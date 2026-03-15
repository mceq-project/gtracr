import sys
import os
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil

from gtracr.trajectory import Trajectory
from gtracr.lib._libgtracr import TrajectoryTracer as CppTrajectoryTracer
from gtracr.lib.constants import ELEMENTARY_CHARGE, KG_PER_GEVC2, KG_M_S_PER_GEVC


def _default_workers():
    physical = psutil.cpu_count(logical=False)
    return physical if physical is not None else max(1, psutil.cpu_count(logical=True) // 2)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)


def _evaluate_single_direction(args):
    '''
    Evaluate the cutoff rigidity for a single random (zenith, azimuth) direction.
    Designed to run in a worker process (no shared state required).

    Parameters
    ----------
    args : tuple
        (location, plabel, bfield_type, date_str, palt, rigidity_list, dt, max_time, seed)

    Returns
    -------
    (azimuth, zenith, rcutoff) or (azimuth, zenith, 0.0) if no cutoff found
    '''
    location, plabel, bfield_type, date_str, palt, rigidity_list, dt, max_time, seed = args
    rng = np.random.default_rng(seed)
    azimuth, zenith = rng.random(2) * np.array([360., 180.])

    # Build Trajectory once for geometry; position and momentum direction are
    # the same for all rigidities — only momentum magnitude changes.
    traj = Trajectory(
        plabel=plabel,
        location_name=location,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=palt,
        rigidity=rigidity_list[0],
        bfield_type=bfield_type,
        date=date_str,
    )

    max_step = int(np.ceil(max_time / dt))
    charge_si = traj.charge * ELEMENTARY_CHARGE
    mass_si = traj.mass * KG_PER_GEVC2

    # Build one TrajectoryTracer (loads IGRF once for the whole rigidity sweep).
    tracer = CppTrajectoryTracer(
        charge_si, mass_si,
        traj.start_alt, traj.esc_alt,
        dt, max_step,
        traj.bfield_type, traj.igrf_params,
    )

    # Precompute momentum direction unit vector (fixed across rigidities).
    ref_mom_si = traj.particle.momentum * KG_M_S_PER_GEVC
    pos = traj.particle_sixvector[:3]
    mom_unit = traj.particle_sixvector[3:] / ref_mom_si

    for rigidity in rigidity_list:
        traj.particle.set_from_rigidity(rigidity)
        mom_si = traj.particle.momentum * KG_M_S_PER_GEVC
        vec0 = list(pos) + list(mom_unit * mom_si)
        tracer.reset()
        tracer.evaluate(0.0, vec0)
        if tracer.particle_escaped:
            return (azimuth, zenith, rigidity)

    return (azimuth, zenith, 0.0)


class GMRC():
    '''
    Evaluates the geomagnetic cutoff rigidities associated to a specific location on the globe for each zenith and azimuthal angle (a zenith angle > 90 degrees are for upward-moving particles, that is, for cosmic rays coming from the other side of Earth).

    The cutoff rigidities are evaluated using a Monte-Carlo sampling scheme, combined with a 2-dimensional linear interpolation using `scipy.interpolate`.

    The resulting cutoffs can be plotted as 2-dimensional heatmap.

    Parameters
    -----------

    - location : str
        The location in which the geomagnetic cutoff rigidities are evaluated (default = "Kamioka"). The names must be one of the locations contained in `location_dict`, which is configured in `gtracr.utils`.
    - particle_altitude : float
        The altitude in which the cosmic ray interacts with the atmosphere in km (default = 100).
    - iter_num : int
        The number of iterations to perform for the Monte-Carlo sampling routine (default = 10000)
    - bfield_type : str
        The type of magnetic field model to use for the evaluation of the cutoff rigidities (default = "igrf"). Set to "dipole" to use the dipole approximation of the geomagnetic field instead.
    - particle_type : str
        The type of particle of the cosmic ray (default  ="p+").
    - date : str
        The specific date in which the geomagnetic rigidity cutoffs are evaluated. Defaults to the current date.
    - min_rigidity : float
        The minimum rigidity to which we evaluate the cutoff rigidities for (default = 5 GV).
    - max_rigidity : float
        The maximum rigidity to which we evaluate the cutoff rigidities for (default = 55 GV).
    - delta_rigidity : float
        The spacing between each rigidity (default = 5 GV). Sets the coarseness of the rigidity sample space.
    - n_workers : int, optional
        Number of parallel worker processes to use. Defaults to the number of CPU cores.
        Set to 1 to disable parallelism (useful for debugging).
    '''
    def __init__(self,
                 location="Kamioka",
                 particle_altitude=100,
                 iter_num=10000,
                 bfield_type="igrf",
                 particle_type="p+",
                 date=str(date.today()),
                 min_rigidity=5.,
                 max_rigidity=55.,
                 delta_rigidity=1.,
                 n_workers=_default_workers()):
        # set class attributes
        self.location = location
        self.palt = particle_altitude
        self.iter_num = iter_num
        self.bfield_type = bfield_type
        self.plabel = particle_type
        self.date = date
        self.n_workers = n_workers  # None = use all CPU cores
        '''
        Rigidity configurations
        '''
        self.rmin = min_rigidity
        self.rmax = max_rigidity
        self.rdelta = delta_rigidity

        # generate list of rigidities
        self.rigidity_list = np.arange(self.rmin, self.rmax, self.rdelta)

        # initialize container for rigidity cutoffs
        self.data_dict = {
            "azimuth": np.zeros(self.iter_num),
            "zenith": np.zeros(self.iter_num),
            "rcutoff": np.zeros(self.iter_num)
        }

    def evaluate(self, dt=1e-5, max_time=1):
        '''
        Evaluate the rigidity cutoff value at some provided location
        on Earth for a given cosmic ray particle.

        Uses parallel worker processes (one per CPU core by default) to
        evaluate independent Monte Carlo samples concurrently.

        Parameters
        ----------

        - dt : float
            The stepsize of each trajectory evaluation (default = 1e-5)
        - max_time : float
            The maximal time of each trajectory evaluation (default = 1.).

        '''
        rigidity_list = list(self.rigidity_list)

        # Build argument list for each MC sample, using distinct random seeds
        # for reproducibility across different worker processes
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31, size=self.iter_num)

        args_list = [
            (self.location, self.plabel, self.bfield_type, self.date,
             self.palt, rigidity_list, dt, max_time, int(seeds[i]))
            for i in range(self.iter_num)
        ]

        n_workers = self.n_workers
        use_parallel = (n_workers is None or n_workers > 1)

        if use_parallel:
            # Parallel evaluation: each worker process evaluates one MC sample
            results = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(_evaluate_single_direction, args): i
                    for i, args in enumerate(args_list)
                }
                for future in tqdm(as_completed(future_to_idx),
                                   total=self.iter_num,
                                   desc="GMRC evaluation"):
                    i = future_to_idx[future]
                    results.append((i, future.result()))

            for i, (az, zen, rc) in results:
                self.data_dict["azimuth"][i] = az
                self.data_dict["zenith"][i] = zen
                self.data_dict["rcutoff"][i] = rc
        else:
            # Sequential fallback (n_workers=1), useful for debugging
            for i in tqdm(range(self.iter_num)):
                az, zen, rc = _evaluate_single_direction(args_list[i])
                self.data_dict["azimuth"][i] = az
                self.data_dict["zenith"][i] = zen
                self.data_dict["rcutoff"][i] = rc

    def interpolate_results(self,
                            method="linear",
                            ngrid_azimuth=70,
                            ngrid_zenith=70):
        '''
        Interpolate the rigidity cutoffs using `scipy.interpolate.griddata`

        Parameters
        ----------
        - method : str
            The type of linear interpolation used for `griddata` (default = "linear"). Choices are between "nearest", "linear", and "cubic".
        - ngrid_azimuth, ngrid_zenith : int
            The number of grids for the azimuth and zenith angles used for the interpolation (default = 70).

        Returns
        --------

        Returns a tuple of the following objects:

        - azimuth_grid : np.array(float), size ngrid_azimuth
            The linearly spaced values of the azimuthal angle
        - zenith_grid : np.array(float), size ngrid_zenith
            The linearly spaced values of the zenith angle
        - rcutoff_grid : np.array(float), size ngrid_azimuth x ngrid_zenith
            The interpolated geomagnetic cutoff rigidities.
        '''

        azimuth_grid = np.linspace(np.min(self.data_dict["azimuth"]),
                                   np.max(self.data_dict["azimuth"]),
                                   ngrid_azimuth)
        zenith_grid = np.linspace(np.max(self.data_dict["zenith"]),
                                  np.min(self.data_dict["zenith"]),
                                  ngrid_zenith)

        rcutoff_grid = griddata(points=(self.data_dict["azimuth"],
                                        self.data_dict["zenith"]),
                                values=self.data_dict["rcutoff"],
                                xi=(azimuth_grid[None, :], zenith_grid[:,
                                                                       None]),
                                method=method)

        return (azimuth_grid, zenith_grid, rcutoff_grid)
