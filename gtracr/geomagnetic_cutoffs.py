import sys
import os
import threading
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
from datetime import date
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from gtracr.trajectory import Trajectory
from gtracr.lib._libgtracr import TrajectoryTracer as CppTrajectoryTracer
from gtracr.lib.constants import ELEMENTARY_CHARGE, KG_PER_GEVC2, KG_M_S_PER_GEVC, EARTH_RADIUS

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

_thread_local = threading.local()

# Directory for cached IGRF tables (next to the data files).
_TABLE_CACHE_DIR = os.path.join(CURRENT_DIR, "data")


def _get_or_generate_igrf_table(datapath, dec_date):
    '''Load a cached IGRF table from disk, or generate and cache it.

    The table depends on the decimal year and the compile-time grid
    constants (Nr, Ntheta, Nphi).  The cache files (.npy for the table,
    .npz for the params) are stored alongside igrf13.json.
    '''
    from gtracr.lib._libgtracr import (
        generate_igrf_table as _gen_table, TableParams)

    cache_table = os.path.join(
        _TABLE_CACHE_DIR, f"igrf_table_{dec_date:.4f}.npy")
    cache_params = os.path.join(
        _TABLE_CACHE_DIR, f"igrf_table_{dec_date:.4f}_params.npz")

    if os.path.isfile(cache_table) and os.path.isfile(cache_params):
        table_flat = np.load(cache_table)
        d = np.load(cache_params)
        table_params = TableParams()
        table_params.r_min = float(d['r_min'])
        table_params.r_max = float(d['r_max'])
        table_params.log_r_min = float(d['log_r_min'])
        table_params.log_r_max = float(d['log_r_max'])
        table_params.Nr = int(d['Nr'])
        table_params.Ntheta = int(d['Ntheta'])
        table_params.Nphi = int(d['Nphi'])
        return table_flat, table_params

    # Generate from scratch and cache.
    table_flat, table_params = _gen_table(datapath, dec_date)
    np.save(cache_table, table_flat)
    np.savez(cache_params,
             r_min=table_params.r_min, r_max=table_params.r_max,
             log_r_min=table_params.log_r_min, log_r_max=table_params.log_r_max,
             Nr=table_params.Nr, Ntheta=table_params.Ntheta, Nphi=table_params.Nphi)
    return table_flat, table_params


def _evaluate_single_direction(args):
    '''
    Evaluate the cutoff rigidity for a single random (zenith, azimuth) direction.
    Designed to run in a worker process (no shared state required).

    Parameters
    ----------
    args : tuple
        (location, plabel, bfield_type, date_str, palt, rigidity_list, dt, max_time,
         seed, solver_char, atol, rtol)

    Returns
    -------
    (azimuth, zenith, rcutoff) or (azimuth, zenith, 0.0) if no cutoff found
    '''
    location, plabel, bfield_type, date_str, palt, rigidity_list, dt, max_time, \
        seed, solver_char, atol, rtol = args
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
        solver_char, atol, rtol,
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


def _evaluate_direction_cpp_only(args):
    '''
    Lightweight thread worker for the tabulated IGRF case.
    All Python/numpy work (Trajectory, coordinate transforms) is done in the
    main thread; this function only builds a C++ tracer and calls
    find_cutoff_rigidity() which runs the entire rigidity loop in C++ with
    the GIL released.

    The tracer is cached per thread via threading.local() so that the
    expensive IGRF JSON parse (in the C++ constructor) happens only once
    per pool thread instead of once per direction.
    '''
    (shared_table, table_params, igrf_params,
     charge_si, mass_si, start_alt, esc_alt,
     dt, max_step, solver_char, atol, rtol,
     pos, mom_unit, rigidity_list, mom_factor,
     azimuth, zenith) = args

    tracer = getattr(_thread_local, 'tracer', None)
    if tracer is None:
        tracer = CppTrajectoryTracer(
            shared_table, table_params,
            charge_si, mass_si,
            start_alt, esc_alt,
            dt, max_step,
            igrf_params,
            solver_char, atol, rtol,
        )
        _thread_local.tracer = tracer
    else:
        tracer.set_start_altitude(start_alt)

    rcutoff = tracer.find_cutoff_rigidity(pos, mom_unit, rigidity_list, mom_factor)
    return (azimuth, zenith, rcutoff)


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
        Number of parallel worker processes to use. Defaults to os.cpu_count()
        (all logical CPUs). Set to 1 to disable parallelism (useful for debugging).
    - solver : str, optional
        Integration method: "rk4" (frozen-field RK4, default), "boris" (Boris pusher),
        or "rk45" (adaptive Dormand-Prince).
    - atol, rtol : float, optional
        Absolute and relative tolerances for the RK45 adaptive solver (defaults 1e-3, 1e-6).
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
                 n_workers=None,
                 solver="rk4",
                 atol=1e-3,
                 rtol=1e-6):
        # set class attributes
        self.location = location
        self.palt = particle_altitude
        self.iter_num = iter_num
        self.bfield_type = bfield_type
        self.plabel = particle_type
        self.date = date
        self.n_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)
        _SOLVER_CHARS = {"rk4": "r", "boris": "b", "rk45": "a"}
        self.solver_char = _SOLVER_CHARS.get(solver.lower(), "r")
        self.atol = atol
        self.rtol = rtol
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

        n_workers = self.n_workers
        use_parallel = (n_workers > 1)
        use_threads = (use_parallel and self.bfield_type[0] == 't')

        if use_threads:
            # Tabulated IGRF: generate table once, share across threads.
            # All Python/numpy work (Trajectory construction, coordinate
            # transforms) is done here in the main thread so the thread
            # workers only execute GIL-released C++ code.
            from gtracr.utils import ymd_to_dec
            datapath = os.path.join(CURRENT_DIR, "data")
            dec_date = float(ymd_to_dec(self.date))
            igrf_params = (datapath, dec_date)
            shared_table, table_params = _get_or_generate_igrf_table(
                datapath, dec_date)

            max_step = int(np.ceil(max_time / dt))

            # Pre-compute initial conditions for every MC direction (main thread).
            args_list = []
            for i in range(self.iter_num):
                rng_i = np.random.default_rng(int(seeds[i]))
                azimuth, zenith = rng_i.random(2) * np.array([360., 180.])

                traj = Trajectory(
                    plabel=self.plabel,
                    location_name=self.location,
                    zenith_angle=zenith,
                    azimuth_angle=azimuth,
                    particle_altitude=self.palt,
                    rigidity=rigidity_list[0],
                    bfield_type="igrf",  # only for coordinate transform
                    date=self.date,
                )

                charge_si = traj.charge * ELEMENTARY_CHARGE
                mass_si = traj.mass * KG_PER_GEVC2
                mom_factor = float(np.abs(traj.charge)) * KG_M_S_PER_GEVC
                ref_mom_si = traj.particle.momentum * KG_M_S_PER_GEVC
                pos = tuple(float(x) for x in traj.particle_sixvector[:3])
                mom_unit = tuple(float(x) for x in traj.particle_sixvector[3:] / ref_mom_si)

                args_list.append((
                    shared_table, table_params, igrf_params,
                    charge_si, mass_si,
                    traj.start_alt, traj.esc_alt,
                    dt, max_step, self.solver_char, self.atol, self.rtol,
                    pos, mom_unit, rigidity_list, mom_factor,
                    azimuth, zenith,
                ))

            results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(_evaluate_direction_cpp_only, args): i
                    for i, args in enumerate(args_list)
                }
                for future in tqdm(as_completed(future_to_idx),
                                   total=self.iter_num,
                                   desc="GMRC evaluation (threaded)"):
                    i = future_to_idx[future]
                    results.append((i, future.result()))

            for i, (az, zen, rc) in results:
                self.data_dict["azimuth"][i] = az
                self.data_dict["zenith"][i] = zen
                self.data_dict["rcutoff"][i] = rc
        else:
            args_list = [
                (self.location, self.plabel, self.bfield_type, self.date,
                 self.palt, rigidity_list, dt, max_time, int(seeds[i]),
                 self.solver_char, self.atol, self.rtol)
                for i in range(self.iter_num)
            ]

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

    def evaluate_batch(self, dt=1e-5, max_time=1., base_seed=None):
        '''
        Evaluate rigidity cutoffs entirely in C++ (batch mode).

        The entire MC loop — RNG, coordinate transforms, rigidity scanning,
        and threading — runs in a single C++ call, eliminating all Python
        overhead per ray.

        Parameters
        ----------
        dt : float
            Step size for trajectory integration (default 1e-5 s).
        max_time : float
            Maximum integration time per trajectory (default 1 s).
        base_seed : int, optional
            RNG seed for reproducibility. If None, a random seed is used.
        '''
        import time as _time
        from gtracr.lib._libgtracr import (
            BatchGMRCParams, TableParams,
            batch_gmrc_evaluate as _batch_eval)
        from gtracr.utils import location_dict, particle_dict, ymd_to_dec

        t_wall_start = _time.monotonic()

        use_table = (self.bfield_type[0] == 't')
        field_label = "table" if use_table else "direct IGRF"
        print(f"Initializing batch GMRC for {self.location} "
              f"({self.iter_num} samples, {self.n_workers} threads, "
              f"{field_label})...")

        datapath = os.path.join(CURRENT_DIR, "data")
        dec_date = float(ymd_to_dec(self.date))
        igrf_params = (datapath, dec_date)

        if use_table:
            shared_table, table_params = _get_or_generate_igrf_table(
                datapath, dec_date)
        else:
            shared_table = None
            table_params = TableParams()

        loc = location_dict[self.location]
        particle = particle_dict[self.plabel]

        if base_seed is None:
            base_seed = int(np.random.default_rng().integers(0, 2**63))

        p = BatchGMRCParams()
        p.latitude = loc.latitude
        p.longitude = loc.longitude
        p.detector_alt = loc.altitude       # km
        p.particle_alt = self.palt           # km (GMRC stores km)
        p.escape_radius = 10.0 * EARTH_RADIUS
        p.charge = float(particle.charge)    # units of e
        p.mass = particle.mass               # GeV/c^2
        p.min_rigidity = self.rmin
        p.max_rigidity = self.rmax
        p.delta_rigidity = self.rdelta
        p.dt = dt
        p.max_time = max_time
        p.solver_type = self.solver_char
        p.bfield_type = self.bfield_type[0]
        p.atol = self.atol
        p.rtol = self.rtol
        p.n_samples = self.iter_num
        p.n_threads = self.n_workers
        p.max_attempts_factor = 30
        p.base_seed = base_seed

        t_init_done = _time.monotonic()
        print(f"Initialized in {t_init_done - t_wall_start:.2f}s. "
              f"Running cutoff calculations...")

        zenith, azimuth, rcutoff, total_traj = _batch_eval(
            shared_table, table_params, igrf_params, p)

        t_calc_done = _time.monotonic()
        n = len(zenith)
        calc_elapsed = t_calc_done - t_init_done
        total_elapsed = t_calc_done - t_wall_start
        ktraj_per_s = (total_traj / 1000.) / calc_elapsed if calc_elapsed > 0 else float('inf')

        print(f"Completed {n} cutoffs in {calc_elapsed:.2f}s "
              f"({total_traj} trajectories, {ktraj_per_s:.1f}k traj/s "
              f"across {self.n_workers} threads, {total_elapsed:.2f}s total)")

        # Store results (may be shorter than iter_num if safety limit hit)
        self.data_dict = {
            "azimuth": azimuth,
            "zenith": zenith,
            "rcutoff": rcutoff,
        }
        if n < self.iter_num:
            import warnings
            warnings.warn(
                f"Batch GMRC: only {n}/{self.iter_num} successful samples "
                f"(safety limit reached). Results may be sparse.")

    def interpolate_results(self,
                            method="linear",
                            ngrid_azimuth=70,
                            ngrid_zenith=70):
        '''
        Interpolate the rigidity cutoffs using `scipy.interpolate.griddata`

        Legacy method kept for backward compatibility; prefer
        ``bin_results`` for large sample counts.

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

    def bin_results(self, nbins_azimuth=72, nbins_zenith=36):
        '''
        Bin the MC samples into a regular (azimuth, zenith) grid and compute
        the mean cutoff rigidity per bin.  Much faster than scattered
        interpolation and more appropriate when the sample count is large.

        Parameters
        ----------
        nbins_azimuth : int
            Number of azimuth bins (default 72, i.e. 5-degree bins).
        nbins_zenith : int
            Number of zenith bins (default 36, i.e. 5-degree bins).

        Returns
        -------
        (azimuth_centres, zenith_centres, rcutoff_grid) where
        azimuth_centres and zenith_centres are 1-D bin-centre arrays and
        rcutoff_grid is shape (nbins_zenith, nbins_azimuth).  Bins with
        no samples are filled with NaN.
        '''
        az  = self.data_dict["azimuth"]
        zen = self.data_dict["zenith"]
        rc  = self.data_dict["rcutoff"]

        az_edges  = np.linspace(0., 360., nbins_azimuth + 1)
        zen_edges = np.linspace(0., 180., nbins_zenith + 1)

        # Digitize: bin index 1..N for in-range, 0 or N+1 for out-of-range
        az_idx  = np.digitize(az,  az_edges)  - 1  # 0-based
        zen_idx = np.digitize(zen, zen_edges) - 1

        # Clamp to valid range
        az_idx  = np.clip(az_idx,  0, nbins_azimuth  - 1)
        zen_idx = np.clip(zen_idx, 0, nbins_zenith - 1)

        # Accumulate sum and count per bin
        sum_grid   = np.zeros((nbins_zenith, nbins_azimuth))
        count_grid = np.zeros((nbins_zenith, nbins_azimuth))
        np.add.at(sum_grid,   (zen_idx, az_idx), rc)
        np.add.at(count_grid, (zen_idx, az_idx), 1)

        with np.errstate(invalid='ignore'):
            rcutoff_grid = sum_grid / count_grid  # NaN where count == 0

        az_centres  = 0.5 * (az_edges[:-1]  + az_edges[1:])
        zen_centres = 0.5 * (zen_edges[:-1] + zen_edges[1:])

        return (az_centres, zen_centres, rcutoff_grid)
