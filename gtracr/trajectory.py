import os
from datetime import date

import numpy as np

from gtracr.lib._libgtracr import TrajectoryTracer
from gtracr.lib.constants import (
    EARTH_RADIUS,
    ELEMENTARY_CHARGE,
    KG_M_S_PER_GEVC,
    KG_PER_GEVC2,
    RAD_PER_DEG,
)
from gtracr.lib.trajectory_tracer import pTrajectoryTracer
from gtracr.utils import location_dict, particle_dict, ymd_to_dec

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")


class Trajectory:
    """
    Class that controls the trajectory of a particle at some given energy / rigidity

    Required Parameters
    -------------------
    - zenith_angle : float
        the angle of the cosmic ray trajectory from the local zenith, with 0 being at the local zenith
    - azimuth_angle : float
        the angle of the cosmic ray trajectory with 0 being in the direction of the geographic North in the local tangent plane
    - energy : float
        the cosmic ray energy. Cannot be used concurrently with rigidity (default = None).
    - rigidity : float
        the cosmic ray rigidity. Cannot be used concurrently with energy (default = None).

    Optional Parameters
    --------------------
    - particle_altitude : float
        the altitude in which the cosmic ray hits Earth's atmosphere and creates showers (default = 100km)
    - latitude : float
         the geographic latitude of the detector, with 0 defined at the equator in degrees
    - longitude : float
        the geographic longitude of the detector, with 0 defined at the Prime Meridian in degrees
    - detector_altitude : float
        the height of the detector from sea level in km (default = 0km)
    - location_name : str
        the location name as stored in location_dict (default = None). Available as an alternative option to initialize the location of the trajectory.
    - bfield_type : str
        the type of bfield to evaluate the trajectory with (either 'dipole' or 'igrf', default = igrf)
    - date : str
        the date in which the field is evaluated in (defaults to the current date). Date must be formatted in "yyyy-mm-dd" format.
    - plabel : str
         the label of the particle defined in particle_dict (default = "p+"). Available options are "p+", "p-", "e+", "e-".
    - escape_altitude : float
        the altitude in which the particle has "escaped" Earth in meters (default = 10 * RE)
    """

    # Solver name → C++ char mapping
    _SOLVER_CHARS = {"rk4": "r", "boris": "b", "rk45": "a"}

    def __init__(
        self,
        zenith_angle,
        azimuth_angle,
        energy=None,
        rigidity=None,
        particle_altitude=100.0,
        latitude=0.0,
        longitude=0.0,
        detector_altitude=0.0,
        location_name=None,
        bfield_type="igrf",
        date=str(date.today()),
        plabel="p+",
        escape_altitude=10.0 * EARTH_RADIUS,
        solver="rk4",
        atol=1e-3,
        rtol=1e-6,
    ):
        """
        Cosmic ray direction configurations
        """
        self.zangle = zenith_angle
        self.azangle = azimuth_angle
        self.palt = particle_altitude * (1e3)  # convert to meters
        self.esc_alt = escape_altitude
        """
        Particle type configuration
        """
        # define particle from particle_dict

        self.particle = particle_dict[plabel]
        self.charge = self.particle.charge
        self.mass = self.particle.mass
        """
        Geodesic coordinate configuration
        """
        # only import location dictionary and use those values if location_name is not None
        if location_name is not None:
            # location_dict = set_locationdict()
            loc = location_dict[location_name]

            latitude = loc.latitude
            longitude = loc.longitude
            detector_altitude = loc.altitude

        self.lat = latitude
        self.lng = longitude
        self.dalt = detector_altitude * (1e3)  # convert to meters

        self.start_alt = self.dalt + self.palt
        """
        Cosmic ray energy / rigidity / momentum configuration
        """
        # define rigidity and energy only if they are provided, evaluate for the other member
        # also set momentum in each case
        if rigidity is None and energy is not None:
            self.particle.set_from_energy(energy)
            self.rigidity = self.particle.rigidity
            self.energy = energy
        elif energy is None and rigidity is not None:
            self.particle.set_from_rigidity(rigidity)
            self.rigidity = rigidity
            self.energy = self.particle.get_energy_rigidity()
        # elif rigidity is None and energy is None:
        else:
            raise Exception("Provide either energy or rigidity as input, not both!")
        """
        Magnetic Field Model configuration
        """
        # type of bfield to use
        # take only first character for compatibility with char in c++
        self.bfield_type = bfield_type[0]

        # find the path to the data and set current date for igrf bfield
        datapath = os.path.abspath(DATA_DIR)
        # print(datapath)
        dec_date = float(ymd_to_dec(date))
        self.igrf_params = (datapath, dec_date)
        """
        Other set-ups
        """
        # Solver configuration
        solver_key = solver.lower() if isinstance(solver, str) else solver
        self.solver_char = self._SOLVER_CHARS.get(solver_key, "r")
        self.atol = atol
        self.rtol = rtol

        self.particle_escaped = False  # check if trajectory is allowed or not
        # final time and six-vector, used for testing purposes
        self.final_time = 0.0
        self.final_sixvector = np.zeros(6)

        # get the 6-vector for the particle, initially defined in
        # detector frame, and transform it to geocentric
        # coordinates
        self.particle_sixvector = self.detector_to_geocentric()

    def get_trajectory(
        self, dt=1e-5, max_time=1, max_step=None, get_data=False, use_python=False
    ):
        """
        Evaluate the trajectory of the particle within Earth's magnetic field
        and determines whether particle has escaped or not.
        Optionally also returns the information of the trajectory (the duration
        and the six-vector in spherical coordinates) if `get_data == True`.

        Parameters
        ----------

        dt : float
            the time step between each iteration of the integration (default: 1e-5)
        max_time : float
            the maximum duration in which the integration would occur in seconds (default: 10)
        max_step : int, optional
            maximum number of steps to integrate for (default None). If `max_step` is not `None`,
            then `max_step` will override the evaluation of maximum number of steps based on `max_time`.
        get_data : bool, optional
            decides whether we want to extract the information (time and six vector)
            for the whole trajectory for e.g. debugging purposes (default: False)
        use_python : bool, optional
            decides whether to use the python implementation for the TrajectoryTracer class instead of
            that implemented in C++. This is mainly enabled for debugging/testing purposes (default: False)

        Returns
        ---------

        - trajdata_dict : dict
            a dictionary that contains the information of the whole trajectory in
            spherical coordinates.
            Keys are ["t", "r", "theta", "phi", "pr", "ptheta", "pphi"]
            - only returned when `get_data` is True
        """

        # evaluate max_step only when max_time is given, else use the user-given
        # max step
        max_step = int(np.ceil(max_time / dt)) if max_step is None else max_step

        # start iteration process

        # Convert to SI units using local variables (not mutating object state,
        # so get_trajectory() can be called multiple times safely)
        charge_si = self.charge * ELEMENTARY_CHARGE
        mass_si = self.mass * KG_PER_GEVC2

        # initialize trajectory tracer
        if use_python:
            # the python trajectory tracer version (for testing/debugging only)
            traj_tracer = pTrajectoryTracer(
                charge_si,
                mass_si,
                self.start_alt,
                self.esc_alt,
                dt,
                max_step,
                self.bfield_type,
                self.igrf_params,
            )
        else:
            # the C++ vectorized trajectory tracer (primary implementation)
            traj_tracer = TrajectoryTracer(
                charge_si,
                mass_si,
                self.start_alt,
                self.esc_alt,
                dt,
                max_step,
                self.bfield_type,
                self.igrf_params,
                self.solver_char,
                self.atol,
                self.rtol,
            )

        # set initial values
        particle_t0 = 0.0
        particle_vec0 = self.particle_sixvector

        if get_data:
            # evaluate the trajectory tracer
            # get data dictionary of the trajectory
            trajectory_datadict = traj_tracer.evaluate_and_get_trajectory(
                particle_t0, particle_vec0
            )

            # convert all data to numpy arrays for computations etc
            # this should be done within C++ in future versions
            for key, arr in list(trajectory_datadict.items()):
                trajectory_datadict[key] = np.array(arr)

            # add the Cartesian components to the dictionary
            # for plotting purposes
            self.convert_to_cartesian(trajectory_datadict)

            # lastly get the boolean of if the particle has escaped or not
            # in binary format
            # this helps with the geomagnetic cutoff procedure
            # alternatively this can be inside the geomagnetic things
            self.particle_escaped = traj_tracer.particle_escaped

            # set the final time and six-vector from the evaluator
            self.final_time = traj_tracer.final_time
            self.final_sixvector = np.array(traj_tracer.final_sixvector)

            return trajectory_datadict

        else:
            # simply evaluate without returning the dictionary
            traj_tracer.evaluate(particle_t0, particle_vec0)
            # lastly get the boolean of if the particle has escaped or not
            # in binary format
            # this helps with the geomagnetic cutoff procedure
            # alternatively this can be inside the geomagnetic things
            self.particle_escaped = traj_tracer.particle_escaped

            # set the final time and six-vector from the evaluator
            self.final_time = traj_tracer.final_time
            self.final_sixvector = np.array(traj_tracer.final_sixvector)

            return None

    def convert_to_cartesian(self, trajectory_data):
        r_arr = trajectory_data["r"] / EARTH_RADIUS
        theta_arr = trajectory_data["theta"]
        phi_arr = trajectory_data["phi"]

        # convert to cartesian & add to dict
        trajectory_data["x"] = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
        trajectory_data["y"] = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
        trajectory_data["z"] = r_arr * np.cos(theta_arr)

    # get the initial trajectory points based on the latitude, longitude, altitude, zenith, and azimuth
    # returns tuple of 2 trajectory points (the initial one and the first one relating to that of the zenith and azimuth one)
    def detector_to_geocentric(self):
        """
        Convert the coordinates defined in the detector frame (the coordinate system
        defined in the local tangent plane to Earth's surface at some specified
        latitude and longitude) to geocentric (Earth-centered, Earth-fixed) coordinates.

        Returns
        -------
        particle_tp : np.array(float), size 6
            The six vector of the particle, evaluated
            based on the location of the detector, the direction in which the particle
            comes from, and the altitude in which a shower occurs.

        """

        # transformation process for coordinate
        detector_coord = self.geodesic_to_cartesian()

        # change particle initial location if zenith angle is > 90
        # so that we only consider upward moving particles
        if self.zangle > 90.0:
            # here we count both altitude and magnitude as a whole
            # for ease of computation

            self.start_alt = (
                self.start_alt
                * np.cos(self.zangle * RAD_PER_DEG)
                * np.cos(self.zangle * RAD_PER_DEG)
            )

            particle_coord = self.get_particle_coord(
                altitude=0.0,
                magnitude=-(2.0 * EARTH_RADIUS + self.palt)
                * np.cos(self.zangle * RAD_PER_DEG),
            )

        elif self.zangle <= 90.0:
            particle_coord = self.get_particle_coord(
                altitude=self.palt, magnitude=1e-10
            )

        # print(detector_coord, particle_coord)
        # print(self.tf_matrix())
        transformed_cart_coord = self.transform(detector_coord, particle_coord)

        # transformation for momentum
        # need to convert from natural units to SI units
        detector_momentum = np.zeros(3)
        particle_momentum = self.get_particle_coord(
            altitude=0.0, magnitude=self.particle.momentum * KG_M_S_PER_GEVC
        )

        transformed_cart_mmtm = self.transform(detector_momentum, particle_momentum)

        # create new trajectory point and set the new coordinate and momentum
        particle_sixvector = self.cartesian_to_spherical(
            transformed_cart_coord, transformed_cart_mmtm
        )

        # return particle_tp
        return particle_sixvector

    # convert between detector coordinates to geocentric coordinates
    def transform(self, detector_coord, particle_coord):
        return detector_coord + np.dot(self.transform_matrix(), particle_coord)

    # get the detector coordinates (in Cartesian) from zenith and azimuthal angles
    def get_particle_coord(self, altitude, magnitude):
        xi = self.zangle * RAD_PER_DEG
        alpha = self.azangle * RAD_PER_DEG
        # alpha = (self.azangle + 90.) * DEG_TO_RAD

        # xt and yt are flipped from usual conversions from spherical coordinates
        # to allow azimuth = 0 to point to the geographic north pole
        # (if we use normal spherical coordinate conversion, azimuth = 0
        #  means pointing west in detector coordinates)

        # the below coordinate convention for detector coordinates are used
        # to follow the convention used in Honda's 2002 paper, in which
        # an azimuth angle of 0 correlates to direction of the
        # geographic South Pole, and + azimuth will indicate particles
        # coming from the west
        xt = magnitude * np.sin(xi) * np.sin(alpha)
        yt = -magnitude * np.sin(xi) * np.cos(alpha)
        zt = magnitude * np.cos(xi) + altitude

        return np.array([xt, yt, zt])

    def transform_matrix(self):
        """
        Returns the transformation matrix for transforming between coordinates in the local tangent plane (detector coordinates) and geocentric coordinates.
        """
        lmbda = self.lat * RAD_PER_DEG
        eta = self.lng * RAD_PER_DEG

        # print(lmbda, eta)

        row1 = np.array(
            [-np.sin(eta), -np.cos(eta) * np.sin(lmbda), np.cos(lmbda) * np.cos(eta)]
        )
        row2 = np.array(
            [np.cos(eta), -np.sin(lmbda) * np.sin(eta), np.cos(lmbda) * np.sin(eta)]
        )
        row3 = np.array([0.0, np.cos(lmbda), np.sin(lmbda)])

        return np.array([row1, row2, row3])

    def geodesic_to_cartesian(self):
        """
        Transforms vectors in geodesic coordinates into Cartesian components

        Returns
        -------

        - cart_vals : np.array(float), size 3
            the coordinate vector in cartesian coordinates (Earth-centered, Earth-fixed coordinates)
        """
        r = EARTH_RADIUS + self.dalt
        theta = (90.0 - self.lat) * RAD_PER_DEG
        phi = self.lng * RAD_PER_DEG

        cart_vals = np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        )

        return cart_vals

    def cartesian_to_spherical(self, cart_coord, cart_mmtm):
        """
        Transforms coordinate and momentum vectors from Cartesian coordinates to Spherical coordinates.

        Parameters
        -----------

        - cart_coord : np.array(float), size 3
            the coordinate vector in cartesian coordinates
        - cart_mmtm : np.arrray(float), size 3
            the momentum vector in cartesian coordianates

        Returns
        -------

        - sph_sixvector : np.array(float), size 6
            the six-vector (coordinate and momentum) in spherical coordinates
        """
        # first set x, y, z for readability
        x, y, z = cart_coord

        # convert coordinates to spherical
        r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        # define transformation matrix for momentum
        tfmat_cart_to_sph = np.array(
            [
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
                [
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    -np.sin(theta),
                ],
                [-np.sin(phi), np.cos(phi), 0.0],
            ]
        )

        # # get spherical momentum
        sph_mmtm = np.dot(tfmat_cart_to_sph, cart_mmtm)

        # store both results into an array
        sph_sixvector = np.hstack((np.array([r, theta, phi]), sph_mmtm))

        return sph_sixvector
