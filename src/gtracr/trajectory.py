"""Single cosmic ray trajectory evaluation through Earth's geomagnetic field."""

from datetime import date
from pathlib import Path

import numpy as np

from gtracr._fallback import pTrajectoryTracer
from gtracr._libgtracr import TrajectoryTracer
from gtracr.constants import (
    EARTH_RADIUS,
    ELEMENTARY_CHARGE,
    KG_M_S_PER_GEVC,
    KG_PER_GEVC2,
    RAD_PER_DEG,
    SOLVER_CHARS,
)
from gtracr.utils import location_dict, particle_dict, ymd_to_dec

_DATA_DIR = Path(__file__).parent / "data"


class Trajectory:
    """
    Evaluate a single cosmic ray trajectory through Earth's geomagnetic field.

    Constructs initial conditions from a particle type, arrival direction,
    energy or rigidity, and geographic location, then integrates the
    relativistic Lorentz force equation in geocentric spherical coordinates.

    Parameters
    ----------
    zenith_angle : float
        Angle from local zenith in degrees. 0 = directly overhead,
        90 = horizontal, >90 = upward-moving (from other side of Earth).
    azimuth_angle : float
        Angle from geographic north in the local tangent plane, in degrees.
        0 = south, 90 = west, 180 = north, 270 = east.
    energy : float, optional
        Cosmic ray kinetic energy in GeV. Mutually exclusive with *rigidity*.
    rigidity : float, optional
        Cosmic ray rigidity (momentum / charge) in GV. Mutually exclusive
        with *energy*.
    particle_altitude : float, optional
        Altitude at which the cosmic ray enters the atmosphere, in km
        (default 100).
    latitude : float, optional
        Geographic latitude of the detector in decimal degrees (default 0).
    longitude : float, optional
        Geographic longitude of the detector in decimal degrees (default 0).
    detector_altitude : float, optional
        Detector altitude above sea level in km (default 0).
    location_name : str, optional
        Name of a predefined location (e.g. ``"Kamioka"``, ``"IceCube"``).
        Overrides *latitude*, *longitude*, and *detector_altitude*.
    bfield_type : str, optional
        Magnetic field model: ``"igrf"`` (default), ``"dipole"``, or
        ``"table"`` (tabulated IGRF for speed).
    date : str, optional
        Date for IGRF evaluation in ``"yyyy-mm-dd"`` format
        (default: today).
    plabel : str, optional
        Particle label: ``"p+"`` (default), ``"p-"``, ``"e+"``, ``"e-"``.
    escape_altitude : float, optional
        Radial distance (m) beyond which the particle is considered escaped
        (default ``10 * EARTH_RADIUS``).
    solver : str, optional
        Integration method: ``"rk4"`` (default, frozen-field Runge-Kutta 4),
        ``"boris"`` (Boris pusher, ~30 %% faster), or ``"rk45"`` (adaptive
        Dormand-Prince).
    atol : float, optional
        Absolute tolerance for the RK45 solver (default 1e-3).
    rtol : float, optional
        Relative tolerance for the RK45 solver (default 1e-6).

    Attributes
    ----------
    particle_escaped : bool
        ``True`` if the most recent trajectory escaped Earth's field.
    final_time : float
        Integration time at termination (seconds).
    final_sixvector : numpy.ndarray
        Final ``(r, theta, phi, pr, ptheta, pphi)`` state vector.

    Examples
    --------
    >>> from gtracr.trajectory import Trajectory
    >>> traj = Trajectory(
    ...     zenith_angle=45., azimuth_angle=0., rigidity=20.,
    ...     location_name="Kamioka", bfield_type="igrf",
    ... )
    >>> data = traj.get_trajectory(get_data=True)
    >>> traj.particle_escaped
    True
    """

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
        self.zangle = zenith_angle
        self.azangle = azimuth_angle
        self.palt = particle_altitude * (1e3)  # convert to meters
        self.esc_alt = escape_altitude
        # Particle type configuration
        # define particle from particle_dict

        self.particle = particle_dict[plabel]
        self.charge = self.particle.charge
        self.mass = self.particle.mass
        # Geodesic coordinate configuration
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
        # Cosmic ray energy / rigidity / momentum configuration
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
        else:
            raise Exception("Provide either energy or rigidity as input, not both!")
        # Magnetic field model configuration
        # type of bfield to use
        # take only first character for compatibility with char in c++
        self.bfield_type = bfield_type[0]

        datapath = str(_DATA_DIR.resolve())
        dec_date = float(ymd_to_dec(date))
        self.igrf_params = (datapath, dec_date)
        solver_key = solver.lower() if isinstance(solver, str) else solver
        self.solver_char = SOLVER_CHARS.get(solver_key, "r")
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

            for key, arr in list(trajectory_datadict.items()):
                trajectory_datadict[key] = np.array(arr)
            self.convert_to_cartesian(trajectory_datadict)

            self.particle_escaped = traj_tracer.particle_escaped
            self.final_time = traj_tracer.final_time
            self.final_sixvector = np.array(traj_tracer.final_sixvector)

            return trajectory_datadict

        else:
            traj_tracer.evaluate(particle_t0, particle_vec0)
            self.particle_escaped = traj_tracer.particle_escaped
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

    def get_particle_coord(self, altitude, magnitude):
        """Convert zenith/azimuth angles to a detector-frame Cartesian vector."""
        xi = self.zangle * RAD_PER_DEG
        alpha = self.azangle * RAD_PER_DEG
        # Convention: azimuth=0 points south (Honda 2002); xt/yt are swapped
        # from standard spherical so that north pole is at azimuth=180°.
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
