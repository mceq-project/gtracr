"""
Pure-Python fallback trajectory integrator.

Implements the same relativistic Lorentz ODE as the C++ ``TrajectoryTracer``
using a fixed-step RK4 scheme in Python/NumPy.  This is used for testing,
debugging, and as a reference implementation.  The production integrator is
``gtracr._libgtracr.TrajectoryTracer`` (C++ via pybind11), which is ~100×
faster.
"""

import numpy as np

from gtracr.constants import EARTH_RADIUS, SPEED_OF_LIGHT


class pTrajectoryTracer:
    """
    Pure-Python trajectory tracer (RK4).

    Mirrors the C++ ``TrajectoryTracer`` interface so it can be used as a
    drop-in for testing.  B-field support: dipole (``"d"``) and IGRF-13
    (``"i"``).

    Parameters
    ----------
    charge : float
        Particle charge in SI (coulombs, already multiplied by *e*).
    mass : float
        Particle rest mass in SI (kg).
    start_altitude : float
        Atmosphere entry altitude in km (default 100).
    escape_radius : float
        Escape radius in metres from Earth's centre (default 10 RE).
    stepsize : float
        Integration step size in seconds (default 1e-5).
    max_step : int
        Maximum number of integration steps (default 10 000).
    bfield_type : str
        ``"d"`` for dipole or ``"i"`` for IGRF-13 (default ``"d"``).
    igrf_params : tuple or None
        ``(data_dir, decimal_year)`` required when *bfield_type* is ``"i"``.
    """

    def __init__(
        self,
        charge,
        mass,
        start_altitude=100.0,
        escape_radius=10.0 * EARTH_RADIUS,
        stepsize=1e-5,
        max_step=10000,
        bfield_type="d",
        igrf_params=None,
    ):
        from gtracr.bfield import IGRF13, MagneticField

        self.charge = charge
        self.mass = mass
        self.start_altitude = start_altitude
        self.escape_radius = escape_radius
        self.stepsize = stepsize
        self.max_step = max_step
        self.particle_escaped = False

        if bfield_type.find("d") != -1:
            self.bfield = MagneticField()
        elif bfield_type.find("i") != -1:
            if igrf_params is None:
                raise ValueError(
                    "igrf_params=(data_dir, decimal_year) required for bfield_type='i'"
                )
            year = igrf_params[1]
            self.bfield = IGRF13(year, nmax=13)
        else:
            raise ValueError(f"Unknown bfield_type {bfield_type!r}: use 'd' or 'i'")

        self.final_time = 0.0
        self.final_sixvector = np.zeros(6)

    def ode_lrz(self, t, vec):
        """
        Relativistic Lorentz ODE in geocentric spherical coordinates.

        Parameters
        ----------
        t : float
            Current time (seconds).
        vec : numpy.ndarray, shape (6,)
            State vector ``(r, θ, φ, pᵣ, pθ, pφ)``.

        Returns
        -------
        numpy.ndarray, shape (6,)
            Time derivative of the state vector.
        """
        r, theta, phi, pr, ptheta, pphi = vec

        pmag = np.sqrt(pr**2.0 + ptheta**2.0 + pphi**2.0)
        gamma = np.sqrt(1.0 + (pmag / (self.mass * SPEED_OF_LIGHT)) ** 2.0)
        rel_mass = self.mass * gamma

        bf_r, bf_theta, bf_phi = self.bfield.values(r, theta, phi)

        dprdt = (
            -self.charge * (ptheta * bf_phi - bf_theta * pphi)
            + (ptheta**2.0 + pphi**2.0) / r
        )
        dpthetadt = (
            self.charge * (pr * bf_phi - bf_r * pphi)
            + (pphi**2.0 * np.cos(theta)) / (r * np.sin(theta))
            - (pr * ptheta) / r
        )
        dpphidt = (
            -self.charge * (pr * bf_theta - bf_r * ptheta)
            - (pr * pphi) / r
            - (ptheta * pphi * np.cos(theta)) / (r * np.sin(theta))
        )

        return (
            np.array(
                [
                    pr,
                    ptheta / r,
                    pphi / (r * np.sin(theta)),
                    dprdt,
                    dpthetadt,
                    dpphidt,
                ]
            )
            / rel_mass
        )

    def evaluate(self, t0, vec0):
        """
        Integrate the trajectory without storing intermediate points.

        Parameters
        ----------
        t0 : float
            Initial time in seconds.
        vec0 : numpy.ndarray, shape (6,)
            Initial state ``(r, θ, φ, pᵣ, pθ, pφ)``.
        """
        t = t0
        vec = np.array(vec0, dtype=float)
        h = self.stepsize
        for _ in range(self.max_step):
            k1 = h * self.ode_lrz(t, vec)
            k2 = h * self.ode_lrz(t + 0.5 * h, vec + 0.5 * k1)
            k3 = h * self.ode_lrz(t + 0.5 * h, vec + 0.5 * k2)
            k4 = h * self.ode_lrz(t + h, vec + k3)
            vec += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += h
            r = vec[0]
            if r > self.escape_radius:
                self.particle_escaped = True
                break
            if r < self.start_altitude + EARTH_RADIUS:
                break
        self.final_time = t
        self.final_sixvector = vec

    def evaluate_and_get_trajectory(self, t0, vec0):
        """
        Integrate and return the full trajectory as a dictionary.

        Parameters
        ----------
        t0 : float
            Initial time in seconds.
        vec0 : numpy.ndarray, shape (6,)
            Initial state ``(r, θ, φ, pᵣ, pθ, pφ)``.

        Returns
        -------
        dict
            Keys: ``"t"``, ``"r"``, ``"theta"``, ``"phi"``,
            ``"pr"``, ``"ptheta"``, ``"pphi"``.
        """
        t = t0
        vec = np.array(vec0, dtype=float)
        h = self.stepsize
        t_arr = []
        vec_arr = []
        for _ in range(self.max_step):
            t_arr.append(t)
            vec_arr.append(vec.copy())
            k1 = h * self.ode_lrz(t, vec)
            k2 = h * self.ode_lrz(t + 0.5 * h, vec + 0.5 * k1)
            k3 = h * self.ode_lrz(t + 0.5 * h, vec + 0.5 * k2)
            k4 = h * self.ode_lrz(t + h, vec + k3)
            vec += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += h
            r = vec[0]
            if r > self.escape_radius:
                self.particle_escaped = True
                break
            if r < self.start_altitude + EARTH_RADIUS:
                break
        self.final_time = t
        self.final_sixvector = vec

        arr = np.array(vec_arr)
        r_arr, theta_arr, phi_arr, pr_arr, ptheta_arr, pphi_arr = arr.T
        return {
            "t": t_arr,
            "r": r_arr,
            "theta": theta_arr,
            "phi": phi_arr,
            "pr": pr_arr,
            "ptheta": ptheta_arr,
            "pphi": pphi_arr,
        }
