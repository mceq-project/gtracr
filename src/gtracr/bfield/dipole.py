"""Ideal magnetic dipole field model."""

import numpy as np

from gtracr.constants import EARTH_RADIUS, G10


class MagneticField:
    """
    Earth's geomagnetic field in the ideal dipole approximation.

    Uses the first-order term of the spherical harmonic expansion of the
    magnetic scalar potential (1/r³ falloff).  No external currents assumed.

    Notes
    -----
    The dipole coefficient *G10* is the IGRF-2020 value (29 404.8 nT).
    """

    def __init__(self):
        pass

    def values(self, r, theta, phi):
        """
        Dipole B-field components at geocentric spherical coordinates.

        Parameters
        ----------
        r : float
            Radial distance from Earth's centre in metres.
        theta : float
            Colatitude (polar angle) in radians.
        phi : float
            Longitude (azimuthal angle) in radians (unused for pure dipole).

        Returns
        -------
        numpy.ndarray, shape (3,)
            ``(Br, Btheta, Bphi)`` in Tesla.
        """
        Br = -2.0 * (EARTH_RADIUS / r) ** 3.0 * G10 * np.cos(theta)
        Btheta = -((EARTH_RADIUS / r) ** 3.0) * G10 * np.sin(theta)
        Bphi = 0.0
        return np.array([Br, Btheta, Bphi])
