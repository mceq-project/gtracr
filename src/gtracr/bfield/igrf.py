"""IGRF-14 geomagnetic field model (degree-13 spherical harmonics)."""

import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

import gtracr.bfield._utils as iuf
from gtracr.constants import DEG_PER_RAD, EARTH_RADIUS, IGRF_SHC_FILE  # noqa: F401

_DATA_DIR = Path(__file__).parent.parent / "data"


class IGRF14:
    """
    IGRF-14 geomagnetic field model.

    Evaluates Earth's magnetic field via the International Geomagnetic
    Reference Field 14th generation (1900–2030; definitive to 2020, the 2025
    main field extrapolated with its secular variation to 2030), using
    degree-13 Schmidt quasi-normalized spherical harmonic coefficients.

    Parameters
    ----------
    curr_year : float
        Decimal year for coefficient interpolation (e.g. ``2024.5``).
    nmax : int, optional
        Truncation degree for the series expansion.  Defaults to the
        maximum degree in the IGRF data file (13).

    References
    ----------
    Alken et al., Earth Planets Space (2025), IGRF-14;
    https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field
    """

    def __init__(self, curr_year, nmax=None):
        self.curr_year = curr_year
        fpath = _DATA_DIR / IGRF_SHC_FILE

        leap_year = self.curr_year % 4 == 0
        igrf = iuf.load_shcfile(str(fpath), leap_year=leap_year)

        interp_coeffs = interp1d(igrf.time, igrf.coeffs, kind="linear")
        self.nmax = igrf.parameters["nmax"] if nmax is None else nmax
        self.igrf_coeffs = interp_coeffs(self.curr_year).T

    def values(self, r, theta, phi):
        """
        Evaluate the IGRF field at a point.

        Parameters
        ----------
        r : float
            Radial distance from Earth's centre in metres.
        theta : float
            Colatitude (polar angle) in radians.
        phi : float
            Longitude (azimuthal angle) in radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            ``(Br, Btheta, Bphi)`` in Tesla.
        """
        r_km = r * 1e-3
        theta_deg = theta * DEG_PER_RAD
        phi_deg = phi * DEG_PER_RAD

        if theta_deg > 180.0:
            theta_deg = theta_deg % 180.0
        if phi_deg > 360.0:
            phi_deg = phi_deg % 360.0

        Br, Btheta, Bphi = iuf.synth_values(
            self.igrf_coeffs, r_km, theta_deg, phi_deg, nmax=self.nmax
        )
        return np.array([Br, Btheta, Bphi]) * 1e-9  # nT → T


class IGRF13(IGRF14):
    """Deprecated alias of :class:`IGRF14` (gtracr now ships IGRF-14 only)."""

    def __init__(self, curr_year, nmax=None):
        warnings.warn(
            "IGRF13 is deprecated and evaluates IGRF-14; use IGRF14",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(curr_year, nmax=nmax)
