import gtracr.lib.igrf_utils as iuf
from gtracr.lib.constants import DEG_PER_RAD, EARTH_RADIUS, G10

"""
Library that controls the equation for the Earth's magnetic field.
"""

import os

import numpy as np
from scipy.interpolate import interp1d

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)


class MagneticField:
    """
    The base class that contains the properties of
    Earth's geomagnetic field defined in spherical coordinates.

    The base class uses the ideal dipole approximation
    in which the first-order approximation of the series expansion
    of the magnetic potential of Earth's geomagnetic field
    is used.

    Notes
    ------
    - The Earth's magnetic field is assumed to consist of
    no external currents, i.e. curl(B) = 0

    """

    def __init__(self):
        pass

    def values(self, r, theta, phi):
        """
        The values of the magnetic field in spherical
        coordinates at a given location (r, theta, phi)

        Parameters
        ----------
        - r : float
            the radial component in meters
        - theta : float
            the polar component in radians
        - phi : float
            the phi component in radians

        Returns
        -------
        - Br, Btheta, Bphi: np.array(float), size 3
            The spherical components of the magnetic field
        """

        Br = -2.0 * (EARTH_RADIUS / r) ** 3.0 * G10 * np.cos(theta)
        Btheta = -((EARTH_RADIUS / r) ** 3.0) * G10 * np.sin(theta)
        Bphi = 0.0

        return np.array([Br, Btheta, Bphi])


class IGRF13(MagneticField):
    """
    Derived class from MagneticField class that
    describes Earth's geomagnetic field using the
    International Geomagnetic Reference Field (IGRF)
    of the 13th Edition (2020).

    The scripts that produces the IGRF magnetic field
    are obtained from the Python scripts that are
    officially constructed by the V-MOD group in the
    International Union of Geodesy and Geophysics.

    More information about the IGRF-13 model can be obtained
    here: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html

    Members
    -------
    - curr_year : float
        the current year in years
    - igrf_coeffs : np.array, float
        the Gauss coefficients for the year specified in curr_year.
    - nmax : int
        the number of truncation

    Initialization Parameters
    -------------------------
    - curr_year : float
        the current year in years. This is required as the
        magnetic field values vary in a linear fashion per year.

    - nmax : int, optional
        the number to truncate the series expansion of the
        magnetic field. The default value is obtained from
        the .shc file.

    """

    def __init__(self, curr_year, nmax=None):
        """
        Parameters
        ----------
        curr_year : float
            Decimal year for coefficient interpolation (e.g. 2024.5).
        nmax : int, optional
            Truncation degree for the spherical harmonic expansion.
            Defaults to the maximum degree in the IGRF-13 data file.
        """
        # override MagneticField __init__ function
        # not necessary, but for decorative sake
        # super().__init__(self)

        self.curr_year = curr_year
        fpath = os.path.join(PARENT_DIR, "data", "IGRF13.shc")

        leap_year = False

        # check if current year is a leap year or not
        if self.curr_year % 4 == 0:
            leap_year = True

        # import the coefficients from .shc file
        igrf = iuf.load_shcfile(fpath, leap_year=leap_year)

        # interpolate to account for any year
        # linear interpolation since time variation in
        # coefficients is a linear one
        interp_coeffs = interp1d(
            igrf.time,
            igrf.coeffs,
            kind="linear",
        )

        # get the number of truncation based on the
        # igrf .shc file
        self.nmax = igrf.parameters["nmax"] if nmax is None else nmax

        self.igrf_coeffs = interp_coeffs(self.curr_year).T

    def values(self, r, theta, phi):
        """Evaluate the IGRF-13 magnetic field at a point.

        Parameters
        ----------
        r : float
            Radial distance from Earth's center in meters.
        theta : float
            Colatitude (polar angle) in radians.
        phi : float
            Longitude (azimuthal angle) in radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            ``(Br, Btheta, Bphi)`` in Tesla.
        """
        # transform theta, phi to degrees and r into kilometers
        r *= 1e-3
        theta *= DEG_PER_RAD
        phi *= DEG_PER_RAD

        # check if theta / phi is > 180 or > 360 respectively
        # this is to prevent errors to be raised by the igrf evaluator
        if theta > 180.0:
            theta = theta % 180.0  # normalize by nearest remainder from 180

        if phi > 360.0:
            phi = phi % 360.0  # normalize by nearest remainder from 360

        # obtaining the values can easily be done by using
        # the function synth_values from igrf_utils.py
        Br, Btheta, Bphi = iuf.synth_values(
            self.igrf_coeffs, r, theta, phi, nmax=self.nmax
        )

        # take first index since synth_values return arrays, not scalars
        return np.array([Br, Btheta, Bphi]) * (1e-9)  # convert to teslas
