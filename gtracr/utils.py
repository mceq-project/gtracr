"""
Utility class for gtracr.
Contains conversions between geodesic coordinates and debuggers for the package.
may also contain other things (not known as of now)
"""

import os
import pickle

import numpy as np

from gtracr.lib.location import Location
from gtracr.lib.particle import Particle

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

# set global dictionaries here
# global location_dict
# global particle_dict


def dec_to_dms(lat_dec, lng_dec):
    """
    Converts geodesic coordinates expressed in decimal notation
    to DMS notation (degree, minutes, seconds).

    Parameters
    ----------
    - lat_dec : float
        the latitude in decimal notation ([-90, 90])
    - lng_dec : float
        the longitude in decimal notation ([-180,180])

    Returns
    --------
    - lat_dms : str
        latitude in DMS notation
    - lng_dms : str
        longitude in DMS notation

    The evaluation is performed with reference to the DMS to decimal calculator:
    https://www.rapidtables.com/convert/number/degrees-to-degrees-minutes-seconds.html

    The convention follows the WGS 84 convention as posed here
    : https://gisgeography.com/latitude-longitude-coordinates/

    The convention shows that:
    - A positive latitude (latitude > 0) is towards the North Pole
    - A negative latitude (latitude < 0) is towards the South Pole
    - A positive longitude (longitude > 0) is towards the East relative to one's location
    - A negative longitude (longitude < 0) is towards the West relative to one's location

    So with this, we see that:
    - latitude goes counter-clockwise from the equator
    - longitude goes counter-clockwise from the Prime Meridian
    """
    # latitude in dms
    lat_deg = int(np.floor(lat_dec))
    lat_min = int(np.floor((lat_dec - lat_deg) * 60.0))
    lat_sec = int(np.floor((lat_dec - lat_deg - (lat_min / 60.0)) * 60.0))

    # add north or south depending on sign of lat_dec
    lat_symb = "N" if lat_dec >= 0 else "S"

    lat_dms = f"{lat_deg:d}°{lat_min:d}'{lat_sec:d}\"{lat_symb:s}"

    # longitude in dms
    lng_deg = int(np.floor(lng_dec))
    lng_min = int(np.floor((lng_dec - lng_deg) * 60.0))
    lng_sec = int(np.floor((lng_dec - lng_deg - (lng_min / 60.0)) * 60.0))

    # add east or west depending on sign of lng_dec
    lng_symb = "E" if lng_dec >= 0 else "W"

    lng_dms = f"{lng_deg:d}°{lng_min:d}'{lng_sec:d}\"{lng_symb:s}"

    return lat_dms, lng_dms


def ymd_to_dec(ymd_date):
    """
    Converts a date in yyyy-mm-dd format into decimal format in
    units of years.

    Parameters
    ----------

    - ymd : str
        the year, month, and date of the specified date in yyyy-mm-dd
        format.

    Returns
    ---------

    - dec_date : float
        the date in decimal format, in units of years.

    """
    # break down the str to get year, month, date separately
    year, month, day = [float(val) for val in ymd_date.split("-")]

    # the number of days in each month
    # not considering leap years right now
    days_per_mth = np.array(
        [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
    )

    # get the total number of days based on the month + day
    ndays = float(days_per_mth[int(month) - 1]) + day

    # check if year is a leap year or not
    # this will change the maximum # days in a year
    # also will change the number of days to the given date
    max_ndays = 365
    if year % 4 == 0:
        max_ndays += 1
        ndays += 1

    # convert month into decimal years
    dec_mth = month / 12.0

    # convert days into decimal years
    dec_days = ndays / max_ndays

    # return the sum of year, month, day
    return year + dec_mth + dec_days


def import_dict(fname):
    """
    Load a pickled dictionary from disk.

    Parameters
    ----------
    fname : str
        Absolute or relative path to the ``.pkl`` file.

    Returns
    -------
    dict
        The deserialized dictionary.
    """
    with open(fname, "rb") as f:
        the_dict = pickle.load(f)
    return the_dict


# def get_locationdict():
#     '''
#     Get the location dictionary from the .pkl file
#     '''
#     fpath = os.path.join(DATA_DIR, "location_dict.pkl")
#     location_dict = import_dict(fpath)
#     return location_dict


def set_locationdict():
    """
    Build the dictionary of predefined detector locations.

    Returns
    -------
    dict of str -> Location
        Mapping from location name to ``Location`` instance.
        Predefined sites: Kamioka, IceCube, SNOLAB, UofA,
        CTA-North, CTA-South, ORCA, ANTARES, Baikal-GVD, TA.
    """
    location_dict = {}

    locations = [
        Location("Kamioka", 36.434800, 137.276599, 0.0),
        Location("IceCube", -89.99, 0.0, 0.0),
        Location("SNOLAB", 46.471983, -81.186701, 0.0),
        Location("UofA", 53.523230, -113.526319, 0.0),
        Location("CTA-North", 28.76216, -17.89201, 0.0),
        Location("CTA-South", -24.68342778, -24.68342778, 0.0),
        Location("ORCA", 42.80000000, 6.0333333, 0.0),
        Location("ANTARES", 42.8, 6.1666666, 0.0),
        Location("Baikal-GVD", 51.77139, 104.3978, 0.0),
        Location("TA", 39.208099, -113.121285, 0.0),
    ]

    # add location to location_dict if it does not exist
    for loc in locations:
        if loc.name not in list(location_dict.keys()):
            location_dict[loc.name] = loc
        else:
            continue

    return location_dict


def set_particledict():
    """
    Build the dictionary of supported cosmic ray particles.

    Returns
    -------
    dict of str -> Particle
        Mapping from particle label to ``Particle`` instance.
        Supported labels: ``"p+"``, ``"p-"``, ``"e+"``, ``"e-"``.
    """
    particle_dict = {}

    # different cosmic ray particles
    particles = [
        Particle("positron", -11, 0.5109 * (1e-3), 1, "e+"),
        Particle("electron", 11, 0.5109 * (1e-3), -1, "e-"),
        Particle("proton", 2212, 0.937272, 1, "p+"),
        Particle("anti-proton", -2212, 0.937272, -1, "p-"),
    ]

    # add particle to particle_dict if it does not exist
    for particle in particles:
        if particle.name not in list(particle_dict.keys()):
            particle_dict[particle.label] = particle
        else:
            continue

    return particle_dict


location_dict = set_locationdict()
particle_dict = set_particledict()
