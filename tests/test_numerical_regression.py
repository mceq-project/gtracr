"""
Numerical regression tests with tight tolerances.
Golden values were captured from the frozen-field RK4 implementation (2026-03).
These tests verify that optimizations do not change physical results.
"""

from pathlib import Path

import numpy as np
import pytest

from gtracr._libgtracr import IGRF
from gtracr.trajectory import Trajectory

DATA_PATH = str(
    Path(__file__).parent.parent / "src" / "gtracr" / "data" / "igrf13.json"
)

# ---------------------------------------------------------------------------
# B-field component tests
# ---------------------------------------------------------------------------

# (r [m], theta [rad], phi [rad])
BFIELD_COORDS = [
    (6.471e6, 1.570796326794897, 0.000000000000000),
    (6.471e6, 0.785398163397448, 1.570796326794897),
    (6.471e6, 2.356194490192345, 3.141592653589793),
    (6.471e6, 0.523598775598299, -1.047197551196598),
    (6.471e6, 2.617993877991494, 0.523598775598299),
    (7.000e6, 1.570796326794897, 0.785398163397448),
    (8.000e6, 1.047197551196598, -0.523598775598299),
    (6.471e6, 0.100000000000000, 0.500000000000000),
    (6.471e6, 3.041592653589793, 1.500000000000000),
    (6.600e6, 1.570796326794897, 3.141592653589793),
]

# (Br [T], Btheta [T], Bphi [T])
EXPECTED_BFIELD = [
    (1.4855695312886571e-05, -2.6275042651584387e-05, -2.1783012782847714e-06),
    (-5.022698287666843e-05, -2.2136022738173563e-05, 7.670865558988388e-07),
    (4.998055401773263e-05, -1.7598037567259555e-05, 8.735265501917444e-06),
    (-5.159088313804225e-05, -1.0660607633772066e-05, -4.726147369878846e-06),
    (2.9451320469060627e-05, -1.23928974362497e-05, -1.116463946819446e-05),
    (7.098511625973886e-06, -2.428599076702391e-05, -7.740370143861933e-07),
    (-1.3781812451778208e-05, -1.3714018905197889e-05, -2.416042461267865e-06),
    (-5.3446224075424234e-05, -3.7948639669908606e-06, 1.5139925094950205e-06),
    (5.042536466649683e-05, 4.981942197366722e-06, -1.4294509205006488e-05),
    (3.086411652001845e-06, -3.0081399087441946e-05, 5.111542735065936e-06),
]


@pytest.mark.parametrize("idx", range(len(BFIELD_COORDS)))
def test_igrf_bfield_components(idx):
    r, theta, phi = BFIELD_COORDS[idx]
    br_exp, btheta_exp, bphi_exp = EXPECTED_BFIELD[idx]
    igrf = IGRF(DATA_PATH, 2020.0)
    br, btheta, bphi = igrf.values(r, theta, phi)
    assert np.isclose(br, br_exp, rtol=1e-10), f"Br mismatch at coord {idx}"
    assert np.isclose(btheta, btheta_exp, rtol=1e-10), f"Btheta mismatch at coord {idx}"
    assert np.isclose(bphi, bphi_exp, rtol=1e-10), f"Bphi mismatch at coord {idx}"


# ---------------------------------------------------------------------------
# IGRF trajectory sixvector tests
# ---------------------------------------------------------------------------

# (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, energy)
INITIAL_VARIABLES = [
    ("p+", 90.0, 90.0, 100.0, 0.0, 0.0, 0.0, 30.0, None),
    ("p+", 120.0, 90.0, 100.0, 0.0, 0.0, -1.0, 30.0, None),
    ("p+", 0.0, 25.0, 100.0, 50.0, 100.0, 0.0, 50.0, None),
    ("p+", 90.0, 5.0, 100.0, 89.0, 20.0, 0.0, 20.0, None),
    ("p+", 90.0, 5.0, 100.0, -90.0, 20.0, 0.0, 20.0, None),
    ("e-", 90.0, 5.0, 100.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 265.0, 0.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 180.0, 10.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 0.0, 0.0, 89.0, 0.0, 0.0, 20.0, None),
    ("p+", 45.0, 0.0, 0.0, 0.0, 180.0, 100.0, 20.0, None),
    ("p+", 45.0, 0.0, 0.0, 0.0, 180.0, 100.0, 5.0, None),
    ("p+", 45.0, 0.0, 0.0, 0.0, 180.0, 100.0, None, 10.0),
    ("p+", 9.0, 80.0, 0.0, 50.0, 260.0, 100.0, None, 50.0),
]

EXPECTED_IGRF_SIXVEC = [
    [
        6471199.68050388,
        1.570796257719829,
        0.00046304787079076373,
        -5.143545002470065e-21,
        -7.196133850247816e-21,
        1.6079998378093046e-17,
    ],
    [
        6395271.144827872,
        1.3992367009947384,
        1.9815374626089006,
        -8.576473297451113e-18,
        -3.0191953439754613e-18,
        1.3264459191915995e-17,
    ],
    [
        63712214.8539477,
        0.8074387437774729,
        2.1770700088482826,
        2.6731276114383172e-17,
        8.661272008667169e-19,
        1.7137976762272475e-18,
    ],
    [
        63713739.96693872,
        0.2072419641715477,
        -1.995676058214853,
        1.0713377464628803e-17,
        -4.1091609699408715e-19,
        -2.4775757546605917e-19,
    ],
    [
        63712428.50813812,
        4.610658305768503,
        62847852133.84116,
        3.58811687217861e-06,
        3.6627401823020585e-07,
        1.740677399716434e-11,
    ],
    [
        63712061.66126494,
        1.4614031647765793,
        -2.856421095363587,
        1.0628653420170669e-17,
        2.785097024529514e-19,
        -1.3921362950657391e-18,
    ],
    [
        63713161.19193234,
        1.0062287310793574,
        -2.298573719569141,
        1.0607067318186453e-17,
        1.0148524314729071e-18,
        1.1916195694514073e-18,
    ],
    [
        63714529.34146243,
        1.789972410364906,
        -1.7889805515157322,
        1.0520225644299797e-17,
        1.6731228863883088e-18,
        1.2365177382975789e-18,
    ],
    [
        63713046.41897766,
        0.353574856242518,
        -1.5676986454616886,
        1.0717719040448463e-17,
        9.910880434013549e-20,
        -2.378961477644391e-19,
    ],
    [
        63714509.338436544,
        1.3926079941190597,
        5.802183827322609,
        1.027698127671675e-17,
        -1.9494152261863959e-19,
        3.0505090860606305e-18,
    ],
    [
        6471152.09785546,
        1.7062074962095228,
        3.2713491909868777,
        -2.357155002479004e-18,
        7.356247048921283e-19,
        1.0525224252410776e-18,
    ],
    [
        6470839.9779103,
        1.788557541472736,
        3.5009963288796593,
        -4.656278933674043e-18,
        -1.061712549427715e-18,
        2.3911519244280324e-18,
    ],
    [
        63714791.309966356,
        0.8980221451744953,
        -1.3191453766497798,
        2.673863545597222e-17,
        1.0610689069329414e-18,
        1.3822910964929718e-18,
    ],
]

EXPECTED_IGRF_ESCAPED = [
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
]


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_sixvector(idx):
    plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en = INITIAL_VARIABLES[idx]
    traj = Trajectory(
        plabel=plabel,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=palt,
        latitude=lat,
        longitude=lng,
        detector_altitude=dalt,
        rigidity=rig,
        energy=en,
        bfield_type="igrf",
    )
    traj.get_trajectory(dt=1e-5, max_time=1.0)
    assert np.allclose(traj.final_sixvector, EXPECTED_IGRF_SIXVEC[idx], rtol=1e-4), (
        f"sixvector mismatch at case {idx}"
    )


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_escaped_flag(idx):
    plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en = INITIAL_VARIABLES[idx]
    traj = Trajectory(
        plabel=plabel,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=palt,
        latitude=lat,
        longitude=lng,
        detector_altitude=dalt,
        rigidity=rig,
        energy=en,
        bfield_type="igrf",
    )
    traj.get_trajectory(dt=1e-5, max_time=1.0)
    assert traj.particle_escaped == EXPECTED_IGRF_ESCAPED[idx], (
        f"escaped flag mismatch at case {idx}: got {traj.particle_escaped}, expected {EXPECTED_IGRF_ESCAPED[idx]}"
    )
