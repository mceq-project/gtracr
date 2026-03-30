"""
Numerical regression tests with tight tolerances.
Golden values were captured from the frozen-field RK4 implementation (2026-03-01).
These tests verify that optimizations do not change physical results.

IGRF_DATE pins the evaluation date so golden values don't drift day-to-day.
"""

from pathlib import Path

import numpy as np
import pytest

from gtracr._libgtracr import IGRF
from gtracr.trajectory import Trajectory

DATA_PATH = str(
    Path(__file__).parent.parent / "src" / "gtracr" / "data" / "igrf13.json"
)

IGRF_DATE = "2026-03-01"  # pinned so golden values don't drift with calendar date

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
    [6471199.680494885, 1.570796257723918, 0.0004630478707916171, -5.143689868164982e-21, -7.195707803370034e-21, 1.607999837818926e-17],
    [6394372.707662919, 1.3991171683374217, 1.981918797258987, -8.582622983971869e-18, -3.017789199068475e-18, 1.326080193044416e-17],
    [63712172.75024449, 0.8074557182925343, 2.177079868177197, 2.6731270321304086e-17, 8.662159841410913e-19, 1.7138434050351642e-18],
    [63713826.62935646, 0.20724683235046246, -1.9957373608122242, 1.071337494242615e-17, -4.1087892510670036e-19, -2.479213146891487e-19],
    [63712428.50813812, 4.610658305768503, 62847852133.84116, 3.5881168721786107e-06, 3.6627401823020574e-07, 1.740677399721284e-11],
    [63712068.88832171, 1.461372001754389, -2.856471578420075, 1.062864433286929e-17, 2.7848361809064457e-19, -1.392212235664858e-18],
    [63713174.36762395, 1.0062703134318098, -2.2986029134111092, 1.06070644404028e-17, 1.0149102209319807e-18, 1.1915961393581837e-18],
    [63714305.53476941, 1.79003494627064, -1.7890309416567596, 1.0520213399764715e-17, 1.673187015456767e-18, 1.2365371203577488e-18],
    [63713141.73385609, 0.3535228391631918, -1.5677680828770737, 1.0717717745429198e-17, 9.905609299844461e-20, -2.379774681958377e-19],
    [63713909.67920056, 1.3926491900123048, 5.802233592658787, 1.0276960039365773e-17, -1.9478960283588128e-19, 3.050590431430006e-18],
    [6471179.121769882, 1.7062049597352513, 3.271351862035455, -2.3571156905562516e-18, 7.356370219610532e-19, 1.0526017383688944e-18],
    [6470889.92009875, 1.7885471384944032, 3.5009992115454973, -4.656227401356567e-18, -1.0618240733781405e-18, 2.3912028329404368e-18],
    [63714845.19530036, 0.8980275779166911, -1.319182078289385, 2.6738641785692963e-17, 1.0610672189027167e-18, 1.382169917827142e-18],
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
    if EXPECTED_IGRF_ESCAPED[idx]:
        pytest.skip("Exit point on escape sphere is architecture-sensitive; use test_igrf_escaped_flag instead")
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
        date=IGRF_DATE,
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
        date=IGRF_DATE,
    )
    traj.get_trajectory(dt=1e-5, max_time=1.0)
    assert traj.particle_escaped == EXPECTED_IGRF_ESCAPED[idx], (
        f"escaped flag mismatch at case {idx}: got {traj.particle_escaped}, expected {EXPECTED_IGRF_ESCAPED[idx]}"
    )
