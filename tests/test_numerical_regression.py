"""
Numerical regression tests with tight tolerances.
Golden values were captured from the frozen-field RK4 implementation (2026-03-01).
These tests verify that optimizations do not change physical results.

IGRF_DATE pins the evaluation date so golden values don't drift day-to-day.
Trajectory goldens were regenerated after correcting Gregorian date conversion;
the calendar date and numerical tolerances are unchanged.
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

EXPECTED_IGRF_SIXVEC = [[6471199.680456355,
  1.5707962577414334,
  0.00046304787079527173,
  -5.144310375920068e-21,
  -7.19388290247321e-21,
  1.6079998378601282e-17],
 [6395774.990369979,
  1.3988939334057546,
  1.982264322973147,
  -8.573826687664324e-18,
  -3.020328504238415e-18,
  1.3265910309277999e-17],
 [63714982.05340226,
  0.8075299472971341,
  2.177126268327159,
  2.6731251199688484e-17,
  8.665622155098491e-19,
  1.7139676581595662e-18],
 [63714197.890034966,
  0.2072676236808022,
  -1.9959996403436684,
  1.0713364122692878e-17,
  -4.1071953689853587e-19,
  -2.486223899937321e-19],
 [63712428.50813812,
  4.610658305768505,
  62847852133.84114,
  3.588116872178611e-06,
  3.6627401823020564e-07,
  1.7406773997429297e-11],
 [63712099.59329914,
  1.4612385315147998,
  -2.856687795139834,
  1.0628605404760392e-17,
  2.7837194080058235e-19,
  -1.3925374841813664e-18],
 [63713230.46539686,
  1.0064484329310972,
  -2.2987279163763943,
  1.0607052097103232e-17,
  1.015157818998031e-18,
  1.1914958366262065e-18],
 [63713345.94274482,
  1.7903028423218497,
  -1.7892466789458248,
  1.0520160919662004e-17,
  1.6734617005772495e-18,
  1.2366203790520562e-18],
 [63713549.723770596,
  0.35330004917116414,
  -1.5680657397137965,
  1.0717712186285232e-17,
  9.883048027216264e-20,
  -2.3832582898344056e-19],
 [63714210.67007905,
  1.3928248647576091,
  5.802460380604856,
  1.02769061939912e-17,
  -1.9412229216114648e-19,
  3.0508147625321976e-18],
 [6468704.66970633,
  1.7063186854370245,
  3.271542981567034,
  -2.3633140714604986e-18,
  7.31365553704417e-19,
  1.0416519779311658e-18],
 [6471103.760603782,
  1.7885025740721798,
  3.5010115508365405,
  -4.656006830689544e-18,
  -1.0623019265637343e-18,
  2.3914204271286833e-18],
 [63712084.86533006,
  0.8980489849287697,
  -1.3193423766547505,
  2.6738664055695216e-17,
  1.061106199662757e-18,
  1.381708969581137e-18]]

EXPECTED_IGRF_ESCAPED = [False, False, True, True, True, True, True, True, True, True, False, False, True]


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_sixvector(idx):
    if EXPECTED_IGRF_ESCAPED[idx]:
        pytest.skip(
            "Exit point on escape sphere is architecture-sensitive; use test_igrf_escaped_flag instead"
        )
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
