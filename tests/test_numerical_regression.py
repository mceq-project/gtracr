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
    Path(__file__).parent.parent / "src" / "gtracr" / "data" / "igrf14.json"
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
    (1.4852121367402633e-05, -2.627407683753786e-05, -2.1805862019785087e-06),
    (-5.02219791204919e-05, -2.213627814232616e-05, 7.682106101764621e-07),
    (4.997621329132202e-05, -1.7596922640250742e-05, 8.737362085077639e-06),
    (-5.1588841191851474e-05, -1.0660400897229365e-05, -4.726079450346835e-06),
    (2.9451477836378822e-05, -1.2392519505573896e-05, -1.1164718976348897e-05),
    (7.099986168808814e-06, -2.4285462616006496e-05, -7.75133746274345e-07),
    (-1.3781881449317958e-05, -1.371293313738948e-05, -2.4159727076148905e-06),
    (-5.3446368127335265e-05, -3.7966017205398845e-06, 1.5133607666670793e-06),
    (5.042280414963028e-05, 4.981648524825708e-06, -1.4293861798941648e-05),
    (3.084126612425399e-06, -3.0080308502720382e-05, 5.1166523163944504e-06),
]


@pytest.mark.parametrize("idx", range(len(BFIELD_COORDS)))
def test_igrf_bfield_components(idx):
    r, theta, phi = BFIELD_COORDS[idx]
    br_exp, btheta_exp, bphi_exp = EXPECTED_BFIELD[idx]
    igrf = IGRF(DATA_PATH, 2020.0)
    br, btheta, bphi = igrf.values(r, theta, phi)
    assert np.isclose(br, br_exp, rtol=1e-10, atol=0.0), f"Br mismatch at coord {idx}"
    assert np.isclose(btheta, btheta_exp, rtol=1e-10, atol=0.0), (
        f"Btheta mismatch at coord {idx}"
    )
    assert np.isclose(bphi, bphi_exp, rtol=1e-10, atol=0.0), f"Bphi mismatch at coord {idx}"


# ---------------------------------------------------------------------------
# IGRF trajectory sixvector tests
# ---------------------------------------------------------------------------

# (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, energy)
INITIAL_VARIABLES = [
    ("p+", 90.0, 90.0, 100.0, 0.0, 0.0, 0.0, 30.0, None),
    ("p+", 120.0, 90.0, 100.0, 0.0, 0.0, -1.0, 30.0, None),
    ("p+", 0.0, 155.0, 100.0, 50.0, 100.0, 0.0, 50.0, None),
    ("p+", 90.0, 175.0, 100.0, 89.0, 20.0, 0.0, 20.0, None),
    ("p+", 90.0, 175.0, 100.0, -90.0, 20.0, 0.0, 20.0, None),
    ("e-", 90.0, 175.0, 100.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 275.0, 0.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 0.0, 10.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 180.0, 0.0, 89.0, 0.0, 0.0, 20.0, None),
    ("p+", 45.0, 180.0, 0.0, 0.0, 180.0, 100.0, 20.0, None),
    ("p+", 45.0, 180.0, 0.0, 0.0, 180.0, 100.0, 5.0, None),
    ("p+", 45.0, 180.0, 0.0, 0.0, 180.0, 100.0, None, 10.0),
    ("p+", 9.0, 100.0, 0.0, 50.0, 260.0, 100.0, None, 50.0),
]

EXPECTED_IGRF_SIXVEC = [
                           [6471199.519801404, 1.5707962240044842, 0.00046304539138271593, -5.138766157141422e-21, -7.118082397146252e-21, 1.603285557440793e-17],
                           [6394824.277506614, 1.3951728534006413, 1.9838215722565786, -8.541200421650536e-18, -3.020489715207225e-18, 1.3227889936847815e-17],
                           [63713434.38880278, 0.809253646516784, 2.1782367999584706, 2.665215845610666e-17, 8.734356623037231e-19, 1.7130029371949903e-18],
                           [63713986.16464133, 0.20365614585776015, -2.0048931673980106, 1.0677698566849564e-17, -4.116979358644733e-19, -2.50657557744288e-19],
                           [63712283.078748524, 2.674998996066176, -0.028448326647251044, 1.066952159839563e-17, -3.959173162967669e-19, -5.001294094300288e-19],
                           [63714589.637032256, 1.4611777044970051, -2.8588574079677227, 1.0593713876010334e-17, 2.785657363336611e-19, -1.3932690084540737e-18],
                           [63714111.52138361, 1.00927044447581, -2.2959883780268706, 1.0573224431183495e-17, 1.0192770266728855e-18, 1.188936733754539e-18],
                           [63714751.86456817, 1.7962943491383543, -1.7895792772770232, 1.048456357205545e-17, 1.6695978108111967e-18, 1.2377160174332313e-18],
                           [63712507.42945745, 0.35120267648952025, -1.5725287839823827, 1.0685467074107351e-17, 9.573290192038856e-20, -2.39164599825254e-19],
                           [63714456.33589988, 1.3938807214817477, -0.4712439333096505, 1.0243665391648447e-17, -1.788590673988123e-19, 3.0464569361971628e-18],
                           [6469657.857121274, 1.7062474860381585, -3.011992838728641, -2.3492455396083863e-18, 7.386382470634177e-19, 1.0372112260715485e-18],
                           [6468703.613481506, 1.7881670474704021, -2.782641064303045, -4.6471770040471645e-18, -1.0572081755360313e-18, 2.3655883049401822e-18],
                           [63713709.35653473, 0.8986594746995084, -1.3193122522350769, 2.6660081017552627e-17, 1.06075950885779e-18, 1.3777689140946872e-18],
                       ]

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
