'''
Numerical regression tests with tight tolerances.
Golden values were captured from the frozen-field RK4 implementation (2026-03).
These tests verify that optimizations do not change physical results.
'''

import math
import numpy as np
import pytest

from gtracr.lib._libgtracr import IGRF
from gtracr.trajectory import Trajectory

DATA_PATH = 'gtracr/data/igrf13.json'

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
    (1.485569531288654e+04, -2.627504265158440e+04, -2.178301278284770e+03),
    (-5.022698287666843e+04, -2.213602273817358e+04, 7.670865558988398e+02),
    (4.998055401773263e+04, -1.759803756725955e+04, 8.735265501917444e+03),
    (-5.159088313804222e+04, -1.066060763377206e+04, -4.726147369878844e+03),
    (2.945132046906063e+04, -1.239289743624970e+04, -1.116463946819445e+04),
    (7.098511625973858e+03, -2.428599076702392e+04, -7.740370143861921e+02),
    (-1.378181245177822e+04, -1.371401890519788e+04, -2.416042461267863e+03),
    (-5.344622407542423e+04, -3.794863966990860e+03, 1.513992509495020e+03),
    (5.042536466649683e+04, 4.981942197366722e+03, -1.429450920500649e+04),
    (3.086411652001816e+03, -3.008139908744193e+04, 5.111542735065936e+03),
]


@pytest.mark.parametrize("idx", range(len(BFIELD_COORDS)))
def test_igrf_bfield_components(idx):
    r, theta, phi = BFIELD_COORDS[idx]
    br_exp, btheta_exp, bphi_exp = EXPECTED_BFIELD[idx]
    igrf = IGRF(DATA_PATH, 2020.0)
    br, btheta, bphi = igrf.values(r, theta, phi)
    assert np.isclose(br,     br_exp,     rtol=1e-10), f"Br mismatch at coord {idx}"
    assert np.isclose(btheta, btheta_exp, rtol=1e-10), f"Btheta mismatch at coord {idx}"
    assert np.isclose(bphi,   bphi_exp,   rtol=1e-10), f"Bphi mismatch at coord {idx}"


# ---------------------------------------------------------------------------
# IGRF trajectory sixvector tests
# ---------------------------------------------------------------------------

# (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, energy)
INITIAL_VARIABLES = [
    ("p+", 90., 90., 100., 0., 0., 0., 30., None),
    ("p+", 120., 90., 100., 0., 0., -1., 30., None),
    ("p+", 0., 25., 100., 50., 100., 0., 50., None),
    ("p+", 90., 5., 100., 89., 20., 0., 20., None),
    ("p+", 90., 5., 100., -90., 20., 0., 20., None),
    ("e-", 90., 5., 100., 40., 200., 0., 20., None),
    ("p+", 45., 265., 0., 40., 200., 0., 20., None),
    ("p+", 45., 180., 10., 40., 200., 0., 20., None),
    ("p+", 45., 0., 0., 89., 0., 0., 20., None),
    ("p+", 45., 0., 0., 0., 180., 100., 20., None),
    ("p+", 45., 0., 0., 0., 180., 100., 5., None),
    ("p+", 45., 0., 0., 0., 180., 100., None, 10.),
    ("p+", 9., 80., 0., 50., 260., 100., None, 50.),
]

EXPECTED_IGRF_SIXVEC = [
    [6.471199625735792e+06, 1.570796326794897e+00, 4.630478780291231e-04, -6.025217124157224e-21, 9.867217328767675e-34, 1.607999924744147e-17],
    [6.371335346026322e+07, 1.570796326794897e+00, 5.409689836827150e+00, 1.555372674920359e-17, 1.791510012189174e-33, 4.082046860194116e-18],
    [6.371259734681733e+07, 7.979748146625527e-01, 2.131001440948860e+00, 2.674803830858280e-17, 7.529699129062378e-19, 1.490904422574872e-18],
    [6.371487178493770e+07, 2.582901355932140e-01, -1.904753969480752e+00, 1.071803571936567e-17, -3.563146559925421e-19, -6.652912354216870e-20],
    [6.371242850812197e+07, 4.610658305770948e+00, 6.284785213384116e+10, 3.588116872178534e-06, 3.662740182309772e-07, 1.740677235989093e-11],
    [6.371412812167440e+07, 1.612670192113047e+00, -2.794639932662458e+00, 1.062056305255945e-17, 5.723174576497950e-19, -1.363073814864442e-18],
    [6.371458356654958e+07, 9.053003311980364e-01, -2.189599300665786e+00, 1.060965169469279e-17, 8.883874272417792e-19, 1.268268268595269e-18],
    [6.371206915991611e+07, 1.731913552668525e+00, -1.620008247289393e+00, 1.051351327064909e-17, 1.503893150800216e-18, 1.485453225462638e-18],
    [6.371385701953655e+07, 4.016896087540504e-01, -1.527572317641708e+00, 1.071857097728965e-17, 1.874818133514414e-19, -1.150944458384313e-19],
    [6.371217143996966e+07, 1.552615389242537e+00, 4.897759649299538e+00, 1.033667870875700e-17, -1.022349522613819e-18, 2.655746939432413e-18],
    [6.468822113847768e+06, 1.737427109977924e+00, 3.335047788120788e+00, -2.455716101088284e-18, 7.264663185259690e-19, 8.019729334895227e-19],
    [6.470696745738055e+06, 1.755643930128702e+00, 3.693730080306008e+00, -4.842999089642077e-18, -2.015885549881965e-18, 1.000837272629210e-18],
    [6.371450809864879e+07, 9.081240198957020e-01, -1.241211445318186e+00, 2.671781397638439e-17, 1.189280561508053e-18, 1.654218899379113e-18],
]

EXPECTED_IGRF_ESCAPED = [
    False, True, True, True, True, True, True, True, True, True, False, False, True,
]


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_sixvector(idx):
    plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en = INITIAL_VARIABLES[idx]
    traj = Trajectory(
        plabel=plabel, zenith_angle=zenith, azimuth_angle=azimuth,
        particle_altitude=palt, latitude=lat, longitude=lng,
        detector_altitude=dalt, rigidity=rig, energy=en,
        bfield_type="igrf",
    )
    traj.get_trajectory(dt=1e-5, max_time=1.)
    assert np.allclose(traj.final_sixvector, EXPECTED_IGRF_SIXVEC[idx], rtol=1e-10), \
        f"sixvector mismatch at case {idx}"


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_escaped_flag(idx):
    plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en = INITIAL_VARIABLES[idx]
    traj = Trajectory(
        plabel=plabel, zenith_angle=zenith, azimuth_angle=azimuth,
        particle_altitude=palt, latitude=lat, longitude=lng,
        detector_altitude=dalt, rigidity=rig, energy=en,
        bfield_type="igrf",
    )
    traj.get_trajectory(dt=1e-5, max_time=1.)
    assert traj.particle_escaped == EXPECTED_IGRF_ESCAPED[idx], \
        f"escaped flag mismatch at case {idx}: got {traj.particle_escaped}, expected {EXPECTED_IGRF_ESCAPED[idx]}"
