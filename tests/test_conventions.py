"""Pin the arrival-direction convention (azimuth from north, clockwise)."""

import numpy as np
import pytest

from gtracr.trajectory import Trajectory


def _escapes(azimuth, rigidity, zenith=60.0):
    traj = Trajectory(
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        rigidity=rigidity,
        latitude=0.0,
        longitude=0.0,
        bfield_type="dipole",
    )
    traj.get_trajectory(dt=1e-5, max_time=0.3)
    return traj.particle_escaped


def _cutoff(azimuth, rigidities):
    for rigidity in rigidities:
        if _escapes(azimuth, rigidity):
            return rigidity
    return np.inf


@pytest.mark.parametrize(
    "azimuth, source",
    [
        (0.0, (0.0, 1.0, 0.0)),  # from north
        (90.0, (1.0, 0.0, 0.0)),  # from east
        (180.0, (0.0, -1.0, 0.0)),  # from south
        (270.0, (-1.0, 0.0, 0.0)),  # from west
    ],
)
def test_local_frame(azimuth, source):
    """Unit vector toward the source in the local frame x = E, y = N, z = U."""
    traj = Trajectory(
        zenith_angle=90.0, azimuth_angle=azimuth, rigidity=10.0, latitude=0.0
    )
    np.testing.assert_allclose(
        traj.get_particle_coord(altitude=0.0, magnitude=1.0), source, atol=1e-12
    )


def test_upgoing_start_point():
    """An up-going arrival from the north starts on the Earth's far side, north."""
    lat, lng = 10.0, 20.0
    start = {}
    for azimuth in [0.0, 90.0]:
        traj = Trajectory(
            zenith_angle=120.0,
            azimuth_angle=azimuth,
            rigidity=10.0,
            latitude=lat,
            longitude=lng,
            particle_altitude=100.0,
        )
        theta, phi = traj.particle_sixvector[1:3]
        start[azimuth] = (90.0 - np.degrees(theta), np.degrees(phi))
    assert start[0.0][0] > lat + 30.0 and start[0.0][1] == pytest.approx(lng)
    assert start[90.0][1] > lng + 30.0


def test_east_west_effect():
    """Protons from the east (azimuth 90) have higher cutoffs than from the west."""
    rigidities = np.arange(10.0, 61.0, 2.5)
    assert _cutoff(90.0, rigidities) > _cutoff(270.0, rigidities) + 10.0
