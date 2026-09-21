"""Custom geometry supplied by downstream atmosphere models is authoritative."""

import numpy as np

from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.location import Location
from gtracr.trajectory import Trajectory


def test_trajectory_custom_location_matches_coordinates():
    # Deliberately reuse a registered name with different coordinates.
    loc = Location("Kamioka", 32.1, -80.7, 0.005)
    common = dict(
        zenith_angle=30.0, azimuth_angle=60.0, rigidity=20.0, date="2000-01-01"
    )
    explicit = Trajectory(location_name=loc, **common)
    numeric = Trajectory(
        latitude=loc.latitude,
        longitude=loc.longitude,
        detector_altitude=loc.altitude,
        **common,
    )
    np.testing.assert_array_equal(
        explicit.particle_sixvector, numeric.particle_sixvector
    )
    assert explicit.lat == loc.latitude
    assert explicit.lng == loc.longitude


def test_batch_uses_explicit_coordinates(monkeypatch):
    import gtracr._libgtracr as lib

    captured = {}

    def evaluate(table, params, igrf, p):
        captured.update(lat=p.latitude, lon=p.longitude, altitude=p.detector_alt)
        return np.array([0.0]), np.array([0.0]), np.array([5.0]), 1

    monkeypatch.setattr(lib, "batch_gmrc_evaluate", evaluate)
    loc = Location("Kamioka", 32.1, -80.7, 0.005)
    gmrc = GMRC(
        location=loc, iter_num=1, bfield_type="igrf", date="2000-01-01", n_workers=1
    )
    gmrc.evaluate_batch(base_seed=42)
    assert captured == dict(lat=loc.latitude, lon=loc.longitude, altitude=loc.altitude)
    assert gmrc.data_dict["rcutoff"][0] == 5.0
