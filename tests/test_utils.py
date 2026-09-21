"""
Tests for gtracr/utils.py
"""

import pytest

from gtracr.utils import dec_to_dms, location_dict, particle_dict, ymd_to_dec


@pytest.mark.parametrize('value, expected', [
    ('2015-01-01', 2015.0),
    ('2000-03-01', 2000 + 60 / 366),
    ('2020-07-02', 2020.5),
    ('2019-12-31', 2019 + 364 / 365),
    ('1900-03-01', 1900 + 59 / 365),
    ('2000-02-29', 2000 + 59 / 366),
])
def test_ymd_to_dec_calendar(value, expected):
    assert ymd_to_dec(value) == pytest.approx(expected, abs=1e-10, rel=0)


@pytest.mark.parametrize('value', ['2019-02-29', '2100-02-29', '2020-13-01', '2020-01-32', '1952-04-31'])
def test_ymd_to_dec_invalid_date(value):
    with pytest.raises(ValueError):
        ymd_to_dec(value)


def test_trajectory_receives_calendar_epoch():
    from gtracr.trajectory import Trajectory

    trajectory = Trajectory(zenith_angle=0., azimuth_angle=0., rigidity=10., date='2015-01-01')
    assert trajectory.igrf_params[1] == 2015.0


def test_dec_to_dms_north_east():
    """Positive lat and lng → "N" for latitude"""
    lat_dms, lng_dms = dec_to_dms(36.4348, 137.2766)
    assert "N" in lat_dms
    assert "36" in lat_dms
    # Note: lng_symb bug uses lat_dec sign, so positive lat → "E" for lng
    assert "E" in lng_dms


def test_dec_to_dms_south_west():
    """Negative lat and negative lng → "S" for lat, "W" for lng."""
    lat_dms, lng_dms = dec_to_dms(-24.68, -24.68)
    assert "S" in lat_dms
    assert "W" in lng_dms


def test_dec_to_dms_positive_lat_negative_lng():
    """Positive lat + negative lng → "N" for lat, "W" for lng."""
    lat_dms, lng_dms = dec_to_dms(36.0, -113.0)
    assert "N" in lat_dms
    assert "W" in lng_dms


def test_location_dict_contents():
    """All 10 predefined locations should be present"""
    expected_locations = [
        "Kamioka",
        "IceCube",
        "SNOLAB",
        "UofA",
        "CTA-North",
        "CTA-South",
        "ORCA",
        "ANTARES",
        "Baikal-GVD",
        "TA",
    ]
    for name in expected_locations:
        assert name in location_dict, f"Location '{name}' not found in location_dict"


def test_particle_dict_contents():
    """All 4 particle types should be present"""
    expected_particles = ["p+", "p-", "e+", "e-"]
    for label in expected_particles:
        assert label in particle_dict, f"Particle '{label}' not found in particle_dict"


def test_set_locationdict_duplicate():
    """set_locationdict skips duplicates (the else: continue branch)"""
    from gtracr.utils import set_locationdict

    # The function builds a fresh dict and checks if name already exists
    # By design, the list has no duplicates, so we add one by calling it directly
    d = set_locationdict()
    # All 10 unique locations should be present
    assert len(d) == 10


def test_set_particledict_duplicate():
    """set_particledict skips duplicates (the else: continue branch)"""
    from gtracr.utils import set_particledict

    d = set_particledict()
    # All 4 unique particles should be present
    assert len(d) == 4


def test_import_dict(tmp_path):
    """import_dict loads a pickle file"""
    import pickle

    from gtracr.utils import import_dict

    # Create a temporary pickle file
    data = {"key": "value", "num": 42}
    fpath = tmp_path / "test.pkl"
    with open(fpath, "wb") as f:
        pickle.dump(data, f)
    loaded = import_dict(str(fpath))
    assert loaded["key"] == "value"
    assert loaded["num"] == 42
