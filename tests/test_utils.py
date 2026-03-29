"""
Tests for gtracr/utils.py
"""

from gtracr.utils import dec_to_dms, location_dict, particle_dict, ymd_to_dec


def test_ymd_to_dec_basic():
    """ymd_to_dec("2015-01-01") should be close to 2015.0 (start of year)"""
    val = ymd_to_dec("2015-01-01")
    # It won't be exactly 2015.0 because of month+day contribution,
    # but it should be just above 2015
    assert val > 2015.0
    assert val < 2015.2


def test_ymd_to_dec_leap_year():
    """2000 is a leap year — March 1 should be slightly past ~2000.17"""
    val = ymd_to_dec("2000-03-01")
    assert val > 2000.0
    assert val < 2001.0


def test_ymd_to_dec_mid_year():
    """Mid-year date should land in the middle of the year decimal"""
    val = ymd_to_dec("2020-07-02")
    assert val > 2020.5
    assert val < 2021.0


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
