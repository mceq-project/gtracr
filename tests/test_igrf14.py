"""IGRF-14 coefficients and epoch handling (C++ and Python evaluators)."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from gtracr._libgtracr import IGRF
from gtracr.bfield import IGRF13, IGRF14
from gtracr.constants import IGRF_JSON_FILE

DATA_DIR = Path(__file__).parent.parent / "src" / "gtracr" / "data"
POINTS = [(6.4e6, 0.9, 2.4), (6.4e6, 2.0, 5.0), (7.0e6, 1.4, 0.3), (6.5e6, 0.05, 1.0)]


@pytest.mark.parametrize(
    "year",
    [1902.0, 1903.0, 1957.5, 2011.0, 2013.0, 2016.0, 2017.0, 2019.0, 2021.0, 2024.0,
     2025.0, 2026.5, 2029.9],
)
def test_cpp_matches_python(year):
    """Both evaluators interpolate between the bracketing epochs (and
    extrapolate the 2025 model with its secular variation after 2025)."""
    cpp = IGRF(str(DATA_DIR / IGRF_JSON_FILE), year)
    py = IGRF14(year)
    for point in POINTS:
        b_cpp = np.array(cpp.values(*point))
        b_py = py.values(*point)
        assert np.linalg.norm(b_cpp - b_py) / np.linalg.norm(b_py) < 1e-5


def test_epoch_coefficients():
    """The coefficient files carry IGRF-14's 2020 (definitive) and 2025 models."""
    import json

    with open(DATA_DIR / IGRF_JSON_FILE) as f:
        coeffs = json.load(f)
    assert coeffs["2020.000000"]["model"] == "DGRF2020"
    assert coeffs["2020.000000"]["gh"][0] == pytest.approx(-29403.41)
    assert coeffs["2025.000000"]["model"] == "IGRF2025"
    assert coeffs["2025.000000"]["gh"][0] == pytest.approx(-29350.0)
    assert coeffs["2025.000000"]["gh_sv"][0] == pytest.approx(12.6)


def test_igrf13_alias_is_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        field = IGRF13(2020.0)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    np.testing.assert_allclose(field.values(*POINTS[0]), IGRF14(2020.0).values(*POINTS[0]))
