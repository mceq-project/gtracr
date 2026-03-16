"""
Tests for Python B-field wrappers: MagneticField (dipole), IGRF13, and IGRFTable.
"""

from pathlib import Path

import numpy as np
import pytest

from gtracr.constants import EARTH_RADIUS

DATA_DIR = Path(__file__).parent.parent / "src" / "gtracr" / "data"


# ---------------------------------------------------------------------------
# MagneticField (dipole) — Python wrapper
# ---------------------------------------------------------------------------


def test_dipole_equator_finite():
    """MagneticField at equator (theta=pi/2) returns finite values."""
    from gtracr.bfield.dipole import MagneticField

    bf = MagneticField()
    result = bf.values(EARTH_RADIUS, np.pi / 2.0, 0.0)
    assert len(result) == 3
    assert all(np.isfinite(v) for v in result)


def test_dipole_pole_finite():
    """MagneticField at north pole (theta≈0) returns finite values."""
    from gtracr.bfield.dipole import MagneticField

    bf = MagneticField()
    result = bf.values(EARTH_RADIUS, 1e-9, 0.0)
    assert len(result) == 3
    assert all(np.isfinite(v) for v in result)


def test_dipole_analytical():
    """Spot-check: dipole B at equator (theta=pi/2) — Br should be zero."""
    from gtracr.bfield.dipole import MagneticField

    bf = MagneticField()
    br, bth, bph = bf.values(EARTH_RADIUS, np.pi / 2.0, 0.0)
    # At equator the radial component of a dipole is zero.
    # Use atol=1e-15 T to accommodate floating-point error in cos(pi/2).
    assert np.isclose(br, 0.0, atol=1e-15)


def test_dipole_r_scaling():
    """Dipole B magnitude scales as 1/r^3."""
    from gtracr.bfield.dipole import MagneticField

    bf = MagneticField()
    b1 = np.linalg.norm(bf.values(EARTH_RADIUS, np.pi / 2.0, 0.0))
    b2 = np.linalg.norm(bf.values(2.0 * EARTH_RADIUS, np.pi / 2.0, 0.0))
    ratio = b1 / b2
    assert np.isclose(ratio, 8.0, rtol=1e-6)  # (2RE/RE)^3 = 8


# ---------------------------------------------------------------------------
# IGRF13 — Python wrapper
# ---------------------------------------------------------------------------


def test_igrf13_default_nmax():
    """IGRF13 default nmax is 13."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(2020.0)
    assert igrf.nmax == 13


def test_igrf13_custom_nmax():
    """IGRF13 with nmax=5 initializes without error."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(2020.0, nmax=5)
    assert igrf.nmax == 5
    result = igrf.values(EARTH_RADIUS, np.pi / 2.0, 0.0)
    assert all(np.isfinite(v) for v in result)


@pytest.mark.parametrize("year", [1900, 1950, 2020, 2024])
def test_igrf13_date_parametrize(year):
    """IGRF13 returns finite values across all epochs."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(float(year))
    result = igrf.values(EARTH_RADIUS, np.pi / 2.0, np.pi)
    assert all(np.isfinite(v) for v in result)


def test_igrf13_values_finite():
    """Spot-check multiple (r, theta, phi) coordinates."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(2020.0)
    coords = [
        (EARTH_RADIUS, np.pi / 2.0, 0.0),
        (2.0 * EARTH_RADIUS, 1.0, np.pi / 4.0),
        (5.0 * EARTH_RADIUS, np.pi / 3.0, np.pi),
    ]
    for r, theta, phi in coords:
        result = igrf.values(r, theta, phi)
        assert all(np.isfinite(v) for v in result), (
            f"Non-finite at ({r}, {theta}, {phi})"
        )


def test_igrf13_leap_year():
    """IGRF13 with a leap year initializes correctly."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(2000)  # 2000 is a leap year
    result = igrf.values(EARTH_RADIUS, np.pi / 2.0, np.pi)
    assert len(result) == 3


def test_igrf13_large_theta_normalization():
    """IGRF13 normalizes theta_deg > 180 without crashing."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(2015)
    # theta_rad = 4.0 → ~229 degrees > 180, triggers normalization
    result = igrf.values(EARTH_RADIUS, 4.0, np.pi)
    assert len(result) == 3


def test_igrf13_large_phi_normalization():
    """IGRF13 normalizes phi_deg > 360 without crashing."""
    from gtracr.bfield.igrf import IGRF13

    igrf = IGRF13(2015)
    # phi_rad = 7.5 → ~430 degrees > 360, triggers normalization
    result = igrf.values(EARTH_RADIUS, np.pi / 2.0, 7.5)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# IGRFTable — Python lookup table wrapper
# ---------------------------------------------------------------------------


def test_igrf_table_values_finite():
    """IGRFTable.values() returns finite floats at a sample point."""
    from gtracr.bfield.table import NPHI, NR, NTHETA, IGRFTable

    # Build a constant table manually to avoid running the full build loop
    table = IGRFTable.__new__(IGRFTable)
    table._r_min = EARTH_RADIUS
    table._r_max = 10.0 * EARTH_RADIUS
    table._log_r_min = np.log(EARTH_RADIUS)
    table._log_r_max = np.log(10.0 * EARTH_RADIUS)
    table._table = np.ones((3, NR, NTHETA, NPHI), dtype=np.float32) * 1e-5

    result = table.values(EARTH_RADIUS, np.pi / 2.0, 0.0)
    assert len(result) == 3
    assert all(np.isfinite(v) for v in result)


def test_igrf_table_phi_periodicity():
    """IGRFTable.values() at phi=0 and phi=2*pi returns the same result."""
    from gtracr.bfield.table import NPHI, NR, NTHETA, IGRFTable

    table = IGRFTable.__new__(IGRFTable)
    table._r_min = EARTH_RADIUS
    table._r_max = 10.0 * EARTH_RADIUS
    table._log_r_min = np.log(EARTH_RADIUS)
    table._log_r_max = np.log(10.0 * EARTH_RADIUS)
    # Set a non-uniform table so interpolation results depend on phi
    rng = np.random.default_rng(42)
    table._table = rng.random((3, NR, NTHETA, NPHI)).astype(np.float32)

    r = EARTH_RADIUS * 2.0
    theta = np.pi / 2.0
    result0 = table.values(r, theta, 0.0)
    result2pi = table.values(r, theta, 2.0 * np.pi)
    # phi=0 and phi=2pi should map to the same cell (ip0=0 in both cases)
    for v0, v2 in zip(result0, result2pi):
        assert np.isclose(v0, v2, rtol=1e-5)


def test_igrf_table_shape_constant():
    """IGRFTable NR, NTHETA, NPHI match expected C++ grid dimensions."""
    from gtracr.bfield.table import NPHI, NR, NTHETA

    assert NR == 64
    assert NTHETA == 128
    assert NPHI == 256


def test_igrf_table_r_clamping():
    """IGRFTable.values() clamps r below r_min without crashing."""
    from gtracr.bfield.table import NPHI, NR, NTHETA, IGRFTable

    table = IGRFTable.__new__(IGRFTable)
    table._r_min = EARTH_RADIUS
    table._r_max = 10.0 * EARTH_RADIUS
    table._log_r_min = np.log(EARTH_RADIUS)
    table._log_r_max = np.log(10.0 * EARTH_RADIUS)
    table._table = np.ones((3, NR, NTHETA, NPHI), dtype=np.float32)

    # r below r_min should be clamped
    result = table.values(EARTH_RADIUS * 0.5, np.pi / 2.0, 0.0)
    assert all(np.isfinite(v) for v in result)


def test_igrf_table_r_max_clamping():
    """IGRFTable.values() clamps r above r_max without crashing."""
    from gtracr.bfield.table import NPHI, NR, NTHETA, IGRFTable

    table = IGRFTable.__new__(IGRFTable)
    table._r_min = EARTH_RADIUS
    table._r_max = 10.0 * EARTH_RADIUS
    table._log_r_min = np.log(EARTH_RADIUS)
    table._log_r_max = np.log(10.0 * EARTH_RADIUS)
    table._table = np.ones((3, NR, NTHETA, NPHI), dtype=np.float32)

    # r above r_max should be clamped
    result = table.values(EARTH_RADIUS * 20.0, np.pi / 2.0, 0.0)
    assert all(np.isfinite(v) for v in result)
