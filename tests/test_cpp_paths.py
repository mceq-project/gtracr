"""Tests for the pybind11 C++ extension interface (gtracr._libgtracr)."""

from pathlib import Path

import numpy as np
import pytest

from gtracr._libgtracr import (
    MagneticField,
    TableParams,
    TrajectoryTracer,
    generate_igrf_table,
)
from gtracr.constants import EARTH_RADIUS, ELEMENTARY_CHARGE, KG_PER_GEVC2

DATA_DIR = str((Path(__file__).parent.parent / "src" / "gtracr" / "data").resolve())
_IGRF_PARAMS = (DATA_DIR, 2020.0)

# Proton at 100 km altitude, equatorial, moderate downward momentum
_VEC0 = [EARTH_RADIUS + 100e3, np.pi / 2.0, 0.0, 0.0, 0.0, 5.36e-19]


def _make_tracer(bfield_type="d", solver="r"):
    return TrajectoryTracer(
        ELEMENTARY_CHARGE,
        0.938 * KG_PER_GEVC2,
        EARTH_RADIUS + 100e3,
        10.0 * EARTH_RADIUS,
        1e-4,
        1000,
        bfield_type,
        _IGRF_PARAMS,
        solver,
        1e-3,
        1e-6,
    )


# ---------------------------------------------------------------------------
# Constructor variants
# ---------------------------------------------------------------------------


def test_cpp_tracer_constructor_igrf():
    """TrajectoryTracer constructs with bfield_type='i' (direct IGRF)."""
    tt = _make_tracer("i")
    assert tt is not None


def test_cpp_tracer_constructor_dipole():
    """TrajectoryTracer constructs with bfield_type='d' (dipole)."""
    tt = _make_tracer("d")
    assert tt is not None


def test_cpp_tracer_constructor_table():
    """TrajectoryTracer constructs with bfield_type='t' (tabulated IGRF)."""
    tt = _make_tracer("t")
    assert tt is not None


def test_cpp_tracer_constructor_shared_table():
    """TrajectoryTracer shared-table constructor accepts pre-built table."""
    table_flat, table_params = generate_igrf_table(DATA_DIR, 2020.0)
    tt = TrajectoryTracer(
        table_flat,
        table_params,
        ELEMENTARY_CHARGE,
        0.938 * KG_PER_GEVC2,
        EARTH_RADIUS + 100e3,
        10.0 * EARTH_RADIUS,
        1e-4,
        1000,
        _IGRF_PARAMS,
        "r",
        1e-3,
        1e-6,
    )
    assert tt is not None


# ---------------------------------------------------------------------------
# evaluate() and evaluate_and_get_trajectory()
# ---------------------------------------------------------------------------


def test_cpp_evaluate_runs():
    """evaluate() sets particle_escaped, final_time, and final_sixvector."""
    tt = _make_tracer()
    tt.evaluate(0.0, _VEC0)
    assert isinstance(tt.particle_escaped, bool)
    assert tt.final_time > 0.0
    assert len(tt.final_sixvector) == 6


def test_cpp_evaluate_and_get_trajectory():
    """evaluate_and_get_trajectory() returns a dict with all required keys."""
    tt = _make_tracer()
    data = tt.evaluate_and_get_trajectory(0.0, _VEC0)
    for key in ["t", "r", "theta", "phi", "pr", "ptheta", "pphi"]:
        assert key in data
    assert len(data["t"]) > 0


# ---------------------------------------------------------------------------
# Solver variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_char", ["r", "b", "a"])
def test_cpp_solvers(solver_char):
    """All three solvers (RK4, Boris, RK45) complete without error."""
    tt = _make_tracer(solver=solver_char)
    tt.evaluate(0.0, _VEC0)
    assert tt.final_time > 0.0


# ---------------------------------------------------------------------------
# find_cutoff_rigidity()
# ---------------------------------------------------------------------------


def test_find_cutoff_rigidity():
    """find_cutoff_rigidity() returns a non-negative float."""
    tt = _make_tracer("d")
    pos = tuple(_VEC0[:3])
    mom_unit = (0.0, 0.0, 1.0)
    mom_factor = ELEMENTARY_CHARGE * 5.36e-19
    rc = tt.find_cutoff_rigidity(pos, mom_unit, [5.0, 10.0, 50.0], mom_factor)
    assert rc >= 0.0


# ---------------------------------------------------------------------------
# generate_igrf_table() and TableParams
# ---------------------------------------------------------------------------


def test_generate_igrf_table():
    """generate_igrf_table() returns an ndarray and a TableParams object."""
    table_flat, table_params = generate_igrf_table(DATA_DIR, 2020.0)
    assert isinstance(table_flat, np.ndarray)
    assert isinstance(table_params, TableParams)
    assert table_params.Nr > 0
    assert table_params.Ntheta > 0
    assert table_params.Nphi > 0


# ---------------------------------------------------------------------------
# MagneticField (C++ dipole)
# ---------------------------------------------------------------------------


def test_cpp_magfield_values():
    """C++ MagneticField.values() returns a 3-element non-zero vector."""
    mf = MagneticField()
    vals = mf.values(EARTH_RADIUS, np.pi / 2.0, 0.0)
    assert len(vals) == 3
    assert np.linalg.norm(vals) > 0.0


# ---------------------------------------------------------------------------
# set_start_altitude() and reset()
# ---------------------------------------------------------------------------


def test_cpp_tracer_reset_and_set_altitude():
    """set_start_altitude() and reset() allow re-running the tracer."""
    tt = _make_tracer()
    tt.evaluate(0.0, _VEC0)

    tt.set_start_altitude(EARTH_RADIUS + 200e3)
    tt.reset()
    tt.evaluate(0.0, _VEC0)

    # Should have run again; final_time is reset by evaluate
    assert tt.final_time > 0.0
