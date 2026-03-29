"""Tests for the pure-Python fallback trajectory tracer (pTrajectoryTracer)."""

import numpy as np
import pytest

from gtracr._fallback import pTrajectoryTracer
from gtracr.constants import (
    EARTH_RADIUS,
    ELEMENTARY_CHARGE,
    KG_M_S_PER_GEVC,
    KG_PER_GEVC2,
)

_DATA_DIR = str(
    (
        __import__("pathlib").Path(__file__).parent.parent / "src" / "gtracr" / "data"
    ).resolve()
)
_IGRF_PARAMS = (_DATA_DIR, 2020.0)

# pTrajectoryTracer uses start_altitude as altitude above Earth's surface (metres).
_START_ALT = 100e3  # 100 km in metres

# A proton at 100 km altitude, equatorial, with moderate momentum
_VEC0 = np.array([EARTH_RADIUS + _START_ALT, np.pi / 2.0, 0.0, 0.0, 0.0, 5.36e-19])


def _make_tracer(bfield_type="d", max_step=500):
    return pTrajectoryTracer(
        charge=ELEMENTARY_CHARGE,
        mass=0.938 * KG_PER_GEVC2,
        start_altitude=_START_ALT,
        escape_radius=10.0 * EARTH_RADIUS,
        stepsize=1e-4,
        max_step=max_step,
        bfield_type=bfield_type,
        igrf_params=_IGRF_PARAMS,
    )


def test_ptracer_dipole_evaluate():
    """evaluate() runs without error and sets final_time and final_sixvector."""
    tt = _make_tracer("d")
    tt.evaluate(0.0, _VEC0.copy())
    assert tt.final_time > 0.0
    assert len(tt.final_sixvector) == 6
    assert all(np.isfinite(v) for v in tt.final_sixvector)


def test_ptracer_igrf_evaluate():
    """evaluate() works with bfield_type='i' (IGRF-13)."""
    tt = _make_tracer("i")
    tt.evaluate(0.0, _VEC0.copy())
    assert tt.final_time > 0.0


def test_ptracer_dipole_get_trajectory():
    """evaluate_and_get_trajectory() returns dict with all required keys."""
    tt = _make_tracer("d")
    data = tt.evaluate_and_get_trajectory(0.0, _VEC0.copy())
    for key in ["t", "r", "theta", "phi", "pr", "ptheta", "pphi"]:
        assert key in data
    assert len(data["t"]) > 0


def test_ptracer_escaped_branch():
    """Radially outward high-momentum proton escapes (exercises lines 156-157)."""
    # Radial outward momentum at ~0.9998c guarantees escape within max_step steps.
    mom_si = 50.0 * KG_M_S_PER_GEVC
    vec0 = np.array([EARTH_RADIUS + _START_ALT, np.pi / 2.0, 0.0, mom_si, 0.0, 0.0])
    tt = pTrajectoryTracer(
        charge=ELEMENTARY_CHARGE,
        mass=0.938 * KG_PER_GEVC2,
        start_altitude=_START_ALT,
        escape_radius=10.0 * EARTH_RADIUS,
        stepsize=1e-5,
        max_step=200000,
        bfield_type="d",
    )
    tt.evaluate(0.0, vec0)
    assert tt.particle_escaped is True


def test_ptracer_forbidden_branch_evaluate():
    """Low-rigidity trajectory terminates at atmosphere (line 159 in evaluate)."""
    # Near-zero momentum: particle barely moves, will be pulled back inward.
    mom_si = 1e-22
    vec0 = np.array([EARTH_RADIUS + _START_ALT, np.pi / 2.0, 0.0, 0.0, 0.0, mom_si])
    tt = pTrajectoryTracer(
        charge=ELEMENTARY_CHARGE,
        mass=0.938 * KG_PER_GEVC2,
        start_altitude=_START_ALT,
        escape_radius=10.0 * EARTH_RADIUS,
        stepsize=1e-5,
        max_step=200000,
        bfield_type="d",
    )
    tt.evaluate(0.0, vec0)
    assert tt.particle_escaped is False


def test_ptracer_forbidden_branch_get_trajectory():
    """Forbidden trajectory in evaluate_and_get_trajectory() (line 199)."""
    mom_si = 1e-22
    vec0 = np.array([EARTH_RADIUS + _START_ALT, np.pi / 2.0, 0.0, 0.0, 0.0, mom_si])
    tt = pTrajectoryTracer(
        charge=ELEMENTARY_CHARGE,
        mass=0.938 * KG_PER_GEVC2,
        start_altitude=_START_ALT,
        escape_radius=10.0 * EARTH_RADIUS,
        stepsize=1e-5,
        max_step=200000,
        bfield_type="d",
    )
    data = tt.evaluate_and_get_trajectory(0.0, vec0)
    assert tt.particle_escaped is False
    assert len(data["t"]) > 0


def test_ptracer_escaped_branch_get_trajectory():
    """Escaped trajectory in evaluate_and_get_trajectory() (exercises lines 196-197)."""
    mom_si = 50.0 * KG_M_S_PER_GEVC
    vec0 = np.array([EARTH_RADIUS + _START_ALT, np.pi / 2.0, 0.0, mom_si, 0.0, 0.0])
    tt = pTrajectoryTracer(
        charge=ELEMENTARY_CHARGE,
        mass=0.938 * KG_PER_GEVC2,
        start_altitude=_START_ALT,
        escape_radius=10.0 * EARTH_RADIUS,
        stepsize=1e-5,
        max_step=200000,
        bfield_type="d",
    )
    data = tt.evaluate_and_get_trajectory(0.0, vec0)
    assert tt.particle_escaped is True
    assert len(data["t"]) > 0


def test_ptracer_invalid_bfield_raises():
    """Unknown bfield_type raises ValueError."""
    with pytest.raises(ValueError):
        pTrajectoryTracer(
            charge=ELEMENTARY_CHARGE,
            mass=0.938 * KG_PER_GEVC2,
            bfield_type="x",
        )


def test_ptracer_igrf_missing_params_raises():
    """bfield_type='i' without igrf_params raises ValueError."""
    with pytest.raises(ValueError, match="igrf_params"):
        pTrajectoryTracer(
            charge=ELEMENTARY_CHARGE,
            mass=0.938 * KG_PER_GEVC2,
            bfield_type="i",
            igrf_params=None,
        )


def test_trajectory_use_python_flag():
    """Trajectory.get_trajectory(use_python=True) exercises pTrajectoryTracer."""
    from gtracr.trajectory import Trajectory

    # Use a fixed IGRF-13 date (within its range of 1900-2025).
    traj = Trajectory(
        zenith_angle=45.0,
        azimuth_angle=90.0,
        rigidity=10.0,
        location_name="Kamioka",
        bfield_type="igrf",
        date="2020-06-15",
    )
    traj.get_trajectory(dt=1e-4, max_time=0.05, use_python=True)
    assert traj.final_time > 0.0
