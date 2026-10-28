"""
Tests for the rigidity-aware escape criterion and the RK45 error norm.
"""

from pathlib import Path

import numpy as np
import pytest

import gtracr
from gtracr._libgtracr import TrajectoryTracer
from gtracr.constants import (
    EARTH_RADIUS,
    ELEMENTARY_CHARGE,
    KG_M_S_PER_GEVC,
    KG_PER_GEVC2,
    SPEED_OF_LIGHT,
)
from gtracr.escape import (
    DEFAULT_ESCAPE_FACTOR,
    DEFAULT_TIME_FACTOR,
    resolve_escape_scaling,
)
from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.location import Location
from gtracr.trajectory import Trajectory

DATA = str(Path(gtracr.__file__).parent / "data")
MASS = 0.938272 * KG_PER_GEVC2


def _tracer(bfield="d", solver="a", max_iter=100000, dt=1e-5, atol=1e-8, rtol=1e-8):
    return TrajectoryTracer(
        ELEMENTARY_CHARGE,
        MASS,
        100e3,
        10 * EARTH_RADIUS,
        dt,
        max_iter,
        bfield,
        (DATA, 2024.0),
        solver,
        atol,
        rtol,
    )


def _vertical_state(lat_deg, rigidity, palt_km=100.0):
    r = EARTH_RADIUS + palt_km * 1e3
    p = rigidity * KG_M_S_PER_GEVC
    return [r, np.radians(90.0 - lat_deg), 0.0, p, 0.0, 0.0]


def test_dipole_strength():
    """B0 from the degree-1 quadrature: exact for the dipole, |g1| for IGRF."""
    assert _tracer("d").dipole_b0() == pytest.approx(29404.8e-9, rel=1e-12)
    b0 = _tracer("i").dipole_b0()
    assert 29.0e-6 < b0 < 30.5e-6


def test_run_limits():
    """Escape radius and time budget follow max(10 RE, k C_St) and
    max(max_time, N r_esc / (beta c)); high rigidities keep the fixed values
    when the factors are small, and 0/0 restores the fixed criterion."""
    tr = _tracer("d", max_iter=100000, dt=1e-5)  # max_time = 1 s
    k, n = DEFAULT_ESCAPE_FACTOR, DEFAULT_TIME_FACTOR
    assert (tr.escape_factor, tr.time_factor) == (k, n)
    for rig in [0.2, 1.0, 5.0, 50.0]:
        p = rig * KG_M_S_PER_GEVC
        c_st = EARTH_RADIUS * np.sqrt(29404.8e-9 * EARTH_RADIUS * ELEMENTARY_CHARGE / p)
        assert tr.stormer_length(p) == pytest.approx(c_st, rel=1e-12)
        tr.reset()
        tr.evaluate(0.0, _vertical_state(80.0, rig))
        r_esc = max(10 * EARTH_RADIUS, k * c_st)
        beta = p / np.hypot(p, MASS * SPEED_OF_LIGHT)
        t_max = max(1.0, n * r_esc / (beta * SPEED_OF_LIGHT))
        assert tr.run_escape_radius == pytest.approx(r_esc, rel=1e-12)
        assert tr.run_max_time == pytest.approx(t_max, rel=1e-12)

    tr.set_escape_scaling(0.0, 0.0)
    tr.reset()
    tr.evaluate(0.0, _vertical_state(80.0, 0.2))
    assert tr.run_escape_radius == 10 * EARTH_RADIUS
    assert tr.run_max_time == pytest.approx(1.0)

    tr.set_escape_scaling(1.0, 0.0)  # radius scaled, fixed time
    tr.reset()
    tr.evaluate(0.0, _vertical_state(80.0, 0.2))
    assert tr.run_escape_radius == pytest.approx(
        tr.stormer_length(0.2 * KG_M_S_PER_GEVC)
    )
    assert tr.run_max_time == pytest.approx(1.0)


def test_slow_escape_counted_allowed():
    """Dipole, 60 deg vertical, 0.9 GV: above the Störmer cutoff (0.85 GV) the
    trajectory needs more than 1 s to reach 10 RE. The fixed criterion calls it
    forbidden; the rigidity-aware one lets it escape."""
    state = _vertical_state(60.0, 0.9)
    tr = _tracer("d")
    tr.set_escape_scaling(0.0, 0.0)
    tr.evaluate(0.0, list(state))
    assert not tr.particle_escaped
    tr = _tracer("d")
    tr.evaluate(0.0, list(state))
    assert tr.particle_escaped
    assert tr.final_time > 1.0


def test_dipole_lower_cutoff_approaches_stormer():
    """With the default criterion, R_L of the vertical dipole cutoff at 60 deg
    lies within 1 % of Störmer's value; the fixed criterion is > 10 % high."""
    palt = 100.0
    r = EARTH_RADIUS + palt * 1e3
    rv = (
        SPEED_OF_LIGHT
        * 29404.8e-9
        * EARTH_RADIUS**3
        * np.cos(np.radians(60.0)) ** 4
        / (4 * r * r)
    )
    stormer = rv * ELEMENTARY_CHARGE / (KG_M_S_PER_GEVC * SPEED_OF_LIGHT)

    def r_lower(**kw):
        g = GMRC(
            location=Location("t", 60.0, 0.0, 0.0),
            particle_altitude=palt,
            bfield_type="dipole",
            solver="rk45",
            method="penumbra",
            min_rigidity=0.6,
            max_rigidity=1.5,
            ladder_delta=0.005,
            coarse=1,
            n_workers=4,
            **kw,
        )
        g.evaluate_batch(zenith=[0.0], azimuth=[0.0])
        return g.data_dict["r_lower"][0]

    assert r_lower() / stormer == pytest.approx(1.0, abs=0.01)
    assert r_lower(escape_factor=0, time_factor=0) / stormer > 1.10


def test_python_plumbing():
    with pytest.raises(ValueError):
        resolve_escape_scaling(-1.0, None)
    assert resolve_escape_scaling() == (DEFAULT_ESCAPE_FACTOR, DEFAULT_TIME_FACTOR)
    g = GMRC(escape_factor=0, time_factor=0)
    assert (g.escape_factor, g.time_factor) == (0.0, 0.0)

    def final(**kw):
        tr = Trajectory(
            zenith_angle=0.0,
            azimuth_angle=0.0,
            rigidity=0.9,
            location_name=Location("t", 60.0, 0.0, 0.0),
            bfield_type="dipole",
            solver="rk45",
            **kw,
        )
        tr.get_trajectory(dt=1e-5, max_time=1.0)
        return tr.particle_escaped

    assert final()
    assert not final(escape_factor=0, time_factor=0)


def test_rk45_tolerance_controls_momentum():
    """The RK45 norm measures momentum relative to |p|, so the drift of |p|
    (the field does no work) over a 1 s bound trajectory scales with rtol."""
    drift = {}
    for rtol in [1e-5, 1e-8]:
        tr = _tracer("d", solver="a", atol=0.0, rtol=rtol, max_iter=100000)
        tr.set_escape_scaling(0.0, 0.0)
        state = _vertical_state(60.0, 0.5)  # deep below the cutoff: bound
        tr.evaluate(0.0, list(state))
        assert not tr.particle_escaped
        assert tr.final_time == pytest.approx(1.0)
        f = np.asarray(tr.final_sixvector)
        drift[rtol] = abs(np.linalg.norm(f[3:]) / state[3] - 1.0)
        assert drift[rtol] < 300 * rtol
    assert drift[1e-8] < drift[1e-5] / 100
