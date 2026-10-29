"""Analytic and cross-direction checks for backward intensity transport."""

import numpy as np
import pytest

from gtracr.mutracr import M_MU, TAU_MU, MuonTracer

RG = 6.3712e8
C = 2.99792458e10


def initial(energy=2.0, height=1e6):
    return [[0, 0, RG + height]], [[0, 0, -np.sqrt(energy**2 - M_MU**2)]]


def test_vacuum_survival_and_source_quadrature():
    energy, length = 2.0, 2e5
    mt = MuonTracer(bfield="none", atmosphere="none", dedx=0)
    x, p = initial(energy)
    out = mt.trace_backward(x, p, [-1], path_max=length, ctl_path=1000)
    lam = M_MU / (np.linalg.norm(p) * C * TAU_MU)
    assert out["fate"][0] == 5
    assert out["path"][0] == pytest.approx(length, abs=1e-7)
    assert out["x"][0, 2] == pytest.approx(x[0][2] + length, abs=1e-6)
    assert out["survival"][0] == pytest.approx(np.exp(-lam * length), rel=1e-12)
    assert out["jacobian"][0] == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(out["p"], p, rtol=1e-13, atol=1e-13)
    integral = -np.expm1(-lam * length) / lam
    assert out["bank"]["w"].sum() == pytest.approx(integral, rel=1e-6)
    assert np.all(out["bank"]["phat"][:, 2] < 0)


def test_constant_density_loss_range_decay():
    energy, end_energy = 0.4, 1.8
    a, rho = 2e-3, 1e-3
    mt = MuonTracer(bfield="none", atmosphere=("uniform", rho), dedx=a)
    x, p = initial(energy)
    out = mt.trace_backward(
        x,
        p,
        [1],
        e_max=end_energy,
        ctl_path=1000,
        ctl_loss=0.002,
        ctl_decay=0.002,
        survival_min=0,
    )
    length = (end_energy - energy) / (a * rho)
    hazard = (
        M_MU
        / (a * rho * C * TAU_MU)
        * (np.arccosh(end_energy / M_MU) - np.arccosh(energy / M_MU))
    )
    assert out["fate"][0] == 2
    assert out["path"][0] == pytest.approx(length, rel=1e-10)
    assert out["survival"][0] == pytest.approx(np.exp(-hazard), rel=1e-9)
    assert out["jacobian"][0] == pytest.approx(1, abs=1e-12)
    assert np.hypot(np.linalg.norm(out["p"]), M_MU) == pytest.approx(end_energy)


def test_variable_loss_focusing():
    from scipy.integrate import solve_ivp

    e0, rho, length = 0.5, 1e-3, 3e5
    grid = np.geomspace(0.05, 10, 101)
    a0, b = 0.002, 0.0004
    av = a0 + b * np.log(grid)
    mt = MuonTracer(bfield="none", atmosphere=("uniform", rho), dedx=(grid, av))
    x, p = initial(e0)
    out = mt.trace_backward(
        x, p, [-1], path_max=length, ctl_path=1000, ctl_loss=0.002, ctl_decay=0.002
    )
    exact = solve_ivp(
        lambda s, y: [rho * (a0 + b * np.log(y[0] - M_MU))],
        (0, length),
        [e0],
        rtol=1e-11,
        atol=1e-13,
    ).y[0, -1]
    ef = np.hypot(np.linalg.norm(out["p"]), M_MU)
    assert ef == pytest.approx(exact, rel=1e-10)
    jac = (a0 + b * np.log(ef - M_MU)) / (a0 + b * np.log(e0 - M_MU))
    assert out["jacobian"][0] == pytest.approx(jac, rel=1e-11)


def test_uniform_field_forward_backward_roundtrip():
    energy, length, nstep = 5.0, 1e6, 200
    mt = MuonTracer(bfield=("uniform", (0, 2e-4, 0)), atmosphere="none", dedx=0)
    x, p = initial(energy, height=5e6)
    back = mt.trace_backward(x, p, [1], path_max=length, ctl_path=length / nstep)
    beta = np.linalg.norm(p) / energy
    forward = mt.trace(
        back["x"],
        back["p"],
        [1],
        w_min=0,
        max_steps=nstep,
        dt_max=length / (nstep * beta * C),
        ctl_path=1e7,
    )
    np.testing.assert_allclose(forward["x"], x, atol=3e-6, rtol=0)
    np.testing.assert_allclose(forward["p"], p, atol=1e-10, rtol=0)


def test_absorbing_boundaries_and_fading():
    mt = MuonTracer(bfield="none", atmosphere="none", h_escape=2e6)
    x, p = initial()
    top = mt.trace_backward(x, p, [1], ctl_path=3e5)
    assert top["fate"][0] == 4
    assert np.linalg.norm(top["x"]) == pytest.approx(RG + 2e6, abs=1e-6)
    ground = mt.trace_backward(x, -np.asarray(p), [1], ctl_path=3e5)
    assert ground["fate"][0] == 1
    assert np.linalg.norm(ground["x"]) == pytest.approx(RG, abs=1e-6)
    fade = mt.trace_backward(x, p, [1], survival_min=0.8)
    assert fade["fate"][0] == 3
    assert fade["survival"][0] == pytest.approx(0.8, rel=1e-10)
    zero = mt.trace_backward([[0, 0, RG]], [[0, 0, 1]], [1])
    assert zero["fate"][0] == 1 and zero["path"][0] == 0


def test_thread_invariance_and_fused_signed_sources():
    mt = MuonTracer(bfield="none", atmosphere="none", dedx=0)
    x, p = initial()
    x = np.tile(x, (12, 1))
    p = np.tile(p, (12, 1))
    q = np.tile([1, -1], 6)
    vals = np.ones((2, 2, 2, 2, 3))
    vals[..., 1] = -2
    vals[0, ..., 2] = 3
    vals[1, ..., 2] = -3
    source = dict(h_cm=[0, 1.2e7], cos_theta=[-1, 1], e_mu=[0.2, 10], values=vals)
    one = mt.trace_backward(
        x, p, q, path_max=1e5, ctl_path=1000, n_threads=1, source=source
    )
    many = mt.trace_backward(
        x, p, q, path_max=1e5, ctl_path=1000, n_threads=3, source=source
    )
    for key in (
        "x",
        "p",
        "survival",
        "jacobian",
        "path",
        "steps_per_ray",
        "fate",
        "source_integral",
    ):
        np.testing.assert_array_equal(one[key], many[key])
    for key in one["bank"]:
        np.testing.assert_array_equal(one["bank"][key], many["bank"][key])
    base = np.bincount(one["bank"]["ray"], weights=one["bank"]["w"])
    np.testing.assert_allclose(one["source_integral"][:, 0], base, rtol=1e-14)
    np.testing.assert_allclose(one["source_integral"][:, 1], -2 * base, rtol=1e-14)
    np.testing.assert_allclose(one["source_integral"][:, 2], 3 * q * base, rtol=1e-14)
    no_bank = mt.trace_backward(
        x, p, q, path_max=1e5, ctl_path=1000, n_threads=2, bank=False, source=source
    )
    assert "bank" not in no_bank
    np.testing.assert_array_equal(no_bank["source_integral"], one["source_integral"])


def test_grazing_crossing_cannot_skip_ground():
    # Both the proposed step endpoint and midpoint are above ground, but the
    # short grazing chord crosses it twice BEFORE the midpoint.
    mt = MuonTracer(bfield="none", atmosphere="none", dedx=0)
    height, cosine, momentum = 0.1, 2e-5, 100.0
    r = RG + height
    x = [[r, 0, 0]]
    p = [[momentum * cosine, momentum * np.sqrt(1 - cosine**2), 0]]
    length = r * cosine - np.sqrt((r * cosine) ** 2 - height * (2 * RG + height))
    out = mt.trace_backward(x, p, [1], ctl_path=2e5, survival_min=0)
    assert out["fate"][0] == 1
    assert np.linalg.norm(out["x"]) == pytest.approx(RG, abs=1e-6)
    # The geometry subtracts Earth-scale radii: use an absolute cm tolerance.
    assert out["path"][0] == pytest.approx(length, abs=0.02)


@pytest.mark.parametrize("boundary", ["ground", "toa"])
@pytest.mark.parametrize("charge", [1, -1])
def test_exactly_tangent_boundary_uses_radial_acceleration(boundary, charge):
    mt = MuonTracer(
        bfield=("uniform", (0, 0, 5e-5)),
        atmosphere="none",
        dedx=0,
        h_escape=1.128e7,
    )
    radius = RG if boundary == "ground" else RG + mt.h_escape
    out = mt.trace_backward(
        [[radius, 0, 0]],
        [[0, 1, 0]],
        [charge],
        path_max=1e4,
        survival_min=0,
        ctl_path=1e4,
    )
    # q+ curves radially outward into the past; q- curves inward.
    exits_shell = (boundary == "ground" and charge == -1) or (
        boundary == "toa" and charge == 1
    )
    if exits_shell:
        assert out["fate"][0] == (1 if boundary == "ground" else 4)
        assert out["path"][0] == 0
        assert out["steps_per_ray"][0] == 0
        np.testing.assert_array_equal(out["x"], [[radius, 0, 0]])
        assert out["bank"]["w"].size == 0
    else:
        assert out["fate"][0] == 5
        assert out["path"][0] == 1e4
        assert np.linalg.norm(out["x"]) > RG
        assert np.linalg.norm(out["x"]) < RG + mt.h_escape


def test_unresolved_near_rest_energy_rejected():
    mt = MuonTracer(bfield="none", atmosphere="none", dedx=0)
    x, _ = initial()
    with pytest.raises(ValueError, match="kinetic energy"):
        mt.trace_backward(x, [[0, 0, 1e-12]], [1])


@pytest.mark.parametrize("bank", [False, True])
def test_empty_batch_and_compiled_source(bank):
    mt = MuonTracer(bfield="none", atmosphere="none")
    source = dict(
        h_cm=[0, 1e7],
        cos_theta=[-1, 1],
        e_mu=[0.2, 2],
        values=np.zeros((2, 2, 2, 2, 2)),
    )
    out = mt.trace_backward(
        np.empty((0, 3)), np.empty((0, 3)), [], bank=bank, source=source
    )
    assert out["x"].shape == (0, 3)
    assert out["steps_per_ray"].shape == (0,)
    assert out["source_integral"].shape == (0, 2)
    if bank:
        assert out["bank"]["x"].shape == (0, 3)
        assert out["bank"]["w"].shape == (0,)


@pytest.mark.parametrize(
    "bad_source",
    [
        {"h_cm": [0, 0]},
        {"cos_theta": [-1, 1.1]},
        {"e_mu": [M_MU, 2]},
        {"e_mu": [2, 1]},
        {"values": np.ones((2, 2, 2, 2))},
        {"values": np.ones((2, 2, 2, 2, 0))},
        {"values": np.full((2, 2, 2, 2, 1), np.nan)},
    ],
)
def test_malformed_compiled_source_rejected(bad_source):
    mt = MuonTracer(bfield="none", atmosphere="none")
    x, p = initial()
    source = dict(
        h_cm=[0, 1e7],
        cos_theta=[-1, 1],
        e_mu=[0.2, 2],
        values=np.ones((2, 2, 2, 2, 1)),
    )
    source.update(bad_source)
    with pytest.raises(ValueError):
        mt.trace_backward(x, p, [1], source=source)


@pytest.mark.parametrize(
    "kw",
    [
        {"ctl_path": 0},
        {"survival_min": 1},
        {"e_max": np.nan},
        {"path_max": -1},
        {"max_steps": 1.5},
        {"n_threads": -1},
    ],
)
def test_invalid_controls(kw):
    mt = MuonTracer(bfield="none", atmosphere="none")
    x, p = initial()
    with pytest.raises(ValueError):
        mt.trace_backward(x, p, [1], **kw)


def test_invalid_states_and_source():
    mt = MuonTracer(bfield="none", atmosphere="none")
    x, p = initial()
    for xx, pp, qq in [
        (x, p, [0]),
        (x, [[0, 0, 0]], [1]),
        ([[0, 0, RG - 1]], p, [1]),
        (x, [[0, 0, np.nan]], [1]),
        (x, np.tile(p, (2, 1)), [1]),
    ]:
        with pytest.raises(ValueError):
            mt.trace_backward(xx, pp, qq)
    with pytest.raises(ValueError):
        mt.trace_backward(
            x,
            p,
            [1],
            source=dict(
                h_cm=[0, 1],
                cos_theta=[-1, 1],
                e_mu=[1, 0.2],
                values=np.ones((2, 2, 2, 2, 1)),
            ),
        )
