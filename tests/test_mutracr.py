"""
Tests for MuonTracer ("mutracr") — batch atmospheric-muon transport.

Layers:
1. analytic anchors (helix, CSDA range, vacuum decay weight),
2. exact parity against the vendored numpy reference
   (gtracr._mu_reference, the nu3d stage-C tracer),
3. deposit consumers (bank / spectrum / angular) cross-consistency,
4. field/atmosphere/dedx probes,
5. threading invariance,
6. Michel kernel norms and moments.
"""

import numpy as np
import pytest

from gtracr import _mu_reference as ref
from gtracr.mutracr import M_MU, TAU_MU, MuonTracer, lab_spectrum, rest_kernel

C_CMS = ref.C_CMS
KP = ref.KP
R_G = 6.3712e8  # cm

# ---------------------------------------------------------------------------
# Analytic anchors
# ---------------------------------------------------------------------------


def test_helix_uniform_field():
    """Gyroradius, orbit closure, |p| conservation, bend sign."""
    b0 = 3e-4
    mt = MuonTracer(bfield=("uniform", (0.0, 0.0, b0)), atmosphere="none")
    p0 = 1.0
    e0 = np.hypot(p0, M_MU)
    beta = p0 / e0
    omega = KP * beta * b0 / p0
    r_gyro = beta * C_CMS / omega
    period = 2 * np.pi / omega
    x0 = np.array([[R_G + 5e6, 0.0, 0.0]])
    pv = np.array([[p0, 0.0, 0.0]])

    res = mt.trace(x0, pv, [1], dt_max=period / 2000, max_steps=1000, w_min=0.0)
    diam = np.linalg.norm(res["x"][0] - x0[0])
    assert abs(diam - 2 * r_gyro) / (2 * r_gyro) < 1e-5
    assert res["x"][0, 1] < 0.0  # q>0, B along z, p along x -> curves to -y

    res = mt.trace(x0, pv, [1], dt_max=period / 2000, max_steps=2000, w_min=0.0)
    assert np.linalg.norm(res["x"][0] - x0[0]) / r_gyro < 1e-4
    assert abs(np.linalg.norm(res["p"][0]) - p0) / p0 < 1e-9


def test_csda_range():
    """B=0 constant-density range: (E0 - E_stop) / (a rho)."""
    rho0, a_loss = 1.2e-3, 2.0e-3
    mt = MuonTracer(
        bfield="none", atmosphere=("uniform", rho0), dedx=a_loss, r_ground=R_G
    )
    x0 = np.array([[0.0, 0.0, R_G + 1.0e7]])
    pv = np.array([[0.0, 0.0, -np.sqrt(1.0 - M_MU**2)]])
    res = mt.trace(x0, pv, [1], e_min=0.2, w_min=0.0)
    e_end = np.hypot(np.linalg.norm(res["p"][0]), M_MU)
    dist = np.linalg.norm(res["x"][0] - x0[0])
    l_pred = (1.0 - e_end) / (a_loss * rho0)
    assert abs(dist - l_pred) / l_pred < 1e-4
    assert res["fate"][0] == 2  # stopped


def test_vacuum_decay_weight():
    """w(t) = exp(-m t / (E tau)); audit closes exactly in vacuum."""
    mt = MuonTracer(bfield="none", atmosphere="none")
    e0 = np.hypot(1.0, M_MU)
    n_steps, dt = 200, 1e-7
    res = mt.trace(
        [[R_G + 1e5, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [1],
        dt_max=dt,
        max_steps=n_steps,
        w_min=0.0,
        ctl_decay=1e9,
        deposit=dict(kind="bank"),
    )
    w_pred = np.exp(-M_MU * n_steps * dt / (e0 * TAU_MU))
    assert abs(res["w"][0] - w_pred) < 1e-12
    assert abs(res["weight_audit"] - 1.0) < 1e-12
    assert len(res["bank"]["w"]) == n_steps
    assert abs(res["bank"]["w"].sum() - res["deposited"]) < 1e-15


def test_ground_pullback_on_sphere():
    """'ground' rays end exactly on the ground sphere."""
    mt = MuonTracer(bfield="none", atmosphere="usstd", r_ground=R_G)
    res = mt.trace(
        [[0.0, 0.0, R_G + 2e6]],
        [[0.0, 0.0, -3.0]],
        [-1],
        w_min=0.0,
    )
    assert res["fate"][0] == 1
    r_end = np.linalg.norm(res["x"][0])
    assert abs(r_end - R_G) / R_G < 1e-9


def test_escape_fate():
    """Upward ray in vacuum escapes above r_ground + h_escape."""
    mt = MuonTracer(bfield="none", atmosphere="none", r_ground=R_G)
    res = mt.trace([[0.0, 0.0, R_G + 1e5]], [[0.0, 0.0, 10.0]], [1], w_min=0.0)
    assert res["fate"][0] == 4
    assert res["fate_w"]["escaped"] > 0.0


def test_emin_clamped_above_muon_mass():
    """e_min <= m_mu must still terminate as 'stopped' (clamp)."""
    mt = MuonTracer(
        bfield="none", atmosphere=("uniform", 1.2e-3), dedx=2.0e-3, r_ground=R_G
    )
    res = mt.trace(
        [[0.0, 0.0, R_G + 1.0e7]],
        [[0.0, 0.0, -0.5]],
        [1],
        e_min=0.0,  # at/below m_mu — would never trigger without clamp
        w_min=0.0,
        max_steps=50000,
    )
    assert res["fate"][0] == 2


# ---------------------------------------------------------------------------
# Parity against the vendored numpy reference
# ---------------------------------------------------------------------------


def _random_batch(n, rng, r_ground):
    """Mixed source: 5-30 km altitudes, 0-70 deg tilts, 0.2-20 GeV."""
    h = rng.uniform(5e5, 3e6, n)
    tilt = np.radians(rng.uniform(0, 70, n))
    az = rng.uniform(0, 2 * np.pi, n)
    lat = np.arccos(rng.uniform(-0.9, 0.9, n))
    lon = rng.uniform(0, 2 * np.pi, n)
    r = r_ground + h
    x = np.stack(
        [r * np.sin(lat) * np.cos(lon), r * np.sin(lat) * np.sin(lon), r * np.cos(lat)],
        1,
    )
    rhat = x / r[:, None]
    a = np.zeros_like(x)
    a[:, 0] = 1.0
    b = np.cross(rhat, a)
    nb = np.linalg.norm(b, axis=1)
    m = nb < 1e-8
    a[m, 0], a[m, 1] = 0.0, 1.0
    b[m] = np.cross(rhat[m], a[m])
    e1 = b / np.linalg.norm(b, axis=1)[:, None]
    e2 = np.cross(rhat, e1)
    d = -np.cos(tilt)[:, None] * rhat + np.sin(tilt)[:, None] * (
        np.cos(az)[:, None] * e1 + np.sin(az)[:, None] * e2
    )
    ekin = np.exp(rng.uniform(np.log(0.2), np.log(20), n))
    pm = np.sqrt((ekin + M_MU) ** 2 - M_MU**2)
    p = pm[:, None] * d
    q = rng.choice([-1, 1], n)
    pol = rng.uniform(-1, 1, n)
    w = rng.uniform(0.5, 2.0, n)
    return x, p, q, pol, w


def test_parity_vs_numpy_reference():
    """Dipole + US-Std + decays: final states and fates match the
    numpy reference at float-rounding level."""
    rng = np.random.default_rng(42)
    r_ground = 6.391e8  # MCEq ground radius
    x, p, q, pol, w = _random_batch(300, rng, r_ground)

    batch = ref.MuonBatch(x.copy(), p.copy(), q, pol=pol, w=w.copy())
    bank_ref = ref.DecayBank()
    dref = ref.trace(
        batch,
        ref.AxialDipole(),
        ref.USStdAtmosphere(),
        deposit=bank_ref,
        r_ground=r_ground,
        e_min=0.115,
        w_min=0.0,
    )

    mt = MuonTracer(bfield="dipole", atmosphere="usstd", r_ground=r_ground)
    res = mt.trace(
        x,
        p,
        q,
        pol=pol,
        w=w,
        e_min=0.115,
        w_min=0.0,
        n_threads=1,
        deposit=dict(kind="bank"),
    )

    assert res["steps"] == dref["steps"]
    assert abs(res["deposited"] - dref["deposited"]) / dref["deposited"] < 1e-12
    np.testing.assert_allclose(res["x"], batch.x, rtol=0, atol=1e-6)  # cm
    np.testing.assert_allclose(res["p"], batch.p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["w"], batch.w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["th2"], batch.th2, rtol=1e-10, atol=1e-18)
    for k in ("ground", "stopped", "faded", "escaped", "maxsteps"):
        assert abs(res["fate_w"][k] - dref["fate"][k]) < 1e-9
    assert len(res["bank"]["w"]) == len(bank_ref.arrays()["w"])


def test_parity_decay_beta_one():
    """The pre-fix MCEq decay convention flag matches the reference."""
    rng = np.random.default_rng(7)
    x, p, q, pol, w = _random_batch(50, rng, R_G)
    batch = ref.MuonBatch(x.copy(), p.copy(), q, pol=pol, w=w.copy())
    dref = ref.trace(
        batch,
        ref.UniformField([0.0, 0.0, 0.0]),
        ref.USStdAtmosphere(),
        r_ground=R_G,
        w_min=0.0,
        decay_beta_one=True,
    )
    mt = MuonTracer(bfield="none", atmosphere="usstd", r_ground=R_G)
    res = mt.trace(x, p, q, pol=pol, w=w, w_min=0.0, decay_beta_one=True, n_threads=1)
    assert res["steps"] == dref["steps"]
    np.testing.assert_allclose(res["w"], batch.w, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Deposit consumers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traced_with_bank():
    rng = np.random.default_rng(11)
    x, p, q, pol, w = _random_batch(200, rng, R_G)
    mt = MuonTracer(bfield="dipole", atmosphere="usstd", r_ground=R_G)
    kw = dict(pol=pol, w=w, e_min=0.115, w_min=0.0)
    e_edges = np.geomspace(M_MU * 1.001 + 0.02, 40.0 + M_MU, 97)
    d_edges = np.concatenate([[0.0], np.geomspace(2e-3, 1.6, 76), [np.pi]])
    axis = (0.0, 0.0, 1.0)
    bank = mt.trace(x, p, q, deposit=dict(kind="bank"), **kw)
    spec = mt.trace(x, p, q, deposit=dict(kind="spectrum", e_edges=e_edges), **kw)
    ang = mt.trace(
        x,
        p,
        q,
        deposit=dict(kind="angular", e_edges=e_edges, d_edges=d_edges, axis=axis),
        **kw,
    )
    return bank, spec, ang, e_edges, d_edges, np.asarray(axis)


def test_spectrum_matches_bank_histogram(traced_with_bank):
    """C++ spectrum tally == numpy histogram of the bank rows."""
    bank, spec, _, e_edges, _, _ = traced_with_bank
    rows = bank["bank"]
    for charge in (+1, -1):
        m = rows["q"] == charge
        h_w = np.histogram(rows["e_mu"][m], bins=e_edges, weights=rows["w"][m])[0]
        h_wp = np.histogram(
            rows["e_mu"][m], bins=e_edges, weights=(rows["w"] * rows["pol"])[m]
        )[0]
        np.testing.assert_allclose(
            spec["spectrum"]["w"][charge], h_w, rtol=1e-10, atol=1e-13
        )
        np.testing.assert_allclose(
            spec["spectrum"]["wp"][charge], h_wp, rtol=1e-10, atol=1e-13
        )


def test_angular_matches_bank_histogram(traced_with_bank):
    """C++ angular tally == numpy 2D histogram of the bank rows,
    including the delta overflow convention."""
    bank, _, ang, e_edges, d_edges, axis = traced_with_bank
    rows = bank["bank"]
    delta = np.arccos(np.clip(rows["phat"] @ axis, -1.0, 1.0))
    over = delta >= d_edges[-1]
    assert abs(ang["angular"]["overflow"] - rows["w"][over].sum()) < 1e-12
    for charge in (+1, -1):
        m = (rows["q"] == charge) & ~over
        for key, wt in (
            ("w", rows["w"]),
            ("wp", rows["w"] * rows["pol"]),
            ("ws2", rows["w"] * rows["sig_mcs"] ** 2),
        ):
            h = np.histogram2d(
                rows["e_mu"][m], delta[m], bins=(e_edges, d_edges), weights=wt[m]
            )[0]
            np.testing.assert_allclose(
                ang["angular"][key][charge], h, rtol=1e-10, atol=1e-15
            )


def test_bank_threshold():
    """w_thresh filters rows but total deposit audit is unaffected."""
    rng = np.random.default_rng(5)
    x, p, q, pol, w = _random_batch(50, rng, R_G)
    mt = MuonTracer(bfield="none", atmosphere="usstd", r_ground=R_G)
    all_rows = mt.trace(x, p, q, pol=pol, w=w, w_min=0.0, deposit=dict(kind="bank"))
    thr = np.median(all_rows["bank"]["w"])
    cut = mt.trace(
        x, p, q, pol=pol, w=w, w_min=0.0, deposit=dict(kind="bank", w_thresh=thr)
    )
    assert len(cut["bank"]["w"]) < len(all_rows["bank"]["w"])
    assert (cut["bank"]["w"] > thr).all()
    assert abs(cut["deposited"] - all_rows["deposited"]) < 1e-12


# ---------------------------------------------------------------------------
# Probes: field / atmosphere / dedx
# ---------------------------------------------------------------------------


def test_dipole_probe_matches_reference():
    mt = MuonTracer(bfield="dipole")
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.5, 1.5, (50, 3)) * 6.4e8
    x = x[np.linalg.norm(x, axis=1) > 6.4e8 * 0.5]
    np.testing.assert_allclose(
        mt.bfield(x), ref.AxialDipole()(x), rtol=1e-14, atol=1e-20
    )


def test_usstd_probe_matches_reference():
    mt = MuonTracer(atmosphere="usstd")
    h = np.linspace(-1e5, 1.2e7, 3001)
    np.testing.assert_allclose(
        mt.rho(h), ref.USStdAtmosphere().rho(h), rtol=1e-14, atol=0
    )


def test_atmosphere_table_interp():
    h_tab = np.linspace(0.0, 1.128e7, 512)
    rho_tab = 1.2e-3 * np.exp(-h_tab / 8.0e5)
    mt = MuonTracer(atmosphere=("table", h_tab, rho_tab))
    h = np.linspace(-1e5, 1.2e7, 2001)
    expect = np.interp(h, h_tab, rho_tab)
    expect[(h >= 1.128e7) | (h < 0.0)] = 0.0
    np.testing.assert_allclose(mt.rho(h), expect, rtol=1e-14, atol=0)


def test_dedx_table_interp():
    e_kin = np.geomspace(0.01, 100.0, 61)
    a = 2.0e-3 * (1.0 + 0.05 * np.log(e_kin / 0.1) ** 2)
    mt = MuonTracer(dedx=(e_kin, a))
    e_tot = np.geomspace(0.05, 300.0, 500) + M_MU
    ekin_c = np.maximum(e_tot - M_MU, e_kin[0])
    expect = np.interp(np.log(ekin_c), np.log(e_kin), a)
    np.testing.assert_allclose(mt.dedx(e_tot), expect, rtol=1e-13, atol=0)


def test_shell_field_close_to_dipole_scale():
    """IGRF shell magnitude is geomagnetic-scale (20-70 uT at ground)."""
    mt = MuonTracer(bfield="shell", r_ground=6.391e8)
    rng = np.random.default_rng(1)
    th = np.arccos(rng.uniform(-0.95, 0.95, 20))
    ph = rng.uniform(0, 2 * np.pi, 20)
    r = 6.391e8 + rng.uniform(0, 1e6, 20)
    x = np.stack(
        [r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)], 1
    )
    bmag = np.linalg.norm(mt.bfield(x), axis=1)
    assert (bmag > 2.0e-5).all() and (bmag < 7.5e-5).all()


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------


def test_thread_invariance():
    """Per-ray results identical for 1 vs 4 threads; tallies at
    rounding level."""
    rng = np.random.default_rng(9)
    x, p, q, pol, w = _random_batch(200, rng, R_G)
    mt = MuonTracer(bfield="dipole", atmosphere="usstd", r_ground=R_G)
    e_edges = np.geomspace(0.15, 41.0, 65)
    kw = dict(pol=pol, w=w, w_min=0.0, deposit=dict(kind="spectrum", e_edges=e_edges))
    r1 = mt.trace(x, p, q, n_threads=1, **kw)
    r4 = mt.trace(x, p, q, n_threads=4, **kw)
    np.testing.assert_array_equal(r1["x"], r4["x"])
    np.testing.assert_array_equal(r1["w"], r4["w"])
    np.testing.assert_array_equal(r1["fate"], r4["fate"])
    for c in (+1, -1):
        np.testing.assert_allclose(
            r1["spectrum"]["w"][c], r4["spectrum"]["w"][c], rtol=1e-13, atol=1e-300
        )


# ---------------------------------------------------------------------------
# Michel kernels
# ---------------------------------------------------------------------------


def test_michel_rest_norms_and_moments():
    """int F0 dx dOmega = 1; <x> = 7/10 (numu), 3/5 (nue)... in the
    Lipari normalisation: <E*>/E_max with E_max = m/2: 0.7 and 0.6."""
    x = (np.arange(4000) + 0.5) / 4000
    for kind, xmean in (("numu", 0.7), ("nue", 0.6)):
        f0 = 4.0 * np.pi * rest_kernel(x, 0.0, 0.0, kind)
        assert abs(f0.sum() / 4000 - 1.0) < 1e-6
        assert abs((f0 * x).sum() / 4000 - xmean) < 1e-6


def test_michel_lab_norm_and_mean():
    """Boosted spectrum: number conservation and <E> = 0.35 / 0.30 E_mu
    (unpolarized, high gamma)."""
    e_mu = 20.0
    e = np.geomspace(1e-4, e_mu, 20000)
    for kind, frac in (("numu", 0.35), ("nue", 0.30)):
        s = lab_spectrum(e, e_mu, 0.0, kind)
        norm = np.trapezoid(s, e)
        emean = np.trapezoid(s * e, e) / norm
        assert abs(norm - 1.0) < 1e-3
        assert abs(emean / e_mu - frac) < 2e-3


def test_michel_polarization_norm_invariance():
    """The polarized term integrates to zero over the solid angle."""
    e_mu = 5.0
    e = np.geomspace(1e-4, e_mu, 8000)
    for kind in ("numu", "nue"):
        s0 = lab_spectrum(e, e_mu, 0.0, kind)
        s1 = lab_spectrum(e, e_mu, 1.0, kind)
        assert abs(np.trapezoid(s1 - s0, e)) < 1e-3
