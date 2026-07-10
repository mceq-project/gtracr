"""
Pure-numpy reference implementation of the muon batch tracer.

Vendored, near-verbatim, from the nu3d stage-C tracer
(mceq-em-integration nu3d/mutracer/{tracer,fields,atmosphere}.py) — the
correctness reference that ``gtracr._libgtracr.MuonTracer`` was ported
from.  Used by ``tests/test_mutracr.py`` to pin the C++ engine
step-for-step; it is NOT the production API (use ``gtracr.mutracr``).

State per ray (SoA): x (N,3) cm | p (N,3) GeV | q (N) | pol (N) |
w (N) decay-survival weight | th2 (N) rad^2 MCS variance | alive (N).

Dynamics (RK4 on (x, p)):
  dx/dt = beta c p_hat
  dp/dt = KP q beta (p_hat x B[T]) - dedx rho(h) c p_hat.

Per step the expected decay weight dW = w (1 - exp(-lambda dt)),
lambda = m_mu / (E tau_mu), is deposited at the midpoint state and w is
multiplied by exp(-lambda dt).  MCS: d<th^2>/dX =
(13.6 MeV / (beta p))^2 / X0 (X0 air 36.62 g/cm^2, no log term).
Termination fates: 'ground' (pulled back onto the sphere), 'stopped',
'faded', 'escaped', 'maxsteps'; deposited + residuals == sum(w_in).
"""

import numpy as np

M_MU = 0.1056583755  # GeV
TAU_MU = 2.1969811e-6  # s
C_CMS = 2.99792458e10  # cm/s
KP = 8.98755179e7  # GeV/(s T) — 1e-9 * c[m/s]^2
X0_AIR = 36.62  # g/cm^2
ES_MCS = 0.0136  # GeV — MCS scale (Highland, no log term)
H_ESC = 1.20e7  # cm — escape shell above ground
H_TOP = 1.128e7  # cm — atmosphere top


class UniformField:
    """Constant field, for analytic gates (helix, CSDA)."""

    def __init__(self, b_vec_T):
        self.b = np.asarray(b_vec_T, float)

    def __call__(self, x_cm):
        return np.broadcast_to(self.b, np.shape(x_cm)).copy()


class AxialDipole:
    """gtracr's ideal centred dipole, vectorised, Cartesian output."""

    def __init__(self, g10_T=29404.8e-9, r_ref_cm=6.3712e8):
        self.g10 = float(g10_T)
        self.a = float(r_ref_cm)

    def __call__(self, x_cm):
        x = np.atleast_2d(np.asarray(x_cm, float))
        r = np.linalg.norm(x, axis=1)
        rhat = x / r[:, None]
        cz = rhat[:, 2]
        fac = -self.g10 * (self.a / r) ** 3
        b = fac[:, None] * (3.0 * cz[:, None] * rhat)
        b[:, 2] -= fac
        return b


# Linsley/CORSIKA US-Std layers: boundaries [cm] and parameters.
_HB = np.array([0.0, 4e5, 1e6, 4e6, 1e7])
_B = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])  # g/cm^2
_C = np.array([994186.38, 878153.55, 636143.04, 772170.16, 1e9])  # cm


class USStdAtmosphere:
    def rho(self, h_cm):
        """Density [g/cm^3]; zero above the atmosphere top."""
        h = np.asarray(h_cm, float)
        i = np.clip(np.searchsorted(_HB, h, side="right") - 1, 0, 4)
        r = np.where(i < 4, _B[i] / _C[i] * np.exp(-h / _C[i]), _B[4] / _C[4])
        return np.where((h >= H_TOP) | (h < 0.0), 0.0, r)


class UniformAtmosphere:
    """Constant density everywhere below H_TOP — analytic-gate toy."""

    def __init__(self, rho0=1.2e-3):
        self.rho0 = float(rho0)

    def rho(self, h_cm):
        h = np.asarray(h_cm, float)
        return np.where(h < H_TOP, self.rho0, 0.0)


class MuonBatch:
    """Source-node ensemble; all arrays length N (float64, int8)."""

    def __init__(self, x, p, q, pol=None, w=None):
        self.x = np.array(x, float)
        self.p = np.array(p, float)
        n = len(self.x)
        self.q = np.array(q, np.int8)
        self.pol = np.zeros(n) if pol is None else np.array(pol, float)
        self.w = np.ones(n) if w is None else np.array(w, float)
        self.th2 = np.zeros(n)
        self.alive = np.ones(n, bool)


class DecayBank:
    """Expected-decay deposits (x, phat, e_mu, q, pol, sig_mcs, w)."""

    def __init__(self, w_thresh=0.0):
        self.w_thresh = float(w_thresh)
        self._rows = []

    def add(self, x, phat, e_mu, q, pol, sig_mcs, w):
        m = w > self.w_thresh
        if m.any():
            self._rows.append((x[m], phat[m], e_mu[m], q[m], pol[m], sig_mcs[m], w[m]))
        return w.sum()

    def arrays(self):
        if not self._rows:
            return None
        cols = [np.concatenate(c) for c in zip(*self._rows)]
        return dict(zip(("x", "phat", "e_mu", "q", "pol", "sig_mcs", "w"), cols))

    @property
    def total_weight(self):
        return sum(r[-1].sum() for r in self._rows)


def _as_dedx(dedx):
    """Accept a constant [GeV/(g/cm^2)] or a callable dedx(E)."""
    if callable(dedx):
        return dedx
    a = float(dedx)
    return lambda e: np.full(np.shape(e), a)


def _deriv(x, p, q, field, atmo, dedx_fn, r_ground):
    pm = np.maximum(np.linalg.norm(p, axis=1), 1e-12)
    e = np.sqrt(pm**2 + M_MU**2)
    phat = p / pm[:, None]
    beta = pm / e
    dx = C_CMS * beta[:, None] * phat
    dp = KP * (q * beta)[:, None] * np.cross(phat, field(x))
    if atmo is not None:
        h = np.linalg.norm(x, axis=1) - r_ground
        dp = dp - (dedx_fn(e) * C_CMS * atmo.rho(h))[:, None] * phat
    return dx, dp


def trace(
    batch,
    field,
    atmo,
    deposit=None,
    *,
    r_ground=6.3712e8,
    dedx=2.0e-3,
    e_min=0.11,
    w_min=1e-6,
    dt_max=np.inf,
    max_steps=100_000,
    ctl_bend=0.05,
    ctl_loss=0.05,
    ctl_decay=0.05,
    ctl_path=2e5,
    decay_beta_one=False,
):
    """March the ensemble until every ray is dead; returns diagnostics.

    Verbatim nu3d reference semantics; see the module docstring.  The
    only difference from nu3d is that the ground radius is an argument
    (nu3d imports its geometry constant).
    """
    e_min = max(float(e_min), M_MU * (1.0 + 1e-3))
    dedx_fn = _as_dedx(dedx)
    b = batch
    fate = {k: 0.0 for k in ("ground", "stopped", "faded", "escaped", "maxsteps")}
    deposited = 0.0
    steps = 0
    while b.alive.any() and steps < max_steps:
        steps += 1
        a = b.alive
        x0, p0 = b.x[a], b.p[a]
        q0, w0 = b.q[a], b.w[a]
        pm = np.maximum(np.linalg.norm(p0, axis=1), 1e-12)
        e0 = np.sqrt(pm**2 + M_MU**2)
        beta = pm / e0

        # per-ray dt from the fastest local scale
        bmag = np.maximum(np.linalg.norm(field(x0), axis=1), 1e-30)
        dt = np.minimum(ctl_bend * pm / (KP * beta * bmag), dt_max)
        dt = np.minimum(dt, ctl_decay * e0 * TAU_MU / M_MU)
        dt = np.minimum(dt, ctl_path / (beta * C_CMS))
        if atmo is not None:
            h0 = np.linalg.norm(x0, axis=1) - r_ground
            rho = atmo.rho(h0)
            dt = np.minimum(
                dt, ctl_loss * pm / (dedx_fn(e0) * C_CMS * np.maximum(rho, 1e-30))
            )

        # RK4
        args = (q0, field, atmo, dedx_fn, r_ground)
        k1x, k1p = _deriv(x0, p0, *args)
        k2x, k2p = _deriv(
            x0 + 0.5 * dt[:, None] * k1x, p0 + 0.5 * dt[:, None] * k1p, *args
        )
        k3x, k3p = _deriv(
            x0 + 0.5 * dt[:, None] * k2x, p0 + 0.5 * dt[:, None] * k2p, *args
        )
        k4x, k4p = _deriv(x0 + dt[:, None] * k3x, p0 + dt[:, None] * k3p, *args)
        x1 = x0 + dt[:, None] / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        p1 = p0 + dt[:, None] / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)

        pm1 = np.maximum(np.linalg.norm(p1, axis=1), 1e-12)
        e1 = np.sqrt(pm1**2 + M_MU**2)
        e_mid = 0.5 * (e0 + e1)

        # decay weight (exact for E ~ const over the step)
        lam = M_MU / (e_mid * TAU_MU)
        if decay_beta_one:
            lam = lam * (0.5 * (pm + pm1) / e_mid)
        surv = np.exp(-lam * dt)
        dw = w0 * (1.0 - surv)
        if deposit is not None:
            x_mid = 0.5 * (x0 + x1)
            pv = p0 + p1
            phat_mid = pv / np.maximum(np.linalg.norm(pv, axis=1), 1e-12)[:, None]
            deposit.add(x_mid, phat_mid, e_mid, q0, b.pol[a], np.sqrt(b.th2[a]), dw)
        deposited += dw.sum()

        # MCS variance along the step
        th2 = b.th2[a]
        if atmo is not None:
            h_mid = np.linalg.norm(0.5 * (x0 + x1), axis=1) - r_ground
            dxg = atmo.rho(h_mid) * beta * C_CMS * dt  # g/cm^2
            th2 = th2 + (ES_MCS / (beta * pm)) ** 2 * dxg / X0_AIR

        w1 = w0 * surv

        # ground crossing: pull the final state back onto the sphere
        r0n = np.linalg.norm(x0, axis=1)
        r1 = np.linalg.norm(x1, axis=1)
        dead_g = r1 < r_ground
        if dead_g.any():
            g = dead_g
            f = ((r0n[g] - r_ground) / np.maximum(r0n[g] - r1[g], 1e-30))[:, None]
            x1[g] = x0[g] + f * (x1[g] - x0[g])
            p1[g] = p0[g] + f * (p1[g] - p0[g])
            w1[g] = w0[g] * surv[g] ** f[:, 0]
            r1[g] = np.linalg.norm(x1[g], axis=1)

        b.x[a], b.p[a], b.w[a], b.th2[a] = x1, p1, w1, th2

        # fates
        out = np.einsum("ij,ij->i", x1, p1) > 0.0
        dead_s = ~dead_g & (e1 < e_min)
        dead_f = ~dead_g & ~dead_s & (w1 < w_min)
        dead_e = ~dead_g & ~dead_s & ~dead_f & (r1 > r_ground + H_ESC) & out
        for key, m in (
            ("ground", dead_g),
            ("stopped", dead_s),
            ("faded", dead_f),
            ("escaped", dead_e),
        ):
            fate[key] += w1[m].sum()
        sub = a.nonzero()[0]
        b.alive[sub[dead_g | dead_s | dead_f | dead_e]] = False

    fate["maxsteps"] = b.w[b.alive].sum()
    audit = deposited + sum(fate.values())
    return dict(
        steps=steps, deposited=deposited, fate=fate, weight_audit=audit, n=len(b.w)
    )
