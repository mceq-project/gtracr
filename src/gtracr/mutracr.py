"""
mutracr — batch transport of atmospheric muons between production and decay.

``MuonTracer`` marches an ensemble of weighted muon rays through the
atmosphere and geomagnetic field: RK4 on (x, p) with per-ray adaptive
time steps, continuous dE/dX energy loss, per-step *expected*-decay
weight deposition (never sampled — a weighted source ensemble is an
exact quadrature, no Monte Carlo variance), and accumulated Fermi-Eyges
multiple-scattering variance.  The C++ engine
(``gtracr._libgtracr.MuonTracer``) is a port of the nu3d stage-C numpy
tracer, which this package vendors as ``gtracr._mu_reference`` for
regression testing.

Units (muon-physics convention, NOT the SI of ``Trajectory``):
positions cm, time s, energy/momentum GeV, B Tesla, density g/cm^3,
dE/dX GeV/(g/cm^2).  Positions are Earth-centred Cartesian with z along
the geographic north pole and x through Greenwich.

Example
-------
>>> import numpy as np
>>> from gtracr.mutracr import MuonTracer
>>> mt = MuonTracer(bfield="dipole", atmosphere="usstd")
>>> x = np.array([[0.0, 0.0, mt.r_ground + 1.75e6]])   # 17.5 km up
>>> p = np.array([[0.0, 0.0, -1.0]])                    # 1 GeV/c down
>>> res = mt.trace(x, p, q=[-1])
>>> res["fate"][0]                                      # 1 == ground
1

The decay deposits compose with the polarized Michel kernels below
(``lab_spectrum`` for angle-integrated fluxes, ``lab_kernel`` toward a
detector direction) to yield the muon-chain neutrino source.
"""

import numpy as np

from gtracr._libgtracr import MuonTracer as _CppMuonTracer
from gtracr.utils import _DATA_DIR

M_MU = 0.1056583755  # muon mass (GeV)
TAU_MU = 2.1969811e-6  # muon lifetime (s)
EARTH_RADIUS_CM = 6.3712e8  # gtracr EARTH_RADIUS in cm

#: fate codes in the ``trace()`` result
FATES = {0: "maxsteps", 1: "ground", 2: "stopped", 3: "faded", 4: "escaped"}


class MuonTracer:
    """
    Batch muon transport through atmosphere and geomagnetic field.

    Parameters
    ----------
    bfield : str or tuple
        Magnetic field model:

        - ``"none"`` — B = 0.
        - ``"dipole"`` — gtracr's ideal centred dipole (IGRF-2020
          g10, reference radius 6371.2 km; the reference radius is
          field normalisation only, independent of `r_ground`).
        - ``"shell"`` — IGRF-13 tabulated on an (h, colat, lon) grid
          over the atmospheric shell [0, `shell_h_max`] above
          `r_ground`, trilinearly interpolated (recommended for
          physics runs; the table is built once at construction).
        - ``("uniform", (bx, by, bz))`` — constant Cartesian field in
          Tesla (analytic tests).
    atmosphere : str or tuple
        Density model rho(h), h the altitude above `r_ground`:

        - ``"none"`` — vacuum.
        - ``"usstd"`` — Linsley/CORSIKA US-Std analytic layers.
        - ``("uniform", rho0)`` — constant density below `h_top`.
        - ``("table", h_cm, rho)`` — linear interpolation of a sampled
          profile (e.g. MSIS); clamped inside, zero outside
          [0, `h_top`).
    dedx : float or tuple
        Muon stopping power, positive, in GeV/(g/cm^2): a constant, or
        ``(e_kin_grid, a)`` interpolated linearly in ln(E_kin) and
        clamped at the ends (e.g. from an MCEq continuous-loss table).
    r_ground : float
        Ground-sphere radius in cm (default gtracr's Earth radius;
        MCEq-matched work wants 6.391e8).
    h_escape : float
        Rays above ``r_ground + h_escape`` moving outward terminate as
        'escaped' (default 1.2e7 cm).
    h_top : float
        Atmosphere top in cm — rho = 0 above it (default 1.128e7).
    igrf_year : float
        Decimal year for the ``"shell"`` field (default 2020.0).
    shell_grid : tuple of int
        (n_h, n_colat, n_lon) nodes of the ``"shell"`` table
        (default (13, 91, 181), the nu3d gate-C0b-validated grid).
    shell_h_max : float
        Radial extent of the ``"shell"`` table in cm (default 1.3e7,
        just above `h_escape`); queries outside clamp to the boundary.

    Notes
    -----
    Rays are independent; ``trace`` parallelises over ``n_threads``
    std::thread workers (0 = all cores).  Per-ray trajectories are
    bit-identical regardless of thread count; tally sums can differ at
    float rounding level.
    """

    def __init__(
        self,
        bfield="none",
        atmosphere="none",
        dedx=2.0e-3,
        r_ground=EARTH_RADIUS_CM,
        h_escape=1.20e7,
        h_top=1.128e7,
        igrf_year=2020.0,
        shell_grid=(13, 91, 181),
        shell_h_max=1.3e7,
    ):
        kw = dict(
            r_ground=float(r_ground), h_escape=float(h_escape), atmo_h_top=float(h_top)
        )

        if isinstance(bfield, str):
            if bfield not in ("none", "dipole", "shell"):
                raise ValueError(f"unknown bfield {bfield!r}")
            kw["bfield_type"] = bfield
            if bfield == "shell":
                kw["igrf_params"] = (str(_DATA_DIR.resolve()), float(igrf_year))
                kw["shell_n_h"], kw["shell_n_colat"], kw["shell_n_lon"] = map(
                    int, shell_grid
                )
                kw["shell_h_max"] = float(shell_h_max)
        elif bfield[0] == "uniform":
            kw["bfield_type"] = "uniform"
            kw["b_uniform"] = tuple(float(b) for b in bfield[1])
        else:
            raise ValueError(f"unknown bfield {bfield!r}")

        if isinstance(atmosphere, str):
            if atmosphere not in ("none", "usstd"):
                raise ValueError(f"unknown atmosphere {atmosphere!r}")
            kw["atmo_type"] = atmosphere
        elif atmosphere[0] == "uniform":
            kw["atmo_type"] = "uniform"
            kw["atmo_rho0"] = float(atmosphere[1])
        elif atmosphere[0] == "table":
            kw["atmo_type"] = "table"
            kw["atmo_h"] = np.ascontiguousarray(atmosphere[1], dtype=float)
            kw["atmo_rho"] = np.ascontiguousarray(atmosphere[2], dtype=float)
        else:
            raise ValueError(f"unknown atmosphere {atmosphere!r}")

        if np.isscalar(dedx):
            kw["dedx_const"] = float(dedx)
        else:
            kw["dedx_e_kin"] = np.ascontiguousarray(dedx[0], dtype=float)
            kw["dedx_a"] = np.ascontiguousarray(dedx[1], dtype=float)

        self._cpp = _CppMuonTracer(**kw)
        self.r_ground = float(r_ground)
        self.h_escape = float(h_escape)

    # -- probes (validation / plotting) ---------------------------------

    def bfield(self, x):
        """B [T] at Cartesian positions `x` [cm], shape (N, 3)."""
        return self._cpp.bfield(np.atleast_2d(np.asarray(x, float)))

    def rho(self, h):
        """Density [g/cm^3] at altitudes `h` [cm] above the ground sphere."""
        return self._cpp.rho(np.atleast_1d(np.asarray(h, float)))

    def dedx(self, e_tot):
        """dE/dX [GeV/(g/cm^2)] at total muon energies `e_tot` [GeV]."""
        return self._cpp.dedx(np.atleast_1d(np.asarray(e_tot, float)))

    # -- transport -------------------------------------------------------

    def trace(
        self,
        x,
        p,
        q,
        pol=None,
        w=None,
        *,
        e_min=0.11,
        w_min=1e-6,
        dt_max=None,
        max_steps=100000,
        ctl_bend=0.05,
        ctl_loss=0.05,
        ctl_decay=0.05,
        ctl_path=2e5,
        decay_beta_one=False,
        n_threads=0,
        deposit=None,
    ):
        """
        March the ensemble until every ray is dead.

        Parameters
        ----------
        x, p : array_like, shape (N, 3)
            Positions [cm] and momenta [GeV/c].
        q : array_like, shape (N,)
            Charges in units of e (+1 / -1).
        pol : array_like, optional
            Spin projection on p_hat (defaults to 0).
        w : array_like, optional
            Source weights (defaults to 1); all deposits/fates are in
            these absolute units.
        e_min, w_min : float
            'stopped' / 'faded' thresholds (total energy GeV, weight).
            `e_min` is clamped above the muon mass internally.
        dt_max : float, optional
            Hard per-step time cap in s (None = uncapped).
        max_steps : int
            Per-ray step cap; leftovers terminate as 'maxsteps'.
        ctl_bend, ctl_loss, ctl_decay, ctl_path
            Adaptive-dt controls: max bend angle [rad], fractional
            energy loss, fractional decay probability per step, and
            path length [cm].
        decay_beta_one : bool
            Use decay probability per unit length 1/(gamma c tau)
            instead of the exact 1/(beta gamma c tau) — reproduces
            pre-2026-07 MCEq references only.
        n_threads : int
            C++ worker threads (0 = all cores).
        deposit : dict, optional
            Expected-decay consumer.  ``None`` disables recording
            (weights still evolve).  Otherwise ``deposit["kind"]``:

            - ``"bank"`` — raw rows; optional ``w_thresh``.  Result key
              ``"bank"``: x, phat, e_mu, q, pol, sig_mcs, w arrays.
            - ``"spectrum"`` — E_mu histograms; requires ``e_edges``
              (total energy, ascending).  Result key ``"spectrum"``:
              ``{"w": {+1: .., -1: ..}, "wp": {+1: .., -1: ..}}``.
            - ``"angular"`` — (E_mu, delta-to-axis) histograms;
              requires ``e_edges``, ``d_edges`` [rad] and ``axis``.
              Result key ``"angular"``: ``{"w"|"wp"|"ws2":
              {+1: (nE, nD), -1: ...}, "overflow": float}`` where
              deposits with delta >= d_edges[-1] land in overflow.

        Returns
        -------
        dict
            Final state ``x, p, w, th2, fate`` (fate codes in
            :data:`FATES`), diagnostics ``steps`` (max per ray),
            ``deposited``, ``fate_w`` (weight per fate),
            ``weight_audit`` (deposited + fates; equals sum(w) up to
            the ground-crossing deposit convention), ``n``, and the
            deposit product described above.
        """
        x = np.atleast_2d(np.asarray(x, float))
        p = np.atleast_2d(np.asarray(p, float))
        n = len(x)
        q = np.asarray(q, np.int64).reshape(n)
        pol = np.zeros(n) if pol is None else np.asarray(pol, float).reshape(n)
        w = np.ones(n) if w is None else np.asarray(w, float).reshape(n)

        dep_kw = dict(deposit="none")
        if deposit is not None:
            kind = deposit["kind"]
            dep_kw["deposit"] = kind
            if kind == "bank":
                dep_kw["w_thresh"] = float(deposit.get("w_thresh", 0.0))
            elif kind == "spectrum":
                dep_kw["e_edges"] = np.ascontiguousarray(deposit["e_edges"], float)
            elif kind == "angular":
                dep_kw["e_edges"] = np.ascontiguousarray(deposit["e_edges"], float)
                dep_kw["d_edges"] = np.ascontiguousarray(deposit["d_edges"], float)
                dep_kw["axis"] = tuple(float(a) for a in deposit["axis"])
            else:
                raise ValueError(f"unknown deposit kind {kind!r}")

        res = self._cpp.trace(
            x,
            p,
            q,
            pol,
            w,
            e_min=float(e_min),
            w_min=float(w_min),
            dt_max=0.0 if dt_max is None else float(dt_max),
            max_steps=int(max_steps),
            ctl_bend=float(ctl_bend),
            ctl_loss=float(ctl_loss),
            ctl_decay=float(ctl_decay),
            ctl_path=float(ctl_path),
            decay_beta_one=bool(decay_beta_one),
            n_threads=int(n_threads),
            **dep_kw,
        )
        if "spectrum" in res:
            s = res["spectrum"]
            res["spectrum"] = {
                "w": {+1: s["w_pos"], -1: s["w_neg"]},
                "wp": {+1: s["wp_pos"], -1: s["wp_neg"]},
            }
        if "angular" in res:
            a = res["angular"]
            res["angular"] = {
                "w": {+1: a["w_pos"], -1: a["w_neg"]},
                "wp": {+1: a["wp_pos"], -1: a["wp_neg"]},
                "ws2": {+1: a["ws2_pos"], -1: a["ws2_neg"]},
                "overflow": a["overflow"],
            }
        return res


# ---------------------------------------------------------------------------
# Polarized mu -> nu Michel kernels (rest frame + boosted lab frame).
#
# Rest frame, x = 2 E*_nu / m_mu in (0, 1], theta_s measured from the
# muon SPIN direction, P in [-1, 1] the spin projection:
#
#     dN/(dx dOmega) = (1/4pi) [ F0(x) + P cos(theta_s) F1(x) ]
#
# for a mu^- (Lipari 1993 / Gaisser structure):
#     nu_mu   : F0 = 2 x^2 (3 - 2x),   F1 = 2 x^2 (1 - 2x)
#     nubar_e : F0 = 12 x^2 (1 - x),   F1 = 12 x^2 (1 - x)
# For mu^+ (-> nubar_mu, nu_e) CP gives the same formulas with P -> -P.
# The sign choice is pinned against the MCEq decay-DB helicity
# polynomials to <2e-3 (nu3d gate C0/T6); the flipped sign deviates by
# 0.3-2.1.
#
# Lab frame: the polarization is carried as a scalar along p_hat (spin
# follows momentum for g ~= 2), so boost and spin axes coincide.  With
# massless nu, E dN/d^3p is the invariant density, so
# dN_lab/(dE dOmega) = (E_lab/E*) dN*/(dE* dOmega*) — FIRST power (the
# naive square over-counts by exactly gamma; nu3d gate C0/T5).
# ---------------------------------------------------------------------------


def rest_kernel(x, cos_th_s, pol, kind):
    """dN/(dx dOmega) in the muon rest frame; vectorised, broadcasts.

    kind: 'numu' (mu-type nu) or 'nue' (e-type nu); the polarization
    convention is P for mu^-, pass -P for mu^+.
    """
    x = np.asarray(x, float)
    inside = (x > 0.0) & (x <= 1.0)
    xs = np.where(inside, x, 0.0)
    if kind == "numu":
        f0 = 2.0 * xs**2 * (3.0 - 2.0 * xs)
        f1 = 2.0 * xs**2 * (1.0 - 2.0 * xs)
    elif kind == "nue":
        f0 = 12.0 * xs**2 * (1.0 - xs)
        f1 = f0
    else:
        raise ValueError(kind)
    return np.where(inside, (f0 + pol * cos_th_s * f1) / (4.0 * np.pi), 0.0)


def lab_spectrum(e_nu, e_mu, pol, kind, charge=-1, n_x=512):
    """Angle-integrated lab spectrum dN/dE_nu [GeV^-1] per decay.

    Exact solid-angle integral evaluated in the rest frame: for fixed
    x the lab energy is uniform on [gamma E*(1-beta), gamma E*(1+beta)]
    (isotropic 2-body boost).  Midpoint rule in x; normalisation
    int dN/dE dE = 1 per decay.  Vectorised over `e_nu`; `e_mu` (TOTAL
    energy) and `pol` scalars.
    """
    e_nu = np.atleast_1d(np.asarray(e_nu, float))
    p_eff = pol if charge == -1 else -pol
    p_mu = np.sqrt(max(e_mu**2 - M_MU**2, 0.0))
    gam = e_mu / M_MU
    bet = max(p_mu / e_mu, 1e-12)
    x = (np.arange(n_x) + 0.5) / n_x
    e_star = 0.5 * M_MU * x
    if kind == "numu":
        f0 = 2.0 * x**2 * (3.0 - 2.0 * x)
        f1 = 2.0 * x**2 * (1.0 - 2.0 * x)
    elif kind == "nue":
        f0 = 12.0 * x**2 * (1.0 - x)
        f1 = f0.copy()
    else:
        raise ValueError(kind)
    cs = (e_nu[:, None] / (gam * e_star[None, :]) - 1.0) / bet
    inside = np.abs(cs) <= 1.0
    integ = np.where(
        inside,
        (f0[None, :] + p_eff * cs * f1[None, :]) / (2.0 * gam * bet * e_star[None, :]),
        0.0,
    )
    return integ.sum(axis=1) / n_x


def lab_kernel(e_nu, cos_th_lab, e_mu, pol, kind, charge=-1):
    """dN/(dE_nu dOmega_lab) [GeV^-1 sr^-1] toward a lab direction.

    cos_th_lab: angle between the nu lab direction and p_hat_mu.
    pol: spin projection along p_hat_mu (scalar); the mu^+ CP mirror is
    applied via `charge` (+1 or -1).  All arguments broadcast.
    """
    e_nu = np.asarray(e_nu, float)
    p_eff = np.where(charge == -1, pol, -pol)
    p_mu = np.sqrt(np.maximum(e_mu**2 - M_MU**2, 0.0))
    gam = e_mu / M_MU
    bet = p_mu / e_mu
    doppler = gam * (1.0 - bet * cos_th_lab)  # E* = E_lab * doppler
    e_star = e_nu * doppler
    x = 2.0 * e_star / M_MU
    cos_star = (cos_th_lab - bet) / (1.0 - bet * cos_th_lab)
    rest = rest_kernel(x, cos_star, p_eff, kind) * (2.0 / M_MU)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(e_star > 0.0, e_nu / np.maximum(e_star, 1e-300), 0.0)
    return rest * jac
