"""
Numerical comparison of RK4 (frozen-field), Boris pusher, and adaptive RK45.

Runs three classes of tests:

  1. Accuracy — single reference trajectory at various step sizes / tolerances.
     Reports final-position divergence, momentum-magnitude drift, and
     number of B-field evaluations relative to RK45 (reference).

  2. Energy conservation — plots |p(t)| − |p₀| over a long trajectory.
     Boris is exact for pure-magnetic fields; RK4/RK45 have numerical drift.

  3. Performance — wall-clock time for N trajectory evaluations with each solver.

Usage
-----
  python examples/eval_solver_comparison.py [--no-plots] [--n-perf N]

Outputs
-------
  Console: structured comparison table.
  Plots (unless --no-plots): accuracy curves, energy drift, timing bar chart.
"""

import argparse
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend; use TkAgg locally if desired
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PLOT_DIR = os.path.join(PARENT_DIR, "..", "gtracr_plots")

sys.path.insert(0, PARENT_DIR)
from gtracr.lib._libgtracr import TrajectoryTracer as CppTrajectoryTracer
from gtracr.lib.constants import (
    EARTH_RADIUS,
    ELEMENTARY_CHARGE,
    KG_PER_GEVC2,
)
from gtracr.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Reference particle / location
# ---------------------------------------------------------------------------
PLABEL = "p+"
LOCATION = "Kamioka"
# Accuracy section uses a forbidden trajectory (R below cutoff) so the
# final position is well-defined and comparable across solvers.
# Kamioka median cutoff ~11 GV; R=5 GV is safely below the cutoff.
RIGIDITY_FORBIDDEN = 5.0  # GV — forbidden (particle returns to atmosphere)
RIGIDITY_ALLOWED = 30.0  # GV — well above cutoff (particle escapes)
BFIELD_TYPE = "igrf"
DT_REF = 1e-5  # reference step size for RK4/Boris
MAX_TIME = 1.0  # seconds → up to 100 000 steps


def run_trajectory(
    solver,
    rigidity=RIGIDITY_FORBIDDEN,
    dt=DT_REF,
    max_time=MAX_TIME,
    atol=1e-3,
    rtol=1e-6,
):
    """Return full trajectory data dict + metadata."""
    traj = Trajectory(
        plabel=PLABEL,
        location_name=LOCATION,
        zenith_angle=45.0,
        azimuth_angle=90.0,
        rigidity=rigidity,
        bfield_type=BFIELD_TYPE,
        solver=solver,
        atol=atol,
        rtol=rtol,
    )
    t0 = time.perf_counter()
    data = traj.get_trajectory(dt=dt, max_time=max_time, get_data=True)
    elapsed = time.perf_counter() - t0
    return data, traj, elapsed


# ---------------------------------------------------------------------------
# 1. Accuracy comparison
# ---------------------------------------------------------------------------


def section_accuracy(show_plots):
    print("\n" + "=" * 72)
    print("SECTION 1: Accuracy vs. B-field evaluations")
    print(f"  Using FORBIDDEN trajectory (R={RIGIDITY_FORBIDDEN} GV) so the")
    print("  atmosphere-return endpoint is well-defined across all solvers.")
    print("=" * 72)

    # Use RK45 at tight tolerance as the reference solution.
    # For a forbidden (atmosphere-hit) trajectory the endpoint is well-defined.
    data_ref, traj_ref, _ = run_trajectory(
        "rk45", rigidity=RIGIDITY_FORBIDDEN, dt=DT_REF, atol=1e-8, rtol=1e-10
    )
    r_final_ref = traj_ref.final_sixvector[0]
    th_final_ref = traj_ref.final_sixvector[1]
    traj_ref.final_sixvector[2]

    print("\nReference (RK45, atol=1e-8, rtol=1e-10):")
    print(f"  steps={len(data_ref['t'])}, escaped={traj_ref.particle_escaped}")
    print(f"  final r = {r_final_ref / EARTH_RADIUS:.6f} RE")
    print(f"  final θ = {np.degrees(th_final_ref):.4f}°")

    def _err(tr):
        fsv = tr.final_sixvector
        dr = abs(fsv[0] - r_final_ref) / EARTH_RADIUS
        dth = abs(np.degrees(fsv[1]) - np.degrees(th_final_ref))
        return dr, dth

    # RK4 at various step sizes
    dt_vals = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6]
    print("\nRK4 (frozen-field) convergence  [1 B-eval per step]:")
    print(f"  {'dt':>10s}  {'steps':>7s}  {'Δr/RE':>12s}  {'Δθ (°)':>12s}")
    rk4_bfevals, rk4_errs = [], []
    for dt in dt_vals:
        d, tr, _ = run_trajectory("rk4", rigidity=RIGIDITY_FORBIDDEN, dt=dt)
        dr, dth = _err(tr)
        n = len(d["t"])
        print(f"  {dt:>10.1e}  {n:>7d}  {dr:>12.2e}  {dth:>12.2e}")
        rk4_bfevals.append(n)
        rk4_errs.append(max(dr, dth))

    # Boris at same step sizes
    print("\nBoris pusher convergence  [1 B-eval per step]:")
    print(f"  {'dt':>10s}  {'steps':>7s}  {'Δr/RE':>12s}  {'Δθ (°)':>12s}")
    boris_bfevals, boris_errs = [], []
    for dt in dt_vals:
        d, tr, _ = run_trajectory("boris", rigidity=RIGIDITY_FORBIDDEN, dt=dt)
        dr, dth = _err(tr)
        n = len(d["t"])
        print(f"  {dt:>10.1e}  {n:>7d}  {dr:>12.2e}  {dth:>12.2e}")
        boris_bfevals.append(n)
        boris_errs.append(max(dr, dth))

    # RK45 at various tolerances
    tol_vals = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    print("\nAdaptive RK45 convergence  [~6 B-evals per accepted step (FSAL)]:")
    print(f"  {'atol=rtol':>10s}  {'steps':>7s}  {'Δr/RE':>12s}  {'Δθ (°)':>12s}")
    rk45_bfevals, rk45_errs = [], []
    for tol in tol_vals:
        d, tr, _ = run_trajectory(
            "rk45", rigidity=RIGIDITY_FORBIDDEN, dt=DT_REF, atol=tol, rtol=tol
        )
        dr, dth = _err(tr)
        n = len(d["t"])
        print(f"  {tol:>10.1e}  {n:>7d}  {dr:>12.2e}  {dth:>12.2e}")
        rk45_bfevals.append(6 * n)
        rk45_errs.append(max(dr, dth))

    if show_plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(rk4_bfevals, rk4_errs, "o-", label="RK4 (frozen-field)")
        ax.loglog(boris_bfevals, boris_errs, "s-", label="Boris pusher")
        ax.loglog(rk45_bfevals, rk45_errs, "^-", label="RK45 (adaptive)")
        ax.set_xlabel("B-field evaluations")
        ax.set_ylabel("Max position error (Δr/RE or Δθ in °)")
        ax.set_title("Accuracy vs. work (B-field evaluations)")
        ax.legend()
        ax.grid(True, which="both", ls=":")
        _savefig(fig, "solver_accuracy.png")


# ---------------------------------------------------------------------------
# 2. Energy (momentum magnitude) conservation
# ---------------------------------------------------------------------------


def section_energy_conservation(show_plots):
    print("\n" + "=" * 72)
    print("SECTION 2: Momentum-magnitude conservation  |p(t)| / |p₀| − 1")
    print("=" * 72)

    solvers = [
        ("rk4", "RK4 (frozen-field)", "C0"),
        ("boris", "Boris pusher", "C1"),
        ("rk45", "RK45 (tol=1e-4)", "C2"),
    ]

    results = {}
    for key, label, _ in solvers:
        kw = dict(rigidity=RIGIDITY_ALLOWED, dt=DT_REF, max_time=MAX_TIME)
        if key == "rk45":
            kw.update(atol=1e-4, rtol=1e-4)
        d, tr, _ = run_trajectory(key, **kw)
        t_arr = d["t"]
        pr = d["pr"]
        pth = d["ptheta"]
        pph = d["pphi"]
        pmag = np.sqrt(pr**2 + pth**2 + pph**2)
        p0 = pmag[0]
        drift = (pmag - p0) / p0
        results[key] = (t_arr, drift)
        rms = np.sqrt(np.mean(drift**2))
        peak = np.max(np.abs(drift))
        print(f"  {label:<30s}  RMS drift = {rms:.2e},  peak = {peak:.2e}")

    if show_plots:
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label, color in solvers:
            t_arr, drift = results[key]
            ax.plot(t_arr, drift, label=label, color=color, linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("|p(t)| / |p₀| − 1")
        ax.set_title("Momentum-magnitude drift (energy conservation)")
        ax.legend()
        ax.grid(True, ls=":")
        _savefig(fig, "solver_energy.png")


# ---------------------------------------------------------------------------
# 3. Performance
# ---------------------------------------------------------------------------


def section_performance(n_perf, show_plots):
    print("\n" + "=" * 72)
    print(f"SECTION 3: Wall-clock time for {n_perf} trajectory evaluations")
    print("=" * 72)

    loc_name = LOCATION

    # Precompute initial conditions once (shared across solvers).
    traj0 = Trajectory(
        plabel=PLABEL,
        location_name=loc_name,
        zenith_angle=45.0,
        azimuth_angle=90.0,
        rigidity=RIGIDITY_FORBIDDEN,
        bfield_type=BFIELD_TYPE,
        solver="rk4",
    )
    charge_si = float(traj0.charge * ELEMENTARY_CHARGE)
    mass_si = float(traj0.mass * KG_PER_GEVC2)
    vec0_orig = [float(x) for x in traj0.particle_sixvector]
    max_step = int(np.ceil(MAX_TIME / DT_REF))

    print(
        f"\n  {'Solver':<20s}  {'Total time (s)':>14s}  {'Per traj (ms)':>14s}  "
        f"{'Total steps':>12s}"
    )
    timings = {}
    steps_total = {}
    for sname, schar in [("rk4", "r"), ("boris", "b"), ("rk45", "a")]:
        tracer = CppTrajectoryTracer(
            charge_si,
            mass_si,
            float(traj0.start_alt),
            float(traj0.esc_alt),
            DT_REF,
            int(max_step),
            traj0.bfield_type,
            traj0.igrf_params,
            schar,
            1e-3,
            1e-6,
        )
        total_steps = 0
        t_start = time.perf_counter()
        for _ in range(n_perf):
            vec0 = vec0_orig.copy()
            tracer.reset()
            tracer.evaluate(0.0, vec0)
            total_steps += tracer.nsteps
        elapsed = time.perf_counter() - t_start
        timings[sname] = elapsed
        steps_total[sname] = total_steps
        per_ms = 1000.0 * elapsed / n_perf
        print(f"  {sname:<20s}  {elapsed:>14.3f}  {per_ms:>14.3f}  {total_steps:>12d}")

    if show_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(timings.keys())
        vals = [timings[k] for k in names]
        colors = ["C0", "C1", "C2"]
        ax.bar(names, vals, color=colors)
        ax.set_ylabel(f"Total time for {n_perf} evaluations (s)")
        ax.set_title("Integration method performance")
        ax.grid(True, axis="y", ls=":")
        _savefig(fig, "solver_performance.png")


# ---------------------------------------------------------------------------
# 4. Trajectory overlay (visual check)
# ---------------------------------------------------------------------------


def section_trajectory_overlay(show_plots):
    if not show_plots:
        return
    print("\n" + "=" * 72)
    print("SECTION 4: Trajectory overlay (visual check)")
    print("=" * 72)

    fig = plt.figure(figsize=(12, 5))
    titles = ["X–Z plane", "X–Y plane"]
    axes_pairs = [("x", "z"), ("x", "y")]
    ax_list = [fig.add_subplot(1, 2, i + 1) for i in range(2)]

    solvers = [
        ("rk4", "RK4 (frozen)", "C0", 1.5),
        ("boris", "Boris", "C1", 1.2),
        ("rk45", "RK45 (tol=1e-4)", "C2", 0.8),
    ]

    for key, label, color, lw in solvers:
        kw = dict(rigidity=RIGIDITY_ALLOWED, dt=DT_REF, max_time=MAX_TIME)
        if key == "rk45":
            kw.update(atol=1e-4, rtol=1e-4)
        d, _, _ = run_trajectory(key, **kw)
        for ax, (cx, cy) in zip(ax_list, axes_pairs):
            ax.plot(d[cx], d[cy], label=label, color=color, lw=lw, alpha=0.85)

    # Draw Earth circle on X–Z and X–Y planes
    theta_circ = np.linspace(0, 2 * np.pi, 300)
    for ax in ax_list:
        ax.plot(np.cos(theta_circ), np.sin(theta_circ), "k--", lw=0.8, label="Earth")
        ax.set_aspect("equal")
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    for ax, title in zip(ax_list, titles):
        ax.set_title(title)

    fig.suptitle("Trajectory comparison (position in units of RE)")
    _savefig(fig, "solver_trajectories.png")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _savefig(fig, fname):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fpath = os.path.join(PLOT_DIR, fname)
    fig.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  → saved {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RK4, Boris, and RK45 integrators for gtracr."
    )
    parser.add_argument(
        "--no-plots",
        dest="no_plots",
        action="store_true",
        help="Skip saving comparison plots.",
    )
    parser.add_argument(
        "--n-perf",
        dest="n_perf",
        type=int,
        default=200,
        help="Number of trajectories for the performance benchmark (default 200).",
    )
    args = parser.parse_args()
    show_plots = not args.no_plots

    print("gtracr solver comparison")
    print(
        f"  particle   : {PLABEL}, R_forbidden={RIGIDITY_FORBIDDEN} GV / R_allowed={RIGIDITY_ALLOWED} GV"
    )
    print(f"  location   : {LOCATION}")
    print(f"  bfield     : {BFIELD_TYPE}")
    print(f"  max_time   : {MAX_TIME} s  (dt_ref = {DT_REF:.0e})")

    section_accuracy(show_plots)
    section_energy_conservation(show_plots)
    section_performance(args.n_perf, show_plots)
    section_trajectory_overlay(show_plots)

    print("\nDone.")
