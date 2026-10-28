# Solvers

gtracr provides three integration methods for the relativistic Lorentz force
equation, all implemented in C++.

## Comparison

| Solver | `solver=` | B-field evals/step | Step size | Key properties |
|--------|-----------|-------------------|-----------|----------------|
| Frozen-field RK4 | `"rk4"` | 1 | Fixed | Default; O(h^4) with frozen-field O(h^2) error |
| Boris pusher | `"boris"` | 1 | Fixed | Symplectic; ~30% faster than RK4; 2x better momentum conservation |
| Adaptive RK45 | `"rk45"` | ~6 (FSAL) | Adaptive | Dormand-Prince; uses `atol`/`rtol`; ~100x fewer steps for escaping trajectories |

## When to use each solver

### Single trajectory visualization

Use **`"rk4"`** or **`"boris"`** with `bfield_type="igrf"`:

- Fixed time steps produce regularly spaced output points, good for plotting.
- All three solvers advance the state in Cartesian (ECEF) coordinates, which
  are regular at the geographic poles, and report the spherical six-vector.
- RK4 is the safe default.

### GMRC cutoff scanning

Use **`"rk45"`** with `bfield_type="table"`:

- Adaptive stepping takes far fewer steps for escaping trajectories.
- Combined with the tabulated IGRF (~7x faster field evaluation), this gives
  the highest throughput (~35k trajectories/s in batch mode).
- Default tolerances (`atol=0`, `rtol=1e-7`) resolve cutoffs to the 0.5 %
  ladder rung (see below).

### High-precision work

Tighten the RK45 tolerance:

```python
traj = Trajectory(..., solver="rk45", rtol=1e-9)
```

## RK45 error norm and tolerances

The RK45 step error is measured on the dimensionless state. Position errors are
lengths in units of R_E, with tolerance `atol + rtol·r/R_E`. Momentum errors are
in units of |p|, with tolerance `atol + rtol`. The norm is the RMS over the six
components, applied to the error vectors, so it does not depend on the coordinate
system..



The error is measured on the dimensionless state (position / R_E,
momentum / |p_0|), so `atol = 1e-3` allows ~6 km in position and 0.1 % in
|p| per step.

| `atol` | `rtol` | Use case |
|--------|--------|----------|
| `0` | `1e-6` | Quick maps (≤ 1 rung in R_eff) |
| `0` | `1e-7` | Default: cutoffs converged on a 0.5 % ladder |
| `0` | `1e-10` | Reference solutions |

## Benchmarking solvers

Run the solver comparison script to compare accuracy and performance:

```bash
python examples/eval_solver_comparison.py --n-perf 100
```
