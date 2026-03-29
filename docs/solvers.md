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
- Boris is faster and conserves energy better, but outputs Cartesian
  internally (converted back to spherical).
- RK4 is the safe default.

### GMRC cutoff scanning

Use **`"rk45"`** with `bfield_type="table"`:

- Adaptive stepping takes far fewer steps for escaping trajectories.
- Combined with the tabulated IGRF (~7x faster field evaluation), this gives
  the highest throughput (~35k trajectories/s in batch mode).
- Default tolerances (`atol=1e-3`, `rtol=1e-6`) work well for cutoff
  determination.

### High-precision work

For position accuracy comparable to RK4 at `dt=1e-5`, tighten the RK45
tolerances:

```python
traj = Trajectory(..., solver="rk45", atol=1e-6, rtol=1e-6)
```

## Tolerance tuning (RK45)

| `atol` | `rtol` | Use case |
|--------|--------|----------|
| `1e-3` | `1e-6` | Default — good for GMRC cutoff determination |
| `1e-6` | `1e-6` | High precision — position accuracy matching RK4 at dt=1e-5 |
| `1e-8` | `1e-8` | Reference solutions for validation |

## Benchmarking solvers

Run the solver comparison script to compare accuracy and performance:

```bash
python examples/eval_solver_comparison.py --n-perf 100
```
