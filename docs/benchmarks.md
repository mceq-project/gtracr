# Benchmarks

Measured on an Apple M3 Pro (12 threads) with `min_rigidity=1`, `max_rigidity=55`,
`delta_rigidity=0.5`, `dt=1e-5`, `max_time=1.0`.

## GMRC throughput

| Mode | Field | Solver | Samples | Successful traj/s | Total traj evaluated | Total traj/s | Time |
|------|-------|--------|---------|-------------------|----------------------|--------------|------|
| `evaluate_batch()` (C++) | Table | RK45 | 100k | 48k/s | ~839k | ~400k/s | 2.1s |
| `evaluate()` (ThreadPool) | Table | RK45 | 10k | 11k/s | ~84k (est.) | ~93k/s | 0.9s |
| `evaluate()` (ProcessPool) | IGRF | RK4 | 1k | 67/s | ~8k (est.) | ~540/s | 14.9s |

"Total traj evaluated" is the number of individual trajectory integrations
(each MC direction requires scanning multiple rigidities to find the cutoff).
The batch mode reports this exactly; Python modes are estimated from the
~8.4 trajectories/direction ratio observed in batch mode.

## Solver performance

| Solver | B-field | Relative speed | Notes |
|--------|---------|---------------|-------|
| RK4 | IGRF | 1× (baseline) | Fixed step, dt=1e-5 |
| Boris | IGRF | ~1.3× | Fixed step, dt=1e-5 |
| RK45 | IGRF | ~5–50× | Adaptive; depends on trajectory length |
| RK4 | Table | ~7× | Tabulated field eliminates SH recursion |
| RK45 | Table | ~50–500× | Best combination for batch work |

## Running benchmarks

```bash
# Solver comparison (accuracy + timing)
python examples/eval_solver_comparison.py --n-perf 100

# Full GMRC benchmark
python examples/eval_gmcutoff.py --iter-num 100000 --bfield-type table --solver rk45
```
