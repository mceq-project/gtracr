# Benchmarks

## Solver performance

Measured with the solver comparison script
(`examples/eval_solver_comparison.py`):

| Solver | B-field | Relative speed | Notes |
|--------|---------|---------------|-------|
| RK4 | IGRF | 1x (baseline) | Fixed step, dt=1e-5 |
| Boris | IGRF | ~1.3x | Fixed step, dt=1e-5 |
| RK45 | IGRF | ~5-50x | Adaptive; depends on trajectory length |
| RK4 | Table | ~7x | Tabulated field eliminates SH recursion |
| RK45 | Table | ~50-500x | Best combination for batch work |

## GMRC throughput

| Mode | Field | Solver | Throughput |
|------|-------|--------|------------|
| `evaluate()` (ProcessPool) | IGRF | RK4 | ~500 traj/s |
| `evaluate()` (ThreadPool) | Table | RK4 | ~5k traj/s |
| `evaluate_batch()` (C++) | Table | RK45 | ~35k traj/s |

Throughput measured on an 8-core Apple M1 with 10,000 MC samples.

## Running benchmarks

```bash
# Solver comparison (accuracy + timing)
python examples/eval_solver_comparison.py --n-perf 100

# Full GMRC benchmark
python examples/eval_benchmarks.py
```
