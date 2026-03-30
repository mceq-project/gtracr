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

Single forbidden trajectory at Kamioka, zenith=45°, azimuth=90°, rigidity=7 GV,
dt=1e-5, max_time=1 s. Table pre-loaded once; numbers measure integration only.

| Solver | B-field | Steps | Eval/s | Ratio to RK4+IGRF |
|--------|---------|------:|-------:|:-----------------:|
| RK4    | IGRF    | 9372  | 216    | 1.00×             |
| Boris  | IGRF    | 9159  | 228    | 1.06×             |
| RK45   | IGRF    | 56    | 5393   | 25×               |
| RK4    | Dipole  | 11823 | 906    | 4.2×              |
| Boris  | Dipole  | 11794 | 1047   | 4.8×              |
| RK45   | Dipole  | 39    | 98156  | 454×              |
| RK4    | Table   | 9300  | 1067   | 4.9×              |
| Boris  | Table   | 9091  | 1339   | 6.2×              |
| RK45   | Table   | 55    | 59446  | 275×              |

## Running benchmarks

```bash
# Solver comparison (accuracy + timing)
python examples/eval_solver_comparison.py --n-perf 100

# Full GMRC benchmark
python examples/eval_gmcutoff.py --iter-num 100000 --bfield-type table --solver rk45
```
