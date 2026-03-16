# gtracr — Claude Code Project Guide

## What This Project Does

**gtracr** simulates cosmic ray trajectories through Earth's geomagnetic field. Given a particle's
arrival direction (zenith and azimuth angles), energy or rigidity, and a geographic location, it
back-traces the particle path by integrating the relativistic Lorentz force equation:

```
dp/dt = q (v × B)
```

The primary scientific use case is computing **geomagnetic rigidity cutoffs** (GMRC): for a given
location, what is the minimum rigidity a cosmic ray must have to reach Earth from a given direction?
This is determined via Monte Carlo: thousands of directions are sampled, and for each one the code
finds the minimum rigidity that allows the particle to escape Earth's magnetic field.

---

## Build

The C++ core is compiled as a Python extension module (`_libgtracr`) via pybind11.
The build system uses **meson-python** (PEP-517) with pybind11 as a git submodule (`subprojects/pybind11`).

```bash
# Initialize submodule after clone
git submodule update --init

# Install in editable mode (triggers meson build)
pip install -e . --no-build-isolation
```

**Requirements**: Python ≥ 3.10, a C++14 compiler (GCC, Clang, MSVC), meson ≥ 1.1, ninja,
and the packages in `requirements.txt` (numpy, scipy, tqdm).

---

## Formatting

**You MUST run formatting before every commit. CI will reject unformatted code.**

### Python (ruff)

Config is in `pyproject.toml` under `[tool.ruff]`.

```bash
# Lint + auto-fix, then format
ruff check --fix gtracr/ examples/ && ruff format gtracr/ examples/
```

### C++ (clang-format)

Config is in `.clang-format` (project root). Vendored headers in `gtracr/lib/extern/` are
excluded via a local `.clang-format` with `DisableFormat: true`.

```bash
find gtracr/lib/src gtracr/lib/include gtracr/lib/gpu \
  -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

---

## CI

GitHub Actions workflow at `.github/workflows/ci.yml`.

- **lint** job (ubuntu-latest, Python 3.12): runs `ruff check`, `ruff format --check`, and
  `clang-format --dry-run --Werror` on all C++ files (excluding `extern/`).
- **test** job (depends on lint): full matrix of 5 runners × 5 Python versions:
  - Runners: `macos-latest` (arm64), `macos-13` (Intel), `ubuntu-latest` (x86_64),
    `ubuntu-24.04-arm` (aarch64), `windows-latest` (x86_64)
  - Python: 3.10, 3.11, 3.12, 3.13, 3.14
- `-march=native` is disabled in CI via `meson.options` (`-Dnative_arch=false`).

---

## Test

```bash
# Full test suite
pytest gtracr/tests/ -v

# Individual test files
pytest gtracr/tests/test_trajectories.py -v   # trajectory cases (dipole + IGRF)
pytest gtracr/tests/test_bfield.py -v          # B-field magnitude tests
pytest gtracr/tests/test_solvers.py -v         # solver comparison (RK4, Boris, RK45)
pytest gtracr/tests/test_gmrc.py -v            # GMRC integration tests
pytest gtracr/tests/test_numerical_regression.py -v  # regression against saved baselines
pytest gtracr/tests/test_trajectory_coverage.py -v   # broad trajectory coverage
pytest gtracr/tests/test_particle_location.py -v     # particle/location combinations
pytest gtracr/tests/test_utils.py -v           # utility function tests
```

---

## Run Examples

```bash
# Single trajectory
python examples/eval_trajectory.py

# Geomagnetic cutoff rigidities for a location
python examples/eval_gmcutoff.py

# Benchmarks
python examples/eval_benchmarks.py
```

---

## Architecture

```
User (Python)
  │
  ├── Trajectory (gtracr/trajectory.py)
  │     Sets up initial conditions (6-vector in spherical geocentric coords),
  │     selects the integrator, calls get_trajectory()
  │
  ├── GMRC (gtracr/geomagnetic_cutoffs.py)
  │     Monte Carlo over 10,000 random (zenith, azimuth) angles;
  │     for each direction, scans rigidities to find the cutoff.
  │     Two evaluation modes:
  │       evaluate()       — Python-orchestrated (ProcessPool or ThreadPool)
  │       evaluate_batch() — entire MC loop in C++ (BatchGMRC)
  │
  └── pybind11 extension: gtracr.lib._libgtracr
        │
        ├── TrajectoryTracer (C++)       ← PRIMARY integrator
        │     RK4/Boris/RK45 integration of the Lorentz ODE in spherical coords.
        │     Supports three B-field backends: dipole, direct IGRF, tabulated IGRF.
        │
        ├── BatchGMRC (C++)              ← BATCH evaluator
        │     Entire GMRC MC loop in C++: RNG, coordinate transforms, rigidity
        │     scanning, std::thread parallelism. Eliminates all Python overhead.
        │
        ├── IGRF (C++)                   ← B-field model (direct)
        │     Degree-13 spherical harmonic expansion (IGRF-13).
        │     Coefficients loaded from gtracr/data/igrf13.json at construction.
        │
        └── IGRF Table (C++)             ← B-field model (tabulated)
              3D lookup table (64×128×256 grid, 24 MB) with trilinear
              interpolation. Generated from IGRF, cached to disk as .npy.
              Located in gtracr/lib/gpu/igrf_table.{hpp,cpp}.
```

### Coordinate System

All integration is in **geocentric spherical coordinates** `(r, θ, φ)`:
- `r` — radial distance from Earth's center (meters)
- `θ` — polar angle / colatitude (radians; 0 = north pole)
- `φ` — azimuthal angle / longitude (radians)

The 6-vector state is `(r, θ, φ, pᵣ, pθ, pφ)` where `p` is relativistic momentum.

### Integration

- **Method**: 4th-order Runge-Kutta (RK4), fixed step size (default `dt = 1e-5 s`)
- **Termination**: particle escapes (`r > 10 RE`) → trajectory *allowed*; particle returns to
  atmosphere (`r < start_altitude + RE`) → trajectory *forbidden*
- **Max iterations**: `max_iter = ceil(max_time / dt)`, default `max_time = 1 s` → 100,000 steps

### Magnetic Field Models

| Type | `bfield_type=` | Class | Description |
|------|----------------|-------|-------------|
| Dipole | `'dipole'` | `MagneticField` (C++) | Ideal dipole, 1/r³ falloff |
| Direct IGRF | `'igrf'` | `IGRF` (C++) | IGRF-13 spherical harmonics, degree 13, 1900–2025 |
| Tabulated IGRF | `'table'` | `igrf_table` (C++) | 3D lookup table (64×128×256), trilinear interpolation; ~7× faster than direct IGRF |

The tabulated field is generated once from the direct IGRF model and cached to disk
(`gtracr/data/igrf_table_<year>.npy`). It is the default for `GMRC.evaluate()` when
`bfield_type="table"` and is required for `evaluate_batch()`.

---

## Key Classes

### `Trajectory` (`gtracr/trajectory.py`)

```python
traj = Trajectory(
    zenith_angle=0.,       # degrees from local zenith
    azimuth_angle=0.,      # degrees from geographic north
    rigidity=10.,          # GV  (or energy= in GeV)
    location_name="IceCube",
    bfield_type="igrf",    # "igrf" or "dipole"
    plabel="p+",           # "p+", "p-", "e+", "e-"
)
traj.get_trajectory(dt=1e-5, max_time=1.)
print(traj.particle_escaped)   # True = allowed trajectory
```

### `GMRC` (`gtracr/geomagnetic_cutoffs.py`)

```python
# Python-orchestrated evaluation (ProcessPool for igrf/dipole, ThreadPool for table)
gmrc = GMRC(location="Kamioka", iter_num=10000, bfield_type="igrf",
            solver="rk4", n_workers=8)
gmrc.evaluate(dt=1e-5, max_time=1.)
az_grid, zen_grid, cutoff_grid = gmrc.interpolate_results()

# C++ batch mode — entire MC loop in C++, fastest option (~35k traj/s with table+rk45)
gmrc = GMRC(location="Kamioka", iter_num=10000, bfield_type="table",
            solver="rk45", atol=1e-3, rtol=1e-6)
gmrc.evaluate_batch(dt=1e-5, max_time=1.)
az_centres, zen_centres, cutoff_grid = gmrc.bin_results()
```

**`evaluate()` threading modes:**
- `bfield_type="igrf"` or `"dipole"` → `ProcessPoolExecutor` (GIL-bound)
- `bfield_type="table"` → `ThreadPoolExecutor` (GIL released in C++), shared table in memory

**`evaluate_batch()`:** Calls `BatchGMRC` (C++) — RNG, coordinate transforms, rigidity
scanning, and `std::thread` parallelism all in C++. No Python overhead per trajectory.

**Result methods:**
- `interpolate_results()` — scipy `griddata` scattered interpolation (legacy)
- `bin_results()` — fast binning into regular azimuth/zenith grid (preferred for large N)

---

## Known Issues and Technical Debt

### Horizontal trajectories at the equator terminate immediately
For `zenith_angle=90` (horizontal) at equatorial latitudes, the Lorentz force pushes the
back-traced proton radially inward in the first RK4 step, immediately triggering the atmosphere
termination condition. This is physically correct (East-West geomagnetic asymmetry): the
Störmer cutoff for eastward horizontal at the equator is ~52 GV (IGRF), so particles below
that rigidity are genuinely forbidden. The 1-step termination happens because `r₀ = threshold`
exactly — any inward motion fires the condition. Trajectories are useless for visualization
in this regime; use mid-latitude locations (e.g. `location_name="Kamioka"`) or
`azimuth_angle=270` (westward) for horizontal trajectory visualization.

---

## Performance Notes and Bottlenecks

### Completed Optimizations
- **Parallel MC loop**: `GMRC.evaluate()` uses `ProcessPoolExecutor` (direct IGRF) or
  `ThreadPoolExecutor` (tabulated IGRF, GIL released); default `n_workers=None` uses all CPU cores
- **Frozen-field RK4**: B-field evaluated once per step (not 4×); reduces IGRF calls 4× per step
- **TrajectoryTracer caching in GMRC**: one `TrajectoryTracer` built per thread; `reset()+evaluate()`
  loops over rigidities without reloading `igrf13.json`
- **std::vector pre-allocation**: `reserve(max_iter_)` already present in `evaluate_and_get_trajectory()`
- **3D IGRF lookup table**: 64×128×256 grid (24 MB) with trilinear interpolation replaces
  Legendre polynomial recursion; generated once, cached to disk as `.npy`
- **BatchGMRC (C++)**: entire GMRC MC loop in C++ — RNG, coordinate transforms, rigidity scanning,
  `std::thread` parallelism — eliminates all Python overhead per trajectory (~35k traj/s with table+RK45)
- **Table disk caching**: IGRF tables are cached as `gtracr/data/igrf_table_<year>.npy` + `_params.npz`
  to avoid regeneration across runs
- **Legacy code removed**: `uTrajectoryTracer` (C++), `gtracr/legacy/` Python modules, vendored
  pybind11 headers (now via submodule)

### Additional Solvers (completed)

Three integration methods are now available in `TrajectoryTracer` (C++), selectable via
the `solver=` keyword in `Trajectory`, `GMRC`, and all CLI scripts (`--solver`):

| Solver | `solver=` | B-evals/step | Notes |
|--------|-----------|--------------|-------|
| Frozen-field RK4 | `"rk4"` | 1 | Default; O(h⁴) with frozen-field O(h²) error |
| Boris pusher | `"boris"` | 1 | Symplectic; Cartesian internally; ~30% faster |
| Adaptive RK45 | `"rk45"` | ~6 (FSAL) | Dormand-Prince; uses `atol=`, `rtol=` |

Numerical comparison script: `examples/eval_solver_comparison.py --n-perf N`

Key findings:
- Boris is **30% faster** than RK4 at equal step count, with **2× better momentum conservation**
- RK45 is extremely fast for coarse GMRC (~100× fewer steps) but needs tight tolerances
  (`atol≲1e-6, rtol≲1e-6`) for position accuracy comparable to RK4 at dt=1e-5

### Improvement Roadmap

**Next step: GPU acceleration** (see `GPU-plan.md` for detailed design)
- HIP portability layer (CUDA + ROCm) with one thread per trajectory
- The 3D IGRF lookup table (already implemented on CPU) is the key enabler — no branching,
  just trilinear interpolation in device global memory
- Warp-level early exit via `__ballot_sync` for variable-length trajectories
- Estimated 1,000–10,000× speedup for 10,000+ trajectory batches

---

## Data Files

| File | Description |
|------|-------------|
| `gtracr/data/IGRF13.COF` | Original IGRF-13 coefficient file |
| `gtracr/data/IGRF13.shc` | Spherical harmonic coefficient format |
| `gtracr/data/igrf13.json` | JSON format used by C++ `IGRF` class at runtime |
| `gtracr/data/igrf_table_<year>.npy` | Cached 3D IGRF lookup table (auto-generated) |
| `gtracr/data/igrf_table_<year>_params.npz` | Grid metadata for cached table |
| `gtracr/data/benchmark_data.pkl` | Saved benchmark/regression baselines |

---

## Pre-defined Locations

Kamioka, IceCube, SNOLAB, UofA, CTA-North, CTA-South, ORCA, ANTARES, Baikal-GVD, TA.

See `gtracr/utils.py` for coordinates.

---

## Documentation Maintenance

**When changing any public Python API (adding/removing/renaming parameters, changing defaults,
adding new classes or methods), you MUST update:**

1. The numpy-style docstring on the changed function/class
2. The relevant MkDocs page in `docs/` (trajectory.md, geomagnetic_cutoffs.md, solvers.md)
3. The quickstart examples in `docs/index.md` if the basic usage pattern changed
4. README.md if the change affects the high-level feature description
