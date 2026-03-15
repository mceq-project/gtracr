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
The build system uses **meson-python** (PEP-517) with pybind11 v3.0.2 as a git submodule.

```bash
# Initialize submodule after clone
git submodule update --init

# Install in editable mode (triggers meson build)
pip install -e . --no-build-isolation
```

**Requirements**: Python ≥ 3.6, a C++11 compiler (GCC, Clang, MSVC), meson ≥ 1.1, ninja,
and the packages in `requirements.txt` (numpy, scipy, tqdm).

---

## Test

```bash
# Full test suite
pytest gtracr/tests/ -v

# Individual test files
pytest gtracr/tests/test_trajectories.py -v   # 13 trajectory cases (dipole + IGRF)
pytest gtracr/tests/test_bfield.py -v          # 10 B-field magnitude tests
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
  │     for each direction, scans rigidities to find the cutoff
  │
  └── pybind11 extension: gtracr.lib._libgtracr
        │
        ├── TrajectoryTracer (C++)       ← PRIMARY integrator
        │     RK4 integration of the Lorentz ODE in spherical coordinates.
        │     Uses std::array<double,6> vector operations.
        │
        └── IGRF (C++)                   ← B-field model
              Degree-13 spherical harmonic expansion (IGRF-13).
              Coefficients loaded from gtracr/data/igrf13.json at construction.
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

| Type | Class | Description |
|------|-------|-------------|
| `'dipole'` | `MagneticField` (C++) | Ideal dipole, 1/r³ falloff |
| `'igrf'` | `IGRF` (C++) | IGRF-13 spherical harmonics, degree 13, 1900–2025 |

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
gmrc = GMRC(location="Kamioka", iter_num=10000, bfield_type="igrf")
gmrc.evaluate(dt=1e-5, max_time=1.)
az_grid, zen_grid, cutoff_grid = gmrc.interpolate_results()
```

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
- **Frozen-field RK4**: B-field evaluated once per step (not 4×); reduces IGRF calls 4× per step
- **TrajectoryTracer caching in GMRC**: one `TrajectoryTracer` built per direction; `reset()+evaluate()`
  loops over rigidities without reloading `igrf13.json`
- **std::vector pre-allocation**: `reserve(max_iter_)` already present in `evaluate_and_get_trajectory()`

### Bottleneck Map (remaining)

| # | Bottleneck | Location | Impact |
|---|-----------|----------|--------|
| 1 | IGRF Legendre evaluation per RK step | `igrf.cpp:shval3` | Medium |

### Improvement Roadmap

**Medium-term (weeks):**
- Precompute a 3D B-field grid (r, θ, φ) → 10–30× speedup on field evaluation; eliminates
  the recursive Legendre polynomial evaluation (4× per RK4 step) with trilinear interpolation
- Implement [Boris integrator](https://en.wikipedia.org/wiki/Boris_method) (1 B-eval/step vs 4
  for RK4, better energy conservation)
- Implement adaptive RK45 (Dormand-Prince) for fewer total steps

**Long-term (GPU, months):**
- The computation is *embarrassingly parallel* at the trajectory level — each trajectory is
  completely independent
- **JAX + vmap**: `jit(vmap(simulate_trajectory))(batch_of_ics)` runs all trajectories in parallel
  on GPU; requires replacing IGRF Legendre recursion with precomputed table lookup (no branching)
- **Numba CUDA kernel**: `@cuda.jit` kernel with one CUDA thread per trajectory; precomputed table
  in device memory
- **Custom CUDA C++ kernel**: highest performance; one thread per trajectory; IGRF table in shared
  memory per block; 500–2000× speedup estimated for 10,000+ trajectory batches

The key enabler for GPU is the **precomputed 3D IGRF table**: the recursive Legendre polynomial
evaluation has loop-carried dependencies that cannot be parallelized across field points, but a
3D lookup table with trilinear interpolation requires only 8 multiplications per query.

---

## Data Files

| File | Description |
|------|-------------|
| `gtracr/data/IGRF13.COF` | Original IGRF-13 coefficient file |
| `gtracr/data/IGRF13.shc` | Spherical harmonic coefficient format |
| `gtracr/data/igrf13.json` | JSON format used by C++ `IGRF` class at runtime |

---

## Pre-defined Locations

Kamioka, IceCube, SNOLAB, UofA, CTA-North, CTA-South, ORCA, ANTARES, Baikal-GVD, TA.

See `gtracr/utils.py` for coordinates.
