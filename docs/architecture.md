# Architecture

## Overview

```
User (Python)
  │
  ├── Trajectory (gtracr/trajectory.py)
  │     Sets up initial conditions (6-vector in spherical geocentric coords),
  │     selects the integrator, calls get_trajectory()
  │
  ├── GMRC (gtracr/geomagnetic_cutoffs.py)
  │     Monte Carlo over random (zenith, azimuth) angles;
  │     for each direction, scans rigidities to find the cutoff.
  │     Two evaluation modes:
  │       evaluate()       — Python-orchestrated (ProcessPool or ThreadPool)
  │       evaluate_batch() — entire MC loop in C++ (BatchGMRC)
  │
  └── pybind11 extension: gtracr.lib._libgtracr
        │
        ├── TrajectoryTracer (C++)       ← primary integrator
        │     RK4/Boris/RK45 integration of the Lorentz ODE.
        │
        ├── BatchGMRC (C++)              ← batch evaluator
        │     Entire GMRC MC loop in C++: RNG, coordinate transforms,
        │     rigidity scanning, std::thread parallelism.
        │
        ├── IGRF (C++)                   ← direct B-field model
        │     Degree-13 spherical harmonic expansion (IGRF-13).
        │
        └── IGRF Table (C++)             ← tabulated B-field model
              3D lookup table with trilinear interpolation.
```

## Coordinate system

All integration is in **geocentric spherical coordinates** `(r, theta, phi)`:

| Component | Description | Units |
|-----------|-------------|-------|
| `r` | Radial distance from Earth's center | meters |
| `theta` | Polar angle / colatitude (0 = north pole) | radians |
| `phi` | Azimuthal angle / longitude | radians |

The 6-vector state is `(r, theta, phi, p_r, p_theta, p_phi)` where `p` is
relativistic momentum in SI units.

## Magnetic field models

| Model | `bfield_type=` | C++ class | Description |
|-------|----------------|-----------|-------------|
| Dipole | `"dipole"` | `MagneticField` | Ideal dipole, 1/r^3 falloff |
| Direct IGRF | `"igrf"` | `IGRF` | IGRF-13 spherical harmonics, degree 13 |
| Tabulated IGRF | `"table"` | `igrf_table` | 3D lookup table (64x128x256), trilinear interpolation |

The tabulated field is ~7x faster than direct IGRF evaluation. It is
generated once from the direct model and cached to disk as
`gtracr/data/igrf_table_<year>.npy`.

## Python / C++ boundary

The `_libgtracr` pybind11 extension exposes:

- **`TrajectoryTracer`** — constructs from physical parameters (charge, mass,
  field type, solver type), then `evaluate()` or
  `evaluate_and_get_trajectory()` integrates one trajectory.
- **`batch_gmrc_evaluate()`** — runs the full GMRC Monte Carlo in C++ with
  `std::thread` parallelism.
- **`generate_igrf_table()`** — generates the 3D lookup table from IGRF
  coefficients.

The GIL is released during C++ integration, allowing true parallelism with
`ThreadPoolExecutor` when using the tabulated field.

## Integration termination

- **Escaped**: `r > 10 * EARTH_RADIUS` — trajectory is *allowed*
- **Returned**: `r < start_altitude + EARTH_RADIUS` — trajectory is *forbidden*
- **Timeout**: `max_iter` steps reached — trajectory is *forbidden*

## Data files

| File | Description |
|------|-------------|
| `gtracr/data/igrf13.json` | IGRF-13 coefficients (loaded by C++ at runtime) |
| `gtracr/data/IGRF13.shc` | Spherical harmonic coefficients (Python IGRF) |
| `gtracr/data/igrf_table_<year>.npy` | Cached 3D lookup table (auto-generated) |
| `gtracr/data/igrf_table_<year>_params.npz` | Grid metadata for cached table |
