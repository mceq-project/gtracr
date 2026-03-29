# Magnetic Field Models

gtracr exposes three Python magnetic field classes via `gtracr.bfield`.
All three share the same interface and can be used directly — for example,
to query the field at a point or to drive the pure-Python fallback integrator.

```python
from gtracr.bfield import MagneticField, IGRF13, IGRFTable
```

## Classes

### `MagneticField` — ideal dipole

Evaluates the first-order term of the geomagnetic scalar potential
(1/r³ falloff). Fastest to evaluate; useful for coarse studies or
sanity checks. Uses the IGRF-2020 G10 coefficient (29 404.8 nT).

```python
from gtracr.bfield import MagneticField

bf = MagneticField()
Br, Btheta, Bphi = bf.values(r=6.371e6, theta=1.0, phi=0.0)
```

### `IGRF13` — spherical harmonic model

Full IGRF-13 evaluation up to degree 13 using time-interpolated
Schmidt quasi-normalized coefficients (valid 1900–2025).

```python
from gtracr.bfield import IGRF13
from gtracr.utils import ymd_to_dec

bf = IGRF13(curr_year=float(ymd_to_dec("2024-01-01")))
Br, Btheta, Bphi = bf.values(r=6.371e6, theta=1.0, phi=0.5)
```

| Parameter | Description |
|-----------|-------------|
| `curr_year` | Decimal year for coefficient interpolation (e.g. `2024.5`). |
| `nmax` | Truncation degree (default: 13, the maximum in IGRF-13). |

### `IGRFTable` — pre-computed lookup table

Trilinear interpolation over a pre-computed 3D grid
(64 × 128 × 256 in r, θ, φ). ~7x faster than `IGRF13` for repeated
queries. Covers 1–10 Earth radii; the dipole approximation is used outside
this range.

The table is generated automatically on first use and cached to
`gtracr/data/igrf_table_<year>.npy` and `igrf_table_<year>_params.npz`.
Subsequent runs load the cache directly.

```python
from gtracr.bfield import IGRFTable

bf = IGRFTable()  # builds or loads cached table for today's date
Br, Btheta, Bphi = bf.values(r=6.371e6, theta=1.0, phi=0.5)
```

| Parameter | Description |
|-----------|-------------|
| `igrf_obj` | Optional pre-built C++ `IGRF` object. Built from `igrf13.json` if `None`. |
| `verbose` | Print progress while generating the table (default `False`). |

## Common interface

All three classes implement:

```python
bf.values(r, theta, phi) -> numpy.ndarray shape (3,)
```

Returns `(Br, Btheta, Bphi)` in Tesla at geocentric spherical coordinates
`(r [m], theta [rad], phi [rad])`.

## Clearing the table cache

If you need to regenerate the cached lookup table (e.g. after updating the
IGRF coefficients or changing the evaluation date), delete the relevant files
from the data directory:

```bash
rm gtracr/data/igrf_table_*.npy gtracr/data/igrf_table_*_params.npz
```

The table will be regenerated on the next run.
