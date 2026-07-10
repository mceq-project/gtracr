# Muon Transport (mutracr)

`MuonTracer` — nicknamed **mutracr** — transports batches of atmospheric
muons between production and decay: RK4 integration of the Lorentz force
with continuous dE/dX energy loss, per-step *expected*-decay weight
deposition, and accumulated multiple-scattering (Fermi–Eyges) variance.

Because muons below ~100 GeV do not reinteract between production and
decay, transport is a single-particle ODE and a **weighted ensemble of
source rays is an exact quadrature** — no Monte Carlo sampling, no
variance. Decays are never sampled: each step deposits the expected
decay weight `w·(1 − e^{−λ dt})` at the step midpoint, and the ray's
weight decays by the survival factor.

The engine is the C++ port of the nu3d stage-C numpy tracer
(the pure-numpy reference is vendored as `gtracr._mu_reference` and the
test suite pins the two against each other at float-rounding level).

## Units

`MuonTracer` uses the atmospheric-muon literature convention — **not**
the SI units of `Trajectory`:

| quantity | unit |
|----------|------|
| position | cm (Earth-centred Cartesian, z = geographic north) |
| momentum / energy | GeV |
| magnetic field | Tesla |
| density | g/cm³ |
| dE/dX | GeV/(g/cm²) |
| time | s |

## Quickstart

```python
import numpy as np
from gtracr import MuonTracer

mt = MuonTracer(bfield="shell", atmosphere="usstd", r_ground=6.3712e8)

# one 3 GeV/c muon, 20 km above the pole, going straight down
x = np.array([[0.0, 0.0, mt.r_ground + 2.0e6]])
p = np.array([[0.0, 0.0, -3.0]])
res = mt.trace(x, p, q=[-1], deposit=dict(kind="bank"))

print(res["fate"][0])          # 1 = reached ground (see mutracr.FATES)
print(res["deposited"])        # expected decays along the path
print(res["bank"]["e_mu"][:3]) # decay deposit rows
```

## Configuration

Constructor arguments:

- `bfield` — `"none"`, `"dipole"` (gtracr's ideal centred dipole),
  `"shell"` (IGRF-13 tabulated on an (h, colat, lon) grid over the
  atmospheric shell, trilinear interpolation, built once at
  construction; validated against direct IGRF to <0.1%), or
  `("uniform", (bx, by, bz))` in Tesla.
- `atmosphere` — `"none"`, `"usstd"` (Linsley/CORSIKA US-Std analytic
  layers), `("uniform", rho0)`, or `("table", h_cm, rho)` for sampled
  profiles (e.g. MSIS); evenly-spaced tables get a direct-index fast
  path.
- `dedx` — constant stopping power or `(e_kin_grid, a)` table
  (interpolated linearly in ln E_kin, clamped), e.g. from MCEq's
  continuous-loss table.
- `r_ground` — ground-sphere radius in cm. Default 6371.2 km (gtracr's
  Earth radius); MCEq-matched work uses 6391 km.

## Tracing and deposit consumers

`trace(x, p, q, pol=None, w=None, ...)` marches all rays to
termination (`ground` / `stopped` / `faded` / `escaped` / `maxsteps`)
with per-ray adaptive time steps (`ctl_bend`, `ctl_loss`, `ctl_decay`,
`ctl_path` control the per-step bend angle, fractional energy loss,
decay probability and path length). Rays crossing the ground within a
step are pulled back onto the sphere. The returned dict contains the
final states, per-fate weights, and a `weight_audit` (deposited +
residual weights ≈ total source weight).

Decay deposits are consumed by one of three built-in accumulators
(selected with the `deposit=` argument):

- `dict(kind="bank", w_thresh=0.0)` — raw deposit rows (position,
  direction, energy, charge, polarization, MCS sigma, weight) for
  downstream composition, e.g. with the Michel kernels below.
- `dict(kind="spectrum", e_edges=...)` — per-charge E_mu histograms of
  Σw and Σw·pol. Memory-light; use for large batches.
- `dict(kind="angular", e_edges=..., d_edges=..., axis=(0,0,1))` —
  per-charge (E_mu, angle-to-axis) histograms of Σw, Σw·pol, Σw·σ²_MCS
  plus an overflow tally.

Rays are independent, so `trace` parallelises over `n_threads`
std::thread workers (default: all cores) with the GIL released.
Per-ray results are bit-identical for any thread count.

## Michel kernels

`gtracr.mutracr` also provides the polarized μ → ν decay kernels used
to turn decay deposits into neutrino fluxes:

- `rest_kernel(x, cos_th_s, pol, kind)` — rest-frame
  dN/(dx dΩ), `kind` in `("numu", "nue")`.
- `lab_spectrum(e_nu, e_mu, pol, kind, charge)` — angle-integrated lab
  spectrum per decay (normalised to 1).
- `lab_kernel(e_nu, cos_th_lab, e_mu, pol, kind, charge)` — lab-frame
  double-differential kernel toward a detector direction.

The polarization sign convention is pinned against the MCEq decay
database helicity polynomials (nu3d gate C0/T6).

## Performance

Workload: muons injected at 5–30 km, 0–70° tilts, 0.2–20 GeV, traced
through the US-Std atmosphere to ground/decay (32-core x86_64 node,
GCC 11, `-O3 -ffast-math`):

| engine | N rays | threads | rays/s |
|--------|-------:|--------:|-------:|
| numpy reference (bank) | 100 000 | – | ~1 300 |
| C++ dipole (spectrum) | 100 000 | 1 | ~38 000 |
| C++ dipole (spectrum) | 1 000 000 | 32 | ~780 000 |
| C++ shell-IGRF (spectrum) | 1 000 000 | 32 | ~310 000 |
| C++ dipole (bank) | 1 000 000 | 32 | ~85 000 |

The `bank` consumer is allocation-bound (it stores every deposit row);
prefer the `spectrum`/`angular` tallies for production runs.

Reproduce with `python examples/eval_mutracr_benchmark.py --full`.
