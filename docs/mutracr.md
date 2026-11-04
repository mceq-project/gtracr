# Muon Transport (mutracr)

`MuonTracer` — nicknamed **mutracr** — transports batches of atmospheric
muons between production and decay: RK4 integration of the Lorentz force
with continuous dE/dX energy loss, accumulated multiple-scattering
(Fermi–Eyges) variance, and deposition of the *expected* decay weight at
each step midpoint (decays are never sampled).

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
  `"shell"` (IGRF-14 tabulated on an (h, colat, lon) grid over the
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
decay probability and path length). A step crossing the ground is cut
back to the chord crossing (interpolated linearly in r), so grazing
rays end within ~10 cm below the sphere. The returned dict contains the
final states, per-fate weights, and a `weight_audit` (deposited +
residual weights ≈ total source weight).

Decay deposits are consumed by one of three built-in accumulators
(selected with the `deposit=` argument):

- `dict(kind="bank", w_thresh=0.0)` — raw deposit rows (position,
  direction, energy, charge, polarization, MCS sigma, weight) for
  downstream composition, e.g. with the Michel kernels below. The MCS
  sigma is the value before the step's own increment.
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
database helicity polynomials.

## Performance

About 780 000 rays/s on 32 cores (dipole field, `spectrum` tally;
`python examples/eval_mutracr_benchmark.py --full`); the `bank` consumer
is allocation-bound, so prefer the tallies for large runs.
