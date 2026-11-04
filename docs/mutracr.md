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

## Backward characteristics and source integrals

`MuonTracer.trace_backward(x, p, q, ...)` traces from an observation state
toward possible production points. Momentum is always the **physical forward
momentum**; do not reverse the input momentum or charge. Backward propagation
reverses spatial drift and magnetic curvature, increases energy under inverse
continuous losses, and decreases the probability of surviving to observation.
The upper absorbing sphere is `r_ground + h_escape`; set `h_escape=1.128e7`
for a 112.8 km atmosphere. Forward `trace` is unchanged.

With backward arc length s [cm], physical unit direction u, total energy E,
stopping power a(E), density rho and momentum magnitude p, the equations are

```text
dx/ds = -u
du/ds = -(KP / C_CMS) q (u x B) / p
dE/ds = rho a(E)
dA/ds = m_mu / (p C_CMS tau_mu)
survival = exp(-A)
```

The phase-space focusing of the differential intensity is
J = a(E_past)/a(E_obs) for these separable losses (J = 1 when a = 0); it is
not a derivative of endpoint energies between neighbouring curved rays.

The returned arrays are `x, p, survival, jacobian, path, steps_per_ray, fate`.
`e_max` is a **total** energy limit [GeV]; `path_max` is cm (None means no
path cap). `survival_min`, `max_steps` and positive `ctl_bend, ctl_loss,
ctl_decay, ctl_path` bound the work. Final steps are clipped at the ground,
TOA, energy and survival limits. `BACKWARD_FATES` maps codes to
`maxsteps, ground, e_max, faded, toa, path_max`.

By default `bank=True` also returns midpoint quadrature rows with
`x, phat, e_mu, q, ray, ds, survival, jacobian, w`, where
`w=survival*jacobian*ds`. A source integral is the sum of `w*q_source`
for each ray. Source values must be per cm; include density if a cascade
source is per g/cm². Bank rows preserve ray/path order across thread counts.
This source quadrature has finite numerical error; refine the step controls
to establish convergence.

For a fused compiled source without path storage, pass `bank=False` and
`source=dict(h_cm=..., cos_theta=..., e_mu=..., values=...)`. Axes must be
strictly increasing, with at least two finite nodes each; `e_mu` is total
energy above the muon mass. Values have shape `(2,nH,nMu,nE,nCol)`, with
charge order (+1,-1). `cos_theta` is the local **zenith cosine of the arrival
direction**, `-phat·r̂` (positive for downgoing muons, as in MCEq), so a grid
on [0, 1] covers all downgoing directions. Interpolation is linear in
altitude, cosine and ln(total energy); the source vanishes outside the grid.
Signed response columns are supported. The output `source_integral` has
shape (N,nCol). The C++ `BackwardSourceConsumer` interface supports more
general source evaluators; consumers may run concurrently on distinct ray
indices.

See `examples/mutracr_backward.py` for a nominal plus signed source example.

Backward transport follows deterministic characteristics only: no adjoint
angular diffusion for multiple scattering and no spin precession. Ground
geometry is spherical and the density profile radial.
