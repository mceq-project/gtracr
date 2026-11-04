# CHANGELOG

Presents the version history of the package.

### Version 3.0.0

Breaking changes:

- **Azimuth convention.** `azimuth_angle` (Trajectory) and the GMRC
  azimuths are now the azimuth of the arrival direction measured from
  geographic north, clockwise (0 = from north, 90 = from east; astropy AltAz,
  MCEq). Before, 0 meant from south (Honda's convention) while the docstring
  claimed a north-based angle. Convert old inputs and maps with
  `(180 - azimuth_old) % 360`; see docs/conventions.md.
- `batch_gmrc_evaluate` returns `(zenith, azimuth, rcutoff, above_max,
  total_trajectories)`; GMRC keeps directions whose cutoff exceeds
  `max_rigidity` (clipped to it, flagged in `data_dict["above_rmax"]`)
  instead of dropping them (batch) or storing 0 (Python). The Python
  rigidity list includes `max_rigidity`.
- IGRF-14 replaces IGRF-13 (`igrf14.json`, `IGRF14.shc`, `IGRF14.COF`);
  `gtracr.bfield.IGRF13` is a deprecated alias of `IGRF14`.
- **Escape criterion.** A trajectory is allowed when it reaches
  `max(escape_radius, k·C_St(p))` (Störmer length C_St of the field model's dipole)
  within `max(max_time, N·r_esc/(βc))`; defaults k = 4, N = 80. `max_time` and the
  escape radius are now lower bounds; `escape_factor=0, time_factor=0` restores the
  fixed criterion. Changes cutoffs below about 10 GV (South Pole: 0.20 → 0.067 GV);
  validity and cost in docs/geomagnetic_cutoffs.md.
- **RK45 defaults and norm.** The error is measured on the dimensionless state
  (position in R_E with tolerance atol + rtol·r/R_E, momentum in |p| with
  atol + rtol); defaults change from `atol=1e-3, rtol=1e-6` to `atol=0, rtol=1e-7`.

New:

- **MuonTracer ("mutracr")**: batch transport of atmospheric muons between
  production and decay (RK4, continuous dE/dX loss, expected-decay weight
  deposits, Fermi–Eyges MCS variance, std::thread parallelism); see
  docs/mutracr.md.
- Polarized Michel mu → nu kernels in `gtracr.mutracr`.
- Benchmark: `examples/eval_mutracr_benchmark.py`.
- `MuonTracer.trace_backward`: backward muon characteristics with signed,
  multi-column tabulated production sources and the CSDA intensity Jacobian;
  example `examples/mutracr_backward.py`.
- `GMRC(method="penumbra")`: Cooke et al. (1991) R_L, R_U, R_eff per direction on a
  geometric rigidity ladder (adaptive scan) in the C++ batch engine
  (`batch_gmrc_penumbra`); explicit direction grids via
  `evaluate_batch(zenith=..., azimuth=...)`; `store_ladder=True` returns the
  allowed/forbidden ladder; `flag` marks cutoffs outside the ladder (1, 2) and
  penumbrae reaching below it (3). The default `method="bisection"` is unchanged.
- Escape criterion parameters `escape_factor` / `time_factor` on `Trajectory`, `GMRC`
  and `TrajectoryTracer.set_escape_scaling`; logic in `IntegratorControl.hpp` and
  `gtracr.escape`.

Fixes:

- Exact SI constants (rigidity labels were 0.30 % low); proton mass 0.938272088 GeV.
- RK4 last stage used `y + h*k3` with `k3` already scaled by h (not fourth order).
- RK4/RK45 integrate in Cartesian coordinates (no NaN or |p| blow-up near the poles).
- C++ IGRF epoch selection interpolated from the wrong epoch after 2.5 years.
- RK45 tolerances were applied to raw SI components.
- Dipole model: g10 from IGRF-14 at 2025.0 (29 350.0 nT, was IGRF-2020).
- CTA-South longitude (#23); batch mode builds the dipole tracer directly for `bfield_type="dipole"`.

### Version 1.0.0

Initial release of the gtracr package.
Major bugs are now fixed and is now ready for initial release.

- Examples are cleaned up and are ready to be used with users.
- Plotter now plots multiple trajectories
- Created a new class for array storage / operator overloading in replace to std::array
- Bug fixes for compatibility with UNIX systems and Windows

### version 0.10.0

Refactoring of the code was performed:

- Evaluation of the geomagnetic cutoffs are now done by class-level objects
- Plotters are separated with the package / evaluation process

### version 0.7.0

The initial pre-release of the beta version.

- The code evaluates the cosmic ray trajectories and the geomagnetic cutoff rigidities.
- It does _not_ support any sort of parallelization (GPU / CPU).
