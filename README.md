# gtracr

![CI](https://github.com/kwat0308/gtracr/actions/workflows/ci.yml/badge.svg)

**gtracr** simulates cosmic ray trajectories through Earth's geomagnetic field
and computes geomagnetic rigidity cutoffs (GMRC).

## Features

- **IGRF-13 magnetic field** — full degree-13 spherical harmonic model, plus a
  fast 3D tabulated lookup (~7x speedup)
- **Three solvers** — RK4, Boris pusher (~30% faster), and adaptive RK45
  (~100x fewer steps for escaping trajectories)
- **C++ batch mode** — entire GMRC Monte Carlo loop in C++ with std::thread
  parallelism (~35k trajectories/s)
- **10 predefined locations** — Kamioka, IceCube, SNOLAB, and more

## Installation

Requires Python >= 3.10, a C++14 compiler, meson >= 1.1, and ninja.

```bash
git clone https://github.com/kwat0308/gtracr.git
cd gtracr
git submodule update --init
pip install -e . --no-build-isolation
```

## Quickstart

```python
from gtracr.trajectory import Trajectory

traj = Trajectory(
    zenith_angle=45., azimuth_angle=0., rigidity=20.,
    location_name="Kamioka", bfield_type="igrf",
)
data = traj.get_trajectory(get_data=True)
print(traj.particle_escaped)  # True = allowed trajectory
```

```python
from gtracr.geomagnetic_cutoffs import GMRC

gmrc = GMRC(location="Kamioka", iter_num=10000,
            bfield_type="table", solver="rk45")
gmrc.evaluate_batch(dt=1e-5, max_time=1.)
az, zen, cutoffs = gmrc.bin_results()
```

## Documentation

See the [full documentation](https://kwat0308.github.io/gtracr/) for detailed
guides on trajectories, cutoff maps, solvers, and architecture.

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
