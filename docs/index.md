# gtracr

**gtracr** simulates cosmic ray trajectories through Earth's geomagnetic field
and computes geomagnetic rigidity cutoffs (GMRC).

It back-traces particle paths by integrating the relativistic Lorentz force
equation in geocentric spherical coordinates, using the IGRF-13 magnetic field
model. Three integration solvers are available (RK4, Boris, RK45), and a
tabulated IGRF lookup table provides ~7x speedup over direct spherical
harmonic evaluation.

## Installation

### Requirements

- Python >= 3.10
- C++14 compiler (GCC, Clang, or MSVC)
- meson >= 1.1, ninja
- numpy, scipy, tqdm

### From source

```bash
git clone https://github.com/kwat0308/gtracr.git
cd gtracr
git submodule update --init
pip install -e . --no-build-isolation
```

## Quickstart

### Single trajectory

```python
from gtracr.trajectory import Trajectory

traj = Trajectory(
    zenith_angle=45., azimuth_angle=0., rigidity=20.,
    location_name="Kamioka", bfield_type="igrf",
)
data = traj.get_trajectory(get_data=True)
print(traj.particle_escaped)  # True = allowed trajectory
```

### Geomagnetic cutoff map

```python
from gtracr.geomagnetic_cutoffs import GMRC

gmrc = GMRC(location="Kamioka", iter_num=10000,
            bfield_type="table", solver="rk45")
gmrc.evaluate_batch(dt=1e-5, max_time=1.)
az, zen, cutoffs = gmrc.bin_results()
```

See the detailed guides for [single trajectories](trajectory.md) and
[cutoff maps](geomagnetic_cutoffs.md).
