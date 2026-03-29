# Single Trajectories

The `Trajectory` class evaluates a single cosmic ray trajectory through
Earth's geomagnetic field.

## Basic usage

```python
from gtracr.trajectory import Trajectory

traj = Trajectory(
    zenith_angle=45.,
    azimuth_angle=0.,
    rigidity=20.,
    location_name="Kamioka",
)
data = traj.get_trajectory(get_data=True)
```

The returned dictionary contains time and the six-vector in spherical
coordinates:

| Key | Description |
|-----|-------------|
| `t` | Time array (seconds) |
| `r` | Radial distance from Earth's center (m) |
| `theta` | Colatitude (rad) |
| `phi` | Longitude (rad) |
| `pr` | Radial momentum (kg m/s) |
| `ptheta` | Polar momentum (kg m/s) |
| `pphi` | Azimuthal momentum (kg m/s) |
| `x`, `y`, `z` | Cartesian coordinates (Earth radii) |

After evaluation, check `traj.particle_escaped` to determine if the
trajectory was allowed (`True`) or forbidden (`False`).

## Parameters

### Required (one of energy or rigidity)

| Parameter | Type | Description |
|-----------|------|-------------|
| `zenith_angle` | float | Angle from local zenith (degrees). 0 = overhead, 90 = horizontal. |
| `azimuth_angle` | float | Angle from geographic south in the tangent plane (degrees). |
| `energy` | float | Total energy in GeV (mutually exclusive with `rigidity`). |
| `rigidity` | float | Rigidity in GV (mutually exclusive with `energy`). |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `location_name` | `None` | Predefined location name (overrides lat/lon/alt). |
| `latitude` | `0.` | Geographic latitude (degrees). |
| `longitude` | `0.` | Geographic longitude (degrees). |
| `detector_altitude` | `0.` | Detector altitude above sea level (km). |
| `particle_altitude` | `100.` | Atmosphere interaction altitude (km). |
| `plabel` | `"p+"` | Particle type: `"p+"`, `"p-"`, `"e+"`, `"e-"`. |
| `bfield_type` | `"igrf"` | Field model: `"igrf"`, `"dipole"`, or `"table"`. |
| `date` | today | IGRF evaluation date (`"yyyy-mm-dd"`). |
| `solver` | `"rk4"` | Integrator: `"rk4"`, `"boris"`, or `"rk45"`. |
| `atol` | `1e-3` | Absolute tolerance (RK45 only). |
| `rtol` | `1e-6` | Relative tolerance (RK45 only). |
| `escape_altitude` | `10 * RE` | Escape radius in meters. |

### Integration parameters (`get_trajectory`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | `1e-5` | Time step in seconds. |
| `max_time` | `1.` | Maximum integration time in seconds. |
| `get_data` | `False` | Return trajectory data dictionary. |
| `use_python` | `False` | Use the pure-Python fallback integrator instead of the C++ extension. Useful for debugging; slower. |

## Solver choice

For **single trajectory visualization**, use `solver="rk4"` or
`solver="boris"` with `bfield_type="igrf"`. These fixed-step solvers produce
regularly spaced output, which is better for plotting.

- **Boris** is ~30% faster than RK4 with 2x better momentum conservation.
- **RK45** uses adaptive stepping (variable output spacing) and is best
  suited for fast cutoff scanning in [GMRC evaluation](geomagnetic_cutoffs.md).

See the [Solvers guide](solvers.md) for a detailed comparison.

## Predefined locations

Kamioka, IceCube, SNOLAB, UofA, CTA-North, CTA-South, ORCA, ANTARES,
Baikal-GVD, TA.

## Example

```python
from gtracr.trajectory import Trajectory

# 20 GV proton from the west at Kamioka, using Boris pusher
traj = Trajectory(
    zenith_angle=45., azimuth_angle=90., rigidity=20.,
    location_name="Kamioka", solver="boris",
)
data = traj.get_trajectory(get_data=True)

if traj.particle_escaped:
    print("Allowed trajectory")
    print(f"  Final radius: {data['r'][-1]:.0f} m")
    print(f"  Steps: {len(data['t'])}")
```
