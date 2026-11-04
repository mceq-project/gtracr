# Conventions

## Arrival direction

`zenith_angle` and `azimuth_angle` give the **arrival direction**: the point
on the sky the particle comes from, seen from the detector.

- Zenith: 0 = from overhead, 90 = horizontal, > 90 = up-going (the particle
  crossed the Earth).
- Azimuth: measured from **geographic north, clockwise** seen from above:
  **0 = from north, 90 = from east, 180 = from south, 270 = from west**.
  This is the astropy `AltAz` convention and the one of MCEq's
  location-centred atmospheres. In the local tangent frame x = east,
  y = north, z = up the unit vector toward the source is
  `(sin z sin a, sin z cos a, cos z)`.

For positive particles the east–west effect makes cutoffs from the east
(azimuth 90) higher than from the west (270).

### Converting from other conventions

| Source | Its azimuth `b` | gtracr azimuth |
|--------|-----------------|----------------|
| gtracr < 3.0 | 0 = from south, 90 = from east (Honda) | `(180 - b) % 360` |
| Honda / HKKM flux tables | 0 = from south, 90 = from east | `(180 - b) % 360` |
| CORSIKA `PHI` | momentum azimuth, counter-clockwise from magnetic north (x = N, y = W) | `(180 - b) % 360`, then add the magnetic declination for geographic north |

East (90) and west (270) are the same in all of these; north and south swap.

## Units and constants

- Rigidity R in GV, R = p c / (|Z| e); momentum p in GeV/c; `energy` is the
  total energy in GeV.
- SI conversions use the exact SI values: e = 1.602176634e-19 C,
  c = 299792458 m/s, 1 GeV/c = 1e9 e / c kg m/s, 1 GeV/c^2 = 1e9 e / c^2 kg.
- Earth radius 6371.2 km (the IGRF reference radius); altitudes are above
  this sphere.
