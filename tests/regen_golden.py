"""Regenerate golden values for test_numerical_regression.py and test_trajectories.py.

Run this script whenever the C++ integrator or IGRF coefficients change and
the golden values need to be updated. It prints ready-to-paste replacements
for EXPECTED_IGRF_SIXVEC, EXPECTED_IGRF_ESCAPED, and expected_times.

The date is pinned to match IGRF_DATE in test_numerical_regression.py so
values are stable across calendar dates.

Usage
-----
  python tests/regen_golden.py
"""

from gtracr.trajectory import Trajectory

DATE = "2026-03-01"  # must match IGRF_DATE in test_numerical_regression.py

INITIAL_VARIABLES = [
    ("p+", 90.0, 90.0, 100.0, 0.0, 0.0, 0.0, 30.0, None),
    ("p+", 120.0, 90.0, 100.0, 0.0, 0.0, -1.0, 30.0, None),
    ("p+", 0.0, 25.0, 100.0, 50.0, 100.0, 0.0, 50.0, None),
    ("p+", 90.0, 5.0, 100.0, 89.0, 20.0, 0.0, 20.0, None),
    ("p+", 90.0, 5.0, 100.0, -90.0, 20.0, 0.0, 20.0, None),
    ("e-", 90.0, 5.0, 100.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 265.0, 0.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 180.0, 10.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 0.0, 0.0, 89.0, 0.0, 0.0, 20.0, None),
    ("p+", 45.0, 0.0, 0.0, 0.0, 180.0, 100.0, 20.0, None),
    ("p+", 45.0, 0.0, 0.0, 0.0, 180.0, 100.0, 5.0, None),
    ("p+", 45.0, 0.0, 0.0, 0.0, 180.0, 100.0, None, 10.0),
    ("p+", 9.0, 80.0, 0.0, 50.0, 260.0, 100.0, None, 50.0),
]

print(f"Regenerating golden values with date={DATE!r} ...\n")

sixvecs = []
escaped_flags = []
final_times = []

for plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en in INITIAL_VARIABLES:
    traj = Trajectory(
        plabel=plabel,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=palt,
        latitude=lat,
        longitude=lng,
        detector_altitude=dalt,
        rigidity=rig,
        energy=en,
        bfield_type="igrf",
        date=DATE,
    )
    traj.get_trajectory(dt=1e-5, max_time=1.0)
    sixvecs.append(traj.final_sixvector.tolist())
    escaped_flags.append(traj.particle_escaped)
    final_times.append(traj.final_time)

print("# Paste into tests/test_numerical_regression.py")
print("EXPECTED_IGRF_SIXVEC = [")
for v in sixvecs:
    print(f"    {v},")
print("]")
print()
print("EXPECTED_IGRF_ESCAPED = [")
for e in escaped_flags:
    print(f"    {e},")
print("]")
print()
print("# Paste the forbidden-trajectory times into tests/test_trajectories.py")
print("# (escaped cases are skipped in test_trajectories_igrf, so only these matter)")
for i, (t, escaped) in enumerate(zip(final_times, escaped_flags)):
    status = "escaped — skip" if escaped else "FORBIDDEN — update expected_times"
    print(f"  [{i}] {t!r}  # {status}")
print()
print(f"expected_times = {final_times!r}")
