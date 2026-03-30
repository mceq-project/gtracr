"""
Benchmark individual trajectory solvers at fixed angle/rigidity.
Table mode pre-loads the IGRF table once to measure integration time only.
"""
import time
from pathlib import Path

import numpy as np

from gtracr._libgtracr import TrajectoryTracer as CppTT
from gtracr.constants import ELEMENTARY_CHARGE, KG_PER_GEVC2
from gtracr.geomagnetic_cutoffs import _get_or_generate_igrf_table
from gtracr.trajectory import Trajectory

N_REPEAT = 500

# Fixed trajectory parameters
TRAJ_KWARGS = dict(
    zenith_angle=45., azimuth_angle=90., rigidity=7.,
    location_name="Kamioka", bfield_type="igrf",
)
DT = 1e-5
MAX_TIME = 1.0
MAX_STEP = int(np.ceil(MAX_TIME / DT))

# Build initial sixvector from Trajectory (field type doesn't matter for coords)
_ref = Trajectory(**TRAJ_KWARGS)
CHARGE_SI = _ref.charge * ELEMENTARY_CHARGE
MASS_SI   = _ref.mass * KG_PER_GEVC2
START_ALT = _ref.start_alt
ESC_ALT   = _ref.esc_alt
VEC0      = _ref.particle_sixvector

IGRF_PARAMS = _ref.igrf_params  # (data_path, dec_date) tuple

# Pre-load table once
print("Pre-loading IGRF table...")
shared_table, _table_params = _get_or_generate_igrf_table(*IGRF_PARAMS)
print("Ready.\n")

solvers      = ["rk4",  "boris", "rk45"]
bfields      = ["igrf", "dipole", "table"]
solver_chars = {"rk4": "r", "boris": "b", "rk45": "a"}
bfield_chars = {"igrf": "i", "dipole": "d", "table": "t"}

results = {}

for bfield in bfields:
    for solver in solvers:
        key = f"{solver}+{bfield}"
        bc = bfield_chars[bfield]
        sc = solver_chars[solver]

        def make_tracer():
            if bfield == "table":
                return CppTT(shared_table, _table_params,
                             CHARGE_SI, MASS_SI, START_ALT, ESC_ALT,
                             DT, MAX_STEP, IGRF_PARAMS, sc)
            else:
                return CppTT(CHARGE_SI, MASS_SI, START_ALT, ESC_ALT,
                             DT, MAX_STEP, bc, IGRF_PARAMS, sc)

        # One run to get step count
        tt = make_tracer()
        data = tt.evaluate_and_get_trajectory(0.0, VEC0)
        n_steps = len(data["t"])
        escaped = tt.particle_escaped

        # Timing loop — reuse same tracer to avoid measuring table copy overhead
        tt_bench = make_tracer()
        t0 = time.monotonic()
        for _ in range(N_REPEAT):
            tt_bench.reset()
            tt_bench.evaluate(0.0, VEC0)
        elapsed = time.monotonic() - t0
        evals_per_s = N_REPEAT / elapsed

        results[key] = {"n_steps": n_steps, "evals_per_s": evals_per_s, "escaped": escaped}
        print(f"{key:20s}  steps={n_steps:6d}  escaped={escaped}  {evals_per_s:.1f} eval/s")

baseline = results["rk4+igrf"]["evals_per_s"]
print("\n--- Ratios to rk4+igrf ---")
for key, v in results.items():
    print(f"{key:20s}  {v['evals_per_s'] / baseline:.2f}x")
