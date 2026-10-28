"""
Benchmark MuonTracer ("mutracr") — C++ engine vs the vendored numpy
reference, across batch sizes, thread counts and field/deposit modes.

The workload mimics the nu3d stage-C closure gates: muons injected
between 5 and 30 km with 0-70 deg tilts and 0.2-20 GeV kinetic energy,
traced through the US-Std atmosphere to ground/decay with expected-
decay deposits.

Run:  python examples/eval_mutracr_benchmark.py [--full]
"""

import argparse
import time

import numpy as np

from gtracr import _mu_reference as ref
from gtracr.mutracr import M_MU, MuonTracer

R_G = 6.391e8  # cm (MCEq ground radius, the nu3d convention)


def make_batch(n, seed=42):
    rng = np.random.default_rng(seed)
    h = rng.uniform(5e5, 3e6, n)
    tilt = np.radians(rng.uniform(0, 70, n))
    az = rng.uniform(0, 2 * np.pi, n)
    lat = np.arccos(rng.uniform(-0.9, 0.9, n))
    lon = rng.uniform(0, 2 * np.pi, n)
    r = R_G + h
    x = np.stack(
        [r * np.sin(lat) * np.cos(lon), r * np.sin(lat) * np.sin(lon), r * np.cos(lat)],
        1,
    )
    rhat = x / r[:, None]
    a = np.zeros_like(x)
    a[:, 0] = 1.0
    b = np.cross(rhat, a)
    nb = np.linalg.norm(b, axis=1)
    m = nb < 1e-8
    a[m, 0], a[m, 1] = 0.0, 1.0
    b[m] = np.cross(rhat[m], a[m])
    e1 = b / np.linalg.norm(b, axis=1)[:, None]
    e2 = np.cross(rhat, e1)
    d = -np.cos(tilt)[:, None] * rhat + np.sin(tilt)[:, None] * (
        np.cos(az)[:, None] * e1 + np.sin(az)[:, None] * e2
    )
    ekin = np.exp(rng.uniform(np.log(0.2), np.log(20), n))
    pm = np.sqrt((ekin + M_MU) ** 2 - M_MU**2)
    return x, pm[:, None] * d, rng.choice([-1, 1], n), rng.uniform(-1, 1, n), np.ones(n)


def bench_numpy(n, deposit):
    x, p, q, pol, w = make_batch(n)
    batch = ref.MuonBatch(x, p, q, pol=pol, w=w)
    bank = ref.DecayBank() if deposit else None
    t0 = time.perf_counter()
    d = ref.trace(
        batch,
        ref.AxialDipole(),
        ref.USStdAtmosphere(),
        deposit=bank,
        r_ground=R_G,
        e_min=0.115,
        w_min=0.0,
    )
    dt = time.perf_counter() - t0
    return dt, d["steps"]


def bench_cpp(n, n_threads, bfield, deposit):
    x, p, q, pol, w = make_batch(n)
    mt = MuonTracer(bfield=bfield, atmosphere="usstd", r_ground=R_G)
    dep = None
    if deposit == "bank":
        dep = dict(kind="bank")
    elif deposit == "spectrum":
        dep = dict(kind="spectrum", e_edges=np.geomspace(0.13, 41.0, 193))
    t0 = time.perf_counter()
    res = mt.trace(
        x, p, q, pol=pol, w=w, e_min=0.115, w_min=0.0, n_threads=n_threads, deposit=dep
    )
    dt = time.perf_counter() - t0
    return dt, res["steps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true", help="larger batches + numpy up to 100k"
    )
    args = ap.parse_args()

    sizes_np = [1000, 10000, 100000] if args.full else [1000, 10000]
    sizes_cpp = [1000, 10000, 100000, 1000000] if args.full else [1000, 10000, 100000]

    print(f"{'engine':<26}{'N rays':>10}{'threads':>9}{'time [s]':>11}{'rays/s':>12}")
    print("-" * 68)
    for n in sizes_np:
        dt, _ = bench_numpy(n, deposit=True)
        print(
            f"{'numpy reference (bank)':<26}{n:>10}{'-':>9}{dt:>11.3f}{n / dt:>12.0f}"
        )
    for n in sizes_cpp:
        for nt in (1, 0):
            for dep in ("none", "spectrum", "bank"):
                dt, _ = bench_cpp(n, nt, "dipole", dep)
                label = f"cpp dipole ({dep})"
                nts = "all" if nt == 0 else str(nt)
                print(f"{label:<26}{n:>10}{nts:>9}{dt:>11.3f}{n / dt:>12.0f}")
        dt, _ = bench_cpp(n, 0, "shell", "spectrum")
        print(
            f"{'cpp shell-IGRF (spectrum)':<26}{n:>10}{'all':>9}{dt:>11.3f}{n / dt:>12.0f}"
        )


if __name__ == "__main__":
    main()
