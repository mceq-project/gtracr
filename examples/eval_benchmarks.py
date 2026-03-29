"""
Evaluates the benchmarks between different versions of the code.
"""

import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gtracr.trajectory import Trajectory

DATA_DIR = Path(__file__).parent.parent / "src" / "gtracr" / "data"
PLOT_DIR = Path(__file__).parent.parent.parent / "gtracr_plots"


def read_pkl(fpath):
    with open(fpath, "rb") as f:
        benchmark_data = pickle.load(f)
    return benchmark_data


def write_pkl(fpath, datadict):
    with open(fpath, "wb") as f:
        pickle.dump(datadict, f, protocol=-1)


def get_evaltime(iter_num, initial_variables, bfield_type, use_python=False):
    """
    Evaluate iter_num trajectories and return the average wall-clock time.

    Parameters
    ----------
    iter_num : int
        Number of iterations to evaluate.
    initial_variables : tuple
        (plabel, zenith, azimuth, part_alt, lat, lng, dec_alt, rig)
    bfield_type : str
        Magnetic field model to use ('igrf', 'dipole', or 'table').
    use_python : bool
        Use pure-Python integrator instead of C++ (default False).
    """
    plabel, zenith, azimuth, part_alt, lat, lng, dec_alt, rig = initial_variables

    trajectory = Trajectory(
        plabel=plabel,
        latitude=lat,
        longitude=lng,
        detector_altitude=dec_alt,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=part_alt,
        rigidity=rig,
        bfield_type=bfield_type,
    )

    eval_time = 0.0
    dt = 1e-5
    max_step = 10000

    for _ in range(int(np.floor(iter_num))):
        start_time = time.perf_counter()
        trajectory.get_trajectory(dt=dt, max_step=max_step, use_python=use_python)
        stop_time = time.perf_counter()
        eval_time += stop_time - start_time

    return eval_time / iter_num


def get_evaltime_data(iternum_list):
    """
    Evaluate the average trajectory time for each iteration count in the list.

    Parameters
    ----------
    iternum_list : list
        List of iteration counts to benchmark.
    """
    initial_variables = ("p+", 20.0, -30.0, 100.0, 0.0, 0.0, 0.0, 40.0)

    avg_evaltime_dict = {
        "cppdip_vec": {
            "values": np.zeros(len(iternum_list)),
            "label": "C++, Dipole",
        },
        "cppigrf_vec": {
            "values": np.zeros(len(iternum_list)),
            "label": "C++, IGRF",
        },
    }

    for i, iter_num in enumerate(iternum_list):
        avg_evaltime_dict["cppdip_vec"]["values"][i] = get_evaltime(
            iter_num, initial_variables, bfield_type="dipole"
        )
        avg_evaltime_dict["cppigrf_vec"]["values"][i] = get_evaltime(
            iter_num, initial_variables, bfield_type="igrf"
        )
        print(f"Finished benchmarking with {int(np.floor(iter_num))} iterations.")

    return avg_evaltime_dict


def plot_benchmarks(benchmark_data, iternum_list):
    """Plot benchmark results for each iteration count."""
    color_arr = ["b", "m", "g", "c", "r", "y"]

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

    for i, (code_type, avg_evaltime_dict) in enumerate(
        sorted(list(benchmark_data.items()))
    ):
        ax.loglog(
            iternum_list,
            avg_evaltime_dict["values"],
            label=avg_evaltime_dict["label"],
            color=color_arr[i],
            marker="o",
            ms=3.0,
        )

    ax.set_xlabel("Number of Iterations", fontsize=14)
    ax.set_ylabel("Average Evaluation Time [s]", fontsize=14)
    ax.set_title("Performance Benchmarks for Trajectory Evaluations", fontsize=16)
    ax.legend()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / "benchmark_plot.png")


if __name__ == "__main__":
    initial_variables = ("p+", 20.0, 25.0, 100.0, 0.0, 0.0, 0.0, 10.0)

    iternum_list = np.logspace(0, np.log10(3000), 10)

    reset = True

    fpath = DATA_DIR / "benchmark_data.pkl"
    if fpath.exists() and not reset:
        benchmark_data = read_pkl(fpath)
    else:
        benchmark_data = get_evaltime_data(iternum_list)
        write_pkl(fpath, benchmark_data)

    plot_benchmarks(benchmark_data, iternum_list)
