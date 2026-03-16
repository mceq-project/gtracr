"""
Evaluate and plot a single cosmic ray trajectory.

Supports interactive display or file output.
"""

import argparse
from pathlib import Path

import numpy as np

from gtracr.constants import EARTH_RADIUS
from gtracr.plotting import plot_2dtraj, plot_3dtraj, plot_traj_momentum
from gtracr.trajectory import Trajectory
from gtracr.utils import dec_to_dms

PLOT_DIR = Path(__file__).parent.parent.parent / "gtracr_plots"


def convert_to_cartesian(trajectory_data):
    r_arr = trajectory_data["r"] / EARTH_RADIUS
    theta_arr = trajectory_data["theta"]
    phi_arr = trajectory_data["phi"]

    trajectory_data["x"] = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    trajectory_data["y"] = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
    trajectory_data["z"] = r_arr * np.cos(theta_arr)


def plot_trajectory(traj_datadict, title, check_3dtraj=False, show_plot=False):
    convert_to_cartesian(traj_datadict)
    plot_2dtraj([traj_datadict], plotdir_path=str(PLOT_DIR))
    if check_3dtraj:
        plot_3dtraj([traj_datadict], title_name=title, plotdir_path=str(PLOT_DIR))


def run(args):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    traj = Trajectory(
        plabel=args.particle,
        location_name=args.location,
        zenith_angle=args.zenith,
        azimuth_angle=args.azimuth,
        rigidity=args.rigidity,
        bfield_type=args.bfield_type,
        solver=args.solver,
    )

    if args.bfield_type == "table":
        print("Building IGRF lookup table (64×128×256 grid)…", flush=True)

    traj_datadict = traj.get_trajectory(
        dt=args.dt, max_time=args.max_time, get_data=True, use_python=False
    )

    if args.bfield_type == "table":
        print("Done.", flush=True)

    steps = len(traj_datadict["t"])
    print(f"particle_escaped = {traj.particle_escaped}  steps = {steps}")

    lat = traj.latitude
    lng = traj.longitude
    lat_dms, lng_dms = dec_to_dms(lat, lng)

    title = (
        f"Particle Trajectory at {lat_dms:s}, {lng_dms:s} "
        f"with Zenith {args.zenith:.1f}°, Azimuth {args.azimuth:.1f}°, "
        f"R = {args.rigidity:.1f} GV [{args.bfield_type}]"
    )

    plot_traj_momentum(traj_datadict, args.rigidity, args.show_plot)
    plot_trajectory(traj_datadict, title, check_3dtraj=True, show_plot=args.show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and plot a single cosmic ray trajectory."
    )
    parser.add_argument(
        "--location",
        default="Kamioka",
        help="Detector location name (default: Kamioka).",
    )
    parser.add_argument(
        "--particle",
        default="p+",
        choices=["p+", "p-", "e+", "e-"],
        help="Particle label (default: p+).",
    )
    parser.add_argument(
        "--rigidity",
        type=float,
        default=10.0,
        help="Particle rigidity in GV (default: 10.0).",
    )
    parser.add_argument(
        "--zenith",
        type=float,
        default=0.0,
        help="Zenith angle in degrees (default: 0.0).",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Azimuth angle in degrees (default: 0.0).",
    )
    parser.add_argument(
        "--solver",
        default="rk4",
        choices=["rk4", "boris", "rk45"],
        help="Integration method: rk4 (default), boris, or rk45 (adaptive).",
    )
    parser.add_argument(
        "--bfield-type",
        dest="bfield_type",
        default="igrf",
        choices=["igrf", "dipole", "table"],
        help="Magnetic field model (default: igrf).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-5,
        help="Integration step size in seconds (default: 1e-5).",
    )
    parser.add_argument(
        "--max-time",
        dest="max_time",
        type=float,
        default=1.0,
        help="Maximum integration time in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--show-plot",
        dest="show_plot",
        action="store_true",
        help="Show plots in an interactive window.",
    )
    args = parser.parse_args()
    run(args)
