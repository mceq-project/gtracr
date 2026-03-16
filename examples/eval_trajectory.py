"""
A command-line interface to obtain the trajectory plots,
both the projections and the 3-D plot.

This should support some html format with interactive window
like PlotLy in the future.
"""

import argparse
import os

import numpy as np

from gtracr.lib.constants import EARTH_RADIUS
from gtracr.plotting import plot_2dtraj, plot_3dtraj, plot_traj_momentum
from gtracr.trajectory import Trajectory
from gtracr.utils import dec_to_dms

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

PLOT_DIR = os.path.join(PARENT_DIR, "..", "gtracr_plots")


def convert_to_cartesian(trajectory_data):
    r_arr = trajectory_data["r"] / EARTH_RADIUS
    theta_arr = trajectory_data["theta"]
    phi_arr = trajectory_data["phi"]

    # convert to cartesian & add to dict
    trajectory_data["x"] = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    trajectory_data["y"] = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
    trajectory_data["z"] = r_arr * np.cos(theta_arr)


def plot_trajectory(traj_datadict, title, check_3dtraj=False, show_plot=False):
    # # convert to cartesian coordinates
    convert_to_cartesian(traj_datadict)

    # plot the projections
    plot_2dtraj([traj_datadict], plotdir_path=PLOT_DIR)

    # plot the 3-d trajectory with wireframe sphere as the earth
    if check_3dtraj:
        plot_3dtraj([traj_datadict], title_name=title, plotdir_path=PLOT_DIR)


def get_trajectory(solver="rk4", field_mode="igrf"):
    # set parameters

    # parameters for trajectory
    # particle is assumed to be proton
    q = 1
    plabel = "p+"

    # initial momentum
    p0 = 30.0
    rigidity = p0 / np.abs(q)  # convert to rigidity

    # location of detector
    lat = 10.0
    lng = 40.0
    detector_alt = 0.0

    # 3-vector of particle
    zenith = 90.0
    azimuth = 0.0
    particle_alt = 100.0

    # set integration parameters
    dt = 1e-5
    max_time = 1.0
    max_step = 10000

    # control variables for the code
    check_pmag = True  # if we want to check the momentum magnitude
    check_3dtraj = True  # if we want to check the 3d trajectory or not
    show_plot = False  # if we want to show the plot on some GUI or not

    # first create plot directory if it doesnt exist
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    # "igrf" maps to bfield_type[0]='i'; "table" maps to 't' — both use the
    # C++ TrajectoryTracer; "table" additionally builds the 3-D lookup table.
    bfield_type = "table" if field_mode == "table" else "igrf"

    # initialize trajectory
    traj = Trajectory(
        plabel=plabel,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=particle_alt,
        latitude=lat,
        longitude=lng,
        detector_altitude=detector_alt,
        rigidity=rigidity,
        bfield_type=bfield_type,
        solver=solver,
    )

    if field_mode == "table":
        print("Building IGRF lookup table (64×128×256 grid)…", flush=True)

    traj_datadict = traj.get_trajectory(
        dt=dt, max_time=max_time, get_data=True, max_step=max_step, use_python=False
    )
    if field_mode == "table":
        print("Done.", flush=True)

    print(
        f"particle_escaped = {traj.particle_escaped}  steps = {len(traj_datadict['t'])}"
    )

    # convert lat, long in decimal notation to dms
    lat_dms, lng_dms = dec_to_dms(lat, lng)

    title = (
        f"Particle Trajectory at {lat_dms:s}, {lng_dms:s} with Zenith Angle {zenith:.1f}°,"
        f"\n Azimuth Angle {azimuth:.1f}° and Rigidity R = {rigidity:.1f}GV"
        f" [field: {field_mode:s}]"
    )

    # get momentum only if check_pmag is true
    if check_pmag:
        plot_traj_momentum(traj_datadict, p0, show_plot)

    # plot the trajectory
    plot_trajectory(traj_datadict, title, check_3dtraj, show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and plot a single cosmic ray trajectory."
    )
    parser.add_argument(
        "--solver",
        default="rk4",
        choices=["rk4", "boris", "rk45"],
        help="Integration method: rk4 (frozen-field RK4, default), boris, or rk45 (adaptive).",
    )
    parser.add_argument(
        "--field-mode",
        dest="field_mode",
        default="igrf",
        choices=["igrf", "table"],
        help="Field evaluation mode: igrf (direct IGRF, default) or "
        "table (precomputed 3-D lookup table with trilinear interpolation).",
    )
    args = parser.parse_args()
    get_trajectory(solver=args.solver, field_mode=args.field_mode)
