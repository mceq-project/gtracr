from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from gtracr.constants import KG_M_S_PER_GEVC

PLOT_DIR = Path(__file__).parent.parent.parent / "gtracr_plots"

COLOR_LIST = ["b", "r", "c", "m", "y", "g", "k"]


def plot_3dtraj(
    trajectory_datalist,
    title_name="Particle Trajectory",
    file_name="test_trajectory_3d.html",
    mpl=False,
    plotdir_path=PLOT_DIR,
    show_plot=False,
):
    """
    Plots trajectories using a 3-dimensional plot using PlotLy, an interactive HTML plotting module. Options are available to plot in usual matplotlib as well.

    Parameters
    ----------

    - trajectory_datalist : list of dict(str, np.array)
        The list of dictionaries that stores the information of the trajectory in Cartesian coordinates.
    - title_name : str
        The title name of the plot. Default is set to `"Particle Trajectory"`.
    - file_name : str
        The name of the file that we want to save. This defaults to a filename with a .html extension. *The filename should not be .html if `mpl=True`.*
    - mpl : bool
        Enables plotting with matplotlib instead of PlotLy (default = False).
    - plotdir_path : str
        The path to the directory in which the plots are stored in. Default is set to a directory `gtracr_plots` placed in parallel with the root directory.
    - show_plot : bool
        Boolean whether to show the plot or not
    """

    # unpack dictionary
    data_list = []
    max_tarr_list = []

    for i, trajectory_data in enumerate(trajectory_datalist):
        data_list.append(
            (
                trajectory_data["t"],
                trajectory_data["x"],
                trajectory_data["y"],
                trajectory_data["z"],
            )
        )
        max_tarr_list.append(np.max(trajectory_data["t"]))

    # get maximal array of time for plotting purposes
    max_tarr_index = np.argmax(max_tarr_list, axis=0)
    cbar_tarr = data_list[max_tarr_index][0]

    print(
        f"Maximal and minimal time values: {np.max(cbar_tarr):.3e}, {np.min(cbar_tarr):.3e}"
    )

    # set the earth wireframe
    u, v = np.mgrid[
        0 : 2 * np.pi : 100j, 0 : np.pi : 100j
    ]  # x in a:b:xj is number of points in [a,b]
    x_sphere = np.sin(u) * np.cos(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(u)

    # plot using matplotlib
    if mpl:
        # set the figure and axes
        fig_3d = plt.figure(figsize=(12, 6))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        # plot the sphere
        ax_3d.plot_wireframe(x_sphere, y_sphere, z_sphere, color="k")

        # plot the trajectory
        for i, (t_arr, x_arr, y_arr, z_arr) in enumerate(data_list):
            cm_3d = ax_3d.scatter(x_arr, y_arr, z_arr, c=t_arr, marker="o", s=1.0)
            if i == max_tarr_index:
                cm3d_max = cm_3d

        # labels, colorbars, limits whatnot
        cbar_3d = fig_3d.colorbar(cm3d_max, ax=ax_3d)
        ax_3d.set_xlim([-10.0, 10.0])
        ax_3d.set_ylim([-10.0, 10.0])
        ax_3d.set_zlim([-10.0, 10.0])
        ax_3d.set_xlabel(r"x [$R_E$]", fontsize=12)
        ax_3d.set_ylabel(r"y [$R_E$]", fontsize=12)
        ax_3d.set_zlabel(r"z [$R_E$]", fontsize=12)
        cbar_3d.ax.set_ylabel("Time [s]", fontsize=12)
        ax_3d.set_title(title_name, fontsize=16)

        # make file extension to png if it is not png or jpg
        if file_name.find("png") < 0 or file_name.find("jpg") < 0:
            file_name = file_name.split(".")[0] + ".png"

        # mpld3.save_html(fig_3d, Path(PLOT_DIR) / "test_trajectory_3d.html")
        plt.savefig(Path(plotdir_path) / file_name)

        if show_plot:
            plt.show()

    # plot using PlotLy
    else:
        # construct trajectory plot object
        traj_plots = []
        for i, (t_arr, x_arr, y_arr, z_arr) in enumerate(data_list):
            # marker setings for trajectory plot

            traj_colorbar = (
                dict(thickness=20, title="Time [s]") if i == max_tarr_index else None
            )
            traj_marker = dict(
                size=4,
                color=t_arr,  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.8,
                colorbar=traj_colorbar,
            )

            traj_plot = go.Scatter3d(
                x=x_arr, y=y_arr, z=z_arr, mode="markers", marker=traj_marker
            )
            traj_plots.append(traj_plot)

        # construct wireframe for sphere
        lines = []
        line_marker = dict(color="#0066FF", width=2)
        for i, j, k in zip(x_sphere, y_sphere, z_sphere):
            lines.append(go.Scatter3d(x=i, y=j, z=k, mode="lines", line=line_marker))

        # append them all and plot
        data = lines + traj_plots
        fig = go.Figure(data=data)

        # additional configurations
        # somehow latex axis labels are still not enabled with plotly?
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene_aspectmode="cube",
            scene=dict(
                xaxis=dict(nticks=6, range=[-10.0, 10.0], title=r"x [Re]"),
                yaxis=dict(nticks=6, range=[-10.0, 10.0], title=r"y [Re]"),
                zaxis=dict(nticks=6, range=[-10.0, 10.0], title=r"z [Re]"),
            ),
            font=dict(family="Courier New, monospace", size=18),
            showlegend=False,
        )

        # make file extension to html if it is not html
        if file_name.find("html") < 0:
            file_name = file_name.split(".")[0] + ".html"

        if show_plot:
            fig.show()

        # write to html
        fig.write_html(Path(plotdir_path) / file_name)


def plot_2dtraj(
    trajectory_datalist,
    dim1="x",
    dim2="y",
    title_name="Particle Trajectory",
    file_name="test_trajectory_proj.png",
    mpl=False,
    plotdir_path=PLOT_DIR,
    show_plot=False,
):
    """
    Plots the projections of the trajectory in three different planes, and the time evolution of the magnitude of the trajectory. Only supported with matplotlib for now.

    Parameters
    ----------

    - trajectory_datalist : list of dict(str, np.array)
        The list of dictionaries that stores the information of the trajectory in Cartesian coordinates.
    - dim1, dim2 : str
        The two dimensions in which we want to plot the trajectories on.
    - title_name : str
        The title name of the plot. Default is setot `"Particle Trajectory"`.
    - file_name : str
        The name of the file that we want to save. This defaults to a filename with a .html extension. *The filename should not be .html if `mpl=True`.*
    - mpl : bool
        Enables plotting with matplotlib instead of PlotLy (default = False).
    - plotdir_path : str
        The path to the directory in which the plots are stored in. Default is set to a directory `gtracr_plots` placed in parallel with the root directory.
    - show_plot : bool
        Boolean whether to show the plot or not
    """
    data_list = []
    max_tarr_list = []

    for i, trajectory_data in enumerate(trajectory_datalist):
        data_list.append(
            (trajectory_data["t"], trajectory_data[dim1], trajectory_data[dim2])
        )
        max_tarr_list.append(np.max(trajectory_data["t"]))

    # get maximal array of time for plotting purposes
    max_tarr_index = np.argmax(max_tarr_list, axis=0)
    cbar_tarr = data_list[max_tarr_index][0]

    print(
        f"Maximal and minimal time values: {np.max(cbar_tarr):.3e}, {np.min(cbar_tarr):.3e}"
    )

    fig_2d, ax_2d = plt.subplots(figsize=(12, 9), constrained_layout=True)

    for i, (t_arr, dim1_arr, dim2_arr) in enumerate(data_list):
        plot_colorbar = True if i == max_tarr_index else False
        plot_traj_projection(
            dim1_arr, dim2_arr, t_arr, fig_2d, ax_2d, dim1, dim2, plot_colorbar
        )

    # fig_proj.suptitle(title_name, fontsize=16)
    if show_plot:
        plt.show()
    # html doesnt work with latex labels and suptitle, so its deprecated for now
    # mpld3.save_html(fig_proj,
    #                 Path(PLOT_DIR) / "test_trajectory_proj.html")
    plt.savefig(Path(PLOT_DIR) / "test_trajectory_proj.png")


def plot_traj_projection(arr1, arr2, t_arr, fig, ax, label1, label2, plot_colorbar):
    """
    Plot the projection of the trajectory.

    Parameters
    ----------
    arr1 (numpy array):
        the array that goes on the x-axis
    arr2 (numpy array):
        the array that goes on the y-axis
    t_arr (numpy array):
        the time array
    fig, ax:
        matplotlib plt.Figure and plt.Axes.axes objects
    label1, label2 (str):
        the labels that indicate the identity of arr1 and arr2 resp.
    plot_colorbar (bool):
        boolean that decides whether to plot colorbar or not
    """
    cm = ax.scatter(arr1, arr2, c=t_arr)
    circ = patches.Circle(
        (0.0, 0.0), 1.0, alpha=0.8, fc="#1f77b4", linestyle="-", ec="b", lw=4.0
    )
    ax.add_patch(circ)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel(rf"{label1:s} [$R_E$]", fontsize=17)
    ax.set_ylabel(rf"{label2:s} [$R_E$]", fontsize=17)

    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=17)

    if plot_colorbar:
        cbar = fig.colorbar(cm, ax=ax)
        cbar.ax.set_ylabel("Time [s]", fontsize=17)
        cbar.ax.tick_params(axis="y", labelsize=17)
    ax.set_title(f"Trajectory projected onto {label1:s}-{label2:s} plane", fontsize=17)


def plot_traj_momentum(trajectory_data, p0, show_plot=False):
    """
    Plot the time evolution of the magnitude of the momentum.
    This is mainly used for debugging purposes as a cross-check
    to the accuracy of our trajectory simulation, as |p| should
    be constant throughout the trajectory.

    Parameters
    -----------

    - trajectory_data : dict(str, np.array)
        The dictionary that contains the momentum components of the
        trajectory.
    - p0 : float
        The initial momentum of the trajectory
    - show_plot : bool
        Decide whether to show the plot in a GUI or not (default = False).
    """
    # check momentum magnitude vs steps since |p| should be
    # constant throughout the trajectory
    t_arr = trajectory_data["t"]
    p_arr = (
        np.sqrt(
            trajectory_data["pr"] ** 2.0
            + trajectory_data["ptheta"] ** 2.0
            + trajectory_data["pphi"] ** 2.0
        )
        / KG_M_S_PER_GEVC
    )
    p_ratio = p_arr / p0

    # figure for momentum vs steps
    # for physics checking
    fig_pmag, ax_pmag = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # momentum ratio vs steps
    ax_pmag.plot(t_arr, p_ratio, color="b", marker="o", ms=3.0)
    # ax_pmag.set_ylim([0.5, 1.5])
    # ax_pmag.set_ylim([-3, 3])
    ax_pmag.set_xlabel(r"Time [s]", fontsize=14)
    ax_pmag.set_ylabel(r"$p/p_0$", fontsize=14)
    ax_pmag.set_title(
        r"Time Variation of Momentum Magnnitude Throughout Trajectory", fontsize=15
    )

    if show_plot:
        plt.show()
    plt.savefig(Path(PLOT_DIR) / "pmag_plot.png")


def plot_gmrc_scatter(
    gmrc_data,
    locname,
    plabel,
    bfield_type,
    iter_num,
    show_plot=False,
    plotdir_path=PLOT_DIR,
):
    """
    Plot the scatter plot of the geomagnetic rigidity cutoffs at the specified location and type of particle that the cosmic ray constitutes of. Currently only supports matplotlib.

    Parameters
    ----------
    - gmrc_data : dict(str, np.array)
        A dictionary that consists of the azimuthal and zenith components of the particle trajectory and its corresponding rigidity cutoff in the following order: (azimuth, zenith, rigidity cutoff)

    - locname : str
        The name of the detector location.

    - plabel : str
        The label of the particle that constitutes the cosmic ray.

    - show_plot : bool
        Decides to choose to show the generated plot or not. If True, presents the plot in a GUI window.

    - plotdir_path : str
        The path to the directory in which the plots are stored in. Default is set to a directory `gtracr_plots` placed in parallel with the root directory.

    """

    # get azimuth, zenith, and rigidity cutoff arrays
    azimuth_arr = gmrc_data["azimuth"]
    zenith_arr = gmrc_data["zenith"]
    rcutoff_arr = gmrc_data["rcutoff"]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sc = ax.scatter(azimuth_arr, zenith_arr, c=rcutoff_arr, s=3.0)
    ax.set_xlabel("Azimuthal Angle [Degrees]")
    ax.set_ylabel("Zenith Angle [Degrees]")
    ax.set_title(
        f"Geomagnetic Rigidity Cutoffs at {locname} for {plabel} with N = {iter_num}"
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel("Rigidity [GV]")

    ax.set_xlim([0.0, 360.0])
    ax.set_ylim([180.0, 0.0])

    plt.savefig(
        Path(plotdir_path) / f"{locname}_{plabel}_{bfield_type}_scatterplot.png",
        dpi=800,
    )

    if show_plot:
        plt.show()


def plot_gmrc_heatmap(
    gmrc_grids,
    rigidity_list,
    locname,
    plabel,
    bfield_type,
    show_plot=False,
    plotdir_path=PLOT_DIR,
):
    """
    Plot the heatmap of the geomagnetic rigidity cutoffs at the specified
    location and particle type.

    Works with both the legacy ``interpolate_results`` output (irregularly
    spaced grids fed to ``imshow``) and the new ``bin_results`` output
    (regular bin-centre arrays fed to ``pcolormesh`` + ``contour``).

    Parameters
    ----------
    - gmrc_grids : tuple(np.array(float))
        (azimuth_grid, zenith_grid, rcutoff_grid)

    - rigidity_list : np.array(float)
        Rigidities evaluated in the MC (for colorbar limits / contour levels).

    - locname, plabel, bfield_type : str
    - show_plot : bool
    - plotdir_path : str
    """

    (azimuth_grid, zenith_grid, rcutoff_grid) = gmrc_grids
    vmin, vmax = np.min(rigidity_list), np.max(rigidity_list)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # pcolormesh handles NaN bins gracefully (transparent), and works
    # directly with bin-centre arrays from bin_results().
    mesh = ax.pcolormesh(
        azimuth_grid,
        zenith_grid,
        rcutoff_grid,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        alpha=0.8,
    )

    # Contour lines over the heatmap.  Mask NaN so contour doesn't choke.
    masked_grid = np.ma.masked_invalid(rcutoff_grid)
    # nlevels = max(4, int(4 * len(rigidity_list) / 6))
    nlevels = [2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
    cs = ax.contour(
        azimuth_grid,
        zenith_grid,
        masked_grid,
        colors="k",
        linewidths=1.0,
        levels=nlevels,
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt="%.0f")

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Rigidity [GV]")

    # ylim from 180 to 0 to follow convention in Honda 2002 paper
    ax.set_xlim([0.0, 360.0])
    ax.set_ylim([180.0, 0.0])

    ax.set_xlabel("Azimuthal Angle [Degrees]")
    ax.set_ylabel("Zenith Angle [Degrees]")
    ax.set_title(f"Geomagnetic Rigidity Cutoffs at {locname} for {plabel}")

    if show_plot:
        plt.show()

    plt.savefig(Path(plotdir_path) / f"{locname}_{plabel}_{bfield_type}_cutoffplot.png")
