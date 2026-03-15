import sys
import os
import numpy as np
import pickle
import argparse

from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.utils import location_dict
from gtracr.plotting import plot_gmrc_scatter, plot_gmrc_heatmap

# add filepath of gtracr to sys.path
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PLOT_DIR = os.path.join(PARENT_DIR, "..", "gtracr_plots")

# create directory if gtracr_plots dir does not exist
if not os.path.isdir(PLOT_DIR):
    os.mkdir(PLOT_DIR)


def export_as_pkl(fpath, ds):
    with open(fpath, "wb") as f:
        pickle.dump(ds, f, protocol=-1)


def eval_gmrc(args):
    # create particle trajectory with desired particle and energy
    plabel = "p+"
    particle_altitude = 100.

    # number of points for azimuth / zenith grid
    ngrid_azimuth = 360
    ngrid_zenith = 180

    # change initial parameters if debug mode is set
    if args.debug_mode:
        args.iter_num = 10
        args.show_plot = True

    if args.eval_all:
        for locname in list(location_dict.keys()):
            # evaluate rigidity cutoffs at the location for
            # the specified particle
            gmrc = GMRC(location=locname,
                        iter_num=args.iter_num,
                        particle_altitude=particle_altitude,
                        bfield_type=args.bfield_type,
                        particle_type=plabel,
                        n_workers=args.n_workers)

            gmrc.evaluate()

            # create a debugger / checker as a scatter plot
            # of the dataset
            # if args.debug_mode:
            plot_gmrc_scatter(gmrc.data_dict,
                              locname,
                              plabel,
                              bfield_type=args.bfield_type,
                              iter_num=args.iter_num,
                              show_plot=args.show_plot)

            interpd_gmrc_data = gmrc.interpolate_results(
                ngrid_azimuth=ngrid_azimuth,
                ngrid_zenith=ngrid_zenith,
            )

            plot_gmrc_heatmap(interpd_gmrc_data,
                              gmrc.rigidity_list,
                              locname=locname,
                              plabel=plabel,
                              bfield_type=args.bfield_type,
                              show_plot=args.show_plot)

    else:
        # evaluate rigidity cutoffs at the location for
        # the specified particle
        gmrc = GMRC(location=args.locname,
                    iter_num=args.iter_num,
                    particle_altitude=particle_altitude,
                    bfield_type=args.bfield_type,
                    particle_type=plabel,
                    n_workers=args.n_workers)

        gmrc.evaluate()

        # create a debugger / checker as a scatter plot
        # of the dataset
        # if args.debug_mode:
        plot_gmrc_scatter(gmrc.data_dict,
                          args.locname,
                          plabel,
                          bfield_type=args.bfield_type,
                          iter_num=args.iter_num,
                          show_plot=args.show_plot)

        interpd_gmrc_data = gmrc.interpolate_results(
            ngrid_azimuth=ngrid_azimuth,
            ngrid_zenith=ngrid_zenith,
        )

        plot_gmrc_heatmap(interpd_gmrc_data,
                          gmrc.rigidity_list,
                          locname=args.locname,
                          plabel=plabel,
                          bfield_type=args.bfield_type,
                          show_plot=args.show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Evaluates the geomagnetic cutoff rigidities of some location for N iterations using a Monte-Carlo sampling scheme, and produces a heatmap for such geomagnetic cutoff rigidities.'
    )
    parser.add_argument('-ln',
                        '--locname',
                        dest="locname",
                        default="Kamioka",
                        type=str,
                        help="Detector location to evaluate GM cutoffs.")
    parser.add_argument('-n',
                        '--iter_num',
                        dest="iter_num",
                        default=50000,
                        type=int,
                        help="Number of iterations for Monte-Carlo.")
    parser.add_argument('-bf',
                        '--bfield',
                        dest="bfield_type",
                        default="igrf",
                        type=str,
                        help="The geomagnetic field model used.")
    parser.add_argument('-a',
                        '--all',
                        dest="eval_all",
                        action="store_true",
                        help="Evaluate GM cutoffs for all locations.")
    parser.add_argument('--show',
                        dest="show_plot",
                        action="store_true",
                        help="Show the plot in an external display.")
    parser.add_argument(
        '-d',
        '--debug',
        dest="debug_mode",
        action="store_true",
        help="Enable debug mode. Sets N = 10 and enable --show=True.")
    parser.add_argument(
        '-w',
        '--workers',
        dest="n_workers",
        default=None,
        type=int,
        help="Number of parallel worker processes (default: physical core count).")
    # parser.add_argument('-c',
    #                     '--clean',
    #                     dest="clean_dict",
    #                     action="store_true",
    #                     help='Clean the dataset. ')

    args = parser.parse_args()
    eval_gmrc(args)
