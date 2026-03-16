"""
Evaluate geomagnetic rigidity cutoffs (GMRC) for a location via Monte Carlo
and produce scatter and heatmap plots.
"""

import argparse
import pickle
from pathlib import Path

from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.plotting import plot_gmrc_heatmap, plot_gmrc_scatter
from gtracr.utils import location_dict

PLOT_DIR = Path(__file__).parent.parent.parent / "gtracr_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def export_as_pkl(fpath, ds):
    with open(fpath, "wb") as f:
        pickle.dump(ds, f, protocol=-1)


def _run_gmrc(gmrc, args):
    """Evaluate gmrc, then plot."""
    plabel = args.particle
    ngrid_azimuth = 360
    ngrid_zenith = 180
    locname = gmrc.location

    if args.eval_mode == "batch":
        gmrc.evaluate_batch(dt=args.dt, max_time=args.max_time)
    else:
        gmrc.evaluate(dt=args.dt, max_time=args.max_time)

    print("Plotting...")

    if args.eval_mode == "batch":
        gmrc_grids = gmrc.bin_results(
            nbins_azimuth=ngrid_azimuth,
            nbins_zenith=ngrid_zenith,
        )
    else:
        plot_gmrc_scatter(
            gmrc.data_dict,
            locname,
            plabel,
            bfield_type=args.bfield_type,
            iter_num=args.iter_num,
            show_plot=args.show_plot,
        )
        gmrc_grids = gmrc.interpolate_results(
            ngrid_azimuth=ngrid_azimuth,
            ngrid_zenith=ngrid_zenith,
        )

    plot_gmrc_heatmap(
        gmrc_grids,
        gmrc.rigidity_list,
        locname=locname,
        plabel=plabel,
        bfield_type=args.bfield_type,
        show_plot=args.show_plot,
    )

    print("Done.")


def eval_gmrc(args):
    locations = list(location_dict.keys()) if args.eval_all else [args.location]

    for locname in locations:
        gmrc = GMRC(
            location=locname,
            iter_num=args.iter_num,
            bfield_type=args.bfield_type,
            particle_type=args.particle,
            n_workers=args.n_workers,
            solver=args.solver,
        )
        _run_gmrc(gmrc, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate geomagnetic cutoff rigidities via Monte Carlo "
            "and produce heatmap plots."
        )
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
        "--iter-num",
        dest="iter_num",
        default=50000,
        type=int,
        help="Number of Monte Carlo iterations (default: 50000).",
    )
    parser.add_argument(
        "--bfield-type",
        dest="bfield_type",
        default="igrf",
        choices=["igrf", "dipole", "table"],
        help="Magnetic field model (default: igrf).",
    )
    parser.add_argument(
        "--solver",
        default="rk4",
        choices=["rk4", "boris", "rk45"],
        help="Integration method: rk4 (default), boris, rk45 (adaptive).",
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
        "--eval-mode",
        dest="eval_mode",
        default="batch",
        choices=["batch", "legacy"],
        help="Evaluation mode: batch (C++ MC loop, default) or legacy (Python loop).",
    )
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        default=None,
        type=int,
        help="Number of parallel workers (default: all CPU cores).",
    )
    parser.add_argument(
        "--all",
        dest="eval_all",
        action="store_true",
        help="Evaluate GMRC for all pre-defined locations.",
    )
    parser.add_argument(
        "--show-plot",
        dest="show_plot",
        action="store_true",
        help="Show plots in an interactive window.",
    )
    args = parser.parse_args()
    eval_gmrc(args)
