"""
gtracr — cosmic ray trajectory simulation through Earth's geomagnetic field.

Simulates cosmic ray trajectories through Earth's geomagnetic field using
the IGRF-13 model and computes geomagnetic rigidity cutoffs (GMRC) via
Monte Carlo sampling.

Main classes
------------
Trajectory : Single cosmic ray trajectory evaluation.
GMRC : Geomagnetic rigidity cutoff map evaluation.
MuonTracer : Batch atmospheric-muon transport with decay tallies
    ("mutracr"; see gtracr.mutracr).
"""

from importlib.metadata import version

from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.mutracr import MuonTracer
from gtracr.trajectory import Trajectory

__version__ = version("gtracr")

__all__ = ["Trajectory", "GMRC", "MuonTracer", "__version__"]
