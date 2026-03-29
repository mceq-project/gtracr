"""
gtracr — cosmic ray trajectory simulation through Earth's geomagnetic field.

Simulates cosmic ray trajectories through Earth's geomagnetic field using
the IGRF-13 model and computes geomagnetic rigidity cutoffs (GMRC) via
Monte Carlo sampling.

Main classes
------------
Trajectory : Single cosmic ray trajectory evaluation.
GMRC : Geomagnetic rigidity cutoff map evaluation.
"""

from importlib.metadata import version

from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.trajectory import Trajectory

__version__ = version("gtracr")

__all__ = ["Trajectory", "GMRC", "__version__"]
