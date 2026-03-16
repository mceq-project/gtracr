"""
Magnetic field models for Earth's geomagnetic field.

Classes
-------
MagneticField : ideal dipole approximation.
IGRF13 : IGRF-13 spherical harmonic model (degree 13, 1900–2025).
IGRFTable : pre-computed 3-D lookup table with trilinear interpolation.
"""

from gtracr.bfield.dipole import MagneticField
from gtracr.bfield.igrf import IGRF13
from gtracr.bfield.table import IGRFTable

__all__ = ["MagneticField", "IGRF13", "IGRFTable"]
