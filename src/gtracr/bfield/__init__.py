"""
Magnetic field models for Earth's geomagnetic field.

Classes
-------
MagneticField : ideal dipole approximation.
IGRF14 : IGRF-14 spherical harmonic model (degree 13, 1900–2030).
IGRF13 : deprecated alias of IGRF14.
IGRFTable : pre-computed 3-D lookup table with trilinear interpolation.
"""

from gtracr.bfield.dipole import MagneticField
from gtracr.bfield.igrf import IGRF13, IGRF14
from gtracr.bfield.table import IGRFTable

__all__ = ["MagneticField", "IGRF14", "IGRF13", "IGRFTable"]
