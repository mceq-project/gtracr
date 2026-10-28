"""Physical constants and unit conversions for gtracr."""

import numpy as np
import scipy.constants as sc

SPEED_OF_LIGHT = sc.c  # speed of light in vacuum (m/s)

ELEMENTARY_CHARGE = sc.e  # elementary charge (coulombs, SI exact)

EARTH_RADIUS = 6371.2 * (1e3)  # Earth radius (meters)
G10 = 29404.8 * (1e-9)  # IGRF-2020 first dipole coefficient (Tesla)

RAD_PER_DEG = np.pi / 180.0
DEG_PER_RAD = 180.0 / np.pi

KG_PER_GEVC2 = 1e9 * sc.e / sc.c**2  # kg per GeV/c² (exact in SI)
KG_M_S_PER_GEVC = 1e9 * sc.e / sc.c  # kg·m/s per GeV/c (exact in SI)

# Lower bound on |sin(theta)| in the spherical equations of motion (polar axis)
MIN_SIN_THETA = 1e-10

# IGRF coefficient files in gtracr/data (14th generation)
IGRF_GENERATION = 14
IGRF_JSON_FILE = "igrf14.json"  # read by the C++ IGRF class
IGRF_SHC_FILE = "IGRF14.shc"  # read by gtracr.bfield.IGRF14

# Solver name → C++ single-character mapping (shared by Trajectory and GMRC)
SOLVER_CHARS: dict[str, str] = {"rk4": "r", "boris": "b", "rk45": "a"}
