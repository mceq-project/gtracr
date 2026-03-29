"""Physical constants and unit conversions for gtracr."""

import numpy as np
import scipy.constants as sc

SPEED_OF_LIGHT = sc.c  # speed of light in vacuum (m/s)

ELEMENTARY_CHARGE = 1.602e-19  # elementary charge (coulombs)

EARTH_RADIUS = 6371.2 * (1e3)  # Earth radius (meters)
G10 = 29404.8 * (1e-9)  # IGRF-2020 first dipole coefficient (Tesla)

RAD_PER_DEG = np.pi / 180.0
DEG_PER_RAD = 180.0 / np.pi

KG_PER_GEVC2 = 1.78e-27  # kg per GeV/c²
KG_M_S_PER_GEVC = 5.36e-19  # kg·m/s per GeV/c

# Solver name → C++ single-character mapping (shared by Trajectory and GMRC)
SOLVER_CHARS: dict[str, str] = {"rk4": "r", "boris": "b", "rk45": "a"}
