"""
Library that contain relavant constants for this package
"""

import numpy as np
import scipy.constants as sc

SPEED_OF_LIGHT = sc.c  # speed of light in vacuum

ELEMENTARY_CHARGE = 1.602e-19  # elementary charge (in coulombs)

EARTH_RADIUS = 6371.2 * (1e3)  # earth radius (meters)
G10 = 29404.8 * (1e-9)  # B-field parameter from IGRF 2020 (in Teslas)

RAD_PER_DEG = np.pi / 180.0
DEG_PER_RAD = 180.0 / np.pi

KG_PER_GEVC2 = 1.78e-27  # conversion factor between kg and GeV/c^2
# conversion factor between kg m/s (momentum SI units) and GeV/c
KG_M_S_PER_GEVC = 5.36e-19
