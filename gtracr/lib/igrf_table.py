"""
Python-level IGRF lookup table.

Mirrors the logic in gtracr/lib/gpu/igrf_table.cpp using numpy so that the
tabulated field can be used from Python (e.g. via pTrajectoryTracer) without
requiring GPU support.

Grid layout (matches the C++ constants in igrf_table.hpp):
  Nr=64    log-spaced radial points  [1 RE … 10 RE]
  Ntheta=128  linear colatitude       [0 … π]
  Nphi=256    linear longitude        [0 … 2π)
  Layout:  table[comp, ir, itheta, iphi]  — component-major, shape (3, Nr, Ntheta, Nphi)
"""

import os
from datetime import date as _date

import numpy as np

from gtracr.lib.constants import EARTH_RADIUS
from gtracr.utils import ymd_to_dec

# Grid dimensions — keep in sync with igrf_table.hpp
NR = 64
NTHETA = 128
NPHI = 256

_R_MIN = EARTH_RADIUS  # 1 RE  in metres
_R_MAX = 10.0 * EARTH_RADIUS  # 10 RE in metres

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class IGRFTable:
    """
    Precomputed IGRF lookup table with trilinear interpolation.

    Provides the same ``values(r, theta, phi) -> (Br, Btheta, Bphi)``
    interface as ``MagneticField`` / ``IGRF13`` so it can be used as a
    drop-in replacement in ``pTrajectoryTracer``.

    Parameters
    ----------
    igrf_obj : _libgtracr.IGRF, optional
        An already-initialised C++ IGRF object.  When *None* a new one is
        constructed from the bundled igrf13.json using today's date.
    verbose : bool
        Print progress while building the table (default False).
    """

    def __init__(self, igrf_obj=None, verbose=False):
        if igrf_obj is None:
            from gtracr.lib._libgtracr import IGRF

            datapath = os.path.abspath(_DATA_DIR)
            dec_date = float(ymd_to_dec(str(_date.today())))
            igrf_obj = IGRF(datapath, dec_date)

        self._r_min = _R_MIN
        self._r_max = _R_MAX
        self._log_r_min = np.log(_R_MIN)
        self._log_r_max = np.log(_R_MAX)

        self._table = self._build(igrf_obj, verbose)

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build(self, igrf, verbose):
        """Evaluate igrf.values() on every grid point; return (3,Nr,Nt,Np) float32 array."""
        table = np.empty((3, NR, NTHETA, NPHI), dtype=np.float32)

        # Precompute grid vectors
        t_vec = np.arange(NR, dtype=np.float64) / (NR - 1)  # 0..1
        r_vec = _R_MIN * (_R_MAX / _R_MIN) ** t_vec  # log-spaced [m]
        th_vec = np.arange(NTHETA, dtype=np.float64) / (NTHETA - 1) * np.pi
        ph_vec = np.arange(NPHI, dtype=np.float64) / NPHI * 2.0 * np.pi

        total = NR * NTHETA
        done = 0
        for ir, r in enumerate(r_vec):
            for itheta, theta in enumerate(th_vec):
                for iphi, phi in enumerate(ph_vec):
                    b = igrf.values(r, theta, phi)
                    table[0, ir, itheta, iphi] = b[0]
                    table[1, ir, itheta, iphi] = b[1]
                    table[2, ir, itheta, iphi] = b[2]
                done += 1
                if verbose and done % (total // 10) == 0:
                    print(f"  igrf_table: {100 * done // total}% done", flush=True)

        return table

    # ------------------------------------------------------------------
    # Lookup (trilinear interpolation)
    # ------------------------------------------------------------------

    def values(self, r, theta, phi):
        """
        Interpolate B-field from the table.

        Parameters
        ----------
        r, theta, phi : float
            Position in geocentric spherical coordinates.
            r in metres, theta in radians [0,π], phi in radians (any range).

        Returns
        -------
        tuple of float : (Br, Btheta, Bphi)  in Tesla
        """
        # ---- radial index (log-spaced) --------------------------------
        log_r = np.log(max(r, self._r_min))
        fr = (log_r - self._log_r_min) / (self._log_r_max - self._log_r_min) * (NR - 1)
        fr = float(np.clip(fr, 0.0, NR - 1 - 1e-6))
        ir0 = int(fr)
        wr = fr - ir0

        # ---- theta index (linear) ------------------------------------
        ft = theta * (NTHETA - 1) / np.pi
        ft = float(np.clip(ft, 0.0, NTHETA - 1 - 1e-6))
        it0 = int(ft)
        wt = ft - it0

        # ---- phi index (linear, periodic) ----------------------------
        two_pi = 2.0 * np.pi
        phi_w = phi - two_pi * np.floor(phi / two_pi)  # wrap to [0, 2π)
        fp = phi_w * NPHI / two_pi
        fp = float(np.clip(fp, 0.0, NPHI - 1e-6))
        ip0 = int(fp) % NPHI
        ip1 = (ip0 + 1) % NPHI
        wp = fp - int(fp)

        # ---- trilinear interpolation ---------------------------------
        t = self._table
        result = []
        for c in range(3):
            v000 = float(t[c, ir0, it0, ip0])
            v001 = float(t[c, ir0, it0, ip1])
            v010 = float(t[c, ir0, it0 + 1, ip0])
            v011 = float(t[c, ir0, it0 + 1, ip1])
            v100 = float(t[c, ir0 + 1, it0, ip0])
            v101 = float(t[c, ir0 + 1, it0, ip1])
            v110 = float(t[c, ir0 + 1, it0 + 1, ip0])
            v111 = float(t[c, ir0 + 1, it0 + 1, ip1])

            along_phi_00 = (1.0 - wp) * v000 + wp * v001
            along_phi_01 = (1.0 - wp) * v010 + wp * v011
            along_phi_10 = (1.0 - wp) * v100 + wp * v101
            along_phi_11 = (1.0 - wp) * v110 + wp * v111

            along_theta_0 = (1.0 - wt) * along_phi_00 + wt * along_phi_01
            along_theta_1 = (1.0 - wt) * along_phi_10 + wt * along_phi_11

            result.append((1.0 - wr) * along_theta_0 + wr * along_theta_1)

        return result[0], result[1], result[2]

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self, igrf_obj, n=10000, rng_seed=42):
        """
        Compare table interpolation against direct ``igrf_obj.values()`` at
        *n* random points.

        Returns
        -------
        max_rel_err : float
            Maximum relative error across all components and test points.
            Target: < 0.001 (0.1 %).
        """
        rng = np.random.default_rng(rng_seed)
        r_pts = rng.uniform(self._r_min, self._r_max, n)
        th_pts = rng.uniform(0.0, np.pi, n)
        ph_pts = rng.uniform(0.0, 2.0 * np.pi, n)

        eps = 1e-30
        max_rel = 0.0
        for r, theta, phi in zip(r_pts, th_pts, ph_pts):
            direct = igrf_obj.values(r, theta, phi)
            interp = self.values(r, theta, phi)
            for d, v in zip(direct, interp):
                rel = abs(v - d) / (abs(d) + eps)
                if rel > max_rel:
                    max_rel = rel

        return max_rel
