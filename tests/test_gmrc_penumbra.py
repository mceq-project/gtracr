"""
Tests for the penumbra-resolving GMRC mode (method="penumbra").
"""

import numpy as np
import pytest

from gtracr._libgtracr import TrajectoryTracer
from gtracr.constants import (
    EARTH_RADIUS,
    ELEMENTARY_CHARGE,
    KG_M_S_PER_GEVC,
    KG_PER_GEVC2,
)
from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.location import Location
from gtracr.trajectory import Trajectory

SPEED_OF_LIGHT = 299792458.0
DIPOLE_B0 = 29404.8e-9  # |B0| of gtracr's dipole (MagneticField.hpp), tesla


def _penumbra(**kw):
    opts = dict(
        bfield_type="dipole",
        solver="rk45",
        atol=1e-6,
        rtol=1e-6,
        method="penumbra",
        n_workers=4,
    )
    opts.update(kw)
    return GMRC(**opts)


def _stormer_vertical_label(lat_deg, palt_km):
    """Störmer vertical cutoff of gtracr's centred, axis-aligned dipole at r = RE + palt,
    expressed in gtracr's rigidity label (p_SI = R * KG_M_S_PER_GEVC)."""
    r = EARTH_RADIUS + palt_km * 1e3
    rigidity_volt = (
        SPEED_OF_LIGHT
        * DIPOLE_B0
        * EARTH_RADIUS**3
        * np.cos(np.radians(lat_deg)) ** 4
        / (4.0 * r * r)
    )
    return rigidity_volt * ELEMENTARY_CHARGE / (KG_M_S_PER_GEVC * SPEED_OF_LIGHT)


# ---------------------------------------------------------------------------
# Ladder summary
# ---------------------------------------------------------------------------


def test_summary_matches_ladder():
    """R_L, R_U, R_eff and flag follow from the stored allowed/forbidden ladder."""
    delta = 0.01
    g = _penumbra(
        location=Location("t", 30.0, 0.0, 0.0),
        min_rigidity=3.0,
        max_rigidity=30.0,
        ladder_delta=delta,
        coarse=1,
        store_ladder=True,
    )
    g.evaluate_batch(zenith=[0.0, 30.0, 60.0], azimuth=[0.0, 90.0, 270.0])
    d = g.data_dict
    q = 1.0 + delta
    rk = g.ladder
    np.testing.assert_allclose(rk, 3.0 * q ** np.arange(len(rk)))
    lo, hi = rk / np.sqrt(q), rk * np.sqrt(q)
    allowed = g.ladder_allowed
    assert allowed.shape == (3, len(rk))
    for i in range(3):
        a = allowed[i]
        forb = np.nonzero(~a)[0]
        assert d["flag"][i] == 0
        assert d["r_upper"][i] == pytest.approx(hi[forb[-1]])
        assert d["r_lower"][i] == pytest.approx(lo[np.argmax(a)])
        assert d["r_eff"][i] == pytest.approx(lo[0] + np.sum((hi - lo)[forb]))
        assert d["r_lower"][i] <= d["r_eff"][i] <= d["r_upper"][i]
    np.testing.assert_array_equal(d["rcutoff"], d["r_eff"])


def test_flags_outside_ladder():
    """Flag 1: cutoff below the ladder; flag 2: cutoff at or above the ladder top."""
    g = _penumbra(
        location=Location("t", 0.0, 0.0, 0.0),
        min_rigidity=1.0,
        max_rigidity=5.0,
        ladder_delta=0.05,
    )
    g.evaluate_batch(zenith=[0.0], azimuth=[0.0])
    assert g.data_dict["flag"][0] == 2
    assert g.ladder[-1] >= 5.0
    assert g.data_dict["r_upper"][0] == pytest.approx(g.ladder[-1] * np.sqrt(1.05))

    g = _penumbra(
        location=Location("t", 80.0, 0.0, 0.0),
        min_rigidity=5.0,
        max_rigidity=50.0,
        ladder_delta=0.05,
    )
    g.evaluate_batch(zenith=[0.0], azimuth=[0.0])
    d = g.data_dict
    assert d["flag"][0] == 1
    assert d["r_lower"][0] == d["r_upper"][0] == d["r_eff"][0]
    assert d["r_eff"][0] == pytest.approx(5.0 / np.sqrt(1.05))


# ---------------------------------------------------------------------------
# Dipole: Störmer bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lat", [40.0, 45.0, 50.0])
def test_dipole_stormer_lower_cutoff(lat):
    """In a dipole, Störmer's cutoff bounds the allowed region from below and the
    lowest allowed rigidity R_L sits on it (the Earth's shadow only adds forbidden
    bands above it). In the meridian plane (azimuth 0/180) Störmer's cutoff does not
    depend on zenith. Latitudes are chosen so the Störmer length stays well inside
    the 10 RE escape sphere; max_time = 10 s lets slow escapes near the cutoff finish.
    """
    delta = 0.005
    palt = 100.0
    g = _penumbra(
        location=Location("t", lat, 0.0, 0.0),
        particle_altitude=palt,
        min_rigidity=0.5,
        max_rigidity=20.0,
        ladder_delta=delta,
        coarse=1,
        atol=0.0,
        rtol=1e-9,
    )
    zen = [0.0, 30.0, 0.0, 30.0]
    az = [0.0, 0.0, 180.0, 180.0]
    g.evaluate_batch(zenith=zen, azimuth=az, max_time=10.0)
    d = g.data_dict
    ratio = d["r_lower"] / _stormer_vertical_label(lat, palt)
    assert np.all(d["flag"] == 0)
    assert np.all(ratio >= 1.0 - 2 * delta)
    assert np.all(ratio <= 1.03)
    assert np.all(d["r_lower"] <= d["r_eff"])
    assert np.all(d["r_eff"] <= d["r_upper"])
    # vertical: azimuth is irrelevant
    assert d["r_lower"][0] == d["r_lower"][2]
    assert d["r_eff"][0] == d["r_eff"][2]


# ---------------------------------------------------------------------------
# IGRF: C++ ladder scan vs brute-force single-trajectory scan
# ---------------------------------------------------------------------------


def _brute_force_ladder(loc, palt, date, zenith, azimuth, ladder):
    traj = Trajectory(
        plabel="p+",
        location_name=loc,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        rigidity=ladder[-1],
        bfield_type="igrf",
        particle_altitude=palt,
        date=date,
        solver="rk45",
        atol=1e-6,
        rtol=1e-6,
    )
    tracer = TrajectoryTracer(
        traj.charge * ELEMENTARY_CHARGE,
        traj.mass * KG_PER_GEVC2,
        traj.start_alt,
        traj.esc_alt,
        1e-5,
        100000,
        traj.bfield_type,
        traj.igrf_params,
        traj.solver_char,
        1e-6,
        1e-6,
    )
    ref = traj.particle.momentum * KG_M_S_PER_GEVC
    pos = list(traj.particle_sixvector[:3])
    mom_unit = traj.particle_sixvector[3:] / ref
    allowed = np.zeros(len(ladder), bool)
    for k, rig in enumerate(ladder):
        traj.particle.set_from_rigidity(rig)
        mom = traj.particle.momentum * KG_M_S_PER_GEVC
        tracer.reset()
        tracer.evaluate(0.0, pos + list(mom_unit * mom))
        allowed[k] = tracer.particle_escaped
    return allowed


def test_igrf_full_ladder_matches_single_trajectories():
    """The C++ batch scan (coarse=1) reproduces a rung-by-rung scan with the
    single-trajectory tracer and Python initial conditions exactly; the adaptive
    scan agrees on R_eff to better than 1 %."""
    loc = Location("Kamioka", 36.4348, 137.276599, 0.0)
    palt, date = 100.0, "2024-01-15"
    zen = [0.0, 60.0]
    az = [0.0, 270.0]
    kw = dict(
        location=loc,
        particle_altitude=palt,
        bfield_type="igrf",
        date=date,
        min_rigidity=5.0,
        max_rigidity=20.0,
        ladder_delta=0.01,
    )
    full = _penumbra(coarse=1, store_ladder=True, **kw)
    full.evaluate_batch(zenith=zen, azimuth=az)
    adapt = _penumbra(coarse=8, guard=2, **kw)
    adapt.evaluate_batch(zenith=zen, azimuth=az)
    assert adapt.total_trajectories < full.total_trajectories
    for i in range(len(zen)):
        brute = _brute_force_ladder(loc, palt, date, zen[i], az[i], full.ladder)
        np.testing.assert_array_equal(full.ladder_allowed[i], brute)
        assert full.data_dict["flag"][i] == 0
    np.testing.assert_allclose(
        adapt.data_dict["r_eff"], full.data_dict["r_eff"], rtol=0.01
    )


# ---------------------------------------------------------------------------
# Random sampling and API
# ---------------------------------------------------------------------------


def test_random_directions_reproducible():
    """Without explicit directions, iter_num directions are drawn and all are kept."""
    kw = dict(
        location="Kamioka",
        iter_num=6,
        min_rigidity=2.0,
        max_rigidity=40.0,
        ladder_delta=0.05,
    )
    a = _penumbra(**kw)
    a.evaluate_batch(base_seed=11, max_time=0.2)
    b = _penumbra(**kw)
    b.evaluate_batch(base_seed=11, max_time=0.2)
    assert len(a.data_dict["r_eff"]) == 6
    assert np.all((a.data_dict["zenith"] >= 0) & (a.data_dict["zenith"] < 180))
    assert np.all((a.data_dict["azimuth"] >= 0) & (a.data_dict["azimuth"] < 360))
    for key in ("zenith", "azimuth", "r_lower", "r_upper", "r_eff", "flag"):
        np.testing.assert_array_equal(a.data_dict[key], b.data_dict[key])
    assert a.ladder_allowed is None


def test_penumbra_defaults_and_errors():
    g = GMRC(method="penumbra")
    assert (g.rmin, g.rmax) == (0.3, 100.0)
    assert (GMRC().rmin, GMRC().rmax) == (5.0, 55.0)
    with pytest.raises(ValueError):
        GMRC(method="stoermer")
    with pytest.raises(ValueError):
        GMRC(method="penumbra", min_rigidity=0.0)
    with pytest.raises(NotImplementedError):
        GMRC(method="penumbra", iter_num=1).evaluate()
    with pytest.raises(ValueError):
        GMRC(bfield_type="dipole", iter_num=1).evaluate_batch(
            zenith=[0.0], azimuth=[0.0]
        )
    with pytest.raises(ValueError):
        _penumbra().evaluate_batch(zenith=[0.0, 1.0], azimuth=[0.0])


def test_batch_bisection_dipole():
    """Batch bisection with the dipole field uses the dipole tracer directly."""
    g = GMRC(
        location=Location("t", 45.0, 0.0, 0.0),
        bfield_type="dipole",
        iter_num=4,
        n_workers=2,
        min_rigidity=1.0,
        max_rigidity=20.0,
        delta_rigidity=0.5,
    )
    g.evaluate_batch(base_seed=3, max_time=0.2)
    assert len(g.data_dict["rcutoff"]) == 4
    assert np.all(g.data_dict["rcutoff"] > 0)
