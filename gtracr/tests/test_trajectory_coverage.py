"""
Tests for gtracr/trajectory.py — branch and feature coverage.
"""

import os

import numpy as np
import pytest

from gtracr.lib.constants import EARTH_RADIUS
from gtracr.trajectory import Trajectory

# ---------------------------------------------------------------------------
# get_trajectory — get_data=True
# ---------------------------------------------------------------------------


def test_get_data_keys():
    """get_trajectory(get_data=True) should return a dict with the expected keys"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        latitude=0.0,
        longitude=0.0,
        bfield_type="dipole",
    )
    result = traj.get_trajectory(dt=1e-4, max_time=0.05, get_data=True)
    assert result is not None
    for key in ["t", "r", "theta", "phi", "pr", "ptheta", "pphi"]:
        assert key in result
    # convert_to_cartesian adds x, y, z
    for key in ["x", "y", "z"]:
        assert key in result


def test_get_data_array_types():
    """Values in the returned dict should be numpy arrays"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        bfield_type="dipole",
    )
    result = traj.get_trajectory(dt=1e-4, max_time=0.05, get_data=True)
    for key in ["r", "theta", "phi", "pr", "ptheta", "pphi"]:
        assert isinstance(result[key], np.ndarray)


# ---------------------------------------------------------------------------
# Constructor branches
# ---------------------------------------------------------------------------


def test_location_name_param():
    """Construct Trajectory with location_name instead of lat/lng/alt"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=10.0,
        location_name="Kamioka",
        bfield_type="dipole",
    )
    # Kamioka latitude ≈ 36.43
    assert np.isclose(traj.lat, 36.4348)
    assert np.isclose(traj.lng, 137.276599)


def test_energy_input():
    """Construct Trajectory with energy= instead of rigidity="""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        energy=10.0,
        bfield_type="dipole",
    )
    assert traj.energy == 10.0
    assert traj.rigidity > 0.0


def test_error_both_none():
    """Providing neither energy nor rigidity should raise Exception"""
    with pytest.raises(Exception):
        Trajectory(
            zenith_angle=0.0,
            azimuth_angle=0.0,
            energy=None,
            rigidity=None,
            bfield_type="dipole",
        )


def test_particle_escaped_low_rigidity():
    """Very low rigidity (0.1 GV) proton at equator should NOT escape"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=0.1,
        latitude=0.0,
        longitude=0.0,
        bfield_type="dipole",
    )
    traj.get_trajectory(dt=1e-4, max_time=0.05)
    assert traj.particle_escaped is False


def test_particle_escaped_high_rigidity():
    """High rigidity (50 GV) proton pointing up should escape"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        latitude=0.0,
        longitude=0.0,
        bfield_type="dipole",
    )
    traj.get_trajectory(dt=1e-4, max_time=0.5)
    assert traj.particle_escaped is True


def test_electron_minus():
    '''Construct Trajectory with plabel="e-"'''
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=10.0,
        plabel="e-",
        bfield_type="dipole",
    )
    assert traj.charge == -1


def test_proton_minus():
    '''Construct Trajectory with plabel="p-"'''
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=10.0,
        plabel="p-",
        bfield_type="dipole",
    )
    assert traj.charge == -1


# ---------------------------------------------------------------------------
# use_python branch
# ---------------------------------------------------------------------------


def test_use_python_runs():
    """use_python=True should run without error (smoke test)"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        bfield_type="dipole",
    )
    # Use very short integration to keep it fast
    traj.get_trajectory(dt=1e-4, max_time=0.02, use_python=True)
    assert traj.final_time > 0.0


# ---------------------------------------------------------------------------
# Coordinate transformation methods
# ---------------------------------------------------------------------------


def test_convert_to_cartesian():
    """convert_to_cartesian should add x, y, z keys to the dict"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        bfield_type="dipole",
    )
    data = {
        "r": np.array([EARTH_RADIUS]),
        "theta": np.array([np.pi / 2.0]),
        "phi": np.array([0.0]),
    }
    traj.convert_to_cartesian(data)
    assert "x" in data
    assert "y" in data
    assert "z" in data
    # At theta=pi/2, phi=0, x should equal r/EARTH_RADIUS
    assert np.isclose(data["x"][0], 1.0, atol=1e-10)


def test_geodesic_to_cartesian():
    """geodesic_to_cartesian at lat=0, lng=0 should give (RE, 0, 0)"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        latitude=0.0,
        longitude=0.0,
        detector_altitude=0.0,
        bfield_type="dipole",
    )
    cart = traj.geodesic_to_cartesian()
    assert np.isclose(cart[0], EARTH_RADIUS, rtol=1e-6)
    assert np.isclose(cart[1], 0.0, atol=1.0)
    assert np.isclose(cart[2], 0.0, atol=1.0)


def test_transform_matrix_identity():
    """At lat=0, lng=0 the transform matrix has specific known structure"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        latitude=0.0,
        longitude=0.0,
        bfield_type="dipole",
    )
    mat = traj.transform_matrix()
    assert mat.shape == (3, 3)
    # At lat=0, lng=0: row3 = [0, cos(0), sin(0)] = [0, 1, 0]
    assert np.isclose(mat[2, 0], 0.0)
    assert np.isclose(mat[2, 1], 1.0)
    assert np.isclose(mat[2, 2], 0.0)


def test_cartesian_to_spherical():
    """cartesian_to_spherical: point on x-axis at radius R"""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=50.0,
        bfield_type="dipole",
    )
    R = EARTH_RADIUS * 2.0
    cart_coord = np.array([R, 0.0, 0.0])
    cart_mmtm = np.array([0.0, 0.0, 0.0])
    sv = traj.cartesian_to_spherical(cart_coord, cart_mmtm)
    assert np.isclose(sv[0], R)  # r
    assert np.isclose(sv[1], np.pi / 2.0, atol=1e-10)  # theta = pi/2 for z=0
    assert np.isclose(sv[2], 0.0, atol=1e-10)  # phi = arctan2(0,R) = 0


# ---------------------------------------------------------------------------
# pTrajectoryTracer tests
# ---------------------------------------------------------------------------


def test_ptrajectorytracer_dipole():
    """
    pTrajectoryTracer direct instantiation with raw (un-converted) values.
    The constructor multiplies charge * ELEMENTARY_CHARGE and mass * KG_PER_GEVC2,
    so pass raw natural units (charge in e, mass in GeV/c²) for correct SI.

    Note: start_altitude is an altitude in meters from surface (not absolute radius).
    The termination check uses r < start_altitude + EARTH_RADIUS.
    """
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    charge_raw = 1  # elementary charges → constructor converts to coulombs
    mass_raw = 0.937272  # GeV/c² → constructor converts to kg
    start_altitude_m = 100e3  # 100 km altitude from surface
    escape_radius = 10.0 * EARTH_RADIUS  # escape at 10 RE

    tracer = pTrajectoryTracer(
        charge=charge_raw,
        mass=mass_raw,
        start_altitude=start_altitude_m,
        escape_radius=escape_radius,
        stepsize=1e-4,
        max_step=200,
        bfield_type="d",
    )

    # Build initial 6-vector: particle at 100 km altitude (absolute r) pointing up
    momentum_si = 0.937272 * 5.36e-19  # 1 GeV/c in kg m/s
    start_r = EARTH_RADIUS + start_altitude_m  # absolute radius
    vec0 = np.array([start_r, np.pi / 2.0, 0.0, momentum_si, 0.0, 0.0])
    tracer.evaluate(0.0, vec0)
    assert tracer.final_time > 0.0


def test_ptrajectorytracer_get_trajectory():
    """pTrajectoryTracer.evaluate_and_get_trajectory() returns a dict"""
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    charge_raw = 1
    mass_raw = 0.937272
    start_altitude_m = 100e3
    escape_radius = 10.0 * EARTH_RADIUS
    momentum_si = 0.937272 * 5.36e-19

    tracer = pTrajectoryTracer(
        charge=charge_raw,
        mass=mass_raw,
        start_altitude=start_altitude_m,
        escape_radius=escape_radius,
        stepsize=1e-4,
        max_step=50,
        bfield_type="d",
    )
    start_r = EARTH_RADIUS + start_altitude_m
    vec0 = np.array([start_r, np.pi / 2.0, 0.0, momentum_si, 0.0, 0.0])
    data = tracer.evaluate_and_get_trajectory(0.0, vec0)
    for key in ["t", "r", "theta", "phi", "pr", "ptheta", "pphi"]:
        assert key in data


def test_ptrajectorytracer_igrf():
    """pTrajectoryTracer with igrf bfield_type should initialize without error"""
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data"
    )

    charge_raw = 1
    mass_raw = 0.937272
    start_altitude_m = 100e3
    escape_radius = 10.0 * EARTH_RADIUS
    momentum_si = 0.937272 * 5.36e-19 * 10.0  # higher momentum

    tracer = pTrajectoryTracer(
        charge=charge_raw,
        mass=mass_raw,
        start_altitude=start_altitude_m,
        escape_radius=escape_radius,
        stepsize=1e-4,
        max_step=50,
        bfield_type="i",
        igrf_params=(data_dir, 2015.0),
    )
    start_r = EARTH_RADIUS + start_altitude_m
    vec0 = np.array([start_r, np.pi / 2.0, 0.0, momentum_si, 0.0, 0.0])
    tracer.evaluate(0.0, vec0)
    assert tracer.final_time > 0.0


def test_igrf13_leap_year():
    """IGRF13 with a leap year date covers the leap_year=True branch (line 111)"""
    from gtracr.lib.magnetic_field import IGRF13

    pyigrf = IGRF13(2000)  # 2000 is a leap year
    # Just verify it initializes and returns valid values
    from gtracr.lib.constants import EARTH_RADIUS

    bf = pyigrf.values(EARTH_RADIUS, np.pi / 2.0, np.pi)
    assert len(bf) == 3


def test_igrf13_large_theta():
    """IGRF13.values() with theta > pi normalizes it (covers line 140)"""
    from gtracr.lib.magnetic_field import IGRF13

    pyigrf = IGRF13(2015)
    from gtracr.lib.constants import EARTH_RADIUS

    # theta in radians: convert to degrees inside values()
    # theta_deg = theta_rad * DEG_PER_RAD
    # theta_rad = 4.0 → theta_deg ≈ 229 > 180, triggers normalization
    theta_large = 4.0  # radians → ~229 degrees > 180
    bf = pyigrf.values(EARTH_RADIUS, theta_large, np.pi)
    assert len(bf) == 3


def test_igrf13_large_phi():
    """IGRF13.values() with phi > 2*pi normalizes it (covers line 143)"""
    from gtracr.lib.magnetic_field import IGRF13

    pyigrf = IGRF13(2015)
    from gtracr.lib.constants import EARTH_RADIUS

    # phi_rad = 7.5 → phi_deg ≈ 430 > 360, triggers normalization
    phi_large = 7.5  # radians → ~430 degrees > 360
    bf = pyigrf.values(EARTH_RADIUS, np.pi / 2.0, phi_large)
    assert len(bf) == 3


def test_ptrajectorytracer_invalid_bfield():
    """pTrajectoryTracer with invalid bfield_type raises Exception"""
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    with pytest.raises(Exception):
        pTrajectoryTracer(
            charge=1,
            mass=0.937272,
            bfield_type="x",
        )


def test_ptrajectorytracer_igrf_no_params():
    """pTrajectoryTracer with igrf but no igrf_params raises Exception"""
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    with pytest.raises(Exception):
        pTrajectoryTracer(
            charge=1,
            mass=0.937272,
            bfield_type="i",
            igrf_params=None,
        )


def test_ptrajectorytracer_escaped():
    """pTrajectoryTracer evaluate: particle escapes (covers lines 209-210).

    The pTrajectoryTracer escape check is: r > EARTH_RADIUS + self.escape_radius
    To make a particle escape quickly we set escape_radius to a small value
    (e.g. 300 km) so the threshold is EARTH_RADIUS + 300 km = 6671200 m.
    The particle starts at r = EARTH_RADIUS + 100 km = 6471200 m and is given
    large outward radial momentum so it moves outward quickly past 6671200.
    start_altitude is set to a small value so the lower bound check doesn't fire.
    """
    from gtracr.lib.constants import KG_M_S_PER_GEVC
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    charge_raw = 1
    mass_raw = 0.937272
    start_altitude_m = 100e3  # lower termination: r < EARTH_RADIUS + 100 km
    small_escape = 300e3  # upper termination: r > EARTH_RADIUS + 300 km
    momentum_si = 50.0 * KG_M_S_PER_GEVC  # large outward momentum

    tracer = pTrajectoryTracer(
        charge=charge_raw,
        mass=mass_raw,
        start_altitude=start_altitude_m,
        escape_radius=small_escape,
        stepsize=1e-4,
        max_step=100000,
        bfield_type="d",
    )
    # start at 100 km altitude, moving outward
    start_r = EARTH_RADIUS + start_altitude_m
    vec0 = np.array([start_r, np.pi / 2.0, 0.0, momentum_si, 0.0, 0.0])
    tracer.evaluate(0.0, vec0)
    assert tracer.particle_escaped is True


def test_ptrajectorytracer_get_traj_escaped():
    """pTrajectoryTracer evaluate_and_get_trajectory: particle escapes (lines 274-275)"""
    from gtracr.lib.constants import KG_M_S_PER_GEVC
    from gtracr.lib.trajectory_tracer import pTrajectoryTracer

    charge_raw = 1
    mass_raw = 0.937272
    start_altitude_m = 100e3
    small_escape = 300e3
    momentum_si = 50.0 * KG_M_S_PER_GEVC

    tracer = pTrajectoryTracer(
        charge=charge_raw,
        mass=mass_raw,
        start_altitude=start_altitude_m,
        escape_radius=small_escape,
        stepsize=1e-4,
        max_step=100000,
        bfield_type="d",
    )
    start_r = EARTH_RADIUS + start_altitude_m
    vec0 = np.array([start_r, np.pi / 2.0, 0.0, momentum_si, 0.0, 0.0])
    data = tracer.evaluate_and_get_trajectory(0.0, vec0)
    assert tracer.particle_escaped is True
    assert "r" in data
