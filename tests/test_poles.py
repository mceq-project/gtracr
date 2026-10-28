"""Trajectories at and near the geographic poles (polar-axis regularity)."""

import numpy as np
import pytest

from gtracr.trajectory import Trajectory


@pytest.mark.parametrize("latitude", [90.0, -90.0])
@pytest.mark.parametrize("bfield_type", ["dipole", "igrf"])
def test_pole_trajectory_is_finite(latitude, bfield_type):
    """A vertical start on the polar axis must not produce NaN (sin(theta) = 0)."""
    traj = Trajectory(
        zenith_angle=0.0,
        azimuth_angle=0.0,
        rigidity=20.0,
        latitude=latitude,
        longitude=0.0,
        bfield_type=bfield_type,
        date="2024-01-15",
    )
    data = traj.get_trajectory(dt=1e-5, max_time=0.3, get_data=True)
    for key in ("r", "theta", "phi", "pr", "ptheta", "pphi"):
        assert np.all(np.isfinite(data[key])), key
    # 20 GV is far above the polar cutoff
    assert traj.particle_escaped


@pytest.mark.parametrize("solver", ["rk4", "rk45", "boris"])
def test_pole_matches_near_pole(solver):
    """Escape on the axis agrees with a start 0.01 deg off the axis."""
    for rigidity in [0.5, 2.0, 20.0]:
        flags = []
        for latitude in [90.0, 89.99]:
            traj = Trajectory(
                zenith_angle=0.0,
                azimuth_angle=0.0,
                rigidity=rigidity,
                latitude=latitude,
                bfield_type="dipole",
                solver=solver,
            )
            traj.get_trajectory(dt=1e-5, max_time=0.3)
            flags.append(traj.particle_escaped)
        assert flags[0] == flags[1]


@pytest.mark.parametrize("solver", ["rk4", "rk45"])
def test_momentum_conserved_near_south_pole(solver):
    """|p| is conserved in a static field, also for paths that pass the axis.

    The RK integrators used to advance the spherical six-vector, whose
    1/sin(theta) terms made |p| grow by orders of magnitude near a pole.
    """
    tolerances = {"atol": 1e-6, "rtol": 1e-6} if solver == "rk45" else {}
    rng = np.random.default_rng(1)
    for _ in range(10):
        zenith, azimuth = rng.uniform(0.0, 90.0), rng.uniform(0.0, 360.0)
        traj = Trajectory(
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            rigidity=1.0,
            location_name="IceCube",
            bfield_type="dipole",
            solver=solver,
            **tolerances,
        )
        traj.get_trajectory(dt=1e-5, max_time=0.1)
        p0 = np.linalg.norm(traj.particle_sixvector[3:])
        p1 = np.linalg.norm(traj.final_sixvector[3:])
        assert p1 / p0 == pytest.approx(1.0, abs=1e-3)


def test_rk4_matches_boris_near_south_pole():
    """Escape flags of RK4 and Boris agree at IceCube for low rigidities."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        zenith, azimuth = rng.uniform(0.0, 90.0), rng.uniform(0.0, 360.0)
        flags = []
        for solver in ["rk4", "boris"]:
            traj = Trajectory(
                zenith_angle=zenith,
                azimuth_angle=azimuth,
                rigidity=0.3,
                location_name="IceCube",
                bfield_type="dipole",
                solver=solver,
            )
            traj.get_trajectory(dt=1e-5, max_time=0.5)
            flags.append(traj.particle_escaped)
        assert flags[0] == flags[1]
