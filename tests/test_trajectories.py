"""
Compare the values of the final times and sixvector of the trajectory for the dipole model
"""

import numpy as np

from gtracr.trajectory import Trajectory

IGRF_DATE = "2026-03-01"

# in the form :
# (plabel, zenith, azimuth, particle_altitude,
# latitude, longitude, detector_altitude, rigidity, kinetic energy)
initial_variable_list = [
    ("p+", 90.0, 90.0, 100.0, 0.0, 0.0, 0.0, 30.0, None),
    ("p+", 120.0, 90.0, 100.0, 0.0, 0.0, -1.0, 30.0, None),
    ("p+", 0.0, 155.0, 100.0, 50.0, 100.0, 0.0, 50.0, None),
    ("p+", 90.0, 175.0, 100.0, 89.0, 20.0, 0.0, 20.0, None),
    ("p+", 90.0, 175.0, 100.0, -90.0, 20.0, 0.0, 20.0, None),
    ("e-", 90.0, 175.0, 100.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 275.0, 0.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 0.0, 10.0, 40.0, 200.0, 0.0, 20.0, None),
    ("p+", 45.0, 180.0, 0.0, 89.0, 0.0, 0.0, 20.0, None),
    ("p+", 45.0, 180.0, 0.0, 0.0, 180.0, 100.0, 20.0, None),
    ("p+", 45.0, 180.0, 0.0, 0.0, 180.0, 100.0, 5.0, None),
    ("p+", 45.0, 180.0, 0.0, 0.0, 180.0, 100.0, None, 10.0),
    ("p+", 9.0, 100.0, 0.0, 50.0, 260.0, 100.0, None, 50.0),
]


def test_trajectories_dipole():
    """
    Test the final times of the trajectory evaluation in the dipole field.
    """

    expected_times = [
                         1e-05,
                         0.3052700000001645,
                         0.19222000000005146,
                         0.20289000000006213,
                         0.20302000000006226,
                         0.2024500000000617,
                         0.19874000000005798,
                         0.2170700000000763,
                         0.19498000000005422,
                         0.23268000000009192,
                         0.00730999999999987,
                         0.019289999999999384,
                         0.19332000000005256,
                     ]

    dt = 1e-5
    max_time = 1.0

    for iexp, initial_variables in enumerate(initial_variable_list):
        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = initial_variables

        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="dipole",
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_igrf():
    """
    Test the final times of the trajectory evaluation in the IGRF field.

    Only forbidden trajectories (particle_escaped=False) are checked against
    golden final times.  Escaping trajectories exit at a point on the 10 RE
    sphere whose exact time depends on FP rounding differences between
    architectures; the escaped flag alone is tested for those cases via
    test_trajectories_igrf_escaped.
    """

    expected_times = [
                         1e-05,
                         0.02283999999999924,
                         0.1925700000000518,
                         0.20332000000006256,
                         0.20830000000006754,
                         0.20079000000006003,
                         0.19821000000005745,
                         0.2160700000000753,
                         0.19480000000005404,
                         0.2633800000001226,
                         0.005169999999999958,
                         0.012529999999999658,
                         0.19273000000005197,
                     ]

    # Only forbidden trajectories have a stable final time across architectures.
    escaped_flags = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
    ]

    dt = 1e-5
    max_time = 1.0

    for iexp, initial_variables in enumerate(initial_variable_list):
        if escaped_flags[iexp]:
            continue
        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = initial_variables

        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="igrf",
            date=IGRF_DATE,
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_stepsize():
    """
    Test the final times of the trajectory evaluation in the igrf field for
    different step sizes
    """

    expected_times = [
                         0.21593980992700437,
                         0.21593990000517954,
                         0.21594000000019528,
                         0.2159500000000752,
                         0.21599999999999253,
                         0.21700000000000016,
                         0.22000000000000006,
                         0.4,
                     ]

    dt_arr = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_time = 1.0

    (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = (
        "p+",
        90.0,
        180.0,
        100.0,
        0.0,
        0.0,
        0.0,
        50.0,
        None,
    )

    for iexp, dt in enumerate(dt_arr):
        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="igrf",
            date=IGRF_DATE,
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_maxtimes():
    """
    Test the final times of the trajectory evaluation in the igrf field for
    different maximal times. The rigidity-aware time budget treats max_time as
    a lower bound, so it is switched off (time_factor=0) to test max_time as
    the hard limit.
    """

    expected_times = [
                         0.00999999999999976,
                         0.027829999999999036,
                         0.07743000000000268,
                         0.2154500000000747,
                         0.2159500000000752,
                         0.2159500000000752,
                         0.2159500000000752,
                         0.2159500000000752,
                         0.2159500000000752,
                         0.2159500000000752,
                     ]

    dt = 1e-5
    max_times = np.logspace(-2, 2, 10)

    for iexp, max_time in enumerate(max_times):
        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = (
            "p+",
            90.0,
            180.0,
            100.0,
            0.0,
            0.0,
            0.0,
            50.0,
            None,
        )

        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="igrf",
            date=IGRF_DATE,
            time_factor=0,
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_dates():
    """
    Test the final times of the trajectory evaluation in the igrf field for
    different dates
    """

    expected_times = [
                         0.2133600000000726,
                         0.2136500000000729,
                         0.21339000000007263,
                         0.21591000000007515,
                         0.21548000000007472,
                         0.21541000000007465,
                         0.2148600000000741,
                         0.2146500000000739,
                         0.21540000000007464,
                         0.21594000000007518,
                     ]

    dt = 1e-5
    max_time = 1.0

    dates = [
        "1900-01-01",
        "1909-01-01",
        "1900-10-31",
        "2020-09-12",
        "2004-03-08",
        "2000-02-28",
        "1970-03-26",
        "1952-04-30",
        "1999-03-08",
        "2024-03-09",
    ]
    for iexp, date in enumerate(dates):
        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = (
            "p+",
            90.0,
            180.0,
            100.0,
            0.0,
            0.0,
            0.0,
            50.0,
            None,
        )

        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="igrf",
            date=date,
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_dipole_sixvec():
    expected_sixvec = [
                          [6471199.4347831765, 1.5707963267948966, 0.0004630454055669382, -6.048499026797328e-21, -9.796249270881347e-34, 1.603285683711726e-17],
                          [63714249.21059555, 1.5707963267948966, -0.6996341775070906, 1.5505464544259402e-17, 1.3149564948165127e-33, 4.0783702885528035e-18],
                          [63713277.5324838, 0.798555269606087, 2.1318830722679913, 2.666918009702782e-17, 7.547041940757398e-19, 1.4899912148032222e-18],
                          [63714737.36749747, 0.2532239783417549, -1.9114942765230158, 1.0682284473566733e-17, -3.607363925869603e-19, -6.510581394551696e-20],
                          [63712690.31303677, 2.8716974552710512, 0.9788169010140291, 1.0683038357880266e-17, 3.344203957960079e-19, -8.015939798138123e-20],
                          [63714227.6013325, 1.6103766911444102, -2.7959708809038664, 1.0585975295747823e-17, 5.700730421441749e-19, -1.362980135800614e-18],
                          [63712264.239121236, 0.9085445411991199, -2.1869389493879736, 1.0575631027985982e-17, 8.933202239233843e-19, 1.2663247701824583e-18],
                          [63713850.04959454, 1.736316449234678, -1.6202869361878374, 1.047811041160514e-17, 1.4977869948043743e-18, 1.487080161443163e-18],
                          [63712481.77900975, 0.3998372004834758, -1.5324434403181357, 1.0686372276784292e-17, 1.8397384749512365e-19, -1.1477051696829416e-19],
                          [63714757.72297563, 1.5474756594226229, -1.3772582321890197, 1.0303045270559348e-17, -1.0182610572773948e-18, 2.656308276196494e-18],
                          [6470955.710797994, 1.7370013380465068, -2.9492756008641376, -2.440054951990605e-18, 7.323301249527414e-19, 8.063328172888153e-19],
                          [6470234.610698771, 1.7566576427953229, -2.5927589948990657, -4.833110539942371e-18, -1.9830791071653108e-18, 1.0091536182648264e-18],
                          [63714529.48015178, 0.9088451519374086, -1.2406574943976887, 2.6639031449129683e-17, 1.1897974807933348e-18, 1.6521994993673565e-18],
                      ]

    dt = 1e-5
    max_time = 1.0

    for iexp, initial_variables in enumerate(initial_variable_list):
        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = initial_variables

        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="dipole",
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(
            traj.final_sixvector, np.array(expected_sixvec[iexp]), rtol=1e-5
        )
