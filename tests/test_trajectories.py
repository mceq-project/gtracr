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
        0.30199000000016124,
        0.19222000000005146,
        0.20289000000006213,
        0.20302000000006226,
        0.2024500000000617,
        0.19871000000005795,
        0.21701000000007625,
        0.19499000000005423,
        0.23247000000009171,
        0.00732999999999987,
        0.01935999999999938,
        0.19331000000005255,
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
        [6471199.437129436, 1.5707963267948966, 0.0004630454056019748, -6.023391231544523e-21, -9.79628001904842e-34, 1.6032856846569677e-17],
        [63713556.59941996, 1.5707963267948966, -0.8094055241343621, 1.550677095082434e-17, 1.0995078207314884e-33, 4.073400253167595e-18],
        [63714631.4838648, 0.7982116428172709, 2.1313311140212856, 2.666936839158172e-17, 7.52311741100198e-19, 1.487829266667849e-18],
        [63714682.532078445, 0.25601931549347673, -1.9076989423180275, 1.0682383312312213e-17, -3.576532639325838e-19, -6.58938372090488e-20],
        [63712450.14428246, 2.868954712695283, 0.9829378294845721, 1.0683129418086942e-17, 3.3134211029793697e-19, -8.080426040531458e-20],
        [63712929.43207792, 1.611671313912196, -2.795144444245858, 1.0586286076839033e-17, 5.703993455457252e-19, -1.360427422568017e-18],
        [63713242.15087602, 0.9066448814733582, -2.1885120293390687, 1.0576152027038227e-17, 8.8878661156518e-19, 1.265163130283141e-18],
        [63712026.15437999, 1.733760299118192, -1.6201727190454, 1.0478515486757649e-17, 1.4985373926238789e-18, 1.4834655246545783e-18],
        [63714241.71332026, 0.40086871125261875, -1.5296791895701212, 1.068634187806382e-17, 1.8568784997655415e-19, -1.1484064464040173e-19],
        [63712578.96243469, 1.550568203123201, -1.3821633242717883, 1.0304290859163843e-17, -1.0189163197890228e-18, 2.6512205696943613e-18],
        [6469993.253658975, 1.7372674341162628, -2.948737389675609, -2.4420125408402106e-18, 7.29143531856483e-19, 8.032886487527875e-19],
        [6469496.584202795, 1.7560771799259651, -2.590919727554374, -4.829599916839137e-18, -1.9973217847713886e-18, 9.978083950236573e-19],
        [63713290.24231917, 0.9084165493963997, -1.2410259096287855, 2.663924305630752e-17, 1.1874268313190225e-18, 1.6504922556510603e-18],
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
