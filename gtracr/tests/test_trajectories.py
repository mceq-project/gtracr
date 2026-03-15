'''
Compare the values of the final times and sixvector of the trajectory for the dipole model
'''

import os
import sys
import numpy as np
import pytest

from gtracr.trajectory import Trajectory

# in the form :
# (plabel, zenith, azimuth, particle_altitude,
# latitude, longitude, detector_altitude, rigidity, kinetic energy)
initial_variable_list = [
    ("p+", 90., 90., 100., 0., 0., 0., 30., None),
    ("p+", 120., 90., 100., 0., 0., -1., 30., None),
    ("p+", 0., 25., 100., 50., 100., 0., 50., None),
    ("p+", 90., 5., 100., 89., 20., 0., 20., None),
    ("p+", 90., 5., 100., -90., 20., 0., 20., None),
    ("e-", 90., 5., 100., 40., 200., 0., 20., None),
    ("p+", 45., 265., 0., 40., 200., 0., 20., None),
    ("p+", 45., 180., 10., 40., 200., 0., 20., None),
    ("p+", 45., 0., 0., 89., 0., 0., 20., None),
    ("p+", 45., 0., 0., 0., 180., 100., 20., None),
    ("p+", 45., 0., 0., 0., 180., 100., 5., None),
    ("p+", 45., 0., 0., 0., 180., 100., None, 10.),
    ("p+", 9., 80., 0., 50., 260., 100., None, 50.),
]


def test_trajectories_dipole():
    '''
    Test the final times of the trajectory evaluation in the dipole field.
    '''

    expected_times = [
        1e-05, 0.3000700000001593, 0.19221000000005145, 0.20289000000006213,
        0.21143000000007067, 0.2024600000000617, 0.19869000000005793,
        0.2169700000000762, 0.19499000000005423, 0.23233000000009157,
        0.007349999999999869, 0.01940999999999938, 0.19331000000005255
    ]

    dt = 1e-5
    max_time = 1.

    for iexp, initial_variables in enumerate(initial_variable_list):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = initial_variables

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="dipole")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_igrf():
    '''
    Test the final times of the trajectory evaluation in the IGRF field.
    '''

    expected_times = [
        1e-05, 0.3000700000001593, 0.19221000000005145, 0.20289000000006213,
        0.21143000000007067, 0.2024600000000617, 0.19869000000005793,
        0.2169700000000762, 0.19499000000005423, 0.23233000000009157,
        0.007349999999999869, 0.01940999999999938, 0.19331000000005255
    ]

    dt = 1e-5
    max_time = 1.

    for iexp, initial_variables in enumerate(initial_variable_list):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = initial_variables

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_stepsize():
    '''
    Test the final times of the trajectory evaluation in the igrf field for
    different step sizes
    '''

    expected_times = [
        0.22073792992447885, 0.22073800000531751, 0.22073800000020008,
        0.22074000000007998, 0.220799999999992, 0.22100000000000017,
        0.23000000000000007, 0.30000000000000004
    ]

    dt_arr = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_time = 1.

    (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
     en) = ("p+", 90., 0., 100., 0., 0., 0., 50., None)

    for iexp, dt in enumerate(dt_arr):

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_maxtimes():
    '''
    Test the final times of the trajectory evaluation in the igrf field for
    different maximal times
    '''

    expected_times = [
        0.00999999999999976, 0.027829999999999036, 0.07743000000000268,
        0.2154500000000747, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998
    ]

    dt = 1e-5
    max_times = np.logspace(-2, 2, 10)

    for iexp, max_time in enumerate(max_times):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = ("p+", 90., 0., 100., 0., 0., 0., 50., None)

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])



def test_trajectories_dates():
    '''
    Test the final times of the trajectory evaluation in the igrf field for
    different dates
    '''

    expected_times = [
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998
    ]

    dt = 1e-5
    max_time = 1.

    dates = [
        "1900-01-01", "1909-01-01", "1900-10-31", "2020-09-12", "2004-03-08",
        "2000-02-28", "1970-03-26", "1952-04-31", "1999-03-08", "2024-03-09"
    ]
    for iexp, date in enumerate(dates):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = ("p+", 90., 0., 100., 0., 0., 0., 50., None)

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf",
                          date=date)

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_dipole_sixvec():

    expected_sixvec = [
        [6.471199625735792e+06, 1.570796326794897e+00, 4.630478780291231e-04, -6.025217124157224e-21, 9.867217328767675e-34, 1.607999924744147e-17],
        [6.371335346026322e+07, 1.570796326794897e+00, 5.409689836827150e+00, 1.555372674920359e-17, 1.791510012189174e-33, 4.082046860194116e-18],
        [6.371259734681733e+07, 7.979748146625527e-01, 2.131001440948860e+00, 2.674803830858280e-17, 7.529699129062378e-19, 1.490904422574872e-18],
        [6.371487178493770e+07, 2.582901355932140e-01, -1.904753969480752e+00, 1.071803571936567e-17, -3.563146559925421e-19, -6.652912354216870e-20],
        [6.371242850812197e+07, 4.610658305770948e+00, 6.284785213384116e+10, 3.588116872178534e-06, 3.662740182309772e-07, 1.740677235989093e-11],
        [6.371412812167440e+07, 1.612670192113047e+00, -2.794639932662458e+00, 1.062056305255945e-17, 5.723174576497950e-19, -1.363073814864442e-18],
        [6.371458356654958e+07, 9.053003311980364e-01, -2.189599300665786e+00, 1.060965169469279e-17, 8.883874272417792e-19, 1.268268268595269e-18],
        [6.371206915991611e+07, 1.731913552668525e+00, -1.620008247289393e+00, 1.051351327064909e-17, 1.503893150800216e-18, 1.485453225462638e-18],
        [6.371385701953655e+07, 4.016896087540504e-01, -1.527572317641708e+00, 1.071857097728965e-17, 1.874818133514414e-19, -1.150944458384313e-19],
        [6.371217143996966e+07, 1.552615389242537e+00, 4.897759649299538e+00, 1.033667870875700e-17, -1.022349522613819e-18, 2.655746939432413e-18],
        [6.468822113847768e+06, 1.737427109977924e+00, 3.335047788120788e+00, -2.455716101088284e-18, 7.264663185259690e-19, 8.019729334895227e-19],
        [6.470696745738055e+06, 1.755643930128702e+00, 3.693730080306008e+00, -4.842999089642077e-18, -2.015885549881965e-18, 1.000837272629210e-18],
        [6.371450809864879e+07, 9.081240198957020e-01, -1.241211445318186e+00, 2.671781397638439e-17, 1.189280561508053e-18, 1.654218899379113e-18],
    ]

    dt = 1e-5
    max_time = 1.

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

        assert np.allclose(traj.final_sixvector, np.array(expected_sixvec[iexp]), rtol=1e-5)
