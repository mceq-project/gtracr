"""
Tests for gtracr/lib/particle.py and gtracr/lib/location.py
"""

import numpy as np

from gtracr.location import Location
from gtracr.particle import Particle

# ---------------------------------------------------------------------------
# Particle tests
# ---------------------------------------------------------------------------


def test_particle_set_from_rigidity():
    """Proton with R=10 GV: momentum = R * |charge| = 10 GV"""
    p = Particle("proton", 2212, 0.937272, 1, "p+")
    p.set_from_rigidity(10.0)
    assert np.isclose(p.rigidity, 10.0)
    assert np.isclose(p.momentum, 10.0)  # charge = 1


def test_particle_set_from_energy():
    """Proton with E=10 GeV: momentum = sqrt(E^2 - m^2)"""
    p = Particle("proton", 2212, 0.937272, 1, "p+")
    p.set_from_energy(10.0)
    expected_momentum = np.sqrt(10.0**2 - 0.937272**2)
    assert np.isclose(p.momentum, expected_momentum)
    assert np.isclose(p.rigidity, expected_momentum)  # charge = 1


def test_particle_set_from_momentum():
    """Direct momentum assignment"""
    p = Particle("proton", 2212, 0.937272, 1, "p+")
    p.set_from_momentum(5.0)
    assert np.isclose(p.momentum, 5.0)
    assert np.isclose(p.rigidity, 5.0)


def test_particle_get_energy_rigidity():
    """Relativistic energy = sqrt((R*|q|)^2 + m^2) + m"""
    p = Particle("proton", 2212, 0.937272, 1, "p+")
    p.set_from_rigidity(10.0)
    energy = p.get_energy_rigidity()
    expected = np.sqrt((10.0 * 1) ** 2 + 0.937272**2) + 0.937272
    assert np.isclose(energy, expected)


def test_particle_str():
    """__str__ should run without error and contain particle name"""
    p = Particle("proton", 2212, 0.937272, 1, "p+")
    p.set_from_rigidity(10.0)
    s = str(p)
    assert "proton" in s
    assert "2212" in s


# ---------------------------------------------------------------------------
# Location tests
# ---------------------------------------------------------------------------


def test_location_attributes():
    """Location stores name, latitude, longitude, altitude"""
    loc = Location("TestSite", 36.43, 137.28, 0.0)
    assert loc.name == "TestSite"
    assert np.isclose(loc.latitude, 36.43)
    assert np.isclose(loc.longitude, 137.28)
    assert np.isclose(loc.altitude, 0.0)


def test_location_default_altitude():
    """Default altitude is 0"""
    loc = Location("TestSite", 36.43, 137.28)
    assert np.isclose(loc.altitude, 0.0)


def test_location_str():
    """__str__ should run without error and contain location name"""
    loc = Location("Kamioka", 36.4348, 137.2766, 0.0)
    s = str(loc)
    assert "Kamioka" in s
    assert "36.4348" in s
