"""Cosmic ray particle data class."""

import numpy as np


class Particle:
    """
    Cosmic ray particle with kinematic properties.

    Parameters
    ----------
    name : str
        Human-readable particle name (e.g. ``"proton"``).
    pid : int
        PDG Monte Carlo particle ID.
    mass : float
        Rest mass in GeV/c².
    charge : int
        Electric charge in units of elementary charge *e*.
    label : str
        Short label used as dictionary key (e.g. ``"p+"``).

    Notes
    -----
    PDG IDs: http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf

    Examples
    --------
    >>> proton = Particle("proton", 2212, 0.938272, 1, "p+")
    """

    def __init__(self, name, pid, mass, charge, label):
        self.name = name
        self.pid = pid
        self.mass = mass
        self.charge = charge
        self.label = label
        self.momentum = 0.0
        self.rigidity = 0.0

    def set_from_energy(self, energy):
        """Set momentum and rigidity from total relativistic energy.

        Parameters
        ----------
        energy : float
            Total relativistic energy in GeV.
        """
        self.momentum = np.sqrt(energy**2.0 - self.mass**2.0)
        self.rigidity = self.momentum / np.abs(self.charge)

    def set_from_rigidity(self, rigidity):
        """Set momentum and rigidity from magnetic rigidity.

        Parameters
        ----------
        rigidity : float
            Magnetic rigidity in GV (momentum / charge).
        """
        self.momentum = rigidity * np.abs(self.charge)
        self.rigidity = rigidity

    def set_from_momentum(self, momentum):
        """Set momentum and compute rigidity from relativistic momentum.

        Parameters
        ----------
        momentum : float
            Relativistic momentum in GeV/c.
        """
        self.momentum = momentum
        self.rigidity = self.momentum / np.abs(self.charge)

    def get_energy_rigidity(self):
        """Compute total energy from the current rigidity.

        Returns
        -------
        float
            Total relativistic energy in GeV.
        """
        return (
            np.sqrt((self.rigidity * np.abs(self.charge)) ** 2.0 + self.mass**2.0)
            + self.mass
        )

    def __str__(self):
        return (
            f"{self.name}: PID = {self.pid}, m = {self.mass:.6f} GeV, "
            f"Z = {self.charge}e\n"
            f"  Momentum = {self.momentum:.6e}, Rigidity = {self.rigidity:.6e}"
        )
