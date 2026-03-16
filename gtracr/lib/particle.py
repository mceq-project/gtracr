import numpy as np


class Particle:
    """
    Utility class for cosmic ray particles

    Parameters
    ------------

    - name : str
        the name of the particle
    - pid : int
        the particle id as in the PDG database
    - mass : float
        the particle's rest mass in GeV
    - charge : int
        the particle's charge in e
    - label : str
        the shorthand name for the particle

    Notes:
    - PDGID obtained from here: http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
    - The mass of the particles are also obtained from PDG

    Example:
    proton: proton = Particle("Proton", 2212, 0.938272, "p+")
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
        """Set momentum and rigidity from total energy.

        Parameters
        ----------
        energy : float
            Total relativistic energy in GeV.
        """
        self.momentum = np.sqrt(energy**2.0 - self.mass**2.0)
        self.rigidity = self.momentum / np.abs(self.charge)

    def set_from_rigidity(self, rigidity):
        """Set momentum and rigidity from rigidity.

        Parameters
        ----------
        rigidity : float
            Magnetic rigidity in GV (momentum / charge).
        """
        self.momentum = rigidity * np.abs(self.charge)
        self.rigidity = rigidity

    def set_from_momentum(self, momentum):
        """Set momentum and compute rigidity from momentum.

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

    # string represetation for print output
    def __str__(self):
        return f"{self.name:s}: PID = {self.pid:d}, m = {self.mass:.6f}GeV, Z = {self.charge:d}e \n Momentum = {self.momentum:.6e}, Rigidity = {self.rigidity:.6e}"


# example using proton
if __name__ == "__main__":
    proton = Particle("Proton", 2122, 0.937272, 1, "p+")
    print(proton)
    print(proton.momentum)
