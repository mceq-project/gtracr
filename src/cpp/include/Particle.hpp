// Header file for Particle class
#ifndef __PARTICLE_HPP_
#define __PARTICLE_HPP_

#include <string>

class Particle {
 private:
  std::string nm;   // Name
  int pid;          // PDG ID
  double m;         // mass
  int ch;           // charge
  std::string lbl;  // label

  double p;  // momentum
  double v;  // velocity
  double R;  // rigidity

 public:
  // Default constructor, set to Proton (PID:2122, charge:+1e, mass:0.938272,
  // label:p+)
  Particle();
  // Constructs a particle with some given name, pdgid, mass, charge, and label
  // to associate the particle with. The particle kinematics are set with the
  // default energy of 1GeV.
  Particle(const std::string& name, const int pdgid, const double& mass, const int charge,
           const std::string& label);
  // Constructs a particle with some given name, pdgid, mass, charge, and label
  // to associate the particle with. The particle kinematics can either be set
  // with user inputs of rigidity or energy, but not both (or none), otherwise
  // this will throw an error
  Particle(const std::string& name, const int pdgid, const double& mass, const int charge,
           const std::string& label, const double& energy, const double& rigidity);
  // destructor
  ~Particle();
  // copy constructor / operator
  Particle(const Particle&);
  Particle& operator=(const Particle&);

  // getters
  // the name of the particle
  const std::string& name() const { return nm; }
  // the mass of the particle
  const double& mass() const { return m; }
  // the charge of the particle
  const int charge() const { return ch; }
  // the PDG ID of the particle, defined by PDG
  const int pdgid() const { return pid; }
  // the label associated with the particle
  const std::string& label() const { return lbl; }
  // the particle momentum
  const double& momentum() const { return p; }
  // the particle velocity in magnitude
  const double& velocity() const { return v; }
  // the particle rigidity
  const double& rigidity() const { return R; }

  // setters
  // modify the momentum given another value of the momentum
  void set_momentum(const double& _p) { p = _p; }
  // modify the velocity given another value of velocity
  void set_velocity(const double& _v) { v = _v; }
  // modify the rigidity given another value of rigidity
  void set_rigidity(const double& _R) { R = _R; }

  // setters for kinematical variables (momentum, velocity, rigidity)
  // set kinematical variables from energy
  void set_from_energy(const double& energy);
  // set kinematical variables from momentum
  void set_from_momentum(const double& momentum);
  // set kinematical variables from rigidity
  void set_from_rigidity(const double& rigidity);
  // set kinematical variables from velocity
  void set_from_velocity(const double& velocity);

  // utility function
  // the lorentz factor evaluated from the particle's velocity
  const double& gamma();
  // the lorentz factor evaluated from user-inputted velocity
  const double& gamma(const double& velocity);
  // print the members of the Particle instance
  void print();

  // other member functions
  // get the value of the energy from the particle's rigidity
  double get_energy_rigidity();
};

#endif  //__PARTICLE_HPP_