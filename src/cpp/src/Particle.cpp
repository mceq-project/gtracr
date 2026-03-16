/*
Utility class for cosmic ray particles
Members:
- name: the name of the particle (char*)
- pid: the particle id as in pdg (int)
- mass: the particle rest mass (double) [units of GeV / c^2]
- charge: particle's charge Z (int) [units of elementary charge]
- label: the shorthand name for the particle (char*)

The below members require the particle energy or one of the below members as
additional information
- momentum: Particle momentum
- velocity: particle velocity
- rigidity: particle rigidity

*Default is set to Proton

Notes:
- PDGID obtained from here: http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
- The mass of the particles are also obtained from PDG

Example:
proton: proton = Particle("Proton", 2212, 0.938272, "p+")
*/
#include "Particle.hpp"

#include <math.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "constants.hpp"

// Constructors
// Default constructor, set to proton
// Default energy set to 1GeV
Particle::Particle() : nm{"Proton"}, pid{2212}, m{0.938272}, ch{1}, lbl{"p+"} {
  p = sqrt(m * m + 1.);
  R = (p) / abs(ch);
  v = (p * constants::SPEED_OF_LIGHT) /
      sqrt((m * constants::SPEED_OF_LIGHT * m * constants::SPEED_OF_LIGHT) + p * p) *
      constants::SPEED_OF_LIGHT;
}

// Construct with given initial configurations
// default energy set to 1GeV
Particle::Particle(const std::string& name, const int pdgid, const double& mass, const int charge,
                   const std::string& label)
    : nm{name}, pid{pdgid}, m{mass}, ch{charge}, lbl{label} {
  p = sqrt(m * m + 1.);
  R = (p) / abs(ch);
  v = (p * constants::SPEED_OF_LIGHT) /
      sqrt((m * constants::SPEED_OF_LIGHT * m * constants::SPEED_OF_LIGHT) + p * p) *
      constants::SPEED_OF_LIGHT;
}

// Constructor with some provided energy
Particle::Particle(const std::string& name, const int pdgid, const double& mass, const int charge,
                   const std::string& label, const double& energy = 0., const double& rigidity = 0.)
    : nm{name}, pid{pdgid}, m{mass}, ch{charge}, lbl{label} {
  // set kinematical variables if energy xor rigidity is given
  if (abs(energy) < 1e-10) {
    set_from_energy(energy);
  } else if (abs(rigidity) < 1e-10) {
    set_from_rigidity(rigidity);
  } else {
    throw std::runtime_error("Input energy or rigidity, but not both or none!");
  }
}

// Destructor
Particle::~Particle() {}

// copy constructor
Particle::Particle(const Particle& part)
    : nm{part.nm}, pid{part.pid}, m{part.m}, ch{part.ch}, lbl{part.lbl} {
  p = part.p;
  v = part.v;
  R = part.R;
}

// copy assignment operator
Particle& Particle::operator=(const Particle& part) {
  nm = part.nm;
  pid = part.pid;
  m = part.m;
  ch = part.ch;
  lbl = part.lbl;

  p = part.p;
  v = part.v;
  R = part.R;
  return *this;
}

// Lorentz factor
const double& Particle::gamma() {
  return 1. / sqrt(1. - (v / constants::SPEED_OF_LIGHT) * (v / constants::SPEED_OF_LIGHT));
}

const double& Particle::gamma(const double& vel) {
  return 1. / sqrt(1. - (vel / constants::SPEED_OF_LIGHT) * (vel / constants::SPEED_OF_LIGHT));
}

// setters
void Particle::set_from_energy(const double& energy) {
  p = sqrt(energy * energy - m * m);
  R = (p) / abs(ch);
  v = (p * constants::SPEED_OF_LIGHT) /
      sqrt((m * constants::SPEED_OF_LIGHT * m * constants::SPEED_OF_LIGHT) + p * p) *
      constants::SPEED_OF_LIGHT;
}

void Particle::set_from_momentum(const double& _p) {
  p = _p;
  R = (p) / abs(ch);
  v = (p * constants::SPEED_OF_LIGHT) /
      sqrt((m * constants::SPEED_OF_LIGHT * m * constants::SPEED_OF_LIGHT) + p * p) *
      constants::SPEED_OF_LIGHT;
}

void Particle::set_from_rigidity(const double& _R) {
  p = (_R * abs(ch));
  R = _R;
  v = (p * constants::SPEED_OF_LIGHT) /
      sqrt((m * constants::SPEED_OF_LIGHT * m * constants::SPEED_OF_LIGHT) + p * p) *
      constants::SPEED_OF_LIGHT;
}

void Particle::set_from_velocity(const double& _v) {
  p = gamma(_v) * m * _v;
  v = _v;
  R = (p) / abs(ch);
}

// other member functions
// obtain energy from rigidity
double Particle::get_energy_rigidity() {
  double enrgy = sqrt(R * abs(ch) * R * abs(ch) + m * m) + m;
  return enrgy;
}

// print contents
void Particle::print() {
  std::cout << "Particle: " << nm << " (" << lbl << "), "
            << "PDG ID: " << pid << ", "
            << "Mass [GeV/c^2]: " << m << ", "
            << "Charge [e]: " << ch << std::endl;
  std::cout << "Current Momentum [GeV/c]: " << p
            << ", "
            //   << "Current Velocity: " << v << ", "
            << "Current Rigidity [GV]: " << R << std::endl;
}