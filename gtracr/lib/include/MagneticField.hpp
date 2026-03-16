// header file for Magnetic Field
#ifndef __MAGNETICFIELD_HPP_
#define __MAGNETICFIELD_HPP_

#include <math.h>

#include <array>

#include "constants.hpp"

class MagneticField {
 private:
  double B0;  // mean value of the magnetic field at the magnetic equator

 public:
  // Constructor
  MagneticField() : B0{-29404.8 * (1e-9)} {}
  /*
    Obtain the values of the magnetic field for a specified (r, theta, phi) value
    in the dipole field approximation.

    Parameters
    -----------

    - r (const double &) : the radial component
    - theta (const double &) : the polar component
    - phi (const double &) : the azimuthal component

    Returns
    --------

    - val (std::array<double, 3>) :
        array containing the magnetic field values [Br, Btheta, Bphi]
  */
  inline std::array<double, 3> values(const double& r, const double& theta, const double& phi) {
    std::array<double, 3> val;
    val[0] = 2. * (constants::RE / r) * (constants::RE / r) * (constants::RE / r) * B0 * cos(theta);
    val[1] = (constants::RE / r) * (constants::RE / r) * (constants::RE / r) * B0 * sin(theta);
    val[2] = 0.;
    return val;
  }
};

#endif  // __MAGNETICFIELD_HPP_