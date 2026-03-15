/*
Trajectory Tracer class that traces the trajectory of the particle
by performing a 4th-order Runge Kutta numerical integration algorithm.
*/
#ifndef __TRAJECTORYTRACER_HPP_
#define __TRAJECTORYTRACER_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "MagneticField.hpp"
#include "constants.hpp"
#include "igrf.hpp"

class TrajectoryTracer {
  /*
Trajectory Tracer class that traces the trajectory of the particle
by performing a 4th-order Runge Kutta numerical integration algorithm.

Class Members
--------------
  - bfield_ (MagneticField instance):
      The MagneticField object that contains the information pertaining the
      Earth's magnetic field model (default is dipole model).
  - charge_ (double) :
      The charge of the particle in units of Coulombs (default 1.602e-19 C, i.e.
  1e).
  - mass_ (double) :
      The (rest) mass of the particle in units of kilograms
  (default 1.672e-27 kg, i.e. proton mass).
  - escape_radius_ (double) :
      The radial distance relative to Earth's center in which the particle is
      set to have "escaped" Earth, i.e. a valid cosmic ray coming from some
      astrophysical source. Units are in kilometers (default 10*Earth's radius)
  - stepsize_ (double) :
      The stepsize for the integration process (default 1e-5)
  - max_iter_ (int) :
      The maximum number of iterations performed for the integration process
      (default 10000)
  - particle_escaped_ (bool) :
      True if particle has effectively "escaped" from Earth (i.e. when the
      radial component of the trajectory > escape_radius_) (default false).
*/
 private:
  MagneticField bfield_;  // the magnetic field
  double charge_;         // charge of the particle in coulumbs
  double mass_;           // mass of the particle in kg
  double start_altitude_;   // starting altitude of particle
  double escape_radius_;  // radius in which we set for particle to escape
  double stepsize_;       // step size
  int max_iter_;          // maximum step size
  // binary value to store if particle has escaped or not
  bool particle_escaped_;

  double final_time_;  // the final time of the trajectory
  std::array<double, 6>
      final_sixvector_;  // the final six-vector of the trajectory

  // SixVector objects to store each RK parameter
//   SixVector k1_vec;
//   SixVector k2_vec;
//   SixVector k3_vec;
//   SixVector k4_vec;
//   SixVector k_vec;



  /* The ordinary differential equations that describes the motion
of charge particles in Earth's magnetic field via the Lorentz force
in spherical coordinates.

Parameters
-----------
- t (double) :
    the time
- vec (std::array<double, 6>) :
     the six-vector (r, theta, phi, pr, ptheta, pphi) at time t

Returns
--------
- ode_lrz (std::array<double, 6>) :
     the ordinary differential equation for the six vector based on the
Lorentz force equation
*/
  std::array<double, 6> ode_lrz(const double t, const std::array<double, 6> &vec);

  // ode_lrz variant that accepts a pre-evaluated B-field (avoids a second
  // bfield_.values() call when the field is frozen across RK4 sub-steps).
  std::array<double, 6> ode_lrz_bf(const double t,
                                    const std::array<double, 6> &vec,
                                    const std::array<double, 3> &bf);

 public:
  /* Default Constructor for TrajectoryTracer class

  Creates an instance of the TrajectoryTracer, that is, the object that keeps
  track of a single particle trajectory in Earth's magnetic field.

  The default constructor initializes the object with the default values
  provided for the members.

  Parameters
  ------------
  None
*/
  TrajectoryTracer();
  /* Constructor for TrajectoryTracer class

  Creates an instance of the TrajectoryTracer, that is, the object that keeps
  track of a single particle trajectory in Earth's magnetic field.

  This constructor requires 2 parameters, and the rest may be optional.

  Required Parameters
  -------------------
  - charge (int) :
        The charge of the particle in units of electrons.
  - mass (double) :
        The mass of the particle in units of GeV.

  Optional Parameters
  -------------------
  - start_altitude (double) :
        The starting altitude of the particle, i.e. the altitude in which
        the cosmic ray has collided with the atmosphere (default 100km)
  - escape_radius (double) :
        The radius in which the particle has "escaped" relative to
        Earth's center in units of km (default 10*RE)
  - stepsize (double) :
        The step size of the integration (default 1e-5)
  - max_iter (int) :
        The maximum number of iterations performed in the integration process
        (default 10000)
  - bfield_type (char) :
        The type of Magnetic Field to evaluate the trajectory with. Only types
        'd' (for dipole model) or 'i' (IGRF model) are allowed (default 'd').
  - igrf_params (std::pair<std::string, double>) :
        Parameters required for instantiating the IGRF model. The first entry
        contains the path of the directory in which the .COF data file is
        stored (default to the author's path to the directory). The second entry
        contains the date in which the evaluation of the trajectory is requested
        in decimal date (default 2020.).
 */
  TrajectoryTracer( double charge,  
                  double mass = 1.67e-27,
                   double start_altitude = 100. * (1e3),
                    double escape_radius = 10. * constants::RE,
                    double stepsize = 1e-5,  int max_iter = 10000,
                   const char bfield_type = 'i',
                   const std::pair<std::string, double> &igrf_params = {
                       "/home/keito/devel/gtracr/data", 2020.});

  // Reset state between reuses of the same tracer (clears particle_escaped_
  // so the object can be called with evaluate() for a new rigidity without
  // constructing a fresh TrajectoryTracer and reloading igrf13.json).
  void reset() { particle_escaped_ = false; }

  /* the charge of the particle associated with the Runge-Kutta
   integrator */
  const double &charge() { return charge_; }
  // the mass of the particle associated with the Runge-Kutta integrator
  const double &mass() { return mass_; }
  // starting altitude of the particle
  const double &start_altitude() { return start_altitude_; }
  // the escape radius of the tracer
  const double &escape_radius() { return escape_radius_; }
  // the step size of the Runge-Kutta integrator
  const double &stepsize() { return stepsize_; }
  // the maximum number of steps of the integrator
  int max_iter() { return max_iter_; }
  // the boolean if particle has escaped or not
  bool particle_escaped() { return particle_escaped_; }
  // the final time of the trajectory
  const double &final_time() { return final_time_; }
  // the final sixvector of the trajectory
  const std::array<double, 6> &final_sixvector() { return final_sixvector_; }

  /*
  The differential equation for the momentum in
  the radial component (dprdt)

  Parameters
  -----------
  - r, theta, phi (double):
        the coordinates at time t
  - br, btheta, bphi (double):
        the magnetic field components at time t
  */

  /*
  Returns the lorentz factor, evaluated from the momentum

  Parameters
  ----------
  - pr (const double &) :
        the momentum in the radial component
  - ptheta (const double &) :
        the momentum in the polar direction
  - pphi (const double &) :
        the momentum in the azimuthal direction

  Returns
  -------
  - gamma (const double) :
        The lorentz factor for the particular momentum
  */
  inline double lorentz_factor(const double &pr, const double &ptheta,
                               const double &pphi);
  /* Evaluates the trajectory of the particle using a 4th-order Runge Kutta
  algorithm.

  Parameters
  -----------
  - t0 (double) :
      the initial time in seconds
  - vec0 (std::array<double, 6>) :
       the initial six-vector (r0, theta0, phi0, pr0, ptheta0, pphi0) at time t0

  Returns
  --------
  None

  */
  void evaluate(const double &t0, std::array<double, 6> &vec0);

  /* Evaluates the trajectory of the particle using a 4th-order Runge Kutta
  algorithm and return a map that contains the information of the particle
  trajectory. This will most often be used for debugging purposes to see the
  actual trajectory.

  Parameters
  -----------
  - t0 (double) :
      the initial time in seconds
  - vec0 (std::array<double, 6>) :
       the initial six-vector (r0, theta0, phi0, pr0, ptheta0, pphi0) at time t0

  Returns
  --------
  - trajectory_data (std::map<std::string, std::vector<double> >) :
      the trajectory information, that is, the time array of the trajectory
      and the six-vector of the trajectory in std::vectors.
      Notes:
      - keys are ["t", "r", "theta", "phi", "pr", "ptheta", "pphi"]
      - the final point of the trajectory is also contained in the map
      - std::vectors (dynamic arrays) are used since the length of each
        trajectory is not known at compile time.

  */
  std::map<std::string, std::vector<double>> evaluate_and_get_trajectory(
      double &t0, std::array<double, 6> &vec0);
};

#endif  //__TRAJECTORYTRACER_HPP_