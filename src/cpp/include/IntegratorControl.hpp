/*
 * IntegratorControl — integrator-independent run control for TrajectoryTracer:
 *
 *  1. Rigidity-aware escape criterion. A back-traced trajectory is allowed when
 *     it reaches r_esc = max(escape_radius, k * C_St(p)) within the time budget
 *     t_max = max(max_time, N * r_esc / (beta c)). C_St = RE sqrt(B0 RE |q| / p)
 *     is the Störmer length and B0 = sqrt(g10^2 + g11^2 + h11^2) the dipole
 *     strength of the active field model. k = N = 0 gives the fixed criterion.
 *
 *  2. RK45 error norm on the dimensionless state: position errors in units of
 *     RE with tolerance atol + rtol * r / RE, momentum errors in units of |p|
 *     with tolerance atol + rtol. Both are applied to the error *vectors*, so
 *     the norm is rotation invariant and identical whether the integrator
 *     advances Cartesian or spherical components.
 */
#ifndef __INTEGRATORCONTROL_HPP_
#define __INTEGRATORCONTROL_HPP_

#include <algorithm>
#include <cmath>

#include "constants.hpp"

namespace integrator_control {

constexpr double DEFAULT_ESCAPE_FACTOR = 4.0;
constexpr double DEFAULT_TIME_FACTOR = 80.0;
// RK45 tolerances (dimensionless, see rk45_error_norm)
constexpr double DEFAULT_ATOL = 0.0;
constexpr double DEFAULT_RTOL = 1e-7;

struct RunLimits {
  double escape_radius;  // m
  double max_time;       // s
  int max_iter;          // steps of size stepsize
};

// Dipole strength (tesla) from the radial field on the sphere r = RE:
//   g10 = 3/(8 pi) int B_r cos(theta) dOmega, g11/h11 with sin(theta) cos/sin(phi).
// br(theta, phi) returns B_r at r = RE. Gauss-Legendre in cos(theta) (8 nodes) x
// 32 uniform phi nodes is exact for spherical-harmonic degree <= 13.
template <class BrFn>
double dipole_b0(BrFn br) {
  constexpr double kPi = 3.14159265358979323846;
  static const double x[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                              -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
                              0.7966664774136267,  0.9602898564975363};
  static const double w[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                              0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                              0.2223810344533745, 0.1012285362903763};
  const int nphi = 32;
  const double dphi = 2. * kPi / nphi;
  double g10 = 0., g11 = 0., h11 = 0.;
  for (int i = 0; i < 8; ++i) {
    const double ct = x[i], st = std::sqrt(1. - ct * ct), theta = std::acos(ct);
    for (int j = 0; j < nphi; ++j) {
      const double phi = (j + 0.5) * dphi;
      const double b = br(theta, phi) * w[i] * dphi;
      g10 += b * ct;
      g11 += b * st * std::cos(phi);
      h11 += b * st * std::sin(phi);
    }
  }
  return 3. / (8. * kPi) * std::sqrt(g10 * g10 + g11 * g11 + h11 * h11);
}

// Störmer length (m) for dipole strength b0 (T), charge (C) and momentum p (kg m/s).
inline double stormer_length(double b0, double charge, double p) {
  return constants::RE * std::sqrt(b0 * constants::RE * std::abs(charge) / p);
}

// Escape radius, time budget and step budget of one run with momentum p.
inline RunLimits run_limits(double escape_radius, int max_iter, double stepsize,
                            double escape_factor, double time_factor, double b0, double charge,
                            double mass, double p) {
  RunLimits L{escape_radius, max_iter * stepsize, max_iter};
  if (!(p > 0.) || (escape_factor <= 0. && time_factor <= 0.)) return L;
  if (escape_factor > 0.)
    L.escape_radius = std::max(escape_radius, escape_factor * stormer_length(b0, charge, p));
  if (time_factor > 0.) {
    const double mc = mass * constants::SPEED_OF_LIGHT;
    const double beta = p / std::sqrt(p * p + mc * mc);
    const double t_need = time_factor * L.escape_radius / (beta * constants::SPEED_OF_LIGHT);
    if (t_need > L.max_time) {
      L.max_time = t_need;
      // the RK45 guard multiplies the step budget by 10; stay well inside int
      L.max_iter = static_cast<int>(std::min(std::ceil(t_need / stepsize), 2.0e8));
    }
  }
  return L;
}

// RMS-equivalent RK45 error norm (<= 1 accepts the step).
//   pos_err2: |delta x|^2 (m^2), mom_err2: |delta p|^2, r: radius (m), p: |p|.
inline double rk45_error_norm(double pos_err2, double mom_err2, double r, double p, double atol,
                              double rtol) {
  const double tol_pos = (atol + rtol * r / constants::RE) * constants::RE;
  const double tol_mom = (atol + rtol) * p;
  return std::sqrt((pos_err2 / (tol_pos * tol_pos) + mom_err2 / (tol_mom * tol_mom)) / 6.0);
}

}  // namespace integrator_control

#endif  // __INTEGRATORCONTROL_HPP_
