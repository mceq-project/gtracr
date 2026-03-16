/*
Trajectory Tracer — integrators: frozen-field RK4, Boris pusher, adaptive RK45.
*/

#include "TrajectoryTracer.hpp"

// ---------------------------------------------------------------------------
// std::array operator overloads (translation-unit local to avoid ODR issues)
// ---------------------------------------------------------------------------

namespace {

inline std::array<double, 6> operator+(std::array<double, 6> lh,
                                       std::array<double, 6> rh) {
  std::transform(lh.begin(), lh.end(), rh.begin(), lh.begin(),
                 std::plus<double>());
  return lh;
}

inline std::array<double, 6> operator*(std::array<double, 6> lh,
                                       std::array<double, 6> rh) {
  std::transform(lh.begin(), lh.end(), rh.begin(), lh.begin(),
                 std::multiplies<double>());
  return lh;
}

inline std::array<double, 6> operator*(const double lh_val,
                                       std::array<double, 6> rh) {
  std::transform(rh.cbegin(), rh.cend(), rh.begin(),
                 std::bind(std::multiplies<double>(), std::placeholders::_1, lh_val));
  return rh;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

TrajectoryTracer::TrajectoryTracer()
    : bfield_type_{'d'},
      charge_{constants::ELEMENTARY_CHARGE},
      mass_{0.938 * constants::KG_PER_GEVC2},
      start_altitude_{100. * (1e3)},
      escape_radius_{10. * constants::RE},
      stepsize_{1e-5},
      max_iter_{10000},
      particle_escaped_{false},
      solver_type_{'r'},
      atol_{1e-3},
      rtol_{1e-6},
      nsteps_{0} {}

TrajectoryTracer::TrajectoryTracer(
    double charge, double mass, double start_altitude, double escape_radius,
    double stepsize, int max_iter, const char bfield_type,
    const std::pair<std::string, double>& igrf_params,
    const char solver_type, double atol, double rtol)
    : bfield_type_{bfield_type},
      charge_{charge},
      mass_{mass},
      start_altitude_{start_altitude},
      escape_radius_{escape_radius},
      stepsize_{stepsize},
      max_iter_{max_iter},
      particle_escaped_{false},
      solver_type_{solver_type},
      atol_{atol},
      rtol_{rtol},
      nsteps_{0} {
  switch (bfield_type) {
    case 'd':
      break;  // dipole_ is default-constructed; no IGRF needed
    case 't':
      igrf_ = std::make_unique<IGRF>(igrf_params.first + "/igrf13.json",
                                     igrf_params.second);
      table_ = generate_igrf_table(*igrf_, table_params_);
      break;
    case 'i':
    default:
      bfield_type_ = 'i';
      igrf_ = std::make_unique<IGRF>(igrf_params.first + "/igrf13.json",
                                     igrf_params.second);
      break;
  }
}

// Shared-table constructor: borrows an external table pointer, no allocation.
TrajectoryTracer::TrajectoryTracer(
    const float* shared_table, const TableParams& table_params,
    double charge, double mass, double start_altitude, double escape_radius,
    double stepsize, int max_iter,
    const std::pair<std::string, double>& igrf_params,
    const char solver_type, double atol, double rtol)
    : bfield_type_{'t'},
      shared_table_ptr_{shared_table},
      table_params_{table_params},
      charge_{charge},
      mass_{mass},
      start_altitude_{start_altitude},
      escape_radius_{escape_radius},
      stepsize_{stepsize},
      max_iter_{max_iter},
      particle_escaped_{false},
      solver_type_{solver_type},
      atol_{atol},
      rtol_{rtol},
      nsteps_{0} {
  // Own IGRF for out-of-range fallback (IGRF::values writes to member state,
  // so each thread needs its own instance).
  igrf_ = std::make_unique<IGRF>(igrf_params.first + "/igrf13.json",
                                   igrf_params.second);
}

// ---------------------------------------------------------------------------
// B-field dispatch
// ---------------------------------------------------------------------------


std::array<double, 3> TrajectoryTracer::bfield_at(double r, double theta,
                                                    double phi) {
  // Guard against NaN/Inf from RK45 intermediate stages.  Return zero field
  // so the adaptive step-size controller rejects the step gracefully.
  if (!std::isfinite(r) || !std::isfinite(theta) || !std::isfinite(phi) || r <= 0.0) {
    return {0.0, 0.0, 0.0};
  }

  switch (bfield_type_) {
    case 't': {
      // Fall back to direct IGRF for r outside the table's valid range so
      // that RK45 intermediate-stage evaluations beyond 10 RE receive
      // physically correct, diminishing field values rather than the clamped
      // boundary value.
      float r_f = static_cast<float>(r);
      if (r_f < table_params_.r_min || r_f > table_params_.r_max) {
        return igrf_->values(r, theta, phi);
      }
      const float* tbl = shared_table_ptr_ ? shared_table_ptr_ : table_.data();
      auto b = table_lookup(tbl, table_params_,
                            r_f,
                            static_cast<float>(theta),
                            static_cast<float>(phi));
      return {static_cast<double>(b[0]),
              static_cast<double>(b[1]),
              static_cast<double>(b[2])};
    }
    case 'i':
      return igrf_->values(r, theta, phi);
    default:   // 'd'
      return dipole_.values(r, theta, phi);
  }
}

// ---------------------------------------------------------------------------
// ODE right-hand sides
// ---------------------------------------------------------------------------

std::array<double, 6> TrajectoryTracer::ode_lrz_bf(
    const double t, const std::array<double, 6>& vec,
    const std::array<double, 3>& bf) {

  double r = vec[0], theta = vec[1], phi = vec[2];
  double pr = vec[3], ptheta = vec[4], pphi = vec[5];

  double gmma     = lorentz_factor(pr, ptheta, pphi);
  double rel_mass = mass_ * gmma;

  double bf_r = bf[0], bf_theta = bf[1], bf_phi = bf[2];

  double dprdt_lrz    = -1. * charge_ * ((ptheta * bf_phi) - (bf_theta * pphi));
  double dprdt_sphcmp = (((ptheta * ptheta) + (pphi * pphi)) / r);
  double dprdt        = dprdt_lrz + dprdt_sphcmp;

  double dpthetadt_lrz    = charge_ * ((pr * bf_phi) - (bf_r * pphi));
  double dpthetadt_sphcmp =
      ((pphi * pphi * cos(theta)) / (r * sin(theta))) - ((pr * ptheta) / r);
  double dpthetadt = dpthetadt_lrz + dpthetadt_sphcmp;

  double dpphidt_lrz    = -1. * charge_ * ((pr * bf_theta) - (bf_r * ptheta));
  double dpphidt_sphcmp =
      ((pr * pphi) / r) + ((ptheta * pphi * cos(theta)) / (r * sin(theta)));
  double dpphidt = dpphidt_lrz - dpphidt_sphcmp;

  std::array<double, 6> result = {{
      pr, (ptheta / r), (pphi / (r * sin(theta))), dprdt, dpthetadt, dpphidt}};
  return (1. / rel_mass) * result;
}

// ode_lrz evaluates the B-field internally; used by RK45 stages.
std::array<double, 6> TrajectoryTracer::ode_lrz(const double t,
                                                  const std::array<double, 6>& vec) {
  return ode_lrz_bf(t, vec, bfield_at(vec[0], vec[1], vec[2]));
}

inline double TrajectoryTracer::lorentz_factor(const double& pr,
                                               const double& ptheta,
                                               const double& pphi) {
  double pmag = sqrt((pr * pr) + (ptheta * ptheta) + (pphi * pphi));
  double pm_ratio = pmag / (mass_ * constants::SPEED_OF_LIGHT);
  return sqrt(1. + (pm_ratio * pm_ratio));
}

// ---------------------------------------------------------------------------
// Dispatcher: evaluate()
// ---------------------------------------------------------------------------

void TrajectoryTracer::evaluate(const double& t0, std::array<double, 6>& vec0) {
  TrajRecorder rec(false);
  switch (solver_type_) {
    case 'b': run_boris<false>(t0, vec0, rec); break;
    case 'a': run_rk45<false>(t0, vec0, rec);  break;
    default:  run_rk4<false>(t0, vec0, rec);   break;
  }
}

// ---------------------------------------------------------------------------
// Dispatcher: evaluate_and_get_trajectory()
// ---------------------------------------------------------------------------

std::map<std::string, std::vector<double>>
TrajectoryTracer::evaluate_and_get_trajectory(double& t0,
                                               std::array<double, 6>& vec0) {
  TrajRecorder rec(true);
  // For adaptive RK45, estimate far fewer steps than max_iter_.
  int reserve_hint = (solver_type_ == 'a') ? std::max(100, max_iter_ / 10)
                                           : max_iter_;
  rec.reserve(reserve_hint);

  switch (solver_type_) {
    case 'b': run_boris<true>(t0, vec0, rec); break;
    case 'a': run_rk45<true>(t0, vec0, rec);  break;
    default:  run_rk4<true>(t0, vec0, rec);   break;
  }
  return make_traj_map(rec.t, rec.r, rec.theta, rec.phi, rec.pr, rec.ptheta, rec.pphi);
}

// Helper to build the return map.
std::map<std::string, std::vector<double>>
TrajectoryTracer::make_traj_map(
    std::vector<double>& t_arr,     std::vector<double>& r_arr,
    std::vector<double>& theta_arr, std::vector<double>& phi_arr,
    std::vector<double>& pr_arr,    std::vector<double>& ptheta_arr,
    std::vector<double>& pphi_arr) {
  return {{"t", t_arr},     {"r", r_arr},   {"theta", theta_arr},
          {"phi", phi_arr}, {"pr", pr_arr}, {"ptheta", ptheta_arr},
          {"pphi", pphi_arr}};
}

// ---------------------------------------------------------------------------
// find_cutoff_rigidity — scans rigidities for a single direction
// ---------------------------------------------------------------------------

double TrajectoryTracer::find_cutoff_rigidity(
    const std::array<double, 3>& pos,
    const std::array<double, 3>& mom_unit,
    const std::vector<double>& rigidities,
    double mom_factor) {
  for (double rig : rigidities) {
    double mom_si = rig * mom_factor;
    std::array<double, 6> vec0 = {{
        pos[0], pos[1], pos[2],
        mom_unit[0] * mom_si, mom_unit[1] * mom_si, mom_unit[2] * mom_si}};
    particle_escaped_ = false;
    nsteps_ = 0;
    evaluate(0.0, vec0);
    if (particle_escaped_) return rig;
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// find_cutoff_rigidity_bisect — binary search variant
// ---------------------------------------------------------------------------

double TrajectoryTracer::find_cutoff_rigidity_bisect(
    const std::array<double, 3>& pos,
    const std::array<double, 3>& mom_unit,
    const std::vector<double>& rigidities_asc,
    double mom_factor) {
  if (rigidities_asc.empty()) return 0.0;

  auto test_escape = [&](int idx) -> bool {
    double rig = rigidities_asc[idx];
    double mom_si = rig * mom_factor;
    std::array<double, 6> vec0 = {{
        pos[0], pos[1], pos[2],
        mom_unit[0] * mom_si, mom_unit[1] * mom_si, mom_unit[2] * mom_si}};
    particle_escaped_ = false;
    nsteps_ = 0;
    evaluate(0.0, vec0);
    return particle_escaped_;
  };

  int lo = 0;                                             // lowest rigidity
  int hi = static_cast<int>(rigidities_asc.size()) - 1;  // highest rigidity

  // Fast path: if min rigidity already escapes, cutoff ≤ min
  if (test_escape(lo)) return rigidities_asc[lo];
  // If max rigidity doesn't escape, cutoff > max
  if (!test_escape(hi)) return 0.0;

  // Invariant: rigidities_asc[lo] forbidden, rigidities_asc[hi] escapes
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (test_escape(mid))
      hi = mid;
    else
      lo = mid;
  }

  return rigidities_asc[hi];
}

// ---------------------------------------------------------------------------
// Frozen-field RK4 (templated on Record)
//
// B-field is evaluated once per step at the step-start position and held
// fixed across all four RK4 stages. This gives O(h^4) position accuracy with
// an O(h^2) frozen-field truncation error — acceptable for the step sizes used
// in GMRC (dt ≤ 1e-5 s). Both evaluate() and evaluate_and_get_trajectory()
// use the same integrator so results are consistent regardless of get_data.
// ---------------------------------------------------------------------------

template <bool Record>
void TrajectoryTracer::run_rk4(const double& t0, std::array<double, 6>& vec0,
                                TrajRecorder& rec) {
  double h = stepsize_;
  double t = t0;
  std::array<double, 6> vec = vec0;

  for (int i = 0; i < max_iter_; ++i) {
    if (Record) rec.record(t, vec);

    std::array<double, 3> bf = bfield_at(vec[0], vec[1], vec[2]);

    std::array<double, 6> k1 = h * ode_lrz_bf(t,           vec,              bf);
    std::array<double, 6> k2 = h * ode_lrz_bf(t + 0.5*h,   vec + (0.5*k1),   bf);
    std::array<double, 6> k3 = h * ode_lrz_bf(t + 0.5*h,   vec + (0.5*k2),   bf);
    std::array<double, 6> k4 = h * ode_lrz_bf(t + h,        vec + h*k3,        bf);
    vec = vec + (1./6.) * (k1 + (2.*k2) + (2.*k3) + k4);
    t  += h;
    ++nsteps_;

    const double r = vec[0];
    if (r > escape_radius_) { particle_escaped_ = true; break; }
    if (r < start_altitude_ + constants::RE)              break;
  }
  final_time_      = t;
  final_sixvector_ = vec;
}

// ---------------------------------------------------------------------------
// Boris pusher (templated on Record)
//
// Relativistic Boris algorithm in Cartesian coordinates:
//   1. Convert state (r,θ,φ,p_r,p_θ,p_φ) → Cartesian (x,y,z,px,py,pz)
//   2. Get B-field in Cartesian (1 evaluation per step)
//   3. Apply Boris rotation: preserves |p| exactly (symplectic, 2nd-order)
//   4. Advance position with updated velocity
//   5. Convert back to spherical
//
// Sign convention: uses −charge_ for the Lorentz force (backtracking mode),
// consistent with the existing ode_lrz_bf implementation.
// ---------------------------------------------------------------------------

template <bool Record>
void TrajectoryTracer::run_boris(const double& t0, std::array<double, 6>& vec0,
                                  TrajRecorder& rec) {
  const double c2  = constants::SPEED_OF_LIGHT * constants::SPEED_OF_LIGHT;
  const double mc2 = mass_ * mass_ * c2;

  double h = stepsize_;
  double t = t0;
  std::array<double, 6> vec = vec0;

  for (int i = 0; i < max_iter_; ++i) {
    if (Record) rec.record(t, vec);

    const double r = vec[0], theta = vec[1], phi = vec[2];
    const double pr = vec[3], ptheta = vec[4], pphi = vec[5];

    const double sT = sin(theta), cT = cos(theta);
    const double sP = sin(phi),   cP = cos(phi);

    const double x = r * sT * cP;
    const double y = r * sT * sP;
    const double z = r * cT;

    // ê_r = (sT·cP, sT·sP, cT), ê_θ = (cT·cP, cT·sP, −sT), ê_φ = (−sP, cP, 0)
    const double px = pr*sT*cP + ptheta*cT*cP - pphi*sP;
    const double py = pr*sT*sP + ptheta*cT*sP + pphi*cP;
    const double pz = pr*cT             - ptheta*sT;

    const auto bf_sph = bfield_at(r, theta, phi);
    const double Br = bf_sph[0], Bt = bf_sph[1], Bp = bf_sph[2];
    const double Bx = Br*sT*cP + Bt*cT*cP - Bp*sP;
    const double By = Br*sT*sP + Bt*cT*sP + Bp*cP;
    const double Bz = Br*cT             - Bt*sT;

    // Boris rotation: t = −q·B·dt / (2·γ·m)   [−q for backtracking]
    const double gamma = sqrt(1.0 + (px*px + py*py + pz*pz) / mc2);
    const double fac = -charge_ * h / (2.0 * gamma * mass_);
    const double tx = fac * Bx, ty = fac * By, tz = fac * Bz;

    const double t2   = tx*tx + ty*ty + tz*tz;
    const double sfac = 2.0 / (1.0 + t2);
    const double sx = sfac*tx, sy = sfac*ty, sz = sfac*tz;

    const double ppx = px + (py*tz - pz*ty);
    const double ppy = py + (pz*tx - px*tz);
    const double ppz = pz + (px*ty - py*tx);

    const double px_n = px + (ppy*sz - ppz*sy);
    const double py_n = py + (ppz*sx - ppx*sz);
    const double pz_n = pz + (ppx*sy - ppy*sx);

    const double gamma_n = sqrt(1.0 + (px_n*px_n + py_n*py_n + pz_n*pz_n) / mc2);
    const double vfac = h / (gamma_n * mass_);
    const double x_n = x + px_n * vfac;
    const double y_n = y + py_n * vfac;
    const double z_n = z + pz_n * vfac;

    const double r_n = sqrt(x_n*x_n + y_n*y_n + z_n*z_n);
    const double theta_n = acos(z_n / r_n);
    const double phi_n   = atan2(y_n, x_n);
    const double sT_n = sin(theta_n), cT_n = cos(theta_n);
    const double sP_n = sin(phi_n),   cP_n = cos(phi_n);

    const double pr_n     =  px_n*sT_n*cP_n + py_n*sT_n*sP_n + pz_n*cT_n;
    const double ptheta_n =  px_n*cT_n*cP_n + py_n*cT_n*sP_n - pz_n*sT_n;
    const double pphi_n   = -px_n*sP_n       + py_n*cP_n;

    vec = {r_n, theta_n, phi_n, pr_n, ptheta_n, pphi_n};
    t  += h;
    ++nsteps_;

    if (r_n > escape_radius_)                      { particle_escaped_ = true; break; }
    if (r_n < start_altitude_ + constants::RE)       break;
  }
  final_time_      = t;
  final_sixvector_ = vec;
}

// ---------------------------------------------------------------------------
// Adaptive RK45 (Dormand-Prince / DOPRI5) (templated on Record)
//
// Evaluates B-field at each of the 7 stages (no frozen-field approximation)
// for accurate error estimates. Uses FSAL: k7 of step n is k1 of step n+1.
//
// Step-size control: h_new = h · clamp(S·ε^{−1/5}, 0.1, 5)
//   where ε = RMS of component-wise scaled error.
//
// Dormand-Prince error coefficients (e = b5 − b4):
//   e1 = 71/57600,  e3 = −71/16695,  e4 = 71/1920,
//   e5 = −17253/339200,  e6 = 22/525,  e7 = −1/40
// ---------------------------------------------------------------------------

template <bool Record>
void TrajectoryTracer::run_rk45(const double& t0, std::array<double, 6>& vec0,
                                  TrajRecorder& rec) {
  // Dormand-Prince Butcher tableau
  constexpr double A21 = 1./5.;
  constexpr double A31 = 3./40.,       A32 = 9./40.;
  constexpr double A41 = 44./45.,      A42 = -56./15.,     A43 = 32./9.;
  constexpr double A51 = 19372./6561., A52 = -25360./2187., A53 = 64448./6561., A54 = -212./729.;
  constexpr double A61 = 9017./3168.,  A62 = -355./33.,    A63 = 46732./5247.,
                   A64 = 49./176.,     A65 = -5103./18656.;
  // 5th-order weights (also the FSAL advance)
  constexpr double B1 = 35./384., B3 = 500./1113., B4 = 125./192.,
                   B5 = -2187./6784., B6 = 11./84.;
  // Error = b5 − b4
  constexpr double E1 =  71./57600.,   E3 = -71./16695.,  E4 =  71./1920.,
                   E5 = -17253./339200., E6 = 22./525.,   E7 = -1./40.;

  constexpr double SAFETY = 0.9, MAX_FAC = 2.0, MIN_FAC = 0.1;
  constexpr double ERR_EXP = -0.2;  // −1/5
  // Cap step size so intermediate RK45 stages stay within IGRF table range.
  // Without this cap, large h causes every stage to land outside [1 RE,10 RE],
  // forcing slow direct-IGRF fallback for all 7 evaluations per step.
  // MAX_FAC=2 (reduced from 5) prevents aggressive step-growth overshoot that
  // causes a high rejection rate and millions of wasted iterations per trajectory.
  constexpr double MAX_STEP = 1e-2;  // 10 ms

  double h = stepsize_;
  double t = t0;
  std::array<double, 6> vec = vec0;

  // Compute k1 once; FSAL reuses it as k1 of the next step.
  std::array<double, 6> k1 = ode_lrz(t, vec);

  int accepted_steps = 0;
  // Hard upper bound on total iterations (accepted + rejected) to prevent
  // indefinite looping when the step-size controller oscillates pathologically.
  // With sin_theta clamping this limit is rarely approached; it is a safety net.
  const int max_total_iters = max_iter_ * 10;
  int total_iters = 0;
  int nan_streak  = 0;  // consecutive NaN error estimates

  while (accepted_steps < max_iter_) {
    if (++total_iters > max_total_iters) break;
    // Clamp step to avoid overshooting max_time (approximated by max_iter_*stepsize_).
    const double t_end = t0 + max_iter_ * stepsize_;
    if (t + h > t_end) h = t_end - t;
    if (!(h > 0.0)) break;  // catches h<=0 AND h=NaN

    std::array<double, 6> k2 = ode_lrz(t + h*1./5.,
        vec + (h*A21)*k1);
    std::array<double, 6> k3 = ode_lrz(t + h*3./10.,
        vec + (h*A31)*k1 + (h*A32)*k2);
    std::array<double, 6> k4 = ode_lrz(t + h*4./5.,
        vec + (h*A41)*k1 + (h*A42)*k2 + (h*A43)*k3);
    std::array<double, 6> k5 = ode_lrz(t + h*8./9.,
        vec + (h*A51)*k1 + (h*A52)*k2 + (h*A53)*k3 + (h*A54)*k4);
    std::array<double, 6> k6 = ode_lrz(t + h,
        vec + (h*A61)*k1 + (h*A62)*k2 + (h*A63)*k3 + (h*A64)*k4 + (h*A65)*k5);

    // 5th-order solution (also used as the advance)
    std::array<double, 6> y_new = vec +
        (h*B1)*k1 + (h*B3)*k3 + (h*B4)*k4 + (h*B5)*k5 + (h*B6)*k6;

    // k7 = f(y_new) — needed for error estimate and FSAL
    std::array<double, 6> k7 = ode_lrz(t + h, y_new);

    // Error estimate (RMS of scaled component errors)
    double err_sq = 0.0;
    for (int j = 0; j < 6; ++j) {
      double err_j = h * (E1*k1[j] + E3*k3[j] + E4*k4[j] +
                          E5*k5[j] + E6*k6[j] + E7*k7[j]);
      double sc = atol_ + rtol_ * std::max(std::abs(vec[j]), std::abs(y_new[j]));
      err_sq += (err_j / sc) * (err_j / sc);
    }
    double err_norm = sqrt(err_sq / 6.0);

    if (!(err_norm >= 0.0)) {  // NaN/Inf guard
      if (++nan_streak > 20) break;  // state is unrecoverable; abort trajectory
      h *= MIN_FAC; continue;
    }
    nan_streak = 0;

    double fac = SAFETY * pow(err_norm, ERR_EXP);
    fac = std::min(MAX_FAC, std::max(MIN_FAC, fac));

    if (err_norm <= 1.0) {
      if (Record) rec.record(t, vec);
      vec = y_new;
      t  += h;
      k1  = k7;   // FSAL
      ++accepted_steps;
      ++nsteps_;

      const double r = vec[0];
      if (r > escape_radius_) { particle_escaped_ = true; break; }
      if (r < start_altitude_ + constants::RE)              break;
    }
    h *= fac;
    if (h > MAX_STEP) h = MAX_STEP;  // keep intermediate stages in IGRF table range
  }

  final_time_      = t;
  final_sixvector_ = vec;
}
