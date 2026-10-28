/*
Trajectory Tracer class that traces the trajectory of the particle
by performing numerical integration of the relativistic Lorentz force ODE.
*/
#ifndef __TRAJECTORYTRACER_HPP_
#define __TRAJECTORYTRACER_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "IntegratorControl.hpp"
#include "MagneticField.hpp"
#include "constants.hpp"
#include "igrf.hpp"
#include "igrf_table.hpp"

class TrajectoryTracer {
 private:
  // B-field dispatch: 'd' = dipole, 'i' = IGRF direct, 't' = IGRF table
  char bfield_type_;
  MagneticField dipole_;                    // always valid (trivial ctor)
  std::unique_ptr<IGRF> igrf_;              // non-null for 'i' and 't'
  std::vector<float> table_;                // non-empty for 't' (owned table)
  const float* shared_table_ptr_{nullptr};  // non-null when borrowing external table
  TableParams table_params_;                // valid for 't'

  // Evaluate B-field at (r, theta, phi) using the active field model.
  std::array<double, 3> bfield_at(double r, double theta, double phi);

  double charge_;          // charge of the particle in coulombs
  double mass_;            // mass of the particle in kg
  double start_altitude_;  // starting altitude of particle
  double escape_radius_;   // radius in which we set for particle to escape
  double stepsize_;        // step size
  int max_iter_;           // maximum number of steps
  bool particle_escaped_;  // true if particle has escaped

  char solver_type_;  // 'r' = RK4 (frozen-field), 'b' = Boris, 'a' = adaptive RK45
  double atol_;       // absolute tolerance (RK45 only)
  double rtol_;       // relative tolerance (RK45 only)
  int nsteps_;        // number of accepted integration steps taken

  double final_time_;
  std::array<double, 6> final_sixvector_;

  // Rigidity-aware escape criterion (IntegratorControl.hpp, set_escape_scaling).
  double escape_factor_{integrator_control::DEFAULT_ESCAPE_FACTOR};
  double time_factor_{integrator_control::DEFAULT_TIME_FACTOR};
  double dipole_b0_{-1.};  // tesla; computed lazily from the field model
  integrator_control::RunLimits run_limits_{0., 0., 0};
  void prepare_run(const std::array<double, 6>& vec0);

  // --------------------------------------------------------------------------
  // Optional trajectory recorder passed to the run_*<Record> templates.
  // When Record=false the compiler eliminates all if(Record) blocks; when
  // Record=true every accepted step is appended to the vectors.
  // --------------------------------------------------------------------------
  struct TrajRecorder {
    std::vector<double> t, r, theta, phi, pr, ptheta, pphi;
    bool active;

    explicit TrajRecorder(bool a) : active(a) {}

    void record(double t_, const std::array<double, 6>& v) {
      if (active) {
        t.push_back(t_);
        r.push_back(v[0]);
        theta.push_back(v[1]);
        phi.push_back(v[2]);
        pr.push_back(v[3]);
        ptheta.push_back(v[4]);
        pphi.push_back(v[5]);
      }
    }

    void reserve(std::size_t n) {
      if (active) {
        t.reserve(n);
        r.reserve(n);
        theta.reserve(n);
        phi.reserve(n);
        pr.reserve(n);
        ptheta.reserve(n);
        pphi.reserve(n);
      }
    }
  };

  // Per-solver templated implementations.
  // Record=false → evaluate() path (no recording overhead).
  // Record=true  → evaluate_and_get_trajectory() path (records each step).
  template <bool Record>
  void run_rk4(const double& t0, std::array<double, 6>& vec0, TrajRecorder& rec);

  template <bool Record>
  void run_boris(const double& t0, std::array<double, 6>& vec0, TrajRecorder& rec);

  template <bool Record>
  void run_rk45(const double& t0, std::array<double, 6>& vec0, TrajRecorder& rec);

  // Helper: pack trajectory vectors into a map.
  static std::map<std::string, std::vector<double>> make_traj_map(
      std::vector<double>& t_arr, std::vector<double>& r_arr, std::vector<double>& theta_arr,
      std::vector<double>& phi_arr, std::vector<double>& pr_arr, std::vector<double>& ptheta_arr,
      std::vector<double>& pphi_arr);

 public:
  // Defaults of the rigidity-aware escape criterion (set_escape_scaling).
  static constexpr double DEFAULT_ESCAPE_FACTOR = integrator_control::DEFAULT_ESCAPE_FACTOR;
  static constexpr double DEFAULT_TIME_FACTOR = integrator_control::DEFAULT_TIME_FACTOR;

  TrajectoryTracer();

  // Full constructor.
  // solver_type: 'r' = frozen-field RK4 (default), 'b' = Boris pusher,
  //              'a' = adaptive RK45 (Dormand-Prince)
  // atol, rtol: error tolerances used only by the RK45 adaptive solver.
  // igrf_params: {data_directory, decimal_year} — must be provided when
  //              bfield_type is 'i' or 't'; the empty-string default will
  //              throw at construction time if IGRF is requested.
  TrajectoryTracer(double charge, double mass = 1.67e-27, double start_altitude = 100. * (1e3),
                   double escape_radius = 10. * constants::RE, double stepsize = 1e-5,
                   int max_iter = 10000, const char bfield_type = 'i',
                   const std::pair<std::string, double>& igrf_params = {"", 2020.},
                   const char solver_type = 'r', double atol = integrator_control::DEFAULT_ATOL,
                   double rtol = integrator_control::DEFAULT_RTOL);

  // Shared-table constructor: borrows an external IGRF table instead of
  // generating its own.  Each tracer still owns its own IGRF object (needed
  // for out-of-range fallback; IGRF::values() is NOT thread-safe).
  TrajectoryTracer(const float* shared_table, const TableParams& table_params, double charge,
                   double mass, double start_altitude, double escape_radius, double stepsize,
                   int max_iter, const std::pair<std::string, double>& igrf_params,
                   const char solver_type = 'r', double atol = integrator_control::DEFAULT_ATOL,
                   double rtol = integrator_control::DEFAULT_RTOL);

  // Reset state for reuse (clears particle_escaped_ and nsteps_).
  void reset() {
    particle_escaped_ = false;
    nsteps_ = 0;
  }

  // Update the termination altitude for the next evaluate() call.
  // Required when reusing a shared tracer across directions with different
  // zenith angles (zenith > 90 reduces start_altitude via cos² scaling).
  void set_start_altitude(double alt) { start_altitude_ = alt; }

  const double& charge() { return charge_; }
  const double& mass() { return mass_; }
  const double& start_altitude() { return start_altitude_; }
  const double& escape_radius() { return escape_radius_; }

  // Rigidity-aware escape criterion: escape_factor k scales the escape radius
  // with the Störmer length, time_factor N sets the time budget to N escape
  // radii of path. escape_radius and max_iter * stepsize remain lower bounds;
  // k = N = 0 restores the fixed criterion.
  void set_escape_scaling(double escape_factor, double time_factor) {
    escape_factor_ = escape_factor;
    time_factor_ = time_factor;
  }
  double escape_factor() const { return escape_factor_; }
  double time_factor() const { return time_factor_; }
  // Dipole strength B0 (tesla) of the active field model at r = RE.
  double dipole_b0();
  // Störmer length (m) for momentum p (kg m/s).
  double stormer_length(double p);
  // Escape radius (m), time budget (s) and step budget of the last run.
  double run_escape_radius() const { return run_limits_.escape_radius; }
  double run_max_time() const { return run_limits_.max_time; }
  int run_max_iter() const { return run_limits_.max_iter; }
  const double& stepsize() { return stepsize_; }
  int max_iter() { return max_iter_; }
  bool particle_escaped() { return particle_escaped_; }
  const double& final_time() { return final_time_; }
  const std::array<double, 6>& final_sixvector() { return final_sixvector_; }
  int nsteps() { return nsteps_; }
  char solver_type() { return solver_type_; }

  inline double lorentz_factor(const double& pr, const double& ptheta, const double& pphi);

  // Evaluate trajectory (no data stored) — dispatcher for all solver types.
  void evaluate(const double& t0, std::array<double, 6>& vec0);

  // Evaluate trajectory and return full history — dispatcher for all solver types.
  std::map<std::string, std::vector<double>> evaluate_and_get_trajectory(
      double& t0, std::array<double, 6>& vec0);

  // Scan rigidities for a single direction and return the cutoff.
  // pos: {r, theta, phi}, mom_unit: {pr, ptheta, pphi} normalised direction,
  // rigidities: descending or ascending list of rigidities to scan,
  // mom_factor: |charge| * KG_M_S_PER_GEVC (converts rigidity to SI momentum).
  // Returns the first rigidity for which the particle escapes, or 0.0.
  double find_cutoff_rigidity(const std::array<double, 3>& pos,
                              const std::array<double, 3>& mom_unit,
                              const std::vector<double>& rigidities, double mom_factor);

  // Binary-search variant: expects rigidities sorted ascending (LOW → HIGH).
  // Tests min rigidity first (cheap escape for most of the sky).
  // Returns the lowest rigidity for which the particle escapes, or 0.0.
  double find_cutoff_rigidity_bisect(const std::array<double, 3>& pos,
                                     const std::array<double, 3>& mom_unit,
                                     const std::vector<double>& rigidities_asc, double mom_factor);
};

#endif  //__TRAJECTORYTRACER_HPP_
