#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "IntegratorControl.hpp"
#include "igrf_table.hpp"

struct BatchGMRCParams {
  double latitude, longitude;                         // degrees
  double detector_alt, particle_alt;                  // km (converted to m internally)
  double escape_radius;                               // meters (default 10*RE)
  double charge;                                      // units of e (+1 for proton)
  double mass;                                        // GeV/c^2
  double min_rigidity, max_rigidity, delta_rigidity;  // GV
  double dt, max_time;                                // seconds
  char solver_type;                                   // 'r', 'b', 'a'
  char bfield_type;                                   // 'i' = direct IGRF, 't' = table (default)
  double atol, rtol;
  int n_samples, n_threads;  // 0 threads = hardware_concurrency()
  int max_attempts_factor;   // safety limit: max attempts = n_samples * this (default 30)
  uint64_t base_seed;
  // Rigidity-aware escape criterion (TrajectoryTracer::set_escape_scaling);
  // escape_radius and max_time remain lower bounds, 0/0 = fixed criterion.
  double escape_factor = integrator_control::DEFAULT_ESCAPE_FACTOR;
  double time_factor = integrator_control::DEFAULT_TIME_FACTOR;
  // Optional Ctrl+C stop flag: set to true to request cooperative cancellation.
  // Threads check this flag each iteration and exit early if set.
  // Nullptr means no cancellation support (default).
  std::atomic<bool>* stop_flag = nullptr;
};

struct BatchGMRCResult {
  std::vector<double> zenith, azimuth, rcutoff;  // each size <= n_samples
  // 1 where no tested rigidity escaped (cutoff > max_rigidity); rcutoff then
  // holds max_rigidity, a lower bound.
  std::vector<uint8_t> above_max;
  int64_t total_trajectories;  // total evaluate() calls across all threads
};

// shared_table may be nullptr when params.bfield_type is 'i' (direct IGRF) or 'd' (dipole).
BatchGMRCResult batch_gmrc_evaluate(const float* shared_table, const TableParams& table_params,
                                    const std::pair<std::string, double>& igrf_params,
                                    const BatchGMRCParams& params);

// -------------------------------------------------------------------------
// Penumbra-resolving cutoffs (Cooke et al. 1991) on a geometric rigidity ladder
// R_k = r_lo (1 + delta)^k, k = 0 .. K-1 (K = ceil(ln(r_hi/r_lo) / ln(1+delta)) + 1).
// Rung k covers [R_k / sqrt(q), R_k sqrt(q)], q = 1 + delta.
//   R_U   : upper edge of the highest forbidden rung (upper cutoff)
//   R_L   : lower edge of the lowest allowed rung (lower cutoff)
//   R_eff : R_U - sum of allowed rung widths in [R_L, R_U]
//           (= bottom edge of the ladder + sum of all forbidden rung widths)
//   flag  : 0 = resolved, 1 = no forbidden rung (cutoff below the ladder; R_L = R_U = R_eff =
//           bottom edge), 2 = top rung forbidden (R_U and R_eff are lower bounds)
// Adaptive scan: every `coarse`-th rung is traced first; coarse intervals within `guard`
// intervals of an allowed/forbidden change are then filled rung by rung, the others inherit
// the common value of their end rungs. coarse = 1 traces every rung (exact on the ladder).
// The rigidity parameters min/max/delta_rigidity of BatchGMRCParams are not used.
// -------------------------------------------------------------------------

struct PenumbraParams {
  double r_lo = 0.3, r_hi = 100.0, delta = 0.005;  // GV, GV, relative rung spacing
  int coarse = 8, guard = 2;
  bool store_ladder = false;  // return the allowed/forbidden flags of every rung
};

struct PenumbraResult {
  std::vector<double> zenith, azimuth;          // degrees, size n
  std::vector<double> r_lower, r_upper, r_eff;  // GV, size n
  std::vector<int32_t> flag;                    // size n
  std::vector<double> ladder;                   // rung centres R_k, size K
  std::vector<uint8_t> allowed;                 // n x K (row-major), only if store_ladder
  int64_t total_trajectories = 0;
};

// Directions: if zenith_in/azimuth_in are non-empty (equal sizes) they are evaluated as given;
// otherwise params.n_samples directions are drawn uniformly in zenith [0, 180) and azimuth
// [0, 360) as in batch_gmrc_evaluate. Every direction is returned (none is dropped).
PenumbraResult batch_gmrc_penumbra(const float* shared_table, const TableParams& table_params,
                                   const std::pair<std::string, double>& igrf_params,
                                   const BatchGMRCParams& params, const PenumbraParams& pparams,
                                   const std::vector<double>& zenith_in,
                                   const std::vector<double>& azimuth_in);
