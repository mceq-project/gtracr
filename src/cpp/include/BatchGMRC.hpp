#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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
  // Optional Ctrl+C stop flag: set to true to request cooperative cancellation.
  // Threads check this flag each iteration and exit early if set.
  // Nullptr means no cancellation support (default).
  std::atomic<bool>* stop_flag = nullptr;
};

struct BatchGMRCResult {
  std::vector<double> zenith, azimuth, rcutoff;  // each size <= n_samples
  int64_t total_trajectories;                    // total evaluate() calls across all threads
};

// shared_table may be nullptr when params.bfield_type == 'i' (direct IGRF).
BatchGMRCResult batch_gmrc_evaluate(const float* shared_table, const TableParams& table_params,
                                    const std::pair<std::string, double>& igrf_params,
                                    const BatchGMRCParams& params);
