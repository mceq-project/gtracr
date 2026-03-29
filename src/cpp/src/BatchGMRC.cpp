/*
 * BatchGMRC — C++ batch evaluator for geomagnetic rigidity cutoffs.
 *
 * Moves the entire GMRC Monte Carlo loop (RNG, coordinate transforms,
 * rigidity scanning, threading) into C++, eliminating all Python overhead.
 */

#include "BatchGMRC.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "TrajectoryTracer.hpp"
#include "constants.hpp"

// =========================================================================
// A. Coordinate transform (C++ port of Trajectory.detector_to_geocentric)
// =========================================================================

struct TransformContext {
  // 3x3 transformation matrix (row-major: mat[row][col])
  double mat[3][3];
  // Detector position in Cartesian ECEF (metres)
  double det_x, det_y, det_z;
  // Particle altitude in metres
  double palt_m;
  // Detector altitude in metres
  double dalt_m;
};

static TransformContext build_transform_context(double lat_deg, double lng_deg, double dalt_km,
                                                double palt_km) {
  TransformContext ctx;
  ctx.palt_m = palt_km * 1e3;
  ctx.dalt_m = dalt_km * 1e3;

  const double lmbda = lat_deg * constants::DEG_TO_RAD;
  const double eta = lng_deg * constants::DEG_TO_RAD;

  const double sl = std::sin(lmbda), cl = std::cos(lmbda);
  const double se = std::sin(eta), ce = std::cos(eta);

  // Transform matrix (same as Trajectory.transform_matrix)
  ctx.mat[0][0] = -se;
  ctx.mat[0][1] = -ce * sl;
  ctx.mat[0][2] = cl * ce;
  ctx.mat[1][0] = ce;
  ctx.mat[1][1] = -sl * se;
  ctx.mat[1][2] = cl * se;
  ctx.mat[2][0] = 0.0;
  ctx.mat[2][1] = cl;
  ctx.mat[2][2] = sl;

  // Detector position in Cartesian (geodesic_to_cartesian)
  const double r_det = constants::RE + ctx.dalt_m;
  const double theta_det = (90.0 - lat_deg) * constants::DEG_TO_RAD;
  const double phi_det = lng_deg * constants::DEG_TO_RAD;

  ctx.det_x = r_det * std::sin(theta_det) * std::cos(phi_det);
  ctx.det_y = r_det * std::sin(theta_det) * std::sin(phi_det);
  ctx.det_z = r_det * std::cos(theta_det);

  return ctx;
}

// Compute particle_coord in detector frame (get_particle_coord)
static void get_particle_coord(double zenith_deg, double azimuth_deg, double altitude,
                               double magnitude, double& xt, double& yt, double& zt) {
  const double xi = zenith_deg * constants::DEG_TO_RAD;
  const double alpha = azimuth_deg * constants::DEG_TO_RAD;

  xt = magnitude * std::sin(xi) * std::sin(alpha);
  yt = -magnitude * std::sin(xi) * std::cos(alpha);
  zt = magnitude * std::cos(xi) + altitude;
}

// Apply transform: detector_coord + mat * particle_coord
static void apply_transform(const TransformContext& ctx, double px, double py, double pz, double ox,
                            double oy, double oz, double& rx, double& ry, double& rz) {
  rx = ox + ctx.mat[0][0] * px + ctx.mat[0][1] * py + ctx.mat[0][2] * pz;
  ry = oy + ctx.mat[1][0] * px + ctx.mat[1][1] * py + ctx.mat[1][2] * pz;
  rz = oz + ctx.mat[2][0] * px + ctx.mat[2][1] * py + ctx.mat[2][2] * pz;
}

// Cartesian to spherical (coordinates)
static void cart_to_sph_coord(double x, double y, double z, double& r, double& theta, double& phi) {
  r = std::sqrt(x * x + y * y + z * z);
  theta = std::acos(z / r);
  phi = std::atan2(y, x);
}

// Cartesian momentum to spherical momentum
static void cart_to_sph_mom(double theta, double phi, double px, double py, double pz, double& pr,
                            double& ptheta, double& pphi) {
  const double st = std::sin(theta), ct = std::cos(theta);
  const double sp = std::sin(phi), cp = std::cos(phi);

  pr = st * cp * px + st * sp * py + ct * pz;
  ptheta = ct * cp * px + ct * sp * py - st * pz;
  pphi = -sp * px + cp * py;
}

struct DirectionIC {
  std::array<double, 3> pos;       // {r, theta, phi}
  std::array<double, 3> mom_unit;  // unit momentum direction in spherical
  double start_altitude;           // start_altitude for termination condition
};

static DirectionIC compute_direction_ic(const TransformContext& ctx, double zenith_deg,
                                        double azimuth_deg, double ref_momentum_gevc) {
  DirectionIC ic;

  double start_alt = ctx.dalt_m + ctx.palt_m;

  // Position
  double pc_x, pc_y, pc_z;
  if (zenith_deg > 90.0) {
    start_alt = start_alt * std::cos(zenith_deg * constants::DEG_TO_RAD) *
                std::cos(zenith_deg * constants::DEG_TO_RAD);

    double magnitude =
        -(2.0 * constants::RE + ctx.palt_m) * std::cos(zenith_deg * constants::DEG_TO_RAD);
    get_particle_coord(zenith_deg, azimuth_deg, 0.0, magnitude, pc_x, pc_y, pc_z);
  } else {
    get_particle_coord(zenith_deg, azimuth_deg, ctx.palt_m, 1e-10, pc_x, pc_y, pc_z);
  }

  double cart_x, cart_y, cart_z;
  apply_transform(ctx, pc_x, pc_y, pc_z, ctx.det_x, ctx.det_y, ctx.det_z, cart_x, cart_y, cart_z);

  double r, theta, phi;
  cart_to_sph_coord(cart_x, cart_y, cart_z, r, theta, phi);
  ic.pos = {r, theta, phi};

  // Momentum direction (unit vector in spherical coords)
  double mom_si = ref_momentum_gevc * constants::KG_M_S_PER_GEVC;
  double mp_x, mp_y, mp_z;
  get_particle_coord(zenith_deg, azimuth_deg, 0.0, mom_si, mp_x, mp_y, mp_z);

  // Transform momentum (offset = 0,0,0 for momentum)
  double cart_px, cart_py, cart_pz;
  apply_transform(ctx, mp_x, mp_y, mp_z, 0.0, 0.0, 0.0, cart_px, cart_py, cart_pz);

  double pr, ptheta, pphi;
  cart_to_sph_mom(theta, phi, cart_px, cart_py, cart_pz, pr, ptheta, pphi);

  // Normalise to unit direction
  double pmag = std::sqrt(pr * pr + ptheta * ptheta + pphi * pphi);
  if (pmag > 0.0) {
    ic.mom_unit = {pr / pmag, ptheta / pmag, pphi / pmag};
  } else {
    ic.mom_unit = {0.0, 0.0, 0.0};
  }

  ic.start_altitude = start_alt;
  return ic;
}

// =========================================================================
// B. Binary search cutoff rigidity
// =========================================================================

// rigidities_asc is sorted LOW → HIGH (e.g. 5, 6, ..., 55 GV).
// Most of the sky has low cutoffs, so testing min rigidity first (cheap
// escape) avoids the expensive max-rigidity forbidden trajectory.
// n_evals is incremented by the number of evaluate() calls made.
static double find_cutoff_rigidity_bisect(
    TrajectoryTracer& tracer, const std::array<double, 3>& pos,
    const std::array<double, 3>& mom_unit,
    const std::vector<double>& rigidities_asc,  // sorted LOW → HIGH
    double mom_factor, int64_t& n_evals) {
  if (rigidities_asc.empty()) return 0.0;

  auto escapes = [&](int idx) -> bool {
    double rig = rigidities_asc[idx];
    double mom_si = rig * mom_factor;
    std::array<double, 6> vec0 = {
        {pos[0], pos[1], pos[2], mom_unit[0] * mom_si, mom_unit[1] * mom_si, mom_unit[2] * mom_si}};
    tracer.reset();
    tracer.evaluate(0.0, vec0);
    ++n_evals;
    return tracer.particle_escaped();
  };

  int lo = 0;                                            // lowest rigidity
  int hi = static_cast<int>(rigidities_asc.size()) - 1;  // highest rigidity

  // Fast path: if min rigidity already escapes, cutoff ≤ min → return min
  if (escapes(lo)) return rigidities_asc[lo];
  // If max rigidity doesn't escape, cutoff > max → return 0
  if (!escapes(hi)) return 0.0;

  // Invariant: rigidities_asc[lo] is forbidden, rigidities_asc[hi] escapes.
  // Binary search for the transition.
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (escapes(mid))
      hi = mid;  // escape → cutoff is at lower rigidity (lower index)
    else
      lo = mid;  // forbidden → cutoff is at higher rigidity (higher index)
  }

  // rigidities_asc[hi] is the first escaping rigidity
  return rigidities_asc[hi];
}

// =========================================================================
// C. Thread worker
// =========================================================================

struct WorkerResult {
  std::vector<double> zenith, azimuth, rcutoff;
  int64_t n_trajectories = 0;
};

static WorkerResult thread_worker(const float* shared_table, const TableParams& table_params,
                                  const std::pair<std::string, double>& igrf_params,
                                  const TransformContext& ctx,
                                  const std::vector<double>& rigidities_asc,
                                  double ref_momentum_gevc, double mom_factor, double charge_si,
                                  double mass_si, double escape_radius, double dt, int max_step,
                                  char solver_type, char bfield_type, double atol, double rtol,
                                  int quota, int max_attempts, uint64_t seed,
                                  const std::atomic<bool>* stop_flag, int thread_id) {
#ifdef GTRACR_DEBUG_PRINT
  std::fprintf(stderr,
               "[Thread %d] Starting: quota=%d max_attempts=%d "
               "solver=%c bfield=%c atol=%.0e rtol=%.0e dt=%.0e max_step=%d\n",
               thread_id, quota, max_attempts, solver_type, bfield_type, atol, rtol, dt, max_step);
  std::fflush(stderr);
#endif

  WorkerResult result;
  result.zenith.reserve(quota);
  result.azimuth.reserve(quota);
  result.rcutoff.reserve(quota);

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist_az(0.0, 360.0);
  std::uniform_real_distribution<double> dist_zen(0.0, 180.0);

  // Build thread-local TrajectoryTracer.
  // Each thread owns its own IGRF instance (IGRF::values is not thread-safe).
  // For 'i' (direct IGRF): use the regular constructor with bfield_type='i'.
  // For 't' (table):       use the shared-table constructor.
  auto make_tracer = [&]() -> TrajectoryTracer {
    if (bfield_type == 'i') {
      return TrajectoryTracer(charge_si, mass_si, 0.0, escape_radius, dt, max_step, 'i',
                              igrf_params, solver_type, atol, rtol);
    } else {
      return TrajectoryTracer(shared_table, table_params, charge_si, mass_si, 0.0, escape_radius,
                              dt, max_step, igrf_params, solver_type, atol, rtol);
    }
  };
  TrajectoryTracer tracer = make_tracer();

#ifdef GTRACR_DEBUG_PRINT
  std::fprintf(stderr, "[Thread %d] TrajectoryTracer constructed\n", thread_id);
  std::fflush(stderr);
#endif

  int successes = 0;
  int attempts = 0;
#ifdef GTRACR_DEBUG_PRINT
  int last_report = 0;
#endif

  while (successes < quota && attempts < max_attempts &&
         !(stop_flag && stop_flag->load(std::memory_order_relaxed))) {
    double azimuth = dist_az(rng);
    double zenith = dist_zen(rng);
    ++attempts;

    // Print a progress line every 100 successes or 500 attempts, whichever comes first.
#ifdef GTRACR_DEBUG_PRINT
    if (successes - last_report >= 100 || (attempts % 500 == 0)) {
      std::fprintf(stderr,
                   "[Thread %d] Progress: %d/%d successes, %d attempts, "
                   "%lld traj_evals so far\n",
                   thread_id, successes, quota, attempts, (long long)result.n_trajectories);
      std::fflush(stderr);
      last_report = successes;
    }
#endif

    DirectionIC ic = compute_direction_ic(ctx, zenith, azimuth, ref_momentum_gevc);
    tracer.set_start_altitude(ic.start_altitude);

    double rc = find_cutoff_rigidity_bisect(tracer, ic.pos, ic.mom_unit, rigidities_asc, mom_factor,
                                            result.n_trajectories);

    if (rc > 0.0) {
      result.zenith.push_back(zenith);
      result.azimuth.push_back(azimuth);
      result.rcutoff.push_back(rc);
      ++successes;
    }
  }

#ifdef GTRACR_DEBUG_PRINT
  std::fprintf(stderr,
               "[Thread %d] Finished: %d/%d successes, %d attempts, "
               "%lld total traj_evals\n",
               thread_id, successes, quota, attempts, (long long)result.n_trajectories);
  std::fflush(stderr);
#endif

  return result;
}

// =========================================================================
// D. Main function
// =========================================================================

BatchGMRCResult batch_gmrc_evaluate(const float* shared_table, const TableParams& table_params,
                                    const std::pair<std::string, double>& igrf_params,
                                    const BatchGMRCParams& params) {
  // Build transform context
  TransformContext ctx = build_transform_context(params.latitude, params.longitude,
                                                 params.detector_alt, params.particle_alt);

  // Build rigidity list sorted ascending (LOW → HIGH).
  // Most of the sky has low cutoffs, so testing min rigidity first
  // (cheap escape) is much faster than starting from the top.
  std::vector<double> rigidities_asc;
  for (double r = params.min_rigidity; r <= params.max_rigidity + 1e-12;
       r += params.delta_rigidity) {
    rigidities_asc.push_back(r);
  }
  // Ensure max_rigidity is included
  if (rigidities_asc.empty() || rigidities_asc.back() < params.max_rigidity - 1e-12) {
    rigidities_asc.push_back(params.max_rigidity);
  }

  // Derived constants
  double charge_abs = std::abs(params.charge);
  double ref_momentum_gevc = params.max_rigidity * charge_abs;  // for IC computation
  double mom_factor = charge_abs * constants::KG_M_S_PER_GEVC;
  double charge_si = params.charge * constants::ELEMENTARY_CHARGE;
  double mass_si = params.mass * constants::KG_PER_GEVC2;
  int max_step = static_cast<int>(std::ceil(params.max_time / params.dt));

  // Thread count
  int n_threads = params.n_threads;
  if (n_threads <= 0) {
    n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) n_threads = 1;
  }

  // Divide samples into per-thread quotas
  std::vector<int> quotas(n_threads, params.n_samples / n_threads);
  int remainder = params.n_samples % n_threads;
  for (int i = 0; i < remainder; ++i) {
    quotas[i]++;
  }

  int max_attempts_factor = params.max_attempts_factor > 0 ? params.max_attempts_factor : 30;

  // Spawn threads
  std::vector<std::thread> threads(n_threads);
  std::vector<WorkerResult> results(n_threads);

  for (int t = 0; t < n_threads; ++t) {
    int quota = quotas[t];
    int max_attempts = quota * max_attempts_factor;
    uint64_t seed = params.base_seed + static_cast<uint64_t>(t) * 1000003ULL;

    threads[t] = std::thread([&, t, quota, max_attempts, seed]() {
      results[t] =
          thread_worker(shared_table, table_params, igrf_params, ctx, rigidities_asc,
                        ref_momentum_gevc, mom_factor, charge_si, mass_si, params.escape_radius,
                        params.dt, max_step, params.solver_type, params.bfield_type, params.atol,
                        params.rtol, quota, max_attempts, seed, params.stop_flag, t);
    });
  }

  // Join all threads
  for (int t = 0; t < n_threads; ++t) {
    threads[t].join();
  }

  // Concatenate results
  BatchGMRCResult output;
  output.total_trajectories = 0;
  int total = 0;
  for (auto& r : results) {
    total += static_cast<int>(r.zenith.size());
    output.total_trajectories += r.n_trajectories;
  }
  output.zenith.reserve(total);
  output.azimuth.reserve(total);
  output.rcutoff.reserve(total);

  for (auto& r : results) {
    output.zenith.insert(output.zenith.end(), r.zenith.begin(), r.zenith.end());
    output.azimuth.insert(output.azimuth.end(), r.azimuth.begin(), r.azimuth.end());
    output.rcutoff.insert(output.rcutoff.end(), r.rcutoff.begin(), r.rcutoff.end());
  }

  return output;
}
