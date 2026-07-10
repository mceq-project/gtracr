/*
MuonTracer ("mutracr") — batch transport of atmospheric muons between
production and decay.

Physics: muons below ~100 GeV do not reinteract between production and
decay — transport is a single-particle ODE (Lorentz bending + continuous
dE/dX loss) and decay is deposited per step as *expected* decay weight
(survival x decay rate x dt), never sampled.  A weighted ensemble of
source rays is therefore an exact quadrature: no Monte Carlo, no
variance.  This is the C++ port of the nu3d stage-C numpy tracer
(mceq-em-integration nu3d/mutracer/tracer.py), which remains the
correctness reference; the per-step algebra here replicates it exactly,
including two documented quirks:

  * the decay deposit of a ground-crossing step uses the full-step
    survival (the fate weight uses the path-fraction survival), so the
    weight audit closes only to O(per-step decay probability) on the
    crossing step;
  * MCS variance is accumulated per step but folded only at decay
    (deposits carry sigma = sqrt(th2) *before* the step's increment);
    the MCS-bend correlation along the path is not carried.

State per ray (SoA): x [cm, Earth-centred Cartesian, z = geographic
north] | p [GeV/c] | q [charge/e] | pol [spin projection on p_hat] |
w [decay-survival weight] | th2 [rad^2 accumulated Fermi-Eyges MCS
variance].

Dynamics (RK4 on (x, p), per-ray adaptive dt):
  dx/dt = beta c p_hat
  dp/dt = KP q beta (p_hat x B[T]) - dedx(E) rho(h) c p_hat
with KP = 1e-9 c[m/s]^2 GeV/(s T); the loss term uses dp/dt =
(E/p) dE/dt = -dedx rho c exactly (beta E = p).

Units: cm, s, GeV, Tesla, g/cm^3, dE/dX in GeV/(g/cm^2) — the
atmospheric-muon literature convention, NOT the SI units of
TrajectoryTracer.  The Python wrapper (gtracr.mutracr) documents this.

Termination fates: 'ground' (r < r_ground, state pulled back onto the
sphere), 'stopped' (E < e_min), 'faded' (w < w_min), 'escaped' (above
r_ground + h_escape moving outward), 'maxsteps' (still alive at the
step cap).  deposited + sum(fate weights) == sum(source weights) up to
the crossing-step quirk above (audited in the result).

Deposit consumers (selected per trace() call):
  none     — weights still evolve, nothing recorded
  bank     — raw deposit rows (x_mid, phat, e_mu, q, pol, sig_mcs, w)
  spectrum — per-charge E_mu histograms of sum(w) and sum(w*pol)
  angular  — per-charge (E_mu, delta-to-axis) histograms of sum(w),
             sum(w*pol), sum(w*sig_mcs^2), plus a delta-overflow tally

Threading: rays are independent; trace() splits the batch over
std::thread workers with per-thread accumulators merged at the end.
Per-ray trajectories are bit-identical regardless of n_threads; tally
sums may differ at floating-point rounding level.
*/
#ifndef __MUONTRACER_HPP_
#define __MUONTRACER_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace muon {

// Constants copied bit-exact from the numpy reference (nu3d).
constexpr double M_MU = 0.1056583755;    // GeV
constexpr double TAU_MU = 2.1969811e-6;  // s
constexpr double C_CMS = 2.99792458e10;  // cm/s
constexpr double KP = 8.98755179e7;      // GeV/(s T) — 1e-9 * c[m/s]^2
constexpr double X0_AIR = 36.62;         // g/cm^2
constexpr double ES_MCS = 0.0136;        // GeV — Highland scale, no log term

// Fate codes (result.fate per ray).
enum Fate : int8_t {
  FATE_ALIVE = 0,  // still alive at max_steps ('maxsteps' weight)
  FATE_GROUND = 1,
  FATE_STOPPED = 2,
  FATE_FADED = 3,
  FATE_ESCAPED = 4,
};

// ---------------------------------------------------------------------------
// Magnetic field: 'z' zero, 'u' uniform, 'd' axial dipole (gtracr
// convention), 's' IGRF-13 thin-shell table (trilinear in h/colat/lon,
// generated from the C++ IGRF at construction).
// Cartesian in/out, positions in cm, B in Tesla.
// ---------------------------------------------------------------------------
class MuonField {
 public:
  // Zero or uniform field.
  explicit MuonField(const std::array<double, 3>& b_uniform);
  // Dipole: g10 in Tesla, reference radius in cm (field normalisation
  // only — distinct from the tracer's ground radius).
  MuonField(double g10_T, double r_ref_cm);
  // IGRF-13 shell table: nodes at r = r_ground + h for h in
  // [0, h_max_cm] (n_h linear), colat [0, 180] deg (n_colat), lon
  // [0, 360] deg inclusive (n_lon; the 360 column duplicates 0 so the
  // interpolation never wraps).  Queries outside the h range clamp to
  // the boundary shell.
  MuonField(const std::string& igrf_json_path, double year, double r_ground_cm, int n_h,
            int n_colat, int n_lon, double h_max_cm);

  std::array<double, 3> at(const std::array<double, 3>& x_cm) const;
  bool is_zero() const { return type_ == 'z'; }

 private:
  char type_;                       // 'z', 'u', 'd', 's'
  std::array<double, 3> b_uni_{};   // 'u'
  double g10_ = 0.0, r_ref_ = 0.0;  // 'd'
  // 's' — flat [3][n_h][n_colat][n_lon] table in Tesla; the axes are
  // uniform (h in cm, colat/lon in deg), stored as origin + 1/spacing.
  std::vector<double> tbl_;
  double r_ground_ = 0.0;
  double inv_dh_ = 0.0, inv_dc_ = 0.0, inv_dl_ = 0.0;
  int nh_ = 0, nc_ = 0, nl_ = 0;
};

// ---------------------------------------------------------------------------
// Atmosphere: 'n' none (vacuum), 'l' Linsley/CORSIKA US-Std layers,
// 'u' uniform, 't' tabulated rho(h) with linear interpolation.
// rho in g/cm^3 as a function of altitude h [cm] above the ground
// sphere; zero above h_top.  Layer conventions replicate the numpy
// reference exactly (uniform: rho0 for all h < h_top including h < 0;
// linsley/table: zero for h < 0).
// ---------------------------------------------------------------------------
class MuonAtmosphere {
 public:
  MuonAtmosphere();                                    // 'n'
  explicit MuonAtmosphere(double rho0, double h_top);  // 'u'
  static MuonAtmosphere us_std();                      // 'l'
  MuonAtmosphere(const std::vector<double>& h_cm, const std::vector<double>& rho,
                 double h_top);  // 't'

  double rho(double h_cm) const;
  bool is_none() const { return type_ == 'n'; }

 private:
  char type_;
  double rho0_ = 0.0;
  double h_top_ = 1.128e7;  // cm — MCEq atmosphere top
  std::vector<double> h_, r_;
  bool uniform_h_ = false;  // 't' fast path when h is evenly spaced
  double inv_dh_ = 0.0;
};

// ---------------------------------------------------------------------------
// dE/dX: constant, or table a(E_kin) interpolated linearly in
// ln(E_kin), clamped at both ends (numpy np.interp semantics).
// Positive values, GeV/(g/cm^2); evaluated at TOTAL energy.
// ---------------------------------------------------------------------------
class MuonDedx {
 public:
  explicit MuonDedx(double a_const);
  MuonDedx(const std::vector<double>& e_kin_grid, const std::vector<double>& a);
  double at(double e_tot) const;

 private:
  bool constant_;
  double a0_;
  std::vector<double> ln_e_, a_;
};

// ---------------------------------------------------------------------------
// Deposit configuration + result containers
// ---------------------------------------------------------------------------
struct DepositConfig {
  char kind = 'n';                         // 'n' none, 'b' bank, 's' spectrum, 'a' angular
  double w_thresh = 0.0;                   // bank
  std::vector<double> e_edges;             // spectrum / angular (E_mu total)
  std::vector<double> d_edges;             // angular (delta rad)
  std::array<double, 3> axis = {0, 0, 1};  // angular (normalised in trace)
};

struct MuTraceResult {
  // Final per-ray state (x, p flattened N*3).
  std::vector<double> x, p, w, th2;
  std::vector<int8_t> fate;
  long steps = 0;  // max steps taken by any ray (== numpy lockstep count)
  double deposited = 0.0;
  // ground, stopped, faded, escaped, maxsteps
  std::array<double, 5> fate_w = {0, 0, 0, 0, 0};
  double weight_audit = 0.0;  // deposited + sum(fate_w); == sum(w_in) - quirk

  // bank ('b'): rows above w_thresh
  std::vector<double> bank_x, bank_phat, bank_e, bank_pol, bank_sig, bank_w;
  std::vector<int8_t> bank_q;
  double bank_total_w = 0.0;  // includes sub-threshold rows

  // spectrum ('s'): [n_e] per charge
  std::vector<double> spec_w_pos, spec_wp_pos, spec_w_neg, spec_wp_neg;

  // angular ('a'): flat [n_e][n_d] per charge + overflow
  std::vector<double> ang_w_pos, ang_wp_pos, ang_ws2_pos;
  std::vector<double> ang_w_neg, ang_wp_neg, ang_ws2_neg;
  double ang_overflow = 0.0;
};

struct TraceOptions {
  double e_min = 0.11;  // GeV total; clamped above m_mu internally
  double w_min = 1e-6;
  double dt_max = 0.0;  // <= 0 means no cap
  long max_steps = 100000;
  double ctl_bend = 0.05;       // rad
  double ctl_loss = 0.05;       // fractional dE/E
  double ctl_decay = 0.05;      // fractional decay probability
  double ctl_path = 2e5;        // cm
  bool decay_beta_one = false;  // pre-fix MCEq convention (1/(gamma c tau))
  int n_threads = 0;            // 0 = hardware concurrency
};

// ---------------------------------------------------------------------------
// MuonTracer — owns field, atmosphere, dE/dX; trace() is stateless
// per batch and thread-safe (all evaluation methods are const).
// ---------------------------------------------------------------------------
class MuonTracer {
 public:
  MuonTracer(MuonField field, MuonAtmosphere atmo, MuonDedx dedx, double r_ground_cm,
             double h_escape_cm = 1.20e7);

  // March the ensemble until every ray is dead.  Inputs are length-N
  // SoA arrays (x, p flattened N*3).
  MuTraceResult trace(const std::vector<double>& x, const std::vector<double>& p,
                      const std::vector<int8_t>& q, const std::vector<double>& pol,
                      const std::vector<double>& w, const TraceOptions& opt,
                      const DepositConfig& dep) const;

  double r_ground() const { return r_ground_; }
  double h_escape() const { return h_escape_; }
  // Field/atmosphere/dE/dX probes (Python-side validation + tests).
  std::array<double, 3> bfield_at(const std::array<double, 3>& x_cm) const {
    return field_.at(x_cm);
  }
  double rho_at(double h_cm) const { return atmo_.rho(h_cm); }
  double dedx_at(double e_tot) const { return dedx_.at(e_tot); }

 private:
  MuonField field_;
  MuonAtmosphere atmo_;
  MuonDedx dedx_;
  double r_ground_;
  double h_escape_;

  // Worker: trace rays [i0, i1) into res.  Final-state slots are
  // disjoint per worker; the accumulator merge at the end is
  // serialised by merge_mutex.
  void trace_range(std::size_t i0, std::size_t i1, const std::vector<double>& x,
                   const std::vector<double>& p, const std::vector<int8_t>& q,
                   const std::vector<double>& pol, const std::vector<double>& w,
                   const TraceOptions& opt, const DepositConfig& dep, MuTraceResult& res,
                   std::mutex* merge_mutex) const;
};

}  // namespace muon

#endif  // __MUONTRACER_HPP_
