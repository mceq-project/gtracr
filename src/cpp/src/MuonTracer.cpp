/*
MuonTracer — batch atmospheric-muon transport (see MuonTracer.hpp).

The per-step algebra replicates the numpy reference
(mceq-em-integration nu3d/mutracer/tracer.py) exactly, in the same
order: dt control -> RK4 (field re-evaluated per stage) -> decay
deposit at the full-step midpoint -> MCS variance increment ->
survival -> ground pullback -> fate checks.
*/

#include "MuonTracer.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "igrf.hpp"

namespace muon {

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;

inline double norm3(const double* v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline double dot3(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// numpy _cell: cell index + clamped fractional coordinate on one axis.
// The shell axes are uniform (linspace), so the cell is found by direct
// division instead of a binary search; the clamped result is identical.
inline void cell_uniform(double g0, double inv_dg, int n, double v, int& i, double& f) {
  const double t = (v - g0) * inv_dg;
  i = std::min(std::max(static_cast<int>(t), 0), n - 2);
  f = std::min(std::max(t - i, 0.0), 1.0);
}

// np.interp-style bin index on an arbitrary ascending grid.
inline int hist_cell(const std::vector<double>& grid, double v) {
  return static_cast<int>(std::upper_bound(grid.begin(), grid.end(), v) - grid.begin()) - 1;
}

// np.histogram bin index: [e_i, e_{i+1}) with the last bin closed.
// Returns -1 when out of range.
inline int hist_bin(const std::vector<double>& edges, double v) {
  if (v < edges.front() || v > edges.back()) return -1;
  int idx = static_cast<int>(std::upper_bound(edges.begin(), edges.end(), v) - edges.begin()) - 1;
  const int n_bins = static_cast<int>(edges.size()) - 1;
  if (idx == n_bins) idx = n_bins - 1;  // v == last edge (closed)
  return idx;
}

}  // namespace

// ---------------------------------------------------------------------------
// MuonField
// ---------------------------------------------------------------------------

MuonField::MuonField(const std::array<double, 3>& b_uniform) : b_uni_{b_uniform} {
  const double m = std::sqrt(b_uniform[0] * b_uniform[0] + b_uniform[1] * b_uniform[1] +
                             b_uniform[2] * b_uniform[2]);
  type_ = (m == 0.0) ? 'z' : 'u';
}

MuonField::MuonField(double g10_T, double r_ref_cm) : type_{'d'}, g10_{g10_T}, r_ref_{r_ref_cm} {}

MuonField::MuonField(const std::string& igrf_json_path, double year, double r_ground_cm, int n_h,
                     int n_colat, int n_lon, double h_max_cm)
    : type_{'s'}, r_ground_{r_ground_cm}, nh_{n_h}, nc_{n_colat}, nl_{n_lon} {
  if (n_h < 2 || n_colat < 2 || n_lon < 2) {
    throw std::invalid_argument("MuonField shell: need at least 2 nodes per axis");
  }
  IGRF igrf(igrf_json_path, year);
  inv_dh_ = (n_h - 1) / h_max_cm;
  inv_dc_ = (n_colat - 1) / 180.0;
  inv_dl_ = (n_lon - 1) / 360.0;
  tbl_.resize(static_cast<std::size_t>(3) * n_h * n_colat * n_lon);
  // IGRF::values is stateful (not thread-safe) — build single-threaded.
  for (int ih = 0; ih < n_h; ++ih) {
    const double r_m = (r_ground_cm + h_max_cm * ih / (n_h - 1)) * 1e-2;
    for (int ic = 0; ic < n_colat; ++ic) {
      const double colat = 180.0 * ic / (n_colat - 1);
      for (int il = 0; il < n_lon; ++il) {
        const double lon = 360.0 * il / (n_lon - 1);
        const auto b = igrf.values(r_m, colat * DEG2RAD, lon * DEG2RAD);
        // interleaved layout [h][colat][lon][3]: one node's components
        // share a cache line (trilinear reads 8 nodes x 3 components)
        const std::size_t base = 3 * ((static_cast<std::size_t>(ih) * nc_ + ic) * nl_ + il);
        tbl_[base] = b[0];
        tbl_[base + 1] = b[1];
        tbl_[base + 2] = b[2];
      }
    }
  }
}

std::array<double, 3> MuonField::at(const std::array<double, 3>& x) const {
  switch (type_) {
    case 'z':
      return {0.0, 0.0, 0.0};
    case 'u':
      return b_uni_;
    case 'd': {
      const double r = norm3(x.data());
      const double rx = x[0] / r, ry = x[1] / r, rz = x[2] / r;
      const double fac = -g10_ * (r_ref_ / r) * (r_ref_ / r) * (r_ref_ / r);
      return {fac * 3.0 * rz * rx, fac * 3.0 * rz * ry, fac * (3.0 * rz * rz - 1.0)};
    }
    default: {  // 's'
      const double r = norm3(x.data());
      double ct = x[2] / r;
      ct = std::min(std::max(ct, -1.0), 1.0);
      const double th = std::acos(ct);
      double ph = std::atan2(x[1], x[0]);
      if (ph < 0.0) ph += 2.0 * PI;  // numpy % (2 pi)
      int ih, ic, il;
      double fh, fc, fl;
      cell_uniform(0.0, inv_dh_, nh_, r - r_ground_, ih, fh);
      cell_uniform(0.0, inv_dc_, nc_, th / DEG2RAD, ic, fc);
      cell_uniform(0.0, inv_dl_, nl_, ph / DEG2RAD, il, fl);
      double sph[3] = {0.0, 0.0, 0.0};
      for (int dh = 0; dh < 2; ++dh) {
        for (int dc = 0; dc < 2; ++dc) {
          for (int dl = 0; dl < 2; ++dl) {
            const double wgt = (dh ? fh : 1.0 - fh) * (dc ? fc : 1.0 - fc) * (dl ? fl : 1.0 - fl);
            const std::size_t base =
                3 * ((static_cast<std::size_t>(ih + dh) * nc_ + (ic + dc)) * nl_ + (il + dl));
            sph[0] += wgt * tbl_[base];
            sph[1] += wgt * tbl_[base + 1];
            sph[2] += wgt * tbl_[base + 2];
          }
        }
      }
      // st = sin(acos(ct)), (cp, sp) = (cos, sin)(atan2(y, x)) — computed
      // algebraically instead of via trig calls (identical up to rounding).
      const double st = std::sqrt(std::max(1.0 - ct * ct, 0.0));
      const double rxy = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      const double cp = (rxy > 0.0) ? x[0] / rxy : 1.0;
      const double sp = (rxy > 0.0) ? x[1] / rxy : 0.0;
      return {sph[0] * st * cp + sph[1] * ct * cp - sph[2] * sp,
              sph[0] * st * sp + sph[1] * ct * sp + sph[2] * cp, sph[0] * ct - sph[1] * st};
    }
  }
}

// ---------------------------------------------------------------------------
// MuonAtmosphere
// ---------------------------------------------------------------------------

namespace {
// Linsley/CORSIKA US-Std layer boundaries [cm] and parameters.
constexpr double LIN_HB[5] = {0.0, 4e5, 1e6, 4e6, 1e7};
constexpr double LIN_B[5] = {1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0};   // g/cm^2
constexpr double LIN_C[5] = {994186.38, 878153.55, 636143.04, 772170.16, 1e9};  // cm
}  // namespace

MuonAtmosphere::MuonAtmosphere() : type_{'n'} {}

MuonAtmosphere::MuonAtmosphere(double rho0, double h_top)
    : type_{'u'}, rho0_{rho0}, h_top_{h_top} {}

MuonAtmosphere MuonAtmosphere::us_std() {
  MuonAtmosphere a;
  a.type_ = 'l';
  return a;
}

MuonAtmosphere::MuonAtmosphere(const std::vector<double>& h_cm, const std::vector<double>& rho,
                               double h_top)
    : type_{'t'}, h_top_{h_top}, h_{h_cm}, r_{rho} {
  if (h_.size() != r_.size() || h_.size() < 2) {
    throw std::invalid_argument("MuonAtmosphere table: h and rho must have equal size >= 2");
  }
  // Evenly-spaced tables (the common case: linspace-sampled MSIS/MCEq
  // profiles) get a direct-index fast path instead of a binary search.
  const double dh = (h_.back() - h_.front()) / (h_.size() - 1);
  uniform_h_ = dh > 0.0;
  for (std::size_t i = 0; i + 1 < h_.size() && uniform_h_; ++i) {
    if (std::abs(h_[i + 1] - h_[i] - dh) > 1e-9 * std::abs(dh)) uniform_h_ = false;
  }
  if (uniform_h_) inv_dh_ = 1.0 / dh;
}

double MuonAtmosphere::rho(double h) const {
  switch (type_) {
    case 'n':
      return 0.0;
    case 'u':
      // numpy UniformAtmosphere: rho0 for ALL h < h_top (incl. h < 0).
      return (h < h_top_) ? rho0_ : 0.0;
    case 'l': {
      if (h >= h_top_ || h < 0.0) return 0.0;
      int i = 0;
      while (i < 4 && h >= LIN_HB[i + 1]) ++i;
      if (i < 4) return LIN_B[i] / LIN_C[i] * std::exp(-h / LIN_C[i]);
      return LIN_B[4] / LIN_C[4];
    }
    default: {  // 't' — np.interp semantics (clamped), zero outside [0, h_top)
      if (h >= h_top_ || h < 0.0) return 0.0;
      if (h <= h_.front()) return r_.front();
      if (h >= h_.back()) return r_.back();
      int i;
      if (uniform_h_) {
        i = std::min(static_cast<int>((h - h_.front()) * inv_dh_), static_cast<int>(h_.size()) - 2);
      } else {
        i = hist_cell(h_, h);
      }
      const double f = (h - h_[i]) / (h_[i + 1] - h_[i]);
      return r_[i] + f * (r_[i + 1] - r_[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// MuonDedx
// ---------------------------------------------------------------------------

MuonDedx::MuonDedx(double a_const) : constant_{true}, a0_{a_const} {}

MuonDedx::MuonDedx(const std::vector<double>& e_kin_grid, const std::vector<double>& a)
    : constant_{false}, a0_{0.0}, a_{a} {
  if (e_kin_grid.size() != a.size() || a.size() < 2) {
    throw std::invalid_argument("MuonDedx table: e_kin and a must have equal size >= 2");
  }
  ln_e_.resize(e_kin_grid.size());
  for (std::size_t i = 0; i < e_kin_grid.size(); ++i) ln_e_[i] = std::log(e_kin_grid[i]);
}

double MuonDedx::at(double e_tot) const {
  if (constant_) return a0_;
  // numpy dedx_from_mceq: ekin = max(e_tot - m_mu, eg[0]); linear
  // interp of a vs ln(ekin), clamped at both ends.
  const double le = std::log(std::max(e_tot - M_MU, std::exp(ln_e_.front())));
  if (le <= ln_e_.front()) return a_.front();
  if (le >= ln_e_.back()) return a_.back();
  const int i =
      static_cast<int>(std::upper_bound(ln_e_.begin(), ln_e_.end(), le) - ln_e_.begin()) - 1;
  const double f = (le - ln_e_[i]) / (ln_e_[i + 1] - ln_e_[i]);
  return a_[i] + f * (a_[i + 1] - a_[i]);
}

// ---------------------------------------------------------------------------
// MuonTracer
// ---------------------------------------------------------------------------

MuonTracer::MuonTracer(MuonField field, MuonAtmosphere atmo, MuonDedx dedx, double r_ground_cm,
                       double h_escape_cm)
    : field_{std::move(field)},
      atmo_{std::move(atmo)},
      dedx_{std::move(dedx)},
      r_ground_{r_ground_cm},
      h_escape_{h_escape_cm} {}

// Per-worker deposit accumulator; merged into MuTraceResult afterwards.
namespace {
struct Accum {
  double deposited = 0.0;
  std::array<double, 5> fate_w = {0, 0, 0, 0, 0};
  long steps = 0;

  // bank
  std::vector<double> bank_x, bank_phat, bank_e, bank_pol, bank_sig, bank_w;
  std::vector<int8_t> bank_q;
  double bank_total_w = 0.0;
  // spectrum / angular
  std::vector<double> spec_w_pos, spec_wp_pos, spec_w_neg, spec_wp_neg;
  std::vector<double> ang_w_pos, ang_wp_pos, ang_ws2_pos;
  std::vector<double> ang_w_neg, ang_wp_neg, ang_ws2_neg;
  double ang_overflow = 0.0;

  void init(const DepositConfig& dep) {
    if (dep.kind == 's') {
      const std::size_t n = dep.e_edges.size() - 1;
      spec_w_pos.assign(n, 0.0);
      spec_wp_pos.assign(n, 0.0);
      spec_w_neg.assign(n, 0.0);
      spec_wp_neg.assign(n, 0.0);
    } else if (dep.kind == 'a') {
      const std::size_t n = (dep.e_edges.size() - 1) * (dep.d_edges.size() - 1);
      ang_w_pos.assign(n, 0.0);
      ang_wp_pos.assign(n, 0.0);
      ang_ws2_pos.assign(n, 0.0);
      ang_w_neg.assign(n, 0.0);
      ang_wp_neg.assign(n, 0.0);
      ang_ws2_neg.assign(n, 0.0);
    }
  }

  void deposit(const DepositConfig& dep, const double* x_mid, const double* phat, double e_mu,
               int8_t q, double pol, double sig, double dw) {
    switch (dep.kind) {
      case 'b':
        bank_total_w += dw;
        if (dw > dep.w_thresh) {
          bank_x.insert(bank_x.end(), x_mid, x_mid + 3);
          bank_phat.insert(bank_phat.end(), phat, phat + 3);
          bank_e.push_back(e_mu);
          bank_q.push_back(q);
          bank_pol.push_back(pol);
          bank_sig.push_back(sig);
          bank_w.push_back(dw);
        }
        break;
      case 's': {
        const int i = hist_bin(dep.e_edges, e_mu);
        if (i >= 0) {
          if (q > 0) {
            spec_w_pos[i] += dw;
            spec_wp_pos[i] += dw * pol;
          } else {
            spec_w_neg[i] += dw;
            spec_wp_neg[i] += dw * pol;
          }
        }
        break;
      }
      case 'a': {
        double cd = dot3(phat, dep.axis.data());
        cd = std::min(std::max(cd, -1.0), 1.0);
        const double delta = std::acos(cd);
        if (delta >= dep.d_edges.back()) {
          ang_overflow += dw;
          break;
        }
        const int ie = hist_bin(dep.e_edges, e_mu);
        const int id = hist_bin(dep.d_edges, delta);
        if (ie >= 0 && id >= 0) {
          const std::size_t k = static_cast<std::size_t>(ie) * (dep.d_edges.size() - 1) + id;
          if (q > 0) {
            ang_w_pos[k] += dw;
            ang_wp_pos[k] += dw * pol;
            ang_ws2_pos[k] += dw * sig * sig;
          } else {
            ang_w_neg[k] += dw;
            ang_wp_neg[k] += dw * pol;
            ang_ws2_neg[k] += dw * sig * sig;
          }
        }
        break;
      }
      default:
        break;
    }
  }
};

inline void add_into(std::vector<double>& dst, const std::vector<double>& src) {
  if (dst.empty()) {
    dst = src;
    return;
  }
  for (std::size_t i = 0; i < src.size(); ++i) dst[i] += src[i];
}
}  // namespace

void MuonTracer::trace_range(std::size_t i0, std::size_t i1, const std::vector<double>& x_in,
                             const std::vector<double>& p_in, const std::vector<int8_t>& q_in,
                             const std::vector<double>& pol_in, const std::vector<double>& w_in,
                             const TraceOptions& opt, const DepositConfig& dep, MuTraceResult& res,
                             std::mutex* merge_mutex) const {
  Accum acc;
  acc.init(dep);

  const double e_min = std::max(opt.e_min, M_MU * (1.0 + 1e-3));
  const double dt_max = (opt.dt_max > 0.0) ? opt.dt_max : std::numeric_limits<double>::infinity();
  const bool has_atmo = !atmo_.is_none();

  // ODE right-hand side (numpy _deriv, single ray).
  auto deriv = [&](const double* x, const double* p, int8_t q, double* dx, double* dp) {
    const double pm = std::max(norm3(p), 1e-12);
    const double e = std::sqrt(pm * pm + M_MU * M_MU);
    const double beta = pm / e;
    const double ph0 = p[0] / pm, ph1 = p[1] / pm, ph2 = p[2] / pm;
    dx[0] = C_CMS * beta * ph0;
    dx[1] = C_CMS * beta * ph1;
    dx[2] = C_CMS * beta * ph2;
    const auto b = field_.at({x[0], x[1], x[2]});
    const double kqb = KP * q * beta;
    dp[0] = kqb * (ph1 * b[2] - ph2 * b[1]);
    dp[1] = kqb * (ph2 * b[0] - ph0 * b[2]);
    dp[2] = kqb * (ph0 * b[1] - ph1 * b[0]);
    if (has_atmo) {
      const double h = norm3(x) - r_ground_;
      const double loss = dedx_.at(e) * C_CMS * atmo_.rho(h);
      dp[0] -= loss * ph0;
      dp[1] -= loss * ph1;
      dp[2] -= loss * ph2;
    }
  };

  for (std::size_t i = i0; i < i1; ++i) {
    double x[3] = {x_in[3 * i], x_in[3 * i + 1], x_in[3 * i + 2]};
    double p[3] = {p_in[3 * i], p_in[3 * i + 1], p_in[3 * i + 2]};
    const int8_t q = q_in[i];
    const double pol = pol_in[i];
    double w = w_in[i];
    double th2 = 0.0;
    int8_t fate = FATE_ALIVE;
    long step = 0;

    while (step < opt.max_steps) {
      ++step;
      const double pm = std::max(norm3(p), 1e-12);
      const double e0 = std::sqrt(pm * pm + M_MU * M_MU);
      const double beta = pm / e0;

      // per-ray dt from the fastest local scale
      const auto b0 = field_.at({x[0], x[1], x[2]});
      const double bmag = std::max(norm3(b0.data()), 1e-30);
      double dt = std::min(opt.ctl_bend * pm / (KP * beta * bmag), dt_max);
      dt = std::min(dt, opt.ctl_decay * e0 * TAU_MU / M_MU);
      dt = std::min(dt, opt.ctl_path / (beta * C_CMS));
      if (has_atmo) {
        const double h0 = norm3(x) - r_ground_;
        const double rho = atmo_.rho(h0);
        dt = std::min(dt, opt.ctl_loss * pm / (dedx_.at(e0) * C_CMS * std::max(rho, 1e-30)));
      }

      // RK4 (field/atmosphere re-evaluated at every stage, as numpy)
      double k1x[3], k1p[3], k2x[3], k2p[3], k3x[3], k3p[3], k4x[3], k4p[3];
      double xs[3], ps[3];
      deriv(x, p, q, k1x, k1p);
      for (int j = 0; j < 3; ++j) {
        xs[j] = x[j] + 0.5 * dt * k1x[j];
        ps[j] = p[j] + 0.5 * dt * k1p[j];
      }
      deriv(xs, ps, q, k2x, k2p);
      for (int j = 0; j < 3; ++j) {
        xs[j] = x[j] + 0.5 * dt * k2x[j];
        ps[j] = p[j] + 0.5 * dt * k2p[j];
      }
      deriv(xs, ps, q, k3x, k3p);
      for (int j = 0; j < 3; ++j) {
        xs[j] = x[j] + dt * k3x[j];
        ps[j] = p[j] + dt * k3p[j];
      }
      deriv(xs, ps, q, k4x, k4p);
      double x1[3], p1[3];
      for (int j = 0; j < 3; ++j) {
        x1[j] = x[j] + dt / 6.0 * (k1x[j] + 2.0 * k2x[j] + 2.0 * k3x[j] + k4x[j]);
        p1[j] = p[j] + dt / 6.0 * (k1p[j] + 2.0 * k2p[j] + 2.0 * k3p[j] + k4p[j]);
      }

      const double pm1 = std::max(norm3(p1), 1e-12);
      const double e1 = std::sqrt(pm1 * pm1 + M_MU * M_MU);
      const double e_mid = 0.5 * (e0 + e1);

      // decay weight (exact for E ~ const over the step)
      double lam = M_MU / (e_mid * TAU_MU);
      if (opt.decay_beta_one) lam *= 0.5 * (pm + pm1) / e_mid;
      const double surv = std::exp(-lam * dt);
      const double dw = w * (1.0 - surv);
      acc.deposited += dw;
      if (dep.kind != 'n') {
        const double x_mid[3] = {0.5 * (x[0] + x1[0]), 0.5 * (x[1] + x1[1]), 0.5 * (x[2] + x1[2])};
        double pv[3] = {p[0] + p1[0], p[1] + p1[1], p[2] + p1[2]};
        const double pvn = std::max(norm3(pv), 1e-12);
        pv[0] /= pvn;
        pv[1] /= pvn;
        pv[2] /= pvn;
        acc.deposit(dep, x_mid, pv, e_mid, q, pol, std::sqrt(th2), dw);
      }

      // MCS variance along the step (pre-pullback midpoint, as numpy)
      if (has_atmo) {
        const double xm[3] = {0.5 * (x[0] + x1[0]), 0.5 * (x[1] + x1[1]), 0.5 * (x[2] + x1[2])};
        const double h_mid = norm3(xm) - r_ground_;
        const double dxg = atmo_.rho(h_mid) * beta * C_CMS * dt;  // g/cm^2
        const double s = ES_MCS / (beta * pm);
        th2 += s * s * dxg / X0_AIR;
      }

      double w1 = w * surv;

      // ground crossing: pull the final state back onto the sphere
      const double r0n = norm3(x);
      const double r1 = norm3(x1);
      if (r1 < r_ground_) {
        const double f = (r0n - r_ground_) / std::max(r0n - r1, 1e-30);
        for (int j = 0; j < 3; ++j) {
          x1[j] = x[j] + f * (x1[j] - x[j]);
          p1[j] = p[j] + f * (p1[j] - p[j]);
        }
        w1 = w * std::pow(surv, f);
        fate = FATE_GROUND;
      } else if (e1 < e_min) {
        fate = FATE_STOPPED;
      } else if (w1 < opt.w_min) {
        fate = FATE_FADED;
      } else if (r1 > r_ground_ + h_escape_ && dot3(x1, p1) > 0.0) {
        fate = FATE_ESCAPED;
      }

      x[0] = x1[0];
      x[1] = x1[1];
      x[2] = x1[2];
      p[0] = p1[0];
      p[1] = p1[1];
      p[2] = p1[2];
      w = w1;

      if (fate != FATE_ALIVE) {
        acc.fate_w[fate - 1] += w;
        break;
      }
    }
    if (fate == FATE_ALIVE) acc.fate_w[4] += w;  // maxsteps
    acc.steps = std::max(acc.steps, step);

    res.x[3 * i] = x[0];
    res.x[3 * i + 1] = x[1];
    res.x[3 * i + 2] = x[2];
    res.p[3 * i] = p[0];
    res.p[3 * i + 1] = p[1];
    res.p[3 * i + 2] = p[2];
    res.w[i] = w;
    res.th2[i] = th2;
    res.fate[i] = fate;
  }

  // Merge accumulators into the shared result.  Workers write disjoint
  // final-state slots above (no race); this merge is the only shared
  // mutation and is serialised by the caller's mutex.
  std::lock_guard<std::mutex> lock(*merge_mutex);
  res.deposited += acc.deposited;
  for (int k = 0; k < 5; ++k) res.fate_w[k] += acc.fate_w[k];
  res.steps = std::max(res.steps, acc.steps);
  res.bank_total_w += acc.bank_total_w;
  res.bank_x.insert(res.bank_x.end(), acc.bank_x.begin(), acc.bank_x.end());
  res.bank_phat.insert(res.bank_phat.end(), acc.bank_phat.begin(), acc.bank_phat.end());
  res.bank_e.insert(res.bank_e.end(), acc.bank_e.begin(), acc.bank_e.end());
  res.bank_q.insert(res.bank_q.end(), acc.bank_q.begin(), acc.bank_q.end());
  res.bank_pol.insert(res.bank_pol.end(), acc.bank_pol.begin(), acc.bank_pol.end());
  res.bank_sig.insert(res.bank_sig.end(), acc.bank_sig.begin(), acc.bank_sig.end());
  res.bank_w.insert(res.bank_w.end(), acc.bank_w.begin(), acc.bank_w.end());
  add_into(res.spec_w_pos, acc.spec_w_pos);
  add_into(res.spec_wp_pos, acc.spec_wp_pos);
  add_into(res.spec_w_neg, acc.spec_w_neg);
  add_into(res.spec_wp_neg, acc.spec_wp_neg);
  add_into(res.ang_w_pos, acc.ang_w_pos);
  add_into(res.ang_wp_pos, acc.ang_wp_pos);
  add_into(res.ang_ws2_pos, acc.ang_ws2_pos);
  add_into(res.ang_w_neg, acc.ang_w_neg);
  add_into(res.ang_wp_neg, acc.ang_wp_neg);
  add_into(res.ang_ws2_neg, acc.ang_ws2_neg);
  res.ang_overflow += acc.ang_overflow;
}

MuTraceResult MuonTracer::trace(const std::vector<double>& x, const std::vector<double>& p,
                                const std::vector<int8_t>& q, const std::vector<double>& pol,
                                const std::vector<double>& w, const TraceOptions& opt,
                                const DepositConfig& dep) const {
  const std::size_t n = q.size();
  if (x.size() != 3 * n || p.size() != 3 * n || pol.size() != n || w.size() != n) {
    throw std::invalid_argument("MuonTracer::trace: inconsistent array lengths");
  }
  if ((dep.kind == 's' || dep.kind == 'a') && dep.e_edges.size() < 2) {
    throw std::invalid_argument("MuonTracer::trace: deposit needs e_edges");
  }
  if (dep.kind == 'a' && dep.d_edges.size() < 2) {
    throw std::invalid_argument("MuonTracer::trace: angular deposit needs d_edges");
  }
  DepositConfig dep_n = dep;  // normalise the angular axis
  if (dep.kind == 'a') {
    const double an = norm3(dep.axis.data());
    if (an <= 0.0) throw std::invalid_argument("MuonTracer::trace: angular axis is zero");
    for (int j = 0; j < 3; ++j) dep_n.axis[j] = dep.axis[j] / an;
  }

  MuTraceResult res;
  res.x.resize(3 * n);
  res.p.resize(3 * n);
  res.w.resize(n);
  res.th2.resize(n);
  res.fate.assign(n, FATE_ALIVE);
  {
    Accum shape;
    shape.init(dep_n);
    res.spec_w_pos = shape.spec_w_pos;
    res.spec_wp_pos = shape.spec_wp_pos;
    res.spec_w_neg = shape.spec_w_neg;
    res.spec_wp_neg = shape.spec_wp_neg;
    res.ang_w_pos = shape.ang_w_pos;
    res.ang_wp_pos = shape.ang_wp_pos;
    res.ang_ws2_pos = shape.ang_ws2_pos;
    res.ang_w_neg = shape.ang_w_neg;
    res.ang_wp_neg = shape.ang_wp_neg;
    res.ang_ws2_neg = shape.ang_ws2_neg;
  }

  int n_threads = opt.n_threads;
  if (n_threads <= 0) n_threads = static_cast<int>(std::thread::hardware_concurrency());
  if (n_threads < 1) n_threads = 1;
  n_threads = static_cast<int>(
      std::min<std::size_t>(static_cast<std::size_t>(n_threads), std::max<std::size_t>(n, 1)));

  std::mutex merge_mutex;
  if (n_threads == 1 || n < 64) {
    trace_range(0, n, x, p, q, pol, w, opt, dep_n, res, &merge_mutex);
  } else {
    // Workers write disjoint final-state slots directly into res;
    // accumulator merges are serialised by merge_mutex.
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    const std::size_t chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
      const std::size_t i0 = static_cast<std::size_t>(t) * chunk;
      const std::size_t i1 = std::min(n, i0 + chunk);
      if (i0 >= i1) break;
      workers.emplace_back(
          [this, i0, i1, &x, &p, &q, &pol, &w, &opt, &dep_n, &res, &merge_mutex]() {
            trace_range(i0, i1, x, p, q, pol, w, opt, dep_n, res, &merge_mutex);
          });
    }
    for (auto& th : workers) th.join();
  }

  res.weight_audit =
      res.deposited + res.fate_w[0] + res.fate_w[1] + res.fate_w[2] + res.fate_w[3] + res.fate_w[4];
  return res;
}

}  // namespace muon
