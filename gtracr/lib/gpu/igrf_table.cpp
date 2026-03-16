#include "igrf_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "igrf.hpp"
#include "constants.hpp"

// ---------------------------------------------------------------------------
// generate_igrf_table
// ---------------------------------------------------------------------------
std::vector<float> generate_igrf_table(IGRF& igrf, TableParams& params) {
    const int Nr     = IGRF_NR;
    const int Ntheta = IGRF_NTHETA;
    const int Nphi   = IGRF_NPHI;

    // Radial range: 1 RE to 10 RE (in meters)
    const double r_min = constants::RE;           // 6,371,200 m
    const double r_max = 10.0 * constants::RE;    // 63,712,000 m

    params.r_min      = static_cast<float>(r_min);
    params.r_max      = static_cast<float>(r_max);
    params.log_r_min  = static_cast<float>(std::log(r_min));
    params.log_r_max  = static_cast<float>(std::log(r_max));
    params.Nr         = Nr;
    params.Ntheta     = Ntheta;
    params.Nphi       = Nphi;

    std::vector<float> table(IGRF_TABLE_SIZE, 0.0f);

    const int slice = Nr * Ntheta * Nphi;  // size of one component block

    for (int ir = 0; ir < Nr; ++ir) {
        // Log-spaced r: r[i] = r_min * (r_max/r_min)^(i/(Nr-1))
        double t = static_cast<double>(ir) / (Nr - 1);
        double r = r_min * std::pow(r_max / r_min, t);

        for (int itheta = 0; itheta < Ntheta; ++itheta) {
            double theta = itheta * constants::pi / (Ntheta - 1);

            for (int iphi = 0; iphi < Nphi; ++iphi) {
                double phi = iphi * 2.0 * constants::pi / Nphi;

                std::array<double, 3> bfield = igrf.values(r, theta, phi);

                int base = ir * Ntheta * Nphi + itheta * Nphi + iphi;
                table[0 * slice + base] = static_cast<float>(bfield[0]);  // Br
                table[1 * slice + base] = static_cast<float>(bfield[1]);  // Btheta
                table[2 * slice + base] = static_cast<float>(bfield[2]);  // Bphi
            }
        }
    }

    return table;
}

// ---------------------------------------------------------------------------
// table_lookup  (CPU version — trilinear interpolation)
// ---------------------------------------------------------------------------
std::array<float, 3> table_lookup(const float* table, const TableParams& params,
                                   float r, float theta, float phi) {
    const int Nr     = params.Nr;
    const int Nt     = params.Ntheta;
    const int Np     = params.Nphi;
    const int slice  = Nr * Nt * Np;

    // Guard against NaN/Inf/negative inputs from RK45 intermediate stages.
    // std::isfinite returns false for NaN and Inf; r <= 0 would produce
    // NaN from std::log.  Return zero field to let the adaptive step-size
    // controller reject the step.
    if (!std::isfinite(r) || !std::isfinite(theta) || !std::isfinite(phi) || r <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    // ---- radial index (log-spaced) ----------------------------------------
    float log_r = std::log(r);
    float fr = (log_r - params.log_r_min) / (params.log_r_max - params.log_r_min)
               * (Nr - 1);
    // Clamp so ir0 is in [0, Nr-2], ensuring ir0+1 < Nr for trilinear interp.
    int ir0 = static_cast<int>(std::max(0.0f, std::min(static_cast<float>(Nr - 2), std::floor(fr))));
    float wr = std::max(0.0f, std::min(1.0f, fr - static_cast<float>(ir0)));

    // ---- theta index (linear) ---------------------------------------------
    float ft = theta * static_cast<float>(Nt - 1) / static_cast<float>(constants::pi);
    int it0 = static_cast<int>(std::max(0.0f, std::min(static_cast<float>(Nt - 2), std::floor(ft))));
    float wt = std::max(0.0f, std::min(1.0f, ft - static_cast<float>(it0)));

    // ---- phi index (linear, periodic) -------------------------------------
    // Normalise phi to [0, 2π)
    const float two_pi = static_cast<float>(2.0 * constants::pi);
    phi = phi - two_pi * std::floor(phi / two_pi);
    float fp = phi * static_cast<float>(Np) / two_pi;
    int ip0 = static_cast<int>(std::max(0.0f, std::min(static_cast<float>(Np - 1), std::floor(fp)))) % Np;
    int ip1 = (ip0 + 1) % Np;   // periodic wrap
    float wp = std::max(0.0f, std::min(1.0f, fp - std::floor(fp)));

    // ---- helper lambda to index into the flat array -----------------------
    auto idx = [&](int comp, int ir, int it, int ip) -> int {
        return comp * slice + ir * Nt * Np + it * Np + ip;
    };

    // ---- trilinear interpolation over (r, theta, phi) ---------------------
    std::array<float, 3> result;
    for (int c = 0; c < 3; ++c) {
        float v000 = table[idx(c, ir0,   it0,   ip0)];
        float v001 = table[idx(c, ir0,   it0,   ip1)];
        float v010 = table[idx(c, ir0,   it0+1, ip0)];
        float v011 = table[idx(c, ir0,   it0+1, ip1)];
        float v100 = table[idx(c, ir0+1, it0,   ip0)];
        float v101 = table[idx(c, ir0+1, it0,   ip1)];
        float v110 = table[idx(c, ir0+1, it0+1, ip0)];
        float v111 = table[idx(c, ir0+1, it0+1, ip1)];

        float along_phi_00 = (1.0f - wp) * v000 + wp * v001;
        float along_phi_01 = (1.0f - wp) * v010 + wp * v011;
        float along_phi_10 = (1.0f - wp) * v100 + wp * v101;
        float along_phi_11 = (1.0f - wp) * v110 + wp * v111;

        float along_theta_0 = (1.0f - wt) * along_phi_00 + wt * along_phi_01;
        float along_theta_1 = (1.0f - wt) * along_phi_10 + wt * along_phi_11;

        result[c] = (1.0f - wr) * along_theta_0 + wr * along_theta_1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// validate_igrf_table
// ---------------------------------------------------------------------------
double validate_igrf_table(const float* table, const TableParams& params,
                            IGRF& igrf, int N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rand_r(params.r_min, params.r_max);
    std::uniform_real_distribution<double> rand_theta(0.0, constants::pi);
    std::uniform_real_distribution<double> rand_phi(0.0, 2.0 * constants::pi);

    double max_rel_err = 0.0;
    const double eps = 1e-30;  // guard against near-zero denominator

    for (int i = 0; i < N; ++i) {
        double r     = rand_r(rng);
        double theta = rand_theta(rng);
        double phi   = rand_phi(rng);

        std::array<double, 3> direct = igrf.values(r, theta, phi);
        std::array<float, 3>  interp = table_lookup(table, params,
                                                     static_cast<float>(r),
                                                     static_cast<float>(theta),
                                                     static_cast<float>(phi));

        for (int c = 0; c < 3; ++c) {
            double ref = direct[c];
            double err = std::abs(static_cast<double>(interp[c]) - ref);
            double rel = err / (std::abs(ref) + eps);
            if (rel > max_rel_err) {
                max_rel_err = rel;
            }
        }
    }

    return max_rel_err;
}
