#pragma once

#include <array>
#include <vector>

// ---------------------------------------------------------------------------
// Grid dimensions for the 3-D IGRF lookup table
// ---------------------------------------------------------------------------
static constexpr int IGRF_NR = 64;       // log-spaced radial points (1–10 RE)
static constexpr int IGRF_NTHETA = 128;  // linear colatitude points [0, π]
static constexpr int IGRF_NPHI = 256;    // linear longitude points  [0, 2π)

// Total number of floats: 3 components × Nr × Ntheta × Nphi ≈ 6 M floats = 24 MB
static constexpr int IGRF_TABLE_SIZE = 3 * IGRF_NR * IGRF_NTHETA * IGRF_NPHI;

// ---------------------------------------------------------------------------
// Metadata needed for both the CPU generator and the GPU kernel
// ---------------------------------------------------------------------------
struct TableParams {
  float r_min;      // minimum radius [m]
  float r_max;      // maximum radius [m]
  float log_r_min;  // std::log(r_min), precomputed for fast lookup
  float log_r_max;  // std::log(r_max)
  int Nr;
  int Ntheta;
  int Nphi;
};

// ---------------------------------------------------------------------------
// CPU-side functions (declared here, defined in igrf_table.cpp)
// ---------------------------------------------------------------------------

// Forward declaration — igrf_table.cpp includes igrf.hpp internally.
class IGRF;

// Build the lookup table from an already-initialised IGRF object.
// Fills `params` with the grid metadata.
// Returns a flat float array with component-major layout [3][Nr][Ntheta][Nphi]:
//   index = comp * Nr*Ntheta*Nphi  +  ir * Ntheta*Nphi  +  itheta * Nphi  +  iphi
std::vector<float> generate_igrf_table(IGRF& igrf, TableParams& params);

// Trilinear interpolation inside the table (CPU version).
// r     : radial distance [m], must be in [r_min, r_max]
// theta : colatitude [rad], [0, π]
// phi   : longitude  [rad], arbitrary (wrapped to [0, 2π))
// Returns {Br, Btheta, Bphi} in Tesla (same units as IGRF::values()).
std::array<float, 3> table_lookup(const float* table, const TableParams& params, float r,
                                  float theta, float phi);

// Validate the table against direct IGRF evaluation at `N` random points.
// Returns the maximum relative error found (target: < 0.001 i.e. 0.1%).
double validate_igrf_table(const float* table, const TableParams& params, IGRF& igrf,
                           int N = 10000);
