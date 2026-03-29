# AVX-512 Vectorization Plan for gtracr

## Context

**Current state: NOT vectorized.** Each `std::thread` in BatchGMRC runs one trajectory
at a time — all scalar. The compiler may auto-vectorize the 6-element `std::array<double,6>`
operations (`std::transform` with `std::plus`), but 6 elements is too short for meaningful
SIMD. The real opportunity is processing **8 independent trajectories simultaneously**
in AVX-512 registers (8 × double = 512 bits).

**Why now:** The 3D IGRF table is already implemented (trilinear interpolation = pure
arithmetic, no branches) — this is the key enabler for SIMD, just as it is for GPU.
The tabulated path eliminates the Legendre recursion which has loop-carried dependencies
that prevent vectorization.

**Expected speedup:** 4–6× per core (8× theoretical, minus gather/masking/trig overhead).
At current 35k traj/s → ~140–200k traj/s on 8 cores.

---

## Target: RK4 Frozen-Field + Tabulated IGRF

This is the best SIMD candidate because:
- **1 B-field eval per step** (frozen field)
- **No inverse trig** (unlike Boris which needs acos + atan2)
- **No adaptive step control** (unlike RK45 where lanes reject at different rates)
- **Pure arithmetic** in ode_lrz_bf: cross products, divisions, one sqrt

Boris is a secondary target (moderate benefit, needs vectorized inverse trig).
RK45 is a poor SIMD target (step rejection causes lane divergence).

---

## Phase 1: SoA Data Layout

### New struct: `BatchState8` (8 trajectories packed for AVX-512)

```cpp
// Structure-of-Arrays for 8 simultaneous trajectories
struct alignas(64) BatchState8 {
    __m512d r, theta, phi;        // positions
    __m512d pr, ptheta, pphi;     // momenta
    __m512d t;                    // current time per trajectory
    __mmask8 active;              // bitmask: 1 = still integrating
};
```

**Why SoA:** AoS would put trajectory 0's r next to trajectory 0's theta — useless for
SIMD. SoA puts all 8 r-values contiguous → single `_mm512_load_pd` loads all 8.

### Files to modify:
- **New file**: `gtracr/lib/include/simd_batch.hpp` — BatchState8, SIMD helpers
- **New file**: `gtracr/lib/src/simd_batch.cpp` — batch_evaluate_rk4(), batch_table_lookup()

---

## Phase 2: Vectorized Table Lookup

The table is `float[3][Nr][Ntheta][Nphi]`. For 8 trajectories at 8 different (r,θ,φ):

```
1. Index computation (all vectorized):
   log_r  = _mm512_log_pd(r)          // 8 log() calls → ~20 cycles with SVML
   fr     = (log_r - log_r_min) * inv_range * (Nr-1)
   ft     = theta * (Nt-1) / pi
   fp     = phi * Np / (2*pi)          // + periodic wrap
   ir0    = _mm512_cvttpd_epi32(fr)    // truncate to int (8 indices)

2. Gather 8 corners × 3 components = 24 gathers:
   For each component c ∈ {Br, Bθ, Bφ}:
     For each corner (dr, dt, dp) ∈ {0,1}³:
       idx = c*slice + (ir0+dr)*Nt*Np + (it0+dt)*Np + (ip0+dp)
       v_corner = _mm256_i32gather_ps(table, idx_vec, 4)  // 8 float32 values

3. Trilinear interpolation (vectorized FMA):
   along_phi   = fma(wp, v001-v000, v000)     // 8 lerps in parallel
   along_theta = fma(wt, along_phi_1-along_phi_0, along_phi_0)
   along_r     = fma(wr, along_theta_1-along_theta_0, along_theta_0)

4. Convert float32 → float64:
   Br_d = _mm512_cvtps_pd(Br_f)  // 8 floats → 8 doubles
```

**Gather performance:** `_mm256_i32gather_ps` is ~12 cycles on Skylake-X. With 24
gathers per step, that's ~288 cycles. But the FMA interpolation overlaps with gather
latency. Total table lookup: ~400 cycles for 8 trajectories = **50 cycles/trajectory**
(vs ~120 cycles scalar → 2.4× speedup just from lookup).

### Files to modify:
- `gtracr/lib/gpu/igrf_table.hpp` — add `batch_table_lookup_8()` declaration
- `gtracr/lib/gpu/igrf_table.cpp` — implement with AVX-512 intrinsics

---

## Phase 3: Vectorized ODE (ode_lrz_bf)

The ODE right-hand side for 8 trajectories simultaneously:

```cpp
// All operations are element-wise across 8 trajectories
void ode_lrz_bf_8(
    const BatchState8& st,
    const __m512d Br, const __m512d Bt, const __m512d Bp,  // B-field (8 values each)
    __m512d& dr, __m512d& dtheta, __m512d& dphi,
    __m512d& dpr, __m512d& dptheta, __m512d& dpphi)
{
    // Lorentz factor: gamma = sqrt(1 + |p|²/(mc)²)
    __m512d p2 = _mm512_fmadd_pd(st.pr, st.pr,
                 _mm512_fmadd_pd(st.ptheta, st.ptheta,
                                 _mm512_mul_pd(st.pphi, st.pphi)));
    __m512d gamma = _mm512_sqrt_pd(_mm512_fmadd_pd(p2, inv_mc2, one));
    __m512d inv_rm = _mm512_div_pd(one, _mm512_mul_pd(mass, gamma));

    // sin(theta), cos(theta) — vectorized trig
    __m512d sin_t = _mm512_sin_pd(st.theta);  // SVML or Sleef
    __m512d cos_t = _mm512_cos_pd(st.theta);

    // Lorentz force cross-product terms (all FMA)
    __m512d dpr_lrz = _mm512_mul_pd(neg_q,
        _mm512_fmsub_pd(st.ptheta, Bp, _mm512_mul_pd(Bt, st.pphi)));
    // ... (remaining terms follow same pattern)

    // Spherical correction terms
    __m512d inv_r = _mm512_div_pd(one, st.r);
    __m512d cot_t = _mm512_div_pd(cos_t, sin_t);  // cos/sin = cot
    // ...

    // Scale by 1/(gamma * mass)
    dr     = _mm512_mul_pd(st.pr, inv_rm);
    dtheta = _mm512_mul_pd(_mm512_mul_pd(st.ptheta, inv_r), inv_rm);
    // ...
}
```

**Trig strategy:**
- **Option A — Intel SVML**: `_mm512_sin_pd` / `_mm512_cos_pd`. Available with
  `-mveclibabi=svml -lsvml` on GCC/Clang. Fast (~20 cycles for 8 values) but
  requires Intel runtime library.
- **Option B — Sleef**: Open-source, BSD-licensed. `Sleef_sind8_u10avx512f`.
  Similar performance. Portable. **Recommended.**
- **Option C — sincos combined**: `Sleef_sincosd8_u10avx512f` returns both
  sin and cos in one call — perfect since we always need both.

### Files to modify:
- `gtracr/lib/src/simd_batch.cpp` — vectorized ODE implementation

---

## Phase 4: Vectorized RK4 Integration Loop

```cpp
// Run 8 trajectories through RK4 in lockstep
void batch_rk4_8(
    BatchState8& st,
    const float* table, const TableParams& tp,
    double h, int max_iter,
    double escape_radius, double start_alt_plus_RE,
    /* outputs */ bool escaped[8], double final_time[8])
{
    __m512d esc_r = _mm512_set1_pd(escape_radius);
    __m512d atm_r = _mm512_set1_pd(start_alt_plus_RE);
    __m512d hv    = _mm512_set1_pd(h);
    __m512d half_h = _mm512_set1_pd(0.5 * h);

    for (int step = 0; step < max_iter; ++step) {
        if (st.active == 0) break;  // all 8 done

        // B-field lookup for 8 positions
        __m512d Br, Bt, Bp;
        batch_table_lookup_8(table, tp, st.r, st.theta, st.phi, Br, Bt, Bp);

        // k1 = h * f(t, y, B)
        __m512d k1_r, k1_t, k1_p, k1_pr, k1_pt, k1_pp;
        ode_lrz_bf_8(st, Br, Bt, Bp, k1_r, k1_t, k1_p, k1_pr, k1_pt, k1_pp);
        // scale by h...

        // k2 = h * f(t+h/2, y+k1/2, B)  [frozen B]
        // ... (same pattern, with intermediate state)

        // k3, k4 similarly

        // Combine: y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        // (all masked to only update active lanes)
        st.r = _mm512_mask_add_pd(st.r, st.active, st.r, dr_total);
        // ...

        // Termination check
        __mmask8 esc = _mm512_cmp_pd_mask(st.r, esc_r, _CMP_GT_OQ);
        __mmask8 atm = _mm512_cmp_pd_mask(st.r, atm_r, _CMP_LT_OQ);

        // Record escaped status
        // ...

        // Deactivate terminated lanes
        st.active &= ~(esc | atm);
    }
}
```

**Early termination strategy — "let them idle":**
When a trajectory terminates, zero its mask bit. The SIMD instructions still execute
for all 8 lanes but masked writes prevent state corruption. The terminated lane
idles until all 8 finish. Waste is bounded: if trajectories have similar length
(which they often do at the same rigidity), waste is minimal. Worst case: 7/8 idle
= 12.5% efficiency. Average case with mixed trajectory lengths: ~70–80% efficiency.

**Alternative — lane refill:** When a lane terminates, load a new trajectory into it.
More complex (bookkeeping which lane maps to which output) but eliminates idle waste.
Worth implementing as a second pass if profiling shows significant idle time.

### Files to modify:
- `gtracr/lib/src/simd_batch.cpp` — main integration loop

---

## Phase 5: Integration with BatchGMRC

Modify `thread_worker()` in `BatchGMRC.cpp`:

```cpp
// Current: process 1 trajectory at a time
while (successes < quota) {
    // ... generate direction
    double rc = find_cutoff_rigidity_bisect(tracer, ...);  // scalar
}

// New: process 8 directions at a time
while (successes < quota) {
    // Generate 8 random directions
    DirectionIC ics[8];
    for (int i = 0; i < 8; ++i)
        ics[i] = compute_direction_ic(ctx, zen[i], az[i], ref_mom);

    // Bisect all 8 in lockstep (test same rigidity level for all 8)
    double rc[8];
    find_cutoff_rigidity_bisect_8(batch_tracer, ics, rigidities, rc);
}
```

**Bisection batching:** All 8 directions start with the same bisection bounds
(lo=0, hi=N-1). At each bisection step, all 8 test the same rigidity index (mid).
After evaluation, each lane independently updates its lo/hi based on escape status.
Lanes converge at different rates but share the same rigidity test — this works
because `escapes(mid)` is evaluated for all 8 simultaneously via `batch_rk4_8`.

### Files to modify:
- `gtracr/lib/src/BatchGMRC.cpp` — batch bisection, 8-wide thread worker

---

## Phase 6: Build System

```meson
# meson.build additions
avx512_args = []
if cpp.has_argument('-mavx512f')
  avx512_args += ['-mavx512f', '-mavx512dq']
  # Optional: SVML for vectorized trig
  if cpp.has_argument('-mveclibabi=svml')
    avx512_args += ['-mveclibabi=svml']
  endif
endif

# Compile SIMD sources with AVX-512 flags
simd_sources = files('src/simd_batch.cpp')
# ... add to library with avx512_args
```

**Runtime dispatch:** Compile two versions of the batch functions — one with AVX-512,
one scalar fallback. Check `__builtin_cpu_supports("avx512f")` at runtime.
Use `__attribute__((target("avx512f")))` for GCC/Clang function multiversioning.

**Sleef integration:** Add as a meson subproject or find_library(). Sleef has
native meson support.

### Files to modify:
- `meson.build` — AVX-512 detection, Sleef dependency
- `gtracr/lib/meson.build` — compile SIMD sources with correct flags

---

## New Files

| File | Purpose |
|------|---------|
| `gtracr/lib/include/simd_batch.hpp` | BatchState8 struct, SIMD helper declarations |
| `gtracr/lib/src/simd_batch.cpp` | batch_rk4_8(), batch_table_lookup_8(), ode_lrz_bf_8() |

## Modified Files

| File | Changes |
|------|---------|
| `gtracr/lib/gpu/igrf_table.hpp` | Add batch_table_lookup_8() declaration |
| `gtracr/lib/gpu/igrf_table.cpp` | AVX-512 gather-based table lookup |
| `gtracr/lib/src/BatchGMRC.cpp` | 8-wide thread worker, batched bisection |
| `gtracr/lib/include/BatchGMRC.hpp` | Updated params/API |
| `gtracr/lib/src/pybind11_wrapper.cpp` | Expose SIMD availability check |
| `meson.build` | AVX-512 flags, Sleef dependency |
| `gtracr/lib/meson.build` | Compile SIMD sources |

---

## Verification

1. **Correctness**: Run `pytest gtracr/tests/ -v` — all existing tests must pass
2. **Numerical agreement**: SIMD batch results vs scalar results must agree within
   floating-point tolerance (~1e-12 relative for double precision)
3. **Benchmark**: Compare `evaluate_batch()` throughput before/after
4. **Edge cases**: Test with < 8 remaining trajectories (partial batch), trajectories
   that terminate in 1 step, trajectories that run full max_iter
5. **Fallback**: Verify scalar fallback works on non-AVX512 CPUs

---

## Estimated Effort

| Phase | Effort | Speedup contribution |
|-------|--------|---------------------|
| SoA layout + batch struct | Small | Foundation |
| Vectorized table lookup | Medium | ~2× (gather + interp) |
| Vectorized ODE | Medium | ~3–4× (arithmetic) |
| Vectorized RK4 loop | Medium | Ties it together |
| BatchGMRC integration | Medium | ~1.2× (eliminates scalar overhead) |
| Build system + dispatch | Small | Portability |
| **Total** | **~1 week** | **4–6× per core** |
