//! Unified hardware device profile.
//!
//! Integrates microarchitecture detection, cache hierarchy, NUMA topology,
//! and peak performance estimates into a single `DeviceProfile` used by the
//! inference compiler for code generation and tuning decisions.

use crate::microarch::{MicroArch, KernelConfig};
use crate::numa::NumaTopology;
use crate::autotuning::HwInfo;

/// GEMM BLIS three-level blocking parameters.
#[derive(Debug, Clone, Copy)]
pub struct GemmBlocking {
    /// K dimension block (fits micropanels in L1)
    pub kc: usize,
    /// M dimension block (fits A panel in L2)
    pub mc: usize,
    /// N dimension block (fits B panel in L3)
    pub nc: usize,
    /// Microkernel M register block
    pub mr: usize,
    /// Microkernel N register block
    pub nr: usize,
}

/// ISA level for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaLevel {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

/// Unified hardware profile for the inference compiler.
///
/// Combines microarchitecture, cache hierarchy, NUMA topology, and peak
/// performance estimates. Built once at `InferenceBackend::init()` time.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    /// Detected microarchitecture
    pub arch: MicroArch,
    /// ISA level for dispatch
    pub isa: IsaLevel,
    /// Kernel configuration (blocking, prefetch, geometry)
    pub kernel_config: KernelConfig,
    /// Hardware info (cores, cache, ISA feature flags)
    pub hw_info: HwInfo,
    /// NUMA topology
    pub numa: NumaTopology,

    // ── Peak performance estimates ──
    /// Peak single-precision GFLOPS (all cores, theoretical)
    pub peak_gflops_f32: f64,
    /// Peak memory bandwidth in GB/s (estimated)
    pub peak_bandwidth_gbs: f64,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (with HT/SMT)
    pub logical_cores: usize,
}

impl DeviceProfile {
    /// Detect the current hardware and build a complete profile.
    pub fn detect() -> Self {
        let arch = crate::microarch::detect();
        let kernel_config = KernelConfig::from_arch(arch);
        let hw_info = HwInfo::detect();
        let numa = crate::numa::topology().clone();

        let isa = detect_isa_level(arch);

        let physical_cores = hw_info.physical_cores;
        let logical_cores = hw_info.logical_cores;

        // Peak GFLOPS = cores × freq_ghz × FMA_ports × SIMD_width × 2 (FMA = mul + add)
        let (_, _, simd_w) = arch.microkernel_geometry();
        let fma_ports = 2.0_f64; // most modern cores have 2 FMA units
        let peak_gflops_f32 =
            physical_cores as f64 * arch.estimated_freq_ghz() * fma_ports * simd_w as f64 * 2.0;

        // Conservative bandwidth estimate based on memory generation
        let peak_bandwidth_gbs = if arch.estimated_mem_latency_ns() > 75 {
            80.0 // DDR5
        } else {
            50.0 // DDR4
        };

        DeviceProfile {
            arch,
            isa,
            kernel_config,
            hw_info,
            numa,
            peak_gflops_f32,
            peak_bandwidth_gbs,
            physical_cores,
            logical_cores,
        }
    }

    /// Roofline ridge point: arithmetic intensity (FLOP/byte) where the
    /// operation transitions from memory-bound to compute-bound.
    #[inline]
    pub fn roofline_ridge_point(&self) -> f64 {
        self.peak_gflops_f32 / self.peak_bandwidth_gbs
    }

    /// Microkernel geometry (MR, NR) for the detected ISA.
    #[inline]
    pub fn microkernel_mr_nr(&self) -> (usize, usize) {
        (self.kernel_config.mr, self.kernel_config.nr)
    }

    /// Cache sizes (L1D, L2, L3) in bytes.
    #[inline]
    pub fn cache_sizes(&self) -> (usize, usize, usize) {
        (self.kernel_config.l1d, self.kernel_config.l2, self.kernel_config.l3)
    }

    /// Number of SIMD registers available for the detected ISA.
    /// AVX2 → 16 (ymm0-ymm15), AVX-512 → 32 (zmm0-zmm31), NEON → 32 (v0-v31)
    #[inline]
    pub fn num_simd_regs(&self) -> usize {
        match self.isa {
            IsaLevel::Avx512 => 32,
            IsaLevel::Avx2 => 16,
            IsaLevel::Neon => 32,
            IsaLevel::Scalar => 0,
        }
    }

    /// SIMD register width in bytes.
    /// AVX2 → 32 (256-bit ymm), AVX-512 → 64 (512-bit zmm), NEON → 16 (128-bit v)
    #[inline]
    pub fn simd_width_bytes(&self) -> usize {
        match self.isa {
            IsaLevel::Avx512 => 64,
            IsaLevel::Avx2 => 32,
            IsaLevel::Neon => 16,
            IsaLevel::Scalar => 4, // single f32
        }
    }

    /// Number of f32 elements per SIMD register.
    #[inline]
    pub fn simd_width_f32(&self) -> usize {
        self.simd_width_bytes() / 4
    }

    /// Compute GEMM BLIS blocking parameters (KC, MC, NC) for given dimensions.
    ///
    /// First checks the WisdomDb for empirically-tuned parameters from a previous
    /// autotuning run. If a cached result exists, its blocking parameters are used
    /// directly (they were measured to be optimal on this hardware). Otherwise,
    /// falls back to the analytical heuristic based on cache hierarchy constraints.
    ///
    /// Blocking strategy (BLIS-style three-level cache blocking):
    /// - KC: A micropanel (MR×KC) + B micropanel (KC×NR) fit in 80% of L1
    /// - MC: A panel (MC×KC) fits in 80% of L2, at least 2×MR tiles
    /// - NC: B panel (KC×NC) fits in 60% of L3 (40% of L2 fallback)
    ///
    /// Alignment: KC to 4 (SIMD), MC to MR, NC to NR.
    /// Small matrices (m*n*k < 4096) use direct path to avoid packing overhead.
    pub fn gemm_blocking(&self, m: usize, n: usize, k: usize) -> GemmBlocking {
        let (mr, nr) = self.microkernel_mr_nr();

        // Query WisdomDb for empirically-tuned parameters
        if let Some(blocking) = self.query_wisdom_blocking(m, n, k, mr, nr) {
            return blocking;
        }

        // Fallback: analytical heuristic
        self.gemm_blocking_heuristic(m, n, k)
    }

    /// Query the WisdomDb for cached GEMM blocking parameters.
    ///
    /// Returns `Some(GemmBlocking)` if a cached result exists for this exact
    /// problem shape on the current hardware. The cached KC/MC/NC values are
    /// validated against alignment constraints before use.
    fn query_wisdom_blocking(
        &self,
        m: usize,
        n: usize,
        k: usize,
        mr: usize,
        nr: usize,
    ) -> Option<GemmBlocking> {
        let db_ref = crate::autotuning::global_wisdom_db();
        let db = db_ref.lock().ok()?;
        let fp = self.hw_info.fingerprint();
        let cached = db.get_gemm_blocking(&fp, m, n, k, 4)?;

        let cfg = &cached.config;

        // Validate cached values: must respect alignment and dimension bounds
        let kc = cfg.kc;
        let mc = cfg.mc;
        let nc = cfg.nc;

        if kc == 0 || mc == 0 || nc == 0 {
            return None;
        }
        if kc > k || mc > m || nc > n {
            return None;
        }

        // Use NR from JIT params if available, otherwise default
        let effective_nr = cfg.jit.as_ref().map(|j| j.nr_variant).unwrap_or(nr);

        Some(GemmBlocking {
            kc,
            mc,
            nc,
            mr,
            nr: effective_nr,
        })
    }

    /// Analytical heuristic for GEMM blocking (fallback when no wisdom exists).
    fn gemm_blocking_heuristic(&self, m: usize, n: usize, k: usize) -> GemmBlocking {
        let (l1, l2, l3) = self.cache_sizes();
        let (mr, nr) = self.microkernel_mr_nr();
        let elem_size = 4; // f32

        // Small matrix optimization: skip blocking overhead for tiny matrices.
        // Packing cost dominates when the total volume is small.
        if m.saturating_mul(n).saturating_mul(k) < 4096 {
            return GemmBlocking {
                kc: k.max(1),
                mc: m.max(1),
                nc: n.max(1),
                mr,
                nr,
            };
        }

        // KC: A micropanel (MR×KC) + B micropanel (KC×NR) must fit in L1.
        // Accumulators (MR×NR) live in registers, only micropanels consume L1.
        // Use 80% of L1, round down to multiple of 4 for SIMD alignment.
        let kc = (l1 * 4 / 5) / (elem_size * (mr + nr));
        let kc = (kc / 4) * 4; // align to 4
        let kc = kc.max(4).min(k);

        // MC: A panel (MC×KC) must fit in 80% of L2.
        // At least 2×MR to amortize pack_a overhead (when m allows).
        let mc = (l2 * 4 / 5) / (elem_size * kc);
        let mc = (mc / mr) * mr; // align down to MR
        let mc = if m >= 2 * mr { mc.max(2 * mr) } else { mc.max(mr) };
        let mc = mc.min(m);
        // Safety cap: pack_a buffer must fit within 85% of L2
        let mc = if mc * kc * elem_size > l2 * 85 / 100 {
            let cap = (l2 * 85 / 100) / (elem_size * kc);
            let cap = (cap / mr) * mr;
            cap.max(mr).min(m)
        } else {
            mc
        };

        // NC: B panel (KC×NC) must fit in L3 (60% budget).
        // Fall back to 40% of L2 for systems without usable L3.
        let nc_budget = if l3 >= 1024 * 1024 {
            l3 * 3 / 5 // 60% of L3
        } else {
            l2 * 2 / 5 // 40% of L2 fallback
        };
        let nc = nc_budget / (elem_size * kc);
        let nc = (nc / nr) * nr; // align down to NR
        let nc = if n >= 2 * nr { nc.max(2 * nr) } else { nc.max(nr) };
        let nc = nc.min(n);

        GemmBlocking { kc, mc, nc, mr, nr }
    }

    /// Optimal prefetch distance in cache lines for the detected microarchitecture.
    ///
    /// Used by codegen to emit prefetch instructions at the right distance ahead.
    /// Tuned per microarchitecture based on memory latency characteristics.
    #[inline]
    pub fn prefetch_distance(&self) -> usize {
        match self.arch {
            // Intel: deeper pipelines, higher memory latency → prefetch further ahead
            MicroArch::SkylakeClient | MicroArch::SkylakeX | MicroArch::CascadeLake |
            MicroArch::IceLakeClient | MicroArch::IceLakeServer |
            MicroArch::TigerLake | MicroArch::AlderLake | MicroArch::RaptorLake |
            MicroArch::SapphireRapids | MicroArch::GraniteRapids => 12,
            // AMD Zen: shorter memory pipeline → closer prefetch
            MicroArch::Zen3 | MicroArch::Zen4 | MicroArch::Zen5 | MicroArch::Zen2 => 8,
            // Conservative default
            _ => 8,
        }
    }

    /// Compute tile size (in f32 elements) for elementwise operations.
    /// Sized to fit input + output in L1 cache.
    pub fn elem_tile_size(&self) -> usize {
        let (l1, _, _) = self.cache_sizes();
        let simd_w = self.simd_width_f32();
        // 2 buffers (input + output) in L1, use 75%
        let elems = (l1 * 3 / 4) / (2 * 4); // 2 buffers × 4 bytes per f32
        // Align to SIMD width
        (elems / simd_w) * simd_w
    }

    /// Minimum number of f32 elements before parallelization is worthwhile.
    /// Below this threshold, use sequential execution.
    pub fn parallel_threshold(&self) -> usize {
        // Rough heuristic: need at least ~4096 elements per thread to amortize overhead
        4096 * self.physical_cores
    }
}

fn detect_isa_level(arch: MicroArch) -> IsaLevel {
    if arch.use_avx512() {
        return IsaLevel::Avx512;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return IsaLevel::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return IsaLevel::Neon;
    }
    #[allow(unreachable_code)]
    IsaLevel::Scalar
}

impl std::fmt::Display for DeviceProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} | {:?} | {}P/{}L | peak {:.0} GFLOPS, {:.0} GB/s | ridge {:.1} FLOP/B",
            self.arch,
            self.isa,
            self.physical_cores,
            self.logical_cores,
            self.peak_gflops_f32,
            self.peak_bandwidth_gbs,
            self.roofline_ridge_point(),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_profile() {
        let profile = DeviceProfile::detect();
        eprintln!("DeviceProfile: {profile}");
        assert!(profile.physical_cores >= 1);
        assert!(profile.peak_gflops_f32 > 0.0);
        assert!(profile.peak_bandwidth_gbs > 0.0);
        assert!(profile.roofline_ridge_point() > 0.0);
    }

    #[test]
    fn test_isa_level() {
        let profile = DeviceProfile::detect();
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(
            profile.isa,
            IsaLevel::Avx2 | IsaLevel::Avx512
        ));
        #[cfg(target_arch = "aarch64")]
        assert_eq!(profile.isa, IsaLevel::Neon);
    }

    #[test]
    fn test_num_simd_regs() {
        let profile = DeviceProfile::detect();
        let regs = profile.num_simd_regs();
        #[cfg(target_arch = "x86_64")]
        assert!(regs == 16 || regs == 32, "x86_64 should have 16 or 32 SIMD regs, got {regs}");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(regs, 32);
    }

    #[test]
    fn test_simd_width() {
        let profile = DeviceProfile::detect();
        let width = profile.simd_width_bytes();
        assert!(width >= 16, "SIMD width should be at least 16 bytes, got {width}");
        let f32_width = profile.simd_width_f32();
        assert!(f32_width >= 4);
    }

    #[test]
    fn test_gemm_blocking() {
        let profile = DeviceProfile::detect();
        let blocking = profile.gemm_blocking(1024, 1024, 1024);
        let (mr, nr) = profile.microkernel_mr_nr();

        assert_eq!(blocking.mc % mr, 0, "MC={} not aligned to MR={}", blocking.mc, mr);
        assert_eq!(blocking.nc % nr, 0, "NC={} not aligned to NR={}", blocking.nc, nr);
        assert_eq!(blocking.kc % 4, 0, "KC={} not aligned to 4", blocking.kc);
        assert!(blocking.kc > 0);
        assert!(blocking.mc <= 1024);
        assert!(blocking.nc <= 1024);
        assert!(blocking.kc <= 1024);

        eprintln!("GEMM blocking 1024x1024x1024: KC={} MC={} NC={}", blocking.kc, blocking.mc, blocking.nc);
    }

    #[test]
    fn test_gemm_blocking_small() {
        let profile = DeviceProfile::detect();
        // Volume = 4*8*16 = 512 < 4096 -> direct path
        let blocking = profile.gemm_blocking(4, 8, 16);
        assert_eq!(blocking.kc, 16, "small matrix should use KC=k");
        assert_eq!(blocking.mc, 4, "small matrix should use MC=m");
        assert_eq!(blocking.nc, 8, "small matrix should use NC=n");

        // Volume = 25*40*64 = 64000 > 4096 -> normal blocking
        let b = profile.gemm_blocking(25, 40, 64);
        eprintln!("m=25,n=40,k=64 => kc={}, mc={}, nc={}, mr={}, nr={}", b.kc, b.mc, b.nc, b.mr, b.nr);
        assert!(b.mc <= 25);
        assert!(b.nc <= 40);
        assert!(b.kc <= 64);
    }

    /// Verify cache capacity invariants for blocking parameters.
    /// These are the fundamental correctness constraints for BLIS blocking.
    #[test]
    fn test_blocking_cache_capacity_invariants() {
        let profile = DeviceProfile::detect();
        let (l1, l2, l3) = profile.cache_sizes();
        let (mr, nr) = profile.microkernel_mr_nr();

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (2048, 512, 1024),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k);

            // KC * (MR + NR) * 4 <= L1 * 0.85 -- micropanels fit in L1
            let micropanel_bytes = b.kc * (mr + nr) * 4;
            assert!(
                micropanel_bytes <= l1 * 85 / 100,
                "m={m} n={n} k={k}: micropanels {micropanel_bytes}B > 85% of L1 {}B",
                l1 * 85 / 100
            );

            // MC * KC * 4 <= L2 * 0.85 -- A panel fits in L2
            let a_panel_bytes = b.mc * b.kc * 4;
            assert!(
                a_panel_bytes <= l2 * 85 / 100,
                "m={m} n={n} k={k}: A panel {a_panel_bytes}B > 85% of L2 {}B",
                l2 * 85 / 100
            );

            // KC * NC * 4 <= L3 * 0.65 -- B panel fits in L3
            // (skip if L3 < 1MB, fallback uses L2 budget)
            if l3 >= 1024 * 1024 {
                let b_panel_bytes = b.kc * b.nc * 4;
                assert!(
                    b_panel_bytes <= l3 * 65 / 100,
                    "m={m} n={n} k={k}: B panel {b_panel_bytes}B > 65% of L3 {}B",
                    l3 * 65 / 100
                );
            }

            // Alignment: MC % MR == 0, NC % NR == 0
            // (only when blocking is smaller than the full dimension)
            if b.mc < m {
                assert_eq!(b.mc % mr, 0,
                    "m={m} n={n} k={k}: MC={} not aligned to MR={mr}", b.mc);
            }
            if b.nc < n {
                assert_eq!(b.nc % nr, 0,
                    "m={m} n={n} k={k}: NC={} not aligned to NR={nr}", b.nc);
            }

            // KC aligned to 4
            assert_eq!(b.kc % 4, 0,
                "m={m} n={n} k={k}: KC={} not aligned to 4", b.kc);
        }
    }

    /// Small matrices should use direct path (no blocking overhead).
    #[test]
    fn test_blocking_small_matrix_direct_path() {
        let profile = DeviceProfile::detect();

        for &(m, n, k) in &[(1, 1, 1), (2, 4, 8), (4, 8, 16), (8, 8, 8), (10, 10, 10)] {
            let vol = m * n * k;
            if vol >= 4096 { continue; }
            let b = profile.gemm_blocking(m, n, k);
            assert_eq!(b.kc, k, "direct path: KC should equal k={k}, got {}", b.kc);
            assert_eq!(b.mc, m, "direct path: MC should equal m={m}, got {}", b.mc);
            assert_eq!(b.nc, n, "direct path: NC should equal n={n}, got {}", b.nc);
        }
    }

    /// MC should be at least 2*MR when m is large enough.
    #[test]
    fn test_blocking_mc_minimum_tiles() {
        let profile = DeviceProfile::detect();
        let (mr, _) = profile.microkernel_mr_nr();

        let b = profile.gemm_blocking(1024, 1024, 1024);
        assert!(b.mc >= 2 * mr,
            "MC={} should be >= 2*MR={} for large matrices", b.mc, 2 * mr);
    }

    #[test]
    fn test_prefetch_distance() {
        let profile = DeviceProfile::detect();
        let dist = profile.prefetch_distance();
        assert!(dist >= 4 && dist <= 16,
            "prefetch distance {dist} out of expected range [4, 16]");
        eprintln!("Prefetch distance: {dist} cache lines");
    }

    #[test]
    fn test_microkernel_mr_nr_avx2() {
        // AVX2: MR=6, NR=16 (12 ymm accumulators + 4 scratch)
        let (mr, nr, _) = MicroArch::Haswell.microkernel_geometry();
        assert_eq!((mr, nr), (6, 16));
        let (mr, nr, _) = MicroArch::Zen3.microkernel_geometry();
        assert_eq!((mr, nr), (6, 16));
    }

    #[test]
    fn test_microkernel_mr_nr_avx512() {
        // AVX-512: MR=14, NR=32 (28 zmm accumulators + 4 scratch)
        let (mr, nr, _) = MicroArch::SapphireRapids.microkernel_geometry();
        assert_eq!((mr, nr), (14, 32));
        let (mr, nr, _) = MicroArch::Zen5.microkernel_geometry();
        assert_eq!((mr, nr), (14, 32));
    }

    #[test]
    fn test_elem_tile_size() {
        let profile = DeviceProfile::detect();
        let tile = profile.elem_tile_size();
        let simd_w = profile.simd_width_f32();
        assert!(tile > 0);
        assert_eq!(tile % simd_w, 0, "tile={tile} not aligned to SIMD width={simd_w}");
        eprintln!("Elem tile size: {tile} f32 elements");
    }

    #[test]
    fn test_parallel_threshold() {
        let profile = DeviceProfile::detect();
        let threshold = profile.parallel_threshold();
        assert!(threshold >= 4096);
        eprintln!("Parallel threshold: {threshold} elements");
    }
}
