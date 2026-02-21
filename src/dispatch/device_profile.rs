//! Unified hardware device profile.
//!
//! Integrates microarchitecture detection, cache hierarchy, NUMA topology,
//! and peak performance estimates into a single `DeviceProfile` used by the
//! inference compiler for code generation and tuning decisions.

use crate::microarch::{MicroArch, KernelConfig};
use crate::numa::NumaTopology;
use crate::autotuning::HwInfo;

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
}
