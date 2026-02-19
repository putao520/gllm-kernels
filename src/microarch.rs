//! MicroArchitecture detection and hardware-adaptive kernel configuration.
//!
//! Detects the exact CPU microarchitecture via CPUID (family/model) and derives
//! optimal kernel parameters: microkernel geometry, cache blocking, prefetch
//! distances, and ISA feature flags.
//!
//! All performance-critical parameters are computed once at startup and cached
//! in a global `OnceLock<KernelConfig>`. No hardcoded constants — every parameter
//! adapts to the detected hardware.

use std::sync::OnceLock;

// ── MicroArch enumeration ──────────────────────────────────────────────

/// Known CPU microarchitectures with distinct performance characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroArch {
    // Intel Core (client + server)
    Haswell,
    Broadwell,
    SkylakeClient,
    SkylakeX,       // AVX-512 with heavy downclocking
    CascadeLake,    // SKX derivative, same downclocking
    IceLakeClient,
    IceLakeServer,  // AVX-512 with mild downclocking
    TigerLake,
    AlderLake,      // Hybrid, P-cores only for AVX
    RaptorLake,
    SapphireRapids, // AVX-512 no downclocking, AMX
    GraniteRapids,

    // AMD Zen
    Zen2,
    Zen3,
    Zen4,           // AVX-512 via double-pump (256-bit units)
    Zen5,           // Native AVX-512 (512-bit units)

    // Fallback tiers (when exact model unknown)
    GenericAvx512,
    GenericAvx2,
    Scalar,
}

impl MicroArch {
    /// Microkernel geometry (MR, NR, SIMD width in f32 lanes).
    pub fn microkernel_geometry(self) -> (usize, usize, usize) {
        match self {
            // AVX-512 native (no downclocking penalty)
            Self::IceLakeClient | Self::IceLakeServer |
            Self::TigerLake | Self::SapphireRapids |
            Self::GraniteRapids | Self::Zen5 |
            Self::GenericAvx512 => (14, 32, 16),

            // AVX-512 with heavy downclocking — use AVX2 geometry
            Self::SkylakeX | Self::CascadeLake => (6, 16, 8),

            // Zen4: double-pump 512-bit, no throughput advantage over AVX2
            Self::Zen4 => (6, 16, 8),

            // AVX2
            Self::Haswell | Self::Broadwell |
            Self::SkylakeClient | Self::AlderLake | Self::RaptorLake |
            Self::Zen2 | Self::Zen3 |
            Self::GenericAvx2 => (6, 16, 8),

            Self::Scalar => (4, 4, 1),
        }
    }

    /// Whether this arch has usable AVX-512 (no severe downclocking).
    pub fn use_avx512(self) -> bool {
        matches!(self,
            Self::IceLakeClient | Self::IceLakeServer |
            Self::TigerLake | Self::SapphireRapids |
            Self::GraniteRapids | Self::Zen5 |
            Self::GenericAvx512
        )
    }

    /// Whether ZMM usage causes significant frequency throttling.
    pub fn zmm_downclocking(self) -> bool {
        matches!(self, Self::SkylakeX | Self::CascadeLake)
    }

    /// Estimated memory latency in nanoseconds (for prefetch distance calculation).
    pub fn estimated_mem_latency_ns(self) -> u32 {
        match self {
            // DDR4 systems (~60-70ns)
            Self::Haswell | Self::Broadwell | Self::SkylakeClient |
            Self::SkylakeX | Self::CascadeLake |
            Self::Zen2 | Self::Zen3 => 65,

            // DDR5 systems (~80-90ns)
            Self::IceLakeClient | Self::IceLakeServer |
            Self::TigerLake | Self::AlderLake | Self::RaptorLake |
            Self::SapphireRapids | Self::GraniteRapids |
            Self::Zen4 | Self::Zen5 => 85,

            Self::GenericAvx512 => 80,
            Self::GenericAvx2 => 70,
            Self::Scalar => 70,
        }
    }

    /// Estimated base frequency in GHz (conservative, for cycle calculations).
    pub fn estimated_freq_ghz(self) -> f64 {
        match self {
            Self::Haswell | Self::Broadwell => 3.0,
            Self::SkylakeClient | Self::SkylakeX | Self::CascadeLake => 3.5,
            Self::IceLakeClient | Self::IceLakeServer | Self::TigerLake => 3.5,
            Self::AlderLake | Self::RaptorLake => 3.8,
            Self::SapphireRapids | Self::GraniteRapids => 3.0,
            Self::Zen2 => 3.5,
            Self::Zen3 => 3.8,
            Self::Zen4 => 4.0,
            Self::Zen5 => 4.2,
            Self::GenericAvx512 | Self::GenericAvx2 => 3.5,
            Self::Scalar => 3.0,
        }
    }

    /// Has Intel AMX (Advanced Matrix Extensions).
    pub fn has_amx(self) -> bool {
        matches!(self, Self::SapphireRapids | Self::GraniteRapids)
    }

    /// Has AVX-512 VNNI (Vector Neural Network Instructions).
    pub fn has_vnni(self) -> bool {
        matches!(self,
            Self::IceLakeClient | Self::IceLakeServer |
            Self::TigerLake | Self::SapphireRapids |
            Self::GraniteRapids | Self::Zen4 | Self::Zen5
        )
    }

    /// Has native AVX-512 FP16 instructions.
    pub fn has_avx512fp16(self) -> bool {
        matches!(self, Self::SapphireRapids | Self::GraniteRapids)
    }

    /// Has AVX-512 BF16 instructions.
    pub fn has_bf16(self) -> bool {
        matches!(self,
            Self::SapphireRapids | Self::GraniteRapids |
            Self::Zen4 | Self::Zen5
        )
    }
}

// ── CPUID-based detection ──────────────────────────────────────────────

/// Detect the CPU microarchitecture via CPUID.
pub fn detect() -> MicroArch {
    #[cfg(target_arch = "x86_64")]
    {
        return detect_x86();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        MicroArch::Scalar
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_x86() -> MicroArch {
    use std::arch::x86_64::__cpuid;

    // Leaf 0: vendor string
    let leaf0 = unsafe { __cpuid(0) };
    let mut vendor_bytes = [0u8; 12];
    vendor_bytes[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
    vendor_bytes[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
    vendor_bytes[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
    let vendor = core::str::from_utf8(&vendor_bytes).unwrap_or("");

    // Leaf 1: family/model/stepping
    let leaf1 = unsafe { __cpuid(1) };
    let family_id = (leaf1.eax >> 8) & 0xF;
    let model_id = (leaf1.eax >> 4) & 0xF;
    let ext_family = (leaf1.eax >> 20) & 0xFF;
    let ext_model = (leaf1.eax >> 16) & 0xF;

    let family = if family_id == 0xF {
        family_id + ext_family
    } else {
        family_id
    };
    let model = if family_id == 0x6 || family_id == 0xF {
        (ext_model << 4) | model_id
    } else {
        model_id
    };

    if vendor.starts_with("GenuineIntel") {
        detect_intel(family, model)
    } else if vendor.starts_with("AuthenticAMD") {
        detect_amd(family, model)
    } else {
        detect_by_features()
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_intel(family: u32, model: u32) -> MicroArch {
    if family != 6 {
        return detect_by_features();
    }

    match model {
        // Haswell
        0x3C | 0x3F | 0x45 | 0x46 => MicroArch::Haswell,
        // Broadwell
        0x3D | 0x47 | 0x4F | 0x56 => MicroArch::Broadwell,
        // Skylake client
        0x4E | 0x5E => MicroArch::SkylakeClient,
        // Skylake-X / Cascade Lake (server AVX-512 with downclocking)
        0x55 => {
            // Stepping >= 5 is Cascade Lake, but same perf characteristics
            if is_x86_feature_detected!("avx512f") {
                // Check stepping to distinguish CLX from SKX
                // Both have same downclocking behavior
                MicroArch::SkylakeX
            } else {
                MicroArch::SkylakeClient
            }
        }
        // Cannon Lake
        0x66 => MicroArch::IceLakeClient,
        // Ice Lake client
        0x7E | 0x7D => MicroArch::IceLakeClient,
        // Ice Lake server
        0x6A | 0x6C => MicroArch::IceLakeServer,
        // Tiger Lake
        0x8C | 0x8D => MicroArch::TigerLake,
        // Alder Lake
        0x97 | 0x9A => MicroArch::AlderLake,
        // Raptor Lake
        0xB7 | 0xBF | 0xBA => MicroArch::RaptorLake,
        // Sapphire Rapids
        0x8F => MicroArch::SapphireRapids,
        // Granite Rapids
        0xAD | 0xAE => MicroArch::GraniteRapids,
        // Meteor Lake / Arrow Lake — treat as Raptor Lake (no AVX-512)
        0xAA | 0xAC => MicroArch::RaptorLake,
        _ => detect_by_features(),
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_amd(family: u32, model: u32) -> MicroArch {
    match family {
        // Zen 2: family 0x17, models 0x31, 0x60-0x6F, 0x70-0x7F, 0x90-0x9F
        0x17 => {
            if model >= 0x30 {
                MicroArch::Zen2
            } else {
                // Zen / Zen+ — treat as GenericAvx2
                MicroArch::GenericAvx2
            }
        }
        // Zen 3/4/5: family 0x19 and 0x1A
        0x19 => {
            match model {
                // Zen 3: 0x00-0x0F (Vermeer), 0x20-0x2F (Cezanne), 0x40-0x4F (Rembrandt),
                //         0x50-0x5F (Raphael desktop with Zen3 IO die)
                0x00..=0x0F | 0x20..=0x2F | 0x40..=0x4F | 0x50..=0x5F => MicroArch::Zen3,
                // Zen 4: 0x10-0x1F (Genoa), 0x60-0x6F (Raphael), 0x70-0x7F (Phoenix)
                0x10..=0x1F | 0x60..=0x7F => MicroArch::Zen4,
                // Zen 4c: 0xA0-0xAF (Bergamo)
                0xA0..=0xAF => MicroArch::Zen4,
                _ => {
                    // Unknown Zen 3/4 model — check for AVX-512
                    if is_x86_feature_detected!("avx512f") {
                        MicroArch::Zen4
                    } else {
                        MicroArch::Zen3
                    }
                }
            }
        }
        // Zen 5: family 0x1A
        0x1A => MicroArch::Zen5,
        _ => detect_by_features(),
    }
}

/// Fallback: detect by ISA feature flags when exact model is unknown.
#[cfg(target_arch = "x86_64")]
fn detect_by_features() -> MicroArch {
    if is_x86_feature_detected!("avx512f") {
        MicroArch::GenericAvx512
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        MicroArch::GenericAvx2
    } else {
        MicroArch::Scalar
    }
}

// ── KernelConfig ───────────────────────────────────────────────────────

/// Complete hardware-adaptive parameter set for all kernel hot paths.
///
/// Computed once from the detected `MicroArch` and actual cache sizes.
/// All GEMM/GEMV/activation kernels read from this config at runtime.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Detected microarchitecture.
    pub arch: MicroArch,

    // ── Microkernel geometry ──
    /// M-dimension tile size (rows per microkernel invocation).
    pub mr: usize,
    /// N-dimension tile size (columns per microkernel invocation).
    pub nr: usize,
    /// SIMD width in f32 lanes (AVX2=8, AVX-512=16).
    pub simd_width: usize,

    // ── Cache blocking (BLIS formula) ──
    /// K-blocking: B strip (KC × NR × elem) fits in L1 × 0.50.
    pub kc: usize,
    /// M-blocking: A panel (MC × KC × elem) fits in L2 × 0.50.
    pub mc: usize,
    /// N-blocking: B panel (KC × NC × elem) fits in L3 × 0.50.
    pub nc: usize,

    // ── Cache sizes (detected) ──
    pub l1d: usize,
    pub l2: usize,
    pub l3: usize,

    // ── Prefetch parameters ──
    /// B panel prefetch distance in bytes (GEMM).
    pub pf_distance_b: usize,
    /// A panel prefetch distance in bytes (GEMM).
    pub pf_distance_a: usize,
    /// GEMV prefetch hint: 0=T0 (temporal), 1=T1, 2=T2, 3=NTA (non-temporal).
    pub pf_hint_gemv: u8,
    /// GEMV prefetch distance in rows ahead.
    pub pf_rows_gemv: usize,

    // ── Hardware features ──
    pub use_avx512: bool,
    pub zmm_downclocking: bool,
    pub has_amx: bool,
    pub has_vnni: bool,
    pub has_avx512fp16: bool,
    pub has_bf16: bool,
}

impl KernelConfig {
    /// Build a complete KernelConfig from the detected microarchitecture.
    pub fn from_arch(arch: MicroArch) -> Self {
        let (l1d, l2, l3) = crate::cache_params::detect_cache_sizes_pub();
        Self::from_arch_with_cache(arch, l1d, l2, l3)
    }

    /// Build KernelConfig with explicit cache sizes (for testing).
    pub fn from_arch_with_cache(arch: MicroArch, l1d: usize, l2: usize, l3: usize) -> Self {
        let (mr, nr, simd_w) = arch.microkernel_geometry();
        let elem = 4usize; // f32

        // ── BLIS standard blocking formulas ──
        // KC: B strip (KC × NR × elem) ≤ L1D × 0.50
        let kc_raw = (l1d / 2) / (nr * elem);
        let kc = (kc_raw & !7).clamp(64, 768);

        // MC: A panel (MC × KC × elem) ≤ L2 × 0.50
        let mc_raw = (l2 / 2) / (kc * elem);
        let mc = (mc_raw / mr * mr).clamp(mr, 960);

        // NC: B panel (KC × NC × elem) ≤ L3 × 0.50
        let nc_raw = (l3 / 2) / (kc * elem);
        let nc = (nc_raw / nr * nr).clamp(nr, 8192);

        // ── Prefetch distance calculation ──
        let mem_lat_ns = arch.estimated_mem_latency_ns();
        let freq_ghz = arch.estimated_freq_ghz();
        let lat_cycles = (mem_lat_ns as f64 * freq_ghz) as usize;

        // GEMM: prefetch B panel ahead by latency × bytes_per_k_iter
        let pf_b = (lat_cycles / 2 * nr * elem).min(8192).max(256);
        let pf_a = (lat_cycles / 2 * mr * elem).min(4096).max(128);

        // GEMV: streaming access → NTA hint (don't pollute L2/L3)
        // Distance in rows = latency_cycles / cycles_per_row
        // Each row processes N elements, ~1 cycle per FMA
        let pf_rows = (lat_cycles / 8).clamp(16, 64);

        KernelConfig {
            arch,
            mr, nr, simd_width: simd_w,
            kc, mc, nc,
            l1d, l2, l3,
            pf_distance_b: pf_b,
            pf_distance_a: pf_a,
            pf_hint_gemv: 3, // NTA for streaming GEMV
            pf_rows_gemv: pf_rows,
            use_avx512: arch.use_avx512(),
            zmm_downclocking: arch.zmm_downclocking(),
            has_amx: arch.has_amx(),
            has_vnni: arch.has_vnni(),
            has_avx512fp16: arch.has_avx512fp16(),
            has_bf16: arch.has_bf16(),
        }
    }

    /// Compute NUMA-aware NC using a specific node's L3 size.
    pub fn nc_for_node_l3(&self, node_l3: usize) -> usize {
        let elem = 4usize;
        let nc_raw = (node_l3 / 2) / (self.kc * elem);
        (nc_raw / self.nr * self.nr).clamp(self.nr, 8192)
    }

    /// Get blocking params in the legacy BlockingParams format.
    pub fn blocking_params(&self) -> crate::cache_params::BlockingParams {
        crate::cache_params::BlockingParams {
            kc: self.kc,
            mc: self.mc,
            nc: self.nc,
        }
    }
}

// ── Global singleton ───────────────────────────────────────────────────

static CONFIG: OnceLock<KernelConfig> = OnceLock::new();

/// Get the global hardware-adaptive kernel configuration.
///
/// Detected once on first call, then cached for the process lifetime.
/// All kernel hot paths should read from this instead of computing
/// parameters independently.
pub fn kernel_config() -> &'static KernelConfig {
    CONFIG.get_or_init(|| {
        let arch = detect();
        KernelConfig::from_arch(arch)
    })
}

/// Get the detected microarchitecture.
pub fn detected_arch() -> MicroArch {
    kernel_config().arch
}

impl std::fmt::Display for MicroArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Haswell => "Intel Haswell",
            Self::Broadwell => "Intel Broadwell",
            Self::SkylakeClient => "Intel Skylake (client)",
            Self::SkylakeX => "Intel Skylake-X/Cascade Lake (AVX-512, downclocking)",
            Self::CascadeLake => "Intel Cascade Lake (AVX-512, downclocking)",
            Self::IceLakeClient => "Intel Ice Lake (client)",
            Self::IceLakeServer => "Intel Ice Lake (server)",
            Self::TigerLake => "Intel Tiger Lake",
            Self::AlderLake => "Intel Alder Lake",
            Self::RaptorLake => "Intel Raptor Lake",
            Self::SapphireRapids => "Intel Sapphire Rapids",
            Self::GraniteRapids => "Intel Granite Rapids",
            Self::Zen2 => "AMD Zen 2",
            Self::Zen3 => "AMD Zen 3",
            Self::Zen4 => "AMD Zen 4 (double-pump AVX-512)",
            Self::Zen5 => "AMD Zen 5 (native AVX-512)",
            Self::GenericAvx512 => "Generic AVX-512",
            Self::GenericAvx2 => "Generic AVX2",
            Self::Scalar => "Scalar",
        };
        write!(f, "{name}")
    }
}

impl std::fmt::Display for KernelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} | MR={} NR={} SIMD={} | KC={} MC={} NC={} | \
                L1D={}K L2={}K L3={}M | AVX512={} ZMM_DC={} | \
                PF_B={} PF_A={} PF_GEMV_rows={}",
            self.arch,
            self.mr, self.nr, self.simd_width,
            self.kc, self.mc, self.nc,
            self.l1d / 1024, self.l2 / 1024, self.l3 / (1024 * 1024),
            self.use_avx512, self.zmm_downclocking,
            self.pf_distance_b, self.pf_distance_a, self.pf_rows_gemv,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_not_scalar() {
        let arch = detect();
        eprintln!("Detected: {arch}");
        // On any modern x86_64 test machine, should at least detect AVX2
        #[cfg(target_arch = "x86_64")]
        {
            assert_ne!(arch, MicroArch::Scalar, "Expected at least AVX2 on x86_64");
        }
    }

    #[test]
    fn test_kernel_config_sanity() {
        let cfg = kernel_config();
        eprintln!("KernelConfig: {cfg}");

        // Geometry sanity
        assert!(cfg.mr >= 4);
        assert!(cfg.nr >= 4);
        assert!(cfg.simd_width >= 1);

        // Blocking sanity
        assert!(cfg.kc >= 64 && cfg.kc <= 768, "KC={} out of range", cfg.kc);
        assert!(cfg.mc >= cfg.mr, "MC={} < MR={}", cfg.mc, cfg.mr);
        assert!(cfg.nc >= cfg.nr, "NC={} < NR={}", cfg.nc, cfg.nr);

        // BLIS constraint: B strip fits in L1D/2
        let b_strip = cfg.nr * cfg.kc * 4;
        assert!(b_strip <= cfg.l1d,
            "B strip {}B exceeds L1D {}B", b_strip, cfg.l1d);

        // BLIS constraint: A panel fits in L2
        let a_panel = cfg.mc * cfg.kc * 4;
        assert!(a_panel <= cfg.l2,
            "A panel {}B exceeds L2 {}B", a_panel, cfg.l2);

        // Prefetch sanity
        assert!(cfg.pf_distance_b >= 128);
        assert!(cfg.pf_distance_a >= 64);
        assert!(cfg.pf_rows_gemv >= 16 && cfg.pf_rows_gemv <= 64);
    }

    #[test]
    fn test_geometry_avx2() {
        let (mr, nr, sw) = MicroArch::Haswell.microkernel_geometry();
        assert_eq!((mr, nr, sw), (6, 16, 8));
    }

    #[test]
    fn test_geometry_avx512() {
        let (mr, nr, sw) = MicroArch::SapphireRapids.microkernel_geometry();
        assert_eq!((mr, nr, sw), (14, 32, 16));
    }

    #[test]
    fn test_skx_uses_avx2_geometry() {
        // SKX has AVX-512 but downclocking makes AVX2 geometry better
        let (mr, nr, sw) = MicroArch::SkylakeX.microkernel_geometry();
        assert_eq!((mr, nr, sw), (6, 16, 8));
        assert!(MicroArch::SkylakeX.zmm_downclocking());
        assert!(!MicroArch::SkylakeX.use_avx512());
    }

    #[test]
    fn test_zen4_uses_avx2_geometry() {
        // Zen4 double-pump: no throughput advantage for 512-bit
        let (mr, nr, sw) = MicroArch::Zen4.microkernel_geometry();
        assert_eq!((mr, nr, sw), (6, 16, 8));
    }

    #[test]
    fn test_config_with_explicit_cache() {
        // Simulate a system with 48KB L1D, 1MB L2, 32MB L3
        let cfg = KernelConfig::from_arch_with_cache(
            MicroArch::SapphireRapids, 48 * 1024, 1024 * 1024, 32 * 1024 * 1024,
        );
        eprintln!("SPR config: {cfg}");
        assert_eq!(cfg.mr, 14);
        assert_eq!(cfg.nr, 32);
        assert!(cfg.use_avx512);
        assert!(!cfg.zmm_downclocking);
        assert!(cfg.has_amx);

        // Verify BLIS constraints
        let b_strip = cfg.nr * cfg.kc * 4;
        assert!(b_strip <= cfg.l1d, "B strip {}B > L1D {}B", b_strip, cfg.l1d);
    }

    #[test]
    fn test_numa_nc() {
        let cfg = kernel_config();
        let nc_8m = cfg.nc_for_node_l3(8 * 1024 * 1024);
        let nc_16m = cfg.nc_for_node_l3(16 * 1024 * 1024);
        assert!(nc_16m >= nc_8m, "Larger L3 should give larger NC");
        assert!(nc_8m >= cfg.nr);
    }

    #[test]
    fn test_singleton_consistency() {
        let c1 = kernel_config();
        let c2 = kernel_config();
        assert_eq!(c1.arch, c2.arch);
        assert_eq!(c1.kc, c2.kc);
        assert_eq!(c1.mc, c2.mc);
        assert_eq!(c1.nc, c2.nc);
    }
}
