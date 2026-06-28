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
    CometLake,      // 10th gen, 14nm, AVX2 only, DDR4
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
            Self::SkylakeClient | Self::CometLake | Self::AlderLake | Self::RaptorLake |
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
            Self::CometLake | Self::SkylakeX | Self::CascadeLake |
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
            Self::CometLake => 3.7,
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

    /// Has F16C (vcvtph2ps / vcvtps2ph — F16↔F32 conversion).
    ///
    /// F16C 是 Ivy Bridge+ 的基线 x86 指令 (2012), 与 AVX-512 FP16 (has_avx512fp16)
    /// 不同: F16C 只做转换, 不做 F16 计算; AVX-512 FP16 做 F16 native 计算。
    /// REQ-HW-TIER-001: 细粒度 flag, 区分"能转换 F16"和"能计算 F16"。
    // @trace REQ-HW-TIER-001 [req:MicroArch-has_f16c] F16C 转换指令细粒度 flag
    pub fn has_f16c(self) -> bool {
        // F16C 自 Ivy Bridge (client) / Haswell (server) 后成为 x86 基线。
        // Scalar 兜底无 SIMD → false。其余所有 AVX2+ 微架构都有 F16C。
        !matches!(self, Self::Scalar)
    }

    /// Number of FMA execution ports (pipes) per core.
    ///
    /// Most modern x86 cores have 2 FMA ports. Zen5 is the notable exception
    /// with 3 FMA pipes. Scalar fallback has 1.
    pub fn fma_ports(self) -> u32 {
        match self {
            Self::Zen5 => 3,
            Self::Scalar => 1,
            _ => 2,
        }
    }

    // ── AMX+ / AVX10 / APX 检测 (Granite Rapids / Diamond Rapids) ──

    /// AMX-FP16: TDPFP16PS (Granite Rapids+).
    pub fn has_amx_fp16(self) -> bool {
        matches!(self, Self::GraniteRapids)
    }

    /// AMX-COMPLEX: TCMMIMFP16PS / TCMMRLFP16PS (Diamond Rapids).
    /// 当前无 MicroArch 变体对应 Diamond Rapids，预留 false。
    pub fn has_amx_complex(self) -> bool {
        false // Diamond Rapids MicroArch variant TBD
    }

    /// AMX-TRANSPOSE: T2RPNTLVWZ series (Diamond Rapids).
    pub fn has_amx_transpose(self) -> bool {
        false // Diamond Rapids MicroArch variant TBD
    }

    /// AMX-FP8: FP8 tile dot product (Diamond Rapids).
    pub fn has_amx_fp8(self) -> bool {
        false // Diamond Rapids MicroArch variant TBD
    }

    /// AVX10.2: 256-bit 统一 SIMD (Arrow Lake+).
    pub fn has_avx10_2(self) -> bool {
        false // Arrow Lake+ MicroArch variant TBD
    }

    /// APX: Advanced Performance Extensions — 31 GPR (r16-r30).
    pub fn has_apx(self) -> bool {
        false // APX detection via CPUID leaf 7 subleaf 1 TBD
    }

    /// VP2INTERSECT: hardware sparse mask intersection.
    pub fn has_sparse_mask_intersect(self) -> bool {
        false // Tiger Lake+ but rarely enabled; CPUID leaf 7 bit TBD
    }
}

// ── CPUID-based detection ──────────────────────────────────────────────

/// Detect the CPU microarchitecture via CPUID.
pub fn detect() -> MicroArch {
    #[cfg(target_arch = "x86_64")]
    {
        detect_x86()
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
    let leaf0 = __cpuid(0);
    let mut vendor_bytes = [0u8; 12];
    vendor_bytes[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
    vendor_bytes[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
    vendor_bytes[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
    let vendor = core::str::from_utf8(&vendor_bytes).unwrap_or("");

    // Leaf 1: family/model/stepping
    let leaf1 = __cpuid(1);
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
        // Comet Lake (10th gen, 14nm, AVX2 only)
        0xA5 | 0xA6 => MicroArch::CometLake,
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
    pub has_f16c: bool,
    /// ARM SVE (Scalable Vector Extension) available.
    pub has_sve: bool,
    /// ARM SVE2 available.
    pub has_sve2: bool,
    /// ARM SVE vector length in bytes (0 if SVE not available).
    pub sve_vl_bytes: usize,

    // ── AMX+ (Granite Rapids / Diamond Rapids) ──
    /// AMX-FP16: TDPFP16PS instruction (Granite Rapids+).
    pub has_amx_fp16: bool,
    /// AMX-COMPLEX: TCMMIMFP16PS / TCMMRLFP16PS (Diamond Rapids).
    pub has_amx_complex: bool,
    /// AMX-TRANSPOSE: T2RPNTLVWZ series (Diamond Rapids).
    pub has_amx_transpose: bool,
    /// AMX-FP8: FP8 tile dot product (Diamond Rapids).
    pub has_amx_fp8: bool,

    // ── AVX10 / APX ──
    /// AVX10.2: 256-bit unified SIMD (Arrow Lake+).
    pub has_avx10_2: bool,
    /// APX: Advanced Performance Extensions — 31 GPR (r16-r30).
    pub has_apx: bool,
    /// VP2INTERSECT: hardware sparse mask intersection.
    pub has_sparse_mask_intersect: bool,
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

        // ── Optimized BLIS blocking formulas ──
        // KC: B strip (KC × NR × elem) ≤ L1D × 0.875
        // A is streaming (not resident in L1), so B strip can use 7/8 of L1D.
        // Larger KC reduces KC-iteration count, cutting C matrix load/store traffic.
        // Example: L1D=32KB, NR=16 → KC=448 (was 384), B strip=28KB (fits in 32KB L1D).
        let kc_raw = (l1d * 7 / 8) / (nr * elem);
        let kc = (kc_raw & !7).clamp(64, 768);

        // MC: A panel (MC × KC × elem) ≤ L2 × 0.50
        let mc_raw = (l2 / 2) / (kc * elem);
        let mc = (mc_raw / mr * mr).clamp(mr, 960);

        // NC: B panel shared across cores in L3
        // L3 is shared across all cores on single-socket (UMA) systems.
        // Only divide by NUMA nodes for multi-socket systems.
        let num_numa_nodes = crate::numa::num_nodes().max(1);
        let effective_l3 = l3 / num_numa_nodes;
        let nc_raw = (effective_l3 / 2) / (kc * elem);
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

        #[allow(unused_mut)]
        let mut cfg = KernelConfig {
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
            has_f16c: arch.has_f16c(),
            has_sve: false,
            has_sve2: false,
            sve_vl_bytes: 0,
            // AMX+ (detected via CPUID at runtime)
            has_amx_fp16: arch.has_amx_fp16(),
            has_amx_complex: arch.has_amx_complex(),
            has_amx_transpose: arch.has_amx_transpose(),
            has_amx_fp8: arch.has_amx_fp8(),
            // AVX10 / APX
            has_avx10_2: arch.has_avx10_2(),
            has_apx: arch.has_apx(),
            has_sparse_mask_intersect: arch.has_sparse_mask_intersect(),
        };

        // Detect ARM SVE from HwInfo if on aarch64.
        #[cfg(target_arch = "aarch64")]
        {
            let hw = crate::autotuning::HwInfo::detect();
            cfg.has_sve = hw.isa.sve;
            cfg.has_sve2 = hw.isa.sve2;
            cfg.sve_vl_bytes = hw.isa.sve_vl_bytes;
        }

        cfg
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
            Self::CometLake => "Intel Comet Lake",
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

        // BLIS constraint: B strip fits in L1D (≤ 7/8 of L1D)
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

    #[test]
    fn prefetch_varies_by_microarch() {
        let hsw = KernelConfig::from_arch(MicroArch::Haswell);
        let spr = KernelConfig::from_arch(MicroArch::SapphireRapids);
        // Different microarchitectures should produce different prefetch distances
        // due to different memory latency and frequency estimates.
        assert_ne!(hsw.pf_distance_b, spr.pf_distance_b,
            "Haswell and SPR should have different prefetch distances");
    }

    #[test]
    fn spr_has_bf16() {
        let spr = KernelConfig::from_arch(MicroArch::SapphireRapids);
        assert!(spr.has_bf16, "SPR should have BF16 support");
        assert!(spr.has_vnni, "SPR should have VNNI support");
    }

    // ── New tests ──────────────────────────────────────────────────────

    #[test]
    fn scalar_geometry_is_minimal() {
        // Scalar fallback uses the smallest possible tile.
        let (mr, nr, sw) = MicroArch::Scalar.microkernel_geometry();
        assert_eq!((mr, nr, sw), (4, 4, 1));
    }

    #[test]
    fn fma_ports_per_arch() {
        // Zen5 has 3 FMA pipes, Scalar has 1, everything else has 2.
        assert_eq!(MicroArch::Zen5.fma_ports(), 3);
        assert_eq!(MicroArch::Scalar.fma_ports(), 1);
        assert_eq!(MicroArch::Haswell.fma_ports(), 2);
        assert_eq!(MicroArch::SapphireRapids.fma_ports(), 2);
        assert_eq!(MicroArch::Zen3.fma_ports(), 2);
        assert_eq!(MicroArch::GenericAvx512.fma_ports(), 2);
    }

    #[test]
    fn has_amx_only_spr_and_granite() {
        // AMX is exclusive to Sapphire Rapids and Granite Rapids.
        assert!(MicroArch::SapphireRapids.has_amx());
        assert!(MicroArch::GraniteRapids.has_amx());
        // Everything else: no AMX.
        for arch in [
            MicroArch::Haswell, MicroArch::IceLakeServer,
            MicroArch::Zen5, MicroArch::GenericAvx512,
        ] {
            assert!(!arch.has_amx(), "{arch:?} should not have AMX");
        }
    }

    #[test]
    fn has_vnni_coverage() {
        // VNNI present on Ice Lake+, Tiger Lake, SPR, Granite Rapids, Zen4, Zen5.
        let with_vnni = [
            MicroArch::IceLakeClient, MicroArch::IceLakeServer,
            MicroArch::TigerLake, MicroArch::SapphireRapids,
            MicroArch::GraniteRapids, MicroArch::Zen4, MicroArch::Zen5,
        ];
        for arch in with_vnni {
            assert!(arch.has_vnni(), "{arch:?} should have VNNI");
        }
        // Pre-Ice-Lake Intel and early AMD lack VNNI.
        let without_vnni = [
            MicroArch::Haswell, MicroArch::SkylakeClient,
            MicroArch::SkylakeX, MicroArch::Zen2, MicroArch::Zen3,
        ];
        for arch in without_vnni {
            assert!(!arch.has_vnni(), "{arch:?} should not have VNNI");
        }
    }

    #[test]
    fn has_avx512fp16_limited_to_spr_granite() {
        assert!(MicroArch::SapphireRapids.has_avx512fp16());
        assert!(MicroArch::GraniteRapids.has_avx512fp16());
        // Ice Lake has AVX-512 but not FP16 instructions.
        assert!(!MicroArch::IceLakeServer.has_avx512fp16());
        assert!(!MicroArch::Zen5.has_avx512fp16());
    }

    #[test]
    fn has_bf16_includes_amd_zen4_zen5() {
        // BF16: SPR, Granite Rapids, Zen4, Zen5.
        assert!(MicroArch::SapphireRapids.has_bf16());
        assert!(MicroArch::GraniteRapids.has_bf16());
        assert!(MicroArch::Zen4.has_bf16());
        assert!(MicroArch::Zen5.has_bf16());
        // Zen3 and Ice Lake server do not have BF16.
        assert!(!MicroArch::Zen3.has_bf16());
        assert!(!MicroArch::IceLakeServer.has_bf16());
    }

    #[test]
    fn mem_latency_ddr4_vs_ddr5() {
        // DDR4-era architectures: 65 ns.
        let ddr4 = [
            MicroArch::Haswell, MicroArch::SkylakeX, MicroArch::Zen2,
        ];
        for arch in ddr4 {
            assert_eq!(arch.estimated_mem_latency_ns(), 65,
                "{arch:?} should be 65 ns DDR4");
        }
        // DDR5-era architectures: 85 ns.
        let ddr5 = [
            MicroArch::AlderLake, MicroArch::SapphireRapids,
            MicroArch::Zen4, MicroArch::Zen5,
        ];
        for arch in ddr5 {
            assert_eq!(arch.estimated_mem_latency_ns(), 85,
                "{arch:?} should be 85 ns DDR5");
        }
    }

    #[test]
    fn amx_future_extensions_always_false() {
        // AMX-COMPLEX, AMX-TRANSPOSE, AMX-FP8 are reserved for Diamond Rapids (TBD).
        for arch in [
            MicroArch::SapphireRapids, MicroArch::GraniteRapids,
            MicroArch::Zen5, MicroArch::Scalar,
        ] {
            assert!(!arch.has_amx_complex(), "{arch:?} AMX-COMPLEX should be false");
            assert!(!arch.has_amx_transpose(), "{arch:?} AMX-TRANSPOSE should be false");
            assert!(!arch.has_amx_fp8(), "{arch:?} AMX-FP8 should be false");
        }
    }

    #[test]
    fn avx10_apx_sparse_mask_intersect_always_false() {
        // These features are not yet tied to any MicroArch variant.
        for arch in [
            MicroArch::GraniteRapids, MicroArch::SapphireRapids,
            MicroArch::Zen5, MicroArch::GenericAvx512,
        ] {
            assert!(!arch.has_avx10_2(), "{arch:?} AVX10.2 should be false");
            assert!(!arch.has_apx(), "{arch:?} APX should be false");
            assert!(!arch.has_sparse_mask_intersect(), "{arch:?} VP2INTERSECT should be false");
        }
    }

    #[test]
    fn display_microarch_has_expected_labels() {
        // Verify Display output contains recognizable substrings.
        assert!(MicroArch::Haswell.to_string().contains("Haswell"));
        assert!(MicroArch::Zen4.to_string().contains("Zen 4"));
        assert!(MicroArch::SapphireRapids.to_string().contains("Sapphire"));
        assert!(MicroArch::Scalar.to_string().contains("Scalar"));
        assert!(MicroArch::SkylakeX.to_string().contains("downclocking"));
    }

    #[test]
    fn display_kernel_config_includes_key_fields() {
        let cfg = KernelConfig::from_arch_with_cache(
            MicroArch::Haswell, 32 * 1024, 256 * 1024, 8 * 1024 * 1024,
        );
        let s = cfg.to_string();
        // Must contain MR, NR, KC, MC, NC, L1D, L2, L3 values.
        assert!(s.contains("MR="), "Display should contain MR=");
        assert!(s.contains("NR="), "Display should contain NR=");
        assert!(s.contains("KC="), "Display should contain KC=");
        assert!(s.contains("AVX512="), "Display should contain AVX512=");
        assert!(s.contains("Haswell"), "Display should contain arch name");
    }

    #[test]
    fn blocking_params_roundtrip() {
        let cfg = KernelConfig::from_arch_with_cache(
            MicroArch::SapphireRapids, 48 * 1024, 1024 * 1024, 32 * 1024 * 1024,
        );
        let bp = cfg.blocking_params();
        assert_eq!(bp.kc, cfg.kc);
        assert_eq!(bp.mc, cfg.mc);
        assert_eq!(bp.nc, cfg.nc);
    }

    #[test]
    fn nc_for_node_l3_scales_and_aligns() {
        let cfg = KernelConfig::from_arch_with_cache(
            MicroArch::GenericAvx2, 32 * 1024, 256 * 1024, 16 * 1024 * 1024,
        );
        let nc_small = cfg.nc_for_node_l3(4 * 1024 * 1024);
        let nc_large = cfg.nc_for_node_l3(64 * 1024 * 1024);
        // Larger L3 must yield NC >= smaller L3 NC.
        assert!(nc_large >= nc_small, "NC should grow with L3");
        // NC must be a multiple of NR (alignment guarantee).
        assert_eq!(nc_small % cfg.nr, 0, "NC should be aligned to NR");
        assert_eq!(nc_large % cfg.nr, 0, "NC should be aligned to NR");
        // NC must be clamped to [nr, 8192].
        assert!(nc_small >= cfg.nr);
        assert!(nc_large <= 8192);
    }

    #[test]
    fn detected_arch_matches_global_config() {
        // detected_arch() must return the same arch as kernel_config().
        assert_eq!(detected_arch(), kernel_config().arch);
    }

    // ── Additional tests for uncovered logic paths ──

    #[test]
    fn estimated_freq_ghz_all_positive() {
        // Every MicroArch variant must report a positive frequency.
        let arches = [
            MicroArch::Haswell, MicroArch::Broadwell,
            MicroArch::SkylakeClient, MicroArch::CometLake,
            MicroArch::SkylakeX, MicroArch::CascadeLake,
            MicroArch::IceLakeClient, MicroArch::IceLakeServer,
            MicroArch::TigerLake, MicroArch::AlderLake, MicroArch::RaptorLake,
            MicroArch::SapphireRapids, MicroArch::GraniteRapids,
            MicroArch::Zen2, MicroArch::Zen3, MicroArch::Zen4, MicroArch::Zen5,
            MicroArch::GenericAvx512, MicroArch::GenericAvx2, MicroArch::Scalar,
        ];
        for arch in arches {
            let freq = arch.estimated_freq_ghz();
            assert!(freq > 0.0, "{arch:?} frequency must be positive, got {freq}");
        }
    }

    #[test]
    fn zen_freq_monotonically_increasing() {
        // AMD Zen generations should have non-decreasing frequency estimates.
        let zen2 = MicroArch::Zen2.estimated_freq_ghz();
        let zen3 = MicroArch::Zen3.estimated_freq_ghz();
        let zen4 = MicroArch::Zen4.estimated_freq_ghz();
        let zen5 = MicroArch::Zen5.estimated_freq_ghz();
        assert!(zen3 >= zen2, "Zen3 freq {zen3} should >= Zen2 freq {zen2}");
        assert!(zen4 >= zen3, "Zen4 freq {zen4} should >= Zen3 freq {zen3}");
        assert!(zen5 >= zen4, "Zen5 freq {zen5} should >= Zen4 freq {zen4}");
    }

    #[test]
    fn intel_client_freq_increases_with_generation() {
        // Intel client generations: Haswell → CometLake should be non-decreasing.
        let hsw = MicroArch::Haswell.estimated_freq_ghz();
        let skl = MicroArch::SkylakeClient.estimated_freq_ghz();
        let cml = MicroArch::CometLake.estimated_freq_ghz();
        let adl = MicroArch::AlderLake.estimated_freq_ghz();
        assert!(skl > hsw, "Skylake freq should exceed Haswell");
        assert!(cml > skl, "CometLake freq should exceed Skylake");
        assert!(adl > cml, "AlderLake freq should exceed CometLake");
    }

    #[test]
    fn use_avx512_false_for_avx2_only_arches() {
        // AVX2-only architectures must not claim AVX-512 usability.
        let avx2_only = [
            MicroArch::Haswell, MicroArch::Broadwell,
            MicroArch::SkylakeClient, MicroArch::CometLake,
            MicroArch::AlderLake, MicroArch::RaptorLake,
            MicroArch::Zen2, MicroArch::Zen3,
            MicroArch::GenericAvx2,
        ];
        for arch in avx2_only {
            assert!(!arch.use_avx512(), "{arch:?} should not claim usable AVX-512");
        }
    }

    #[test]
    fn use_avx512_true_for_native_avx512_arches() {
        // Architectures with native AVX-512 (no downclocking) must claim it.
        let avx512_native = [
            MicroArch::IceLakeClient, MicroArch::IceLakeServer,
            MicroArch::TigerLake, MicroArch::SapphireRapids,
            MicroArch::GraniteRapids, MicroArch::Zen5,
            MicroArch::GenericAvx512,
        ];
        for arch in avx512_native {
            assert!(arch.use_avx512(), "{arch:?} should claim usable AVX-512");
        }
    }

    #[test]
    fn zmm_downclocking_only_skx_cascadelake() {
        // Only Skylake-X and CascadeLake have severe ZMM downclocking.
        assert!(MicroArch::SkylakeX.zmm_downclocking());
        assert!(MicroArch::CascadeLake.zmm_downclocking());
        // All other arches: no downclocking.
        let no_dc = [
            MicroArch::IceLakeServer, MicroArch::SapphireRapids,
            MicroArch::Zen5, MicroArch::Haswell, MicroArch::Zen4,
            MicroArch::GenericAvx512, MicroArch::Scalar,
        ];
        for arch in no_dc {
            assert!(!arch.zmm_downclocking(), "{arch:?} should not have ZMM downclocking");
        }
    }

    #[test]
    fn has_amx_fp16_only_granite_rapids() {
        // AMX-FP16 is exclusive to Granite Rapids.
        assert!(MicroArch::GraniteRapids.has_amx_fp16());
        assert!(!MicroArch::SapphireRapids.has_amx_fp16());
        assert!(!MicroArch::Zen5.has_amx_fp16());
        assert!(!MicroArch::IceLakeServer.has_amx_fp16());
    }

    #[test]
    fn kernel_config_from_arch_scalar_minimum_values() {
        // Scalar arch: geometry (4,4,1), all fields should reflect minimal hardware.
        let cfg = KernelConfig::from_arch_with_cache(
            MicroArch::Scalar, 16 * 1024, 128 * 1024, 4 * 1024 * 1024,
        );
        assert_eq!(cfg.mr, 4);
        assert_eq!(cfg.nr, 4);
        assert_eq!(cfg.simd_width, 1);
        assert!(!cfg.use_avx512);
        assert!(!cfg.has_amx);
        assert!(!cfg.has_bf16);
        assert!(!cfg.has_vnni);
        assert_eq!(cfg.arch.fma_ports(), 1); // Scalar arch has 1 FMA port
    }

    #[test]
    fn kernel_config_from_arch_zen4_reflects_no_avx512_flag() {
        // Zen4 double-pump: KernelConfig.use_avx512 must be false.
        let cfg = KernelConfig::from_arch_with_cache(
            MicroArch::Zen4, 32 * 1024, 512 * 1024, 16 * 1024 * 1024,
        );
        assert_eq!(cfg.mr, 6);
        assert_eq!(cfg.nr, 16);
        assert_eq!(cfg.simd_width, 8);
        assert!(!cfg.use_avx512, "Zen4 should not enable AVX-512 usage");
        assert!(!cfg.zmm_downclocking);
        assert!(cfg.has_bf16, "Zen4 should have BF16");
        assert!(cfg.has_vnni, "Zen4 should have VNNI");
    }

    #[test]
    fn display_all_microarch_variants_no_panic() {
        // Ensure Display is implemented for all variants without panic.
        let arches = [
            MicroArch::Haswell, MicroArch::Broadwell,
            MicroArch::SkylakeClient, MicroArch::CometLake,
            MicroArch::SkylakeX, MicroArch::CascadeLake,
            MicroArch::IceLakeClient, MicroArch::IceLakeServer,
            MicroArch::TigerLake, MicroArch::AlderLake, MicroArch::RaptorLake,
            MicroArch::SapphireRapids, MicroArch::GraniteRapids,
            MicroArch::Zen2, MicroArch::Zen3, MicroArch::Zen4, MicroArch::Zen5,
            MicroArch::GenericAvx512, MicroArch::GenericAvx2, MicroArch::Scalar,
        ];
        for arch in arches {
            let s = arch.to_string();
            assert!(!s.is_empty(), "{arch:?} Display produced empty string");
        }
    }

    #[test]
    fn display_amd_arches_contain_zen() {
        // AMD arches should display with "Zen" in the name.
        assert!(MicroArch::Zen2.to_string().contains("Zen 2"));
        assert!(MicroArch::Zen3.to_string().contains("Zen 3"));
        assert!(MicroArch::Zen4.to_string().contains("Zen 4"));
        assert!(MicroArch::Zen5.to_string().contains("Zen 5"));
    }

    #[test]
    fn estimated_mem_latency_consistent_within_memory_gen() {
        // All DDR4-era arches must have the same latency; all DDR5-era must match.
        // DDR4 Intel
        assert_eq!(MicroArch::Broadwell.estimated_mem_latency_ns(),
                   MicroArch::Haswell.estimated_mem_latency_ns());
        // DDR5 Intel
        assert_eq!(MicroArch::SapphireRapids.estimated_mem_latency_ns(),
                   MicroArch::AlderLake.estimated_mem_latency_ns());
        // DDR5 AMD
        assert_eq!(MicroArch::Zen5.estimated_mem_latency_ns(),
                   MicroArch::Zen4.estimated_mem_latency_ns());
    }
}
