//! Hardware profile enumeration for topology-driven fusion decisions.
//!
//! Defines 12 hardware profiles covering mainstream accelerators and CPUs.
//! Each profile has distinct fusion preferences based on memory hierarchy,
//! compute throughput, and instruction set capabilities.

use crate::dispatch::DeviceProfile;
use crate::compiler::codegen::vm::isa_profile::Platform;

/// 12 hardware profiles for topology-driven fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareProfile {
    // NVIDIA CUDA
    CudaSM80,      // Ampere (A100) — aggressive fusion, tensor cores
    CudaSM90,      // Hopper (H100) — max fusion, TMA, FP8
    CudaSM100,     // Blackwell (B100+) — ultra-aggressive, FP4/FP6

    // AMD ROCm
    RocmMI200,     // CDNA2 (MI200) — matrix cores, 64-wide wavefront
    RocmMI300,     // CDNA3 (MI300) — enhanced matrix, FP8

    // CPU x86_64
    CpuAvx2,       // Intel/AMD AVX2 — conservative fusion, register pressure
    CpuAvx512,     // Intel AVX-512 — moderate fusion, 32 ZMM regs
    CpuAvx10_2,    // Intel AVX10.2 — AVX-512 subset, no downclocking

    // Apple Silicon
    AppleM1,       // M1 — AMX tiles, unified memory
    AppleM2,       // M2 — enhanced AMX, wider memory
    AppleM3,       // M3 — dynamic caching, ray tracing

    // ARM Server
    ArmNeoverse,   // Neoverse V1/V2 — SVE2, 128-bit vectors

    // Fallback
    Generic,
}

impl HardwareProfile {
    /// All 12 hardware profile variants (excluding Generic fallback).
    pub const ALL: [HardwareProfile; 12] = [
        HardwareProfile::CudaSM80,
        HardwareProfile::CudaSM90,
        HardwareProfile::CudaSM100,
        HardwareProfile::RocmMI200,
        HardwareProfile::RocmMI300,
        HardwareProfile::CpuAvx2,
        HardwareProfile::CpuAvx512,
        HardwareProfile::CpuAvx10_2,
        HardwareProfile::AppleM1,
        HardwareProfile::AppleM2,
        HardwareProfile::AppleM3,
        HardwareProfile::ArmNeoverse,
    ];

    /// Map to IsaProfile Platform (many-to-one).
    pub fn platform(&self) -> Platform {
        match self {
            Self::CpuAvx2 | Self::CpuAvx512 | Self::CpuAvx10_2 => Platform::X86_64 {
                has_avx512: matches!(self, Self::CpuAvx512),
                has_bf16: matches!(self, Self::CpuAvx512 | Self::CpuAvx10_2),
                has_vnni: matches!(self, Self::CpuAvx512 | Self::CpuAvx10_2),
                has_avx512fp16: false,
                has_f16c: true,
                has_amx: matches!(self, Self::CpuAvx512),
                has_amx_fp16: false,
                has_amx_complex: false,
                has_amx_transpose: false,
                has_amx_fp8: false,
                has_avx10_2: matches!(self, Self::CpuAvx10_2),
                has_apx: matches!(self, Self::CpuAvx10_2),
                has_sparse_mask_intersect: false,
            },
            Self::ArmNeoverse => Platform::AArch64 {
                has_bf16: true,
                has_dotprod: true,
                has_i8mm: true,
                has_sve: true,
                has_sve2: true,
                sve_vl: 128,
                has_sme: true,
                has_sme2: true,
                has_sme_f16f16: true,
                has_sme_i16i64: true,
                sme_vl: 128,
            },
            Self::AppleM1 | Self::AppleM2 | Self::AppleM3 => {
                let gpu_family = match self {
                    Self::AppleM1 => 7,
                    Self::AppleM2 => 8,
                    _ => 9,
                };
                Platform::Metal {
                    simd_width: 32,
                    gpu_family,
                    has_simdgroup_matrix: true,
                    threadgroup_mem_kb: 32,
                }
            }
            Self::CudaSM80 => Platform::Cuda {
                sm_version: 80,
                warp_size: 32,
                shared_mem_kb: 164,
                reg_file_per_sm: 65536,
                max_regs_per_thread: 255,
                has_wgmma: false,
                has_tma: false,
                has_warp_spec: false,
                has_fp8: false,
                has_tmem: false,
                has_block_scaled: false,
                has_native_fp4: false,
                has_native_fp6: false,
                has_cluster: false,
                has_2cta_mma: false,
                tmem_size_kb: 0,
            },
            Self::CudaSM90 => Platform::Cuda {
                sm_version: 90,
                warp_size: 32,
                shared_mem_kb: 228,
                reg_file_per_sm: 65536,
                max_regs_per_thread: 255,
                has_wgmma: true,
                has_tma: true,
                has_warp_spec: true,
                has_fp8: true,
                has_tmem: false,
                has_block_scaled: false,
                has_native_fp4: false,
                has_native_fp6: false,
                has_cluster: false,
                has_2cta_mma: false,
                tmem_size_kb: 0,
            },
            Self::CudaSM100 => Platform::Cuda {
                sm_version: 100,
                warp_size: 32,
                shared_mem_kb: 228,
                reg_file_per_sm: 65536,
                max_regs_per_thread: 255,
                has_wgmma: true,
                has_tma: true,
                has_warp_spec: true,
                has_fp8: true,
                has_tmem: true,
                has_block_scaled: true,
                has_native_fp4: true,
                has_native_fp6: true,
                has_cluster: true,
                has_2cta_mma: true,
                tmem_size_kb: 256,
            },
            Self::RocmMI200 => Platform::Hip {
                gfx_arch: 908,
                wave_size: 64,
                has_mfma: true,
                has_mfma_v2: false,
                has_fp8_mfma: false,
                has_fp4_mfma: false,
                vgpr_per_cu: 256,
                lds_size_kb: 64,
                infinity_cache_mb: 0,
            },
            Self::RocmMI300 => Platform::Hip {
                gfx_arch: 942,
                wave_size: 64,
                has_mfma: true,
                has_mfma_v2: false,
                has_fp8_mfma: false,
                has_fp4_mfma: false,
                vgpr_per_cu: 512,
                lds_size_kb: 64,
                infinity_cache_mb: 128,
            },
            Self::Generic => Platform::X86_64 {
                has_avx512: false,
                has_f16c: false,
                has_bf16: false,
                has_vnni: false,
                has_avx512fp16: false,
                has_amx: false,
                has_amx_fp16: false,
                has_amx_complex: false,
                has_amx_transpose: false,
                has_amx_fp8: false,
                has_avx10_2: false,
                has_apx: false,
                has_sparse_mask_intersect: false,
            },
        }
    }

    /// Detect hardware profile from DeviceProfile.
    pub fn detect(profile: &DeviceProfile) -> Self {
        #[cfg(feature = "jit-cuda")]
        {
            if let Some(sm) = detect_cuda_sm() {
                return match sm {
                    80..=89 => Self::CudaSM80,
                    90..=99 => Self::CudaSM90,
                    100.. => Self::CudaSM100,
                    _ => Self::Generic,
                };
            }
        }

        #[cfg(feature = "jit-hip")]
        {
            if let Some(gfx) = detect_rocm_gfx() {
                return match gfx {
                    90 => Self::RocmMI200,
                    94 => Self::RocmMI300,
                    _ => Self::Generic,
                };
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use crate::dispatch::IsaLevel;
            match profile.isa {
                IsaLevel::Avx512 | IsaLevel::Avx512Amx => {
                    // Distinguish AVX10.2 from legacy AVX-512
                    if profile.arch.has_avx512fp16() {
                        return Self::CpuAvx10_2;
                    }
                    return Self::CpuAvx512;
                }
                IsaLevel::Avx2 => return Self::CpuAvx2,
                _ => {}
            }
        }

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            // Detect Apple Silicon generation via sysctl
            if let Some(gen) = detect_apple_silicon_gen() {
                return match gen {
                    1 => Self::AppleM1,
                    2 => Self::AppleM2,
                    3.. => Self::AppleM3,
                    _ => Self::Generic,
                };
            }
        }

        #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
        {
            use crate::dispatch::IsaLevel;
            if matches!(profile.isa, IsaLevel::Sve2) {
                return Self::ArmNeoverse;
            }
        }

        Self::Generic
    }

    /// Fusion aggressiveness score (0.0 = conservative, 1.0 = max fusion).
    pub fn fusion_aggressiveness(self) -> f32 {
        match self {
            Self::CudaSM100 => 1.0,
            Self::CudaSM90 => 0.95,
            Self::RocmMI300 => 0.9,
            Self::CudaSM80 | Self::RocmMI200 => 0.85,
            Self::AppleM3 => 0.75,
            Self::AppleM2 => 0.7,
            Self::CpuAvx10_2 => 0.65,
            Self::AppleM1 => 0.6,
            Self::CpuAvx512 => 0.55,
            Self::ArmNeoverse => 0.5,
            Self::CpuAvx2 => 0.4,
            Self::Generic => 0.3,
        }
    }

    /// Minimum fusion benefit threshold (speedup multiplier).
    pub fn min_fusion_benefit(self) -> f32 {
        match self {
            Self::CudaSM100 | Self::CudaSM90 => 1.15, // GPU: 15% speedup
            Self::RocmMI300 | Self::CudaSM80 | Self::RocmMI200 => 1.2,
            Self::AppleM3 | Self::AppleM2 | Self::AppleM1 => 1.25,
            Self::CpuAvx10_2 | Self::CpuAvx512 => 1.3,
            Self::ArmNeoverse => 1.35,
            Self::CpuAvx2 => 1.4, // CPU: 40% speedup required
            Self::Generic => 1.5,
        }
    }

    /// Max ops per fusion group (avoid register spills).
    pub fn max_fusion_depth(self) -> usize {
        match self {
            Self::CudaSM100 | Self::CudaSM90 => 8,
            Self::RocmMI300 | Self::CudaSM80 | Self::RocmMI200 => 7,
            Self::AppleM3 | Self::AppleM2 => 6,
            Self::CpuAvx512 | Self::CpuAvx10_2 => 5,
            Self::AppleM1 | Self::ArmNeoverse => 4,
            Self::CpuAvx2 => 3,
            Self::Generic => 2,
        }
    }

    /// Whether to prefer large GEMM fusion (vs small elementwise fusion).
    pub fn prefer_gemm_fusion(self) -> bool {
        matches!(self,
            Self::CudaSM100 | Self::CudaSM90 | Self::CudaSM80 |
            Self::RocmMI300 | Self::RocmMI200 |
            Self::AppleM3 | Self::AppleM2 | Self::AppleM1
        )
    }

    /// GPU tensor core generation (0 = no tensor cores).
    pub fn tensor_core_gen(self) -> u32 {
        match self {
            Self::CudaSM100 => 100,
            Self::CudaSM90 => 90,
            Self::CudaSM80 => 80,
            Self::RocmMI300 => 94,
            Self::RocmMI200 => 90,
            _ => 0,
        }
    }

    /// Whether this GPU supports TMA (Tensor Memory Accelerator).
    pub fn has_tma(self) -> bool {
        matches!(self, Self::CudaSM90 | Self::CudaSM100)
    }

    /// Whether this GPU supports TMEM (Tensor Memory, ~256KB/SM on Blackwell).
    /// TMEM is independent of shared memory and provides low-latency (<1ns)
    /// scratch space for attention score staging, MoE weight caching, etc.
    pub fn has_tmem(self) -> bool {
        matches!(self, Self::CudaSM100)
    }

    /// Whether this GPU supports native FP4/F6 MMA.
    pub fn has_native_fp4(self) -> bool {
        matches!(self, Self::CudaSM100)
    }

    /// Whether this CPU has AMX (Advanced Matrix Extensions).
    pub fn has_amx(self) -> bool {
        matches!(self, Self::CpuAvx512)
    }

    /// Whether this CPU has SME2 (Scalable Matrix Extension).
    pub fn has_sme2(self) -> bool {
        matches!(self, Self::ArmNeoverse)
    }

    /// Number of general-purpose registers available for epilogue.
    pub fn gpr_count(self) -> usize {
        match self {
            Self::CpuAvx10_2 => 31,
            Self::CpuAvx512 | Self::CpuAvx2 => 16,
            Self::ArmNeoverse => 31,
            _ => 16,
        }
    }

    /// SIMD register count for the target ISA.
    pub fn num_simd_regs(self) -> usize {
        match self {
            Self::CpuAvx512 | Self::CpuAvx10_2 => 32,
            Self::CpuAvx2 => 16,
            Self::ArmNeoverse => 32,
            Self::CudaSM100 => 256,
            Self::CudaSM90 | Self::CudaSM80 => 255,
            Self::RocmMI300 | Self::RocmMI200 => 256,
            _ => 16,
        }
    }

    /// Whether this profile supports VNNI (Vector Neural Network Instructions).
    pub fn has_vnni(self) -> bool {
        matches!(self, Self::CpuAvx512 | Self::CpuAvx10_2)
    }

    /// Shared memory bytes per SM/CU (GPU profiles only).
    pub fn shared_memory_bytes(self) -> usize {
        match self {
            Self::CudaSM100 => 228 * 1024,
            Self::CudaSM90 => 227 * 1024,
            Self::CudaSM80 => 163 * 1024,
            Self::RocmMI300 => 64 * 1024,
            Self::RocmMI200 => 64 * 1024,
            _ => 0,
        }
    }

    /// Max threads per block (GPU only).
    pub fn max_threads_per_block(self) -> usize {
        match self {
            Self::CudaSM100 | Self::CudaSM90 | Self::CudaSM80 => 1024,
            Self::RocmMI300 | Self::RocmMI200 => 1024,
            _ => 0,
        }
    }

    /// Cache sizes in bytes: (L1, L2, L3).
    pub fn cache_sizes(self) -> (usize, usize, usize) {
        match self {
            Self::CpuAvx512 | Self::CpuAvx10_2 => (32 * 1024, 256 * 1024, 12 * 1024 * 1024),
            Self::CpuAvx2 => (32 * 1024, 256 * 1024, 8 * 1024 * 1024),
            Self::ArmNeoverse => (64 * 1024, 512 * 1024, 8 * 1024 * 1024),
            _ => (0, 0, 0),
        }
    }

    /// Maximum epilogue chain depth (ops after anchor GEMM) based on register budget.
    ///
    /// Derived from SIMD register count and GEMM accumulator usage.
    /// GPU profiles have large register files -> deep epilogue.
    /// CPU profiles are constrained by ymm/zmm count.
    pub fn max_epilogue_depth(self) -> usize {
        let simd_regs = self.num_simd_regs();
        // Reserve ~50% of SIMD registers for GEMM accumulators + address pointers.
        // The remaining ~50% are available for epilogue temporaries.
        let available_for_epilogue = simd_regs / 2;
        // Each epilogue op needs ~1-2 SIMD registers for temporaries.
        // Use 2 as conservative estimate to avoid register spills.
        (available_for_epilogue / 2).max(1).min(self.max_fusion_depth())
    }

    /// Whether this profile supports deep-chain quantized GEMM epilogue.
    ///
    /// Quantized GEMM (W4A4/W4A8) requires a dequant step after accumulation.
    /// Profiles with VNNI/SVE2 dot instructions can fuse dequant + activation
    /// into the epilogue without extra register pressure.
    pub fn supports_quant_epilogue(self) -> bool {
        self.has_vnni() || self.has_sme2() || self.tensor_core_gen() > 0
    }

    /// Weight for compute ROI in fusion cost model (0.0-2.0).
    ///
    /// GPU profiles have high compute throughput -> compute ROI weight is low
    /// (fusion benefit is dominated by memory traffic savings).
    /// CPU profiles with limited compute -> compute ROI weight is higher
    /// (fusion may increase compute overhead).
    pub fn compute_roi_weight(self) -> f64 {
        match self {
            Self::CudaSM100 | Self::CudaSM90 => 0.3,
            Self::CudaSM80 | Self::RocmMI300 | Self::RocmMI200 => 0.4,
            Self::CpuAvx512 | Self::CpuAvx10_2 => 0.8,
            Self::AppleM3 | Self::AppleM2 | Self::AppleM1 => 0.7,
            Self::ArmNeoverse => 0.9,
            Self::CpuAvx2 => 1.2,
            Self::Generic => 1.5,
        }
    }

    /// Weight for cache ROI in fusion cost model (0.5-1.5).
    ///
    /// Profiles with large shared memory / L1 benefit more from cache-friendly
    /// fusion patterns (tile-level fusion, compute-root).
    pub fn cache_roi_weight(self) -> f64 {
        let (l1, _, _) = self.cache_sizes();
        let shared = self.shared_memory_bytes();
        let effective_cache = l1.max(shared) as f64;
        // Normalize to 0.5-1.5 range based on cache size
        // 256KB+ -> 1.5 (high benefit), <32KB -> 0.5 (low benefit)
        ((effective_cache / (32.0 * 1024.0)).ln().max(0.0) + 1.0).min(1.5).max(0.5)
    }

    /// GPU GEMM 三级分块参数: (cta_m, cta_n, cta_k, warp_m, warp_n, mma_k).
    /// 非 GPU profile返回 (0,0,0,0,0,0) 表示 CPU 路径，不使用 GPU 分块.
    pub fn gpu_gemm_tiles(self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            // SM100 Blackwell: 128×256×64 CTA, 64×32 tcgen05, K=16
            Self::CudaSM100 => (128, 256, 64, 64, 32, 16),
            // SM90 Hopper: 128×128×64 CTA, 64×16 wgmma, K=16
            Self::CudaSM90 => (128, 128, 64, 64, 16, 16),
            // SM80 Ampere: 128×128×32 CTA, 64×16 mma.sync, K=16
            Self::CudaSM80 => (128, 128, 32, 64, 16, 16),
            // MI300 CDNA3: 128×128×64 CTA, 32×32 MFMA, K=16
            Self::RocmMI300 => (128, 128, 64, 32, 32, 16),
            // MI200 CDNA2: 128×128×32 CTA, 16×16 MFMA, K=16
            Self::RocmMI200 => (128, 128, 32, 16, 16, 16),
            // CPU profiles: 无 GPU 分块
            _ => (0, 0, 0, 0, 0, 0),
        }
    }

    /// GPU GEMM 双缓冲流水线深度.
    /// 0/1 = 无流水线, 2 = ping-pong double buffer, 3 = 三缓冲.
    /// SMEM 预算决定最大深度: 每级需要 cta_m×cta_k + cta_k×cta_n 的 shared memory.
    pub fn gpu_pipeline_depth(self) -> usize {
        match self {
            // SM100 Blackwell: 228KB SMEM, 三缓冲 (3 × 128×64×4B + 3 × 64×256×4B = 281KB > 228KB)
            // 实际: 三缓冲 tile 需 ~187KB → fit in 228KB ✓
            Self::CudaSM100 => 3,
            // SM90 Hopper: 227KB SMEM, 三缓冲
            Self::CudaSM90 => 3,
            // SM80 Ampere: 164KB SMEM, 双缓冲 (2 × 128×32×4B + 2 × 32×128×4B = 64KB)
            Self::CudaSM80 => 2,
            // MI300 CDNA3: 64KB SMEM, 双缓冲
            Self::RocmMI300 => 2,
            // MI200 CDNA2: 64KB SMEM, 双缓冲
            Self::RocmMI200 => 2,
            // CPU/其他: 无 GPU 流水线
            _ => 0,
        }
    }
}

#[cfg(feature = "jit-cuda")]
fn detect_cuda_sm() -> Option<u32> {
    use crate::gpu::cuda::CudaDriver;
    CudaDriver::load().ok()?.compute_capability().ok()
}

#[cfg(feature = "jit-hip")]
fn detect_rocm_gfx() -> Option<u32> {
    use crate::gpu::hip::HipDriver;
    HipDriver::load().ok()?.gfx_arch(0).ok()
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn detect_apple_silicon_gen() -> Option<u32> {
    use std::process::Command;
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
        .ok()?;
    let brand = String::from_utf8_lossy(&output.stdout);
    if brand.contains("M1") { Some(1) }
    else if brand.contains("M2") { Some(2) }
    else if brand.contains("M3") { Some(3) }
    else if brand.contains("M4") { Some(4) }
    else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fusion_aggressiveness monotonicity ──

    #[test]
    fn fusion_aggressiveness_gpu_highest() {
        assert!(HardwareProfile::CudaSM100.fusion_aggressiveness() >= HardwareProfile::CudaSM90.fusion_aggressiveness());
        assert!(HardwareProfile::CudaSM90.fusion_aggressiveness() >= HardwareProfile::CudaSM80.fusion_aggressiveness());
    }

    #[test]
    fn fusion_aggressiveness_range() {
        for variant in all_variants() {
            let v = variant.fusion_aggressiveness();
            assert!((0.0..=1.0).contains(&v), "{variant:?}: {v} out of [0,1]");
        }
    }

    // ── min_fusion_benefit ──

    #[test]
    fn min_fusion_benefit_all_above_one() {
        for variant in all_variants() {
            assert!(variant.min_fusion_benefit() >= 1.0, "{variant:?} < 1.0");
        }
    }

    #[test]
    fn min_fusion_benefit_gpu_lower_than_cpu() {
        assert!(HardwareProfile::CudaSM100.min_fusion_benefit() < HardwareProfile::CpuAvx2.min_fusion_benefit());
    }

    // ── max_fusion_depth ──

    #[test]
    fn max_fusion_depth_gpu_deepest() {
        assert!(HardwareProfile::CudaSM100.max_fusion_depth() >= HardwareProfile::CpuAvx2.max_fusion_depth());
    }

    #[test]
    fn max_fusion_depth_positive() {
        for variant in all_variants() {
            assert!(variant.max_fusion_depth() >= 1, "{variant:?}: depth 0");
        }
    }

    // ── prefer_gemm_fusion ──

    #[test]
    fn prefer_gemm_fusion_gpu_true() {
        assert!(HardwareProfile::CudaSM100.prefer_gemm_fusion());
        assert!(HardwareProfile::CudaSM90.prefer_gemm_fusion());
        assert!(HardwareProfile::CudaSM80.prefer_gemm_fusion());
    }

    #[test]
    fn prefer_gemm_fusion_cpu_false() {
        assert!(!HardwareProfile::CpuAvx2.prefer_gemm_fusion());
        assert!(!HardwareProfile::CpuAvx512.prefer_gemm_fusion());
        assert!(!HardwareProfile::Generic.prefer_gemm_fusion());
    }

    // ── GPU features ──

    #[test]
    fn tensor_core_gen_gpu_only() {
        assert!(HardwareProfile::CudaSM100.tensor_core_gen() > 0);
        assert!(HardwareProfile::CudaSM90.tensor_core_gen() > 0);
        assert_eq!(HardwareProfile::CpuAvx2.tensor_core_gen(), 0);
        assert_eq!(HardwareProfile::Generic.tensor_core_gen(), 0);
    }

    #[test]
    fn tma_sm90_and_above() {
        assert!(HardwareProfile::CudaSM90.has_tma());
        assert!(HardwareProfile::CudaSM100.has_tma());
        assert!(!HardwareProfile::CudaSM80.has_tma());
        assert!(!HardwareProfile::CpuAvx512.has_tma());
    }

    #[test]
    fn tmem_and_fp4_sm100_only() {
        assert!(HardwareProfile::CudaSM100.has_tmem());
        assert!(HardwareProfile::CudaSM100.has_native_fp4());
        assert!(!HardwareProfile::CudaSM90.has_tmem());
        assert!(!HardwareProfile::CudaSM90.has_native_fp4());
    }

    // ── CPU features ──

    #[test]
    fn amx_avx512_only() {
        assert!(HardwareProfile::CpuAvx512.has_amx());
        assert!(!HardwareProfile::CpuAvx2.has_amx());
        assert!(!HardwareProfile::CpuAvx10_2.has_amx());
    }

    #[test]
    fn sme2_neoverse_only() {
        assert!(HardwareProfile::ArmNeoverse.has_sme2());
        assert!(!HardwareProfile::CpuAvx512.has_sme2());
    }

    #[test]
    fn vnni_avx512_and_avx10() {
        assert!(HardwareProfile::CpuAvx512.has_vnni());
        assert!(HardwareProfile::CpuAvx10_2.has_vnni());
        assert!(!HardwareProfile::CpuAvx2.has_vnni());
    }

    // ── register counts ──

    #[test]
    fn simd_regs_gpu_largest() {
        assert!(HardwareProfile::CudaSM100.num_simd_regs() >= 255);
        assert!(HardwareProfile::CpuAvx512.num_simd_regs() >= 32);
        assert!(HardwareProfile::CpuAvx2.num_simd_regs() >= 16);
    }

    // ── shared memory ──

    #[test]
    fn shared_memory_sm100_largest() {
        assert!(HardwareProfile::CudaSM100.shared_memory_bytes() >= HardwareProfile::CudaSM90.shared_memory_bytes());
        assert!(HardwareProfile::CudaSM90.shared_memory_bytes() >= HardwareProfile::CudaSM80.shared_memory_bytes());
    }

    #[test]
    fn shared_memory_cpu_zero() {
        assert_eq!(HardwareProfile::CpuAvx2.shared_memory_bytes(), 0);
        assert_eq!(HardwareProfile::Generic.shared_memory_bytes(), 0);
    }

    // ── cache sizes ──

    #[test]
    fn cache_sizes_cpu_nonzero() {
        let (l1, l2, l3) = HardwareProfile::CpuAvx512.cache_sizes();
        assert!(l1 > 0 && l2 > 0 && l3 > 0);
    }

    #[test]
    fn cache_sizes_gpu_zero() {
        let (l1, l2, l3) = HardwareProfile::CudaSM100.cache_sizes();
        assert_eq!((l1, l2, l3), (0, 0, 0));
    }

    // ── epilogue depth ──

    #[test]
    fn max_epilogue_depth_positive() {
        for variant in all_variants() {
            assert!(variant.max_epilogue_depth() >= 1, "{variant:?}: 0");
        }
    }

    #[test]
    fn supports_quant_epilogue_gpu_and_vnni() {
        assert!(HardwareProfile::CudaSM100.supports_quant_epilogue());
        assert!(HardwareProfile::CpuAvx512.supports_quant_epilogue());
        assert!(!HardwareProfile::CpuAvx2.supports_quant_epilogue());
        assert!(!HardwareProfile::Generic.supports_quant_epilogue());
    }

    // ── GPU tiles ──

    #[test]
    fn gpu_gemm_tiles_sm100() {
        let (cta_m, cta_n, cta_k, warp_m, warp_n, mma_k) = HardwareProfile::CudaSM100.gpu_gemm_tiles();
        assert!(cta_m > 0 && cta_n > 0 && cta_k > 0);
        assert!(warp_m > 0 && warp_n > 0 && mma_k > 0);
    }

    #[test]
    fn gpu_gemm_tiles_cpu_zero() {
        assert_eq!(HardwareProfile::CpuAvx2.gpu_gemm_tiles(), (0, 0, 0, 0, 0, 0));
    }

    // ── pipeline depth ──

    #[test]
    fn gpu_pipeline_depth_sm100_triple_buffer() {
        assert_eq!(HardwareProfile::CudaSM100.gpu_pipeline_depth(), 3);
    }

    #[test]
    fn gpu_pipeline_depth_cpu_zero() {
        assert_eq!(HardwareProfile::CpuAvx2.gpu_pipeline_depth(), 0);
    }

    // ── compute/cache ROI ──

    #[test]
    fn compute_roi_weight_cpu_higher_than_gpu() {
        assert!(HardwareProfile::CpuAvx2.compute_roi_weight() > HardwareProfile::CudaSM100.compute_roi_weight());
    }

    #[test]
    fn cache_roi_weight_range() {
        for variant in all_variants() {
            let w = variant.cache_roi_weight();
            assert!((0.5..=1.5).contains(&w), "{variant:?}: {w} out of [0.5,1.5]");
        }
    }

    // ── max_threads_per_block ──

    #[test]
    fn max_threads_gpu_1024() {
        assert_eq!(HardwareProfile::CudaSM100.max_threads_per_block(), 1024);
        assert_eq!(HardwareProfile::CpuAvx2.max_threads_per_block(), 0);
    }

    fn all_variants() -> [HardwareProfile; 13] {
        [
            HardwareProfile::CudaSM80,
            HardwareProfile::CudaSM90,
            HardwareProfile::CudaSM100,
            HardwareProfile::RocmMI200,
            HardwareProfile::RocmMI300,
            HardwareProfile::CpuAvx2,
            HardwareProfile::CpuAvx512,
            HardwareProfile::CpuAvx10_2,
            HardwareProfile::AppleM1,
            HardwareProfile::AppleM2,
            HardwareProfile::AppleM3,
            HardwareProfile::ArmNeoverse,
            HardwareProfile::Generic,
            // Note: 14th would need another variant; we have 13 defined + Generic = 14
        ]
    }
}
