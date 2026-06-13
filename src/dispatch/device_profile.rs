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
    /// AVX-512 + Intel AMX tile instructions (Sapphire Rapids+).
    Avx512Amx,
    Neon,
    /// ARM SVE (Scalable Vector Extension) — runtime vector length 128..2048 bits.
    Sve,
    /// ARM SVE2 — superset of SVE with additional integer/crypto/bitmanip instructions.
    Sve2,
    /// Apple AMX (M1+): 32x32 matrix block operations via undocumented coprocessor.
    NeonAmx,
}

/// Hardware dot-product capability — determines which native compute instructions
/// are available for quantized and floating-point GEMM microkernels.
///
/// Query order: NativeBf16 > NativeFp16 > NativeInt4x8 > NativeFp4 > NativeInt8* >
///              SimdAssisted > SimdBasic > None
///
/// Each level implies all lower levels are also achievable (e.g., NativeBf16
/// hardware can always do F32 FMA as fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DotProductCap {
    // === Native floating-point dot-product ===
    /// Hardware-native BF16 dot-product (x86 VDPBF16PS / ARM BFMMLA / GPU HMMA bf16).
    /// FP32 accumulate, no software conversion needed.
    NativeBf16,
    /// Hardware-native FP16 dot-product (ARM FMMLA / GPU HMMA fp16).
    /// x86 has no native FP16 compute (AVX-512 FP16 exists but rare).
    NativeFp16,

    // === Native integer dot-product ===
    /// Hardware-native 4-bit×8-bit (AMD GFX12 a4w8 WMMA).
    NativeInt4x8,
    /// Hardware-native FP4 tensor core (NVIDIA SM100+ tcgen05).
    NativeFp4,
    /// Hardware-native INT8 tensor core (SM80/SM90 IMMA/WGMMA).
    NativeInt8Tc,
    /// Hardware-native INT8 SIMD dot-product (x86 VNNI VPDPBUSD / ARM SVE2 SDOT).
    NativeInt8Simd,
    /// Hardware-native INT8 tile (Intel AMX TDPBSSD / TDPBF16PS for BF16).
    NativeInt8Tile,

    // === Software-assisted ===
    /// Has SIMD (256-bit+) but no dot-product instruction (AVX2 FMA, NEON FMLA).
    /// INT8 compute: manual multiply-add in SIMD registers.
    SimdAssisted,
    /// Basic SIMD (128-bit only, SSE2-level).
    SimdBasic,
    /// No SIMD — scalar only.
    None,
}

/// Quantization capability declaration for the device.
///
/// Declares which quantized formats the hardware can execute natively
/// (hardware dot-product instructions) vs. assisted (software emulation).
/// Used by the compiler to select optimal quantized GEMM codegen paths.
#[derive(Debug, Clone)]
pub struct QuantCapability {
    /// Quantization formats with hardware-native dot-product support.
    /// E.g., BF16 on VDPBF16PS/BFMMLA/HMMA, FP16 on FMMLA/HMMA fp16.
    pub native_formats: Vec<crate::quant::QuantType>,
    /// Quantization formats feasible via software-assisted paths.
    /// E.g., INT4/INT8 on SIMD without native dot-product instructions.
    pub assisted_formats: Vec<crate::quant::QuantType>,
    /// Tensor core generation, if available:
    /// - `None`: no tensor core / tile matrix hardware
    /// - `Some(1)`: first-gen (AMX TDPBSSD/TDPBF16PS, SM80+ Tensor Core)
    /// - `Some(2)`: second-gen (AMX FP16, SM90+ WGMMA)
    /// - `Some(3)`: third-gen (SM100+ FP4/FP6 TC)
    pub tensor_core_gen: Option<u8>,
}

/// ISV (Independent Software Vendor) library availability — runtime detected.
///
/// Only third-party libraries belong here. Hardware ISA extensions (AMX, AVX-512, etc.)
/// are handled via `IsaLevel` and `KernelConfig` flags.
#[derive(Debug, Clone, Default)]
pub struct IsvCapabilities {
    /// Intel oneDNN (MKL-DNN) — x86_64 GEMM acceleration
    pub onednn_available: bool,
    /// Apple Accelerate (vDSP/BLAS)
    pub accelerate_available: bool,
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
    /// Quantization capability declaration (native vs. assisted formats)
    pub quant_capabilities: QuantCapability,
    /// ISV library availability (runtime detected)
    pub isv: IsvCapabilities,
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
        let fma_ports = arch.fma_ports() as f64;
        let peak_gflops_f32 =
            physical_cores as f64 * arch.estimated_freq_ghz() * fma_ports * simd_w as f64 * 2.0;

        // Conservative bandwidth estimate based on memory generation
        let peak_bandwidth_gbs = if arch.estimated_mem_latency_ns() > 75 {
            80.0 // DDR5
        } else {
            50.0 // DDR4
        };

        let quant_capabilities = detect_quant_capabilities(&arch, &hw_info, isa);

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
            quant_capabilities,
            isv: detect_isv_capabilities(isa),
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
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 32,
            IsaLevel::Avx2 => 16,
            IsaLevel::Neon | IsaLevel::NeonAmx => 32,
            IsaLevel::Sve | IsaLevel::Sve2 => 32, // z0-z31
            IsaLevel::Scalar => 0,
        }
    }

    /// SIMD register width in bytes.
    /// AVX2 → 32 (256-bit ymm), AVX-512 → 64 (512-bit zmm), NEON → 16 (128-bit v)
    #[inline]
    pub fn simd_width_bytes(&self) -> usize {
        match self.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 64,
            IsaLevel::Avx2 => 32,
            IsaLevel::Neon | IsaLevel::NeonAmx => 16,
            // SVE: runtime-determined, return minimum guaranteed (128-bit).
            // Actual VL is queried at runtime via RDVL.
            IsaLevel::Sve | IsaLevel::Sve2 => self.hw_info.isa.sve_vl_bytes.max(16),
            IsaLevel::Scalar => 4, // single f32
        }
    }

    /// Number of f32 elements per SIMD register.
    #[inline]
    pub fn simd_width_f32(&self) -> usize {
        self.simd_width_bytes() / 4
    }

    /// Per-element SIMD width for a given DType.
    #[inline]
    pub fn simd_width(&self, dtype: crate::types::DType) -> usize {
        self.simd_width_bytes() / dtype.size_bytes()
    }

    /// Peak GFLOPS estimate for a given DType.
    /// F16/BF16 on ISAs with native half-precision support get ~2x throughput.
    pub fn peak_gflops(&self, dtype: crate::types::DType) -> f64 {
        use crate::types::DType;
        match dtype {
            DType::F32 => self.peak_gflops_f32,
            DType::F16 | DType::BF16 => match self.isa {
                IsaLevel::Avx512 | IsaLevel::Avx512Amx => self.peak_gflops_f32 * 2.0,
                IsaLevel::Neon | IsaLevel::NeonAmx | IsaLevel::Sve | IsaLevel::Sve2 => self.peak_gflops_f32 * 2.0,
                _ => self.peak_gflops_f32,
            },
            // FP8: sub-byte types processed via quantized GEMM; throughput >= FP16
            DType::F8E4M3 | DType::F8E5M2 => match self.isa {
                IsaLevel::Avx512 | IsaLevel::Avx512Amx => self.peak_gflops_f32 * 4.0,
                IsaLevel::Neon | IsaLevel::NeonAmx | IsaLevel::Sve | IsaLevel::Sve2 => self.peak_gflops_f32 * 4.0,
                _ => self.peak_gflops_f32 * 2.0,
            },
            // FP6/FP4: sub-byte packed; throughput modeled as >= FP8
            DType::F6E3M2 | DType::F6E2M3 | DType::F4E2M1 => match self.isa {
                IsaLevel::Avx512 | IsaLevel::Avx512Amx => self.peak_gflops_f32 * 4.0,
                IsaLevel::Neon | IsaLevel::NeonAmx | IsaLevel::Sve | IsaLevel::Sve2 => self.peak_gflops_f32 * 4.0,
                _ => self.peak_gflops_f32 * 2.0,
            },
            DType::U8 => self.peak_gflops_f32,
        }
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
    /// F16/BF16 elements are 2 bytes vs F32's 4 bytes, so micropanels fit
    /// ~2× more elements in the same cache budget → KC doubles → higher reuse.
    ///
    /// Alignment: KC to 4 (SIMD), MC to MR, NC to NR.
    /// Small matrices (m*n*k < 4096) use direct path to avoid packing overhead.
    pub fn gemm_blocking(&self, m: usize, n: usize, k: usize, dtype: crate::types::DType) -> GemmBlocking {
        let (mr, nr) = self.microkernel_mr_nr();
        let elem_bytes = dtype.size_bytes();

        // Query WisdomDb for empirically-tuned parameters
        let dtype_id = dtype.elem_id();
        if let Some(blocking) = self.query_wisdom_blocking(m, n, k, mr, nr, elem_bytes, dtype_id) {
            return blocking;
        }

        // Fallback: analytical heuristic
        self.gemm_blocking_heuristic(m, n, k, dtype)
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
        elem_bytes: usize,
        dtype_id: u8,
    ) -> Option<GemmBlocking> {
        let db_ref = crate::autotuning::global_wisdom_db();
        let db = db_ref.lock().ok()?;
        let fp = self.hw_info.fingerprint();
        let cached = db.get_gemm_blocking(&fp, m, n, k, elem_bytes, dtype_id)?;

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

    /// Query the WisdomDb for cached JIT codegen parameters for a GEMM shape.
    ///
    /// Returns `Some(JitParams)` if a cached autotuning result exists with
    /// JIT-specific parameters for this exact problem shape on the current hardware.
    pub fn query_wisdom_jit_params(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: crate::types::DType,
    ) -> Option<crate::autotuning::search_space::JitParams> {
        let db_ref = crate::autotuning::global_wisdom_db();
        let db = db_ref.lock().ok()?;
        let fp = self.hw_info.fingerprint();
        let cached = db.get_gemm_blocking(&fp, m, n, k, dtype.size_bytes(), dtype.elem_id())?;
        cached.config.jit.clone()
    }

    /// Analytical heuristic for GEMM blocking with dtype awareness.
    fn gemm_blocking_heuristic(&self, m: usize, n: usize, k: usize, dtype: crate::types::DType) -> GemmBlocking {
        let (l1, l2, l3) = self.cache_sizes();
        let (mr, nr) = self.microkernel_mr_nr();
        let elem_size = dtype.size_bytes();

        // Small matrix optimization: skip blocking overhead for tiny matrices.
        // Packing cost dominates when the total volume is small.
        if m.saturating_mul(n).saturating_mul(k) < 4096 {
            let (l1, _, _) = self.cache_sizes();
            // Clamp kc so micropanels (MR×KC + NR×KC) still fit in L1
            let max_kc = l1 / (elem_size * (mr + nr));
            let kc = k.max(1).min(max_kc.max(1));
            return GemmBlocking {
                kc,
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
        let elems = (l1 * 3 / 4) / (2 * std::mem::size_of::<f32>()); // 2 buffers × sizeof(f32)
        // Align to SIMD width
        (elems / simd_w) * simd_w
    }

    /// Whether hardware has efficient 2D transpose instructions.
    /// AVX-512: VPERM*. NEON: TRN*. SVE: compact/unsqueeze.
    pub fn has_hw_transpose(&self) -> bool {
        matches!(self.isa, IsaLevel::Avx512 | IsaLevel::Avx512Amx | IsaLevel::Sve | IsaLevel::Sve2)
    }

    /// Whether hardware has efficient arbitrary-lane shuffle/permute.
    /// AVX-512: VPERMI2*. AVX2: VPERMD (limited). NEON: TBL.
    pub fn has_hw_permute(&self) -> bool {
        matches!(self.isa, IsaLevel::Avx512 | IsaLevel::Avx512Amx | IsaLevel::Sve | IsaLevel::Sve2)
    }

    /// Whether hardware supports memory-level broadcast without explicit replication.
    /// AVX-512: VBROADCAST*. NEON: DUP. SVE: DUP.
    pub fn has_hw_broadcast(&self) -> bool {
        matches!(self.isa,
            IsaLevel::Avx2 | IsaLevel::Avx512 | IsaLevel::Avx512Amx |
            IsaLevel::Neon | IsaLevel::NeonAmx | IsaLevel::Sve | IsaLevel::Sve2
        )
    }

    /// L1 data cache budget ratio (0.0-1.0) for TileLevelFusion working set.
    /// Higher values for architectures with larger L1D or hardware prefetch.
    pub fn l1_budget_ratio(&self) -> f64 {
        match self.isa {
            // Sapphire Rapids+ has 48KB L1D + excellent prefetch
            IsaLevel::Avx512Amx => 0.90,
            // Standard 32KB L1D
            IsaLevel::Avx512 => 0.85,
            // AVX2 machines often have smaller L1D, be conservative
            IsaLevel::Avx2 => 0.80,
            // ARM cores vary; 64KB on Neoverse, 32KB on Cortex
            IsaLevel::Neon | IsaLevel::NeonAmx => 0.85,
            IsaLevel::Sve | IsaLevel::Sve2 => 0.85,
            IsaLevel::Scalar => 0.75,
        }
    }

    /// K-dimension unroll factor for GEMM inner loop.
    /// Driven by FMA throughput and register pressure.
    pub fn k_unroll_factor(&self) -> usize {
        match self.isa {
            // 32 zmm regs → can afford 8-accumulator unroll
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 8,
            // 16 ymm regs → 4-accumulator unroll
            IsaLevel::Avx2 => 4,
            // NEON: 32 q-regs but typically 4-6 accumulator
            IsaLevel::Neon | IsaLevel::NeonAmx => 4,
            // SVE: VL-dependent but 4 accumulators safe baseline
            IsaLevel::Sve | IsaLevel::Sve2 => 4,
            IsaLevel::Scalar => 1,
        }
    }

    /// Register pressure cost factor for fusion scoring (§0.2.6).
    /// Higher = more conservative fusion (more register pressure).
    /// AVX-512 has 32 zmm → lower cost; AVX2 has 16 ymm → higher cost.
    pub fn reg_cost_factor(&self) -> f64 {
        match self.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 0.0005,
            IsaLevel::Avx2 => 0.001,
            IsaLevel::Neon | IsaLevel::NeonAmx => 0.0008,
            IsaLevel::Sve | IsaLevel::Sve2 => 0.0006,
            IsaLevel::Scalar => 0.002,
        }
    }

    /// Reserved SIMD registers for GEMM microkernel accumulator.
    pub fn gemm_accumulator_regs(&self) -> usize {
        match self.isa {
            // BLIS-style MR×NR tiling uses MR*NR accumulators
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 14 * 4, // mr=14, packed 4-wide
            IsaLevel::Avx2 => 6 * 3, // mr=6, packed 3-wide
            IsaLevel::Neon | IsaLevel::NeonAmx => 8 * 4,
            IsaLevel::Sve | IsaLevel::Sve2 => 4 * 4,
            IsaLevel::Scalar => 1,
        }
    }

    /// Minimum number of f32 elements before parallelization is worthwhile.
    /// Below this threshold, use sequential execution.
    pub fn parallel_threshold(&self) -> usize {
        // Rough heuristic: need at least ~4096 elements per thread to amortize overhead
        4096 * self.physical_cores
    }

    /// §0.2.9 GEMM tile sizes for ExecPattern derivation.
    /// Returns (tile_m, tile_n, tile_k) parameterized on ISA level.
    pub fn gemm_tile_sizes(&self) -> (usize, usize, usize) {
        match self.isa {
            IsaLevel::Avx512Amx => (14, 32, 256),
            IsaLevel::Avx512 => (14, 32, 256),
            IsaLevel::Avx2 => (6, 16, 128),
            IsaLevel::NeonAmx => (8, 12, 128),
            IsaLevel::Neon => (8, 12, 128),
            IsaLevel::Sve => (4, 16, 128),
            IsaLevel::Sve2 => (4, 16, 128),
            IsaLevel::Scalar => (1, 1, 1),
        }
    }

    /// Highest-performance dot-product capability available on this hardware.
    ///
    /// Drives quantized GEMM microkernel instruction selection (SPEC/23 §3.2).
    /// Codegen matches on `DotProductCap` variants, never on hardware model names.
    pub fn dot_product_cap(&self) -> DotProductCap {
        // Query order: BF16 > FP16 > INT4 > FP4 > INT8 TC > INT8 SIMD > INT8 tile >
        //              SIMD assisted > basic > none
        if self.arch.has_bf16() {
            return DotProductCap::NativeBf16;
        }
        if self.hw_info.isa.avx512fp16 {
            return DotProductCap::NativeFp16;
        }
        // GFX12 a4w8: detected via GPU path (not yet in CPU MicroArch)
        // if self.has_wmma_a4w8() { return DotProductCap::NativeInt4x8; }
        // SM100+ FP4 TC: detected via GPU path
        // if self.has_fp4_tc() { return DotProductCap::NativeFp4; }
        // SM80/SM90 INT8 TC: detected via GPU path
        // if self.has_int8_tc() { return DotProductCap::NativeInt8Tc; }
        if self.arch.has_vnni() || self.hw_info.isa.avx512vnni {
            return DotProductCap::NativeInt8Simd;
        }
        if self.arch.has_amx() {
            return DotProductCap::NativeInt8Tile;
        }
        if matches!(self.isa, IsaLevel::Sve2) {
            // SVE2 has SDOT/UDOT for INT8 dot-product
            return DotProductCap::NativeInt8Simd;
        }
        let sw = self.simd_width_bytes();
        if sw >= 32 {
            return DotProductCap::SimdAssisted; // AVX2 256-bit, NEON+SVE
        }
        if sw >= 16 {
            return DotProductCap::SimdBasic; // SSE2 128-bit, NEON 128-bit
        }
        DotProductCap::None
    }
}

fn detect_isa_level(arch: MicroArch) -> IsaLevel {
    if arch.use_avx512() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("amx-tile") && is_x86_feature_detected!("amx-bf16") {
            return IsaLevel::Avx512Amx;
        }
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
        let hw = crate::autotuning::HwInfo::detect();
        if hw.isa.sve2 {
            return IsaLevel::Sve2;
        }
        if hw.isa.sve {
            return IsaLevel::Sve;
        }
        // Apple AMX is available on all Apple Silicon (M1+) — detect via macOS target.
        #[cfg(target_os = "macos")]
        return IsaLevel::NeonAmx;
        #[cfg(not(target_os = "macos"))]
        return IsaLevel::Neon;
    }
    #[allow(unreachable_code)]
    IsaLevel::Scalar
}

/// Detect ISV library availability at runtime.
fn detect_isv_capabilities(isa: IsaLevel) -> IsvCapabilities {
    let onednn_available = {
        #[cfg(feature = "onednn")]
        { crate::isv::onednn::OneDnnBackend::is_available() }
        #[cfg(not(feature = "onednn"))]
        { false }
    };

    let accelerate_available = {
        #[cfg(feature = "accelerate")]
        { crate::isv::accelerate::AccelerateBackend::is_available() }
        #[cfg(not(feature = "accelerate"))]
        { false }
    };

    IsvCapabilities {
        onednn_available,
        accelerate_available,
    }
}

/// Detect quantization capabilities from hardware features.
///
/// Determines which `QuantType` formats have native dot-product support
/// and which require software-assisted paths. Also detects tensor core
/// generation (AMX gen, GPU SM version).
fn detect_quant_capabilities(
    arch: &MicroArch,
    hw_info: &HwInfo,
    isa: IsaLevel,
) -> QuantCapability {
    let mut native = Vec::new();
    let mut assisted = Vec::new();

    // F32 is always natively supported via FMA
    native.push(crate::quant::QuantType::F32);

    // BF16 native: VDPBF16PS (x86), BFMMLA (ARM), HMMA bf16 (GPU)
    if arch.has_bf16() {
        native.push(crate::quant::QuantType::Bf16);
    }

    // FP16 native: AVX-512 FP16, ARM FMMLA, GPU HMMA fp16
    if hw_info.isa.avx512fp16 {
        native.push(crate::quant::QuantType::F16);
    }

    // K-Quant and classic formats: always available via software assist
    // on any ISA with SIMD width >= 128-bit
    let simd_bytes = match isa {
        IsaLevel::Avx512 | IsaLevel::Avx512Amx => 64,
        IsaLevel::Avx2 => 32,
        IsaLevel::Neon | IsaLevel::NeonAmx => 16,
        IsaLevel::Sve | IsaLevel::Sve2 => hw_info.isa.sve_vl_bytes.max(16),
        IsaLevel::Scalar => 4,
    };

    if simd_bytes >= 16 {
        // All K-Quant and IQ formats are assisted on capable SIMD hardware
        assisted.push(crate::quant::QuantType::Q2K);
        assisted.push(crate::quant::QuantType::Q3K);
        assisted.push(crate::quant::QuantType::Q4K);
        assisted.push(crate::quant::QuantType::Q5K);
        assisted.push(crate::quant::QuantType::Q6K);
        assisted.push(crate::quant::QuantType::Q8K);
        assisted.push(crate::quant::QuantType::IQ1S);
        assisted.push(crate::quant::QuantType::IQ1M);
        assisted.push(crate::quant::QuantType::IQ2XXS);
        assisted.push(crate::quant::QuantType::IQ2XS);
        assisted.push(crate::quant::QuantType::IQ2S);
        assisted.push(crate::quant::QuantType::IQ3XXS);
        assisted.push(crate::quant::QuantType::IQ3S);
        assisted.push(crate::quant::QuantType::IQ4NL);
        assisted.push(crate::quant::QuantType::IQ4XS);
        // Classic GGML formats
        assisted.push(crate::quant::QuantType::Q4_0);
        assisted.push(crate::quant::QuantType::Q4_1);
        assisted.push(crate::quant::QuantType::Q5_0);
        assisted.push(crate::quant::QuantType::Q5_1);
        assisted.push(crate::quant::QuantType::Q8_0);
        assisted.push(crate::quant::QuantType::Q8_1);
        // External formats
        assisted.push(crate::quant::QuantType::AWQ4);
        assisted.push(crate::quant::QuantType::GPTQ4);
        assisted.push(crate::quant::QuantType::Squeeze);
        assisted.push(crate::quant::QuantType::TQ1_0);
        assisted.push(crate::quant::QuantType::TQ2_0);
        assisted.push(crate::quant::QuantType::Mxfp4 { block_size: 32 });
        assisted.push(crate::quant::QuantType::Nvfp4);
    }

    // Tensor core / tile matrix generation detection
    let tensor_core_gen = if arch.has_amx() {
        // AMX first-gen: Sapphire Rapids+ (TDPBSSD/TDPBF16PS)
        Some(1u8)
    } else if arch.has_amx_fp16() {
        // AMX second-gen: Granite Rapids+ (AMX FP16)
        Some(2u8)
    } else {
        // GPU tensor core detection is handled by the GPU-specific codegen path
        // (Cuda SM version, HIP gfx arch). CPU-side defaults to None.
        None
    };

    QuantCapability {
        native_formats: native,
        assisted_formats: assisted,
        tensor_core_gen,
    }
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
            IsaLevel::Avx2 | IsaLevel::Avx512 | IsaLevel::Avx512Amx
        ));
        #[cfg(target_arch = "aarch64")]
        assert!(matches!(
            profile.isa,
            IsaLevel::Neon | IsaLevel::NeonAmx | IsaLevel::Sve | IsaLevel::Sve2
        ));
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
        let blocking = profile.gemm_blocking(1024, 1024, 1024, crate::types::DType::F32);
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
        let blocking = profile.gemm_blocking(4, 8, 16, crate::types::DType::F32);
        assert_eq!(blocking.kc, 16, "small matrix should use KC=k");
        assert_eq!(blocking.mc, 4, "small matrix should use MC=m");
        assert_eq!(blocking.nc, 8, "small matrix should use NC=n");

        // Volume = 25*40*64 = 64000 > 4096 -> normal blocking
        let b = profile.gemm_blocking(25, 40, 64, crate::types::DType::F32);
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
            let b = profile.gemm_blocking(m, n, k, crate::types::DType::F32);

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
            let b = profile.gemm_blocking(m, n, k, crate::types::DType::F32);
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

        let b = profile.gemm_blocking(1024, 1024, 1024, crate::types::DType::F32);
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

    #[test]
    fn test_dot_product_cap() {
        let profile = DeviceProfile::detect();
        let cap = profile.dot_product_cap();
        // On any real machine, should be at least SimdBasic
        assert!(
            !matches!(cap, DotProductCap::None),
            "dot_product_cap should not be None on real hardware, got {cap:?}"
        );
        eprintln!("DotProductCap: {cap:?}");
    }

    #[test]
    fn test_dot_product_cap_isa_consistency() {
        let profile = DeviceProfile::detect();
        let cap = profile.dot_product_cap();

        // Verify consistency with IsaLevel
        match profile.isa {
            IsaLevel::Avx512Amx => {
                // AMX implies at least NativeInt8Tile or NativeBf16
                assert!(
                    matches!(cap, DotProductCap::NativeBf16 | DotProductCap::NativeInt8Tile | DotProductCap::NativeInt8Simd),
                    "Avx512Amx should have NativeBf16/NativeInt8Tile/NativeInt8Simd, got {cap:?}"
                );
            }
            IsaLevel::Avx512 => {
                assert!(
                    matches!(cap, DotProductCap::NativeBf16 | DotProductCap::NativeInt8Simd | DotProductCap::SimdAssisted),
                    "Avx512 should have NativeBf16/NativeInt8Simd/SimdAssisted, got {cap:?}"
                );
            }
            IsaLevel::Avx2 => {
                assert!(
                    matches!(cap, DotProductCap::SimdAssisted),
                    "Avx2 should have SimdAssisted, got {cap:?}"
                );
            }
            IsaLevel::Sve2 => {
                assert!(
                    matches!(cap, DotProductCap::NativeInt8Simd),
                    "Sve2 should have NativeInt8Simd, got {cap:?}"
                );
            }
            _ => {}
        }
    }

    #[test]
    fn test_gemm_blocking_copy_derives() {
        let original = GemmBlocking { kc: 256, mc: 128, nc: 512, mr: 6, nr: 16 };
        let copied = original;
        assert_eq!(original.kc, copied.kc);
        assert_eq!(original.mc, copied.mc);
        assert_eq!(original.nc, copied.nc);
        assert_eq!(original.mr, copied.mr);
        assert_eq!(original.nr, copied.nr);
    }

    #[test]
    fn test_isa_level_equality_and_hash() {
        use std::collections::HashSet;
        let levels = [
            IsaLevel::Scalar, IsaLevel::Avx2, IsaLevel::Avx512,
            IsaLevel::Avx512Amx, IsaLevel::Neon, IsaLevel::Sve,
            IsaLevel::Sve2, IsaLevel::NeonAmx,
        ];
        let set: HashSet<IsaLevel> = levels.iter().copied().collect();
        assert_eq!(set.len(), levels.len());
        assert_eq!(IsaLevel::Avx512, IsaLevel::Avx512);
        assert_ne!(IsaLevel::Avx2, IsaLevel::Avx512);
    }

    #[test]
    fn test_dot_product_cap_ordering_invariants() {
        let profile = DeviceProfile::detect();
        let cap = profile.dot_product_cap();
        match cap {
            DotProductCap::NativeBf16 => {
                assert!(profile.arch.has_bf16());
            }
            DotProductCap::NativeInt8Simd => {
                assert!(profile.arch.has_vnni() || profile.hw_info.isa.avx512vnni || matches!(profile.isa, IsaLevel::Sve2));
            }
            DotProductCap::NativeInt8Tile => {
                assert!(profile.arch.has_amx());
            }
            DotProductCap::SimdAssisted => {
                assert!(profile.simd_width_bytes() >= 32);
            }
            DotProductCap::SimdBasic => {
                let sw = profile.simd_width_bytes();
                assert!(sw >= 16 && sw < 32);
            }
            _ => {}
        }
    }

    #[test]
    fn test_peak_gflops_dtype_scaling() {
        let profile = DeviceProfile::detect();
        let f32_gflops = profile.peak_gflops(crate::types::DType::F32);
        assert!(f32_gflops > 0.0);
        assert_eq!(f32_gflops, profile.peak_gflops_f32);
        let u8_gflops = profile.peak_gflops(crate::types::DType::U8);
        assert_eq!(u8_gflops, profile.peak_gflops_f32);
    }

    #[test]
    fn test_peak_gflops_half_precision_on_avx512() {
        let profile = DeviceProfile::detect();
        let f32_gflops = profile.peak_gflops_f32;
        let bf16_gflops = profile.peak_gflops(crate::types::DType::BF16);
        let f16_gflops = profile.peak_gflops(crate::types::DType::F16);
        match profile.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => {
                assert!(bf16_gflops > f32_gflops, "BF16 should be faster on AVX-512");
                assert!(f16_gflops > f32_gflops, "FP16 should be faster on AVX-512");
            }
            _ => {
                assert_eq!(bf16_gflops, f32_gflops);
                assert_eq!(f16_gflops, f32_gflops);
            }
        }
    }

    #[test]
    fn test_simd_width_dtype_dispatch() {
        let profile = DeviceProfile::detect();
        let bytes = profile.simd_width_bytes();
        let f32_lanes = profile.simd_width(crate::types::DType::F32);
        assert_eq!(f32_lanes, bytes / 4);
        let bf16_lanes = profile.simd_width(crate::types::DType::BF16);
        assert_eq!(bf16_lanes, bytes / 2);
        let u8_lanes = profile.simd_width(crate::types::DType::U8);
        assert_eq!(u8_lanes, bytes / 1);
    }

    #[test]
    fn test_gemm_blocking_bf16_larger_kc_than_f32() {
        let profile = DeviceProfile::detect();
        let b_f32 = profile.gemm_blocking(2048, 2048, 2048, crate::types::DType::F32);
        let b_bf16 = profile.gemm_blocking(2048, 2048, 2048, crate::types::DType::BF16);
        assert!(b_bf16.kc >= b_f32.kc,
            "BF16 KC={} should be >= F32 KC={} (2x smaller elements in same cache budget)",
            b_bf16.kc, b_f32.kc);
    }

    #[test]
    fn test_display_profile_contains_key_fields() {
        let profile = DeviceProfile::detect();
        let display = format!("{profile}");
        assert!(display.contains("GFLOPS"), "Display should contain GFLOPS");
        assert!(display.contains("GB/s"), "Display should contain GB/s");
        assert!(display.contains("FLOP/B"), "Display should contain FLOP/B");
        assert!(display.contains(&profile.physical_cores.to_string()));
        assert!(display.contains(&profile.logical_cores.to_string()));
    }

    #[test]
    fn test_roofline_ridge_point_arithmetic() {
        let profile = DeviceProfile::detect();
        let ridge = profile.roofline_ridge_point();
        let expected = profile.peak_gflops_f32 / profile.peak_bandwidth_gbs;
        assert!((ridge - expected).abs() < 1e-10);
        assert!(ridge > 0.0);
    }

    #[test]
    fn test_l1_budget_ratio_range() {
        let profile = DeviceProfile::detect();
        let ratio = profile.l1_budget_ratio();
        assert!(ratio >= 0.75 && ratio <= 0.90,
            "L1 budget ratio {ratio} should be in [0.75, 0.90]");
    }

    #[test]
    fn test_k_unroll_factor_matches_isa() {
        let profile = DeviceProfile::detect();
        let factor = profile.k_unroll_factor();
        match profile.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => assert_eq!(factor, 8),
            IsaLevel::Avx2 => assert_eq!(factor, 4),
            IsaLevel::Neon | IsaLevel::NeonAmx => assert_eq!(factor, 4),
            IsaLevel::Sve | IsaLevel::Sve2 => assert_eq!(factor, 4),
            IsaLevel::Scalar => assert_eq!(factor, 1),
        }
    }

    #[test]
    fn test_quant_capabilities_always_contains_f32() {
        let profile = DeviceProfile::detect();
        assert!(profile.quant_capabilities.native_formats.contains(&crate::quant::QuantType::F32),
            "F32 must always be in native_formats");
    }

    #[test]
    fn test_gemm_tile_sizes_all_positive() {
        let profile = DeviceProfile::detect();
        let (tile_m, tile_n, tile_k) = profile.gemm_tile_sizes();
        assert!(tile_m > 0, "tile_m must be positive");
        assert!(tile_n > 0, "tile_n must be positive");
        assert!(tile_k > 0, "tile_k must be positive");
        match profile.isa {
            IsaLevel::Scalar => {
                assert_eq!((tile_m, tile_n, tile_k), (1, 1, 1));
            }
            _ => {
                assert!(tile_m >= 4, "Non-scalar tile_m should be >= 4");
                assert!(tile_n >= 12, "Non-scalar tile_n should be >= 12");
                assert!(tile_k >= 128, "Non-scalar tile_k should be >= 128");
            }
        }
    }

    // ── 10 new tests for uncovered logic paths ──

    /// Test has_hw_transpose: on real x86_64 hardware with AVX-512 it should be true,
    /// on AVX2 it should be false. On ARM SVE it should be true.
    #[test]
    fn test_has_hw_transpose_matches_isa() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let result = profile.has_hw_transpose();

        // Assert: consistency with IsaLevel
        let expected = matches!(
            profile.isa,
            IsaLevel::Avx512 | IsaLevel::Avx512Amx | IsaLevel::Sve | IsaLevel::Sve2
        );
        assert_eq!(result, expected,
            "has_hw_transpose={result} inconsistent with isa={:?}", profile.isa);
    }

    /// Test has_hw_permute: mirrors has_hw_transpose on current ISA variants.
    #[test]
    fn test_has_hw_permute_matches_isa() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let result = profile.has_hw_permute();

        // Assert
        let expected = matches!(
            profile.isa,
            IsaLevel::Avx512 | IsaLevel::Avx512Amx | IsaLevel::Sve | IsaLevel::Sve2
        );
        assert_eq!(result, expected,
            "has_hw_permute={result} inconsistent with isa={:?}", profile.isa);
    }

    /// Test has_hw_broadcast: all SIMD-capable ISAs (except Scalar) should broadcast.
    #[test]
    fn test_has_hw_broadcast_true_for_all_simd() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let result = profile.has_hw_broadcast();

        // Assert: Scalar has no broadcast; every other ISA should
        if matches!(profile.isa, IsaLevel::Scalar) {
            assert!(!result, "Scalar should not have hw broadcast");
        } else {
            assert!(result, "{:?} should have hw broadcast", profile.isa);
        }
    }

    /// Test reg_cost_factor: must be positive and ISA-consistent.
    /// AVX-512 has lowest cost (most registers), Scalar highest.
    #[test]
    fn test_reg_cost_factor_positive_and_isa_ordered() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let factor = profile.reg_cost_factor();

        // Assert
        assert!(factor > 0.0, "reg_cost_factor must be positive, got {factor}");
        let expected = match profile.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 0.0005,
            IsaLevel::Avx2 => 0.001,
            IsaLevel::Neon | IsaLevel::NeonAmx => 0.0008,
            IsaLevel::Sve | IsaLevel::Sve2 => 0.0006,
            IsaLevel::Scalar => 0.002,
        };
        assert!((factor - expected).abs() < 1e-15,
            "reg_cost_factor={factor} != expected={expected} for {:?}", profile.isa);
    }

    /// Test gemm_accumulator_regs: must be positive and consistent with ISA.
    /// Scalar = 1, AVX2 = 6*3=18, AVX-512 = 14*4=56.
    #[test]
    fn test_gemm_accumulator_regs_positive_and_isa_consistent() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let regs = profile.gemm_accumulator_regs();

        // Assert
        assert!(regs >= 1, "accumulator regs must be >= 1, got {regs}");
        let expected = match profile.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx => 14 * 4,
            IsaLevel::Avx2 => 6 * 3,
            IsaLevel::Neon | IsaLevel::NeonAmx => 8 * 4,
            IsaLevel::Sve | IsaLevel::Sve2 => 4 * 4,
            IsaLevel::Scalar => 1,
        };
        assert_eq!(regs, expected,
            "accumulator regs={regs} != expected={expected} for {:?}", profile.isa);
    }

    /// Test peak_gflops for sub-byte types (FP8, FP6, FP4).
    /// On AVX-512 they should get 4x F32 throughput; on AVX2 they get 2x.
    #[test]
    fn test_peak_gflops_fp8_fp6_fp4_scaling() {
        // Arrange
        let profile = DeviceProfile::detect();
        let f32_gflops = profile.peak_gflops_f32;

        // Act
        let fp8 = profile.peak_gflops(crate::types::DType::F8E4M3);
        let fp6 = profile.peak_gflops(crate::types::DType::F6E3M2);
        let fp4 = profile.peak_gflops(crate::types::DType::F4E2M1);

        // Assert
        let multiplier = match profile.isa {
            IsaLevel::Avx512 | IsaLevel::Avx512Amx |
            IsaLevel::Neon | IsaLevel::NeonAmx |
            IsaLevel::Sve | IsaLevel::Sve2 => 4.0,
            _ => 2.0,
        };
        let expected = f32_gflops * multiplier;
        assert!((fp8 - expected).abs() < 1e-6,
            "FP8 gflops={fp8} != expected={expected}");
        assert!((fp6 - expected).abs() < 1e-6,
            "FP6 gflops={fp6} != expected={expected}");
        assert!((fp4 - expected).abs() < 1e-6,
            "FP4 gflops={fp4} != expected={expected}");
    }

    /// Test that quant_capabilities assisted_formats is non-empty on any SIMD-capable ISA.
    /// Scalar ISA should have empty assisted_formats.
    #[test]
    fn test_quant_capabilities_assisted_formats_presence() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let assisted = &profile.quant_capabilities.assisted_formats;

        // Assert
        if matches!(profile.isa, IsaLevel::Scalar) {
            assert!(assisted.is_empty(),
                "Scalar should have no assisted formats, got {} entries", assisted.len());
        } else {
            assert!(!assisted.is_empty(),
                "{:?} should have assisted quant formats", profile.isa);
            // Verify specific entries exist: Q4_0, Q8_0, AWQ4, GPTQ4
            assert!(assisted.contains(&crate::quant::QuantType::Q4_0), "Q4_0 missing");
            assert!(assisted.contains(&crate::quant::QuantType::Q8_0), "Q8_0 missing");
            assert!(assisted.contains(&crate::quant::QuantType::AWQ4), "AWQ4 missing");
            assert!(assisted.contains(&crate::quant::QuantType::GPTQ4), "GPTQ4 missing");
        }
    }

    /// Test tensor_core_gen consistency: AMX-capable arch should have gen>=1,
    /// non-AMX arch should have None (on CPU path).
    #[test]
    fn test_tensor_core_gen_amx_consistency() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let tc_gen = profile.quant_capabilities.tensor_core_gen;

        // Assert
        let has_amx = profile.arch.has_amx();
        let has_amx_fp16 = profile.arch.has_amx_fp16();
        if has_amx_fp16 {
            assert_eq!(tc_gen, Some(2),
                "AMX FP16 arch should report tensor_core_gen=2, got {tc_gen:?}");
        } else if has_amx {
            assert_eq!(tc_gen, Some(1),
                "AMX arch should report tensor_core_gen=1, got {tc_gen:?}");
        } else {
            assert_eq!(tc_gen, None,
                "Non-AMX CPU arch should have None tensor_core_gen, got {tc_gen:?}");
        }
    }

    /// Test isv capabilities defaults: on standard builds (no onednn/accelerate features),
    /// both flags should be false.
    #[test]
    fn test_isv_capabilities_default() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let isv = &profile.isv;

        // Assert: without explicit feature enablement, both should be false
        #[cfg(not(feature = "onednn"))]
        assert!(!isv.onednn_available, "oneDNN should not be available without feature flag");
        #[cfg(not(feature = "accelerate"))]
        assert!(!isv.accelerate_available, "Accelerate should not be available without feature flag");
    }

    /// Test GemmBlocking Debug derive produces non-empty output.
    /// Also verify cache_sizes returns consistent triple with l1 <= l2 <= l3.
    #[test]
    fn test_gemm_blocking_debug_and_cache_hierarchy() {
        // Arrange
        let profile = DeviceProfile::detect();
        let blocking = profile.gemm_blocking(512, 512, 512, crate::types::DType::F32);

        // Act
        let debug_str = format!("{blocking:?}");

        // Assert: Debug output should contain all field names
        assert!(debug_str.contains("kc"), "Debug should contain kc");
        assert!(debug_str.contains("mc"), "Debug should contain mc");
        assert!(debug_str.contains("nc"), "Debug should contain nc");

        // Assert: cache hierarchy monotonicity
        let (l1, l2, l3) = profile.cache_sizes();
        assert!(l1 > 0, "L1D should be > 0");
        assert!(l2 >= l1, "L2={l2} should be >= L1={l1}");
        assert!(l3 >= l2, "L3={l3} should be >= L2={l2}");
    }

    // ── 10 additional tests for remaining uncovered logic paths ──

    /// Test DotProductCap PartialEq/Eq/Hash for all variants.
    /// Ensures every enum variant is distinct and can be used in collections.
    #[test]
    fn test_dot_product_cap_all_variants_distinct() {
        // Arrange
        use std::collections::HashSet;
        let all_variants = [
            DotProductCap::NativeBf16,
            DotProductCap::NativeFp16,
            DotProductCap::NativeInt4x8,
            DotProductCap::NativeFp4,
            DotProductCap::NativeInt8Tc,
            DotProductCap::NativeInt8Simd,
            DotProductCap::NativeInt8Tile,
            DotProductCap::SimdAssisted,
            DotProductCap::SimdBasic,
            DotProductCap::None,
        ];

        // Act
        let set: HashSet<DotProductCap> = all_variants.iter().copied().collect();

        // Assert: all 10 variants are distinct (HashSet deduplicates on Eq)
        assert_eq!(set.len(), 10, "All 10 DotProductCap variants should be distinct");

        // Assert: reflexivity
        assert_eq!(DotProductCap::NativeBf16, DotProductCap::NativeBf16);
        assert_ne!(DotProductCap::NativeBf16, DotProductCap::SimdAssisted);
    }

    /// Test GemmBlocking field values are never zero for real blocking results.
    /// Also verify KC is always a multiple of 4 (SIMD alignment contract).
    #[test]
    fn test_gemm_blocking_kc_aligned_for_various_shapes() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act & Assert: test multiple shapes to exercise blocking heuristic
        for &(m, n, k) in &[
            (64, 64, 64),
            (128, 256, 512),
            (4096, 1024, 768),
            (2048, 128, 4096),
            (1, 4096, 4096),
        ] {
            let b = profile.gemm_blocking(m, n, k, crate::types::DType::F32);
            assert!(b.kc > 0, "KC must be > 0 for m={m} n={n} k={k}");
            assert!(b.mc > 0, "MC must be > 0 for m={m} n={n} k={k}");
            assert!(b.nc > 0, "NC must be > 0 for m={m} n={n} k={k}");
            assert!(b.mr > 0, "MR must be > 0");
            assert!(b.nr > 0, "NR must be > 0");
            assert_eq!(b.kc % 4, 0,
                "KC={} must be aligned to 4 for m={m} n={n} k={k}", b.kc);
        }
    }

    /// Test cache_sizes returns values consistent with kernel_config fields.
    #[test]
    fn test_cache_sizes_matches_kernel_config() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let (l1, l2, l3) = profile.cache_sizes();

        // Assert: must match kernel_config exactly
        assert_eq!(l1, profile.kernel_config.l1d,
            "cache_sizes L1D should match kernel_config.l1d");
        assert_eq!(l2, profile.kernel_config.l2,
            "cache_sizes L2 should match kernel_config.l2");
        assert_eq!(l3, profile.kernel_config.l3,
            "cache_sizes L3 should match kernel_config.l3");
    }

    /// Test simd_width_f32 is exactly simd_width_bytes / 4.
    #[test]
    fn test_simd_width_f32_matches_bytes_div_4() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let bytes = profile.simd_width_bytes();
        let f32_lanes = profile.simd_width_f32();

        // Assert: mathematical identity
        assert_eq!(f32_lanes, bytes / 4,
            "simd_width_f32={f32_lanes} should equal simd_width_bytes/4={}", bytes / 4);
    }

    /// Test parallel_threshold scales with physical_cores.
    #[test]
    fn test_parallel_threshold_scales_with_cores() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let threshold = profile.parallel_threshold();

        // Assert: threshold = 4096 * physical_cores
        let expected = 4096 * profile.physical_cores;
        assert_eq!(threshold, expected,
            "parallel_threshold={threshold} should equal 4096 * physical_cores={expected}");
    }

    /// Test microkernel_mr_nr returns values from kernel_config.
    #[test]
    fn test_microkernel_mr_nr_matches_kernel_config() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let (mr, nr) = profile.microkernel_mr_nr();

        // Assert
        assert_eq!(mr, profile.kernel_config.mr,
            "microkernel_mr_nr MR={mr} should match kernel_config.mr={}", profile.kernel_config.mr);
        assert_eq!(nr, profile.kernel_config.nr,
            "microkernel_mr_nr NR={nr} should match kernel_config.nr={}", profile.kernel_config.nr);
    }

    /// Test peak_gflops returns positive values for all DType variants.
    #[test]
    fn test_peak_gflops_all_dtypes_positive() {
        // Arrange
        use crate::types::DType;
        let profile = DeviceProfile::detect();
        let dtypes = [
            DType::F32, DType::F16, DType::BF16,
            DType::F8E4M3, DType::F8E5M2,
            DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
            DType::U8,
        ];

        // Act & Assert
        for &dt in &dtypes {
            let gflops = profile.peak_gflops(dt);
            assert!(gflops > 0.0,
                "peak_gflops for {dt:?} must be positive, got {gflops}");
        }
    }

    /// Test IsvCapabilities Default trait produces both fields as false.
    #[test]
    fn test_isv_capabilities_default_trait() {
        // Arrange & Act
        let isv = IsvCapabilities::default();

        // Assert
        assert!(!isv.onednn_available, "default onednn_available should be false");
        assert!(!isv.accelerate_available, "default accelerate_available should be false");
    }

    /// Test query_wisdom_jit_params returns None for a shape not in WisdomDb.
    #[test]
    fn test_query_wisdom_jit_params_returns_none_for_uncached_shape() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act: use an arbitrary shape unlikely to be pre-cached
        let result = profile.query_wisdom_jit_params(12345, 67890, 11111, crate::types::DType::F32);

        // Assert: no cached JIT params for this shape
        assert!(result.is_none(),
            "query_wisdom_jit_params should return None for uncached shape");
    }

    /// Test GemmBlocking with BF16 dtype: MC should be aligned to MR,
    /// and NC aligned to NR, matching F32 alignment contracts.
    #[test]
    fn test_gemm_blocking_bf16_alignment_contracts() {
        // Arrange
        let profile = DeviceProfile::detect();
        let (mr, nr) = profile.microkernel_mr_nr();

        // Act
        let b = profile.gemm_blocking(2048, 2048, 2048, crate::types::DType::BF16);

        // Assert: same alignment contracts as F32
        if b.mc < 2048 {
            assert_eq!(b.mc % mr, 0,
                "BF16 MC={} must be aligned to MR={mr}", b.mc);
        }
        if b.nc < 2048 {
            assert_eq!(b.nc % nr, 0,
                "BF16 NC={} must be aligned to NR={nr}", b.nc);
        }
        assert_eq!(b.kc % 4, 0, "BF16 KC={} must be aligned to 4", b.kc);
        assert!(b.kc > 0 && b.mc > 0 && b.nc > 0,
            "All blocking params must be positive: KC={} MC={} NC={}", b.kc, b.mc, b.nc);
    }

    // ── Additional tests for uncovered logic paths ──

    /// Test GemmBlocking Clone derive works correctly.
    #[test]
    fn test_gemm_blocking_clone_independent() {
        let original = GemmBlocking { kc: 128, mc: 64, nc: 256, mr: 6, nr: 16 };
        let cloned = original.clone();
        assert_eq!(original.kc, cloned.kc);
        assert_eq!(original.mc, cloned.mc);
        assert_eq!(original.nc, cloned.nc);
        assert_eq!(original.mr, cloned.mr);
        assert_eq!(original.nr, cloned.nr);
    }

    /// Test IsaLevel Debug derive produces recognizable output for all variants.
    #[test]
    fn test_isa_level_debug_output() {
        let levels = [
            IsaLevel::Scalar, IsaLevel::Avx2, IsaLevel::Avx512,
            IsaLevel::Avx512Amx, IsaLevel::Neon, IsaLevel::Sve,
            IsaLevel::Sve2, IsaLevel::NeonAmx,
        ];
        for level in levels {
            let debug_str = format!("{level:?}");
            assert!(!debug_str.is_empty(), "IsaLevel Debug should not be empty");
        }
    }

    /// Test DotProductCap Debug derive for all variants.
    #[test]
    fn test_dot_product_cap_debug_output() {
        let caps = [
            DotProductCap::NativeBf16, DotProductCap::NativeFp16,
            DotProductCap::NativeInt4x8, DotProductCap::NativeFp4,
            DotProductCap::NativeInt8Tc, DotProductCap::NativeInt8Simd,
            DotProductCap::NativeInt8Tile, DotProductCap::SimdAssisted,
            DotProductCap::SimdBasic, DotProductCap::None,
        ];
        for cap in caps {
            let debug_str = format!("{cap:?}");
            assert!(!debug_str.is_empty(), "DotProductCap Debug should not be empty");
        }
    }

    /// Test DeviceProfile Display contains arch name and ISA level.
    #[test]
    fn test_device_profile_display_contains_arch() {
        let profile = DeviceProfile::detect();
        let display = format!("{profile}");
        // Must contain the arch name from MicroArch Display
        let arch_display = profile.arch.to_string();
        assert!(display.contains(&arch_display),
            "DeviceProfile Display should contain arch name '{arch_display}'");
    }

    /// Test DeviceProfile.arch matches kernel_config.arch (consistency).
    #[test]
    fn test_device_profile_arch_consistency() {
        let profile = DeviceProfile::detect();
        assert_eq!(profile.arch, profile.kernel_config.arch,
            "DeviceProfile.arch must match kernel_config.arch");
    }

    /// Test peak_gflops F16 and BF16 are equal (same throughput class).
    #[test]
    fn test_peak_gflops_f16_equals_bf16() {
        let profile = DeviceProfile::detect();
        let f16_gflops = profile.peak_gflops(crate::types::DType::F16);
        let bf16_gflops = profile.peak_gflops(crate::types::DType::BF16);
        assert!((f16_gflops - bf16_gflops).abs() < 1e-10,
            "F16 and BF16 should have same peak GFLOPS, got F16={f16_gflops} BF16={bf16_gflops}");
    }

    /// Test peak_gflops F8E4M3 equals F8E5M2 (same throughput class).
    #[test]
    fn test_peak_gflops_fp8_variants_equal() {
        let profile = DeviceProfile::detect();
        let e4m3 = profile.peak_gflops(crate::types::DType::F8E4M3);
        let e5m2 = profile.peak_gflops(crate::types::DType::F8E5M2);
        assert!((e4m3 - e5m2).abs() < 1e-10,
            "FP8 E4M3 and E5M2 should have same peak GFLOPS");
    }

    /// Test QuantCapability Debug derive produces output.
    #[test]
    fn test_quant_capability_debug() {
        let profile = DeviceProfile::detect();
        let debug_str = format!("{:?}", profile.quant_capabilities);
        assert!(debug_str.contains("native_formats"), "Debug should contain native_formats");
        assert!(debug_str.contains("assisted_formats"), "Debug should contain assisted_formats");
    }

    /// Test IsvCapabilities Clone derive.
    #[test]
    fn test_isv_capabilities_clone() {
        let profile = DeviceProfile::detect();
        let cloned = profile.isv.clone();
        assert_eq!(profile.isv.onednn_available, cloned.onednn_available);
        assert_eq!(profile.isv.accelerate_available, cloned.accelerate_available);
    }

    /// Test elem_tile_size is SIMD-aligned and fits in L1 budget.
    #[test]
    fn test_elem_tile_size_l1_budget() {
        let profile = DeviceProfile::detect();
        let tile = profile.elem_tile_size();
        let (l1, _, _) = profile.cache_sizes();
        let simd_w = profile.simd_width_f32();

        // tile * 2 * sizeof(f32) should fit in 75% of L1
        let bytes_used = tile * 2 * 4;
        assert!(bytes_used <= l1 * 3 / 4 + simd_w * 4, // allow rounding
            "elem_tile bytes {bytes_used} should fit in 75% of L1={l1}");
        assert_eq!(tile % simd_w, 0, "tile must be SIMD-aligned");
    }

    /// Test gemm_blocking with all DType variants produces valid results.
    #[test]
    fn test_gemm_blocking_all_dtypes_positive() {
        use crate::types::DType;
        let profile = DeviceProfile::detect();
        let dtypes = [
            DType::F32, DType::F16, DType::BF16,
            DType::F8E4M3, DType::F8E5M2,
            DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
            DType::U8,
        ];
        for &dt in &dtypes {
            let b = profile.gemm_blocking(512, 512, 512, dt);
            assert!(b.kc > 0, "KC must be > 0 for {dt:?}");
            assert!(b.mc > 0, "MC must be > 0 for {dt:?}");
            assert!(b.nc > 0, "NC must be > 0 for {dt:?}");
            assert!(b.mr > 0, "MR must be > 0 for {dt:?}");
            assert!(b.nr > 0, "NR must be > 0 for {dt:?}");
        }
    }

    /// Test l1_budget_ratio: Scalar is 0.75 (lowest), Avx512Amx is 0.90 (highest).
    #[test]
    fn test_l1_budget_ratio_scalar_vs_amx() {
        let profile = DeviceProfile::detect();
        let ratio = profile.l1_budget_ratio();
        // Ratio must be in the documented range.
        assert!(ratio >= 0.75 && ratio <= 0.90,
            "l1_budget_ratio {ratio} outside [0.75, 0.90]");
        // Verify exact match with ISA level.
        let expected = match profile.isa {
            IsaLevel::Avx512Amx => 0.90,
            IsaLevel::Avx512 => 0.85,
            IsaLevel::Avx2 => 0.80,
            IsaLevel::Neon | IsaLevel::NeonAmx => 0.85,
            IsaLevel::Sve | IsaLevel::Sve2 => 0.85,
            IsaLevel::Scalar => 0.75,
        };
        assert!((ratio - expected).abs() < 1e-15,
            "ratio {ratio} != expected {expected}");
    }
}
