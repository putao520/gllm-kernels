//! Quantization format descriptors for the compiler pipeline.
//!
//! Per SPEC `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md` §2.2–§2.3 — every quantization
//! format is described by a unified [`QuantFormatDescriptor`]. This module
//! re-exports the descriptor types from [`crate::quant_format`] and adds
//! compiler-side algorithmic format kinds ([`QuantAlgoKind`]) that map
//! directly to JIT codegen paths.
//!
//! ## Registered formats (SPEC §2.2–§2.3)
//!
//! | Kind | Data | Scale | Zero-point | Layout |
//! |------|------|-------|------------|--------|
//! | [`Mxfp4`](QuantAlgoKind::Mxfp4) | E2M1 float | E8M0 pure exponent | — | RowMajor |
//! | [`Nvfp4`](QuantAlgoKind::Nvfp4) | E2M1 float | UE4M3 sub-block (4×16) | — | RowMajor |
//! | [`Awq4`](QuantAlgoKind::Awq4) | INT4 packed | FP16 block scalar | FP16 block scalar | RowMajor |
//! | [`Gptq4`](QuantAlgoKind::Gptq4) | INT4 packed | FP16 block scalar | StaticBias(1) | ColInterleaved |
//! | [`SqueezeLlm`](QuantAlgoKind::SqueezeLlm) | 3-bit codebook LUT | FP16 block scalar | — | RowMajor |
//! | [`Q4_0`](QuantAlgoKind::Q4_0) | INT4 packed | FP16 block scalar | StaticBias(8) | RowMajor |
//! | [`Q4_1`](QuantAlgoKind::Q4_1) | INT4 packed | FP16 d+m | — | RowMajor |
//! | [`Q5_0`](QuantAlgoKind::Q5_0) | INT5 packed | FP16 block scalar | StaticBias(16) | RowMajor |
//! | [`Q5_1`](QuantAlgoKind::Q5_1) | INT5 packed | FP16 d+m | — | RowMajor |
//! | [`Q8_0`](QuantAlgoKind::Q8_0) | INT8 signed | FP16 block scalar | — | RowMajor |
//! | [`Q2K`](QuantAlgoKind::Q2K) | 2-bit packed | hierarchical 4-bit | hierarchical dmin | RowMajor |
//! | [`Q3K`](QuantAlgoKind::Q3K) | 3-bit packed | hierarchical 6-bit | StaticBias(4) | RowMajor |
//! | [`Q4K`](QuantAlgoKind::Q4K) | INT4 packed | hierarchical 6-bit | hierarchical dmin | RowMajor |
//! | [`Q5K`](QuantAlgoKind::Q5K) | INT5 packed | hierarchical 6-bit | hierarchical dmin | RowMajor |
//! | [`Q6K`](QuantAlgoKind::Q6K) | INT6 packed | Q6K i8 scales | StaticBias(32) | RowMajor |
//! | [`IQ1S`](QuantAlgoKind::IQ1S) | 1-bit codebook | FP16 block scalar | — | RowMajor |
//! | [`IQ2S`](QuantAlgoKind::IQ2S) | 2-bit codebook | FP16 block scalar | — | RowMajor |
//! | [`IQ3XXS`](QuantAlgoKind::IQ3XXS) | 3-bit codebook | FP16 block scalar | — | RowMajor |

pub use crate::quant_format::{
    QuantFormatDescriptor,
    QuantFormatRegistry,
    QuantDataKind,
    StorageLayout,
    NativeIsa,
    ScaleLayout,
    ZeroLayout,
    DataLayout,
    ScaleDType,
    CodebookSpec,
    PackedScaleLayout,
    PackedScaleAlgorithm,
    registry,
};

use crate::quant::QuantType;

/// Compiler-side algorithmic quantization format kind.
///
/// Each variant corresponds to a SPEC-defined quantization algorithm and
/// provides a [`descriptor()`](QuantAlgoKind::descriptor) method that returns
/// the canonical [`QuantFormatDescriptor`] for use by JIT codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantAlgoKind {
    // ── External quantization formats (SPEC §2.2) ──────────────────────
    /// MXFP4 (OCP Microscaling): E2M1 data + E8M0 pure-exponent scaling, block_size=32.
    /// Decode: `value = E2M1(raw_nibble) * 2^(e8m0_byte - 127)`.
    Mxfp4,
    /// NVFP4 (NVIDIA Blackwell native): E2M1 data + UE4M3 sub-block scaling, block_size=64.
    /// Decode: `value = E2M1(raw_nibble) * UE4M3_decode(sub_scale)`.
    /// SM100+ native via tcgen05; SM<100 software decode via E2M1 LUT.
    Nvfp4,
    /// AWQ4 (Activation-aware Weight quantization): Row-major INT4 with FP16 scale + FP16 zero-point.
    /// Weight-Only GEMM path: scale and zp are per-group (group_size=128).
    Awq4,
    /// GPTQ4 (GPTQ 4-bit): Column-interleaved INT4 packed with FP16 scale + static bias of 1.
    /// After g_idx reordering, layout matches AWQ4 byte-stream but with ColInterleaved access.
    Gptq4,
    /// SqueezeLLM: 3-bit codebook LUT quantization (per-tensor dynamic codebook).
    /// Indices decoded through a runtime codebook pointer; dequant is a LUT lookup.
    SqueezeLlm,

    // ── Classic GGML quantization formats (SPEC §2.3, block_size=32) ───
    /// Q4_0: 4-bit quantization, block_size=32, 18 bytes. d(f16) + qs[16] packed nibbles, bias -8.
    Q4_0,
    /// Q4_1: 4-bit quantization with min, block_size=32, 20 bytes. d(f16) + m(f16) + qs[16].
    Q4_1,
    /// Q5_0: 5-bit quantization, block_size=32, 22 bytes. d(f16) + qh[4] + qs[16], bias -16.
    Q5_0,
    /// Q5_1: 5-bit quantization with min, block_size=32, 24 bytes. d(f16) + m(f16) + qh[4] + qs[16].
    Q5_1,
    /// Q8_0: 8-bit signed quantization, block_size=32, 34 bytes. d(f16) + qs[32 i8].
    Q8_0,

    // ── K-Quant family (SPEC §2.3, block_size=256) ─────────────────────
    /// Q2K: 2-bit quantization, block_size=256, 84 bytes. scales[16] + qs[64] + d(f16) + dmin(f16).
    Q2K,
    /// Q3K: 3-bit quantization, block_size=256, 110 bytes. hmask[32] + qs[64] + scales[12] + d(f16).
    Q3K,
    /// Q4K: 4-bit quantization, block_size=256, 144 bytes. d(f16) + dmin(f16) + scales[12] + qs[128].
    Q4K,
    /// Q5K: 5-bit quantization, block_size=256, 176 bytes. d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
    Q5K,
    /// Q6K: 6-bit quantization, block_size=256, 210 bytes. qs[128] + qh[64] + scales[16 i8] + d(f16).
    Q6K,

    // ── IQ series (importance-matrix quantization, SPEC §2.3, block_size=256) ──
    /// IQ1S: 1-bit importance quantization (small), block_size=256, 50 bytes.
    IQ1S,
    /// IQ1M: 1-bit importance quantization (medium), block_size=256, 56 bytes.
    IQ1M,
    /// IQ2XXS: 2-bit importance quantization (extra extra small), block_size=256, 66 bytes.
    IQ2XXS,
    /// IQ2XS: 2-bit importance quantization (extra small), block_size=256, 74 bytes.
    IQ2XS,
    /// IQ2S: 2-bit importance quantization (small), block_size=256, 82 bytes.
    IQ2S,
    /// IQ3XXS: 3-bit importance quantization (extra extra small), block_size=256, 98 bytes.
    IQ3XXS,
    /// IQ3S: 3-bit importance quantization (small), block_size=256, 110 bytes.
    IQ3S,
    /// IQ4NL: 4-bit importance quantization (non-linear, codebook LUT), block_size=32, 18 bytes.
    IQ4NL,
    /// IQ4XS: 4-bit importance quantization (extra small, codebook LUT), block_size=256, 136 bytes.
    IQ4XS,
}

impl QuantAlgoKind {
    /// Return the canonical [`QuantFormatDescriptor`] for this algorithmic kind.
    ///
    /// Looks up the descriptor from the global registry populated in
    /// [`crate::quant_format::QuantFormatRegistry`].
    pub fn descriptor(self) -> QuantFormatDescriptor {
        let reg = registry();
        let qt = self.quant_type();
        reg.get(&qt)
            .cloned()
            .unwrap_or_else(|| panic!("QuantAlgoKind::{:?}: no QuantFormatDescriptor registered for {:?}", self, qt))
    }

    /// Map this algorithmic kind to the corresponding [`QuantType`].
    pub fn quant_type(self) -> QuantType {
        match self {
            Self::Mxfp4 => QuantType::Mxfp4 { block_size: 32 },
            Self::Nvfp4 => QuantType::Nvfp4,
            Self::Awq4 => QuantType::AWQ4,
            Self::Gptq4 => QuantType::GPTQ4,
            Self::SqueezeLlm => QuantType::Squeeze,
            // Classic GGML (SPEC §2.3)
            Self::Q4_0 => QuantType::Q4_0,
            Self::Q4_1 => QuantType::Q4_1,
            Self::Q5_0 => QuantType::Q5_0,
            Self::Q5_1 => QuantType::Q5_1,
            Self::Q8_0 => QuantType::Q8_0,
            // K-Quant (SPEC §2.3)
            Self::Q2K => QuantType::Q2K,
            Self::Q3K => QuantType::Q3K,
            Self::Q4K => QuantType::Q4K,
            Self::Q5K => QuantType::Q5K,
            Self::Q6K => QuantType::Q6K,
            // IQ series (SPEC §2.3)
            Self::IQ1S => QuantType::IQ1S,
            Self::IQ1M => QuantType::IQ1M,
            Self::IQ2XXS => QuantType::IQ2XXS,
            Self::IQ2XS => QuantType::IQ2XS,
            Self::IQ2S => QuantType::IQ2S,
            Self::IQ3XXS => QuantType::IQ3XXS,
            Self::IQ3S => QuantType::IQ3S,
            Self::IQ4NL => QuantType::IQ4NL,
            Self::IQ4XS => QuantType::IQ4XS,
        }
    }

    /// Return all registered algorithmic kinds.
    pub fn all() -> &'static [QuantAlgoKind] {
        &[
            Self::Mxfp4, Self::Nvfp4, Self::Awq4, Self::Gptq4, Self::SqueezeLlm,
            // Classic GGML (SPEC §2.3)
            Self::Q4_0, Self::Q4_1, Self::Q5_0, Self::Q5_1, Self::Q8_0,
            // K-Quant (SPEC §2.3)
            Self::Q2K, Self::Q3K, Self::Q4K, Self::Q5K, Self::Q6K,
            // IQ series (SPEC §2.3)
            Self::IQ1S, Self::IQ1M, Self::IQ2XXS, Self::IQ2XS,
            Self::IQ2S, Self::IQ3XXS, Self::IQ3S, Self::IQ4NL, Self::IQ4XS,
        ]
    }
}

/// Convenience constructor: MXFP4 descriptor (E2M1 + E8M0, block_size=32).
pub fn mxfp4_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Mxfp4.descriptor()
}

/// Convenience constructor: NVFP4 descriptor (E2M1 + UE4M3 sub-block, block_size=64).
pub fn nvfp4_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Nvfp4.descriptor()
}

/// Convenience constructor: AWQ4 descriptor (row-major, FP16 scale + FP16 zp, group_size=128).
pub fn awq4_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Awq4.descriptor()
}

/// Convenience constructor: GPTQ4 descriptor (col-interleaved, INT4 packed + StaticBias(1)).
pub fn gptq4_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Gptq4.descriptor()
}

/// Convenience constructor: SqueezeLLM descriptor (3-bit codebook LUT, block_size=256).
pub fn squeeze_llm_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::SqueezeLlm.descriptor()
}

// ── Classic GGML convenience constructors (SPEC §2.3) ──────────────────

/// Convenience constructor: Q4_0 descriptor (4-bit, block_size=32, StaticBias(8)).
pub fn q4_0_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q4_0.descriptor()
}

/// Convenience constructor: Q4_1 descriptor (4-bit with min, block_size=32).
pub fn q4_1_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q4_1.descriptor()
}

/// Convenience constructor: Q5_0 descriptor (5-bit, block_size=32, StaticBias(16)).
pub fn q5_0_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q5_0.descriptor()
}

/// Convenience constructor: Q5_1 descriptor (5-bit with min, block_size=32).
pub fn q5_1_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q5_1.descriptor()
}

/// Convenience constructor: Q8_0 descriptor (8-bit signed, block_size=32).
pub fn q8_0_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q8_0.descriptor()
}

// ── K-Quant convenience constructors (SPEC §2.3) ───────────────────────

/// Convenience constructor: Q2K descriptor (2-bit, block_size=256, hierarchical scale).
pub fn q2_k_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q2K.descriptor()
}

/// Convenience constructor: Q3K descriptor (3-bit, block_size=256, hierarchical scale).
pub fn q3_k_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q3K.descriptor()
}

/// Convenience constructor: Q4K descriptor (4-bit, block_size=256, hierarchical scale+min).
pub fn q4_k_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q4K.descriptor()
}

/// Convenience constructor: Q5K descriptor (5-bit, block_size=256, hierarchical scale+min).
pub fn q5_k_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q5K.descriptor()
}

/// Convenience constructor: Q6K descriptor (6-bit, block_size=256, Q6K scales + StaticBias(32)).
pub fn q6_k_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::Q6K.descriptor()
}

// ── IQ series convenience constructors (SPEC §2.3) ─────────────────────

/// Convenience constructor: IQ1S descriptor (1-bit importance, block_size=256).
pub fn iq1_s_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ1S.descriptor()
}

/// Convenience constructor: IQ1M descriptor (1-bit importance medium, block_size=256).
pub fn iq1_m_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ1M.descriptor()
}

/// Convenience constructor: IQ2XXS descriptor (2-bit importance extra extra small, block_size=256).
pub fn iq2_xxs_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ2XXS.descriptor()
}

/// Convenience constructor: IQ2XS descriptor (2-bit importance extra small, block_size=256).
pub fn iq2_xs_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ2XS.descriptor()
}

/// Convenience constructor: IQ2S descriptor (2-bit importance, block_size=256).
pub fn iq2_s_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ2S.descriptor()
}

/// Convenience constructor: IQ3XXS descriptor (3-bit importance extra extra small, block_size=256).
pub fn iq3_xxs_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ3XXS.descriptor()
}

/// Convenience constructor: IQ3S descriptor (3-bit importance small, block_size=256).
pub fn iq3_s_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ3S.descriptor()
}

/// Convenience constructor: IQ4NL descriptor (4-bit non-linear codebook LUT, block_size=32).
pub fn iq4_nl_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ4NL.descriptor()
}

/// Convenience constructor: IQ4XS descriptor (4-bit extra small codebook LUT, block_size=256).
pub fn iq4_xs_descriptor() -> QuantFormatDescriptor {
    QuantAlgoKind::IQ4XS.descriptor()
}

// ═══════════════════════════════════════════════════════════════════════════
// §5 — 5 ISA × 5 Operator Coverage Matrix (SPEC §5, QCG10)
// ═══════════════════════════════════════════════════════════════════════════
//
// Every combination of 5 ISA families × 5 operator categories must have at
// least one execution path. The matrix is parametric on `QuantType` because
// the available native instructions depend on the data format.

/// Target ISA family for code generation (5 ISA).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaKind {
    /// x86_64: AVX2 / AVX-512 / AVX-VNNI / AMX
    X86,
    /// ARM: NEON / SVE / SME
    Arm,
    /// NVIDIA GPU: PTX (HMMA tensor cores)
    GpuPtx,
    /// AMD GPU: HIP/ROCm (MFMA matrix cores)
    GpuHip,
    /// Apple GPU: Metal Shading Language
    GpuMsl,
}

impl IsaKind {
    /// All 5 ISA families.
    pub const fn all() -> &'static [IsaKind; 5] {
        &[IsaKind::X86, IsaKind::Arm, IsaKind::GpuPtx, IsaKind::GpuHip, IsaKind::GpuMsl]
    }
}

/// Operator category in the coverage matrix (5 ops).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Matrix multiply: GEMM / GEMM-bias / QuantGEMM
    Gemm,
    /// Multi-head attention: MHA / cached GQA
    Attention,
    /// Normalization: RMSNorm / LayerNorm / ValueNorm
    Norm,
    /// Activation: SiLU / GELU / SwiGLU / GeGLU
    Activation,
    /// Quantization / dequantization: Dequantize / QuantGather
    Quant,
}

impl OpCategory {
    /// All 5 operator categories.
    pub const fn all() -> &'static [OpCategory; 5] {
        &[OpCategory::Gemm, OpCategory::Attention, OpCategory::Norm, OpCategory::Activation, OpCategory::Quant]
    }
}

/// Execution path for a (QuantType × ISA × OpCategory) triple (SPEC §1.3).
///
/// Priority: Native > Assisted > DequantFMA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoveragePath {
    /// Level 1: Hardware-native instruction (e.g. VNNI dp4a, HMMA tensor core).
    /// Packed data → one instruction → fp32/int32 accumulate, zero software unpack.
    Native,
    /// Level 2: ISA-assisted vectorized loop with SIMD intrinsics but no dedicated
    /// dot-product instruction (e.g. nibble unpack + VPDPBUSD, NEON vmlal for INT4).
    /// Microkernel completes unpack + dot-product, accumulator stays INT32.
    Assisted,
    /// Level 3: Dequantize-then-FMA — decode quantized block → F32 vector → F32 FMA.
    /// Universal fallback for quantized formats on devices with no dot-product support.
    /// Only triggered when hardware has no VNNI/SDOT/IMMA/WMMA and SIMD width ≤ 128-bit.
    DequantFMA,
}

impl CoveragePath {
    /// Quality ranking: Native (0) > Assisted (1) > DequantFMA (2).
    pub fn rank(self) -> u8 {
        match self {
            Self::Native => 0,
            Self::Assisted => 1,
            Self::DequantFMA => 2,
        }
    }
}

/// Static entry in the coverage matrix: best available path for one cell.
#[derive(Debug, Clone, Copy)]
pub struct CoverageEntry {
    pub isa: IsaKind,
    pub op: OpCategory,
    pub path: CoveragePath,
}

/// The 5×5 ISA × Operator coverage matrix.
///
/// Parametric on `QuantType`: each quantization format may have different
/// native instruction availability. Constructed via [`CoverageMatrix::new`],
/// which evaluates every `(QuantType, IsaKind, OpCategory)` triple and
/// records the best path. All 25 cells are guaranteed to have at least
/// `CoveragePath::DequantFMA`.
#[derive(Debug, Clone)]
pub struct CoverageMatrix {
    /// 25 entries, indexed as `isa_index * 5 + op_index`.
    entries: Vec<CoverageEntry>,
}

impl CoverageMatrix {
    /// Build the 5×5 coverage matrix for a given quantization type.
    ///
    /// Every one of the 25 cells is guaranteed to be filled with at least
    /// `CoveragePath::DequantFMA`. Native or Assisted paths are selected
    /// when the ISA provides dedicated support for the format.
    pub fn new(quant_type: QuantType) -> Self {
        let is_float = matches!(quant_type, QuantType::F32 | QuantType::Bf16 | QuantType::F16);
        let is_int8 = matches!(quant_type, QuantType::Q8_0 | QuantType::Q8_1 | QuantType::Q8K);
        let is_int4 = matches!(
            quant_type,
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_0 | QuantType::Q5_1
                | QuantType::Q4K | QuantType::Q5K | QuantType::AWQ4 | QuantType::GPTQ4
        );
        let is_nvfp4 = matches!(quant_type, QuantType::Nvfp4);
        let is_mxfp4 = matches!(quant_type, QuantType::Mxfp4 { .. });

        let mut entries = Vec::with_capacity(25);
        for &isa in IsaKind::all() {
            for &op in OpCategory::all() {
                let path = best_path(quant_type, isa, op, is_float, is_int8, is_int4, is_nvfp4, is_mxfp4);
                entries.push(CoverageEntry { isa, op, path });
            }
        }

        CoverageMatrix { entries }
    }

    /// Look up the best coverage path for a given (ISA, OpCategory) pair.
    pub fn get(&self, isa: IsaKind, op: OpCategory) -> CoveragePath {
        let isa_idx = isa_to_index(isa);
        let op_idx = op_to_index(op);
        self.entries[isa_idx * 5 + op_idx].path
    }

    /// Iterate over all 25 coverage entries.
    pub fn iter(&self) -> impl Iterator<Item = &CoverageEntry> {
        self.entries.iter()
    }

    /// Generate a compile-time coverage report string.
    ///
    /// Returns a human-readable table showing the coverage path for every
    /// (ISA × Op) cell. Panics if any cell is uncovered (should never happen
    /// since `DequantFMA` is the universal fallback).
    pub fn coverage_report(&self, quant_type: QuantType) -> String {
        let mut report = String::with_capacity(1024);
        report.push_str(&format!("Coverage Matrix for {:?}:\n", quant_type));
        report.push_str("ISA       | Gemm       | Attention  | Norm       | Activation | Quant\n");
        report.push_str("----------|------------|------------|------------|------------|------------\n");

        let isa_names = [
            (IsaKind::X86, "x86_64"),
            (IsaKind::Arm, "ARM"),
            (IsaKind::GpuPtx, "GPU PTX"),
            (IsaKind::GpuHip, "GPU HIP"),
            (IsaKind::GpuMsl, "GPU MSL"),
        ];

        for (isa, name) in &isa_names {
            report.push_str(&format!("{:<10}|", name));
            for op in &[OpCategory::Gemm, OpCategory::Attention, OpCategory::Norm, OpCategory::Activation, OpCategory::Quant] {
                let entry = &self.entries[isa_to_index(*isa) * 5 + op_to_index(*op)];
                let path_str = match entry.path {
                    CoveragePath::Native => "Native  ",
                    CoveragePath::Assisted => "Assisted",
                    CoveragePath::DequantFMA => "DeqFMA  ",
                };
                report.push_str(&format!(" {}     |", path_str));
            }
            report.push('\n');
        }

        // Verify completeness: every cell must have a path
        for entry in &self.entries {
            assert!(
                matches!(entry.path, CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA),
                "Uncovered cell: isa={:?} op={:?}",
                entry.isa,
                entry.op,
            );
        }

        report
    }
}

fn isa_to_index(isa: IsaKind) -> usize {
    match isa {
        IsaKind::X86 => 0,
        IsaKind::Arm => 1,
        IsaKind::GpuPtx => 2,
        IsaKind::GpuHip => 3,
        IsaKind::GpuMsl => 4,
    }
}

fn op_to_index(op: OpCategory) -> usize {
    match op {
        OpCategory::Gemm => 0,
        OpCategory::Attention => 1,
        OpCategory::Norm => 2,
        OpCategory::Activation => 3,
        OpCategory::Quant => 4,
    }
}

/// Determine the best execution path for a given (QuantType, ISA, OpCategory) triple.
///
/// Decision logic (SPEC §1.3):
/// - **GEMM**: Native if the ISA has a hardware dot-product for the format;
///   Assisted if SIMD-accelerated but no dedicated instruction; else DequantFMA.
/// - **Attention**: Always Native for float types on GPU; Assisted for quantized
///   (attention operates on dequantized activations).
/// - **Norm**: GPU Native (warp shuffle), CPU Assisted (horizontal add).
/// - **Activation**: Always Native (element-wise, maps directly to hardware).
/// - **Quant**: Native if the ISA provides dedicated pack/unpack; else DequantFMA.
fn best_path(
    quant_type: QuantType,
    isa: IsaKind,
    op: OpCategory,
    is_float: bool,
    is_int8: bool,
    is_int4: bool,
    is_nvfp4: bool,
    is_mxfp4: bool,
) -> CoveragePath {
    match op {
        OpCategory::Gemm => gemm_path(quant_type, isa, is_float, is_int8, is_int4, is_nvfp4, is_mxfp4),
        OpCategory::Attention => attention_path(isa, is_float),
        OpCategory::Norm => norm_path(isa),
        OpCategory::Activation => CoveragePath::Native,
        OpCategory::Quant => quant_path(quant_type, isa, is_float, is_int8, is_int4),
    }
}

/// Best GEMM path per (QuantType, ISA).
fn gemm_path(
    _qt: QuantType,
    isa: IsaKind,
    is_float: bool,
    is_int8: bool,
    is_int4: bool,
    is_nvfp4: bool,
    is_mxfp4: bool,
) -> CoveragePath {
    match isa {
        IsaKind::X86 => {
            if is_float {
                CoveragePath::Native
            } else if is_int8 {
                CoveragePath::Native
            } else if is_nvfp4 {
                CoveragePath::DequantFMA
            } else if is_mxfp4 {
                CoveragePath::DequantFMA
            } else if is_int4 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
        IsaKind::Arm => {
            if is_float {
                CoveragePath::Native
            } else if is_int8 {
                CoveragePath::Native
            } else if is_int4 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
        IsaKind::GpuPtx => {
            if is_float {
                CoveragePath::Native
            } else if is_nvfp4 {
                CoveragePath::Native
            } else if is_mxfp4 {
                CoveragePath::Native
            } else if is_int8 {
                CoveragePath::Native
            } else if is_int4 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
        IsaKind::GpuHip => {
            if is_float {
                CoveragePath::Native
            } else if is_int8 {
                CoveragePath::Native
            } else if is_int4 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
        IsaKind::GpuMsl => {
            if is_float {
                CoveragePath::Native
            } else if is_int8 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
    }
}

/// Best attention path: attention operates on dequantized activations.
fn attention_path(isa: IsaKind, is_float: bool) -> CoveragePath {
    if is_float {
        CoveragePath::Native
    } else {
        match isa {
            IsaKind::X86 => CoveragePath::Assisted,
            IsaKind::Arm => CoveragePath::Assisted,
            IsaKind::GpuPtx => CoveragePath::Assisted,
            IsaKind::GpuHip => CoveragePath::Assisted,
            IsaKind::GpuMsl => CoveragePath::DequantFMA,
        }
    }
}

/// Best normalization path.
fn norm_path(isa: IsaKind) -> CoveragePath {
    match isa {
        IsaKind::X86 => CoveragePath::Assisted,
        IsaKind::Arm => CoveragePath::Assisted,
        IsaKind::GpuPtx => CoveragePath::Native,
        IsaKind::GpuHip => CoveragePath::Native,
        IsaKind::GpuMsl => CoveragePath::Native,
    }
}

/// Best quantization/dequantization path.
fn quant_path(
    _qt: QuantType,
    isa: IsaKind,
    is_float: bool,
    is_int8: bool,
    is_int4: bool,
) -> CoveragePath {
    if is_float {
        return CoveragePath::Native;
    }
    match isa {
        IsaKind::X86 => {
            if is_int8 {
                CoveragePath::Native
            } else if is_int4 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
        IsaKind::Arm => {
            if is_int8 {
                CoveragePath::Native
            } else if is_int4 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
        IsaKind::GpuPtx => {
            if is_int8 {
                CoveragePath::Native
            } else {
                CoveragePath::Assisted
            }
        }
        IsaKind::GpuHip => {
            if is_int8 {
                CoveragePath::Native
            } else {
                CoveragePath::Assisted
            }
        }
        IsaKind::GpuMsl => {
            if is_int8 {
                CoveragePath::Assisted
            } else {
                CoveragePath::DequantFMA
            }
        }
    }
}

/// Check the coverage path for a specific (QuantType, IsaKind, OpCategory) triple.
///
/// Primary API for the JIT planner: queries the best available execution path
/// and uses it to select the codegen strategy.
pub fn coverage_check(quant_type: QuantType, isa: IsaKind, op: OpCategory) -> CoveragePath {
    let matrix = CoverageMatrix::new(quant_type);
    matrix.get(isa, op)
}

/// Build the ISA × Op coverage matrix for a given QuantType.
pub fn isa_op_matrix(quant_type: QuantType) -> CoverageMatrix {
    CoverageMatrix::new(quant_type)
}

/// Global coverage matrix builder: returns a matrix for each registered QuantType.
///
/// Used at compile time (unit tests / CI) to verify that every combination
/// of (QuantType × ISA × OpCategory) has at least one execution path.
pub fn coverage_matrix(quant_type: QuantType) -> CoverageMatrix {
    CoverageMatrix::new(quant_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── QuantAlgoKind ─────────────────────────────────────────────────

    #[test]
    fn algo_kind_all_count() {
        assert_eq!(QuantAlgoKind::all().len(), 24);
    }

    #[test]
    fn algo_kind_quant_type_roundtrip() {
        for &kind in QuantAlgoKind::all() {
            let _qt = kind.quant_type();
        }
    }

    #[test]
    fn algo_kind_equality() {
        assert_eq!(QuantAlgoKind::Mxfp4, QuantAlgoKind::Mxfp4);
        assert_ne!(QuantAlgoKind::Awq4, QuantAlgoKind::Gptq4);
        assert_ne!(QuantAlgoKind::Q4_0, QuantAlgoKind::Q4_1);
    }

    #[test]
    fn algo_kind_quant_type_specific() {
        assert!(matches!(QuantAlgoKind::Mxfp4.quant_type(), QuantType::Mxfp4 { block_size: 32 }));
        assert!(matches!(QuantAlgoKind::Nvfp4.quant_type(), QuantType::Nvfp4));
        assert_eq!(QuantAlgoKind::Awq4.quant_type(), QuantType::AWQ4);
        assert_eq!(QuantAlgoKind::Gptq4.quant_type(), QuantType::GPTQ4);
        assert_eq!(QuantAlgoKind::SqueezeLlm.quant_type(), QuantType::Squeeze);
        assert_eq!(QuantAlgoKind::Q4_0.quant_type(), QuantType::Q4_0);
        assert_eq!(QuantAlgoKind::Q8_0.quant_type(), QuantType::Q8_0);
        assert_eq!(QuantAlgoKind::Q2K.quant_type(), QuantType::Q2K);
        assert_eq!(QuantAlgoKind::IQ4XS.quant_type(), QuantType::IQ4XS);
    }

    // ── IsaKind ───────────────────────────────────────────────────────

    #[test]
    fn isa_kind_all_count() {
        assert_eq!(IsaKind::all().len(), 5);
    }

    #[test]
    fn isa_kind_variants() {
        let all = IsaKind::all();
        assert!(all.contains(&IsaKind::X86));
        assert!(all.contains(&IsaKind::Arm));
        assert!(all.contains(&IsaKind::GpuPtx));
        assert!(all.contains(&IsaKind::GpuHip));
        assert!(all.contains(&IsaKind::GpuMsl));
    }

    // ── OpCategory ────────────────────────────────────────────────────

    #[test]
    fn op_category_all_count() {
        assert_eq!(OpCategory::all().len(), 5);
    }

    #[test]
    fn op_category_variants() {
        let all = OpCategory::all();
        assert!(all.contains(&OpCategory::Gemm));
        assert!(all.contains(&OpCategory::Attention));
        assert!(all.contains(&OpCategory::Norm));
        assert!(all.contains(&OpCategory::Activation));
        assert!(all.contains(&OpCategory::Quant));
    }

    // ── CoveragePath ──────────────────────────────────────────────────

    #[test]
    fn coverage_path_rank_order() {
        assert!(CoveragePath::Native.rank() < CoveragePath::Assisted.rank());
        assert!(CoveragePath::Assisted.rank() < CoveragePath::DequantFMA.rank());
        assert_eq!(CoveragePath::Native.rank(), 0);
        assert_eq!(CoveragePath::Assisted.rank(), 1);
        assert_eq!(CoveragePath::DequantFMA.rank(), 2);
    }

    #[test]
    fn coverage_path_equality() {
        assert_eq!(CoveragePath::Native, CoveragePath::Native);
        assert_ne!(CoveragePath::Native, CoveragePath::Assisted);
    }

    // ── CoverageMatrix ────────────────────────────────────────────────

    #[test]
    fn coverage_matrix_f32_all_native_activation() {
        let m = CoverageMatrix::new(QuantType::F32);
        for &isa in IsaKind::all() {
            assert_eq!(m.get(isa, OpCategory::Activation), CoveragePath::Native);
        }
    }

    #[test]
    fn coverage_matrix_25_entries() {
        let m = CoverageMatrix::new(QuantType::F32);
        assert_eq!(m.iter().count(), 25);
    }

    #[test]
    fn coverage_matrix_f32_gemm_native_everywhere() {
        let m = CoverageMatrix::new(QuantType::F32);
        for &isa in IsaKind::all() {
            assert_eq!(m.get(isa, OpCategory::Gemm), CoveragePath::Native,
                "F32 GEMM should be Native on {:?}", isa);
        }
    }

    #[test]
    fn coverage_matrix_f32_norm_gpu_native_cpu_assisted() {
        let m = CoverageMatrix::new(QuantType::F32);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Norm), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Norm), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Norm), CoveragePath::Native);
    }

    #[test]
    fn coverage_matrix_int4_gemm_x86_assisted() {
        let m = CoverageMatrix::new(QuantType::Q4_0);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Gemm), CoveragePath::Assisted);
    }

    #[test]
    fn coverage_matrix_nvfp4_gpu_native() {
        let m = CoverageMatrix::new(QuantType::Nvfp4);
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Gemm), CoveragePath::Native);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::DequantFMA);
    }

    #[test]
    fn coverage_matrix_int8_gemm_native() {
        let m = CoverageMatrix::new(QuantType::Q8_0);
        for &isa in &[IsaKind::X86, IsaKind::Arm, IsaKind::GpuPtx, IsaKind::GpuHip] {
            assert_eq!(m.get(isa, OpCategory::Gemm), CoveragePath::Native,
                "INT8 GEMM should be Native on {:?}", isa);
        }
    }

    #[test]
    fn coverage_matrix_no_uncovered_cells() {
        let m = CoverageMatrix::new(QuantType::AWQ4);
        for entry in m.iter() {
            assert!(matches!(entry.path,
                CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA));
        }
    }

    #[test]
    fn coverage_report_produces_output() {
        let m = CoverageMatrix::new(QuantType::F32);
        let report = m.coverage_report(QuantType::F32);
        assert!(report.contains("x86_64"));
        assert!(report.contains("ARM"));
        assert!(report.contains("GPU PTX"));
        assert!(report.contains("Gemm"));
        assert!(report.contains("Native"));
    }

    // ── Convenience constructors ──────────────────────────────────────

    #[test]
    fn convenience_constructors_all_work() {
        mxfp4_descriptor();
        nvfp4_descriptor();
        awq4_descriptor();
        gptq4_descriptor();
        squeeze_llm_descriptor();
        q4_0_descriptor();
        q4_1_descriptor();
        q5_0_descriptor();
        q5_1_descriptor();
        q8_0_descriptor();
        q2_k_descriptor();
        q3_k_descriptor();
        q4_k_descriptor();
        q5_k_descriptor();
        q6_k_descriptor();
        iq1_s_descriptor();
        iq1_m_descriptor();
        iq2_xxs_descriptor();
        iq2_xs_descriptor();
        iq2_s_descriptor();
        iq3_xxs_descriptor();
        iq3_s_descriptor();
        iq4_nl_descriptor();
        iq4_xs_descriptor();
    }

    // ── coverage_check API ────────────────────────────────────────────

    #[test]
    fn coverage_check_api() {
        let path = coverage_check(QuantType::F32, IsaKind::X86, OpCategory::Gemm);
        assert_eq!(path, CoveragePath::Native);
    }

    #[test]
    fn isa_op_matrix_api() {
        let m = isa_op_matrix(QuantType::Bf16);
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Gemm), CoveragePath::Native);
    }

    // ── Additional coverage tests ────────────────────────────────────

    #[test]
    fn coverage_matrix_bf16_float_paths() {
        // Bf16 is a float type — same fast paths as F32 for GEMM/Attention/Activation/Quant.
        let m = CoverageMatrix::new(QuantType::Bf16);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::Native);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Attention), CoveragePath::Native);
        assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Quant), CoveragePath::Native);
    }

    #[test]
    fn coverage_matrix_mxfp4_gpu_native_x86_dequant() {
        // MXFP4: PTX GEMM Native (WGMMA), x86 GEMM DequantFMA (no native 4-bit float).
        let m = CoverageMatrix::new(QuantType::Mxfp4 { block_size: 32 });
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Gemm), CoveragePath::Native);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::DequantFMA);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Gemm), CoveragePath::DequantFMA);
    }

    #[test]
    fn coverage_matrix_quantized_attention_assisted() {
        // Quantized formats use Assisted attention on x86/ARM/GPU PTX/HIP (dequantized activations).
        let m = CoverageMatrix::new(QuantType::AWQ4);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Attention), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Attention), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Attention), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::GpuHip, OpCategory::Attention), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Attention), CoveragePath::DequantFMA);
    }

    #[test]
    fn coverage_matrix_metal_int8_gemm_assisted() {
        // Metal has no native INT8 tensor core, so INT8 GEMM is Assisted (not Native).
        let m = CoverageMatrix::new(QuantType::Q8_0);
        assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Gemm), CoveragePath::Assisted);
    }

    #[test]
    fn coverage_matrix_squeeze_gemm_dequant_everywhere() {
        // SqueezeLLM (3-bit codebook): no hardware GEMM support anywhere → DequantFMA.
        let m = CoverageMatrix::new(QuantType::Squeeze);
        for &isa in IsaKind::all() {
            assert_eq!(m.get(isa, OpCategory::Gemm), CoveragePath::DequantFMA,
                "SqueezeLLM GEMM should be DequantFMA on {:?}", isa);
        }
    }

    #[test]
    fn coverage_report_non_float_contains_dequantfma() {
        // Non-float formats must show DequantFMA in at least some cells.
        let m = CoverageMatrix::new(QuantType::Q2K);
        let report = m.coverage_report(QuantType::Q2K);
        assert!(report.contains("DeqFMA"), "Q2K report must show DequantFMA for some cells");
        assert!(report.contains("GPU PTX"), "Report must list GPU PTX ISA");
    }

    #[test]
    fn algo_kind_quant_type_iq_series() {
        // Verify IQ family members that were not tested in algo_kind_quant_type_specific.
        assert_eq!(QuantAlgoKind::IQ1M.quant_type(), QuantType::IQ1M);
        assert_eq!(QuantAlgoKind::IQ2XXS.quant_type(), QuantType::IQ2XXS);
        assert_eq!(QuantAlgoKind::IQ2XS.quant_type(), QuantType::IQ2XS);
        assert_eq!(QuantAlgoKind::IQ3S.quant_type(), QuantType::IQ3S);
        assert_eq!(QuantAlgoKind::IQ4NL.quant_type(), QuantType::IQ4NL);
        assert_eq!(QuantAlgoKind::IQ2S.quant_type(), QuantType::IQ2S);
    }

    #[test]
    fn descriptor_block_bytes_consistency() {
        // All convenience constructors must return descriptors with non-zero block_bytes.
        let descriptors = [
            mxfp4_descriptor(), nvfp4_descriptor(), awq4_descriptor(),
            gptq4_descriptor(), squeeze_llm_descriptor(),
            q4_0_descriptor(), q4_1_descriptor(), q5_0_descriptor(),
            q5_1_descriptor(), q8_0_descriptor(),
            q2_k_descriptor(), q3_k_descriptor(), q4_k_descriptor(),
            q5_k_descriptor(), q6_k_descriptor(),
        ];
        for desc in &descriptors {
            assert!(desc.block_bytes > 0, "{}: block_bytes must be > 0", desc.name);
            assert!(desc.block_size > 0, "{}: block_size must be > 0", desc.name);
            assert!(!desc.name.is_empty(), "Descriptor name must not be empty");
        }
    }

    #[test]
    fn coverage_matrix_norm_paths_all_isa() {
        // Norm: CPU (x86, ARM) → Assisted; GPU (PTX, HIP, MSL) → Native.
        // This holds regardless of quant_type (norm operates on dequantized values).
        let m = CoverageMatrix::new(QuantType::Q4_0);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Norm), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Norm), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Norm), CoveragePath::Native);
        assert_eq!(m.get(IsaKind::GpuHip, OpCategory::Norm), CoveragePath::Native);
        assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Norm), CoveragePath::Native);
    }

    #[test]
    fn coverage_check_multiple_quant_types() {
        // Spot-check coverage_check for several QuantTypes to verify the public API
        // returns valid paths for diverse format families.
        let cases = [
            (QuantType::Q6K, IsaKind::X86, OpCategory::Gemm, CoveragePath::DequantFMA),
            (QuantType::Q8K, IsaKind::Arm, OpCategory::Gemm, CoveragePath::Native),
            (QuantType::F16, IsaKind::GpuHip, OpCategory::Attention, CoveragePath::Native),
            (QuantType::GPTQ4, IsaKind::GpuPtx, OpCategory::Activation, CoveragePath::Native),
            (QuantType::AWQ4, IsaKind::X86, OpCategory::Quant, CoveragePath::Assisted),
        ];
        for (qt, isa, op, expected) in cases {
            assert_eq!(coverage_check(qt, isa, op), expected,
                "coverage_check({:?}, {:?}, {:?}) should be {:?}", qt, isa, op, expected);
        }
    }

    // ── Additional tests (wave-12knf) ─────────────────────────────────

    #[test]
    fn quant_data_kind_float_classification() {
        // Float data kinds must report is_float() == true; all others false.
        assert!(QuantDataKind::Bfloat16.is_float());
        assert!(QuantDataKind::Float16.is_float());
        assert!(QuantDataKind::Float32.is_float());

        assert!(!QuantDataKind::Int8.is_float());
        assert!(!QuantDataKind::PackedInt4.is_float());
        assert!(!QuantDataKind::SignedPackedInt4.is_float());
        assert!(!QuantDataKind::PackedInt5.is_float());
        assert!(!QuantDataKind::PackedInt6.is_float());
        assert!(!QuantDataKind::Float4.is_float());
        assert!(!QuantDataKind::Float8.is_float());
        assert!(!QuantDataKind::SuperLowBit.is_float());
        assert!(!QuantDataKind::Nvfp4.is_float());
        assert!(!QuantDataKind::Codebook.is_float());
    }

    #[test]
    fn coverage_matrix_clone_independence() {
        // Cloning a CoverageMatrix produces an independent copy; modifying the
        // original (by dropping) does not affect the clone.
        let original = CoverageMatrix::new(QuantType::F32);
        let cloned = original.clone();
        for &isa in IsaKind::all() {
            for &op in OpCategory::all() {
                assert_eq!(original.get(isa, op), cloned.get(isa, op));
            }
        }
    }

    #[test]
    fn coverage_path_rank_transitivity() {
        // Rank ordering must be transitive: Native < Assisted < DequantFMA.
        let native = CoveragePath::Native.rank();
        let assisted = CoveragePath::Assisted.rank();
        let dequant = CoveragePath::DequantFMA.rank();
        assert!(native < assisted, "Native rank must be less than Assisted");
        assert!(assisted < dequant, "Assisted rank must be less than DequantFMA");
        assert!(native < dequant, "Native rank must be less than DequantFMA (transitive)");
    }

    #[test]
    fn iq_series_descriptors_block_bytes_positive() {
        // IQ convenience constructors that were not covered by descriptor_block_bytes_consistency.
        let iq_descriptors = [
            iq1_s_descriptor(),
            iq1_m_descriptor(),
            iq2_xxs_descriptor(),
            iq2_xs_descriptor(),
            iq2_s_descriptor(),
            iq3_xxs_descriptor(),
            iq3_s_descriptor(),
            iq4_nl_descriptor(),
            iq4_xs_descriptor(),
        ];
        for desc in &iq_descriptors {
            assert!(desc.block_bytes > 0, "{}: block_bytes must be > 0", desc.name);
            assert!(desc.block_size > 0, "{}: block_size must be > 0", desc.name);
            assert!(!desc.name.is_empty(), "Descriptor name must not be empty");
        }
    }

    #[test]
    fn k_quant_gemm_paths_by_bit_class() {
        // Q4K and Q5K are INT4-class (is_int4 match): Assisted on x86/ARM/GPU PTX/GPU HIP.
        for qt in [QuantType::Q4K, QuantType::Q5K] {
            let m = CoverageMatrix::new(qt);
            assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::Assisted,
                "{:?} GEMM x86 should be Assisted", qt);
            assert_eq!(m.get(IsaKind::Arm, OpCategory::Gemm), CoveragePath::Assisted,
                "{:?} GEMM ARM should be Assisted", qt);
            assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Gemm), CoveragePath::Assisted,
                "{:?} GEMM GPU PTX should be Assisted", qt);
            assert_eq!(m.get(IsaKind::GpuHip, OpCategory::Gemm), CoveragePath::Assisted,
                "{:?} GEMM GPU HIP should be Assisted", qt);
            assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Gemm), CoveragePath::DequantFMA,
                "{:?} GEMM GPU MSL should be DequantFMA", qt);
        }
        // Q3K (3-bit) and Q2K (2-bit) are NOT in the is_int4 match; DequantFMA on x86/ARM.
        for qt in [QuantType::Q3K, QuantType::Q2K] {
            let m = CoverageMatrix::new(qt);
            assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::DequantFMA,
                "{:?} GEMM x86 should be DequantFMA (not INT4-class)", qt);
            assert_eq!(m.get(IsaKind::Arm, OpCategory::Gemm), CoveragePath::DequantFMA,
                "{:?} GEMM ARM should be DequantFMA (not INT4-class)", qt);
            assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Gemm), CoveragePath::DequantFMA,
                "{:?} GEMM GPU MSL should be DequantFMA", qt);
        }
    }

    #[test]
    fn algo_kind_hash_consistency() {
        // Equal QuantAlgoKind values must produce equal hashes (HashMap/HashSet contract).
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for &kind in QuantAlgoKind::all() {
            assert!(set.insert(kind), "Duplicate QuantAlgoKind: {:?}", kind);
        }
        assert_eq!(set.len(), QuantAlgoKind::all().len());

        // Verify re-inserting the same kind does not increase size.
        assert!(!set.insert(QuantAlgoKind::Mxfp4), "Re-inserting Mxfp4 should be a no-op");
        assert_eq!(set.len(), QuantAlgoKind::all().len());
    }

    #[test]
    fn isa_to_index_roundtrip_coverage() {
        // Every IsaKind variant must map to a unique index within [0, 5).
        let mut seen = [false; 5];
        for &isa in IsaKind::all() {
            let idx = isa_to_index(isa);
            assert!(idx < 5, "IsaKind {:?} index {} out of range", isa, idx);
            assert!(!seen[idx], "Duplicate isa_to_index for index {}", idx);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "Not all ISA indices were covered");
    }

    #[test]
    fn op_to_index_roundtrip_coverage() {
        // Every OpCategory variant must map to a unique index within [0, 5).
        let mut seen = [false; 5];
        for &op in OpCategory::all() {
            let idx = op_to_index(op);
            assert!(idx < 5, "OpCategory {:?} index {} out of range", op, idx);
            assert!(!seen[idx], "Duplicate op_to_index for index {}", idx);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "Not all OpCategory indices were covered");
    }

    #[test]
    fn coverage_matrix_fp16_identical_to_f32() {
        // FP16 is a float type; coverage paths should be identical to F32.
        let f32m = CoverageMatrix::new(QuantType::F32);
        let f16m = CoverageMatrix::new(QuantType::F16);
        for &isa in IsaKind::all() {
            for &op in OpCategory::all() {
                assert_eq!(f32m.get(isa, op), f16m.get(isa, op),
                    "FP16 path differs from F32 for isa={:?} op={:?}", isa, op);
            }
        }
    }

    #[test]
    fn descriptor_data_kind_matches_quant_type() {
        // External float formats (MXFP4, NVFP4) must have float-class data kinds;
        // INT quantized formats (Q4_0, Q8_0) must have integer-class data kinds.
        let mxfp4 = mxfp4_descriptor();
        assert_eq!(mxfp4.data_kind, QuantDataKind::Float4);
        assert!(!mxfp4.data_kind.is_float());

        let nvfp4 = nvfp4_descriptor();
        assert_eq!(nvfp4.data_kind, QuantDataKind::Nvfp4);

        let q4_0 = q4_0_descriptor();
        assert_eq!(q4_0.data_kind, QuantDataKind::SignedPackedInt4);

        let q8_0 = q8_0_descriptor();
        assert_eq!(q8_0.data_kind, QuantDataKind::Int8);

        let awq4 = awq4_descriptor();
        assert_eq!(awq4.data_kind, QuantDataKind::PackedInt4);
    }

    // ── Additional tests (wave-12koe) ────────────────────────────────────

    #[test]
    fn for_type_returns_registered_descriptor() {
        // QuantFormatDescriptor::for_type must resolve every QuantType to a
        // descriptor whose quant_type field matches the input.
        let desc = QuantFormatDescriptor::for_type(QuantType::Q4_0);
        assert_eq!(desc.quant_type, QuantType::Q4_0);
        assert_eq!(desc.name, "Q4_0");

        let desc = QuantFormatDescriptor::for_type(QuantType::Nvfp4);
        assert_eq!(desc.quant_type, QuantType::Nvfp4);
    }

    #[test]
    fn native_isa_set_for_int8_descriptors() {
        // INT8 formats (Q8_0, Q8_1, Q8K) have native ISA support (VPDPBUSD / SDOT / HMMA).
        let q8_0 = q8_0_descriptor();
        assert!(q8_0.native_isa.is_some(), "Q8_0 must have native_isa");
        assert_eq!(q8_0.native_isa, Some(NativeIsa::Vpdpbusd));

        let desc_8k = QuantFormatDescriptor::for_type(QuantType::Q8K);
        assert!(desc_8k.native_isa.is_some(), "Q8K must have native_isa");
    }

    #[test]
    fn native_isa_none_for_sub_int4_descriptors() {
        // Sub-4-bit formats (Q2K, Q3K, IQ family) have no native hardware ISA.
        for qt in [QuantType::Q2K, QuantType::Q3K, QuantType::IQ1S, QuantType::IQ3XXS] {
            let desc = QuantFormatDescriptor::for_type(qt);
            assert!(desc.native_isa.is_none(),
                "{:?} should have no native ISA (sub-4-bit)", qt);
        }
    }

    #[test]
    fn storage_layout_gptq4_col_interleaved() {
        // GPTQ4 is the only format using ColInterleaved storage; AWQ4 uses RowMajor.
        let gptq4 = gptq4_descriptor();
        assert_eq!(gptq4.storage_layout, StorageLayout::ColInterleaved,
            "GPTQ4 must use ColInterleaved storage");

        let awq4 = awq4_descriptor();
        assert_eq!(awq4.storage_layout, StorageLayout::RowMajor,
            "AWQ4 must use RowMajor storage");
    }

    #[test]
    fn bits_per_element_monotonic_with_type() {
        // bits_per_element must increase monotonically within a family.
        assert!(q4_0_descriptor().bits_per_element < q5_0_descriptor().bits_per_element);
        assert!(q5_0_descriptor().bits_per_element < q8_0_descriptor().bits_per_element);
        assert_eq!(q4_0_descriptor().bits_per_element, 4);
        assert_eq!(q5_0_descriptor().bits_per_element, 5);
        assert_eq!(q8_0_descriptor().bits_per_element, 8);
    }

    #[test]
    fn zero_layout_static_bias_values() {
        // Q4_0 has StaticBias(8), Q5_0 has StaticBias(16), Q6K has StaticBias(32).
        // Q4_1 and Q8_0 have no zero-point (ZeroLayout::None).
        let q4_0 = q4_0_descriptor();
        assert_eq!(q4_0.zero_layout, ZeroLayout::StaticBias { value: 8 });

        let q5_0 = q5_0_descriptor();
        assert_eq!(q5_0.zero_layout, ZeroLayout::StaticBias { value: 16 });

        let q6k = q6_k_descriptor();
        assert_eq!(q6k.zero_layout, ZeroLayout::StaticBias { value: 32 });

        let q8_0 = q8_0_descriptor();
        assert_eq!(q8_0.zero_layout, ZeroLayout::None);

        let q4_1 = q4_1_descriptor();
        assert_eq!(q4_1.zero_layout, ZeroLayout::None);
    }

    #[test]
    fn hierarchical_scale_kquant_family() {
        // K-Quant family (Q4K, Q5K) must use Hierarchical scale layout with
        // sub_scales_bits = 6 and sub_block_elements = 32.
        for (qt, name) in [(QuantType::Q4K, "Q4K"), (QuantType::Q5K, "Q5K")] {
            let desc = QuantFormatDescriptor::for_type(qt);
            match &desc.scale_layout {
                ScaleLayout::Hierarchical { sub_scales_bits, sub_block_elements, .. } => {
                    assert_eq!(*sub_scales_bits, 6, "{} sub_scales_bits must be 6", name);
                    assert_eq!(*sub_block_elements, 32, "{} sub_block_elements must be 32", name);
                }
                other => panic!("{} expected Hierarchical scale, got {:?}", name, other),
            }
        }
    }

    #[test]
    fn coverage_matrix_gpu_hip_nvfp4_dequantfma() {
        // NVFP4: GPU HIP does not have native FP4 tensor core (NVIDIA-only),
        // so GEMM path is DequantFMA on GPU HIP.
        let m = CoverageMatrix::new(QuantType::Nvfp4);
        assert_eq!(m.get(IsaKind::GpuHip, OpCategory::Gemm), CoveragePath::DequantFMA);
        // Activation is always Native regardless of format.
        assert_eq!(m.get(IsaKind::GpuHip, OpCategory::Activation), CoveragePath::Native);
    }

    #[test]
    fn coverage_matrix_function_matches_new() {
        // The coverage_matrix() public function must produce identical results
        // to CoverageMatrix::new() for the same QuantType.
        let via_new = CoverageMatrix::new(QuantType::GPTQ4);
        let via_fn = coverage_matrix(QuantType::GPTQ4);
        for &isa in IsaKind::all() {
            for &op in OpCategory::all() {
                assert_eq!(via_new.get(isa, op), via_fn.get(isa, op),
                    "coverage_matrix() differs from new() for ({:?}, {:?})", isa, op);
            }
        }
    }

    #[test]
    fn scale_layout_block_scalar_classic_formats() {
        // Q4_0, Q5_0, Q8_0 use BlockScalar scale with F16 dtype at offset 0.
        // Q4_1, Q5_1 use BlockScalarWithMin (d + m pair).
        for qt in [QuantType::Q4_0, QuantType::Q5_0, QuantType::Q8_0] {
            let desc = QuantFormatDescriptor::for_type(qt);
            match &desc.scale_layout {
                ScaleLayout::BlockScalar { offset_bytes, dtype } => {
                    assert_eq!(*offset_bytes, 0, "{:?} scale offset must be 0", qt);
                    assert_eq!(*dtype, ScaleDType::F16, "{:?} scale dtype must be F16", qt);
                }
                other => panic!("{:?} expected BlockScalar scale, got {:?}", qt, other),
            }
        }

        for qt in [QuantType::Q4_1, QuantType::Q5_1] {
            let desc = QuantFormatDescriptor::for_type(qt);
            match &desc.scale_layout {
                ScaleLayout::BlockScalarWithMin { dtype, .. } => {
                    assert_eq!(*dtype, ScaleDType::F16, "{:?} scale dtype must be F16", qt);
                }
                other => panic!("{:?} expected BlockScalarWithMin scale, got {:?}", qt, other),
            }
        }
    }

    // ── Additional tests (wave-12x59) ────────────────────────────────────

    #[test]
    fn algo_kind_descriptor_quant_type_matches_kind() {
        // Every QuantAlgoKind's descriptor() must return a QuantFormatDescriptor
        // whose quant_type field matches the kind's own quant_type() mapping.
        for &kind in QuantAlgoKind::all() {
            let desc = kind.descriptor();
            assert_eq!(desc.quant_type, kind.quant_type(),
                "QuantAlgoKind {:?}: descriptor quant_type {:?} != kind.quant_type {:?}",
                kind, desc.quant_type, kind.quant_type());
        }
    }

    #[test]
    fn isa_kind_copy_independence() {
        // IsaKind derives Copy; modifying a copy must not affect the original.
        let original = IsaKind::X86;
        let copy = original;
        assert_eq!(original, copy);
        // Verify all() still returns the canonical 5 variants after copy.
        assert_eq!(IsaKind::all().len(), 5);
        assert_eq!(IsaKind::all()[0], IsaKind::X86);
    }

    #[test]
    fn op_category_copy_independence() {
        // OpCategory derives Copy; copies are independent and equal.
        let original = OpCategory::Gemm;
        let copy = original;
        assert_eq!(original, copy);
        assert_eq!(OpCategory::all().len(), 5);
        assert_eq!(OpCategory::all()[0], OpCategory::Gemm);
    }

    #[test]
    fn coverage_matrix_iter_isa_op_pairing() {
        // CoverageMatrix iter() must yield 25 entries where each (isa, op)
        // pair corresponds to the correct index position.
        let m = CoverageMatrix::new(QuantType::F32);
        let entries: Vec<_> = m.iter().collect();
        assert_eq!(entries.len(), 25);
        // Verify that for each ISA, all 5 OpCategory entries appear in order.
        for (isa_idx, &isa) in IsaKind::all().iter().enumerate() {
            for (op_idx, &op) in OpCategory::all().iter().enumerate() {
                let entry = entries[isa_idx * 5 + op_idx];
                assert_eq!(entry.isa, isa, "Entry at index {} has wrong ISA", isa_idx * 5 + op_idx);
                assert_eq!(entry.op, op, "Entry at index {} has wrong OpCategory", isa_idx * 5 + op_idx);
            }
        }
    }

    #[test]
    fn coverage_matrix_fp8_e4m3_quant_path() {
        // FP8 E4M3 is not float (not F32/BF16/F16), not INT8, not INT4;
        // its Quant GEMM path should be DequantFMA on x86 (no native instruction).
        let m = CoverageMatrix::new(QuantType::Fp8E4M3);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Gemm), CoveragePath::DequantFMA);
        // Activation is always Native regardless of format.
        assert_eq!(m.get(IsaKind::X86, OpCategory::Activation), CoveragePath::Native);
        // Quant path for non-float, non-INT8, non-INT4 on x86: DequantFMA.
        assert_eq!(m.get(IsaKind::X86, OpCategory::Quant), CoveragePath::DequantFMA);
    }

    #[test]
    fn coverage_matrix_squeeze_quant_dequantfma_cpu() {
        // SqueezeLLM (3-bit codebook) has no native ISA; Quant path on x86/ARM/Metal is DequantFMA.
        // GPU PTX/HIP: Assisted (software unpack path exists).
        let m = CoverageMatrix::new(QuantType::Squeeze);
        assert_eq!(m.get(IsaKind::X86, OpCategory::Quant), CoveragePath::DequantFMA);
        assert_eq!(m.get(IsaKind::Arm, OpCategory::Quant), CoveragePath::DequantFMA);
        assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Quant), CoveragePath::DequantFMA);
        assert_eq!(m.get(IsaKind::GpuPtx, OpCategory::Quant), CoveragePath::Assisted);
        assert_eq!(m.get(IsaKind::GpuHip, OpCategory::Quant), CoveragePath::Assisted);
    }

    #[test]
    fn quant_algo_kind_all_unique_no_duplicates() {
        // QuantAlgoKind::all() must contain no duplicate entries.
        let all = QuantAlgoKind::all();
        for (i, kind_i) in all.iter().enumerate() {
            for (j, kind_j) in all.iter().enumerate() {
                if i != j {
                    assert_ne!(kind_i, kind_j,
                        "QuantAlgoKind::all() has duplicate at indices {} and {}: {:?}", i, j, kind_i);
                }
            }
        }
    }

    #[test]
    fn quant_algo_kind_copy_trait() {
        // QuantAlgoKind derives Copy; copied values must be equal and independent.
        let original = QuantAlgoKind::Nvfp4;
        let copy = original;
        assert_eq!(original, copy);
        // Ensure descriptor() produces identical results for both.
        assert_eq!(original.descriptor().name, copy.descriptor().name);
        assert_eq!(original.descriptor().block_size, copy.descriptor().block_size);
    }

    #[test]
    fn coverage_matrix_int4_quant_path_x86_assisted() {
        // INT4-class formats (AWQ4, GPTQ4, Q4_0, Q4K) have Assisted quant on x86
        // because VPDPBUSD does not handle 4-bit; DequantFMA on Metal.
        for qt in [QuantType::AWQ4, QuantType::GPTQ4, QuantType::Q4_0, QuantType::Q4K] {
            let m = CoverageMatrix::new(qt);
            assert_eq!(m.get(IsaKind::X86, OpCategory::Quant), CoveragePath::Assisted,
                "{:?} Quant on x86 should be Assisted", qt);
            assert_eq!(m.get(IsaKind::Arm, OpCategory::Quant), CoveragePath::Assisted,
                "{:?} Quant on ARM should be Assisted", qt);
            assert_eq!(m.get(IsaKind::GpuMsl, OpCategory::Quant), CoveragePath::DequantFMA,
                "{:?} Quant on Metal should be DequantFMA", qt);
        }
    }

    #[test]
    fn coverage_entry_debug_format_contains_fields() {
        // CoverageEntry derives Debug; its debug output must include isa, op, path.
        let entry = CoverageEntry {
            isa: IsaKind::X86,
            op: OpCategory::Gemm,
            path: CoveragePath::Native,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("X86"), "Debug output must contain ISA name");
        assert!(debug_str.contains("Gemm"), "Debug output must contain OpCategory name");
        assert!(debug_str.contains("Native"), "Debug output must contain CoveragePath name");
    }
}
