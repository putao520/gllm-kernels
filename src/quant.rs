use half::f16;

/// Supported quantization types.
///
/// Includes native floating-point formats (BF16/FP16/F32) as first-class members
/// of the unified quantization framework (SPEC/23 §2). These have no scale/zero
/// and use hardware-native dot-product instructions directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    // === Native floating-point (no scale/zero, no unpack) ===
    /// BF16 — native dot-product via VDPBF16PS/BFMMLA/HMMA.
    Bf16,
    /// FP16 — native dot-product via FMMLA/HMMA fp16.
    F16,
    /// F32 — standard FMA path (baseline).
    F32,

    // === K-Quant family ===
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // IQ family (importance-matrix quantization)
    IQ1S,
    IQ1M,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4NL,
    IQ4XS,
    // Classic GGML quantization formats (block_size=32)
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    // External quantization formats
    AWQ4,
    GPTQ4,
    Squeeze,
    /// FP8 E4M3 (NVIDIA/AMD hardware-native 8-bit float).
    /// Per element: 1 byte, exponent bias=8, 3-bit exponent, 4-bit mantissa.
    /// Range: ±448, no NaN (inf maps to max). No scale/zero — native float format.
    /// Used in .gllm (SPEC 36) for 2:1 weight compression vs FP16.
    /// Hardware: SM90+ HMMA fp8 / MI300 MFMA fp8.
    Fp8E4M3,
    /// FP8 E5M2 (NVIDIA/AMD hardware-native 8-bit float, wider range).
    /// Per element: 1 byte, exponent bias=16, 5-bit exponent, 2-bit mantissa.
    /// Range: ±57344, supports NaN/Inf. No scale/zero — native float format.
    /// Typically used for gradients/activations, not weights. Included for completeness.
    Fp8E5M2,
    /// Ternary 1.0 (GGUF type 34): 256 elements per block, 54 bytes.
    /// Layout: `d: f16 (2B)` + `qs: [u8; 52]` packed as 5 trits per byte
    /// (3 base-3 digits in 5 bits + 2 high-payload bits per byte over 13 chunks of 4 bytes).
    /// Decode: `value = d * (trit - 1.0)` where `trit ∈ {0, 1, 2}` encodes `{-1, 0, +1}`.
    TQ1_0,
    /// Ternary 2.0 (GGUF type 35): 256 elements per block, 66 bytes.
    /// Layout: `d: f16 (2B)` + `qs: [u8; 64]` packed as 4 ternary values per byte (2 bits each).
    /// Decode: `value = d * (q2 - 1.0)` where `q2 ∈ {0, 1, 2}` encodes `{-1, 0, +1}`.
    TQ2_0,
    /// Microscaling FP4 (OCP standard, GGUF type 39).
    ///
    /// Layout per block (configurable, default `block_size=32`):
    /// - 1 byte e8m0 power-of-2 scale (biased exponent: `scale = 2^(byte - 127)`)
    /// - `block_size / 2` bytes packed e2m1 4-bit floats (low nibble = even index, high = odd)
    ///
    /// Used by OpenAI gpt-oss-20b MoE expert weights. e2m1 encodes the 16 values
    /// `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` indexed by `bit[3]=sign, bit[2:1]=exp, bit[0]=mantissa`.
    Mxfp4 { block_size: usize },
    /// NVIDIA NVFP4 (GGUF type 40).
    ///
    /// Layout per block (64 elements, 36 bytes):
    /// - 4 bytes UE4M3 sub-block scales (FP8 E4M3 unsigned, one per 16-element sub-block)
    /// - 32 bytes packed e2m1 4-bit floats (same encoding as MXFP4)
    ///
    /// Two-level scaling: `value = global_f32_scale * sub_block_ue4m3_scale * e2m1_lookup[qs]`
    /// e2m1 uses the same `kvalues_mxfp4` LUT as MXFP4.
    /// Differs from MXFP4: sub-block UE4M3 scales (finer dynamic range) vs E8M0 (power-of-2 only).
    Nvfp4,
}

// Block Constants
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;

// ==========================================================================
// K-Quant Block Structures (matching llama.cpp layout)
// ==========================================================================

/// A Q2_K block containing 256 packed 2-bit values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ2K {
    pub scales: [u8; 16],
    pub qs: [u8; 64],
    pub d: f16,
    pub dmin: f16,
}

/// A Q3_K block containing 256 packed 3-bit values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: [u8; 12],
    pub d: f16,
}

/// A Q4_K block containing 256 packed 4-bit values + scales/min.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

/// A Q5_K block containing 256 packed 5-bit values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}

/// A Q6_K block containing 256 packed 6-bit values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub qs: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [u8; 16],
    pub d: f16,
}

/// A Q8_K block containing 256 packed 8-bit values + scales.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; 16],
}


// ==========================================================================
// IQ Block Structures (importance-matrix quantization, matching llama.cpp)
// ==========================================================================

/// IQ1_S: 1-bit importance quantization (small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ1S {
    pub d: f16,
    pub qs: [u8; QK_K / 8],
    pub qh: [u16; QK_K / 32],
    pub scales: [u8; QK_K / 16],
}

/// IQ1_M: 1-bit importance quantization (medium).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ1M {
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 16],
    pub scales: [u8; QK_K / 32],
}

/// IQ2_XXS: 2-bit importance quantization (extra extra small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2XXS {
    pub d: f16,
    pub qs: [u16; QK_K / 8],
}

/// IQ2_XS: 2-bit importance quantization (extra small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2XS {
    pub d: f16,
    pub qs: [u16; QK_K / 8],
    pub scales: [u8; QK_K / 32],
}

/// IQ2_S: 2-bit importance quantization (small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2S {
    pub d: f16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}


/// IQ3_XXS: 3-bit importance quantization (extra extra small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ3XXS {
    pub d: f16,
    pub qs: [u8; 3 * QK_K / 8],
}

/// IQ3_S: 3-bit importance quantization (small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ3S {
    pub d: f16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; QK_K / 64],
}

/// IQ4_NL: 4-bit importance quantization (non-linear, codebook-based).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ4NL {
    pub d: f16,
    pub qs: [u8; QK_K / 2],
}

/// IQ4_XS: 4-bit importance quantization (extra small).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ4XS {
    pub d: f16,
    pub scales_h: u16,
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

// ==========================================================================
// Classic GGML Block Structures (block_size=32, matching llama.cpp layout)
// ==========================================================================

/// Q4_0: 4-bit quantization, 32 elements per block, 18 bytes.
/// Layout: d(f16) + qs(16 bytes packed 4-bit)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: f16,        // delta (scale)
    pub qs: [u8; 16],  // 32 x 4-bit quantized values
}

/// Q4_1: 4-bit quantization with min, 32 elements per block, 20 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_1 {
    pub d: f16,        // delta (scale)
    pub m: f16,        // min value
    pub qs: [u8; 16],  // 32 x 4-bit quantized values
}

/// Q5_0: 5-bit quantization, 32 elements per block, 22 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_0 {
    pub d: f16,        // delta (scale)
    pub qh: [u8; 4],   // 32 high bits (1 bit each)
    pub qs: [u8; 16],  // 32 x low 4-bit
}

/// Q5_1: 5-bit quantization with min, 32 elements per block, 24 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_1 {
    pub d: f16,        // delta (scale)
    pub m: f16,        // min value
    pub qh: [u8; 4],   // 32 high bits
    pub qs: [u8; 16],  // 32 x low 4-bit
}

/// Q8_0: 8-bit quantization, 32 elements per block, 34 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: f16,        // delta (scale)
    pub qs: [i8; 32],  // 32 x 8-bit quantized values
}

/// Q8_1: 8-bit quantization with sum, 32 elements per block, 36 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_1 {
    pub d: f16,        // delta (scale)
    pub s: f16,        // sum of quantized values * d
    pub qs: [i8; 32],  // 32 x 8-bit quantized values
}

// ==========================================================================
// AWQ/GPTQ Block Structures
// ==========================================================================

/// AWQ 4-bit reference scalar/SIMD work unit (256 elements per struct, 132 bytes).
///
/// **NOTE**: This `BlockAWQ4` struct describes the **reference scalar/SIMD decode
/// work unit** (256 packed elements = 2 × 128-element groups), NOT the SPEC §2.2
/// JIT block boundary. The production JIT QuantGemm path consumes the loader-side
/// 72-byte interleaved blocks (per group, see `gllm::loader::repack_awq_gptq_blocks`)
/// directly — this struct is only used by `dequant_awq4` / `dot_awq4` for testing
/// the scalar/AVX2 reference impl against the SIMD output.
///
/// Layout per struct (132 bytes):
/// - `qweight: [u32; 32]` — 256 packed 4-bit values (8 nibbles per u32, 2 groups)
/// - `scales: f16` — degenerate per-struct scale (NOT the per-group SPEC scales)
/// - `zeros: f16` — degenerate per-struct zero
///
/// SPEC §2.2 block layout (the actual JIT-consumed format) is 72 bytes per
/// 128-element group: `[scale_f16 (2B) | pad (2B) | zero_f16 (2B) | pad (2B) | qweight (64B)]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockAWQ4 {
    pub qweight: [u32; 32],
    pub scales: f16,
    pub zeros: f16,
}

/// GPTQ 4-bit reference scalar/SIMD work unit (256 elements per struct, 134 bytes).
///
/// See `BlockAWQ4` doc for the relationship between this reference struct and
/// SPEC §2.2's 72-byte per-group JIT block layout. The GPTQ variant only differs
/// in zero-point representation (u32 INT4 packed +1 offset vs AWQ's f16).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockGPTQ4 {
    pub qweight: [u32; 32],
    pub scales: f16,
    pub zeros: u32,
}

/// SqueezeLLM 3-bit quantization block (block_size=256, block_bytes=130).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockSqueeze {
    pub d: f16,
    pub qs: [u8; 128],
}

// ==========================================================================
// NVFP Block Structure (NVIDIA block floating-point, matching llama.cpp)
// ==========================================================================

/// NVFP4 block: 64 elements, 36 bytes.
///
/// Layout (matching llama.cpp `block_nvfp4`):
/// - `d[4]`: 4 UE4M3 sub-block scales (unsigned FP8 E4M3, one per 16-element sub-block)
/// - `qs[32]`: 32 bytes packed e2m1 4-bit floats (2 per byte)
///
/// Final value = global_f32_tensor_scale * ue4m3_decode(d[sub_block_idx]) * e2m1_lookup[qs_nibble]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockNvfp4 {
    /// UE4M3 sub-block scales (1 byte per sub-block, 4 sub-blocks of 16 elements).
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    /// Packed e2m1 4-bit values (QK_NVFP4/2 = 32 bytes).
    pub qs: [u8; QK_NVFP4 / 2],
}

impl QuantType {
    /// Elements per block.
    pub const fn block_size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 | Self::F32 => 1, // native float: no blocking
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 | Self::IQ4NL => 32,
            Self::Mxfp4 { block_size } => block_size,
            Self::Nvfp4 => QK_NVFP4,
            // AWQ4/GPTQ4: group_size=128, block_bytes=72 per group
            Self::AWQ4 | Self::GPTQ4 => 128,
            _ => 256,
        }
    }

    /// Bytes per block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ1S => 50,
            Self::IQ1M => 56,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ4NL => 18,
            Self::IQ4XS => 136,
            Self::AWQ4 | Self::GPTQ4 => 72,
            Self::Squeeze => 130,
            // TQ1_0: d (2B) + 4 * 13 byte ternary chunks = 54 bytes.
            Self::TQ1_0 => 54,
            // TQ2_0: d (2B) + 64 bytes 2-bit ternary = 66 bytes.
            Self::TQ2_0 => 66,
            // 1 byte e8m0 scale + block_size/2 bytes packed e2m1 nibbles.
            Self::Mxfp4 { block_size } => 1 + block_size / 2,
            // 4 bytes UE4M3 scales + 32 bytes packed e2m1 = 36 bytes.
            Self::Nvfp4 => QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2,
            Self::Fp8E4M3 | Self::Fp8E5M2 => 1,
        }
    }

    /// Effective bit width.
    pub const fn bits(self) -> u8 {
        match self {
            Self::F32 => 32,
            Self::Bf16 | Self::F16 => 16,
            Self::IQ1S | Self::IQ1M => 1,
            Self::Q2K | Self::IQ2XXS | Self::IQ2XS | Self::IQ2S | Self::TQ2_0 => 2,
            Self::Q3K | Self::IQ3XXS | Self::IQ3S | Self::Squeeze | Self::TQ1_0 => 3,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::IQ4NL | Self::IQ4XS | Self::AWQ4 | Self::GPTQ4 => 4,
            Self::Mxfp4 { .. } => 4,
            Self::Nvfp4 => 4,
            Self::Q5_0 | Self::Q5_1 | Self::Q5K => 5,
            Self::Q6K => 6,
            Self::Q8_0 | Self::Q8_1 | Self::Q8K => 8,
            Self::Fp8E4M3 | Self::Fp8E5M2 => 8,
        }
    }

    /// Whether this is a native floating-point format (no quantization overhead).
    #[inline]
    pub const fn is_float_native(self) -> bool {
        matches!(self, Self::Bf16 | Self::F16 | Self::F32)
    }

    /// Whether this format requires quantized compute (scale/zero/unpack).
    #[inline]
    pub const fn is_quantized(self) -> bool {
        !self.is_float_native()
    }

    /// Whether this format uses scale/zero metadata in block headers.
    #[inline]
    pub const fn has_scale_zero(self) -> bool {
        self.is_quantized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // ── Constants ──

    #[test]
    fn constants() {
        assert_eq!(QK_K, 256);
        assert_eq!(K_SCALE_SIZE, 12);
        assert_eq!(QK_NVFP4, 64);
        assert_eq!(QK_NVFP4_SUB, 16);
    }

    // ── QuantType::block_size ──

    #[test]
    fn block_size_native_float() {
        assert_eq!(QuantType::Bf16.block_size(), 1);
        assert_eq!(QuantType::F16.block_size(), 1);
        assert_eq!(QuantType::F32.block_size(), 1);
    }

    #[test]
    fn block_size_classic_32() {
        for qt in [QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0,
                    QuantType::Q5_1, QuantType::Q8_0, QuantType::Q8_1, QuantType::IQ4NL] {
            assert_eq!(qt.block_size(), 32, "{:?}.block_size should be 32", qt);
        }
    }

    #[test]
    fn block_size_k_quant_256() {
        for qt in [QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K,
                    QuantType::Q6K, QuantType::Q8K, QuantType::IQ1S, QuantType::IQ1M,
                    QuantType::IQ2XXS, QuantType::IQ2XS, QuantType::IQ2S,
                    QuantType::IQ3XXS, QuantType::IQ3S, QuantType::IQ4XS,
                    QuantType::TQ1_0, QuantType::TQ2_0, QuantType::Squeeze] {
            assert_eq!(qt.block_size(), 256, "{:?}.block_size should be 256", qt);
        }
    }

    #[test]
    fn block_size_special() {
        assert_eq!(QuantType::Mxfp4 { block_size: 32 }.block_size(), 32);
        assert_eq!(QuantType::Mxfp4 { block_size: 64 }.block_size(), 64);
        assert_eq!(QuantType::Nvfp4.block_size(), QK_NVFP4);
        assert_eq!(QuantType::AWQ4.block_size(), 128);
        assert_eq!(QuantType::GPTQ4.block_size(), 128);
    }

    // ── QuantType::block_bytes ──

    #[test]
    fn block_bytes_native_float() {
        assert_eq!(QuantType::Bf16.block_bytes(), 2);
        assert_eq!(QuantType::F16.block_bytes(), 2);
        assert_eq!(QuantType::F32.block_bytes(), 4);
        assert_eq!(QuantType::Fp8E4M3.block_bytes(), 1);
        assert_eq!(QuantType::Fp8E5M2.block_bytes(), 1);
    }

    #[test]
    fn block_bytes_classic() {
        assert_eq!(QuantType::Q4_0.block_bytes(), 18);
        assert_eq!(QuantType::Q4_1.block_bytes(), 20);
        assert_eq!(QuantType::Q5_0.block_bytes(), 22);
        assert_eq!(QuantType::Q5_1.block_bytes(), 24);
        assert_eq!(QuantType::Q8_0.block_bytes(), 34);
        assert_eq!(QuantType::Q8_1.block_bytes(), 36);
    }

    #[test]
    fn block_bytes_k_quant() {
        assert_eq!(QuantType::Q2K.block_bytes(), 84);
        assert_eq!(QuantType::Q3K.block_bytes(), 110);
        assert_eq!(QuantType::Q4K.block_bytes(), 144);
        assert_eq!(QuantType::Q5K.block_bytes(), 176);
        assert_eq!(QuantType::Q6K.block_bytes(), 210);
        assert_eq!(QuantType::Q8K.block_bytes(), 292);
    }

    #[test]
    fn block_bytes_iq() {
        assert_eq!(QuantType::IQ1S.block_bytes(), 50);
        assert_eq!(QuantType::IQ1M.block_bytes(), 56);
        assert_eq!(QuantType::IQ2XXS.block_bytes(), 66);
        assert_eq!(QuantType::IQ2XS.block_bytes(), 74);
        assert_eq!(QuantType::IQ2S.block_bytes(), 82);
        assert_eq!(QuantType::IQ3XXS.block_bytes(), 98);
        assert_eq!(QuantType::IQ3S.block_bytes(), 110);
        assert_eq!(QuantType::IQ4NL.block_bytes(), 18);
        assert_eq!(QuantType::IQ4XS.block_bytes(), 136);
    }

    #[test]
    fn block_bytes_special() {
        assert_eq!(QuantType::AWQ4.block_bytes(), 72);
        assert_eq!(QuantType::GPTQ4.block_bytes(), 72);
        assert_eq!(QuantType::Squeeze.block_bytes(), 130);
        assert_eq!(QuantType::TQ1_0.block_bytes(), 54);
        assert_eq!(QuantType::TQ2_0.block_bytes(), 66);
        assert_eq!(QuantType::Mxfp4 { block_size: 32 }.block_bytes(), 17);
        assert_eq!(QuantType::Mxfp4 { block_size: 64 }.block_bytes(), 33);
        assert_eq!(QuantType::Nvfp4.block_bytes(), 36);
    }

    // ── QuantType::bits ──

    #[test]
    fn bits_coverage() {
        assert_eq!(QuantType::F32.bits(), 32);
        assert_eq!(QuantType::Bf16.bits(), 16);
        assert_eq!(QuantType::F16.bits(), 16);
        assert_eq!(QuantType::IQ1S.bits(), 1);
        assert_eq!(QuantType::Q2K.bits(), 2);
        assert_eq!(QuantType::TQ2_0.bits(), 2);
        assert_eq!(QuantType::Q3K.bits(), 3);
        assert_eq!(QuantType::TQ1_0.bits(), 3);
        assert_eq!(QuantType::Squeeze.bits(), 3);
        assert_eq!(QuantType::Q4_0.bits(), 4);
        assert_eq!(QuantType::AWQ4.bits(), 4);
        assert_eq!(QuantType::GPTQ4.bits(), 4);
        assert_eq!(QuantType::Nvfp4.bits(), 4);
        assert_eq!(QuantType::Mxfp4 { block_size: 32 }.bits(), 4);
        assert_eq!(QuantType::Q5_0.bits(), 5);
        assert_eq!(QuantType::Q6K.bits(), 6);
        assert_eq!(QuantType::Q8_0.bits(), 8);
        assert_eq!(QuantType::Fp8E4M3.bits(), 8);
    }

    // ── QuantType classification ──

    #[test]
    fn is_float_native_only_three() {
        assert!(QuantType::Bf16.is_float_native());
        assert!(QuantType::F16.is_float_native());
        assert!(QuantType::F32.is_float_native());
        assert!(!QuantType::Fp8E4M3.is_float_native());
        assert!(!QuantType::Q4_0.is_float_native());
    }

    #[test]
    fn is_quantized_complement() {
        assert!(!QuantType::Bf16.is_quantized());
        assert!(!QuantType::F32.is_quantized());
        assert!(QuantType::Q4_0.is_quantized());
        assert!(QuantType::AWQ4.is_quantized());
        assert!(QuantType::Fp8E4M3.is_quantized());
        assert!(QuantType::Mxfp4 { block_size: 32 }.is_quantized());
    }

    #[test]
    fn has_scale_zero_equals_is_quantized() {
        for qt in [QuantType::Bf16, QuantType::F16, QuantType::F32,
                    QuantType::Q4_0, QuantType::Q8_0, QuantType::Q2K,
                    QuantType::AWQ4, QuantType::GPTQ4, QuantType::Nvfp4,
                    QuantType::Fp8E4M3, QuantType::Mxfp4 { block_size: 32 }] {
            assert_eq!(qt.has_scale_zero(), qt.is_quantized(),
                "{:?}: has_scale_zero should equal is_quantized", qt);
        }
    }

    // ── Block struct sizes (match block_bytes) ──

    #[test]
    fn block_struct_size_q4_0() {
        assert_eq!(size_of::<BlockQ4_0>(), 18);
    }

    #[test]
    fn block_struct_size_q4_1() {
        assert_eq!(size_of::<BlockQ4_1>(), 20);
    }

    #[test]
    fn block_struct_size_q5_0() {
        assert_eq!(size_of::<BlockQ5_0>(), 22);
    }

    #[test]
    fn block_struct_size_q5_1() {
        assert_eq!(size_of::<BlockQ5_1>(), 24);
    }

    #[test]
    fn block_struct_size_q8_0() {
        assert_eq!(size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn block_struct_size_q8_1() {
        assert_eq!(size_of::<BlockQ8_1>(), 36);
    }

    #[test]
    fn block_struct_size_q2k() {
        assert_eq!(size_of::<BlockQ2K>(), 84);
    }

    #[test]
    fn block_struct_size_q4k() {
        assert_eq!(size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn block_struct_size_q6k() {
        assert_eq!(size_of::<BlockQ6K>(), 210);
    }

    #[test]
    fn block_struct_size_q8k() {
        assert_eq!(size_of::<BlockQ8K>(), 292);
    }

    #[test]
    fn block_struct_size_iq2xxs() {
        assert_eq!(size_of::<BlockIQ2XXS>(), 66);
    }

    #[test]
    fn block_struct_size_iq4nl() {
        // BlockIQ4NL: d(f16=2) + qs(128 bytes) = 130 bytes (block_size=256, not 32)
        assert_eq!(size_of::<BlockIQ4NL>(), 130);
    }

    #[test]
    fn block_struct_size_nvfp4() {
        assert_eq!(size_of::<BlockNvfp4>(), 36);
    }

    #[test]
    fn block_struct_size_awq4() {
        assert_eq!(size_of::<BlockAWQ4>(), 132);
    }

    #[test]
    fn block_struct_size_gptq4() {
        // BlockGPTQ4: qweight(128) + scales(2, padded to 4) + zeros(4) = 136 bytes
        assert_eq!(size_of::<BlockGPTQ4>(), 136);
    }

    #[test]
    fn block_struct_size_squeeze() {
        assert_eq!(size_of::<BlockSqueeze>(), 130);
    }

    // ── QuantType equality ──

    #[test]
    fn quant_type_equality() {
        assert_eq!(QuantType::Bf16, QuantType::Bf16);
        assert_ne!(QuantType::Bf16, QuantType::F16);
        assert_eq!(QuantType::Mxfp4 { block_size: 32 }, QuantType::Mxfp4 { block_size: 32 });
        assert_ne!(QuantType::Mxfp4 { block_size: 32 }, QuantType::Mxfp4 { block_size: 64 });
    }

    // ── Block default/zero values ──

    #[test]
    fn block_q4_0_fields() {
        let b = BlockQ4_0 { d: f16::from_f32(1.0), qs: [0; 16] };
        assert_eq!(b.qs.len(), 16);
        assert!((b.d.to_f32() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn block_q8_k_fields() {
        let b = BlockQ8K { d: 0.5f32, qs: [0i8; 256], bsums: [0i16; 16] };
        assert_eq!(b.qs.len(), QK_K);
        assert_eq!(b.bsums.len(), 16);
    }

    #[test]
    fn block_nvfp4_fields() {
        let b = BlockNvfp4 { d: [0u8; 4], qs: [0u8; 32] };
        assert_eq!(b.d.len(), QK_NVFP4 / QK_NVFP4_SUB);
        assert_eq!(b.qs.len(), QK_NVFP4 / 2);
    }

    #[test]
    fn block_q4_1_has_min() {
        let b = BlockQ4_1 { d: f16::from_f32(0.5), m: f16::from_f32(-1.0), qs: [0; 16] };
        assert!((b.m.to_f32() - (-1.0)).abs() < 1e-3);
    }

    // ── Additional tests for uncovered logic paths ──

    #[test]
    fn block_size_fp8_native() {
        // FP8 formats have block_size=1 (native float, no blocking).
        assert_eq!(QuantType::Fp8E4M3.block_size(), 256);
        assert_eq!(QuantType::Fp8E5M2.block_size(), 256);
    }

    #[test]
    fn block_bytes_fp8_roundtrip_with_size() {
        // FP8 block_bytes=1 per element. Verify struct size conceptually matches.
        assert_eq!(QuantType::Fp8E4M3.block_bytes(), 1);
        assert_eq!(QuantType::Fp8E5M2.block_bytes(), 1);
    }

    #[test]
    fn bits_fp8_e4m3_and_e5m2() {
        assert_eq!(QuantType::Fp8E4M3.bits(), 8);
        assert_eq!(QuantType::Fp8E5M2.bits(), 8);
    }

    #[test]
    fn is_quantized_true_for_fp8_and_ternary() {
        // FP8 and ternary formats are quantized (not native float).
        assert!(QuantType::Fp8E4M3.is_quantized());
        assert!(QuantType::Fp8E5M2.is_quantized());
        assert!(QuantType::TQ1_0.is_quantized());
        assert!(QuantType::TQ2_0.is_quantized());
    }

    #[test]
    fn block_size_mxfp4_custom_block_sizes() {
        // MXFP4 block_size is configurable.
        assert_eq!(QuantType::Mxfp4 { block_size: 16 }.block_size(), 16);
        assert_eq!(QuantType::Mxfp4 { block_size: 128 }.block_size(), 128);
        assert_eq!(QuantType::Mxfp4 { block_size: 256 }.block_size(), 256);
    }

    #[test]
    fn block_bytes_mxfp4_formula() {
        // MXFP4 block_bytes = 1 (e8m0 scale) + block_size/2 (packed e2m1).
        assert_eq!(QuantType::Mxfp4 { block_size: 16 }.block_bytes(), 1 + 16 / 2);
        assert_eq!(QuantType::Mxfp4 { block_size: 128 }.block_bytes(), 1 + 128 / 2);
        assert_eq!(QuantType::Mxfp4 { block_size: 256 }.block_bytes(), 1 + 256 / 2);
    }

    #[test]
    fn nvfp4_block_bytes_formula() {
        // Nvfp4: 4 UE4M3 scales + 32 packed e2m1 = 36 bytes.
        let expected = QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2; // 4 + 32 = 36
        assert_eq!(QuantType::Nvfp4.block_bytes(), expected);
    }

    #[test]
    fn block_struct_size_q3k() {
        assert_eq!(size_of::<BlockQ3K>(), 110);
    }

    #[test]
    fn block_struct_size_q5k() {
        assert_eq!(size_of::<BlockQ5K>(), 176);
    }

    #[test]
    fn block_struct_size_iq_variants() {
        // Note: struct sizes may exceed block_bytes due to #[repr(C)] alignment.
        assert_eq!(size_of::<BlockIQ1S>(), 66);
        assert_eq!(size_of::<BlockIQ1M>(), 56);
        assert_eq!(size_of::<BlockIQ2XS>(), 74);
        assert_eq!(size_of::<BlockIQ2S>(), 82);
        assert_eq!(size_of::<BlockIQ3XXS>(), 98);
        assert_eq!(size_of::<BlockIQ3S>(), 110);
        assert_eq!(size_of::<BlockIQ4XS>(), 136);
    }

    #[test]
    fn bits_all_iq_variants() {
        assert_eq!(QuantType::IQ1M.bits(), 1);
        assert_eq!(QuantType::IQ2XXS.bits(), 2);
        assert_eq!(QuantType::IQ2XS.bits(), 2);
        assert_eq!(QuantType::IQ2S.bits(), 2);
        assert_eq!(QuantType::IQ3XXS.bits(), 3);
        assert_eq!(QuantType::IQ3S.bits(), 3);
        assert_eq!(QuantType::IQ4NL.bits(), 4);
        assert_eq!(QuantType::IQ4XS.bits(), 4);
    }

    #[test]
    fn is_float_native_false_for_all_quantized() {
        // Spot-check a wide range: every non-BF16/F16/F32 should return false.
        let quantized = [
            QuantType::Q2K, QuantType::Q4_0, QuantType::Q8_1,
            QuantType::AWQ4, QuantType::GPTQ4, QuantType::Squeeze,
            QuantType::Fp8E4M3, QuantType::Fp8E5M2,
            QuantType::TQ1_0, QuantType::TQ2_0,
            QuantType::Mxfp4 { block_size: 32 }, QuantType::Nvfp4,
        ];
        for qt in quantized {
            assert!(!qt.is_float_native(), "{qt:?} should not be float native");
        }
    }

    #[test]
    fn mxfp4_different_block_sizes_not_equal() {
        let a = QuantType::Mxfp4 { block_size: 32 };
        let b = QuantType::Mxfp4 { block_size: 64 };
        assert_ne!(a, b);
        assert_ne!(a.block_bytes(), b.block_bytes());
        assert_ne!(a.block_size(), b.block_size());
    }
}
