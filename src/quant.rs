use half::f16;

/// Supported quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    // K-Quant family
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
}

// Block Constants
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

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
    pub d: f32,
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
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
    pub d: f16,
    pub dmin: f16,
}

/// A Q6_K block containing 256 packed 6-bit values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub qs: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [u8; 16],
    pub d: f32,
}

/// A Q8_K block containing 256 packed 8-bit values + scales.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; 16],
}

// PLACEHOLDER_IQ_BLOCKS

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

// PLACEHOLDER_IQ_BLOCKS_2

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

/// AWQ 4-bit quantization block (group_size=128).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockAWQ4 {
    pub qweight: [u32; 32],
    pub scales: f16,
    pub zeros: f16,
}

/// GPTQ 4-bit quantization block (group_size=128).
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

impl QuantType {
    /// Elements per block.
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 | Self::IQ4NL => 32,
            _ => 256,
        }
    }

    /// Bytes per block.
    pub const fn block_bytes(self) -> usize {
        match self {
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
        }
    }

    /// Effective bit width.
    pub const fn bits(self) -> u8 {
        match self {
            Self::IQ1S | Self::IQ1M => 1,
            Self::Q2K | Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 2,
            Self::Q3K | Self::IQ3XXS | Self::IQ3S | Self::Squeeze => 3,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::IQ4NL | Self::IQ4XS | Self::AWQ4 | Self::GPTQ4 => 4,
            Self::Q5_0 | Self::Q5_1 | Self::Q5K => 5,
            Self::Q6K => 6,
            Self::Q8_0 | Self::Q8_1 | Self::Q8K => 8,
        }
    }
}
