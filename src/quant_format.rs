//! Algorithmic quantization format descriptor.
//!
//! Per SPEC `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md` — every quantization format is
//! described by a unified [`QuantFormatDescriptor`] capturing its byte-stream
//! layout (scale, zero, data, codebook). JIT codegen consumes the descriptor
//! through [`DecodeTraceBuilder`] (in `compiler/codegen/vm/quant_decode.rs`)
//! to generate per-format dequant trace, eliminating hand-written codegen
//! per (format × ISA × operator) combination.

use crate::quant::QuantType;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Storage layout of quantized weight tensors in memory.
///
/// Per SPEC `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md` §2.2 — every format has a
/// canonical storage order that affects memory access patterns during JIT codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageLayout {
    /// Standard row-major: elements within a block are contiguous,
    /// blocks are laid out sequentially along the output dimension.
    RowMajor,
    /// Column-interleaved: after g_idx reordering (e.g. GPTQ), bytes are
    /// interleaved for batched access along the reduction dimension.
    ColInterleaved,
    /// Dense packed: no interleaving, but multi-byte packing within elements
    /// (e.g. 3-bit packed across byte boundaries).
    Packed,
}

impl QuantFormatDescriptor {
    /// Look up the canonical descriptor for a given [`QuantType`] from the global registry.
    ///
    /// Panics if no descriptor is registered for the type (all 31+ types are registered
    /// at library init via [`QuantFormatRegistry::new`]).
    pub fn for_type(quant_type: QuantType) -> QuantFormatDescriptor {
        let reg = registry();
        reg.get(&quant_type)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "QuantFormatDescriptor::for_type: no descriptor registered for {:?}",
                    quant_type
                )
            })
    }
}

/// Hardware-native instruction that accelerates a quantization format.
///
/// Per SPEC §1.1 — each format may have a canonical native dot-product or
/// tensor-core instruction. `None` means the format requires software unpack
/// (Level 2 Assisted or Level 3 DequantFMA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeIsa {
    // === x86 ===
    /// VDPBF16PS: AVX-512 BF16 dot-product (bf16×bf16→f32).
    Vdpbf16ps,
    /// VPDPBUSD: AVX-VNNI INT8 dot-product (u8×i8→i32).
    Vpdpbusd,
    /// VFMADD: AVX-512 F32 FMA.
    Vfmadd,
    /// TDPBF16PS: AMX BF16 tile dot-product.
    Tdpbf16ps,
    /// TDPBSSD: AMX INT8 tile dot-product.
    Tdpbssd,
    // === ARM ===
    /// BFMMLA: SVE2/SME2 BF16 matrix multiply-accumulate.
    Bfmmla,
    /// FMMLA: SVE2/SME2 FP16 matrix multiply-accumulate.
    Fmmla,
    /// FMLA: SVE2/SME2 FP32 FMA.
    Fmla,
    /// SDOT: SVE2 INT8 signed dot-product.
    Sdot,
    // === NVIDIA PTX ===
    /// HMMA: SM80 tensor core (bf16/fp16).
    Hmma,
    /// WGMMA: SM90+ warp-group matrix multiply-accumulate.
    Wgmma,
    /// tcgen05 FP4: SM100+ Blackwell FP4 tensor core.
    Tcgen05Fp4,
    // === AMD HIP ===
    /// WMMA: GFX12 wave matrix multiply-accumulate (bf16/fp16).
    Wmma,
    /// v_wmma_i32_16x16x16_iu4_iu8: GFX12 INT4×INT8 WMMA.
    Gfx12WmmaIU4IU8,
}

/// How the quantized data maps to hardware instructions.
/// Drives `emit_unpack_weight` and `emit_dot_product` in the parameterized microkernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDataKind {
    // === Native float: no scale/zero, no unpack ===
    Bfloat16,
    Float16,
    Float32,
    // === Integer quantized: need scale/zero + possible unpack ===
    Int8,
    PackedInt4,
    SignedPackedInt4,
    PackedInt5,
    PackedInt6,
    Float4,
    /// FP8 E4M3/E5M2: hardware-native 8-bit float, no scale/zero, no unpack.
    /// Direct byte→compute_dtype conversion (e.g. F8→F16/BF16/F32).
    Float8,
    SuperLowBit,
    /// NVIDIA NVFP4: E2M1 packed data with per-sub-block UE4M3 FP8 scales.
    /// Two-level scaling: global F32 × sub-block UE4M3 × E2M1 lookup.
    Nvfp4,
    /// Per-tensor dynamic codebook (e.g. SqueezeLLM original paper).
    /// Indices are packed like SuperLowBit but decoded via runtime codebook
    /// pointer passed as an additional ABI argument, not a static LUT.
    /// See SPEC §2.2 SqueezeLLM evolution path.
    Codebook,
}

impl QuantDataKind {
    pub fn is_float(self) -> bool {
        matches!(self, Self::Bfloat16 | Self::Float16 | Self::Float32)
    }
}

/// Algebraic description of a quantization format's byte-stream layout.
///
/// Per SPEC `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md` §2 — unified descriptor for
/// all 28+ quantization formats. JIT codegen consumes this descriptor through
/// [`DecodeTraceBuilder`] to generate per-format dequant traces.
#[derive(Debug, Clone)]
pub struct QuantFormatDescriptor {
    /// Human-readable format name (e.g. "Q4_0", "BF16").
    pub name: &'static str,
    /// The canonical [`QuantType`] variant.
    pub quant_type: QuantType,
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// Total bytes per quantization block (scale + zero + data + padding).
    pub block_bytes: usize,
    /// Effective bits per element (e.g. Q4_0 = 4, BF16 = 16).
    pub bits_per_element: u8,
    /// Scale metadata layout within the block.
    pub scale_layout: ScaleLayout,
    /// Zero-point metadata layout within the block.
    pub zero_layout: ZeroLayout,
    /// Quantized data layout within the block.
    pub data_layout: DataLayout,
    /// Semantic data kind driving unpack/dot-product selection.
    pub data_kind: QuantDataKind,
    /// Optional static codebook for LUT-based formats (IQ family).
    pub codebook: Option<CodebookSpec>,
    /// Canonical storage order of the weight tensor (REQ-QCG1-001).
    pub storage_layout: StorageLayout,
    /// Hardware-native instruction that accelerates this format, if any (REQ-QCG1-002).
    ///
    /// `None` means the format requires software-assisted unpack or DequantFMA
    /// fallback on all known ISAs (e.g. super-low-bit formats without native instructions).
    pub native_isa: Option<NativeIsa>,
}

#[derive(Debug, Clone)]
pub enum ScaleLayout {
    /// No scale (native float formats: BF16, FP16, F32).
    None,
    /// Single f16 / f32 / bf16 scale at fixed offset, broadcast to all lanes.
    BlockScalar { offset_bytes: usize, dtype: ScaleDType },
    /// Block-level d + min, both scalars broadcast.
    BlockScalarWithMin {
        d_offset: usize,
        m_offset: usize,
        dtype: ScaleDType,
    },
    /// K-quant: block-level d + sub-block 6-bit packed scales.
    Hierarchical {
        block_d_offset: usize,
        block_dmin_offset: Option<usize>,
        sub_scales_offset: usize,
        sub_scales_bits: u8,
        sub_scales_count: usize,
        sub_block_elements: usize,
        packed_layout: PackedScaleLayout,
    },
    /// Q6_K style: block-level d + sub-block i8 scales (no packing).
    Q6KScales {
        block_d_offset: usize,
        sub_scales_offset: usize,
        sub_scales_count: usize,
        sub_block_elements: usize,
    },
    /// MXFP4: scales stored in a separate array (external pointer).
    ExternalArray { stride: usize, dtype: ScaleDType },
    /// NVFP4: multiple sub-block scalar scales stored inline in the block.
    /// One scale per sub-block (e.g. 4 UE4M3 scales for NVFP4's 16-element sub-blocks).
    SubBlockScalars {
        offset_bytes: usize,
        sub_block_size: usize,
        dtype: ScaleDType,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ZeroLayout {
    /// No zero-point (Q4_0, Q5_0, Q8_0, Q6_K).
    None,
    /// Block-level scalar zero/min.
    BlockScalar {
        offset_bytes: usize,
        dtype: ScaleDType,
    },
    /// Q4_K / Q5_K: block dmin × sub-block m.
    Hierarchical {
        dmin_offset: usize,
        sub_m_offset: usize,
    },
    /// Static integer subtraction (e.g. Q4_0: nibble - 8, Q6_K: q - 32).
    StaticBias { value: i32 },
    /// Q4_1 / Q5_1 / Q8_1: block-level min offset added AFTER scale.
    /// `value = d * quantized + m` where m is a scalar f16 at `offset_bytes`.
    /// Unlike `BlockScalar` (which is PreScaleSubtract for AWQ/GPTQ),
    /// this is PostScaleAdd: the min is added after scaling.
    BlockMin {
        offset_bytes: usize,
        dtype: ScaleDType,
    },
}

#[derive(Debug, Clone)]
pub enum DataLayout {
    /// Packed nibbles in `lanes/2` consecutive bytes.
    PackedNibbles { offset: usize, low_first: bool },
    /// Low nibbles + high bit-plane (Q5_0/Q5_1 have 1 high bit, Q6_K has 2).
    NibbleWithHighBits {
        low_offset: usize,
        high_offset: usize,
        high_bits_per_elem: u8,
    },
    /// Full bytes (Q8_0 signed, Q8_K signed, Q8_1 signed).
    Bytes { offset: usize, signed: bool },
    /// Codebook indices (IQ family).
    CodebookIndex { offset: usize, index_bits: u8 },
    /// Q3_K: 2-bit values at variable shifts + conditional bias from hmask.
    /// Layout: hmask[32] + qs[64] + scales[12] + d(f16) = 110 bytes, 256 elements.
    /// Elements accessed as: seg*128 + j*32 + run*16 + l (0..16).
    /// qs_val = (qs[seg*32 + run*16 + l] >> (j*2)) & 3
    /// bias = 0 if (hmask[run*16 + l] & (1 << (seg*4 + j))) else 4
    /// result = (qs_val - bias) * dl   where dl = d * (scale - 32)
    TwoBitConditionalBias {
        qs_offset: usize,    // offset to qs[] within block (32 for Q3_K)
        hmask_offset: usize, // offset to hmask[] within block (0 for Q3_K)
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDType {
    F16,
    F32,
    BF16,
    U8Range,
    I8Range,
    F8E4M3,
    F8E5M2,
    /// OCP MX E8M0: 8-bit pure exponent, scale = 2^(byte - 127). No mantissa.
    /// Used by MXFP4 per-block scaling.
    E8M0,
}

#[derive(Debug, Clone)]
pub struct CodebookSpec {
    pub codebook_data: &'static [i8],
    pub vector_size: usize,
    pub bits_per_entry: u8,
}

#[derive(Debug, Clone)]
pub struct PackedScaleLayout {
    pub algorithm: PackedScaleAlgorithm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedScaleAlgorithm {
    /// Q4_K / Q5_K: 12-byte scales[12] decoded into 8 (sc, m) pairs.
    KQuant6Bit,
    /// Q3_K: hmask + scales[12] with extended decoding.
    Q3KExtended,
}

/// Registry of all known quantization formats.
pub struct QuantFormatRegistry {
    descriptors: HashMap<u32, QuantFormatDescriptor>,
}

fn quant_type_key(qt: &QuantType) -> u32 {
    match qt {
        QuantType::F32 => 0,
        QuantType::Bf16 => 50,
        QuantType::F16 => 51,
        QuantType::Q4_0 => 1,
        QuantType::Q4_1 => 2,
        QuantType::Q5_0 => 3,
        QuantType::Q5_1 => 4,
        QuantType::Q8_0 => 5,
        QuantType::Q8_1 => 6,
        QuantType::Q2K => 10,
        QuantType::Q3K => 11,
        QuantType::Q4K => 12,
        QuantType::Q5K => 13,
        QuantType::Q6K => 14,
        QuantType::Q8K => 15,
        QuantType::IQ1S => 20,
        QuantType::IQ1M => 21,
        QuantType::IQ2XXS => 22,
        QuantType::IQ2XS => 23,
        QuantType::IQ2S => 24,
        QuantType::IQ3XXS => 25,
        QuantType::IQ3S => 26,
        QuantType::IQ4NL => 27,
        QuantType::IQ4XS => 28,
        QuantType::AWQ4 => 30,
        QuantType::GPTQ4 => 31,
        QuantType::Squeeze => 32,
        QuantType::Mxfp4 { .. } => 33,
        QuantType::Nvfp4 => 34,
        QuantType::TQ1_0 => 35,
        QuantType::TQ2_0 => 36,
        QuantType::Fp8E4M3 => 37,
        QuantType::Fp8E5M2 => 38,
    }
}

impl QuantFormatRegistry {
    pub fn new() -> Self {
        let mut r = Self { descriptors: HashMap::new() };
        r.register_float();
        r.register_classic();
        r.register_kquant();
        r.register_iquant();
        r.register_iquant_extended();
        r.register_external();
        r
    }

    pub fn get(&self, qt: &QuantType) -> Option<&QuantFormatDescriptor> {
        self.descriptors.get(&quant_type_key(qt))
    }

    fn insert(&mut self, desc: QuantFormatDescriptor) {
        self.descriptors.insert(quant_type_key(&desc.quant_type), desc);
    }

    fn register_classic(&mut self) {
        // Q4_0: d(f16) + qs[16] = 18 bytes, 32 elements, nibble - 8
        // DecodeTraceBuilder applies the -8 bias via ZeroLayout::StaticBias.
        self.insert(QuantFormatDescriptor {
            name: "Q4_0",
            quant_type: QuantType::Q4_0,
            block_size: 32,
            block_bytes: 18,
            bits_per_element: 4,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::StaticBias { value: 8 },
            data_layout: DataLayout::PackedNibbles { offset: 2, low_first: true },
            data_kind: QuantDataKind::SignedPackedInt4,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT4: software-assisted on all ISAs
        });

        // Q4_1: d(f16) + m(f16) + qs[16] = 20 bytes
        // value = d * quantized + m  (BlockScalarWithMin scale + BlockMin zero)
        self.insert(QuantFormatDescriptor {
            name: "Q4_1",
            quant_type: QuantType::Q4_1,
            block_size: 32,
            block_bytes: 20,
            bits_per_element: 4,
            scale_layout: ScaleLayout::BlockScalarWithMin {
                d_offset: 0,
                m_offset: 2,
                dtype: ScaleDType::F16,
            },
            zero_layout: ZeroLayout::BlockMin { offset_bytes: 2, dtype: ScaleDType::F16 },
            data_layout: DataLayout::PackedNibbles { offset: 4, low_first: true },
            data_kind: QuantDataKind::PackedInt4,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT4: software-assisted on all ISAs
        });

        // Q5_0: d(f16) + qh[4] + qs[16] = 22 bytes, low 4 + high 1 bit, value - 16
        self.insert(QuantFormatDescriptor {
            name: "Q5_0",
            quant_type: QuantType::Q5_0,
            block_size: 32,
            block_bytes: 22,
            bits_per_element: 5,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::StaticBias { value: 16 },
            data_layout: DataLayout::NibbleWithHighBits {
                low_offset: 6,
                high_offset: 2,
                high_bits_per_elem: 1,
            },
            data_kind: QuantDataKind::PackedInt5,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT5: software-assisted on all ISAs
        });

        // Q5_1: d + m + qh[4] + qs[16] = 24 bytes
        // value = d * quantized + m  (BlockScalarWithMin scale + BlockMin zero)
        self.insert(QuantFormatDescriptor {
            name: "Q5_1",
            quant_type: QuantType::Q5_1,
            block_size: 32,
            block_bytes: 24,
            bits_per_element: 5,
            scale_layout: ScaleLayout::BlockScalarWithMin {
                d_offset: 0,
                m_offset: 2,
                dtype: ScaleDType::F16,
            },
            zero_layout: ZeroLayout::BlockMin { offset_bytes: 2, dtype: ScaleDType::F16 },
            data_layout: DataLayout::NibbleWithHighBits {
                low_offset: 8,
                high_offset: 4,
                high_bits_per_elem: 1,
            },
            data_kind: QuantDataKind::PackedInt5,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT5: software-assisted on all ISAs
        });

        // Q8_0: d(f16) + qs[32 i8] = 34 bytes
        self.insert(QuantFormatDescriptor {
            name: "Q8_0",
            quant_type: QuantType::Q8_0,
            block_size: 32,
            block_bytes: 34,
            bits_per_element: 8,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 2, signed: true },
            data_kind: QuantDataKind::Int8,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: Some(NativeIsa::Vpdpbusd), // INT8: VNNI VPDPBUSD native
        });

        // Q8_1: d(f16) + s(f16) + qs[32 i8] = 36 bytes
        // value = d * quantized + m  (BlockScalarWithMin scale + BlockMin zero)
        self.insert(QuantFormatDescriptor {
            name: "Q8_1",
            quant_type: QuantType::Q8_1,
            block_size: 32,
            block_bytes: 36,
            bits_per_element: 8,
            scale_layout: ScaleLayout::BlockScalarWithMin {
                d_offset: 0,
                m_offset: 2,
                dtype: ScaleDType::F16,
            },
            zero_layout: ZeroLayout::BlockMin { offset_bytes: 2, dtype: ScaleDType::F16 },
            data_layout: DataLayout::Bytes { offset: 4, signed: true },
            data_kind: QuantDataKind::Int8,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: Some(NativeIsa::Vpdpbusd), // INT8: VNNI VPDPBUSD native
        });
    }

    fn register_kquant(&mut self) {
        // Q4_K: d(f16) + dmin(f16) + scales[12] + qs[128] = 144 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "Q4_K",
            quant_type: QuantType::Q4K,
            block_size: 256,
            block_bytes: 144,
            bits_per_element: 4,
            scale_layout: ScaleLayout::Hierarchical {
                block_d_offset: 0,
                block_dmin_offset: Some(2),
                sub_scales_offset: 4,
                sub_scales_bits: 6,
                sub_scales_count: 8,
                sub_block_elements: 32,
                packed_layout: PackedScaleLayout {
                    algorithm: PackedScaleAlgorithm::KQuant6Bit,
                },
            },
            zero_layout: ZeroLayout::Hierarchical {
                dmin_offset: 2,
                sub_m_offset: 4,
            },
            data_layout: DataLayout::PackedNibbles { offset: 16, low_first: true },
            data_kind: QuantDataKind::PackedInt4,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT4 hierarchical: software-assisted
        });

        // Q5_K: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] = 176 bytes
        self.insert(QuantFormatDescriptor {
            name: "Q5_K",
            quant_type: QuantType::Q5K,
            block_size: 256,
            block_bytes: 176,
            bits_per_element: 5,
            scale_layout: ScaleLayout::Hierarchical {
                block_d_offset: 0,
                block_dmin_offset: Some(2),
                sub_scales_offset: 4,
                sub_scales_bits: 6,
                sub_scales_count: 8,
                sub_block_elements: 32,
                packed_layout: PackedScaleLayout {
                    algorithm: PackedScaleAlgorithm::KQuant6Bit,
                },
            },
            zero_layout: ZeroLayout::Hierarchical {
                dmin_offset: 2,
                sub_m_offset: 4,
            },
            data_layout: DataLayout::NibbleWithHighBits {
                low_offset: 48,
                high_offset: 16,
                high_bits_per_elem: 1,
            },
            data_kind: QuantDataKind::PackedInt5,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT5 hierarchical: software-assisted
        });

        // Q6_K: qs[128] + qh[64] + scales[16 i8] + d(f16) = 210 bytes
        // value = (low4 | (high2 << 4)) - 32
        self.insert(QuantFormatDescriptor {
            name: "Q6_K",
            quant_type: QuantType::Q6K,
            block_size: 256,
            block_bytes: 210,
            bits_per_element: 6,
            scale_layout: ScaleLayout::Q6KScales {
                block_d_offset: 208,
                sub_scales_offset: 192,
                sub_scales_count: 16,
                sub_block_elements: 16,
            },
            zero_layout: ZeroLayout::StaticBias { value: 32 },
            data_layout: DataLayout::NibbleWithHighBits {
                low_offset: 0,
                high_offset: 128,
                high_bits_per_elem: 2,
            },
            data_kind: QuantDataKind::PackedInt6,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT6: software-assisted on all ISAs
        });

        // Q2_K: scales[16] + qs[64] + d(f16) + dmin(f16) = 84 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "Q2_K",
            quant_type: QuantType::Q2K,
            block_size: 256,
            block_bytes: 84,
            bits_per_element: 2,
            scale_layout: ScaleLayout::Hierarchical {
                block_d_offset: 80,
                block_dmin_offset: Some(82),
                sub_scales_offset: 0,
                sub_scales_bits: 4,
                sub_scales_count: 16,
                sub_block_elements: 16,
                packed_layout: PackedScaleLayout { algorithm: PackedScaleAlgorithm::KQuant6Bit },
            },
            zero_layout: ZeroLayout::Hierarchical { dmin_offset: 82, sub_m_offset: 0 },
            data_layout: DataLayout::PackedNibbles { offset: 16, low_first: true },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 2-bit: DequantFMA on all ISAs
        });

        // Q3_K: hmask[32] + qs[64] + scales[12] + d(f16) = 110 bytes, 256 elements
        // 3-bit value = (qs 2-bit) | (hmask 1-bit as sign/high)
        // result = (3-bit_value - 4) × d × scale_6bit
        self.insert(QuantFormatDescriptor {
            name: "Q3_K",
            quant_type: QuantType::Q3K,
            block_size: 256,
            block_bytes: 110,
            bits_per_element: 3,
            scale_layout: ScaleLayout::Hierarchical {
                block_d_offset: 108,
                block_dmin_offset: None,
                sub_scales_offset: 96,
                sub_scales_bits: 6,
                sub_scales_count: 16,
                sub_block_elements: 16,
                packed_layout: PackedScaleLayout { algorithm: PackedScaleAlgorithm::Q3KExtended },
            },
            zero_layout: ZeroLayout::StaticBias { value: 4 },
            data_layout: DataLayout::TwoBitConditionalBias {
                qs_offset: 32,
                hmask_offset: 0,
            },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 3-bit conditional: DequantFMA on all ISAs
        });

        // Q8_K: d(f32) + qs[256 i8] + bsums[16 i16] = 292 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "Q8_K",
            quant_type: QuantType::Q8K,
            block_size: 256,
            block_bytes: 292,
            bits_per_element: 8,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F32 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 4, signed: true },
            data_kind: QuantDataKind::Int8,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: Some(NativeIsa::Vpdpbusd), // INT8: VNNI VPDPBUSD native
        });
    }

    fn register_iquant(&mut self) {
        // IQ4_NL: block_size=32, 18 bytes (same layout as Q4_0 but with non-linear codebook)
        self.insert(QuantFormatDescriptor {
            name: "IQ4_NL",
            quant_type: QuantType::IQ4NL,
            block_size: 32,
            block_bytes: 18,
            bits_per_element: 4,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 2, index_bits: 4 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: Some(CodebookSpec {
                codebook_data: &IQ4_NL_CODEBOOK,
                vector_size: 1,
                bits_per_entry: 4,
            }),
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // codebook-based: no native hardware
        });

        // IQ4_XS: block_size=256, 136 bytes
        self.insert(QuantFormatDescriptor {
            name: "IQ4_XS",
            quant_type: QuantType::IQ4XS,
            block_size: 256,
            block_bytes: 136,
            bits_per_element: 4,
            scale_layout: ScaleLayout::Hierarchical {
                block_d_offset: 0,
                block_dmin_offset: None,
                sub_scales_offset: 2,
                sub_scales_bits: 6,
                sub_scales_count: 8,
                sub_block_elements: 32,
                packed_layout: PackedScaleLayout { algorithm: PackedScaleAlgorithm::KQuant6Bit },
            },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 8, index_bits: 4 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: Some(CodebookSpec {
                codebook_data: &IQ4_NL_CODEBOOK,
                vector_size: 1,
                bits_per_entry: 4,
            }),
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // codebook-based: no native hardware
        });
    }

    fn register_iquant_extended(&mut self) {
        // IQ1_S: d(f16)[2] + qs[32] + qh[16 u16] + scales[16] = 50 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ1_S",
            quant_type: QuantType::IQ1S,
            block_size: 256,
            block_bytes: 50,
            bits_per_element: 1,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 34, index_bits: 1 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 1-bit: DequantFMA on all ISAs
        });

        // IQ1_M: qs[32] + qh[16] + scales[8] = 56 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ1_M",
            quant_type: QuantType::IQ1M,
            block_size: 256,
            block_bytes: 56,
            bits_per_element: 1,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 40, dtype: ScaleDType::U8Range },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 0, index_bits: 1 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 1-bit: DequantFMA on all ISAs
        });

        // IQ2_XXS: d(f16)[2] + qs[64] = 66 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ2_XXS",
            quant_type: QuantType::IQ2XXS,
            block_size: 256,
            block_bytes: 66,
            bits_per_element: 2,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 2, index_bits: 2 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 2-bit importance: DequantFMA on all ISAs
        });

        // IQ2_XS: d(f16)[2] + qs[64] + scales[8] = 74 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ2_XS",
            quant_type: QuantType::IQ2XS,
            block_size: 256,
            block_bytes: 74,
            bits_per_element: 2,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 2, index_bits: 2 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 2-bit importance: DequantFMA on all ISAs
        });

        // IQ2_S: d(f16)[2] + qs[64] + qh[8] + scales[8] = 82 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ2_S",
            quant_type: QuantType::IQ2S,
            block_size: 256,
            block_bytes: 82,
            bits_per_element: 2,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 2, index_bits: 2 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 2-bit importance: DequantFMA on all ISAs
        });

        // IQ3_XXS: d(f16)[2] + qs[96] = 98 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ3_XXS",
            quant_type: QuantType::IQ3XXS,
            block_size: 256,
            block_bytes: 98,
            bits_per_element: 3,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 2, index_bits: 3 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 3-bit importance: DequantFMA on all ISAs
        });

        // IQ3_S: d(f16)[2] + qs[64] + qh[8] + signs[32] + scales[4] = 110 bytes, 256 elements
        self.insert(QuantFormatDescriptor {
            name: "IQ3_S",
            quant_type: QuantType::IQ3S,
            block_size: 256,
            block_bytes: 110,
            bits_per_element: 3,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 106, dtype: ScaleDType::U8Range },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::CodebookIndex { offset: 2, index_bits: 3 },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 3-bit importance: DequantFMA on all ISAs
        });
    }

    fn register_external(&mut self) {
        // AWQ4: group_size=128, 72 bytes per group.
        // Layout: [scale(f16, 2B) | pad(2B) | zero(f16, 2B) | pad(2B) | qweight(64B)]
        self.insert(QuantFormatDescriptor {
            name: "AWQ4",
            quant_type: QuantType::AWQ4,
            block_size: 128,
            block_bytes: 72,
            bits_per_element: 4,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::BlockScalar { offset_bytes: 4, dtype: ScaleDType::F16 },
            data_layout: DataLayout::PackedNibbles { offset: 8, low_first: true },
            data_kind: QuantDataKind::PackedInt4,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // INT4: software-assisted on all ISAs
        });

        // GPTQ4: same layout as AWQ4 after g_idx column reordering during repacking.
        self.insert(QuantFormatDescriptor {
            name: "GPTQ4",
            quant_type: QuantType::GPTQ4,
            block_size: 128,
            block_bytes: 72,
            bits_per_element: 4,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::BlockScalar { offset_bytes: 4, dtype: ScaleDType::F16 },
            data_layout: DataLayout::PackedNibbles { offset: 8, low_first: true },
            data_kind: QuantDataKind::PackedInt4,
            codebook: None,
            storage_layout: StorageLayout::ColInterleaved, // g_idx reordering
            native_isa: None, // INT4: software-assisted on all ISAs
        });

        // SqueezeLLM: d(f16) + qs[128] = 130 bytes, 256 elements
        // 3-bit packed values (256*3/8=96 bytes used from qs[128]).
        // Decode: out[i] = d * (q3 - 4.0) where q3 is a 3-bit value [0..7].
        // Linear quantization with range [-4d, +3d].
        self.insert(QuantFormatDescriptor {
            name: "Squeeze",
            quant_type: QuantType::Squeeze,
            block_size: 256,
            block_bytes: 130,
            bits_per_element: 3,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::StaticBias { value: 4 },
            data_layout: DataLayout::PackedNibbles { offset: 2, low_first: true },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // 3-bit codebook: DequantFMA on all ISAs
        });

        // MXFP4 (OCP standard): 1 byte e8m0 scale + packed e2m1 data
        // block_size configurable (default 32). block_bytes = 1 + block_size/2.
        // e2m1: 4-bit float with 2 exp bits + 1 mantissa bit + 1 sign bit.
        // E8M0 scale: pure exponent format, scale = 2^(byte_value - 127).
        self.insert(QuantFormatDescriptor {
            name: "MXFP4",
            quant_type: QuantType::Mxfp4 { block_size: 32 },
            block_size: 32,
            block_bytes: 17, // 1 byte scale + 32/2 = 16 bytes packed
            bits_per_element: 4,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::E8M0 },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::PackedNibbles { offset: 1, low_first: true },
            data_kind: QuantDataKind::Float4,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // OCP standard: no single canonical native ISA
        });

        // NVFP4 (NVIDIA): 64 elements, 36 bytes.
        // 4 bytes UE4M3 sub-block scales (FP8 E4M3 unsigned, one per 16-element sub-block)
        // + 32 bytes packed e2m1 (same encoding as MXFP4).
        // Two-level scaling: global_f32 * ue4m3_sub_block_scale * e2m1_lookup[qs]
        self.insert(QuantFormatDescriptor {
            name: "NVFP4",
            quant_type: QuantType::Nvfp4,
            block_size: 64,
            block_bytes: 36,
            bits_per_element: 4,
            scale_layout: ScaleLayout::SubBlockScalars {
                offset_bytes: 0,
                sub_block_size: 16,
                dtype: ScaleDType::F8E4M3,
            },
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::PackedNibbles { offset: 4, low_first: true },
            data_kind: QuantDataKind::Nvfp4,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: Some(NativeIsa::Tcgen05Fp4), // SM100+ Blackwell FP4 tensor core
        });

        // TQ1_0 (Ternary 1.0, GGUF type 34): 256 elements, 54 bytes per block.
        // Layout: d: f16 (offset 0, 2B) + qs: [u8; 52] packed ternary (offset 2).
        // 5-trit packing: 3 base-3 digits in 5 low bits + 2 base-3 digits as MSB across 5 bytes.
        // Decode (linear, simplified): value = d * (trit - 1.0), trit ∈ {0, 1, 2}.
        self.insert(QuantFormatDescriptor {
            name: "TQ1_0",
            quant_type: QuantType::TQ1_0,
            block_size: 256,
            block_bytes: 54,
            bits_per_element: 3,  // log2(3) ≈ 1.58 → engineering aligned to 3-bit upper bound
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::StaticBias { value: 1 },  // trit - 1 (symmetric)
            data_layout: DataLayout::PackedNibbles { offset: 2, low_first: true },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::Packed, // 5-trit packing across byte boundaries
            native_isa: None, // ternary: no native hardware
        });

        // TQ2_0 (Ternary 2.0, GGUF type 35): 256 elements, 66 bytes per block.
        // Layout: d: f16 (offset 0, 2B) + qs: [u8; 64] 2-bit packed ternary (offset 2).
        // Each byte stores 4 ternary values (2 bits each); decode: value = d * (q2 - 1.0).
        self.insert(QuantFormatDescriptor {
            name: "TQ2_0",
            quant_type: QuantType::TQ2_0,
            block_size: 256,
            block_bytes: 66,
            bits_per_element: 2,
            scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
            zero_layout: ZeroLayout::StaticBias { value: 1 },
            data_layout: DataLayout::PackedNibbles { offset: 2, low_first: true },
            data_kind: QuantDataKind::SuperLowBit,
            codebook: None,
            storage_layout: StorageLayout::Packed, // 2-bit packed ternary
            native_isa: None, // ternary: no native hardware
        });

        // FP8 E4M3 (NVIDIA/AMD hardware-native 8-bit float).
        // Per element: 1 byte, no scale, no zero, no block structure.
        // Range: ±448, no NaN (inf maps to max mantissa).
        // Used in .gllm (SPEC 36) for 2:1 weight compression vs FP16.
        self.insert(QuantFormatDescriptor {
            name: "FP8_E4M3",
            quant_type: QuantType::Fp8E4M3,
            block_size: 1,
            block_bytes: 1,
            bits_per_element: 8,
            scale_layout: ScaleLayout::None,
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 0, signed: false },
            data_kind: QuantDataKind::Float8,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // FP8 dot-product requires hardware-specific lowering (SM90+ HMMA fp8 / MI300 MFMA fp8)
        });

        // FP8 E5M2 (wider range, lower precision).
        // Per element: 1 byte, no scale, no zero, no block structure.
        // Range: ±57344, supports NaN/Inf.
        self.insert(QuantFormatDescriptor {
            name: "FP8_E5M2",
            quant_type: QuantType::Fp8E5M2,
            block_size: 1,
            block_bytes: 1,
            bits_per_element: 8,
            scale_layout: ScaleLayout::None,
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 0, signed: false },
            data_kind: QuantDataKind::Float8,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None,
        });
    }

    fn register_float(&mut self) {
        // BF16: native float, block_size=1, no scale/zero
        self.insert(QuantFormatDescriptor {
            name: "BF16",
            quant_type: QuantType::Bf16,
            block_size: 1,
            block_bytes: 2,
            bits_per_element: 16,
            scale_layout: ScaleLayout::None,
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 0, signed: false },
            data_kind: QuantDataKind::Bfloat16,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: Some(NativeIsa::Vdpbf16ps), // VDPBF16PS native dot-product
        });

        // FP16: native float, block_size=1, no scale/zero
        self.insert(QuantFormatDescriptor {
            name: "FP16",
            quant_type: QuantType::F16,
            block_size: 1,
            block_bytes: 2,
            bits_per_element: 16,
            scale_layout: ScaleLayout::None,
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 0, signed: false },
            data_kind: QuantDataKind::Float16,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: Some(NativeIsa::Fmmla), // FMMLA native dot-product
        });

        // F32: baseline float, block_size=1, no scale/zero
        self.insert(QuantFormatDescriptor {
            name: "F32",
            quant_type: QuantType::F32,
            block_size: 1,
            block_bytes: 4,
            bits_per_element: 32,
            scale_layout: ScaleLayout::None,
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 0, signed: false },
            data_kind: QuantDataKind::Float32,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None, // F32 FMA is universal — no single canonical native ISA
        });
    }
}

/// IQ4_NL non-linear codebook (16 entries, maps 4-bit index → signed value).
/// From llama.cpp ggml-quants.c: kvalues_iq4nl[16].
static IQ4_NL_CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Process-wide singleton registry (lazy-initialized).
static GLOBAL_REGISTRY: OnceLock<QuantFormatRegistry> = OnceLock::new();

pub fn registry() -> &'static QuantFormatRegistry {
    GLOBAL_REGISTRY.get_or_init(QuantFormatRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_formats_registered() {
        let r = registry();
        for qt in &[
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0,
            QuantType::Q5_1, QuantType::Q8_0, QuantType::Q8_1,
        ] {
            let d = r.get(qt).expect("classic format must be registered");
            assert_eq!(d.block_size, 32);
            assert_eq!(d.block_bytes, qt.block_bytes());
        }
    }

    #[test]
    fn kquant_formats_registered() {
        let r = registry();
        for qt in &[QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K, QuantType::Q6K, QuantType::Q8K] {
            let d = r.get(qt).expect("k-quant format must be registered");
            assert_eq!(d.block_size, 256);
            assert_eq!(d.block_bytes, qt.block_bytes());
        }
    }

    #[test]
    fn iq_formats_registered() {
        let r = registry();
        for qt in &[QuantType::IQ4NL, QuantType::IQ4XS] {
            let d = r.get(qt).expect("IQ format must be registered");
            assert!(d.codebook.is_some());
        }
    }

    /// REQ-QCG-008: All 9 IQ QuantAlgoKind variants map to correct QuantType.
    #[test]
    fn iq_algo_kind_coverage() {
        use crate::compiler::quant_format::QuantAlgoKind;
        let iq_kinds = [
            QuantAlgoKind::IQ1S, QuantAlgoKind::IQ1M,
            QuantAlgoKind::IQ2XXS, QuantAlgoKind::IQ2XS, QuantAlgoKind::IQ2S,
            QuantAlgoKind::IQ3XXS, QuantAlgoKind::IQ3S,
            QuantAlgoKind::IQ4NL, QuantAlgoKind::IQ4XS,
        ];
        assert_eq!(iq_kinds.len(), 9, "all 9 IQ QuantAlgoKind variants present");
        for kind in &iq_kinds {
            let desc = kind.descriptor();
            assert!(
                matches!(desc.data_kind, QuantDataKind::SuperLowBit),
                "{:?}: expected SuperLowBit data_kind, got {:?}",
                kind, desc.data_kind,
            );
        }
        // All 9 must appear in QuantAlgoKind::all()
        let all = QuantAlgoKind::all();
        for kind in &iq_kinds {
            assert!(all.contains(kind), "QuantAlgoKind::all() missing {:?}", kind);
        }
    }

    #[test]
    fn external_formats_registered() {
        let r = registry();
        for qt in &[QuantType::AWQ4, QuantType::GPTQ4] {
            let d = r.get(qt).expect("external format must be registered");
            assert_eq!(d.bits_per_element, 4);
        }
    }

    /// REQ-QCG-001: All 31 QuantType variants must have a QuantFormatDescriptor.
    #[test]
    fn all_quant_types_have_descriptor() {
        let r = registry();
        let all_qt = [
            QuantType::F32, QuantType::Bf16, QuantType::F16,
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1,
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K,
            QuantType::Q6K, QuantType::Q8K,
            QuantType::IQ1S, QuantType::IQ1M, QuantType::IQ2XXS, QuantType::IQ2XS,
            QuantType::IQ2S, QuantType::IQ3XXS, QuantType::IQ3S,
            QuantType::IQ4NL, QuantType::IQ4XS,
            QuantType::AWQ4, QuantType::GPTQ4,
            QuantType::Squeeze,
            QuantType::Mxfp4 { block_size: 32 },
            QuantType::Nvfp4,
            QuantType::TQ1_0, QuantType::TQ2_0,
        ];
        assert_eq!(all_qt.len(), 31);
        for qt in &all_qt {
            let desc = r.get(qt).unwrap_or_else(|| panic!("missing descriptor for {:?}", qt));
            assert_eq!(desc.block_size, qt.block_size(),
                "{:?}: descriptor.block_size {} != qt.block_size() {}", qt, desc.block_size, qt.block_size());
            assert_eq!(desc.block_bytes, qt.block_bytes(),
                "{:?}: descriptor.block_bytes {} != qt.block_bytes() {}", qt, desc.block_bytes, qt.block_bytes());
            assert_eq!(desc.bits_per_element, qt.bits(),
                "{:?}: descriptor.bits {} != qt.bits() {}", qt, desc.bits_per_element, qt.bits());
        }
    }

    /// Verify NVFP4 descriptor: 4 UE4M3 sub-block scales + E2M1 data, two-level scaling.
    #[test]
    fn nvfp4_descriptor_layout() {
        let r = registry();
        let d = r.get(&QuantType::Nvfp4).expect("NVFP4 registered");
        assert_eq!(d.block_size, 64);
        assert_eq!(d.block_bytes, 36);
        assert!(matches!(d.data_kind, QuantDataKind::Nvfp4));
        assert!(matches!(d.zero_layout, ZeroLayout::None));
        assert!(matches!(d.scale_layout, ScaleLayout::SubBlockScalars { .. }));
    }

    /// Verify Squeeze descriptor: 3-bit linear with StaticBias(4).
    #[test]
    fn squeeze_descriptor_layout() {
        let r = registry();
        let d = r.get(&QuantType::Squeeze).expect("Squeeze registered");
        assert_eq!(d.block_size, 256);
        assert_eq!(d.block_bytes, 130);
        assert_eq!(d.bits_per_element, 3);
        assert!(matches!(d.data_kind, QuantDataKind::SuperLowBit));
        assert!(matches!(d.zero_layout, ZeroLayout::StaticBias { value: 4 }));
    }

    /// Verify TQ1_0/TQ2_0 descriptors: ternary with StaticBias(1).
    #[test]
    fn ternary_descriptor_layout() {
        let r = registry();
        let tq10 = r.get(&QuantType::TQ1_0).expect("TQ1_0 registered");
        assert_eq!(tq10.block_bytes, 54);
        assert!(matches!(tq10.data_kind, QuantDataKind::SuperLowBit));
        assert!(matches!(tq10.zero_layout, ZeroLayout::StaticBias { value: 1 }));

        let tq20 = r.get(&QuantType::TQ2_0).expect("TQ2_0 registered");
        assert_eq!(tq20.block_bytes, 66);
        assert!(matches!(tq20.data_kind, QuantDataKind::SuperLowBit));
        assert!(matches!(tq20.zero_layout, ZeroLayout::StaticBias { value: 1 }));
    }

    /// Verify BF16/FP16/F32 descriptors: native float, no scale/zero.
    #[test]
    fn native_float_descriptors() {
        let r = registry();
        for (qt, bytes, kind) in [
            (QuantType::Bf16, 2, QuantDataKind::Bfloat16),
            (QuantType::F16, 2, QuantDataKind::Float16),
            (QuantType::F32, 4, QuantDataKind::Float32),
        ] {
            let d = r.get(&qt).unwrap();
            assert_eq!(d.block_size, 1);
            assert_eq!(d.block_bytes, bytes);
            assert!(matches!(d.scale_layout, ScaleLayout::None));
            assert!(matches!(d.zero_layout, ZeroLayout::None));
            assert!(matches!(d.data_kind, kind));
        }
    }

    // ========== StorageLayout tests ==========

    #[test]
    fn storage_layout_equality_and_copy() {
        let a = StorageLayout::RowMajor;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(StorageLayout::RowMajor, StorageLayout::RowMajor);
        assert_ne!(StorageLayout::RowMajor, StorageLayout::ColInterleaved);
        assert_ne!(StorageLayout::RowMajor, StorageLayout::Packed);
        assert_ne!(StorageLayout::ColInterleaved, StorageLayout::Packed);
    }

    #[test]
    fn storage_layout_all_variants_distinct() {
        let variants = [StorageLayout::RowMajor, StorageLayout::ColInterleaved, StorageLayout::Packed];
        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                assert_eq!(i == j, v1 == v2, "variant {} == {} mismatch", i, j);
            }
        }
    }

    // ========== NativeIsa tests ==========

    #[test]
    fn native_isa_equality() {
        assert_eq!(NativeIsa::Vdpbf16ps, NativeIsa::Vdpbf16ps);
        assert_eq!(NativeIsa::Tcgen05Fp4, NativeIsa::Tcgen05Fp4);
        assert_ne!(NativeIsa::Vdpbf16ps, NativeIsa::Vpdpbusd);
        assert_ne!(NativeIsa::Hmma, NativeIsa::Wgmma);
    }

    #[test]
    fn native_isa_x86_variants_distinct() {
        let x86 = [
            NativeIsa::Vdpbf16ps,
            NativeIsa::Vpdpbusd,
            NativeIsa::Vfmadd,
            NativeIsa::Tdpbf16ps,
            NativeIsa::Tdpbssd,
        ];
        for (i, a) in x86.iter().enumerate() {
            for (j, b) in x86.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn native_isa_arm_variants_distinct() {
        let arm = [
            NativeIsa::Bfmmla,
            NativeIsa::Fmmla,
            NativeIsa::Fmla,
            NativeIsa::Sdot,
        ];
        for (i, a) in arm.iter().enumerate() {
            for (j, b) in arm.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn native_isa_gpu_variants_distinct() {
        let gpu = [NativeIsa::Hmma, NativeIsa::Wgmma, NativeIsa::Tcgen05Fp4, NativeIsa::Wmma, NativeIsa::Gfx12WmmaIU4IU8];
        for (i, a) in gpu.iter().enumerate() {
            for (j, b) in gpu.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    // ========== QuantDataKind tests ==========

    #[test]
    fn quant_data_kind_is_float_true_cases() {
        assert!(QuantDataKind::Bfloat16.is_float());
        assert!(QuantDataKind::Float16.is_float());
        assert!(QuantDataKind::Float32.is_float());
    }

    #[test]
    fn quant_data_kind_is_float_false_cases() {
        let non_float = [
            QuantDataKind::Int8,
            QuantDataKind::PackedInt4,
            QuantDataKind::SignedPackedInt4,
            QuantDataKind::PackedInt5,
            QuantDataKind::PackedInt6,
            QuantDataKind::Float4,
            QuantDataKind::Float8,
            QuantDataKind::SuperLowBit,
            QuantDataKind::Nvfp4,
            QuantDataKind::Codebook,
        ];
        for kind in &non_float {
            assert!(!kind.is_float(), "{:?} should not be float", kind);
        }
    }

    #[test]
    fn quant_data_kind_equality() {
        assert_eq!(QuantDataKind::Int8, QuantDataKind::Int8);
        assert_eq!(QuantDataKind::Nvfp4, QuantDataKind::Nvfp4);
        assert_ne!(QuantDataKind::PackedInt4, QuantDataKind::SignedPackedInt4);
        assert_ne!(QuantDataKind::Float4, QuantDataKind::Float8);
    }

    // ========== ScaleDType tests ==========

    #[test]
    fn scale_dtype_equality() {
        assert_eq!(ScaleDType::F16, ScaleDType::F16);
        assert_eq!(ScaleDType::F32, ScaleDType::F32);
        assert_eq!(ScaleDType::E8M0, ScaleDType::E8M0);
        assert_ne!(ScaleDType::F16, ScaleDType::F32);
        assert_ne!(ScaleDType::F8E4M3, ScaleDType::F8E5M2);
        assert_ne!(ScaleDType::U8Range, ScaleDType::I8Range);
    }

    #[test]
    fn scale_dtype_all_variants_distinct() {
        let all = [
            ScaleDType::F16, ScaleDType::F32, ScaleDType::BF16,
            ScaleDType::U8Range, ScaleDType::I8Range,
            ScaleDType::F8E4M3, ScaleDType::F8E5M2, ScaleDType::E8M0,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "ScaleDType index {} vs {}", i, j);
            }
        }
    }

    // ========== ZeroLayout tests ==========

    #[test]
    fn zero_layout_none_variants() {
        assert_eq!(ZeroLayout::None, ZeroLayout::None);
    }

    #[test]
    fn zero_layout_static_bias_values() {
        let bias8 = ZeroLayout::StaticBias { value: 8 };
        let bias16 = ZeroLayout::StaticBias { value: 16 };
        let bias32 = ZeroLayout::StaticBias { value: 32 };
        assert_eq!(bias8, ZeroLayout::StaticBias { value: 8 });
        assert_ne!(bias8, bias16);
        assert_ne!(bias16, bias32);
    }

    #[test]
    fn zero_layout_block_scalar() {
        let layout = ZeroLayout::BlockScalar { offset_bytes: 4, dtype: ScaleDType::F16 };
        assert!(matches!(&layout, ZeroLayout::BlockScalar { offset_bytes: 4, dtype: ScaleDType::F16 }));
        assert!(!matches!(&layout, ZeroLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 }));
    }

    // ========== DataLayout tests ==========

    #[test]
    fn data_layout_packed_nibbles_fields() {
        let layout = DataLayout::PackedNibbles { offset: 2, low_first: true };
        assert!(matches!(&layout, DataLayout::PackedNibbles { offset: 2, low_first: true }));
        assert!(!matches!(&layout, DataLayout::PackedNibbles { offset: 2, low_first: false }));
    }

    #[test]
    fn data_layout_bytes_signed_vs_unsigned() {
        let signed = DataLayout::Bytes { offset: 0, signed: true };
        let unsigned = DataLayout::Bytes { offset: 0, signed: false };
        assert!(matches!(&signed, DataLayout::Bytes { signed: true, .. }));
        assert!(matches!(&unsigned, DataLayout::Bytes { signed: false, .. }));
    }

    #[test]
    fn data_layout_codebook_index_bits() {
        let idx1 = DataLayout::CodebookIndex { offset: 34, index_bits: 1 };
        let idx2 = DataLayout::CodebookIndex { offset: 2, index_bits: 2 };
        let idx3 = DataLayout::CodebookIndex { offset: 2, index_bits: 3 };
        let idx4 = DataLayout::CodebookIndex { offset: 2, index_bits: 4 };
        assert!(matches!(&idx1, DataLayout::CodebookIndex { index_bits: 1, .. }));
        assert!(matches!(&idx2, DataLayout::CodebookIndex { index_bits: 2, .. }));
        assert!(matches!(&idx3, DataLayout::CodebookIndex { index_bits: 3, .. }));
        assert!(matches!(&idx4, DataLayout::CodebookIndex { index_bits: 4, .. }));
    }

    // ========== PackedScaleAlgorithm tests ==========

    #[test]
    fn packed_scale_algorithm_variants() {
        let kq = PackedScaleAlgorithm::KQuant6Bit;
        let q3 = PackedScaleAlgorithm::Q3KExtended;
        // Must be constructible and matchable
        assert!(matches!(kq, PackedScaleAlgorithm::KQuant6Bit));
        assert!(matches!(q3, PackedScaleAlgorithm::Q3KExtended));
    }

    // ========== ScaleLayout variant coverage tests ==========

    #[test]
    fn scale_layout_none_for_float_formats() {
        let r = registry();
        for qt in &[QuantType::Bf16, QuantType::F16, QuantType::F32] {
            let d = r.get(qt).unwrap();
            assert!(matches!(d.scale_layout, ScaleLayout::None), "{:?} should have ScaleLayout::None", qt);
        }
    }

    #[test]
    fn scale_layout_block_scalar_for_classic() {
        let r = registry();
        for qt in &[QuantType::Q4_0, QuantType::Q5_0, QuantType::Q8_0] {
            let d = r.get(qt).unwrap();
            assert!(matches!(d.scale_layout, ScaleLayout::BlockScalar { .. }),
                "{:?} should have BlockScalar scale", qt);
        }
    }

    #[test]
    fn scale_layout_block_scalar_with_min_for_q41_q51_q81() {
        let r = registry();
        for qt in &[QuantType::Q4_1, QuantType::Q5_1, QuantType::Q8_1] {
            let d = r.get(qt).unwrap();
            assert!(matches!(d.scale_layout, ScaleLayout::BlockScalarWithMin { .. }),
                "{:?} should have BlockScalarWithMin scale", qt);
        }
    }

    #[test]
    fn scale_layout_hierarchical_for_kquant() {
        let r = registry();
        for qt in &[QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K] {
            let d = r.get(qt).unwrap();
            assert!(matches!(d.scale_layout, ScaleLayout::Hierarchical { .. }),
                "{:?} should have Hierarchical scale", qt);
        }
    }

    #[test]
    fn scale_layout_q6k_scales() {
        let r = registry();
        let d = r.get(&QuantType::Q6K).unwrap();
        assert!(matches!(d.scale_layout, ScaleLayout::Q6KScales { .. }));
    }

    #[test]
    fn scale_layout_sub_block_scalars_for_nvfp4() {
        let r = registry();
        let d = r.get(&QuantType::Nvfp4).unwrap();
        match &d.scale_layout {
            ScaleLayout::SubBlockScalars { offset_bytes, sub_block_size, dtype } => {
                assert_eq!(*offset_bytes, 0);
                assert_eq!(*sub_block_size, 16);
                assert_eq!(*dtype, ScaleDType::F8E4M3);
            }
            other => panic!("expected SubBlockScalars, got {:?}", other),
        }
    }

    // ========== QuantFormatDescriptor::for_type tests ==========

    #[test]
    fn for_type_returns_registered_descriptor() {
        let desc = QuantFormatDescriptor::for_type(QuantType::Q4_0);
        assert_eq!(desc.name, "Q4_0");
        assert_eq!(desc.quant_type, QuantType::Q4_0);
        assert_eq!(desc.block_size, 32);
        assert_eq!(desc.block_bytes, 18);
        assert_eq!(desc.bits_per_element, 4);
    }

    #[test]
    fn for_type_consistent_with_registry_get() {
        let r = registry();
        for qt in [QuantType::Bf16, QuantType::Q8_0, QuantType::Nvfp4, QuantType::TQ1_0] {
            let via_for_type = QuantFormatDescriptor::for_type(qt);
            let via_registry = r.get(&qt).unwrap();
            assert_eq!(via_for_type.name, via_registry.name);
            assert_eq!(via_for_type.block_size, via_registry.block_size);
            assert_eq!(via_for_type.block_bytes, via_registry.block_bytes);
        }
    }

    // ========== QuantFormatRegistry tests ==========

    #[test]
    fn registry_new_returns_populated() {
        let r = QuantFormatRegistry::new();
        // Spot-check a few known types
        assert!(r.get(&QuantType::F32).is_some());
        assert!(r.get(&QuantType::Q4_0).is_some());
        assert!(r.get(&QuantType::Nvfp4).is_some());
        assert!(r.get(&QuantType::TQ2_0).is_some());
    }

    // ========== Codebook presence tests ==========

    #[test]
    fn codebook_present_for_iq4nl_and_iq4xs() {
        let r = registry();
        for qt in &[QuantType::IQ4NL, QuantType::IQ4XS] {
            let d = r.get(qt).unwrap();
            let cb = d.codebook.as_ref().expect("IQ4 formats must have codebook");
            assert_eq!(cb.bits_per_entry, 4);
            assert_eq!(cb.vector_size, 1);
            assert_eq!(cb.codebook_data.len(), 16);
        }
    }

    #[test]
    fn codebook_absent_for_classic_and_kquant() {
        let r = registry();
        for qt in &[
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1,
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K,
            QuantType::Q6K, QuantType::Q8K,
        ] {
            let d = r.get(qt).unwrap();
            assert!(d.codebook.is_none(), "{:?} should have no codebook", qt);
        }
    }

    #[test]
    fn codebook_absent_for_external_formats() {
        let r = registry();
        for qt in &[QuantType::AWQ4, QuantType::GPTQ4, QuantType::Squeeze, QuantType::Nvfp4] {
            let d = r.get(qt).unwrap();
            assert!(d.codebook.is_none(), "{:?} should have no codebook", qt);
        }
    }

    #[test]
    fn codebook_absent_for_float_formats() {
        let r = registry();
        for qt in &[QuantType::Bf16, QuantType::F16, QuantType::F32] {
            let d = r.get(qt).unwrap();
            assert!(d.codebook.is_none());
        }
    }

    #[test]
    fn codebook_absent_for_fp8_formats() {
        let r = registry();
        for qt in &[QuantType::Fp8E4M3, QuantType::Fp8E5M2] {
            let d = r.get(qt).unwrap();
            assert!(d.codebook.is_none());
        }
    }

    // ========== CodebookSpec data validation ==========

    #[test]
    fn iq4nl_codebook_data_known_values() {
        let r = registry();
        let d = r.get(&QuantType::IQ4NL).unwrap();
        let cb = d.codebook.as_ref().unwrap();
        // First and last entries from the known IQ4_NL codebook
        assert_eq!(cb.codebook_data[0], -127);
        assert_eq!(cb.codebook_data[15], 113);
        assert_eq!(cb.codebook_data.len(), 16);
    }

    // ========== Storage layout per format ==========

    #[test]
    fn gptq4_is_col_interleaved() {
        let r = registry();
        let d = r.get(&QuantType::GPTQ4).unwrap();
        assert_eq!(d.storage_layout, StorageLayout::ColInterleaved);
    }

    #[test]
    fn awq4_is_row_major_not_col_interleaved() {
        let r = registry();
        let d = r.get(&QuantType::AWQ4).unwrap();
        assert_eq!(d.storage_layout, StorageLayout::RowMajor);
    }

    #[test]
    fn ternary_formats_are_packed() {
        let r = registry();
        let tq10 = r.get(&QuantType::TQ1_0).unwrap();
        let tq20 = r.get(&QuantType::TQ2_0).unwrap();
        assert_eq!(tq10.storage_layout, StorageLayout::Packed);
        assert_eq!(tq20.storage_layout, StorageLayout::Packed);
    }

    // ========== native_isa presence tests ==========

    #[test]
    fn native_isa_present_for_int8_formats() {
        let r = registry();
        for qt in &[QuantType::Q8_0, QuantType::Q8_1, QuantType::Q8K] {
            let d = r.get(qt).unwrap();
            assert_eq!(d.native_isa, Some(NativeIsa::Vpdpbusd),
                "{:?} should have VPDPBUSD native ISA", qt);
        }
    }

    #[test]
    fn native_isa_present_for_bf16() {
        let d = QuantFormatDescriptor::for_type(QuantType::Bf16);
        assert_eq!(d.native_isa, Some(NativeIsa::Vdpbf16ps));
    }

    #[test]
    fn native_isa_present_for_fp16() {
        let d = QuantFormatDescriptor::for_type(QuantType::F16);
        assert_eq!(d.native_isa, Some(NativeIsa::Fmmla));
    }

    #[test]
    fn native_isa_present_for_nvfp4() {
        let d = QuantFormatDescriptor::for_type(QuantType::Nvfp4);
        assert_eq!(d.native_isa, Some(NativeIsa::Tcgen05Fp4));
    }

    #[test]
    fn native_isa_none_for_low_bit_formats() {
        let r = registry();
        for qt in &[QuantType::Q4_0, QuantType::Q5_0, QuantType::Q6K, QuantType::Q2K, QuantType::Q3K] {
            let d = r.get(qt).unwrap();
            assert!(d.native_isa.is_none(), "{:?} should have no native ISA", qt);
        }
    }

    #[test]
    fn native_isa_none_for_fp8() {
        let r = registry();
        for qt in &[QuantType::Fp8E4M3, QuantType::Fp8E5M2] {
            let d = r.get(qt).unwrap();
            assert!(d.native_isa.is_none());
        }
    }

    // ========== FP8 descriptor details ==========

    #[test]
    fn fp8_e4m3_descriptor() {
        let d = QuantFormatDescriptor::for_type(QuantType::Fp8E4M3);
        assert_eq!(d.name, "FP8_E4M3");
        assert_eq!(d.block_size, 1);
        assert_eq!(d.block_bytes, 1);
        assert_eq!(d.bits_per_element, 8);
        assert!(matches!(d.data_kind, QuantDataKind::Float8));
        assert!(matches!(d.data_layout, DataLayout::Bytes { offset: 0, signed: false }));
    }

    #[test]
    fn fp8_e5m2_descriptor() {
        let d = QuantFormatDescriptor::for_type(QuantType::Fp8E5M2);
        assert_eq!(d.name, "FP8_E5M2");
        assert_eq!(d.block_size, 1);
        assert_eq!(d.block_bytes, 1);
        assert!(matches!(d.data_kind, QuantDataKind::Float8));
    }

    // ========== IQ extended format coverage ==========

    #[test]
    fn iq1s_descriptor_fields() {
        let d = QuantFormatDescriptor::for_type(QuantType::IQ1S);
        assert_eq!(d.bits_per_element, 1);
        assert_eq!(d.block_size, 256);
        assert!(matches!(d.data_layout, DataLayout::CodebookIndex { index_bits: 1, .. }));
    }

    #[test]
    fn iq2xxs_descriptor_fields() {
        let d = QuantFormatDescriptor::for_type(QuantType::IQ2XXS);
        assert_eq!(d.bits_per_element, 2);
        assert!(matches!(d.data_layout, DataLayout::CodebookIndex { index_bits: 2, .. }));
    }

    #[test]
    fn iq3xxs_descriptor_fields() {
        let d = QuantFormatDescriptor::for_type(QuantType::IQ3XXS);
        assert_eq!(d.bits_per_element, 3);
        assert!(matches!(d.data_layout, DataLayout::CodebookIndex { index_bits: 3, .. }));
    }

    // ========== MXFP4 descriptor details ==========

    #[test]
    fn mxfp4_descriptor() {
        let d = QuantFormatDescriptor::for_type(QuantType::Mxfp4 { block_size: 32 });
        assert_eq!(d.name, "MXFP4");
        assert_eq!(d.block_size, 32);
        assert_eq!(d.block_bytes, 17);
        assert_eq!(d.bits_per_element, 4);
        assert!(matches!(d.data_kind, QuantDataKind::Float4));
        assert!(matches!(d.scale_layout, ScaleLayout::BlockScalar { dtype: ScaleDType::E8M0, .. }));
    }

    // ========== QuantType round-trip through registry ==========

    #[test]
    fn registry_round_trip_all_registered() {
        let all_qt = [
            QuantType::F32, QuantType::Bf16, QuantType::F16,
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1,
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K,
            QuantType::Q6K, QuantType::Q8K,
            QuantType::IQ1S, QuantType::IQ1M, QuantType::IQ2XXS, QuantType::IQ2XS,
            QuantType::IQ2S, QuantType::IQ3XXS, QuantType::IQ3S,
            QuantType::IQ4NL, QuantType::IQ4XS,
            QuantType::AWQ4, QuantType::GPTQ4,
            QuantType::Squeeze,
            QuantType::Mxfp4 { block_size: 32 },
            QuantType::Nvfp4,
            QuantType::TQ1_0, QuantType::TQ2_0,
            QuantType::Fp8E4M3, QuantType::Fp8E5M2,
        ];
        // 33 types (31 from original test + 2 FP8)
        assert!(all_qt.len() >= 31);
        let r = QuantFormatRegistry::new();
        for qt in &all_qt {
            assert!(r.get(qt).is_some(), "fresh registry missing {:?}", qt);
        }
    }

    // ========== Hierarchical scale field validation ==========

    #[test]
    fn q4k_hierarchical_scale_fields() {
        let r = registry();
        let d = r.get(&QuantType::Q4K).unwrap();
        match &d.scale_layout {
            ScaleLayout::Hierarchical {
                block_d_offset, block_dmin_offset, sub_scales_offset,
                sub_scales_bits, sub_scales_count, sub_block_elements,
                packed_layout,
            } => {
                assert_eq!(*block_d_offset, 0);
                assert_eq!(*block_dmin_offset, Some(2));
                assert_eq!(*sub_scales_offset, 4);
                assert_eq!(*sub_scales_bits, 6);
                assert_eq!(*sub_scales_count, 8);
                assert_eq!(*sub_block_elements, 32);
                assert!(matches!(packed_layout.algorithm, PackedScaleAlgorithm::KQuant6Bit));
            }
            other => panic!("expected Hierarchical, got {:?}", other),
        }
    }

    #[test]
    fn q3k_hierarchical_scale_no_dmin() {
        let r = registry();
        let d = r.get(&QuantType::Q3K).unwrap();
        match &d.scale_layout {
            ScaleLayout::Hierarchical { block_dmin_offset, packed_layout, .. } => {
                assert_eq!(*block_dmin_offset, None);
                assert!(matches!(packed_layout.algorithm, PackedScaleAlgorithm::Q3KExtended));
            }
            other => panic!("expected Hierarchical, got {:?}", other),
        }
    }

    // ========== Zero layout per format coverage ==========

    #[test]
    fn zero_layout_static_bias_per_format() {
        let r = registry();
        let cases: &[(QuantType, i32)] = &[
            (QuantType::Q4_0, 8),
            (QuantType::Q5_0, 16),
            (QuantType::Q6K, 32),
            (QuantType::Squeeze, 4),
            (QuantType::TQ1_0, 1),
            (QuantType::TQ2_0, 1),
        ];
        for (qt, expected_bias) in cases {
            let d = r.get(qt).unwrap();
            assert!(
                matches!(&d.zero_layout, ZeroLayout::StaticBias { value } if *value == *expected_bias),
                "{:?}: expected StaticBias({}), got {:?}", qt, expected_bias, d.zero_layout
            );
        }
    }

    #[test]
    fn zero_layout_hierarchical_for_q4k_q5k() {
        let r = registry();
        for qt in &[QuantType::Q4K, QuantType::Q5K] {
            let d = r.get(qt).unwrap();
            assert!(matches!(d.zero_layout, ZeroLayout::Hierarchical { .. }),
                "{:?} should have Hierarchical zero layout", qt);
        }
    }

    #[test]
    fn zero_layout_block_scalar_for_awq4_gptq4() {
        let r = registry();
        for qt in &[QuantType::AWQ4, QuantType::GPTQ4] {
            let d = r.get(qt).unwrap();
            assert!(matches!(d.zero_layout, ZeroLayout::BlockScalar { .. }),
                "{:?} should have BlockScalar zero layout", qt);
        }
    }
}
