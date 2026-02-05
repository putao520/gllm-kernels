use half::f16;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::backend_trait::{BackendError, BackendResult};
use crate::kernel_types::PackedBits;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Block<const N: usize> {
    pub scale: f16,
    pub data: [u8; N],
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockwiseMatrix<const N: usize> {
    pub blocks: Vec<Block<N>>,
    pub rows: usize,
    pub cols: usize,
    pub block_values: usize,
    pub bits: u8,
}

impl<const N: usize> BlockwiseMatrix<N> {
    pub fn blocks_per_row(&self) -> usize {
        if self.block_values == 0 {
            0
        } else {
            (self.cols + self.block_values - 1) / self.block_values
        }
    }
}

// ===== Quantized storage trait (Phase 1) =====

#[inline(always)]
const fn values_per_byte_const(bits: u8) -> usize {
    match bits {
        1 | 2 | 4 | 8 => 8 / bits as usize,
        _ => 0,
    }
}

#[inline(always)]
const fn block_values_const<const N: usize, const BITS: u8>() -> usize {
    let per_byte = values_per_byte_const(BITS);
    if per_byte == 0 {
        0
    } else {
        N * per_byte
    }
}

#[inline(always)]
fn decode_signed_const<const BITS: u8>(raw: u8) -> i8 {
    match BITS {
        1 => {
            if raw & 0x01 != 0 { 1 } else { -1 }
        }
        2 | 4 | 8 => {
            let mask = (1u16 << BITS) - 1;
            let sign_bit = 1u16 << (BITS - 1);
            let value = (raw as u16) & mask;
            if value & sign_bit != 0 {
                (value as i16 - (1i16 << BITS)) as i8
            } else {
                value as i8
            }
        }
        _ => 0,
    }
}

/// Unified quantized storage trait with block dequantization.
///
/// This trait enables zero-cost, const-generic specialization for different
/// bit widths while sharing a single linear implementation.
pub trait QuantizedStorage<const N: usize, const BITS: u8> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn blocks(&self) -> &[Block<N>];

    #[inline(always)]
    fn block_values(&self) -> usize {
        block_values_const::<N, BITS>()
    }

    #[inline(always)]
    fn blocks_per_row(&self) -> usize {
        let block_values = self.block_values();
        if block_values == 0 {
            0
        } else {
            (self.cols() + block_values - 1) / block_values
        }
    }

    /// Dequantize one block into `out`.
    ///
    /// `valid` indicates how many values are valid in the block (for tail blocks).
    fn dequantize_block(block: &Block<N>, valid: usize, out: &mut [f32]);
}

impl<const N: usize, const BITS: u8> QuantizedStorage<N, BITS> for BlockwiseMatrix<N> {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn blocks(&self) -> &[Block<N>] {
        &self.blocks
    }

    #[inline(always)]
    fn block_values(&self) -> usize {
        let expected = block_values_const::<N, BITS>();
        debug_assert!(expected > 0, "unsupported quant bits");
        debug_assert_eq!(self.bits, BITS);
        debug_assert_eq!(self.block_values, expected);
        expected
    }

    #[inline(always)]
    fn dequantize_block(block: &Block<N>, valid: usize, out: &mut [f32]) {
        let per_byte = values_per_byte_const(BITS);
        debug_assert!(per_byte > 0, "unsupported quant bits");
        debug_assert!(valid <= out.len());
        let scale = block.scale.to_f32();
        for idx in 0..valid {
            let byte = block.data[idx / per_byte];
            let shift = (idx % per_byte) * BITS as usize;
            let mask = ((1u16 << BITS) - 1) as u8;
            let raw = (byte >> shift) & mask;
            let q = decode_signed_const::<BITS>(raw);
            out[idx] = q as f32 * scale;
        }
    }
}

// ===== GGUF block quantization types (Phase 2) =====

/// GGUF Q4_0/Q8_0 block sizes (values per block).
pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

/// GGUF K-quant block size (values per block).
pub const QK_K: usize = 256;

/// Number of bytes used for packed scale/min values in K-quant blocks.
pub const K_SCALE_SIZE: usize = 12;

/// Q4_0 block payload bytes (packed 4-bit values).
pub const Q4_0_BLOCK_BYTES: usize = QK4_0 / 2; // 16 bytes

/// Q8_0 block payload bytes (int8 values).
pub const Q8_0_BLOCK_BYTES: usize = QK8_0; // 32 bytes

/// Q5_K block payload bytes stored after `scale` (d): dmin + scales + qh + qs.
pub const Q5_K_BLOCK_BYTES: usize = 2 + K_SCALE_SIZE + (QK_K / 8) + (QK_K / 2); // 174 bytes

/// Quantized tensor type identifiers for GGUF-compatible block formats.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedType {
    Q4_0,
    Q8_0,
    Q5_K,
}

#[allow(non_camel_case_types)]
pub type Q4_0Block = Block<Q4_0_BLOCK_BYTES>;
#[allow(non_camel_case_types)]
pub type Q8_0Block = Block<Q8_0_BLOCK_BYTES>;
#[allow(non_camel_case_types)]
pub type Q5_KBlock = Block<Q5_K_BLOCK_BYTES>;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq)]
pub struct Q4_0Matrix {
    pub blocks: Vec<Q4_0Block>,
    pub rows: usize,
    pub cols: usize,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0Matrix {
    pub blocks: Vec<Q8_0Block>,
    pub rows: usize,
    pub cols: usize,
}

/// GGUF Q5_K (used by Q5_K_M) matrix storage.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq)]
pub struct Q5_KMatrix {
    pub blocks: Vec<Q5_KBlock>,
    pub rows: usize,
    pub cols: usize,
}

impl Q4_0Matrix {
    #[inline(always)]
    pub fn blocks_per_row(&self) -> usize {
        (self.cols + QK4_0 - 1) / QK4_0
    }
}

impl Q8_0Matrix {
    #[inline(always)]
    pub fn blocks_per_row(&self) -> usize {
        (self.cols + QK8_0 - 1) / QK8_0
    }
}

impl Q5_KMatrix {
    #[inline(always)]
    pub fn blocks_per_row(&self) -> usize {
        (self.cols + QK_K - 1) / QK_K
    }
}

impl QuantizedStorage<Q4_0_BLOCK_BYTES, 4> for Q4_0Matrix {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn blocks(&self) -> &[Block<Q4_0_BLOCK_BYTES>] {
        &self.blocks
    }

    #[inline(always)]
    fn block_values(&self) -> usize {
        QK4_0
    }

    #[inline(always)]
    fn dequantize_block(block: &Block<Q4_0_BLOCK_BYTES>, valid: usize, out: &mut [f32]) {
        dequantize_block_q4_0(block, valid, out);
    }
}

impl QuantizedStorage<Q8_0_BLOCK_BYTES, 8> for Q8_0Matrix {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn blocks(&self) -> &[Block<Q8_0_BLOCK_BYTES>] {
        &self.blocks
    }

    #[inline(always)]
    fn block_values(&self) -> usize {
        QK8_0
    }

    #[inline(always)]
    fn dequantize_block(block: &Block<Q8_0_BLOCK_BYTES>, valid: usize, out: &mut [f32]) {
        dequantize_block_q8_0(block, valid, out);
    }
}

// NOTE: Q5_K uses 5-bit values, but we bind BITS=4 to reuse generic blockwise
// helpers that require a packed bit width dividing 8. The dequantization logic
// below handles the actual 5-bit reconstruction.
impl QuantizedStorage<Q5_K_BLOCK_BYTES, 4> for Q5_KMatrix {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn blocks(&self) -> &[Block<Q5_K_BLOCK_BYTES>] {
        &self.blocks
    }

    #[inline(always)]
    fn block_values(&self) -> usize {
        QK_K
    }

    #[inline(always)]
    fn dequantize_block(block: &Block<Q5_K_BLOCK_BYTES>, valid: usize, out: &mut [f32]) {
        dequantize_block_q5_k(block, valid, out);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedU8 {
    pub bits: PackedBits,
    pub data: Vec<u8>,
    pub values: usize,
}

impl PackedU8 {
    pub fn from_i8(values: &[i8], bits: PackedBits) -> BackendResult<Self> {
        let values_len = values.len();
        let (min_q, max_q, _) = quant_bounds(bits.bits() as u8)?;
        let bytes = packed_storage_bytes(values_len, bits.bits() as u8)?;
        let mut data = vec![0u8; bytes];
        for (idx, value) in values.iter().enumerate() {
            let mut q = *value;
            if q < min_q {
                q = min_q;
            } else if q > max_q {
                q = max_q;
            }
            pack_value(bits.bits() as u8, idx, q, &mut data)?;
        }
        Ok(Self {
            bits,
            data,
            values: values_len,
        })
    }

    pub fn unpack_i8(&self, output: &mut [i8]) -> BackendResult<()> {
        if output.len() < self.values {
            return Err(BackendError::InvalidConfig(
                "packed output too small".into(),
            ));
        }
        for idx in 0..self.values {
            output[idx] = unpack_value(self.bits.bits() as u8, idx, &self.data)?;
        }
        Ok(())
    }

    pub fn values_per_byte(&self) -> usize {
        self.bits.values_per_byte()
    }
}

#[inline]
fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ===== GGUF dequantization helpers =====

#[inline(always)]
fn unpack_scales_mins_k(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    debug_assert!(scales.len() >= K_SCALE_SIZE);
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    for i in 0..4 {
        sc[i] = scales[i] & 0x3f;
        mn[i] = scales[i + 4] & 0x3f;
    }

    for i in 0..4 {
        let low2_sc = scales[i] >> 6;
        let high4_sc = scales[8 + i] & 0x0f;
        sc[i + 4] = low2_sc | (high4_sc << 2);

        let low2_mn = scales[i + 4] >> 6;
        let high4_mn = scales[8 + i] >> 4;
        mn[i + 4] = low2_mn | (high4_mn << 2);
    }

    (sc, mn)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_i8x32_avx2(src: *const i8, dst: *mut f32, scale: f32) {
    let scale_vec = _mm256_set1_ps(scale);
    for i in 0..4 {
        let bytes = _mm_loadl_epi64(src.add(i * 8) as *const __m128i);
        let ints = _mm256_cvtepi8_epi32(bytes);
        let floats = _mm256_cvtepi32_ps(ints);
        let scaled = _mm256_mul_ps(floats, scale_vec);
        _mm256_storeu_ps(dst.add(i * 8), scaled);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_u8x32_avx2(src: *const u8, dst: *mut f32, scale: f32, bias: f32) {
    let scale_vec = _mm256_set1_ps(scale);
    let bias_vec = _mm256_set1_ps(bias);
    for i in 0..4 {
        let bytes = _mm_loadl_epi64(src.add(i * 8) as *const __m128i);
        let ints = _mm256_cvtepu8_epi32(bytes);
        let floats = _mm256_cvtepi32_ps(ints);
        let scaled = _mm256_add_ps(_mm256_mul_ps(floats, scale_vec), bias_vec);
        _mm256_storeu_ps(dst.add(i * 8), scaled);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequantize_i8x32_neon(src: *const i8, dst: *mut f32, scale: f32) {
    let scale_vec = vdupq_n_f32(scale);
    for chunk in 0..2 {
        let v = vld1q_s8(src.add(chunk * 16));
        let lo = vmovl_s8(vget_low_s8(v));
        let hi = vmovl_s8(vget_high_s8(v));

        let lo0 = vmovl_s16(vget_low_s16(lo));
        let lo1 = vmovl_s16(vget_high_s16(lo));
        let hi0 = vmovl_s16(vget_low_s16(hi));
        let hi1 = vmovl_s16(vget_high_s16(hi));

        let f0 = vmulq_f32(vcvtq_f32_s32(lo0), scale_vec);
        let f1 = vmulq_f32(vcvtq_f32_s32(lo1), scale_vec);
        let f2 = vmulq_f32(vcvtq_f32_s32(hi0), scale_vec);
        let f3 = vmulq_f32(vcvtq_f32_s32(hi1), scale_vec);

        let out = dst.add(chunk * 16);
        vst1q_f32(out, f0);
        vst1q_f32(out.add(4), f1);
        vst1q_f32(out.add(8), f2);
        vst1q_f32(out.add(12), f3);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequantize_u8x32_neon(src: *const u8, dst: *mut f32, scale: f32, bias: f32) {
    let scale_vec = vdupq_n_f32(scale);
    let bias_vec = vdupq_n_f32(bias);
    for chunk in 0..2 {
        let v = vld1q_u8(src.add(chunk * 16));
        let lo = vmovl_u8(vget_low_u8(v));
        let hi = vmovl_u8(vget_high_u8(v));

        let lo0 = vmovl_u16(vget_low_u16(lo));
        let lo1 = vmovl_u16(vget_high_u16(lo));
        let hi0 = vmovl_u16(vget_low_u16(hi));
        let hi1 = vmovl_u16(vget_high_u16(hi));

        let f0 = vmlaq_f32(bias_vec, vcvtq_f32_u32(lo0), scale_vec);
        let f1 = vmlaq_f32(bias_vec, vcvtq_f32_u32(lo1), scale_vec);
        let f2 = vmlaq_f32(bias_vec, vcvtq_f32_u32(hi0), scale_vec);
        let f3 = vmlaq_f32(bias_vec, vcvtq_f32_u32(hi1), scale_vec);

        let out = dst.add(chunk * 16);
        vst1q_f32(out, f0);
        vst1q_f32(out.add(4), f1);
        vst1q_f32(out.add(8), f2);
        vst1q_f32(out.add(12), f3);
    }
}

#[inline(always)]
fn dequantize_block_q4_0(block: &Q4_0Block, valid: usize, out: &mut [f32]) {
    debug_assert!(valid <= QK4_0);
    let scale = block.scale.to_f32();

    if valid == QK4_0 {
        if is_avx2_available() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let mut tmp = [0i8; QK4_0];
                for idx in 0..QK4_0 {
                    let byte = block.data[idx / 2];
                    let raw = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    tmp[idx] = if raw & 0x08 != 0 {
                        (raw as i8) - 16
                    } else {
                        raw as i8
                    };
                }
                dequantize_i8x32_avx2(tmp.as_ptr(), out.as_mut_ptr(), scale);
                return;
            }
        }
        if is_neon_available() {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let mut tmp = [0i8; QK4_0];
                for idx in 0..QK4_0 {
                    let byte = block.data[idx / 2];
                    let raw = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    tmp[idx] = if raw & 0x08 != 0 {
                        (raw as i8) - 16
                    } else {
                        raw as i8
                    };
                }
                dequantize_i8x32_neon(tmp.as_ptr(), out.as_mut_ptr(), scale);
                return;
            }
        }
    }

    for idx in 0..valid {
        let byte = block.data[idx / 2];
        let raw = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        let q = if raw & 0x08 != 0 { (raw as i8) - 16 } else { raw as i8 };
        out[idx] = q as f32 * scale;
    }
}

#[inline(always)]
fn dequantize_block_q8_0(block: &Q8_0Block, valid: usize, out: &mut [f32]) {
    debug_assert!(valid <= QK8_0);
    let scale = block.scale.to_f32();

    if valid == QK8_0 {
        if is_avx2_available() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let src = block.data.as_ptr() as *const i8;
                dequantize_i8x32_avx2(src, out.as_mut_ptr(), scale);
                return;
            }
        }
        if is_neon_available() {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let src = block.data.as_ptr() as *const i8;
                dequantize_i8x32_neon(src, out.as_mut_ptr(), scale);
                return;
            }
        }
    }

    for idx in 0..valid {
        let q = block.data[idx] as i8;
        out[idx] = q as f32 * scale;
    }
}

#[inline(always)]
fn dequantize_block_q5_k(block: &Q5_KBlock, valid: usize, out: &mut [f32]) {
    debug_assert!(valid <= QK_K);
    let d = block.scale.to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block.data[0], block.data[1]])).to_f32();

    let scales = &block.data[2..2 + K_SCALE_SIZE];
    let qh = &block.data[2 + K_SCALE_SIZE..2 + K_SCALE_SIZE + (QK_K / 8)];
    let qs = &block.data[2 + K_SCALE_SIZE + (QK_K / 8)..];

    let (sc, mn) = unpack_scales_mins_k(scales);

    for sb in 0..8 {
        let base = sb * 32;
        if base >= valid {
            break;
        }
        let sb_valid = (valid - base).min(32);
        let scale = d * sc[sb] as f32;
        let min = dmin * mn[sb] as f32;

        if sb_valid == 32 {
            let mut tmp = [0u8; 32];
            for i in 0..32 {
                let idx = base + i;
                let byte = qs[idx / 2];
                let low4 = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let high = (qh[idx / 8] >> (idx % 8)) & 0x01;
                tmp[i] = low4 | (high << 4);
            }

            if is_avx2_available() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    dequantize_u8x32_avx2(tmp.as_ptr(), out[base..].as_mut_ptr(), scale, min);
                    continue;
                }
            }
            if is_neon_available() {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    dequantize_u8x32_neon(tmp.as_ptr(), out[base..].as_mut_ptr(), scale, min);
                    continue;
                }
            }

            for i in 0..32 {
                out[base + i] = tmp[i] as f32 * scale + min;
            }
        } else {
            for i in 0..sb_valid {
                let idx = base + i;
                let byte = qs[idx / 2];
                let low4 = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let high = (qh[idx / 8] >> (idx % 8)) & 0x01;
                let q = low4 | (high << 4);
                out[base + i] = q as f32 * scale + min;
            }
        }
    }
}

fn packed_storage_bytes(values: usize, bits: u8) -> BackendResult<usize> {
    let per_byte = values_per_byte(bits)?;
    if values == 0 {
        return Ok(0);
    }
    values
        .checked_add(per_byte - 1)
        .map(|v| v / per_byte)
        .ok_or_else(|| BackendError::InvalidConfig("packed storage overflow".into()))
}

fn values_per_byte(bits: u8) -> BackendResult<usize> {
    match bits {
        1 | 2 | 4 | 8 => Ok(8 / bits as usize),
        _ => Err(BackendError::InvalidConfig(
            "unsupported packed bits".into(),
        )),
    }
}

fn quant_bounds(bits: u8) -> BackendResult<(i8, i8, f32)> {
    match bits {
        1 => Ok((-1, 1, 1.0)),
        2 | 4 | 8 => {
            let max = (1i16 << (bits - 1)) - 1;
            let min = -1i16 << (bits - 1);
            Ok((min as i8, max as i8, max as f32))
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn quant_bounds_const<const BITS: u8>() -> BackendResult<(i8, i8, f32)> {
    match BITS {
        1 => Ok((-1, 1, 1.0)),
        2 | 4 | 8 => {
            let max = (1i16 << (BITS - 1)) - 1;
            let min = -1i16 << (BITS - 1);
            Ok((min as i8, max as i8, max as f32))
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn encode_signed(bits: u8, value: i8) -> BackendResult<u8> {
    match bits {
        1 => Ok(if value >= 0 { 1 } else { 0 }),
        2 | 4 | 8 => {
            let mask = (1u16 << bits) - 1;
            Ok((value as i16 as u16 & mask) as u8)
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn decode_signed(bits: u8, raw: u8) -> BackendResult<i8> {
    match bits {
        1 => Ok(if raw & 0x1 != 0 { 1 } else { -1 }),
        2 | 4 | 8 => {
            let mask = (1u16 << bits) - 1;
            let sign_bit = 1u16 << (bits - 1);
            let value = (raw as u16) & mask;
            if value & sign_bit != 0 {
                Ok((value as i16 - (1i16 << bits)) as i8)
            } else {
                Ok(value as i8)
            }
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn pack_value(bits: u8, idx: usize, value: i8, data: &mut [u8]) -> BackendResult<()> {
    let per_byte = values_per_byte(bits)?;
    let byte_idx = idx / per_byte;
    if byte_idx >= data.len() {
        return Err(BackendError::InvalidConfig("packed write overflow".into()));
    }
    let shift = ((idx % per_byte) * bits as usize) as u32;
    let mask = ((1u16 << bits) - 1) as u8;
    let encoded = encode_signed(bits, value)? & mask;
    let slot = data[byte_idx] & !(mask << shift);
    data[byte_idx] = slot | (encoded << shift);
    Ok(())
}

fn unpack_value(bits: u8, idx: usize, data: &[u8]) -> BackendResult<i8> {
    let per_byte = values_per_byte(bits)?;
    let byte_idx = idx / per_byte;
    if byte_idx >= data.len() {
        return Err(BackendError::InvalidConfig("packed read overflow".into()));
    }
    let shift = ((idx % per_byte) * bits as usize) as u32;
    let mask = ((1u16 << bits) - 1) as u8;
    let raw = (data[byte_idx] >> shift) & mask;
    decode_signed(bits, raw)
}

fn block_values_checked<const N: usize, const BITS: u8>() -> BackendResult<usize> {
    if N == 0 {
        return Err(BackendError::InvalidConfig("block size must be > 0".into()));
    }
    let per_byte = values_per_byte_const(BITS);
    if per_byte == 0 {
        return Err(BackendError::InvalidConfig("unsupported packed bits".into()));
    }
    N.checked_mul(per_byte)
        .ok_or_else(|| BackendError::InvalidConfig("block size overflow".into()))
}

pub fn quantize_blockwise_bits<const N: usize, const BITS: u8>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    let block_values = block_values_checked::<N, BITS>()?;
    let total = rows.saturating_mul(cols);
    if input.len() < total {
        return Err(BackendError::InvalidConfig(
            "blockwise quantize input too small".into(),
        ));
    }

    let (min_q, max_q, max_q_f) = quant_bounds_const::<BITS>()?;
    let blocks_per_row = if block_values == 0 {
        0
    } else {
        (cols + block_values - 1) / block_values
    };
    let mut blocks = Vec::with_capacity(rows.saturating_mul(blocks_per_row));

    for row in 0..rows {
        let row_base = row * cols;
        for block_idx in 0..blocks_per_row {
            let block_start = row_base + block_idx * block_values;
            let mut max_abs = 0.0f32;
            for t in 0..block_values {
                let idx = block_start + t;
                if idx >= row_base + cols {
                    break;
                }
                let v = input[idx].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = if max_abs > 0.0 {
                max_abs / max_q_f
            } else {
                1.0
            };
            let inv_scale = 1.0 / scale;
            let mut data = [0u8; N];
            for t in 0..block_values {
                let idx = block_start + t;
                let v = if idx < row_base + cols {
                    input[idx]
                } else {
                    0.0
                };
                let mut q = (v * inv_scale).round() as i32;
                if q < min_q as i32 {
                    q = min_q as i32;
                } else if q > max_q as i32 {
                    q = max_q as i32;
                }
                pack_value(BITS, t, q as i8, &mut data)?;
            }
            blocks.push(Block {
                scale: f16::from_f32(scale),
                data,
            });
        }
    }

    Ok(BlockwiseMatrix {
        blocks,
        rows,
        cols,
        block_values,
        bits: BITS,
    })
}

pub fn quantize_blockwise<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
) -> BackendResult<BlockwiseMatrix<N>> {
    match bits {
        8 => quantize_blockwise_bits::<N, 8>(input, rows, cols),
        4 => quantize_blockwise_bits::<N, 4>(input, rows, cols),
        2 => quantize_blockwise_bits::<N, 2>(input, rows, cols),
        1 => quantize_blockwise_bits::<N, 1>(input, rows, cols),
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

pub fn quantize_blockwise_int8<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise_bits::<N, 8>(input, rows, cols)
}

pub fn quantize_blockwise_int4<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise_bits::<N, 4>(input, rows, cols)
}

pub fn quantize_blockwise_int2<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise_bits::<N, 2>(input, rows, cols)
}

pub fn quantize_blockwise_int1<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise_bits::<N, 1>(input, rows, cols)
}

pub fn dequantize_blockwise<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    match matrix.bits {
        8 => dequantize_blockwise_generic::<N, 8, _>(matrix, output),
        4 => dequantize_blockwise_generic::<N, 4, _>(matrix, output),
        2 => dequantize_blockwise_generic::<N, 2, _>(matrix, output),
        1 => dequantize_blockwise_generic::<N, 1, _>(matrix, output),
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

pub fn dequantize_q4_0(matrix: &Q4_0Matrix, output: &mut [f32]) -> BackendResult<()> {
    dequantize_blockwise_generic::<Q4_0_BLOCK_BYTES, 4, _>(matrix, output)
}

pub fn dequantize_q8_0(matrix: &Q8_0Matrix, output: &mut [f32]) -> BackendResult<()> {
    dequantize_blockwise_generic::<Q8_0_BLOCK_BYTES, 8, _>(matrix, output)
}

pub fn dequantize_q5_k(matrix: &Q5_KMatrix, output: &mut [f32]) -> BackendResult<()> {
    dequantize_blockwise_generic::<Q5_K_BLOCK_BYTES, 4, _>(matrix, output)
}

pub fn dequantize_blockwise_generic<const N: usize, const BITS: u8, S: QuantizedStorage<N, BITS>>(
    matrix: &S,
    output: &mut [f32],
) -> BackendResult<()> {
    if values_per_byte_const(BITS) == 0 {
        return Err(BackendError::InvalidConfig("unsupported packed bits".into()));
    }
    let rows = matrix.rows();
    let cols = matrix.cols();
    let total = rows.saturating_mul(cols);
    if output.len() < total {
        return Err(BackendError::InvalidConfig(
            "blockwise dequantize output too small".into(),
        ));
    }

    let blocks_per_row = matrix.blocks_per_row();
    let blocks = matrix.blocks();
    if blocks.len() < rows.saturating_mul(blocks_per_row) {
        return Err(BackendError::InvalidConfig(
            "blockwise matrix storage too small".into(),
        ));
    }

    let block_values = matrix.block_values();
    if block_values == 0 {
        return Err(BackendError::InvalidConfig(
            "blockwise matrix block size mismatch".into(),
        ));
    }

    for row in 0..rows {
        let row_base = row * cols;
        for block_idx in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + block_idx];
            let block_start = row_base + block_idx * block_values;
            let remaining = cols.saturating_sub(block_idx * block_values);
            let valid = remaining.min(block_values);
            let out_slice = &mut output[block_start..block_start + valid];
            S::dequantize_block(block, valid, out_slice);
        }
    }
    Ok(())
}

pub fn dequantize_blockwise_int8<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 8 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 8-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise_generic::<N, 8, _>(matrix, output)
}

pub fn dequantize_blockwise_int4<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 4 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 4-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise_generic::<N, 4, _>(matrix, output)
}

pub fn dequantize_blockwise_int2<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 2 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 2-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise_generic::<N, 2, _>(matrix, output)
}

pub fn dequantize_blockwise_int1<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 1 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 1-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise_generic::<N, 1, _>(matrix, output)
}

pub fn linear_blockwise<const N: usize, const BITS: u8, S: QuantizedStorage<N, BITS>>(
    input: &[f32],
    weight: &S,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if values_per_byte_const(BITS) == 0 {
        return Err(BackendError::InvalidConfig("unsupported packed bits".into()));
    }
    let k = weight.cols();
    let n = weight.rows();
    let input_len = m.saturating_mul(k);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "blockwise matmul buffer size mismatch".into(),
        ));
    }

    let blocks_per_row = weight.blocks_per_row();
    let blocks = weight.blocks();
    if blocks.len() < n.saturating_mul(blocks_per_row) {
        return Err(BackendError::InvalidConfig(
            "blockwise matmul weight storage too small".into(),
        ));
    }

    let block_values = weight.block_values();
    if block_values == 0 {
        return Err(BackendError::InvalidConfig(
            "blockwise matmul block size mismatch".into(),
        ));
    }

    let mut values = vec![0f32; block_values];
    for row in 0..m {
        let row_base = row * k;
        for col in 0..n {
            let mut sum = 0.0f32;
            let weight_base = col * blocks_per_row;
            for block_idx in 0..blocks_per_row {
                let block = &blocks[weight_base + block_idx];
                let block_start = block_idx * block_values;
                let valid = k.saturating_sub(block_start).min(block_values);
                S::dequantize_block(block, valid, &mut values);
                let input_slice = &input[row_base + block_start..row_base + block_start + valid];
                sum += dot_f32(input_slice, &values[..valid]);
            }
            output[row * n + col] = sum;
        }
    }
    Ok(())
}

pub fn matmul_blockwise<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    match weight.bits {
        8 => linear_blockwise::<N, 8, _>(input, weight, output, m),
        4 => linear_blockwise::<N, 4, _>(input, weight, output, m),
        2 => linear_blockwise::<N, 2, _>(input, weight, output, m),
        1 => linear_blockwise::<N, 1, _>(input, weight, output, m),
        _ => Err(BackendError::InvalidConfig("unsupported packed bits".into())),
    }
}

pub fn matmul_blockwise_int8<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 8 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 8-bit blockwise matrix".into(),
        ));
    }
    linear_blockwise::<N, 8, _>(input, weight, output, m)
}

pub fn matmul_blockwise_int4<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 4 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 4-bit blockwise matrix".into(),
        ));
    }
    linear_blockwise::<N, 4, _>(input, weight, output, m)
}

pub fn matmul_blockwise_int2<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 2 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 2-bit blockwise matrix".into(),
        ));
    }
    linear_blockwise::<N, 2, _>(input, weight, output, m)
}

pub fn matmul_blockwise_int1<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 1 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 1-bit blockwise matrix".into(),
        ));
    }
    linear_blockwise::<N, 1, _>(input, weight, output, m)
}

fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    if is_avx2_available() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if std::arch::is_x86_feature_detected!("fma") {
                return dot_avx2_fma(lhs, rhs, len);
            }
            return dot_avx2(lhs, rhs, len);
        }
    }
    if is_neon_available() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return dot_neon(lhs, rhs, len);
        }
    }
    dot_scalar(lhs, rhs, len)
}

fn dot_scalar(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += lhs[i] * rhs[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += lhs[i] * rhs[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
        acc = _mm256_fmadd_ps(a, b, acc);
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += lhs[i] * rhs[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let a = vld1q_f32(lhs.as_ptr().add(i));
        let b = vld1q_f32(rhs.as_ptr().add(i));
        acc = vmlaq_f32(acc, a, b);
        i += 4;
    }
    let mut tmp = [0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += lhs[i] * rhs[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_u8_roundtrip() {
        let values = vec![-2i8, -1, 0, 1, 1, 0, -1, -2, 1];
        let packed = PackedU8::from_i8(&values, PackedBits::Int2).unwrap();
        let mut output = vec![0i8; values.len()];
        packed.unpack_i8(&mut output).unwrap();
        assert_eq!(values, output);

        let values = vec![-1i8, 1, -1, 1, 1, -1, 1, -1];
        let packed = PackedU8::from_i8(&values, PackedBits::Int1).unwrap();
        let mut output = vec![0i8; values.len()];
        packed.unpack_i8(&mut output).unwrap();
        assert_eq!(values, output);
    }

    #[test]
    fn blockwise_int4_roundtrip() {
        let rows = 2;
        let cols = 9;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 10.0) * 0.25).collect();
        let matrix = quantize_blockwise_int4::<16>(&input, rows, cols).unwrap();
        let mut output = vec![0f32; rows * cols];
        dequantize_blockwise_int4(&matrix, &mut output).unwrap();
        for (orig, deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.5);
        }
    }

    #[test]
    fn blockwise_int4_matmul_matches_dequantized() {
        let rows = 3;
        let cols = 8;
        let input_rows = 2;
        let input: Vec<f32> = (0..input_rows * cols)
            .map(|i| (i as f32 - 5.0) * 0.1)
            .collect();
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.2) - 1.5).collect();
        let matrix = quantize_blockwise_int4::<16>(&weight, rows, cols).unwrap();

        let mut dequant = vec![0f32; rows * cols];
        dequantize_blockwise_int4(&matrix, &mut dequant).unwrap();

        let mut expected = vec![0f32; input_rows * rows];
        for i in 0..input_rows {
            for j in 0..rows {
                let mut sum = 0.0f32;
                for k in 0..cols {
                    sum += input[i * cols + k] * dequant[j * cols + k];
                }
                expected[i * rows + j] = sum;
            }
        }

        let mut output = vec![0f32; input_rows * rows];
        matmul_blockwise_int4(&input, &matrix, &mut output, input_rows).unwrap();

        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn blockwise_matmul_multiple_bits() {
        fn run<const N: usize>(bits: u8) {
            let rows = 4;
            let cols = 7;
            let input_rows = 3;
            let input: Vec<f32> = (0..input_rows * cols)
                .map(|i| (i as f32 - 4.0) * 0.2)
                .collect();
            let weight: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.15) - 1.0).collect();
            let matrix = quantize_blockwise::<N>(&weight, rows, cols, bits).unwrap();

            let mut dequant = vec![0f32; rows * cols];
            dequantize_blockwise(&matrix, &mut dequant).unwrap();

            let mut expected = vec![0f32; input_rows * rows];
            for i in 0..input_rows {
                for j in 0..rows {
                    let mut sum = 0.0f32;
                    for k in 0..cols {
                        sum += input[i * cols + k] * dequant[j * cols + k];
                    }
                    expected[i * rows + j] = sum;
                }
            }

            let mut output = vec![0f32; input_rows * rows];
            matmul_blockwise(&input, &matrix, &mut output, input_rows).unwrap();

            for (a, b) in output.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }

        run::<16>(8);
        run::<16>(4);
        run::<16>(2);
        run::<16>(1);
    }

    #[test]
    fn q4_0_dequant_basic() {
        let mut data = [0u8; Q4_0_BLOCK_BYTES];
        data[0] = 0x8f; // low nibble = 0xF (-1), high nibble = 0x8 (-8)
        let block = Block {
            scale: f16::from_f32(1.0),
            data,
        };
        let matrix = Q4_0Matrix {
            blocks: vec![block],
            rows: 1,
            cols: QK4_0,
        };
        let mut out = vec![0f32; QK4_0];
        dequantize_q4_0(&matrix, &mut out).unwrap();
        assert_eq!(out[0], -1.0);
        assert_eq!(out[1], -8.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn q8_0_dequant_basic() {
        let mut data = [0u8; Q8_0_BLOCK_BYTES];
        data[0] = (-4i8) as u8;
        data[1] = 3i8 as u8;
        let block = Block {
            scale: f16::from_f32(0.5),
            data,
        };
        let matrix = Q8_0Matrix {
            blocks: vec![block],
            rows: 1,
            cols: QK8_0,
        };
        let mut out = vec![0f32; QK8_0];
        dequantize_q8_0(&matrix, &mut out).unwrap();
        assert!((out[0] + 2.0).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
    }

    fn pack_scales_mins_k(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; K_SCALE_SIZE] {
        let mut out = [0u8; K_SCALE_SIZE];
        for i in 0..4 {
            out[i] = sc[i] & 0x3f;
            out[i + 4] = mn[i] & 0x3f;

            out[i] |= (sc[i + 4] & 0x03) << 6;
            out[i + 4] |= (mn[i + 4] & 0x03) << 6;

            out[8 + i] = ((sc[i + 4] >> 2) & 0x0f) | ((mn[i + 4] >> 2) << 4);
        }
        out
    }

    #[test]
    fn q5_k_dequant_basic() {
        let mut data = [0u8; Q5_K_BLOCK_BYTES];
        let dmin_bits = f16::from_f32(0.0).to_bits().to_le_bytes();
        data[0] = dmin_bits[0];
        data[1] = dmin_bits[1];

        let sc = [1u8; 8];
        let mn = [0u8; 8];
        let scales = pack_scales_mins_k(&sc, &mn);
        data[2..2 + K_SCALE_SIZE].copy_from_slice(&scales);

        let mut qh = [0u8; QK_K / 8];
        let mut qs = [0u8; QK_K / 2];
        for idx in 0..QK_K {
            let q = (idx % 32) as u8;
            let low4 = q & 0x0f;
            let high = (q >> 4) & 0x01;
            let byte_idx = idx / 2;
            if idx % 2 == 0 {
                qs[byte_idx] = (qs[byte_idx] & 0xf0) | low4;
            } else {
                qs[byte_idx] = (qs[byte_idx] & 0x0f) | (low4 << 4);
            }
            if high != 0 {
                qh[idx / 8] |= 1 << (idx % 8);
            }
        }
        let qh_offset = 2 + K_SCALE_SIZE;
        let qs_offset = qh_offset + (QK_K / 8);
        data[qh_offset..qh_offset + (QK_K / 8)].copy_from_slice(&qh);
        data[qs_offset..qs_offset + (QK_K / 2)].copy_from_slice(&qs);

        let block = Block {
            scale: f16::from_f32(1.0),
            data,
        };
        let matrix = Q5_KMatrix {
            blocks: vec![block],
            rows: 1,
            cols: QK_K,
        };
        let mut out = vec![0f32; QK_K];
        dequantize_q5_k(&matrix, &mut out).unwrap();
        for idx in 0..QK_K {
            assert!((out[idx] - (idx % 32) as f32).abs() < 1e-6);
        }
    }
}
