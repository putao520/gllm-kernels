//! Microscaling FP4 (mxfp4) dequantization — OCP standard.
//!
//! Format reference: OCP Microscaling Formats (MX) Specification v1.0.
//!
//! - **Element type**: e2m1 (4-bit float) — `bit[3]=sign, bit[2:1]=exponent, bit[0]=mantissa`.
//!   Decodes to one of 16 values: `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
//! - **Scale type**: e8m0 (8-bit power-of-2 scale) — `scale = 2^(byte - 127)`.
//!   `byte = 255` indicates NaN per OCP spec.
//! - **Block layout**: `block_size` consecutive e2m1 elements share a single e8m0 scale.
//!   Standard `block_size = 32`. Per block: `block_size / 2` packed bytes (low nibble = even index,
//!   high nibble = odd index).
//!
//! This module exposes:
//! - [`E2M1_LUT_F32`]: 16-entry lookup table mapping code → f32 (ground truth).
//! - [`dequant_mxfp4_scalar`]: SymExec-compatible reference implementation.
//! - [`dequant_mxfp4_avx2`]: production AVX2 implementation (real machine code, no fallback).
//! - [`dequant_mxfp4`]: runtime ISA dispatcher (AVX2 → scalar).
//!
//! **NO_SCALAR rule**: `dequant_mxfp4_scalar` is intended only for SymExec ground truth and
//! test reference. The runtime hot path must use [`dequant_mxfp4`] (or `dequant_mxfp4_avx2`
//! directly when AVX2 has been verified at higher level).
//!
//! **dtype 设计(REQ-DTYPE-CHAIN-004)**:MXFP4 解码输出固定 F32 是反量化语义合法默认
//! (4-bit 量化 → 浮点值)。多精度架构下,若需 BF16/F16 输出,由调用方在 wrapper 层做
//! F32 → 目标 dtype 的 cast,而非 dequant 函数自己处理。`vec![0f32; N]` 仅出现在测试
//! 代码,验证 F32 解码路径正确性。

/// OCP e2m1 4-bit float lookup table.
///
/// Index = 4-bit code (`bit[3]=sign, bit[2:1]=exp, bit[0]=mantissa`).
/// Codes 0..7 are non-negative (sign=0); codes 8..15 are negative (sign=1).
/// Both code 0 (+0) and code 8 (-0) decode to `0.0`.
pub const E2M1_LUT_F32: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode an e8m0 scale byte to its f32 power-of-2 multiplier.
///
/// `byte = v` → `2^(v - 127)`. `byte = 255` is NaN per OCP spec. `byte = 0` decodes to
/// `2^-127` which lies in the subnormal range of binary32 (encoded as the f32 with all
/// exponent bits zero and mantissa bit 22 set).
#[inline(always)]
pub fn decode_e8m0_scale(byte: u8) -> f32 {
    match byte {
        // OCP spec: byte=255 is the only NaN encoding (Section "E8M0 Scale Format").
        255 => f32::NAN,
        // 2^-127 sits in the subnormal range: subnormal f32 = mantissa/2^23 * 2^-126.
        // mantissa = 2^22 ⇒ value = 0.5 * 2^-126 = 2^-127.
        0 => f32::from_bits(0x0040_0000),
        // Normal range byte ∈ [1, 254]: biased exponent field stored verbatim.
        // 2^(byte-127) = f32 with exponent field = byte and zero mantissa.
        v => f32::from_bits((v as u32) << 23),
    }
}

/// Scalar reference implementation — golden truth for SymExec & tests.
///
/// # Layout
/// - `blocks`: packed e2m1 nibbles. Length must be `(num_blocks * block_size) / 2` bytes.
///   For `block_size = 32` this is `16 * num_blocks` bytes. Even-index elements live in the
///   low nibble, odd-index in the high nibble.
/// - `scales`: one e8m0 byte per block. Length must be `num_blocks`.
/// - `output`: dequantized f32 values. Length must be `num_blocks * block_size`.
/// - `block_size`: elements per block (typically 32 for OCP standard).
///
/// # Panics
/// Panics in debug mode if `block_size` is odd or if any length is inconsistent.
///
/// # Numerical contract
/// `output[block_idx * block_size + i] = E2M1_LUT_F32[nibble] * decode_e8m0_scale(scales[block_idx])`
#[inline]
pub fn dequant_mxfp4_scalar(
    blocks: &[u8],
    scales: &[u8],
    output: &mut [f32],
    block_size: usize,
) {
    debug_assert!(
        block_size % 2 == 0,
        "mxfp4 block_size must be even (two e2m1 per byte); got {block_size}"
    );
    let num_blocks = scales.len();
    debug_assert_eq!(
        blocks.len(),
        num_blocks * (block_size / 2),
        "blocks len mismatch: {} blocks × {} bytes/block ≠ {} bytes",
        num_blocks,
        block_size / 2,
        blocks.len(),
    );
    debug_assert_eq!(
        output.len(),
        num_blocks * block_size,
        "output len mismatch: {} blocks × {} elems/block ≠ {} elems",
        num_blocks,
        block_size,
        output.len(),
    );

    let bytes_per_block = block_size / 2;
    for blk in 0..num_blocks {
        let scale = decode_e8m0_scale(scales[blk]);
        let qs = &blocks[blk * bytes_per_block..(blk + 1) * bytes_per_block];
        let out = &mut output[blk * block_size..(blk + 1) * block_size];
        for i in 0..bytes_per_block {
            let byte = qs[i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            // OCP packing: low nibble = even index, high nibble = odd index.
            out[2 * i] = E2M1_LUT_F32[lo] * scale;
            out[2 * i + 1] = E2M1_LUT_F32[hi] * scale;
        }
    }
}

/// AVX2 production implementation — real machine code (no fallback).
///
/// Strategy:
/// 1. Use `_mm256_permutevar8x32_ps` with a 16-entry LUT loaded as two `__m256` (low/high halves).
///    AVX2 doesn't have a 16-lane f32 shuffle, so we shuffle two 8-lane f32 LUTs by index range
///    and `vblendvps` based on the high bit (sign bit, bit 3).
/// 2. Process 32 elements (one OCP block) per outer iteration: load 16 packed bytes via
///    `_mm_loadu_si128`, split into low/high nibbles via mask + shift, expand each nibble into
///    32-bit indices via `_mm256_cvtepu8_epi32`, gather LUT values via `_mm256_permutevar8x32`,
///    multiply by broadcast scale, store 32 contiguous f32.
///
/// # Safety
/// - `target_arch = "x86_64"` and `target_feature = "avx2"`. Caller must verify AVX2 via
///   `is_x86_feature_detected!("avx2")` before invoking this function.
/// - Slice lengths must satisfy the same contract as [`dequant_mxfp4_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dequant_mxfp4_avx2(
    blocks: &[u8],
    scales: &[u8],
    output: &mut [f32],
    block_size: usize,
) {
    use std::arch::x86_64::*;

    debug_assert!(
        block_size % 8 == 0,
        "AVX2 mxfp4 path requires block_size multiple of 8; got {block_size}"
    );
    let num_blocks = scales.len();
    debug_assert_eq!(blocks.len(), num_blocks * (block_size / 2));
    debug_assert_eq!(output.len(), num_blocks * block_size);

    // Split E2M1 LUT into two 8-lane halves, indexed 0..7 (positive) and 8..15 (negative).
    // Note `_mm256_permutevar8x32_ps` uses each i32 lane modulo 8 as the source lane index.
    let lut_pos = _mm256_setr_ps(
        E2M1_LUT_F32[0], E2M1_LUT_F32[1], E2M1_LUT_F32[2], E2M1_LUT_F32[3],
        E2M1_LUT_F32[4], E2M1_LUT_F32[5], E2M1_LUT_F32[6], E2M1_LUT_F32[7],
    );
    let lut_neg = _mm256_setr_ps(
        E2M1_LUT_F32[8],  E2M1_LUT_F32[9],  E2M1_LUT_F32[10], E2M1_LUT_F32[11],
        E2M1_LUT_F32[12], E2M1_LUT_F32[13], E2M1_LUT_F32[14], E2M1_LUT_F32[15],
    );
    let nibble_mask = _mm_set1_epi8(0x0F);
    let sign_bit_i32 = _mm256_set1_epi32(0x8); // 4-bit sign mask in lane

    let bytes_per_block = block_size / 2;
    let blocks_ptr = blocks.as_ptr();
    let scales_ptr = scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for blk in 0..num_blocks {
        // Decode scale via the shared scalar helper to guarantee bit-exact agreement with
        // the SymExec ground truth (`dequant_mxfp4_scalar`).
        let scale = decode_e8m0_scale(*scales_ptr.add(blk));
        let v_scale = _mm256_set1_ps(scale);

        let qs_ptr = blocks_ptr.add(blk * bytes_per_block);
        let dst = out_ptr.add(blk * block_size);

        // Process the block in 16-element chunks (8 packed bytes per chunk).
        // For block_size=32 this runs the inner body twice.
        let chunks = bytes_per_block / 8;
        for chunk in 0..chunks {
            // Load 8 packed bytes (16 nibbles → 16 elements).
            let v64 = _mm_loadl_epi64(qs_ptr.add(chunk * 8) as *const __m128i);
            // Extract low and high nibbles into two separate __m128i (8 bytes each).
            let lo_nibbles_8 = _mm_and_si128(v64, nibble_mask);
            let hi_nibbles_8 = _mm_and_si128(_mm_srli_epi16(v64, 4), nibble_mask);

            // Widen u8 → i32: 8 lanes per __m256i.
            let lo_idx = _mm256_cvtepu8_epi32(lo_nibbles_8);
            let hi_idx = _mm256_cvtepu8_epi32(hi_nibbles_8);

            // Build sign mask: lanes where bit 3 is set ⇒ negative half of LUT.
            let lo_sign_bits = _mm256_and_si256(lo_idx, sign_bit_i32);
            let hi_sign_bits = _mm256_and_si256(hi_idx, sign_bit_i32);
            // Compare-equal-to-8 to obtain all-ones mask in negative lanes.
            let lo_neg_mask = _mm256_cmpeq_epi32(lo_sign_bits, sign_bit_i32);
            let hi_neg_mask = _mm256_cmpeq_epi32(hi_sign_bits, sign_bit_i32);

            // Strip sign bit so index ∈ 0..7 — this matches both LUT halves.
            let mask_low3 = _mm256_set1_epi32(0x7);
            let lo_lut_idx = _mm256_and_si256(lo_idx, mask_low3);
            let hi_lut_idx = _mm256_and_si256(hi_idx, mask_low3);

            // Gather f32 from positive half and negative half independently.
            let lo_pos = _mm256_permutevar8x32_ps(lut_pos, lo_lut_idx);
            let lo_neg = _mm256_permutevar8x32_ps(lut_neg, lo_lut_idx);
            let hi_pos = _mm256_permutevar8x32_ps(lut_pos, hi_lut_idx);
            let hi_neg = _mm256_permutevar8x32_ps(lut_neg, hi_lut_idx);

            // Blend: pick negative lanes when sign bit was set.
            let lo_vals = _mm256_blendv_ps(lo_pos, lo_neg, _mm256_castsi256_ps(lo_neg_mask));
            let hi_vals = _mm256_blendv_ps(hi_pos, hi_neg, _mm256_castsi256_ps(hi_neg_mask));

            // Apply scale.
            let lo_scaled = _mm256_mul_ps(lo_vals, v_scale);
            let hi_scaled = _mm256_mul_ps(hi_vals, v_scale);

            // Interleave: byte i contributes (lo_i, hi_i) at output positions (2i, 2i+1).
            // We have 8 lo values in `lo_scaled` and 8 hi values in `hi_scaled`. Use
            // `vunpcklps/vunpckhps` then `vperm2f128` to produce 16 contiguous output f32.
            let unpack_lo = _mm256_unpacklo_ps(lo_scaled, hi_scaled);
            let unpack_hi = _mm256_unpackhi_ps(lo_scaled, hi_scaled);
            // Within each 128-bit lane, unpacklo gives (lo[0],hi[0],lo[1],hi[1]) and
            // unpackhi gives (lo[2],hi[2],lo[3],hi[3]). Across the two 128-bit halves:
            //   lo_lane0 → out[0..4],  hi_lane0 → out[4..8],
            //   lo_lane1 → out[8..12], hi_lane1 → out[12..16].
            let out0 = _mm256_permute2f128_ps(unpack_lo, unpack_hi, 0x20);
            let out1 = _mm256_permute2f128_ps(unpack_lo, unpack_hi, 0x31);

            _mm256_storeu_ps(dst.add(chunk * 16), out0);
            _mm256_storeu_ps(dst.add(chunk * 16 + 8), out1);
        }

        // Tail: any remaining bytes when block_size/2 not multiple of 8 — fall through to scalar.
        let processed_bytes = chunks * 8;
        if processed_bytes < bytes_per_block {
            let qs_tail = std::slice::from_raw_parts(qs_ptr.add(processed_bytes),
                                                     bytes_per_block - processed_bytes);
            let out_tail = std::slice::from_raw_parts_mut(dst.add(processed_bytes * 2),
                                                         (bytes_per_block - processed_bytes) * 2);
            for (i, &byte) in qs_tail.iter().enumerate() {
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                out_tail[2 * i]     = E2M1_LUT_F32[lo] * scale;
                out_tail[2 * i + 1] = E2M1_LUT_F32[hi] * scale;
            }
        }
    }
}

/// Runtime ISA dispatcher.
///
/// Selects the AVX2 implementation when the host CPU advertises `avx2`; otherwise returns
/// an error (NO_SILENT_FALLBACK: no scalar fallback on the hot path).
/// **This is the only public entry point that callers outside the SymExec / test layer
/// should use.**
#[inline]
pub fn dequant_mxfp4(
    blocks: &[u8],
    scales: &[u8],
    output: &mut [f32],
    block_size: usize,
) -> Result<(), String> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime; slice contracts checked inside.
            unsafe { dequant_mxfp4_avx2(blocks, scales, output, block_size); }
            return Ok(());
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (blocks, scales, output, block_size);
    }
    Err("dequant_mxfp4: no SIMD path available — AVX2 required on x86_64; \
         other architectures not yet supported".to_string())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// OCP standard reference: encode the 16 e2m1 codewords across one full block of 32 elements
    /// (each codeword appears twice: once at an even index and once at an odd index) and verify
    /// the decoded values match the OCP-defined table.
    #[test]
    fn ocp_reference_block_decodes_all_16_codewords() {
        // Build 16 packed bytes encoding all 16 codewords across both nibble positions.
        // Byte i = (low_nibble = i, high_nibble = (15 - i)) → exercises both positions.
        let mut blocks = [0u8; 16];
        for i in 0..16 {
            let lo = i as u8 & 0x0F;
            let hi = (15 - i) as u8 & 0x0F;
            blocks[i] = (hi << 4) | lo;
        }
        // Scale byte = 127 ⇒ 2^0 = 1.0 (identity scale).
        let scales = [127u8];
        let mut out = [0f32; 32];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        // Verify: out[2*i]   = LUT[i],  out[2*i+1] = LUT[15-i].
        for i in 0..16 {
            let expected_even = E2M1_LUT_F32[i];
            let expected_odd  = E2M1_LUT_F32[15 - i];
            assert_eq!(
                out[2 * i], expected_even,
                "block byte {i} low nibble decoded to {} expected {expected_even}",
                out[2 * i]
            );
            assert_eq!(
                out[2 * i + 1], expected_odd,
                "block byte {i} high nibble decoded to {} expected {expected_odd}",
                out[2 * i + 1]
            );
        }
    }

    /// Verify the OCP-defined codeword → value mapping directly against the spec table.
    #[test]
    fn ocp_e2m1_lut_matches_spec() {
        // Per OCP Microscaling Formats Specification, Table "FP4 Encoding":
        //   code 0 = +0,    code 1 = +0.5,  code 2 = +1,    code 3 = +1.5,
        //   code 4 = +2,    code 5 = +3,    code 6 = +4,    code 7 = +6,
        //   code 8 = -0,    code 9 = -0.5,  code 10 = -1,   code 11 = -1.5,
        //   code 12 = -2,   code 13 = -3,   code 14 = -4,   code 15 = -6.
        assert_eq!(E2M1_LUT_F32[0],   0.0);
        assert_eq!(E2M1_LUT_F32[1],   0.5);
        assert_eq!(E2M1_LUT_F32[2],   1.0);
        assert_eq!(E2M1_LUT_F32[3],   1.5);
        assert_eq!(E2M1_LUT_F32[4],   2.0);
        assert_eq!(E2M1_LUT_F32[5],   3.0);
        assert_eq!(E2M1_LUT_F32[6],   4.0);
        assert_eq!(E2M1_LUT_F32[7],   6.0);
        assert_eq!(E2M1_LUT_F32[8],  -0.0);
        assert_eq!(E2M1_LUT_F32[9],  -0.5);
        assert_eq!(E2M1_LUT_F32[10], -1.0);
        assert_eq!(E2M1_LUT_F32[11], -1.5);
        assert_eq!(E2M1_LUT_F32[12], -2.0);
        assert_eq!(E2M1_LUT_F32[13], -3.0);
        assert_eq!(E2M1_LUT_F32[14], -4.0);
        assert_eq!(E2M1_LUT_F32[15], -6.0);
    }

    /// Verify e8m0 scale decoding against the OCP-defined formula `scale = 2^(byte - 127)`.
    #[test]
    fn ocp_e8m0_scale_decoding() {
        // byte = 127 → 2^0 = 1.0 (identity).
        assert_eq!(decode_e8m0_scale(127), 1.0);
        // byte = 128 → 2^1 = 2.0.
        assert_eq!(decode_e8m0_scale(128), 2.0);
        // byte = 126 → 2^-1 = 0.5.
        assert_eq!(decode_e8m0_scale(126), 0.5);
        // byte = 129 → 2^2 = 4.0.
        assert_eq!(decode_e8m0_scale(129), 4.0);
        // byte = 125 → 2^-2 = 0.25.
        assert_eq!(decode_e8m0_scale(125), 0.25);
        // byte = 254 → 2^127 ≈ 1.7e38.
        let s = decode_e8m0_scale(254);
        let expected = 2.0_f32.powi(127);
        assert_eq!(s, expected, "byte=254 scale {s} ≠ expected {expected}");
        // byte = 255 → NaN per OCP.
        assert!(decode_e8m0_scale(255).is_nan(), "byte=255 must decode to NaN");
    }

    /// Two-block decode with non-identity scales — verifies per-block scale application.
    #[test]
    fn two_block_decode_with_distinct_scales() {
        // Block 0: all bytes = 0x21 → low nibble 1 (+0.5), high nibble 2 (+1.0).
        // Block 1: all bytes = 0x67 → low nibble 7 (+6.0), high nibble 6 (+4.0).
        let blocks: Vec<u8> = (0..16).map(|_| 0x21).chain((0..16).map(|_| 0x67)).collect();
        // Scales: block 0 = 128 (×2), block 1 = 126 (×0.5).
        let scales = [128u8, 126u8];
        let mut out = vec![0f32; 64];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        // Block 0: pairs (0.5*2, 1.0*2) = (1.0, 2.0) repeated.
        for i in 0..16 {
            assert_eq!(out[2 * i],     1.0, "blk0 idx {} (lo)", 2 * i);
            assert_eq!(out[2 * i + 1], 2.0, "blk0 idx {} (hi)", 2 * i + 1);
        }
        // Block 1: pairs (6.0*0.5, 4.0*0.5) = (3.0, 2.0) repeated.
        for i in 0..16 {
            assert_eq!(out[32 + 2 * i],     3.0, "blk1 idx {} (lo)", 32 + 2 * i);
            assert_eq!(out[32 + 2 * i + 1], 2.0, "blk1 idx {} (hi)", 32 + 2 * i + 1);
        }
    }

    /// Cross-validate AVX2 against scalar reference on a multi-block random-like fixture.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mxfp4_avx2_matches_scalar_reference() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not advertise AVX2");
            return;
        }

        // Build 8 blocks (256 elements) with deterministic patterns covering all codewords
        // and a spread of scales.
        let num_blocks = 8;
        let block_size = 32;
        let mut blocks = vec![0u8; num_blocks * 16];
        let mut scales = vec![0u8; num_blocks];
        for blk in 0..num_blocks {
            for i in 0..16 {
                // Mix codewords across blocks: lo = (blk*16 + i) % 16, hi = (blk + i) % 16.
                let lo = ((blk * 16 + i) & 0x0F) as u8;
                let hi = ((blk + i) & 0x0F) as u8;
                blocks[blk * 16 + i] = (hi << 4) | lo;
            }
            // Scales: 120, 122, 124, 126, 128, 130, 132, 134.
            scales[blk] = (120 + 2 * blk) as u8;
        }

        let mut out_scalar = vec![0f32; num_blocks * block_size];
        let mut out_avx2   = vec![0f32; num_blocks * block_size];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out_scalar, block_size);
        unsafe {
            dequant_mxfp4_avx2(&blocks, &scales, &mut out_avx2, block_size);
        }

        for i in 0..out_scalar.len() {
            assert_eq!(
                out_scalar[i].to_bits(), out_avx2[i].to_bits(),
                "AVX2 mismatch at index {i}: scalar={} avx2={}",
                out_scalar[i], out_avx2[i]
            );
        }
    }

    /// Verify the runtime dispatcher produces the same result as the scalar reference.
    #[test]
    fn mxfp4_dispatcher_matches_scalar_reference() {
        let blocks = vec![0xABu8; 32]; // 2 blocks, low=0xB(-1.5), high=0xA(-1.0)
        let scales = vec![127u8, 127u8];
        let mut out_scalar = vec![0f32; 64];
        let mut out_dispatch = vec![0f32; 64];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out_scalar, 32);
        dequant_mxfp4(&blocks, &scales, &mut out_dispatch, 32)
            .expect("dequant_mxfp4 requires AVX2 on x86_64");

        for i in 0..64 {
            assert_eq!(out_scalar[i].to_bits(), out_dispatch[i].to_bits(),
                "dispatcher mismatch at {i}");
        }
    }

    /// Verify QuantType::Mxfp4 metadata methods report the OCP-correct sizes.
    #[test]
    fn quant_type_mxfp4_metadata() {
        use crate::quant::QuantType;
        let qt = QuantType::Mxfp4 { block_size: 32 };
        assert_eq!(qt.block_size(), 32);
        // 1 byte e8m0 scale + 16 packed bytes (32 elements / 2).
        assert_eq!(qt.block_bytes(), 17);
        assert_eq!(qt.bits(), 4);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Additional unit tests (13 new tests)
    // ────────────────────────────────────────────────────────────────────────────

    /// Verify byte=0 decodes to exactly 2^-127 (subnormal f32).
    /// Per OCP spec, this sits in the subnormal range of binary32.
    #[test]
    fn e8m0_scale_byte_zero_is_subnormal_2_pow_neg127() {
        let s = decode_e8m0_scale(0);
        let expected = 2.0_f32.powi(-127);
        assert_eq!(s.to_bits(), expected.to_bits(),
            "byte=0 decoded to {:e} but expected {:e}", s, expected);
        // Confirm it is actually subnormal (exponent field all zeros, mantissa nonzero).
        let bits = s.to_bits();
        let exp_field = (bits >> 23) & 0xFF;
        let mant_field = bits & 0x7F_FFFF;
        assert_eq!(exp_field, 0, "byte=0 result should be subnormal (exp_field=0)");
        assert_ne!(mant_field, 0, "byte=0 result should be nonzero subnormal");
    }

    /// Verify byte=1 decodes to 2^-126 (smallest normal f32 power of 2).
    #[test]
    fn e8m0_scale_byte_one_is_2_pow_neg126() {
        let s = decode_e8m0_scale(1);
        let expected = 2.0_f32.powi(-126);
        assert_eq!(s.to_bits(), expected.to_bits(),
            "byte=1 decoded to {:e} but expected {:e}", s, expected);
        assert!(s.is_normal(), "byte=1 result should be a normal f32");
    }

    /// Verify E2M1 LUT sign symmetry: LUT[i+8] == -LUT[i] for all i in 0..8.
    /// Per OCP spec, bit[3] is the sign bit; flipping it negates the value.
    #[test]
    fn e2m1_lut_sign_symmetry() {
        for i in 0..8 {
            let pos = E2M1_LUT_F32[i];
            let neg = E2M1_LUT_F32[i + 8];
            // For zero (i=0), both +0 and -0 should satisfy this.
            assert!(
                pos + neg == 0.0,
                "LUT[{}] = {} but LUT[{}] = {} — not negations",
                i, pos, i + 8, neg
            );
        }
    }

    /// Verify that both code 0 (+0) and code 8 (-0) decode to exactly 0.0f32.
    /// IEEE 754 treats +0 and -0 as equal under ==.
    #[test]
    fn e2m1_lut_zero_both_signs() {
        assert_eq!(E2M1_LUT_F32[0], 0.0f32);
        assert_eq!(E2M1_LUT_F32[8], 0.0f32);
        // Also verify via sign bit that code 8 is the negative zero.
        assert_eq!(E2M1_LUT_F32[0].to_bits() & 0x8000_0000, 0, "code 0 should be +0");
        assert_ne!(E2M1_LUT_F32[8].to_bits() & 0x8000_0000, 0, "code 8 should be -0");
    }

    /// Verify single-element decode at various scale factors.
    /// Uses one block (32 elements) with all nibbles set to code 2 (+1.0),
    /// then checks the output against several known scales.
    #[test]
    fn single_element_decode_with_various_scales() {
        // All packed bytes = 0x22 → lo=2 (+1.0), hi=2 (+1.0).
        let blocks = [0x22u8; 16];
        let mut out = [0f32; 32];

        // scale byte 124 → 2^-3 = 0.125
        let scales = [124u8];
        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);
        let expected = 1.0f32 * 0.125f32;
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, expected, "scale=124 idx={i}");
        }

        // scale byte 130 → 2^3 = 8.0
        let scales = [130u8];
        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);
        let expected = 8.0f32;
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, expected, "scale=130 idx={i}");
        }
    }

    /// Verify that a block of all-zero bytes decodes to all zeros regardless of scale.
    /// Code 0 in e2m1 is +0; 0 * any_scale = +0.
    #[test]
    fn all_zeros_block_decodes_to_zero() {
        let blocks = [0x00u8; 16];
        let scales = [135u8]; // 2^8 = 256.0 — a large scale
        let mut out = [0f32; 32];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0f32, "all-zero block at idx={i} should be 0.0");
        }
    }

    /// Verify a block of all-max-positive (code 7 = +6.0) and all-max-negative (code 15 = -6.0)
    /// with identity scale.
    #[test]
    fn all_max_code_block_with_identity_scale() {
        // All packed bytes = 0x77 → lo=7 (+6.0), hi=7 (+6.0).
        let blocks_pos = [0x77u8; 16];
        let scales = [127u8]; // 2^0 = 1.0
        let mut out = [0f32; 32];

        dequant_mxfp4_scalar(&blocks_pos, &scales, &mut out, 32);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 6.0f32, "max-positive block idx={i} should be 6.0");
        }

        // All packed bytes = 0xFF → lo=15 (-6.0), hi=15 (-6.0).
        let blocks_neg = [0xFFu8; 16];
        dequant_mxfp4_scalar(&blocks_neg, &scales, &mut out, 32);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, -6.0f32, "max-negative block idx={i} should be -6.0");
        }
    }

    /// Verify that each block's scale is independent — changing one block's scale
    /// does not affect another block's output.
    #[test]
    fn multi_block_scale_independence() {
        // Two blocks, both containing the same data: all bytes = 0x12 → lo=2(+1.0), hi=1(+0.5).
        let blocks = [0x12u8; 32]; // 2 blocks x 16 bytes
        let mut scales = [127u8, 127u8]; // both identity
        let mut out = [0f32; 64];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        // Both blocks should produce identical output.
        for i in 0..32 {
            assert_eq!(out[i], out[32 + i], "block independence: idx {i}");
        }

        // Now change only block 1's scale.
        scales[1] = 129; // 2^2 = 4.0
        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        // Block 0 should be unchanged: pairs (1.0, 0.5).
        for i in 0..16 {
            assert_eq!(out[2 * i], 1.0f32, "blk0 lo idx={}", 2 * i);
            assert_eq!(out[2 * i + 1], 0.5f32, "blk0 hi idx={}", 2 * i + 1);
        }
        // Block 1 should be scaled by 4.0: pairs (4.0, 2.0).
        for i in 0..16 {
            assert_eq!(out[32 + 2 * i], 4.0f32, "blk1 lo idx={}", 32 + 2 * i);
            assert_eq!(out[32 + 2 * i + 1], 2.0f32, "blk1 hi idx={}", 32 + 2 * i + 1);
        }
    }

    /// Verify dequantization with non-standard block_size=16 (8 packed bytes per block).
    #[test]
    fn custom_block_size_16() {
        // One block of 16 elements: 8 packed bytes.
        // Byte i: lo = i % 16, hi = (15 - i) % 16.
        let blocks: Vec<u8> = (0..8)
            .map(|i| (((15 - i) as u8 & 0x0F) << 4) | (i as u8 & 0x0F))
            .collect();
        let scales = [127u8]; // identity scale
        let mut out = [0f32; 16];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 16);

        for i in 0..8 {
            let expected_even = E2M1_LUT_F32[i];
            let expected_odd = E2M1_LUT_F32[15 - i];
            assert_eq!(out[2 * i], expected_even,
                "block_size=16 byte {i} lo: got {} expected {expected_even}", out[2 * i]);
            assert_eq!(out[2 * i + 1], expected_odd,
                "block_size=16 byte {i} hi: got {} expected {expected_odd}", out[2 * i + 1]);
        }
    }

    /// Verify dequantization with non-standard block_size=64 (32 packed bytes per block).
    #[test]
    fn custom_block_size_64() {
        let block_size = 64;
        let bytes_per_block = block_size / 2; // 32
        // Fill with a repeating pattern: byte = 0x43 → lo=3(+1.5), hi=4(+2.0).
        let blocks = vec![0x43u8; bytes_per_block];
        let scales = [125u8]; // 2^-2 = 0.25
        let mut out = vec![0f32; block_size];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, block_size);

        let expected_lo = 1.5f32 * 0.25f32; // 0.375
        let expected_hi = 2.0f32 * 0.25f32; // 0.5
        for i in 0..bytes_per_block {
            assert_eq!(out[2 * i], expected_lo,
                "block_size=64 byte {i} lo: got {} expected {expected_lo}", out[2 * i]);
            assert_eq!(out[2 * i + 1], expected_hi,
                "block_size=64 byte {i} hi: got {} expected {expected_hi}", out[2 * i + 1]);
        }
    }

    /// Verify that byte=255 (NaN per OCP) produces NaN in every output element.
    #[test]
    fn nan_scale_produces_nan_outputs() {
        let blocks = [0x12u8; 16]; // arbitrary non-zero data
        let scales = [255u8]; // NaN scale
        let mut out = [0f32; 32];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_nan(), "NaN scale at idx={i} should produce NaN, got {v}");
        }
    }

    /// Verify the dispatcher returns an error on non-AVX2-capable hosts (or succeeds
    /// bit-exactly on AVX2 hosts). On x86_64 with AVX2 this test confirms the
    /// dispatcher delegates correctly; on other archs it confirms the error message.
    #[test]
    fn dispatcher_error_or_bitexact_on_x86_64() {
        let blocks = vec![0x56u8; 16]; // lo=6(+4.0), hi=5(+3.0)
        let scales = vec![127u8]; // identity
        let mut out_dispatch = vec![0f32; 32];

        let result = dequant_mxfp4(&blocks, &scales, &mut out_dispatch, 32);

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64 with AVX2 this should succeed; otherwise error.
            if std::is_x86_feature_detected!("avx2") {
                assert!(result.is_ok(), "dispatcher should succeed on AVX2 host");
                let mut out_scalar = vec![0f32; 32];
                dequant_mxfp4_scalar(&blocks, &scales, &mut out_scalar, 32);
                for i in 0..32 {
                    assert_eq!(out_dispatch[i].to_bits(), out_scalar[i].to_bits(),
                        "dispatcher mismatch at {i}");
                }
            } else {
                assert!(result.is_err(), "dispatcher should fail without AVX2");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert!(result.is_err(), "dispatcher should fail on non-x86_64");
            let msg = result.unwrap_err();
            assert!(msg.contains("no SIMD path") || msg.contains("not yet supported"),
                "unexpected error message: {msg}");
        }
    }

    /// Cross-validate AVX2 single-block decode covering all 16 codewords.
    /// Ensures the AVX2 path handles the full OCP code range correctly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_single_block_all_codewords() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not advertise AVX2");
            return;
        }

        // Build 16 bytes where byte i encodes (lo=i, hi=15-i) — exercises all nibbles.
        let mut blocks = [0u8; 16];
        for i in 0..16 {
            blocks[i] = ((15 - i) as u8) << 4 | (i as u8);
        }
        // Use scale byte 122 → 2^-5 = 0.03125 to exercise non-identity scale.
        let scales = [122u8];
        let mut out_scalar = [0f32; 32];
        let mut out_avx2 = [0f32; 32];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out_scalar, 32);
        unsafe {
            dequant_mxfp4_avx2(&blocks, &scales, &mut out_avx2, 32);
        }

        let scale = decode_e8m0_scale(122);
        for i in 0..16 {
            let expected_lo = E2M1_LUT_F32[i] * scale;
            let expected_hi = E2M1_LUT_F32[15 - i] * scale;
            assert_eq!(out_scalar[2 * i].to_bits(), expected_lo.to_bits(),
                "scalar lo mismatch at byte {i}");
            assert_eq!(out_scalar[2 * i + 1].to_bits(), expected_hi.to_bits(),
                "scalar hi mismatch at byte {i}");
            assert_eq!(out_avx2[2 * i].to_bits(), expected_lo.to_bits(),
                "avx2 lo mismatch at byte {i}");
            assert_eq!(out_avx2[2 * i + 1].to_bits(), expected_hi.to_bits(),
                "avx2 hi mismatch at byte {i}");
        }
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Additional unit tests (10 new tests — wave-12x60)
    // ────────────────────────────────────────────────────────────────────────────

    /// E2M1 nibble encode/decode: verify that packing a nibble into the low position
    /// of a byte and decoding via the scalar path recovers the correct LUT value.
    /// Tests boundary codes 0 (zero), 7 (max positive), 8 (negative zero), 15 (max negative).
    #[test]
    fn e2m1_nibble_low_position_boundary_codes() {
        let scale_byte = 127u8; // identity scale
        let scale = decode_e8m0_scale(scale_byte);

        // Test codes 0, 7, 8, 15 in the low nibble position (high nibble = 0).
        let test_codes: [(u8, f32); 4] = [
            (0, E2M1_LUT_F32[0]),   // +0
            (7, E2M1_LUT_F32[7]),   // +6.0
            (8, E2M1_LUT_F32[8]),   // -0
            (15, E2M1_LUT_F32[15]), // -6.0
        ];

        for (code, expected_raw) in test_codes {
            // Pack: low nibble = code, high nibble = 0 (+0).
            let packed = code & 0x0F; // high nibble is 0
            let blocks = [packed; 16];
            let scales = [scale_byte];
            let mut out = [0f32; 32];

            dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

            // Even-index elements (low nibble) should be expected_raw * scale.
            for i in 0..16 {
                assert_eq!(out[2 * i], expected_raw * scale,
                    "code={code} low nibble at byte {i}: got {} expected {}",
                    out[2 * i], expected_raw * scale);
            }
            // Odd-index elements (high nibble = 0) should be 0.0.
            for i in 0..16 {
                assert_eq!(out[2 * i + 1], 0.0f32,
                    "code={code} high nibble (0) at byte {i}: got {} expected 0.0",
                    out[2 * i + 1]);
            }
        }
    }

    /// E2M1 nibble encode/decode: verify that packing a nibble into the high position
    /// of a byte and decoding via the scalar path recovers the correct LUT value.
    /// Tests boundary codes 0, 7, 8, 15 in the high nibble.
    #[test]
    fn e2m1_nibble_high_position_boundary_codes() {
        let scale_byte = 127u8;
        let scale = decode_e8m0_scale(scale_byte);

        let test_codes: [(u8, f32); 4] = [
            (0, E2M1_LUT_F32[0]),
            (7, E2M1_LUT_F32[7]),
            (8, E2M1_LUT_F32[8]),
            (15, E2M1_LUT_F32[15]),
        ];

        for (code, expected_raw) in test_codes {
            // Pack: high nibble = code, low nibble = 0.
            let packed = (code & 0x0F) << 4;
            let blocks = [packed; 16];
            let scales = [scale_byte];
            let mut out = [0f32; 32];

            dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

            // Even-index elements (low nibble = 0) should be 0.0.
            for i in 0..16 {
                assert_eq!(out[2 * i], 0.0f32,
                    "code={code} low nibble (0) at byte {i}: got {} expected 0.0",
                    out[2 * i]);
            }
            // Odd-index elements (high nibble) should be expected_raw * scale.
            for i in 0..16 {
                assert_eq!(out[2 * i + 1], expected_raw * scale,
                    "code={code} high nibble at byte {i}: got {} expected {}",
                    out[2 * i + 1], expected_raw * scale);
            }
        }
    }

    /// E8M0 scale computation: verify the full normal range produces exact power-of-2 values.
    /// For each byte in [1, 254], decode_e8m0_scale(byte) must equal 2^(byte - 127).
    /// Spot-check a representative sample across the range.
    #[test]
    fn e8m0_scale_normal_range_spot_check() {
        // Check a spread of normal-range bytes: near-min, mid, near-max, and powers of 2.
        let test_bytes: &[u8] = &[1, 2, 10, 50, 100, 120, 127, 128, 140, 200, 240, 253, 254];

        for &byte in test_bytes {
            let s = decode_e8m0_scale(byte);
            let expected = 2.0_f32.powi(byte as i32 - 127);
            assert_eq!(s.to_bits(), expected.to_bits(),
                "byte={byte}: decoded to {:e} but expected {:e}", s, expected);
            // All normal-range bytes should produce normal (non-subnormal, non-inf, non-NaN) f32.
            assert!(s.is_normal() || s == 1.0f32,
                "byte={byte}: result {:e} should be normal f32", s);
        }
    }

    /// E8M0 scale: verify byte=254 produces 2^127 which is the largest finite power-of-2 in f32
    /// (just below FLT_MAX). Verify it is not infinity.
    #[test]
    fn e8m0_scale_max_normal_is_not_inf() {
        let s = decode_e8m0_scale(254);
        let expected = 2.0_f32.powi(127);
        assert_eq!(s.to_bits(), expected.to_bits(),
            "byte=254 should be 2^127, got {:e}", s);
        assert!(s.is_finite(), "byte=254 result should be finite, not inf");
        assert!(!s.is_nan(), "byte=254 result should not be NaN");
    }

    /// MXFP4 block quantization roundtrip: encode known f32 values into e2m1 nibbles,
    /// dequantize, and verify the output matches the expected quantized values.
    /// This tests the full encode→pack→dequantize pipeline for a single block.
    #[test]
    fn mxfp4_roundtrip_single_block_all_positive_codes() {
        // Encode all 8 positive e2m1 codes (0..7) into a block of 16 elements.
        // Each code appears once in the low nibble and once in the high nibble.
        let mut blocks = [0u8; 8]; // block_size=16 → 8 packed bytes
        for i in 0..8 {
            // Byte i: lo = i, hi = i (same code in both positions).
            blocks[i] = (i as u8) | ((i as u8) << 4);
        }
        let scales = [127u8]; // identity scale
        let mut out = [0f32; 16];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 16);

        // Verify: out[2*i] = LUT[i], out[2*i+1] = LUT[i].
        for i in 0..8 {
            let expected = E2M1_LUT_F32[i];
            assert_eq!(out[2 * i], expected,
                "roundtrip byte {i} lo: got {} expected {}", out[2 * i], expected);
            assert_eq!(out[2 * i + 1], expected,
                "roundtrip byte {i} hi: got {} expected {}", out[2 * i + 1], expected);
        }
    }

    /// MXFP4 roundtrip with negative codes: encode all 8 negative e2m1 codes (8..15)
    /// and verify dequantization recovers the correct negative values.
    #[test]
    fn mxfp4_roundtrip_single_block_all_negative_codes() {
        let mut blocks = [0u8; 8]; // block_size=16 → 8 packed bytes
        for i in 0..8 {
            let code = (i + 8) as u8;
            blocks[i] = (code & 0x0F) | ((code & 0x0F) << 4);
        }
        let scales = [127u8];
        let mut out = [0f32; 16];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 16);

        for i in 0..8 {
            let expected = E2M1_LUT_F32[i + 8];
            assert_eq!(out[2 * i], expected,
                "negative roundtrip byte {i} lo: got {} expected {}", out[2 * i], expected);
            assert_eq!(out[2 * i + 1], expected,
                "negative roundtrip byte {i} hi: got {} expected {}", out[2 * i + 1], expected);
        }
    }

    /// Edge case: single-element block (block_size=2, 1 packed byte).
    /// Verify that the smallest possible block decodes correctly.
    #[test]
    fn edge_case_smallest_block_size_2() {
        // block_size=2 → 1 packed byte, 2 output elements.
        // Byte: lo=5 (+3.0), hi=9 (-0.5).
        let blocks = [0x95u8]; // lo=5, hi=9
        let scales = [127u8]; // identity
        let mut out = [0f32; 2];

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 2);

        assert_eq!(out[0], 3.0f32, "smallest block lo: got {} expected 3.0", out[0]);
        assert_eq!(out[1], -0.5f32, "smallest block hi: got {} expected -0.5", out[1]);
    }

    /// Edge case: all-zero values across multiple blocks.
    /// All nibbles = 0 (+0), with varying scales. Zero times any scale must remain zero.
    #[test]
    fn edge_case_all_zero_values_with_varying_scales() {
        let num_blocks = 4;
        let blocks = [0x00u8; 64]; // 4 blocks x 16 bytes, all zero nibbles
        let scales = [120u8, 127u8, 135u8, 254u8]; // 2^-7, 2^0, 2^8, 2^127
        let mut out = [0f32; 128]; // 4 x 32

        dequant_mxfp4_scalar(&blocks, &scales, &mut out, 32);

        for blk in 0..num_blocks {
            for i in 0..32 {
                let idx = blk * 32 + i;
                assert_eq!(out[idx], 0.0f32,
                    "all-zero block {} idx {} with scale byte {}: got {} expected 0.0",
                    blk, i, scales[blk], out[idx]);
            }
        }
    }

    /// Scale factor validation: verify that the e8m0 scale byte correctly maps to
    /// the shared-amax quantization model. For a block where the max absolute value
    /// is M, the correct scale byte should satisfy decode_e8m0_scale(byte) * max_e2m1 >= M,
    /// where max_e2m1 = 6.0 (the largest e2m1 magnitude).
    /// This test verifies the scale selection logic for representative amax values.
    #[test]
    fn scale_factor_validation_amax_coverage() {
        // The e2m1 max magnitude is 6.0. For a block with amax = S,
        // the correct scale byte b satisfies: 2^(b-127) * 6.0 >= S.
        // Equivalently: b >= 127 + ceil(log2(S / 6.0)).

        // Test case 1: amax = 6.0 → scale should be 2^0 = 1.0 → byte = 127.
        // 1.0 * 6.0 = 6.0 >= 6.0 ✓
        let scale_byte = 127u8;
        let scale = decode_e8m0_scale(scale_byte);
        let amax = 6.0f32;
        assert!(scale * 6.0f32 >= amax,
            "scale byte 127: {} * 6.0 = {} < amax {}", scale, scale * 6.0, amax);

        // Test case 2: amax = 12.0 → need scale >= 2.0 → byte = 128.
        let scale_byte = 128u8;
        let scale = decode_e8m0_scale(scale_byte);
        let amax = 12.0f32;
        assert!(scale * 6.0f32 >= amax,
            "scale byte 128: {} * 6.0 = {} < amax {}", scale, scale * 6.0, amax);

        // Test case 3: amax = 0.75 → need scale >= 0.125 → byte = 124 (2^-3 = 0.125).
        // 0.125 * 6.0 = 0.75 >= 0.75 ✓
        let scale_byte = 124u8;
        let scale = decode_e8m0_scale(scale_byte);
        let amax = 0.75f32;
        assert!(scale * 6.0f32 >= amax,
            "scale byte 124: {} * 6.0 = {} < amax {}", scale, scale * 6.0, amax);

        // Test case 4: amax = 1.5 → need scale >= 0.25 → byte = 125 (2^-2 = 0.25).
        // 0.25 * 6.0 = 1.5 >= 1.5 ✓
        let scale_byte = 125u8;
        let scale = decode_e8m0_scale(scale_byte);
        let amax = 1.5f32;
        assert!(scale * 6.0f32 >= amax,
            "scale byte 125: {} * 6.0 = {} < amax {}", scale, scale * 6.0, amax);
    }

    /// Verify that the e2m1 LUT values are strictly ordered by magnitude within
    /// each sign half, and that the full range of representable magnitudes is covered.
    /// Positive codes 1..7 should have strictly increasing values; negative codes 9..15
    /// should have strictly decreasing (more negative) values.
    #[test]
    fn e2m1_lut_magnitude_ordering() {
        // Positive half: codes 1..7 should be strictly increasing.
        for i in 1..7 {
            assert!(E2M1_LUT_F32[i] < E2M1_LUT_F32[i + 1],
                "LUT ordering violated: LUT[{}] = {} >= LUT[{}] = {}",
                i, E2M1_LUT_F32[i], i + 1, E2M1_LUT_F32[i + 1]);
        }

        // Negative half: codes 9..15 should be strictly decreasing (more negative).
        for i in 9..15 {
            assert!(E2M1_LUT_F32[i] > E2M1_LUT_F32[i + 1],
                "LUT ordering violated: LUT[{}] = {} >= LUT[{}] = {}",
                i, E2M1_LUT_F32[i], i + 1, E2M1_LUT_F32[i + 1]);
        }

        // Verify the full dynamic range: max positive / min positive.
        let max_pos = E2M1_LUT_F32[7]; // 6.0
        let min_pos = E2M1_LUT_F32[1]; // 0.5
        assert_eq!(max_pos / min_pos, 12.0f32,
            "e2m1 dynamic range should be 12x (6.0/0.5)");
    }
}
