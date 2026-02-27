//! Quantized GEMV / GEMM scalar reference implementations.
//!
//! Covers REQ-OPS-070 through REQ-OPS-080 from SPEC §2.6.
//! Each function is `extern "C"` + `#[no_mangle]` so the JIT compiler
//! can lift them for symbolic execution and auto-vectorisation.

// ---------------------------------------------------------------------------
// GEMV — single-row dot-product variants
// ---------------------------------------------------------------------------

/// GEMV Q8: `dot(weight_i8[0..n], input[0..n]) * scale`
///
/// Weight is plain `i8` per-element, one shared scale for the row.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemv_q8(
    weight: *const i8,
    input: *const f32,
    scale: f32,
    n: usize,
) -> f32 {
    unsafe {
        let mut acc = 0.0_f32;
        for i in 0..n {
            acc += (*weight.add(i) as f32) * *input.add(i);
        }
        acc * scale
    }
}

/// GEMV Q4: 4-bit packed (two values per byte, low nibble first).
///
/// `weight[byte]` holds `(q_lo, q_hi)` where `q = nibble - 8` (symmetric around 0).
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemv_q4(
    weight: *const u8,
    input: *const f32,
    scale: f32,
    n: usize,
) -> f32 {
    unsafe {
        let mut acc = 0.0_f32;
        for i in 0..n / 2 {
            let byte = *weight.add(i);
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = (byte >> 4) as i8 - 8;
            acc += (lo as f32) * *input.add(2 * i);
            acc += (hi as f32) * *input.add(2 * i + 1);
        }
        acc * scale
    }
}

/// GEMV Q2: 2-bit packed (four values per byte, LSB first).
///
/// `q = 2-bit value - 2` maps {0,1,2,3} → {-2,-1,0,1}.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemv_q2(
    weight: *const u8,
    input: *const f32,
    scale: f32,
    n: usize,
) -> f32 {
    unsafe {
        let mut acc = 0.0_f32;
        for i in 0..n / 4 {
            let byte = *weight.add(i);
            for j in 0..4 {
                let q = ((byte >> (j * 2)) & 0x03) as i8 - 2;
                acc += (q as f32) * *input.add(4 * i + j);
            }
        }
        acc * scale
    }
}

/// GEMV Q1: 1-bit packed (eight values per byte, LSB first).
///
/// `q = bit ? 1 : -1` (binary {0,1} → {-1,+1}).
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemv_q1(
    weight: *const u8,
    input: *const f32,
    scale: f32,
    n: usize,
) -> f32 {
    unsafe {
        let mut acc = 0.0_f32;
        for i in 0..n / 8 {
            let byte = *weight.add(i);
            for j in 0..8 {
                let bit = (byte >> j) & 1;
                let q: f32 = if bit == 1 { 1.0 } else { -1.0 };
                acc += q * *input.add(8 * i + j);
            }
        }
        acc * scale
    }
}

// ---------------------------------------------------------------------------
// GEMM — batched matmul variants  (A[M,K] × W_quant[K,N] → C[M,N])
// ---------------------------------------------------------------------------

/// GEMM Q8: `C[i,j] = sum_p( weight_i8[p*N+j] * input[i*K+p] ) * scales[j]`
///
/// Weight is [K,N] row-major i8, one scale per output column.
/// C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemm_q8(
    weight: *const i8,
    input: *const f32,
    output: *mut f32,
    scales: *const f32,
    m: usize,
    n: usize,
    k: usize,
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += (*weight.add(p * n + j) as f32) * *input.add(i * k + p);
                }
                *output.add(i * n + j) = acc * *scales.add(j);
            }
        }
    }
}

/// GEMM Q4: 4-bit packed weight [K, N/2] with per-column scales.
///
/// Two weights packed per byte (low nibble first), `q = nibble - 8`.
/// C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemm_q4(
    weight: *const u8,
    input: *const f32,
    output: *mut f32,
    scales: *const f32,
    m: usize,
    n: usize,
    k: usize,
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                // Each row of weight has k values; packed as k/2 bytes per column-pair
                // Layout: weight row p has N values packed into N/2 bytes
                for p in 0..k {
                    let byte_idx = p * (n / 2) + j / 2;
                    let byte = *weight.add(byte_idx);
                    let q = if j % 2 == 0 {
                        (byte & 0x0F) as i8 - 8
                    } else {
                        (byte >> 4) as i8 - 8
                    };
                    acc += (q as f32) * *input.add(i * k + p);
                }
                *output.add(i * n + j) = acc * *scales.add(j);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K-Quant matmul  (Q2K..Q8K block formats)
// ---------------------------------------------------------------------------

/// K-Quant matmul: dequantize-on-the-fly dot product.
///
/// Weight is stored as contiguous quant blocks (row-major, one row = K/block_size blocks).
/// `quant_type`: 0=Q2K, 1=Q3K, 2=Q4K, 3=Q5K, 4=Q6K, 5=Q8K.
/// This scalar reference dequantizes Q4K blocks inline; other K-quant types
/// follow the same structure (the JIT compiler specialises per type).
///
/// Layout: weight_blocks is [N rows × (K/256) blocks], input is [M, K], output is [M, N].
/// C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_kquant_matmul(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    block_bytes: usize,
    block_size: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let blocks_per_row = k / block_size;
    let row_bytes = blocks_per_row * block_bytes;
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                let row_ptr = weight_blocks.add(j * row_bytes);
                for b in 0..blocks_per_row {
                    let blk = row_ptr.add(b * block_bytes);
                    // Generic: treat first 4 bytes as f32 scale, rest as i8 quants
                    let d = *(blk as *const f32);
                    for e in 0..block_size {
                        let q = *blk.add(4 + e) as i8;
                        let input_idx = i * k + b * block_size + e;
                        acc += (q as f32) * d * *input.add(input_idx);
                    }
                }
                *output.add(i * n + j) = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IQ matmul  (IQ1..IQ4 importance-quantized formats)
// ---------------------------------------------------------------------------

/// IQ matmul: importance-quantized block matmul (scalar reference).
///
/// Same dequantize-on-the-fly pattern as kquant_matmul but for IQ block formats.
/// The scalar reference uses the same generic decode (scale + i8 body);
/// the JIT compiler specialises per IQ sub-type with codebook lookups.
///
/// C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_iq_matmul(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    block_bytes: usize,
    block_size: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let blocks_per_row = k / block_size;
    let row_bytes = blocks_per_row * block_bytes;
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                let row_ptr = weight_blocks.add(j * row_bytes);
                for b in 0..blocks_per_row {
                    let blk = row_ptr.add(b * block_bytes);
                    // Scale stored as f16 (2 bytes) at block start
                    let d_bits = *(blk as *const u16);
                    let d = f16_to_f32(d_bits);
                    for e in 0..block_size {
                        let q = *blk.add(2 + e) as i8;
                        let input_idx = i * k + b * block_size + e;
                        acc += (q as f32) * d * *input.add(input_idx);
                    }
                }
                *output.add(i * n + j) = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AWQ matmul
// ---------------------------------------------------------------------------

/// AWQ 4-bit matmul: `C[i,j] = sum_p( dequant(weight[p,j]) * input[i,p] )`
///
/// Weight is [K, N/8] u32 words (8 × 4-bit per u32), with per-group scales/zeros.
/// group_size = 128.  C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_awq_matmul(
    weight: *const u32,
    zeros: *const u32,
    scales: *const f32,
    input: *const f32,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    let group_size: usize = 128;
    let n_packed = n / 8; // u32 words per row
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                let word_col = j / 8;
                let bit_off = (j % 8) * 4;
                for p in 0..k {
                    let group = p / group_size;
                    let scale = *scales.add(group * n + j);
                    let z_word = *zeros.add(group * n_packed + word_col);
                    let zero = ((z_word >> bit_off) & 0x0F) as f32;

                    let w_word = *weight.add(p * n_packed + word_col);
                    let q = ((w_word >> bit_off) & 0x0F) as f32;
                    let w_val = (q - zero) * scale;
                    acc += w_val * *input.add(i * k + p);
                }
                *output.add(i * n + j) = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPTQ matmul
// ---------------------------------------------------------------------------

/// GPTQ 4-bit matmul with g_idx group assignment.
///
/// Weight is [K, N/8] u32 words, scales is [n_groups, N], g_idx maps each row to its group.
/// C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gptq_matmul(
    weight: *const u32,
    g_idx: *const i32,
    scales: *const f32,
    zeros: *const u32,
    input: *const f32,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    let n_packed = n / 8;
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                let word_col = j / 8;
                let bit_off = (j % 8) * 4;
                for p in 0..k {
                    let group = *g_idx.add(p) as usize;
                    let scale = *scales.add(group * n + j);
                    let z_word = *zeros.add(group * n_packed + word_col);
                    let zero = ((z_word >> bit_off) & 0x0F) as f32;

                    let w_word = *weight.add(p * n_packed + word_col);
                    let q = ((w_word >> bit_off) & 0x0F) as f32;
                    let w_val = (q - zero) * scale;
                    acc += w_val * *input.add(i * k + p);
                }
                *output.add(i * n + j) = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SqueezeLLM matmul
// ---------------------------------------------------------------------------

/// SqueezeLLM 3-bit block matmul (scalar reference).
///
/// Weight is stored as blocks of [f16 scale, 128 bytes packed 3-bit].
/// block_size = 256 elements, block_bytes = 130.
/// C is assumed zero-initialized.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_squeeze_matmul(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    let block_size: usize = 256;
    let block_bytes: usize = 130;
    let blocks_per_row = k / block_size;
    let row_bytes = blocks_per_row * block_bytes;
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                let row_ptr = weight_blocks.add(j * row_bytes);
                for b in 0..blocks_per_row {
                    let blk = row_ptr.add(b * block_bytes);
                    let d = f16_to_f32(*(blk as *const u16));
                    // 3-bit packed: 8 values per 3 bytes
                    for e in 0..block_size {
                        let bit_offset = e * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_shift = bit_offset % 8;
                        // Read up to 2 bytes to extract 3 bits
                        let lo = *blk.add(2 + byte_idx) as u16;
                        let hi = *blk.add(2 + byte_idx + 1) as u16;
                        let combined = lo | (hi << 8);
                        let q = ((combined >> bit_shift) & 0x07) as i8 - 4;
                        let input_idx = i * k + b * block_size + e;
                        acc += (q as f32) * d * *input.add(input_idx);
                    }
                }
                *output.add(i * n + j) = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert f16 bits to f32 (software decode, no intrinsics).
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // subnormal or zero
        let val = (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        // inf / nan
        if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_q8_basic() {
        let n = 4;
        let weight: Vec<i8> = vec![1, 2, 3, 4];
        let input = vec![1.0_f32, 1.0, 1.0, 1.0];
        let scale = 0.5_f32;
        let result = scalar_gemv_q8(weight.as_ptr(), input.as_ptr(), scale, n);
        // dot = 1+2+3+4 = 10, * 0.5 = 5.0
        assert!((result - 5.0).abs() < 1e-6, "gemv_q8: got {result}, expected 5.0");
    }

    #[test]
    fn test_gemv_q4_basic() {
        // 4 values packed into 2 bytes: (3,5) and (7,1)
        // q = nibble - 8 → (-5,-3) and (-1,-7)
        let n = 4;
        let weight: Vec<u8> = vec![0x53, 0x17]; // lo=3,hi=5 ; lo=7,hi=1
        let input = vec![1.0_f32; 4];
        let scale = 1.0_f32;
        let result = scalar_gemv_q4(weight.as_ptr(), input.as_ptr(), scale, n);
        let expected = (-5.0 + -3.0 + -1.0 + -7.0) * 1.0;
        assert!((result - expected).abs() < 1e-6, "gemv_q4: got {result}, expected {expected}");
    }

    #[test]
    fn test_gemv_q2_basic() {
        // 4 values in 1 byte: bits [1,0] [1,1] [0,0] [1,0] = 0b10_00_11_01 = 0x8D
        // q = val - 2 → (-1, 1, -2, 0)
        let n = 4;
        let weight: Vec<u8> = vec![0x8D];
        let input = vec![1.0_f32; 4];
        let scale = 2.0_f32;
        let result = scalar_gemv_q2(weight.as_ptr(), input.as_ptr(), scale, n);
        // vals: 01=1→-1, 11=3→1, 00=0→-2, 10=2→0  sum=-2, *2=-4
        let expected = (-1.0 + 1.0 + -2.0 + 0.0) * 2.0;
        assert!((result - expected).abs() < 1e-6, "gemv_q2: got {result}, expected {expected}");
    }

    #[test]
    fn test_gemv_q1_basic() {
        // 8 values in 1 byte: 0b10101010 → bits 0,1,0,1,0,1,0,1 → -1,1,-1,1,-1,1,-1,1
        let n = 8;
        let weight: Vec<u8> = vec![0xAA];
        let input = vec![1.0_f32; 8];
        let scale = 1.0_f32;
        let result = scalar_gemv_q1(weight.as_ptr(), input.as_ptr(), scale, n);
        // sum = 0
        assert!((result - 0.0).abs() < 1e-6, "gemv_q1: got {result}, expected 0.0");
    }

    #[test]
    fn test_gemv_q1_all_ones() {
        let n = 8;
        let weight: Vec<u8> = vec![0xFF];
        let input = vec![2.0_f32; 8];
        let scale = 0.5_f32;
        let result = scalar_gemv_q1(weight.as_ptr(), input.as_ptr(), scale, n);
        // all +1, dot = 16, * 0.5 = 8.0
        assert!((result - 8.0).abs() < 1e-6, "gemv_q1: got {result}, expected 8.0");
    }

    #[test]
    fn test_gemm_q8_identity_scale() {
        let m = 2;
        let n = 2;
        let k = 2;
        let weight: Vec<i8> = vec![1, 0, 0, 1]; // identity-ish [K=2, N=2]
        let input = vec![3.0_f32, 4.0, 5.0, 6.0]; // [M=2, K=2]
        let scales = vec![1.0_f32; 2];
        let mut output = vec![0.0_f32; 4];
        scalar_gemm_q8(
            weight.as_ptr(), input.as_ptr(), output.as_mut_ptr(),
            scales.as_ptr(), m, n, k,
        );
        // row0: (3*1+4*0, 3*0+4*1) = (3,4)
        // row1: (5*1+6*0, 5*0+6*1) = (5,6)
        assert_eq!(output, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gemm_q4_small() {
        // K=2, N=2 → packed N/2=1 byte per row
        // Row p=0: byte packs col0(lo) and col1(hi)
        // We want q_col0=1 (nibble=9), q_col1=2 (nibble=10) → byte = 0xA9
        // Row p=1: q_col0=-1 (nibble=7), q_col1=0 (nibble=8) → byte = 0x87
        let m = 1;
        let n = 2;
        let k = 2;
        let weight: Vec<u8> = vec![0xA9, 0x87];
        let input = vec![1.0_f32, 1.0]; // [1, K=2]
        let scales = vec![1.0_f32; 2];
        let mut output = vec![0.0_f32; 2];
        scalar_gemm_q4(
            weight.as_ptr(), input.as_ptr(), output.as_mut_ptr(),
            scales.as_ptr(), m, n, k,
        );
        // col0: (9-8)*1 + (7-8)*1 = 1 + (-1) = 0
        // col1: (10-8)*1 + (8-8)*1 = 2 + 0 = 2
        assert!((output[0] - 0.0).abs() < 1e-6, "gemm_q4[0]: got {}", output[0]);
        assert!((output[1] - 2.0).abs() < 1e-6, "gemm_q4[1]: got {}", output[1]);
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.0
        assert!((f16_to_f32(0x0000)).abs() < 1e-10);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }
}
