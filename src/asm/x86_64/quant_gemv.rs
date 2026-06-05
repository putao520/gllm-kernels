//! Fused multi-row quantized GEMV microkernels for AVX2 and AVX-512.
//!
//! Key optimization: for GEMV (n=1), process multiple M rows simultaneously.
//! Each row shares the same input vector, so input loads are amortized across rows.
//! 4-row unroll gives 4x bandwidth saving on input loads.
//!
//! Optimizations applied:
//! - Flattened array accumulators to named variables (guarantees YMM/ZMM register allocation)
//! - 2x block unrolling in main loops (overlaps prefetch with compute, reduces loop overhead)
//! - Q4K tail uses 4 independent accumulators (breaks serial FMA dependency chain)
//! - AVX-512 variants using ZMM registers (2x wider SIMD)
//! - Q8K AVX2: global_asm! microkernel for precise register control and software pipelining
//!
//! Supported formats: Q8_K, Q4_K.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::global_asm;

const Q8K_BLOCK_BYTES: usize = 292;
const Q8K_QS_OFFSET: usize = 4;
const QK_K: usize = 256;

/// Horizontal sum of __m256 register.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let h1 = _mm256_hadd_ps(v, v);
    let h2 = _mm256_hadd_ps(h1, h1);
    let hi = _mm256_extractf128_ps(h2, 1);
    let lo = _mm256_castps256_ps128(h2);
    _mm_cvtss_f32(_mm_add_ps(hi, lo))
}

/// Horizontal sum of __m512 register — reduces to AVX2 hsum.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx512(v: __m512) -> f32 {
    let lo = _mm512_castps512_ps256(v);
    let hi = _mm256_castsi256_ps(_mm512_extracti32x8_epi32(_mm512_castps_si512(v), 1));
    hsum_avx2(_mm256_add_ps(lo, hi))
}

/// Inner block dot for one row: accumulate raw i8*f32 dot into 4 ymm accumulators.
/// Returns the reduced scalar (before d multiplication).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn q8k_block_dot_1row(qs: *const i8, input: *const f32) -> f32 {
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i < 256 {
        let o0 = _mm256_loadu_ps(input.add(i));
        let o1 = _mm256_loadu_ps(input.add(i + 8));
        let o2 = _mm256_loadu_ps(input.add(i + 16));
        let o3 = _mm256_loadu_ps(input.add(i + 24));
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(i) as *const _)));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(i + 8) as *const _)));
        let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(i + 16) as *const _)));
        let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(i + 24) as *const _)));
        a0 = _mm256_fmadd_ps(q0, o0, a0);
        a1 = _mm256_fmadd_ps(q1, o1, a1);
        a2 = _mm256_fmadd_ps(q2, o2, a2);
        a3 = _mm256_fmadd_ps(q3, o3, a3);
        i += 32;
    }
    let s01 = _mm256_add_ps(a0, a1);
    let s23 = _mm256_add_ps(a2, a3);
    hsum_avx2(_mm256_add_ps(s01, s23))
}

// ============================================================================
// Q8_K Fused GEMV — AVX2
// ============================================================================

/// Process one Q8K block for 4 rows with flattened accumulators.
/// Accumulates into the 16 named accumulators (4 per row).
#[cfg(target_arch = "x86_64")]
macro_rules! q8k_block_4row_avx2 {
    ($inp:expr, $qs0:expr, $qs1:expr, $qs2:expr, $qs3:expr,
     $a00:expr, $a01:expr, $a02:expr, $a03:expr,
     $a10:expr, $a11:expr, $a12:expr, $a13:expr,
     $a20:expr, $a21:expr, $a22:expr, $a23:expr,
     $a30:expr, $a31:expr, $a32:expr, $a33:expr) => {{
        let mut i = 0usize;
        while i < 256 {
            // Shared input load (amortized across 4 rows)
            let o0 = _mm256_loadu_ps($inp.add(i));
            let o1 = _mm256_loadu_ps($inp.add(i + 8));
            let o2 = _mm256_loadu_ps($inp.add(i + 16));
            let o3 = _mm256_loadu_ps($inp.add(i + 24));

            // Row 0
            $a00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs0.add(i) as *const _))), o0, $a00);
            $a01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs0.add(i+8) as *const _))), o1, $a01);
            $a02 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs0.add(i+16) as *const _))), o2, $a02);
            $a03 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs0.add(i+24) as *const _))), o3, $a03);
            // Row 1
            $a10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs1.add(i) as *const _))), o0, $a10);
            $a11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs1.add(i+8) as *const _))), o1, $a11);
            $a12 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs1.add(i+16) as *const _))), o2, $a12);
            $a13 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs1.add(i+24) as *const _))), o3, $a13);
            // Row 2
            $a20 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs2.add(i) as *const _))), o0, $a20);
            $a21 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs2.add(i+8) as *const _))), o1, $a21);
            $a22 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs2.add(i+16) as *const _))), o2, $a22);
            $a23 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs2.add(i+24) as *const _))), o3, $a23);
            // Row 3
            $a30 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs3.add(i) as *const _))), o0, $a30);
            $a31 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs3.add(i+8) as *const _))), o1, $a31);
            $a32 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs3.add(i+16) as *const _))), o2, $a32);
            $a33 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($qs3.add(i+24) as *const _))), o3, $a33);

            i += 32;
        }
    }};
}

/// Reduce 4 named accumulators to scalar via hsum_avx2.
#[cfg(target_arch = "x86_64")]
macro_rules! reduce4_avx2 {
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
        hsum_avx2(_mm256_add_ps(_mm256_add_ps($a0, $a1), _mm256_add_ps($a2, $a3)))
    };
}

// ============================================================================
// Q8_K GEMV — AVX2 global_asm! microkernel
// ============================================================================
//
// `_gllm_gemv_q8k_block_4row_avx2`:
//   Process one Q8K block (256 i8 elements) for 4 rows simultaneously.
//   Input loads are shared across all 4 rows (4x amortization).
//
// Calling convention (System V AMD64 ABI, extern "C"):
//   rdi = qs0: *const i8    (row 0 quantized weights, 256 bytes)
//   rsi = qs1: *const i8    (row 1)
//   rdx = qs2: *const i8    (row 2)
//   rcx = qs3: *const i8    (row 3)
//   r8  = input: *const f32 (256 f32 values, shared across rows)
//   r9  = out: *mut f32     (4 f32 output slots, written with partial dot results)
//
// Register allocation (16 YMM registers):
//   ymm0 - ymm1  : row0 accumulators (2 × 8 f32 = 16 lanes)
//   ymm2 - ymm3  : row1 accumulators
//   ymm4 - ymm5  : row2 accumulators
//   ymm6 - ymm7  : row3 accumulators
//   ymm8          : shared input[i..i+8]    (8 f32, reused each iter)
//   ymm9          : shared input[i+8..i+16] (8 f32, reused each iter)
//   ymm10         : i8→f32 scratch (low 8 elements)
//   ymm11         : i8→f32 scratch (high 8 elements)
//   xmm12         : vmovq scratch for 8-byte i8 loads
//
// Inner loop: 16 iterations × 16 elements = 256 elements per block.
// Each iteration:
//   1. Load input[i..i+8] → ymm8, input[i+8..i+16] → ymm9  (shared, 1 load per row)
//   2. For each row r (0..3):
//      vmovq   xmm12, [qsr + i]       ; load 8 i8
//      vpmovsxbd ymm10, xmm12         ; sign-extend i8→i32 (8 lanes)
//      vcvtdq2ps ymm10, ymm10         ; i32→f32
//      vfmadd231ps ymmR0, ymm10, ymm8 ; acc_lo += q_lo * inp_lo
//      vmovq   xmm12, [qsr + i + 8]   ; load next 8 i8
//      vpmovsxbd ymm11, xmm12
//      vcvtdq2ps ymm11, ymm11
//      vfmadd231ps ymmR1, ymm11, ymm9 ; acc_hi += q_hi * inp_hi
//
// Horizontal reduction: vaddps pair + vhaddps×2 + vextractf128 + vaddss
// → 4 scalars written to out[0..3].

#[cfg(target_arch = "x86_64")]
global_asm!(
    ".text",
    ".align 64",
    ".global _gllm_gemv_q8k_block_4row_avx2",
    ".type _gllm_gemv_q8k_block_4row_avx2, @function",
    "_gllm_gemv_q8k_block_4row_avx2:",

    // All YMM registers are caller-saved in SysV AMD64 ABI — no push/pop needed.

    // Zero all 8 accumulators (2 per row × 4 rows)
    "vxorps ymm0, ymm0, ymm0",
    "vxorps ymm1, ymm1, ymm1",
    "vxorps ymm2, ymm2, ymm2",
    "vxorps ymm3, ymm3, ymm3",
    "vxorps ymm4, ymm4, ymm4",
    "vxorps ymm5, ymm5, ymm5",
    "vxorps ymm6, ymm6, ymm6",
    "vxorps ymm7, ymm7, ymm7",

    // r10 = loop counter: 16 iterations × 16 elements = 256 total
    "mov r10, 16",

    ".align 32",
    ".Lq8k_4row_loop:",

    // ── Shared input load: 16 f32 = 64 bytes ──
    "vmovups ymm8, [r8]",
    "vmovups ymm9, [r8 + 32]",

    // ── Row 0 (qs0 = rdi): low 8 then high 8 ──
    "vmovq    xmm12, [rdi]",
    "vpmovsxbd ymm10, xmm12",
    "vcvtdq2ps ymm10, ymm10",
    "vfmadd231ps ymm0, ymm10, ymm8",
    "vmovq    xmm12, [rdi + 8]",
    "vpmovsxbd ymm11, xmm12",
    "vcvtdq2ps ymm11, ymm11",
    "vfmadd231ps ymm1, ymm11, ymm9",

    // ── Row 1 (qs1 = rsi) ──
    "vmovq    xmm12, [rsi]",
    "vpmovsxbd ymm10, xmm12",
    "vcvtdq2ps ymm10, ymm10",
    "vfmadd231ps ymm2, ymm10, ymm8",
    "vmovq    xmm12, [rsi + 8]",
    "vpmovsxbd ymm11, xmm12",
    "vcvtdq2ps ymm11, ymm11",
    "vfmadd231ps ymm3, ymm11, ymm9",

    // ── Row 2 (qs2 = rdx) ──
    "vmovq    xmm12, [rdx]",
    "vpmovsxbd ymm10, xmm12",
    "vcvtdq2ps ymm10, ymm10",
    "vfmadd231ps ymm4, ymm10, ymm8",
    "vmovq    xmm12, [rdx + 8]",
    "vpmovsxbd ymm11, xmm12",
    "vcvtdq2ps ymm11, ymm11",
    "vfmadd231ps ymm5, ymm11, ymm9",

    // ── Row 3 (qs3 = rcx) ──
    "vmovq    xmm12, [rcx]",
    "vpmovsxbd ymm10, xmm12",
    "vcvtdq2ps ymm10, ymm10",
    "vfmadd231ps ymm6, ymm10, ymm8",
    "vmovq    xmm12, [rcx + 8]",
    "vpmovsxbd ymm11, xmm12",
    "vcvtdq2ps ymm11, ymm11",
    "vfmadd231ps ymm7, ymm11, ymm9",

    // Advance all pointers: qs += 16 bytes (16 i8), input += 64 bytes (16 f32)
    "add rdi, 16",
    "add rsi, 16",
    "add rdx, 16",
    "add rcx, 16",
    "add r8,  64",

    "dec r10",
    "jnz .Lq8k_4row_loop",

    // ── Horizontal reduce each row pair → scalar ──
    // Pattern: vaddps(acc_lo, acc_hi) → vhaddps×2 → vextractf128 → vaddss
    // This is the same hsum_avx2 pattern used in the intrinsics path.

    // Row 0: ymm0 + ymm1 → out[0]
    "vaddps ymm0, ymm0, ymm1",
    "vhaddps ymm0, ymm0, ymm0",
    "vhaddps ymm0, ymm0, ymm0",
    "vextractf128 xmm1, ymm0, 1",
    "vaddss xmm0, xmm0, xmm1",
    "vmovss [r9],      xmm0",

    // Row 1: ymm2 + ymm3 → out[1]
    "vaddps ymm2, ymm2, ymm3",
    "vhaddps ymm2, ymm2, ymm2",
    "vhaddps ymm2, ymm2, ymm2",
    "vextractf128 xmm3, ymm2, 1",
    "vaddss xmm2, xmm2, xmm3",
    "vmovss [r9 + 4],  xmm2",

    // Row 2: ymm4 + ymm5 → out[2]
    "vaddps ymm4, ymm4, ymm5",
    "vhaddps ymm4, ymm4, ymm4",
    "vhaddps ymm4, ymm4, ymm4",
    "vextractf128 xmm5, ymm4, 1",
    "vaddss xmm4, xmm4, xmm5",
    "vmovss [r9 + 8],  xmm4",

    // Row 3: ymm6 + ymm7 → out[3]
    "vaddps ymm6, ymm6, ymm7",
    "vhaddps ymm6, ymm6, ymm6",
    "vhaddps ymm6, ymm6, ymm6",
    "vextractf128 xmm7, ymm6, 1",
    "vaddss xmm6, xmm6, xmm7",
    "vmovss [r9 + 12], xmm6",

    "vzeroupper",
    "ret",

    ".size _gllm_gemv_q8k_block_4row_avx2, . - _gllm_gemv_q8k_block_4row_avx2",
);

#[cfg(target_arch = "x86_64")]
extern "C" {
    /// Process one Q8K block for 4 rows: 256 i8 × f32 dot products per row.
    /// Writes 4 partial scalar results (before d scaling) to out[0..4].
    fn _gllm_gemv_q8k_block_4row_avx2(
        qs0: *const i8,
        qs1: *const i8,
        qs2: *const i8,
        qs3: *const i8,
        input: *const f32,
        out: *mut f32,
    );
}

/// Fused 4-row Q8K GEMV — AVX2 hand-written assembly path.
///
/// Uses `global_asm!` microkernel for precise register control:
/// - 8 YMM accumulators (2 per row), no register spills
/// - Input loads shared across all 4 rows (4x amortization)
/// - i8→f32 conversion interleaved with FMA in the hot loop
///
/// The outer block loop (scale loading, prefetch, 2x unrolling) is in Rust;
/// the hot inner path (256-element dot product) is in hand-written ASM.
///
/// # Safety
/// - weight_blocks: m * blocks_per_row * 292 bytes of contiguous Q8K data
/// - input: k f32 values
/// - output: m f32 values (written, not accumulated)
/// - k must be a multiple of 256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q8k_fused_avx2_asm(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K;
    let row_bytes = bpr * Q8K_BLOCK_BYTES;
    let m4 = m & !3;

    // Scratch buffer for 4 partial dot results from the ASM kernel.
    let mut partial = [0.0f32; 4];

    // ---- 4-row main loop ----
    let mut row = 0usize;
    while row < m4 {
        let base0 = weight_blocks.add(row * row_bytes);
        let base1 = weight_blocks.add((row + 1) * row_bytes);
        let base2 = weight_blocks.add((row + 2) * row_bytes);
        let base3 = weight_blocks.add((row + 3) * row_bytes);

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let bpr2 = bpr & !1;
        let mut b = 0usize;

        // ---- 2x block unrolled loop ----
        while b < bpr2 {
            // Block b
            let boff = b * Q8K_BLOCK_BYTES;
            let d0_b0 = *(base0.add(boff) as *const f32);
            let d1_b0 = *(base1.add(boff) as *const f32);
            let d2_b0 = *(base2.add(boff) as *const f32);
            let d3_b0 = *(base3.add(boff) as *const f32);

            // Prefetch block b+1
            let noff = (b + 1) * Q8K_BLOCK_BYTES;
            _mm_prefetch(base0.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base1.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base2.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base3.add(noff) as *const i8, _MM_HINT_T0);

            _gllm_gemv_q8k_block_4row_avx2(
                base0.add(boff + Q8K_QS_OFFSET) as *const i8,
                base1.add(boff + Q8K_QS_OFFSET) as *const i8,
                base2.add(boff + Q8K_QS_OFFSET) as *const i8,
                base3.add(boff + Q8K_QS_OFFSET) as *const i8,
                input.add(b * QK_K),
                partial.as_mut_ptr(),
            );
            sum0 += d0_b0 * partial[0];
            sum1 += d1_b0 * partial[1];
            sum2 += d2_b0 * partial[2];
            sum3 += d3_b0 * partial[3];

            // Block b+1
            let boff1 = noff;
            let d0_b1 = *(base0.add(boff1) as *const f32);
            let d1_b1 = *(base1.add(boff1) as *const f32);
            let d2_b1 = *(base2.add(boff1) as *const f32);
            let d3_b1 = *(base3.add(boff1) as *const f32);

            // Prefetch block b+2
            if b + 2 < bpr {
                let noff2 = (b + 2) * Q8K_BLOCK_BYTES;
                _mm_prefetch(base0.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base1.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base2.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base3.add(noff2) as *const i8, _MM_HINT_T0);
            }

            _gllm_gemv_q8k_block_4row_avx2(
                base0.add(boff1 + Q8K_QS_OFFSET) as *const i8,
                base1.add(boff1 + Q8K_QS_OFFSET) as *const i8,
                base2.add(boff1 + Q8K_QS_OFFSET) as *const i8,
                base3.add(boff1 + Q8K_QS_OFFSET) as *const i8,
                input.add((b + 1) * QK_K),
                partial.as_mut_ptr(),
            );
            sum0 += d0_b1 * partial[0];
            sum1 += d1_b1 * partial[1];
            sum2 += d2_b1 * partial[2];
            sum3 += d3_b1 * partial[3];

            b += 2;
        }

        // ---- Odd block remainder ----
        if bpr & 1 != 0 {
            let boff = b * Q8K_BLOCK_BYTES;
            let d0 = *(base0.add(boff) as *const f32);
            let d1 = *(base1.add(boff) as *const f32);
            let d2 = *(base2.add(boff) as *const f32);
            let d3 = *(base3.add(boff) as *const f32);

            _gllm_gemv_q8k_block_4row_avx2(
                base0.add(boff + Q8K_QS_OFFSET) as *const i8,
                base1.add(boff + Q8K_QS_OFFSET) as *const i8,
                base2.add(boff + Q8K_QS_OFFSET) as *const i8,
                base3.add(boff + Q8K_QS_OFFSET) as *const i8,
                input.add(b * QK_K),
                partial.as_mut_ptr(),
            );
            sum0 += d0 * partial[0];
            sum1 += d1 * partial[1];
            sum2 += d2 * partial[2];
            sum3 += d3 * partial[3];
        }

        *output.add(row)     = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // ---- Tail: 1 row at a time ----
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let boff = b * Q8K_BLOCK_BYTES;
            let d = *(base.add(boff) as *const f32);
            let raw = q8k_block_dot_1row(
                base.add(boff + Q8K_QS_OFFSET) as *const i8,
                input.add(b * QK_K),
            );
            sum += d * raw;
        }
        *output.add(row) = sum;
        row += 1;
    }
}

/// Fused 4-row Q8K GEMV — intrinsics fallback.
///
/// Flattened accumulators (16 named YMM variables) guarantee register allocation.
/// 2x block unrolling when bpr >= 2 overlaps prefetch with compute.
///
/// # Safety
/// - weight_blocks: m * blocks_per_row * 292 bytes of contiguous Q8K data
/// - input: k f32 values
/// - output: m f32 values (written, not accumulated)
/// - k must be a multiple of 256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q8k_fused_avx2_intrinsics(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K; // blocks per row
    let row_bytes = bpr * Q8K_BLOCK_BYTES;
    let m4 = m & !3; // round down to multiple of 4

    // ---- 4-row main loop ----
    let mut row = 0usize;
    while row < m4 {
        let base0 = weight_blocks.add(row * row_bytes);
        let base1 = weight_blocks.add((row + 1) * row_bytes);
        let base2 = weight_blocks.add((row + 2) * row_bytes);
        let base3 = weight_blocks.add((row + 3) * row_bytes);

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let bpr2 = bpr & !1; // round down to even for 2x unrolling
        let mut b = 0usize;

        // ---- 2x block unrolled loop ----
        while b < bpr2 {
            let zero = _mm256_setzero_ps();
            // 16 flattened accumulators: 4 per row
            let (mut a00, mut a01, mut a02, mut a03) = (zero, zero, zero, zero);
            let (mut a10, mut a11, mut a12, mut a13) = (zero, zero, zero, zero);
            let (mut a20, mut a21, mut a22, mut a23) = (zero, zero, zero, zero);
            let (mut a30, mut a31, mut a32, mut a33) = (zero, zero, zero, zero);

            // Block b
            let boff = b * Q8K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0_b0 = *(base0.add(boff) as *const f32);
            let d1_b0 = *(base1.add(boff) as *const f32);
            let d2_b0 = *(base2.add(boff) as *const f32);
            let d3_b0 = *(base3.add(boff) as *const f32);

            // Prefetch block b+1
            let noff = (b + 1) * Q8K_BLOCK_BYTES;
            _mm_prefetch(base0.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base1.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base2.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base3.add(noff) as *const i8, _MM_HINT_T0);

            let qs0 = base0.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs1 = base1.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs2 = base2.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs3 = base3.add(boff + Q8K_QS_OFFSET) as *const i8;

            q8k_block_4row_avx2!(inp, qs0, qs1, qs2, qs3,
                a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33);

            // Reduce block b and apply d
            sum0 += d0_b0 * reduce4_avx2!(a00, a01, a02, a03);
            sum1 += d1_b0 * reduce4_avx2!(a10, a11, a12, a13);
            sum2 += d2_b0 * reduce4_avx2!(a20, a21, a22, a23);
            sum3 += d3_b0 * reduce4_avx2!(a30, a31, a32, a33);

            // Block b+1 — reset accumulators
            let (mut a00, mut a01, mut a02, mut a03) = (zero, zero, zero, zero);
            let (mut a10, mut a11, mut a12, mut a13) = (zero, zero, zero, zero);
            let (mut a20, mut a21, mut a22, mut a23) = (zero, zero, zero, zero);
            let (mut a30, mut a31, mut a32, mut a33) = (zero, zero, zero, zero);

            let boff1 = noff;
            let inp1 = input.add((b + 1) * QK_K);
            let d0_b1 = *(base0.add(boff1) as *const f32);
            let d1_b1 = *(base1.add(boff1) as *const f32);
            let d2_b1 = *(base2.add(boff1) as *const f32);
            let d3_b1 = *(base3.add(boff1) as *const f32);

            // Prefetch block b+2
            if b + 2 < bpr {
                let noff2 = (b + 2) * Q8K_BLOCK_BYTES;
                _mm_prefetch(base0.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base1.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base2.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base3.add(noff2) as *const i8, _MM_HINT_T0);
            }

            let qs0 = base0.add(boff1 + Q8K_QS_OFFSET) as *const i8;
            let qs1 = base1.add(boff1 + Q8K_QS_OFFSET) as *const i8;
            let qs2 = base2.add(boff1 + Q8K_QS_OFFSET) as *const i8;
            let qs3 = base3.add(boff1 + Q8K_QS_OFFSET) as *const i8;

            q8k_block_4row_avx2!(inp1, qs0, qs1, qs2, qs3,
                a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33);

            sum0 += d0_b1 * reduce4_avx2!(a00, a01, a02, a03);
            sum1 += d1_b1 * reduce4_avx2!(a10, a11, a12, a13);
            sum2 += d2_b1 * reduce4_avx2!(a20, a21, a22, a23);
            sum3 += d3_b1 * reduce4_avx2!(a30, a31, a32, a33);

            b += 2;
        }

        // ---- Odd block remainder ----
        if bpr & 1 != 0 {
            let zero = _mm256_setzero_ps();
            let (mut a00, mut a01, mut a02, mut a03) = (zero, zero, zero, zero);
            let (mut a10, mut a11, mut a12, mut a13) = (zero, zero, zero, zero);
            let (mut a20, mut a21, mut a22, mut a23) = (zero, zero, zero, zero);
            let (mut a30, mut a31, mut a32, mut a33) = (zero, zero, zero, zero);

            let boff = b * Q8K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0 = *(base0.add(boff) as *const f32);
            let d1 = *(base1.add(boff) as *const f32);
            let d2 = *(base2.add(boff) as *const f32);
            let d3 = *(base3.add(boff) as *const f32);

            let qs0 = base0.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs1 = base1.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs2 = base2.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs3 = base3.add(boff + Q8K_QS_OFFSET) as *const i8;

            q8k_block_4row_avx2!(inp, qs0, qs1, qs2, qs3,
                a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33);

            sum0 += d0 * reduce4_avx2!(a00, a01, a02, a03);
            sum1 += d1 * reduce4_avx2!(a10, a11, a12, a13);
            sum2 += d2 * reduce4_avx2!(a20, a21, a22, a23);
            sum3 += d3 * reduce4_avx2!(a30, a31, a32, a33);
        }

        *output.add(row) = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // ---- Tail: 1 row at a time ----
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let boff = b * Q8K_BLOCK_BYTES;
            let d = *(base.add(boff) as *const f32);
            let raw = q8k_block_dot_1row(
                base.add(boff + Q8K_QS_OFFSET) as *const i8,
                input.add(b * QK_K),
            );
            sum += d * raw;
        }
        *output.add(row) = sum;
        row += 1;
    }
}

// ============================================================================
// Q4_K Fused GEMV — AVX2
// ============================================================================

const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_D_OFFSET: usize = 0;     // f16 d at byte 0
const Q4K_QS_OFFSET: usize = 16;   // qs starts after d(2) + dmin(2) + scales(12)

/// Unpack 16 bytes of Q4K nibbles and FMA against 4 input vectors.
/// Accumulates into 4 named accumulators for one row.
#[cfg(target_arch = "x86_64")]
macro_rules! q4k_row_fma_avx2 {
    ($qs:expr, $byte_off:expr, $o0:expr, $o1:expr, $o2:expr, $o3:expr,
     $mask_0f:expr, $na0:expr, $na1:expr, $na2:expr, $na3:expr) => {{
        let v = _mm_loadu_si128($qs.add($byte_off) as *const _);
        let lo = _mm_and_si128(v, $mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), $mask_0f);

        let fl0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
        let fl1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
        let fh0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
        let fh1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));

        let t0 = _mm256_unpacklo_ps(fl0, fh0);
        let t1 = _mm256_unpackhi_ps(fl0, fh0);
        let n0 = _mm256_permute2f128_ps(t0, t1, 0x20);
        let n1 = _mm256_permute2f128_ps(t0, t1, 0x31);
        let t2 = _mm256_unpacklo_ps(fl1, fh1);
        let t3 = _mm256_unpackhi_ps(fl1, fh1);
        let n2 = _mm256_permute2f128_ps(t2, t3, 0x20);
        let n3 = _mm256_permute2f128_ps(t2, t3, 0x31);

        $na0 = _mm256_fmadd_ps(n0, $o0, $na0);
        $na1 = _mm256_fmadd_ps(n1, $o1, $na1);
        $na2 = _mm256_fmadd_ps(n2, $o2, $na2);
        $na3 = _mm256_fmadd_ps(n3, $o3, $na3);
    }};
}

/// Process one Q4K block for 4 rows with flattened accumulators + shared input-sum.
#[cfg(target_arch = "x86_64")]
macro_rules! q4k_block_4row_avx2 {
    ($inp:expr, $qs0:expr, $qs1:expr, $qs2:expr, $qs3:expr, $mask_0f:expr,
     $na00:expr, $na01:expr, $na02:expr, $na03:expr,
     $na10:expr, $na11:expr, $na12:expr, $na13:expr,
     $na20:expr, $na21:expr, $na22:expr, $na23:expr,
     $na30:expr, $na31:expr, $na32:expr, $na33:expr,
     $oa0:expr, $oa1:expr) => {{
        let mut i = 0usize;
        while i < 8 {
            let byte_off = i * 16;
            let val_off = i * 32;

            let o0 = _mm256_loadu_ps($inp.add(val_off));
            let o1 = _mm256_loadu_ps($inp.add(val_off + 8));
            let o2 = _mm256_loadu_ps($inp.add(val_off + 16));
            let o3 = _mm256_loadu_ps($inp.add(val_off + 24));

            $oa0 = _mm256_add_ps($oa0, _mm256_add_ps(o0, o1));
            $oa1 = _mm256_add_ps($oa1, _mm256_add_ps(o2, o3));

            q4k_row_fma_avx2!($qs0, byte_off, o0, o1, o2, o3, $mask_0f,
                $na00, $na01, $na02, $na03);
            q4k_row_fma_avx2!($qs1, byte_off, o0, o1, o2, o3, $mask_0f,
                $na10, $na11, $na12, $na13);
            q4k_row_fma_avx2!($qs2, byte_off, o0, o1, o2, o3, $mask_0f,
                $na20, $na21, $na22, $na23);
            q4k_row_fma_avx2!($qs3, byte_off, o0, o1, o2, o3, $mask_0f,
                $na30, $na31, $na32, $na33);

            i += 1;
        }
    }};
}

/// Fused 4-row Q4K GEMV using deferred-scale trick (intrinsics fallback).
///
/// For each block: dot = d * sum(nib_i * input_i) - d*8 * sum(input_i)
/// The sum(input_i) is computed once and shared across all 4 rows.
///
/// Flattened accumulators (16 named YMM variables) guarantee register allocation.
/// 2x block unrolling when bpr >= 2.
/// Tail uses 4 independent accumulators to break serial FMA dependency.
///
/// # Safety
/// Same requirements as gemv_q8k_fused_avx2_asm but for Q4K blocks (144 bytes each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4k_fused_avx2_intrinsics(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K;
    let row_bytes = bpr * Q4K_BLOCK_BYTES;
    let m4 = m & !3;
    let mask_0f = _mm_set1_epi8(0x0F);

    // ---- 4-row main loop ----
    let mut row = 0usize;
    while row < m4 {
        let base0 = weight_blocks.add(row * row_bytes);
        let base1 = weight_blocks.add((row + 1) * row_bytes);
        let base2 = weight_blocks.add((row + 2) * row_bytes);
        let base3 = weight_blocks.add((row + 3) * row_bytes);

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let bpr2 = bpr & !1;
        let mut b = 0usize;

        // ---- 2x block unrolled loop ----
        while b < bpr2 {
            // --- Block b ---
            let zero = _mm256_setzero_ps();
            let (mut na00, mut na01, mut na02, mut na03) = (zero, zero, zero, zero);
            let (mut na10, mut na11, mut na12, mut na13) = (zero, zero, zero, zero);
            let (mut na20, mut na21, mut na22, mut na23) = (zero, zero, zero, zero);
            let (mut na30, mut na31, mut na32, mut na33) = (zero, zero, zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let noff = (b + 1) * Q4K_BLOCK_BYTES;
            _mm_prefetch(base0.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base1.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base2.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base3.add(noff) as *const i8, _MM_HINT_T0);

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            q4k_block_4row_avx2!(inp, qs0, qs1, qs2, qs3, mask_0f,
                na00, na01, na02, na03, na10, na11, na12, na13,
                na20, na21, na22, na23, na30, na31, na32, na33,
                oa0, oa1);

            let nib0 = reduce4_avx2!(na00, na01, na02, na03);
            let nib1 = reduce4_avx2!(na10, na11, na12, na13);
            let nib2 = reduce4_avx2!(na20, na21, na22, na23);
            let nib3 = reduce4_avx2!(na30, na31, na32, na33);
            let oth = hsum_avx2(_mm256_add_ps(oa0, oa1));

            sum0 += d0 * nib0 - d0 * 8.0 * oth;
            sum1 += d1 * nib1 - d1 * 8.0 * oth;
            sum2 += d2 * nib2 - d2 * 8.0 * oth;
            sum3 += d3 * nib3 - d3 * 8.0 * oth;

            // --- Block b+1 ---
            let (mut na00, mut na01, mut na02, mut na03) = (zero, zero, zero, zero);
            let (mut na10, mut na11, mut na12, mut na13) = (zero, zero, zero, zero);
            let (mut na20, mut na21, mut na22, mut na23) = (zero, zero, zero, zero);
            let (mut na30, mut na31, mut na32, mut na33) = (zero, zero, zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let boff1 = noff;
            let inp1 = input.add((b + 1) * QK_K);
            let d0_b1 = half::f16::from_bits(*(base0.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1_b1 = half::f16::from_bits(*(base1.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2_b1 = half::f16::from_bits(*(base2.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3_b1 = half::f16::from_bits(*(base3.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();

            if b + 2 < bpr {
                let noff2 = (b + 2) * Q4K_BLOCK_BYTES;
                _mm_prefetch(base0.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base1.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base2.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base3.add(noff2) as *const i8, _MM_HINT_T0);
            }

            let qs0 = base0.add(boff1 + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff1 + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff1 + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff1 + Q4K_QS_OFFSET);

            q4k_block_4row_avx2!(inp1, qs0, qs1, qs2, qs3, mask_0f,
                na00, na01, na02, na03, na10, na11, na12, na13,
                na20, na21, na22, na23, na30, na31, na32, na33,
                oa0, oa1);

            let nib0 = reduce4_avx2!(na00, na01, na02, na03);
            let nib1 = reduce4_avx2!(na10, na11, na12, na13);
            let nib2 = reduce4_avx2!(na20, na21, na22, na23);
            let nib3 = reduce4_avx2!(na30, na31, na32, na33);
            let oth = hsum_avx2(_mm256_add_ps(oa0, oa1));

            sum0 += d0_b1 * nib0 - d0_b1 * 8.0 * oth;
            sum1 += d1_b1 * nib1 - d1_b1 * 8.0 * oth;
            sum2 += d2_b1 * nib2 - d2_b1 * 8.0 * oth;
            sum3 += d3_b1 * nib3 - d3_b1 * 8.0 * oth;

            b += 2;
        }

        // ---- Odd block remainder ----
        if bpr & 1 != 0 {
            let zero = _mm256_setzero_ps();
            let (mut na00, mut na01, mut na02, mut na03) = (zero, zero, zero, zero);
            let (mut na10, mut na11, mut na12, mut na13) = (zero, zero, zero, zero);
            let (mut na20, mut na21, mut na22, mut na23) = (zero, zero, zero, zero);
            let (mut na30, mut na31, mut na32, mut na33) = (zero, zero, zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            q4k_block_4row_avx2!(inp, qs0, qs1, qs2, qs3, mask_0f,
                na00, na01, na02, na03, na10, na11, na12, na13,
                na20, na21, na22, na23, na30, na31, na32, na33,
                oa0, oa1);

            let nib0 = reduce4_avx2!(na00, na01, na02, na03);
            let nib1 = reduce4_avx2!(na10, na11, na12, na13);
            let nib2 = reduce4_avx2!(na20, na21, na22, na23);
            let nib3 = reduce4_avx2!(na30, na31, na32, na33);
            let oth = hsum_avx2(_mm256_add_ps(oa0, oa1));

            sum0 += d0 * nib0 - d0 * 8.0 * oth;
            sum1 += d1 * nib1 - d1 * 8.0 * oth;
            sum2 += d2 * nib2 - d2 * 8.0 * oth;
            sum3 += d3 * nib3 - d3 * 8.0 * oth;
        }

        *output.add(row) = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // ---- Tail: 1 row at a time (4 independent accumulators) ----
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let boff = b * Q4K_BLOCK_BYTES;
            let d = half::f16::from_bits(*(base.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let qs = base.add(boff + Q4K_QS_OFFSET);
            let inp = input.add(b * QK_K);

            let zero = _mm256_setzero_ps();
            let (mut na0, mut na1, mut na2, mut na3) = (zero, zero, zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let mut i = 0usize;
            while i < 8 {
                let byte_off = i * 16;
                let val_off = i * 32;

                let o0 = _mm256_loadu_ps(inp.add(val_off));
                let o1 = _mm256_loadu_ps(inp.add(val_off + 8));
                let o2 = _mm256_loadu_ps(inp.add(val_off + 16));
                let o3 = _mm256_loadu_ps(inp.add(val_off + 24));

                oa0 = _mm256_add_ps(oa0, _mm256_add_ps(o0, o1));
                oa1 = _mm256_add_ps(oa1, _mm256_add_ps(o2, o3));

                q4k_row_fma_avx2!(qs, byte_off, o0, o1, o2, o3, mask_0f,
                    na0, na1, na2, na3);

                i += 1;
            }

            let nib_sum = reduce4_avx2!(na0, na1, na2, na3);
            let oth_sum = hsum_avx2(_mm256_add_ps(oa0, oa1));
            sum += d * nib_sum - d * 8.0 * oth_sum;
        }
        *output.add(row) = sum;
        row += 1;
    }
}

// ============================================================================
// Q4_K Fused GEMV — AVX2 hand-written assembly microkernel
// ============================================================================
//
// _gllm_q4k_block_1row_avx2: processes one Q4K block (128 bytes of qs,
// 256 f32 inputs) for a single row.
//
// Calling convention (System V AMD64 ABI, extern "C"):
//   rdi = *const u8  : qs pointer (128 bytes of packed nibbles)
//   rsi = *const f32 : input pointer (256 f32 values)
//   rdx = *mut f32   : nib_sum output (sum of nib_i * input_i)
//   rcx = *mut f32   : oth_sum output (sum of all input_i)
//
// Register allocation (16 YMM):
//   ymm0-ymm3  : nib accumulators (4 x 8 f32, reduced at end)
//   ymm4-ymm5  : oth accumulators (input sum, 2 x 8 f32)
//   ymm6-ymm9  : input loads o0-o3 (4 x 8 f32 = 32 values per iteration)
//   xmm10      : nibble mask 0x0F (16 bytes)
//   xmm11      : raw 16-byte qs load / scratch
//   xmm12      : lo nibbles / scratch
//   xmm13      : hi nibbles / scratch
//   ymm14-ymm15: f32 conversion temporaries / interleave
//
// Inner loop: 8 iterations, each processes 16 bytes qs (32 nibbles = 32 values)
// against 32 f32 inputs (4 x ymm loads of 8 f32 each).
//
// Nibble interleave matches intrinsics q4k_row_fma_avx2! exactly:
//   lo[0..7]  -> vpmovzxbd ymm -> vcvtdq2ps -> fl0 (8 f32)
//   lo[8..15] -> vpsrldq 8 + vpmovzxbd -> fl1 (8 f32)
//   hi[0..7]  -> vpmovzxbd ymm -> vcvtdq2ps -> fh0 (8 f32)
//   hi[8..15] -> vpsrldq 8 + vpmovzxbd -> fh1 (8 f32)
//   n0 = vperm2f128(vunpcklps(fl0,fh0), vunpckhps(fl0,fh0), 0x20) -> values 0..7
//   n1 = vperm2f128(vunpcklps(fl0,fh0), vunpckhps(fl0,fh0), 0x31) -> values 8..15
//   n2 = vperm2f128(vunpcklps(fl1,fh1), vunpckhps(fl1,fh1), 0x20) -> values 16..23
//   n3 = vperm2f128(vunpcklps(fl1,fh1), vunpckhps(fl1,fh1), 0x31) -> values 24..31

#[cfg(target_arch = "x86_64")]
global_asm!(
    ".text",
    ".align 64",
    ".global _gllm_q4k_block_1row_avx2",
    ".type _gllm_q4k_block_1row_avx2, @function",
    "_gllm_q4k_block_1row_avx2:",

    // Zero nib accumulators ymm0-ymm3
    "vxorps ymm0, ymm0, ymm0",
    "vxorps ymm1, ymm1, ymm1",
    "vxorps ymm2, ymm2, ymm2",
    "vxorps ymm3, ymm3, ymm3",
    // Zero oth accumulators ymm4-ymm5
    "vxorps ymm4, ymm4, ymm4",
    "vxorps ymm5, ymm5, ymm5",

    // Build nibble mask 0x0F in xmm10 via vpbroadcastd
    "mov eax, 0x0F0F0F0F",
    "vmovd xmm10, eax",
    "vpbroadcastd xmm10, xmm10",

    // r8d = loop counter (8 iterations)
    // r9d = byte offset into qs (0, 16, 32, ..., 112)
    // r10d = float element offset into input (0, 32, 64, ..., 224)
    "mov r8d, 8",
    "xor r9d, r9d",
    "xor r10d, r10d",

    ".align 32",
    ".Lq4k_1row_loop:",

    // Load 4 x 8 f32 input values (32 f32 = 128 bytes)
    "vmovups ymm6, [rsi + r10*4]",
    "vmovups ymm7, [rsi + r10*4 + 32]",
    "vmovups ymm8, [rsi + r10*4 + 64]",
    "vmovups ymm9, [rsi + r10*4 + 96]",

    // Accumulate input sum: oth += (o0+o1) + (o2+o3)
    "vaddps ymm14, ymm6, ymm7",
    "vaddps ymm15, ymm8, ymm9",
    "vaddps ymm4, ymm4, ymm14",
    "vaddps ymm5, ymm5, ymm15",

    // Load 16 bytes of packed nibbles
    "vmovdqu xmm11, [rdi + r9]",

    // lo = xmm11 & 0x0F
    "vpand xmm12, xmm11, xmm10",
    // hi = (xmm11 >> 4) & 0x0F
    "vpsrlw xmm13, xmm11, 4",
    "vpand xmm13, xmm13, xmm10",

    // fl0 = cvtepu8_epi32(lo[0..7]) -> f32  (vpmovzxbd uses low 8 bytes of xmm12)
    "vpmovzxbd ymm14, xmm12",
    "vcvtdq2ps ymm14, ymm14",
    // fl1 = cvtepu8_epi32(lo[8..15]) -> f32
    "vpsrldq xmm12, xmm12, 8",
    "vpmovzxbd ymm15, xmm12",
    "vcvtdq2ps ymm15, ymm15",

    // Save hi[8..15] into xmm11 before consuming xmm13
    "vpsrldq xmm11, xmm13, 8",
    // fh0 = cvtepu8_epi32(hi[0..7]) -> f32  (into ymm12)
    "vpmovzxbd ymm12, xmm13",
    "vcvtdq2ps ymm12, ymm12",
    // fh1 = cvtepu8_epi32(hi[8..15]) -> f32  (into ymm13)
    "vpmovzxbd ymm13, xmm11",
    "vcvtdq2ps ymm13, ymm13",

    // Interleave fl0(ymm14) / fh0(ymm12) -> n0, n1
    // t0 = vunpcklps(fl0, fh0): [fl0[0],fh0[0],fl0[1],fh0[1] | fl0[4],fh0[4],fl0[5],fh0[5]]
    // t1 = vunpckhps(fl0, fh0): [fl0[2],fh0[2],fl0[3],fh0[3] | fl0[6],fh0[6],fl0[7],fh0[7]]
    // n0 = vperm2f128(t0,t1,0x20): lo128(t0)|lo128(t1) = sequential values 0..7
    // n1 = vperm2f128(t0,t1,0x31): hi128(t0)|hi128(t1) = sequential values 8..15
    "vunpcklps ymm11, ymm14, ymm12",
    "vunpckhps ymm12, ymm14, ymm12",
    "vperm2f128 ymm14, ymm11, ymm12, 0x20",
    "vperm2f128 ymm11, ymm11, ymm12, 0x31",

    // FMA: nib_acc[0] += n0 * o0,  nib_acc[1] += n1 * o1
    "vfmadd231ps ymm0, ymm14, ymm6",
    "vfmadd231ps ymm1, ymm11, ymm7",

    // Interleave fl1(ymm15) / fh1(ymm13) -> n2, n3
    "vunpcklps ymm11, ymm15, ymm13",
    "vunpckhps ymm12, ymm15, ymm13",
    "vperm2f128 ymm14, ymm11, ymm12, 0x20",
    "vperm2f128 ymm11, ymm11, ymm12, 0x31",

    // FMA: nib_acc[2] += n2 * o2,  nib_acc[3] += n3 * o3
    "vfmadd231ps ymm2, ymm14, ymm8",
    "vfmadd231ps ymm3, ymm11, ymm9",

    // Advance: r9 += 16 (bytes), r10 += 32 (f32 elements)
    "add r9d, 16",
    "add r10d, 32",
    "dec r8d",
    "jnz .Lq4k_1row_loop",

    // Reduce nib accumulators ymm0-ymm3 -> scalar
    "vaddps ymm0, ymm0, ymm1",
    "vaddps ymm2, ymm2, ymm3",
    "vaddps ymm0, ymm0, ymm2",
    "vhaddps ymm0, ymm0, ymm0",
    "vhaddps ymm0, ymm0, ymm0",
    "vextractf128 xmm1, ymm0, 1",
    "vaddss xmm0, xmm0, xmm1",
    "vmovss [rdx], xmm0",

    // Reduce oth accumulators ymm4-ymm5 -> scalar
    "vaddps ymm4, ymm4, ymm5",
    "vhaddps ymm4, ymm4, ymm4",
    "vhaddps ymm4, ymm4, ymm4",
    "vextractf128 xmm5, ymm4, 1",
    "vaddss xmm4, xmm4, xmm5",
    "vmovss [rcx], xmm4",

    "vzeroupper",
    "ret",

    ".size _gllm_q4k_block_1row_avx2, . - _gllm_q4k_block_1row_avx2",
);

#[cfg(target_arch = "x86_64")]
extern "C" {
    /// Hand-written AVX2 microkernel: processes one Q4K block for one row.
    ///
    /// Computes:
    ///   *nib_sum = sum over 256 elements of (nibble_i * input_i)
    ///   *oth_sum = sum over 256 elements of input_i
    ///
    /// # Safety
    /// - qs must point to 128 bytes of packed Q4K nibble data
    /// - input must point to 256 f32 values
    /// - nib_sum and oth_sum must be valid writable f32 pointers
    fn _gllm_q4k_block_1row_avx2(
        qs: *const u8,
        input: *const f32,
        nib_sum: *mut f32,
        oth_sum: *mut f32,
    );
}

/// Fused 4-row Q4K GEMV — AVX2 hand-written assembly path.
///
/// Uses `_gllm_q4k_block_1row_avx2` for the inner nibble-unpack + FMA loop.
/// The assembly kernel gives precise register allocation and software pipelining
/// that the compiler cannot guarantee from intrinsics.
///
/// For each block: dot = d * nib_sum - d*8 * oth_sum
/// oth_sum is identical for all 4 rows (same input vector), computed from row0.
///
/// # Safety
/// Same requirements as gemv_q4k_fused_avx2_intrinsics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4k_fused_avx2_asm(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K;
    let row_bytes = bpr * Q4K_BLOCK_BYTES;
    let m4 = m & !3;

    let mut row = 0usize;
    while row < m4 {
        let base0 = weight_blocks.add(row * row_bytes);
        let base1 = weight_blocks.add((row + 1) * row_bytes);
        let base2 = weight_blocks.add((row + 2) * row_bytes);
        let base3 = weight_blocks.add((row + 3) * row_bytes);

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let bpr2 = bpr & !1;
        let mut b = 0usize;

        // ---- 2x block unrolled loop ----
        while b < bpr2 {
            // --- Block b ---
            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);

            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let noff = (b + 1) * Q4K_BLOCK_BYTES;
            _mm_prefetch(base0.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base1.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base2.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base3.add(noff) as *const i8, _MM_HINT_T0);

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            // oth_sum is identical for all rows (same input); use oth from row0
            let mut nib0 = 0.0f32;
            let mut oth = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs0, inp, &mut nib0, &mut oth);

            let mut nib1 = 0.0f32;
            let mut oth1 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs1, inp, &mut nib1, &mut oth1);

            let mut nib2 = 0.0f32;
            let mut oth2 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs2, inp, &mut nib2, &mut oth2);

            let mut nib3 = 0.0f32;
            let mut oth3 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs3, inp, &mut nib3, &mut oth3);

            // All rows share the same input, so oth is identical; use oth from row0
            sum0 += d0 * nib0 - d0 * 8.0 * oth;
            sum1 += d1 * nib1 - d1 * 8.0 * oth;
            sum2 += d2 * nib2 - d2 * 8.0 * oth;
            sum3 += d3 * nib3 - d3 * 8.0 * oth;

            // --- Block b+1 ---
            let boff1 = noff;
            let inp1 = input.add((b + 1) * QK_K);

            let d0_b1 = half::f16::from_bits(*(base0.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1_b1 = half::f16::from_bits(*(base1.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2_b1 = half::f16::from_bits(*(base2.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3_b1 = half::f16::from_bits(*(base3.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();

            if b + 2 < bpr {
                let noff2 = (b + 2) * Q4K_BLOCK_BYTES;
                _mm_prefetch(base0.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base1.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base2.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base3.add(noff2) as *const i8, _MM_HINT_T0);
            }

            let qs0 = base0.add(boff1 + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff1 + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff1 + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff1 + Q4K_QS_OFFSET);

            let mut nib0_b1 = 0.0f32;
            let mut oth_b1 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs0, inp1, &mut nib0_b1, &mut oth_b1);

            let mut nib1_b1 = 0.0f32;
            let mut oth1_b1 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs1, inp1, &mut nib1_b1, &mut oth1_b1);

            let mut nib2_b1 = 0.0f32;
            let mut oth2_b1 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs2, inp1, &mut nib2_b1, &mut oth2_b1);

            let mut nib3_b1 = 0.0f32;
            let mut oth3_b1 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs3, inp1, &mut nib3_b1, &mut oth3_b1);

            sum0 += d0_b1 * nib0_b1 - d0_b1 * 8.0 * oth_b1;
            sum1 += d1_b1 * nib1_b1 - d1_b1 * 8.0 * oth_b1;
            sum2 += d2_b1 * nib2_b1 - d2_b1 * 8.0 * oth_b1;
            sum3 += d3_b1 * nib3_b1 - d3_b1 * 8.0 * oth_b1;

            b += 2;
        }

        // ---- Odd block remainder ----
        if bpr & 1 != 0 {
            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);

            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            let mut nib0 = 0.0f32;
            let mut oth = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs0, inp, &mut nib0, &mut oth);

            let mut nib1 = 0.0f32;
            let mut oth1 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs1, inp, &mut nib1, &mut oth1);

            let mut nib2 = 0.0f32;
            let mut oth2 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs2, inp, &mut nib2, &mut oth2);

            let mut nib3 = 0.0f32;
            let mut oth3 = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs3, inp, &mut nib3, &mut oth3);

            sum0 += d0 * nib0 - d0 * 8.0 * oth;
            sum1 += d1 * nib1 - d1 * 8.0 * oth;
            sum2 += d2 * nib2 - d2 * 8.0 * oth;
            sum3 += d3 * nib3 - d3 * 8.0 * oth;
        }

        *output.add(row) = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // ---- Tail: 1 row at a time ----
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let boff = b * Q4K_BLOCK_BYTES;
            let d = half::f16::from_bits(*(base.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let qs = base.add(boff + Q4K_QS_OFFSET);
            let inp = input.add(b * QK_K);

            let mut nib_sum = 0.0f32;
            let mut oth_sum = 0.0f32;
            _gllm_q4k_block_1row_avx2(qs, inp, &mut nib_sum, &mut oth_sum);
            sum += d * nib_sum - d * 8.0 * oth_sum;
        }
        *output.add(row) = sum;
        row += 1;
    }
}

/// Public entry point for Q4K AVX2 GEMV: dispatches to ASM microkernel.
///
/// # Safety
/// Same requirements as gemv_q4k_fused_avx2_intrinsics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4k_fused_avx2(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    gemv_q4k_fused_avx2_asm(weight_blocks, input, output, m, k)
}

// ============================================================================
// Q8_K Fused GEMV — AVX-512
// ============================================================================

/// Reduce 4 named ZMM accumulators to scalar via hsum_avx512.
#[cfg(target_arch = "x86_64")]
macro_rules! reduce4_avx512 {
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
        hsum_avx512(_mm512_add_ps(_mm512_add_ps($a0, $a1), _mm512_add_ps($a2, $a3)))
    };
}

/// Process one Q8K block for 4 rows with flattened ZMM accumulators.
/// Each ZMM = 16 f32, so 4 accumulators × 16 = 64 elements per inner iteration.
/// 256 elements / 64 = 4 iterations per block.
#[cfg(target_arch = "x86_64")]
macro_rules! q8k_block_4row_avx512 {
    ($inp:expr, $qs0:expr, $qs1:expr, $qs2:expr, $qs3:expr,
     $a00:expr, $a01:expr, $a02:expr, $a03:expr,
     $a10:expr, $a11:expr, $a12:expr, $a13:expr,
     $a20:expr, $a21:expr, $a22:expr, $a23:expr,
     $a30:expr, $a31:expr, $a32:expr, $a33:expr) => {{
        let mut i = 0usize;
        while i < 256 {
            // Shared input load: 4 × 16 f32 = 64 elements
            let o0 = _mm512_loadu_ps($inp.add(i));
            let o1 = _mm512_loadu_ps($inp.add(i + 16));
            let o2 = _mm512_loadu_ps($inp.add(i + 32));
            let o3 = _mm512_loadu_ps($inp.add(i + 48));

            // Row 0: load 16 i8 → 16 i32 → 16 f32, FMA
            $a00 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs0.add(i) as *const _))), o0, $a00);
            $a01 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs0.add(i+16) as *const _))), o1, $a01);
            $a02 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs0.add(i+32) as *const _))), o2, $a02);
            $a03 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs0.add(i+48) as *const _))), o3, $a03);
            // Row 1
            $a10 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs1.add(i) as *const _))), o0, $a10);
            $a11 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs1.add(i+16) as *const _))), o1, $a11);
            $a12 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs1.add(i+32) as *const _))), o2, $a12);
            $a13 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs1.add(i+48) as *const _))), o3, $a13);
            // Row 2
            $a20 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs2.add(i) as *const _))), o0, $a20);
            $a21 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs2.add(i+16) as *const _))), o1, $a21);
            $a22 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs2.add(i+32) as *const _))), o2, $a22);
            $a23 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs2.add(i+48) as *const _))), o3, $a23);
            // Row 3
            $a30 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs3.add(i) as *const _))), o0, $a30);
            $a31 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs3.add(i+16) as *const _))), o1, $a31);
            $a32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs3.add(i+32) as *const _))), o2, $a32);
            $a33 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128($qs3.add(i+48) as *const _))), o3, $a33);

            i += 64;
        }
    }};
}

/// Inner block dot for one row using AVX-512: 4 ZMM accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn q8k_block_dot_1row_avx512(qs: *const i8, input: *const f32) -> f32 {
    let mut a0 = _mm512_setzero_ps();
    let mut a1 = _mm512_setzero_ps();
    let mut a2 = _mm512_setzero_ps();
    let mut a3 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i < 256 {
        let o0 = _mm512_loadu_ps(input.add(i));
        let o1 = _mm512_loadu_ps(input.add(i + 16));
        let o2 = _mm512_loadu_ps(input.add(i + 32));
        let o3 = _mm512_loadu_ps(input.add(i + 48));
        let q0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(qs.add(i) as *const _)));
        let q1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(qs.add(i + 16) as *const _)));
        let q2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(qs.add(i + 32) as *const _)));
        let q3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(qs.add(i + 48) as *const _)));
        a0 = _mm512_fmadd_ps(q0, o0, a0);
        a1 = _mm512_fmadd_ps(q1, o1, a1);
        a2 = _mm512_fmadd_ps(q2, o2, a2);
        a3 = _mm512_fmadd_ps(q3, o3, a3);
        i += 64;
    }
    reduce4_avx512!(a0, a1, a2, a3)
}

/// Fused 4-row Q8K GEMV — AVX-512 variant.
///
/// Uses ZMM registers (16 f32 per vector). 4 accumulators per row = 64 elements
/// per inner iteration, so only 4 iterations per 256-element block.
/// 2x block unrolling with prefetch overlap.
///
/// # Safety
/// Same as gemv_q8k_fused_avx2_asm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn gemv_q8k_fused_avx512(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K;
    let row_bytes = bpr * Q8K_BLOCK_BYTES;
    let m4 = m & !3;

    let mut row = 0usize;
    while row < m4 {
        let base0 = weight_blocks.add(row * row_bytes);
        let base1 = weight_blocks.add((row + 1) * row_bytes);
        let base2 = weight_blocks.add((row + 2) * row_bytes);
        let base3 = weight_blocks.add((row + 3) * row_bytes);

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let bpr2 = bpr & !1;
        let mut b = 0usize;

        while b < bpr2 {
            let zero = _mm512_setzero_ps();
            let (mut a00, mut a01, mut a02, mut a03) = (zero, zero, zero, zero);
            let (mut a10, mut a11, mut a12, mut a13) = (zero, zero, zero, zero);
            let (mut a20, mut a21, mut a22, mut a23) = (zero, zero, zero, zero);
            let (mut a30, mut a31, mut a32, mut a33) = (zero, zero, zero, zero);

            // Block b
            let boff = b * Q8K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0_b0 = *(base0.add(boff) as *const f32);
            let d1_b0 = *(base1.add(boff) as *const f32);
            let d2_b0 = *(base2.add(boff) as *const f32);
            let d3_b0 = *(base3.add(boff) as *const f32);

            let noff = (b + 1) * Q8K_BLOCK_BYTES;
            _mm_prefetch(base0.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base1.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base2.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base3.add(noff) as *const i8, _MM_HINT_T0);

            let qs0 = base0.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs1 = base1.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs2 = base2.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs3 = base3.add(boff + Q8K_QS_OFFSET) as *const i8;

            q8k_block_4row_avx512!(inp, qs0, qs1, qs2, qs3,
                a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33);

            sum0 += d0_b0 * reduce4_avx512!(a00, a01, a02, a03);
            sum1 += d1_b0 * reduce4_avx512!(a10, a11, a12, a13);
            sum2 += d2_b0 * reduce4_avx512!(a20, a21, a22, a23);
            sum3 += d3_b0 * reduce4_avx512!(a30, a31, a32, a33);

            // Block b+1
            let (mut a00, mut a01, mut a02, mut a03) = (zero, zero, zero, zero);
            let (mut a10, mut a11, mut a12, mut a13) = (zero, zero, zero, zero);
            let (mut a20, mut a21, mut a22, mut a23) = (zero, zero, zero, zero);
            let (mut a30, mut a31, mut a32, mut a33) = (zero, zero, zero, zero);

            let boff1 = noff;
            let inp1 = input.add((b + 1) * QK_K);
            let d0_b1 = *(base0.add(boff1) as *const f32);
            let d1_b1 = *(base1.add(boff1) as *const f32);
            let d2_b1 = *(base2.add(boff1) as *const f32);
            let d3_b1 = *(base3.add(boff1) as *const f32);

            if b + 2 < bpr {
                let noff2 = (b + 2) * Q8K_BLOCK_BYTES;
                _mm_prefetch(base0.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base1.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base2.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base3.add(noff2) as *const i8, _MM_HINT_T0);
            }

            let qs0 = base0.add(boff1 + Q8K_QS_OFFSET) as *const i8;
            let qs1 = base1.add(boff1 + Q8K_QS_OFFSET) as *const i8;
            let qs2 = base2.add(boff1 + Q8K_QS_OFFSET) as *const i8;
            let qs3 = base3.add(boff1 + Q8K_QS_OFFSET) as *const i8;

            q8k_block_4row_avx512!(inp1, qs0, qs1, qs2, qs3,
                a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33);

            sum0 += d0_b1 * reduce4_avx512!(a00, a01, a02, a03);
            sum1 += d1_b1 * reduce4_avx512!(a10, a11, a12, a13);
            sum2 += d2_b1 * reduce4_avx512!(a20, a21, a22, a23);
            sum3 += d3_b1 * reduce4_avx512!(a30, a31, a32, a33);

            b += 2;
        }

        // Odd block remainder
        if bpr & 1 != 0 {
            let zero = _mm512_setzero_ps();
            let (mut a00, mut a01, mut a02, mut a03) = (zero, zero, zero, zero);
            let (mut a10, mut a11, mut a12, mut a13) = (zero, zero, zero, zero);
            let (mut a20, mut a21, mut a22, mut a23) = (zero, zero, zero, zero);
            let (mut a30, mut a31, mut a32, mut a33) = (zero, zero, zero, zero);

            let boff = b * Q8K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0 = *(base0.add(boff) as *const f32);
            let d1 = *(base1.add(boff) as *const f32);
            let d2 = *(base2.add(boff) as *const f32);
            let d3 = *(base3.add(boff) as *const f32);

            let qs0 = base0.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs1 = base1.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs2 = base2.add(boff + Q8K_QS_OFFSET) as *const i8;
            let qs3 = base3.add(boff + Q8K_QS_OFFSET) as *const i8;

            q8k_block_4row_avx512!(inp, qs0, qs1, qs2, qs3,
                a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33);

            sum0 += d0 * reduce4_avx512!(a00, a01, a02, a03);
            sum1 += d1 * reduce4_avx512!(a10, a11, a12, a13);
            sum2 += d2 * reduce4_avx512!(a20, a21, a22, a23);
            sum3 += d3 * reduce4_avx512!(a30, a31, a32, a33);
        }

        *output.add(row) = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // Tail: 1 row at a time
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let boff = b * Q8K_BLOCK_BYTES;
            let d = *(base.add(boff) as *const f32);
            let raw = q8k_block_dot_1row_avx512(
                base.add(boff + Q8K_QS_OFFSET) as *const i8,
                input.add(b * QK_K),
            );
            sum += d * raw;
        }
        *output.add(row) = sum;
        row += 1;
    }
}

// ============================================================================
// Q4_K Fused GEMV — AVX-512
// ============================================================================

/// Unpack 16 bytes of Q4K nibbles and FMA against 2 ZMM input vectors for one row.
/// AVX-512: 16 bytes = 32 nibbles = 32 values → 2 ZMM vectors of 16 f32 each.
/// 8 iterations per block (128 bytes = 256 values).
/// 2 accumulators per row (na0, na1).
///
/// Interleaving uses _mm512_permutex2var_ps to correctly merge 128-bit lanes
/// from unpacklo/unpackhi results into sequential order.
#[cfg(target_arch = "x86_64")]
macro_rules! q4k_row_fma_avx512 {
    ($qs:expr, $byte_off:expr, $o0:expr, $o1:expr,
     $mask_0f:expr, $idx_n0:expr, $idx_n1:expr, $na0:expr, $na1:expr) => {{
        let v = _mm_loadu_si128($qs.add($byte_off) as *const _);
        let lo = _mm_and_si128(v, $mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), $mask_0f);

        // Expand 16 low nibbles → 16 i32 → 16 f32 (ZMM)
        let fl = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo));
        // Expand 16 high nibbles → 16 i32 → 16 f32 (ZMM)
        let fh = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi));

        // unpacklo/hi interleave pairs within each 128-bit lane:
        // il = [l0,h0,l1,h1 | l4,h4,l5,h5 | l8,h8,l9,h9 | l12,h12,l13,h13]
        // ih = [l2,h2,l3,h3 | l6,h6,l7,h7 | l10,h10,l11,h11 | l14,h14,l15,h15]
        let il = _mm512_unpacklo_ps(fl, fh);
        let ih = _mm512_unpackhi_ps(fl, fh);

        // Cross-lane permute to get sequential order:
        // n0 = il.lane0, ih.lane0, il.lane1, ih.lane1 = [l0,h0,...,l7,h7]
        // n1 = il.lane2, ih.lane2, il.lane3, ih.lane3 = [l8,h8,...,l15,h15]
        let n0 = _mm512_permutex2var_ps(il, $idx_n0, ih);
        let n1 = _mm512_permutex2var_ps(il, $idx_n1, ih);

        $na0 = _mm512_fmadd_ps(n0, $o0, $na0);
        $na1 = _mm512_fmadd_ps(n1, $o1, $na1);
    }};
}

/// Reduce 2 named ZMM accumulators to scalar.
#[cfg(target_arch = "x86_64")]
macro_rules! reduce2_avx512 {
    ($a0:expr, $a1:expr) => {
        hsum_avx512(_mm512_add_ps($a0, $a1))
    };
}

/// Fused 4-row Q4K GEMV — AVX-512 variant.
///
/// Uses ZMM registers. Each iteration processes 16 bytes of qs = 32 nibbles = 32 values
/// into 2 ZMM vectors. 8 iterations per block (128 bytes = 256 values).
/// 2x block unrolling with prefetch overlap.
///
/// # Safety
/// Same as gemv_q4k_fused_avx2_intrinsics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn gemv_q4k_fused_avx512(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K;
    let row_bytes = bpr * Q4K_BLOCK_BYTES;
    let m4 = m & !3;
    let mask_0f = _mm_set1_epi8(0x0F);

    // Permutation indices for cross-lane interleave fix after unpacklo/unpackhi.
    // n0 picks: il.lane0(0..3), ih.lane0(16..19), il.lane1(4..7), ih.lane1(20..23)
    // n1 picks: il.lane2(8..11), ih.lane2(24..27), il.lane3(12..15), ih.lane3(28..31)
    let idx_n0 = _mm512_setr_epi32(0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
    let idx_n1 = _mm512_setr_epi32(8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);

    let mut row = 0usize;
    while row < m4 {
        let base0 = weight_blocks.add(row * row_bytes);
        let base1 = weight_blocks.add((row + 1) * row_bytes);
        let base2 = weight_blocks.add((row + 2) * row_bytes);
        let base3 = weight_blocks.add((row + 3) * row_bytes);

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let bpr2 = bpr & !1;
        let mut b = 0usize;

        // 2x block unrolled loop
        while b < bpr2 {
            // --- Block b ---
            let zero = _mm512_setzero_ps();
            // 2 accumulators per row (ZMM = 16 f32, 2×16 = 32 per iteration, 8 iters = 256)
            let (mut na00, mut na01) = (zero, zero);
            let (mut na10, mut na11) = (zero, zero);
            let (mut na20, mut na21) = (zero, zero);
            let (mut na30, mut na31) = (zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let noff = (b + 1) * Q4K_BLOCK_BYTES;
            _mm_prefetch(base0.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base1.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base2.add(noff) as *const i8, _MM_HINT_T0);
            _mm_prefetch(base3.add(noff) as *const i8, _MM_HINT_T0);

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            let mut i = 0usize;
            while i < 8 {
                let byte_off = i * 16;
                let val_off = i * 32;

                // 2 ZMM input loads = 32 f32 values
                let o0 = _mm512_loadu_ps(inp.add(val_off));
                let o1 = _mm512_loadu_ps(inp.add(val_off + 16));

                oa0 = _mm512_add_ps(oa0, o0);
                oa1 = _mm512_add_ps(oa1, o1);

                q4k_row_fma_avx512!(qs0, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na00, na01);
                q4k_row_fma_avx512!(qs1, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na10, na11);
                q4k_row_fma_avx512!(qs2, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na20, na21);
                q4k_row_fma_avx512!(qs3, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na30, na31);

                i += 1;
            }

            let nib0 = reduce2_avx512!(na00, na01);
            let nib1 = reduce2_avx512!(na10, na11);
            let nib2 = reduce2_avx512!(na20, na21);
            let nib3 = reduce2_avx512!(na30, na31);
            let oth = hsum_avx512(_mm512_add_ps(oa0, oa1));

            sum0 += d0 * nib0 - d0 * 8.0 * oth;
            sum1 += d1 * nib1 - d1 * 8.0 * oth;
            sum2 += d2 * nib2 - d2 * 8.0 * oth;
            sum3 += d3 * nib3 - d3 * 8.0 * oth;

            // --- Block b+1 ---
            let (mut na00, mut na01) = (zero, zero);
            let (mut na10, mut na11) = (zero, zero);
            let (mut na20, mut na21) = (zero, zero);
            let (mut na30, mut na31) = (zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let boff1 = noff;
            let inp1 = input.add((b + 1) * QK_K);
            let d0_b1 = half::f16::from_bits(*(base0.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1_b1 = half::f16::from_bits(*(base1.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2_b1 = half::f16::from_bits(*(base2.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3_b1 = half::f16::from_bits(*(base3.add(boff1 + Q4K_D_OFFSET) as *const u16)).to_f32();

            if b + 2 < bpr {
                let noff2 = (b + 2) * Q4K_BLOCK_BYTES;
                _mm_prefetch(base0.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base1.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base2.add(noff2) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base3.add(noff2) as *const i8, _MM_HINT_T0);
            }

            let qs0 = base0.add(boff1 + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff1 + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff1 + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff1 + Q4K_QS_OFFSET);

            let mut i = 0usize;
            while i < 8 {
                let byte_off = i * 16;
                let val_off = i * 32;

                let o0 = _mm512_loadu_ps(inp1.add(val_off));
                let o1 = _mm512_loadu_ps(inp1.add(val_off + 16));

                oa0 = _mm512_add_ps(oa0, o0);
                oa1 = _mm512_add_ps(oa1, o1);

                q4k_row_fma_avx512!(qs0, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na00, na01);
                q4k_row_fma_avx512!(qs1, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na10, na11);
                q4k_row_fma_avx512!(qs2, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na20, na21);
                q4k_row_fma_avx512!(qs3, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na30, na31);

                i += 1;
            }

            let nib0 = reduce2_avx512!(na00, na01);
            let nib1 = reduce2_avx512!(na10, na11);
            let nib2 = reduce2_avx512!(na20, na21);
            let nib3 = reduce2_avx512!(na30, na31);
            let oth = hsum_avx512(_mm512_add_ps(oa0, oa1));

            sum0 += d0_b1 * nib0 - d0_b1 * 8.0 * oth;
            sum1 += d1_b1 * nib1 - d1_b1 * 8.0 * oth;
            sum2 += d2_b1 * nib2 - d2_b1 * 8.0 * oth;
            sum3 += d3_b1 * nib3 - d3_b1 * 8.0 * oth;

            b += 2;
        }

        // Odd block remainder
        if bpr & 1 != 0 {
            let zero = _mm512_setzero_ps();
            let (mut na00, mut na01) = (zero, zero);
            let (mut na10, mut na11) = (zero, zero);
            let (mut na20, mut na21) = (zero, zero);
            let (mut na30, mut na31) = (zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            let mut i = 0usize;
            while i < 8 {
                let byte_off = i * 16;
                let val_off = i * 32;

                let o0 = _mm512_loadu_ps(inp.add(val_off));
                let o1 = _mm512_loadu_ps(inp.add(val_off + 16));

                oa0 = _mm512_add_ps(oa0, o0);
                oa1 = _mm512_add_ps(oa1, o1);

                q4k_row_fma_avx512!(qs0, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na00, na01);
                q4k_row_fma_avx512!(qs1, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na10, na11);
                q4k_row_fma_avx512!(qs2, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na20, na21);
                q4k_row_fma_avx512!(qs3, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na30, na31);

                i += 1;
            }

            let nib0 = reduce2_avx512!(na00, na01);
            let nib1 = reduce2_avx512!(na10, na11);
            let nib2 = reduce2_avx512!(na20, na21);
            let nib3 = reduce2_avx512!(na30, na31);
            let oth = hsum_avx512(_mm512_add_ps(oa0, oa1));

            sum0 += d0 * nib0 - d0 * 8.0 * oth;
            sum1 += d1 * nib1 - d1 * 8.0 * oth;
            sum2 += d2 * nib2 - d2 * 8.0 * oth;
            sum3 += d3 * nib3 - d3 * 8.0 * oth;
        }

        *output.add(row) = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // Tail: 1 row at a time (2 ZMM accumulators)
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let boff = b * Q4K_BLOCK_BYTES;
            let d = half::f16::from_bits(*(base.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let qs = base.add(boff + Q4K_QS_OFFSET);
            let inp = input.add(b * QK_K);

            let zero = _mm512_setzero_ps();
            let (mut na0, mut na1) = (zero, zero);
            let (mut oa0, mut oa1) = (zero, zero);

            let mut i = 0usize;
            while i < 8 {
                let byte_off = i * 16;
                let val_off = i * 32;

                let o0 = _mm512_loadu_ps(inp.add(val_off));
                let o1 = _mm512_loadu_ps(inp.add(val_off + 16));

                oa0 = _mm512_add_ps(oa0, o0);
                oa1 = _mm512_add_ps(oa1, o1);

                q4k_row_fma_avx512!(qs, byte_off, o0, o1, mask_0f, idx_n0, idx_n1, na0, na1);

                i += 1;
            }

            let nib_sum = reduce2_avx512!(na0, na1);
            let oth_sum = hsum_avx512(_mm512_add_ps(oa0, oa1));
            sum += d * nib_sum - d * 8.0 * oth_sum;
        }
        *output.add(row) = sum;
        row += 1;
    }
}
