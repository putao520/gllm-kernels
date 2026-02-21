//! Hand-written NEON Q4K GEMV microkernel (aarch64).
//!
//! Implements fused 4-row Q4K GEMV using global_asm! for the hot inner loop.
//!
//! Q4K block layout (144 bytes total, matching BlockQ4K):
//!   offset  0: d     (f16, 2 bytes)  — block scale
//!   offset  2: dmin  (f16, 2 bytes)  — block min scale
//!   offset  4: scales[12] (u8)       — sub-block scales (6 scales + 6 mins packed)
//!   offset 16: qs[128] (u8)          — 256 nibble-packed 4-bit values
//!
//! NOTE: This first version uses a simplified deferred-scale approach.
//! The full Q4K decode with sub-block scales/mins will be added in a follow-up.
//!
//! Assembly microkernel: _gllm_gemv_q4k_block_4row_neon
//!   Processes one Q4K block (128 bytes of qs = 256 nibbles) for 4 rows.
//!   8 iterations, each handling 16 bytes of qs (32 nibbles = 32 values).
//!
//! Register allocation (32 NEON 128-bit V registers):
//!   v0  - v1  : row0 nibble accumulators
//!   v2  - v3  : row1 nibble accumulators
//!   v4  - v5  : row2 nibble accumulators
//!   v6  - v7  : row3 nibble accumulators
//!   v8  - v9  : input sum accumulators (shared)
//!   v10 - v13 : input vectors (current iteration)
//!   v14       : nibble mask (0x0F × 16)
//!   v15 - v24 : nibble unpack + conversion temporaries
//!   v25       : horizontal reduction scratch
//!   v27 - v30 : qs load for 4 rows

#[cfg(target_arch = "aarch64")]
use std::arch::global_asm;

/// Q4K block layout constants.
pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q4K_D_OFFSET: usize = 0;
pub const Q4K_DMIN_OFFSET: usize = 2;
pub const Q4K_SCALES_OFFSET: usize = 4;
pub const Q4K_QS_OFFSET: usize = 16;
pub const QK_K: usize = 256;

// ============================================================================
// Assembly microkernel: one Q4K block × 4 rows
// ============================================================================
//
// Processes 128 bytes of packed nibbles (= 256 4-bit values) for 4 rows.
// Each iteration: 16 bytes qs → 32 nibbles → 32 f32 values.
// 8 iterations total.
//
// Nibble unpack for 16 bytes → 32 values:
//   lo = v AND 0x0F          (16 × u8 low nibbles)
//   hi = (v >> 4) AND 0x0F   (16 × u8 high nibbles)
//   interleave via zip1/zip2 to get sequential order
//   convert u8 → u16 → u32 → f32 via UXTL + SCVTF
//
// Calling convention (AAPCS64):
//   x0 = qs0 (*const u8, row 0, 128 bytes)
//   x1 = qs1 (*const u8, row 1, 128 bytes)
//   x2 = qs2 (*const u8, row 2, 128 bytes)
//   x3 = qs3 (*const u8, row 3, 128 bytes)
//   x4 = input (*const f32, 256 values)
//   x5 = out_nib (*mut f32, 4 scalars: nibble dot for each row)
//   x6 = out_sum (*mut f32, 1 scalar: shared input sum)
#[cfg(target_arch = "aarch64")]
global_asm!(
    ".text",
    ".align 6",
    ".global _gllm_gemv_q4k_block_4row_neon",
    ".type _gllm_gemv_q4k_block_4row_neon, %function",
    "_gllm_gemv_q4k_block_4row_neon:",

    // Save callee-saved NEON registers d8-d15
    "stp d8,  d9,  [sp, #-64]!",
    "stp d10, d11, [sp, #16]",
    "stp d12, d13, [sp, #32]",
    "stp d14, d15, [sp, #48]",

    // Zero all accumulators
    "movi v0.4s,  #0",
    "movi v1.4s,  #0",
    "movi v2.4s,  #0",
    "movi v3.4s,  #0",
    "movi v4.4s,  #0",
    "movi v5.4s,  #0",
    "movi v6.4s,  #0",
    "movi v7.4s,  #0",
    "movi v8.4s,  #0",
    "movi v9.4s,  #0",

    // Nibble mask: 0x0F in all 16 bytes
    "movi v14.16b, #0x0F",

    // Loop counter: 8 iterations
    "mov x7, #8",

    ".align 5",
    ".Lq4k_neon_block_loop:",

    // ---- Load 16 bytes of qs for each row ----
    "ldr q27, [x0], #16",
    "ldr q28, [x1], #16",
    "ldr q29, [x2], #16",
    "ldr q30, [x3], #16",

    // ---- Load first 16 f32 input values ----
    "ldp q10, q11, [x4]",
    "ldp q12, q13, [x4, #32]",

    // ---- Accumulate input sum ----
    "fadd v8.4s, v8.4s, v10.4s",
    "fadd v9.4s, v9.4s, v11.4s",
    "fadd v8.4s, v8.4s, v12.4s",
    "fadd v9.4s, v9.4s, v13.4s",

    // ==== Row 0: unpack v27 nibbles, FMA into v0-v1 ====
    "and  v15.16b, v27.16b, v14.16b",       // lo nibbles
    "ushr v16.16b, v27.16b, #4",            // hi nibbles
    "and  v16.16b, v16.16b, v14.16b",
    "zip1 v17.16b, v15.16b, v16.16b",       // interleave first half
    "zip2 v18.16b, v15.16b, v16.16b",       // interleave second half

    // v17 → 16 u8 → 16 f32 (4 vectors)
    "uxtl  v19.8h, v17.8b",
    "uxtl2 v20.8h, v17.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v0.4s, v21.4s, v10.4s",
    "fmla v0.4s, v22.4s, v11.4s",
    "fmla v1.4s, v23.4s, v12.4s",
    "fmla v1.4s, v24.4s, v13.4s",

    // Load second 16 f32 input
    "ldp q10, q11, [x4, #64]",
    "ldp q12, q13, [x4, #96]",

    // Accumulate input sum for second half
    "fadd v8.4s, v8.4s, v10.4s",
    "fadd v9.4s, v9.4s, v11.4s",
    "fadd v8.4s, v8.4s, v12.4s",
    "fadd v9.4s, v9.4s, v13.4s",

    // v18 → 16 u8 → 16 f32
    "uxtl  v19.8h, v18.8b",
    "uxtl2 v20.8h, v18.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v0.4s, v21.4s, v10.4s",
    "fmla v0.4s, v22.4s, v11.4s",
    "fmla v1.4s, v23.4s, v12.4s",
    "fmla v1.4s, v24.4s, v13.4s",

    // ==== Row 1: unpack v28 nibbles, FMA into v2-v3 ====
    "and  v15.16b, v28.16b, v14.16b",
    "ushr v16.16b, v28.16b, #4",
    "and  v16.16b, v16.16b, v14.16b",
    "zip1 v17.16b, v15.16b, v16.16b",
    "zip2 v18.16b, v15.16b, v16.16b",

    // Reload first 16 f32
    "ldp q10, q11, [x4]",
    "ldp q12, q13, [x4, #32]",

    "uxtl  v19.8h, v17.8b",
    "uxtl2 v20.8h, v17.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v2.4s, v21.4s, v10.4s",
    "fmla v2.4s, v22.4s, v11.4s",
    "fmla v3.4s, v23.4s, v12.4s",
    "fmla v3.4s, v24.4s, v13.4s",

    "ldp q10, q11, [x4, #64]",
    "ldp q12, q13, [x4, #96]",

    "uxtl  v19.8h, v18.8b",
    "uxtl2 v20.8h, v18.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v2.4s, v21.4s, v10.4s",
    "fmla v2.4s, v22.4s, v11.4s",
    "fmla v3.4s, v23.4s, v12.4s",
    "fmla v3.4s, v24.4s, v13.4s",

    // ==== Row 2: unpack v29 nibbles, FMA into v4-v5 ====
    "and  v15.16b, v29.16b, v14.16b",
    "ushr v16.16b, v29.16b, #4",
    "and  v16.16b, v16.16b, v14.16b",
    "zip1 v17.16b, v15.16b, v16.16b",
    "zip2 v18.16b, v15.16b, v16.16b",

    "ldp q10, q11, [x4]",
    "ldp q12, q13, [x4, #32]",

    "uxtl  v19.8h, v17.8b",
    "uxtl2 v20.8h, v17.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v4.4s, v21.4s, v10.4s",
    "fmla v4.4s, v22.4s, v11.4s",
    "fmla v5.4s, v23.4s, v12.4s",
    "fmla v5.4s, v24.4s, v13.4s",

    "ldp q10, q11, [x4, #64]",
    "ldp q12, q13, [x4, #96]",

    "uxtl  v19.8h, v18.8b",
    "uxtl2 v20.8h, v18.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v4.4s, v21.4s, v10.4s",
    "fmla v4.4s, v22.4s, v11.4s",
    "fmla v5.4s, v23.4s, v12.4s",
    "fmla v5.4s, v24.4s, v13.4s",

    // ==== Row 3: unpack v30 nibbles, FMA into v6-v7 ====
    "and  v15.16b, v30.16b, v14.16b",
    "ushr v16.16b, v30.16b, #4",
    "and  v16.16b, v16.16b, v14.16b",
    "zip1 v17.16b, v15.16b, v16.16b",
    "zip2 v18.16b, v15.16b, v16.16b",

    "ldp q10, q11, [x4]",
    "ldp q12, q13, [x4, #32]",

    "uxtl  v19.8h, v17.8b",
    "uxtl2 v20.8h, v17.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v6.4s, v21.4s, v10.4s",
    "fmla v6.4s, v22.4s, v11.4s",
    "fmla v7.4s, v23.4s, v12.4s",
    "fmla v7.4s, v24.4s, v13.4s",

    "ldp q10, q11, [x4, #64]",
    "ldp q12, q13, [x4, #96]",

    "uxtl  v19.8h, v18.8b",
    "uxtl2 v20.8h, v18.16b",
    "uxtl  v21.4s, v19.4h",
    "uxtl2 v22.4s, v19.8h",
    "uxtl  v23.4s, v20.4h",
    "uxtl2 v24.4s, v20.8h",
    "ucvtf v21.4s, v21.4s",
    "ucvtf v22.4s, v22.4s",
    "ucvtf v23.4s, v23.4s",
    "ucvtf v24.4s, v24.4s",
    "fmla v6.4s, v21.4s, v10.4s",
    "fmla v6.4s, v22.4s, v11.4s",
    "fmla v7.4s, v23.4s, v12.4s",
    "fmla v7.4s, v24.4s, v13.4s",

    // Advance input pointer by 32 f32 = 128 bytes
    "add x4, x4, #128",

    // Loop
    "subs x7, x7, #1",
    "b.ne .Lq4k_neon_block_loop",

    // ---- Horizontal reduce accumulators → scalars ----
    // row0: v0 + v1 → scalar
    "fadd v25.4s, v0.4s, v1.4s",
    "faddp v25.4s, v25.4s, v25.4s",
    "faddp s25, v25.2s",
    "str s25, [x5, #0]",

    // row1
    "fadd v25.4s, v2.4s, v3.4s",
    "faddp v25.4s, v25.4s, v25.4s",
    "faddp s25, v25.2s",
    "str s25, [x5, #4]",

    // row2
    "fadd v25.4s, v4.4s, v5.4s",
    "faddp v25.4s, v25.4s, v25.4s",
    "faddp s25, v25.2s",
    "str s25, [x5, #8]",

    // row3
    "fadd v25.4s, v6.4s, v7.4s",
    "faddp v25.4s, v25.4s, v25.4s",
    "faddp s25, v25.2s",
    "str s25, [x5, #12]",

    // input sum
    "fadd v25.4s, v8.4s, v9.4s",
    "faddp v25.4s, v25.4s, v25.4s",
    "faddp s25, v25.2s",
    "str s25, [x6]",

    // Restore callee-saved registers
    "ldp d10, d11, [sp, #16]",
    "ldp d12, d13, [sp, #32]",
    "ldp d14, d15, [sp, #48]",
    "ldp d8,  d9,  [sp], #64",
    "ret",

    ".size _gllm_gemv_q4k_block_4row_neon, . - _gllm_gemv_q4k_block_4row_neon",
);

// ============================================================================
// extern "C" declaration
// ============================================================================

#[cfg(target_arch = "aarch64")]
extern "C" {
    /// Process one Q4K block for 4 rows using NEON assembly.
    ///
    /// # Safety
    /// - qs0..qs3: each must point to 128 bytes of valid nibble data
    /// - inp: must point to 256 valid f32 values
    /// - out_nib: must point to 4 writable f32 slots
    /// - out_sum: must point to 1 writable f32 slot
    fn _gllm_gemv_q4k_block_4row_neon(
        qs0: *const u8,
        qs1: *const u8,
        qs2: *const u8,
        qs3: *const u8,
        inp: *const f32,
        out_nib: *mut f32,
        out_sum: *mut f32,
    );
}

// ============================================================================
// Fused 4-row Q4K GEMV — NEON
// ============================================================================

/// Fused 4-row Q4K GEMV using NEON assembly inner loop.
///
/// Uses simplified deferred-scale: dot = d * nib_sum - d * 8 * input_sum.
/// NOTE: Full sub-block scale decode will be added in a follow-up.
///
/// # Safety
/// - `weight_blocks`: `m * (k/256) * 144` bytes of contiguous Q4K block data
/// - `input`: `k` f32 values
/// - `output`: `m` f32 values (written, not accumulated)
/// - `k` must be a multiple of 256
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_q4k_fused_neon(
    weight_blocks: *const u8,
    input: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
) {
    let bpr = k / QK_K;
    let row_bytes = bpr * Q4K_BLOCK_BYTES;
    let m4 = m & !3;

    let mut out_nib = [0.0f32; 4];
    let mut out_sum = 0.0f32;

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

        for b in 0..bpr {
            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);

            // Read d (f16) for each row
            let d0 = half::f16::from_bits(*(base0.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d1 = half::f16::from_bits(*(base1.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d2 = half::f16::from_bits(*(base2.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let d3 = half::f16::from_bits(*(base3.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();

            let qs0 = base0.add(boff + Q4K_QS_OFFSET);
            let qs1 = base1.add(boff + Q4K_QS_OFFSET);
            let qs2 = base2.add(boff + Q4K_QS_OFFSET);
            let qs3 = base3.add(boff + Q4K_QS_OFFSET);

            _gllm_gemv_q4k_block_4row_neon(
                qs0, qs1, qs2, qs3,
                inp,
                out_nib.as_mut_ptr(),
                &mut out_sum,
            );

            // Simplified deferred-scale
            sum0 += d0 * out_nib[0] - d0 * 8.0 * out_sum;
            sum1 += d1 * out_nib[1] - d1 * 8.0 * out_sum;
            sum2 += d2 * out_nib[2] - d2 * 8.0 * out_sum;
            sum3 += d3 * out_nib[3] - d3 * 8.0 * out_sum;
        }

        *output.add(row)     = sum0;
        *output.add(row + 1) = sum1;
        *output.add(row + 2) = sum2;
        *output.add(row + 3) = sum3;
        row += 4;
    }

    // ---- Tail: 1 row at a time (scalar fallback) ----
    while row < m {
        let base = weight_blocks.add(row * row_bytes);
        let mut sum = 0.0f32;

        for b in 0..bpr {
            let boff = b * Q4K_BLOCK_BYTES;
            let inp = input.add(b * QK_K);
            let d = half::f16::from_bits(*(base.add(boff + Q4K_D_OFFSET) as *const u16)).to_f32();
            let qs = base.add(boff + Q4K_QS_OFFSET);

            let mut nib_sum = 0.0f32;
            let mut inp_sum = 0.0f32;
            for i in 0..128 {
                let byte = *qs.add(i);
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                let in0 = *inp.add(i * 2);
                let in1 = *inp.add(i * 2 + 1);
                nib_sum += lo * in0 + hi * in1;
                inp_sum += in0 + in1;
            }
            sum += d * nib_sum - d * 8.0 * inp_sum;
        }

        *output.add(row) = sum;
        row += 1;
    }
}
