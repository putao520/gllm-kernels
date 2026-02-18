//! Hand-written NEON GEMM 8x12 microkernel (aarch64).
//!
//! Microkernel geometry: 8 rows x 12 columns (3 x float32x4_t vectors)
//! = 24 accumulator registers (v0-v23)
//!
//! Register allocation (32 NEON registers total):
//!   v0  - v2  : C row 0 (3 vectors x 4 lanes = 12 columns)
//!   v3  - v5  : C row 1
//!   v6  - v8  : C row 2
//!   v9  - v11 : C row 3
//!   v12 - v14 : C row 4
//!   v15 - v17 : C row 5
//!   v18 - v20 : C row 6
//!   v21 - v23 : C row 7
//!   v24 - v27 : A elements (broadcast scalars for 4 rows, software pipeline)
//!   v28 - v30 : B vectors (3 vectors for current k)
//!   v31       : temp / B prefetch
//!
//! Software pipeline strategy:
//!   - Load B[k+1] while computing FMA for B[k]
//!   - Interleave A loads with FMA to hide load latency
//!   - Prefetch A and B data 8 cache lines ahead
//!
//! Calling convention (Rust extern "C"):
//!   x0 = *const f32  : packed_a (TM x KC, column-major panels)
//!   x1 = *const f32  : packed_b (KC x TN, row-major panels)
//!   x2 = *mut f32    : c_ptr (row-major, stride = ldc)
//!   x3 = usize       : kc (K-dimension loop count)
//!   x4 = usize       : ldc (C matrix row stride in elements, NOT bytes)
//!   x5 = usize       : accumulate (0 = zero C first, 1 = load existing C)

#[cfg(target_arch = "aarch64")]
use std::arch::global_asm;

/// Microkernel tile dimensions
pub const MR: usize = 8;
pub const NR: usize = 12;

// The assembly microkernel: gemm_kernel_8x12_neon
//
// This is the innermost loop of the GEMM. It computes:
//   C[8x12] += A[8xKC] * B[KCx12]
//
// A is packed column-major: a[row + k*MR] (MR=8 contiguous rows per k)
// B is packed row-major: b[col + k*NR] (NR=12 contiguous cols per k)
// C is row-major with stride ldc.
#[cfg(target_arch = "aarch64")]
global_asm!(
    ".text",
    ".align 6",  // 64-byte alignment for cache line
    ".global _gllm_gemm_8x12_neon_f32",
    ".type _gllm_gemm_8x12_neon_f32, %function",
    "_gllm_gemm_8x12_neon_f32:",

    // Save callee-saved NEON registers (v8-v15 per AAPCS64)
    "stp d8, d9, [sp, #-64]!",
    "stp d10, d11, [sp, #16]",
    "stp d12, d13, [sp, #32]",
    "stp d14, d15, [sp, #48]",

    // x0=packed_a, x1=packed_b, x2=c_ptr, x3=kc, x4=ldc, x5=accumulate
    // Convert ldc from elements to bytes: ldc_bytes = ldc * 4
    "lsl x4, x4, #2",

    // Compute C row pointers: x6..x13 = c + row*ldc_bytes
    "mov x6, x2",                // row 0
    "add x7, x6, x4",           // row 1
    "add x8, x7, x4",           // row 2
    "add x9, x8, x4",           // row 3
    "add x10, x9, x4",          // row 4
    "add x11, x10, x4",         // row 5
    "add x12, x11, x4",         // row 6
    "add x13, x12, x4",         // row 7

    // Branch: accumulate or zero-init C accumulators
    "cbz x5, .Lzero_c_f32",

    // Load existing C values into accumulators
    "ldp q0, q1, [x6]",
    "ldr q2, [x6, #32]",
    "ldp q3, q4, [x7]",
    "ldr q5, [x7, #32]",
    "ldp q6, q7, [x8]",
    "ldr q8, [x8, #32]",
    "ldp q9, q10, [x9]",
    "ldr q11, [x9, #32]",
    "ldp q12, q13, [x10]",
    "ldr q14, [x10, #32]",
    "ldp q15, q16, [x11]",
    "ldr q17, [x11, #32]",
    "ldp q18, q19, [x12]",
    "ldr q20, [x12, #32]",
    "ldp q21, q22, [x13]",
    "ldr q23, [x13, #32]",
    "b .Lk_loop_setup_f32",

    ".Lzero_c_f32:",
    // Zero all 24 accumulator registers
    "movi v0.4s, #0",
    "movi v1.4s, #0",
    "movi v2.4s, #0",
    "movi v3.4s, #0",
    "movi v4.4s, #0",
    "movi v5.4s, #0",
    "movi v6.4s, #0",
    "movi v7.4s, #0",
    "movi v8.4s, #0",
    "movi v9.4s, #0",
    "movi v10.4s, #0",
    "movi v11.4s, #0",
    "movi v12.4s, #0",
    "movi v13.4s, #0",
    "movi v14.4s, #0",
    "movi v15.4s, #0",
    "movi v16.4s, #0",
    "movi v17.4s, #0",
    "movi v18.4s, #0",
    "movi v19.4s, #0",
    "movi v20.4s, #0",
    "movi v21.4s, #0",
    "movi v22.4s, #0",
    "movi v23.4s, #0",

    ".Lk_loop_setup_f32:",
    // Check if kc >= 4 for unrolled loop
    "cmp x3, #4",
    "b.lt .Lk_remainder_f32",

    // Preload first B panel (3 vectors = 12 floats = 48 bytes)
    "ldp q28, q29, [x1]",
    "ldr q30, [x1, #32]",

    // Main K-loop: 4x unrolled with software pipeline
    // Each iteration: load A[k], compute FMA with B[k], load B[k+1]
    "sub x3, x3, #4",  // pre-decrement for pipeline

    ".align 5",
    ".Lk_loop_4x_f32:",
    // ---- k+0 ----
    // Load A column k+0 (8 floats = 32 bytes)
    "ldp q24, q25, [x0]",       // a[0..3], a[4..7] for k+0

    // Prefetch A and B ahead
    "prfm pldl1keep, [x0, #256]",
    "prfm pldl1keep, [x1, #256]",

    // FMA: row 0-3 with B[k+0]
    "fmla v0.4s, v28.4s, v24.s[0]",
    "fmla v1.4s, v29.4s, v24.s[0]",
    "fmla v2.4s, v30.4s, v24.s[0]",
    "fmla v3.4s, v28.4s, v24.s[1]",
    "fmla v4.4s, v29.4s, v24.s[1]",
    "fmla v5.4s, v30.4s, v24.s[1]",
    "fmla v6.4s, v28.4s, v24.s[2]",
    "fmla v7.4s, v29.4s, v24.s[2]",
    "fmla v8.4s, v30.4s, v24.s[2]",
    "fmla v9.4s, v28.4s, v24.s[3]",
    "fmla v10.4s, v29.4s, v24.s[3]",
    "fmla v11.4s, v30.4s, v24.s[3]",

    // FMA: row 4-7 with B[k+0]
    "fmla v12.4s, v28.4s, v25.s[0]",
    "fmla v13.4s, v29.4s, v25.s[0]",
    "fmla v14.4s, v30.4s, v25.s[0]",
    "fmla v15.4s, v28.4s, v25.s[1]",
    "fmla v16.4s, v29.4s, v25.s[1]",
    "fmla v17.4s, v30.4s, v25.s[1]",
    "fmla v18.4s, v28.4s, v25.s[2]",
    "fmla v19.4s, v29.4s, v25.s[2]",
    "fmla v20.4s, v30.4s, v25.s[2]",
    "fmla v21.4s, v28.4s, v25.s[3]",
    "fmla v22.4s, v29.4s, v25.s[3]",
    "fmla v23.4s, v30.4s, v25.s[3]",

    // Load B[k+1] (pipeline: load next B while FMA for k+0 completes)
    "ldp q28, q29, [x1, #48]",
    "ldr q30, [x1, #80]",

    // ---- k+1 ----
    "ldp q24, q25, [x0, #32]",  // A column k+1

    "fmla v0.4s, v28.4s, v24.s[0]",
    "fmla v1.4s, v29.4s, v24.s[0]",
    "fmla v2.4s, v30.4s, v24.s[0]",
    "fmla v3.4s, v28.4s, v24.s[1]",
    "fmla v4.4s, v29.4s, v24.s[1]",
    "fmla v5.4s, v30.4s, v24.s[1]",
    "fmla v6.4s, v28.4s, v24.s[2]",
    "fmla v7.4s, v29.4s, v24.s[2]",
    "fmla v8.4s, v30.4s, v24.s[2]",
    "fmla v9.4s, v28.4s, v24.s[3]",
    "fmla v10.4s, v29.4s, v24.s[3]",
    "fmla v11.4s, v30.4s, v24.s[3]",

    "fmla v12.4s, v28.4s, v25.s[0]",
    "fmla v13.4s, v29.4s, v25.s[0]",
    "fmla v14.4s, v30.4s, v25.s[0]",
    "fmla v15.4s, v28.4s, v25.s[1]",
    "fmla v16.4s, v29.4s, v25.s[1]",
    "fmla v17.4s, v30.4s, v25.s[1]",
    "fmla v18.4s, v28.4s, v25.s[2]",
    "fmla v19.4s, v29.4s, v25.s[2]",
    "fmla v20.4s, v30.4s, v25.s[2]",
    "fmla v21.4s, v28.4s, v25.s[3]",
    "fmla v22.4s, v29.4s, v25.s[3]",
    "fmla v23.4s, v30.4s, v25.s[3]",

    // Load B[k+2]
    "ldp q28, q29, [x1, #96]",
    "ldr q30, [x1, #128]",

    // ---- k+2 ----
    "ldp q24, q25, [x0, #64]",  // A column k+2

    "prfm pldl1keep, [x0, #384]",
    "prfm pldl1keep, [x1, #384]",

    "fmla v0.4s, v28.4s, v24.s[0]",
    "fmla v1.4s, v29.4s, v24.s[0]",
    "fmla v2.4s, v30.4s, v24.s[0]",
    "fmla v3.4s, v28.4s, v24.s[1]",
    "fmla v4.4s, v29.4s, v24.s[1]",
    "fmla v5.4s, v30.4s, v24.s[1]",
    "fmla v6.4s, v28.4s, v24.s[2]",
    "fmla v7.4s, v29.4s, v24.s[2]",
    "fmla v8.4s, v30.4s, v24.s[2]",
    "fmla v9.4s, v28.4s, v24.s[3]",
    "fmla v10.4s, v29.4s, v24.s[3]",
    "fmla v11.4s, v30.4s, v24.s[3]",

    "fmla v12.4s, v28.4s, v25.s[0]",
    "fmla v13.4s, v29.4s, v25.s[0]",
    "fmla v14.4s, v30.4s, v25.s[0]",
    "fmla v15.4s, v28.4s, v25.s[1]",
    "fmla v16.4s, v29.4s, v25.s[1]",
    "fmla v17.4s, v30.4s, v25.s[1]",
    "fmla v18.4s, v28.4s, v25.s[2]",
    "fmla v19.4s, v29.4s, v25.s[2]",
    "fmla v20.4s, v30.4s, v25.s[2]",
    "fmla v21.4s, v28.4s, v25.s[3]",
    "fmla v22.4s, v29.4s, v25.s[3]",
    "fmla v23.4s, v30.4s, v25.s[3]",

    // Load B[k+3]
    "ldp q28, q29, [x1, #144]",
    "ldr q30, [x1, #176]",

    // ---- k+3 ----
    "ldp q24, q25, [x0, #96]",  // A column k+3

    "fmla v0.4s, v28.4s, v24.s[0]",
    "fmla v1.4s, v29.4s, v24.s[0]",
    "fmla v2.4s, v30.4s, v24.s[0]",
    "fmla v3.4s, v28.4s, v24.s[1]",
    "fmla v4.4s, v29.4s, v24.s[1]",
    "fmla v5.4s, v30.4s, v24.s[1]",
    "fmla v6.4s, v28.4s, v24.s[2]",
    "fmla v7.4s, v29.4s, v24.s[2]",
    "fmla v8.4s, v30.4s, v24.s[2]",
    "fmla v9.4s, v28.4s, v24.s[3]",
    "fmla v10.4s, v29.4s, v24.s[3]",
    "fmla v11.4s, v30.4s, v24.s[3]",

    "fmla v12.4s, v28.4s, v25.s[0]",
    "fmla v13.4s, v29.4s, v25.s[0]",
    "fmla v14.4s, v30.4s, v25.s[0]",
    "fmla v15.4s, v28.4s, v25.s[1]",
    "fmla v16.4s, v29.4s, v25.s[1]",
    "fmla v17.4s, v30.4s, v25.s[1]",
    "fmla v18.4s, v28.4s, v25.s[2]",
    "fmla v19.4s, v29.4s, v25.s[2]",
    "fmla v20.4s, v30.4s, v25.s[2]",
    "fmla v21.4s, v28.4s, v25.s[3]",
    "fmla v22.4s, v29.4s, v25.s[3]",
    "fmla v23.4s, v30.4s, v25.s[3]",

    // Advance pointers: A += 4*MR*4 = 128 bytes, B += 4*NR*4 = 192 bytes
    "add x0, x0, #128",
    "add x1, x1, #192",

    // Loop: pre-load next B[k+0] for next iteration
    "subs x3, x3, #4",
    "b.lt .Lk_loop_4x_done_f32",
    "ldp q28, q29, [x1]",
    "ldr q30, [x1, #32]",
    "b .Lk_loop_4x_f32",

    ".Lk_loop_4x_done_f32:",
    // Restore kc remainder: x3 = x3 + 4 (was pre-decremented)
    "add x3, x3, #4",

    ".Lk_remainder_f32:",
    // Handle remaining k iterations (0..3)
    "cbz x3, .Lstore_c_f32",

    ".Lk_tail_f32:",
    // Load B[k] (3 vectors)
    "ldp q28, q29, [x1]",
    "ldr q30, [x1, #32]",
    // Load A[k] (8 elements = 2 vectors)
    "ldp q24, q25, [x0]",

    // FMA all 8 rows
    "fmla v0.4s, v28.4s, v24.s[0]",
    "fmla v1.4s, v29.4s, v24.s[0]",
    "fmla v2.4s, v30.4s, v24.s[0]",
    "fmla v3.4s, v28.4s, v24.s[1]",
    "fmla v4.4s, v29.4s, v24.s[1]",
    "fmla v5.4s, v30.4s, v24.s[1]",
    "fmla v6.4s, v28.4s, v24.s[2]",
    "fmla v7.4s, v29.4s, v24.s[2]",
    "fmla v8.4s, v30.4s, v24.s[2]",
    "fmla v9.4s, v28.4s, v24.s[3]",
    "fmla v10.4s, v29.4s, v24.s[3]",
    "fmla v11.4s, v30.4s, v24.s[3]",

    "fmla v12.4s, v28.4s, v25.s[0]",
    "fmla v13.4s, v29.4s, v25.s[0]",
    "fmla v14.4s, v30.4s, v25.s[0]",
    "fmla v15.4s, v28.4s, v25.s[1]",
    "fmla v16.4s, v29.4s, v25.s[1]",
    "fmla v17.4s, v30.4s, v25.s[1]",
    "fmla v18.4s, v28.4s, v25.s[2]",
    "fmla v19.4s, v29.4s, v25.s[2]",
    "fmla v20.4s, v30.4s, v25.s[2]",
    "fmla v21.4s, v28.4s, v25.s[3]",
    "fmla v22.4s, v29.4s, v25.s[3]",
    "fmla v23.4s, v30.4s, v25.s[3]",

    // Advance: A += MR*4 = 32 bytes, B += NR*4 = 48 bytes
    "add x0, x0, #32",
    "add x1, x1, #48",
    "subs x3, x3, #1",
    "b.ne .Lk_tail_f32",

    ".Lstore_c_f32:",
    // Store C accumulators back to memory
    "stp q0, q1, [x6]",
    "str q2, [x6, #32]",
    "stp q3, q4, [x7]",
    "str q5, [x7, #32]",
    "stp q6, q7, [x8]",
    "str q8, [x8, #32]",
    "stp q9, q10, [x9]",
    "str q11, [x9, #32]",
    "stp q12, q13, [x10]",
    "str q14, [x10, #32]",
    "stp q15, q16, [x11]",
    "str q17, [x11, #32]",
    "stp q18, q19, [x12]",
    "str q20, [x12, #32]",
    "stp q21, q22, [x13]",
    "str q23, [x13, #32]",

    // Restore callee-saved registers
    "ldp d10, d11, [sp, #16]",
    "ldp d12, d13, [sp, #32]",
    "ldp d14, d15, [sp, #48]",
    "ldp d8, d9, [sp], #64",
    "ret",

    ".size _gllm_gemm_8x12_neon_f32, . - _gllm_gemm_8x12_neon_f32",
);

// FP16 microkernel: same 8x12 geometry but with float16x8_t vectors
// NR = 24 (3 vectors x 8 f16 lanes) -- different tile shape
// For now, f16 uses the macro-generated path; native f16 FMA microkernel
// will be added when targeting ARMv8.2-A+ with FP16 extension.

extern "C" {
    /// Raw assembly entry point for the 8x12 f32 NEON GEMM microkernel.
    ///
    /// # Safety
    /// - All pointers must be valid and properly aligned (16-byte for NEON)
    /// - packed_a must contain at least kc * MR f32 elements
    /// - packed_b must contain at least kc * NR f32 elements
    /// - c_ptr must point to a buffer with at least MR rows of ldc elements each
    /// - kc must be > 0
    #[cfg(target_arch = "aarch64")]
    fn _gllm_gemm_8x12_neon_f32(
        packed_a: *const f32,
        packed_b: *const f32,
        c_ptr: *mut f32,
        kc: usize,
        ldc: usize,
        accumulate: usize,
    );
}

/// Safe Rust wrapper for the 8x12 NEON GEMM microkernel.
///
/// Computes: C[8x12] += A[8xKC] * B[KCx12]  (if accumulate=true)
///       or: C[8x12]  = A[8xKC] * B[KCx12]  (if accumulate=false)
///
/// # Arguments
/// - `packed_a`: Packed A panel, column-major within panel (MR elements per k)
/// - `packed_b`: Packed B panel, row-major within panel (NR elements per k)
/// - `c_ptr`: Pointer to C matrix tile (row-major)
/// - `kc`: Number of K iterations
/// - `ldc`: Leading dimension of C (row stride in elements)
/// - `accumulate`: If true, add to existing C; if false, overwrite C
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn gemm_kernel_8x12_f32(
    packed_a: *const f32,
    packed_b: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    ldc: usize,
    accumulate: bool,
) {
    _gllm_gemm_8x12_neon_f32(
        packed_a,
        packed_b,
        c_ptr,
        kc,
        ldc,
        accumulate as usize,
    );
}
