//! Hand-written AVX2 GEMM 6x16 microkernel (x86_64).
//!
//! Microkernel geometry: 6 rows x 16 columns (2 x YMM vectors per row)
//! = 12 accumulator registers (ymm0-ymm11)
//!
//! Register allocation (16 YMM registers total):
//!   ymm0  - ymm1  : C row 0 (2 vectors x 8 lanes = 16 columns)
//!   ymm2  - ymm3  : C row 1
//!   ymm4  - ymm5  : C row 2
//!   ymm6  - ymm7  : C row 3
//!   ymm8  - ymm9  : C row 4
//!   ymm10 - ymm11 : C row 5
//!   ymm12          : A broadcast (current element)
//!   ymm13          : B vector 0 (current k)
//!   ymm14          : B vector 1 (current k)
//!   ymm15          : temp / prefetch
//!
//! Calling convention (System V AMD64 ABI, extern "C"):
//!   rdi = *const f32 : packed_a (MR x KC, column-major panels: a[row + k*MR])
//!   rsi = *const f32 : packed_b (KC x NR, row-major panels: b[col + k*NR])
//!   rdx = *mut f32   : c_ptr (row-major, stride = ldc)
//!   rcx = usize      : kc (K-dimension loop count)
//!   r8  = usize      : ldc (C matrix row stride in elements, NOT bytes)
//!   r9  = usize      : accumulate (0 = zero C first, 1 = load existing C)

#[cfg(target_arch = "x86_64")]
use std::arch::global_asm;

/// Microkernel tile dimensions.
pub const MR: usize = 6;
pub const NR: usize = 16;

// ── AVX2 6x16 f32 GEMM microkernel ─────────────────────────────────────
//
// Computes: C[6x16] += A[6xKC] * B[KCx16]  (if accumulate=1)
//       or: C[6x16]  = A[6xKC] * B[KCx16]  (if accumulate=0)
//
// A is packed column-major: a[row + k*MR] (MR=6 contiguous rows per k)
// B is packed row-major:    b[col + k*NR] (NR=16 contiguous cols per k)
// C is row-major with stride ldc.
//
// The 4x-unrolled K-loop processes 4 K-iterations per pass.
// Software pipeline: prefetch A and B 256 bytes ahead.
#[cfg(target_arch = "x86_64")]
global_asm!(
    ".text",
    ".align 64",
    ".global _gllm_gemm_6x16_avx2_f32",
    ".type _gllm_gemm_6x16_avx2_f32, @function",
    "_gllm_gemm_6x16_avx2_f32:",
    // System V AMD64: rdi=packed_a, rsi=packed_b, rdx=c_ptr,
    //                 rcx=kc, r8=ldc, r9=accumulate
    // Callee-saved: rbx, rbp, r12-r15. YMM registers are caller-saved on System V.
    // We use r10-r15 for C row pointers, so save r12-r15.
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",

    // Convert ldc from elements to bytes: ldc_bytes = ldc * 4
    "shl r8, 2",

    // Compute C row pointers: rdx=row0, r10..r14=row1..row5
    "mov r10, rdx",
    "add r10, r8",          // row 1
    "lea r11, [r10 + r8]",  // row 2
    "lea r12, [r11 + r8]",  // row 3
    "lea r13, [r12 + r8]",  // row 4
    "lea r14, [r13 + r8]",  // row 5

    // Branch: accumulate or zero-init C accumulators
    "test r9, r9",
    "jz .Lavx2_zero_c",

    // ── Load existing C into accumulators ──
    "vmovups ymm0, [rdx]",
    "vmovups ymm1, [rdx + 32]",
    "vmovups ymm2, [r10]",
    "vmovups ymm3, [r10 + 32]",
    "vmovups ymm4, [r11]",
    "vmovups ymm5, [r11 + 32]",
    "vmovups ymm6, [r12]",
    "vmovups ymm7, [r12 + 32]",
    "vmovups ymm8, [r13]",
    "vmovups ymm9, [r13 + 32]",
    "vmovups ymm10, [r14]",
    "vmovups ymm11, [r14 + 32]",
    "jmp .Lavx2_k_setup",

    ".Lavx2_zero_c:",
    // Zero all 12 accumulator registers
    "vxorps ymm0, ymm0, ymm0",
    "vxorps ymm1, ymm1, ymm1",
    "vxorps ymm2, ymm2, ymm2",
    "vxorps ymm3, ymm3, ymm3",
    "vxorps ymm4, ymm4, ymm4",
    "vxorps ymm5, ymm5, ymm5",
    "vxorps ymm6, ymm6, ymm6",
    "vxorps ymm7, ymm7, ymm7",
    "vxorps ymm8, ymm8, ymm8",
    "vxorps ymm9, ymm9, ymm9",
    "vxorps ymm10, ymm10, ymm10",
    "vxorps ymm11, ymm11, ymm11",

    ".Lavx2_k_setup:",
    // r15 = kc (loop counter)
    "mov r15, rcx",

    // Check if kc >= 4 for unrolled loop
    "cmp r15, 4",
    "jl .Lavx2_k_remainder",

    // ── Main K-loop: 4x unrolled ──
    ".align 32",
    ".Lavx2_k_loop_4x:",

    // ---- k+0 ----
    // Load B[k+0]: 2 YMM vectors (16 floats = 64 bytes)
    "vmovups ymm13, [rsi]",
    "vmovups ymm14, [rsi + 32]",

    // Prefetch A and B 256 bytes ahead
    "prefetcht0 [rdi + 256]",
    "prefetcht0 [rsi + 256]",

    // Broadcast A[row, k+0] and FMA for each of 6 rows
    // A is packed column-major: a[row + k*MR], MR=6, so stride = 6*4 = 24 bytes per k
    // k+0: A starts at [rdi], elements at offsets 0,4,8,12,16,20
    "vbroadcastss ymm12, [rdi]",       // a[0,k+0]
    "vfmadd231ps ymm0, ymm12, ymm13",
    "vfmadd231ps ymm1, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 4]",   // a[1,k+0]
    "vfmadd231ps ymm2, ymm12, ymm13",
    "vfmadd231ps ymm3, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 8]",   // a[2,k+0]
    "vfmadd231ps ymm4, ymm12, ymm13",
    "vfmadd231ps ymm5, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 12]",  // a[3,k+0]
    "vfmadd231ps ymm6, ymm12, ymm13",
    "vfmadd231ps ymm7, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 16]",  // a[4,k+0]
    "vfmadd231ps ymm8, ymm12, ymm13",
    "vfmadd231ps ymm9, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 20]",  // a[5,k+0]
    "vfmadd231ps ymm10, ymm12, ymm13",
    "vfmadd231ps ymm11, ymm12, ymm14",

    // ---- k+1 ----
    // B[k+1] at [rsi + NR*4] = [rsi + 64]
    "vmovups ymm13, [rsi + 64]",
    "vmovups ymm14, [rsi + 96]",

    // A[k+1] at [rdi + MR*4] = [rdi + 24]
    "vbroadcastss ymm12, [rdi + 24]",
    "vfmadd231ps ymm0, ymm12, ymm13",
    "vfmadd231ps ymm1, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 28]",
    "vfmadd231ps ymm2, ymm12, ymm13",
    "vfmadd231ps ymm3, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 32]",
    "vfmadd231ps ymm4, ymm12, ymm13",
    "vfmadd231ps ymm5, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 36]",
    "vfmadd231ps ymm6, ymm12, ymm13",
    "vfmadd231ps ymm7, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 40]",
    "vfmadd231ps ymm8, ymm12, ymm13",
    "vfmadd231ps ymm9, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 44]",
    "vfmadd231ps ymm10, ymm12, ymm13",
    "vfmadd231ps ymm11, ymm12, ymm14",

    // ---- k+2 ----
    "vmovups ymm13, [rsi + 128]",
    "vmovups ymm14, [rsi + 160]",

    "prefetcht0 [rdi + 384]",
    "prefetcht0 [rsi + 384]",

    // A[k+2] at [rdi + 2*MR*4] = [rdi + 48]
    "vbroadcastss ymm12, [rdi + 48]",
    "vfmadd231ps ymm0, ymm12, ymm13",
    "vfmadd231ps ymm1, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 52]",
    "vfmadd231ps ymm2, ymm12, ymm13",
    "vfmadd231ps ymm3, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 56]",
    "vfmadd231ps ymm4, ymm12, ymm13",
    "vfmadd231ps ymm5, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 60]",
    "vfmadd231ps ymm6, ymm12, ymm13",
    "vfmadd231ps ymm7, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 64]",
    "vfmadd231ps ymm8, ymm12, ymm13",
    "vfmadd231ps ymm9, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 68]",
    "vfmadd231ps ymm10, ymm12, ymm13",
    "vfmadd231ps ymm11, ymm12, ymm14",

    // ---- k+3 ----
    "vmovups ymm13, [rsi + 192]",
    "vmovups ymm14, [rsi + 224]",

    // A[k+3] at [rdi + 3*MR*4] = [rdi + 72]
    "vbroadcastss ymm12, [rdi + 72]",
    "vfmadd231ps ymm0, ymm12, ymm13",
    "vfmadd231ps ymm1, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 76]",
    "vfmadd231ps ymm2, ymm12, ymm13",
    "vfmadd231ps ymm3, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 80]",
    "vfmadd231ps ymm4, ymm12, ymm13",
    "vfmadd231ps ymm5, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 84]",
    "vfmadd231ps ymm6, ymm12, ymm13",
    "vfmadd231ps ymm7, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 88]",
    "vfmadd231ps ymm8, ymm12, ymm13",
    "vfmadd231ps ymm9, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 92]",
    "vfmadd231ps ymm10, ymm12, ymm13",
    "vfmadd231ps ymm11, ymm12, ymm14",

    // Advance pointers: A += 4*MR*4 = 96 bytes, B += 4*NR*4 = 256 bytes
    "add rdi, 96",
    "add rsi, 256",

    "sub r15, 4",
    "cmp r15, 4",
    "jge .Lavx2_k_loop_4x",

    // ── K-remainder loop (1x) ──
    ".Lavx2_k_remainder:",
    "test r15, r15",
    "jz .Lavx2_store_c",

    ".Lavx2_k_tail:",
    // Load B[k]: 2 YMM vectors
    "vmovups ymm13, [rsi]",
    "vmovups ymm14, [rsi + 32]",

    // Broadcast A[row, k] and FMA
    "vbroadcastss ymm12, [rdi]",
    "vfmadd231ps ymm0, ymm12, ymm13",
    "vfmadd231ps ymm1, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 4]",
    "vfmadd231ps ymm2, ymm12, ymm13",
    "vfmadd231ps ymm3, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 8]",
    "vfmadd231ps ymm4, ymm12, ymm13",
    "vfmadd231ps ymm5, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 12]",
    "vfmadd231ps ymm6, ymm12, ymm13",
    "vfmadd231ps ymm7, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 16]",
    "vfmadd231ps ymm8, ymm12, ymm13",
    "vfmadd231ps ymm9, ymm12, ymm14",
    "vbroadcastss ymm12, [rdi + 20]",
    "vfmadd231ps ymm10, ymm12, ymm13",
    "vfmadd231ps ymm11, ymm12, ymm14",

    // Advance: A += MR*4 = 24 bytes, B += NR*4 = 64 bytes
    "add rdi, 24",
    "add rsi, 64",
    "dec r15",
    "jnz .Lavx2_k_tail",

    // ── Store C accumulators ──
    ".Lavx2_store_c:",
    "vmovups [rdx], ymm0",
    "vmovups [rdx + 32], ymm1",
    "vmovups [r10], ymm2",
    "vmovups [r10 + 32], ymm3",
    "vmovups [r11], ymm4",
    "vmovups [r11 + 32], ymm5",
    "vmovups [r12], ymm6",
    "vmovups [r12 + 32], ymm7",
    "vmovups [r13], ymm8",
    "vmovups [r13 + 32], ymm9",
    "vmovups [r14], ymm10",
    "vmovups [r14 + 32], ymm11",

    // Clean up AVX state
    "vzeroupper",

    // Restore callee-saved registers
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "ret",

    ".size _gllm_gemm_6x16_avx2_f32, . - _gllm_gemm_6x16_avx2_f32",
);

#[cfg(target_arch = "x86_64")]
extern "C" {
    /// Raw assembly entry point for the 6x16 f32 AVX2 GEMM microkernel.
    ///
    /// # Safety
    /// - All pointers must be valid and properly aligned (32-byte for AVX2)
    /// - packed_a must contain at least kc * MR f32 elements
    /// - packed_b must contain at least kc * NR f32 elements
    /// - c_ptr must point to a buffer with at least MR rows of ldc elements each
    /// - kc must be > 0
    fn _gllm_gemm_6x16_avx2_f32(
        packed_a: *const f32,
        packed_b: *const f32,
        c_ptr: *mut f32,
        kc: usize,
        ldc: usize,
        accumulate: usize,
    );
}

/// Safe Rust wrapper for the 6x16 AVX2 GEMM microkernel.
///
/// Computes: C[6x16] += A[6xKC] * B[KCx16]  (if accumulate=true)
///       or: C[6x16]  = A[6xKC] * B[KCx16]  (if accumulate=false)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn gemm_kernel_6x16_f32(
    packed_a: *const f32,
    packed_b: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    ldc: usize,
    accumulate: bool,
) {
    _gllm_gemm_6x16_avx2_f32(packed_a, packed_b, c_ptr, kc, ldc, accumulate as usize);
}
