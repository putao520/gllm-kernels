//! Hand-written AVX2 GEMM 6x16 microkernel (x86_64).
//!
//! Phase 4 rewrite: BLIS double-buffer pattern.
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
//!   ymm12 - ymm13 : A broadcast double-buffer (alternating, eliminates false dep)
//!   ymm14 - ymm15 : B load double-buffer (software pipeline: load B[k+1] during FMA[k])
//!
//! Key optimizations vs Phase 3:
//! - A broadcast uses TWO registers (ymm12/ymm13) alternating per row-pair,
//!   breaking the 6-deep WAW dependency chain on a single register.
//! - B vectors pre-loaded before loop; each k-step loads B[k+1] at the end,
//!   overlapping the load latency (~5 cycles) with the last FMAs of step k.
//! - Prefetch distances tuned for typical L1D (32-48KB).
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

#[cfg(target_arch = "x86_64")]
global_asm!(
    ".text",
    ".align 64",
    ".global _gllm_gemm_6x16_avx2_f32",
    ".type _gllm_gemm_6x16_avx2_f32, @function",
    "_gllm_gemm_6x16_avx2_f32:",
    // Save callee-saved registers
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",

    // Convert ldc from elements to bytes
    "shl r8, 2",

    // Compute C row pointers: rdx=row0, r10..r14=row1..row5
    "mov r10, rdx",
    "add r10, r8",
    "lea r11, [r10 + r8]",
    "lea r12, [r11 + r8]",
    "lea r13, [r12 + r8]",
    "lea r14, [r13 + r8]",

    // Branch: accumulate or zero-init
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
    "mov r15, rcx",
    "cmp r15, 4",
    "jl .Lavx2_k_remainder",

    // ── Pre-load B[k=0] for software pipeline ──
    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",

    // ── Main K-loop: 4x unrolled with double-buffer ──
    ".align 32",
    ".Lavx2_k_loop_4x:",

    // ──── k+0 ──── B[k+0] already in ymm14/ymm15
    "prefetcht0 [rdi + 192]",       // prefetch A ~8 k-steps ahead
    "prefetcht0 [rsi + 384]",       // prefetch B ~6 k-steps ahead

    "vbroadcastss ymm12, [rdi]",       // A[0,k]
    "vbroadcastss ymm13, [rdi + 4]",   // A[1,k]
    "vfmadd231ps ymm0, ymm12, ymm14",
    "vfmadd231ps ymm1, ymm12, ymm15",
    "vfmadd231ps ymm2, ymm13, ymm14",
    "vfmadd231ps ymm3, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 8]",   // A[2,k]
    "vbroadcastss ymm13, [rdi + 12]",  // A[3,k]
    "vfmadd231ps ymm4, ymm12, ymm14",
    "vfmadd231ps ymm5, ymm12, ymm15",
    "vfmadd231ps ymm6, ymm13, ymm14",
    "vfmadd231ps ymm7, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 16]",  // A[4,k]
    "vbroadcastss ymm13, [rdi + 20]",  // A[5,k]
    "vfmadd231ps ymm8, ymm12, ymm14",
    "vfmadd231ps ymm9, ymm12, ymm15",
    "vfmadd231ps ymm10, ymm13, ymm14",
    "vfmadd231ps ymm11, ymm13, ymm15",

    // Software pipeline: load B[k+1]
    "vmovups ymm14, [rsi + 64]",
    "vmovups ymm15, [rsi + 96]",

    // ──── k+1 ──── B[k+1] in ymm14/ymm15
    "vbroadcastss ymm12, [rdi + 24]",
    "vbroadcastss ymm13, [rdi + 28]",
    "vfmadd231ps ymm0, ymm12, ymm14",
    "vfmadd231ps ymm1, ymm12, ymm15",
    "vfmadd231ps ymm2, ymm13, ymm14",
    "vfmadd231ps ymm3, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 32]",
    "vbroadcastss ymm13, [rdi + 36]",
    "vfmadd231ps ymm4, ymm12, ymm14",
    "vfmadd231ps ymm5, ymm12, ymm15",
    "vfmadd231ps ymm6, ymm13, ymm14",
    "vfmadd231ps ymm7, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 40]",
    "vbroadcastss ymm13, [rdi + 44]",
    "vfmadd231ps ymm8, ymm12, ymm14",
    "vfmadd231ps ymm9, ymm12, ymm15",
    "vfmadd231ps ymm10, ymm13, ymm14",
    "vfmadd231ps ymm11, ymm13, ymm15",

    // Load B[k+2]
    "vmovups ymm14, [rsi + 128]",
    "vmovups ymm15, [rsi + 160]",

    // ──── k+2 ────
    "prefetcht0 [rdi + 384]",
    "prefetcht0 [rsi + 512]",

    "vbroadcastss ymm12, [rdi + 48]",
    "vbroadcastss ymm13, [rdi + 52]",
    "vfmadd231ps ymm0, ymm12, ymm14",
    "vfmadd231ps ymm1, ymm12, ymm15",
    "vfmadd231ps ymm2, ymm13, ymm14",
    "vfmadd231ps ymm3, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 56]",
    "vbroadcastss ymm13, [rdi + 60]",
    "vfmadd231ps ymm4, ymm12, ymm14",
    "vfmadd231ps ymm5, ymm12, ymm15",
    "vfmadd231ps ymm6, ymm13, ymm14",
    "vfmadd231ps ymm7, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 64]",
    "vbroadcastss ymm13, [rdi + 68]",
    "vfmadd231ps ymm8, ymm12, ymm14",
    "vfmadd231ps ymm9, ymm12, ymm15",
    "vfmadd231ps ymm10, ymm13, ymm14",
    "vfmadd231ps ymm11, ymm13, ymm15",

    // Load B[k+3]
    "vmovups ymm14, [rsi + 192]",
    "vmovups ymm15, [rsi + 224]",

    // ──── k+3 ────
    "vbroadcastss ymm12, [rdi + 72]",
    "vbroadcastss ymm13, [rdi + 76]",
    "vfmadd231ps ymm0, ymm12, ymm14",
    "vfmadd231ps ymm1, ymm12, ymm15",
    "vfmadd231ps ymm2, ymm13, ymm14",
    "vfmadd231ps ymm3, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 80]",
    "vbroadcastss ymm13, [rdi + 84]",
    "vfmadd231ps ymm4, ymm12, ymm14",
    "vfmadd231ps ymm5, ymm12, ymm15",
    "vfmadd231ps ymm6, ymm13, ymm14",
    "vfmadd231ps ymm7, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 88]",
    "vbroadcastss ymm13, [rdi + 92]",
    "vfmadd231ps ymm8, ymm12, ymm14",
    "vfmadd231ps ymm9, ymm12, ymm15",
    "vfmadd231ps ymm10, ymm13, ymm14",
    "vfmadd231ps ymm11, ymm13, ymm15",

    // Advance pointers: A += 4*MR*4 = 96, B += 4*NR*4 = 256
    "add rdi, 96",
    "add rsi, 256",

    "sub r15, 4",
    "cmp r15, 4",
    "jl .Lavx2_k_remainder",

    // Pre-load B for next 4x iteration
    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",
    "jmp .Lavx2_k_loop_4x",

    // ── K-remainder loop (1x) ──
    ".Lavx2_k_remainder:",
    "test r15, r15",
    "jz .Lavx2_store_c",

    ".Lavx2_k_tail:",
    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",

    "vbroadcastss ymm12, [rdi]",
    "vbroadcastss ymm13, [rdi + 4]",
    "vfmadd231ps ymm0, ymm12, ymm14",
    "vfmadd231ps ymm1, ymm12, ymm15",
    "vfmadd231ps ymm2, ymm13, ymm14",
    "vfmadd231ps ymm3, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 8]",
    "vbroadcastss ymm13, [rdi + 12]",
    "vfmadd231ps ymm4, ymm12, ymm14",
    "vfmadd231ps ymm5, ymm12, ymm15",
    "vfmadd231ps ymm6, ymm13, ymm14",
    "vfmadd231ps ymm7, ymm13, ymm15",

    "vbroadcastss ymm12, [rdi + 16]",
    "vbroadcastss ymm13, [rdi + 20]",
    "vfmadd231ps ymm8, ymm12, ymm14",
    "vfmadd231ps ymm9, ymm12, ymm15",
    "vfmadd231ps ymm10, ymm13, ymm14",
    "vfmadd231ps ymm11, ymm13, ymm15",

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

    "vzeroupper",

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
