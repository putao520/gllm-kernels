//! Hand-written AVX-512 GEMM 14x32 microkernel (x86_64).
//!
//! Phase 5 rewrite: 4x unrolled K-loop with dual-distance prefetch
//! and C row prefetch before stores.
//!
//! Microkernel geometry: 14 rows x 32 columns (2 x ZMM vectors per row)
//! = 28 accumulator registers (zmm0-zmm27)
//!
//! Register allocation (32 ZMM registers):
//!   zmm0  - zmm1  : C row 0
//!   zmm2  - zmm3  : C row 1
//!   zmm4  - zmm5  : C row 2
//!   zmm6  - zmm7  : C row 3
//!   zmm8  - zmm9  : C row 4
//!   zmm10 - zmm11 : C row 5
//!   zmm12 - zmm13 : C row 6
//!   zmm14 - zmm15 : C row 7
//!   zmm16 - zmm17 : C row 8
//!   zmm18 - zmm19 : C row 9
//!   zmm20 - zmm21 : C row 10
//!   zmm22 - zmm23 : C row 11
//!   zmm24 - zmm25 : C row 12
//!   zmm26 - zmm27 : C row 13
//!   zmm28 - zmm29 : A broadcast temporaries
//!   zmm30 - zmm31 : B load double-buffer (software pipeline)
//!
//! K-loop: 4x unrolled main loop, 2x remainder, 1x tail.
//! Prefetch: dual-distance L1 (prefetcht0) + L2 (prefetcht1) schedule.
//!
//! Calling convention (System V AMD64 ABI):
//!   rdi = packed_a, rsi = packed_b, rdx = c_ptr,
//!   rcx = kc, r8 = ldc (elements), r9 = accumulate

#[cfg(target_arch = "x86_64")]
use std::arch::global_asm;

pub const MR: usize = 14;
pub const NR: usize = 32;

#[cfg(target_arch = "x86_64")]
global_asm!(
    ".text",
    ".align 64",
    ".global _gllm_gemm_14x32_avx512_f32",
    ".type _gllm_gemm_14x32_avx512_f32, @function",
    "_gllm_gemm_14x32_avx512_f32:",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "push rbp",
    // Stack: 8 pointers for C rows 6-13
    "sub rsp, 64",
    // ldc: elements -> bytes
    "shl r8, 2",
    // C row pointers: row0=rdx, row1-5 in r10-r14
    "mov r10, rdx",
    "add r10, r8",
    "lea r11, [r10 + r8]",
    "lea r12, [r11 + r8]",
    "lea r13, [r12 + r8]",
    "lea r14, [r13 + r8]",
    // rows 6-13 on stack
    "lea rbx, [r14 + r8]",
    "mov [rsp], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 8], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 16], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 24], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 32], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 40], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 48], rbx",
    "lea rbx, [rbx + r8]",
    "mov [rsp + 56], rbx",

    "test r9, r9",
    "jz .Lavx512_zero_c",

    // -- Load existing C --
    "vmovups zmm0, [rdx]",
    "vmovups zmm1, [rdx + 64]",
    "vmovups zmm2, [r10]",
    "vmovups zmm3, [r10 + 64]",
    "vmovups zmm4, [r11]",
    "vmovups zmm5, [r11 + 64]",
    "vmovups zmm6, [r12]",
    "vmovups zmm7, [r12 + 64]",
    "vmovups zmm8, [r13]",
    "vmovups zmm9, [r13 + 64]",
    "vmovups zmm10, [r14]",
    "vmovups zmm11, [r14 + 64]",
    "mov rbx, [rsp + 0]",
    "vmovups zmm12, [rbx]",
    "vmovups zmm13, [rbx + 64]",
    "mov rbx, [rsp + 8]",
    "vmovups zmm14, [rbx]",
    "vmovups zmm15, [rbx + 64]",
    "mov rbx, [rsp + 16]",
    "vmovups zmm16, [rbx]",
    "vmovups zmm17, [rbx + 64]",
    "mov rbx, [rsp + 24]",
    "vmovups zmm18, [rbx]",
    "vmovups zmm19, [rbx + 64]",
    "mov rbx, [rsp + 32]",
    "vmovups zmm20, [rbx]",
    "vmovups zmm21, [rbx + 64]",
    "mov rbx, [rsp + 40]",
    "vmovups zmm22, [rbx]",
    "vmovups zmm23, [rbx + 64]",
    "mov rbx, [rsp + 48]",
    "vmovups zmm24, [rbx]",
    "vmovups zmm25, [rbx + 64]",
    "mov rbx, [rsp + 56]",
    "vmovups zmm26, [rbx]",
    "vmovups zmm27, [rbx + 64]",
    "jmp .Lavx512_k_setup",

    ".Lavx512_zero_c:",
    "vpxord zmm0, zmm0, zmm0",
    "vpxord zmm1, zmm1, zmm1",
    "vpxord zmm2, zmm2, zmm2",
    "vpxord zmm3, zmm3, zmm3",
    "vpxord zmm4, zmm4, zmm4",
    "vpxord zmm5, zmm5, zmm5",
    "vpxord zmm6, zmm6, zmm6",
    "vpxord zmm7, zmm7, zmm7",
    "vpxord zmm8, zmm8, zmm8",
    "vpxord zmm9, zmm9, zmm9",
    "vpxord zmm10, zmm10, zmm10",
    "vpxord zmm11, zmm11, zmm11",
    "vpxord zmm12, zmm12, zmm12",
    "vpxord zmm13, zmm13, zmm13",
    "vpxord zmm14, zmm14, zmm14",
    "vpxord zmm15, zmm15, zmm15",
    "vpxord zmm16, zmm16, zmm16",
    "vpxord zmm17, zmm17, zmm17",
    "vpxord zmm18, zmm18, zmm18",
    "vpxord zmm19, zmm19, zmm19",
    "vpxord zmm20, zmm20, zmm20",
    "vpxord zmm21, zmm21, zmm21",
    "vpxord zmm22, zmm22, zmm22",
    "vpxord zmm23, zmm23, zmm23",
    "vpxord zmm24, zmm24, zmm24",
    "vpxord zmm25, zmm25, zmm25",
    "vpxord zmm26, zmm26, zmm26",
    "vpxord zmm27, zmm27, zmm27",

    ".Lavx512_k_setup:",
    "mov r15, rcx",
    "cmp r15, 4",
    "jl .Lavx512_k_check_2x",

    // Pre-load B[k=0]
    "vmovups zmm30, [rsi]",
    "vmovups zmm31, [rsi + 64]",

    // -- Main K-loop: 4x unrolled with software pipeline --
    ".align 64",
    ".Lavx512_k_loop_4x:",
    // ---- k+0 with L1 prefetch ---- B in zmm30/zmm31
    "prefetcht0 [rdi + 256]",
    "prefetcht0 [rsi + 512]",

    "vbroadcastss zmm28, [rdi + 0]",
    "vbroadcastss zmm29, [rdi + 4]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 8]",
    "vbroadcastss zmm29, [rdi + 12]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 16]",
    "vbroadcastss zmm29, [rdi + 20]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 24]",
    "vbroadcastss zmm29, [rdi + 28]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 32]",
    "vbroadcastss zmm29, [rdi + 36]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 40]",
    "vbroadcastss zmm29, [rdi + 44]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 48]",
    "vbroadcastss zmm29, [rdi + 52]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    // Software pipeline: load B[k+1]
    "vmovups zmm30, [rsi + 128]",
    "vmovups zmm31, [rsi + 192]",

    // ---- k+1 with L2 prefetch ---- B in zmm30/zmm31
    "prefetcht1 [rdi + 768]",
    "prefetcht1 [rsi + 1536]",

    "vbroadcastss zmm28, [rdi + 56]",
    "vbroadcastss zmm29, [rdi + 60]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 64]",
    "vbroadcastss zmm29, [rdi + 68]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 72]",
    "vbroadcastss zmm29, [rdi + 76]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 80]",
    "vbroadcastss zmm29, [rdi + 84]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 88]",
    "vbroadcastss zmm29, [rdi + 92]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 96]",
    "vbroadcastss zmm29, [rdi + 100]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 104]",
    "vbroadcastss zmm29, [rdi + 108]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    // Software pipeline: load B[k+2]
    "vmovups zmm30, [rsi + 256]",
    "vmovups zmm31, [rsi + 320]",

    // ---- k+2 with L1 prefetch (second distance) ---- B in zmm30/zmm31
    "prefetcht0 [rdi + 448]",
    "prefetcht0 [rsi + 768]",

    "vbroadcastss zmm28, [rdi + 112]",
    "vbroadcastss zmm29, [rdi + 116]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 120]",
    "vbroadcastss zmm29, [rdi + 124]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 128]",
    "vbroadcastss zmm29, [rdi + 132]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 136]",
    "vbroadcastss zmm29, [rdi + 140]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 144]",
    "vbroadcastss zmm29, [rdi + 148]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 152]",
    "vbroadcastss zmm29, [rdi + 156]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 160]",
    "vbroadcastss zmm29, [rdi + 164]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    // Software pipeline: load B[k+3]
    "vmovups zmm30, [rsi + 384]",
    "vmovups zmm31, [rsi + 448]",

    // ---- k+3 (no prefetch) ---- B in zmm30/zmm31

    "vbroadcastss zmm28, [rdi + 168]",
    "vbroadcastss zmm29, [rdi + 172]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 176]",
    "vbroadcastss zmm29, [rdi + 180]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 184]",
    "vbroadcastss zmm29, [rdi + 188]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 192]",
    "vbroadcastss zmm29, [rdi + 196]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 200]",
    "vbroadcastss zmm29, [rdi + 204]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 208]",
    "vbroadcastss zmm29, [rdi + 212]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 216]",
    "vbroadcastss zmm29, [rdi + 220]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    // Advance: A += 4*MR*4 = 224, B += 4*NR*4 = 512
    "add rdi, 224",
    "add rsi, 512",

    "sub r15, 4",
    "cmp r15, 4",
    "jl .Lavx512_k_check_2x",

    // Pre-load B for next 4x iteration
    "vmovups zmm30, [rsi]",
    "vmovups zmm31, [rsi + 64]",
    "jmp .Lavx512_k_loop_4x",

    // -- K-remainder (2x) --
    ".Lavx512_k_check_2x:",
    "cmp r15, 2",
    "jl .Lavx512_k_remainder",

    "vmovups zmm30, [rsi]",
    "vmovups zmm31, [rsi + 64]",

    // ---- 2x remainder: k+0 ---- B in zmm30/zmm31

    "vbroadcastss zmm28, [rdi + 0]",
    "vbroadcastss zmm29, [rdi + 4]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 8]",
    "vbroadcastss zmm29, [rdi + 12]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 16]",
    "vbroadcastss zmm29, [rdi + 20]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 24]",
    "vbroadcastss zmm29, [rdi + 28]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 32]",
    "vbroadcastss zmm29, [rdi + 36]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 40]",
    "vbroadcastss zmm29, [rdi + 44]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 48]",
    "vbroadcastss zmm29, [rdi + 52]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    // Software pipeline: load B[k+1]
    "vmovups zmm30, [rsi + 128]",
    "vmovups zmm31, [rsi + 192]",

    // ---- 2x remainder: k+1 ---- B in zmm30/zmm31

    "vbroadcastss zmm28, [rdi + 56]",
    "vbroadcastss zmm29, [rdi + 60]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 64]",
    "vbroadcastss zmm29, [rdi + 68]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 72]",
    "vbroadcastss zmm29, [rdi + 76]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 80]",
    "vbroadcastss zmm29, [rdi + 84]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 88]",
    "vbroadcastss zmm29, [rdi + 92]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 96]",
    "vbroadcastss zmm29, [rdi + 100]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 104]",
    "vbroadcastss zmm29, [rdi + 108]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    "add rdi, 112",
    "add rsi, 256",
    "sub r15, 2",

    // -- K-remainder (1x) --
    ".Lavx512_k_remainder:",
    "test r15, r15",
    "jz .Lavx512_prefetch_c",

    ".Lavx512_k_tail:",
    "vmovups zmm30, [rsi]",
    "vmovups zmm31, [rsi + 64]",

    // ---- 1x tail ---- B in zmm30/zmm31

    "vbroadcastss zmm28, [rdi + 0]",
    "vbroadcastss zmm29, [rdi + 4]",
    "vfmadd231ps zmm0, zmm28, zmm30",
    "vfmadd231ps zmm1, zmm28, zmm31",
    "vfmadd231ps zmm2, zmm29, zmm30",
    "vfmadd231ps zmm3, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 8]",
    "vbroadcastss zmm29, [rdi + 12]",
    "vfmadd231ps zmm4, zmm28, zmm30",
    "vfmadd231ps zmm5, zmm28, zmm31",
    "vfmadd231ps zmm6, zmm29, zmm30",
    "vfmadd231ps zmm7, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 16]",
    "vbroadcastss zmm29, [rdi + 20]",
    "vfmadd231ps zmm8, zmm28, zmm30",
    "vfmadd231ps zmm9, zmm28, zmm31",
    "vfmadd231ps zmm10, zmm29, zmm30",
    "vfmadd231ps zmm11, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 24]",
    "vbroadcastss zmm29, [rdi + 28]",
    "vfmadd231ps zmm12, zmm28, zmm30",
    "vfmadd231ps zmm13, zmm28, zmm31",
    "vfmadd231ps zmm14, zmm29, zmm30",
    "vfmadd231ps zmm15, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 32]",
    "vbroadcastss zmm29, [rdi + 36]",
    "vfmadd231ps zmm16, zmm28, zmm30",
    "vfmadd231ps zmm17, zmm28, zmm31",
    "vfmadd231ps zmm18, zmm29, zmm30",
    "vfmadd231ps zmm19, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 40]",
    "vbroadcastss zmm29, [rdi + 44]",
    "vfmadd231ps zmm20, zmm28, zmm30",
    "vfmadd231ps zmm21, zmm28, zmm31",
    "vfmadd231ps zmm22, zmm29, zmm30",
    "vfmadd231ps zmm23, zmm29, zmm31",

    "vbroadcastss zmm28, [rdi + 48]",
    "vbroadcastss zmm29, [rdi + 52]",
    "vfmadd231ps zmm24, zmm28, zmm30",
    "vfmadd231ps zmm25, zmm28, zmm31",
    "vfmadd231ps zmm26, zmm29, zmm30",
    "vfmadd231ps zmm27, zmm29, zmm31",

    "add rdi, 56",
    "add rsi, 128",
    "dec r15",
    "jnz .Lavx512_k_tail",

    // -- Prefetch C rows before store (reduce RFO latency) --
    ".Lavx512_prefetch_c:",
    "prefetcht0 [rdx]",
    "prefetcht0 [rdx + 64]",
    "prefetcht0 [r10]",
    "prefetcht0 [r10 + 64]",
    "prefetcht0 [r11]",
    "prefetcht0 [r11 + 64]",
    "prefetcht0 [r12]",
    "prefetcht0 [r12 + 64]",
    "prefetcht0 [r13]",
    "prefetcht0 [r13 + 64]",
    "prefetcht0 [r14]",
    "prefetcht0 [r14 + 64]",
    "mov rbx, [rsp + 0]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 8]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 16]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 24]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 32]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 40]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 48]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",
    "mov rbx, [rsp + 56]",
    "prefetcht0 [rbx]",
    "prefetcht0 [rbx + 64]",

    // -- Store C --
    ".Lavx512_store_c:",

    // ── Fuse bias if bias pointer (7th arg) is non-null ──
    // Stack layout: sub rsp 64 + 6 pushes (48) + return addr (8) = 120
    "mov rax, [rsp + 120]",
    "test rax, rax",
    "jz .Lavx512_store_no_bias",

    // Load bias vector (32 floats = 2 zmm)
    "vmovups zmm28, [rax]",
    "vmovups zmm29, [rax + 64]",

    // Add bias to all 14 rows
    "vaddps zmm0, zmm0, zmm28",
    "vaddps zmm1, zmm1, zmm29",
    "vaddps zmm2, zmm2, zmm28",
    "vaddps zmm3, zmm3, zmm29",
    "vaddps zmm4, zmm4, zmm28",
    "vaddps zmm5, zmm5, zmm29",
    "vaddps zmm6, zmm6, zmm28",
    "vaddps zmm7, zmm7, zmm29",
    "vaddps zmm8, zmm8, zmm28",
    "vaddps zmm9, zmm9, zmm29",
    "vaddps zmm10, zmm10, zmm28",
    "vaddps zmm11, zmm11, zmm29",
    "vaddps zmm12, zmm12, zmm28",
    "vaddps zmm13, zmm13, zmm29",
    "vaddps zmm14, zmm14, zmm28",
    "vaddps zmm15, zmm15, zmm29",
    "vaddps zmm16, zmm16, zmm28",
    "vaddps zmm17, zmm17, zmm29",
    "vaddps zmm18, zmm18, zmm28",
    "vaddps zmm19, zmm19, zmm29",
    "vaddps zmm20, zmm20, zmm28",
    "vaddps zmm21, zmm21, zmm29",
    "vaddps zmm22, zmm22, zmm28",
    "vaddps zmm23, zmm23, zmm29",
    "vaddps zmm24, zmm24, zmm28",
    "vaddps zmm25, zmm25, zmm29",
    "vaddps zmm26, zmm26, zmm28",
    "vaddps zmm27, zmm27, zmm29",

    ".Lavx512_store_no_bias:",

    "vmovups [rdx], zmm0",
    "vmovups [rdx + 64], zmm1",
    "vmovups [r10], zmm2",
    "vmovups [r10 + 64], zmm3",
    "vmovups [r11], zmm4",
    "vmovups [r11 + 64], zmm5",
    "vmovups [r12], zmm6",
    "vmovups [r12 + 64], zmm7",
    "vmovups [r13], zmm8",
    "vmovups [r13 + 64], zmm9",
    "vmovups [r14], zmm10",
    "vmovups [r14 + 64], zmm11",
    "mov rbx, [rsp + 0]",
    "vmovups [rbx], zmm12",
    "vmovups [rbx + 64], zmm13",
    "mov rbx, [rsp + 8]",
    "vmovups [rbx], zmm14",
    "vmovups [rbx + 64], zmm15",
    "mov rbx, [rsp + 16]",
    "vmovups [rbx], zmm16",
    "vmovups [rbx + 64], zmm17",
    "mov rbx, [rsp + 24]",
    "vmovups [rbx], zmm18",
    "vmovups [rbx + 64], zmm19",
    "mov rbx, [rsp + 32]",
    "vmovups [rbx], zmm20",
    "vmovups [rbx + 64], zmm21",
    "mov rbx, [rsp + 40]",
    "vmovups [rbx], zmm22",
    "vmovups [rbx + 64], zmm23",
    "mov rbx, [rsp + 48]",
    "vmovups [rbx], zmm24",
    "vmovups [rbx + 64], zmm25",
    "mov rbx, [rsp + 56]",
    "vmovups [rbx], zmm26",
    "vmovups [rbx + 64], zmm27",

    "add rsp, 64",
    "vzeroupper",
    "pop rbp",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "ret",

    ".size _gllm_gemm_14x32_avx512_f32, . - _gllm_gemm_14x32_avx512_f32",
);

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn _gllm_gemm_14x32_avx512_f32(
        packed_a: *const f32,
        packed_b: *const f32,
        c_ptr: *mut f32,
        kc: usize,
        ldc: usize,
        accumulate: usize,
        bias: *const f32,
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn gemm_kernel_14x32_f32(
    packed_a: *const f32,
    packed_b: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    ldc: usize,
    accumulate: bool,
) {
    _gllm_gemm_14x32_avx512_f32(packed_a, packed_b, c_ptr, kc, ldc, accumulate as usize, std::ptr::null());
}

/// AVX-512 GEMM microkernel with fused bias addition.
/// `bias` must point to NR (32) floats for the current tile columns.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn gemm_kernel_14x32_f32_bias(
    packed_a: *const f32,
    packed_b: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    ldc: usize,
    accumulate: bool,
    bias: *const f32,
) {
    _gllm_gemm_14x32_avx512_f32(packed_a, packed_b, c_ptr, kc, ldc, accumulate as usize, bias);
}
