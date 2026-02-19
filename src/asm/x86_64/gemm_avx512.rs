//! Hand-written AVX-512 GEMM 14x32 microkernel (x86_64).
//!
//! Microkernel geometry: 14 rows x 32 columns (2 x ZMM vectors per row)
//! = 28 accumulator registers (zmm0-zmm27)
//!
//! Register allocation (32 ZMM registers):
//!   zmm0  - zmm1  : C row 0  (2 vectors x 16 lanes = 32 columns)
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
//!   zmm28          : A broadcast (current element)
//!   zmm29          : B vector 0 (current k)
//!   zmm30          : B vector 1 (current k)
//!   zmm31          : temp
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
pub const MR: usize = 14;
pub const NR: usize = 32;

// ── AVX-512 14x32 f32 GEMM microkernel ─────────────────────────────────
//
// Computes: C[14x32] += A[14xKC] * B[KCx32]  (if accumulate=1)
//       or: C[14x32]  = A[14xKC] * B[KCx32]  (if accumulate=0)
//
// A is packed column-major: a[row + k*MR] (MR=14 contiguous rows per k)
// B is packed row-major:    b[col + k*NR] (NR=32 contiguous cols per k)
// C is row-major with stride ldc.
//
// A stride per k = MR * 4 = 56 bytes
// B stride per k = NR * 4 = 128 bytes
//
// We need 14 C row pointers. We use: rdx(row0), r10-r14(row1-5),
// and spill row6-13 pointers to the stack.
#[cfg(target_arch = "x86_64")]
global_asm!(
    ".text",
    ".align 64",
    ".global _gllm_gemm_14x32_avx512_f32",
    ".type _gllm_gemm_14x32_avx512_f32, @function",
    "_gllm_gemm_14x32_avx512_f32:",
    // System V AMD64: rdi=packed_a, rsi=packed_b, rdx=c_ptr,
    //                 rcx=kc, r8=ldc, r9=accumulate
    // Save callee-saved registers
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "push rbp",

    // Allocate stack space for C row pointers (rows 6-13 = 8 pointers = 64 bytes)
    "sub rsp, 64",

    // Convert ldc from elements to bytes
    "shl r8, 2",

    // Compute all 14 C row pointers
    // row0 = rdx (kept in rdx)
    // row1-5 in r10-r14
    "mov r10, rdx",
    "add r10, r8",           // row 1
    "lea r11, [r10 + r8]",   // row 2
    "lea r12, [r11 + r8]",   // row 3
    "lea r13, [r12 + r8]",   // row 4
    "lea r14, [r13 + r8]",   // row 5

    // rows 6-13 on stack: [rsp+0]..[rsp+56]
    "lea rbx, [r14 + r8]",   // row 6
    "mov [rsp], rbx",
    "lea rbx, [rbx + r8]",   // row 7
    "mov [rsp + 8], rbx",
    "lea rbx, [rbx + r8]",   // row 8
    "mov [rsp + 16], rbx",
    "lea rbx, [rbx + r8]",   // row 9
    "mov [rsp + 24], rbx",
    "lea rbx, [rbx + r8]",   // row 10
    "mov [rsp + 32], rbx",
    "lea rbx, [rbx + r8]",   // row 11
    "mov [rsp + 40], rbx",
    "lea rbx, [rbx + r8]",   // row 12
    "mov [rsp + 48], rbx",
    "lea rbx, [rbx + r8]",   // row 13
    "mov [rsp + 56], rbx",

    // Branch: accumulate or zero-init
    "test r9, r9",
    "jz .Lavx512_zero_c",

    // ── Load existing C into accumulators ──
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
    // rows 6-13 from stack pointers
    "mov rbx, [rsp]",
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
    // Zero all 28 accumulator registers
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
    "jl .Lavx512_k_remainder",

    // ── Main K-loop: 4x unrolled ──
    // A stride per k = MR * 4 = 56 bytes
    // B stride per k = NR * 4 = 128 bytes
    ".align 32",
    ".Lavx512_k_loop_4x:",

    // ---- k+0 ----
    // Load B[k+0]: 2 ZMM vectors (32 floats = 128 bytes)
    "vmovups zmm29, [rsi]",
    "vmovups zmm30, [rsi + 64]",

    "prefetcht0 [rdi + 256]",
    "prefetcht0 [rsi + 512]",

    // Broadcast A[row, k+0] and FMA for 14 rows
    // A offsets: row_i at [rdi + i*4], i=0..13
    "vbroadcastss zmm28, [rdi]",
    "vfmadd231ps zmm0, zmm28, zmm29",
    "vfmadd231ps zmm1, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 4]",
    "vfmadd231ps zmm2, zmm28, zmm29",
    "vfmadd231ps zmm3, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 8]",
    "vfmadd231ps zmm4, zmm28, zmm29",
    "vfmadd231ps zmm5, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 12]",
    "vfmadd231ps zmm6, zmm28, zmm29",
    "vfmadd231ps zmm7, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 16]",
    "vfmadd231ps zmm8, zmm28, zmm29",
    "vfmadd231ps zmm9, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 20]",
    "vfmadd231ps zmm10, zmm28, zmm29",
    "vfmadd231ps zmm11, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 24]",
    "vfmadd231ps zmm12, zmm28, zmm29",
    "vfmadd231ps zmm13, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 28]",
    "vfmadd231ps zmm14, zmm28, zmm29",
    "vfmadd231ps zmm15, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 32]",
    "vfmadd231ps zmm16, zmm28, zmm29",
    "vfmadd231ps zmm17, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 36]",
    "vfmadd231ps zmm18, zmm28, zmm29",
    "vfmadd231ps zmm19, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 40]",
    "vfmadd231ps zmm20, zmm28, zmm29",
    "vfmadd231ps zmm21, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 44]",
    "vfmadd231ps zmm22, zmm28, zmm29",
    "vfmadd231ps zmm23, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 48]",
    "vfmadd231ps zmm24, zmm28, zmm29",
    "vfmadd231ps zmm25, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 52]",
    "vfmadd231ps zmm26, zmm28, zmm29",
    "vfmadd231ps zmm27, zmm28, zmm30",

    // ---- k+1 ----
    // B[k+1] at [rsi + 128], A[k+1] at [rdi + 56]
    "vmovups zmm29, [rsi + 128]",
    "vmovups zmm30, [rsi + 192]",

    "vbroadcastss zmm28, [rdi + 56]",
    "vfmadd231ps zmm0, zmm28, zmm29",
    "vfmadd231ps zmm1, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 60]",
    "vfmadd231ps zmm2, zmm28, zmm29",
    "vfmadd231ps zmm3, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 64]",
    "vfmadd231ps zmm4, zmm28, zmm29",
    "vfmadd231ps zmm5, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 68]",
    "vfmadd231ps zmm6, zmm28, zmm29",
    "vfmadd231ps zmm7, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 72]",
    "vfmadd231ps zmm8, zmm28, zmm29",
    "vfmadd231ps zmm9, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 76]",
    "vfmadd231ps zmm10, zmm28, zmm29",
    "vfmadd231ps zmm11, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 80]",
    "vfmadd231ps zmm12, zmm28, zmm29",
    "vfmadd231ps zmm13, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 84]",
    "vfmadd231ps zmm14, zmm28, zmm29",
    "vfmadd231ps zmm15, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 88]",
    "vfmadd231ps zmm16, zmm28, zmm29",
    "vfmadd231ps zmm17, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 92]",
    "vfmadd231ps zmm18, zmm28, zmm29",
    "vfmadd231ps zmm19, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 96]",
    "vfmadd231ps zmm20, zmm28, zmm29",
    "vfmadd231ps zmm21, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 100]",
    "vfmadd231ps zmm22, zmm28, zmm29",
    "vfmadd231ps zmm23, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 104]",
    "vfmadd231ps zmm24, zmm28, zmm29",
    "vfmadd231ps zmm25, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 108]",
    "vfmadd231ps zmm26, zmm28, zmm29",
    "vfmadd231ps zmm27, zmm28, zmm30",

    // ---- k+2 ----
    // B[k+2] at [rsi + 256], A[k+2] at [rdi + 112]
    "vmovups zmm29, [rsi + 256]",
    "vmovups zmm30, [rsi + 320]",

    "prefetcht0 [rdi + 384]",
    "prefetcht0 [rsi + 768]",

    "vbroadcastss zmm28, [rdi + 112]",
    "vfmadd231ps zmm0, zmm28, zmm29",
    "vfmadd231ps zmm1, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 116]",
    "vfmadd231ps zmm2, zmm28, zmm29",
    "vfmadd231ps zmm3, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 120]",
    "vfmadd231ps zmm4, zmm28, zmm29",
    "vfmadd231ps zmm5, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 124]",
    "vfmadd231ps zmm6, zmm28, zmm29",
    "vfmadd231ps zmm7, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 128]",
    "vfmadd231ps zmm8, zmm28, zmm29",
    "vfmadd231ps zmm9, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 132]",
    "vfmadd231ps zmm10, zmm28, zmm29",
    "vfmadd231ps zmm11, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 136]",
    "vfmadd231ps zmm12, zmm28, zmm29",
    "vfmadd231ps zmm13, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 140]",
    "vfmadd231ps zmm14, zmm28, zmm29",
    "vfmadd231ps zmm15, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 144]",
    "vfmadd231ps zmm16, zmm28, zmm29",
    "vfmadd231ps zmm17, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 148]",
    "vfmadd231ps zmm18, zmm28, zmm29",
    "vfmadd231ps zmm19, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 152]",
    "vfmadd231ps zmm20, zmm28, zmm29",
    "vfmadd231ps zmm21, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 156]",
    "vfmadd231ps zmm22, zmm28, zmm29",
    "vfmadd231ps zmm23, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 160]",
    "vfmadd231ps zmm24, zmm28, zmm29",
    "vfmadd231ps zmm25, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 164]",
    "vfmadd231ps zmm26, zmm28, zmm29",
    "vfmadd231ps zmm27, zmm28, zmm30",

    // ---- k+3 ----
    // B[k+3] at [rsi + 384], A[k+3] at [rdi + 168]
    "vmovups zmm29, [rsi + 384]",
    "vmovups zmm30, [rsi + 448]",

    "vbroadcastss zmm28, [rdi + 168]",
    "vfmadd231ps zmm0, zmm28, zmm29",
    "vfmadd231ps zmm1, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 172]",
    "vfmadd231ps zmm2, zmm28, zmm29",
    "vfmadd231ps zmm3, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 176]",
    "vfmadd231ps zmm4, zmm28, zmm29",
    "vfmadd231ps zmm5, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 180]",
    "vfmadd231ps zmm6, zmm28, zmm29",
    "vfmadd231ps zmm7, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 184]",
    "vfmadd231ps zmm8, zmm28, zmm29",
    "vfmadd231ps zmm9, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 188]",
    "vfmadd231ps zmm10, zmm28, zmm29",
    "vfmadd231ps zmm11, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 192]",
    "vfmadd231ps zmm12, zmm28, zmm29",
    "vfmadd231ps zmm13, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 196]",
    "vfmadd231ps zmm14, zmm28, zmm29",
    "vfmadd231ps zmm15, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 200]",
    "vfmadd231ps zmm16, zmm28, zmm29",
    "vfmadd231ps zmm17, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 204]",
    "vfmadd231ps zmm18, zmm28, zmm29",
    "vfmadd231ps zmm19, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 208]",
    "vfmadd231ps zmm20, zmm28, zmm29",
    "vfmadd231ps zmm21, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 212]",
    "vfmadd231ps zmm22, zmm28, zmm29",
    "vfmadd231ps zmm23, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 216]",
    "vfmadd231ps zmm24, zmm28, zmm29",
    "vfmadd231ps zmm25, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 220]",
    "vfmadd231ps zmm26, zmm28, zmm29",
    "vfmadd231ps zmm27, zmm28, zmm30",

    // Advance pointers: A += 4*MR*4 = 224 bytes, B += 4*NR*4 = 512 bytes
    "add rdi, 224",
    "add rsi, 512",

    "sub r15, 4",
    "cmp r15, 4",
    "jge .Lavx512_k_loop_4x",

    // ── K-remainder loop (1x) ──
    ".Lavx512_k_remainder:",
    "test r15, r15",
    "jz .Lavx512_store_c",

    ".Lavx512_k_tail:",
    "vmovups zmm29, [rsi]",
    "vmovups zmm30, [rsi + 64]",

    "vbroadcastss zmm28, [rdi]",
    "vfmadd231ps zmm0, zmm28, zmm29",
    "vfmadd231ps zmm1, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 4]",
    "vfmadd231ps zmm2, zmm28, zmm29",
    "vfmadd231ps zmm3, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 8]",
    "vfmadd231ps zmm4, zmm28, zmm29",
    "vfmadd231ps zmm5, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 12]",
    "vfmadd231ps zmm6, zmm28, zmm29",
    "vfmadd231ps zmm7, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 16]",
    "vfmadd231ps zmm8, zmm28, zmm29",
    "vfmadd231ps zmm9, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 20]",
    "vfmadd231ps zmm10, zmm28, zmm29",
    "vfmadd231ps zmm11, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 24]",
    "vfmadd231ps zmm12, zmm28, zmm29",
    "vfmadd231ps zmm13, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 28]",
    "vfmadd231ps zmm14, zmm28, zmm29",
    "vfmadd231ps zmm15, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 32]",
    "vfmadd231ps zmm16, zmm28, zmm29",
    "vfmadd231ps zmm17, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 36]",
    "vfmadd231ps zmm18, zmm28, zmm29",
    "vfmadd231ps zmm19, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 40]",
    "vfmadd231ps zmm20, zmm28, zmm29",
    "vfmadd231ps zmm21, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 44]",
    "vfmadd231ps zmm22, zmm28, zmm29",
    "vfmadd231ps zmm23, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 48]",
    "vfmadd231ps zmm24, zmm28, zmm29",
    "vfmadd231ps zmm25, zmm28, zmm30",
    "vbroadcastss zmm28, [rdi + 52]",
    "vfmadd231ps zmm26, zmm28, zmm29",
    "vfmadd231ps zmm27, zmm28, zmm30",

    // Advance: A += MR*4 = 56 bytes, B += NR*4 = 128 bytes
    "add rdi, 56",
    "add rsi, 128",
    "dec r15",
    "jnz .Lavx512_k_tail",

    // ── Store C accumulators ──
    ".Lavx512_store_c:",
    // rows 0-5 via registers
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
    // rows 6-13 via stack pointers
    "mov rbx, [rsp]",
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

    // Clean up
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
    /// Raw assembly entry point for the 14x32 f32 AVX-512 GEMM microkernel.
    ///
    /// # Safety
    /// - All pointers must be valid and properly aligned (64-byte for AVX-512)
    /// - packed_a must contain at least kc * MR f32 elements
    /// - packed_b must contain at least kc * NR f32 elements
    /// - c_ptr must point to a buffer with at least MR rows of ldc elements each
    /// - kc must be > 0
    fn _gllm_gemm_14x32_avx512_f32(
        packed_a: *const f32,
        packed_b: *const f32,
        c_ptr: *mut f32,
        kc: usize,
        ldc: usize,
        accumulate: usize,
    );
}

/// Safe Rust wrapper for the 14x32 AVX-512 GEMM microkernel.
///
/// Computes: C[14x32] += A[14xKC] * B[KCx32]  (if accumulate=true)
///       or: C[14x32]  = A[14xKC] * B[KCx32]  (if accumulate=false)
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
    _gllm_gemm_14x32_avx512_f32(packed_a, packed_b, c_ptr, kc, ldc, accumulate as usize);
}
