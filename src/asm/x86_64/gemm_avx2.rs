//! Hand-written AVX2 GEMM 6x16 microkernel (x86_64).
//!
//! Phase 6 rewrite: 8x unrolled K-loop with minimal prefetch.
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
//!   ymm12 - ymm13 : A broadcast double-buffer
//!   ymm14 - ymm15 : B load double-buffer
//!
//! Skylake uop analysis per K step (steady state):
//!   6 vbroadcastss [mem] = 6 uops (p23)
//!   12 vfmadd231ps       = 12 uops (p01)
//!   2 vmovups [mem]       = 2 uops (p23)
//!   Total: 20 uops/K → 5 cycles frontend, 6 cycles FMA-bound
//!
//! 8x unroll: 160 FMA uops + 64 load uops + 4 prefetch + 8 loop = 236 uops
//!   FMA: 96/2 = 48 cycles (bottleneck)
//!   Frontend: 236/4 = 59 cycles (< 48? No: 236/4=59 > 48)
//!   Actually: need to recount. 8K × 20 = 160 core + 4 pf + 8 loop = 172
//!   Frontend: 172/4 = 43 cycles < 48 FMA cycles → FMA-bound ✓
//!
//! Calling convention (System V AMD64 ABI, extern "C"):
//!   rdi = *const f32 : packed_a
//!   rsi = *const f32 : packed_b
//!   rdx = *mut f32   : c_ptr
//!   rcx = usize      : kc
//!   r8  = usize      : ldc (elements)
//!   r9  = usize      : accumulate (0=zero, 1=load)

#[cfg(target_arch = "x86_64")]
use std::arch::global_asm;

/// Microkernel tile dimensions.
pub const MR: usize = 6;
pub const NR: usize = 16;

/// Epilogue variant selector for fused GEMM microkernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmEpilogue {
    /// No epilogue — plain GEMM (optional bias via pointer).
    None,
    /// Fused SiLU: C = silu(A*B) or C = silu(A*B + bias).
    Silu,
    /// Fused GELU: C = gelu(A*B) or C = gelu(A*B + bias).
    Gelu,
    /// Fused residual add: C = A*B + residual.
    Residual,
}

// Helper macro: one K-step of 6x16 FMA with B already in ymm14/ymm15.
// B[k+1] load interleaved after row 2 FMAs (between 4th and 5th FMA).
// $a_off = byte offset into packed_a for this K step
// $b_next_lo, $b_next_hi = byte offsets for next B load (or "" to skip)
macro_rules! k_step_asm {
    // With B preload for next step
    ($a_off:literal, $b_lo:literal, $b_hi:literal) => {
        concat!(
            "vbroadcastss ymm12, [rdi + ", $a_off, "]\n",
            "vfmadd231ps ymm0, ymm12, ymm14\n",
            "vfmadd231ps ymm1, ymm12, ymm15\n",
            "vbroadcastss ymm13, [rdi + ", $a_off, " + 4]\n",
            "vfmadd231ps ymm2, ymm13, ymm14\n",
            "vbroadcastss ymm12, [rdi + ", $a_off, " + 8]\n",
            "vfmadd231ps ymm3, ymm13, ymm15\n",
            "vfmadd231ps ymm4, ymm12, ymm14\n",
            "vbroadcastss ymm13, [rdi + ", $a_off, " + 12]\n",
            "vfmadd231ps ymm5, ymm12, ymm15\n",
            "vfmadd231ps ymm6, ymm13, ymm14\n",
            "vbroadcastss ymm12, [rdi + ", $a_off, " + 16]\n",
            "vfmadd231ps ymm7, ymm13, ymm15\n",
            "vfmadd231ps ymm8, ymm12, ymm14\n",
            "vbroadcastss ymm13, [rdi + ", $a_off, " + 20]\n",
            "vfmadd231ps ymm9, ymm12, ymm15\n",
            "vfmadd231ps ymm10, ymm13, ymm14\n",
            "vfmadd231ps ymm11, ymm13, ymm15\n",
            "vmovups ymm14, [rsi + ", $b_lo, "]\n",
            "vmovups ymm15, [rsi + ", $b_hi, "]\n",
        )
    };
    // Last step: no B preload
    (last $a_off:literal) => {
        concat!(
            "vbroadcastss ymm12, [rdi + ", $a_off, "]\n",
            "vfmadd231ps ymm0, ymm12, ymm14\n",
            "vfmadd231ps ymm1, ymm12, ymm15\n",
            "vbroadcastss ymm13, [rdi + ", $a_off, " + 4]\n",
            "vfmadd231ps ymm2, ymm13, ymm14\n",
            "vbroadcastss ymm12, [rdi + ", $a_off, " + 8]\n",
            "vfmadd231ps ymm3, ymm13, ymm15\n",
            "vfmadd231ps ymm4, ymm12, ymm14\n",
            "vbroadcastss ymm13, [rdi + ", $a_off, " + 12]\n",
            "vfmadd231ps ymm5, ymm12, ymm15\n",
            "vfmadd231ps ymm6, ymm13, ymm14\n",
            "vbroadcastss ymm12, [rdi + ", $a_off, " + 16]\n",
            "vfmadd231ps ymm7, ymm13, ymm15\n",
            "vfmadd231ps ymm8, ymm12, ymm14\n",
            "vbroadcastss ymm13, [rdi + ", $a_off, " + 20]\n",
            "vfmadd231ps ymm9, ymm12, ymm15\n",
            "vfmadd231ps ymm10, ymm13, ymm14\n",
            "vfmadd231ps ymm11, ymm13, ymm15\n",
        )
    };
}

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
    "cmp r15, 8",
    "jl .Lavx2_k_check_4x",

    // ── Pre-load B[k=0] for software pipeline ──
    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",

    // ── Main K-loop: 8x unrolled ──
    // B strip (KC×NR×4 = 24KB) is L1-resident → NO B prefetch needed.
    // A streams from L2 → prefetch A to L1 every 4 K-steps (~24 cycles ahead).
    // 8K: 96 FMA + 48 bcast + 16 Bload + 2 prefetch + 8 loop = 170 uops
    // FMA: 96/2 = 48 cycles. Frontend: 170/4 = 42.5 cycles. → FMA-bound ✓
    ".align 32",
    ".Lavx2_k_loop_8x:",

    // k+0: B in ymm14/ymm15, prefetch A to L1 (~8 K-steps ahead)
    "prefetcht0 [rdi + 192]",
    k_step_asm!("0", "64", "96"),

    // k+1
    k_step_asm!("24", "128", "160"),

    // k+2
    k_step_asm!("48", "192", "224"),

    // k+3
    k_step_asm!("72", "256", "288"),

    // k+4: prefetch A to L1 for second half
    "prefetcht0 [rdi + 288]",
    k_step_asm!("96", "320", "352"),

    // k+5
    k_step_asm!("120", "384", "416"),

    // k+6
    k_step_asm!("144", "448", "480"),

    // k+7: last step, no B preload needed (done below)
    k_step_asm!(last "168"),

    // Advance: A += 8*MR*4 = 192, B += 8*NR*4 = 512
    "add rdi, 192",
    "add rsi, 512",

    "sub r15, 8",

    // Pre-load B for next iteration BEFORE the branch check.
    // This gives 5+ cycles of ALU/branch between load issue and
    // first FMA use at loop top, hiding L1 load latency (4-5 cy).
    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",

    "cmp r15, 8",
    "jge .Lavx2_k_loop_8x",

    // ── K-remainder: 4x unrolled ──
    ".Lavx2_k_check_4x:",
    "cmp r15, 4",
    "jl .Lavx2_k_remainder",

    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",

    k_step_asm!("0", "64", "96"),
    k_step_asm!("24", "128", "160"),
    k_step_asm!("48", "192", "224"),
    k_step_asm!(last "72"),

    "add rdi, 96",
    "add rsi, 256",
    "sub r15, 4",

    // ── K-remainder loop (1x) ──
    ".Lavx2_k_remainder:",
    "test r15, r15",
    "jz .Lavx2_store_c",

    ".Lavx2_k_tail:",
    "vmovups ymm14, [rsi]",
    "vmovups ymm15, [rsi + 32]",

    "vbroadcastss ymm12, [rdi]",
    "vfmadd231ps ymm0, ymm12, ymm14",
    "vfmadd231ps ymm1, ymm12, ymm15",
    "vbroadcastss ymm13, [rdi + 4]",
    "vfmadd231ps ymm2, ymm13, ymm14",
    "vbroadcastss ymm12, [rdi + 8]",
    "vfmadd231ps ymm3, ymm13, ymm15",
    "vfmadd231ps ymm4, ymm12, ymm14",
    "vbroadcastss ymm13, [rdi + 12]",
    "vfmadd231ps ymm5, ymm12, ymm15",
    "vfmadd231ps ymm6, ymm13, ymm14",
    "vbroadcastss ymm12, [rdi + 16]",
    "vfmadd231ps ymm7, ymm13, ymm15",
    "vfmadd231ps ymm8, ymm12, ymm14",
    "vbroadcastss ymm13, [rdi + 20]",
    "vfmadd231ps ymm9, ymm12, ymm15",
    "vfmadd231ps ymm10, ymm13, ymm14",
    "vfmadd231ps ymm11, ymm13, ymm15",

    "add rdi, 24",
    "add rsi, 64",
    "dec r15",
    "jnz .Lavx2_k_tail",

    // ── Prefetch C rows to reduce RFO latency on stores ──
    ".Lavx2_store_c:",
    "prefetcht0 [rdx]",
    "prefetcht0 [rdx + 32]",
    "prefetcht0 [r10]",
    "prefetcht0 [r10 + 32]",
    "prefetcht0 [r11]",
    "prefetcht0 [r11 + 32]",
    "prefetcht0 [r12]",
    "prefetcht0 [r12 + 32]",
    "prefetcht0 [r13]",
    "prefetcht0 [r13 + 32]",
    "prefetcht0 [r14]",
    "prefetcht0 [r14 + 32]",

    // ── Fuse bias if bias pointer (7th arg) is non-null ──
    // 7th arg is on stack: [rsp + 48] (5 pushes × 8 + return addr × 8 = 48)
    "mov rax, [rsp + 48]",
    "test rax, rax",
    "jz .Lavx2_store_no_bias",

    // Load bias vector (16 floats = 2 ymm)
    "vmovups ymm12, [rax]",
    "vmovups ymm13, [rax + 32]",

    // Add bias to all 6 rows
    "vaddps ymm0, ymm0, ymm12",
    "vaddps ymm1, ymm1, ymm13",
    "vaddps ymm2, ymm2, ymm12",
    "vaddps ymm3, ymm3, ymm13",
    "vaddps ymm4, ymm4, ymm12",
    "vaddps ymm5, ymm5, ymm13",
    "vaddps ymm6, ymm6, ymm12",
    "vaddps ymm7, ymm7, ymm13",
    "vaddps ymm8, ymm8, ymm12",
    "vaddps ymm9, ymm9, ymm13",
    "vaddps ymm10, ymm10, ymm12",
    "vaddps ymm11, ymm11, ymm13",

    ".Lavx2_store_no_bias:",

    // ── Store C accumulators ──
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
        bias: *const f32,
    );
}

/// Safe Rust wrapper for the 6x16 AVX2 GEMM microkernel.
/// When `bias` is non-null and points to NR (16) floats, bias is fused
/// into the store stage (zero extra cache misses).
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
    _gllm_gemm_6x16_avx2_f32(packed_a, packed_b, c_ptr, kc, ldc, accumulate as usize, std::ptr::null());
}

/// GEMM microkernel with fused bias addition.
/// `bias` must point to NR (16) floats for the current tile columns.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn gemm_kernel_6x16_f32_bias(
    packed_a: *const f32,
    packed_b: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    ldc: usize,
    accumulate: bool,
    bias: *const f32,
) {
    _gllm_gemm_6x16_avx2_f32(packed_a, packed_b, c_ptr, kc, ldc, accumulate as usize, bias);
}
