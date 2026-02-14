//! # CPU Kernel Macro Architecture
//!
//! 4-layer macro system for ISA-dispatched SIMD kernels:
//!
//! ## Layer 1: `simd_primitive!` (src/macros/simd_primitive.rs)
//! Maps abstract ops to hardware intrinsics: `simd_primitive!(avx2, f32, add, a, b)`.
//! Covers scalar/avx2/avx512/neon × f32/f16/bf16.
//!
//! ## Layer 2: `operator_templates` (src/macros/operator_templates.rs)
//! Generic operator bodies parameterized by ISA+Element:
//! `define_element_wise_ops!`, `define_blas1_ops!`, `define_norm_ops!`, etc.
//!
//! ## Layer 3: `quant_primitive!` (src/macros/quant_primitive.rs)
//! Quantization decode/dot primitives: `quant_primitive!(neon, q4_k, decode, blk, dst)`.
//! Each `(isa, format, op)` tuple is a separate macro branch.
//!
//! ## Layer 4: `expand_isa_impls!` (src/macros/expand.rs)
//! Generates per-ISA modules: `expand_isa_impls!(avx2_f32, avx2, f32)`.
//!
//! ## Dispatch macros (this file)
//! Runtime ISA selection via `get_isa_level()`:
//!
//! | Macro | Signature | Used by |
//! |---|---|---|
//! | `define_quant_dot_k!` | `(name, qfmt, BlockTy)` | K-quant dot (Q2-Q8_K) |
//! | `define_quant_dot_iq!` | `(name, qfmt, BlockTy)` | IQ dot (avx2+scalar only) |
//! | `define_quant_decode_k!` | `(name, qfmt, BlockTy)` | K-quant decode |
//! | `define_quant_decode_scalar!` | `(name, qfmt, BlockTy)` | IQ decode (scalar only) |
//! | `dispatch_binary_op!` | `(fn, op_name)` | vec add/sub/mul/div |
//! | `dispatch_unary_op!` | `(fn, op_name)` | neg/abs/recip |
//! | `dispatch_reduce_op!` | `(fn, op_name)` | sum/max/min |
//! | `dispatch_with_scalar!` | `(fn, op_name)` | scalar_add/mul |
//! | `dispatch_with_eps!` | `(fn, op_name)` | rms_norm/layer_norm |
//! | `dispatch_rope!` | `(fn, op_name)` | rope |
//! | `dispatch_scale!` | `(fn, op_name)` | vec_scale |

use crate::traits::{Element, Kernels};
use std::marker::PhantomData;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaLevel {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

static ISA_LEVEL: OnceLock<IsaLevel> = OnceLock::new();

pub fn get_isa_level() -> IsaLevel {
    *ISA_LEVEL.get_or_init(detect_isa_features)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_isa_features() -> IsaLevel {
    if is_x86_feature_detected!("avx512f") {
        IsaLevel::Avx512
    } else if is_x86_feature_detected!("avx2") {
        IsaLevel::Avx2
    } else {
        IsaLevel::Scalar
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_isa_features() -> IsaLevel {
    IsaLevel::Neon
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_isa_features() -> IsaLevel {
    IsaLevel::Scalar
}


/// Convert any Element slice to f32 vec (for quantization paths that need f32 input)
#[inline]
fn elem_to_f32_vec<E: Element>(src: &[E]) -> Vec<f32> {
    if let Some(f) = E::as_f32_slice(src) {
        return f.to_vec();
    }
    src.iter().map(|v| v.to_f32()).collect()
}

pub struct CpuKernels<E: Element> {
    _phantom: PhantomData<E>,
}

impl<E: Element> CpuKernels<E> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

// ============================================================================
// Quant dispatch macros — eliminate repetitive ISA match boilerplate
// ============================================================================

/// Generate a dot_$fn_name function with ISA dispatch.
/// Pattern: avx512 + avx2 + scalar (K-Quant formats)
macro_rules! define_quant_dot_k {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], other: &[f32]) -> f32 {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let src = other.as_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 => { crate::quant_primitive!(avx512, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, dot, blk, src) }
                _ => { crate::quant_primitive!(scalar, $qfmt, dot, blk, src) }
            }
        }
    };
}

/// Generate a dot_$fn_name function: avx2 + scalar (IQ formats, avx512 falls through to avx2)
macro_rules! define_quant_dot_iq {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], other: &[f32]) -> f32 {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let src = other.as_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 | IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, dot, blk, src) }
                _ => { crate::quant_primitive!(scalar, $qfmt, dot, blk, src) }
            }
        }
    };
}

/// Generate a dequant_$fn_name function with ISA dispatch (avx512 + avx2 + scalar)
macro_rules! define_quant_decode_k {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], out: &mut [f32]) {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let dst = out.as_mut_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 => { crate::quant_primitive!(avx512, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, decode, blk, dst); }
                _ => { crate::quant_primitive!(scalar, $qfmt, decode, blk, dst); }
            }
        }
    };
}

/// Generate a scalar-only dequant function
macro_rules! define_quant_decode_scalar {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], out: &mut [f32]) {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let dst = out.as_mut_ptr();
            crate::quant_primitive!(scalar, $qfmt, decode, blk, dst);
        }
    };
}

/// Generate a dequant function: avx512 + neon + scalar (no avx2 decode)
macro_rules! define_quant_decode_avx512 {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], out: &mut [f32]) {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let dst = out.as_mut_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 => { crate::quant_primitive!(avx512, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, decode, blk, dst); }
                _ => { crate::quant_primitive!(scalar, $qfmt, decode, blk, dst); }
            }
        }
    };
}

// ============================================================================
// Private helper methods on CpuKernels
// ============================================================================

impl<E: Element> CpuKernels<E> {
    // K-Quant dot: avx512 + avx2 + scalar
    define_quant_dot_k!(dot_q4_k, q4_k, BlockQ4K);
    define_quant_dot_k!(dot_q8_k, q8_k, BlockQ8K);
    define_quant_dot_k!(dot_q2_k, q2_k, BlockQ2K);
    define_quant_dot_k!(dot_q3_k, q3_k, BlockQ3K);
    define_quant_dot_k!(dot_q5_k, q5_k, BlockQ5K);
    define_quant_dot_k!(dot_q6_k, q6_k, BlockQ6K);

    // IQ dot: avx2 + scalar (avx512 falls through to avx2)
    define_quant_dot_iq!(dot_iq1_s, iq1_s, BlockIQ1S);
    define_quant_dot_iq!(dot_iq1_m, iq1_m, BlockIQ1M);
    define_quant_dot_iq!(dot_iq2_xxs, iq2_xxs, BlockIQ2XXS);
    define_quant_dot_iq!(dot_iq2_xs, iq2_xs, BlockIQ2XS);
    define_quant_dot_iq!(dot_iq2_s, iq2_s, BlockIQ2S);
    define_quant_dot_iq!(dot_iq3_xxs, iq3_xxs, BlockIQ3XXS);
    define_quant_dot_iq!(dot_iq3_s, iq3_s, BlockIQ3S);
    define_quant_dot_iq!(dot_iq4_nl, iq4_nl, BlockIQ4NL);
    define_quant_dot_iq!(dot_iq4_xs, iq4_xs, BlockIQ4XS);

    // Scalar-only dot
    fn dot_awq4(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockAWQ4;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, awq4, dot, blk, src)
    }
    fn dot_gptq4(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockGPTQ4;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, gptq4, dot, blk, src)
    }

    // ========================================================================
    // Hot-path inner loops: const-generic block_bytes/block_size + fn pointer
    // Eliminates match quant_type inside the inner loop (SPEC §8 compliance)
    // ========================================================================

    #[inline(always)]
    fn fused_dequant_gemv_inner<const BLOCK_BYTES: usize, const BLOCK_SIZE: usize>(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, k: usize,
        dot_fn: fn(&Self, &[u8], &[f32]) -> f32,
    ) {
        let blocks_per_row = k / BLOCK_SIZE;
        // Zero-copy for f32, single allocation for f16/bf16
        let owned_f32: Vec<f32>;
        let in_f32: &[f32] = if let Some(f) = E::as_f32_slice(input) {
            f
        } else {
            owned_f32 = input.iter().map(|v| v.to_f32()).collect();
            &owned_f32
        };
        for i in 0..m {
            let mut sum = 0.0f32;
            for b in 0..blocks_per_row {
                let off = i * blocks_per_row * BLOCK_BYTES + b * BLOCK_BYTES;
                let blk = &weight_blocks[off..off + BLOCK_BYTES];
                let in_slice = &in_f32[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                sum += dot_fn(self, blk, in_slice);
            }
            output[i] = E::from_f32(sum);
        }
    }

    #[inline(always)]
    fn quant_matmul_inner<const BLOCK_BYTES: usize, const BLOCK_SIZE: usize>(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
        dot_fn: fn(&Self, &[u8], &[f32]) -> f32,
    ) {
        let blocks_per_row = k / BLOCK_SIZE;
        // Pre-allocate column buffer once, reuse across all j iterations
        let mut in_f32_col = vec![0.0f32; k];
        for j in 0..n {
            // Extract column j from input (stride access: input[p * n + j])
            for p in 0..k {
                in_f32_col[p] = input[p * n + j].to_f32();
            }
            for i in 0..m {
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let off = i * blocks_per_row * BLOCK_BYTES + b * BLOCK_BYTES;
                    let blk = &weight_blocks[off..off + BLOCK_BYTES];
                    let in_slice = &in_f32_col[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                    sum += dot_fn(self, blk, in_slice);
                }
                output[i * n + j] = E::from_f32(sum);
            }
        }
    }
}

// Import the kernel modules
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;

// ============================================================================
// Dispatch macros: use ELEM_ID for zero-cost type dispatch
// ELEM_ID is a compile-time constant, so dead branches are eliminated.
// For f16/bf16, we transmute slices and call the native f16/bf16 ISA modules
// which handle load/store conversion internally via simd_primitive.
// ============================================================================

/// Transmute a &[E] to &[$target] when E::ELEM_ID matches.
/// Safety: caller must ensure E is actually $target (checked via ELEM_ID).
macro_rules! as_typed_slice {
    ($slice:expr, $target:ty) => {
        unsafe { std::slice::from_raw_parts($slice.as_ptr() as *const $target, $slice.len()) }
    };
}

macro_rules! as_typed_slice_mut {
    ($slice:expr, $target:ty) => {
        unsafe { std::slice::from_raw_parts_mut($slice.as_mut_ptr() as *mut $target, $slice.len()) }
    };
}

macro_rules! dispatch_binary_op {
    ($a:expr, $b:expr, $out:expr, $op:ident) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32)),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16)),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            _ => unreachable!(),
        }
    };
}

macro_rules! dispatch_unary_op {
    ($a:expr, $out:expr, $op:ident) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32), as_typed_slice_mut!($out, f32)),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice_mut!($out, half::f16)),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice_mut!($out, half::bf16)),
            _ => unreachable!(),
        }
    };
}

macro_rules! dispatch_reduce_op {
    ($a:expr, $op:ident) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16)).to_f32(),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32)).to_f32(),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16)).to_f32(),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16)).to_f32(),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32)).to_f32(),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16)).to_f32(),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16)).to_f32(),
            _ => unreachable!(),
        }
    };
}

/// Dispatch with scalar parameter conversion (for axpy, etc.)
/// Usage: dispatch_with_scalar!(op_name, scalar_f32_value, x_slice, y_mut_slice)
macro_rules! dispatch_with_scalar {
    ($op:ident, $scalar:expr, $x:expr, $y:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op($scalar, as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(half::f16::from_f32($scalar), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(half::bf16::from_f32($scalar), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op($scalar, as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(half::f16::from_f32($scalar), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(half::bf16::from_f32($scalar), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op($scalar, as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(half::f16::from_f32($scalar), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(half::bf16::from_f32($scalar), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16)),
            (_, 0) => scalar::scalar_f32::$op($scalar, as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32)),
            (_, 1) => scalar::scalar_f16::$op(half::f16::from_f32($scalar), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16)),
            (_, 2) => scalar::scalar_bf16::$op(half::bf16::from_f32($scalar), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16)),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for norm ops with eps parameter (rms_norm, layer_norm).
/// eps is f32 for f32 path, converted via from_f32 for f16/bf16.
/// 2-slice + out + eps variant (rms_norm)
macro_rules! dispatch_with_eps {
    ($op:ident, $x:expr, $w:expr, $out:expr, $eps:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($w, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($w, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($w, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($w, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($w, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($w, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($w, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($w, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($w, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($w, f32), as_typed_slice_mut!($out, f32), $eps),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($w, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($w, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            _ => unreachable!(),
        }
    };
    // 3-slice + out + eps variant (layer_norm: x, gamma, beta, out, eps)
    ($op:ident, $x:expr, $g:expr, $b:expr, $out:expr, $eps:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($g, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($g, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($g, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($g, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($g, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($g, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($g, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($g, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($g, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($g, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32), $eps),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($g, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($g, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for operations with extra usize parameters (gemv, gemm, etc.)
/// Usage: dispatch_with_dims!(gemv, a, x, y, m, n)
macro_rules! dispatch_with_dims {
    // For gemv: 3 slices + 2 dims
    ($op:ident, $a:expr, $x:expr, $y:expr, $m:expr, $n:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16), $m, $n),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32), $m, $n),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16), $m, $n),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16), $m, $n),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32), $m, $n),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16), $m, $n),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16), $m, $n),
            _ => unreachable!(),
        }
    };
    // For gemm: 3 slices + 3 dims
    ($op:ident, $a:expr, $b:expr, $c:expr, $m:expr, $n:expr, $k:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for rope: 1 mut slice + 2 read slices + usize
macro_rules! dispatch_rope {
    ($op:ident, $qk:expr, $cos:expr, $sin:expr, $hd:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for in-place scale: read slice + f32 scalar + mut out, then copy back
macro_rules! dispatch_scale {
    ($op:ident, $x:expr, $sf:expr, $tmp:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::$op(as_typed_slice!($x, f32), $sf, as_typed_slice_mut!($tmp, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($x, half::f16), half::f16::from_f32($sf), as_typed_slice_mut!($tmp, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::$op(as_typed_slice!($x, half::bf16), half::bf16::from_f32($sf), as_typed_slice_mut!($tmp, half::bf16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($x, f32), $sf, as_typed_slice_mut!($tmp, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($x, half::f16), half::f16::from_f32($sf), as_typed_slice_mut!($tmp, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($x, half::bf16), half::bf16::from_f32($sf), as_typed_slice_mut!($tmp, half::bf16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($x, f32), $sf, as_typed_slice_mut!($tmp, f32)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($x, half::f16), half::f16::from_f32($sf), as_typed_slice_mut!($tmp, half::f16)),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($x, half::bf16), half::bf16::from_f32($sf), as_typed_slice_mut!($tmp, half::bf16)),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($x, f32), $sf, as_typed_slice_mut!($tmp, f32)),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($x, half::f16), half::f16::from_f32($sf), as_typed_slice_mut!($tmp, half::f16)),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($x, half::bf16), half::bf16::from_f32($sf), as_typed_slice_mut!($tmp, half::bf16)),
            _ => unreachable!(),
        }
    };
}

// ============================================================================
// Flash Attention dispatch macros
// ============================================================================

macro_rules! dispatch_flash_attention {
    ($q:expr, $k:expr, $v:expr, $out:expr, $seq_len:expr, $num_heads:expr, $head_dim:expr, $scale:expr, $causal:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::flash_attention(as_typed_slice!($q, f32), as_typed_slice!($k, f32), as_typed_slice!($v, f32), as_typed_slice_mut!($out, f32), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::flash_attention(as_typed_slice!($q, half::f16), as_typed_slice!($k, half::f16), as_typed_slice!($v, half::f16), as_typed_slice_mut!($out, half::f16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::flash_attention(as_typed_slice!($q, half::bf16), as_typed_slice!($k, half::bf16), as_typed_slice!($v, half::bf16), as_typed_slice_mut!($out, half::bf16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::flash_attention(as_typed_slice!($q, f32), as_typed_slice!($k, f32), as_typed_slice!($v, f32), as_typed_slice_mut!($out, f32), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::flash_attention(as_typed_slice!($q, half::f16), as_typed_slice!($k, half::f16), as_typed_slice!($v, half::f16), as_typed_slice_mut!($out, half::f16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::flash_attention(as_typed_slice!($q, half::bf16), as_typed_slice!($k, half::bf16), as_typed_slice!($v, half::bf16), as_typed_slice_mut!($out, half::bf16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::flash_attention(as_typed_slice!($q, f32), as_typed_slice!($k, f32), as_typed_slice!($v, f32), as_typed_slice_mut!($out, f32), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::flash_attention(as_typed_slice!($q, half::f16), as_typed_slice!($k, half::f16), as_typed_slice!($v, half::f16), as_typed_slice_mut!($out, half::f16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::flash_attention(as_typed_slice!($q, half::bf16), as_typed_slice!($k, half::bf16), as_typed_slice!($v, half::bf16), as_typed_slice_mut!($out, half::bf16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            (_, 0) => scalar::scalar_f32::flash_attention(as_typed_slice!($q, f32), as_typed_slice!($k, f32), as_typed_slice!($v, f32), as_typed_slice_mut!($out, f32), $seq_len, $num_heads, $head_dim, $scale, $causal),
            (_, 1) => scalar::scalar_f16::flash_attention(as_typed_slice!($q, half::f16), as_typed_slice!($k, half::f16), as_typed_slice!($v, half::f16), as_typed_slice_mut!($out, half::f16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            (_, 2) => scalar::scalar_bf16::flash_attention(as_typed_slice!($q, half::bf16), as_typed_slice!($k, half::bf16), as_typed_slice!($v, half::bf16), as_typed_slice_mut!($out, half::bf16), $seq_len, $num_heads, $head_dim, $scale, $causal),
            _ => unreachable!(),
        }
    };
}

macro_rules! dispatch_flash_attention_paged {
    ($q:expr, $kc:expr, $vc:expr, $pt:expr, $out:expr, $sl:expr, $cl:expr, $nh:expr, $nkv:expr, $hd:expr, $ps:expr, $sc:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 0) => avx512::avx512_f32::flash_attention_paged(as_typed_slice!($q, f32), as_typed_slice!($kc, f32), as_typed_slice!($vc, f32), $pt, as_typed_slice_mut!($out, f32), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::flash_attention_paged(as_typed_slice!($q, half::f16), as_typed_slice!($kc, half::f16), as_typed_slice!($vc, half::f16), $pt, as_typed_slice_mut!($out, half::f16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 2) => avx512::avx512_bf16::flash_attention_paged(as_typed_slice!($q, half::bf16), as_typed_slice!($kc, half::bf16), as_typed_slice!($vc, half::bf16), $pt, as_typed_slice_mut!($out, half::bf16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::flash_attention_paged(as_typed_slice!($q, f32), as_typed_slice!($kc, f32), as_typed_slice!($vc, f32), $pt, as_typed_slice_mut!($out, f32), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::flash_attention_paged(as_typed_slice!($q, half::f16), as_typed_slice!($kc, half::f16), as_typed_slice!($vc, half::f16), $pt, as_typed_slice_mut!($out, half::f16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::flash_attention_paged(as_typed_slice!($q, half::bf16), as_typed_slice!($kc, half::bf16), as_typed_slice!($vc, half::bf16), $pt, as_typed_slice_mut!($out, half::bf16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::flash_attention_paged(as_typed_slice!($q, f32), as_typed_slice!($kc, f32), as_typed_slice!($vc, f32), $pt, as_typed_slice_mut!($out, f32), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::flash_attention_paged(as_typed_slice!($q, half::f16), as_typed_slice!($kc, half::f16), as_typed_slice!($vc, half::f16), $pt, as_typed_slice_mut!($out, half::f16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::flash_attention_paged(as_typed_slice!($q, half::bf16), as_typed_slice!($kc, half::bf16), as_typed_slice!($vc, half::bf16), $pt, as_typed_slice_mut!($out, half::bf16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            (_, 0) => scalar::scalar_f32::flash_attention_paged(as_typed_slice!($q, f32), as_typed_slice!($kc, f32), as_typed_slice!($vc, f32), $pt, as_typed_slice_mut!($out, f32), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            (_, 1) => scalar::scalar_f16::flash_attention_paged(as_typed_slice!($q, half::f16), as_typed_slice!($kc, half::f16), as_typed_slice!($vc, half::f16), $pt, as_typed_slice_mut!($out, half::f16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            (_, 2) => scalar::scalar_bf16::flash_attention_paged(as_typed_slice!($q, half::bf16), as_typed_slice!($kc, half::bf16), as_typed_slice!($vc, half::bf16), $pt, as_typed_slice_mut!($out, half::bf16), $sl, $cl, $nh, $nkv, $hd, $ps, $sc),
            _ => unreachable!(),
        }
    };
}

// ============================================================================
// Kernels implementation
// ============================================================================

impl<E: Element> Kernels<E> for CpuKernels<E> {

    // BLAS-1
    fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]) { dispatch_binary_op!(a, b, out, add); }
    fn vec_sub(&self, a: &[E], b: &[E], out: &mut [E]) { dispatch_binary_op!(a, b, out, sub); }
    fn vec_mul(&self, a: &[E], b: &[E], out: &mut [E]) { dispatch_binary_op!(a, b, out, mul); }

    fn vec_dot(&self, a: &[E], b: &[E]) -> E {
        let len = a.len();
        assert!(b.len() == len);
        let mut tmp = vec![E::ZERO; len];
        self.vec_mul(a, b, &mut tmp);
        self.vec_sum(&tmp)
    }

    fn vec_scale(&self, x: &mut [E], s: E) {
        let mut tmp = vec![E::ZERO; x.len()];
        let sf = s.to_f32();
        dispatch_scale!(scale, x, sf, &mut tmp);
        x.copy_from_slice(&tmp);
    }

    fn vec_axpy(&self, y: &mut [E], a: E, x: &[E]) {
        let af = a.to_f32();
        dispatch_with_scalar!(axpy, af, x, y);
    }

    fn vec_sum(&self, x: &[E]) -> E { E::from_f32(dispatch_reduce_op!(x, sum)) }
    fn vec_max(&self, x: &[E]) -> E { E::from_f32(dispatch_reduce_op!(x, max_val)) }
    fn vec_sum_squares(&self, x: &[E]) -> E { E::from_f32(dispatch_reduce_op!(x, sum_squares)) }

    // BLAS-2/3
    fn gemv(&self, a: &[E], x: &[E], y: &mut [E], m: usize, n: usize) {
        dispatch_with_dims!(gemv, a, x, y, m, n);
    }

    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        dispatch_with_dims!(matmul, a, b, c, m, n, k);
    }

    fn gemm_bias(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        self.gemm(a, b, c, m, n, k);
        assert!(c.len() == m * n && bias.len() == n);
        for i in 0..m {
            let row = &mut c[i*n..(i+1)*n];
            self.vec_axpy(row, E::ONE, bias);
        }
    }

    // Activations
    fn silu(&self, a: &[E], out: &mut [E]) { dispatch_unary_op!(a, out, silu); }
    fn relu(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, relu); }
    fn gelu(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, gelu); }
    fn tanh(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, tanh); }
    fn exp(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, exp); }
    fn softmax(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, softmax); }

    fn swiglu(&self, gate: &[E], up: &[E], out: &mut [E]) {
        let len = gate.len();
        let mut silu_out = vec![E::ZERO; len];
        self.silu(gate, &mut silu_out);
        self.vec_mul(&silu_out, up, out);
    }

    // Normalization
    fn rms_norm(&self, x: &[E], weight: &[E], out: &mut [E], eps: f32) {
        dispatch_with_eps!(rms_norm, x, weight, out, eps);
    }

    fn layer_norm(&self, x: &[E], gamma: &[E], beta: &[E], out: &mut [E], eps: f32) {
        dispatch_with_eps!(layer_norm, x, gamma, beta, out, eps);
    }

    // Positional
    fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, _interleaved: bool) {
        dispatch_rope!(rope, qk, cos, sin, head_dim);
    }

    // Embedding
    fn embedding_lookup(&self, ids: &[u32], table: &[E], output: &mut [E], _vocab_size: usize, hidden_size: usize) {
        match E::ELEM_ID {
            0 => scalar::scalar_f32::embedding_lookup(as_typed_slice!(table, f32), ids, as_typed_slice_mut!(output, f32), hidden_size),
            1 => scalar::scalar_f16::embedding_lookup(as_typed_slice!(table, half::f16), ids, as_typed_slice_mut!(output, half::f16), hidden_size),
            2 => scalar::scalar_bf16::embedding_lookup(as_typed_slice!(table, half::bf16), ids, as_typed_slice_mut!(output, half::bf16), hidden_size),
            _ => unreachable!(),
        }
    }

    // Sampling
    // Quantization
    // K-Quant decode: avx512 + avx2 + scalar
    define_quant_decode_k!(dequant_q4_k, q4_k, BlockQ4K);
    define_quant_decode_k!(dequant_q3_k, q3_k, BlockQ3K);
    define_quant_decode_k!(dequant_q5_k, q5_k, BlockQ5K);
    define_quant_decode_k!(dequant_q6_k, q6_k, BlockQ6K);

    // Q2_K/Q8_K decode: avx512 + scalar (no avx2 decode branch)
    define_quant_decode_avx512!(dequant_q2_k, q2_k, BlockQ2K);
    define_quant_decode_avx512!(dequant_q8_k, q8_k, BlockQ8K);

    // Quantized GEMV/GEMM
    fn gemv_q4(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        assert!(n % 256 == 0);
        let blocks = n / 256;
        let block_size = std::mem::size_of::<crate::quant::BlockQ4K>();
        if let Some(in_f32) = E::as_f32_slice(input) {
            let mut sum = 0.0f32;
            for b in 0..blocks {
                let blk_slice = &weight[b * block_size..(b + 1) * block_size];
                sum += self.dot_q4_k(blk_slice, &in_f32[b*256..(b+1)*256]);
            }
            E::from_f32(sum * scale)
        } else {
            let in_f32 = elem_to_f32_vec(input);
            let mut sum = 0.0f32;
            for b in 0..blocks {
                let blk_slice = &weight[b * block_size..(b + 1) * block_size];
                sum += self.dot_q4_k(blk_slice, &in_f32[b*256..(b+1)*256]);
            }
            E::from_f32(sum * scale)
        }
    }

    fn gemv_q8(&self, weight: &[i8], input: &[E], scale: f32, n: usize) -> E {
        assert!(n % 256 == 0);
        let blocks = n / 256;
        let block_size = std::mem::size_of::<crate::quant::BlockQ8K>();
        let w_u8 = unsafe { std::slice::from_raw_parts(weight.as_ptr() as *const u8, weight.len()) };
        if let Some(in_f32) = E::as_f32_slice(input) {
            let mut sum = 0.0f32;
            for b in 0..blocks {
                let blk_slice = &w_u8[b * block_size..(b + 1) * block_size];
                sum += self.dot_q8_k(blk_slice, &in_f32[b*256..(b+1)*256]);
            }
            E::from_f32(sum * scale)
        } else {
            let in_f32 = elem_to_f32_vec(input);
            let mut sum = 0.0f32;
            for b in 0..blocks {
                let blk_slice = &w_u8[b * block_size..(b + 1) * block_size];
                sum += self.dot_q8_k(blk_slice, &in_f32[b*256..(b+1)*256]);
            }
            E::from_f32(sum * scale)
        }
    }

    fn gemm_q4(&self, weight: &[u8], input: &[E], output: &mut [E], scales: &[f32], m: usize, n: usize, k: usize) {
        assert!(k % 256 == 0);
        let blocks_per_row = k / 256;
        let block_size = std::mem::size_of::<crate::quant::BlockQ4K>();
        let row_stride = blocks_per_row * block_size;
        let per_channel = scales.len() == n;
        let per_tensor = scales.len() == 1;

        let in_f32: Vec<f32> = if let Some(f) = E::as_f32_slice(input) {
            f.to_vec()
        } else {
            input.iter().map(|v| v.to_f32()).collect()
        };

        for i in 0..m {
            let in_row = &in_f32[i*k..(i+1)*k];
            for j in 0..n {
                let w_start = j * row_stride;
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk_slice = &weight[w_start + b * block_size..w_start + (b + 1) * block_size];
                    sum += self.dot_q4_k(blk_slice, &in_row[b*256..(b+1)*256]);
                }
                let scale = if per_channel { scales[j] } else if per_tensor { scales[0] } else { 1.0 };
                if let Some(of) = E::as_f32_slice_mut(output) {
                    of[i*n + j] = sum * scale;
                } else {
                    output[i*n + j] = E::from_f32(sum * scale);
                }
            }
        }
    }

    fn gemm_q8(&self, weight: &[i8], input: &[E], output: &mut [E], scales: &[f32], m: usize, n: usize, k: usize) {
        assert!(k % 256 == 0);
        let blocks_per_row = k / 256;
        let block_size = std::mem::size_of::<crate::quant::BlockQ8K>();
        let row_stride = blocks_per_row * block_size;
        let w_u8 = unsafe { std::slice::from_raw_parts(weight.as_ptr() as *const u8, weight.len()) };
        let per_channel = scales.len() == n;
        let per_tensor = scales.len() == 1;

        let in_f32: Vec<f32> = if let Some(f) = E::as_f32_slice(input) {
            f.to_vec()
        } else {
            input.iter().map(|v| v.to_f32()).collect()
        };

        for i in 0..m {
            let in_row = &in_f32[i*k..(i+1)*k];
            for j in 0..n {
                let w_start = j * row_stride;
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk_slice = &w_u8[w_start + b * block_size..w_start + (b + 1) * block_size];
                    sum += self.dot_q8_k(blk_slice, &in_row[b*256..(b+1)*256]);
                }
                let scale = if per_channel { scales[j] } else if per_tensor { scales[0] } else { 1.0 };
                if let Some(of) = E::as_f32_slice_mut(output) {
                    of[i*n + j] = sum * scale;
                } else {
                    output[i*n + j] = E::from_f32(sum * scale);
                }
            }
        }
    }

    // Fused operators
    fn fused_qkv_rope(
        &self, input: &[E],
        wq: &[E], wk: &[E], wv: &[E],
        cos: &[E], sin: &[E],
        q_out: &mut [E], k_out: &mut [E], v_out: &mut [E],
        seq_len: usize, _hidden_size: usize,
        _num_heads: usize, _num_kv_heads: usize, head_dim: usize,
        _rotary_dim: usize, interleaved: bool,
    ) {
        let hidden = input.len() / seq_len;
        let q_cols = wq.len() / hidden;
        let k_cols = wk.len() / hidden;
        let v_cols = wv.len() / hidden;
        self.gemm(input, wq, q_out, seq_len, q_cols, hidden);
        self.gemm(input, wk, k_out, seq_len, k_cols, hidden);
        self.gemm(input, wv, v_out, seq_len, v_cols, hidden);
        self.rope(q_out, cos, sin, head_dim, interleaved);
        self.rope(k_out, cos, sin, head_dim, interleaved);
    }

    fn fused_gate_up_swiglu(
        &self, input: &[E], gate_weight: &[E], up_weight: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize,
    ) {
        assert_eq!(input.len(), seq_len * hidden_size);
        assert_eq!(output.len(), seq_len * ffn_dim);
        assert_eq!(gate_weight.len(), hidden_size * ffn_dim);
        let mut gate_out = vec![E::ZERO; seq_len * ffn_dim];
        let mut up_out = vec![E::ZERO; seq_len * ffn_dim];
        self.gemm(input, gate_weight, &mut gate_out, seq_len, ffn_dim, hidden_size);
        self.gemm(input, up_weight, &mut up_out, seq_len, ffn_dim, hidden_size);
        self.swiglu(&gate_out, &up_out, output);
    }

    fn fused_ffn_q4(
        &self, input: &[E],
        gate: &[u8], up: &[u8], down: &[u8],
        gate_scales: &[f32], up_scales: &[f32], down_scales: &[f32],
        residual: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize,
    ) {
        // gate_out = input × gate (quantized), up_out = input × up (quantized)
        // intermediate = SiLU(gate_out) * up_out
        // output = intermediate × down (quantized) + residual
        let mut gate_out = vec![E::ZERO; seq_len * ffn_dim];
        let mut up_out = vec![E::ZERO; seq_len * ffn_dim];
        self.gemm_q4(gate, input, &mut gate_out, gate_scales, seq_len, ffn_dim, hidden_size);
        self.gemm_q4(up, input, &mut up_out, up_scales, seq_len, ffn_dim, hidden_size);
        let mut intermediate = vec![E::ZERO; seq_len * ffn_dim];
        self.swiglu(&gate_out, &up_out, &mut intermediate);
        let mut down_out = vec![E::ZERO; seq_len * hidden_size];
        self.gemm_q4(down, &intermediate, &mut down_out, down_scales, seq_len, hidden_size, ffn_dim);
        for i in 0..seq_len * hidden_size {
            output[i] = E::from_f32(down_out[i].to_f32() + residual[i].to_f32());
        }
    }

    // ========================================================================
    // IQ Dequantization (ISA-dispatched via crate::quant_primitive!)
    // ========================================================================

    // IQ scalar-only decode
    define_quant_decode_scalar!(dequant_iq1_s, iq1_s, BlockIQ1S);
    define_quant_decode_scalar!(dequant_iq1_m, iq1_m, BlockIQ1M);
    define_quant_decode_scalar!(dequant_iq2_xxs, iq2_xxs, BlockIQ2XXS);
    define_quant_decode_scalar!(dequant_iq2_xs, iq2_xs, BlockIQ2XS);
    define_quant_decode_scalar!(dequant_iq2_s, iq2_s, BlockIQ2S);
    define_quant_decode_scalar!(dequant_iq3_xxs, iq3_xxs, BlockIQ3XXS);
    define_quant_decode_scalar!(dequant_iq3_s, iq3_s, BlockIQ3S);
    define_quant_decode_scalar!(dequant_iq4_nl, iq4_nl, BlockIQ4NL);
    define_quant_decode_scalar!(dequant_iq4_xs, iq4_xs, BlockIQ4XS);

    // ========================================================================
    // AWQ/GPTQ/Squeeze Dequantization + Dot
    // ========================================================================

    fn dequant_awq4(&self, packed: &[u8], zeros: &[u8], scales: &[half::f16], out: &mut [f32]) {
        // AWQ4: group_size=128, dequantized[i] = (nibble - zeros[group]) * scales[group]
        let blk = unsafe { &*(packed.as_ptr() as *const crate::quant::BlockAWQ4) };
        let group_size = 128usize;
        for w in 0..32 {
            let word = blk.qweight[w];
            for nib in 0..8 {
                let idx = w * 8 + nib;
                let group = idx / group_size;
                let q = ((word >> (nib * 4)) & 0xF) as f32;
                let zero = if group < zeros.len() { zeros[group] as f32 } else { 8.0 };
                let scale = if group < scales.len() { scales[group].to_f32() } else { blk.scales.to_f32() };
                out[idx] = (q - zero) * scale;
            }
        }
    }
    fn dequant_gptq4(&self, packed: &[u8], g_idx: &[i32], scales: &[half::f16], out: &mut [f32]) {
        // GPTQ4: dequantized[i] = (nibble - 8) * scales[g_idx[i]]
        let blk = unsafe { &*(packed.as_ptr() as *const crate::quant::BlockGPTQ4) };
        for w in 0..32 {
            let word = blk.qweight[w];
            for nib in 0..8 {
                let idx = w * 8 + nib;
                let q = ((word >> (nib * 4)) & 0xF) as f32;
                let group = if idx < g_idx.len() { g_idx[idx] as usize } else { 0 };
                let scale = if group < scales.len() { scales[group].to_f32() } else { blk.scales.to_f32() };
                out[idx] = (q - 8.0) * scale;
            }
        }
    }
    fn dequant_squeeze(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockSqueeze;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, squeeze, decode, blk, dst);
    }

    // ========================================================================
    // Position encoding: rope_with_pos
    // ========================================================================

    fn rope_with_pos(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, position: usize, _interleaved: bool) {
        let half = head_dim / 2;
        let seq_len = qk.len() / head_dim;
        for pos in 0..seq_len {
            let actual_pos = pos + position;
            let base = pos * head_dim;
            for i in 0..half {
                let x0 = qk[base + i];
                let x1 = qk[base + i + half];
                let c = cos[actual_pos * half + i];
                let s = sin[actual_pos * half + i];
                qk[base + i] = E::from_f32(x0.to_f32() * c.to_f32() - x1.to_f32() * s.to_f32());
                qk[base + i + half] = E::from_f32(x0.to_f32() * s.to_f32() + x1.to_f32() * c.to_f32());
            }
        }
    }

    // ========================================================================
    // Quantized GEMV: gemv_q2, gemv_q1
    // ========================================================================

    fn gemv_q2(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        let mut sum = 0.0f32;
        for i in 0..n {
            let byte_idx = i / 4;
            let shift = (i % 4) * 2;
            let q = ((weight[byte_idx] >> shift) & 0x03) as f32;
            sum += (scale * (q - 1.5)) * input[i].to_f32();
        }
        E::from_f32(sum)
    }

    fn gemv_q1(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        let mut sum = 0.0f32;
        for i in 0..n {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let q = (weight[byte_idx] >> bit_idx) & 1;
            let val = if q == 0 { -1.0f32 } else { 1.0f32 };
            sum += (scale * val) * input[i].to_f32();
        }
        E::from_f32(sum)
    }

    // ========================================================================
    // FP Fused operators (SPEC §2.3)
    // ========================================================================

    fn fused_ffn(
        &self, input: &[E],
        gate_weight: &[E], up_weight: &[E], down_weight: &[E],
        residual: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize,
    ) {
        let mut gate_out = vec![E::ZERO; seq_len * ffn_dim];
        let mut up_out = vec![E::ZERO; seq_len * ffn_dim];
        self.gemm(input, gate_weight, &mut gate_out, seq_len, ffn_dim, hidden_size);
        self.gemm(input, up_weight, &mut up_out, seq_len, ffn_dim, hidden_size);
        let mut intermediate = vec![E::ZERO; seq_len * ffn_dim];
        self.swiglu(&gate_out, &up_out, &mut intermediate);
        let mut down_out = vec![E::ZERO; seq_len * hidden_size];
        self.gemm(&intermediate, down_weight, &mut down_out, seq_len, hidden_size, ffn_dim);
        for i in 0..output.len() {
            output[i] = E::from_f32(down_out[i].to_f32() + residual[i].to_f32());
        }
    }

    fn fused_linear_residual_rmsnorm(
        &self, input: &[E], weight: &[E],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    ) {
        let mut linear_out = vec![E::ZERO; seq_len * out_features];
        self.gemm(input, weight, &mut linear_out, seq_len, out_features, in_features);
        let mut with_residual = vec![E::ZERO; seq_len * out_features];
        for i in 0..with_residual.len() {
            with_residual[i] = E::from_f32(linear_out[i].to_f32() + residual[i].to_f32());
        }
        for s in 0..seq_len {
            let row = &with_residual[s * out_features..(s + 1) * out_features];
            let out_row = &mut output[s * out_features..(s + 1) * out_features];
            self.rms_norm(row, norm_weight, out_row, eps);
        }
    }

    fn flash_attention(
        &self, q: &[E], k: &[E], v: &[E], output: &mut [E],
        seq_len: usize, num_heads: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) {
        dispatch_flash_attention!(q, k, v, output, seq_len, num_heads, head_dim, scale, causal);
    }

    fn flash_attention_paged(
        &self, q: &[E], k_cache: &[E], v_cache: &[E],
        page_table: &[usize], output: &mut [E],
        seq_len: usize, cache_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        page_size: usize, scale: f32,
    ) {
        dispatch_flash_attention_paged!(q, k_cache, v_cache, page_table, output, seq_len, cache_len, num_heads, num_kv_heads, head_dim, page_size, scale);
    }

    fn fused_ffn_rmsnorm(
        &self, input: &[E],
        gate_weight: &[E], up_weight: &[E], down_weight: &[E],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize, eps: f32,
    ) {
        let mut ffn_out = vec![E::ZERO; seq_len * hidden_size];
        self.fused_ffn(input, gate_weight, up_weight, down_weight, residual, &mut ffn_out, seq_len, hidden_size, ffn_dim);
        for s in 0..seq_len {
            let row = &ffn_out[s * hidden_size..(s + 1) * hidden_size];
            let out_row = &mut output[s * hidden_size..(s + 1) * hidden_size];
            self.rms_norm(row, norm_weight, out_row, eps);
        }
    }

    fn fused_linear_bias_residual_rmsnorm(
        &self, input: &[E], weight: &[E], bias: &[E],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    ) {
        let mut linear_out = vec![E::ZERO; seq_len * out_features];
        self.gemm(input, weight, &mut linear_out, seq_len, out_features, in_features);
        let mut with_residual = vec![E::ZERO; seq_len * out_features];
        for i in 0..with_residual.len() {
            let s = i / out_features;
            let j = i % out_features;
            let _ = s;
            with_residual[i] = E::from_f32(linear_out[i].to_f32() + bias[j].to_f32() + residual[i].to_f32());
        }
        for s in 0..seq_len {
            let row = &with_residual[s * out_features..(s + 1) * out_features];
            let out_row = &mut output[s * out_features..(s + 1) * out_features];
            self.rms_norm(row, norm_weight, out_row, eps);
        }
    }

    // ========================================================================
    // Quantized fused operators (SPEC §2.3)
    // ========================================================================

    fn fused_qkv_rope_q4(
        &self, input: &[E],
        wq: &[u8], wk: &[u8], wv: &[u8],
        scales_q: &[f32], scales_k: &[f32], scales_v: &[f32],
        cos: &[E], sin: &[E],
        q_out: &mut [E], k_out: &mut [E], v_out: &mut [E],
        seq_len: usize, hidden_size: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        rotary_dim: usize, interleaved: bool,
    ) {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        self.gemm_q4(wq, input, q_out, scales_q, seq_len, q_dim, hidden_size);
        self.gemm_q4(wk, input, k_out, scales_k, seq_len, kv_dim, hidden_size);
        self.gemm_q4(wv, input, v_out, scales_v, seq_len, kv_dim, hidden_size);
        let _ = rotary_dim;
        self.rope(q_out, cos, sin, head_dim, interleaved);
        self.rope(k_out, cos, sin, head_dim, interleaved);
    }

    fn fused_dequant_gemv(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: crate::quant::QuantType, m: usize, _n: usize, k: usize,
    ) {
        use crate::quant::QuantType;
        // Match once at entry, dispatch to format-specific loop (no match in hot path)
        match quant_type {
            QuantType::Q2K => self.fused_dequant_gemv_inner::<84, 256>(weight_blocks, input, output, m, k, Self::dot_q2_k),
            QuantType::Q3K => self.fused_dequant_gemv_inner::<110, 256>(weight_blocks, input, output, m, k, Self::dot_q3_k),
            QuantType::Q4K => self.fused_dequant_gemv_inner::<144, 256>(weight_blocks, input, output, m, k, Self::dot_q4_k),
            QuantType::Q5K => self.fused_dequant_gemv_inner::<176, 256>(weight_blocks, input, output, m, k, Self::dot_q5_k),
            QuantType::Q6K => self.fused_dequant_gemv_inner::<210, 256>(weight_blocks, input, output, m, k, Self::dot_q6_k),
            QuantType::Q8K => self.fused_dequant_gemv_inner::<292, 256>(weight_blocks, input, output, m, k, Self::dot_q8_k),
            QuantType::IQ1S => self.fused_dequant_gemv_inner::<50, 256>(weight_blocks, input, output, m, k, Self::dot_iq1_s),
            QuantType::IQ1M => self.fused_dequant_gemv_inner::<56, 256>(weight_blocks, input, output, m, k, Self::dot_iq1_m),
            QuantType::IQ2XXS => self.fused_dequant_gemv_inner::<66, 256>(weight_blocks, input, output, m, k, Self::dot_iq2_xxs),
            QuantType::IQ2XS => self.fused_dequant_gemv_inner::<74, 256>(weight_blocks, input, output, m, k, Self::dot_iq2_xs),
            QuantType::IQ2S => self.fused_dequant_gemv_inner::<82, 256>(weight_blocks, input, output, m, k, Self::dot_iq2_s),
            QuantType::IQ3XXS => self.fused_dequant_gemv_inner::<98, 256>(weight_blocks, input, output, m, k, Self::dot_iq3_xxs),
            QuantType::IQ3S => self.fused_dequant_gemv_inner::<110, 256>(weight_blocks, input, output, m, k, Self::dot_iq3_s),
            QuantType::IQ4NL => self.fused_dequant_gemv_inner::<18, 32>(weight_blocks, input, output, m, k, Self::dot_iq4_nl),
            QuantType::IQ4XS => self.fused_dequant_gemv_inner::<136, 256>(weight_blocks, input, output, m, k, Self::dot_iq4_xs),
            QuantType::AWQ4 => self.fused_dequant_gemv_inner::<72, 128>(weight_blocks, input, output, m, k, Self::dot_awq4),
            QuantType::GPTQ4 => self.fused_dequant_gemv_inner::<72, 128>(weight_blocks, input, output, m, k, Self::dot_gptq4),
            _ => unimplemented!("unsupported quant type for fused_dequant_gemv"),
        }
    }

    fn fused_int8_linear_residual_rmsnorm(
        &self, input: &[E], weight: &[i8], scales: &[f32],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    ) {
        let mut linear_out = vec![E::ZERO; seq_len * out_features];
        self.gemm_q8(weight, input, &mut linear_out, scales, seq_len, out_features, in_features);
        let mut with_residual = vec![E::ZERO; seq_len * out_features];
        for i in 0..with_residual.len() {
            with_residual[i] = E::from_f32(linear_out[i].to_f32() + residual[i].to_f32());
        }
        for s in 0..seq_len {
            let row = &with_residual[s * out_features..(s + 1) * out_features];
            let out_row = &mut output[s * out_features..(s + 1) * out_features];
            self.rms_norm(row, norm_weight, out_row, eps);
        }
    }

    fn fused_int4_linear_residual_rmsnorm(
        &self, input: &[E], weight: &[u8], scales: &[f32],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    ) {
        let mut linear_out = vec![E::ZERO; seq_len * out_features];
        self.gemm_q4(weight, input, &mut linear_out, scales, seq_len, out_features, in_features);
        let mut with_residual = vec![E::ZERO; seq_len * out_features];
        for i in 0..with_residual.len() {
            with_residual[i] = E::from_f32(linear_out[i].to_f32() + residual[i].to_f32());
        }
        for s in 0..seq_len {
            let row = &with_residual[s * out_features..(s + 1) * out_features];
            let out_row = &mut output[s * out_features..(s + 1) * out_features];
            self.rms_norm(row, norm_weight, out_row, eps);
        }
    }

    // ========================================================================
    // Quantized format-specific matmul (SPEC §2.3)
    // ========================================================================

    fn kquant_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: crate::quant::QuantType, m: usize, n: usize, k: usize,
    ) {
        use crate::quant::QuantType;
        // Match once at entry, dispatch to format-specific loop (no match in hot path)
        match quant_type {
            QuantType::Q2K => self.quant_matmul_inner::<84, 256>(weight_blocks, input, output, m, n, k, Self::dot_q2_k),
            QuantType::Q3K => self.quant_matmul_inner::<110, 256>(weight_blocks, input, output, m, n, k, Self::dot_q3_k),
            QuantType::Q4K => self.quant_matmul_inner::<144, 256>(weight_blocks, input, output, m, n, k, Self::dot_q4_k),
            QuantType::Q5K => self.quant_matmul_inner::<176, 256>(weight_blocks, input, output, m, n, k, Self::dot_q5_k),
            QuantType::Q6K => self.quant_matmul_inner::<210, 256>(weight_blocks, input, output, m, n, k, Self::dot_q6_k),
            QuantType::Q8K => self.quant_matmul_inner::<292, 256>(weight_blocks, input, output, m, n, k, Self::dot_q8_k),
            _ => unimplemented!("unsupported quant type for kquant_matmul"),
        }
    }

    fn iq_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: crate::quant::QuantType, m: usize, n: usize, k: usize,
    ) {
        use crate::quant::QuantType;
        // Match once at entry, dispatch to format-specific loop (no match in hot path)
        match quant_type {
            QuantType::IQ1S => self.quant_matmul_inner::<50, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq1_s),
            QuantType::IQ1M => self.quant_matmul_inner::<56, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq1_m),
            QuantType::IQ2XXS => self.quant_matmul_inner::<66, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq2_xxs),
            QuantType::IQ2XS => self.quant_matmul_inner::<74, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq2_xs),
            QuantType::IQ2S => self.quant_matmul_inner::<82, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq2_s),
            QuantType::IQ3XXS => self.quant_matmul_inner::<98, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq3_xxs),
            QuantType::IQ3S => self.quant_matmul_inner::<110, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq3_s),
            QuantType::IQ4NL => self.quant_matmul_inner::<18, 32>(weight_blocks, input, output, m, n, k, Self::dot_iq4_nl),
            QuantType::IQ4XS => self.quant_matmul_inner::<136, 256>(weight_blocks, input, output, m, n, k, Self::dot_iq4_xs),
            _ => unimplemented!("unsupported quant type for iq_matmul"),
        }
    }

    fn awq_matmul(
        &self, weight: &[u8], zeros: &[u8], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        // AWQ4: group_size=128, dequantized[i] = (nibble - zeros[group]) * scales[group]
        let group_size = 128usize;
        let num_groups_per_row = k / group_size;
        // Pre-allocate column buffer once
        let mut in_f32_col = vec![0.0f32; k];
        for j in 0..n {
            for p in 0..k { in_f32_col[p] = input[p * n + j].to_f32(); }
            for i in 0..m {
                let mut sum = 0.0f32;
                let row_offset = i * k;
                for idx in 0..k {
                    let byte_pos = row_offset / 2 + idx / 2;
                    let nibble = if idx % 2 == 0 {
                        (weight[byte_pos] & 0x0F) as f32
                    } else {
                        (weight[byte_pos] >> 4) as f32
                    };
                    let group = (i * num_groups_per_row) + idx / group_size;
                    let zero = if group < zeros.len() { zeros[group] as f32 } else { 8.0 };
                    let scale = if group < scales.len() { scales[group].to_f32() } else { 1.0 };
                    sum += (nibble - zero) * scale * in_f32_col[idx];
                }
                output[i * n + j] = E::from_f32(sum);
            }
        }
    }

    fn gptq_matmul(
        &self, weight: &[u8], g_idx: &[i32], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        // GPTQ4: dequantized[i] = (nibble - 8) * scales[g_idx[i]]
        let mut in_f32_col = vec![0.0f32; k];
        for j in 0..n {
            for p in 0..k { in_f32_col[p] = input[p * n + j].to_f32(); }
            for i in 0..m {
                let mut sum = 0.0f32;
                let row_offset = i * k;
                for idx in 0..k {
                    let byte_pos = row_offset / 2 + idx / 2;
                    let nibble = if idx % 2 == 0 {
                        (weight[byte_pos] & 0x0F) as f32
                    } else {
                        (weight[byte_pos] >> 4) as f32
                    };
                    let group = if idx < g_idx.len() { g_idx[idx] as usize } else { 0 };
                    let scale = if group < scales.len() { scales[group].to_f32() } else { 1.0 };
                    sum += (nibble - 8.0) * scale * in_f32_col[idx];
                }
                output[i * n + j] = E::from_f32(sum);
            }
        }
    }

    fn squeeze_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        let block_size = 256usize;
        let block_bytes = 130usize;
        let blocks_per_row = k / block_size;
        let mut in_f32_col = vec![0.0f32; k];
        for j in 0..n {
            for p in 0..k { in_f32_col[p] = input[p * n + j].to_f32(); }
            for i in 0..m {
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let off = i * blocks_per_row * block_bytes + b * block_bytes;
                    let blk = &weight_blocks[off..off + block_bytes];
                    let blk_ptr = blk.as_ptr() as *const crate::quant::BlockSqueeze;
                    let src = in_f32_col[b * block_size..(b + 1) * block_size].as_ptr();
                    sum += crate::quant_primitive!(scalar, squeeze, dot, blk_ptr, src);
                }
                output[i * n + j] = E::from_f32(sum);
            }
        }
    }

    fn fused_iq1_s_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        self.iq_matmul(weight_blocks, input, output, crate::quant::QuantType::IQ1S, m, n, k);
    }

    fn fused_iq2_xxs_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        self.iq_matmul(weight_blocks, input, output, crate::quant::QuantType::IQ2XXS, m, n, k);
    }

    fn fused_awq4_matmul(
        &self, weight: &[u8], zeros: &[u8], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        self.awq_matmul(weight, zeros, scales, input, output, m, n, k);
    }

    fn fused_gptq4_matmul(
        &self, weight: &[u8], g_idx: &[i32], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        self.gptq_matmul(weight, g_idx, scales, input, output, m, n, k);
    }

    fn fused_squeeze_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        self.squeeze_matmul(weight_blocks, input, output, m, n, k);
    }
}
