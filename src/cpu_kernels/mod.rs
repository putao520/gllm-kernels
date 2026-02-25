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
    /// AVX-512 with native FP16 support (Sapphire Rapids+).
    /// 32 x f16 per register, native FMA/add/mul — 2x throughput vs F16C conversion path.
    Avx512Fp16,
    Neon,
}

static ISA_LEVEL: OnceLock<IsaLevel> = OnceLock::new();

pub fn get_isa_level() -> IsaLevel {
    *ISA_LEVEL.get_or_init(detect_isa_features)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_isa_features() -> IsaLevel {
    if is_x86_feature_detected!("avx512fp16") {
        IsaLevel::Avx512Fp16
    } else if is_x86_feature_detected!("avx512f") {
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
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, dot, blk, src) }
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
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, dot, blk, src) }
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
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, decode, blk, dst); }
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
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, decode, blk, dst); }
                _ => { crate::quant_primitive!(scalar, $qfmt, decode, blk, dst); }
            }
        }
    };
}

/// Generate a dequant function: avx512 + avx2 + neon + scalar (IQ1-IQ3 decode, full ISA coverage)
macro_rules! define_quant_decode_iq {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], out: &mut [f32]) {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let dst = out.as_mut_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, decode, blk, dst); }
                _ => { crate::quant_primitive!(scalar, $qfmt, decode, blk, dst); }
            }
        }
    };
}

/// Generate a dot function: avx512 + avx2 + neon + scalar (IQ4 formats with full ISA coverage)
macro_rules! define_quant_dot_iq4 {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], other: &[f32]) -> f32 {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let src = other.as_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, dot, blk, src) }
                _ => { crate::quant_primitive!(scalar, $qfmt, dot, blk, src) }
            }
        }
    };
}

/// Generate a dequant function: avx512 + avx2 + scalar (IQ4 decode with full x86 coverage)
macro_rules! define_quant_decode_iq4 {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], out: &mut [f32]) {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let dst = out.as_mut_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, decode, blk, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, decode, blk, dst); }
                _ => { crate::quant_primitive!(scalar, $qfmt, decode, blk, dst); }
            }
        }
    };
}

/// Generate a dot function: avx512 + avx2 + neon + scalar (commercial formats AWQ4/GPTQ4)
macro_rules! define_quant_dot_commercial {
    ($fn_name:ident, $qfmt:ident, $block_ty:ident) => {
        fn $fn_name(&self, block: &[u8], other: &[f32]) -> f32 {
            let blk = block.as_ptr() as *const crate::quant::$block_ty;
            let src = other.as_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => { crate::quant_primitive!(avx512, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, $qfmt, dot, blk, src) }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, $qfmt, dot, blk, src) }
                _ => { crate::quant_primitive!(scalar, $qfmt, dot, blk, src) }
            }
        }
    };
}

// ============================================================================
// Private helper methods on CpuKernels
// ============================================================================

impl<E: Element> CpuKernels<E> {
    // Classic GGML dot: avx512 + avx2 + neon + scalar (block_size=32)
    define_quant_dot_k!(dot_q4_0, q4_0, BlockQ4_0);
    define_quant_dot_k!(dot_q4_1, q4_1, BlockQ4_1);
    define_quant_dot_k!(dot_q5_0, q5_0, BlockQ5_0);
    define_quant_dot_k!(dot_q5_1, q5_1, BlockQ5_1);
    define_quant_dot_k!(dot_q8_0, q8_0, BlockQ8_0);
    define_quant_dot_k!(dot_q8_1, q8_1, BlockQ8_1);

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

    // IQ4 dot: avx512 + avx2 + neon + scalar (full ISA coverage)
    define_quant_dot_iq4!(dot_iq4_nl, iq4_nl, BlockIQ4NL);
    define_quant_dot_iq4!(dot_iq4_xs, iq4_xs, BlockIQ4XS);

    // Commercial dot: avx512 + avx2 + neon + scalar
    define_quant_dot_commercial!(dot_awq4, awq4, BlockAWQ4);
    define_quant_dot_commercial!(dot_gptq4, gptq4, BlockGPTQ4);
    define_quant_dot_commercial!(dot_squeeze, squeeze, BlockSqueeze);

    /// SIMD-dispatched f32 dot product (used by AWQ/GPTQ matmul after dequant).
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => avx512::avx512_f32::dot(a, b).to_f32(),
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => avx2::avx2_f32::dot(a, b).to_f32(),
            #[cfg(target_arch = "aarch64")]
            IsaLevel::Neon => neon::neon_f32::dot(a, b).to_f32(),
            _ => scalar::scalar_f32::dot(a, b).to_f32(),
        }
    }

    // ========================================================================
    // Hot-path inner loops: const-generic block_bytes/block_size + fn pointer
    // Eliminates match quant_type inside the inner loop (SPEC §8 compliance)
    // ========================================================================

    #[inline(always)]
    fn quant_matmul_inner<const BLOCK_BYTES: usize, const BLOCK_SIZE: usize, F>(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
        dot_fn: F,
    ) where F: Fn(&Self, &[u8], &[f32]) -> f32 {
        let blocks_per_row = k / BLOCK_SIZE;
        // Thread-local buffer: avoids heap allocation on every call
        thread_local! {
            static INPUT_T: std::cell::Cell<crate::cache_params::AlignedVec<f32>>
                = std::cell::Cell::new(crate::cache_params::AlignedVec::new());
        }
        INPUT_T.with(|cell| {
            let mut buf = cell.take();
            buf.resize_zeroed(k * n);
            let input_t = buf.as_mut_slice();
            // Pre-transpose input from [k, n] col-major to [n, k] row-major
            for p in 0..k {
                for j in 0..n {
                    input_t[j * k + p] = input[p * n + j].to_f32();
                }
            }
            for j in 0..n {
                let in_row = &input_t[j * k..];
                for i in 0..m {
                    let mut sum = 0.0f32;
                    for b in 0..blocks_per_row {
                        let off = i * blocks_per_row * BLOCK_BYTES + b * BLOCK_BYTES;
                        let blk = &weight_blocks[off..off + BLOCK_BYTES];
                        let in_slice = &in_row[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                        sum += dot_fn(self, blk, in_slice);
                    }
                    output[i * n + j] = E::from_f32(sum);
                }
            }
            cell.set(buf);
        });
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
        #[allow(unused_unsafe)]
        unsafe { std::slice::from_raw_parts($slice.as_ptr() as *const $target, $slice.len()) }
    };
}

macro_rules! as_typed_slice_mut {
    ($slice:expr, $target:ty) => {
        #[allow(unused_unsafe)]
        unsafe { std::slice::from_raw_parts_mut($slice.as_mut_ptr() as *mut $target, $slice.len()) }
    };
}

macro_rules! dispatch_binary_op {
    ($a:expr, $b:expr, $out:expr, $op:ident) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16)),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice_mut!($out, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice_mut!($out, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice_mut!($out, half::bf16)),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16)).to_f32(),
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

/// Dispatch for dot product: 2 read slices → scalar (single pass FMA + reduce)
macro_rules! dispatch_dot_op {
    ($a:expr, $b:expr, $op:ident) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16)).to_f32(),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16)).to_f32(),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32)).to_f32(),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16)).to_f32(),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16)).to_f32(),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32)).to_f32(),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16)).to_f32(),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16)).to_f32(),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(half::f16::from_f32($scalar), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op($scalar, as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(half::f16::from_f32($scalar), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(half::bf16::from_f32($scalar), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16)),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($w, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($w, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($w, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($w, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($g, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($x, f32), as_typed_slice!($g, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($out, f32), $eps),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($x, half::f16), as_typed_slice!($g, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($out, half::f16), half::f16::from_f32($eps)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($x, half::bf16), as_typed_slice!($g, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($out, half::bf16), half::bf16::from_f32($eps)),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($x, f32), as_typed_slice_mut!($y, f32), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($x, half::f16), as_typed_slice_mut!($y, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($x, half::bf16), as_typed_slice_mut!($y, half::bf16), $m, $n),
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
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
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

/// Dispatch streaming GEMV (M=1): gemv_streaming(a, b, c, n_size, k_size)
macro_rules! dispatch_gemv_streaming {
    ($a:expr, $b:expr, $c:expr, $n:expr, $k:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::gemv_streaming(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::gemv_streaming(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::gemv_streaming(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::gemv_streaming(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::gemv_streaming(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::gemv_streaming(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::gemv_streaming(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::gemv_streaming(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::gemv_streaming(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::gemv_streaming(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $n, $k),
            (_, 0) => scalar::scalar_f32::gemv_streaming(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $n, $k),
            (_, 1) => scalar::scalar_f16::gemv_streaming(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $n, $k),
            (_, 2) => scalar::scalar_bf16::gemv_streaming(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $n, $k),
            _ => unreachable!(),
        }
    };
}

// Dispatch skinny GEMM (M=2..32): gemm_skinny(a, b, c, m, n, k)
macro_rules! dispatch_gemm_skinny {
    ($a:expr, $b:expr, $c:expr, $m:expr, $n:expr, $k:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::gemm_skinny(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::gemm_skinny(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::gemm_skinny(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::gemm_skinny(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::gemm_skinny(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::gemm_skinny(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::gemm_skinny(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::gemm_skinny(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::gemm_skinny(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::gemm_skinny(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            (_, 0) => scalar::scalar_f32::gemm_skinny(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            (_, 1) => scalar::scalar_f16::gemm_skinny(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            (_, 2) => scalar::scalar_bf16::gemm_skinny(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            _ => unreachable!(),
        }
    };
}

// Dispatch skinny GEMM bt (M=2..32): gemm_skinny_bt(a, b_t, c, m, n, k)
macro_rules! dispatch_gemm_skinny_bt {
    ($a:expr, $b_t:expr, $c:expr, $m:expr, $n:expr, $k:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::gemm_skinny_bt(as_typed_slice!($a, half::f16), as_typed_slice!($b_t, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::gemm_skinny_bt(as_typed_slice!($a, f32), as_typed_slice!($b_t, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::gemm_skinny_bt(as_typed_slice!($a, half::f16), as_typed_slice!($b_t, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::gemm_skinny_bt(as_typed_slice!($a, half::bf16), as_typed_slice!($b_t, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::gemm_skinny_bt(as_typed_slice!($a, f32), as_typed_slice!($b_t, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::gemm_skinny_bt(as_typed_slice!($a, half::f16), as_typed_slice!($b_t, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::gemm_skinny_bt(as_typed_slice!($a, half::bf16), as_typed_slice!($b_t, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::gemm_skinny_bt(as_typed_slice!($a, f32), as_typed_slice!($b_t, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::gemm_skinny_bt(as_typed_slice!($a, half::f16), as_typed_slice!($b_t, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::gemm_skinny_bt(as_typed_slice!($a, half::bf16), as_typed_slice!($b_t, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            (_, 0) => scalar::scalar_f32::gemm_skinny_bt(as_typed_slice!($a, f32), as_typed_slice!($b_t, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            (_, 1) => scalar::scalar_f16::gemm_skinny_bt(as_typed_slice!($a, half::f16), as_typed_slice!($b_t, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            (_, 2) => scalar::scalar_bf16::gemm_skinny_bt(as_typed_slice!($a, half::bf16), as_typed_slice!($b_t, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for rope: 1 mut slice + 2 read slices + usize
macro_rules! dispatch_rope {
    ($op:ident, $qk:expr, $cos:expr, $sin:expr, $hd:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd),
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

/// Dispatch for rope_with_pos: 1 mut slice + 2 read slices + usize + usize
macro_rules! dispatch_rope_with_pos {
    ($op:ident, $qk:expr, $cos:expr, $sin:expr, $hd:expr, $pos:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd, $pos),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd, $pos),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd, $pos),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd, $pos),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd, $pos),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd, $pos),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd, $pos),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd, $pos),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd, $pos),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd, $pos),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice_mut!($qk, f32), as_typed_slice!($cos, f32), as_typed_slice!($sin, f32), $hd, $pos),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice_mut!($qk, half::f16), as_typed_slice!($cos, half::f16), as_typed_slice!($sin, half::f16), $hd, $pos),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice_mut!($qk, half::bf16), as_typed_slice!($cos, half::bf16), as_typed_slice!($sin, half::bf16), $hd, $pos),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for in-place scale: read slice + f32 scalar + mut out, then copy back
macro_rules! dispatch_scale {
    ($op:ident, $x:expr, $sf:expr, $tmp:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($x, half::f16), half::f16::from_f32($sf), as_typed_slice_mut!($tmp, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($x, f32), $sf, as_typed_slice_mut!($tmp, f32)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($x, half::f16), half::f16::from_f32($sf), as_typed_slice_mut!($tmp, half::f16)),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($x, half::bf16), half::bf16::from_f32($sf), as_typed_slice_mut!($tmp, half::bf16)),
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

/// Dispatch for bias_rows: 1 mut slice + 1 read slice + 2 dims
#[allow(unused_macros)]
macro_rules! dispatch_bias_rows {
    ($op:ident, $c:expr, $bias:expr, $m:expr, $n:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice_mut!($c, half::f16), as_typed_slice!($bias, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice_mut!($c, f32), as_typed_slice!($bias, f32), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice_mut!($c, half::f16), as_typed_slice!($bias, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice_mut!($c, half::bf16), as_typed_slice!($bias, half::bf16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice_mut!($c, f32), as_typed_slice!($bias, f32), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice_mut!($c, half::f16), as_typed_slice!($bias, half::f16), $m, $n),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice_mut!($c, half::bf16), as_typed_slice!($bias, half::bf16), $m, $n),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice_mut!($c, f32), as_typed_slice!($bias, f32), $m, $n),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice_mut!($c, half::f16), as_typed_slice!($bias, half::f16), $m, $n),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice_mut!($c, half::bf16), as_typed_slice!($bias, half::bf16), $m, $n),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice_mut!($c, f32), as_typed_slice!($bias, f32), $m, $n),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice_mut!($c, half::f16), as_typed_slice!($bias, half::f16), $m, $n),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice_mut!($c, half::bf16), as_typed_slice!($bias, half::bf16), $m, $n),
            _ => unreachable!(),
        }
    };
}

/// Dispatch for matmul_bias: 3 read slices + 1 mut slice + 3 dims (fused C = A*B + bias)
macro_rules! dispatch_matmul_bias {
    ($op:ident, $a:expr, $b:expr, $bias:expr, $c:expr, $m:expr, $n:expr, $k:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice!($bias, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice!($bias, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice!($bias, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice!($bias, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => neon::neon_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice!($bias, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => neon::neon_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => neon::neon_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice!($bias, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            (_, 0) => scalar::scalar_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice!($bias, f32), as_typed_slice_mut!($c, f32), $m, $n, $k),
            (_, 1) => scalar::scalar_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k),
            (_, 2) => scalar::scalar_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice!($bias, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k),
            _ => unreachable!(),
        }
    };
}

macro_rules! dispatch_matmul_bias_act {
    ($op:ident, $a:expr, $b:expr, $bias:expr, $c:expr, $m:expr, $n:expr, $k:expr, $act:expr) => {{
        let act = $act;
        if matches!(act, crate::Activation::None) {
            dispatch_matmul_bias!(matmul_bias, $a, $b, $bias, $c, $m, $n, $k);
            return;
        }
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => avx512::avx512fp16_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k, act),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => avx512::avx512_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice!($bias, f32), as_typed_slice_mut!($c, f32), $m, $n, $k, act),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => avx512::avx512_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k, act),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => avx512::avx512_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice!($bias, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k, act),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => avx2::avx2_f32::$op(as_typed_slice!($a, f32), as_typed_slice!($b, f32), as_typed_slice!($bias, f32), as_typed_slice_mut!($c, f32), $m, $n, $k, act),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => avx2::avx2_f16::$op(as_typed_slice!($a, half::f16), as_typed_slice!($b, half::f16), as_typed_slice!($bias, half::f16), as_typed_slice_mut!($c, half::f16), $m, $n, $k, act),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => avx2::avx2_bf16::$op(as_typed_slice!($a, half::bf16), as_typed_slice!($b, half::bf16), as_typed_slice!($bias, half::bf16), as_typed_slice_mut!($c, half::bf16), $m, $n, $k, act),
            // Fallback: matmul_bias + scalar activation in-place
            _ => {
                dispatch_matmul_bias!(matmul_bias, $a, $b, $bias, $c, $m, $n, $k);
                let len = $m * $n;
                match act {
                    crate::Activation::Relu => { for i in 0..len { if $c[i].to_f32() < 0.0 { $c[i] = E::ZERO; } } },
                    crate::Activation::Silu => { for i in 0..len { let v = $c[i].to_f32(); $c[i] = E::from_f32(v / (1.0 + (-v).exp())); } },
                    crate::Activation::Gelu => { for i in 0..len { let x = $c[i].to_f32(); let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x); $c[i] = E::from_f32(0.5 * x * (1.0 + inner.tanh())); } },
                    _ => {},
                }
            },
        }
    }};
}

/// Transmute Vec<$concrete> to Vec<E> (same size/alignment guaranteed by Element trait).
macro_rules! transmute_vec {
    ($vec:expr, $concrete:ty) => {
        unsafe {
            let mut v = $vec;
            let ptr = v.as_mut_ptr() as *mut E;
            let len = v.len();
            let cap = v.capacity();
            std::mem::forget(v);
            Vec::from_raw_parts(ptr, len, cap)
        }
    };
}

/// Dispatch for pack_b: returns Vec<E>
macro_rules! dispatch_pack_b {
    ($op:ident, $b:expr, $n:expr, $k:expr) => {
        match (get_isa_level(), E::ELEM_ID) {
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512Fp16, 1) => transmute_vec!(avx512::avx512fp16_f16::$op(as_typed_slice!($b, half::f16), $n, $k), half::f16),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 0) => transmute_vec!(avx512::avx512_f32::$op(as_typed_slice!($b, f32), $n, $k), f32),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512, 1) => transmute_vec!(avx512::avx512_f16::$op(as_typed_slice!($b, half::f16), $n, $k), half::f16),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx512 | IsaLevel::Avx512Fp16, 2) => transmute_vec!(avx512::avx512_bf16::$op(as_typed_slice!($b, half::bf16), $n, $k), half::bf16),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 0) => transmute_vec!(avx2::avx2_f32::$op(as_typed_slice!($b, f32), $n, $k), f32),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 1) => transmute_vec!(avx2::avx2_f16::$op(as_typed_slice!($b, half::f16), $n, $k), half::f16),
            #[cfg(target_arch = "x86_64")]
            (IsaLevel::Avx2, 2) => transmute_vec!(avx2::avx2_bf16::$op(as_typed_slice!($b, half::bf16), $n, $k), half::bf16),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 0) => transmute_vec!(neon::neon_f32::$op(as_typed_slice!($b, f32), $n, $k), f32),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 1) => transmute_vec!(neon::neon_f16::$op(as_typed_slice!($b, half::f16), $n, $k), half::f16),
            #[cfg(target_arch = "aarch64")]
            (IsaLevel::Neon, 2) => transmute_vec!(neon::neon_bf16::$op(as_typed_slice!($b, half::bf16), $n, $k), half::bf16),
            (_, 0) => transmute_vec!(scalar::scalar_f32::$op(as_typed_slice!($b, f32), $n, $k), f32),
            (_, 1) => transmute_vec!(scalar::scalar_f16::$op(as_typed_slice!($b, half::f16), $n, $k), half::f16),
            (_, 2) => transmute_vec!(scalar::scalar_bf16::$op(as_typed_slice!($b, half::bf16), $n, $k), half::bf16),
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
        E::from_f32(dispatch_dot_op!(a, b, dot))
    }

    fn vec_scale(&self, x: &mut [E], s: E) {
        // SAFETY: scale reads a[i] then writes out[i] per lane — no cross-lane
        // dependency, so aliasing the same buffer is safe and avoids a heap alloc.
        let alias: &[E] = unsafe { std::slice::from_raw_parts(x.as_ptr(), x.len()) };
        let sf = s.to_f32();
        dispatch_scale!(scale, alias, sf, x);
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
        // M=1: streaming GEMV — zero-packing, bandwidth-optimal for LLM decode
        if m == 1 {
            dispatch_gemv_streaming!(a, b, c, n, k);
            return;
        }
        // M=2..32: pack-free skinny GEMM — no packing overhead, register-tiled
        if m <= 32 {
            dispatch_gemm_skinny!(a, b, c, m, n, k);
            return;
        }
        // NEON f32: route to hand-written asm microkernel for peak performance
        #[cfg(target_arch = "aarch64")]
        if E::ELEM_ID == 0 {
            // ELEM_ID 0 = f32
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            crate::asm::aarch64::gemm_asm_f32(a_f32, b_f32, c_f32, m, n, k);
            return;
        }
        // x86_64 f32: hand-written ASM microkernels for peak performance
        #[cfg(target_arch = "x86_64")]
        if E::ELEM_ID == 0 {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            match get_isa_level() {
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => {
                    crate::asm::x86_64::gemm_asm_f32_avx512(a_f32, b_f32, c_f32, m, n, k);
                }
                IsaLevel::Avx2 => {
                    crate::asm::x86_64::gemm_asm_f32_avx2(a_f32, b_f32, c_f32, m, n, k);
                }
                _ => {
                    // Scalar fallback: use intrinsics path
                    dispatch_with_dims!(matmul, a, b, c, m, n, k);
                }
            }
            return;
        }
        dispatch_with_dims!(matmul, a, b, c, m, n, k);
    }

    fn gemm_bt(&self, a: &[E], b_t: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        assert!(a.len() >= m * k);
        assert!(b_t.len() >= n * k);
        assert!(c.len() >= m * n);
        assert!(m >= 2 && m <= 32, "gemm_bt currently supports M=2..32");
        dispatch_gemm_skinny_bt!(a, b_t, c, m, n, k);
    }

    fn gemm_bias(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        assert!(c.len() == m * n && bias.len() == n);
        // x86_64 f32 M>32: hand-written ASM microkernels with fused bias
        #[cfg(target_arch = "x86_64")]
        if E::ELEM_ID == 0 && m > 32 {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let bias_f32 = unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, bias.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            match get_isa_level() {
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => {
                    crate::asm::x86_64::gemm_bias_asm_f32_avx512(a_f32, b_f32, bias_f32, c_f32, m, n, k);
                    return;
                }
                IsaLevel::Avx2 => {
                    crate::asm::x86_64::gemm_bias_asm_f32_avx2(a_f32, b_f32, bias_f32, c_f32, m, n, k);
                    return;
                }
                _ => {}
            }
        }
        dispatch_matmul_bias!(matmul_bias, a, b, bias, c, m, n, k);
    }

    fn gemm_bias_act(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize, act: crate::Activation) {
        assert!(c.len() == m * n && bias.len() == n);
        dispatch_matmul_bias_act!(matmul_bias_act, a, b, bias, c, m, n, k, act);
    }

    fn pack_b(&self, b: &[E], n: usize, k: usize) -> Vec<E> {
        // x86_64 f32: use ASM driver's pack_b format (NR-wide strips, full K contiguous)
        #[cfg(target_arch = "x86_64")]
        if E::ELEM_ID == 0 {
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let packed = match get_isa_level() {
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => {
                    crate::asm::x86_64::pack_b_asm_f32_avx512(b_f32, n, k)
                }
                IsaLevel::Avx2 => {
                    crate::asm::x86_64::pack_b_asm_f32_avx2(b_f32, n, k)
                }
                _ => {
                    return dispatch_pack_b!(pack_b, b, n, k);
                }
            };
            return transmute_vec!(packed, f32);
        }
        #[cfg(target_arch = "aarch64")]
        if E::ELEM_ID == 0 {
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let packed = crate::asm::aarch64::pack_b_asm_f32_neon(b_f32, n, k);
            return transmute_vec!(packed, f32);
        }
        dispatch_pack_b!(pack_b, b, n, k)
    }

    fn gemm_prepacked(&self, a: &[E], packed_b: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        // x86_64 f32: pack_b always produces ASM format, so prepacked must use ASM driver
        #[cfg(target_arch = "x86_64")]
        if E::ELEM_ID == 0 {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let pb_f32 = unsafe { std::slice::from_raw_parts(packed_b.as_ptr() as *const f32, packed_b.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            match get_isa_level() {
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => {
                    crate::asm::x86_64::gemm_prepacked_asm_f32_avx512(a_f32, pb_f32, c_f32, m, n, k);
                    return;
                }
                IsaLevel::Avx2 => {
                    crate::asm::x86_64::gemm_prepacked_asm_f32_avx2(a_f32, pb_f32, c_f32, m, n, k);
                    return;
                }
                _ => {}
            }
        }
        #[cfg(target_arch = "aarch64")]
        if E::ELEM_ID == 0 {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let pb_f32 = unsafe { std::slice::from_raw_parts(packed_b.as_ptr() as *const f32, packed_b.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            crate::asm::aarch64::gemm_prepacked_asm_f32(a_f32, pb_f32, c_f32, m, n, k);
            return;
        }
        dispatch_with_dims!(matmul_prepacked, a, packed_b, c, m, n, k);
    }

    fn gemm_bias_prepacked(&self, a: &[E], packed_b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        // x86_64 f32: pack_b always produces ASM format, so prepacked must use ASM driver
        #[cfg(target_arch = "x86_64")]
        if E::ELEM_ID == 0 {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let pb_f32 = unsafe { std::slice::from_raw_parts(packed_b.as_ptr() as *const f32, packed_b.len()) };
            let bias_f32 = unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, bias.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            match get_isa_level() {
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => {
                    crate::asm::x86_64::gemm_bias_prepacked_asm_f32_avx512(a_f32, pb_f32, bias_f32, c_f32, m, n, k);
                    return;
                }
                IsaLevel::Avx2 => {
                    crate::asm::x86_64::gemm_bias_prepacked_asm_f32_avx2(a_f32, pb_f32, bias_f32, c_f32, m, n, k);
                    return;
                }
                _ => {}
            }
        }
        #[cfg(target_arch = "aarch64")]
        if E::ELEM_ID == 0 {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let pb_f32 = unsafe { std::slice::from_raw_parts(packed_b.as_ptr() as *const f32, packed_b.len()) };
            let bias_f32 = unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, bias.len()) };
            let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
            crate::asm::aarch64::gemm_bias_prepacked_asm_f32(a_f32, pb_f32, bias_f32, c_f32, m, n, k);
            return;
        }
        dispatch_matmul_bias!(matmul_bias_prepacked, a, packed_b, bias, c, m, n, k);
    }

    // Activations
    fn silu(&self, a: &[E], out: &mut [E]) { dispatch_unary_op!(a, out, silu); }
    fn relu(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, relu); }
    fn gelu(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, gelu); }
    fn tanh(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, tanh); }
    fn exp(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, exp); }
    fn softmax(&self, x: &[E], out: &mut [E]) { dispatch_unary_op!(x, out, softmax); }

    fn swiglu(&self, gate: &[E], up: &[E], out: &mut [E]) {
        dispatch_binary_op!(gate, up, out, swiglu);
    }

    // Normalization
    fn rms_norm(&self, x: &[E], weight: &[E], out: &mut [E], eps: f32) {
        dispatch_with_eps!(rms_norm, x, weight, out, eps);
    }

    fn layer_norm(&self, x: &[E], gamma: &[E], beta: &[E], out: &mut [E], eps: f32) {
        dispatch_with_eps!(layer_norm, x, gamma, beta, out, eps);
    }

    // Positional
    fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, interleaved: bool) {
        if interleaved {
            dispatch_rope!(rope_interleaved, qk, cos, sin, head_dim);
        } else {
            dispatch_rope!(rope, qk, cos, sin, head_dim);
        }
    }

    // Sampling
    // Quantization
    // Classic GGML decode: avx512 + avx2 + neon + scalar (block_size=32)
    define_quant_decode_k!(dequant_q4_0, q4_0, BlockQ4_0);
    define_quant_decode_k!(dequant_q4_1, q4_1, BlockQ4_1);
    define_quant_decode_k!(dequant_q5_0, q5_0, BlockQ5_0);
    define_quant_decode_k!(dequant_q5_1, q5_1, BlockQ5_1);
    define_quant_decode_k!(dequant_q8_0, q8_0, BlockQ8_0);
    define_quant_decode_k!(dequant_q8_1, q8_1, BlockQ8_1);

    // K-Quant decode: avx512 + avx2 + scalar
    define_quant_decode_k!(dequant_q4_k, q4_k, BlockQ4K);
    define_quant_decode_k!(dequant_q3_k, q3_k, BlockQ3K);
    define_quant_decode_k!(dequant_q5_k, q5_k, BlockQ5K);
    define_quant_decode_k!(dequant_q6_k, q6_k, BlockQ6K);

    // Q2_K decode: avx512 + avx2 + neon + scalar (per-sub-block scales fixed)
    define_quant_decode_k!(dequant_q2_k, q2_k, BlockQ2K);
    // Q8_K decode: avx512 + avx2 + neon + scalar (full ISA coverage)
    define_quant_decode_k!(dequant_q8_k, q8_k, BlockQ8K);

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

    // ========================================================================
    // IQ Dequantization (ISA-dispatched via crate::quant_primitive!)
    // ========================================================================

    // IQ decode: avx2 + scalar (avx512 falls through to avx2)
    define_quant_decode_iq!(dequant_iq1_s, iq1_s, BlockIQ1S);
    define_quant_decode_iq!(dequant_iq1_m, iq1_m, BlockIQ1M);
    define_quant_decode_iq!(dequant_iq2_xxs, iq2_xxs, BlockIQ2XXS);
    define_quant_decode_iq!(dequant_iq2_xs, iq2_xs, BlockIQ2XS);
    define_quant_decode_iq!(dequant_iq2_s, iq2_s, BlockIQ2S);
    define_quant_decode_iq!(dequant_iq3_xxs, iq3_xxs, BlockIQ3XXS);
    define_quant_decode_iq!(dequant_iq3_s, iq3_s, BlockIQ3S);

    // IQ4 decode: avx512 + avx2 + scalar (full x86 coverage)
    define_quant_decode_iq4!(dequant_iq4_nl, iq4_nl, BlockIQ4NL);
    define_quant_decode_iq4!(dequant_iq4_xs, iq4_xs, BlockIQ4XS);

    // ========================================================================
    // AWQ/GPTQ/Squeeze Dequantization + Dot
    // ========================================================================

    fn dequant_awq4(&self, packed: &[u8], zeros: &[u8], scales: &[half::f16], out: &mut [f32]) {
        let blk = unsafe { &*(packed.as_ptr() as *const crate::quant::BlockAWQ4) };
        if zeros.is_empty() && scales.is_empty() {
            // Block-level scale only — use ISA-dispatched SIMD decode
            let blk_ptr = blk as *const crate::quant::BlockAWQ4;
            let dst = out.as_mut_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 => { crate::quant_primitive!(avx512, awq4, decode, blk_ptr, dst); }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, awq4, decode, blk_ptr, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, awq4, decode, blk_ptr, dst); }
                _ => { crate::quant_primitive!(scalar, awq4, decode, blk_ptr, dst); }
            }
        } else {
            // Per-group zeros/scales — scalar path
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
    }
    fn dequant_gptq4(&self, packed: &[u8], g_idx: &[i32], scales: &[half::f16], out: &mut [f32]) {
        let blk = unsafe { &*(packed.as_ptr() as *const crate::quant::BlockGPTQ4) };
        if g_idx.is_empty() && scales.is_empty() {
            // Block-level scale only — use ISA-dispatched SIMD decode
            let blk_ptr = blk as *const crate::quant::BlockGPTQ4;
            let dst = out.as_mut_ptr();
            match get_isa_level() {
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx512 => { crate::quant_primitive!(avx512, gptq4, decode, blk_ptr, dst); }
                #[cfg(target_arch = "x86_64")]
                IsaLevel::Avx2 => { crate::quant_primitive!(avx2, gptq4, decode, blk_ptr, dst); }
                #[cfg(target_arch = "aarch64")]
                IsaLevel::Neon => { crate::quant_primitive!(neon, gptq4, decode, blk_ptr, dst); }
                _ => { crate::quant_primitive!(scalar, gptq4, decode, blk_ptr, dst); }
            }
        } else {
            // Per-element g_idx — scalar path
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
    }
    fn dequant_squeeze(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockSqueeze;
        let dst = out.as_mut_ptr();
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => { crate::quant_primitive!(avx512, squeeze, decode, blk, dst); }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => { crate::quant_primitive!(avx2, squeeze, decode, blk, dst); }
            #[cfg(target_arch = "aarch64")]
            IsaLevel::Neon => { crate::quant_primitive!(neon, squeeze, decode, blk, dst); }
            _ => { crate::quant_primitive!(scalar, squeeze, decode, blk, dst); }
        }
    }

    // ========================================================================
    // Position encoding: rope_with_pos
    // ========================================================================

    fn rope_with_pos(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, position: usize, interleaved: bool) {
        if interleaved {
            dispatch_rope_with_pos!(rope_interleaved_with_pos, qk, cos, sin, head_dim, position);
        } else {
            dispatch_rope_with_pos!(rope_with_pos, qk, cos, sin, head_dim, position);
        }
    }

    // ========================================================================
    // Quantized GEMV: gemv_q2, gemv_q1
    // ========================================================================

    fn gemv_q2(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        // Fused dequant-dot: accumulate directly without intermediate buffer
        let in_f32 = elem_to_f32_vec(input);
        let mut sum = 0.0f32;
        for i in 0..n {
            let byte_idx = i / 4;
            let shift = (i % 4) * 2;
            let q = ((weight[byte_idx] >> shift) & 0x03) as f32;
            sum += scale * (q - 1.5) * in_f32[i];
        }
        E::from_f32(sum)
    }

    fn gemv_q1(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        // Fused dequant-dot: accumulate directly without intermediate buffer
        let in_f32 = elem_to_f32_vec(input);
        let mut sum = 0.0f32;
        for i in 0..n {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let q = (weight[byte_idx] >> bit_idx) & 1;
            let w = if q == 0 { -scale } else { scale };
            sum += w * in_f32[i];
        }
        E::from_f32(sum)
    }

    // ========================================================================
    // Quantized format-specific matmul (SPEC §2.3)
    // ========================================================================

    fn kquant_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: crate::quant::QuantType, m: usize, n: usize, k: usize,
    ) {
        use crate::quant::QuantType;

        // Fast path: fused multi-row GEMV for n==1 (bypasses transpose + fn-ptr overhead)
        // AVX-512 preferred when available, falls back to AVX2.
        #[cfg(target_arch = "x86_64")]
        if n == 1 && k % 256 == 0 {
            let isa = get_isa_level();
            match isa {
                IsaLevel::Avx512 | IsaLevel::Avx512Fp16 => {
                    match quant_type {
                        QuantType::Q8K => {
                            if let Some(in_f32) = E::as_f32_slice(input) {
                                let mut out_f32 = vec![0.0f32; m];
                                unsafe {
                                    crate::asm::x86_64::quant_gemv::gemv_q8k_fused_avx512(
                                        weight_blocks.as_ptr(),
                                        in_f32.as_ptr(),
                                        out_f32.as_mut_ptr(),
                                        m, k,
                                    );
                                }
                                for i in 0..m {
                                    output[i] = E::from_f32(out_f32[i]);
                                }
                                return;
                            }
                        }
                        QuantType::Q4K => {
                            if let Some(in_f32) = E::as_f32_slice(input) {
                                let mut out_f32 = vec![0.0f32; m];
                                unsafe {
                                    crate::asm::x86_64::quant_gemv::gemv_q4k_fused_avx512(
                                        weight_blocks.as_ptr(),
                                        in_f32.as_ptr(),
                                        out_f32.as_mut_ptr(),
                                        m, k,
                                    );
                                }
                                for i in 0..m {
                                    output[i] = E::from_f32(out_f32[i]);
                                }
                                return;
                            }
                        }
                        _ => {}
                    }
                }
                IsaLevel::Avx2 => {
                    match quant_type {
                        QuantType::Q8K => {
                            if let Some(in_f32) = E::as_f32_slice(input) {
                                let mut out_f32 = vec![0.0f32; m];
                                unsafe {
                                    crate::asm::x86_64::quant_gemv::gemv_q8k_fused_avx2_asm(
                                        weight_blocks.as_ptr(),
                                        in_f32.as_ptr(),
                                        out_f32.as_mut_ptr(),
                                        m, k,
                                    );
                                }
                                for i in 0..m {
                                    output[i] = E::from_f32(out_f32[i]);
                                }
                                return;
                            }
                        }
                        QuantType::Q4K => {
                            if let Some(in_f32) = E::as_f32_slice(input) {
                                let mut out_f32 = vec![0.0f32; m];
                                unsafe {
                                    crate::asm::x86_64::quant_gemv::gemv_q4k_fused_avx2(
                                        weight_blocks.as_ptr(),
                                        in_f32.as_ptr(),
                                        out_f32.as_mut_ptr(),
                                        m, k,
                                    );
                                }
                                for i in 0..m {
                                    output[i] = E::from_f32(out_f32[i]);
                                }
                                return;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Generic path: transpose + per-block dot
        match quant_type {
            QuantType::Q2K => self.quant_matmul_inner::<84, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_q2_k),
            QuantType::Q3K => self.quant_matmul_inner::<110, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_q3_k),
            QuantType::Q4K => self.quant_matmul_inner::<144, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_q4_k),
            QuantType::Q5K => self.quant_matmul_inner::<176, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_q5_k),
            QuantType::Q6K => self.quant_matmul_inner::<210, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_q6_k),
            QuantType::Q8K => self.quant_matmul_inner::<292, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_q8_k),
            // Classic GGML formats — use classic_matmul()
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_0
            | QuantType::Q5_1 | QuantType::Q8_0 | QuantType::Q8_1 =>
                unimplemented!("kquant_matmul does not support classic format {:?}, use classic_matmul()", quant_type),
            // IQ formats — use iq_matmul()
            QuantType::IQ1S | QuantType::IQ1M | QuantType::IQ2XXS | QuantType::IQ2XS
            | QuantType::IQ2S | QuantType::IQ3XXS | QuantType::IQ3S
            | QuantType::IQ4NL | QuantType::IQ4XS =>
                unimplemented!("kquant_matmul does not support IQ format {:?}, use iq_matmul()", quant_type),
            // External formats — use dedicated matmul
            QuantType::AWQ4 => unimplemented!("kquant_matmul does not support AWQ4, use awq_matmul()"),
            QuantType::GPTQ4 => unimplemented!("kquant_matmul does not support GPTQ4, use gptq_matmul()"),
            QuantType::Squeeze => unimplemented!("kquant_matmul does not support Squeeze, use squeeze_matmul()"),
        }
    }

    fn classic_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: crate::quant::QuantType, m: usize, n: usize, k: usize,
    ) {
        use crate::quant::QuantType;
        match quant_type {
            QuantType::Q4_0 => self.quant_matmul_inner::<18, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_q4_0),
            QuantType::Q4_1 => self.quant_matmul_inner::<20, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_q4_1),
            QuantType::Q5_0 => self.quant_matmul_inner::<22, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_q5_0),
            QuantType::Q5_1 => self.quant_matmul_inner::<24, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_q5_1),
            QuantType::Q8_0 => self.quant_matmul_inner::<34, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_q8_0),
            QuantType::Q8_1 => self.quant_matmul_inner::<36, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_q8_1),
            // K-Quant formats — use kquant_matmul()
            QuantType::Q2K | QuantType::Q3K | QuantType::Q4K
            | QuantType::Q5K | QuantType::Q6K | QuantType::Q8K =>
                unimplemented!("classic_matmul does not support K-Quant format {:?}, use kquant_matmul()", quant_type),
            // IQ formats — use iq_matmul()
            QuantType::IQ1S | QuantType::IQ1M | QuantType::IQ2XXS | QuantType::IQ2XS
            | QuantType::IQ2S | QuantType::IQ3XXS | QuantType::IQ3S
            | QuantType::IQ4NL | QuantType::IQ4XS =>
                unimplemented!("classic_matmul does not support IQ format {:?}, use iq_matmul()", quant_type),
            // External formats — use dedicated matmul
            QuantType::AWQ4 => unimplemented!("classic_matmul does not support AWQ4, use awq_matmul()"),
            QuantType::GPTQ4 => unimplemented!("classic_matmul does not support GPTQ4, use gptq_matmul()"),
            QuantType::Squeeze => unimplemented!("classic_matmul does not support Squeeze, use squeeze_matmul()"),
        }
    }

    fn iq_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: crate::quant::QuantType, m: usize, n: usize, k: usize,
    ) {
        use crate::quant::QuantType;
        // Match once at entry, dispatch to format-specific loop (no match in hot path)
        match quant_type {
            QuantType::IQ1S => self.quant_matmul_inner::<50, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq1_s),
            QuantType::IQ1M => self.quant_matmul_inner::<56, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq1_m),
            QuantType::IQ2XXS => self.quant_matmul_inner::<66, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq2_xxs),
            QuantType::IQ2XS => self.quant_matmul_inner::<74, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq2_xs),
            QuantType::IQ2S => self.quant_matmul_inner::<82, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq2_s),
            QuantType::IQ3XXS => self.quant_matmul_inner::<98, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq3_xxs),
            QuantType::IQ3S => self.quant_matmul_inner::<110, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq3_s),
            QuantType::IQ4NL => self.quant_matmul_inner::<18, 32, _>(weight_blocks, input, output, m, n, k, Self::dot_iq4_nl),
            QuantType::IQ4XS => self.quant_matmul_inner::<136, 256, _>(weight_blocks, input, output, m, n, k, Self::dot_iq4_xs),
            // K-Quant formats — use kquant_matmul()
            QuantType::Q2K | QuantType::Q3K | QuantType::Q4K
            | QuantType::Q5K | QuantType::Q6K | QuantType::Q8K =>
                unimplemented!("iq_matmul does not support K-Quant format {:?}, use kquant_matmul()", quant_type),
            // Classic GGML formats — use classic_matmul()
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_0
            | QuantType::Q5_1 | QuantType::Q8_0 | QuantType::Q8_1 =>
                unimplemented!("iq_matmul does not support classic format {:?}, use classic_matmul()", quant_type),
            // External formats — use dedicated matmul
            QuantType::AWQ4 => unimplemented!("iq_matmul does not support AWQ4, use awq_matmul()"),
            QuantType::GPTQ4 => unimplemented!("iq_matmul does not support GPTQ4, use gptq_matmul()"),
            QuantType::Squeeze => unimplemented!("iq_matmul does not support Squeeze, use squeeze_matmul()"),
        }
    }

    fn awq_matmul(
        &self, weight: &[u8], zeros: &[u8], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        // AWQ4: dequant row -> f32 buffer, then SIMD dot with pre-transposed input
        let group_size = 128usize;
        let num_groups_per_row = k / group_size;
        thread_local! {
            static AWQ_INPUT_T: std::cell::Cell<crate::cache_params::AlignedVec<f32>>
                = std::cell::Cell::new(crate::cache_params::AlignedVec::new());
            static AWQ_DEQUANT: std::cell::Cell<crate::cache_params::AlignedVec<f32>>
                = std::cell::Cell::new(crate::cache_params::AlignedVec::new());
        }
        AWQ_INPUT_T.with(|cell_t| {
        AWQ_DEQUANT.with(|cell_d| {
            let mut buf_t = cell_t.take();
            let mut buf_d = cell_d.take();
            buf_t.resize_zeroed(k * n);
            buf_d.resize_zeroed(k);
            let input_t = buf_t.as_mut_slice();
            let dequant_row = buf_d.as_mut_slice();
            // Pre-transpose input from [k, n] col-major to [n, k] row-major
            for p in 0..k {
                for j in 0..n {
                    input_t[j * k + p] = input[p * n + j].to_f32();
                }
            }
            let k_bytes = k / 2;
            for j in 0..n {
                let in_row = &input_t[j * k..(j + 1) * k];
                for i in 0..m {
                    let row_byte_offset = i * k_bytes;
                    for byte_idx in 0..k_bytes {
                        let b = weight[row_byte_offset + byte_idx];
                        let lo = (b & 0x0F) as f32;
                        let hi = (b >> 4) as f32;
                        let idx = byte_idx * 2;
                        let group_lo = (i * num_groups_per_row) + idx / group_size;
                        let group_hi = (i * num_groups_per_row) + (idx + 1) / group_size;
                        let zero_lo = if group_lo < zeros.len() { zeros[group_lo] as f32 } else { 8.0 };
                        let zero_hi = if group_hi < zeros.len() { zeros[group_hi] as f32 } else { 8.0 };
                        let scale_lo = if group_lo < scales.len() { scales[group_lo].to_f32() } else { 1.0 };
                        let scale_hi = if group_hi < scales.len() { scales[group_hi].to_f32() } else { 1.0 };
                        dequant_row[idx] = (lo - zero_lo) * scale_lo;
                        dequant_row[idx + 1] = (hi - zero_hi) * scale_hi;
                    }
                    output[i * n + j] = E::from_f32(self.dot_f32(dequant_row, in_row));
                }
            }
            cell_t.set(buf_t);
            cell_d.set(buf_d);
        });
        });
    }

    fn gptq_matmul(
        &self, weight: &[u8], g_idx: &[i32], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        // GPTQ4: dequant row -> f32 buffer, then SIMD dot with pre-transposed input
        thread_local! {
            static GPTQ_INPUT_T: std::cell::Cell<crate::cache_params::AlignedVec<f32>>
                = std::cell::Cell::new(crate::cache_params::AlignedVec::new());
            static GPTQ_DEQUANT: std::cell::Cell<crate::cache_params::AlignedVec<f32>>
                = std::cell::Cell::new(crate::cache_params::AlignedVec::new());
        }
        GPTQ_INPUT_T.with(|cell_t| {
        GPTQ_DEQUANT.with(|cell_d| {
            let mut buf_t = cell_t.take();
            let mut buf_d = cell_d.take();
            buf_t.resize_zeroed(k * n);
            buf_d.resize_zeroed(k);
            let input_t = buf_t.as_mut_slice();
            let dequant_row = buf_d.as_mut_slice();
            for p in 0..k {
                for j in 0..n {
                    input_t[j * k + p] = input[p * n + j].to_f32();
                }
            }
            let k_bytes = k / 2;
            for j in 0..n {
                let in_row = &input_t[j * k..(j + 1) * k];
                for i in 0..m {
                    let row_byte_offset = i * k_bytes;
                    for byte_idx in 0..k_bytes {
                        let b = weight[row_byte_offset + byte_idx];
                        let lo = (b & 0x0F) as f32;
                        let hi = (b >> 4) as f32;
                        let idx = byte_idx * 2;
                        let group_lo = if idx < g_idx.len() { g_idx[idx] as usize } else { 0 };
                        let group_hi = if (idx + 1) < g_idx.len() { g_idx[idx + 1] as usize } else { 0 };
                        let scale_lo = if group_lo < scales.len() { scales[group_lo].to_f32() } else { 1.0 };
                        let scale_hi = if group_hi < scales.len() { scales[group_hi].to_f32() } else { 1.0 };
                        dequant_row[idx] = (lo - 8.0) * scale_lo;
                        dequant_row[idx + 1] = (hi - 8.0) * scale_hi;
                    }
                    output[i * n + j] = E::from_f32(self.dot_f32(dequant_row, in_row));
                }
            }
            cell_t.set(buf_t);
            cell_d.set(buf_d);
        });
        });
    }

    fn squeeze_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    ) {
        let block_size = 256usize;
        let block_bytes = 130usize;
        let blocks_per_row = k / block_size;
        thread_local! {
            static SQ_INPUT_T: std::cell::Cell<crate::cache_params::AlignedVec<f32>>
                = std::cell::Cell::new(crate::cache_params::AlignedVec::new());
        }
        SQ_INPUT_T.with(|cell| {
            let mut buf = cell.take();
            buf.resize_zeroed(k * n);
            let input_t = buf.as_mut_slice();
            for p in 0..k {
                for j in 0..n {
                    input_t[j * k + p] = input[p * n + j].to_f32();
                }
            }
            for j in 0..n {
                let in_row = &input_t[j * k..];
                for i in 0..m {
                    let mut sum = 0.0f32;
                    for b in 0..blocks_per_row {
                        let off = i * blocks_per_row * block_bytes + b * block_bytes;
                        let blk = &weight_blocks[off..off + block_bytes];
                        let blk_ptr = blk.as_ptr() as *const crate::quant::BlockSqueeze;
                        let src = in_row[b * block_size..(b + 1) * block_size].as_ptr();
                        sum += match get_isa_level() {
                            #[cfg(target_arch = "x86_64")]
                            IsaLevel::Avx512 => crate::quant_primitive!(avx512, squeeze, dot, blk_ptr, src),
                            #[cfg(target_arch = "x86_64")]
                            IsaLevel::Avx2 => crate::quant_primitive!(avx2, squeeze, dot, blk_ptr, src),
                            #[cfg(target_arch = "aarch64")]
                            IsaLevel::Neon => crate::quant_primitive!(neon, squeeze, dot, blk_ptr, src),
                            _ => crate::quant_primitive!(scalar, squeeze, dot, blk_ptr, src),
                        };
                    }
                    output[i * n + j] = E::from_f32(sum);
                }
            }
            cell.set(buf);
        });
    }

}
