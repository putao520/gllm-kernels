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

pub struct CpuKernels<E: Element> {
    _phantom: PhantomData<E>,
}

impl<E: Element> CpuKernels<E> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }

    // ========================================================================
    // Private helper methods: dot_* quantization primitives
    // These are called by public Kernels trait methods
    // ========================================================================

    fn dot_q4_k(&self, block: &[u8], other: &[f32]) -> f32 {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ4K;
        let other_ptr = other.as_ptr();
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                crate::quant_primitive!(avx2, q4_k, dot, block_ptr, other_ptr)
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                crate::quant_primitive!(avx2, q4_k, dot, block_ptr, other_ptr)
            }
            _ => {
                crate::quant_primitive!(scalar, q4_k, dot, block_ptr, other_ptr)
            }
        }
    }

    fn dot_q8_k(&self, block: &[u8], other: &[f32]) -> f32 {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ8K;
        let other_ptr = other.as_ptr();
        crate::quant_primitive!(scalar, q8_k, dot, block_ptr, other_ptr)
    }

    fn dot_q2_k(&self, block: &[u8], other: &[f32]) -> f32 {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ2K;
        let other_ptr = other.as_ptr();
        crate::quant_primitive!(scalar, q2_k, dot, block_ptr, other_ptr)
    }

    fn dot_q3_k(&self, block: &[u8], other: &[f32]) -> f32 {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ3K;
        let other_ptr = other.as_ptr();
        crate::quant_primitive!(scalar, q3_k, dot, block_ptr, other_ptr)
    }

    fn dot_q5_k(&self, block: &[u8], other: &[f32]) -> f32 {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ5K;
        let other_ptr = other.as_ptr();
        crate::quant_primitive!(scalar, q5_k, dot, block_ptr, other_ptr)
    }

    fn dot_q6_k(&self, block: &[u8], other: &[f32]) -> f32 {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ6K;
        let other_ptr = other.as_ptr();
        crate::quant_primitive!(scalar, q6_k, dot, block_ptr, other_ptr)
    }

    fn dot_iq1_s(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ1S;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq1_s, dot, blk, src)
    }

    fn dot_iq1_m(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ1M;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq1_m, dot, blk, src)
    }

    fn dot_iq2_xxs(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ2XXS;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq2_xxs, dot, blk, src)
    }

    fn dot_iq2_xs(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ2XS;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq2_xs, dot, blk, src)
    }

    fn dot_iq2_s(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ2S;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq2_s, dot, blk, src)
    }

    fn dot_iq3_xxs(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ3XXS;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq3_xxs, dot, blk, src)
    }

    fn dot_iq3_s(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ3S;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq3_s, dot, blk, src)
    }

    fn dot_iq4_nl(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ4NL;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq4_nl, dot, blk, src)
    }

    fn dot_iq4_xs(&self, block: &[u8], other: &[f32]) -> f32 {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ4XS;
        let src = other.as_ptr();
        crate::quant_primitive!(scalar, iq4_xs, dot, blk, src)
    }

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
        let in_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
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
        for i in 0..m {
            for j in 0..n {
                let in_f32: Vec<f32> = (0..k).map(|p| input[p * n + j].to_f32()).collect();
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let off = i * blocks_per_row * BLOCK_BYTES + b * BLOCK_BYTES;
                    let blk = &weight_blocks[off..off + BLOCK_BYTES];
                    let in_slice = &in_f32[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
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
// Dispatch macros: use Element::as_f32_slice for zero-cost type dispatch
// (No TypeId — compile-time monomorphization via Element trait methods)
// ============================================================================

macro_rules! dispatch_binary_op {
    ($a:expr, $b:expr, $out:expr, $op:ident) => {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(af), Some(bf), Some(of)) = (
                    E::as_f32_slice($a), E::as_f32_slice($b), E::as_f32_slice_mut($out)
                ) {
                    avx512::avx512_f32::$op(af, bf, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = $b.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    avx512::avx512_f32::$op(&ac, &bc, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(af), Some(bf), Some(of)) = (
                    E::as_f32_slice($a), E::as_f32_slice($b), E::as_f32_slice_mut($out)
                ) {
                    avx2::avx2_f32::$op(af, bf, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = $b.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    avx2::avx2_f32::$op(&ac, &bc, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "aarch64")]
            IsaLevel::Neon => {
                if let (Some(af), Some(bf), Some(of)) = (
                    E::as_f32_slice($a), E::as_f32_slice($b), E::as_f32_slice_mut($out)
                ) {
                    neon::neon_f32::$op(af, bf, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = $b.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    neon::neon_f32::$op(&ac, &bc, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            _ => {
                if let (Some(af), Some(bf), Some(of)) = (
                    E::as_f32_slice($a), E::as_f32_slice($b), E::as_f32_slice_mut($out)
                ) {
                    scalar::scalar_f32::$op(af, bf, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = $b.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    scalar::scalar_f32::$op(&ac, &bc, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
        }
    };
}

macro_rules! dispatch_unary_op {
    ($a:expr, $out:expr, $op:ident) => {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(af), Some(of)) = (E::as_f32_slice($a), E::as_f32_slice_mut($out)) {
                    avx512::avx512_f32::$op(af, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    avx512::avx512_f32::$op(&ac, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(af), Some(of)) = (E::as_f32_slice($a), E::as_f32_slice_mut($out)) {
                    avx2::avx2_f32::$op(af, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    avx2::avx2_f32::$op(&ac, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "aarch64")]
            IsaLevel::Neon => {
                if let (Some(af), Some(of)) = (E::as_f32_slice($a), E::as_f32_slice_mut($out)) {
                    neon::neon_f32::$op(af, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    neon::neon_f32::$op(&ac, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            _ => {
                if let (Some(af), Some(of)) = (E::as_f32_slice($a), E::as_f32_slice_mut($out)) {
                    scalar::scalar_f32::$op(af, of);
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; $out.len()];
                    scalar::scalar_f32::$op(&ac, &mut oc);
                    for (o, v) in $out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
        }
    };
}

macro_rules! dispatch_reduce_op {
    ($a:expr, $op:ident) => {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let Some(af) = E::as_f32_slice($a) {
                    avx512::avx512_f32::$op(af)
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    avx512::avx512_f32::$op(&ac)
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let Some(af) = E::as_f32_slice($a) {
                    avx2::avx2_f32::$op(af)
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    avx2::avx2_f32::$op(&ac)
                }
            }
            #[cfg(target_arch = "aarch64")]
            IsaLevel::Neon => {
                if let Some(af) = E::as_f32_slice($a) {
                    neon::neon_f32::$op(af)
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    neon::neon_f32::$op(&ac)
                }
            }
            _ => {
                if let Some(af) = E::as_f32_slice($a) {
                    scalar::scalar_f32::$op(af)
                } else {
                    let ac: Vec<f32> = $a.iter().map(|v| v.to_f32()).collect();
                    scalar::scalar_f32::$op(&ac)
                }
            }
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
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(xf), Some(sf)) = (E::as_f32_slice_mut(x), E::as_f32_ref(&s)) {
                    let mut out = vec![0.0f32; xf.len()];
                    avx512::avx512_f32::scale(xf, *sf, &mut out);
                    xf.copy_from_slice(&out);
                } else {
                    let sf = s.to_f32();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut out = vec![0.0f32; xc.len()];
                    avx512::avx512_f32::scale(&xc, sf, &mut out);
                    for (xv, ov) in x.iter_mut().zip(out.iter()) { *xv = E::from_f32(*ov); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(xf), Some(sf)) = (E::as_f32_slice_mut(x), E::as_f32_ref(&s)) {
                    let mut out = vec![0.0f32; xf.len()];
                    avx2::avx2_f32::scale(xf, *sf, &mut out);
                    xf.copy_from_slice(&out);
                } else {
                    let sf = s.to_f32();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut out = vec![0.0f32; xc.len()];
                    avx2::avx2_f32::scale(&xc, sf, &mut out);
                    for (xv, ov) in x.iter_mut().zip(out.iter()) { *xv = E::from_f32(*ov); }
                }
            }
            _ => {
                if let (Some(xf), Some(sf)) = (E::as_f32_slice_mut(x), E::as_f32_ref(&s)) {
                    let mut out = vec![0.0f32; xf.len()];
                    scalar::scalar_f32::scale(xf, *sf, &mut out);
                    xf.copy_from_slice(&out);
                } else {
                    let sf = s.to_f32();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut out = vec![0.0f32; xc.len()];
                    scalar::scalar_f32::scale(&xc, sf, &mut out);
                    for (xv, ov) in x.iter_mut().zip(out.iter()) { *xv = E::from_f32(*ov); }
                }
            }
        }
    }

    fn vec_axpy(&self, y: &mut [E], a: E, x: &[E]) {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(yf), Some(xf), Some(af)) = (
                    E::as_f32_slice_mut(y), E::as_f32_slice(x), E::as_f32_ref(&a)
                ) {
                    avx512::avx512_f32::axpy(*af, xf, yf);
                } else {
                    let af = a.to_f32();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut yc: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                    avx512::avx512_f32::axpy(af, &xc, &mut yc);
                    for (yv, ov) in y.iter_mut().zip(yc.iter()) { *yv = E::from_f32(*ov); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(yf), Some(xf), Some(af)) = (
                    E::as_f32_slice_mut(y), E::as_f32_slice(x), E::as_f32_ref(&a)
                ) {
                    avx2::avx2_f32::axpy(*af, xf, yf);
                } else {
                    let af = a.to_f32();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut yc: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                    avx2::avx2_f32::axpy(af, &xc, &mut yc);
                    for (yv, ov) in y.iter_mut().zip(yc.iter()) { *yv = E::from_f32(*ov); }
                }
            }
            _ => {
                if let (Some(yf), Some(xf), Some(af)) = (
                    E::as_f32_slice_mut(y), E::as_f32_slice(x), E::as_f32_ref(&a)
                ) {
                    scalar::scalar_f32::axpy(*af, xf, yf);
                } else {
                    let af = a.to_f32();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut yc: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                    scalar::scalar_f32::axpy(af, &xc, &mut yc);
                    for (yv, ov) in y.iter_mut().zip(yc.iter()) { *yv = E::from_f32(*ov); }
                }
            }
        }
    }

    fn vec_sum(&self, x: &[E]) -> E { E::from_f32(dispatch_reduce_op!(x, sum)) }
    fn vec_max(&self, x: &[E]) -> E { E::from_f32(dispatch_reduce_op!(x, max_val)) }
    fn vec_sum_squares(&self, x: &[E]) -> E { E::from_f32(dispatch_reduce_op!(x, sum_squares)) }

    // BLAS-2/3
    fn gemv(&self, a: &[E], x: &[E], y: &mut [E], m: usize, n: usize) {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(af), Some(xf), Some(yf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(x), E::as_f32_slice_mut(y)
                ) {
                    avx512::avx512_f32::gemv(af, xf, yf, m, n);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut yc: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                    avx512::avx512_f32::gemv(&ac, &xc, &mut yc, m, n);
                    for (yv, ov) in y.iter_mut().zip(yc.iter()) { *yv = E::from_f32(*ov); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(af), Some(xf), Some(yf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(x), E::as_f32_slice_mut(y)
                ) {
                    avx2::avx2_f32::gemv(af, xf, yf, m, n);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut yc: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                    avx2::avx2_f32::gemv(&ac, &xc, &mut yc, m, n);
                    for (yv, ov) in y.iter_mut().zip(yc.iter()) { *yv = E::from_f32(*ov); }
                }
            }
            _ => {
                if let (Some(af), Some(xf), Some(yf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(x), E::as_f32_slice_mut(y)
                ) {
                    scalar::scalar_f32::gemv(af, xf, yf, m, n);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let mut yc: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                    scalar::scalar_f32::gemv(&ac, &xc, &mut yc, m, n);
                    for (yv, ov) in y.iter_mut().zip(yc.iter()) { *yv = E::from_f32(*ov); }
                }
            }
        }
    }

    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(af), Some(bf), Some(cf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(b), E::as_f32_slice_mut(c)
                ) {
                    avx512::avx512_f32::matmul(af, bf, cf, m, n, k);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
                    let mut cc = vec![0.0f32; c.len()];
                    avx512::avx512_f32::matmul(&ac, &bc, &mut cc, m, n, k);
                    for (cv, ov) in c.iter_mut().zip(cc.iter()) { *cv = E::from_f32(*ov); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(af), Some(bf), Some(cf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(b), E::as_f32_slice_mut(c)
                ) {
                    avx2::avx2_f32::matmul(af, bf, cf, m, n, k);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
                    let mut cc = vec![0.0f32; c.len()];
                    avx2::avx2_f32::matmul(&ac, &bc, &mut cc, m, n, k);
                    for (cv, ov) in c.iter_mut().zip(cc.iter()) { *cv = E::from_f32(*ov); }
                }
            }
            #[cfg(target_arch = "aarch64")]
            IsaLevel::Neon => {
                if let (Some(af), Some(bf), Some(cf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(b), E::as_f32_slice_mut(c)
                ) {
                    neon::neon_f32::matmul(af, bf, cf, m, n, k);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
                    let mut cc = vec![0.0f32; c.len()];
                    neon::neon_f32::matmul(&ac, &bc, &mut cc, m, n, k);
                    for (cv, ov) in c.iter_mut().zip(cc.iter()) { *cv = E::from_f32(*ov); }
                }
            }
            _ => {
                if let (Some(af), Some(bf), Some(cf)) = (
                    E::as_f32_slice(a), E::as_f32_slice(b), E::as_f32_slice_mut(c)
                ) {
                    scalar::scalar_f32::matmul(af, bf, cf, m, n, k);
                } else {
                    let ac: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
                    let mut cc = vec![0.0f32; c.len()];
                    scalar::scalar_f32::matmul(&ac, &bc, &mut cc, m, n, k);
                    for (cv, ov) in c.iter_mut().zip(cc.iter()) { *cv = E::from_f32(*ov); }
                }
            }
        }
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
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(xf), Some(wf), Some(of)) = (
                    E::as_f32_slice(x), E::as_f32_slice(weight), E::as_f32_slice_mut(out)
                ) {
                    avx512::avx512_f32::rms_norm(xf, wf, of, eps);
                } else {
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let wc: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; out.len()];
                    avx512::avx512_f32::rms_norm(&xc, &wc, &mut oc, eps);
                    for (o, v) in out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(xf), Some(wf), Some(of)) = (
                    E::as_f32_slice(x), E::as_f32_slice(weight), E::as_f32_slice_mut(out)
                ) {
                    avx2::avx2_f32::rms_norm(xf, wf, of, eps);
                } else {
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let wc: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; out.len()];
                    avx2::avx2_f32::rms_norm(&xc, &wc, &mut oc, eps);
                    for (o, v) in out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            _ => {
                if let (Some(xf), Some(wf), Some(of)) = (
                    E::as_f32_slice(x), E::as_f32_slice(weight), E::as_f32_slice_mut(out)
                ) {
                    scalar::scalar_f32::rms_norm(xf, wf, of, eps);
                } else {
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let wc: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; out.len()];
                    scalar::scalar_f32::rms_norm(&xc, &wc, &mut oc, eps);
                    for (o, v) in out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
        }
    }

    fn layer_norm(&self, x: &[E], gamma: &[E], beta: &[E], out: &mut [E], eps: f32) {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(xf), Some(gf), Some(bf), Some(of)) = (
                    E::as_f32_slice(x), E::as_f32_slice(gamma), E::as_f32_slice(beta), E::as_f32_slice_mut(out)
                ) {
                    avx512::avx512_f32::layer_norm(xf, gf, bf, of, eps);
                } else {
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let gc: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = beta.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; out.len()];
                    avx512::avx512_f32::layer_norm(&xc, &gc, &bc, &mut oc, eps);
                    for (o, v) in out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(xf), Some(gf), Some(bf), Some(of)) = (
                    E::as_f32_slice(x), E::as_f32_slice(gamma), E::as_f32_slice(beta), E::as_f32_slice_mut(out)
                ) {
                    avx2::avx2_f32::layer_norm(xf, gf, bf, of, eps);
                } else {
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let gc: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = beta.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; out.len()];
                    avx2::avx2_f32::layer_norm(&xc, &gc, &bc, &mut oc, eps);
                    for (o, v) in out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
            _ => {
                if let (Some(xf), Some(gf), Some(bf), Some(of)) = (
                    E::as_f32_slice(x), E::as_f32_slice(gamma), E::as_f32_slice(beta), E::as_f32_slice_mut(out)
                ) {
                    scalar::scalar_f32::layer_norm(xf, gf, bf, of, eps);
                } else {
                    let xc: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                    let gc: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
                    let bc: Vec<f32> = beta.iter().map(|v| v.to_f32()).collect();
                    let mut oc = vec![0.0f32; out.len()];
                    scalar::scalar_f32::layer_norm(&xc, &gc, &bc, &mut oc, eps);
                    for (o, v) in out.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
                }
            }
        }
    }

    // Positional
    fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, _interleaved: bool) {
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                if let (Some(qf), Some(cf), Some(sf)) = (
                    E::as_f32_slice_mut(qk), E::as_f32_slice(cos), E::as_f32_slice(sin)
                ) {
                    avx512::avx512_f32::rope(qf, cf, sf, head_dim);
                } else {
                    let mut qc: Vec<f32> = qk.iter().map(|v| v.to_f32()).collect();
                    let cc: Vec<f32> = cos.iter().map(|v| v.to_f32()).collect();
                    let sc: Vec<f32> = sin.iter().map(|v| v.to_f32()).collect();
                    avx512::avx512_f32::rope(&mut qc, &cc, &sc, head_dim);
                    for (q, v) in qk.iter_mut().zip(qc.iter()) { *q = E::from_f32(*v); }
                }
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                if let (Some(qf), Some(cf), Some(sf)) = (
                    E::as_f32_slice_mut(qk), E::as_f32_slice(cos), E::as_f32_slice(sin)
                ) {
                    avx2::avx2_f32::rope(qf, cf, sf, head_dim);
                } else {
                    let mut qc: Vec<f32> = qk.iter().map(|v| v.to_f32()).collect();
                    let cc: Vec<f32> = cos.iter().map(|v| v.to_f32()).collect();
                    let sc: Vec<f32> = sin.iter().map(|v| v.to_f32()).collect();
                    avx2::avx2_f32::rope(&mut qc, &cc, &sc, head_dim);
                    for (q, v) in qk.iter_mut().zip(qc.iter()) { *q = E::from_f32(*v); }
                }
            }
            _ => {
                if let (Some(qf), Some(cf), Some(sf)) = (
                    E::as_f32_slice_mut(qk), E::as_f32_slice(cos), E::as_f32_slice(sin)
                ) {
                    scalar::scalar_f32::rope(qf, cf, sf, head_dim);
                } else {
                    let mut qc: Vec<f32> = qk.iter().map(|v| v.to_f32()).collect();
                    let cc: Vec<f32> = cos.iter().map(|v| v.to_f32()).collect();
                    let sc: Vec<f32> = sin.iter().map(|v| v.to_f32()).collect();
                    scalar::scalar_f32::rope(&mut qc, &cc, &sc, head_dim);
                    for (q, v) in qk.iter_mut().zip(qc.iter()) { *q = E::from_f32(*v); }
                }
            }
        }
    }

    // Embedding
    fn embedding_lookup(&self, ids: &[u32], table: &[E], output: &mut [E], _vocab_size: usize, hidden_size: usize) {
        if let (Some(tf), Some(of)) = (E::as_f32_slice(table), E::as_f32_slice_mut(output)) {
            scalar::scalar_f32::embedding_lookup(tf, ids, of, hidden_size);
        } else {
            let tc: Vec<f32> = table.iter().map(|v| v.to_f32()).collect();
            let mut oc = vec![0.0f32; output.len()];
            scalar::scalar_f32::embedding_lookup(&tc, ids, &mut oc, hidden_size);
            for (o, v) in output.iter_mut().zip(oc.iter()) { *o = E::from_f32(*v); }
        }
    }

    // Sampling
    // Quantization
    fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]) {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ4K;
        let out_ptr = out.as_mut_ptr();
        match get_isa_level() {
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx512 => {
                crate::quant_primitive!(avx2, q4_k, decode, block_ptr, out_ptr);
            }
            #[cfg(target_arch = "x86_64")]
            IsaLevel::Avx2 => {
                crate::quant_primitive!(avx2, q4_k, decode, block_ptr, out_ptr);
            }
            _ => {
                crate::quant_primitive!(scalar, q4_k, decode, block_ptr, out_ptr);
            }
        }
    }

    fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]) {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ8K;
        let out_ptr = out.as_mut_ptr();
        crate::quant_primitive!(scalar, q8_k, decode, block_ptr, out_ptr);
    }

    fn dequant_q2_k(&self, block: &[u8], out: &mut [f32]) {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ2K;
        let out_ptr = out.as_mut_ptr();
        crate::quant_primitive!(scalar, q2_k, decode, block_ptr, out_ptr);
    }

    fn dequant_q3_k(&self, block: &[u8], out: &mut [f32]) {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ3K;
        let out_ptr = out.as_mut_ptr();
        crate::quant_primitive!(scalar, q3_k, decode, block_ptr, out_ptr);
    }

    fn dequant_q5_k(&self, block: &[u8], out: &mut [f32]) {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ5K;
        let out_ptr = out.as_mut_ptr();
        crate::quant_primitive!(scalar, q5_k, decode, block_ptr, out_ptr);
    }

    fn dequant_q6_k(&self, block: &[u8], out: &mut [f32]) {
        let block_ptr = block.as_ptr() as *const crate::quant::BlockQ6K;
        let out_ptr = out.as_mut_ptr();
        crate::quant_primitive!(scalar, q6_k, decode, block_ptr, out_ptr);
    }

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
            let in_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
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
            let in_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
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

    fn dequant_iq1_s(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ1S;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq1_s, decode, blk, dst);
    }
    fn dequant_iq1_m(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ1M;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq1_m, decode, blk, dst);
    }
    fn dequant_iq2_xxs(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ2XXS;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq2_xxs, decode, blk, dst);
    }
    fn dequant_iq2_xs(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ2XS;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq2_xs, decode, blk, dst);
    }
    fn dequant_iq2_s(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ2S;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq2_s, decode, blk, dst);
    }
    fn dequant_iq3_xxs(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ3XXS;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq3_xxs, decode, blk, dst);
    }
    fn dequant_iq3_s(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ3S;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq3_s, decode, blk, dst);
    }
    fn dequant_iq4_nl(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ4NL;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq4_nl, decode, blk, dst);
    }
    fn dequant_iq4_xs(&self, block: &[u8], out: &mut [f32]) {
        let blk = block.as_ptr() as *const crate::quant::BlockIQ4XS;
        let dst = out.as_mut_ptr();
        crate::quant_primitive!(scalar, iq4_xs, decode, blk, dst);
    }

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
        for h in 0..num_heads {
            for i in 0..seq_len {
                let q_off = h * seq_len * head_dim + i * head_dim;
                let o_off = q_off;
                let max_j = if causal { i + 1 } else { seq_len };
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(max_j);
                for j in 0..max_j {
                    let k_off = h * seq_len * head_dim + j * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d].to_f32() * k[k_off + d].to_f32();
                    }
                    let s = dot * scale;
                    if s > max_val { max_val = s; }
                    scores.push(s);
                }
                let mut sum_exp = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp;
                let mut acc = vec![0.0f32; head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let v_off = h * seq_len * head_dim + j * head_dim;
                    let a = w * inv_sum;
                    for d in 0..head_dim {
                        acc[d] += a * v[v_off + d].to_f32();
                    }
                }
                for d in 0..head_dim {
                    output[o_off + d] = E::from_f32(acc[d]);
                }
            }
        }
    }

    fn flash_attention_paged(
        &self, q: &[E], k_cache: &[E], v_cache: &[E],
        page_table: &[usize], output: &mut [E],
        seq_len: usize, cache_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        page_size: usize, scale: f32,
    ) {
        let heads_per_kv = num_heads / num_kv_heads;
        let num_pages = (cache_len + page_size - 1) / page_size;
        let _ = num_pages;
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            for i in 0..seq_len {
                let q_off = h * seq_len * head_dim + i * head_dim;
                let o_off = q_off;
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(cache_len);
                for j in 0..cache_len {
                    let page_idx = j / page_size;
                    let page_off = j % page_size;
                    let phys_page = page_table[kv_h * ((cache_len + page_size - 1) / page_size) + page_idx];
                    let k_off = phys_page * page_size * head_dim + page_off * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d].to_f32() * k_cache[k_off + d].to_f32();
                    }
                    let s = dot * scale;
                    if s > max_val { max_val = s; }
                    scores.push(s);
                }
                let mut sum_exp = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp;
                let mut acc = vec![0.0f32; head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let page_idx = j / page_size;
                    let page_off = j % page_size;
                    let phys_page = page_table[kv_h * ((cache_len + page_size - 1) / page_size) + page_idx];
                    let v_off = phys_page * page_size * head_dim + page_off * head_dim;
                    let a = w * inv_sum;
                    for d in 0..head_dim {
                        acc[d] += a * v_cache[v_off + d].to_f32();
                    }
                }
                for d in 0..head_dim {
                    output[o_off + d] = E::from_f32(acc[d]);
                }
            }
        }
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
        for i in 0..m {
            for j in 0..n {
                let in_f32: Vec<f32> = (0..k).map(|p| input[p * n + j].to_f32()).collect();
                let mut sum = 0.0f32;
                // Process weight row i, element by element
                let row_offset = i * k;
                // Weight is packed as 4-bit nibbles: 2 values per byte
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
                    sum += (nibble - zero) * scale * in_f32[idx];
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
        for i in 0..m {
            for j in 0..n {
                let in_f32: Vec<f32> = (0..k).map(|p| input[p * n + j].to_f32()).collect();
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
                    sum += (nibble - 8.0) * scale * in_f32[idx];
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
        for i in 0..m {
            for j in 0..n {
                let in_f32: Vec<f32> = (0..k).map(|p| input[p * n + j].to_f32()).collect();
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let off = i * blocks_per_row * block_bytes + b * block_bytes;
                    let blk = &weight_blocks[off..off + block_bytes];
                    let blk_ptr = blk.as_ptr() as *const crate::quant::BlockSqueeze;
                    let src = in_f32[b * block_size..(b + 1) * block_size].as_ptr();
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
