use std::fmt::Debug;
use half::{f16, bf16};
use crate::quant::QuantType;

/// Activation function selector for fused GEMM+activation epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    Relu,
    Silu,
    Gelu,
}

/// Represents a device-specific representation of a tensor or buffer.
pub trait DeviceRepr: Debug + Clone + Send + Sync + 'static {}

/// A blanket implementation for any type that meets the criteria.
impl<T> DeviceRepr for T where T: Debug + Clone + Send + Sync + 'static {}

/// Core element trait for tensor operations (SPEC/03 §2.1).
///
/// Provides a unified interface for scalar operations across precisions (f32, f16, bf16).
/// Compile-time monomorphization, zero runtime overhead.
pub trait Element:
    Debug + Clone + Copy + Send + Sync + Default + 'static + DeviceRepr
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
    + PartialOrd
{
    const ZERO: Self;
    const ONE: Self;
    /// Element type discriminant: 0=f32, 1=f16, 2=bf16
    const ELEM_ID: u8;

    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;

    // Fused multiply-add: self + a * b
    fn mul_add(self, a: Self, b: Self) -> Self;

    // Arithmetic (explicit methods for SIMD macro compatibility)
    fn elem_add(self, other: Self) -> Self;
    fn elem_sub(self, other: Self) -> Self;
    fn elem_mul(self, other: Self) -> Self;
    fn elem_div(self, other: Self) -> Self;
    fn neg(self) -> Self;

    // Comparison
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    // Math functions
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn recip(self) -> Self;
    fn abs(self) -> Self;
    fn tanh(self) -> Self;

    /// Zero-cost transmute to f32 slice when Self == f32.
    /// Returns None for non-f32 types (caller must convert element-by-element).
    fn as_f32_slice(s: &[Self]) -> Option<&[f32]>;

    /// Zero-cost transmute to mutable f32 slice when Self == f32.
    fn as_f32_slice_mut(s: &mut [Self]) -> Option<&mut [f32]>;

    /// Reinterpret a scalar reference as f32 (zero-cost for f32, None for others).
    fn as_f32_ref(v: &Self) -> Option<&f32>;
}

impl Element for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const ELEM_ID: u8 = 0;

    #[inline(always)] fn from_f32(v: f32) -> Self { v }
    #[inline(always)] fn to_f32(self) -> f32 { self }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { f32::mul_add(a, b, self) }

    #[inline(always)] fn elem_add(self, other: Self) -> Self { self + other }
    #[inline(always)] fn elem_sub(self, other: Self) -> Self { self - other }
    #[inline(always)] fn elem_mul(self, other: Self) -> Self { self * other }
    #[inline(always)] fn elem_div(self, other: Self) -> Self { self / other }
    #[inline(always)] fn neg(self) -> Self { -self }

    #[inline(always)] fn max(self, other: Self) -> Self { f32::max(self, other) }
    #[inline(always)] fn min(self, other: Self) -> Self { f32::min(self, other) }

    #[inline(always)] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline(always)] fn exp(self) -> Self { f32::exp(self) }
    #[inline(always)] fn recip(self) -> Self { 1.0 / self }
    #[inline(always)] fn abs(self) -> Self { f32::abs(self) }
    #[inline(always)] fn tanh(self) -> Self { f32::tanh(self) }

    #[inline(always)]
    fn as_f32_slice(s: &[Self]) -> Option<&[f32]> { Some(s) }

    #[inline(always)]
    fn as_f32_slice_mut(s: &mut [Self]) -> Option<&mut [f32]> { Some(s) }

    #[inline(always)]
    fn as_f32_ref(v: &Self) -> Option<&f32> { Some(v) }
}

impl Element for f16 {
    const ZERO: Self = f16::ZERO;
    const ONE: Self = f16::ONE;
    const ELEM_ID: u8 = 1;

    #[inline(always)] fn from_f32(v: f32) -> Self { f16::from_f32(v) }
    #[inline(always)] fn to_f32(self) -> f32 { f16::to_f32(self) }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self {
        f16::from_f32(f32::mul_add(a.to_f32(), b.to_f32(), self.to_f32()))
    }

    #[inline(always)] fn elem_add(self, other: Self) -> Self { f16::from_f32(self.to_f32() + other.to_f32()) }
    #[inline(always)] fn elem_sub(self, other: Self) -> Self { f16::from_f32(self.to_f32() - other.to_f32()) }
    #[inline(always)] fn elem_mul(self, other: Self) -> Self { f16::from_f32(self.to_f32() * other.to_f32()) }
    #[inline(always)] fn elem_div(self, other: Self) -> Self { f16::from_f32(self.to_f32() / other.to_f32()) }
    #[inline(always)] fn neg(self) -> Self { f16::from_f32(-self.to_f32()) }

    #[inline(always)] fn max(self, other: Self) -> Self { if self.to_f32() >= other.to_f32() { self } else { other } }
    #[inline(always)] fn min(self, other: Self) -> Self { if self.to_f32() <= other.to_f32() { self } else { other } }

    #[inline(always)] fn sqrt(self) -> Self { f16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)] fn exp(self) -> Self { f16::from_f32(self.to_f32().exp()) }
    #[inline(always)] fn recip(self) -> Self { f16::from_f32(1.0 / self.to_f32()) }
    #[inline(always)] fn abs(self) -> Self { f16::from_f32(self.to_f32().abs()) }
    #[inline(always)] fn tanh(self) -> Self { f16::from_f32(self.to_f32().tanh()) }

    #[inline(always)] fn as_f32_slice(_s: &[Self]) -> Option<&[f32]> { None }
    #[inline(always)] fn as_f32_slice_mut(_s: &mut [Self]) -> Option<&mut [f32]> { None }
    #[inline(always)] fn as_f32_ref(_v: &Self) -> Option<&f32> { None }
}

impl Element for bf16 {
    const ZERO: Self = bf16::ZERO;
    const ONE: Self = bf16::ONE;
    const ELEM_ID: u8 = 2;

    #[inline(always)] fn from_f32(v: f32) -> Self { bf16::from_f32(v) }
    #[inline(always)] fn to_f32(self) -> f32 { bf16::to_f32(self) }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self {
        bf16::from_f32(f32::mul_add(a.to_f32(), b.to_f32(), self.to_f32()))
    }

    #[inline(always)] fn elem_add(self, other: Self) -> Self { bf16::from_f32(self.to_f32() + other.to_f32()) }
    #[inline(always)] fn elem_sub(self, other: Self) -> Self { bf16::from_f32(self.to_f32() - other.to_f32()) }
    #[inline(always)] fn elem_mul(self, other: Self) -> Self { bf16::from_f32(self.to_f32() * other.to_f32()) }
    #[inline(always)] fn elem_div(self, other: Self) -> Self { bf16::from_f32(self.to_f32() / other.to_f32()) }
    #[inline(always)] fn neg(self) -> Self { bf16::from_f32(-self.to_f32()) }

    #[inline(always)] fn max(self, other: Self) -> Self { if self.to_f32() >= other.to_f32() { self } else { other } }
    #[inline(always)] fn min(self, other: Self) -> Self { if self.to_f32() <= other.to_f32() { self } else { other } }

    #[inline(always)] fn sqrt(self) -> Self { bf16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)] fn exp(self) -> Self { bf16::from_f32(self.to_f32().exp()) }
    #[inline(always)] fn recip(self) -> Self { bf16::from_f32(1.0 / self.to_f32()) }
    #[inline(always)] fn abs(self) -> Self { bf16::from_f32(self.to_f32().abs()) }
    #[inline(always)] fn tanh(self) -> Self { bf16::from_f32(self.to_f32().tanh()) }

    #[inline(always)] fn as_f32_slice(_s: &[Self]) -> Option<&[f32]> { None }
    #[inline(always)] fn as_f32_slice_mut(_s: &mut [Self]) -> Option<&mut [f32]> { None }
    #[inline(always)] fn as_f32_ref(_v: &Self) -> Option<&f32> { None }
}

// ==========================================================================
// Backend Trait (SPEC/03 §2.2)
// ==========================================================================

/// Abstract computation backend.
///
/// Connects a specific device backend (CPU) with its kernel implementation.
pub trait Backend: Send + Sync + 'static {
    const NAME: &'static str;

    /// Associated kernel type, parameterized by Element precision.
    type Kernels<E: Element>: Kernels<E>;

    /// Initialize kernels for a given precision.
    fn init<E: Element>() -> Self::Kernels<E>;
}

// ==========================================================================
// Kernels Trait (SPEC/03 §2.3) — 70+ operators
// ==========================================================================

/// The complete set of compute kernels.
///
/// All operators from SPEC/03 §2.3. Methods with default bodies are stubs
/// (`unimplemented!`) to allow incremental implementation.
pub trait Kernels<E: Element>: Send + Sync {

    // ======================================================================
    // BLAS-1: Vector operations
    // ======================================================================

    fn vec_dot(&self, _a: &[E], _b: &[E]) -> E { unimplemented!("vec_dot") }
    fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]);  // required
    fn vec_sub(&self, _a: &[E], _b: &[E], _out: &mut [E]) { unimplemented!("vec_sub") }
    fn vec_mul(&self, a: &[E], b: &[E], out: &mut [E]);  // required
    fn vec_scale(&self, _x: &mut [E], _s: E) { unimplemented!("vec_scale") }
    fn vec_axpy(&self, _y: &mut [E], _a: E, _x: &[E]) { unimplemented!("vec_axpy") }
    fn vec_sum(&self, _x: &[E]) -> E { unimplemented!("vec_sum") }
    fn vec_max(&self, _x: &[E]) -> E { unimplemented!("vec_max") }
    fn vec_sum_squares(&self, _x: &[E]) -> E { unimplemented!("vec_sum_squares") }

    // ======================================================================
    // BLAS-2/3: Matrix operations
    // ======================================================================

    fn gemv(&self, _a: &[E], _x: &[E], _y: &mut [E], _m: usize, _n: usize) {
        unimplemented!("gemv")
    }
    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize); // required
    fn gemm_bias(&self, _a: &[E], _b: &[E], _bias: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("gemm_bias")
    }
    /// Fused GEMM+bias+activation: C = act(A*B + bias)
    /// Activation is applied in-register before writeback, avoiding an extra C read/write pass.
    fn gemm_bias_act(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize, act: Activation) {
        // Default: unfused fallback — compute gemm_bias, then apply activation in-place
        self.gemm_bias(a, b, bias, c, m, n, k);
        match act {
            Activation::None => {},
            _ => {
                let len = m * n;
                for i in 0..len {
                    c[i] = match act {
                        Activation::Relu => E::max(c[i], E::ZERO),
                        Activation::Silu => {
                            let v = c[i].to_f32();
                            E::from_f32(v / (1.0 + (-v).exp()))
                        },
                        Activation::Gelu => {
                            let x = c[i].to_f32();
                            let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
                            E::from_f32(0.5 * x * (1.0 + inner.tanh()))
                        },
                        Activation::None => unreachable!(),
                    };
                }
            }
        }
    }
    fn pack_b(&self, _b: &[E], _n: usize, _k: usize) -> Vec<E> {
        unimplemented!("pack_b")
    }
    fn gemm_prepacked(&self, _a: &[E], _packed_b: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("gemm_prepacked")
    }
    fn gemm_bias_prepacked(&self, _a: &[E], _packed_b: &[E], _bias: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("gemm_bias_prepacked")
    }

    // ======================================================================
    // Activation functions
    // ======================================================================

    fn silu(&self, a: &[E], out: &mut [E]); // required
    fn gelu(&self, _x: &[E], _out: &mut [E]) { unimplemented!("gelu") }
    fn relu(&self, _x: &[E], _out: &mut [E]) { unimplemented!("relu") }
    fn tanh(&self, _x: &[E], _out: &mut [E]) { unimplemented!("tanh") }
    fn swiglu(&self, _gate: &[E], _up: &[E], _out: &mut [E]) { unimplemented!("swiglu") }
    fn softmax(&self, _x: &[E], _out: &mut [E]) { unimplemented!("softmax") }
    fn softmax_online(&self, _x: &[E], _out: &mut [E]) {
        // Default: fall back to regular softmax
        self.softmax(_x, _out);
    }
    fn softmax_3pass(&self, _x: &[E], _out: &mut [E]) {
        // Default: fall back to regular softmax
        self.softmax(_x, _out);
    }
    fn exp(&self, _x: &[E], _out: &mut [E]) { unimplemented!("exp") }

    // ======================================================================
    // Normalization
    // ======================================================================

    fn rms_norm(&self, _x: &[E], _weight: &[E], _out: &mut [E], _eps: f32) {
        unimplemented!("rms_norm")
    }
    fn layer_norm(&self, _x: &[E], _gamma: &[E], _beta: &[E], _out: &mut [E], _eps: f32) {
        unimplemented!("layer_norm")
    }

    // ======================================================================
    // Positional encoding
    // ======================================================================

    fn rope(&self, _qk: &mut [E], _cos: &[E], _sin: &[E], _head_dim: usize, _interleaved: bool) {
        unimplemented!("rope")
    }

    fn rope_with_pos(&self, _qk: &mut [E], _cos: &[E], _sin: &[E], _head_dim: usize, _position: usize, _interleaved: bool) {
        unimplemented!("rope_with_pos")
    }

    // ======================================================================
    // Dequantization (output fixed f32)
    // ======================================================================

    fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]); // required
    fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]); // required
    fn dequant_q2_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q2_k") }
    fn dequant_q3_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q3_k") }
    fn dequant_q5_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q5_k") }
    fn dequant_q6_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q6_k") }

    // Classic GGML dequantization (block_size=32)
    fn dequant_q4_0(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q4_0") }
    fn dequant_q4_1(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q4_1") }
    fn dequant_q5_0(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q5_0") }
    fn dequant_q5_1(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q5_1") }
    fn dequant_q8_0(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q8_0") }
    fn dequant_q8_1(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_q8_1") }

    // AWQ/GPTQ/Squeeze dequantization (SPEC §2.3)
    fn dequant_awq4(&self, _packed: &[u8], _zeros: &[u8], _scales: &[f16], _out: &mut [f32]) {
        unimplemented!("dequant_awq4")
    }
    fn dequant_gptq4(&self, _packed: &[u8], _g_idx: &[i32], _scales: &[f16], _out: &mut [f32]) {
        unimplemented!("dequant_gptq4")
    }
    fn dequant_squeeze(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_squeeze") }

    // ======================================================================
    // Quantized GEMV
    // ======================================================================

    fn gemv_q8(&self, _weight: &[i8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("gemv_q8")
    }
    fn gemv_q4(&self, _weight: &[u8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("gemv_q4")
    }
    fn gemv_q2(&self, _weight: &[u8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("gemv_q2")
    }
    fn gemv_q1(&self, _weight: &[u8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("gemv_q1")
    }
    fn gemm_q8(&self, _weight: &[i8], _input: &[E], _output: &mut [E], _scales: &[f32], _m: usize, _n: usize, _k: usize) {
        unimplemented!("gemm_q8")
    }
    fn gemm_q4(&self, _weight: &[u8], _input: &[E], _output: &mut [E], _scales: &[f32], _m: usize, _n: usize, _k: usize) {
        unimplemented!("gemm_q4")
    }

    // ======================================================================
    // IQ Quantized operations
    // ======================================================================

    fn dequant_iq1_s(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq1_s") }
    fn dequant_iq1_m(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq1_m") }
    fn dequant_iq2_xxs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq2_xxs") }
    fn dequant_iq2_xs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq2_xs") }
    fn dequant_iq2_s(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq2_s") }
    fn dequant_iq3_xxs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq3_xxs") }
    fn dequant_iq3_s(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq3_s") }
    fn dequant_iq4_nl(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq4_nl") }
    fn dequant_iq4_xs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("dequant_iq4_xs") }

    // ======================================================================
    // Quantized format-specific matmul (SPEC §2.3)
    // ======================================================================

    fn kquant_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _quant_type: QuantType, _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("kquant_matmul")
    }

    fn classic_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _quant_type: QuantType, _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("classic_matmul")
    }

    fn iq_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _quant_type: QuantType, _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("iq_matmul")
    }

    fn awq_matmul(
        &self, _weight: &[u8], _zeros: &[u8], _scales: &[f16],
        _input: &[E], _output: &mut [E],
        _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("awq_matmul")
    }

    fn gptq_matmul(
        &self, _weight: &[u8], _g_idx: &[i32], _scales: &[f16],
        _input: &[E], _output: &mut [E],
        _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("gptq_matmul")
    }

    fn squeeze_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("squeeze_matmul")
    }

}
