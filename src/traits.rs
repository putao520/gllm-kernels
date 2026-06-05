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
    /// GeGLU: gelu(gate) * up — used by Gemma
    GeGlu,
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

    fn vec_dot(&self, _a: &[E], _b: &[E]) -> E { unimplemented!("Kernels::vec_dot not implemented for this backend") }
    fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]);  // required
    fn vec_sub(&self, _a: &[E], _b: &[E], _out: &mut [E]) { unimplemented!("Kernels::vec_sub not implemented for this backend") }
    fn vec_mul(&self, a: &[E], b: &[E], out: &mut [E]);  // required
    fn vec_scale(&self, _x: &mut [E], _s: E) { unimplemented!("Kernels::vec_scale not implemented for this backend") }
    fn vec_axpy(&self, _y: &mut [E], _a: E, _x: &[E]) { unimplemented!("Kernels::vec_axpy not implemented for this backend") }
    fn vec_sum(&self, _x: &[E]) -> E { unimplemented!("Kernels::vec_sum not implemented for this backend") }
    fn vec_max(&self, _x: &[E]) -> E { unimplemented!("Kernels::vec_max not implemented for this backend") }
    fn vec_sum_squares(&self, _x: &[E]) -> E { unimplemented!("Kernels::vec_sum_squares not implemented for this backend") }

    // ======================================================================
    // BLAS-2/3: Matrix operations
    // ======================================================================

    fn gemv(&self, _a: &[E], _x: &[E], _y: &mut [E], _m: usize, _n: usize) {
        unimplemented!("Kernels::gemv not implemented for this backend")
    }
    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize); // required
    /// GEMM with transposed B: C[M×N] = A[M×K] * B^T[N×K].
    /// b_t is stored as [N×K] row-major, i.e. b_t[j*k + ki] == original B[ki*n + j].
    fn gemm_bt(&self, _a: &[E], _b_t: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("Kernels::gemm_bt not implemented for this backend")
    }
    fn gemm_bias(&self, _a: &[E], _b: &[E], _bias: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("Kernels::gemm_bias not implemented for this backend")
    }
    /// Fused GEMM+bias+activation: C = act(A*B + bias)
    /// Activation is applied in-register before writeback, avoiding an extra C read/write pass.
    fn gemm_bias_act(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize, act: Activation) {
        // Default: unfused fallback — compute gemm_bias, then apply activation in-place
        self.gemm_bias(a, b, bias, c, m, n, k);
        let len = m * n;
        if matches!(act, Activation::None) {
            return;
        }
        // Single allocation reused for all activation types
        let input = c[..len].to_vec();
        match act {
            Activation::None => unreachable!(),
            Activation::Relu => self.relu(&input, &mut c[..len]),
            Activation::Silu => self.silu(&input, &mut c[..len]),
            Activation::Gelu | Activation::GeGlu => self.gelu(&input, &mut c[..len]),
        }
    }
    fn pack_b(&self, _b: &[E], _n: usize, _k: usize) -> Vec<E> {
        unimplemented!("Kernels::pack_b not implemented for this backend")
    }
    fn gemm_prepacked(&self, _a: &[E], _packed_b: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("Kernels::gemm_prepacked not implemented for this backend")
    }
    fn gemm_bias_prepacked(&self, _a: &[E], _packed_b: &[E], _bias: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
        unimplemented!("Kernels::gemm_bias_prepacked not implemented for this backend")
    }

    // ======================================================================
    // Activation functions
    // ======================================================================

    fn silu(&self, a: &[E], out: &mut [E]); // required
    fn gelu(&self, _x: &[E], _out: &mut [E]) { unimplemented!("Kernels::gelu not implemented for this backend") }
    fn relu(&self, _x: &[E], _out: &mut [E]) { unimplemented!("Kernels::relu not implemented for this backend") }
    fn tanh(&self, _x: &[E], _out: &mut [E]) { unimplemented!("Kernels::tanh not implemented for this backend") }
    fn swiglu(&self, _gate: &[E], _up: &[E], _out: &mut [E]) { unimplemented!("Kernels::swiglu not implemented for this backend") }
    fn softmax(&self, _x: &[E], _out: &mut [E]) { unimplemented!("Kernels::softmax not implemented for this backend") }
    fn exp(&self, _x: &[E], _out: &mut [E]) { unimplemented!("Kernels::exp not implemented for this backend") }

    // ======================================================================
    // Normalization
    // ======================================================================

    fn rms_norm(&self, _x: &[E], _weight: &[E], _out: &mut [E], _eps: f32) {
        unimplemented!("Kernels::rms_norm not implemented for this backend")
    }
    fn layer_norm(&self, _x: &[E], _gamma: &[E], _beta: &[E], _out: &mut [E], _eps: f32) {
        unimplemented!("Kernels::layer_norm not implemented for this backend")
    }

    // ======================================================================
    // Positional encoding
    // ======================================================================

    fn rope(&self, _qk: &mut [E], _cos: &[E], _sin: &[E], _head_dim: usize, _interleaved: bool) {
        unimplemented!("Kernels::rope not implemented for this backend")
    }

    fn rope_with_pos(&self, _qk: &mut [E], _cos: &[E], _sin: &[E], _head_dim: usize, _position: usize, _interleaved: bool) {
        unimplemented!("Kernels::rope_with_pos not implemented for this backend")
    }

    // ======================================================================
    // Dequantization (output fixed f32)
    // ======================================================================

    fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]); // required
    fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]); // required
    fn dequant_q2_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q2_k not implemented for this backend") }
    fn dequant_q3_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q3_k not implemented for this backend") }
    fn dequant_q5_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q5_k not implemented for this backend") }
    fn dequant_q6_k(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q6_k not implemented for this backend") }

    // Classic GGML dequantization (block_size=32)
    fn dequant_q4_0(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q4_0 not implemented for this backend") }
    fn dequant_q4_1(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q4_1 not implemented for this backend") }
    fn dequant_q5_0(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q5_0 not implemented for this backend") }
    fn dequant_q5_1(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q5_1 not implemented for this backend") }
    fn dequant_q8_0(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q8_0 not implemented for this backend") }
    fn dequant_q8_1(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_q8_1 not implemented for this backend") }

    // AWQ/GPTQ/Squeeze dequantization (SPEC §2.3)
    fn dequant_awq4(&self, _packed: &[u8], _zeros: &[u8], _scales: &[f16], _out: &mut [f32]) {
        unimplemented!("Kernels::dequant_awq4 not implemented for this backend")
    }
    fn dequant_gptq4(&self, _packed: &[u8], _g_idx: &[i32], _scales: &[f16], _out: &mut [f32]) {
        unimplemented!("Kernels::dequant_gptq4 not implemented for this backend")
    }
    fn dequant_squeeze(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_squeeze not implemented for this backend") }

    // ======================================================================
    // Quantized GEMV
    // ======================================================================

    fn gemv_q8(&self, _weight: &[i8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("Kernels::gemv_q8 not implemented for this backend")
    }
    fn gemv_q4(&self, _weight: &[u8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("Kernels::gemv_q4 not implemented for this backend")
    }
    fn gemv_q2(&self, _weight: &[u8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("Kernels::gemv_q2 not implemented for this backend")
    }
    fn gemv_q1(&self, _weight: &[u8], _input: &[E], _scale: f32, _n: usize) -> E {
        unimplemented!("Kernels::gemv_q1 not implemented for this backend")
    }
    fn gemm_q8(&self, _weight: &[i8], _input: &[E], _output: &mut [E], _scales: &[f32], _m: usize, _n: usize, _k: usize) {
        unimplemented!("Kernels::gemm_q8 not implemented for this backend")
    }
    fn gemm_q4(&self, _weight: &[u8], _input: &[E], _output: &mut [E], _scales: &[f32], _m: usize, _n: usize, _k: usize) {
        unimplemented!("Kernels::gemm_q4 not implemented for this backend")
    }

    // ======================================================================
    // IQ Quantized operations
    // ======================================================================

    fn dequant_iq1_s(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq1_s not implemented for this backend") }
    fn dequant_iq1_m(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq1_m not implemented for this backend") }
    fn dequant_iq2_xxs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq2_xxs not implemented for this backend") }
    fn dequant_iq2_xs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq2_xs not implemented for this backend") }
    fn dequant_iq2_s(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq2_s not implemented for this backend") }
    fn dequant_iq3_xxs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq3_xxs not implemented for this backend") }
    fn dequant_iq3_s(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq3_s not implemented for this backend") }
    fn dequant_iq4_nl(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq4_nl not implemented for this backend") }
    fn dequant_iq4_xs(&self, _block: &[u8], _out: &mut [f32]) { unimplemented!("Kernels::dequant_iq4_xs not implemented for this backend") }

    // ======================================================================
    // Quantized format-specific matmul (SPEC §2.3)
    // ======================================================================

    fn kquant_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _quant_type: QuantType, _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("Kernels::kquant_matmul not implemented for this backend")
    }

    fn classic_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _quant_type: QuantType, _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("Kernels::classic_matmul not implemented for this backend")
    }

    fn iq_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _quant_type: QuantType, _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("Kernels::iq_matmul not implemented for this backend")
    }

    fn awq_matmul(
        &self, _weight: &[u8], _zeros: &[u8], _scales: &[f16],
        _input: &[E], _output: &mut [E],
        _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("Kernels::awq_matmul not implemented for this backend")
    }

    fn gptq_matmul(
        &self, _weight: &[u8], _g_idx: &[i32], _scales: &[f16],
        _input: &[E], _output: &mut [E],
        _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("Kernels::gptq_matmul not implemented for this backend")
    }

    fn squeeze_matmul(
        &self, _weight_blocks: &[u8], _input: &[E], _output: &mut [E],
        _m: usize, _n: usize, _k: usize,
    ) {
        unimplemented!("Kernels::squeeze_matmul not implemented for this backend")
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Activation ──

    #[test]
    fn activation_variants() {
        assert_eq!(Activation::None, Activation::None);
        assert_ne!(Activation::Relu, Activation::Silu);
        assert_ne!(Activation::Gelu, Activation::GeGlu);
    }

    #[test]
    fn activation_count() {
        let all = [Activation::None, Activation::Relu, Activation::Silu, Activation::Gelu, Activation::GeGlu];
        assert_eq!(all.len(), 5);
    }

    // ── Element for f32 ──

    #[test]
    fn f32_element_constants() {
        assert_eq!(f32::ZERO, 0.0);
        assert_eq!(f32::ONE, 1.0);
        assert_eq!(f32::ELEM_ID, 0);
    }

    #[test]
    fn f32_element_roundtrip() {
        let v = 3.14f32;
        assert!((f32::from_f32(v).to_f32() - v).abs() < 1e-6);
    }

    #[test]
    fn f32_element_arithmetic() {
        let a = 5.0f32;
        let b = 3.0f32;
        assert!((a.elem_add(b) - 8.0).abs() < 1e-6);
        assert!((a.elem_sub(b) - 2.0).abs() < 1e-6);
        assert!((a.elem_mul(b) - 15.0).abs() < 1e-6);
        assert!((a.elem_div(b) - 5.0/3.0).abs() < 1e-6);
        assert!((a.neg() - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn f32_element_mul_add() {
        // mul_add(a, b) = self * a + b (fused multiply-add)
        let a = 2.0f32;
        assert!((a.mul_add(3.0, 4.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn f32_element_math() {
        let v = 4.0f32;
        assert!((v.sqrt() - 2.0).abs() < 1e-6);
        assert!((v.exp() - 4.0f64.exp() as f32).abs() < 1e-3);
        assert!((v.recip() - 0.25).abs() < 1e-6);
        assert!((v.abs() - 4.0).abs() < 1e-6);
        assert!((v.tanh() - 1.0).abs() < 0.01);
    }

    #[test]
    fn f32_element_min_max() {
        assert!((f32::max(1.0, 2.0) - 2.0).abs() < 1e-6);
        assert!((f32::min(1.0, 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f32_as_f32_slice() {
        let data = vec![1.0f32, 2.0, 3.0];
        let slice = f32::as_f32_slice(&data).unwrap();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[1], 2.0);

        let mut data_mut = vec![1.0f32, 2.0];
        let slice_mut = f32::as_f32_slice_mut(&mut data_mut).unwrap();
        slice_mut[0] = 99.0;
        assert_eq!(data_mut[0], 99.0);

        let v = 42.0f32;
        assert_eq!(*f32::as_f32_ref(&v).unwrap(), 42.0);
    }

    // ── Element for f16 ──

    #[test]
    fn f16_element_constants() {
        assert_eq!(f16::ELEM_ID, 1);
        assert!((f16::ZERO.to_f32()).abs() < 1e-6);
        assert!((f16::ONE.to_f32() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn f16_element_roundtrip() {
        let v = 2.5f32;
        let f = f16::from_f32(v);
        assert!((f.to_f32() - v).abs() < 0.01);
    }

    #[test]
    fn f16_element_arithmetic() {
        let a = f16::from_f32(6.0);
        let b = f16::from_f32(2.0);
        assert!((a.elem_add(b).to_f32() - 8.0).abs() < 0.1);
        assert!((a.elem_sub(b).to_f32() - 4.0).abs() < 0.1);
        assert!((a.elem_mul(b).to_f32() - 12.0).abs() < 0.1);
        assert!((a.elem_div(b).to_f32() - 3.0).abs() < 0.1);
    }

    #[test]
    fn f16_as_f32_slice_none() {
        let data = vec![f16::ZERO; 4];
        assert!(f16::as_f32_slice(&data).is_none());
        assert!(f16::as_f32_ref(&f16::ZERO).is_none());
    }

    // ── Element for bf16 ──

    #[test]
    fn bf16_element_constants() {
        assert_eq!(bf16::ELEM_ID, 2);
        assert!((bf16::ZERO.to_f32()).abs() < 1e-6);
        assert!((bf16::ONE.to_f32() - 1.0).abs() < 1e-2);
    }

    #[test]
    fn bf16_element_roundtrip() {
        let v = 3.14f32;
        let b = bf16::from_f32(v);
        assert!((b.to_f32() - v).abs() < 0.1);
    }

    #[test]
    fn bf16_element_arithmetic() {
        let a = bf16::from_f32(10.0);
        let b = bf16::from_f32(3.0);
        assert!((a.elem_add(b).to_f32() - 13.0).abs() < 0.5);
        assert!((a.elem_sub(b).to_f32() - 7.0).abs() < 0.5);
        assert!((a.elem_mul(b).to_f32() - 30.0).abs() < 1.0);
    }

    #[test]
    fn bf16_as_f32_slice_none() {
        let data = vec![bf16::ZERO; 4];
        assert!(bf16::as_f32_slice(&data).is_none());
    }

    // ── Element ELEM_ID uniqueness ──

    #[test]
    fn elem_id_unique() {
        let ids = [f32::ELEM_ID, f16::ELEM_ID, bf16::ELEM_ID];
        let mut sorted = ids.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "ELEM_ID values must be unique");
    }

    // ── Activation Debug, Clone, Copy ──

    #[test]
    fn activation_debug_format() {
        // Arrange & Act
        let debug_none = format!("{:?}", Activation::None);
        let debug_relu = format!("{:?}", Activation::Relu);
        let debug_silu = format!("{:?}", Activation::Silu);
        let debug_gelu = format!("{:?}", Activation::Gelu);
        let debug_geglu = format!("{:?}", Activation::GeGlu);
        // Assert
        assert!(!debug_none.is_empty());
        assert!(!debug_relu.is_empty());
        assert!(!debug_silu.is_empty());
        assert!(!debug_gelu.is_empty());
        assert!(!debug_geglu.is_empty());
    }

    #[test]
    fn activation_clone_copy() {
        // Arrange
        let a = Activation::Silu;
        // Act — Copy
        let b = a;
        // Assert
        assert_eq!(a, b);
        // Act — Clone
        let c = a.clone();
        assert_eq!(a, c);
    }

    #[test]
    fn activation_equality_all_pairs() {
        // Verify each variant equals itself and differs from others.
        let variants = [
            Activation::None, Activation::Relu, Activation::Silu,
            Activation::Gelu, Activation::GeGlu,
        ];
        for (i, vi) in variants.iter().enumerate() {
            for (j, vj) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(vi, vj);
                } else {
                    assert_ne!(vi, vj);
                }
            }
        }
    }

    // ── f32 Element: Default ──

    #[test]
    fn f32_element_default() {
        // Arrange & Act
        let default: f32 = Default::default();
        // Assert
        assert_eq!(default, 0.0f32);
    }

    // ── f16 Element: mul_add, neg, max, min, math functions ──

    #[test]
    fn f16_element_mul_add() {
        // Arrange
        let a = f16::from_f32(2.0);
        // Act: mul_add(a, b) = self + a * b
        let result = a.mul_add(f16::from_f32(3.0), f16::from_f32(4.0));
        // Assert: 2.0 + 3.0 * 4.0 = 14.0
        assert!((result.to_f32() - 14.0).abs() < 0.5);
    }

    #[test]
    fn f16_element_neg() {
        // Arrange
        let a = f16::from_f32(5.0);
        // Act
        let result = a.neg();
        // Assert
        assert!((result.to_f32() - (-5.0)).abs() < 0.1);
    }

    #[test]
    fn f16_element_max_min() {
        // Arrange
        let a = f16::from_f32(1.0);
        let b = f16::from_f32(3.0);
        // Act & Assert
        assert!((a.max(b).to_f32() - 3.0).abs() < 0.1);
        assert!((a.min(b).to_f32() - 1.0).abs() < 0.1);
    }

    #[test]
    fn f16_element_math_functions() {
        // Arrange
        let v = f16::from_f32(4.0);
        // Act & Assert
        assert!((v.sqrt().to_f32() - 2.0).abs() < 0.1);
        assert!((v.abs().to_f32() - 4.0).abs() < 0.1);
        assert!((v.recip().to_f32() - 0.25).abs() < 0.05);
        // exp(4) ≈ 54.6
        let exp_val = v.exp().to_f32();
        assert!((exp_val - 54.6).abs() < 2.0, "exp(4) ≈ 54.6, got {exp_val}");
        // tanh(4) ≈ 0.999
        assert!((v.tanh().to_f32() - 0.999).abs() < 0.01);
    }

    #[test]
    fn f16_as_f32_slice_mut_none() {
        // Arrange
        let mut data = vec![f16::ZERO; 4];
        // Act
        let result = f16::as_f32_slice_mut(&mut data);
        // Assert
        assert!(result.is_none());
    }

    // ── bf16 Element: mul_add, neg, max, min, math, div, slice_mut, as_f32_ref ──

    #[test]
    fn bf16_element_mul_add() {
        // Arrange
        let a = bf16::from_f32(2.0);
        // Act: mul_add(a, b) = self + a * b
        let result = a.mul_add(bf16::from_f32(3.0), bf16::from_f32(4.0));
        // Assert: 2.0 + 3.0 * 4.0 = 14.0
        assert!((result.to_f32() - 14.0).abs() < 0.5);
    }

    #[test]
    fn bf16_element_neg() {
        // Arrange
        let a = bf16::from_f32(5.0);
        // Act
        let result = a.neg();
        // Assert
        assert!((result.to_f32() - (-5.0)).abs() < 0.1);
    }

    #[test]
    fn bf16_element_max_min() {
        // Arrange
        let a = bf16::from_f32(1.0);
        let b = bf16::from_f32(3.0);
        // Act & Assert
        assert!((a.max(b).to_f32() - 3.0).abs() < 0.1);
        assert!((a.min(b).to_f32() - 1.0).abs() < 0.1);
    }

    #[test]
    fn bf16_element_div() {
        // Arrange
        let a = bf16::from_f32(10.0);
        let b = bf16::from_f32(4.0);
        // Act
        let result = a.elem_div(b);
        // Assert
        assert!((result.to_f32() - 2.5).abs() < 0.2);
    }

    #[test]
    fn bf16_element_math_functions() {
        // Arrange
        let v = bf16::from_f32(4.0);
        // Act & Assert
        assert!((v.sqrt().to_f32() - 2.0).abs() < 0.1);
        assert!((v.abs().to_f32() - 4.0).abs() < 0.1);
        assert!((v.recip().to_f32() - 0.25).abs() < 0.05);
        assert!((v.tanh().to_f32() - 0.999).abs() < 0.01);
    }

    #[test]
    fn bf16_as_f32_slice_mut_none() {
        // Arrange
        let mut data = vec![bf16::ZERO; 4];
        // Act
        let result = bf16::as_f32_slice_mut(&mut data);
        // Assert
        assert!(result.is_none());
    }

    #[test]
    fn bf16_as_f32_ref_none() {
        // Arrange
        let v = bf16::from_f32(1.0);
        // Act & Assert
        assert!(bf16::as_f32_ref(&v).is_none());
    }

    // ── DeviceRepr blanket impl ──

    #[test]
    fn device_repr_blanket_impl() {
        // f32 satisfies DeviceRepr via blanket impl.
        fn assert_device_repr<T: DeviceRepr>() {}
        assert_device_repr::<f32>();
        assert_device_repr::<f16>();
        assert_device_repr::<bf16>();
    }

    // ── Element Send + Sync bounds ──

    #[test]
    fn element_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<f32>();
        assert_send_sync::<f16>();
        assert_send_sync::<bf16>();
    }
}
