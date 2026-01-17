//! Zero-cost activation functions.
//!
//! Common activation functions used in neural networks, implemented as
//! direct slice operations with compile-time inlining.
//!
//! # Supported Activations
//!
//! - **SiLU/Swish**: `x * sigmoid(x)` - Used in LLaMA, Mistral
//! - **GELU**: Gaussian Error Linear Unit - Used in BERT, GPT
//! - **ReLU**: Rectified Linear Unit - Classic activation
//! - **Sigmoid**: Logistic function
//! - **Tanh**: Hyperbolic tangent
//!
//! # Design
//!
//! All functions have both in-place and out-of-place variants.
//! In-place variants modify the input directly, avoiding allocation.

// ============================================================================
// SiLU (Swish) Activation: x * sigmoid(x)
// ============================================================================

/// SiLU (Swish) activation in-place: `x = x * sigmoid(x)`.
///
/// Used in LLaMA, Mistral, and other modern LLMs.
#[inline(always)]
pub fn silu_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = *x * sigmoid_scalar(*x);
    }
}

/// SiLU (Swish) activation: `output = input * sigmoid(input)`.
#[inline(always)]
pub fn silu(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = x * sigmoid_scalar(x);
    }
}

/// Fused SiLU with element-wise multiplication: `x = silu(x) * gate`.
///
/// Common pattern in LLaMA MLP: `silu(gate_proj(x)) * up_proj(x)`.
#[inline(always)]
pub fn silu_mul_inplace(x: &mut [f32], gate: &[f32]) {
    debug_assert_eq!(x.len(), gate.len());
    for (xi, &gi) in x.iter_mut().zip(gate.iter()) {
        *xi = *xi * sigmoid_scalar(*xi) * gi;
    }
}

// ============================================================================
// GELU Activation: Gaussian Error Linear Unit
// ============================================================================

/// GELU activation in-place (approximate): `x = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
///
/// Uses the fast tanh approximation. Used in BERT, GPT-2.
#[inline(always)]
pub fn gelu_inplace(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEF: f32 = 0.044715;

    for x in data.iter_mut() {
        let x3 = *x * *x * *x;
        *x = 0.5 * *x * (1.0 + fast_tanh(SQRT_2_OVER_PI * (*x + COEF * x3)));
    }
}

/// GELU activation (approximate).
#[inline(always)]
pub fn gelu(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEF: f32 = 0.044715;

    for (i, &x) in input.iter().enumerate() {
        let x3 = x * x * x;
        output[i] = 0.5 * x * (1.0 + fast_tanh(SQRT_2_OVER_PI * (x + COEF * x3)));
    }
}

/// GELU activation in-place (exact using erf).
///
/// More accurate but slower: `x = 0.5 * x * (1 + erf(x / sqrt(2)))`.
#[inline(always)]
pub fn gelu_exact_inplace(data: &mut [f32]) {
    const INV_SQRT2: f32 = 0.7071067811865476;

    for x in data.iter_mut() {
        *x = 0.5 * *x * (1.0 + erf(*x * INV_SQRT2));
    }
}

/// GELU activation (exact using erf).
#[inline(always)]
pub fn gelu_exact(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    const INV_SQRT2: f32 = 0.7071067811865476;

    for (i, &x) in input.iter().enumerate() {
        output[i] = 0.5 * x * (1.0 + erf(x * INV_SQRT2));
    }
}

// ============================================================================
// ReLU Activation: max(0, x)
// ============================================================================

/// ReLU activation in-place: `x = max(0, x)`.
#[inline(always)]
pub fn relu_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = x.max(0.0);
    }
}

/// ReLU activation.
#[inline(always)]
pub fn relu(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.max(0.0);
    }
}

/// Leaky ReLU activation in-place: `x = max(alpha * x, x)`.
#[inline(always)]
pub fn leaky_relu_inplace(data: &mut [f32], alpha: f32) {
    for x in data.iter_mut() {
        *x = if *x > 0.0 { *x } else { alpha * *x };
    }
}

/// Leaky ReLU activation.
#[inline(always)]
pub fn leaky_relu(input: &[f32], output: &mut [f32], alpha: f32) {
    debug_assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = if x > 0.0 { x } else { alpha * x };
    }
}

// ============================================================================
// Sigmoid Activation: 1 / (1 + exp(-x))
// ============================================================================

/// Sigmoid activation in-place: `x = 1 / (1 + exp(-x))`.
#[inline(always)]
pub fn sigmoid_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = sigmoid_scalar(*x);
    }
}

/// Sigmoid activation.
#[inline(always)]
pub fn sigmoid(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = sigmoid_scalar(x);
    }
}

/// Sigmoid for a single value (numerically stable).
#[inline(always)]
pub fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ============================================================================
// Tanh Activation
// ============================================================================

/// Tanh activation in-place.
#[inline(always)]
pub fn tanh_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = x.tanh();
    }
}

/// Tanh activation.
#[inline(always)]
pub fn tanh_activation(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.tanh();
    }
}

/// Fast tanh approximation using rational function.
///
/// Accurate to ~1e-4 for |x| < 5.
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
    // Clamp to avoid overflow
    let x = x.clamp(-5.0, 5.0);
    let x2 = x * x;
    let a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    a / b
}

// ============================================================================
// Softplus: log(1 + exp(x))
// ============================================================================

/// Softplus activation in-place: `x = log(1 + exp(x))`.
#[inline(always)]
pub fn softplus_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = softplus_scalar(*x);
    }
}

/// Softplus activation.
#[inline(always)]
pub fn softplus(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = softplus_scalar(x);
    }
}

/// Softplus for a single value (numerically stable).
#[inline(always)]
pub fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 {
        x // Asymptotically linear
    } else if x < -20.0 {
        0.0 // Asymptotically zero
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Error function approximation (Abramowitz and Stegun).
#[inline(always)]
fn erf(x: f32) -> f32 {
    // Constants
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Element-wise multiplication in-place.
#[inline(always)]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai *= bi;
    }
}

/// Element-wise addition in-place.
#[inline(always)]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn test_silu() {
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut output = vec![0.0; 4];

        silu(&input, &mut output);

        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((output[0] - 0.0).abs() < EPS);
        // silu(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((output[1] - 0.731).abs() < 0.01);
        // silu(-1) = -1 * sigmoid(-1) ≈ -0.269
        assert!((output[2] - (-0.269)).abs() < 0.01);
    }

    #[test]
    fn test_silu_inplace() {
        let mut data = vec![0.0, 1.0, -1.0];
        silu_inplace(&mut data);

        assert!((data[0] - 0.0).abs() < EPS);
        assert!((data[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        gelu(&input, &mut output);

        // gelu(0) ≈ 0
        assert!((output[0] - 0.0).abs() < EPS);
        // gelu(1) ≈ 0.841
        assert!((output[1] - 0.841).abs() < 0.01);
        // gelu(-1) ≈ -0.159
        assert!((output[2] - (-0.159)).abs() < 0.01);
    }

    #[test]
    fn test_gelu_exact() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        gelu_exact(&input, &mut output);

        // Should be close to approximate version
        assert!((output[0] - 0.0).abs() < EPS);
        assert!((output[1] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        relu(&input, &mut output);

        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_inplace() {
        let mut data = vec![-1.0, 0.0, 1.0];
        relu_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        leaky_relu(&input, &mut output, 0.1);

        assert!((output[0] - (-0.2)).abs() < EPS);
        assert!((output[1] - (-0.1)).abs() < EPS);
        assert!((output[2] - 0.0).abs() < EPS);
        assert!((output[3] - 1.0).abs() < EPS);
        assert!((output[4] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_sigmoid() {
        let input = vec![0.0, 1.0, -1.0, 10.0, -10.0];
        let mut output = vec![0.0; 5];

        sigmoid(&input, &mut output);

        assert!((output[0] - 0.5).abs() < EPS);
        assert!((output[1] - 0.731).abs() < 0.01);
        assert!((output[2] - 0.269).abs() < 0.01);
        assert!((output[3] - 1.0).abs() < 0.001); // Very close to 1
        assert!((output[4] - 0.0).abs() < 0.001); // Very close to 0
    }

    #[test]
    fn test_sigmoid_numerical_stability() {
        // Test extreme values
        assert!(sigmoid_scalar(100.0).is_finite());
        assert!(sigmoid_scalar(-100.0).is_finite());
        assert!((sigmoid_scalar(100.0) - 1.0).abs() < 1e-10);
        assert!((sigmoid_scalar(-100.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tanh() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        tanh_activation(&input, &mut output);

        assert!((output[0] - 0.0).abs() < EPS);
        assert!((output[1] - 0.7616).abs() < 0.01);
        assert!((output[2] - (-0.7616)).abs() < 0.01);
    }

    #[test]
    fn test_fast_tanh() {
        // Compare with std tanh
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let fast = fast_tanh(x);
            let std = x.tanh();
            assert!(
                (fast - std).abs() < 0.01,
                "fast_tanh({}) = {}, std = {}",
                x,
                fast,
                std
            );
        }
    }

    #[test]
    fn test_softplus() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        softplus(&input, &mut output);

        // softplus(0) = ln(2) ≈ 0.693
        assert!((output[0] - 0.693).abs() < 0.01);
        // softplus(1) ≈ 1.313
        assert!((output[1] - 1.313).abs() < 0.01);
    }

    #[test]
    fn test_softplus_numerical_stability() {
        // Test extreme values
        assert!(softplus_scalar(100.0).is_finite());
        assert!(softplus_scalar(-100.0).is_finite());
        assert!((softplus_scalar(100.0) - 100.0).abs() < 0.01);
        assert!((softplus_scalar(-100.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_mul_inplace() {
        let mut x = vec![1.0, 2.0, 3.0];
        let gate = vec![2.0, 2.0, 2.0];

        silu_mul_inplace(&mut x, &gate);

        // silu(1) * 2 ≈ 0.731 * 2 ≈ 1.462
        assert!((x[0] - 1.462).abs() < 0.02);
    }

    #[test]
    fn test_mul_inplace() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];

        mul_inplace(&mut a, &b);

        assert_eq!(a, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_add_inplace() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];

        add_inplace(&mut a, &b);

        assert_eq!(a, vec![11.0, 22.0, 33.0]);
    }
}
