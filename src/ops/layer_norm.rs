//! Zero-cost Layer Normalization operations.
//!
//! Standard Layer Normalization as used in BERT, GPT, and other Transformer models.
//!
//! # Formula
//!
//! ```text
//! mean = mean(x)
//! var = var(x)
//! output = (x - mean) / sqrt(var + eps) * gamma + beta
//! ```
//!
//! # Design
//!
//! - Direct slice operations with `#[inline(always)]`
//! - No heap allocation during forward pass
//! - Uses Welford's algorithm for numerical stability

/// Layer Normalization forward pass.
///
/// Computes `(input - mean) / sqrt(var + eps) * gamma + beta`.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, hidden]`
/// * `gamma` - Scale parameter `[hidden]`
/// * `beta` - Shift parameter `[hidden]`
/// * `output` - Output tensor `[batch, hidden]` (pre-allocated)
/// * `batch` - Batch size
/// * `hidden` - Hidden dimension size
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Panics
///
/// Panics in debug mode if slice lengths don't match expected sizes.
#[inline(always)]
pub fn layer_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch * hidden);
    debug_assert_eq!(gamma.len(), hidden);
    debug_assert_eq!(beta.len(), hidden);
    debug_assert_eq!(output.len(), batch * hidden);

    for b in 0..batch {
        let in_row = &input[b * hidden..(b + 1) * hidden];
        let out_row = &mut output[b * hidden..(b + 1) * hidden];

        // Compute mean and variance using Welford's algorithm
        let (mean, var) = welford_mean_var(in_row);

        let inv_std = 1.0 / ((var + eps).sqrt());

        // Normalize, scale, and shift
        for i in 0..hidden {
            out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

/// Layer Normalization forward pass (in-place).
///
/// Modifies input tensor in-place.
///
/// # Arguments
///
/// * `data` - Input/output tensor `[batch, hidden]`
/// * `gamma` - Scale parameter `[hidden]`
/// * `beta` - Shift parameter `[hidden]`
/// * `batch` - Batch size
/// * `hidden` - Hidden dimension size
/// * `eps` - Small constant for numerical stability
#[inline(always)]
pub fn layer_norm_inplace(
    data: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    debug_assert_eq!(data.len(), batch * hidden);
    debug_assert_eq!(gamma.len(), hidden);
    debug_assert_eq!(beta.len(), hidden);

    for b in 0..batch {
        let row = &mut data[b * hidden..(b + 1) * hidden];

        // Compute mean and variance
        let (mean, var) = welford_mean_var(row);
        let inv_std = 1.0 / ((var + eps).sqrt());

        // Normalize, scale, and shift in-place
        for i in 0..hidden {
            row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

/// Layer Normalization without affine parameters.
///
/// Just normalizes to zero mean and unit variance.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, hidden]`
/// * `output` - Output tensor `[batch, hidden]` (pre-allocated)
/// * `batch` - Batch size
/// * `hidden` - Hidden dimension size
/// * `eps` - Small constant for numerical stability
#[inline(always)]
pub fn layer_norm_no_affine(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch * hidden);
    debug_assert_eq!(output.len(), batch * hidden);

    for b in 0..batch {
        let in_row = &input[b * hidden..(b + 1) * hidden];
        let out_row = &mut output[b * hidden..(b + 1) * hidden];

        let (mean, var) = welford_mean_var(in_row);
        let inv_std = 1.0 / ((var + eps).sqrt());

        for i in 0..hidden {
            out_row[i] = (in_row[i] - mean) * inv_std;
        }
    }
}

/// Welford's online algorithm for computing mean and variance.
///
/// Numerically stable single-pass algorithm.
///
/// # Arguments
///
/// * `data` - Input slice
///
/// # Returns
///
/// Tuple of (mean, variance)
#[inline(always)]
pub fn welford_mean_var(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;

    for (i, &x) in data.iter().enumerate() {
        let x64 = x as f64;
        let delta = x64 - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x64 - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / data.len() as f64;
    (mean as f32, variance as f32)
}

/// Compute mean of a slice.
///
/// # Arguments
///
/// * `data` - Input slice
///
/// # Returns
///
/// The arithmetic mean
#[inline(always)]
pub fn compute_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // Kahan summation for numerical stability
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;

    for &x in data {
        let y = x as f64 - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    (sum / data.len() as f64) as f32
}

/// Compute variance of a slice (population variance).
///
/// # Arguments
///
/// * `data` - Input slice
/// * `mean` - Pre-computed mean
///
/// # Returns
///
/// The population variance
#[inline(always)]
pub fn compute_variance(data: &[f32], mean: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mean64 = mean as f64;
    let mut sum_sq = 0.0f64;

    for &x in data {
        let diff = x as f64 - mean64;
        sum_sq += diff * diff;
    }

    (sum_sq / data.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn test_layer_norm_basic() {
        // input = [1, 2, 3, 4], mean = 2.5, var = 1.25
        // normalized = [-1.34, -0.45, 0.45, 1.34] (approx)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 4];

        layer_norm_forward(&input, &gamma, &beta, &mut output, 1, 4, EPS);

        // Check that mean is ~0 and std is ~1
        let (mean, var) = welford_mean_var(&output);
        assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
        assert!((var - 1.0).abs() < 1e-2, "Var should be ~1, got {}", var);
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 2.0, 2.0, 2.0]; // scale by 2
        let beta = vec![1.0, 1.0, 1.0, 1.0]; // shift by 1
        let mut output = vec![0.0; 4];

        layer_norm_forward(&input, &gamma, &beta, &mut output, 1, 4, EPS);

        // After scaling by 2 and shifting by 1
        // mean should be 1, var should be 4
        let (mean, var) = welford_mean_var(&output);
        assert!((mean - 1.0).abs() < 1e-4, "Mean should be ~1, got {}", mean);
        assert!((var - 4.0).abs() < 1e-1, "Var should be ~4, got {}", var);
    }

    #[test]
    fn test_layer_norm_batch() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // [2, 4]
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 8];

        layer_norm_forward(&input, &gamma, &beta, &mut output, 2, 4, EPS);

        // Each row should be normalized independently
        let (mean1, var1) = welford_mean_var(&output[0..4]);
        let (mean2, var2) = welford_mean_var(&output[4..8]);

        assert!(mean1.abs() < 1e-4, "Row 1 mean should be ~0, got {}", mean1);
        assert!((var1 - 1.0).abs() < 1e-2, "Row 1 var should be ~1, got {}", var1);
        assert!(mean2.abs() < 1e-4, "Row 2 mean should be ~0, got {}", mean2);
        assert!((var2 - 1.0).abs() < 1e-2, "Row 2 var should be ~1, got {}", var2);
    }

    #[test]
    fn test_layer_norm_inplace() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        layer_norm_inplace(&mut data, &gamma, &beta, 1, 4, EPS);

        let (mean, var) = welford_mean_var(&data);
        assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
        assert!((var - 1.0).abs() < 1e-2, "Var should be ~1, got {}", var);
    }

    #[test]
    fn test_layer_norm_no_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        layer_norm_no_affine(&input, &mut output, 1, 4, EPS);

        let (mean, var) = welford_mean_var(&output);
        assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
        assert!((var - 1.0).abs() < 1e-2, "Var should be ~1, got {}", var);
    }

    #[test]
    fn test_welford_mean_var() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var) = welford_mean_var(&data);

        // mean = 3, var = 2
        assert!((mean - 3.0).abs() < 1e-5, "Mean should be 3, got {}", mean);
        assert!((var - 2.0).abs() < 1e-5, "Var should be 2, got {}", var);
    }

    #[test]
    fn test_compute_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = compute_mean(&data);
        assert!((mean - 3.0).abs() < 1e-5, "Mean should be 3, got {}", mean);
    }

    #[test]
    fn test_compute_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = compute_variance(&data, 3.0);
        assert!((var - 2.0).abs() < 1e-5, "Var should be 2, got {}", var);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with large values
        let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 4];

        layer_norm_forward(&input, &gamma, &beta, &mut output, 1, 4, EPS);

        // Should not produce NaN or Inf
        for o in &output {
            assert!(o.is_finite(), "Output should be finite, got {}", o);
        }

        let (mean, var) = welford_mean_var(&output);
        assert!(mean.abs() < 1e-3, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_empty_inputs() {
        assert_eq!(welford_mean_var(&[]), (0.0, 0.0));
        assert_eq!(compute_mean(&[]), 0.0);
        assert_eq!(compute_variance(&[], 0.0), 0.0);
    }
}
