//! Zero-cost RMS Normalization operations.
//!
//! Root Mean Square Layer Normalization is commonly used in Transformer architectures
//! (LLaMA, Mistral, etc.) as a simpler alternative to LayerNorm.
//!
//! # Formula
//!
//! ```text
//! rms = sqrt(mean(x^2) + eps)
//! output = (x / rms) * weight
//! ```
//!
//! # Design
//!
//! - Direct slice operations with `#[inline(always)]`
//! - No heap allocation during forward pass
//! - Numerically stable computation

/// RMS Normalization forward pass.
///
/// Computes `(input / rms) * weight` where `rms = sqrt(mean(input^2) + eps)`.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, hidden]`
/// * `weight` - Scale parameter `[hidden]` (gamma)
/// * `output` - Output tensor `[batch, hidden]` (pre-allocated)
/// * `batch` - Batch size
/// * `hidden` - Hidden dimension size
/// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
///
/// # Panics
///
/// Panics in debug mode if slice lengths don't match expected sizes.
#[inline(always)]
pub fn rms_norm_forward(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch * hidden);
    debug_assert_eq!(weight.len(), hidden);
    debug_assert_eq!(output.len(), batch * hidden);

    for b in 0..batch {
        let in_row = &input[b * hidden..(b + 1) * hidden];
        let out_row = &mut output[b * hidden..(b + 1) * hidden];

        // Compute mean of squares using Kahan summation for numerical stability
        let mut sum_sq = 0.0f64;
        let mut compensation = 0.0f64;

        for &x in in_row {
            let x64 = x as f64;
            let y = x64 * x64 - compensation;
            let t = sum_sq + y;
            compensation = (t - sum_sq) - y;
            sum_sq = t;
        }

        let mean_sq = sum_sq / hidden as f64;
        let rms = ((mean_sq + eps as f64).sqrt()) as f32;
        let inv_rms = 1.0 / rms;

        // Normalize and scale
        for i in 0..hidden {
            out_row[i] = in_row[i] * inv_rms * weight[i];
        }
    }
}

/// RMS Normalization forward pass (in-place).
///
/// Modifies input tensor in-place.
///
/// # Arguments
///
/// * `data` - Input/output tensor `[batch, hidden]`
/// * `weight` - Scale parameter `[hidden]` (gamma)
/// * `batch` - Batch size
/// * `hidden` - Hidden dimension size
/// * `eps` - Small constant for numerical stability
#[inline(always)]
pub fn rms_norm_inplace(data: &mut [f32], weight: &[f32], batch: usize, hidden: usize, eps: f32) {
    debug_assert_eq!(data.len(), batch * hidden);
    debug_assert_eq!(weight.len(), hidden);

    for b in 0..batch {
        let row = &mut data[b * hidden..(b + 1) * hidden];

        // Compute RMS with Kahan summation
        let mut sum_sq = 0.0f64;
        let mut compensation = 0.0f64;

        for &x in row.iter() {
            let x64 = x as f64;
            let y = x64 * x64 - compensation;
            let t = sum_sq + y;
            compensation = (t - sum_sq) - y;
            sum_sq = t;
        }

        let mean_sq = sum_sq / hidden as f64;
        let rms = ((mean_sq + eps as f64).sqrt()) as f32;
        let inv_rms = 1.0 / rms;

        // Normalize and scale in-place
        for i in 0..hidden {
            row[i] = row[i] * inv_rms * weight[i];
        }
    }
}

/// Compute RMS value without normalization.
///
/// Useful for debugging or when you need the RMS value separately.
///
/// # Arguments
///
/// * `input` - Input slice
/// * `eps` - Small constant for numerical stability
///
/// # Returns
///
/// The RMS value: `sqrt(mean(input^2) + eps)`
#[inline(always)]
pub fn compute_rms(input: &[f32], eps: f32) -> f32 {
    if input.is_empty() {
        return eps.sqrt();
    }

    let mut sum_sq = 0.0f64;
    let mut compensation = 0.0f64;

    for &x in input {
        let x64 = x as f64;
        let y = x64 * x64 - compensation;
        let t = sum_sq + y;
        compensation = (t - sum_sq) - y;
        sum_sq = t;
    }

    let mean_sq = sum_sq / input.len() as f64;
    ((mean_sq + eps as f64).sqrt()) as f32
}

/// RMS Normalization with optional bias (less common variant).
///
/// Some models use `output = (input / rms) * weight + bias`.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, hidden]`
/// * `weight` - Scale parameter `[hidden]` (gamma)
/// * `bias` - Optional bias parameter `[hidden]` (beta)
/// * `output` - Output tensor `[batch, hidden]` (pre-allocated)
/// * `batch` - Batch size
/// * `hidden` - Hidden dimension size
/// * `eps` - Small constant for numerical stability
#[inline(always)]
pub fn rms_norm_forward_with_bias(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch * hidden);
    debug_assert_eq!(weight.len(), hidden);
    debug_assert_eq!(output.len(), batch * hidden);

    for b in 0..batch {
        let in_row = &input[b * hidden..(b + 1) * hidden];
        let out_row = &mut output[b * hidden..(b + 1) * hidden];

        // Compute RMS
        let mut sum_sq = 0.0f64;
        for &x in in_row {
            sum_sq += (x as f64) * (x as f64);
        }

        let mean_sq = sum_sq / hidden as f64;
        let rms = ((mean_sq + eps as f64).sqrt()) as f32;
        let inv_rms = 1.0 / rms;

        // Normalize, scale, and optionally add bias
        match bias {
            Some(b) => {
                debug_assert_eq!(b.len(), hidden);
                for i in 0..hidden {
                    out_row[i] = in_row[i] * inv_rms * weight[i] + b[i];
                }
            }
            None => {
                for i in 0..hidden {
                    out_row[i] = in_row[i] * inv_rms * weight[i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn test_rms_norm_basic() {
        // Simple case: input = [1, 1, 1, 1], weight = [1, 1, 1, 1]
        // mean(x^2) = 1, rms = 1, output = input / 1 * 1 = input
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];

        rms_norm_forward(&input, &weight, &mut output, 1, 4, EPS);

        for o in &output {
            assert!((o - 1.0).abs() < 1e-4, "Expected ~1.0, got {}", o);
        }
    }

    #[test]
    fn test_rms_norm_scaling() {
        // input = [2, 2, 2, 2], weight = [1, 1, 1, 1]
        // mean(x^2) = 4, rms = 2, output = 2/2 * 1 = 1
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];

        rms_norm_forward(&input, &weight, &mut output, 1, 4, EPS);

        for o in &output {
            assert!((o - 1.0).abs() < 1e-4, "Expected ~1.0, got {}", o);
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        // input = [1, 1, 1, 1], weight = [2, 2, 2, 2]
        // mean(x^2) = 1, rms = 1, output = 1/1 * 2 = 2
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0; 4];

        rms_norm_forward(&input, &weight, &mut output, 1, 4, EPS);

        for o in &output {
            assert!((o - 2.0).abs() < 1e-4, "Expected ~2.0, got {}", o);
        }
    }

    #[test]
    fn test_rms_norm_batch() {
        // Batch of 2
        let input = vec![1.0, 1.0, 2.0, 2.0]; // [2, 2]
        let weight = vec![1.0, 1.0];
        let mut output = vec![0.0; 4];

        rms_norm_forward(&input, &weight, &mut output, 2, 2, EPS);

        // First row: input=[1,1], rms=1, output=[1,1]
        assert!((output[0] - 1.0).abs() < 1e-4);
        assert!((output[1] - 1.0).abs() < 1e-4);

        // Second row: input=[2,2], rms=2, output=[1,1]
        assert!((output[2] - 1.0).abs() < 1e-4);
        assert!((output[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_inplace() {
        let mut data = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        rms_norm_inplace(&mut data, &weight, 1, 4, EPS);

        for d in &data {
            assert!((d - 1.0).abs() < 1e-4, "Expected ~1.0, got {}", d);
        }
    }

    #[test]
    fn test_compute_rms() {
        // input = [3, 4], mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5) ≈ 3.536
        let input = vec![3.0, 4.0];
        let rms = compute_rms(&input, EPS);

        let expected = (12.5f32 + EPS).sqrt();
        assert!(
            (rms - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            rms
        );
    }

    #[test]
    fn test_rms_norm_with_bias() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, 0.5, 0.5, 0.5];
        let mut output = vec![0.0; 4];

        rms_norm_forward_with_bias(&input, &weight, Some(&bias), &mut output, 1, 4, EPS);

        // rms = 1, output = 1/1 * 1 + 0.5 = 1.5
        for o in &output {
            assert!((o - 1.5).abs() < 1e-4, "Expected ~1.5, got {}", o);
        }
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small values
        let input = vec![1e-10, 1e-10, 1e-10, 1e-10];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];

        rms_norm_forward(&input, &weight, &mut output, 1, 4, EPS);

        // Should not produce NaN or Inf
        for o in &output {
            assert!(o.is_finite(), "Output should be finite, got {}", o);
        }
    }

    #[test]
    fn test_empty_compute_rms() {
        let rms = compute_rms(&[], EPS);
        assert!((rms - EPS.sqrt()).abs() < 1e-6);
    }
}
