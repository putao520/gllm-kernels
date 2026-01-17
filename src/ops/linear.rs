//! Zero-cost Linear layer operations.
//!
//! Provides compile-time inlined matrix multiplication for neural network linear layers.
//!
//! # Design
//!
//! Unlike Burn's `Linear<B>` which uses generic backends:
//! - Direct slice operations with `#[inline(always)]`
//! - No heap allocation during forward pass
//! - CPU fallback with optional BLAS acceleration
//!
//! # Example
//!
//! ```ignore
//! use gllm_kernels::ops::linear::{linear_forward, linear_forward_transposed};
//!
//! let input = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
//! let weight = vec![0.5, 0.5, 0.5, 0.5]; // [2, 2]
//! let mut output = vec![0.0; 4];
//!
//! linear_forward(&input, &weight, None, &mut output, 2, 2, 2);
//! ```

/// Linear forward pass: output = input @ weight^T + bias
///
/// Weight is stored as `[out_features, in_features]` (row-major), so we compute
/// `input @ weight^T` which is equivalent to `input @ (weight transposed)`.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, in_features]`
/// * `weight` - Weight matrix `[out_features, in_features]`
/// * `bias` - Optional bias vector `[out_features]`
/// * `output` - Output tensor `[batch, out_features]` (pre-allocated)
/// * `batch` - Batch size
/// * `in_features` - Number of input features
/// * `out_features` - Number of output features
///
/// # Panics
///
/// Panics in debug mode if slice lengths don't match expected sizes.
#[inline(always)]
pub fn linear_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) {
    debug_assert_eq!(input.len(), batch * in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), batch * out_features);

    // output[b, o] = sum_i(input[b, i] * weight[o, i]) + bias[o]
    for b in 0..batch {
        let in_row = &input[b * in_features..(b + 1) * in_features];
        let out_row = &mut output[b * out_features..(b + 1) * out_features];

        for o in 0..out_features {
            let weight_row = &weight[o * in_features..(o + 1) * in_features];
            let mut sum = 0.0f32;

            // Dot product with manual unrolling for better vectorization
            let chunks = in_features / 4;
            let remainder = in_features % 4;

            for c in 0..chunks {
                let idx = c * 4;
                sum += in_row[idx] * weight_row[idx]
                    + in_row[idx + 1] * weight_row[idx + 1]
                    + in_row[idx + 2] * weight_row[idx + 2]
                    + in_row[idx + 3] * weight_row[idx + 3];
            }

            let base = chunks * 4;
            for r in 0..remainder {
                sum += in_row[base + r] * weight_row[base + r];
            }

            out_row[o] = sum;
        }

        // Add bias if present
        if let Some(b) = bias {
            for o in 0..out_features {
                out_row[o] += b[o];
            }
        }
    }
}

/// Linear forward pass with pre-transposed weights.
///
/// Use this when weights are already stored as `[in_features, out_features]`
/// (column-major for the original weight matrix).
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, in_features]`
/// * `weight_t` - Transposed weight matrix `[in_features, out_features]`
/// * `bias` - Optional bias vector `[out_features]`
/// * `output` - Output tensor `[batch, out_features]` (pre-allocated)
/// * `batch` - Batch size
/// * `in_features` - Number of input features
/// * `out_features` - Number of output features
#[inline(always)]
pub fn linear_forward_transposed(
    input: &[f32],
    weight_t: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) {
    debug_assert_eq!(input.len(), batch * in_features);
    debug_assert_eq!(weight_t.len(), in_features * out_features);
    debug_assert_eq!(output.len(), batch * out_features);

    // Initialize output with bias or zeros
    if let Some(b) = bias {
        for batch_idx in 0..batch {
            let out_row = &mut output[batch_idx * out_features..(batch_idx + 1) * out_features];
            out_row.copy_from_slice(b);
        }
    } else {
        output.fill(0.0);
    }

    // output[b, o] = sum_i(input[b, i] * weight_t[i, o])
    // This is more cache-friendly when iterating over i first
    for b in 0..batch {
        let in_row = &input[b * in_features..(b + 1) * in_features];
        let out_row = &mut output[b * out_features..(b + 1) * out_features];

        for i in 0..in_features {
            let in_val = in_row[i];
            let weight_row = &weight_t[i * out_features..(i + 1) * out_features];

            // Vectorizable inner loop
            for o in 0..out_features {
                out_row[o] += in_val * weight_row[o];
            }
        }
    }
}

/// Add bias to output in-place.
///
/// # Arguments
///
/// * `output` - Output tensor `[batch, features]`
/// * `bias` - Bias vector `[features]`
/// * `batch` - Batch size
/// * `features` - Number of features
#[inline(always)]
pub fn add_bias(output: &mut [f32], bias: &[f32], batch: usize, features: usize) {
    debug_assert_eq!(output.len(), batch * features);
    debug_assert_eq!(bias.len(), features);

    for b in 0..batch {
        let row = &mut output[b * features..(b + 1) * features];
        for (o, b_val) in row.iter_mut().zip(bias.iter()) {
            *o += b_val;
        }
    }
}

/// Fused linear + activation (for MLP layers).
///
/// Computes `activation(input @ weight^T + bias)` in a single pass.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch, in_features]`
/// * `weight` - Weight matrix `[out_features, in_features]`
/// * `bias` - Optional bias vector `[out_features]`
/// * `output` - Output tensor `[batch, out_features]` (pre-allocated)
/// * `batch` - Batch size
/// * `in_features` - Number of input features
/// * `out_features` - Number of output features
/// * `activation` - Activation function to apply element-wise
#[inline(always)]
pub fn linear_forward_fused<F>(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    activation: F,
) where
    F: Fn(f32) -> f32,
{
    debug_assert_eq!(input.len(), batch * in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), batch * out_features);

    for b in 0..batch {
        let in_row = &input[b * in_features..(b + 1) * in_features];
        let out_row = &mut output[b * out_features..(b + 1) * out_features];

        for o in 0..out_features {
            let weight_row = &weight[o * in_features..(o + 1) * in_features];
            let mut sum = 0.0f32;

            for i in 0..in_features {
                sum += in_row[i] * weight_row[i];
            }

            if let Some(b) = bias {
                sum += b[o];
            }

            out_row[o] = activation(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_identity() {
        // Identity matrix weight
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity
        let mut output = vec![0.0; 4];

        linear_forward(&input, &weight, None, &mut output, 2, 2, 2);

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linear_forward_simple() {
        // input = [[1, 2], [3, 4]]
        // weight = [[1, 1], [1, 1]]
        // output = [[3, 3], [7, 7]]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];

        linear_forward(&input, &weight, None, &mut output, 2, 2, 2);

        assert_eq!(output, vec![3.0, 3.0, 7.0, 7.0]);
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![10.0, 20.0];
        let mut output = vec![0.0; 4];

        linear_forward(&input, &weight, Some(&bias), &mut output, 2, 2, 2);

        assert_eq!(output, vec![13.0, 23.0, 17.0, 27.0]);
    }

    #[test]
    fn test_linear_forward_transposed() {
        // Same as test_linear_forward_simple but with transposed weights
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Original weight = [[1, 1], [1, 1]]
        // Transposed weight_t = [[1, 1], [1, 1]] (same for this symmetric case)
        let weight_t = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];

        linear_forward_transposed(&input, &weight_t, None, &mut output, 2, 2, 2);

        assert_eq!(output, vec![3.0, 3.0, 7.0, 7.0]);
    }

    #[test]
    fn test_linear_fused_relu() {
        let input = vec![1.0, -2.0, -3.0, 4.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0; 4];

        linear_forward_fused(
            &input,
            &weight,
            None,
            &mut output,
            2,
            2,
            2,
            |x| x.max(0.0), // ReLU
        );

        assert_eq!(output, vec![1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn test_add_bias() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![10.0, 20.0];

        add_bias(&mut output, &bias, 2, 2);

        assert_eq!(output, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_linear_projection() {
        // Project from 3 features to 2 features
        let input = vec![1.0, 2.0, 3.0]; // [1, 3]
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3]
        let mut output = vec![0.0; 2]; // [1, 2]

        linear_forward(&input, &weight, None, &mut output, 1, 3, 2);

        assert_eq!(output, vec![1.0, 2.0]);
    }
}
