//! Generic QKV projection with RoPE - automatically selects optimal implementation.
//!
//! This module provides a unified interface for QKV projection that:
//! - Supports both separate and fused weight formats
//! - Automatically selects the optimal computation path
//! - Works with all data types through compile-time monomorphization
//!
//! # Weight Format Support
//!
//! ## Separate Weights (Optimal Path)
//! When q_weight, k_weight, v_weight are all provided:
//! - Uses 3 independent small matrix multiplications (optimal cache performance)
//! - Direct output to head-major format, no remapping needed
//! - Supports models like Qwen, Llama (separate q_proj/k_proj/v_proj weights)
//!
//! ## Fused Weight (Fallback Path)
//! When only fused_qkv_weight is provided:
//! - Automatically splits into Q/K/V parts, then does 3 independent linear calls
//! - Still better than old fused_qkv_rope because output is already separated
//! - Supports models with fused QKV weights

use crate::backend_trait::BackendError;
use crate::cpu_kernels::traits::{DTypeTrait, QkvWeightFormat};
use crate::cpu_kernels::quantization::linear_generic;
use half::f16;

/// Generic QKV projection with RoPE - unified interface for all data types.
///
/// This function automatically selects the optimal path:
/// - **Separate weights**: 3× independent small matrix multiplications (optimal)
/// - **Fused weights**: 1× large matrix multiplication split into 3 parts
///
/// # Output Format
///
/// Always outputs Q, K, V in head-major format:
/// - Q: [seq_len * num_heads * head_dim]
/// - K: [seq_len * num_kv_heads * head_dim]
/// - V: [seq_len * num_kv_heads * head_dim]
///
/// # Type Parameters
///
/// - `T`: Data type implementing `DTypeTrait`
///
/// # Performance
///
/// For SmolLM2 (hidden=576, q_out=576, kv_out=192, seq_len=5):
/// - Separate path: 3 × matmul([5, 576] × [576/192, 576]) = 3 × ~331K ops
/// - Fused path: 1 × matmul([5, 576] × [960, 576]) = ~553K ops, then remap overhead
/// - Separate path is ~2x faster due to better cache utilization
#[allow(clippy::too_many_arguments)]
pub fn qkv_projection_rope_generic<'a, T: DTypeTrait>(
    // Input
    input: &[f32],
    // Weights
    qkv_weights: &QkvWeightFormat<'a, T>,
    // Bias (optional)
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    v_bias: Option<&[f32]>,
    // Output buffers (must be pre-allocated)
    q_output: &mut [f32],
    k_output: &mut [f32],
    v_output: &mut [f32],
    // Dimensions
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scale: f32,
    rope_interleaved: bool,
    positions: &[i32],
) -> Result<(), BackendError> {
    let q_out = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| BackendError::InvalidConfig("q_out overflow".into()))?;
    let kv_out = num_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| BackendError::InvalidConfig("kv_out overflow".into()))?;

    match qkv_weights {
        QkvWeightFormat::Separated {
            q_weight,
            k_weight,
            v_weight,
            q_scales,
            k_scales,
            v_scales,
        } => {
            // Optimal path: 3 independent small matrix multiplications
            let q_expected = q_out * hidden_size;
            let kv_expected = kv_out * hidden_size;

            if q_weight.len() < q_expected || k_weight.len() < kv_expected || v_weight.len() < kv_expected {
                return Err(BackendError::InvalidConfig(
                    "separate QKV weight size mismatch".into(),
                ));
            }

            // Use empty scales for non-quantized weights
            let empty_scales: &[f16] = &[];

            // Q projection
            linear_generic::<T>(
                input,
                q_weight,
                q_scales.unwrap_or(empty_scales),
                q_bias,
                q_output,
                seq_len,
                q_out,
                hidden_size,
            )?;

            // K projection
            linear_generic::<T>(
                input,
                k_weight,
                k_scales.unwrap_or(empty_scales),
                k_bias,
                k_output,
                seq_len,
                kv_out,
                hidden_size,
            )?;

            // V projection
            linear_generic::<T>(
                input,
                v_weight,
                v_scales.unwrap_or(empty_scales),
                v_bias,
                v_output,
                seq_len,
                kv_out,
                hidden_size,
            )?;

            // Apply RoPE to separated Q and K buffers
            apply_rope_separated_generic(
                q_output,
                k_output,
                positions,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                rope_theta,
                rope_scale,
                rope_interleaved,
            );
        }
        QkvWeightFormat::Fused {
            qkv_weight,
            scales,
        } => {
            // Fallback path: split fused weight into 3 parts
            let fused_out = q_out + 2 * kv_out;
            let fused_expected = fused_out * hidden_size;

            if qkv_weight.len() < fused_expected {
                return Err(BackendError::InvalidConfig(
                    "fused QKV weight size mismatch".into(),
                ));
            }

            let empty_scales: &[f16] = &[];
            let scales = scales.unwrap_or(empty_scales);

            // Extract Q weight: first q_out rows
            linear_generic::<T>(
                input,
                &qkv_weight[..q_out * hidden_size],
                if !scales.is_empty() { &scales[..q_out] } else { empty_scales },
                q_bias,
                q_output,
                seq_len,
                q_out,
                hidden_size,
            )?;

            // Extract K weight: next kv_out rows
            linear_generic::<T>(
                input,
                &qkv_weight[q_out * hidden_size..q_out * hidden_size + kv_out * hidden_size],
                if !scales.is_empty() { &scales[q_out..q_out + kv_out] } else { empty_scales },
                k_bias,
                k_output,
                seq_len,
                kv_out,
                hidden_size,
            )?;

            // Extract V weight: last kv_out rows
            linear_generic::<T>(
                input,
                &qkv_weight[q_out * hidden_size + kv_out * hidden_size..],
                if !scales.is_empty() { &scales[q_out + kv_out..] } else { empty_scales },
                v_bias,
                v_output,
                seq_len,
                kv_out,
                hidden_size,
            )?;

            // Apply RoPE to separated Q and K buffers
            apply_rope_separated_generic(
                q_output,
                k_output,
                positions,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                rope_theta,
                rope_scale,
                rope_interleaved,
            );
        }
    }
    Ok(())
}

/// Apply RoPE to separated Q and K buffers (head-major format).
///
/// This is the generic version that works directly on the output buffers
/// without needing a RopeCache. For cached RoPE values, see the
/// apply_rope_separated function in mod.rs.
fn apply_rope_separated_generic(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[i32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scale: f32,
    _rope_interleaved: bool,
) {
    if rotary_dim == 0 || rotary_dim > head_dim {
        return;
    }

    let half = rotary_dim / 2;

    for pos_idx in 0..seq_len {
        let pos = positions[pos_idx];
        let pos_usize = if pos < 0 { 0 } else { pos as usize };

        // Apply RoPE to Q (all heads)
        for head in 0..num_heads {
            let q_base = pos_idx * num_heads * head_dim + head * head_dim;
            for dim in 0..half {
                let idx0 = dim;
                let idx1 = half + dim;
                let freq = rope_theta.powf(-2.0 * dim as f32 / rotary_dim as f32);

                let (cos_val, sin_val) = if pos == 0 {
                    (1.0f32, 0.0f32)
                } else {
                    let angle = pos_usize as f32 * rope_scale * freq;
                    (angle.cos(), angle.sin())
                };

                let q0 = q[q_base + idx0];
                let q1 = q[q_base + idx1];
                q[q_base + idx0] = q0 * cos_val - q1 * sin_val;
                q[q_base + idx1] = q0 * sin_val + q1 * cos_val;
            }
        }

        // Apply RoPE to K (all KV heads)
        for head in 0..num_kv_heads {
            let k_base = pos_idx * num_kv_heads * head_dim + head * head_dim;
            for dim in 0..half {
                let idx0 = dim;
                let idx1 = half + dim;
                let freq = rope_theta.powf(-2.0 * dim as f32 / rotary_dim as f32);

                let (cos_val, sin_val) = if pos == 0 {
                    (1.0f32, 0.0f32)
                } else {
                    let angle = pos_usize as f32 * rope_scale * freq;
                    (angle.cos(), angle.sin())
                };

                let k0 = k[k_base + idx0];
                let k1 = k[k_base + idx1];
                k[k_base + idx0] = k0 * cos_val - k1 * sin_val;
                k[k_base + idx1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_kernels::traits::*;

    #[test]
    fn test_qkv_projection_separated_f32() {
        let input = [1.0f32; 2 * 6]; // seq_len=2, hidden_size=6
        let q_weight = vec![1.0f32; 8 * 6]; // q_out=8 (num_heads=4 * head_dim=2)
        let k_weight = vec![1.0f32; 4 * 6]; // kv_out=4 (num_kv_heads=2 * head_dim=2)
        let v_weight = vec![1.0f32; 4 * 6];

        let mut q = vec![0.0f32; 2 * 8];
        let mut k = vec![0.0f32; 2 * 4];
        let mut v = vec![0.0f32; 2 * 4];
        let positions = [0i32, 1];

        let weights: QkvWeightFormat<'_, F32Type> = QkvWeightFormat::Separated {
            q_weight: &q_weight,
            k_weight: &k_weight,
            v_weight: &v_weight,
            q_scales: None,
            k_scales: None,
            v_scales: None,
        };

        qkv_projection_rope_generic::<F32Type>(
            &input,
            &weights,
            None, None, None,
            &mut q, &mut k, &mut v,
            2, 6, 4, 2, 2,
            2, 100000.0, 1.0, false, &positions,
        )
        .unwrap();

        // Verify outputs are non-zero
        assert!(q.iter().any(|&x| x != 0.0));
        assert!(k.iter().any(|&x| x != 0.0));
        assert!(v.iter().all(|&x| x != 0.0)); // V doesn't have RoPE, should be non-zero
    }

    #[test]
    fn test_qkv_projection_fused_f32() {
        let input = [1.0f32; 2 * 6]; // seq_len=2, hidden_size=6
        let qkv_weight = vec![1.0f32; (8 + 2 * 4) * 6]; // q_out=8, kv_out=4

        let mut q = vec![0.0f32; 2 * 8];
        let mut k = vec![0.0f32; 2 * 4];
        let mut v = vec![0.0f32; 2 * 4];
        let positions = [0i32, 1];

        let weights: QkvWeightFormat<'_, F32Type> = QkvWeightFormat::Fused {
            qkv_weight: &qkv_weight,
            scales: None,
        };

        qkv_projection_rope_generic::<F32Type>(
            &input,
            &weights,
            None, None, None,
            &mut q, &mut k, &mut v,
            2, 6, 4, 2, 2,
            2, 100000.0, 1.0, false, &positions,
        )
        .unwrap();

        // Verify outputs are non-zero
        assert!(q.iter().any(|&x| x != 0.0));
        assert!(k.iter().any(|&x| x != 0.0));
        assert!(v.iter().all(|&x| x != 0.0));
    }

    #[test]
    fn test_apply_rope_separated_position_0_unchanged() {
        let mut q = [1.0f32, 2.0, 3.0, 4.0];
        let mut k = [5.0f32, 6.0, 7.0, 8.0];
        let positions = [0i32];

        apply_rope_separated_generic(
            &mut q, &mut k, &positions,
            1, 1, 1, 2, 2,
            100000.0, 1.0, false,
        );

        // Position 0: cos=1, sin=0, values should be unchanged
        assert_eq!(q[0], 1.0);
        assert_eq!(q[1], 2.0);
        assert_eq!(q[2], 3.0);
        assert_eq!(q[3], 4.0);
        assert_eq!(k[0], 5.0);
        assert_eq!(k[1], 6.0);
        assert_eq!(k[2], 7.0);
        assert_eq!(k[3], 8.0);
    }

    #[test]
    fn test_apply_rope_separated_position_1_rotated() {
        let mut q = [1.0f32; 8];
        let mut k = [1.0f32; 4];
        let positions = [0i32, 1];

        // Copy original values for position 1
        let q_orig = q.clone();
        let k_orig = k.clone();

        apply_rope_separated_generic(
            &mut q, &mut k, &positions,
            2, 2, 1, 2, 2,
            100000.0, 1.0, false,
        );

        // Position 0 should be unchanged
        assert_eq!(q[0], 1.0);
        assert_eq!(k[0], 1.0);

        // Position 1 should be rotated (values changed)
        let q_changed = q.iter().zip(q_orig.iter()).any(|(&a, &b)| a != b);
        let k_changed = k.iter().zip(k_orig.iter()).any(|(&a, &b)| a != b);

        assert!(q_changed, "Q position 1 should be rotated");
        assert!(k_changed, "K position 1 should be rotated");
    }
}
