//! Rotary Position Embedding (RoPE) implementation.
//!
//! RoPE encodes position information directly into the attention mechanism
//! by rotating query and key vectors based on their position in the sequence.
//!
//! This is a pure Rust implementation without framework dependencies.
//!
//! Reference: https://arxiv.org/abs/2104.09864

use crate::kernel_dispatcher::KernelFloat;

/// Configuration for RoPE operations.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Hidden dimension (head_dim, must be divisible by 2).
    pub dim: usize,
    /// Maximum sequence length for precomputed frequencies.
    pub max_seq_len: usize,
    /// Base frequency for position encoding (default: 10000.0).
    pub theta: f64,
    /// Optional NTK scaling factor for long context.
    pub ntk_factor: Option<f64>,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            max_seq_len: 8192,
            theta: 10000.0,
            ntk_factor: None,
        }
    }
}

/// Precompute cos/sin frequency tables for RoPE.
///
/// # Arguments
/// * `cos_out` - Output buffer for cosine values: [max_seq_len, dim/2]
/// * `sin_out` - Output buffer for sine values: [max_seq_len, dim/2]
/// * `config` - RoPE configuration
///
/// # Panics
/// Panics if output buffer sizes don't match expected dimensions.
pub fn rope_precompute(cos_out: &mut [f32], sin_out: &mut [f32], config: &RoPEConfig) {
    let half_dim = config.dim / 2;
    let expected_size = config.max_seq_len * half_dim;

    assert_eq!(
        cos_out.len(),
        expected_size,
        "cos_out size mismatch: expected {}, got {}",
        expected_size,
        cos_out.len()
    );
    assert_eq!(
        sin_out.len(),
        expected_size,
        "sin_out size mismatch: expected {}, got {}",
        expected_size,
        sin_out.len()
    );

    // Apply NTK scaling if specified
    let theta = if let Some(factor) = config.ntk_factor {
        config.theta * factor
    } else {
        config.theta
    };

    // Compute inverse frequencies: theta^(-2i/dim) for i in 0..dim/2
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| {
            let exponent = -2.0 * (i as f64) / (config.dim as f64);
            theta.powf(exponent)
        })
        .collect();

    // Compute frequencies for all positions
    for pos in 0..config.max_seq_len {
        for (i, &inv_f) in inv_freq.iter().enumerate() {
            let freq = (pos as f64) * inv_f;
            let idx = pos * half_dim + i;
            cos_out[idx] = freq.cos() as f32;
            sin_out[idx] = freq.sin() as f32;
        }
    }
}

/// Apply RoPE to query and key tensors (CPU implementation).
///
/// # Arguments
/// * `q` - Query tensor: [batch, seq, heads, head_dim]
/// * `k` - Key tensor: [batch, seq, kv_heads, head_dim]
/// * `cos_cache` - Precomputed cosine values: [max_seq, dim/2]
/// * `sin_cache` - Precomputed sine values: [max_seq, dim/2]
/// * `q_out` - Output query tensor (same shape as q)
/// * `k_out` - Output key tensor (same shape as k)
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `num_q_heads` - Number of query heads
/// * `num_kv_heads` - Number of key-value heads
/// * `head_dim` - Head dimension
/// * `position_offset` - Starting position offset
pub fn rope_apply<T: KernelFloat>(
    q: &[T],
    k: &[T],
    cos_cache: &[f32],
    sin_cache: &[f32],
    q_out: &mut [T],
    k_out: &mut [T],
    batch_size: usize,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    position_offset: usize,
) {
    let half_dim = head_dim / 2;

    // Validate input sizes
    let q_size = batch_size * seq_len * num_q_heads * head_dim;
    let k_size = batch_size * seq_len * num_kv_heads * head_dim;
    assert_eq!(q.len(), q_size, "q size mismatch");
    assert_eq!(k.len(), k_size, "k size mismatch");
    assert_eq!(q_out.len(), q_size, "q_out size mismatch");
    assert_eq!(k_out.len(), k_size, "k_out size mismatch");

    // Apply RoPE to query tensor
    for b in 0..batch_size {
        for s in 0..seq_len {
            let pos = position_offset + s;
            let cos_sin_offset = pos * half_dim;

            // Process query heads
            for h in 0..num_q_heads {
                let q_offset = ((b * seq_len + s) * num_q_heads + h) * head_dim;
                rotate_half(
                    &q[q_offset..q_offset + head_dim],
                    &mut q_out[q_offset..q_offset + head_dim],
                    &cos_cache[cos_sin_offset..cos_sin_offset + half_dim],
                    &sin_cache[cos_sin_offset..cos_sin_offset + half_dim],
                );
            }

            // Process key heads
            for h in 0..num_kv_heads {
                let k_offset = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
                rotate_half(
                    &k[k_offset..k_offset + head_dim],
                    &mut k_out[k_offset..k_offset + head_dim],
                    &cos_cache[cos_sin_offset..cos_sin_offset + half_dim],
                    &sin_cache[cos_sin_offset..cos_sin_offset + half_dim],
                );
            }
        }
    }
}

/// Apply rotation to a single head vector.
///
/// Rotation formula:
/// - x1' = x1 * cos - x2 * sin
/// - x2' = x2 * cos + x1 * sin
#[inline(always)]
fn rotate_half<T: KernelFloat>(x: &[T], out: &mut [T], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(half, cos.len());
    debug_assert_eq!(half, sin.len());

    for i in 0..half {
        let x1 = x[i].to_f32();
        let x2 = x[half + i].to_f32();
        let c = cos[i];
        let s = sin[i];

        // Apply rotation
        out[i] = T::from_f32(x1 * c - x2 * s);
        out[half + i] = T::from_f32(x2 * c + x1 * s);
    }
}

/// Apply RoPE to a single tensor in-place.
///
/// # Arguments
/// * `x` - Input/output tensor: [batch, seq, heads, head_dim]
/// * `cos_cache` - Precomputed cosine values: [max_seq, dim/2]
/// * `sin_cache` - Precomputed sine values: [max_seq, dim/2]
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `num_heads` - Number of heads
/// * `head_dim` - Head dimension
/// * `position_offset` - Starting position offset
pub fn rope_apply_inplace<T: KernelFloat>(
    x: &mut [T],
    cos_cache: &[f32],
    sin_cache: &[f32],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: usize,
) {
    let half_dim = head_dim / 2;

    for b in 0..batch_size {
        for s in 0..seq_len {
            let pos = position_offset + s;
            let cos_sin_offset = pos * half_dim;

            for h in 0..num_heads {
                let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                rotate_half_inplace(
                    &mut x[offset..offset + head_dim],
                    &cos_cache[cos_sin_offset..cos_sin_offset + half_dim],
                    &sin_cache[cos_sin_offset..cos_sin_offset + half_dim],
                );
            }
        }
    }
}

/// Apply rotation in-place.
#[inline(always)]
fn rotate_half_inplace<T: KernelFloat>(x: &mut [T], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;

    for i in 0..half {
        let x1 = x[i].to_f32();
        let x2 = x[half + i].to_f32();
        let c = cos[i];
        let s = sin[i];

        x[i] = T::from_f32(x1 * c - x2 * s);
        x[half + i] = T::from_f32(x2 * c + x1 * s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_precompute_basic() {
        let config = RoPEConfig {
            dim: 8,
            max_seq_len: 4,
            theta: 10000.0,
            ntk_factor: None,
        };

        let half_dim = config.dim / 2;
        let size = config.max_seq_len * half_dim;
        let mut cos = vec![0.0f32; size];
        let mut sin = vec![0.0f32; size];

        rope_precompute(&mut cos, &mut sin, &config);

        // Position 0 should have cos=1, sin=0 for all frequencies
        for i in 0..half_dim {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[{}] = {} (expected 1.0)", i, cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[{}] = {} (expected 0.0)", i, sin[i]);
        }

        // Non-zero positions should have varying cos/sin values
        for i in 0..half_dim {
            let idx = 1 * half_dim + i; // position 1
            assert!(
                cos[idx].abs() <= 1.0 && sin[idx].abs() <= 1.0,
                "cos/sin should be in [-1, 1]"
            );
        }
    }

    #[test]
    fn test_rope_apply_preserves_norm() {
        let config = RoPEConfig {
            dim: 8,
            max_seq_len: 16,
            theta: 10000.0,
            ntk_factor: None,
        };

        let half_dim = config.dim / 2;
        let cache_size = config.max_seq_len * half_dim;
        let mut cos = vec![0.0f32; cache_size];
        let mut sin = vec![0.0f32; cache_size];
        rope_precompute(&mut cos, &mut sin, &config);

        // Create test data: batch=1, seq=2, heads=1, head_dim=8
        let batch = 1;
        let seq = 2;
        let heads = 1;
        let head_dim = config.dim;
        let size = batch * seq * heads * head_dim;

        let q: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let k: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let mut q_out = vec![0.0f32; size];
        let mut k_out = vec![0.0f32; size];

        rope_apply(
            &q, &k, &cos, &sin, &mut q_out, &mut k_out,
            batch, seq, heads, heads, head_dim, 0,
        );

        // Verify norm is preserved (rotation shouldn't change norm)
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    let offset = ((b * seq + s) * heads + h) * head_dim;

                    // Compute original norm
                    let q_norm_orig: f32 = q[offset..offset + head_dim]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt();

                    // Compute rotated norm
                    let q_norm_rot: f32 = q_out[offset..offset + head_dim]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt();

                    assert!(
                        (q_norm_orig - q_norm_rot).abs() < 1e-4,
                        "Norm changed: {} -> {}",
                        q_norm_orig,
                        q_norm_rot
                    );
                }
            }
        }
    }

    #[test]
    fn test_rope_apply_inplace() {
        let config = RoPEConfig {
            dim: 8,
            max_seq_len: 8,
            theta: 10000.0,
            ntk_factor: None,
        };

        let half_dim = config.dim / 2;
        let cache_size = config.max_seq_len * half_dim;
        let mut cos = vec![0.0f32; cache_size];
        let mut sin = vec![0.0f32; cache_size];
        rope_precompute(&mut cos, &mut sin, &config);

        let batch = 1;
        let seq = 2;
        let heads = 1;
        let head_dim = config.dim;
        let size = batch * seq * heads * head_dim;

        // Test with copy vs inplace
        let original: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let mut x_copy = original.clone();
        let x_input = original.clone();
        let mut q_out = vec![0.0f32; size];
        let mut k_out = vec![0.0f32; size];

        // Use rope_apply with separate q and k (same data for comparison)
        rope_apply(
            &x_input, &x_input, &cos, &sin, &mut q_out, &mut k_out,
            batch, seq, heads, heads, head_dim, 0,
        );

        rope_apply_inplace(
            &mut x_copy, &cos, &sin,
            batch, seq, heads, head_dim, 0,
        );

        // Results should be the same (q_out should match x_copy since same input)
        for i in 0..size {
            assert!(
                (x_copy[i] - q_out[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                x_copy[i],
                q_out[i]
            );
        }
    }

    #[test]
    fn test_rope_with_ntk_scaling() {
        let config_base = RoPEConfig {
            dim: 8,
            max_seq_len: 4,
            theta: 10000.0,
            ntk_factor: None,
        };

        let config_ntk = RoPEConfig {
            dim: 8,
            max_seq_len: 4,
            theta: 10000.0,
            ntk_factor: Some(2.0),
        };

        let half_dim = config_base.dim / 2;
        let size = config_base.max_seq_len * half_dim;

        let mut cos_base = vec![0.0f32; size];
        let mut sin_base = vec![0.0f32; size];
        let mut cos_ntk = vec![0.0f32; size];
        let mut sin_ntk = vec![0.0f32; size];

        rope_precompute(&mut cos_base, &mut sin_base, &config_base);
        rope_precompute(&mut cos_ntk, &mut sin_ntk, &config_ntk);

        // NTK scaling should produce different frequencies
        let mut different = false;
        for i in 0..size {
            if (cos_base[i] - cos_ntk[i]).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "NTK scaling should produce different frequencies");
    }
}
