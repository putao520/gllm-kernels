//! Scalar normalization functions.

/// RMSNorm: two-pass — compute RMS then scale.
///
/// `out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_rms_norm(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
) {
    unsafe {
        // Pass 1: sum of squares
        let mut sum_sq = 0.0_f32;
        for i in 0..n {
            let v = *x.add(i);
            sum_sq += v * v;
        }
        let scale = 1.0 / (sum_sq / n as f32 + eps).sqrt();

        // Pass 2: normalize and scale by weight
        for i in 0..n {
            *out.add(i) = *x.add(i) * scale * *weight.add(i);
        }
    }
}

/// LayerNorm: mean -> variance -> normalize -> scale + shift.
///
/// `out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_layer_norm(
    x: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
) {
    unsafe {
        // Pass 1: mean
        let mut sum = 0.0_f32;
        for i in 0..n {
            sum += *x.add(i);
        }
        let mean = sum / n as f32;

        // Pass 2: variance
        let mut var = 0.0_f32;
        for i in 0..n {
            let d = *x.add(i) - mean;
            var += d * d;
        }
        var /= n as f32;

        let scale = 1.0 / (var + eps).sqrt();

        // Pass 3: normalize, scale, shift
        for i in 0..n {
            let normed = (*x.add(i) - mean) * scale;
            *out.add(i) = normed * *weight.add(i) + *bias.add(i);
        }
    }
}


/// Value-Normalization: RMSNorm without learned scale.
///
/// `out[i] = x[i] / sqrt(mean(x^2) + eps)`
///
/// Identical to RmsNorm except there is no weight multiplication step.
/// Used by Gemma 4 for Value vector normalization.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_value_norm(
    input: *const f32,
    output: *mut f32,
    len: usize,
    eps: f32,
) {
    unsafe {
        let mut sum_sq = 0.0_f32;
        for i in 0..len {
            let v = *input.add(i);
            sum_sq += v * v;
        }
        let rms = (sum_sq / len as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..len {
            let v = *input.add(i);
            *output.add(i) = v * inv_rms;
        }
    }
}

/// L2 Normalize: two-pass — compute L2 norm then scale.
///
/// `out[i] = x[i] / sqrt(sum(x^2) + eps)`
///
/// Structure is NormLike (same as RmsNorm): reduce(sum_sq) → finalize(rsqrt) → transform(scale).
/// No weight parameter — pure geometric normalization.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_l2_normalize(
    x: *const f32,
    out: *mut f32,
    n: usize,
) {
    unsafe {
        // Pass 1: sum of squares
        let mut sum_sq = 0.0_f32;
        for i in 0..n {
            let v = *x.add(i);
            sum_sq += v * v;
        }
        let inv_norm = 1.0 / (sum_sq + 1e-12_f32).sqrt();

        // Pass 2: normalize
        for i in 0..n {
            *out.add(i) = *x.add(i) * inv_norm;
        }
    }
}
/// QK-Normalization: L2 normalize per-head then scale by √head_dim.
///
/// Used in Gemma 4 to replace Softcap. Each head vector is independently
/// L2-normalized and then scaled by √head_dim.
///
/// `Q_out[i] = Q[i] / ‖Q_head‖₂ × √head_dim` (same for K)
///
/// Input: [seq_len * num_heads * head_dim] (Q or K vector, contiguous per-head)
/// Output: same shape, normalized per-head.
///
/// Structure is NormLike: reduce(sum_sq per head) → finalize(rsqrt + scale) → transform(mul).
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_qk_norm(
    input: *const f32,
    output: *mut f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let scale = (head_dim as f32).sqrt();
    let total = seq_len * num_heads;
    for i in 0..total {
        let offset = i * head_dim;
        // L2 norm
        let mut sum_sq = 0.0f32;
        for d in 0..head_dim {
            let v = unsafe { *input.add(offset + d) };
            sum_sq += v * v;
        }
        let inv_norm = 1.0 / (sum_sq.sqrt() + 1e-6);
        // normalize + scale
        for d in 0..head_dim {
            let v = unsafe { *input.add(offset + d) };
            unsafe { *output.add(offset + d) = v * inv_norm * scale };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_ops_rms_norm() {
        let n = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![1.0_f32; n];
        let mut out = vec![0.0_f32; n];
        let eps = 1e-5_f32;

        scalar_rms_norm(x.as_ptr(), w.as_ptr(), out.as_mut_ptr(), n, eps);

        // Manual: rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps)
        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + eps).sqrt();
        for i in 0..n {
            let expected = x[i] / rms;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "rms_norm[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_ops_rms_norm_with_weight() {
        let n = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![2.0_f32, 0.5, 1.0, 3.0];
        let mut out = vec![0.0_f32; n];
        let eps = 1e-5_f32;

        scalar_rms_norm(x.as_ptr(), w.as_ptr(), out.as_mut_ptr(), n, eps);

        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + eps).sqrt();
        for i in 0..n {
            let expected = x[i] * w[i] / rms;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "rms_norm_w[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_ops_layer_norm() {
        let n = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![1.0_f32; n];
        let b = vec![0.0_f32; n];
        let mut out = vec![0.0_f32; n];
        let eps = 1e-5_f32;

        scalar_layer_norm(x.as_ptr(), w.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, eps);

        let mean = 2.5_f32;
        let var = ((1.0 - 2.5_f32).powi(2)
            + (2.0 - 2.5_f32).powi(2)
            + (3.0 - 2.5_f32).powi(2)
            + (4.0 - 2.5_f32).powi(2))
            / 4.0;
        let scale = 1.0 / (var + eps).sqrt();

        for i in 0..n {
            let expected = (x[i] - mean) * scale;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "layer_norm[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_ops_layer_norm_with_affine() {
        let n = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![2.0_f32, 0.5, 1.0, 3.0];
        let b = vec![0.1_f32, 0.2, 0.3, 0.4];
        let mut out = vec![0.0_f32; n];
        let eps = 1e-5_f32;

        scalar_layer_norm(x.as_ptr(), w.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, eps);

        let mean = 2.5_f32;
        let var = ((1.0 - 2.5_f32).powi(2)
            + (2.0 - 2.5_f32).powi(2)
            + (3.0 - 2.5_f32).powi(2)
            + (4.0 - 2.5_f32).powi(2))
            / 4.0;
        let scale = 1.0 / (var + eps).sqrt();

        for i in 0..n {
            let expected = (x[i] - mean) * scale * w[i] + b[i];
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "layer_norm_affine[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_l2_normalize() {
        let n = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; n];

        scalar_l2_normalize(x.as_ptr(), out.as_mut_ptr(), n);

        let norm = (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt();
        for i in 0..n {
            let expected = x[i] / norm;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "l2_normalize[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_l2_normalize_unit_vector() {
        // Already unit vector — should be unchanged
        let x = vec![0.0_f32, 0.0, 1.0, 0.0];
        let mut out = vec![0.0_f32; 4];

        scalar_l2_normalize(x.as_ptr(), out.as_mut_ptr(), 4);

        for i in 0..4 {
            assert!(
                (out[i] - x[i]).abs() < 1e-5,
                "l2_normalize_unit[{i}]: got {}, expected {}",
                out[i], x[i]
            );
        }
    }

    #[test]
    fn test_scalar_l2_normalize_zero_vector() {
        // Zero vector — should produce near-zero (eps prevents div-by-zero)
        let x = vec![0.0_f32; 4];
        let mut out = vec![1.0_f32; 4];

        scalar_l2_normalize(x.as_ptr(), out.as_mut_ptr(), 4);

        for i in 0..4 {
            assert!(
                out[i].abs() < 1e-3,
                "l2_normalize_zero[{i}]: got {}, expected ~0",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_value_norm() {
        let n = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; n];
        let eps = 1e-5_f32;

        scalar_value_norm(x.as_ptr(), out.as_mut_ptr(), n, eps);

        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + eps).sqrt();
        for i in 0..n {
            let expected = x[i] / rms;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "value_norm[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_qk_norm() {
        let head_dim = 4;
        let num_heads = 2;
        let seq_len = 1;
        // Two heads: [1,2,3,4] and [2,0,0,0]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0_f32; 8];

        scalar_qk_norm(x.as_ptr(), out.as_mut_ptr(), seq_len, num_heads, head_dim);

        let scale = (head_dim as f32).sqrt(); // 2.0

        // Head 0: norm = sqrt(1+4+9+16) = sqrt(30)
        let norm0 = (30.0_f32).sqrt();
        let inv0 = 1.0 / (norm0 + 1e-6);
        for d in 0..head_dim {
            let expected = x[d] * inv0 * scale;
            assert!(
                (out[d] - expected).abs() < 1e-5,
                "qk_norm head0[{d}]: got {}, expected {expected}",
                out[d]
            );
        }

        // Head 1: norm = sqrt(4) = 2.0
        let norm1 = 2.0_f32;
        let inv1 = 1.0 / (norm1 + 1e-6);
        for d in 0..head_dim {
            let expected = x[head_dim + d] * inv1 * scale;
            assert!(
                (out[head_dim + d] - expected).abs() < 1e-5,
                "qk_norm head1[{d}]: got {}, expected {expected}",
                out[head_dim + d]
            );
        }
    }

    #[test]
    fn test_scalar_qk_norm_multi_seq() {
        let head_dim = 2;
        let num_heads = 1;
        let seq_len = 2;
        let x = vec![3.0_f32, 4.0, 0.0, 1.0];
        let mut out = vec![0.0_f32; 4];

        scalar_qk_norm(x.as_ptr(), out.as_mut_ptr(), seq_len, num_heads, head_dim);

        let scale = (head_dim as f32).sqrt(); // sqrt(2)

        // Seq 0: norm = sqrt(9+16) = 5
        let inv0 = 1.0 / (5.0 + 1e-6);
        assert!((out[0] - 3.0 * inv0 * scale).abs() < 1e-5);
        assert!((out[1] - 4.0 * inv0 * scale).abs() < 1e-5);

        // Seq 1: norm = sqrt(0+1) = 1
        let inv1 = 1.0 / (1.0 + 1e-6);
        assert!((out[2] - 0.0 * inv1 * scale).abs() < 1e-5);
        assert!((out[3] - 1.0 * inv1 * scale).abs() < 1e-5);
    }
}
