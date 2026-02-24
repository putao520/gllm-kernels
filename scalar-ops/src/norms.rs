//! Scalar normalization functions.

/// RMSNorm: two-pass — compute RMS then scale.
///
/// `out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_rms_norm(
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
pub extern "C" fn scalar_layer_norm(
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
}
