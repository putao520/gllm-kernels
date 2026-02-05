use crate::backend_trait::{BackendError, BackendResult};

pub fn layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> BackendResult<()> {
    if cols == 0 {
        return Err(BackendError::InvalidConfig(
            "layer_norm requires non-zero hidden size".into(),
        ));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| BackendError::InvalidConfig("layer_norm size overflow".into()))?;
    if input.len() < total || output.len() < total || weight.len() < cols {
        return Err(BackendError::InvalidConfig(
            "layer_norm buffer size mismatch".into(),
        ));
    }
    if let Some(bias) = bias {
        if bias.len() < cols {
            return Err(BackendError::InvalidConfig(
                "layer_norm bias size mismatch".into(),
            ));
        }
    }
    for row in 0..rows {
        let base = row * cols;
        let in_row = &input[base..base + cols];
        let out_row = &mut output[base..base + cols];
        let (mean, var) = mean_variance(in_row);
        let inv = (var + eps).sqrt().recip();
        apply_row(in_row, weight, bias, out_row, mean, inv);
    }
    Ok(())
}

fn mean_variance(slice: &[f32]) -> (f32, f32) {
    let len = slice.len() as f32;
    let mean = slice.iter().sum::<f32>() / len;
    let mut var = 0.0f32;
    for &v in slice {
        let diff = v - mean;
        var += diff * diff;
    }
    (mean, var / len)
}

fn apply_row(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    mean: f32,
    inv_std: f32,
) {
    match bias {
        Some(bias) => {
            for i in 0..input.len() {
                output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
            }
        }
        None => {
            for i in 0..input.len() {
                output[i] = (input[i] - mean) * inv_std * weight[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_matches_expected() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32, 1.0];
        let bias = [0.5f32, -0.5];
        let mut output = [0.0f32; 4];
        layer_norm(&input, &weight, Some(&bias), &mut output, 2, 2, 0.0).unwrap();
        let expected = [-0.5, 0.5, -0.5, 0.5];
        for (out, exp) in output.iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn layer_norm_supports_no_bias() {
        let input = [1.0f32, 2.0];
        let weight = [1.0f32, 1.0];
        let mut output = [0.0f32; 2];
        layer_norm(&input, &weight, None, &mut output, 1, 2, 0.0).unwrap();
        assert!((output[0] + 1.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_rejects_mismatch() {
        let input = [1.0f32, 2.0];
        let weight = [1.0f32; 1];
        let mut output = [0.0f32; 2];
        assert!(layer_norm(&input, &weight, None, &mut output, 1, 2, 1e-5).is_err());
    }
}
