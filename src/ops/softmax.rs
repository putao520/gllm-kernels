use crate::backend_trait::{BackendError, BackendResult};

pub fn softmax(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) -> BackendResult<()> {
    if cols == 0 {
        return Err(BackendError::InvalidConfig(
            "softmax requires non-zero cols".into(),
        ));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| BackendError::InvalidConfig("softmax size overflow".into()))?;
    if input.len() < total || output.len() < total {
        return Err(BackendError::InvalidConfig(
            "softmax buffer size mismatch".into(),
        ));
    }
    for row in 0..rows {
        let base = row * cols;
        let in_row = &input[base..base + cols];
        let out_row = &mut output[base..base + cols];
        softmax_row(in_row, out_row);
    }
    Ok(())
}

fn softmax_row(input: &[f32], output: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for (out, &v) in output.iter_mut().zip(input.iter()) {
        let exp = (v - max_val).exp();
        *out = exp;
        sum += exp;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for out in output.iter_mut() {
            *out *= inv;
        }
    } else {
        output.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_normalize() {
        let input = [0.0f32, 0.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];
        softmax(&input, &mut output, 2, 2).unwrap();
        for row in 0..2 {
            let base = row * 2;
            let sum = output[base] + output[base + 1];
            assert!((sum - 1.0).abs() < 1e-6);
        }
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let input = [1000.0f32, 1000.0];
        let mut output = [0.0f32; 2];
        softmax(&input, &mut output, 1, 2).unwrap();
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_rejects_invalid_dims() {
        let input = [0.0f32, 1.0];
        let mut output = [0.0f32; 2];
        assert!(softmax(&input, &mut output, 1, 0).is_err());
    }
}
