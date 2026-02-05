use crate::backend_trait::{BackendError, BackendResult};

pub fn mul(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BackendResult<()> {
    let len = ensure_same_len(lhs, rhs, output, "mul")?;
    for i in 0..len {
        output[i] = lhs[i] * rhs[i];
    }
    Ok(())
}

pub fn div(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BackendResult<()> {
    let len = ensure_same_len(lhs, rhs, output, "div")?;
    for i in 0..len {
        output[i] = lhs[i] / rhs[i];
    }
    Ok(())
}

pub fn sub(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BackendResult<()> {
    let len = ensure_same_len(lhs, rhs, output, "sub")?;
    for i in 0..len {
        output[i] = lhs[i] - rhs[i];
    }
    Ok(())
}

fn ensure_same_len(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    op: &str,
) -> BackendResult<usize> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(BackendError::InvalidConfig(format!(
            "{op} buffer size mismatch",
        )));
    }
    Ok(lhs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_matches_expected() {
        let lhs = [2.0f32, -1.0, 0.5];
        let rhs = [3.0f32, 4.0, -2.0];
        let mut output = [0.0f32; 3];
        mul(&lhs, &rhs, &mut output).unwrap();
        assert_eq!(output, [6.0, -4.0, -1.0]);
    }

    #[test]
    fn div_matches_expected() {
        let lhs = [6.0f32, -4.0, 1.0];
        let rhs = [3.0f32, 2.0, 4.0];
        let mut output = [0.0f32; 3];
        div(&lhs, &rhs, &mut output).unwrap();
        assert_eq!(output, [2.0, -2.0, 0.25]);
    }

    #[test]
    fn sub_matches_expected() {
        let lhs = [6.0f32, -4.0, 1.0];
        let rhs = [3.0f32, 2.0, 4.0];
        let mut output = [0.0f32; 3];
        sub(&lhs, &rhs, &mut output).unwrap();
        assert_eq!(output, [3.0, -6.0, -3.0]);
    }

    #[test]
    fn elementwise_rejects_mismatch() {
        let lhs = [1.0f32, 2.0];
        let rhs = [1.0f32; 3];
        let mut output = [0.0f32; 2];
        assert!(mul(&lhs, &rhs, &mut output).is_err());
    }
}
