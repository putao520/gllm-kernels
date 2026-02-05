use crate::backend_trait::{BackendError, BackendResult};
use libm::erff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluApproximation {
    Exact,
    Tanh,
}

pub fn gelu(
    input: &[f32],
    output: &mut [f32],
    approximation: GeluApproximation,
) -> BackendResult<()> {
    if input.len() != output.len() {
        return Err(BackendError::InvalidConfig(
            "gelu buffer size mismatch".into(),
        ));
    }
    match approximation {
        GeluApproximation::Exact => {
            for (out, &x) in output.iter_mut().zip(input.iter()) {
                *out = gelu_exact(x);
            }
        }
        GeluApproximation::Tanh => {
            for (out, &x) in output.iter_mut().zip(input.iter()) {
                *out = gelu_tanh(x);
            }
        }
    }
    Ok(())
}

fn gelu_exact(x: f32) -> f32 {
    const INV_SQRT_2: f32 = 0.7071067811865476;
    0.5 * x * (1.0 + erff(x * INV_SQRT_2))
}

fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const GELU_COEFF: f32 = 0.044715;
    let x3 = x * x * x;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + GELU_COEFF * x3)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_exact_matches_formula() {
        let input = [0.0f32, 1.0, -1.5];
        let mut output = [0.0f32; 3];
        gelu(&input, &mut output, GeluApproximation::Exact).unwrap();
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| 0.5 * x * (1.0 + erff(x * 0.7071067811865476)))
            .collect();
        for (out, exp) in output.iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_tanh_matches_formula() {
        let input = [0.0f32, 1.0, -1.5];
        let mut output = [0.0f32; 3];
        gelu(&input, &mut output, GeluApproximation::Tanh).unwrap();
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x3)).tanh())
            })
            .collect();
        for (out, exp) in output.iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_rejects_mismatch() {
        let input = [0.0f32, 1.0];
        let mut output = [0.0f32; 3];
        assert!(gelu(&input, &mut output, GeluApproximation::Exact).is_err());
    }
}
