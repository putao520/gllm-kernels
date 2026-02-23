//! Scalar activation functions.

/// SiLU: `out[i] = x[i] / (1 + exp(-x[i]))`
#[no_mangle]
pub extern "C" fn scalar_silu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            let v = *x.add(i);
            *out.add(i) = v / (1.0 + (-v).exp());
        }
    }
}

/// GELU (tanh approximation):
/// `out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
#[no_mangle]
pub extern "C" fn scalar_gelu(x: *const f32, out: *mut f32, n: usize) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
    const COEFF: f32 = 0.044715_f32;
    for i in 0..n {
        unsafe {
            let v = *x.add(i);
            let inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
            *out.add(i) = 0.5 * v * (1.0 + inner.tanh());
        }
    }
}

/// ReLU: `out[i] = max(0, x[i])`
#[no_mangle]
pub extern "C" fn scalar_relu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            let v = *x.add(i);
            *out.add(i) = if v > 0.0 { v } else { 0.0 };
        }
    }
}

/// SwiGLU: `out[i] = silu(gate[i]) * up[i]`
#[no_mangle]
pub extern "C" fn scalar_swiglu(
    gate: *const f32,
    up: *const f32,
    out: *mut f32,
    n: usize,
) {
    for i in 0..n {
        unsafe {
            let g = *gate.add(i);
            let u = *up.add(i);
            let silu_g = g / (1.0 + (-g).exp());
            *out.add(i) = silu_g * u;
        }
    }
}

/// GeGLU: `out[i] = gelu(gate[i]) * up[i]`
#[no_mangle]
pub extern "C" fn scalar_geglu(
    gate: *const f32,
    up: *const f32,
    out: *mut f32,
    n: usize,
) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
    const COEFF: f32 = 0.044715_f32;
    for i in 0..n {
        unsafe {
            let g = *gate.add(i);
            let u = *up.add(i);
            let inner = SQRT_2_OVER_PI * (g + COEFF * g * g * g);
            let gelu_g = 0.5 * g * (1.0 + inner.tanh());
            *out.add(i) = gelu_g * u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_scalar(f: unsafe extern "C" fn(*const f32, *mut f32, usize), input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; input.len()];
        unsafe { f(input.as_ptr(), out.as_mut_ptr(), input.len()) };
        out
    }

    #[test]
    fn test_scalar_ops_silu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let out = run_scalar(scalar_silu, &input);
        // silu(0) = 0
        assert!((out[2]).abs() < 1e-6);
        // silu(x) ≈ x for large positive x
        assert!((out[4] - 2.0 * (1.0 / (1.0 + (-2.0_f32).exp()))).abs() < 1e-5);
        // silu is odd-ish: silu(-x) = -x * sigmoid(-x) != -silu(x), but silu(0)=0
        assert!(out[0] < 0.0); // silu(-2) < 0
        assert!(out[3] > 0.0); // silu(1) > 0
    }

    #[test]
    fn test_scalar_ops_gelu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let out = run_scalar(scalar_gelu, &input);
        // gelu(0) = 0
        assert!((out[2]).abs() < 1e-6);
        // gelu(x) ≈ x for large positive x
        assert!(out[4] > 1.9);
        // gelu(-2) ≈ -0.0454
        assert!((out[0] - (-0.0454)).abs() < 0.01);
    }

    #[test]
    fn test_scalar_ops_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let out = run_scalar(scalar_relu, &input);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_scalar_ops_swiglu() {
        let gate = vec![0.0, 1.0, 2.0, -1.0];
        let up = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        scalar_swiglu(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), 4);
        // swiglu(0, 1) = silu(0) * 1 = 0
        assert!((out[0]).abs() < 1e-6);
        // swiglu(1, 2) = silu(1) * 2
        let silu_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!((out[1] - silu_1 * 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_ops_geglu() {
        let gate = vec![0.0, 1.0, 2.0, -1.0];
        let up = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        scalar_geglu(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), 4);
        // geglu(0, 1) = gelu(0) * 1 = 0
        assert!((out[0]).abs() < 1e-6);
        // geglu(1, 2) = gelu(1) * 2
        let inner = 0.7978845608_f32 * (1.0 + 0.044715_f32);
        let gelu_1 = 0.5 * (1.0 + inner.tanh());
        assert!((out[1] - gelu_1 * 2.0).abs() < 1e-4);
    }
}
