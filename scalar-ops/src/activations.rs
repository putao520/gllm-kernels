//! Scalar activation functions.

/// SiLU: `out[i] = x[i] / (1 + exp(-x[i]))`
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_silu(x: *const f32, out: *mut f32, n: usize) {
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
#[inline(never)]
pub unsafe extern "C" fn scalar_gelu(x: *const f32, out: *mut f32, n: usize) {
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

/// Tanh: `out[i] = tanh(x[i])` — RobertaClassificationHead 的中间激活。
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_tanh(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            *out.add(i) = (*x.add(i)).tanh();
        }
    }
}

/// ReLU: `out[i] = max(0, x[i])`
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_relu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            let v = *x.add(i);
            *out.add(i) = if v > 0.0 { v } else { 0.0 };
        }
    }
}

/// SwiGLU: `out[i] = silu(gate[i]) * up[i]`
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_swiglu(
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

/// Clipped SwiGLU (OpenAI gpt-oss-20b style):
/// `gate' = clamp(gate[i], -limit, +limit)`,
/// `up'   = clamp(up[i],   -limit, +limit)`,
/// `out[i] = silu(gate') * up'`.
///
/// `limit` is the symmetric magnitude clipping threshold
/// (e.g. `swiglu_limit = 7.0` in gpt-oss-20b).
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_swiglu_clipped(
    gate: *const f32,
    up: *const f32,
    out: *mut f32,
    n: usize,
    limit: f32,
) {
    for i in 0..n {
        unsafe {
            let g_raw = *gate.add(i);
            let u_raw = *up.add(i);
            // Symmetric clamp to [-limit, +limit].
            let g = g_raw.max(-limit).min(limit);
            let u = u_raw.max(-limit).min(limit);
            let silu_g = g / (1.0 + (-g).exp());
            *out.add(i) = silu_g * u;
        }
    }
}

/// GeGLU: `out[i] = gelu(gate[i]) * up[i]`
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_geglu(
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
    fn test_scalar_ops_swiglu_clipped() {
        // limit=7.0 behaves like unclipped for |x| <= 7, and saturates for |x| > 7.
        let gate = vec![0.0_f32, 1.0, 100.0, -100.0, 7.0, -7.0];
        let up   = vec![1.0_f32, 2.0, 3.0,    4.0,   5.0, 6.0];
        let mut out = vec![0.0_f32; 6];
        let limit = 7.0_f32;
        scalar_swiglu_clipped(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), gate.len(), limit);

        // swiglu(0, 1) = silu(0) * 1 = 0
        assert!(out[0].abs() < 1e-6);

        // swiglu(1, 2) — neither gate nor up exceeds ±7, so same as unclipped.
        let silu_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!((out[1] - silu_1 * 2.0).abs() < 1e-5);

        // swiglu(100, 3) → gate clamped to +7, up stays at 3 (|3| <= 7).
        // out = silu(7) * 3
        let silu_7 = 7.0_f32 / (1.0 + (-7.0_f32).exp());
        assert!((out[2] - silu_7 * 3.0).abs() < 1e-4,
            "got={}, expected={}", out[2], silu_7 * 3.0);

        // swiglu(-100, 4) → gate clamped to -7, up stays at 4.
        // out = silu(-7) * 4
        let silu_neg7 = -7.0_f32 / (1.0 + 7.0_f32.exp());
        assert!((out[3] - silu_neg7 * 4.0).abs() < 1e-5);

        // swiglu(7, 5) — boundary, neither clamped.
        assert!((out[4] - silu_7 * 5.0).abs() < 1e-4);

        // swiglu(-7, 6) — boundary on gate; up=6 within range.
        assert!((out[5] - silu_neg7 * 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_ops_swiglu_clipped_up_saturates() {
        // Verify clamp on `up` operand (up=100 saturates to +limit).
        let gate = vec![1.0_f32];
        let up   = vec![100.0_f32];
        let mut out = vec![0.0_f32; 1];
        let limit = 7.0_f32;
        scalar_swiglu_clipped(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), 1, limit);
        let silu_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        // up clamped from 100 → 7
        let expected = silu_1 * 7.0;
        assert!((out[0] - expected).abs() < 1e-5,
            "got={}, expected={}", out[0], expected);
    }

    #[test]
    fn test_scalar_ops_swiglu_clipped_differs_from_unclipped() {
        // At large magnitudes, clipped result must differ from unclipped.
        let gate = vec![50.0_f32];
        let up   = vec![50.0_f32];
        let mut out_clipped = vec![0.0_f32; 1];
        let mut out_unclipped = vec![0.0_f32; 1];
        scalar_swiglu(gate.as_ptr(), up.as_ptr(), out_unclipped.as_mut_ptr(), 1);
        scalar_swiglu_clipped(gate.as_ptr(), up.as_ptr(), out_clipped.as_mut_ptr(), 1, 7.0);
        // Unclipped ≈ 50 * 50 = 2500, clipped ≈ silu(7) * 7 ≈ 6.994 * 7 ≈ 48.96.
        assert!((out_unclipped[0] - out_clipped[0]).abs() > 100.0,
            "clipped must differ substantially: unclipped={}, clipped={}",
            out_unclipped[0], out_clipped[0]);
        // Clipped value bounded.
        assert!(out_clipped[0].abs() < 100.0);
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
