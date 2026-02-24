/// Vector add: `out[i] = a[i] + b[i]`
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_vec_add(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            *out.add(i) = *a.add(i) + *b.add(i);
        }
    }
}

/// Vector mul: `out[i] = a[i] * b[i]`
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_vec_mul(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            *out.add(i) = *a.add(i) * *b.add(i);
        }
    }
}

/// Exp: `out[i] = exp(x[i])`
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_exp(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            *out.add(i) = (*x.add(i)).exp();
        }
    }
}

/// Softmax: `out[i] = exp(x[i] - max) / sum(exp(x - max))`
///
/// Numerically stable three-pass: max -> exp-sum -> normalize.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_softmax(x: *const f32, out: *mut f32, n: usize) {
    if n == 0 {
        return;
    }
    unsafe {
        let mut max_val = *x;
        for i in 1..n {
            let v = *x.add(i);
            if v > max_val {
                max_val = v;
            }
        }

        let mut sum = 0.0_f32;
        for i in 0..n {
            let e = (*x.add(i) - max_val).exp();
            *out.add(i) = e;
            sum += e;
        }

        let inv_sum = 1.0 / sum;
        for i in 0..n {
            *out.add(i) *= inv_sum;
        }
    }
}

/// GEMM: `C[i][j] += A[i][k] * B[k][j]` (row-major)
///
/// Naive triple-loop. A is [M,K], B is [K,N], C is [M,N].
/// C is assumed zero-initialized by the caller.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemm(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += *a.add(i * k + p) * *b.add(p * n + j);
                }
                *c.add(i * n + j) += acc;
            }
        }
    }
}

/// GEMM + bias: `C[i][j] = A[i][k] * B[k][j] + bias[j]` (row-major)
///
/// A is [M,K], B is [K,N], bias is [N], C is [M,N].
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_gemm_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += *a.add(i * k + p) * *b.add(p * n + j);
                }
                *c.add(i * n + j) = acc + *bias.add(j);
            }
        }
    }
}

/// Transpose 2D: `out[j][i] = input[i][j]`
///
/// input is [rows, cols], out is [cols, rows].
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_transpose_2d(
    input: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
) {
    unsafe {
        for i in 0..rows {
            for j in 0..cols {
                *out.add(j * rows + i) = *input.add(i * cols + j);
            }
        }
    }
}

/// Reshape (identity copy): `out[i] = input[i]`
///
/// Reshape is a layout-only operation; the scalar version is a memcpy.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_reshape(input: *const f32, out: *mut f32, n: usize) {
    unsafe {
        for i in 0..n {
            *out.add(i) = *input.add(i);
        }
    }
}

/// Dequantize (simplified uniform quantization): `out[i] = scale * quant[i]`
///
/// Minimal scalar reference for uniform block quantization.
/// Real quantization formats (Q4_0, Q8_0, etc.) have format-specific decode logic.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_dequantize(
    quant: *const f32,
    scale: *const f32,
    out: *mut f32,
    n: usize,
    block_size: usize,
) {
    unsafe {
        for i in 0..n {
            let block_idx = i / block_size;
            *out.add(i) = *quant.add(i) * *scale.add(block_idx);
        }
    }
}

/// Quantized GEMM (scalar reference): dequantize B on-the-fly during matmul.
///
/// A is [M,K] f32, B_quant is [K,N] quantized, B_scale is [K*N/block_size], C is [M,N].
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_quant_gemm(
    a: *const f32,
    b_quant: *const f32,
    b_scale: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    let b_idx = p * n + j;
                    let block_idx = b_idx / block_size;
                    let b_val = *b_quant.add(b_idx) * *b_scale.add(block_idx);
                    acc += *a.add(i * k + p) * b_val;
                }
                *c.add(i * n + j) = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_ops_vec_add() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0_f32; 4];
        scalar_vec_add(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_scalar_ops_vec_mul() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0_f32; 4];
        scalar_vec_mul(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        assert_eq!(out, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_scalar_ops_exp() {
        let x = vec![0.0_f32, 1.0, -1.0];
        let mut out = vec![0.0_f32; 3];
        scalar_exp(x.as_ptr(), out.as_mut_ptr(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((out[2] - (-1.0_f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_ops_softmax() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        scalar_softmax(x.as_ptr(), out.as_mut_ptr(), 4);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");

        for i in 1..4 {
            assert!(out[i] > out[i - 1], "softmax not monotonic at {i}");
        }
    }

    #[test]
    fn test_scalar_ops_softmax_numerical_stability() {
        let x = vec![1000.0_f32, 1001.0, 1002.0];
        let mut out = vec![0.0_f32; 3];
        scalar_softmax(x.as_ptr(), out.as_mut_ptr(), 3);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
        assert!(out.iter().all(|v| v.is_finite()), "softmax produced non-finite");
    }

    #[test]
    fn test_scalar_ops_gemm_identity() {
        let a = vec![1.0_f32, 0.0, 0.0, 1.0]; // I
        let b = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0_f32; 4];
        scalar_gemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_ops_gemm_small() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0_f32; 4];
        scalar_gemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_scalar_ops_gemm_rectangular() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let mut c = vec![0.0_f32; 1];
        scalar_gemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 1, 3);
        assert_eq!(c, vec![32.0]);
    }

    #[test]
    fn test_scalar_ops_gemm_bias() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let bias = vec![10.0_f32, 20.0];
        let mut c = vec![0.0_f32; 4];
        scalar_gemm_bias(a.as_ptr(), b.as_ptr(), bias.as_ptr(), c.as_mut_ptr(), 2, 2, 2);
        // C = A*B + bias = [19+10, 22+20; 43+10, 50+20] = [29, 42, 53, 70]
        assert_eq!(c, vec![29.0, 42.0, 53.0, 70.0]);
    }

    #[test]
    fn test_scalar_ops_transpose_2d() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let mut out = vec![0.0_f32; 6]; // [3,2]
        scalar_transpose_2d(input.as_ptr(), out.as_mut_ptr(), 2, 3);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_scalar_ops_reshape() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        scalar_reshape(input.as_ptr(), out.as_mut_ptr(), 4);
        assert_eq!(out, input);
    }

    #[test]
    fn test_scalar_ops_dequantize() {
        // 4 elements, block_size=2, 2 blocks
        let quant = vec![1.0_f32, 2.0, 3.0, 4.0];
        let scale = vec![0.5_f32, 2.0];
        let mut out = vec![0.0_f32; 4];
        scalar_dequantize(quant.as_ptr(), scale.as_ptr(), out.as_mut_ptr(), 4, 2);
        assert_eq!(out, vec![0.5, 1.0, 6.0, 8.0]);
    }
}
