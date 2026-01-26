//! CPU reference implementation of matrix multiplication.

use crate::kernel_types::{KernelFloat, MatmulConfig};

/// CPU reference implementation of matrix multiplication (GEMM).
///
/// C = alpha * A * B + beta * C
///
/// Where:
/// - A is [M, K]
/// - B is [K, N] (or [N, K] if transpose_b=true)
/// - C is [M, N]
#[inline(always)]
pub fn cpu_matmul<T: KernelFloat>(a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
    let MatmulConfig { m, k, n, transpose_b, alpha, beta } = config;

    debug_assert_eq!(a.len(), m * k, "A matrix size mismatch");
    debug_assert_eq!(c.len(), m * n, "C matrix size mismatch");

    if transpose_b {
        // B is stored as [N, K], we want B^T which is [K, N]
        debug_assert_eq!(b.len(), n * k, "B matrix size mismatch (transposed)");

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    // A[i, l] = a[i * k + l]
                    // B^T[l, j] = B[j, l] = b[j * k + l]
                    let a_val = a[i * k + l].to_f32() as f64;
                    let b_val = b[j * k + l].to_f32() as f64;
                    sum += a_val * b_val;
                }
                let c_idx = i * n + j;
                let c_val = if beta != 0.0 {
                    beta as f64 * c[c_idx].to_f32() as f64
                } else {
                    0.0
                };
                c[c_idx] = T::from_f32((alpha as f64 * sum + c_val) as f32);
            }
        }
    } else {
        // B is stored as [K, N]
        debug_assert_eq!(b.len(), k * n, "B matrix size mismatch");

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    // A[i, l] = a[i * k + l]
                    // B[l, j] = b[l * n + j]
                    let a_val = a[i * k + l].to_f32() as f64;
                    let b_val = b[l * n + j].to_f32() as f64;
                    sum += a_val * b_val;
                }
                let c_idx = i * n + j;
                let c_val = if beta != 0.0 {
                    beta as f64 * c[c_idx].to_f32() as f64
                } else {
                    0.0
                };
                c[c_idx] = T::from_f32((alpha as f64 * sum + c_val) as f32);
            }
        }
    }
}
