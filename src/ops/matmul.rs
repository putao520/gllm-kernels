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
    let MatmulConfig {
        m,
        k,
        n,
        transpose_a,
        transpose_b,
        alpha,
        beta,
    } = config;

    debug_assert_eq!(a.len(), m * k, "A matrix size mismatch");
    debug_assert_eq!(c.len(), m * n, "C matrix size mismatch");

    // B is stored as [K, N] unless transpose_b, but size is always k * n.
    debug_assert_eq!(b.len(), k * n, "B matrix size mismatch");

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..k {
                // A[i, l] = a[i * k + l] unless transpose_a, then A is [K, M].
                let a_idx = if transpose_a { l * m + i } else { i * k + l };
                // B[l, j] = b[l * n + j] unless transpose_b, then B is [N, K].
                let b_idx = if transpose_b { j * k + l } else { l * n + j };
                let a_val = a[a_idx].to_f32() as f64;
                let b_val = b[b_idx].to_f32() as f64;
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

#[cfg(test)]
mod tests {
    use super::cpu_matmul;
    use crate::kernel_types::MatmulConfig;

    #[test]
    fn cpu_matmul_no_transpose() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        let config = MatmulConfig {
            m: 2,
            k: 3,
            n: 2,
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        };

        cpu_matmul(&a, &b, &mut c, config);

        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_transpose_b() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_t = vec![7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0];
        let mut c = vec![0.0f32; 4];
        let config = MatmulConfig {
            m: 2,
            k: 3,
            n: 2,
            transpose_a: false,
            transpose_b: true,
            alpha: 1.0,
            beta: 0.0,
        };

        cpu_matmul(&a, &b_t, &mut c, config);

        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_transpose_a() {
        let a_t = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        let config = MatmulConfig {
            m: 2,
            k: 3,
            n: 2,
            transpose_a: true,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        };

        cpu_matmul(&a_t, &b, &mut c, config);

        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_alpha_beta() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let mut c = vec![10.0f32];
        let config = MatmulConfig {
            m: 1,
            k: 2,
            n: 1,
            transpose_a: false,
            transpose_b: false,
            alpha: 2.0,
            beta: 0.5,
        };

        cpu_matmul(&a, &b, &mut c, config);

        assert_eq!(c, vec![27.0]);
    }
}
