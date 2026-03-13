//! Scalar pooling functions.

/// MeanPool: average over the sequence dimension.
///
/// Input: [seq_len, hidden] (row-major), Output: [hidden]
/// `out[j] = sum(x[i * hidden + j] for i in 0..seq_len) / seq_len`
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_mean_pool(
    x: *const f32,
    out: *mut f32,
    seq_len: usize,
    hidden: usize,
) {
    unsafe {
        // Initialize output to zero
        for j in 0..hidden {
            *out.add(j) = 0.0;
        }

        // Accumulate rows
        for i in 0..seq_len {
            for j in 0..hidden {
                *out.add(j) += *x.add(i * hidden + j);
            }
        }

        // Divide by seq_len
        let inv_seq = 1.0_f32 / seq_len as f32;
        for j in 0..hidden {
            *out.add(j) *= inv_seq;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_mean_pool_basic() {
        // 2 rows, 3 cols: [[1, 2, 3], [4, 5, 6]] -> mean = [2.5, 3.5, 4.5]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0_f32; 3];

        scalar_mean_pool(x.as_ptr(), out.as_mut_ptr(), 2, 3);

        assert!((out[0] - 2.5).abs() < 1e-5);
        assert!((out[1] - 3.5).abs() < 1e-5);
        assert!((out[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_mean_pool_single_row() {
        // 1 row: identity
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];

        scalar_mean_pool(x.as_ptr(), out.as_mut_ptr(), 1, 4);

        for i in 0..4 {
            assert!(
                (out[i] - x[i]).abs() < 1e-5,
                "mean_pool_single[{i}]: got {}, expected {}",
                out[i], x[i]
            );
        }
    }

    #[test]
    fn test_scalar_mean_pool_uniform() {
        // All same value -> mean = that value
        let x = vec![3.0_f32; 12]; // 4 rows x 3 cols, all 3.0
        let mut out = vec![0.0_f32; 3];

        scalar_mean_pool(x.as_ptr(), out.as_mut_ptr(), 4, 3);

        for i in 0..3 {
            assert!(
                (out[i] - 3.0).abs() < 1e-5,
                "mean_pool_uniform[{i}]: got {}, expected 3.0",
                out[i]
            );
        }
    }
}
