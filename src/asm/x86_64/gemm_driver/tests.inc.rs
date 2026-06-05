#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;

    /// Naive reference GEMM for correctness checking.
    fn gemm_ref(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn check_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(
                diff / scale < tol,
                "{label}[{i}]: asm={x}, ref={y}, diff={diff}"
            );
        }
    }

    // ── AVX2 tests ──

    #[test]
    fn test_avx2_exact_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx2_exact_tile");
    }

    #[test]
    fn test_avx2_multi_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (24, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx2_multi_tile");
    }

    #[test]
    fn test_avx2_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let m_padded = ((m + 5) / 6) * 6;
        let n_padded = ((n + 15) / 16) * 16;
        let mut c_asm = vec![0.0f32; m_padded * n_padded.max(n)];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let asm_val = c_asm[i * n + j];
                let ref_val = c_ref[i * n + j];
                let diff = (asm_val - ref_val).abs();
                let scale = asm_val.abs().max(ref_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_non_aligned[{i},{j}]: asm={asm_val}, ref={ref_val}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_avx2_identity() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 16);
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let expected = b[i * n + j];
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "avx2_identity[{i},{j}]: got={}, expected={expected}", c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx2_bias() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_asm_f32_avx2(&a, &b, &bias, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        check_close(&c_asm, &c_ref, 1e-4, "avx2_bias");
    }

    // ── AVX-512 tests ──

    #[test]
    fn test_avx512_exact_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx512_exact_tile");
    }

    #[test]
    fn test_avx512_multi_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx512_multi_tile");
    }

    #[test]
    fn test_avx512_non_aligned() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (15, 37, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let m_padded = ((m + 13) / 14) * 14;
        let n_padded = ((n + 31) / 32) * 32;
        let mut c_asm = vec![0.0f32; m_padded * n_padded.max(n)];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let asm_val = c_asm[i * n + j];
                let ref_val = c_ref[i * n + j];
                let diff = (asm_val - ref_val).abs();
                let scale = asm_val.abs().max(ref_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx512_non_aligned[{i},{j}]: asm={asm_val}, ref={ref_val}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_avx512_identity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 32);
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let expected = b[i * n + j];
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "avx512_identity[{i},{j}]: got={}, expected={expected}", c[i * n + j]
                );
            }
        }
    }

    // ── Prepacked AVX2 tests ──

    #[test]
    fn test_avx2_prepacked_exact_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx2_prepacked_exact_tile");
    }

    #[test]
    fn test_avx2_prepacked_multi_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (24, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx2_prepacked_multi_tile");
    }

    #[test]
    fn test_avx2_prepacked_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_pp[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_pp[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_prepacked_non_aligned[{i},{j}]: pp={}, ref={}, diff={diff}",
                    c_pp[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx2_prepacked_vs_regular() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (48, 96, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_asm_f32_avx2(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_pp, &c_reg, 1e-5, "avx2_prepacked_vs_regular");
    }

    #[test]
    fn test_avx2_prepacked_bias() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (12, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_prepacked_asm_f32_avx2(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        check_close(&c_pp, &c_ref, 1e-4, "avx2_prepacked_bias");
    }

    #[test]
    fn test_avx2_prepacked_bias_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_prepacked_asm_f32_avx2(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        for i in 0..m {
            for j in 0..n {
                let diff = (c_pp[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_pp[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_prepacked_bias_non_aligned[{i},{j}]: pp={}, ref={}, diff={diff}",
                    c_pp[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    // ── Prepacked AVX-512 tests ──

    #[test]
    fn test_avx512_prepacked_exact_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx512_prepacked_exact_tile");
    }

    #[test]
    fn test_avx512_prepacked_multi_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx512_prepacked_multi_tile");
    }

    #[test]
    fn test_avx512_prepacked_non_aligned() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (15, 37, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_pp[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_pp[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx512_prepacked_non_aligned[{i},{j}]: pp={}, ref={}, diff={diff}",
                    c_pp[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx512_prepacked_vs_regular() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_asm_f32_avx512(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_pp, &c_reg, 1e-5, "avx512_prepacked_vs_regular");
    }

    #[test]
    fn test_avx512_prepacked_bias() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (28, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_prepacked_asm_f32_avx512(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        check_close(&c_pp, &c_ref, 1e-4, "avx512_prepacked_bias");
    }

    // ── Prepacked AB (shared pack_a) AVX2 tests ──

    #[test]
    fn test_avx2_prepacked_ab_exact_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_ab, &c_ref, 1e-4, "avx2_prepacked_ab_exact_tile");
    }

    #[test]
    fn test_avx2_prepacked_ab_multi_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (24, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_ab, &c_ref, 1e-4, "avx2_prepacked_ab_multi_tile");
    }

    #[test]
    fn test_avx2_prepacked_ab_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_ab[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_ab[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_prepacked_ab_non_aligned[{i},{j}]: ab={}, ref={}, diff={diff}",
                    c_ab[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx2_prepacked_ab_vs_regular() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (48, 96, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_asm_f32_avx2(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_ab, &c_reg, 1e-5, "avx2_prepacked_ab_vs_regular");
    }

    /// Verify pack_a output matches what the internal pack_a_f32 produces
    /// for a single KC-block.
    #[test]
    fn test_avx2_pack_a_matches_internal() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use super::super::gemm_avx2::MR;
        let kc = crate::microarch::kernel_config().kc;
        let (m, k) = (13, 37); // non-aligned

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.07).collect();
        let shared = pack_a_asm_f32_avx2(&a, m, k);

        // Manually pack the first KC-block using pack_a_f32 and compare
        let kc_len = kc.min(k);
        let n_mr_strips = (m + MR - 1) / MR;
        let mut manual = vec![0.0f32; n_mr_strips * MR * kc_len];
        unsafe {
            pack_a_f32(a.as_ptr(), k, manual.as_mut_ptr(), m, kc_len, MR);
        }

        // The first KC-block in shared should match manual
        // shared layout: kc_block=0 → offset 0, each strip is kc*MR floats
        // manual layout: each strip is kc_len*MR floats (contiguous)
        for strip in 0..n_mr_strips {
            let shared_off = strip * kc * MR; // kc (max) stride in shared
            let manual_off = strip * kc_len * MR;
            for kl in 0..kc_len {
                for r in 0..MR {
                    let sv = shared.as_slice()[shared_off + kl * MR + r];
                    let mv = manual[manual_off + kl * MR + r];
                    assert!(
                        (sv - mv).abs() < 1e-6,
                        "pack_a mismatch strip={strip} k={kl} r={r}: shared={sv}, manual={mv}"
                    );
                }
            }
        }
    }

    /// QKV-style shared pack_a reuse: pack A once, multiply with 3 different B matrices.
    #[test]
    fn test_avx2_shared_pack_a_qkv() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (12, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let bq: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let bk: Vec<f32> = (0..k * n).map(|i| ((i % 71) as f32 - 35.0) * 0.03).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i % 59) as f32 - 29.0) * 0.04).collect();

        // Pack A once
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_bq = pack_b_asm_f32_avx2(&bq, n, k);
        let packed_bk = pack_b_asm_f32_avx2(&bk, n, k);
        let packed_bv = pack_b_asm_f32_avx2(&bv, n, k);

        // Compute Q, K, V with shared packed A
        let mut cq = vec![0.0f32; m * n];
        let mut ck = vec![0.0f32; m * n];
        let mut cv = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_bq, &mut cq, m, n, k);
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_bk, &mut ck, m, n, k);
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_bv, &mut cv, m, n, k);

        // Reference
        let mut cq_ref = vec![0.0f32; m * n];
        let mut ck_ref = vec![0.0f32; m * n];
        let mut cv_ref = vec![0.0f32; m * n];
        gemm_ref(&a, &bq, &mut cq_ref, m, n, k);
        gemm_ref(&a, &bk, &mut ck_ref, m, n, k);
        gemm_ref(&a, &bv, &mut cv_ref, m, n, k);

        check_close(&cq, &cq_ref, 1e-4, "shared_pack_a_Q");
        check_close(&ck, &ck_ref, 1e-4, "shared_pack_a_K");
        check_close(&cv, &cv_ref, 1e-4, "shared_pack_a_V");
    }

    // ── Prepacked AB AVX-512 tests ──

    #[test]
    fn test_avx512_prepacked_ab_exact_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx512(&a, m, k);
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx512(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_ab, &c_ref, 1e-4, "avx512_prepacked_ab_exact_tile");
    }

    #[test]
    fn test_avx512_prepacked_ab_non_aligned() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (15, 37, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_a = pack_a_asm_f32_avx512(&a, m, k);
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx512(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_ab[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_ab[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx512_prepacked_ab_non_aligned[{i},{j}]: ab={}, ref={}, diff={diff}",
                    c_ab[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx512_prepacked_ab_vs_regular() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx512(&a, m, k);
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx512(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_asm_f32_avx512(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_ab, &c_reg, 1e-5, "avx512_prepacked_ab_vs_regular");
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod pack_tests {
    use super::*;

    /// Scalar reference pack_a (known correct)
    fn pack_a_ref(a: &[f32], lda: usize, mc: usize, kc: usize, mr: usize) -> Vec<f32> {
        let mut packed = vec![0.0f32; mc.div_ceil(mr) * mr * kc];
        let mut p = 0usize;
        let mut i = 0usize;
        while i + mr <= mc {
            for r in 0..mr {
                for k in 0..kc {
                    packed[p + k * mr + r] = a[(i + r) * lda + k];
                }
            }
            p += mr * kc;
            i += mr;
        }
        if i < mc {
            let rem = mc - i;
            for r in 0..rem {
                for k in 0..kc {
                    packed[p + k * mr + r] = a[(i + r) * lda + k];
                }
            }
        }
        packed
    }

    #[test]
    fn test_pack_a_avx2_vs_scalar() {
        let mr = 6usize;
        // Test various mc/kc combinations including non-multiples of 8
        for &(mc, kc) in &[(6, 8), (6, 16), (12, 296), (128, 296), (6, 7), (12, 9), (128, 216)] {
            let lda = kc;
            let a: Vec<f32> = (0..mc * lda).map(|i| (i as f32) * 0.01 - 5.0).collect();
            
            let expected = pack_a_ref(&a, lda, mc, kc, mr);
            
            let mut packed = vec![0.0f32; expected.len()];
            unsafe {
                gllm_pack_a_f32(a.as_ptr(), lda, packed.as_mut_ptr(), mc, kc, mr);
            }
            
            for idx in 0..expected.len() {
                let diff = (packed[idx] - expected[idx]).abs();
                assert!(
                    diff < 1e-6,
                    "pack_a mismatch at idx={} for mc={} kc={}: got={}, expected={}, diff={}",
                    idx, mc, kc, packed[idx], expected[idx], diff,
                );
            }
            eprintln!("pack_a mc={} kc={}: OK", mc, kc);
        }
    }

    #[test]
    fn test_pack_b_correctness() {
        let nr = 16usize;
        for &(kc, nc) in &[(8usize, 16usize), (296, 256), (216, 256), (8, 15), (296, 17)] {
            let ldb = nc;
            let b: Vec<f32> = (0..kc * ldb).map(|i| (i as f32) * 0.01 - 3.0).collect();
            
            // Reference pack_b
            let num_panels = nc.div_ceil(nr);
            let mut expected = vec![0.0f32; num_panels * kc * nr];
            let mut p = 0usize;
            let mut j = 0usize;
            while j + nr <= nc {
                for k in 0..kc {
                    for c in 0..nr {
                        expected[p + k * nr + c] = b[k * ldb + j + c];
                    }
                }
                p += kc * nr;
                j += nr;
            }
            if j < nc {
                let rem = nc - j;
                for k in 0..kc {
                    for c in 0..rem {
                        expected[p + k * nr + c] = b[k * ldb + j + c];
                    }
                }
            }
            
            let mut packed = vec![0.0f32; expected.len()];
            unsafe {
                gllm_pack_b_f32(b.as_ptr(), ldb, packed.as_mut_ptr(), kc, nc, nr);
            }
            
            for idx in 0..expected.len() {
                let diff = (packed[idx] - expected[idx]).abs();
                assert!(
                    diff < 1e-6,
                    "pack_b mismatch at idx={} for kc={} nc={}: got={}, expected={}, diff={}",
                    idx, kc, nc, packed[idx], expected[idx], diff,
                );
            }
            eprintln!("pack_b kc={} nc={}: OK", kc, nc);
        }
    }
}
