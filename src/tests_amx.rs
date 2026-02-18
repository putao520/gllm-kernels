#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests_amx_bf16 {
    use crate::cpu_kernels::avx512::amx_bf16;
    use half::bf16;

    /// Reference scalar bf16 matmul: C[m,n] = A[m,k] * B[k,n], accumulated in f32
    fn ref_matmul_bf16(a: &[bf16], b: &[bf16], m: usize, n: usize, k: usize) -> Vec<bf16> {
        let mut c = vec![bf16::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
                }
                c[i * n + j] = bf16::from_f32(acc);
            }
        }
        c
    }

    /// Reference scalar bf16 matmul with bias: C[m,n] = A[m,k] * B[k,n] + bias[n]
    fn ref_matmul_bias_bf16(a: &[bf16], b: &[bf16], bias: &[bf16],
                            m: usize, n: usize, k: usize) -> Vec<bf16> {
        let mut c = vec![bf16::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
                }
                c[i * n + j] = bf16::from_f32(acc + bias[j].to_f32());
            }
        }
        c
    }

    fn max_abs_diff_bf16(a: &[bf16], b: &[bf16]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0f32, f32::max)
    }

    fn max_rel_diff_bf16(a: &[bf16], b: &[bf16]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| {
                let xf = x.to_f32();
                let yf = y.to_f32();
                let denom = xf.abs().max(yf.abs()).max(1e-6);
                (xf - yf).abs() / denom
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_amx_bf16_availability() {
        let available = amx_bf16::is_available();
        eprintln!("AMX BF16 available: {available}");
        eprintln!("  amx-tile: {}", is_x86_feature_detected!("amx-tile"));
        eprintln!("  amx-bf16: {}", is_x86_feature_detected!("amx-bf16"));
        // This test always passes -- it's informational
    }

    #[test]
    fn test_amx_bf16_matmul_small() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        let (m, n, k) = (4, 8, 16);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32((i % 10) as f32 * 0.1)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32((i % 7) as f32 * 0.2 - 0.5)).collect();
        let expected = ref_matmul_bf16(&a, &b, m, n, k);

        let mut c = vec![bf16::ZERO; m * n];
        amx_bf16::matmul(&a, &b, &mut c, m, n, k);

        let max_diff = max_abs_diff_bf16(&c, &expected);
        assert!(max_diff < 0.5, "small matmul max_abs_diff={max_diff} (expected < 0.5)");
    }

    #[test]
    fn test_amx_bf16_matmul_exact_tile() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        // Exact 16x32 tile, K=32 (matches TILE_M x TILE_N x TILE_K)
        let (m, n, k) = (16, 32, 32);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32((i % 13) as f32 * 0.05)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32((i % 11) as f32 * 0.1 - 0.5)).collect();
        let expected = ref_matmul_bf16(&a, &b, m, n, k);

        let mut c = vec![bf16::ZERO; m * n];
        amx_bf16::matmul(&a, &b, &mut c, m, n, k);

        let max_diff = max_abs_diff_bf16(&c, &expected);
        assert!(max_diff < 1.0, "exact tile matmul max_abs_diff={max_diff} (expected < 1.0)");
    }

    #[test]
    fn test_amx_bf16_matmul_remainder() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        // M not multiple of 16, N not multiple of 32, K not multiple of 32
        let (m, n, k) = (19, 37, 50);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32((i % 17) as f32 * 0.03)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32((i % 19) as f32 * 0.04 - 0.3)).collect();
        let expected = ref_matmul_bf16(&a, &b, m, n, k);

        let mut c = vec![bf16::ZERO; m * n];
        amx_bf16::matmul(&a, &b, &mut c, m, n, k);

        let max_diff = max_abs_diff_bf16(&c, &expected);
        assert!(max_diff < 1.5, "remainder matmul max_abs_diff={max_diff} (expected < 1.5)");
    }

    #[test]
    fn test_amx_bf16_matmul_large() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        // Large enough to trigger multi-chunk K blocking
        let (m, n, k) = (64, 128, 256);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(((i * 7 + 3) % 100) as f32 * 0.01)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32(((i * 11 + 5) % 100) as f32 * 0.01 - 0.5)).collect();
        let expected = ref_matmul_bf16(&a, &b, m, n, k);

        let mut c = vec![bf16::ZERO; m * n];
        amx_bf16::matmul(&a, &b, &mut c, m, n, k);

        let max_rel = max_rel_diff_bf16(&c, &expected);
        assert!(max_rel < 0.05, "large matmul max_rel_diff={max_rel} (expected < 0.05)");
    }

    #[test]
    fn test_amx_bf16_matmul_bias() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        let (m, n, k) = (16, 32, 64);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32((i % 10) as f32 * 0.1)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32((i % 7) as f32 * 0.1 - 0.3)).collect();
        let bias: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 * 0.01)).collect();
        let expected = ref_matmul_bias_bf16(&a, &b, &bias, m, n, k);

        let mut c = vec![bf16::ZERO; m * n];
        amx_bf16::matmul_bias(&a, &b, &bias, &mut c, m, n, k);

        let max_diff = max_abs_diff_bf16(&c, &expected);
        assert!(max_diff < 1.0, "matmul_bias max_abs_diff={max_diff} (expected < 1.0)");
    }

    #[test]
    fn test_amx_bf16_matmul_bias_act_relu() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        let (m, n, k) = (16, 32, 64);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32((i % 10) as f32 * 0.1)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32((i % 7) as f32 * 0.1 - 0.3)).collect();
        let bias: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 * 0.01 - 0.2)).collect();

        let mut c = vec![bf16::ZERO; m * n];
        amx_bf16::matmul_bias_act(&a, &b, &bias, &mut c, m, n, k, crate::Activation::Relu);

        // Verify all outputs are >= 0 (ReLU property)
        for i in 0..m * n {
            assert!(c[i].to_f32() >= 0.0, "ReLU output negative at index {i}: {}", c[i].to_f32());
        }
    }

    #[test]
    fn test_amx_bf16_pack_b_roundtrip() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        // Verify pack_b + matmul_prepacked matches matmul
        let (m, n, k) = (16, 32, 64);
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32((i % 10) as f32 * 0.1)).collect();
        let b: Vec<bf16> = (0..k * n).map(|i| bf16::from_f32((i % 7) as f32 * 0.1 - 0.3)).collect();

        let mut c_direct = vec![bf16::ZERO; m * n];
        amx_bf16::matmul(&a, &b, &mut c_direct, m, n, k);

        let packed_b = amx_bf16::pack_b(&b, n, k);
        let mut c_prepacked = vec![bf16::ZERO; m * n];
        amx_bf16::matmul_prepacked(&a, &packed_b, &mut c_prepacked, m, n, k);

        let max_diff = max_abs_diff_bf16(&c_direct, &c_prepacked);
        assert!(max_diff < 0.01, "prepacked vs direct max_abs_diff={max_diff} (expected < 0.01)");
    }

    #[test]
    fn test_amx_bf16_matmul_zero() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        // Zero dimensions should not panic
        let mut c = vec![bf16::ZERO; 0];
        amx_bf16::matmul(&[], &[], &mut c, 0, 0, 0);
        amx_bf16::matmul(&[], &[], &mut c, 0, 4, 4);
    }

    #[test]
    fn test_amx_bf16_matmul_identity() {
        if !amx_bf16::is_available() {
            eprintln!("skipping: AMX BF16 not available");
            return;
        }
        // A * I = A (for square matrices)
        let n = 32;
        let a: Vec<bf16> = (0..n * n).map(|i| bf16::from_f32((i % 10) as f32 * 0.1)).collect();
        let mut identity = vec![bf16::ZERO; n * n];
        for i in 0..n {
            identity[i * n + i] = bf16::ONE;
        }

        let mut c = vec![bf16::ZERO; n * n];
        amx_bf16::matmul(&a, &identity, &mut c, n, n, n);

        let max_diff = max_abs_diff_bf16(&c, &a);
        assert!(max_diff < 0.1, "identity matmul max_abs_diff={max_diff} (expected < 0.1)");
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests_amx_int8 {
    use crate::cpu_kernels::avx512::amx_int8;

    /// Reference scalar i32 matmul: C[m,n] += A[m,k](u8) * B[k,n](i8)
    fn ref_matmul_i32(a: &[u8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += a[i * k + p] as i32 * b[p * n + j] as i32;
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn test_amx_int8_availability() {
        let available = amx_int8::is_available();
        eprintln!("AMX INT8 available: {available}");
        eprintln!("  amx-tile: {}", is_x86_feature_detected!("amx-tile"));
        eprintln!("  amx-int8: {}", is_x86_feature_detected!("amx-int8"));
    }

    #[test]
    fn test_amx_int8_gemm_small() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        let (m, n, k) = (4, 8, 64);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 128) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 64) as i8 - 32).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = amx_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            amx_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "small AMX INT8 4x8 k=64 mismatch");
    }

    #[test]
    fn test_amx_int8_gemm_exact_tile() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        // Exact 16x32 tile, K=64 (matches TILE_M x TILE_N x TILE_K)
        let (m, n, k) = (16, 32, 64);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 100) as i8 - 50).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = amx_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            amx_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "exact tile AMX INT8 16x32 k=64 mismatch");
    }

    #[test]
    fn test_amx_int8_gemm_remainder() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        // M not multiple of 16, N not multiple of 32, K not multiple of 64
        let (m, n, k) = (19, 37, 100);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 250) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 120) as i8 - 60).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = amx_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            amx_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "remainder AMX INT8 19x37 k=100 mismatch");
    }

    #[test]
    fn test_amx_int8_gemm_large() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        // Large enough to trigger multi-chunk K blocking
        let (m, n, k) = (64, 128, 512);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 255) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 127) as i8 - 63).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = amx_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            amx_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "large AMX INT8 64x128 k=512 mismatch");
    }

    #[test]
    fn test_amx_int8_dequantize() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        let n = 35; // not multiple of 16 to test scalar tail
        let c_i32: Vec<i32> = (0..n).map(|i| i as i32 * 100 - 1700).collect();
        let mut c_f32 = vec![0.0f32; n];
        let scale_a = 0.5f32;
        let scale_b = 0.25f32;

        unsafe {
            amx_int8::dequantize_i32_to_f32(
                &c_i32, &mut c_f32, 1, n,
                scale_a, scale_b, None, crate::Activation::None,
            );
        }

        for i in 0..n {
            let expected = c_i32[i] as f32 * scale_a * scale_b;
            assert!(
                (c_f32[i] - expected).abs() < 1e-5,
                "AMX INT8 dequant mismatch at {i}: got {} expected {expected}", c_f32[i]
            );
        }
    }

    #[test]
    fn test_amx_int8_dequantize_with_bias_relu() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        let (m, n) = (2, 32);
        let c_i32: Vec<i32> = (0..m * n).map(|i| i as i32 * 50 - 1600).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut c_f32 = vec![0.0f32; m * n];
        let scale_a = 0.1f32;
        let scale_b = 0.2f32;

        unsafe {
            amx_int8::dequantize_i32_to_f32(
                &c_i32, &mut c_f32, m, n,
                scale_a, scale_b, Some(&bias), crate::Activation::Relu,
            );
        }

        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let raw = c_i32[idx] as f32 * scale_a * scale_b + bias[j];
                let expected = if raw < 0.0 { 0.0 } else { raw };
                assert!(
                    (c_f32[idx] - expected).abs() < 1e-4,
                    "AMX INT8 dequant+bias+relu mismatch at [{i},{j}]: got {} expected {expected}",
                    c_f32[idx]
                );
            }
        }
    }

    #[test]
    fn test_amx_int8_gemm_single_row() {
        if !amx_int8::is_available() {
            eprintln!("skipping: AMX INT8 not available");
            return;
        }
        // M=1: tests edge_m_kernel path
        let (m, n, k) = (1, 32, 64);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 128) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 64) as i8 - 32).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = amx_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            amx_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "single row AMX INT8 1x32 k=64 mismatch");
    }
}
