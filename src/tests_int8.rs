#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests_int8 {
    use crate::cpu_kernels::avx512::avx512_int8;

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
    fn test_int8_pack_b_and_gemm_small() {
        if !is_x86_feature_detected!("avx512vnni") {
            eprintln!("skipping: avx512vnni not available");
            return;
        }
        let (m, n, k) = (2, 4, 8);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 128) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 64) as i8 - 32).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = avx512_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            avx512_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "small 2x4 k=8 mismatch");
    }

    #[test]
    fn test_int8_gemm_exact_tile() {
        if !is_x86_feature_detected!("avx512vnni") {
            eprintln!("skipping: avx512vnni not available");
            return;
        }
        // Exact 14×32 tile, K=64 (multiple of 4)
        let (m, n, k) = (14, 32, 64);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 100) as i8 - 50).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = avx512_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            avx512_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "exact tile 14x32 k=64 mismatch");
    }

    #[test]
    fn test_int8_gemm_remainder() {
        if !is_x86_feature_detected!("avx512vnni") {
            eprintln!("skipping: avx512vnni not available");
            return;
        }
        // M not multiple of 14, N not multiple of 32, K not multiple of 4
        let (m, n, k) = (17, 37, 13);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 250) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 120) as i8 - 60).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = avx512_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            avx512_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "remainder 17x37 k=13 mismatch");
    }

    #[test]
    fn test_int8_gemm_large() {
        if !is_x86_feature_detected!("avx512vnni") {
            eprintln!("skipping: avx512vnni not available");
            return;
        }
        // Large enough to trigger multi-chunk blocking
        let (m, n, k) = (64, 128, 512);
        let a: Vec<u8> = (0..m * k).map(|i| (i % 255) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 127) as i8 - 63).collect();
        let expected = ref_matmul_i32(&a, &b, m, n, k);

        let packed_b = avx512_int8::pack_b(&b, n, k);
        let mut c = vec![0i32; m * n];
        unsafe {
            avx512_int8::gemm_int8(&a, &packed_b, &b, &mut c, m, n, k);
        }
        assert_eq!(c, expected, "large 64x128 k=512 mismatch");
    }

    #[test]
    fn test_int8_dequantize() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: avx512f not available");
            return;
        }
        let n = 35; // not multiple of 16 to test scalar tail
        let c_i32: Vec<i32> = (0..n).map(|i| i as i32 * 100 - 1700).collect();
        let mut c_f32 = vec![0.0f32; n];
        let scale_a = 0.5f32;
        let scale_b = 0.25f32;

        unsafe {
            avx512_int8::dequantize_i32_to_f32(
                &c_i32, &mut c_f32, 1, n,
                scale_a, scale_b, None, crate::Activation::None,
            );
        }

        for i in 0..n {
            let expected = c_i32[i] as f32 * scale_a * scale_b;
            assert!(
                (c_f32[i] - expected).abs() < 1e-5,
                "dequant mismatch at {}: got {} expected {}", i, c_f32[i], expected
            );
        }
    }
}
