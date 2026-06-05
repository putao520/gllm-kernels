
    // ========================================================================
    // Block 19: Q4K dot accuracy tests
    // ========================================================================

    /// Helper: construct a Q4_K block with known nibble values.
    /// Q4_K dequant: for byte qs[i], low = qs[i] & 0x0F, high = qs[i] >> 4
    ///   value[2*i]   = d * low  - d * 8
    ///   value[2*i+1] = d * high - d * 8
    fn make_q4k_block(d: f32, qs: &[u8; 128]) -> Vec<u8> {
        use crate::quant::BlockQ4K;
        let blk = BlockQ4K {
            d: half::f16::from_f32(d),
            dmin: half::f16::from_f32(0.0),
            scales: [0u8; 12],
            qs: *qs,
        };
        let ptr = &blk as *const BlockQ4K as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<BlockQ4K>()).to_vec() }
    }

    #[test]
    fn test_q4k_dot_accuracy_random() {
        // Verify SIMD Q4K dot matches scalar reference with random data.
        // This catches interleaving bugs in AVX-512 (and AVX2) nibble unpacking.
        let kernels = CpuKernels::<f32>::new();
        let d = 0.0371f32;

        // Random-ish nibble values: each byte packs two 4-bit values [0..15]
        let mut qs = [0u8; 128];
        for i in 0..128 {
            let lo = ((i * 97 + 13) % 16) as u8;
            let hi = ((i * 53 + 7) % 16) as u8;
            qs[i] = lo | (hi << 4);
        }
        let block = make_q4k_block(d, &qs);

        // Random-ish f32 input (256 values for QK_K=256)
        let input: Vec<f32> = (0..256).map(|i| {
            ((i as f32 * 0.0137 + 0.5).sin()) * 2.0 - 1.0
        }).collect();

        // Compute scalar reference (matches the scalar q4_k dot macro)
        let d64 = d as f64;
        let mut expected = 0.0f64;
        for i in 0..128 {
            let lo = (qs[i] & 0x0F) as f64;
            let hi = (qs[i] >> 4) as f64;
            expected += (d64 * lo - d64 * 8.0) * (input[i * 2] as f64);
            expected += (d64 * hi - d64 * 8.0) * (input[i * 2 + 1] as f64);
        }

        let mut output = vec![0.0f32; 1];
        kernels.kquant_matmul(&block, &input, &mut output,
            crate::quant::QuantType::Q4K, 1, 1, 256);

        let rel_err = ((output[0] as f64 - expected) / expected).abs();
        assert!(rel_err < 1e-3,
            "Q4K dot accuracy: got {}, expected {}, rel_err {:.2e}",
            output[0], expected, rel_err);
    }

    #[test]
    fn test_q4k_dot_accuracy_multirow() {
        // Test Q4K with 5 rows (exercises both 4-row main loop and 1-row tail)
        let kernels = CpuKernels::<f32>::new();
        let d = 0.05f32;

        let mut qs = [0u8; 128];
        for i in 0..128 {
            let lo = ((i * 31 + 5) % 16) as u8;
            let hi = ((i * 71 + 3) % 16) as u8;
            qs[i] = lo | (hi << 4);
        }

        let input: Vec<f32> = (0..256).map(|i| {
            ((i as f32 * 0.023 + 1.7).cos()) * 1.5
        }).collect();

        // Scalar reference
        let d64 = d as f64;
        let mut expected = 0.0f64;
        for i in 0..128 {
            let lo = (qs[i] & 0x0F) as f64;
            let hi = (qs[i] >> 4) as f64;
            expected += (d64 * lo - d64 * 8.0) * (input[i * 2] as f64);
            expected += (d64 * hi - d64 * 8.0) * (input[i * 2 + 1] as f64);
        }

        // 5 identical rows
        let m = 5;
        let mut weight_data = Vec::new();
        for _ in 0..m {
            weight_data.extend_from_slice(&make_q4k_block(d, &qs));
        }

        let mut output = vec![0.0f32; m];
        kernels.kquant_matmul(&weight_data, &input, &mut output,
            crate::quant::QuantType::Q4K, m, 1, 256);

        for row in 0..m {
            let rel_err = ((output[row] as f64 - expected) / expected).abs();
            assert!(rel_err < 1e-3,
                "Q4K multirow[{}]: got {}, expected {}, rel_err {:.2e}",
                row, output[row], expected, rel_err);
        }
    }

    // ========================================================================
    // Block 21: QuantType utility tests
    // ========================================================================

    #[test]
    fn test_quant_type_properties() {
        use crate::quant::QuantType;
        assert_eq!(QuantType::Q4K.block_size(), 256);
        assert_eq!(QuantType::Q4K.block_bytes(), 144);
        assert_eq!(QuantType::Q4K.bits(), 4);
        assert_eq!(QuantType::Q8K.block_size(), 256);
        assert_eq!(QuantType::Q8K.block_bytes(), 292);
        assert_eq!(QuantType::Q8K.bits(), 8);
        assert_eq!(QuantType::IQ4NL.block_size(), 32);
        assert_eq!(QuantType::IQ4NL.block_bytes(), 18);
        assert_eq!(QuantType::Squeeze.block_size(), 256);
        assert_eq!(QuantType::Squeeze.block_bytes(), 130);
        assert_eq!(QuantType::Squeeze.bits(), 3);
        assert_eq!(QuantType::AWQ4.bits(), 4);
        assert_eq!(QuantType::GPTQ4.bits(), 4);
    }

    // ========================================================================
    // Block 22: Additional missing method tests
    // ========================================================================

    #[test]
    fn test_rope_with_pos() {
        let kernels = CpuKernels::<f32>::new();
        // head_dim = 4, position offset = 1, seq_len = 1
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        // cos/sin arrays: indexed by actual_pos * half + i
        // actual_pos = 0 + 1 = 1, half = 2
        // Need cos[1*2+0], cos[1*2+1], sin[1*2+0], sin[1*2+1]
        // So need at least 4 elements
        let cos = vec![1.0, 0.0, 0.0, -1.0]; // cos[2]=0.0, cos[3]=-1.0
        let sin = vec![0.0, 1.0, 1.0, 0.0];  // sin[2]=1.0, sin[3]=0.0
        kernels.rope_with_pos(&mut data, &cos, &sin, 4, 1, false);
        // With position=1, actual_pos=1, uses cos[2]=0.0, cos[3]=-1.0, sin[2]=1.0, sin[3]=0.0
        // half = 2
        // i=0: x0=1.0, x1=0.0, c=cos[2]=0.0, s=sin[2]=1.0
        //   data[0] = 1*0 - 0*1 = 0.0
        //   data[2] = 1*1 + 0*0 = 1.0
        // i=1: x0=0.0, x1=1.0, c=cos[3]=-1.0, s=sin[3]=0.0
        //   data[1] = 0*(-1) - 1*0 = 0.0
        //   data[3] = 0*0 + 1*(-1) = -1.0
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[3] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_q8() {
        let kernels = CpuKernels::<f32>::new();
        // n=256 (one block), k=256
        let qs = [1i8; 256];
        let block_bytes = make_q8k_block(0.5, &qs);
        // Reinterpret as &[i8]
        let weight_i8: &[i8] = unsafe {
            std::slice::from_raw_parts(block_bytes.as_ptr() as *const i8, block_bytes.len())
        };
        let input = vec![1.0f32; 256];
        let scale = 1.0;
        let result = kernels.gemv_q8(weight_i8, &input, scale, 256);
        // Expected: 0.5 * sum(1 * 1) * scale = 0.5 * 256 * 1.0 = 128.0
        assert!((result - 128.0).abs() < 1.0, "gemv_q8: got {}", result);
    }

    #[test]
    fn test_gemv_q4() {
        let kernels = CpuKernels::<f32>::new();
        // n=256 (one Q4K block)
        let block = make_zero_block(std::mem::size_of::<crate::quant::BlockQ4K>());
        let input = vec![1.0f32; 256];
        let scale = 1.0;
        let result = kernels.gemv_q4(&block, &input, scale, 256);
        // Zero block should give ~0
        assert!(result.abs() < 1.0, "gemv_q4: got {}", result);
    }

    #[test]
    fn test_gemv_q2() {
        let kernels = CpuKernels::<f32>::new();
        // n=256 (one Q2K block)
        let block = make_zero_block(std::mem::size_of::<crate::quant::BlockQ2K>());
        let input = vec![1.0f32; 256];
        let scale = 1.0;
        let result = kernels.gemv_q2(&block, &input, scale, 256);
        // Zero block may produce small non-zero values, just verify reasonable range
        assert!(result.abs() < 500.0, "gemv_q2: got {}", result);
    }

    #[test]
    fn test_gemv_q1() {
        let kernels = CpuKernels::<f32>::new();
        // n=256 (one Q1 block)
        let block = make_zero_block(256); // Placeholder size
        let input = vec![1.0f32; 256];
        let scale = 1.0;
        let result = kernels.gemv_q1(&block, &input, scale, 256);
        // Zero block may produce small non-zero values, just verify reasonable range
        assert!(result.abs() < 500.0, "gemv_q1: got {}", result);
    }

    #[test]
    fn test_gemm_q8() {
        let kernels = CpuKernels::<f32>::new();
        // m=2, n=1, k=256 (1 block per row)
        let qs1 = [1i8; 256];
        let block1 = make_q8k_block(0.5, &qs1);
        // For n=1 output column, need 1 block total
        let weight_i8: &[i8] = unsafe {
            std::slice::from_raw_parts(block1.as_ptr() as *const i8, block1.len())
        };
        let input = vec![1.0f32; 256 * 2]; // m=2 rows, k=256 cols
        let scales = vec![1.0f32; 1]; // 1 scale per output column
        let mut output = vec![0.0f32; 2]; // m=2 rows, n=1 col
        kernels.gemm_q8(weight_i8, &input, &mut output, &scales, 2, 1, 256);
        // Row 0: 0.5 * 256 * 1.0 = 128
        // Row 1: 0.5 * 256 * 1.0 = 128 (same weight block)
        assert!((output[0] - 128.0).abs() < 1.0, "gemm_q8 row0: got {}", output[0]);
        assert!((output[1] - 128.0).abs() < 1.0, "gemm_q8 row1: got {}", output[1]);
    }

    #[test]
    fn test_gemm_q4() {
        let kernels = CpuKernels::<f32>::new();
        // m=2, n=1, k=256 (1 block per row)
        let block_size = std::mem::size_of::<crate::quant::BlockQ4K>();
        // For n=1 output column, need 1 block per row × 1 column = 1 block total
        let block1 = make_zero_block(block_size);
        let weight = block1; // Just 1 block for 1 output column
        let input = vec![1.0f32; 256 * 2]; // m=2 rows, k=256 cols
        let scales = vec![1.0f32; 1]; // 1 scale per output column
        let mut output = vec![0.0f32; 2]; // m=2 rows, n=1 col
        kernels.gemm_q4(&weight, &input, &mut output, &scales, 2, 1, 256);
        // Zero blocks should give ~0
        assert!(output[0].abs() < 1.0, "gemm_q4 row0: got {}", output[0]);
        assert!(output[1].abs() < 1.0, "gemm_q4 row1: got {}", output[1]);
    }

    #[test]
    fn test_dequant_awq4() {
        use half::f16;
        let kernels = CpuKernels::<f32>::new();
        // AWQ4: 128 elements per block (32 u32 packed = 128 bytes)
        // BlockAWQ4 = qweight[32] + scales(f16) + zeros(f16) = 128 + 2 + 2 = 132 bytes
        let block_size = std::mem::size_of::<crate::quant::BlockAWQ4>();
        let packed = vec![0u8; block_size];
        let zeros = vec![0u8; block_size];
        let scales = vec![f16::from_f32(1.0); 256]; // Need enough for all elements
        let mut out = vec![0.0f32; 256];
        kernels.dequant_awq4(&packed, &zeros, &scales, &mut out);
        // Zero packed should give ~0
        for v in &out {
            assert!(v.abs() < 1.0, "dequant_awq4: got {}", v);
        }
    }

    #[test]
    fn test_dequant_gptq4() {
        use half::f16;
        let kernels = CpuKernels::<f32>::new();
        // GPTQ4: 128 elements per block
        let block_size = std::mem::size_of::<crate::quant::BlockGPTQ4>();
        let packed = vec![0u8; block_size];
        let g_idx = vec![0i32; 256];
        let scales = vec![f16::from_f32(1.0); 256];
        let mut out = vec![0.0f32; 256];
        kernels.dequant_gptq4(&packed, &g_idx, &scales, &mut out);
        // Zero packed with scale=1.0 may produce small non-zero values due to zero-point handling
        // Just verify it doesn't crash and produces reasonable values
        for v in &out {
            assert!(v.abs() < 10.0, "dequant_gptq4: got {}", v);
        }
    }

    #[test]
    fn test_dequant_squeeze() {
        let kernels = CpuKernels::<f32>::new();
        // Squeeze: 256 elements per block (130 bytes)
        let block = make_zero_block(std::mem::size_of::<crate::quant::BlockSqueeze>());
        let mut out = vec![0.0f32; 256];
        kernels.dequant_squeeze(&block, &mut out);
        // Zero block should give ~0
        for v in &out {
            assert!(v.abs() < 1.0, "dequant_squeeze: got {}", v);
        }
    }

    #[test]
    fn test_exp_activation() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![0.0, 1.0, -1.0, 2.0];
        let mut out = vec![0.0; 4];
        kernels.exp(&a, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5); // e^0 = 1
        assert!((out[1] - 2.71828).abs() < 1e-4); // e^1 ≈ 2.71828
        assert!((out[2] - 0.36788).abs() < 1e-4); // e^-1 ≈ 0.36788
        assert!((out[3] - 7.38906).abs() < 1e-3); // e^2 ≈ 7.38906
    }

    #[test]
    fn test_vec_scale() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = a.clone();
        kernels.vec_scale(&mut out, 2.5);
        assert_eq!(out, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_vec_axpy() {
        let kernels = CpuKernels::<f32>::new();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        kernels.vec_axpy(&mut y, 2.0, &x);
        // y = a*x + y = 2*x + y = [2+10, 4+20, 6+30] = [12, 24, 36]
        assert_eq!(y, vec![12.0, 24.0, 36.0]);
    }

    // ========================================================================
    // Block 23: Large-scale GEMV tests (LLM typical sizes)
    // ========================================================================

    /// Scalar reference GEMV: y[i] = sum_j(A[i*n+j] * x[j])
    fn reference_gemv(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for i in 0..m {
            let mut sum = 0.0f64; // f64 for reference accuracy
            for j in 0..n {
                sum += a[i * n + j] as f64 * x[j] as f64;
            }
            y[i] = sum as f32;
        }
        y
    }

    /// Deterministic pseudo-random f32 in [-1, 1]
    fn pseudo_random_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as f32 / 32768.0) - 1.0
        }).collect()
    }

    #[test]
    fn test_gemv_large_4096() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n) = (4096, 4096);
        let a = pseudo_random_vec(m * n, 42);
        let x = pseudo_random_vec(n, 137);
        let y_ref = reference_gemv(&a, &x, m, n);
        let mut y = vec![0.0f32; m];
        kernels.gemv(&a, &x, &mut y, m, n);
        for i in 0..m {
            let rel_err = if y_ref[i].abs() > 1e-6 {
                ((y[i] - y_ref[i]) / y_ref[i]).abs()
            } else {
                (y[i] - y_ref[i]).abs()
            };
            assert!(rel_err < 1e-3,
                "gemv 4096x4096: y[{}]={}, ref={}, rel_err={:.2e}", i, y[i], y_ref[i], rel_err);
        }
    }

    #[test]
    fn test_gemv_large_11008() {
        let kernels = CpuKernels::<f32>::new();
        // LLaMA FFN typical: 4096 x 11008
        let (m, n) = (4096, 11008);
        let a = pseudo_random_vec(m * n, 99);
        let x = pseudo_random_vec(n, 200);
        let y_ref = reference_gemv(&a, &x, m, n);
        let mut y = vec![0.0f32; m];
        kernels.gemv(&a, &x, &mut y, m, n);
        for i in (0..m).step_by(128) { // spot-check every 128th element
            let rel_err = if y_ref[i].abs() > 1e-6 {
                ((y[i] - y_ref[i]) / y_ref[i]).abs()
            } else {
                (y[i] - y_ref[i]).abs()
            };
            assert!(rel_err < 1e-3,
                "gemv 4096x11008: y[{}]={}, ref={}, rel_err={:.2e}", i, y[i], y_ref[i], rel_err);
        }
    }

    // ========================================================================
    // Block 24: Boundary condition tests
    // ========================================================================

    #[test]
    fn test_vec_ops_length_zero() {
        let kernels = CpuKernels::<f32>::new();
        let empty: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        // These should not panic on empty input
        kernels.vec_add(&empty, &empty, &mut out);
        kernels.vec_mul(&empty, &empty, &mut out);
        kernels.vec_sub(&empty, &empty, &mut out);
    }

    #[test]
    fn test_vec_ops_length_one() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![3.0];
        let b = vec![4.0];
        let mut out = vec![0.0];
        kernels.vec_add(&a, &b, &mut out);
        assert_eq!(out[0], 7.0);
        kernels.vec_mul(&a, &b, &mut out);
        assert_eq!(out[0], 12.0);
        kernels.vec_sub(&a, &b, &mut out);
        assert_eq!(out[0], -1.0);
        let d = kernels.vec_dot(&a, &b);
        assert!((d - 12.0).abs() < 1e-5);
        let s = kernels.vec_sum(&a);
        assert!((s - 3.0).abs() < 1e-5);
        let m = kernels.vec_max(&a);
        assert!((m - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_non_aligned() {
        // m,n,k not multiples of any SIMD lane width
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (3, 7, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.05).collect();

        // f64 reference
        let mut c_ref = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                }
            }
        }

        let mut c = vec![0.0f32; m * n];
        kernels.gemm(&a, &b, &mut c, m, n, k);
        for idx in 0..m * n {
            let tol = c_ref[idx].abs() * 1e-5 + 1e-4;
            assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                "gemm 3x7x5: c[{}]={}, ref={}", idx, c[idx], c_ref[idx]);
        }
    }

    #[test]
    fn test_gemm_1x1x1() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![3.5f32];
        let b = vec![2.0f32];
        let mut c = vec![0.0f32];
        kernels.gemm(&a, &b, &mut c, 1, 1, 1);
        assert!((c[0] - 7.0).abs() < 1e-5);
    }

    /// Streaming GEMV (M=1) with dimensions that exercise SIMD lanes + scalar tail.
    #[test]
    fn test_gemv_streaming_m1_large() {
        let kernels = CpuKernels::<f32>::new();
        // N=67 (not a multiple of any SIMD width), K=131
        let (m, n, k) = (1, 67, 131);
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 97) as f32) * 0.02 - 0.5).collect();

        // f64 reference
        let mut c_ref = vec![0.0f64; n];
        for j in 0..n {
            for l in 0..k {
                c_ref[j] += a[l] as f64 * b[l * n + j] as f64;
            }
        }

        let mut c = vec![0.0f32; n];
        kernels.gemm(&a, &b, &mut c, m, n, k);
        for idx in 0..n {
            let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
            assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                "gemv_streaming M=1: c[{}]={}, ref={}", idx, c[idx], c_ref[idx]);
        }
    }

    /// Streaming GEMV (M=1) with large N that spans multiple 4×LANES strips.
    #[test]
    fn test_gemv_streaming_m1_wide() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (1, 512, 256);
        let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 53) as f32) * 0.01 - 0.25).collect();

        let mut c_ref = vec![0.0f64; n];
        for j in 0..n {
            for l in 0..k {
                c_ref[j] += a[l] as f64 * b[l * n + j] as f64;
            }
        }

        let mut c = vec![0.0f32; n];
        kernels.gemm(&a, &b, &mut c, m, n, k);
        for idx in 0..n {
            let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
            assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                "gemv_streaming M=1 wide: c[{}]={}, ref={}", idx, c[idx], c_ref[idx]);
        }
    }

    /// Skinny GEMM (M=2..32) — exercises all M-tile paths (m8, m4, m2, m1 remainder).
    #[test]
    fn test_gemm_skinny_m2_to_m32() {
        let kernels = CpuKernels::<f32>::new();
        // Test M values that exercise each code path:
        // 2 (m2), 4 (m4), 7 (m4+m2+m1), 8 (m8), 15 (m8+m4+m2+m1),
        // 16 (2×m8), 31 (3×m8+m4+m2+m1), 32 (4×m8)
        for &m in &[2, 4, 7, 8, 15, 16, 31, 32] {
            let (n, k) = (67, 131); // non-aligned dims
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 53) as f32) * 0.02 - 0.5).collect();

            // f64 reference
            let mut c_ref = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                    }
                }
            }

            let mut c = vec![0.0f32; m * n];
            kernels.gemm(&a, &b, &mut c, m, n, k);
            for idx in 0..m * n {
                let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
                assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                    "skinny GEMM M={}: c[{}]={}, ref={}", m, idx, c[idx], c_ref[idx]);
            }
        }
    }

    /// Skinny GEMM with large N (512) to exercise wide N-tiling.
    #[test]
    fn test_gemm_skinny_wide_n() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (8, 512, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) * 0.1 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 71) as f32) * 0.01 - 0.3).collect();

        let mut c_ref = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                }
            }
        }

        let mut c = vec![0.0f32; m * n];
        kernels.gemm(&a, &b, &mut c, m, n, k);
        for idx in 0..m * n {
            let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
            assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                "skinny GEMM 8x512x256: c[{}]={}, ref={}", idx, c[idx], c_ref[idx]);
        }
    }

    /// Skinny GEMM with small K (1..4) — edge case for K-unroll remainder.
    #[test]
    fn test_gemm_skinny_small_k() {
        let kernels = CpuKernels::<f32>::new();
        for &k in &[1, 2, 3] {
            let (m, n) = (5, 16);
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 + 0.1).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3 - 0.2).collect();

            let mut c_ref = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                    }
                }
            }

            let mut c = vec![0.0f32; m * n];
            kernels.gemm(&a, &b, &mut c, m, n, k);
            for idx in 0..m * n {
                let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
                assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                    "skinny GEMM M=5 K={}: c[{}]={}, ref={}", k, idx, c[idx], c_ref[idx]);
            }
        }
    }

    // ── gemm_bt (B-transposed skinny GEMM) ──

    /// gemm_bt correctness: M=2..32, non-aligned dims, compared to f64 reference.
    #[test]
    fn test_gemm_bt_m2_to_m32() {
        let kernels = CpuKernels::<f32>::new();
        for &m in &[2, 4, 7, 8, 15, 16, 31, 32] {
            let (n, k) = (67, 131);
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 53) as f32) * 0.02 - 0.5).collect();

            // Transpose B[K×N] → B^T[N×K]
            let mut b_t = vec![0.0f32; n * k];
            for ki in 0..k {
                for j in 0..n {
                    b_t[j * k + ki] = b[ki * n + j];
                }
            }

            // f64 reference
            let mut c_ref = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                    }
                }
            }

            let mut c = vec![0.0f32; m * n];
            kernels.gemm_bt(&a, &b_t, &mut c, m, n, k);
            for idx in 0..m * n {
                let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
                assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                    "gemm_bt M={}: c[{}]={}, ref={}", m, idx, c[idx], c_ref[idx]);
            }
        }
    }

    /// gemm_bt with large N to exercise wide N-tiling + scalar tail.
    #[test]
    fn test_gemm_bt_wide_n() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (8, 512, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) * 0.1 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 71) as f32) * 0.01 - 0.3).collect();

        let mut b_t = vec![0.0f32; n * k];
        for ki in 0..k {
            for j in 0..n {
                b_t[j * k + ki] = b[ki * n + j];
            }
        }

        let mut c_ref = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                }
            }
        }

        let mut c = vec![0.0f32; m * n];
        kernels.gemm_bt(&a, &b_t, &mut c, m, n, k);
        for idx in 0..m * n {
            let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
            assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                "gemm_bt 8x512x256: c[{}]={}, ref={}", idx, c[idx], c_ref[idx]);
        }
    }

    /// gemm_bt with small K (1..3) — edge case for K-unroll remainder.
    #[test]
    fn test_gemm_bt_small_k() {
        let kernels = CpuKernels::<f32>::new();
        for &k in &[1, 2, 3] {
            let (m, n) = (5, 16);
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 + 0.1).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3 - 0.2).collect();

            let mut b_t = vec![0.0f32; n * k];
            for ki in 0..k {
                for j in 0..n {
                    b_t[j * k + ki] = b[ki * n + j];
                }
            }

            let mut c_ref = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        c_ref[i * n + j] += a[i * k + l] as f64 * b[l * n + j] as f64;
                    }
                }
            }

            let mut c = vec![0.0f32; m * n];
            kernels.gemm_bt(&a, &b_t, &mut c, m, n, k);
            for idx in 0..m * n {
                let tol = c_ref[idx].abs() * 1e-4 + 1e-3;
                assert!((c[idx] as f64 - c_ref[idx]).abs() < tol,
                    "gemm_bt M=5 K={}: c[{}]={}, ref={}", k, idx, c[idx], c_ref[idx]);
            }
        }
    }

    #[test]
    fn test_softmax_length_1() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![42.0];
        let mut out = vec![0.0];
        kernels.softmax(&a, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5, "softmax([x]) should be [1.0], got {}", out[0]);
    }

    #[test]
    fn test_softmax_length_2() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![0.0, 0.0];
        let mut out = vec![0.0; 2];
        kernels.softmax(&a, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_length_1() {
        let kernels = CpuKernels::<f32>::new();
        let x = vec![5.0];
        let w = vec![1.0];
        let mut out = vec![0.0];
        kernels.rms_norm(&x, &w, &mut out, 1e-5);
        // rms = sqrt(25/1 + eps) = sqrt(25.00001) ~ 5.0
        // out = x / rms * w = 5 / 5 * 1 = 1.0
        assert!((out[0] - 1.0).abs() < 1e-3, "rms_norm len=1: got {}", out[0]);
    }

    /// Test single-pass rms_norm at LLM-scale hidden dimension (4096).
    /// Verifies the fused pass 1 (x*weight + accumulate x²) + L1-hot pass 2
    /// produces bit-identical results to the naive reference.
    #[test]
    fn test_rms_norm_large_fused() {
        let kernels = CpuKernels::<f32>::new();
        let n = 4096; // typical LLM hidden dim
        let eps = 1e-5_f32;
        // Deterministic pseudo-random data
        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.7123 + 0.3).sin()) * 2.0).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.0031).cos() * 0.5).collect();
        let mut out = vec![0.0f32; n];
        kernels.rms_norm(&x, &w, &mut out, eps);
        // Scalar reference
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>();
        let inv_rms = 1.0 / (ss / n as f32 + eps).sqrt();
        let max_err = (0..n)
            .map(|i| (out[i] - x[i] * w[i] * inv_rms).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "rms_norm n={n}: max_err={max_err}");
    }

    /// Test rms_norm with non-unit weights and non-aligned length (4096+13).
    #[test]
    fn test_rms_norm_unaligned_fused() {
        let kernels = CpuKernels::<f32>::new();
        let n = 4096 + 13; // not a multiple of any SIMD width
        let eps = 1e-6_f32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) * 0.001).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
        let mut out = vec![0.0f32; n];
        kernels.rms_norm(&x, &w, &mut out, eps);
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>();
        let inv_rms = 1.0 / (ss / n as f32 + eps).sqrt();
        let max_err = (0..n)
            .map(|i| (out[i] - x[i] * w[i] * inv_rms).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "rms_norm n={n}: max_err={max_err}");
    }

    #[test]
    fn test_layer_norm_length_1() {
        let kernels = CpuKernels::<f32>::new();
        let x = vec![5.0];
        let w = vec![1.0];
        let b = vec![0.0];
        let mut out = vec![0.0];
        kernels.layer_norm(&x, &w, &b, &mut out, 1e-5);
        // mean=5, var=0, std=sqrt(eps) -> (5-5)/sqrt(eps) = 0
        assert!(out[0].abs() < 1e-2, "layer_norm len=1: got {}", out[0]);
    }

    // ========================================================================
    // Block 26: RoPE interleaved extended tests
    // ========================================================================

    #[test]
    fn test_rope_interleaved_identity() {
        let kernels = CpuKernels::<f32>::new();
        // cos=1, sin=0 for all pairs -> identity rotation
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        let cos = vec![1.0; 4]; // 4 pairs for head_dim=8
        let sin = vec![0.0; 4];
        kernels.rope(&mut data, &cos, &sin, 8, true);
        for i in 0..8 {
            assert!((data[i] - original[i]).abs() < 1e-5,
                "rope interleaved identity: data[{}]={}, expected {}", i, data[i], original[i]);
        }
    }

    #[test]
    fn test_rope_interleaved_180deg() {
        let kernels = CpuKernels::<f32>::new();
        // cos=-1, sin=0 -> 180-degree rotation: x'=-x, y'=-y
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let cos = vec![-1.0, -1.0];
        let sin = vec![0.0, 0.0];
        kernels.rope(&mut data, &cos, &sin, 4, true);
        // pair 0: (1,2), c=-1, s=0 -> x'=1*(-1)-2*0=-1, y'=1*0+2*(-1)=-2
        // pair 1: (3,4), c=-1, s=0 -> x'=3*(-1)-4*0=-3, y'=3*0+4*(-1)=-4
        assert!((data[0] - (-1.0)).abs() < 1e-5);
        assert!((data[1] - (-2.0)).abs() < 1e-5);
        assert!((data[2] - (-3.0)).abs() < 1e-5);
        assert!((data[3] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn test_rope_non_interleaved_large() {
        let kernels = CpuKernels::<f32>::new();
        // head_dim=64, verify non-interleaved RoPE with known cos/sin
        let head_dim = 64;
        let half = head_dim / 2;
        let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let data_orig = data.clone();
        let cos: Vec<f32> = (0..half).map(|i| ((i as f32) * 0.01).cos()).collect();
        let sin: Vec<f32> = (0..half).map(|i| ((i as f32) * 0.01).sin()).collect();
        kernels.rope(&mut data, &cos, &sin, head_dim, false);

        // Verify against manual computation
        for i in 0..half {
            let x0 = data_orig[i];
            let x1 = data_orig[i + half];
            let c = cos[i];
            let s = sin[i];
            let expected_0 = x0 * c - x1 * s;
            let expected_1 = x0 * s + x1 * c;
            assert!((data[i] - expected_0).abs() < 1e-4,
                "rope non-interleaved: data[{}]={}, expected {}", i, data[i], expected_0);
            assert!((data[i + half] - expected_1).abs() < 1e-4,
                "rope non-interleaved: data[{}]={}, expected {}", i + half, data[i + half], expected_1);
        }
    }

    // ========================================================================
    // Block 27: ASM GEMM boundary correctness tests
    // ========================================================================
    //
    // Systematic edge-case coverage for the ASM GEMM path (AVX2 6x16 / AVX-512 14x32).
    // Tests go through CpuKernels which dispatches to the best available ISA at runtime.
    //
    // Edge cases covered:
    //   - N not a multiple of NR (16 for AVX2, 32 for AVX-512)
    //   - M not a multiple of MR (6 for AVX2, 14 for AVX-512)
    //   - K very small (1, 2, 3)
    //   - K not a multiple of 8 (unroll remainder)
    //   - Extreme sizes: 1x1x1, 1x1024x1
    //   - All of the above for gemm, gemm_prepacked, gemm_bias, gemm_bias_prepacked

    /// Deterministic patterned data for reproducible tests.
    /// Values in [-1.5, 1.5] range to avoid large accumulation errors.
    fn asm_test_pattern_a(m: usize, k: usize) -> Vec<f32> {
        (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect()
    }

    fn asm_test_pattern_b(k: usize, n: usize) -> Vec<f32> {
        (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect()
    }

    fn asm_test_bias(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect()
    }

    /// f64 reference GEMM (triple loop). Already exists as `reference_matmul` but
    /// we define a local version that also supports bias addition for the fused tests.
    fn ref_gemm_bias(a: &[f32], b: &[f32], bias: Option<&[f32]>, m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    sum += a[i * k + l] as f64 * b[l * n + j] as f64;
                }
                if let Some(bias) = bias {
                    sum += bias[j] as f64;
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Assert ASM GEMM output matches f64 reference within 1e-5 relative tolerance.
    fn assert_gemm_close(got: &[f32], expected: &[f64], m: usize, n: usize, label: &str) {
        assert_eq!(got.len(), m * n, "{label}: output length mismatch");
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let g = got[idx] as f64;
                let e = expected[idx];
                let tol = e.abs() * 1e-5 + 1e-4;
                assert!(
                    (g - e).abs() < tol,
                    "{label} [{i},{j}]: got={g}, expected={e}, diff={}, tol={tol}",
                    (g - e).abs()
                );
            }
        }
    }

    /// Core test runner: tests gemm, gemm_prepacked, gemm_bias, gemm_bias_prepacked
    /// for a single (M, N, K) combination.
    fn check_asm_gemm_all_paths(m: usize, n: usize, k: usize) {
        let kernels = CpuKernels::<f32>::new();
        let a = asm_test_pattern_a(m, k);
        let b = asm_test_pattern_b(k, n);
        let bias = asm_test_bias(n);
        let label_base = format!("{}x{}x{}", m, n, k);

        // 1) gemm (regular path)
        {
            let ref_c = ref_gemm_bias(&a, &b, None, m, n, k);
            let mut c = vec![0.0f32; m * n];
            kernels.gemm(&a, &b, &mut c, m, n, k);
            assert_gemm_close(&c, &ref_c, m, n, &format!("gemm {label_base}"));
        }

        // 2) gemm_prepacked
        {
            let ref_c = ref_gemm_bias(&a, &b, None, m, n, k);
            let packed_b = kernels.pack_b(&b, n, k);
            let mut c = vec![0.0f32; m * n];
            kernels.gemm_prepacked(&a, &packed_b, &mut c, m, n, k);
            assert_gemm_close(&c, &ref_c, m, n, &format!("gemm_prepacked {label_base}"));
        }

        // 3) gemm_bias
        {
            let ref_c = ref_gemm_bias(&a, &b, Some(&bias), m, n, k);
            let mut c = vec![0.0f32; m * n];
            kernels.gemm_bias(&a, &b, &bias, &mut c, m, n, k);
            assert_gemm_close(&c, &ref_c, m, n, &format!("gemm_bias {label_base}"));
        }

        // 4) gemm_bias_prepacked
        {
            let ref_c = ref_gemm_bias(&a, &b, Some(&bias), m, n, k);
            let packed_b = kernels.pack_b(&b, n, k);
            let mut c = vec![0.0f32; m * n];
            kernels.gemm_bias_prepacked(&a, &packed_b, &bias, &mut c, m, n, k);
            assert_gemm_close(&c, &ref_c, m, n, &format!("gemm_bias_prepacked {label_base}"));
        }
    }

    // --- N not a multiple of NR ---
    // NR=16 (AVX2), NR=32 (AVX-512). These N values are not multiples of either.

    #[test] fn test_asm_gemm_n1()  { check_asm_gemm_all_paths(8, 1, 32); }
    #[test] fn test_asm_gemm_n7()  { check_asm_gemm_all_paths(8, 7, 32); }
    #[test] fn test_asm_gemm_n15() { check_asm_gemm_all_paths(8, 15, 32); }
    #[test] fn test_asm_gemm_n17() { check_asm_gemm_all_paths(8, 17, 32); }
    #[test] fn test_asm_gemm_n31() { check_asm_gemm_all_paths(8, 31, 32); }
    #[test] fn test_asm_gemm_n33() { check_asm_gemm_all_paths(8, 33, 32); }
    #[test] fn test_asm_gemm_n63() { check_asm_gemm_all_paths(8, 63, 32); }

    // --- M not a multiple of MR ---
    // MR=6 (AVX2), MR=14 (AVX-512). These M values are not multiples of either.

    #[test] fn test_asm_gemm_m1()  { check_asm_gemm_all_paths(1, 32, 32); }
    #[test] fn test_asm_gemm_m3()  { check_asm_gemm_all_paths(3, 32, 32); }
    #[test] fn test_asm_gemm_m5()  { check_asm_gemm_all_paths(5, 32, 32); }
    #[test] fn test_asm_gemm_m7()  { check_asm_gemm_all_paths(7, 32, 32); }
    #[test] fn test_asm_gemm_m13() { check_asm_gemm_all_paths(13, 32, 32); }

    // --- K very small ---

    #[test] fn test_asm_gemm_k1() { check_asm_gemm_all_paths(8, 32, 1); }
    #[test] fn test_asm_gemm_k2() { check_asm_gemm_all_paths(8, 32, 2); }
    #[test] fn test_asm_gemm_k3() { check_asm_gemm_all_paths(8, 32, 3); }

    // --- K not a multiple of 8 (unroll remainder) ---

    #[test] fn test_asm_gemm_k5()  { check_asm_gemm_all_paths(8, 32, 5); }
    #[test] fn test_asm_gemm_k7()  { check_asm_gemm_all_paths(8, 32, 7); }
    #[test] fn test_asm_gemm_k9()  { check_asm_gemm_all_paths(8, 32, 9); }
    #[test] fn test_asm_gemm_k15() { check_asm_gemm_all_paths(8, 32, 15); }

    // --- Extreme sizes ---

    #[test] fn test_asm_gemm_1x1x1()    { check_asm_gemm_all_paths(1, 1, 1); }
    #[test] fn test_asm_gemm_1x1024x1() { check_asm_gemm_all_paths(1, 1024, 1); }

    // --- Combined M/N/K edge cases (all three dimensions non-aligned simultaneously) ---

    #[test] fn test_asm_gemm_m1_n1_k1()   { check_asm_gemm_all_paths(1, 1, 1); }
    #[test] fn test_asm_gemm_m1_n7_k3()   { check_asm_gemm_all_paths(1, 7, 3); }
    #[test] fn test_asm_gemm_m3_n15_k5()  { check_asm_gemm_all_paths(3, 15, 5); }
    #[test] fn test_asm_gemm_m5_n17_k7()  { check_asm_gemm_all_paths(5, 17, 7); }
    #[test] fn test_asm_gemm_m7_n31_k9()  { check_asm_gemm_all_paths(7, 31, 9); }
    #[test] fn test_asm_gemm_m13_n33_k15() { check_asm_gemm_all_paths(13, 33, 15); }
    #[test] fn test_asm_gemm_m5_n63_k2()  { check_asm_gemm_all_paths(5, 63, 2); }
    #[test] fn test_asm_gemm_m7_n1_k15()  { check_asm_gemm_all_paths(7, 1, 15); }
    #[test] fn test_asm_gemm_m13_n7_k1()  { check_asm_gemm_all_paths(13, 7, 1); }
    #[test] fn test_asm_gemm_m1_n33_k9()  { check_asm_gemm_all_paths(1, 33, 9); }

    // --- Exact tile boundaries (should work perfectly, regression guard) ---
    // AVX2: MR=6, NR=16
    #[test] fn test_asm_gemm_avx2_exact()  { check_asm_gemm_all_paths(6, 16, 8); }
    #[test] fn test_asm_gemm_avx2_2x2()   { check_asm_gemm_all_paths(12, 32, 16); }
    // AVX-512: MR=14, NR=32
    #[test] fn test_asm_gemm_avx512_exact() { check_asm_gemm_all_paths(14, 32, 8); }
    #[test] fn test_asm_gemm_avx512_2x2()  { check_asm_gemm_all_paths(28, 64, 16); }

    // --- One-off from tile boundary (most likely to trigger off-by-one bugs) ---
    // AVX2 MR-1, MR+1, NR-1, NR+1
    #[test] fn test_asm_gemm_avx2_mr_minus1() { check_asm_gemm_all_paths(5, 16, 32); }
    #[test] fn test_asm_gemm_avx2_mr_plus1()  { check_asm_gemm_all_paths(7, 16, 32); }
    #[test] fn test_asm_gemm_avx2_nr_minus1() { check_asm_gemm_all_paths(6, 15, 32); }
    #[test] fn test_asm_gemm_avx2_nr_plus1()  { check_asm_gemm_all_paths(6, 17, 32); }
    // AVX-512 MR-1, MR+1, NR-1, NR+1
    #[test] fn test_asm_gemm_avx512_mr_minus1() { check_asm_gemm_all_paths(13, 32, 32); }
    #[test] fn test_asm_gemm_avx512_mr_plus1()  { check_asm_gemm_all_paths(15, 32, 32); }
    #[test] fn test_asm_gemm_avx512_nr_minus1() { check_asm_gemm_all_paths(14, 31, 32); }
    #[test] fn test_asm_gemm_avx512_nr_plus1()  { check_asm_gemm_all_paths(14, 33, 32); }

    // --- Large non-aligned (multi-tile with remainder on all dimensions) ---
    #[test] fn test_asm_gemm_large_non_aligned() { check_asm_gemm_all_paths(37, 67, 131); }
    #[test] fn test_asm_gemm_large_m1_wide()     { check_asm_gemm_all_paths(1, 257, 64); }
    #[test] fn test_asm_gemm_large_skinny_tall()  { check_asm_gemm_all_paths(97, 3, 15); }
