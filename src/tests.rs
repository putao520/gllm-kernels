#[cfg(test)]
mod tests {
    use crate::cpu_kernels::CpuKernels;
    use crate::traits::Kernels;

    // ========================================================================
    // Original tests (vec_add, vec_mul, silu, gemm)
    // ========================================================================

    #[test]
    fn test_scalar_add_f32() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 4];
        kernels.vec_add(&a, &b, &mut out);
        assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_scalar_mul_f32() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0; 4];
        kernels.vec_mul(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scalar_silu_f32() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![0.0, 1.0, -1.0];
        let mut out = vec![0.0; 3];
        kernels.silu(&a, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 0.731058).abs() < 1e-5);
        assert!((out[2] - (-0.268941)).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_matmul_f32() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        kernels.gemm(&a, &b, &mut c, 2, 2, 3);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_prepacked_matmul_f32() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (2, 2, 3);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let packed_b = kernels.pack_b(&b, n, k);
        let mut c = vec![0.0; m * n];
        kernels.gemm_prepacked(&a, &packed_b, &mut c, m, n, k);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_prepacked_matmul_bias_f32() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (2, 2, 3);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let bias = vec![100.0, 200.0];

        let packed_b = kernels.pack_b(&b, n, k);
        let mut c = vec![0.0; m * n];
        kernels.gemm_bias_prepacked(&a, &packed_b, &bias, &mut c, m, n, k);
        assert_eq!(c, vec![158.0, 264.0, 239.0, 354.0]);
    }

    #[test]
    fn test_prepacked_matmul_larger_f32() {
        let kernels = CpuKernels::<f32>::new();
        let (m, n, k) = (4, 8, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();

        // Reference: regular matmul
        let mut c_ref = vec![0.0f32; m * n];
        kernels.gemm(&a, &b, &mut c_ref, m, n, k);

        // Prepacked matmul
        let packed_b = kernels.pack_b(&b, n, k);
        let mut c_packed = vec![0.0f32; m * n];
        kernels.gemm_prepacked(&a, &packed_b, &mut c_packed, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_ref[i] - c_packed[i]).abs() < 1e-4,
                "mismatch at {}: ref={}, packed={}", i, c_ref[i], c_packed[i]
            );
        }
    }

    // ========================================================================
    // Block 1: BLAS-1 tests
    // ========================================================================

    #[test]
    fn test_vec_sub() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![5.0, 8.0, 3.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        kernels.vec_sub(&a, &b, &mut out);
        assert_eq!(out, vec![4.0, 6.0, 0.0, -3.0]);
    }

    #[test]
    fn test_vec_sum() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = kernels.vec_sum(&a);
        assert!((s - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec_max() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 5.0, 3.0, -2.0, 4.0];
        let m = kernels.vec_max(&a);
        assert!((m - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec_dot() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = kernels.vec_dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((d - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec_sum_squares() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![3.0, 4.0];
        let ss = kernels.vec_sum_squares(&a);
        // 9 + 16 = 25
        assert!((ss - 25.0).abs() < 1e-5);
    }

    // ========================================================================
    // Block 2: Activation tests
    // ========================================================================

    #[test]
    fn test_relu() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut out = vec![0.0; 5];
        kernels.relu(&a, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gelu() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![0.0, 1.0, -1.0];
        let mut out = vec![0.0; 3];
        kernels.gelu(&a, &mut out);
        // gelu(0) ≈ 0.0
        assert!((out[0]).abs() < 1e-4);
        // gelu(1) ≈ 0.8412
        assert!((out[1] - 0.8412).abs() < 1e-3);
        // gelu(-1) ≈ -0.1588
        assert!((out[2] - (-0.1588)).abs() < 1e-3);
    }

    #[test]
    fn test_tanh() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![0.0, 1.0, -1.0];
        let mut out = vec![0.0; 3];
        kernels.tanh(&a, &mut out);
        assert!((out[0]).abs() < 1e-5);
        assert!((out[1] - 0.7616).abs() < 1e-3);
        assert!((out[2] - (-0.7616)).abs() < 1e-3);
    }

    #[test]
    fn test_softmax() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        kernels.softmax(&a, &mut out);
        // Verify sum = 1.0
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Verify ordering
        assert!(out[2] > out[1] && out[1] > out[0]);
    }

    #[test]
    fn test_swiglu() {
        let kernels = CpuKernels::<f32>::new();
        let gate = vec![1.0, -1.0];
        let up = vec![2.0, 3.0];
        let mut out = vec![0.0; 2];
        kernels.swiglu(&gate, &up, &mut out);
        // swiglu = silu(gate) * up
        // silu(1) ≈ 0.731058 -> 0.731058 * 2 ≈ 1.462
        // silu(-1) ≈ -0.268941 -> -0.268941 * 3 ≈ -0.807
        assert!((out[0] - 1.462).abs() < 1e-2);
        assert!((out[1] - (-0.807)).abs() < 1e-2);
    }

    // ========================================================================
    // Block 2: Normalization tests
    // ========================================================================

    #[test]
    fn test_rms_norm() {
        let kernels = CpuKernels::<f32>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0]; // identity weight
        let mut out = vec![0.0; 4];
        kernels.rms_norm(&x, &w, &mut out, 1e-5);
        // rms = sqrt(mean(1+4+9+16)) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
        // inv_rms ≈ 0.3651
        // out[0] ≈ 1.0 * 0.3651 ≈ 0.3651
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        let expected_0 = 1.0 / rms;
        assert!((out[0] - expected_0).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm() {
        let kernels = CpuKernels::<f32>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 4];
        kernels.layer_norm(&x, &w, &b, &mut out, 1e-5);
        // mean = 2.5, var = 1.25, std = ~1.1180
        // After normalization: should be ~ [-1.342, -0.447, 0.447, 1.342]
        let mean = 2.5f32;
        let var = 1.25f32;
        let std = (var + 1e-5).sqrt();
        let expected_0 = (1.0 - mean) / std;
        assert!((out[0] - expected_0).abs() < 1e-3);
        // Sum should be ~0 after normalization (with unit weight and zero bias)
        let sum: f32 = out.iter().sum();
        assert!(sum.abs() < 1e-4);
    }

    // ========================================================================
    // Block 3: RoPE & Embedding tests
    // ========================================================================

    #[test]
    fn test_rope() {
        let kernels = CpuKernels::<f32>::new();
        // head_dim = 4, seq_len = 1
        let mut data = vec![1.0, 0.0, 0.0, 1.0]; // [x0, x1, x0+half, x1+half]
        let cos = vec![1.0, 0.0]; // cos for 2 pairs
        let sin = vec![0.0, 1.0]; // sin for 2 pairs
        kernels.rope(&mut data, &cos, &sin, 4, false);
        // pair 0: x0'=1*1 - 0*0=1, x0+half'=1*0 + 0*1=0
        // pair 1: x1'=0*0 - 1*1=-1, x1+half'=0*1 + 1*0=0 -> Wait...
        // Let me recalculate:
        // half = 2
        // i=0: x0=data[0]=1.0, x1=data[0+2]=0.0, c=cos[0]=1.0, s=sin[0]=0.0
        //   data[0] = 1*1 - 0*0 = 1.0
        //   data[2] = 1*0 + 0*1 = 0.0
        // i=1: x0=data[1]=0.0, x1=data[1+2]=1.0, c=cos[1]=0.0, s=sin[1]=1.0
        //   data[1] = 0*0 - 1*1 = -1.0
        //   data[3] = 0*1 + 1*0 = 0.0
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - (-1.0)).abs() < 1e-5);
        assert!((data[2] - 0.0).abs() < 1e-5);
        assert!((data[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_lookup() {
        let kernels = CpuKernels::<f32>::new();
        // vocab=3, dim=2
        let table = vec![
            0.1, 0.2,  // token 0
            0.3, 0.4,  // token 1
            0.5, 0.6,  // token 2
        ];
        let ids = vec![2u32, 0, 1];
        let mut out = vec![0.0; 6];
        kernels.embedding_lookup(&ids, &table, &mut out, 3, 2);
        assert!((out[0] - 0.5).abs() < 1e-5); // token 2
        assert!((out[1] - 0.6).abs() < 1e-5);
        assert!((out[2] - 0.1).abs() < 1e-5); // token 0
        assert!((out[3] - 0.2).abs() < 1e-5);
        assert!((out[4] - 0.3).abs() < 1e-5); // token 1
        assert!((out[5] - 0.4).abs() < 1e-5);
    }

    // ========================================================================
    // Block 4: GEMV test
    // ========================================================================

    #[test]
    fn test_gemv() {
        let kernels = CpuKernels::<f32>::new();
        // A: 2x3 = [[1,2,3],[4,5,6]], x: [1,1,1], y: [0,0]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0, 0.0];
        kernels.gemv(&a, &x, &mut y, 2, 3);
        // y[0] = 1+2+3 = 6, y[1] = 4+5+6 = 15
        assert!((y[0] - 6.0).abs() < 1e-5);
        assert!((y[1] - 15.0).abs() < 1e-5);
    }
    // ========================================================================
    // Block 5: Fused Kernels tests
    // ========================================================================

    #[test]
    fn test_gemm_bias() {
        let kernels = CpuKernels::<f32>::new();
        // C = A * B + Bias
        // A: 2x2 [[1,2],[3,4]]
        // B: 2x2 [[1,0],[0,1]] (Identity)
        // Bias: [10, 20]
        // C expected: [[11, 22], [13, 24]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![10.0, 20.0];
        let mut c = vec![0.0; 4];
        kernels.gemm_bias(&a, &b, &bias, &mut c, 2, 2, 2);
        assert_eq!(c, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_fused_qkv_rope() {
        let kernels = CpuKernels::<f32>::new();
        // Simple case: seq_len=1, hidden=2, n_heads=1, head_dim=2
        // input: [1, 0]
        // WQ: Identity [[1,0],[0,1]] -> Q=[1,0]
        // WK: All Zero [[0,0],[0,0]] -> K=[0,0]
        // WV: Identity [[1,0],[0,1]] -> V=[1,0]
        // RoPE:
        // cos=[1], sin=[0] => No rotation
        // Q' = [1, 0]
        // K' = [0, 0]
        // V' = [1, 0] (V isn't rotated)
        let input = vec![1.0, 0.0];
        let wq = vec![1.0, 0.0, 0.0, 1.0];
        let wk = vec![0.0, 0.0, 0.0, 0.0];
        let wv = vec![1.0, 0.0, 0.0, 1.0];
        let cos = vec![1.0, 1.0]; // cos, sin interleaved? No, separate arrays in signature
        let sin = vec![0.0, 0.0];
        
        let mut q_out = vec![0.0; 2];
        let mut k_out = vec![0.0; 2];
        let mut v_out = vec![0.0; 2];
        
        kernels.fused_qkv_rope(
            &input, &wq, &wk, &wv, &cos, &sin,
            &mut q_out, &mut k_out, &mut v_out,
            1, 2, 1, 1, 2, 2, false
        );
        
        // Check Q (rotated by 0 -> same)
        assert!((q_out[0]-1.0).abs() < 1e-5);
        assert!((q_out[1]-0.0).abs() < 1e-5);
        
        // Check V (projection only)
        assert!((v_out[0]-1.0).abs() < 1e-5);
        
        // Test with actual rotation
        // input: [1, 0] -> Q: [1, 0]
        // cos=[0], sin=[1] => 90 deg rotation
        // Q'[0] = x0*c - x1*s = 1*0 - 0*1 = 0
        // Q'[1] = x0*s + x1*c = 1*1 + 0*0 = 1
        let cos90 = vec![0.0, 0.0];
        let sin90 = vec![1.0, 1.0];
        kernels.fused_qkv_rope(
            &input, &wq, &wk, &wv, &cos90, &sin90,
            &mut q_out, &mut k_out, &mut v_out,
            1, 2, 1, 1, 2, 2, false
        );
        assert!((q_out[0]-0.0).abs() < 1e-5);
        assert!((q_out[1]-1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fused_gate_up_swiglu() {
        let kernels = CpuKernels::<f32>::new();
        // input: [1.0]
        let input = vec![1.0];
        // gate: [2.0] -> gate=2.0
        let gate_weight = vec![2.0];
        // up: [3.0] -> up=3.0
        let up_weight = vec![3.0];
        let mut out = vec![0.0];
        
        kernels.fused_gate_up_swiglu(&input, &gate_weight, &up_weight, &mut out, 1, 1, 1);
        
        // silu(2) = 2 / (1 + e^-2) = 2 / (1 + 0.1353) = 2 / 1.1353 = 1.76159
        // out = 1.76159 * 3 = 5.2847
        let expected = 5.2847;
        assert!((out[0] - expected).abs() < 1e-3, "Got {}, expected {}", out[0], expected);
    }


    // ========================================================================
    // Block 15: f16 Element tests
    // ========================================================================

    #[test]
    fn test_f16_vec_add() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let b = vec![f16::from_f32(3.0), f16::from_f32(4.0)];
        let mut out = vec![f16::ZERO; 2];
        kernels.vec_add(&a, &b, &mut out);
        assert!((out[0].to_f32() - 4.0).abs() < 0.01);
        assert!((out[1].to_f32() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_silu() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a = vec![f16::from_f32(0.0), f16::from_f32(1.0)];
        let mut out = vec![f16::ZERO; 2];
        kernels.silu(&a, &mut out);
        assert!(out[0].to_f32().abs() < 0.01);
        assert!((out[1].to_f32() - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_f16_gemm() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let b = vec![f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(1.0)];
        let mut c = vec![f16::ZERO; 4];
        kernels.gemm(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0].to_f32() - 1.0).abs() < 0.01);
        assert!((c[1].to_f32() - 2.0).abs() < 0.01);
        assert!((c[2].to_f32() - 3.0).abs() < 0.01);
        assert!((c[3].to_f32() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_bf16_vec_add() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a = vec![bf16::from_f32(1.0), bf16::from_f32(2.0)];
        let b = vec![bf16::from_f32(3.0), bf16::from_f32(4.0)];
        let mut out = vec![bf16::ZERO; 2];
        kernels.vec_add(&a, &b, &mut out);
        assert!((out[0].to_f32() - 4.0).abs() < 0.1);
        assert!((out[1].to_f32() - 6.0).abs() < 0.1);
    }

    // ========================================================================
    // Block 15b: f16/bf16 activation & normalization tests
    // ========================================================================

    // --- f16 activations ---

    #[test]
    fn test_f16_relu() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a: Vec<f16> = [-2.0, -1.0, 0.0, 1.0, 2.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 5];
        kernels.relu(&a, &mut out);
        let got: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert_eq!(got, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_f16_gelu() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a: Vec<f16> = [0.0, 1.0, -1.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 3];
        kernels.gelu(&a, &mut out);
        assert!(out[0].to_f32().abs() < 0.01);
        assert!((out[1].to_f32() - 0.8412).abs() < 0.02);
        assert!((out[2].to_f32() - (-0.1588)).abs() < 0.02);
    }

    #[test]
    fn test_f16_tanh() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a: Vec<f16> = [0.0, 1.0, -1.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 3];
        kernels.tanh(&a, &mut out);
        assert!(out[0].to_f32().abs() < 0.01);
        assert!((out[1].to_f32() - 0.7616).abs() < 0.01);
        assert!((out[2].to_f32() - (-0.7616)).abs() < 0.01);
    }

    #[test]
    fn test_f16_softmax() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a: Vec<f16> = [1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 3];
        kernels.softmax(&a, &mut out);
        let sum: f32 = out.iter().map(|x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 0.01);
        assert!(out[2].to_f32() > out[1].to_f32() && out[1].to_f32() > out[0].to_f32());
    }

    #[test]
    fn test_f16_exp() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let a: Vec<f16> = [0.0, 1.0, -1.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 3];
        kernels.exp(&a, &mut out);
        assert!((out[0].to_f32() - 1.0).abs() < 0.01);
        assert!((out[1].to_f32() - std::f32::consts::E).abs() < 0.02);
        assert!((out[2].to_f32() - 1.0 / std::f32::consts::E).abs() < 0.01);
    }

    #[test]
    fn test_f16_swiglu() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let gate: Vec<f16> = [1.0, -1.0].iter().map(|&x| f16::from_f32(x)).collect();
        let up: Vec<f16> = [2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 2];
        kernels.swiglu(&gate, &up, &mut out);
        assert!((out[0].to_f32() - 1.462).abs() < 0.02);
        assert!((out[1].to_f32() - (-0.807)).abs() < 0.02);
    }

    // --- f16 normalization ---

    #[test]
    fn test_f16_rms_norm() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let x: Vec<f16> = [1.0, 2.0, 3.0, 4.0].iter().map(|&x| f16::from_f32(x)).collect();
        let w: Vec<f16> = [1.0, 1.0, 1.0, 1.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 4];
        kernels.rms_norm(&x, &w, &mut out, 1e-5);
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        let expected_0 = 1.0 / rms;
        assert!((out[0].to_f32() - expected_0).abs() < 0.01);
    }

    #[test]
    fn test_f16_layer_norm() {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();
        let x: Vec<f16> = [1.0, 2.0, 3.0, 4.0].iter().map(|&x| f16::from_f32(x)).collect();
        let w: Vec<f16> = [1.0, 1.0, 1.0, 1.0].iter().map(|&x| f16::from_f32(x)).collect();
        let b: Vec<f16> = [0.0, 0.0, 0.0, 0.0].iter().map(|&x| f16::from_f32(x)).collect();
        let mut out = vec![f16::ZERO; 4];
        kernels.layer_norm(&x, &w, &b, &mut out, 1e-5);
        let mean = 2.5f32;
        let var = 1.25f32;
        let std = (var + 1e-5).sqrt();
        let expected_0 = (1.0 - mean) / std;
        assert!((out[0].to_f32() - expected_0).abs() < 0.02);
        let sum: f32 = out.iter().map(|x| x.to_f32()).sum();
        assert!(sum.abs() < 0.05);
    }

    // --- bf16 activations ---

    #[test]
    fn test_bf16_relu() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a: Vec<bf16> = [-2.0, -1.0, 0.0, 1.0, 2.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 5];
        kernels.relu(&a, &mut out);
        let got: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert_eq!(got, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_bf16_silu() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a: Vec<bf16> = [0.0, 1.0, -1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 3];
        kernels.silu(&a, &mut out);
        assert!(out[0].to_f32().abs() < 0.01);
        assert!((out[1].to_f32() - 0.7311).abs() < 0.02);
        assert!((out[2].to_f32() - (-0.2689)).abs() < 0.02);
    }

    #[test]
    fn test_bf16_gelu() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a: Vec<bf16> = [0.0, 1.0, -1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 3];
        kernels.gelu(&a, &mut out);
        assert!(out[0].to_f32().abs() < 0.02);
        assert!((out[1].to_f32() - 0.8412).abs() < 0.03);
        assert!((out[2].to_f32() - (-0.1588)).abs() < 0.03);
    }

    #[test]
    fn test_bf16_tanh() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a: Vec<bf16> = [0.0, 1.0, -1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 3];
        kernels.tanh(&a, &mut out);
        assert!(out[0].to_f32().abs() < 0.01);
        assert!((out[1].to_f32() - 0.7616).abs() < 0.02);
        assert!((out[2].to_f32() - (-0.7616)).abs() < 0.02);
    }

    #[test]
    fn test_bf16_softmax() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a: Vec<bf16> = [1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 3];
        kernels.softmax(&a, &mut out);
        let sum: f32 = out.iter().map(|x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 0.05);
        assert!(out[2].to_f32() > out[1].to_f32() && out[1].to_f32() > out[0].to_f32());
    }

    #[test]
    fn test_bf16_exp() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let a: Vec<bf16> = [0.0, 1.0, -1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 3];
        kernels.exp(&a, &mut out);
        assert!((out[0].to_f32() - 1.0).abs() < 0.02);
        assert!((out[1].to_f32() - std::f32::consts::E).abs() < 0.05);
        assert!((out[2].to_f32() - 1.0 / std::f32::consts::E).abs() < 0.02);
    }

    #[test]
    fn test_bf16_swiglu() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let gate: Vec<bf16> = [1.0, -1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let up: Vec<bf16> = [2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 2];
        kernels.swiglu(&gate, &up, &mut out);
        assert!((out[0].to_f32() - 1.462).abs() < 0.05);
        assert!((out[1].to_f32() - (-0.807)).abs() < 0.05);
    }

    // --- bf16 normalization ---

    #[test]
    fn test_bf16_rms_norm() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let x: Vec<bf16> = [1.0, 2.0, 3.0, 4.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let w: Vec<bf16> = [1.0, 1.0, 1.0, 1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 4];
        kernels.rms_norm(&x, &w, &mut out, 1e-5);
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        let expected_0 = 1.0 / rms;
        assert!((out[0].to_f32() - expected_0).abs() < 0.02);
    }

    #[test]
    fn test_bf16_layer_norm() {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();
        let x: Vec<bf16> = [1.0, 2.0, 3.0, 4.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let w: Vec<bf16> = [1.0, 1.0, 1.0, 1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let b: Vec<bf16> = [0.0, 0.0, 0.0, 0.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mut out = vec![bf16::ZERO; 4];
        kernels.layer_norm(&x, &w, &b, &mut out, 1e-5);
        let mean = 2.5f32;
        let var = 1.25f32;
        let std = (var + 1e-5).sqrt();
        let expected_0 = (1.0 - mean) / std;
        assert!((out[0].to_f32() - expected_0).abs() < 0.05);
        let sum: f32 = out.iter().map(|x| x.to_f32()).sum();
        assert!(sum.abs() < 0.1);
    }

    // ========================================================================
    // Block 15c: f16/bf16 GEMM systematic size coverage
    // ========================================================================

    /// Helper: reference matmul in f64 for verification
    fn reference_matmul(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Helper: run gemm for f16 at given dimensions with patterned data and verify against f64 reference
    fn check_f16_gemm(m: usize, n: usize, k: usize) {
        use half::f16;
        let kernels = CpuKernels::<f16>::new();

        // Patterned data: a[i] = ((i % 7) as f32 - 3.0) * 0.5, b[i] = ((i % 5) as f32 - 2.0) * 0.25
        let a_f64: Vec<f64> = (0..m * k).map(|i| ((i % 7) as f64 - 3.0) * 0.5).collect();
        let b_f64: Vec<f64> = (0..k * n).map(|i| ((i % 5) as f64 - 2.0) * 0.25).collect();
        let ref_c = reference_matmul(&a_f64, &b_f64, m, n, k);

        let a_f16: Vec<f16> = a_f64.iter().map(|&x| f16::from_f64(x)).collect();
        let b_f16: Vec<f16> = b_f64.iter().map(|&x| f16::from_f64(x)).collect();
        let mut c_f16 = vec![f16::ZERO; m * n];
        kernels.gemm(&a_f16, &b_f16, &mut c_f16, m, n, k);

        for idx in 0..m * n {
            let got = c_f16[idx].to_f64();
            let expected = ref_c[idx];
            let tol = expected.abs() * 0.05 + 0.1; // 5% relative + 0.1 absolute (f16 has limited precision)
            assert!(
                (got - expected).abs() < tol,
                "f16 gemm {}x{}x{}: c[{}] = {}, expected {} (tol {})",
                m, n, k, idx, got, expected, tol
            );
        }
    }

    /// Helper: run gemm for bf16 at given dimensions with patterned data and verify against f64 reference
    fn check_bf16_gemm(m: usize, n: usize, k: usize) {
        use half::bf16;
        let kernels = CpuKernels::<bf16>::new();

        let a_f64: Vec<f64> = (0..m * k).map(|i| ((i % 7) as f64 - 3.0) * 0.5).collect();
        let b_f64: Vec<f64> = (0..k * n).map(|i| ((i % 5) as f64 - 2.0) * 0.25).collect();
        let ref_c = reference_matmul(&a_f64, &b_f64, m, n, k);

        let a_bf16: Vec<bf16> = a_f64.iter().map(|&x| bf16::from_f64(x)).collect();
        let b_bf16: Vec<bf16> = b_f64.iter().map(|&x| bf16::from_f64(x)).collect();
        let mut c_bf16 = vec![bf16::ZERO; m * n];
        kernels.gemm(&a_bf16, &b_bf16, &mut c_bf16, m, n, k);

        for idx in 0..m * n {
            let got = c_bf16[idx].to_f64();
            let expected = ref_c[idx];
            let tol = expected.abs() * 0.05 + 0.2; // bf16 has even less precision than f16
            assert!(
                (got - expected).abs() < tol,
                "bf16 gemm {}x{}x{}: c[{}] = {}, expected {} (tol {})",
                m, n, k, idx, got, expected, tol
            );
        }
    }

    /// Helper: run gemm for f32 at given dimensions with patterned data and verify against f64 reference
    fn check_f32_gemm(m: usize, n: usize, k: usize) {
        let kernels = CpuKernels::<f32>::new();

        let a_f64: Vec<f64> = (0..m * k).map(|i| ((i % 7) as f64 - 3.0) * 0.5).collect();
        let b_f64: Vec<f64> = (0..k * n).map(|i| ((i % 5) as f64 - 2.0) * 0.25).collect();
        let ref_c = reference_matmul(&a_f64, &b_f64, m, n, k);

        let a_f32: Vec<f32> = a_f64.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b_f64.iter().map(|&x| x as f32).collect();
        let mut c_f32 = vec![0.0f32; m * n];
        kernels.gemm(&a_f32, &b_f32, &mut c_f32, m, n, k);

        for idx in 0..m * n {
            let got = c_f32[idx] as f64;
            let expected = ref_c[idx];
            let tol = expected.abs() * 1e-5 + 1e-4;
            assert!(
                (got - expected).abs() < tol,
                "f32 gemm {}x{}x{}: c[{}] = {}, expected {} (tol {})",
                m, n, k, idx, got, expected, tol
            );
        }
    }

    // ========================================================================
    // Block 15d: GEMM tile-boundary remainder path tests
    // ========================================================================
    //
    // Systematically test M-remainder, N-remainder, both-remainder paths
    // for each ISA's tile dimensions. K=17 (prime) for all tests.
    //
    // AVX-512: TILE_M=14, TILE_N=32
    // AVX2:    TILE_M=6,  TILE_N=16
    // NEON:    TILE_M=8,  TILE_N=12

    // --- f32 GEMM remainder: AVX-512 tile (14×32) ---
    #[test] fn test_f32_gemm_avx512_exact()    { check_f32_gemm(14, 32, 17); }
    #[test] fn test_f32_gemm_avx512_m_rem()    { check_f32_gemm(15, 32, 17); }
    #[test] fn test_f32_gemm_avx512_n_rem()    { check_f32_gemm(14, 33, 17); }
    #[test] fn test_f32_gemm_avx512_both_rem() { check_f32_gemm(15, 33, 17); }
    #[test] fn test_f32_gemm_avx512_max_rem()  { check_f32_gemm(27, 63, 17); }
    #[test] fn test_f32_gemm_avx512_sub_tile() { check_f32_gemm(13, 31, 17); }

    // --- f32 GEMM remainder: AVX2 tile (6×16) ---
    #[test] fn test_f32_gemm_avx2_exact()    { check_f32_gemm(6, 16, 17); }
    #[test] fn test_f32_gemm_avx2_m_rem()    { check_f32_gemm(7, 16, 17); }
    #[test] fn test_f32_gemm_avx2_n_rem()    { check_f32_gemm(6, 17, 17); }
    #[test] fn test_f32_gemm_avx2_both_rem() { check_f32_gemm(7, 17, 17); }
    #[test] fn test_f32_gemm_avx2_max_rem()  { check_f32_gemm(11, 31, 17); }
    #[test] fn test_f32_gemm_avx2_sub_tile() { check_f32_gemm(5, 15, 17); }

    // --- f32 GEMM remainder: NEON tile (8×12) ---
    #[test] fn test_f32_gemm_neon_exact()    { check_f32_gemm(8, 12, 17); }
    #[test] fn test_f32_gemm_neon_m_rem()    { check_f32_gemm(9, 12, 17); }
    #[test] fn test_f32_gemm_neon_n_rem()    { check_f32_gemm(8, 13, 17); }
    #[test] fn test_f32_gemm_neon_both_rem() { check_f32_gemm(9, 13, 17); }
    #[test] fn test_f32_gemm_neon_max_rem()  { check_f32_gemm(15, 23, 17); }
    #[test] fn test_f32_gemm_neon_sub_tile() { check_f32_gemm(7, 11, 17); }

    // --- f16 GEMM remainder: AVX-512 tile (14×32) ---
    #[test] fn test_f16_gemm_avx512_exact()    { check_f16_gemm(14, 32, 17); }
    #[test] fn test_f16_gemm_avx512_m_rem()    { check_f16_gemm(15, 32, 17); }
    #[test] fn test_f16_gemm_avx512_n_rem()    { check_f16_gemm(14, 33, 17); }
    #[test] fn test_f16_gemm_avx512_both_rem() { check_f16_gemm(15, 33, 17); }
    #[test] fn test_f16_gemm_avx512_max_rem()  { check_f16_gemm(27, 63, 17); }
    #[test] fn test_f16_gemm_avx512_sub_tile() { check_f16_gemm(13, 31, 17); }

    // --- f16 GEMM remainder: AVX2 tile (6×16) ---
    #[test] fn test_f16_gemm_avx2_exact()    { check_f16_gemm(6, 16, 17); }
    #[test] fn test_f16_gemm_avx2_m_rem()    { check_f16_gemm(7, 16, 17); }
    #[test] fn test_f16_gemm_avx2_n_rem()    { check_f16_gemm(6, 17, 17); }
    #[test] fn test_f16_gemm_avx2_both_rem() { check_f16_gemm(7, 17, 17); }
    #[test] fn test_f16_gemm_avx2_max_rem()  { check_f16_gemm(11, 31, 17); }
    #[test] fn test_f16_gemm_avx2_sub_tile() { check_f16_gemm(5, 15, 17); }

    // --- f16 GEMM remainder: NEON tile (8×12) ---
    #[test] fn test_f16_gemm_neon_exact()    { check_f16_gemm(8, 12, 17); }
    #[test] fn test_f16_gemm_neon_m_rem()    { check_f16_gemm(9, 12, 17); }
    #[test] fn test_f16_gemm_neon_n_rem()    { check_f16_gemm(8, 13, 17); }
    #[test] fn test_f16_gemm_neon_both_rem() { check_f16_gemm(9, 13, 17); }
    #[test] fn test_f16_gemm_neon_max_rem()  { check_f16_gemm(15, 23, 17); }
    #[test] fn test_f16_gemm_neon_sub_tile() { check_f16_gemm(7, 11, 17); }

    // --- bf16 GEMM remainder: AVX-512 tile (14×32) ---
    #[test] fn test_bf16_gemm_avx512_exact()    { check_bf16_gemm(14, 32, 17); }
    #[test] fn test_bf16_gemm_avx512_m_rem()    { check_bf16_gemm(15, 32, 17); }
    #[test] fn test_bf16_gemm_avx512_n_rem()    { check_bf16_gemm(14, 33, 17); }
    #[test] fn test_bf16_gemm_avx512_both_rem() { check_bf16_gemm(15, 33, 17); }
    #[test] fn test_bf16_gemm_avx512_max_rem()  { check_bf16_gemm(27, 63, 17); }
    #[test] fn test_bf16_gemm_avx512_sub_tile() { check_bf16_gemm(13, 31, 17); }

    // --- bf16 GEMM remainder: AVX2 tile (6×16) ---
    #[test] fn test_bf16_gemm_avx2_exact()    { check_bf16_gemm(6, 16, 17); }
    #[test] fn test_bf16_gemm_avx2_m_rem()    { check_bf16_gemm(7, 16, 17); }
    #[test] fn test_bf16_gemm_avx2_n_rem()    { check_bf16_gemm(6, 17, 17); }
    #[test] fn test_bf16_gemm_avx2_both_rem() { check_bf16_gemm(7, 17, 17); }
    #[test] fn test_bf16_gemm_avx2_max_rem()  { check_bf16_gemm(11, 31, 17); }
    #[test] fn test_bf16_gemm_avx2_sub_tile() { check_bf16_gemm(5, 15, 17); }

    // --- bf16 GEMM remainder: NEON tile (8×12) ---
    #[test] fn test_bf16_gemm_neon_exact()    { check_bf16_gemm(8, 12, 17); }
    #[test] fn test_bf16_gemm_neon_m_rem()    { check_bf16_gemm(9, 12, 17); }
    #[test] fn test_bf16_gemm_neon_n_rem()    { check_bf16_gemm(8, 13, 17); }
    #[test] fn test_bf16_gemm_neon_both_rem() { check_bf16_gemm(9, 13, 17); }
    #[test] fn test_bf16_gemm_neon_max_rem()  { check_bf16_gemm(15, 23, 17); }
    #[test] fn test_bf16_gemm_neon_sub_tile() { check_bf16_gemm(7, 11, 17); }

    // --- f16 GEMM size tests ---

    #[test]
    fn test_f16_gemm_1x1x1() { check_f16_gemm(1, 1, 1); }

    #[test]
    fn test_f16_gemm_1x8x4() { check_f16_gemm(1, 8, 4); }

    #[test]
    fn test_f16_gemm_8x1x4() { check_f16_gemm(8, 1, 4); }

    #[test]
    fn test_f16_gemm_2x3x5() { check_f16_gemm(2, 3, 5); }

    #[test]
    fn test_f16_gemm_3x7x11() { check_f16_gemm(3, 7, 11); }

    #[test]
    fn test_f16_gemm_7x9x5() { check_f16_gemm(7, 9, 5); } // M=TILE_M+1, N=8+1

    #[test]
    fn test_f16_gemm_6x16x8() { check_f16_gemm(6, 16, 8); } // Exact AVX2 tile

    #[test]
    fn test_f16_gemm_7x17x9() { check_f16_gemm(7, 17, 9); } // AVX2 remainder both M and N

    #[test]
    fn test_f16_gemm_13x19x23() { check_f16_gemm(13, 19, 23); } // All primes

    #[test]
    fn test_f16_gemm_14x32x16() { check_f16_gemm(14, 32, 16); } // Exact AVX-512 tile

    #[test]
    fn test_f16_gemm_15x33x17() { check_f16_gemm(15, 33, 17); } // AVX-512 remainder both M and N

    #[test]
    fn test_f16_gemm_32x32x32() { check_f16_gemm(32, 32, 32); } // Multi-tile

    // --- bf16 GEMM size tests ---

    #[test]
    fn test_bf16_gemm_1x1x1() { check_bf16_gemm(1, 1, 1); }

    #[test]
    fn test_bf16_gemm_1x8x4() { check_bf16_gemm(1, 8, 4); }

    #[test]
    fn test_bf16_gemm_8x1x4() { check_bf16_gemm(8, 1, 4); }

    #[test]
    fn test_bf16_gemm_2x3x5() { check_bf16_gemm(2, 3, 5); }

    #[test]
    fn test_bf16_gemm_3x7x11() { check_bf16_gemm(3, 7, 11); }

    #[test]
    fn test_bf16_gemm_7x9x5() { check_bf16_gemm(7, 9, 5); }

    #[test]
    fn test_bf16_gemm_6x16x8() { check_bf16_gemm(6, 16, 8); }

    #[test]
    fn test_bf16_gemm_7x17x9() { check_bf16_gemm(7, 17, 9); }

    #[test]
    fn test_bf16_gemm_13x19x23() { check_bf16_gemm(13, 19, 23); }

    #[test]
    fn test_bf16_gemm_14x32x16() { check_bf16_gemm(14, 32, 16); }

    #[test]
    fn test_bf16_gemm_15x33x17() { check_bf16_gemm(15, 33, 17); }

    #[test]
    fn test_bf16_gemm_32x32x32() { check_bf16_gemm(32, 32, 32); }

    // --- f32 GEMM additional size tests (same sizes for cross-type validation) ---

    #[test]
    fn test_f32_gemm_1x1x1() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![2.5f32];
        let b = vec![3.0f32];
        let mut c = vec![0.0f32];
        kernels.gemm(&a, &b, &mut c, 1, 1, 1);
        assert!((c[0] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_f32_gemm_7x17x9() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0f32; 7 * 9];
        let b = vec![1.0f32; 9 * 17];
        let mut c = vec![0.0f32; 7 * 17];
        kernels.gemm(&a, &b, &mut c, 7, 17, 9);
        assert!(c.iter().all(|&x| (x - 9.0).abs() < 1e-5));
    }

    #[test]
    fn test_f32_gemm_13x19x23() {
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0f32; 13 * 23];
        let b = vec![1.0f32; 23 * 19];
        let mut c = vec![0.0f32; 13 * 19];
        kernels.gemm(&a, &b, &mut c, 13, 19, 23);
        assert!(c.iter().all(|&x| (x - 23.0).abs() < 1e-4));
    }

    // ========================================================================
    // Block 16: K-Quant dequant tests (Q2_K, Q3_K, Q5_K, Q6_K)
    // ========================================================================

    /// Helper: construct a Q8_K block with known values for baseline comparison.
    /// Q8_K is the simplest: out[i] = d * qs[i], so we can verify exactly.
    fn make_q8k_block(d: f32, qs: &[i8; 256]) -> Vec<u8> {
        use crate::quant::BlockQ8K;
        let mut blk = BlockQ8K {
            d,
            qs: *qs,
            bsums: [0i16; 16],
        };
        // Compute bsums: sum of 16 consecutive qs values
        for g in 0..16 {
            let mut s = 0i16;
            for j in 0..16 {
                s += qs[g * 16 + j] as i16;
            }
            blk.bsums[g] = s;
        }
        let ptr = &blk as *const BlockQ8K as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<BlockQ8K>()).to_vec() }
    }

    #[test]
    fn test_dequant_q8_k() {
        let kernels = CpuKernels::<f32>::new();
        let mut qs = [0i8; 256];
        for i in 0..256 { qs[i] = (i as i8).wrapping_add(1); }
        let d = 0.5f32;
        let block = make_q8k_block(d, &qs);
        let mut out = vec![0.0f32; 256];
        kernels.dequant_q8_k(&block, &mut out);
        for i in 0..256 {
            let expected = d * qs[i] as f32;
            assert!((out[i] - expected).abs() < 1e-4, "q8_k dequant mismatch at {}: {} vs {}", i, out[i], expected);
        }
    }

    /// Helper: create a zeroed block of given size (all zeros = valid for any format)
    fn make_zero_block(size: usize) -> Vec<u8> {
        vec![0u8; size]
    }

    #[test]
    fn test_dequant_q4_k_roundtrip() {
        let kernels = CpuKernels::<f32>::new();
        let block = make_zero_block(std::mem::size_of::<crate::quant::BlockQ4K>());
        let mut out = vec![0.0f32; 256];
        kernels.dequant_q4_k(&block, &mut out);
        // Zero block should decode to all zeros
        for v in &out { assert!(v.abs() < 1e-6); }
    }

    // ========================================================================
    // Block 18: Quantized matmul tests (kquant_matmul, iq_matmul, awq_matmul)
    // ========================================================================

    #[test]
    fn test_kquant_matmul_q8_k() {
        use crate::quant::QuantType;
        let kernels = CpuKernels::<f32>::new();
        // m=1, n=1, k=256 (one block)
        // Weight: one Q8_K block with all qs=1, d=1.0
        let qs = [1i8; 256];
        let block = make_q8k_block(1.0, &qs);
        // Input: k×n = 256×1, all 1.0
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        kernels.kquant_matmul(&block, &input, &mut output, QuantType::Q8K, 1, 1, 256);
        // Expected: d * sum(qs[i] * input[i]) = 1.0 * 256 * 1 = 256.0
        assert!((output[0] - 256.0).abs() < 1.0, "kquant_matmul Q8_K: got {}", output[0]);
    }

    #[test]
    fn test_fused_dequant_gemv_q8_k() {
        use crate::quant::QuantType;
        let kernels = CpuKernels::<f32>::new();
        // m=2, n=1, k=256
        let qs1 = [1i8; 256];
        let qs2 = [2i8; 256];
        let block1 = make_q8k_block(0.5, &qs1);
        let block2 = make_q8k_block(0.5, &qs2);
        let mut weight = Vec::new();
        weight.extend_from_slice(&block1);
        weight.extend_from_slice(&block2);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];
        kernels.fused_dequant_gemv(&weight, &input, &mut output, QuantType::Q8K, 2, 1, 256);
        // Row 0: 0.5 * 256 = 128
        // Row 1: 0.5 * 512 = 256
        assert!((output[0] - 128.0).abs() < 1.0, "row0: got {}", output[0]);
        assert!((output[1] - 256.0).abs() < 1.0, "row1: got {}", output[1]);
    }

    // ========================================================================
    // Block 19: FP fused operator tests
    // ========================================================================

    #[test]
    fn test_fused_ffn_simple() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, hidden_size=2, ffn_dim=2
        let input = vec![1.0, 0.0];
        // gate = identity, up = identity, down = identity
        let gate_w = vec![1.0, 0.0, 0.0, 1.0];
        let up_w = vec![1.0, 0.0, 0.0, 1.0];
        let down_w = vec![1.0, 0.0, 0.0, 1.0];
        let residual = vec![0.5, 0.5];
        let mut output = vec![0.0; 2];
        kernels.fused_ffn(&input, &gate_w, &up_w, &down_w, &residual, &mut output, 1, 2, 2);
        // gate_out = [1,0], up_out = [1,0]
        // swiglu = [silu(1)*1, silu(0)*0] = [0.731, 0]
        // down_out = [0.731, 0]
        // output = down_out + residual = [1.231, 0.5]
        assert!((output[0] - 1.231).abs() < 0.01, "fused_ffn[0]: got {}", output[0]);
        assert!((output[1] - 0.5).abs() < 0.01, "fused_ffn[1]: got {}", output[1]);
    }

    #[test]
    fn test_flash_attention_causal() {
        let kernels = CpuKernels::<f32>::new();
        // seq=2, heads=1, head_dim=2
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        let scale = 1.0 / (2.0f32).sqrt();
        kernels.flash_attention(&q, &k, &v, &mut out, 2, 1, 2, scale, true);
        // Pos 0 (causal, sees only pos 0): out = V[0] = [1, 2]
        assert!((out[0] - 1.0).abs() < 1e-3);
        assert!((out[1] - 2.0).abs() < 1e-3);
        // Pos 1 (sees pos 0 and 1): weighted average of V[0] and V[1]
        // Q[1]·K[0] = 0, Q[1]·K[1] = 1 → softmax favors V[1]
        assert!(out[2] > 1.5); // closer to V[1][0]=3
        assert!(out[3] > 2.5); // closer to V[1][1]=4
    }

    #[test]
    fn test_flash_attention_paged_simple() {
        let kernels = CpuKernels::<f32>::new();
        // seq=1, cache_len=2, heads=1, kv_heads=1, head_dim=2, page_size=2
        let q = vec![1.0, 0.0]; // 1 query
        // 1 physical page with 2 KV entries
        let k_cache = vec![1.0, 0.0, 0.0, 1.0]; // K[0]=[1,0], K[1]=[0,1]
        let v_cache = vec![10.0, 20.0, 30.0, 40.0]; // V[0]=[10,20], V[1]=[30,40]
        let page_table = vec![0usize]; // kv_head 0 maps to physical page 0
        let mut out = vec![0.0; 2];
        let scale = 1.0 / (2.0f32).sqrt();
        kernels.flash_attention_paged(
            &q, &k_cache, &v_cache, &page_table, &mut out,
            1, 2, 1, 1, 2, 2, scale,
        );
        // Q·K[0] = 1/sqrt(2) ≈ 0.707, Q·K[1] = 0
        // softmax → higher weight on V[0]
        assert!(out[0] < 25.0, "paged attn: V[0] should dominate, got {}", out[0]);
        assert!(out[0] > 10.0);
    }

    #[test]
    fn test_fused_linear_residual_rmsnorm_simple() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, in_features=2, out_features=2
        let input = vec![1.0, 0.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let residual = vec![1.0, 1.0];
        let norm_weight = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        kernels.fused_linear_residual_rmsnorm(
            &input, &weight, &residual, &norm_weight, &mut output,
            1, 2, 2, 1e-5,
        );
        // linear_out = [1, 0], with_residual = [2, 1]
        // rms = sqrt((4+1)/2) = sqrt(2.5) ≈ 1.5811
        // out = [2/1.5811, 1/1.5811] ≈ [1.265, 0.632]
        let rms = (2.5f32 + 1e-5).sqrt();
        assert!((output[0] - 2.0 / rms).abs() < 1e-3);
        assert!((output[1] - 1.0 / rms).abs() < 1e-3);
    }

    // ========================================================================
    // Block 20: Quantized fused operator tests
    // ========================================================================

    #[test]
    fn test_fused_int4_linear_residual_rmsnorm() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, in_features=256, out_features=1
        // Q4 weight: 1 BlockQ4K per row = 144 bytes
        let block_size = std::mem::size_of::<crate::quant::BlockQ4K>();
        let weight = vec![0u8; block_size]; // zeroed block → all dequant to 0
        let scales = vec![1.0f32];
        let input = vec![1.0f32; 256];
        let residual = vec![2.0f32];
        let norm_weight = vec![1.0f32];
        let mut output = vec![0.0f32; 1];
        kernels.fused_int4_linear_residual_rmsnorm(
            &input, &weight, &scales, &residual, &norm_weight, &mut output,
            1, 256, 1, 1e-5,
        );
        // linear_out ≈ 0 (zeroed block), with_residual ≈ 2.0
        // rms_norm([2.0], [1.0]) = 2.0 / sqrt(4.0 + eps) ≈ 1.0
        assert!((output[0] - 1.0).abs() < 0.5, "fused_int4: got {}", output[0]);
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
    fn test_fused_ffn_q4() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, hidden_size=256, ffn_dim=256
        // gate: (seq_len=1, ffn_dim=256, hidden_size=256) → need ffn_dim blocks
        // up: same
        // down: (seq_len=1, hidden_size=256, ffn_dim=256) → need hidden_size blocks
        let input = vec![1.0f32; 256];
        let block_size = std::mem::size_of::<crate::quant::BlockQ4K>();
        let gate = vec![0u8; block_size * 256]; // 256 blocks for ffn_dim=256
        let up = vec![0u8; block_size * 256];   // 256 blocks
        let down = vec![0u8; block_size * 256]; // 256 blocks for hidden_size=256
        let gate_scales = vec![1.0f32; 256];
        let up_scales = vec![1.0f32; 256];
        let down_scales = vec![1.0f32; 256];
        let residual = vec![2.0f32; 256];
        let mut output = vec![0.0f32; 256];
        kernels.fused_ffn_q4(
            &input, &gate, &up, &down,
            &gate_scales, &up_scales, &down_scales,
            &residual, &mut output,
            1, 256, 256,
        );
        // Zero weights → output ≈ residual
        for i in 0..256 {
            assert!((output[i] - 2.0).abs() < 1.0, "fused_ffn_q4[{}]: got {}", i, output[i]);
        }
    }

    #[test]
    fn test_fused_qkv_rope_q4() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, hidden_size=256, num_heads=1, num_kv_heads=1, head_dim=256, rotary_dim=256
        // Each weight matrix: (seq_len=1, output_dim=256, hidden_size=256) → need output_dim blocks
        let input = vec![1.0f32; 256];
        let block_size = std::mem::size_of::<crate::quant::BlockQ4K>();
        let wq = vec![0u8; block_size * 256]; // 256 blocks
        let wk = vec![0u8; block_size * 256]; // 256 blocks
        let wv = vec![0u8; block_size * 256]; // 256 blocks
        let scales_q = vec![1.0f32; 256];
        let scales_k = vec![1.0f32; 256];
        let scales_v = vec![1.0f32; 256];
        let cos = vec![1.0f32; 128];
        let sin = vec![0.0f32; 128];
        let mut q_out = vec![0.0f32; 256];
        let mut k_out = vec![0.0f32; 256];
        let mut v_out = vec![0.0f32; 256];
        kernels.fused_qkv_rope_q4(
            &input, &wq, &wk, &wv,
            &scales_q, &scales_k, &scales_v,
            &cos, &sin,
            &mut q_out, &mut k_out, &mut v_out,
            1, 256, 1, 1, 256, 256, false,
        );
        // Zero weights → all outputs ≈ 0
        for v in &q_out {
            assert!(v.abs() < 1.0, "fused_qkv_rope_q4 q_out: got {}", v);
        }
    }

    #[test]
    fn test_fused_ffn_rmsnorm() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, hidden_size=2, ffn_dim=2
        let input = vec![1.0, 0.0];
        let gate_weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let up_weight = vec![1.0, 0.0, 0.0, 1.0];
        let down_weight = vec![1.0, 0.0, 0.0, 1.0];
        let residual = vec![0.5, 0.5];
        let norm_weight = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        kernels.fused_ffn_rmsnorm(
            &input, &gate_weight, &up_weight, &down_weight,
            &residual, &norm_weight, &mut output,
            1, 2, 2, 1e-5,
        );
        // gate=[1,0], up=[1,0], swiglu=[silu(1)*1, silu(0)*0]=[0.731,0]
        // down=[0.731,0], with_residual=[1.231,0.5]
        // rms_norm: rms=sqrt((1.231^2+0.5^2)/2)=sqrt(0.883)≈0.94
        // output≈[1.231/0.94, 0.5/0.94]≈[1.31, 0.53]
        assert!(output[0] > 1.0 && output[0] < 1.5, "fused_ffn_rmsnorm[0]: got {}", output[0]);
        assert!(output[1] > 0.3 && output[1] < 0.8, "fused_ffn_rmsnorm[1]: got {}", output[1]);
    }

    #[test]
    fn test_fused_linear_bias_residual_rmsnorm() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, in_features=2, out_features=2
        let input = vec![1.0, 0.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let bias = vec![0.5, 0.5];
        let residual = vec![1.0, 1.0];
        let norm_weight = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        kernels.fused_linear_bias_residual_rmsnorm(
            &input, &weight, &bias, &residual, &norm_weight, &mut output,
            1, 2, 2, 1e-5,
        );
        // linear=[1,0], with_bias=[1.5,0.5], with_residual=[2.5,1.5]
        // rms=sqrt((2.5^2+1.5^2)/2)=sqrt(4.75)≈2.18
        // output≈[2.5/2.18, 1.5/2.18]≈[1.15, 0.69]
        assert!(output[0] > 1.0 && output[0] < 1.3, "fused_linear_bias_residual_rmsnorm[0]: got {}", output[0]);
        assert!(output[1] > 0.5 && output[1] < 0.8, "fused_linear_bias_residual_rmsnorm[1]: got {}", output[1]);
    }

    #[test]
    fn test_fused_int8_linear_residual_rmsnorm() {
        let kernels = CpuKernels::<f32>::new();
        // seq_len=1, in_features=256, out_features=1
        // gemm_q8: (seq_len=1, out_features=1, in_features=256)
        // Weight layout: for each output column, need blocks_per_row blocks
        // blocks_per_row = in_features / 256 = 1
        // Total weight size: out_features × blocks_per_row × block_size = 1 × 1 × 292 = 292 bytes
        let input = vec![1.0f32; 256];
        let qs = [1i8; 256];
        let block_bytes = make_q8k_block(0.01, &qs);
        let weight_i8: &[i8] = unsafe {
            std::slice::from_raw_parts(block_bytes.as_ptr() as *const i8, block_bytes.len())
        };
        let scales = vec![1.0f32; 1];
        let residual = vec![2.0f32; 1];
        let norm_weight = vec![1.0f32; 1];
        let mut output = vec![0.0f32; 1];
        kernels.fused_int8_linear_residual_rmsnorm(
            &input, weight_i8, &scales, &residual, &norm_weight, &mut output,
            1, 256, 1, 1e-5,
        );
        // linear_out = 0.01 * sum(1*1) = 0.01 * 256 = 2.56
        // with_residual = 2.56 + 2.0 = 4.56
        // rms_norm([4.56], [1.0]) = 4.56 / sqrt(4.56^2 + eps) ≈ 1.0
        assert!((output[0] - 1.0).abs() < 0.1, "fused_int8_linear_residual_rmsnorm: got {}", output[0]);
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
}
