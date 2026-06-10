//! TurboQuant FWHT Epilogue Fusion (SPEC §11.1)
//!
//! Fast Walsh-Hadamard Transform 内联到 GEMM epilogue，实现零额外内存访问的在线旋转。
//! 3 个插入点：Attention Epilogue / FFN Epilogue / KV Write 阶段。

/// FWHT 量化配置
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FwhtQuantBits {
    Q4 = 4,
    Q8 = 8,
}

/// FWHT epilogue 融合规则
pub struct FwhtEpilogueFusion {
    pub bits: FwhtQuantBits,
    pub block_size: usize,
}

impl FwhtEpilogueFusion {
    pub fn new(bits: FwhtQuantBits) -> Self {
        Self {
            bits,
            block_size: 32,
        }
    }

    /// 判断是否可以融合 FWHT（维度必须是 2 的幂）
    pub fn can_fuse(&self, m: usize, n: usize) -> bool {
        n.is_power_of_two() && n >= 32 && m > 0
    }

    /// 计算量化后的字节数
    pub fn quantized_bytes(&self, elements: usize) -> usize {
        let bits = self.bits as usize;
        (elements * bits + 7) / 8
    }

    /// 生成 GPU PTX/HIP/MSL FWHT kernel 代码（内联到 epilogue）
    pub fn emit_gpu_fwht_kernel(&self, out: &mut String, n: usize) {
        out.push_str(&format!("    // FWHT butterfly n={} (SPEC §11.1)\n", n));
        out.push_str("    {\n");
        out.push_str("        int stride = 1;\n");
        out.push_str(&format!("        while (stride < {}) {{\n", n));
        out.push_str("            for (int i = 0; i < n; i += stride * 2) {\n");
        out.push_str("                for (int j = 0; j < stride; j++) {\n");
        out.push_str("                    float a = data[i + j];\n");
        out.push_str("                    float b = data[i + j + stride];\n");
        out.push_str("                    data[i + j] = a + b;\n");
        out.push_str("                    data[i + j + stride] = a - b;\n");
        out.push_str("                }\n");
        out.push_str("            }\n");
        out.push_str("            stride *= 2;\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
    }

    /// 生成 x86_64 scalar FWHT（Scalar + SymExec 参考实现，供 SymExec trace 提取）
    pub fn scalar_fwht_inplace(data: &mut [f32]) {
        let n = data.len();
        debug_assert!(n.is_power_of_two());
        let mut stride = 1;
        while stride < n {
            let mut i = 0;
            while i < n {
                for j in 0..stride {
                    let a = data[i + j];
                    let b = data[i + j + stride];
                    data[i + j] = a + b;
                    data[i + j + stride] = a - b;
                }
                i += stride * 2;
            }
            stride *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_fuse() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        assert!(!fusion.can_fuse(1, 16));
        assert!(fusion.can_fuse(1, 32));
        assert!(fusion.can_fuse(8, 64));
        assert!(!fusion.can_fuse(8, 100));
    }

    #[test]
    fn test_quantized_bytes() {
        let fusion_q4 = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        assert_eq!(fusion_q4.quantized_bytes(32), 16);

        let fusion_q8 = FwhtEpilogueFusion::new(FwhtQuantBits::Q8);
        assert_eq!(fusion_q8.quantized_bytes(32), 32);
    }

    #[test]
    fn test_scalar_fwht_correctness() {
        // FWHT([1,0,0,0]) = [1,1,1,1]
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]);

        // FWHT is involutory: FWHT(FWHT(x)) = n*x
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data2 = original.clone();
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data2);
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data2);
        let n = original.len() as f32;
        for (a, b) in data2.iter().zip(original.iter()) {
            assert!((a - b * n).abs() < 1e-4);
        }
    }

    #[test]
    fn test_emit_gpu_fwht_kernel() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        let mut code = String::new();
        fusion.emit_gpu_fwht_kernel(&mut code, 64);
        assert!(code.contains("FWHT butterfly"));
        assert!(code.contains("stride"));
        assert!(code.contains("a + b"));
        assert!(code.contains("a - b"));
    }

    #[test]
    fn test_fwht_quant_bits_discriminant_values() {
        // Verify discriminant values match expected bit widths
        assert_eq!(FwhtQuantBits::Q4 as usize, 4);
        assert_eq!(FwhtQuantBits::Q8 as usize, 8);
    }

    #[test]
    fn test_fwht_quant_bits_derive_traits() {
        // Debug trait produces meaningful output
        let q4 = FwhtQuantBits::Q4;
        let debug_str = format!("{:?}", q4);
        assert!(debug_str.contains("Q4"));

        // Clone produces equal value
        let cloned = q4.clone();
        assert_eq!(cloned, FwhtQuantBits::Q4);

        // PartialEq works correctly
        assert_eq!(FwhtQuantBits::Q4, FwhtQuantBits::Q4);
        assert_ne!(FwhtQuantBits::Q4, FwhtQuantBits::Q8);
    }

    #[test]
    fn test_new_default_block_size() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q8);
        assert_eq!(fusion.block_size, 32);
        assert_eq!(fusion.bits, FwhtQuantBits::Q8);
    }

    #[test]
    fn test_can_fuse_boundary_cases() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);

        // m=0: no rows, cannot fuse
        assert!(!fusion.can_fuse(0, 32));

        // n=1: power of two but below minimum 32
        assert!(!fusion.can_fuse(1, 1));

        // n=2: power of two but below minimum 32
        assert!(!fusion.can_fuse(1, 2));

        // n=31: not a power of two
        assert!(!fusion.can_fuse(1, 31));

        // n=32: minimum valid power of two
        assert!(fusion.can_fuse(1, 32));

        // n=64: valid
        assert!(fusion.can_fuse(4, 64));

        // n=128: valid
        assert!(fusion.can_fuse(1, 128));
    }

    #[test]
    fn test_quantized_bytes_edge_cases() {
        let q4 = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        let q8 = FwhtEpilogueFusion::new(FwhtQuantBits::Q8);

        // Zero elements
        assert_eq!(q4.quantized_bytes(0), 0);
        assert_eq!(q8.quantized_bytes(0), 0);

        // One element: Q4 rounds up to 1 byte, Q8 = 1 byte
        assert_eq!(q4.quantized_bytes(1), 1);
        assert_eq!(q8.quantized_bytes(1), 1);

        // Odd elements with Q4: 5 elements * 4 bits = 20 bits = 3 bytes (ceil)
        assert_eq!(q4.quantized_bytes(5), 3);

        // Odd elements with Q8: 5 elements * 8 bits = 40 bits = 5 bytes
        assert_eq!(q8.quantized_bytes(5), 5);

        // Large count
        assert_eq!(q4.quantized_bytes(1024), 512);
        assert_eq!(q8.quantized_bytes(1024), 1024);
    }

    #[test]
    fn test_scalar_fwht_all_zeros() {
        let mut data = vec![0.0f32; 8];
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        for v in &data {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_scalar_fwht_constant_vector() {
        // FWHT of constant vector [c, c, c, c] = [4c, 0, 0, 0]
        let mut data = vec![2.0f32; 4];
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        assert!((data[0] - 8.0).abs() < 1e-4);
        assert!(data[1].abs() < 1e-4);
        assert!(data[2].abs() < 1e-4);
        assert!(data[3].abs() < 1e-4);
    }

    #[test]
    fn test_scalar_fwht_larger_power_of_two() {
        // Verify involutory property for n=16
        let original: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.5).collect();
        let mut data = original.clone();
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        let n = original.len() as f32;
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b * n).abs() < 1e-3, "expected {} got {}", b * n, a);
        }
    }

    #[test]
    fn test_emit_gpu_fwht_kernel_contains_n() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q8);
        let mut code = String::new();
        fusion.emit_gpu_fwht_kernel(&mut code, 128);
        assert!(code.contains("n=128"));
        assert!(code.contains("stride < 128"));
    }

    #[test]
    fn test_emit_gpu_fwht_kernel_appends_to_existing() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        let mut code = String::from("// prefix\n");
        fusion.emit_gpu_fwht_kernel(&mut code, 32);
        assert!(code.starts_with("// prefix\n"));
        assert!(code.contains("FWHT butterfly n=32"));
    }

    // ── Additional tests ──

    #[test]
    fn test_fwht_quant_bits_copy_semantics() {
        // FwhtQuantBits derives Copy, verify value semantics
        let a = FwhtQuantBits::Q4;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_can_fuse_large_power_of_two() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q8);
        // 2^20 = 1048576 — very large power of two, still valid
        assert!(fusion.can_fuse(1, 1048576));
        // 2^30 = 1073741824
        assert!(fusion.can_fuse(1, 1073741824));
    }

    #[test]
    fn test_can_fuse_multiple_rows() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        // Multiple rows with valid n
        assert!(fusion.can_fuse(128, 64));
        assert!(fusion.can_fuse(1024, 32));
    }

    #[test]
    fn test_quantized_bytes_rounding_behavior() {
        let q4 = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        // 7 elements * 4 bits = 28 bits -> ceil(28/8) = 4 bytes
        assert_eq!(q4.quantized_bytes(7), 4);
        // 9 elements * 4 bits = 36 bits -> ceil(36/8) = 5 bytes
        assert_eq!(q4.quantized_bytes(9), 5);
        // 16 elements * 4 bits = 64 bits -> 8 bytes (no remainder)
        assert_eq!(q4.quantized_bytes(16), 8);
    }

    #[test]
    fn test_scalar_fwht_orthogonality() {
        // FWHT preserves energy: sum of squares is n times original
        let mut data = vec![3.0f32, -1.0, 4.0, 1.0, -5.0, 9.0, -2.0, 6.0];
        let input_energy: f32 = data.iter().map(|x| x * x).sum();
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        let output_energy: f32 = data.iter().map(|x| x * x).sum();
        // Parseval's theorem: output_energy = n * input_energy
        let n = 8.0f32;
        assert!((output_energy - n * input_energy).abs() < 1e-2,
            "energy not preserved: input={}, output={}, expected={}", input_energy, output_energy, n * input_energy);
    }

    #[test]
    fn test_scalar_fwht_linearity() {
        // FWHT(a*x + b*y) = a*FWHT(x) + b*FWHT(y) for scalars a, b
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![4.0f32, 3.0, 2.0, 1.0];
        let a = 2.0f32;
        let b = 3.0f32;

        // Compute a*x + b*y, then FWHT
        let mut combined: Vec<f32> = x.iter().zip(y.iter()).map(|(&xi, &yi)| a * xi + b * yi).collect();
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut combined);

        // Compute a*FWHT(x) + b*FWHT(y) separately
        let mut fx = x.clone();
        let mut fy = y.clone();
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut fx);
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut fy);
        let expected: Vec<f32> = fx.iter().zip(fy.iter()).map(|(&fxi, &fyi)| a * fxi + b * fyi).collect();

        for (i, (c, e)) in combined.iter().zip(expected.iter()).enumerate() {
            assert!((c - e).abs() < 1e-3, "linearity violated at index {}: got {}, expected {}", i, c, e);
        }
    }

    #[test]
    fn test_new_sets_bits_correctly() {
        let fusion_q4 = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        assert_eq!(fusion_q4.bits, FwhtQuantBits::Q4);
        assert_eq!(fusion_q4.bits as usize, 4);

        let fusion_q8 = FwhtEpilogueFusion::new(FwhtQuantBits::Q8);
        assert_eq!(fusion_q8.bits, FwhtQuantBits::Q8);
        assert_eq!(fusion_q8.bits as usize, 8);
    }

    #[test]
    fn test_scalar_fwht_single_element() {
        // n=1 is power of two; FWHT([c]) = [c]
        let mut data = vec![42.0f32];
        FwhtEpilogueFusion::scalar_fwht_inplace(&mut data);
        assert_eq!(data[0], 42.0);
    }

    #[test]
    fn test_emit_gpu_fwht_kernel_structure_completeness() {
        let fusion = FwhtEpilogueFusion::new(FwhtQuantBits::Q4);
        let mut code = String::new();
        fusion.emit_gpu_fwht_kernel(&mut code, 32);
        // Must contain all the structural elements of a butterfly network
        assert!(code.contains("stride = 1"));
        assert!(code.contains("stride *= 2"));
        assert!(code.contains("data[i + j]"));
        assert!(code.contains("data[i + j + stride]"));
        assert!(code.contains("SPEC §11.1"));
    }
}
