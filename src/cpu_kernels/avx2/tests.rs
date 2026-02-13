#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use crate::cpu_kernels::avx2::avx2_f32;
    use crate::cpu_kernels::{CpuKernels, get_isa_level, IsaLevel};
    use crate::traits::Kernels;

    #[test]
    fn test_avx2_add_f32() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test: AVX2 not supported");
            return;
        }

        let a = vec![1.0; 8];
        let b = vec![2.0; 8];
        let mut out = vec![0.0; 8];

        avx2_f32::add(&a, &b, &mut out);

        assert_eq!(out, vec![3.0; 8]);
    }

    #[test]
    fn test_avx2_dispatch() {
        if get_isa_level() != IsaLevel::Avx2 && get_isa_level() != IsaLevel::Avx512 {
             println!("Skipping AVX2 dispatch test: Current level {:?} < Avx2", get_isa_level());
             return;
        }
        
        // If we are here, get_isa_level() >= Avx2.
        // CpuKernels should dispatch to AVX2 impl (or better).
        let kernels = CpuKernels::<f32>::new();
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let mut out = vec![0.0; 16];

        kernels.vec_add(&a, &b, &mut out);
        assert_eq!(out, vec![3.0; 16]);
    }

    #[test]
    fn test_avx2_matmul_aligned() {
        if !is_x86_feature_detected!("avx2") {
             println!("Skipping AVX2 matmul test");
             return;
        }
        
        // M=6, N=16, K=16 (Matches one tile exactly)
        let m = 6;
        let n = 16;
        let k = 16;
        
        // A filled with 1.0
        let a = vec![1.0; m * k];
        // B filled with 1.0
        let b = vec![1.0; k * n];
        let mut c = vec![0.0; m * n];
        
        avx2_f32::matmul(&a, &b, &mut c, m, n, k);
        
        // Expected: Each element of C is dot product of row A (1s) and col B (1s) length K.
        // So C[i] = K * 1.0 = 16.0
        assert!(c.iter().all(|&x| (x - 16.0f32).abs() < 1e-5), "Expected all 16.0, got {:?}", &c[0..8]);
    }
    
    #[test]
    fn test_avx2_matmul_unaligned() {
        if !is_x86_feature_detected!("avx2") {
             return;
        }
        
        // M=7, N=17, K=17 (Triggers edge handling)
        let m = 7;
        let n = 17;
        let k = 17;
        
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let mut c = vec![0.0; m * n];
        
        avx2_f32::matmul(&a, &b, &mut c, m, n, k);
        
        assert!(c.iter().all(|&x| (x - 17.0f32).abs() < 1e-5));
    }
}
