#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use crate::cpu_kernels::neon::neon_f32;
    use crate::cpu_kernels::{CpuKernels, get_isa_level, IsaLevel};
    use crate::traits::{Kernels, Backend};
    
    // On aarch64, we assume NEON is always available for this project scope.

    #[test]
    fn test_neon_add_f32() {
        let a = vec![1.0; 8];
        let b = vec![2.0; 8];
        let mut out = vec![0.0; 8];

        neon_f32::add(&a, &b, &mut out);

        assert_eq!(out, vec![3.0; 8]);
    }
}
