#[cfg(test)]
mod tests {
    use crate::quant::{BlockQ4K, BlockQ8K, BlockQ2K, BlockQ3K, BlockQ5K, BlockQ6K};
    use crate::Kernels;
    use crate::cpu_kernels::CpuKernels;
    use half::f16;
    use std::slice;
    use std::mem::size_of;

    fn as_u8_slice<T>(v: &T) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                v as *const T as *const u8,
                size_of::<T>()
            )
        }
    }

    #[test]
    fn test_dequant_q4_k_scalar() {
        let kernels = CpuKernels::<f32>::new();
        let mut block = BlockQ4K {
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };
        block.qs[0] = 0x88;
        block.qs[1] = 0x97;

        let mut out = vec![0.0f32; 256];
        kernels.dequant_q4_k(as_u8_slice(&block), &mut out);
        
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], -1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn test_dequant_q8_k() {
        let kernels = CpuKernels::<f32>::new();
        let mut block = BlockQ8K {
            d: 2.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        };
        block.qs[0] = 10;
        block.qs[1] = -5;
        
        let mut out = vec![0.0f32; 256];
        kernels.dequant_q8_k(as_u8_slice(&block), &mut out);
        
        assert_eq!(out[0], 20.0);
        assert_eq!(out[1], -10.0);
    }

    #[test]
    fn test_dequant_q2_k() {
        let kernels = CpuKernels::<f32>::new();
        let mut block = BlockQ2K {
            scales: [0u8; 16],
            qs: [0u8; 64],
            d: f16::from_f32(2.0),
            dmin: f16::from_f32(0.0),
        };
        block.qs[0] = 0x1B;

        let mut out = vec![0.0f32; 256];
        kernels.dequant_q2_k(as_u8_slice(&block), &mut out);

        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], 4.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn test_dequant_q3_k() {
        let kernels = CpuKernels::<f32>::new();
        let mut block = BlockQ3K {
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: [0u8; 12],
            d: 1.0,
        };
        block.qs[0] = 0x03; 
        block.hmask[0] = 0x01;

        let mut out = vec![0.0f32; 256];
        kernels.dequant_q3_k(as_u8_slice(&block), &mut out);
        
        assert_eq!(out[0], 3.0);
    }

    #[test]
    fn test_dequant_q5_k() {
        let kernels = CpuKernels::<f32>::new();
        let mut block = BlockQ5K {
            scales: [0u8; 12],
            qh: [0u8; 32],
            qs: [0u8; 128],
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
        };
        block.qs[0] = 0x0F;
        block.qh[0] = 0x01;
        block.qs[0] |= 0xF0;
        block.qh[0] |= 0x02;

        let mut out = vec![0.0f32; 256];
        kernels.dequant_q5_k(as_u8_slice(&block), &mut out);

        assert_eq!(out[0], 31.0);
        assert_eq!(out[1], 31.0);
    }

    #[test]
    fn test_dequant_q6_k() {
        let kernels = CpuKernels::<f32>::new();
        let mut block = BlockQ6K {
            qs: [0u8; 128],
            qh: [0u8; 64],
            scales: [0u8; 16],
            d: 1.0,
        };
        block.qs[0] = 0x0F;
        block.qh[0] = 0x03;

        let mut out = vec![0.0f32; 256];
        kernels.dequant_q6_k(as_u8_slice(&block), &mut out);

        assert_eq!(out[0], 31.0);
    }

    #[test]
    fn test_gemm_q4() {
        let kernels = CpuKernels::<f32>::new();
        let m = 2; // batch
        let n = 2; // out_features
        let k = 256; // in_features (must be multiple of 256)
        
        let block_size = size_of::<BlockQ4K>();
        let mut weight_data = vec![0u8; n * (k / 256) * block_size];
        
        let w_ptr = weight_data.as_mut_ptr() as *mut BlockQ4K;
        
        unsafe {
            // Block 0 (Row 0)
            let b0 = &mut *w_ptr.add(0);
            b0.d = f16::from_f32(1.0);
            b0.dmin = f16::from_f32(0.0);
            b0.qs[0] = 0x88;
            b0.qs[1] = 0x97;
            
            // Block 1 (Row 1)
            let b1 = &mut *w_ptr.add(1);
            b1.d = f16::from_f32(2.0);
            b1.dmin = f16::from_f32(0.0);
            b1.qs[0] = 0x88;
            b1.qs[1] = 0x97;
        }
        
        let mut input = vec![0.0f32; m * k];
        input[2] = -1.0;
        input[3] = 1.0;
        input[k + 2] = -1.0;
        input[k + 3] = 1.0;
        
        let mut output = vec![0.0f32; m * n];
        
        kernels.gemm_q4(&weight_data, &input, &mut output, &[], m, n, k);
        
        assert_eq!(output[0], 2.0);
        assert_eq!(output[1], 4.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 4.0);
    }

    #[test]
    fn test_gemm_q4_scaled() {
        let kernels = CpuKernels::<f32>::new();
        let m = 1; 
        let n = 2; 
        let k = 256; 
        
        let block_size = size_of::<BlockQ4K>();
        let mut weight_data = vec![0u8; n * block_size];
        let w_ptr = weight_data.as_mut_ptr() as *mut BlockQ4K;
        unsafe {
             (*w_ptr.add(0)).d = f16::from_f32(1.0);
             (*w_ptr.add(0)).dmin = f16::from_f32(0.0);
             (*w_ptr.add(0)).qs[0] = 0x88;
             
             (*w_ptr.add(1)).d = f16::from_f32(1.0);
             (*w_ptr.add(1)).dmin = f16::from_f32(0.0);
             (*w_ptr.add(1)).qs[0] = 0x88;
        }
        
        let mut input = vec![0.0f32; m * k];
        input[0] = -1.0;
        input[1] = 1.0; 
        
        unsafe {
            (*w_ptr.add(0)).qs[0] = 0x97; // Block 0 -> 2.0
            (*w_ptr.add(1)).qs[0] = 0x97; // Block 1 -> 2.0
        }

        let mut output = vec![0.0f32; m * n];
        let scales = vec![10.0, 0.5]; 
        
        kernels.gemm_q4(&weight_data, &input, &mut output, &scales, m, n, k);
        
        assert_eq!(output[0], 20.0); 
        assert_eq!(output[1], 1.0);  
    }
}
