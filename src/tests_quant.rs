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
        // Set per-sub-block scale for sub-block 0: sc=1 (low nibble), m=0 (high nibble)
        // Formula: out = d * sc * q - dmin * m = 2.0 * 1 * q - 0 = 2*q
        block.scales[0] = 0x01;
        block.qs[0] = 0x1B; // q values: 3, 2, 1, 0

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

    // ========================================================================
    // Classic GGML dequant correctness tests (block_size=32)
    // ========================================================================

    #[test]
    fn test_dequant_q4_0_zero() {
        use crate::quant::BlockQ4_0;
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, qs all 0x88 -> each nibble = 8, val = 1.0*(8-8) = 0
        let block = BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs: [0x88; 16],
        };
        let mut out = vec![0.0f32; 32];
        kernels.dequant_q4_0(as_u8_slice(&block), &mut out);
        for i in 0..32 {
            assert!((out[i]).abs() < 1e-6, "q4_0 zero: out[{}] = {}", i, out[i]);
        }
    }

    #[test]
    fn test_dequant_q4_0_known_values() {
        use crate::quant::BlockQ4_0;
        let kernels = CpuKernels::<f32>::new();
        // d=0.5, qs[0] = 0x12 -> lo nibble=2, hi nibble=1
        // out[0] = 0.5*(2-8) = -3.0, out[1] = 0.5*(1-8) = -3.5
        let mut block = BlockQ4_0 {
            d: f16::from_f32(0.5),
            qs: [0x88; 16], // default zero-point
        };
        block.qs[0] = 0x12;
        block.qs[1] = 0xFA; // lo=0xA=10, hi=0xF=15 -> out[2]=0.5*(10-8)=1.0, out[3]=0.5*(15-8)=3.5

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q4_0(as_u8_slice(&block), &mut out);
        assert!((out[0] - (-3.0)).abs() < 1e-3, "q4_0: out[0]={}", out[0]);
        assert!((out[1] - (-3.5)).abs() < 1e-3, "q4_0: out[1]={}", out[1]);
        assert!((out[2] - 1.0).abs() < 1e-3, "q4_0: out[2]={}", out[2]);
        assert!((out[3] - 3.5).abs() < 1e-3, "q4_0: out[3]={}", out[3]);
    }

    #[test]
    fn test_dequant_q4_0_full_range() {
        use crate::quant::BlockQ4_0;
        let kernels = CpuKernels::<f32>::new();
        // d=2.0, fill qs with ramp pattern to test all nibble values
        let mut block = BlockQ4_0 {
            d: f16::from_f32(2.0),
            qs: [0; 16],
        };
        // qs[i] = (i*2+1) << 4 | (i*2) for i=0..8 -> covers nibbles 0..15
        for i in 0..8 {
            let lo = (i * 2) as u8;
            let hi = (i * 2 + 1) as u8;
            block.qs[i] = lo | (hi << 4);
        }
        let mut out = vec![0.0f32; 32];
        kernels.dequant_q4_0(as_u8_slice(&block), &mut out);
        for i in 0..8 {
            let lo = (i * 2) as f32;
            let hi = (i * 2 + 1) as f32;
            let expected_lo = 2.0 * (lo - 8.0);
            let expected_hi = 2.0 * (hi - 8.0);
            assert!((out[i * 2] - expected_lo).abs() < 1e-3,
                "q4_0 full: out[{}]={}, expected {}", i * 2, out[i * 2], expected_lo);
            assert!((out[i * 2 + 1] - expected_hi).abs() < 1e-3,
                "q4_0 full: out[{}]={}, expected {}", i * 2 + 1, out[i * 2 + 1], expected_hi);
        }
    }

    #[test]
    fn test_dequant_q4_1() {
        use crate::quant::BlockQ4_1;
        let kernels = CpuKernels::<f32>::new();
        // d=0.5, m=1.0, qs[0]=0x23 -> lo=3, hi=2
        // out[0] = 0.5*3 + 1.0 = 2.5, out[1] = 0.5*2 + 1.0 = 2.0
        let mut block = BlockQ4_1 {
            d: f16::from_f32(0.5),
            m: f16::from_f32(1.0),
            qs: [0; 16],
        };
        block.qs[0] = 0x23;
        block.qs[1] = 0x0F; // lo=15, hi=0 -> out[2]=0.5*15+1=8.5, out[3]=0.5*0+1=1.0

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q4_1(as_u8_slice(&block), &mut out);
        assert!((out[0] - 2.5).abs() < 1e-3, "q4_1: out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-3, "q4_1: out[1]={}", out[1]);
        assert!((out[2] - 8.5).abs() < 1e-3, "q4_1: out[2]={}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-3, "q4_1: out[3]={}", out[3]);
    }

    #[test]
    fn test_dequant_q4_1_zero_scale() {
        use crate::quant::BlockQ4_1;
        let kernels = CpuKernels::<f32>::new();
        // d=0, m=5.0 -> all outputs = 5.0
        let block = BlockQ4_1 {
            d: f16::from_f32(0.0),
            m: f16::from_f32(5.0),
            qs: [0xFF; 16], // all nibbles = 15, but d=0 so doesn't matter
        };
        let mut out = vec![0.0f32; 32];
        kernels.dequant_q4_1(as_u8_slice(&block), &mut out);
        for i in 0..32 {
            assert!((out[i] - 5.0).abs() < 1e-3, "q4_1 zero_scale: out[{}]={}", i, out[i]);
        }
    }

    #[test]
    fn test_dequant_q5_0() {
        use crate::quant::BlockQ5_0;
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, zero_point=16
        // Element 0: lo nibble from qs[0] low = 0xF=15, hi bit from qh[0] bit0 = 1
        //   q = (1<<4)|15 = 31, val = 1.0*(31-16) = 15.0
        // Element 1: hi nibble from qs[0] high = 0x0=0, hi bit from qh[0] bit1 = 0
        //   q = (0<<4)|0 = 0, val = 1.0*(0-16) = -16.0
        let mut block = BlockQ5_0 {
            d: f16::from_f32(1.0),
            qh: [0; 4],
            qs: [0; 16],
        };
        block.qs[0] = 0x0F; // lo nibble=15, hi nibble=0
        block.qh[0] = 0x01; // bit 0 set (element 0 hi bit = 1)

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q5_0(as_u8_slice(&block), &mut out);
        assert!((out[0] - 15.0).abs() < 1e-3, "q5_0: out[0]={}", out[0]);
        assert!((out[1] - (-16.0)).abs() < 1e-3, "q5_0: out[1]={}", out[1]);
    }

    #[test]
    fn test_dequant_q5_0_all_bits() {
        use crate::quant::BlockQ5_0;
        let kernels = CpuKernels::<f32>::new();
        // d=0.5, test element 8: qs[4] lo nibble, qh[1] bit 0
        let mut block = BlockQ5_0 {
            d: f16::from_f32(0.5),
            qh: [0; 4],
            qs: [0; 16],
        };
        // Element 8: qs[4] lo nibble = 7, qh[1] bit 0 = 1
        // q = (1<<4)|7 = 23, val = 0.5*(23-16) = 3.5
        block.qs[4] = 0x07;
        block.qh[1] = 0x01;

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q5_0(as_u8_slice(&block), &mut out);
        assert!((out[8] - 3.5).abs() < 1e-3, "q5_0 all_bits: out[8]={}", out[8]);
    }

    #[test]
    fn test_dequant_q5_1() {
        use crate::quant::BlockQ5_1;
        let kernels = CpuKernels::<f32>::new();
        // d=0.5, m=2.0
        // Element 0: qs[0] lo=0xA=10, qh[0] bit0=1 -> q=(1<<4)|10=26
        //   val = 0.5*26 + 2.0 = 15.0
        // Element 1: qs[0] hi=0x3=3, qh[0] bit1=0 -> q=(0<<4)|3=3
        //   val = 0.5*3 + 2.0 = 3.5
        let mut block = BlockQ5_1 {
            d: f16::from_f32(0.5),
            m: f16::from_f32(2.0),
            qh: [0; 4],
            qs: [0; 16],
        };
        block.qs[0] = 0x3A; // lo=0xA, hi=0x3
        block.qh[0] = 0x01; // bit 0 set

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q5_1(as_u8_slice(&block), &mut out);
        assert!((out[0] - 15.0).abs() < 1e-3, "q5_1: out[0]={}", out[0]);
        assert!((out[1] - 3.5).abs() < 1e-3, "q5_1: out[1]={}", out[1]);
    }

    #[test]
    fn test_dequant_q8_0() {
        use crate::quant::BlockQ8_0;
        let kernels = CpuKernels::<f32>::new();
        // d=0.5, qs[i] = i as i8
        let mut block = BlockQ8_0 {
            d: f16::from_f32(0.5),
            qs: [0i8; 32],
        };
        block.qs[0] = 10;
        block.qs[1] = -20;
        block.qs[31] = 127;

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q8_0(as_u8_slice(&block), &mut out);
        assert!((out[0] - 5.0).abs() < 1e-3, "q8_0: out[0]={}", out[0]);
        assert!((out[1] - (-10.0)).abs() < 1e-3, "q8_0: out[1]={}", out[1]);
        assert!((out[31] - 63.5).abs() < 1e-3, "q8_0: out[31]={}", out[31]);
    }

    #[test]
    fn test_dequant_q8_0_extreme() {
        use crate::quant::BlockQ8_0;
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, qs alternating -128 and 127
        let mut block = BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: [0i8; 32],
        };
        for i in 0..32 {
            block.qs[i] = if i % 2 == 0 { -128 } else { 127 };
        }
        let mut out = vec![0.0f32; 32];
        kernels.dequant_q8_0(as_u8_slice(&block), &mut out);
        for i in 0..32 {
            let expected = if i % 2 == 0 { -128.0 } else { 127.0 };
            assert!((out[i] - expected).abs() < 1e-3, "q8_0 extreme: out[{}]={}", i, out[i]);
        }
    }

    #[test]
    fn test_dequant_q8_1() {
        use crate::quant::BlockQ8_1;
        let kernels = CpuKernels::<f32>::new();
        // d=2.0, s is precomputed sum (unused in decode), qs[0]=5, qs[1]=-3
        let mut block = BlockQ8_1 {
            d: f16::from_f32(2.0),
            s: f16::from_f32(0.0), // unused in decode
            qs: [0i8; 32],
        };
        block.qs[0] = 5;
        block.qs[1] = -3;
        block.qs[15] = 100;

        let mut out = vec![0.0f32; 32];
        kernels.dequant_q8_1(as_u8_slice(&block), &mut out);
        assert!((out[0] - 10.0).abs() < 1e-3, "q8_1: out[0]={}", out[0]);
        assert!((out[1] - (-6.0)).abs() < 1e-3, "q8_1: out[1]={}", out[1]);
        assert!((out[15] - 200.0).abs() < 1e-3, "q8_1: out[15]={}", out[15]);
    }

    // ========================================================================
    // Classic GGML dot product tests
    // ========================================================================

    #[test]
    fn test_dot_q4_0() {
        use crate::quant::{BlockQ4_0, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // Build a block with known pattern, compute dot against known input
        let mut block = BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs: [0x88; 16], // all nibbles = 8, val = 0
        };
        // Set qs[0] = 0x9A -> lo=0xA=10, hi=0x9=9
        // val[0] = 1*(10-8) = 2, val[1] = 1*(9-8) = 1
        block.qs[0] = 0x9A;

        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        input[1] = 4.0;
        // Expected dot = 2*3 + 1*4 = 10.0
        let mut output = vec![0.0f32; 1];
        kernels.classic_matmul(as_u8_slice(&block), &input, &mut output, QuantType::Q4_0, 1, 1, 32);
        assert!((output[0] - 10.0).abs() < 1e-2, "dot_q4_0: got {}", output[0]);
    }

    #[test]
    fn test_dot_q4_1() {
        use crate::quant::{BlockQ4_1, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, m=0.5
        // qs[0] = 0x21 -> lo=1, hi=2
        // val[0] = 1*1 + 0.5 = 1.5, val[1] = 1*2 + 0.5 = 2.5
        let mut block = BlockQ4_1 {
            d: f16::from_f32(1.0),
            m: f16::from_f32(0.5),
            qs: [0; 16],
        };
        block.qs[0] = 0x21;

        let mut input = vec![0.0f32; 32];
        input[0] = 2.0;
        input[1] = 1.0;
        // Expected: 1.5*2 + 2.5*1 + 0.5*(sum of remaining 30 zeros in input) = 3.0 + 2.5 = 5.5
        let mut output = vec![0.0f32; 1];
        kernels.classic_matmul(as_u8_slice(&block), &input, &mut output, QuantType::Q4_1, 1, 1, 32);
        assert!((output[0] - 5.5).abs() < 1e-2, "dot_q4_1: got {}", output[0]);
    }

    #[test]
    fn test_dot_q5_0() {
        use crate::quant::{BlockQ5_0, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, element 0: qs[0] lo=5, qh[0] bit0=1 -> q=(1<<4)|5=21, val=21-16=5
        let mut block = BlockQ5_0 {
            d: f16::from_f32(1.0),
            qh: [0; 4],
            qs: [0; 16],
        };
        block.qs[0] = 0x05; // lo nibble=5, hi nibble=0
        block.qh[0] = 0x01; // element 0 hi bit = 1

        let mut input = vec![0.0f32; 32];
        input[0] = 2.0;
        // Element 0: val=5, Element 1: qs[0] hi=0, qh bit1=0 -> q=0, val=0-16=-16
        // dot = 5*2 + (-16)*0 = 10.0
        let mut output = vec![0.0f32; 1];
        kernels.classic_matmul(as_u8_slice(&block), &input, &mut output, QuantType::Q5_0, 1, 1, 32);
        assert!((output[0] - 10.0).abs() < 1e-2, "dot_q5_0: got {}", output[0]);
    }

    #[test]
    fn test_dot_q5_1() {
        use crate::quant::{BlockQ5_1, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, m=0.0
        // Element 0: qs[0] lo=3, qh[0] bit0=1 -> q=(1<<4)|3=19, val=1*19+0=19
        let mut block = BlockQ5_1 {
            d: f16::from_f32(1.0),
            m: f16::from_f32(0.0),
            qh: [0; 4],
            qs: [0; 16],
        };
        block.qs[0] = 0x03;
        block.qh[0] = 0x01;

        let mut input = vec![0.0f32; 32];
        input[0] = 1.0;
        // dot = 19*1 = 19.0
        let mut output = vec![0.0f32; 1];
        kernels.classic_matmul(as_u8_slice(&block), &input, &mut output, QuantType::Q5_1, 1, 1, 32);
        assert!((output[0] - 19.0).abs() < 1e-2, "dot_q5_1: got {}", output[0]);
    }

    #[test]
    fn test_dot_q8_0() {
        use crate::quant::{BlockQ8_0, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // d=0.5, qs[0]=10, qs[1]=-4
        let mut block = BlockQ8_0 {
            d: f16::from_f32(0.5),
            qs: [0i8; 32],
        };
        block.qs[0] = 10;
        block.qs[1] = -4;

        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        input[1] = 2.0;
        // val[0]=0.5*10=5, val[1]=0.5*(-4)=-2
        // dot = 5*3 + (-2)*2 = 15 - 4 = 11.0
        let mut output = vec![0.0f32; 1];
        kernels.classic_matmul(as_u8_slice(&block), &input, &mut output, QuantType::Q8_0, 1, 1, 32);
        assert!((output[0] - 11.0).abs() < 1e-2, "dot_q8_0: got {}", output[0]);
    }

    #[test]
    fn test_dot_q8_1() {
        use crate::quant::{BlockQ8_1, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // d=1.0, qs[0]=7, qs[1]=-3
        let mut block = BlockQ8_1 {
            d: f16::from_f32(1.0),
            s: f16::from_f32(0.0),
            qs: [0i8; 32],
        };
        block.qs[0] = 7;
        block.qs[1] = -3;

        let mut input = vec![0.0f32; 32];
        input[0] = 2.0;
        input[1] = 5.0;
        // val[0]=7, val[1]=-3
        // dot = 7*2 + (-3)*5 = 14 - 15 = -1.0
        let mut output = vec![0.0f32; 1];
        kernels.classic_matmul(as_u8_slice(&block), &input, &mut output, QuantType::Q8_1, 1, 1, 32);
        assert!((output[0] - (-1.0)).abs() < 1e-2, "dot_q8_1: got {}", output[0]);
    }

    // ========================================================================
    // Classic matmul multi-row test
    // ========================================================================

    #[test]
    fn test_classic_matmul_q4_0_multi_row() {
        use crate::quant::{BlockQ4_0, QuantType};
        let kernels = CpuKernels::<f32>::new();
        // classic_matmul: m = weight rows (output features), n = batch columns, k = inner dim
        // input layout: [k, n] col-major, so input.len() = k * n
        // output layout: [m, n], so output.len() = m * n
        // weight layout: m * blocks_per_row * block_bytes
        let m = 2; // 2 output features (weight rows)
        let n = 1; // 1 batch column
        let k = 32; // inner dimension (1 block)
        let block_size = size_of::<BlockQ4_0>();
        let mut weight_data = vec![0u8; m * block_size];
        let w_ptr = weight_data.as_mut_ptr() as *mut BlockQ4_0;
        unsafe {
            // Row 0: d=1.0, all zero-point (nibble=8)
            let b0 = &mut *w_ptr.add(0);
            b0.d = f16::from_f32(1.0);
            b0.qs = [0x88; 16];
            b0.qs[0] = 0x9A; // val[0]=2, val[1]=1

            // Row 1: d=2.0, all zero-point
            let b1 = &mut *w_ptr.add(1);
            b1.d = f16::from_f32(2.0);
            b1.qs = [0x88; 16];
            b1.qs[0] = 0x9A; // val[0]=4, val[1]=2
        }

        // input: [k, n] = [32, 1], col-major
        let mut input = vec![0.0f32; k * n];
        input[0] = 1.0; // input[0*n+0] = 1.0
        input[1] = 1.0; // input[1*n+0] = 1.0
        // Row 0 dot: 2*1 + 1*1 = 3.0
        // Row 1 dot: 4*1 + 2*1 = 6.0
        let mut output = vec![0.0f32; m * n];
        kernels.classic_matmul(&weight_data, &input, &mut output, QuantType::Q4_0, m, n, k);
        assert!((output[0] - 3.0).abs() < 1e-2, "classic_matmul q4_0 row0: got {}", output[0]);
        assert!((output[1] - 6.0).abs() < 1e-2, "classic_matmul q4_0 row1: got {}", output[1]);
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
