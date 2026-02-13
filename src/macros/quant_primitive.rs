/// Defines quantization primitives (decode, dot).
/// 
/// Layer 3 of the macro architecture.
#[macro_export]
macro_rules! quant_primitive {
    // ------------------------------------------------------------------------
    // Q4_K Decoding
    // ------------------------------------------------------------------------

    (scalar, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
             let block = unsafe { &*$block_ptr };
             let d: f32 = block.d.to_f32();
             // Q4_K has 256 weights. qs has 128 bytes.
             for i in 0..128 {
                 let b = block.qs[i];
                 let l = b & 0x0F;
                 let h = b >> 4;
                 unsafe {
                     *$out_ptr.add(i*2) = d * (l as f32 - 8.0);
                     *$out_ptr.add(i*2+1) = d * (h as f32 - 8.0);
                 }
             }
        }
    };

    (avx2, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "x86_64")]
             {
                use std::arch::x86_64::*;
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32(); 
                let vd = _mm256_set1_ps(d);
                let v_offset = _mm256_set1_ps(8.0);
                
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                
                for i in 0..8 {
                    // Load 16 bytes (32 nibbles)
                    let v128 = _mm_loadu_si128(qs_ptr.add(i*16) as *const _);
                    
                    // Extract Low/High using masks
                    let mask = _mm_set1_epi8(0x0F);
                    let low_bytes = _mm_and_si128(v128, mask);
                    let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                    
                    // Convert Low 8 bytes -> 8 i32 -> 8 f32
                    let ints_l_0 = _mm256_cvtepu8_epi32(low_bytes); 
                    let ints_l_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8));
                    
                    let f_l_0 = _mm256_cvtepi32_ps(ints_l_0);
                    let f_l_1 = _mm256_cvtepi32_ps(ints_l_1);
                    
                    // Convert High 8 bytes -> 8 i32 -> 8 f32
                    let ints_h_0 = _mm256_cvtepu8_epi32(high_bytes);
                    let ints_h_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8));
                    
                    let f_h_0 = _mm256_cvtepi32_ps(ints_h_0);
                    let f_h_1 = _mm256_cvtepi32_ps(ints_h_1);
                    
                    // Dequantize: d * (val - 8.0)
                    let res_l_0 = _mm256_mul_ps(vd, _mm256_sub_ps(f_l_0, v_offset));
                    let res_l_1 = _mm256_mul_ps(vd, _mm256_sub_ps(f_l_1, v_offset));
                    let res_h_0 = _mm256_mul_ps(vd, _mm256_sub_ps(f_h_0, v_offset));
                    let res_h_1 = _mm256_mul_ps(vd, _mm256_sub_ps(f_h_1, v_offset));
                    
                    // Interleave: Store Low/High pairs
                    // We want [l0, h0, l1, h1...]
                    let out_0 = _mm256_unpacklo_ps(res_l_0, res_h_0);
                    let out_1 = _mm256_unpackhi_ps(res_l_0, res_h_0);
                    let out_2 = _mm256_unpacklo_ps(res_l_1, res_h_1);
                    let out_3 = _mm256_unpackhi_ps(res_l_1, res_h_1);
                    
                    // Permute to correct linear order for store
                    let final_0 = _mm256_permute2f128_ps(out_0, out_1, 0x20);
                    let final_1 = _mm256_permute2f128_ps(out_0, out_1, 0x31);
                    let final_2 = _mm256_permute2f128_ps(out_2, out_3, 0x20);
                    let final_3 = _mm256_permute2f128_ps(out_2, out_3, 0x31);
                    
                    _mm256_storeu_ps(out_ptr.add(i*32 + 0), final_0);
                    _mm256_storeu_ps(out_ptr.add(i*32 + 8), final_1);
                    _mm256_storeu_ps(out_ptr.add(i*32 + 16), final_2);
                    _mm256_storeu_ps(out_ptr.add(i*32 + 24), final_3);
                }
             }
        }
    };

    (neon, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "aarch64")]
             {
                use std::arch::aarch64::*;
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32(); 
                let vd = vdupq_n_f32(d);
                let v_offset = vdupq_n_f32(8.0);
                
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                
                // 128 bytes qs -> 256 nibbles. 
                // Each iteration processes 16 bytes = 32 values.
                // 128/16 = 8 iterations.
                for i in 0..8 {
                    // Load 16 bytes
                    let v128 = vld1q_u8(qs_ptr.add(i*16));
                    
                    // Low nibbles
                    let mask = vdupq_n_u8(0x0F);
                    let low_bytes = vandq_u8(v128, mask);
                    // High nibbles
                    let high_bytes = vshrq_n_u8(v128, 4); 
                    // (vshr shifts in 0s for unsigned)
                    
                    // Expand low_bytes (16x u8) -> 4x float32x4_t
                    // u8 -> u16
                    let low_u16_lo = vmovl_u8(vget_low_u8(low_bytes)); // low 8 -> u16x8
                    let low_u16_hi = vmovl_high_u8(low_bytes);         // high 8 -> u16x8
                    
                    // u16 -> u32
                    let low_u32_0 = vmovl_u16(vget_low_u16(low_u16_lo));
                    let low_u32_1 = vmovl_high_u16(low_u16_lo);
                    let low_u32_2 = vmovl_u16(vget_low_u16(low_u16_hi));
                    let low_u32_3 = vmovl_high_u16(low_u16_hi);
                    
                    // u32 -> f32
                    let f_l_0 = vcvtq_f32_u32(low_u32_0);
                    let f_l_1 = vcvtq_f32_u32(low_u32_1);
                    let f_l_2 = vcvtq_f32_u32(low_u32_2);
                    let f_l_3 = vcvtq_f32_u32(low_u32_3);
                    
                    // HIGH Expand
                    let high_u16_lo = vmovl_u8(vget_low_u8(high_bytes));
                    let high_u16_hi = vmovl_high_u8(high_bytes);
                    
                    let high_u32_0 = vmovl_u16(vget_low_u16(high_u16_lo));
                    let high_u32_1 = vmovl_high_u16(high_u16_lo);
                    let high_u32_2 = vmovl_u16(vget_low_u16(high_u16_hi));
                    let high_u32_3 = vmovl_high_u16(high_u16_hi);
                    
                    let f_h_0 = vcvtq_f32_u32(high_u32_0);
                    let f_h_1 = vcvtq_f32_u32(high_u32_1);
                    let f_h_2 = vcvtq_f32_u32(high_u32_2);
                    let f_h_3 = vcvtq_f32_u32(high_u32_3);
                    
                    // Dequant: d * (val - 8.0)
                    // FMA: -8*d + val*d ? Or sub then mul.
                    // vsubq_f32, then vmulq_f32.
                    // Or vfmaq_f32.
                    // let res = vfmaq_f32(vnegq_f32(vmulq_f32(vd, v_offset)), val, vd)
                    // Simpler: (val - 8.0) * d
                    let res_l_0 = vmulq_f32(vsubq_f32(f_l_0, v_offset), vd);
                    let res_l_1 = vmulq_f32(vsubq_f32(f_l_1, v_offset), vd);
                    let res_l_2 = vmulq_f32(vsubq_f32(f_l_2, v_offset), vd);
                    let res_l_3 = vmulq_f32(vsubq_f32(f_l_3, v_offset), vd);
                    
                    let res_h_0 = vmulq_f32(vsubq_f32(f_h_0, v_offset), vd);
                    let res_h_1 = vmulq_f32(vsubq_f32(f_h_1, v_offset), vd);
                    let res_h_2 = vmulq_f32(vsubq_f32(f_h_2, v_offset), vd);
                    let res_h_3 = vmulq_f32(vsubq_f32(f_h_3, v_offset), vd);
                    
                    // Interleave stores!
                    // We have l0..3 and h0..3.
                    // We want [l0, h0, l1, h1, l2, h2, l3, h3] for each pair.
                    // zip l0, h0 -> 8 floats. Store.
                    // st2 (store 2 interleaved) is perfect for this!
                    // vst2q_f32(ptr, float32x4x2_t { val: [res_l_0, res_h_0] })
                    
                    // 0..3
                    let pair0 = float32x4x2_t(res_l_0, res_h_0);
                    vst2q_f32(out_ptr.add(i*32 + 0), pair0);
                    
                    // 4..7
                    let pair1 = float32x4x2_t(res_l_1, res_h_1);
                    vst2q_f32(out_ptr.add(i*32 + 8), pair1);
                    
                    // 8..11
                    let pair2 = float32x4x2_t(res_l_2, res_h_2);
                    vst2q_f32(out_ptr.add(i*32 + 16), pair2);
                    
                    // 12..15
                    let pair3 = float32x4x2_t(res_l_3, res_h_3);
                    vst2q_f32(out_ptr.add(i*32 + 24), pair3);
                }
             }
        }
    };
    
    // ------------------------------------------------------------------------
    // Q8_K Decoding
    // ------------------------------------------------------------------------
    
     (scalar, q8_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
             let block = unsafe { &*$block_ptr };
             let d = block.d;
             for i in 0..256 {
                 let q = block.qs[i]; // i8
                 unsafe { *$out_ptr.add(i) = d * (q as f32); }
             }
        }
    };

    // ------------------------------------------------------------------------
    // Q2_K Decoding
    // ------------------------------------------------------------------------
    (scalar, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let dmin: f32 = block.dmin.to_f32();
            // qs: 64 bytes, 256 vals. 4 vals/byte.
            for i in 0..64 {
                let b = block.qs[i];
                let v0 = b & 0x03;
                let v1 = (b >> 2) & 0x03;
                let v2 = (b >> 4) & 0x03;
                let v3 = (b >> 6) & 0x03;
                unsafe {
                    *$out_ptr.add(i*4+0) = d * (v0 as f32) - dmin;
                    *$out_ptr.add(i*4+1) = d * (v1 as f32) - dmin;
                    *$out_ptr.add(i*4+2) = d * (v2 as f32) - dmin;
                    *$out_ptr.add(i*4+3) = d * (v3 as f32) - dmin;
                }
            }
        }
    };

    // ------------------------------------------------------------------------
    // Q3_K Decoding
    // ------------------------------------------------------------------------
    (scalar, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d;
            for i in 0..256 {
                let low = (block.qs[i/4] >> ((i%4)*2)) & 3;
                let high = (block.hmask[i/8] >> (i%8)) & 1;
                let val = (high << 2) | low;
                unsafe { *$out_ptr.add(i) = d * (val as f32 - 4.0); }
            }
        }
    };

    // ------------------------------------------------------------------------
    // Q5_K Decoding
    // ------------------------------------------------------------------------
    (scalar, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let dmin: f32 = block.dmin.to_f32();
            for i in 0..256 {
                let low = (block.qs[i/2] >> ((i%2)*4)) & 0xF;
                let high = (block.qh[i/8] >> (i%8)) & 1;
                let val = (high << 4) | low;
                unsafe { *$out_ptr.add(i) = d * (val as f32) - dmin; }
            }
        }
    };

    // ------------------------------------------------------------------------
    // Q6_K Decoding
    // ------------------------------------------------------------------------
    (scalar, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d;
            for i in 0..256 {
                let low = (block.qs[i/2] >> ((i%2)*4)) & 0xF;
                let high = (block.qh[i/4] >> ((i%4)*2)) & 3;
                let val = (high << 4) | low;
                unsafe { *$out_ptr.add(i) = d * (val as f32 - 32.0); }
            }
        }
    };


    // ------------------------------------------------------------------------
    // Dot Products (Scalar)
    // ------------------------------------------------------------------------

    (avx2, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "x86_64")]
             {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32(); 
                let vd = _mm256_set1_ps(d);
                let v_offset = _mm256_set1_ps(8.0);
                
                let qs_ptr = block.qs.as_ptr();
                let other_ptr = $other_ptr;
                let mut acc = _mm256_setzero_ps();
                
                for i in 0..8 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i*16) as *const _);
                    let mask = _mm_set1_epi8(0x0F);
                    let low_bytes = _mm_and_si128(v128, mask);
                    let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                    
                    let f_l_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_bytes));
                    let f_l_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8)));
                    let f_h_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_bytes));
                    let f_h_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8)));
                    
                    let dq_l_0 = _mm256_mul_ps(vd, _mm256_sub_ps(f_l_0, v_offset));
                    let dq_l_1 = _mm256_mul_ps(vd, _mm256_sub_ps(f_l_1, v_offset));
                    let dq_h_0 = _mm256_mul_ps(vd, _mm256_sub_ps(f_h_0, v_offset));
                    let dq_h_1 = _mm256_mul_ps(vd, _mm256_sub_ps(f_h_1, v_offset));
                    
                    let out_0 = _mm256_unpacklo_ps(dq_l_0, dq_h_0);
                    let out_1 = _mm256_unpackhi_ps(dq_l_0, dq_h_0);
                    let final_0 = _mm256_permute2f128_ps(out_0, out_1, 0x20); 
                    let final_1 = _mm256_permute2f128_ps(out_0, out_1, 0x31);
                    
                    let out_2 = _mm256_unpacklo_ps(dq_l_1, dq_h_1);
                    let out_3 = _mm256_unpackhi_ps(dq_l_1, dq_h_1);
                    let final_2 = _mm256_permute2f128_ps(out_2, out_3, 0x20); 
                    let final_3 = _mm256_permute2f128_ps(out_2, out_3, 0x31);
                    
                    let other_0 = _mm256_loadu_ps(other_ptr.add(i*32 + 0));
                    let other_1 = _mm256_loadu_ps(other_ptr.add(i*32 + 8));
                    let other_2 = _mm256_loadu_ps(other_ptr.add(i*32 + 16));
                    let other_3 = _mm256_loadu_ps(other_ptr.add(i*32 + 24));
                    
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(final_0, other_0));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(final_1, other_1));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(final_2, other_2));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(final_3, other_3));
                }
                
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let t3 = _mm256_extractf128_ps(t2, 1);
                let t4 = _mm256_castps256_ps128(t2);
                let res = _mm_add_ps(t3, t4);
                _mm_cvtss_f32(res)
             }
        }
    };

    (scalar, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..128 {
                 let b = block.qs[i];
                 let l = (b & 0x0F) as f32;
                 let h = (b >> 4) as f32;
                 unsafe {
                     sum += (d * (l - 8.0)) * *other.add(i*2);
                     sum += (d * (h - 8.0)) * *other.add(i*2+1);
                 }
            }
            sum
        }
    };

    (scalar, q8_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..256 {
                let q = block.qs[i] as f32;
                unsafe { sum += (d * q) * *other.add(i); }
            }
            sum
        }
    };

    (scalar, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let dmin: f32 = block.dmin.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..64 {
                let b = block.qs[i];
                let v0 = (b & 0x03) as f32;
                let v1 = ((b >> 2) & 0x03) as f32;
                let v2 = ((b >> 4) & 0x03) as f32;
                let v3 = ((b >> 6) & 0x03) as f32;
                unsafe {
                    sum += (d * v0 - dmin) * *other.add(i*4);
                    sum += (d * v1 - dmin) * *other.add(i*4+1);
                    sum += (d * v2 - dmin) * *other.add(i*4+2);
                    sum += (d * v3 - dmin) * *other.add(i*4+3);
                }
            }
            sum
        }
    };

    (scalar, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..256 {
                let l = (block.qs[i/4] >> ((i%4)*2)) & 3;
                let h = (block.hmask[i/8] >> (i%8)) & 1;
                let val = ((h << 2) | l) as f32;
                unsafe { sum += (d * (val - 4.0)) * *other.add(i); }
            }
            sum
        }
    };

    (scalar, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..256 {
                let l = (block.qs[i/2] >> ((i%2)*4)) & 0xF;
                let h = (block.qh[i/8] >> (i%8)) & 1;
                let val = ((h << 4) | l) as f32;
                unsafe { sum += (d * val - dmin) * *other.add(i); }
            }
            sum
        }
    };

    (scalar, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..256 {
                let l = (block.qs[i/2] >> ((i%2)*4)) & 0xF;
                let h = (block.qh[i/4] >> ((i%4)*2)) & 3;
                let val = ((h << 4) | l) as f32;
                unsafe { sum += (d * (val - 32.0)) * *other.add(i); }
            }
            sum
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q2_K
    // ========================================================================
    (avx2, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = _mm256_set1_ps(d);
                let v_dmin = _mm256_set1_ps(dmin);
                let mask2 = _mm_set1_epi8(0x03);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                // 64 bytes qs, 4 values per byte = 256 values
                // Process 16 bytes at a time = 64 values
                for i in 0..4 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let v0 = _mm_and_si128(v128, mask2);
                    let v1 = _mm_and_si128(_mm_srli_epi16(v128, 2), mask2);
                    let v2 = _mm_and_si128(_mm_srli_epi16(v128, 4), mask2);
                    let v3 = _mm_and_si128(_mm_srli_epi16(v128, 6), mask2);
                    // Convert each 16 bytes to 2x8 f32 and store
                    for (j, vv) in [(0usize, v0), (1, v1), (2, v2), (3, v3)] {
                        let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vv));
                        let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vv, 8)));
                        let r_lo = _mm256_sub_ps(_mm256_mul_ps(vd, lo), v_dmin);
                        let r_hi = _mm256_sub_ps(_mm256_mul_ps(vd, hi), v_dmin);
                        let base = i * 64 + j * 16;
                        _mm256_storeu_ps(out_ptr.add(base), r_lo);
                        _mm256_storeu_ps(out_ptr.add(base + 8), r_hi);
                    }
                }
            }
        }
    };

    (avx2, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = _mm256_set1_ps(d);
                let v_dmin = _mm256_set1_ps(dmin);
                let mask2 = _mm_set1_epi8(0x03);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = _mm256_setzero_ps();
                for i in 0..4 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let v0 = _mm_and_si128(v128, mask2);
                    let v1 = _mm_and_si128(_mm_srli_epi16(v128, 2), mask2);
                    let v2 = _mm_and_si128(_mm_srli_epi16(v128, 4), mask2);
                    let v3 = _mm_and_si128(_mm_srli_epi16(v128, 6), mask2);
                    for (j, vv) in [(0usize, v0), (1, v1), (2, v2), (3, v3)] {
                        let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vv));
                        let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vv, 8)));
                        let dq_lo = _mm256_sub_ps(_mm256_mul_ps(vd, lo), v_dmin);
                        let dq_hi = _mm256_sub_ps(_mm256_mul_ps(vd, hi), v_dmin);
                        let base = i * 64 + j * 16;
                        let o_lo = _mm256_loadu_ps(other.add(base));
                        let o_hi = _mm256_loadu_ps(other.add(base + 8));
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(dq_lo, o_lo));
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(dq_hi, o_hi));
                    }
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q3_K
    // ========================================================================
    (avx2, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        // Q3_K: 3-bit with high mask. Scalar fallback for AVX2 (complex bit layout).
        $crate::quant_primitive!(scalar, q3_k, decode, $block_ptr, $out_ptr)
    };

    (avx2, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive!(scalar, q3_k, dot, $block_ptr, $other_ptr)
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q5_K
    // ========================================================================
    (avx2, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        // Q5_K: 5-bit with high bit mask. Scalar fallback for AVX2.
        $crate::quant_primitive!(scalar, q5_k, decode, $block_ptr, $out_ptr)
    };

    (avx2, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive!(scalar, q5_k, dot, $block_ptr, $other_ptr)
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q6_K
    // ========================================================================
    (avx2, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        $crate::quant_primitive!(scalar, q6_k, decode, $block_ptr, $out_ptr)
    };

    (avx2, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive!(scalar, q6_k, dot, $block_ptr, $other_ptr)
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q8_K
    // ========================================================================
    (avx2, q8_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = _mm256_set1_ps(d);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                // 256 i8 values. Process 8 at a time = 32 iterations.
                for i in 0..32 {
                    // Load 8 i8 -> i32 -> f32
                    let v64 = _mm_loadl_epi64(qs_ptr.add(i * 8) as *const _);
                    let vi32 = _mm256_cvtepi8_epi32(v64);
                    let vf = _mm256_cvtepi32_ps(vi32);
                    let res = _mm256_mul_ps(vd, vf);
                    _mm256_storeu_ps(out_ptr.add(i * 8), res);
                }
            }
        }
    };

    (avx2, q8_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = _mm256_set1_ps(d);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = _mm256_setzero_ps();
                for i in 0..32 {
                    let v64 = _mm_loadl_epi64(qs_ptr.add(i * 8) as *const _);
                    let vi32 = _mm256_cvtepi8_epi32(v64);
                    let vf = _mm256_cvtepi32_ps(vi32);
                    let dq = _mm256_mul_ps(vd, vf);
                    let vo = _mm256_loadu_ps(other.add(i * 8));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(dq, vo));
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    // ========================================================================
    // IQ Scalar decode/dot
    // ========================================================================

    // IQ1_S: 1-bit with IQ1S_GRID codebook lookup
    (scalar, iq1_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;      // [u8; 32]
            let qh = &block.qh;      // [u16; 8]
            let scales = &block.scales; // [u8; 16]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 32 (8 groups total)
            for group in 0..8 {
                // Extract grid index from qs (simplified: use byte pairs as index)
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF; // 11 bits for 2048 entries
                let grid_val = crate::codebooks::IQ1S_GRID[idx];

                // Extract scale for this group
                let scale = scales[group * 2] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    // Apply sign from qh
                    let sign_bit = (qh[group] >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 32 + j) = group_d * (signed_v as f32); }
                }

                // Second half of the group (next 8 values)
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_bit = (qh[group] >> (j + 8)) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 32 + 8 + j) = group_d * (signed_v as f32); }
                }

                // Remaining 16 values (simplified: reuse grid pattern)
                for j in 0..16 {
                    let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 32 + 16 + j) = group_d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let qh = &block.qh;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..8 {
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                let grid_val = crate::codebooks::IQ1S_GRID[idx];
                let scale = scales[group * 2] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_bit = (qh[group] >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 32 + j); }
                }

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_bit = (qh[group] >> (j + 8)) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 32 + 8 + j); }
                }

                for j in 0..16 {
                    let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                    unsafe { sum += (group_d * (v as f32)) * *other.add(group * 32 + 16 + j); }
                }
            }
            sum
        }
    };

    // IQ1_M: 1-bit with IQ1S_GRID codebook (same grid as IQ1_S)
    (scalar, iq1_m, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let qs = &block.qs;      // [u8; 32]
            let qh = &block.qh;      // [u8; 16]
            let scales = &block.scales; // [u8; 8]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 32 (8 groups total)
            for group in 0..8 {
                // Extract grid index from qs
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                let grid_val = crate::codebooks::IQ1S_GRID[idx];

                // Extract scale for this group
                let scale = scales[group] as f32 / 255.0;

                // Unpack 8 int8 values from u64 grid entry (4 times for 32 values)
                for rep in 0..4 {
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        // Apply sign from qh
                        let qh_idx = group * 2 + rep / 2;
                        let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                        let signed_v = if sign_bit == 0 { v } else { -v };
                        unsafe { *out_ptr.add(group * 32 + rep * 8 + j) = scale * (signed_v as f32); }
                    }
                }
            }
        }
    };

    (scalar, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let qs = &block.qs;
            let qh = &block.qh;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..8 {
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                let grid_val = crate::codebooks::IQ1S_GRID[idx];
                let scale = scales[group] as f32 / 255.0;

                for rep in 0..4 {
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let qh_idx = group * 2 + rep / 2;
                        let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                        let signed_v = if sign_bit == 0 { v } else { -v };
                        unsafe { sum += (scale * (signed_v as f32)) * *other.add(group * 32 + rep * 8 + j); }
                    }
                }
            }
            sum
        }
    };

    // IQ2_XXS: 2-bit with IQ2XXS_GRID codebook
    (scalar, iq2_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u16; 32]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 8 (32 groups total)
            for group in 0..32 {
                // Extract grid index from qs (low 8 bits)
                let idx = (qs[group] & 0xFF) as usize;
                let grid_val = crate::codebooks::IQ2XXS_GRID[idx];

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 8 + j) = d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u16; 32]
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..32 {
                let idx = (qs[group] & 0xFF) as usize;
                let grid_val = crate::codebooks::IQ2XXS_GRID[idx];

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { sum += (d * (v as f32)) * *other.add(group * 8 + j); }
                }
            }
            sum
        }
    };

    // IQ2_XS: 2-bit with IQ2XS_GRID codebook
    (scalar, iq2_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u16; 32]
            let scales = &block.scales; // [u8; 8]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 8 (32 groups total)
            for group in 0..32 {
                // Extract grid index from qs (low 9 bits for 512 entries)
                let idx = (qs[group] & 0x1FF) as usize;
                let grid_val = crate::codebooks::IQ2XS_GRID[idx];

                // Extract scale for this group
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    // Apply sign from KSIGNS_IQ2XS
                    let sign_byte = crate::codebooks::KSIGNS_IQ2XS[((qs[group] >> 9) & 0x7F) as usize];
                    let sign_bit = (sign_byte >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 8 + j) = group_d * (signed_v as f32); }
                }
            }
        }
    };

    (scalar, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..32 {
                let idx = (qs[group] & 0x1FF) as usize;
                let grid_val = crate::codebooks::IQ2XS_GRID[idx];
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_byte = crate::codebooks::KSIGNS_IQ2XS[((qs[group] >> 9) & 0x7F) as usize];
                    let sign_bit = (sign_byte >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 8 + j); }
                }
            }
            sum
        }
    };

    // IQ2_S: 2-bit with IQ2S_GRID codebook
    (scalar, iq2_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 64]
            let qh = &block.qh; // [u8; 8]
            let scales = &block.scales; // [u8; 8]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 8 (32 groups total)
            for group in 0..32 {
                // Extract grid index from qs (10 bits: 8 from qs + 2 from qh)
                let qs_idx = group * 2;
                let qh_idx = group / 4;
                let qh_shift = (group % 4) * 2;
                let idx = ((qs[qs_idx] as usize) |
                          ((qs[qs_idx + 1] as usize) << 8) |
                          (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF; // 10 bits for 1024 entries
                let grid_val = crate::codebooks::IQ2S_GRID[idx];

                // Extract scale for this group
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 8 + j) = group_d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 64]
            let qh = &block.qh;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..32 {
                let qs_idx = group * 2;
                let qh_idx = group / 4;
                let qh_shift = (group % 4) * 2;
                let idx = ((qs[qs_idx] as usize) |
                          ((qs[qs_idx + 1] as usize) << 8) |
                          (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                let grid_val = crate::codebooks::IQ2S_GRID[idx];
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { sum += (group_d * (v as f32)) * *other.add(group * 8 + j); }
                }
            }
            sum
        }
    };

    // IQ3_XXS: 3-bit with IQ3XXS_GRID codebook
    (scalar, iq3_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 96] = 3*QK_K/8
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 4 (64 groups total)
            // Each grid entry (u32) contains 4 packed int8 values
            for group in 0..64 {
                // Extract 8-bit index from qs
                let idx = qs[group] as usize;
                let grid_val = crate::codebooks::IQ3XXS_GRID[idx];

                // Unpack 4 int8 values from u32 grid entry
                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 4 + j) = d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..64 {
                let idx = qs[group] as usize;
                let grid_val = crate::codebooks::IQ3XXS_GRID[idx];

                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { sum += (d * (v as f32)) * *other.add(group * 4 + j); }
                }
            }
            sum
        }
    };

    // IQ3_S: 3-bit with IQ3S_GRID codebook
    (scalar, iq3_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 64]
            let qh = &block.qh; // [u8; 8]
            let signs = &block.signs; // [u8; 32]
            let scales = &block.scales; // [u8; 4]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 4 (64 groups total)
            for group in 0..64 {
                // Extract 9-bit index from qs (8 bits) + qh (1 bit)
                let qh_idx = group / 8;
                let qh_shift = group % 8;
                let idx = ((qs[group] as usize) |
                          (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF; // 9 bits for 512 entries
                let grid_val = crate::codebooks::IQ3S_GRID[idx];

                // Extract scale for this group
                let scale_idx = group / 16;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 4 int8 values from u32 grid entry
                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    // Apply sign from signs array
                    let sign_idx = group / 2;
                    let sign_shift = (group % 2) * 4 + j;
                    let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 4 + j) = group_d * (signed_v as f32); }
                }
            }
        }
    };

    (scalar, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let qh = &block.qh;
            let signs = &block.signs;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..64 {
                let qh_idx = group / 8;
                let qh_shift = group % 8;
                let idx = ((qs[group] as usize) |
                          (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                let grid_val = crate::codebooks::IQ3S_GRID[idx];
                let scale_idx = group / 16;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_idx = group / 2;
                    let sign_shift = (group % 2) * 4 + j;
                    let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 4 + j); }
                }
            }
            sum
        }
    };

    // IQ4_NL: 4-bit non-linear codebook lookup
    (scalar, iq4_nl, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let out_ptr = $out_ptr;
            // IQ4_NL block_size=32, qs has 16 bytes
            for i in 0..16 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    *out_ptr.add(i * 2) = d * (crate::codebooks::KVALUES_IQ4NL[lo] as f32);
                    *out_ptr.add(i * 2 + 1) = d * (crate::codebooks::KVALUES_IQ4NL[hi] as f32);
                }
            }
        }
    };

    (scalar, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..16 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[lo] as f32) * *other.add(i * 2);
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[hi] as f32) * *other.add(i * 2 + 1);
                }
            }
            sum
        }
    };

    // IQ4_XS: uses same codebook as IQ4_NL but block_size=256
    (scalar, iq4_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let out_ptr = $out_ptr;
            for i in 0..128 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    *out_ptr.add(i * 2) = d * (crate::codebooks::KVALUES_IQ4NL[lo] as f32);
                    *out_ptr.add(i * 2 + 1) = d * (crate::codebooks::KVALUES_IQ4NL[hi] as f32);
                }
            }
        }
    };

    (scalar, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..128 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[lo] as f32) * *other.add(i * 2);
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[hi] as f32) * *other.add(i * 2 + 1);
                }
            }
            sum
        }
    };

    // ========================================================================
    // AWQ4/GPTQ4/Squeeze Scalar decode/dot
    // ========================================================================

    // AWQ4: group-wise 4-bit with scales+zeros
    (scalar, awq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let out_ptr = $out_ptr;
            // AWQ4 block_size=128: 32 u32 words, each holding 8 nibbles
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { *out_ptr.add(w * 8 + nib) = d * (q - 8.0); }
                }
            }
        }
    };

    (scalar, awq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { sum += (d * (q - 8.0)) * *other.add(w * 8 + nib); }
                }
            }
            sum
        }
    };

    // GPTQ4: same layout as AWQ4 for decode/dot
    (scalar, gptq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let out_ptr = $out_ptr;
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { *out_ptr.add(w * 8 + nib) = d * (q - 8.0); }
                }
            }
        }
    };

    (scalar, gptq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { sum += (d * (q - 8.0)) * *other.add(w * 8 + nib); }
                }
            }
            sum
        }
    };

    // Squeeze: 3-bit SqueezeLLM
    (scalar, squeeze, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let out_ptr = $out_ptr;
            // 3-bit packed: 256 values in 96 bytes (256*3/8=96), but block has 128 bytes qs
            for i in 0..256 {
                let bit_offset = i * 3;
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let q = if bit_idx <= 5 {
                    (qs[byte_idx] >> bit_idx) & 0x07
                } else {
                    let lo = qs[byte_idx] >> bit_idx;
                    let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                    (lo | hi) & 0x07
                };
                unsafe { *out_ptr.add(i) = d * (q as f32 - 4.0); }
            }
        }
    };

    (scalar, squeeze, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..256 {
                let bit_offset = i * 3;
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let q = if bit_idx <= 5 {
                    (qs[byte_idx] >> bit_idx) & 0x07
                } else {
                    let lo = qs[byte_idx] >> bit_idx;
                    let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                    (lo | hi) & 0x07
                };
                unsafe { sum += (d * (q as f32 - 4.0)) * *other.add(i); }
            }
            sum
        }
    };

    // Generic fallback for any other arch/op
    ($isa:ident, $qty:ident, $op:ident, $block:expr, $out_ptr:expr) => {
        $crate::quant_primitive!(scalar, $qty, $op, $block, $out_ptr)
    };
    
    // Dot fallback
    ($isa:ident, $qty:ident, dot, $block:expr, $out_ptr:expr) => {
        $crate::quant_primitive!(scalar, $qty, dot, $block, $out_ptr)
    };
}
