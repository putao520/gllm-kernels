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
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx2, f32, splat, 8.0);
                
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
                    let res_l_0 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_l_0, v_offset));
                    let res_l_1 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_l_1, v_offset));
                    let res_h_0 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_h_0, v_offset));
                    let res_h_1 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_h_1, v_offset));
                    
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
                    
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i*32 + 0), final_0);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i*32 + 8), final_1);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i*32 + 16), final_2);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i*32 + 24), final_3);
                }
             }
        }
    };

    (avx512, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "x86_64")]
             {
                use std::arch::x86_64::*;

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx512, f32, splat, 8.0);

                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;

                // Process 32 bytes at a time (64 nibbles = 64 f32 outputs)
                for i in 0..4 {
                    let v256 = _mm256_loadu_si256(qs_ptr.add(i*32) as *const _);

                    // Extract low and high nibbles
                    let mask = _mm256_set1_epi8(0x0F);
                    let low_bytes = _mm256_and_si256(v256, mask);
                    let high_bytes = _mm256_and_si256(_mm256_srli_epi16(v256, 4), mask);

                    // Convert low 16 bytes -> 16 i32 -> 16 f32
                    let low_128 = _mm256_castsi256_si128(low_bytes);
                    let ints_l = _mm512_cvtepu8_epi32(low_128);
                    let f_l = _mm512_cvtepi32_ps(ints_l);

                    // Convert high 16 bytes -> 16 i32 -> 16 f32
                    let high_128 = _mm256_castsi256_si128(high_bytes);
                    let ints_h = _mm512_cvtepu8_epi32(high_128);
                    let f_h = _mm512_cvtepi32_ps(ints_h);

                    // Dequantize: d * (val - 8.0)
                    let res_l = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, f_l, v_offset));
                    let res_h = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, f_h, v_offset));

                    // Interleave low and high: [l0,h0,l1,h1,...]
                    let res_lo = _mm512_unpacklo_ps(res_l, res_h);
                    let res_hi = _mm512_unpackhi_ps(res_l, res_h);

                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i*64 + 0), res_lo);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i*64 + 16), res_hi);

                    // Process upper 16 bytes of the 32-byte chunk
                    let low_128_hi = _mm256_extracti128_si256(low_bytes, 1);
                    let ints_l_hi = _mm512_cvtepu8_epi32(low_128_hi);
                    let f_l_hi = _mm512_cvtepi32_ps(ints_l_hi);

                    let high_128_hi = _mm256_extracti128_si256(high_bytes, 1);
                    let ints_h_hi = _mm512_cvtepu8_epi32(high_128_hi);
                    let f_h_hi = _mm512_cvtepi32_ps(ints_h_hi);

                    let res_l_hi = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, f_l_hi, v_offset));
                    let res_h_hi = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, f_h_hi, v_offset));

                    let res_lo_hi = _mm512_unpacklo_ps(res_l_hi, res_h_hi);
                    let res_hi_hi = _mm512_unpackhi_ps(res_l_hi, res_h_hi);

                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i*64 + 32), res_lo_hi);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i*64 + 48), res_hi_hi);
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
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_offset = $crate::simd_primitive!(neon, f32, splat, 8.0);

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
                    let res_l_0 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_l_0, v_offset), vd);
                    let res_l_1 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_l_1, v_offset), vd);
                    let res_l_2 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_l_2, v_offset), vd);
                    let res_l_3 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_l_3, v_offset), vd);

                    let res_h_0 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_h_0, v_offset), vd);
                    let res_h_1 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_h_1, v_offset), vd);
                    let res_h_2 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_h_2, v_offset), vd);
                    let res_h_3 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, f_h_3, v_offset), vd);
                    
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

    (neon, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_offset = $crate::simd_primitive!(neon, f32, splat, 8.0);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for i in 0..8 {
                    let v128 = vld1q_u8(qs_ptr.add(i * 16));
                    let mask = vdupq_n_u8(0x0F);
                    let low_bytes = vandq_u8(v128, mask);
                    let high_bytes = vshrq_n_u8(v128, 4);

                    // Expand low nibbles
                    let lo_u16_lo = vmovl_u8(vget_low_u8(low_bytes));
                    let lo_u16_hi = vmovl_high_u8(low_bytes);
                    let l_u32_0 = vmovl_u16(vget_low_u16(lo_u16_lo));
                    let l_u32_1 = vmovl_high_u16(lo_u16_lo);
                    let l_u32_2 = vmovl_u16(vget_low_u16(lo_u16_hi));
                    let l_u32_3 = vmovl_high_u16(lo_u16_hi);
                    let fl0 = vcvtq_f32_u32(l_u32_0);
                    let fl1 = vcvtq_f32_u32(l_u32_1);
                    let fl2 = vcvtq_f32_u32(l_u32_2);
                    let fl3 = vcvtq_f32_u32(l_u32_3);

                    // Expand high nibbles
                    let hi_u16_lo = vmovl_u8(vget_low_u8(high_bytes));
                    let hi_u16_hi = vmovl_high_u8(high_bytes);
                    let h_u32_0 = vmovl_u16(vget_low_u16(hi_u16_lo));
                    let h_u32_1 = vmovl_high_u16(hi_u16_lo);
                    let h_u32_2 = vmovl_u16(vget_low_u16(hi_u16_hi));
                    let h_u32_3 = vmovl_high_u16(hi_u16_hi);
                    let fh0 = vcvtq_f32_u32(h_u32_0);
                    let fh1 = vcvtq_f32_u32(h_u32_1);
                    let fh2 = vcvtq_f32_u32(h_u32_2);
                    let fh3 = vcvtq_f32_u32(h_u32_3);

                    // Dequant: d * (val - 8.0)
                    let dl0 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fl0, v_offset), vd);
                    let dh0 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fh0, v_offset), vd);
                    let dl1 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fl1, v_offset), vd);
                    let dh1 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fh1, v_offset), vd);
                    let dl2 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fl2, v_offset), vd);
                    let dh2 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fh2, v_offset), vd);
                    let dl3 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fl3, v_offset), vd);
                    let dh3 = $crate::simd_primitive!(neon, f32, mul, $crate::simd_primitive!(neon, f32, sub, fh3, v_offset), vd);

                    // Interleaved dot: [l0,h0,l1,h1,...] matches memory layout
                    // Use vzip to interleave, then FMA with other
                    let z0 = vzipq_f32(dl0, dh0); // .0=[l0,h0,l1,h1], .1=[l2,h2,l3,h3]
                    let z1 = vzipq_f32(dl1, dh1);
                    let z2 = vzipq_f32(dl2, dh2);
                    let z3 = vzipq_f32(dl3, dh3);

                    let base = i * 32;
                    acc = $crate::simd_primitive!(neon, f32, fma, z0.0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z0.1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z1.0, $crate::simd_primitive!(neon, f32, load, other.add(base + 8)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z1.1, $crate::simd_primitive!(neon, f32, load, other.add(base + 12)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z2.0, $crate::simd_primitive!(neon, f32, load, other.add(base + 16)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z2.1, $crate::simd_primitive!(neon, f32, load, other.add(base + 20)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z3.0, $crate::simd_primitive!(neon, f32, load, other.add(base + 24)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, z3.1, $crate::simd_primitive!(neon, f32, load, other.add(base + 28)), acc);
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
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

    (neon, q8_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let out_ptr = $out_ptr;

                // Process 256 i8 values, 16 at a time (4 NEON vectors per iteration)
                for i in 0..16 {
                    // Load 16 i8 values
                    let v_i8 = vld1q_s8(qs_ptr.add(i * 16));

                    // Expand i8 -> i16 -> i32 -> f32
                    let v_i16_lo = vmovl_s8(vget_low_s8(v_i8));
                    let v_i16_hi = vmovl_high_s8(v_i8);

                    let v_i32_0 = vmovl_s16(vget_low_s16(v_i16_lo));
                    let v_i32_1 = vmovl_high_s16(v_i16_lo);
                    let v_i32_2 = vmovl_s16(vget_low_s16(v_i16_hi));
                    let v_i32_3 = vmovl_high_s16(v_i16_hi);

                    let v_f32_0 = vcvtq_f32_s32(v_i32_0);
                    let v_f32_1 = vcvtq_f32_s32(v_i32_1);
                    let v_f32_2 = vcvtq_f32_s32(v_i32_2);
                    let v_f32_3 = vcvtq_f32_s32(v_i32_3);

                    // Multiply by d
                    let res_0 = $crate::simd_primitive!(neon, f32, mul, vd, v_f32_0);
                    let res_1 = $crate::simd_primitive!(neon, f32, mul, vd, v_f32_1);
                    let res_2 = $crate::simd_primitive!(neon, f32, mul, vd, v_f32_2);
                    let res_3 = $crate::simd_primitive!(neon, f32, mul, vd, v_f32_3);

                    // Store
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 0), res_0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 4), res_1);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 8), res_2);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 12), res_3);
                }
            }
        }
    };

    (neon, q8_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                // 256 i8 values, 16 at a time
                for i in 0..16 {
                    let v_i8 = vld1q_s8(qs_ptr.add(i * 16));

                    let v_i16_lo = vmovl_s8(vget_low_s8(v_i8));
                    let v_i16_hi = vmovl_high_s8(v_i8);

                    let v_i32_0 = vmovl_s16(vget_low_s16(v_i16_lo));
                    let v_i32_1 = vmovl_high_s16(v_i16_lo);
                    let v_i32_2 = vmovl_s16(vget_low_s16(v_i16_hi));
                    let v_i32_3 = vmovl_high_s16(v_i16_hi);

                    let vf0 = vcvtq_f32_s32(v_i32_0);
                    let vf1 = vcvtq_f32_s32(v_i32_1);
                    let vf2 = vcvtq_f32_s32(v_i32_2);
                    let vf3 = vcvtq_f32_s32(v_i32_3);

                    let dq0 = $crate::simd_primitive!(neon, f32, mul, vd, vf0);
                    let dq1 = $crate::simd_primitive!(neon, f32, mul, vd, vf1);
                    let dq2 = $crate::simd_primitive!(neon, f32, mul, vd, vf2);
                    let dq3 = $crate::simd_primitive!(neon, f32, mul, vd, vf3);

                    let base = i * 16;
                    acc = $crate::simd_primitive!(neon, f32, fma, dq0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, dq1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, dq2, $crate::simd_primitive!(neon, f32, load, other.add(base + 8)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, dq3, $crate::simd_primitive!(neon, f32, load, other.add(base + 12)), acc);
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (avx512, q8_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let out_ptr = $out_ptr;

                // Process 16 i8 values at a time
                for i in 0..16 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i*16) as *const _);
                    let ints = _mm512_cvtepi8_epi32(v128);
                    let vf = _mm512_cvtepi32_ps(ints);
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i*16), res);
                }
            }
        }
    };

    (avx512, q8_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for i in 0..16 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i*16) as *const _);
                    let ints = _mm512_cvtepi8_epi32(v128);
                    let vf = _mm512_cvtepi32_ps(ints);
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(i*16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
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

    (avx512, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "x86_64")]
             {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx512, f32, splat, 8.0);

                let qs_ptr = block.qs.as_ptr();
                let other_ptr = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                // Process 16 bytes at a time (32 nibbles = 32 values)
                for i in 0..8 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i*16) as *const _);

                    // Extract low and high nibbles
                    let mask = _mm_set1_epi8(0x0F);
                    let low_bytes = _mm_and_si128(v128, mask);
                    let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);

                    // Convert 16 low nibbles -> 16 i32 -> 16 f32
                    let ints_l = _mm512_cvtepu8_epi32(low_bytes);
                    let f_l = _mm512_cvtepi32_ps(ints_l);

                    // Convert 16 high nibbles -> 16 i32 -> 16 f32
                    let ints_h = _mm512_cvtepu8_epi32(high_bytes);
                    let f_h = _mm512_cvtepi32_ps(ints_h);

                    // Dequantize: d * (val - 8.0)
                    let dq_l = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, f_l, v_offset));
                    let dq_h = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, f_h, v_offset));

                    // Interleave low and high
                    let res_lo = _mm512_unpacklo_ps(dq_l, dq_h);
                    let res_hi = _mm512_unpackhi_ps(dq_l, dq_h);

                    // Load other values and accumulate
                    let other_lo = $crate::simd_primitive!(avx512, f32, load, other_ptr.add(i*32 + 0));
                    let other_hi = $crate::simd_primitive!(avx512, f32, load, other_ptr.add(i*32 + 16));

                    acc = $crate::simd_primitive!(avx512, f32, fma, res_lo, other_lo, acc);
                    acc = $crate::simd_primitive!(avx512, f32, fma, res_hi, other_hi, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
             }
        }
    };

    (avx2, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "x86_64")]
             {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx2, f32, splat, 8.0);

                let qs_ptr = block.qs.as_ptr();
                let other_ptr = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);

                for i in 0..8 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i*16) as *const _);
                    let mask = _mm_set1_epi8(0x0F);
                    let low_bytes = _mm_and_si128(v128, mask);
                    let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);

                    let f_l_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_bytes));
                    let f_l_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8)));
                    let f_h_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_bytes));
                    let f_h_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8)));

                    let dq_l_0 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_l_0, v_offset));
                    let dq_l_1 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_l_1, v_offset));
                    let dq_h_0 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_h_0, v_offset));
                    let dq_h_1 = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, f_h_1, v_offset));

                    let out_0 = _mm256_unpacklo_ps(dq_l_0, dq_h_0);
                    let out_1 = _mm256_unpackhi_ps(dq_l_0, dq_h_0);
                    let final_0 = _mm256_permute2f128_ps(out_0, out_1, 0x20);
                    let final_1 = _mm256_permute2f128_ps(out_0, out_1, 0x31);

                    let out_2 = _mm256_unpacklo_ps(dq_l_1, dq_h_1);
                    let out_3 = _mm256_unpackhi_ps(dq_l_1, dq_h_1);
                    let final_2 = _mm256_permute2f128_ps(out_2, out_3, 0x20);
                    let final_3 = _mm256_permute2f128_ps(out_2, out_3, 0x31);

                    let other_0 = $crate::simd_primitive!(avx2, f32, load, other_ptr.add(i*32 + 0));
                    let other_1 = $crate::simd_primitive!(avx2, f32, load, other_ptr.add(i*32 + 8));
                    let other_2 = $crate::simd_primitive!(avx2, f32, load, other_ptr.add(i*32 + 16));
                    let other_3 = $crate::simd_primitive!(avx2, f32, load, other_ptr.add(i*32 + 24));

                    acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, final_0, other_0));
                    acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, final_1, other_1));
                    acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, final_2, other_2));
                    acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, final_3, other_3));
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
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx2, f32, splat, dmin);
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
                        let r_lo = $crate::simd_primitive!(avx2, f32, sub, $crate::simd_primitive!(avx2, f32, mul, vd, lo), v_dmin);
                        let r_hi = $crate::simd_primitive!(avx2, f32, sub, $crate::simd_primitive!(avx2, f32, mul, vd, hi), v_dmin);
                        let base = i * 64 + j * 16;
                        $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), r_lo);
                        $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 8), r_hi);
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
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx2, f32, splat, dmin);
                let mask2 = _mm_set1_epi8(0x03);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..4 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let v0 = _mm_and_si128(v128, mask2);
                    let v1 = _mm_and_si128(_mm_srli_epi16(v128, 2), mask2);
                    let v2 = _mm_and_si128(_mm_srli_epi16(v128, 4), mask2);
                    let v3 = _mm_and_si128(_mm_srli_epi16(v128, 6), mask2);
                    for (j, vv) in [(0usize, v0), (1, v1), (2, v2), (3, v3)] {
                        let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vv));
                        let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vv, 8)));
                        let dq_lo = $crate::simd_primitive!(avx2, f32, sub, $crate::simd_primitive!(avx2, f32, mul, vd, lo), v_dmin);
                        let dq_hi = $crate::simd_primitive!(avx2, f32, sub, $crate::simd_primitive!(avx2, f32, mul, vd, hi), v_dmin);
                        let base = i * 64 + j * 16;
                        let o_lo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                        let o_hi = $crate::simd_primitive!(avx2, f32, load, other.add(base + 8));
                        acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, dq_lo, o_lo));
                        acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, dq_hi, o_hi));
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

    (avx512, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx512, f32, splat, dmin);
                let mask2 = _mm_set1_epi8(0x03);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;

                // Process 16 bytes at a time = 64 values
                for i in 0..4 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let v0 = _mm_and_si128(v128, mask2);
                    let v1 = _mm_and_si128(_mm_srli_epi16(v128, 2), mask2);
                    let v2 = _mm_and_si128(_mm_srli_epi16(v128, 4), mask2);
                    let v3 = _mm_and_si128(_mm_srli_epi16(v128, 6), mask2);

                    for (j, vv) in [(0usize, v0), (1, v1), (2, v2), (3, v3)] {
                        let ints = _mm512_cvtepu8_epi32(vv);
                        let vf = _mm512_cvtepi32_ps(ints);
                        let res = $crate::simd_primitive!(avx512, f32, sub, $crate::simd_primitive!(avx512, f32, mul, vd, vf), v_dmin);
                        let base = i * 64 + j * 16;
                        $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
                    }
                }
            }
        }
    };

    (avx512, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx512, f32, splat, dmin);
                let mask2 = _mm_set1_epi8(0x03);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for i in 0..4 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let v0 = _mm_and_si128(v128, mask2);
                    let v1 = _mm_and_si128(_mm_srli_epi16(v128, 2), mask2);
                    let v2 = _mm_and_si128(_mm_srli_epi16(v128, 4), mask2);
                    let v3 = _mm_and_si128(_mm_srli_epi16(v128, 6), mask2);

                    for (j, vv) in [(0usize, v0), (1, v1), (2, v2), (3, v3)] {
                        let ints = _mm512_cvtepu8_epi32(vv);
                        let vf = _mm512_cvtepi32_ps(ints);
                        let dq = $crate::simd_primitive!(avx512, f32, sub, $crate::simd_primitive!(avx512, f32, mul, vd, vf), v_dmin);
                        let base = i * 64 + j * 16;
                        let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                        acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                    }
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q3_K
    // ========================================================================
    (avx2, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx2, f32, splat, 4.0);
                let out_ptr = $out_ptr;
                // Q3_K: 256 values. qs has 64 bytes (2 bits per value in low part),
                // hmask has 32 bytes (1 high bit per value).
                // val = ((hmask_bit << 2) | qs_2bits) - 4
                // Process 8 values at a time
                for i in 0..32 {
                    // Load 8 values: extract 2-bit from qs, 1-bit from hmask
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let low = (block.qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                        let high = (block.hmask[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 2) | low) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, vf, v_offset));
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx2, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx2, f32, splat, 4.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..32 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let low = (block.qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                        let high = (block.hmask[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 2) | low) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, vf, v_offset));
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx512, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx512, f32, splat, 4.0);
                let out_ptr = $out_ptr;

                // Process 16 values at a time
                for i in 0..16 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let low = (block.qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                        let high = (block.hmask[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 2) | low) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, vf, v_offset));
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx512, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx512, f32, splat, 4.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for i in 0..16 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let low = (block.qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                        let high = (block.hmask[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 2) | low) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, vf, v_offset));
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q5_K
    // ========================================================================
    (avx2, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx2, f32, splat, dmin);
                let out_ptr = $out_ptr;
                // Q5_K: 256 values. qs[128] has low 4 bits, qh[32] has high 1 bit.
                // val = (high << 4) | low4, out = d * val - dmin
                for i in 0..32 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx2, f32, sub, $crate::simd_primitive!(avx2, f32, mul, vd, vf), v_dmin);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx2, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx2, f32, splat, dmin);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..32 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx2, f32, sub, $crate::simd_primitive!(avx2, f32, mul, vd, vf), v_dmin);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx512, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx512, f32, splat, dmin);
                let out_ptr = $out_ptr;

                for i in 0..16 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx512, f32, sub, $crate::simd_primitive!(avx512, f32, mul, vd, vf), v_dmin);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx512, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d.to_f32();
                let dmin = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(avx512, f32, splat, dmin);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for i in 0..16 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx512, f32, sub, $crate::simd_primitive!(avx512, f32, mul, vd, vf), v_dmin);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q6_K
    // ========================================================================
    (avx2, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx2, f32, splat, 32.0);
                let out_ptr = $out_ptr;
                // Q6_K: 256 values. qs[128] has low 4 bits, qh[64] has high 2 bits.
                // val = (high << 4) | low4, out = d * (val - 32)
                for i in 0..32 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, vf, v_offset));
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx2, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx2, f32, splat, 32.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..32 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, $crate::simd_primitive!(avx2, f32, sub, vf, v_offset));
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx512, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx512, f32, splat, 32.0);
                let out_ptr = $out_ptr;

                for i in 0..16 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, vf, v_offset));
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx512, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_offset = $crate::simd_primitive!(avx512, f32, splat, 32.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for i in 0..16 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, $crate::simd_primitive!(avx512, f32, sub, vf, v_offset));
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q2_K
    // ========================================================================
    (neon, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(neon, f32, splat, dmin);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let mask2 = vdupq_n_u8(0x03);

                // 64 bytes qs, 4 values per byte = 256 values
                // Process 16 bytes at a time = 64 values, 4 iterations
                for i in 0..4 {
                    let v128 = vld1q_u8(qs_ptr.add(i * 16));
                    let v0 = vandq_u8(v128, mask2);
                    let v1 = vandq_u8(vshrq_n_u8(v128, 2), mask2);
                    let v2 = vandq_u8(vshrq_n_u8(v128, 4), mask2);
                    let v3 = vandq_u8(vshrq_n_u8(v128, 6), mask2);

                    // For each shift variant, expand 16 u8 -> 4x4 f32 and store
                    let shifts = [v0, v1, v2, v3];
                    for j in 0..4usize {
                        let vv = shifts[j];
                        // u8 -> u16 -> u32 -> f32
                        let lo_u16 = vmovl_u8(vget_low_u8(vv));
                        let hi_u16 = vmovl_high_u8(vv);

                        let u32_0 = vmovl_u16(vget_low_u16(lo_u16));
                        let u32_1 = vmovl_high_u16(lo_u16);
                        let u32_2 = vmovl_u16(vget_low_u16(hi_u16));
                        let u32_3 = vmovl_high_u16(hi_u16);

                        let f0 = vcvtq_f32_u32(u32_0);
                        let f1 = vcvtq_f32_u32(u32_1);
                        let f2 = vcvtq_f32_u32(u32_2);
                        let f3 = vcvtq_f32_u32(u32_3);

                        // d * val - dmin
                        let r0 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f0), v_dmin);
                        let r1 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f1), v_dmin);
                        let r2 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f2), v_dmin);
                        let r3 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f3), v_dmin);

                        let base = i * 64 + j * 16;
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), r0);
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 4), r1);
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 8), r2);
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 12), r3);
                    }
                }
            }
        }
    };

    (neon, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(neon, f32, splat, dmin);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mask2 = vdupq_n_u8(0x03);
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for i in 0..4 {
                    let v128 = vld1q_u8(qs_ptr.add(i * 16));
                    let v0 = vandq_u8(v128, mask2);
                    let v1 = vandq_u8(vshrq_n_u8(v128, 2), mask2);
                    let v2 = vandq_u8(vshrq_n_u8(v128, 4), mask2);
                    let v3 = vandq_u8(vshrq_n_u8(v128, 6), mask2);

                    let shifts = [v0, v1, v2, v3];
                    for j in 0..4usize {
                        let vv = shifts[j];
                        let lo_u16 = vmovl_u8(vget_low_u8(vv));
                        let hi_u16 = vmovl_high_u8(vv);

                        let u32_0 = vmovl_u16(vget_low_u16(lo_u16));
                        let u32_1 = vmovl_high_u16(lo_u16);
                        let u32_2 = vmovl_u16(vget_low_u16(hi_u16));
                        let u32_3 = vmovl_high_u16(hi_u16);

                        let f0 = vcvtq_f32_u32(u32_0);
                        let f1 = vcvtq_f32_u32(u32_1);
                        let f2 = vcvtq_f32_u32(u32_2);
                        let f3 = vcvtq_f32_u32(u32_3);

                        let dq0 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f0), v_dmin);
                        let dq1 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f1), v_dmin);
                        let dq2 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f2), v_dmin);
                        let dq3 = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, f3), v_dmin);

                        let base = i * 64 + j * 16;
                        let o0 = $crate::simd_primitive!(neon, f32, load, other.add(base));
                        let o1 = $crate::simd_primitive!(neon, f32, load, other.add(base + 4));
                        let o2 = $crate::simd_primitive!(neon, f32, load, other.add(base + 8));
                        let o3 = $crate::simd_primitive!(neon, f32, load, other.add(base + 12));

                        acc = $crate::simd_primitive!(neon, f32, fma, dq0, o0, acc);
                        acc = $crate::simd_primitive!(neon, f32, fma, dq1, o1, acc);
                        acc = $crate::simd_primitive!(neon, f32, fma, dq2, o2, acc);
                        acc = $crate::simd_primitive!(neon, f32, fma, dq3, o3, acc);
                    }
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q3_K
    // ========================================================================
    (neon, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_offset = $crate::simd_primitive!(neon, f32, splat, 4.0);
                let out_ptr = $out_ptr;

                // Q3_K: 256 values. qs[64] has 2 low bits, hmask[32] has 1 high bit.
                // val = ((hmask_bit << 2) | qs_2bits) - 4
                // Process 4 values at a time via NEON
                for i in 0..64 {
                    let base = i * 4;
                    let mut vals = [0i32; 4];
                    for j in 0..4 {
                        let idx = base + j;
                        let low = (block.qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                        let high = (block.hmask[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 2) | low) as i32;
                    }
                    let vi = vld1q_s32(vals.as_ptr());
                    let vf = vcvtq_f32_s32(vi);
                    let res = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, sub, vf, v_offset));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (neon, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_offset = $crate::simd_primitive!(neon, f32, splat, 4.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for i in 0..64 {
                    let base = i * 4;
                    let mut vals = [0i32; 4];
                    for j in 0..4 {
                        let idx = base + j;
                        let low = (block.qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                        let high = (block.hmask[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 2) | low) as i32;
                    }
                    let vi = vld1q_s32(vals.as_ptr());
                    let vf = vcvtq_f32_s32(vi);
                    let dq = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, sub, vf, v_offset));
                    let vo = $crate::simd_primitive!(neon, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(neon, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q5_K
    // ========================================================================
    (neon, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(neon, f32, splat, dmin);
                let out_ptr = $out_ptr;

                // Q5_K: 256 values. qs[128] has low 4 bits, qh[32] has high 1 bit.
                // val = (high << 4) | low4, out = d * val - dmin
                for i in 0..64 {
                    let base = i * 4;
                    let mut vals = [0i32; 4];
                    for j in 0..4 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = vld1q_s32(vals.as_ptr());
                    let vf = vcvtq_f32_s32(vi);
                    let res = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, vf), v_dmin);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (neon, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let dmin: f32 = block.dmin.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_dmin = $crate::simd_primitive!(neon, f32, splat, dmin);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for i in 0..64 {
                    let base = i * 4;
                    let mut vals = [0i32; 4];
                    for j in 0..4 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = vld1q_s32(vals.as_ptr());
                    let vf = vcvtq_f32_s32(vi);
                    let dq = $crate::simd_primitive!(neon, f32, sub, $crate::simd_primitive!(neon, f32, mul, vd, vf), v_dmin);
                    let vo = $crate::simd_primitive!(neon, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(neon, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q6_K
    // ========================================================================
    (neon, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_offset = $crate::simd_primitive!(neon, f32, splat, 32.0);
                let out_ptr = $out_ptr;

                // Q6_K: 256 values. qs[128] has low 4 bits, qh[64] has high 2 bits.
                // val = (high << 4) | low4, out = d * (val - 32)
                for i in 0..64 {
                    let base = i * 4;
                    let mut vals = [0i32; 4];
                    for j in 0..4 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = vld1q_s32(vals.as_ptr());
                    let vf = vcvtq_f32_s32(vi);
                    let res = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, sub, vf, v_offset));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (neon, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d = block.d;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_offset = $crate::simd_primitive!(neon, f32, splat, 32.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for i in 0..64 {
                    let base = i * 4;
                    let mut vals = [0i32; 4];
                    for j in 0..4 {
                        let idx = base + j;
                        let low = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let high = (block.qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                        vals[j] = ((high << 4) | low) as i32;
                    }
                    let vi = vld1q_s32(vals.as_ptr());
                    let vf = vcvtq_f32_s32(vi);
                    let dq = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, sub, vf, v_offset));
                    let vo = $crate::simd_primitive!(neon, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(neon, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
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
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                // 256 i8 values. Process 8 at a time = 32 iterations.
                for i in 0..32 {
                    // Load 8 i8 -> i32 -> f32
                    let v64 = _mm_loadl_epi64(qs_ptr.add(i * 8) as *const _);
                    let vi32 = _mm256_cvtepi8_epi32(v64);
                    let vf = _mm256_cvtepi32_ps(vi32);
                    let res = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 8), res);
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
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..32 {
                    let v64 = _mm_loadl_epi64(qs_ptr.add(i * 8) as *const _);
                    let vi32 = _mm256_cvtepi8_epi32(v64);
                    let vf = _mm256_cvtepi32_ps(vi32);
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(i * 8));
                    acc = $crate::simd_primitive!(avx2, f32, add, acc, $crate::simd_primitive!(avx2, f32, mul, dq, vo));
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
    // NEON IQ dot implementations (scalar decode + NEON FMA accumulation)
    // ========================================================================

    (neon, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 32];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..16 {
                        let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                        vals[16 + j] = v as f32;
                    }
                    let base = group * 32;
                    for c in 0..8 {
                        let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(c * 4)));
                        let vo = $crate::simd_primitive!(neon, f32, load, other.add(base + c * 4));
                        acc = $crate::simd_primitive!(neon, f32, fma, vf, vo, acc);
                    }
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, scale);
                    let mut vals = [0.0f32; 32];
                    for rep in 0..4 {
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                            vals[rep * 8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                    }
                    let base = group * 32;
                    for c in 0..8 {
                        let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(c * 4)));
                        let vo = $crate::simd_primitive!(neon, f32, load, other.add(base + c * 4));
                        acc = $crate::simd_primitive!(neon, f32, fma, vf, vo, acc);
                    }
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                for group in 0..32 {
                    let idx = (qs[group] & 0xFF) as usize;
                    let grid_val = crate::codebooks::IQ2XXS_GRID[idx];
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, vf1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..32 {
                    let idx = (qs[group] & 0x1FF) as usize;
                    let grid_val = crate::codebooks::IQ2XS_GRID[idx.min(511)];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, vf1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..32 {
                    let qs_idx = group * 2;
                    let qh_idx = group / 4;
                    let qh_shift = (group % 4) * 2;
                    let idx = ((qs[qs_idx] as usize) |
                              ((qs[qs_idx + 1] as usize) << 8) |
                              (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                    let grid_val = crate::codebooks::IQ2S_GRID[idx];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, vf1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                for group in 0..64 {
                    let idx = qs[group] as usize;
                    let grid_val = crate::codebooks::IQ3XXS_GRID[idx];
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 4;
                    let vf = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..64 {
                    let qh_idx = group / 8;
                    let qh_shift = group % 8;
                    let idx = ((qs[group] as usize) |
                              (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                    let grid_val = crate::codebooks::IQ3S_GRID[idx];
                    let scale_idx = group / 16;
                    let scale = scales[scale_idx] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_idx = group / 2;
                        let sign_shift = (group % 2) * 4 + j;
                        let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let base = group * 4;
                    let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                // Load 16-entry codebook into NEON register for vqtbl1q lookup
                let lut = vld1q_s8(crate::codebooks::KVALUES_IQ4NL.as_ptr());
                let mask_lo = vdupq_n_u8(0x0F);

                // Process 16 bytes (32 values) at a time — IQ4_NL block_size=32, qs has 16 bytes
                let raw = vld1q_u8(qs.as_ptr());
                let lo_idx = vandq_u8(raw, mask_lo);
                let hi_idx = vshrq_n_u8::<4>(raw);

                // Table lookup: 16 parallel codebook lookups each
                let lo_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(lo_idx)));
                let hi_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(hi_idx)));

                // Interleave lo/hi to get correct order: [lo0, hi0, lo1, hi1, ...]
                let interleaved_lo = vzip1q_s8(lo_vals, hi_vals);
                let interleaved_hi = vzip2q_s8(lo_vals, hi_vals);

                // Convert i8 → f32 and FMA, 4 elements at a time
                // First 16 values (interleaved_lo)
                let lo8 = vget_low_s8(interleaved_lo);
                let hi8 = vget_high_s8(interleaved_lo);
                let lo16_0 = vmovl_s8(lo8);
                let lo16_1 = vmovl_s8(hi8);

                // 4 groups of 4 from interleaved_lo
                let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0)));
                let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0)));
                let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1)));
                let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1)));

                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f0), $crate::simd_primitive!(neon, f32, load, other), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f1), $crate::simd_primitive!(neon, f32, load, other.add(4)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f2), $crate::simd_primitive!(neon, f32, load, other.add(8)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f3), $crate::simd_primitive!(neon, f32, load, other.add(12)), acc);

                // Next 16 values (interleaved_hi)
                let lo8b = vget_low_s8(interleaved_hi);
                let hi8b = vget_high_s8(interleaved_hi);
                let lo16_2 = vmovl_s8(lo8b);
                let lo16_3 = vmovl_s8(hi8b);

                let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_2)));
                let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_2)));
                let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_3)));
                let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_3)));

                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f4), $crate::simd_primitive!(neon, f32, load, other.add(16)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f5), $crate::simd_primitive!(neon, f32, load, other.add(20)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f6), $crate::simd_primitive!(neon, f32, load, other.add(24)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f7), $crate::simd_primitive!(neon, f32, load, other.add(28)), acc);

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc0 = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let mut acc1 = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                // Load 16-entry codebook into NEON register
                let lut = vld1q_s8(crate::codebooks::KVALUES_IQ4NL.as_ptr());
                let mask_lo = vdupq_n_u8(0x0F);

                // IQ4_XS block_size=256, qs has 128 bytes → 16 groups of 8 bytes
                for i in 0..8 {
                    // Load 16 packed bytes → 32 values
                    let raw = vld1q_u8(qs.as_ptr().add(i * 16));
                    let lo_idx = vandq_u8(raw, mask_lo);
                    let hi_idx = vshrq_n_u8::<4>(raw);

                    let lo_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(lo_idx)));
                    let hi_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(hi_idx)));

                    let interleaved_lo = vzip1q_s8(lo_vals, hi_vals);
                    let interleaved_hi = vzip2q_s8(lo_vals, hi_vals);

                    let base = i * 32;

                    // interleaved_lo → 16 f32 values
                    let lo8 = vget_low_s8(interleaved_lo);
                    let hi8 = vget_high_s8(interleaved_lo);
                    let s16_0 = vmovl_s8(lo8);
                    let s16_1 = vmovl_s8(hi8);

                    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_0)));
                    let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_0)));
                    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_1)));
                    let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_1)));

                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f0), $crate::simd_primitive!(neon, f32, load, other.add(base)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f1), $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc1);
                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f2), $crate::simd_primitive!(neon, f32, load, other.add(base + 8)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f3), $crate::simd_primitive!(neon, f32, load, other.add(base + 12)), acc1);

                    // interleaved_hi → 16 f32 values
                    let lo8b = vget_low_s8(interleaved_hi);
                    let hi8b = vget_high_s8(interleaved_hi);
                    let s16_2 = vmovl_s8(lo8b);
                    let s16_3 = vmovl_s8(hi8b);

                    let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_2)));
                    let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_2)));
                    let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_3)));
                    let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_3)));

                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f4), $crate::simd_primitive!(neon, f32, load, other.add(base + 16)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f5), $crate::simd_primitive!(neon, f32, load, other.add(base + 20)), acc1);
                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f6), $crate::simd_primitive!(neon, f32, load, other.add(base + 24)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f7), $crate::simd_primitive!(neon, f32, load, other.add(base + 28)), acc1);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, $crate::simd_primitive!(neon, f32, add, acc0, acc1))
            }
        }
    };

    // ========================================================================
    // AVX2 IQ4_NL decode/dot (16-entry codebook, vpshufb lookup)
    // ========================================================================
    (avx2, iq4_nl, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let mask_lo = _mm_set1_epi8(0x0F);
                // Process 16 bytes at a time (32 values)
                for i in 0..8 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo_bytes = _mm_and_si128(v128, mask_lo);
                    let hi_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask_lo);
                    // Convert nibble indices to codebook values via scalar lookup
                    // (vpshufb only works for byte->byte, we need float output)
                    for chunk in 0..2 {
                        let bytes = if chunk == 0 { lo_bytes } else { hi_bytes };
                        // Extract 16 bytes, lookup codebook, convert to f32
                        let mut indices = [0u8; 16];
                        _mm_storeu_si128(indices.as_mut_ptr() as *mut _, bytes);
                        for g in 0..2 {
                            let mut vals = [0.0f32; 8];
                            for j in 0..8 {
                                vals[j] = crate::codebooks::KVALUES_IQ4NL[indices[g * 8 + j] as usize] as f32;
                            }
                            let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                            let res = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                            let out_idx = i * 32 + chunk * 16 + g * 8;
                            $crate::simd_primitive!(avx2, f32, store, out_ptr.add(out_idx), res);
                        }
                    }
                }
            }
        }
    };

    (avx2, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mask_lo = _mm_set1_epi8(0x0F);
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..8 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo_bytes = _mm_and_si128(v128, mask_lo);
                    let hi_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask_lo);
                    for chunk in 0..2 {
                        let bytes = if chunk == 0 { lo_bytes } else { hi_bytes };
                        let mut indices = [0u8; 16];
                        _mm_storeu_si128(indices.as_mut_ptr() as *mut _, bytes);
                        for g in 0..2 {
                            let mut vals = [0.0f32; 8];
                            for j in 0..8 {
                                vals[j] = crate::codebooks::KVALUES_IQ4NL[indices[g * 8 + j] as usize] as f32;
                            }
                            let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                            let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                            let out_idx = i * 32 + chunk * 16 + g * 8;
                            let vo = $crate::simd_primitive!(avx2, f32, load, other.add(out_idx));
                            acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                        }
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
    // AVX2 IQ4_XS decode/dot (same codebook as IQ4_NL, block_size=256)
    // ========================================================================
    (avx2, iq4_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                // 128 bytes qs, 2 nibbles each = 256 values
                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[(b >> 4) as usize] as f32;
                    }
                    let v0 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let v1 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr().add(8));
                    let r0 = $crate::simd_primitive!(avx2, f32, mul, vd, v0);
                    let r1 = $crate::simd_primitive!(avx2, f32, mul, vd, v1);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16), r0);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16 + 8), r1);
                }
            }
        }
    };

    (avx2, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[(b >> 4) as usize] as f32;
                    }
                    let v0 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let v1 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr().add(8));
                    let dq0 = $crate::simd_primitive!(avx2, f32, mul, vd, v0);
                    let dq1 = $crate::simd_primitive!(avx2, f32, mul, vd, v1);
                    let o0 = $crate::simd_primitive!(avx2, f32, load, other.add(i * 16));
                    let o1 = $crate::simd_primitive!(avx2, f32, load, other.add(i * 16 + 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq0, o0, acc);
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq1, o1, acc);
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
    // AVX2 IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S dot (vpgatherdd codebook)
    // ========================================================================
    // For grid-based IQ formats, the bit extraction is complex but the FMA
    // accumulation benefits from AVX2. We extract values to a temp buffer
    // then do vectorized FMA.

    (avx2, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // 32 groups of 8 values
                for group in 0..32 {
                    let idx = (qs[group] & 0xFF) as usize;
                    let grid_val = crate::codebooks::IQ2XXS_GRID[idx];
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for group in 0..32 {
                    let idx = (qs[group] & 0x1FF) as usize;
                    let grid_val = crate::codebooks::IQ2XS_GRID[idx.min(511)];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // 256 values in groups of 32 (8 groups)
                for group in 0..8 {
                    let scale = scales[group / 2] as f32 / 15.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    // Process 32 values in 4 chunks of 8
                    for chunk in 0..4 {
                        let base = group * 32 + chunk * 8;
                        let mut vals = [0.0f32; 8];
                        for j in 0..8 {
                            let idx = base + j;
                            let low = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                            let high = (qh[idx / 8] >> (idx % 8)) & 1;
                            let val = (high << 2) | low;
                            let sign_byte = signs[idx / 8];
                            let sign_bit = (sign_byte >> (idx % 8)) & 1;
                            let signed_val = if sign_bit == 0 { val as f32 } else { -(val as f32) };
                            vals[j] = signed_val;
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                        acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
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

    (avx2, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // IQ3_XXS: 3-bit packed in qs, 256 values
                for i in 0..32 {
                    let base = i * 8;
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // 256 values in groups of 32 (8 groups)
                for group in 0..8 {
                    let scale = scales[group] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    for chunk in 0..4 {
                        let base = group * 32 + chunk * 8;
                        let mut vals = [0.0f32; 8];
                        for j in 0..8 {
                            let idx = base + j;
                            let q = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                            let h = (qh[idx / 8] >> (idx % 8)) & 1;
                            let val = ((h << 2) | q) as f32 - 4.0;
                            vals[j] = val;
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                        acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
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

    (avx2, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    // First 8 values
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                    // Second 8 values
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32 + 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                    // Remaining 16 values (2 chunks of 8)
                    for chunk in 0..2 {
                        for j in 0..8 {
                            let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                            vals[j] = v as f32;
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32 + 16 + chunk * 8));
                        acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
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

    (avx2, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, scale);
                    for rep in 0..4 {
                        let mut vals = [0.0f32; 8];
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                            vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32 + rep * 8));
                        acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
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
}
