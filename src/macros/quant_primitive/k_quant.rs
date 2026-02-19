/// K-Quant quantization primitives (Q2_K through Q8_K).
///
/// Sub-macro of the Layer 3 quant_primitive dispatcher.
/// Covers all ISA variants: scalar, avx2, avx512, neon.
#[macro_export]
macro_rules! quant_primitive_kquant {
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
                     let neg_d8 = -d * 8.0;
                     *$out_ptr.add(i*2) = (l as f32).mul_add(d, neg_d8);
                     *$out_ptr.add(i*2+1) = (h as f32).mul_add(d, neg_d8);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 8.0);

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

                    // Dequantize: d * val + (-d * 8.0) via FMA
                    let res_l_0 = $crate::simd_primitive!(avx2, f32, fma, vd, f_l_0, v_neg_d_offset);
                    let res_l_1 = $crate::simd_primitive!(avx2, f32, fma, vd, f_l_1, v_neg_d_offset);
                    let res_h_0 = $crate::simd_primitive!(avx2, f32, fma, vd, f_h_0, v_neg_d_offset);
                    let res_h_1 = $crate::simd_primitive!(avx2, f32, fma, vd, f_h_1, v_neg_d_offset);
                    
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
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 8.0);

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

                    // Dequantize: d * val + (-d * 8.0) via FMA
                    let res_l = $crate::simd_primitive!(avx512, f32, fma, vd, f_l, v_neg_d_offset);
                    let res_h = $crate::simd_primitive!(avx512, f32, fma, vd, f_h, v_neg_d_offset);

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

                    let res_l_hi = $crate::simd_primitive!(avx512, f32, fma, vd, f_l_hi, v_neg_d_offset);
                    let res_h_hi = $crate::simd_primitive!(avx512, f32, fma, vd, f_h_hi, v_neg_d_offset);

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
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 8.0);

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

                    // Dequant: d * val + (-d * 8.0) via FMA
                    let res_l_0 = $crate::simd_primitive!(neon, f32, fma, vd, f_l_0, v_neg_d_offset);
                    let res_l_1 = $crate::simd_primitive!(neon, f32, fma, vd, f_l_1, v_neg_d_offset);
                    let res_l_2 = $crate::simd_primitive!(neon, f32, fma, vd, f_l_2, v_neg_d_offset);
                    let res_l_3 = $crate::simd_primitive!(neon, f32, fma, vd, f_l_3, v_neg_d_offset);

                    let res_h_0 = $crate::simd_primitive!(neon, f32, fma, vd, f_h_0, v_neg_d_offset);
                    let res_h_1 = $crate::simd_primitive!(neon, f32, fma, vd, f_h_1, v_neg_d_offset);
                    let res_h_2 = $crate::simd_primitive!(neon, f32, fma, vd, f_h_2, v_neg_d_offset);
                    let res_h_3 = $crate::simd_primitive!(neon, f32, fma, vd, f_h_3, v_neg_d_offset);
                    
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
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 8.0);
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

                    // Dequant: d * val + (-d * 8.0) via FMA
                    let dl0 = $crate::simd_primitive!(neon, f32, fma, vd, fl0, v_neg_d_offset);
                    let dh0 = $crate::simd_primitive!(neon, f32, fma, vd, fh0, v_neg_d_offset);
                    let dl1 = $crate::simd_primitive!(neon, f32, fma, vd, fl1, v_neg_d_offset);
                    let dh1 = $crate::simd_primitive!(neon, f32, fma, vd, fh1, v_neg_d_offset);
                    let dl2 = $crate::simd_primitive!(neon, f32, fma, vd, fl2, v_neg_d_offset);
                    let dh2 = $crate::simd_primitive!(neon, f32, fma, vd, fh2, v_neg_d_offset);
                    let dl3 = $crate::simd_primitive!(neon, f32, fma, vd, fl3, v_neg_d_offset);
                    let dh3 = $crate::simd_primitive!(neon, f32, fma, vd, fh3, v_neg_d_offset);

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
            // Q2_K: 16 sub-blocks of 16 values each = 256 values
            // block.scales[16]: each byte encodes sc (low 4 bits) and m (high 4 bits)
            // Formula: out = d * sc * q - dmin * m
            for sb in 0..16 {
                let sc = (block.scales[sb] & 0xF) as f32;
                let m = (block.scales[sb] >> 4) as f32;
                let d_sc = d * sc;
                let neg_dmin_m = -dmin * m;
                for j in 0..16 {
                    let idx = sb * 16 + j;
                    let byte_idx = idx / 4;
                    let shift = (idx % 4) * 2;
                    let q = ((block.qs[byte_idx] >> shift) & 0x03) as f32;
                    unsafe {
                        *$out_ptr.add(idx) = q.mul_add(d_sc, neg_dmin_m);
                    }
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
                unsafe { *$out_ptr.add(i) = (val as f32).mul_add(d, -d * 4.0); }
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
                unsafe { *$out_ptr.add(i) = (val as f32).mul_add(d, -dmin); }
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
                unsafe { *$out_ptr.add(i) = (val as f32).mul_add(d, -d * 32.0); }
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
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 8.0);

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

                    // Dequantize: d * val + (-d * 8.0) via FMA
                    let dq_l = $crate::simd_primitive!(avx512, f32, fma, vd, f_l, v_neg_d_offset);
                    let dq_h = $crate::simd_primitive!(avx512, f32, fma, vd, f_h, v_neg_d_offset);

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

    // -----------------------------------------------------------------------
    // AVX2 Q4_K dot -- optimized microkernel.
    //
    // Deferred scale:  dot = d * sum(nib_i * other_i) - d*8 * sum(other_i)
    //   Eliminates 4 dequant FMAs per iteration (was: fma(vd, nibble, -d*8)).
    // 4 independent accumulators break the FMA dependency chain
    //   (original: 4 dependent FMAs on 1 acc = 20c latency-bound per iter).
    // 2x unrolled loop for better OoO scheduling.
    // -----------------------------------------------------------------------
    (avx2, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
             #[cfg(target_arch = "x86_64")]
             {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();

                let qs_ptr = block.qs.as_ptr();
                let other_ptr = $other_ptr;
                let mask = _mm_set1_epi8(0x0F);

                // 4 independent accumulators for nib*other
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                // 2 independent accumulators for sum(other)
                let mut acc_oth0 = _mm256_setzero_ps();
                let mut acc_oth1 = _mm256_setzero_ps();

                // 2x unrolled: 4 iterations processing 2x16 bytes = 64 elements each
                let mut i = 0usize;
                while i < 8 {
                    // --- First half (i) ---
                    let v_a = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo_a  = _mm_and_si128(v_a, mask);
                    let hi_a = _mm_and_si128(_mm_srli_epi16(v_a, 4), mask);

                    let fl0a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_a));
                    let fl1a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo_a, 8)));
                    let fh0a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_a));
                    let fh1a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi_a, 8)));

                    let t0a = _mm256_unpacklo_ps(fl0a, fh0a);
                    let t1a = _mm256_unpackhi_ps(fl0a, fh0a);
                    let n0a = _mm256_permute2f128_ps(t0a, t1a, 0x20);
                    let n1a = _mm256_permute2f128_ps(t0a, t1a, 0x31);
                    let t2a = _mm256_unpacklo_ps(fl1a, fh1a);
                    let t3a = _mm256_unpackhi_ps(fl1a, fh1a);
                    let n2a = _mm256_permute2f128_ps(t2a, t3a, 0x20);
                    let n3a = _mm256_permute2f128_ps(t2a, t3a, 0x31);

                    let o0a = _mm256_loadu_ps(other_ptr.add(i * 32));
                    let o1a = _mm256_loadu_ps(other_ptr.add(i * 32 + 8));
                    let o2a = _mm256_loadu_ps(other_ptr.add(i * 32 + 16));
                    let o3a = _mm256_loadu_ps(other_ptr.add(i * 32 + 24));

                    acc0 = _mm256_fmadd_ps(n0a, o0a, acc0);
                    acc1 = _mm256_fmadd_ps(n1a, o1a, acc1);
                    acc2 = _mm256_fmadd_ps(n2a, o2a, acc2);
                    acc3 = _mm256_fmadd_ps(n3a, o3a, acc3);

                    acc_oth0 = _mm256_add_ps(acc_oth0,
                        _mm256_add_ps(_mm256_add_ps(o0a, o1a), _mm256_add_ps(o2a, o3a)));

                    // --- Second half (i+1) ---
                    let v_b = _mm_loadu_si128(qs_ptr.add((i + 1) * 16) as *const _);
                    let lo_b  = _mm_and_si128(v_b, mask);
                    let hi_b = _mm_and_si128(_mm_srli_epi16(v_b, 4), mask);

                    let fl0b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_b));
                    let fl1b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo_b, 8)));
                    let fh0b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_b));
                    let fh1b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi_b, 8)));

                    let t0b = _mm256_unpacklo_ps(fl0b, fh0b);
                    let t1b = _mm256_unpackhi_ps(fl0b, fh0b);
                    let n0b = _mm256_permute2f128_ps(t0b, t1b, 0x20);
                    let n1b = _mm256_permute2f128_ps(t0b, t1b, 0x31);
                    let t2b = _mm256_unpacklo_ps(fl1b, fh1b);
                    let t3b = _mm256_unpackhi_ps(fl1b, fh1b);
                    let n2b = _mm256_permute2f128_ps(t2b, t3b, 0x20);
                    let n3b = _mm256_permute2f128_ps(t2b, t3b, 0x31);

                    let o0b = _mm256_loadu_ps(other_ptr.add((i + 1) * 32));
                    let o1b = _mm256_loadu_ps(other_ptr.add((i + 1) * 32 + 8));
                    let o2b = _mm256_loadu_ps(other_ptr.add((i + 1) * 32 + 16));
                    let o3b = _mm256_loadu_ps(other_ptr.add((i + 1) * 32 + 24));

                    acc0 = _mm256_fmadd_ps(n0b, o0b, acc0);
                    acc1 = _mm256_fmadd_ps(n1b, o1b, acc1);
                    acc2 = _mm256_fmadd_ps(n2b, o2b, acc2);
                    acc3 = _mm256_fmadd_ps(n3b, o3b, acc3);

                    acc_oth1 = _mm256_add_ps(acc_oth1,
                        _mm256_add_ps(_mm256_add_ps(o0b, o1b), _mm256_add_ps(o2b, o3b)));

                    i += 2;
                }

                // Merge accumulators
                let acc_nib = _mm256_add_ps(
                    _mm256_add_ps(acc0, acc1),
                    _mm256_add_ps(acc2, acc3),
                );
                let acc_oth = _mm256_add_ps(acc_oth0, acc_oth1);

                let nib_sum = $crate::simd_primitive!(avx2, f32, reduce_sum, acc_nib);
                let oth_sum = $crate::simd_primitive!(avx2, f32, reduce_sum, acc_oth);

                // Final: d * sum(nib*other) - d*8 * sum(other)
                d * nib_sum - d * 8.0 * oth_sum
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
                     let neg_d8 = -d * 8.0;
                     sum += (l).mul_add(d, neg_d8) * *other.add(i*2);
                     sum += (h).mul_add(d, neg_d8) * *other.add(i*2+1);
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
            // Q2_K: 16 sub-blocks of 16 values each
            for sb in 0..16 {
                let sc = (block.scales[sb] & 0xF) as f32;
                let m = (block.scales[sb] >> 4) as f32;
                let d_sc = d * sc;
                let neg_dmin_m = -dmin * m;
                for j in 0..16 {
                    let idx = sb * 16 + j;
                    let byte_idx = idx / 4;
                    let shift = (idx % 4) * 2;
                    let q = ((block.qs[byte_idx] >> shift) & 0x03) as f32;
                    unsafe {
                        sum += q.mul_add(d_sc, neg_dmin_m) * *other.add(idx);
                    }
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
                unsafe { sum += (val).mul_add(d, -d * 4.0) * *other.add(i); }
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
                unsafe { sum += val.mul_add(d, -dmin) * *other.add(i); }
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
                unsafe { sum += (val).mul_add(d, -d * 32.0) * *other.add(i); }
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
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let mask2_128 = _mm_set1_epi8(0x03);
                // Q2_K: 16 sub-blocks of 16 values each = 256 values
                // Each sub-block: 4 bytes of qs, per-sub-block scale and min
                // Process one sub-block (16 values = 2 x ymm of 8 f32) per iteration
                for sb in 0..16 {
                    let sc = (block.scales[sb] & 0xF) as f32;
                    let m = (block.scales[sb] >> 4) as f32;
                    let vd_sc = $crate::simd_primitive!(avx2, f32, splat, d * sc);
                    let v_neg_dmin_m = $crate::simd_primitive!(avx2, f32, splat, -dmin * m);
                    // Load 4 bytes for this sub-block
                    let qs_off = sb * 4;
                    // Broadcast 4 bytes into 128-bit register
                    let raw32 = _mm_set1_epi32(*(qs_ptr.add(qs_off) as *const i32));
                    // Replicate each byte 4 times: byte0 x4, byte1 x4, byte2 x4, byte3 x4
                    let shuf = _mm_setr_epi8(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
                    let replicated = _mm_shuffle_epi8(raw32, shuf);
                    // Shift each group of 4 identical bytes by 0,2,4,6 bits
                    // Use 32-bit variable shift: each 32-bit lane shifts its 4 bytes uniformly
                    let shift_amounts = _mm_setr_epi32(0, 2, 4, 6);
                    let shifted = _mm_srlv_epi32(replicated, shift_amounts);
                    // Mask to 2 bits
                    let masked = _mm_and_si128(shifted, mask2_128);
                    // Now masked has 16 bytes, each 0-3, in the correct order:
                    // [byte0>>0, byte0>>0, byte0>>0, byte0>>0,  <- only low byte matters per lane
                    //  byte1>>2, ..., byte2>>4, ..., byte3>>6, ...]
                    // Actually each 32-bit lane has 4 bytes but only the low byte of each lane
                    // has the correct value after the 32-bit shift. The other 3 bytes in each lane
                    // are garbage. We need a different approach.
                    //
                    // Better: unpack 4 bytes into 16 values using byte-level operations.
                    // Load 4 bytes, create 16 values by combining shift+mask at byte granularity.
                    let b0 = *qs_ptr.add(qs_off);
                    let b1 = *qs_ptr.add(qs_off + 1);
                    let b2 = *qs_ptr.add(qs_off + 2);
                    let b3 = *qs_ptr.add(qs_off + 3);
                    // Pack 16 2-bit values into a __m128i as bytes
                    let vals = _mm_setr_epi8(
                        (b0 & 3) as i8, (b0 >> 2 & 3) as i8, (b0 >> 4 & 3) as i8, (b0 >> 6 & 3) as i8,
                        (b1 & 3) as i8, (b1 >> 2 & 3) as i8, (b1 >> 4 & 3) as i8, (b1 >> 6 & 3) as i8,
                        (b2 & 3) as i8, (b2 >> 2 & 3) as i8, (b2 >> 4 & 3) as i8, (b2 >> 6 & 3) as i8,
                        (b3 & 3) as i8, (b3 >> 2 & 3) as i8, (b3 >> 4 & 3) as i8, (b3 >> 6 & 3) as i8,
                    );
                    // Convert low 8 bytes -> 8 x f32 (ymm)
                    let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vals));
                    // Convert high 8 bytes -> 8 x f32 (ymm)
                    let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vals, 8)));
                    // out = d * sc * q - dmin * m
                    let r_lo = $crate::simd_primitive!(avx2, f32, fma, vd_sc, lo, v_neg_dmin_m);
                    let r_hi = $crate::simd_primitive!(avx2, f32, fma, vd_sc, hi, v_neg_dmin_m);
                    let base = sb * 16;
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), r_lo);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 8), r_hi);
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
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // Q2_K: 16 sub-blocks of 16 values each
                for sb in 0..16 {
                    let sc = (block.scales[sb] & 0xF) as f32;
                    let m = (block.scales[sb] >> 4) as f32;
                    let vd_sc = $crate::simd_primitive!(avx2, f32, splat, d * sc);
                    let v_neg_dmin_m = $crate::simd_primitive!(avx2, f32, splat, -dmin * m);
                    let qs_off = sb * 4;
                    let b0 = *qs_ptr.add(qs_off);
                    let b1 = *qs_ptr.add(qs_off + 1);
                    let b2 = *qs_ptr.add(qs_off + 2);
                    let b3 = *qs_ptr.add(qs_off + 3);
                    let vals = _mm_setr_epi8(
                        (b0 & 3) as i8, (b0 >> 2 & 3) as i8, (b0 >> 4 & 3) as i8, (b0 >> 6 & 3) as i8,
                        (b1 & 3) as i8, (b1 >> 2 & 3) as i8, (b1 >> 4 & 3) as i8, (b1 >> 6 & 3) as i8,
                        (b2 & 3) as i8, (b2 >> 2 & 3) as i8, (b2 >> 4 & 3) as i8, (b2 >> 6 & 3) as i8,
                        (b3 & 3) as i8, (b3 >> 2 & 3) as i8, (b3 >> 4 & 3) as i8, (b3 >> 6 & 3) as i8,
                    );
                    let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vals));
                    let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vals, 8)));
                    // dq = d * sc * q - dmin * m
                    let dq_lo = $crate::simd_primitive!(avx2, f32, fma, vd_sc, lo, v_neg_dmin_m);
                    let dq_hi = $crate::simd_primitive!(avx2, f32, fma, vd_sc, hi, v_neg_dmin_m);
                    let base = sb * 16;
                    let o_lo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    let o_hi = $crate::simd_primitive!(avx2, f32, load, other.add(base + 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq_lo, o_lo, acc);
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq_hi, o_hi, acc);
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
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                // Q2_K: 16 sub-blocks of 16 values each = 256 values
                // AVX-512: process one sub-block (16 values = 1 zmm) per iteration
                for sb in 0..16 {
                    let sc = (block.scales[sb] & 0xF) as f32;
                    let m = (block.scales[sb] >> 4) as f32;
                    let vd_sc = $crate::simd_primitive!(avx512, f32, splat, d * sc);
                    let v_neg_dmin_m = $crate::simd_primitive!(avx512, f32, splat, -dmin * m);
                    let qs_off = sb * 4;
                    let b0 = *qs_ptr.add(qs_off);
                    let b1 = *qs_ptr.add(qs_off + 1);
                    let b2 = *qs_ptr.add(qs_off + 2);
                    let b3 = *qs_ptr.add(qs_off + 3);
                    let vals = _mm_setr_epi8(
                        (b0 & 3) as i8, (b0 >> 2 & 3) as i8, (b0 >> 4 & 3) as i8, (b0 >> 6 & 3) as i8,
                        (b1 & 3) as i8, (b1 >> 2 & 3) as i8, (b1 >> 4 & 3) as i8, (b1 >> 6 & 3) as i8,
                        (b2 & 3) as i8, (b2 >> 2 & 3) as i8, (b2 >> 4 & 3) as i8, (b2 >> 6 & 3) as i8,
                        (b3 & 3) as i8, (b3 >> 2 & 3) as i8, (b3 >> 4 & 3) as i8, (b3 >> 6 & 3) as i8,
                    );
                    let ints = _mm512_cvtepu8_epi32(vals);
                    let vf = _mm512_cvtepi32_ps(ints);
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd_sc, vf, v_neg_dmin_m);
                    let base = sb * 16;
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
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
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                // Q2_K: 16 sub-blocks of 16 values each
                for sb in 0..16 {
                    let sc = (block.scales[sb] & 0xF) as f32;
                    let m = (block.scales[sb] >> 4) as f32;
                    let vd_sc = $crate::simd_primitive!(avx512, f32, splat, d * sc);
                    let v_neg_dmin_m = $crate::simd_primitive!(avx512, f32, splat, -dmin * m);
                    let qs_off = sb * 4;
                    let b0 = *qs_ptr.add(qs_off);
                    let b1 = *qs_ptr.add(qs_off + 1);
                    let b2 = *qs_ptr.add(qs_off + 2);
                    let b3 = *qs_ptr.add(qs_off + 3);
                    let vals = _mm_setr_epi8(
                        (b0 & 3) as i8, (b0 >> 2 & 3) as i8, (b0 >> 4 & 3) as i8, (b0 >> 6 & 3) as i8,
                        (b1 & 3) as i8, (b1 >> 2 & 3) as i8, (b1 >> 4 & 3) as i8, (b1 >> 6 & 3) as i8,
                        (b2 & 3) as i8, (b2 >> 2 & 3) as i8, (b2 >> 4 & 3) as i8, (b2 >> 6 & 3) as i8,
                        (b3 & 3) as i8, (b3 >> 2 & 3) as i8, (b3 >> 4 & 3) as i8, (b3 >> 6 & 3) as i8,
                    );
                    let ints = _mm512_cvtepu8_epi32(vals);
                    let vf = _mm512_cvtepi32_ps(ints);
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd_sc, vf, v_neg_dmin_m);
                    let base = sb * 16;
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 4.0);
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
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 4.0);
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
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 4.0);
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
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 4.0);
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
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_dmin = $crate::simd_primitive!(avx2, f32, splat, -dmin);
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
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_neg_dmin);
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
                let v_neg_dmin = $crate::simd_primitive!(avx2, f32, splat, -dmin);
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
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_neg_dmin);
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
                let v_neg_dmin = $crate::simd_primitive!(avx512, f32, splat, -dmin);
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
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_neg_dmin);
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
                let v_neg_dmin = $crate::simd_primitive!(avx512, f32, splat, -dmin);
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
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_neg_dmin);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 32.0);
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
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 32.0);
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
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 32.0);
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
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 32.0);
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
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_neg_d_offset);
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
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                // Q2_K: 16 sub-blocks of 16 values each = 256 values
                // Process one sub-block (16 values = 4 x float32x4) per iteration
                for sb in 0..16 {
                    let sc = (block.scales[sb] & 0xF) as f32;
                    let m = (block.scales[sb] >> 4) as f32;
                    let vd_sc = $crate::simd_primitive!(neon, f32, splat, d * sc);
                    let v_neg_dmin_m = $crate::simd_primitive!(neon, f32, splat, -dmin * m);
                    let qs_off = sb * 4;
                    // Unpack 4 bytes -> 16 2-bit values -> 4 x float32x4
                    let mut vals = [0u32; 16];
                    for bi in 0..4 {
                        let byte_val = *qs_ptr.add(qs_off + bi);
                        vals[bi * 4 + 0] = (byte_val & 0x03) as u32;
                        vals[bi * 4 + 1] = ((byte_val >> 2) & 0x03) as u32;
                        vals[bi * 4 + 2] = ((byte_val >> 4) & 0x03) as u32;
                        vals[bi * 4 + 3] = ((byte_val >> 6) & 0x03) as u32;
                    }
                    let base = sb * 16;
                    for chunk in 0..4 {
                        let vi = vld1q_u32(vals.as_ptr().add(chunk * 4));
                        let vf = vcvtq_f32_u32(vi);
                        let r = $crate::simd_primitive!(neon, f32, fma, vd_sc, vf, v_neg_dmin_m);
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + chunk * 4), r);
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
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                // Q2_K: 16 sub-blocks of 16 values each
                for sb in 0..16 {
                    let sc = (block.scales[sb] & 0xF) as f32;
                    let m = (block.scales[sb] >> 4) as f32;
                    let vd_sc = $crate::simd_primitive!(neon, f32, splat, d * sc);
                    let v_neg_dmin_m = $crate::simd_primitive!(neon, f32, splat, -dmin * m);
                    let qs_off = sb * 4;
                    let mut vals = [0u32; 16];
                    for bi in 0..4 {
                        let byte_val = *qs_ptr.add(qs_off + bi);
                        vals[bi * 4 + 0] = (byte_val & 0x03) as u32;
                        vals[bi * 4 + 1] = ((byte_val >> 2) & 0x03) as u32;
                        vals[bi * 4 + 2] = ((byte_val >> 4) & 0x03) as u32;
                        vals[bi * 4 + 3] = ((byte_val >> 6) & 0x03) as u32;
                    }
                    let base = sb * 16;
                    for chunk in 0..4 {
                        let vi = vld1q_u32(vals.as_ptr().add(chunk * 4));
                        let vf = vcvtq_f32_u32(vi);
                        let dq = $crate::simd_primitive!(neon, f32, fma, vd_sc, vf, v_neg_dmin_m);
                        let vo = $crate::simd_primitive!(neon, f32, load, other.add(base + chunk * 4));
                        acc = $crate::simd_primitive!(neon, f32, fma, dq, vo, acc);
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
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 4.0);
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
                    let res = $crate::simd_primitive!(neon, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 4.0);
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
                    let dq = $crate::simd_primitive!(neon, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_dmin = $crate::simd_primitive!(neon, f32, splat, -dmin);
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
                    let res = $crate::simd_primitive!(neon, f32, fma, vd, vf, v_neg_dmin);
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
                let v_neg_dmin = $crate::simd_primitive!(neon, f32, splat, -dmin);
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
                    let dq = $crate::simd_primitive!(neon, f32, fma, vd, vf, v_neg_dmin);
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
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 32.0);
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
                    let res = $crate::simd_primitive!(neon, f32, fma, vd, vf, v_neg_d_offset);
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
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 32.0);
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
                    let dq = $crate::simd_primitive!(neon, f32, fma, vd, vf, v_neg_d_offset);
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
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let out_ptr = $out_ptr;

                // Process 16 i8 values per iteration (2x8 f32), 16 iterations total
                for i in 0..16 {
                    // Low 8 i8 -> 8 i32 -> 8 f32
                    let vi32_lo = _mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(i * 16) as *const _)
                    );
                    let res_lo = $crate::simd_primitive!(avx2, f32, mul, vd, _mm256_cvtepi32_ps(vi32_lo));
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16), res_lo);

                    // High 8 i8 -> 8 i32 -> 8 f32
                    let vi32_hi = _mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(i * 16 + 8) as *const _)
                    );
                    let res_hi = $crate::simd_primitive!(avx2, f32, mul, vd, _mm256_cvtepi32_ps(vi32_hi));
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16 + 8), res_hi);
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
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;

                // 4 independent accumulators to hide FMA latency (5c lat / 0.5c tput).
                // Each iteration processes 32 elements (4x8 i8 -> 4x8 f32).
                // 8 iterations total for 256 elements.
                // Scale factor d is deferred to the final reduction (factored out of inner loop).
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                // Unrolled: 8 iterations x 32 elements = 256
                let mut i = 0usize;
                while i < 256 {
                    // Prefetch next iteration's data (2 cache lines ahead)
                    _mm_prefetch(qs_ptr.add(i + 64) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(other.add(i + 64) as *const i8, _MM_HINT_T0);

                    // Lane 0: elements [i..i+8]
                    let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(i) as *const _)));
                    let o0 = _mm256_loadu_ps(other.add(i));
                    acc0 = _mm256_fmadd_ps(q0, o0, acc0);

                    // Lane 1: elements [i+8..i+16]
                    let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(i + 8) as *const _)));
                    let o1 = _mm256_loadu_ps(other.add(i + 8));
                    acc1 = _mm256_fmadd_ps(q1, o1, acc1);

                    // Lane 2: elements [i+16..i+24]
                    let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(i + 16) as *const _)));
                    let o2 = _mm256_loadu_ps(other.add(i + 16));
                    acc2 = _mm256_fmadd_ps(q2, o2, acc2);

                    // Lane 3: elements [i+24..i+32]
                    let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(i + 24) as *const _)));
                    let o3 = _mm256_loadu_ps(other.add(i + 24));
                    acc3 = _mm256_fmadd_ps(q3, o3, acc3);

                    i += 32;
                }

                // Merge 4 accumulators (pairwise to maintain precision)
                let sum01 = _mm256_add_ps(acc0, acc1);
                let sum23 = _mm256_add_ps(acc2, acc3);
                let sum_all = _mm256_add_ps(sum01, sum23);

                // Horizontal reduction
                let t1 = _mm256_hadd_ps(sum_all, sum_all);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                let raw_sum = _mm_cvtss_f32(_mm_add_ps(hi128, lo128));

                // Apply scale factor once at the end
                d * raw_sum
            }
        }
    };
}
