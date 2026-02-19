/// Classic GGML quantization primitives (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1).
///
/// Sub-macro of the Layer 3 quant_primitive dispatcher.
/// Block size = 32 (unlike K-Quant which uses 256).
/// Scale stored as f16 (unlike K-Quant which mixes f16/f32).
#[macro_export]
macro_rules! quant_primitive_classic {
    // ========================================================================
    // Q4_0: d(f16) + qs[16] packed 4-bit, zero_point = 8
    // out[i] = d * (nibble - 8)
    // ========================================================================

    (scalar, q4_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            for i in 0..16 {
                let b = block.qs[i];
                let lo = (b & 0x0F) as f32;
                let hi = (b >> 4) as f32;
                unsafe {
                    *$out_ptr.add(i * 2) = d * (lo - 8.0);
                    *$out_ptr.add(i * 2 + 1) = d * (hi - 8.0);
                }
            }
        }
    };

    (scalar, q4_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..16 {
                let b = block.qs[i];
                let lo = (b & 0x0F) as f32;
                let hi = (b >> 4) as f32;
                unsafe {
                    sum += d * (lo - 8.0) * *other.add(i * 2);
                    sum += d * (hi - 8.0) * *other.add(i * 2 + 1);
                }
            }
            sum
        }
    };

    (avx2, q4_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx2, f32, splat, -d * 8.0);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let mask = _mm_set1_epi8(0x0F);
                // 16 bytes -> 32 values, process in one pass
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                // Low 8 + High 8
                let il0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_bytes));
                let il1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8)));
                let ih0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_bytes));
                let ih1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8)));
                let rl0 = $crate::simd_primitive!(avx2, f32, fma, vd, il0, v_off);
                let rl1 = $crate::simd_primitive!(avx2, f32, fma, vd, il1, v_off);
                let rh0 = $crate::simd_primitive!(avx2, f32, fma, vd, ih0, v_off);
                let rh1 = $crate::simd_primitive!(avx2, f32, fma, vd, ih1, v_off);
                // Interleave [l0,h0,l1,h1,...]
                let t0 = _mm256_unpacklo_ps(rl0, rh0);
                let t1 = _mm256_unpackhi_ps(rl0, rh0);
                let t2 = _mm256_unpacklo_ps(rl1, rh1);
                let t3 = _mm256_unpackhi_ps(rl1, rh1);
                let f0 = _mm256_permute2f128_ps(t0, t1, 0x20);
                let f1 = _mm256_permute2f128_ps(t0, t1, 0x31);
                let f2 = _mm256_permute2f128_ps(t2, t3, 0x20);
                let f3 = _mm256_permute2f128_ps(t2, t3, 0x31);
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(0), f0);
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(8), f1);
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(16), f2);
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(24), f3);
            }
        }
    };

    (avx2, q4_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mask = _mm_set1_epi8(0x0F);
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let il0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_bytes));
                let il1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8)));
                let ih0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_bytes));
                let ih1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8)));
                // Interleave
                let t0 = _mm256_unpacklo_ps(il0, ih0);
                let t1 = _mm256_unpackhi_ps(il0, ih0);
                let t2 = _mm256_unpacklo_ps(il1, ih1);
                let t3 = _mm256_unpackhi_ps(il1, ih1);
                let n0 = _mm256_permute2f128_ps(t0, t1, 0x20);
                let n1 = _mm256_permute2f128_ps(t0, t1, 0x31);
                let n2 = _mm256_permute2f128_ps(t2, t3, 0x20);
                let n3 = _mm256_permute2f128_ps(t2, t3, 0x31);
                // Deferred scale: d * sum(nib*other) - d*8 * sum(other)
                let o0 = _mm256_loadu_ps(other.add(0));
                let o1 = _mm256_loadu_ps(other.add(8));
                let o2 = _mm256_loadu_ps(other.add(16));
                let o3 = _mm256_loadu_ps(other.add(24));
                let mut acc = _mm256_mul_ps(n0, o0);
                acc = _mm256_fmadd_ps(n1, o1, acc);
                acc = _mm256_fmadd_ps(n2, o2, acc);
                acc = _mm256_fmadd_ps(n3, o3, acc);
                let acc_oth = _mm256_add_ps(_mm256_add_ps(o0, o1), _mm256_add_ps(o2, o3));
                let nib_sum = $crate::simd_primitive!(avx2, f32, reduce_sum, acc);
                let oth_sum = $crate::simd_primitive!(avx2, f32, reduce_sum, acc_oth);
                d * nib_sum - d * 8.0 * oth_sum
            }
        }
    };

    (avx512, q4_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx512, f32, splat, -d * 8.0);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let mask = _mm_set1_epi8(0x0F);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let fl = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(low_bytes));
                let fh = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(high_bytes));
                let rl = $crate::simd_primitive!(avx512, f32, fma, vd, fl, v_off);
                let rh = $crate::simd_primitive!(avx512, f32, fma, vd, fh, v_off);
                let lo = _mm512_unpacklo_ps(rl, rh);
                let hi = _mm512_unpackhi_ps(rl, rh);
                $crate::simd_primitive!(avx512, f32, store, out_ptr.add(0), lo);
                $crate::simd_primitive!(avx512, f32, store, out_ptr.add(16), hi);
            }
        }
    };

    (avx512, q4_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx512, f32, splat, -d * 8.0);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let mask = _mm_set1_epi8(0x0F);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let fl = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(low_bytes));
                let fh = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(high_bytes));
                let rl = $crate::simd_primitive!(avx512, f32, fma, vd, fl, v_off);
                let rh = $crate::simd_primitive!(avx512, f32, fma, vd, fh, v_off);
                let lo = _mm512_unpacklo_ps(rl, rh);
                let hi = _mm512_unpackhi_ps(rl, rh);
                let o_lo = $crate::simd_primitive!(avx512, f32, load, other.add(0));
                let o_hi = $crate::simd_primitive!(avx512, f32, load, other.add(16));
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                acc = $crate::simd_primitive!(avx512, f32, fma, lo, o_lo, acc);
                acc = $crate::simd_primitive!(avx512, f32, fma, hi, o_hi, acc);
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (neon, q4_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_off = $crate::simd_primitive!(neon, f32, splat, -d * 8.0);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let v128 = vld1q_u8(qs_ptr);
                let mask = vdupq_n_u8(0x0F);
                let low_bytes = vandq_u8(v128, mask);
                let high_bytes = vshrq_n_u8(v128, 4);
                let lo_u16_lo = vmovl_u8(vget_low_u8(low_bytes));
                let lo_u16_hi = vmovl_high_u8(low_bytes);
                let lo_u32_0 = vmovl_u16(vget_low_u16(lo_u16_lo));
                let lo_u32_1 = vmovl_high_u16(lo_u16_lo);
                let lo_u32_2 = vmovl_u16(vget_low_u16(lo_u16_hi));
                let lo_u32_3 = vmovl_high_u16(lo_u16_hi);
                let fl0 = vcvtq_f32_u32(lo_u32_0);
                let fl1 = vcvtq_f32_u32(lo_u32_1);
                let fl2 = vcvtq_f32_u32(lo_u32_2);
                let fl3 = vcvtq_f32_u32(lo_u32_3);
                let hi_u16_lo = vmovl_u8(vget_low_u8(high_bytes));
                let hi_u16_hi = vmovl_high_u8(high_bytes);
                let hi_u32_0 = vmovl_u16(vget_low_u16(hi_u16_lo));
                let hi_u32_1 = vmovl_high_u16(hi_u16_lo);
                let hi_u32_2 = vmovl_u16(vget_low_u16(hi_u16_hi));
                let hi_u32_3 = vmovl_high_u16(hi_u16_hi);
                let fh0 = vcvtq_f32_u32(hi_u32_0);
                let fh1 = vcvtq_f32_u32(hi_u32_1);
                let fh2 = vcvtq_f32_u32(hi_u32_2);
                let fh3 = vcvtq_f32_u32(hi_u32_3);
                let rl0 = $crate::simd_primitive!(neon, f32, fma, vd, fl0, v_off);
                let rl1 = $crate::simd_primitive!(neon, f32, fma, vd, fl1, v_off);
                let rl2 = $crate::simd_primitive!(neon, f32, fma, vd, fl2, v_off);
                let rl3 = $crate::simd_primitive!(neon, f32, fma, vd, fl3, v_off);
                let rh0 = $crate::simd_primitive!(neon, f32, fma, vd, fh0, v_off);
                let rh1 = $crate::simd_primitive!(neon, f32, fma, vd, fh1, v_off);
                let rh2 = $crate::simd_primitive!(neon, f32, fma, vd, fh2, v_off);
                let rh3 = $crate::simd_primitive!(neon, f32, fma, vd, fh3, v_off);
                let p0 = float32x4x2_t(rl0, rh0);
                vst2q_f32(out_ptr.add(0), p0);
                let p1 = float32x4x2_t(rl1, rh1);
                vst2q_f32(out_ptr.add(8), p1);
                let p2 = float32x4x2_t(rl2, rh2);
                vst2q_f32(out_ptr.add(16), p2);
                let p3 = float32x4x2_t(rl3, rh3);
                vst2q_f32(out_ptr.add(24), p3);
            }
        }
    };

    (neon, q4_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_off = $crate::simd_primitive!(neon, f32, splat, -d * 8.0);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let v128 = vld1q_u8(qs_ptr);
                let mask = vdupq_n_u8(0x0F);
                let low_bytes = vandq_u8(v128, mask);
                let high_bytes = vshrq_n_u8(v128, 4);
                let lo_u16_lo = vmovl_u8(vget_low_u8(low_bytes));
                let lo_u16_hi = vmovl_high_u8(low_bytes);
                let l0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_u16_lo)));
                let l1 = vcvtq_f32_u32(vmovl_high_u16(lo_u16_lo));
                let l2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_u16_hi)));
                let l3 = vcvtq_f32_u32(vmovl_high_u16(lo_u16_hi));
                let hi_u16_lo = vmovl_u8(vget_low_u8(high_bytes));
                let hi_u16_hi = vmovl_high_u8(high_bytes);
                let h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_u16_lo)));
                let h1 = vcvtq_f32_u32(vmovl_high_u16(hi_u16_lo));
                let h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_u16_hi)));
                let h3 = vcvtq_f32_u32(vmovl_high_u16(hi_u16_hi));
                let dl0 = $crate::simd_primitive!(neon, f32, fma, vd, l0, v_off);
                let dh0 = $crate::simd_primitive!(neon, f32, fma, vd, h0, v_off);
                let dl1 = $crate::simd_primitive!(neon, f32, fma, vd, l1, v_off);
                let dh1 = $crate::simd_primitive!(neon, f32, fma, vd, h1, v_off);
                let dl2 = $crate::simd_primitive!(neon, f32, fma, vd, l2, v_off);
                let dh2 = $crate::simd_primitive!(neon, f32, fma, vd, h2, v_off);
                let dl3 = $crate::simd_primitive!(neon, f32, fma, vd, l3, v_off);
                let dh3 = $crate::simd_primitive!(neon, f32, fma, vd, h3, v_off);
                let z0 = vzipq_f32(dl0, dh0);
                let z1 = vzipq_f32(dl1, dh1);
                let z2 = vzipq_f32(dl2, dh2);
                let z3 = vzipq_f32(dl3, dh3);
                acc = $crate::simd_primitive!(neon, f32, fma, z0.0, $crate::simd_primitive!(neon, f32, load, other.add(0)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z0.1, $crate::simd_primitive!(neon, f32, load, other.add(4)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z1.0, $crate::simd_primitive!(neon, f32, load, other.add(8)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z1.1, $crate::simd_primitive!(neon, f32, load, other.add(12)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z2.0, $crate::simd_primitive!(neon, f32, load, other.add(16)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z2.1, $crate::simd_primitive!(neon, f32, load, other.add(20)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z3.0, $crate::simd_primitive!(neon, f32, load, other.add(24)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, z3.1, $crate::simd_primitive!(neon, f32, load, other.add(28)), acc);
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // Q4_1: d(f16) + m(f16) + qs[16] packed 4-bit, with min
    // out[i] = d * nibble + m
    // ========================================================================

    (scalar, q4_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let m: f32 = block.m.to_f32();
            for i in 0..16 {
                let b = block.qs[i];
                let lo = (b & 0x0F) as f32;
                let hi = (b >> 4) as f32;
                unsafe {
                    *$out_ptr.add(i * 2) = d * lo + m;
                    *$out_ptr.add(i * 2 + 1) = d * hi + m;
                }
            }
        }
    };

    (scalar, q4_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let m: f32 = block.m.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..16 {
                let b = block.qs[i];
                let lo = (b & 0x0F) as f32;
                let hi = (b >> 4) as f32;
                unsafe {
                    sum += (d * lo + m) * *other.add(i * 2);
                    sum += (d * hi + m) * *other.add(i * 2 + 1);
                }
            }
            sum
        }
    };

    (avx2, q4_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let vm = $crate::simd_primitive!(avx2, f32, splat, m);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let mask = _mm_set1_epi8(0x0F);
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let il0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_bytes));
                let il1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8)));
                let ih0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_bytes));
                let ih1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8)));
                let rl0 = $crate::simd_primitive!(avx2, f32, fma, vd, il0, vm);
                let rl1 = $crate::simd_primitive!(avx2, f32, fma, vd, il1, vm);
                let rh0 = $crate::simd_primitive!(avx2, f32, fma, vd, ih0, vm);
                let rh1 = $crate::simd_primitive!(avx2, f32, fma, vd, ih1, vm);
                let t0 = _mm256_unpacklo_ps(rl0, rh0);
                let t1 = _mm256_unpackhi_ps(rl0, rh0);
                let t2 = _mm256_unpacklo_ps(rl1, rh1);
                let t3 = _mm256_unpackhi_ps(rl1, rh1);
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(0), _mm256_permute2f128_ps(t0, t1, 0x20));
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(8), _mm256_permute2f128_ps(t0, t1, 0x31));
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(16), _mm256_permute2f128_ps(t2, t3, 0x20));
                $crate::simd_primitive!(avx2, f32, store, out_ptr.add(24), _mm256_permute2f128_ps(t2, t3, 0x31));
            }
        }
    };

    (avx2, q4_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mask = _mm_set1_epi8(0x0F);
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let il0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_bytes));
                let il1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low_bytes, 8)));
                let ih0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_bytes));
                let ih1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high_bytes, 8)));
                let t0 = _mm256_unpacklo_ps(il0, ih0);
                let t1 = _mm256_unpackhi_ps(il0, ih0);
                let t2 = _mm256_unpacklo_ps(il1, ih1);
                let t3 = _mm256_unpackhi_ps(il1, ih1);
                let n0 = _mm256_permute2f128_ps(t0, t1, 0x20);
                let n1 = _mm256_permute2f128_ps(t0, t1, 0x31);
                let n2 = _mm256_permute2f128_ps(t2, t3, 0x20);
                let n3 = _mm256_permute2f128_ps(t2, t3, 0x31);
                let o0 = _mm256_loadu_ps(other.add(0));
                let o1 = _mm256_loadu_ps(other.add(8));
                let o2 = _mm256_loadu_ps(other.add(16));
                let o3 = _mm256_loadu_ps(other.add(24));
                let mut acc = _mm256_mul_ps(n0, o0);
                acc = _mm256_fmadd_ps(n1, o1, acc);
                acc = _mm256_fmadd_ps(n2, o2, acc);
                acc = _mm256_fmadd_ps(n3, o3, acc);
                let acc_oth = _mm256_add_ps(_mm256_add_ps(o0, o1), _mm256_add_ps(o2, o3));
                let nib_sum = $crate::simd_primitive!(avx2, f32, reduce_sum, acc);
                let oth_sum = $crate::simd_primitive!(avx2, f32, reduce_sum, acc_oth);
                d * nib_sum + m * oth_sum
            }
        }
    };

    (avx512, q4_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let vm = $crate::simd_primitive!(avx512, f32, splat, m);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let mask = _mm_set1_epi8(0x0F);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let fl = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(low_bytes));
                let fh = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(high_bytes));
                let rl = $crate::simd_primitive!(avx512, f32, fma, vd, fl, vm);
                let rh = $crate::simd_primitive!(avx512, f32, fma, vd, fh, vm);
                let lo = _mm512_unpacklo_ps(rl, rh);
                let hi = _mm512_unpackhi_ps(rl, rh);
                $crate::simd_primitive!(avx512, f32, store, out_ptr.add(0), lo);
                $crate::simd_primitive!(avx512, f32, store, out_ptr.add(16), hi);
            }
        }
    };

    (avx512, q4_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let vm = $crate::simd_primitive!(avx512, f32, splat, m);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let v128 = _mm_loadu_si128(qs_ptr as *const _);
                let mask = _mm_set1_epi8(0x0F);
                let low_bytes = _mm_and_si128(v128, mask);
                let high_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask);
                let fl = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(low_bytes));
                let fh = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(high_bytes));
                let rl = $crate::simd_primitive!(avx512, f32, fma, vd, fl, vm);
                let rh = $crate::simd_primitive!(avx512, f32, fma, vd, fh, vm);
                let lo = _mm512_unpacklo_ps(rl, rh);
                let hi = _mm512_unpackhi_ps(rl, rh);
                let o_lo = $crate::simd_primitive!(avx512, f32, load, other.add(0));
                let o_hi = $crate::simd_primitive!(avx512, f32, load, other.add(16));
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                acc = $crate::simd_primitive!(avx512, f32, fma, lo, o_lo, acc);
                acc = $crate::simd_primitive!(avx512, f32, fma, hi, o_hi, acc);
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (neon, q4_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        // Reuse Q4_0 NEON pattern but with +m instead of -d*8
        $crate::quant_primitive_classic!(scalar, q4_1, decode, $block_ptr, $out_ptr);
    };

    (neon, q4_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive_classic!(scalar, q4_1, dot, $block_ptr, $other_ptr)
    };

    // ========================================================================
    // Q5_0: d(f16) + qh[4] + qs[16], zero_point = 16
    // out[i] = d * ((hi_bit << 4 | lo_nibble) - 16)
    // ========================================================================

    (scalar, q5_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            for i in 0..32 {
                let lo = (block.qs[i / 2] >> ((i % 2) * 4)) & 0xF;
                let hi = (block.qh[i / 8] >> (i % 8)) & 1;
                let q = (hi << 4) | lo;
                unsafe { *$out_ptr.add(i) = d * (q as f32 - 16.0); }
            }
        }
    };

    (scalar, q5_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..32 {
                let lo = (block.qs[i / 2] >> ((i % 2) * 4)) & 0xF;
                let hi = (block.qh[i / 8] >> (i % 8)) & 1;
                let q = (hi << 4) | lo;
                unsafe { sum += d * (q as f32 - 16.0) * *other.add(i); }
            }
            sum
        }
    };

    (avx2, q5_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx2, f32, splat, -d * 16.0);
                let out_ptr = $out_ptr;
                // Extract 32 x 5-bit values into i32 array, then SIMD convert
                for i in 0..4 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_off);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx2, q5_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx2, f32, splat, -d * 16.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..4 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vf, v_off);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx2, f32, reduce_sum, acc)
            }
        }
    };

    (avx512, q5_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx512, f32, splat, -d * 16.0);
                let out_ptr = $out_ptr;
                for i in 0..2 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_off);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx512, q5_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_off = $crate::simd_primitive!(avx512, f32, splat, -d * 16.0);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                for i in 0..2 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vf, v_off);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (neon, q5_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        $crate::quant_primitive_classic!(scalar, q5_0, decode, $block_ptr, $out_ptr);
    };

    (neon, q5_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive_classic!(scalar, q5_0, dot, $block_ptr, $other_ptr)
    };

    // ========================================================================
    // Q5_1: d(f16) + m(f16) + qh[4] + qs[16], with min
    // out[i] = d * ((hi_bit << 4 | lo_nibble)) + m
    // ========================================================================

    (scalar, q5_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let m: f32 = block.m.to_f32();
            for i in 0..32 {
                let lo = (block.qs[i / 2] >> ((i % 2) * 4)) & 0xF;
                let hi = (block.qh[i / 8] >> (i % 8)) & 1;
                let q = ((hi << 4) | lo) as f32;
                unsafe { *$out_ptr.add(i) = d * q + m; }
            }
        }
    };

    (scalar, q5_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let m: f32 = block.m.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..32 {
                let lo = (block.qs[i / 2] >> ((i % 2) * 4)) & 0xF;
                let hi = (block.qh[i / 8] >> (i % 8)) & 1;
                let q = ((hi << 4) | lo) as f32;
                unsafe { sum += (d * q + m) * *other.add(i); }
            }
            sum
        }
    };

    (avx2, q5_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let vm = $crate::simd_primitive!(avx2, f32, splat, m);
                let out_ptr = $out_ptr;
                for i in 0..4 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vf, vm);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx2, q5_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let vm = $crate::simd_primitive!(avx2, f32, splat, m);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..4 {
                    let base = i * 8;
                    let mut vals = [0i32; 8];
                    for j in 0..8 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm256_loadu_si256(vals.as_ptr() as *const _);
                    let vf = _mm256_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vf, vm);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx2, f32, reduce_sum, acc)
            }
        }
    };

    (avx512, q5_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let vm = $crate::simd_primitive!(avx512, f32, splat, m);
                let out_ptr = $out_ptr;
                for i in 0..2 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vf, vm);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), res);
                }
            }
        }
    };

    (avx512, q5_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let m: f32 = block.m.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let vm = $crate::simd_primitive!(avx512, f32, splat, m);
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                for i in 0..2 {
                    let base = i * 16;
                    let mut vals = [0i32; 16];
                    for j in 0..16 {
                        let idx = base + j;
                        let lo = (block.qs[idx / 2] >> ((idx % 2) * 4)) & 0xF;
                        let hi = (block.qh[idx / 8] >> (idx % 8)) & 1;
                        vals[j] = ((hi << 4) | lo) as i32;
                    }
                    let vi = _mm512_loadu_si512(vals.as_ptr() as *const _);
                    let vf = _mm512_cvtepi32_ps(vi);
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vf, vm);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(base));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (neon, q5_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        $crate::quant_primitive_classic!(scalar, q5_1, decode, $block_ptr, $out_ptr);
    };

    (neon, q5_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive_classic!(scalar, q5_1, dot, $block_ptr, $other_ptr)
    };

    // ========================================================================
    // Q8_0: d(f16) + qs[32] signed i8
    // out[i] = d * qs[i]
    // ========================================================================

    (scalar, q8_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            for i in 0..32 {
                unsafe { *$out_ptr.add(i) = d * (block.qs[i] as f32); }
            }
        }
    };

    (scalar, q8_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..32 {
                unsafe { sum += d * (block.qs[i] as f32) * *other.add(i); }
            }
            sum
        }
    };

    (avx2, q8_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let out_ptr = $out_ptr;
                for i in 0..2 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v128));
                    let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v128, 8)));
                    let r_lo = $crate::simd_primitive!(avx2, f32, mul, vd, lo);
                    let r_hi = $crate::simd_primitive!(avx2, f32, mul, vd, hi);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16), r_lo);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16 + 8), r_hi);
                }
            }
        }
    };

    (avx2, q8_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..2 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v128));
                    let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v128, 8)));
                    let dq_lo = $crate::simd_primitive!(avx2, f32, mul, vd, lo);
                    let dq_hi = $crate::simd_primitive!(avx2, f32, mul, vd, hi);
                    let o_lo = $crate::simd_primitive!(avx2, f32, load, other.add(i * 16));
                    let o_hi = $crate::simd_primitive!(avx2, f32, load, other.add(i * 16 + 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq_lo, o_lo, acc);
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq_hi, o_hi, acc);
                }
                $crate::simd_primitive!(avx2, f32, reduce_sum, acc)
            }
        }
    };

    (avx512, q8_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let out_ptr = $out_ptr;
                for i in 0..2 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let ints = _mm512_cvtepi8_epi32(v128);
                    let vf = _mm512_cvtepi32_ps(ints);
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i * 16), res);
                }
            }
        }
    };

    (avx512, q8_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                for i in 0..2 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let ints = _mm512_cvtepi8_epi32(v128);
                    let vf = _mm512_cvtepi32_ps(ints);
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(i * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (neon, q8_0, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let out_ptr = $out_ptr;
                for i in 0..2 {
                    let v_i8 = vld1q_s8(qs_ptr.add(i * 16));
                    let v_i16_lo = vmovl_s8(vget_low_s8(v_i8));
                    let v_i16_hi = vmovl_high_s8(v_i8);
                    let v0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_i16_lo)));
                    let v1 = vcvtq_f32_s32(vmovl_high_s16(v_i16_lo));
                    let v2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_i16_hi)));
                    let v3 = vcvtq_f32_s32(vmovl_high_s16(v_i16_hi));
                    let r0 = $crate::simd_primitive!(neon, f32, mul, vd, v0);
                    let r1 = $crate::simd_primitive!(neon, f32, mul, vd, v1);
                    let r2 = $crate::simd_primitive!(neon, f32, mul, vd, v2);
                    let r3 = $crate::simd_primitive!(neon, f32, mul, vd, v3);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16), r0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 4), r1);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 8), r2);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 16 + 12), r3);
                }
            }
        }
    };

    (neon, q8_0, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let qs_ptr = block.qs.as_ptr() as *const i8;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                for i in 0..2 {
                    let v_i8 = vld1q_s8(qs_ptr.add(i * 16));
                    let v_i16_lo = vmovl_s8(vget_low_s8(v_i8));
                    let v_i16_hi = vmovl_high_s8(v_i8);
                    let vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_i16_lo)));
                    let vf1 = vcvtq_f32_s32(vmovl_high_s16(v_i16_lo));
                    let vf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_i16_hi)));
                    let vf3 = vcvtq_f32_s32(vmovl_high_s16(v_i16_hi));
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

    // ========================================================================
    // Q8_1: d(f16) + s(f16) + qs[32] signed i8
    // out[i] = d * qs[i]  (s is precomputed sum, used for dot optimization)
    // ========================================================================

    (scalar, q8_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            for i in 0..32 {
                unsafe { *$out_ptr.add(i) = d * (block.qs[i] as f32); }
            }
        }
    };

    (scalar, q8_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..32 {
                unsafe { sum += d * (block.qs[i] as f32) * *other.add(i); }
            }
            sum
        }
    };

    // Q8_1 SIMD: same as Q8_0 (s field unused in decode/dot against f32)
    (avx2, q8_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        $crate::quant_primitive_classic!(avx2, q8_0, decode, $block_ptr, $out_ptr);
    };
    (avx2, q8_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive_classic!(avx2, q8_0, dot, $block_ptr, $other_ptr)
    };
    (avx512, q8_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        $crate::quant_primitive_classic!(avx512, q8_0, decode, $block_ptr, $out_ptr);
    };
    (avx512, q8_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive_classic!(avx512, q8_0, dot, $block_ptr, $other_ptr)
    };
    (neon, q8_1, decode, $block_ptr:expr, $out_ptr:expr) => {
        $crate::quant_primitive_classic!(neon, q8_0, decode, $block_ptr, $out_ptr);
    };
    (neon, q8_1, dot, $block_ptr:expr, $other_ptr:expr) => {
        $crate::quant_primitive_classic!(neon, q8_0, dot, $block_ptr, $other_ptr)
    };
}
