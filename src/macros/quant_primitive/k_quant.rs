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
            let dmin: f32 = block.dmin.to_f32();

            // get_scale_min_k4 equivalent (matches llama.cpp)
            #[inline(always)]
            fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (f32, f32) {
                if j < 4 {
                    ((scales[j] & 63) as f32, (scales[j + 4] & 63) as f32)
                } else {
                    let sc = ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as f32;
                    let m  = ((scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)) as f32;
                    (sc, m)
                }
            }

            let mut is = 0usize;
            for group in 0..4usize {
                let q_off = group * 32;
                let out_off = group * 64;

                let (sc1, m1) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc1;
                let neg_m1 = dmin * m1;

                let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc2;
                let neg_m2 = dmin * m2;

                // First 32 values: low nibbles with scale is+0
                for l in 0..32usize {
                    let val = (block.qs[q_off + l] & 0xF) as f32;
                    unsafe {
                        *$out_ptr.add(out_off + l) = d1 * val - neg_m1;
                    }
                }
                // Next 32 values: high nibbles with scale is+1
                for l in 0..32usize {
                    let val = (block.qs[q_off + l] >> 4) as f32;
                    unsafe {
                        *$out_ptr.add(out_off + 32 + l) = d2 * val - neg_m2;
                    }
                }

                is += 2;
            }
        }
    };


    (avx2, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated for proper Q4_K
            $crate::quant_primitive_kquant!(scalar, q4_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx512, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated for proper Q4_K
            $crate::quant_primitive_kquant!(scalar, q4_k, decode, $block_ptr, $out_ptr)
        }
    };

    (neon, q4_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated for proper Q4_K
            $crate::quant_primitive_kquant!(scalar, q4_k, decode, $block_ptr, $out_ptr)
        }
    };


    (neon, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q4_k, dot, $block_ptr, $other_ptr)
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

                // 4 iterations × 64 elements = 256 total
                // Each iteration: 4 subblocks of 16 elements
                // All loads first, then batch stores — breaks store dependency chain
                let mut base = 0usize;
                while base < 256 {
                    // Prefetch next iteration's data
                    _mm_prefetch(qs_ptr.add(base + 64) as *const i8, _MM_HINT_T0);

                    // Load + convert 4 groups of 16
                    let v0 = _mm_loadu_si128(qs_ptr.add(base) as *const _);
                    let v1 = _mm_loadu_si128(qs_ptr.add(base + 16) as *const _);
                    let v2 = _mm_loadu_si128(qs_ptr.add(base + 32) as *const _);
                    let v3 = _mm_loadu_si128(qs_ptr.add(base + 48) as *const _);

                    let f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v0));
                    let f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v1));
                    let f2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v2));
                    let f3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v3));

                    // Scale all 4 vectors
                    let r0 = $crate::simd_primitive!(avx512, f32, mul, vd, f0);
                    let r1 = $crate::simd_primitive!(avx512, f32, mul, vd, f1);
                    let r2 = $crate::simd_primitive!(avx512, f32, mul, vd, f2);
                    let r3 = $crate::simd_primitive!(avx512, f32, mul, vd, f3);

                    // Batch store
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base), r0);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base + 16), r1);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base + 32), r2);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(base + 48), r3);

                    base += 64;
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

            let qs = &block.qs;
            let mut is = 0usize;
            let mut out_idx = 0usize;

            // 2 groups of 128 values
            for n_group in 0..2usize {
                let q_off = n_group * 32;
                let mut shift: u32 = 0;

                // 4 sub-iterations per group, each producing 32 values (16 + 16)
                for _j in 0..4usize {
                    let sc_byte = block.scales[is];
                    let dl = d * (sc_byte & 0xF) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;
                    for l in 0..16usize {
                        let q = ((qs[q_off + l] >> shift) & 3) as f32;
                        unsafe { *$out_ptr.add(out_idx) = dl * q - ml; }
                        out_idx += 1;
                    }

                    let sc_byte = block.scales[is];
                    let dl = d * (sc_byte & 0xF) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;
                    for l in 0..16usize {
                        let q = ((qs[q_off + l + 16] >> shift) & 3) as f32;
                        unsafe { *$out_ptr.add(out_idx) = dl * q - ml; }
                        out_idx += 1;
                    }

                    shift += 2;
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
            let d_all = block.d.to_f32();

            // Unpack 16 x 6-bit scales from scales[12] using aux/kmask trick (matches llama.cpp)
            let kmask1: u32 = 0x03030303;
            let kmask2: u32 = 0x0f0f0f0f;

            let mut aux = [0u32; 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    block.scales.as_ptr(),
                    aux.as_mut_ptr() as *mut u8,
                    12,
                );
            }
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            let scales: &[i8; 16] = unsafe { &*(aux.as_ptr() as *const [i8; 16]) };

            let q = &block.qs;
            let hm = &block.hmask;
            let mut m: u8 = 1;
            let mut is = 0usize;
            let mut out_idx = 0usize;

            for n_group in 0..2usize {
                let q_off = n_group * 32;
                let mut shift = 0u32;
                for _j in 0..4usize {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16usize {
                        let qval = ((q[q_off + l] >> shift) & 3) as i8;
                        let hbit = if (hm[l] & m) != 0 { 0i8 } else { 4i8 };
                        unsafe { *$out_ptr.add(out_idx) = dl * (qval - hbit) as f32; }
                        out_idx += 1;
                    }

                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16usize {
                        let qval = ((q[q_off + l + 16] >> shift) & 3) as i8;
                        let hbit = if (hm[l + 16] & m) != 0 { 0i8 } else { 4i8 };
                        unsafe { *$out_ptr.add(out_idx) = dl * (qval - hbit) as f32; }
                        out_idx += 1;
                    }

                    shift += 2;
                    m <<= 1;
                }
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
            let min: f32 = block.dmin.to_f32();

            // get_scale_min_k4 equivalent (matches llama.cpp)
            #[inline(always)]
            fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (f32, f32) {
                if j < 4 {
                    ((scales[j] & 63) as f32, (scales[j + 4] & 63) as f32)
                } else {
                    let sc = ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as f32;
                    let m  = ((scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)) as f32;
                    (sc, m)
                }
            }

            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            for group in 0..4usize {
                let ql_off = group * 32;
                let out_off = group * 64;

                let (sc1, m1) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc1;
                let neg_m1 = min * m1;

                let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc2;
                let neg_m2 = min * m2;

                // First 32 values: low nibbles + high bit from qh
                for l in 0..32usize {
                    let val = (ql[ql_off + l] & 0xF) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                    unsafe {
                        *$out_ptr.add(out_off + l) = d1 * (val as f32) - neg_m1;
                    }
                }
                // Next 32 values: high nibbles + high bit from qh
                for l in 0..32usize {
                    let val = (ql[ql_off + l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                    unsafe {
                        *$out_ptr.add(out_off + 32 + l) = d2 * (val as f32) - neg_m2;
                    }
                }

                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    };

    // ------------------------------------------------------------------------
    // Q6_K Decoding
    // ------------------------------------------------------------------------
    (scalar, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d.to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let sc = &block.scales;

            // Process 256 values in two groups of 128
            for n_group in 0..2usize {
                let ql_off = n_group * 64;
                let qh_off = n_group * 32;
                let sc_off = n_group * 8;
                let out_off = n_group * 128;

                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = ((ql[ql_off + l]      & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[ql_off + l + 32]  & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[ql_off + l]      >> 4)   | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[ql_off + l + 32]  >> 4)   | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
                    unsafe {
                        *$out_ptr.add(out_off + l)      = d * (sc[sc_off + is] as i8 as f32) * (q1 as f32);
                        *$out_ptr.add(out_off + l + 32)  = d * (sc[sc_off + is + 2] as i8 as f32) * (q2 as f32);
                        *$out_ptr.add(out_off + l + 64)  = d * (sc[sc_off + is + 4] as i8 as f32) * (q3 as f32);
                        *$out_ptr.add(out_off + l + 96)  = d * (sc[sc_off + is + 6] as i8 as f32) * (q4 as f32);
                    }
                }
            }
        }
    };


    // ------------------------------------------------------------------------
    // Dot Products (Scalar)
    // ------------------------------------------------------------------------

    (avx512, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q4_k, dot, $block_ptr, $other_ptr)
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
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q4_k, dot, $block_ptr, $other_ptr)
        }
    };

    (scalar, q4_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let dmin: f32 = block.dmin.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;

            #[inline(always)]
            fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (f32, f32) {
                if j < 4 {
                    ((scales[j] & 63) as f32, (scales[j + 4] & 63) as f32)
                } else {
                    let sc = ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as f32;
                    let m  = ((scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)) as f32;
                    (sc, m)
                }
            }

            let mut is = 0usize;
            for group in 0..4usize {
                let q_off = group * 32;
                let out_off = group * 64;

                let (sc1, m1) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc1;
                let neg_m1 = dmin * m1;

                let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc2;
                let neg_m2 = dmin * m2;

                for l in 0..32usize {
                    let val = (block.qs[q_off + l] & 0xF) as f32;
                    unsafe { sum += (d1 * val - neg_m1) * *other.add(out_off + l); }
                }
                for l in 0..32usize {
                    let val = (block.qs[q_off + l] >> 4) as f32;
                    unsafe { sum += (d2 * val - neg_m2) * *other.add(out_off + 32 + l); }
                }

                is += 2;
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

            let qs = &block.qs;
            let mut is = 0usize;
            let mut out_idx = 0usize;

            for n_group in 0..2usize {
                let q_off = n_group * 32;
                let mut shift: u32 = 0;

                for _j in 0..4usize {
                    let sc_byte = block.scales[is];
                    let dl = d * (sc_byte & 0xF) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;
                    for l in 0..16usize {
                        let q = ((qs[q_off + l] >> shift) & 3) as f32;
                        unsafe { sum += (dl * q - ml) * *other.add(out_idx); }
                        out_idx += 1;
                    }

                    let sc_byte = block.scales[is];
                    let dl = d * (sc_byte & 0xF) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;
                    for l in 0..16usize {
                        let q = ((qs[q_off + l + 16] >> shift) & 3) as f32;
                        unsafe { sum += (dl * q - ml) * *other.add(out_idx); }
                        out_idx += 1;
                    }

                    shift += 2;
                }
            }
            sum
        }
    };

    (scalar, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d_all = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;

            let kmask1: u32 = 0x03030303;
            let kmask2: u32 = 0x0f0f0f0f;

            let mut aux = [0u32; 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    block.scales.as_ptr(),
                    aux.as_mut_ptr() as *mut u8,
                    12,
                );
            }
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            let scales: &[i8; 16] = unsafe { &*(aux.as_ptr() as *const [i8; 16]) };

            let q = &block.qs;
            let hm = &block.hmask;
            let mut m: u8 = 1;
            let mut is = 0usize;
            let mut out_idx = 0usize;

            for n_group in 0..2usize {
                let q_off = n_group * 32;
                let mut shift = 0u32;
                for _j in 0..4usize {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16usize {
                        let qval = ((q[q_off + l] >> shift) & 3) as i8;
                        let hbit = if (hm[l] & m) != 0 { 0i8 } else { 4i8 };
                        unsafe { sum += (dl * (qval - hbit) as f32) * *other.add(out_idx); }
                        out_idx += 1;
                    }

                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16usize {
                        let qval = ((q[q_off + l + 16] >> shift) & 3) as i8;
                        let hbit = if (hm[l + 16] & m) != 0 { 0i8 } else { 4i8 };
                        unsafe { sum += (dl * (qval - hbit) as f32) * *other.add(out_idx); }
                        out_idx += 1;
                    }

                    shift += 2;
                    m <<= 1;
                }
            }
            sum
        }
    };

    (scalar, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let min: f32 = block.dmin.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;

            #[inline(always)]
            fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (f32, f32) {
                if j < 4 {
                    ((scales[j] & 63) as f32, (scales[j + 4] & 63) as f32)
                } else {
                    let sc = ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as f32;
                    let m  = ((scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)) as f32;
                    (sc, m)
                }
            }

            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            for group in 0..4usize {
                let ql_off = group * 32;
                let out_off = group * 64;

                let (sc1, m1) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc1;
                let neg_m1 = min * m1;

                let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc2;
                let neg_m2 = min * m2;

                for l in 0..32usize {
                    let val = (ql[ql_off + l] & 0xF) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                    unsafe { sum += (d1 * (val as f32) - neg_m1) * *other.add(out_off + l); }
                }
                for l in 0..32usize {
                    let val = (ql[ql_off + l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                    unsafe { sum += (d2 * (val as f32) - neg_m2) * *other.add(out_off + 32 + l); }
                }

                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            sum
        }
    };

    (scalar, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d = block.d.to_f32();
            let other = $other_ptr;
            let mut sum = 0.0f32;
            let ql = &block.qs;
            let qh = &block.qh;
            let sc = &block.scales;

            for n_group in 0..2usize {
                let ql_off = n_group * 64;
                let qh_off = n_group * 32;
                let sc_off = n_group * 8;
                let out_off = n_group * 128;

                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = ((ql[ql_off + l]      & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[ql_off + l + 32]  & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[ql_off + l]      >> 4)   | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[ql_off + l + 32]  >> 4)   | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
                    unsafe {
                        sum += d * (sc[sc_off + is] as i8 as f32) * (q1 as f32) * *other.add(out_off + l);
                        sum += d * (sc[sc_off + is + 2] as i8 as f32) * (q2 as f32) * *other.add(out_off + l + 32);
                        sum += d * (sc[sc_off + is + 4] as i8 as f32) * (q3 as f32) * *other.add(out_off + l + 64);
                        sum += d * (sc[sc_off + is + 6] as i8 as f32) * (q4 as f32) * *other.add(out_off + l + 96);
                    }
                }
            }
            sum
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q2_K
    // ========================================================================
    (avx2, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q2_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx2, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q2_k, dot, $block_ptr, $other_ptr)
        }
    };

    (avx512, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q2_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx512, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q2_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q3_K
    // ========================================================================
    (avx2, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q3_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx2, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q3_k, dot, $block_ptr, $other_ptr)
        }
    };

    (avx512, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q3_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx512, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q3_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q5_K
    // ========================================================================
    (avx2, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q5_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx2, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q5_k, dot, $block_ptr, $other_ptr)
        }
    };

    (avx512, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q5_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx512, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q5_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // AVX2 K-Quant decode/dot: Q6_K
    // ========================================================================
    (avx2, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q6_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx2, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q6_k, dot, $block_ptr, $other_ptr)
        }
    };

    (avx512, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q6_k, decode, $block_ptr, $out_ptr)
        }
    };

    (avx512, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q6_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q2_K
    // ========================================================================
    (neon, q2_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q2_k, decode, $block_ptr, $out_ptr)
        }
    };

    (neon, q2_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q2_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q3_K
    // ========================================================================
    (neon, q3_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q3_k, decode, $block_ptr, $out_ptr)
        }
    };

    (neon, q3_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q3_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q5_K
    // ========================================================================
    (neon, q5_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q5_k, decode, $block_ptr, $out_ptr)
        }
    };

    (neon, q5_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q5_k, dot, $block_ptr, $other_ptr)
        }
    };

    // ========================================================================
    // NEON K-Quant decode/dot: Q6_K
    // ========================================================================
    (neon, q6_k, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q6_k, decode, $block_ptr, $out_ptr)
        }
    };

    (neon, q6_k, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            // Correctness-first: use scalar logic until SIMD is updated
            $crate::quant_primitive_kquant!(scalar, q6_k, dot, $block_ptr, $other_ptr)
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

                // 4 iterations × 64 elements = 256 total
                // Each iteration: 4 subblocks of 16 elements (lo+hi 8 each)
                // All loads first, then converts+muls, then batch stores — breaks store dependency chain
                let mut base = 0usize;
                while base < 256 {
                    // Prefetch next iteration's data (64 bytes = 1 cache line ahead)
                    _mm_prefetch(qs_ptr.add(base + 64) as *const i8, _MM_HINT_T0);

                    // Load + convert all 8 groups (4 subblocks × lo/hi)
                    let q0l = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base) as *const _)));
                    let q0h = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 8) as *const _)));
                    let q1l = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 16) as *const _)));
                    let q1h = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 24) as *const _)));
                    let q2l = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 32) as *const _)));
                    let q2h = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 40) as *const _)));
                    let q3l = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 48) as *const _)));
                    let q3h = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(qs_ptr.add(base + 56) as *const _)));

                    // Scale all 8 vectors
                    let r0 = $crate::simd_primitive!(avx2, f32, mul, vd, q0l);
                    let r1 = $crate::simd_primitive!(avx2, f32, mul, vd, q0h);
                    let r2 = $crate::simd_primitive!(avx2, f32, mul, vd, q1l);
                    let r3 = $crate::simd_primitive!(avx2, f32, mul, vd, q1h);
                    let r4 = $crate::simd_primitive!(avx2, f32, mul, vd, q2l);
                    let r5 = $crate::simd_primitive!(avx2, f32, mul, vd, q2h);
                    let r6 = $crate::simd_primitive!(avx2, f32, mul, vd, q3l);
                    let r7 = $crate::simd_primitive!(avx2, f32, mul, vd, q3h);

                    // Batch store all 8 results
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base), r0);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 8), r1);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 16), r2);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 24), r3);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 32), r4);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 40), r5);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 48), r6);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(base + 56), r7);

                    base += 64;
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
