/// Maps abstract SIMD operations to concrete hardware intrinsics or fallback implementations.
///
/// # Architecture
/// This macro is the "Layer 1" of the macro architecture. It provides a unified interface
/// for:
/// - Architecture constants (lanes, register counts, cache parameters)
/// - Compute primitives (add, fma, load, store, etc.)
///
/// # Usage
/// ```ignore
/// simd_primitive!(scalar, f32, add, a, b) // -> a + b
/// simd_primitive!(avx2, f32, add, a, b)   // -> _mm256_add_ps(a, b)
/// ```
#[macro_export]
macro_rules! simd_primitive {
    // ========================================================================
    // Scalar Fallback (Baseline)
    // ========================================================================

    // --- f32 Architecture Constants ---
    (scalar, f32, num_regs) => { usize::MAX };
    (scalar, f32, optimal_tile_m) => { 1 };
    (scalar, f32, optimal_tile_n_vecs) => { 1 };
    (scalar, f32, prefetch_distance) => { 0 };
    (scalar, f32, has_native_fp16) => { false };
    (scalar, f32, has_native_bf16) => { false };

    // --- f32 Compute Primitives ---
    (scalar, f32, lanes) => { 1 };
    (scalar, f32, zero) => { 0.0f32 };
    (scalar, f32, splat, $v:expr) => { $v };
    (scalar, f32, load, $p:expr) => { unsafe { *$p } };
    (scalar, f32, store, $p:expr, $v:expr) => { unsafe { *$p = $v } };
    (scalar, f32, add, $a:expr, $b:expr) => { $a + $b };
    (scalar, f32, sub, $a:expr, $b:expr) => { $a - $b };
    (scalar, f32, mul, $a:expr, $b:expr) => { $a * $b };
    (scalar, f32, div, $a:expr, $b:expr) => { $a / $b };
    (scalar, f32, fma, $a:expr, $b:expr, $c:expr) => { $c + $a * $b };
    (scalar, f32, fnmadd, $a:expr, $b:expr, $c:expr) => { $c - $a * $b };
    (scalar, f32, neg, $a:expr) => { -$a };
    (scalar, f32, max, $a:expr, $b:expr) => { $a.max($b) };
    (scalar, f32, min, $a:expr, $b:expr) => { $a.min($b) };
    (scalar, f32, reduce_sum, $v:expr) => { $v };
    (scalar, f32, reduce_max, $v:expr) => { $v };
    (scalar, f32, abs, $a:expr) => { $a.abs() };
    (scalar, f32, exp, $a:expr) => { $a.exp() };
    (scalar, f32, exp_fast, $a:expr) => { $a.exp() }; // scalar fallback: same as full exp
    (scalar, f32, recip, $a:expr) => { 1.0 / $a };
    (scalar, f32, sqrt, $a:expr) => { $a.sqrt() };
    (scalar, f32, rsqrt, $a:expr) => { 1.0 / $a.sqrt() };
    (scalar, f32, prefetch, $p:expr, $dist:expr) => { };
    (scalar, f32, prefetch_t1, $p:expr) => { };
    (scalar, f32, prefetch_nta, $p:expr) => { };
    // Scalar stream: just a regular store
    (scalar, f32, stream, $p:expr, $v:expr) => { unsafe { *$p = $v } };

    // --- Integer Primitives (Scalar Fallback) ---
    (scalar, i32, splat, $v:expr) => { $v };
    (scalar, i32, load, $p:expr) => { unsafe { *($p as *const i32) } };
    (scalar, i32, add, $a:expr, $b:expr) => { $a.wrapping_add($b) };
    (scalar, i32, and, $a:expr, $b:expr) => { $a & $b };
    (scalar, i32, or, $a:expr, $b:expr) => { $a | $b };
    (scalar, i32, xor, $a:expr, $b:expr) => { $a ^ $b };
    (scalar, i32, shl, $a:expr, $shift:expr) => { $a << $shift };
    (scalar, i32, shl_23, $a:expr) => { $a << 23 };
    (scalar, i32, shr, $a:expr, $shift:expr) => { $a >> $shift }; // arithmetic shift
    (scalar, i32, cast_f32, $a:expr) => { $a as f32 };
    (scalar, i32, cast_bits_f32, $a:expr) => { f32::from_bits($a as u32) };
    (scalar, f32, round_nearest, $a:expr) => { $a.round() };
    (scalar, f32, cvt_to_i32, $a:expr) => { $a as i32 };
    (scalar, f32, cvt_from_i32, $a:expr) => { $a as f32 };
    // Shuffle: scalar shuffle is identity or specific index? 
    // Usually shuffle is typically used for rearranging bytes.
    // For scalar, we might need a custom implementation if operating on bytes, but scalar mode 
    // usually means we process 1 element at a time, so shuffle is irrelevant.
    (scalar, i8, shuffle, $a:expr, $mask:expr) => { $a }; // Stub for scalar loop


    // ========================================================================
    // AVX2 Implementation
    // ========================================================================

    // --- f32 Architecture Constants ---
    (avx2, f32, num_regs) => { 16 };
    (avx2, f32, optimal_tile_m) => { 6 };
    (avx2, f32, optimal_tile_n_vecs) => { 2 };
    (avx2, f32, prefetch_distance) => { 256 };
    (avx2, f32, has_native_fp16) => { false };
    (avx2, f32, has_native_bf16) => { false };

    // --- f32 Compute Primitives ---
    (avx2, f32, lanes) => { 8 };
    (avx2, f32, zero) => { std::arch::x86_64::_mm256_setzero_ps() };
    (avx2, f32, splat, $v:expr) => { std::arch::x86_64::_mm256_set1_ps($v) };
    (avx2, f32, load, $p:expr) => { std::arch::x86_64::_mm256_loadu_ps($p) };
    (avx2, f32, loadu, $p:expr) => { std::arch::x86_64::_mm256_loadu_ps($p) };
    (avx2, f32, store, $p:expr, $v:expr) => { std::arch::x86_64::_mm256_storeu_ps($p, $v) };
    (avx2, f32, storeu, $p:expr, $v:expr) => { std::arch::x86_64::_mm256_storeu_ps($p, $v) };
    (avx2, f32, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_add_ps($a, $b) };
    (avx2, f32, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_sub_ps($a, $b) };
    (avx2, f32, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_mul_ps($a, $b) };
    (avx2, f32, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_div_ps($a, $b) };
    (avx2, f32, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c) };
    (avx2, f32, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm256_fnmadd_ps($a, $b, $c) };
    (avx2, f32, neg, $a:expr) => { std::arch::x86_64::_mm256_sub_ps(std::arch::x86_64::_mm256_setzero_ps(), $a) }; // 0 - a
    (avx2, f32, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_max_ps($a, $b) };
    (avx2, f32, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_min_ps($a, $b) };
    
    // Reduce sum: horizontal add
    (avx2, f32, reduce_sum, $v:expr) => { 
        {
            let h1 = std::arch::x86_64::_mm256_hadd_ps($v, $v);
            let h2 = std::arch::x86_64::_mm256_hadd_ps(h1, h1);
            let t1 = std::arch::x86_64::_mm256_extractf128_ps(h2, 1);
            let t2 = std::arch::x86_64::_mm256_castps256_ps128(h2);
            let res = std::arch::x86_64::_mm_add_ps(t1, t2);
            std::arch::x86_64::_mm_cvtss_f32(res)
        }
    };
    
    // Exp, Recip, Sqrt etc.
    (avx2, f32, sqrt, $a:expr) => { std::arch::x86_64::_mm256_sqrt_ps($a) };
    // rsqrt with one Newton-Raphson refinement: ~23-bit accuracy
    (avx2, f32, rsqrt, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let r = _mm256_rsqrt_ps($a);
            let half = _mm256_set1_ps(0.5);
            let three = _mm256_set1_ps(3.0);
            let ar = _mm256_mul_ps($a, r);
            // r * 0.5 * (3 - a * r * r)
            _mm256_mul_ps(_mm256_mul_ps(r, half), _mm256_fnmadd_ps(ar, r, three))
        }
    };
    // recip with one Newton-Raphson refinement: ~23-bit accuracy
    (avx2, f32, recip, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let r = _mm256_rcp_ps($a);
            let two = _mm256_set1_ps(2.0);
            // r * (2 - a * r)
            _mm256_mul_ps(r, _mm256_fnmadd_ps($a, r, two))
        }
    };
    // EXP
    (avx2, f32, exp, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_f32($a) };
    // EXP_FAST: degree-3 polynomial, ~12-bit accuracy, 2-3x faster
    (avx2, f32, exp_fast, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_fast_f32($a) };

    // Reduce max: horizontal max
    (avx2, f32, reduce_max, $v:expr) => {
        {
            // Compare high/low 128-bit halves
            let hi = std::arch::x86_64::_mm256_extractf128_ps($v, 1);
            let lo = std::arch::x86_64::_mm256_castps256_ps128($v);
            let m1 = std::arch::x86_64::_mm_max_ps(hi, lo);
            // Shuffle and max within 128-bit
            let m2 = std::arch::x86_64::_mm_movehl_ps(m1, m1);
            let m3 = std::arch::x86_64::_mm_max_ps(m1, m2);
            let m4 = std::arch::x86_64::_mm_shuffle_ps(m3, m3, 1);
            let m5 = std::arch::x86_64::_mm_max_ss(m3, m4);
            std::arch::x86_64::_mm_cvtss_f32(m5)
        }
    };
    (avx2, f32, abs, $a:expr) => { std::arch::x86_64::_mm256_andnot_ps(std::arch::x86_64::_mm256_set1_ps(-0.0), $a) };

    // Non-temporal (streaming) store: bypasses cache, useful for write-only data
    (avx2, f32, stream, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm256_stream_ps($p, $v)
    };
    (avx2, f32, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx2, f32, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx2, f32, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };
    // Masked load: load `count` f32 elements (0..8), rest are zero
    (avx2, f32, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            // Build mask: lane i gets -1 (0xFFFFFFFF) if i < count, else 0
            let idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let thresh = _mm256_set1_epi32($count as i32);
            let mask = _mm256_cmpgt_epi32(thresh, idx); // mask[i] = (count > i) ? -1 : 0
            _mm256_maskload_ps($p, mask)
        }
    };
    // Masked store: store `count` f32 elements (0..8)
    (avx2, f32, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let thresh = _mm256_set1_epi32($count as i32);
            let mask = _mm256_cmpgt_epi32(thresh, idx);
            _mm256_maskstore_ps($p, mask, $v);
        }
    };
    // Note: We use __m256i for integer vectors.
    (avx2, i32, zero) => { std::arch::x86_64::_mm256_setzero_si256() };
    (avx2, i32, splat, $v:expr) => { std::arch::x86_64::_mm256_set1_epi32($v) };
    (avx2, i32, load, $p:expr) => { std::arch::x86_64::_mm256_loadu_si256($p as *const std::arch::x86_64::__m256i) };
    (avx2, i32, store, $p:expr, $v:expr) => { std::arch::x86_64::_mm256_storeu_si256($p as *mut std::arch::x86_64::__m256i, $v) };
    
    (avx2, i32, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_add_epi32($a, $b) };
    (avx2, i32, and, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_and_si256($a, $b) };
    (avx2, i32, or, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_or_si256($a, $b) };
    (avx2, i32, xor, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_xor_si256($a, $b) };
    (avx2, i32, shl, $a:expr, $shift:expr) => { std::arch::x86_64::_mm256_slli_epi32($a, $shift) };
    (avx2, i32, shl_23, $a:expr) => { std::arch::x86_64::_mm256_slli_epi32($a, 23) };
    (avx2, i32, shr, $a:expr, $shift:expr) => { std::arch::x86_64::_mm256_srai_epi32($a, $shift) }; // arithmetic right shift
    (avx2, i32, shr_u, $a:expr, $shift:expr) => { std::arch::x86_64::_mm256_srli_epi32($a, $shift) }; // logical right shift

    (avx2, i32, cast_f32, $a:expr) => { std::arch::x86_64::_mm256_cvtepi32_ps($a) };
    (avx2, i32, cast_bits_f32, $a:expr) => { std::arch::x86_64::_mm256_castsi256_ps($a) };
    (avx2, f32, round_nearest, $a:expr) => { std::arch::x86_64::_mm256_round_ps($a, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT | std::arch::x86_64::_MM_FROUND_NO_EXC) };
    (avx2, f32, cvt_to_i32, $a:expr) => { std::arch::x86_64::_mm256_cvtps_epi32($a) };
    (avx2, f32, cvt_from_i32, $a:expr) => { std::arch::x86_64::_mm256_cvtepi32_ps($a) };
    
    // Shuffle bytes (pshufb) - operates on 128-bit memory lanes!
    // _mm256_shuffle_epi8
    (avx2, i8, shuffle, $a:expr, $mask:expr) => { std::arch::x86_64::_mm256_shuffle_epi8($a, $mask) };

    // ========================================================================
    // AVX-512 Implementation
    // ========================================================================

    // --- f32 Architecture Constants ---
    (avx512, f32, num_regs) => { 32 };
    (avx512, f32, optimal_tile_m) => { 14 };
    (avx512, f32, optimal_tile_n_vecs) => { 2 };
    (avx512, f32, prefetch_distance) => { 512 };
    (avx512, f32, has_native_fp16) => { false }; // AVX512-FP16 is separate extension
    (avx512, f32, has_native_bf16) => { false };

    // --- f32 Compute Primitives ---
    (avx512, f32, lanes) => { 16 };
    (avx512, f32, zero) => { std::arch::x86_64::_mm512_setzero_ps() };
    (avx512, f32, splat, $v:expr) => { std::arch::x86_64::_mm512_set1_ps($v) };
    (avx512, f32, load, $p:expr) => { std::arch::x86_64::_mm512_loadu_ps($p) };
    (avx512, f32, loadu, $p:expr) => { std::arch::x86_64::_mm512_loadu_ps($p) };
    (avx512, f32, store, $p:expr, $v:expr) => { std::arch::x86_64::_mm512_storeu_ps($p, $v) };
    (avx512, f32, storeu, $p:expr, $v:expr) => { std::arch::x86_64::_mm512_storeu_ps($p, $v) };
    // Masked load: load `count` elements (0..16), rest are zero
    (avx512, f32, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask16 = (1u32.wrapping_shl($count as u32) - 1) as __mmask16;
            _mm512_maskz_loadu_ps(mask, $p)
        }
    };
    // Masked store: store `count` elements (0..16)
    (avx512, f32, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask16 = (1u32.wrapping_shl($count as u32) - 1) as __mmask16;
            _mm512_mask_storeu_ps($p, mask, $v);
        }
    };
    // Non-temporal (streaming) store: bypasses cache, useful for write-only data
    (avx512, f32, stream, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm512_stream_ps($p, $v)
    };
    (avx512, f32, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_add_ps($a, $b) };
    (avx512, f32, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_sub_ps($a, $b) };
    (avx512, f32, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_mul_ps($a, $b) };
    (avx512, f32, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_div_ps($a, $b) };
    (avx512, f32, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fmadd_ps($a, $b, $c) };
    (avx512, f32, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fnmadd_ps($a, $b, $c) };
    (avx512, f32, neg, $a:expr) => { std::arch::x86_64::_mm512_sub_ps(std::arch::x86_64::_mm512_setzero_ps(), $a) };
    (avx512, f32, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_max_ps($a, $b) };
    (avx512, f32, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_min_ps($a, $b) };

    // Reduce sum: native reduce
    (avx512, f32, reduce_sum, $v:expr) => { std::arch::x86_64::_mm512_reduce_add_ps($v) };
    // Reduce max: native reduce
    (avx512, f32, reduce_max, $v:expr) => { std::arch::x86_64::_mm512_reduce_max_ps($v) };

    (avx512, f32, abs, $a:expr) => { std::arch::x86_64::_mm512_abs_ps($a) };
    (avx512, f32, sqrt, $a:expr) => { std::arch::x86_64::_mm512_sqrt_ps($a) };
    // rsqrt with one Newton-Raphson refinement: ~23-bit accuracy
    (avx512, f32, rsqrt, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let r = _mm512_rsqrt14_ps($a);
            let half = _mm512_set1_ps(0.5);
            let three = _mm512_set1_ps(3.0);
            let ar = _mm512_mul_ps($a, r);
            _mm512_mul_ps(_mm512_mul_ps(r, half), _mm512_fnmadd_ps(ar, r, three))
        }
    };
    // recip with one Newton-Raphson refinement: ~23-bit accuracy
    (avx512, f32, recip, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let r = _mm512_rcp14_ps($a);
            let two = _mm512_set1_ps(2.0);
            _mm512_mul_ps(r, _mm512_fnmadd_ps($a, r, two))
        }
    };

    // EXP: native AVX-512 polynomial approximation
    (avx512, f32, exp, $a:expr) => {
        $crate::cpu_kernels::avx512::math::avx512_exp_f32($a)
    };
    (avx512, f32, exp_fast, $a:expr) => {
        $crate::cpu_kernels::avx512::math::avx512_exp_fast_f32($a)
    };

    (avx512, f32, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx512, f32, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx512, f32, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };

    // --- AVX-512 f16 (F16C via 256-bit halves, compute in f32) ---
    (avx512, f16, lanes) => { 16 };
    (avx512, f16, num_regs) => { 32 };
    (avx512, f16, zero) => { std::arch::x86_64::_mm512_setzero_ps() };
    (avx512, f16, splat, $v:expr) => { { std::arch::x86_64::_mm512_set1_ps($v.to_f32()) } };
    (avx512, f16, load, $p:expr) => {
        {
            // Load 16 f16 as two 128-bit chunks, convert each to 256-bit f32, combine to 512-bit
            let lo = std::arch::x86_64::_mm256_cvtph_ps(
                std::arch::x86_64::_mm_loadu_si128($p as *const std::arch::x86_64::__m128i)
            );
            let hi = std::arch::x86_64::_mm256_cvtph_ps(
                std::arch::x86_64::_mm_loadu_si128(($p as *const std::arch::x86_64::__m128i).add(1))
            );
            let mut result = std::arch::x86_64::_mm512_castps256_ps512(lo);
            result = std::arch::x86_64::_mm512_insertf32x8(result, hi, 1);
            result
        }
    };
    (avx512, f16, store, $p:expr, $v:expr) => {
        {
            let lo = std::arch::x86_64::_mm512_castps512_ps256($v);
            let hi = std::arch::x86_64::_mm512_extractf32x8_ps($v, 1);
            let lo_h = std::arch::x86_64::_mm256_cvtps_ph(lo, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT);
            let hi_h = std::arch::x86_64::_mm256_cvtps_ph(hi, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT);
            std::arch::x86_64::_mm_storeu_si128($p as *mut std::arch::x86_64::__m128i, lo_h);
            std::arch::x86_64::_mm_storeu_si128(($p as *mut std::arch::x86_64::__m128i).add(1), hi_h);
        }
    };
    (avx512, f16, optimal_tile_m) => { 14 };
    (avx512, f16, optimal_tile_n_vecs) => { 2 };
    (avx512, f16, prefetch_distance) => { 512 };
    (avx512, f16, has_native_fp16) => { false };
    (avx512, f16, has_native_bf16) => { false };
    (avx512, f16, loadu, $p:expr) => { $crate::simd_primitive!(avx512, f16, load, $p) };
    (avx512, f16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(avx512, f16, store, $p, $v) };
    (avx512, f16, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_add_ps($a, $b) };
    (avx512, f16, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_sub_ps($a, $b) };
    (avx512, f16, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_mul_ps($a, $b) };
    (avx512, f16, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_div_ps($a, $b) };
    (avx512, f16, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fmadd_ps($a, $b, $c) };
    (avx512, f16, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fnmadd_ps($a, $b, $c) };
    (avx512, f16, neg, $a:expr) => { std::arch::x86_64::_mm512_sub_ps(std::arch::x86_64::_mm512_setzero_ps(), $a) };
    (avx512, f16, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_max_ps($a, $b) };
    (avx512, f16, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_min_ps($a, $b) };
    (avx512, f16, reduce_sum, $v:expr) => { std::arch::x86_64::_mm512_reduce_add_ps($v) };
    (avx512, f16, reduce_max, $v:expr) => { std::arch::x86_64::_mm512_reduce_max_ps($v) };
    (avx512, f16, abs, $a:expr) => { std::arch::x86_64::_mm512_abs_ps($a) };
    (avx512, f16, sqrt, $a:expr) => { std::arch::x86_64::_mm512_sqrt_ps($a) };
    (avx512, f16, rsqrt, $a:expr) => { $crate::simd_primitive!(avx512, f32, rsqrt, $a) };
    (avx512, f16, recip, $a:expr) => { $crate::simd_primitive!(avx512, f32, recip, $a) };
    (avx512, f16, exp, $a:expr) => { $crate::simd_primitive!(avx512, f32, exp, $a) };
    (avx512, f16, exp_fast, $a:expr) => { $crate::simd_primitive!(avx512, f32, exp_fast, $a) };
    (avx512, f16, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx512, f16, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx512, f16, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };
    // Non-temporal store for f16: convert f32→f16 then stream 256-bit
    (avx512, f16, stream, $p:expr, $v:expr) => {
        {
            let lo = std::arch::x86_64::_mm512_castps512_ps256($v);
            let hi = std::arch::x86_64::_mm512_extractf32x8_ps($v, 1);
            let lo_h = std::arch::x86_64::_mm256_cvtps_ph(lo, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT);
            let hi_h = std::arch::x86_64::_mm256_cvtps_ph(hi, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT);
            std::arch::x86_64::_mm_stream_si128($p as *mut std::arch::x86_64::__m128i, lo_h);
            std::arch::x86_64::_mm_stream_si128(($p as *mut std::arch::x86_64::__m128i).add(1), hi_h);
        }
    };
    // Masked load: load `count` f16 elements (0..16), convert to f32, rest are zero
    (avx512, f16, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask16 = (1u32.wrapping_shl($count as u32) - 1) as __mmask16;
            // Masked load 16-bit integers, zero-extend to 32-bit, then F16C convert
            let raw = _mm256_maskz_loadu_epi16(mask, $p as *const i16);
            // Convert __m256i (8×f16 in low 128 + 8×f16 in high 128) via two F16C calls
            let lo128 = _mm256_castsi256_si128(raw);
            let hi128 = _mm256_extracti128_si256(raw, 1);
            let lo_f32 = _mm256_cvtph_ps(lo128);
            let hi_f32 = _mm256_cvtph_ps(hi128);
            // Combine into __m512
            let combined = _mm512_castps256_ps512(lo_f32);
            _mm512_insertf32x8(combined, hi_f32, 1)
        }
    };
    // Masked store: store `count` f16 elements (0..16) from f32 vector
    (avx512, f16, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask16 = (1u32.wrapping_shl($count as u32) - 1) as __mmask16;
            let lo = _mm512_castps512_ps256($v);
            let hi = _mm512_extractf32x8_ps($v, 1);
            let lo_h = _mm256_cvtps_ph(lo, _MM_FROUND_TO_NEAREST_INT);
            let hi_h = _mm256_cvtps_ph(hi, _MM_FROUND_TO_NEAREST_INT);
            // Pack two __m128i into __m256i
            let packed = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_h), hi_h, 1);
            _mm256_mask_storeu_epi16($p as *mut i16, mask, packed);
        }
    };

    // --- AVX-512 bf16 (bit-shift conversion, compute in f32) ---
    (avx512, bf16, lanes) => { 16 };
    (avx512, bf16, num_regs) => { 32 };
    (avx512, bf16, zero) => { std::arch::x86_64::_mm512_setzero_ps() };
    (avx512, bf16, splat, $v:expr) => { { std::arch::x86_64::_mm512_set1_ps($v.to_f32()) } };
    (avx512, bf16, load, $p:expr) => {
        {
            // Load 16 bf16 (256-bit), zero-extend to 16 u32 (512-bit), shift left 16
            let v256 = std::arch::x86_64::_mm256_loadu_si256($p as *const std::arch::x86_64::__m256i);
            let v512 = std::arch::x86_64::_mm512_cvtepu16_epi32(v256);
            let shifted = std::arch::x86_64::_mm512_slli_epi32(v512, 16);
            std::arch::x86_64::_mm512_castsi512_ps(shifted)
        }
    };
    (avx512, bf16, store, $p:expr, $v:expr) => {
        {
            let vi = std::arch::x86_64::_mm512_castps_si512($v);
            let shifted = std::arch::x86_64::_mm512_srli_epi32(vi, 16);
            let packed = std::arch::x86_64::_mm512_cvtepi32_epi16(shifted);
            std::arch::x86_64::_mm256_storeu_si256($p as *mut std::arch::x86_64::__m256i, packed);
        }
    };
    (avx512, bf16, optimal_tile_m) => { 14 };
    (avx512, bf16, optimal_tile_n_vecs) => { 2 };
    (avx512, bf16, prefetch_distance) => { 512 };
    (avx512, bf16, has_native_fp16) => { false };
    (avx512, bf16, has_native_bf16) => { false };
    (avx512, bf16, loadu, $p:expr) => { $crate::simd_primitive!(avx512, bf16, load, $p) };
    (avx512, bf16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(avx512, bf16, store, $p, $v) };
    (avx512, bf16, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_add_ps($a, $b) };
    (avx512, bf16, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_sub_ps($a, $b) };
    (avx512, bf16, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_mul_ps($a, $b) };
    (avx512, bf16, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_div_ps($a, $b) };
    (avx512, bf16, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fmadd_ps($a, $b, $c) };
    (avx512, bf16, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fnmadd_ps($a, $b, $c) };
    (avx512, bf16, neg, $a:expr) => { std::arch::x86_64::_mm512_sub_ps(std::arch::x86_64::_mm512_setzero_ps(), $a) };
    (avx512, bf16, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_max_ps($a, $b) };
    (avx512, bf16, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_min_ps($a, $b) };
    (avx512, bf16, reduce_sum, $v:expr) => { std::arch::x86_64::_mm512_reduce_add_ps($v) };
    (avx512, bf16, reduce_max, $v:expr) => { std::arch::x86_64::_mm512_reduce_max_ps($v) };
    (avx512, bf16, abs, $a:expr) => { std::arch::x86_64::_mm512_abs_ps($a) };
    (avx512, bf16, sqrt, $a:expr) => { std::arch::x86_64::_mm512_sqrt_ps($a) };
    (avx512, bf16, rsqrt, $a:expr) => { $crate::simd_primitive!(avx512, f32, rsqrt, $a) };
    (avx512, bf16, recip, $a:expr) => { $crate::simd_primitive!(avx512, f32, recip, $a) };
    (avx512, bf16, exp, $a:expr) => { $crate::simd_primitive!(avx512, f32, exp, $a) };
    (avx512, bf16, exp_fast, $a:expr) => { $crate::simd_primitive!(avx512, f32, exp_fast, $a) };
    (avx512, bf16, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx512, bf16, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx512, bf16, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };
    // Non-temporal store for bf16: convert f32→bf16 then stream 256-bit
    (avx512, bf16, stream, $p:expr, $v:expr) => {
        {
            let vi = std::arch::x86_64::_mm512_castps_si512($v);
            let shifted = std::arch::x86_64::_mm512_srli_epi32(vi, 16);
            let packed = std::arch::x86_64::_mm512_cvtepi32_epi16(shifted);
            std::arch::x86_64::_mm256_stream_si256($p as *mut std::arch::x86_64::__m256i, packed);
        }
    };
    // Masked load: load `count` bf16 elements (0..16), convert to f32, rest are zero
    (avx512, bf16, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask16 = (1u32.wrapping_shl($count as u32) - 1) as __mmask16;
            // Masked load 16-bit bf16 values as integers
            let raw = _mm256_maskz_loadu_epi16(mask, $p as *const i16);
            let v512 = _mm512_cvtepu16_epi32(raw);
            let shifted = _mm512_slli_epi32(v512, 16);
            _mm512_castsi512_ps(shifted)
        }
    };
    // Masked store: store `count` bf16 elements (0..16) from f32 vector
    (avx512, bf16, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask16 = (1u32.wrapping_shl($count as u32) - 1) as __mmask16;
            // f32→bf16: shift right 16 (truncate), pack 32→16
            let vi = _mm512_castps_si512($v);
            let shifted = _mm512_srli_epi32(vi, 16);
            let packed = _mm512_cvtepi32_epi16(shifted);
            _mm256_mask_storeu_epi16($p as *mut i16, mask, packed);
        }
    };

    // --- AVX-512 BF16 native dot-product (avx512bf16 extension) ---
    // vdpbf16ps: src += a(bf16_pairs) * b(bf16_pairs), 2 bf16 MACs per 32-bit lane
    // Requires: avx512bf16 target feature (Cooper Lake / Sapphire Rapids+)
    // Usage: accumulator is __m512 (f32), operands are __m512bh (packed bf16 pairs)
    // The caller must load raw bf16 data as __m512bh (not convert to f32).
    //
    // Primitive entries for future specialized bf16 GEMM microkernel:
    //   simd_primitive!(avx512, bf16, dpbf16ps, acc, a_bh, b_bh)
    //   simd_primitive!(avx512, bf16, load_raw, ptr)  -- load 32 bf16 as __m512bh
    //
    // These are NOT used by the current define_matmul_x86! macro because:
    // 1. dpbf16ps processes PAIRS of bf16 values per 32-bit lane (2x throughput)
    // 2. The K-loop must advance by 2 per iteration, not 1
    // 3. Input layout must be interleaved bf16 pairs, not scalar-broadcast
    // 4. A dedicated matmul_bf16_native! macro is needed
    //
    // When implementing, the microkernel structure would be:
    //   for k in (0..K).step_by(2) {
    //       let a_pair: __m512bh = load_bf16_pair(A, row, k);  // broadcast 2 bf16 from A
    //       let b_pair: __m512bh = load_raw(B_packed, k*TN);   // 32 bf16 from packed B
    //       acc = _mm512_dpbf16_ps(acc, a_pair, b_pair);
    //   }

    // --- AVX-512 VNNI INT8 dot-product (avx512vnni extension) ---
    // vpdpbusd: src += a(u8x4) * b(i8x4), 4 u8*i8 MACs per 32-bit lane
    // Requires: avx512vnni target feature (Cascade Lake+)
    // Usage: all operands are __m512i, accumulator is i32 lanes
    //
    // Primitive entries for future specialized INT8 GEMM microkernel:
    //   simd_primitive!(avx512, i8, dpbusd, acc, a_u8, b_i8)
    //   simd_primitive!(avx512, i8, load_i512, ptr)
    //
    // These are NOT used by the current define_matmul_x86! macro because:
    // 1. dpbusd processes 4 u8*i8 products per 32-bit lane (4x throughput vs f32)
    // 2. Requires quantized (u8/i8) input data, not float
    // 3. K-loop advances by 4 per iteration
    // 4. Accumulator is i32, needs post-loop dequantization to f32
    // 5. A dedicated matmul_int8_vnni! macro is needed


    // --- AVX-512 Integer Primitives ---
    (avx512, i32, zero) => { std::arch::x86_64::_mm512_setzero_si512() };
    (avx512, i32, splat, $v:expr) => { std::arch::x86_64::_mm512_set1_epi32($v) };
    (avx512, i32, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_add_epi32($a, $b) };
    (avx512, i32, shl, $a:expr, $shift:expr) => { std::arch::x86_64::_mm512_slli_epi32($a, $shift) };
    (avx512, i32, shl_23, $a:expr) => { std::arch::x86_64::_mm512_slli_epi32($a, 23) };
    (avx512, i32, shr, $a:expr, $shift:expr) => { std::arch::x86_64::_mm512_srai_epi32($a, $shift) };
    (avx512, i32, cast_f32, $a:expr) => { std::arch::x86_64::_mm512_cvtepi32_ps($a) };
    (avx512, i32, cast_bits_f32, $a:expr) => { std::arch::x86_64::_mm512_castsi512_ps($a) };
    (avx512, f32, round_nearest, $a:expr) => { std::arch::x86_64::_mm512_roundscale_ps($a, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT | std::arch::x86_64::_MM_FROUND_NO_EXC) };
    (avx512, f32, cvt_to_i32, $a:expr) => { std::arch::x86_64::_mm512_cvtps_epi32($a) };
    (avx512, f32, cvt_from_i32, $a:expr) => { std::arch::x86_64::_mm512_cvtepi32_ps($a) };


    // ========================================================================
    // NEON Implementation (aarch64)
    // ========================================================================

    // --- f32 Architecture Constants ---
    (neon, f32, num_regs) => { 32 };
    (neon, f32, optimal_tile_m) => { 8 }; // TBD
    (neon, f32, optimal_tile_n_vecs) => { 3 };
    (neon, f32, prefetch_distance) => { 128 };
    (neon, f32, has_native_fp16) => { true }; // ARMv8.2+ usually
    (neon, f32, has_native_bf16) => { false }; // Check target feature

    // --- f32 Compute Primitives ---
    (neon, f32, lanes) => { 4 };
    (neon, f32, zero) => { unsafe { std::arch::aarch64::vdupq_n_f32(0.0) } };
    (neon, f32, splat, $v:expr) => { unsafe { std::arch::aarch64::vdupq_n_f32($v) } };
    (neon, f32, load, $p:expr) => { unsafe { std::arch::aarch64::vld1q_f32($p) } };
    (neon, f32, loadu, $p:expr) => { unsafe { std::arch::aarch64::vld1q_f32($p) } }; // NEON loads are generally unaligned safe
    (neon, f32, store, $p:expr, $v:expr) => { unsafe { std::arch::aarch64::vst1q_f32($p, $v) } };
    (neon, f32, storeu, $p:expr, $v:expr) => { unsafe { std::arch::aarch64::vst1q_f32($p, $v) } };
    (neon, f32, add, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vaddq_f32($a, $b) } };
    (neon, f32, sub, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vsubq_f32($a, $b) } };
    (neon, f32, mul, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmulq_f32($a, $b) } };
    (neon, f32, div, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vdivq_f32($a, $b) } };
    (neon, f32, fma, $a:expr, $b:expr, $c:expr) => { unsafe { std::arch::aarch64::vfmaq_f32($c, $a, $b) } }; // c + a * b
    (neon, f32, fnmadd, $a:expr, $b:expr, $c:expr) => { unsafe { std::arch::aarch64::vfmsq_f32($c, $a, $b) } }; // c - a * b
    (neon, f32, neg, $a:expr) => { unsafe { std::arch::aarch64::vnegq_f32($a) } };
    (neon, f32, max, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmaxq_f32($a, $b) } };
    (neon, f32, min, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vminq_f32($a, $b) } };
    
    // Reduce sum
    (neon, f32, reduce_sum, $v:expr) => { unsafe { std::arch::aarch64::vaddvq_f32($v) } };
    (neon, f32, reduce_max, $v:expr) => { unsafe { std::arch::aarch64::vmaxvq_f32($v) } };
    (neon, f32, abs, $a:expr) => { unsafe { std::arch::aarch64::vabsq_f32($a) } };
    
    // Ops
    (neon, f32, sqrt, $a:expr) => { unsafe { std::arch::aarch64::vsqrtq_f32($a) } };
    // rsqrt with one Newton-Raphson step via vrsqrtsq_f32
    (neon, f32, rsqrt, $a:expr) => {
        unsafe {
            let r = std::arch::aarch64::vrsqrteq_f32($a);
            // vrsqrtsq_f32(a, r*r) computes (3 - a * r * r) / 2
            let step = std::arch::aarch64::vrsqrtsq_f32($a, std::arch::aarch64::vmulq_f32(r, r));
            std::arch::aarch64::vmulq_f32(r, step)
        }
    };
    // recip with one Newton-Raphson step via vrecpsq_f32
    (neon, f32, recip, $a:expr) => {
        unsafe {
            let r = std::arch::aarch64::vrecpeq_f32($a);
            // vrecpsq_f32(a, r) computes (2 - a * r)
            let step = std::arch::aarch64::vrecpsq_f32($a, r);
            std::arch::aarch64::vmulq_f32(r, step)
        }
    };
    
    // EXP
    (neon, f32, exp, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) };
    (neon, f32, exp_fast, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) }; // NEON: same impl (already fast)

    (neon, f32, prefetch, $p:expr, $dist:expr) => {
        unsafe { core::arch::asm!("prfm pldl1keep, [{addr}]", addr = in(reg) ($p as *const u8).add($dist), options(nostack, preserves_flags)) }
    };
    (neon, f32, prefetch_t1, $p:expr) => {
        unsafe { core::arch::asm!("prfm pldl2keep, [{addr}]", addr = in(reg) ($p as *const u8), options(nostack, preserves_flags)) }
    };
    (neon, f32, prefetch_nta, $p:expr) => {
        unsafe { core::arch::asm!("prfm pldl1strm, [{addr}]", addr = in(reg) ($p as *const u8), options(nostack, preserves_flags)) }
    };
    // NEON non-temporal store: use STNP (store non-temporal pair) for 128-bit
    (neon, f32, stream, $p:expr, $v:expr) => {
        unsafe {
            let ptr = $p as *mut u8;
            let pair: [u64; 2] = std::mem::transmute($v);
            core::arch::asm!(
                "stnp {lo}, {hi}, [{ptr}]",
                lo = in(reg) pair[0],
                hi = in(reg) pair[1],
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    };

    // --- NEON f16 (ARMv8.2 native FP16 when available, else convert) ---
    (neon, f16, lanes) => { 4 };
    (neon, f16, num_regs) => { 32 };
    (neon, f16, zero) => { unsafe { std::arch::aarch64::vdupq_n_f32(0.0) } };
    (neon, f16, splat, $v:expr) => { { use crate::traits::Element as _; unsafe { std::arch::aarch64::vdupq_n_f32($v.to_f32()) } } };
    // Load 4 f16 → 4 f32 via vcvt
    (neon, f16, load, $p:expr) => {
        unsafe {
            let h = std::arch::aarch64::vld1_f16($p as *const std::arch::aarch64::float16_t);
            std::arch::aarch64::vcvt_f32_f16(h)
        }
    };
    (neon, f16, store, $p:expr, $v:expr) => {
        unsafe {
            let h = std::arch::aarch64::vcvt_f16_f32($v);
            std::arch::aarch64::vst1_f16($p as *mut std::arch::aarch64::float16_t, h);
        }
    };
    (neon, f16, optimal_tile_m) => { 8 };
    (neon, f16, optimal_tile_n_vecs) => { 3 };
    (neon, f16, prefetch_distance) => { 128 };
    (neon, f16, has_native_fp16) => { true };
    (neon, f16, has_native_bf16) => { false };
    (neon, f16, loadu, $p:expr) => { $crate::simd_primitive!(neon, f16, load, $p) };
    (neon, f16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(neon, f16, store, $p, $v) };
    (neon, f16, add, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vaddq_f32($a, $b) } };
    (neon, f16, sub, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vsubq_f32($a, $b) } };
    (neon, f16, mul, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmulq_f32($a, $b) } };
    (neon, f16, div, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vdivq_f32($a, $b) } };
    (neon, f16, fma, $a:expr, $b:expr, $c:expr) => { unsafe { std::arch::aarch64::vfmaq_f32($c, $a, $b) } };
    (neon, f16, fnmadd, $a:expr, $b:expr, $c:expr) => { unsafe { std::arch::aarch64::vfmsq_f32($c, $a, $b) } };
    (neon, f16, neg, $a:expr) => { unsafe { std::arch::aarch64::vnegq_f32($a) } };
    (neon, f16, max, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmaxq_f32($a, $b) } };
    (neon, f16, min, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vminq_f32($a, $b) } };
    (neon, f16, reduce_sum, $v:expr) => { unsafe { std::arch::aarch64::vaddvq_f32($v) } };
    (neon, f16, reduce_max, $v:expr) => { unsafe { std::arch::aarch64::vmaxvq_f32($v) } };
    (neon, f16, abs, $a:expr) => { unsafe { std::arch::aarch64::vabsq_f32($a) } };
    (neon, f16, sqrt, $a:expr) => { unsafe { std::arch::aarch64::vsqrtq_f32($a) } };
    (neon, f16, rsqrt, $a:expr) => { $crate::simd_primitive!(neon, f32, rsqrt, $a) };
    (neon, f16, recip, $a:expr) => { $crate::simd_primitive!(neon, f32, recip, $a) };
    (neon, f16, exp, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) };
    (neon, f16, exp_fast, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) };
    (neon, f16, prefetch, $p:expr, $dist:expr) => {
        unsafe { core::arch::asm!("prfm pldl1keep, [{addr}]", addr = in(reg) ($p as *const u8).add($dist), options(nostack, preserves_flags)) }
    };
    (neon, f16, prefetch_t1, $p:expr) => {
        unsafe { core::arch::asm!("prfm pldl2keep, [{addr}]", addr = in(reg) ($p as *const u8), options(nostack, preserves_flags)) }
    };
    (neon, f16, prefetch_nta, $p:expr) => {
        unsafe { core::arch::asm!("prfm pldl1strm, [{addr}]", addr = in(reg) ($p as *const u8), options(nostack, preserves_flags)) }
    };
    // NEON f16 non-temporal store: convert f32→f16 then STNP 64-bit
    (neon, f16, stream, $p:expr, $v:expr) => {
        unsafe {
            let h = std::arch::aarch64::vcvt_f16_f32($v);
            let ptr = $p as *mut u8;
            let bits: u64 = std::mem::transmute(h);
            let lo = bits as u32;
            let hi = (bits >> 32) as u32;
            core::arch::asm!(
                "stnp {lo:w}, {hi:w}, [{ptr}]",
                lo = in(reg) lo,
                hi = in(reg) hi,
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    };

    // --- NEON bf16 (convert via bit-shift) ---
    (neon, bf16, lanes) => { 4 };
    (neon, bf16, num_regs) => { 32 };
    (neon, bf16, zero) => { unsafe { std::arch::aarch64::vdupq_n_f32(0.0) } };
    (neon, bf16, splat, $v:expr) => { { use crate::traits::Element as _; unsafe { std::arch::aarch64::vdupq_n_f32($v.to_f32()) } } };
    (neon, bf16, load, $p:expr) => {
        unsafe {
            // Load 4 u16, zero-extend to 4 u32, shift left 16, reinterpret as f32
            let v = std::arch::aarch64::vld1_u16($p as *const u16);
            let v32 = std::arch::aarch64::vmovl_u16(v);
            let shifted = std::arch::aarch64::vshlq_n_u32(v32, 16);
            std::arch::aarch64::vreinterpretq_f32_u32(shifted)
        }
    };
    (neon, bf16, store, $p:expr, $v:expr) => {
        unsafe {
            let vi = std::arch::aarch64::vreinterpretq_u32_f32($v);
            let shifted = std::arch::aarch64::vshrq_n_u32(vi, 16);
            let narrow = std::arch::aarch64::vmovn_u32(shifted);
            std::arch::aarch64::vst1_u16($p as *mut u16, narrow);
        }
    };
    (neon, bf16, optimal_tile_m) => { 8 };
    (neon, bf16, optimal_tile_n_vecs) => { 3 };
    (neon, bf16, prefetch_distance) => { 128 };
    (neon, bf16, has_native_fp16) => { false };
    (neon, bf16, has_native_bf16) => { false };
    (neon, bf16, loadu, $p:expr) => { $crate::simd_primitive!(neon, bf16, load, $p) };
    (neon, bf16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(neon, bf16, store, $p, $v) };
    (neon, bf16, add, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vaddq_f32($a, $b) } };
    (neon, bf16, sub, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vsubq_f32($a, $b) } };
    (neon, bf16, mul, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmulq_f32($a, $b) } };
    (neon, bf16, div, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vdivq_f32($a, $b) } };
    (neon, bf16, fma, $a:expr, $b:expr, $c:expr) => { unsafe { std::arch::aarch64::vfmaq_f32($c, $a, $b) } };
    (neon, bf16, fnmadd, $a:expr, $b:expr, $c:expr) => { unsafe { std::arch::aarch64::vfmsq_f32($c, $a, $b) } };
    (neon, bf16, neg, $a:expr) => { unsafe { std::arch::aarch64::vnegq_f32($a) } };
    (neon, bf16, max, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmaxq_f32($a, $b) } };
    (neon, bf16, min, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vminq_f32($a, $b) } };
    (neon, bf16, reduce_sum, $v:expr) => { unsafe { std::arch::aarch64::vaddvq_f32($v) } };
    (neon, bf16, reduce_max, $v:expr) => { unsafe { std::arch::aarch64::vmaxvq_f32($v) } };
    (neon, bf16, abs, $a:expr) => { unsafe { std::arch::aarch64::vabsq_f32($a) } };
    (neon, bf16, sqrt, $a:expr) => { unsafe { std::arch::aarch64::vsqrtq_f32($a) } };
    (neon, bf16, rsqrt, $a:expr) => { $crate::simd_primitive!(neon, f32, rsqrt, $a) };
    (neon, bf16, recip, $a:expr) => { $crate::simd_primitive!(neon, f32, recip, $a) };
    (neon, bf16, exp, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) };
    (neon, bf16, exp_fast, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) };
    (neon, bf16, prefetch, $p:expr, $dist:expr) => {
        unsafe { core::arch::asm!("prfm pldl1keep, [{addr}]", addr = in(reg) ($p as *const u8).add($dist), options(nostack, preserves_flags)) }
    };
    (neon, bf16, prefetch_t1, $p:expr) => {
        unsafe { core::arch::asm!("prfm pldl2keep, [{addr}]", addr = in(reg) ($p as *const u8), options(nostack, preserves_flags)) }
    };
    (neon, bf16, prefetch_nta, $p:expr) => {
        unsafe { core::arch::asm!("prfm pldl1strm, [{addr}]", addr = in(reg) ($p as *const u8), options(nostack, preserves_flags)) }
    };
    // NEON bf16 non-temporal store: convert f32→bf16 then STNP 64-bit
    (neon, bf16, stream, $p:expr, $v:expr) => {
        unsafe {
            let vi = std::arch::aarch64::vreinterpretq_u32_f32($v);
            let shifted = std::arch::aarch64::vshrq_n_u32(vi, 16);
            let narrow = std::arch::aarch64::vmovn_u32(shifted);
            let ptr = $p as *mut u8;
            let bits: u64 = std::mem::transmute(narrow);
            let lo = bits as u32;
            let hi = (bits >> 32) as u32;
            core::arch::asm!(
                "stnp {lo:w}, {hi:w}, [{ptr}]",
                lo = in(reg) lo,
                hi = in(reg) hi,
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    };

    // --- Integer Primitives (NEON) ---
    // using int32x4_t / uint32x4_t
    (neon, i32, zero) => { unsafe { std::arch::aarch64::vdupq_n_s32(0) } };
    (neon, i32, splat, $v:expr) => { unsafe { std::arch::aarch64::vdupq_n_s32($v) } };
    (neon, i32, load, $p:expr) => { unsafe { std::arch::aarch64::vld1q_s32($p) } };
    (neon, i32, store, $p:expr, $v:expr) => { unsafe { std::arch::aarch64::vst1q_s32($p, $v) } };

    (neon, i32, and, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vandq_s32($a, $b) } };
    (neon, i32, or, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vorrq_s32($a, $b) } };
    (neon, i32, xor, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::veorq_s32($a, $b) } };
    (neon, i32, add, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vaddq_s32($a, $b) } };
    (neon, i32, shl, $a:expr, $shift:expr) => { unsafe { std::arch::aarch64::shlq_n_s32($a, $shift) } }; // Wait, intrinsics for immediate shift are tricky with rust macros
    // Better use vshlq_n_s32 if shift is const, or vshlq_s32 if variable.
    // Assuming immediate:
    // Rust's std::arch::aarch64 doesn't support immediate argument const generics easily in old versions, but let's try.
    // Actually vshlq_n_s32 expects constant. If $shift is expr, it might fail.
    // Use dynamic shift: vshlq_s32(a, vdupq_n_s32(shift))
    (neon, i32, shl_dyn, $a:expr, $shift:expr) => { unsafe { std::arch::aarch64::vshlq_s32($a, std::arch::aarch64::vdupq_n_s32($shift)) } };
    (neon, i32, shl_23, $a:expr) => { unsafe { std::arch::aarch64::vshlq_n_s32::<23>($a) } };

    (neon, i32, cast_f32, $a:expr) => { unsafe { std::arch::aarch64::vcvtq_f32_s32($a) } };
    (neon, i32, cast_bits_f32, $a:expr) => { unsafe { std::arch::aarch64::vreinterpretq_f32_s32($a) } };
    (neon, f32, round_nearest, $a:expr) => { unsafe { std::arch::aarch64::vrndnq_f32($a) } };
    (neon, f32, cvt_to_i32, $a:expr) => { unsafe { std::arch::aarch64::vcvtq_s32_f32($a) } };
    (neon, f32, cvt_from_i32, $a:expr) => { unsafe { std::arch::aarch64::vcvtq_f32_s32($a) } };
    
    // Shuffle: vqtbl1q_u8 (lookup table)
    (neon, i8, shuffle, $tbl:expr, $idx:expr) => { unsafe { std::arch::aarch64::vqtbl1q_u8($tbl, $idx) } };

    // ========================================================================
    // f16 Support (compute in f32, load/store with conversion)
    // ========================================================================

    // --- Scalar f16 (convert to f32 for all compute) ---
    (scalar, f16, lanes) => { 1 };
    (scalar, f16, num_regs) => { usize::MAX };
    (scalar, f16, optimal_tile_m) => { 1 };
    (scalar, f16, optimal_tile_n_vecs) => { 1 };
    (scalar, f16, prefetch_distance) => { 0 };
    (scalar, f16, has_native_fp16) => { false };
    (scalar, f16, has_native_bf16) => { false };
    (scalar, f16, zero) => { 0.0f32 };
    (scalar, f16, splat, $v:expr) => { { $v.to_f32() } };
    (scalar, f16, load, $p:expr) => { unsafe { (*$p).to_f32() } };
    (scalar, f16, loadu, $p:expr) => { unsafe { (*$p).to_f32() } };
    (scalar, f16, store, $p:expr, $v:expr) => { unsafe { *$p = half::f16::from_f32($v) } };
    (scalar, f16, storeu, $p:expr, $v:expr) => { unsafe { *$p = half::f16::from_f32($v) } };
    (scalar, f16, add, $a:expr, $b:expr) => { $a + $b };
    (scalar, f16, sub, $a:expr, $b:expr) => { $a - $b };
    (scalar, f16, mul, $a:expr, $b:expr) => { $a * $b };
    (scalar, f16, div, $a:expr, $b:expr) => { $a / $b };
    (scalar, f16, fma, $a:expr, $b:expr, $c:expr) => { $c + $a * $b };
    (scalar, f16, fnmadd, $a:expr, $b:expr, $c:expr) => { $c - $a * $b };
    (scalar, f16, neg, $a:expr) => { -$a };
    (scalar, f16, max, $a:expr, $b:expr) => { $a.max($b) };
    (scalar, f16, min, $a:expr, $b:expr) => { $a.min($b) };
    (scalar, f16, reduce_sum, $v:expr) => { $v };
    (scalar, f16, reduce_max, $v:expr) => { $v };
    (scalar, f16, abs, $a:expr) => { $a.abs() };
    (scalar, f16, exp, $a:expr) => { $a.exp() };
    (scalar, f16, exp_fast, $a:expr) => { $a.exp() };
    (scalar, f16, recip, $a:expr) => { 1.0 / $a };
    (scalar, f16, sqrt, $a:expr) => { $a.sqrt() };
    (scalar, f16, rsqrt, $a:expr) => { 1.0 / $a.sqrt() };
    (scalar, f16, prefetch, $p:expr, $dist:expr) => { };
    (scalar, f16, prefetch_t1, $p:expr) => { };
    (scalar, f16, prefetch_nta, $p:expr) => { };
    (scalar, f16, stream, $p:expr, $v:expr) => { unsafe { *$p = half::f16::from_f32($v) } };

    // --- AVX2 f16 (F16C: load f16→f32, compute f32, store f32→f16) ---
    (avx2, f16, lanes) => { 8 };
    (avx2, f16, num_regs) => { 16 };
    (avx2, f16, optimal_tile_m) => { 6 };
    (avx2, f16, optimal_tile_n_vecs) => { 2 };
    (avx2, f16, prefetch_distance) => { 256 };
    (avx2, f16, has_native_fp16) => { false };
    (avx2, f16, has_native_bf16) => { false };
    (avx2, f16, zero) => { std::arch::x86_64::_mm256_setzero_ps() };
    (avx2, f16, splat, $v:expr) => { { std::arch::x86_64::_mm256_set1_ps($v.to_f32()) } };
    // F16C: load 8 f16 (128-bit) → 8 f32 (256-bit)
    (avx2, f16, load, $p:expr) => {
        std::arch::x86_64::_mm256_cvtph_ps(
            std::arch::x86_64::_mm_loadu_si128($p as *const std::arch::x86_64::__m128i)
        )
    };
    (avx2, f16, loadu, $p:expr) => { $crate::simd_primitive!(avx2, f16, load, $p) };
    // F16C: store 8 f32 (256-bit) → 8 f16 (128-bit)
    (avx2, f16, store, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm_storeu_si128(
            $p as *mut std::arch::x86_64::__m128i,
            std::arch::x86_64::_mm256_cvtps_ph($v, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT)
        )
    };
    (avx2, f16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(avx2, f16, store, $p, $v) };
    // All compute ops are f32 (same as avx2 f32)
    (avx2, f16, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_add_ps($a, $b) };
    (avx2, f16, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_sub_ps($a, $b) };
    (avx2, f16, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_mul_ps($a, $b) };
    (avx2, f16, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_div_ps($a, $b) };
    (avx2, f16, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c) };
    (avx2, f16, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm256_fnmadd_ps($a, $b, $c) };
    (avx2, f16, neg, $a:expr) => { std::arch::x86_64::_mm256_sub_ps(std::arch::x86_64::_mm256_setzero_ps(), $a) };
    (avx2, f16, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_max_ps($a, $b) };
    (avx2, f16, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_min_ps($a, $b) };
    (avx2, f16, reduce_sum, $v:expr) => { $crate::simd_primitive!(avx2, f32, reduce_sum, $v) };
    (avx2, f16, reduce_max, $v:expr) => { $crate::simd_primitive!(avx2, f32, reduce_max, $v) };
    (avx2, f16, abs, $a:expr) => { std::arch::x86_64::_mm256_andnot_ps(std::arch::x86_64::_mm256_set1_ps(-0.0), $a) };
    (avx2, f16, exp, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_f32($a) };
    (avx2, f16, exp_fast, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_fast_f32($a) };
    (avx2, f16, sqrt, $a:expr) => { std::arch::x86_64::_mm256_sqrt_ps($a) };
    (avx2, f16, rsqrt, $a:expr) => { $crate::simd_primitive!(avx2, f32, rsqrt, $a) };
    (avx2, f16, recip, $a:expr) => { $crate::simd_primitive!(avx2, f32, recip, $a) };
    (avx2, f16, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx2, f16, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx2, f16, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };
    // Non-temporal store for f16: convert f32→f16 then stream 128-bit
    (avx2, f16, stream, $p:expr, $v:expr) => {
        {
            let h = std::arch::x86_64::_mm256_cvtps_ph($v, std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT);
            std::arch::x86_64::_mm_stream_si128($p as *mut std::arch::x86_64::__m128i, h);
        }
    };
    // Masked load: load `count` f16 elements (0..8), convert to f32, rest are zero
    (avx2, f16, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            // Build 32-bit mask for maskload_ps (sign bit selects)
            let idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let thresh = _mm256_set1_epi32($count as i32);
            let mask32 = _mm256_cmpgt_epi32(thresh, idx);
            // Load 8 f32 via maskload, then we need f16 approach:
            // Load all 8 f16 (128-bit), convert to f32, then zero masked lanes
            let raw = _mm_loadu_si128($p as *const __m128i);
            let converted = _mm256_cvtph_ps(raw);
            // Zero out lanes beyond count using the mask
            _mm256_and_ps(converted, _mm256_castsi256_ps(mask32))
        }
    };
    // Masked store: store `count` f16 elements (0..8) from f32 vector
    (avx2, f16, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            // Convert f32→f16
            let h = _mm256_cvtps_ph($v, _MM_FROUND_TO_NEAREST_INT);
            // Scalar store only the valid lanes
            let mut tmp = [0u16; 8];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, h);
            let dst = $p as *mut u16;
            for i in 0..($count as usize) {
                *dst.add(i) = tmp[i];
            }
        }
    };

    // ========================================================================
    // bf16 Support (compute in f32, load/store with bit-shift conversion)
    // ========================================================================

    // --- Scalar bf16 ---
    (scalar, bf16, lanes) => { 1 };
    (scalar, bf16, num_regs) => { usize::MAX };
    (scalar, bf16, optimal_tile_m) => { 1 };
    (scalar, bf16, optimal_tile_n_vecs) => { 1 };
    (scalar, bf16, prefetch_distance) => { 0 };
    (scalar, bf16, has_native_fp16) => { false };
    (scalar, bf16, has_native_bf16) => { false };
    (scalar, bf16, zero) => { 0.0f32 };
    (scalar, bf16, splat, $v:expr) => { { $v.to_f32() } };
    (scalar, bf16, load, $p:expr) => { unsafe { (*$p).to_f32() } };
    (scalar, bf16, loadu, $p:expr) => { unsafe { (*$p).to_f32() } };
    (scalar, bf16, store, $p:expr, $v:expr) => { unsafe { *$p = half::bf16::from_f32($v) } };
    (scalar, bf16, storeu, $p:expr, $v:expr) => { unsafe { *$p = half::bf16::from_f32($v) } };
    (scalar, bf16, add, $a:expr, $b:expr) => { $a + $b };
    (scalar, bf16, sub, $a:expr, $b:expr) => { $a - $b };
    (scalar, bf16, mul, $a:expr, $b:expr) => { $a * $b };
    (scalar, bf16, div, $a:expr, $b:expr) => { $a / $b };
    (scalar, bf16, fma, $a:expr, $b:expr, $c:expr) => { $c + $a * $b };
    (scalar, bf16, fnmadd, $a:expr, $b:expr, $c:expr) => { $c - $a * $b };
    (scalar, bf16, neg, $a:expr) => { -$a };
    (scalar, bf16, max, $a:expr, $b:expr) => { $a.max($b) };
    (scalar, bf16, min, $a:expr, $b:expr) => { $a.min($b) };
    (scalar, bf16, reduce_sum, $v:expr) => { $v };
    (scalar, bf16, reduce_max, $v:expr) => { $v };
    (scalar, bf16, abs, $a:expr) => { $a.abs() };
    (scalar, bf16, exp, $a:expr) => { $a.exp() };
    (scalar, bf16, exp_fast, $a:expr) => { $a.exp() };
    (scalar, bf16, recip, $a:expr) => { 1.0 / $a };
    (scalar, bf16, sqrt, $a:expr) => { $a.sqrt() };
    (scalar, bf16, rsqrt, $a:expr) => { 1.0 / $a.sqrt() };
    (scalar, bf16, prefetch, $p:expr, $dist:expr) => { };
    (scalar, bf16, prefetch_t1, $p:expr) => { };
    (scalar, bf16, prefetch_nta, $p:expr) => { };
    (scalar, bf16, stream, $p:expr, $v:expr) => { unsafe { *$p = half::bf16::from_f32($v) } };

    // --- AVX2 bf16 (bit-shift conversion: load bf16→f32, compute f32, store f32→bf16) ---
    (avx2, bf16, lanes) => { 8 };
    (avx2, bf16, num_regs) => { 16 };
    (avx2, bf16, optimal_tile_m) => { 6 };
    (avx2, bf16, optimal_tile_n_vecs) => { 2 };
    (avx2, bf16, prefetch_distance) => { 256 };
    (avx2, bf16, has_native_fp16) => { false };
    (avx2, bf16, has_native_bf16) => { false };
    (avx2, bf16, zero) => { std::arch::x86_64::_mm256_setzero_ps() };
    (avx2, bf16, splat, $v:expr) => { { std::arch::x86_64::_mm256_set1_ps($v.to_f32()) } };
    // bf16→f32: load 8 u16, zero-extend to 8 u32, shift left 16 → reinterpret as f32
    (avx2, bf16, load, $p:expr) => {
        {
            let v128 = std::arch::x86_64::_mm_loadu_si128($p as *const std::arch::x86_64::__m128i);
            let v256 = std::arch::x86_64::_mm256_cvtepu16_epi32(v128);
            let shifted = std::arch::x86_64::_mm256_slli_epi32(v256, 16);
            std::arch::x86_64::_mm256_castsi256_ps(shifted)
        }
    };
    (avx2, bf16, loadu, $p:expr) => { $crate::simd_primitive!(avx2, bf16, load, $p) };
    // f32→bf16: truncate (shift right 16), pack to 8 u16, store 128-bit
    (avx2, bf16, store, $p:expr, $v:expr) => {
        {
            let vi = std::arch::x86_64::_mm256_castps_si256($v);
            let shifted = std::arch::x86_64::_mm256_srli_epi32(vi, 16);
            let lo = std::arch::x86_64::_mm256_castsi256_si128(shifted);
            let hi = std::arch::x86_64::_mm256_extracti128_si256(shifted, 1);
            let packed = std::arch::x86_64::_mm_packus_epi32(lo, hi);
            std::arch::x86_64::_mm_storeu_si128($p as *mut std::arch::x86_64::__m128i, packed)
        }
    };
    (avx2, bf16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(avx2, bf16, store, $p, $v) };
    // All compute ops are f32 (same as avx2 f32)
    (avx2, bf16, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_add_ps($a, $b) };
    (avx2, bf16, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_sub_ps($a, $b) };
    (avx2, bf16, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_mul_ps($a, $b) };
    (avx2, bf16, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_div_ps($a, $b) };
    (avx2, bf16, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c) };
    (avx2, bf16, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm256_fnmadd_ps($a, $b, $c) };
    (avx2, bf16, neg, $a:expr) => { std::arch::x86_64::_mm256_sub_ps(std::arch::x86_64::_mm256_setzero_ps(), $a) };
    (avx2, bf16, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_max_ps($a, $b) };
    (avx2, bf16, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_min_ps($a, $b) };
    (avx2, bf16, reduce_sum, $v:expr) => { $crate::simd_primitive!(avx2, f32, reduce_sum, $v) };
    (avx2, bf16, reduce_max, $v:expr) => { $crate::simd_primitive!(avx2, f32, reduce_max, $v) };
    (avx2, bf16, abs, $a:expr) => { std::arch::x86_64::_mm256_andnot_ps(std::arch::x86_64::_mm256_set1_ps(-0.0), $a) };
    (avx2, bf16, exp, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_f32($a) };
    (avx2, bf16, exp_fast, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_fast_f32($a) };
    (avx2, bf16, sqrt, $a:expr) => { std::arch::x86_64::_mm256_sqrt_ps($a) };
    (avx2, bf16, rsqrt, $a:expr) => { $crate::simd_primitive!(avx2, f32, rsqrt, $a) };
    (avx2, bf16, recip, $a:expr) => { $crate::simd_primitive!(avx2, f32, recip, $a) };
    (avx2, bf16, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx2, bf16, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx2, bf16, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };
    // Non-temporal store for bf16: convert f32→bf16 then stream 128-bit
    (avx2, bf16, stream, $p:expr, $v:expr) => {
        {
            let vi = std::arch::x86_64::_mm256_castps_si256($v);
            let shifted = std::arch::x86_64::_mm256_srli_epi32(vi, 16);
            let lo = std::arch::x86_64::_mm256_castsi256_si128(shifted);
            let hi = std::arch::x86_64::_mm256_extracti128_si256(shifted, 1);
            let packed = std::arch::x86_64::_mm_packus_epi32(lo, hi);
            std::arch::x86_64::_mm_stream_si128($p as *mut std::arch::x86_64::__m128i, packed);
        }
    };
    // Masked load: load `count` bf16 elements (0..8), convert to f32, rest are zero
    (avx2, bf16, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let thresh = _mm256_set1_epi32($count as i32);
            let mask32 = _mm256_cmpgt_epi32(thresh, idx);
            // Load 8 bf16 (128-bit), zero-extend to 8 u32 (256-bit), shift left 16
            let raw = _mm_loadu_si128($p as *const __m128i);
            let extended = _mm256_cvtepu16_epi32(raw);
            let shifted = _mm256_slli_epi32(extended, 16);
            let as_f32 = _mm256_castsi256_ps(shifted);
            // Zero out lanes beyond count
            _mm256_and_ps(as_f32, _mm256_castsi256_ps(mask32))
        }
    };
    // Masked store: store `count` bf16 elements (0..8) from f32 vector
    (avx2, bf16, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            // f32→bf16: shift right 16, pack to u16
            let vi = _mm256_castps_si256($v);
            let shifted = _mm256_srli_epi32(vi, 16);
            let lo = _mm256_castsi256_si128(shifted);
            let hi = _mm256_extracti128_si256(shifted, 1);
            let packed = _mm_packus_epi32(lo, hi);
            // Scalar store only the valid lanes
            let mut tmp = [0u16; 8];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, packed);
            let dst = $p as *mut u16;
            for i in 0..($count as usize) {
                *dst.add(i) = tmp[i];
            }
        }
    };

    // ========================================================================
    // AVX-512 FP16 Native Implementation (Sapphire Rapids+)
    //
    // Uses native __m512h intrinsics via core::f16 + stdarch_x86_avx512_f16.
    // Public API uses half::f16; we transmute pointers at load/store boundaries.
    // half::f16 and core::f16 are both IEEE 754 binary16 — same memory layout.
    //
    // Key difference from avx512+f16 (F16C path):
    //   - SIMD type: __m512h (not __m512)
    //   - Lanes: 32 (not 16) — 2x throughput
    //   - All arithmetic native f16 (no f32 conversion overhead)
    // ========================================================================

    // --- Architecture Constants ---
    (avx512fp16, f16, lanes) => { 32 };
    (avx512fp16, f16, num_regs) => { 32 };
    (avx512fp16, f16, optimal_tile_m) => { 14 };
    (avx512fp16, f16, optimal_tile_n_vecs) => { 2 };
    (avx512fp16, f16, prefetch_distance) => { 512 };
    (avx512fp16, f16, has_native_fp16) => { true };
    (avx512fp16, f16, has_native_bf16) => { false };

    // --- Zero / Splat ---
    (avx512fp16, f16, zero) => { std::arch::x86_64::_mm512_setzero_ph() };
    // _mm512_set1_ph takes primitive f16; transmute from half::f16 (same layout)
    (avx512fp16, f16, splat, $v:expr) => {
        {
            let bits: u16 = ($v).to_bits();
            let cf16: $crate::prim_f16 = $crate::prim_f16::from_bits(bits);
            std::arch::x86_64::_mm512_set1_ph(cf16)
        }
    };

    // --- Load / Store ---
    // _mm512_loadu_ph takes *const primitive f16; cast half::f16 ptr
    (avx512fp16, f16, load, $p:expr) => {
        std::arch::x86_64::_mm512_loadu_ph($p as *const $crate::prim_f16)
    };
    (avx512fp16, f16, loadu, $p:expr) => { $crate::simd_primitive!(avx512fp16, f16, load, $p) };
    // _mm512_storeu_ph takes *mut primitive f16
    (avx512fp16, f16, store, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm512_storeu_ph($p as *mut $crate::prim_f16, $v)
    };
    (avx512fp16, f16, storeu, $p:expr, $v:expr) => { $crate::simd_primitive!(avx512fp16, f16, store, $p, $v) };

    // --- Arithmetic (native __m512h) ---
    (avx512fp16, f16, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_add_ph($a, $b) };
    (avx512fp16, f16, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_sub_ph($a, $b) };
    (avx512fp16, f16, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_mul_ph($a, $b) };
    (avx512fp16, f16, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_div_ph($a, $b) };
    (avx512fp16, f16, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fmadd_ph($a, $b, $c) };
    (avx512fp16, f16, fnmadd, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fnmadd_ph($a, $b, $c) };
    (avx512fp16, f16, neg, $a:expr) => { std::arch::x86_64::_mm512_sub_ph(std::arch::x86_64::_mm512_setzero_ph(), $a) };
    (avx512fp16, f16, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_max_ph($a, $b) };
    (avx512fp16, f16, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_min_ph($a, $b) };

    // --- Reduce (return f32 for compatibility with operator templates) ---
    // _mm512_reduce_add_ph returns primitive f16; convert to f32 via `as`
    (avx512fp16, f16, reduce_sum, $v:expr) => { std::arch::x86_64::_mm512_reduce_add_ph($v) as f32 };
    // _mm512_reduce_max_ph returns primitive f16; convert to f32 via `as`
    (avx512fp16, f16, reduce_max, $v:expr) => { std::arch::x86_64::_mm512_reduce_max_ph($v) as f32 };

    // --- Math ---
    (avx512fp16, f16, abs, $a:expr) => { std::arch::x86_64::_mm512_abs_ph($a) };
    (avx512fp16, f16, sqrt, $a:expr) => { std::arch::x86_64::_mm512_sqrt_ph($a) };
    // rsqrt: _mm512_rsqrt_ph + Newton-Raphson refinement
    (avx512fp16, f16, rsqrt, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let r = _mm512_rsqrt_ph($a);
            let half_v = _mm512_set1_ph($crate::prim_f16::from_bits(0x3800u16)); // 0.5
            let three = _mm512_set1_ph($crate::prim_f16::from_bits(0x4200u16)); // 3.0
            let ar = _mm512_mul_ph($a, r);
            // r * 0.5 * (3.0 - a*r*r)
            _mm512_mul_ph(_mm512_mul_ph(r, half_v), _mm512_fnmadd_ph(ar, r, three))
        }
    };
    // recip: _mm512_rcp_ph + Newton-Raphson refinement
    (avx512fp16, f16, recip, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let r = _mm512_rcp_ph($a);
            let two = _mm512_set1_ph($crate::prim_f16::from_bits(0x4000u16)); // 2.0
            // r * (2.0 - a*r)
            _mm512_mul_ph(r, _mm512_fnmadd_ph($a, r, two))
        }
    };
    // exp: no native f16 exp — split 32xf16 into 2x16xf32, compute, convert back
    (avx512fp16, f16, exp, $a:expr) => {
        {
            use std::arch::x86_64::*;
            // Split __m512h into low/high __m256h, convert each to __m512 (f32)
            let lo_256h: __m256h = _mm512_castph512_ph256($a);
            let hi_256h: __m256h = {
                // Extract upper 256 bits: cast to __m512 and extract high __m256
                let as_ps: __m512 = _mm512_castph_ps($a);
                let hi_ps: __m256 = _mm256_castpd_ps(_mm512_extractf64x4_pd(
                    _mm512_castps_pd(as_ps), 1
                ));
                std::mem::transmute::<__m256, __m256h>(hi_ps)
            };
            let lo_f32: __m512 = _mm512_cvtxph_ps(lo_256h);
            let hi_f32: __m512 = _mm512_cvtxph_ps(hi_256h);
            // Compute exp in f32
            let exp_lo = $crate::cpu_kernels::avx512::math::avx512_exp_f32(lo_f32);
            let exp_hi = $crate::cpu_kernels::avx512::math::avx512_exp_f32(hi_f32);
            // Convert back to f16
            let lo_ph: __m256h = _mm512_cvtxps_ph(exp_lo);
            let hi_ph: __m256h = _mm512_cvtxps_ph(exp_hi);
            // Combine two __m256h into __m512h
            let lo_512h: __m512h = _mm512_castph256_ph512(lo_ph);
            let hi_as_ps: __m256 = std::mem::transmute::<__m256h, __m256>(hi_ph);
            let lo_as_ps: __m512 = _mm512_castph_ps(lo_512h);
            let combined_ps: __m512 = _mm512_insertf32x8(lo_as_ps, hi_as_ps, 1);
            _mm512_castps_ph(combined_ps)
        }
    };
    // exp_fast: same split strategy but using fast exp in f32
    (avx512fp16, f16, exp_fast, $a:expr) => {
        {
            use std::arch::x86_64::*;
            let lo_256h: __m256h = _mm512_castph512_ph256($a);
            let hi_256h: __m256h = {
                let as_ps: __m512 = _mm512_castph_ps($a);
                let hi_ps: __m256 = _mm256_castpd_ps(_mm512_extractf64x4_pd(
                    _mm512_castps_pd(as_ps), 1
                ));
                std::mem::transmute::<__m256, __m256h>(hi_ps)
            };
            let lo_f32: __m512 = _mm512_cvtxph_ps(lo_256h);
            let hi_f32: __m512 = _mm512_cvtxph_ps(hi_256h);
            let exp_lo = $crate::cpu_kernels::avx512::math::avx512_exp_fast_f32(lo_f32);
            let exp_hi = $crate::cpu_kernels::avx512::math::avx512_exp_fast_f32(hi_f32);
            let lo_ph: __m256h = _mm512_cvtxps_ph(exp_lo);
            let hi_ph: __m256h = _mm512_cvtxps_ph(exp_hi);
            let lo_512h: __m512h = _mm512_castph256_ph512(lo_ph);
            let hi_as_ps: __m256 = std::mem::transmute::<__m256h, __m256>(hi_ph);
            let lo_as_ps: __m512 = _mm512_castph_ps(lo_512h);
            let combined_ps: __m512 = _mm512_insertf32x8(lo_as_ps, hi_as_ps, 1);
            _mm512_castps_ph(combined_ps)
        }
    };

    // --- Prefetch ---
    (avx512fp16, f16, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    (avx512fp16, f16, prefetch_t1, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T1) };
    (avx512fp16, f16, prefetch_nta, $p:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_NTA) };

    // --- Stream store (non-temporal) ---
    (avx512fp16, f16, stream, $p:expr, $v:expr) => {
        {
            use std::arch::x86_64::*;
            let vi: __m512i = _mm512_castph_si512($v);
            _mm512_stream_si512($p as *mut __m512i, vi);
        }
    };

    // --- Masked load (count lanes, rest zero) ---
    (avx512fp16, f16, maskload, $p:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask32 = (1u64.wrapping_shl($count as u32) - 1) as __mmask32;
            let vi = _mm512_maskz_loadu_epi16(mask, $p as *const i16);
            _mm512_castsi512_ph(vi)
        }
    };

    // --- Masked store (count lanes) ---
    (avx512fp16, f16, maskstore, $p:expr, $v:expr, $count:expr) => {
        {
            use std::arch::x86_64::*;
            let mask: __mmask32 = (1u64.wrapping_shl($count as u32) - 1) as __mmask32;
            let vi: __m512i = _mm512_castph_si512($v);
            _mm512_mask_storeu_epi16($p as *mut i16, mask, vi);
        }
    };

}
