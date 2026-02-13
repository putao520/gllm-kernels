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
    (scalar, f32, neg, $a:expr) => { -$a };
    (scalar, f32, max, $a:expr, $b:expr) => { $a.max($b) };
    (scalar, f32, min, $a:expr, $b:expr) => { $a.min($b) };
    (scalar, f32, reduce_sum, $v:expr) => { $v };
    (scalar, f32, reduce_max, $v:expr) => { $v };
    (scalar, f32, abs, $a:expr) => { $a.abs() };
    (scalar, f32, exp, $a:expr) => { $a.exp() };
    (scalar, f32, recip, $a:expr) => { 1.0 / $a };
    (scalar, f32, sqrt, $a:expr) => { $a.sqrt() };
    (scalar, f32, rsqrt, $a:expr) => { 1.0 / $a.sqrt() };
    (scalar, f32, prefetch, $p:expr, $dist:expr) => { };

    // --- Integer Primitives (Scalar Fallback) ---
    (scalar, i32, splat, $v:expr) => { $v };
    (scalar, i32, load, $p:expr) => { unsafe { *($p as *const i32) } };
    (scalar, i32, and, $a:expr, $b:expr) => { $a & $b };
    (scalar, i32, or, $a:expr, $b:expr) => { $a | $b };
    (scalar, i32, xor, $a:expr, $b:expr) => { $a ^ $b };
    (scalar, i32, shl, $a:expr, $shift:expr) => { $a << $shift };
    (scalar, i32, shr, $a:expr, $shift:expr) => { $a >> $shift }; // arithmetic shift
    (scalar, i32, cast_f32, $a:expr) => { $a as f32 };
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
    (avx2, f32, rsqrt, $a:expr) => { std::arch::x86_64::_mm256_rsqrt_ps($a) };
    (avx2, f32, recip, $a:expr) => { std::arch::x86_64::_mm256_rcp_ps($a) };
    // EXP
    (avx2, f32, exp, $a:expr) => { $crate::cpu_kernels::avx2::math::avx2_exp_f32($a) };

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

    (avx2, f32, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };

    // --- Integer Primitives (AVX2) ---
    // Note: We use __m256i for integer vectors.
    (avx2, i32, zero) => { std::arch::x86_64::_mm256_setzero_si256() };
    (avx2, i32, splat, $v:expr) => { std::arch::x86_64::_mm256_set1_epi32($v) };
    (avx2, i32, load, $p:expr) => { std::arch::x86_64::_mm256_loadu_si256($p as *const std::arch::x86_64::__m256i) };
    (avx2, i32, store, $p:expr, $v:expr) => { std::arch::x86_64::_mm256_storeu_si256($p as *mut std::arch::x86_64::__m256i, $v) };
    
    (avx2, i32, and, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_and_si256($a, $b) };
    (avx2, i32, or, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_or_si256($a, $b) };
    (avx2, i32, xor, $a:expr, $b:expr) => { std::arch::x86_64::_mm256_xor_si256($a, $b) };
    (avx2, i32, shl, $a:expr, $shift:expr) => { std::arch::x86_64::_mm256_slli_epi32($a, $shift) };
    (avx2, i32, shr, $a:expr, $shift:expr) => { std::arch::x86_64::_mm256_srai_epi32($a, $shift) }; // arithmetic right shift
    (avx2, i32, shr_u, $a:expr, $shift:expr) => { std::arch::x86_64::_mm256_srli_epi32($a, $shift) }; // logical right shift

    (avx2, i32, cast_f32, $a:expr) => { std::arch::x86_64::_mm256_cvtepi32_ps($a) };
    
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
    (avx512, f32, add, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_add_ps($a, $b) };
    (avx512, f32, sub, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_sub_ps($a, $b) };
    (avx512, f32, mul, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_mul_ps($a, $b) };
    (avx512, f32, div, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_div_ps($a, $b) };
    (avx512, f32, fma, $a:expr, $b:expr, $c:expr) => { std::arch::x86_64::_mm512_fmadd_ps($a, $b, $c) };
    (avx512, f32, neg, $a:expr) => { std::arch::x86_64::_mm512_sub_ps(std::arch::x86_64::_mm512_setzero_ps(), $a) };
    (avx512, f32, max, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_max_ps($a, $b) };
    (avx512, f32, min, $a:expr, $b:expr) => { std::arch::x86_64::_mm512_min_ps($a, $b) };

    // Reduce sum: native reduce
    (avx512, f32, reduce_sum, $v:expr) => { std::arch::x86_64::_mm512_reduce_add_ps($v) };
    // Reduce max: native reduce
    (avx512, f32, reduce_max, $v:expr) => { std::arch::x86_64::_mm512_reduce_max_ps($v) };

    (avx512, f32, abs, $a:expr) => { std::arch::x86_64::_mm512_abs_ps($a) };
    (avx512, f32, sqrt, $a:expr) => { std::arch::x86_64::_mm512_sqrt_ps($a) };
    (avx512, f32, rsqrt, $a:expr) => { std::arch::x86_64::_mm512_rsqrt14_ps($a) };
    (avx512, f32, recip, $a:expr) => { std::arch::x86_64::_mm512_rcp14_ps($a) };

    // EXP: use AVX2 exp on lower/upper halves (no native 512-bit exp)
    (avx512, f32, exp, $a:expr) => {
        {
            let lo = std::arch::x86_64::_mm512_castps512_ps256($a);
            let hi = std::arch::x86_64::_mm512_extractf32x8_ps($a, 1);
            let exp_lo = $crate::cpu_kernels::avx2::math::avx2_exp_f32(lo);
            let exp_hi = $crate::cpu_kernels::avx2::math::avx2_exp_f32(hi);
            let mut result = std::arch::x86_64::_mm512_castps256_ps512(exp_lo);
            result = std::arch::x86_64::_mm512_insertf32x8(result, exp_hi, 1);
            result
        }
    };

    (avx512, f32, prefetch, $p:expr, $dist:expr) => { std::arch::x86_64::_mm_prefetch($p as *const i8, std::arch::x86_64::_MM_HINT_T0) };


    // ========================================================================
    // NEON Implementation (aarch64)
    // ========================================================================

    // --- f32 Architecture Constants ---
    (neon, f32, num_regs) => { 32 };
    (neon, f32, optimal_tile_m) => { 8 }; // TBD
    (neon, f32, optimal_tile_n_vecs) => { 2 };
    (neon, f32, prefetch_distance) => { 0 }; // manual prefetch often tricky
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
    (neon, f32, neg, $a:expr) => { unsafe { std::arch::aarch64::vnegq_f32($a) } };
    (neon, f32, max, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vmaxq_f32($a, $b) } };
    (neon, f32, min, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vminq_f32($a, $b) } };
    
    // Reduce sum
    (neon, f32, reduce_sum, $v:expr) => { unsafe { std::arch::aarch64::vaddvq_f32($v) } };
    (neon, f32, reduce_max, $v:expr) => { unsafe { std::arch::aarch64::vmaxvq_f32($v) } };
    (neon, f32, abs, $a:expr) => { unsafe { std::arch::aarch64::vabsq_f32($a) } };
    
    // Ops
    (neon, f32, sqrt, $a:expr) => { unsafe { std::arch::aarch64::vsqrtq_f32($a) } };
    (neon, f32, rsqrt, $a:expr) => { unsafe { std::arch::aarch64::vrsqrteq_f32($a) } }; // Estimate?
    (neon, f32, recip, $a:expr) => { unsafe { std::arch::aarch64::vrecpeq_f32($a) } }; // Estimate?
    
    // EXP
    (neon, f32, exp, $a:expr) => { $crate::cpu_kernels::neon::math::exp_ps($a) };

    (neon, f32, prefetch, $p:expr, $dist:expr) => { };

    // --- Integer Primitives (NEON) ---
    // using int32x4_t / uint32x4_t
    (neon, i32, zero) => { unsafe { std::arch::aarch64::vdupq_n_s32(0) } };
    (neon, i32, splat, $v:expr) => { unsafe { std::arch::aarch64::vdupq_n_s32($v) } };
    (neon, i32, load, $p:expr) => { unsafe { std::arch::aarch64::vld1q_s32($p) } };
    (neon, i32, store, $p:expr, $v:expr) => { unsafe { std::arch::aarch64::vst1q_s32($p, $v) } };

    (neon, i32, and, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vandq_s32($a, $b) } };
    (neon, i32, or, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::vorrq_s32($a, $b) } };
    (neon, i32, xor, $a:expr, $b:expr) => { unsafe { std::arch::aarch64::veorq_s32($a, $b) } };
    (neon, i32, shl, $a:expr, $shift:expr) => { unsafe { std::arch::aarch64::shlq_n_s32($a, $shift) } }; // Wait, intrinsics for immediate shift are tricky with rust macros
    // Better use vshlq_n_s32 if shift is const, or vshlq_s32 if variable.
    // Assuming immediate:
    // Rust's std::arch::aarch64 doesn't support immediate argument const generics easily in old versions, but let's try.
    // Actually vshlq_n_s32 expects constant. If $shift is expr, it might fail.
    // Use dynamic shift: vshlq_s32(a, vdupq_n_s32(shift))
    (neon, i32, shl_dyn, $a:expr, $shift:expr) => { unsafe { std::arch::aarch64::vshlq_s32($a, std::arch::aarch64::vdupq_n_s32($shift)) } };
    
    (neon, i32, cast_f32, $a:expr) => { unsafe { std::arch::aarch64::vcvtq_f32_s32($a) } };
    
    // Shuffle: vqtbl1q_u8 (lookup table)
    (neon, i8, shuffle, $tbl:expr, $idx:expr) => { unsafe { std::arch::aarch64::vqtbl1q_u8($tbl, $idx) } };


}
