/// Unified math function templates using simd_primitive! abstraction.
/// Eliminates duplicate exp implementations across AVX2/AVX-512/NEON.

/// Generates the body of a Cephes-style exp(x) approximation for f32 SIMD vectors.
///
/// Algorithm: clamp → t = x * log2e → k = round(t) → Cody-Waite range reduction
///            → degree-5 Horner polynomial → 2^k via IEEE-754 exponent manipulation.
///
/// All ISAs use identical constants and polynomial coefficients.
/// The only difference is the underlying intrinsics, abstracted by simd_primitive!.
///
/// Usage:
/// ```ignore
/// define_exp_f32!(avx2);   // generates avx2_exp_f32
/// define_exp_f32!(avx512); // generates avx512_exp_f32
/// define_exp_f32!(neon);   // generates neon_exp_f32
/// ```
#[macro_export]
macro_rules! define_exp_f32 {
    ($isa:ident) => {
        /// Fast vectorized exp(x) for f32 SIMD vectors.
        /// Cephes-style degree-5 polynomial with Cody-Waite range reduction.
        /// Input clamped to [-88.376, 88.376] to avoid NaN/Inf.
        #[inline(always)]
        pub unsafe fn exp_f32_impl(x: $crate::define_exp_f32!(@vec_type $isa)) -> $crate::define_exp_f32!(@vec_type $isa) {
            // Clamp input to avoid overflow/underflow in 2^k computation
            let x = $crate::simd_primitive!($isa, f32, min,
                $crate::simd_primitive!($isa, f32, max, x,
                    $crate::simd_primitive!($isa, f32, splat, -88.376_f32)),
                $crate::simd_primitive!($isa, f32, splat, 88.376_f32));

            let v_log2e = $crate::simd_primitive!($isa, f32, splat, 1.442_695_04_f32);
            let v_127 = $crate::simd_primitive!($isa, i32, splat, 127);

            // Cody-Waite range reduction: ln2 = c1 + c2 (c1 exact in float)
            let c1 = $crate::simd_primitive!($isa, f32, splat, -0.693_359_375_f32);
            let c2 = $crate::simd_primitive!($isa, f32, splat, 2.121_944_4e-4_f32);

            // k = round(x * log2e)
            let t = $crate::simd_primitive!($isa, f32, mul, x, v_log2e);
            let k = $crate::simd_primitive!($isa, f32, cvt_to_i32,
                $crate::simd_primitive!($isa, f32, round_nearest, t));
            let k_ps = $crate::simd_primitive!($isa, i32, cast_f32, k);

            // y = x - k*ln2 (two-step for precision)
            let mut y = $crate::simd_primitive!($isa, f32, fma, k_ps, c1, x);
            y = $crate::simd_primitive!($isa, f32, fma, k_ps, c2, y);

            // Degree-5 minimax polynomial (Horner's method)
            let p0 = $crate::simd_primitive!($isa, f32, splat, 1.987_569_15E-4_f32);
            let p1 = $crate::simd_primitive!($isa, f32, splat, 1.398_199_950_7E-3_f32);
            let p2 = $crate::simd_primitive!($isa, f32, splat, 8.333_451_907_3E-3_f32);
            let p3 = $crate::simd_primitive!($isa, f32, splat, 4.166_579_589_4E-2_f32);
            let p4 = $crate::simd_primitive!($isa, f32, splat, 1.666_666_545_9E-1_f32);
            let p5 = $crate::simd_primitive!($isa, f32, splat, 5.000_000_120_1E-1_f32);
            let one = $crate::simd_primitive!($isa, f32, splat, 1.0_f32);

            let mut p = p0;
            p = $crate::simd_primitive!($isa, f32, fma, p, y, p1);
            p = $crate::simd_primitive!($isa, f32, fma, p, y, p2);
            p = $crate::simd_primitive!($isa, f32, fma, p, y, p3);
            p = $crate::simd_primitive!($isa, f32, fma, p, y, p4);
            p = $crate::simd_primitive!($isa, f32, fma, p, y, p5);
            p = $crate::simd_primitive!($isa, f32, fma, p, y, one);
            p = $crate::simd_primitive!($isa, f32, fma, p, y, one);

            // 2^k via IEEE-754 exponent manipulation: (k + 127) << 23
            let v_exp = $crate::simd_primitive!($isa, i32, shl_23,
                $crate::simd_primitive!($isa, i32, add, k, v_127));
            let fact = $crate::simd_primitive!($isa, i32, cast_bits_f32, v_exp);

            $crate::simd_primitive!($isa, f32, mul, p, fact)
        }
    };

    // Vector type mapping: ISA → concrete SIMD type
    (@vec_type avx2) => { std::arch::x86_64::__m256 };
    (@vec_type avx512) => { std::arch::x86_64::__m512 };
    (@vec_type neon) => { std::arch::aarch64::float32x4_t };
}

/// Generates `store_f32_as_bf16` and `load_bf16_as_f32` helper functions
/// using `simd_primitive!(avx512, bf16, ...)`. Invoke once per module that
/// needs AVX-512 bf16↔f32 conversion helpers.
#[macro_export]
macro_rules! define_bf16_helpers {
    () => {
        /// Store 16 f32 values as 16 bf16 (truncation via bit-shift).
        #[inline(always)]
        unsafe fn store_f32_as_bf16(ptr: *mut half::bf16, v: std::arch::x86_64::__m512) {
            $crate::simd_primitive!(avx512, bf16, store, ptr, v);
        }

        /// Load 16 bf16 values as f32 (zero-extend + shift).
        #[inline(always)]
        unsafe fn load_bf16_as_f32(ptr: *const half::bf16) -> std::arch::x86_64::__m512 {
            $crate::simd_primitive!(avx512, bf16, load, ptr)
        }
    };
}

/// Helper functions for AVX-512 FP16 native GEMM microkernel.
/// Provides load/store conversions between half::f16 memory and __m512h registers.
#[macro_export]
macro_rules! define_f16_helpers {
    () => {
        /// Store 16 f32 values as 16 f16 (via F16C conversion).
        #[inline(always)]
        unsafe fn store_f32_as_f16(ptr: *mut half::f16, v: std::arch::x86_64::__m512) {
            $crate::simd_primitive!(avx512, f16, store, ptr, v);
        }

        /// Load 16 f16 values as f32 (via F16C conversion).
        #[inline(always)]
        unsafe fn load_f16_as_f32(ptr: *const half::f16) -> std::arch::x86_64::__m512 {
            $crate::simd_primitive!(avx512, f16, load, ptr)
        }
    };
}
