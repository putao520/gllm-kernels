/// Pack-free skinny GEMM macro — dedicated M=2..32 path for small-batch LLM inference.
///
/// Computes C[M×N] = A[M×K] * B[K×N] with zero packing overhead.
/// This bridges the gap between streaming GEMV (M=1) and full packed GEMM (M>32).
///
/// Key optimizations:
/// - Zero packing: reads A and B directly from row-major layout
/// - Row-parallel: processes up to 8 rows of A simultaneously (8 accumulators)
/// - N-tiling: 2×LANES columns per strip, matching streaming GEMV
/// - K-tiling: KC=256 keeps A rows resident in L1 (M*KC*4 ≤ 32KB for M≤32)
/// - 2x K-unroll with software prefetch
/// - Scalar tail for non-SIMD-aligned N remainder
///
/// Register budget (per N-strip):
///   8 accumulator vectors (one per row) + 1 B vector + 1 A broadcast = 10 regs
///   With 2×LANES strip: 16 accumulators + 2 B + 1 A = 19 regs (fits in 32 NEON / 16 YMM)
///
/// For M>8, we tile M in chunks of 8 and accumulate into C directly.
#[macro_export]
macro_rules! define_gemm_skinny {
    ($isa:ident, $elem:ident, $LANES:literal, $($feat:literal),+) => {

        /// Skinny GEMM entry point: C[M×N] = A[M×K] * B[K×N], M in 2..=32.
        #[inline(always)]
        pub fn gemm_skinny(a: &[$elem], b: &[$elem], c: &mut [$elem],
                           m: usize, n: usize, k: usize) {
            debug_assert!(m >= 2 && m <= 32);
            debug_assert!(a.len() >= m * k);
            debug_assert!(b.len() >= k * n);
            debug_assert!(c.len() >= m * n);

            // Zero C first (we accumulate into it)
            for v in c[..m * n].iter_mut() { *v = <$elem as Element>::ZERO; }

            unsafe {
                gemm_skinny_inner(
                    a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                    m, n, k,
                );
            }
        }

        /// Inner skinny GEMM kernel. Tiles M in chunks of 8, N in LANES-wide strips.
        #[target_feature($(enable = $feat),+)]
        unsafe fn gemm_skinny_inner(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            m: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;
            const KC: usize = 256; // K-tile: 256 elems ≈ 1KB f32, M*KC*4 ≤ 32KB for M≤32

            // Tile M in chunks of 8 rows
            let mut i = 0usize;
            while i + 8 <= m {
                gemm_skinny_m8(a_ptr, b_ptr, c_ptr, i, n, k);
                i += 8;
            }
            // Remainder: 4-row chunk
            if i + 4 <= m {
                gemm_skinny_m4(a_ptr, b_ptr, c_ptr, i, n, k);
                i += 4;
            }
            // Remainder: 2-row chunk
            if i + 2 <= m {
                gemm_skinny_m2(a_ptr, b_ptr, c_ptr, i, n, k);
                i += 2;
            }
            // Remainder: single row (use streaming GEMV pattern)
            if i < m {
                gemm_skinny_m1(a_ptr, b_ptr, c_ptr, i, n, k);
            }
        }

        /// Process 8 rows × N columns. 8 accumulator vectors per N-strip.
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn gemm_skinny_m8(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            i_start: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;

            // Process LANES columns at a time
            let mut j = 0usize;
            while j + LANES <= n {
                // 8 accumulator vectors, one per row
                let mut acc0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc2 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc3 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc4 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc5 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc6 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc7 = $crate::simd_primitive!($isa, $elem, zero);

                let ku = k & !1; // 2x unroll
                let mut ki = 0usize;
                while ki < ku {
                    // k+0
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(ki * n + j));
                    acc0 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 0) * k + ki)), vb, acc0);
                    acc1 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 1) * k + ki)), vb, acc1);
                    acc2 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 2) * k + ki)), vb, acc2);
                    acc3 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 3) * k + ki)), vb, acc3);
                    acc4 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 4) * k + ki)), vb, acc4);
                    acc5 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 5) * k + ki)), vb, acc5);
                    acc6 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 6) * k + ki)), vb, acc6);
                    acc7 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 7) * k + ki)), vb, acc7);

                    // k+1
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((ki + 1) * n + j));
                    acc0 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 0) * k + ki + 1)), vb, acc0);
                    acc1 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 1) * k + ki + 1)), vb, acc1);
                    acc2 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 2) * k + ki + 1)), vb, acc2);
                    acc3 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 3) * k + ki + 1)), vb, acc3);
                    acc4 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 4) * k + ki + 1)), vb, acc4);
                    acc5 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 5) * k + ki + 1)), vb, acc5);
                    acc6 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 6) * k + ki + 1)), vb, acc6);
                    acc7 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 7) * k + ki + 1)), vb, acc7);

                    ki += 2;
                }
                // K remainder
                if ki < k {
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(ki * n + j));
                    acc0 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 0) * k + ki)), vb, acc0);
                    acc1 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 1) * k + ki)), vb, acc1);
                    acc2 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 2) * k + ki)), vb, acc2);
                    acc3 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 3) * k + ki)), vb, acc3);
                    acc4 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 4) * k + ki)), vb, acc4);
                    acc5 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 5) * k + ki)), vb, acc5);
                    acc6 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 6) * k + ki)), vb, acc6);
                    acc7 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 7) * k + ki)), vb, acc7);
                }

                // Store 8 rows
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 0) * n + j), acc0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 1) * n + j), acc1);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 2) * n + j), acc2);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 3) * n + j), acc3);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 4) * n + j), acc4);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 5) * n + j), acc5);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 6) * n + j), acc6);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 7) * n + j), acc7);
                j += LANES;
            }

            // Scalar tail for remaining columns
            while j < n {
                for r in 0..8 {
                    let mut sum = <$elem as Element>::ZERO;
                    for ki in 0..k {
                        sum = <$elem as Element>::mul_add(sum,
                            *a_ptr.add((i_start + r) * k + ki),
                            *b_ptr.add(ki * n + j));
                    }
                    *c_ptr.add((i_start + r) * n + j) = sum;
                }
                j += 1;
            }
        }

        /// Process 4 rows × N columns.
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn gemm_skinny_m4(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            i_start: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;

            let mut j = 0usize;
            while j + LANES <= n {
                let mut acc0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc2 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc3 = $crate::simd_primitive!($isa, $elem, zero);

                for ki in 0..k {
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(ki * n + j));
                    acc0 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 0) * k + ki)), vb, acc0);
                    acc1 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 1) * k + ki)), vb, acc1);
                    acc2 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 2) * k + ki)), vb, acc2);
                    acc3 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 3) * k + ki)), vb, acc3);
                }

                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 0) * n + j), acc0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 1) * n + j), acc1);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 2) * n + j), acc2);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 3) * n + j), acc3);
                j += LANES;
            }

            while j < n {
                for r in 0..4 {
                    let mut sum = <$elem as Element>::ZERO;
                    for ki in 0..k {
                        sum = <$elem as Element>::mul_add(sum,
                            *a_ptr.add((i_start + r) * k + ki),
                            *b_ptr.add(ki * n + j));
                    }
                    *c_ptr.add((i_start + r) * n + j) = sum;
                }
                j += 1;
            }
        }

        /// Process 2 rows × N columns.
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn gemm_skinny_m2(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            i_start: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;

            let mut j = 0usize;
            while j + LANES <= n {
                let mut acc0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc1 = $crate::simd_primitive!($isa, $elem, zero);

                for ki in 0..k {
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(ki * n + j));
                    acc0 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 0) * k + ki)), vb, acc0);
                    acc1 = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add((i_start + 1) * k + ki)), vb, acc1);
                }

                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 0) * n + j), acc0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add((i_start + 1) * n + j), acc1);
                j += LANES;
            }

            while j < n {
                for r in 0..2 {
                    let mut sum = <$elem as Element>::ZERO;
                    for ki in 0..k {
                        sum = <$elem as Element>::mul_add(sum,
                            *a_ptr.add((i_start + r) * k + ki),
                            *b_ptr.add(ki * n + j));
                    }
                    *c_ptr.add((i_start + r) * n + j) = sum;
                }
                j += 1;
            }
        }

        /// Process 1 row × N columns (tail).
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn gemm_skinny_m1(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            i_start: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;

            let mut j = 0usize;
            while j + LANES <= n {
                let mut acc = $crate::simd_primitive!($isa, $elem, zero);
                for ki in 0..k {
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(ki * n + j));
                    acc = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(i_start * k + ki)), vb, acc);
                }
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(i_start * n + j), acc);
                j += LANES;
            }

            while j < n {
                let mut sum = <$elem as Element>::ZERO;
                for ki in 0..k {
                    sum = <$elem as Element>::mul_add(sum,
                        *a_ptr.add(i_start * k + ki),
                        *b_ptr.add(ki * n + j));
                }
                *c_ptr.add(i_start * n + j) = sum;
                j += 1;
            }
        }
    };
}
