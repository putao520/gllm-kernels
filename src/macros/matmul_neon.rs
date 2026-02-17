/// NEON matmul template: 8x12 microkernel (8 rows x 3 vecs of 4 lanes)
#[macro_export]
macro_rules! define_matmul_neon {
    ($elem:ident) => {
        $crate::define_matmul_neon!(@body $elem, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    };
    (@body $elem:ident, [$($R:literal),+]) => {
        const TM_: usize = 10;
        const LANES_: usize = $crate::simd_primitive!(neon, $elem, lanes);
        const TN_: usize = 3 * LANES_;

        /// Cached blocking parameters for this NEON backend.
        #[inline(always)]
        fn _blocking() -> $crate::cache_params::BlockingParams {
            static BP: std::sync::OnceLock<$crate::cache_params::BlockingParams> = std::sync::OnceLock::new();
            *BP.get_or_init(|| $crate::cache_params::blocking_params(
                TM_, 3, LANES_, std::mem::size_of::<$elem>(),
            ))
        }

        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            let kc_max = _blocking().kc;
            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TN_ - 1) / TN_;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let cs = n_strips * kc_max * TN_;
            let mut packed = vec![<$elem as Element>::ZERO; n_chunks * cs];
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let base = ch * cs;
                for (i, ns) in (0..n_size).step_by(TN_).enumerate() {
                    let an = TN_.min(n_size - ns);
                    for k in 0..kc {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add((ks + k) * n_size + ns),
                                packed.as_mut_ptr().add(base + i * kc_max * TN_ + k * TN_),
                                an,
                            );
                        }
                    }
                }
                ks += kc_max; ch += 1;
            }
            packed
        }

        // ── Small-M no-pack path: read B directly from row-major layout ──
        #[inline(always)]
        fn neon_matmul_nopack(a: &[$elem], b: &[$elem], c: &mut [$elem],
                              m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);

            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = c.as_mut_ptr();

            let mut m = 0usize;
            while m + TM_ <= m_size {
                let mut n = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe { paste::paste! {
                        $(
                            let mut [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, zero);
                            let mut [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, zero);
                            let mut [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, zero);
                        )+
                        let mut _k = 0usize;
                        let ku = k_size & !7;
                        while _k < ku {
                            // Prefetch B 16 rows ahead, A 32 elements ahead
                            $crate::simd_primitive!(neon, $elem, prefetch, bp.add((_k + 16) * n_size + n) as *const u8, 0);
                            $($crate::simd_primitive!(neon, $elem, prefetch, ap.add((m + $R) * k_size + _k + 32) as *const u8, 0);)+
                            // k+0
                            let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n));
                            let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_));
                            let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_2, [<c_ $R _2>]);
                            )+
                            // k+1
                            let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 1) * n_size + n));
                            let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 1) * n_size + n + LANES_));
                            let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 1) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 1));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_2, [<c_ $R _2>]);
                            )+
                            // k+2
                            let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 2) * n_size + n));
                            let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 2) * n_size + n + LANES_));
                            let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 2) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 2));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_2, [<c_ $R _2>]);
                            )+
                            // k+3
                            let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 3) * n_size + n));
                            let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 3) * n_size + n + LANES_));
                            let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 3) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 3));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_2, [<c_ $R _2>]);
                            )+
                            // Mid-prefetch: B 20 rows ahead
                            $crate::simd_primitive!(neon, $elem, prefetch, bp.add((_k + 20) * n_size + n) as *const u8, 0);
                            // k+4
                            let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 4) * n_size + n));
                            let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 4) * n_size + n + LANES_));
                            let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 4) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 4));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_2, [<c_ $R _2>]);
                            )+
                            // k+5
                            let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 5) * n_size + n));
                            let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 5) * n_size + n + LANES_));
                            let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 5) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 5));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_2, [<c_ $R _2>]);
                            )+
                            // k+6
                            let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 6) * n_size + n));
                            let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 6) * n_size + n + LANES_));
                            let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 6) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 6));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_2, [<c_ $R _2>]);
                            )+
                            // k+7
                            let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 7) * n_size + n));
                            let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 7) * n_size + n + LANES_));
                            let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 7) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 7));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_2, [<c_ $R _2>]);
                            )+
                            _k += 8;
                        }
                        // Remainder
                        while _k < k_size {
                            let vb0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n));
                            let vb1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_));
                            let vb2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2, [<c_ $R _2>]);
                            )+
                            _k += 1;
                        }
                        $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R _0>]);
                          $crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n + LANES_), [<c_ $R _1>]);
                          $crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n + LANES_ * 2), [<c_ $R _2>]);)+
                    }}
                    n += TN_;
                }
                // N-remainder: LANES-wide
                while n + LANES_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe { paste::paste! {
                        $(
                            let mut [<c_ $R>] = $crate::simd_primitive!(neon, $elem, zero);
                        )+
                        for k in 0..k_size {
                            let vb = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R>] = $crate::simd_primitive!(neon, $elem, fma, va, vb, [<c_ $R>]);
                            )+
                        }
                        $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R>]);)+
                    }}
                    n += LANES_;
                }
                // N-remainder: scalar
                while n < n_size {
                    $(
                    {
                        let mut s = <$elem as Element>::ZERO;
                        for k in 0..k_size { s = <$elem as Element>::mul_add(s, a[(m + $R) * k_size + k], b[k * n_size + n]); }
                        c[(m + $R) * n_size + n] = s;
                    }
                    )+
                    n += 1;
                }
                m += TM_;
            }
            // M-remainder
            while m < m_size {
                let mut n = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut c0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c2 = $crate::simd_primitive!(neon, $elem, zero);
                        for k in 0..k_size {
                            let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add(m * k_size + k));
                            let vb0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n + LANES_));
                            let vb2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n + LANES_ * 2));
                            c0 = $crate::simd_primitive!(neon, $elem, fma, va, vb0, c0);
                            c1 = $crate::simd_primitive!(neon, $elem, fma, va, vb1, c1);
                            c2 = $crate::simd_primitive!(neon, $elem, fma, va, vb2, c2);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n), c0);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n + LANES_), c1);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n + LANES_ * 2), c2);
                    }
                    n += TN_;
                }
                while n + LANES_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut acc = $crate::simd_primitive!(neon, $elem, zero);
                        for k in 0..k_size {
                            let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add(m * k_size + k));
                            let vb = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n));
                            acc = $crate::simd_primitive!(neon, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n), acc);
                    }
                    n += LANES_;
                }
                while n < n_size {
                    let mut s = <$elem as Element>::ZERO;
                    for k in 0..k_size { s = <$elem as Element>::mul_add(s, a[m * k_size + k], b[k * n_size + n]); }
                    c[m * n_size + n] = s;
                    n += 1;
                }
                m += 1;
            }
        }

        // ── Small-M no-pack path with bias ──
        #[inline(always)]
        fn neon_matmul_bias_nopack(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                                    m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = c.as_mut_ptr();
            let biasp = bias.as_ptr();

            let mut m = 0usize;
            while m + TM_ <= m_size {
                let mut n = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe { paste::paste! {
                        $(
                            let mut [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n));
                            let mut [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n + LANES_));
                            let mut [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n + LANES_ * 2));
                        )+
                        let mut _k = 0usize;
                        let ku = k_size & !7;
                        while _k < ku {
                            // Prefetch B 16 rows ahead, A 32 elements ahead
                            $crate::simd_primitive!(neon, $elem, prefetch, bp.add((_k + 16) * n_size + n) as *const u8, 0);
                            $($crate::simd_primitive!(neon, $elem, prefetch, ap.add((m + $R) * k_size + _k + 32) as *const u8, 0);)+
                            // k+0
                            let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n));
                            let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_));
                            let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_2, [<c_ $R _2>]);
                            )+
                            // k+1
                            let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 1) * n_size + n));
                            let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 1) * n_size + n + LANES_));
                            let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 1) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 1));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_2, [<c_ $R _2>]);
                            )+
                            // k+2
                            let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 2) * n_size + n));
                            let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 2) * n_size + n + LANES_));
                            let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 2) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 2));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_2, [<c_ $R _2>]);
                            )+
                            // k+3
                            let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 3) * n_size + n));
                            let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 3) * n_size + n + LANES_));
                            let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 3) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 3));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_2, [<c_ $R _2>]);
                            )+
                            // Mid-prefetch: B 20 rows ahead
                            $crate::simd_primitive!(neon, $elem, prefetch, bp.add((_k + 20) * n_size + n) as *const u8, 0);
                            // k+4
                            let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 4) * n_size + n));
                            let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 4) * n_size + n + LANES_));
                            let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 4) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 4));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_2, [<c_ $R _2>]);
                            )+
                            // k+5
                            let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 5) * n_size + n));
                            let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 5) * n_size + n + LANES_));
                            let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 5) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 5));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_2, [<c_ $R _2>]);
                            )+
                            // k+6
                            let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 6) * n_size + n));
                            let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 6) * n_size + n + LANES_));
                            let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 6) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 6));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_2, [<c_ $R _2>]);
                            )+
                            // k+7
                            let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 7) * n_size + n));
                            let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 7) * n_size + n + LANES_));
                            let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add((_k + 7) * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k + 7));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_2, [<c_ $R _2>]);
                            )+
                            _k += 8;
                        }
                        // Remainder
                        while _k < k_size {
                            let vb0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n));
                            let vb1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_));
                            let vb2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(_k * n_size + n + LANES_ * 2));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + _k));
                                [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1, [<c_ $R _1>]);
                                [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2, [<c_ $R _2>]);
                            )+
                            _k += 1;
                        }
                        $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R _0>]);
                          $crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n + LANES_), [<c_ $R _1>]);
                          $crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n + LANES_ * 2), [<c_ $R _2>]);)+
                    }}
                    n += TN_;
                }
                while n + LANES_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe { paste::paste! {
                        $(
                            let mut [<c_ $R>] = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n));
                        )+
                        for k in 0..k_size {
                            let vb = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n));
                            $(
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R>] = $crate::simd_primitive!(neon, $elem, fma, va, vb, [<c_ $R>]);
                            )+
                        }
                        $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R>]);)+
                    }}
                    n += LANES_;
                }
                while n < n_size {
                    $(
                    {
                        let mut s = bias[n];
                        for k in 0..k_size { s = <$elem as Element>::mul_add(s, a[(m + $R) * k_size + k], b[k * n_size + n]); }
                        c[(m + $R) * n_size + n] = s;
                    }
                    )+
                    n += 1;
                }
                m += TM_;
            }
            while m < m_size {
                let mut n = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut c0 = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n));
                        let mut c1 = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n + LANES_));
                        let mut c2 = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n + LANES_ * 2));
                        for k in 0..k_size {
                            let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add(m * k_size + k));
                            let vb0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n + LANES_));
                            let vb2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n + LANES_ * 2));
                            c0 = $crate::simd_primitive!(neon, $elem, fma, va, vb0, c0);
                            c1 = $crate::simd_primitive!(neon, $elem, fma, va, vb1, c1);
                            c2 = $crate::simd_primitive!(neon, $elem, fma, va, vb2, c2);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n), c0);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n + LANES_), c1);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n + LANES_ * 2), c2);
                    }
                    n += TN_;
                }
                while n + LANES_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut acc = $crate::simd_primitive!(neon, $elem, loadu, biasp.add(n));
                        for k in 0..k_size {
                            let va = $crate::simd_primitive!(neon, $elem, splat, *ap.add(m * k_size + k));
                            let vb = $crate::simd_primitive!(neon, $elem, loadu, bp.add(k * n_size + n));
                            acc = $crate::simd_primitive!(neon, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m * n_size + n), acc);
                    }
                    n += LANES_;
                }
                while n < n_size {
                    let mut s = bias[n];
                    for k in 0..k_size { s = <$elem as Element>::mul_add(s, a[m * k_size + k], b[k * n_size + n]); }
                    c[m * n_size + n] = s;
                    n += 1;
                }
                m += 1;
            }
        }

        const SMALL_M_THRESHOLD_: usize = 4 * TM_;

        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            let kc_max = _blocking().kc;
            if m_size <= SMALL_M_THRESHOLD_ {
                neon_matmul_nopack(a, b, c, m_size, n_size, k_size);
                return;
            }
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TN_ - 1) / TN_;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let cs = n_strips * kc_max * TN_;
            let tp = n_chunks * cs;

            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut pb = WS.with(|c| c.take());
            pb.clear(); pb.resize(tp, <$elem as Element>::ZERO);
            {
                let mut ks = 0usize; let mut ch = 0usize;
                while ks < k_size {
                    let kc = kc_max.min(k_size - ks);
                    let base = ch * cs;
                    for i in 0..n_strips {
                        let ns = i * TN_;
                        let an = TN_.min(n_size.saturating_sub(ns));
                        for k in 0..kc {
                            let d = base + i * kc_max * TN_ + k * TN_;
                            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr().add((ks+k)*n_size+ns), pb.as_mut_ptr().add(d), an); }
                        }
                    }
                    ks += kc_max; ch += 1;
                }
            }

            let cp = c.as_mut_ptr();
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let mut m = 0;
                while m + TM_ <= m_size {
                    let mut n = 0; let mut si = 0;
                    while n + TN_ <= n_size {
                        #[allow(unused_unsafe)]
                        unsafe { paste::paste! {
                            $(
                            let (mut [<c_ $R _0>], mut [<c_ $R _1>], mut [<c_ $R _2>]) = if ch == 0 {(
                                $crate::simd_primitive!(neon, $elem, zero),
                                $crate::simd_primitive!(neon, $elem, zero),
                                $crate::simd_primitive!(neon, $elem, zero),
                            )} else {(
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_*2)),
                            )};
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*kc_max*TN_);
                            // Prefetch C output tile into L1 before K-loop
                            $($crate::simd_primitive!(neon, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const u8, 0);)+
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead + A rows ~128 bytes ahead
                                $crate::simd_primitive!(neon, $elem, prefetch, bp.add(TN_*16) as *const u8, 0);
                                $($crate::simd_primitive!(neon, $elem, prefetch, ac.add(k_size*$R + 32) as *const u8, 0);)+
                                let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_));
                                let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_));
                                let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_+LANES_));
                                let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2));
                                let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2+LANES_));
                                let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3));
                                let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3+LANES_));
                                let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4));
                                let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4+LANES_));
                                let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5));
                                let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5+LANES_));
                                let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6));
                                let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6+LANES_));
                                let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7));
                                let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7+LANES_));
                                let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN_*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_));
                                let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN_); _k += 1;
                            }
                            $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_), [<c_ $R _1>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_*2), [<c_ $R _2>]);)+
                        }}
                        n += TN_; si += 1;
                    }
                    m += TM_;
                }
                ks += kc_max; ch += 1;
            }
            // Remainder N
            let nm = (n_size / TN_) * TN_;
            if nm < n_size {
                let mut n = nm;
                while n + LANES_ <= n_size {
                    for m in 0..m_size {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let mut acc = $crate::simd_primitive!(neon, $elem, zero);
                            let ar = a.as_ptr().add(m*k_size);
                            for k in 0..k_size {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ar.add(k));
                                let vb = $crate::simd_primitive!(neon, $elem, loadu, b.as_ptr().add(k*n_size+n));
                                acc = $crate::simd_primitive!(neon, $elem, fma, va, vb, acc);
                            }
                            $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add(m*n_size+n), acc);
                        }
                    }
                    n += LANES_;
                }
                for m in 0..m_size {
                    for nn in n..n_size {
                        let mut s = <$elem as Element>::ZERO;
                        for k in 0..k_size { s = <$elem as Element>::mul_add(s, a[m*k_size+k], b[k*n_size+nn]); }
                        c[m*n_size+nn] = s;
                    }
                }
            }
            // Remainder M
            let mm = (m_size / TM_) * TM_;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut c0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut ac = a.as_ptr().add(m*k_size);
                        let mut ks2 = 0usize; let mut ci = 0usize;
                        while ks2 < k_size {
                            let kc = kc_max.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+si*kc_max*TN_);
                            for _ in 0..kc {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ac);
                                let v0 = $crate::simd_primitive!(neon, $elem, loadu, bpp);
                                let v1 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_));
                                let v2 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_*2));
                                c0 = $crate::simd_primitive!(neon, $elem, fma, va, v0, c0);
                                c1 = $crate::simd_primitive!(neon, $elem, fma, va, v1, c1);
                                c2 = $crate::simd_primitive!(neon, $elem, fma, va, v2, c2);
                                ac = ac.add(1); bpp = bpp.add(TN_);
                            }
                            ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n), c0);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_), c1);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_*2), c2);
                    }
                    n += TN_; si += 1;
                }
            }
            WS.with(|c| c.set(pb));
        }

        // ── matmul_bias: C = A * B + bias ──────────────────────────────
        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            let kc_max = _blocking().kc;
            if m_size <= SMALL_M_THRESHOLD_ {
                neon_matmul_bias_nopack(a, b, bias, c, m_size, n_size, k_size);
                return;
            }
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TN_ - 1) / TN_;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let cs = n_strips * kc_max * TN_;
            let tp = n_chunks * cs;

            thread_local! { static WS2: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut pb = WS2.with(|c| c.take());
            pb.clear(); pb.resize(tp, <$elem as Element>::ZERO);
            {
                let mut ks = 0usize; let mut ch = 0usize;
                while ks < k_size {
                    let kc = kc_max.min(k_size - ks);
                    let base = ch * cs;
                    for i in 0..n_strips {
                        let ns = i * TN_;
                        let an = TN_.min(n_size.saturating_sub(ns));
                        for k in 0..kc {
                            let d = base + i * kc_max * TN_ + k * TN_;
                            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr().add((ks+k)*n_size+ns), pb.as_mut_ptr().add(d), an); }
                        }
                    }
                    ks += kc_max; ch += 1;
                }
            }

            // Init C with bias
            let cp = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), cp.add(m * n_size), n_size); }
            }

            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let mut m = 0;
                while m + TM_ <= m_size {
                    let mut n = 0; let mut si = 0;
                    while n + TN_ <= n_size {
                        #[allow(unused_unsafe)]
                        unsafe { paste::paste! {
                            $(
                            let (mut [<c_ $R _0>], mut [<c_ $R _1>], mut [<c_ $R _2>]) = (
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_*2)),
                            );
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*kc_max*TN_);
                            // Prefetch C output tile into L1 before K-loop
                            $($crate::simd_primitive!(neon, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const u8, 0);)+
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead + A rows ~128 bytes ahead
                                $crate::simd_primitive!(neon, $elem, prefetch, bp.add(TN_*16) as *const u8, 0);
                                $($crate::simd_primitive!(neon, $elem, prefetch, ac.add(k_size*$R + 32) as *const u8, 0);)+
                                let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_));
                                let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_));
                                let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_+LANES_));
                                let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2));
                                let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2+LANES_));
                                let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3));
                                let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3+LANES_));
                                let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4));
                                let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4+LANES_));
                                let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5));
                                let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5+LANES_));
                                let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6));
                                let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6+LANES_));
                                let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7));
                                let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7+LANES_));
                                let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN_*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_));
                                let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN_); _k += 1;
                            }
                            $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_), [<c_ $R _1>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_*2), [<c_ $R _2>]);)+
                        }}
                        n += TN_; si += 1;
                    }
                    m += TM_;
                }
                ks += kc_max; ch += 1;
            }
            // Remainder N with bias
            let nm = (n_size / TN_) * TN_;
            if nm < n_size {
                let mut n = nm;
                while n + LANES_ <= n_size {
                    for m in 0..m_size {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let mut acc = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n));
                            let ar = a.as_ptr().add(m*k_size);
                            for k in 0..k_size {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ar.add(k));
                                let vb = $crate::simd_primitive!(neon, $elem, loadu, b.as_ptr().add(k*n_size+n));
                                acc = $crate::simd_primitive!(neon, $elem, fma, va, vb, acc);
                            }
                            $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add(m*n_size+n), acc);
                        }
                    }
                    n += LANES_;
                }
                for m in 0..m_size {
                    for nn in n..n_size {
                        let mut s = bias[nn];
                        for k in 0..k_size { s = <$elem as Element>::mul_add(s, a[m*k_size+k], b[k*n_size+nn]); }
                        c[m*n_size+nn] = s;
                    }
                }
            }
            // Remainder M with bias
            let mm = (m_size / TM_) * TM_;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let bp2 = bias.as_ptr().add(n);
                        let mut c0 = $crate::simd_primitive!(neon, $elem, loadu, bp2);
                        let mut c1 = $crate::simd_primitive!(neon, $elem, loadu, bp2.add(LANES_));
                        let mut c2 = $crate::simd_primitive!(neon, $elem, loadu, bp2.add(LANES_*2));
                        let mut ac = a.as_ptr().add(m*k_size);
                        let mut ks2 = 0usize; let mut ci = 0usize;
                        while ks2 < k_size {
                            let kc = kc_max.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+si*kc_max*TN_);
                            for _ in 0..kc {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ac);
                                let v0 = $crate::simd_primitive!(neon, $elem, loadu, bpp);
                                let v1 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_));
                                let v2 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_*2));
                                c0 = $crate::simd_primitive!(neon, $elem, fma, va, v0, c0);
                                c1 = $crate::simd_primitive!(neon, $elem, fma, va, v1, c1);
                                c2 = $crate::simd_primitive!(neon, $elem, fma, va, v2, c2);
                                ac = ac.add(1); bpp = bpp.add(TN_);
                            }
                            ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n), c0);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_), c1);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_*2), c2);
                    }
                    n += TN_; si += 1;
                }
            }
            WS2.with(|c| c.set(pb));
        }

        // ── matmul_prepacked: C = A * packed_B ─────────────────────────
        #[inline(always)]
        pub fn matmul_prepacked(a: &[$elem], pb: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            let kc_max = _blocking().kc;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TN_ - 1) / TN_;
            let cs = n_strips * kc_max * TN_;
            let cp = c.as_mut_ptr();

            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let mut m = 0;
                while m + TM_ <= m_size {
                    let mut n = 0; let mut si = 0;
                    while n + TN_ <= n_size {
                        #[allow(unused_unsafe)]
                        unsafe { paste::paste! {
                            $(
                            let (mut [<c_ $R _0>], mut [<c_ $R _1>], mut [<c_ $R _2>]) = if ch == 0 {(
                                $crate::simd_primitive!(neon, $elem, zero),
                                $crate::simd_primitive!(neon, $elem, zero),
                                $crate::simd_primitive!(neon, $elem, zero),
                            )} else {(
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_*2)),
                            )};
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*kc_max*TN_);
                            // Prefetch C output tile into L1 before K-loop
                            $($crate::simd_primitive!(neon, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const u8, 0);)+
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead + A rows ~128 bytes ahead
                                $crate::simd_primitive!(neon, $elem, prefetch, bp.add(TN_*16) as *const u8, 0);
                                $($crate::simd_primitive!(neon, $elem, prefetch, ac.add(k_size*$R + 32) as *const u8, 0);)+
                                let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_));
                                let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_));
                                let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_+LANES_));
                                let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2));
                                let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2+LANES_));
                                let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*2+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3));
                                let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3+LANES_));
                                let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*3+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4));
                                let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4+LANES_));
                                let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*4+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5));
                                let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5+LANES_));
                                let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*5+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6));
                                let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6+LANES_));
                                let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*6+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7));
                                let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7+LANES_));
                                let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(TN_*7+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN_*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_));
                                let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, bp.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN_); _k += 1;
                            }
                            $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_), [<c_ $R _1>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_*2), [<c_ $R _2>]);)+
                        }}
                        n += TN_; si += 1;
                    }
                    m += TM_;
                }
                ks += kc_max; ch += 1;
            }
            // Remainder N
            let nm = (n_size / TN_) * TN_;
            if nm < n_size {
                let nr = n_size - nm;
                let ls = nm / TN_;
                let mut nos = 0usize;
                while nos + LANES_ <= nr {
                    for m in 0..m_size {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let mut acc = $crate::simd_primitive!(neon, $elem, zero);
                            let mut ac = a.as_ptr().add(m*k_size);
                            let mut ks2 = 0usize; let mut ci = 0usize;
                            while ks2 < k_size {
                                let kc = kc_max.min(k_size-ks2);
                                let mut bpp = pb.as_ptr().add(ci*cs+ls*kc_max*TN_+nos);
                                for _ in 0..kc {
                                    let va = $crate::simd_primitive!(neon, $elem, splat, *ac);
                                    let vb = $crate::simd_primitive!(neon, $elem, loadu, bpp);
                                    acc = $crate::simd_primitive!(neon, $elem, fma, va, vb, acc);
                                    ac = ac.add(1); bpp = bpp.add(TN_);
                                }
                                ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                            }
                            $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add(m*n_size+nm+nos), acc);
                        }
                    }
                    nos += LANES_;
                }
                for m in 0..m_size {
                    for no in nos..nr {
                        let mut s = <$elem as Element>::ZERO;
                        let mut k = 0usize; let mut ci = 0usize;
                        while k < k_size {
                            let kc = kc_max.min(k_size-k);
                            let base = ci*cs+ls*kc_max*TN_;
                            for ki in 0..kc { s = <$elem as Element>::mul_add(s, a[m*k_size+k+ki], pb[base+ki*TN_+no]); }
                            k += kc; ci += 1;
                        }
                        c[m*n_size+nm+no] = s;
                    }
                }
            }
            // Remainder M
            let mm = (m_size / TM_) * TM_;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut c0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut ac = a.as_ptr().add(m*k_size);
                        let mut ks2 = 0usize; let mut ci = 0usize;
                        while ks2 < k_size {
                            let kc = kc_max.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+si*kc_max*TN_);
                            for _ in 0..kc {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ac);
                                let v0 = $crate::simd_primitive!(neon, $elem, loadu, bpp);
                                let v1 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_));
                                let v2 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_*2));
                                c0 = $crate::simd_primitive!(neon, $elem, fma, va, v0, c0);
                                c1 = $crate::simd_primitive!(neon, $elem, fma, va, v1, c1);
                                c2 = $crate::simd_primitive!(neon, $elem, fma, va, v2, c2);
                                ac = ac.add(1); bpp = bpp.add(TN_);
                            }
                            ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n), c0);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_), c1);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_*2), c2);
                    }
                    n += TN_; si += 1;
                }
            }
        }

        // ── matmul_bias_prepacked: C = A * packed_B + bias ─────────────
        #[inline(always)]
        pub fn matmul_bias_prepacked(a: &[$elem], pb: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            let kc_max = _blocking().kc;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TN_ - 1) / TN_;
            let cs = n_strips * kc_max * TN_;
            let cp = c.as_mut_ptr();

            // Init C with bias
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), cp.add(m * n_size), n_size); }
            }

            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let mut m = 0;
                while m + TM_ <= m_size {
                    let mut n = 0; let mut si = 0;
                    while n + TN_ <= n_size {
                        #[allow(unused_unsafe)]
                        unsafe { paste::paste! {
                            $(
                            let (mut [<c_ $R _0>], mut [<c_ $R _1>], mut [<c_ $R _2>]) = (
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_)),
                                $crate::simd_primitive!(neon, $elem, loadu, cp.add((m+$R)*n_size+n+LANES_*2)),
                            );
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bptr = pb.as_ptr().add(ch*cs+si*kc_max*TN_);
                            // Prefetch C output tile into L1 before K-loop
                            $($crate::simd_primitive!(neon, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const u8, 0);)+
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead + A rows ~128 bytes ahead
                                $crate::simd_primitive!(neon, $elem, prefetch, bptr.add(TN_*16) as *const u8, 0);
                                $($crate::simd_primitive!(neon, $elem, prefetch, ac.add(k_size*$R + 32) as *const u8, 0);)+
                                let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr);
                                let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(LANES_));
                                let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb0_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_));
                                let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_+LANES_));
                                let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb1_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*2));
                                let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*2+LANES_));
                                let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*2+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb2_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*3));
                                let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*3+LANES_));
                                let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*3+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb3_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*4));
                                let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*4+LANES_));
                                let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*4+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb4_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*5));
                                let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*5+LANES_));
                                let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*5+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb5_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*6));
                                let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*6+LANES_));
                                let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*6+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb6_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*7));
                                let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*7+LANES_));
                                let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(TN_*7+LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb7_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bptr = bptr.add(TN_*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, bptr);
                                let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(LANES_));
                                let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, bptr.add(LANES_*2));
                                $(
                                        let va = $crate::simd_primitive!(neon, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        [<c_ $R _2>] = $crate::simd_primitive!(neon, $elem, fma, va, vb_2, [<c_ $R _2>]);
                                )+; ac = ac.add(1);
                                bptr = bptr.add(TN_); _k += 1;
                            }
                            $($crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_), [<c_ $R _1>]);
                              $crate::simd_primitive!(neon, $elem, storeu, cp.add((m+$R)*n_size+n+LANES_*2), [<c_ $R _2>]);)+
                        }}
                        n += TN_; si += 1;
                    }
                    m += TM_;
                }
                ks += kc_max; ch += 1;
            }
            // Remainder M with bias
            let mm = (m_size / TM_) * TM_;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN_ <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let bp2 = bias.as_ptr().add(n);
                        let mut c0 = $crate::simd_primitive!(neon, $elem, loadu, bp2);
                        let mut c1 = $crate::simd_primitive!(neon, $elem, loadu, bp2.add(LANES_));
                        let mut c2 = $crate::simd_primitive!(neon, $elem, loadu, bp2.add(LANES_*2));
                        let mut ac = a.as_ptr().add(m*k_size);
                        let mut ks2 = 0usize; let mut ci = 0usize;
                        while ks2 < k_size {
                            let kc = kc_max.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+si*kc_max*TN_);
                            for _ in 0..kc {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *ac);
                                let v0 = $crate::simd_primitive!(neon, $elem, loadu, bpp);
                                let v1 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_));
                                let v2 = $crate::simd_primitive!(neon, $elem, loadu, bpp.add(LANES_*2));
                                c0 = $crate::simd_primitive!(neon, $elem, fma, va, v0, c0);
                                c1 = $crate::simd_primitive!(neon, $elem, fma, va, v1, c1);
                                c2 = $crate::simd_primitive!(neon, $elem, fma, va, v2, c2);
                                ac = ac.add(1); bpp = bpp.add(TN_);
                            }
                            ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                        }
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n), c0);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_), c1);
                        $crate::simd_primitive!(neon, $elem, storeu, cp.add(m*n_size+n+LANES_*2), c2);
                    }
                    n += TN_; si += 1;
                }
            }
        }
    };
}
