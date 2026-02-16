/// Unified x86 matmul template — generates AVX-512 (TILE_M=14, LANES=16) and AVX2 (TILE_M=6, LANES=8)
/// from a single macro body parameterized by ISA constants and target_feature.
/// Uses paste::paste! for identifier concatenation (c_0_0, c_1_0, etc.)
#[macro_export]
macro_rules! define_matmul_x86 {
    ($isa:ident, $elem:ident, 14, $LANES:literal, $NV:literal, $MC:literal, $($feat:literal),+) => {
        $crate::define_matmul_x86!(@body $isa, $elem, 14, $LANES, $NV, $MC, [$($feat),+],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    };
    ($isa:ident, $elem:ident, 6, $LANES:literal, $NV:literal, $MC:literal, $($feat:literal),+) => {
        $crate::define_matmul_x86!(@body $isa, $elem, 6, $LANES, $NV, $MC, [$($feat),+],
            [0, 1, 2, 3, 4, 5]);
    };

    (@body $isa:ident, $elem:ident, $TM:literal, $LANES:literal, $NV:literal, $MC:literal,
     [$($feat:literal),+], [$($R:literal),+]) => {
        // ── pack_b (public) ────────────────────────────────────────────
        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            const TN: usize = $NV * $LANES;
            const KC: usize = 256;
            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + KC - 1) / KC;
            let cs = n_strips * KC * TN;
            let mut packed = vec![<$elem as Element>::ZERO; n_chunks * cs];
            let mut ks = 0usize;
            let mut ch = 0usize;
            while ks < k_size {
                let kc = KC.min(k_size - ks);
                let base = ch * cs;
                for (i, ns) in (0..n_size).step_by(TN).enumerate() {
                    let an = TN.min(n_size - ns);
                    for k in 0..kc {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add((ks + k) * n_size + ns),
                                packed.as_mut_ptr().add(base + i * KC * TN + k * TN),
                                an,
                            );
                        }
                    }
                }
                ks += KC; ch += 1;
            }
            packed
        }

        // ── matmul: C = A * B ──────────────────────────────────────────
        #[target_feature($(enable = $feat),+)]
        pub unsafe fn x86_matmul_impl(a: &[$elem], b: &[$elem], c: &mut [$elem],
                                       m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            const KC: usize = 256;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + KC - 1) / KC;
            let cs = n_strips * KC * TN;
            let tp = n_chunks * cs;

            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut pb = WS.with(|c| c.take());
            if pb.capacity() < tp { pb.reserve(tp - pb.len()); }
            unsafe { pb.set_len(tp); }
            {
                let mut ks = 0usize; let mut ch = 0usize;
                while ks < k_size {
                    let kc = KC.min(k_size - ks);
                    let base = ch * cs;
                    for i in 0..n_strips {
                        let ns = i * TN;
                        let an = TN.min(n_size.saturating_sub(ns));
                        for k in 0..kc {
                            let d = base + i * KC * TN + k * TN;
                            std::ptr::copy_nonoverlapping(b.as_ptr().add((ks+k)*n_size+ns), pb.as_mut_ptr().add(d), an);
                        }
                    }
                    ks += KC; ch += 1;
                }
            }

            let cp = c.as_mut_ptr();
            const MC: usize = $MC;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = KC.min(k_size - ks);
                let mut m_block = 0usize;
                while m_block < m_size {
                let m_end = (m_block + MC).min(m_size);
                let mut m = m_block;
                while m + TM <= m_end {
                    let mut n = 0; let mut si = 0;
                    while n + TN <= n_size {
                        unsafe { paste::paste! {
                            // Init accumulators
                            $(
                                let (mut [<c_ $R _0>], mut [<c_ $R _1>]) = if ch == 0 {(
                                    $crate::simd_primitive!($isa, $elem, zero),
                                    $crate::simd_primitive!($isa, $elem, zero),
                                )} else {(
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n+$LANES)),
                                )};
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*KC*TN);
                            // Prefetch C output tile into L1 before K-loop
                            $($crate::simd_primitive!($isa, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const i8, 0);)+
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead
                                $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                // Prefetch A rows ~256 bytes ahead (once per 8-iter block)
                                $($crate::simd_primitive!($isa, $elem, prefetch, ac.add(k_size*$R + 64) as *const i8, 0);)+
                                let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN); _k += 1;
                            }
                            // Store
                            $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                        }}
                        n += TN; si += 1;
                    }
                    m += TM;
                }
                m_block = m_end;
                } // MC block
                ks += KC; ch += 1;
            }
            // Remainder N
            let nm = (n_size / TN) * TN;
            if nm < n_size {
                let mut n = nm;
                while n + $LANES <= n_size {
                    for m in 0..m_size { unsafe {
                        let mut acc = $crate::simd_primitive!($isa, $elem, zero);
                        let ar = a.as_ptr().add(m*k_size);
                        for k in 0..k_size {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ar.add(k));
                            let vb = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k*n_size+n));
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(m*n_size+n), acc);
                    }}
                    n += $LANES;
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
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN <= n_size { unsafe {
                    let mut c0 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut ac = a.as_ptr().add(m*k_size);
                    let mut ks2 = 0usize; let mut ci = 0usize;
                    while ks2 < k_size {
                        let kc = KC.min(k_size-ks2);
                        let mut bp = pb.as_ptr().add(ci*cs+si*KC*TN);
                        for _ in 0..kc {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ac);
                            let v0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                            let v1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                            c0 = $crate::simd_primitive!($isa, $elem, fma, va, v0, c0);
                            c1 = $crate::simd_primitive!($isa, $elem, fma, va, v1, c1);
                            ac = ac.add(1); bp = bp.add(TN);
                        }
                        ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n+$LANES), c1);
                    n += TN; si += 1;
                }}
            }
            WS.with(|c| c.set(pb));
        }

        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { x86_matmul_impl(a, b, c, m_size, n_size, k_size); }
        }

        // ── matmul_bias: C = A * B + bias ──────────────────────────────
        #[target_feature($(enable = $feat),+)]
        pub unsafe fn x86_matmul_bias_impl(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                                            m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            const KC: usize = 256;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + KC - 1) / KC;
            let cs = n_strips * KC * TN;
            let tp = n_chunks * cs;

            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut pb = WS.with(|c| c.take());
            if pb.capacity() < tp { pb.reserve(tp - pb.len()); }
            unsafe { pb.set_len(tp); }
            {
                let mut ks = 0usize; let mut ch = 0usize;
                while ks < k_size {
                    let kc = KC.min(k_size - ks);
                    let base = ch * cs;
                    for i in 0..n_strips {
                        let ns = i * TN;
                        let an = TN.min(n_size.saturating_sub(ns));
                        for k in 0..kc {
                            let d = base + i * KC * TN + k * TN;
                            std::ptr::copy_nonoverlapping(b.as_ptr().add((ks+k)*n_size+ns), pb.as_mut_ptr().add(d), an);
                        }
                    }
                    ks += KC; ch += 1;
                }
            }

            // Init C with bias
            let cp = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), cp.add(m * n_size), n_size); }
            }

            const MC: usize = $MC;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = KC.min(k_size - ks);
                let mut m_block = 0usize;
                while m_block < m_size {
                let m_end = (m_block + MC).min(m_size);
                let mut m = m_block;
                while m + TM <= m_end {
                    let mut n = 0; let mut si = 0;
                    while n + TN <= n_size {
                        unsafe { paste::paste! {
                            // Load C tile (bias already in C)
                            $(
                                let (mut [<c_ $R _0>], mut [<c_ $R _1>]) = (
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n+$LANES)),
                                );
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*KC*TN);
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead
                                $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                // Prefetch A rows ~256 bytes ahead (once per 8-iter block)
                                $($crate::simd_primitive!($isa, $elem, prefetch, ac.add(k_size*$R + 64) as *const i8, 0);)+
                                let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN); _k += 1;
                            }
                            $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                        }}
                        n += TN; si += 1;
                    }
                    m += TM;
                }
                m_block = m_end;
                } // MC block
                ks += KC; ch += 1;
            }
            // Remainder N with bias
            let nm = (n_size / TN) * TN;
            if nm < n_size {
                let mut n = nm;
                while n + $LANES <= n_size {
                    for m in 0..m_size { unsafe {
                        let mut acc = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(n));
                        let ar = a.as_ptr().add(m*k_size);
                        for k in 0..k_size {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ar.add(k));
                            let vb = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k*n_size+n));
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(m*n_size+n), acc);
                    }}
                    n += $LANES;
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
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN <= n_size { unsafe {
                    let bp2 = bias.as_ptr().add(n);
                    let mut c0 = $crate::simd_primitive!($isa, $elem, loadu, bp2);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, loadu, bp2.add($LANES));
                    let mut ac = a.as_ptr().add(m*k_size);
                    let mut ks2 = 0usize; let mut ci = 0usize;
                    while ks2 < k_size {
                        let kc = KC.min(k_size-ks2);
                        let mut bpp = pb.as_ptr().add(ci*cs+si*KC*TN);
                        for _ in 0..kc {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ac);
                            let v0 = $crate::simd_primitive!($isa, $elem, loadu, bpp);
                            let v1 = $crate::simd_primitive!($isa, $elem, loadu, bpp.add($LANES));
                            c0 = $crate::simd_primitive!($isa, $elem, fma, va, v0, c0);
                            c1 = $crate::simd_primitive!($isa, $elem, fma, va, v1, c1);
                            ac = ac.add(1); bpp = bpp.add(TN);
                        }
                        ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n+$LANES), c1);
                    n += TN; si += 1;
                }}
            }
            WS.with(|c| c.set(pb));
        }

        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { x86_matmul_bias_impl(a, b, bias, c, m_size, n_size, k_size); }
        }

        // ── matmul_prepacked: C = A * packed_B ─────────────────────────
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_prepacked_impl(a: &[$elem], pb: &[$elem], c: &mut [$elem],
                                             m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            const KC: usize = 256;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TN - 1) / TN;
            let cs = n_strips * KC * TN;
            let cp = c.as_mut_ptr();

            const MC: usize = $MC;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = KC.min(k_size - ks);
                let mut m_block = 0usize;
                while m_block < m_size {
                let m_end = (m_block + MC).min(m_size);
                let mut m = m_block;
                while m + TM <= m_end {
                    let mut n = 0; let mut si = 0;
                    while n + TN <= n_size {
                        unsafe { paste::paste! {
                            $(
                                let (mut [<c_ $R _0>], mut [<c_ $R _1>]) = if ch == 0 {(
                                    $crate::simd_primitive!($isa, $elem, zero),
                                    $crate::simd_primitive!($isa, $elem, zero),
                                )} else {(
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n+$LANES)),
                                )};
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*KC*TN);
                            // Prefetch C output tile into L1 before K-loop
                            $($crate::simd_primitive!($isa, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const i8, 0);)+
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead
                                $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                // Prefetch A rows ~256 bytes ahead (once per 8-iter block)
                                $($crate::simd_primitive!($isa, $elem, prefetch, ac.add(k_size*$R + 64) as *const i8, 0);)+
                                let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN); _k += 1;
                            }
                            $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                        }}
                        n += TN; si += 1;
                    }
                    m += TM;
                }
                m_block = m_end;
                } // MC block
                ks += KC; ch += 1;
            }
            // Remainder N
            let nm = (n_size / TN) * TN;
            if nm < n_size {
                let nr = n_size - nm;
                let ls = nm / TN;
                let mut nos = 0usize;
                while nos + $LANES <= nr {
                    for m in 0..m_size { unsafe {
                        let mut acc = $crate::simd_primitive!($isa, $elem, zero);
                        let mut ac = a.as_ptr().add(m*k_size);
                        let mut ks2 = 0usize; let mut ci = 0usize;
                        while ks2 < k_size {
                            let kc = KC.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+ls*KC*TN+nos);
                            for _ in 0..kc {
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ac);
                                let vb = $crate::simd_primitive!($isa, $elem, loadu, bpp);
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                                ac = ac.add(1); bpp = bpp.add(TN);
                            }
                            ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(m*n_size+nm+nos), acc);
                    }}
                    nos += $LANES;
                }
                for m in 0..m_size {
                    for no in nos..nr {
                        let mut s = <$elem as Element>::ZERO;
                        let mut k = 0usize; let mut ci = 0usize;
                        while k < k_size {
                            let kc = KC.min(k_size-k);
                            let base = ci*cs+ls*KC*TN;
                            for ki in 0..kc { s = <$elem as Element>::mul_add(s, a[m*k_size+k+ki], pb[base+ki*TN+no]); }
                            k += kc; ci += 1;
                        }
                        c[m*n_size+nm+no] = s;
                    }
                }
            }
            // Remainder M
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN <= n_size { unsafe {
                    let mut c0 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut ac = a.as_ptr().add(m*k_size);
                    let mut ks2 = 0usize; let mut ci = 0usize;
                    while ks2 < k_size {
                        let kc = KC.min(k_size-ks2);
                        let mut bpp = pb.as_ptr().add(ci*cs+si*KC*TN);
                        for _ in 0..kc {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ac);
                            let v0 = $crate::simd_primitive!($isa, $elem, loadu, bpp);
                            let v1 = $crate::simd_primitive!($isa, $elem, loadu, bpp.add($LANES));
                            c0 = $crate::simd_primitive!($isa, $elem, fma, va, v0, c0);
                            c1 = $crate::simd_primitive!($isa, $elem, fma, va, v1, c1);
                            ac = ac.add(1); bpp = bpp.add(TN);
                        }
                        ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n+$LANES), c1);
                    n += TN; si += 1;
                }}
            }
        }

        #[inline(always)]
        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { x86_matmul_prepacked_impl(a, packed_b, c, m_size, n_size, k_size); }
        }

        // ── matmul_bias_prepacked: C = A * packed_B + bias ─────────────
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_bias_prepacked_impl(a: &[$elem], pb: &[$elem], bias: &[$elem], c: &mut [$elem],
                                                  m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            const KC: usize = 256;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TN - 1) / TN;
            let cs = n_strips * KC * TN;
            let cp = c.as_mut_ptr();

            // Init C with bias
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), cp.add(m * n_size), n_size); }
            }

            const MC: usize = $MC;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = KC.min(k_size - ks);
                let mut m_block = 0usize;
                while m_block < m_size {
                let m_end = (m_block + MC).min(m_size);
                let mut m = m_block;
                while m + TM <= m_end {
                    let mut n = 0; let mut si = 0;
                    while n + TN <= n_size {
                        unsafe { paste::paste! {
                            $(
                                let (mut [<c_ $R _0>], mut [<c_ $R _1>]) = (
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                    $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n+$LANES)),
                                );
                            )+
                            let mut ac = a.as_ptr().add(m*k_size+ks);
                            let mut bp = pb.as_ptr().add(ch*cs+si*KC*TN);
                            let mut _k = 0usize;
                            let ku = kc & !7;
                            while _k < ku {
                                // Prefetch B ahead
                                $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                // Prefetch A rows ~256 bytes ahead (once per 8-iter block)
                                $($crate::simd_primitive!($isa, $elem, prefetch, ac.add(k_size*$R + 64) as *const i8, 0);)+
                                let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN*8); _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                $(
                                        let va = $crate::simd_primitive!($isa, $elem, splat, *ac.add(k_size*$R));
                                        [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                        [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                )+; ac = ac.add(1);
                                bp = bp.add(TN); _k += 1;
                            }
                            $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                              $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                        }}
                        n += TN; si += 1;
                    }
                    m += TM;
                }
                m_block = m_end;
                } // MC block
                ks += KC; ch += 1;
            }
            // Remainder N with bias
            let nm = (n_size / TN) * TN;
            if nm < n_size {
                let nr = n_size - nm;
                let ls = nm / TN;
                let mut nos = 0usize;
                while nos + $LANES <= nr {
                    for m in 0..m_size { unsafe {
                        let mut acc = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(nm+nos));
                        let mut ac = a.as_ptr().add(m*k_size);
                        let mut ks2 = 0usize; let mut ci = 0usize;
                        while ks2 < k_size {
                            let kc = KC.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+ls*KC*TN+nos);
                            for _ in 0..kc {
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ac);
                                let vb = $crate::simd_primitive!($isa, $elem, loadu, bpp);
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                                ac = ac.add(1); bpp = bpp.add(TN);
                            }
                            ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(m*n_size+nm+nos), acc);
                    }}
                    nos += $LANES;
                }
                for m in 0..m_size {
                    for no in nos..nr {
                        let mut s = bias[nm+no];
                        let mut k = 0usize; let mut ci = 0usize;
                        while k < k_size {
                            let kc = KC.min(k_size-k);
                            let base = ci*cs+ls*KC*TN;
                            for ki in 0..kc { s = <$elem as Element>::mul_add(s, a[m*k_size+k+ki], pb[base+ki*TN+no]); }
                            k += kc; ci += 1;
                        }
                        c[m*n_size+nm+no] = s;
                    }
                }
            }
            // Remainder M with bias
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN <= n_size { unsafe {
                    let bp2 = bias.as_ptr().add(n);
                    let mut c0 = $crate::simd_primitive!($isa, $elem, loadu, bp2);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, loadu, bp2.add($LANES));
                    let mut ac = a.as_ptr().add(m*k_size);
                    let mut ks2 = 0usize; let mut ci = 0usize;
                    while ks2 < k_size {
                        let kc = KC.min(k_size-ks2);
                        let mut bpp = pb.as_ptr().add(ci*cs+si*KC*TN);
                        for _ in 0..kc {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ac);
                            let v0 = $crate::simd_primitive!($isa, $elem, loadu, bpp);
                            let v1 = $crate::simd_primitive!($isa, $elem, loadu, bpp.add($LANES));
                            c0 = $crate::simd_primitive!($isa, $elem, fma, va, v0, c0);
                            c1 = $crate::simd_primitive!($isa, $elem, fma, va, v1, c1);
                            ac = ac.add(1); bpp = bpp.add(TN);
                        }
                        ks2 += kc; ci += 1; ac = a.as_ptr().add(m*k_size+ks2);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n+$LANES), c1);
                    n += TN; si += 1;
                }}
            }
        }

        #[inline(always)]
        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { x86_matmul_bias_prepacked_impl(a, packed_b, bias, c, m_size, n_size, k_size); }
        }
    };
}
