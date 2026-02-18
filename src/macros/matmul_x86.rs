/// Unified x86 matmul template — generates AVX-512 (TILE_M=14, LANES=16) and AVX2 (TILE_M=6, LANES=8)
/// from a single macro body parameterized by ISA constants and target_feature.
/// Uses paste::paste! for identifier concatenation (c_0_0, c_1_0, etc.)
#[macro_export]
macro_rules! define_matmul_x86 {
    ($isa:ident, $elem:ident, 16, $LANES:literal, $NV:literal, $MC:literal, $($feat:literal),+) => {
        $crate::define_matmul_x86!(@body $isa, $elem, 16, $LANES, $NV, $MC, [$($feat),+],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    };
    ($isa:ident, $elem:ident, 14, $LANES:literal, $NV:literal, $MC:literal, $($feat:literal),+) => {
        $crate::define_matmul_x86!(@body $isa, $elem, 14, $LANES, $NV, $MC, [$($feat),+],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    };
    ($isa:ident, $elem:ident, 7, $LANES:literal, $NV:literal, $MC:literal, $($feat:literal),+) => {
        $crate::define_matmul_x86!(@body $isa, $elem, 7, $LANES, $NV, $MC, [$($feat),+],
            [0, 1, 2, 3, 4, 5, 6]);
    };
    ($isa:ident, $elem:ident, 6, $LANES:literal, $NV:literal, $MC:literal, $($feat:literal),+) => {
        $crate::define_matmul_x86!(@body $isa, $elem, 6, $LANES, $NV, $MC, [$($feat),+],
            [0, 1, 2, 3, 4, 5]);
    };

    (@body $isa:ident, $elem:ident, $TM:literal, $LANES:literal, $NV:literal, $MC:literal,
     [$($feat:literal),+], [$($R:literal),+]) => {
        /// Cached blocking parameters for this backend (kc_max, MC).
        #[inline(always)]
        fn _blocking() -> $crate::cache_params::BlockingParams {
            static BP: std::sync::OnceLock<$crate::cache_params::BlockingParams> = std::sync::OnceLock::new();
            *BP.get_or_init(|| $crate::cache_params::blocking_params(
                $TM, $NV, $LANES, std::mem::size_of::<$elem>(),
            ))
        }

        // ── pack_b (public) ────────────────────────────────────────────
        // SIMD-optimized: uses vector load/NT-store for full strips, L2 prefetch for next row.
        // NT stores bypass cache for write-once packed data (read later in GEMM).
        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            #[target_feature($(enable = $feat),+)]
            unsafe fn pack_b_simd(b: &[$elem], n_size: usize, k_size: usize, packed: &mut [$elem]) {
                const TN: usize = $NV * $LANES;
                const LANES: usize = $LANES;
                let kc_max = _blocking().kc;
                let n_strips = (n_size + TN - 1) / TN;
                let cs = n_strips * kc_max * TN;
                let bp = b.as_ptr();
                let pp = packed.as_mut_ptr();
                // Check if packed buffer is SIMD-aligned for NT stores
                let align = std::mem::size_of::<$elem>() * LANES; // 64B for AVX-512 f32, 32B for AVX2 f32
                let nt_ok = (pp as usize) % align == 0;
                let mut ks = 0usize;
                let mut ch = 0usize;
                while ks < k_size {
                    let kc = kc_max.min(k_size - ks);
                    let base = ch * cs;
                    for i in 0..n_strips {
                        let ns = i * TN;
                        let an = TN.min(n_size.saturating_sub(ns));
                        if an == TN && nt_ok {
                            // Full strip + aligned: SIMD load / NT-store (NV vectors per row)
                            for k in 0..kc {
                                let src = bp.add((ks + k) * n_size + ns);
                                let dst = pp.add(base + i * kc_max * TN + k * TN);
                                if k + 4 < kc {
                                    $crate::simd_primitive!($isa, $elem, prefetch,
                                        bp.add((ks + k + 4) * n_size + ns) as *const i8, 0);
                                }
                                let v0 = $crate::simd_primitive!($isa, $elem, loadu, src);
                                $crate::simd_primitive!($isa, $elem, stream, dst, v0);
                                let v1 = $crate::simd_primitive!($isa, $elem, loadu, src.add(LANES));
                                $crate::simd_primitive!($isa, $elem, stream, dst.add(LANES), v1);
                            }
                        } else if an == TN {
                            // Full strip but unaligned: regular SIMD store
                            for k in 0..kc {
                                let src = bp.add((ks + k) * n_size + ns);
                                let dst = pp.add(base + i * kc_max * TN + k * TN);
                                if k + 4 < kc {
                                    $crate::simd_primitive!($isa, $elem, prefetch,
                                        bp.add((ks + k + 4) * n_size + ns) as *const i8, 0);
                                }
                                let v0 = $crate::simd_primitive!($isa, $elem, loadu, src);
                                $crate::simd_primitive!($isa, $elem, storeu, dst, v0);
                                let v1 = $crate::simd_primitive!($isa, $elem, loadu, src.add(LANES));
                                $crate::simd_primitive!($isa, $elem, storeu, dst.add(LANES), v1);
                            }
                        } else {
                            // Partial strip (tail): SIMD for LANES-aligned portion, scalar remainder
                            for k in 0..kc {
                                let src = bp.add((ks + k) * n_size + ns);
                                let dst = pp.add(base + i * kc_max * TN + k * TN);
                                let mut j = 0usize;
                                while j + LANES <= an {
                                    let v = $crate::simd_primitive!($isa, $elem, loadu, src.add(j));
                                    $crate::simd_primitive!($isa, $elem, storeu, dst.add(j), v);
                                    j += LANES;
                                }
                                // Masked SIMD tail
                                let rem = an - j;
                                if rem > 0 {
                                    let v = $crate::simd_primitive!($isa, $elem, maskload, src.add(j), rem);
                                    $crate::simd_primitive!($isa, $elem, maskstore, dst.add(j), v, rem);
                                }
                            }
                        }
                    }
                    ks += kc_max; ch += 1;
                }
                // Fence: ensure all NT stores are globally visible before returning
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_sfence();
            }

            const TN: usize = $NV * $LANES;
            let kc_max = _blocking().kc;
            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let cs = n_strips * kc_max * TN;
            let mut packed = vec![<$elem as Element>::ZERO; n_chunks * cs];
            unsafe { pack_b_simd(b, n_size, k_size, &mut packed); }
            packed
        }

        // ── matmul: C = A * B (NC-blocked) ───────────────────────────────
        #[target_feature($(enable = $feat),+)]
        pub unsafe fn x86_matmul_impl(a: &[$elem], b: &[$elem], c: &mut [$elem],
                                       m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            let bp_ = _blocking();
            let kc_max = bp_.kc;
            let nc_max = bp_.nc;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            // NC-blocked packed B buffer: only nc_max columns at a time
            let nc_strips_max = (nc_max + TN - 1) / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let cs_nc = nc_strips_max * kc_max * TN; // chunk stride for NC-sized panel
            let tp = n_chunks * cs_nc;

            thread_local! { static WS: std::cell::Cell<$crate::cache_params::AlignedVec<$elem>> = std::cell::Cell::new($crate::cache_params::AlignedVec::new()); }
            let mut pb = WS.with(|c| c.take());
            if pb.capacity() < tp { pb.reserve(tp); }
            unsafe { pb.set_len(tp); }

            let cp = c.as_mut_ptr();
            let mc_base = _blocking().mc;
            let nthreads = rayon::current_num_threads().max(1);
            let mc_max = {
                let blocks_base = (m_size + mc_base - 1) / mc_base;
                // Ensure at least 3x oversubscription for load balancing
                let target_blocks = nthreads * 3;
                if blocks_base < target_blocks && m_size >= TM * 2 {
                    let mc_small = ((m_size + target_blocks - 1) / target_blocks) / TM * TM;
                    mc_small.max(TM)
                } else {
                    mc_base
                }
            };

            // ── KC outer loop ──
            let n_full = (n_size / TN) * TN; // full-TN portion
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);

                // ── NC inner loop ──
                let mut ns = 0usize;
                while ns < n_full {
                let nc = nc_max.min(n_full - ns);
                let nc_strips = nc / TN; // always exact since both nc and ns are TN-aligned
                let cs_local = nc_strips * kc_max * TN;

                    // Pack B for columns [ns, ns+nc) this KC strip — parallel across strips
                    {
                        let bptr_addr = b.as_ptr() as usize;
                        let pp_addr = pb.as_mut_ptr() as usize;
                        let pack_strip = |i: usize| {
                            let bptr = bptr_addr as *const $elem;
                            let pp = pp_addr as *mut $elem;
                            let col = ns + i * TN;
                            unsafe { for k in 0..kc {
                                let src = bptr.add((ks + k) * n_size + col);
                                let dst = pp.add(i * kc_max * TN + k * TN);
                                if k + 4 < kc {
                                    $crate::simd_primitive!($isa, $elem, prefetch,
                                        bptr.add((ks + k + 4) * n_size + col) as *const i8, 0);
                                }
                                let v0 = $crate::simd_primitive!($isa, $elem, loadu, src);
                                $crate::simd_primitive!($isa, $elem, storeu, dst, v0);
                                let v1 = $crate::simd_primitive!($isa, $elem, loadu, src.add($LANES));
                                $crate::simd_primitive!($isa, $elem, storeu, dst.add($LANES), v1);
                            }}
                        };
                        if nc_strips >= 4 && nthreads > 1 {
                            use rayon::prelude::*;
                            (0..nc_strips).into_par_iter().for_each(|i| pack_strip(i));
                        } else {
                            for i in 0..nc_strips { pack_strip(i); }
                        }
                    }

                    // M-parallel microkernel
                    let num_blocks = (m_size + mc_max - 1) / mc_max;
                    let cp_addr = cp as usize;
                    let ap_addr = a.as_ptr() as usize;
                    let pb_addr = pb.as_ptr() as usize;
                    let mc_body = |bi: usize| {
                        let cp = cp_addr as *mut $elem;
                        let a_ptr = ap_addr as *const $elem;
                        let pb_ptr = pb_addr as *const $elem;
                        let m_block = bi * mc_max;
                        let m_end = (m_block + mc_max).min(m_size);
                        let mc = m_end - m_block;
                        thread_local! { static WS_A: std::cell::Cell<$crate::cache_params::AlignedVec<$elem>> = std::cell::Cell::new($crate::cache_params::AlignedVec::new()); }
                        let mut pa = WS_A.with(|c| c.take());
                        // Only pack A on the first NC iteration of this KC chunk
                        if ns == 0 {
                            let pa_cap = mc_max * kc_max;
                            if pa.capacity() < pa_cap { pa.reserve(pa_cap); }
                            unsafe { pa.set_len(pa_cap); }
                            unsafe {
                                let pp = pa.as_mut_ptr();
                                let n_tiles = mc / TM;
                                for t in 0..n_tiles {
                                    let tile_off = t * TM * kc;
                                    for k in 0..kc {
                                        $(
                                            *pp.add(tile_off + k * TM + $R) =
                                                *a_ptr.add((m_block + t * TM + $R) * k_size + ks + k);
                                        )+
                                    }
                                }
                            }
                        }
                        let mut m = m_block;
                        while m + TM <= m_end {
                            let pa_tile = ((m - m_block) / TM) * TM * kc;
                            let mut si = 0usize;
                            while si < nc_strips {
                                let n = ns + si * TN;
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
                                    let mut pac = pa.as_ptr().add(pa_tile);
                                    let mut bp = pb_ptr.add(si*kc_max*TN);
                                    $($crate::simd_primitive!($isa, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const i8, 0);)+
                                    let mut _k = 0usize;
                                    let ku = kc & !7;
                                    while _k < ku {
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                        $crate::simd_primitive!($isa, $elem, prefetch_nta, pac.add(TM*8) as *const i8);
                                        let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                        let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                        let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                        let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*20) as *const i8, 0);
                                        let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                        let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                        let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                        let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                        let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN*8); _k += 8;
                                    }
                                    while _k < kc {
                                        let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN); _k += 1;
                                    }
                                    $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                                      $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                                }}
                                si += 1;
                            }
                            m += TM;
                        }
                        WS_A.with(|c| c.set(pa));
                    };
                    if num_blocks >= 2 && nthreads > 1 {
                        use rayon::prelude::*;
                        (0..num_blocks).into_par_iter().for_each(|bi| mc_body(bi));
                    } else {
                        for bi in 0..num_blocks { mc_body(bi); }
                    }
                    ns += nc;
                } // NC inner loop
                ks += kc_max; ch += 1;
            } // KC outer loop

            // ── N-remainder: columns [n_full, n_size) — KC-blocked ──
            if n_full < n_size {
                let nr = n_size - n_full;
                // LANES-wide chunks
                let mut nos = 0usize;
                while nos + $LANES <= nr {
                    for m in 0..m_size { unsafe {
                        let mut acc = $crate::simd_primitive!($isa, $elem, zero);
                        let mut ks2 = 0usize;
                        while ks2 < k_size {
                            let kc2 = kc_max.min(k_size - ks2);
                            for k in ks2..ks2+kc2 {
                                let va = $crate::simd_primitive!($isa, $elem, splat, *a.as_ptr().add(m * k_size + k));
                                let vb = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k * n_size + n_full + nos));
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                            }
                            ks2 += kc2;
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n_full + nos), acc);
                    }}
                    nos += $LANES;
                }
                // Scalar tail
                for m in 0..m_size {
                    for no in nos..nr {
                        let mut s = <$elem as Element>::ZERO;
                        let mut ks2 = 0usize;
                        while ks2 < k_size {
                            let kc2 = kc_max.min(k_size - ks2);
                            for k in ks2..ks2+kc2 {
                                s = <$elem as Element>::mul_add(s, a[m * k_size + k], b[k * n_size + n_full + no]);
                            }
                            ks2 += kc2;
                        }
                        c[m * n_size + n_full + no] = s;
                    }
                }
            }
            // ── M-remainder: rows [mm, m_size) — KC-blocked ──
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize;
                while n + TN <= n_size { unsafe {
                    let mut c0 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut ks2 = 0usize;
                    while ks2 < k_size {
                        let kc2 = kc_max.min(k_size - ks2);
                        for k in ks2..ks2+kc2 {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *a.as_ptr().add(m * k_size + k));
                            let v0 = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k * n_size + n));
                            let v1 = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k * n_size + n + $LANES));
                            c0 = $crate::simd_primitive!($isa, $elem, fma, va, v0, c0);
                            c1 = $crate::simd_primitive!($isa, $elem, fma, va, v1, c1);
                        }
                        ks2 += kc2;
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n+$LANES), c1);
                    n += TN;
                }}
            }
            WS.with(|c| c.set(pb));
        }

        // ── Small-M no-pack path: read B directly from row-major layout ──
        // Avoids O(K*N) packing overhead when M is small (e.g. autoregressive decode).
        // Uses TM×TN microkernel with strided B loads (stride = n_size).
        // When M is small and N is large, parallelizes across the N dimension using rayon.
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_nopack_impl(a: &[$elem], b: &[$elem], c: &mut [$elem],
                                          m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);

            // ── N-parallel dispatch: when M is small and N is large, split N across threads ──
            let nthreads = rayon::current_num_threads().max(1);
            if n_size >= TN * 4 && nthreads > 1 {
                // Split N into nthreads chunks, each aligned to TN (last chunk may be smaller)
                let n_per_thread = ((n_size + nthreads - 1) / nthreads + TN - 1) / TN * TN;
                let num_n_blocks = (n_size + n_per_thread - 1) / n_per_thread;
                let cp_addr = c.as_mut_ptr() as usize;
                let ap_addr = a.as_ptr() as usize;
                let bp_addr = b.as_ptr() as usize;

                use rayon::prelude::*;
                (0..num_n_blocks).into_par_iter().for_each(|ni| {
                    let n_start = ni * n_per_thread;
                    let n_end = (n_start + n_per_thread).min(n_size);
                    let ap = ap_addr as *const $elem;
                    let bp = bp_addr as *const $elem;
                    let cp = cp_addr as *mut $elem;
                    unsafe {
                        x86_matmul_nopack_range(ap, bp, cp, m_size, n_size, k_size, n_start, n_end);
                    }
                });
                return;
            }

            // Single-thread fallback: process full N range
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = c.as_mut_ptr();
            unsafe { x86_matmul_nopack_range(ap, bp, cp, m_size, n_size, k_size, 0, n_size); }
        }

        /// Inner nopack kernel operating on N range [n_start, n_end).
        /// Each thread calls this on disjoint column ranges — no synchronization needed.
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_nopack_range(
            ap: *const $elem, bp: *const $elem, cp: *mut $elem,
            m_size: usize, n_size: usize, k_size: usize,
            n_start: usize, n_end: usize,
        ) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;

            let kc_max = _blocking().kc;
            let mut ks = 0usize;
            let mut ch = 0u32; // KC chunk index

            while ks < k_size {
                let kc = (k_size - ks).min(kc_max);

            // ── Main TM×TN tiles ──
            let mut m = 0usize;
            while m + TM <= m_size {
                let mut n = n_start;
                while n + TN <= n_end {
                    paste::paste! {
                        // Init accumulators: zero on first chunk, load C on subsequent
                        $(
                            let mut [<c_ $R _0>] = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                                else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n)) };
                            let mut [<c_ $R _1>] = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                                else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n + $LANES)) };
                        )+
                        // 8-way unrolled K-loop with prefetch
                        let mut ki = 0usize;
                        let ku = kc & !7;
                        while ki < ku {
                            let k = ks + ki;
                            // Prefetch B 16 rows ahead, A 32 elements ahead
                            $crate::simd_primitive!($isa, $elem, prefetch, bp.wrapping_add((k + 16) * n_size + n) as *const i8, 0);
                            $($crate::simd_primitive!($isa, $elem, prefetch, ap.wrapping_add((m + $R) * k_size + k + 32) as *const i8, 0);)+
                            // k+0
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+1
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+1) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+1) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 1));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+2
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+2) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+2) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 2));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+3
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+3) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+3) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 3));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // Mid-group prefetch (every 4 steps)
                            $crate::simd_primitive!($isa, $elem, prefetch, bp.wrapping_add((k + 20) * n_size + n) as *const i8, 0);
                            // k+4
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+4) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+4) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 4));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+5
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+5) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+5) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 5));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+6
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+6) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+6) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 6));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+7
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+7) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+7) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 7));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            ki += 8;
                        }
                        // Remainder
                        while ki < kc {
                            let k = ks + ki;
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            ki += 1;
                        }
                        // Store
                        $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R _0>]);
                          $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n + $LANES), [<c_ $R _1>]);)+
                    }
                    n += TN;
                }
                // N-remainder: LANES-wide
                while n + $LANES <= n_end {
                    $(
                        let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                            else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n)) };
                        for ki in 0..kc {
                            let k = ks + ki;
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                            let vb = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), acc);
                    )+
                    n += $LANES;
                }
                // N-remainder: masked SIMD
                {
                    let rem = n_end - n;
                    if rem > 0 { unsafe {
                        $(
                        {
                            let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                                else { $crate::simd_primitive!($isa, $elem, maskload, cp.add((m + $R) * n_size + n), rem) };
                            for ki in 0..kc {
                                let k = ks + ki;
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                let vb = $crate::simd_primitive!($isa, $elem, maskload, bp.add(k * n_size + n), rem);
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                            }
                            $crate::simd_primitive!($isa, $elem, maskstore, cp.add((m + $R) * n_size + n), acc, rem);
                        }
                        )+
                    }}
                }
                m += TM;
            }
            // M-remainder: 1×TN tiles
            while m < m_size {
                let mut n = n_start;
                while n + TN <= n_end {
                    let mut c0 = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n)) };
                    let mut c1 = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n + $LANES)) };
                    for ki in 0..kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                        c0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, c0);
                        c1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, c1);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n + $LANES), c1);
                    n += TN;
                }
                while n + $LANES <= n_end {
                    let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n)) };
                    for ki in 0..kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                        let vb = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                        acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), acc);
                    n += $LANES;
                }
                {
                    let rem = n_end - n;
                    if rem > 0 {
                        let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, zero) }
                            else { $crate::simd_primitive!($isa, $elem, maskload, cp.add(m * n_size + n), rem) };
                        for ki in 0..kc {
                            let k = ks + ki;
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                            let vb = $crate::simd_primitive!($isa, $elem, maskload, bp.add(k * n_size + n), rem);
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!($isa, $elem, maskstore, cp.add(m * n_size + n), acc, rem);
                    }
                }
                m += 1;
            }

            ks += kc; ch += 1;
            } // KC loop
        }

        // ── Small-M no-pack path with bias ──
        // When M is small and N is large, parallelizes across the N dimension using rayon.
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_bias_nopack_impl(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                                               m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            // ── N-parallel dispatch ──
            let nthreads = rayon::current_num_threads().max(1);
            if n_size >= TN * 4 && nthreads > 1 {
                let n_per_thread = ((n_size + nthreads - 1) / nthreads + TN - 1) / TN * TN;
                let num_n_blocks = (n_size + n_per_thread - 1) / n_per_thread;
                let cp_addr = c.as_mut_ptr() as usize;
                let ap_addr = a.as_ptr() as usize;
                let bp_addr = b.as_ptr() as usize;
                let biasp_addr = bias.as_ptr() as usize;

                use rayon::prelude::*;
                (0..num_n_blocks).into_par_iter().for_each(|ni| {
                    let n_start = ni * n_per_thread;
                    let n_end = (n_start + n_per_thread).min(n_size);
                    let ap = ap_addr as *const $elem;
                    let bp = bp_addr as *const $elem;
                    let cp = cp_addr as *mut $elem;
                    let biasp = biasp_addr as *const $elem;
                    unsafe {
                        x86_matmul_bias_nopack_range(ap, bp, biasp, cp, m_size, n_size, k_size, n_start, n_end);
                    }
                });
                return;
            }

            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = c.as_mut_ptr();
            let biasp = bias.as_ptr();
            unsafe { x86_matmul_bias_nopack_range(ap, bp, biasp, cp, m_size, n_size, k_size, 0, n_size); }
        }

        /// Inner bias nopack kernel operating on N range [n_start, n_end).
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_bias_nopack_range(
            ap: *const $elem, bp: *const $elem, biasp: *const $elem, cp: *mut $elem,
            m_size: usize, n_size: usize, k_size: usize,
            n_start: usize, n_end: usize,
        ) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;

            let kc_max = _blocking().kc;
            let mut ks = 0usize;
            let mut ch = 0u32;

            while ks < k_size {
                let kc = (k_size - ks).min(kc_max);

            let mut m = 0usize;
            while m + TM <= m_size {
                let mut n = n_start;
                while n + TN <= n_end {
                    paste::paste! {
                        // Init: bias on first chunk, load C on subsequent
                        $(
                            let mut [<c_ $R _0>] = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                                else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n)) };
                            let mut [<c_ $R _1>] = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n + $LANES)) }
                                else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n + $LANES)) };
                        )+
                        // 8-way unrolled K-loop with prefetch
                        let mut ki = 0usize;
                        let ku = kc & !7;
                        while ki < ku {
                            let k = ks + ki;
                            // Prefetch B 16 rows ahead, A 32 elements ahead
                            $crate::simd_primitive!($isa, $elem, prefetch, bp.wrapping_add((k + 16) * n_size + n) as *const i8, 0);
                            $($crate::simd_primitive!($isa, $elem, prefetch, ap.wrapping_add((m + $R) * k_size + k + 32) as *const i8, 0);)+
                            // k+0
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+1
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+1) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+1) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 1));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+2
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+2) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+2) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 2));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+3
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+3) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+3) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 3));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // Mid-group prefetch (every 4 steps)
                            $crate::simd_primitive!($isa, $elem, prefetch, bp.wrapping_add((k + 20) * n_size + n) as *const i8, 0);
                            // k+4
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+4) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+4) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 4));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+5
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+5) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+5) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 5));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+6
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+6) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+6) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 6));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+7
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+7) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+7) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 7));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            ki += 8;
                        }
                        // Remainder
                        while ki < kc {
                            let k = ks + ki;
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            ki += 1;
                        }
                        $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R _0>]);
                          $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n + $LANES), [<c_ $R _1>]);)+
                    }
                    n += TN;
                }
                while n + $LANES <= n_end {
                    $(
                        let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                            else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n)) };
                        for ki in 0..kc {
                            let k = ks + ki;
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                            let vb = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), acc);
                    )+
                    n += $LANES;
                }
                {
                    let rem = n_end - n;
                    if rem > 0 { unsafe {
                        $(
                        {
                            let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, maskload, biasp.add(n), rem) }
                                else { $crate::simd_primitive!($isa, $elem, maskload, cp.add((m + $R) * n_size + n), rem) };
                            for ki in 0..kc {
                                let k = ks + ki;
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                let vb = $crate::simd_primitive!($isa, $elem, maskload, bp.add(k * n_size + n), rem);
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                            }
                            $crate::simd_primitive!($isa, $elem, maskstore, cp.add((m + $R) * n_size + n), acc, rem);
                        }
                        )+
                    }}
                }
                m += TM;
            }
            while m < m_size {
                let mut n = n_start;
                while n + TN <= n_end {
                    let mut c0 = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n)) };
                    let mut c1 = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n + $LANES)) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n + $LANES)) };
                    for ki in 0..kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                        c0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, c0);
                        c1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, c1);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n + $LANES), c1);
                    n += TN;
                }
                while n + $LANES <= n_end {
                    let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n)) };
                    for ki in 0..kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                        let vb = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                        acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), acc);
                    n += $LANES;
                }
                {
                    let rem = n_end - n;
                    if rem > 0 {
                        let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, maskload, biasp.add(n), rem) }
                            else { $crate::simd_primitive!($isa, $elem, maskload, cp.add(m * n_size + n), rem) };
                        for ki in 0..kc {
                            let k = ks + ki;
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                            let vb = $crate::simd_primitive!($isa, $elem, maskload, bp.add(k * n_size + n), rem);
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        $crate::simd_primitive!($isa, $elem, maskstore, cp.add(m * n_size + n), acc, rem);
                    }
                }
                m += 1;
            }

            ks += kc; ch += 1;
            } // KC loop
        }

        /// Threshold: skip B-packing when M ≤ 4*TM (e.g. 56 for AVX-512, 24 for AVX2).
        const SMALL_M_THRESHOLD: usize = 4 * $TM;

        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            if m_size <= SMALL_M_THRESHOLD {
                unsafe { x86_matmul_nopack_impl(a, b, c, m_size, n_size, k_size); }
            } else {
                unsafe { x86_matmul_impl(a, b, c, m_size, n_size, k_size); }
            }
        }

        // ── matmul_bias: C = A * B + bias (NC-blocked) ─────────────────
        #[target_feature($(enable = $feat),+)]
        pub unsafe fn x86_matmul_bias_impl(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                                            m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            let bp_ = _blocking();
            let kc_max = bp_.kc;
            let nc_max = bp_.nc;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            // Init C with bias (full matrix)
            let cp = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), cp.add(m * n_size), n_size); }
            }

            // NC-blocked packed B buffer
            let nc_strips_max = (nc_max + TN - 1) / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let cs_nc = nc_strips_max * kc_max * TN;
            let tp = n_chunks * cs_nc;

            thread_local! { static WS: std::cell::Cell<$crate::cache_params::AlignedVec<$elem>> = std::cell::Cell::new($crate::cache_params::AlignedVec::new()); }
            let mut pb = WS.with(|c| c.take());
            if pb.capacity() < tp { pb.reserve(tp); }
            unsafe { pb.set_len(tp); }

            let mc_base_b = _blocking().mc;
            let nthreads = rayon::current_num_threads().max(1);
            let mc_max = {
                let blocks_base = (m_size + mc_base_b - 1) / mc_base_b;
                // Ensure at least 3x oversubscription for load balancing
                let target_blocks = nthreads * 3;
                if blocks_base < target_blocks && m_size >= TM * 2 {
                    let mc_small = ((m_size + target_blocks - 1) / target_blocks) / TM * TM;
                    mc_small.max(TM)
                } else {
                    mc_base_b
                }
            };

            // ── KC outer loop ──
            let n_full = (n_size / TN) * TN;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);

                // ── NC inner loop ──
                let mut ns = 0usize;
                while ns < n_full {
                let nc = nc_max.min(n_full - ns);
                let nc_strips = nc / TN;
                let cs_local = nc_strips * kc_max * TN;

                    // Pack B for columns [ns, ns+nc) — parallel across strips
                    {
                        let bptr_addr = b.as_ptr() as usize;
                        let pp_addr = pb.as_mut_ptr() as usize;
                        let pack_strip = |i: usize| {
                            let bptr = bptr_addr as *const $elem;
                            let pp = pp_addr as *mut $elem;
                            let col = ns + i * TN;
                            unsafe { for k in 0..kc {
                                let src = bptr.add((ks + k) * n_size + col);
                                let dst = pp.add(i * kc_max * TN + k * TN);
                                if k + 4 < kc {
                                    $crate::simd_primitive!($isa, $elem, prefetch,
                                        bptr.add((ks + k + 4) * n_size + col) as *const i8, 0);
                                }
                                let v0 = $crate::simd_primitive!($isa, $elem, loadu, src);
                                $crate::simd_primitive!($isa, $elem, storeu, dst, v0);
                                let v1 = $crate::simd_primitive!($isa, $elem, loadu, src.add($LANES));
                                $crate::simd_primitive!($isa, $elem, storeu, dst.add($LANES), v1);
                            }}
                        };
                        if nc_strips >= 4 && nthreads > 1 {
                            use rayon::prelude::*;
                            (0..nc_strips).into_par_iter().for_each(|i| pack_strip(i));
                        } else {
                            for i in 0..nc_strips { pack_strip(i); }
                        }
                    }

                    // M-parallel microkernel
                    let num_blocks = (m_size + mc_max - 1) / mc_max;
                    let cp_addr = cp as usize;
                    let ap_addr = a.as_ptr() as usize;
                    let pb_addr = pb.as_ptr() as usize;
                    let mc_body = |bi: usize| {
                        let cp = cp_addr as *mut $elem;
                        let a_ptr = ap_addr as *const $elem;
                        let pb_ptr = pb_addr as *const $elem;
                        let m_block = bi * mc_max;
                        let m_end = (m_block + mc_max).min(m_size);
                        let mc = m_end - m_block;
                        thread_local! { static WS_A2: std::cell::Cell<$crate::cache_params::AlignedVec<$elem>> = std::cell::Cell::new($crate::cache_params::AlignedVec::new()); }
                        let mut pa = WS_A2.with(|c| c.take());
                        // Only pack A on the first NC iteration of this KC chunk
                        if ns == 0 {
                            let pa_cap = mc_max * kc_max;
                            if pa.capacity() < pa_cap { pa.reserve(pa_cap); }
                            unsafe { pa.set_len(pa_cap); }
                            unsafe {
                                let pp = pa.as_mut_ptr();
                                let n_tiles = mc / TM;
                                for t in 0..n_tiles {
                                    let tile_off = t * TM * kc;
                                    for k in 0..kc {
                                        $(
                                            *pp.add(tile_off + k * TM + $R) =
                                                *a_ptr.add((m_block + t * TM + $R) * k_size + ks + k);
                                        )+
                                    }
                                }
                            }
                        }
                        let mut m = m_block;
                        while m + TM <= m_end {
                            let pa_tile = ((m - m_block) / TM) * TM * kc;
                            let mut si = 0usize;
                            while si < nc_strips {
                                let n = ns + si * TN;
                                unsafe { paste::paste! {
                                    // Always load from C (bias already there, or partial sums from prior KC)
                                    $(
                                        let (mut [<c_ $R _0>], mut [<c_ $R _1>]) = (
                                            $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                            $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n+$LANES)),
                                        );
                                    )+
                                    let mut pac = pa.as_ptr().add(pa_tile);
                                    let mut bp = pb_ptr.add(si*kc_max*TN);
                                    $($crate::simd_primitive!($isa, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const i8, 0);)+
                                    let mut _k = 0usize;
                                    let ku = kc & !7;
                                    while _k < ku {
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                        $crate::simd_primitive!($isa, $elem, prefetch_nta, pac.add(TM*8) as *const i8);
                                        let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                        let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                        let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                        let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*20) as *const i8, 0);
                                        let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                        let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                        let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                        let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                        let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN*8); _k += 8;
                                    }
                                    while _k < kc {
                                        let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN); _k += 1;
                                    }
                                    $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                                      $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                                }}
                                si += 1;
                            }
                            m += TM;
                        }
                        WS_A2.with(|c| c.set(pa));
                    };
                    if num_blocks >= 2 && nthreads > 1 {
                        use rayon::prelude::*;
                        (0..num_blocks).into_par_iter().for_each(|bi| mc_body(bi));
                    } else {
                        for bi in 0..num_blocks { mc_body(bi); }
                    }
                    ns += nc;
                } // NC inner loop
                ks += kc_max; ch += 1;
            } // KC outer loop

            // ── N-remainder with bias: columns [n_full, n_size) — KC-blocked ──
            if n_full < n_size {
                let nr = n_size - n_full;
                let mut nos = 0usize;
                while nos + $LANES <= nr {
                    for m in 0..m_size { unsafe {
                        let mut acc = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(n_full + nos));
                        let mut ks2 = 0usize;
                        while ks2 < k_size {
                            let kc2 = kc_max.min(k_size - ks2);
                            for k in ks2..ks2+kc2 {
                                let va = $crate::simd_primitive!($isa, $elem, splat, *a.as_ptr().add(m * k_size + k));
                                let vb = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k * n_size + n_full + nos));
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                            }
                            ks2 += kc2;
                        }
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n_full + nos), acc);
                    }}
                    nos += $LANES;
                }
                for m in 0..m_size {
                    for no in nos..nr {
                        let mut s = bias[n_full + no];
                        let mut ks2 = 0usize;
                        while ks2 < k_size {
                            let kc2 = kc_max.min(k_size - ks2);
                            for k in ks2..ks2+kc2 {
                                s = <$elem as Element>::mul_add(s, a[m * k_size + k], b[k * n_size + n_full + no]);
                            }
                            ks2 += kc2;
                        }
                        c[m * n_size + n_full + no] = s;
                    }
                }
            }
            // ── M-remainder with bias — KC-blocked ──
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize;
                while n + TN <= n_size { unsafe {
                    let bp2 = bias.as_ptr().add(n);
                    let mut c0 = $crate::simd_primitive!($isa, $elem, loadu, bp2);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, loadu, bp2.add($LANES));
                    let mut ks2 = 0usize;
                    while ks2 < k_size {
                        let kc2 = kc_max.min(k_size - ks2);
                        for k in ks2..ks2+kc2 {
                            let va = $crate::simd_primitive!($isa, $elem, splat, *a.as_ptr().add(m * k_size + k));
                            let v0 = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k * n_size + n));
                            let v1 = $crate::simd_primitive!($isa, $elem, loadu, b.as_ptr().add(k * n_size + n + $LANES));
                            c0 = $crate::simd_primitive!($isa, $elem, fma, va, v0, c0);
                            c1 = $crate::simd_primitive!($isa, $elem, fma, va, v1, c1);
                        }
                        ks2 += kc2;
                    }
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n), c0);
                    $crate::simd_primitive!($isa, $elem, storeu, cp.add(m*n_size+n+$LANES), c1);
                    n += TN;
                }}
            }
            WS.with(|c| c.set(pb));
        }

        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            if m_size <= SMALL_M_THRESHOLD {
                unsafe { x86_matmul_bias_nopack_impl(a, b, bias, c, m_size, n_size, k_size); }
            } else {
                unsafe { x86_matmul_bias_impl(a, b, bias, c, m_size, n_size, k_size); }
            }
        }

        // ── matmul_prepacked: C = A * packed_B (NC-blocked) ────────────
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_prepacked_impl(a: &[$elem], pb: &[$elem], c: &mut [$elem],
                                             m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            let bp_ = _blocking();
            let kc_max = bp_.kc;
            let nc_max = bp_.nc;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TN - 1) / TN;
            let cs = n_strips * kc_max * TN; // chunk stride for FULL N (packed layout)
            let cp = c.as_mut_ptr();
            let mc_base_p = _blocking().mc;
            let nthreads = rayon::current_num_threads().max(1);
            let mc_max = {
                let blocks_base = (m_size + mc_base_p - 1) / mc_base_p;
                // Ensure at least 3x oversubscription for load balancing
                let target_blocks = nthreads * 3;
                if blocks_base < target_blocks && m_size >= TM * 2 {
                    let mc_small = ((m_size + target_blocks - 1) / target_blocks) / TM * TM;
                    mc_small.max(TM)
                } else {
                    mc_base_p
                }
            };

            // ── KC outer loop over prepacked B ──
            let n_full = (n_size / TN) * TN;
            let nc_strips_max = (nc_max + TN - 1) / TN;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);

                // ── NC inner loop over full-TN strips ──
                let mut strip_start = 0usize;
                while strip_start * TN < n_full {
                let strips_left = (n_full / TN) - strip_start;
                let nc_strips = strips_left.min(nc_strips_max);

                    let num_blocks = (m_size + mc_max - 1) / mc_max;
                    let cp_addr = cp as usize;
                    let ap_addr = a.as_ptr() as usize;
                    let pb_addr = pb.as_ptr() as usize;
                    let mc_body = |bi: usize| {
                        let cp = cp_addr as *mut $elem;
                        let a_ptr = ap_addr as *const $elem;
                        let pb_ptr = pb_addr as *const $elem;
                        let m_block = bi * mc_max;
                        let m_end = (m_block + mc_max).min(m_size);
                        let mc = m_end - m_block;
                        thread_local! { static WS_A3: std::cell::Cell<$crate::cache_params::AlignedVec<$elem>> = std::cell::Cell::new($crate::cache_params::AlignedVec::new()); }
                        let mut pa = WS_A3.with(|c| c.take());
                        // Only pack A on the first NC iteration of this KC chunk
                        if strip_start == 0 {
                            let pa_cap = mc_max * kc_max;
                            if pa.capacity() < pa_cap { pa.reserve(pa_cap); }
                            unsafe { pa.set_len(pa_cap); }
                            unsafe {
                                let pp = pa.as_mut_ptr();
                                let n_tiles = mc / TM;
                                for t in 0..n_tiles {
                                    let tile_off = t * TM * kc;
                                    for k in 0..kc {
                                        $(
                                            *pp.add(tile_off + k * TM + $R) =
                                                *a_ptr.add((m_block + t * TM + $R) * k_size + ks + k);
                                        )+
                                    }
                                }
                            }
                        }
                        let mut m = m_block;
                        while m + TM <= m_end {
                            let pa_tile = ((m - m_block) / TM) * TM * kc;
                            let mut si_local = 0usize;
                            while si_local < nc_strips {
                                let si = strip_start + si_local; // global strip index
                                let n = si * TN;
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
                                    let mut pac = pa.as_ptr().add(pa_tile);
                                    let mut bp = pb_ptr.add(ch*cs+si*kc_max*TN);
                                    $($crate::simd_primitive!($isa, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const i8, 0);)+
                                    let mut _k = 0usize;
                                    let ku = kc & !7;
                                    while _k < ku {
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                        $crate::simd_primitive!($isa, $elem, prefetch_nta, pac.add(TM*8) as *const i8);
                                        let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                        let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                        let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                        let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                        let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                        let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                        let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                        let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN*8); _k += 8;
                                    }
                                    while _k < kc {
                                        let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN); _k += 1;
                                    }
                                    $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                                      $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                                }}
                                si_local += 1;
                            }
                            m += TM;
                        }
                        WS_A3.with(|c| c.set(pa));
                    };
                    if num_blocks >= 2 && nthreads > 1 {
                        use rayon::prelude::*;
                        (0..num_blocks).into_par_iter().for_each(|bi| mc_body(bi));
                    } else {
                        for bi in 0..num_blocks { mc_body(bi); }
                    }
                    strip_start += nc_strips;
                } // NC inner loop
                ks += kc_max; ch += 1;
            } // KC outer loop

            // ── N-remainder: columns [n_full, n_size) — read from packed B ──
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
                            let kc = kc_max.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+ls*kc_max*TN+nos);
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
                            let kc = kc_max.min(k_size-k);
                            let base = ci*cs+ls*kc_max*TN;
                            for ki in 0..kc { s = <$elem as Element>::mul_add(s, a[m*k_size+k+ki], pb[base+ki*TN+no]); }
                            k += kc; ci += 1;
                        }
                        c[m*n_size+nm+no] = s;
                    }
                }
            }
            // ── M-remainder ──
            let mm = (m_size / TM) * TM;
            for m in mm..m_size {
                let mut n = 0usize; let mut si = 0usize;
                while n + TN <= n_size { unsafe {
                    let mut c0 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut c1 = $crate::simd_primitive!($isa, $elem, zero);
                    let mut ac = a.as_ptr().add(m*k_size);
                    let mut ks2 = 0usize; let mut ci = 0usize;
                    while ks2 < k_size {
                        let kc = kc_max.min(k_size-ks2);
                        let mut bpp = pb.as_ptr().add(ci*cs+si*kc_max*TN);
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

        // ── matmul_bias_prepacked: C = A * packed_B + bias (NC-blocked) ─
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_bias_prepacked_impl(a: &[$elem], pb: &[$elem], bias: &[$elem], c: &mut [$elem],
                                                  m_size: usize, n_size: usize, k_size: usize) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            let bp_ = _blocking();
            let kc_max = bp_.kc;
            let nc_max = bp_.nc;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TN - 1) / TN;
            let cs = n_strips * kc_max * TN;
            let cp = c.as_mut_ptr();

            // Init C with bias
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), cp.add(m * n_size), n_size); }
            }

            let mc_base_bp = _blocking().mc;
            let nthreads = rayon::current_num_threads().max(1);
            let mc_max = {
                let blocks_base = (m_size + mc_base_bp - 1) / mc_base_bp;
                // Ensure at least 3x oversubscription for load balancing
                let target_blocks = nthreads * 3;
                if blocks_base < target_blocks && m_size >= TM * 2 {
                    let mc_small = ((m_size + target_blocks - 1) / target_blocks) / TM * TM;
                    mc_small.max(TM)
                } else {
                    mc_base_bp
                }
            };

            // ── KC outer loop over prepacked B with bias ──
            let n_full = (n_size / TN) * TN;
            let nc_strips_max = (nc_max + TN - 1) / TN;
            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);

                // ── NC inner loop over full-TN strips ──
                let mut strip_start = 0usize;
                while strip_start * TN < n_full {
                let strips_left = (n_full / TN) - strip_start;
                let nc_strips = strips_left.min(nc_strips_max);

                    let num_blocks = (m_size + mc_max - 1) / mc_max;
                    let cp_addr = cp as usize;
                    let ap_addr = a.as_ptr() as usize;
                    let pb_addr = pb.as_ptr() as usize;
                    let mc_body = |bi: usize| {
                        let cp = cp_addr as *mut $elem;
                        let a_ptr = ap_addr as *const $elem;
                        let pb_ptr = pb_addr as *const $elem;
                        let m_block = bi * mc_max;
                        let m_end = (m_block + mc_max).min(m_size);
                        let mc = m_end - m_block;
                        thread_local! { static WS_A4: std::cell::Cell<$crate::cache_params::AlignedVec<$elem>> = std::cell::Cell::new($crate::cache_params::AlignedVec::new()); }
                        let mut pa = WS_A4.with(|c| c.take());
                        // Only pack A on the first NC iteration of this KC chunk
                        if strip_start == 0 {
                            let pa_cap = mc_max * kc_max;
                            if pa.capacity() < pa_cap { pa.reserve(pa_cap); }
                            unsafe { pa.set_len(pa_cap); }
                            unsafe {
                                let pp = pa.as_mut_ptr();
                                let n_tiles = mc / TM;
                                for t in 0..n_tiles {
                                    let tile_off = t * TM * kc;
                                    for k in 0..kc {
                                        $(
                                            *pp.add(tile_off + k * TM + $R) =
                                                *a_ptr.add((m_block + t * TM + $R) * k_size + ks + k);
                                        )+
                                    }
                                }
                            }
                        }
                        let mut m = m_block;
                        while m + TM <= m_end {
                            let pa_tile = ((m - m_block) / TM) * TM * kc;
                            let mut si_local = 0usize;
                            while si_local < nc_strips {
                                let si = strip_start + si_local;
                                let n = si * TN;
                                unsafe { paste::paste! {
                                    // Always load from C (bias already there, or partial sums)
                                    $(
                                        let (mut [<c_ $R _0>], mut [<c_ $R _1>]) = (
                                            $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n)),
                                            $crate::simd_primitive!($isa, $elem, loadu, cp.add((m+$R)*n_size+n+$LANES)),
                                        );
                                    )+
                                    let mut pac = pa.as_ptr().add(pa_tile);
                                    let mut bp = pb_ptr.add(ch*cs+si*kc_max*TN);
                                    $($crate::simd_primitive!($isa, $elem, prefetch, cp.add((m+$R)*n_size+n) as *const i8, 0);)+
                                    let mut _k = 0usize;
                                    let ku = kc & !7;
                                    while _k < ku {
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
                                        $crate::simd_primitive!($isa, $elem, prefetch_nta, pac.add(TM*8) as *const i8);
                                        let vb0_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb0_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb1_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN));
                                        let vb1_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb2_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2));
                                        let vb2_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*2+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb2_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb3_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3));
                                        let vb3_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*3+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb3_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        $crate::simd_primitive!($isa, $elem, prefetch, bp.add(TN*20) as *const i8, 0);
                                        let vb4_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4));
                                        let vb4_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*4+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb4_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb5_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5));
                                        let vb5_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*5+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb5_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb6_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6));
                                        let vb6_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*6+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb6_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        let vb7_0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7));
                                        let vb7_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(TN*7+$LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb7_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN*8); _k += 8;
                                    }
                                    while _k < kc {
                                        let vb_0 = $crate::simd_primitive!($isa, $elem, loadu, bp);
                                        let vb_1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add($LANES));
                                        $(
                                                let va = $crate::simd_primitive!($isa, $elem, splat, *pac.add($R));
                                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_0, [<c_ $R _0>]);
                                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb_1, [<c_ $R _1>]);
                                        )+; pac = pac.add(TM);
                                        bp = bp.add(TN); _k += 1;
                                    }
                                    $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n), [<c_ $R _0>]);
                                      $crate::simd_primitive!($isa, $elem, storeu, cp.add((m+$R)*n_size+n+$LANES), [<c_ $R _1>]);)+
                                }}
                                si_local += 1;
                            }
                            m += TM;
                        }
                        WS_A4.with(|c| c.set(pa));
                    };
                    if num_blocks >= 2 && nthreads > 1 {
                        use rayon::prelude::*;
                        (0..num_blocks).into_par_iter().for_each(|bi| mc_body(bi));
                    } else {
                        for bi in 0..num_blocks { mc_body(bi); }
                    }
                    strip_start += nc_strips;
                } // NC inner loop
                ks += kc_max; ch += 1;
            } // KC outer loop

            // ── N-remainder with bias ──
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
                            let kc = kc_max.min(k_size-ks2);
                            let mut bpp = pb.as_ptr().add(ci*cs+ls*kc_max*TN+nos);
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
                            let kc = kc_max.min(k_size-k);
                            let base = ci*cs+ls*kc_max*TN;
                            for ki in 0..kc { s = <$elem as Element>::mul_add(s, a[m*k_size+k+ki], pb[base+ki*TN+no]); }
                            k += kc; ci += 1;
                        }
                        c[m*n_size+nm+no] = s;
                    }
                }
            }
            // ── M-remainder with bias ──
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
                        let kc = kc_max.min(k_size-ks2);
                        let mut bpp = pb.as_ptr().add(ci*cs+si*kc_max*TN);
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

        // ══════════════════════════════════════════════════════════════════
        // Fused GEMM+bias+activation: C = act(A*B + bias)
        // Activation applied in-register before writeback (nopack path)
        // or immediately after last kc_max chunk (packed path, C still in cache).
        // ══════════════════════════════════════════════════════════════════

        // ── Fused nopack: C = act(A*B + bias), activation in-register ──
        // When M is small and N is large, parallelizes across the N dimension using rayon.
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_bias_act_nopack_impl<const ACT: u8>(
            a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
            m_size: usize, n_size: usize, k_size: usize,
        ) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            // ── N-parallel dispatch ──
            let nthreads = rayon::current_num_threads().max(1);
            if n_size >= TN * 4 && nthreads > 1 {
                let n_per_thread = ((n_size + nthreads - 1) / nthreads + TN - 1) / TN * TN;
                let num_n_blocks = (n_size + n_per_thread - 1) / n_per_thread;
                let cp_addr = c.as_mut_ptr() as usize;
                let ap_addr = a.as_ptr() as usize;
                let bp_addr = b.as_ptr() as usize;
                let biasp_addr = bias.as_ptr() as usize;

                use rayon::prelude::*;
                (0..num_n_blocks).into_par_iter().for_each(|ni| {
                    let n_start = ni * n_per_thread;
                    let n_end = (n_start + n_per_thread).min(n_size);
                    let ap = ap_addr as *const $elem;
                    let bp = bp_addr as *const $elem;
                    let cp = cp_addr as *mut $elem;
                    let biasp = biasp_addr as *const $elem;
                    unsafe {
                        x86_matmul_bias_act_nopack_range::<ACT>(ap, bp, biasp, cp, m_size, n_size, k_size, n_start, n_end);
                    }
                });
                return;
            }

            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = c.as_mut_ptr();
            let biasp = bias.as_ptr();
            unsafe { x86_matmul_bias_act_nopack_range::<ACT>(ap, bp, biasp, cp, m_size, n_size, k_size, 0, n_size); }
        }

        /// Inner fused bias+act nopack kernel operating on N range [n_start, n_end).
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_matmul_bias_act_nopack_range<const ACT: u8>(
            ap: *const $elem, bp: *const $elem, biasp: *const $elem, cp: *mut $elem,
            m_size: usize, n_size: usize, k_size: usize,
            n_start: usize, n_end: usize,
        ) {
            const TM: usize = $TM;
            const TN: usize = $NV * $LANES;

            let kc_max = _blocking().kc;
            let mut ks = 0usize;
            let mut ch = 0u32;
            let last_ch = (k_size + kc_max - 1) / kc_max - 1;

            while ks < k_size {
                let kc = (k_size - ks).min(kc_max);

            // ── Main TM×TN tiles ──
            let mut m = 0usize;
            while m + TM <= m_size {
                let mut n = n_start;
                while n + TN <= n_end {
                    paste::paste! {
                        // Init: bias on first chunk, load C on subsequent
                        $(
                            let mut [<c_ $R _0>] = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                                else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n)) };
                            let mut [<c_ $R _1>] = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n + $LANES)) }
                                else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n + $LANES)) };
                        )+
                        // 8-way unrolled K-loop with prefetch
                        let mut ki = 0usize;
                        let ku = kc & !7;
                        while ki < ku {
                            let k = ks + ki;
                            // Prefetch B 16 rows ahead, A 32 elements ahead
                            $crate::simd_primitive!($isa, $elem, prefetch, bp.wrapping_add((k + 16) * n_size + n) as *const i8, 0);
                            $($crate::simd_primitive!($isa, $elem, prefetch, ap.wrapping_add((m + $R) * k_size + k + 32) as *const i8, 0);)+
                            // k+0
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+1
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+1) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+1) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 1));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+2
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+2) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+2) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 2));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+3
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+3) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+3) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 3));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // Mid-group prefetch (every 4 steps)
                            $crate::simd_primitive!($isa, $elem, prefetch, bp.wrapping_add((k + 20) * n_size + n) as *const i8, 0);
                            // k+4
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+4) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+4) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 4));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+5
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+5) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+5) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 5));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+6
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+6) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+6) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 6));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            // k+7
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+7) * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add((k+7) * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k + 7));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            ki += 8;
                        }
                        // Remainder
                        while ki < kc {
                            let k = ks + ki;
                            let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                            $(
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                [<c_ $R _0>] = $crate::simd_primitive!($isa, $elem, fma, va, vb0, [<c_ $R _0>]);
                                [<c_ $R _1>] = $crate::simd_primitive!($isa, $elem, fma, va, vb1, [<c_ $R _1>]);
                            )+
                            ki += 1;
                        }
                        if ch as usize == last_ch {
                            $({
                                let r0 = match ACT {
                                    0 => $crate::apply_act!($isa, $elem, [<c_ $R _0>], relu),
                                    1 => $crate::apply_act!($isa, $elem, [<c_ $R _0>], silu),
                                    _ => $crate::apply_act!($isa, $elem, [<c_ $R _0>], gelu),
                                };
                                let r1 = match ACT {
                                    0 => $crate::apply_act!($isa, $elem, [<c_ $R _1>], relu),
                                    1 => $crate::apply_act!($isa, $elem, [<c_ $R _1>], silu),
                                    _ => $crate::apply_act!($isa, $elem, [<c_ $R _1>], gelu),
                                };
                                $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), r0);
                                $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n + $LANES), r1);
                            })+
                        } else {
                            $($crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), [<c_ $R _0>]);
                              $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n + $LANES), [<c_ $R _1>]);)+
                        }
                    }
                    n += TN;
                }
                // N-remainder: LANES-wide
                while n + $LANES <= n_end {
                    $(
                        let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                            else { $crate::simd_primitive!($isa, $elem, loadu, cp.add((m + $R) * n_size + n)) };
                        for ki in 0..kc {
                            let k = ks + ki;
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                            let vb = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        if ch as usize == last_ch {
                            let r = match ACT {
                                0 => $crate::apply_act!($isa, $elem, acc, relu),
                                1 => $crate::apply_act!($isa, $elem, acc, silu),
                                _ => $crate::apply_act!($isa, $elem, acc, gelu),
                            };
                            $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), r);
                        } else {
                            $crate::simd_primitive!($isa, $elem, storeu, cp.add((m + $R) * n_size + n), acc);
                        }
                    )+
                    n += $LANES;
                }
                // N-remainder: masked
                {
                    let rem = n_end - n;
                    if rem > 0 { unsafe {
                        $(
                        {
                            let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, maskload, biasp.add(n), rem) }
                                else { $crate::simd_primitive!($isa, $elem, maskload, cp.add((m + $R) * n_size + n), rem) };
                            for ki in 0..kc {
                                let k = ks + ki;
                                let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add((m + $R) * k_size + k));
                                let vb = $crate::simd_primitive!($isa, $elem, maskload, bp.add(k * n_size + n), rem);
                                acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                            }
                            if ch as usize == last_ch {
                                let r = match ACT {
                                    0 => $crate::apply_act!($isa, $elem, acc, relu),
                                    1 => $crate::apply_act!($isa, $elem, acc, silu),
                                    _ => $crate::apply_act!($isa, $elem, acc, gelu),
                                };
                                $crate::simd_primitive!($isa, $elem, maskstore, cp.add((m + $R) * n_size + n), r, rem);
                            } else {
                                $crate::simd_primitive!($isa, $elem, maskstore, cp.add((m + $R) * n_size + n), acc, rem);
                            }
                        }
                        )+
                    }}
                }
                m += TM;
            }
            // M-remainder: 1×TN
            while m < m_size {
                let mut n = n_start;
                while n + TN <= n_end {
                    let mut c0 = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n)) };
                    let mut c1 = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n + $LANES)) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n + $LANES)) };
                    for ki in 0..kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n + $LANES));
                        c0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, c0);
                        c1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, c1);
                    }
                    if ch as usize == last_ch {
                        let r0 = match ACT {
                            0 => $crate::apply_act!($isa, $elem, c0, relu),
                            1 => $crate::apply_act!($isa, $elem, c0, silu),
                            _ => $crate::apply_act!($isa, $elem, c0, gelu),
                        };
                        let r1 = match ACT {
                            0 => $crate::apply_act!($isa, $elem, c1, relu),
                            1 => $crate::apply_act!($isa, $elem, c1, silu),
                            _ => $crate::apply_act!($isa, $elem, c1, gelu),
                        };
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), r0);
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n + $LANES), r1);
                    } else {
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), c0);
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n + $LANES), c1);
                    }
                    n += TN;
                }
                // M-remainder: LANES-wide
                while n + $LANES <= n_end {
                    let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, loadu, biasp.add(n)) }
                        else { $crate::simd_primitive!($isa, $elem, loadu, cp.add(m * n_size + n)) };
                    for ki in 0..kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                        let vb = $crate::simd_primitive!($isa, $elem, loadu, bp.add(k * n_size + n));
                        acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                    }
                    if ch as usize == last_ch {
                        let r = match ACT {
                            0 => $crate::apply_act!($isa, $elem, acc, relu),
                            1 => $crate::apply_act!($isa, $elem, acc, silu),
                            _ => $crate::apply_act!($isa, $elem, acc, gelu),
                        };
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), r);
                    } else {
                        $crate::simd_primitive!($isa, $elem, storeu, cp.add(m * n_size + n), acc);
                    }
                    n += $LANES;
                }
                // M-remainder: masked
                {
                    let rem = n_end - n;
                    if rem > 0 {
                        let mut acc = if ch == 0 { $crate::simd_primitive!($isa, $elem, maskload, biasp.add(n), rem) }
                            else { $crate::simd_primitive!($isa, $elem, maskload, cp.add(m * n_size + n), rem) };
                        for ki in 0..kc {
                            let k = ks + ki;
                            let va = $crate::simd_primitive!($isa, $elem, splat, *ap.add(m * k_size + k));
                            let vb = $crate::simd_primitive!($isa, $elem, maskload, bp.add(k * n_size + n), rem);
                            acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                        }
                        if ch as usize == last_ch {
                            let r = match ACT {
                                0 => $crate::apply_act!($isa, $elem, acc, relu),
                                1 => $crate::apply_act!($isa, $elem, acc, silu),
                                _ => $crate::apply_act!($isa, $elem, acc, gelu),
                            };
                            $crate::simd_primitive!($isa, $elem, maskstore, cp.add(m * n_size + n), r, rem);
                        } else {
                            $crate::simd_primitive!($isa, $elem, maskstore, cp.add(m * n_size + n), acc, rem);
                        }
                    }
                }
                m += 1;
            }

            ks += kc; ch += 1;
            } // KC loop
        }

        /// Dispatch matmul_bias_act to the appropriate fused implementation.
        #[inline(always)]
        pub fn matmul_bias_act(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                               m_size: usize, n_size: usize, k_size: usize, act: $crate::Activation) {
            match act {
                $crate::Activation::None => matmul_bias(a, b, bias, c, m_size, n_size, k_size),
                $crate::Activation::Relu => {
                    if m_size <= SMALL_M_THRESHOLD {
                        unsafe { x86_matmul_bias_act_nopack_impl::<{0u8}>(a, b, bias, c, m_size, n_size, k_size); }
                    } else {
                        matmul_bias(a, b, bias, c, m_size, n_size, k_size);
                        unsafe { x86_apply_act_inplace::<{0u8}>(c, m_size * n_size); }
                    }
                },
                $crate::Activation::Silu => {
                    if m_size <= SMALL_M_THRESHOLD {
                        unsafe { x86_matmul_bias_act_nopack_impl::<{1u8}>(a, b, bias, c, m_size, n_size, k_size); }
                    } else {
                        matmul_bias(a, b, bias, c, m_size, n_size, k_size);
                        unsafe { x86_apply_act_inplace::<{1u8}>(c, m_size * n_size); }
                    }
                },
                $crate::Activation::Gelu => {
                    if m_size <= SMALL_M_THRESHOLD {
                        unsafe { x86_matmul_bias_act_nopack_impl::<{2u8}>(a, b, bias, c, m_size, n_size, k_size); }
                    } else {
                        matmul_bias(a, b, bias, c, m_size, n_size, k_size);
                        unsafe { x86_apply_act_inplace::<{2u8}>(c, m_size * n_size); }
                    }
                },
            }
        }

        /// In-place activation pass over C (used for packed path where C is still hot in cache).
        #[target_feature($(enable = $feat),+)]
        unsafe fn x86_apply_act_inplace<const ACT: u8>(c: &mut [$elem], len: usize) {
            const LANES: usize = $NV * $LANES;
            let cp = c.as_mut_ptr();
            let mut i = 0usize;
            while i + LANES <= len {
                let v0 = $crate::simd_primitive!($isa, $elem, loadu, cp.add(i));
                let v1 = $crate::simd_primitive!($isa, $elem, loadu, cp.add(i + $LANES));
                let r0 = match ACT {
                    0 => $crate::apply_act!($isa, $elem, v0, relu),
                    1 => $crate::apply_act!($isa, $elem, v0, silu),
                    _ => $crate::apply_act!($isa, $elem, v0, gelu),
                };
                let r1 = match ACT {
                    0 => $crate::apply_act!($isa, $elem, v1, relu),
                    1 => $crate::apply_act!($isa, $elem, v1, silu),
                    _ => $crate::apply_act!($isa, $elem, v1, gelu),
                };
                $crate::simd_primitive!($isa, $elem, storeu, cp.add(i), r0);
                $crate::simd_primitive!($isa, $elem, storeu, cp.add(i + $LANES), r1);
                i += LANES;
            }
            while i + $LANES <= len {
                let v = $crate::simd_primitive!($isa, $elem, loadu, cp.add(i));
                let r = match ACT {
                    0 => $crate::apply_act!($isa, $elem, v, relu),
                    1 => $crate::apply_act!($isa, $elem, v, silu),
                    _ => $crate::apply_act!($isa, $elem, v, gelu),
                };
                $crate::simd_primitive!($isa, $elem, storeu, cp.add(i), r);
                i += $LANES;
            }
            while i < len {
                let val = *cp.add(i);
                *cp.add(i) = match ACT {
                    0 => <$elem as Element>::max(val, <$elem as Element>::ZERO),
                    1 => {
                        let v = val.to_f32();
                        <$elem as Element>::from_f32(v / (1.0 + (-v).exp()))
                    },
                    _ => {
                        let x = val.to_f32();
                        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
                        <$elem as Element>::from_f32(0.5 * x * (1.0 + inner.tanh()))
                    },
                };
                i += 1;
            }
        }
    };
}
