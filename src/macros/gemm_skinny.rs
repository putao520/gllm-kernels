/// Skinny GEMM macro with K-tiling + B micro-packing + 2×LANES N-strip + software prefetch.
///
/// Computes C[M×N] = A[M×K] * B[K×N] for small-batch LLM inference (M=2..32).
///
/// Key optimizations:
/// - K-tiling: KC auto-tuned per ISA so pack buffer fits half L1d (16KB)
/// - B micro-packing: packs B[KC×TN] strip into contiguous buffer per tile
/// - 2×LANES N-strip (TN): doubles N-width per strip, halving broadcast pressure
///   m4×2LANES: 4 broadcasts + 8 FMAs per K iter (port 5 and FMA ports balanced)
///   vs m8×1LANES: 8 broadcasts + 8 FMAs (port 5 bottleneck on AVX2)
/// - Software prefetch in B packing: stride-N access defeats HW prefetcher
/// - 4x K-unroll in m4_2x for ILP
///
/// Register budget (m4×2LANES):
///   8 acc (4 rows × 2 vectors) + 2 B loads + 1 A broadcast = 11 regs
///   Fits AVX2 (16 YMM), AVX-512 (32 ZMM), NEON (32 regs)
///
/// KC auto-tuning (pack_buf = KC × TN × sizeof(elem) ≤ 16KB):
///   AVX2 f32 (LANES=8):   KC = 256, pack = 16KB
///   AVX-512 f32 (LANES=16): KC = 128, pack = 16KB
///   NEON f32 (LANES=4):   KC = 512, pack = 16KB

/// Helper: generate a 2×LANES skinny kernel for $R rows with $U-way K-unroll.
///
/// Each row has 2 accumulator vectors (lo + hi), processing TN = 2×LANES columns.
/// A scalar broadcast is shared across both lo and hi FMAs (halving port 5 pressure).
#[macro_export]
#[doc(hidden)]
macro_rules! define_skinny_mN_2x {
    (
        $isa:ident, $elem:ident, $LANES:literal, [$($feat:literal),+],
        fn $name:ident, rows = $R:literal, unroll = $U:literal
    ) => {
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn $name(
            a_ptr: *const $elem, pack_ptr: *const $elem, c_ptr: *mut $elem,
            i_start: usize, j: usize, n: usize, k: usize,
            kc: usize, kc_len: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;

            // Load 2 partial sums per row from C & compute A row base pointers
            seq_macro::seq!(R in 0..$R {
                let mut acc_lo~R = $crate::simd_primitive!($isa, $elem, loadu,
                    c_ptr.add((i_start + R) * n + j));
                let mut acc_hi~R = $crate::simd_primitive!($isa, $elem, loadu,
                    c_ptr.add((i_start + R) * n + j + LANES));
                let a~R = a_ptr.add((i_start + R) * k + kc);
            });

            // U-way unrolled K loop
            let ku = kc_len & !($U - 1);
            let mut ki = 0usize;
            while ki < ku {
                seq_macro::seq!(U in 0..$U {
                    let vb_lo~U = $crate::simd_primitive!($isa, $elem, loadu,
                        pack_ptr.add((ki + U) * TN));
                    let vb_hi~U = $crate::simd_primitive!($isa, $elem, loadu,
                        pack_ptr.add((ki + U) * TN + LANES));
                    seq_macro::seq!(R in 0..$R {
                        // Broadcast A[row][ki+U] once, reuse for both lo and hi FMAs
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a~R.add(ki + U));
                        acc_lo~R = $crate::simd_primitive!($isa, $elem, fma, va, vb_lo~U, acc_lo~R);
                        acc_hi~R = $crate::simd_primitive!($isa, $elem, fma, va, vb_hi~U, acc_hi~R);
                    });
                });
                ki += $U;
            }

            // K remainder
            while ki < kc_len {
                let vb_lo = $crate::simd_primitive!($isa, $elem, loadu,
                    pack_ptr.add(ki * TN));
                let vb_hi = $crate::simd_primitive!($isa, $elem, loadu,
                    pack_ptr.add(ki * TN + LANES));
                seq_macro::seq!(R in 0..$R {
                    let va = $crate::simd_primitive!($isa, $elem, splat, *a~R.add(ki));
                    acc_lo~R = $crate::simd_primitive!($isa, $elem, fma, va, vb_lo, acc_lo~R);
                    acc_hi~R = $crate::simd_primitive!($isa, $elem, fma, va, vb_hi, acc_hi~R);
                });
                ki += 1;
            }

            // Store 2 vectors per row back to C
            seq_macro::seq!(R in 0..$R {
                $crate::simd_primitive!($isa, $elem, storeu,
                    c_ptr.add((i_start + R) * n + j), acc_lo~R);
                $crate::simd_primitive!($isa, $elem, storeu,
                    c_ptr.add((i_start + R) * n + j + LANES), acc_hi~R);
            });
        }
    };
}

/// Helper: generate a 1×LANES skinny kernel for $R rows with $U-way K-unroll.
/// Used for N-remainder when N % TN >= LANES.
#[macro_export]
#[doc(hidden)]
macro_rules! define_skinny_mN_1x {
    (
        $isa:ident, $elem:ident, $LANES:literal, [$($feat:literal),+],
        fn $name:ident, rows = $R:literal, unroll = $U:literal
    ) => {
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn $name(
            a_ptr: *const $elem, pack_ptr: *const $elem, c_ptr: *mut $elem,
            i_start: usize, j: usize, n: usize, k: usize,
            kc: usize, kc_len: usize,
        ) {
            const LANES: usize = $LANES;

            seq_macro::seq!(R in 0..$R {
                let mut acc~R = $crate::simd_primitive!($isa, $elem, loadu,
                    c_ptr.add((i_start + R) * n + j));
                let a~R = a_ptr.add((i_start + R) * k + kc);
            });

            let ku = kc_len & !($U - 1);
            let mut ki = 0usize;
            while ki < ku {
                seq_macro::seq!(U in 0..$U {
                    let vb~U = $crate::simd_primitive!($isa, $elem, loadu,
                        pack_ptr.add((ki + U) * LANES));
                    seq_macro::seq!(R in 0..$R {
                        acc~R = $crate::simd_primitive!($isa, $elem, fma,
                            $crate::simd_primitive!($isa, $elem, splat, *a~R.add(ki + U)),
                            vb~U, acc~R);
                    });
                });
                ki += $U;
            }

            while ki < kc_len {
                let vb = $crate::simd_primitive!($isa, $elem, loadu,
                    pack_ptr.add(ki * LANES));
                seq_macro::seq!(R in 0..$R {
                    acc~R = $crate::simd_primitive!($isa, $elem, fma,
                        $crate::simd_primitive!($isa, $elem, splat, *a~R.add(ki)),
                        vb, acc~R);
                });
                ki += 1;
            }

            seq_macro::seq!(R in 0..$R {
                $crate::simd_primitive!($isa, $elem, storeu,
                    c_ptr.add((i_start + R) * n + j), acc~R);
            });
        }
    };
}

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

            // Zero C first (we accumulate into it across KC tiles)
            for v in c[..m * n].iter_mut() { *v = <$elem as Element>::ZERO; }

            unsafe {
                gemm_skinny_inner(
                    a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                    m, n, k,
                );
            }
        }

        /// Inner skinny GEMM kernel with K-tiling + 2×LANES N-strip + B micro-packing.
        ///
        /// Loop order: K-tile → N-strip (2×LANES) → M-tile (m4)
        /// For each KC tile, pack B strip with prefetch, then process all M rows.
        #[target_feature($(enable = $feat),+)]
        unsafe fn gemm_skinny_inner(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            m: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;
            // KC auto-tuned: pack_buf = KC × TN × sizeof(elem) ≤ 16KB (half L1d)
            const KC: usize = 16384 / (TN * std::mem::size_of::<$elem>());
            // Panel width: pack NP strips per B-row visit to reduce DRAM row activations.
            // NP=4: reads 4×TN contiguous bytes per row (256B for AVX2 f32).
            // Total panel pack buffer: NP × KC × TN × sizeof = 64KB, fits L2.
            const NP: usize = 4;

            // Pack buffers on stack, reused across all KC tiles
            // panel_bufs[0] doubles as per-strip pack buffer; [1..NP] used for panel packing
            let mut panel_bufs = [[<$elem as Element>::ZERO; KC * TN]; NP];
            let mut pack_buf_1x = [<$elem as Element>::ZERO; KC * LANES];
            let use_panel = n * std::mem::size_of::<$elem>() > 32768;

            // K-tiling: process K in chunks of KC
            let mut kc = 0usize;
            while kc < k {
                let kc_len = if kc + KC <= k { KC } else { k - kc };

                // ── Primary: 2×LANES N-strips with packed B ──
                let n_full_strips = n / TN;
                let mut j = 0usize;
                let mut strip = 0usize;

                // Panel packing for large N: reduces DRAM row activations
                if use_panel {
                    while strip + NP <= n_full_strips {
                        pack_b_panel_4(b_ptr,
                            panel_bufs[0].as_mut_ptr(), panel_bufs[1].as_mut_ptr(),
                            panel_bufs[2].as_mut_ptr(), panel_bufs[3].as_mut_ptr(),
                            kc, kc_len, j, n);

                        for s in 0..NP {
                            let js = j + s * TN;
                            let pp = panel_bufs[s].as_ptr();
                            let mut i = 0usize;
                            while i + 4 <= m {
                                skinny_m4_2x(a_ptr, pp, c_ptr, i, js, n, k, kc, kc_len);
                                i += 4;
                            }
                            if i + 2 <= m {
                                skinny_m2_2x(a_ptr, pp, c_ptr, i, js, n, k, kc, kc_len);
                                i += 2;
                            }
                            if i < m {
                                skinny_m1_2x(a_ptr, pp, c_ptr, i, js, n, k, kc, kc_len);
                            }
                        }

                        j += NP * TN;
                        strip += NP;
                    }
                }

                // ── Remaining (or all, for small N) 2×LANES strips ──
                while j + TN <= n {
                    pack_b_strip_2x(b_ptr, panel_bufs[0].as_mut_ptr(), kc, kc_len, j, n);

                    let mut i = 0usize;
                    while i + 4 <= m {
                        skinny_m4_2x(a_ptr, panel_bufs[0].as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 4;
                    }
                    if i + 2 <= m {
                        skinny_m2_2x(a_ptr, panel_bufs[0].as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 2;
                    }
                    if i < m {
                        skinny_m1_2x(a_ptr, panel_bufs[0].as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                    }

                    j += TN;
                }

                // ── Remainder: 1×LANES N-strip ──
                if j + LANES <= n {
                    pack_b_strip_1x(b_ptr, pack_buf_1x.as_mut_ptr(), kc, kc_len, j, n);

                    let mut i = 0usize;
                    while i + 4 <= m {
                        skinny_m4_1x(a_ptr, pack_buf_1x.as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 4;
                    }
                    if i + 2 <= m {
                        skinny_m2_1x(a_ptr, pack_buf_1x.as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 2;
                    }
                    if i < m {
                        skinny_m1_1x(a_ptr, pack_buf_1x.as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                    }

                    j += LANES;
                }

                // ── Scalar tail: remaining columns < LANES ──
                while j < n {
                    for i in 0..m {
                        let mut sum = *c_ptr.add(i * n + j);
                        for ki in 0..kc_len {
                            sum = <$elem as Element>::mul_add(sum,
                                *a_ptr.add(i * k + kc + ki),
                                *b_ptr.add((kc + ki) * n + j));
                        }
                        *c_ptr.add(i * n + j) = sum;
                    }
                    j += 1;
                }

                kc += KC;
            }
        }

        /// Pack B[kc..kc+kc_len][j..j+TN] into contiguous buffer (2×LANES wide).
        /// Software prefetch 8 rows ahead — stride-N access defeats HW prefetcher.
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn pack_b_strip_2x(
            b_ptr: *const $elem, pack_ptr: *mut $elem,
            kc: usize, kc_len: usize, j: usize, n: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;
            const PF_AHEAD: usize = 8;

            for ki in 0..kc_len {
                if ki + PF_AHEAD < kc_len {
                    $crate::simd_primitive!($isa, $elem, prefetch,
                        b_ptr.add((kc + ki + PF_AHEAD) * n + j), 0);
                }
                std::ptr::copy_nonoverlapping(
                    b_ptr.add((kc + ki) * n + j),
                    pack_ptr.add(ki * TN),
                    TN,
                );
            }
        }

        /// Pack B[kc..kc+kc_len][j..j+LANES] into contiguous buffer (1×LANES wide).
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn pack_b_strip_1x(
            b_ptr: *const $elem, pack_ptr: *mut $elem,
            kc: usize, kc_len: usize, j: usize, n: usize,
        ) {
            const LANES: usize = $LANES;
            const PF_AHEAD: usize = 8;

            for ki in 0..kc_len {
                if ki + PF_AHEAD < kc_len {
                    $crate::simd_primitive!($isa, $elem, prefetch,
                        b_ptr.add((kc + ki + PF_AHEAD) * n + j), 0);
                }
                std::ptr::copy_nonoverlapping(
                    b_ptr.add((kc + ki) * n + j),
                    pack_ptr.add(ki * LANES),
                    LANES,
                );
            }
        }

        /// Pack NP strips in one pass over B rows (panel packing).
        /// Reads NP×TN contiguous bytes per row → better DRAM row buffer utilization.
        /// Reduces DRAM row activations by NP× vs per-strip packing.
        ///
        /// pack_ptrs: array of NP pointers to separate pack buffers (each KC×TN).
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn pack_b_panel_4(
            b_ptr: *const $elem,
            pp0: *mut $elem, pp1: *mut $elem, pp2: *mut $elem, pp3: *mut $elem,
            kc: usize, kc_len: usize, j_panel: usize, n: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;
            const PF_AHEAD: usize = 8;

            for ki in 0..kc_len {
                let row = (kc + ki) * n + j_panel;
                // Prefetch: one prefetch covers the whole panel row (sequential within page)
                if ki + PF_AHEAD < kc_len {
                    $crate::simd_primitive!($isa, $elem, prefetch,
                        b_ptr.add((kc + ki + PF_AHEAD) * n + j_panel), 0);
                }
                // Copy 4 contiguous strips from this B row into separate pack buffers
                let src = b_ptr.add(row);
                let off = ki * TN;
                std::ptr::copy_nonoverlapping(src,            pp0.add(off), TN);
                std::ptr::copy_nonoverlapping(src.add(TN),    pp1.add(off), TN);
                std::ptr::copy_nonoverlapping(src.add(TN * 2), pp2.add(off), TN);
                std::ptr::copy_nonoverlapping(src.add(TN * 3), pp3.add(off), TN);
            }
        }


        // ==================================================================
        // B-transposed variant: b_t is B^T[N×K], pack via transpose then
        // reuse the same broadcast-A/vector-B kernels as the non-transposed path.
        // ==================================================================

        /// Skinny GEMM with transposed B: C[M×N] = A[M×K] * B^T[N×K], M in 2..=32.
        ///
        /// Transpose-packs B^T strips into ki-major layout identical to pack_b_strip_2x,
        /// then reuses the same broadcast-A/vector-B compute kernels (skinny_m4_2x etc.).
        /// B^T rows are K-contiguous, so packing reads are sequential (no stride-N misses).
        #[inline(always)]
        pub fn gemm_skinny_bt(a: &[$elem], b_t: &[$elem], c: &mut [$elem],
                               m: usize, n: usize, k: usize) {
            debug_assert!(m >= 2 && m <= 32);
            debug_assert!(a.len() >= m * k);
            debug_assert!(b_t.len() >= n * k);
            debug_assert!(c.len() >= m * n);

            for v in c[..m * n].iter_mut() { *v = <$elem as Element>::ZERO; }

            unsafe {
                gemm_skinny_inner_bt(
                    a.as_ptr(), b_t.as_ptr(), c.as_mut_ptr(),
                    m, n, k,
                );
            }
        }

        /// Inner bt kernel: adaptive dispatch between two strategies.
        ///
        /// - M ≤ 8: j-tiled row-dot (pack-free, FMA:load=2:1, zero packing overhead)
        /// - M > 8: transpose-pack + broadcast kernels (higher compute efficiency,
        ///   packing cost amortized over many M rows)
        #[target_feature($(enable = $feat),+)]
        unsafe fn gemm_skinny_inner_bt(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            m: usize, n: usize, k: usize,
        ) {
            // N-dependent dispatch: transpose-pack cost scales with N,
            // so for large N the pack-free bt_dot path wins at higher M.
            // Threshold: M≤8 always bt_dot; M≤16 bt_dot when N>8192; else pack.
            let use_dot = if n > 8192 { m <= 16 } else { m <= 8 };
            if use_dot {
                gemm_skinny_inner_bt_dot(a_ptr, b_t_ptr, c_ptr, m, n, k);
            } else {
                gemm_skinny_inner_bt_pack(a_ptr, b_t_ptr, c_ptr, m, n, k);
            }
        }

        /// Pack-free j-tiled row-dot path for small M (≤8).
        /// No packing overhead. FMA:load = 2:1 with 4-column j-tiling.
        #[target_feature($(enable = $feat),+)]
        unsafe fn gemm_skinny_inner_bt_dot(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            m: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;
            const KC: usize = 16384 / (4 * std::mem::size_of::<$elem>());

            let mut kc = 0usize;
            while kc < k {
                let kc_end = if kc + KC <= k { kc + KC } else { k };

                let mut i = 0usize;
                while i + 4 <= m {
                    let mut j = 0usize;
                    while j + 4 <= n {
                        bt_dot_m4_j4(a_ptr, b_t_ptr, c_ptr, i, j, n, k, kc, kc_end);
                        j += 4;
                    }
                    while j < n {
                        bt_dot_m4_j1(a_ptr, b_t_ptr, c_ptr, i, j, n, k, kc, kc_end);
                        j += 1;
                    }
                    i += 4;
                }
                if i + 2 <= m {
                    let mut j = 0usize;
                    while j + 4 <= n {
                        bt_dot_m2_j4(a_ptr, b_t_ptr, c_ptr, i, j, n, k, kc, kc_end);
                        j += 4;
                    }
                    while j < n {
                        bt_dot_m2_j1(a_ptr, b_t_ptr, c_ptr, i, j, n, k, kc, kc_end);
                        j += 1;
                    }
                    i += 2;
                }
                while i < m {
                    let mut j = 0usize;
                    while j + 4 <= n {
                        bt_dot_m1_j4(a_ptr, b_t_ptr, c_ptr, i, j, n, k, kc, kc_end);
                        j += 4;
                    }
                    while j < n {
                        bt_dot_m1_j1(a_ptr, b_t_ptr, c_ptr, i, j, n, k, kc, kc_end);
                        j += 1;
                    }
                    i += 1;
                }

                kc += KC;
            }
        }

        /// Transpose-pack + broadcast path for large M (>8).
        /// Packing cost amortized over many rows; broadcast kernel has no reduce_sum.
        #[target_feature($(enable = $feat),+)]
        unsafe fn gemm_skinny_inner_bt_pack(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            m: usize, n: usize, k: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;
            const KC: usize = 16384 / (TN * std::mem::size_of::<$elem>());
            const NP: usize = 4;

            let mut panel_bufs = [[<$elem as Element>::ZERO; KC * TN]; NP];
            let mut pack_buf_1x = [<$elem as Element>::ZERO; KC * LANES];
            let use_panel = n * std::mem::size_of::<$elem>() > 32768;

            let mut kc = 0usize;
            while kc < k {
                let kc_len = if kc + KC <= k { KC } else { k - kc };

                let n_full_strips = n / TN;
                let mut j = 0usize;
                let mut strip = 0usize;

                if use_panel {
                    while strip + NP <= n_full_strips {
                        pack_bt_panel_4(b_t_ptr,
                            panel_bufs[0].as_mut_ptr(), panel_bufs[1].as_mut_ptr(),
                            panel_bufs[2].as_mut_ptr(), panel_bufs[3].as_mut_ptr(),
                            kc, kc_len, j, k);

                        for s in 0..NP {
                            let js = j + s * TN;
                            let pp = panel_bufs[s].as_ptr();
                            let mut i = 0usize;
                            while i + 4 <= m {
                                skinny_m4_2x(a_ptr, pp, c_ptr, i, js, n, k, kc, kc_len);
                                i += 4;
                            }
                            if i + 2 <= m {
                                skinny_m2_2x(a_ptr, pp, c_ptr, i, js, n, k, kc, kc_len);
                                i += 2;
                            }
                            if i < m {
                                skinny_m1_2x(a_ptr, pp, c_ptr, i, js, n, k, kc, kc_len);
                            }
                        }

                        j += NP * TN;
                        strip += NP;
                    }
                }

                while j + TN <= n {
                    pack_bt_strip_2x(b_t_ptr, panel_bufs[0].as_mut_ptr(), kc, kc_len, j, k);

                    let mut i = 0usize;
                    while i + 4 <= m {
                        skinny_m4_2x(a_ptr, panel_bufs[0].as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 4;
                    }
                    if i + 2 <= m {
                        skinny_m2_2x(a_ptr, panel_bufs[0].as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 2;
                    }
                    if i < m {
                        skinny_m1_2x(a_ptr, panel_bufs[0].as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                    }

                    j += TN;
                }

                if j + LANES <= n {
                    pack_bt_strip_1x(b_t_ptr, pack_buf_1x.as_mut_ptr(), kc, kc_len, j, k);

                    let mut i = 0usize;
                    while i + 4 <= m {
                        skinny_m4_1x(a_ptr, pack_buf_1x.as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 4;
                    }
                    if i + 2 <= m {
                        skinny_m2_1x(a_ptr, pack_buf_1x.as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                        i += 2;
                    }
                    if i < m {
                        skinny_m1_1x(a_ptr, pack_buf_1x.as_ptr(), c_ptr, i, j, n, k, kc, kc_len);
                    }

                    j += LANES;
                }

                while j < n {
                    for i in 0..m {
                        let mut sum = *c_ptr.add(i * n + j);
                        for ki in 0..kc_len {
                            sum = <$elem as Element>::mul_add(sum,
                                *a_ptr.add(i * k + kc + ki),
                                *b_t_ptr.add(j * k + kc + ki));
                        }
                        *c_ptr.add(i * n + j) = sum;
                    }
                    j += 1;
                }

                kc += KC;
            }
        }

        /// Transpose-pack B^T[j..j+TN][kc..kc+kc_len] into ki-major pack buffer.
        /// Reads: B^T[j+lane][kc+ki] (stride-K per lane, sequential per ki)
        /// Writes: pack[ki * TN + lane] (sequential)
        /// Pack buffer fits in L1 (<=16KB), so writes are always L1 hits.
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn pack_bt_strip_2x(
            b_t_ptr: *const $elem, pack_ptr: *mut $elem,
            kc: usize, kc_len: usize, j: usize, k: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;

            for ki in 0..kc_len {
                let dst = pack_ptr.add(ki * TN);
                for lane in 0..TN {
                    *dst.add(lane) = *b_t_ptr.add((j + lane) * k + kc + ki);
                }
            }
        }

        /// Transpose-pack B^T[j..j+LANES][kc..kc+kc_len] into ki-major pack buffer (1×LANES).
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn pack_bt_strip_1x(
            b_t_ptr: *const $elem, pack_ptr: *mut $elem,
            kc: usize, kc_len: usize, j: usize, k: usize,
        ) {
            const LANES: usize = $LANES;

            for ki in 0..kc_len {
                let dst = pack_ptr.add(ki * LANES);
                for lane in 0..LANES {
                    *dst.add(lane) = *b_t_ptr.add((j + lane) * k + kc + ki);
                }
            }
        }

        /// Panel transpose-pack: pack 4 consecutive TN-wide strips from B^T in one pass.
        #[target_feature($(enable = $feat),+)]
        #[inline]
        unsafe fn pack_bt_panel_4(
            b_t_ptr: *const $elem,
            p0: *mut $elem, p1: *mut $elem, p2: *mut $elem, p3: *mut $elem,
            kc: usize, kc_len: usize, j: usize, k: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = 2 * LANES;

            for ki in 0..kc_len {
                let off = kc + ki;
                // Strip 0
                let dst0 = p0.add(ki * TN);
                for lane in 0..TN {
                    *dst0.add(lane) = *b_t_ptr.add((j + lane) * k + off);
                }
                // Strip 1
                let dst1 = p1.add(ki * TN);
                for lane in 0..TN {
                    *dst1.add(lane) = *b_t_ptr.add((j + TN + lane) * k + off);
                }
                // Strip 2
                let dst2 = p2.add(ki * TN);
                for lane in 0..TN {
                    *dst2.add(lane) = *b_t_ptr.add((j + 2 * TN + lane) * k + off);
                }
                // Strip 3
                let dst3 = p3.add(ki * TN);
                for lane in 0..TN {
                    *dst3.add(lane) = *b_t_ptr.add((j + 3 * TN + lane) * k + off);
                }
            }
        }

        // ── B-transposed dot-product kernels ──
        // These compute C[rows × cols] += A[rows × K] · B_t[cols × K]
        // where B is stored transposed (row = output column).
        // Each kernel: SIMD dot loop + scalar tail + f32 accumulate into C.

        /// bt_dot_m4_j4: 4 rows × 4 columns dot product
        /// Optimized: per row, 4 cols simultaneous + 2-way K-unroll (8 acc vectors)
        #[target_feature($(enable = $feat),+)]
        unsafe fn bt_dot_m4_j4(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            i: usize, j: usize, n: usize, k: usize, kc: usize, kc_end: usize,
        ) {
            const LANES: usize = $LANES;
            let kc_len = kc_end - kc;
            let simd_end = kc_len & !(LANES - 1);
            let simd_end2 = kc_len & !(2 * LANES - 1);

            let b0 = b_t_ptr.add(j * k + kc);
            let b1 = b_t_ptr.add((j + 1) * k + kc);
            let b2 = b_t_ptr.add((j + 2) * k + kc);
            let b3 = b_t_ptr.add((j + 3) * k + kc);

            for ri in 0..4usize {
                let a_row = a_ptr.add((i + ri) * k + kc);
                // 4 cols x 2-way K-unroll = 8 accumulators
                let mut ac0_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac0_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac1_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac1_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac2_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac2_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac3_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac3_1 = $crate::simd_primitive!($isa, $elem, zero);

                let mut si = 0usize;
                while si < simd_end2 {
                    let va0 = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si));
                    let va1 = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si + LANES));
                    ac0_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si)), ac0_0);
                    ac0_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si + LANES)), ac0_1);
                    ac1_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si)), ac1_0);
                    ac1_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si + LANES)), ac1_1);
                    ac2_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si)), ac2_0);
                    ac2_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si + LANES)), ac2_1);
                    ac3_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si)), ac3_0);
                    ac3_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si + LANES)), ac3_1);
                    si += 2 * LANES;
                }
                while si < simd_end {
                    let va = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si));
                    ac0_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si)), ac0_0);
                    ac1_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si)), ac1_0);
                    ac2_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si)), ac2_0);
                    ac3_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si)), ac3_0);
                    si += LANES;
                }
                let mut d0: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac0_0, ac0_1));
                let mut d1: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac1_0, ac1_1));
                let mut d2: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac2_0, ac2_1));
                let mut d3: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac3_0, ac3_1));
                let mut ti = simd_end;
                while ti < kc_len {
                    let a_val = (*a_row.add(ti)).to_f32();
                    d0 += a_val * (*b0.add(ti)).to_f32();
                    d1 += a_val * (*b1.add(ti)).to_f32();
                    d2 += a_val * (*b2.add(ti)).to_f32();
                    d3 += a_val * (*b3.add(ti)).to_f32();
                    ti += 1;
                }
                let c_row = c_ptr.add((i + ri) * n + j);
                *c_row += <$elem>::from_f32(d0);
                *c_row.add(1) += <$elem>::from_f32(d1);
                *c_row.add(2) += <$elem>::from_f32(d2);
                *c_row.add(3) += <$elem>::from_f32(d3);
            }
        }

        /// bt_dot_m4_j1: 4 rows × 1 column dot product
        /// Optimized: 4 rows simultaneous, B loaded once reused 4x, 2-way K-unroll (8 acc vectors)
        #[target_feature($(enable = $feat),+)]
        unsafe fn bt_dot_m4_j1(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            i: usize, j: usize, n: usize, k: usize, kc: usize, kc_end: usize,
        ) {
            const LANES: usize = $LANES;
            let kc_len = kc_end - kc;
            let simd_end = kc_len & !(LANES - 1);
            let simd_end2 = kc_len & !(2 * LANES - 1);
            let b_row = b_t_ptr.add(j * k + kc);

            let a0 = a_ptr.add((i) * k + kc);
            let a1 = a_ptr.add((i + 1) * k + kc);
            let a2 = a_ptr.add((i + 2) * k + kc);
            let a3 = a_ptr.add((i + 3) * k + kc);

            // 4 rows x 2-way K-unroll = 8 accumulators
            let mut ac0_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac0_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac2_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac2_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac3_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac3_1 = $crate::simd_primitive!($isa, $elem, zero);

            let mut si = 0usize;
            while si < simd_end2 {
                let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si));
                let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + LANES));
                ac0_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si)), vb0, ac0_0);
                ac0_1 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si + LANES)), vb1, ac0_1);
                ac1_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si)), vb0, ac1_0);
                ac1_1 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si + LANES)), vb1, ac1_1);
                ac2_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a2.add(si)), vb0, ac2_0);
                ac2_1 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a2.add(si + LANES)), vb1, ac2_1);
                ac3_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a3.add(si)), vb0, ac3_0);
                ac3_1 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a3.add(si + LANES)), vb1, ac3_1);
                si += 2 * LANES;
            }
            while si < simd_end {
                let vb = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si));
                ac0_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si)), vb, ac0_0);
                ac1_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si)), vb, ac1_0);
                ac2_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a2.add(si)), vb, ac2_0);
                ac3_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a3.add(si)), vb, ac3_0);
                si += LANES;
            }
            let mut d0: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac0_0, ac0_1));
            let mut d1: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac1_0, ac1_1));
            let mut d2: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac2_0, ac2_1));
            let mut d3: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac3_0, ac3_1));
            let mut ti = simd_end;
            while ti < kc_len {
                let b_val = (*b_row.add(ti)).to_f32();
                d0 += (*a0.add(ti)).to_f32() * b_val;
                d1 += (*a1.add(ti)).to_f32() * b_val;
                d2 += (*a2.add(ti)).to_f32() * b_val;
                d3 += (*a3.add(ti)).to_f32() * b_val;
                ti += 1;
            }
            *c_ptr.add((i) * n + j) += <$elem>::from_f32(d0);
            *c_ptr.add((i + 1) * n + j) += <$elem>::from_f32(d1);
            *c_ptr.add((i + 2) * n + j) += <$elem>::from_f32(d2);
            *c_ptr.add((i + 3) * n + j) += <$elem>::from_f32(d3);
        }

        /// bt_dot_m2_j4: 2 rows × 4 columns dot product
        /// Optimized: per row, 4 cols simultaneous + 2-way K-unroll (8 acc vectors)
        #[target_feature($(enable = $feat),+)]
        unsafe fn bt_dot_m2_j4(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            i: usize, j: usize, n: usize, k: usize, kc: usize, kc_end: usize,
        ) {
            const LANES: usize = $LANES;
            let kc_len = kc_end - kc;
            let simd_end = kc_len & !(LANES - 1);
            let simd_end2 = kc_len & !(2 * LANES - 1);

            let b0 = b_t_ptr.add(j * k + kc);
            let b1 = b_t_ptr.add((j + 1) * k + kc);
            let b2 = b_t_ptr.add((j + 2) * k + kc);
            let b3 = b_t_ptr.add((j + 3) * k + kc);

            for ri in 0..2usize {
                let a_row = a_ptr.add((i + ri) * k + kc);
                let mut ac0_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac0_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac1_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac1_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac2_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac2_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac3_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut ac3_1 = $crate::simd_primitive!($isa, $elem, zero);

                let mut si = 0usize;
                while si < simd_end2 {
                    let va0 = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si));
                    let va1 = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si + LANES));
                    ac0_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si)), ac0_0);
                    ac0_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si + LANES)), ac0_1);
                    ac1_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si)), ac1_0);
                    ac1_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si + LANES)), ac1_1);
                    ac2_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si)), ac2_0);
                    ac2_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si + LANES)), ac2_1);
                    ac3_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si)), ac3_0);
                    ac3_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si + LANES)), ac3_1);
                    si += 2 * LANES;
                }
                while si < simd_end {
                    let va = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si));
                    ac0_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si)), ac0_0);
                    ac1_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si)), ac1_0);
                    ac2_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si)), ac2_0);
                    ac3_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si)), ac3_0);
                    si += LANES;
                }
                let mut d0: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac0_0, ac0_1));
                let mut d1: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac1_0, ac1_1));
                let mut d2: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac2_0, ac2_1));
                let mut d3: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac3_0, ac3_1));
                let mut ti = simd_end;
                while ti < kc_len {
                    let a_val = (*a_row.add(ti)).to_f32();
                    d0 += a_val * (*b0.add(ti)).to_f32();
                    d1 += a_val * (*b1.add(ti)).to_f32();
                    d2 += a_val * (*b2.add(ti)).to_f32();
                    d3 += a_val * (*b3.add(ti)).to_f32();
                    ti += 1;
                }
                let c_row = c_ptr.add((i + ri) * n + j);
                *c_row += <$elem>::from_f32(d0);
                *c_row.add(1) += <$elem>::from_f32(d1);
                *c_row.add(2) += <$elem>::from_f32(d2);
                *c_row.add(3) += <$elem>::from_f32(d3);
            }
        }

        /// bt_dot_m2_j1: 2 rows × 1 column dot product
        /// Optimized: 2 rows simultaneous + 4-way K-unroll (8 acc vectors)
        #[target_feature($(enable = $feat),+)]
        unsafe fn bt_dot_m2_j1(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            i: usize, j: usize, n: usize, k: usize, kc: usize, kc_end: usize,
        ) {
            const LANES: usize = $LANES;
            let kc_len = kc_end - kc;
            let simd_end = kc_len & !(LANES - 1);
            let simd_end4 = kc_len & !(4 * LANES - 1);
            let b_row = b_t_ptr.add(j * k + kc);

            let a0 = a_ptr.add((i) * k + kc);
            let a1 = a_ptr.add((i + 1) * k + kc);

            // 2 rows x 4-way K-unroll = 8 accumulators
            let mut ac0_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac0_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac0_2 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac0_3 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_2 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_3 = $crate::simd_primitive!($isa, $elem, zero);

            let mut si = 0usize;
            while si < simd_end4 {
                let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si));
                let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + LANES));
                let vb2 = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + 2 * LANES));
                let vb3 = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + 3 * LANES));
                ac0_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si)), vb0, ac0_0);
                ac0_1 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si + LANES)), vb1, ac0_1);
                ac0_2 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si + 2 * LANES)), vb2, ac0_2);
                ac0_3 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si + 3 * LANES)), vb3, ac0_3);
                ac1_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si)), vb0, ac1_0);
                ac1_1 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si + LANES)), vb1, ac1_1);
                ac1_2 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si + 2 * LANES)), vb2, ac1_2);
                ac1_3 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si + 3 * LANES)), vb3, ac1_3);
                si += 4 * LANES;
            }
            while si < simd_end {
                let vb = $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si));
                ac0_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a0.add(si)), vb, ac0_0);
                ac1_0 = $crate::simd_primitive!($isa, $elem, fma, $crate::simd_primitive!($isa, $elem, loadu, a1.add(si)), vb, ac1_0);
                si += LANES;
            }
            // Reduce: merge 4 accumulators per row
            let r0_01 = $crate::simd_primitive!($isa, $elem, add, ac0_0, ac0_1);
            let r0_23 = $crate::simd_primitive!($isa, $elem, add, ac0_2, ac0_3);
            let mut d0: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, r0_01, r0_23));
            let r1_01 = $crate::simd_primitive!($isa, $elem, add, ac1_0, ac1_1);
            let r1_23 = $crate::simd_primitive!($isa, $elem, add, ac1_2, ac1_3);
            let mut d1: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, r1_01, r1_23));
            let mut ti = simd_end;
            while ti < kc_len {
                let b_val = (*b_row.add(ti)).to_f32();
                d0 += (*a0.add(ti)).to_f32() * b_val;
                d1 += (*a1.add(ti)).to_f32() * b_val;
                ti += 1;
            }
            *c_ptr.add((i) * n + j) += <$elem>::from_f32(d0);
            *c_ptr.add((i + 1) * n + j) += <$elem>::from_f32(d1);
        }

        /// bt_dot_m1_j4: 1 row × 4 columns dot product
        /// Optimized: 4 cols simultaneous + 2-way K-unroll (8 acc vectors)
        #[target_feature($(enable = $feat),+)]
        unsafe fn bt_dot_m1_j4(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            i: usize, j: usize, n: usize, k: usize, kc: usize, kc_end: usize,
        ) {
            const LANES: usize = $LANES;
            let kc_len = kc_end - kc;
            let simd_end = kc_len & !(LANES - 1);
            let simd_end2 = kc_len & !(2 * LANES - 1);
            let a_row = a_ptr.add(i * k + kc);

            let b0 = b_t_ptr.add(j * k + kc);
            let b1 = b_t_ptr.add((j + 1) * k + kc);
            let b2 = b_t_ptr.add((j + 2) * k + kc);
            let b3 = b_t_ptr.add((j + 3) * k + kc);

            // 4 cols x 2-way K-unroll = 8 accumulators
            let mut ac0_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac0_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac2_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac2_1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac3_0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac3_1 = $crate::simd_primitive!($isa, $elem, zero);

            let mut si = 0usize;
            while si < simd_end2 {
                let va0 = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si));
                let va1 = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si + LANES));
                ac0_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si)), ac0_0);
                ac0_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si + LANES)), ac0_1);
                ac1_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si)), ac1_0);
                ac1_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si + LANES)), ac1_1);
                ac2_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si)), ac2_0);
                ac2_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si + LANES)), ac2_1);
                ac3_0 = $crate::simd_primitive!($isa, $elem, fma, va0, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si)), ac3_0);
                ac3_1 = $crate::simd_primitive!($isa, $elem, fma, va1, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si + LANES)), ac3_1);
                si += 2 * LANES;
            }
            while si < simd_end {
                let va = $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si));
                ac0_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b0.add(si)), ac0_0);
                ac1_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b1.add(si)), ac1_0);
                ac2_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b2.add(si)), ac2_0);
                ac3_0 = $crate::simd_primitive!($isa, $elem, fma, va, $crate::simd_primitive!($isa, $elem, loadu, b3.add(si)), ac3_0);
                si += LANES;
            }
            let mut d0: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac0_0, ac0_1));
            let mut d1: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac1_0, ac1_1));
            let mut d2: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac2_0, ac2_1));
            let mut d3: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, ac3_0, ac3_1));
            let mut ti = simd_end;
            while ti < kc_len {
                let a_val = (*a_row.add(ti)).to_f32();
                d0 += a_val * (*b0.add(ti)).to_f32();
                d1 += a_val * (*b1.add(ti)).to_f32();
                d2 += a_val * (*b2.add(ti)).to_f32();
                d3 += a_val * (*b3.add(ti)).to_f32();
                ti += 1;
            }
            let c_row = c_ptr.add(i * n + j);
            *c_row += <$elem>::from_f32(d0);
            *c_row.add(1) += <$elem>::from_f32(d1);
            *c_row.add(2) += <$elem>::from_f32(d2);
            *c_row.add(3) += <$elem>::from_f32(d3);
        }

        /// bt_dot_m1_j1: 1 row × 1 column dot product
        /// Optimized: 4-way K-unroll (4 acc vectors)
        #[target_feature($(enable = $feat),+)]
        unsafe fn bt_dot_m1_j1(
            a_ptr: *const $elem, b_t_ptr: *const $elem, c_ptr: *mut $elem,
            i: usize, j: usize, n: usize, k: usize, kc: usize, kc_end: usize,
        ) {
            const LANES: usize = $LANES;
            let kc_len = kc_end - kc;
            let simd_end = kc_len & !(LANES - 1);
            let simd_end4 = kc_len & !(4 * LANES - 1);
            let a_row = a_ptr.add(i * k + kc);
            let b_row = b_t_ptr.add(j * k + kc);

            let mut ac0 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac1 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac2 = $crate::simd_primitive!($isa, $elem, zero);
            let mut ac3 = $crate::simd_primitive!($isa, $elem, zero);

            let mut si = 0usize;
            while si < simd_end4 {
                ac0 = $crate::simd_primitive!($isa, $elem, fma,
                    $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si)),
                    $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si)), ac0);
                ac1 = $crate::simd_primitive!($isa, $elem, fma,
                    $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si + LANES)),
                    $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + LANES)), ac1);
                ac2 = $crate::simd_primitive!($isa, $elem, fma,
                    $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si + 2 * LANES)),
                    $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + 2 * LANES)), ac2);
                ac3 = $crate::simd_primitive!($isa, $elem, fma,
                    $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si + 3 * LANES)),
                    $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si + 3 * LANES)), ac3);
                si += 4 * LANES;
            }
            while si < simd_end {
                ac0 = $crate::simd_primitive!($isa, $elem, fma,
                    $crate::simd_primitive!($isa, $elem, loadu, a_row.add(si)),
                    $crate::simd_primitive!($isa, $elem, loadu, b_row.add(si)), ac0);
                si += LANES;
            }
            let r01 = $crate::simd_primitive!($isa, $elem, add, ac0, ac1);
            let r23 = $crate::simd_primitive!($isa, $elem, add, ac2, ac3);
            let mut dot: f32 = $crate::simd_primitive!($isa, $elem, reduce_sum, $crate::simd_primitive!($isa, $elem, add, r01, r23));
            let mut ti = simd_end;
            while ti < kc_len {
                dot += (*a_row.add(ti)).to_f32() * (*b_row.add(ti)).to_f32();
                ti += 1;
            }
            *c_ptr.add(i * n + j) += <$elem>::from_f32(dot);
        }

        // ── 2×LANES compute kernels ──
        // m4×2LANES: 8 acc + 2 B + 1 A = 11 regs, 4x K-unroll
        // m2×2LANES: 4 acc + 2 B + 1 A = 7 regs, 2x K-unroll
        // m1×2LANES: 2 acc + 2 B + 1 A = 5 regs, no unroll
        $crate::define_skinny_mN_2x!($isa, $elem, $LANES, [$($feat),+],
            fn skinny_m4_2x, rows = 4, unroll = 4);
        $crate::define_skinny_mN_2x!($isa, $elem, $LANES, [$($feat),+],
            fn skinny_m2_2x, rows = 2, unroll = 2);
        $crate::define_skinny_mN_2x!($isa, $elem, $LANES, [$($feat),+],
            fn skinny_m1_2x, rows = 1, unroll = 1);

        // ── 1×LANES compute kernels (N remainder) ──
        $crate::define_skinny_mN_1x!($isa, $elem, $LANES, [$($feat),+],
            fn skinny_m4_1x, rows = 4, unroll = 2);
        $crate::define_skinny_mN_1x!($isa, $elem, $LANES, [$($feat),+],
            fn skinny_m2_1x, rows = 2, unroll = 1);
        $crate::define_skinny_mN_1x!($isa, $elem, $LANES, [$($feat),+],
            fn skinny_m1_1x, rows = 1, unroll = 1);
    };
}
