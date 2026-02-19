/// Streaming GEMV macro — dedicated M=1 path for LLM decode.
///
/// Computes c[1×N] = a[1×K] * B[K×N] with zero packing overhead.
/// This is the hottest path in autoregressive decode (M=1 per token).
///
/// Optimizations vs generic nopack matmul:
/// - Zero packing: reads B directly from row-major layout
/// - Wide N-processing: uses all available SIMD registers as accumulators
///   (AVX2: 6×2=12 accumulators, AVX-512: 6×2=12 accumulators for 6*TN columns)
/// - K-tiling: KC=512 keeps A vector resident in L1
/// - 4x K-unroll with software prefetch 8 cache lines ahead
/// - N-parallel via rayon for large N
/// - Fused epilogue: bias + activation applied before store
///
/// Memory bandwidth bound: theoretical peak = DRAM_BW / (K*N*elem_bytes + K*elem_bytes)
/// Target: ≥90% of practical DRAM bandwidth (~38-40 GB/s DDR4-3200 dual-channel)
#[macro_export]
macro_rules! define_gemv_streaming {
    ($isa:ident, $elem:ident, $LANES:literal, $($feat:literal),+) => {

        /// Streaming GEMV entry point: c[1×N] = a[1×K] * B[K×N]
        /// Called from matmul dispatch when M=1.
        #[inline(always)]
        pub fn gemv_streaming(a: &[$elem], b: &[$elem], c: &mut [$elem],
                              n_size: usize, k_size: usize) {
            assert!(a.len() >= k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= n_size);

            // N-parallel dispatch: split N across threads when large enough
            let nthreads = rayon::current_num_threads().max(1);
            const TN: usize = $LANES * 2; // Process 2 SIMD vectors per N-chunk
            if n_size >= TN * 8 && nthreads > 1 {
                let n_per_thread = ((n_size + nthreads - 1) / nthreads + TN - 1) / TN * TN;
                let num_blocks = (n_size + n_per_thread - 1) / n_per_thread;
                let ap = a.as_ptr() as usize;
                let bp = b.as_ptr() as usize;
                let cp = c.as_mut_ptr() as usize;

                use rayon::prelude::*;
                (0..num_blocks).into_par_iter().for_each(|ni| {
                    let n_start = ni * n_per_thread;
                    let n_end = (n_start + n_per_thread).min(n_size);
                    unsafe {
                        gemv_streaming_range(
                            ap as *const $elem, bp as *const $elem, cp as *mut $elem,
                            n_size, k_size, n_start, n_end,
                        );
                    }
                });
            } else {
                unsafe {
                    gemv_streaming_range(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        n_size, k_size, 0, n_size,
                    );
                }
            }
        }

        /// Inner streaming GEMV kernel for N range [n_start, n_end).
        /// Uses 6 × 2-vector accumulator groups (12 SIMD regs) to process
        /// 6*TN output columns simultaneously, hiding FMA latency.
        ///
        /// Prefetch distance and hint type are adaptive via `kernel_config()`:
        /// - Distance: `pf_rows_gemv` (derived from memory latency / frequency)
        /// - Hint: NTA (non-temporal) for streaming GEMV to avoid L2/L3 pollution
        #[target_feature($(enable = $feat),+)]
        unsafe fn gemv_streaming_range(
            a_ptr: *const $elem, b_ptr: *const $elem, c_ptr: *mut $elem,
            n_size: usize, k_size: usize,
            n_start: usize, n_end: usize,
        ) {
            const LANES: usize = $LANES;
            const TN: usize = LANES * 2;  // 2 vectors per accumulator group
            const KC: usize = 512;        // K-tile: 512 elements ≈ 2KB f32, fits L1

            // Adaptive prefetch distance from kernel_config
            let pf_rows = $crate::microarch::kernel_config().pf_rows_gemv;

            // ── Process 3×TN columns at a time (6 accumulator vectors) ──
            let mut n = n_start;
            while n + TN * 3 <= n_end {
                let mut acc0_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc0_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc1_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc1_1 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc2_0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc2_1 = $crate::simd_primitive!($isa, $elem, zero);

                let mut ks = 0usize;
                while ks < k_size {
                    let kc = KC.min(k_size - ks);
                    let ku = kc & !3; // 4x unroll
                    let mut ki = 0usize;

                    while ki < ku {
                        let k = ks + ki;
                        // NTA prefetch: streaming access, don't pollute L2/L3
                        $crate::simd_primitive!($isa, $elem, prefetch_nta,
                            b_ptr.add((k + pf_rows) * n_size + n));
                        $crate::simd_primitive!($isa, $elem, prefetch_nta,
                            b_ptr.add((k + pf_rows) * n_size + n + TN));
                        $crate::simd_primitive!($isa, $elem, prefetch_nta,
                            b_ptr.add((k + pf_rows) * n_size + n + TN * 2));

                        // k+0
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + LANES));
                        let vb2 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN));
                        let vb3 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN + LANES));
                        let vb4 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN * 2));
                        let vb5 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN * 2 + LANES));
                        acc0_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, acc0_0);
                        acc0_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, acc0_1);
                        acc1_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb2, acc1_0);
                        acc1_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb3, acc1_1);
                        acc2_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb4, acc2_0);
                        acc2_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb5, acc2_1);

                        // k+1
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k + 1));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n + LANES));
                        let vb2 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n + TN));
                        let vb3 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n + TN + LANES));
                        let vb4 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n + TN * 2));
                        let vb5 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n + TN * 2 + LANES));
                        acc0_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, acc0_0);
                        acc0_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, acc0_1);
                        acc1_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb2, acc1_0);
                        acc1_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb3, acc1_1);
                        acc2_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb4, acc2_0);
                        acc2_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb5, acc2_1);

                        // k+2
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k + 2));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n + LANES));
                        let vb2 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n + TN));
                        let vb3 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n + TN + LANES));
                        let vb4 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n + TN * 2));
                        let vb5 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n + TN * 2 + LANES));
                        acc0_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, acc0_0);
                        acc0_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, acc0_1);
                        acc1_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb2, acc1_0);
                        acc1_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb3, acc1_1);
                        acc2_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb4, acc2_0);
                        acc2_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb5, acc2_1);

                        // k+3
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k + 3));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n + LANES));
                        let vb2 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n + TN));
                        let vb3 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n + TN + LANES));
                        let vb4 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n + TN * 2));
                        let vb5 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n + TN * 2 + LANES));
                        acc0_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, acc0_0);
                        acc0_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, acc0_1);
                        acc1_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb2, acc1_0);
                        acc1_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb3, acc1_1);
                        acc2_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb4, acc2_0);
                        acc2_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb5, acc2_1);

                        ki += 4;
                    }
                    // K remainder
                    while ki < kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k));
                        let vb0 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n));
                        let vb1 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + LANES));
                        let vb2 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN));
                        let vb3 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN + LANES));
                        let vb4 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN * 2));
                        let vb5 = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + TN * 2 + LANES));
                        acc0_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb0, acc0_0);
                        acc0_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb1, acc0_1);
                        acc1_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb2, acc1_0);
                        acc1_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb3, acc1_1);
                        acc2_0 = $crate::simd_primitive!($isa, $elem, fma, va, vb4, acc2_0);
                        acc2_1 = $crate::simd_primitive!($isa, $elem, fma, va, vb5, acc2_1);
                        ki += 1;
                    }
                    ks += KC;
                }
                // Store
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n), acc0_0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n + LANES), acc0_1);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n + TN), acc1_0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n + TN + LANES), acc1_1);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n + TN * 2), acc2_0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n + TN * 2 + LANES), acc2_1);
                n += TN * 3;
            }

            // ── 1×TN remainder ──
            while n + TN <= n_end {
                let mut acc0 = $crate::simd_primitive!($isa, $elem, zero);
                let mut acc1 = $crate::simd_primitive!($isa, $elem, zero);

                let mut ks = 0usize;
                while ks < k_size {
                    let kc = KC.min(k_size - ks);
                    let ku = kc & !3;
                    let mut ki = 0usize;
                    while ki < ku {
                        let k = ks + ki;
                        $crate::simd_primitive!($isa, $elem, prefetch_nta,
                            b_ptr.add((k + pf_rows) * n_size + n));
                        // k+0
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n)), acc0);
                        acc1 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + LANES)), acc1);
                        // k+1
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k + 1));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n)), acc0);
                        acc1 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+1) * n_size + n + LANES)), acc1);
                        // k+2
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k + 2));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n)), acc0);
                        acc1 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+2) * n_size + n + LANES)), acc1);
                        // k+3
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k + 3));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n)), acc0);
                        acc1 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add((k+3) * n_size + n + LANES)), acc1);
                        ki += 4;
                    }
                    while ki < kc {
                        let k = ks + ki;
                        let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n)), acc0);
                        acc1 = $crate::simd_primitive!($isa, $elem, fma, va,
                            $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n + LANES)), acc1);
                        ki += 1;
                    }
                    ks += KC;
                }
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n), acc0);
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n + LANES), acc1);
                n += TN;
            }

            // ── LANES-wide remainder ──
            while n + LANES <= n_end {
                let mut acc = $crate::simd_primitive!($isa, $elem, zero);
                for k in 0..k_size {
                    let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k));
                    let vb = $crate::simd_primitive!($isa, $elem, loadu, b_ptr.add(k * n_size + n));
                    acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                }
                $crate::simd_primitive!($isa, $elem, storeu, c_ptr.add(n), acc);
                n += LANES;
            }

            // ── Masked SIMD tail ──
            let rem = n_end - n;
            if rem > 0 {
                let mut acc = $crate::simd_primitive!($isa, $elem, zero);
                for k in 0..k_size {
                    let va = $crate::simd_primitive!($isa, $elem, splat, *a_ptr.add(k));
                    let vb = $crate::simd_primitive!($isa, $elem, maskload, b_ptr.add(k * n_size + n), rem);
                    acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                }
                $crate::simd_primitive!($isa, $elem, maskstore, c_ptr.add(n), acc, rem);
            }
        }

        /// Streaming GEMV with bias: c[1×N] = a[1×K] * B[K×N] + bias[N]
        #[inline(always)]
        pub fn gemv_streaming_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                                   n_size: usize, k_size: usize) {
            // Compute GEMV first, then add bias in-place
            gemv_streaming(a, b, c, n_size, k_size);
            // Fused bias add
            const LANES: usize = $LANES;
            let mut i = 0usize;
            #[allow(unused_unsafe)]
            unsafe {
                while i + LANES * 4 <= n_size {
                    let c0 = $crate::simd_primitive!($isa, $elem, loadu, c.as_ptr().add(i));
                    let c1 = $crate::simd_primitive!($isa, $elem, loadu, c.as_ptr().add(i + LANES));
                    let c2 = $crate::simd_primitive!($isa, $elem, loadu, c.as_ptr().add(i + LANES * 2));
                    let c3 = $crate::simd_primitive!($isa, $elem, loadu, c.as_ptr().add(i + LANES * 3));
                    let b0 = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(i));
                    let b1 = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(i + LANES));
                    let b2 = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(i + LANES * 2));
                    let b3 = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(i + LANES * 3));
                    $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(i),
                        $crate::simd_primitive!($isa, $elem, add, c0, b0));
                    $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(i + LANES),
                        $crate::simd_primitive!($isa, $elem, add, c1, b1));
                    $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(i + LANES * 2),
                        $crate::simd_primitive!($isa, $elem, add, c2, b2));
                    $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(i + LANES * 3),
                        $crate::simd_primitive!($isa, $elem, add, c3, b3));
                    i += LANES * 4;
                }
                while i + LANES <= n_size {
                    let cv = $crate::simd_primitive!($isa, $elem, loadu, c.as_ptr().add(i));
                    let bv = $crate::simd_primitive!($isa, $elem, loadu, bias.as_ptr().add(i));
                    $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(i),
                        $crate::simd_primitive!($isa, $elem, add, cv, bv));
                    i += LANES;
                }
            }
            while i < n_size {
                c[i] = c[i] + bias[i];
                i += 1;
            }
        }

        /// Streaming GEMV with bias + fused activation: c = act(a*B + bias)
        #[inline(always)]
        pub fn gemv_streaming_bias_act(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem],
                                       n_size: usize, k_size: usize, act: $crate::Activation) {
            // Compute GEMV + bias
            gemv_streaming_bias(a, b, bias, c, n_size, k_size);
            // Apply activation in-place (fused — single pass over C)
            if matches!(act, $crate::Activation::None) { return; }
            const LANES: usize = $LANES;
            let mut i = 0usize;
            #[allow(unused_unsafe)]
            unsafe {
                while i + LANES <= n_size {
                    let v = $crate::simd_primitive!($isa, $elem, loadu, c.as_ptr().add(i));
                    let r = $crate::apply_act_runtime!($isa, $elem, v, act);
                    $crate::simd_primitive!($isa, $elem, storeu, c.as_mut_ptr().add(i), r);
                    i += LANES;
                }
            }
            while i < n_size {
                let v = c[i].to_f32();
                c[i] = <$elem as Element>::from_f32($crate::apply_act_scalar_runtime!(v, act));
                i += 1;
            }
        }
    };
}
