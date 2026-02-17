/// AVX-512 VNNI INT8 GEMM microkernel using vpdpbusd.
///
/// vpdpbusd semantics per 32-bit lane i:
///   acc[i] += sum_{j=0..3}( a_u8[j] * b_i8[j] )
/// where a is unsigned 8-bit, b is signed 8-bit, accumulator is signed 32-bit.
///
/// Microkernel tile: TM rows × TN=NV*LANES output columns (NV=2, LANES=16 → TN=32)
///   - NV=2 accumulator vectors per row (each __m512i has 16 i32 lanes)
///   - 14×2 = 28 accum regs + 2 B loads + 1 A broadcast = 31 zmm regs
///
/// K-loop advances by 4: each vpdpbusd processes 4 consecutive u8/i8 pairs.
///
/// Packed B layout (K-quad interleaved):
///   For each K-quad (k..k+3) and N-strip of TN columns:
///     [b(k,n0)b(k+1,n0)b(k+2,n0)b(k+3,n0) | b(k,n1)... | ... b(k,n15)...]  ← __m512i vec 0
///     [b(k,n16)b(k+1,n16)... b(k+3,n31)]                                      ← __m512i vec 1
///   = 128 bytes per K-quad per N-strip (NV × 64 bytes)
///
/// A broadcast: 4 consecutive u8 from A[m, k..k+3] replicated to all 16 lanes as i32.
#[macro_export]
macro_rules! define_matmul_x86_int8 {
    () => {
        use std::arch::x86_64::*;

        const TM: usize = 14;
        const LANES: usize = 16;  // i32 lanes per __m512i
        const NV: usize = 2;      // accumulator vectors per row
        const TN: usize = NV * LANES;  // 32 output N-columns per strip
        // Packed B stride per K-quad = NV * 64 bytes = 128 bytes
        const BK_STRIDE: usize = NV * 64;  // bytes per K-quad in packed B

        /// Cached blocking parameters for INT8 VNNI backend.
        #[inline(always)]
        fn _blocking() -> $crate::cache_params::BlockingParams {
            static BP: std::sync::OnceLock<$crate::cache_params::BlockingParams> = std::sync::OnceLock::new();
            *BP.get_or_init(|| $crate::cache_params::blocking_params(
                TM, NV, LANES, 1, // elem_bytes = 1 for int8
            ))
        }

        /// Broadcast 4 consecutive u8 from A as a single i32 to all 16 lanes.
        #[inline(always)]
        unsafe fn broadcast_a_quad(ptr: *const u8) -> __m512i {
            let quad = std::ptr::read_unaligned(ptr as *const i32);
            _mm512_set1_epi32(quad)
        }

        // ── pack_b (K-quad interleaved for vpdpbusd) ────────────────────
        // Layout: [chunk][strip][kc_quads][BK_STRIDE]
        //   chunk  = kc_max-sized block along K dimension
        //   strip  = TN-wide block along N dimension
        //   kc_quads = ceil(kc/4) quads per chunk
        //   BK_STRIDE = 128 bytes per K-quad (NV × 64 bytes)
        //
        // Within each K-quad block (128 bytes):
        //   [0..64)   = __m512i for N cols 0..15  (16 × 4 bytes)
        //   [64..128) = __m512i for N cols 16..31 (16 × 4 bytes)
        // Each i32 lane = (b[k,n], b[k+1,n], b[k+2,n], b[k+3,n])
        pub fn pack_b(b: &[i8], n_size: usize, k_size: usize) -> Vec<i8> {
            let kc_max = _blocking().kc;
            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let kc_quads_max = (kc_max + 3) / 4;
            let strip_size = kc_quads_max * BK_STRIDE;
            let cs = n_strips * strip_size;
            let total = n_chunks * cs;
            let mut packed = vec![0i8; total];

            let mut ks = 0usize;
            let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let chunk_base = ch * cs;
                for si in 0..n_strips {
                    let ns = si * TN;
                    let strip_base = chunk_base + si * strip_size;
                    let mut kk = 0usize;
                    let mut kq = 0usize;
                    // Full K-quads
                    while kk + 3 < kc {
                        let quad_base = strip_base + kq * BK_STRIDE;
                        // Vec 0: N columns ns..ns+15
                        for nn in 0..LANES.min(n_size.saturating_sub(ns)) {
                            let dst = quad_base + nn * 4;
                            packed[dst]     = b[(ks + kk)     * n_size + ns + nn];
                            packed[dst + 1] = b[(ks + kk + 1) * n_size + ns + nn];
                            packed[dst + 2] = b[(ks + kk + 2) * n_size + ns + nn];
                            packed[dst + 3] = b[(ks + kk + 3) * n_size + ns + nn];
                        }
                        // Vec 1: N columns ns+16..ns+31
                        for nn in 0..LANES.min(n_size.saturating_sub(ns + LANES)) {
                            let dst = quad_base + 64 + nn * 4;
                            packed[dst]     = b[(ks + kk)     * n_size + ns + LANES + nn];
                            packed[dst + 1] = b[(ks + kk + 1) * n_size + ns + LANES + nn];
                            packed[dst + 2] = b[(ks + kk + 2) * n_size + ns + LANES + nn];
                            packed[dst + 3] = b[(ks + kk + 3) * n_size + ns + LANES + nn];
                        }
                        kk += 4; kq += 1;
                    }
                    // Remainder K (1..3 leftover): pad with zeros (already zeroed)
                    if kk < kc {
                        let quad_base = strip_base + kq * BK_STRIDE;
                        let rem = kc - kk;
                        for nn in 0..LANES.min(n_size.saturating_sub(ns)) {
                            let dst = quad_base + nn * 4;
                            for r in 0..rem {
                                packed[dst + r] = b[(ks + kk + r) * n_size + ns + nn];
                            }
                        }
                        for nn in 0..LANES.min(n_size.saturating_sub(ns + LANES)) {
                            let dst = quad_base + 64 + nn * 4;
                            for r in 0..rem {
                                packed[dst + r] = b[(ks + kk + r) * n_size + ns + LANES + nn];
                            }
                        }
                    }
                }
                ks += kc_max; ch += 1;
            }
            packed
        }

        // ── vpdpbusd microkernel (14×32 tile) ──────────────────────────
        // Processes one kc chunk for one (m_block, n_strip) tile.
        // ac: pointer to A[m, ks] (row-major u8, stride = k_size)
        // bp: pointer to packed B for this strip/chunk (K-quad interleaved i8)
        // cp: pointer to C[m, n] (row-major i32, stride = n_size)
        // kc_quads: number of K-quad iterations (kc / 4, rounded up)
        // k_size: A row stride (in elements)
        // n_size: C row stride (in elements)
        // first_chunk: if true, zero-init accumulators; else load from C
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        unsafe fn microkernel_14x32(
            ac: *const u8, bp: *const i8,
            cp: *mut i32,
            kc_quads: usize, k_size: usize, n_size: usize,
            first_chunk: bool,
        ) {
            macro_rules! init {
                ($r:expr) => {
                    if first_chunk {
                        (_mm512_setzero_si512(), _mm512_setzero_si512())
                    } else {
                        (_mm512_loadu_si512(cp.add($r * n_size) as *const __m512i),
                         _mm512_loadu_si512(cp.add($r * n_size + LANES) as *const __m512i))
                    }
                };
            }
            let (mut c0_0, mut c0_1) = init!(0);
            let (mut c1_0, mut c1_1) = init!(1);
            let (mut c2_0, mut c2_1) = init!(2);
            let (mut c3_0, mut c3_1) = init!(3);
            let (mut c4_0, mut c4_1) = init!(4);
            let (mut c5_0, mut c5_1) = init!(5);
            let (mut c6_0, mut c6_1) = init!(6);
            let (mut c7_0, mut c7_1) = init!(7);
            let (mut c8_0, mut c8_1) = init!(8);
            let (mut c9_0, mut c9_1) = init!(9);
            let (mut c10_0, mut c10_1) = init!(10);
            let (mut c11_0, mut c11_1) = init!(11);
            let (mut c12_0, mut c12_1) = init!(12);
            let (mut c13_0, mut c13_1) = init!(13);

            let mut a_ptr = ac;
            let mut b_ptr = bp;

            for _kq in 0..kc_quads {
                // Load 2 B vectors for this K-quad (each 64 bytes = 16 × i32)
                let vb0 = _mm512_loadu_si512(b_ptr as *const __m512i);           // N cols 0..15
                let vb1 = _mm512_loadu_si512(b_ptr.add(64) as *const __m512i);   // N cols 16..31

                // Prefetch next B quad
                _mm_prefetch(b_ptr.add(BK_STRIDE * 2) as *const i8, _MM_HINT_T0);

                macro_rules! dp_row {
                    ($r:expr, $c0:ident, $c1:ident) => {
                        let va = broadcast_a_quad(a_ptr.add($r * k_size));
                        $c0 = _mm512_dpbusd_epi32($c0, va, vb0);
                        $c1 = _mm512_dpbusd_epi32($c1, va, vb1);
                    };
                }
                dp_row!(0, c0_0, c0_1);
                dp_row!(1, c1_0, c1_1);
                dp_row!(2, c2_0, c2_1);
                dp_row!(3, c3_0, c3_1);
                dp_row!(4, c4_0, c4_1);
                dp_row!(5, c5_0, c5_1);
                dp_row!(6, c6_0, c6_1);
                dp_row!(7, c7_0, c7_1);
                dp_row!(8, c8_0, c8_1);
                dp_row!(9, c9_0, c9_1);
                dp_row!(10, c10_0, c10_1);
                dp_row!(11, c11_0, c11_1);
                dp_row!(12, c12_0, c12_1);
                dp_row!(13, c13_0, c13_1);

                a_ptr = a_ptr.add(4);       // advance 4 K positions in A
                b_ptr = b_ptr.add(BK_STRIDE); // advance to next K-quad in packed B
            }

            // Store i32 accumulators back to C
            macro_rules! store_row {
                ($r:expr, $c0:ident, $c1:ident) => {
                    _mm512_storeu_si512(cp.add($r * n_size) as *mut __m512i, $c0);
                    _mm512_storeu_si512(cp.add($r * n_size + LANES) as *mut __m512i, $c1);
                };
            }
            store_row!(0, c0_0, c0_1);
            store_row!(1, c1_0, c1_1);
            store_row!(2, c2_0, c2_1);
            store_row!(3, c3_0, c3_1);
            store_row!(4, c4_0, c4_1);
            store_row!(5, c5_0, c5_1);
            store_row!(6, c6_0, c6_1);
            store_row!(7, c7_0, c7_1);
            store_row!(8, c8_0, c8_1);
            store_row!(9, c9_0, c9_1);
            store_row!(10, c10_0, c10_1);
            store_row!(11, c11_0, c11_1);
            store_row!(12, c12_0, c12_1);
            store_row!(13, c13_0, c13_1);
        }

        // ── 1×32 edge microkernel (single row) ──────────────────────────
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        unsafe fn microkernel_1x32(
            ac: *const u8, bp: *const i8,
            cp: *mut i32,
            kc_quads: usize, _k_size: usize, _n_size: usize,
            first_chunk: bool,
        ) {
            let (mut c0, mut c1) = if first_chunk {
                (_mm512_setzero_si512(), _mm512_setzero_si512())
            } else {
                (_mm512_loadu_si512(cp as *const __m512i),
                 _mm512_loadu_si512(cp.add(LANES) as *const __m512i))
            };
            let mut a_ptr = ac;
            let mut b_ptr = bp;
            for _ in 0..kc_quads {
                let vb0 = _mm512_loadu_si512(b_ptr as *const __m512i);
                let vb1 = _mm512_loadu_si512(b_ptr.add(64) as *const __m512i);
                let va = broadcast_a_quad(a_ptr);
                c0 = _mm512_dpbusd_epi32(c0, va, vb0);
                c1 = _mm512_dpbusd_epi32(c1, va, vb1);
                a_ptr = a_ptr.add(4);
                b_ptr = b_ptr.add(BK_STRIDE);
            }
            _mm512_storeu_si512(cp as *mut __m512i, c0);
            _mm512_storeu_si512(cp.add(LANES) as *mut __m512i, c1);
        }

        // ── Scalar edge for N-tail (< TN columns) ──────────────────────
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        unsafe fn edge_n_scalar(
            a: &[u8], b: &[i8], c: &mut [i32],
            m_start: usize, m_end: usize, n_start: usize, n_size: usize,
            ks: usize, kc: usize, k_size: usize, first_chunk: bool,
        ) {
            let n_rem = n_size - n_start;
            for mi in m_start..m_end {
                for ni in 0..n_rem {
                    let mut acc: i32 = if first_chunk { 0 } else { c[mi * n_size + n_start + ni] };
                    for ki in 0..kc {
                        let av = a[mi * k_size + ks + ki] as i32;
                        let bv = b[(ks + ki) * n_size + n_start + ni] as i32;
                        acc += av * bv;
                    }
                    c[mi * n_size + n_start + ni] = acc;
                }
            }
        }

        // ── Top-level GEMM: C (i32) = A (u8) × packed_B (i8) ───────────
        // Uses 3-loop (K-chunks → M-blocks → N-strips) with packed B.
        // A is read directly (row-major u8), B must be pre-packed via pack_b().
        // `b_orig` is the original unpacked B (row-major i8) needed for N-tail scalar fallback.
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        pub unsafe fn gemm_int8(
            a: &[u8], b_packed: &[i8], b_orig: &[i8], c: &mut [i32],
            m_size: usize, n_size: usize, k_size: usize,
        ) {
            let bp_params = _blocking();
            let kc_max = bp_params.kc;
            let mc_max = bp_params.mc;
            let n_strips = (n_size + TN - 1) / TN;
            let kc_quads_max = (kc_max + 3) / 4;
            let strip_size = kc_quads_max * BK_STRIDE;
            let cs = n_strips * strip_size;

            let ap = a.as_ptr();
            let cp = c.as_mut_ptr();

            let mut ks = 0usize;
            let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let kc_quads = (kc + 3) / 4;
                let first_chunk = ch == 0;

                let mut m_block = 0usize;
                while m_block < m_size {
                    let m_end = (m_block + mc_max).min(m_size);

                    // Full TM-row tiles
                    let mut m = m_block;
                    while m + TM <= m_end {
                        let mut si = 0usize;
                        let mut n = 0usize;
                        while n + TN <= n_size {
                            let ac = ap.add(m * k_size + ks);
                            let bp_ptr = b_packed.as_ptr().add(ch * cs + si * strip_size);
                            let c_out = cp.add(m * n_size + n);
                            microkernel_14x32(ac, bp_ptr, c_out, kc_quads, k_size, n_size, first_chunk);
                            n += TN; si += 1;
                        }
                        m += TM;
                    }

                    // Remainder M rows (< TM): use 1×32 microkernel
                    while m < m_end {
                        let mut si = 0usize;
                        let mut n = 0usize;
                        while n + TN <= n_size {
                            let ac = ap.add(m * k_size + ks);
                            let bp_ptr = b_packed.as_ptr().add(ch * cs + si * strip_size);
                            let c_out = cp.add(m * n_size + n);
                            microkernel_1x32(ac, bp_ptr, c_out, kc_quads, k_size, n_size, first_chunk);
                            n += TN; si += 1;
                        }
                        m += 1;
                    }

                    m_block = m_end;
                }

                // N-tail (< TN columns): scalar fallback
                let nm = (n_size / TN) * TN;
                if nm < n_size {
                    edge_n_scalar(a, b_orig, c,
                                  0, m_size, nm, n_size, ks, kc, k_size, first_chunk);
                }

                ks += kc_max; ch += 1;
            }
        }

        // ── Dequantize + activation: i32 → f32 with fused epilogue ──────
        // C_f32 = scale_a * scale_b * C_i32 + bias, then activation
        #[target_feature(enable = "avx512f")]
        pub unsafe fn dequantize_i32_to_f32(
            c_i32: &[i32], c_f32: &mut [f32],
            m: usize, n: usize,
            scale_a: f32, scale_b: f32,
            bias: Option<&[f32]>,
            activation: $crate::Activation,
        ) {
            let combined_scale = _mm512_set1_ps(scale_a * scale_b);
            let src = c_i32.as_ptr();
            let dst = c_f32.as_mut_ptr();

            for i in 0..m {
                let mut j = 0usize;
                // Vectorized dequant: 16 i32 → 16 f32 per iteration
                while j + 16 <= n {
                    let idx = i * n + j;
                    let vi = _mm512_loadu_si512(src.add(idx) as *const __m512i);
                    let mut vf = _mm512_cvtepi32_ps(vi);
                    vf = _mm512_mul_ps(vf, combined_scale);
                    if let Some(b) = bias {
                        let vb = _mm512_loadu_ps(b.as_ptr().add(j));
                        vf = _mm512_add_ps(vf, vb);
                    }
                    // Fused activation (fully vectorized)
                    vf = $crate::apply_act_runtime!(avx512, f32, vf, activation);
                    _mm512_storeu_ps(dst.add(idx), vf);
                    j += 16;
                }
                // Scalar tail
                while j < n {
                    let idx = i * n + j;
                    let mut val = c_i32[idx] as f32 * scale_a * scale_b;
                    if let Some(b) = bias { val += b[j]; }
                    val = $crate::apply_act_scalar_runtime!(val, activation);
                    c_f32[idx] = val;
                    j += 1;
                }
            }
        }
    };
}
