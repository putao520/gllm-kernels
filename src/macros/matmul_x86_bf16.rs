/// Native AVX-512 BF16 GEMM microkernel using vdpbf16ps (VDPBF16PS).
///
/// vdpbf16ps semantics per 32-bit lane i:
///   acc[i] += a_bh[i].lo * b_bh[i].lo + a_bh[i].hi * b_bh[i].hi
/// where .lo/.hi are the two bf16 halves of each 32-bit element.
///
/// Microkernel tile: TM=14 rows × TN=32 output columns
///   - NV=2 accumulator vectors per row (each __m512 has 16 f32 lanes)
///   - 14×2 = 28 accum regs + 2 B loads + 1 A broadcast = 31 zmm regs
///
/// K-loop advances by 2: each dpbf16ps processes a (k, k+1) pair.
///
/// Packed B layout (K-interleaved pairs):
///   For each K-pair (k, k+1) and N-strip of 32 columns:
///     [b(k,n0) b(k+1,n0) b(k,n1) b(k+1,n1) ... b(k,n15) b(k+1,n15)]  ← __m512bh vec 0
///     [b(k,n16) b(k+1,n16) ... b(k,n31) b(k+1,n31)]                    ← __m512bh vec 1
///   = 64 bf16 per K-pair per N-strip
///
/// A broadcast: (a[m,k], a[m,k+1]) replicated to all 16 lanes of __m512bh
#[macro_export]
macro_rules! define_matmul_x86_bf16_native {
    () => {
        use std::arch::x86_64::*;

        const TM: usize = 14;
        const LANES: usize = 16;  // f32 lanes per __m512
        const NV: usize = 2;      // accumulator vectors per row
        const TN: usize = NV * LANES;  // 32 output N-columns per strip
        // Packed B stride per K-pair step = NV * 32 bf16 = 2 * 32 = 64 bf16
        // (each __m512bh = 32 bf16, we load NV of them)
        const BK_STRIDE: usize = NV * 32;  // = 64 bf16 per K-pair

        /// Cached blocking parameters for bf16 native backend.
        #[inline(always)]
        fn _blocking() -> $crate::cache_params::BlockingParams {
            static BP: std::sync::OnceLock<$crate::cache_params::BlockingParams> = std::sync::OnceLock::new();
            *BP.get_or_init(|| $crate::cache_params::blocking_params(
                TM, NV, LANES, std::mem::size_of::<half::bf16>(),
            ))
        }

        /// Broadcast (a[k], a[k+1]) bf16 pair to all 16 lanes of __m512bh.
        #[inline(always)]
        unsafe fn broadcast_a_pair(ptr: *const half::bf16) -> __m512bh {
            let pair = std::ptr::read_unaligned(ptr as *const u32);
            let v = _mm512_set1_epi32(pair as i32);
            std::mem::transmute(v)
        }

        /// Load 32 bf16 as __m512bh (for B operand of dpbf16ps).
        #[inline(always)]
        unsafe fn load_b_vec(ptr: *const half::bf16) -> __m512bh {
            let v = _mm512_loadu_si512(ptr as *const __m512i);
            std::mem::transmute(v)
        }

        $crate::define_bf16_helpers!();

        // ── pack_b (K-interleaved pairs for vdpbf16ps) ───────────────
        // Layout: [chunk][strip][kc_pairs][BK_STRIDE]
        //   chunk  = kc_max-sized block along K dimension
        //   strip  = TN-wide block along N dimension
        //   kc_pairs = ceil(kc/2) pairs per chunk
        //   BK_STRIDE = 64 bf16 per K-pair (NV × 32 bf16)
        //
        // Within each BK_STRIDE block:
        //   [0..32)  = __m512bh for N cols 0..15  (16 bf16 pairs)
        //   [32..64) = __m512bh for N cols 16..31 (16 bf16 pairs)
        // Each bf16 pair = (b[k, n_col], b[k+1, n_col])
        pub fn pack_b(b: &[half::bf16], n_size: usize, k_size: usize) -> Vec<half::bf16> {
            let kc_max = _blocking().kc;
            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let kc_pairs_max = (kc_max + 1) / 2;
            let strip_size = kc_pairs_max * BK_STRIDE;
            let cs = n_strips * strip_size;
            let total = n_chunks * cs;
            let mut packed = vec![half::bf16::ZERO; total];
            let mut ks = 0usize;
            let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let chunk_base = ch * cs;
                for si in 0..n_strips {
                    let ns = si * TN;
                    let strip_base = chunk_base + si * strip_size;
                    let mut kk = 0usize;
                    let mut kp = 0usize; // pair index
                    while kk + 1 < kc {
                        let pair_base = strip_base + kp * BK_STRIDE;
                        // Vec 0: N columns ns..ns+15
                        for nn in 0..LANES.min(n_size.saturating_sub(ns)) {
                            let dst = pair_base + nn * 2;
                            packed[dst]     = b[(ks + kk) * n_size + ns + nn];
                            packed[dst + 1] = b[(ks + kk + 1) * n_size + ns + nn];
                        }
                        // Vec 1: N columns ns+16..ns+31
                        for nn in 0..LANES.min(n_size.saturating_sub(ns + LANES)) {
                            let dst = pair_base + 32 + nn * 2;
                            packed[dst]     = b[(ks + kk) * n_size + ns + LANES + nn];
                            packed[dst + 1] = b[(ks + kk + 1) * n_size + ns + LANES + nn];
                        }
                        kk += 2; kp += 1;
                    }
                    // Odd K remainder: pair with zero (already zeroed)
                    if kk < kc {
                        let pair_base = strip_base + kp * BK_STRIDE;
                        for nn in 0..LANES.min(n_size.saturating_sub(ns)) {
                            let dst = pair_base + nn * 2;
                            packed[dst] = b[(ks + kk) * n_size + ns + nn];
                        }
                        for nn in 0..LANES.min(n_size.saturating_sub(ns + LANES)) {
                            let dst = pair_base + 32 + nn * 2;
                            packed[dst] = b[(ks + kk) * n_size + ns + LANES + nn];
                        }
                    }
                }
                ks += kc_max; ch += 1;
            }
            packed
        }

        // ── dpbf16ps microkernel (14×32 tile) ────────────────────────
        // Processes one kc_max chunk for one (m_block, n_strip) tile.
        // ac: pointer to A[m, ks] (row-major, stride = k_size)
        // bp: pointer to packed B for this strip/chunk
        // cp: pointer to C[m, n] (row-major, stride = n_size)
        // kc_pairs: number of K-pair iterations
        // k_size: A row stride
        // n_size: C row stride
        // first_chunk: if true, zero-init accumulators; else load from C
        #[target_feature(enable = "avx512f,avx512bf16,avx512bw")]
        unsafe fn microkernel_14x32(
            ac: *const half::bf16, bp: *const half::bf16,
            cp: *mut half::bf16,
            kc_pairs: usize, k_size: usize, n_size: usize,
            first_chunk: bool,
        ) {
            macro_rules! init {
                ($r:expr) => {
                    if first_chunk {
                        (_mm512_setzero_ps(), _mm512_setzero_ps())
                    } else {
                        (load_bf16_as_f32(cp.add($r * n_size)),
                         load_bf16_as_f32(cp.add($r * n_size + LANES)))
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

            for _kp in 0..kc_pairs {
                // Load 2 B vectors for this K-pair
                let vb0 = load_b_vec(b_ptr);          // N cols 0..15
                let vb1 = load_b_vec(b_ptr.add(32));   // N cols 16..31

                // Prefetch next B pair
                _mm_prefetch(b_ptr.add(BK_STRIDE * 2) as *const i8, _MM_HINT_T0);

                macro_rules! dp_row {
                    ($r:expr, $c0:ident, $c1:ident) => {
                        let va = broadcast_a_pair(a_ptr.add($r * k_size));
                        $c0 = _mm512_dpbf16_ps($c0, va, vb0);
                        $c1 = _mm512_dpbf16_ps($c1, va, vb1);
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

                a_ptr = a_ptr.add(2);  // advance 2 K positions in A
                b_ptr = b_ptr.add(BK_STRIDE);  // advance to next K-pair in packed B
            }

            // Store accumulators back as bf16
            macro_rules! store_row {
                ($r:expr, $c0:ident, $c1:ident) => {
                    store_f32_as_bf16(cp.add($r * n_size), $c0);
                    store_f32_as_bf16(cp.add($r * n_size + LANES), $c1);
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

        // ── 1×32 edge microkernel (single row) ──────────────────────
        #[target_feature(enable = "avx512f,avx512bf16,avx512bw")]
        unsafe fn microkernel_1x32(
            ac: *const half::bf16, bp: *const half::bf16,
            cp: *mut half::bf16,
            kc_pairs: usize, _k_size: usize, _n_size: usize,
            first_chunk: bool,
        ) {
            let (mut c0, mut c1) = if first_chunk {
                (_mm512_setzero_ps(), _mm512_setzero_ps())
            } else {
                (load_bf16_as_f32(cp), load_bf16_as_f32(cp.add(LANES)))
            };
            let mut a_ptr = ac;
            let mut b_ptr = bp;
            for _ in 0..kc_pairs {
                let vb0 = load_b_vec(b_ptr);
                let vb1 = load_b_vec(b_ptr.add(32));
                let va = broadcast_a_pair(a_ptr);
                c0 = _mm512_dpbf16_ps(c0, va, vb0);
                c1 = _mm512_dpbf16_ps(c1, va, vb1);
                a_ptr = a_ptr.add(2);
                b_ptr = b_ptr.add(BK_STRIDE);
            }
            store_f32_as_bf16(cp, c0);
            store_f32_as_bf16(cp.add(LANES), c1);
        }

        // ── Scalar edge for N-tail (< TN columns) ──────────────────
        #[target_feature(enable = "avx512f,avx512bf16,avx512bw")]
        unsafe fn edge_n_scalar(
            a: &[half::bf16], b: &[half::bf16], c: &mut [half::bf16],
            m_start: usize, m_end: usize, n_start: usize, n_size: usize,
            ks: usize, kc: usize, k_size: usize, first_chunk: bool,
        ) {
            let n_rem = n_size - n_start;
            for mi in m_start..m_end {
                for ni in 0..n_rem {
                    let mut acc: f32 = if first_chunk { 0.0 } else { c[mi * n_size + n_start + ni].to_f32() };
                    for ki in 0..kc {
                        let av = a[mi * k_size + ks + ki].to_f32();
                        let bv = b[(ks + ki) * n_size + n_start + ni].to_f32();
                        acc += av * bv;
                    }
                    c[mi * n_size + n_start + ni] = half::bf16::from_f32(acc);
                }
            }
        }

        // ── Small-M no-pack path: bf16→f32 convert + FMA, reads B directly ──
        // Avoids O(K*N) K-interleaved packing overhead when M is small.
        // Uses AVX-512 bf16→f32 widening + vfmadd instead of dpbf16ps (no pair layout needed).
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn nopack_impl(a: &[half::bf16], b: &[half::bf16], c: &mut [half::bf16],
                               m_size: usize, n_size: usize, k_size: usize) {
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = c.as_mut_ptr();

            // Process TM rows at a time
            let mut m = 0usize;
            while m + TM <= m_size {
                let mut n = 0usize;
                while n + TN <= n_size {
                    // 14×32 tile: 28 f32 accumulators
                    let mut acc = [[_mm512_setzero_ps(); 2]; TM];
                    for k in 0..k_size {
                        // Load B[k, n..n+32] as f32
                        let vb0 = load_bf16_as_f32(bp.add(k * n_size + n));
                        let vb1 = load_bf16_as_f32(bp.add(k * n_size + n + LANES));
                        for r in 0..TM {
                            let a_val = (*ap.add((m + r) * k_size + k)).to_f32();
                            let va = _mm512_set1_ps(a_val);
                            acc[r][0] = _mm512_fmadd_ps(va, vb0, acc[r][0]);
                            acc[r][1] = _mm512_fmadd_ps(va, vb1, acc[r][1]);
                        }
                    }
                    // Store as bf16
                    for r in 0..TM {
                        store_f32_as_bf16(cp.add((m + r) * n_size + n), acc[r][0]);
                        store_f32_as_bf16(cp.add((m + r) * n_size + n + LANES), acc[r][1]);
                    }
                    n += TN;
                }
                // N-remainder: 16-wide
                while n + LANES <= n_size {
                    let mut acc = [_mm512_setzero_ps(); TM];
                    for k in 0..k_size {
                        let vb = load_bf16_as_f32(bp.add(k * n_size + n));
                        for r in 0..TM {
                            let a_val = (*ap.add((m + r) * k_size + k)).to_f32();
                            let va = _mm512_set1_ps(a_val);
                            acc[r] = _mm512_fmadd_ps(va, vb, acc[r]);
                        }
                    }
                    for r in 0..TM {
                        store_f32_as_bf16(cp.add((m + r) * n_size + n), acc[r]);
                    }
                    n += LANES;
                }
                // N-remainder: scalar
                while n < n_size {
                    for r in 0..TM {
                        let mut s: f32 = 0.0;
                        for k in 0..k_size {
                            s += a[(m + r) * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                        }
                        c[(m + r) * n_size + n] = half::bf16::from_f32(s);
                    }
                    n += 1;
                }
                m += TM;
            }
            // M-remainder: 1 row at a time
            while m < m_size {
                let mut n = 0usize;
                while n + TN <= n_size {
                    let mut c0 = _mm512_setzero_ps();
                    let mut c1 = _mm512_setzero_ps();
                    for k in 0..k_size {
                        let a_val = (*ap.add(m * k_size + k)).to_f32();
                        let va = _mm512_set1_ps(a_val);
                        let vb0 = load_bf16_as_f32(bp.add(k * n_size + n));
                        let vb1 = load_bf16_as_f32(bp.add(k * n_size + n + LANES));
                        c0 = _mm512_fmadd_ps(va, vb0, c0);
                        c1 = _mm512_fmadd_ps(va, vb1, c1);
                    }
                    store_f32_as_bf16(cp.add(m * n_size + n), c0);
                    store_f32_as_bf16(cp.add(m * n_size + n + LANES), c1);
                    n += TN;
                }
                while n + LANES <= n_size {
                    let mut acc = _mm512_setzero_ps();
                    for k in 0..k_size {
                        let a_val = (*ap.add(m * k_size + k)).to_f32();
                        let va = _mm512_set1_ps(a_val);
                        let vb = load_bf16_as_f32(bp.add(k * n_size + n));
                        acc = _mm512_fmadd_ps(va, vb, acc);
                    }
                    store_f32_as_bf16(cp.add(m * n_size + n), acc);
                    n += LANES;
                }
                while n < n_size {
                    let mut s: f32 = 0.0;
                    for k in 0..k_size {
                        s += a[m * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                    }
                    c[m * n_size + n] = half::bf16::from_f32(s);
                    n += 1;
                }
                m += 1;
            }
        }

        /// Threshold: skip B-packing when M ≤ 4*TM (56 rows).
        const SMALL_M_THRESHOLD: usize = 4 * TM;

        // ── Main matmul entry point ─────────────────────────────────
        pub fn matmul(a: &[half::bf16], b: &[half::bf16], c: &mut [half::bf16],
                      m_size: usize, n_size: usize, k_size: usize) {
            let kc_max = _blocking().kc;
            let mc_max = _blocking().mc;
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);

            if m_size == 0 || n_size == 0 || k_size == 0 { return; }

            // Small-M fast path: skip B-packing entirely
            if m_size <= SMALL_M_THRESHOLD {
                unsafe { nopack_impl(a, b, c, m_size, n_size, k_size); }
                return;
            }

            // Pack B into thread-local workspace
            let n_strips = (n_size + TN - 1) / TN;
            let n_full_strips = n_size / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let kc_pairs_max = (kc_max + 1) / 2;
            let strip_size = kc_pairs_max * BK_STRIDE;
            let cs = n_strips * strip_size;
            let total = n_chunks * cs;

            thread_local! { static WS: std::cell::RefCell<Vec<half::bf16>> = std::cell::RefCell::new(Vec::new()); }
            WS.with(|cell| {
                let mut pb = cell.borrow_mut();
                if pb.len() < total {
                    pb.resize(total, half::bf16::ZERO);
                }
                // Zero for clean pair padding
                pb[..total].fill(half::bf16::ZERO);

                // Pack B
                let mut ks = 0usize;
                let mut ch = 0usize;
                while ks < k_size {
                    let kc = kc_max.min(k_size - ks);
                    let chunk_base = ch * cs;
                    for si in 0..n_strips {
                        let ns = si * TN;
                        let strip_base = chunk_base + si * strip_size;
                        let mut kk = 0usize;
                        let mut kp = 0usize;
                        while kk + 1 < kc {
                            let pair_base = strip_base + kp * BK_STRIDE;
                            for nn in 0..LANES.min(n_size.saturating_sub(ns)) {
                                pb[pair_base + nn * 2]     = b[(ks + kk) * n_size + ns + nn];
                                pb[pair_base + nn * 2 + 1] = b[(ks + kk + 1) * n_size + ns + nn];
                            }
                            if ns + LANES < n_size {
                                for nn in 0..LANES.min(n_size - ns - LANES) {
                                    pb[pair_base + 32 + nn * 2]     = b[(ks + kk) * n_size + ns + LANES + nn];
                                    pb[pair_base + 32 + nn * 2 + 1] = b[(ks + kk + 1) * n_size + ns + LANES + nn];
                                }
                            }
                            kk += 2; kp += 1;
                        }
                        if kk < kc {
                            let pair_base = strip_base + kp * BK_STRIDE;
                            for nn in 0..LANES.min(n_size.saturating_sub(ns)) {
                                pb[pair_base + nn * 2] = b[(ks + kk) * n_size + ns + nn];
                            }
                            if ns + LANES < n_size {
                                for nn in 0..LANES.min(n_size - ns - LANES) {
                                    pb[pair_base + 32 + nn * 2] = b[(ks + kk) * n_size + ns + LANES + nn];
                                }
                            }
                        }
                    }
                    ks += kc_max; ch += 1;
                }

                // Compute C = A * packed_B
                let mut ks = 0usize;
                let mut ch = 0usize;
                while ks < k_size {
                    let kc = kc_max.min(k_size - ks);
                    let kc_pairs = (kc + 1) / 2;
                    let first = ch == 0;
                    let mut m_block = 0usize;
                    while m_block < m_size {
                        let m_end = (m_block + mc_max).min(m_size);
                        // Full TM rows
                        let mut m = m_block;
                        while m + TM <= m_end {
                            // Full N strips
                            for si in 0..n_full_strips {
                                let n = si * TN;
                                unsafe {
                                    microkernel_14x32(
                                        a.as_ptr().add(m * k_size + ks),
                                        pb.as_ptr().add(ch * cs + si * strip_size),
                                        c.as_mut_ptr().add(m * n_size + n),
                                        kc_pairs, k_size, n_size, first,
                                    );
                                }
                            }
                            // N-tail (scalar fallback)
                            if n_full_strips < n_strips {
                                unsafe {
                                    edge_n_scalar(a, b, c, m, m + TM, n_full_strips * TN, n_size, ks, kc, k_size, first);
                                }
                            }
                            m += TM;
                        }
                        // M-tail: process remaining rows one at a time
                        while m < m_end {
                            for si in 0..n_full_strips {
                                let n = si * TN;
                                unsafe {
                                    microkernel_1x32(
                                        a.as_ptr().add(m * k_size + ks),
                                        pb.as_ptr().add(ch * cs + si * strip_size),
                                        c.as_mut_ptr().add(m * n_size + n),
                                        kc_pairs, k_size, n_size, first,
                                    );
                                }
                            }
                            if n_full_strips < n_strips {
                                unsafe {
                                    edge_n_scalar(a, b, c, m, m + 1, n_full_strips * TN, n_size, ks, kc, k_size, first);
                                }
                            }
                            m += 1;
                        }
                        m_block += mc_max;
                    }
                    ks += kc_max; ch += 1;
                }
            });
        }

        // ── matmul_prepacked: C = A * packed_B ──────────────────────
        // packed_B must have been produced by pack_b() above (K-interleaved pair layout).
        pub fn matmul_prepacked(a: &[half::bf16], packed_b: &[half::bf16], c: &mut [half::bf16],
                                m_size: usize, n_size: usize, k_size: usize) {
            let kc_max = _blocking().kc;
            let mc_max = _blocking().mc;
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            if m_size == 0 || n_size == 0 || k_size == 0 { return; }

            let n_strips = (n_size + TN - 1) / TN;
            let n_full_strips = n_size / TN;
            let kc_pairs_max = (kc_max + 1) / 2;
            let strip_size = kc_pairs_max * BK_STRIDE;
            let cs = n_strips * strip_size;
            let cp = c.as_mut_ptr();

            let mut ks = 0usize; let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let kc_pairs = (kc + 1) / 2;
                let first = ch == 0;
                let mut m_block = 0usize;
                while m_block < m_size {
                    let m_end = (m_block + mc_max).min(m_size);
                    let mut m = m_block;
                    while m + TM <= m_end {
                        for si in 0..n_full_strips {
                            let n = si * TN;
                            unsafe {
                                microkernel_14x32(
                                    a.as_ptr().add(m * k_size + ks),
                                    packed_b.as_ptr().add(ch * cs + si * strip_size),
                                    cp.add(m * n_size + n),
                                    kc_pairs, k_size, n_size, first,
                                );
                            }
                        }
                        // N-tail: scalar fallback using unpacked B
                        // Note: for prepacked, we don't have original B, so we decode from packed layout
                        if n_full_strips < n_strips {
                            let si = n_full_strips;
                            let n = si * TN;
                            let n_rem = n_size - n;
                            if n_rem >= TN {
                                // Full strip, use microkernel
                                unsafe {
                                    microkernel_14x32(
                                        a.as_ptr().add(m * k_size + ks),
                                        packed_b.as_ptr().add(ch * cs + si * strip_size),
                                        cp.add(m * n_size + n),
                                        kc_pairs, k_size, n_size, first,
                                    );
                                }
                            } else {
                                // Partial strip: use scalar with packed B decode
                                let pb_base = packed_b.as_ptr();
                                for mi in m..m+TM {
                                    for ni in 0..n_rem {
                                        let mut acc: f32 = if first { 0.0 } else { c[mi * n_size + n + ni].to_f32() };
                                        let vec_idx = ni / LANES;
                                        let lane = ni % LANES;
                                        let mut kk = 0usize;
                                        let mut kp = 0usize;
                                        while kk + 1 < kc {
                                            unsafe {
                                                let bp_off = ch * cs + si * strip_size + kp * BK_STRIDE + vec_idx * 32 + lane * 2;
                                                let bk0 = (*pb_base.add(bp_off)).to_f32();
                                                let bk1 = (*pb_base.add(bp_off + 1)).to_f32();
                                                let ak0 = a[mi * k_size + ks + kk].to_f32();
                                                let ak1 = a[mi * k_size + ks + kk + 1].to_f32();
                                                acc += ak0 * bk0 + ak1 * bk1;
                                            }
                                            kk += 2; kp += 1;
                                        }
                                        if kk < kc {
                                            unsafe {
                                                let bp_off = ch * cs + si * strip_size + kp * BK_STRIDE + vec_idx * 32 + lane * 2;
                                                let bk0 = (*pb_base.add(bp_off)).to_f32();
                                                let ak0 = a[mi * k_size + ks + kk].to_f32();
                                                acc += ak0 * bk0;
                                            }
                                        }
                                        c[mi * n_size + n + ni] = half::bf16::from_f32(acc);
                                    }
                                }
                            }
                        }
                        m += TM;
                    }
                    // M-tail
                    while m < m_end {
                        for si in 0..n_full_strips {
                            let n = si * TN;
                            unsafe {
                                microkernel_1x32(
                                    a.as_ptr().add(m * k_size + ks),
                                    packed_b.as_ptr().add(ch * cs + si * strip_size),
                                    cp.add(m * n_size + n),
                                    kc_pairs, k_size, n_size, first,
                                );
                            }
                        }
                        if n_full_strips < n_strips {
                            let si = n_full_strips;
                            let n = si * TN;
                            let n_rem = n_size - n;
                            let pb_base = packed_b.as_ptr();
                            for ni in 0..n_rem {
                                let mut acc: f32 = if first { 0.0 } else { c[m * n_size + n + ni].to_f32() };
                                let vec_idx = ni / LANES;
                                let lane = ni % LANES;
                                let mut kk = 0usize; let mut kp = 0usize;
                                while kk + 1 < kc {
                                    unsafe {
                                        let bp_off = ch * cs + si * strip_size + kp * BK_STRIDE + vec_idx * 32 + lane * 2;
                                        let bk0 = (*pb_base.add(bp_off)).to_f32();
                                        let bk1 = (*pb_base.add(bp_off + 1)).to_f32();
                                        acc += a[m * k_size + ks + kk].to_f32() * bk0 + a[m * k_size + ks + kk + 1].to_f32() * bk1;
                                    }
                                    kk += 2; kp += 1;
                                }
                                if kk < kc {
                                    unsafe {
                                        let bp_off = ch * cs + si * strip_size + kp * BK_STRIDE + vec_idx * 32 + lane * 2;
                                        acc += a[m * k_size + ks + kk].to_f32() * (*pb_base.add(bp_off)).to_f32();
                                    }
                                }
                                c[m * n_size + n + ni] = half::bf16::from_f32(acc);
                            }
                        }
                        m += 1;
                    }
                    m_block += mc_max;
                }
                ks += kc_max; ch += 1;
            }
        }

        // ── matmul_bias: C = A * B + bias ───────────────────────────
        pub fn matmul_bias(a: &[half::bf16], b: &[half::bf16], bias: &[half::bf16],
                           c: &mut [half::bf16], m_size: usize, n_size: usize, k_size: usize) {
            // First compute C = A * B
            matmul(a, b, c, m_size, n_size, k_size);
            // Then add bias to each row
            for m in 0..m_size {
                for n in 0..n_size {
                    let val = c[m * n_size + n].to_f32() + bias[n].to_f32();
                    c[m * n_size + n] = half::bf16::from_f32(val);
                }
            }
        }

        // ── matmul_bias_prepacked: C = A * packed_B + bias ──────────
        pub fn matmul_bias_prepacked(a: &[half::bf16], packed_b: &[half::bf16], bias: &[half::bf16],
                                     c: &mut [half::bf16], m_size: usize, n_size: usize, k_size: usize) {
            // First compute C = A * packed_B
            matmul_prepacked(a, packed_b, c, m_size, n_size, k_size);
            // Then add bias to each row
            for m in 0..m_size {
                for n in 0..n_size {
                    let val = c[m * n_size + n].to_f32() + bias[n].to_f32();
                    c[m * n_size + n] = half::bf16::from_f32(val);
                }
            }
        }

        pub fn matmul_bias_act(a: &[half::bf16], b: &[half::bf16], bias: &[half::bf16],
                               c: &mut [half::bf16], m_size: usize, n_size: usize, k_size: usize,
                               act: $crate::Activation) {
            matmul_bias(a, b, bias, c, m_size, n_size, k_size);
            let len = m_size * n_size;
            match act {
                $crate::Activation::Relu => {
                    for i in 0..len { if c[i].to_f32() < 0.0 { c[i] = half::bf16::from_f32(0.0); } }
                }
                $crate::Activation::Silu => {
                    for i in 0..len { let v = c[i].to_f32(); c[i] = half::bf16::from_f32(v / (1.0 + (-v).exp())); }
                }
                $crate::Activation::Gelu => {
                    for i in 0..len {
                        let x = c[i].to_f32();
                        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
                        c[i] = half::bf16::from_f32(0.5 * x * (1.0 + inner.tanh()));
                    }
                }
                _ => {}
            }
        }
    }; // end of macro arm
} // end of macro_rules!
