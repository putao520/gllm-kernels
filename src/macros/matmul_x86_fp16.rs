/// Native AVX-512 FP16 GEMM microkernel using vfmaddph (VFMADD231PH).
///
/// vfmaddph semantics per 16-bit lane i:
///   acc[i] = a[i] * b[i] + acc[i]
/// where all operands are f16 (IEEE 754 half-precision).
///
/// Microkernel tile: TM=14 rows x TN=64 output columns
///   - NV=2 accumulator vectors per row (each __m512h has 32 f16 lanes)
///   - 14x2 = 28 accum regs + 2 B loads + 1 A broadcast = 31 zmm regs
///
/// K-loop advances by 1: each fmadd processes one K element.
///
/// Packed B layout (standard column-panel):
///   For each K step and N-strip of 64 columns:
///     [b(k,n0) b(k,n1) ... b(k,n31)]  <- __m512h vec 0
///     [b(k,n32) ... b(k,n63)]          <- __m512h vec 1
///   = 64 f16 per K step per N-strip
///
/// A broadcast: a[m,k] replicated to all 32 lanes of __m512h
#[macro_export]
macro_rules! define_matmul_x86_fp16_native {
    () => {
        use std::arch::x86_64::*;

        const TM: usize = 14;
        const LANES: usize = 32;  // f16 lanes per __m512h
        const NV: usize = 2;      // accumulator vectors per row
        const TN: usize = NV * LANES;  // 64 output N-columns per strip

        /// Cached blocking parameters for fp16 native backend.
        #[inline(always)]
        fn _blocking() -> $crate::cache_params::BlockingParams {
            static BP: std::sync::OnceLock<$crate::cache_params::BlockingParams> = std::sync::OnceLock::new();
            *BP.get_or_init(|| $crate::cache_params::blocking_params(
                TM, NV, LANES, std::mem::size_of::<half::f16>(),
            ))
        }

        /// Broadcast a[m,k] f16 scalar to all 32 lanes of __m512h.
        #[inline(always)]
        unsafe fn broadcast_a(ptr: *const half::f16) -> __m512h {
            let val = std::ptr::read_unaligned(ptr as *const u16);
            let vi = _mm512_set1_epi16(val as i16);
            std::mem::transmute(vi)
        }

        /// Load 32 f16 as __m512h.
        #[inline(always)]
        unsafe fn load_b_vec(ptr: *const half::f16) -> __m512h {
            _mm512_loadu_ph(ptr as *const _)
        }

        // ── pack_b (standard column-panel for native f16) ────────────────
        pub fn pack_b(b: &[half::f16], n_size: usize, k_size: usize) -> Vec<half::f16> {
            let kc_max = _blocking().kc;
            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TN - 1) / TN;
            let n_chunks = (k_size + kc_max - 1) / kc_max;
            let strip_size = kc_max * TN;
            let cs = n_strips * strip_size;
            let total = n_chunks * cs;
            let mut packed = vec![half::f16::ZERO; total];

            let mut ks = 0usize;
            let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let base = ch * cs;
                for i in 0..n_strips {
                    let ns = i * TN;
                    let an = TN.min(n_size.saturating_sub(ns));
                    for k in 0..kc {
                        let dst_off = base + i * strip_size + k * TN;
                        for n in 0..an {
                            packed[dst_off + n] = b[(ks + k) * n_size + ns + n];
                        }
                        // Zero-pad remainder
                        for n in an..TN {
                            packed[dst_off + n] = half::f16::ZERO;
                        }
                    }
                }
                ks += kc;
                ch += 1;
            }
            packed
        }

        // ── microkernel: TM x TN tile ────────────────────────────────────
        #[target_feature(enable = "avx512fp16")]
        unsafe fn microkernel(
            a_panel: *const half::f16, a_stride: usize,
            b_panel: *const half::f16,
            c_ptr: *mut half::f16, c_stride: usize,
            mr: usize, kc: usize,
        ) {
            // Initialize TM x NV accumulators
            let mut acc: [[__m512h; NV]; TM] = [[_mm512_setzero_ph(); NV]; TM];

            for k in 0..kc {
                let b_base = b_panel.add(k * TN);
                let bv0 = load_b_vec(b_base);
                let bv1 = load_b_vec(b_base.add(LANES));

                macro_rules! row_fma {
                    ($r:expr) => {
                        if $r < mr {
                            let av = broadcast_a(a_panel.add($r * a_stride + k));
                            acc[$r][0] = _mm512_fmadd_ph(av, bv0, acc[$r][0]);
                            acc[$r][1] = _mm512_fmadd_ph(av, bv1, acc[$r][1]);
                        }
                    };
                }
                row_fma!(0);  row_fma!(1);  row_fma!(2);  row_fma!(3);
                row_fma!(4);  row_fma!(5);  row_fma!(6);  row_fma!(7);
                row_fma!(8);  row_fma!(9);  row_fma!(10); row_fma!(11);
                row_fma!(12); row_fma!(13);
            }

            // Store accumulators to C
            for r in 0..mr {
                let c_row = c_ptr.add(r * c_stride);
                _mm512_storeu_ph(c_row as *mut _, acc[r][0]);
                _mm512_storeu_ph(c_row.add(LANES) as *mut _, acc[r][1]);
            }
        }

        // ── microkernel with bias epilogue ────────────────────────────────
        #[target_feature(enable = "avx512fp16")]
        unsafe fn microkernel_bias(
            a_panel: *const half::f16, a_stride: usize,
            b_panel: *const half::f16,
            bias_ptr: *const half::f16,
            c_ptr: *mut half::f16, c_stride: usize,
            mr: usize, kc: usize,
        ) {
            let mut acc: [[__m512h; NV]; TM] = [[_mm512_setzero_ph(); NV]; TM];

            for k in 0..kc {
                let b_base = b_panel.add(k * TN);
                let bv0 = load_b_vec(b_base);
                let bv1 = load_b_vec(b_base.add(LANES));

                macro_rules! row_fma {
                    ($r:expr) => {
                        if $r < mr {
                            let av = broadcast_a(a_panel.add($r * a_stride + k));
                            acc[$r][0] = _mm512_fmadd_ph(av, bv0, acc[$r][0]);
                            acc[$r][1] = _mm512_fmadd_ph(av, bv1, acc[$r][1]);
                        }
                    };
                }
                row_fma!(0);  row_fma!(1);  row_fma!(2);  row_fma!(3);
                row_fma!(4);  row_fma!(5);  row_fma!(6);  row_fma!(7);
                row_fma!(8);  row_fma!(9);  row_fma!(10); row_fma!(11);
                row_fma!(12); row_fma!(13);
            }

            // Add bias and store
            let bv0 = load_b_vec(bias_ptr);
            let bv1 = load_b_vec(bias_ptr.add(LANES));
            for r in 0..mr {
                let c_row = c_ptr.add(r * c_stride);
                let r0 = _mm512_add_ph(acc[r][0], bv0);
                let r1 = _mm512_add_ph(acc[r][1], bv1);
                _mm512_storeu_ph(c_row as *mut _, r0);
                _mm512_storeu_ph(c_row.add(LANES) as *mut _, r1);
            }
        }

        // ── microkernel for edge N-strip (partial columns) ────────────────
        #[target_feature(enable = "avx512fp16")]
        unsafe fn microkernel_edge(
            a_panel: *const half::f16, a_stride: usize,
            b_panel: *const half::f16,
            c_ptr: *mut half::f16, c_stride: usize,
            mr: usize, an: usize, kc: usize,
        ) {
            // For partial N-strips, compute full tile then store only valid columns
            let mut acc: [[__m512h; NV]; TM] = [[_mm512_setzero_ph(); NV]; TM];

            for k in 0..kc {
                let b_base = b_panel.add(k * TN);
                let bv0 = load_b_vec(b_base);
                let bv1 = if an > LANES { load_b_vec(b_base.add(LANES)) } else { _mm512_setzero_ph() };

                macro_rules! row_fma {
                    ($r:expr) => {
                        if $r < mr {
                            let av = broadcast_a(a_panel.add($r * a_stride + k));
                            acc[$r][0] = _mm512_fmadd_ph(av, bv0, acc[$r][0]);
                            acc[$r][1] = _mm512_fmadd_ph(av, bv1, acc[$r][1]);
                        }
                    };
                }
                row_fma!(0);  row_fma!(1);  row_fma!(2);  row_fma!(3);
                row_fma!(4);  row_fma!(5);  row_fma!(6);  row_fma!(7);
                row_fma!(8);  row_fma!(9);  row_fma!(10); row_fma!(11);
                row_fma!(12); row_fma!(13);
            }

            // Store with masking for partial columns
            for r in 0..mr {
                let c_row = c_ptr.add(r * c_stride);
                if an >= LANES {
                    _mm512_storeu_ph(c_row as *mut _, acc[r][0]);
                    if an > LANES {
                        // Partial second vector
                        let rem = an - LANES;
                        let mut tmp = [half::f16::ZERO; 32];
                        _mm512_storeu_ph(tmp.as_mut_ptr() as *mut _, acc[r][1]);
                        for j in 0..rem {
                            *c_row.add(LANES + j) = tmp[j];
                        }
                    }
                } else {
                    // Partial first vector
                    let mut tmp = [half::f16::ZERO; 32];
                    _mm512_storeu_ph(tmp.as_mut_ptr() as *mut _, acc[r][0]);
                    for j in 0..an {
                        *c_row.add(j) = tmp[j];
                    }
                }
            }
        }

        // ── matmul: C = A * B ─────────────────────────────────────────────
        pub fn matmul(a: &[half::f16], b: &[half::f16], c: &mut [half::f16],
                      m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            assert!(c.len() >= m_size * n_size);

            let bp = _blocking();
            let kc_max = bp.kc;
            let n_strips = (n_size + TN - 1) / TN;

            // Pack B
            let packed_b = pack_b(b, n_size, k_size);

            let mut ks = 0usize;
            let mut ch = 0usize;
            let cs = n_strips * kc_max * TN;

            // Zero C on first K-chunk
            for v in c.iter_mut() { *v = half::f16::ZERO; }

            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let pb_base = &packed_b[ch * cs..];

                for ms in (0..m_size).step_by(TM) {
                    let mr = TM.min(m_size - ms);
                    for i in 0..n_strips {
                        let ns = i * TN;
                        let an = TN.min(n_size.saturating_sub(ns));
                        let b_panel = pb_base[i * kc_max * TN..].as_ptr();

                        unsafe {
                            if an == TN {
                                // Accumulate into temp, then add to C
                                let mut tmp = vec![half::f16::ZERO; mr * TN];
                                microkernel(
                                    a[ms * k_size + ks..].as_ptr(), k_size,
                                    b_panel,
                                    tmp.as_mut_ptr(), TN,
                                    mr, kc,
                                );
                                // Accumulate into C
                                for r in 0..mr {
                                    for j in 0..TN {
                                        let ci = (ms + r) * n_size + ns + j;
                                        c[ci] = half::f16::from_f32(c[ci].to_f32() + tmp[r * TN + j].to_f32());
                                    }
                                }
                            } else {
                                let mut tmp = vec![half::f16::ZERO; mr * TN];
                                microkernel_edge(
                                    a[ms * k_size + ks..].as_ptr(), k_size,
                                    b_panel,
                                    tmp.as_mut_ptr(), TN,
                                    mr, an, kc,
                                );
                                for r in 0..mr {
                                    for j in 0..an {
                                        let ci = (ms + r) * n_size + ns + j;
                                        c[ci] = half::f16::from_f32(c[ci].to_f32() + tmp[r * TN + j].to_f32());
                                    }
                                }
                            }
                        }
                    }
                }
                ks += kc;
                ch += 1;
            }
        }

        // ── matmul_bias: C = A * B + bias ─────────────────────────────────
        pub fn matmul_bias(a: &[half::f16], b: &[half::f16], bias: &[half::f16],
                           c: &mut [half::f16], m_size: usize, n_size: usize, k_size: usize) {
            // Compute C = A * B first, then add bias
            matmul(a, b, c, m_size, n_size, k_size);
            for m in 0..m_size {
                for n in 0..n_size {
                    let idx = m * n_size + n;
                    c[idx] = half::f16::from_f32(c[idx].to_f32() + bias[n].to_f32());
                }
            }
        }

        // ── matmul_prepacked: C = A * packed_B ────────────────────────────
        pub fn matmul_prepacked(a: &[half::f16], packed_b: &[half::f16], c: &mut [half::f16],
                                m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let bp = _blocking();
            let kc_max = bp.kc;
            let n_strips = (n_size + TN - 1) / TN;
            let cs = n_strips * kc_max * TN;

            for v in c.iter_mut() { *v = half::f16::ZERO; }

            let mut ks = 0usize;
            let mut ch = 0usize;
            while ks < k_size {
                let kc = kc_max.min(k_size - ks);
                let pb_base = &packed_b[ch * cs..];

                for ms in (0..m_size).step_by(TM) {
                    let mr = TM.min(m_size - ms);
                    for i in 0..n_strips {
                        let ns = i * TN;
                        let an = TN.min(n_size.saturating_sub(ns));
                        let b_panel = pb_base[i * kc_max * TN..].as_ptr();

                        unsafe {
                            let mut tmp = vec![half::f16::ZERO; mr * TN];
                            if an == TN {
                                microkernel(
                                    a[ms * k_size + ks..].as_ptr(), k_size,
                                    b_panel,
                                    tmp.as_mut_ptr(), TN,
                                    mr, kc,
                                );
                            } else {
                                microkernel_edge(
                                    a[ms * k_size + ks..].as_ptr(), k_size,
                                    b_panel,
                                    tmp.as_mut_ptr(), TN,
                                    mr, an, kc,
                                );
                            }
                            let cols = if an == TN { TN } else { an };
                            for r in 0..mr {
                                for j in 0..cols {
                                    let ci = (ms + r) * n_size + ns + j;
                                    c[ci] = half::f16::from_f32(c[ci].to_f32() + tmp[r * TN + j].to_f32());
                                }
                            }
                        }
                    }
                }
                ks += kc;
                ch += 1;
            }
        }

        // ── matmul_bias_prepacked: C = A * packed_B + bias ────────────────
        pub fn matmul_bias_prepacked(a: &[half::f16], packed_b: &[half::f16], bias: &[half::f16],
                                     c: &mut [half::f16], m_size: usize, n_size: usize, k_size: usize) {
            matmul_prepacked(a, packed_b, c, m_size, n_size, k_size);
            for m in 0..m_size {
                for n in 0..n_size {
                    let idx = m * n_size + n;
                    c[idx] = half::f16::from_f32(c[idx].to_f32() + bias[n].to_f32());
                }
            }
        }

        // ── matmul_bias_act: C = act(A * B + bias) ───────────────────────
        pub fn matmul_bias_act(a: &[half::f16], b: &[half::f16], bias: &[half::f16],
                               c: &mut [half::f16], m_size: usize, n_size: usize, k_size: usize,
                               act: $crate::Activation) {
            matmul_bias(a, b, bias, c, m_size, n_size, k_size);
            let len = m_size * n_size;
            match act {
                $crate::Activation::None => {},
                $crate::Activation::Relu => {
                    for i in 0..len {
                        if c[i].to_f32() < 0.0 { c[i] = half::f16::ZERO; }
                    }
                },
                $crate::Activation::Silu => {
                    for i in 0..len {
                        let v = c[i].to_f32();
                        c[i] = half::f16::from_f32(v / (1.0 + (-v).exp()));
                    }
                },
                $crate::Activation::Gelu => {
                    for i in 0..len {
                        let x = c[i].to_f32();
                        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
                        c[i] = half::f16::from_f32(0.5 * x * (1.0 + inner.tanh()));
                    }
                },
            }
        }
    };
}
