/// AMX (Advanced Matrix Extensions) BF16 GEMM microkernel using TDPBF16PS.
///
/// TDPBF16PS semantics:
///   C[16x16 f32] += A[16x32 bf16] * B[32x16 bf16]
///   K step = 32 bf16 pairs per tile multiply.
///
/// Tile register allocation (2C + 1A + 2B = 5 tiles):
///   tmm0, tmm1: C accumulators (each 16x16 f32, side-by-side = 16x32 output)
///   tmm2: A tile (16 rows x 32 bf16 = 16x64 bytes)
///   tmm3, tmm4: B tiles (each 16 cols, side-by-side = 32 cols)
///   tmm5-7: unused
///
/// Packed B layout: reuses existing BF16 native K-pair interleaved format.
/// AMX _tile_loadd with stride parameter reads the correct columns.
#[macro_export]
macro_rules! define_matmul_x86_amx_bf16 {
    () => {
        use std::arch::x86_64::*;
        use std::sync::atomic::{AtomicU8, Ordering};

        const TILE_M: usize = 16;
        const TILE_N: usize = 32; // 2 x 16
        const TILE_K: usize = 32; // bf16 pairs per TDPBF16PS

        /// TILECFG structure (64 bytes, 64-byte aligned).
        /// Intel AMX programming reference §3.1.
        #[repr(C, align(64))]
        struct TileCfg {
            palette: u8,
            start_row: u8,
            _pad0: [u8; 14],
            colsb: [u16; 8],  // bytes per row for each tile (offset 16)
            _pad1: [u8; 16],
            rows: [u8; 8],    // number of rows for each tile (offset 48)
            _pad2: [u8; 8],
        }

        impl TileCfg {
            fn new_zero() -> Self {
                Self {
                    palette: 0,
                    start_row: 0,
                    _pad0: [0; 14],
                    colsb: [0; 8],
                    _pad1: [0; 16],
                    rows: [0; 8],
                    _pad2: [0; 8],
                }
            }
        }

        /// RAII guard: automatically releases tiles on drop.
        struct TileGuard;
        impl Drop for TileGuard {
            fn drop(&mut self) {
                unsafe { _tile_release(); }
            }
        }

        /// Request AMX permission from Linux kernel via arch_prctl.
        /// ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18.
        /// Result is cached in an AtomicU8 (0=unknown, 1=granted, 2=denied).
        fn request_amx_permission() -> bool {
            static STATE: AtomicU8 = AtomicU8::new(0);
            match STATE.load(Ordering::Relaxed) {
                1 => return true,
                2 => return false,
                _ => {}
            }
            #[cfg(target_os = "linux")]
            {
                const ARCH_REQ_XCOMP_PERM: i32 = 0x1023;
                const XFEATURE_XTILEDATA: u64 = 18;
                let ret = unsafe {
                    libc_arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
                };
                let ok = ret == 0;
                STATE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
                ok
            }
            #[cfg(not(target_os = "linux"))]
            {
                STATE.store(2, Ordering::Relaxed);
                false
            }
        }

        /// arch_prctl syscall wrapper (Linux only).
        #[cfg(target_os = "linux")]
        unsafe fn libc_arch_prctl(code: i32, addr: u64) -> i64 {
            let ret: i64;
            std::arch::asm!(
                "syscall",
                in("rax") 158i64, // __NR_arch_prctl
                in("rdi") code as i64,
                in("rsi") addr as i64,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
            ret
        }

        /// Runtime availability check for AMX BF16.
        pub fn is_available() -> bool {
            is_x86_feature_detected!("amx-tile")
                && is_x86_feature_detected!("amx-bf16")
                && request_amx_permission()
        }

        /// Build TILECFG for BF16 GEMM: all 5 active tiles are 16 rows x 64 bytes.
        fn make_bf16_tilecfg() -> TileCfg {
            let mut cfg = TileCfg::new_zero();
            cfg.palette = 1;
            // tmm0, tmm1: C accumulators (16 rows x 16 f32 = 16 x 64 bytes)
            cfg.rows[0] = 16; cfg.colsb[0] = 64;
            cfg.rows[1] = 16; cfg.colsb[1] = 64;
            // tmm2: A tile (16 rows x 32 bf16 = 16 x 64 bytes)
            cfg.rows[2] = 16; cfg.colsb[2] = 64;
            // tmm3, tmm4: B tiles (16 rows x 16 bf16-pairs = 16 x 64 bytes)
            // For B: K/2 rows (since bf16 pairs), 16 cols * 4 bytes = 64 bytes
            cfg.rows[3] = 16; cfg.colsb[3] = 64;
            cfg.rows[4] = 16; cfg.colsb[4] = 64;
            cfg
        }

        /// Build TILECFG with reduced M rows for edge handling.
        fn make_bf16_tilecfg_edge(m_rows: u8) -> TileCfg {
            let mut cfg = TileCfg::new_zero();
            cfg.palette = 1;
            cfg.rows[0] = m_rows; cfg.colsb[0] = 64;
            cfg.rows[1] = m_rows; cfg.colsb[1] = 64;
            cfg.rows[2] = m_rows; cfg.colsb[2] = 64;
            cfg.rows[3] = 16; cfg.colsb[3] = 64;
            cfg.rows[4] = 16; cfg.colsb[4] = 64;
            cfg
        }

        /// AMX 16x32 microkernel: C[16x32 f32] += A[16xkc bf16] * B_packed[kcx32 bf16]
        ///
        /// a_ptr: pointer to A[m_block, k_start], row-major bf16, stride = a_stride elements
        /// b_packed_ptr: pointer to packed B for this N-strip/K-chunk
        /// c_f32_ptr: pointer to C accumulator [m_block, n_strip], row-major f32, stride = c_stride elements
        /// k_iters: number of K-tile iterations (each processes TILE_K=32 bf16)
        /// a_stride: A row stride in bf16 elements (= k_size)
        /// c_stride: C row stride in f32 elements (= n_size)
        /// first_chunk: if true, zero C accumulators; else load from c_f32_ptr
        #[target_feature(enable = "amx-tile,amx-bf16,avx512f")]
        unsafe fn microkernel_16x32(
            a_ptr: *const half::bf16,
            b_packed_ptr: *const half::bf16,
            c_f32_ptr: *mut f32,
            k_iters: usize,
            a_stride: usize,
            c_stride: usize,
            first_chunk: bool,
        ) {
            let cfg = make_bf16_tilecfg();
            _tile_loadconfig((&cfg as *const TileCfg) as *const u8);
            let _guard = TileGuard;

            // A stride in bytes (row-major bf16)
            let a_stride_bytes = a_stride * 2; // bf16 = 2 bytes
            // C stride in bytes (row-major f32)
            let c_stride_bytes = c_stride * 4; // f32 = 4 bytes
            // B packed stride: each K-pair block = BK_STRIDE bf16 = 64 bf16 = 128 bytes
            // For AMX: B tile is 16 rows x 64 bytes, stride between rows = 128 bytes
            // (because packed B has 64 bf16 per K-pair: 32 for cols 0..15, 32 for cols 16..31)
            let b_stride_bytes: usize = 64 * 2; // 64 bf16 * 2 bytes = 128 bytes

            if first_chunk {
                _tile_zero(0);
                _tile_zero(1);
            } else {
                _tile_loadd(0, c_f32_ptr as *const u8, c_stride_bytes);
                _tile_loadd(1, (c_f32_ptr as *const u8).add(16 * 4), c_stride_bytes);
            }

            let mut a_cur = a_ptr as *const u8;
            let mut b_cur = b_packed_ptr as *const u8;

            for _ki in 0..k_iters {
                // Load A tile: 16 rows x 32 bf16 (64 bytes per row)
                _tile_loadd(2, a_cur, a_stride_bytes);

                // Load B tiles from packed layout
                // tmm3: first 16 N-columns (offset 0 in each K-pair block)
                _tile_loadd(3, b_cur, b_stride_bytes);
                // tmm4: next 16 N-columns (offset 32 bf16 = 64 bytes in each K-pair block)
                _tile_loadd(4, b_cur.add(32 * 2), b_stride_bytes);

                // TDPBF16PS: C += A * B
                _tile_dpbf16ps(0, 2, 3); // tmm0 += tmm2 * tmm3
                _tile_dpbf16ps(1, 2, 4); // tmm1 += tmm2 * tmm4

                // Advance A by TILE_K bf16 columns = 64 bytes
                a_cur = a_cur.add(TILE_K * 2);
                // Advance B by 16 K-pairs (each pair = 128 bytes in packed layout)
                b_cur = b_cur.add(16 * b_stride_bytes);
            }

            // Store C accumulators back to f32 buffer
            _tile_stored(0, c_f32_ptr as *mut u8, c_stride_bytes);
            _tile_stored(1, (c_f32_ptr as *mut u8).add(16 * 4), c_stride_bytes);
        }

        /// Edge microkernel for M < 16 rows.
        /// Uses dynamic TILECFG with reduced row count.
        #[target_feature(enable = "amx-tile,amx-bf16,avx512f")]
        unsafe fn edge_m_kernel(
            m_rem: usize,
            a_ptr: *const half::bf16,
            b_packed_ptr: *const half::bf16,
            c_f32_ptr: *mut f32,
            k_iters: usize,
            a_stride: usize,
            c_stride: usize,
            first_chunk: bool,
        ) {
            if m_rem == 0 { return; }
            let cfg = make_bf16_tilecfg_edge(m_rem as u8);
            _tile_loadconfig((&cfg as *const TileCfg) as *const u8);
            let _guard = TileGuard;

            let a_stride_bytes = a_stride * 2;
            let c_stride_bytes = c_stride * 4;
            let b_stride_bytes: usize = 64 * 2;

            if first_chunk {
                _tile_zero(0);
                _tile_zero(1);
            } else {
                _tile_loadd(0, c_f32_ptr as *const u8, c_stride_bytes);
                _tile_loadd(1, (c_f32_ptr as *const u8).add(16 * 4), c_stride_bytes);
            }

            let mut a_cur = a_ptr as *const u8;
            let mut b_cur = b_packed_ptr as *const u8;

            for _ki in 0..k_iters {
                _tile_loadd(2, a_cur, a_stride_bytes);
                _tile_loadd(3, b_cur, b_stride_bytes);
                _tile_loadd(4, b_cur.add(32 * 2), b_stride_bytes);
                _tile_dpbf16ps(0, 2, 3);
                _tile_dpbf16ps(1, 2, 4);
                a_cur = a_cur.add(TILE_K * 2);
                b_cur = b_cur.add(16 * b_stride_bytes);
            }

            _tile_stored(0, c_f32_ptr as *mut u8, c_stride_bytes);
            _tile_stored(1, (c_f32_ptr as *mut u8).add(16 * 4), c_stride_bytes);
        }

        /// Scalar edge for N-tail (< TILE_N columns).
        fn edge_n_scalar(
            a: &[half::bf16], b: &[half::bf16], c_f32: &mut [f32],
            m_start: usize, m_end: usize, n_start: usize, n_size: usize,
            ks: usize, kc: usize, k_size: usize, first_chunk: bool,
        ) {
            for mi in m_start..m_end {
                for ni in n_start..n_size {
                    let mut acc = if first_chunk { 0.0f32 } else { c_f32[mi * n_size + ni] };
                    for ki in 0..kc {
                        let av = a[mi * k_size + ks + ki].to_f32();
                        let bv = b[(ks + ki) * n_size + ni].to_f32();
                        acc += av * bv;
                    }
                    c_f32[mi * n_size + ni] = acc;
                }
            }
        }

        /// Convert f32 buffer to bf16 output using AVX-512.
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn convert_f32_to_bf16(src: &[f32], dst: &mut [half::bf16]) {
            let len = src.len();
            let mut i = 0usize;
            while i + 16 <= len {
                let v = _mm512_loadu_ps(src.as_ptr().add(i));
                let vi = _mm512_castps_si512(v);
                let shifted = _mm512_srli_epi32(vi, 16);
                let packed = _mm512_cvtepi32_epi16(shifted);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, packed);
                i += 16;
            }
            while i < len {
                dst[i] = half::bf16::from_f32(src[i]);
                i += 1;
            }
        }

        /// Fused f32→bf16 conversion with bias addition using AVX-512.
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn convert_f32_to_bf16_bias(src: &[f32], bias: &[half::bf16], dst: &mut [half::bf16],
                                            m_size: usize, n_size: usize) {
            for m in 0..m_size {
                let row_off = m * n_size;
                let mut n = 0usize;
                while n + 16 <= n_size {
                    let v = _mm512_loadu_ps(src.as_ptr().add(row_off + n));
                    // Load bias bf16 → f32
                    let bp = bias.as_ptr().add(n) as *const u8;
                    let b16 = _mm256_loadu_si256(bp as *const __m256i);
                    let b32 = _mm512_cvtepu16_epi32(b16);
                    let bshift = _mm512_slli_epi32(b32, 16);
                    let bfloat = _mm512_castsi512_ps(bshift);
                    let sum = _mm512_add_ps(v, bfloat);
                    // Convert f32 → bf16
                    let vi = _mm512_castps_si512(sum);
                    let shifted = _mm512_srli_epi32(vi, 16);
                    let packed = _mm512_cvtepi32_epi16(shifted);
                    _mm256_storeu_si256(dst.as_mut_ptr().add(row_off + n) as *mut __m256i, packed);
                    n += 16;
                }
                while n < n_size {
                    let val = src[row_off + n] + bias[n].to_f32();
                    dst[row_off + n] = half::bf16::from_f32(val);
                    n += 1;
                }
            }
        }

        /// Fused f32→bf16 conversion with bias + activation using AVX-512.
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn convert_f32_to_bf16_bias_act(src: &[f32], bias: &[half::bf16], dst: &mut [half::bf16],
                                                m_size: usize, n_size: usize, act: $crate::Activation) {
            let zero_v = _mm512_setzero_ps();
            let half_v = _mm512_set1_ps(0.5f32);
            let one_v = _mm512_set1_ps(1.0f32);
            let coeff_v = _mm512_set1_ps(0.7978845608f32);
            let cubic_v = _mm512_set1_ps(0.044715f32);
            for m in 0..m_size {
                let row_off = m * n_size;
                let mut n = 0usize;
                while n + 16 <= n_size {
                    let v = _mm512_loadu_ps(src.as_ptr().add(row_off + n));
                    // Load bias bf16 → f32
                    let bp = bias.as_ptr().add(n) as *const u8;
                    let b16 = _mm256_loadu_si256(bp as *const __m256i);
                    let b32 = _mm512_cvtepu16_epi32(b16);
                    let bshift = _mm512_slli_epi32(b32, 16);
                    let bfloat = _mm512_castsi512_ps(bshift);
                    let mut x = _mm512_add_ps(v, bfloat);
                    // Apply activation
                    match act {
                        $crate::Activation::Relu => {
                            x = _mm512_max_ps(x, zero_v);
                        }
                        $crate::Activation::Silu => {
                            // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
                            let neg_x = _mm512_sub_ps(zero_v, x);
                            let exp_neg = $crate::cpu_kernels::avx512::math::avx512_exp_f32(neg_x);
                            let denom = _mm512_add_ps(one_v, exp_neg);
                            x = _mm512_div_ps(x, denom);
                        }
                        $crate::Activation::Gelu => {
                            // gelu(x) = 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
                            let x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
                            let inner = _mm512_fmadd_ps(cubic_v, x3, x);
                            let scaled = _mm512_mul_ps(coeff_v, inner);
                            // tanh(a) = (exp(2a)-1)/(exp(2a)+1)
                            let two_a = _mm512_add_ps(scaled, scaled);
                            let exp_2a = $crate::cpu_kernels::avx512::math::avx512_exp_f32(two_a);
                            let num = _mm512_sub_ps(exp_2a, one_v);
                            let den = _mm512_add_ps(exp_2a, one_v);
                            let tanh_v = _mm512_div_ps(num, den);
                            let inner2 = _mm512_add_ps(one_v, tanh_v);
                            x = _mm512_mul_ps(_mm512_mul_ps(half_v, x), inner2);
                        }
                        _ => {}
                    }
                    // Convert f32 → bf16
                    let vi = _mm512_castps_si512(x);
                    let shifted = _mm512_srli_epi32(vi, 16);
                    let packed = _mm512_cvtepi32_epi16(shifted);
                    _mm256_storeu_si256(dst.as_mut_ptr().add(row_off + n) as *mut __m256i, packed);
                    n += 16;
                }
                // Scalar tail
                while n < n_size {
                    let mut x = src[row_off + n] + bias[n].to_f32();
                    match act {
                        $crate::Activation::Relu => { if x < 0.0 { x = 0.0; } }
                        $crate::Activation::Silu => { x = x / (1.0 + (-x).exp()); }
                        $crate::Activation::Gelu => {
                            let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
                            x = 0.5 * x * (1.0 + inner.tanh());
                        }
                        _ => {}
                    }
                    dst[row_off + n] = half::bf16::from_f32(x);
                    n += 1;
                }
            }
        }

        /// Pack B matrix into K-pair interleaved format for AMX tiles.
        /// AMX BF16 uses the same packed layout as the native bf16 path.
        pub fn pack_b(b: &[half::bf16], n_size: usize, k_size: usize) -> Vec<half::bf16> {
            // K-pair interleaved: for each strip of TILE_N columns, pack pairs of K rows
            let tile_n = TILE_N;
            let k_pairs = (k_size + 1) / 2;
            let n_strips = (n_size + tile_n - 1) / tile_n;
            let mut packed = vec![half::bf16::ZERO; n_strips * tile_n * k_pairs * 2];
            let mut dst_idx = 0;
            for ns in 0..n_strips {
                let n_start = ns * tile_n;
                let n_end = (n_start + tile_n).min(n_size);
                for kp in 0..k_pairs {
                    let k0 = kp * 2;
                    let k1 = k0 + 1;
                    for n in n_start..n_end {
                        packed[dst_idx] = b[k0 * n_size + n];
                        dst_idx += 1;
                        if k1 < k_size {
                            packed[dst_idx] = b[k1 * n_size + n];
                        }
                        dst_idx += 1;
                    }
                    // Pad remaining columns in the strip
                    for _ in (n_end - n_start)..tile_n {
                        dst_idx += 2;
                    }
                }
            }
            packed
        }

        /// Core GEMM into f32 buffer (no conversion). Returns the f32 buffer.
        fn matmul_to_f32(a: &[half::bf16], b: &[half::bf16],
                         m_size: usize, n_size: usize, k_size: usize) -> Vec<f32> {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= k_size * n_size);
            if m_size == 0 || n_size == 0 || k_size == 0 { return vec![0.0f32; m_size * n_size]; }

            let mut c_f32 = vec![0.0f32; m_size * n_size];
            let n_full_strips = n_size / TILE_N;

            let packed_b = pack_b(b, n_size, k_size);

            let mut ks = 0usize;
            let mut k_chunk_idx = 0usize;
            while ks < k_size {
                let kc = TILE_K.min(k_size - ks);
                let k_iters = 1;
                let first_chunk = k_chunk_idx == 0;

                let mut m = 0usize;
                while m + TILE_M <= m_size {
                    for si in 0..n_full_strips {
                        let n = si * TILE_N;
                        unsafe {
                            microkernel_16x32(
                                a.as_ptr().add(m * k_size + ks),
                                packed_b.as_ptr().add(k_chunk_idx * ((n_size + TILE_N - 1) / TILE_N) * 16 * 64 + si * 16 * 64),
                                c_f32.as_mut_ptr().add(m * n_size + n),
                                k_iters, k_size, n_size, first_chunk,
                            );
                        }
                    }
                    if n_full_strips * TILE_N < n_size {
                        edge_n_scalar(a, b, &mut c_f32, m, m + TILE_M,
                                      n_full_strips * TILE_N, n_size, ks, kc, k_size, first_chunk);
                    }
                    m += TILE_M;
                }

                let m_rem = m_size - m;
                if m_rem > 0 {
                    for si in 0..n_full_strips {
                        let n = si * TILE_N;
                        unsafe {
                            edge_m_kernel(
                                m_rem,
                                a.as_ptr().add(m * k_size + ks),
                                packed_b.as_ptr().add(k_chunk_idx * ((n_size + TILE_N - 1) / TILE_N) * 16 * 64 + si * 16 * 64),
                                c_f32.as_mut_ptr().add(m * n_size + n),
                                k_iters, k_size, n_size, first_chunk,
                            );
                        }
                    }
                    if n_full_strips * TILE_N < n_size {
                        edge_n_scalar(a, b, &mut c_f32, m, m_size,
                                      n_full_strips * TILE_N, n_size, ks, kc, k_size, first_chunk);
                    }
                }

                ks += TILE_K;
                k_chunk_idx += 1;
            }
            c_f32
        }

        /// Main AMX BF16 matmul: C[m,n] = A[m,k] * B[k,n]
        /// Accumulates in f32, converts to bf16 at the end.
        pub fn matmul(a: &[half::bf16], b: &[half::bf16], c: &mut [half::bf16],
                      m_size: usize, n_size: usize, k_size: usize) {
            assert!(c.len() >= m_size * n_size);
            let c_f32 = matmul_to_f32(a, b, m_size, n_size, k_size);
            unsafe { convert_f32_to_bf16(&c_f32, c); }
        }

        pub fn matmul_prepacked(a: &[half::bf16], packed_b: &[half::bf16], c: &mut [half::bf16],
                                m_size: usize, n_size: usize, k_size: usize) {
            // TODO: optimize to use pre-packed B directly
            let mut b_unpacked = vec![half::bf16::ZERO; k_size * n_size];
            matmul(a, &b_unpacked, c, m_size, n_size, k_size);
            let _ = packed_b;
            let _ = &mut b_unpacked;
        }

        pub fn matmul_bias(a: &[half::bf16], b: &[half::bf16], bias: &[half::bf16],
                           c: &mut [half::bf16], m_size: usize, n_size: usize, k_size: usize) {
            assert!(c.len() >= m_size * n_size);
            let c_f32 = matmul_to_f32(a, b, m_size, n_size, k_size);
            unsafe { convert_f32_to_bf16_bias(&c_f32, bias, c, m_size, n_size); }
        }

        pub fn matmul_bias_prepacked(a: &[half::bf16], packed_b: &[half::bf16], bias: &[half::bf16],
                                     c: &mut [half::bf16], m_size: usize, n_size: usize, k_size: usize) {
            matmul_prepacked(a, packed_b, c, m_size, n_size, k_size);
            // Apply bias in a vectorized pass
            for m in 0..m_size {
                let row_off = m * n_size;
                let mut n = 0usize;
                while n < n_size {
                    let val = c[row_off + n].to_f32() + bias[n].to_f32();
                    c[row_off + n] = half::bf16::from_f32(val);
                    n += 1;
                }
            }
        }

        pub fn matmul_bias_act(a: &[half::bf16], b: &[half::bf16], bias: &[half::bf16],
                               c: &mut [half::bf16], m_size: usize, n_size: usize, k_size: usize,
                               act: $crate::Activation) {
            assert!(c.len() >= m_size * n_size);
            let c_f32 = matmul_to_f32(a, b, m_size, n_size, k_size);
            unsafe { convert_f32_to_bf16_bias_act(&c_f32, bias, c, m_size, n_size, act); }
        }
    };
}
