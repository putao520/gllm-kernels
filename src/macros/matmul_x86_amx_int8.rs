/// AMX INT8 GEMM microkernel using TDPBUSD (u8 × i8 → i32).
///
/// TDPBUSD semantics:
///   C[16x16 i32] += A[16x64 u8] * B[64x16 i8]
///   K step = 64 bytes per tile multiply.
///
/// Tile register allocation (2C + 1A + 2B = 5 tiles):
///   tmm0, tmm1: C accumulators (each 16x16 i32 = 16x64 bytes)
///   tmm2: A tile (16 rows x 64 u8 = 16x64 bytes)
///   tmm3, tmm4: B tiles (each 16 cols of i32, packed as 64x16 i8)
///
/// Reuses the existing INT8 K-quad interleaved packed B layout.
#[macro_export]
macro_rules! define_matmul_x86_amx_int8 {
    () => {
        use std::arch::x86_64::*;
        use std::sync::atomic::{AtomicU8, Ordering};

        const TILE_M: usize = 16;
        const TILE_N: usize = 32; // 2 x 16
        const TILE_K: usize = 64; // bytes per TDPBUSD A-tile row

        /// TILECFG structure (64 bytes, 64-byte aligned).
        #[repr(C, align(64))]
        struct TileCfg {
            palette: u8,
            start_row: u8,
            _pad0: [u8; 14],
            colsb: [u16; 8],
            _pad1: [u8; 16],
            rows: [u8; 8],
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

        struct TileGuard;
        impl Drop for TileGuard {
            fn drop(&mut self) {
                unsafe { _tile_release(); }
            }
        }

        /// Request AMX permission from Linux kernel.
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
                let ret = unsafe { libc_arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) };
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

        #[cfg(target_os = "linux")]
        unsafe fn libc_arch_prctl(code: i32, addr: u64) -> i64 {
            let ret: i64;
            std::arch::asm!(
                "syscall",
                in("rax") 158i64,
                in("rdi") code as i64,
                in("rsi") addr as i64,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
            ret
        }

        /// Runtime availability check for AMX INT8.
        pub fn is_available() -> bool {
            is_x86_feature_detected!("amx-tile")
                && is_x86_feature_detected!("amx-int8")
                && request_amx_permission()
        }

        /// Build TILECFG for INT8 GEMM.
        /// tmm0,1: C accumulators (16 rows x 64 bytes = 16 x 16 i32)
        /// tmm2: A tile (16 rows x 64 bytes = 16 x 64 u8)
        /// tmm3,4: B tiles (16 rows x 64 bytes = 16 x 16 i32-packed-i8)
        fn make_int8_tilecfg() -> TileCfg {
            let mut cfg = TileCfg::new_zero();
            cfg.palette = 1;
            for t in 0..5u8 {
                cfg.rows[t as usize] = 16;
                cfg.colsb[t as usize] = 64;
            }
            cfg
        }

        fn make_int8_tilecfg_edge(m_rows: u8) -> TileCfg {
            let mut cfg = TileCfg::new_zero();
            cfg.palette = 1;
            cfg.rows[0] = m_rows; cfg.colsb[0] = 64;
            cfg.rows[1] = m_rows; cfg.colsb[1] = 64;
            cfg.rows[2] = m_rows; cfg.colsb[2] = 64;
            cfg.rows[3] = 16; cfg.colsb[3] = 64;
            cfg.rows[4] = 16; cfg.colsb[4] = 64;
            cfg
        }

        /// Delegate pack_b to existing INT8 K-quad interleaved packer.
        pub fn pack_b(b: &[i8], n_size: usize, k_size: usize) -> Vec<i8> {
            crate::cpu_kernels::avx512::avx512_int8::pack_b(b, n_size, k_size)
        }

        /// AMX 16x32 INT8 microkernel.
        /// a_ptr: pointer to A[m, k_start] (row-major u8, stride = a_stride bytes)
        /// b_packed_ptr: pointer to packed B for this strip/chunk
        /// c_ptr: pointer to C[m, n] (row-major i32, stride = c_stride i32 elements)
        /// k_iters: number of K-tile iterations (each processes TILE_K=64 u8)
        /// a_stride: A row stride in bytes (= k_size, since u8)
        /// c_stride: C row stride in i32 elements (= n_size)
        #[target_feature(enable = "amx-tile,amx-int8,avx512f")]
        unsafe fn microkernel_16x32(
            a_ptr: *const u8,
            b_packed_ptr: *const i8,
            c_ptr: *mut i32,
            k_iters: usize,
            a_stride: usize,
            c_stride: usize,
            first_chunk: bool,
        ) {
            let cfg = make_int8_tilecfg();
            _tile_loadconfig((&cfg as *const TileCfg) as *const u8);
            let _guard = TileGuard;

            let a_stride_bytes = a_stride; // u8, so stride in bytes = stride in elements
            let c_stride_bytes = c_stride * 4; // i32 = 4 bytes
            // B packed: K-quad interleaved, BK_STRIDE = 128 bytes per K-quad
            // For AMX: we need 64 bytes per B-tile row, stride between rows = 128 bytes
            let b_stride_bytes: usize = 128;

            if first_chunk {
                _tile_zero(0);
                _tile_zero(1);
            } else {
                _tile_loadd(0, c_ptr as *const u8, c_stride_bytes);
                _tile_loadd(1, (c_ptr as *const u8).add(16 * 4), c_stride_bytes);
            }

            let mut a_cur = a_ptr;
            let mut b_cur = b_packed_ptr as *const u8;

            for _ki in 0..k_iters {
                _tile_loadd(2, a_cur, a_stride_bytes);
                _tile_loadd(3, b_cur, b_stride_bytes);
                _tile_loadd(4, b_cur.add(64), b_stride_bytes);

                _tile_dpbusd(0, 2, 3);
                _tile_dpbusd(1, 2, 4);

                a_cur = a_cur.add(TILE_K);
                b_cur = b_cur.add(16 * b_stride_bytes);
            }

            _tile_stored(0, c_ptr as *mut u8, c_stride_bytes);
            _tile_stored(1, (c_ptr as *mut u8).add(16 * 4), c_stride_bytes);
        }

        /// Edge microkernel for M < 16 rows.
        #[target_feature(enable = "amx-tile,amx-int8,avx512f")]
        unsafe fn edge_m_kernel(
            m_rem: usize,
            a_ptr: *const u8,
            b_packed_ptr: *const i8,
            c_ptr: *mut i32,
            k_iters: usize,
            a_stride: usize,
            c_stride: usize,
            first_chunk: bool,
        ) {
            if m_rem == 0 { return; }
            let cfg = make_int8_tilecfg_edge(m_rem as u8);
            _tile_loadconfig((&cfg as *const TileCfg) as *const u8);
            let _guard = TileGuard;

            let a_stride_bytes = a_stride;
            let c_stride_bytes = c_stride * 4;
            let b_stride_bytes: usize = 128;

            if first_chunk {
                _tile_zero(0);
                _tile_zero(1);
            } else {
                _tile_loadd(0, c_ptr as *const u8, c_stride_bytes);
                _tile_loadd(1, (c_ptr as *const u8).add(16 * 4), c_stride_bytes);
            }

            let mut a_cur = a_ptr;
            let mut b_cur = b_packed_ptr as *const u8;

            for _ki in 0..k_iters {
                _tile_loadd(2, a_cur, a_stride_bytes);
                _tile_loadd(3, b_cur, b_stride_bytes);
                _tile_loadd(4, b_cur.add(64), b_stride_bytes);
                _tile_dpbusd(0, 2, 3);
                _tile_dpbusd(1, 2, 4);
                a_cur = a_cur.add(TILE_K);
                b_cur = b_cur.add(16 * b_stride_bytes);
            }

            _tile_stored(0, c_ptr as *mut u8, c_stride_bytes);
            _tile_stored(1, (c_ptr as *mut u8).add(16 * 4), c_stride_bytes);
        }

        /// Scalar edge for N-tail (< TILE_N columns).
        fn edge_n_scalar(
            a: &[u8], b: &[i8], c: &mut [i32],
            m_start: usize, m_end: usize, n_start: usize, n_size: usize,
            ks: usize, kc: usize, k_size: usize, first_chunk: bool,
        ) {
            for mi in m_start..m_end {
                for ni in n_start..n_size {
                    let mut acc = if first_chunk { 0i32 } else { c[mi * n_size + ni] };
                    for ki in 0..kc {
                        acc += a[mi * k_size + ks + ki] as i32 * b[(ks + ki) * n_size + ni] as i32;
                    }
                    c[mi * n_size + ni] = acc;
                }
            }
        }

        /// Top-level AMX INT8 GEMM: C[i32] = A[u8] * B[i8]
        /// b_orig: original unpacked B for N-tail scalar fallback.
        #[target_feature(enable = "amx-tile,amx-int8,avx512f")]
        pub unsafe fn gemm_int8(
            a: &[u8], b_packed: &[i8], b_orig: &[i8], c: &mut [i32],
            m_size: usize, n_size: usize, k_size: usize,
        ) {
            if m_size == 0 || n_size == 0 || k_size == 0 { return; }

            let n_full_strips = n_size / TILE_N;
            let _k_tiles = (k_size + TILE_K - 1) / TILE_K;

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
                        microkernel_16x32(
                            a.as_ptr().add(m * k_size + ks),
                            b_packed.as_ptr().add(k_chunk_idx * n_full_strips * 16 * 128 + si * 16 * 128),
                            c.as_mut_ptr().add(m * n_size + n),
                            k_iters, k_size, n_size, first_chunk,
                        );
                    }
                    if n_full_strips * TILE_N < n_size {
                        edge_n_scalar(a, b_orig, c, m, m + TILE_M,
                                      n_full_strips * TILE_N, n_size, ks, kc, k_size, first_chunk);
                    }
                    m += TILE_M;
                }

                let m_rem = m_size - m;
                if m_rem > 0 {
                    for si in 0..n_full_strips {
                        let n = si * TILE_N;
                        edge_m_kernel(
                            m_rem,
                            a.as_ptr().add(m * k_size + ks),
                            b_packed.as_ptr().add(k_chunk_idx * n_full_strips * 16 * 128 + si * 16 * 128),
                            c.as_mut_ptr().add(m * n_size + n),
                            k_iters, k_size, n_size, first_chunk,
                        );
                    }
                    if n_full_strips * TILE_N < n_size {
                        edge_n_scalar(a, b_orig, c, m, m_size,
                                      n_full_strips * TILE_N, n_size, ks, kc, k_size, first_chunk);
                    }
                }

                ks += TILE_K;
                k_chunk_idx += 1;
            }
        }

        /// Dequantize i32 -> f32 with fused epilogue (reuses AVX-512 VNNI version).
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
                while j + 16 <= n {
                    let idx = i * n + j;
                    let vi = _mm512_loadu_si512(src.add(idx) as *const __m512i);
                    let mut vf = _mm512_cvtepi32_ps(vi);
                    vf = _mm512_mul_ps(vf, combined_scale);
                    if let Some(b) = bias {
                        let vb = _mm512_loadu_ps(b.as_ptr().add(j));
                        vf = _mm512_add_ps(vf, vb);
                    }
                    vf = match activation {
                        $crate::Activation::None => vf,
                        $crate::Activation::Relu => _mm512_max_ps(vf, _mm512_setzero_ps()),
                        $crate::Activation::Silu => {
                            let neg_v = _mm512_sub_ps(_mm512_setzero_ps(), vf);
                            let exp_neg = $crate::cpu_kernels::avx512::math::avx512_exp_f32(neg_v);
                            let one = _mm512_set1_ps(1.0);
                            let denom = _mm512_add_ps(one, exp_neg);
                            let sigmoid = _mm512_div_ps(one, denom);
                            _mm512_mul_ps(vf, sigmoid)
                        }
                        $crate::Activation::Gelu => {
                            let half = _mm512_set1_ps(0.5);
                            let one = _mm512_set1_ps(1.0);
                            let coeff = _mm512_set1_ps(0.044715);
                            let sqrt2pi = _mm512_set1_ps(0.7978845608);
                            let two = _mm512_set1_ps(2.0);
                            let x2 = _mm512_mul_ps(vf, vf);
                            let x3 = _mm512_mul_ps(x2, vf);
                            let inner = _mm512_fmadd_ps(coeff, x3, vf);
                            let scaled = _mm512_mul_ps(sqrt2pi, inner);
                            let two_x = _mm512_mul_ps(two, scaled);
                            let exp_2x = $crate::cpu_kernels::avx512::math::avx512_exp_f32(two_x);
                            let num = _mm512_sub_ps(exp_2x, one);
                            let den = _mm512_add_ps(exp_2x, one);
                            let tanh_val = _mm512_div_ps(num, den);
                            let one_plus_tanh = _mm512_add_ps(one, tanh_val);
                            let half_x = _mm512_mul_ps(half, vf);
                            _mm512_mul_ps(half_x, one_plus_tanh)
                        }
                    };
                    _mm512_storeu_ps(dst.add(idx), vf);
                    j += 16;
                }
                while j < n {
                    let idx = i * n + j;
                    let mut val = c_i32[idx] as f32 * scale_a * scale_b;
                    if let Some(b) = bias { val += b[j]; }
                    val = match activation {
                        $crate::Activation::None => val,
                        $crate::Activation::Relu => val.max(0.0),
                        $crate::Activation::Gelu => {
                            let inner = 0.7978845608f32 * (val + 0.044715f32 * val * val * val);
                            0.5 * val * (1.0 + inner.tanh())
                        }
                        $crate::Activation::Silu => val / (1.0 + (-val).exp()),
                    };
                    c_f32[idx] = val;
                    j += 1;
                }
            }
        }
    };
}
