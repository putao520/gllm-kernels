/// Full GEMM using the hand-written 6x16 AVX2 microkernel.
///
/// C = A * B  (row-major, C is m x n, A is m x k, B is k x n)
#[cfg(target_arch = "x86_64")]
pub fn gemm_asm_f32_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx2::{MR, NR, gemm_kernel_6x16_f32};

    gemm_driver_f32_mt(a, b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, _bias| unsafe {
        gemm_kernel_6x16_f32(pa, pb, cp, kc, ldc, acc);
    }, std::ptr::null());
}

/// Full GEMM using the hand-written 14x32 AVX-512 microkernel.
///
/// C = A * B  (row-major, C is m x n, A is m x k, B is k x n)
#[cfg(target_arch = "x86_64")]
pub fn gemm_asm_f32_avx512(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx512::{MR, NR, gemm_kernel_14x32_f32};

    gemm_driver_f32_mt(a, b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, _bias| unsafe {
        gemm_kernel_14x32_f32(pa, pb, cp, kc, ldc, acc);
    }, std::ptr::null());
}

/// GEMM with fused bias using AVX2: C = A * B + bias (bias fused in microkernel)
#[cfg(target_arch = "x86_64")]
pub fn gemm_bias_asm_f32_avx2(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx2::{MR, NR, gemm_kernel_6x16_f32_bias, gemm_kernel_6x16_f32};

    gemm_driver_f32_mt(a, b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, bias_ptr| unsafe {
        if bias_ptr.is_null() {
            gemm_kernel_6x16_f32(pa, pb, cp, kc, ldc, acc);
        } else {
            gemm_kernel_6x16_f32_bias(pa, pb, cp, kc, ldc, acc, bias_ptr);
        }
    }, bias.as_ptr());
}

/// GEMM with fused bias using AVX-512: C = A * B + bias (bias fused in microkernel)
#[cfg(target_arch = "x86_64")]
pub fn gemm_bias_asm_f32_avx512(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx512::{MR, NR, gemm_kernel_14x32_f32, gemm_kernel_14x32_f32_bias};

    gemm_driver_f32_mt(a, b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, bias_ptr| unsafe {
        if bias_ptr.is_null() {
            gemm_kernel_14x32_f32(pa, pb, cp, kc, ldc, acc);
        } else {
            gemm_kernel_14x32_f32_bias(pa, pb, cp, kc, ldc, acc, bias_ptr);
        }
    }, bias.as_ptr());
}

// ============================================================================
// Pre-packed B support: pack_b + prepacked GEMM drivers
// ============================================================================

/// Pack the entire B matrix [K x N] into KC-blocked, NR-wide strips.
///
/// Layout (KC-blocked):
///   For each KC-block p (p=0, KC, 2*KC, ...):
///     For each NR-wide strip j (j=0, NR, 2*NR, ...):
///       packed[p_block * n_strips * kc_len * NR + strip * kc_len * NR + k_local * NR + j_local]
///         = B[p*KC + k_local, j + j_local]
///
/// This layout ensures that within a KC iteration, all NR-strips for that
/// KC-block are contiguous in memory. The prefetcher sees a linear scan
/// of size (n_strips * kc_len * NR * 4) bytes per KC-block, eliminating
/// the 131KB inter-strip jumps of the old flat layout.
///
/// Total size: n_kc_blocks * n_strips * kc_len * NR + NR (padding).
/// The `kc` parameter must match the driver's KC blocking (from `kernel_config().kc`).
#[cfg(target_arch = "x86_64")]
fn pack_b_full_f32(b: &[f32], n: usize, k: usize, nr: usize, kc: usize) -> Vec<f32> {
    assert!(b.len() >= k * n);
    let n_strips = (n + nr - 1) / nr;
    let n_kc_blocks = (k + kc - 1) / kc;
    // +nr padding: microkernel speculatively loads B[next] before checking loop exit
    let total = n_kc_blocks * n_strips * kc * nr + nr;
    let mut packed = vec![0.0f32; total];

    let b_ptr = b.as_ptr();
    let p_ptr = packed.as_mut_ptr();

    unsafe {
        for kc_block in 0..n_kc_blocks {
            let pc = kc_block * kc;
            let kc_len = kc.min(k - pc); // actual rows in this KC-block

            for strip in 0..n_strips {
                let j = strip * nr;
                let col_rem = nr.min(n - j); // actual columns in this strip

                // Base offset for this (kc_block, strip) tile
                let tile_base = kc_block * n_strips * kc * nr + strip * kc * nr;

                if col_rem == nr {
                    // Full NR-wide strip: copy NR elements per row
                    for k_local in 0..kc_len {
                        let src = b_ptr.add((pc + k_local) * n + j);
                        let dst = p_ptr.add(tile_base + k_local * nr);
                        std::ptr::copy_nonoverlapping(src, dst, nr);
                    }
                    // Zero-pad remaining kc rows (kc_len..kc) — already zero from vec init
                } else {
                    // Remainder strip: copy actual columns, rest already zero
                    for k_local in 0..kc_len {
                        let src = b_ptr.add((pc + k_local) * n + j);
                        let dst = p_ptr.add(tile_base + k_local * nr);
                        std::ptr::copy_nonoverlapping(src, dst, col_rem);
                        // Remaining (nr - col_rem) elements are already 0
                    }
                }
            }
        }
    }

    packed
}

/// Pack B for the AVX2 ASM driver (NR=16).
#[cfg(target_arch = "x86_64")]
pub fn pack_b_asm_f32_avx2(b: &[f32], n: usize, k: usize) -> Vec<f32> {
    use super::gemm_avx2::NR;
    let kc = crate::microarch::kernel_config().kc;
    pack_b_full_f32(b, n, k, NR, kc)
}

/// Pack B for the AVX-512 ASM driver (NR=32).
#[cfg(target_arch = "x86_64")]
pub fn pack_b_asm_f32_avx512(b: &[f32], n: usize, k: usize) -> Vec<f32> {
    use super::gemm_avx512::NR;
    let kc = crate::microarch::kernel_config().kc;
    pack_b_full_f32(b, n, k, NR, kc)
}

/// Single-threaded BLIS-style GEMM driver with pre-packed B.
///
/// C = A * packed_B  (row-major, C is m x n, A is m x k)
///
/// `packed_b` must have been produced by `pack_b_full_f32` with the same NR and KC.
/// The driver skips the pack_b step and indexes directly into the KC-blocked pre-packed buffer.
///
/// KC-blocked B layout: packed_b[kc_block * n_strips * kc * nr + strip * kc * nr + k_local * nr]
/// This gives sequential memory access within each KC iteration (no inter-strip jumps).
#[cfg(target_arch = "x86_64")]
fn gemm_prepacked_driver_f32(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    microkernel: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, bool, *const f32),
    bias: *const f32,
) {
    assert!(a.len() >= m * k);
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let cfg = crate::microarch::kernel_config();
    let mc_max = cfg.mc;
    let nc_max = cfg.nc;

    // Dynamic KC: when M is small (few MC blocks), increase KC to reduce
    // KC-block count and C matrix load/store traffic. Constraint: A panel
    // (MC × KC × 4 bytes) must fit in L2 cache.
    // NOTE: kc_max must match the kc used in pack_b_full_f32 (cfg.kc).
    // Dynamic KC expansion is disabled for prepacked B because the buffer
    // was packed with cfg.kc. Use cfg.kc always.
    let kc_max = cfg.kc;

    // Thread-local aligned pack_a buffer
    thread_local! {
        static PACK_A_PP: std::cell::Cell<AlignedVec<f32>> = std::cell::Cell::new(AlignedVec::new());
    }

    let pack_a_size = mc_max * kc_max;
    let mut pa = PACK_A_PP.with(|c| c.take());
    if pa.capacity() < pack_a_size { pa.reserve(pack_a_size); }
    unsafe { pa.set_len(pack_a_size); }

    let a_ptr = a.as_ptr();
    let c_ptr = c.as_mut_ptr();
    let pb_base = packed_b.as_ptr();

    let n_strips_total = (n + nr - 1) / nr;

    // NC loop
    let mut jc = 0usize;
    while jc < n {
        let nc = nc_max.min(n - jc);
        let strip_start = jc / nr; // First NR-strip in this NC chunk

        // KC loop
        let mut pc = 0usize;
        while pc < k {
            let kc = kc_max.min(k - pc);
            let first_kc = pc == 0;
            let last_kc = pc + kc_max >= k;
            let kc_block_idx = pc / kc_max;

            // MC loop
            let mut ic = 0usize;
            while ic < m {
                let mc = mc_max.min(m - ic);

                // Pack A panel
                unsafe {
                    pack_a_f32(
                        a_ptr.add(ic * k + pc), k,
                        pa.as_mut_ptr(), mc, kc, mr,
                    );
                }

                let n_mr = (mc + mr - 1) / mr;
                let n_nr = (nc + nr - 1) / nr;

                let mut c_tmp = [0.0f32; 14 * 32];

                for jr in 0..n_nr {
                    let col_start = jc + jr * nr;
                    let col_rem = n.saturating_sub(col_start).min(nr);
                    let is_edge_col = col_rem < nr;

                    // KC-blocked B indexing:
                    // tile base = kc_block_idx * n_strips_total * kc_max * nr
                    //           + global_strip * kc_max * nr
                    // Within tile, k_local offset is already baked in (k_local * nr).
                    // The microkernel strides by nr per k step, matching kc_max*nr tile size.
                    let global_strip = strip_start + jr;
                    let pb_tile_base = unsafe {
                        pb_base.add(
                            kc_block_idx * n_strips_total * kc_max * nr
                            + global_strip * kc_max * nr,
                        )
                    };

                    // Opt 4: prefetch next strip's B data into L2 while computing current strip
                    if jr + 1 < n_nr {
                        let next_strip = global_strip + 1;
                        let next_pb = unsafe {
                            pb_base.add(
                                kc_block_idx * n_strips_total * kc_max * nr
                                + next_strip * kc_max * nr,
                            )
                        };
                        // Prefetch kc*nr floats = kc*nr*4 bytes, one cache line (64B) at a time
                        let prefetch_bytes = kc * nr * F32_BYTES;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / F32_BYTES) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T1,
                                );
                            }
                            pf_off += 64;
                        }
                    }

                    for ir in 0..n_mr {
                        let row_start = ic + ir * mr;
                        let row_rem = m.saturating_sub(row_start).min(mr);
                        let is_edge_row = row_rem < mr;

                        unsafe {
                            let pa_tile = pa.as_ptr().add(ir * mr * kc);

                            if is_edge_row || is_edge_col {
                                if !first_kc {
                                    for ri in 0..row_rem {
                                        for ci in 0..col_rem {
                                            c_tmp[ri * nr + ci] =
                                                *c_ptr.add((row_start + ri) * n + col_start + ci);
                                        }
                                    }
                                } else {
                                    for v in c_tmp[..mr * nr].iter_mut() { *v = 0.0; }
                                }

                                microkernel(
                                    pa_tile, pb_tile_base, c_tmp.as_mut_ptr(),
                                    kc, nr, !first_kc, std::ptr::null(),
                                );

                                for ri in 0..row_rem {
                                    for ci in 0..col_rem {
                                        let mut val = c_tmp[ri * nr + ci];
                                        if last_kc && !bias.is_null() {
                                            val += *bias.add(col_start + ci);
                                        }
                                        *c_ptr.add((row_start + ri) * n + col_start + ci) = val;
                                    }
                                }
                            } else {
                                let c_tile = c_ptr.add(row_start * n + col_start);
                                let tile_bias = if last_kc && !bias.is_null() {
                                    bias.add(col_start)
                                } else {
                                    std::ptr::null()
                                };
                                microkernel(
                                    pa_tile, pb_tile_base, c_tile,
                                    kc, n, !first_kc, tile_bias,
                                );
                            }
                        }
                    }
                }

                ic += mc_max;
            }

            pc += kc_max;
        }

        jc += nc_max;
    }

    PACK_A_PP.with(|c| c.set(pa));
}

/// Multi-threaded BLIS-style GEMM driver with pre-packed B.
///
/// Same structure as `gemm_driver_f32_mt` but skips pack_b entirely.
/// Falls back to single-threaded `gemm_prepacked_driver_f32` for small problems.
/// Uses KC-blocked B layout: packed_b[kc_block * n_strips * kc * nr + strip * kc * nr + k_local * nr].
#[cfg(target_arch = "x86_64")]
fn gemm_prepacked_driver_f32_mt(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    microkernel: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, bool, *const f32),
    bias: *const f32,
) {
    assert!(a.len() >= m * k);
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let cfg = crate::microarch::kernel_config();
    let mc_max = cfg.mc;
    let nc_max = cfg.nc;
    // kc_max must match the kc used in pack_b_full_f32 (always cfg.kc for prepacked)
    let kc_max = cfg.kc;

    let nthreads = rayon::current_num_threads().max(1);
    let phys_cores = (nthreads / 2).max(1);

    let mc_max = {
        let num_m_blocks_default = (m + mc_max - 1) / mc_max;
        if num_m_blocks_default < phys_cores && m >= mr * phys_cores {
            (m / phys_cores / mr) * mr
        } else {
            mc_max
        }
    };

    let num_m_blocks = (m + mc_max - 1) / mc_max;
    let n_strips_total = (n + nr - 1) / nr;

    if num_m_blocks < 2 || nthreads <= 1 {
        return gemm_prepacked_driver_f32(a, packed_b, c, m, n, k, mr, nr, microkernel, bias);
    }

    let use_2d = num_m_blocks < phys_cores;

    let a_ptr = a.as_ptr();
    let c_ptr = c.as_mut_ptr();
    let pb_base = packed_b.as_ptr();

    // NC loop
    let mut jc = 0usize;
    while jc < n {
        let nc = nc_max.min(n - jc);
        let strip_start = jc / nr;

        // KC loop
        let mut pc = 0usize;
        while pc < k {
            let kc = kc_max.min(k - pc);
            let first_kc = pc == 0;
            let last_kc = pc + kc_max >= k;
            let kc_block_idx = pc / kc_max;

            let a_addr = a_ptr as usize;
            let c_addr = c_ptr as usize;
            let pb_addr = pb_base as usize;
            let bias_addr = bias as usize;

            use rayon::prelude::*;

            if use_2d {
                let n_nr_blocks = (nc + nr - 1) / nr;
                let target_tiles = phys_cores * 4;
                let nr_per_chunk = (n_nr_blocks * num_m_blocks / target_tiles).max(1);
                let n_chunks = (n_nr_blocks + nr_per_chunk - 1) / nr_per_chunk;
                let total_tiles = num_m_blocks * n_chunks;

                // Pre-pack all A blocks
                let pack_a_size = mc_max * kc;
                let all_pa: Vec<cache_params::AlignedVec<f32>> = (0..num_m_blocks)
                    .map(|block_idx| {
                        let ic = block_idx * mc_max;
                        if ic >= m {
                            return cache_params::AlignedVec::new();
                        }
                        let mc = mc_max.min(m - ic);
                        let mut pa = cache_params::AlignedVec::new();
                        pa.reserve(pack_a_size);
                        unsafe {
                            pa.set_len(pack_a_size);
                            pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc, mr);
                        }
                        pa
                    })
                    .collect();

                let pa_addrs: Vec<usize> = all_pa.iter().map(|pa| pa.as_ptr() as usize).collect();

                (0..total_tiles).into_par_iter().for_each(|tile_idx| {
                    let ir_block = tile_idx / n_chunks;
                    let jc_chunk = tile_idx % n_chunks;

                    let ic = ir_block * mc_max;
                    if ic >= m { return; }
                    let mc = mc_max.min(m - ic);

                    let pa_ptr = pa_addrs[ir_block] as *const f32;
                    let pb_ptr = pb_addr as *const f32;
                    let c_ptr = c_addr as *mut f32;

                    let n_mr = (mc + mr - 1) / mr;
                    let mut c_tmp = [0.0f32; 14 * 32];

                    let jr_start = jc_chunk * nr_per_chunk;
                    let jr_end = (jr_start + nr_per_chunk).min(n_nr_blocks);

                    for jr in jr_start..jr_end {
                        let col_start = jc + jr * nr;
                        let col_rem = n.saturating_sub(col_start).min(nr);
                        if col_rem == 0 { continue; }
                        let is_edge_col = col_rem < nr;

                        let global_strip = strip_start + jr;
                        let pb_tile_base = unsafe {
                            pb_ptr.add(
                                kc_block_idx * n_strips_total * kc_max * nr
                                + global_strip * kc_max * nr,
                            )
                        };

                        // Prefetch next strip's B data
                        if jr + 1 < jr_end {
                            let next_strip = global_strip + 1;
                            let next_pb = unsafe {
                                pb_ptr.add(
                                    kc_block_idx * n_strips_total * kc_max * nr
                                    + next_strip * kc_max * nr,
                                )
                            };
                            let prefetch_bytes = kc * nr * F32_BYTES;
                            let mut pf_off = 0usize;
                            while pf_off < prefetch_bytes {
                                unsafe {
                                    #[cfg(target_arch = "x86_64")]
                                    std::arch::x86_64::_mm_prefetch(
                                        next_pb.add(pf_off / F32_BYTES) as *const i8,
                                        std::arch::x86_64::_MM_HINT_T1,
                                    );
                                }
                                pf_off += 64;
                            }
                        }

                        for ir in 0..n_mr {
                            let row_start = ic + ir * mr;
                            let row_rem = m.saturating_sub(row_start).min(mr);
                            let is_edge_row = row_rem < mr;

                            unsafe {
                                let pa_tile = pa_ptr.add(ir * mr * kc);

                                if is_edge_row || is_edge_col {
                                    if !first_kc {
                                        for ri in 0..row_rem {
                                            for ci in 0..col_rem {
                                                c_tmp[ri * nr + ci] =
                                                    *c_ptr.add((row_start + ri) * n + col_start + ci);
                                            }
                                        }
                                    } else {
                                        for v in c_tmp[..mr * nr].iter_mut() { *v = 0.0; }
                                    }

                                    microkernel(
                                        pa_tile, pb_tile_base, c_tmp.as_mut_ptr(),
                                        kc, nr, !first_kc, std::ptr::null(),
                                    );

                                    let bias_ptr = bias_addr as *const f32;
                                    let fuse_bias = last_kc && !bias_ptr.is_null();

                                    for ri in 0..row_rem {
                                        for ci in 0..col_rem {
                                            let mut val = c_tmp[ri * nr + ci];
                                            if fuse_bias {
                                                val += *bias_ptr.add(col_start + ci);
                                            }
                                            *c_ptr.add((row_start + ri) * n + col_start + ci) = val;
                                        }
                                    }
                                } else {
                                    let c_tile = c_ptr.add(row_start * n + col_start);
                                    let bias_ptr = bias_addr as *const f32;
                                    let tile_bias = if last_kc && !bias_ptr.is_null() {
                                        bias_ptr.add(col_start)
                                    } else {
                                        std::ptr::null()
                                    };
                                    microkernel(
                                        pa_tile, pb_tile_base, c_tile,
                                        kc, n, !first_kc, tile_bias,
                                    );
                                }
                            }
                        }
                    }
                });

                drop(all_pa);
            } else {
            // 1D parallel: parallelize over M-blocks
            (0..num_m_blocks).into_par_iter().for_each(|block_idx| {
                let ic = block_idx * mc_max;
                if ic >= m { return; }
                let mc = mc_max.min(m - ic);

                thread_local! {
                    static PACK_A_PP_MT: std::cell::Cell<AlignedVec<f32>> =
                        std::cell::Cell::new(AlignedVec::new());
                }
                let pack_a_size = mc_max * kc_max;
                let mut pa = PACK_A_PP_MT.with(|c| c.take());
                if pa.capacity() < pack_a_size { pa.reserve(pack_a_size); }
                unsafe { pa.set_len(pack_a_size); }

                let a_ptr = a_addr as *const f32;
                let c_ptr = c_addr as *mut f32;
                let pb_ptr = pb_addr as *const f32;

                unsafe {
                    pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc, mr);
                }

                let n_mr = (mc + mr - 1) / mr;
                let n_nr = (nc + nr - 1) / nr;
                let mut c_tmp = [0.0f32; 14 * 32];

                for jr in 0..n_nr {
                    let col_start = jc + jr * nr;
                    let col_rem = n.saturating_sub(col_start).min(nr);
                    let is_edge_col = col_rem < nr;

                    let global_strip = strip_start + jr;
                    let pb_tile_base = unsafe {
                        pb_ptr.add(
                            kc_block_idx * n_strips_total * kc_max * nr
                            + global_strip * kc_max * nr,
                        )
                    };

                    // Prefetch next strip's B data
                    if jr + 1 < n_nr {
                        let next_strip = global_strip + 1;
                        let next_pb = unsafe {
                            pb_ptr.add(
                                kc_block_idx * n_strips_total * kc_max * nr
                                + next_strip * kc_max * nr,
                            )
                        };
                        let prefetch_bytes = kc * nr * F32_BYTES;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / F32_BYTES) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T1,
                                );
                            }
                            pf_off += 64;
                        }
                    }

                    for ir in 0..n_mr {
                        let row_start = ic + ir * mr;
                        let row_rem = m.saturating_sub(row_start).min(mr);
                        let is_edge_row = row_rem < mr;

                        unsafe {
                            let pa_tile = pa.as_ptr().add(ir * mr * kc);

                            if is_edge_row || is_edge_col {
                                if !first_kc {
                                    for ri in 0..row_rem {
                                        for ci in 0..col_rem {
                                            c_tmp[ri * nr + ci] =
                                                *c_ptr.add((row_start + ri) * n + col_start + ci);
                                        }
                                    }
                                } else {
                                    for v in c_tmp[..mr * nr].iter_mut() { *v = 0.0; }
                                }

                                microkernel(
                                    pa_tile, pb_tile_base, c_tmp.as_mut_ptr(),
                                    kc, nr, !first_kc, std::ptr::null(),
                                );

                                let bias_ptr = bias_addr as *const f32;
                                let fuse_bias = last_kc && !bias_ptr.is_null();

                                for ri in 0..row_rem {
                                    for ci in 0..col_rem {
                                        let mut val = c_tmp[ri * nr + ci];
                                        if fuse_bias {
                                            val += *bias_ptr.add(col_start + ci);
                                        }
                                        *c_ptr.add((row_start + ri) * n + col_start + ci) = val;
                                    }
                                }
                            } else {
                                let c_tile = c_ptr.add(row_start * n + col_start);
                                let bias_ptr = bias_addr as *const f32;
                                let tile_bias = if last_kc && !bias_ptr.is_null() {
                                    bias_ptr.add(col_start)
                                } else {
                                    std::ptr::null()
                                };
                                microkernel(
                                    pa_tile, pb_tile_base, c_tile,
                                    kc, n, !first_kc, tile_bias,
                                );
                            }
                        }
                    }
                }

                PACK_A_PP_MT.with(|c| c.set(pa));
            });
            } // end else (1D parallel)

            pc += kc_max;
        }

        jc += nc_max;
    }
}

/// Prepacked GEMM using AVX2: C = A * packed_B
#[cfg(target_arch = "x86_64")]
pub fn gemm_prepacked_asm_f32_avx2(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx2::{MR, NR, gemm_kernel_6x16_f32};

    gemm_prepacked_driver_f32_mt(a, packed_b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, _bias| unsafe {
        gemm_kernel_6x16_f32(pa, pb, cp, kc, ldc, acc);
    }, std::ptr::null());
}

/// Prepacked GEMM using AVX-512: C = A * packed_B
#[cfg(target_arch = "x86_64")]
pub fn gemm_prepacked_asm_f32_avx512(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx512::{MR, NR, gemm_kernel_14x32_f32};

    gemm_prepacked_driver_f32_mt(a, packed_b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, _bias| unsafe {
        gemm_kernel_14x32_f32(pa, pb, cp, kc, ldc, acc);
    }, std::ptr::null());
}

/// Prepacked GEMM with fused bias using AVX2: C = A * packed_B + bias
#[cfg(target_arch = "x86_64")]
pub fn gemm_bias_prepacked_asm_f32_avx2(
    a: &[f32],
    packed_b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx2::{MR, NR, gemm_kernel_6x16_f32, gemm_kernel_6x16_f32_bias};

    gemm_prepacked_driver_f32_mt(a, packed_b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, bias_ptr| unsafe {
        if bias_ptr.is_null() {
            gemm_kernel_6x16_f32(pa, pb, cp, kc, ldc, acc);
        } else {
            gemm_kernel_6x16_f32_bias(pa, pb, cp, kc, ldc, acc, bias_ptr);
        }
    }, bias.as_ptr());
}

/// Prepacked GEMM with fused bias using AVX-512: C = A * packed_B + bias
#[cfg(target_arch = "x86_64")]
pub fn gemm_bias_prepacked_asm_f32_avx512(
    a: &[f32],
    packed_b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx512::{MR, NR, gemm_kernel_14x32_f32, gemm_kernel_14x32_f32_bias};

    gemm_prepacked_driver_f32_mt(a, packed_b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, bias_ptr| unsafe {
        if bias_ptr.is_null() {
            gemm_kernel_14x32_f32(pa, pb, cp, kc, ldc, acc);
        } else {
            gemm_kernel_14x32_f32_bias(pa, pb, cp, kc, ldc, acc, bias_ptr);
        }
    }, bias.as_ptr());
}

// ============================================================================
// Pre-packed A support: pack_a + SharedPackA + prepacked AB GEMM drivers
// ============================================================================

/// Shared pre-packed A buffer for QKV-style reuse.
///
/// In transformer inference, Q/K/V projections share the same input activation
/// matrix A. Packing A once and reusing it across all three GEMMs saves ~3x
/// pack_a bandwidth.
///
/// Create via `SharedPackA::pack()`, then pass to `gemm_prepacked_ab_asm_f32_*`.
#[cfg(target_arch = "x86_64")]
pub struct SharedPackA {
    packed: Vec<f32>,
    m: usize,
    k: usize,
    mr: usize,
    kc: usize,
}

#[cfg(target_arch = "x86_64")]
impl SharedPackA {
    /// Pack a row-major A matrix [m x k] into KC-blocked, MR-wide strips.
    pub fn pack(a: &[f32], m: usize, k: usize, mr: usize, kc: usize) -> Self {
        let packed = pack_a_full_f32(a, m, k, mr, kc);
        Self { packed, m, k, mr, kc }
    }

    /// Get the packed data slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.packed
    }

    /// Dimensions.
    #[inline]
    pub fn m(&self) -> usize { self.m }
    #[inline]
    pub fn k(&self) -> usize { self.k }
    #[inline]
    pub fn mr(&self) -> usize { self.mr }
    #[inline]
    pub fn kc(&self) -> usize { self.kc }
}

/// Pack the entire A matrix [M x K] into KC-blocked, MR-wide strips.
///
/// Layout (KC-blocked, mirrors `pack_b_full_f32`):
///   For each KC-block p (p=0, KC, 2*KC, ...):
///     For each MR-wide strip i (i=0, MR, 2*MR, ...):
///       packed[kc_block * n_mr_strips * kc * mr + strip * kc * mr + k_local * mr + i_local]
///         = A[i + i_local, p*KC + k_local]
///
/// Total size: n_kc_blocks * n_mr_strips * kc * mr.
#[cfg(target_arch = "x86_64")]
fn pack_a_full_f32(a: &[f32], m: usize, k: usize, mr: usize, kc: usize) -> Vec<f32> {
    assert!(a.len() >= m * k);
    let n_mr_strips = (m + mr - 1) / mr;
    let n_kc_blocks = (k + kc - 1) / kc;
    let total = n_kc_blocks * n_mr_strips * kc * mr;
    let mut packed = vec![0.0f32; total];

    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();

    unsafe {
        for kc_block in 0..n_kc_blocks {
            let pc = kc_block * kc;
            let kc_len = kc.min(k - pc);

            for strip in 0..n_mr_strips {
                let i = strip * mr;
                let mc = mr.min(m - i);

                let tile_base = kc_block * n_mr_strips * kc * mr + strip * kc * mr;

                // Reuse the optimized pack_a_f32 (AVX2 6x8 transpose for MR=6).
                // pack_a_f32 writes mc (up to mr) rows x kc_len cols into
                // packed[k_local * mr + i_local] layout, zero-padding remainder rows.
                // Remaining (kc - kc_len) * mr floats are already zero from vec init.
                pack_a_f32(
                    a_ptr.add(i * k + pc), k,
                    p_ptr.add(tile_base),
                    mc, kc_len, mr,
                );
            }
        }
    }

    packed
}

/// Pack A for the AVX2 ASM driver (MR=6).
#[cfg(target_arch = "x86_64")]
pub fn pack_a_asm_f32_avx2(a: &[f32], m: usize, k: usize) -> SharedPackA {
    use super::gemm_avx2::MR;
    let kc = crate::microarch::kernel_config().kc;
    SharedPackA::pack(a, m, k, MR, kc)
}

/// Pack A for the AVX-512 ASM driver (MR=14).
#[cfg(target_arch = "x86_64")]
pub fn pack_a_asm_f32_avx512(a: &[f32], m: usize, k: usize) -> SharedPackA {
    use super::gemm_avx512::MR;
    let kc = crate::microarch::kernel_config().kc;
    SharedPackA::pack(a, m, k, MR, kc)
}

/// Single-threaded BLIS-style GEMM driver with both pre-packed A and pre-packed B.
///
/// C = packed_A * packed_B  (row-major C, m x n)
///
/// Both `packed_a` and `packed_b` must have been produced by `pack_a_full_f32`
/// and `pack_b_full_f32` respectively, with matching MR/NR/KC parameters.
/// The driver skips all packing and indexes directly into the KC-blocked buffers.
#[cfg(target_arch = "x86_64")]
fn gemm_prepacked_ab_driver_f32(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    microkernel: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, bool, *const f32),
    bias: *const f32,
) {
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let cfg = crate::microarch::kernel_config();
    let kc_max = cfg.kc;
    let nc_max = cfg.nc;

    let c_ptr = c.as_mut_ptr();
    let pa_base = packed_a.as_ptr();
    let pb_base = packed_b.as_ptr();

    let n_mr_strips_total = (m + mr - 1) / mr;
    let n_nr_strips_total = (n + nr - 1) / nr;

    // NC loop
    let mut jc = 0usize;
    while jc < n {
        let nc = nc_max.min(n - jc);
        let strip_start_n = jc / nr;

        // KC loop
        let mut pc = 0usize;
        while pc < k {
            let kc = kc_max.min(k - pc);
            let first_kc = pc == 0;
            let last_kc = pc + kc_max >= k;
            let kc_block_idx = pc / kc_max;

            let n_nr = (nc + nr - 1) / nr;

            // MR-strip loop (no MC blocking needed — A is already packed)
            for mr_strip in 0..n_mr_strips_total {
                let row_start = mr_strip * mr;
                let row_rem = mr.min(m - row_start);
                let is_edge_row = row_rem < mr;

                let pa_strip = unsafe {
                    pa_base.add(
                        kc_block_idx * n_mr_strips_total * kc_max * mr
                        + mr_strip * kc_max * mr,
                    )
                };

                let mut c_tmp = [0.0f32; 14 * 32];

                for jr in 0..n_nr {
                    let col_start = jc + jr * nr;
                    let col_rem = n.saturating_sub(col_start).min(nr);
                    let is_edge_col = col_rem < nr;

                    let global_strip_n = strip_start_n + jr;
                    let pb_tile = unsafe {
                        pb_base.add(
                            kc_block_idx * n_nr_strips_total * kc_max * nr
                            + global_strip_n * kc_max * nr,
                        )
                    };

                    // Prefetch next B strip
                    if jr + 1 < n_nr {
                        let next_strip = global_strip_n + 1;
                        let next_pb = unsafe {
                            pb_base.add(
                                kc_block_idx * n_nr_strips_total * kc_max * nr
                                + next_strip * kc_max * nr,
                            )
                        };
                        let prefetch_bytes = kc * nr * F32_BYTES;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / F32_BYTES) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T1,
                                );
                            }
                            pf_off += 64;
                        }
                    }

                    unsafe {
                        if is_edge_row || is_edge_col {
                            if !first_kc {
                                for ri in 0..row_rem {
                                    for ci in 0..col_rem {
                                        c_tmp[ri * nr + ci] =
                                            *c_ptr.add((row_start + ri) * n + col_start + ci);
                                    }
                                }
                            } else {
                                for v in c_tmp[..mr * nr].iter_mut() { *v = 0.0; }
                            }

                            microkernel(
                                pa_strip, pb_tile, c_tmp.as_mut_ptr(),
                                kc, nr, !first_kc, std::ptr::null(),
                            );

                            for ri in 0..row_rem {
                                for ci in 0..col_rem {
                                    let mut val = c_tmp[ri * nr + ci];
                                    if last_kc && !bias.is_null() {
                                        val += *bias.add(col_start + ci);
                                    }
                                    *c_ptr.add((row_start + ri) * n + col_start + ci) = val;
                                }
                            }
                        } else {
                            let c_tile = c_ptr.add(row_start * n + col_start);
                            let tile_bias = if last_kc && !bias.is_null() {
                                bias.add(col_start)
                            } else {
                                std::ptr::null()
                            };
                            microkernel(
                                pa_strip, pb_tile, c_tile,
                                kc, n, !first_kc, tile_bias,
                            );
                        }
                    }
                }
            }

            pc += kc_max;
        }

        jc += nc_max;
    }
}

/// Multi-threaded GEMM driver with both pre-packed A and pre-packed B.
///
/// Since both A and B are pre-packed, no per-thread packing buffers are needed.
/// Parallelizes directly over MR-strips (1D) or (MR-strip, NR-chunk) tiles (2D).
#[cfg(target_arch = "x86_64")]
fn gemm_prepacked_ab_driver_f32_mt(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    microkernel: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, bool, *const f32),
    bias: *const f32,
) {
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let cfg = crate::microarch::kernel_config();
    let kc_max = cfg.kc;
    let nc_max = cfg.nc;

    let nthreads = rayon::current_num_threads().max(1);
    let n_mr_strips_total = (m + mr - 1) / mr;
    let n_nr_strips_total = (n + nr - 1) / nr;

    // Fall back to single-threaded for small problems
    if n_mr_strips_total < 2 || nthreads <= 1 {
        return gemm_prepacked_ab_driver_f32(
            packed_a, packed_b, c, m, n, k, mr, nr, microkernel, bias,
        );
    }

    let pa_base = packed_a.as_ptr();
    let pb_base = packed_b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // NC loop
    let mut jc = 0usize;
    while jc < n {
        let nc = nc_max.min(n - jc);
        let strip_start_n = jc / nr;

        // KC loop
        let mut pc = 0usize;
        while pc < k {
            let kc = kc_max.min(k - pc);
            let first_kc = pc == 0;
            let last_kc = pc + kc_max >= k;
            let kc_block_idx = pc / kc_max;

            let n_nr = (nc + nr - 1) / nr;

            let pa_addr = pa_base as usize;
            let pb_addr = pb_base as usize;
            let c_addr = c_ptr as usize;
            let bias_addr = bias as usize;

            use rayon::prelude::*;

            // Parallel over MR-strips — no packing buffers needed
            (0..n_mr_strips_total).into_par_iter().for_each(|mr_strip| {
                let row_start = mr_strip * mr;
                let row_rem = mr.min(m - row_start);
                let is_edge_row = row_rem < mr;

                let pa_ptr = pa_addr as *const f32;
                let pb_ptr = pb_addr as *const f32;
                let c_ptr = c_addr as *mut f32;

                let pa_strip = unsafe {
                    pa_ptr.add(
                        kc_block_idx * n_mr_strips_total * kc_max * mr
                        + mr_strip * kc_max * mr,
                    )
                };

                let mut c_tmp = [0.0f32; 14 * 32];

                for jr in 0..n_nr {
                    let col_start = jc + jr * nr;
                    let col_rem = n.saturating_sub(col_start).min(nr);
                    if col_rem == 0 { continue; }
                    let is_edge_col = col_rem < nr;

                    let global_strip_n = strip_start_n + jr;
                    let pb_tile = unsafe {
                        pb_ptr.add(
                            kc_block_idx * n_nr_strips_total * kc_max * nr
                            + global_strip_n * kc_max * nr,
                        )
                    };

                    // Prefetch next B strip
                    if jr + 1 < n_nr {
                        let next_strip = global_strip_n + 1;
                        let next_pb = unsafe {
                            pb_ptr.add(
                                kc_block_idx * n_nr_strips_total * kc_max * nr
                                + next_strip * kc_max * nr,
                            )
                        };
                        let prefetch_bytes = kc * nr * F32_BYTES;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / F32_BYTES) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T1,
                                );
                            }
                            pf_off += 64;
                        }
                    }

                    unsafe {
                        if is_edge_row || is_edge_col {
                            if !first_kc {
                                for ri in 0..row_rem {
                                    for ci in 0..col_rem {
                                        c_tmp[ri * nr + ci] =
                                            *c_ptr.add((row_start + ri) * n + col_start + ci);
                                    }
                                }
                            } else {
                                for v in c_tmp[..mr * nr].iter_mut() { *v = 0.0; }
                            }

                            microkernel(
                                pa_strip, pb_tile, c_tmp.as_mut_ptr(),
                                kc, nr, !first_kc, std::ptr::null(),
                            );

                            let bias_ptr = bias_addr as *const f32;
                            let fuse_bias = last_kc && !bias_ptr.is_null();

                            for ri in 0..row_rem {
                                for ci in 0..col_rem {
                                    let mut val = c_tmp[ri * nr + ci];
                                    if fuse_bias {
                                        val += *bias_ptr.add(col_start + ci);
                                    }
                                    *c_ptr.add((row_start + ri) * n + col_start + ci) = val;
                                }
                            }
                        } else {
                            let c_tile = c_ptr.add(row_start * n + col_start);
                            let bias_ptr = bias_addr as *const f32;
                            let tile_bias = if last_kc && !bias_ptr.is_null() {
                                bias_ptr.add(col_start)
                            } else {
                                std::ptr::null()
                            };
                            microkernel(
                                pa_strip, pb_tile, c_tile,
                                kc, n, !first_kc, tile_bias,
                            );
                        }
                    }
                }
            });

            pc += kc_max;
        }

        jc += nc_max;
    }
}

/// Prepacked AB GEMM using AVX2: C = packed_A * packed_B
///
/// Both A and B must be pre-packed via `pack_a_asm_f32_avx2` / `pack_b_asm_f32_avx2`.
/// Use this for QKV-style shared-A reuse: pack A once, call this 3x with different packed_B.
#[cfg(target_arch = "x86_64")]
pub fn gemm_prepacked_ab_asm_f32_avx2(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx2::{MR, NR, gemm_kernel_6x16_f32};

    gemm_prepacked_ab_driver_f32_mt(packed_a, packed_b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, _bias| unsafe {
        gemm_kernel_6x16_f32(pa, pb, cp, kc, ldc, acc);
    }, std::ptr::null());
}

/// Prepacked AB GEMM using AVX-512: C = packed_A * packed_B
#[cfg(target_arch = "x86_64")]
pub fn gemm_prepacked_ab_asm_f32_avx512(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use super::gemm_avx512::{MR, NR, gemm_kernel_14x32_f32};

    gemm_prepacked_ab_driver_f32_mt(packed_a, packed_b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc, _bias| unsafe {
        gemm_kernel_14x32_f32(pa, pb, cp, kc, ldc, acc);
    }, std::ptr::null());
}
