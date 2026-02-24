//! NEON GEMM driver using the hand-written 8x12 assembly microkernel.
//!
//! This module provides the complete GEMM implementation:
//! - Pack A into column-major MR-wide panels (r-inner / k-outer order)
//! - Pack B into row-major NR-wide panels (KC-blocked full-matrix prepacking)
//! - Tile the M/N/K dimensions with cache-aware blocking
//! - Call the assembly microkernel for each tile
//! - Multi-threaded MC loop via Rayon
//!
//! Two entry points:
//! - `gemm_asm_f32`:           standard GEMM (packs B on the fly)
//! - `gemm_prepacked_asm_f32`: GEMM with pre-packed B (skips pack_b step)
//!
//! Pre-packing helpers:
//! - `pack_b_asm_f32_neon`:    pack the full B matrix for repeated use

#[cfg(target_arch = "aarch64")]
use crate::asm::aarch64::{MR, NR, gemm_kernel_8x12_f32};
#[cfg(target_arch = "aarch64")]
use crate::cache_params;

/// Pack a panel of A (mc x kc) into column-major MR-wide strips.
///
/// Input A is row-major: a[i*lda + k]
/// Output: packed[k*MR + i_local] for each MR-wide strip
///
/// This layout ensures the microkernel can load MR consecutive A elements
/// with a single `ldp` instruction pair.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pack_a_f32(
    a: *const f32,
    lda: usize,
    packed: *mut f32,
    mc: usize,
    kc: usize,
) {
    let mut p = packed;
    let mut i = 0usize;

    // Full MR-wide panels
    while i + MR <= mc {
        for k in 0..kc {
            // Load MR elements from column k of the A panel
            for r in 0..MR {
                *p.add(r) = *a.add((i + r) * lda + k);
            }
            p = p.add(MR);
        }
        i += MR;
    }

    // Remainder rows (< MR): zero-pad to MR width
    if i < mc {
        let rem = mc - i;
        for k in 0..kc {
            for r in 0..rem {
                *p.add(r) = *a.add((i + r) * lda + k);
            }
            for r in rem..MR {
                *p.add(r) = 0.0;
            }
            p = p.add(MR);
        }
    }
}

/// Pack a panel of B (kc x nc) into row-major NR-wide strips.
///
/// Input B is row-major: b[k*ldb + j]
/// Output: packed[k*NR + j_local] for each NR-wide strip
///
/// This layout ensures the microkernel can load NR consecutive B elements
/// with `ldp` + `ldr` (48 bytes = 12 floats).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pack_b_f32(
    b: *const f32,
    ldb: usize,
    packed: *mut f32,
    kc: usize,
    nc: usize,
) {
    let mut p = packed;
    let mut j = 0usize;

    // Full NR-wide panels
    while j + NR <= nc {
        for k in 0..kc {
            let src = b.add(k * ldb + j);
            // Copy NR elements: 3 vectors of 4 floats
            std::ptr::copy_nonoverlapping(src, p, NR);
            p = p.add(NR);
        }
        j += NR;
    }

    // Remainder columns (< NR): zero-pad to NR width
    if j < nc {
        let rem = nc - j;
        for k in 0..kc {
            let src = b.add(k * ldb + j);
            std::ptr::copy_nonoverlapping(src, p, rem);
            for c in rem..NR {
                *p.add(c) = 0.0;
            }
            p = p.add(NR);
        }
    }
}

/// Inner microkernel loop over MR×NR tiles for a single MC×NC block.
///
/// Shared by both single-threaded and multi-threaded paths.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn compute_mc_block(
    pa: *const f32,
    pb: *const f32,
    c_ptr: *mut f32,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    n: usize,
    first_kc: bool,
) {
    let n_mr = (mc + MR - 1) / MR;
    let n_nr = (nc + NR - 1) / NR;
    let mut tmp = [0.0f32; 8 * 12]; // MR * NR

    for jr in 0..n_nr {
        let nc_cur = NR.min(nc - jr * NR);

        for ir in 0..n_mr {
            let mc_cur = MR.min(mc - ir * MR);

            let pa_tile = pa.add(ir * MR * kc);
            let pb_tile = pb.add(jr * NR * kc);

            if mc_cur == MR && nc_cur == NR {
                let c_tile = c_ptr.add((ic + ir * MR) * n + (jc + jr * NR));
                gemm_kernel_8x12_f32(pa_tile, pb_tile, c_tile, kc, n, !first_kc);
            } else {
                if !first_kc {
                    for r in 0..mc_cur {
                        for col in 0..nc_cur {
                            tmp[r * NR + col] =
                                *c_ptr.add((ic + ir * MR + r) * n + (jc + jr * NR + col));
                        }
                    }
                } else {
                    tmp.fill(0.0);
                }

                gemm_kernel_8x12_f32(pa_tile, pb_tile, tmp.as_mut_ptr(), kc, NR, !first_kc);

                for r in 0..mc_cur {
                    for col in 0..nc_cur {
                        *c_ptr.add((ic + ir * MR + r) * n + (jc + jr * NR + col)) =
                            tmp[r * NR + col];
                    }
                }
            }
        }
    }
}

/// Full GEMM using the hand-written 8x12 NEON microkernel.
///
/// C = A * B  (row-major, C is m x n, A is m x k, B is k x n)
///
/// Uses BLIS-style 3-level blocking:
/// - NC blocking on N (B panel fits in L3)
/// - KC blocking on K (A panel fits in L2, B strip fits in L1)
/// - MC blocking on M (A panel fits in L2)
///
/// Multi-threaded: MC loop is parallelized via Rayon when there are
/// enough M-blocks to fill available cores. B is packed once on the
/// main thread and shared read-only; each worker packs its own A block.
#[cfg(target_arch = "aarch64")]
pub fn gemm_asm_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let bp = cache_params::blocking_params(MR, 3, 4, 4);
    let kc_max = bp.kc;
    let mc_max = bp.mc;
    let nc_max = bp.nc;

    let nthreads = rayon::current_num_threads().max(1);
    let num_m_blocks = (m + mc_max - 1) / mc_max;
    let use_mt = num_m_blocks >= 2 && nthreads > 1;

    // Shared pack_b buffer (main thread packs, workers read)
    thread_local! {
        static PACK_B: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }

    let pack_b_size = kc_max * nc_max;

    PACK_B.with(|pb_cell| {
        let mut pb = pb_cell.borrow_mut();
        if pb.len() < pack_b_size {
            pb.resize(pack_b_size, 0.0);
        }

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        // NC loop
        let mut jc = 0usize;
        while jc < n {
            let nc = nc_max.min(n - jc);

            // KC loop
            let mut pc = 0usize;
            while pc < k {
                let kc = kc_max.min(k - pc);
                let first_kc = pc == 0;

                // Pack B once on main thread
                unsafe {
                    pack_b_f32(b_ptr.add(pc * n + jc), n, pb.as_mut_ptr(), kc, nc);
                }

                if use_mt {
                    // Multi-threaded MC loop
                    let pb_addr = pb.as_ptr() as usize;
                    let a_addr = a_ptr as usize;
                    let c_addr = c_ptr as usize;

                    use rayon::prelude::*;
                    (0..num_m_blocks).into_par_iter().for_each(|block_idx| {
                        let ic = block_idx * mc_max;
                        if ic >= m { return; }
                        let mc = mc_max.min(m - ic);

                        // Thread-local pack_a
                        thread_local! {
                            static PACK_A_MT: std::cell::RefCell<Vec<f32>> =
                                std::cell::RefCell::new(Vec::new());
                        }
                        let pack_a_size = mc_max * kc_max;
                        PACK_A_MT.with(|pa_cell| {
                            let mut pa = pa_cell.borrow_mut();
                            if pa.len() < pack_a_size {
                                pa.resize(pack_a_size, 0.0);
                            }

                            let a_ptr = a_addr as *const f32;
                            let c_ptr = c_addr as *mut f32;
                            let pb_ptr = pb_addr as *const f32;

                            unsafe {
                                pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc);
                                compute_mc_block(
                                    pa.as_ptr(), pb_ptr, c_ptr,
                                    ic, jc, mc, nc, kc, n, first_kc,
                                );
                            }
                        });
                    });
                } else {
                    // Single-threaded MC loop
                    thread_local! {
                        static PACK_A: std::cell::RefCell<Vec<f32>> =
                            std::cell::RefCell::new(Vec::new());
                    }
                    let pack_a_size = mc_max * kc_max;

                    PACK_A.with(|pa_cell| {
                        let mut pa = pa_cell.borrow_mut();
                        if pa.len() < pack_a_size {
                            pa.resize(pack_a_size, 0.0);
                        }

                        let mut ic = 0usize;
                        while ic < m {
                            let mc = mc_max.min(m - ic);

                            unsafe {
                                pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc);
                                compute_mc_block(
                                    pa.as_ptr(), pb.as_ptr(), c_ptr,
                                    ic, jc, mc, nc, kc, n, first_kc,
                                );
                            }

                            ic += mc_max;
                        }
                    });
                }

                pc += kc_max;
            }

            jc += nc_max;
        }
    });
}

/// GEMM with bias: C = A * B + bias
#[cfg(target_arch = "aarch64")]
pub fn gemm_bias_asm_f32(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // First compute C = A * B
    gemm_asm_f32(a, b, c, m, n, k);

    // Then add bias to each row
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
}

/// Pack the full B matrix (k x n) into NEON ASM tile format for repeated use.
///
/// Layout: for each NR-wide strip j, data is stored as k*NR contiguous floats.
/// i.e. packed[j_strip * NR * k + kk * NR .. + NR] = B[kk, j_strip*NR .. j_strip*NR+NR]
#[cfg(target_arch = "aarch64")]
pub fn pack_b_asm_f32_neon(b: &[f32], n: usize, k: usize) -> Vec<f32> {
    let n_nr = (n + NR - 1) / NR;
    let packed_size = n_nr * NR * k;
    let mut packed = vec![0.0f32; packed_size];
    unsafe {
        pack_b_f32(b.as_ptr(), n, packed.as_mut_ptr(), k, n);
    }
    packed
}

/// GEMM with pre-packed B: C = A * packed_B
///
/// packed_B must have been produced by `pack_b_asm_f32_neon`.
/// Layout: packed_b[j_strip * NR * k + kk * NR + j_local]
///
/// Multi-threaded: MC loop is parallelized via Rayon.
#[cfg(target_arch = "aarch64")]
pub fn gemm_prepacked_asm_f32(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.len() >= m * k);
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let bp = cache_params::blocking_params(MR, 3, 4, 4);
    let kc_max = bp.kc;
    let mc_max = bp.mc;

    let nthreads = rayon::current_num_threads().max(1);
    let num_m_blocks = (m + mc_max - 1) / mc_max;
    let use_mt = num_m_blocks >= 2 && nthreads > 1;
    let n_nr = (n + NR - 1) / NR;

    let a_ptr = a.as_ptr();
    let c_ptr = c.as_mut_ptr();
    let pb_ptr = packed_b.as_ptr();

    // KC loop
    let mut pc = 0usize;
    while pc < k {
        let kc = kc_max.min(k - pc);
        let first_kc = pc == 0;

        if use_mt {
            let a_addr = a_ptr as usize;
            let c_addr = c_ptr as usize;
            let pb_addr = pb_ptr as usize;

            use rayon::prelude::*;
            (0..num_m_blocks).into_par_iter().for_each(|block_idx| {
                let ic = block_idx * mc_max;
                if ic >= m { return; }
                let mc = mc_max.min(m - ic);

                thread_local! {
                    static PACK_A_PP_MT: std::cell::RefCell<Vec<f32>> =
                        std::cell::RefCell::new(Vec::new());
                }
                let pack_a_size = mc_max * kc_max;
                PACK_A_PP_MT.with(|pa_cell| {
                    let mut pa = pa_cell.borrow_mut();
                    if pa.len() < pack_a_size {
                        pa.resize(pack_a_size, 0.0);
                    }

                    let a_ptr = a_addr as *const f32;
                    let c_ptr = c_addr as *mut f32;
                    let pb_ptr = pb_addr as *const f32;

                    unsafe {
                        pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc);
                        compute_mc_block_prepacked(
                            pa.as_ptr(), pb_ptr, c_ptr,
                            ic, mc, n, k, kc, pc, n_nr, first_kc,
                        );
                    }
                });
            });
        } else {
            // Single-threaded MC loop
            thread_local! {
                static PACK_A_PP: std::cell::RefCell<Vec<f32>> =
                    std::cell::RefCell::new(Vec::new());
            }
            let pack_a_size = mc_max * kc_max;

            PACK_A_PP.with(|pa_cell| {
                let mut pa = pa_cell.borrow_mut();
                if pa.len() < pack_a_size {
                    pa.resize(pack_a_size, 0.0);
                }

                let mut ic = 0usize;
                while ic < m {
                    let mc = mc_max.min(m - ic);

                    unsafe {
                        pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc);
                        compute_mc_block_prepacked(
                            pa.as_ptr(), pb_ptr, c_ptr,
                            ic, mc, n, k, kc, pc, n_nr, first_kc,
                        );
                    }

                    ic += mc_max;
                }
            });
        }

        pc += kc_max;
    }
}

/// Inner microkernel loop for prepacked-B path.
///
/// B offset: `jr * NR * k + pc * NR` (full-K prepacked layout).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn compute_mc_block_prepacked(
    pa: *const f32,
    pb_ptr: *const f32,
    c_ptr: *mut f32,
    ic: usize,
    mc: usize,
    n: usize,
    k: usize,
    kc: usize,
    pc: usize,
    n_nr: usize,
    first_kc: bool,
) {
    let n_mr = (mc + MR - 1) / MR;
    let mut tmp = [0.0f32; 8 * 12]; // MR * NR

    for jr in 0..n_nr {
        let nc_cur = NR.min(n.saturating_sub(jr * NR));
        if nc_cur == 0 { continue; }

        for ir in 0..n_mr {
            let mc_cur = MR.min(mc - ir * MR);

            let pa_tile = pa.add(ir * MR * kc);
            let pb_tile = pb_ptr.add(jr * NR * k + pc * NR);

            if mc_cur == MR && nc_cur == NR {
                let c_tile = c_ptr.add((ic + ir * MR) * n + jr * NR);
                gemm_kernel_8x12_f32(pa_tile, pb_tile, c_tile, kc, n, !first_kc);
            } else {
                if !first_kc {
                    for r in 0..mc_cur {
                        for col in 0..nc_cur {
                            tmp[r * NR + col] =
                                *c_ptr.add((ic + ir * MR + r) * n + jr * NR + col);
                        }
                    }
                } else {
                    tmp.fill(0.0);
                }

                gemm_kernel_8x12_f32(pa_tile, pb_tile, tmp.as_mut_ptr(), kc, NR, !first_kc);

                for r in 0..mc_cur {
                    for col in 0..nc_cur {
                        *c_ptr.add((ic + ir * MR + r) * n + jr * NR + col) =
                            tmp[r * NR + col];
                    }
                }
            }
        }
    }
}

/// GEMM with pre-packed B and bias: C = A * packed_B + bias
#[cfg(target_arch = "aarch64")]
pub fn gemm_bias_prepacked_asm_f32(
    a: &[f32],
    packed_b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    gemm_prepacked_asm_f32(a, packed_b, c, m, n, k);
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
}

/// C-ABI wrapper around [`pack_a_f32`] for JIT-emitted GEMM code.
///
/// Signature matches the x86_64 `gllm_pack_a_f32` so the JIT compiler
/// can use the same calling convention on both architectures.
/// The `_mr` parameter is accepted for API compatibility but ignored
/// (the NEON packer always uses MR=8).
#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub unsafe extern "C" fn gllm_pack_a_f32_neon(
    a: *const f32,
    lda: usize,
    packed: *mut f32,
    mc: usize,
    kc: usize,
    _mr: usize,
) {
    pack_a_f32(a, lda, packed, mc, kc);
}

/// C-ABI wrapper around [`pack_b_f32`] for JIT-emitted GEMM code.
///
/// The `_nr` parameter is accepted for API compatibility but ignored
/// (the NEON packer always uses NR=12).
#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub unsafe extern "C" fn gllm_pack_b_f32_neon(
    b: *const f32,
    ldb: usize,
    packed: *mut f32,
    kc: usize,
    nc: usize,
    _nr: usize,
) {
    pack_b_f32(b, ldb, packed, kc, nc);
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Naive reference GEMM for correctness checking.
    fn gemm_ref(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn check_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(
                diff / scale < tol,
                "{label}[{i}]: asm={x}, ref={y}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_asm_gemm_exact_tile() {
        // Exact MR x NR tile: 8x12 with k=16
        let (m, n, k) = (8, 12, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();

        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);

        check_close(&c_asm, &c_ref, 1e-4, "exact_tile");
    }

    #[test]
    fn test_asm_gemm_multi_tile() {
        // Multiple tiles: 32x48 with k=64
        let (m, n, k) = (32, 48, 64);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();

        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);

        check_close(&c_asm, &c_ref, 1e-4, "multi_tile");
    }

    #[test]
    fn test_asm_gemm_non_aligned() {
        // Non-aligned dimensions: m=7, n=11, k=13
        // Tests edge-tile handling with zero-padding
        let (m, n, k) = (7, 11, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();

        // Allocate C with padding for edge tiles (MR x NR overshoot)
        let m_padded = ((m + MR - 1) / MR) * MR;
        let n_padded = ((n + NR - 1) / NR) * NR;
        let mut c_asm = vec![0.0f32; m_padded * n_padded.max(n)];
        let mut c_ref = vec![0.0f32; m * n];

        // Note: for non-aligned dims, the asm driver writes to padded C.
        // We only check the valid m x n region.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);

        for i in 0..m {
            for j in 0..n {
                let asm_val = c_asm[i * n + j];
                let ref_val = c_ref[i * n + j];
                let diff = (asm_val - ref_val).abs();
                let scale = asm_val.abs().max(ref_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "non_aligned[{i},{j}]: asm={asm_val}, ref={ref_val}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_asm_gemm_identity() {
        // A = I (identity), C should equal B
        let n = 12;
        let k = 12;
        let m = 8;
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];

        gemm_asm_f32(&a, &b, &mut c, m, n, k);

        for i in 0..m {
            for j in 0..n {
                let expected = b[i * n + j]; // since A is identity for first m rows
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "identity[{i},{j}]: got={}, expected={expected}", c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_asm_gemm_bias() {
        let (m, n, k) = (8, 12, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();

        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        gemm_bias_asm_f32(&a, &b, &bias, &mut c_asm, m, n, k);

        // Reference: naive gemm + bias
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }

        check_close(&c_asm, &c_ref, 1e-4, "bias");
    }
}
