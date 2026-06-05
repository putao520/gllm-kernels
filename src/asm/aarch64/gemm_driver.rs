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

    // ---- 13 new tests below ----

    #[test]
    fn test_mr_nr_constants() {
        // Arrange: MR and NR define the microkernel tile geometry.
        // Act: read the constants directly.
        let mr = MR;
        let nr = NR;
        // Assert: NEON microkernel is 8x12.
        assert_eq!(mr, 8, "MR must be 8 for NEON 8x12 microkernel");
        assert_eq!(nr, 12, "NR must be 12 for NEON 8x12 microkernel");
        // MR * NR = 96 accumulator elements fits in 24 NEON Q registers.
        assert_eq!(mr * nr, 96, "MR*NR must equal 96 (24 Q-registers x 4 lanes)");
    }

    #[test]
    fn test_pack_b_exact_tile() {
        // Arrange: B is kc x nc where nc is an exact multiple of NR=12.
        let (kc, nc) = (4, 12);
        let b: Vec<f32> = (0..kc * nc).map(|i| i as f32).collect();
        // Act: pack the full B matrix.
        let packed = pack_b_asm_f32_neon(&b, nc, kc);
        // Assert: packed size is n_nr * NR * kc = 1 * 12 * 4 = 48.
        assert_eq!(packed.len(), 48, "packed size for exact tile");
        // Layout is row-major within NR-wide strip: packed[k*NR + j_local].
        for k in 0..kc {
            for j in 0..nc {
                assert_eq!(
                    packed[k * NR + j], b[k * nc + j],
                    "packed[{}*NR+{}]={}, expected b[{}*nc+{}]={}",
                    k, j, packed[k * NR + j], k, j, b[k * nc + j]
                );
            }
        }
    }

    #[test]
    fn test_pack_b_remainder_columns() {
        // Arrange: nc=7 < NR=12, so 5 zero-padded columns per strip.
        let (kc, nc) = (3, 7);
        let b: Vec<f32> = (0..kc * nc).map(|i| (i as f32) * 0.1).collect();
        // Act.
        let packed = pack_b_asm_f32_neon(&b, nc, kc);
        // Assert: 1 strip x NR x kc = 12 * 3 = 36 total.
        assert_eq!(packed.len(), 36, "packed size for remainder columns");
        for k in 0..kc {
            // First 7 values match B.
            for j in 0..nc {
                assert!(
                    (packed[k * NR + j] - b[k * nc + j]).abs() < 1e-7,
                    "packed[{}*NR+{}]={}, expected b[{}*nc+{}]={}",
                    k, j, packed[k * NR + j], k, j, b[k * nc + j]
                );
            }
            // Remaining 5 are zero-padded.
            for j in nc..NR {
                assert_eq!(
                    packed[k * NR + j], 0.0,
                    "zero-pad at packed[{}*NR+{}]", k, j
                );
            }
        }
    }

    #[test]
    fn test_gemm_prepacked_matches_standard() {
        // Arrange: compute C = A*B via both standard and prepacked paths.
        let (m, n, k) = (16, 24, 32);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32 - 15.0) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 37) as f32 - 18.0) * 0.04).collect();
        let mut c_std = vec![0.0f32; m * n];
        let mut c_pp = vec![0.0f32; m * n];
        // Act: standard GEMM.
        gemm_asm_f32(&a, &b, &mut c_std, m, n, k);
        // Act: prepacked B then GEMM.
        let packed_b = pack_b_asm_f32_neon(&b, n, k);
        gemm_prepacked_asm_f32(&a, &packed_b, &mut c_pp, m, n, k);
        // Assert: both paths must agree.
        check_close(&c_std, &c_pp, 1e-5, "prepacked vs standard");
    }

    #[test]
    fn test_gemm_bias_prepacked_correctness() {
        // Arrange: C = A * packed_B + bias.
        let (m, n, k) = (8, 12, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7).collect();
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        // Act.
        let packed_b = pack_b_asm_f32_neon(&b, n, k);
        gemm_bias_prepacked_asm_f32(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        // Reference.
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        // Assert.
        check_close(&c_pp, &c_ref, 1e-4, "bias_prepacked");
    }

    #[test]
    fn test_gemm_zero_dimension_m() {
        // Arrange: m=0 should produce an all-zero C and not panic.
        let (m, n, k) = (0, 12, 4);
        let a: Vec<f32> = Vec::new();
        let b: Vec<f32> = vec![1.0; k * n];
        let mut c = vec![0.0f32; m.max(1) * n]; // at least 1 row to observe
        let c_before = c.clone();
        // Act.
        gemm_asm_f32(&a, &b, &mut c, m, n, k);
        // Assert: C unchanged (early return).
        assert_eq!(c, c_before, "C must be unchanged when m=0");
    }

    #[test]
    fn test_gemm_unit_dimensions() {
        // Arrange: 1x1 GEMM with k=1 → C[0] = A[0]*B[0].
        let (m, n, k) = (1, 1, 1);
        let a = vec![3.0f32];
        let b = vec![7.0f32];
        let mut c = vec![0.0f32; 1];
        // Act.
        gemm_asm_f32(&a, &b, &mut c, m, n, k);
        // Assert.
        let expected = 3.0f32 * 7.0f32;
        assert!(
            (c[0] - expected).abs() < 1e-5,
            "unit gemm: got {}, expected {}", c[0], expected
        );
    }

    #[test]
    fn test_gemm_float_small_values() {
        // Arrange: very small values to verify no catastrophic cancellation.
        let (m, n, k) = (8, 12, 4);
        let a: Vec<f32> = (0..m * k).map(|i| 1e-6 * (i as f32)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| 1e-7 * (i as f32)).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        // Act.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        // Assert: relative tolerance adapted for small magnitudes.
        check_close(&c_asm, &c_ref, 1e-3, "small_values");
    }

    #[test]
    fn test_gemm_float_large_values() {
        // Arrange: large values to verify no overflow in intermediate sums.
        let (m, n, k) = (8, 12, 4);
        let a: Vec<f32> = (0..m * k).map(|i| 1e3 * ((i % 7) as f32 - 3.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| 1e3 * ((i % 5) as f32 - 2.0)).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        // Act.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        // Assert.
        check_close(&c_asm, &c_ref, 1e-4, "large_values");
    }

    #[test]
    fn test_blocking_params_debug_format() {
        // Arrange: create a BlockingParams via blocking_params().
        let bp = crate::cache_params::blocking_params(MR, 3, 4, 4);
        // Act: format with Debug.
        let debug_str = format!("{:?}", bp);
        // Assert: must contain field names.
        assert!(
            debug_str.contains("kc"),
            "Debug output must contain 'kc': got {}", debug_str
        );
        assert!(
            debug_str.contains("mc"),
            "Debug output must contain 'mc': got {}", debug_str
        );
        assert!(
            debug_str.contains("nc"),
            "Debug output must contain 'nc': got {}", debug_str
        );
    }

    #[test]
    fn test_blocking_params_clone_copy() {
        // Arrange.
        let bp = crate::cache_params::blocking_params(MR, 3, 4, 4);
        // Act: Copy trait (implicit on assignment).
        let bp_copy = bp;
        // Act: Clone trait.
        let bp_clone = bp.clone();
        // Assert: all three are bitwise equal.
        assert_eq!(bp_copy.kc, bp.kc, "Copy: kc mismatch");
        assert_eq!(bp_copy.mc, bp.mc, "Copy: mc mismatch");
        assert_eq!(bp_copy.nc, bp.nc, "Copy: nc mismatch");
        assert_eq!(bp_clone.kc, bp.kc, "Clone: kc mismatch");
        assert_eq!(bp_clone.mc, bp.mc, "Clone: mc mismatch");
        assert_eq!(bp_clone.nc, bp.nc, "Clone: nc mismatch");
    }

    #[test]
    fn test_blocking_params_struct_update_syntax() {
        // Arrange.
        let bp = crate::cache_params::blocking_params(MR, 3, 4, 4);
        // Act: use struct update syntax to override one field.
        let bp_custom = crate::cache_params::BlockingParams {
            kc: 128,
            ..bp
        };
        // Assert: overridden field is 128, others inherited.
        assert_eq!(bp_custom.kc, 128, "overridden kc must be 128");
        assert_eq!(bp_custom.mc, bp.mc, "mc must be inherited from bp");
        assert_eq!(bp_custom.nc, bp.nc, "nc must be inherited from bp");
    }

    #[test]
    fn test_blocking_params_field_values_bounded() {
        // Arrange: compute blocking params for the NEON 8x12 microkernel.
        let bp = crate::cache_params::blocking_params(MR, 3, 4, 4);
        // Assert: kc is multiple of 8 and within [64, 768].
        assert_eq!(bp.kc % 8, 0, "kc must be a multiple of 8");
        assert!(bp.kc >= 64, "kc must be >= 64, got {}", bp.kc);
        assert!(bp.kc <= 768, "kc must be <= 768, got {}", bp.kc);
        // mc is multiple of MR and within [MR, 960].
        assert_eq!(bp.mc % MR, 0, "mc must be a multiple of MR={}", MR);
        assert!(bp.mc >= MR, "mc must be >= MR={}, got {}", MR, bp.mc);
        assert!(bp.mc <= 960, "mc must be <= 960, got {}", bp.mc);
        // nc is multiple of NR and within [NR, 8192].
        assert_eq!(bp.nc % NR, 0, "nc must be a multiple of NR={}", NR);
        assert!(bp.nc >= NR, "nc must be >= NR={}, got {}", NR, bp.nc);
        assert!(bp.nc <= 8192, "nc must be <= 8192, got {}", bp.nc);
    }

    // ---- 10 additional edge-case tests ----

    #[test]
    fn test_gemm_zero_dimension_n() {
        // Arrange: n=0 should produce an all-zero C and not panic.
        let (m, n, k) = (8, 0, 4);
        let a: Vec<f32> = vec![1.0; m * k];
        let b: Vec<f32> = Vec::new();
        let mut c = vec![0.0f32; 16]; // extra capacity to observe no write
        let c_before = c.clone();
        // Act.
        gemm_asm_f32(&a, &b, &mut c, m, n, k);
        // Assert: C unchanged (early return when any dimension is zero).
        assert_eq!(c, c_before, "C must be unchanged when n=0");
    }

    #[test]
    fn test_gemm_zero_dimension_k() {
        // Arrange: k=0 should produce an all-zero C and not panic.
        let (m, n, k) = (4, 12, 0);
        let a: Vec<f32> = Vec::new();
        let b: Vec<f32> = Vec::new();
        let mut c = vec![0.0f32; m * n];
        let c_before = c.clone();
        // Act.
        gemm_asm_f32(&a, &b, &mut c, m, n, k);
        // Assert: C unchanged (early return when k=0).
        assert_eq!(c, c_before, "C must be unchanged when k=0");
    }

    #[test]
    fn test_gemm_single_row_matrix() {
        // Arrange: m=1, n=25, k=4 — single row, n spans multiple NR tiles.
        let (m, n, k) = (1, 25, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        // Act.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        // Assert.
        check_close(&c_asm, &c_ref, 1e-4, "single_row");
    }

    #[test]
    fn test_gemm_single_column_matrix() {
        // Arrange: m=17, n=1, k=4 — single column, m has MR remainder.
        let (m, n, k) = (17, 1, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        // Act.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        // Assert.
        check_close(&c_asm, &c_ref, 1e-4, "single_column");
    }

    #[test]
    fn test_pack_b_multiple_nr_strips() {
        // Arrange: nc=25 → 3 NR strips (12+12+1), with remainder in last.
        let (kc, nc) = (2, 25);
        let b: Vec<f32> = (0..kc * nc).map(|i| i as f32).collect();
        // Act.
        let packed = pack_b_asm_f32_neon(&b, nc, kc);
        // Assert: 3 strips × NR × kc = 3 * 12 * 2 = 72.
        let n_nr = (nc + NR - 1) / NR;
        assert_eq!(n_nr, 3, "must have 3 NR strips");
        assert_eq!(packed.len(), n_nr * NR * kc, "packed size");
        // First strip: rows 0-11, exact NR.
        for k in 0..kc {
            for j in 0..NR {
                assert_eq!(
                    packed[k * NR + j], b[k * nc + j],
                    "strip0[k={},j={}]", k, j
                );
            }
        }
    }

    #[test]
    fn test_pack_b_layout_column_contiguous() {
        // Arrange: verify packed B is column-contiguous within each NR strip.
        let (kc, nc) = (3, 12); // exact one NR strip
        let b: Vec<f32> = (0..kc * nc).map(|i| (i + 1) as f32).collect();
        // Act.
        let packed = pack_b_asm_f32_neon(&b, nc, kc);
        // Assert: packed[k * NR + j] == b[k * nc + j] for all k, j.
        // Within each column k, NR elements are contiguous.
        for k in 0..kc {
            let col_start = k * NR;
            for j in 0..nc {
                assert!(
                    (packed[col_start + j] - b[k * nc + j]).abs() < 1e-7,
                    "layout[k={},j={}]: packed={}, b={}",
                    k, j, packed[col_start + j], b[k * nc + j]
                );
            }
        }
    }

    #[test]
    fn test_gemm_prepacked_non_aligned_dims() {
        // Arrange: non-aligned m=5, n=7, k=3 — both m and n have remainders.
        let (m, n, k) = (5, 7, 3);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
        let mut c_std = vec![0.0f32; m * n];
        let mut c_pp = vec![0.0f32; m * n];
        // Act: standard GEMM.
        gemm_asm_f32(&a, &b, &mut c_std, m, n, k);
        // Act: prepacked B then GEMM.
        let packed_b = pack_b_asm_f32_neon(&b, n, k);
        gemm_prepacked_asm_f32(&a, &packed_b, &mut c_pp, m, n, k);
        // Assert: both paths must agree.
        check_close(&c_std, &c_pp, 1e-5, "prepacked_non_aligned");
    }

    #[test]
    fn test_gemm_k_spanning_kc_blocks() {
        // Arrange: large k to force multiple KC blocks in the KC loop.
        let (m, n, k) = (8, 12, 512);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        // Act.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        // Assert: k accumulation across KC blocks must be correct.
        check_close(&c_asm, &c_ref, 5e-4, "k_spanning_kc");
    }

    #[test]
    fn test_gemm_1x1_larger_k() {
        // Arrange: 1x1 output but k=24 → dot product of two length-24 vectors.
        let (m, n, k) = (1, 1, 24);
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..k).map(|i| ((i as f32) + 1.0) * 0.2).collect();
        let mut c_asm = vec![0.0f32; 1];
        let mut c_ref = vec![0.0f32; 1];
        // Act.
        gemm_asm_f32(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        // Assert.
        let diff = (c_asm[0] - c_ref[0]).abs();
        let scale = c_asm[0].abs().max(c_ref[0].abs()).max(1.0);
        assert!(
            diff / scale < 1e-4,
            "1x1_k24: asm={}, ref={}, diff={}", c_asm[0], c_ref[0], diff
        );
    }

    #[test]
    fn test_gemm_prepacked_zero_k() {
        // Arrange: k=0 with prepacked B (empty).
        let (m, n, k) = (4, 12, 0);
        let a: Vec<f32> = Vec::new();
        let packed_b: Vec<f32> = Vec::new();
        let mut c = vec![0.0f32; m * n];
        let c_before = c.clone();
        // Act.
        gemm_prepacked_asm_f32(&a, &packed_b, &mut c, m, n, k);
        // Assert: C unchanged (early return when k=0).
        assert_eq!(c, c_before, "C must be unchanged when k=0 with prepacked B");
    }
}
