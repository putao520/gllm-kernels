//! x86_64 GEMM drivers using hand-written AVX2/AVX-512 assembly microkernels.
//!
//! This module provides the complete GEMM implementation:
//! - Pack A into column-major MR-wide panels
//! - Pack B into row-major NR-wide panels
//! - Tile the M/N/K dimensions with cache-aware blocking
//! - Call the assembly microkernel for each tile
//!
//! Two entry points:
//! - `gemm_asm_f32_avx2`:   uses the 6x16 AVX2 microkernel
//! - `gemm_asm_f32_avx512`: uses the 14x32 AVX-512 microkernel

#[cfg(target_arch = "x86_64")]
use crate::cache_params::{self, AlignedVec};

/// Pack a panel of A (mc x kc) into column-major MR-wide strips.
///
/// Input A is row-major: a[i*lda + k]
/// Output: packed[k*MR + i_local] for each MR-wide strip
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn pack_a_f32(
    a: *const f32,
    lda: usize,
    packed: *mut f32,
    mc: usize,
    kc: usize,
    mr: usize,
) {
    let mut p = packed;
    let mut i = 0usize;

    while i + mr <= mc {
        for k in 0..kc {
            for r in 0..mr {
                *p.add(r) = *a.add((i + r) * lda + k);
            }
            p = p.add(mr);
        }
        i += mr;
    }

    if i < mc {
        let rem = mc - i;
        for k in 0..kc {
            for r in 0..rem {
                *p.add(r) = *a.add((i + r) * lda + k);
            }
            for r in rem..mr {
                *p.add(r) = 0.0;
            }
            p = p.add(mr);
        }
    }
}

/// Pack a panel of B (kc x nc) into row-major NR-wide strips.
///
/// Input B is row-major: b[k*ldb + j]
/// Output: packed[k*NR + j_local] for each NR-wide strip
///
/// This layout ensures the microkernel can load NR consecutive B elements
/// with `vmovups` (AVX2: 2x ymm, AVX-512: 2x zmm).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn pack_b_f32(
    b: *const f32,
    ldb: usize,
    packed: *mut f32,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    let mut p = packed;
    let mut j = 0usize;

    // Full NR-wide panels
    while j + nr <= nc {
        for k in 0..kc {
            let src = b.add(k * ldb + j);
            std::ptr::copy_nonoverlapping(src, p, nr);
            p = p.add(nr);
        }
        j += nr;
    }

    // Remainder columns (< NR): zero-pad to NR width
    if j < nc {
        let rem = nc - j;
        for k in 0..kc {
            let src = b.add(k * ldb + j);
            std::ptr::copy_nonoverlapping(src, p, rem);
            for c in rem..nr {
                *p.add(c) = 0.0;
            }
            p = p.add(nr);
        }
    }
}

/// Generic BLIS-style GEMM driver parameterized by microkernel geometry.
///
/// C = A * B  (row-major, C is m x n, A is m x k, B is k x n)
///
/// Uses 3-level blocking (NC/KC/MC) with parameters from `kernel_config()`.
/// Pack buffers use thread-local `AlignedVec` (64-byte aligned, zero allocation
/// on hot path after first call).
#[cfg(target_arch = "x86_64")]
fn gemm_driver_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    microkernel: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, bool),
) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // Get blocking params from kernel_config (BLIS formula, computed once)
    let cfg = crate::microarch::kernel_config();
    let kc_max = cfg.kc;
    let mc_max = cfg.mc;
    let nc_max = cfg.nc;

    // Thread-local aligned packing buffers (zero alloc on hot path)
    thread_local! {
        static PACK_A: std::cell::Cell<AlignedVec<f32>> = std::cell::Cell::new(AlignedVec::new());
        static PACK_B: std::cell::Cell<AlignedVec<f32>> = std::cell::Cell::new(AlignedVec::new());
    }

    let pack_a_size = mc_max * kc_max;
    let pack_b_size = kc_max * nc_max;

    let mut pa = PACK_A.with(|c| c.take());
    let mut pb = PACK_B.with(|c| c.take());

    if pa.capacity() < pack_a_size { pa.reserve(pack_a_size); }
    if pb.capacity() < pack_b_size { pb.reserve(pack_b_size); }
    unsafe { pa.set_len(pack_a_size); pb.set_len(pack_b_size); }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // NC loop: iterate over N in chunks of nc_max
    let mut jc = 0usize;
    while jc < n {
        let nc = nc_max.min(n - jc);

        // KC loop: iterate over K in chunks of kc_max
        let mut pc = 0usize;
        while pc < k {
            let kc = kc_max.min(k - pc);
            let first_kc = pc == 0;

            // Pack B panel: B[pc..pc+kc, jc..jc+nc] -> packed_b
            unsafe {
                pack_b_f32(
                    b_ptr.add(pc * n + jc), n,
                    pb.as_mut_ptr(), kc, nc, nr,
                );
            }

            // MC loop: iterate over M in chunks of mc_max
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

                // Temp buffer for edge tiles
                let mut c_tmp = [0.0f32; 14 * 32];

                for jr in 0..n_nr {
                    let col_start = jc + jr * nr;
                    let col_rem = n.saturating_sub(col_start).min(nr);
                    let is_edge_col = col_rem < nr;

                    for ir in 0..n_mr {
                        let row_start = ic + ir * mr;
                        let row_rem = m.saturating_sub(row_start).min(mr);
                        let is_edge_row = row_rem < mr;

                        unsafe {
                            let pa_tile = pa.as_ptr().add(ir * mr * kc);
                            let pb_tile = pb.as_ptr().add(jr * nr * kc);

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
                                    pa_tile, pb_tile, c_tmp.as_mut_ptr(),
                                    kc, nr, !first_kc,
                                );

                                for ri in 0..row_rem {
                                    for ci in 0..col_rem {
                                        *c_ptr.add((row_start + ri) * n + col_start + ci) =
                                            c_tmp[ri * nr + ci];
                                    }
                                }
                            } else {
                                let c_tile = c_ptr.add(row_start * n + col_start);
                                microkernel(
                                    pa_tile, pb_tile, c_tile,
                                    kc, n, !first_kc,
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

    // Return buffers to thread-local storage
    PACK_A.with(|c| c.set(pa));
    PACK_B.with(|c| c.set(pb));
}

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

    gemm_driver_f32(a, b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc| unsafe {
        gemm_kernel_6x16_f32(pa, pb, cp, kc, ldc, acc);
    });
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

    gemm_driver_f32(a, b, c, m, n, k, MR, NR, |pa, pb, cp, kc, ldc, acc| unsafe {
        gemm_kernel_14x32_f32(pa, pb, cp, kc, ldc, acc);
    });
}

/// GEMM with bias using AVX2: C = A * B + bias
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
    gemm_asm_f32_avx2(a, b, c, m, n, k);
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
}

/// GEMM with bias using AVX-512: C = A * B + bias
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
    gemm_asm_f32_avx512(a, b, c, m, n, k);
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
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

    // ── AVX2 tests ──

    #[test]
    fn test_avx2_exact_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx2_exact_tile");
    }

    #[test]
    fn test_avx2_multi_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (24, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx2_multi_tile");
    }

    #[test]
    fn test_avx2_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let m_padded = ((m + 5) / 6) * 6;
        let n_padded = ((n + 15) / 16) * 16;
        let mut c_asm = vec![0.0f32; m_padded * n_padded.max(n)];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let asm_val = c_asm[i * n + j];
                let ref_val = c_ref[i * n + j];
                let diff = (asm_val - ref_val).abs();
                let scale = asm_val.abs().max(ref_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_non_aligned[{i},{j}]: asm={asm_val}, ref={ref_val}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_avx2_identity() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 16);
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_asm_f32_avx2(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let expected = b[i * n + j];
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "avx2_identity[{i},{j}]: got={}, expected={expected}", c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx2_bias() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_asm_f32_avx2(&a, &b, &bias, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        check_close(&c_asm, &c_ref, 1e-4, "avx2_bias");
    }

    // ── AVX-512 tests ──

    #[test]
    fn test_avx512_exact_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx512_exact_tile");
    }

    #[test]
    fn test_avx512_multi_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let mut c_asm = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_asm, &c_ref, 1e-4, "avx512_multi_tile");
    }

    #[test]
    fn test_avx512_non_aligned() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (15, 37, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let m_padded = ((m + 13) / 14) * 14;
        let n_padded = ((n + 31) / 32) * 32;
        let mut c_asm = vec![0.0f32; m_padded * n_padded.max(n)];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c_asm, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let asm_val = c_asm[i * n + j];
                let ref_val = c_ref[i * n + j];
                let diff = (asm_val - ref_val).abs();
                let scale = asm_val.abs().max(ref_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx512_non_aligned[{i},{j}]: asm={asm_val}, ref={ref_val}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_avx512_identity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 32);
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_asm_f32_avx512(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let expected = b[i * n + j];
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "avx512_identity[{i},{j}]: got={}, expected={expected}", c[i * n + j]
                );
            }
        }
    }
}
