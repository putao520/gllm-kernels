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
///
/// For MR=6 (AVX2) uses a vectorized 6×8 transpose kernel:
///   - Reads 6 rows × 8 columns with vmovups (sequential, cache-friendly)
///   - Transposes with vunpcklps/vunpckhps/vperm2f128 (no memory traffic)
///   - Writes 8 packed columns of 6 elements each
/// This eliminates the scalar gather pattern and saturates L1D bandwidth.
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
    // AVX2 fast path for MR=6: vectorized 6×8 transpose.
    // Runtime check so this works regardless of compile-time target-cpu.
    if mr == 6 && is_x86_feature_detected!("avx2") {
        // Safety: we just confirmed AVX2 is available at runtime.
        pack_a_f32_avx2_mr6(a, lda, packed, mc, kc);
        return;
    }

    // Scalar fallback for other MR values or non-AVX2 hosts
    let mut p = packed;
    let mut i = 0usize;

    while i + mr <= mc {
        for r in 0..mr {
            let a_row = a.add((i + r) * lda);
            for k in 0..kc {
                *p.add(k * mr + r) = *a_row.add(k);
            }
        }
        p = p.add(mr * kc);
        i += mr;
    }

    if i < mc {
        let rem = mc - i;
        std::ptr::write_bytes(p, 0, mr * kc);
        for r in 0..rem {
            let a_row = a.add((i + r) * lda);
            for k in 0..kc {
                *p.add(k * mr + r) = *a_row.add(k);
            }
        }
    }
}

/// AVX2 vectorized pack_a for MR=6: 6×8 transpose kernel.
///
/// Processes 8 K-columns at a time using AVX2 shuffle instructions.
/// Each iteration reads 6×8 = 48 floats and writes 48 floats in packed layout.
///
/// Transpose of 6×8 block:
///   Input rows:  r0[k0..k7], r1[k0..k7], ..., r5[k0..k7]
///   Output cols: packed[k*6 + 0..5] for k in 0..8
///
/// Uses vunpcklps/vunpckhps to interleave pairs, then vperm2f128 to
/// combine low/high 128-bit halves across the 256-bit registers.
///
/// Marked with #[target_feature(enable = "avx2")] so it always compiles
/// but is only called after a runtime `is_x86_feature_detected!("avx2")` check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_a_f32_avx2_mr6(
    a: *const f32,
    lda: usize,
    packed: *mut f32,
    mc: usize,
    kc: usize,
) {
    use std::arch::x86_64::*;
    const MR: usize = 6;

    let mut p = packed;
    let mut i = 0usize;

    // Full MR=6 panels
    while i + MR <= mc {
        let row0 = a.add((i + 0) * lda);
        let row1 = a.add((i + 1) * lda);
        let row2 = a.add((i + 2) * lda);
        let row3 = a.add((i + 3) * lda);
        let row4 = a.add((i + 4) * lda);
        let row5 = a.add((i + 5) * lda);

        let mut k = 0usize;

        // Process 8 K-columns at a time with AVX2 transpose
        while k + 8 <= kc {
            // Load 8 floats from each of the 6 rows
            let r0 = _mm256_loadu_ps(row0.add(k));
            let r1 = _mm256_loadu_ps(row1.add(k));
            let r2 = _mm256_loadu_ps(row2.add(k));
            let r3 = _mm256_loadu_ps(row3.add(k));
            let r4 = _mm256_loadu_ps(row4.add(k));
            let r5 = _mm256_loadu_ps(row5.add(k));

            // Transpose 6×8 → 8×6 using shuffle/permute
            // Step 1: interleave pairs within each 128-bit lane
            // unpacklo: [a0,b0,a1,b1, a4,b4,a5,b5]
            // unpackhi: [a2,b2,a3,b3, a6,b6,a7,b7]
            let u01l = _mm256_unpacklo_ps(r0, r1); // [r0[0],r1[0],r0[1],r1[1], r0[4],r1[4],r0[5],r1[5]]
            let u01h = _mm256_unpackhi_ps(r0, r1); // [r0[2],r1[2],r0[3],r1[3], r0[6],r1[6],r0[7],r1[7]]
            let u23l = _mm256_unpacklo_ps(r2, r3);
            let u23h = _mm256_unpackhi_ps(r2, r3);
            let u45l = _mm256_unpacklo_ps(r4, r5);
            let u45h = _mm256_unpackhi_ps(r4, r5);

            // Step 2: interleave quads
            // shuffle(u01l, u23l, 0x44) = [u01l[0..1], u23l[0..1], u01l[4..5], u23l[4..5]]
            //   = [r0[0],r1[0],r2[0],r3[0], r0[4],r1[4],r2[4],r3[4]]
            let q0 = _mm256_shuffle_ps::<0x44>(u01l, u23l); // cols 0,4: r0-r3
            let q1 = _mm256_shuffle_ps::<0xEE>(u01l, u23l); // cols 1,5: r0-r3
            let q2 = _mm256_shuffle_ps::<0x44>(u01h, u23h); // cols 2,6: r0-r3
            let q3 = _mm256_shuffle_ps::<0xEE>(u01h, u23h); // cols 3,7: r0-r3

            // Step 3: extract low/high 128-bit halves to get contiguous k-columns
            // For k-column 0: [r0[0],r1[0],r2[0],r3[0]] from low half of q0
            // For k-column 4: [r0[4],r1[4],r2[4],r3[4]] from high half of q0
            let col0_03 = _mm256_permute2f128_ps::<0x20>(q0, q1); // low halves: k=0,1
            let col2_03 = _mm256_permute2f128_ps::<0x20>(q2, q3); // low halves: k=2,3
            let col4_03 = _mm256_permute2f128_ps::<0x31>(q0, q1); // high halves: k=4,5
            let col6_03 = _mm256_permute2f128_ps::<0x31>(q2, q3); // high halves: k=6,7

            // Extract 128-bit lanes for the 4-element (r0-r3) parts of each k-column
            let c0_03 = _mm256_castps256_ps128(col0_03);      // [r0[0],r1[0],r2[0],r3[0]]
            let c1_03 = _mm256_extractf128_ps::<1>(col0_03);  // [r0[1],r1[1],r2[1],r3[1]]
            let c2_03 = _mm256_castps256_ps128(col2_03);
            let c3_03 = _mm256_extractf128_ps::<1>(col2_03);
            let c4_03 = _mm256_castps256_ps128(col4_03);
            let c5_03 = _mm256_extractf128_ps::<1>(col4_03);
            let c6_03 = _mm256_castps256_ps128(col6_03);
            let c7_03 = _mm256_extractf128_ps::<1>(col6_03);

            // r4,r5 contribution: 2 rows only, use 128-bit ops
            let u45l_lo = _mm256_castps256_ps128(u45l);
            let u45l_hi = _mm256_extractf128_ps::<1>(u45l);
            let u45h_lo = _mm256_castps256_ps128(u45h);
            let u45h_hi = _mm256_extractf128_ps::<1>(u45h);

            // Store the 128-bit registers to temp arrays for scalar extraction.
            // u45l_lo = [r4[0], r5[0], r4[1], r5[1]]
            // u45h_lo = [r4[2], r5[2], r4[3], r5[3]]
            // u45l_hi = [r4[4], r5[4], r4[5], r5[5]]
            // u45h_hi = [r4[6], r5[6], r4[7], r5[7]]
            let mut t45ll = [0.0f32; 4];
            let mut t45lh = [0.0f32; 4];
            let mut t45hl = [0.0f32; 4];
            let mut t45hh = [0.0f32; 4];
            _mm_storeu_ps(t45ll.as_mut_ptr(), u45l_lo);
            _mm_storeu_ps(t45lh.as_mut_ptr(), u45l_hi);
            _mm_storeu_ps(t45hl.as_mut_ptr(), u45h_lo);
            _mm_storeu_ps(t45hh.as_mut_ptr(), u45h_hi);

            // Write each k-column: 4 floats (r0-r3) via 128-bit store, then 2 floats (r4-r5) scalar
            // k+0
            _mm_storeu_ps(p.add((k + 0) * MR), c0_03);
            *p.add((k + 0) * MR + 4) = t45ll[0];
            *p.add((k + 0) * MR + 5) = t45ll[1];
            // k+1
            _mm_storeu_ps(p.add((k + 1) * MR), c1_03);
            *p.add((k + 1) * MR + 4) = t45ll[2];
            *p.add((k + 1) * MR + 5) = t45ll[3];
            // k+2
            _mm_storeu_ps(p.add((k + 2) * MR), c2_03);
            *p.add((k + 2) * MR + 4) = t45hl[0];
            *p.add((k + 2) * MR + 5) = t45hl[1];
            // k+3
            _mm_storeu_ps(p.add((k + 3) * MR), c3_03);
            *p.add((k + 3) * MR + 4) = t45hl[2];
            *p.add((k + 3) * MR + 5) = t45hl[3];
            // k+4
            _mm_storeu_ps(p.add((k + 4) * MR), c4_03);
            *p.add((k + 4) * MR + 4) = t45lh[0];
            *p.add((k + 4) * MR + 5) = t45lh[1];
            // k+5
            _mm_storeu_ps(p.add((k + 5) * MR), c5_03);
            *p.add((k + 5) * MR + 4) = t45lh[2];
            *p.add((k + 5) * MR + 5) = t45lh[3];
            // k+6
            _mm_storeu_ps(p.add((k + 6) * MR), c6_03);
            *p.add((k + 6) * MR + 4) = t45hh[0];
            *p.add((k + 6) * MR + 5) = t45hh[1];
            // k+7
            _mm_storeu_ps(p.add((k + 7) * MR), c7_03);
            *p.add((k + 7) * MR + 4) = t45hh[2];
            *p.add((k + 7) * MR + 5) = t45hh[3];

            k += 8;
        }

        // Scalar tail for remaining k columns
        while k < kc {
            *p.add(k * MR + 0) = *row0.add(k);
            *p.add(k * MR + 1) = *row1.add(k);
            *p.add(k * MR + 2) = *row2.add(k);
            *p.add(k * MR + 3) = *row3.add(k);
            *p.add(k * MR + 4) = *row4.add(k);
            *p.add(k * MR + 5) = *row5.add(k);
            k += 1;
        }

        p = p.add(MR * kc);
        i += MR;
    }

    // Remainder panel (< MR rows): zero-pad to MR width
    if i < mc {
        let rem = mc - i;
        std::ptr::write_bytes(p, 0, MR * kc);
        for r in 0..rem {
            let a_row = a.add((i + r) * lda);
            for k in 0..kc {
                *p.add(k * MR + r) = *a_row.add(k);
            }
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

// ── C-ABI wrappers for JIT codegen ──────────────────────────────────
//
// The JIT compiler emits `mov rax, fn_ptr; call rax` to invoke these.
// They must use the C calling convention so the JIT-generated code can
// set up arguments in the standard System V AMD64 registers.

/// C-ABI wrapper around [`pack_a_f32`] for JIT-emitted GEMM code.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
pub unsafe extern "C" fn gllm_pack_a_f32(
    a: *const f32,
    lda: usize,
    packed: *mut f32,
    mc: usize,
    kc: usize,
    mr: usize,
) {
    pack_a_f32(a, lda, packed, mc, kc, mr);
}

/// C-ABI wrapper around [`pack_b_f32`] for JIT-emitted GEMM code.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
pub unsafe extern "C" fn gllm_pack_b_f32(
    b: *const f32,
    ldb: usize,
    packed: *mut f32,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    pack_b_f32(b, ldb, packed, kc, nc, nr);
}

/// Apply bias to a block of C: C[i,j] += bias[j].
/// Uses slices so the compiler can auto-vectorize with -C target-cpu=native.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn add_bias_block_f32(
    c: *mut f32, ldc: usize,
    row_start: usize, col_start: usize,
    rows: usize, cols: usize,
    bias: *const f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let bias_base = bias.add(col_start);

        for i in 0..rows {
            let c_row = c.add((row_start + i) * ldc + col_start);
            let mut j = 0usize;

            // AVX2: 8 floats per iteration, 4x unrolled = 32 floats
            while j + 32 <= cols {
                let b0 = _mm256_loadu_ps(bias_base.add(j));
                let b1 = _mm256_loadu_ps(bias_base.add(j + 8));
                let b2 = _mm256_loadu_ps(bias_base.add(j + 16));
                let b3 = _mm256_loadu_ps(bias_base.add(j + 24));
                let c0 = _mm256_loadu_ps(c_row.add(j));
                let c1 = _mm256_loadu_ps(c_row.add(j + 8));
                let c2 = _mm256_loadu_ps(c_row.add(j + 16));
                let c3 = _mm256_loadu_ps(c_row.add(j + 24));
                _mm256_storeu_ps(c_row.add(j),      _mm256_add_ps(c0, b0));
                _mm256_storeu_ps(c_row.add(j + 8),  _mm256_add_ps(c1, b1));
                _mm256_storeu_ps(c_row.add(j + 16), _mm256_add_ps(c2, b2));
                _mm256_storeu_ps(c_row.add(j + 24), _mm256_add_ps(c3, b3));
                j += 32;
            }
            // 8-float tail
            while j + 8 <= cols {
                let b = _mm256_loadu_ps(bias_base.add(j));
                let cv = _mm256_loadu_ps(c_row.add(j));
                _mm256_storeu_ps(c_row.add(j), _mm256_add_ps(cv, b));
                j += 8;
            }
            // Scalar tail
            while j < cols {
                *c_row.add(j) += *bias_base.add(j);
                j += 1;
            }
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
    microkernel: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, bool, *const f32),
    bias: *const f32,
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
    // +nr padding: microkernel speculatively loads B[next] before checking loop exit
    let pack_b_size = kc_max * nc_max + nr;

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
            let last_kc = pc + kc_max >= k;

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
                                    pa_tile, pb_tile, c_tile,
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

    // Return buffers to thread-local storage
    PACK_A.with(|c| c.set(pa));
    PACK_B.with(|c| c.set(pb));
}

/// Multi-threaded BLIS-style GEMM driver. Parallelizes the MC loop with Rayon.
///
/// C = A * B  (row-major, C is m x n, A is m x k, B is k x n)
///
/// The NC/KC outer loops run on the main thread. Pack B is done once per KC
/// iteration. The MC loop is distributed across Rayon worker threads, each
/// using a thread-local pack_a buffer.
///
/// Falls back to single-threaded `gemm_driver_f32` when the problem is too
/// small to benefit from parallelism.
#[cfg(target_arch = "x86_64")]
fn gemm_driver_f32_mt(
    a: &[f32],
    b: &[f32],
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
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let cfg = crate::microarch::kernel_config();
    let kc_max = cfg.kc;
    let mc_max = cfg.mc;
    // For MT: cap NC so that pack_b + C working set fits in L3.
    // cfg.nc was sized for pack_b alone (L3*0.5 / (KC*elem)), so
    // L3 ≈ cfg.nc * cfg.kc * elem * 2.  We want:
    //   NC * KC * 4 + M * NC * 4 ≤ L3 * 0.6
    //   NC ≤ L3 * 0.6 / (4 * (KC + M))
    let nc_max = {
        let l3_est = cfg.nc * cfg.kc * 4 * 2; // reconstruct L3 from NC formula
        let budget = l3_est * 6 / 10; // 60%
        let nc_for_l3 = budget / (4 * (kc_max + m));
        let nr = 16usize; // TN alignment
        let nc_capped = (nc_for_l3 / nr * nr).max(nr);
        nc_capped.min(cfg.nc)
    };

    let nthreads = rayon::current_num_threads().max(1);
    // For compute-bound GEMM, HT provides no benefit. Use physical core count
    // for parallelization decisions (approximate: logical_threads / 2).
    let phys_cores = (nthreads / 2).max(1);

    // No adaptive MC — shrinking MC reduces B-panel reuse and hurts per-core
    // throughput. Instead, when M-blocks < phys_cores, we use 2D parallelism
    // to distribute work across both M and N dimensions.

    let num_m_blocks = (m + mc_max - 1) / mc_max;

    // Fall back to single-threaded for small problems
    if num_m_blocks < 2 || nthreads <= 1 {
        return gemm_driver_f32(a, b, c, m, n, k, mr, nr, microkernel, bias);
    }

    // 2D parallelism: when M-blocks can't fill physical cores
    // (e.g., M=128 MC=84 → 2 blocks, or M=512 MC=84 → 7 blocks for 10 cores)
    let use_2d = num_m_blocks < phys_cores;

    // Shared pack_b buffer (allocated once on main thread, read by all workers)
    thread_local! {
        static PACK_B_MT: std::cell::Cell<AlignedVec<f32>> = std::cell::Cell::new(AlignedVec::new());
    }

    // +nr padding: microkernel speculatively loads B[next] before checking loop exit
    let pack_b_size = kc_max * nc_max + nr;
    let mut pb = PACK_B_MT.with(|c| c.take());
    let prev_cap = pb.capacity();
    if prev_cap < pack_b_size { pb.reserve(pack_b_size); }
    unsafe { pb.set_len(pack_b_size); }

    // NUMA: interleave shared pack_b across nodes (only on first alloc)
    if prev_cap < pack_b_size {
        let byte_size = pack_b_size * std::mem::size_of::<f32>();
        let _ = crate::numa::mbind_interleave(pb.as_mut_ptr() as *mut u8, byte_size);
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
            let last_kc = pc + kc_max >= k;

            // Pack B ONCE on the main thread
            unsafe {
                pack_b_f32(b_ptr.add(pc * n + jc), n, pb.as_mut_ptr(), kc, nc, nr);
            }

            // Send raw pointers across threads via usize (safe: all point into
            // caller-owned slices that outlive the parallel region).
            let a_addr = a_ptr as usize;
            let c_addr = c_ptr as usize;
            let pb_addr = pb.as_ptr() as usize;
            let bias_addr = bias as usize;

            use rayon::prelude::*;

            if use_2d {
                // 2D parallel: pre-pack all A blocks on main thread, then
                // parallelize compute over (m_block, n_chunk) tiles.
                // Each n_chunk covers multiple NR strips to reduce scheduling overhead.
                let n_nr_blocks = (nc + nr - 1) / nr;
                // Target: ~4x physical cores total tiles for good load balance
                let target_tiles = phys_cores * 4;
                let nr_per_chunk = (n_nr_blocks * num_m_blocks / target_tiles).max(1);
                let n_chunks = (n_nr_blocks + nr_per_chunk - 1) / nr_per_chunk;
                let total_tiles = num_m_blocks * n_chunks;

                // Pre-pack all A blocks (few blocks since M is small)
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

                        for ir in 0..n_mr {
                            let row_start = ic + ir * mr;
                            let row_rem = m.saturating_sub(row_start).min(mr);
                            let is_edge_row = row_rem < mr;

                            unsafe {
                                let pa_tile = pa_ptr.add(ir * mr * kc);
                                let pb_tile = pb_ptr.add(jr * nr * kc);

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
                                        pa_tile, pb_tile, c_tile,
                                        kc, n, !first_kc, tile_bias,
                                    );
                                }
                            }
                        }
                    }
                });

                // Keep all_pa alive until parallel region completes (it does, par_iter is blocking)
                drop(all_pa);
            } else {
            // 1D parallel: parallelize over M-blocks (original path)
            (0..num_m_blocks).into_par_iter().for_each(|block_idx| {
                let ic = block_idx * mc_max;
                if ic >= m { return; }
                let mc = mc_max.min(m - ic);

                // Thread-local pack_a buffer (separate TLS key from the
                // single-threaded driver to avoid conflicts)
                thread_local! {
                    static PACK_A_MT: std::cell::Cell<AlignedVec<f32>> =
                        std::cell::Cell::new(AlignedVec::new());
                }
                let pack_a_size = mc_max * kc_max;
                let mut pa = PACK_A_MT.with(|c| c.take());
                if pa.capacity() < pack_a_size { pa.reserve(pack_a_size); }
                unsafe { pa.set_len(pack_a_size); }

                let a_ptr = a_addr as *const f32;
                let c_ptr = c_addr as *mut f32;
                let pb_ptr = pb_addr as *const f32;

                // Pack A for this M-block
                unsafe {
                    pack_a_f32(a_ptr.add(ic * k + pc), k, pa.as_mut_ptr(), mc, kc, mr);
                }

                let n_mr = (mc + mr - 1) / mr;
                let n_nr = (nc + nr - 1) / nr;
                let mut c_tmp = [0.0f32; 14 * 32]; // max MR*NR

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
                            let pb_tile = pb_ptr.add(jr * nr * kc);

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
                                    kc, nr, !first_kc, std::ptr::null(),
                                );

                                // Fuse bias into edge-tile store
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
                                    pa_tile, pb_tile, c_tile,
                                    kc, n, !first_kc, tile_bias,
                                );
                            }
                        }
                    }
                }

                // Return pack_a to thread-local storage
                PACK_A_MT.with(|c| c.set(pa));
            });
            } // end else (1D parallel)

            pc += kc_max;
        }

        jc += nc_max;
    }

    // Return pack_b to thread-local storage
    PACK_B_MT.with(|c| c.set(pb));
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
                        let prefetch_bytes = kc * nr * 4;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / 4) as *const i8,
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
                            let prefetch_bytes = kc * nr * 4;
                            let mut pf_off = 0usize;
                            while pf_off < prefetch_bytes {
                                unsafe {
                                    #[cfg(target_arch = "x86_64")]
                                    std::arch::x86_64::_mm_prefetch(
                                        next_pb.add(pf_off / 4) as *const i8,
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
                        let prefetch_bytes = kc * nr * 4;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / 4) as *const i8,
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
                        let prefetch_bytes = kc * nr * 4;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / 4) as *const i8,
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
                        let prefetch_bytes = kc * nr * 4;
                        let mut pf_off = 0usize;
                        while pf_off < prefetch_bytes {
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                std::arch::x86_64::_mm_prefetch(
                                    next_pb.add(pf_off / 4) as *const i8,
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

    // ── Prepacked AVX2 tests ──

    #[test]
    fn test_avx2_prepacked_exact_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx2_prepacked_exact_tile");
    }

    #[test]
    fn test_avx2_prepacked_multi_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (24, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx2_prepacked_multi_tile");
    }

    #[test]
    fn test_avx2_prepacked_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_pp[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_pp[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_prepacked_non_aligned[{i},{j}]: pp={}, ref={}, diff={diff}",
                    c_pp[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx2_prepacked_vs_regular() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (48, 96, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx2(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_asm_f32_avx2(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_pp, &c_reg, 1e-5, "avx2_prepacked_vs_regular");
    }

    #[test]
    fn test_avx2_prepacked_bias() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (12, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_prepacked_asm_f32_avx2(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        check_close(&c_pp, &c_ref, 1e-4, "avx2_prepacked_bias");
    }

    #[test]
    fn test_avx2_prepacked_bias_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_prepacked_asm_f32_avx2(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        for i in 0..m {
            for j in 0..n {
                let diff = (c_pp[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_pp[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_prepacked_bias_non_aligned[{i},{j}]: pp={}, ref={}, diff={diff}",
                    c_pp[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    // ── Prepacked AVX-512 tests ──

    #[test]
    fn test_avx512_prepacked_exact_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx512_prepacked_exact_tile");
    }

    #[test]
    fn test_avx512_prepacked_multi_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_pp, &c_ref, 1e-4, "avx512_prepacked_multi_tile");
    }

    #[test]
    fn test_avx512_prepacked_non_aligned() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (15, 37, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_pp[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_pp[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx512_prepacked_non_aligned[{i},{j}]: pp={}, ref={}, diff={diff}",
                    c_pp[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx512_prepacked_vs_regular() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_asm_f32_avx512(&a, &packed_b, &mut c_pp, m, n, k);
        gemm_asm_f32_avx512(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_pp, &c_reg, 1e-5, "avx512_prepacked_vs_regular");
    }

    #[test]
    fn test_avx512_prepacked_bias() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (28, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_pp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_bias_prepacked_asm_f32_avx512(&a, &packed_b, &bias, &mut c_pp, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias[j];
            }
        }
        check_close(&c_pp, &c_ref, 1e-4, "avx512_prepacked_bias");
    }

    // ── Prepacked AB (shared pack_a) AVX2 tests ──

    #[test]
    fn test_avx2_prepacked_ab_exact_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (6, 16, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_ab, &c_ref, 1e-4, "avx2_prepacked_ab_exact_tile");
    }

    #[test]
    fn test_avx2_prepacked_ab_multi_tile() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (24, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_ab, &c_ref, 1e-4, "avx2_prepacked_ab_multi_tile");
    }

    #[test]
    fn test_avx2_prepacked_ab_non_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (7, 19, 13);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_ab[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_ab[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx2_prepacked_ab_non_aligned[{i},{j}]: ab={}, ref={}, diff={diff}",
                    c_ab[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx2_prepacked_ab_vs_regular() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (48, 96, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_b = pack_b_asm_f32_avx2(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_asm_f32_avx2(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_ab, &c_reg, 1e-5, "avx2_prepacked_ab_vs_regular");
    }

    /// Verify pack_a output matches what the internal pack_a_f32 produces
    /// for a single KC-block.
    #[test]
    fn test_avx2_pack_a_matches_internal() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use super::super::gemm_avx2::MR;
        let kc = crate::microarch::kernel_config().kc;
        let (m, k) = (13, 37); // non-aligned

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.07).collect();
        let shared = pack_a_asm_f32_avx2(&a, m, k);

        // Manually pack the first KC-block using pack_a_f32 and compare
        let kc_len = kc.min(k);
        let n_mr_strips = (m + MR - 1) / MR;
        let mut manual = vec![0.0f32; n_mr_strips * MR * kc_len];
        unsafe {
            pack_a_f32(a.as_ptr(), k, manual.as_mut_ptr(), m, kc_len, MR);
        }

        // The first KC-block in shared should match manual
        // shared layout: kc_block=0 → offset 0, each strip is kc*MR floats
        // manual layout: each strip is kc_len*MR floats (contiguous)
        for strip in 0..n_mr_strips {
            let shared_off = strip * kc * MR; // kc (max) stride in shared
            let manual_off = strip * kc_len * MR;
            for kl in 0..kc_len {
                for r in 0..MR {
                    let sv = shared.as_slice()[shared_off + kl * MR + r];
                    let mv = manual[manual_off + kl * MR + r];
                    assert!(
                        (sv - mv).abs() < 1e-6,
                        "pack_a mismatch strip={strip} k={kl} r={r}: shared={sv}, manual={mv}"
                    );
                }
            }
        }
    }

    /// QKV-style shared pack_a reuse: pack A once, multiply with 3 different B matrices.
    #[test]
    fn test_avx2_shared_pack_a_qkv() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (m, n, k) = (12, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let bq: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let bk: Vec<f32> = (0..k * n).map(|i| ((i % 71) as f32 - 35.0) * 0.03).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i % 59) as f32 - 29.0) * 0.04).collect();

        // Pack A once
        let packed_a = pack_a_asm_f32_avx2(&a, m, k);
        let packed_bq = pack_b_asm_f32_avx2(&bq, n, k);
        let packed_bk = pack_b_asm_f32_avx2(&bk, n, k);
        let packed_bv = pack_b_asm_f32_avx2(&bv, n, k);

        // Compute Q, K, V with shared packed A
        let mut cq = vec![0.0f32; m * n];
        let mut ck = vec![0.0f32; m * n];
        let mut cv = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_bq, &mut cq, m, n, k);
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_bk, &mut ck, m, n, k);
        gemm_prepacked_ab_asm_f32_avx2(packed_a.as_slice(), &packed_bv, &mut cv, m, n, k);

        // Reference
        let mut cq_ref = vec![0.0f32; m * n];
        let mut ck_ref = vec![0.0f32; m * n];
        let mut cv_ref = vec![0.0f32; m * n];
        gemm_ref(&a, &bq, &mut cq_ref, m, n, k);
        gemm_ref(&a, &bk, &mut ck_ref, m, n, k);
        gemm_ref(&a, &bv, &mut cv_ref, m, n, k);

        check_close(&cq, &cq_ref, 1e-4, "shared_pack_a_Q");
        check_close(&ck, &ck_ref, 1e-4, "shared_pack_a_K");
        check_close(&cv, &cv_ref, 1e-4, "shared_pack_a_V");
    }

    // ── Prepacked AB AVX-512 tests ──

    #[test]
    fn test_avx512_prepacked_ab_exact_tile() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (14, 32, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx512(&a, m, k);
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx512(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        check_close(&c_ab, &c_ref, 1e-4, "avx512_prepacked_ab_exact_tile");
    }

    #[test]
    fn test_avx512_prepacked_ab_non_aligned() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (15, 37, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.04).collect();
        let packed_a = pack_a_asm_f32_avx512(&a, m, k);
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx512(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_ref(&a, &b, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let diff = (c_ab[i * n + j] - c_ref[i * n + j]).abs();
                let scale = c_ab[i * n + j].abs().max(c_ref[i * n + j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-4,
                    "avx512_prepacked_ab_non_aligned[{i},{j}]: ab={}, ref={}, diff={diff}",
                    c_ab[i * n + j], c_ref[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_avx512_prepacked_ab_vs_regular() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let (m, n, k) = (56, 128, 256);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 83) as f32 - 41.0) * 0.02).collect();
        let packed_a = pack_a_asm_f32_avx512(&a, m, k);
        let packed_b = pack_b_asm_f32_avx512(&b, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        let mut c_reg = vec![0.0f32; m * n];
        gemm_prepacked_ab_asm_f32_avx512(packed_a.as_slice(), &packed_b, &mut c_ab, m, n, k);
        gemm_asm_f32_avx512(&a, &b, &mut c_reg, m, n, k);
        check_close(&c_ab, &c_reg, 1e-5, "avx512_prepacked_ab_vs_regular");
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod pack_tests {
    use super::*;

    /// Scalar reference pack_a (known correct)
    fn pack_a_ref(a: &[f32], lda: usize, mc: usize, kc: usize, mr: usize) -> Vec<f32> {
        let mut packed = vec![0.0f32; mc.div_ceil(mr) * mr * kc];
        let mut p = 0usize;
        let mut i = 0usize;
        while i + mr <= mc {
            for r in 0..mr {
                for k in 0..kc {
                    packed[p + k * mr + r] = a[(i + r) * lda + k];
                }
            }
            p += mr * kc;
            i += mr;
        }
        if i < mc {
            let rem = mc - i;
            for r in 0..rem {
                for k in 0..kc {
                    packed[p + k * mr + r] = a[(i + r) * lda + k];
                }
            }
        }
        packed
    }

    #[test]
    fn test_pack_a_avx2_vs_scalar() {
        let mr = 6usize;
        // Test various mc/kc combinations including non-multiples of 8
        for &(mc, kc) in &[(6, 8), (6, 16), (12, 296), (128, 296), (6, 7), (12, 9), (128, 216)] {
            let lda = kc;
            let a: Vec<f32> = (0..mc * lda).map(|i| (i as f32) * 0.01 - 5.0).collect();
            
            let expected = pack_a_ref(&a, lda, mc, kc, mr);
            
            let mut packed = vec![0.0f32; expected.len()];
            unsafe {
                gllm_pack_a_f32(a.as_ptr(), lda, packed.as_mut_ptr(), mc, kc, mr);
            }
            
            for idx in 0..expected.len() {
                let diff = (packed[idx] - expected[idx]).abs();
                assert!(
                    diff < 1e-6,
                    "pack_a mismatch at idx={} for mc={} kc={}: got={}, expected={}, diff={}",
                    idx, mc, kc, packed[idx], expected[idx], diff,
                );
            }
            eprintln!("pack_a mc={} kc={}: OK", mc, kc);
        }
    }

    #[test]
    fn test_pack_b_correctness() {
        let nr = 16usize;
        for &(kc, nc) in &[(8usize, 16usize), (296, 256), (216, 256), (8, 15), (296, 17)] {
            let ldb = nc;
            let b: Vec<f32> = (0..kc * ldb).map(|i| (i as f32) * 0.01 - 3.0).collect();
            
            // Reference pack_b
            let num_panels = nc.div_ceil(nr);
            let mut expected = vec![0.0f32; num_panels * kc * nr];
            let mut p = 0usize;
            let mut j = 0usize;
            while j + nr <= nc {
                for k in 0..kc {
                    for c in 0..nr {
                        expected[p + k * nr + c] = b[k * ldb + j + c];
                    }
                }
                p += kc * nr;
                j += nr;
            }
            if j < nc {
                let rem = nc - j;
                for k in 0..kc {
                    for c in 0..rem {
                        expected[p + k * nr + c] = b[k * ldb + j + c];
                    }
                }
            }
            
            let mut packed = vec![0.0f32; expected.len()];
            unsafe {
                gllm_pack_b_f32(b.as_ptr(), ldb, packed.as_mut_ptr(), kc, nc, nr);
            }
            
            for idx in 0..expected.len() {
                let diff = (packed[idx] - expected[idx]).abs();
                assert!(
                    diff < 1e-6,
                    "pack_b mismatch at idx={} for kc={} nc={}: got={}, expected={}, diff={}",
                    idx, kc, nc, packed[idx], expected[idx], diff,
                );
            }
            eprintln!("pack_b kc={} nc={}: OK", kc, nc);
        }
    }
}
