
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast vectorized exp(x) for AVX-512.
/// Cephes-style degree-5 polynomial with Cody-Waite range reduction.
/// Input clamped to [-88.376, 88.376] to avoid NaN/Inf.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_exp_f32(x: __m512) -> __m512 {
    // Clamp input to avoid overflow/underflow
    let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-88.376_f32)), _mm512_set1_ps(88.376_f32));

    let v_log2e = _mm512_set1_ps(1.442_695_04_f32);
    let v_127 = _mm512_set1_epi32(127);

    let c1 = _mm512_set1_ps(-0.693_359_375_f32);
    let c2 = _mm512_set1_ps(2.121_944_4e-4_f32);

    let t = _mm512_mul_ps(x, v_log2e);
    let k = _mm512_cvtps_epi32(_mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let k_ps = _mm512_cvtepi32_ps(k);

    let mut y = x;
    y = _mm512_fmadd_ps(k_ps, c1, y);
    y = _mm512_fmadd_ps(k_ps, c2, y);

    let p0 = _mm512_set1_ps(1.987_569_15E-4);
    let p1 = _mm512_set1_ps(1.398_199_950_7E-3);
    let p2 = _mm512_set1_ps(8.333_451_907_3E-3);
    let p3 = _mm512_set1_ps(4.166_579_589_4E-2);
    let p4 = _mm512_set1_ps(1.666_666_545_9E-1);
    let p5 = _mm512_set1_ps(5.000_000_120_1E-1);

    let mut p = p0;
    p = _mm512_fmadd_ps(p, y, p1);
    p = _mm512_fmadd_ps(p, y, p2);
    p = _mm512_fmadd_ps(p, y, p3);
    p = _mm512_fmadd_ps(p, y, p4);
    p = _mm512_fmadd_ps(p, y, p5);
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps(1.0));
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps(1.0));

    let v_exp = _mm512_slli_epi32(_mm512_add_epi32(k, v_127), 23);
    let fact = _mm512_castsi512_ps(v_exp);

    _mm512_mul_ps(p, fact)
}
