
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast vectorized exp(x) for AVX2+FMA.
/// Cephes-style degree-5 polynomial with Cody-Waite range reduction.
/// Input clamped to [-88.376, 88.376] to avoid NaN/Inf.
/// Max error: ~1 ULP.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn avx2_exp_f32(x: __m256) -> __m256 {
    // Clamp input to avoid overflow/underflow in 2^k computation
    let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.376_f32)), _mm256_set1_ps(88.376_f32));

    let v_log2e = _mm256_set1_ps(1.442_695_04_f32);
    let v_127 = _mm256_set1_epi32(127);

    // Cody-Waite range reduction: ln2 = c1 + c2 (c1 exact in float)
    let c1 = _mm256_set1_ps(-0.693_359_375_f32);
    let c2 = _mm256_set1_ps(2.121_944_4e-4_f32);

    // k = round(x * log2e)
    let t = _mm256_mul_ps(x, v_log2e);
    let k = _mm256_cvtps_epi32(_mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let k_ps = _mm256_cvtepi32_ps(k);

    // y = x - k*ln2 (two-step for precision)
    let mut y = x;
    y = _mm256_fmadd_ps(k_ps, c1, y);
    y = _mm256_fmadd_ps(k_ps, c2, y);

    // Degree-5 minimax polynomial (Horner's method)
    let p0 = _mm256_set1_ps(1.987_569_15E-4);
    let p1 = _mm256_set1_ps(1.398_199_950_7E-3);
    let p2 = _mm256_set1_ps(8.333_451_907_3E-3);
    let p3 = _mm256_set1_ps(4.166_579_589_4E-2);
    let p4 = _mm256_set1_ps(1.666_666_545_9E-1);
    let p5 = _mm256_set1_ps(5.000_000_120_1E-1);

    let mut p = p0;
    p = _mm256_fmadd_ps(p, y, p1);
    p = _mm256_fmadd_ps(p, y, p2);
    p = _mm256_fmadd_ps(p, y, p3);
    p = _mm256_fmadd_ps(p, y, p4);
    p = _mm256_fmadd_ps(p, y, p5);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0));

    // 2^k via IEEE-754 exponent manipulation
    let v_exp = _mm256_slli_epi32(_mm256_add_epi32(k, v_127), 23);
    let fact = _mm256_castsi256_ps(v_exp);

    _mm256_mul_ps(p, fact)
}
