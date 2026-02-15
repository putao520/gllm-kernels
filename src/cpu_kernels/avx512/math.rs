
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_exp_f32(x: __m512) -> __m512 {
    let v_log2e = _mm512_set1_ps(1.44269504089f32);
    let v_127 = _mm512_set1_epi32(127);

    // Cody-Waite range reduction constants
    let c1 = _mm512_set1_ps(-0.693359375f32);
    let c2 = _mm512_set1_ps(2.12194440e-4f32);

    // k = round(x * log2e)
    let t = _mm512_mul_ps(x, v_log2e);
    let k = _mm512_cvtps_epi32(_mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let k_ps = _mm512_cvtepi32_ps(k);

    // Range reduction: y = x - k*c1 - k*c2
    let mut y = x;
    y = _mm512_fmadd_ps(k_ps, c1, y);
    y = _mm512_fmadd_ps(k_ps, c2, y);

    // Degree-5 polynomial (Horner's method)
    let p0 = _mm512_set1_ps(1.9875691500E-4);
    let p1 = _mm512_set1_ps(1.3981999507E-3);
    let p2 = _mm512_set1_ps(8.3334519073E-3);
    let p3 = _mm512_set1_ps(4.1665795894E-2);
    let p4 = _mm512_set1_ps(1.6666665459E-1);
    let p5 = _mm512_set1_ps(5.0000001201E-1);

    let mut p = p0;
    p = _mm512_fmadd_ps(p, y, p1);
    p = _mm512_fmadd_ps(p, y, p2);
    p = _mm512_fmadd_ps(p, y, p3);
    p = _mm512_fmadd_ps(p, y, p4);
    p = _mm512_fmadd_ps(p, y, p5);
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps(1.0));
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps(1.0));

    // 2^k via IEEE-754 exponent manipulation
    let v_exp = _mm512_slli_epi32(_mm512_add_epi32(k, v_127), 23);
    let fact = _mm512_castsi512_ps(v_exp);

    _mm512_mul_ps(p, fact)
}
