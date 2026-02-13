
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn avx2_exp_f32(x: __m256) -> __m256 {
    // Constants
    let v_log2e = _mm256_set1_ps(1.44269504089f32);
    // let v_ln2 = _mm256_set1_ps(0.69314718056f32); // Unused
    let v_127 = _mm256_set1_epi32(127);
    
    // Coefficients for polynomial P(y)
    let c1 = _mm256_set1_ps(-0.693359375f32); // ln2_hi ? Simplified
    let c2 = _mm256_set1_ps(2.12194440e-4f32); // ln2_lo
    
    // Degree 3 or something?
    // Let's use a standard fast approx found in effectively every SIMD lib (e.g. fpl_ll)
    // exp(x) = 2^k * 2^f
    
    // k = round(x * log2e)
    let t = _mm256_mul_ps(x, v_log2e);
    let k = _mm256_cvtps_epi32(_mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    let k_ps = _mm256_cvtepi32_ps(k);
    
    // Range reduction: y = x - k * ln2
    // Use multi-step for precision
    // y = x - k * c1 - k * c2
    let mut y = x;
    y = _mm256_fmadd_ps(k_ps, c1, y);
    y = _mm256_fmadd_ps(k_ps, c2, y);
    
    // Polynomial approximation (Degree 5/6)
    // p = ...
     let p0 = _mm256_set1_ps(1.9875691500E-4);
     let p1 = _mm256_set1_ps(1.3981999507E-3);
     let p2 = _mm256_set1_ps(8.3334519073E-3);
     let p3 = _mm256_set1_ps(4.1665795894E-2);
     let p4 = _mm256_set1_ps(1.6666665459E-1);
     let p5 = _mm256_set1_ps(5.0000001201E-1);
     
     let mut p = p0;
     p = _mm256_fmadd_ps(p, y, p1);
     p = _mm256_fmadd_ps(p, y, p2);
     p = _mm256_fmadd_ps(p, y, p3);
     p = _mm256_fmadd_ps(p, y, p4);
     p = _mm256_fmadd_ps(p, y, p5);
     p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0)); // + y
     p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0)); // + 1
     
     // 2^k
     let v_exp = _mm256_slli_epi32(_mm256_add_epi32(k, v_127), 23);
     let fact = _mm256_castsi256_ps(v_exp);
     
     _mm256_mul_ps(p, fact)
}
