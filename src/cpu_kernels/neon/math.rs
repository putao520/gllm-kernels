#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON polynomial exp approximation (matches AVX2 version).
/// Range reduction x → k*ln2 + y, then degree-5 polynomial on y, then 2^k scale.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
    let v_log2e = vdupq_n_f32(1.44269504089f32);
    let v_127 = vdupq_n_s32(127);

    // Cody-Waite range reduction constants
    let c1 = vdupq_n_f32(-0.693359375f32);
    let c2 = vdupq_n_f32(2.12194440e-4f32);

    // k = round(x * log2e)
    let t = vmulq_f32(x, v_log2e);
    let k = vcvtq_s32_f32(vrndnq_f32(t));
    let k_ps = vcvtq_f32_s32(k);

    // Range reduction: y = x - k*c1 - k*c2
    let mut y = vfmaq_f32(x, k_ps, c1);
    y = vfmaq_f32(y, k_ps, c2);

    // Degree-5 polynomial: Horner's method
    let p0 = vdupq_n_f32(1.9875691500E-4);
    let p1 = vdupq_n_f32(1.3981999507E-3);
    let p2 = vdupq_n_f32(8.3334519073E-3);
    let p3 = vdupq_n_f32(4.1665795894E-2);
    let p4 = vdupq_n_f32(1.6666665459E-1);
    let p5 = vdupq_n_f32(5.0000001201E-1);
    let one = vdupq_n_f32(1.0);

    let mut p = p0;
    p = vfmaq_f32(p1, p, y);
    p = vfmaq_f32(p2, p, y);
    p = vfmaq_f32(p3, p, y);
    p = vfmaq_f32(p4, p, y);
    p = vfmaq_f32(p5, p, y);
    p = vfmaq_f32(one, p, y);
    p = vfmaq_f32(one, p, y);

    // 2^k via IEEE-754 exponent manipulation
    let v_exp = vshlq_n_s32::<23>(vaddq_s32(k, v_127));
    let fact = vreinterpretq_f32_s32(v_exp);

    vmulq_f32(p, fact)
}
