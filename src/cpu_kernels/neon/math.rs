#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
    // Fallback implementation: unpack, exp, pack
    let mut tmp = [0.0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), x);
    for i in 0..4 {
        tmp[i] = tmp[i].exp();
    }
    vld1q_f32(tmp.as_ptr())
}
