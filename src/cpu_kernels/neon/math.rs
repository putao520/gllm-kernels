#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// Generate the unified exp implementation via macro
#[cfg(target_arch = "aarch64")]
crate::define_exp_f32!(neon);

/// NEON polynomial exp approximation.
/// Range reduction x → k*ln2 + y, then degree-5 polynomial on y, then 2^k scale.
/// Input clamped to [-88.376, 88.376] to avoid NaN/Inf.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
    exp_f32_impl(x)
}
