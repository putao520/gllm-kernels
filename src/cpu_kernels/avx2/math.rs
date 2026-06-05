
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Generate the unified exp implementation via macro
#[cfg(target_arch = "x86_64")]
crate::define_exp_f32!(avx2);

// Generate the fast exp implementation via macro
#[cfg(target_arch = "x86_64")]
crate::define_exp_fast_f32!(avx2);

/// Fast vectorized exp(x) for AVX2+FMA.
/// Cephes-style degree-5 polynomial with Cody-Waite range reduction.
/// Input clamped to [-88.376, 88.376] to avoid NaN/Inf.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn avx2_exp_f32(x: __m256) -> __m256 {
    exp_f32_impl(x)
}

/// Fast exp(x) for AVX2+FMA — degree-3 polynomial, ~12-bit accuracy.
/// 2-3x faster than full exp. Use for softmax/sigmoid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn avx2_exp_fast_f32(x: __m256) -> __m256 {
    exp_fast_f32_impl(x)
}
