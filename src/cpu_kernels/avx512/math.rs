
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Generate the unified exp implementation via macro
#[cfg(target_arch = "x86_64")]
crate::define_exp_f32!(avx512);

/// Fast vectorized exp(x) for AVX-512.
/// Cephes-style degree-5 polynomial with Cody-Waite range reduction.
/// Input clamped to [-88.376, 88.376] to avoid NaN/Inf.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_exp_f32(x: __m512) -> __m512 {
    exp_f32_impl(x)
}
