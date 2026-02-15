pub mod math;

// Expand AVX-512 float-32 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_f32, avx512, f32);
// Expand AVX-512 f16 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_f16, avx512, f16);
// Expand AVX-512 bf16 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_bf16, avx512, bf16);
