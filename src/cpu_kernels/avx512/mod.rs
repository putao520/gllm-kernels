// Expand AVX-512 float-32 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_f32, avx512, f32);
