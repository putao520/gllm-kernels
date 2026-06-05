pub mod math;
// Expand AVX2 float-32 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx2_f32, avx2, f32);
// Expand AVX2 f16 implementations (F16C load/store, compute in f32)
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx2_f16, avx2, f16);
// Expand AVX2 bf16 implementations (bit-shift load/store, compute in f32)
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx2_bf16, avx2, bf16);

#[cfg(test)]
mod tests;
