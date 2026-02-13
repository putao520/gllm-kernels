pub mod math;
// Expand AVX2 float-32 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx2_f32, avx2, f32);

#[cfg(test)]
mod tests;
