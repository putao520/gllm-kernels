// Expand NEON float-32 implementations
#[cfg(target_arch = "aarch64")]
crate::expand_isa_impls!(neon_f32, neon, f32);
