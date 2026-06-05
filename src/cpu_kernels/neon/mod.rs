#[cfg(target_arch = "aarch64")]
pub mod math;

// Expand NEON float-32 implementations
#[cfg(target_arch = "aarch64")]
crate::expand_isa_impls!(neon_f32, neon, f32);
// Expand NEON f16 implementations
#[cfg(target_arch = "aarch64")]
crate::expand_isa_impls!(neon_f16, neon, f16);
// Expand NEON bf16 implementations
#[cfg(target_arch = "aarch64")]
crate::expand_isa_impls!(neon_bf16, neon, bf16);
