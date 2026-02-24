//! AArch64 hand-written assembly microkernels.

pub mod gemm_neon;
pub mod gemm_driver;
pub mod quant_gemv;

pub use gemm_neon::{MR, NR, gemm_kernel_8x12_f32};
pub use gemm_driver::{gemm_asm_f32, gemm_bias_asm_f32, pack_b_asm_f32_neon, gemm_prepacked_asm_f32, gemm_bias_prepacked_asm_f32};
#[cfg(target_arch = "aarch64")]
pub use gemm_driver::{gllm_pack_a_f32_neon, gllm_pack_b_f32_neon};
