//! AArch64 hand-written assembly microkernels.

pub mod gemm_neon;
pub mod gemm_driver;

pub use gemm_neon::{MR, NR, gemm_kernel_8x12_f32};
pub use gemm_driver::{gemm_asm_f32, gemm_bias_asm_f32};
