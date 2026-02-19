//! x86_64 hand-written assembly microkernels.

pub mod gemm_avx2;
pub mod gemm_avx512;
pub mod gemm_driver;
pub mod quant_gemv;

pub use gemm_avx2::{MR as MR_AVX2, NR as NR_AVX2};
pub use gemm_avx512::{MR as MR_AVX512, NR as NR_AVX512};
pub use gemm_driver::{gemm_asm_f32_avx2, gemm_asm_f32_avx512};
pub use gemm_driver::{gemm_bias_asm_f32_avx2, gemm_bias_asm_f32_avx512};
pub use gemm_driver::{pack_b_asm_f32_avx2, pack_b_asm_f32_avx512};
pub use gemm_driver::{gemm_prepacked_asm_f32_avx2, gemm_prepacked_asm_f32_avx512};
pub use gemm_driver::{gemm_bias_prepacked_asm_f32_avx2, gemm_bias_prepacked_asm_f32_avx512};
