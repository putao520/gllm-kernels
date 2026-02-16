//! Macro system for gllm-kernels.
//!
//! Follows strict 4-layer architecture:
//! 1. simd_primitive! (Hardware Primitives)
//! 2. operator_templates! (Operator Logic)
//! 3. quant_primitive! (Quantization Logic)
//! 4. expand_all! (Batch Expansion)

#[macro_use]
pub mod gemm_epilogue;
#[macro_use]
pub mod simd_primitive;
#[macro_use]
pub mod matmul_x86;
#[macro_use]
pub mod matmul_x86_bf16;
#[macro_use]
pub mod matmul_neon;
#[macro_use]
pub mod operator_templates;
#[macro_use]
pub mod expand;
#[macro_use]
pub mod quant_primitive;
