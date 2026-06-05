//! x86_64 GEMM drivers using hand-written AVX2/AVX-512 assembly microkernels.
//!
//! This module provides the complete GEMM implementation:
//! - Pack A into column-major MR-wide panels
//! - Pack B into row-major NR-wide panels
//! - Tile the M/N/K dimensions with cache-aware blocking
//! - Call the assembly microkernel for each tile
//!
//! Two entry points:
//! - `gemm_asm_f32_avx2`:   uses the 6x16 AVX2 microkernel
//! - `gemm_asm_f32_avx512`: uses the 14x32 AVX-512 microkernel
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 3 个片段):
//! - `gemm_driver/pack_driver.inc.rs` — pack functions + single/MT BLIS driver
//! - `gemm_driver/prepacked.inc.rs`   — prepacked driver variants + public API
//! - `gemm_driver/tests.inc.rs`       — test modules

/// Size of a single `f32` element in bytes.
const F32_BYTES: usize = std::mem::size_of::<f32>();

include!("gemm_driver/pack_driver.inc.rs");
include!("gemm_driver/prepacked.inc.rs");

#[cfg(test)]
include!("gemm_driver/tests.inc.rs");
