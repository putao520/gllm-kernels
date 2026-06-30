//! Register VM E2E 测试——完整管线验证。
//!
//! compile_layer → CompiledLayer → 执行 → 数值验证
//!
//! 代码组织 (include! 模式):
//! - `e2e_tests_fragments/tests.inc.rs`              — 核心编译+执行测试
//! - `e2e_tests_fragments/gather_tests.inc.rs`       — Gather 编译测试
//! - `e2e_tests_fragments/quant_gemv_tests.inc.rs`   — 量化 GEMV JIT 执行测试
//! - `e2e_tests_fragments/p05_dtype_matrix_tests.inc.rs` — P0.5 dtype-matrix 防假完成护栏

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
include!("e2e_tests_fragments/tests.inc.rs");

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
include!("e2e_tests_fragments/gather_tests.inc.rs");

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
include!("e2e_tests_fragments/quant_gemv_tests.inc.rs");

#[cfg(test)]
include!("e2e_tests_fragments/p05_dtype_matrix_tests.inc.rs");
