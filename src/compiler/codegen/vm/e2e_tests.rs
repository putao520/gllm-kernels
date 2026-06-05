//! Register VM E2E 测试——完整管线验证。
//!
//! compile_layer → CompiledLayer → 执行 → 数值验证
//!
//! 代码组织 (include! 模式):
//! - `e2e_tests_fragments/tests.inc.rs`        — 核心编译+执行测试
//! - `e2e_tests_fragments/gather_tests.inc.rs` — Gather 编译测试

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
include!("e2e_tests_fragments/tests.inc.rs");

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
include!("e2e_tests_fragments/gather_tests.inc.rs");
