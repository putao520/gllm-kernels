use std::collections::HashMap;
use crate::compiler::graph::OpKind;
use crate::compiler::trace::{build_quant_gather_trace, classify_pattern, ComputePattern, OpTrace, ReductionSecondPass, ScalarFnSignature, ScalarParam, TraceOp, ValueId};
use crate::compiler::symexec::SymExecError;
use crate::types::DType;

// 代码组织 (include! 模式 — 编译为单模块，物理分散到 5 个片段):
// - `registry_fragments/types.inc.rs`      — OpKindKey, RegistryError
// - `registry_fragments/core.inc.rs`       — ScalarOpRegistry struct + core methods
// - `registry_fragments/defaults.inc.rs`   — with_defaults() registration body
// - `registry_fragments/auto.inc.rs`       — auto_register methods
// - `registry_fragments/tests.inc.rs`      — test module

include!("registry_fragments/types.inc.rs");
include!("registry_fragments/core.inc.rs");
include!("registry_fragments/defaults.inc.rs");
include!("registry_fragments/auto.inc.rs");

#[cfg(test)]
include!("registry_fragments/tests.inc.rs");
