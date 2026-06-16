//! CompilerGraph — DAG representation for the JIT inference compiler.
//!
//! The graph captures the computation of a single transformer layer as a
//! directed acyclic graph of typed operations. Each operation reads from
//! input tensors and produces output tensors. Tensors carry shape metadata
//!
//! 代码组织 (include! 模式):
//! - `graph_fragments/types.inc.rs`    — SymDim, ShapeBinding, WeightLayout, small types
//! - `graph_fragments/op_kind.inc.rs`  — OpKind enum + QTap + CompilerOp/CompilerGraph structs
//! - `graph_fragments/graph_impl.inc.rs` — impl CompilerGraph + Default + Display
//! - `graph_fragments/tests.inc.rs`    — test module

use std::hash::Hash;

include!("graph_fragments/types.inc.rs");
include!("graph_fragments/op_kind.inc.rs");
include!("graph_fragments/op.inc.rs");
include!("graph_fragments/graph_impl.inc.rs");

#[cfg(test)]
include!("graph_fragments/tests.inc.rs");
