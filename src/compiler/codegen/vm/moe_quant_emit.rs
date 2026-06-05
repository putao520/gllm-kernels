//! MoE routing/dispatch + quantized GEMM inline lowering.
//!
//! 代码组织 (include! 模式):
//! - `moe_quant_fragments/moe_emit.inc.rs`       — MoE functions + QuantGemmPlan struct
//! - `moe_quant_fragments/quant_gemm.inc.rs`     — quant GEMM + DequantFMAPath
//! - `moe_quant_fragments/template_driven.inc.rs` — template-driven MoE bridge
//! - `moe_quant_fragments/tests.inc.rs`          — test module

use super::instr::*;
use super::isa_hook;
use crate::compiler::trace::{QuantPrecision, TraceOp, ReduceKind, ValueId};
use crate::types::CompilerError;
use crate::dispatch::device_profile::DotProductCap;

include!("moe_quant_fragments/moe_emit.inc.rs");
include!("moe_quant_fragments/quant_gemm.inc.rs");
include!("moe_quant_fragments/template_driven.inc.rs");

#[cfg(test)]
include!("moe_quant_fragments/tests.inc.rs");
