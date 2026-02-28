//! Platform-agnostic algorithm layer for code generation.
//!
//! This module contains the shared logic that was previously duplicated across
//! x86_64.rs, aarch64.rs, and aarch64_dynasm.rs:
//!
//! - `emit_plan`: iterate fusion groups and dispatch
//! - `emit_group`: dispatch by FusionMode
//! - `collect_trace_info`: gather TraceOp bodies from the registry
//! - `emit_trace_op`: map a single TraceOp to SimdOps calls
//! - `emit_trace_body_on_reg`: execute a TraceOp body on a virtual register
//! - `emit_epilogue_on_accumulators`: apply epilogue traces to accumulator regs
//! - `emit_vectorized_loop`: emit a SIMD loop with scalar tail
//!
//! All functions are generic over `E: SimdOps`, so they work with any backend.

use super::simd_ops::{SimdOps, VReg};
use super::math_approx;
use crate::compiler::trace::{TraceOp, ComputePattern};
use crate::compiler::fusion::FusionGroup;
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::registry::ScalarOpRegistry;

// ── Trace body collection ───────────────────────────────────────────────────

/// Collected trace info: (body, is_binary) pairs for a fusion group.
pub struct TraceInfo {
    pub bodies: Vec<(Vec<TraceOp>, bool)>,
}

/// Collect TraceOp bodies from the registry for a fusion group's anchor + epilogue ops.
///
/// This logic was duplicated in every backend's `emit_elementwise_chain` and
/// `emit_gemm_with_epilogue`. Now it's shared.
pub fn collect_trace_info(
    group: &FusionGroup,
    graph: &CompilerGraph,
    registry: Option<&ScalarOpRegistry>,
) -> TraceInfo {
    let mut bodies = Vec::new();

    if let Some(reg) = registry {
        // Anchor op
        if let Some(anchor_op) = graph.op(group.anchor) {
            let key = ScalarOpRegistry::key_from_op_kind(&anchor_op.kind);
            if let Some(trace) = reg.get_trace(&key) {
                if let Some(body) = trace.pattern.body() {
                    let is_binary = matches!(
                        trace.pattern,
                        ComputePattern::BinaryElementwise { .. }
                    );
                    bodies.push((body.to_vec(), is_binary));
                }
            }
        }

        // Epilogue ops
        for &epi_id in &group.epilogue {
            if let Some(epi_op) = graph.op(epi_id) {
                let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                if let Some(trace) = reg.get_trace(&key) {
                    if let Some(body) = trace.pattern.body() {
                        let is_binary = matches!(
                            trace.pattern,
                            ComputePattern::BinaryElementwise { .. }
                        );
                        bodies.push((body.to_vec(), is_binary));
                    }
                }
            }
        }
    }

    TraceInfo { bodies }
}

/// Collect epilogue TraceOp bodies (without the anchor) for GEMM epilogue injection.
pub fn collect_epilogue_bodies<'a>(
    group: &FusionGroup,
    graph: &CompilerGraph,
    registry: Option<&'a ScalarOpRegistry>,
) -> Vec<&'a [TraceOp]> {
    let mut bodies = Vec::new();
    if let Some(reg) = registry {
        for &epi_id in &group.epilogue {
            if let Some(epi_op) = graph.op(epi_id) {
                let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                if let Some(trace) = reg.get_trace(&key) {
                    if let Some(body) = trace.pattern.body() {
                        bodies.push(body);
                    }
                }
            }
        }
    }
    bodies
}

/// Check whether any epilogue body references `Input(idx)` with idx > 0,
/// indicating an external tensor (e.g. bias vector).
pub fn epilogue_has_external_input(bodies: &[&[TraceOp]]) -> bool {
    bodies.iter().any(|body| {
        body.iter().any(|op| matches!(op, TraceOp::Input(i) if *i > 0))
    })
}

// ── TraceOp → SimdOps mapping ───────────────────────────────────────────────

/// Map a single TraceOp to SimdOps calls, writing the result to `dst`.
///
/// `reg_map` maps SSA indices to VRegs for previously computed values.
/// `scratch` provides 3 scratch registers for math approximations.
///
/// This replaces the duplicated match blocks in:
/// - emit_trace_ops_avx2 / emit_trace_ops_avx512
/// - emit_trace_ops_neon
pub fn emit_trace_op<E: SimdOps>(
    e: &mut E,
    op: &TraceOp,
    dst: VReg,
    reg_map: &[VReg],
    scratch: [VReg; 3],
) -> Result<(), String> {
    match op {
        TraceOp::Input(_) => {
            // Input is pre-loaded by the caller; zero as placeholder
            e.vzero(dst)?;
        }
        TraceOp::Const(v) => {
            e.vbroadcast_const(dst, *v as f32)?;
        }
        TraceOp::Add(a, b) => {
            e.vadd(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
        TraceOp::Sub(a, b) => {
            e.vsub(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
        TraceOp::Mul(a, b) => {
            e.vmul(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
        TraceOp::Div(a, b) => {
            e.vdiv(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
        TraceOp::Fma(a, b, c) => {
            e.vmov(dst, reg_map[*c as usize])?;
            e.vfmadd231(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
        TraceOp::Neg(a) => {
            e.vneg(dst, reg_map[*a as usize])?;
        }
        TraceOp::Abs(a) => {
            e.vabs(dst, reg_map[*a as usize])?;
        }
        TraceOp::Exp(a) => {
            math_approx::emit_exp(e, dst, reg_map[*a as usize], scratch)?;
        }
        TraceOp::Sqrt(a) => {
            e.vsqrt(dst, reg_map[*a as usize])?;
        }
        TraceOp::Rsqrt(a) => {
            e.vrsqrt(dst, reg_map[*a as usize])?;
        }
        TraceOp::Tanh(a) => {
            math_approx::emit_tanh(e, dst, reg_map[*a as usize], scratch)?;
        }
        TraceOp::Recip(a) => {
            e.vrecip(dst, reg_map[*a as usize])?;
        }
        TraceOp::Log(a) => {
            math_approx::emit_log(e, dst, reg_map[*a as usize], scratch)?;
        }
        TraceOp::Max(a, b) => {
            e.vmax(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
        TraceOp::Min(a, b) => {
            e.vmin(dst, reg_map[*a as usize], reg_map[*b as usize])?;
        }
    }
    Ok(())
}

/// Execute a full TraceOp body, mapping SSA indices to sequential VRegs.
///
/// Returns the VReg holding the final result.
/// `base_vreg` is the starting VReg index for SSA allocation.
/// `scratch` provides 3 scratch registers for math.
///
/// This replaces emit_trace_ops_avx2, emit_trace_ops_avx512, emit_trace_ops_neon.
pub fn emit_trace_body<E: SimdOps>(
    e: &mut E,
    ops: &[TraceOp],
    base_vreg: u8,
    scratch: [VReg; 3],
) -> Result<Vec<VReg>, String> {
    let mut reg_map: Vec<VReg> = Vec::with_capacity(ops.len());

    for (i, op) in ops.iter().enumerate() {
        let dst = VReg(base_vreg + i as u8);
        emit_trace_op(e, op, dst, &reg_map, scratch)?;
        reg_map.push(dst);
    }

    Ok(reg_map)
}

/// Get the element count for a fusion group's anchor op output.
pub fn group_elem_count(group: &FusionGroup, graph: &CompilerGraph) -> usize {
    if let Some(op) = graph.op(group.anchor) {
        if let Some(&out_id) = op.outputs.first() {
            return graph.tensor_numel(out_id).unwrap_or(0);
        }
    }
    0
}

/// Extract GEMM dimensions (m, n, k) from an op, if it's a GEMM/QuantGemm.
pub fn gemm_dims(group: &FusionGroup, graph: &CompilerGraph) -> Option<(usize, usize, usize)> {
    let op = graph.op(group.anchor)?;
    match &op.kind {
        OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
            Some((*m, *n, *k))
        }
        _ => None,
    }
}

/// Check if an op is a GEMM variant.
pub fn is_gemm_op(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Gemm { .. } | OpKind::QuantGemm { .. })
}

/// Extract norm epsilon from an op, if it's RmsNorm.
pub fn norm_eps(graph: &CompilerGraph, op_id: crate::compiler::graph::OpId) -> Option<f32> {
    let op = graph.op(op_id)?;
    match &op.kind {
        OpKind::RmsNorm { eps } => Some(*eps),
        _ => None,
    }
}
