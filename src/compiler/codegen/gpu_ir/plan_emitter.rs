//! Unified `gpu_emit_plan<D>` — generic ComputePattern dispatch.
//!
//! Phase 3 of the GPU codegen unification. Replaces the triplicated
//! `emit_plan` match arms in `ptx.rs`, `hip.rs`, and `air.rs` with a
//! single generic function parameterised over `GpuDialect`.

use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::ComputePattern;
use super::trace_emitter::GpuDialect;

/// Emit GPU code for every group in `plan`, dispatching on `ComputePattern`.
///
/// The caller is responsible for:
/// 1. Constructing the dialect (`PtxDialect`, `HipDialect`, `MslDialect`).
/// 2. Emitting the file header via `dialect.emit_header()` before calling this.
/// 3. Wrapping the returned bytes in a `CodegenOutput`.
///
/// This function appends kernel text to `out` and returns `Ok(())` on success.
pub fn gpu_emit_plan<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    plan: &FusionPlan,
    graph: &CompilerGraph,
    registry: Option<&ScalarOpRegistry>,
) -> Result<(), String> {
    for group in &plan.groups {
        let anchor_op = graph.op(group.anchor).ok_or_else(|| {
            format!("gpu_emit_plan: anchor op {:?} not found in graph", group.anchor)
        })?;

        let kernel_name = format!("group_{}", group.id);
        let op_kind = &anchor_op.kind;

        // Reshape/Transpose are metadata-only — NOP on GPU
        if matches!(op_kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
            continue;
        }

        let registry = registry.ok_or_else(|| {
            format!("gpu_emit_plan: ScalarOpRegistry required for {:?}", op_kind)
        })?;
        let key = ScalarOpRegistry::key_from_op_kind(op_kind);
        let trace = registry.get_trace(&key).ok_or_else(|| {
            format!("gpu_emit_plan: no OpTrace for {:?}", op_kind)
        })?;

        match &trace.pattern {
            ComputePattern::Elementwise { body } => {
                dialect.emit_elementwise_kernel(out, &kernel_name, body);
            }
            ComputePattern::BinaryElementwise { body } => {
                dialect.emit_binary_elementwise_kernel(out, &kernel_name, body);
            }
            ComputePattern::NormLike { reduce, finalize, transform } => {
                let eps_override = match op_kind {
                    OpKind::RmsNorm { eps } => Some(*eps),
                    OpKind::LayerNorm { eps } => Some(*eps),
                    _ => None,
                };
                let has_weight = matches!(
                    op_kind,
                    OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }
                );
                let has_bias = matches!(op_kind, OpKind::LayerNorm { .. });
                dialect.emit_normlike_kernel(
                    out,
                    &kernel_name,
                    reduce,
                    finalize,
                    transform,
                    has_weight,
                    has_bias,
                    eps_override,
                );
            }
            ComputePattern::Reduction { combine, identity, .. } => {
                match op_kind {
                    OpKind::Softmax => {
                        dialect.emit_softmax_kernel(out, &kernel_name);
                    }
                    OpKind::MeanPool { seq_len, hidden } => {
                        dialect.emit_meanpool_kernel(out, &kernel_name, *seq_len, *hidden);
                    }
                    _ => {
                        dialect.emit_reduction_kernel(
                            out, &kernel_name, *identity, combine,
                        )?;
                    }
                }
            }
            ComputePattern::Gemm => {
                match op_kind {
                    OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                        dialect.emit_mha_kernel(
                            out, &kernel_name, *seq_len, *num_heads, *head_dim,
                        );
                    }
                    _ => {
                        dialect.emit_gemm_kernel(out, &kernel_name, op_kind)?;
                    }
                }
            }
            ComputePattern::Injective { body, num_inputs, num_outputs } => {
                match op_kind {
                    OpKind::RoPE { head_dim, theta } => {
                        dialect.emit_rope_kernel(out, &kernel_name, *head_dim, *theta);
                    }
                    _ => {
                        if body.is_empty() {
                            continue;
                        }
                        dialect.emit_injective_kernel(
                            out, &kernel_name, body, *num_inputs, *num_outputs,
                        )?;
                    }
                }
            }
            ComputePattern::QuantDecode { block_size, .. } => {
                match op_kind {
                    OpKind::Dequantize { num_elements, block_size: bs, bits } => {
                        dialect.emit_dequantize_kernel(
                            out, &kernel_name, *num_elements, *bs, *bits as u8,
                        );
                    }
                    _ => {
                        return Err(format!(
                            "gpu_emit_plan: unsupported QuantDecode op {:?} (block_size={})",
                            op_kind, block_size
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "jit-hip"))]
mod tests {
    use super::*;
    use super::super::trace_emitter::HipDialect;
    use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
    use crate::compiler::graph::{CompilerGraph, OpId, OpKind};
    use crate::compiler::registry::{OpKindKey, ScalarOpRegistry};
    use crate::compiler::trace::{ComputePattern, OpTrace, ScalarFnSignature, ScalarParam, TraceOp};
    use crate::types::DType;
    use std::collections::HashMap;

    /// Build a minimal graph with a single op and return (graph, op_id).
    fn one_op_graph(kind: OpKind) -> (CompilerGraph, OpId) {
        let mut g = CompilerGraph::new();
        let t_in = g.add_tensor("in", vec![1024], DType::F32);
        let t_out = g.add_tensor("out", vec![1024], DType::F32);
        let op = g.add_op(kind, vec![t_in], vec![t_out], "test_op");
        (g, op)
    }

    /// Build a FusionPlan with a single standalone group.
    fn one_group_plan(anchor: OpId) -> FusionPlan {
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![anchor],
            }],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(anchor, 0);
                m
            },
        }
    }

    /// Dummy signature for test traces.
    fn dummy_sig() -> ScalarFnSignature {
        ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        }
    }

    /// SiLU body: x / (1 + exp(-x))
    fn silu_body() -> Vec<TraceOp> {
        vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
            TraceOp::Exp(1),
            TraceOp::Const(1.0),
            TraceOp::Add(2, 3),
            TraceOp::Div(0, 4),
        ]
    }

    /// Simple add body: a + b
    fn add_body() -> Vec<TraceOp> {
        vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(0, 1),
        ]
    }

    // ── Elementwise dispatch ────────────────────────────────────────

    #[test]
    fn elementwise_dispatches_to_injective_codegen() {
        let (graph, op) = one_op_graph(OpKind::Silu);
        let plan = one_group_plan(op);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Silu, OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body: silu_body() },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("expf("), "missing expf (SiLU body):\n{out}");
    }

    // ── BinaryElementwise dispatch ──────────────────────────────────

    #[test]
    fn binary_elementwise_dispatches() {
        let (graph, op) = one_op_graph(OpKind::Add);
        let plan = one_group_plan(op);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Add, OpTrace {
            op_kind: OpKind::Add,
            pattern: ComputePattern::BinaryElementwise { body: add_body() },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(
            out.contains("input0") || out.contains("__restrict__ A"),
            "missing first input param:\n{out}",
        );
        assert!(
            out.contains("input1") || out.contains("__restrict__ B"),
            "missing second input param:\n{out}",
        );
    }

    // ── NormLike dispatch ───────────────────────────────────────────

    #[test]
    fn normlike_dispatches_rmsnorm() {
        let (graph, op) = one_op_graph(OpKind::RmsNorm { eps: 1e-6 });
        let plan = one_group_plan(op);

        let reduce = vec![TraceOp::Input(0), TraceOp::Mul(0, 0)];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(0)];
        let transform = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Mul(0, 1),
            TraceOp::Input(2),
            TraceOp::Mul(2, 3),
        ];

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::RmsNorm, OpTrace {
            op_kind: OpKind::RmsNorm { eps: 1e-6 },
            pattern: ComputePattern::NormLike { reduce, finalize, transform },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("__shared__ float smem["), "missing shared mem:\n{out}");
        assert!(out.contains("rsqrtf("), "missing rsqrt:\n{out}");
        assert!(out.contains("weight"), "missing weight param:\n{out}");
    }

    // ── Softmax dispatch ────────────────────────────────────────────

    #[test]
    fn softmax_dispatches_via_reduction() {
        let (graph, op) = one_op_graph(OpKind::Softmax);
        let plan = one_group_plan(op);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Softmax, OpTrace {
            op_kind: OpKind::Softmax,
            pattern: ComputePattern::Reduction {
                identity: f64::NEG_INFINITY,
                combine: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Max(0, 1)],
                second_pass: None,
                normalize: None,
            },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        // Softmax uses the dedicated softmax emitter, not generic reduction
        assert!(out.contains("fmaxf(") || out.contains("expf("),
            "missing softmax ops:\n{out}");
    }

    // ── Gemm dispatch ───────────────────────────────────────────────

    #[test]
    fn gemm_dispatches() {
        let kind = OpKind::Gemm { m: 64, n: 64, k: 64 };
        let (graph, op) = one_op_graph(kind.clone());
        let plan = one_group_plan(op);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Gemm, OpTrace {
            op_kind: kind,
            pattern: ComputePattern::Gemm,
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
    }

    // ── Reshape NOP skip ────────────────────────────────────────────

    #[test]
    fn reshape_skipped_no_kernel() {
        let kind = OpKind::Reshape { target_shape: vec![32, 32] };
        let (graph, op) = one_op_graph(kind);
        let plan = one_group_plan(op);

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let header_len = out.len();

        // No registry needed — reshape should be skipped before registry lookup
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, None);

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        // Nothing should have been appended after the header
        assert_eq!(out.len(), header_len, "reshape should produce no kernel output");
    }

    // ── Injective dispatch ──────────────────────────────────────────

    #[test]
    fn injective_dispatches_concat() {
        let (graph, op) = one_op_graph(OpKind::Add); // reuse Add as anchor
        let plan = one_group_plan(op);

        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(0, 1),
        ];

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Add, OpTrace {
            op_kind: OpKind::Add,
            pattern: ComputePattern::Injective {
                body,
                num_inputs: 2,
                num_outputs: 1,
            },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("input0"), "missing input0 param:\n{out}");
        assert!(out.contains("input1"), "missing input1 param:\n{out}");
        assert!(out.contains("output0"), "missing output0 param:\n{out}");
    }

    // ── Injective with empty body is skipped ────────────────────────

    #[test]
    fn injective_empty_body_skipped() {
        let (graph, op) = one_op_graph(OpKind::Add);
        let plan = one_group_plan(op);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Add, OpTrace {
            op_kind: OpKind::Add,
            pattern: ComputePattern::Injective {
                body: vec![],
                num_inputs: 1,
                num_outputs: 1,
            },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let header_len = out.len();

        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg));

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert_eq!(out.len(), header_len, "empty injective body should produce no kernel");
    }
}
