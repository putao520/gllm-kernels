//! Unified `gpu_emit_plan<D>` — generic ComputePattern dispatch.
//!
//! Phase 3 of the GPU codegen unification. Replaces the triplicated
//! `emit_plan` match arms in `ptx.rs`, `hip.rs`, and `air.rs` with a
//! single generic function parameterised over `GpuDialect`.

use crate::compiler::fusion::{FusionMode, FusionPlan};
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::ComputePattern;
use crate::gpu::GpuDeviceProfile;
use super::trace_emitter::GpuDialect;

/// Emit GPU code for every group in `plan`, dispatching on `FusionMode` and
/// `ComputePattern`.
///
/// The caller is responsible for:
/// 1. Constructing the dialect (`PtxDialect`, `HipDialect`, `MslDialect`).
/// 2. Emitting the file header via `dialect.emit_header()` before calling this.
/// 3. Wrapping the returned bytes in a `CodegenOutput`.
///
/// Fusion modes:
/// - `Standalone`: dispatch on the anchor op's `ComputePattern` (default path).
/// - `EpilogueInjection`: emit a fused GEMM + epilogue kernel where activation
///   ops (GELU, SiLU, etc.) are applied on the accumulator in-register before
///   the final store, avoiding an intermediate global memory round-trip.
///
/// This function appends kernel text to `out` and returns `Ok(())` on success.
pub fn gpu_emit_plan<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    plan: &FusionPlan,
    graph: &CompilerGraph,
    registry: Option<&ScalarOpRegistry>,
    gpu_profile: Option<&GpuDeviceProfile>,
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

        // ── FusionMode dispatch ──
        match group.mode {
            FusionMode::EpilogueInjection => {
                emit_epilogue_injection(
                    dialect, out, &kernel_name, group, graph, registry,
                )?;
                continue;
            }
            FusionMode::Standalone
            | FusionMode::LoopFusion
            | FusionMode::QkvSharedInput
            | FusionMode::NormIntoGemm => {
                // Fall through to ComputePattern dispatch below
            }
            FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                let pred_op = graph.op(predecessor).ok_or_else(|| {
                    format!("gpu_emit_plan: TileLevelFusion predecessor {:?} not found", predecessor)
                })?;
                let registry = registry.ok_or(
                    "gpu_emit_plan: ScalarOpRegistry required for TileLevelFusion"
                )?;
                let key = ScalarOpRegistry::key_from_op_kind(&pred_op.kind);
                let trace = registry.get_trace(&key).ok_or_else(|| {
                    format!("gpu_emit_plan: no OpTrace for predecessor {:?}", pred_op.kind)
                })?;
                let (reduce, finalize, transform) = match &trace.pattern {
                    ComputePattern::NormLike { reduce, finalize, transform } => {
                        (reduce.as_slice(), finalize.as_slice(), transform.as_slice())
                    }
                    _ => return Err(format!(
                        "gpu_emit_plan: TileLevelFusion predecessor must be NormLike, got {:?}",
                        trace.pattern
                    )),
                };
                let shared_budget = gpu_profile
                    .map(|p| (p.shared_mem_per_block as usize) * 75 / 100)
                    .unwrap_or(32768);
                dialect.emit_tile_level_fusion_kernel(
                    out, &kernel_name, &pred_op.kind, op_kind,
                    tile_rows, reduce, finalize, transform, shared_budget,
                )?;
                continue;
            }
            FusionMode::ComputeRoot { predecessor } => {
                let pred_op = graph.op(predecessor).ok_or_else(|| {
                    format!("gpu_emit_plan: ComputeRoot predecessor {:?} not found", predecessor)
                })?;
                let registry = registry.ok_or(
                    "gpu_emit_plan: ScalarOpRegistry required for ComputeRoot"
                )?;
                let key = ScalarOpRegistry::key_from_op_kind(&pred_op.kind);
                let trace = registry.get_trace(&key).ok_or_else(|| {
                    format!("gpu_emit_plan: no OpTrace for predecessor {:?}", pred_op.kind)
                })?;
                let (reduce, finalize, transform) = match &trace.pattern {
                    ComputePattern::NormLike { reduce, finalize, transform } => {
                        (reduce.as_slice(), finalize.as_slice(), transform.as_slice())
                    }
                    _ => return Err(format!(
                        "gpu_emit_plan: ComputeRoot predecessor must be NormLike, got {:?}",
                        trace.pattern
                    )),
                };
                let shared_budget = gpu_profile
                    .map(|p| (p.shared_mem_per_block as usize) * 75 / 100)
                    .unwrap_or(32768);
                dialect.emit_compute_root_kernel(
                    out, &kernel_name, &pred_op.kind, op_kind,
                    reduce, finalize, transform, shared_budget,
                )?;
                continue;
            }
        }

        // ── ComputePattern dispatch (Standalone / LoopFusion / etc.) ──
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

// ── EpilogueInjection dispatch ───────────────────────────────────────────────

/// Emit a fused GEMM + epilogue kernel for an `EpilogueInjection` group.
///
/// Extracts epilogue trace bodies from the registry and delegates to the
/// dialect's `emit_gemm_epilogue_kernel`.
fn emit_epilogue_injection<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    group: &crate::compiler::fusion::FusionGroup,
    graph: &CompilerGraph,
    registry: Option<&ScalarOpRegistry>,
) -> Result<(), String> {
    let anchor_op = graph.op(group.anchor).ok_or_else(|| {
        format!(
            "emit_epilogue_injection: anchor op {:?} not found in graph",
            group.anchor
        )
    })?;
    let op_kind = &anchor_op.kind;

    // Validate that the anchor is a GEMM-family op
    match op_kind {
        OpKind::Gemm { .. } | OpKind::GemmBias { .. } => {}
        _ => {
            return Err(format!(
                "emit_epilogue_injection: anchor op {:?} is not a GEMM-family op \
                 (expected Gemm or GemmBias), fusion planner bug",
                op_kind
            ));
        }
    }

    // Collect epilogue trace bodies from the registry
    let registry = registry.ok_or_else(|| {
        "emit_epilogue_injection: ScalarOpRegistry required for epilogue lookup".to_string()
    })?;

    let mut epilogue_bodies: Vec<&[crate::compiler::trace::TraceOp]> = Vec::new();
    for &epi_id in &group.epilogue {
        let epi_op = graph.op(epi_id).ok_or_else(|| {
            format!(
                "emit_epilogue_injection: epilogue op {:?} not found in graph",
                epi_id
            )
        })?;
        let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
        let trace = registry.get_trace(&key).ok_or_else(|| {
            format!(
                "emit_epilogue_injection: no OpTrace for epilogue op {:?}",
                epi_op.kind
            )
        })?;
        let body = trace.pattern.body().ok_or_else(|| {
            format!(
                "emit_epilogue_injection: epilogue op {:?} has no trace body \
                 (only elementwise ops are valid epilogue candidates)",
                epi_op.kind
            )
        })?;
        epilogue_bodies.push(body);
    }

    let epi_refs: Vec<&[crate::compiler::trace::TraceOp]> =
        epilogue_bodies.iter().copied().collect();

    dialect.emit_gemm_epilogue_kernel(out, kernel_name, op_kind, &epi_refs)
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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("__shared__ float smem["), "missing shared mem:\n{out}");
        assert!(out.contains("rsqrtf("), "missing rsqrt:\n{out}");
        assert!(out.contains("weight"), "missing weight param:\n{out}");
    }

    // ── NormLike dispatch: L2Normalize ─────────────────────────────

    #[test]
    fn normlike_dispatches_l2normalize() {
        let (graph, op) = one_op_graph(OpKind::L2Normalize { hidden: 1024 });
        let plan = one_group_plan(op);

        // L2Normalize NormLike trace: reduce=sum_sq, finalize=rsqrt(sum_sq+eps), transform=x*inv_norm
        let reduce = vec![TraceOp::Input(0), TraceOp::Mul(0, 0)];
        let finalize = vec![
            TraceOp::Input(0),       // [0] sum_sq
            TraceOp::Const(1e-12),   // [1] eps (baked into trace)
            TraceOp::Add(0, 1),      // [2] sum_sq + eps
            TraceOp::Rsqrt(2),       // [3] 1/sqrt(sum_sq + eps)
        ];
        let transform = vec![
            TraceOp::Input(0),  // [0] x
            TraceOp::Input(1),  // [1] inv_norm (from finalize)
            TraceOp::Mul(0, 1), // [2] x * inv_norm
        ];

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::L2Normalize, OpTrace {
            op_kind: OpKind::L2Normalize { hidden: 1024 },
            pattern: ComputePattern::NormLike { reduce, finalize, transform },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("__shared__ float smem["), "missing shared mem:\n{out}");
        assert!(out.contains("rsqrtf("), "missing rsqrt (L2Normalize finalize):\n{out}");
        // L2Normalize has NO weight and NO bias
        assert!(!out.contains("__restrict__ weight"), "unexpected weight param in L2Normalize:\n{out}");
        assert!(!out.contains("__restrict__ bias"), "unexpected bias param in L2Normalize:\n{out}");
        // Kernel text should be non-empty (beyond header)
        let header_only = {
            let mut h = String::new();
            dialect.emit_header(&mut h);
            h.len()
        };
        assert!(out.len() > header_only, "kernel output should be non-empty");
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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
    }

    // ── MHA dispatch ──────────────────────────────────────────────────

    #[test]
    fn mha_dispatches_via_gemm_pattern() {
        let kind = OpKind::MultiHeadAttention { seq_len: 32, num_heads: 8, head_dim: 64 };
        let (graph, op) = one_op_graph(kind.clone());
        let plan = one_group_plan(op);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::MultiHeadAttention, OpTrace {
            op_kind: kind,
            pattern: ComputePattern::Gemm,
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan MHA failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        // MHA kernel should have Q, K, V parameters
        assert!(out.contains("__restrict__ Q"), "missing Q param:\n{out}");
        assert!(out.contains("__restrict__ K"), "missing K param:\n{out}");
        assert!(out.contains("__restrict__ V"), "missing V param:\n{out}");
        // Should have softmax operations
        assert!(out.contains("fmaxf(") || out.contains("expf("),
            "missing softmax ops in MHA kernel:\n{out}");
        // Should have shared memory for scores
        assert!(out.contains("__shared__ float smem_scores["),
            "missing shared memory for scores:\n{out}");
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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, None, None);

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
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

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

        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan failed: {:?}", result);
        assert_eq!(out.len(), header_len, "empty injective body should produce no kernel");
    }

    // ── EpilogueInjection: GemmBias + GELU ─────────────────────────

    /// GELU trace body (tanh approximation):
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu_body() -> Vec<TraceOp> {
        vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Mul(0, 0),      // [1] x^2
            TraceOp::Mul(1, 0),      // [2] x^3
            TraceOp::Const(0.044715),// [3] coeff
            TraceOp::Mul(2, 3),      // [4] 0.044715 * x^3
            TraceOp::Add(0, 4),      // [5] x + 0.044715 * x^3
            TraceOp::Const(0.7978845608), // [6] sqrt(2/pi)
            TraceOp::Mul(5, 6),      // [7] sqrt(2/pi) * (...)
            TraceOp::Tanh(7),        // [8] tanh(...)
            TraceOp::Const(1.0),     // [9] 1.0
            TraceOp::Add(8, 9),      // [10] 1 + tanh(...)
            TraceOp::Const(0.5),     // [11] 0.5
            TraceOp::Mul(0, 11),     // [12] 0.5 * x
            TraceOp::Mul(12, 10),    // [13] 0.5 * x * (1 + tanh(...))
        ]
    }

    /// Build an EpilogueInjection plan: GemmBias anchor + epilogue ops.
    fn epilogue_plan(
        anchor: OpId,
        epilogue: Vec<OpId>,
    ) -> FusionPlan {
        let mut ops = vec![anchor];
        ops.extend(epilogue.iter());
        let mut op_to_group = HashMap::new();
        for &id in &ops {
            op_to_group.insert(id, 0);
        }
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor,
                epilogue,
                mode: FusionMode::EpilogueInjection,
                ops,
            }],
            op_to_group,
        }
    }

    #[test]
    fn epilogue_injection_gemmbias_gelu() {
        // Build a graph: GemmBias -> GELU
        let mut g = CompilerGraph::new();
        let t_a = g.add_tensor("A", vec![64, 64], DType::F32);
        let t_b = g.add_tensor("B", vec![64, 64], DType::F32);
        let t_gemm_out = g.add_tensor("gemm_out", vec![64, 64], DType::F32);
        let t_gelu_out = g.add_tensor("gelu_out", vec![64, 64], DType::F32);

        let gemm_op = g.add_op(
            OpKind::GemmBias { m: 64, n: 64, k: 64 },
            vec![t_a, t_b],
            vec![t_gemm_out],
            "gemmbias",
        );
        let gelu_op = g.add_op(
            OpKind::Gelu,
            vec![t_gemm_out],
            vec![t_gelu_out],
            "gelu",
        );

        let plan = epilogue_plan(gemm_op, vec![gelu_op]);

        // Register the GELU trace
        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Gelu, OpTrace {
            op_kind: OpKind::Gelu,
            pattern: ComputePattern::Elementwise { body: gelu_body() },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &g, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan EpilogueInjection failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        // The kernel should contain bias parameter (GemmBias)
        assert!(out.contains("bias"), "missing bias param in fused kernel:\n{out}");
        // The kernel should contain GELU operations (tanh)
        assert!(out.contains("tanhf("), "missing tanhf (GELU epilogue):\n{out}");
        // The kernel should be a single fused GEMM kernel (with shared memory)
        assert!(out.contains("__shared__ float"), "missing shared mem (tiled GEMM):\n{out}");
        // Should have fmaf for the GEMM accumulation
        assert!(out.contains("fmaf("), "missing fmaf (GEMM accumulate):\n{out}");
    }

    #[test]
    fn epilogue_injection_gemm_silu() {
        // Build a graph: Gemm -> SiLU
        let mut g = CompilerGraph::new();
        let t_a = g.add_tensor("A", vec![64, 64], DType::F32);
        let t_b = g.add_tensor("B", vec![64, 64], DType::F32);
        let t_gemm_out = g.add_tensor("gemm_out", vec![64, 64], DType::F32);
        let t_silu_out = g.add_tensor("silu_out", vec![64, 64], DType::F32);

        let gemm_op = g.add_op(
            OpKind::Gemm { m: 64, n: 64, k: 64 },
            vec![t_a, t_b],
            vec![t_gemm_out],
            "gemm",
        );
        let silu_op = g.add_op(
            OpKind::Silu,
            vec![t_gemm_out],
            vec![t_silu_out],
            "silu",
        );

        let plan = epilogue_plan(gemm_op, vec![silu_op]);

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Silu, OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body: silu_body() },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &g, Some(&reg), None);

        assert!(result.is_ok(), "gpu_emit_plan EpilogueInjection failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        // No bias parameter (plain Gemm, not GemmBias)
        assert!(!out.contains("__restrict__ bias"), "unexpected bias in Gemm kernel:\n{out}");
        // SiLU epilogue: expf
        assert!(out.contains("expf("), "missing expf (SiLU epilogue):\n{out}");
    }

    #[test]
    fn epilogue_injection_rejects_non_gemm_anchor() {
        // EpilogueInjection with non-GEMM anchor should fail
        let (graph, op) = one_op_graph(OpKind::Silu);

        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op,
                epilogue: vec![],
                mode: FusionMode::EpilogueInjection,
                ops: vec![op],
            }],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op, 0);
                m
            },
        };

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, None, None);

        assert!(result.is_err(), "EpilogueInjection with non-GEMM anchor should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("not a GEMM-family op"),
            "error should mention non-GEMM anchor: {err}",
        );
    }

    #[test]
    fn epilogue_injection_no_epilogue_still_emits_gemm() {
        // EpilogueInjection with zero epilogue ops should still produce a valid GEMM kernel
        let mut g = CompilerGraph::new();
        let t_a = g.add_tensor("A", vec![64, 64], DType::F32);
        let t_b = g.add_tensor("B", vec![64, 64], DType::F32);
        let t_out = g.add_tensor("out", vec![64, 64], DType::F32);

        let gemm_op = g.add_op(
            OpKind::Gemm { m: 64, n: 64, k: 64 },
            vec![t_a, t_b],
            vec![t_out],
            "gemm",
        );

        let plan = epilogue_plan(gemm_op, vec![]);

        let reg = ScalarOpRegistry::new();

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &g, Some(&reg), None);

        assert!(result.is_ok(), "EpilogueInjection with no epilogue should succeed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("__shared__ float"), "missing shared mem:\n{out}");
        assert!(out.contains("fmaf("), "missing fmaf:\n{out}");
    }

    // ── TileLevelFusion dispatch ────────────────────────────────────

    /// Helper: build a two-op graph (RmsNorm → Gemm) and return (graph, norm_id, gemm_id).
    fn norm_gemm_graph() -> (CompilerGraph, OpId, OpId) {
        let mut g = CompilerGraph::new();
        let t_in = g.add_tensor("in", vec![64, 64], DType::F32);
        let t_normed = g.add_tensor("normed", vec![64, 64], DType::F32);
        let t_weight = g.add_tensor("weight", vec![64, 64], DType::F32);
        let t_out = g.add_tensor("out", vec![64, 64], DType::F32);

        let norm_op = g.add_op(
            OpKind::RmsNorm { eps: 1e-6 },
            vec![t_in],
            vec![t_normed],
            "rmsnorm",
        );
        let gemm_op = g.add_op(
            OpKind::Gemm { m: 64, n: 64, k: 64 },
            vec![t_normed, t_weight],
            vec![t_out],
            "gemm",
        );
        (g, norm_op, gemm_op)
    }

    /// RmsNorm NormLike trace components.
    fn rmsnorm_trace() -> (Vec<TraceOp>, Vec<TraceOp>, Vec<TraceOp>) {
        let reduce = vec![TraceOp::Input(0), TraceOp::Mul(0, 0)];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(0)];
        let transform = vec![
            TraceOp::Input(0),  // x
            TraceOp::Input(1),  // inv_rms
            TraceOp::Mul(0, 1), // x * inv_rms
            TraceOp::Input(2),  // weight
            TraceOp::Mul(2, 3), // (x * inv_rms) * weight
        ];
        (reduce, finalize, transform)
    }

    #[test]
    fn tile_level_fusion_emits_fused_kernel() {
        let (graph, norm_op, gemm_op) = norm_gemm_graph();
        let (reduce, finalize, transform) = rmsnorm_trace();

        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_op,
                epilogue: vec![],
                mode: FusionMode::TileLevelFusion {
                    predecessor: norm_op,
                    tile_rows: 32,
                },
                ops: vec![norm_op, gemm_op],
            }],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(norm_op, 0);
                m.insert(gemm_op, 0);
                m
            },
        };

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::RmsNorm, OpTrace {
            op_kind: OpKind::RmsNorm { eps: 1e-6 },
            pattern: ComputePattern::NormLike { reduce, finalize, transform },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "TileLevelFusion failed: {:?}", result);
        assert!(out.contains("group_0"), "missing kernel name:\n{out}");
        assert!(out.contains("__shared__"), "missing shared memory:\n{out}");
        assert!(out.contains("__syncthreads"), "missing barrier:\n{out}");
        assert!(out.contains("rsqrtf("), "missing rsqrtf (norm finalize):\n{out}");
        assert!(out.contains("fmaf("), "missing fmaf (GEMM accumulate):\n{out}");
        assert!(out.contains("norm_tile"), "missing norm_tile shared array:\n{out}");
    }

    #[test]
    fn compute_root_emits_two_phase_kernel() {
        let (graph, norm_op, gemm_op) = norm_gemm_graph();
        let (reduce, finalize, transform) = rmsnorm_trace();

        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_op,
                epilogue: vec![],
                mode: FusionMode::ComputeRoot {
                    predecessor: norm_op,
                },
                ops: vec![norm_op, gemm_op],
            }],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(norm_op, 0);
                m.insert(gemm_op, 0);
                m
            },
        };

        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::RmsNorm, OpTrace {
            op_kind: OpKind::RmsNorm { eps: 1e-6 },
            pattern: ComputePattern::NormLike { reduce, finalize, transform },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &graph, Some(&reg), None);

        assert!(result.is_ok(), "ComputeRoot failed: {:?}", result);
        // ComputeRoot emits two kernels: norm + gemm
        assert!(out.contains("group_0_norm"), "missing norm kernel:\n{out}");
        assert!(out.contains("group_0_gemm"), "missing gemm kernel:\n{out}");
        assert!(out.contains("rsqrtf("), "missing rsqrtf (norm):\n{out}");
        assert!(out.contains("fmaf("), "missing fmaf (GEMM):\n{out}");
        assert!(out.contains("__shared__"), "missing shared memory:\n{out}");
    }

    #[test]
    fn tile_level_fusion_rejects_non_normlike_predecessor() {
        let mut g = CompilerGraph::new();
        let t_in = g.add_tensor("in", vec![64, 64], DType::F32);
        let t_silu_out = g.add_tensor("silu_out", vec![64, 64], DType::F32);
        let t_weight = g.add_tensor("weight", vec![64, 64], DType::F32);
        let t_out = g.add_tensor("out", vec![64, 64], DType::F32);

        let silu_op = g.add_op(OpKind::Silu, vec![t_in], vec![t_silu_out], "silu");
        let gemm_op = g.add_op(
            OpKind::Gemm { m: 64, n: 64, k: 64 },
            vec![t_silu_out, t_weight],
            vec![t_out],
            "gemm",
        );

        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_op,
                epilogue: vec![],
                mode: FusionMode::TileLevelFusion {
                    predecessor: silu_op,
                    tile_rows: 32,
                },
                ops: vec![silu_op, gemm_op],
            }],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(silu_op, 0);
                m.insert(gemm_op, 0);
                m
            },
        };

        // Register SiLU as Elementwise (not NormLike)
        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(OpKindKey::Silu, OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body: silu_body() },
            signature: dummy_sig(),
        });

        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        let result = gpu_emit_plan(&dialect, &mut out, &plan, &g, Some(&reg), None);

        assert!(result.is_err(), "TileLevelFusion with non-NormLike predecessor should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("NormLike"),
            "error should mention NormLike requirement: {err}",
        );
    }
}
