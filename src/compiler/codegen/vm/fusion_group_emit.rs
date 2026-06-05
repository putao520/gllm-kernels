//! Fusion group emission — layout transform + per-mode fusion group lowering.

use super::instr::*;
use super::vm_state::AbiPtrs;
use super::plan_lower::{
    LoweringContext, TensorPtrResolver,
    emit_standalone_op, emit_elementwise_inline,
    extract_gemm_dims_sym, collect_epilogue_trace,
    load_op_scratch_ptr, extract_op_trace, infer_output_shape_sym,
};
use super::gemm_emit::{emit_gemm_inline_with_hook, emit_gemm_inline_with_epilogue};

use crate::compiler::fusion::{FusionGroup, FusionMode};
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::compiler::layout_negotiator::MovementType;
use crate::types::CompilerError;

/// §0.2.11 InterOpTransform 布局变换 VmInstr 发射
///
/// 当 R1.5 布局协商检测到两个相邻 op 的布局不兼容且不在免费变换窗口时，
/// 需要在它们之间插入显式布局变换。这仅发生在 RegisterDirect 场景下。
///
/// 变换类型映射:
/// - RowMajor <-> ColMajor -> emit Transpose2D loop
/// - RowMajor <-> HeadSplit -> zero-cost reshape (no VmInstr needed)
/// - PanelPacked/Vnni/AmxTile -> handled by PackMap (§0.2.7)
pub(super) fn emit_layout_transform(
    prog: &mut VmProgram,
    xform: &crate::compiler::layout_negotiator::LayoutTransform,
    _graph: &CompilerGraph,
    _alloc: &BufferAllocation,
    _resolver: &TensorPtrResolver,
    _abi: &AbiPtrs,
) -> Result<(), CompilerError> {
    use crate::compiler::accel_registry::LayoutConstraint;

    match (&xform.source, &xform.target) {
        // RowMajor <-> ColMajor: 需要 2D transpose
        (LayoutConstraint::RowMajor { .. }, LayoutConstraint::ColMajor { .. }) |
        (LayoutConstraint::ColMajor { .. }, LayoutConstraint::RowMajor { .. }) => {
            // Transpose 由下游 op 的 stride 计算隐式处理 — 不需要额外的内存搬运。
            // 融合模式中的 stride 重计算已在 emit_gemm/emit_elementwise 中完成。
            // 对于 Standalone 模式，TensorPtrResolver 的 offset 计算已考虑布局差异。
            // 此处标记已消费即可。
        }
        // HeadSplit is a zero-cost reshape of RowMajor — no transform needed
        (LayoutConstraint::RowMajor { .. }, LayoutConstraint::HeadSplit { .. }) |
        (LayoutConstraint::HeadSplit { .. }, LayoutConstraint::RowMajor { .. }) => {}
        // PanelPacked/Vnni/AmxTile — handled by PackMap (§0.2.7)
        (LayoutConstraint::PanelPacked { .. }, _) |
        (_, LayoutConstraint::PanelPacked { .. }) |
        (LayoutConstraint::VnniPacked4, _) |
        (_, LayoutConstraint::VnniPacked4) |
        (LayoutConstraint::AmxTileBF16 { .. }, _) |
        (_, LayoutConstraint::AmxTileBF16 { .. }) => {}
        // GPU layouts — handled by GPU codegen path
        (LayoutConstraint::SharedMemTile { .. }, _) |
        (_, LayoutConstraint::SharedMemTile { .. }) |
        (LayoutConstraint::TmaAligned2D { .. }, _) |
        (_, LayoutConstraint::TmaAligned2D { .. }) => {}
        // Any is always compatible — no transform
        (LayoutConstraint::Any, _) | (_, LayoutConstraint::Any) => {}
        // InterleavedPairs is a stride-level transform consumed by SwiGLU auto_select
        (LayoutConstraint::InterleavedPairs, _) |
        (_, LayoutConstraint::InterleavedPairs) => {}
        _ => {}
    }
    Ok(())
}

/// 单个 fusion group 的 FusionMode 分派 — 被 emit_fusion_groups 和 compile_layer_type_body 共用。
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_fusion_group_by_mode(
    prog: &mut VmProgram,
    group: &FusionGroup,
    anchor_op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    ctx: &LoweringContext,
    group_input_ptr: VRegId,
    group_weight_ptr: VRegId,
    group_output_ptr: VRegId,
    scratch_base: VRegId,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    rope_cache_offset: Option<usize>,
    seq_bound_override: Option<&BoundExpr>,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
) -> Result<(), CompilerError> {
    let width = ctx.width;
    let sym_map = ctx.sym_map;
    let registry = ctx.registry;

    // §0.2.11: 获取当前融合组的布局协商结果
    let group_layout = ctx.layout
        .and_then(|la| la.group_assignments.iter().find(|ga| ga.group_id == group.id));

    // §0.2.11: 处理 InterOpTransform — 在 op 间插入布局变换 VmInstr
    // 仅处理 RegisterDirect 不兼容的 case（需要显式变换）。
    // RegisterToMemory/MemoryToMemory 是免费变换窗口，stride 在 store 时自然改变。
    if let Some(gl) = group_layout {
        for xform in &gl.inter_op_transforms {
            if xform.movement == MovementType::RegisterDirect && xform.transform.cost > 0.0 {
                emit_layout_transform(prog, &xform.transform, graph, alloc, resolver, abi)?;
            }
        }
    }

    // QuantGemm anchor ops can only be lowered via emit_standalone_op (which
    // dispatches to emit_quant_gemm_inline).  Fusion modes like NormIntoGemm,
    // QkvSharedInput, EpilogueInjection, etc. call emit_gemm_inline_with_hook
    // which assumes F32 Gemm.  When the anchor is QuantGemm, break the group
    // into per-op standalone lowering.
    if matches!(anchor_op.kind, OpKind::QuantGemm { .. })
        && !matches!(group.mode, FusionMode::Standalone | FusionMode::LoopFusion)
    {
        eprintln!("[QGEM-FALLBACK] anchor='{}' mode={:?} ops_count={} epilogue={:?} abi.wp={:?}",
            anchor_op.label, group.mode, group.ops.len(),
            group.epilogue.iter().filter_map(|&oid| graph.op(oid).map(|o| o.label.clone())).collect::<Vec<_>>(),
            abi.weight_ptr);
        for &op_id in &group.ops {
            let op = graph.op(op_id).ok_or_else(|| CompilerError::CodegenViolation(
                format!("QuantGemm fallback: op {:?} not found", op_id)))?;
            let op_input = op.inputs.first()
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let op_weight = op.inputs.get(1)
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            let op_output = op.outputs.first()
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(output_ptr);
            emit_standalone_op(prog, op, graph, ctx,
                op_input, op_weight, op_output, rope_cache_offset,
                resolver, abi)?;
        }
        return Ok(());
    }

    match &group.mode {
        FusionMode::Standalone | FusionMode::LoopFusion => {
            emit_standalone_op(prog, anchor_op, graph, ctx,
                group_input_ptr, group_weight_ptr, group_output_ptr, rope_cache_offset,
                resolver, abi)?;
        }

        FusionMode::EpilogueInjection => {
            let (m_dim, n, k) = extract_gemm_dims_sym(anchor_op)?;
            let epi_trace = collect_epilogue_trace(group, graph, registry)?;
            for op in &epi_trace {
                if let crate::compiler::trace::TraceOp::Input(n) = op {
                    if *n >= 1 {
                        let epi_kinds: Vec<_> = group.epilogue.iter()
                            .filter_map(|&oid| graph.op(oid))
                            .map(|o| format!("{:?}(inputs={})", o.kind, o.inputs.len()))
                            .collect();
                        return Err(CompilerError::CodegenViolation(format!(
                            "EpilogueInjection: GEMM epilogue trace 引用 Input({}) — \
                             GEMM 累加器只能提供 Input(0)。group.epilogue ops: {:?}; trace: {:?}",
                            n, epi_kinds, epi_trace
                        )));
                    }
                }
            }
            let terminal_op_id = group.epilogue.last().copied().unwrap_or(anchor_op.id);
            let gemm_output_ptr = if matches!(anchor_op.kind, OpKind::Gemm { .. } | OpKind::GemmBias { .. })
                && anchor_op.outputs.first() != graph.op(terminal_op_id).and_then(|op| op.outputs.first())
            {
                group_output_ptr
            } else {
                graph.op(terminal_op_id)
                    .and_then(|op| op.outputs.first().copied())
                    .and_then(|tid| resolver.materialize(prog, tid, abi))
                    .unwrap_or(group_output_ptr)
            };
            let gemm_trans_b = match &anchor_op.kind {
                OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                _ => false,
            };
            emit_gemm_inline_with_epilogue(prog, &m_dim, n, k, width,
                group_input_ptr, group_weight_ptr, gemm_output_ptr,
                &epi_trace, sym_map, graph.telemetry.gemm_row_stats, seq_bound_override, ctx.dtype, gemm_trans_b,
                super::isa_hook::EpiloguePlace::OnAccumulators)?;
        }

        FusionMode::NormIntoGemm => {
            let norm_op = group.ops.iter()
                .filter_map(|&oid| graph.op(oid))
                .find(|op| matches!(op.kind,
                    OpKind::RmsNorm{..} | OpKind::LayerNorm{..} | OpKind::ValueNorm{..}))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    "NormIntoGemm: group.ops 中未找到 Norm op".into()))?;
            let norm_output_tid = norm_op.outputs.first().copied()
                .ok_or_else(|| CompilerError::CodegenViolation(
                    "NormIntoGemm: Norm op 无输出张量".into()))?;
            let scratch_offset = alloc.offset_of(norm_output_tid)
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("NormIntoGemm: BufferAllocation 中未找到张量 {:?} 的 scratchpad 偏移", norm_output_tid)))?;
            let scratch_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: scratch_ptr, src: PtrExpr::VRegPlusConst(scratch_base, scratch_offset) });

            let norm_input_ptr = norm_op.inputs.first()
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let norm_weight_ptr = norm_op.inputs.get(1)
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            prog.emit_scope(|p| -> Result<(), CompilerError> {
                emit_standalone_op(p, norm_op, graph, ctx,
                    norm_input_ptr, norm_weight_ptr, scratch_ptr, rope_cache_offset,
                    resolver, abi)?;
                let (m_dim, n, k) = extract_gemm_dims_sym(anchor_op)?;
                let pm = ctx.pack_map_for_gemm(anchor_op.inputs.get(1).copied());
                let norm_into_gemm_trans_b = match &anchor_op.kind {
                    OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                    _ => false,
                };
                emit_gemm_inline_with_hook(p, &m_dim, n, k, ctx,
                    scratch_ptr, group_weight_ptr, group_output_ptr, seq_bound_override, Some(anchor_op.id), pm, norm_into_gemm_trans_b)?;
                Ok(())
            })?;
        }

        FusionMode::QkvSharedInput => {
            for &op_id in &group.ops {
                if let Some(op) = graph.op(op_id) {
                    if let Ok((m_dim, n, k)) = extract_gemm_dims_sym(op) {
                        let out_ptr = load_op_scratch_ptr(prog, scratch_base, op, alloc, resolver, abi)?;
                        let gemm_input = op.inputs.first().copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(input_ptr);
                        let gemm_weight = op.inputs.get(1).copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(weight_ptr);
                        let pm = ctx.pack_map_for_gemm(op.inputs.get(1).copied());
                        let qkv_trans_b = match &op.kind {
                            OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                            _ => false,
                        };
                        prog.emit_scope(|p| -> Result<(), CompilerError> {
                            emit_gemm_inline_with_hook(p, &m_dim, n, k, ctx, gemm_input, gemm_weight, out_ptr, seq_bound_override, Some(op.id), pm, qkv_trans_b)?;
                            Ok(())
                        })?;
                    }
                }
            }
        }

        FusionMode::TileLevelFusion { predecessor, .. } => {
            let pre_op = graph.op(*predecessor).ok_or_else(|| CompilerError::CodegenViolation(
                format!("TileLevelFusion predecessor {:?} not found", predecessor)))?;
            let pre_scratch = load_op_scratch_ptr(prog, scratch_base, pre_op, alloc, resolver, abi)?;

            let pre_input_ptr = pre_op.inputs.first()
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let pre_weight_ptr = pre_op.inputs.get(1)
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            emit_standalone_op(prog, pre_op, graph, ctx,
                pre_input_ptr, pre_weight_ptr, pre_scratch, rope_cache_offset,
                resolver, abi)?;

            let (m_dim, n, k) = extract_gemm_dims_sym(anchor_op)?;
            let pm = ctx.pack_map_for_gemm(anchor_op.inputs.get(1).copied());
            let pre_fusion_trans_b = match &anchor_op.kind {
                OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                _ => false,
            };
            emit_gemm_inline_with_hook(prog, &m_dim, n, k, ctx,
                pre_scratch, group_weight_ptr, group_output_ptr, seq_bound_override, Some(anchor_op.id), pm, pre_fusion_trans_b)?;
        }

        FusionMode::ComputeRoot { predecessor } => {
            let pre_op = graph.op(*predecessor).ok_or_else(|| CompilerError::CodegenViolation(
                format!("ComputeRoot predecessor {:?} not found", predecessor)))?;
            let pre_scratch = load_op_scratch_ptr(prog, scratch_base, pre_op, alloc, resolver, abi)?;

            let pre_input_ptr = pre_op.inputs.first()
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let pre_weight_ptr = pre_op.inputs.get(1)
                .and_then(|&tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            prog.emit_scope(|p| -> Result<(), CompilerError> {
                emit_standalone_op(p, pre_op, graph, ctx,
                    pre_input_ptr, pre_weight_ptr, pre_scratch, rope_cache_offset,
                    resolver, abi)?;
                Ok(())
            })?;

            let (m_dim, n, k) = extract_gemm_dims_sym(anchor_op)?;
            let pm = ctx.pack_map_for_gemm(anchor_op.inputs.get(1).copied());
            let pre_fusion_trans_b = match &anchor_op.kind {
                OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                _ => false,
            };
            emit_gemm_inline_with_hook(prog, &m_dim, n, k, ctx,
                pre_scratch, group_weight_ptr, group_output_ptr, seq_bound_override, Some(anchor_op.id), pm, pre_fusion_trans_b)?;
        }

        FusionMode::FFNBlock { gate_gemm, up_gemm, activation, combine } => {
            let gate_op = graph.op(*gate_gemm).ok_or_else(|| CompilerError::CodegenViolation(
                format!("FFNBlock gate_gemm op {:?} not found", gate_gemm)))?;
            let up_op = graph.op(*up_gemm).ok_or_else(|| CompilerError::CodegenViolation(
                format!("FFNBlock up_gemm op {:?} not found", up_gemm)))?;
            let gate_scratch = load_op_scratch_ptr(prog, scratch_base, gate_op, alloc, resolver, abi)?;
            let up_scratch = load_op_scratch_ptr(prog, scratch_base, up_op, alloc, resolver, abi)?;

            let gate_input = gate_op.inputs.first().copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let gate_weight = gate_op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            if let Ok((m_dim, n, k)) = extract_gemm_dims_sym(gate_op) {
                let pm = ctx.pack_map_for_gemm(gate_op.inputs.get(1).copied());
                let gate_trans_b = match &gate_op.kind {
                    OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                    _ => false,
                };
                prog.emit_scope(|p| -> Result<(), CompilerError> {
                    emit_gemm_inline_with_hook(p, &m_dim, n, k, ctx, gate_input, gate_weight, gate_scratch, seq_bound_override, Some(gate_op.id), pm, gate_trans_b)?;
                    Ok(())
                })?;
            }

            let up_input = up_op.inputs.first().copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let up_weight = up_op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            if let Ok((m_dim, n, k)) = extract_gemm_dims_sym(up_op) {
                let pm = ctx.pack_map_for_gemm(up_op.inputs.get(1).copied());
                let up_trans_b = match &up_op.kind {
                    OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                    _ => false,
                };
                prog.emit_scope(|p| -> Result<(), CompilerError> {
                    emit_gemm_inline_with_hook(p, &m_dim, n, k, ctx, up_input, up_weight, up_scratch, seq_bound_override, Some(up_op.id), pm, up_trans_b)?;
                    Ok(())
                })?;
            }

            let act_op = graph.op(*activation).ok_or_else(|| CompilerError::CodegenViolation(
                format!("FFNBlock activation op {:?} not found", activation)))?;
            let act_scratch = load_op_scratch_ptr(prog, scratch_base, act_op, alloc, resolver, abi)?;
            let act_trace = extract_op_trace(act_op, registry)?;
            let (act_shape, _) = infer_output_shape_sym(act_op, graph)?;
            emit_elementwise_inline(prog, &act_trace, &act_shape, width, false,
                false,
                gate_scratch, weight_ptr, act_scratch, sym_map, seq_bound_override, ctx.dtype)?;
            let combine_op = graph.op(*combine).ok_or_else(|| CompilerError::CodegenViolation(
                format!("FFNBlock combine op {:?} not found", combine)))?;
            let combine_trace = extract_op_trace(combine_op, registry)?;
            let (combine_shape, _) = infer_output_shape_sym(combine_op, graph)?;
            let combine_output = combine_op.outputs.first().copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(output_ptr);
            emit_elementwise_inline(prog, &combine_trace, &combine_shape, width, true,
                false,
                act_scratch, up_scratch, combine_output, sym_map, seq_bound_override, ctx.dtype)?;
        }

        FusionMode::CrossLayerResidual { residual, norm } => {
            let res_op = graph.op(*residual).ok_or_else(|| CompilerError::CodegenViolation(
                format!("CrossLayerResidual residual {:?} not found", residual)))?;
            let norm_op = graph.op(*norm).ok_or_else(|| CompilerError::CodegenViolation(
                format!("CrossLayerResidual norm {:?} not found", norm)))?;
            let res_scratch = load_op_scratch_ptr(prog, scratch_base, res_op, alloc, resolver, abi)?;

            let res_input0 = res_op.inputs.first().copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let res_input1 = res_op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);

            let res_trace = extract_op_trace(res_op, registry)?;
            let (res_shape, _) = infer_output_shape_sym(res_op, graph)?;
            let res_is_binary = res_op.inputs.len() > 1;
            emit_elementwise_inline(prog, &res_trace, &res_shape, width, res_is_binary,
                false,
                res_input0, res_input1, res_scratch, sym_map, seq_bound_override, ctx.dtype)?;

            let norm_input0 = norm_op.inputs.first().copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(res_scratch);
            let norm_weight = norm_op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(weight_ptr);
            let norm_output = norm_op.outputs.first().copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(output_ptr);

            let norm_trace = extract_op_trace(norm_op, registry)?;
            let (norm_shape, _) = infer_output_shape_sym(norm_op, graph)?;
            let norm_is_binary = norm_op.inputs.len() > 1;
            emit_elementwise_inline(prog, &norm_trace, &norm_shape, width, norm_is_binary,
                false,
                norm_input0, norm_weight, norm_output, sym_map, seq_bound_override, ctx.dtype)?;
        }

        FusionMode::FusedQkvNormRope { gemm_q, gemm_k, gemm_v, .. } => {
            for &op_id in &[*gemm_q, *gemm_k, *gemm_v] {
                if let Some(op) = graph.op(op_id) {
                    if let Ok((m_dim, n, k)) = extract_gemm_dims_sym(op) {
                        let out_ptr = load_op_scratch_ptr(prog, scratch_base, op, alloc, resolver, abi)?;
                        let gemm_input = op.inputs.first().copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(input_ptr);
                        let gemm_weight = op.inputs.get(1).copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(weight_ptr);
                        let pm = ctx.pack_map_for_gemm(op.inputs.get(1).copied());
                        let fqnr_trans_b = match &op.kind {
                            OpKind::Gemm { trans_b, .. } | OpKind::GemmBias { trans_b, .. } => *trans_b,
                            _ => false,
                        };
                        prog.emit_scope(|p| -> Result<(), CompilerError> {
                            emit_gemm_inline_with_hook(p, &m_dim, n, k, ctx, gemm_input, gemm_weight, out_ptr, seq_bound_override, Some(op.id), pm, fqnr_trans_b)?;
                            Ok(())
                        })?;
                    }
                }
            }
            for &op_id in &group.ops {
                if let Some(op) = graph.op(op_id) {
                    if !matches!(op.kind, OpKind::Gemm{..} | OpKind::GemmBias{..}) {
                        let op_input = op.inputs.first().copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(input_ptr);
                        let op_weight = op.inputs.get(1).copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(weight_ptr);
                        let op_output = op.outputs.first().copied()
                            .and_then(|tid| resolver.materialize(prog, tid, abi))
                            .unwrap_or(output_ptr);
                        emit_standalone_op(prog, op, graph, ctx,
                            op_input, op_weight, op_output, rope_cache_offset,
                            resolver, abi)?;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::layout_negotiator::{
        LayoutTransform, InterOpTransform, MovementType, GroupLayoutAssignment,
        LayoutAssignment,
    };
    use crate::compiler::accel_registry::LayoutConstraint;
    use crate::compiler::fusion::{FusionGroup, FusionMode, FusionCost};
    use crate::compiler::graph::{OpId, MultiOutputConfig};
    use std::collections::HashMap;

    /// Helper: build a minimal AbiPtrs with dummy VRegIds for emit_layout_transform tests.
    fn make_test_abi() -> AbiPtrs {
        AbiPtrs {
            input_ptr: VRegId(0),
            weight_ptr: Some(VRegId(1)),
            weight_abi_expr: None,
            output_ptr: VRegId(2),
            scratch_ptr: Some(VRegId(3)),
            gen_loop_counter: None,
            layer_loop_counter: None,
            mega_decode_seq_len: None,
            hook_ctx_ptr: None,
            sg_detect_scratch_offset: None,
            sg_knowledge_scratch_offset: None,
            callback_table_ptr: None,
            page_table_ptr: None,
            kv_load_mode: None,
            kv_cache_ptr: None,
            activation_ping_ptr: None,
            activation_pong_ptr: None,
        }
    }

    /// Helper: build minimal TensorPtrResolver for an empty graph.
    fn make_test_resolver() -> TensorPtrResolver {
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        TensorPtrResolver::build(&graph, &alloc)
    }

    // ── emit_layout_transform: all layout pairs return Ok and emit nothing ──

    #[test]
    fn layout_transform_row_major_to_col_major_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 64 },
            target: LayoutConstraint::ColMajor { align_bytes: 64 },
            cost: 1.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_row_major_to_head_split_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 32 },
            target: LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 64 },
            cost: 0.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_any_source_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::Any,
            target: LayoutConstraint::RowMajor { align_bytes: 64 },
            cost: 0.5,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_interleaved_pairs_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::InterleavedPairs,
            target: LayoutConstraint::RowMajor { align_bytes: 32 },
            cost: 0.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_panel_packed_to_row_major_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::PanelPacked { mr: 6, nr: 4 },
            target: LayoutConstraint::RowMajor { align_bytes: 64 },
            cost: 2.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_vnni_packed_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::VnniPacked4,
            target: LayoutConstraint::RowMajor { align_bytes: 64 },
            cost: 1.5,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    // ── FusionGroup construction and field access ──

    #[test]
    fn fusion_group_fields_are_set_correctly() {
        // Arrange
        let anchor = OpId(0);
        let op1 = OpId(1);
        let op2 = OpId(2);

        // Act
        let group = FusionGroup {
            id: 42,
            anchor,
            epilogue: vec![op1, op2],
            mode: FusionMode::EpilogueInjection,
            ops: vec![anchor, op1, op2],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };

        // Assert
        assert_eq!(group.id, 42);
        assert_eq!(group.anchor, OpId(0));
        assert_eq!(group.epilogue, vec![OpId(1), OpId(2)]);
        assert_eq!(group.ops.len(), 3);
        assert_eq!(group.mode, FusionMode::EpilogueInjection);
        assert!(group.dominant_dtype.is_none());
        assert!(!group.multi_output.is_multi_output());
    }

    #[test]
    fn fusion_mode_ffn_block_equality() {
        // Arrange
        let gate = OpId(10);
        let up = OpId(11);
        let act = OpId(12);
        let combine = OpId(13);

        // Act
        let mode_a = FusionMode::FFNBlock {
            gate_gemm: gate,
            up_gemm: up,
            activation: act,
            combine,
        };
        let mode_b = FusionMode::FFNBlock {
            gate_gemm: gate,
            up_gemm: up,
            activation: act,
            combine,
        };
        let mode_c = FusionMode::FFNBlock {
            gate_gemm: OpId(99),
            up_gemm: up,
            activation: act,
            combine,
        };

        // Assert
        assert_eq!(mode_a, mode_b);
        assert_ne!(mode_a, mode_c);
    }

    #[test]
    fn fusion_mode_tile_level_fusion_fields() {
        // Arrange & Act
        let pred = OpId(5);
        let mode = FusionMode::TileLevelFusion { predecessor: pred, tile_rows: 64 };

        // Assert
        match mode {
            FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                assert_eq!(predecessor, OpId(5));
                assert_eq!(tile_rows, 64);
            }
            _ => panic!("Expected TileLevelFusion variant"),
        }
    }

    // ── MovementType and InterOpTransform ──

    #[test]
    fn movement_type_variants_are_distinct() {
        // Arrange & Act & Assert
        assert_ne!(MovementType::RegisterDirect, MovementType::RegisterToMemory);
        assert_ne!(MovementType::RegisterToMemory, MovementType::MemoryToMemory);
        assert_ne!(MovementType::MemoryToMemory, MovementType::GpuGlobalToShared);
        assert_eq!(MovementType::RegisterDirect, MovementType::RegisterDirect);
    }

    #[test]
    fn inter_op_transform_fields_match_constructor() {
        // Arrange
        let transform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 64 },
            target: LayoutConstraint::ColMajor { align_bytes: 64 },
            cost: 3.0,
        };

        // Act
        let iot = InterOpTransform {
            producer: OpId(0),
            consumer: OpId(1),
            transform,
            movement: MovementType::RegisterDirect,
        };

        // Assert
        assert_eq!(iot.producer, OpId(0));
        assert_eq!(iot.consumer, OpId(1));
        assert_eq!(iot.movement, MovementType::RegisterDirect);
        assert_eq!(iot.transform.cost, 3.0);
    }

    // ── GroupLayoutAssignment filter logic (mirrors emit_fusion_group_by_mode lines 98-109) ──

    #[test]
    fn group_layout_assignment_find_by_group_id() {
        // Arrange
        let xform = InterOpTransform {
            producer: OpId(0),
            consumer: OpId(1),
            transform: LayoutTransform {
                source: LayoutConstraint::RowMajor { align_bytes: 64 },
                target: LayoutConstraint::ColMajor { align_bytes: 64 },
                cost: 1.0,
            },
            movement: MovementType::RegisterDirect,
        };
        let gla = GroupLayoutAssignment {
            group_id: 7,
            op_layouts: HashMap::new(),
            inter_op_transforms: vec![xform],
            total_benefit: 10.0,
            total_transform_cost: 1.0,
        };
        let assignments = vec![gla];

        // Act
        let found = assignments.iter().find(|ga| ga.group_id == 7);

        // Assert
        assert!(found.is_some());
        assert_eq!(found.unwrap().total_benefit, 10.0);
        // Simulate the filter from emit_fusion_group_by_mode line 106:
        // only RegisterDirect + cost > 0
        let costly_direct: Vec<_> = found.unwrap().inter_op_transforms.iter()
            .filter(|t| t.movement == MovementType::RegisterDirect && t.transform.cost > 0.0)
            .collect();
        assert_eq!(costly_direct.len(), 1);
    }

    #[test]
    fn fusion_cost_construction() {
        // Arrange & Act
        let cost = FusionCost {
            bytes_saved: 8192,
            extra_regs: 4,
            scratch_bytes: 1024,
            benefit: 8192 - 1024,
        };

        // Assert
        assert!(cost.benefit > 0);
        assert_eq!(cost.bytes_saved, 8192);
        assert_eq!(cost.extra_regs, 4);
        assert_eq!(cost.scratch_bytes, 1024);
    }

    // ── Additional tests ──

    #[test]
    fn layout_transform_gpu_shared_mem_tile_is_ok() {
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::SharedMemTile { tile_rows: 16, tile_cols: 16, padding_bytes: 4 },
            target: LayoutConstraint::RowMajor { align_bytes: 128 },
            cost: 2.5,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_amx_tile_to_row_major_is_ok() {
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 },
            target: LayoutConstraint::RowMajor { align_bytes: 64 },
            cost: 1.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_tma_aligned_2d_is_ok() {
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 128 },
            target: LayoutConstraint::TmaAligned2D { tile_m: 64, tile_n: 64 },
            cost: 0.5,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn fusion_mode_compute_root_equality() {
        let pred = OpId(7);
        let mode_a = FusionMode::ComputeRoot { predecessor: pred };
        let mode_b = FusionMode::ComputeRoot { predecessor: pred };
        let mode_c = FusionMode::ComputeRoot { predecessor: OpId(99) };

        assert_eq!(mode_a, mode_b);
        assert_ne!(mode_a, mode_c);
    }

    #[test]
    fn fusion_mode_cross_layer_residual_equality() {
        let residual = OpId(20);
        let norm = OpId(21);
        let mode_a = FusionMode::CrossLayerResidual { residual, norm };
        let mode_b = FusionMode::CrossLayerResidual { residual, norm };
        let mode_c = FusionMode::CrossLayerResidual { residual: OpId(0), norm };

        assert_eq!(mode_a, mode_b);
        assert_ne!(mode_a, mode_c);
    }

    #[test]
    fn fusion_mode_fused_qkv_norm_rope_equality() {
        let mode_a = FusionMode::FusedQkvNormRope {
            gemm_q: OpId(0),
            gemm_k: OpId(1),
            gemm_v: OpId(2),
            qk_norm_q: OpId(3),
            qk_norm_k: OpId(4),
            value_norm_v: OpId(5),
            rope_q: OpId(6),
            rope_k: OpId(7),
        };
        let mode_b = FusionMode::FusedQkvNormRope {
            gemm_q: OpId(0),
            gemm_k: OpId(1),
            gemm_v: OpId(2),
            qk_norm_q: OpId(3),
            qk_norm_k: OpId(4),
            value_norm_v: OpId(5),
            rope_q: OpId(6),
            rope_k: OpId(7),
        };
        let mode_c = FusionMode::FusedQkvNormRope {
            gemm_q: OpId(99),
            gemm_k: OpId(1),
            gemm_v: OpId(2),
            qk_norm_q: OpId(3),
            qk_norm_k: OpId(4),
            value_norm_v: OpId(5),
            rope_q: OpId(6),
            rope_k: OpId(7),
        };

        assert_eq!(mode_a, mode_b);
        assert_ne!(mode_a, mode_c);
    }

    #[test]
    fn fusion_group_infer_dominant_dtype_is_none_for_empty_graph() {
        let mut group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };
        let graph = CompilerGraph::new();

        group.infer_dominant_dtype(&graph);

        assert!(group.dominant_dtype.is_none());
    }

    #[test]
    fn fusion_plan_display_shows_group_count() {
        use crate::compiler::fusion::FusionPlan;
        let op0 = OpId(0);
        let group = FusionGroup {
            id: 0,
            anchor: op0,
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op0],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };
        let plan = FusionPlan {
            groups: vec![group],
            op_to_group: HashMap::from([(op0, 0)]),
        };

        let display = format!("{}", plan);

        assert!(display.contains("1 groups"));
        assert!(display.contains("Standalone"));
        assert!(display.contains("anchor=Op(0)"));
    }

    #[test]
    fn fusion_cost_zero_benefit_when_scratch_equals_saved() {
        let cost = FusionCost {
            bytes_saved: 4096,
            extra_regs: 0,
            scratch_bytes: 4096,
            benefit: 0,
        };

        assert_eq!(cost.benefit, 0);
        assert_eq!(cost.bytes_saved, cost.scratch_bytes);
    }

    #[test]
    fn fusion_cost_negative_benefit_when_scratch_exceeds_saved() {
        let cost = FusionCost {
            bytes_saved: 1024,
            extra_regs: 8,
            scratch_bytes: 2048,
            benefit: -1024,
        };

        assert!(cost.benefit < 0);
    }

    #[test]
    fn layout_constraint_compatible_with_any() {
        let any = LayoutConstraint::Any;
        let row_major = LayoutConstraint::RowMajor { align_bytes: 64 };
        let col_major = LayoutConstraint::ColMajor { align_bytes: 64 };

        assert!(any.compatible_with(&row_major));
        assert!(row_major.compatible_with(&any));
        assert!(any.compatible_with(&col_major));
        assert!(any.compatible_with(&LayoutConstraint::VnniPacked4));
    }

    #[test]
    fn group_layout_assignment_filter_excludes_zero_cost_register_direct() {
        let free_xform = InterOpTransform {
            producer: OpId(0),
            consumer: OpId(1),
            transform: LayoutTransform {
                source: LayoutConstraint::RowMajor { align_bytes: 64 },
                target: LayoutConstraint::ColMajor { align_bytes: 64 },
                cost: 0.0,
            },
            movement: MovementType::RegisterDirect,
        };
        let memory_xform = InterOpTransform {
            producer: OpId(2),
            consumer: OpId(3),
            transform: LayoutTransform {
                source: LayoutConstraint::RowMajor { align_bytes: 64 },
                target: LayoutConstraint::ColMajor { align_bytes: 64 },
                cost: 5.0,
            },
            movement: MovementType::RegisterToMemory,
        };
        let gla = GroupLayoutAssignment {
            group_id: 1,
            op_layouts: HashMap::new(),
            inter_op_transforms: vec![free_xform, memory_xform],
            total_benefit: 10.0,
            total_transform_cost: 5.0,
        };

        let costly_direct: Vec<_> = gla.inter_op_transforms.iter()
            .filter(|t| t.movement == MovementType::RegisterDirect && t.transform.cost > 0.0)
            .collect();

        assert_eq!(costly_direct.len(), 0);
    }

    // ── 10 new tests covering untested logic paths ──

    #[test]
    fn layout_transform_col_major_to_col_major_falls_through_to_wildcard() {
        // Arrange: ColMajor→ColMajor hits the wildcard `_ => {}` arm (not the
        // explicit RowMajor<->ColMajor branches).  Verify Ok + empty program.
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::ColMajor { align_bytes: 64 },
            target: LayoutConstraint::ColMajor { align_bytes: 128 },
            cost: 1.5,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_any_target_always_ok() {
        // Arrange: Any as target should match the `(_, LayoutConstraint::Any)` arm.
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::ColMajor { align_bytes: 64 },
            target: LayoutConstraint::Any,
            cost: 0.1,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_interleaved_pairs_to_vnni_hits_wildcard() {
        // Arrange: InterleavedPairs→VnniPacked4 does not match any explicit arm,
        // so it falls through to the wildcard `_ => {}`.  Verify it still returns Ok.
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::InterleavedPairs,
            target: LayoutConstraint::VnniPacked4,
            cost: 0.5,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn layout_transform_gpu_shared_mem_tile_as_target() {
        // Arrange: RowMajor→SharedMemTile matches the GPU layout arm
        // `(_, LayoutConstraint::SharedMemTile { .. })`.
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 64 },
            target: LayoutConstraint::SharedMemTile { tile_rows: 32, tile_cols: 32, padding_bytes: 8 },
            cost: 3.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn infer_dominant_dtype_succeeds_with_bf16_tensor() {
        // Arrange: Build a real graph with a BF16 tensor as the anchor op's input.
        use crate::compiler::graph::SymDim;
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor_concrete("hidden", &[1, 512], DType::BF16);
        let out_tid = graph.add_tensor_concrete("output", &[1, 512], DType::BF16);
        let op_id = graph.add_op(
            OpKind::RmsNorm { eps: 1e-5 },
            vec![tid],
            vec![out_tid],
            "norm",
        );
        let mut group = FusionGroup {
            id: 0,
            anchor: op_id,
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };

        // Act
        group.infer_dominant_dtype(&graph);

        // Assert
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::BF16));
    }

    #[test]
    fn fusion_plan_group_of_returns_correct_group_in_multi_group_plan() {
        // Arrange: two groups with different modes.
        use crate::compiler::fusion::FusionPlan;
        let op0 = OpId(0);
        let op1 = OpId(1);
        let op2 = OpId(2);
        let g0 = FusionGroup {
            id: 0, anchor: op0, epilogue: vec![], mode: FusionMode::Standalone,
            ops: vec![op0], multi_output: MultiOutputConfig::single(), dominant_dtype: None,
        };
        let g1 = FusionGroup {
            id: 1, anchor: op1, epilogue: vec![op2], mode: FusionMode::EpilogueInjection,
            ops: vec![op1, op2], multi_output: MultiOutputConfig::single(), dominant_dtype: None,
        };
        let plan = FusionPlan {
            groups: vec![g0, g1],
            op_to_group: HashMap::from([(op0, 0), (op1, 1), (op2, 1)]),
        };

        // Act & Assert
        assert_eq!(plan.group_of(op0).unwrap().mode, FusionMode::Standalone);
        assert_eq!(plan.group_of(op1).unwrap().mode, FusionMode::EpilogueInjection);
        assert_eq!(plan.group_of(op2).unwrap().id, 1);
        assert!(plan.group_of(OpId(99)).is_none());
    }

    #[test]
    fn fusion_plan_num_fused_ops_counts_all_non_standalone_modes() {
        // Arrange: one Standalone (excluded), one LoopFusion (2 ops), one QkvSharedInput (3 ops).
        use crate::compiler::fusion::FusionPlan;
        let ops: Vec<OpId> = (0..6).map(OpId).collect();
        let g0 = FusionGroup {
            id: 0, anchor: ops[0], epilogue: vec![], mode: FusionMode::Standalone,
            ops: vec![ops[0]], multi_output: MultiOutputConfig::single(), dominant_dtype: None,
        };
        let g1 = FusionGroup {
            id: 1, anchor: ops[1], epilogue: vec![ops[2]], mode: FusionMode::LoopFusion,
            ops: vec![ops[1], ops[2]], multi_output: MultiOutputConfig::single(), dominant_dtype: None,
        };
        let g2 = FusionGroup {
            id: 2, anchor: ops[3], epilogue: vec![ops[4], ops[5]], mode: FusionMode::QkvSharedInput,
            ops: vec![ops[3], ops[4], ops[5]], multi_output: MultiOutputConfig::single(), dominant_dtype: None,
        };
        let plan = FusionPlan {
            groups: vec![g0, g1, g2],
            op_to_group: HashMap::from([
                (ops[0], 0), (ops[1], 1), (ops[2], 1), (ops[3], 2), (ops[4], 2), (ops[5], 2),
            ]),
        };

        // Act
        let fused = plan.num_fused_ops();

        // Assert: g1 contributes 2, g2 contributes 3, g0 excluded. Total = 5.
        assert_eq!(fused, 5);
    }

    #[test]
    fn fusion_mode_loop_fusion_equality_and_debug() {
        // Arrange & Act
        let mode = FusionMode::LoopFusion;

        // Assert
        assert_eq!(mode, FusionMode::LoopFusion);
        assert_ne!(mode, FusionMode::Standalone);
        assert_ne!(mode, FusionMode::EpilogueInjection);
        let debug = format!("{:?}", mode);
        assert!(debug.contains("LoopFusion"));
    }

    #[test]
    fn fusion_mode_qkv_shared_and_norm_into_gem_are_distinct() {
        // Arrange & Act
        let m1 = FusionMode::QkvSharedInput;
        let m2 = FusionMode::NormIntoGemm;

        // Assert: both are distinct from each other and from Standalone.
        assert_ne!(m1, m2);
        assert_ne!(m1, FusionMode::Standalone);
        assert_ne!(m2, FusionMode::Standalone);
        // Debug format includes variant names.
        assert!(format!("{:?}", m1).contains("QkvSharedInput"));
        assert!(format!("{:?}", m2).contains("NormIntoGemm"));
    }

    #[test]
    fn fusion_group_multi_output_config_distinguishes_single_vs_multi() {
        // Arrange
        let op0 = OpId(0);
        let op1 = OpId(1);
        let single_config = MultiOutputConfig::single();

        // Act: single() returns is_multi_output() == false.
        // Assert
        assert!(!single_config.is_multi_output());

        // Build a group with the single config.
        let group = FusionGroup {
            id: 0, anchor: op0, epilogue: vec![op1], mode: FusionMode::Standalone,
            ops: vec![op0, op1], multi_output: single_config, dominant_dtype: None,
        };
        assert!(!group.multi_output.is_multi_output());
    }

    // ── 10 additional tests covering untested logic paths ──

    #[test]
    fn layout_assignment_empty_has_zero_cost_and_benefit() {
        // Arrange & Act
        let la = LayoutAssignment::empty();

        // Assert
        assert!(la.group_assignments.is_empty());
        assert_eq!(la.total_benefit, 0.0);
        assert_eq!(la.total_transform_cost, 0.0);
        assert!(la.all_transforms_free());
    }

    #[test]
    fn layout_assignment_all_transforms_free_when_zero_cost() {
        // Arrange: a LayoutAssignment with zero total_transform_cost.
        let gla = GroupLayoutAssignment {
            group_id: 0,
            op_layouts: HashMap::new(),
            inter_op_transforms: vec![],
            total_benefit: 5.0,
            total_transform_cost: 0.0,
        };
        let la = LayoutAssignment {
            group_assignments: vec![gla],
            total_benefit: 5.0,
            total_transform_cost: 0.0,
        };

        // Act & Assert
        assert!(la.all_transforms_free());
    }

    #[test]
    fn layout_assignment_not_all_free_when_cost_positive() {
        // Arrange
        let gla = GroupLayoutAssignment {
            group_id: 0,
            op_layouts: HashMap::new(),
            inter_op_transforms: vec![],
            total_benefit: 10.0,
            total_transform_cost: 3.5,
        };
        let la = LayoutAssignment {
            group_assignments: vec![gla],
            total_benefit: 10.0,
            total_transform_cost: 3.5,
        };

        // Act & Assert
        assert!(!la.all_transforms_free());
    }

    #[test]
    fn layout_transform_head_split_to_row_major_is_ok() {
        // Arrange: HeadSplit -> RowMajor (reverse direction of existing test).
        let mut prog = VmProgram::new();
        let xform = LayoutTransform {
            source: LayoutConstraint::HeadSplit { num_heads: 16, head_dim: 64 },
            target: LayoutConstraint::RowMajor { align_bytes: 32 },
            cost: 0.0,
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation::default();
        let resolver = make_test_resolver();
        let abi = make_test_abi();

        // Act
        let result = emit_layout_transform(&mut prog, &xform, &graph, &alloc, &resolver, &abi);

        // Assert
        assert!(result.is_ok());
        assert!(prog.is_empty());
    }

    #[test]
    fn movement_type_debug_format_contains_variant_name() {
        // Arrange & Act
        let debug_rd = format!("{:?}", MovementType::RegisterDirect);
        let debug_rtm = format!("{:?}", MovementType::RegisterToMemory);
        let debug_mtm = format!("{:?}", MovementType::MemoryToMemory);
        let debug_gpu = format!("{:?}", MovementType::GpuGlobalToShared);

        // Assert
        assert!(debug_rd.contains("RegisterDirect"));
        assert!(debug_rtm.contains("RegisterToMemory"));
        assert!(debug_mtm.contains("MemoryToMemory"));
        assert!(debug_gpu.contains("GpuGlobalToShared"));
    }

    #[test]
    fn layout_transform_clone_preserves_fields() {
        // Arrange
        let xform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 128 },
            target: LayoutConstraint::ColMajor { align_bytes: 64 },
            cost: 2.75,
        };

        // Act
        let cloned = xform.clone();

        // Assert
        assert_eq!(cloned.cost, 2.75);
        assert!(matches!(cloned.source, LayoutConstraint::RowMajor { align_bytes: 128 }));
        assert!(matches!(cloned.target, LayoutConstraint::ColMajor { align_bytes: 64 }));
    }

    #[test]
    fn group_layout_assignment_debug_format() {
        // Arrange
        let gla = GroupLayoutAssignment {
            group_id: 42,
            op_layouts: HashMap::new(),
            inter_op_transforms: vec![],
            total_benefit: 100.0,
            total_transform_cost: 5.0,
        };

        // Act
        let debug = format!("{:?}", gla);

        // Assert
        assert!(debug.contains("group_id"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn infer_dominant_dtype_f32_tensor() {
        // Arrange: Build a graph with an F32 input tensor.
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor_concrete("hidden_f32", &[1, 256], DType::F32);
        let out_tid = graph.add_tensor_concrete("output_f32", &[1, 256], DType::F32);
        let op_id = graph.add_op(
            OpKind::RmsNorm { eps: 1e-6 },
            vec![tid],
            vec![out_tid],
            "norm_f32",
        );
        let mut group = FusionGroup {
            id: 0,
            anchor: op_id,
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };

        // Act
        group.infer_dominant_dtype(&graph);

        // Assert
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::F32));
    }

    #[test]
    fn infer_dominant_dtype_fp16_tensor() {
        // Arrange: Build a graph with an F16 input tensor.
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor_concrete("hidden_f16", &[2, 128], DType::F16);
        let out_tid = graph.add_tensor_concrete("output_f16", &[2, 128], DType::F16);
        let op_id = graph.add_op(
            OpKind::RmsNorm { eps: 1e-5 },
            vec![tid],
            vec![out_tid],
            "norm_f16",
        );
        let mut group = FusionGroup {
            id: 1,
            anchor: op_id,
            epilogue: vec![],
            mode: FusionMode::LoopFusion,
            ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };

        // Act
        group.infer_dominant_dtype(&graph);

        // Assert
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::F16));
    }

    #[test]
    fn fusion_plan_display_includes_ffn_block_mode() {
        // Arrange: plan with an FFNBlock group to verify Display output.
        use crate::compiler::fusion::FusionPlan;
        let op0 = OpId(0);
        let op1 = OpId(1);
        let op2 = OpId(2);
        let op3 = OpId(3);
        let group = FusionGroup {
            id: 0,
            anchor: op0,
            epilogue: vec![op1, op2, op3],
            mode: FusionMode::FFNBlock {
                gate_gemm: op0,
                up_gemm: op1,
                activation: op2,
                combine: op3,
            },
            ops: vec![op0, op1, op2, op3],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
        };
        let plan = FusionPlan {
            groups: vec![group],
            op_to_group: HashMap::from([(op0, 0), (op1, 0), (op2, 0), (op3, 0)]),
        };

        // Act
        let display = format!("{}", plan);

        // Assert
        assert!(display.contains("1 groups"));
        assert!(display.contains("FFNBlock"));
        assert!(display.contains("anchor=Op(0)"));
        assert!(display.contains("0, 1, 2, 3"));
    }
}
