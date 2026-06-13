// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// lower_op_v2 — Op v2 lowering 入口（胖 opcode 驱动，非 OpKind 反查）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Phase 4+: 从 Op Spec 直接获取 lowering 参数，消除 dispatch_emit.rs 中
// `match OpKind { RmsNorm { feature_dim, .. } => ... }` 的 OpKind 反查。
//
// include! 模式：plan_lower.rs 已 use 大部分类型。本文件补 import Op v2 类型。
// emit_normlike_inline/NormKind 已在 plan_lower.rs use，不重复。

use crate::compiler::graph::{CompilerOp, Op, NormSpec};

/// Phase 4+: Op v2 驱动的 lowering 入口。
///
/// 返回 Ok(true) 表示已处理，Ok(false) 表示非本函数处理的类别（调用方 fallback）。
///
/// Norm 类：从 NormSpec.feature_dim/dtype 直接获取参数，不反查 OpKind。
/// 其他类别：Phase 5-7 扩展。
pub(crate) fn lower_op_v2(
    prog: &mut VmProgram,
    op: &CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
) -> Result<bool, CompilerError> {
    // 尝试用 Op v2 转换（Norm/Activation 类别）
    let op_v2 = Op::from_op_kind_norm_activation(op, graph);

    let Some(op_v2) = op_v2 else {
        return Ok(false); // 非 Norm/Activation，Phase 5-7 处理
    };

    match op_v2 {
        Op::RmsNorm(ref spec) => lower_norm_v2(prog, op, graph, ctx, resolver, abi, spec, NormKind::RmsNorm),
        Op::LayerNorm(ref spec) => lower_norm_v2(prog, op, graph, ctx, resolver, abi, spec, NormKind::LayerNorm),
        Op::ValueNorm(ref spec) => lower_norm_v2(prog, op, graph, ctx, resolver, abi, spec, NormKind::ValueNorm),
        Op::HeadRmsNorm { .. } => Ok(false), // Phase 4 续：head_dim 特化路径
        _ => Ok(false), // Activation 走 trace-lookup 路径
    }
}

/// Norm lowering（Op v2 驱动）。
///
/// 从 NormSpec 获取 feature_dim/dtype/has_weight，结合 registry 的 NormLike pattern，
/// 调用 emit_normlike_inline。消除 dispatch_emit.rs 的 OpKind::RmsNorm 反查。
fn lower_norm_v2(
    prog: &mut VmProgram,
    op: &CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    spec: &NormSpec,
    norm_kind: NormKind,
) -> Result<bool, CompilerError> {
    // 从 registry 获取 NormLike pattern（trace 驱动，auto_select 架构）
    let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
    let trace = ctx.session.registry
        .and_then(|r| r.get_trace(&key))
        .ok_or_else(|| CompilerError::CodegenViolation(format!(
            "lower_norm_v2: 无 registry trace for {:?}", key
        )))?;

    let ComputePattern::NormLike { .. } = &trace.pattern else {
        return Err(CompilerError::CodegenViolation(format!(
            "lower_norm_v2: 期望 NormLike pattern，实际 {:?}", trace.pattern
        )));
    };

    // 从 NormSpec 直接获取参数（胖 opcode，不反查 OpKind）
    let feature_dim = spec.feature_dim;
    let dtype = spec.dtype.to_quant_precision();

    // 输出 tensor 的 seq 维度（用于循环 bound）
    let out_tid = op.outputs.first().copied().ok_or_else(|| CompilerError::CodegenViolation(
        format!("Norm op {:?}: 无输出张量", op.id)))?;
    let out_tensor = graph.tensor(out_tid).ok_or_else(|| CompilerError::CodegenViolation(
        format!("Norm op {:?}: 输出张量 {:?} 不存在", op.id, out_tid)))?;
    let seq_dim = out_tensor.shape.first().cloned().unwrap_or(SymDim::Concrete(1));
    let seq_bound = resolve_sym_dim(&seq_dim, abi, ctx.session.sym_map);

    // 物化输入/权重/输出指针
    let input_tid = op.inputs.first().copied().ok_or_else(|| CompilerError::CodegenViolation(
        format!("Norm op {:?}: 无输入张量", op.id)))?;
    let input_ptr = resolver.materialize(prog, input_tid, abi)
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("Norm op {:?}: 输入 tensor {:?} 无法 materialize", op.id, input_tid)))?;
    let output_ptr = resolver.materialize(prog, out_tid, abi)
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("Norm op {:?}: 输出 tensor {:?} 无法 materialize", op.id, out_tid)))?;

    // 权重指针：has_weight=true 时从 inputs[1] 获取，否则用 input_ptr（无权重 norm）
    let weight_ptr = if spec.has_weight {
        op.inputs.get(1).copied()
            .and_then(|tid| resolver.materialize(prog, tid, abi))
            .unwrap_or(input_ptr)
    } else {
        input_ptr
    };

    emit_normlike_inline(
        prog,
        &trace.pattern,
        feature_dim,
        1, // groups_per_row
        spec.has_weight, // broadcast_weight
        norm_kind,
        ctx.session.width,
        seq_bound,
        input_ptr,
        weight_ptr,
        output_ptr,
        dtype,
    )?;

    Ok(true)
}
