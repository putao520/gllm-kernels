// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// lower_op_v2 — Op v2 lowering 入口（胖 opcode 驱动，非 OpKind 反查）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Phase 4+: 从 Op Spec 直接获取 lowering 参数，消除 dispatch_emit.rs 中
// `match OpKind { RmsNorm { feature_dim, .. } => ... }` 的 OpKind 反查。
//
// include! 模式：plan_lower.rs 已 use 大部分类型。本文件补 import Op v2 类型。
// emit_normlike_inline/NormKind 已在 plan_lower.rs use，不重复。

use crate::compiler::graph::{CompilerOp, Op, NormSpec, AttentionSpec, KvSource};
use super::attention_emit::emit_tiled_attention_inline;

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
    // 统一 Op v2 转换入口（Phase 4-7 覆盖所有类别）
    let op_v2 = Op::from_op_kind(op, graph);

    let Some(op_v2) = op_v2 else {
        return Ok(false);
    };

    match op_v2 {
        Op::RmsNorm(ref spec) => lower_norm_v2(prog, op, graph, ctx, resolver, abi, spec, NormKind::RmsNorm),
        Op::LayerNorm(ref spec) => lower_norm_v2(prog, op, graph, ctx, resolver, abi, spec, NormKind::LayerNorm),
        Op::ValueNorm(ref spec) => lower_norm_v2(prog, op, graph, ctx, resolver, abi, spec, NormKind::ValueNorm),
        Op::Gemm(ref spec) | Op::GemmBias(ref spec) => {
            lower_gemm_v2(prog, op, graph, ctx, resolver, abi, spec)
        }
        Op::MultiHeadAttention(ref spec) => lower_attention_v2(prog, op, graph, ctx, resolver, abi, spec),
        Op::HeadRmsNorm { .. } => Ok(false),
        _ => Ok(false), // 其他类别走现有路径（Phase 6-7 续迁移）
    }
}

/// Gemm lowering（Op v2 驱动）。
///
/// 从 GemmSpec 获取 m/n/k/trans_b/has_bias，结合 ctx.pack_map_for_gemm，
/// 调用 emit_gemm_inline_with_hook。has_bias 时额外 emit bias add。
fn lower_gemm_v2(
    prog: &mut VmProgram,
    op: &CompilerOp,
    _graph: &CompilerGraph,
    ctx: &LoweringContext,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    spec: &crate::compiler::graph::GemmSpec,
) -> Result<bool, CompilerError> {
    // 物化 a/b/c 指针（通过 resolver，幂等）
    let a_ptr = op.inputs.first().copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("Gemm op {:?}: 输入 tensor 无法 materialize", op.id)))?;
    let b_ptr = op.inputs.get(1).copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("Gemm op {:?}: 权重 tensor 无法 materialize", op.id)))?;
    let c_ptr = op.outputs.first().copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("Gemm op {:?}: 输出 tensor 无法 materialize", op.id)))?;

    // pack_map 从 ctx 获取（权重 tensor 的 packing 信息）
    let weight_tid = op.inputs.get(1).copied();
    let pm = ctx.pack_map_for_gemm(weight_tid);

    // seq_bound_override: mega-kernel decode 时 M=1
    let seq_bound_override = if abi.mega_decode_seq_len.is_some() {
        Some(BoundExpr::Const(1))
    } else {
        None
    };

    emit_gemm_inline_with_hook(
        prog,
        &spec.m, spec.n, spec.k,
        ctx,
        a_ptr, b_ptr, c_ptr,
        seq_bound_override.as_ref(),
        Some(op.id),
        pm,
        spec.trans_b,
    )?;

    // GemmBias: bias add（output += bias broadcast across M rows）
    // GemmBias: bias add（output[i,j] += bias[j]，broadcast across M rows）
    if spec.has_bias {
        if let Some(&bias_tid) = op.inputs.get(2) {
            let bias_ptr = resolver.materialize(prog, bias_tid, abi)
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("GemmBias op {:?}: bias tensor {:?} 无法 materialize", op.id, bias_tid)
                ))?;

            let n_elem = spec.n;
            let elem_bytes = spec.dtype.size_bytes();
            let m_bound = if abi.mega_decode_seq_len.is_some() {
                BoundExpr::Const(1)
            } else {
                resolve_sym_dim(&spec.m, abi, ctx.session.sym_map)
            };
            let row_bytes = n_elem * elem_bytes;
            let row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let width = ctx.session.width;
            let lanes = width.f32_lanes().max(1);
            let dtype_qp = spec.dtype.to_quant_precision();

            prog.emit_loop(m_bound, row_bytes, |prog, _row_ctr, row_off| {
                prog.emit(VmInstr::GprBinOp {
                    dst: row_ptr, a: c_ptr, b: GprOperand::VReg(row_off), op: GprOp::Add,
                });
                let n_vec = n_elem / lanes;
                for vj in 0..n_vec {
                    let byte_off = vj * lanes * elem_bytes;
                    let b_data = prog.alloc_vreg(VRegKind::Vec, width);
                    let c_data = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecLoad { dst: b_data, base: bias_ptr, offset: OffsetExpr::Const(byte_off), width, dtype: dtype_qp });
                    prog.emit(VmInstr::VecLoad { dst: c_data, base: row_ptr, offset: OffsetExpr::Const(byte_off), width, dtype: dtype_qp });
                    prog.emit(VmInstr::VecBinOp { dst: c_data, a: c_data, b: b_data, op: VecOp::Add, dtype: dtype_qp });
                    prog.emit(VmInstr::VecStore { base: row_ptr, offset: OffsetExpr::Const(byte_off), src: c_data, width, dtype: dtype_qp });
                }
                let rem_start = n_vec * lanes;
                for jj in rem_start..n_elem {
                    let byte_off = jj * elem_bytes;
                    let b_s = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
                    let c_s = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
                    prog.emit(VmInstr::VecLoad { dst: b_s, base: bias_ptr, offset: OffsetExpr::Const(byte_off), width: SimdWidth::Scalar, dtype: dtype_qp });
                    prog.emit(VmInstr::VecLoad { dst: c_s, base: row_ptr, offset: OffsetExpr::Const(byte_off), width: SimdWidth::Scalar, dtype: dtype_qp });
                    prog.emit(VmInstr::VecBinOp { dst: c_s, a: c_s, b: b_s, op: VecOp::Add, dtype: dtype_qp });
                    prog.emit(VmInstr::VecStore { base: row_ptr, offset: OffsetExpr::Const(byte_off), src: c_s, width: SimdWidth::Scalar, dtype: dtype_qp });
                }
            });
        }
    }

    Ok(true)
}

/// Attention lowering（Op v2 驱动）。
///
/// 从 AttentionSpec 获取 geometry/mask/kv_source/sinks/dtype，
/// 调用 emit_tiled_attention_inline。
///
/// kv_source=FromCache 走现有路径（KV cache copy 逻辑复杂，Phase 6 续迁移）。
/// kv_source=FromTensor 直接处理（conformer/vision self-attention）。
fn lower_attention_v2(
    prog: &mut VmProgram,
    op: &CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    spec: &AttentionSpec,
) -> Result<bool, CompilerError> {
    // Q/K/V 指针
    let q_ptr = op.inputs.first().copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("MHA op {:?}: Q tensor 无法 materialize", op.id)))?;
    let k_ptr = op.inputs.get(1).copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("MHA op {:?}: K tensor 无法 materialize", op.id)))?;
    let v_ptr = op.inputs.get(2).copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("MHA op {:?}: V tensor 无法 materialize", op.id)))?;
    let output_ptr = op.outputs.first().copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("MHA op {:?}: output tensor 无法 materialize", op.id)))?;

    // sinks
    let sinks_ptr = if matches!(spec.sinks, crate::compiler::graph::SinksSpec::Learnable) {
        op.inputs.get(3).copied()
            .and_then(|tid| resolver.materialize(prog, tid, abi))
    } else {
        None
    };

    // seq bound
    let (q_bound, kv_bound) = if let Some(seq_vreg) = abi.mega_decode_seq_len {
        (BoundExpr::Const(1), BoundExpr::DynamicVReg(seq_vreg))
    } else {
        let bound = resolve_sym_dim(&spec.seq_len, abi, ctx.session.sym_map);
        (bound.clone(), bound)
    };

    let dtype = spec.dtype.to_quant_precision();

    // TMA/TMEM detection (GPU only)
    let use_tma = {
        use crate::compiler::hardware_profile::HardwareProfile;
        use crate::dispatch::DeviceProfile;
        HardwareProfile::detect(&DeviceProfile::detect()).has_tma()
    };
    let use_tmem = {
        use crate::compiler::hardware_profile::HardwareProfile;
        use crate::dispatch::DeviceProfile;
        HardwareProfile::detect(&DeviceProfile::detect()).has_tmem()
    };

    let causal = matches!(spec.mask, crate::compiler::graph::AttentionMask::Causal);

    // KV cache copy（FromCache 路径）— 从 AttentionSpec 获取参数（胖 opcode）
    let (k_attn_ptr, v_attn_ptr) = match spec.kv_source {
        KvSource::FromCache => {
            let kv_cache_ptr = abi.kv_cache_ptr.ok_or_else(|| CompilerError::CodegenViolation(
                format!("MHA op {:?}: kv_source=FromCache 但 ABI 中无 kv_cache_ptr", op.id)))?;

            let layer_ctr = abi.layer_loop_counter.unwrap_or_else(|| {
                let zero = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: zero, value: 0 });
                zero
            });
            let gen_ctr = abi.gen_loop_counter.unwrap_or_else(|| {
                let zero = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: zero, value: 0 });
                zero
            });

            let kv_row_stride = spec.geometry.num_kv_heads * spec.geometry.head_dim * dtype.elem_bytes();
            let max_seq = graph.max_seq_len;
            let kv_layer_stride = 2 * max_seq * kv_row_stride;

            // K cache base = kv_cache_ptr + layer_ctr * kv_layer_stride
            let layer_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: layer_off, a: layer_ctr, b: GprOperand::Imm(kv_layer_stride as i64), op: GprOp::Mul });
            let k_cache_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: k_cache_base, a: kv_cache_ptr, b: GprOperand::VReg(layer_off), op: GprOp::Add });
            let pos_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: pos_off, a: gen_ctr, b: GprOperand::Imm(kv_row_stride as i64), op: GprOp::Mul });

            // Copy K rows to cache
            let k_copy_dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let k_copy_src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let k_off_tmp = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit_loop(q_bound.clone(), kv_row_stride, |prog, _ctr, byte_off| {
                prog.emit(VmInstr::GprBinOp { dst: k_copy_src, a: k_ptr, b: GprOperand::VReg(byte_off), op: GprOp::Add });
                prog.emit(VmInstr::GprBinOp { dst: k_off_tmp, a: pos_off, b: GprOperand::VReg(byte_off), op: GprOp::Add });
                prog.emit(VmInstr::GprBinOp { dst: k_copy_dst, a: k_cache_base, b: GprOperand::VReg(k_off_tmp), op: GprOp::Add });
                prog.emit(VmInstr::MemCopy { dst: k_copy_dst, src: k_copy_src, bytes: kv_row_stride });
            });

            // V cache base = K cache base + max_seq * kv_row_stride
            let v_offset_gpr = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: v_offset_gpr, value: max_seq * kv_row_stride });
            let v_cache_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: v_cache_base, a: k_cache_base, b: GprOperand::VReg(v_offset_gpr), op: GprOp::Add });
            let v_copy_dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let v_copy_src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let v_off_tmp = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit_loop(q_bound.clone(), kv_row_stride, |prog, _ctr, byte_off| {
                prog.emit(VmInstr::GprBinOp { dst: v_copy_src, a: v_ptr, b: GprOperand::VReg(byte_off), op: GprOp::Add });
                prog.emit(VmInstr::GprBinOp { dst: v_off_tmp, a: pos_off, b: GprOperand::VReg(byte_off), op: GprOp::Add });
                prog.emit(VmInstr::GprBinOp { dst: v_copy_dst, a: v_cache_base, b: GprOperand::VReg(k_off_tmp), op: GprOp::Add });
                prog.emit(VmInstr::MemCopy { dst: v_copy_dst, src: v_copy_src, bytes: kv_row_stride });
            });

            (k_cache_base, v_cache_base)
        }
        KvSource::FromTensor => (k_ptr, v_ptr),
    };

    emit_tiled_attention_inline(
        prog, q_bound, kv_bound,
        spec.geometry.num_q_heads, spec.geometry.num_kv_heads, spec.geometry.head_dim,
        ctx.session.width,
        q_ptr, k_attn_ptr, v_attn_ptr, output_ptr,
        ctx.session.hook, causal, sinks_ptr, dtype,
        abi.page_table_ptr, ctx.session.page_size,
        abi.kv_load_mode.unwrap_or_default(), None,
        ctx.session.batch_ctx_ptr, abi.kv_cache_ptr,
        use_tma, use_tmem,
    )?;

    Ok(true)
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
