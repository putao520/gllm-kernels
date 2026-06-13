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
        Op::QuantGemm(ref spec) => {
            let m_bound = if abi.mega_decode_seq_len.is_some() {
                BoundExpr::Const(1)
            } else {
                resolve_sym_dim(&spec.m, abi, ctx.session.sym_map)
            };
            let a_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("QuantGemm op {:?}: input 无法 materialize", op.id))
            })?;
            let b_ptr = resolver.materialize(prog, op.inputs[1], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("QuantGemm op {:?}: weight 无法 materialize", op.id))
            })?;
            let c_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("QuantGemm op {:?}: output 无法 materialize", op.id))
            })?;
            super::moe_quant_emit::emit_quant_gemm_inline(prog, m_bound, spec.n, spec.k, spec.quant_type,
                ctx.session.width, a_ptr, b_ptr, c_ptr, ctx.dtype, ctx.session.dot_cap)?;
            Ok(true)
        }
        Op::Residual => {
            let (out_shape, feature_dim) = infer_output_shape_sym(op, graph)?;
            let in_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("Residual op {:?}: input 无法 materialize", op.id))
            })?;
            let w_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(in_ptr);
            let out_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("Residual op {:?}: output 无法 materialize", op.id))
            })?;
            let telemetry_ptr = if graph.telemetry.residual_cosine_sim {
                ctx.session.sym_map.resolve("telemetry").map(|expr| {
                    let ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr_vreg, src: expr.clone() });
                    ptr_vreg
                })
            } else {
                None
            };
            super::telemetry_emit::emit_residual_with_telemetry(
                prog, &out_shape, feature_dim, ctx.session.width,
                in_ptr, w_ptr, out_ptr, ctx.session.sym_map, telemetry_ptr,
                None, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::Argmax { vocab_size } => {
            let logits_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("Argmax op {:?}: logits 无法 materialize", op.id))
            })?;
            let argmax_dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            let vocab_bytes = vocab_size * ctx.dtype.elem_bytes();
            prog.emit(VmInstr::Argmax {
                dst: argmax_dst, logits_ptr, vocab_bytes, width: ctx.session.width,
            });
            let out_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("Argmax op {:?}: output 无法 materialize", op.id))
            })?;
            prog.emit(VmInstr::ScalarStore {
                base: out_ptr, src: argmax_dst, offset: OffsetExpr::Const(0),
            });
            Ok(true)
        }
        Op::GuardrailCheck { probe_offset } => {
            let scratch = match abi.scratch_ptr {
                Some(v) => v, None => return Ok(true),
            };
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("GuardrailCheck op {:?}: input 无法 materialize", op.id))
            })?;
            super::structural_builder::StructuralOpBuilder::emit_conditional_guard(
                prog, scratch, probe_offset, input_ptr,
            )?;
            Ok(true)
        }
        Op::CotStepCheck { shared_mem_offset } => {
            let scratch = match abi.scratch_ptr {
                Some(v) => v, None => return Ok(true),
            };
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("CotStepCheck op {:?}: input 无法 materialize", op.id))
            })?;
            super::structural_builder::StructuralOpBuilder::emit_conditional_guard(
                prog, scratch, shared_mem_offset, input_ptr,
            )?;
            Ok(true)
        }
        Op::WriteLogits { ref target_indices } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("WriteLogits op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("WriteLogits op {:?}: output 无法 materialize", op.id))
            })?;
            super::structural_builder::StructuralOpBuilder::emit_scalar_writeback(
                prog, input_ptr, output_ptr, target_indices,
            )?;
            Ok(true)
        }
        Op::EarlyExit { anchor_layer } => {
            let layer_ctr = abi.layer_loop_counter.ok_or_else(|| CompilerError::CodegenViolation(
                "EarlyExit: layer_loop_counter is None".into()))?;
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("EarlyExit op {:?}: input 无法 materialize", op.id))
            })?;
            prog.emit(VmInstr::GprCondAction {
                cond: GprCondition::CmpEq(layer_ctr, anchor_layer as u64),
                action: GprBranchAction::Exit(input_ptr),
            });
            Ok(true)
        }
        Op::ScaleConst { value } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("ScaleConst op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("ScaleConst op {:?}: output 无法 materialize", op.id))
            })?;
            super::dispatch_emit::lower_scale_const(prog, op, graph, ctx,
                input_ptr, output_ptr, resolver, abi, value)?;
            Ok(true)
        }
        Op::SessionKvRestore => {
            let out_tid = op.outputs.first().ok_or_else(|| CompilerError::CodegenViolation(
                "SessionKvRestore: no output tensor".into()))?;
            let out_shape = &graph.tensor(*out_tid).unwrap().shape;
            let feature_dim: usize = out_shape.iter().filter_map(|d| d.as_concrete()).sum();
            let elem_b = ctx.dtype.elem_bytes();
            let width = ctx.session.width;
            let step = width.f32_lanes() * elem_b;
            let iters = (feature_dim * elem_b + step - 1) / step;
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("SessionKvRestore op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, *out_tid, abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("SessionKvRestore op {:?}: output 无法 materialize", op.id))
            })?;
            let ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
            let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::LoopBegin {
                counter: ctr, byte_offset: byte_off,
                bound: BoundExpr::Const(iters), step_bytes: step,
            });
            let vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: vec, base: input_ptr,
                offset: OffsetExpr::LoopOffset(byte_off), width, dtype: ctx.dtype,
            });
            prog.emit(VmInstr::VecStore {
                base: output_ptr, src: vec,
                offset: OffsetExpr::LoopOffset(byte_off), width, dtype: ctx.dtype,
            });
            prog.emit(VmInstr::LoopEnd);
            Ok(true)
        }
        Op::ColumnSlice { seq_len, input_inner, start, slice_dim } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("ColumnSlice op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("ColumnSlice op {:?}: output 无法 materialize", op.id))
            })?;
            let seq_bound = resolve_sym_dim(&seq_len, abi, ctx.session.sym_map);
            super::structural_emit::emit_column_slice_inline(
                prog, seq_bound, input_inner, start, slice_dim,
                ctx.session.width, input_ptr, output_ptr, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::Gather { table_rows: _, embed_dim, ref index_dim, ref indices_kind, scale } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("Gather op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("Gather op {:?}: output 无法 materialize", op.id))
            })?;
            let seq_bound = if abi.mega_decode_seq_len.is_some() {
                BoundExpr::Const(1)
            } else {
                resolve_sym_dim(index_dim, abi, ctx.session.sym_map)
            };
            let telemetry_ptr = if graph.telemetry.embed_l2_norm {
                ctx.session.sym_map.resolve("telemetry").map(|expr| {
                    let ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr_vreg, src: expr.clone() });
                    ptr_vreg
                })
            } else { None };
            let weight_dtype = op.inputs.get(1)
                .and_then(|&tid| graph.tensor(tid))
                .map(|t| t.dtype.to_quant_precision())
                .unwrap_or(ctx.dtype);
            super::structural_emit::emit_gather_inline(
                prog, seq_bound, embed_dim, ctx.session.width,
                input_ptr, weight_ptr, output_ptr, telemetry_ptr, scale,
                indices_kind.clone(), ctx.dtype, weight_dtype,
            )?;
            Ok(true)
        }
        Op::QuantGather { vocab_size, hidden_dim, ref index_dim, quant_type, scale } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("QuantGather op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .unwrap_or(input_ptr);
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("QuantGather op {:?}: output 无法 materialize", op.id))
            })?;
            let seq_bound = if abi.mega_decode_seq_len.is_some() {
                BoundExpr::Const(1)
            } else {
                resolve_sym_dim(index_dim, abi, ctx.session.sym_map)
            };
            super::quant_gather_emit::emit_quant_gather_inline(
                prog, seq_bound, vocab_size, hidden_dim, quant_type,
                ctx.session.width, input_ptr, weight_ptr, output_ptr,
                ctx.dtype, scale,
            )?;
            Ok(true)
        }
        Op::RoPE(ref spec) => {
            // rope_cache_offset 从 ctx.rope_req 获取（op-level 自描述替代外部参数）
            let rope_req = ctx.rope_req.ok_or_else(|| CompilerError::CodegenViolation(
                format!("RoPE op {:?}: ctx.rope_req 未配置", op.id)))?;
            let cos_sin_offset = rope_req.cache_offset;

            let out_tid = op.outputs.first().copied().ok_or_else(|| CompilerError::CodegenViolation(
                format!("RoPE op {:?}: 无输出张量", op.id)))?;
            let out_tensor = graph.tensor(out_tid).ok_or_else(|| CompilerError::CodegenViolation(
                format!("RoPE op {:?}: 输出张量不存在", op.id)))?;
            let seq_dim = out_tensor.shape.iter().find(|d| d.is_symbolic()).cloned()
                .or_else(|| out_tensor.shape.first().cloned())
                .ok_or_else(|| CompilerError::CodegenViolation(format!(
                    "RoPE op {:?}: 输出 shape 为空", op.id)))?;
            let seq_bound = resolve_sym_dim(&seq_dim, abi, ctx.session.sym_map);

            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("RoPE op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, out_tid, abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("RoPE op {:?}: output 无法 materialize", op.id))
            })?;

            // Mega-kernel: RoPE processes 1 token (current position)
            let (rope_seq_bound, rope_pos_offset) = if let Some(gen_ctr) = abi.gen_loop_counter {
                (BoundExpr::Const(1), Some(gen_ctr))
            } else {
                (seq_bound, None)
            };

            super::structural_emit::emit_rope_inline(
                prog, rope_seq_bound, spec.num_heads, spec.head_dim,
                spec.partial, ctx.session.width,
                input_ptr, output_ptr, cos_sin_offset, ctx.session.sym_map,
                ctx.dtype, rope_pos_offset,
            )?;
            Ok(true)
        }
        Op::MultiHeadAttention(ref spec) => lower_attention_v2(prog, op, graph, ctx, resolver, abi, spec),
        // NOP variants — 元数据 op，不生成 VmInstr（与 dispatch_structural:236 等价）
        Op::Transpose { .. } | Op::Reshape { .. } | Op::SliceView { .. } => Ok(true),

        // Generation control flow — 从 Op v2 路由，逻辑等价（unit 变体，无 Spec 参数）
        Op::StoreToken => {
            let token_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "StoreToken op {:?}: token tensor 无法 materialize", op.id))
            })?;
            let counter = abi.gen_loop_counter.ok_or_else(|| {
                CompilerError::CodegenViolation("StoreToken: gen_loop_counter is None".into())
            })?;
            let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: output_tokens_ptr,
                src: ctx.session.sym_map.resolve("output_tokens_ptr").cloned().unwrap(),
            });
            let input_ids_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: input_ids_ptr,
                src: ctx.session.sym_map.resolve("prompt_len").cloned().unwrap(),
            });
            let prompt_len_bytes = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: prompt_len_bytes,
                src: ctx.session.sym_map.resolve("scratchpad").cloned().unwrap(),
            });
            prog.emit(VmInstr::StoreToken {
                token_id: token_ptr, output_buf: output_tokens_ptr,
                counter, input_ids_ptr, prompt_len_bytes,
            });
            Ok(true)
        }
        Op::CheckStopCondition => {
            let token_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "CheckStopCondition op {:?}: token tensor 无法 materialize", op.id))
            })?;
            let counter = abi.gen_loop_counter.ok_or_else(|| {
                CompilerError::CodegenViolation("CheckStopCondition: gen_loop_counter is None".into())
            })?;
            let eos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: eos_ptr,
                src: ctx.session.sym_map.resolve("eos_token_id").cloned().unwrap(),
            });
            let max_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: max_tokens_ptr,
                src: ctx.session.sym_map.resolve("max_new_tokens").cloned().unwrap(),
            });
            prog.emit(VmInstr::CheckStopCondition {
                token_id: token_ptr, counter, eos_ptr, max_tokens_ptr,
            });
            Ok(true)
        }
        Op::SgInject { knowledge_offset: _, dim } => {
            let sg_base = abi.hook_ctx_ptr.ok_or_else(|| CompilerError::CodegenViolation(
                "SgInject: hook_ctx_ptr is None".into()))?;
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("SgInject op {:?}: input 无法 materialize", op.id))
            })?;
            super::structural_builder::StructuralOpBuilder::emit_simd_injection(
                prog, input_ptr, sg_base,
                12, 16 + dim * 4, dim, ctx.session.width,
            )?;
            Ok(true)
        }
        Op::SgDetect { detect_offset: _, hidden_dim } => {
            let sg_base = abi.hook_ctx_ptr.ok_or_else(|| CompilerError::CodegenViolation(
                "SgDetect: hook_ctx_ptr is None".into()))?;
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("SgDetect op {:?}: input 无法 materialize", op.id))
            })?;
            let detect_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: detect_ptr, src: PtrExpr::VRegPlusConst(sg_base, 16),
            });
            super::structural_builder::StructuralOpBuilder::emit_side_channel_copy(
                prog, input_ptr, detect_ptr, 0, hidden_dim, ctx.session.width,
            )?;
            if abi.callback_table_ptr.is_some() {
                let cb_table = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: cb_table,
                    src: ctx.session.sym_map.resolve("callback_table_ptr").cloned().expect("ABI: callback_table_ptr"),
                });
                prog.emit(VmInstr::GprCondAction { cond: GprCondition::IsNull(cb_table), action: GprBranchAction::Skip(4) });
                let fn_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                let ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadCallbackEntry { table_ptr: cb_table, slot_id: 0, fn_ptr_out: fn_ptr, ctx_out: ctx_ptr });
                prog.emit(VmInstr::MemFence { order: crate::compiler::codegen::vm::instr::MemFenceOrder::Release });
                let ret_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr });
                prog.emit(VmInstr::MemFence { order: crate::compiler::codegen::vm::instr::MemFenceOrder::Acquire });
            }
            Ok(true)
        }
        Op::QTapSTG { sink_ptr, step_index_ptr, dtype, ref q_dim, position, num_slots } => {
            let q_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("QTapSTG op {:?}: Q tensor 无法 materialize", op.id))
            })?;
            let q_dim_concrete = match q_dim {
                SymDim::Concrete(v) => *v,
                SymDim::Symbolic { max_value, .. } => max_value.ok_or_else(|| {
                    CompilerError::CodegenViolation(format!("QTapSTG op {:?}: q_dim Symbolic 无 max_value", op.id))
                })?,
            };
            let q_tensor = graph.tensor(op.inputs[0]).ok_or_else(|| CompilerError::CodegenViolation(
                format!("QTapSTG op {:?}: Q tensor 不存在", op.id)))?;
            let seq_dim = q_tensor.shape.iter().find(|d| d.is_symbolic()).cloned()
                .or_else(|| q_tensor.shape.first().cloned())
                .ok_or_else(|| CompilerError::CodegenViolation(format!("QTapSTG op {:?}: Q shape 为空", op.id)))?;
            let seq_bound = resolve_sym_dim(&seq_dim, abi, ctx.session.sym_map);
            super::lower::lower_qtap_stg(
                prog, sink_ptr, step_index_ptr, dtype,
                q_dim_concrete, seq_bound, position, num_slots, ctx.session.width, q_ptr,
            )?;
            Ok(true)
        }
        Op::DepthwiseConv1D { channels, kernel_size, causal } => {
            let req = ctx.dwc_req.ok_or_else(|| CompilerError::CodegenViolation(
                format!("DepthwiseConv1D op {:?}: ctx.dwc_req 未配置", op.id)))?;
            if req.channels != channels || req.kernel_size != kernel_size || req.causal != causal {
                return Err(CompilerError::CodegenViolation(format!(
                    "DepthwiseConv1D: 签名与 dwc_req 不一致")));
            }
            super::vision_audio_emit::lower_depthwise_conv1d(
                prog, op, graph, ctx.session.width, channels, kernel_size, causal,
                ctx.session.sym_map, resolver, abi, req, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::PatchEmbed { patch_size, embed_dim, in_channels, image_size } => {
            let image_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("PatchEmbed op {:?}: image 无法 materialize", op.id))
            })?;
            let kernel_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("PatchEmbed op {:?}: kernel 无法 materialize", op.id)))?;
            let patches_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("PatchEmbed op {:?}: output 无法 materialize", op.id))
            })?;
            super::vision_audio_emit::lower_patch_embed(
                prog, patch_size, embed_dim, in_channels, image_size,
                image_ptr, kernel_ptr, patches_ptr, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::MoEGate { seq_len: _, num_experts, hidden: _, top_k } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MoEGate op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MoEGate op {:?}: output 无法 materialize", op.id))
            })?;
            let telemetry_ptr = if graph.telemetry.moe_hit_counter {
                ctx.session.sym_map.resolve("telemetry").map(|expr| {
                    let ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr_vreg, src: expr.clone() });
                    ptr_vreg
                })
            } else { None };
            super::norm_softmax_emit::emit_softmax_inline(
                prog, num_experts, ctx.session.width, input_ptr, input_ptr, ctx.dtype,
            )?;
            super::moe_quant_emit::emit_moe_topk_dispatch_inline(
                prog, num_experts, top_k, ctx.session.width,
                input_ptr, output_ptr, ctx.session.hook, telemetry_ptr, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::AltUpPredict { ref seq_len, num_preds, hidden } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("AltUpPredict op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi)).unwrap_or(input_ptr);
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("AltUpPredict op {:?}: output 无法 materialize", op.id))
            })?;
            super::dispatch_emit::lower_altup_predict(
                prog, op, graph, ctx, input_ptr, weight_ptr, output_ptr, resolver, abi,
                seq_len.clone(), num_preds, hidden,
            )?;
            Ok(true)
        }
        Op::AltUpCorrect { ref seq_len, num_preds, hidden } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("AltUpCorrect op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi)).unwrap_or(input_ptr);
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("AltUpCorrect op {:?}: output 无法 materialize", op.id))
            })?;
            super::dispatch_emit::lower_altup_correct(
                prog, op, graph, ctx, input_ptr, weight_ptr, output_ptr, resolver, abi,
                seq_len.clone(), num_preds, hidden,
            )?;
            Ok(true)
        }
        Op::AltUpInject { ref seq_len, num_preds, hidden } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("AltUpInject op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi)).unwrap_or(input_ptr);
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("AltUpInject op {:?}: output 无法 materialize", op.id))
            })?;
            super::dispatch_emit::lower_altup_inject(
                prog, op, graph, ctx, input_ptr, weight_ptr, output_ptr, resolver, abi,
                seq_len.clone(), num_preds, hidden,
            )?;
            Ok(true)
        }
        Op::MoERouter { num_experts, top_k, hidden, seq_len: _ } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MoERouter op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_vreg = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MoERouter op {:?}: weight 无法 materialize", op.id)))?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MoERouter op {:?}: output 无法 materialize", op.id))
            })?;
            let logits_off = top_k * 2 * ctx.dtype.elem_bytes();
            let logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: logits_ptr, src: PtrExpr::VRegPlusConst(output_ptr, logits_off) });
            super::moe_quant_emit::emit_moe_router_gemv_inline(
                prog, num_experts, hidden, ctx.session.width,
                input_ptr, weight_vreg, logits_ptr, ctx.dtype,
            )?;
            super::norm_softmax_emit::emit_softmax_inline(
                prog, num_experts, ctx.session.width, logits_ptr, logits_ptr, ctx.dtype,
            )?;
            super::moe_quant_emit::emit_moe_topk_dispatch_inline(
                prog, num_experts, top_k, ctx.session.width,
                logits_ptr, output_ptr, ctx.session.hook, None, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::MlaKvCompress { ref m, d_c, hidden } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaKvCompress op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MlaKvCompress op {:?}: weight 无法 materialize", op.id)))?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaKvCompress op {:?}: output 无法 materialize", op.id))
            })?;
            let m_bound = if abi.mega_decode_seq_len.is_some() {
                BoundExpr::Const(1)
            } else {
                resolve_sym_dim(m, abi, ctx.session.sym_map)
            };
            emit_gemm_inline_with_hook(prog, m, d_c, hidden, ctx,
                input_ptr, weight_ptr, output_ptr,
                Some(&m_bound), Some(op.id), None, false)?;
            Ok(true)
        }
        Op::MlaQAbsorb { ref seq_len, num_heads, head_dim, d_c } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaQAbsorb op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MlaQAbsorb op {:?}: weight 无法 materialize", op.id)))?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaQAbsorb op {:?}: output 无法 materialize", op.id))
            })?;
            let m_for_gemm = if abi.mega_decode_seq_len.is_some() {
                SymDim::Concrete(num_heads)
            } else { seq_len.clone() };
            emit_gemm_inline_with_hook(prog, &m_for_gemm, d_c, head_dim, ctx,
                input_ptr, weight_ptr, output_ptr, None, Some(op.id), None, true)?;
            Ok(true)
        }
        Op::MlaVRestore { ref seq_len, num_heads, head_dim, d_c } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaVRestore op {:?}: input 无法 materialize", op.id))
            })?;
            let weight_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MlaVRestore op {:?}: weight 无法 materialize", op.id)))?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaVRestore op {:?}: output 无法 materialize", op.id))
            })?;
            let m_for_gemm = if abi.mega_decode_seq_len.is_some() {
                SymDim::Concrete(num_heads)
            } else { seq_len.clone() };
            emit_gemm_inline_with_hook(prog, &m_for_gemm, head_dim, d_c, ctx,
                input_ptr, weight_ptr, output_ptr, None, Some(op.id), None, false)?;
            Ok(true)
        }
        Op::MlaRopeMerge { ref seq_len, d_c, d_rope } => {
            let c_kv_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaRopeMerge op {:?}: c_kv 无法 materialize", op.id))
            })?;
            let k_pe_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MlaRopeMerge op {:?}: k_pe 无法 materialize", op.id)))?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaRopeMerge op {:?}: output 无法 materialize", op.id))
            })?;
            let _ = seq_len;
            let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: cos_ptr,
                src: ctx.session.sym_map.resolve("rope_cos_sin_table").cloned().ok_or_else(|| CompilerError::CodegenViolation(
                    "MlaRopeMerge: rope_cos_sin_table not in sym_map".into()))?,
            });
            let sin_ptr = cos_ptr;
            let position = abi.gen_loop_counter.unwrap_or_else(|| {
                prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar)
            });
            super::mla_emit::emit_mla_rope_merge_inline(
                prog, d_c, d_rope,
                &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, position],
                ctx.session.width, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::MmHiddenInject { hidden_dim } => {
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MmHiddenInject op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MmHiddenInject op {:?}: output 无法 materialize", op.id))
            })?;
            let width = ctx.session.width;
            let total_bytes = hidden_dim * ctx.dtype.elem_bytes();
            let step = width.f32_lanes() * 4;
            let iters = (total_bytes + step - 1) / step;
            let ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
            let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::LoopBegin { counter: ctr, byte_offset: byte_off, bound: BoundExpr::Const(iters), step_bytes: step });
            let src_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad { dst: src_vec, base: input_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype: ctx.dtype });
            let mm_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad { dst: mm_vec, base: input_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype: ctx.dtype });
            let result = prog.alloc_vreg(VRegKind::Vec, width);
            let ple_add_body: Vec<TraceOp> = vec![
                TraceOp::Input(0), TraceOp::Input(1),
                TraceOp::Add(ValueId(0), ValueId(1)),
            ];
            super::auto_select::auto_lower_trace_into(prog, &ple_add_body, &[src_vec, mm_vec], result, width, QuantPrecision::F32)
                .map_err(|e| CompilerError::CodegenViolation(format!("MmHiddenInject auto_lower: {e}")))?;
            prog.emit(VmInstr::VecStore { base: output_ptr, src: result, offset: OffsetExpr::LoopOffset(byte_off), width, dtype: ctx.dtype });
            prog.emit(VmInstr::LoopEnd);
            Ok(true)
        }
        Op::MlaAttention(ref spec) => {
            let q_absorbed_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaAttention op {:?}: q_absorbed 无法 materialize", op.id))
            })?;
            let kv_cache_ptr = op.inputs.get(1).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MlaAttention op {:?}: kv_cache 无法 materialize", op.id)))?;
            let w_uv_ptr = op.inputs.get(2).copied()
                .and_then(|tid| resolver.materialize(prog, tid, abi))
                .ok_or_else(|| CompilerError::CodegenViolation(
                    format!("MlaAttention op {:?}: w_uv 无法 materialize", op.id)))?;
            let output_ptr = resolver.materialize(prog, op.outputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("MlaAttention op {:?}: output 无法 materialize", op.id))
            })?;
            let kv_len_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            if let Some(gen_ctr) = abi.gen_loop_counter {
                prog.emit(VmInstr::GprBinOp { dst: kv_len_vreg, a: gen_ctr, b: GprOperand::Imm(1), op: GprOp::Add });
            } else {
                let seq_val = resolve_sym_dim(&spec.seq_len, abi, ctx.session.sym_map);
                match seq_val {
                    BoundExpr::Const(c) => { prog.emit(VmInstr::GprLoadImm { dst: kv_len_vreg, value: c }); },
                    BoundExpr::DynamicVReg(vr) => {
                        prog.emit(VmInstr::GprBinOp { dst: kv_len_vreg, a: vr, b: GprOperand::Imm(1), op: GprOp::Add });
                    },
                    _ => return Err(CompilerError::CodegenViolation("MlaAttention: unsupported seq_len bound".into())),
                }
            }
            super::mla_emit::emit_mla_attn_score_inline(
                prog, spec.num_heads, spec.head_dim, spec.d_c, spec.d_rope,
                &[q_absorbed_ptr, kv_cache_ptr, w_uv_ptr, output_ptr],
                kv_len_vreg, ctx.session.width, ctx.dtype,
            )?;
            Ok(true)
        }
        Op::DualRoPE(ref spec) => {
            let rope_req = ctx.rope_req.ok_or_else(|| CompilerError::CodegenViolation(
                format!("DualRoPE op {:?}: ctx.rope_req 未配置", op.id)))?;
            let base_offset = rope_req.cache_offset;
            let (sliding_cos_offset, global_cos_offset) = if let Some(ref sec) = rope_req.secondary_cache {
                (base_offset, sec.cache_offset)
            } else {
                return Err(CompilerError::CodegenViolation(
                    "DualRoPE: requires RopeCacheRequirement with secondary_cache".into()));
            };
            let out_tid = op.outputs.first().copied().ok_or_else(|| CompilerError::CodegenViolation(
                format!("DualRoPE op {:?}: 无输出张量", op.id)))?;
            let out_tensor = graph.tensor(out_tid).ok_or_else(|| CompilerError::CodegenViolation(
                format!("DualRoPE op {:?}: 输出张量不存在", op.id)))?;
            let seq_dim = out_tensor.shape.iter().find(|d| d.is_symbolic()).cloned()
                .or_else(|| out_tensor.shape.first().cloned())
                .ok_or_else(|| CompilerError::CodegenViolation(format!("DualRoPE op {:?}: shape 为空", op.id)))?;
            let seq_bound = resolve_sym_dim(&seq_dim, abi, ctx.session.sym_map);
            let input_ptr = resolver.materialize(prog, op.inputs[0], abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("DualRoPE op {:?}: input 无法 materialize", op.id))
            })?;
            let output_ptr = resolver.materialize(prog, out_tid, abi).ok_or_else(|| {
                CompilerError::CodegenViolation(format!("DualRoPE op {:?}: output 无法 materialize", op.id))
            })?;
            let width = ctx.session.width;
            let (rope_seq_bound, rope_pos_offset) = if let Some(gen_ctr) = abi.gen_loop_counter {
                (BoundExpr::Const(1), Some(gen_ctr))
            } else { (seq_bound, None) };
            let layer_ctr = abi.layer_loop_counter.ok_or_else(|| CompilerError::CodegenViolation(
                "DualRoPE: layer_loop_counter is None".into()))?;
            let label_global = prog.alloc_label();
            let label_end = prog.alloc_label();
            let temp_add = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: temp_add, a: layer_ctr, b: GprOperand::Imm(spec.layer_offset as i64), op: GprOp::Add });
            let quotient = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: quotient, a: temp_add, b: GprOperand::Imm(spec.layer_divisor as i64), op: GprOp::Div });
            let product = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: product, a: quotient, b: GprOperand::Imm(spec.layer_divisor as i64), op: GprOp::Mul });
            let remainder = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: remainder, a: temp_add, b: GprOperand::VReg(product), op: GprOp::Sub });
            prog.emit(VmInstr::GprCondAction { cond: GprCondition::CmpEq(remainder, spec.layer_remainder as u64), action: GprBranchAction::JumpToLabel(label_global) });
            super::structural_emit::emit_rope_inline(prog, rope_seq_bound.clone(), spec.num_heads, spec.head_dim,
                spec.sliding_partial, width, input_ptr, output_ptr, sliding_cos_offset, ctx.session.sym_map, ctx.dtype, rope_pos_offset)?;
            prog.emit(VmInstr::GprCondAction { cond: GprCondition::IsNonNull(layer_ctr), action: GprBranchAction::JumpToLabel(label_end) });
            prog.emit(VmInstr::MarkLabel { label_id: label_global });
            super::structural_emit::emit_rope_inline(prog, rope_seq_bound, spec.num_heads, spec.head_dim,
                spec.global_partial, width, input_ptr, output_ptr, global_cos_offset, ctx.session.sym_map, ctx.dtype, rope_pos_offset)?;
            prog.emit(VmInstr::MarkLabel { label_id: label_end });
            Ok(true)
        }
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
