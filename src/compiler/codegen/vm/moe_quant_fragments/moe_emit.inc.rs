
pub(crate) fn emit_moe_router_gemv_inline(
    prog: &mut VmProgram,
    num_experts: usize,
    hidden: usize,
    width: SimdWidth,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    logits_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if num_experts == 0 || hidden == 0 {
        return Err(CompilerError::CodegenViolation(
            format!("emit_moe_router_gemv_inline: invalid params (experts={num_experts}, hidden={hidden})")));
    }
    let elem = dtype.elem_bytes();
    let dot_body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1)), TraceOp::Input(2), TraceOp::Add(ValueId(2), ValueId(3))];

    for ei in 0..num_experts {
        let acc_reg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
        prog.emit(VmInstr::Broadcast { dst: acc_reg, src: ScalarExpr::Const(0.0), width: SimdWidth::Scalar, dtype, });
        let weight_row_off = ei * hidden * elem;
        prog.emit_loop(BoundExpr::Const(hidden), elem, |prog, _d_ctr, d_off| {
            let h_val = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::VecLoad { dst: h_val, base: input_ptr, offset: OffsetExpr::LoopOffset(d_off), width: SimdWidth::Scalar, dtype, });
            let w_val = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::VecLoad { dst: w_val, base: weight_ptr,
                offset: OffsetExpr::Add(Box::new(OffsetExpr::Const(weight_row_off)), Box::new(OffsetExpr::LoopOffset(d_off))),
                width: SimdWidth::Scalar, dtype, });
            super::auto_select::auto_lower_trace_into(
                prog, &dot_body, &[h_val, w_val, acc_reg], acc_reg, SimdWidth::Scalar, QuantPrecision::F32,
            ).expect("moe gemv: dot_body auto_lower failed");
        });
        prog.emit(VmInstr::VecStore { base: logits_ptr, offset: OffsetExpr::Const(ei * elem), src: acc_reg, width: SimdWidth::Scalar, dtype, });
    }
    Ok(())
}

/// 替代 `lower::lower_moe_topk_dispatch` — top-k 选择 + expert dispatch。
/// top_k ≤ UNROLL_THRESHOLD: Rust for 展开合法（编译时确定的小常量）。
/// 算术比较用 ConditionalSkip（结构性控制流）。
pub(crate) fn emit_moe_topk_dispatch_inline(
    prog: &mut VmProgram,
    num_experts: usize,
    top_k: usize,
    width: SimdWidth,
    gate_ptr: VRegId,
    output_ptr: VRegId,
    hook: Option<&dyn isa_hook::IsaHook>,
    telemetry_ptr: Option<VRegId>,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if num_experts == 0 || top_k == 0 || top_k > num_experts {
        return Err(CompilerError::CodegenViolation(
            format!("emit_moe_topk_dispatch_inline: invalid (experts={num_experts}, top_k={top_k})")));
    }
    let elem = dtype.elem_bytes();
    let dispatch = hook.map(|h| h.moe_dispatch(num_experts))
        .unwrap_or_else(|| {
            if num_experts <= 8 { isa_hook::MoeDispatchStrategy::CmpChain }
            else { isa_hook::MoeDispatchStrategy::JmpTable }
        });

    let neg_inf = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: neg_inf, src: ScalarExpr::Const(f32::NEG_INFINITY), width, dtype, });
    let indices_offset = top_k * elem;
    let scalar_w = SimdWidth::Scalar;

    // TraceOp body for top-k update:
    //   Input(0) = best_val, Input(1) = cur_val, Input(2) = best_idx, Input(3) = ei_scalar
    //   Output slots: [4]=new_best, [5]=raw_diff, [6]=clamped_diff, [7]=updated_idx
    let update_body: Vec<TraceOp> = vec![
        TraceOp::Max(ValueId(0), ValueId(1)),      // 4: new_best = max(best_val, cur_val)
        TraceOp::Sub(ValueId(4), ValueId(0)),      // 5: diff = new_best - best_val
        TraceOp::Max(ValueId(5), ValueId(5)),      // 6: diff = max(diff, 0) — identity if positive (Const not needed, Max(x,x)=x hack)
    ];
    // We need a proper clamp: max(diff, 0) then min(result, 1)
    let update_body: Vec<TraceOp> = vec![
        TraceOp::Max(ValueId(0), ValueId(1)),      // 4: new_best
        TraceOp::Sub(ValueId(4), ValueId(0)),      // 5: diff = new_best - best_val
        TraceOp::Const(0.0),     // 6: zero
        TraceOp::Max(ValueId(5), ValueId(6)),      // 7: diff_clamp_lo = max(diff, 0)
        TraceOp::Const(1.0),     // 8: one
        TraceOp::Min(ValueId(7), ValueId(8)),      // 9: clamped_diff = min(diff_clamp_lo, 1)
        TraceOp::Sub(ValueId(3), ValueId(2)),      // 10: delta = ei_scalar - best_idx
        TraceOp::Mul(ValueId(9), ValueId(10)),     // 11: contrib = clamped_diff * delta
        TraceOp::Add(ValueId(11), ValueId(2)),     // 12: updated_idx = contrib + best_idx
    ];
    // Slot indices: 4=new_best, 12=updated_idx

    // TraceOp body for mask-out match detection:
    //   Input(0) = cur_val, Input(1) = best_val
    //   Output slot: [4]=is_match (positive if cur_val ≈ best_val)
    let mask_body: Vec<TraceOp> = vec![
        TraceOp::Sub(ValueId(0), ValueId(1)),      // 2: diff = cur_val - best_val
        TraceOp::Abs(ValueId(2)),         // 3: abs_diff
        TraceOp::Const(1e-6),    // 4: epsilon
        TraceOp::Sub(ValueId(4), ValueId(3)),      // 5: is_match = epsilon - abs_diff
    ];
    // Slot index: 5=is_match

    for ki in 0..top_k {
        // Control flow: initialize accumulators (Broadcast = parameter setup)
        let best_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
        prog.emit(VmInstr::Broadcast { dst: best_val, src: ScalarExpr::Const(f32::NEG_INFINITY), width: scalar_w, dtype, });
        let best_idx = prog.alloc_vreg(VRegKind::Vec, scalar_w);
        prog.emit(VmInstr::Broadcast { dst: best_idx, src: ScalarExpr::Const(0.0), width: scalar_w, dtype, });

        for ei in 0..num_experts {
            // Control flow: load current expert value
            let cur_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::VecLoad { dst: cur_val, base: gate_ptr, offset: OffsetExpr::Const(ei * elem), width: scalar_w, dtype, });
            // Control flow: set up expert index scalar
            let ei_scalar = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::Broadcast { dst: ei_scalar, src: ScalarExpr::Const(ei as f32), width: scalar_w, dtype, });

            // Arithmetic: top-k update via auto_lower_trace_raw
            let slots = super::auto_select::auto_lower_trace_raw(
                prog, &update_body, &[best_val, cur_val, best_idx, ei_scalar], scalar_w, QuantPrecision::F32)?;
            let new_best = slots[4];
            let updated_idx = slots[12];
            // Copy results back to accumulators via auto_lower_trace_into
            let identity: Vec<TraceOp> = vec![TraceOp::Input(0)];
            super::auto_select::auto_lower_trace_into(prog, &identity, &[updated_idx], best_idx, scalar_w, QuantPrecision::F32).expect("best_idx identity auto_lower failed");
            super::auto_select::auto_lower_trace_into(prog, &identity, &[new_best], best_val, scalar_w, QuantPrecision::F32).expect("best_val identity auto_lower failed");
        }

        // Control flow: store results
        prog.emit(VmInstr::VecStore { base: output_ptr, offset: OffsetExpr::Const(ki * elem), src: best_val, width: scalar_w, dtype, });
        prog.emit(VmInstr::VecStore { base: output_ptr, offset: OffsetExpr::Const(indices_offset + ki * elem), src: best_idx, width: scalar_w, dtype, });

        // Telemetry: expert hit counts (control flow + arithmetic via TraceOp)
        if let Some(tel_ptr) = telemetry_ptr {
            let base_offset = crate::compiler::graph::telemetry_offsets::EXPERT_HIT_COUNTS_OFFSET;
            let tel_body: Vec<TraceOp> = vec![
                TraceOp::Sub(ValueId(0), ValueId(1)),      // 2: diff = best_idx - e_scalar
                TraceOp::Abs(ValueId(2)),         // 3: abs_diff
                TraceOp::Const(0.5),     // 4: half
                TraceOp::Min(ValueId(3), ValueId(4)),      // 5: clamped = min(abs_diff, 0.5)
                TraceOp::Sub(ValueId(4), ValueId(5)),      // 6: negated = 0.5 - clamped (positive if match)
            ];
            for e in 0..num_experts {
                let e_scalar = prog.alloc_vreg(VRegKind::Vec, scalar_w);
                prog.emit(VmInstr::Broadcast { dst: e_scalar, src: ScalarExpr::Const(e as f32), width: scalar_w, dtype, });
                let slots = super::auto_select::auto_lower_trace_raw(
                    prog, &tel_body, &[best_idx, e_scalar], scalar_w, QuantPrecision::F32)?;
                // Control flow: conditional skip + atomic update
                prog.emit(VmInstr::ConditionalSkip { mask: slots[6], skip_count: 1 });
                prog.emit(VmInstr::AtomicAdd { base: tel_ptr, offset: OffsetExpr::Const(base_offset + e * elem), value: 1, elem_width: 4 });
            }
        }

        // Mask out selected expert (control flow + arithmetic via TraceOp)
        for ei in 0..num_experts {
            let cur_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::VecLoad { dst: cur_val, base: gate_ptr, offset: OffsetExpr::Const(ei * elem), width: scalar_w, dtype, });
            let slots = super::auto_select::auto_lower_trace_raw(
                prog, &mask_body, &[cur_val, best_val], scalar_w, QuantPrecision::F32)?;
            // Control flow: conditional skip (if match, overwrite with -inf)
            prog.emit(VmInstr::ConditionalSkip { mask: slots[5], skip_count: 1 });
            prog.emit(VmInstr::VecStore { base: gate_ptr, offset: OffsetExpr::Const(ei * elem), src: neg_inf, width: scalar_w, dtype, });
        }
    }

    // Expert dispatch (pure control flow)
    let expert_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad { dst: expert_idx, base: output_ptr, offset: OffsetExpr::Const(0) });
    match dispatch {
        isa_hook::MoeDispatchStrategy::JmpTable | isa_hook::MoeDispatchStrategy::InKernelJmp => {
            let targets: Vec<JumpTarget> = (0..num_experts).map(|e| JumpTarget { expert_id: e, instr_index: 0 }).collect();
            prog.emit(VmInstr::IndirectJump { index: expert_idx, targets });
        }
        isa_hook::MoeDispatchStrategy::CmpChain => {
            for e in 0..num_experts {
                let cmp_val = prog.alloc_vreg(VRegKind::Mask, SimdWidth::Scalar);
                prog.emit(VmInstr::Broadcast { dst: cmp_val, src: ScalarExpr::Const(e as f32), width: SimdWidth::Scalar, dtype, });
                prog.emit(VmInstr::ConditionalSkip { mask: cmp_val, skip_count: 1 });
                prog.emit(VmInstr::HotpatchSlot { slot_id: e as u32, initial_target: HotpatchTarget::InstrIndex(0), alternatives: vec![] });
            }
        }
        isa_hook::MoeDispatchStrategy::Predicated => {
            let targets: Vec<JumpTarget> = (0..num_experts).map(|e| JumpTarget { expert_id: e, instr_index: 0 }).collect();
            prog.emit(VmInstr::IndirectJump { index: expert_idx, targets });
        }
    }
    Ok(())
}

/// 替代 `lower::lower_moe_dispatch_packed` — 完整 MoE 管线 (GEMV + softmax + top-k +
/// dequant + SwiGLU + down GEMV + weighted accumulate)，所有循环使用 `emit_loop`。
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_moe_packed_inline(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    num_experts: usize,
    top_k: usize,
    mxfp4_block_size: usize,
    swiglu_limit: f32,
    intermediate_size: usize,
    hidden: usize,
    width: SimdWidth,
    hidden_input_ptr: VRegId,
    router_weight_ptr: VRegId,
    router_bias_ptr: VRegId,
    gate_up_blocks_ptr: VRegId,
    gate_up_scales_ptr: VRegId,
    gate_up_bias_ptr: VRegId,
    down_blocks_ptr: VRegId,
    down_scales_ptr: VRegId,
    down_bias_ptr: VRegId,
    output_ptr: VRegId,
    scratchpad_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if mxfp4_block_size == 0 || mxfp4_block_size % 2 != 0 {
        return Err(CompilerError::CodegenViolation(
            format!("emit_moe_packed_inline: mxfp4_block_size must be even and > 0, got {mxfp4_block_size}")));
    }
    if intermediate_size % mxfp4_block_size != 0 || hidden % mxfp4_block_size != 0 {
        return Err(CompilerError::CodegenViolation(
            format!("emit_moe_packed_inline: intermediate_size {intermediate_size} and hidden {hidden} must be divisible by block_size {mxfp4_block_size}")));
    }

    let elem = dtype.elem_bytes();
    let lanes = width.f32_lanes().max(1);

    let gu_size = 2 * intermediate_size;
    let gu_bytes = gu_size * elem;
    let gu_blocks_per_expert = gu_size / mxfp4_block_size;
    let gu_block_bytes = mxfp4_block_size / 2;
    let gu_vecs_per_block = mxfp4_block_size / lanes;

    let intermediate_vecs = intermediate_size / lanes;
    let intermediate_row_packed_bytes = intermediate_size / 2;

    let down_blocks_per_expert = hidden / mxfp4_block_size;
    let down_block_bytes = intermediate_size * mxfp4_block_size / 2;

    let gu_scales_stride = gu_blocks_per_expert;
    let gu_blocks_stride = gu_blocks_per_expert * gu_block_bytes;
    let gu_bias_stride = gu_size * elem;
    let down_scales_stride = down_blocks_per_expert;
    let down_blocks_stride = down_blocks_per_expert * down_block_bytes;
    let down_bias_stride = hidden * elem;

    // Scratchpad layout
    let router_weights_off: usize = 0;
    let router_indices_off: usize = top_k * elem;
    let logits_off: usize = 2 * top_k * elem;
    let logits_bytes: usize = num_experts * elem;
    let gu_off: usize = logits_off + logits_bytes;
    let activ_offset: usize = gu_off + gu_bytes;

    // Broadcast limit constants for clipped SwiGLU
    let limit_neg = prog.alloc_vreg(VRegKind::Vec, width);
    let limit_pos = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: limit_neg, src: ScalarExpr::Const(-swiglu_limit), width, dtype, });
    prog.emit(VmInstr::Broadcast { dst: limit_pos, src: ScalarExpr::Const(swiglu_limit), width, dtype, });

    let seq_step = hidden * elem;
    let scalar_w = SimdWidth::Scalar;

    // TraceOp bodies for auto instruction selection
    let gemv_dot_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
        TraceOp::Mul(ValueId(0), ValueId(1)),  // 3: prod = h * w
        TraceOp::Add(ValueId(2), ValueId(3)),  // 4: new_acc = acc + prod
    ];
    let softmax_max_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Max(ValueId(0), ValueId(1)),  // 2: new_max
    ];
    let softmax_exp_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Sub(ValueId(0), ValueId(1)),  // 2: shifted = cur - max
        TraceOp::Exp(ValueId(2)),     // 3: exp_val
    ];
    let softmax_sum_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Add(ValueId(0), ValueId(1)), // 2: new_sum
    ];
    let softmax_norm_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Mul(ValueId(0), ValueId(1)), // 2: normalized
    ];
    let topk_compare_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Sub(ValueId(0), ValueId(1)),  // 2: diff = cur - best
        TraceOp::Const(0.0), // 3: zero
        TraceOp::Max(ValueId(2), ValueId(3)),  // 4: is_better = max(diff, 0)
    ];
    let topk_mask_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
        TraceOp::Sub(ValueId(0), ValueId(1)),  // 3: idx_diff = ei - best_idx
        TraceOp::Abs(ValueId(3)),     // 4: abs_diff
        TraceOp::Min(ValueId(4), ValueId(2)),  // 5: clamped
        TraceOp::Sub(ValueId(2), ValueId(5)),  // 6: negated = half - clamped
    ];
    let dequant_bias_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Add(ValueId(0), ValueId(1)), // 2: dequant + bias
    ];
    let swiglu_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), // 0: gate
        TraceOp::Input(1), // 1: up
        TraceOp::Input(2), // 2: limit_neg
        TraceOp::Input(3), // 3: limit_pos
        TraceOp::Max(ValueId(0), ValueId(2)),  // 4: gate_lo
        TraceOp::Min(ValueId(4), ValueId(3)),  // 5: gate_clamped
        TraceOp::Max(ValueId(1), ValueId(2)),  // 6: up_lo
        TraceOp::Min(ValueId(6), ValueId(3)),  // 7: up_clamped
        TraceOp::Sigmoid(ValueId(5)), // 8: sigmoid_gate
        TraceOp::Mul(ValueId(5), ValueId(8)),  // 9: silu
        TraceOp::Mul(ValueId(9), ValueId(7)),  // 10: activ
    ];
    let down_accumulate_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), // 0: acc_reduced
        TraceOp::Input(1), // 1: bias
        TraceOp::Input(2), // 2: weight
        TraceOp::Input(3), // 3: old_out
        TraceOp::Add(ValueId(0), ValueId(1)),  // 4: with_bias
        TraceOp::Mul(ValueId(4), ValueId(2)),  // 5: weighted
        TraceOp::Add(ValueId(5), ValueId(3)),  // 6: new_out
    ];

    prog.emit_loop_try(seq_bound, seq_step, |prog, _seq_ctr, seq_off| -> Result<(), CompilerError> {
        // ═══ Phase A: Inline routing (GEMV + softmax + top-k) ═══
        // A1: GEMV — logits[e] = hidden · weight[e] + bias[e]
        prog.emit_loop_try(BoundExpr::Const(num_experts), hidden * elem, |prog, e_ctr, e_off| -> Result<(), CompilerError> {
            let acc_reg = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::Broadcast { dst: acc_reg, src: ScalarExpr::Const(0.0), width: scalar_w, dtype, });
            prog.emit_loop_try(BoundExpr::Const(hidden), elem, |prog, _d_ctr, d_off| -> Result<(), CompilerError> {
                let h_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
                prog.emit(VmInstr::VecLoad { dst: h_val, base: hidden_input_ptr, offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)), Box::new(OffsetExpr::LoopOffset(d_off)),
                ), width: scalar_w, dtype, });
                let w_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
                prog.emit(VmInstr::VecLoad { dst: w_val, base: router_weight_ptr, offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(e_off)), Box::new(OffsetExpr::LoopOffset(d_off)),
                ), width: scalar_w, dtype, });
                super::auto_select::auto_lower_trace_into(prog, &gemv_dot_body, &[h_val, w_val, acc_reg], acc_reg, scalar_w, QuantPrecision::F32)?;
                Ok(())
            })?;
            // Add bias
            let bias_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::VecLoad { dst: bias_val, base: router_bias_ptr,
                offset: OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(e_ctr)), elem),
                width: scalar_w, dtype, });
            let slots = super::auto_select::auto_lower_trace_raw(prog, &dequant_bias_body, &[acc_reg, bias_val], scalar_w, QuantPrecision::F32)?;
            prog.emit(VmInstr::VecStore { base: scratchpad_ptr, offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(logits_off)),
                Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(e_ctr)), elem)),
            ), src: slots[2], width: scalar_w, dtype, });
            Ok(())
        })?;

        // A2: Softmax over num_experts
        let max_acc = prog.alloc_vreg(VRegKind::Vec, scalar_w);
        prog.emit(VmInstr::Broadcast { dst: max_acc, src: ScalarExpr::Const(f32::NEG_INFINITY), width: scalar_w, dtype, });
        prog.emit_loop_try(BoundExpr::Const(num_experts), elem, |prog, _e_ctr, e_off| -> Result<(), CompilerError> {
            let cur = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::VecLoad { dst: cur, base: scratchpad_ptr, offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(logits_off)), Box::new(OffsetExpr::LoopOffset(e_off)),
            ), width: scalar_w, dtype, });
            super::auto_select::auto_lower_trace_into(prog, &softmax_max_body, &[max_acc, cur], max_acc, scalar_w, QuantPrecision::F32)?;
            Ok(())
        })?;

        let sum_acc = prog.alloc_vreg(VRegKind::Vec, scalar_w);
        prog.emit(VmInstr::Broadcast { dst: sum_acc, src: ScalarExpr::Const(0.0), width: scalar_w, dtype, });
        prog.emit_loop_try(BoundExpr::Const(num_experts), elem, |prog, _e_ctr, e_off| -> Result<(), CompilerError> {
            let cur = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::VecLoad { dst: cur, base: scratchpad_ptr, offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(logits_off)), Box::new(OffsetExpr::LoopOffset(e_off)),
            ), width: scalar_w, dtype, });
            let slots = super::auto_select::auto_lower_trace_raw(prog, &softmax_exp_body, &[cur, max_acc], scalar_w, QuantPrecision::F32)?;
            let exp_val = slots[3];
            prog.emit(VmInstr::VecStore { base: scratchpad_ptr, offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(logits_off)), Box::new(OffsetExpr::LoopOffset(e_off)),
            ), src: exp_val, width: scalar_w, dtype, });
            super::auto_select::auto_lower_trace_into(prog, &softmax_sum_body, &[sum_acc, exp_val], sum_acc, scalar_w, QuantPrecision::F32)?;
            Ok(())
        })?;
        // Recip(sum) for normalization — arithmetic via TraceOp
        let recip_body: Vec<TraceOp> = vec![TraceOp::Input(0), TraceOp::Recip(ValueId(0))];
        super::auto_select::auto_lower_trace_into(prog, &recip_body, &[sum_acc], sum_acc, scalar_w, QuantPrecision::F32)?;

        prog.emit_loop_try(BoundExpr::Const(num_experts), elem, |prog, _e_ctr, e_off| -> Result<(), CompilerError> {
            let cur = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::VecLoad { dst: cur, base: scratchpad_ptr, offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(logits_off)), Box::new(OffsetExpr::LoopOffset(e_off)),
            ), width: scalar_w, dtype, });
            let slots = super::auto_select::auto_lower_trace_raw(prog, &softmax_norm_body, &[cur, sum_acc], scalar_w, QuantPrecision::F32)?;
            prog.emit(VmInstr::VecStore { base: scratchpad_ptr, offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(logits_off)), Box::new(OffsetExpr::LoopOffset(e_off)),
            ), src: slots[2], width: scalar_w, dtype, });
            Ok(())
        })?;

        // A3: Top-K selection
        let neg_inf = prog.alloc_vreg(VRegKind::Vec, scalar_w);
        prog.emit(VmInstr::Broadcast { dst: neg_inf, src: ScalarExpr::Const(f32::NEG_INFINITY), width: scalar_w, dtype, });
        let half_v = prog.alloc_vreg(VRegKind::Vec, scalar_w);
        prog.emit(VmInstr::Broadcast { dst: half_v, src: ScalarExpr::Const(0.5), width: scalar_w, dtype, });

        for ki in 0..top_k {
            let best_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::Broadcast { dst: best_val, src: ScalarExpr::Const(f32::NEG_INFINITY), width: scalar_w, dtype, });
            let best_idx = prog.alloc_vreg(VRegKind::Vec, scalar_w);
            prog.emit(VmInstr::Broadcast { dst: best_idx, src: ScalarExpr::Const(0.0), width: scalar_w, dtype, });

            for ei in 0..num_experts {
                // Control flow: load current expert value
                let cur_val = prog.alloc_vreg(VRegKind::Vec, scalar_w);
                prog.emit(VmInstr::VecLoad { dst: cur_val, base: scratchpad_ptr, offset: OffsetExpr::Const(logits_off + ei * elem), width: scalar_w, dtype, });
                // Arithmetic: compare via auto_lower_trace_raw
                let slots = super::auto_select::auto_lower_trace_raw(prog, &topk_compare_body, &[cur_val, best_val], scalar_w, QuantPrecision::F32)?;
                // Control flow: conditional update
                prog.emit(VmInstr::ConditionalSkip { mask: slots[4], skip_count: 2 });
                prog.emit(VmInstr::Broadcast { dst: best_idx, src: ScalarExpr::Const(ei as f32), width: scalar_w, dtype, });
                let identity: Vec<TraceOp> = vec![TraceOp::Input(0)];
                super::auto_select::auto_lower_trace_into(prog, &identity, &[cur_val], best_val, scalar_w, QuantPrecision::F32).expect("best_val copy auto_lower failed");
            }

            prog.emit(VmInstr::VecStore { base: scratchpad_ptr, offset: OffsetExpr::Const(router_weights_off + ki * elem), src: best_val, width: scalar_w, dtype, });
            prog.emit(VmInstr::VecStore { base: scratchpad_ptr, offset: OffsetExpr::Const(router_indices_off + ki * elem), src: best_idx, width: scalar_w, dtype, });

            // Mask out selected expert
            for ei in 0..num_experts {
                let ei_scalar = prog.alloc_vreg(VRegKind::Vec, scalar_w);
                prog.emit(VmInstr::Broadcast { dst: ei_scalar, src: ScalarExpr::Const(ei as f32), width: scalar_w, dtype, });
                let slots = super::auto_select::auto_lower_trace_raw(prog, &topk_mask_body, &[ei_scalar, best_idx, half_v], scalar_w, QuantPrecision::F32)?;
                prog.emit(VmInstr::ConditionalSkip { mask: slots[6], skip_count: 1 });
                prog.emit(VmInstr::VecStore { base: scratchpad_ptr, offset: OffsetExpr::Const(logits_off + ei * elem), src: neg_inf, width: scalar_w, dtype, });
            }
        }

        // ═══ Phase B: Expert dispatch (dequant → SwiGLU → down project) ═══
        for k in 0..top_k {
            // B1: Load expert_id and weight
            let idx_scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: idx_scalar, base: scratchpad_ptr, offset: OffsetExpr::Const(router_indices_off + k * elem) });
            let expert_id_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: expert_id_gpr, src: idx_scalar, stride: 1 });

            let weight_scalar = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::VecLoad { dst: weight_scalar, base: scratchpad_ptr, offset: OffsetExpr::Const(router_weights_off + k * elem), width: SimdWidth::Scalar, dtype, });
            let weight_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: weight_vec, src: ScalarExpr::ExtractLane0(weight_scalar), width, dtype, });

            // B2: Compute expert base pointers
            let gu_scales_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: gu_scales_off, src: idx_scalar, stride: gu_scales_stride });
            let expert_gu_scales_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: expert_gu_scales_ptr, src: PtrExpr::VRegPlusVReg(gate_up_scales_ptr, gu_scales_off) });

            let gu_blocks_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: gu_blocks_off, src: idx_scalar, stride: gu_blocks_stride });
            let expert_gu_blocks_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: expert_gu_blocks_ptr, src: PtrExpr::VRegPlusVReg(gate_up_blocks_ptr, gu_blocks_off) });

            let gu_bias_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: gu_bias_off, src: idx_scalar, stride: gu_bias_stride });
            let expert_gu_bias_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: expert_gu_bias_ptr, src: PtrExpr::VRegPlusVReg(gate_up_bias_ptr, gu_bias_off) });

            // B3: Dequant gate_up → scratchpad[gu_off]
            prog.emit_loop(BoundExpr::Const(gu_blocks_per_expert), 1, |prog, _blk_ctr, blk_off| {
                let scale_byte_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarByteLoad { dst: scale_byte_gpr, base: expert_gu_scales_ptr, offset: OffsetExpr::LoopOffset(blk_off) });

                for vec_idx in 0..gu_vecs_per_block {
                    // Dequant via TraceOp::Mxfp4Dequant (offset = blk_off * gu_block_bytes + vec_idx * lanes/2)
                    let dequant_body: Vec<TraceOp> = vec![
                        TraceOp::Mxfp4Dequant {
                            data: ValueId(0), scales: ValueId(1),
                            off_a: Some(ValueId(2)), stride_a: gu_block_bytes,
                            off_b: None, stride_b: 0,
                            off_c: None,
                            const_off: vec_idx * (lanes / 2),
                            block_size: mxfp4_block_size,
                        },
                    ];
                    let dequant_slots = super::auto_select::auto_lower_trace_raw(
                        prog, &dequant_body, &[expert_gu_blocks_ptr, scale_byte_gpr, blk_off], width, QuantPrecision::F32).expect("gate_up dequant auto_lower failed");
                    let dequant_vec = dequant_slots[0];

                    let bias_vec = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecLoad { dst: bias_vec, base: expert_gu_bias_ptr,
                        offset: OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(blk_off)), mxfp4_block_size * elem)),
                            Box::new(OffsetExpr::Const(vec_idx * lanes * elem)),
                        ), width, dtype, });
                    // Arithmetic: dequant + bias via auto_lower_trace_raw
                    let slots = super::auto_select::auto_lower_trace_raw(prog, &dequant_bias_body, &[dequant_vec, bias_vec], width, QuantPrecision::F32).expect("dequant_bias_body auto_lower failed");

                    prog.emit(VmInstr::VecStore { base: scratchpad_ptr,
                        offset: OffsetExpr::Add(
                            Box::new(OffsetExpr::Const(gu_off)),
                            Box::new(OffsetExpr::Add(
                                Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(blk_off)), mxfp4_block_size * elem)),
                                Box::new(OffsetExpr::Const(vec_idx * lanes * elem)),
                            )),
                        ), src: slots[2], width, dtype, });
                }
            });

            // B4: Clipped SwiGLU — clamp + sigmoid + gate*sigmoid*up via auto_lower_trace_raw
            prog.emit_loop(BoundExpr::Const(intermediate_vecs), lanes * elem, |prog, _v_ctr, v_off| {
                let gate_vec = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: gate_vec, base: scratchpad_ptr,
                    offset: OffsetExpr::Add(Box::new(OffsetExpr::Const(gu_off)), Box::new(OffsetExpr::LoopOffset(v_off))), width, dtype, });

                let up_vec = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: up_vec, base: scratchpad_ptr,
                    offset: OffsetExpr::Add(Box::new(OffsetExpr::Const(gu_off + intermediate_size * elem)), Box::new(OffsetExpr::LoopOffset(v_off))), width, dtype, });

                // Arithmetic: full SwiGLU (clamp + sigmoid + gate*scaled + silu*up) via auto_lower_trace_raw
                let slots = super::auto_select::auto_lower_trace_raw(
                    prog, &swiglu_body, &[gate_vec, up_vec, limit_neg, limit_pos], width, QuantPrecision::F32).expect("swiglu_body auto_lower failed");

                prog.emit(VmInstr::VecStore { base: scratchpad_ptr,
                    offset: OffsetExpr::Add(Box::new(OffsetExpr::Const(activ_offset)), Box::new(OffsetExpr::LoopOffset(v_off))),
                    src: slots[10], width, dtype, });
            });

            // B5: Down GEMV + bias + weighted accumulate
            let down_scales_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: down_scales_off, src: idx_scalar, stride: down_scales_stride });
            let expert_down_scales_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: expert_down_scales_ptr, src: PtrExpr::VRegPlusVReg(down_scales_ptr, down_scales_off) });

            let down_blocks_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: down_blocks_off, src: idx_scalar, stride: down_blocks_stride });
            let expert_down_blocks_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: expert_down_blocks_ptr, src: PtrExpr::VRegPlusVReg(down_blocks_ptr, down_blocks_off) });

            let down_bias_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarToIndex { dst: down_bias_off, src: idx_scalar, stride: down_bias_stride });
            let expert_down_bias_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: expert_down_bias_ptr, src: PtrExpr::VRegPlusVReg(down_bias_ptr, down_bias_off) });

            prog.emit_loop(BoundExpr::Const(down_blocks_per_expert), 1, |prog, _blk_ctr, blk_off| {
                let scale_byte_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarByteLoad { dst: scale_byte_gpr, base: expert_down_scales_ptr, offset: OffsetExpr::LoopOffset(blk_off) });

                prog.emit_loop(BoundExpr::Const(mxfp4_block_size), elem, |prog, _row_ctr, row_off| {
                    let acc = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype, });

                    let dot_step = lanes / 2;
                    prog.emit_loop(BoundExpr::Const(intermediate_vecs), dot_step, |prog, _sv_ctr, sv_off| {
                        // Dequant via TraceOp::Mxfp4Dequant (3-component offset)
                        // offset = blk_off * down_block_bytes + row_off * (intermediate_row_packed_bytes/elem) + sv_off
                        let down_dequant_body: Vec<TraceOp> = vec![
                            TraceOp::Mxfp4Dequant {
                                data: ValueId(0), scales: ValueId(1),
                                off_a: Some(ValueId(2)), stride_a: down_block_bytes,
                                off_b: Some(ValueId(3)), stride_b: intermediate_row_packed_bytes / elem,
                                off_c: Some(ValueId(4)),
                                const_off: 0,
                                block_size: mxfp4_block_size,
                            },
                        ];
                        let dq_slots = super::auto_select::auto_lower_trace_raw(
                            prog, &down_dequant_body,
                            &[expert_down_blocks_ptr, scale_byte_gpr, blk_off, row_off, sv_off],
                            width, QuantPrecision::F32).expect("down dequant auto_lower failed");
                        let down_vec = dq_slots[0];

                        let activ_vec = prog.alloc_vreg(VRegKind::Vec, width);
                        prog.emit(VmInstr::VecLoad { dst: activ_vec, base: scratchpad_ptr,
                            offset: OffsetExpr::Add(
                                Box::new(OffsetExpr::Const(activ_offset)),
                                Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(sv_off)), lanes * elem / dot_step)),
                            ), width, dtype, });

                        // Arithmetic: FMA via auto_lower_trace_raw
                        let fma_body: Vec<TraceOp> = vec![
                            TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
                            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)), // 3: new_acc = down * activ + acc
                        ];
                        super::auto_select::auto_lower_trace_into(prog, &fma_body, &[down_vec, activ_vec, acc], acc, width, QuantPrecision::F32).expect("down FMA auto_lower failed");
                    });

                    // Arithmetic: HReduce via auto_lower_trace_raw (scalar output → broadcast)
                    let mha_hr_body = vec![TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
                    let hr_slots = super::auto_select::auto_lower_trace_raw(prog, &mha_hr_body, &[acc], width, QuantPrecision::F32).expect("MoE packed HReduce auto_lower failed");
                    let acc_reduced = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::Broadcast { dst: acc_reduced, src: ScalarExpr::ExtractLane0(hr_slots[1]), width, dtype, });

                    // Control flow: load bias
                    let bias_val = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
                    prog.emit(VmInstr::VecLoad { dst: bias_val, base: expert_down_bias_ptr,
                        offset: OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(blk_off)), mxfp4_block_size * elem)),
                            Box::new(OffsetExpr::LoopOffset(row_off)),
                        ), width: SimdWidth::Scalar, dtype, });
                    let bias_broadcast = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::Broadcast { dst: bias_broadcast, src: ScalarExpr::ExtractLane0(bias_val), width, dtype, });

                    // Control flow: load old output
                    let out_offset = OffsetExpr::Add(
                        Box::new(OffsetExpr::LoopOffset(seq_off)),
                        Box::new(OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(blk_off)), mxfp4_block_size * elem)),
                            Box::new(OffsetExpr::LoopOffset(row_off)),
                        )),
                    );
                    let old_out = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
                    prog.emit(VmInstr::VecLoad { dst: old_out, base: output_ptr, offset: out_offset.clone(), width: SimdWidth::Scalar, dtype, });
                    let old_out_broadcast = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::Broadcast { dst: old_out_broadcast, src: ScalarExpr::ExtractLane0(old_out), width, dtype, });

                    // Arithmetic: bias + weight + accumulate via auto_lower_trace_raw
                    let slots = super::auto_select::auto_lower_trace_raw(
                        prog, &down_accumulate_body, &[acc_reduced, bias_broadcast, weight_vec, old_out_broadcast], width, QuantPrecision::F32).expect("down accumulate auto_lower failed");
                    let new_out_scalar = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
                    prog.emit(VmInstr::Broadcast { dst: new_out_scalar, src: ScalarExpr::ExtractLane0(slots[6]), width: SimdWidth::Scalar, dtype, });
                    prog.emit(VmInstr::VecStore { base: output_ptr, offset: out_offset, src: new_out_scalar, width: SimdWidth::Scalar, dtype, });
                });
            });
        }
        Ok(())
    })?;

    Ok(())
}

/// 量化 GEMM — SPEC §23-QUANT-CODEGEN-ALGO §3 参数化微核三组件模板.
///
/// 根据 `(desc.data_kind, dot_cap)` 驱动实例化:
/// - **Float (BF16/FP16/F32)**: 无 prologue/epilogue，直接用 DotProduct(Bf16/Fp16)/FMA
/// - **Int8**: scale prologue + GgufInt8Load + DotProduct(Int8) + ScaleApply epilogue
/// - **其他量化格式**: 保留 DecodeTraceBuilder fallback（N2-N6 逐步替换为原生路径）
///
/// 循环结构: M × N × (K/block_size) × (block_size/lanes)
pub(crate) fn emit_quant_gemm_inline(
    prog: &mut VmProgram,
    m_bound: BoundExpr,
    n: usize,
    k: usize,
    quant_type: crate::quant::QuantType,
    width: SimdWidth,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
    dot_cap: DotProductCap,
) -> Result<(), CompilerError> {
    let desc = crate::quant_format::registry()
        .get(&quant_type)
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("emit_quant_gemm_inline: quant_type={:?} not registered", quant_type)
        ))?;

    // Unified parameterized microkernel — SPEC §3.3
    emit_quant_gemm_tiled(prog, m_bound, n, k, desc, width, input_ptr, weight_ptr, output_ptr, dtype, dot_cap)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3.3 统一声明式 GEMM — 完全自动化推导，零手写 VmInstr
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 内层核心策略 — 由 (QuantDataKind, DotProductCap) 自动决定。
#[derive(Debug)]
pub(crate) enum GemmKernel {
    /// 浮点路径: VecLoad A + VecLoad B + DotProduct(Bf16/Fp16)/Fma
    Float,
    /// INT8 硬件原生: Int8Load + DotProduct(Int8) + ScaleApply epilogue
    Int8Native { scale_offset: usize, data_offset: usize },
    /// Assisted 半硬件辅助: 寄存器内 nibble unpack + dequant + FMA (REQ-QCG5)
    /// 用于 INT4 weight-only + SimdAssisted/SimdBasic dot_cap:
    ///   1. QuantBlockLoad(UnsignedNibbleLow/High) or (SignedNibbleLow/High)
    ///   2. QuantInterleave(lo, hi) → sequential order
    ///   3. VecUnaryOp(IntToFloat) if needed
    ///   4. Scale/mul + FMA with activation
    Assisted { scale_offset: usize, data_offset: usize },
    /// DequantFMA 路径 (REQ-QCG6): 纯软件反量化，公式 (qw - zp) × scale。
    /// DecodeTraceBuilder → auto_lower_trace_raw + vfmadd231ps FMA。
    /// 参见 [`DequantFMAPath`]。
    DequantFma,
    /// HighBitMerge 路径 (REQ-QCG-006): Q5_0/Q5_1/Q5_K/Q6_K 等 INT5/INT6 格式。
    /// 使用 QuantBiPlaneLoad(Low5/Low6) 完成 nibble + high-bit 合并 + 偏置减法，
    /// 然后乘 scale + FMA 累加。
    HighBitMerge {
        scale_offset: usize,
        low_offset: usize,
        high_offset: usize,
        bias: f32,
        high_bits: u8,
    },
}

/// GEMM 模式 — 由 m_bound 自动决定。
#[derive(Debug, Clone, Copy)]
pub(crate) enum GemmMode {
    /// GEMV M=1: 3 层循环 (j × blk × ei)，省略 m 循环
    Gemv,
    /// 通用 M>1: 4 层循环 (m × j × blk × ei)
    General,
    /// 浮点无块: 3 层循环 (m × j × k_iters)，无量化块结构
    Float,
}

/// 完全从 (QuantFormatDescriptor, shape, width, dtype) 自动推导的 GEMM 参数。
/// 所有循环边界、步长从这里读取，零手写公式。
pub(crate) struct QuantGemmPlan {
    m_bound: BoundExpr,
    n: usize,
    k: usize,
    width: SimdWidth,
    dtype: QuantPrecision,
    kernel: GemmKernel,
    mode: GemmMode,
    block_size: usize,
    block_bytes: usize,
    bits_per_elem: usize,
    lanes: usize,
    elem: usize,
    gguf_num_blocks: usize,
    quant_row_stride: usize,
    iters_per_block: usize,
    data_bytes_per_block: usize,
    data_step: usize,
    a_row_stride: usize,
    c_row_stride: usize,
}

impl QuantGemmPlan {
    fn derive(
        m_bound: BoundExpr,
        n: usize,
        k: usize,
        desc: &crate::quant_format::QuantFormatDescriptor,
        width: SimdWidth,
        dtype: QuantPrecision,
        dot_cap: DotProductCap,
    ) -> Result<Self, CompilerError> {
        use crate::quant_format::{QuantDataKind as DK, DataLayout, ScaleLayout};

        let lanes = width.f32_lanes().max(1);
        let elem = dtype.elem_bytes();
        let block_size = desc.block_size;
        let block_bytes = desc.block_bytes;
        let bits_per_elem = desc.bits_per_element as usize;

        if n == 0 || k == 0 {
            return Err(CompilerError::CodegenViolation("quant_gemm: n=0 or k=0".into()));
        }

        let (kernel, mode) = if desc.data_kind.is_float() {
            (GemmKernel::Float, GemmMode::Float)
        } else {
            if block_size == 0 {
                return Err(CompilerError::CodegenViolation(
                    format!("quant_gemm: block_size=0 for {}", desc.name)
                ));
            }
            if k % block_size != 0 {
                return Err(CompilerError::CodegenViolation(
                    format!("quant_gemm: k={} not divisible by block_size={} for {}",
                        k, block_size, desc.name)
                ));
            }
            if block_size % lanes != 0 {
                return Err(CompilerError::CodegenViolation(
                    format!("quant_gemm: block_size={} not divisible by lanes={} for {}",
                        block_size, lanes, desc.name)
                ));
            }

            let is_int4 = matches!(desc.data_kind, DK::PackedInt4 | DK::SignedPackedInt4);
            let is_assisted = is_int4 && matches!(dot_cap,
                DotProductCap::SimdAssisted | DotProductCap::SimdBasic);

            let kernel = if matches!(desc.data_kind, DK::Int8) && matches!(dot_cap,
                DotProductCap::NativeInt8Tc | DotProductCap::NativeInt8Simd | DotProductCap::NativeInt8Tile)
            {
                let data_offset = match &desc.data_layout { DataLayout::Bytes { offset, .. } => *offset, _ => 0 };
                let scale_offset = match &desc.scale_layout { ScaleLayout::BlockScalar { offset_bytes, .. } => *offset_bytes, _ => 0 };
                GemmKernel::Int8Native { scale_offset, data_offset }
            } else if is_assisted {
                let data_offset = match &desc.data_layout { DataLayout::Bytes { offset, .. } | DataLayout::PackedNibbles { offset, .. } => *offset, _ => 0 };
                let scale_offset = match &desc.scale_layout { ScaleLayout::BlockScalar { offset_bytes, .. } => *offset_bytes, _ => 0 };
                GemmKernel::Assisted { scale_offset, data_offset }
            } else if matches!(desc.data_kind, DK::PackedInt5 | DK::PackedInt6)
                && matches!(
                    &desc.scale_layout,
                    ScaleLayout::BlockScalar { .. } | ScaleLayout::BlockScalarWithMin { .. }
                )
            {
                // INT5/INT6 HighBitMerge path (REQ-QCG-006):
                // Uses QuantBiPlaneLoad to merge low nibbles + high bit-plane.
                // Only selected for flat scale layouts (BlockScalar / BlockScalarWithMin).
                // Hierarchical (Q5_K) / Q6KScales (Q6_K) formats fall through to DequantFma
                // which handles complex scale + zero layouts via DecodeTraceBuilder.
                let (low_offset, high_offset, high_bits) = match &desc.data_layout {
                    DataLayout::NibbleWithHighBits { low_offset, high_offset, high_bits_per_elem } =>
                        (*low_offset, *high_offset, *high_bits_per_elem),
                    _ => return Err(CompilerError::CodegenViolation(
                        format!("quant_gemm: PackedInt5/6 requires NibbleWithHighBits layout, got {:?}", desc.data_layout)
                    )),
                };
                let scale_offset = match &desc.scale_layout {
                    ScaleLayout::BlockScalar { offset_bytes, .. } => *offset_bytes,
                    ScaleLayout::BlockScalarWithMin { d_offset, .. } => *d_offset,
                    _ => unreachable!("guarded by outer match"),
                };
                let bias = match &desc.zero_layout {
                    crate::quant_format::ZeroLayout::StaticBias { value } => *value as f32,
                    _ => 0.0,
                };
                GemmKernel::HighBitMerge { scale_offset, low_offset, high_offset, bias, high_bits }
            } else {
                GemmKernel::DequantFma
            };

            let mode = if matches!(m_bound, BoundExpr::Const(1) | BoundExpr::DynamicVReg(_)) {
                GemmMode::Gemv
            } else {
                GemmMode::General
            };

            (kernel, mode)
        };

        let gguf_num_blocks = if block_size > 0 { k / block_size } else { 0 };
        let quant_row_stride = gguf_num_blocks * block_bytes;
        let iters_per_block = if block_size > 0 && lanes > 0 { block_size / lanes } else { 0 };
        let data_bytes_per_block = block_size * bits_per_elem / 8;
        // data_step is the byte advance of the LOW-PLANE data pointer per ei iteration.
        // For NibbleWithHighBits (Q6_K), data_ptr only walks the low nibble plane,
        // so step = lanes/2 (4 bytes per 8-element iteration), NOT total_bits/8.
        let data_step = match &desc.data_layout {
            crate::quant_format::DataLayout::PackedNibbles { .. }
            | crate::quant_format::DataLayout::NibbleWithHighBits { .. } => {
                // Low nibbles: 2 elements per byte → lanes/2 bytes per iteration
                lanes / 2
            }
            _ => {
                // Bytes, CodebookIndex, etc: use total bit-span
                if iters_per_block > 0 { data_bytes_per_block / iters_per_block } else { 1 }
            }
        };

        Ok(Self {
            m_bound, n, k, width, dtype, kernel, mode,
            block_size, block_bytes, bits_per_elem,
            lanes, elem,
            gguf_num_blocks, quant_row_stride,
            iters_per_block, data_bytes_per_block, data_step,
            a_row_stride: k * elem,
            c_row_stride: n * elem,
        })
    }
}
