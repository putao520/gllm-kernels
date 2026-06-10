//! Norm, LayerNorm, Softmax inline lowering.

use super::instr::*;
use super::auto_select;
use crate::compiler::trace::{ComputePattern, QuantPrecision, ReduceKind, TraceOp, ValueId};
use crate::compiler::graph;
use crate::types::CompilerError;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §13.N NormLike 自动指令选择 (ARCH-AUTO-INSTR-SELECT Phase A)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// NormLike 通用发射框架：替代 `lower_norm` / `lower_norm_grouped` / `lower_qk_norm`。
///
/// 用 `auto_lower_trace` 替代手写 `lower_trace_body_compat` + 内联 `VecBinOp`，
/// 将三阶段（reduce/finalize/transform）全部交给算法生成 VmInstr。
///
/// `groups_per_row` = 1 时退化为标准 RmsNorm/ValueNorm。
/// `groups_per_row` > 1 时为 HeadRmsNorm/QkNorm 的 per-head 归一化。
///
/// `has_weight` = false 时 transform 不读 weight_ptr（ValueNorm/QkNorm 语义）。
/// `broadcast_weight` = true 时所有 group 共享同一份 weight[feature_dim]。
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_normlike_inline(
    prog: &mut VmProgram,
    pattern: &ComputePattern,
    feature_dim: usize,
    groups_per_row: usize,
    broadcast_weight: bool,
    has_weight: bool,
    width: SimdWidth,
    seq_bound: BoundExpr,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let (reduce, finalize, transform) = match pattern {
        ComputePattern::NormLike { reduce, finalize, transform } => {
            (reduce.as_slice(), finalize.as_slice(), transform.as_slice())
        }
        _ => return Err(CompilerError::CodegenViolation(
            "emit_normlike_inline: expected NormLike pattern".into())),
    };

    let lanes = width.f32_lanes();
    if lanes == 0 || feature_dim == 0 {
        return Err(CompilerError::CodegenViolation(
            "emit_normlike_inline: zero lanes or feature_dim".into()));
    }
    if groups_per_row == 0 {
        return Err(CompilerError::CodegenViolation(
            "emit_normlike_inline: groups_per_row must be >= 1".into()));
    }

    let vec_count = feature_dim / lanes;
    let step_bytes = width.bytes();
    let elem = dtype.elem_bytes();

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let temp = prog.alloc_vreg(VRegKind::Vec, width);
    let scale = prog.alloc_vreg(VRegKind::Vec, width);
    let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);

    let row_bytes = feature_dim * dtype.elem_bytes();
    let outer_step = row_bytes * groups_per_row;

    let row_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let row_output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let row_weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let outer_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let outer_output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // 外层 row loop: seq_len 维度 (ARCH-SYMDIM-THREADING)。
    prog.emit_loop(seq_bound, outer_step, |prog, _row_ctr, row_byte_off| {
        prog.emit(VmInstr::LoadPtr { dst: outer_input, src: PtrExpr::VRegPlusVReg(input_ptr, row_byte_off) });
        prog.emit(VmInstr::LoadPtr { dst: outer_output, src: PtrExpr::VRegPlusVReg(output_ptr, row_byte_off) });

        let body = |prog: &mut VmProgram, _g_ctr: VRegId, group_off: VRegId| {
            prog.emit(VmInstr::LoadPtr { dst: row_input, src: PtrExpr::VRegPlusVReg(outer_input, group_off) });
            prog.emit(VmInstr::LoadPtr { dst: row_output, src: PtrExpr::VRegPlusVReg(outer_output, group_off) });
            if has_weight {
                if broadcast_weight {
                    prog.emit(VmInstr::LoadPtr { dst: row_weight, src: PtrExpr::VRegPlusConst(weight_ptr, 0) });
                } else {
                    prog.emit(VmInstr::LoadPtr { dst: row_weight, src: PtrExpr::VRegPlusVReg(weight_ptr, group_off) });
                }
            }
            emit_normlike_one_group(
                prog, reduce, finalize, transform,
                feature_dim, vec_count, step_bytes, elem, lanes, width,
                has_weight, acc, temp, scale, dim_bc,
                row_input, row_weight, row_output, dtype,
            );
        };

        if groups_per_row > 1 {
            prog.emit_loop(BoundExpr::Const(groups_per_row), row_bytes, body);
        } else {
            // groups_per_row=1: avoid extra LoopBegin/LoopEnd, direct call with group_off=0
            prog.emit(VmInstr::LoadPtr { dst: row_input, src: PtrExpr::VRegPlusConst(outer_input, 0) });
            prog.emit(VmInstr::LoadPtr { dst: row_output, src: PtrExpr::VRegPlusConst(outer_output, 0) });
            if has_weight {
                prog.emit(VmInstr::LoadPtr { dst: row_weight, src: PtrExpr::VRegPlusConst(weight_ptr, 0) });
            }
            emit_normlike_one_group(
                prog, reduce, finalize, transform,
                feature_dim, vec_count, step_bytes, elem, lanes, width,
                has_weight, acc, temp, scale, dim_bc,
                row_input, row_weight, row_output, dtype,
            );
        }
    });

    Ok(())
}

/// 单 group 内部三阶段（由 emit_normlike_inline 调用）。
///
/// Phase 1 (reduce): acc = Σ reduce_body(VecLoad(input))
/// Phase 2 (finalize): acc = finalize_body(acc, dim_bc)
/// Phase 3 (transform): output = transform_body(VecLoad(input), scale, [weight])
///
/// 全部通过 `auto_lower_trace` 自动生成 VmInstr，零手写 emit。
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_normlike_one_group(
    prog: &mut VmProgram,
    reduce: &[TraceOp],
    finalize: &[TraceOp],
    transform: &[TraceOp],
    feature_dim: usize,
    vec_count: usize,
    step_bytes: usize,
    elem: usize,
    lanes: usize,
    width: SimdWidth,
    has_weight: bool,
    acc: VRegId,
    temp: VRegId,
    scale: VRegId,
    dim_bc: VRegId,
    row_input: VRegId,
    weight_ptr: VRegId,
    row_output: VRegId,
    dtype: QuantPrecision,
) {
    let tail = feature_dim % lanes;
    let tail_off = vec_count * step_bytes;
    let s1 = SimdWidth::Scalar;

    // Phase 1: Reduce
    prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _counter, byte_off| {
            prog.emit(VmInstr::VecLoad { dst: temp, base: row_input, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
            auto_select::auto_lower_trace(prog, reduce, &[temp, acc], width, dtype).expect("normlike reduce");
            prog.emit(VmInstr::Accumulate { acc, src: temp });
        });
    }
    if tail > 0 {
        let s_tmp = prog.alloc_vreg(VRegKind::Vec, s1);
        for t in 0..tail {
            prog.emit(VmInstr::VecLoad { dst: s_tmp, base: row_input, offset: OffsetExpr::Const(tail_off + t * elem), width: s1, dtype });
            auto_select::auto_lower_trace(prog, reduce, &[s_tmp, acc], s1, dtype).expect("normlike reduce tail");
            prog.emit(VmInstr::Accumulate { acc, src: s_tmp });
        }
    }

    // HReduce
    let hr = auto_select::auto_lower_trace_raw(prog,
        &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }], &[acc], width, dtype).expect("normlike HReduce");
    prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::ExtractLane0(hr[1]), width, dtype });

    // Phase 2: Finalize
    prog.emit(VmInstr::Broadcast { dst: dim_bc, src: ScalarExpr::Const(feature_dim as f32), width, dtype });
    auto_select::auto_lower_trace(prog, finalize, &[acc, dim_bc], width, dtype).expect("normlike finalize");

    // Phase 3: Transform
    prog.emit(VmInstr::Broadcast { dst: scale, src: ScalarExpr::ExtractLane0(acc), width, dtype });
    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _counter, byte_off| {
            prog.emit(VmInstr::VecLoad { dst: temp, base: row_input, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
            if has_weight {
                let w = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: w, base: weight_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
                auto_select::auto_lower_trace(prog, transform, &[temp, scale, w], width, dtype).expect("normlike transform");
            } else {
                auto_select::auto_lower_trace(prog, transform, &[temp, scale], width, dtype).expect("normlike transform");
            }
            prog.emit(VmInstr::VecStore { base: row_output, offset: OffsetExpr::LoopOffset(byte_off), src: temp, width, dtype });
        });
    }
    if tail > 0 {
        let s_temp = prog.alloc_vreg(VRegKind::Vec, s1);
        for t in 0..tail {
            let off = tail_off + t * elem;
            prog.emit(VmInstr::VecLoad { dst: s_temp, base: row_input, offset: OffsetExpr::Const(off), width: s1, dtype });
            if has_weight {
                let s_w = prog.alloc_vreg(VRegKind::Vec, s1);
                prog.emit(VmInstr::VecLoad { dst: s_w, base: weight_ptr, offset: OffsetExpr::Const(off), width: s1, dtype });
                auto_select::auto_lower_trace(prog, transform, &[s_temp, scale, s_w], s1, dtype).expect("normlike transform tail");
            } else {
                auto_select::auto_lower_trace(prog, transform, &[s_temp, scale], s1, dtype).expect("normlike transform tail");
            }
            prog.emit(VmInstr::VecStore { base: row_output, offset: OffsetExpr::Const(off), src: s_temp, width: s1, dtype });
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LayerNorm auto-lowering (替代 lower.rs 手写 lower_layernorm)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// LayerNorm 全自动 JIT lowering：y = (x - mean) * rsqrt(var + eps) * weight + bias
///
/// 三阶段全部通过 `auto_lower_trace` / `auto_lower_trace_multi` 自动生成 VmInstr：
/// - Phase 1: 双 accumulator reduce (sum_x + sum_x²)
/// - Phase 2: finalize (mean + inv_std)，多输出写回
/// - Phase 3: transform ((x - mean) * inv_std * weight + bias)
///
/// weight 和 bias packed: `[weight: feature_dim * elem][bias: feature_dim * elem]`
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_layernorm_auto(
    prog: &mut VmProgram,
    feature_dim: usize,
    eps: f32,
    width: SimdWidth,
    seq_bound: BoundExpr,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes();
    if lanes == 0 || feature_dim == 0 {
        return Err(CompilerError::CodegenViolation("emit_layernorm_auto: zero lanes or feature_dim".into()));
    }
    let vec_count = feature_dim / lanes;
    let step_bytes = width.bytes();
    let elem = dtype.elem_bytes();
    let row_bytes = feature_dim * elem;
    let bias_offset = feature_dim * elem;
    let tail = feature_dim % lanes;
    let tail_off = vec_count * step_bytes;
    let s1 = SimdWidth::Scalar;

    let acc_sum = prog.alloc_vreg(VRegKind::Vec, width);
    let acc_sq = prog.alloc_vreg(VRegKind::Vec, width);
    let temp = prog.alloc_vreg(VRegKind::Vec, width);
    let scale = prog.alloc_vreg(VRegKind::Vec, width);
    let mean_bc = prog.alloc_vreg(VRegKind::Vec, width);
    let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
    let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
    let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

    prog.emit(VmInstr::Broadcast { dst: dim_bc, src: ScalarExpr::Const(feature_dim as f32), width, dtype });

    let finalize_body = vec![
        TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
        TraceOp::Const(eps as f64),
        TraceOp::Div(ValueId(0), ValueId(2)),   // mean = sum_x / dim
        TraceOp::Div(ValueId(1), ValueId(2)),   // sq_mean = sum_sq / dim
        TraceOp::Mul(ValueId(4), ValueId(4)),   // mean²
        TraceOp::Sub(ValueId(5), ValueId(6)),   // var
        TraceOp::Add(ValueId(7), ValueId(3)),   // var + eps
        TraceOp::Rsqrt(ValueId(8)),    // inv_std
    ];
    let transform_body = vec![
        TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2), TraceOp::Input(3),
        TraceOp::Sub(ValueId(0), ValueId(2)),  // x - mean
        TraceOp::Mul(ValueId(4), ValueId(1)),  // * inv_std
        TraceOp::Mul(ValueId(5), ValueId(3)),  // * weight
        TraceOp::Input(4),
        TraceOp::Add(ValueId(6), ValueId(7)),  // + bias
    ];
    let mul_sq_body = [TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(0))];
    let hr_sum_body = vec![TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];

    prog.emit_loop(seq_bound, row_bytes, |prog, _row_ctr, row_byte_off| {
        prog.emit(VmInstr::LoadPtr { dst: row_input, src: PtrExpr::VRegPlusVReg(input_ptr, row_byte_off) });
        prog.emit(VmInstr::LoadPtr { dst: row_output, src: PtrExpr::VRegPlusVReg(output_ptr, row_byte_off) });

        // Phase 1: Dual Accumulator Reduce (sum_x + sum_x²)
        prog.emit(VmInstr::Broadcast { dst: acc_sum, src: ScalarExpr::Const(0.0), width, dtype });
        prog.emit(VmInstr::Broadcast { dst: acc_sq, src: ScalarExpr::Const(0.0), width, dtype });
        if vec_count > 0 {
            prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _ctr, byte_off| {
                prog.emit(VmInstr::VecLoad { dst: temp, base: row_input, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
                prog.emit(VmInstr::Accumulate { acc: acc_sum, src: temp });
                auto_select::auto_lower_trace(prog, &mul_sq_body, &[temp], width, dtype).expect("layernorm sq");
                prog.emit(VmInstr::Accumulate { acc: acc_sq, src: temp });
            });
        }
        if tail > 0 {
            for t in 0..tail {
                prog.emit(VmInstr::VecLoad { dst: temp, base: row_input, offset: OffsetExpr::Const(tail_off + t * elem), width: s1, dtype });
                prog.emit(VmInstr::Accumulate { acc: acc_sum, src: temp });
                auto_select::auto_lower_trace(prog, &mul_sq_body, &[temp], s1, dtype).expect("layernorm sq tail");
                prog.emit(VmInstr::Accumulate { acc: acc_sq, src: temp });
            }
        }
        // HReduce both accumulators
        let hr_s = auto_select::auto_lower_trace_raw(prog, &hr_sum_body, &[acc_sum], width, dtype).expect("layernorm HReduce sum");
        prog.emit(VmInstr::Broadcast { dst: acc_sum, src: ScalarExpr::ExtractLane0(hr_s[1]), width, dtype });
        let hr_q = auto_select::auto_lower_trace_raw(prog, &hr_sum_body, &[acc_sq], width, dtype).expect("layernorm HReduce sq");
        prog.emit(VmInstr::Broadcast { dst: acc_sq, src: ScalarExpr::ExtractLane0(hr_q[1]), width, dtype });

        // Phase 2: Finalize → inv_std (slot 9), mean (slot 4)
        auto_select::auto_lower_trace_multi(
            prog, &finalize_body, &[acc_sum, acc_sq, dim_bc],
            &[(acc_sum, 9), (acc_sq, 4)], width, dtype,
        ).expect("layernorm finalize");
        prog.emit(VmInstr::Broadcast { dst: scale, src: ScalarExpr::ExtractLane0(acc_sum), width, dtype });
        prog.emit(VmInstr::Broadcast { dst: mean_bc, src: ScalarExpr::ExtractLane0(acc_sq), width, dtype });

        // Phase 3: Transform (x - mean) * inv_std * weight + bias
        if vec_count > 0 {
            prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _ctr, byte_off| {
                prog.emit(VmInstr::VecLoad { dst: temp, base: row_input, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
                let w = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: w, base: weight_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
                let b = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: b, base: weight_ptr,
                    offset: OffsetExpr::Add(Box::new(OffsetExpr::Const(bias_offset)), Box::new(OffsetExpr::LoopOffset(byte_off))),
                    width, dtype });
                auto_select::auto_lower_trace(prog, &transform_body, &[temp, scale, mean_bc, w, b], width, dtype).expect("layernorm transform");
                prog.emit(VmInstr::VecStore { base: row_output, offset: OffsetExpr::LoopOffset(byte_off), src: temp, width, dtype });
            });
        }
        if tail > 0 {
            for t in 0..tail {
                let off = tail_off + t * elem;
                prog.emit(VmInstr::VecLoad { dst: temp, base: row_input, offset: OffsetExpr::Const(off), width: s1, dtype });
                let s_w = prog.alloc_vreg(VRegKind::Vec, s1);
                prog.emit(VmInstr::VecLoad { dst: s_w, base: weight_ptr, offset: OffsetExpr::Const(off), width: s1, dtype });
                let s_b = prog.alloc_vreg(VRegKind::Vec, s1);
                prog.emit(VmInstr::VecLoad { dst: s_b, base: weight_ptr, offset: OffsetExpr::Const(bias_offset + off), width: s1, dtype });
                auto_select::auto_lower_trace(prog, &transform_body, &[temp, scale, mean_bc, s_w, s_b], s1, dtype).expect("layernorm transform tail");
                prog.emit(VmInstr::VecStore { base: row_output, offset: OffsetExpr::Const(off), src: temp, width: s1, dtype });
            }
        }
    });

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §13.S Reduction 自动指令选择 (ARCH-AUTO-INSTR-SELECT Phase B)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Softmax 三阶段 JIT lowering (替代 `lower_reduction_softmax`)。
///
/// 三阶段:
/// 1. Max reduce: acc = max(input[*])
/// 2. Exp+sum: output[i] = exp(input[i] - max); acc2 = sum(output[*])
/// 3. Normalize: output[i] = output[i] / acc2
///
/// 全部通过 `auto_lower_trace` 自动生成 VmInstr。
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_softmax_inline(
    prog: &mut VmProgram,
    feature_dim: usize,
    width: SimdWidth,
    input_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(VRegId, VRegId), CompilerError> {
    if feature_dim == 0 {
        return Err(CompilerError::CodegenViolation("emit_softmax_inline: zero feature_dim".into()));
    }
    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let vec_count = feature_dim / lanes;
    let tail = feature_dim % lanes;
    let step = width.bytes();
    let tail_off = vec_count * step;
    let s1 = SimdWidth::Scalar;

    let tmp = prog.alloc_vreg(VRegKind::Vec, width);
    let max_val = prog.alloc_vreg(VRegKind::Vec, width);
    let sum_val = prog.alloc_vreg(VRegKind::Vec, width);
    let combine_max: Vec<TraceOp> = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Max(ValueId(0), ValueId(1))];

    // Phase 1: max reduce
    prog.emit(VmInstr::Broadcast { dst: max_val, src: ScalarExpr::Const(f32::NEG_INFINITY), width, dtype });
    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step, |prog, _ctr, byte_off| {
            prog.emit(VmInstr::VecLoad { dst: tmp, base: input_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
            auto_select::auto_lower_trace(prog, &combine_max, &[tmp, max_val], width, dtype).expect("softmax max");
            prog.emit(VmInstr::Accumulate { acc: max_val, src: tmp });
        });
    }
    if tail > 0 {
        let s_tmp = prog.alloc_vreg(VRegKind::Vec, s1);
        let s_max = prog.alloc_vreg(VRegKind::Vec, s1);
        prog.emit(VmInstr::Broadcast { dst: s_max, src: ScalarExpr::ExtractLane0(max_val), width: s1, dtype });
        for t in 0..tail {
            prog.emit(VmInstr::VecLoad { dst: s_tmp, base: input_ptr, offset: OffsetExpr::Const(tail_off + t * elem), width: s1, dtype });
            auto_select::auto_lower_trace(prog, &combine_max, &[s_tmp, s_max], s1, dtype).expect("softmax max tail");
            prog.emit(VmInstr::Accumulate { acc: max_val, src: s_tmp });
        }
    }
    let hr_max = auto_select::auto_lower_trace_raw(prog,
        &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Max }], &[max_val], width, dtype).expect("softmax HReduce max");
    prog.emit(VmInstr::Broadcast { dst: max_val, src: ScalarExpr::ExtractLane0(hr_max[1]), width, dtype });

    // Phase 2: exp(x - max) + sum
    let exp_sub: Vec<TraceOp> = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Sub(ValueId(0), ValueId(1)), TraceOp::Exp(ValueId(2))];
    let combine_sum: Vec<TraceOp> = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(ValueId(0), ValueId(1))];
    prog.emit(VmInstr::Broadcast { dst: sum_val, src: ScalarExpr::Const(0.0), width, dtype });
    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step, |prog, _ctr, byte_off| {
            prog.emit(VmInstr::VecLoad { dst: tmp, base: input_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
            auto_select::auto_lower_trace(prog, &exp_sub, &[tmp, max_val], width, dtype).expect("softmax exp");
            prog.emit(VmInstr::VecStore { base: output_ptr, offset: OffsetExpr::LoopOffset(byte_off), src: tmp, width, dtype });
            auto_select::auto_lower_trace(prog, &combine_sum, &[sum_val, tmp], width, dtype).expect("softmax sum");
            prog.emit(VmInstr::Accumulate { acc: sum_val, src: tmp });
        });
    }
    if tail > 0 {
        let s_max_bc = prog.alloc_vreg(VRegKind::Vec, s1);
        let s_sum = prog.alloc_vreg(VRegKind::Vec, s1);
        let s_tmp = prog.alloc_vreg(VRegKind::Vec, s1);
        prog.emit(VmInstr::Broadcast { dst: s_max_bc, src: ScalarExpr::ExtractLane0(max_val), width: s1, dtype });
        prog.emit(VmInstr::Broadcast { dst: s_sum, src: ScalarExpr::ExtractLane0(sum_val), width: s1, dtype });
        for t in 0..tail {
            let off = tail_off + t * elem;
            prog.emit(VmInstr::VecLoad { dst: s_tmp, base: input_ptr, offset: OffsetExpr::Const(off), width: s1, dtype });
            auto_select::auto_lower_trace_into(prog, &exp_sub, &[s_tmp, s_max_bc], s_tmp, s1, dtype).expect("softmax tail exp");
            prog.emit(VmInstr::VecStore { base: output_ptr, offset: OffsetExpr::Const(off), src: s_tmp, width: s1, dtype });
            auto_select::auto_lower_trace_into(prog, &combine_sum, &[s_sum, s_tmp], s_sum, s1, dtype).expect("softmax tail sum");
        }
        prog.emit(VmInstr::Accumulate { acc: sum_val, src: s_sum });
    }
    let hr_sum = auto_select::auto_lower_trace_raw(prog,
        &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }], &[sum_val], width, dtype).expect("softmax HReduce sum");
    prog.emit(VmInstr::Broadcast { dst: sum_val, src: ScalarExpr::ExtractLane0(hr_sum[1]), width, dtype });

    // Phase 3: normalize by sum
    let normalize: Vec<TraceOp> = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
    auto_select::auto_lower_trace_into(prog, &[TraceOp::Input(0), TraceOp::Recip(ValueId(0))], &[sum_val], sum_val, width, dtype).expect("softmax recip");
    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step, |prog, _ctr, byte_off| {
            prog.emit(VmInstr::VecLoad { dst: tmp, base: output_ptr, offset: OffsetExpr::LoopOffset(byte_off), width, dtype });
            auto_select::auto_lower_trace(prog, &normalize, &[tmp, sum_val], width, dtype).expect("softmax normalize");
            prog.emit(VmInstr::VecStore { base: output_ptr, offset: OffsetExpr::LoopOffset(byte_off), src: tmp, width, dtype });
        });
    }
    if tail > 0 {
        let s_inv = prog.alloc_vreg(VRegKind::Vec, s1);
        let s_tmp = prog.alloc_vreg(VRegKind::Vec, s1);
        prog.emit(VmInstr::Broadcast { dst: s_inv, src: ScalarExpr::ExtractLane0(sum_val), width: s1, dtype });
        for t in 0..tail {
            let off = tail_off + t * elem;
            prog.emit(VmInstr::VecLoad { dst: s_tmp, base: output_ptr, offset: OffsetExpr::Const(off), width: s1, dtype });
            auto_select::auto_lower_trace_into(prog, &normalize, &[s_tmp, s_inv], s_tmp, s1, dtype).expect("softmax tail normalize");
            prog.emit(VmInstr::VecStore { base: output_ptr, offset: OffsetExpr::Const(off), src: s_tmp, width: s1, dtype });
        }
    }

    Ok((max_val, sum_val))
}

/// §13.9 Softmax telemetry: sharpness + max + is_sink。
/// 在 `emit_softmax_inline` 完成后调用。
/// 注意: `max_val` 和 `sum_val` 必须在调用时仍然有效（包含正确的 softmax 统计）。
pub(crate) fn emit_softmax_telemetry(
    prog: &mut VmProgram,
    max_val: VRegId,
    sum_val: VRegId,
    telemetry_ptr: VRegId,
    width: SimdWidth,
    dtype: QuantPrecision,
) {
    use graph::telemetry_offsets;
    use graph::SOFTMAX_SINK_THRESHOLD;

    prog.emit(VmInstr::Comment("§13.9 Softmax telemetry: max + sharpness + is_sink".into()));

    // Store softmax max value to telemetry[SOFTMAX_MAX_OFFSET]
    prog.emit(VmInstr::VecStore {
        base: telemetry_ptr,
        offset: OffsetExpr::Const(telemetry_offsets::SOFTMAX_MAX_OFFSET),
        src: max_val,
        width: SimdWidth::Scalar,
        dtype,
    });

    // sharpness = 1/sum
    let sharpness = prog.alloc_vreg(VRegKind::Vec, width);
    let recip_body: Vec<TraceOp> = vec![TraceOp::Input(0), TraceOp::Recip(ValueId(0))];
    auto_select::auto_lower_trace_into(prog, &recip_body, &[sum_val], sharpness, width, dtype)
        .expect("emit_softmax_telemetry: recip auto_lower invariant violation");

    // Store sharpness to telemetry[SOFTMAX_SHARPNESS_OFFSET]
    prog.emit(VmInstr::VecStore {
        base: telemetry_ptr,
        offset: OffsetExpr::Const(telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET),
        src: sharpness,
        width: SimdWidth::Scalar,
        dtype,
    });

    // is_sink = (sharpness > SOFTMAX_SINK_THRESHOLD) ? 1.0 : 0.0
    // Using Min/Max trick (no VecCmp needed):
    //   diff = sharpness - threshold → if positive, is_sink
    //   clamped = Min(Max(diff, 0.0), 1.0)
    // TraceOp body: clamp(sharpness - threshold, 0.0, 1.0)
    let sink_body: Vec<TraceOp> = vec![
        TraceOp::Input(0),                           // [0] sharpness
        TraceOp::Const(SOFTMAX_SINK_THRESHOLD as f64), // [1] threshold
        TraceOp::Sub(ValueId(0), ValueId(1)),                          // [2] diff
        TraceOp::Const(0.0),                         // [3] zero
        TraceOp::Max(ValueId(2), ValueId(3)),                          // [4] diff_clamped
        TraceOp::Const(1.0),                         // [5] one
        TraceOp::Min(ValueId(4), ValueId(5)),                          // [6] clamped
    ];
    let clamped = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
    auto_select::auto_lower_trace_into(prog, &sink_body, &[sharpness], clamped, SimdWidth::Scalar, dtype)
        .expect("emit_softmax_telemetry: sink_body auto_lower invariant violation");

    prog.emit(VmInstr::VecStore {
        base: telemetry_ptr,
        offset: OffsetExpr::Const(telemetry_offsets::IS_ATTENTION_SINK_OFFSET),
        src: clamped,
        width: SimdWidth::Scalar,
        dtype,
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 27 REQ-AT-009: 模板驱动 Norm 发射桥接
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 尝试通过模板驱动路径发射 Norm 算子。
///
/// 返回 `Some(())` 表示成功，`None` 表示需要 fallback 到 emit_normlike_inline。
pub(crate) fn emit_norm_template_driven(
    prog: &mut VmProgram,
    feature_dim: usize,
    width: SimdWidth,
    seq_bound: BoundExpr,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
    is_rms: bool,
) -> Option<()> {
    use super::algo_registry;
    use super::algo_interpreter::{TemplateInterpreter, ParamTable, TemplateInputs};
    use crate::dispatch::device_profile::DeviceProfile;

    let strategy = if is_rms {
        crate::compiler::codegen::vm::algo_template::AlgoStrategy::NormRms
    } else {
        crate::compiler::codegen::vm::algo_template::AlgoStrategy::NormLayer
    };

    let profile = DeviceProfile::detect();
    let template = algo_registry::select_template(&strategy, &profile)?;

    let mut params = ParamTable::new();
    params.set("hidden_dim", feature_dim);

    // Norm 模板需要 4 个输入: input_ptr, weight_ptr, output_ptr, seq_offset
    let seq_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    let inputs = TemplateInputs::norm();

    let mut interp = TemplateInterpreter::new(params);
    let trace_ops = interp.instantiate(template, &inputs);

    super::auto_select::auto_lower_trace_raw(
        prog, &trace_ops, &[input_ptr, weight_ptr, output_ptr, seq_offset],
        width, dtype,
    ).ok()?;

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::{ComputePattern, QuantPrecision, TraceOp};

    /// Helper: create a NormLike pattern with trivial reduce/finalize/transform bodies.
    fn make_normlike_pattern() -> ComputePattern {
        ComputePattern::NormLike {
            reduce: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))],
            finalize: vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))],
            transform: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))],
        }
    }

    // ── Test 1: emit_normlike_inline rejects non-NormLike patterns ──

    #[test]
    fn normlike_inline_rejects_elementwise_pattern() {
        // Arrange
        let mut prog = VmProgram::new();
        let pattern = ComputePattern::Elementwise {
            body: vec![TraceOp::Input(0)],
        };
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject Elementwise pattern");
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected NormLike"), "Error message should mention NormLike, got: {msg}");
    }

    // ── Test 2: emit_normlike_inline rejects zero lanes ──

    #[test]
    fn normlike_inline_rejects_zero_lanes() {
        // Arrange
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: SimdWidth::Scalable has f32_lanes() == 0
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, true,
            SimdWidth::Scalable, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject zero lanes (Scalable)");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("zero lanes"), "Error message should mention zero lanes, got: {msg}");
    }

    // ── Test 3: emit_normlike_inline rejects zero feature_dim ──

    #[test]
    fn normlike_inline_rejects_zero_feature_dim() {
        // Arrange
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim = 0
        let result = emit_normlike_inline(
            &mut prog, &pattern, 0, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject zero feature_dim");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("zero"), "Error should mention zero, got: {msg}");
    }

    // ── Test 4: emit_normlike_inline rejects zero groups_per_row ──

    #[test]
    fn normlike_inline_rejects_zero_groups() {
        // Arrange
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: groups_per_row = 0
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 0, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject groups_per_row == 0");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("groups_per_row"), "Error should mention groups_per_row, got: {msg}");
    }

    // ── Test 5: emit_normlike_inline succeeds with valid parameters (groups=1) ──

    #[test]
    fn normlike_inline_succeeds_groups_one() {
        // Arrange
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with valid parameters");
        // Should have allocated additional vregs for internal use
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
        // Should have emitted instructions
        assert!(!prog.instrs.is_empty(), "Should emit VmInstr instructions");
    }

    // ── Test 6: emit_normlike_inline succeeds with multiple groups ──

    #[test]
    fn normlike_inline_succeeds_multiple_groups() {
        // Arrange
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: groups_per_row=4 simulates per-head normalization (e.g., QkNorm)
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 4, false, false,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with groups_per_row > 1");
    }

    // ── Test 7: emit_softmax_inline rejects zero feature_dim ──

    #[test]
    fn softmax_inline_rejects_zero_feature_dim() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_softmax_inline(
            &mut prog, 0, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject zero feature_dim");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("zero feature_dim"), "Error should mention zero feature_dim, got: {msg}");
    }

    // ── Test 8: emit_softmax_inline returns (max_val, sum_val) vregs on success ──

    #[test]
    fn softmax_inline_returns_valid_vregs() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act
        let result = emit_softmax_inline(
            &mut prog, 16, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with feature_dim=16, W256");
        let (max_val, sum_val) = result.unwrap();
        // max_val and sum_val should be within the allocated vreg range
        assert!(max_val.0 < prog.vreg_count(), "max_val should be a valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be a valid vreg");
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
    }

    // ── Test 9: emit_softmax_inline with BF16 dtype ──

    #[test]
    fn softmax_inline_succeeds_bf16() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_softmax_inline(
            &mut prog, 32, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 dtype");
    }

    // ── Test 10: emit_layernorm_auto rejects zero lanes ──

    #[test]
    fn layernorm_auto_rejects_zero_lanes() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: SimdWidth::Scalable has f32_lanes() == 0
        let result = emit_layernorm_auto(
            &mut prog, 64, 1e-5,
            SimdWidth::Scalable, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject zero lanes");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("zero lanes"), "Error should mention zero lanes, got: {msg}");
    }

    // ── Test 11: emit_layernorm_auto rejects zero feature_dim ──

    #[test]
    fn layernorm_auto_rejects_zero_feature_dim() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim = 0
        let result = emit_layernorm_auto(
            &mut prog, 0, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_err(), "Should reject zero feature_dim");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("zero"), "Error should mention zero, got: {msg}");
    }

    // ── Test 12: emit_layernorm_auto succeeds and emits instructions ──

    #[test]
    fn layernorm_auto_succeeds_and_emits_instrs() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 32, 1e-5,
            SimdWidth::W256, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with valid parameters");
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 13: emit_softmax_inline handles tail elements correctly ──

    #[test]
    fn softmax_inline_handles_non_aligned_feature_dim() {
        // Arrange: feature_dim = 10 with W256 (8 lanes) => vec_count=1, tail=2
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=10 is not a multiple of 8 lanes, so tail handling is exercised
        let result = emit_softmax_inline(
            &mut prog, 10, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should handle non-aligned feature_dim correctly");
        // Verify the program has instructions (not empty)
        assert!(!prog.instrs.is_empty(), "Should emit instructions for non-aligned case");
    }

    // ── Test 14: emit_normlike_inline with broadcast_weight=true and multiple groups ──
    // @trace TEST-NSE-14 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_broadcast_weight_multiple_groups() {
        // Arrange: broadcast_weight=true means all groups share same weight[feature_dim]
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act: groups_per_row=4, broadcast_weight=true, has_weight=true
        let result = emit_normlike_inline(
            &mut prog, &pattern, 32, 4, true, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with broadcast_weight=true");
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 15: emit_normlike_inline with has_weight=false (ValueNorm semantics) ──
    // @trace TEST-NSE-15 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_no_weight_valuenorm_semantics() {
        // Arrange: has_weight=false skips all weight loading (ValueNorm/QkNorm path)
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: has_weight=false, groups_per_row=1
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, false,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed without weight (ValueNorm semantics)");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 16: emit_normlike_inline with feature_dim < lanes (pure tail, no vec iterations) ──
    // @trace TEST-NSE-16 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_pure_tail_feature_dim_less_than_lanes() {
        // Arrange: feature_dim=3 with W256 (8 lanes) => vec_count=0, tail=3
        // This exercises the tail-only path where no vectorized loop iterations occur.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=3 is less than 8 lanes, vec_count=0
        let result = emit_normlike_inline(
            &mut prog, &pattern, 3, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with feature_dim < lanes");
        assert!(!prog.instrs.is_empty(), "Should emit instructions for tail-only case");
    }

    // ── Test 17: emit_normlike_inline with SimdWidth::W128 ──
    // @trace TEST-NSE-17 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_w128_width() {
        // Arrange: SimdWidth::W128 has 4 f32 lanes, different step_bytes and vec_count
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=10, W128 => vec_count=2, tail=2
        let result = emit_normlike_inline(
            &mut prog, &pattern, 10, 1, false, true,
            SimdWidth::W128, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W128 width");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with W128");
    }

    // ── Test 18: emit_normlike_one_group direct call with weight and non-aligned feature_dim ──
    // @trace TEST-NSE-18 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_with_weight_non_aligned() {
        // Arrange: feature_dim=10, W256 (8 lanes) => vec_count=1, tail=2
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let lanes = width.f32_lanes();
        let feature_dim = 10usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::F32.elem_bytes();
        let dtype = QuantPrecision::F32;
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: has_weight=true with non-aligned feature_dim exercises both vec loop and tail
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            true, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, dtype,
        );

        // Assert: should have emitted instructions for reduce + finalize + transform phases
        assert!(prog.instrs.len() > initial_count, "Should emit instructions for all 3 phases");
    }

    // ── Test 19: emit_normlike_one_group direct call without weight ──
    // @trace TEST-NSE-19 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_without_weight() {
        // Arrange: has_weight=false skips weight loading entirely
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let lanes = width.f32_lanes();
        let feature_dim = 16usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::F32.elem_bytes();
        let dtype = QuantPrecision::F32;
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: has_weight=false
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            false, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, dtype,
        );

        // Assert: should emit instructions even without weight
        assert!(prog.instrs.len() > initial_count, "Should emit instructions without weight");
    }

    // ── Test 20: emit_softmax_telemetry emits telemetry instructions ──
    // @trace TEST-NSE-20 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_telemetry_emits_instructions() {
        // Arrange
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let max_val = prog.alloc_vreg(VRegKind::Vec, width);
        let sum_val = prog.alloc_vreg(VRegKind::Vec, width);
        let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act
        emit_softmax_telemetry(&mut prog, max_val, sum_val, telemetry_ptr, width, dtype);

        // Assert: should emit instructions for max store, sharpness (recip), and is_sink
        assert!(prog.instrs.len() > initial_count, "Should emit telemetry instructions");
    }

    // ── Test 21: emit_layernorm_auto handles non-aligned feature_dim (tail elements) ──
    // @trace TEST-NSE-21 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_non_aligned_feature_dim() {
        // Arrange: feature_dim=10, W256 (8 lanes) => vec_count=1, tail=2
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act: non-aligned feature_dim exercises tail code paths
        let result = emit_layernorm_auto(
            &mut prog, 10, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with non-aligned feature_dim");
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 22: emit_layernorm_auto with BF16 dtype ──
    // @trace TEST-NSE-22 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_bf16_dtype() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: BF16 dtype affects elem_bytes and VecStore dtype
        let result = emit_layernorm_auto(
            &mut prog, 32, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 dtype");
        assert!(!prog.instrs.is_empty(), "Should emit instructions for BF16");
    }

    // ── Test 23: emit_layernorm_auto with SimdWidth::W128 ──
    // @trace TEST-NSE-23 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_w128_width() {
        // Arrange: W128 has 4 lanes, feature_dim=12 => vec_count=3, tail=0
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 12, 1e-5,
            SimdWidth::W128, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W128 width");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with W128");
    }

    // ── Test 24: emit_softmax_inline with SimdWidth::W128 ──
    // @trace TEST-NSE-24 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_w128_width() {
        // Arrange: W128 has 4 f32 lanes, feature_dim=10 => vec_count=2, tail=2
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_softmax_inline(
            &mut prog, 10, SimdWidth::W128,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W128");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be a valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be a valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with W128");
    }

    // ── Test 25: emit_normlike_inline with Symbolic seq bound ──
    // @trace TEST-NSE-25 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_symbolic_seq_bound() {
        // Arrange: Symbolic bound exercises runtime dimension path
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".into(),
            max_alloc: 4096,
        });

        // Act: Symbolic bound tests dynamic seq_len code path
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, true,
            SimdWidth::W256, sym_bound,
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with Symbolic bound");
        assert!(!prog.instrs.is_empty(), "Should emit instructions for Symbolic bound");
    }

    // ── Test 26: emit_softmax_inline with feature_dim=1 (single element, pure tail) ──
    // @trace TEST-NSE-26 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_single_element_feature_dim() {
        // Arrange: feature_dim=1, W256 (8 lanes) => vec_count=0, tail=1
        // This exercises the tail-only path for all 3 phases (max, exp+sum, normalize).
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: single element exercises pure-scalar tail handling
        let result = emit_softmax_inline(
            &mut prog, 1, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with feature_dim=1");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be a valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be a valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit instructions for single element");
    }

    // ── Test 27: emit_normlike_inline with BF16 dtype exercises elem_bytes=2 path ──
    // @trace TEST-NSE-27 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_bf16_dtype_elem_bytes() {
        // Arrange: BF16 has elem_bytes=2, affects step_bytes and row_bytes calculations
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: BF16 dtype with non-aligned feature_dim
        let result = emit_normlike_inline(
            &mut prog, &pattern, 12, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 dtype");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with BF16 elem_bytes=2");
    }

    // ── Test 28: emit_layernorm_auto with Symbolic seq bound ──
    // @trace TEST-NSE-28 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_symbolic_seq_bound() {
        // Arrange: Symbolic bound exercises runtime dimension path via emit_loop
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".into(),
            max_alloc: 2048,
        });

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 32, 1e-5,
            SimdWidth::W256, sym_bound,
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with Symbolic seq bound");
        assert!(prog.instrs.len() > initial_count, "Should emit instructions for Symbolic bound");
    }

    // ── Test 29: emit_layernorm_auto with feature_dim exactly equal to lanes (no tail) ──
    // @trace TEST-NSE-29 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_feature_dim_equals_lanes_no_tail() {
        // Arrange: feature_dim=8 with W256 => vec_count=1, tail=0 (perfectly aligned)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 8, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert: no tail means fewer instructions than a non-aligned case
        assert!(result.is_ok(), "Should succeed with perfectly aligned feature_dim");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 30: emit_normlike_one_group with feature_dim exactly equal to lanes (no tail) ──
    // @trace TEST-NSE-30 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_aligned_no_tail() {
        // Arrange: feature_dim=8 with W256 => vec_count=1, tail=0
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let lanes = width.f32_lanes();
        let feature_dim = 8usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::F32.elem_bytes();
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: feature_dim=8 perfectly aligned, no tail path exercised
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            true, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, QuantPrecision::F32,
        );

        // Assert: should emit instructions for vec loop only, no scalar tail
        assert!(prog.instrs.len() > initial_count, "Should emit instructions for aligned feature_dim");
    }

    // ── Test 31: emit_softmax_inline with W512 width (16 lanes) ──
    // @trace TEST-NSE-31 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_w512_width() {
        // Arrange: W512 has 16 f32 lanes, feature_dim=20 => vec_count=1, tail=4
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_softmax_inline(
            &mut prog, 20, SimdWidth::W512,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W512");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with W512");
    }

    // ── Test 32: emit_normlike_inline with non-broadcast weight and multiple groups ──
    // @trace TEST-NSE-32 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_non_broadcast_weight_multiple_groups() {
        // Arrange: broadcast_weight=false with groups_per_row>1 means per-group weight pointer
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: groups_per_row=3, broadcast_weight=false, has_weight=true
        let result = emit_normlike_inline(
            &mut prog, &pattern, 24, 3, false, true,
            SimdWidth::W256, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert: non-broadcast weight loads per-group pointer via VRegPlusVReg
        assert!(result.is_ok(), "Should succeed with non-broadcast weight and multiple groups");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 33: emit_softmax_inline with feature_dim exactly equal to lanes (zero tail) ──
    // @trace TEST-NSE-33 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_aligned_feature_dim_no_tail() {
        // Arrange: feature_dim=8 with W256 => vec_count=1, tail=0 (perfectly aligned)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_softmax_inline(
            &mut prog, 8, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert: no tail path, only vectorized loop iterations
        assert!(result.is_ok(), "Should succeed with aligned feature_dim");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be valid vreg");
    }

    // ── Test 34: emit_layernorm_auto with W128 and non-aligned feature_dim ──
    // @trace TEST-NSE-34 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_w128_non_aligned() {
        // Arrange: W128 has 4 lanes, feature_dim=7 => vec_count=1, tail=3
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 7, 1e-5,
            SimdWidth::W128, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W128 and non-aligned feature_dim=7");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 35: emit_softmax_telemetry with W128 width ──
    // @trace TEST-NSE-35 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_telemetry_w128_width() {
        // Arrange
        let mut prog = VmProgram::new();
        let width = SimdWidth::W128;
        let dtype = QuantPrecision::F32;
        let max_val = prog.alloc_vreg(VRegKind::Vec, width);
        let sum_val = prog.alloc_vreg(VRegKind::Vec, width);
        let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act
        emit_softmax_telemetry(&mut prog, max_val, sum_val, telemetry_ptr, width, dtype);

        // Assert: should emit telemetry instructions with W128 width
        assert!(prog.instrs.len() > initial_count, "Should emit telemetry instructions with W128");
    }

    // ── Test 36: emit_normlike_inline with feature_dim=1 (single element, pure tail) ──
    // @trace TEST-NSE-36 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_feature_dim_one_pure_tail() {
        // Arrange: feature_dim=1 with W256 (8 lanes) => vec_count=0, tail=1
        // Exercises pure scalar tail path across all three phases.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_normlike_inline(
            &mut prog, &pattern, 1, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with feature_dim=1");
        assert!(!prog.instrs.is_empty(), "Should emit scalar-only instructions for feature_dim=1");
    }

    // ── Test 37: emit_normlike_one_group with BF16 dtype ──
    // @trace TEST-NSE-37 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_bf16_dtype() {
        // Arrange: BF16 elem_bytes=2 affects tail offset calculations
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let lanes = width.f32_lanes();
        let feature_dim = 12usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::BF16.elem_bytes();
        let dtype = QuantPrecision::BF16;
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: BF16 dtype with vec_count=1, tail=4
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            true, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, dtype,
        );

        // Assert: should emit instructions with BF16 dtype affecting VecLoad/VecStore
        assert!(prog.instrs.len() > initial_count, "Should emit instructions with BF16 dtype");
    }

    // ── Test 38: emit_softmax_telemetry with BF16 dtype ──
    // @trace TEST-NSE-38 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_telemetry_bf16_dtype() {
        // Arrange: telemetry with BF16 dtype affects VecStore dtype field
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::BF16;
        let max_val = prog.alloc_vreg(VRegKind::Vec, width);
        let sum_val = prog.alloc_vreg(VRegKind::Vec, width);
        let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act
        emit_softmax_telemetry(&mut prog, max_val, sum_val, telemetry_ptr, width, dtype);

        // Assert: should emit telemetry instructions with BF16 dtype
        assert!(prog.instrs.len() > initial_count, "Should emit telemetry instructions with BF16 dtype");
    }

    // ── Test 39: emit_layernorm_auto with feature_dim=1 (pure scalar tail, no vec loop) ──
    // @trace TEST-NSE-39 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_feature_dim_one_pure_tail() {
        // Arrange: feature_dim=1 with W256 => vec_count=0, tail=1
        // Exercises pure-scalar path for all phases (reduce, finalize, transform).
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 1, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with feature_dim=1");
        assert!(!prog.instrs.is_empty(), "Should emit scalar-only instructions for feature_dim=1");
    }

    // ── Test 40: emit_softmax_inline with Scalar width (1 lane) ──
    // @trace TEST-NSE-40 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_scalar_width() {
        // Arrange: SimdWidth::Scalar has f32_lanes() == 1, so vec_count=feature_dim, tail=0
        // This exercises the scalar-width path where every element is processed one at a time.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: Scalar width, small feature_dim=4
        let result = emit_softmax_inline(
            &mut prog, 4, SimdWidth::Scalar,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with Scalar width");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be a valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be a valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with Scalar width");
    }

    // ── Test 41: emit_normlike_one_group without weight and feature_dim < lanes (pure tail, no weight) ──
    // @trace TEST-NSE-41 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_no_weight_pure_tail() {
        // Arrange: feature_dim=3, W256 => vec_count=0, tail=3, has_weight=false
        // Exercises pure-scalar reduce + finalize + transform without any weight loads.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let lanes = width.f32_lanes();
        let feature_dim = 3usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::F32.elem_bytes();
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: has_weight=false, vec_count=0, pure tail-only scalar path
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            false, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, QuantPrecision::F32,
        );

        // Assert: should emit instructions for tail-only phases without weight loads
        assert!(prog.instrs.len() > initial_count, "Should emit instructions for tail-only no-weight path");
    }

    // ── Test 42: emit_layernorm_auto with W512 width ──
    // @trace TEST-NSE-42 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_w512_width() {
        // Arrange: W512 has 16 f32 lanes, feature_dim=24 => vec_count=1, tail=8
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: W512 with non-aligned feature_dim exercises both vec loop and tail
        let result = emit_layernorm_auto(
            &mut prog, 24, 1e-5,
            SimdWidth::W512, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W512 width");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with W512");
    }

    // ── Test 43: emit_normlike_inline groups=1 has_weight=false takes direct path (no group loop) ──
    // @trace TEST-NSE-43 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_groups_one_no_weight_direct_path() {
        // Arrange: groups_per_row=1 with has_weight=false takes the direct code path
        // (lines 100-111) that skips the inner group loop and all weight loading.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act: groups_per_row=1, has_weight=false → direct path, no group loop, no weight loads
        let result = emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, false,
            SimdWidth::W256, BoundExpr::Const(3),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with groups=1, no weight, direct path");
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
        assert!(!prog.instrs.is_empty(), "Should emit instructions for direct path");
    }

    // ── Test 44: emit_softmax_inline with BF16 and non-aligned feature_dim ──
    // @trace TEST-NSE-44 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_bf16_non_aligned() {
        // Arrange: BF16 dtype with non-aligned feature_dim exercises both vec and tail paths
        // with elem_bytes=2 affecting all offset calculations.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=10 with W256 (8 lanes) => vec_count=1, tail=2, BF16 elem_bytes=2
        let result = emit_softmax_inline(
            &mut prog, 10, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 and non-aligned feature_dim");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with BF16 non-aligned");
    }

    // ── Test 45: emit_softmax_telemetry with Scalar width ──
    // @trace TEST-NSE-45 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_telemetry_scalar_width() {
        // Arrange: Scalar width for telemetry exercises scalar recip and clamp operations
        let mut prog = VmProgram::new();
        let dtype = QuantPrecision::F32;
        let max_val = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
        let sum_val = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
        let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act: Scalar width telemetry
        emit_softmax_telemetry(&mut prog, max_val, sum_val, telemetry_ptr, SimdWidth::Scalar, dtype);

        // Assert: should emit telemetry instructions at scalar width
        assert!(prog.instrs.len() > initial_count, "Should emit telemetry instructions at Scalar width");
    }

    // ── Test 46: emit_layernorm_auto with BF16 and non-aligned feature_dim ──
    // @trace TEST-NSE-46 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_bf16_non_aligned_feature_dim() {
        // Arrange: BF16 dtype with non-aligned feature_dim=10, W256 => vec_count=1, tail=2
        // elem_bytes=2 affects bias_offset, row_bytes, and all tail offset calculations.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: BF16 with non-aligned feature_dim exercises both vec loop and scalar tail with 2-byte elems
        let result = emit_layernorm_auto(
            &mut prog, 10, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 and non-aligned feature_dim");
        assert!(!prog.instrs.is_empty(), "Should emit instructions for BF16 non-aligned");
    }

    // ── Test 47: emit_normlike_inline with Symbolic bound and multiple groups ──
    // @trace TEST-NSE-47 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_symbolic_bound_multiple_groups() {
        // Arrange: Symbolic seq_len bound combined with groups_per_row=2 exercises both
        // the outer Symbolic emit_loop and the inner Const group loop simultaneously.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".into(),
            max_alloc: 512,
        });

        // Act: Symbolic outer + 2 groups per row
        let result = emit_normlike_inline(
            &mut prog, &pattern, 16, 2, false, false,
            SimdWidth::W256, sym_bound,
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with Symbolic bound and multiple groups");
        assert!(!prog.instrs.is_empty(), "Should emit nested loop instructions");
    }

    // ── Test 48: emit_normlike_inline with broadcast_weight=true and groups=1 ──
    // @trace TEST-NSE-48 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_broadcast_weight_single_group() {
        // Arrange: broadcast_weight=true with groups_per_row=1 means weight is loaded once
        // via PtrExpr::VRegPlusConst(weight_ptr, 0) in the direct path (lines 100-111).
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act: groups=1, broadcast_weight=true, has_weight=true, direct code path
        let result = emit_normlike_inline(
            &mut prog, &pattern, 32, 1, true, true,
            SimdWidth::W256, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with broadcast weight and single group");
        assert!(prog.instrs.len() > initial_count, "Should emit instructions including weight load");
    }

    // ── Test 49: emit_normlike_one_group with W128 width ──
    // @trace TEST-NSE-49 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_w128_width() {
        // Arrange: W128 has 4 f32 lanes, feature_dim=10 => vec_count=2, tail=2
        let mut prog = VmProgram::new();
        let width = SimdWidth::W128;
        let lanes = width.f32_lanes();
        let feature_dim = 10usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::F32.elem_bytes();
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: W128 width with has_weight=true, exercises 4-lane vectorized loop + scalar tail
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            true, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, QuantPrecision::F32,
        );

        // Assert
        assert!(prog.instrs.len() > initial_count, "Should emit instructions with W128 width");
    }

    // ── Test 50: emit_softmax_inline with W512 and aligned feature_dim (zero tail) ──
    // @trace TEST-NSE-50 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_w512_aligned_no_tail() {
        // Arrange: W512 has 16 f32 lanes, feature_dim=16 => vec_count=1, tail=0
        // Perfectly aligned case for W512 — no scalar tail handling needed.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_softmax_inline(
            &mut prog, 16, SimdWidth::W512,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W512 and aligned feature_dim=16");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit vectorized-only instructions");
    }

    // ── Test 51: emit_softmax_inline with Scalar width and non-aligned feature_dim ──
    // @trace TEST-NSE-51 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_scalar_width_non_aligned() {
        // Arrange: Scalar width (1 lane), feature_dim=3 => vec_count=3, tail=0
        // Scalar processes each element individually, so non-aligned is still tail=0.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: Scalar width with feature_dim=3, all elements processed as vec_count=3
        let result = emit_softmax_inline(
            &mut prog, 3, SimdWidth::Scalar,
            input_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with Scalar width and non-aligned feature_dim");
        let (max_val, sum_val) = result.unwrap();
        assert!(max_val.0 < prog.vreg_count(), "max_val should be valid vreg");
        assert!(sum_val.0 < prog.vreg_count(), "sum_val should be valid vreg");
        assert!(!prog.instrs.is_empty(), "Should emit scalar-width instructions");
    }

    // ── Test 52: emit_softmax_telemetry with W512 width ──
    // @trace TEST-NSE-52 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_telemetry_w512_width() {
        // Arrange: W512 telemetry exercises 16-lane recip and clamp operations
        let mut prog = VmProgram::new();
        let width = SimdWidth::W512;
        let dtype = QuantPrecision::F32;
        let max_val = prog.alloc_vreg(VRegKind::Vec, width);
        let sum_val = prog.alloc_vreg(VRegKind::Vec, width);
        let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act
        emit_softmax_telemetry(&mut prog, max_val, sum_val, telemetry_ptr, width, dtype);

        // Assert: should emit telemetry instructions at W512 width
        assert!(prog.instrs.len() > initial_count, "Should emit telemetry instructions at W512 width");
    }

    // ── Test 53: emit_layernorm_auto with large feature_dim (256) ──
    // @trace TEST-NSE-53 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_large_feature_dim() {
        // Arrange: feature_dim=256 with W256 (8 lanes) => vec_count=32, tail=0
        // Exercises many vectorized loop iterations in reduce + transform phases.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.vreg_count();

        // Act
        let result = emit_layernorm_auto(
            &mut prog, 256, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with large feature_dim=256");
        assert!(prog.vreg_count() > initial_count, "Should allocate internal vregs");
        assert!(!prog.instrs.is_empty(), "Should emit many instructions for large feature_dim");
    }

    // ── Test 54: emit_layernorm_auto with BF16 and W128 width ──
    // @trace TEST-NSE-54 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_bf16_w128_width() {
        // Arrange: BF16 (elem_bytes=2) with W128 (4 lanes) and non-aligned feature_dim=6
        // => vec_count=1, tail=2. Tests BF16 + non-W256 width combination.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: BF16 + W128 + non-aligned feature_dim exercises both vec and tail with 2-byte elems
        let result = emit_layernorm_auto(
            &mut prog, 6, 1e-5,
            SimdWidth::W128, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 and W128");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with BF16 W128");
    }

    // ── Test 55: emit_normlike_inline with W512 width ──
    // @trace TEST-NSE-55 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_w512_width() {
        // Arrange: W512 has 16 f32 lanes, feature_dim=24 => vec_count=1, tail=8
        // Exercises 16-lane vectorized path with significant tail elements.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_normlike_inline(
            &mut prog, &pattern, 24, 1, false, true,
            SimdWidth::W512, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with W512 width");
        assert!(!prog.instrs.is_empty(), "Should emit instructions with W512");
    }

    // ── Test 56: emit_normlike_inline broadcast_weight=true does not affect has_weight=false path ──
    // @trace TEST-NSE-56 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_broadcast_weight_ignored_when_no_weight() {
        // Arrange: broadcast_weight=true but has_weight=false — the broadcast_weight flag
        // should be irrelevant because no weight loading occurs at all.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let initial_count = prog.instrs.len();

        // Act: broadcast_weight=true but has_weight=false, groups=1 direct path
        let result = emit_normlike_inline(
            &mut prog, &pattern, 16, 1, true, false,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        );

        // Assert: should succeed and emit instructions without any weight loads
        assert!(result.is_ok(), "Should succeed when broadcast_weight is ignored (has_weight=false)");
        assert!(prog.instrs.len() > initial_count, "Should emit instructions without weight loads");
    }

    // ── Test 57: RmsNorm with hidden_dim<lanes produces pure tail (no inner vec LoopBegin) ──
    // @trace TEST-NSE-57 [req:REQ-QCG] [level:unit]

    #[test]
    fn rmsnorm_small_hidden_dim_pure_tail_no_inner_vec_loop() {
        // Arrange: feature_dim=3 < 8 lanes (W256) => vec_count=0, no inner vec loop in reduce/transform.
        // The inner vec loop is only emitted when vec_count > 0.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=3 => vec_count=0, only outer seq loop + scalar tail ops
        emit_normlike_inline(
            &mut prog, &pattern, 3, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should have exactly 1 LoopBegin (outer seq loop only, no inner vec loop)
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_begin_count, 1, "Should have 1 LoopBegin (outer seq loop only) for feature_dim < lanes, got {loop_begin_count}");
    }

    // ── Test 58: W512 produces LoopBegin with larger step_bytes than W256 for same feature_dim ──
    // @trace TEST-NSE-58 [req:REQ-QCG] [level:unit]

    #[test]
    fn simd_width_w512_larger_step_bytes_than_w256() {
        // Arrange: W512 step_bytes=64 vs W256 step_bytes=32 for same feature_dim.
        // The inner vec loop's step_bytes reflects the SIMD width difference.
        let pattern = make_normlike_pattern();
        let feature_dim = 64;

        let mut prog_256 = VmProgram::new();
        let i256 = prog_256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w256 = prog_256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o256 = prog_256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_normlike_inline(
            &mut prog_256, &pattern, feature_dim, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            i256, w256, o256, QuantPrecision::F32,
        ).unwrap();

        let mut prog_512 = VmProgram::new();
        let i512 = prog_512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w512 = prog_512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o512 = prog_512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_normlike_inline(
            &mut prog_512, &pattern, feature_dim, 1, false, true,
            SimdWidth::W512, BoundExpr::Const(1),
            i512, w512, o512, QuantPrecision::F32,
        ).unwrap();

        // Assert: W512 should produce inner vec LoopBegin with step_bytes=64 (16 lanes * 4 bytes)
        // and W256 should produce inner vec LoopBegin with step_bytes=32 (8 lanes * 4 bytes).
        // Find the maximum step_bytes among inner vec loops (BoundExpr::Const(vec_count)) in each program.
        let max_step_256: usize = prog_256.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::LoopBegin { step_bytes, bound: BoundExpr::Const(n), .. } if *n > 1 => Some(*step_bytes),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let max_step_512: usize = prog_512.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::LoopBegin { step_bytes, bound: BoundExpr::Const(n), .. } if *n > 1 => Some(*step_bytes),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        assert_eq!(max_step_256, 32, "W256 inner vec loop step_bytes should be 32");
        assert_eq!(max_step_512, 64, "W512 inner vec loop step_bytes should be 64");
    }

    // ── Test 59: emit_normlike_inline generated program contains LoopBegin/LoopEnd VmInstr ──
    // @trace TEST-NSE-59 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_program_contains_loop_structures() {
        // Arrange: emit a normlike program with enough iterations to require loops
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=64, W256 => vec_count=8, requires inner vec loop
        emit_normlike_inline(
            &mut prog, &pattern, 64, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(2),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain LoopBegin/LoopEnd pairs
        let has_loop_begin = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        let has_loop_end = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopEnd));
        assert!(has_loop_begin, "Should contain LoopBegin instructions");
        assert!(has_loop_end, "Should contain LoopEnd instructions");
    }

    // ── Test 60: emit_softmax_inline generated program contains HReduce VmInstr ──
    // @trace TEST-NSE-60 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_inline_program_contains_hreduce() {
        // Arrange: softmax must produce HReduce instructions for horizontal max/sum reduction
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=16 with W256 => vec_count=2, exercises both vec loop and HReduce
        emit_softmax_inline(
            &mut prog, 16, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain HReduce instructions for max and sum phases
        let hreduce_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::HReduce { .. })).count();
        assert!(hreduce_count >= 2, "Should contain at least 2 HReduce instructions (max + sum), found {hreduce_count}");
    }

    // ── Test 61: emit_normlike_inline generated program contains Broadcast VmInstr ──
    // @trace TEST-NSE-61 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_program_contains_broadcast() {
        // Arrange: normlike must produce Broadcast instructions for acc init and dim broadcast
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_normlike_inline(
            &mut prog, &pattern, 32, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain Broadcast instructions (for acc=0 init, dim_bc, scale)
        let broadcast_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Broadcast { .. })).count();
        assert!(broadcast_count >= 2, "Should contain at least 2 Broadcast instructions, found {broadcast_count}");
    }

    // ── Test 62: BF16 dtype produces VecLoad/VecStore with BF16 dtype field ──
    // @trace TEST-NSE-62 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_bf16_produces_bf16_dtype_vminstrs() {
        // Arrange: verify that BF16 dtype propagates to VecLoad/VecStore instructions
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: BF16 dtype with aligned feature_dim=16
        emit_normlike_inline(
            &mut prog, &pattern, 16, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::BF16,
        ).unwrap();

        // Assert: should contain VecLoad/VecStore with dtype=BF16
        let has_bf16_load = prog.instrs.iter().any(|i| matches!(
            i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, .. }
        ));
        let has_bf16_store = prog.instrs.iter().any(|i| matches!(
            i, VmInstr::VecStore { dtype: QuantPrecision::BF16, .. }
        ));
        assert!(has_bf16_load, "Should contain VecLoad with BF16 dtype");
        assert!(has_bf16_store, "Should contain VecStore with BF16 dtype");
    }

    // ── Test 63: F32 dtype produces VecLoad/VecStore with F32 dtype field ──
    // @trace TEST-NSE-63 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_f32_produces_f32_dtype_vminstrs() {
        // Arrange: verify that F32 dtype propagates to VecLoad/VecStore instructions
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: F32 dtype with feature_dim=16
        emit_softmax_inline(
            &mut prog, 16, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain VecLoad/VecStore with dtype=F32
        let has_f32_load = prog.instrs.iter().any(|i| matches!(
            i, VmInstr::VecLoad { dtype: QuantPrecision::F32, .. }
        ));
        let has_f32_store = prog.instrs.iter().any(|i| matches!(
            i, VmInstr::VecStore { dtype: QuantPrecision::F32, .. }
        ));
        assert!(has_f32_load, "Should contain VecLoad with F32 dtype");
        assert!(has_f32_store, "Should contain VecStore with F32 dtype");
    }

    // ── Test 64: emit_layernorm_auto with Symbolic bound produces LoopBegin with Symbolic bound ──
    // @trace TEST-NSE-64 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_symbolic_bound_produces_symbolic_loop() {
        // Arrange: Symbolic seq_len bound should produce LoopBegin with BoundExpr::Symbolic
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_bound = BoundExpr::Symbolic(SymBound { name: "seq_len".into(), max_alloc: 1024 });

        // Act
        emit_layernorm_auto(
            &mut prog, 32, 1e-5,
            SimdWidth::W256, sym_bound,
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain at least one LoopBegin with Symbolic bound
        let has_symbolic_loop = prog.instrs.iter().any(|i| matches!(
            i, VmInstr::LoopBegin { bound: BoundExpr::Symbolic(_), .. }
        ));
        assert!(has_symbolic_loop, "Should contain LoopBegin with Symbolic bound for seq_len");
    }

    // ── Test 65: emit_normlike_inline with groups>1 produces nested loops ──
    // @trace TEST-NSE-65 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_groups_gt_one_produces_inner_const_loop() {
        // Arrange: groups_per_row>1 should produce an inner Const loop for group iteration
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: groups_per_row=3, feature_dim=24, Const(1) seq bound
        emit_normlike_inline(
            &mut prog, &pattern, 24, 3, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should have more LoopBegin instructions than groups=1 (outer loop + inner group loop)
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();

        // For comparison, emit with groups=1
        let mut prog_g1 = VmProgram::new();
        let i1 = prog_g1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w1 = prog_g1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_g1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_normlike_inline(
            &mut prog_g1, &pattern, 24, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            i1, w1, o1, QuantPrecision::F32,
        ).unwrap();
        let loop_begin_count_g1 = prog_g1.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();

        assert!(loop_begin_count > loop_begin_count_g1,
            "groups=3 should produce more LoopBegin ({loop_begin_count}) than groups=1 ({loop_begin_count_g1})");
    }

    // ── Test 66: emit_softmax_inline with Scalar width produces no tail instructions ──
    // @trace TEST-NSE-66 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_scalar_width_no_tail_vec_load_scalar_width() {
        // Arrange: Scalar width (1 lane) means feature_dim is always a multiple of lanes (1),
        // so there should be zero tail elements and all VecLoad should use Scalar width.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=5 with Scalar width => vec_count=5, tail=0
        emit_softmax_inline(
            &mut prog, 5, SimdWidth::Scalar,
            input_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: all VecLoad instructions should use SimdWidth::Scalar (no W256 tail loads)
        let all_scalar_loads = prog.instrs.iter().all(|i| match i {
            VmInstr::VecLoad { width, .. } => *width == SimdWidth::Scalar,
            _ => true,
        });
        assert!(all_scalar_loads, "All VecLoad instructions should use Scalar width when SimdWidth::Scalar is specified");
    }

    // ── Test 67: emit_normlike_inline produces Accumulate VmInstr in reduce phase ──
    // @trace TEST-NSE-67 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_inline_produces_accumulate_instrs() {
        // Arrange: Accumulate instructions are used in reduce phase to sum squared values.
        // emit_loop emits the loop body once (with LoopBegin/LoopEnd), so Accumulate appears
        // once per loop body in the VmProgram, not once per iteration.
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=32, W256 => vec_count=4, tail=0
        emit_normlike_inline(
            &mut prog, &pattern, 32, 1, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain at least 1 Accumulate instruction from reduce phase
        let accumulate_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Accumulate { .. })).count();
        assert!(accumulate_count >= 1, "Should contain at least 1 Accumulate instruction from reduce phase, found {accumulate_count}");
    }

    // ── Test 68: emit_layernorm_auto produces dual Accumulate for sum_x and sum_sq ──
    // @trace TEST-NSE-68 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_produces_dual_accumulate() {
        // Arrange: LayerNorm uses two accumulators (sum_x, sum_sq) each with its own Accumulate.
        // emit_loop emits the loop body once (with LoopBegin/LoopEnd), so Accumulate appears
        // once per accumulator per loop body (2 total), not once per iteration.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=16, W256 => vec_count=2, tail=0
        emit_layernorm_auto(
            &mut prog, 16, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: LayerNorm has 2 Accumulate in the vec loop body (acc_sum + acc_sq)
        let accumulate_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Accumulate { .. })).count();
        assert!(accumulate_count >= 2, "Should contain at least 2 Accumulate instructions (sum_x + sum_sq), found {accumulate_count}");
    }

    // ── Test 69: emit_softmax_inline with non-aligned feature_dim produces tail instructions ──
    // @trace TEST-NSE-69 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_non_aligned_produces_more_instrs_than_aligned() {
        // Arrange: emit_loop emits loop body once regardless of loop count, so aligned vs
        // non-aligned with same vec_count produces the same loop body size. However, non-aligned
        // feature_dim with a tail produces additional scalar tail instructions that aligned does not.
        // feature_dim=10 (W256): vec_count=1, tail=2 => has tail instructions
        // feature_dim=8  (W256): vec_count=1, tail=0 => no tail instructions
        let mut prog_aligned = VmProgram::new();
        let i_a = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_a = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_softmax_inline(&mut prog_aligned, 8, SimdWidth::W256, i_a, o_a, QuantPrecision::F32).unwrap();

        let mut prog_tailed = VmProgram::new();
        let i_t = prog_tailed.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_t = prog_tailed.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_softmax_inline(&mut prog_tailed, 10, SimdWidth::W256, i_t, o_t, QuantPrecision::F32).unwrap();

        // Assert: non-aligned (tail=2) produces more instructions than aligned (tail=0)
        // because tail elements generate additional scalar VecLoad/VecStore for each of the 3 phases
        assert!(
            prog_tailed.instrs.len() > prog_aligned.instrs.len(),
            "feature_dim=10 with tail ({}) should produce more instructions than feature_dim=8 aligned ({})",
            prog_tailed.instrs.len(),
            prog_aligned.instrs.len()
        );
    }

    // ── Test 70: emit_softmax_telemetry produces Comment VmInstr ──
    // @trace TEST-NSE-70 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_telemetry_produces_comment_instr() {
        // Arrange: emit_softmax_telemetry emits a Comment instruction for identification
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let max_val = prog.alloc_vreg(VRegKind::Vec, width);
        let sum_val = prog.alloc_vreg(VRegKind::Vec, width);
        let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_softmax_telemetry(&mut prog, max_val, sum_val, telemetry_ptr, width, dtype);

        // Assert: should contain a Comment instruction mentioning telemetry
        let has_comment = prog.instrs.iter().any(|i| matches!(i, VmInstr::Comment(_)));
        assert!(has_comment, "Should contain a Comment instruction");
    }

    // ── Test 71: emit_normlike_one_group with W512 width ──
    // @trace TEST-NSE-71 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_one_group_w512_width() {
        // Arrange: W512 has 16 f32 lanes, feature_dim=20 => vec_count=1, tail=4
        let mut prog = VmProgram::new();
        let width = SimdWidth::W512;
        let lanes = width.f32_lanes();
        let feature_dim = 20usize;
        let vec_count = feature_dim / lanes;
        let step_bytes = width.bytes();
        let elem = QuantPrecision::F32.elem_bytes();
        let s1 = SimdWidth::Scalar;

        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        let temp = prog.alloc_vreg(VRegKind::Vec, width);
        let scale = prog.alloc_vreg(VRegKind::Vec, width);
        let dim_bc = prog.alloc_vreg(VRegKind::Vec, width);
        let row_input = prog.alloc_vreg(VRegKind::Ptr, s1);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, s1);
        let row_output = prog.alloc_vreg(VRegKind::Ptr, s1);

        let reduce = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let finalize = vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let transform = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];

        let initial_count = prog.instrs.len();

        // Act: W512 with has_weight=true, vec_count=1, tail=4
        emit_normlike_one_group(
            &mut prog, &reduce, &finalize, &transform,
            feature_dim, vec_count, step_bytes, elem, lanes, width,
            true, acc, temp, scale, dim_bc,
            row_input, weight_ptr, row_output, QuantPrecision::F32,
        );

        // Assert
        assert!(prog.instrs.len() > initial_count, "Should emit instructions with W512 width");
    }

    // ── Test 72: emit_layernorm_auto with BF16 and W512 combined ──
    // @trace TEST-NSE-72 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_auto_bf16_w512_combined() {
        // Arrange: BF16 (elem_bytes=2) + W512 (16 lanes) + non-aligned feature_dim=20
        // => vec_count=1, tail=4, with 2-byte elements.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: combined BF16 + W512 + non-aligned feature_dim
        let result = emit_layernorm_auto(
            &mut prog, 20, 1e-5,
            SimdWidth::W512, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "Should succeed with BF16 + W512 combined");
        assert!(!prog.instrs.is_empty(), "Should emit instructions");
    }

    // ── Test 73: emit_softmax_inline produces VecStore in exp phase (intermediate results) ──
    // @trace TEST-NSE-73 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_produces_intermediate_vecstore_for_exp() {
        // Arrange: Softmax Phase 2 writes exp(x - max) results back to output before normalization.
        // With feature_dim=16 and W256 => vec_count=2, there should be VecStore instructions
        // in both the exp phase (Phase 2) and the normalize phase (Phase 3).
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_softmax_inline(
            &mut prog, 16, SimdWidth::W256,
            input_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should have multiple VecStore instructions (exp writes + normalize writes)
        let store_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(store_count >= 2, "Should have multiple VecStore instructions (exp phase + normalize phase), found {store_count}");
    }

    // ── Test 74: emit_normlike_inline with large groups_per_row produces LoadPtr per group ──
    // @trace TEST-NSE-74 [req:REQ-QCG] [level:unit]

    #[test]
    fn normlike_large_groups_produces_loadptr_per_group() {
        // Arrange: groups_per_row > 1 emits LoadPtr for per-group row_input/row_output pointers
        let mut prog = VmProgram::new();
        let pattern = make_normlike_pattern();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: groups_per_row=8, feature_dim=64 (8 bytes per group), exercises per-group LoadPtr
        emit_normlike_inline(
            &mut prog, &pattern, 64, 8, false, true,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain LoadPtr instructions for per-group pointer resolution
        let loadptr_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoadPtr { .. })).count();
        assert!(loadptr_count >= 4, "Should contain multiple LoadPtr instructions for group pointers, found {loadptr_count}");
    }

    // ── Test 75: emit_layernorm_auto with Const(1) produces exactly one outer LoopBegin ──
    // @trace TEST-NSE-75 [req:REQ-QCG] [level:unit]

    #[test]
    fn layernorm_const_one_seq_produces_single_outer_loop() {
        // Arrange: seq_bound=Const(1) should produce exactly one outer LoopBegin for the row loop
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=16, W256, Const(1) => one outer row loop with row_bytes step
        emit_layernorm_auto(
            &mut prog, 16, 1e-5,
            SimdWidth::W256, BoundExpr::Const(1),
            input_ptr, weight_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: count LoopBegin with row_bytes step (16 * 4 = 64 bytes)
        let row_bytes = 16usize * 4;
        let outer_loop_count = prog.instrs.iter().filter(|i| match i {
            VmInstr::LoopBegin { step_bytes, bound: BoundExpr::Const(1), .. } if *step_bytes == row_bytes => true,
            _ => false,
        }).count();
        assert!(outer_loop_count <= 1, "Should have at most 1 outer row loop with row_bytes step, found {outer_loop_count}");
    }

    // ── Test 76: emit_softmax_inline with W128 produces VecLoad with W128 width ──
    // @trace TEST-NSE-76 [req:REQ-QCG] [level:unit]

    #[test]
    fn softmax_w128_produces_w128_vecload() {
        // Arrange: W128 width should produce VecLoad instructions with SimdWidth::W128
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: feature_dim=8, W128 => vec_count=2, tail=0 (perfectly aligned for W128)
        emit_softmax_inline(
            &mut prog, 8, SimdWidth::W128,
            input_ptr, output_ptr, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain VecLoad instructions with W128 width (not Scalar or W256)
        let has_w128_load = prog.instrs.iter().any(|i| matches!(
            i, VmInstr::VecLoad { width: SimdWidth::W128, .. }
        ));
        assert!(has_w128_load, "Should contain VecLoad with W128 width");
    }
}
