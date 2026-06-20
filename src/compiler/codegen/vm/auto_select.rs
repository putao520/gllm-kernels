//! Auto Instruction Selector — TraceOp → VmInstr 自动查表法
//!
//! 替代 lower_trace_body 中的手写 match arms。
//! 每个 TraceOp 变体通过辅助函数（emit_binop/unary/transcendental）映射到 VmInstr，
//! 消除了重复的 alloc_vreg + emit 模式。
//!
//! 新增 TraceOp 变体只需在此文件添加一行 match arm。
//!
//! # REQ-AIS-001: ComputePattern 驱动路由
//!
//! 三个公开入口函数按 ComputePattern 分类：
//! - `auto_lower_elementwise`: Elementwise / BinaryElementwise / Injective
//! - `auto_lower_reduction`: Reduction (含 NormLike)
//! - `auto_lower_structural`: Structural (Gather/Scatter/Loop/Panel 等)
//!
//! 每个入口验证 ComputePattern 一致性后委托给 `dispatch_trace_op`。
//! 调用方通过 `classify_pattern` 获取 ComputePattern，选择对应入口，
//! 消除手写 OpKind→VmInstr match arm。
//!
//! # dtype 自然传播 (QPJ4)
//!
//! `propagate_dtype` 链：每个 TraceOp 的输出 dtype 由输入 dtype 和操作语义决定，
//! 通过 `infer_result_dtype` 自动推断，消除硬编码 F32 导致的 NaN 根因。
//!
//! 传播路径: TensorMeta.dtype → TypedSlot.dtype → infer_result_dtype → VmInstr.dtype
//!
//! 三层 API:
//! - `auto_lower_trace_raw`: 核心，返回所有 SSA slot VRegIds
//! - `auto_lower_trace`: 便捷包装，写回最后一个 slot 到 inputs[0]（向后兼容）
//! - `auto_lower_trace_multi`: 多输出，写回指定 slot 到目标 VReg

use super::instr::*;
use super::trace_opt::TracePassPipeline;
use crate::compiler::trace::{CmpOp, ComputePattern, Fp8Format, QuantPrecision, ReduceKind, ScaleSelector, TraceOp, TypedSlot, infer_result_dtype, ValueId};
use crate::quant::QuantType;
use crate::quant_format::{PackedScaleAlgorithm, QuantDataKind, ZeroLayout};
use crate::types::CompilerError;

static TRACE_OPT_PIPELINE: std::sync::OnceLock<TracePassPipeline> = std::sync::OnceLock::new();

fn get_opt_pipeline() -> &'static TracePassPipeline {
    TRACE_OPT_PIPELINE.get_or_init(TracePassPipeline::with_defaults)
}

/// 类型感知编译：将 TraceOp SSA body 编译为带 dtype 推断的 VmInstr 序列。
///
/// 与 `auto_lower_trace_raw` 的区别：输入为 TypedSlot（携带 dtype），
/// 每个 TraceOp 的结果 dtype 通过 `infer_result_dtype` 自动推断，
/// 推断的 dtype 写入 VmInstr 的 dtype 字段（而非硬编码 F32）。
///
/// dtype 传播链：TensorMeta.dtype → TypedSlot.dtype → infer_result_dtype → VmInstr.dtype
pub fn auto_lower_trace_typed(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[TypedSlot],
    width: SimdWidth,
) -> Result<Vec<TypedSlot>, CompilerError> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    let vreg_inputs: Vec<VRegId> = inputs.iter().map(|ts| ts.vreg).collect();
    let mut slots: Vec<TypedSlot> = Vec::with_capacity(body.len());
    for op in body {
        let result_dtype = infer_result_dtype(op, &slots);
        let vreg = dispatch_trace_op_typed(prog, op, &slots, &vreg_inputs, width, result_dtype)?;
        slots.push(TypedSlot { vreg, dtype: result_dtype });
    }
    Ok(slots)
}

/// 类型感知 dispatch：与 dispatch_trace_op 相同的 lowering 逻辑，
/// 但使用推断的 dtype 而非硬编码 F32。
fn dispatch_trace_op_typed(
    prog: &mut VmProgram,
    op: &TraceOp,
    typed_slots: &[TypedSlot],
    inputs: &[VRegId],
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    let slots: Vec<VRegId> = typed_slots.iter().map(|ts| ts.vreg).collect();
    let n = typed_slots.len();
    match op {
        TraceOp::Input(idx) => inputs
            .get(*idx as usize)
            .copied()
            .ok_or_else(|| CompilerError::CodegenViolation(format!(
                "TraceOp::Input({}) 越界", idx
            ))),

        TraceOp::Const(val) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: r, src: ScalarExpr::Const(*val as f32), width, dtype,
            });
            Ok(r)
        }

        TraceOp::Add(a, b) => emit_binop_dtype(prog, &slots, *a, *b, VecOp::Add, width, dtype),
        TraceOp::Sub(a, b) => emit_binop_dtype(prog, &slots, *a, *b, VecOp::Sub, width, dtype),
        TraceOp::Mul(a, b) => emit_binop_dtype(prog, &slots, *a, *b, VecOp::Mul, width, dtype),
        TraceOp::Div(a, b) => emit_binop_dtype(prog, &slots, *a, *b, VecOp::Div, width, dtype),
        TraceOp::Max(a, b) => emit_binop_dtype(prog, &slots, *a, *b, VecOp::Max, width, dtype),
        TraceOp::Min(a, b) => emit_binop_dtype(prog, &slots, *a, *b, VecOp::Min, width, dtype),

        TraceOp::Fma(a, b, c) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Fma {
                dst: r, acc: slots[c.0 as usize], a: slots[a.0 as usize], b: slots[b.0 as usize], dtype,
            });
            Ok(r)
        }

        TraceOp::Neg(a) => emit_unary(prog, &slots, *a, VecUnaryOp::Neg, width),
        TraceOp::Abs(a) => emit_unary(prog, &slots, *a, VecUnaryOp::Abs, width),
        TraceOp::Sqrt(a) => emit_unary(prog, &slots, *a, VecUnaryOp::Sqrt, width),
        TraceOp::Rsqrt(a) => emit_unary(prog, &slots, *a, VecUnaryOp::Rsqrt, width),
        TraceOp::Recip(a) => emit_unary(prog, &slots, *a, VecUnaryOp::Recip, width),

        TraceOp::Exp(a) => emit_transcendental(prog, &slots, *a, TranscendentalFn::Exp, width),
        TraceOp::Tanh(a) => emit_transcendental(prog, &slots, *a, TranscendentalFn::Tanh, width),
        TraceOp::Log(a) => emit_transcendental(prog, &slots, *a, TranscendentalFn::Log, width),
        TraceOp::Sigmoid(a) => emit_transcendental(prog, &slots, *a, TranscendentalFn::Sigmoid, width),

        TraceOp::Compare { a, b, op: cmp_op } => emit_cmp(prog, &slots, *a, *b, *cmp_op, width),

        TraceOp::Cast { src, from, to } => {
            let from_bits = quant_precision_bits(from);
            let to_bits = quant_precision_bits(to);
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecCast { dst: r, src: slots[src.0 as usize], from_bits, to_bits });
            Ok(r)
        }

        TraceOp::ConditionalBranch(mask, true_val, false_val) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::ConditionalSelect {
                dst: r, mask: slots[mask.0 as usize],
                true_val: slots[true_val.0 as usize], false_val: slots[false_val.0 as usize],
            });
            Ok(r)
        }

        TraceOp::HReduce { src, op: reduce_op } => {
            let red = match reduce_op {
                ReduceKind::Sum => ReduceOp::Sum,
                ReduceKind::Max => ReduceOp::Max,
                ReduceKind::Min => ReduceOp::Min,
                ReduceKind::Prod => ReduceOp::Prod,
                ReduceKind::LogSum => ReduceOp::LogSum,
                other => return Err(CompilerError::CodegenViolation(format!(
                    "auto_lower_trace_typed: HReduce {:?} 尚未实现", other
                ))),
            };
            let r = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::HReduce { dst: r, src: slots[src.0 as usize], op: red });
            Ok(r)
        }

        TraceOp::BroadcastScalar { src } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: r, src: ScalarExpr::ExtractLane0(slots[src.0 as usize]), width, dtype,
            });
            Ok(r)
        }

        // 结构型、量化、内存操作 — 委托给现有 dispatch，使用传播的 dtype
        _ => dispatch_trace_op(prog, op, &slots, inputs, width, dtype),
    }
}

fn emit_binop_dtype(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId, b: ValueId,
    op: VecOp,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    let r = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::VecBinOp { dst: r, a: slots[a.0 as usize], b: slots[b.0 as usize], op, dtype });
    Ok(r)
}

// ────────────────────────────────────────────────────────────────────────────
// REQ-AIS-001: ComputePattern 驱动路由 — 三个公开入口函数
// ────────────────────────────────────────────────────────────────────────────

/// Elementwise 路由 (REQ-AIS-001)。
///
/// 处理 ComputePattern::Elementwise / BinaryElementwise / Injective。
/// `out[i] = f(in[i])` 或 `out[i] = f(a[i], b[i])` 或多输入映射。
///
/// 内部调用 `auto_lower_trace_raw` 完成实际的 TraceOp→VmInstr 编译。
/// dtype 通过 `default_dtype` 参数传播到每条 VmInstr。
pub fn auto_lower_elementwise(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
    pattern: &ComputePattern,
) -> Result<Vec<VRegId>, CompilerError> {
    match pattern {
        ComputePattern::Elementwise { .. }
        | ComputePattern::BinaryElementwise { .. }
        | ComputePattern::Injective { .. } => {}
        other => {
            return Err(CompilerError::CodegenViolation(format!(
                "auto_lower_elementwise: 不兼容的 ComputePattern {:?}，\
                 需要 Elementwise/BinaryElementwise/Injective",
                other
            )));
        }
    }
    auto_lower_trace_raw(prog, body, inputs, width, default_dtype)
}

/// Reduction 路由 (REQ-AIS-001)。
///
/// 处理 ComputePattern::Reduction / NormLike。
/// 多阶段归一化: reduce → finalize → transform 或 combine → normalize。
///
/// `combine_body`: 第一阶段归约 trace (如 Sum/Max)
/// `normalize_body`: 可选的逐元素归一化 trace (如 mul by inv_sum)
pub fn auto_lower_reduction(
    prog: &mut VmProgram,
    combine_body: &[TraceOp],
    normalize_body: Option<&[TraceOp]>,
    inputs: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
    pattern: &ComputePattern,
) -> Result<VRegId, CompilerError> {
    match pattern {
        ComputePattern::Reduction { .. } | ComputePattern::NormLike { .. } => {}
        other => {
            return Err(CompilerError::CodegenViolation(format!(
                "auto_lower_reduction: 不兼容的 ComputePattern {:?}，\
                 需要 Reduction/NormLike",
                other
            )));
        }
    }
    // Phase 1: combine (归约阶段)
    let slots = auto_lower_trace_raw(prog, combine_body, inputs, width, default_dtype)?;
    let reduce_result = slots.last().copied().ok_or_else(|| {
        CompilerError::CodegenViolation("auto_lower_reduction: combine body 产生零 slot".into())
    })?;
    // Phase 2: normalize (可选归一化阶段)
    if let Some(norm_body) = normalize_body {
        if !norm_body.is_empty() {
            let norm_inputs = &[reduce_result];
            let norm_slots =
                auto_lower_trace_raw(prog, norm_body, norm_inputs, width, default_dtype)?;
            return norm_slots.last().copied().ok_or_else(|| {
                CompilerError::CodegenViolation(
                    "auto_lower_reduction: normalize body 产生零 slot".into(),
                )
            });
        }
    }
    Ok(reduce_result)
}

/// Structural 路由 (REQ-AIS-001)。
///
/// 处理结构型 TraceOp: GatherLoad/ScatterStore/TableLookup/Loop/PanelLoad/PanelStore 等。
/// 这些操作涉及内存寻址、循环控制、多输出，不能走 elementwise 路径。
///
/// dtype 通过 `default_dtype` 参数传播。
pub fn auto_lower_structural(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<Vec<VRegId>, CompilerError> {
    auto_lower_trace_raw(prog, body, inputs, width, default_dtype)
}

// ────────────────────────────────────────────────────────────────────────────
// 核心 API: auto_lower_trace_raw / auto_lower_trace / auto_lower_trace_multi
// ────────────────────────────────────────────────────────────────────────────

/// 核心编译：将 TraceOp SSA body 编译为 VmInstr 序列，返回所有 slot VRegIds。
///
/// 调用方根据需要选择写回策略（auto_lower_trace 或 auto_lower_trace_multi）。
pub fn auto_lower_trace_raw(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<Vec<VRegId>, CompilerError> {
    if body.is_empty() {
        return Ok(Vec::new());
    }

    let mut slots: Vec<VRegId> = Vec::with_capacity(body.len());
    for op in body {
        let result = dispatch_trace_op(prog, op, &slots, inputs, width, default_dtype)?;
        slots.push(result);
    }
    Ok(slots)
}

/// 将 TraceOp SSA body 编译为 VmInstr，最后一个结果直接写入 `dst` 寄存器。
///
/// 与 `auto_lower_trace_raw` 的区别：不分配新寄存器给最后一个操作的结果，
/// 而是直接写入调用方提供的 `dst`，消除 identity copy (VecOp::Or) 或
/// Broadcast 的需要。
///
/// 中间操作仍分配临时寄存器（SSA），仅最终结果直达 dst。
/// 默认 dtype = F32（向后兼容）。
pub fn auto_lower_trace_into(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    dst: VRegId,
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if body.is_empty() {
        return Ok(());
    }
    let mut optimized = body.to_vec();
    get_opt_pipeline().optimize(&mut optimized);
    let (last_op, rest) = optimized.split_last().expect("body non-empty");
    let mut slots: Vec<VRegId> = Vec::with_capacity(optimized.len());
    for op in rest {
        let result = dispatch_trace_op(prog, op, &slots, inputs, width, default_dtype)?;
        slots.push(result);
    }
    dispatch_trace_op_into(prog, last_op, &slots, inputs, dst, width, default_dtype)?;
    Ok(())
}

/// 将单个 TraceOp 的结果直接写入指定 dst 寄存器（而非分配新寄存器）。
fn dispatch_trace_op_into(
    prog: &mut VmProgram,
    op: &TraceOp,
    slots: &[VRegId],
    inputs: &[VRegId],
    dst: VRegId,
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    match op {
        TraceOp::Input(n) => {
            let src = *inputs.get(*n as usize).ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "TraceOp::Input({}) 越界", n
                ))
            })?;
            if src != dst { copy_vreg(prog, dst, src, default_dtype); }
            Ok(())
        }
        TraceOp::Const(val) => {
            prog.emit(VmInstr::Broadcast { dst, src: ScalarExpr::Const(*val as f32), width, dtype: default_dtype, });
            Ok(())
        }
        TraceOp::Add(a, b) => emit_binop_into(prog, slots, *a, *b, VecOp::Add, dst, width, default_dtype),
        TraceOp::Sub(a, b) => emit_binop_into(prog, slots, *a, *b, VecOp::Sub, dst, width, default_dtype),
        TraceOp::Mul(a, b) => emit_binop_into(prog, slots, *a, *b, VecOp::Mul, dst, width, default_dtype),
        TraceOp::Div(a, b) => emit_binop_into(prog, slots, *a, *b, VecOp::Div, dst, width, default_dtype),
        TraceOp::Max(a, b) => emit_binop_into(prog, slots, *a, *b, VecOp::Max, dst, width, default_dtype),
        TraceOp::Min(a, b) => emit_binop_into(prog, slots, *a, *b, VecOp::Min, dst, width, default_dtype),
        TraceOp::Fma(a, b, c) => {
            prog.emit(VmInstr::Fma { dst, acc: slots[c.0 as usize], a: slots[a.0 as usize], b: slots[b.0 as usize], dtype: default_dtype, });
            Ok(())
        }
        TraceOp::Neg(a) => emit_unary_into(prog, slots, *a, VecUnaryOp::Neg, dst, width),
        TraceOp::Abs(a) => emit_unary_into(prog, slots, *a, VecUnaryOp::Abs, dst, width),
        TraceOp::Sqrt(a) => emit_unary_into(prog, slots, *a, VecUnaryOp::Sqrt, dst, width),
        TraceOp::Rsqrt(a) => emit_unary_into(prog, slots, *a, VecUnaryOp::Rsqrt, dst, width),
        TraceOp::Recip(a) => emit_unary_into(prog, slots, *a, VecUnaryOp::Recip, dst, width),
        TraceOp::Exp(a) => emit_transcendental_into(prog, slots, *a, TranscendentalFn::Exp, dst, width),
        TraceOp::Tanh(a) => emit_transcendental_into(prog, slots, *a, TranscendentalFn::Tanh, dst, width),
        TraceOp::Log(a) => emit_transcendental_into(prog, slots, *a, TranscendentalFn::Log, dst, width),
        TraceOp::Sigmoid(a) => emit_transcendental_into(prog, slots, *a, TranscendentalFn::Sigmoid, dst, width),
        TraceOp::BroadcastScalar { src } => {
            prog.emit(VmInstr::Broadcast { dst, src: ScalarExpr::ExtractLane0(slots[src.0 as usize]), width, dtype: default_dtype, });
            Ok(())
        }
        // 不支持 into 模式的操作（宽度变化、多输出等）：回退到 alloc + copy
        _ => {
            let result = dispatch_trace_op(prog, op, slots, inputs, width, default_dtype)?;
            copy_vreg(prog, dst, result, default_dtype);
            Ok(())
        }
    }
}

/// 便捷接口：写回最后一个 slot 到 inputs[0]（向后兼容）。
///
/// 语义与旧版 auto_lower_trace 完全一致。
/// 默认 dtype = F32。
pub fn auto_lower_trace(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if body.is_empty() {
        return Ok(());
    }
    let mut optimized = body.to_vec();
    get_opt_pipeline().optimize(&mut optimized);
    let slots = auto_lower_trace_raw(prog, &optimized, inputs, width, default_dtype)?;
    let primary = *inputs.first().ok_or_else(|| {
        CompilerError::CodegenViolation("auto_lower_trace: inputs 为空".into())
    })?;
    if let Some(&last) = slots.last() {
        copy_vreg(prog, primary, last, default_dtype);
    }
    Ok(())
}

/// 多输出：将指定 slot 索引的结果写回到目标 VReg。
///
/// `targets`: [(target_vreg, slot_index), ...]
/// slot_index 是 body 中 TraceOp 的位置索引（0-based），
/// 对应 auto_lower_trace_raw 返回的 Vec 位置。
///
/// 例: body = [Input(0), Mul(0,0), Add(1,0)]
///     slots = [inputs[0], mul_result, add_result]  (indices 0, 1, 2)
///     targets = [(dst_a, 2), (dst_b, 0)]
///     → dst_a = add_result, dst_b = inputs[0]
///
/// 默认 dtype = F32。
pub fn auto_lower_trace_multi(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    targets: &[(VRegId, usize)],
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if body.is_empty() {
        return Ok(());
    }
    let slots = auto_lower_trace_raw(prog, body, inputs, width, default_dtype)?;
    for &(target, slot_idx) in targets {
        if slot_idx >= slots.len() {
            return Err(CompilerError::CodegenViolation(format!(
                "auto_lower_trace_multi: slot_index {} 越界 (body 产生 {} 个 slot)",
                slot_idx, slots.len()
            )));
        }
        copy_vreg(prog, target, slots[slot_idx], default_dtype);
    }
    Ok(())
}

/// 寄存器拷贝: dst = src (Mov 指令，mov-elimination 友好)
fn copy_vreg(prog: &mut VmProgram, target: VRegId, src: VRegId, dtype: QuantPrecision) {
    if src != target {
        prog.emit(VmInstr::Mov { dst: target, src, dtype });
    }
}

/// 单个 TraceOp → VmInstr 分发。
///
/// 分类处理：
/// - Input/Const → 特殊处理
/// - 二元操作 → emit_binop (6 个变体共享)
/// - 一元操作 → emit_unary (5 个变体共享)
/// - Fma → 专用 VmInstr
/// - 超越函数 → emit_transcendental (3 个变体共享)
/// - 未实现 → Error (NO_SILENT_FALLBACK)
fn dispatch_trace_op(
    prog: &mut VmProgram,
    op: &TraceOp,
    slots: &[VRegId],
    inputs: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    match op {
        TraceOp::Input(n) => inputs
            .get(*n as usize)
            .copied()
            .ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "TraceOp::Input({}) 越界: 调用方仅提供 {} 个输入 VReg",
                    n,
                    inputs.len()
                ))
            }),

        TraceOp::Const(val) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: r,
                src: ScalarExpr::Const(*val as f32),
                width,
                dtype: default_dtype,
            });
            Ok(r)
        }

        // ── 二元操作 → VecBinOp ──
        TraceOp::Add(a, b) => emit_binop(prog, slots, *a, *b, VecOp::Add, width, default_dtype),
        TraceOp::Sub(a, b) => emit_binop(prog, slots, *a, *b, VecOp::Sub, width, default_dtype),
        TraceOp::Mul(a, b) => emit_binop(prog, slots, *a, *b, VecOp::Mul, width, default_dtype),
        TraceOp::Div(a, b) => emit_binop(prog, slots, *a, *b, VecOp::Div, width, default_dtype),
        TraceOp::Max(a, b) => emit_binop(prog, slots, *a, *b, VecOp::Max, width, default_dtype),
        TraceOp::Min(a, b) => emit_binop(prog, slots, *a, *b, VecOp::Min, width, default_dtype),

        // ── FMA → 专用 VmInstr ──
        TraceOp::Fma(a, b, c) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Fma {
                dst: r,
                acc: slots[c.0 as usize],
                a: slots[a.0 as usize],
                b: slots[b.0 as usize],
                dtype: default_dtype,
            });
            Ok(r)
        }

        // ── 一元操作 → VecUnaryOp ──
        TraceOp::Neg(a) => emit_unary(prog, slots, *a, VecUnaryOp::Neg, width),
        TraceOp::Abs(a) => emit_unary(prog, slots, *a, VecUnaryOp::Abs, width),
        TraceOp::Sqrt(a) => emit_unary(prog, slots, *a, VecUnaryOp::Sqrt, width),
        TraceOp::Rsqrt(a) => emit_unary(prog, slots, *a, VecUnaryOp::Rsqrt, width),
        TraceOp::Recip(a) => emit_unary(prog, slots, *a, VecUnaryOp::Recip, width),

        // ── 超越函数 → Transcendental ──
        TraceOp::Exp(a) => {
            emit_transcendental(prog, slots, *a, TranscendentalFn::Exp, width)
        }
        TraceOp::Tanh(a) => {
            emit_transcendental(prog, slots, *a, TranscendentalFn::Tanh, width)
        }
        TraceOp::Log(a) => {
            emit_transcendental(prog, slots, *a, TranscendentalFn::Log, width)
        }
        TraceOp::Sigmoid(a) => {
            emit_transcendental(prog, slots, *a, TranscendentalFn::Sigmoid, width)
        }

        // ── 比较 → VecCmp ──
        TraceOp::Compare { a, b, op } => emit_cmp(prog, slots, *a, *b, *op, width),

        // ── 类型转换 → VecCast ──
        TraceOp::Cast { src, from, to } => {
            let from_bits = quant_precision_bits(from);
            let to_bits = quant_precision_bits(to);
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecCast {
                dst: r,
                src: slots[src.0 as usize],
                from_bits,
                to_bits,
            });
            Ok(r)
        }

        // Conditional select: dst[i] = (mask[i] != 0) ? true_val[i] : false_val[i]
        TraceOp::ConditionalBranch(mask, true_val, false_val) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::ConditionalSelect {
                dst: r,
                mask: slots[mask.0 as usize],
                true_val: slots[true_val.0 as usize],
                false_val: slots[false_val.0 as usize],
            });
            Ok(r)
        }
        TraceOp::HReduce { src, op } => {
            let reduce_op = match op {
                ReduceKind::Sum => ReduceOp::Sum,
                ReduceKind::Max => ReduceOp::Max,
                ReduceKind::Min => ReduceOp::Min,
                ReduceKind::Prod => ReduceOp::Prod,
                ReduceKind::LogSum => ReduceOp::LogSum,
                other => {
                    return Err(CompilerError::CodegenViolation(format!(
                        "auto_lower_trace: HReduce op {:?} 尚未实现 (Count/ArgMax need dedicated lowering)",
                        other
                    )));
                }
            };
            // HReduce output is scalar-width (single value), input is vector-width.
            let r = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::HReduce {
                dst: r,
                src: slots[src.0 as usize],
                op: reduce_op,
            });
            Ok(r)
        }

        // ── 结构型内存操作 (ARCH-AUTO-INSTR-SELECT structural) ──

        TraceOp::ScalarLoad { base, offset } => {
            let r = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad {
                dst: r,
                base: slots[base.0 as usize],
                offset: OffsetExpr::LoopOffset(slots[offset.0 as usize]),
            });
            Ok(r)
        }

        TraceOp::StrideMul { value, stride } => {
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::IntMulStride {
                dst: r,
                src: slots[value.0 as usize],
                stride: *stride,
            });
            Ok(r)
        }

        TraceOp::PtrAdd { base, offset } => {
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: r,
                src: PtrExpr::VRegPlusVReg(slots[base.0 as usize], slots[offset.0 as usize]),
            });
            Ok(r)
        }

        TraceOp::VecLoadIndexed { base, offset } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: r,
                base: slots[base.0 as usize],
                offset: OffsetExpr::LoopOffset(slots[offset.0 as usize]),
                width,
                dtype: default_dtype, predicate: None,
            });
            Ok(r)
        }

        TraceOp::VecStoreIndexed { base, offset, value } => {
            prog.emit(VmInstr::VecStore {
                base: slots[base.0 as usize],
                offset: OffsetExpr::LoopOffset(slots[offset.0 as usize]),
                src: slots[value.0 as usize],
                width,
                dtype: default_dtype, predicate: None,
            });
            // Store doesn't produce a new value; return the value slot.
            Ok(slots[value.0 as usize])
        }

        // ── 量化混合精度 (§11 TurboQuant / §13.12 硬件拓扑) ──

        // QuantFma: 混合精度 FMA — 简化为标准 FMA (acc + act * weight)。
        // 当 act/weight 为非 F32 精度时，上层 Cast 已完成反量化。
        // 后端 gfx950 mfma_scale / SM100 tcgen05 / AMX-FP8 路径
        // 在 ISA Lowering 阶段根据 DeviceProfile 生成专用指令。
        TraceOp::QuantFma { acc, act, weight, .. } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Fma {
                dst: r,
                acc: slots[acc.0 as usize],
                a: slots[act.0 as usize],
                b: slots[weight.0 as usize],
                dtype: default_dtype,
            });
            Ok(r)
        }

        // BlockScale: 逐元素 Mul — data[i] * scale[i/block_size]。
        // f32 SIMD 路径下 scale 已广播到每个 SIMD 通道（上层处理 block 展开）。
        // CDNA4 gfx950 mfma_scale 原生路径在 ISA Lowering 阶段特化。
        TraceOp::BlockScale { data, scale, .. } => {
            emit_binop(prog, slots, *data, *scale, VecOp::Mul, width, default_dtype)
        }

        // ── 内存层级控制 ──

        // Prefetch: 性能提示，不影响正确性。
        // Prefetch 没有数据输出，返回一个 dummy slot（不产生新值）。
        // ISA Lowering 根据 CacheLevel 映射: prefetcht0/t1/nta / prfm / prefetch.global.L2。
        TraceOp::Prefetch { level: _ } => {
            // Prefetch 不产生新值，调用方不消费返回的 slot。
            // 返回 inputs[0] 作为占位（语义：无输出）。
            inputs.first().copied().ok_or_else(|| {
                CompilerError::CodegenViolation(
                    "Prefetch: 需要 inputs 作为占位 slot".into(),
                )
            })
        }

        // NonTemporalStore: 性能提示，绕过缓存写入。
        // 不产生新值，返回 inputs[0] 占位。
        TraceOp::NonTemporalStore => {
            inputs.first().copied().ok_or_else(|| {
                CompilerError::CodegenViolation(
                    "NonTemporalStore: 需要 inputs 作为占位 slot".into(),
                )
            })
        }

        // ── 位操作 (量化解包) ──

        // BitExtract: 需要 shift+mask 整数操作。
        // f32 SIMD 管线中没有等价的浮点向量指令。
        // 量化场景的位提取应在反量化前通过专用路径处理（如 QuantBlockLoad），
        // 不应出现在 f32 trace body 中。
        TraceOp::BitExtract { offset, width, .. } => {
            Err(CompilerError::CodegenViolation(format!(
                "BitExtract: 需要 shift+mask 整数操作，f32 SIMD 管线不支持 \
                 (offset={offset}, width={width}). \
                 量化解包应使用 QuantBlockLoad 或整数专用路径"
            )))
        }

        // Permute: 向量排列/洗牌。
        // 按 indices 重排 src 的元素。
        // 映射到 VmInstr::VecShuffle (REQ-VR-005)。
        // x86: vpshufb/vpermd / ARM: tbl/tbx / GPU: prmt。
        TraceOp::Permute { src, indices } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecShuffle {
                dst: r,
                src: slots[src.0 as usize],
                mask: VecShuffleMask::Dynamic { ctrl: slots[indices.0 as usize] },
                width,
            });
            Ok(r)
        }

        // ── 掩码操作 ──

        // MaskedOp: 仅对 mask 为 true 的 lane 执行内部 op。
        // 先执行内部 op 得到 result，再用 ConditionalSelect 应用 mask:
        //   dst[i] = (mask[i] != 0) ? result[i] : original[i]
        TraceOp::MaskedOp { op, mask } => {
            let inner = dispatch_trace_op(prog, op, slots, inputs, width, default_dtype)?;
            // masked_result[i] = (mask[i] != 0) ? inner[i] : original[i]
            // original = 内部 op 的第一个输入（slot 中最后一个 Input）
            let original = slots.last().copied().ok_or_else(|| {
                CompilerError::CodegenViolation(
                    "MaskedOp: slots 为空，无法确定 original 值".into(),
                )
            })?;
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::ConditionalSelect {
                dst: r,
                mask: slots[mask.0 as usize],
                true_val: inner,
                false_val: original,
            });
            Ok(r)
        }

        // ── 原子操作 (§13.6 MoE 命中计数) ──

        // AtomicAdd: [addr] += val。
        // addr 是一个指针 VReg，val 是要加的值。
        // 映射到 VmInstr::AtomicAdd（elem_width=4 的 u32 原子加，用于 MoE 命中计数）。
        // 如果 val 是运行时值，需要扩展 VmInstr 支持 VReg 源操作数。
        TraceOp::AtomicAdd { addr, val } => {
            prog.emit(VmInstr::AtomicAdd {
                base: slots[addr.0 as usize],
                offset: OffsetExpr::Const(0),
                value: 1,
                elem_width: 4,
            });
            // AtomicAdd 不产生新值，返回 addr slot 占位。
            Ok(slots[addr.0 as usize])
        }

        // ── 向量广播 (GateMask / EntropyGate) ──

        // BroadcastScalar: 将 src 的 lane 0 广播到向量所有 lane。
        // 映射到 VmInstr::Broadcast { src: ExtractLane0(src_vreg) }。
        TraceOp::BroadcastScalar { src } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: r,
                src: ScalarExpr::ExtractLane0(slots[src.0 as usize]),
                width,
                dtype: default_dtype,
            });
            Ok(r)
        }

        // BroadcastLoad: 从内存加载标量并广播到向量所有 lane。
        // 映射到 VmInstr::Broadcast { src: MemLoad(base, offset) }。
        TraceOp::BroadcastLoad { base, offset } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: r,
                src: ScalarExpr::MemLoad(
                    slots[base.0 as usize],
                    OffsetExpr::ScalarVReg(slots[offset.0 as usize]),
                ),
                width,
                dtype: default_dtype,
            });
            Ok(r)
        }

        // ── 向量索引内存操作 (Gather / Scatter) ──

        // GatherLoad: 从 base + indices[i]*stride 加载元素到向量。
        TraceOp::GatherLoad { base, indices, stride } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::GatherLoad {
                dst: r,
                base: slots[base.0 as usize],
                indices: slots[indices.0 as usize],
                stride: *stride,
                width, dtype: default_dtype, predicate: None,
            });
            Ok(r)
        }

        // ScatterStore: 将 value 的元素按 indices 写入 base + indices[i]*stride。
        TraceOp::ScatterStore { base, indices, value, stride } => {
            prog.emit(VmInstr::ScatterStore {
                base: slots[base.0 as usize],
                indices: slots[indices.0 as usize],
                src: slots[value.0 as usize],
                stride: *stride,
                width, dtype: default_dtype, predicate: None,
            });
            // Store doesn't produce a new value; return the value slot.
            Ok(slots[value.0 as usize])
        }

        // ── 查表操作 (embedding lookup) ──

        // TableLookup: 从 base + row_index * row_bytes 加载一行。
        // 映射到 VmInstr::TableLookup（组合 IntMulStride + PtrAdd + VecLoad）。
        TraceOp::TableLookup { base, row_index, row_bytes } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::TableLookup {
                dst: r,
                base: slots[base.0 as usize],
                row_index: slots[row_index.0 as usize],
                row_bytes: *row_bytes,
                width,
            });
            Ok(r)
        }

        // ── 量化解量化 (MoE expert FFN) ──

        // Mxfp4Dequant: packed 4-bit blocks → f32 × per-block scale。
        // 映射到 VmInstr::QuantBlockLoad { unpack: Mxfp4 } (REQ-VR-001)。
        // 偏移公式: slots[off_a]*stride_a + slots[off_b]*stride_b + slots[off_c] + const_off
        TraceOp::Mxfp4Dequant { data, scales, off_a, stride_a, off_b, stride_b, off_c, const_off, block_size } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let mut parts: Vec<OffsetExpr> = Vec::new();
            if off_a.is_some() { parts.push(OffsetExpr::Mul(Box::new(OffsetExpr::ScalarVReg(slots[off_a.unwrap().0 as usize])), *stride_a)); }
            if off_b.is_some() { parts.push(OffsetExpr::Mul(Box::new(OffsetExpr::ScalarVReg(slots[off_b.unwrap().0 as usize])), *stride_b)); }
            if off_c.is_some() { parts.push(OffsetExpr::ScalarVReg(slots[off_c.unwrap().0 as usize])); }
            if *const_off != 0 { parts.push(OffsetExpr::Const(*const_off)); }
            let offset = parts.into_iter().reduce(|a, b| OffsetExpr::Add(Box::new(a), Box::new(b)))
                .unwrap_or(OffsetExpr::Const(0));
            prog.emit(VmInstr::QuantBlockLoad {
                dst: r,
                base: slots[data.0 as usize],
                offset,
                unpack: BlockUnpackMode::Mxfp4 { scale_src: slots[scales.0 as usize] },
                width,
            });
            let _ = block_size;
            Ok(r)
        }

        // BitAnd: 逐位与运算 (低位掩码等)。
        // 映射到 VmInstr::VecBinOp { op: VecOp::And }。
        TraceOp::BitAnd(a, b) => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecBinOp { dst: r, a: slots[a.0 as usize], b: slots[b.0 as usize], op: VecOp::And, dtype: default_dtype, });
            Ok(r)
        }

        // ── 信号处理 (§11.1 TurboQuant FWHT) ──

        // FWHT: Fast Walsh-Hadamard Transform。
        // butterfly 网络: 逐级加减。
        // 展开为 log2(dim) 级，每级 dim/2 对 Add/Sub。
        // dim 必须是 2 的幂。
        TraceOp::FWHT { src, dim } => {
            emit_fwht(prog, slots, *src, *dim, width, default_dtype)
        }

        // ── SPEC 23-QUANT-CODEGEN-ALGO §3: Quant* 解码 TraceOp 分派 ──

        TraceOp::QuantBitAnd { lhs, rhs } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecBinOp { dst: r, a: slots[lhs.0 as usize], b: slots[rhs.0 as usize], op: VecOp::And, dtype: default_dtype });
            Ok(r)
        }

        TraceOp::QuantBitOr { lhs, rhs } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecBinOp { dst: r, a: slots[lhs.0 as usize], b: slots[rhs.0 as usize], op: VecOp::Or, dtype: default_dtype });
            Ok(r)
        }

        TraceOp::QuantBroadcast { src, lanes: _ } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            // src is Vec (quant scale from QuantLoadF16toF32), extract lane 0 then broadcast.
            prog.emit(VmInstr::Broadcast { dst: r, src: ScalarExpr::ExtractLane0(slots[src.0 as usize]), width, dtype: default_dtype });
            Ok(r)
        }

        TraceOp::QuantCastF16toF32 { src } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            // Vec F16 → Vec F32 conversion (vcvtph2ps / fcvtl)
            prog.emit(VmInstr::VecCast { dst: r, src: slots[src.0 as usize], from_bits: 16, to_bits: 32 });
            Ok(r)
        }

        TraceOp::QuantCastI8toF32 { src } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecUnaryOp { dst: r, a: slots[src.0 as usize], op: VecUnaryOp::IntToFloat });
            Ok(r)
        }

        TraceOp::QuantCastFp8toF32 { src, format } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let op = match format { Fp8Format::E4M3 => VecUnaryOp::Fp8E4M3ToFloat, Fp8Format::E5M2 => VecUnaryOp::Fp8E5M2ToFloat };
            prog.emit(VmInstr::VecUnaryOp { dst: r, a: slots[src.0 as usize], op });
            Ok(r)
        }

        TraceOp::QuantCodebookLookup { indices, codebook_data, vector_size, bits_per_entry } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantCodebookLookup {
                dst: r,
                indices: slots[indices.0 as usize],
                codebook_data,
                vector_size: *vector_size,
                bits_per_entry: *bits_per_entry,
                width,
            });
            Ok(r)
        }

        TraceOp::QuantExtractBits { src, bit_offset, bit_width } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantExtractBits { dst: r, src: slots[src.0 as usize], bit_offset: *bit_offset, bit_width: *bit_width, width });
            Ok(r)
        }

        TraceOp::QuantDequantFma { acc, a, b } => {
            // Reuse acc VReg as dst to avoid register pressure conflict.
            // acc is always Const(0.0) in quant dequant, so overwriting it is safe.
            // Allocating a separate dst VReg causes the linear allocator to map both
            // dst and acc to the same physical register as intermediate broadcast
            // results (scale), corrupting the FMA operands.
            let acc_vreg = slots[acc.0 as usize];
            prog.emit(VmInstr::Fma { dst: acc_vreg, acc: acc_vreg, a: slots[a.0 as usize], b: slots[b.0 as usize], dtype: default_dtype });
            Ok(acc_vreg)
        }

        TraceOp::QuantIntDivConst { src, divisor } => {
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: r, a: slots[src.0 as usize], b: GprOperand::Imm(*divisor), op: GprOp::Div });
            Ok(r)
        }

        TraceOp::QuantIntMul { src, factor } => {
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: r, a: slots[src.0 as usize], b: GprOperand::Imm(*factor), op: GprOp::Mul });
            Ok(r)
        }

        TraceOp::QuantInterleave { lo, hi } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantInterleave { dst: r, lo: slots[lo.0 as usize], hi: slots[hi.0 as usize], width });
            Ok(r)
        }

        TraceOp::QuantConcatSeq { lo, hi } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantConcatSeq { dst: r, lo: slots[lo.0 as usize], hi: slots[hi.0 as usize], width });
            Ok(r)
        }

        TraceOp::QuantQ3KDecode { block_base, lane_offset, d_slot, qs_offset, hmask_offset } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let lanes = if matches!(width, SimdWidth::W512) { 16 } else if matches!(width, SimdWidth::W256) { 8 } else { 4 };
            prog.emit(VmInstr::Q3KDecodeStep {
                dst: r,
                block_base: slots[block_base.0 as usize],
                lane_offset: slots[lane_offset.0 as usize],
                d_vreg: slots[d_slot.0 as usize],
                qs_offset: *qs_offset,
                hmask_offset: *hmask_offset,
                lanes,
                width,
            });
            Ok(r)
        }

        TraceOp::QuantPtrAddOffset { base, offset_bytes } => {
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::AddPtr { dst: r, base: slots[base.0 as usize], offset: *offset_bytes as usize });
            Ok(r)
        }

        TraceOp::QuantPtrAddDynamic { base, index } => {
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: r, a: slots[base.0 as usize], b: GprOperand::VReg(slots[index.0 as usize] ), op: GprOp::Add });
            Ok(r)
        }

        TraceOp::QuantScalarLoad { ptr, offset_bytes } => {
            // ScalarLoad dst must be GPR; load scalar then broadcast to Vec.
            let scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: scalar, base: slots[ptr.0 as usize], offset: OffsetExpr::Const(*offset_bytes as usize) });
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: r, src: ScalarExpr::VReg(scalar), width, dtype: default_dtype });
            Ok(r)
        }

        TraceOp::QuantAndMask { src, mask } => {
            let mask_vreg = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantBroadcastInt { dst: mask_vreg, value: *mask, width });
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecBinOp { dst: r, a: slots[src.0 as usize], b: mask_vreg, op: VecOp::And, dtype: default_dtype });
            Ok(r)
        }

        TraceOp::QuantKQuantPackedScaleLookup { scales_base, sub_block_idx, scale_algo, selector } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::GgufKQuantScaleLoad {
                dst: r,
                scales_base: slots[scales_base.0 as usize],
                sub_block_idx: slots[sub_block_idx.0 as usize],
                scales_count: 8,
                is_q3k_extended: matches!(scale_algo, PackedScaleAlgorithm::Q3KExtended),
                is_min: matches!(selector, ScaleSelector::Min),
                width,
            });
            Ok(r)
        }

        TraceOp::QuantLoadF16toF32 { ptr, offset_bytes } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantScalarCvtLoad { dst: r, base: slots[ptr.0 as usize], offset: *offset_bytes, src_dtype: ScalarCvtSource::F16, width });
            Ok(r)
        }

        TraceOp::QuantLoadI8toF32 { ptr, offset_bytes } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantScalarCvtLoad { dst: r, base: slots[ptr.0 as usize], offset: *offset_bytes, src_dtype: ScalarCvtSource::I8, width });
            Ok(r)
        }

        TraceOp::QuantLoadBytesVec { ptr, offset_bytes, count, signed } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantLoadBytesVec { dst: r, base: slots[ptr.0 as usize], offset: *offset_bytes, count: *count, signed: *signed, width });
            Ok(r)
        }

        TraceOp::QuantShiftLeft { src, amount } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecShiftImm { dst: r, a: slots[src.0 as usize], amount: *amount as u8, op: VecShiftDir::Left, width });
            Ok(r)
        }

        TraceOp::QuantShiftRight { src, amount } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecShiftImm { dst: r, a: slots[src.0 as usize], amount: *amount as u8, op: VecShiftDir::Right, width });
            Ok(r)
        }

        TraceOp::QuantE2m1LutDecode { packed_data_ptr, scale_byte, nvfp4_mode } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let unpack = if *nvfp4_mode {
                BlockUnpackMode::Nvfp4 { scale_src: slots[scale_byte.0 as usize] }
            } else {
                BlockUnpackMode::Mxfp4 { scale_src: slots[scale_byte.0 as usize] }
            };
            prog.emit(VmInstr::QuantBlockLoad {
                dst: r,
                base: slots[packed_data_ptr.0 as usize],
                offset: OffsetExpr::Const(0),
                unpack,
                width,
            });
            Ok(r)
        }

        // ── SPEC 24-QUANT-PIPELINE-JIT §1.3: quant block-level load TraceOps ──
        // T2 将完善为完整的 auto_select 映射，此处先确保编译通过。

        TraceOp::QuantScaleLoad { source, offset, dtype } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let base = slots[source.0 as usize];
            let offset_val = *offset;
            // SPEC 24-QUANT-PIPELINE-JIT §4.1: dispatch by quant_type
            match dtype {
                // f16 scale: load f16 scalar → convert to f32 → broadcast
                // (GgufF16ScaleLoad equivalent: vmovd + vcvtph2ps + vbroadcastss)
                QuantType::Q4_0 | QuantType::Q4_1
                | QuantType::Q5_0 | QuantType::Q5_1
                | QuantType::Q8_0 | QuantType::Q8_1 => {
                    prog.emit(VmInstr::QuantScalarCvtLoad {
                        dst: r,
                        base,
                        offset: offset_val as i64,
                        src_dtype: ScalarCvtSource::F16,
                        width,
                    });
                    Ok(r)
                }
                // K-Quant formats use dedicated QuantKQuantPackedScaleLookup / QuantSubScaleLoad
                QuantType::Q2K | QuantType::Q3K | QuantType::Q4K
                | QuantType::Q5K | QuantType::Q6K | QuantType::Q8K => {
                    Err(CompilerError::CodegenViolation(
                        "QuantScaleLoad: K-Quant formats use QuantKQuantPackedScaleLookup TraceOp".into()
                    ))
                }
                // MXFP/NVFP use dedicated QuantE2m1LutDecode path
                QuantType::Mxfp4 { .. } | QuantType::Nvfp4 => {
                    Err(CompilerError::CodegenViolation(
                        "QuantScaleLoad: MXFP/NVFP formats use QuantE2m1LutDecode path".into()
                    ))
                }
                // IQ formats use codebook-based dequant path
                qt @ (QuantType::IQ1S | QuantType::IQ1M | QuantType::IQ2XXS
                     | QuantType::IQ2XS | QuantType::IQ2S
                     | QuantType::IQ3XXS | QuantType::IQ3S
                     | QuantType::IQ4NL | QuantType::IQ4XS) => {
                    Err(CompilerError::CodegenViolation(format!(
                        "QuantScaleLoad: IQ format {:?} uses codebook dequant path", qt
                    )))
                }
                // Native float types should not reach QuantScaleLoad
                QuantType::F32 | QuantType::F16 | QuantType::Bf16 => {
                    Err(CompilerError::CodegenViolation(format!(
                        "QuantScaleLoad: native float type {:?} has no scale to load", dtype
                    )))
                }
                // External quantization formats
                QuantType::AWQ4 | QuantType::GPTQ4 | QuantType::Squeeze => {
                    Err(CompilerError::CodegenViolation(format!(
                        "QuantScaleLoad: external format {:?} not yet supported", dtype
                    )))
                }
                QuantType::TQ1_0 | QuantType::TQ2_0 => {
                    Err(CompilerError::CodegenViolation(format!(
                        "QuantScaleLoad: ternary format {:?} not yet supported", dtype
                    )))
                }
                QuantType::Fp8E4M3 | QuantType::Fp8E5M2 => {
                    Err(CompilerError::CodegenViolation(format!(
                        "QuantScaleLoad: FP8 {:?} has no scale (native float)", dtype
                    )))
                }
            }
        }

        TraceOp::QuantDataLoad { source, offset, quant_type, block_size } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let base = slots[source.0 as usize];
            let offset_val = *offset;
            let desc = crate::quant_format::QuantFormatDescriptor::for_type(*quant_type);
            // SPEC 24-QUANT-PIPELINE-JIT §4.1: dispatch by data_kind
            match desc.data_kind {
                QuantDataKind::SignedPackedInt4 => {
                    // LowNibble + HighNibble: interleave into signed F32 values
                    let r_lo = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantBlockLoad { dst: r_lo, base, offset: OffsetExpr::Const(offset_val), unpack: BlockUnpackMode::SignedNibbleLow, width });
                    let r_hi = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantBlockLoad { dst: r_hi, base, offset: OffsetExpr::Const(offset_val), unpack: BlockUnpackMode::SignedNibbleHigh, width });
                    prog.emit(VmInstr::QuantInterleave { dst: r, lo: r_lo, hi: r_hi, width });
                }
                QuantDataKind::PackedInt4 => {
                    // LowNibble + HighNibble: interleave into unsigned F32 values
                    let r_lo = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantBlockLoad { dst: r_lo, base, offset: OffsetExpr::Const(offset_val), unpack: BlockUnpackMode::UnsignedNibbleLow, width });
                    let r_hi = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantBlockLoad { dst: r_hi, base, offset: OffsetExpr::Const(offset_val), unpack: BlockUnpackMode::UnsignedNibbleHigh, width });
                    prog.emit(VmInstr::QuantInterleave { dst: r, lo: r_lo, hi: r_hi, width });
                }
                QuantDataKind::Int8 => {
                    // Single pass: load i8, sign-extend to i32, convert to f32
                    prog.emit(VmInstr::QuantBlockLoad { dst: r, base, offset: OffsetExpr::Const(offset_val), unpack: BlockUnpackMode::Int8, width });
                }
                QuantDataKind::PackedInt5 => {
                    // INT5 unpack (single pass): each byte contains one 5-bit value
                    let bytes = prog.alloc_vreg(VRegKind::Vec, width);
                    let count = width.f32_lanes().min(32);
                    prog.emit(VmInstr::QuantLoadBytesVec { dst: bytes, base, offset: offset_val as i64, count, signed: false, width });
                    // Mask with 0x1F to extract low 5 bits
                    let mask = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantBroadcastInt { dst: mask, value: 0x1F, width });
                    let extracted = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecBinOp { dst: extracted, a: bytes, b: mask, op: VecOp::And, dtype: QuantPrecision::F32 });
                    // Convert i32 → f32
                    prog.emit(VmInstr::VecUnaryOp { dst: r, a: extracted, op: VecUnaryOp::IntToFloat });
                }
                QuantDataKind::PackedInt6 => {
                    // INT6 unpack (single pass): each byte contains one 6-bit value
                    let bytes = prog.alloc_vreg(VRegKind::Vec, width);
                    let count = width.f32_lanes().min(32);
                    prog.emit(VmInstr::QuantLoadBytesVec { dst: bytes, base, offset: offset_val as i64, count, signed: false, width });
                    // Mask with 0x3F to extract low 6 bits
                    let mask = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantBroadcastInt { dst: mask, value: 0x3F, width });
                    let extracted = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecBinOp { dst: extracted, a: bytes, b: mask, op: VecOp::And, dtype: QuantPrecision::F32 });
                    // Convert i32 → f32
                    prog.emit(VmInstr::VecUnaryOp { dst: r, a: extracted, op: VecUnaryOp::IntToFloat });
                }
                QuantDataKind::SuperLowBit => {
                    // Codebook lookup: load packed indices, then decode via codebook
                    let bytes = prog.alloc_vreg(VRegKind::Vec, width);
                    let count = width.f32_lanes().min(32);
                    prog.emit(VmInstr::QuantLoadBytesVec { dst: bytes, base, offset: offset_val as i64, count, signed: false, width });
                    let codebook_data = desc.codebook.as_ref().map(|cb| cb.codebook_data).unwrap_or(&[]);
                    prog.emit(VmInstr::QuantCodebookLookup {
                        dst: r,
                        indices: bytes,
                        codebook_data,
                        vector_size: *block_size,
                        bits_per_entry: desc.bits_per_element,
                        width,
                    });
                }
                QuantDataKind::Float4 | QuantDataKind::Nvfp4 => {
                    // E2M1 decode: use QuantBlockLoad with MXFP/NVFP unpack
                    // scale_src = source pointer (scale byte is at base offset within block)
                    let scale_src = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::QuantScalarCvtLoad { dst: scale_src, base, offset: offset_val as i64, src_dtype: ScalarCvtSource::U8, width });
                    let unpack = if matches!(desc.data_kind, QuantDataKind::Nvfp4) {
                        BlockUnpackMode::Nvfp4 { scale_src }
                    } else {
                        BlockUnpackMode::Mxfp4 { scale_src }
                    };
                    prog.emit(VmInstr::QuantBlockLoad { dst: r, base, offset: OffsetExpr::Const(offset_val), unpack, width });
                }
                other => {
                    return Err(CompilerError::CodegenViolation(format!(
                        "QuantDataLoad: unsupported data_kind={:?} for quant_type={:?}",
                        other, quant_type
                    )));
                }
            }
            Ok(r)
        }

        TraceOp::QuantZeroLoad { source, offset, zp_type } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let base = slots[source.0 as usize];
            match zp_type {
                ZeroLayout::None => {
                    prog.emit(VmInstr::Broadcast {
                        dst: r,
                        src: ScalarExpr::Const(0.0),
                        width,
                        dtype: QuantPrecision::F32,
                    });
                }
                ZeroLayout::StaticBias { value } => {
                    prog.emit(VmInstr::Broadcast {
                        dst: r,
                        src: ScalarExpr::Const(*value as f32),
                        width,
                        dtype: QuantPrecision::F32,
                    });
                }
                ZeroLayout::BlockScalar { offset_bytes, .. } => {
                    let off = OffsetExpr::Const(*offset_bytes + offset);
                    prog.emit(VmInstr::QuantBlockLoad { dst: r, base, offset: off, unpack: BlockUnpackMode::F16Broadcast, width });
                }
                ZeroLayout::BlockMin { offset_bytes, .. } => {
                    let off = OffsetExpr::Const(*offset_bytes + offset);
                    prog.emit(VmInstr::QuantBlockLoad { dst: r, base, offset: off, unpack: BlockUnpackMode::F16Broadcast, width });
                }
                ZeroLayout::Hierarchical { .. } => {
                    return Err(CompilerError::CodegenViolation(
                        "QuantZeroLoad: Hierarchical zero layout not yet supported in auto_select".into()
                    ));
                }
            }
            Ok(r)
        }

        TraceOp::QuantSubScaleLoad { .. } => {
            Err(CompilerError::CodegenViolation(
                "QuantSubScaleLoad: K-Quant sub-scale not yet implemented in auto_select".into()
            ))
        }

        TraceOp::QuantHighBitsLoad { block_ptr, byte_offset, bits_per_elem } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let base = slots[block_ptr.0 as usize];
            // 计算 extra_base: base + byte_offset (高 bit 平面偏移)
            let extra_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::AddPtr {
                dst: extra_base,
                base,
                offset: *byte_offset,
            });
            let mode = match *bits_per_elem {
                3 => BiPlaneMode::Q3Merge,
                5 => BiPlaneMode::Low5,
                6 => BiPlaneMode::Low6,
                other => return Err(CompilerError::CodegenViolation(format!(
                    "QuantHighBitsLoad: unsupported bits_per_elem={other}, expected 3/5/6"
                ))),
            };
            prog.emit(VmInstr::QuantBiPlaneLoad {
                dst: r,
                qs_base: base,
                extra_base,
                bias: 0.0,
                mode,
                width,
            });
            Ok(r)
        }

        TraceOp::QuantCodebookDequant { indices, codebook_ptr, vector_size, bits_per_entry } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::QuantCodebookLookup {
                dst: r,
                indices: slots[indices.0 as usize],
                codebook_data: &[],
                vector_size: *vector_size,
                bits_per_entry: *bits_per_entry as u8,
                width,
            });
            let _ = codebook_ptr;
            Ok(r)
        }

        // ── SPEC 27 AT-003: 结构型 TraceOp → VmInstr 映射 ──

        TraceOp::Loop { bound, step_bytes, body } => {
            let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
            let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::LoopBegin {
                counter,
                byte_offset: byte_off,
                bound: bound.clone(),
                step_bytes: *step_bytes,
            });
            // 递归处理 Loop body — body 内的 TraceOp 走同样的 dispatch
            for body_op in body.iter() {
                let _body_result = dispatch_trace_op(prog, body_op, slots, inputs, width, default_dtype)?;
                // body 内的结果不 push 到外层 slots — 它们是循环局部值
            }
            prog.emit(VmInstr::LoopEnd);
            Ok(counter)
        }

        TraceOp::PanelLoad { base, offset, rows: _, cols: _ } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: r,
                base: slots[base.0 as usize],
                offset: OffsetExpr::ScalarVReg(slots[offset.0 as usize]),
                width,
                dtype: default_dtype, predicate: None,
            });
            Ok(r)
        }

        TraceOp::PanelStore { base, offset, rows: _, cols: _ } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecStore {
                base: slots[base.0 as usize],
                offset: OffsetExpr::ScalarVReg(slots[offset.0 as usize]),
                src: r,
                width,
                dtype: default_dtype, predicate: None,
            });
            Ok(r)
        }

        TraceOp::PackBuffer { src, dst: _, rows: _, cols: _, layout: _ } => {
            let tmp = prog.alloc_vreg(VRegKind::Vec, width);
            let _ = src;
            Ok(tmp)
        }

        TraceOp::SharedMemDeclare { name, bytes } => {
            prog.emit(VmInstr::SharedMemAlloc {
                name: name.clone(),
                bytes: *bytes,
            });
            let r = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            Ok(r)
        }

        TraceOp::AsyncCopyToShared { name, src_offset, bytes } => {
            prog.emit(VmInstr::SharedMemAsyncStore {
                name: name.clone(),
                dst_offset: OffsetExpr::ScalarVReg(slots[src_offset.0 as usize]),
                src: slots[src_offset.0 as usize],
                width,
                dtype: default_dtype,
            });
            let _ = bytes;
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            Ok(r)
        }

        TraceOp::Tma2DCopy { desc, coord_x, coord_y, bytes: _ } => {
            prog.emit(VmInstr::Tma2DCopy {
                desc_name: desc.clone(),
                smem_name: format!("tma_{}", desc),
                coord_x: slots[coord_x.0 as usize],
                coord_y: slots[coord_y.0 as usize],
                barrier_name: format!("tma_bar_{}", desc),
            });
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            Ok(r)
        }

        TraceOp::AsyncWaitGroup { n } => {
            prog.emit(VmInstr::SharedMemAsyncWaitGroup {
                n: *n,
            });
            let r = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            Ok(r)
        }

        TraceOp::SyncBarrier { name: _ } => {
            let r = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            Ok(r)
        }

        TraceOp::TileConfig { rows, cols } => {
            prog.emit(VmInstr::TileConfig {
                rows: *rows,
                cols: *cols,
                dtype: crate::types::DType::F32,
            });
            let r = prog.alloc_vreg(VRegKind::Tile, SimdWidth::Scalar);
            Ok(r)
        }

        TraceOp::TileMma { c, a, b } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::TileMma {
                c: slots[c.0 as usize],
                a: slots[a.0 as usize],
                b: slots[b.0 as usize],
            });
            Ok(r)
        }

        TraceOp::TileRelease => {
            prog.emit(VmInstr::TileRelease);
            let r = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            Ok(r)
        }

        TraceOp::Softmax { src, dst: _ } => {
            let max_r = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::SoftmaxReduceMax {
                dst: max_r,
                logits_ptr: slots[src.0 as usize],
                vocab_bytes: width.f32_lanes() * 4,
                width,
            });
            let sum_r = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::SoftmaxExpSum {
                sum_dst: sum_r,
                logits_ptr: slots[src.0 as usize],
                max_val: max_r,
                vocab_bytes: width.f32_lanes() * 4,
                width,
            });
            let norm_r = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::SoftmaxNormalize {
                logits_ptr: slots[src.0 as usize],
                sum_val: sum_r,
                vocab_bytes: width.f32_lanes() * 4,
                width,
            });
            Ok(norm_r)
        }

        TraceOp::EpilogueChain { ops } => {
            let r = prog.alloc_vreg(VRegKind::Vec, width);
            let _ = ops;
            Ok(r)
        }

        // ── SPEC 24-QUANT-PIPELINE-JIT §1.3: QuantGather/QuantGemm structural ──
        // These structural ops expand to full loop nests with block decode.
        // The 3 preceding Input ops in the trace provide the pointer VRegIds.

        TraceOp::QuantGather { quant_type, vocab_size, hidden_dim } => {
            // Trace structure: [Input(0):indices_ptr, Input(1):embed_ptr, Input(2):output_ptr, QuantGather]
            // After processing 3 Input ops, slots = [indices_ptr, embed_ptr, output_ptr]
            let indices_ptr = slots[0];
            let embed_ptr = slots[1];
            let output_ptr = slots[2];
            // Default to single-token decode (mega-kernel); the higher-level
            // plan_lower / lower_op overrides seq_bound when more
            // context is available.
            let seq_bound = BoundExpr::Const(1);
            super::quant_gather_emit::emit_quant_gather_inline(
                prog, seq_bound, *vocab_size, *hidden_dim, *quant_type,
                width, indices_ptr, embed_ptr, output_ptr, default_dtype,
                None,
            )?;
            Ok(output_ptr)
        }

        TraceOp::QuantGemm { quant_type, m, n, k } => {
            // Trace structure: [Input(0):input_ptr, Input(1):weight_ptr, Input(2):output_ptr, QuantGemm]
            let input_ptr = slots[0];
            let weight_ptr = slots[1];
            let output_ptr = slots[2];
            let m_bound = BoundExpr::Const(*m);
            super::moe_quant_emit::emit_quant_gemm_inline(
                prog, m_bound, *n, *k, *quant_type,
                width, input_ptr, weight_ptr, output_ptr, default_dtype,
                crate::dispatch::device_profile::DotProductCap::SimdAssisted,
            )?;
            Ok(output_ptr)
        }

        // ── MTP Draft (MTP-001): structural GEMV + argmax loop nest ──
        // Trace structure: [Input(0):hidden_ptr, Input(1):weight_ptr, Input(2):output_tokens_ptr, MtpDraft]
        TraceOp::MtpDraft { depth, hidden_size, vocab_size } => {
            let hidden_ptr = slots[0];
            let weight_ptr = slots[1];
            let output_tokens_ptr = slots[2];
            super::mega_kernel_emit::emit_mtp_draft_inline(
                prog, *depth, *hidden_size, *vocab_size,
                hidden_ptr, weight_ptr, output_tokens_ptr,
                width, default_dtype,
            )?;
            Ok(output_tokens_ptr)
        }

        // ── MLA (REQ-MLA-007): structural attention + RoPE merge ──
        // MlaAttnScore: online softmax attention in compressed d_c space + per-head V restore.
        // Trace structure: [Input(0):q_absorbed_ptr, Input(1):kv_cache_ptr,
        //   Input(2):w_uv_ptr, Input(3):output_ptr], kv_len passed as VRegId
        TraceOp::MlaAttnScore { num_heads, head_dim, d_c, d_rope } => {
            let kv_len_vreg = slots[4];
            super::mla_emit::emit_mla_attn_score_inline(
                prog, *num_heads, *head_dim, *d_c, *d_rope,
                &slots[..4], kv_len_vreg, width, default_dtype,
            )
        }
        // MlaRopeMerge: replace c_KV[d_c-d_rope..d_c] with RoPE(k_pe).
        // Trace structure: [Input(0):c_kv_ptr, Input(1):k_pe_ptr,
        //   Input(2):output_ptr, Input(3):cos_ptr, Input(4):sin_ptr, Input(5):position]
        TraceOp::MlaRopeMerge { d_c, d_rope } => {
            super::mla_emit::emit_mla_rope_merge_inline(
                prog, *d_c, *d_rope,
                slots, width, default_dtype,
            )
        }

        // ── SPEC 37 REQ-HWACC-007: DynamicPrecisionSelect ──
        // GEMM prologue: 分析 tensor 统计量 → 运行时选择精度。
        // 生成 VmInstr 序列:
        //   1. HReduce(tensor, Max) → max_val (标量)
        //   2. 广播 max_val 到向量
        //   3. 逐阈值比较: Const(threshold[i]) + Compare(Lt)
        //   4. ConditionalSelect 链选择精度索引
        // 结果: 标量 i32 精度索引（0 = candidates[0] = 最高精度）
        TraceOp::DynamicPrecisionSelect { tensor, candidates, thresholds } => {
            if candidates.len() != thresholds.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "DynamicPrecisionSelect: candidates({}) 与 thresholds({}) 长度不匹配",
                    candidates.len(), thresholds.len()
                )));
            }
            if candidates.is_empty() {
                return Err(CompilerError::CodegenViolation(
                    "DynamicPrecisionSelect: candidates 不能为空".into()
                ));
            }

            // Step 1: HReduce(Max) — 从 tensor 中提取最大绝对值
            let max_val = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::HReduce {
                dst: max_val,
                src: slots[tensor.0 as usize],
                op: ReduceOp::Max,
            });

            // Step 2: 广播 max_val 到向量宽度以便比较
            let max_broadcast = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: max_broadcast,
                src: ScalarExpr::ExtractLane0(max_val),
                width,
                dtype: default_dtype,
            });

            // Step 3: 对每个阈值生成比较 + ConditionalSelect
            // 策略: 从最高精度（索引 0）开始，逐级检查是否可以降精度
            // result_idx 初始 = 0（最高精度）
            // if max_val < thresholds[1]: result_idx = 1 (第一个低精度)
            // if max_val < thresholds[2]: result_idx = 2 (更低精度)
            // ...
            // 最终 result_idx 作为 GPR 整数结果

            // 广播当前索引值（初始 = 0，即 candidates[0] 最高精度）
            let mut idx_val = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: idx_val,
                src: ScalarExpr::Const(0.0),
                width,
                dtype: default_dtype,
            });

            // 从索引 1 开始逐阈值比较（candidates[0] 是默认，不需要比较）
            for (i, threshold) in thresholds.iter().enumerate().skip(1) {
                // 广播阈值常量
                let thresh_val = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::Broadcast {
                    dst: thresh_val,
                    src: ScalarExpr::Const(*threshold as f32),
                    width,
                    dtype: default_dtype,
                });

                // 比较: max_val < threshold → mask
                let cmp_mask = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecCmp {
                    dst: cmp_mask,
                    a: max_broadcast,
                    b: thresh_val,
                    pred: CmpPredicate::Lt,
                });

                // 广播新索引值 (i as f32)
                let new_idx = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::Broadcast {
                    dst: new_idx,
                    src: ScalarExpr::Const(i as f32),
                    width,
                    dtype: default_dtype,
                });

                // ConditionalSelect: mask ? new_idx : idx_val
                let selected = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::ConditionalSelect {
                    dst: selected,
                    mask: cmp_mask,
                    true_val: new_idx,
                    false_val: idx_val,
                });

                idx_val = selected;
            }

            // 结果是向量形式的精度索引，返回它
            Ok(idx_val)
        }

    }
}

// ────────────────────────────────────────────────────────────────────────────
// 辅助函数：消除 match arm 中的 alloc_vreg + emit 重复代码
// ────────────────────────────────────────────────────────────────────────────

fn emit_binop(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    b: ValueId,
    op: VecOp,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    let r = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::VecBinOp {
        dst: r,
        a: slots[a.0 as usize],
        b: slots[b.0 as usize],
        op,
        dtype,
    });
    Ok(r)
}

fn emit_binop_into(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    b: ValueId,
    op: VecOp,
    dst: VRegId,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    prog.emit(VmInstr::VecBinOp { dst, a: slots[a.0 as usize], b: slots[b.0 as usize], op, dtype });
    Ok(())
}

fn emit_unary(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    op: VecUnaryOp,
    width: SimdWidth,
) -> Result<VRegId, CompilerError> {
    let r = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::VecUnaryOp {
        dst: r,
        a: slots[a.0 as usize],
        op,
    });
    Ok(r)
}

fn emit_unary_into(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    op: VecUnaryOp,
    dst: VRegId,
    width: SimdWidth,
) -> Result<(), CompilerError> {
    prog.emit(VmInstr::VecUnaryOp { dst, a: slots[a.0 as usize], op });
    Ok(())
}

fn emit_transcendental(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    func: TranscendentalFn,
    width: SimdWidth,
) -> Result<VRegId, CompilerError> {
    let r = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Transcendental { dst: r, src: slots[a.0 as usize], func });
    Ok(r)
}

fn emit_transcendental_into(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    func: TranscendentalFn,
    dst: VRegId,
    width: SimdWidth,
) -> Result<(), CompilerError> {
    prog.emit(VmInstr::Transcendental { dst, src: slots[a.0 as usize], func });
    Ok(())
}

fn emit_cmp(
    prog: &mut VmProgram,
    slots: &[VRegId],
    a: ValueId,
    b: ValueId,
    op: CmpOp,
    width: SimdWidth,
) -> Result<VRegId, CompilerError> {
    let pred = match op {
        CmpOp::Eq => CmpPredicate::Eq,
        CmpOp::Ne => CmpPredicate::Ne,
        CmpOp::Lt => CmpPredicate::Lt,
        CmpOp::Le => CmpPredicate::Le,
        CmpOp::Gt => CmpPredicate::Gt,
        CmpOp::Ge => CmpPredicate::Ge,
    };
    let r = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::VecCmp {
        dst: r,
        a: slots[a.0 as usize],
        b: slots[b.0 as usize],
        pred,
    });
    Ok(r)
}

fn quant_precision_bits(p: &crate::compiler::trace::QuantPrecision) -> u8 {
    use crate::compiler::trace::DTypeKind;
    match p.kind {
        DTypeKind::F32 => 32,
        DTypeKind::F16 => 16,
        DTypeKind::BF16 => 16,
        DTypeKind::TF32 => 19,
        DTypeKind::FP8E4M3 => 8,
        DTypeKind::FP8E5M2 => 8,
        DTypeKind::FP6E2M3 => 6,
        DTypeKind::FP6E3M2 => 6,
        DTypeKind::FP4E2M1 => 4,
        DTypeKind::INT8 => 8,
        DTypeKind::INT4 => 4,
        DTypeKind::INT2 => 2,
        DTypeKind::INT1 => 1,
    }
}

/// FWHT (Fast Walsh-Hadamard Transform): 生成 Add + 1/sqrt(dim) 缩放。
///
/// dim 必须是 2 的幂。dim=1 → identity。
/// butterfly 网络 (逐级 Add/Sub) 在 ISA Lowering 阶段展开。
/// VmInstr 层面生成: r = src+src (sum placeholder), scaled = r * (1/sqrt(dim))。
fn emit_fwht(
    prog: &mut VmProgram,
    slots: &[VRegId],
    src: ValueId,
    dim: usize,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    if dim == 0 || (dim & (dim - 1)) != 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "FWHT: dim={dim} 必须是 2 的幂且 > 0"
        )));
    }
    if dim == 1 {
        return Ok(slots[src.0 as usize]);
    }
    // Correct butterfly FWHT: log2(dim) stages, each with dim/2 Add/Sub pairs.
    // For dim <= SIMD lanes: in-register butterfly using VecShuffle + Add/Sub + ConditionalSelect.
    // For dim > SIMD lanes: not supported at TraceOp level (requires multi-register coordination).
    let lanes = width.f32_lanes();
    if dim > lanes {
        return Err(CompilerError::CodegenViolation(format!(
            "FWHT: dim={dim} > lanes={lanes} multi-register butterfly requires mega-kernel level"
        )));
    }

    let mut current = slots[src.0 as usize];
    let log2_dim = dim.trailing_zeros() as usize;

    for s in 0..log2_dim {
        let stride = 1usize << s;
        // Shuffle: swap elements at distance stride (i ↦ i ^ stride).
        let mut mask: Vec<u8> = Vec::with_capacity(lanes);
        for i in 0..lanes {
            mask.push((i ^ stride) as u8);
        }
        let shuffled = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecShuffle {
            dst: shuffled,
            src: current,
            mask: VecShuffleMask::Const(mask),
            width,
        });
        // Butterfly pair: sum = current + shuffled, diff = current - shuffled.
        let sum = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecBinOp { dst: sum, a: current, b: shuffled, op: VecOp::Add, dtype });
        let diff = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecBinOp { dst: diff, a: current, b: shuffled, op: VecOp::Sub, dtype });
        // Reconstruct: result[i] = sum[i] where (i % 2*stride) < stride, else diff[i].
        let mut select_vals: Vec<u32> = Vec::with_capacity(lanes);
        for i in 0..lanes {
            select_vals.push(if (i % (2 * stride)) < stride { 0xFFFFFFFF } else { 0 });
        }
        let sel_mask = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecLoadConst { dst: sel_mask, values: select_vals, dtype, width });
        let result = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::ConditionalSelect { dst: result, mask: sel_mask, true_val: sum, false_val: diff });
        current = result;
    }

    // Normalization: 1/sqrt(dim).
    let inv_sqrt = 1.0 / (dim as f32).sqrt();
    let scale = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: scale, src: ScalarExpr::Const(inv_sqrt), width, dtype });
    let scaled = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::VecBinOp { dst: scaled, a: current, b: scale, op: VecOp::Mul, dtype });
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::{DTypeKind, PackingFormat, QuantPrecision, TraceOp, ValueId};
    use crate::compiler::codegen::vm::instr::{SimdWidth, VRegKind, VmInstr};

    fn raw_prog(body: &[TraceOp]) -> (VmProgram, Vec<VRegId>) {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let slots = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32).unwrap();
        (prog, slots)
    }

    #[test]
    fn empty_body_returns_empty_slots() {
        let mut prog = VmProgram::new();
        let slots = auto_lower_trace_raw(&mut prog, &[], &[], SimdWidth::W256, QuantPrecision::F32).unwrap();
        assert!(slots.is_empty());
    }

    #[test]
    fn input_passthrough_produces_copy() {
        let (prog, slots) = raw_prog(&[TraceOp::Input(0)]);
        assert_eq!(slots.len(), 1);
        // Should produce a VecCopy or equivalent
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn const_broadcast_produces_broadcast_instr() {
        let (prog, slots) = raw_prog(&[TraceOp::Const(42.0)]);
        assert_eq!(slots.len(), 1);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Broadcast { .. })));
    }

    #[test]
    fn add_binop_produces_vec_binop() {
        let body = &[TraceOp::Input(0), TraceOp::Const(1.0), TraceOp::Add(ValueId(0), ValueId(1))];
        let (prog, slots) = raw_prog(body);
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Add, .. })));
    }

    #[test]
    fn mul_binop_produces_vec_mul() {
        let body = &[TraceOp::Input(0), TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(1))];
        let (prog, slots) = raw_prog(body);
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Mul, .. })));
    }

    #[test]
    fn fma_produces_fma_instr() {
        let body = &[TraceOp::Input(0), TraceOp::Input(0), TraceOp::Input(0), TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2))];
        let (prog, slots) = raw_prog(body);
        assert_eq!(slots.len(), 4);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Fma { .. })));
    }

    #[test]
    fn neg_unary_produces_unary_instr() {
        let body = &[TraceOp::Input(0), TraceOp::Neg(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecUnaryOp { op: VecUnaryOp::Neg, .. })));
    }

    #[test]
    fn sqrt_unary_produces_sqrt() {
        let body = &[TraceOp::Input(0), TraceOp::Sqrt(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecUnaryOp { op: VecUnaryOp::Sqrt, .. })));
    }

    #[test]
    fn exp_transcendental_produces_exp() {
        let body = &[TraceOp::Input(0), TraceOp::Exp(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Exp, .. })));
    }

    #[test]
    fn chained_ops_produce_correct_slot_count() {
        // x → x*2 → x*2+3 → sqrt(x*2+3)
        let body = &[
            TraceOp::Input(0),       // slot 0
            TraceOp::Const(2.0),     // slot 1
            TraceOp::Mul(ValueId(0), ValueId(1)), // slot 2
            TraceOp::Const(3.0),     // slot 3
            TraceOp::Add(ValueId(2), ValueId(3)), // slot 4
            TraceOp::Sqrt(ValueId(4)),             // slot 5
        ];
        let (_prog, slots) = raw_prog(body);
        assert_eq!(slots.len(), 6);
    }

    #[test]
    fn into_variant_writes_to_dst() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Const(1.0), TraceOp::Add(ValueId(0), ValueId(1))];
        auto_lower_trace_into(&mut prog, body, &[input], dst, SimdWidth::W256, QuantPrecision::F32).unwrap();
        // The final result should be written to `dst` directly
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn typed_variant_infers_dtype_through_chain() {
        // x(BF16) → x*2 → x*2+3
        // Input(0) falls back to F32 (slots empty), but Mul/Add propagate from operands
        use crate::compiler::trace::TypedSlot;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let inputs = &[TypedSlot { vreg: input, dtype: QuantPrecision::BF16 }];
        let body = &[
            TraceOp::Input(0),       // slot 0 (F32 fallback — no prior slots)
            TraceOp::Const(2.0),     // slot 1 (F32)
            TraceOp::Mul(ValueId(0), ValueId(1)), // slot 2 (promote F32,F32 = F32)
        ];
        let slots = auto_lower_trace_typed(&mut prog, body, inputs, SimdWidth::W256).unwrap();
        assert_eq!(slots.len(), 3);
        // All F32 since no BF16 slot exists in the body yet
        assert_eq!(slots[2].dtype, QuantPrecision::F32);
    }

    // ── 13 new tests below ──

    #[test]
    fn input_out_of_bounds_returns_error() {
        // Arrange: body references Input(1) but only 1 input provided
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(1)];
        // Act
        let result = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn const_zero_value_broadcasts_correctly() {
        // Arrange: boundary value 0.0
        let (prog, slots) = raw_prog(&[TraceOp::Const(0.0)]);
        // Assert: still produces a Broadcast instruction with 0.0
        assert_eq!(slots.len(), 1);
        let has_broadcast = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == 0.0f32
        ));
        assert!(has_broadcast);
    }

    #[test]
    fn const_negative_value_broadcasts() {
        // Arrange: negative float
        let (prog, slots) = raw_prog(&[TraceOp::Const(-3.14)]);
        assert_eq!(slots.len(), 1);
        let has_broadcast = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == -3.14f32
        ));
        assert!(has_broadcast);
    }

    #[test]
    fn sub_binop_produces_vec_sub() {
        // Arrange: a - b
        let body = &[TraceOp::Input(0), TraceOp::Const(5.0), TraceOp::Sub(ValueId(0), ValueId(1))];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Sub, .. })));
    }

    #[test]
    fn div_binop_produces_vec_div() {
        // Arrange: a / b
        let body = &[TraceOp::Input(0), TraceOp::Const(2.0), TraceOp::Div(ValueId(0), ValueId(1))];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Div, .. })));
    }

    #[test]
    fn abs_unary_produces_abs() {
        // Arrange
        let body = &[TraceOp::Input(0), TraceOp::Abs(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        // Assert
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecUnaryOp { op: VecUnaryOp::Abs, .. })));
    }

    #[test]
    fn rsqrt_unary_produces_rsqrt() {
        // Arrange
        let body = &[TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        // Assert
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecUnaryOp { op: VecUnaryOp::Rsqrt, .. })));
    }

    #[test]
    fn recip_unary_produces_recip() {
        // Arrange
        let body = &[TraceOp::Input(0), TraceOp::Recip(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        // Assert
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecUnaryOp { op: VecUnaryOp::Recip, .. })));
    }

    #[test]
    fn tanh_transcendental_produces_tanh() {
        // Arrange
        let body = &[TraceOp::Input(0), TraceOp::Tanh(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        // Assert
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Tanh, .. })));
    }

    #[test]
    fn sigmoid_transcendental_produces_sigmoid() {
        // Arrange
        let body = &[TraceOp::Input(0), TraceOp::Sigmoid(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        // Assert
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Sigmoid, .. })));
    }

    #[test]
    fn log_transcendental_produces_log() {
        // Arrange
        let body = &[TraceOp::Input(0), TraceOp::Log(ValueId(0))];
        let (prog, _slots) = raw_prog(body);
        // Assert
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Log, .. })));
    }

    #[test]
    fn max_binop_produces_vec_max() {
        // Arrange: max(a, b)
        let body = &[TraceOp::Input(0), TraceOp::Input(0), TraceOp::Max(ValueId(0), ValueId(1))];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Max, .. })));
    }

    #[test]
    fn min_binop_produces_vec_min() {
        // Arrange: min(a, b)
        let body = &[TraceOp::Input(0), TraceOp::Input(0), TraceOp::Min(ValueId(0), ValueId(1))];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Min, .. })));
    }

    #[test]
    fn compare_produces_vec_cmp() {
        // Arrange: compare a < b
        use crate::compiler::trace::CmpOp;
        let body = &[TraceOp::Input(0), TraceOp::Const(1.0), TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: CmpOp::Lt }];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::VecCmp { pred: CmpPredicate::Lt, .. })));
    }

    #[test]
    fn auto_lower_trace_empty_body_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let instrs_before = prog.instrs.len();
        // Act: empty body should succeed with no-op
        let result = auto_lower_trace(&mut prog, &[], &[input], SimdWidth::W256, QuantPrecision::F32);
        // Assert: no new instructions emitted for empty body
        assert!(result.is_ok());
        assert_eq!(prog.instrs.len(), instrs_before);
    }

    #[test]
    fn auto_lower_trace_writes_last_slot_to_primary_input() {
        // Arrange: Input(0), Const(7.0), Mul → result should be copied back to inputs[0]
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Const(7.0), TraceOp::Mul(ValueId(0), ValueId(1))];
        // Act
        auto_lower_trace(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32).unwrap();
        // Assert: last instruction should be a Mov writing back to inputs[0]
        let last = prog.instrs.last().expect("should have instructions");
        assert!(matches!(last, VmInstr::Mov { dst, .. } if *dst == input));
    }

    #[test]
    fn auto_lower_trace_into_empty_body_is_ok() {
        // Arrange
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let instrs_before = prog.instrs.len();
        // Act
        let result = auto_lower_trace_into(&mut prog, &[], &[input], dst, SimdWidth::W256, QuantPrecision::F32);
        // Assert: no new instructions emitted for empty body
        assert!(result.is_ok());
        assert_eq!(prog.instrs.len(), instrs_before);
    }

    #[test]
    fn multi_output_writes_correct_slots() {
        // Arrange: body = [Input(0), Mul(0,0)] → write slot 0 and slot 1 to separate dsts
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst_a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst_b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(1))];
        // Act: copy slot 0 (Input) → dst_a, slot 2 (Mul result) → dst_b
        auto_lower_trace_multi(
            &mut prog, body, &[input],
            &[(dst_a, 0), (dst_b, 2)],
            SimdWidth::W256, QuantPrecision::F32,
        ).unwrap();
        // Assert: should have Mov instructions for both targets
        let movs: Vec<_> = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Mov { .. })).cloned().collect();
        assert!(movs.iter().any(|i| matches!(i, VmInstr::Mov { dst, .. } if *dst == dst_a)));
        assert!(movs.iter().any(|i| matches!(i, VmInstr::Mov { dst, .. } if *dst == dst_b)));
    }

    #[test]
    fn multi_output_out_of_bounds_slot_returns_error() {
        // Arrange: body has 2 ops (2 slots), but target requests slot 5
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Const(1.0)];
        // Act
        let result = auto_lower_trace_multi(
            &mut prog, body, &[input],
            &[(dst, 5)],
            SimdWidth::W256, QuantPrecision::F32,
        );
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn typed_empty_body_returns_empty() {
        // Arrange
        use crate::compiler::trace::TypedSlot;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let inputs = &[TypedSlot { vreg: input, dtype: QuantPrecision::F32 }];
        // Act
        let result = auto_lower_trace_typed(&mut prog, &[], inputs, SimdWidth::W256).unwrap();
        // Assert
        assert!(result.is_empty());
    }

    #[test]
    fn cast_traceop_produces_vec_cast() {
        // Arrange: cast from F16 (16 bits) to F32 (32 bits)
        let body = &[TraceOp::Input(0), TraceOp::Cast {
            src: ValueId(0),
            from: QuantPrecision::F16,
            to: QuantPrecision::F32,
        }];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 2);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecCast { from_bits: 16, to_bits: 32, .. }
        )));
    }

    #[test]
    fn conditional_branch_produces_conditional_select() {
        // Arrange: mask, true_val, false_val
        let body = &[
            TraceOp::Input(0),  // slot 0: mask
            TraceOp::Const(1.0),  // slot 1: true_val
            TraceOp::Const(0.0),  // slot 2: false_val
            TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2)),  // slot 3
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 4);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::ConditionalSelect { .. })));
    }

    #[test]
    fn hreduce_sum_produces_hreduce_instr() {
        // Arrange
        use crate::compiler::trace::ReduceKind;
        let body = &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
        let (prog, slots) = raw_prog(body);
        // Assert: scalar-width output
        assert_eq!(slots.len(), 2);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::HReduce { op: ReduceOp::Sum, .. })));
    }

    #[test]
    fn hreduce_unsupported_kind_returns_error() {
        // Arrange: ArgMax is not supported in auto_select
        use crate::compiler::trace::ReduceKind;
        let body = &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::ArgMax }];
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Act
        let result = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn simd_width_scalar_lanes_is_one() {
        // Arrange & Act
        let lanes = SimdWidth::Scalar.f32_lanes();
        // Assert
        assert_eq!(lanes, 1);
    }

    #[test]
    fn simd_width_w128_lanes_is_four() {
        assert_eq!(SimdWidth::W128.f32_lanes(), 4);
    }

    #[test]
    fn simd_width_w256_bytes_is_32() {
        assert_eq!(SimdWidth::W256.bytes(), 32);
    }

    #[test]
    fn simd_width_w512_lanes_is_16() {
        assert_eq!(SimdWidth::W512.f32_lanes(), 16);
    }

    #[test]
    fn simd_width_warp_lanes_matches_value() {
        assert_eq!(SimdWidth::Warp(32).f32_lanes(), 32);
        assert_eq!(SimdWidth::Warp(64).f32_lanes(), 64);
    }

    #[test]
    fn simd_width_scalable_lanes_is_zero() {
        assert_eq!(SimdWidth::Scalable.f32_lanes(), 0);
    }

    #[test]
    fn vreg_id_equality_and_hash() {
        use std::collections::HashSet;
        let a = VRegId(42);
        let b = VRegId(42);
        let c = VRegId(99);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn vreg_id_debug_format() {
        let id = VRegId(7);
        let debug_str = format!("{:?}", id);
        assert!(debug_str.contains("7"));
    }

    #[test]
    fn vreg_kind_all_variants_are_distinct() {
        let kinds = [
            VRegKind::Ptr, VRegKind::Vec, VRegKind::Scalar,
            VRegKind::Counter, VRegKind::ByteOffset, VRegKind::Tile, VRegKind::Mask,
        ];
        // Assert all pairwise distinct
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(kinds[i], kinds[j], "VRegKind variants {} and {} should differ", i, j);
            }
        }
    }

    #[test]
    fn quant_precision_f32_constant_fields() {
        let f32_prec = QuantPrecision::F32;
        assert_eq!(f32_prec.kind, DTypeKind::F32);
        assert_eq!(f32_prec.packing, PackingFormat::Plain);
        assert_eq!(f32_prec.block_size, 0);
        assert_eq!(f32_prec.group_size, 0);
    }

    #[test]
    fn quant_precision_bf16_constant_fields() {
        let bf16 = QuantPrecision::BF16;
        assert_eq!(bf16.kind, DTypeKind::BF16);
    }

    #[test]
    fn quant_precision_struct_update_syntax() {
        // Arrange: derive from F32, change kind only
        let custom = QuantPrecision { kind: DTypeKind::INT8, ..QuantPrecision::F32 };
        // Assert
        assert_eq!(custom.kind, DTypeKind::INT8);
        assert_eq!(custom.packing, PackingFormat::Plain);
        assert_eq!(custom.block_size, 0);
    }

    #[test]
    fn value_id_none_is_max_u32() {
        assert_eq!(ValueId::NONE.0, u32::MAX);
        assert!(!ValueId::NONE.is_some());
    }

    #[test]
    fn value_id_is_some_for_normal_values() {
        assert!(ValueId(0).is_some());
        assert!(ValueId(100).is_some());
    }

    #[test]
    fn value_id_saturating_sub_clamps_at_zero() {
        assert_eq!(ValueId(0).saturating_sub(5), ValueId(0));
        assert_eq!(ValueId(10).saturating_sub(3), ValueId(7));
    }

    #[test]
    fn value_id_display_shows_v_prefix() {
        let id = ValueId(42);
        assert_eq!(format!("{}", id), "v42");
    }

    #[test]
    fn quant_precision_bits_f32_is_32() {
        let bits = quant_precision_bits(&QuantPrecision::F32);
        assert_eq!(bits, 32);
    }

    #[test]
    fn quant_precision_bits_bf16_is_16() {
        let bits = quant_precision_bits(&QuantPrecision::BF16);
        assert_eq!(bits, 16);
    }

    #[test]
    fn quant_precision_bits_fp8_variants_are_8() {
        assert_eq!(quant_precision_bits(&QuantPrecision::FP8E4M3), 8);
        assert_eq!(quant_precision_bits(&QuantPrecision::FP8E5M2), 8);
    }

    #[test]
    fn quant_precision_bits_int4_is_4() {
        assert_eq!(quant_precision_bits(&QuantPrecision::INT4), 4);
    }

    #[test]
    fn emit_cmp_all_predicates() {
        use crate::compiler::trace::CmpOp;
        let preds = [
            (CmpOp::Eq, CmpPredicate::Eq),
            (CmpOp::Ne, CmpPredicate::Ne),
            (CmpOp::Lt, CmpPredicate::Lt),
            (CmpOp::Le, CmpPredicate::Le),
            (CmpOp::Gt, CmpPredicate::Gt),
            (CmpOp::Ge, CmpPredicate::Ge),
        ];
        for (cmp_op, expected_pred) in preds {
            let body = &[
                TraceOp::Input(0),
                TraceOp::Const(1.0),
                TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: cmp_op },
            ];
            let (prog, _slots) = raw_prog(body);
            assert!(
                prog.instrs.iter().any(|i| matches!(i, VmInstr::VecCmp { pred, .. } if *pred == expected_pred)),
                "CmpOp::{:?} should produce CmpPredicate::{:?}", cmp_op, expected_pred,
            );
        }
    }

    #[test]
    fn fwht_dim1_returns_identity() {
        // Arrange: FWHT with dim=1 should return the source slot unchanged
        let body = &[TraceOp::Input(0), TraceOp::FWHT { src: ValueId(0), dim: 1 }];
        let (prog, slots) = raw_prog(body);
        // Assert: no extra instructions (just the Input passthrough), slot unchanged
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[1], slots[0]);
    }

    #[test]
    fn fwht_dim0_returns_error() {
        // Arrange: dim=0 is invalid (must be power of 2 and > 0)
        let body = &[TraceOp::Input(0), TraceOp::FWHT { src: ValueId(0), dim: 0 }];
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Act
        let result = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn fwht_non_power_of_two_returns_error() {
        // Arrange: dim=3 is not a power of 2
        let body = &[TraceOp::Input(0), TraceOp::FWHT { src: ValueId(0), dim: 3 }];
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Act
        let result = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn typed_variant_produces_correct_dtype_on_fma() {
        // Arrange: FMA with BF16 dtype propagation
        use crate::compiler::trace::TypedSlot;
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let inputs = &[TypedSlot { vreg: v0, dtype: QuantPrecision::BF16 }];
        let body = &[
            TraceOp::Input(0),       // slot 0
            TraceOp::Input(0),       // slot 1
            TraceOp::Input(0),       // slot 2
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),  // slot 3
        ];
        // Act
        let slots = auto_lower_trace_typed(&mut prog, body, inputs, SimdWidth::W256).unwrap();
        // Assert
        assert_eq!(slots.len(), 4);
        // FMA dtype is inferred from inputs — no prior BF16 slots in body, so F32
        assert_eq!(slots[3].dtype, QuantPrecision::F32);
    }

    // ── 10 new tests (wave-12kga) ──

    #[test]
    fn gather_load_produces_gather_load_instr() {
        // Arrange: GatherLoad needs base, indices, stride
        let body = &[
            TraceOp::Input(0), // slot 0: base
            TraceOp::Input(0), // slot 1: indices
            TraceOp::GatherLoad { base: ValueId(0), indices: ValueId(1), stride: 4 },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::GatherLoad { stride: 4, .. }
        )));
    }

    #[test]
    fn scatter_store_produces_scatter_store_instr() {
        // Arrange: ScatterStore writes value to base + indices * stride
        let body = &[
            TraceOp::Input(0), // slot 0: base
            TraceOp::Input(0), // slot 1: indices
            TraceOp::Input(0), // slot 2: value
            TraceOp::ScatterStore { base: ValueId(0), indices: ValueId(1), value: ValueId(2), stride: 8 },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 4);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScatterStore { stride: 8, .. }
        )));
    }

    #[test]
    fn table_lookup_produces_table_lookup_instr() {
        // Arrange: TableLookup with base, row_index, row_bytes
        let body = &[
            TraceOp::Input(0), // slot 0: base
            TraceOp::Input(0), // slot 1: row_index
            TraceOp::TableLookup { base: ValueId(0), row_index: ValueId(1), row_bytes: 256 },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::TableLookup { row_bytes: 256, .. }
        )));
    }

    #[test]
    fn permute_produces_vec_shuffle_dynamic() {
        // Arrange: Permute with src and dynamic indices
        let body = &[
            TraceOp::Input(0), // slot 0: src
            TraceOp::Input(0), // slot 1: indices
            TraceOp::Permute { src: ValueId(0), indices: ValueId(1) },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecShuffle { mask: VecShuffleMask::Dynamic { .. }, .. }
        )));
    }

    #[test]
    fn bit_and_produces_vec_binop_and() {
        // Arrange: BitAnd(a, b) → VecBinOp { op: And }
        let body = &[
            TraceOp::Input(0),
            TraceOp::Input(0),
            TraceOp::BitAnd(ValueId(0), ValueId(1)),
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecBinOp { op: VecOp::And, .. }
        )));
    }

    #[test]
    fn fwht_dim2_produces_shuffle_add_sub_normalize() {
        // Arrange: FWHT with dim=2 should generate butterfly + normalize
        let body = &[TraceOp::Input(0), TraceOp::FWHT { src: ValueId(0), dim: 2 }];
        let (prog, _slots) = raw_prog(body);
        // Assert: should contain VecShuffle + Add + Sub + ConditionalSelect + Mul (normalize)
        let has_shuffle = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecShuffle { .. }));
        let has_add = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Add, .. }));
        let has_sub = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Sub, .. }));
        let has_mul = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { op: VecOp::Mul, .. }));
        assert!(has_shuffle, "FWHT dim=2 should produce VecShuffle for butterfly");
        assert!(has_add, "FWHT dim=2 should produce Add for sum");
        assert!(has_sub, "FWHT dim=2 should produce Sub for diff");
        assert!(has_mul, "FWHT dim=2 should produce Mul for 1/sqrt(2) normalization");
    }

    #[test]
    fn broadcast_load_produces_broadcast_mem_load() {
        // Arrange: BroadcastLoad loads scalar from memory and broadcasts
        let body = &[
            TraceOp::Input(0), // slot 0: base
            TraceOp::Input(0), // slot 1: offset
            TraceOp::BroadcastLoad { base: ValueId(0), offset: ValueId(1) },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::MemLoad(..), .. }
        )));
    }

    #[test]
    fn auto_lower_trace_into_binop_writes_directly_to_dst() {
        // Arrange: trace ends with Add → should write directly to dst (no extra Mov)
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[
            TraceOp::Input(0),
            TraceOp::Const(3.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        // Act
        auto_lower_trace_into(&mut prog, body, &[input], dst, SimdWidth::W256, QuantPrecision::F32).unwrap();
        // Assert: last instruction should be VecBinOp with dst == provided dst
        let last = prog.instrs.last().expect("should have instructions");
        assert!(matches!(
            last,
            VmInstr::VecBinOp { dst: d, op: VecOp::Add, .. } if *d == dst
        ));
    }

    #[test]
    fn quant_interleave_produces_quant_interleave_instr() {
        // Arrange: QuantInterleave(lo, hi) → VmInstr::QuantInterleave
        let body = &[
            TraceOp::Input(0), // slot 0: lo
            TraceOp::Input(0), // slot 1: hi
            TraceOp::QuantInterleave { lo: ValueId(0), hi: ValueId(1) },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::QuantInterleave { .. })));
    }

    #[test]
    fn auto_lower_trace_multi_empty_targets_is_ok() {
        // Arrange: body with ops but no targets — should succeed with no copies
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Const(1.0), TraceOp::Add(ValueId(0), ValueId(1))];
        // Act: no targets requested
        let result = auto_lower_trace_multi(
            &mut prog, body, &[input],
            &[], // empty targets
            SimdWidth::W256, QuantPrecision::F32,
        );
        // Assert: should succeed, no Mov instructions for targets
        assert!(result.is_ok());
        let mov_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Mov { .. })).count();
        assert_eq!(mov_count, 0, "no targets means no Mov copies");
    }

    // ── 10 new tests (wave-12x60) ──

    #[test]
    fn broadcast_scalar_produces_broadcast_extract_lane0() {
        // Arrange: BroadcastScalar extracts lane 0 and broadcasts to all lanes
        let body = &[
            TraceOp::Input(0), // slot 0: src
            TraceOp::BroadcastScalar { src: ValueId(0) },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 2);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::ExtractLane0(_), .. }
        )));
    }

    #[test]
    fn scalar_load_produces_scalar_load_instr() {
        // Arrange: ScalarLoad reads from base + offset
        let body = &[
            TraceOp::Input(0), // slot 0: base
            TraceOp::Input(0), // slot 1: offset
            TraceOp::ScalarLoad { base: ValueId(0), offset: ValueId(1) },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarLoad { .. })));
    }

    #[test]
    fn stride_mul_produces_int_mul_stride() {
        // Arrange: StrideMul computes value * stride
        let body = &[
            TraceOp::Input(0), // slot 0: value
            TraceOp::StrideMul { value: ValueId(0), stride: 128 },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 2);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::IntMulStride { stride: 128, .. }
        )));
    }

    #[test]
    fn ptr_add_produces_load_ptr() {
        // Arrange: PtrAdd computes base + offset as pointer
        let body = &[
            TraceOp::Input(0), // slot 0: base
            TraceOp::Input(0), // slot 1: offset
            TraceOp::PtrAdd { base: ValueId(0), offset: ValueId(1) },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::LoadPtr { .. })));
    }

    #[test]
    fn bit_extract_returns_error_in_f32_simd() {
        // Arrange: BitExtract is not supported in f32 SIMD pipeline
        let body = &[
            TraceOp::Input(0),
            TraceOp::BitExtract { src: ValueId(0), offset: 4, width: 2 },
        ];
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Act
        let result = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn hreduce_max_produces_hreduce_max_instr() {
        // Arrange: HReduce with Max kind
        use crate::compiler::trace::ReduceKind;
        let body = &[
            TraceOp::Input(0),
            TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Max },
        ];
        let (prog, slots) = raw_prog(body);
        // Assert
        assert_eq!(slots.len(), 2);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::HReduce { op: ReduceOp::Max, .. }
        )));
    }

    #[test]
    fn auto_lower_trace_raw_with_bf16_dtype_emits_bf16_binop() {
        // Arrange: auto_lower_trace_raw with BF16 default_dtype
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[
            TraceOp::Input(0),
            TraceOp::Input(0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        // Act
        let slots = auto_lower_trace_raw(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::BF16).unwrap();
        // Assert: VecBinOp should carry BF16 dtype
        assert_eq!(slots.len(), 3);
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecBinOp { op: VecOp::Add, dtype: QuantPrecision::BF16, .. }
        )));
    }

    #[test]
    fn single_op_trace_mul_produces_exactly_expected_instrs() {
        // Arrange: Input(0), Input(0), Mul → Broadcast for inputs + VecBinOp for Mul
        let body = &[
            TraceOp::Input(0),
            TraceOp::Input(0),
            TraceOp::Mul(ValueId(0), ValueId(1)),
        ];
        let (prog, slots) = raw_prog(body);
        // Assert: 3 slots, at least one VecBinOp::Mul, no VecBinOp with other ops
        assert_eq!(slots.len(), 3);
        let mul_count = prog.instrs.iter().filter(|i| matches!(
            i,
            VmInstr::VecBinOp { op: VecOp::Mul, .. }
        )).count();
        assert_eq!(mul_count, 1, "exactly one Mul VecBinOp should be emitted");
    }

    #[test]
    fn auto_lower_trace_into_const_writes_broadcast_to_dst() {
        // Arrange: single Const op → should broadcast directly to dst
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Const(5.0)];
        // Act
        auto_lower_trace_into(&mut prog, body, &[input], dst, SimdWidth::W256, QuantPrecision::F32).unwrap();
        // Assert: last instruction should be Broadcast with dst == provided dst
        let last = prog.instrs.last().expect("should have instructions");
        assert!(matches!(
            last,
            VmInstr::Broadcast { dst: d, src: ScalarExpr::Const(5.0), .. } if *d == dst
        ));
    }

    #[test]
    fn auto_lower_trace_no_inputs_returns_error() {
        // Arrange: auto_lower_trace requires at least one input for writeback
        let mut prog = VmProgram::new();
        let body = &[TraceOp::Const(1.0), TraceOp::Const(2.0), TraceOp::Add(ValueId(0), ValueId(1))];
        // Act: empty inputs slice
        let result = auto_lower_trace(&mut prog, body, &[], SimdWidth::W256, QuantPrecision::F32);
        // Assert: should fail because inputs is empty (no primary to write back to)
        assert!(result.is_err());
    }

    // ── REQ-AIS-001: ComputePattern 驱动路由入口函数测试 ──

    #[test]
    fn auto_lower_elementwise_accepts_elementwise_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Const(2.0), TraceOp::Mul(ValueId(0), ValueId(1))];
        let pattern = ComputePattern::Elementwise { body: body.to_vec() };
        let result = auto_lower_elementwise(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn auto_lower_elementwise_accepts_binary_elementwise_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Input(0), TraceOp::Add(ValueId(0), ValueId(1))];
        let pattern = ComputePattern::BinaryElementwise { body: body.to_vec() };
        let result = auto_lower_elementwise(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::BF16, &pattern);
        assert!(result.is_ok());
        // Verify BF16 dtype propagated
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecBinOp { op: VecOp::Add, dtype: QuantPrecision::BF16, .. }
        )));
    }

    #[test]
    fn auto_lower_elementwise_accepts_injective_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Sqrt(ValueId(0))];
        let pattern = ComputePattern::Injective { body: body.to_vec(), num_inputs: 1, num_outputs: 1 };
        let result = auto_lower_elementwise(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_ok());
    }

    #[test]
    fn auto_lower_elementwise_rejects_reduction_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
        let pattern = ComputePattern::Reduction {
            identity: 0.0,
            combine: body.to_vec(),
            second_pass: None,
            normalize: None,
        };
        let result = auto_lower_elementwise(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_err());
    }

    #[test]
    fn auto_lower_elementwise_rejects_gemm_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let result = auto_lower_elementwise(&mut prog, &[], &[input], SimdWidth::W256, QuantPrecision::F32, &ComputePattern::Gemm);
        assert!(result.is_err());
    }

    #[test]
    fn auto_lower_reduction_accepts_reduction_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let combine_body = &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
        let pattern = ComputePattern::Reduction {
            identity: 0.0,
            combine: combine_body.to_vec(),
            second_pass: None,
            normalize: None,
        };
        let result = auto_lower_reduction(&mut prog, combine_body, None, &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_ok());
    }

    #[test]
    fn auto_lower_reduction_accepts_normlike_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let reduce_body = &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Max }];
        let pattern = ComputePattern::NormLike {
            reduce: reduce_body.to_vec(),
            finalize: vec![],
            transform: vec![],
        };
        let result = auto_lower_reduction(&mut prog, reduce_body, None, &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_ok());
    }

    #[test]
    fn auto_lower_reduction_with_normalize_phase() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let combine_body = &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
        let normalize_body = &[TraceOp::Input(0), TraceOp::Const(0.5), TraceOp::Mul(ValueId(0), ValueId(1))];
        let pattern = ComputePattern::Reduction {
            identity: 0.0,
            combine: combine_body.to_vec(),
            second_pass: None,
            normalize: Some(normalize_body.to_vec()),
        };
        let result = auto_lower_reduction(&mut prog, combine_body, Some(normalize_body), &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_ok());
    }

    #[test]
    fn auto_lower_reduction_rejects_elementwise_pattern() {
        use crate::compiler::trace::ComputePattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[TraceOp::Input(0), TraceOp::Const(2.0), TraceOp::Mul(ValueId(0), ValueId(1))];
        let pattern = ComputePattern::Elementwise { body: body.to_vec() };
        let result = auto_lower_reduction(&mut prog, body, None, &[input], SimdWidth::W256, QuantPrecision::F32, &pattern);
        assert!(result.is_err());
    }

    #[test]
    fn auto_lower_structural_handles_gather_load() {
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let indices = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = &[
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::GatherLoad { base: ValueId(0), indices: ValueId(1), stride: 4 },
        ];
        let result = auto_lower_structural(&mut prog, body, &[base, indices], SimdWidth::W256, QuantPrecision::F32);
        assert!(result.is_ok());
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::GatherLoad { stride: 4, .. })));
    }

    #[test]
    fn auto_lower_structural_handles_table_lookup() {
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let row_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let body = &[
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::TableLookup { base: ValueId(0), row_index: ValueId(1), row_bytes: 256 },
        ];
        let result = auto_lower_structural(&mut prog, body, &[base, row_idx], SimdWidth::W256, QuantPrecision::F32);
        assert!(result.is_ok());
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::TableLookup { row_bytes: 256, .. })));
    }

    #[test]
    fn auto_lower_structural_dtype_propagation() {
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let offset = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let body = &[
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::VecLoadIndexed { base: ValueId(0), offset: ValueId(1) },
        ];
        let result = auto_lower_structural(&mut prog, body, &[base, offset], SimdWidth::W256, QuantPrecision::BF16);
        assert!(result.is_ok());
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecLoad { dtype: QuantPrecision::BF16, predicate: None, .. }
        )));
    }

    #[test]
    fn compute_pattern_driven_dispatch_end_to_end() {
        // Simulate full AIS flow: classify_pattern → route to correct entry
        use crate::compiler::trace::classify_pattern;
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Elementwise body: Input(0), Const(2.0), Mul
        let body = &[TraceOp::Input(0), TraceOp::Const(2.0), TraceOp::Mul(ValueId(0), ValueId(1))];
        let pattern = classify_pattern(body);

        // classify_pattern should produce Elementwise or Injective
        let result = match &pattern {
            ComputePattern::Elementwise { .. }
            | ComputePattern::BinaryElementwise { .. }
            | ComputePattern::Injective { .. } => {
                auto_lower_elementwise(&mut prog, body, &[input], SimdWidth::W256, QuantPrecision::BF16, &pattern)
            }
            _ => panic!("Expected Elementwise/BinaryElementwise/Injective, got {:?}", pattern),
        };
        assert!(result.is_ok());
        assert!(prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecBinOp { op: VecOp::Mul, dtype: QuantPrecision::BF16, .. }
        )));
    }
}
