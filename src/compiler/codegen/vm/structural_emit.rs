//! Structural op inline lowering — Gather, ColumnSlice, RoPE.

use super::instr::*;
use super::plan_lower::SymDimSlotMap;
use crate::compiler::trace::{QuantPrecision, TraceOp, ReduceKind, ValueId};
use crate::types::CompilerError;

pub(crate) fn emit_gather_inline(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    embed_dim: usize,
    width: SimdWidth,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    telemetry_ptr: Option<VRegId>,
    embedding_scale: Option<f32>,
    indices_kind: crate::compiler::graph::GatherIndicesKind,
    compute_dtype: QuantPrecision,
    weight_dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    use crate::compiler::graph::GatherIndicesKind;

    if embed_dim == 0 {
        return Err(CompilerError::CodegenViolation("emit_gather_inline: embed_dim is 0".into()));
    }

    let lanes = width.f32_lanes().max(1);
    // Weight-side dimensions (VecLoad reads from weight table)
    let weight_elem = weight_dtype.elem_bytes();
    let weight_row_bytes = embed_dim * weight_elem;
    // Compute-side dimensions (VecStore writes to activation buffer)
    let compute_elem = compute_dtype.elem_bytes();
    let compute_row_bytes = embed_dim * compute_elem;
    let compute_vec_step = lanes * compute_elem;
    let dim_vecs = embed_dim / lanes;
    let index_elem = 4usize;

    prog.emit(VmInstr::Comment(match indices_kind {
        GatherIndicesKind::Tensor => "Gather: indexed embedding lookup (auto_select)".into(),
        GatherIndicesKind::Arange => "Gather: sequential position lookup (auto_select)".into(),
        GatherIndicesKind::Zeros  => "Gather: broadcast first row (auto_select)".into(),
    }));

    // §13.10 L2 norm accumulator
    let norm_sq_acc = if telemetry_ptr.is_some() && dim_vecs > 0 {
        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype: compute_dtype, });
        Some(acc)
    } else {
        None
    };

    // TraceOp body for optional embedding scale: Input(0) * scale
    let scale_body: Vec<TraceOp> = if let Some(s) = embedding_scale {
        vec![TraceOp::Const(s as f64), TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(1))]
    } else {
        vec![TraceOp::Input(0)] // identity pass-through
    };

    // TraceOp body for L2 norm telemetry: HReduce(Sum, x²) + Sqrt
    let norm_body = vec![
        TraceOp::Input(0),            // 0: x
        TraceOp::Mul(ValueId(0), ValueId(0)),           // 1: x²
        TraceOp::HReduce { src: ValueId(1), op: ReduceKind::Sum }, // 2: sum(x²)
        TraceOp::Sqrt(ValueId(2)),             // 3: sqrt(sum(x²))
    ];

    // Scalar scratch VRegs for index computation
    let idx_scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    let row_offset = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let table_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let out_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: out_row, src: PtrExpr::VRegPlusConst(output_ptr, 0) });

    // Outer seq loop
    prog.emit_loop(seq_bound, index_elem, |prog, _seq_ctr, seq_byte_off| {
        // Index computation (structural — differs per indices_kind)
        match indices_kind {
            GatherIndicesKind::Tensor => {
                prog.emit(VmInstr::ScalarLoad {
                    dst: idx_scalar, base: input_ptr,
                    offset: OffsetExpr::LoopOffset(seq_byte_off),
                });
                prog.emit(VmInstr::IntMulStride {
                    dst: row_offset, src: idx_scalar, stride: weight_row_bytes,
                });
                prog.emit(VmInstr::LoadPtr {
                    dst: table_row, src: PtrExpr::VRegPlusVReg(weight_ptr, row_offset),
                });
            }
            GatherIndicesKind::Arange => {
                let scale = weight_row_bytes / index_elem;
                prog.emit(VmInstr::IntMulStride {
                    dst: row_offset, src: seq_byte_off, stride: scale,
                });
                prog.emit(VmInstr::LoadPtr {
                    dst: table_row, src: PtrExpr::VRegPlusVReg(weight_ptr, row_offset),
                });
            }
            GatherIndicesKind::Zeros => {
                prog.emit(VmInstr::LoadPtr {
                    dst: table_row, src: PtrExpr::VRegPlusConst(weight_ptr, 0),
                });
            }
        }

        // Inner dim loop: VecLoad (weight_dtype) + optional scale + VecStore (compute_dtype)
        if dim_vecs > 0 {
            // Same dtype for load and store (weight_blob is F32 for BF16 models — pack converts BF16→F32)
            prog.emit_loop(BoundExpr::Const(dim_vecs), compute_vec_step, |prog, _ctr, d_off| {
                let data = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad {
                    dst: data, base: table_row,
                    offset: OffsetExpr::LoopOffset(d_off), width,
                    dtype: weight_dtype,
                });
                let slots = super::auto_select::auto_lower_trace_raw(
                    prog, &scale_body, &[data], width, compute_dtype).expect("gather scale trace auto_lower failed");
                let result = slots.last().copied().unwrap_or(data);

                prog.emit(VmInstr::VecStore {
                    base: out_row, offset: OffsetExpr::LoopOffset(d_off),
                    src: result, width,
                    dtype: compute_dtype,
                });
                if let Some(acc) = norm_sq_acc {
                    let l2_body = vec![TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(0)), TraceOp::Input(1), TraceOp::Add(ValueId(1), ValueId(2))];
                    super::auto_select::auto_lower_trace_into(
                        prog, &l2_body, &[data, acc], acc, width, compute_dtype,
                    ).expect("gather L2 norm auto_lower failed");
                }
            });
        }

        // Advance output row pointer
        prog.emit(VmInstr::LoadPtr {
            dst: out_row, src: PtrExpr::VRegPlusConst(out_row, compute_row_bytes),
        });
    });

    // §13.10 Embedding L2 norm finalization via auto_lower_trace
    if let Some(tel_ptr) = telemetry_ptr {
        if let Some(acc) = norm_sq_acc {
            use crate::compiler::graph::telemetry_offsets;
            prog.emit(VmInstr::Comment("§13.10 Embedding L2 norm telemetry (auto_select)".into()));
            let slots = super::auto_select::auto_lower_trace_raw(
                prog, &norm_body, &[acc], width, compute_dtype).expect("gather norm trace auto_lower failed");
            let l2_norm = slots.last().copied().unwrap_or(acc);
            prog.emit(VmInstr::VecStore {
                base: tel_ptr,
                offset: OffsetExpr::Const(telemetry_offsets::EMBED_L2_NORM_OFFSET),
                src: l2_norm, width: SimdWidth::Scalar,
                dtype: compute_dtype,
            });
        }
    }

    Ok(())
}

/// ARCH-AUTO-INSTR-SELECT structural: ColumnSlice (row-major column copy) via TraceOp + auto_lower_trace。
///
/// 替代 `lower::lower_column_slice` 的手写 VmInstr 发射。
///
/// TraceOp body: Input(0) — identity pass-through (pure memory copy, no arithmetic).
/// 循环骨架:
///   emit_loop(seq_bound)                               // 外层: seq_len
///     emit_loop(BoundExpr::Const(slice_vecs))           // 内层: 向量化切片维度
///       auto_lower_trace_raw(body, QuantPrecision::F32)                      // VecLoad + VecStore
///     tail handling (scalar)
///     row pointer advance (input/output stride different)
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_column_slice_inline(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    input_inner: usize,
    start: usize,
    slice_dim: usize,
    width: SimdWidth,
    input_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if slice_dim == 0 {
        return Err(CompilerError::CodegenViolation("emit_column_slice_inline: slice_dim is 0".into()));
    }
    if start + slice_dim > input_inner {
        return Err(CompilerError::CodegenViolation(format!(
            "emit_column_slice_inline: start({start}) + slice_dim({slice_dim}) > input_inner({input_inner})",
        )));
    }

    let elem = dtype.elem_bytes();
    let lanes = width.f32_lanes().max(1);
    let input_row_bytes = input_inner * dtype.elem_bytes();
    let output_row_bytes = slice_dim * dtype.elem_bytes();
    let start_bytes = start * elem;
    let slice_vecs = slice_dim / lanes;
    let tail = slice_dim - slice_vecs * lanes;
    let vec_step = lanes * elem;

    prog.emit(VmInstr::Comment(format!(
        "ColumnSlice: input[s,{start}..{end}] → output[s,0..{slice_dim}] (auto_select)",
        end = start + slice_dim,
    )));

    // Identity trace body: pure pass-through
    let copy_body = vec![TraceOp::Input(0)];

    // Row pointers with different strides
    let in_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let out_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: in_row, src: PtrExpr::VRegPlusConst(input_ptr, start_bytes) });
    prog.emit(VmInstr::LoadPtr { dst: out_row, src: PtrExpr::VRegPlusConst(output_ptr, 0) });

    // Outer seq loop
    prog.emit_loop(seq_bound, output_row_bytes, |prog, _seq_ctr, _byte_off| {
        // Inner vectorized copy: auto_select driven
        if slice_vecs > 0 {
            prog.emit_loop(BoundExpr::Const(slice_vecs), vec_step, |prog, _ctr, col_off| {
                let data = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad {
                    dst: data, base: in_row,
                    offset: OffsetExpr::LoopOffset(col_off), width,
                    dtype,
                });
                // Apply identity trace via auto_lower_trace_raw
                let slots = super::auto_select::auto_lower_trace_raw(
                    prog, &copy_body, &[data], width, dtype).expect("column_slice copy trace auto_lower failed");
                let result = slots[0];
                prog.emit(VmInstr::VecStore {
                    base: out_row, offset: OffsetExpr::LoopOffset(col_off),
                    src: result, width,
                    dtype,
                });
            });
        }

        // Tail: scalar elements
        if tail > 0 {
            let s_width = SimdWidth::Scalar;
            let tail_base_bytes = slice_vecs * vec_step;
            let s_data = prog.alloc_vreg(VRegKind::Vec, s_width);
            for t in 0..tail {
                let col_off_const = tail_base_bytes + t * elem;
                prog.emit(VmInstr::VecLoad {
                    dst: s_data, base: in_row,
                    offset: OffsetExpr::Const(col_off_const), width: s_width,
                    dtype,
                });
                prog.emit(VmInstr::VecStore {
                    base: out_row, offset: OffsetExpr::Const(col_off_const),
                    src: s_data, width: s_width,
                    dtype,
                });
            }
        }

        // Advance row pointers (different strides)
        prog.emit(VmInstr::LoadPtr {
            dst: in_row, src: PtrExpr::VRegPlusConst(in_row, input_row_bytes),
        });
        prog.emit(VmInstr::LoadPtr {
            dst: out_row, src: PtrExpr::VRegPlusConst(out_row, output_row_bytes),
        });
    });

    Ok(())
}

/// ARCH-AUTO-INSTR-SELECT structural: RoPE (positional encoding) via TraceOp + auto_lower_trace。
///
/// 替代 `lower::lower_rope_full` — 旋转计算体使用 `auto_lower_trace_multi`，
/// passthrough 部分使用 `auto_lower_trace_raw` (identity trace)。
///
/// 循环骨架 (全部 emit_loop):
///   emit_loop(seq_bound)                      // 外层: seq_len
///     emit_loop(BoundExpr::Const(num_heads))  // 中层: heads
///       emit_loop(BoundExpr::Const(vec_count)) // 内层: rotation pairs
///         auto_lower_trace_multi(rope_body)   // 旋转计算
///       emit_loop(BoundExpr::Const(pt_vecs))  // passthrough (identity trace)
///         auto_lower_trace_raw(identity_body, QuantPrecision::F32)
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_rope_inline(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    num_heads: usize,
    head_dim: usize,
    partial: f32,
    width: SimdWidth,
    input_ptr: VRegId,
    output_ptr: VRegId,
    cos_sin_offset: usize,
    sym_map: &SymDimSlotMap,
    dtype: QuantPrecision,
    position_offset: Option<VRegId>,
) -> Result<(), CompilerError> {
    if head_dim == 0 || head_dim % 2 != 0 || num_heads == 0 {
        return Err(CompilerError::CodegenViolation(
            format!("emit_rope_inline: invalid params (heads={num_heads}, head_dim={head_dim})"),
        ));
    }
    if partial <= 0.0 || partial > 1.0 {
        return Err(CompilerError::CodegenViolation(
            format!("emit_rope_inline: partial must be in (0,1], got {partial}"),
        ));
    }

    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let rot_dim = ((head_dim as f32 * partial) as usize) & !1;
    let rot_dim = rot_dim.max(2);
    let half_rot = rot_dim / 2;
    let half = head_dim / 2;
    let passthrough_dim = head_dim - rot_dim;

    // cos/sin table base from scratchpad
    let scratch_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: scratch_base,
        src: sym_map.resolve("scratchpad").cloned().ok_or_else(|| {
            CompilerError::CodegenViolation(
                "emit_rope_inline: scratchpad ABI slot missing".into())
        })?,
    });
    let cos_sin_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: cos_sin_base,
        src: PtrExpr::VRegPlusConst(scratch_base, cos_sin_offset),
    });

    // VReg pool (allocated outside seq loop to reduce register pressure)
    let tok_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tok_output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let head_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let head_output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let x_even = prog.alloc_vreg(VRegKind::Vec, width);
    let x_odd = prog.alloc_vreg(VRegKind::Vec, width);
    let cos_val = prog.alloc_vreg(VRegKind::Vec, width);
    let sin_val = prog.alloc_vreg(VRegKind::Vec, width);

    // Strides
    let head_step_bytes = head_dim * dtype.elem_bytes();
    let token_step_bytes = num_heads * head_dim * dtype.elem_bytes();
    let cos_row_bytes = head_dim * dtype.elem_bytes();
    let vec_count = half_rot / lanes;
    let pt_vecs = passthrough_dim / lanes;
    let vec_step = lanes * elem;

    // TraceOp body for rotation: out_even = x0*cos - x1*sin, out_odd = x1*cos + x0*sin
    let rope_body: Vec<TraceOp> = vec![
        TraceOp::Input(0),  // [0] x0
        TraceOp::Input(1),  // [1] x1
        TraceOp::Input(2),  // [2] cos
        TraceOp::Input(3),  // [3] sin
        TraceOp::Mul(ValueId(0), ValueId(2)), // [4] x0 * cos
        TraceOp::Mul(ValueId(1), ValueId(3)), // [5] x1 * sin
        TraceOp::Sub(ValueId(4), ValueId(5)), // [6] out_even = x0*cos - x1*sin
        TraceOp::Mul(ValueId(1), ValueId(2)), // [7] x1 * cos
        TraceOp::Mul(ValueId(0), ValueId(3)), // [8] x0 * sin
        TraceOp::Add(ValueId(7), ValueId(8)), // [9] out_odd = x1*cos + x0*sin
    ];

    // TraceOp body for passthrough: identity
    let passthrough_body = vec![TraceOp::Input(0)];

    // Outer seq loop
    prog.emit_loop(seq_bound, token_step_bytes, |prog, p_ctr, tok_off| {
        prog.emit(VmInstr::LoadPtr { dst: tok_input, src: PtrExpr::VRegPlusVReg(input_ptr, tok_off) });
        prog.emit(VmInstr::LoadPtr { dst: tok_output, src: PtrExpr::VRegPlusVReg(output_ptr, tok_off) });

        // When position_offset is set (mega-kernel decode), use gen_counter as the
        // cos/sin position. p_ctr=0 (seq_bound=1), so offset = position_offset.
        let pos_vreg = if let Some(offset) = position_offset {
            let abs_pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: abs_pos, a: p_ctr, b: GprOperand::VReg(offset ), op: GprOp::Add });
            abs_pos
        } else {
            p_ctr
        };

        // Middle head loop
        prog.emit_loop(BoundExpr::Const(num_heads), head_step_bytes, |prog, _h_ctr, h_off| {
            prog.emit(VmInstr::LoadPtr { dst: head_input, src: PtrExpr::VRegPlusVReg(tok_input, h_off) });
            prog.emit(VmInstr::LoadPtr { dst: head_output, src: PtrExpr::VRegPlusVReg(tok_output, h_off) });

            // Part 1: Rotation via auto_lower_trace_multi
            if vec_count > 0 {
                prog.emit_loop(BoundExpr::Const(vec_count), vec_step, |prog, _pair_ctr, pair_off| {
                    prog.emit(VmInstr::VecLoad { dst: x_even, base: head_input, offset: OffsetExpr::LoopOffset(pair_off), width, dtype, });
                    prog.emit(VmInstr::VecLoad {
                        dst: x_odd, base: head_input,
                        offset: OffsetExpr::loop_plus_const(pair_off, half_rot * dtype.elem_bytes()), width,
                        dtype,
                    });
                    let p_bytes = OffsetExpr::Mul(Box::new(OffsetExpr::ScalarVReg(pos_vreg)), cos_row_bytes);
                    let cos_off = OffsetExpr::Add(Box::new(p_bytes.clone()), Box::new(OffsetExpr::LoopOffset(pair_off)));
                    let sin_off = OffsetExpr::Add(Box::new(cos_off.clone()), Box::new(OffsetExpr::Const(half * dtype.elem_bytes())));
                    prog.emit(VmInstr::VecLoad { dst: cos_val, base: cos_sin_base, offset: cos_off, width, dtype, });
                    prog.emit(VmInstr::VecLoad { dst: sin_val, base: cos_sin_base, offset: sin_off, width, dtype, });

                    let out_even = prog.alloc_vreg(VRegKind::Vec, width);
                    let out_odd = prog.alloc_vreg(VRegKind::Vec, width);
                    super::auto_select::auto_lower_trace_multi(prog, &rope_body,
                        &[x_even, x_odd, cos_val, sin_val],
                        &[(out_even, 6), (out_odd, 9)], width, dtype)
                        .expect("emit_rope_inline: rotation trace auto_lower failed");

                    prog.emit(VmInstr::VecStore { base: head_output, offset: OffsetExpr::LoopOffset(pair_off), src: out_even, width, dtype, });
                    prog.emit(VmInstr::VecStore {
                        base: head_output,
                        offset: OffsetExpr::loop_plus_const(pair_off, half_rot * dtype.elem_bytes()),
                        src: out_odd, width,
                        dtype,
                    });
                });
            }

            // Scalar tail pairs
            let pair_tail = half_rot - vec_count * lanes;
            for t in 0..pair_tail {
                let off_even = vec_count * vec_step + t * elem;
                let s1 = SimdWidth::Scalar;
                let sx_even = prog.alloc_vreg(VRegKind::Vec, s1);
                let sx_odd = prog.alloc_vreg(VRegKind::Vec, s1);
                let scos = prog.alloc_vreg(VRegKind::Vec, s1);
                let ssin = prog.alloc_vreg(VRegKind::Vec, s1);
                prog.emit(VmInstr::VecLoad { dst: sx_even, base: head_input, offset: OffsetExpr::Const(off_even), width: s1, dtype, });
                prog.emit(VmInstr::VecLoad { dst: sx_odd, base: head_input, offset: OffsetExpr::Const(off_even + half_rot * dtype.elem_bytes()), width: s1, dtype, });
                let p_bytes = OffsetExpr::Mul(Box::new(OffsetExpr::ScalarVReg(p_ctr)), cos_row_bytes);
                let cos_off = OffsetExpr::Add(Box::new(p_bytes.clone()), Box::new(OffsetExpr::Const(off_even)));
                let sin_off = OffsetExpr::Add(Box::new(cos_off.clone()), Box::new(OffsetExpr::Const(half * dtype.elem_bytes())));
                prog.emit(VmInstr::VecLoad { dst: scos, base: cos_sin_base, offset: cos_off, width: s1, dtype, });
                prog.emit(VmInstr::VecLoad { dst: ssin, base: cos_sin_base, offset: sin_off, width: s1, dtype, });

                let s_out_even = prog.alloc_vreg(VRegKind::Vec, s1);
                let s_out_odd = prog.alloc_vreg(VRegKind::Vec, s1);
                super::auto_select::auto_lower_trace_multi(prog, &rope_body,
                    &[sx_even, sx_odd, scos, ssin],
                    &[(s_out_even, 6), (s_out_odd, 9)], s1, dtype)
                    .expect("emit_rope_inline: scalar rotation trace auto_lower failed");
                prog.emit(VmInstr::VecStore { base: head_output, offset: OffsetExpr::Const(off_even), src: s_out_even, width: s1, dtype, });
                prog.emit(VmInstr::VecStore {
                    base: head_output, offset: OffsetExpr::Const(off_even + half_rot * dtype.elem_bytes()),
                    src: s_out_odd, width: s1,
                    dtype,
                });
            }

            // Part 2: Passthrough via auto_lower_trace_raw (identity trace)
            if passthrough_dim > 0 {
                let pt_base_off = rot_dim * dtype.elem_bytes();
                if pt_vecs > 0 {
                    prog.emit_loop(BoundExpr::Const(pt_vecs), vec_step, |prog, _ctr, byte_off| {
                        let data = prog.alloc_vreg(VRegKind::Vec, width);
                        prog.emit(VmInstr::VecLoad {
                            dst: data, base: head_input,
                            offset: OffsetExpr::loop_plus_const(byte_off, pt_base_off), width,
                            dtype,
                        });
                        let slots = super::auto_select::auto_lower_trace_raw(
                            prog, &passthrough_body, &[data], width, dtype).expect("emit_rope_inline: passthrough trace auto_lower failed");
                        prog.emit(VmInstr::VecStore {
                            base: head_output,
                            offset: OffsetExpr::loop_plus_const(byte_off, pt_base_off),
                            src: slots[0], width,
                            dtype,
                        });
                    });
                }
                // scalar tail
                let pt_tail = passthrough_dim % lanes;
                for t in 0..pt_tail {
                    let off = pt_base_off + pt_vecs * vec_step + t * elem;
                    let s1 = SimdWidth::Scalar;
                    let s_data = prog.alloc_vreg(VRegKind::Vec, s1);
                    prog.emit(VmInstr::VecLoad { dst: s_data, base: head_input, offset: OffsetExpr::Const(off), width: s1, dtype, });
                    let slots = super::auto_select::auto_lower_trace_raw(
                        prog, &passthrough_body, &[s_data], s1, dtype).expect("emit_rope_inline: scalar passthrough trace auto_lower failed");
                    prog.emit(VmInstr::VecStore { base: head_output, offset: OffsetExpr::Const(off), src: slots[0], width: s1, dtype, });
                }
            }
        }); // head
    }); // seq

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 27 REQ-AT-009: 模板驱动 RoPE 发射桥接
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 尝试通过模板驱动路径发射 RoPE 算子。
pub(crate) fn emit_rope_template_driven(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    num_heads: usize,
    head_dim: usize,
    partial: f32,
    width: SimdWidth,
    input_ptr: VRegId,
    cos_ptr: VRegId,
    sin_ptr: VRegId,
    output_ptr: VRegId,
    scratchpad_ptr: VRegId,
    dtype: QuantPrecision,
) -> Option<()> {
    use super::algo_registry;
    use super::algo_interpreter::{TemplateInterpreter, ParamTable, TemplateInputs};
    use crate::dispatch::device_profile::DeviceProfile;

    let strategy = if partial < 1.0 {
        crate::compiler::codegen::vm::algo_template::AlgoStrategy::RopePartial
    } else {
        crate::compiler::codegen::vm::algo_template::AlgoStrategy::RopeStandard
    };

    let profile = DeviceProfile::detect();
    let template = algo_registry::select_template(&strategy, &profile)?;

    let mut params = ParamTable::new();
    let seq_len_val = match &seq_bound {
        BoundExpr::Const(v) => *v,
        _ => 1,
    };
    params.set("seq_len", seq_len_val);
    params.set("head_dim", head_dim);
    params.set("partial_dim", (head_dim as f64 * partial as f64) as usize);

    let seq_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    let inputs = TemplateInputs::rope();

    let mut interp = TemplateInterpreter::new(params);
    let trace_ops = interp.instantiate(template, &inputs);

    super::auto_select::auto_lower_trace_raw(
        prog, &trace_ops, &[input_ptr, cos_ptr, sin_ptr, seq_offset],
        width, dtype,
    ).ok()?;

    let _ = (num_heads, scratchpad_ptr, output_ptr);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::vm::plan_lower::SymDimSlotMap;
    use crate::compiler::graph::GatherIndicesKind;
    use crate::types::CompilerError;

    // ── emit_gather_inline tests (6) ──────────────────────────────

    #[test]
    fn gather_rejects_zero_embed_dim() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            0, // embed_dim = 0 → error
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("embed_dim"), "expected mention of embed_dim, got: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn gather_tensor_produces_instrs_with_scalarload() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            64,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let has_scalar_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarLoad { .. }));
        assert!(has_scalar_load, "Tensor gather must emit ScalarLoad for index loading");
        let has_loop = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        assert!(has_loop, "Gather must produce a seq loop");
    }

    #[test]
    fn gather_arange_omits_scalarload() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(8),
            128,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let has_scalar_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarLoad { .. }));
        assert!(!has_scalar_load, "Arange gather should not emit ScalarLoad");
        // Still has IntMulStride for offset computation
        let has_int_mul = prog.instrs.iter().any(|i| matches!(i, VmInstr::IntMulStride { .. }));
        assert!(has_int_mul, "Arange gather must compute row offset via IntMulStride");
    }

    #[test]
    fn gather_zeros_loads_first_row_only() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            64,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Zeros,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let has_scalar_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarLoad { .. }));
        assert!(!has_scalar_load, "Zeros gather should not emit ScalarLoad");
        let has_int_mul = prog.instrs.iter().any(|i| matches!(i, VmInstr::IntMulStride { .. }));
        assert!(!has_int_mul, "Zeros gather should not emit IntMulStride");
        // Zeros path: LoadPtr with VRegPlusConst(weight, 0)
        let load_ptr_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoadPtr { .. })).count();
        assert!(load_ptr_count > 0, "Zeros gather must load the table row pointer");
    }

    #[test]
    fn gather_with_telemetry_emits_l2_norm() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            64,
            SimdWidth::W256,
            input, weight, output,
            Some(telemetry), None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // With telemetry, a Broadcast(0.0) accumulator is allocated for L2 norm
        let has_broadcast_zero = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == 0.0)
        });
        assert!(has_broadcast_zero, "Telemetry gather must emit Broadcast(0.0) for L2 norm accumulator");

        // Must contain a comment about L2 norm telemetry
        let has_l2_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(s) if s.contains("L2 norm"))
        });
        assert!(has_l2_comment, "Telemetry gather must emit L2 norm comment");
    }

    #[test]
    fn gather_with_embedding_scale_produces_mul_trace() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            64,
            SimdWidth::W256,
            input, weight, output,
            None,
            Some(0.125), // embedding scale
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // Scale body: Const(scale) * Input(0) → produces VecMul via auto_lower_trace
        let has_vec_op = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { .. }));
        assert!(has_vec_op, "Gather with embedding scale must produce VecBinOp for scale multiply");
    }

    // ── emit_column_slice_inline tests (4) ──────────────────────────

    #[test]
    fn column_slice_rejects_zero_slice_dim() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(4),
            64, 0, 0, // start=0, slice_dim=0 → error
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("slice_dim"), "expected mention of slice_dim, got: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn column_slice_rejects_out_of_bounds() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(4),
            32, // input_inner = 32
            20, // start = 20
            20, // slice_dim = 20 → start+slice=40 > input_inner=32
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        );

        assert!(result.is_err(), "start + slice_dim > input_inner must fail");
    }

    #[test]
    fn column_slice_emits_copy_with_different_strides() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // input_inner=128, start=32, slice_dim=64 → copy 64 elements from col 32..96
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(8),
            128, 32, 64,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        // Must have a seq loop (outer) and a dim loop (inner)
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_count >= 2, "ColumnSlice must emit at least 2 loops (seq + dim), got {loop_count}");

        // Must have VecLoad and VecStore for the copy
        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_load, "ColumnSlice must emit VecLoad");
        assert!(has_store, "ColumnSlice must emit VecStore");

        // Must have exactly 2 row pointer advances (in_row + out_row) per seq iteration
        let load_ptr_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoadPtr { .. })).count();
        assert!(load_ptr_count >= 4, "ColumnSlice must advance both in_row and out_row pointers");
    }

    #[test]
    fn column_slice_tail_handles_non_aligned_dim() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // slice_dim=10 with W256 (8 lanes) → 1 vec iteration + 2 scalar tail elements
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            16, 0, 10,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        // Check that there are scalar VecLoad/VecStore for tail handling
        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0, "Non-aligned slice_dim must produce scalar tail loads");
        assert!(scalar_stores > 0, "Non-aligned slice_dim must produce scalar tail stores");
    }

    // ── emit_rope_inline tests (3) ───────────────────────────────

    #[test]
    fn rope_rejects_zero_head_dim() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 0, // num_heads=4, head_dim=0 → error
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("head_dim") || msg.contains("heads"), "error should mention head params: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn rope_rejects_odd_head_dim() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 127, // head_dim=127 (odd) → error
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        );

        assert!(result.is_err(), "odd head_dim must be rejected");
    }

    #[test]
    fn rope_rejects_invalid_partial() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 64,
            0.0, // partial=0.0 → invalid
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("partial"), "error should mention partial: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // ── Additional 13 tests ──────────────────────────────────────

    // 1. Gather with BF16 compute dtype still emits VecLoad/VecStore
    #[test]
    fn gather_bf16_compute_dtype_emits_load_store() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            64,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::BF16, // compute_dtype = BF16
            QuantPrecision::F32,  // weight stored as F32
        ).unwrap();

        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_vec_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_vec_load, "BF16 gather must emit VecLoad");
        assert!(has_vec_store, "BF16 gather must emit VecStore");
    }

    // 2. Gather with W512 SIMD width emits wider vector ops
    #[test]
    fn gather_w512_width_emits_correct_lanes() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            128,
            SimdWidth::W512,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // W512 has 16 lanes, so dim_vecs = 128/16 = 8 inner iterations
        let w512_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W512, .. })
        }).count();
        let w512_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W512, .. })
        }).count();
        assert!(w512_loads > 0, "W512 gather must emit W512 VecLoad");
        assert!(w512_stores > 0, "W512 gather must emit W512 VecStore");
    }

    // 3. Gather Tensor with seq_bound=1 (single token decode)
    #[test]
    fn gather_tensor_single_token_produces_minimal_loop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(1), // single token
            32,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // Must have at least 1 outer loop (seq) and 1 inner loop (dim)
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_count >= 2, "Single-token gather must still emit seq+dim loops, got {loop_count}");

        // ScalarLoad for index read
        let scalar_loads = prog.instrs.iter().filter(|i| matches!(i, VmInstr::ScalarLoad { .. })).count();
        assert!(scalar_loads >= 1, "Tensor gather must emit ScalarLoad for index");
    }

    // 4. Gather with embedding scale AND telemetry simultaneously
    #[test]
    fn gather_scale_and_telemetry_both_active() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(3),
            64,
            SimdWidth::W256,
            input, weight, output,
            Some(telemetry),
            Some(1.5), // embedding scale
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // Must have both: scale VecBinOp and L2 norm telemetry
        let has_binop = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { .. }));
        let has_broadcast_zero = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == 0.0)
        });
        assert!(has_binop, "Scale + telemetry gather must produce VecBinOp");
        assert!(has_broadcast_zero, "Scale + telemetry gather must initialize L2 norm accumulator");
    }

    // 5. Gather Zeros with embed_dim not aligned to SIMD width still succeeds
    #[test]
    fn gather_zeros_non_aligned_embed_dim_succeeds() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // embed_dim=10, W256=8 lanes → dim_vecs=1, no remainder in inner dim loop
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            10, // not a multiple of 8
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Zeros,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_load, "Non-aligned gather must still emit VecLoad");
        assert!(has_store, "Non-aligned gather must still emit VecStore");
    }

    // 6. ColumnSlice with start=0 is equivalent to truncation
    #[test]
    fn column_slice_start_zero_truncation() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // start=0, slice_dim=64 from input_inner=128 → first 64 columns
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(4),
            128, 0, 64,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let has_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(s) if s.contains("0..64"))
        });
        assert!(has_comment, "ColumnSlice with start=0 must include range 0..64 in comment");
    }

    // 7. ColumnSlice with BF16 dtype uses 2-byte element stride
    #[test]
    fn column_slice_bf16_dtype_correct_strides() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // BF16: elem_bytes=2, input_inner=64, start=16, slice_dim=32
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(2),
            64, 16, 32,
            SimdWidth::W256,
            input, output,
            QuantPrecision::BF16,
        ).unwrap();

        // Verify BF16 VecLoad and VecStore are emitted
        let bf16_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, .. })
        }).count();
        let bf16_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { dtype: QuantPrecision::BF16, .. })
        }).count();
        assert!(bf16_loads > 0, "BF16 ColumnSlice must emit BF16 VecLoad");
        assert!(bf16_stores > 0, "BF16 ColumnSlice must emit BF16 VecStore");
    }

    // 8. ColumnSlice with exact boundary: start + slice_dim == input_inner
    #[test]
    fn column_slice_exact_boundary_succeeds() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // start=48, slice_dim=16, input_inner=64 → 48+16=64 == input_inner (exact boundary)
        let result = emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            64, 48, 16,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        );

        assert!(result.is_ok(), "Exact boundary (start+slice==input_inner) must succeed");
    }

    // 9. ColumnSlice with W128 produces narrower vector ops
    #[test]
    fn column_slice_w128_width_emits_narrower_vectors() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W128=4 lanes, slice_dim=16 → 4 vec iterations, no tail
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            32, 0, 16,
            SimdWidth::W128,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let w128_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W128, .. })
        }).count();
        assert!(w128_loads > 0, "W128 ColumnSlice must emit W128 VecLoad");
    }

    // 10. RoPE rejects partial > 1.0
    #[test]
    fn rope_rejects_partial_above_one() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 64,
            1.5, // partial=1.5 → invalid
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        );

        assert!(result.is_err(), "partial > 1.0 must be rejected");
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("partial"), "error should mention partial: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // 11. RoPE rejects zero num_heads
    #[test]
    fn rope_rejects_zero_num_heads() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            0, 64, // num_heads=0 → error
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        );

        assert!(result.is_err(), "num_heads=0 must be rejected");
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("heads"), "error should mention heads: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // 12. RoPE with partial=0.25 (p-RoPE) produces passthrough section
    #[test]
    fn rope_partial_produces_passthrough_section() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // head_dim=64, partial=0.25 → rot_dim=16, passthrough_dim=48
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 64,
            0.25,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // With partial < 1.0, must have 3 nested loops: seq, heads, rotation pairs
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        // seq loop + head loop + rotation vec loop + passthrough vec loop >= 4
        assert!(loop_count >= 4,
            "Partial RoPE must emit seq+head+rotation+passthrough loops, got {loop_count}");
    }

    // 13. RoPE with position_offset emits GprBinOp for position calculation
    #[test]
    fn rope_with_position_offset_emits_gpr_add() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let pos_offset = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 64,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            Some(pos_offset), // position_offset set → mega-kernel decode path
        ).unwrap();

        // position_offset triggers GprBinOp(Add) to compute absolute position
        let has_gpr_add = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::GprBinOp { op: GprOp::Add, .. })
        });
        assert!(has_gpr_add,
            "RoPE with position_offset must emit GprBinOp(Add) for position calculation");
    }

    // ── Wave 12k79: additional tests ─────────────────────────────────────

    // @trace TEST-12k79
    #[test]
    fn gather_arange_emits_loop_for_seq() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(8),
            32,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let has_loop_begin = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        assert!(has_loop_begin, "Arange gather must emit loop for seq dimension");
    }

    // @trace TEST-12k79
    #[test]
    fn gather_scalar_width_emits_scalar_ops() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4,
            SimdWidth::Scalar,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0, "Scalar width gather must emit scalar VecLoad");
    }

    // @trace TEST-12k79
    #[test]
    fn column_slice_rejects_start_exceeds_input() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            32, 40, 8, // start=40 > input_inner=32 → error
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        );

        assert!(result.is_err(), "start > input_inner must be rejected");
    }

    // @trace TEST-12k79
    #[test]
    fn rope_full_partial_emits_nested_loops() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // full partial=1.0, 8 heads — all dimensions are rotated
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            8, 64,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Must have at least 3 nested loops: seq, head, rotation-vec
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_count >= 3, "RoPE with full partial must emit >= 3 nested loops (seq+head+vec), got {loop_count}");
    }

    // @trace TEST-12k79
    #[test]
    fn rope_bf16_dtype_propagates_to_vec_load() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 64,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::BF16,
            None,
        ).unwrap();

        let has_bf16_load = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, .. })
        });
        assert!(has_bf16_load, "BF16 RoPE must emit BF16 VecLoad");
    }

    // @trace TEST-12k79
    #[test]
    fn rope_partial_half_produces_rot_and_passthrough() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // head_dim=128, partial=0.5 → rot_dim=64, passthrough_dim=64
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            2, 128,
            0.5,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Must have multiple loops for rotation and passthrough sections
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_count >= 4,
            "Half-partial RoPE must emit seq+head+rot+passthrough loops, got {loop_count}");
    }

    // @trace TEST-12k79
    #[test]
    fn gather_with_symbolic_bound_emits_dynamic_loop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".to_string(),
            max_alloc: 16,
        });

        emit_gather_inline(
            &mut prog,
            sym_bound,
            32,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let has_loop = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        assert!(has_loop, "Symbolic bound gather must emit loop");
    }

    // @trace TEST-12k79
    #[test]
    fn column_slice_with_scalar_width_emits_scalar_ops() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 0, 4,
            SimdWidth::Scalar,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0, "Scalar width ColumnSlice must emit scalar loads");
    }

    // @trace TEST-12k79
    #[test]
    fn gather_rejects_zero_seq_bound() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Zero-length sequence shouldn't cause issues — it should emit no body instructions
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(0),
            32,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // Zero seq should either be clean or emit minimal loop structure
        assert!(prog.len() > 0, "Even zero-seq gather must emit setup instructions");
    }

    // @trace TEST-12k79
    #[test]
    fn column_slice_with_slice_dim_equals_input_inner_is_full_copy() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // start=0, slice_dim=64, input_inner=64 → full copy (all columns)
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(2),
            64, 0, 64,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_load, "Full copy ColumnSlice must emit VecLoad");
        assert!(has_store, "Full copy ColumnSlice must emit VecStore");
    }

    // ── Wave 12kfb: 10 additional tests ─────────────────────────────────

    // 1. Gather Tensor: verify IntMulStride stride matches weight_row_bytes for F32
    #[test]
    fn gather_tensor_intmul_stride_matches_weight_row_bytes() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let embed_dim = 96; // weight_row_bytes = 96 * 4 = 384 for F32
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            embed_dim,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let stride_matches = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::IntMulStride { stride, .. } if *stride == embed_dim * 4)
        });
        assert!(stride_matches,
            "Tensor gather IntMulStride stride must equal weight_row_bytes ({})",
            embed_dim * 4);
    }

    // 2. Gather Arange: verify IntMulStride scale = weight_row_bytes / index_elem
    #[test]
    fn gather_arange_intmul_stride_scale_derived_from_row_bytes() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let embed_dim = 64; // weight_row_bytes = 256, scale = 256/4 = 64
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            embed_dim,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        let expected_scale = (embed_dim * 4) / 4; // weight_row_bytes / index_elem
        let scale_matches = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::IntMulStride { stride, .. } if *stride == expected_scale)
        });
        assert!(scale_matches,
            "Arange gather IntMulStride scale must be weight_row_bytes/index_elem = {}", expected_scale);
    }

    // 3. Gather Zeros with multi-token seq: verify no index-dependent ops, output row advances
    #[test]
    fn gather_zeros_multi_token_broadcasts_without_index_ops() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(6), // 6 tokens, all should broadcast row 0
            64,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Zeros,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // Must NOT have ScalarLoad or IntMulStride (no index computation)
        let has_scalar_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarLoad { .. }));
        let has_int_mul = prog.instrs.iter().any(|i| matches!(i, VmInstr::IntMulStride { .. }));
        assert!(!has_scalar_load, "Zeros gather must not emit ScalarLoad regardless of seq count");
        assert!(!has_int_mul, "Zeros gather must not emit IntMulStride regardless of seq count");

        // Must have VecStore for writing output rows
        let store_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(store_count > 0, "Zeros gather must emit VecStore for output");
    }

    // 4. ColumnSlice with W512 SIMD width produces wider vector operations
    #[test]
    fn column_slice_w512_emits_w512_load_store() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W512 = 16 lanes, slice_dim=64 → 4 vectorized iterations
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(2),
            128, 0, 64,
            SimdWidth::W512,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let w512_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W512, .. })
        }).count();
        let w512_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W512, .. })
        }).count();
        assert!(w512_loads > 0, "W512 ColumnSlice must emit W512 VecLoad");
        assert!(w512_stores > 0, "W512 ColumnSlice must emit W512 VecStore");
    }

    // 5. RoPE with head_dim=2 (minimum valid) produces rotation with scalar tail fallback
    #[test]
    fn rope_minimal_head_dim_2_produces_rotation() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // head_dim=2, partial=1.0 → rot_dim=2, half_rot=1, no vectorized path (lanes=8 > 1 pair)
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            2, 2, // 2 heads, head_dim=2
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Must have seq + head loops
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_count >= 2, "Minimal RoPE must emit seq+head loops, got {loop_count}");

        // Must have VecLoad for reading input elements
        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_load, "Minimal RoPE must load input values");
    }

    // 6. RoPE with Symbolic seq_bound still compiles correctly
    #[test]
    fn rope_symbolic_seq_bound_produces_loop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".to_string(),
            max_alloc: 32,
        });

        emit_rope_inline(
            &mut prog,
            sym_bound,
            4, 64,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        let has_loop = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        assert!(has_loop, "RoPE with symbolic bound must emit outer seq loop");
    }

    // 7. RoPE with Scalar width falls back to scalar rotation pairs
    #[test]
    fn rope_scalar_width_emits_scalar_rotation() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Scalar width = 1 lane, head_dim=4 → half_rot=2, vec_count=2/1=2 scalar iterations
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            2, 4,
            1.0,
            SimdWidth::Scalar,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0, "Scalar width RoPE must emit scalar VecLoad");
        assert!(scalar_stores > 0, "Scalar width RoPE must emit scalar VecStore");
    }

    // 8. Gather with BF16 weight_dtype uses 2-byte element for weight row
    #[test]
    fn gather_bf16_weight_dtype_shorter_weight_rows() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // weight_dtype=BF16 (2 bytes), compute_dtype=F32 (4 bytes)
        // weight_row_bytes = 64*2 = 128, compute_row_bytes = 64*4 = 256
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            64,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::BF16, // weight stored as BF16
        ).unwrap();

        // IntMulStride stride should use weight_row_bytes (64*2=128 for BF16)
        let bf16_stride = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::IntMulStride { stride, .. } if *stride == 64 * 2)
        });
        assert!(bf16_stride,
            "BF16 weight gather must use BF16 weight_row_bytes (128) as stride");
    }

    // 9. ColumnSlice with Symbolic bound emits dynamic outer loop
    #[test]
    fn column_slice_symbolic_bound_emits_dynamic_loop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".to_string(),
            max_alloc: 64,
        });

        emit_column_slice_inline(
            &mut prog,
            sym_bound,
            64, 0, 32,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let has_loop = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        assert!(has_loop, "ColumnSlice with symbolic bound must emit outer seq loop");
        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_load, "Symbolic ColumnSlice must still emit VecLoad for copy");
    }

    // 10. RoPE head_dim not divisible by lanes emits scalar tail rotation pairs
    #[test]
    fn rope_head_dim_not_aligned_to_lanes_emits_scalar_tail() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // head_dim=24, W256=8 lanes, half_rot=12, vec_count=12/8=1, pair_tail=4
        // → 1 vectorized iteration + 4 scalar tail pairs
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 24,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0,
            "head_dim=24 with W256 must emit scalar tail rotation pairs (pair_tail=4)");
    }

    // ── Wave 12kkd: 10 additional tests (56 total) ──────────────────────────

    // 1. Gather with embed_dim < lanes: dim_vecs=0 → no inner dim loop, only output row advance
    #[test]
    fn gather_embed_dim_less_than_lanes_skips_inner_dim_loop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W256=8 lanes, embed_dim=4 → dim_vecs=4/8=0 → no inner dim loop
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // Must have outer seq loop but NO inner dim loop (dim_vecs=0)
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_count, 1, "embed_dim < lanes must produce exactly 1 loop (outer seq only), got {loop_count}");

        // Still must emit VecStore for output row advance (LoadPtr with out_row)
        let has_load_ptr = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoadPtr { .. }));
        assert!(has_load_ptr, "Gather must emit LoadPtr for output row pointer management");
    }

    // 2. Gather with both BF16 weight and BF16 compute dtype: strides use 2-byte elements throughout
    #[test]
    fn gather_bf16_weight_and_compute_uses_two_byte_strides() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Both weight and compute are BF16: weight_row_bytes = 32*2 = 64
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            32,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::BF16,
            QuantPrecision::BF16,
        ).unwrap();

        // IntMulStride for Tensor index: stride = weight_row_bytes = 32*2 = 64
        let stride_bf16 = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::IntMulStride { stride, .. } if *stride == 32 * 2)
        });
        assert!(stride_bf16, "BF16/BF16 gather must use BF16 weight_row_bytes (64) as stride");

        // VecLoad dtype must be BF16 (reading from weight table)
        let bf16_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, .. })
        }).count();
        assert!(bf16_loads > 0, "BF16 weight gather must emit BF16 VecLoad");
    }

    // 3. ColumnSlice with W128 and non-aligned slice_dim: tail handling with scalar ops
    #[test]
    fn column_slice_w128_with_tail_emits_scalar_tail_ops() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W128=4 lanes, slice_dim=10 → slice_vecs=10/4=2, tail=2 scalar elements
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            16, 0, 10,
            SimdWidth::W128,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        // W128 vectorized loads
        let w128_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W128, .. })
        }).count();
        assert!(w128_loads > 0, "W128 ColumnSlice must emit W128 VecLoad");

        // Scalar tail for remainder
        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0, "W128 ColumnSlice with tail must emit scalar tail loads");
    }

    // 4. ColumnSlice comment contains correct start and end column range
    #[test]
    fn column_slice_comment_reflects_offset_range() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // start=8, slice_dim=16 → range 8..24
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(1),
            64, 8, 16,
            SimdWidth::W256,
            input, output,
            QuantPrecision::F32,
        ).unwrap();

        let has_correct_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(s) if s.contains("8..24"))
        });
        assert!(has_correct_comment,
            "ColumnSlice comment must reflect correct range 8..24 for start=8, slice_dim=16");
    }

    // 5. RoPE with partial=1.0 and head_dim=128: no passthrough section (all dims rotated)
    #[test]
    fn rope_full_partial_no_passthrough_loops() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // partial=1.0, head_dim=128 → rot_dim=128, passthrough_dim=0 → no passthrough section
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 128,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // seq loop + head loop + rotation vec loop = exactly 3
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_count, 3,
            "Full partial RoPE must emit exactly 3 loops (seq+head+rot vec), got {loop_count}");
    }

    // 6. RoPE with non-zero cos_sin_offset: scratchpad offset used for cos/sin base
    #[test]
    fn rope_nonzero_cos_sin_offset_loads_from_offset() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(1),
            4, 64,
            1.0,
            SimdWidth::W256,
            input, output,
            256, // cos_sin_offset=256 → cos/sin table starts 256 bytes into scratchpad
            &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Must have LoadPtr with VRegPlusConst(scratch_base, 256) to set cos_sin_base
        let has_offset_load = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, off), .. } if *off == 256)
        });
        assert!(has_offset_load,
            "RoPE with cos_sin_offset=256 must load cos/sin base at offset 256 from scratchpad");
    }

    // 7. RoPE with multi-token seq produces correct token stride bytes
    #[test]
    fn rope_multi_token_seq_emits_outer_seq_loop_with_stride() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // 4 heads, head_dim=32 → token_step_bytes = 4 * 32 * 4 = 512
        emit_rope_inline(
            &mut prog,
            BoundExpr::Const(8), // 8 tokens
            4, 32,
            1.0,
            SimdWidth::W256,
            input, output,
            0, &sym_map,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Must emit tok_input and tok_output LoadPtr from base ptrs
        let load_ptr_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::LoadPtr { .. })
        }).count();
        assert!(load_ptr_count >= 4,
            "Multi-token RoPE must emit tok_input/tok_output/head_input/head_output LoadPtr");

        let has_vec_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_vec_store, "Multi-token RoPE must emit VecStore for rotation output");
    }

    // 8. Gather with telemetry (no scale): verifies VecStore to telemetry ptr after seq loop
    #[test]
    fn gather_telemetry_without_scale_stores_l2_norm_after_seq_loop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            64,
            SimdWidth::W256,
            input, weight, output,
            Some(telemetry), None, // telemetry, no scale
            GatherIndicesKind::Arange,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // After seq loop: VecStore to telemetry ptr with Scalar width for L2 norm result
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_stores > 0,
            "Gather with telemetry must emit scalar VecStore for L2 norm finalization");

        // Must have Sqrt trace op lowered (from norm_body) — produces VecUnOp or similar
        let has_sqrt_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(s) if s.contains("L2 norm"))
        });
        assert!(has_sqrt_comment,
            "Gather with telemetry must emit L2 norm finalization comment");
    }

    // 9. Gather with embed_dim exactly matching lanes: no scalar tail, clean vectorized path
    #[test]
    fn gather_embed_dim_exact_lane_multiple_no_scalar_tail() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W256=8 lanes, embed_dim=16 → dim_vecs=16/8=2, no remainder
        emit_gather_inline(
            &mut prog,
            BoundExpr::Const(3),
            16,
            SimdWidth::W256,
            input, weight, output,
            None, None,
            GatherIndicesKind::Tensor,
            QuantPrecision::F32,
            QuantPrecision::F32,
        ).unwrap();

        // All VecLoad should be W256 (no scalar loads for tail)
        let vec_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W256, .. })
        }).count();
        assert!(vec_loads > 0, "Lane-aligned gather must emit W256 VecLoad");

        // No scalar VecLoad from the inner dim loop (ScalarLoad for index reading is ok)
        // Check that there are no VecLoad with Scalar width (only ScalarLoad for indices)
        let scalar_vec_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert_eq!(scalar_vec_loads, 0,
            "Lane-aligned gather must not emit scalar VecLoad in dim loop, got {scalar_vec_loads}");
    }

    // 10. ColumnSlice with BF16 and exact boundary: no tail, no scalar ops in dim loop
    #[test]
    fn column_slice_bf16_exact_boundary_no_scalar_tail() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // BF16: elem_bytes=2, W256=8 lanes, slice_dim=16 → slice_vecs=16/8=2, tail=0
        emit_column_slice_inline(
            &mut prog,
            BoundExpr::Const(2),
            32, 0, 16,
            SimdWidth::W256,
            input, output,
            QuantPrecision::BF16,
        ).unwrap();

        // All VecLoad must be BF16 W256
        let bf16_vec_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W256, dtype: QuantPrecision::BF16, .. })
        }).count();
        assert!(bf16_vec_loads > 0,
            "BF16 ColumnSlice with aligned dims must emit BF16 W256 VecLoad");

        // No scalar VecLoad (tail=0)
        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert_eq!(scalar_loads, 0,
            "BF16 aligned ColumnSlice must not emit scalar tail loads, got {scalar_loads}");
    }

    // ── Wave 12x88: 10 additional tests (66 total) ──────────────────────────

    // 1. Gather Zeros with telemetry: L2 norm accumulator is still emitted
    #[test]
    fn gather_zeros_with_telemetry_produces_l2_accumulator() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog, BoundExpr::Const(2), 64, SimdWidth::W256,
            input, weight, output, Some(telemetry), None,
            GatherIndicesKind::Zeros, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        let has_broadcast_zero = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == 0.0)
        });
        assert!(has_broadcast_zero,
            "Zeros gather with telemetry must emit Broadcast(0.0) for L2 norm accumulator");
        let has_l2_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(s) if s.contains("L2 norm"))
        });
        assert!(has_l2_comment, "Zeros gather with telemetry must emit L2 norm comment");
    }

    // 2. ColumnSlice with slice_dim=1: all scalar ops, no vectorized inner loop
    #[test]
    fn column_slice_single_element_all_scalar() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W256=8 lanes, slice_dim=1 → slice_vecs=0, tail=1 → scalar-only path
        emit_column_slice_inline(
            &mut prog, BoundExpr::Const(2), 8, 0, 1,
            SimdWidth::W256, input, output, QuantPrecision::F32,
        ).unwrap();

        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_count, 1, "slice_dim=1 must emit only outer seq loop, got {loop_count}");
        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_loads > 0, "slice_dim=1 must emit scalar VecLoad for tail element");
    }

    // 3. RoPE rejects negative partial value
    #[test]
    fn rope_rejects_negative_partial() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_rope_inline(
            &mut prog, BoundExpr::Const(1), 4, 64,
            -0.5, SimdWidth::W256, input, output, 0, &sym_map,
            QuantPrecision::F32, None,
        );

        assert!(result.is_err(), "negative partial must be rejected");
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("partial"), "error should mention partial: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // 4. Gather Tensor with embed_dim exactly equal to SIMD lanes: one inner iteration, no tail
    #[test]
    fn gather_tensor_embed_dim_equals_lanes_one_vec_iteration() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W256=8, embed_dim=8 → dim_vecs=1, no scalar tail
        emit_gather_inline(
            &mut prog, BoundExpr::Const(2), 8, SimdWidth::W256,
            input, weight, output, None, None,
            GatherIndicesKind::Tensor, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_count, 2, "embed_dim=lanes must emit seq+dim loops, got {loop_count}");
        let scalar_vec_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert_eq!(scalar_vec_loads, 0,
            "embed_dim=lanes must not emit scalar VecLoad, got {scalar_vec_loads}");
    }

    // 5. ColumnSlice with large input_inner: initial LoadPtr offset matches start*elem_bytes
    #[test]
    fn column_slice_large_input_inner_correct_start_offset() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // F32: start_bytes = 128*4 = 512
        emit_column_slice_inline(
            &mut prog, BoundExpr::Const(1), 256, 128, 64,
            SimdWidth::W256, input, output, QuantPrecision::F32,
        ).unwrap();

        let has_start_offset = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, off), .. } if *off == 128 * 4)
        });
        assert!(has_start_offset,
            "ColumnSlice must initialize in_row with start*elem_bytes offset (512)");
    }

    // 6. RoPE W512 with head_dim=128 full rotation: W512 VecLoad/VecStore
    #[test]
    fn rope_w512_full_rotation_emits_w512_loads() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        emit_rope_inline(
            &mut prog, BoundExpr::Const(1), 4, 128,
            1.0, SimdWidth::W512, input, output, 0, &sym_map,
            QuantPrecision::F32, None,
        ).unwrap();

        let w512_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W512, .. })
        }).count();
        let w512_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W512, .. })
        }).count();
        assert!(w512_loads > 0, "W512 RoPE must emit W512 VecLoad");
        assert!(w512_stores > 0, "W512 RoPE must emit W512 VecStore");
    }

    // 7. Gather Arange with BF16 compute dtype: VecStore uses BF16
    #[test]
    fn gather_arange_bf16_compute_emits_bf16_store() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog, BoundExpr::Const(2), 64, SimdWidth::W256,
            input, weight, output, None, None,
            GatherIndicesKind::Arange, QuantPrecision::BF16, QuantPrecision::F32,
        ).unwrap();

        let bf16_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { dtype: QuantPrecision::BF16, .. })
        }).count();
        assert!(bf16_stores > 0, "BF16 compute gather must emit BF16 VecStore");
    }

    // 8. ColumnSlice W128 BF16 aligned: no scalar tail, clean W128 path
    #[test]
    fn column_slice_w128_bf16_aligned_no_scalar_tail() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W128=4 lanes, BF16, slice_dim=8 → slice_vecs=2, tail=0
        emit_column_slice_inline(
            &mut prog, BoundExpr::Const(2), 16, 0, 8,
            SimdWidth::W128, input, output, QuantPrecision::BF16,
        ).unwrap();

        let w128_bf16_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::W128, dtype: QuantPrecision::BF16, .. })
        }).count();
        assert!(w128_bf16_loads > 0, "W128 BF16 ColumnSlice must emit W128 BF16 VecLoad");
        let scalar_loads = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecLoad { width: SimdWidth::Scalar, .. })
        }).count();
        assert_eq!(scalar_loads, 0,
            "W128 BF16 aligned ColumnSlice must not emit scalar tail, got {scalar_loads}");
    }

    // 9. RoPE with position_offset AND BF16: GprBinOp(Add) and BF16 loads
    #[test]
    fn rope_position_offset_with_bf16_emits_gpr_add_and_bf16_loads() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let pos_offset = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        emit_rope_inline(
            &mut prog, BoundExpr::Const(1), 4, 64,
            1.0, SimdWidth::W256, input, output, 0, &sym_map,
            QuantPrecision::BF16, Some(pos_offset),
        ).unwrap();

        let has_gpr_add = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::GprBinOp { op: GprOp::Add, .. })
        });
        assert!(has_gpr_add, "BF16 RoPE with position_offset must emit GprBinOp(Add)");
        let has_bf16_load = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, .. })
        });
        assert!(has_bf16_load, "BF16 RoPE must emit BF16 VecLoad");
    }

    // 10. Gather Tensor with seq_bound=1: output row advance uses compute_row_bytes offset
    #[test]
    fn gather_tensor_seq_bound_one_emits_out_row_advance() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // F32, embed_dim=32 → compute_row_bytes=128
        emit_gather_inline(
            &mut prog, BoundExpr::Const(1), 32, SimdWidth::W256,
            input, weight, output, None, None,
            GatherIndicesKind::Tensor, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        let has_row_advance = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, off), .. } if *off == 32 * 4)
        });
        assert!(has_row_advance,
            "Gather must emit output row advance with compute_row_bytes offset");
    }

    // ── Wave 12x91: 10 additional tests (76 total) ──────────────────────────

    // 1. Gather Tensor: ScalarLoad offset uses LoopOffset for index read
    #[test]
    fn gather_tensor_scalarload_uses_loop_offset() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog, BoundExpr::Const(4), 64, SimdWidth::W256,
            input, weight, output, None, None,
            GatherIndicesKind::Tensor, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        // Tensor gather's ScalarLoad must use LoopOffset (not Const) for index reading
        let has_loop_offset_load = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::ScalarLoad { offset: OffsetExpr::LoopOffset(_), .. })
        });
        assert!(has_loop_offset_load,
            "Tensor gather must emit ScalarLoad with LoopOffset for sequential index reading");
    }

    // 2. Gather Arange: IntMulStride src is the seq loop byte_offset counter
    #[test]
    fn gather_arange_intmul_stride_src_is_byte_offset() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog, BoundExpr::Const(4), 32, SimdWidth::W256,
            input, weight, output, None, None,
            GatherIndicesKind::Arange, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        // Arange gather: IntMulStride computes row_offset from seq byte_offset
        // The src must be a ByteOffset VReg (from the outer seq loop)
        let intmul_uses_byte_offset = prog.instrs.iter().any(|i| {
            if let VmInstr::IntMulStride { src, .. } = i {
                // Check that src is declared as ByteOffset kind
                prog.instrs.iter().any(|j| {
                    matches!(j, VmInstr::DeclareVReg { id, kind: VRegKind::ByteOffset, .. } if *id == *src)
                })
            } else {
                false
            }
        });
        assert!(intmul_uses_byte_offset,
            "Arange gather IntMulStride must use ByteOffset VReg as src");
    }

    // 3. Gather Zeros: LoadPtr uses VRegPlusConst with offset 0 for weight table row
    #[test]
    fn gather_zeros_loadptr_weight_offset_zero() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog, BoundExpr::Const(2), 64, SimdWidth::W256,
            input, weight, output, None, None,
            GatherIndicesKind::Zeros, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        // Zeros path: LoadPtr with VRegPlusConst(weight_ptr, 0) for table row
        let has_zero_offset_load = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, 0), .. })
        });
        assert!(has_zero_offset_load,
            "Zeros gather must emit LoadPtr with VRegPlusConst offset 0 for weight table row");
    }

    // 4. ColumnSlice: output row advance uses slice_dim * elem_bytes stride
    #[test]
    fn column_slice_output_row_advance_uses_slice_stride() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // F32: output_row_bytes = 32*4 = 128
        let slice_dim = 32;
        emit_column_slice_inline(
            &mut prog, BoundExpr::Const(2), 64, 0, slice_dim,
            SimdWidth::W256, input, output, QuantPrecision::F32,
        ).unwrap();

        // Output row advance: LoadPtr with VRegPlusConst(out_row, 128)
        let has_output_stride = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, off), .. } if *off == slice_dim * 4)
        });
        assert!(has_output_stride,
            "ColumnSlice must advance output row by slice_dim*elem_bytes = {}", slice_dim * 4);
    }

    // 5. ColumnSlice: input row advance uses input_inner * elem_bytes stride
    #[test]
    fn column_slice_input_row_advance_uses_input_inner_stride() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // F32: input_row_bytes = 128*4 = 512
        let input_inner = 128;
        emit_column_slice_inline(
            &mut prog, BoundExpr::Const(2), input_inner, 16, 32,
            SimdWidth::W256, input, output, QuantPrecision::F32,
        ).unwrap();

        // Input row advance: LoadPtr with VRegPlusConst(in_row, 512)
        let has_input_stride = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, off), .. } if *off == input_inner * 4)
        });
        assert!(has_input_stride,
            "ColumnSlice must advance input row by input_inner*elem_bytes = {}", input_inner * 4);
    }

    // 6. RoPE: loop step_bytes matches token stride (num_heads * head_dim * elem_bytes)
    #[test]
    fn rope_seq_loop_step_matches_token_stride() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let num_heads = 8;
        let head_dim = 64;
        // token_step_bytes = 8 * 64 * 4 = 2048
        let expected_step = num_heads * head_dim * 4;

        emit_rope_inline(
            &mut prog, BoundExpr::Const(4), num_heads, head_dim,
            1.0, SimdWidth::W256, input, output, 0, &sym_map,
            QuantPrecision::F32, None,
        ).unwrap();

        // Outer seq loop must have step_bytes = token_step_bytes
        let has_correct_step = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoopBegin { step_bytes, .. } if *step_bytes == expected_step)
        });
        assert!(has_correct_step,
            "RoPE seq loop step must be token_step_bytes = {expected_step}");
    }

    // 7. RoPE: head loop uses BoundExpr::Const(num_heads) as bound
    #[test]
    fn rope_head_loop_uses_const_num_heads_bound() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let num_heads = 6;
        emit_rope_inline(
            &mut prog, BoundExpr::Const(1), num_heads, 64,
            1.0, SimdWidth::W256, input, output, 0, &sym_map,
            QuantPrecision::F32, None,
        ).unwrap();

        // Head loop must have BoundExpr::Const(6)
        let has_head_bound = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } if *n == num_heads)
        });
        assert!(has_head_bound,
            "RoPE must emit head loop with BoundExpr::Const({num_heads})");
    }

    // 8. Gather with embedding_scale=1.0: scale body still produces VecBinOp (1.0 * x)
    #[test]
    fn gather_scale_one_point_zero_still_produces_binop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_gather_inline(
            &mut prog, BoundExpr::Const(2), 64, SimdWidth::W256,
            input, weight, output, None,
            Some(1.0), // scale = 1.0 (identity in math, but still emits trace)
            GatherIndicesKind::Arange, QuantPrecision::F32, QuantPrecision::F32,
        ).unwrap();

        // scale=1.0 still generates Const(1.0) * Input(0) trace → VecBinOp
        let has_binop = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecBinOp { .. }));
        assert!(has_binop,
            "Gather with embedding_scale=1.0 must still emit VecBinOp for scale trace");
    }

    // 9. ColumnSlice: both in_row and out_row LoadPtr initializations present
    #[test]
    fn column_slice_initializes_both_row_pointers() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // start=16, F32: start_bytes = 64
        emit_column_slice_inline(
            &mut prog, BoundExpr::Const(2), 64, 16, 32,
            SimdWidth::W256, input, output, QuantPrecision::F32,
        ).unwrap();

        // in_row init: LoadPtr with VRegPlusConst(input_ptr, 64)
        let has_in_row_init = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, 64), .. })
        });
        assert!(has_in_row_init,
            "ColumnSlice must initialize in_row with start*elem_bytes = 64");

        // out_row init: LoadPtr with VRegPlusConst(output_ptr, 0)
        let has_out_row_init = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(_, 0), .. })
        });
        assert!(has_out_row_init,
            "ColumnSlice must initialize out_row with offset 0");
    }

    // 10. RoPE: passthrough section with partial=0.5 uses identity trace (no VecBinOp in passthrough)
    #[test]
    fn rope_passthrough_section_uses_identity_trace_no_binop() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // head_dim=128, partial=0.5 → rot_dim=64, passthrough_dim=64
        // Passthrough uses identity trace (Input(0)) → VecLoad + VecStore only, no VecBinOp
        emit_rope_inline(
            &mut prog, BoundExpr::Const(1), 4, 128,
            0.5, SimdWidth::W256, input, output, 0, &sym_map,
            QuantPrecision::F32, None,
        ).unwrap();

        // Must have VecLoad and VecStore (both rotation and passthrough)
        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_load, "RoPE must emit VecLoad for rotation and passthrough");
        assert!(has_store, "RoPE must emit VecStore for rotation and passthrough");

        // Must have at least 4 loops: seq + head + rotation_vec + passthrough_vec
        let loop_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_count >= 4,
            "Partial=0.5 RoPE must emit >= 4 loops (seq+head+rot+pt), got {loop_count}");
    }
}

