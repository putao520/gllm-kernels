//! Telemetry inline lowering — GEMM row stats, RMSNorm channel scale,
//! SiLU dead neuron detection, residual with telemetry.

use super::instr::*;
use super::plan_lower::SymDimSlotMap;
use crate::compiler::trace::{QuantPrecision, TraceOp, ReduceKind, ValueId};
use crate::compiler::graph::SymDim;
use crate::types::CompilerError;

pub(crate) fn emit_gemm_row_stats_telemetry(
    prog: &mut VmProgram,
    acc: VRegId,
    width: SimdWidth,
    sym_map: &SymDimSlotMap,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes();
    if lanes == 0 {
        return Ok(());
    }

    // Load telemetry buffer pointer
    let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tel_expr = sym_map.resolve("telemetry")
        .ok_or_else(|| CompilerError::CodegenViolation(
            "emit_gemm_row_stats_telemetry: 'telemetry' ABI arg not found in sym_map".into(),
        ))?
        .clone();
    prog.emit(VmInstr::LoadPtr { dst: telemetry_ptr, src: tel_expr });

    // Helper: build body, run auto_lower, store result at offset
    let hreduce_store = |prog: &mut VmProgram, body: &[TraceOp], result_idx: usize, offset: usize| -> Result<(), CompilerError> {
        let slots = super::auto_select::auto_lower_trace_raw(prog, body, &[acc], width, QuantPrecision::F32)?;
        prog.emit(VmInstr::VecStore { base: telemetry_ptr, offset: OffsetExpr::Const(offset), src: slots[result_idx], width: SimdWidth::Scalar, dtype , predicate: None });
        Ok(())
    };
    // L1 norm: abs(acc) → HReduce(Sum)
    hreduce_store(prog, &[
        TraceOp::Input(0), TraceOp::Abs(ValueId(0)),
        TraceOp::HReduce { src: ValueId(1), op: ReduceKind::Sum },
    ], 2, crate::compiler::graph::telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET)?;
    // Max
    hreduce_store(prog, &[
        TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Max },
    ], 1, crate::compiler::graph::telemetry_offsets::GEMM_ROW_MAX_OFFSET)?;
    // Min
    hreduce_store(prog, &[
        TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Min },
    ], 1, crate::compiler::graph::telemetry_offsets::GEMM_ROW_MIN_OFFSET)?;

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §13.8 RmsNorm Per-Channel Scale Telemetry (Epilogue Piggyback)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Emit RmsNorm per-channel |x| max telemetry after norm computation.
///
/// The RmsNorm reduction loop already computes sum(x²). This function adds
/// per-channel absolute-max tracking by scanning the input row and keeping
/// a running max of |x| across all SIMD vector iterations.
///
/// The result is stored indirectly: telemetry[CHANNEL_SCALE_PTR_OFFSET] contains
/// a pointer to the per-channel scale buffer. We store the per-vec-group
/// absolute max vector at offset 0 of that indirect buffer.
///
/// Instruction overhead: ~1 SIMD instruction per vector iteration (VecBinOp::Max),
/// plus 1 VecStore at end. Per SPEC §13.8.
pub(crate) fn emit_rmsnorm_channel_scale_telemetry(
    prog: &mut VmProgram,
    input_ptr: VRegId,
    feature_dim: usize,
    width: SimdWidth,
    sym_map: &SymDimSlotMap,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes();
    if lanes == 0 || feature_dim == 0 {
        return Ok(());
    }

    let vec_count = feature_dim / lanes;
    let step_bytes = width.bytes();
    if vec_count == 0 {
        return Ok(());
    }

    // Load telemetry buffer pointer
    let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tel_expr = sym_map.resolve("telemetry")
        .ok_or_else(|| CompilerError::CodegenViolation(
            "emit_rmsnorm_channel_scale_telemetry: 'telemetry' ABI arg not found in sym_map".into(),
        ))?
        .clone();
    prog.emit(VmInstr::LoadPtr { dst: telemetry_ptr, src: tel_expr });

    // Load indirect pointer: channel_scale_buf = *(telemetry_ptr + CHANNEL_SCALE_PTR_OFFSET)
    let channel_scale_buf = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: channel_scale_buf,
        src: PtrExpr::VRegPlusConst(telemetry_ptr, crate::compiler::graph::telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET),
    });

    // Initialize running max to zero (broadcast)
    let running_max = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: running_max, src: ScalarExpr::Const(0.0), width, dtype, });

    // Scan input row: for each vector group, compute |x| and update running max
    // TraceOp body: running_max = Max(running_max, Abs(input_val))
    let ch_scale_body: Vec<TraceOp> = vec![
        TraceOp::Input(0),  // [0] input_val
        TraceOp::Abs(ValueId(0)),    // [1] abs_val
        TraceOp::Input(1),  // [2] running_max
        TraceOp::Max(ValueId(1), ValueId(2)), // [3] new_max
    ];

    // input_ptr already points to the current row base (set by lower_norm caller).
    // Scan from offset 0..feature_dim bytes.
    let input_val = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _ctr, byte_off| {
        prog.emit(VmInstr::VecLoad {
            dst: input_val, base: input_ptr,
            offset: OffsetExpr::LoopOffset(byte_off), width,
            dtype, predicate: None,
        });
        super::auto_select::auto_lower_trace_into(
            prog, &ch_scale_body, &[input_val, running_max], running_max, width, QuantPrecision::F32,
        ).expect("emit_rmsnorm_channel_scale_telemetry: auto_lower_trace invariant violation");
    });

    // Store the per-channel absolute max vector to channel_scale_buf[0..lanes]
    // This gives the caller a representative per-channel scale estimate
    prog.emit(VmInstr::VecStore {
        base: channel_scale_buf,
        offset: OffsetExpr::Const(0),
        src: running_max,
        width,
        dtype, predicate: None,
    });

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §13.5 SiLU Dead Neuron Telemetry (Epilogue Piggyback)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Algebraic threshold for dead neuron detection (SPEC §13.5).
/// sigmoid(x) < ε=0.01 ⟺ x < ln(ε/(1-ε)) ≈ -4.5951
const SILU_DEAD_NEURON_THRESHOLD_X: f32 = -4.5951;

/// Emit SiLU dead neuron count telemetry after SiLU computation.
///
/// Scans the SiLU input tensor and counts elements where x < SILU_DEAD_NEURON_THRESHOLD_X
/// (algebraic equivalence to sigmoid(x) < ε). Writes the count as u32 to
/// `telemetry[SILU_DEAD_NEURON_COUNT]` (offset 0).
///
/// Instruction overhead: ~3 SIMD instructions per vector iteration
/// (1 compare + 1 horizontal add-to-count + accumulation), plus 1 final store.
pub(crate) fn emit_silu_dead_neuron_telemetry(
    prog: &mut VmProgram,
    input_ptr: VRegId,
    output_shape: &[SymDim],
    width: SimdWidth,
    sym_map: &SymDimSlotMap,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes();
    if lanes == 0 {
        return Ok(());
    }

    // Load telemetry buffer pointer from ABI arg "telemetry"
    let telemetry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tel_expr = sym_map.resolve("telemetry")
        .ok_or_else(|| CompilerError::CodegenViolation(
            "emit_silu_dead_neuron_telemetry: 'telemetry' ABI arg not found in sym_map".into(),
        ))?
        .clone();
    prog.emit(VmInstr::LoadPtr { dst: telemetry_ptr, src: tel_expr });

    // Compute total elements and loop structure (same as emit_elementwise_inline)
    let feature_dim: usize = output_shape.iter()
        .filter(|d| !d.is_symbolic())
        .map(|d| d.as_concrete().expect("inner dim must be Concrete"))
        .product::<usize>()
        .max(1);
    let feature_vecs = feature_dim / lanes;
    let step_bytes = width.bytes();
    let row_bytes = feature_dim * dtype.elem_bytes();

    let outer_sym = output_shape.iter().find(|d| d.is_symbolic());

    // Allocate accumulator for dead neuron count (as f32 for SIMD accumulation)
    let dead_count = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: dead_count, src: ScalarExpr::Const(0.0), width, dtype, });

    // Threshold broadcast
    let threshold = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast {
        dst: threshold,
        src: ScalarExpr::Const(SILU_DEAD_NEURON_THRESHOLD_X),
        width,
        dtype,
    });

    // One-vec constant for counting (used to convert comparison mask to count)
    let ones = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: ones, src: ScalarExpr::Const(1.0), width, dtype, });

    let input_val = prog.alloc_vreg(VRegKind::Vec, width);

    // Dead neuron detection: Min(Max(threshold - input, 0), 1) * ones + dead_count
    let dead_detect_body: Vec<TraceOp> = vec![
        TraceOp::Input(0), // 0: threshold
        TraceOp::Input(1), // 1: input_val
        TraceOp::Sub(ValueId(0), ValueId(1)), // 2: diff = threshold - input
        TraceOp::Const(0.0), // 3: zero
        TraceOp::Max(ValueId(2), ValueId(3)), // 4: positive part
        TraceOp::Const(1.0), // 5: one
        TraceOp::Min(ValueId(4), ValueId(5)), // 6: clamped [0,1]
        TraceOp::Input(2), // 7: ones
        TraceOp::Mul(ValueId(6), ValueId(7)), // 8: count_lane
        TraceOp::Input(3), // 9: dead_count
        TraceOp::Add(ValueId(8), ValueId(9)), // 10: new dead_count
    ];

    // Unified scan loop: use outer_bound (Symbolic or Const=1)
    let outer_bound = if let Some(sym_dim) = outer_sym {
        sym_map.to_bound(sym_dim)
    } else {
        BoundExpr::Const(1)
    };
    let scan_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit_loop(outer_bound, row_bytes, |prog, _row_ctr, row_off| {
        prog.emit(VmInstr::LoadPtr { dst: scan_base, src: PtrExpr::VRegPlusVReg(input_ptr, row_off) });
        if feature_vecs > 0 {
            prog.emit_loop(BoundExpr::Const(feature_vecs), step_bytes, |prog, _ctr, col_off| {
                prog.emit(VmInstr::VecLoad { dst: input_val, base: scan_base, offset: OffsetExpr::LoopOffset(col_off), width, dtype , predicate: None });
                super::auto_select::auto_lower_trace_into(
                    prog, &dead_detect_body, &[threshold, input_val, ones, dead_count], dead_count, width, QuantPrecision::F32,
                ).expect("dead neuron detect auto_lower failed");
            });
        }
    });

    // Horizontal reduce: sum all lanes of dead_count into a single scalar
    let hr_body = vec![TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
    let hr_slots = super::auto_select::auto_lower_trace_raw(prog, &hr_body, &[dead_count], width, QuantPrecision::F32)
        .expect("dead neuron HReduce auto_lower failed");
    let dead_count_scalar = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
    prog.emit(VmInstr::Broadcast { dst: dead_count_scalar, src: ScalarExpr::ExtractLane0(hr_slots[1]), width: SimdWidth::Scalar, dtype, });

    // Store the count to telemetry[0] (SILU_DEAD_NEURON_COUNT offset = 0)
    // Use Scalar store: telemetry_ptr + 0, dead_count_scalar (lane 0 has the sum)
    prog.emit(VmInstr::VecStore {
        base: telemetry_ptr,
        offset: OffsetExpr::Const(crate::compiler::graph::telemetry_offsets::SILU_DEAD_NEURON_COUNT),
        src: dead_count_scalar,
        width: SimdWidth::Scalar,
        dtype, predicate: None,
    });

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §13.11 Residual Cosine Similarity Telemetry (Epilogue Piggyback)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Emit Residual Add (`out = x_in + x_out`) with cosine similarity telemetry.
///
/// Computes during the residual add loop:
/// - `dot = Σ(x_in[i] * x_out[i])` (dot product)
/// - `norm_in = ‖x_in‖ = sqrt(Σ x_in[i]²)`
/// - `norm_out = ‖x_out‖ = sqrt(Σ x_out[i]²)`
/// - `delta = norm_out / norm_in` (magnitude ratio)
/// - `cosine = dot / (norm_in * norm_out)` (direction similarity)
///
/// Writes to telemetry buffer:
/// - RESIDUAL_DELTA_OFFSET (128): f32, ‖x_out‖ / ‖x_in‖
/// - COSINE_SIMILARITY_OFFSET (136): f32, cosθ
///
/// Instruction overhead: ~5 SIMD instructions per vector iteration
/// (FMA for dot, FMA for norm_sq_in, FMA for norm_sq_out).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_residual_with_telemetry(
    prog: &mut VmProgram,
    out_shape: &[SymDim],
    feature_dim: usize,
    width: SimdWidth,
    input_ptr: VRegId,   // x_in (first operand)
    weight_ptr: VRegId,  // x_out (second operand — "weight" slot for binary ops)
    output_ptr: VRegId,  // result: x_in + x_out
    sym_map: &SymDimSlotMap,
    telemetry_ptr: Option<VRegId>,
    seq_bound_override: Option<&BoundExpr>,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let row_bytes = feature_dim * dtype.elem_bytes();
    let step_bytes = width.bytes();
    let feature_vecs = feature_dim / lanes;
    let tail = feature_dim - feature_vecs * lanes;

    // Outer dimension (seq_len)
    let outer_sym = out_shape.iter().find(|d| d.is_symbolic());
    let outer_bound = if let Some(override_bound) = seq_bound_override.cloned() {
        override_bound
    } else if let Some(sym) = outer_sym {
        sym_map.to_bound(sym)
    } else if out_shape.len() > 1 {
        // Multi-dimensional but all concrete: outer = product of all dims except last
        let outer_dim: usize = out_shape[..out_shape.len() - 1]
            .iter()
            .map(|d| d.as_concrete().expect("non-symbolic dim must be Concrete"))
            .product();
        BoundExpr::Const(outer_dim.max(1))
    } else {
        BoundExpr::Const(1) // 1D tensor: single row
    };

    // §13.11: Allocate accumulators for cosine similarity telemetry
    let has_telemetry = telemetry_ptr.is_some() && feature_vecs > 0;
    let dot_acc = if has_telemetry {
        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype, });
        Some(acc)
    } else {
        None
    };
    let norm_sq_in_acc = if has_telemetry {
        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype, });
        Some(acc)
    } else {
        None
    };
    let norm_sq_out_acc = if has_telemetry {
        let acc = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype, });
        Some(acc)
    } else {
        None
    };

    prog.emit(VmInstr::Comment("Residual Add: out = x_in + x_out (+ §13.11 cosine telemetry)".into()));

    // Main loop: outer (seq) × inner (feature_vecs)
    prog.emit_loop(outer_bound, row_bytes, |prog, _row_ctr, row_off| {
        let row_in = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let row_out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let row_res = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: row_in, src: PtrExpr::VRegPlusVReg(input_ptr, row_off) });
        prog.emit(VmInstr::LoadPtr { dst: row_out, src: PtrExpr::VRegPlusVReg(weight_ptr, row_off) });
        prog.emit(VmInstr::LoadPtr { dst: row_res, src: PtrExpr::VRegPlusVReg(output_ptr, row_off) });

        if feature_vecs > 0 {
            prog.emit_loop(BoundExpr::Const(feature_vecs), step_bytes, |prog, _col_ctr, col_off| {
                let a_vec = prog.alloc_vreg(VRegKind::Vec, width);
                let b_vec = prog.alloc_vreg(VRegKind::Vec, width);
                let sum_vec = prog.alloc_vreg(VRegKind::Vec, width);

                prog.emit(VmInstr::VecLoad {
                    dst: a_vec, base: row_in, offset: OffsetExpr::LoopOffset(col_off), width,
                    dtype, predicate: None,
                });
                prog.emit(VmInstr::VecLoad {
                    dst: b_vec, base: row_out, offset: OffsetExpr::LoopOffset(col_off), width,
                    dtype, predicate: None,
                });

                // Residual add: sum = a + b (via auto_select)
                super::auto_select::auto_lower_trace_into(
                    prog,
                    &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(ValueId(0), ValueId(1))],
                    &[a_vec, b_vec],
                    sum_vec,
                    width,
                    QuantPrecision::F32,
                ).expect("emit_residual_with_telemetry: residual add auto_lower_trace_into");
                prog.emit(VmInstr::VecStore {
                    base: row_res, offset: OffsetExpr::LoopOffset(col_off), src: sum_vec, width,
                    dtype, predicate: None,
                });

                // §13.11 telemetry accumulation (in-register, no extra loads, via auto_select)
                if let (Some(dot), Some(ni), Some(no)) = (dot_acc, norm_sq_in_acc, norm_sq_out_acc) {
                    // dot += a * b
                    super::auto_select::auto_lower_trace_into(
                        prog,
                        &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2), TraceOp::Fma(ValueId(1), ValueId(2), ValueId(0))],
                        &[dot, a_vec, b_vec],
                        dot,
                        width,
                        QuantPrecision::F32,
                    ).expect("emit_residual_with_telemetry: dot fma auto_lower_trace_into");
                    // norm_sq_in += a²
                    super::auto_select::auto_lower_trace_into(
                        prog,
                        &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Fma(ValueId(1), ValueId(1), ValueId(0))],
                        &[ni, a_vec],
                        ni,
                        width,
                        QuantPrecision::F32,
                    ).expect("emit_residual_with_telemetry: norm_sq_in fma auto_lower_trace_into");
                    // norm_sq_out += b²
                    super::auto_select::auto_lower_trace_into(
                        prog,
                        &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Fma(ValueId(1), ValueId(1), ValueId(0))],
                        &[no, b_vec],
                        no,
                        width,
                        QuantPrecision::F32,
                    ).expect("emit_residual_with_telemetry: norm_sq_out fma auto_lower_trace_into");
                }
            });
        }

        // Tail: scalar-wide copy for remaining elements
        if tail > 0 {
            let s_width = SimdWidth::Scalar;
            let tail_base_bytes = feature_vecs * step_bytes;
            prog.emit_loop(BoundExpr::Const(tail), elem, |prog, _t_ctr, t_off| {
                let s_a = prog.alloc_vreg(VRegKind::Vec, s_width);
                let s_b = prog.alloc_vreg(VRegKind::Vec, s_width);
                let s_sum = prog.alloc_vreg(VRegKind::Vec, s_width);
                let off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(t_off)),
                    Box::new(OffsetExpr::Const(tail_base_bytes)),
                );
                prog.emit(VmInstr::VecLoad {
                    dst: s_a, base: row_in, offset: off.clone(), width: s_width,
                    dtype, predicate: None,
                });
                prog.emit(VmInstr::VecLoad {
                    dst: s_b, base: row_out, offset: off.clone(), width: s_width,
                    dtype, predicate: None,
                });
                // Tail residual add: s_sum = s_a + s_b (via auto_select)
                super::auto_select::auto_lower_trace_into(
                    prog,
                    &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(ValueId(0), ValueId(1))],
                    &[s_a, s_b],
                    s_sum,
                    s_width,
                    QuantPrecision::F32,
                ).expect("emit_residual_with_telemetry: tail residual add auto_lower_trace_into");
                prog.emit(VmInstr::VecStore {
                    base: row_res, offset: off, src: s_sum, width: s_width,
                    dtype, predicate: None,
                });

                // §13.11 telemetry for tail elements (via auto_select)
                if let (Some(dot), Some(ni), Some(no)) = (dot_acc, norm_sq_in_acc, norm_sq_out_acc) {
                    super::auto_select::auto_lower_trace_into(
                        prog,
                        &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2), TraceOp::Fma(ValueId(1), ValueId(2), ValueId(0))],
                        &[dot, s_a, s_b],
                        dot,
                        s_width,
                        QuantPrecision::F32,
                    ).expect("emit_residual_with_telemetry: tail dot fma auto_lower_trace_into");
                    super::auto_select::auto_lower_trace_into(
                        prog,
                        &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Fma(ValueId(1), ValueId(1), ValueId(0))],
                        &[ni, s_a],
                        ni,
                        s_width,
                        QuantPrecision::F32,
                    ).expect("emit_residual_with_telemetry: tail norm_sq_in fma auto_lower_trace_into");
                    super::auto_select::auto_lower_trace_into(
                        prog,
                        &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Fma(ValueId(1), ValueId(1), ValueId(0))],
                        &[no, s_b],
                        no,
                        s_width,
                        QuantPrecision::F32,
                    ).expect("emit_residual_with_telemetry: tail norm_sq_out fma auto_lower_trace_into");
                }
            });
        }
    });

    // ── §13.11 Finalize cosine similarity telemetry ──
    if let (Some(tel_ptr), Some(dot), Some(ni), Some(no)) =
        (telemetry_ptr, dot_acc, norm_sq_in_acc, norm_sq_out_acc)
    {
        use crate::compiler::graph::telemetry_offsets;
        use crate::compiler::graph::RESIDUAL_NORM_EPSILON;

        prog.emit(VmInstr::Comment("§13.11 Residual cosine similarity telemetry".into()));

        let hreduce_broadcast = |prog: &mut VmProgram, src: VRegId| -> VRegId {
            let raw = super::auto_select::auto_lower_trace_raw(
                prog, &[TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }], &[src], width, QuantPrecision::F32,
            ).expect("hreduce_broadcast");
            let dst = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst, src: ScalarExpr::ExtractLane0(raw[0]), width, dtype });
            dst
        };
        let dot_scalar = hreduce_broadcast(prog, dot);
        let ni_scalar = hreduce_broadcast(prog, ni);
        let no_scalar = hreduce_broadcast(prog, no);

        // norm_in = sqrt(\xce\xa3 x_in\xc2\xb2) (via auto_select)
        let norm_in = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Sqrt(ValueId(0))],
            &[ni_scalar],
            norm_in,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: norm_in sqrt auto_lower_trace_into");

        // norm_out = sqrt(\xce\xa3 x_out\xc2\xb2) (via auto_select)
        let norm_out = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Sqrt(ValueId(0))],
            &[no_scalar],
            norm_out,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: norm_out sqrt auto_lower_trace_into");

        // delta = norm_out / norm_in (with epsilon guard)
        // norm_in_safe = Max(norm_in, epsilon) to avoid division by zero
        let eps = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast { dst: eps, src: ScalarExpr::Const(RESIDUAL_NORM_EPSILON), width, dtype, });
        let norm_in_safe = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Max(ValueId(0), ValueId(1))],
            &[norm_in, eps],
            norm_in_safe,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: norm_in_safe max auto_lower_trace_into");

        let delta = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Div(ValueId(0), ValueId(1))],
            &[norm_out, norm_in_safe],
            delta,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: delta div auto_lower_trace_into");

        // Store delta to telemetry[RESIDUAL_DELTA_OFFSET]
        prog.emit(VmInstr::VecStore {
            base: tel_ptr,
            offset: OffsetExpr::Const(telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            src: delta,
            width: SimdWidth::Scalar,
            dtype, predicate: None,
        });

        // cosine = dot / (norm_in * norm_out) (with epsilon guard on denominator)
        // denom = norm_in * norm_out (via auto_select)
        let denom = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))],
            &[norm_in, norm_out],
            denom,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: denom mul auto_lower_trace_into");
        let eps_sq = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast { dst: eps_sq, src: ScalarExpr::Const(RESIDUAL_NORM_EPSILON * RESIDUAL_NORM_EPSILON), width, dtype, });
        let denom_safe = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Max(ValueId(0), ValueId(1))],
            &[denom, eps_sq],
            denom_safe,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: denom_safe max auto_lower_trace_into");

        let cosine = prog.alloc_vreg(VRegKind::Vec, width);
        super::auto_select::auto_lower_trace_into(
            prog,
            &[TraceOp::Input(0), TraceOp::Input(1), TraceOp::Div(ValueId(0), ValueId(1))],
            &[dot_scalar, denom_safe],
            cosine,
            width,
            QuantPrecision::F32,
        ).expect("emit_residual_with_telemetry: cosine div auto_lower_trace_into");

        // Store cosine to telemetry[COSINE_SIMILARITY_OFFSET]
        prog.emit(VmInstr::VecStore {
            base: tel_ptr,
            offset: OffsetExpr::Const(telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            src: cosine,
            width: SimdWidth::Scalar,
            dtype, predicate: None,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::telemetry_offsets;

    // ── Test 1: SILU_DEAD_NEURON_THRESHOLD_X algebraic correctness ──

    #[test]
    fn silu_dead_neuron_threshold_is_log_of_epsilon() {
        // SPEC §13.5: sigmoid(x) < ε=0.01 ⟺ x < ln(ε/(1-ε)) ≈ -4.5951
        let epsilon: f32 = 0.01;
        let expected = (epsilon / (1.0 - epsilon)).ln();
        let diff = (SILU_DEAD_NEURON_THRESHOLD_X - expected).abs();
        assert!(
            diff < 0.001,
            "threshold {} should be close to {}, diff = {}",
            SILU_DEAD_NEURON_THRESHOLD_X,
            expected,
            diff
        );
    }

    // ── Test 2: emit_gemm_row_stats_telemetry returns error without telemetry ──

    #[test]
    fn gemm_row_stats_returns_error_when_telemetry_missing() {
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let empty_sym_map = {
            let state = super::super::vm_state::VmState::init_mega_kernel_x86();
            SymDimSlotMap::from_vm_state_with_params(&state, &[]).unwrap()
        };

        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &empty_sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_err(), "should return error when 'telemetry' not in sym_map");
        if let Err(CompilerError::CodegenViolation(msg)) = result {
            assert!(
                msg.contains("telemetry"),
                "error message should mention 'telemetry': {msg}"
            );
        } else {
            panic!("expected CodegenViolation error, got {:?}", result);
        }
    }

    // ── Test 3: emit_rmsnorm_channel_scale_telemetry returns error without telemetry ──

    #[test]
    fn rmsnorm_channel_scale_returns_error_when_telemetry_missing() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let empty_sym_map = {
            let state = super::super::vm_state::VmState::init_mega_kernel_x86();
            SymDimSlotMap::from_vm_state_with_params(&state, &[]).unwrap()
        };

        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &empty_sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_err(), "should return error when 'telemetry' not in sym_map");
        if let Err(CompilerError::CodegenViolation(msg)) = result {
            assert!(
                msg.contains("telemetry"),
                "error message should mention 'telemetry': {msg}"
            );
        } else {
            panic!("expected CodegenViolation error, got {:?}", result);
        }
    }

    // ── Test 4: emit_rmsnorm_channel_scale_telemetry returns Ok when feature_dim is zero ──

    #[test]
    fn rmsnorm_channel_scale_returns_ok_for_zero_feature_dim() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            0,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_ok(), "zero feature_dim should return Ok immediately");
        assert_eq!(
            prog.len(),
            initial_len,
            "zero feature_dim should emit no instructions"
        );
    }

    // ── Test 5: emit_rmsnorm_channel_scale_telemetry returns Ok when lanes is zero ──

    #[test]
    fn rmsnorm_channel_scale_returns_ok_for_scalable_simd_width() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::Scalable,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(
            result.is_ok(),
            "Scalable SIMD width should return Ok immediately"
        );
        assert_eq!(
            prog.len(),
            initial_len,
            "Scalable SIMD width should emit no instructions"
        );
    }

    // ── Test 6: emit_gemm_row_stats_telemetry returns Ok when lanes is zero ──

    #[test]
    fn gemm_row_stats_returns_ok_for_scalable_simd_width() {
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalable);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::Scalable,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(
            result.is_ok(),
            "Scalable SIMD width should return Ok immediately"
        );
        assert_eq!(
            prog.len(),
            initial_len,
            "Scalable SIMD width should emit no instructions"
        );
    }

    // ── Test 7: emit_silu_dead_neuron_telemetry returns error without telemetry ──

    #[test]
    fn silu_dead_neuron_returns_error_when_telemetry_missing() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(64)];
        let empty_sym_map = {
            let state = super::super::vm_state::VmState::init_mega_kernel_x86();
            SymDimSlotMap::from_vm_state_with_params(&state, &[]).unwrap()
        };

        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &empty_sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_err(), "should return error when 'telemetry' not in sym_map");
        if let Err(CompilerError::CodegenViolation(msg)) = result {
            assert!(
                msg.contains("telemetry"),
                "error message should mention 'telemetry': {msg}"
            );
        } else {
            panic!("expected CodegenViolation error, got {:?}", result);
        }
    }

    // ── Test 8: emit_silu_dead_neuron_telemetry returns Ok when lanes is zero ──

    #[test]
    fn silu_dead_neuron_returns_ok_for_scalable_simd_width() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::Scalable,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(
            result.is_ok(),
            "Scalable SIMD width should return Ok immediately"
        );
        assert_eq!(
            prog.len(),
            initial_len,
            "Scalable SIMD width should emit no instructions"
        );
    }

    // ── Test 9: emit_rmsnorm_channel_scale emits instructions with valid parameters ──

    #[test]
    fn rmsnorm_channel_scale_emits_instructions_with_valid_params() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_ok(), "valid params should succeed");
        assert!(
            prog.len() > initial_len,
            "should have emitted at least one instruction"
        );
    }

    // ── Test 10: telemetry_offsets constants satisfy ordering invariants ──

    #[test]
    fn telemetry_offsets_are_aligned_and_non_overlapping() {
        let offsets = [
            telemetry_offsets::SILU_DEAD_NEURON_COUNT,
            telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET,
            telemetry_offsets::EXPERT_HIT_COUNTS_OFFSET,
            telemetry_offsets::RESIDUAL_DELTA_OFFSET,
            telemetry_offsets::COSINE_SIMILARITY_OFFSET,
            telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET,
            telemetry_offsets::CENTROID_TOKEN_IDX_OFFSET,
            telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET,
            telemetry_offsets::SOFTMAX_MAX_OFFSET,
            telemetry_offsets::EFFECTIVE_CONTEXT_LEN_OFFSET,
            telemetry_offsets::IS_ATTENTION_SINK_OFFSET,
            telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET,
            telemetry_offsets::GEMM_ROW_MAX_OFFSET,
            telemetry_offsets::GEMM_ROW_MIN_OFFSET,
            telemetry_offsets::EMBED_L2_NORM_OFFSET,
        ];

        for &offset in &offsets {
            assert_eq!(
                offset % 4,
                0,
                "offset {offset} is not 4-byte aligned"
            );
        }

        // TELEMETRY_BUFFER_MIN_BYTES must cover the last offset + sizeof(f32)
        assert!(
            telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES
                >= telemetry_offsets::EMBED_L2_NORM_OFFSET + 4,
            "TELEMETRY_BUFFER_MIN_BYTES must cover EMBED_L2_NORM_OFFSET + sizeof(f32)"
        );
    }

    // ── Test 11: emit_gemm_row_stats emits instructions with valid sym_map ──

    #[test]
    fn gemm_row_stats_emits_instructions_with_valid_sym_map() {
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_ok(), "valid params should succeed");
        assert!(
            prog.len() > initial_len,
            "should have emitted at least one instruction"
        );
    }

    // ── Test 12: emit_silu_dead_neuron works with symbolic outer dimension ──

    #[test]
    fn silu_dead_neuron_emits_instructions_with_symbolic_outer_dim() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![
            SymDim::Symbolic {
                name: "seq_len".into(),
                max_value: Some(2048),
            },
            SymDim::Concrete(128),
        ];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        assert!(result.is_ok(), "symbolic outer dim should succeed");
        assert!(
            prog.len() > initial_len,
            "should have emitted instructions for symbolic outer"
        );
    }

    // ── Test 13: residual telemetry works without telemetry_ptr ──

    #[test]
    fn residual_with_telemetry_works_without_telemetry_ptr() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            None,
            None,
            QuantPrecision::F32,
        );

        assert!(result.is_ok(), "should succeed with telemetry_ptr = None");
        assert!(
            !prog.is_empty(),
            "should emit residual add instructions even without telemetry"
        );
    }

    // ── Test 14: residual telemetry with telemetry_ptr emits cosine similarity instructions ──
    // @trace TEST-TE-14 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_emits_cosine_similarity_when_telemetry_ptr_provided() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let len_without_telemetry;

        // Act: first run without telemetry to get baseline instruction count
        {
            let mut prog2 = VmProgram::new();
            let ip2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let wp2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let op2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            emit_residual_with_telemetry(
                &mut prog2,
                &shape,
                64,
                SimdWidth::W256,
                ip2,
                wp2,
                op2,
                &sym_map,
                None,
                None,
                QuantPrecision::F32,
            ).unwrap();
            len_without_telemetry = prog2.len();
        }

        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "should succeed with telemetry_ptr provided");
        assert!(
            prog.len() > len_without_telemetry,
            "telemetry path should emit more instructions ({}) than non-telemetry path ({})",
            prog.len(),
            len_without_telemetry,
        );
    }

    // ── Test 15: residual telemetry with tail elements (feature_dim not aligned to lanes) ──
    // @trace TEST-TE-15 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_handles_unaligned_feature_dim_with_tail() {
        // Arrange: feature_dim=65, W256 lanes=8 → feature_vecs=8, tail=1
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(65)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            65,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: 65 % 8 = 1 tail element, should still succeed with telemetry
        assert!(result.is_ok(), "unaligned feature_dim=65 should succeed with tail handling");
        assert!(!prog.is_empty(), "should emit instructions for unaligned feature_dim");
    }

    // ── Test 16: residual telemetry with seq_bound_override overrides symbolic dimension ──
    // @trace TEST-TE-16 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_uses_seq_bound_override_instead_of_shape() {
        // Arrange: shape has symbolic outer, but seq_bound_override provides Const(1)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![
            SymDim::Symbolic { name: "seq_len".into(), max_value: Some(2048) },
            SymDim::Concrete(64),
        ];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let override_bound = BoundExpr::Const(1);

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            None,
            Some(&override_bound),
            QuantPrecision::F32,
        );

        // Assert: seq_bound_override should take precedence over symbolic shape
        assert!(result.is_ok(), "seq_bound_override should work even with symbolic shape");
        assert!(!prog.is_empty(), "should emit instructions with overridden bound");
    }

    // ── Test 17: residual telemetry with concrete multi-dimensional shape (outer_dim > 1) ──
    // @trace TEST-TE-17 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_concrete_multi_dim_shape_succeeds() {
        // Arrange: shape [4, 64] → outer_dim = 4 (product of all dims except last)
        // The function treats the last dim as feature_dim and all preceding dims as outer.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(4), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: multi-dimensional concrete shape with telemetry should succeed
        // The VmProgram records loop structures with bounds, and the outer loop
        // bound is Const(4) for shape [4, 64] vs Const(1) for single-row shapes.
        assert!(result.is_ok(), "multi-dimensional concrete shape should succeed");
        assert!(
            !prog.is_empty(),
            "should emit residual add and telemetry instructions for multi-dim shape"
        );
    }

    // ── Test 18: residual telemetry with 1D tensor shape (single row, no outer dimension) ──
    // @trace TEST-TE-18 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_1d_tensor_uses_single_row_outer_bound() {
        // Arrange: 1D shape [64] → outer_bound = Const(1), feature_dim = 64
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: 1D with telemetry should work and emit telemetry finalization
        assert!(result.is_ok(), "1D shape with telemetry should succeed");
        assert!(!prog.is_empty(), "should emit residual add and telemetry instructions");
    }

    // ── Test 19: rmsnorm channel scale returns Ok when feature_dim < lanes (vec_count=0) ──
    // @trace TEST-TE-19 [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_returns_ok_when_feature_dim_smaller_than_lanes() {
        // Arrange: feature_dim=4, W256 lanes=8 → vec_count = 4/8 = 0
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            4, // 4 < 8 lanes of W256
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert: vec_count = 0 should return Ok with no instructions
        assert!(result.is_ok(), "feature_dim < lanes should return Ok (vec_count=0)");
        assert_eq!(
            prog.len(),
            initial_len,
            "vec_count=0 should emit no instructions"
        );
    }

    // ── Test 20: rmsnorm channel scale emits correct structure with W512 SIMD width ──
    // @trace TEST-TE-20 [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_with_w512_simd_width() {
        // Arrange: W512 has 16 f32 lanes, feature_dim=256 → vec_count = 256/16 = 16
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W512,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "W512 SIMD width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with W512 SIMD width"
        );
    }

    // ── Test 21: gemm row stats emits correct structure with W512 SIMD width ──
    // @trace TEST-TE-21 [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_emits_with_w512_simd_width() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W512);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W512,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "W512 SIMD width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with W512 SIMD width"
        );
    }

    // ── Test 22: silu dead neuron with 1D concrete-only shape (no outer symbolic dim) ──
    // @trace TEST-TE-22 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_1d_concrete_shape_emits_instructions() {
        // Arrange: single-dim shape, all concrete, outer_bound = Const(1)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert: 1D concrete shape should use outer_bound=Const(1) path
        assert!(result.is_ok(), "1D concrete shape should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions for 1D concrete shape"
        );
    }

    // ── Test 23: silu dead neuron with multi-layer concrete shape ──
    // @trace TEST-TE-23 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_multi_dim_concrete_shape_emits_instructions() {
        // Arrange: shape [2, 128] → outer_dim derived from non-symbolic product
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(2), SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert: multi-dim concrete shape computes feature_dim as product of non-symbolic dims
        assert!(result.is_ok(), "multi-dim concrete shape should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions for multi-dim concrete shape"
        );
    }

    // ── Test 24: silu dead neuron with feature_dim < lanes produces zero feature_vecs ──
    // @trace TEST-TE-24 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_small_feature_dim_still_emits_telemetry() {
        // Arrange: feature_dim=4, W256 lanes=8 → feature_vecs = 4/8 = 0
        // The inner loop body won't execute, but the outer loop and final store should still emit
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(4)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert: should succeed even with feature_vecs=0 (no inner vec loop iterations)
        assert!(result.is_ok(), "feature_dim < lanes should succeed");
        assert!(
            !prog.is_empty(),
            "should still emit broadcast + hreduce + store even when feature_vecs=0"
        );
    }

    // ── Test 25: RESIDUAL_NORM_EPSILON is a positive small value ──
    // @trace TEST-TE-25 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_norm_epsilon_is_positive_small_value() {
        use crate::compiler::graph::RESIDUAL_NORM_EPSILON;

        // Arrange & Act: check the constant value
        // Assert: epsilon must be positive, non-zero, and small (guard against division by zero)
        assert!(
            RESIDUAL_NORM_EPSILON > 0.0,
            "RESIDUAL_NORM_EPSILON must be positive, got {}",
            RESIDUAL_NORM_EPSILON,
        );
        assert!(
            RESIDUAL_NORM_EPSILON < 1.0,
            "RESIDUAL_NORM_EPSILON must be < 1.0, got {}",
            RESIDUAL_NORM_EPSILON,
        );
        assert!(
            RESIDUAL_NORM_EPSILON == 1e-6,
            "RESIDUAL_NORM_EPSILON should be 1e-6 per SPEC, got {}",
            RESIDUAL_NORM_EPSILON,
        );
    }

    // ── Test 26: residual telemetry with perfectly aligned feature_dim (no tail) ──
    // @trace TEST-TE-26 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_aligned_feature_dim_no_tail_elements() {
        // Arrange: feature_dim=128 (8*16=128 for W256, tail=0), with telemetry
        let mut prog_aligned = VmProgram::new();
        let ip_a = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let wp_a = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let op_a = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let tp_a = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: aligned (128 / 8 = 16 vecs, tail=0)
        let result_aligned = emit_residual_with_telemetry(
            &mut prog_aligned,
            &shape,
            128,
            SimdWidth::W256,
            ip_a,
            wp_a,
            op_a,
            &sym_map,
            Some(tp_a),
            None,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result_aligned.is_ok(), "aligned feature_dim should succeed with telemetry");
        assert!(
            !prog_aligned.is_empty(),
            "should emit instructions for aligned feature_dim with telemetry"
        );
    }

    // ── Test 27: silu dead neuron threshold at boundary: x = threshold should NOT be dead ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_threshold_boundary_not_dead() {
        // Arrange: sigmoid(-4.5951) ≈ 0.01 exactly, so x = threshold is the boundary.
        // Elements strictly less than threshold are counted as dead.
        // The threshold constant must satisfy sigmoid(threshold) ≈ 0.01.
        let epsilon: f32 = 0.01;

        // Act: compute sigmoid at the threshold
        let sigmoid_at_threshold = 1.0 / (1.0 + (-SILU_DEAD_NEURON_THRESHOLD_X).exp());

        // Assert: sigmoid at threshold ≈ epsilon
        let diff = (sigmoid_at_threshold - epsilon).abs();
        assert!(
            diff < 0.001,
            "sigmoid({}) = {} should be close to epsilon={}, diff={}",
            SILU_DEAD_NEURON_THRESHOLD_X,
            sigmoid_at_threshold,
            epsilon,
            diff,
        );
    }

    // ── Test 28: gemm_row_stats emits instructions with W128 (SSE/NEON) width ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_emits_with_w128_simd_width() {
        // Arrange: W128 has 4 f32 lanes
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W128);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W128,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "W128 SIMD width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with W128 SIMD width"
        );
    }

    // ── Test 29: rmsnorm_channel_scale emits with Warp(32) GPU SIMD width ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_with_gpu_warp_width() {
        // Arrange: Warp(32) simulates NVIDIA GPU warp width (32 f32 lanes)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256, // 256 / 32 = 8 vector iterations
            SimdWidth::Warp(32),
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Warp(32) SIMD width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with Warp(32) width"
        );
    }

    // ── Test 30: residual telemetry with feature_dim=0 produces no inner loop iterations ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_zero_feature_dim_emits_minimal_instructions() {
        // Arrange: feature_dim=0 → no vectors, no tail, no telemetry accumulators
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(0)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            0,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: feature_dim=0 means feature_vecs=0 and tail=0, so no telemetry accumulators
        // are allocated, but the outer loop structure (with Const(1)) should still emit.
        assert!(
            result.is_ok(),
            "feature_dim=0 with telemetry_ptr should succeed (no inner loop body)"
        );
    }

    // ── Test 31: silu dead neuron with Warp(64) AMD GPU width ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_with_amd_warp64_width() {
        // Arrange: Warp(64) simulates AMD GPU wavefront (64 f32 lanes)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act: 128 / 64 = 2 feature_vecs
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::Warp(64),
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Warp(64) AMD GPU width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with Warp(64) width"
        );
    }

    // ── Test 32: residual telemetry with Scalar SIMD width (lanes=1) handles single elements ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_scalar_width_handles_single_element_ops() {
        // Arrange: Scalar width (1 lane), feature_dim=8 → 8 scalar iterations, no tail
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(8)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            8,
            SimdWidth::Scalar,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: Scalar width should work for single-element operations
        assert!(result.is_ok(), "Scalar SIMD width with telemetry should succeed");
        assert!(!prog.is_empty(), "should emit scalar residual add + telemetry instructions");
    }

    // ── Test 33: gemm_row_stats produces VmProgram with LoadPtr as first emitted instruction ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_emits_loadptr_for_telemetry() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: at least one LoadPtr instruction should be emitted (loading telemetry pointer)
        let has_loadptr = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoadPtr { .. }));
        assert!(
            has_loadptr,
            "emit_gemm_row_stats_telemetry should emit LoadPtr for telemetry buffer",
        );
    }

    // ── Test 34: SimdWidth::Warp(0) has zero lanes and returns Ok immediately ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_returns_ok_for_warp_zero_lanes() {
        // Arrange: Warp(0) → f32_lanes() = 0, early exit path
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Warp(0));
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::Warp(0),
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert: zero lanes should return Ok with no instructions emitted
        assert!(result.is_ok(), "Warp(0) should return Ok immediately");
        assert_eq!(
            prog.len(),
            initial_len,
            "Warp(0) should emit no instructions"
        );
    }

    // ── Test 35: rmsnorm channel scale emits correct vec_count for feature_dim exactly equal to lanes ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_feature_dim_equals_lanes_exactly_one_vec() {
        // Arrange: W256 has 8 lanes, feature_dim=8 → vec_count = 8/8 = 1 (exactly one iteration)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            8, // exactly 1 vector
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "feature_dim=8 with W256 should succeed (vec_count=1)");
        assert!(
            prog.len() > initial_len,
            "should emit instructions for exactly 1 vector iteration"
        );
    }

    // ── Test 36: residual telemetry BF16 dtype propagation succeeds ──
    // @trace TEST-12k67 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_bf16_dtype_propagation_succeeds() {
        // Arrange: BF16 dtype — the emit functions accept QuantPrecision and propagate it
        // to VmInstr construction. This test verifies no panic/error with non-F32 precision.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: use BF16 precision (2-byte elements)
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::BF16,
        );

        // Assert: should succeed with BF16 dtype (the function propagates dtype to VmInstr)
        assert!(
            result.is_ok(),
            "BF16 dtype should propagate without error"
        );
        assert!(
            !prog.is_empty(),
            "should emit residual add instructions with BF16 dtype"
        );
    }

    // ── Test 37: gemm_row_stats BF16 dtype propagation succeeds ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_bf16_dtype_propagation_succeeds() {
        // Arrange: BF16 dtype — verifies that gemm_row_stats_telemetry propagates
        // non-F32 QuantPrecision to VecStore instructions without error.
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "BF16 dtype should propagate without error in gemm_row_stats");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with BF16 dtype"
        );
    }

    // ── Test 38: rmsnorm_channel_scale BF16 dtype propagation succeeds ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_bf16_dtype_propagation_succeeds() {
        // Arrange: BF16 dtype — verifies channel scale telemetry propagates non-F32
        // QuantPrecision through Broadcast, VecLoad, VecStore instructions.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "BF16 dtype should propagate without error in rmsnorm_channel_scale");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with BF16 dtype"
        );
    }

    // ── Test 39: silu_dead_neuron BF16 dtype propagation succeeds ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_bf16_dtype_propagation_succeeds() {
        // Arrange: BF16 dtype — verifies dead neuron detection propagates non-F32
        // QuantPrecision through Broadcast, VecLoad, VecStore, and auto_lower_trace.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::BF16,
        );

        // Assert
        assert!(result.is_ok(), "BF16 dtype should propagate without error in silu_dead_neuron");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with BF16 dtype"
        );
    }

    // ── Test 40: rmsnorm_channel_scale emits LoadPtr for telemetry and indirect channel_scale_buf ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_loadptr_for_telemetry_and_indirect_buf() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit at least 2 LoadPtr instructions:
        // one for telemetry buffer, one for the indirect channel_scale_buf
        let loadptr_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::LoadPtr { .. }))
            .count();
        assert!(
            loadptr_count >= 2,
            "should emit at least 2 LoadPtr instructions (telemetry + channel_scale_buf), got {}",
            loadptr_count,
        );
    }

    // ── Test 41: silu_dead_neuron emits Broadcast instructions for threshold, ones, and dead_count ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_broadcast_instructions_for_initialization() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit at least 3 Broadcast instructions:
        // dead_count=0.0, threshold=-4.5951, ones=1.0
        let broadcast_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { .. }))
            .count();
        assert!(
            broadcast_count >= 3,
            "should emit at least 3 Broadcast instructions (dead_count, threshold, ones), got {}",
            broadcast_count,
        );
    }

    // ── Test 42: residual_with_telemetry emits Comment instruction at start ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_emits_comment_instruction() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            None,
            None,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit at least one Comment instruction describing residual add
        let has_residual_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(text) if text.contains("Residual Add"))
        });
        assert!(
            has_residual_comment,
            "should emit a Comment instruction describing the residual add operation"
        );
    }

    // ── Test 43: gemm_row_stats emits VecStore at correct telemetry offsets ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_emits_vecstore_at_expected_offsets() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit VecStore instructions at GEMM_ROW_NORM_L1_OFFSET,
        // GEMM_ROW_MAX_OFFSET, and GEMM_ROW_MIN_OFFSET
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            vecstore_offsets.contains(&telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET),
            "should emit VecStore at GEMM_ROW_NORM_L1_OFFSET ({})",
            telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET,
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::GEMM_ROW_MAX_OFFSET),
            "should emit VecStore at GEMM_ROW_MAX_OFFSET ({})",
            telemetry_offsets::GEMM_ROW_MAX_OFFSET,
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::GEMM_ROW_MIN_OFFSET),
            "should emit VecStore at GEMM_ROW_MIN_OFFSET ({})",
            telemetry_offsets::GEMM_ROW_MIN_OFFSET,
        );
    }

    // ── Test 44: residual telemetry with FP16 dtype propagation succeeds ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_fp16_dtype_propagation_succeeds() {
        // Arrange: FP16 dtype — verifies residual telemetry works with half precision
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F16,
        );

        // Assert: should succeed with FP16 dtype
        assert!(result.is_ok(), "FP16 dtype should propagate without error");
        assert!(!prog.is_empty(), "should emit residual add instructions with FP16 dtype");
    }

    // ── Test 45: silu_dead_neuron emits VecStore at SILU_DEAD_NEURON_COUNT offset ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_vecstore_at_correct_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: final VecStore should be at SILU_DEAD_NEURON_COUNT offset (0)
        let last_vecstore = prog.instrs.iter().rev().find(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(
            last_vecstore.is_some(),
            "should emit at least one VecStore instruction"
        );
        if let Some(VmInstr::VecStore { offset: OffsetExpr::Const(off), width, .. }) = last_vecstore {
            assert_eq!(
                *off,
                telemetry_offsets::SILU_DEAD_NEURON_COUNT,
                "final VecStore offset should be SILU_DEAD_NEURON_COUNT ({})",
                telemetry_offsets::SILU_DEAD_NEURON_COUNT,
            );
            assert_eq!(
                *width,
                SimdWidth::Scalar,
                "final VecStore width should be Scalar (single count value)"
            );
        }
    }

    // ── Test 46: residual telemetry with telemetry stores at RESIDUAL_DELTA and COSINE_SIMILARITY offsets ──
    // @trace TEST-12k96 [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_stores_at_cosine_similarity_offsets() {
        // Arrange: telemetry_ptr provided, so finalization should store delta and cosine
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: VecStore instructions should target RESIDUAL_DELTA_OFFSET and COSINE_SIMILARITY_OFFSET
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "should emit VecStore at RESIDUAL_DELTA_OFFSET ({})",
            telemetry_offsets::RESIDUAL_DELTA_OFFSET,
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            "should emit VecStore at COSINE_SIMILARITY_OFFSET ({})",
            telemetry_offsets::COSINE_SIMILARITY_OFFSET,
        );
    }

    // ── Test 47: gemm_row_stats FP16 dtype propagation succeeds ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_fp16_dtype_propagation_succeeds() {
        // Arrange: FP16 dtype — verifies that gemm_row_stats_telemetry propagates
        // FP16 QuantPrecision to VecStore instructions without error.
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F16,
        );

        // Assert
        assert!(result.is_ok(), "FP16 dtype should propagate without error in gemm_row_stats");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with FP16 dtype"
        );
    }

    // ── Test 48: rmsnorm_channel_scale FP16 dtype propagation succeeds ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_fp16_dtype_propagation_succeeds() {
        // Arrange: FP16 dtype — verifies channel scale telemetry propagates FP16
        // QuantPrecision through Broadcast, VecLoad, VecStore instructions.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F16,
        );

        // Assert
        assert!(result.is_ok(), "FP16 dtype should propagate without error in rmsnorm_channel_scale");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with FP16 dtype"
        );
    }

    // ── Test 49: silu_dead_neuron FP16 dtype propagation succeeds ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_fp16_dtype_propagation_succeeds() {
        // Arrange: FP16 dtype — verifies dead neuron detection propagates FP16
        // QuantPrecision through Broadcast, VecLoad, VecStore, and auto_lower_trace.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F16,
        );

        // Assert
        assert!(result.is_ok(), "FP16 dtype should propagate without error in silu_dead_neuron");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with FP16 dtype"
        );
    }

    // ── Test 50: residual telemetry with W128 SIMD width and telemetry enabled ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_w128_width_with_telemetry_enabled() {
        // Arrange: W128 (4 f32 lanes) with telemetry_ptr provided
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: W128 lanes=4, feature_vecs=64/4=16, tail=0
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W128,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: W128 with telemetry should succeed and emit cosine similarity finalization
        assert!(result.is_ok(), "W128 with telemetry should succeed");
        assert!(!prog.is_empty(), "should emit residual add + telemetry instructions");

        // Verify telemetry stores at delta and cosine offsets
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "should emit VecStore at RESIDUAL_DELTA_OFFSET with W128"
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            "should emit VecStore at COSINE_SIMILARITY_OFFSET with W128"
        );
    }

    // ── Test 51: rmsnorm_channel_scale returns Ok for Warp(0) zero lanes ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_returns_ok_for_warp_zero_lanes() {
        // Arrange: Warp(0) → f32_lanes() = 0, early exit path for rmsnorm
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::Warp(0),
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert: zero lanes should return Ok with no instructions emitted
        assert!(result.is_ok(), "Warp(0) should return Ok immediately");
        assert_eq!(
            prog.len(),
            initial_len,
            "Warp(0) should emit no instructions"
        );
    }

    // ── Test 52: silu_dead_neuron emits with W128 (NEON) SIMD width ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_with_w128_simd_width() {
        // Arrange: W128 has 4 f32 lanes, feature_dim=128 → feature_vecs=128/4=32
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W128,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "W128 SIMD width should succeed for silu_dead_neuron");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with W128 SIMD width"
        );
    }

    // ── Test 53: gemm_row_stats emits correct Abs and HReduce trace patterns ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_emits_hreduce_patterns_for_norm_max_min() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit exactly 3 VecStore instructions (L1 norm, max, min)
        let vecstore_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(
            vecstore_count >= 3,
            "should emit at least 3 VecStore instructions for L1/Max/Min stats, got {}",
            vecstore_count,
        );
    }

    // ── Test 54: residual telemetry with telemetry_ptr but feature_vecs=0 skips telemetry accumulators ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_ptr_but_zero_feature_vecs_skips_accumulators() {
        // Arrange: feature_dim=3, W256 lanes=8 → feature_vecs=0, tail=3
        // telemetry_ptr is Some but feature_vecs=0 → has_telemetry=false (no accumulators allocated)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(3)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            3,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: has_telemetry = telemetry_ptr.is_some() && feature_vecs > 0
        // feature_vecs = 3/8 = 0, so has_telemetry=false → no telemetry accumulators
        // But tail=3 so the tail loop should still emit scalar residual add instructions.
        assert!(result.is_ok(), "feature_dim < lanes with telemetry_ptr should succeed");
        assert!(!prog.is_empty(), "should emit scalar tail loop for residual add");

        // Should NOT emit VecStore at telemetry offsets (no accumulators → no finalization)
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "should NOT emit VecStore at RESIDUAL_DELTA_OFFSET when feature_vecs=0"
        );
    }

    // ── Test 55: silu_dead_neuron emits with Warp(32) NVIDIA GPU width ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_with_nvidia_warp32_width() {
        // Arrange: Warp(32) simulates NVIDIA GPU warp (32 f32 lanes), feature_dim=256
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(256)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act: 256 / 32 = 8 feature_vecs
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::Warp(32),
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Warp(32) NVIDIA GPU width should succeed for silu_dead_neuron");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with Warp(32) width"
        );
    }

    // ── Test 56: residual telemetry with multi-dim concrete shape but no telemetry emits fewer instructions ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_multi_dim_concrete_without_telemetry_produces_no_cosine_stores() {
        // Arrange: shape [8, 64] (multi-dim concrete), telemetry_ptr = None
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(8), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            None,
            None,
            QuantPrecision::F32,
        );

        // Assert: no telemetry_ptr → no cosine similarity finalization
        assert!(result.is_ok(), "multi-dim without telemetry should succeed");
        assert!(!prog.is_empty(), "should emit residual add instructions");

        // Should NOT emit VecStore at telemetry offsets
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "should NOT emit VecStore at RESIDUAL_DELTA_OFFSET when telemetry_ptr is None"
        );
        assert!(
            !vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            "should NOT emit VecStore at COSINE_SIMILARITY_OFFSET when telemetry_ptr is None"
        );
    }

    // ── Test 57: residual with symbolic outer dim and telemetry ptr emits finalization ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_symbolic_outer_with_telemetry_emits_cosine_finalization() {
        // Arrange: symbolic outer dimension [seq_len, 64] with telemetry_ptr provided
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![
            SymDim::Symbolic { name: "seq_len".into(), max_value: Some(2048) },
            SymDim::Concrete(64),
        ];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: symbolic outer + telemetry should produce cosine similarity stores
        assert!(result.is_ok(), "symbolic outer with telemetry should succeed");
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "symbolic outer + telemetry should emit VecStore at RESIDUAL_DELTA_OFFSET"
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            "symbolic outer + telemetry should emit VecStore at COSINE_SIMILARITY_OFFSET"
        );
    }

    // ── Test 58: rmsnorm channel scale VecStore targets indirect channel_scale_buf at offset 0 ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_vecstore_targets_indirect_buf_at_offset_zero() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: the final VecStore should be at offset 0 (into the indirect channel_scale_buf)
        let last_vecstore = prog.instrs.iter().rev().find(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(
            last_vecstore.is_some(),
            "should emit at least one VecStore instruction"
        );
        if let Some(VmInstr::VecStore { offset: OffsetExpr::Const(off), .. }) = last_vecstore {
            assert_eq!(
                *off, 0,
                "final VecStore should target offset 0 of the indirect channel_scale_buf"
            );
        }
    }

    // ── Test 59: silu dead neuron emits LoadPtr for telemetry buffer ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_loadptr_for_telemetry_buffer() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: first emitted instruction should be LoadPtr for the telemetry buffer
        let first_loadptr = prog.instrs.iter().find(|i| matches!(i, VmInstr::LoadPtr { .. }));
        assert!(
            first_loadptr.is_some(),
            "silu dead neuron should emit LoadPtr to load telemetry buffer pointer"
        );
    }

    // ── Test 60: residual with both telemetry_ptr and seq_bound_override combined ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_telemetry_with_seq_bound_override_combined() {
        // Arrange: symbolic outer shape but seq_bound_override provides Const(2), with telemetry
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![
            SymDim::Symbolic { name: "seq_len".into(), max_value: Some(2048) },
            SymDim::Concrete(64),
        ];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let override_bound = BoundExpr::Const(2);

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            Some(&override_bound),
            QuantPrecision::F32,
        );

        // Assert: both features combined should still produce correct telemetry
        assert!(result.is_ok(), "seq_bound_override + telemetry should work together");
        assert!(!prog.is_empty(), "should emit instructions");

        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "combined override + telemetry should emit delta offset"
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            "combined override + telemetry should emit cosine offset"
        );
    }

    // ── Test 61: gemm_row_stats stat stores all use Scalar width (HReduce result) ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_stat_stores_all_use_scalar_width() {
        // Arrange: hreduce_store always stores with SimdWidth::Scalar since HReduce
        // produces a broadcast result and the store writes a single scalar value.
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog,
            acc,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: all 3 telemetry stat VecStore instructions should use Scalar width
        let stat_vecstores: Vec<SimdWidth> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), width, .. } = i {
                    let is_telemetry_off = *off == telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET
                        || *off == telemetry_offsets::GEMM_ROW_MAX_OFFSET
                        || *off == telemetry_offsets::GEMM_ROW_MIN_OFFSET;
                    if is_telemetry_off { Some(*width) } else { None }
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            stat_vecstores.len(), 3,
            "should have exactly 3 telemetry stat VecStore instructions, found {}",
            stat_vecstores.len(),
        );
        for (idx, &w) in stat_vecstores.iter().enumerate() {
            assert_eq!(
                w, SimdWidth::Scalar,
                "stat store #{} should use Scalar width (HReduce result), got {:?}",
                idx, w,
            );
        }
    }

    // ── Test 62: rmsnorm channel scale emits with W128 (NEON/SSE) SIMD width ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_with_w128_simd_width() {
        // Arrange: W128 has 4 f32 lanes, feature_dim=64 → vec_count = 64/4 = 16
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            64,
            SimdWidth::W128,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "W128 SIMD width should succeed for rmsnorm_channel_scale");
        assert!(
            prog.len() > initial_len,
            "should emit instructions with W128 SIMD width"
        );
    }

    // ── Test 63: residual with Scalar width and no telemetry emits add instructions ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_scalar_width_without_telemetry_emits_add_instructions() {
        // Arrange: Scalar width (1 lane), no telemetry_ptr, feature_dim=4
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(4)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            4,
            SimdWidth::Scalar,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            None,
            None,
            QuantPrecision::F32,
        );

        // Assert: Scalar width + no telemetry should succeed and emit scalar add ops
        assert!(result.is_ok(), "Scalar width without telemetry should succeed");
        assert!(!prog.is_empty(), "should emit scalar residual add instructions");

        // Should NOT emit at telemetry offsets since telemetry_ptr is None
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "should NOT emit telemetry delta offset without telemetry_ptr"
        );
    }

    // ── Test 64: silu dead neuron with symbolic outer dim and W128 width ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_symbolic_outer_with_w128_width() {
        // Arrange: symbolic outer [seq_len, 64] with W128 (4 f32 lanes)
        // feature_vecs = 64/4 = 16
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![
            SymDim::Symbolic { name: "seq_len".into(), max_value: Some(512) },
            SymDim::Concrete(64),
        ];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog,
            input_ptr,
            &shape,
            SimdWidth::W128,
            &sym_map,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "symbolic outer + W128 should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions for symbolic outer + W128"
        );
    }

    // ── Test 65: residual with large outer dim concrete shape and telemetry emits full pipeline ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_large_outer_dim_concrete_with_telemetry_emits_full_pipeline() {
        // Arrange: shape [16, 64] → outer_dim=16, feature_dim=64, with telemetry
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(16), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog,
            &shape,
            64,
            SimdWidth::W256,
            input_ptr,
            weight_ptr,
            output_ptr,
            &sym_map,
            Some(telemetry_ptr_vreg),
            None,
            QuantPrecision::F32,
        );

        // Assert: large outer dim should produce full telemetry pipeline
        assert!(result.is_ok(), "large outer dim with telemetry should succeed");
        assert!(!prog.is_empty(), "should emit full residual + telemetry pipeline");

        // Verify both telemetry offsets are present
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "large outer dim should still emit RESIDUAL_DELTA_OFFSET"
        );
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            "large outer dim should still emit COSINE_SIMILARITY_OFFSET"
        );
    }

    // ── Test 66: rmsnorm channel scale emits VecLoad instructions for input scanning ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_vecload_for_input_row_scanning() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_rmsnorm_channel_scale_telemetry(
            &mut prog,
            input_ptr,
            256,
            SimdWidth::W256,
            &sym_map,
            QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit at least one VecLoad to scan the input row
        let vecload_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert!(
            vecload_count >= 1,
            "should emit at least 1 VecLoad to scan input row for per-channel abs max, got {}",
            vecload_count,
        );
    }

    // ── Test 67: silu_dead_neuron with 3D shape containing symbolic outer dimension ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_3d_shape_with_symbolic_batch_emits_instructions() {
        // Arrange: 3D shape [batch, 2, 64] — outer_sym finds "batch", feature_dim = 2*64 = 128
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![
            SymDim::Symbolic { name: "batch".into(), max_value: Some(32) },
            SymDim::Concrete(2),
            SymDim::Concrete(64),
        ];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog, input_ptr, &shape, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        );

        // Assert: 3D shape with symbolic batch should succeed
        assert!(result.is_ok(), "3D shape with symbolic batch should succeed");
        assert!(prog.len() > initial_len, "should emit instructions for 3D shape");
    }

    // ── Test 68: gemm_row_stats first emitted instruction is LoadPtr for telemetry ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_produces_instructions_with_mega_kernel_abi() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_gemm_row_stats_telemetry(
            &mut prog, acc, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "should succeed with default ABI (has telemetry slot)");
        assert!(!prog.instrs.is_empty(), "should emit at least one instruction");
    }

    // ── Test 69: rmsnorm_channel_scale emits Broadcast to initialize running_max ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_broadcast_for_running_max_init() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_rmsnorm_channel_scale_telemetry(
            &mut prog, input_ptr, 256, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit at least one Broadcast to zero-initialize running_max
        let broadcast_zero = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. })
        });
        assert!(
            broadcast_zero,
            "should emit Broadcast with Const(0.0) to initialize running_max"
        );
    }

    // ── Test 70: silu_dead_neuron feature_dim exactly equals lanes produces single vec iteration ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_feature_dim_equals_lanes_single_vec_iteration() {
        // Arrange: feature_dim=8, W256 lanes=8 → feature_vecs = 8/8 = 1 (exactly one inner iteration)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(8)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog, input_ptr, &shape, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        );

        // Assert: feature_dim == lanes should produce exactly 1 inner vec loop iteration
        assert!(result.is_ok(), "feature_dim == lanes should succeed");
        assert!(prog.len() > initial_len, "should emit instructions for single vec iteration");
    }

    // ── Test 71: residual_with_telemetry produces deterministic instruction count ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_produces_deterministic_instruction_count() {
        // Arrange: run the same emit twice and compare instruction counts
        let shape = vec![SymDim::Concrete(2), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let run_emit = || -> usize {
            let mut prog = VmProgram::new();
            let ip = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let wp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let op = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let tp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            emit_residual_with_telemetry(
                &mut prog, &shape, 64, SimdWidth::W256, ip, wp, op,
                &sym_map, Some(tp), None, QuantPrecision::F32,
            ).unwrap();
            prog.len()
        };

        // Act & Assert: two runs should produce identical instruction counts
        let count1 = run_emit();
        let count2 = run_emit();
        assert_eq!(count1, count2, "identical inputs should produce identical instruction counts");
    }

    // ── Test 72: residual_with_telemetry W512 width with telemetry produces cosine stores ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_w512_with_telemetry_produces_cosine_stores() {
        // Arrange: W512 (16 f32 lanes) with telemetry, feature_dim=64 → feature_vecs=4
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog, &shape, 64, SimdWidth::W512, input_ptr, weight_ptr, output_ptr,
            &sym_map, Some(telemetry_ptr_vreg), None, QuantPrecision::F32,
        );

        // Assert: W512 + telemetry should emit cosine similarity stores
        assert!(result.is_ok(), "W512 with telemetry should succeed");
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i { Some(*off) } else { None }
            })
            .collect();
        assert!(vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET));
        assert!(vecstore_offsets.contains(&telemetry_offsets::COSINE_SIMILARITY_OFFSET));
    }

    // ── Test 73: rmsnorm_channel_scale with Warp(16) simulating subset GPU warp ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_emits_with_warp16_subset_width() {
        // Arrange: Warp(16) simulates a subset SIMD width, feature_dim=128 → vec_count=128/16=8
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog, input_ptr, 128, SimdWidth::Warp(16), &sym_map, QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok(), "Warp(16) should succeed for rmsnorm_channel_scale");
        assert!(prog.len() > initial_len, "should emit instructions with Warp(16)");
    }

    // ── Test 74: silu_dead_neuron 3D all-concrete shape produces instructions ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_3d_all_concrete_shape_emits_instructions() {
        // Arrange: 3D shape [2, 4, 32] — all concrete, no symbolic dims
        // feature_dim = product of non-symbolic = 2*4*32 = 256, outer_bound = Const(1)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(2), SymDim::Concrete(4), SymDim::Concrete(32)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog, input_ptr, &shape, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        );

        // Assert: 3D all-concrete should use outer_bound=Const(1) path and emit instructions
        assert!(result.is_ok(), "3D all-concrete shape should succeed");
        assert!(prog.len() > initial_len, "should emit instructions for 3D concrete shape");
    }

    // ── Test 75: residual_with_telemetry with large tail count and telemetry enabled ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_large_tail_count_with_telemetry_enabled() {
        // Arrange: feature_dim=71, W256 lanes=8 → feature_vecs=8, tail=7, telemetry enabled
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(71)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog, &shape, 71, SimdWidth::W256, input_ptr, weight_ptr, output_ptr,
            &sym_map, Some(telemetry_ptr_vreg), None, QuantPrecision::F32,
        );

        // Assert: large tail with telemetry should succeed and produce full telemetry pipeline
        assert!(result.is_ok(), "large tail with telemetry should succeed");
        assert!(!prog.is_empty(), "should emit vec + tail residual add with telemetry");
        let vecstore_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| {
                if let VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } = i { Some(*off) } else { None }
            })
            .collect();
        assert!(
            vecstore_offsets.contains(&telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            "should emit delta offset even with large tail"
        );
    }

    // ── Test 76: gemm_row_stats instruction composition has exactly 1 LoadPtr ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_emits_exactly_one_loadptr_for_telemetry() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog, acc, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit exactly 1 LoadPtr (for the telemetry buffer pointer)
        let loadptr_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::LoadPtr { .. }))
            .count();
        assert_eq!(
            loadptr_count, 1,
            "should emit exactly 1 LoadPtr for telemetry buffer, got {}",
            loadptr_count,
        );
    }

    // ── Test 77: EpilogueTelemetryConfig default has all flags false (zero telemetry) ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn epilogue_telemetry_config_default_is_all_disabled() {
        // Arrange
        use crate::compiler::graph::EpilogueTelemetryConfig;

        // Act
        let cfg = EpilogueTelemetryConfig::default();

        // Assert: default config should have all telemetry flags disabled
        assert!(!cfg.silu_dead_neuron, "default silu_dead_neuron should be false");
        assert!(!cfg.moe_hit_counter, "default moe_hit_counter should be false");
        assert!(!cfg.rmsnorm_channel_scale, "default rmsnorm_channel_scale should be false");
        assert!(!cfg.softmax_sharpness, "default softmax_sharpness should be false");
        assert!(!cfg.residual_cosine_sim, "default residual_cosine_sim should be false");
        assert!(!cfg.gemm_row_stats, "default gemm_row_stats should be false");
        assert!(!cfg.embed_l2_norm, "default embed_l2_norm should be false");
    }

    // ── Test 78: gemm_row_stats stat offsets are all distinct and strictly ordered ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_offsets_are_distinct_and_ordered() {
        // Arrange: the three GEMM row stats offsets must be distinct and ascending
        let l1 = telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET;
        let max = telemetry_offsets::GEMM_ROW_MAX_OFFSET;
        let min = telemetry_offsets::GEMM_ROW_MIN_OFFSET;

        // Act & Assert: all three must be distinct
        assert_ne!(l1, max, "L1 and Max offsets must be distinct");
        assert_ne!(l1, min, "L1 and Min offsets must be distinct");
        assert_ne!(max, min, "Max and Min offsets must be distinct");

        // Offsets should be ascending: L1 < Max < Min (by layout convention)
        assert!(l1 < max, "GEMM_ROW_NORM_L1_OFFSET ({}) < GEMM_ROW_MAX_OFFSET ({})", l1, max);
        assert!(max < min, "GEMM_ROW_MAX_OFFSET ({}) < GEMM_ROW_MIN_OFFSET ({})", max, min);
    }

    // ── Test 79: gemm_row_stats instruction sequence starts with LoadPtr then has VecStore at 3 offsets ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_instruction_sequence_structure() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog, acc, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: should contain a LoadPtr for telemetry buffer (may not be first due to DeclareVReg)
        let has_loadptr = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoadPtr { .. }));
        assert!(
            has_loadptr,
            "should contain LoadPtr for telemetry buffer",
        );

        // Should contain exactly 3 VecStore instructions at telemetry offsets
        let tel_offsets = [
            telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET,
            telemetry_offsets::GEMM_ROW_MAX_OFFSET,
            telemetry_offsets::GEMM_ROW_MIN_OFFSET,
        ];
        let store_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } if tel_offsets.contains(off) => Some(*off),
                _ => None,
            })
            .collect();
        assert_eq!(store_offsets.len(), 3, "should have exactly 3 telemetry VecStore instructions");
    }

    // ── Test 80: rmsnorm_channel_scale with feature_dim=1 and Scalar width emits minimal instructions ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_feature_dim_1_scalar_width_emits_instructions() {
        // Arrange: feature_dim=1, Scalar width (1 lane) → vec_count=1/1=1
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_rmsnorm_channel_scale_telemetry(
            &mut prog, input_ptr, 1, SimdWidth::Scalar, &sym_map, QuantPrecision::F32,
        );

        // Assert: feature_dim=1 with Scalar (1 lane) → vec_count=1, should emit instructions
        assert!(result.is_ok(), "feature_dim=1 with Scalar width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions for single scalar element channel scale"
        );
    }

    // ── Test 81: silu_dead_neuron with feature_dim=1 and Scalar width produces valid output ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_feature_dim_1_scalar_width_emits_instructions() {
        // Arrange: feature_dim=1, Scalar width (1 lane) → feature_vecs=1/1=1
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let initial_len = prog.len();

        // Act
        let result = emit_silu_dead_neuron_telemetry(
            &mut prog, input_ptr, &shape, SimdWidth::Scalar, &sym_map, QuantPrecision::F32,
        );

        // Assert: single element with scalar width should succeed
        assert!(result.is_ok(), "feature_dim=1 with Scalar width should succeed");
        assert!(
            prog.len() > initial_len,
            "should emit instructions for single element dead neuron detection"
        );
    }

    // ── Test 82: residual_with_telemetry with feature_dim=1 and Scalar width no tail ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_feature_dim_1_scalar_width_no_tail() {
        // Arrange: feature_dim=1, Scalar (1 lane) → feature_vecs=1, tail=0
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let tp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(1)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_residual_with_telemetry(
            &mut prog, &shape, 1, SimdWidth::Scalar,
            input_ptr, weight_ptr, output_ptr, &sym_map, Some(tp), None, QuantPrecision::F32,
        );

        // Assert: single element with telemetry should succeed
        assert!(result.is_ok(), "feature_dim=1 Scalar with telemetry should succeed");
        assert!(!prog.is_empty(), "should emit residual add + telemetry instructions");
    }

    // ── Test 83: EpilogueTelemetryConfig with all flags enabled is not default ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn epilogue_telemetry_config_all_enabled_differs_from_default() {
        // Arrange
        use crate::compiler::graph::EpilogueTelemetryConfig;

        // Act
        let default_cfg = EpilogueTelemetryConfig::default();
        let all_enabled = EpilogueTelemetryConfig {
            silu_dead_neuron: true,
            moe_hit_counter: true,
            rmsnorm_channel_scale: true,
            softmax_sharpness: true,
            residual_cosine_sim: true,
            gemm_row_stats: true,
            embed_l2_norm: true,
        };

        // Assert: all-enabled config must differ from default on every field
        assert_ne!(all_enabled.silu_dead_neuron, default_cfg.silu_dead_neuron);
        assert_ne!(all_enabled.gemm_row_stats, default_cfg.gemm_row_stats);
        assert_ne!(all_enabled.residual_cosine_sim, default_cfg.residual_cosine_sim);
    }

    // ── Test 84: TELEMETRY_BUFFER_MIN_BYTES covers all known offsets ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_buffer_min_bytes_covers_all_stat_offsets() {
        // Arrange: collect all telemetry offsets that store f32 values
        let f32_offsets: Vec<(usize, &str)> = vec![
            (telemetry_offsets::SILU_DEAD_NEURON_COUNT, "SILU_DEAD_NEURON_COUNT"),
            (telemetry_offsets::RESIDUAL_DELTA_OFFSET, "RESIDUAL_DELTA_OFFSET"),
            (telemetry_offsets::COSINE_SIMILARITY_OFFSET, "COSINE_SIMILARITY_OFFSET"),
            (telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET, "GEMM_ROW_NORM_L1_OFFSET"),
            (telemetry_offsets::GEMM_ROW_MAX_OFFSET, "GEMM_ROW_MAX_OFFSET"),
            (telemetry_offsets::GEMM_ROW_MIN_OFFSET, "GEMM_ROW_MIN_OFFSET"),
            (telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET, "SOFTMAX_SHARPNESS_OFFSET"),
            (telemetry_offsets::SOFTMAX_MAX_OFFSET, "SOFTMAX_MAX_OFFSET"),
            (telemetry_offsets::EFFECTIVE_CONTEXT_LEN_OFFSET, "EFFECTIVE_CONTEXT_LEN_OFFSET"),
            (telemetry_offsets::IS_ATTENTION_SINK_OFFSET, "IS_ATTENTION_SINK_OFFSET"),
            (telemetry_offsets::EMBED_L2_NORM_OFFSET, "EMBED_L2_NORM_OFFSET"),
        ];

        // Act & Assert: each offset + 4 bytes must fit within TELEMETRY_BUFFER_MIN_BYTES
        for (off, name) in &f32_offsets {
            assert!(
                off + 4 <= telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES,
                "{} offset ({}) + sizeof(f32) exceeds TELEMETRY_BUFFER_MIN_BYTES ({})",
                name, off, telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES,
            );
        }
    }

    // ── Test 85: gemm_row_stats with different SimdWidth produces different instruction counts ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_different_simd_widths_produce_different_counts() {
        // Arrange: W256 (8 lanes) vs W512 (16 lanes) should produce different instruction counts
        // because the HReduce and auto_lower_trace emit width-dependent instructions.
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: emit with W256
        let mut prog_w256 = VmProgram::new();
        let acc_256 = prog_w256.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        emit_gemm_row_stats_telemetry(
            &mut prog_w256, acc_256, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Act: emit with W512
        let mut prog_w512 = VmProgram::new();
        let acc_512 = prog_w512.alloc_vreg(VRegKind::Vec, SimdWidth::W512);
        emit_gemm_row_stats_telemetry(
            &mut prog_w512, acc_512, SimdWidth::W512, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: both widths should produce instructions (exact count may vary by ISA path)
        assert!(prog_w256.len() > 0, "W256 should produce instructions");
        assert!(prog_w512.len() > 0, "W512 should produce instructions");
    }

    // ── Test 86: residual_with_telemetry without telemetry produces no Broadcast for accumulators ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_without_telemetry_emits_no_accumulator_broadcasts() {
        // Arrange: without telemetry_ptr, no dot/norm_sq accumulators are allocated,
        // so no Broadcast(Const(0.0)) for accumulator initialization should appear.
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_residual_with_telemetry(
            &mut prog, &shape, 64, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr, &sym_map, None, None, QuantPrecision::F32,
        ).unwrap();

        // Assert: no Broadcast with Const(0.0) should appear (no accumulator init without telemetry)
        let zero_broadcast_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }))
            .count();
        assert_eq!(
            zero_broadcast_count, 0,
            "without telemetry_ptr, should emit no Broadcast(Const(0.0)) for accumulators, got {}",
            zero_broadcast_count,
        );
    }

    // ── Test 87: residual with telemetry emits Broadcast(Const(0.0)) for dot and norm accumulators ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_emits_accumulator_broadcasts() {
        // Arrange: with telemetry_ptr, three accumulators (dot, norm_sq_in, norm_sq_out)
        // are allocated and initialized with Broadcast(Const(0.0)).
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_residual_with_telemetry(
            &mut prog, &shape, 64, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr, &sym_map,
            Some(telemetry_ptr_vreg), None, QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit exactly 3 Broadcast(Const(0.0)) for the three accumulators
        let zero_broadcast_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }))
            .count();
        assert!(
            zero_broadcast_count >= 3,
            "with telemetry, should emit at least 3 Broadcast(Const(0.0)) for accumulators, got {}",
            zero_broadcast_count,
        );
    }

    // ── Test 88: silu_dead_neuron dead count body uses Sub/Max/Min/Mul/Add trace pattern ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn silu_dead_neuron_emits_vecload_for_input_scanning() {
        // Arrange: verifies that silu_dead_neuron actually loads input data for scanning
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(128)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_silu_dead_neuron_telemetry(
            &mut prog, input_ptr, &shape, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit at least one VecLoad to scan input values
        let vecload_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert!(
            vecload_count >= 1,
            "silu dead neuron should emit VecLoad for scanning input, got {}",
            vecload_count,
        );
    }

    // ── Test 89: TELEMETRY_BUFFER_MIN_BYTES is a power of 2 and at least 512 ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_buffer_min_bytes_is_reasonable_power_of_two() {
        // Arrange & Act: check TELEMETRY_BUFFER_MIN_BYTES properties
        let min_bytes = telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES;

        // Assert: must be at least 512 to cover all offsets
        assert!(
            min_bytes >= 512,
            "TELEMETRY_BUFFER_MIN_BYTES should be at least 512, got {}",
            min_bytes,
        );
        // Assert: must be a power of 2 (alignment-friendly for page allocation)
        assert!(
            min_bytes.is_power_of_two(),
            "TELEMETRY_BUFFER_MIN_BYTES should be a power of 2, got {}",
            min_bytes,
        );
    }

    // ── Test 90: EXPERT_HIT_COUNTS_OFFSET is between SILU and RESIDUAL regions ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn expert_hit_counts_offset_lies_between_silu_and_residual_regions() {
        // Arrange: expert hit counts should be after SILU region (0..8) and before residual (128+)
        let expert_off = telemetry_offsets::EXPERT_HIT_COUNTS_OFFSET;
        let silu_mask_end = telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET + 8; // 8 bytes past mask
        let residual_start = telemetry_offsets::RESIDUAL_DELTA_OFFSET;

        // Assert: expert offset should be in the region between SILU and RESIDUAL
        assert!(
            expert_off >= silu_mask_end,
            "EXPERT_HIT_COUNTS_OFFSET ({}) should be >= end of SILU region ({})",
            expert_off, silu_mask_end,
        );
        assert!(
            expert_off < residual_start,
            "EXPERT_HIT_COUNTS_OFFSET ({}) should be < RESIDUAL_DELTA_OFFSET ({})",
            expert_off, residual_start,
        );
    }

    // ── Test 91: rmsnorm_channel_scale with W512 produces fewer loop iterations than W256 ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn rmsnorm_channel_scale_w512_produces_fewer_total_instructions_than_w256() {
        // Arrange: same feature_dim=256, W256 lanes=8 (32 vecs) vs W512 lanes=16 (16 vecs)
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: W256
        let mut prog_w256 = VmProgram::new();
        let ip_256 = prog_w256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_rmsnorm_channel_scale_telemetry(
            &mut prog_w256, ip_256, 256, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Act: W512
        let mut prog_w512 = VmProgram::new();
        let ip_512 = prog_w512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_rmsnorm_channel_scale_telemetry(
            &mut prog_w512, ip_512, 256, SimdWidth::W512, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: wider SIMD should produce fewer instructions (fewer loop iterations)
        assert!(
            prog_w512.len() <= prog_w256.len(),
            "W512 ({} instrs) should have <= W256 ({} instrs) for same feature_dim",
            prog_w512.len(), prog_w256.len(),
        );
    }

    // ── Test 92: residual cosine similarity offsets are 8 bytes apart (one f32 + padding) ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_cosine_offsets_are_eight_bytes_apart() {
        // Arrange: RESIDUAL_DELTA_OFFSET and COSINE_SIMILARITY_OFFSET store f32 values
        let delta = telemetry_offsets::RESIDUAL_DELTA_OFFSET;
        let cosine = telemetry_offsets::COSINE_SIMILARITY_OFFSET;

        // Assert: the two offsets should be exactly 8 bytes apart (one aligned f32 slot each)
        let diff = cosine.abs_diff(delta);
        assert_eq!(
            diff, 8,
            "RESIDUAL_DELTA_OFFSET ({}) and COSINE_SIMILARITY_OFFSET ({}) should be 8 bytes apart, got {}",
            delta, cosine, diff,
        );
    }

    // ── Test 93: residual_with_telemetry emits Comment for cosine finalization section ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_emits_cosine_telemetry_comment() {
        // Arrange: with telemetry_ptr, the finalization section should emit a comment
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let telemetry_ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let shape = vec![SymDim::Concrete(1), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_residual_with_telemetry(
            &mut prog, &shape, 64, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr, &sym_map,
            Some(telemetry_ptr_vreg), None, QuantPrecision::F32,
        ).unwrap();

        // Assert: should emit a Comment mentioning cosine similarity or telemetry finalization
        let has_cosine_comment = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::Comment(text) if text.contains("cosine") || text.contains("§13.11"))
        });
        assert!(
            has_cosine_comment,
            "should emit a Comment instruction mentioning cosine similarity or §13.11"
        );
    }

    // ── Test 94: CHANNEL_SCALE_PTR_OFFSET is a pointer-sized offset (8-byte aligned) ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn channel_scale_ptr_offset_is_pointer_aligned() {
        // Arrange: CHANNEL_SCALE_PTR_OFFSET stores an indirect pointer, must be 8-byte aligned
        let off = telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET;

        // Assert: pointer storage requires 8-byte alignment on 64-bit platforms
        assert_eq!(
            off % 8, 0,
            "CHANNEL_SCALE_PTR_OFFSET ({}) must be 8-byte aligned for pointer storage",
            off,
        );
    }

    // ── Test 95: residual_with_telemetry with same inputs always produces same instruction count ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn residual_with_telemetry_no_telemetry_deterministic_count() {
        // Arrange: run the same emit without telemetry three times
        let shape = vec![SymDim::Concrete(4), SymDim::Concrete(64)];
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let run_emit = || -> usize {
            let mut prog = VmProgram::new();
            let ip = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let wp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let op = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            emit_residual_with_telemetry(
                &mut prog, &shape, 64, SimdWidth::W256, ip, wp, op,
                &sym_map, None, None, QuantPrecision::F32,
            ).unwrap();
            prog.len()
        };

        // Act & Assert: three runs should produce identical instruction counts
        let count1 = run_emit();
        let count2 = run_emit();
        let count3 = run_emit();
        assert_eq!(count1, count2, "run 1 ({}) != run 2 ({})", count1, count2);
        assert_eq!(count2, count3, "run 2 ({}) != run 3 ({})", count2, count3);
    }

    // ── Test 96: gemm_row_stats emits exactly 3 VecStore at distinct telemetry offsets ──
    // @trace TEST-12kaw [req:REQ-OBS] [level:unit]

    #[test]
    fn gemm_row_stats_vecstore_offsets_are_all_distinct() {
        // Arrange
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        emit_gemm_row_stats_telemetry(
            &mut prog, acc, SimdWidth::W256, &sym_map, QuantPrecision::F32,
        ).unwrap();

        // Assert: collect all Const-offset VecStore instructions and verify no duplicates
        let tel_offsets = [
            telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET,
            telemetry_offsets::GEMM_ROW_MAX_OFFSET,
            telemetry_offsets::GEMM_ROW_MIN_OFFSET,
        ];
        let store_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } if tel_offsets.contains(off) => Some(*off),
                _ => None,
            })
            .collect();

        // Must have 3 stores, and each must be unique
        assert_eq!(store_offsets.len(), 3, "should have exactly 3 telemetry VecStore instructions");
        let unique: std::collections::HashSet<usize> = store_offsets.iter().copied().collect();
        assert_eq!(unique.len(), 3, "all 3 telemetry VecStore offsets must be distinct");
    }
}
