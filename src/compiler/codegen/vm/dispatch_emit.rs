//! Structural and compute pattern dispatch — OpKind routing to emit functions.
//!
//! Extracted from `plan_lower.rs` to isolate the two large dispatch functions
//! that route OpKind variants to the appropriate JIT emit functions.

use super::instr::*;
use super::vm_state::AbiPtrs;
use super::plan_lower::{LoweringContext, TensorPtrResolver};

use crate::compiler::graph::{CompilerGraph, SymDim};
use crate::compiler::trace::QuantPrecision;
use crate::types::CompilerError;

/// Structural op dispatch (ARCH-AUTO-INSTR-SELECT Category C/D).
///
/// **Category C** (手写 lower 委托, awaiting TraceOp extension for auto_select migration):
/// Gather, ColumnSlice, QTapSTG.
///
/// **Category D** (permanent, cannot use auto_select):
/// Residual+telemetry (fused compute+telemetry loop), Argmax (specialized VmInstr),
/// StoreToken, WriteLogits, CheckStopCondition (generation control flow),
/// GuardrailCheck, CotStepCheck, SgInject, SgDetect (in-flight control),
/// EarlyExit, MegaKernelDispatch (dispatch/scheduling), Reshape/Transpose/SliceView (NOP).
#[allow(clippy::too_many_arguments)]
#[allow(unreachable_code)]
pub(crate) fn dispatch_structural(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    seq_bound_override: Option<&BoundExpr>,
) -> Result<(), CompilerError> {
    // Phase 4-7+: Op v2 lowering — 所有 structural ops 走 lower_op_v2（胖 opcode 驱动）。
    if super::plan_lower::lower_op_v2(prog, op, graph, ctx, resolver, abi)? {
        return Ok(());
    }

    // 死代码保护（51064 测试验证零触发）：lower_op_v2 处理所有 dispatch_structural 接收的 ops。
    // 如果到达这里，说明有新 op 未被 lower_op_v2 覆盖——需要补充 Op v2 路径。
    return Err(CompilerError::CodegenViolation(format!(
        "dispatch_structural: op {:?} 未被 lower_op_v2 处理。\
         所有 structural ops 应通过 Op v2 (lower_op_v2) 处理。", op.kind
    )));

}




/// ComputePattern-driven dispatch (ARCH-AUTO-INSTR-SELECT).
///
// ── AltUp lower functions (Gemma 4 E2B/E4B) ──────────────────────────
//
// AltUpPredict/AltUpCorrect/AltUpInject are cross-path arithmetic ops.
// Current implementation: passthrough copy (input → output) to ensure
// pipeline compilation. Full cross-path arithmetic requires dedicated
// TraceOp extensions for P-dimension path-aware addressing.

/// AltUpPredict: fat_buffer [S,P*H] + pred_coefs [S,P²] → predictions [S,P*H]
///
/// Per position s, for each prediction path p:
///   predictions[p] = hidden[p] + Σ_q coefs[p,q] · hidden[q]
/// where hidden[p] = fat_buffer[s, p*H..(p+1)*H] and coefs[p,q] = pred_coefs[s, p*P+q].
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_altup_predict(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    _graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    _weight_ptr: VRegId,
    output_ptr: VRegId,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    seq_len: SymDim,
    num_preds: usize,
    hidden: usize,
) -> Result<(), CompilerError> {
    let seq_bound = ctx.session.sym_map.to_bound(&seq_len);
    let width = ctx.session.width.f32_lanes();
    let p = num_preds;
    let elem_bytes = 4usize;
    let row_bytes = p * hidden * elem_bytes;
    let num_vec = (hidden + width - 1) / width;

    // Resolve coef pointer (op.inputs[1] = pred_coefs [S, P²])
    let coefs_ptr = resolver.materialize(prog, op.inputs[1], abi)
        .ok_or_else(|| CompilerError::CodegenViolation("AltUpPredict: pred_coefs ptr".into()))?;

    prog.emit_loop_try(seq_bound, row_bytes, |prog, _ctr, seq_off| {
        // For each prediction path p_out: predictions[p_out] = hidden[p_out] + Σ_q coefs[p,q]*hidden[q]
        for p_out in 0..p {
            // Step A: Initialize accumulator = hidden[p_out] (copy input → output)
            for v in 0..num_vec {
                let off = p_out * hidden * elem_bytes + v * width * elem_bytes;
                let off_expr = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(off)),
                );
                let data = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: data, base: input_ptr, offset: off_expr.clone(),
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: off_expr, src: data,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
            }

            // Step B: Accumulate Σ_q coefs[p_out, q] * hidden[q] → add to output[p_out]
            for q in 0..p {
                // Broadcast scalar coef[p_out, q] to vector
                let coef_byte_off = (p_out * p + q) * elem_bytes;
                let coef_bc = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::Broadcast {
                    dst: coef_bc,
                    src: ScalarExpr::MemLoad(coefs_ptr, OffsetExpr::Add(
                        Box::new(OffsetExpr::LoopOffset(seq_off)),
                        Box::new(OffsetExpr::Const(coef_byte_off)),
                    )),
                    width: ctx.session.width,
                    dtype: QuantPrecision::F32,
                });

                for v in 0..num_vec {
                    let q_off = q * hidden * elem_bytes + v * width * elem_bytes;
                    let out_off = p_out * hidden * elem_bytes + v * width * elem_bytes;
                    let q_off_expr = OffsetExpr::Add(
                        Box::new(OffsetExpr::LoopOffset(seq_off)),
                        Box::new(OffsetExpr::Const(q_off)),
                    );
                    let out_off_expr = OffsetExpr::Add(
                        Box::new(OffsetExpr::LoopOffset(seq_off)),
                        Box::new(OffsetExpr::Const(out_off)),
                    );

                    // Load hidden[q] chunk
                    let h_q = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                    prog.emit(VmInstr::VecLoad {
                        dst: h_q, base: input_ptr, offset: q_off_expr,
                        width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                    });
                    // scaled = coef * hidden[q]
                    let scaled = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                    prog.emit(VmInstr::VecBinOp {
                        dst: scaled, a: h_q, b: coef_bc,
                        op: VecOp::Mul, dtype: QuantPrecision::F32,
                    });
                    // acc = load output[p_out]
                    let acc = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                    prog.emit(VmInstr::VecLoad {
                        dst: acc, base: output_ptr, offset: out_off_expr.clone(),
                        width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                    });
                    // acc += scaled
                    let new_acc = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                    prog.emit(VmInstr::VecBinOp {
                        dst: new_acc, a: acc, b: scaled,
                        op: VecOp::Add, dtype: QuantPrecision::F32,
                    });
                    prog.emit(VmInstr::VecStore {
                        base: output_ptr, offset: out_off_expr, src: new_acc,
                        width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                    });
                }
            }
        }
        Ok(())
    })
}

/// AltUpCorrect: predictions [S,P*H] + corr_coefs [S,P] + gated [S,H] → corrected [S,P*H]
///
/// corrected[0] = gated (active path direct update)
/// corrected[p] = predictions[p] + corr_coefs[p] · (gated - predictions[0])  for p > 0
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_altup_correct(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    _graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    _weight_ptr: VRegId,
    output_ptr: VRegId,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    seq_len: SymDim,
    num_preds: usize,
    hidden: usize,
) -> Result<(), CompilerError> {
    let seq_bound = ctx.session.sym_map.to_bound(&seq_len);
    let width = ctx.session.width.f32_lanes();
    let p = num_preds;
    let elem_bytes = 4usize;
    let row_bytes = p * hidden * elem_bytes;
    let num_vec = (hidden + width - 1) / width;

    let coefs_ptr = resolver.materialize(prog, op.inputs[1], abi)
        .ok_or_else(|| CompilerError::CodegenViolation("AltUpCorrect: corr_coefs ptr".into()))?;
    let gated_ptr = resolver.materialize(prog, op.inputs[2], abi)
        .ok_or_else(|| CompilerError::CodegenViolation("AltUpCorrect: gated ptr".into()))?;

    prog.emit_loop_try(seq_bound, row_bytes, |prog, _ctr, seq_off| {
        // Step 1: corrected[0] = gated (copy active path)
        for v in 0..num_vec {
            let off = v * width * elem_bytes;
            let off_expr = OffsetExpr::Add(
                Box::new(OffsetExpr::LoopOffset(seq_off)),
                Box::new(OffsetExpr::Const(off)),
            );
            let data = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
            prog.emit(VmInstr::VecLoad {
                dst: data, base: gated_ptr, offset: off_expr.clone(),
                width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
            });
            prog.emit(VmInstr::VecStore {
                base: output_ptr, offset: off_expr, src: data,
                width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
            });
        }

        // Step 2: corrected[p] = predictions[p] + coef[p] · (gated - predictions[0]) for p > 0
        for p_out in 1..p {
            // Broadcast scalar coef[p_out]
            let coef_bc = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
            prog.emit(VmInstr::Broadcast {
                dst: coef_bc,
                src: ScalarExpr::MemLoad(coefs_ptr, OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(p_out * elem_bytes)),
                )),
                width: ctx.session.width,
                dtype: QuantPrecision::F32,
            });

            for v in 0..num_vec {
                let chunk_off = v * width * elem_bytes;
                let base_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(chunk_off)),
                );

                // Load predictions[p_out] chunk
                let pred_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(p_out * hidden * elem_bytes + chunk_off)),
                );
                let pred_v = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: pred_v, base: input_ptr, offset: pred_off,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                // Load gated chunk
                let gated_v = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: gated_v, base: gated_ptr, offset: base_off.clone(),
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                // Load predictions[0] chunk
                let pred0_v = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: pred0_v, base: input_ptr, offset: base_off,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                // innovation = gated - predictions[0]
                let innov = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecBinOp {
                    dst: innov, a: gated_v, b: pred0_v,
                    op: VecOp::Sub, dtype: QuantPrecision::F32,
                });
                // scaled = coef * innovation
                let scaled = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecBinOp {
                    dst: scaled, a: coef_bc, b: innov,
                    op: VecOp::Mul, dtype: QuantPrecision::F32,
                });
                // corrected[p_out] = predictions[p_out] + scaled
                let result = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecBinOp {
                    dst: result, a: pred_v, b: scaled,
                    op: VecOp::Add, dtype: QuantPrecision::F32,
                });
                let out_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(p_out * hidden * elem_bytes + chunk_off)),
                );
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: out_off, src: result,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
            }
        }
        Ok(())
    })
}

/// AltUpInject: corrected [S,P*H] + normalized [S,H] → output [S,P*H]
///
/// Copy corrected to output, then: output[p] += normalized for p > 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_altup_inject(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    _graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    _weight_ptr: VRegId,
    output_ptr: VRegId,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    seq_len: SymDim,
    num_preds: usize,
    hidden: usize,
) -> Result<(), CompilerError> {
    let seq_bound = ctx.session.sym_map.to_bound(&seq_len);
    let width = ctx.session.width.f32_lanes();
    let p = num_preds;
    let elem_bytes = 4usize;
    let row_bytes = p * hidden * elem_bytes;
    let num_vec = (hidden + width - 1) / width;

    let norm_ptr = resolver.materialize(prog, op.inputs[1], abi)
        .ok_or_else(|| CompilerError::CodegenViolation("AltUpInject: normalized ptr".into()))?;

    prog.emit_loop_try(seq_bound, row_bytes, |prog, _ctr, seq_off| {
        // Step 1: Copy all corrected paths to output
        let total_vec = (p * hidden + width - 1) / width;
        for v in 0..total_vec {
            let off = v * width * elem_bytes;
            let off_expr = OffsetExpr::Add(
                Box::new(OffsetExpr::LoopOffset(seq_off)),
                Box::new(OffsetExpr::Const(off)),
            );
            let data = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
            prog.emit(VmInstr::VecLoad {
                dst: data, base: input_ptr, offset: off_expr.clone(),
                width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
            });
            prog.emit(VmInstr::VecStore {
                base: output_ptr, offset: off_expr, src: data,
                width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
            });
        }

        // Step 2: output[p] += normalized for p > 0
        for p_out in 1..p {
            for v in 0..num_vec {
                let chunk_off = v * width * elem_bytes;
                let norm_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(chunk_off)),
                );
                let norm_v = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: norm_v, base: norm_ptr, offset: norm_off,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                let out_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(p_out * hidden * elem_bytes + chunk_off)),
                );
                let out_v = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: out_v, base: output_ptr, offset: out_off.clone(),
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                let sum = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecBinOp {
                    dst: sum, a: out_v, b: norm_v,
                    op: VecOp::Add, dtype: QuantPrecision::F32,
                });
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: out_off, src: sum,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
            }
        }
        Ok(())
    })
}

/// ScaleConst: out = x * value. Broadcast scalar constant to SIMD vector, multiply.
pub(crate) fn lower_scale_const(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    output_ptr: VRegId,
    _resolver: &TensorPtrResolver,
    _abi: &AbiPtrs,
    value: f32,
) -> Result<(), CompilerError> {
    // Infer total element count from output tensor shape
    let out_tid = op.outputs[0];
    let out_tensor = graph.tensor(out_tid)
        .ok_or_else(|| CompilerError::CodegenViolation("ScaleConst: no output tensor".into()))?;
    let seq_dim = out_tensor.shape.iter().find(|d| d.is_symbolic()).cloned();
    let feature_dim: usize = out_tensor.shape.iter()
        .filter(|d| !d.is_symbolic())
        .map(|d| d.as_concrete().unwrap_or(1))
        .product::<usize>()
        .max(1);

    let width = ctx.session.width.f32_lanes();
    let elem_bytes = 4usize;
    let num_vec = (feature_dim + width - 1) / width;
    let row_bytes = feature_dim * elem_bytes;

    // Broadcast constant to vector register
    let const_bc = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
    prog.emit(VmInstr::Broadcast {
        dst: const_bc,
        src: ScalarExpr::Const(value),
        width: ctx.session.width,
        dtype: QuantPrecision::F32,
    });

    if let Some(sym_dim) = seq_dim {
        // Symbolic seq dimension → outer loop
        let seq_bound = ctx.session.sym_map.to_bound(&sym_dim);
        prog.emit_loop_try(seq_bound, row_bytes, |prog, _ctr, seq_off| {
            for v in 0..num_vec {
                let off = v * width * elem_bytes;
                let off_expr = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(seq_off)),
                    Box::new(OffsetExpr::Const(off)),
                );
                let data = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecLoad {
                    dst: data, base: input_ptr, offset: off_expr.clone(),
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
                let scaled = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
                prog.emit(VmInstr::VecBinOp {
                    dst: scaled, a: data, b: const_bc,
                    op: VecOp::Mul, dtype: QuantPrecision::F32,
                });
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: off_expr, src: scaled,
                    width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
                });
            }
            Ok(())
        })
    } else {
        // No symbolic dimension — flat vector multiply
        for v in 0..num_vec {
            let off = v * width * elem_bytes;
            let off_expr = OffsetExpr::Const(off);
            let data = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
            prog.emit(VmInstr::VecLoad {
                dst: data, base: input_ptr, offset: off_expr.clone(),
                width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
            });
            let scaled = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
            prog.emit(VmInstr::VecBinOp {
                dst: scaled, a: data, b: const_bc,
                op: VecOp::Mul, dtype: QuantPrecision::F32,
            });
            prog.emit(VmInstr::VecStore {
                base: output_ptr, offset: off_expr, src: scaled,
                width: ctx.session.width, dtype: QuantPrecision::F32, predicate: None,
            });
        }
        Ok(())
    }
}

/// Emit a VecLoad+VecStore copy loop over total_elem elements per seq step.
/// Uses emit_loop_try for the outer seq dimension, VecLoad/VecStore for inner elements.
fn emit_altup_copy_loop(
    prog: &mut VmProgram,
    input_ptr: VRegId,
    output_ptr: VRegId,
    seq_bound: BoundExpr,
    total_elem: usize,
    ctx: &LoweringContext,
) -> Result<(), CompilerError> {
    let elem_bytes = 4usize; // f32
    let width_val = ctx.session.width.f32_lanes();
    let vec_bytes = width_val * elem_bytes;
    let num_vec_iters = (total_elem + width_val - 1) / width_val;
    let step_bytes = total_elem * elem_bytes;

    prog.emit_loop_try(seq_bound, step_bytes, |prog, _ctr, seq_off| {
        for v in 0..num_vec_iters {
            let off = v * vec_bytes;
            let data = prog.alloc_vreg(VRegKind::Vec, ctx.session.width);
            let offset_expr = OffsetExpr::Add(
                Box::new(OffsetExpr::LoopOffset(seq_off)),
                Box::new(OffsetExpr::Const(off)),
            );
            prog.emit(VmInstr::VecLoad {
                dst: data,
                base: input_ptr,
                offset: offset_expr.clone(),
                width: ctx.session.width,
                dtype: QuantPrecision::F32, predicate: None,
            });
            prog.emit(VmInstr::VecStore {
                base: output_ptr,
                offset: offset_expr,
                src: data,
                width: ctx.session.width,
                dtype: QuantPrecision::F32, predicate: None,
            });
        }
        Ok(())
    })
}

use super::instr::BoundExpr;
use super::instr::OffsetExpr;

/// **ComputePattern 自动分发** (`try_dispatch_by_compute_pattern`): Auto-driven ops dispatched
///   by ComputePattern from registry — NormLike, Reduction, BinaryElementwise.
///
/// **OpKind 专用分发** (match &op.kind): Composite ops with OpKind-specific lowering:
///   Gemm/GemmBias, QuantGemm, MHA, RoPE, MoEGate/MoERouter/MoEDispatchPacked,
///   AltUpPredict/AltUpCorrect/AltUpInject, DepthwiseConv1D, PatchEmbed, SessionKvRestore, MmHiddenInject.
///
/// Returns Ok(true) if handled, Ok(false) if the op should fall through to structural match.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_compute_pattern(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    rope_cache_offset: Option<usize>,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
) -> Result<bool, CompilerError> {
    // Phase 4+: Op v2 lowering 入口（胖 opcode 驱动）。
    // 返回 true 表示已处理，跳过现有 OpKind 反查路径。
    if super::plan_lower::lower_op_v2(prog, op, graph, ctx, resolver, abi)? {
        return Ok(true);
    }

    // 死代码（51064 测试验证零触发）：lower_op_v2 处理所有 ops。
    // 如果到达这里，说明有新 op 未被 lower_op_v2 覆盖——需要补充 Op v2 路径。
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: VRegId equality and hashing ──

    #[test]
    fn vreg_id_equality_and_hash_consistency() {
        // Arrange
        let a = VRegId(0);
        let b = VRegId(0);
        let c = VRegId(1);

        // Act & Assert
        assert_eq!(a, b, "VRegId(0) should equal VRegId(0)");
        assert_ne!(a, c, "VRegId(0) should not equal VRegId(1)");
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(!set.insert(b), "duplicate VRegId(0) should not be re-inserted");
        assert!(set.insert(c));
        assert_eq!(set.len(), 2);
    }

    // ── Test 2: SimdWidth::f32_lanes covers all variants ──

    #[test]
    fn simd_width_f32_lanes_all_variants() {
        // Arrange & Act
        let scalar = SimdWidth::Scalar.f32_lanes();
        let w128 = SimdWidth::W128.f32_lanes();
        let w256 = SimdWidth::W256.f32_lanes();
        let w512 = SimdWidth::W512.f32_lanes();
        let warp = SimdWidth::Warp(32).f32_lanes();
        let scalable = SimdWidth::Scalable.f32_lanes();

        // Assert
        assert_eq!(scalar, 1);
        assert_eq!(w128, 4);
        assert_eq!(w256, 8);
        assert_eq!(w512, 16);
        assert_eq!(warp, 32);
        assert_eq!(scalable, 0, "Scalable has runtime-determined lanes");
    }

    // ── Test 3: SimdWidth::bytes derives from f32_lanes ──

    #[test]
    fn simd_width_bytes_matches_f32_lanes_times_4() {
        // Arrange & Act & Assert
        for width in [SimdWidth::Scalar, SimdWidth::W128, SimdWidth::W256, SimdWidth::W512] {
            assert_eq!(width.bytes(), width.f32_lanes() * 4);
        }
        assert_eq!(SimdWidth::Warp(64).bytes(), 64 * 4);
    }

    // ── Test 4: VRegKind Copy/Send/Sync traits ──

    #[test]
    fn vreg_kind_copy_and_static_traits() {
        // Arrange
        let kind = VRegKind::Ptr;

        // Act: copy
        let copy = kind;

        // Assert: both usable, Copy semantics
        assert_eq!(kind, copy);

        // Verify Send + Sync via type assertion
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VRegKind>();
    }

    // ── Test 5: BoundExpr variants Clone and PartialEq ──

    #[test]
    fn bound_expr_equality_and_clone() {
        // Arrange
        let a = BoundExpr::Const(42);
        let b = a.clone();
        let c = BoundExpr::Const(42);
        let sym = BoundExpr::Symbolic(SymBound { name: "seq_len".into(), max_alloc: 4096 });
        let sym2 = sym.clone();

        // Act & Assert
        assert_eq!(a, b, "cloned BoundExpr should be equal");
        assert_eq!(a, c, "same-value BoundExpr::Const should be equal");
        assert_eq!(sym, sym2, "cloned Symbolic should be equal");
        assert_ne!(a, sym, "Const and Symbolic should not be equal");
    }

    // ── Test 6: OffsetExpr::substitute_loop_offset replaces matching vreg only ──

    #[test]
    fn offset_expr_substitute_loop_offset_targeted() {
        // Arrange
        let target = VRegId(5);
        let other = VRegId(9);
        let expr = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(target)),
            Box::new(OffsetExpr::LoopOffset(other)),
        );

        // Act: substitute only target vreg
        let result = expr.substitute_loop_offset(target, 100);

        // Assert: target replaced with Const, other unchanged
        match &result {
            OffsetExpr::Add(a, b) => {
                assert_eq!(&**a, &OffsetExpr::Const(100), "target LoopOffset should become Const(100)");
                assert_eq!(&**b, &OffsetExpr::LoopOffset(other), "other LoopOffset should remain unchanged");
            }
            _ => panic!("expected Add variant"),
        }
    }

    // ── Test 7: OffsetExpr::loop_plus_const with zero constant simplifies ──

    #[test]
    fn offset_expr_loop_plus_const_zero_simplifies() {
        // Arrange
        let vreg = VRegId(3);

        // Act
        let result = OffsetExpr::loop_plus_const(vreg, 0);

        // Assert: should be plain LoopOffset, not Add
        assert_eq!(result, OffsetExpr::LoopOffset(vreg));
    }

    // ── Test 8: OffsetExpr::loop_plus_const with nonzero constant wraps in Add ──

    #[test]
    fn offset_expr_loop_plus_const_nonzero_wraps_add() {
        // Arrange
        let vreg = VRegId(7);

        // Act
        let result = OffsetExpr::loop_plus_const(vreg, 64);

        // Assert
        match &result {
            OffsetExpr::Add(a, b) => {
                assert_eq!(&**a, &OffsetExpr::LoopOffset(vreg));
                assert_eq!(&**b, &OffsetExpr::Const(64));
            }
            _ => panic!("expected Add variant, got {:?}", result),
        }
    }

    // ── Test 9: GprOperand::remap applies function to VReg variant ──

    #[test]
    fn gpr_operand_remap_vreg_and_imm() {
        // Arrange
        let vreg = GprOperand::VReg(VRegId(10));
        let imm = GprOperand::Imm(42);

        // Act
        let remapped_vreg = vreg.remap(|v| VRegId(v.0 + 100));
        let remapped_imm = imm.remap(|v| VRegId(v.0 + 100));

        // Assert
        assert_eq!(remapped_vreg, GprOperand::VReg(VRegId(110)));
        assert_eq!(remapped_imm, GprOperand::Imm(42), "Imm should be unchanged by remap");
    }

    // ── Test 10: GprOperand::vreg returns Some for VReg, None for Imm ──

    #[test]
    fn gpr_operand_vreg_accessor() {
        // Arrange
        let vreg_op = GprOperand::VReg(VRegId(5));
        let imm_op = GprOperand::Imm(99);

        // Act & Assert
        assert_eq!(vreg_op.vreg(), Some(VRegId(5)));
        assert_eq!(imm_op.vreg(), None);
    }

    // ── Test 11: GprCondition::vregs collects referenced registers ──

    #[test]
    fn gpr_condition_vregs_collects_all_variants() {
        // Arrange
        let v = VRegId(7);
        let is_null = GprCondition::IsNull(v);
        let cmp_eq = GprCondition::CmpEq(v, 10);
        let bit_clear = GprCondition::BitClear(v, 3);

        // Act & Assert
        assert_eq!(is_null.vregs(), vec![v]);
        assert_eq!(cmp_eq.vregs(), vec![v]);
        assert_eq!(bit_clear.vregs(), vec![v]);
    }

    // ── Test 12: BlockUnpackMode::block_bytes returns correct sizes ──

    #[test]
    fn block_unpack_mode_block_bytes_all_variants() {
        // Arrange & Act & Assert
        assert_eq!(BlockUnpackMode::Int8.block_bytes(), 32);
        assert_eq!(BlockUnpackMode::F16Broadcast.block_bytes(), 64);
        assert_eq!(BlockUnpackMode::SignedNibbleLow.block_bytes(), 18);
        assert_eq!(BlockUnpackMode::UnsignedNibbleHigh.block_bytes(), 18);
        assert_eq!(BlockUnpackMode::Bitpack2 { bias: 0.0 }.block_bytes(), 8);
        assert_eq!(BlockUnpackMode::Mxfp4 { scale_src: VRegId(0) }.block_bytes(), 16);
        assert_eq!(BlockUnpackMode::Nvfp4 { scale_src: VRegId(0) }.block_bytes(), 16);
    }

    // ── Test 13: KvLoadMode Default is Direct ──

    #[test]
    fn kv_load_mode_default_is_direct() {
        // Arrange & Act
        let default: KvLoadMode = KvLoadMode::default();

        // Assert
        assert_eq!(default, KvLoadMode::Direct);
    }

    // ── Test 14: VmProgram new starts empty ──

    #[test]
    fn vm_program_new_is_empty() {
        // Arrange & Act
        let prog = VmProgram::new();

        // Assert
        assert!(prog.is_empty());
        assert_eq!(prog.len(), 0);
        assert_eq!(prog.vreg_count(), 0);
    }

    // ── Test 15: VmProgram alloc_vreg emits DeclareVReg and increments counter ──

    #[test]
    fn vm_program_alloc_vreg_emits_declare_and_increments() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act
        let v0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Assert
        assert_eq!(v0, VRegId(0));
        assert_eq!(v1, VRegId(1));
        assert_eq!(prog.vreg_count(), 2);
        assert_eq!(prog.len(), 2);
        assert!(matches!(prog.instrs[0], VmInstr::DeclareVReg { id: VRegId(0), kind: VRegKind::Ptr, .. }));
        assert!(matches!(prog.instrs[1], VmInstr::DeclareVReg { id: VRegId(1), kind: VRegKind::Vec, .. }));
    }

    // ── Test 16: VmProgram emit_loop produces balanced LoopBegin and LoopEnd ──

    #[test]
    fn vm_program_emit_loop_balanced_structure() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act
        prog.emit_loop(BoundExpr::Const(10), 32, |_prog, _ctr, _off| {});

        // Assert
        assert!(prog.validate_structure().is_ok());
        assert!(prog.len() >= 3, "emit_loop should produce at least DeclareVReg*2 + LoopBegin + LoopEnd");
    }

    // ── Test 17: VmProgram validate_provenance catches undeclared vreg ──

    #[test]
    fn vm_program_validate_provenance_rejects_undeclared_vreg() {
        // Arrange
        let mut prog = VmProgram::new();
        let undeclared = VRegId(99);
        prog.emit(VmInstr::LoopEnd); // no vregs referenced
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::VecLoad {
            dst: undeclared,
            base,
            offset: OffsetExpr::Const(0),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let result = prog.validate_provenance();

        // Assert
        assert!(result.is_err(), "should detect undeclared VRegId(99)");
        assert!(result.unwrap_err().contains("v99"));
    }

    // ── Test 18: VmProgram validate_structure detects unmatched LoopEnd ──

    #[test]
    fn vm_program_validate_structure_detects_unmatched_loop_end() {
        // Arrange
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::LoopEnd);

        // Act
        let result = prog.validate_structure();

        // Assert
        assert!(result.is_err(), "unmatched LoopEnd should fail");
    }

    // ── Test 19: VmProgram append remaps vreg ids ──

    #[test]
    fn vm_program_append_remaps_vreg_ids() {
        // Arrange
        let mut prog_a = VmProgram::new();
        let a0 = prog_a.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let mut prog_b = VmProgram::new();
        let _b0 = prog_b.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let _b1 = prog_b.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        prog_a.append(prog_b);

        // Assert
        assert_eq!(prog_a.vreg_count(), 3, "should have 3 total vregs after append");
        // Original a0 still VRegId(0), prog_b's vregs remapped to VRegId(1) and VRegId(2)
        assert_eq!(a0, VRegId(0));
    }

    // ── Test 20: VRegKindCounts default is all None ──

    #[test]
    fn vreg_kind_counts_default_is_none() {
        // Arrange & Act
        let counts = VRegKindCounts::default();

        // Assert
        assert_eq!(counts.gpr_max_id, None);
        assert_eq!(counts.vec_max_id, None);
        assert_eq!(counts.mask_max_id, None);
        assert_eq!(counts.tile_max_id, None);
        assert_eq!(counts.gpr_like(), 0);
        assert_eq!(counts.vec_like(), 0);
    }

    // ── Test 21: VRegKindCounts computed from mixed vreg types ──

    #[test]
    fn vreg_kind_counts_from_mixed_program() {
        // Arrange
        let mut prog = VmProgram::new();
        prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);      // v0 gpr
        prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);        // v1 vec
        prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);  // v2 gpr
        prog.alloc_vreg(VRegKind::Vec, SimdWidth::W512);        // v3 vec
        prog.alloc_vreg(VRegKind::Mask, SimdWidth::W256);       // v4 mask

        // Act
        let counts = prog.vreg_counts_by_kind();

        // Assert
        assert_eq!(counts.gpr_max_id, Some(2), "Ptr(v0) and Counter(v2), max=2");
        assert_eq!(counts.vec_max_id, Some(3), "Vec(v1,v3), max=3");
        assert_eq!(counts.mask_max_id, Some(4), "Mask(v4), max=4");
        assert_eq!(counts.tile_max_id, None, "no Tile vregs");
        assert_eq!(counts.gpr_like(), 3);
        assert_eq!(counts.vec_like(), 4);
    }

    // ── Test 22: GprBranchAction remap and vregs ──

    #[test]
    fn gpr_branch_action_remap_and_vregs() {
        // Arrange
        let exit = GprBranchAction::Exit(VRegId(3));
        let skip = GprBranchAction::Skip(5);

        // Act
        let remapped_exit = exit.clone().remap(|v| VRegId(v.0 + 10));
        let remapped_skip = skip.clone().remap(|v| VRegId(v.0 + 10));

        // Assert
        match remapped_exit {
            GprBranchAction::Exit(v) => assert_eq!(v, VRegId(13)),
            other => panic!("expected Exit, got {:?}", other),
        }
        match remapped_skip {
            GprBranchAction::Skip(n) => assert_eq!(n, 5),
            other => panic!("expected Skip, got {:?}", other),
        }
        assert_eq!(exit.vregs(), vec![VRegId(3)]);
        assert_eq!(skip.vregs(), Vec::<VRegId>::new());
    }

    // ── Test 23: BlockUnpackMode remap_vregs and vregs ──

    #[test]
    fn block_unpack_mode_remap_vregs_and_vregs_accessors() {
        // Arrange
        let scale_src = VRegId(7);
        let mxfp4 = BlockUnpackMode::Mxfp4 { scale_src };
        let plain = BlockUnpackMode::Int8;

        // Act
        let remapped = mxfp4.remap_vregs(|v| VRegId(v.0 * 2));
        let remapped_plain = plain.remap_vregs(|v| VRegId(v.0 * 2));

        // Assert
        match remapped {
            BlockUnpackMode::Mxfp4 { scale_src } => assert_eq!(scale_src, VRegId(14)),
            other => panic!("expected Mxfp4, got {:?}", other),
        }
        match remapped_plain {
            BlockUnpackMode::Int8 => {}
            other => panic!("expected Int8, got {:?}", other),
        }
        assert_eq!(mxfp4.vregs(), vec![VRegId(7)]);
        assert_eq!(plain.vregs(), Vec::<VRegId>::new());
    }

    // ── Test 24: OffsetExpr substitute_loop_offset nested Add/Mul ──

    #[test]
    fn offset_expr_substitute_loop_offset_nested_add_mul() {
        // Arrange
        let target = VRegId(2);
        let inner = OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(target)), 4);
        let expr = OffsetExpr::Add(Box::new(OffsetExpr::Const(100)), Box::new(inner));

        // Act
        let result = expr.substitute_loop_offset(target, 50);

        // Assert
        match &result {
            OffsetExpr::Add(a, b) => {
                assert_eq!(&**a, &OffsetExpr::Const(100));
                match &**b {
                    OffsetExpr::Mul(inner, scale) => {
                        assert_eq!(&**inner, &OffsetExpr::Const(50));
                        assert_eq!(*scale, 4);
                    }
                    other => panic!("expected Mul, got {:?}", other),
                }
            }
            other => panic!("expected Add, got {:?}", other),
        }
    }

    // ── Test 25: GprCondition remap propagates through all variants ──

    #[test]
    fn gpr_condition_remap_all_variants() {
        // Arrange
        let v = VRegId(1);
        let cases = vec![
            GprCondition::IsNull(v),
            GprCondition::IsNonNull(v),
            GprCondition::CmpEq(v, 42),
            GprCondition::CmpLtU(v, 100),
            GprCondition::BitClear(v, 3),
            GprCondition::BitSet(v, 5),
        ];

        // Act & Assert
        for cond in cases {
            let remapped = cond.remap(|id| VRegId(id.0 + 20));
            assert_eq!(remapped.vregs(), vec![VRegId(21)]);
        }
    }

    // ── Test 26: BoundExpr DynamicVReg equality semantics ──

    #[test]
    fn bound_expr_dynamic_vreg_plus_one_distinct_from_dynamic_vreg() {
        // Arrange
        let v = VRegId(5);
        let a = BoundExpr::DynamicVReg(v);
        let b = BoundExpr::DynamicVRegPlusOne(v);

        // Act & Assert
        assert_ne!(a, b, "DynamicVReg and DynamicVRegPlusOne are distinct variants");
        assert_eq!(a.clone(), a, "DynamicVReg clones as equal");
    }

    // ── Test 27: VecOp all variants are distinct and Copy ──

    #[test]
    fn vec_op_all_variants_distinct() {
        // Arrange
        let ops = [
            VecOp::Add, VecOp::Sub, VecOp::Mul, VecOp::Div,
            VecOp::Max, VecOp::Min, VecOp::And, VecOp::Or,
            VecOp::Xor, VecOp::AndNot, VecOp::Shl, VecOp::Shr, VecOp::Not,
        ];

        // Act & Assert — all pairwise distinct
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "VecOp variants {} and {} should differ", i, j);
            }
        }
        // Copy works
        let copy = ops[0];
        assert_eq!(ops[0], copy);
    }

    // ── Test 28: VecShiftDir equality and Copy ──

    #[test]
    fn vec_shift_dir_variants() {
        // Arrange & Act & Assert
        assert_ne!(VecShiftDir::Left, VecShiftDir::Right);
        assert_eq!(VecShiftDir::Left, VecShiftDir::Left);
        let copied = VecShiftDir::Right;
        assert_eq!(VecShiftDir::Right, copied);
    }

    // ── Test 29: VecUnaryOp covers conversion variants ──

    #[test]
    fn vec_unary_op_fp8_conversion_variants() {
        // Arrange
        let ops = [
            VecUnaryOp::Neg, VecUnaryOp::Abs, VecUnaryOp::Sqrt, VecUnaryOp::Rsqrt,
            VecUnaryOp::Recip, VecUnaryOp::Floor, VecUnaryOp::Ceil, VecUnaryOp::Round,
            VecUnaryOp::IntToFloat, VecUnaryOp::Fp8E4M3ToFloat, VecUnaryOp::Fp8E5M2ToFloat,
        ];

        // Act & Assert — all pairwise distinct
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "VecUnaryOp variants {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 30: ReduceOp all variants distinct ──

    #[test]
    fn reduce_op_all_variants_distinct() {
        // Arrange
        let ops = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod, ReduceOp::LogSum];

        // Act & Assert
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "ReduceOp variants {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 31: MemFenceOrder all variants distinct ──

    #[test]
    fn mem_fence_order_all_variants_distinct() {
        // Arrange
        let orders = [
            MemFenceOrder::Release, MemFenceOrder::Acquire,
            MemFenceOrder::AcqRel, MemFenceOrder::SeqCst,
        ];

        // Act & Assert
        for i in 0..orders.len() {
            for j in (i + 1)..orders.len() {
                assert_ne!(orders[i], orders[j], "MemFenceOrder {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 32: DotDtype all variants distinct ──

    #[test]
    fn dot_dtype_all_variants_distinct() {
        // Arrange
        let dtypes = [DotDtype::Bf16, DotDtype::Fp16, DotDtype::Int8, DotDtype::Int4x8, DotDtype::Fp4];

        // Act & Assert
        for i in 0..dtypes.len() {
            for j in (i + 1)..dtypes.len() {
                assert_ne!(dtypes[i], dtypes[j], "DotDtype {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 33: BiPlaneMode all variants distinct ──

    #[test]
    fn bi_plane_mode_all_variants_distinct() {
        // Arrange
        let modes = [BiPlaneMode::Low5, BiPlaneMode::Low6, BiPlaneMode::Q3Merge];

        // Act & Assert
        assert_ne!(modes[0], modes[1]);
        assert_ne!(modes[1], modes[2]);
        assert_ne!(modes[0], modes[2]);
    }

    // ── Test 34: ScalarCvtSource all variants distinct ──

    #[test]
    fn scalar_cvt_source_all_variants_distinct() {
        // Arrange
        let sources = [ScalarCvtSource::F16, ScalarCvtSource::I8, ScalarCvtSource::U8];

        // Act & Assert
        assert_ne!(sources[0], sources[1]);
        assert_ne!(sources[1], sources[2]);
        assert_ne!(sources[0], sources[2]);
    }

    // ── Test 35: VecShuffleMask remap for Const and Dynamic ──

    #[test]
    fn vec_shuffle_mask_remap_const_and_dynamic() {
        // Arrange
        let const_mask = VecShuffleMask::Const(vec![0, 1, 2, 3]);
        let dynamic_mask = VecShuffleMask::Dynamic { ctrl: VRegId(5) };

        // Act
        let remapped_const = const_mask.remap(&|v| VRegId(v.0 + 100));
        let remapped_dynamic = dynamic_mask.remap(&|v| VRegId(v.0 + 100));

        // Assert — Const is unchanged, Dynamic is remapped
        match remapped_const {
            VecShuffleMask::Const(v) => assert_eq!(v, vec![0, 1, 2, 3]),
            other => panic!("expected Const, got {:?}", other),
        }
        match remapped_dynamic {
            VecShuffleMask::Dynamic { ctrl } => assert_eq!(ctrl, VRegId(105)),
            other => panic!("expected Dynamic, got {:?}", other),
        }
    }

    // ── Test 36: MemOrdering all variants distinct ──

    #[test]
    fn mem_ordering_all_variants_distinct() {
        // Arrange
        let orderings = [
            MemOrdering::Relaxed, MemOrdering::Acquire, MemOrdering::Release,
            MemOrdering::AcqRel, MemOrdering::SeqCst,
        ];

        // Act & Assert
        for i in 0..orderings.len() {
            for j in (i + 1)..orderings.len() {
                assert_ne!(orderings[i], orderings[j], "MemOrdering {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 37: HotpatchTarget Debug and Clone ──

    #[test]
    fn hotpatch_target_debug_and_clone() {
        // Arrange
        let a = HotpatchTarget::InstrIndex(42);
        let b = HotpatchTarget::ExternalAddr(0xDEADBEEF);

        // Act
        let a_clone = a.clone();

        // Assert — Debug formats contain variant content
        let debug_a = format!("{:?}", a);
        assert!(debug_a.contains("42"));
        let debug_b = format!("{:?}", b);
        assert!(debug_b.contains("3735928559"), "ExternalAddr debug: {}", debug_b);
        match a_clone {
            HotpatchTarget::InstrIndex(n) => assert_eq!(n, 42),
            other => panic!("expected InstrIndex, got {:?}", other),
        }
    }

    // ── Test 38: JumpTarget field access and Clone ──

    #[test]
    fn jump_target_field_access_and_clone() {
        // Arrange
        let target = JumpTarget { expert_id: 3, instr_index: 128 };

        // Act
        let cloned = target.clone();

        // Assert
        assert_eq!(target.expert_id, 3);
        assert_eq!(target.instr_index, 128);
        assert_eq!(cloned.expert_id, 3);
        assert_eq!(cloned.instr_index, 128);
    }

    // ── Test 39: GprUnaryOpKind all variants distinct ──

    #[test]
    fn gpr_unary_op_kind_all_variants_distinct() {
        // Arrange
        let ops = [GprUnaryOpKind::Not, GprUnaryOpKind::Popcount, GprUnaryOpKind::Clz,
                   GprUnaryOpKind::Bswap, GprUnaryOpKind::Neg];

        // Act & Assert
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "GprUnaryOpKind {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 40: PtrExpr Debug for key variants ──

    #[test]
    fn ptr_expr_debug_formats() {
        // Arrange
        let abi = PtrExpr::AbiArg(0);
        let stack = PtrExpr::StackArg(16);
        let vreg_off = PtrExpr::VRegPlusConst(VRegId(3), 64);
        let named = PtrExpr::NamedArg("telemetry".to_string());
        let abs = PtrExpr::AbsAddr(0xCAFE);

        // Act & Assert — just ensure Debug doesn't panic and contains hints
        assert!(format!("{:?}", abi).contains('0'));
        assert!(format!("{:?}", stack).contains("16"));
        assert!(format!("{:?}", vreg_off).contains("64"));
        assert!(format!("{:?}", named).contains("telemetry"));
        assert!(format!("{:?}", abs).contains("51966"), "AbsAddr debug: {:?}", abs);
    }

    // ── Test 41: ScalarExpr Debug for all variants ──

    #[test]
    fn scalar_expr_debug_formats() {
        // Arrange
        let const_val = ScalarExpr::Const(3.14);
        let mem_load = ScalarExpr::MemLoad(VRegId(1), OffsetExpr::Const(32));
        let extract = ScalarExpr::ExtractLane0(VRegId(2));
        let vreg = ScalarExpr::VReg(VRegId(5));

        // Act & Assert — Debug works without panic
        let _ = format!("{:?}", const_val);
        let _ = format!("{:?}", mem_load);
        let _ = format!("{:?}", extract);
        let _ = format!("{:?}", vreg);
    }

    // ── Test 42: CmpPredicate all variants distinct ──

    #[test]
    fn cmp_predicate_all_variants_distinct() {
        // Arrange
        let preds = [CmpPredicate::Eq, CmpPredicate::Ne, CmpPredicate::Lt,
                     CmpPredicate::Le, CmpPredicate::Gt, CmpPredicate::Ge];

        // Act & Assert
        for i in 0..preds.len() {
            for j in (i + 1)..preds.len() {
                assert_ne!(preds[i], preds[j], "CmpPredicate {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 43: TranscendentalFn all variants distinct ──

    #[test]
    fn transcendental_fn_all_variants_distinct() {
        // Arrange
        let fns = [TranscendentalFn::Exp, TranscendentalFn::Log, TranscendentalFn::Tanh,
                   TranscendentalFn::Sigmoid, TranscendentalFn::Fwht];

        // Act & Assert
        for i in 0..fns.len() {
            for j in (i + 1)..fns.len() {
                assert_ne!(fns[i], fns[j], "TranscendentalFn {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 44: ReturnValue Debug for both variants ──

    #[test]
    fn return_value_debug_for_both_variants() {
        // Arrange
        let const_ret = ReturnValue::Const(0);
        let vreg_ret = ReturnValue::VReg(VRegId(7));

        // Act
        let debug_const = format!("{:?}", const_ret);
        let debug_vreg = format!("{:?}", vreg_ret);

        // Assert
        assert!(debug_const.contains('0'));
        assert!(debug_vreg.contains('7'));
    }

    // ── Test 45: Fp8Kind E4M3 and E5M2 are distinct and Copy ──

    #[test]
    fn fp8_kind_variants_distinct_and_copy() {
        // Arrange
        let e4m3 = Fp8Kind::E4M3;
        let e5m2 = Fp8Kind::E5M2;

        // Act & Assert
        assert_ne!(e4m3, e5m2, "E4M3 and E5M2 should be distinct");
        assert_eq!(e4m3, Fp8Kind::E4M3);
        assert_eq!(e5m2, Fp8Kind::E5M2);
        // Copy works
        let copied = e4m3;
        assert_eq!(e4m3, copied);
    }

    // ── Test 46: TmaSwizzle all variants distinct ──

    #[test]
    fn tma_swizzle_all_variants_distinct() {
        // Arrange
        let modes = [TmaSwizzle::None, TmaSwizzle::Swizzle32,
                     TmaSwizzle::Swizzle64, TmaSwizzle::Swizzle128];

        // Act & Assert
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert_ne!(modes[i], modes[j], "TmaSwizzle {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 47: GprOp all variants distinct and Copy ──

    #[test]
    fn gpr_op_all_variants_distinct() {
        // Arrange
        let ops = [GprOp::Add, GprOp::Sub, GprOp::Mul, GprOp::Div,
                   GprOp::Shl, GprOp::Shr, GprOp::And, GprOp::Or,
                   GprOp::Xor, GprOp::BitTest];

        // Act & Assert
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "GprOp {} and {} should differ", i, j);
            }
        }
        let copied = ops[0];
        assert_eq!(ops[0], copied);
    }

    // ── Test 48: KvLoadMode all variants distinct and Default is Direct ──

    #[test]
    fn kv_load_mode_all_variants_distinct() {
        // Arrange
        let modes = [KvLoadMode::Direct, KvLoadMode::Kivi4, KvLoadMode::Kivi2,
                     KvLoadMode::Sparse, KvLoadMode::Auto];

        // Act & Assert
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert_ne!(modes[i], modes[j], "KvLoadMode {} and {} should differ", i, j);
            }
        }
        assert_eq!(KvLoadMode::default(), KvLoadMode::Direct);
    }

    // ── Test 49: OffsetExpr::ScalarVReg not substituted by substitute_loop_offset ──

    #[test]
    fn offset_expr_scalar_vreg_not_substituted() {
        // Arrange
        let target = VRegId(3);
        let expr = OffsetExpr::ScalarVReg(target);

        // Act: try substituting with target vreg — should be unchanged
        let result = expr.substitute_loop_offset(target, 999);

        // Assert
        assert_eq!(result, OffsetExpr::ScalarVReg(target),
                   "ScalarVReg should not be affected by substitute_loop_offset");
    }

    // ── Test 50: VmProgram alloc_label increments monotonically ──

    #[test]
    fn vm_program_alloc_label_monotonic_increment() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act
        let l0 = prog.alloc_label();
        let l1 = prog.alloc_label();
        let l2 = prog.alloc_label();

        // Assert
        assert!(l0 >= 1000, "first label should start at >= 1000, got {}", l0);
        assert_eq!(l1, l0 + 1, "labels should increment by 1");
        assert_eq!(l2, l0 + 2, "labels should increment by 1");
    }

    // ── Test 51: VmProgram emit_scope produces ScopeBegin and ScopeEnd ──

    #[test]
    fn vm_program_emit_scope_balanced_structure() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act
        let result: Result<(), std::convert::Infallible> = prog.emit_scope(|p| {
            let v = p.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            p.emit(VmInstr::LoopEnd); // placeholder, just needs vreg
            let _ = v;
            Ok(())
        });

        // Assert
        assert!(result.is_ok());
        // Should contain ScopeBegin, DeclareVReg, LoopEnd, ScopeEnd
        assert!(prog.len() >= 4);
        assert!(matches!(prog.instrs.first(), Some(VmInstr::ScopeBegin { .. })));
        assert!(matches!(prog.instrs.last(), Some(VmInstr::ScopeEnd { .. })));
    }

    // ── Test 52: VmInstr is_meta correctly classifies meta instructions ──

    #[test]
    fn vm_instr_is_meta_classification() {
        // Arrange — meta instructions
        let meta_instrs = vec![
            VmInstr::DeclareVReg { id: VRegId(0), kind: VRegKind::Ptr, width: SimdWidth::Scalar },
            VmInstr::ReleaseVReg { id: VRegId(0) },
            VmInstr::MarkLabel { label_id: 0 },
            VmInstr::Comment("test".into()),
        ];
        // Arrange — non-meta instruction
        let non_meta = VmInstr::LoopEnd;

        // Act & Assert
        for instr in &meta_instrs {
            assert!(instr.is_meta(), "{:?} should be meta", instr);
        }
        assert!(!non_meta.is_meta(), "LoopEnd should not be meta");
    }

    // ── Test 53: VmProgram emit_loop_try propagates errors ──

    #[test]
    fn vm_program_emit_loop_try_propagates_error() {
        // Arrange
        let mut prog = VmProgram::new();
        let err_msg = "test error";

        // Act
        let result: Result<(), &str> = prog.emit_loop_try(BoundExpr::Const(4), 16, |_p, _ctr, _off| {
            Err(err_msg)
        });

        // Assert
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), err_msg);
        // LoopBegin should still have been emitted before the error
        assert!(prog.len() >= 2, "should have DeclareVReg*2 + LoopBegin before error");
    }

    // ── Test 54: VRegKind all variants distinct ──

    #[test]
    fn vreg_kind_all_variants_distinct() {
        // Arrange
        let kinds = [VRegKind::Ptr, VRegKind::Vec, VRegKind::Scalar,
                     VRegKind::Counter, VRegKind::ByteOffset,
                     VRegKind::Tile, VRegKind::Mask];

        // Act & Assert
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(kinds[i], kinds[j], "VRegKind {} and {} should differ", i, j);
            }
        }
    }

    // ── Test 55: validate_type_consistency rejects Vec-typed base in VecLoad ──

    #[test]
    fn validate_type_consistency_rejects_vec_base_in_vecload() {
        // Arrange: base is declared as Vec (not GPR class) — invalid for memory base
        let mut prog = VmProgram::new();
        let vec_base = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec_dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec_dst, base: vec_base,
            offset: OffsetExpr::Const(0), width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert
        assert!(result.is_err(), "Vec-typed base in VecLoad should fail type check");
        let err = result.unwrap_err();
        assert!(err.contains("base") && err.contains("Vec"), "error should mention base and Vec, got: {}", err);
    }

    // ── Test 56: validate_type_consistency rejects GPR-typed operands in Fma ──

    #[test]
    fn validate_type_consistency_rejects_gpr_in_fma() {
        // Arrange: acc is Ptr (GPR class), but Fma requires all Vec
        let mut prog = VmProgram::new();
        let acc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Fma { dst, acc, a, b, dtype: QuantPrecision::F32 });

        // Act
        let result = prog.validate_type_consistency();

        // Assert
        assert!(result.is_err(), "GPR-typed acc in Fma should fail type check");
        let err = result.unwrap_err();
        assert!(err.contains("acc") || err.contains("Fma"), "error should reference Fma or acc, got: {}", err);
    }

    // ── Test 57: validate_width_consistency detects mismatch in Fma operands ──

    #[test]
    fn validate_width_consistency_detects_fma_width_mismatch() {
        // Arrange: dst=W256 but acc=W512 — width mismatch
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W512);
        let a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Fma { dst, acc, a, b, dtype: QuantPrecision::F32 });

        // Act
        let result = prog.validate_width_consistency();

        // Assert
        assert!(result.is_err(), "width mismatch in Fma should fail");
        let err = result.unwrap_err();
        assert!(err.contains("Fma"), "error should reference Fma, got: {}", err);
    }

    // ── Test 58: validate_width_consistency passes for consistent VecBinOp program ──

    #[test]
    fn validate_width_consistency_passes_consistent_program() {
        // Arrange: all Vec vregs have same width
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecBinOp { dst, a, b, op: VecOp::Add, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::VecUnaryOp { dst, a, op: VecUnaryOp::Neg });

        // Act
        let result = prog.validate_width_consistency();

        // Assert
        assert!(result.is_ok(), "uniform-width program should pass width check, got: {:?}", result);
    }

    // ── Test 59: validate_declares_before_uses catches late-declared vreg ──

    #[test]
    fn validate_declares_before_uses_catches_late_declare() {
        // Arrange: use VRegId(99) in VecLoad, then DeclareVReg it afterward
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Use v99 before declaring it (simulating an opt pass bug)
        prog.emit(VmInstr::VecLoad {
            dst: VRegId(99), base,
            offset: OffsetExpr::Const(0), width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });
        // Late declare (should have been before the VecLoad)
        prog.emit(VmInstr::DeclareVReg { id: VRegId(99), kind: VRegKind::Vec, width: SimdWidth::W256 });

        // Act
        let result = prog.validate_declares_before_uses();

        // Assert
        assert!(result.is_err(), "late DeclareVReg should be caught");
        let err = result.unwrap_err();
        assert!(err.contains("v99") || err.contains("before"), "error should reference v99 or 'before', got: {}", err);
    }

    // ── Test 60: validate_value_domains rejects VecData used as VecLoad base ──

    #[test]
    fn validate_value_domains_rejects_vecdata_as_base() {
        // Arrange: emit VecBinOp so v1 gets VecData domain, then use v1 as VecLoad base
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);    // v0
        let vec_a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);    // v1
        let vec_b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);    // v2
        let vec_c = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);    // v3
        // v1 = v1 + v2 — now v1 has VecData domain
        prog.emit(VmInstr::VecBinOp { dst: vec_a, a: vec_b, b: vec_c, op: VecOp::Add, dtype: QuantPrecision::F32 });
        // Use v1 (VecData) as base for VecLoad — should be rejected
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);      // v4
        prog.emit(VmInstr::VecLoad {
            dst, base: vec_a,
            offset: OffsetExpr::Const(0), width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let result = prog.validate_value_domains();

        // Assert
        assert!(result.is_err(), "VecData domain as VecLoad base should fail");
        let err = result.unwrap_err();
        assert!(err.contains("VecLoad") && err.contains("SIGSEGV"),
                "error should mention VecLoad and SIGSEGV, got: {}", err);
    }

    // ── Test 61: validate_structure detects unmatched ScopeEnd ──

    #[test]
    fn validate_structure_detects_unmatched_scope_end() {
        // Arrange: emit ScopeEnd without ScopeBegin
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeEnd { scope_id: 42 });

        // Act
        let result = prog.validate_structure();

        // Assert
        assert!(result.is_err(), "unmatched ScopeEnd should fail structure check");
        let err = result.unwrap_err();
        assert!(err.contains("ScopeEnd") || err.contains("scope"),
                "error should reference scope, got: {}", err);
    }

    // ── Test 62: append_with_mapping correctly remaps template vregs in VecBinOp ──

    #[test]
    fn append_with_mapping_remapping_preserves_semantics() {
        // Arrange: main program has v0(Vec), template has v0(Vec)
        let mut main_prog = VmProgram::new();
        let main_v = prog_alloc_vec(&mut main_prog); // v0

        let mut tpl_prog = VmProgram::new();
        let tpl_v0 = prog_alloc_vec(&mut tpl_prog);  // tpl v0
        let tpl_v1 = prog_alloc_vec(&mut tpl_prog);  // tpl v1
        tpl_prog.emit(VmInstr::VecBinOp {
            dst: tpl_v0, a: tpl_v1, b: tpl_v0,
            op: VecOp::Mul, dtype: QuantPrecision::F32,
        });

        // Act: map tpl_v1 → main_v, tpl_v0 auto-allocated
        let pre_len = main_prog.len();
        main_prog.append_with_mapping(tpl_prog, &[(tpl_v1, main_v)]);

        // Assert: new instructions appended after existing ones
        assert!(main_prog.len() > pre_len, "instructions should be appended");
        // The appended VecBinOp should use the mapped vreg for tpl_v1
        let bin_ops: Vec<&VmInstr> = main_prog.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecBinOp { op: VecOp::Mul, .. } => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(bin_ops.len(), 1, "should have exactly one Mul VecBinOp");
        if let VmInstr::VecBinOp { a, .. } = bin_ops[0] {
            assert_eq!(*a, main_v, "tpl_v1 should be mapped to main_v (v0)");
        }
    }

    // ── Test 63: VRegKindCounts tile_like and mask_like from mixed program ──

    #[test]
    fn vreg_kind_counts_tile_and_mask_like() {
        // Arrange: program with Tile and Mask vregs
        let mut prog = VmProgram::new();
        prog.alloc_vreg(VRegKind::Tile, SimdWidth::W256);   // v0
        prog.alloc_vreg(VRegKind::Tile, SimdWidth::W256);   // v1
        prog.alloc_vreg(VRegKind::Mask, SimdWidth::W256);   // v2

        // Act
        let counts = prog.vreg_counts_by_kind();

        // Assert
        assert_eq!(counts.tile_max_id, Some(1), "two Tile vregs: v0, v1, max=1");
        assert_eq!(counts.mask_max_id, Some(2), "one Mask vreg: v2, max=2");
        assert_eq!(counts.tile_like(), 2, "tile_like = max+1 = 2");
        assert_eq!(counts.mask_like(), 3, "mask_like = max+1 = 3");
        assert_eq!(counts.gpr_max_id, None, "no GPR vregs");
    }

    // ── Test 64: validate_provenance passes for well-formed load-store program ──

    #[test]
    fn validate_provenance_passes_well_formed_program() {
        // Arrange: build a minimal valid program: alloc base, alloc dst, VecLoad, VecStore
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let out_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::VecLoad {
            dst, base,
            offset: OffsetExpr::Const(0), width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: out_base, src: dst,
            offset: OffsetExpr::Const(0), width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let result = prog.validate_provenance();

        // Assert
        assert!(result.is_ok(), "well-formed program should pass provenance check, got: {:?}", result);
    }

    // Helper: allocate a Vec W256 vreg (reduces boilerplate in tests)
    fn prog_alloc_vec(prog: &mut VmProgram) -> VRegId {
        prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256)
    }

    // ── Test 65: validate_type_consistency rejects width mismatch in VecNarrow ──

    #[test]
    fn validate_type_consistency_rejects_vec_narrow_width_mismatch() {
        // Arrange: src is W512 but dst is W256 — width mismatch in narrow
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W512);
        prog.emit(VmInstr::VecNarrow {
            dst, src,
            dst_dtype: QuantPrecision::BF16,
            src_dtype: QuantPrecision::F32,
            width: SimdWidth::W256,
        });

        // Act
        let result = prog.validate_width_consistency();

        // Assert
        assert!(result.is_err(), "width mismatch in VecNarrow should fail");
        let err = result.unwrap_err();
        assert!(err.contains("VecNarrow"), "error should reference VecNarrow, got: {}", err);
    }

    // ── Test 66: VmProgram append_with_mapping with empty map auto-remaps all vregs ──

    #[test]
    fn append_with_mapping_empty_map_auto_remaps_all() {
        // Arrange: main program has 2 vregs, template has 3 vregs
        let mut main_prog = VmProgram::new();
        let _main_v0 = prog_alloc_vec(&mut main_prog); // v0
        let _main_v1 = prog_alloc_vec(&mut main_prog); // v1

        let mut tpl = VmProgram::new();
        let tpl_v0 = prog_alloc_vec(&mut tpl); // tpl v0
        let tpl_v1 = prog_alloc_vec(&mut tpl); // tpl v1
        let tpl_v2 = prog_alloc_vec(&mut tpl); // tpl v2
        tpl.emit(VmInstr::VecBinOp {
            dst: tpl_v2, a: tpl_v0, b: tpl_v1,
            op: VecOp::Add, dtype: QuantPrecision::F32,
        });

        // Act: empty mapping -> all template vregs auto-remapped
        let pre_vreg_count = main_prog.vreg_count();
        main_prog.append_with_mapping(tpl, &[]);

        // Assert: total vregs = 2 (main) + 3 (template auto-remapped)
        assert_eq!(main_prog.vreg_count(), pre_vreg_count + 3,
                   "should have main + remapped template vregs");
        // The appended VecBinOp should reference remapped vregs, all >= pre_vreg_count
        let bin_ops: Vec<&VmInstr> = main_prog.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecBinOp { op: VecOp::Add, .. } => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(bin_ops.len(), 1, "should have exactly one Add VecBinOp");
        if let VmInstr::VecBinOp { dst, a, b, .. } = bin_ops[0] {
            // All remapped vregs should be above the original main_prog vregs
            assert!(a.0 >= pre_vreg_count, "tpl_v0 should be remapped above main vregs");
            assert!(b.0 >= pre_vreg_count, "tpl_v1 should be remapped above main vregs");
            assert!(dst.0 >= pre_vreg_count, "tpl_v2 should be remapped above main vregs");
            // All three should be distinct
            assert_ne!(*a, *b, "remapped a and b should be distinct");
            assert_ne!(*a, *dst, "remapped a and dst should be distinct");
            assert_ne!(*b, *dst, "remapped b and dst should be distinct");
        }
    }

    // ── Test 67: VmProgram multiple nested scopes pass structure validation ──

    #[test]
    fn vm_program_nested_scopes_pass_structure_validation() {
        // Arrange: two nested scopes inside a loop
        let mut prog = VmProgram::new();

        // Act: outer loop -> inner scope 1 -> inner scope 2
        prog.emit_loop(BoundExpr::Const(4), 32, |p, _ctr, _off| {
            let result: Result<(), std::convert::Infallible> = p.emit_scope(|inner| {
                let v = inner.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                inner.emit(VmInstr::ScalarStore {
                    base: v, src: v, offset: OffsetExpr::Const(0),
                });
                Ok(())
            });
            assert!(result.is_ok());
            let result2: Result<(), std::convert::Infallible> = p.emit_scope(|inner2| {
                let v2 = inner2.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
                inner2.emit(VmInstr::VecStore {
                    base: v2, src: v2, offset: OffsetExpr::Const(0),
                    width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
                });
                Ok(())
            });
            assert!(result2.is_ok());
        });

        // Assert
        assert!(prog.validate_structure().is_ok(),
                "nested scopes inside loop should pass structure validation");
        assert!(prog.len() > 10, "should have substantial instruction count");
    }

    // ── Test 68: SimdWidth Warp variants with different lane counts ──

    #[test]
    fn simd_width_warp_variants_differ_by_lane_count() {
        // Arrange
        let warp32 = SimdWidth::Warp(32);
        let warp64 = SimdWidth::Warp(64);
        let warp128 = SimdWidth::Warp(128);

        // Act & Assert
        assert_ne!(warp32, warp64, "Warp(32) and Warp(64) should differ");
        assert_ne!(warp64, warp128, "Warp(64) and Warp(128) should differ");
        assert_eq!(warp32.f32_lanes(), 32);
        assert_eq!(warp64.f32_lanes(), 64);
        assert_eq!(warp128.f32_lanes(), 128);
        assert_eq!(warp32.bytes(), 32 * 4);
        assert_eq!(warp64.bytes(), 64 * 4);
    }

    // ── Test 69: BoundExpr::Symbolic SymBound field access and clone ──

    #[test]
    fn bound_expr_symbolic_symbound_fields_and_clone() {
        // Arrange
        let sym = SymBound { name: "batch_size".into(), max_alloc: 8192 };
        let expr = BoundExpr::Symbolic(sym.clone());

        // Act
        let cloned = expr.clone();

        // Assert
        assert_eq!(cloned, expr, "cloned Symbolic should be equal");
        if let BoundExpr::Symbolic(inner) = cloned {
            assert_eq!(inner.name, "batch_size");
            assert_eq!(inner.max_alloc, 8192);
        } else {
            panic!("expected Symbolic variant");
        }
        // Distinct from Const and DynamicVReg
        assert_ne!(expr, BoundExpr::Const(8192));
        assert_ne!(expr, BoundExpr::DynamicVReg(VRegId(0)));
    }

    // ── Test 70: OffsetExpr deeply nested substitute_loop_offset ──

    #[test]
    fn offset_expr_deeply_nested_substitute() {
        // Arrange: Add(Const(16), Add(Mul(LoopOffset(target), 4), LoopOffset(other)))
        let target = VRegId(1);
        let other = VRegId(2);
        let inner = OffsetExpr::Add(
            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(target)), 4)),
            Box::new(OffsetExpr::LoopOffset(other)),
        );
        let outer = OffsetExpr::Add(
            Box::new(OffsetExpr::Const(16)),
            Box::new(inner),
        );

        // Act: substitute target -> 200
        let result = outer.substitute_loop_offset(target, 200);

        // Assert: inner Mul should become Mul(Const(200), 4), other unchanged
        match &result {
            OffsetExpr::Add(a, b) => {
                assert_eq!(&**a, &OffsetExpr::Const(16), "outer Const should be unchanged");
                match &**b {
                    OffsetExpr::Add(inner_a, inner_b) => {
                        match &**inner_a {
                            OffsetExpr::Mul(base, scale) => {
                                assert_eq!(&**base, &OffsetExpr::Const(200),
                                           "target should be substituted to Const(200)");
                                assert_eq!(*scale, 4, "scale should be unchanged");
                            }
                            other => panic!("expected Mul, got {:?}", other),
                        }
                        assert_eq!(&**inner_b, &OffsetExpr::LoopOffset(other),
                                   "other LoopOffset should be unchanged");
                    }
                    other => panic!("expected inner Add, got {:?}", other),
                }
            }
            other => panic!("expected outer Add, got {:?}", other),
        }
    }

    // ── Test 71: VmInstr::Mov type consistency rejects Ptr-to-Vec copy ──

    #[test]
    fn validate_type_consistency_rejects_ptr_to_vec_mov() {
        // Arrange: Mov from Ptr (GPR) to Vec -- type mismatch
        let mut prog = VmProgram::new();
        let ptr_src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Mov {
            dst: vec_dst,
            src: ptr_src,
            dtype: QuantPrecision::F32,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert
        assert!(result.is_err(), "Mov from Ptr to Vec should fail type check");
        let err = result.unwrap_err();
        assert!(err.contains("Mov") || err.contains("src") || err.contains("dst"),
                "error should reference Mov or operands, got: {}", err);
    }

    // ── Test 72: VmProgram emit_loop with DynamicVReg bound creates correct structure ──

    #[test]
    fn vm_program_emit_loop_dynamic_vreg_bound_structure() {
        // Arrange
        let mut prog = VmProgram::new();
        let seq_len_vreg = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let dynamic_bound = BoundExpr::DynamicVReg(seq_len_vreg);

        // Act
        prog.emit_loop(dynamic_bound, 32, |p, _ctr, _off| {
            let v = prog_alloc_vec(p);
            p.emit(VmInstr::VecStore {
                base: v, src: v, offset: OffsetExpr::Const(0),
                width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
            });
        });

        // Assert
        assert!(prog.validate_structure().is_ok(),
                "loop with DynamicVReg bound should pass structure validation");
        // Should contain LoopBegin with DynamicVReg bound
        let has_dynamic_loop = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::LoopBegin { bound: BoundExpr::DynamicVReg(_), .. }
        ));
        assert!(has_dynamic_loop, "should contain LoopBegin with DynamicVReg bound");
    }

    // ── Test 73: BlockUnpackMode SignedNibble variants have same block_bytes ──

    #[test]
    fn block_unpack_mode_nibble_variants_same_block_bytes() {
        // Arrange
        let low = BlockUnpackMode::SignedNibbleLow;
        let high = BlockUnpackMode::UnsignedNibbleHigh;

        // Act
        let low_bytes = low.block_bytes();
        let high_bytes = high.block_bytes();

        // Assert: both nibble variants use the same block size
        assert_eq!(low_bytes, high_bytes,
                   "SignedNibbleLow and UnsignedNibbleHigh should have same block_bytes");
        assert_eq!(low_bytes, 18, "nibble block_bytes should be 18");
    }

    // ── Test 74: VmInstr is_meta correctly classifies ScopeBegin/ScopeEnd as meta ──

    #[test]
    fn vm_instr_is_meta_classifies_scope_instructions() {
        // Arrange
        let scope_begin = VmInstr::ScopeBegin { scope_id: 1 };
        let scope_end = VmInstr::ScopeEnd { scope_id: 1 };
        let loop_begin = VmInstr::LoopBegin {
            counter: VRegId(0), byte_offset: VRegId(1),
            bound: BoundExpr::Const(10), step_bytes: 32,
        };
        let vec_load = VmInstr::VecLoad {
            dst: VRegId(2), base: VRegId(0),
            offset: OffsetExpr::Const(0), width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        };

        // Act & Assert
        assert!(scope_begin.is_meta(), "ScopeBegin should be meta");
        assert!(scope_end.is_meta(), "ScopeEnd should be meta");
        assert!(!loop_begin.is_meta(), "LoopBegin should not be meta");
        assert!(!vec_load.is_meta(), "VecLoad should not be meta");
    }

    // ── Test 75: VmProgram empty passes all validation checks ──

    #[test]
    fn vm_program_empty_passes_all_validations() {
        // Arrange
        let prog = VmProgram::new();

        // Act & Assert
        assert!(prog.validate_structure().is_ok(), "empty program should pass structure check");
        assert!(prog.validate_provenance().is_ok(), "empty program should pass provenance check");
        assert!(prog.validate_type_consistency().is_ok(), "empty program should pass type check");
        assert!(prog.validate_width_consistency().is_ok(), "empty program should pass width check");
        assert!(prog.validate_value_domains().is_ok(), "empty program should pass domain check");
        assert!(prog.validate_declares_before_uses().is_ok(), "empty program should pass declare check");
    }

    // ── Test 76: VmProgram emit_loop body instructions appear between LoopBegin and LoopEnd ──

    #[test]
    fn vm_program_emit_loop_body_instructions_between_begin_end() {
        // Arrange
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        prog.emit_loop(BoundExpr::Const(8), 32, |p, _ctr, _off| {
            p.emit(VmInstr::VecStore {
                base, src,
                offset: OffsetExpr::Const(0),
                width: SimdWidth::W256,
                dtype: QuantPrecision::F32, predicate: None,
            });
        });

        // Assert: VecStore appears between LoopBegin and LoopEnd
        let begin_idx = prog.instrs.iter().position(|i| matches!(i, VmInstr::LoopBegin { .. }))
            .expect("should have LoopBegin");
        let end_idx = prog.instrs.iter().rposition(|i| matches!(i, VmInstr::LoopEnd))
            .expect("should have LoopEnd");
        assert!(begin_idx < end_idx, "LoopBegin should come before LoopEnd");
        let has_store = prog.instrs[begin_idx + 1..end_idx].iter()
            .any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "VecStore should appear between LoopBegin and LoopEnd");
    }

    // ── Test 77: OffsetExpr::Const is unchanged by substitute_loop_offset ──

    #[test]
    fn offset_expr_const_substitute_loop_offset_is_noop() {
        // Arrange
        let expr = OffsetExpr::Const(42);
        let target = VRegId(0);

        // Act
        let result = expr.substitute_loop_offset(target, 100);

        // Assert
        assert_eq!(result, OffsetExpr::Const(42),
            "Const should be unchanged by substitute_loop_offset");
    }

    // ── Test 78: BoundExpr::Const Debug format contains the value ──

    #[test]
    fn bound_expr_const_debug_contains_value() {
        // Arrange
        let expr = BoundExpr::Const(12345);

        // Act
        let debug = format!("{:?}", expr);

        // Assert
        assert!(debug.contains("12345"),
            "Debug should contain the const value, got: {}", debug);
    }

    // ── Test 79: VmProgram alloc_vreg preserves width information ──

    #[test]
    fn vm_program_alloc_vreg_preserves_width_information() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act
        let v_scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let v_w128 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W128);
        let v_w256 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v_w512 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W512);

        // Assert: each DeclareVReg matches the requested width
        let find_width = |id: VRegId| -> Option<SimdWidth> {
            prog.instrs.iter().find_map(|i| match i {
                VmInstr::DeclareVReg { id: did, width, .. } if *did == id => Some(*width),
                _ => None,
            })
        };
        assert_eq!(find_width(v_scalar), Some(SimdWidth::Scalar));
        assert_eq!(find_width(v_w128), Some(SimdWidth::W128));
        assert_eq!(find_width(v_w256), Some(SimdWidth::W256));
        assert_eq!(find_width(v_w512), Some(SimdWidth::W512));
    }

    // ── Test 80: VmInstr compute instructions are not meta ──

    #[test]
    fn vm_instr_compute_instructions_not_meta() {
        // Arrange
        let compute_instrs: Vec<VmInstr> = vec![
            VmInstr::VecBinOp {
                dst: VRegId(0), a: VRegId(1), b: VRegId(2),
                op: VecOp::Add, dtype: QuantPrecision::F32,
            },
            VmInstr::VecLoad {
                dst: VRegId(0), base: VRegId(1),
                offset: OffsetExpr::Const(0), width: SimdWidth::W256,
                dtype: QuantPrecision::F32, predicate: None,
            },
            VmInstr::Fma {
                dst: VRegId(0), acc: VRegId(1), a: VRegId(2), b: VRegId(3),
                dtype: QuantPrecision::F32,
            },
        ];

        // Act & Assert
        for instr in &compute_instrs {
            assert!(!instr.is_meta(), "{:?} should not be meta", instr);
        }
    }

    // ── Test 81: GprCondition IsNull vs IsNonNull are distinct for same vreg ──

    #[test]
    fn gpr_condition_is_null_and_is_non_null_share_vreg() {
        // Arrange
        let v = VRegId(5);
        let is_null = GprCondition::IsNull(v);
        let is_non_null = GprCondition::IsNonNull(v);

        // Act & Assert: both reference the same vreg
        assert_eq!(is_null.vregs(), vec![v]);
        assert_eq!(is_non_null.vregs(), vec![v]);
    }

    // ── Test 82: OffsetExpr::Mul with non-matching target unchanged by substitute ──

    #[test]
    fn offset_expr_mul_non_matching_target_unchanged() {
        // Arrange
        let target = VRegId(1);
        let other = VRegId(2);
        let expr = OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(other)), 8);

        // Act: substitute target (which doesn't match other)
        let result = expr.substitute_loop_offset(target, 500);

        // Assert
        match &result {
            OffsetExpr::Mul(inner, scale) => {
                assert_eq!(&**inner, &OffsetExpr::LoopOffset(other),
                    "non-matching LoopOffset should be unchanged");
                assert_eq!(*scale, 8, "scale should be unchanged");
            }
            other_variant => panic!("expected Mul, got {:?}", other_variant),
        }
    }

    // ── Test 83: VmProgram emit_loop with Symbolic bound passes structure validation ──

    #[test]
    fn vm_program_emit_loop_symbolic_bound_passes_validation() {
        // Arrange
        let mut prog = VmProgram::new();
        let sym_bound = BoundExpr::Symbolic(SymBound { name: "tokens".into(), max_alloc: 2048 });

        // Act
        prog.emit_loop(sym_bound, 16, |p, _ctr, _off| {
            let v = prog_alloc_vec(p);
            p.emit(VmInstr::VecStore {
                base: v, src: v, offset: OffsetExpr::Const(0),
                width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
            });
        });

        // Assert
        assert!(prog.validate_structure().is_ok(),
            "loop with Symbolic bound should pass structure validation");
    }

    // ── Test 84: VmProgram sequential loops pass structure validation ──

    #[test]
    fn vm_program_sequential_loops_pass_structure_validation() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act: two independent sequential loops
        prog.emit_loop(BoundExpr::Const(4), 32, |p, _ctr, _off| {
            let v = prog_alloc_vec(p);
            p.emit(VmInstr::VecLoad {
                dst: v, base: v, offset: OffsetExpr::Const(0),
                width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
            });
        });
        prog.emit_loop(BoundExpr::Const(8), 64, |p, _ctr, _off| {
            let v = prog_alloc_vec(p);
            p.emit(VmInstr::VecStore {
                base: v, src: v, offset: OffsetExpr::Const(0),
                width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
            });
        });

        // Assert
        assert!(prog.validate_structure().is_ok(),
            "two sequential loops should pass structure validation");
        let begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        let ends = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopEnd)).count();
        assert_eq!(begins, 2, "should have 2 LoopBegin");
        assert_eq!(ends, 2, "should have 2 LoopEnd");
    }

    // ── Test 85: GprCondAction emission with IsNull condition (SgDetect callback skip pattern) ──

    #[test]
    fn gpr_cond_action_is_null_skip_pattern() {
        // Arrange: SgDetect uses GprCondAction(IsNull, Skip) to skip callback
        // when callback_table_ptr is NULL (zero overhead when SG disabled).
        let mut prog = VmProgram::new();
        let cb_table = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: emit the same pattern as SgDetect callback skip
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::IsNull(cb_table),
            action: GprBranchAction::Skip(4),
        });

        // Assert: instruction present and fields correct
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::GprCondAction { .. }));
        assert!(instr.is_some(), "should contain GprCondAction");
        if let Some(VmInstr::GprCondAction { cond, action }) = instr {
            assert_eq!(cond.vregs(), vec![cb_table], "condition should reference cb_table vreg");
            match action {
                GprBranchAction::Skip(n) => assert_eq!(*n, 4, "should skip 4 instructions"),
                other => panic!("expected Skip action, got {:?}", other),
            }
        }
    }

    // ── Test 86: GprCondAction emission with CmpEq condition (EarlyExit pattern) ──

    #[test]
    fn gpr_cond_action_cmp_eq_exit_pattern() {
        // Arrange: EarlyExit uses GprCondAction(CmpEq, Exit) to exit mega-kernel
        // when layer_loop_counter equals the anchor_layer.
        let mut prog = VmProgram::new();
        let layer_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let anchor_layer: u64 = 12;

        // Act: emit the same pattern as dispatch_structural EarlyExit
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(layer_ctr, anchor_layer),
            action: GprBranchAction::Exit(output_ptr),
        });

        // Assert: instruction present with correct fields
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::GprCondAction { .. }));
        assert!(instr.is_some(), "should contain GprCondAction");
        if let Some(VmInstr::GprCondAction { cond, action }) = instr {
            match cond {
                GprCondition::CmpEq(v, val) => {
                    assert_eq!(*v, layer_ctr, "condition should reference layer counter");
                    assert_eq!(*val, anchor_layer, "condition should compare against anchor_layer");
                }
                other => panic!("expected CmpEq condition, got {:?}", other),
            }
            match action {
                GprBranchAction::Exit(v) => assert_eq!(*v, output_ptr, "exit should use output_ptr"),
                other => panic!("expected Exit action, got {:?}", other),
            }
        }
    }

    // ── Test 87: IndirectJump emission with multiple targets (MegaKernelDispatch pattern) ──

    #[test]
    fn indirect_jump_multi_target_dispatch_pattern() {
        // Arrange: MegaKernelDispatch uses IndirectJump to route between
        // prefill/decode/chunked based on batch_size.
        let mut prog = VmProgram::new();
        let mode_reg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let targets = vec![
            JumpTarget { expert_id: 0, instr_index: 0 },
            JumpTarget { expert_id: 1, instr_index: 0 },
            JumpTarget { expert_id: 2, instr_index: 0 },
        ];

        // Act
        prog.emit(VmInstr::IndirectJump {
            index: mode_reg,
            targets: targets.clone(),
        });

        // Assert
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::IndirectJump { .. }));
        assert!(instr.is_some(), "should contain IndirectJump");
        if let Some(VmInstr::IndirectJump { index, targets: t }) = instr {
            assert_eq!(*index, mode_reg, "index should be mode_reg");
            assert_eq!(t.len(), 3, "should have 3 dispatch targets");
            assert_eq!(t[0].expert_id, 0);
            assert_eq!(t[1].expert_id, 1);
            assert_eq!(t[2].expert_id, 2);
        }
    }

    // ── Test 88: Argmax instruction type consistency validation ──

    #[test]
    fn argmax_instruction_type_consistency() {
        // Arrange: Argmax requires logits_ptr as GPR class, dst as Scalar
        let mut prog = VmProgram::new();
        let logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        prog.emit(VmInstr::Argmax {
            dst, logits_ptr, vocab_bytes: 32000 * 4, width: SimdWidth::W256,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert: well-typed Argmax should pass
        assert!(result.is_ok(), "Argmax with correct types should pass, got: {:?}", result);
    }

    // ── Test 89: Argmax rejects Vec-typed logits_ptr ──

    #[test]
    fn argmax_rejects_vec_typed_logits_ptr() {
        // Arrange: logits_ptr declared as Vec (not GPR class) — invalid for memory base
        let mut prog = VmProgram::new();
        let bad_logits_ptr = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        prog.emit(VmInstr::Argmax {
            dst, logits_ptr: bad_logits_ptr, vocab_bytes: 256, width: SimdWidth::W256,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert
        assert!(result.is_err(), "Argmax with Vec logits_ptr should fail type check");
        let err = result.unwrap_err();
        assert!(err.contains("Argmax") && err.contains("logits_ptr"),
                "error should mention Argmax and logits_ptr, got: {}", err);
    }

    // ── Test 90: StoreToken instruction type consistency validation ──

    #[test]
    fn store_token_instruction_type_consistency() {
        // Arrange: StoreToken requires token_id(Scalar), output_buf(Ptr),
        // counter(Counter), input_ids_ptr(Ptr), prompt_len_bytes(Ptr)
        let mut prog = VmProgram::new();
        let token_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let output_buf = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let input_ids_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let prompt_len_bytes = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::StoreToken {
            token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert: well-typed StoreToken should pass
        assert!(result.is_ok(), "StoreToken with correct types should pass, got: {:?}", result);
    }

    // ── Test 91: CheckStopCondition instruction type consistency validation ──

    #[test]
    fn check_stop_condition_instruction_type_consistency() {
        // Arrange: CheckStopCondition requires token_id(Scalar), counter(Counter),
        // eos_ptr(Ptr), max_tokens_ptr(Ptr)
        let mut prog = VmProgram::new();
        let token_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let eos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let max_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::CheckStopCondition {
            token_id, counter, eos_ptr, max_tokens_ptr,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert: well-typed CheckStopCondition should pass
        assert!(result.is_ok(), "CheckStopCondition with correct types should pass, got: {:?}", result);
    }

    // ── Test 92: CheckStopCondition rejects Vec-typed eos_ptr ──

    #[test]
    fn check_stop_condition_rejects_vec_eos_ptr() {
        // Arrange: eos_ptr declared as Vec (not GPR class) — invalid for memory address
        let mut prog = VmProgram::new();
        let token_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let bad_eos_ptr = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let max_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::CheckStopCondition {
            token_id, counter, eos_ptr: bad_eos_ptr, max_tokens_ptr,
        });

        // Act
        let result = prog.validate_type_consistency();

        // Assert
        assert!(result.is_err(), "CheckStopCondition with Vec eos_ptr should fail type check");
        let err = result.unwrap_err();
        assert!(err.contains("eos_ptr"),
                "error should mention eos_ptr, got: {}", err);
    }

    // ── Test 93: ConditionalSkip emission and provenance validation ──

    #[test]
    fn conditional_skip_emission_and_provenance() {
        // Arrange: ConditionalSkip is used by gate-first dead neuron mask dispatch.
        // The mask VReg must be declared and of Vec kind.
        let mut prog = VmProgram::new();
        let mask = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act: emit a gate-skip pattern (ConditionalSkip + some body + more)
        prog.emit(VmInstr::ConditionalSkip { mask, skip_count: 2 });
        // Simulated body that would be skipped (2 instructions)
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecBinOp { dst: v, a: v, b: v, op: VecOp::Add, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::VecBinOp { dst: v, a: v, b: v, op: VecOp::Mul, dtype: QuantPrecision::F32 });

        // Assert: provenance check passes
        let result = prog.validate_provenance();
        assert!(result.is_ok(), "ConditionalSkip with declared mask should pass provenance, got: {:?}", result);

        // Verify the ConditionalSkip instruction is present with correct skip_count
        let skip_instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::ConditionalSkip { .. }));
        assert!(skip_instr.is_some(), "should contain ConditionalSkip");
        if let Some(VmInstr::ConditionalSkip { mask: m, skip_count }) = skip_instr {
            assert_eq!(*m, mask, "mask should be the declared Vec vreg");
            assert_eq!(*skip_count, 2, "should skip 2 instructions");
        }
    }

    // ── Test 94: GprCondAction with IsNull preserves vreg references through remap ──

    #[test]
    fn gpr_cond_action_is_null_remap_presures_vreg_tracking() {
        // Arrange: verify that GprCondAction correctly tracks VReg references
        // through GprCondition and GprBranchAction for append/remap operations.
        let cb_table = VRegId(5);
        let output = VRegId(10);

        let cond = GprCondition::IsNull(cb_table);
        let action = GprBranchAction::Exit(output);

        // Act: remap (simulating VmProgram append vreg remapping)
        let remapped_cond = cond.remap(|v| VRegId(v.0 + 100));
        let remapped_action = action.clone().remap(|v| VRegId(v.0 + 100));

        // Assert: remapping propagates to inner VRegIds
        assert_eq!(remapped_cond.vregs(), vec![VRegId(105)],
                   "IsNull condition vreg should be remapped");
        match remapped_action {
            GprBranchAction::Exit(v) => assert_eq!(v, VRegId(110),
                "Exit action vreg should be remapped"),
            other => panic!("expected Exit, got {:?}", other),
        }

        // Original Skip action has no vregs to remap
        let skip_action = GprBranchAction::Skip(3);
        assert_eq!(skip_action.vregs(), Vec::<VRegId>::new(),
                   "Skip action has no vreg references");
        let remapped_skip = skip_action.clone().remap(|v| VRegId(v.0 + 100));
        match remapped_skip {
            GprBranchAction::Skip(n) => assert_eq!(n, 3, "Skip count unchanged by remap"),
            other => panic!("expected Skip, got {:?}", other),
        }
    }

    // ── Test 95: HotpatchSlot allocation and tracking in VmProgram ──

    #[test]
    fn hotpatch_slot_allocation_and_tracking() {
        // Arrange: HotpatchSlot reserves a 5-byte NOP placeholder at a branch point.
        // slot_id identifies the slot, initial_target is the default jump target,
        // alternatives are additional targets the slot can be patched to at runtime.
        let mut prog = VmProgram::new();

        // Act: emit HotpatchSlot with slot 0 (first slot)
        prog.emit(VmInstr::HotpatchSlot {
            slot_id: 0,
            initial_target: HotpatchTarget::InstrIndex(42),
            alternatives: vec![],
        });
        // Emit another hotpatch slot with slot 1 and alternatives
        prog.emit(VmInstr::HotpatchSlot {
            slot_id: 1,
            initial_target: HotpatchTarget::InstrIndex(100),
            alternatives: vec![HotpatchTarget::ExternalAddr(0x1000)],
        });

        // Assert: both HotpatchSlot instructions present with correct slot IDs
        let hotpatches: Vec<&VmInstr> = prog.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::HotpatchSlot { .. } => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(hotpatches.len(), 2, "should have 2 HotpatchSlot instructions");

        if let VmInstr::HotpatchSlot { slot_id, initial_target, alternatives } = hotpatches[0] {
            assert_eq!(*slot_id, 0, "first hotpatch should use slot 0");
            match initial_target {
                HotpatchTarget::InstrIndex(idx) => assert_eq!(*idx, 42),
                other => panic!("expected InstrIndex, got {:?}", other),
            }
            assert!(alternatives.is_empty(), "first slot should have no alternatives");
        }
        if let VmInstr::HotpatchSlot { slot_id, initial_target, alternatives } = hotpatches[1] {
            assert_eq!(*slot_id, 1, "second hotpatch should use slot 1");
            match initial_target {
                HotpatchTarget::InstrIndex(idx) => assert_eq!(*idx, 100),
                other => panic!("expected InstrIndex, got {:?}", other),
            }
            assert_eq!(alternatives.len(), 1, "second slot should have 1 alternative");
            match &alternatives[0] {
                HotpatchTarget::ExternalAddr(addr) => assert_eq!(*addr, 0x1000),
                other => panic!("expected ExternalAddr, got {:?}", other),
            }
        }
    }

    // ── Test 96: HotpatchTarget InstrIndex vs ExternalAddr field access ──

    #[test]
    fn hotpatch_target_variants_field_access_and_clone() {
        // Arrange
        let instr_target = HotpatchTarget::InstrIndex(100);
        let addr_target = HotpatchTarget::ExternalAddr(0xDEAD_BEEF);

        // Act
        let cloned_instr = instr_target.clone();
        let cloned_addr = addr_target.clone();

        // Assert: verify field access through pattern matching
        match instr_target {
            HotpatchTarget::InstrIndex(idx) => assert_eq!(idx, 100),
            other => panic!("expected InstrIndex, got {:?}", other),
        }
        match addr_target {
            HotpatchTarget::ExternalAddr(addr) => assert_eq!(addr, 0xDEAD_BEEF),
            other => panic!("expected ExternalAddr, got {:?}", other),
        }
        // Cloned variants have same field values
        match cloned_instr {
            HotpatchTarget::InstrIndex(idx) => assert_eq!(idx, 100),
            other => panic!("expected InstrIndex, got {:?}", other),
        }
        match cloned_addr {
            HotpatchTarget::ExternalAddr(addr) => assert_eq!(addr, 0xDEAD_BEEF),
            other => panic!("expected ExternalAddr, got {:?}", other),
        }
    }

    // ── Test 97: IndirectJump with single target (minimal dispatch table) ──

    #[test]
    fn indirect_jump_single_target_dispatch_table() {
        // Arrange: minimal dispatch table with just one target
        let mut prog = VmProgram::new();
        let index_reg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let single_target = vec![JumpTarget { expert_id: 0, instr_index: 256 }];

        // Act
        prog.emit(VmInstr::IndirectJump {
            index: index_reg,
            targets: single_target.clone(),
        });

        // Assert
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::IndirectJump { .. }));
        assert!(instr.is_some(), "should contain IndirectJump");
        if let Some(VmInstr::IndirectJump { index, targets }) = instr {
            assert_eq!(*index, index_reg);
            assert_eq!(targets.len(), 1, "should have exactly 1 target");
            assert_eq!(targets[0].instr_index, 256);
        }
    }

    // ── Test 98: JumpTarget expert_id and instr_index are independent fields ──

    #[test]
    fn jump_target_fields_independent() {
        // Arrange: JumpTarget is used in IndirectJump dispatch tables.
        // expert_id identifies the dispatch target (e.g., MoE expert, kernel variant).
        // instr_index is the instruction offset within that target's code.
        let t1 = JumpTarget { expert_id: 0, instr_index: 100 };
        let t2 = JumpTarget { expert_id: 0, instr_index: 200 }; // same expert, different offset
        let t3 = JumpTarget { expert_id: 1, instr_index: 100 }; // different expert, same offset

        // Act & Assert: verify field independence through direct field access
        assert_eq!(t1.expert_id, t2.expert_id, "same expert_id");
        assert_ne!(t1.instr_index, t2.instr_index, "different instr_index");
        assert_ne!(t1.expert_id, t3.expert_id, "different expert_id");
        assert_eq!(t1.instr_index, t3.instr_index, "same instr_index");

        // Clone preserves fields
        let cloned = t2.clone();
        assert_eq!(cloned.expert_id, 0);
        assert_eq!(cloned.instr_index, 200);
    }

    // ── Test 99: GprCondAction with BitClear condition (GuardrailCheck pattern) ──

    #[test]
    fn gpr_cond_action_bit_clear_guardrail_pattern() {
        // Arrange: GuardrailCheck uses BitClear to test flag bits in a status word.
        // If the bit is clear (flag not set), skip the guardrail probe.
        let mut prog = VmProgram::new();
        let status_word = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let bit_pos = 3u8;

        // Act: emit BitClear condition with Skip action
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::BitClear(status_word, bit_pos),
            action: GprBranchAction::Skip(2),
        });

        // Assert
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::GprCondAction { .. }));
        assert!(instr.is_some());
        if let Some(VmInstr::GprCondAction { cond, action }) = instr {
            match cond {
                GprCondition::BitClear(v, b) => {
                    assert_eq!(*v, status_word);
                    assert_eq!(*b, bit_pos);
                }
                other => panic!("expected BitClear, got {:?}", other),
            }
            match action {
                GprBranchAction::Skip(n) => assert_eq!(*n, 2),
                other => panic!("expected Skip, got {:?}", other),
            }
        }
    }

    // ── Test 100: GprCondAction with BitSet condition (CotStepCheck pattern) ──

    #[test]
    fn gpr_cond_action_bit_set_cot_step_pattern() {
        // Arrange: CotStepCheck uses BitSet to test if CoT step flag is set.
        // If the bit is set (flag active), execute the step check logic.
        let mut prog = VmProgram::new();
        let flags_reg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let cot_flag_bit = 5u8;

        // Act: emit BitSet condition with Skip action (skip if flag NOT set)
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::BitSet(flags_reg, cot_flag_bit),
            action: GprBranchAction::Skip(3),
        });

        // Assert
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::GprCondAction { .. }));
        assert!(instr.is_some());
        if let Some(VmInstr::GprCondAction { cond, action }) = instr {
            match cond {
                GprCondition::BitSet(v, b) => {
                    assert_eq!(*v, flags_reg);
                    assert_eq!(*b, cot_flag_bit);
                }
                other => panic!("expected BitSet, got {:?}", other),
            }
            match action {
                GprBranchAction::Skip(n) => assert_eq!(*n, 3),
                other => panic!("expected Skip, got {:?}", other),
            }
        }
    }

    // ── Test 101: GprCondAction with CmpLtU condition (bounds check pattern) ──

    #[test]
    fn gpr_cond_action_cmp_lt_u_bounds_check_pattern() {
        // Arrange: CmpLtU (unsigned less-than) is used for bounds checking.
        // E.g., skip if index < max_valid_index.
        let mut prog = VmProgram::new();
        let index_reg = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let max_valid: u64 = 1024;

        // Act: emit CmpLtU with Exit action (exit early if out of bounds)
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpLtU(index_reg, max_valid),
            action: GprBranchAction::Exit(output_ptr),
        });

        // Assert
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::GprCondAction { .. }));
        assert!(instr.is_some());
        if let Some(VmInstr::GprCondAction { cond, action }) = instr {
            match cond {
                GprCondition::CmpLtU(v, val) => {
                    assert_eq!(*v, index_reg);
                    assert_eq!(*val, max_valid);
                }
                other => panic!("expected CmpLtU, got {:?}", other),
            }
            match action {
                GprBranchAction::Exit(v) => assert_eq!(*v, output_ptr),
                other => panic!("expected Exit, got {:?}", other),
            }
        }
    }

    // ── Test 102: ConditionalSkip emission with declared mask passes provenance ──

    #[test]
    fn conditional_skip_declared_mask_passes_provenance() {
        // Arrange: emit ConditionalSkip with a properly declared mask vreg
        let mut prog = VmProgram::new();
        let mask = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        prog.emit(VmInstr::ConditionalSkip {
            mask,
            skip_count: 2,
        });

        // Assert: provenance should pass since mask is declared
        let result = prog.validate_provenance();
        assert!(result.is_ok(),
                "ConditionalSkip with declared mask should pass provenance, got: {:?}", result);

        // Verify instruction was emitted correctly
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::ConditionalSkip { .. }));
        assert!(instr.is_some());
        if let Some(VmInstr::ConditionalSkip { mask: m, skip_count }) = instr {
            assert_eq!(*m, mask);
            assert_eq!(*skip_count, 2);
        }
    }

    // ── Test 103: IndirectJump provenance validates index register ──

    #[test]
    fn indirect_jump_provenance_validates_index_register() {
        // Arrange: IndirectJump index must be a declared GPR-class vreg
        let mut prog = VmProgram::new();
        let valid_index = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let targets = vec![
            JumpTarget { expert_id: 0, instr_index: 0 },
            JumpTarget { expert_id: 1, instr_index: 64 },
        ];

        // Act
        prog.emit(VmInstr::IndirectJump {
            index: valid_index,
            targets,
        });

        // Assert: provenance passes with declared index
        let result = prog.validate_provenance();
        assert!(result.is_ok(),
                "IndirectJump with declared index should pass provenance, got: {:?}", result);
    }

    // ── Test 104: IndirectJump with multi-expert dispatch table preserves order ──

    #[test]
    fn indirect_jump_multi_expert_dispatch_table_preserves_order() {
        // Arrange: IndirectJump dispatch table maps expert_id to instr_index.
        // The order of targets in the vector determines the dispatch index.
        let mut prog = VmProgram::new();
        let index_reg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let targets = vec![
            JumpTarget { expert_id: 0, instr_index: 10 },
            JumpTarget { expert_id: 1, instr_index: 20 },
            JumpTarget { expert_id: 2, instr_index: 30 },
            JumpTarget { expert_id: 3, instr_index: 40 },
        ];

        // Act
        prog.emit(VmInstr::IndirectJump {
            index: index_reg,
            targets: targets.clone(),
        });

        // Assert: targets preserved in insertion order
        let instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::IndirectJump { .. }));
        assert!(instr.is_some());
        if let Some(VmInstr::IndirectJump { index, targets: t }) = instr {
            assert_eq!(*index, index_reg);
            assert_eq!(t.len(), 4, "should have 4 targets");
            // Verify order is preserved
            for (i, target) in t.iter().enumerate() {
                assert_eq!(target.expert_id, i,
                           "expert_id at position {} should be {}", i, i);
                assert_eq!(target.instr_index, (i + 1) * 10,
                           "instr_index at position {} should be {}", i, (i + 1) * 10);
            }
        }
    }
}
