//! Resource estimator for fusion budget gating (SPEC 15 REQ-JCTX-014).
//!
//! Provides lightweight resource estimation functions that the fusion pass
//! uses to check JitContext budget before committing to a fusion decision.
//! Budget不足 → 拒绝融合（降级为原子 op），而非拒绝编译。
//! jit_ctx=None → 使用 DeviceProfile 保守估计（不阻止融合）。

use crate::compiler::graph::{CompilerGraph, CompilerOp, OpId, OpKind};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass};
use crate::compiler::jit_context::{JitContext, ResourceKind};

/// Register pressure estimate per op class (in SimdVec registers).
///
/// Simplified model based on typical register usage patterns:
/// - GEMM: accumulator tiles + load buffers = 8 SIMD regs
/// - Reduction: accumulator + source = 4 SIMD regs
/// - Norm: mean + variance + output = 3 SIMD regs
/// - Elementwise: source + destination = 2 SIMD regs
/// - Opaque: 1 SIMD reg (metadata-only ops like Reshape/Transpose)
fn op_class_register_pressure(op_class: OpClass) -> usize {
    match op_class {
        OpClass::Gemm => 8,
        OpClass::Reduction => 4,
        OpClass::ElemWise | OpClass::Injective => 2,
        OpClass::Opaque => 1,
    }
}

/// Estimate register pressure for fusing two ops.
///
/// Returns the *additional* SimdVec register pressure beyond what each
/// op would use independently. When fusing, the intermediate tensor
/// is kept in a register instead of written to memory, so the extra
/// cost is approximately 1 register for the intermediate value.
pub fn estimate_register_pressure(op_a: &CompilerOp, op_b: &CompilerOp, dag: Option<&SemanticDAG>) -> usize {
    let class_a = dag.and_then(|d| d.node(op_a.id)).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
    let class_b = dag.and_then(|d| d.node(op_b.id)).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
    // Fused: the intermediate output stays in register instead of being
    // written to memory and re-loaded. Extra cost = 1 register for the
    // intermediate, plus any register pressure from the second op that
    // overlaps with the first's live range.
    op_class_register_pressure(class_a) + op_class_register_pressure(class_b) + 1
}

/// Estimate tile resource demand (in bytes) for TileLevelFusion.
///
/// Tile size = rows × cols × elem_bytes. This must fit within L1 cache
/// (or GPU SharedMem) for tile-level fusion to be profitable.
pub fn estimate_tile_resource_bytes(
    graph: &CompilerGraph,
    anchor: OpId,
    predecessor: OpId,
) -> usize {
    // Use the predecessor's output tensor size as the tile resource demand.
    // If we can't determine it, use a conservative estimate based on the
    // anchor GEMM's dimensions.
    // Try to get output tensor byte count from the predecessor
    if let Some(pred_op) = graph.op(predecessor) {
        if let Some(&out_tid) = pred_op.outputs.first() {
            if let Some(tensor) = graph.tensor(out_tid) {
                let elem_bytes = tensor.dtype.size_bytes();
                let total_elems: usize = tensor.shape.iter()
                    .map(|d| d.max_for_allocation(0))
                    .product();
                return total_elems * elem_bytes;
            }
        }
    }

    // Fallback: estimate from anchor GEMM dimensions
    if let Some(anchor_op) = graph.op(anchor) {
        match &anchor_op.kind {
            OpKind::Gemm { m, n, .. } | OpKind::GemmBias { m, n, .. } | OpKind::QuantGemm { m, n, .. } => {
                let m_val = m.max_for_allocation(0);
                let tile_rows = m_val.min(64);
                return tile_rows * n * 4;
            }
            _ => {}
        }
    }

    0
}

/// Check whether the JitContext budget allows fusing a candidate op
/// into an existing fusion group.
///
/// Returns `true` if fusion is allowed, `false` if budget would be exceeded.
/// When `jit_ctx` is `None`, always returns `true` (no budget constraint).
pub fn can_fuse_with_budget(
    jit_ctx: Option<&JitContext>,
    graph: &CompilerGraph,
    anchor: &CompilerOp,
    candidate: &CompilerOp,
    dag: Option<&SemanticDAG>,
) -> bool {
    let ctx = match jit_ctx {
        Some(c) => c,
        None => return true, // No budget constraint when JitContext unavailable
    };

    let extra_pressure = estimate_register_pressure(anchor, candidate, dag);

    // Check SimdVec register budget
    let available = ctx.available(ResourceKind::SimdVec);
    if extra_pressure > available {
        return false;
    }

    // Check GPR budget (epilogue ops may need scratch GPRs)
    let gpr_needed = 2; // conservative: 1 for loop counter + 1 for address
    if gpr_needed > ctx.available(ResourceKind::Gpr) {
        return false;
    }

    true
}

/// Check whether extending a fusion group with additional ops would exceed
/// the cumulative resource budget.
///
/// `cumulative_regs` is the register pressure already consumed by the group.
/// Returns `true` if the extension is allowed, `false` if budget exceeded.
pub fn can_extend_group_with_budget(
    jit_ctx: Option<&JitContext>,
    cumulative_regs: usize,
    additional_ops: usize,
    dag: Option<&SemanticDAG>,
) -> bool {
    let ctx = match jit_ctx {
        Some(c) => c,
        None => return true,
    };

    // Each additional op consumes ~2 SIMD regs on average
    let extra_regs = additional_ops * 2;
    let total_needed = cumulative_regs + extra_regs;

    // Must not exceed SimdVec capacity
    let simd_capacity = ctx.capacity(ResourceKind::SimdVec);
    if total_needed > simd_capacity {
        return false;
    }

    // Must leave some headroom for ISA lowering infrastructure
    // (loop counters, address calculation, spill/reload temps)
    let min_headroom = 4;
    if total_needed + min_headroom > simd_capacity {
        return false;
    }

    true
}

/// Check whether an epilogue injection is feasible within L1/SharedMem budget.
///
/// The epilogue ops' intermediate data must fit in L1 cache (CPU) or
/// SharedMem (GPU) for the fused kernel to avoid cache thrashing.
pub fn can_inject_epilogue_with_budget(
    jit_ctx: Option<&JitContext>,
    epilogue_bytes: usize,
) -> bool {
    let ctx = match jit_ctx {
        Some(c) => c,
        None => return true,
    };

    // On CPU: check stack budget (L1-resident data lives in registers/stack)
    // On GPU: check SharedMem budget
    if ctx.capacity(ResourceKind::SharedMem) > 0 {
        // GPU path: epilogue data must fit in SharedMem
        let smem_available = ctx.mem_available(ResourceKind::SharedMem);
        if epilogue_bytes > smem_available {
            return false;
        }
    }

    // Register budget for epilogue chain
    // Each epilogue op needs ~2 SIMD regs
    let estimated_epilogue_ops = (epilogue_bytes / 64).max(1); // rough estimate
    let regs_needed = estimated_epilogue_ops * 2;
    if regs_needed > ctx.available(ResourceKind::SimdVec) {
        return false;
    }

    true
}

/// Check whether TileLevelFusion is feasible within L1/SharedMem budget.
///
/// Tile data must fit within the L1 cache line budget (CPU) or
/// SharedMem (GPU) for tile-level fusion to work correctly.
pub fn can_tile_fuse_with_budget(
    jit_ctx: Option<&JitContext>,
    graph: &CompilerGraph,
    anchor: OpId,
    predecessor: OpId,
) -> bool {
    let ctx = match jit_ctx {
        Some(c) => c,
        None => return true,
    };

    let tile_bytes = estimate_tile_resource_bytes(graph, anchor, predecessor);

    if ctx.capacity(ResourceKind::SharedMem) > 0 {
        // GPU: tile must fit in SharedMem
        if tile_bytes > ctx.mem_available(ResourceKind::SharedMem) {
            return false;
        }
    }

    // Register budget for tile: tile_rows × elem_width SIMD regs
    // (each row is loaded into one SIMD register)
    let regs_per_tile_row = 1; // 1 SIMD reg per row
    let tile_rows = estimate_tile_rows(graph, anchor);
    let total_tile_regs = tile_rows * regs_per_tile_row;
    if total_tile_regs > ctx.available(ResourceKind::SimdVec) {
        return false;
    }

    true
}

/// Estimate tile rows from the anchor GEMM's MC blocking factor.
fn estimate_tile_rows(graph: &CompilerGraph, anchor: OpId) -> usize {
    if let Some(op) = graph.op(anchor) {
        match &op.kind {
            OpKind::Gemm { m, .. } | OpKind::GemmBias { m, .. } | OpKind::QuantGemm { m, .. } => {
                // MC = min(m, 64) — typical L1 blocking factor
                return m.max_for_allocation(0).min(64);
            }
            _ => {}
        }
    }
    64 // default MC
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, OpId, OpKind, SymDim};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::semantic_dag::SemanticDAG;
    use crate::compiler::planner::ExecutionPlan;
    use crate::compiler::jit_context::JitContext;
    use crate::compiler::codegen::vm::isa_profile::IsaProfile;
    use crate::dispatch::DeviceProfile;
    use crate::types::DType;

    fn make_profile() -> IsaProfile {
        IsaProfile::from_device_profile(&DeviceProfile::detect())
    }

    fn make_plan() -> ExecutionPlan {
        ExecutionPlan::from_profile(&DeviceProfile::detect())
    }

    // ── estimate_register_pressure ──

    #[test]
    fn estimate_register_pressure_gemm_elemwise() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![mid], "gemm",
        );
        let silu = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let reg = ScalarOpRegistry::new();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let gemm_op = g.op(gemm).unwrap();
        let silu_op = g.op(silu).unwrap();
        let pressure = estimate_register_pressure(gemm_op, silu_op, Some(&dag));
        // GEMM=8 + ElemWise=2 + 1 intermediate = 11
        assert_eq!(pressure, 11);
    }

    #[test]
    fn estimate_register_pressure_elemwise_elemwise() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 64], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);
        let tanh = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let silu = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let reg = ScalarOpRegistry::new();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let tanh_op = g.op(tanh).unwrap();
        let silu_op = g.op(silu).unwrap();
        let pressure = estimate_register_pressure(tanh_op, silu_op, Some(&dag));
        // ElemWise=2 + ElemWise=2 + 1 intermediate = 5
        assert_eq!(pressure, 5);
    }

    #[test]
    fn estimate_register_pressure_without_dag_uses_opaque() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let tanh = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let silu = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let tanh_op = g.op(tanh).unwrap();
        let silu_op = g.op(silu).unwrap();
        let pressure = estimate_register_pressure(tanh_op, silu_op, None);
        // Opaque=1 + Opaque=1 + 1 intermediate = 3 (conservative)
        assert_eq!(pressure, 3);
    }

    // ── can_fuse_with_budget ──

    #[test]
    fn can_fuse_with_budget_no_ctx_always_allows() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let tanh = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let silu = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let tanh_op = g.op(tanh).unwrap();
        let silu_op = g.op(silu).unwrap();
        assert!(can_fuse_with_budget(None, &g, tanh_op, silu_op, None));
    }

    #[test]
    fn can_fuse_with_budget_fresh_ctx_allows() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let tanh = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let silu = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let profile = make_profile();
        let ctx = JitContext::new(&profile);
        let reg = ScalarOpRegistry::new();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let tanh_op = g.op(tanh).unwrap();
        let silu_op = g.op(silu).unwrap();
        // Fresh context has full budget available
        assert!(can_fuse_with_budget(Some(&ctx), &g, tanh_op, silu_op, Some(&dag)));
    }

    #[test]
    fn can_fuse_with_budget_exhausted_regs_rejects() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        // Exhaust all SimdVec registers
        let simd_cap = ctx.capacity(ResourceKind::SimdVec);
        let mut allocated = Vec::new();
        for _ in 0..simd_cap {
            allocated.push(ctx.allocate(ResourceKind::SimdVec, "exhaust").unwrap());
        }

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let tanh = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let silu = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let tanh_op = g.op(tanh).unwrap();
        let silu_op = g.op(silu).unwrap();
        // Budget exhausted — should reject fusion
        assert!(!can_fuse_with_budget(Some(&ctx), &g, tanh_op, silu_op, None));

        // Cleanup
        for idx in allocated {
            ctx.release(ResourceKind::SimdVec, idx);
        }
    }

    // ── can_extend_group_with_budget ──

    #[test]
    fn can_extend_group_no_ctx_always_allows() {
        assert!(can_extend_group_with_budget(None, 10, 5, None));
    }

    #[test]
    fn can_extend_group_fresh_ctx_allows() {
        let profile = make_profile();
        let ctx = JitContext::new(&profile);
        let simd_cap = ctx.capacity(ResourceKind::SimdVec);
        // Use values that fit within any SimdVec capacity (AVX2=16, AVX-512=32)
        // cumulative + additional*2 + headroom(4) <= simd_cap
        let cumulative = simd_cap.saturating_sub(10);
        let additional = 2; // extra = 4, total = cumulative + 4 + 4 headroom
        assert!(can_extend_group_with_budget(Some(&ctx), cumulative, additional, None));
    }

    #[test]
    fn can_extend_group_near_capacity_rejects() {
        let profile = make_profile();
        let ctx = JitContext::new(&profile);
        let simd_cap = ctx.capacity(ResourceKind::SimdVec);

        // Request nearly all capacity — headroom check should reject
        // (need 4 extra for headroom)
        let cumulative = simd_cap.saturating_sub(3);
        let additional = 2;
        // cumulative + additional*2 > simd_cap - headroom
        assert!(!can_extend_group_with_budget(Some(&ctx), cumulative, additional, None));
    }

    // ── can_inject_epilogue_with_budget ──

    #[test]
    fn can_inject_epilogue_no_ctx_always_allows() {
        assert!(can_inject_epilogue_with_budget(None, 4096));
    }

    #[test]
    fn can_inject_epilogue_fresh_ctx_allows() {
        let profile = make_profile();
        let ctx = JitContext::new(&profile);
        // Small epilogue should be fine
        assert!(can_inject_epilogue_with_budget(Some(&ctx), 256));
    }

    // ── can_tile_fuse_with_budget ──

    #[test]
    fn can_tile_fuse_no_ctx_always_allows() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![out], "gemm",
        );

        assert!(can_tile_fuse_with_budget(None, &g, gemm, OpId(999)));
    }

    #[test]
    fn can_tile_fuse_fresh_ctx_allows() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let norm = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![norm_out], "norm");
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![norm_out, w], vec![out], "gemm",
        );

        let profile = make_profile();
        let ctx = JitContext::new(&profile);
        // Small tile should be fine
        assert!(can_tile_fuse_with_budget(Some(&ctx), &g, gemm, norm));
    }

    // ── estimate_tile_resource_bytes ──

    #[test]
    fn estimate_tile_resource_returns_nonzero() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let norm = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![norm_out], "norm");
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![norm_out, w], vec![out], "gemm",
        );

        let bytes = estimate_tile_resource_bytes(&g, gemm, norm);
        // norm output: 1 × 16 × 4 bytes = 64
        assert_eq!(bytes, 64);
    }

    #[test]
    fn estimate_tile_resource_nonexistent_predecessor() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![out], "gemm",
        );

        let bytes = estimate_tile_resource_bytes(&g, gemm, OpId(999));
        // Nonexistent predecessor → fallback to GEMM dimension estimate
        assert!(bytes > 0);
    }
}
