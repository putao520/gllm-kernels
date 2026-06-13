//! Fusion cost model — roofline-based benefit estimation.

use std::collections::HashSet;
use crate::compiler::graph::{CompilerGraph, CompilerOp, OpKind, OpId};
use crate::compiler::trace::{OpTrace, TraceOp, ComputePattern, ScalarParam};
use crate::types::DType;
use crate::compiler::pain_point::OpBottleneckMap;
use crate::compiler::hardware_profile::HardwareProfile;
use super::types::{FusionGroup, FusionMode, FusionCost};

/// Compute roofline scaling factor for a fusion group's benefit.
///
/// When OpBottleneckMap is available (R0), uses the anchor GEMM's precise
/// bottleneck analysis. Otherwise falls back to group-level AI computation.
pub(crate) fn compute_group_roofline_scale(
    group: &FusionGroup,
    graph: &CompilerGraph,
    plan: &crate::compiler::planner::ExecutionPlan,
    bottleneck_map: Option<&OpBottleneckMap>,
) -> f64 {
    // R0 快速路径：anchor GEMM 有精确瓶颈分析
    if let Some(map) = bottleneck_map {
        if let Some(bn) = map.gemm_bottlenecks.get(&group.anchor) {
            return match bn.bottleneck {
                crate::compiler::pain_point::BottleneckType::MemoryBound { .. } => 1.0,
                crate::compiler::pain_point::BottleneckType::ComputeBound { compute_utilization } => {
                    compute_utilization.max(0.1)
                }
                crate::compiler::pain_point::BottleneckType::LatencyBound { .. } => 0.5,
            };
        }
    }

    // 回退：通用 group-level AI 计算
    let ridge = plan.roofline.ridge_point;
    if ridge <= 0.0 {
        return 1.0;
    }
    let ai = compute_group_ai(group, graph);
    if ai <= ridge as f32 {
        1.0
    } else {
        (ridge / ai as f64).clamp(0.1, 1.0)
    }
}

/// Compute arithmetic intensity (FLOP/byte) for a fusion group.
pub(crate) fn compute_group_ai(group: &FusionGroup, graph: &CompilerGraph) -> f32 {
    let mut total_flops = 0usize;
    let mut total_bytes = 0usize;
    for &op_id in &group.ops {
        if let Some(op) = graph.op(op_id) {
            total_flops += estimate_op_flops(op);
            for &tid in op.inputs.iter().chain(op.outputs.iter()) {
                if let Some(t) = graph.tensor(tid) {
                    total_bytes += t.concrete_bytes();
                }
            }
        }
    }
    if total_bytes == 0 { return 0.0; }
    total_flops as f32 / total_bytes as f32
}

/// Estimate FLOPs for a single CompilerOp based on its OpKind.
/// For GEMM variants, returns 2*M*N*K. For other ops, returns 0
/// (their FLOP contribution is negligible for roofline classification).
// ARCH-SYMDIM-DEGRADE: cost model uses max_for_allocation for conservative estimate.
// TODO(G-2): preserve symbolic form for tighter bounds.
fn estimate_op_flops(op: &CompilerOp) -> usize {
    match &op.kind {
        OpKind::Gemm { m, n, k, .. }
        | OpKind::GemmBias { m, n, k, .. }
        | OpKind::QuantGemm { m, n, k, .. } => 2 * m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model") * n * k,
        _ => 0,
    }
}

/// Returns true if the fusion group is memory-bound (AI < ridge point).
pub(crate) fn is_memory_bound_group(
    group: &FusionGroup,
    graph: &CompilerGraph,
    plan: &crate::compiler::planner::ExecutionPlan,
) -> bool {
    let ridge = plan.roofline.ridge_point;
    if ridge <= 0.0 { return true; }
    let ai = compute_group_ai(group, graph);
    (ai as f64) < ridge
}

/// Extract GEMM dtype from the anchor op of a fusion group.
/// ARCH-DTYPE-FULLCHAIN-ORCH: QuantGemm and non-GEMM anchors use graph-level dtype inference.
fn extract_anchor_dtype(group: &FusionGroup, graph: &CompilerGraph) -> crate::types::DType {
    graph.op(group.anchor)
        .and_then(|op| match &op.kind {
            OpKind::Gemm { dtype, .. } | OpKind::GemmBias { dtype, .. } => Some(*dtype),
            OpKind::QuantGemm { .. } => Some(graph.infer_computation_dtype()),
            _ => None,
        })
        .unwrap_or_else(|| graph.infer_computation_dtype())
}

/// Extract GEMM (m, n, k) from the anchor op of a fusion group.
// ARCH-SYMDIM-DEGRADE: cost model uses max_for_allocation for conservative estimate.
// TODO(G-2): preserve symbolic form for tighter bounds.
fn extract_anchor_gemm_dims(group: &FusionGroup, graph: &CompilerGraph) -> (usize, usize, usize) {
    graph.op(group.anchor)
        .and_then(|op| match &op.kind {
            OpKind::Gemm { m, n, k, .. }
            | OpKind::GemmBias { m, n, k, .. }
            | OpKind::QuantGemm { m, n, k, .. } => Some((m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model"), *n, *k)),
            _ => None,
        })
        .unwrap_or((0, 0, 0))
}

/// Estimate the cost/benefit of a fusion group.
///
/// The model is roofline-inspired:
/// - Benefit: bytes of intermediate tensors that no longer need to be written/read
///   from memory (2x the tensor size for write + read-back).
/// - Cost: register pressure increase may cause spills (each spill ~ 64 bytes
///   round-trip to stack), plus scratch buffer allocation overhead.
pub fn estimate_fusion_cost(
    group: &FusionGroup,
    graph: &CompilerGraph,
    plan: &crate::compiler::planner::ExecutionPlan,
    bottleneck_map: Option<&OpBottleneckMap>,
) -> FusionCost {
    let avail_regs = plan.profile.num_simd_regs();
    let (l1, _, _) = plan.profile.cache_sizes();

    // Bytes saved: sum of intermediate tensor sizes that are consumed only within the group
    let group_ops: HashSet<OpId> = group.ops.iter().copied().collect();
    let mut bytes_saved: usize = 0;

    for &op_id in &group.ops {
        let op = match graph.op(op_id) {
            Some(o) => o,
            None => continue,
        };
        for &out_tid in &op.outputs {
            let tensor = match graph.tensor(out_tid) {
                Some(t) => t,
                None => continue,
            };
            // If all consumers are within the group, this intermediate is eliminated
            let all_internal = tensor
                .consumers
                .iter()
                .all(|c| group_ops.contains(c));
            if all_internal && !tensor.consumers.is_empty() {
                // Write + read-back eliminated
                let size = tensor.concrete_bytes();
                bytes_saved += size * 2;
            }
        }
    }

    // Register pressure estimate
    let anchor_dtype = extract_anchor_dtype(group, graph);
    let anchor_dims = extract_anchor_gemm_dims(group, graph);
    let base_regs = match group.mode {
        FusionMode::EpilogueInjection => {
            // GEMM accumulators + epilogue temporaries (dynamic based on register file)
            let blocking = plan.profile.gemm_blocking(anchor_dims.0, anchor_dims.1, anchor_dims.2, anchor_dtype);
            let simd_w = plan.profile.simd_width_bytes() / anchor_dtype.size_bytes();
            let acc = (blocking.mr * blocking.nr) / simd_w.max(1);
            let avail = plan.profile.num_simd_regs();
            let free = avail.saturating_sub(acc + 2);
            let no_spill = group.epilogue.len().min(free);
            let spill_ops = group.epilogue.len().saturating_sub(free);
            acc + 2 + no_spill + spill_ops.min(1)
        }
        FusionMode::TileLevelFusion { .. } => {
            let blocking = plan.profile.gemm_blocking(anchor_dims.0, anchor_dims.1, anchor_dims.2, anchor_dtype);
            let simd_w = plan.profile.simd_width_bytes() / anchor_dtype.size_bytes();
            let acc = (blocking.mr * blocking.nr) / simd_w.max(1);
            acc + 3 // norm scratch: mean, rsqrt, weight
        }
        FusionMode::LoopFusion => {
            // 1 input + 1 output + 1 temp per fused op
            1 + group.ops.len().min(8)
        }
        _ => 0,
    };
    let extra_regs = base_regs.saturating_sub(avail_regs / 2);

    // Scratch buffer for TileLevelFusion
    let scratch_bytes = match group.mode {
        FusionMode::TileLevelFusion { tile_rows, .. } => {
            // Scratch = tile_rows x K x elem_bytes for the tiled norm output
            let k = group.ops.iter().find_map(|&oid| {
                graph.op(oid).and_then(|o| match &o.kind {
                    OpKind::Gemm { k, .. }
                    | OpKind::GemmBias { k, .. }
                    | OpKind::QuantGemm { k, .. } => Some(*k),
                    _ => None,
                })
            }).unwrap_or(0);
            tile_rows * k * anchor_dtype.size_bytes()
        }
        _ => 0,
    };

    // Penalty: spill cost + scratch overhead
    let spill_penalty = (extra_regs as i64) * 64 * 2; // 64B per spill, write+read
    let scratch_penalty = if scratch_bytes > l1 / 2 {
        scratch_bytes as i64 // heavy penalty if scratch exceeds half L1
    } else {
        0
    };

    // Roofline-aware scaling: memory-bound groups get full bytes_saved benefit,
    // compute-bound groups get scaled-down benefit (fusion saves less when
    // the bottleneck is compute, not memory).
    let roofline_scale = compute_group_roofline_scale(group, graph, plan, bottleneck_map);
    // Apply fusion_cost_scale: < 1.0 amplifies savings (fusion cheaper),
    // > 1.0 dampens savings (fusion more expensive). Formula: savings / scale.
    let fusion_scale = plan.strategy_bias.fusion_cost_scale;
    let adjusted_bytes = if fusion_scale > 0.0 {
        bytes_saved as f64 / fusion_scale
    } else {
        bytes_saved as f64
    };
    let benefit = (adjusted_bytes * roofline_scale) as i64 - spill_penalty - scratch_penalty;

    FusionCost {
        bytes_saved,
        extra_regs,
        scratch_bytes,
        benefit,
    }
}

// ── Roofline cost model ─────────────────────────────────────────────

/// Per-operator roofline cost estimate.
///
/// Classifies an operator as compute-bound or memory-bound by comparing
/// estimated compute cycles against memory cycles. Used by the fusion pass
/// to quantify the benefit of eliminating intermediate memory traffic.
#[derive(Debug, Clone)]
pub struct Cost {
    /// Total floating-point operations (per invocation, across all elements).
    pub flops: u64,
    /// Total memory traffic in bytes (all inputs + outputs).
    pub bytes: u64,
    /// Estimated time spent on compute (flops / peak_gflops, in nanoseconds).
    pub compute_cycles: f64,
    /// Estimated time spent on memory (bytes / peak_bandwidth, in nanoseconds).
    pub memory_cycles: f64,
}

/// FLOP cost for a single TraceOp.
///
/// Simple arithmetic = 1 FLOP, FMA = 2, transcendentals = polynomial
/// approximation cost (Exp/Log ~10, Tanh ~12).
fn trace_op_flops(op: &TraceOp) -> u64 {
    match op {
        TraceOp::Input(_) | TraceOp::Const(_) => 0,
        TraceOp::Add(..) | TraceOp::Sub(..) | TraceOp::Mul(..) | TraceOp::Div(..) => 1,
        TraceOp::Neg(..) | TraceOp::Abs(..) | TraceOp::Recip(..) => 1,
        TraceOp::Fma(..) => 2,
        TraceOp::Sqrt(..) | TraceOp::Rsqrt(..) => 2,
        TraceOp::Max(..) | TraceOp::Min(..) => 1,
        TraceOp::Exp(..) | TraceOp::Log(..) => 10,
        TraceOp::Tanh(..) | TraceOp::Sigmoid(..) => 12,
        TraceOp::ConditionalBranch(..) => 4, // compare + and + andnot + or
        // New extended TraceOp variants (§12+§14): conservative cost estimates.
        // QuantFma/BlockScale are mixed-precision FMAs (heavier than plain FMA).
        // Cast is a single-instruction type conversion.
        // HReduce is a horizontal reduction (~log2(width) shuffles).
        // Prefetch/NonTemporalStore are memory hints with no compute cost.
        // BitExtract/Permute are single-instruction bit/shuffle ops.
        // Compare/MaskedOp are predicate generation + application.
        // AtomicAdd has high latency but low FLOP count.
        // FWHT is O(d log d) butterfly — approximate per-element cost as log2(dim).
        TraceOp::QuantFma { .. } => 4,
        TraceOp::BlockScale { .. } => 2,
        TraceOp::Cast { .. } => 1,
        TraceOp::HReduce { .. } => 6,
        TraceOp::Prefetch { .. } | TraceOp::NonTemporalStore => 0,
        TraceOp::BitExtract { .. } | TraceOp::Permute { .. } => 1,
        TraceOp::Compare { .. } => 1,
        TraceOp::MaskedOp { .. } => 2,
        TraceOp::AtomicAdd { .. } => 3,
        TraceOp::FWHT { .. } => 8,
        // Structural memory operations (structural): memory access, not compute.
        TraceOp::ScalarLoad { .. } => 0,
        TraceOp::StrideMul { .. } => 1,
        TraceOp::PtrAdd { .. } => 0,
        TraceOp::VecLoadIndexed { .. } => 0,
        TraceOp::VecStoreIndexed { .. } => 0,
        // Vector broadcast + indexed memory operations
        TraceOp::BroadcastScalar { .. } | TraceOp::BroadcastLoad { .. } => 1,
        TraceOp::GatherLoad { .. } => 0,
        TraceOp::ScatterStore { .. } => 0,
        TraceOp::TableLookup { .. } => 0,
        // Quantization dequant: memory load + compute (counts as ~1 FLOP per element)
        TraceOp::Mxfp4Dequant { .. } => 1,
        // Bitwise AND: 1 op per element
        TraceOp::BitAnd(..) => 1,
        // Quant decode ops: memory load + bit manipulation + dequant algebra
        TraceOp::QuantBitAnd { .. } | TraceOp::QuantBitOr { .. } => 1,
        TraceOp::QuantBroadcast { .. } => 1,
        TraceOp::QuantCastF16toF32 { .. } | TraceOp::QuantCastI8toF32 { .. } | TraceOp::QuantCastFp8toF32 { .. } => 1,
        TraceOp::QuantCodebookLookup { vector_size, .. } => *vector_size as u64,
        TraceOp::QuantExtractBits { .. } => 1,
        TraceOp::QuantDequantFma { .. } => 2,
        TraceOp::QuantIntDivConst { .. } | TraceOp::QuantIntMul { .. } => 1,
        TraceOp::QuantInterleave { .. } | TraceOp::QuantConcatSeq { .. } => 1,
        TraceOp::QuantPtrAddOffset { .. } | TraceOp::QuantPtrAddDynamic { .. } | TraceOp::QuantAndMask { .. } | TraceOp::QuantScalarLoad { .. } | TraceOp::QuantLoadF16toF32 { .. } | TraceOp::QuantLoadI8toF32 { .. } | TraceOp::QuantLoadBytesVec { .. } | TraceOp::QuantKQuantPackedScaleLookup { .. } => 0,
        TraceOp::QuantShiftLeft { .. } | TraceOp::QuantShiftRight { .. } => 1,
        // SPEC 24-QUANT-PIPELINE-JIT: quant block-level load TraceOps
        TraceOp::QuantScaleLoad { .. } | TraceOp::QuantDataLoad { .. }
        | TraceOp::QuantZeroLoad { .. } | TraceOp::QuantSubScaleLoad { .. }
        | TraceOp::QuantHighBitsLoad { .. } | TraceOp::QuantE2m1LutDecode { .. } => 0,
        TraceOp::QuantCodebookDequant { .. } => 4,
        TraceOp::QuantQ3KDecode { .. } => 8,
        // ── SPEC 27 AT-002: 结构型扩展 ──
        TraceOp::Loop { .. } => 0,
        TraceOp::PanelLoad { .. } | TraceOp::PanelStore { .. } => 0,
        TraceOp::PackBuffer { .. } => 0,
        TraceOp::SharedMemDeclare { .. } | TraceOp::AsyncCopyToShared { .. }
        | TraceOp::AsyncWaitGroup { .. } | TraceOp::SyncBarrier { .. } => 0,
        TraceOp::TileConfig { .. } => 0,
        TraceOp::TileMma { .. } => 2,
        TraceOp::TileRelease => 0,
        TraceOp::Softmax { .. } => 20,
        TraceOp::EpilogueChain { .. } => 2,
        // ── SPEC 24-QUANT-PIPELINE-JIT: QuantGather/QuantGemm structural ──
        TraceOp::QuantGather { .. } | TraceOp::QuantGemm { .. } => 0,
        TraceOp::MtpDraft { .. }
        | TraceOp::MlaAttnScore { .. }
        | TraceOp::MlaRopeMerge { .. }
        | TraceOp::Tma2DCopy { .. }
        | TraceOp::DynamicPrecisionSelect { .. } => 0,
    }
}

/// Count per-element FLOPs from a ComputePattern's body.
fn pattern_per_element_flops(pattern: &ComputePattern) -> u64 {
    match pattern {
        ComputePattern::Gemm => 0, // GEMM flops come from M*N*K, handled separately
        ComputePattern::Elementwise { body }
        | ComputePattern::BinaryElementwise { body }
        | ComputePattern::Injective { body, .. }
        | ComputePattern::QuantDecode { decode: body, .. } => {
            body.iter().map(trace_op_flops).sum()
        }
        ComputePattern::NormLike { reduce, finalize, transform } => {
            // reduce and transform run per-element; finalize runs once (amortized to ~0)
            let r: u64 = reduce.iter().map(trace_op_flops).sum();
            let t: u64 = transform.iter().map(trace_op_flops).sum();
            let f: u64 = finalize.iter().map(trace_op_flops).sum();
            r + t + f
        }
        ComputePattern::Reduction { combine, .. } => {
            combine.iter().map(trace_op_flops).sum()
        }
    }
}

/// Extract (max_dim, num_input_ptrs, num_output_ptrs) from a scalar function signature.
fn parse_signature(sig: &crate::compiler::trace::ScalarFnSignature) -> (u64, u64, u64) {
    let mut inputs = 0u64;
    let mut outputs = 0u64;
    let mut dims = Vec::new();
    for p in &sig.params {
        match p {
            ScalarParam::InputPtr | ScalarParam::WeightPtr => inputs += 1,
            ScalarParam::OutputPtr => outputs += 1,
            ScalarParam::Dim(d) => dims.push(*d as u64),
            ScalarParam::Scalar(_) => {}
        }
    }
    let max_dim = dims.iter().copied().max().unwrap_or(0);
    (max_dim, inputs, outputs)
}

impl Cost {
    /// Compute roofline cost for a single operator from its OpTrace.
    ///
    /// For elementwise/norm/reduction patterns, FLOPs are counted from the
    /// TraceOp body and scaled by the dimension from the signature.
    /// For GEMM, FLOPs = 2*M*N*K derived from the signature's Dim params.
    /// Memory bytes = (input_ptrs + output_ptrs) * dim * sizeof(f32).
    pub fn compute(trace: &OpTrace, plan: &crate::compiler::planner::ExecutionPlan) -> Self {
        let (max_dim, input_ptrs, output_ptrs) = parse_signature(&trace.signature);

        // Determine element size from trace dtype hint (default F32 = 4 bytes).
        let elem_bytes: u64 = trace.signature.params.iter().find_map(|p| {
            if let ScalarParam::Scalar(v) = p {
                // dtype encoded as size_bytes (1/2/4) in the scalar slot
                let v = *v as u64;
                if v == 1 || v == 2 || v == 4 { Some(v) } else { None }
            } else {
                None
            }
        }).unwrap_or(DType::F32.size_bytes() as u64); // default F32

        let (flops, bytes) = if matches!(trace.pattern, ComputePattern::Gemm) {
            // GEMM: extract M, N, K from Dim params -> flops = 2*M*N*K
            let dims: Vec<u64> = trace.signature.params.iter().filter_map(|p| {
                if let ScalarParam::Dim(d) = p { Some(*d as u64) } else { None }
            }).collect();
            let (m, n, k) = if dims.len() >= 3 {
                (dims[0], dims[1], dims[2])
            } else {
                (max_dim, max_dim, max_dim)
            };
            let flops = 2 * m * n * k;
            // A(M*K) + B(K*N) + C(M*N), sized by elem_bytes
            let bytes = (m * k + k * n + m * n) * elem_bytes;
            (flops, bytes)
        } else {
            let per_elem = pattern_per_element_flops(&trace.pattern);
            let flops = per_elem * max_dim;
            let bytes = (input_ptrs + output_ptrs) * max_dim * elem_bytes;
            (flops, bytes)
        };

        // Convert to nanoseconds:
        //   peak_gflops is in GFLOP/s = 1e9 FLOP/s
        //   peak_bandwidth_gbs is in GB/s = 1e9 B/s
        //   compute_ns = flops / (peak_gflops * 1e9) * 1e9 = flops / peak_gflops
        //   memory_ns  = bytes / (peak_bw * 1e9) * 1e9     = bytes / peak_bw
        // ARCH-DTYPE-FULLCHAIN-ORCH: use dtype-aware peak_gflops for correct roofline
        let dtype_from_eb = match elem_bytes {
            2 => crate::types::DType::F16,
            _ => crate::types::DType::F32,
        };
        let peak_gflops = plan.profile.peak_gflops(dtype_from_eb);
        let compute_cycles = if peak_gflops > 0.0 {
            flops as f64 / peak_gflops
        } else {
            0.0
        };
        let memory_cycles = if plan.roofline.peak_bandwidth_gbs > 0.0 {
            bytes as f64 / plan.roofline.peak_bandwidth_gbs
        } else {
            0.0
        };

        Cost { flops, bytes, compute_cycles, memory_cycles }
    }

    /// Compute the memory-cycle savings from eliminating intermediate traffic.
    ///
    /// Returns saved nanoseconds (as integer) from not writing + reading back
    /// `eliminated_bytes` of intermediate data.
    pub fn fusion_benefit(eliminated_bytes: usize, plan: &crate::compiler::planner::ExecutionPlan) -> u64 {
        if plan.roofline.peak_bandwidth_gbs > 0.0 {
            (eliminated_bytes as f64 / plan.roofline.peak_bandwidth_gbs) as u64
        } else {
            0
        }
    }

    /// True if this operator is compute-bound (compute time > memory time).
    ///
    /// Equivalently, arithmetic intensity > roofline ridge point.
    #[inline]
    pub fn is_compute_bound(&self) -> bool {
        self.compute_cycles > self.memory_cycles
    }
}

/// Compute eliminated bytes for a candidate elementwise chain fusion.
///
/// Each intermediate tensor between adjacent ops in the chain is written then
/// read back — fusion eliminates both accesses (2x tensor size).
pub(crate) fn chain_eliminated_bytes(graph: &CompilerGraph, anchor: &CompilerOp, chain: &[&CompilerOp]) -> usize {
    if chain.is_empty() {
        return 0;
    }
    let mut eliminated = 0usize;
    // anchor's output -> chain[0]: eliminated
    for &out_tid in &anchor.outputs {
        if let Some(t) = graph.tensor(out_tid) {
            if t.consumers.len() == 1 {
                eliminated += t.concrete_bytes() * 2;
            }
        }
    }
    // Each chain op's output (except the last) -> next chain op: eliminated
    for i in 0..chain.len().saturating_sub(1) {
        for &out_tid in &chain[i].outputs {
            if let Some(t) = graph.tensor(out_tid) {
                if t.consumers.len() == 1 {
                    eliminated += t.concrete_bytes() * 2;
                }
            }
        }
    }
    eliminated
}


// -- REQ-FUS-009: Fusion Cost Model --------------------------------

/// Hardware-weighted fusion cost model (REQ-FUS-009).
///
/// Extends the basic roofline model with profile-specific weights for
/// compute ROI, cache ROI, and latency ROI. Each HardwareProfile has
/// different cost weights reflecting its architecture characteristics.
pub struct FusionCostModel {
    /// Weight for compute overhead vs memory savings (0.0-2.0).
    /// Low = compute is cheap (GPU tensor cores), High = compute is expensive (scalar CPU).
    pub compute_roi_weight: f64,
    /// Weight for cache-friendliness of the fusion (0.5-1.5).
    /// High = cache-friendly fusion is very beneficial (large L1/shared memory).
    pub cache_roi_weight: f64,
    /// Weight for latency reduction (kernel launch overhead or pipeline stall).
    /// GPU: high (kernel launch is expensive). CPU: low (no kernel launch).
    pub latency_roi_weight: f64,
    /// Minimum benefit threshold from HardwareProfile.
    pub min_benefit: f32,
    /// Maximum fusion depth from HardwareProfile.
    pub max_depth: usize,
    /// Whether the profile supports quantized epilogue fusion.
    pub quant_epilogue: bool,
}

impl FusionCostModel {
    /// Build cost model from HardwareProfile.
    pub fn from_profile(hw: HardwareProfile) -> Self {
        let latency_weight = if hw.tensor_core_gen() > 0 {
            // GPU: kernel launch overhead is significant (~5-20us).
            // Fusion eliminates per-kernel launch -> high latency ROI.
            1.5
        } else {
            // CPU: no kernel launch overhead. Fusion benefit is memory-only.
            0.5
        };

        Self {
            compute_roi_weight: hw.compute_roi_weight(),
            cache_roi_weight: hw.cache_roi_weight(),
            latency_roi_weight: latency_weight,
            min_benefit: hw.min_fusion_benefit(),
            max_depth: hw.max_fusion_depth(),
            quant_epilogue: hw.supports_quant_epilogue(),
        }
    }

    /// Evaluate whether a fusion decision is profitable.
    ///
    /// Returns true if the estimated benefit exceeds the minimum threshold
    /// for this hardware profile, accounting for compute/cache/latency weights.
    pub fn is_profitable(&self, raw_benefit: i64, epilogue_depth: usize) -> bool {
        if epilogue_depth > self.max_depth {
            return false;
        }
        if raw_benefit <= 0 {
            return false;
        }
        // Scale benefit by compute weight (higher compute weight -> harder to justify fusion).
        let adjusted = raw_benefit as f64 / self.compute_roi_weight;
        adjusted > (self.min_benefit * 100.0) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::fusion::GroupMarker;
    use crate::compiler::graph::{CompilerGraph, OpId, OpKind, SymDim, MultiOutputConfig};
    use crate::compiler::planner::ExecutionPlan;
    use crate::compiler::hardware_profile::HardwareProfile;
    use crate::compiler::trace::{OpTrace, ComputePattern, ScalarFnSignature, ScalarParam, TraceOp, ValueId};
    use crate::compiler::pain_point::{OpBottleneckMap, GemmBottleneck, GemmRole, BottleneckType, FusionPriority, ExecPattern, ParallelismDesc};
    use crate::dispatch::DeviceProfile;
    use crate::types::DType;
    use super::super::types::FusionGroup;
    use std::collections::HashMap;

    // ── Helpers ──

    fn vid(n: u32) -> ValueId { ValueId(n) }

    fn make_plan() -> ExecutionPlan {
        ExecutionPlan::from_profile(&DeviceProfile::detect())
    }

    fn make_epilogue_group(anchor: OpId, epilogue: Vec<OpId>) -> FusionGroup {
        let mut ops = vec![anchor];
        ops.extend_from_slice(&epilogue);
        FusionGroup {
            id: 0,
            anchor,
            epilogue,
            mode: FusionMode::EpilogueInjection,
            ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }
    }

    fn make_loop_fusion_group(ops: Vec<OpId>) -> FusionGroup {
        let anchor = ops[0];
        let epilogue = ops[1..].to_vec();
        FusionGroup {
            id: 0,
            anchor,
            epilogue,
            mode: FusionMode::LoopFusion,
            ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }
    }

    fn make_standalone_group(op: OpId) -> FusionGroup {
        FusionGroup {
            id: 0,
            anchor: op,
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }
    }

    fn make_tile_level_group(anchor: OpId, predecessor: OpId, tile_rows: usize) -> FusionGroup {
        FusionGroup {
            id: 0,
            anchor,
            epilogue: vec![predecessor],
            mode: FusionMode::TileLevelFusion { predecessor, tile_rows },
            ops: vec![predecessor, anchor],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }
    }

    fn make_bottleneck_map(op_id: OpId, bn: BottleneckType) -> OpBottleneckMap {
        let mut gemm_bottlenecks = HashMap::new();
        gemm_bottlenecks.insert(op_id, GemmBottleneck {
            gemm_role: GemmRole::Other,
            shape: (1, 4096, 4096),
            arithmetic_intensity: 1.0,
            ridge_point: 10.0,
            bottleneck: bn,
            optimal_fusion: FusionPriority::EpilogueInjection,
            fusion_benefits: HashMap::new(),
            exec_pattern: ExecPattern::ScalarLoop,
            parallelism: ParallelismDesc::SimdVectorize { element_width: 4, unroll_factor: 1 },
        });
        OpBottleneckMap {
            gemm_bottlenecks,
            ridge_point: 10.0,
        }
    }

    fn make_sig(params: Vec<ScalarParam>) -> ScalarFnSignature {
        ScalarFnSignature {
            fn_ptr: dummy_fn as *const u8,
            params,
        }
    }

    extern "C" fn dummy_fn() {}

    // ── Cost struct tests ──

    #[test]
    fn cost_construct_and_clone() {
        let c = Cost { flops: 100, bytes: 400, compute_cycles: 10.0, memory_cycles: 20.0 };
        let cloned = c.clone();
        assert_eq!(cloned.flops, 100);
        assert_eq!(cloned.bytes, 400);
        assert_eq!(cloned.compute_cycles, 10.0);
        assert_eq!(cloned.memory_cycles, 20.0);
    }

    #[test]
    fn cost_is_compute_bound_true() {
        let c = Cost { flops: 2000, bytes: 100, compute_cycles: 50.0, memory_cycles: 10.0 };
        assert!(c.is_compute_bound());
    }

    #[test]
    fn cost_is_compute_bound_false() {
        let c = Cost { flops: 100, bytes: 2000, compute_cycles: 5.0, memory_cycles: 50.0 };
        assert!(!c.is_compute_bound());
    }

    #[test]
    fn cost_is_compute_bound_equal() {
        let c = Cost { flops: 100, bytes: 100, compute_cycles: 20.0, memory_cycles: 20.0 };
        assert!(!c.is_compute_bound()); // strictly greater
    }

    #[test]
    fn cost_zero_flops_and_bytes() {
        let c = Cost { flops: 0, bytes: 0, compute_cycles: 0.0, memory_cycles: 0.0 };
        assert!(!c.is_compute_bound());
        assert_eq!(c.flops, 0);
        assert_eq!(c.bytes, 0);
    }

    // ── Cost::compute tests ──

    #[test]
    fn cost_compute_elementwise_pattern() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(1024),
        ]);
        let trace = OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body: vec![TraceOp::Exp(vid(0))] },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // per_elem_flops(Exp) = 10; 10 * 1024 = 10240
        assert_eq!(cost.flops, 10240);
        // bytes = (1 input + 1 output) * 1024 * 4 = 8192
        assert_eq!(cost.bytes, 8192);
    }

    #[test]
    fn cost_compute_gemm_pattern() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(4),
            ScalarParam::Dim(8),
            ScalarParam::Dim(16),
        ]);
        let trace = OpTrace {
            op_kind: OpKind::Gemm { m: SymDim::Concrete(4), n: 8, k: 16, dtype: DType::F32, trans_b: false },
            pattern: ComputePattern::Gemm,
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // flops = 2 * 4 * 8 * 16 = 1024
        assert_eq!(cost.flops, 1024);
        // bytes = (4*16 + 16*8 + 4*8) * 4 = (64+128+32)*4 = 896
        assert_eq!(cost.bytes, 896);
    }

    #[test]
    fn cost_compute_normlike_pattern() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(512),
        ]);
        // NormLike: reduce + finalize + transform
        let trace = OpTrace {
            op_kind: OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            pattern: ComputePattern::NormLike {
                reduce: vec![TraceOp::Mul(vid(0), vid(0))],
                finalize: vec![TraceOp::Sqrt(vid(0))],
                transform: vec![TraceOp::Div(vid(0), vid(1))],
            },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // per_elem = Mul(1) + Sqrt(2) + Div(1) = 4; 4 * 512 = 2048
        assert_eq!(cost.flops, 2048);
        // bytes = (1 input + 1 output) * 512 * 4 = 4096
        assert_eq!(cost.bytes, 4096);
    }

    #[test]
    fn cost_compute_reduction_pattern() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(256),
        ]);
        let trace = OpTrace {
            op_kind: OpKind::Tanh,
            pattern: ComputePattern::Reduction {
                identity: 0.0,
                combine: vec![TraceOp::Add(vid(0), vid(1))],
                second_pass: None,
                normalize: None,
            },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // per_elem = Add(1); 1 * 256 = 256
        assert_eq!(cost.flops, 256);
    }

    #[test]
    fn cost_compute_empty_body_zero_flops() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(128),
        ]);
        let trace = OpTrace {
            op_kind: OpKind::Tanh,
            pattern: ComputePattern::Elementwise { body: vec![] },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        assert_eq!(cost.flops, 0);
        assert_eq!(cost.bytes, 1024); // (1+1)*128*4
    }

    #[test]
    fn cost_compute_no_dims_defaults_to_zero() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
        ]);
        let trace = OpTrace {
            op_kind: OpKind::Tanh,
            pattern: ComputePattern::Elementwise { body: vec![TraceOp::Exp(vid(0))] },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        assert_eq!(cost.flops, 0); // 10 * 0 = 0
        assert_eq!(cost.bytes, 0);  // (1+1)*0*4 = 0
    }

    #[test]
    fn cost_compute_gemm_with_two_dims_falls_back_to_max() {
        // Only 2 Dim params => falls back to (max_dim, max_dim, max_dim)
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(4),
            ScalarParam::Dim(8),
        ]);
        let trace = OpTrace {
            op_kind: OpKind::Gemm { m: SymDim::Concrete(4), n: 8, k: 8, dtype: DType::F32, trans_b: false },
            pattern: ComputePattern::Gemm,
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // max_dim = 8; flops = 2 * 8 * 8 * 8 = 1024
        assert_eq!(cost.flops, 1024);
    }

    // ── Cost::fusion_benefit tests ──

    #[test]
    fn cost_fusion_benefit_nonzero_with_valid_bandwidth() {
        let plan = make_plan();
        if plan.roofline.peak_bandwidth_gbs > 0.0 {
            let saved = Cost::fusion_benefit(1024, &plan);
            let expected = (1024.0 / plan.roofline.peak_bandwidth_gbs) as u64;
            assert_eq!(saved, expected);
        }
        // If bandwidth is 0 (unlikely in test env), just verify it returns 0
    }

    #[test]
    fn cost_fusion_benefit_zero_bytes_yields_zero() {
        let plan = make_plan();
        let saved = Cost::fusion_benefit(0, &plan);
        assert_eq!(saved, 0);
    }

    // ── FusionCostModel tests ──

    #[test]
    fn fusion_cost_model_from_profile_cpu() {
        let model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        assert!(model.compute_roi_weight > 0.0);
        assert!(model.cache_roi_weight > 0.0);
        assert!(model.latency_roi_weight > 0.0);
        assert!(model.min_benefit >= 0.0);
        assert!(model.max_depth > 0);
    }

    #[test]
    fn fusion_cost_model_from_profile_gpu() {
        let model = FusionCostModel::from_profile(HardwareProfile::CudaSM80);
        // GPU has higher latency weight due to kernel launch overhead
        assert!(model.latency_roi_weight > 1.0);
    }

    #[test]
    fn fusion_cost_model_from_profile_all_variants() {
        let profiles = [
            HardwareProfile::CudaSM80,
            HardwareProfile::CudaSM90,
            HardwareProfile::CudaSM100,
            HardwareProfile::RocmMI200,
            HardwareProfile::RocmMI300,
            HardwareProfile::CpuAvx2,
            HardwareProfile::CpuAvx512,
            HardwareProfile::CpuAvx10_2,
            HardwareProfile::AppleM1,
            HardwareProfile::AppleM2,
            HardwareProfile::AppleM3,
            HardwareProfile::ArmNeoverse,
            HardwareProfile::Generic,
        ];
        for hw in &profiles {
            let model = FusionCostModel::from_profile(*hw);
            assert!(model.compute_roi_weight > 0.0, "compute_roi_weight for {:?}", hw);
            assert!(model.max_depth > 0, "max_depth for {:?}", hw);
        }
    }

    #[test]
    fn fusion_cost_model_gpu_latency_weight_higher_than_cpu() {
        let cpu_model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        let gpu_model = FusionCostModel::from_profile(HardwareProfile::CudaSM80);
        assert!(gpu_model.latency_roi_weight > cpu_model.latency_roi_weight);
    }

    // ── FusionCostModel::is_profitable tests ──

    #[test]
    fn is_profitable_rejects_zero() {
        let model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        assert!(!model.is_profitable(0, 0));
    }

    #[test]
    fn is_profitable_rejects_negative() {
        let model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        assert!(!model.is_profitable(-1000, 0));
    }

    #[test]
    fn is_profitable_rejects_depth_exceeding_max() {
        let model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        assert!(!model.is_profitable(1_000_000, model.max_depth + 1));
    }

    #[test]
    fn is_profitable_accepts_at_max_depth() {
        let model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        // Very large benefit should be accepted at exactly max_depth
        assert!(model.is_profitable(100_000_000, model.max_depth));
    }

    #[test]
    fn is_profitable_accepts_large_benefit() {
        let model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        assert!(model.is_profitable(10_000_000, 1));
    }

    #[test]
    fn is_profitable_higher_compute_weight_harder_to_justify() {
        let mut model = FusionCostModel::from_profile(HardwareProfile::CpuAvx2);
        let small_benefit = 100i64;
        // With default weight
        let default_result = model.is_profitable(small_benefit, 0);
        // With very high compute weight, same benefit is rejected
        model.compute_roi_weight = 100.0;
        let high_weight_result = model.is_profitable(small_benefit, 0);
        assert!(!high_weight_result);
        // If default accepted, high weight must not; if both reject, that is also fine
        if default_result {
            assert!(!high_weight_result);
        }
    }

    // ── compute_group_ai tests ──

    #[test]
    fn compute_group_ai_with_gemm_ops() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);

        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let ai = compute_group_ai(&group, &g);
        // flops = 2*1*4096*4096 = 33554432; bytes = (1*4096 + 4096*4096 + 1*4096)*4
        assert!(ai > 0.0, "AI should be positive for GEMM");
    }

    #[test]
    fn compute_group_ai_empty_ops_returns_zero() {
        let group = FusionGroup {
            id: 0,
            anchor: OpId(999),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let g = CompilerGraph::new();
        let ai = compute_group_ai(&group, &g);
        assert_eq!(ai, 0.0);
    }

    #[test]
    fn compute_group_ai_non_gemm_ops_zero_flops() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let group = make_standalone_group(op0);
        let ai = compute_group_ai(&group, &g);
        // Non-GEMM ops have zero flops in estimate_op_flops, but non-zero bytes
        // So AI = 0 / bytes = 0
        assert_eq!(ai, 0.0);
    }

    // ── compute_group_roofline_scale tests ──

    #[test]
    fn compute_group_roofline_scale_memory_bound_returns_one() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let plan = make_plan();
        let bmap = make_bottleneck_map(op0, BottleneckType::MemoryBound { bandwidth_utilization: 0.8 });
        let scale = compute_group_roofline_scale(&group, &g, &plan, Some(&bmap));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn compute_group_roofline_scale_compute_bound_returns_utilization() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let plan = make_plan();
        let bmap = make_bottleneck_map(op0, BottleneckType::ComputeBound { compute_utilization: 0.5 });
        let scale = compute_group_roofline_scale(&group, &g, &plan, Some(&bmap));
        assert!((scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_group_roofline_scale_latency_bound_returns_half() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let plan = make_plan();
        let bmap = make_bottleneck_map(op0, BottleneckType::LatencyBound { estimated_latency_ns: 5.0 });
        let scale = compute_group_roofline_scale(&group, &g, &plan, Some(&bmap));
        assert_eq!(scale, 0.5);
    }

    #[test]
    fn compute_group_roofline_scale_no_bottleneck_map() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let plan = make_plan();
        let scale = compute_group_roofline_scale(&group, &g, &plan, None);
        // Without bottleneck map, scale depends on AI vs ridge point
        assert!(scale > 0.0 && scale <= 1.0);
    }

    #[test]
    fn compute_group_roofline_scale_zero_ridge() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let group = make_standalone_group(op0);
        let plan = make_plan();
        // Ridge point from DeviceProfile is positive, but without bottleneck map
        // and if ridge <= 0, returns 1.0
        let scale = compute_group_roofline_scale(&group, &g, &plan, None);
        assert!(scale > 0.0);
    }

    // ── is_memory_bound_group tests ──

    #[test]
    fn is_memory_bound_group_zero_ridge() {
        // With a plan that has positive ridge, this tests the AI < ridge path
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let group = make_standalone_group(op0);
        let plan = make_plan();
        // Non-GEMM has zero flops, so AI=0 which is always < ridge => memory-bound
        assert!(is_memory_bound_group(&group, &g, &plan));
    }

    #[test]
    fn is_memory_bound_group_small_gemm() {
        // Small GEMM with large weight tensor => low AI => memory-bound
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4], dt);
        let w = g.add_tensor_concrete("w", &[4, 4], dt);
        let out = g.add_tensor_concrete("out", &[1, 4], dt);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let plan = make_plan();
        // flops=2*1*4*4=32; bytes=(4+16+4)*4=96; AI=0.33
        let result = is_memory_bound_group(&group, &g, &plan);
        // With typical ridge ~10, AI=0.33 < 10 => memory-bound
        assert!(result);
    }

    #[test]
    fn is_memory_bound_group_large_gemm() {
        // Large GEMM with high AI may be compute-bound
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[4096, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[4096, 4096], dt);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(4096), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = make_standalone_group(op0);
        let plan = make_plan();
        // flops=2*4096^3; bytes=(4096^2+4096^2+4096^2)*4; AI ~ 2*4096/3 ≈ 2730
        // With typical ridge ~10, AI > 10 => compute-bound
        assert!(!is_memory_bound_group(&group, &g, &plan));
    }

    // ── estimate_fusion_cost tests ──

    #[test]
    fn estimate_fusion_cost_epilogue_with_internal_consumer() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 128], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 128], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 128], dt);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let silu = g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let group = make_epilogue_group(gemm, vec![silu]);
        let plan = make_plan();
        let cost = estimate_fusion_cost(&group, &g, &plan, None);

        // gemm_out is consumed only by silu (internal to group) => bytes saved
        assert!(cost.bytes_saved > 0);
        // EpilogueInjection => scratch_bytes = 0
        assert_eq!(cost.scratch_bytes, 0);
    }

    #[test]
    fn estimate_fusion_cost_external_consumer_no_savings() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 128], dt);
        let b = g.add_tensor_concrete("b", &[1, 128], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 128], dt);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 128], dt);
        let ext_out = g.add_tensor_concrete("ext_out", &[1, 128], dt);

        let add_op = g.add_op(OpKind::Add, vec![a, b], vec![add_out], "add");
        let tanh_op = g.add_op(OpKind::Tanh, vec![add_out], vec![tanh_out], "tanh");
        // External consumer of tanh_out
        g.add_op(OpKind::Tanh, vec![tanh_out], vec![ext_out], "external");

        let group = make_loop_fusion_group(vec![add_op, tanh_op]);
        let plan = make_plan();
        let cost = estimate_fusion_cost(&group, &g, &plan, None);

        // tanh_out has external consumer => not eliminated
        // add_out has only tanh as consumer (internal) => eliminated
        // So bytes_saved = add_out bytes * 2
        let add_out_bytes = 1 * 128 * 4; // F32 intermediate
        assert_eq!(cost.bytes_saved, add_out_bytes * 2);
    }

    #[test]
    fn estimate_fusion_cost_loop_fusion_group() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let mid = g.add_tensor_concrete("mid", &[1, 64], dt);
        let out = g.add_tensor_concrete("out", &[1, 64], dt);

        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let op1 = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let group = make_loop_fusion_group(vec![op0, op1]);
        let plan = make_plan();
        let cost = estimate_fusion_cost(&group, &g, &plan, None);

        // mid is internal (consumed only by op1 within group) => eliminated
        let mid_bytes = 1 * 64 * 4;
        assert_eq!(cost.bytes_saved, mid_bytes * 2);
        assert_eq!(cost.scratch_bytes, 0);
    }

    #[test]
    fn estimate_fusion_cost_standalone_zero_benefit() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 128], dt);
        let out = g.add_tensor_concrete("out", &[1, 128], dt);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let group = make_standalone_group(op0);
        let plan = make_plan();
        let cost = estimate_fusion_cost(&group, &g, &plan, None);

        // Standalone: out has no consumers inside the group (only out itself which is the last output)
        // Actually out might have no consumers at all (no downstream ops)
        assert_eq!(cost.bytes_saved, 0); // no intermediates eliminated
    }

    #[test]
    fn estimate_fusion_cost_with_bottleneck_map() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 128], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 128], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 128], dt);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let silu = g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let group = make_epilogue_group(gemm, vec![silu]);
        let plan = make_plan();
        let bmap = make_bottleneck_map(gemm, BottleneckType::MemoryBound { bandwidth_utilization: 0.9 });
        let cost = estimate_fusion_cost(&group, &g, &plan, Some(&bmap));

        // With memory-bound bottleneck, roofline_scale = 1.0 (full benefit)
        assert!(cost.bytes_saved > 0);
    }

    #[test]
    fn estimate_fusion_cost_tile_level_fusion_has_scratch() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let norm_in = g.add_tensor_concrete("norm_in", &[1, 4096], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);

        let norm = g.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![norm_in], vec![norm_out], "norm");
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );

        let tile_rows = 32;
        let group = make_tile_level_group(gemm, norm, tile_rows);
        let plan = make_plan();
        let cost = estimate_fusion_cost(&group, &g, &plan, None);

        // TileLevelFusion: scratch = tile_rows * K * elem_bytes
        assert_eq!(cost.scratch_bytes, tile_rows * 4096 * 4);
    }

    // ── chain_eliminated_bytes tests ──

    #[test]
    fn chain_eliminated_bytes_empty_chain() {
        let g = CompilerGraph::new();
        let dt = DType::F32;
        // Need an op for anchor
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let out = g.add_tensor_concrete("out", &[1, 64], dt);
        let anchor = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");
        let anchor_op = g.op(anchor).unwrap().clone();
        let eliminated = chain_eliminated_bytes(&g, &anchor_op, &[]);
        assert_eq!(eliminated, 0);
    }

    #[test]
    fn chain_eliminated_bytes_single_consumer() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let mid = g.add_tensor_concrete("mid", &[1, 64], dt);
        let out = g.add_tensor_concrete("out", &[1, 64], dt);

        let anchor = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let chain_op = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let anchor_op = g.op(anchor).unwrap().clone();
        let chain_ref = g.op(chain_op).unwrap();
        let eliminated = chain_eliminated_bytes(&g, &anchor_op, &[chain_ref]);

        // mid has exactly 1 consumer (chain_op) => eliminated
        // mid bytes = 1*64*4 = 256; eliminated = 256 * 2 = 512
        assert_eq!(eliminated, 512);
    }

    #[test]
    fn chain_eliminated_bytes_multi_consumer_not_eliminated() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let mid = g.add_tensor_concrete("mid", &[1, 64], dt);
        let out1 = g.add_tensor_concrete("out1", &[1, 64], dt);
        let out2 = g.add_tensor_concrete("out2", &[1, 64], dt);

        let anchor = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let chain_op = g.add_op(OpKind::Silu, vec![mid], vec![out1], "silu");
        // Second consumer of mid => mid is NOT eliminated
        g.add_op(OpKind::Silu, vec![mid], vec![out2], "silu2");

        let anchor_op = g.op(anchor).unwrap().clone();
        let chain_ref = g.op(chain_op).unwrap();
        let eliminated = chain_eliminated_bytes(&g, &anchor_op, &[chain_ref]);

        // mid has 2 consumers => not eliminated
        assert_eq!(eliminated, 0);
    }

    #[test]
    fn chain_eliminated_bytes_multi_op_chain() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let mid1 = g.add_tensor_concrete("mid1", &[1, 64], dt);
        let mid2 = g.add_tensor_concrete("mid2", &[1, 64], dt);
        let out = g.add_tensor_concrete("out", &[1, 64], dt);

        let anchor = g.add_op(OpKind::Tanh, vec![a], vec![mid1], "tanh");
        let c0 = g.add_op(OpKind::Silu, vec![mid1], vec![mid2], "silu");
        let c1 = g.add_op(OpKind::Tanh, vec![mid2], vec![out], "tanh2");

        let anchor_op = g.op(anchor).unwrap().clone();
        let c0_ref = g.op(c0).unwrap();
        let c1_ref = g.op(c1).unwrap();
        let eliminated = chain_eliminated_bytes(&g, &anchor_op, &[c0_ref, c1_ref]);

        // anchor→c0: mid1 has 1 consumer (c0) => 256*2=512
        // c0→c1: mid2 has 1 consumer (c1) => 256*2=512
        // c1 is last => no elimination for c1 output
        assert_eq!(eliminated, 1024);
    }

    // ── trace_op_flops tests (exercised via pattern_per_element_flops) ──

    #[test]
    fn trace_op_flops_injective_pattern() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(256),
        ]);
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Mul(vid(0), vid(1)),   // 1
            TraceOp::Exp(vid(2)),            // 10
            TraceOp::Add(vid(2), vid(3)),    // 1
        ];
        let trace = OpTrace {
            op_kind: OpKind::Tanh,
            pattern: ComputePattern::Injective { body, num_inputs: 2, num_outputs: 2 },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // per_elem = 1 + 10 + 1 = 12; 12 * 256 = 3072
        assert_eq!(cost.flops, 3072);
    }

    #[test]
    fn trace_op_flops_quant_decode_pattern() {
        let sig = make_sig(vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(512),
        ]);
        let body = vec![
            TraceOp::Input(0),
            TraceOp::QuantBitAnd { lhs: vid(0), rhs: vid(0) },  // 1
            TraceOp::QuantDequantFma { acc: vid(1), a: vid(0), b: vid(0) }, // 2
        ];
        let trace = OpTrace {
            op_kind: OpKind::Tanh,
            pattern: ComputePattern::QuantDecode { block_size: 32, decode: body },
            signature: sig,
        };
        let plan = make_plan();
        let cost = Cost::compute(&trace, &plan);
        // per_elem = 1 + 2 = 3; 3 * 512 = 1536
        assert_eq!(cost.flops, 1536);
    }

    // ── FusionCost tests ──

    #[test]
    fn fusion_cost_clone_and_debug() {
        let fc = FusionCost {
            bytes_saved: 1024,
            extra_regs: 2,
            scratch_bytes: 0,
            benefit: 896,
        };
        let cloned = fc.clone();
        assert_eq!(cloned.bytes_saved, 1024);
        assert_eq!(cloned.extra_regs, 2);
        assert_eq!(cloned.scratch_bytes, 0);
        assert_eq!(cloned.benefit, 896);
        // Debug trait
        let debug_str = format!("{:?}", fc);
        assert!(debug_str.contains("bytes_saved"));
    }

    // ── FusionCostModel manual construction ──

    #[test]
    fn fusion_cost_model_manual_construction() {
        let model = FusionCostModel {
            compute_roi_weight: 1.0,
            cache_roi_weight: 1.0,
            latency_roi_weight: 0.5,
            min_benefit: 0.5,
            max_depth: 4,
            quant_epilogue: false,
        };
        assert_eq!(model.max_depth, 4);
        assert!(!model.quant_epilogue);
        // Small benefit below threshold
        assert!(!model.is_profitable(10, 0));
        // Very large benefit above threshold
        assert!(model.is_profitable(1_000_000, 0));
    }

    #[test]
    fn fusion_cost_model_from_generic_profile() {
        let model = FusionCostModel::from_profile(HardwareProfile::Generic);
        assert!(model.compute_roi_weight > 0.0);
        assert!(model.latency_roi_weight > 0.0);
    }
}


