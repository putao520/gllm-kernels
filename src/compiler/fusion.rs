//! Fusion pass — Phase 2 of the JIT compiler pipeline.
//!
//! Walks the CompilerGraph in topological order and groups adjacent ops
//! into `FusionGroup`s based on semantic compatibility rules from
//! `semantics::can_fuse()`.
//!
//! Fusion groups are the unit of code generation: each group becomes a
//! single loop nest or microkernel call in the emitted machine code.
//!
//! Key fusion patterns:
//! - GEMM + elementwise epilogue (bias, activation, residual add)
//! - Elementwise chain collapse (silu+mul → swiglu already in graph)
//! - QKV shared input: three GEMMs reading the same normed input → single pack_a
//! - RmsNorm → GEMM: norm output feeds GEMM input without memory writeback

use std::collections::{HashMap, HashSet};
use crate::compiler::graph::{CompilerGraph, CompilerOp, OpKind, OpId, TensorId};
use crate::compiler::semantics::{self, OpSemantics};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass};
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::{OpTrace, TraceOp, ComputePattern, ScalarParam};
use crate::dispatch::DeviceProfile;

/// A group of fused operations that will be compiled as a single unit.
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Unique group ID.
    pub id: usize,
    /// The "anchor" op — determines the primary computation pattern.
    /// For GEMM fusion, this is the GEMM op.
    /// For elementwise chains, this is the first op.
    pub anchor: OpId,
    /// Ops absorbed into this group's epilogue (in execution order).
    pub epilogue: Vec<OpId>,
    /// The fusion mode that was applied.
    pub mode: FusionMode,
    /// All op IDs in this group (anchor + epilogue), in execution order.
    pub ops: Vec<OpId>,
}

/// Named fusion modes recognized by the pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionMode {
    /// Single op, no fusion applied.
    Standalone,
    /// GEMM with fused elementwise epilogue (e.g., GEMM + SiLU, GEMM + Add).
    EpilogueInjection,
    /// Chain of elementwise ops collapsed into a single loop.
    LoopFusion,
    /// Tile-level fusion: predecessor tile computation embedded in GEMM MC loop.
    /// Used when predecessor output > 75% L1 — tiles are computed per MC strip.
    TileLevelFusion {
        /// The predecessor op (e.g. RmsNorm) whose output is tiled into the GEMM MC loop.
        predecessor: OpId,
        /// Number of rows per tile (= MC from GEMM blocking).
        tile_rows: usize,
    },
    /// Compute root: predecessor computed fully before GEMM, result stays in L1/L2.
    /// Used when predecessor output ≤ 75% L1.
    ComputeRoot {
        /// The predecessor op computed as a standalone root.
        predecessor: OpId,
    },
    /// Three QKV GEMMs sharing the same input → single pack_a.
    QkvSharedInput,
    /// RmsNorm output feeds directly into GEMM (no intermediate writeback).
    NormIntoGemm,
}

/// Result of the fusion pass.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Fusion groups in execution order.
    pub groups: Vec<FusionGroup>,
    /// Map from OpId → group index (for quick lookup).
    pub op_to_group: std::collections::HashMap<OpId, usize>,
}

impl FusionPlan {
    /// Number of fusion groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get the group containing a specific op.
    pub fn group_of(&self, op: OpId) -> Option<&FusionGroup> {
        self.op_to_group.get(&op).map(|&idx| &self.groups[idx])
    }

    /// Count how many ops were fused (not standalone).
    pub fn num_fused_ops(&self) -> usize {
        self.groups
            .iter()
            .filter(|g| g.mode != FusionMode::Standalone)
            .map(|g| g.ops.len())
            .sum()
    }
}

// ── Fusion cost model (WI-4) ────────────────────────────────────────────

/// Cost estimate for a fusion decision.
#[derive(Debug, Clone)]
pub struct FusionCost {
    /// Bytes of intermediate data eliminated by fusion (saved memory traffic).
    pub bytes_saved: usize,
    /// Extra registers consumed by the fused kernel vs separate kernels.
    pub extra_regs: usize,
    /// Scratch buffer bytes needed for tiled fusion (0 for epilogue/loop fusion).
    pub scratch_bytes: usize,
    /// Net benefit score: positive means fusion is profitable.
    /// `benefit = bytes_saved - penalty`, where penalty accounts for register
    /// spill cost and scratch buffer overhead.
    pub benefit: i64,
}

/// Estimate the cost/benefit of a fusion group.
///
/// The model is roofline-inspired:
/// - Benefit: bytes of intermediate tensors that no longer need to be written/read
///   from memory (2× the tensor size for write + read-back).
/// - Cost: register pressure increase may cause spills (each spill ≈ 64 bytes
///   round-trip to stack), plus scratch buffer allocation overhead.
pub fn estimate_fusion_cost(
    group: &FusionGroup,
    graph: &CompilerGraph,
    profile: &crate::dispatch::DeviceProfile,
) -> FusionCost {
    let avail_regs = profile.num_simd_regs();
    let (l1, _, _) = profile.cache_sizes();

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
                let size = tensor.shape.iter().product::<usize>() * tensor.dtype.size_bytes();
                bytes_saved += size * 2;
            }
        }
    }

    // Register pressure estimate
    let base_regs = match group.mode {
        FusionMode::EpilogueInjection => {
            // GEMM accumulators + epilogue temporaries
            let nr = profile.gemm_blocking(0, 0, 0).nr;
            let mr = profile.gemm_blocking(0, 0, 0).mr;
            let acc = (mr * nr) / (profile.simd_width_bytes() / 4);
            acc + group.epilogue.len().min(4)
        }
        FusionMode::TileLevelFusion { .. } => {
            let nr = profile.gemm_blocking(0, 0, 0).nr;
            let mr = profile.gemm_blocking(0, 0, 0).mr;
            let acc = (mr * nr) / (profile.simd_width_bytes() / 4);
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
            // Scratch = tile_rows × K × sizeof(f32) for the tiled norm output
            let k = group.ops.iter().find_map(|&oid| {
                graph.op(oid).and_then(|o| match &o.kind {
                    OpKind::Gemm { k, .. }
                    | OpKind::GemmBias { k, .. }
                    | OpKind::QuantGemm { k, .. } => Some(*k),
                    _ => None,
                })
            }).unwrap_or(0);
            tile_rows * k * 4 // f32
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

    let benefit = bytes_saved as i64 - spill_penalty - scratch_penalty;

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
        TraceOp::Tanh(..) => 12,
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
            body.iter().map(|op| trace_op_flops(op)).sum()
        }
        ComputePattern::NormLike { reduce, finalize, transform } => {
            // reduce and transform run per-element; finalize runs once (amortized to ~0)
            let r: u64 = reduce.iter().map(|op| trace_op_flops(op)).sum();
            let t: u64 = transform.iter().map(|op| trace_op_flops(op)).sum();
            let f: u64 = finalize.iter().map(|op| trace_op_flops(op)).sum();
            r + t + f
        }
        ComputePattern::Reduction { combine, .. } => {
            combine.iter().map(|op| trace_op_flops(op)).sum()
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
    pub fn compute(trace: &OpTrace, profile: &DeviceProfile) -> Self {
        let (max_dim, input_ptrs, output_ptrs) = parse_signature(&trace.signature);

        let (flops, bytes) = if matches!(trace.pattern, ComputePattern::Gemm) {
            // GEMM: extract M, N, K from Dim params → flops = 2*M*N*K
            let dims: Vec<u64> = trace.signature.params.iter().filter_map(|p| {
                if let ScalarParam::Dim(d) = p { Some(*d as u64) } else { None }
            }).collect();
            let (m, n, k) = if dims.len() >= 3 {
                (dims[0], dims[1], dims[2])
            } else {
                (max_dim, max_dim, max_dim)
            };
            let flops = 2 * m * n * k;
            // A(M×K) + B(K×N) + C(M×N), all f32
            let bytes = (m * k + k * n + m * n) * 4;
            (flops, bytes)
        } else {
            let per_elem = pattern_per_element_flops(&trace.pattern);
            let flops = per_elem * max_dim;
            let bytes = (input_ptrs + output_ptrs) * max_dim * 4;
            (flops, bytes)
        };

        // Convert to nanoseconds:
        //   peak_gflops_f32 is in GFLOP/s = 1e9 FLOP/s
        //   peak_bandwidth_gbs is in GB/s = 1e9 B/s
        //   compute_ns = flops / (peak_gflops * 1e9) * 1e9 = flops / peak_gflops
        //   memory_ns  = bytes / (peak_bw * 1e9) * 1e9     = bytes / peak_bw
        let compute_cycles = if profile.peak_gflops_f32 > 0.0 {
            flops as f64 / profile.peak_gflops_f32
        } else {
            0.0
        };
        let memory_cycles = if profile.peak_bandwidth_gbs > 0.0 {
            bytes as f64 / profile.peak_bandwidth_gbs
        } else {
            0.0
        };

        Cost { flops, bytes, compute_cycles, memory_cycles }
    }

    /// Compute the memory-cycle savings from eliminating intermediate traffic.
    ///
    /// Returns saved nanoseconds (as integer) from not writing + reading back
    /// `eliminated_bytes` of intermediate data.
    pub fn fusion_benefit(eliminated_bytes: usize, profile: &DeviceProfile) -> u64 {
        if profile.peak_bandwidth_gbs > 0.0 {
            (eliminated_bytes as f64 / profile.peak_bandwidth_gbs) as u64
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
/// read back — fusion eliminates both accesses (2× tensor size).
fn chain_eliminated_bytes(graph: &CompilerGraph, anchor: &CompilerOp, chain: &[&CompilerOp]) -> usize {
    if chain.is_empty() {
        return 0;
    }
    let mut eliminated = 0usize;
    // anchor's output → chain[0]: eliminated
    for &out_tid in &anchor.outputs {
        if let Some(t) = graph.tensor(out_tid) {
            if t.consumers.len() == 1 {
                eliminated += t.shape.iter().product::<usize>() * t.dtype.size_bytes() * 2;
            }
        }
    }
    // Each chain op's output (except the last) → next chain op: eliminated
    for i in 0..chain.len().saturating_sub(1) {
        for &out_tid in &chain[i].outputs {
            if let Some(t) = graph.tensor(out_tid) {
                if t.consumers.len() == 1 {
                    eliminated += t.shape.iter().product::<usize>() * t.dtype.size_bytes() * 2;
                }
            }
        }
    }
    eliminated
}

/// Run the fusion pass on a CompilerGraph.
///
/// Returns a `FusionPlan` describing which ops are grouped together.
/// The plan respects data dependencies (topological order) and only
/// fuses ops that `semantics::can_fuse()` approves.
pub fn fuse(graph: &CompilerGraph, profile: &crate::dispatch::DeviceProfile) -> FusionPlan {
    let topo = graph.topological_sort();
    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut op_to_group: std::collections::HashMap<OpId, usize> = std::collections::HashMap::new();
    // Track which ops have been claimed by a group
    let mut claimed: std::collections::HashSet<OpId> = std::collections::HashSet::new();

    // First pass: detect QKV shared input pattern
    let qkv_groups = detect_qkv_shared_input(graph, &topo);
    for qkv in &qkv_groups {
        let gid = groups.len();
        for &op_id in &qkv.ops {
            op_to_group.insert(op_id, gid);
            claimed.insert(op_id);
        }
        groups.push(qkv.clone());
    }

    // Second pass: walk topo order, greedily fuse unclaimed ops
    for &op_id in &topo {
        if claimed.contains(&op_id) {
            continue;
        }

        let op = match graph.op(op_id) {
            Some(o) => o,
            None => continue,
        };

        let sem = semantics::classify(&op.kind);

        // Try to build a fusion group starting from this op
        match sem {
            OpSemantics::Gemm => {
                // Check if preceding op is a Reduction (RmsNorm) feeding only into this GEMM
                let norm_prefix = detect_norm_into_gemm(graph, op);

                // Collect epilogue: downstream elementwise ops that only consume this GEMM's output
                let epilogue = collect_epilogue(graph, op, &claimed);

                if norm_prefix.is_some() || !epilogue.is_empty() {
                    let gid = groups.len();
                    let mut all_ops = Vec::new();

                    let mode = if let Some(norm_id) = norm_prefix {
                        if !epilogue.is_empty() {
                            // Both norm prefix and epilogue
                            FusionMode::EpilogueInjection
                        } else {
                            detect_tile_vs_compute_root(graph, op, norm_id, profile)
                        }
                    } else {
                        FusionMode::EpilogueInjection
                    };

                    if let Some(norm_id) = norm_prefix {
                        if !claimed.contains(&norm_id) {
                            all_ops.push(norm_id);
                            op_to_group.insert(norm_id, gid);
                            claimed.insert(norm_id);
                        }
                    }

                    all_ops.push(op_id);
                    op_to_group.insert(op_id, gid);
                    claimed.insert(op_id);

                    let epilogue_ids: Vec<OpId> = epilogue.iter().map(|o| o.id).collect();
                    for &eid in &epilogue_ids {
                        all_ops.push(eid);
                        op_to_group.insert(eid, gid);
                        claimed.insert(eid);
                    }

                    groups.push(FusionGroup {
                        id: gid,
                        anchor: op_id,
                        epilogue: epilogue_ids,
                        mode,
                        ops: all_ops,
                    });
                } else {
                    // Standalone GEMM
                    let gid = groups.len();
                    op_to_group.insert(op_id, gid);
                    claimed.insert(op_id);
                    groups.push(FusionGroup {
                        id: gid,
                        anchor: op_id,
                        epilogue: Vec::new(),
                        mode: FusionMode::Standalone,
                        ops: vec![op_id],
                    });
                }
            }
            OpSemantics::Elementwise => {
                // Try to chain with downstream elementwise ops
                let chain = collect_elementwise_chain(graph, op, &claimed);
                let gid = groups.len();
                let mut all_ops = vec![op_id];
                let chain_ids: Vec<OpId> = chain.iter().map(|o| o.id).collect();
                all_ops.extend_from_slice(&chain_ids);

                let mode = if chain_ids.is_empty() {
                    FusionMode::Standalone
                } else {
                    FusionMode::LoopFusion
                };

                for &oid in &all_ops {
                    op_to_group.insert(oid, gid);
                    claimed.insert(oid);
                }

                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: chain_ids,
                    mode,
                    ops: all_ops,
                });
            }
            _ => {
                // Reduction, Opaque → standalone
                let gid = groups.len();
                op_to_group.insert(op_id, gid);
                claimed.insert(op_id);
                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op_id],
                });
            }
        }
    }

    // Re-sort groups by the minimum op ID in each group (execution order)
    groups.sort_by_key(|g| g.ops.iter().map(|o| o.0).min().unwrap_or(0));
    // Reassign group IDs and rebuild op_to_group
    let mut new_op_to_group = std::collections::HashMap::new();
    for (i, g) in groups.iter_mut().enumerate() {
        g.id = i;
        for &oid in &g.ops {
            new_op_to_group.insert(oid, i);
        }
    }

    FusionPlan {
        groups,
        op_to_group: new_op_to_group,
    }
}

/// Detect QKV shared input pattern: three consecutive GEMMs reading the same tensor.
fn detect_qkv_shared_input(graph: &CompilerGraph, topo: &[OpId]) -> Vec<FusionGroup> {
    let mut result = Vec::new();

    // Find groups of GEMM ops that share the same first input tensor
    let gemm_ops: Vec<&CompilerOp> = topo
        .iter()
        .filter_map(|&id| graph.op(id))
        .filter(|op| matches!(op.kind, OpKind::Gemm { .. } | OpKind::GemmBias { .. } | OpKind::QuantGemm { .. }))
        .collect();

    // Group by first input tensor (BTreeMap for deterministic iteration order)
    let mut by_input: std::collections::BTreeMap<TensorId, Vec<&CompilerOp>> =
        std::collections::BTreeMap::new();
    for op in &gemm_ops {
        if let Some(&first_input) = op.inputs.first() {
            by_input.entry(first_input).or_default().push(op);
        }
    }

    // QKV pattern: exactly 3 GEMMs sharing the same input (Q, K, V projections)
    for (_input_tid, ops) in &by_input {
        if ops.len() == 3 {
            // Check that the shared input comes from a norm op (typical QKV pattern)
            let shared_input = ops[0].inputs[0];
            let is_from_norm = graph.tensor(shared_input).and_then(|t| t.producer).map_or(
                false,
                |prod_id| {
                    graph
                        .op(prod_id)
                        .map_or(false, |prod_op| {
                            matches!(prod_op.kind, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. })
                        })
                },
            );

            if is_from_norm {
                let all_ops: Vec<OpId> = ops.iter().map(|o| o.id).collect();
                result.push(FusionGroup {
                    id: 0, // will be reassigned
                    anchor: all_ops[0],
                    epilogue: all_ops[1..].to_vec(),
                    mode: FusionMode::QkvSharedInput,
                    ops: all_ops,
                });
            }
        }
    }

    result
}

/// Check if the op's input comes from a RmsNorm/LayerNorm with single consumer.
fn detect_norm_into_gemm(graph: &CompilerGraph, gemm_op: &CompilerOp) -> Option<OpId> {
    let input_tid = gemm_op.inputs.first()?;
    let tensor = graph.tensor(*input_tid)?;
    let producer_id = tensor.producer?;
    let producer = graph.op(producer_id)?;

    // Must be a norm op
    if !matches!(producer.kind, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }) {
        return None;
    }

    // The norm output must feed only into this GEMM (single consumer)
    // Actually for QKV pattern, norm feeds 3 GEMMs — skip if multi-consumer
    if tensor.consumers.len() != 1 {
        return None;
    }

    Some(producer_id)
}

/// Collect downstream elementwise ops that can be fused as epilogue.
///
/// Greedily follows the output chain: if the GEMM's output tensor has
/// exactly one consumer and that consumer is elementwise, absorb it.
fn collect_epilogue<'a>(
    graph: &'a CompilerGraph,
    anchor: &CompilerOp,
    claimed: &std::collections::HashSet<OpId>,
) -> Vec<&'a CompilerOp> {
    let mut epilogue = Vec::new();
    let mut current_outputs = anchor.outputs.clone();

    loop {
        if current_outputs.len() != 1 {
            break;
        }
        let out_tid = current_outputs[0];
        let tensor = match graph.tensor(out_tid) {
            Some(t) => t,
            None => break,
        };

        // Must have exactly one consumer
        if tensor.consumers.len() != 1 {
            break;
        }

        let consumer_id = tensor.consumers[0];
        if claimed.contains(&consumer_id) {
            break;
        }

        let consumer = match graph.op(consumer_id) {
            Some(o) => o,
            None => break,
        };

        // Consumer must be elementwise to be fusable as epilogue
        if semantics::classify(&consumer.kind) != OpSemantics::Elementwise {
            break;
        }

        // Verify all consumer inputs come from the fusion chain or are graph inputs
        let chain_tids: std::collections::HashSet<TensorId> =
            std::iter::once(anchor.outputs[0])
            .chain(epilogue.iter().flat_map(|op: &&CompilerOp| op.outputs.iter().copied()))
            .collect();
        let all_inputs_available = consumer.inputs.iter().all(|tid| {
            chain_tids.contains(tid) || graph.tensor(*tid).map_or(false, |t| t.producer.is_none())
        });
        if !all_inputs_available {
            break;
        }

        epilogue.push(consumer);
        current_outputs = consumer.outputs.clone();
    }

    epilogue
}

/// Collect a chain of elementwise ops starting from the given op.
fn collect_elementwise_chain<'a>(
    graph: &'a CompilerGraph,
    start: &CompilerOp,
    claimed: &std::collections::HashSet<OpId>,
) -> Vec<&'a CompilerOp> {
    let mut chain = Vec::new();
    let mut current_outputs = start.outputs.clone();

    loop {
        if current_outputs.len() != 1 {
            break;
        }
        let out_tid = current_outputs[0];
        let tensor = match graph.tensor(out_tid) {
            Some(t) => t,
            None => break,
        };

        if tensor.consumers.len() != 1 {
            break;
        }

        let consumer_id = tensor.consumers[0];
        if claimed.contains(&consumer_id) {
            break;
        }

        let consumer = match graph.op(consumer_id) {
            Some(o) => o,
            None => break,
        };

        if semantics::classify(&consumer.kind) != OpSemantics::Elementwise {
            break;
        }

        chain.push(consumer);
        current_outputs = consumer.outputs.clone();
    }

    chain
}

// ── DAG-based fusion (Phase 1 path) ─────────────────────────────────

/// Fusion pass based on SemanticDAG (Phase 1 path).
///
/// Uses OpTrace-derived `OpClass` instead of hand-maintained `OpSemantics`.
/// This is the new preferred entry point; `fuse()` is kept for backward compat.
pub fn fuse_with_dag(graph: &CompilerGraph, registry: &ScalarOpRegistry, profile: &crate::dispatch::DeviceProfile) -> FusionPlan {
    let dag = SemanticDAG::from_graph(graph, registry);
    fuse_with_dag_prebuilt(graph, &dag, profile)
}

/// Fusion pass using a pre-built SemanticDAG.
///
/// Use this when the caller already has a `SemanticDAG` to avoid redundant
/// construction (e.g., `jit_compile` builds the DAG once for both analysis
/// and fusion).
pub fn fuse_with_dag_prebuilt(graph: &CompilerGraph, dag: &SemanticDAG, profile: &crate::dispatch::DeviceProfile) -> FusionPlan {
    let topo = graph.topological_sort();
    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut op_to_group: HashMap<OpId, usize> = HashMap::new();
    let mut claimed: HashSet<OpId> = HashSet::new();

    // First pass: QKV shared input detection (reuse existing logic)
    let qkv_groups = detect_qkv_shared_input(graph, &topo);
    for qkv in &qkv_groups {
        let gid = groups.len();
        for &op_id in &qkv.ops {
            op_to_group.insert(op_id, gid);
            claimed.insert(op_id);
        }
        groups.push(qkv.clone());
    }

    // Second pass: walk topo order using OpClass from SemanticDAG
    for &op_id in &topo {
        if claimed.contains(&op_id) {
            continue;
        }

        let op = match graph.op(op_id) {
            Some(o) => o,
            None => continue,
        };

        let node = dag.node(op_id);
        let op_class = node.map(|n| n.op_class).unwrap_or(OpClass::Opaque);

        match op_class {
            OpClass::Gemm => {
                // Check norm prefix
                let norm_prefix = detect_norm_into_gemm_dag(graph, op, &dag);
                // Collect epilogue using OpClass
                let epilogue = collect_epilogue_dag(graph, op, &claimed, &dag);

                if norm_prefix.is_some() || !epilogue.is_empty() {
                    let gid = groups.len();
                    let mut all_ops = Vec::new();
                    let mode = if let Some(norm_id) = norm_prefix {
                        if !epilogue.is_empty() {
                            FusionMode::EpilogueInjection
                        } else {
                            // Decide TileLevelFusion vs ComputeRoot based on L1 capacity
                            detect_tile_vs_compute_root(graph, op, norm_id, profile)
                        }
                    } else {
                        FusionMode::EpilogueInjection
                    };

                    if let Some(norm_id) = norm_prefix {
                        if !claimed.contains(&norm_id) {
                            all_ops.push(norm_id);
                            op_to_group.insert(norm_id, gid);
                            claimed.insert(norm_id);
                        }
                    }

                    all_ops.push(op_id);
                    op_to_group.insert(op_id, gid);
                    claimed.insert(op_id);

                    let epilogue_ids: Vec<OpId> = epilogue.iter().map(|o| o.id).collect();
                    for &eid in &epilogue_ids {
                        all_ops.push(eid);
                        op_to_group.insert(eid, gid);
                        claimed.insert(eid);
                    }

                    groups.push(FusionGroup {
                        id: gid,
                        anchor: op_id,
                        epilogue: epilogue_ids,
                        mode,
                        ops: all_ops,
                    });
                } else {
                    // Standalone GEMM
                    let gid = groups.len();
                    op_to_group.insert(op_id, gid);
                    claimed.insert(op_id);
                    groups.push(FusionGroup {
                        id: gid,
                        anchor: op_id,
                        epilogue: Vec::new(),
                        mode: FusionMode::Standalone,
                        ops: vec![op_id],
                    });
                }
            }
            OpClass::ElemWise | OpClass::Injective => {
                // Try to chain with downstream elementwise/injective ops
                let chain = collect_elementwise_chain_dag(graph, op, &claimed, &dag);

                // Cost-based filter: only fuse if eliminating intermediates saves cycles
                let (accepted_chain, mode) = if chain.is_empty() {
                    (Vec::new(), FusionMode::Standalone)
                } else {
                    let eliminated = chain_eliminated_bytes(graph, op, &chain);
                    if Cost::fusion_benefit(eliminated, profile) > 0 {
                        (chain.iter().map(|o| o.id).collect(), FusionMode::LoopFusion)
                    } else {
                        (Vec::new(), FusionMode::Standalone)
                    }
                };

                let gid = groups.len();
                let mut all_ops = vec![op_id];
                all_ops.extend_from_slice(&accepted_chain);

                for &oid in &all_ops {
                    op_to_group.insert(oid, gid);
                    claimed.insert(oid);
                }

                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: accepted_chain,
                    mode,
                    ops: all_ops,
                });
            }
            OpClass::Reduction | OpClass::Opaque => {
                let gid = groups.len();
                op_to_group.insert(op_id, gid);
                claimed.insert(op_id);
                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op_id],
                });
            }
        }
    }

    // Re-sort groups by execution order
    groups.sort_by_key(|g| g.ops.iter().map(|o| o.0).min().unwrap_or(0));
    let mut new_op_to_group = HashMap::new();
    for (i, g) in groups.iter_mut().enumerate() {
        g.id = i;
        for &oid in &g.ops {
            new_op_to_group.insert(oid, i);
        }
    }

    FusionPlan {
        groups,
        op_to_group: new_op_to_group,
    }
}

/// Check norm prefix using SemanticDAG OpClass.
fn detect_norm_into_gemm_dag(
    graph: &CompilerGraph,
    gemm_op: &CompilerOp,
    dag: &SemanticDAG,
) -> Option<OpId> {
    let input_tid = gemm_op.inputs.first()?;
    let tensor = graph.tensor(*input_tid)?;
    let producer_id = tensor.producer?;

    // Use OpClass from DAG instead of manual classify
    let producer_node = dag.node(producer_id)?;
    if producer_node.op_class != OpClass::Reduction {
        return None;
    }

    // OpClass::Reduction includes Softmax — only accept actual norm ops
    let producer = graph.op(producer_id)?;
    if !matches!(producer.kind, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }) {
        return None;
    }

    // Must be single consumer
    if tensor.consumers.len() != 1 {
        return None;
    }

    Some(producer_id)
}

/// Decide between TileLevelFusion and ComputeRoot for a norm→GEMM pair.
///
/// If the norm output tensor exceeds 75% of L1, the norm must be tiled into
/// the GEMM MC loop (TileLevelFusion). Otherwise, the norm is computed fully
/// first and its result stays in L1 (ComputeRoot).
fn detect_tile_vs_compute_root(
    graph: &CompilerGraph,
    gemm_op: &CompilerOp,
    norm_id: OpId,
    profile: &crate::dispatch::DeviceProfile,
) -> FusionMode {
    let (l1, _, _) = profile.cache_sizes();
    let l1_budget = l1 * 75 / 100; // 75% of L1

    // Compute norm output size in bytes
    let norm_output_bytes = gemm_op
        .inputs
        .first()
        .and_then(|tid| graph.tensor(*tid))
        .map(|t| t.shape.iter().product::<usize>() * t.dtype.size_bytes())
        .unwrap_or(0);

    if norm_output_bytes > l1_budget {
        // Norm output doesn't fit in L1 → tile into GEMM MC loop
        let (m, n, k) = match &gemm_op.kind {
            OpKind::Gemm { m, n, k }
            | OpKind::GemmBias { m, n, k }
            | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
            _ => (0, 0, 0),
        };
        let blocking = profile.gemm_blocking(m, n, k);
        FusionMode::TileLevelFusion {
            predecessor: norm_id,
            tile_rows: blocking.mc,
        }
    } else {
        // Norm output fits in L1 → compute root (standalone norm, result stays in L1)
        FusionMode::ComputeRoot {
            predecessor: norm_id,
        }
    }
}

/// Collect epilogue using DAG OpClass.
fn collect_epilogue_dag<'a>(
    graph: &'a CompilerGraph,
    anchor: &CompilerOp,
    claimed: &HashSet<OpId>,
    dag: &SemanticDAG,
) -> Vec<&'a CompilerOp> {
    let mut epilogue = Vec::new();
    let mut current_outputs = anchor.outputs.clone();

    loop {
        if current_outputs.len() != 1 {
            break;
        }
        let out_tid = current_outputs[0];
        let tensor = match graph.tensor(out_tid) {
            Some(t) => t,
            None => break,
        };
        if tensor.consumers.len() != 1 {
            break;
        }
        let consumer_id = tensor.consumers[0];
        if claimed.contains(&consumer_id) {
            break;
        }
        let consumer = match graph.op(consumer_id) {
            Some(o) => o,
            None => break,
        };

        // Use OpClass from DAG
        let consumer_class = dag
            .node(consumer_id)
            .map(|n| n.op_class)
            .unwrap_or(OpClass::Opaque);

        if !matches!(consumer_class, OpClass::ElemWise | OpClass::Injective) {
            break;
        }

        // Verify all consumer inputs come from the fusion chain or are graph inputs
        let chain_tids: HashSet<TensorId> =
            std::iter::once(anchor.outputs[0])
            .chain(epilogue.iter().flat_map(|op: &&CompilerOp| op.outputs.iter().copied()))
            .collect();
        let all_inputs_available = consumer.inputs.iter().all(|tid| {
            chain_tids.contains(tid) || graph.tensor(*tid).map_or(false, |t| t.producer.is_none())
        });
        if !all_inputs_available {
            break;
        }

        epilogue.push(consumer);
        current_outputs = consumer.outputs.clone();
    }

    epilogue
}

/// Collect elementwise chain using DAG OpClass.
fn collect_elementwise_chain_dag<'a>(
    graph: &'a CompilerGraph,
    start: &CompilerOp,
    claimed: &HashSet<OpId>,
    dag: &SemanticDAG,
) -> Vec<&'a CompilerOp> {
    let mut chain = Vec::new();
    let mut current_outputs = start.outputs.clone();

    loop {
        if current_outputs.len() != 1 {
            break;
        }
        let out_tid = current_outputs[0];
        let tensor = match graph.tensor(out_tid) {
            Some(t) => t,
            None => break,
        };
        if tensor.consumers.len() != 1 {
            break;
        }
        let consumer_id = tensor.consumers[0];
        if claimed.contains(&consumer_id) {
            break;
        }
        let consumer = match graph.op(consumer_id) {
            Some(o) => o,
            None => break,
        };

        let consumer_class = dag
            .node(consumer_id)
            .map(|n| n.op_class)
            .unwrap_or(OpClass::Opaque);

        if !matches!(consumer_class, OpClass::ElemWise | OpClass::Injective) {
            break;
        }

        chain.push(consumer);
        current_outputs = consumer.outputs.clone();
    }

    chain
}

impl std::fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FusionPlan: {} groups", self.groups.len())?;
        for g in &self.groups {
            let ops_str: Vec<String> = g.ops.iter().map(|o| format!("{}", o.0)).collect();
            writeln!(
                f,
                "  [{}] {:?} anchor=Op({}) ops=[{}]",
                g.id,
                g.mode,
                g.anchor.0,
                ops_str.join(", ")
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::CompilerGraph;
    use crate::compiler::ir::LayerIR;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::{DType, ModelConfig};

    #[test]
    fn test_fuse_decoder_layer() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let plan = fuse(&graph, &profile);

        eprintln!("{plan}");

        // Every op should be in exactly one group
        for op in &graph.ops {
            assert!(
                plan.op_to_group.contains_key(&op.id),
                "Op {} not in any group",
                op.id.0
            );
        }

        // Should have QKV shared input group
        let has_qkv = plan
            .groups
            .iter()
            .any(|g| g.mode == FusionMode::QkvSharedInput);
        assert!(has_qkv, "Expected QKV shared input fusion");

        // Should have fewer groups than ops (some ops fused)
        assert!(
            plan.num_groups() < graph.num_ops(),
            "Expected fusion to reduce group count: {} groups vs {} ops",
            plan.num_groups(),
            graph.num_ops()
        );
    }

    #[test]
    fn test_fuse_gemm_epilogue() {
        // Build a minimal graph: GEMM → SiLU
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 4096], dt);
        let silu_out = g.add_tensor("silu_out", vec![1, 4096], dt);

        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        eprintln!("{plan}");

        // Should fuse into one group
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
        assert_eq!(plan.groups[0].ops.len(), 2);
    }

    #[test]
    fn test_standalone_reduction() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);

        g.add_op(OpKind::Softmax, vec![a], vec![b], "softmax");

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone);
    }

    #[test]
    fn test_elementwise_chain() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let c = g.add_tensor("c", vec![1, 4096], dt);
        let d = g.add_tensor("d", vec![1, 4096], dt);

        g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");
        g.add_op(OpKind::Silu, vec![c], vec![d], "silu3");

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::LoopFusion);
        assert_eq!(plan.groups[0].ops.len(), 3);
    }

    #[test]
    fn test_no_fuse_across_reduction() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let c = g.add_tensor("c", vec![1, 4096], dt);

        g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        g.add_op(OpKind::Softmax, vec![b], vec![c], "softmax");

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        // Silu and Softmax should be in separate groups
        assert_eq!(plan.num_groups(), 2);
    }

    #[test]
    fn test_gemma_geglu_fusion() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let plan = fuse(&graph, &profile);

        eprintln!("{plan}");

        // All ops accounted for
        for op in &graph.ops {
            assert!(plan.op_to_group.contains_key(&op.id));
        }
    }

    // ── DAG-based fusion tests ──────────────────────────────────────

    #[test]
    fn test_fuse_with_dag_decoder_layer() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let plan = fuse_with_dag(&graph, &registry, &profile);

        eprintln!("DAG-based fusion:\n{plan}");

        // Every op should be in exactly one group
        for op in &graph.ops {
            assert!(
                plan.op_to_group.contains_key(&op.id),
                "Op {} not in any group",
                op.id.0
            );
        }

        // Should have QKV shared input group
        let has_qkv = plan
            .groups
            .iter()
            .any(|g| g.mode == FusionMode::QkvSharedInput);
        assert!(has_qkv, "Expected QKV shared input fusion");

        // Should have fewer groups than ops
        assert!(plan.num_groups() < graph.num_ops());
    }

    #[test]
    fn test_fuse_with_dag_matches_old_fuse() {
        // The new DAG-based fusion should produce similar results to the old path
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let old_plan = fuse(&graph, &profile);
        let new_plan = fuse_with_dag(&graph, &registry, &profile);

        // Same number of groups (or very close — Injective handling may differ slightly)
        let diff = (old_plan.num_groups() as i32 - new_plan.num_groups() as i32).abs();
        assert!(
            diff <= 2,
            "Old plan has {} groups, new plan has {} groups (diff={})",
            old_plan.num_groups(),
            new_plan.num_groups(),
            diff
        );

        eprintln!(
            "Old: {} groups, New: {} groups",
            old_plan.num_groups(),
            new_plan.num_groups()
        );
    }

    #[test]
    fn test_fuse_with_dag_gemm_epilogue() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 4096], dt);
        let silu_out = g.add_tensor("silu_out", vec![1, 4096], dt);

        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let plan = fuse_with_dag(&g, &registry, &profile);

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
    }

    #[test]
    fn test_fuse_with_dag_injective_chain() {
        // RoPE (Injective) should be fusable in elementwise chains
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 128], dt);
        let cos = g.add_tensor("cos", vec![64], dt);
        let rope_out = g.add_tensor("rope_out", vec![1, 128], dt);
        let silu_out = g.add_tensor("silu_out", vec![1, 128], dt);

        g.add_op(
            OpKind::RoPE { head_dim: 128, theta: 10000.0 },
            vec![a, cos],
            vec![rope_out],
            "rope",
        );
        g.add_op(OpKind::Silu, vec![rope_out], vec![silu_out], "silu");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let plan = fuse_with_dag(&g, &registry, &profile);

        // RoPE + Silu should fuse (both fusable in new path: Injective + ElemWise)
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::LoopFusion);
    }

    // ── QuantGemm fusion tests ──────────────────────────────────────

    #[test]
    fn test_fuse_quant_gemm_epilogue() {
        // QuantGemm + SiLU should fuse as GemmEpilogue
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let w = g.add_tensor("w_q4", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 4096], dt);
        let silu_out = g.add_tensor("silu_out", vec![1, 4096], dt);

        g.add_op(
            OpKind::QuantGemm { m: 1, n: 4096, k: 4096, block_size: 32, bits: 4 },
            vec![a, w],
            vec![gemm_out],
            "qgemm",
        );
        g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
    }

    #[test]
    fn test_fuse_standalone_quant_gemm() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let w = g.add_tensor("w_q4", vec![4096, 4096], dt);
        let out = g.add_tensor("out", vec![1, 4096], dt);

        g.add_op(
            OpKind::QuantGemm { m: 1, n: 4096, k: 4096, block_size: 32, bits: 4 },
            vec![a, w],
            vec![out],
            "qgemm",
        );

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone);
    }

    #[test]
    fn test_fuse_dequantize_standalone() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a_q4", vec![4096], dt);
        let b = g.add_tensor("b_f32", vec![4096], dt);

        g.add_op(
            OpKind::Dequantize { num_elements: 4096, block_size: 32, bits: 4 },
            vec![a],
            vec![b],
            "dequant",
        );

        let profile = DeviceProfile::detect();
        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
    }

    // ── M5: Softmax must not be misidentified as norm prefix ────────

    #[test]
    fn test_softmax_not_norm_prefix_dag() {
        // Softmax → GEMM: Softmax is Reduction but NOT a norm op.
        // detect_norm_into_gemm_dag must reject it.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let softmax_out = g.add_tensor("softmax_out", vec![1, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 4096], dt);

        g.add_op(OpKind::Softmax, vec![a], vec![softmax_out], "softmax");
        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![softmax_out, w],
            vec![gemm_out],
            "gemm",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let plan = fuse_with_dag(&g, &registry, &profile);

        // Softmax and GEMM must be in separate groups — no NormIntoGemm fusion
        assert_eq!(plan.num_groups(), 2, "Softmax should not fuse as norm prefix");
        for group in &plan.groups {
            assert_ne!(
                group.mode,
                FusionMode::NormIntoGemm,
                "Softmax must not produce NormIntoGemm pattern"
            );
        }
    }

    // ── M6: Multi-input consumer with external input must not fuse ──

    #[test]
    fn test_multi_input_consumer_not_fused_as_epilogue() {
        // GEMM → SwiGlu(gemm_out, other_out): SwiGlu has two inputs.
        // If only gemm_out comes from the GEMM chain and other_out is from
        // a different op, SwiGlu must NOT be fused as epilogue.
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 4096], dt);
        let other_out = g.add_tensor("other_out", vec![1, 4096], dt);
        let swiglu_out = g.add_tensor("swiglu_out", vec![1, 4096], dt);

        // Another GEMM produces other_out (not in the fusion chain of the first GEMM)
        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![a, w],
            vec![other_out],
            "gemm_other",
        );
        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm_main",
        );
        // SwiGlu takes gemm_out (from main chain) + other_out (from different op)
        g.add_op(
            OpKind::SwiGlu,
            vec![gemm_out, other_out],
            vec![swiglu_out],
            "swiglu",
        );

        let plan = fuse(&g, &DeviceProfile::detect());
        // SwiGlu must NOT be fused into gemm_main's epilogue
        let gemm_main_group = plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "gemm_main")
            })
        }).expect("gemm_main should have a group");
        assert!(
            !gemm_main_group.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "swiglu")
            }),
            "SwiGlu with external input must not be fused into GEMM epilogue"
        );
    }

    #[test]
    fn test_multi_input_consumer_not_fused_dag() {
        // Same test but for the DAG-based path
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 4096], dt);
        let other_out = g.add_tensor("other_out", vec![1, 4096], dt);
        let swiglu_out = g.add_tensor("swiglu_out", vec![1, 4096], dt);

        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![a, w],
            vec![other_out],
            "gemm_other",
        );
        g.add_op(
            OpKind::Gemm { m: 1, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm_main",
        );
        g.add_op(
            OpKind::SwiGlu,
            vec![gemm_out, other_out],
            vec![swiglu_out],
            "swiglu",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let plan = fuse_with_dag(&g, &registry, &profile);

        let gemm_main_group = plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "gemm_main")
            })
        }).expect("gemm_main should have a group");
        assert!(
            !gemm_main_group.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "swiglu")
            }),
            "SwiGlu with external input must not be fused into GEMM epilogue (DAG path)"
        );
    }

    // ── WI-22: Fusion cost model tests ────────────────────────────────

    /// Helper: find the fusion group containing the op with the given label.
    fn find_group_by_label<'a>(
        plan: &'a FusionPlan,
        graph: &CompilerGraph,
        label: &str,
    ) -> Option<&'a FusionGroup> {
        plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                graph.op(oid).map_or(false, |o| o.label == label)
            })
        })
    }

    /// When norm output > 75% L1, TileLevelFusion is chosen;
    /// when <= 75% L1, ComputeRoot is chosen.
    #[test]
    fn test_tile_vs_compute_root_threshold() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let l1_budget = l1 * 75 / 100;
        let dt = DType::F32;

        // ── Large norm output: exceeds 75% L1 → TileLevelFusion ──
        // Pick K so that [1, K] * 4 bytes > l1_budget
        let k_large = (l1_budget / 4) + 1;
        {
            let mut g = CompilerGraph::new();
            let x = g.add_tensor("x", vec![1, k_large], dt);
            let norm_out = g.add_tensor("norm_out", vec![1, k_large], dt);
            let w = g.add_tensor("w", vec![k_large, k_large], dt);
            let gemm_out = g.add_tensor("gemm_out", vec![1, k_large], dt);

            g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
            g.add_op(
                OpKind::Gemm { m: 1, n: k_large, k: k_large },
                vec![norm_out, w],
                vec![gemm_out],
                "gemm",
            );

            let plan = fuse(&g, &profile);
            let gemm_group = find_group_by_label(&plan, &g, "gemm")
                .expect("GEMM should have a fusion group");
            assert!(
                matches!(gemm_group.mode, FusionMode::TileLevelFusion { .. }),
                "Expected TileLevelFusion for norm output ({} B) > 75% L1 ({} B), got {:?}",
                k_large * 4, l1_budget, gemm_group.mode,
            );
        }

        // ── Small norm output: fits in 75% L1 → ComputeRoot ──
        // Pick K so that [1, K] * 4 bytes <= l1_budget, with m=1
        let k_small = l1_budget / 4;
        {
            let mut g = CompilerGraph::new();
            let x = g.add_tensor("x", vec![1, k_small], dt);
            let norm_out = g.add_tensor("norm_out", vec![1, k_small], dt);
            // Use n=4096 so m*n*k >> 4096 (avoids small-matrix direct path)
            let n_small = 4096;
            let w = g.add_tensor("w", vec![k_small, n_small], dt);
            let gemm_out = g.add_tensor("gemm_out", vec![1, n_small], dt);

            let norm_bytes = k_small * 4;
            assert!(
                norm_bytes <= l1_budget,
                "Test setup error: norm output {norm_bytes}B should be <= l1_budget {l1_budget}B"
            );

            g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
            g.add_op(
                OpKind::Gemm { m: 1, n: n_small, k: k_small },
                vec![norm_out, w],
                vec![gemm_out],
                "gemm",
            );

            let plan = fuse(&g, &profile);
            let gemm_group = find_group_by_label(&plan, &g, "gemm")
                .expect("GEMM should have a fusion group");
            assert!(
                matches!(gemm_group.mode, FusionMode::ComputeRoot { .. }),
                "Expected ComputeRoot for norm output ({} B) <= 75% L1 ({} B), got {:?}",
                norm_bytes, l1_budget, gemm_group.mode,
            );
        }
    }

    /// GEMM followed by elementwise gets EpilogueInjection mode.
    #[test]
    fn test_epilogue_injection_decision() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;

        // GEMM → Gelu (different activation than existing test_fuse_gemm_epilogue)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![4, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![4, 4096], dt);
        let gelu_out = g.add_tensor("gelu_out", vec![4, 4096], dt);

        g.add_op(
            OpKind::Gemm { m: 4, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Gelu, vec![gemm_out], vec![gelu_out], "gelu");

        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(
            plan.groups[0].mode,
            FusionMode::EpilogueInjection,
            "GEMM + elementwise should produce EpilogueInjection"
        );
        assert_eq!(plan.groups[0].epilogue.len(), 1, "epilogue should contain the Gelu op");
    }

    /// Consecutive elementwise ops get LoopFusion mode.
    #[test]
    fn test_loop_fusion_decision() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;

        // Add → Mul chain (different ops than existing test_elementwise_chain)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let add_out = g.add_tensor("add_out", vec![1, 4096], dt);
        let mul_out = g.add_tensor("mul_out", vec![1, 4096], dt);

        g.add_op(OpKind::Add, vec![a, b], vec![add_out], "add");
        g.add_op(OpKind::Silu, vec![add_out], vec![mul_out], "silu");

        let plan = fuse(&g, &profile);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(
            plan.groups[0].mode,
            FusionMode::LoopFusion,
            "Consecutive elementwise ops should produce LoopFusion"
        );
        assert_eq!(plan.groups[0].ops.len(), 2);
    }

    /// 3 GEMMs sharing the same norm output get QkvSharedInput mode.
    #[test]
    fn test_qkv_shared_input_detection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let dim = 4096;

        let mut g = CompilerGraph::new();
        let x = g.add_tensor("x", vec![1, dim], dt);
        let norm_out = g.add_tensor("norm_out", vec![1, dim], dt);
        let wq = g.add_tensor("wq", vec![dim, dim], dt);
        let wk = g.add_tensor("wk", vec![dim, dim], dt);
        let wv = g.add_tensor("wv", vec![dim, dim], dt);
        let q_out = g.add_tensor("q_out", vec![1, dim], dt);
        let k_out = g.add_tensor("k_out", vec![1, dim], dt);
        let v_out = g.add_tensor("v_out", vec![1, dim], dt);

        // RmsNorm produces norm_out
        g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        // 3 GEMMs all read norm_out as first input
        g.add_op(
            OpKind::Gemm { m: 1, n: dim, k: dim },
            vec![norm_out, wq],
            vec![q_out],
            "gemm_q",
        );
        g.add_op(
            OpKind::Gemm { m: 1, n: dim, k: dim },
            vec![norm_out, wk],
            vec![k_out],
            "gemm_k",
        );
        g.add_op(
            OpKind::Gemm { m: 1, n: dim, k: dim },
            vec![norm_out, wv],
            vec![v_out],
            "gemm_v",
        );

        let plan = fuse(&g, &profile);

        let qkv_group = plan.groups.iter().find(|grp| grp.mode == FusionMode::QkvSharedInput);
        assert!(
            qkv_group.is_some(),
            "Expected QkvSharedInput group for 3 GEMMs sharing norm output"
        );
        let qkv = qkv_group.unwrap();
        assert_eq!(qkv.ops.len(), 3, "QKV group should contain exactly 3 GEMM ops");
    }

    /// RmsNorm → GEMM (single consumer, no epilogue) gets a norm-aware
    /// fusion mode (ComputeRoot or TileLevelFusion), not Standalone.
    #[test]
    fn test_norm_into_gemm_decision() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;

        let mut g = CompilerGraph::new();
        let x = g.add_tensor("x", vec![1, 512], dt);
        let norm_out = g.add_tensor("norm_out", vec![1, 512], dt);
        let w = g.add_tensor("w", vec![512, 512], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![1, 512], dt);

        g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        g.add_op(
            OpKind::Gemm { m: 1, n: 512, k: 512 },
            vec![norm_out, w],
            vec![gemm_out],
            "gemm",
        );

        let plan = fuse(&g, &profile);

        // The GEMM group must carry a norm-aware fusion mode
        let gemm_group = find_group_by_label(&plan, &g, "gemm")
            .expect("GEMM should have a fusion group");
        let mode = &gemm_group.mode;
        assert!(
            matches!(mode, FusionMode::ComputeRoot { .. } | FusionMode::TileLevelFusion { .. }),
            "RmsNorm → GEMM should produce ComputeRoot or TileLevelFusion, got {:?}",
            mode,
        );
        // The mode must reference the norm op as predecessor
        match mode {
            FusionMode::ComputeRoot { predecessor } |
            FusionMode::TileLevelFusion { predecessor, .. } => {
                let pred_op = g.op(*predecessor).expect("predecessor op should exist");
                assert!(
                    matches!(pred_op.kind, OpKind::RmsNorm { .. }),
                    "predecessor should be RmsNorm, got {:?}", pred_op.kind,
                );
            }
            _ => unreachable!(),
        }
    }

    /// Verify KC * (MR + NR) * 4 <= L1 * 0.85 for various GEMM sizes.
    #[test]
    fn test_gemm_blocking_l1_constraint() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let (mr, nr) = profile.microkernel_mr_nr();

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k);
            let micropanel_bytes = b.kc * (mr + nr) * 4;
            assert!(
                micropanel_bytes <= l1 * 85 / 100,
                "L1 constraint violated for m={m} n={n} k={k}: \
                 KC({}) * (MR({mr}) + NR({nr})) * 4 = {micropanel_bytes}B > 85% of L1 ({}B)",
                b.kc, l1 * 85 / 100,
            );
        }
    }

    /// Verify MC * KC * 4 <= L2 * 0.85 for various GEMM sizes.
    #[test]
    fn test_gemm_blocking_l2_constraint() {
        let profile = DeviceProfile::detect();
        let (_, l2, _) = profile.cache_sizes();

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k);
            let a_panel_bytes = b.mc * b.kc * 4;
            assert!(
                a_panel_bytes <= l2 * 85 / 100,
                "L2 constraint violated for m={m} n={n} k={k}: \
                 MC({}) * KC({}) * 4 = {a_panel_bytes}B > 85% of L2 ({}B)",
                b.mc, b.kc, l2 * 85 / 100,
            );
        }
    }

    /// Verify KC * NC * 4 <= L3 * 0.65 for various GEMM sizes.
    #[test]
    fn test_gemm_blocking_l3_constraint() {
        let profile = DeviceProfile::detect();
        let (_, _, l3) = profile.cache_sizes();

        // Only meaningful when L3 >= 1MB (otherwise fallback to L2 budget)
        if l3 < 1024 * 1024 {
            eprintln!("Skipping L3 constraint test: L3 = {l3}B < 1MB");
            return;
        }

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k);
            let b_panel_bytes = b.kc * b.nc * 4;
            assert!(
                b_panel_bytes <= l3 * 65 / 100,
                "L3 constraint violated for m={m} n={n} k={k}: \
                 KC({}) * NC({}) * 4 = {b_panel_bytes}B > 65% of L3 ({}B)",
                b.kc, b.nc, l3 * 65 / 100,
            );
        }
    }

    // ── Roofline Cost model tests ──────────────────────────────────────

    #[test]
    fn test_cost_compute_bound() {
        // GEMM is compute-bound: 2*M*N*K FLOPs with relatively small memory footprint
        use crate::compiler::trace::{OpTrace, ComputePattern, ScalarFnSignature, ScalarParam};
        use crate::compiler::graph::OpKind;

        let profile = DeviceProfile::detect();
        let trace = OpTrace {
            op_kind: OpKind::Gemm { m: 1024, n: 1024, k: 1024 },
            pattern: ComputePattern::Gemm,
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![
                    ScalarParam::InputPtr,   // A
                    ScalarParam::InputPtr,   // B
                    ScalarParam::OutputPtr,  // C
                    ScalarParam::Dim(1024),  // M
                    ScalarParam::Dim(1024),  // N
                    ScalarParam::Dim(1024),  // K
                ],
            },
        };

        let cost = Cost::compute(&trace, &profile);
        assert!(cost.is_compute_bound(),
            "GEMM 1024x1024x1024 should be compute-bound: compute_cycles={:.1} > memory_cycles={:.1}",
            cost.compute_cycles, cost.memory_cycles);
        // 2 * 1024^3 = 2,147,483,648 FLOPs
        assert_eq!(cost.flops, 2 * 1024 * 1024 * 1024);
        // A(1024*1024) + B(1024*1024) + C(1024*1024) = 3*1024*1024 f32 = 12 MiB
        assert_eq!(cost.bytes, 3 * 1024 * 1024 * 4);
        eprintln!("GEMM cost: flops={}, bytes={}, compute_ns={:.1}, memory_ns={:.1}, compute_bound={}",
            cost.flops, cost.bytes, cost.compute_cycles, cost.memory_cycles, cost.is_compute_bound());
    }

    #[test]
    fn test_cost_memory_bound() {
        // SiLU is memory-bound: ~13 FLOPs per element, 2 memory accesses (read + write)
        use crate::compiler::trace::{OpTrace, ComputePattern, ScalarFnSignature, ScalarParam, TraceOp};
        use crate::compiler::graph::OpKind;

        let profile = DeviceProfile::detect();
        let body = vec![
            TraceOp::Input(0),   // [0] v
            TraceOp::Neg(0),     // [1] -v
            TraceOp::Exp(1),     // [2] exp(-v)
            TraceOp::Const(1.0), // [3] 1.0
            TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
            TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
        ];
        let trace = OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body },
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![
                    ScalarParam::InputPtr,
                    ScalarParam::OutputPtr,
                    ScalarParam::Dim(4096),
                ],
            },
        };

        let cost = Cost::compute(&trace, &profile);
        assert!(!cost.is_compute_bound(),
            "SiLU on 4096 elements should be memory-bound: compute_cycles={:.1} <= memory_cycles={:.1}",
            cost.compute_cycles, cost.memory_cycles);
        // Per-element: Neg(1) + Exp(10) + Add(1) + Div(1) = 13 FLOPs
        assert_eq!(cost.flops, 13 * 4096);
        // 1 input + 1 output = 2 * 4096 * 4 = 32768 bytes
        assert_eq!(cost.bytes, 2 * 4096 * 4);
        eprintln!("SiLU cost: flops={}, bytes={}, compute_ns={:.1}, memory_ns={:.1}, compute_bound={}",
            cost.flops, cost.bytes, cost.compute_cycles, cost.memory_cycles, cost.is_compute_bound());
    }

    #[test]
    fn test_fusion_benefit() {
        let profile = DeviceProfile::detect();

        // Fusing two elementwise ops on a 4096-element f32 tensor eliminates
        // one intermediate write + read = 2 * 4096 * 4 = 32768 bytes
        let eliminated = 2 * 4096 * 4;
        let benefit = Cost::fusion_benefit(eliminated, &profile);
        assert!(benefit > 0,
            "Eliminating {} bytes should yield positive benefit, got {}",
            eliminated, benefit);

        // Larger tensors yield proportionally larger benefit
        let eliminated_large = 2 * 4096 * 4096 * 4;
        let benefit_large = Cost::fusion_benefit(eliminated_large, &profile);
        assert!(benefit_large > benefit,
            "Larger elimination ({} bytes) should yield larger benefit ({}) than smaller ({} bytes, {})",
            eliminated_large, benefit_large, eliminated, benefit);

        // Zero eliminated bytes → zero benefit
        assert_eq!(Cost::fusion_benefit(0, &profile), 0);

        eprintln!("Fusion benefit: {}B → {} ns, {}B → {} ns",
            eliminated, benefit, eliminated_large, benefit_large);
    }

    #[test]
    fn test_cost_chain_eliminated_bytes() {
        // Build: SiLU → SiLU → SiLU chain, verify eliminated bytes
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let c = g.add_tensor("c", vec![1, 4096], dt);
        let d = g.add_tensor("d", vec![1, 4096], dt);

        g.add_op(OpKind::Silu, vec![a], vec![b], "silu1");
        g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");
        g.add_op(OpKind::Silu, vec![c], vec![d], "silu3");

        let anchor = g.op(OpId(0)).unwrap();
        let chain: Vec<&CompilerOp> = vec![
            g.op(OpId(1)).unwrap(),
            g.op(OpId(2)).unwrap(),
        ];

        let eliminated = chain_eliminated_bytes(&g, anchor, &chain);
        // b is intermediate (silu1→silu2): 4096 * 4 * 2 = 32768
        // c is intermediate (silu2→silu3): 4096 * 4 * 2 = 32768
        // Total = 65536
        assert_eq!(eliminated, 2 * 4096 * 4 * 2,
            "Expected 2 intermediates eliminated, got {} bytes", eliminated);
    }

    #[test]
    fn test_cost_dag_fusion_uses_cost_filter() {
        // Verify that the DAG-based fusion path applies cost filtering
        // (a valid chain should still fuse because benefit > 0)
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let c = g.add_tensor("c", vec![1, 4096], dt);

        g.add_op(OpKind::Silu, vec![a], vec![b], "silu1");
        g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let plan = fuse_with_dag(&g, &registry, &profile);

        // Chain should still fuse (benefit > 0 for 4096-element intermediate)
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::LoopFusion);
    }
}
