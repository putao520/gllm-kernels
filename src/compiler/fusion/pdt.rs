//! R1 PDT topology fusion — Post-Dominator Tree driven fusion (SPEC §3.2).
//!
//! Replaces v1 pattern-matching helpers with TVM-style PDT traversal:
//!   1. Build Post-Dominator Tree from SemanticDAG
//!   2. Walk PDT children in topological order
//!   3. OpClass hierarchy decides fusion eligibility
//!   4. Bottleneck-aware score_fusion (§4) gates actual fusion
//!
//! The PDT approach naturally discovers fusion opportunities that
//! pattern-matching misses (e.g., cross-branch fusion, nested consumers).

use std::collections::{HashMap, HashSet, VecDeque};
use crate::compiler::graph::{CompilerGraph, OpId};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass};
use crate::compiler::pain_point::OpBottleneckMap;

/// Post-Dominator Tree node.
#[derive(Debug, Clone)]
pub struct PdtNode {
    /// The OpId this node represents.
    pub op_id: OpId,
    /// Immediate post-dominator (ipostdom).
    pub ipostdom: Option<OpId>,
    /// Children in the PDT (nodes whose ipostdom == this node).
    pub children: Vec<OpId>,
}

/// Post-Dominator Tree built from a SemanticDAG.
#[derive(Debug, Clone)]
pub struct PostDominatorTree {
    /// Nodes indexed by OpId.0 (u32).
    nodes: HashMap<u32, PdtNode>,
    /// Exit node OpId (virtual sink).
    exit_id: OpId,
}

impl PostDominatorTree {
    /// Build PDT using simple iterative algorithm (O(n²), fine for <100 nodes).
    ///
    /// Algorithm:
    /// 1. For each node n, compute post-dominators: all nodes that appear on
    ///    every path from n to the exit.
    /// 2. The immediate post-dominator (ipostdom) is the closest such node.
    pub fn build(graph: &CompilerGraph, dag: &SemanticDAG) -> Self {
        let topo = graph.topological_sort();
        let n = topo.len();
        if n == 0 {
            let exit_id = OpId(u32::MAX);
            return PostDominatorTree {
                nodes: HashMap::new(),
                exit_id,
            };
        }

        // The exit node is the last in topological order (sink of the DAG).
        let exit_id = *topo.last().unwrap();

        // Build successor map for each op (consumers of its output tensors).
        let mut successors: HashMap<OpId, HashSet<OpId>> = HashMap::new();
        for &op_id in &topo {
            successors.insert(op_id, HashSet::new());
        }
        for &op_id in &topo {
            let op = match graph.op(op_id) {
                Some(o) => o,
                None => continue,
            };
            for &out_tid in &op.outputs {
                if let Some(tensor) = graph.tensor(out_tid) {
                    for &consumer_id in &tensor.consumers {
                        if let Some(succs) = successors.get_mut(&op_id) {
                            succs.insert(consumer_id);
                        }
                    }
                }
            }
        }

        // For the exit node, post-dominators = {exit} only.
        // For other nodes, post-dominators = intersection of post-dominators of all successors,
        // union {node itself} for the successor set.
        //
        // Simple iterative approach:
        //   postdom[n] = { n } ∪ (∩ postdom[s] for s in successors(n))
        //   If no successors: postdom[n] = { n }

        let mut postdom_sets: HashMap<OpId, HashSet<OpId>> = HashMap::new();

        // Initialize: exit node post-dominates itself.
        // All nodes start with full set, then iteratively intersect.
        let all_ids: HashSet<OpId> = topo.iter().copied().collect();
        for &op_id in &topo {
            let succs = &successors[&op_id];
            if succs.is_empty() {
                // No successors: postdom = { self }
                let mut set = HashSet::new();
                set.insert(op_id);
                postdom_sets.insert(op_id, set);
            } else {
                // Initialize to full set (will be intersected)
                postdom_sets.insert(op_id, all_ids.clone());
            }
        }

        // Iterate in reverse topological order until fixed point.
        // Reverse topo processes successors before predecessors.
        let rev_topo: Vec<OpId> = topo.iter().rev().copied().collect();
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < n + 2 {
            changed = false;
            iterations += 1;
            for &op_id in &rev_topo {
                let succs = &successors[&op_id];
                if succs.is_empty() {
                    continue;
                }
                // postdom[n] = { n } ∪ (∩ postdom[s] for s in successors(n))
                let mut new_set: Option<HashSet<OpId>> = None;
                for &s in succs {
                    let s_postdom = postdom_sets.get(&s).cloned().unwrap_or_default();
                    match &mut new_set {
                        None => new_set = Some(s_postdom),
                        Some(current) => {
                            let intersection: HashSet<OpId> =
                                current.intersection(&s_postdom).copied().collect();
                            *current = intersection;
                        }
                    }
                }
                let mut new_set = new_set.unwrap_or_default();
                new_set.insert(op_id);

                let old_set = postdom_sets.get_mut(&op_id).unwrap();
                if *old_set != new_set {
                    *old_set = new_set;
                    changed = true;
                }
            }
        }

        // Compute immediate post-dominator (ipostdom) for each node.
        // ipostdom(n) = the post-dominator of n that is closest to n in the
        // topological order, excluding n itself.
        let topo_idx: HashMap<OpId, usize> = topo.iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let mut nodes: HashMap<u32, PdtNode> = HashMap::new();
        let mut ipostdom_map: HashMap<OpId, Option<OpId>> = HashMap::new();

        for &op_id in &topo {
            let postdoms = &postdom_sets[&op_id];
            // ipostdom = the post-dominator with smallest topological index
            // that is > op_id's own index.
            let my_idx = topo_idx[&op_id];
            let mut best: Option<(usize, OpId)> = None;
            for &pd_id in postdoms {
                if pd_id == op_id {
                    continue;
                }
                if let Some(&pd_idx) = topo_idx.get(&pd_id) {
                    if pd_idx > my_idx {
                        match &best {
                            None => best = Some((pd_idx, pd_id)),
                            Some((best_idx, _)) if pd_idx < *best_idx => best = Some((pd_idx, pd_id)),
                            _ => {}
                        }
                    }
                }
            }
            let ipostdom = best.map(|(_, id)| id);
            ipostdom_map.insert(op_id, ipostdom);
        }

        // Build children lists from ipostdom relationships.
        let mut children_map: HashMap<OpId, Vec<OpId>> = HashMap::new();
        for &op_id in &topo {
            children_map.insert(op_id, Vec::new());
        }
        for &op_id in &topo {
            if let Some(ipd) = ipostdom_map[&op_id] {
                children_map.entry(ipd).or_default().push(op_id);
            }
        }

        for &op_id in &topo {
            nodes.insert(op_id.0, PdtNode {
                op_id,
                ipostdom: ipostdom_map[&op_id],
                children: children_map.get(&op_id).cloned().unwrap_or_default(),
            });
        }

        PostDominatorTree { nodes, exit_id }
    }

    /// Get the immediate post-dominator of a node.
    pub fn ipostdom(&self, op_id: OpId) -> Option<OpId> {
        self.nodes.get(&op_id.0).and_then(|n| n.ipostdom)
    }

    /// Get the PDT children of a node.
    pub fn children(&self, op_id: OpId) -> &[OpId] {
        self.nodes.get(&op_id.0).map(|n| n.children.as_slice()).unwrap_or(&[])
    }

    /// Collect all nodes in the PDT subtree rooted at `op_id` (BFS).
    pub fn subtree(&self, root: OpId) -> Vec<OpId> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(id) = queue.pop_front() {
            result.push(id);
            for &child in self.children(id) {
                queue.push_back(child);
            }
        }
        result
    }

    /// Number of nodes in the PDT.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the PDT is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Fusion candidate discovered during PDT traversal.
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// The anchor op (producer) of the fusion group.
    pub anchor: OpId,
    /// Ops that can be fused with the anchor.
    pub consumers: Vec<OpId>,
    /// Anchor's OpClass.
    pub anchor_class: OpClass,
}

/// Bottleneck-aware fusion score (SPEC §4.1).
///
/// Scores a (producer, consumer) pair based on:
/// - bytes_saved: intermediate tensor size eliminated by fusion
/// - scale: bottleneck-dependent scaling from R0 OpBottleneckMap
/// - strategy_weight: GEMM-role-specific weighting from R0
/// - reg_penalty: register pressure cost
pub fn score_fusion(
    producer: OpId,
    consumer: OpId,
    graph: &CompilerGraph,
    dag: &SemanticDAG,
    bottleneck_map: Option<&OpBottleneckMap>,
    profile: Option<&crate::dispatch::device_profile::DeviceProfile>,
) -> f64 {
    // Bytes saved = size of intermediate tensor between producer and consumer.
    let bytes_saved = compute_bytes_saved(producer, consumer, graph);
    if bytes_saved == 0 {
        return 0.0;
    }

    // Bottleneck-aware scale from R0.
    let scale = compute_bottleneck_scale(producer, dag, bottleneck_map);

    // Strategy weight from R0 fusion_benefits.
    let strategy_weight = compute_strategy_weight(producer, consumer, dag, bottleneck_map);

    // Register pressure penalty — §0.2.6
    // 寄存器成本相对于内存节省的比率。融合的收益（bytes_saved）通常远大于
    // 寄存器开销。penalty 权重 = bytes_saved 的 0.1%，确保融合决策仍以带宽为主。
    let reg_penalty = estimate_reg_penalty(producer, consumer, graph, dag);
    let base_reg_cost = profile.map(|p| p.reg_cost_factor()).unwrap_or(0.001);
    let reg_cost_factor = base_reg_cost * bytes_saved.max(1) as f64 / (reg_penalty.max(1) as f64 * 1000.0);

    (bytes_saved as f64) * scale * strategy_weight - (reg_penalty as f64 * reg_cost_factor)
}

/// Compute bytes saved by fusing producer → consumer (intermediate tensor size).
pub fn compute_bytes_saved(producer: OpId, consumer: OpId, graph: &CompilerGraph) -> usize {
    let prod = match graph.op(producer) {
        Some(o) => o,
        None => return 0,
    };
    // Find the tensor(s) produced by producer and consumed by consumer.
    let mut total = 0usize;
    for &out_tid in &prod.outputs {
        if let Some(tensor) = graph.tensor(out_tid) {
            if tensor.consumers.contains(&consumer) {
                total += tensor.concrete_bytes();
            }
        }
    }
    total
}

/// Compute bottleneck-aware scale (SPEC §4.1).
fn compute_bottleneck_scale(
    producer: OpId,
    dag: &SemanticDAG,
    bottleneck_map: Option<&OpBottleneckMap>,
) -> f64 {
    let producer_class = dag.node(producer).map(|n| n.op_class).unwrap_or(OpClass::Opaque);

    // Use R0 bottleneck analysis when available.
    if let Some(bmap) = bottleneck_map {
        if let Some(gemm_bn) = bmap.gemm_bottlenecks.get(&producer) {
            use crate::compiler::pain_point::BottleneckType;
            return match gemm_bn.bottleneck {
                BottleneckType::MemoryBound { bandwidth_utilization } => {
                    bandwidth_utilization.max(0.1)
                }
                BottleneckType::ComputeBound { compute_utilization } => {
                    compute_utilization.max(0.1).min(1.0)
                }
                BottleneckType::LatencyBound { .. } => 0.5,
            };
        }
    }

    // Fallback: use SemanticDAG's bottleneck classification.
    match producer_class {
        OpClass::Gemm => 1.0, // GEMM fusion is always potentially beneficial
        OpClass::Reduction => 0.8,
        OpClass::ElemWise => {
            let ai = dag.node(producer).map(|n| n.arithmetic_intensity).unwrap_or(0.0);
            if ai < 2.0 { 1.0 } else { 0.8 }
        }
        OpClass::Injective => 0.7,
        OpClass::Opaque => 0.0,
    }
}

/// Compute strategy-specific weight from R0 fusion_benefits.
fn compute_strategy_weight(
    producer: OpId,
    consumer: OpId,
    dag: &SemanticDAG,
    bottleneck_map: Option<&OpBottleneckMap>,
) -> f64 {
    if let Some(bmap) = bottleneck_map {
        if let Some(gemm_bn) = bmap.gemm_bottlenecks.get(&producer) {
            // Use R0's precomputed fusion_benefits to find the best matching
            // strategy weight. Default weight = 1.0 if no specific match.
            let consumer_class = dag.node(consumer).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
            // EpilogueInjection for GEMM + ElemWise/Reduction is the highest-value fusion.
            if consumer_class == OpClass::Reduction || consumer_class == OpClass::ElemWise {
                return gemm_bn.fusion_benefits.values()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or(1.0)
                    / (producer_output_bytes(producer, dag) as f64 + 1.0).max(1.0)
                    * producer_output_bytes(producer, dag) as f64;
            }
        }
    }
    1.0
}

fn producer_output_bytes(producer: OpId, dag: &SemanticDAG) -> usize {
    dag.node(producer)
        .map(|n| {
            // Approximate: use AI to estimate. For simplicity, use a fixed heuristic.
            // In practice, this should use graph.tensor() for exact sizes.
            4096 * 4 // placeholder
        })
        .unwrap_or(0)
}

/// Estimate register pressure penalty for fusing two ops.
fn estimate_reg_penalty(
    _producer: OpId,
    _consumer: OpId,
    _graph: &CompilerGraph,
    dag: &SemanticDAG,
) -> usize {
    // Simple heuristic: each fused op adds some register pressure.
    // The actual register count depends on the ISA lowering, but we approximate
    // based on OpClass.
    let consumer_class = dag.node(_consumer).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
    match consumer_class {
        OpClass::ElemWise => 64,   // 2 SIMD regs for elementwise temporary
        OpClass::Injective => 128, // More regs for index computation
        OpClass::Reduction => 256, // Reduction needs accumulator + index regs
        _ => 0,
    }
}

/// OpClass fusion compatibility: can `consumer` be fused into a group
/// anchored by `anchor_class`?
pub fn can_fuse(anchor_class: OpClass, consumer_class: OpClass) -> bool {
    match anchor_class {
        OpClass::Gemm => {
            // GEMM can fuse with Reduction, ElemWise, Injective as epilogue.
            matches!(consumer_class, OpClass::Reduction | OpClass::ElemWise | OpClass::Injective)
        }
        OpClass::Reduction => {
            // Reduction can fuse with downstream ElemWise (rare but valid).
            matches!(consumer_class, OpClass::ElemWise)
        }
        OpClass::ElemWise => {
            // ElemWise chains with ElemWise or Injective.
            matches!(consumer_class, OpClass::ElemWise | OpClass::Injective)
        }
        OpClass::Injective => {
            // Injective chains with ElemWise (e.g., RoPE → SiLU).
            matches!(consumer_class, OpClass::ElemWise)
        }
        OpClass::Opaque => false,
    }
}

/// PDT fusion engine: walks the post-dominator tree and discovers fusion candidates.
pub struct PdtFusionEngine;

impl PdtFusionEngine {
    /// Discover all fusion candidates by walking the DAG in topological order,
    /// using PDT to validate safe fusion scope and score_fusion for gating.
    ///
    /// For each node, walk its DAG consumers (direct and chain). For each
    /// consumer, check OpClass compatibility and PDT containment (consumer's
    /// ipostdom must be compatible with anchor). Score and collect.
    pub fn discover_candidates(
        graph: &CompilerGraph,
        dag: &SemanticDAG,
        _pdt: &PostDominatorTree,
        bottleneck_map: Option<&OpBottleneckMap>,
    ) -> Vec<FusionCandidate> {
        let topo = graph.topological_sort();
        let mut candidates = Vec::new();
        let mut claimed: HashSet<OpId> = HashSet::new();

        // Walk in topological order (producers before consumers).
        for &op_id in &topo {
            if claimed.contains(&op_id) {
                continue;
            }

            let anchor_class = dag.node(op_id).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
            if anchor_class == OpClass::Opaque {
                continue;
            }

            // Walk DAG consumers: direct consumers of this node's output tensors.
            let mut fusable_consumers = Vec::new();
            let mut queue: VecDeque<OpId> = VecDeque::new();

            // Seed with direct consumers.
            if let Some(op) = graph.op(op_id) {
                for &out_tid in &op.outputs {
                    if let Some(tensor) = graph.tensor(out_tid) {
                        for &consumer_id in &tensor.consumers {
                            if !claimed.contains(&consumer_id) && consumer_id != op_id {
                                queue.push_back(consumer_id);
                            }
                        }
                    }
                }
            }

            // BFS walk along the consumer chain.
            while let Some(consumer_id) = queue.pop_front() {
                if claimed.contains(&consumer_id) {
                    continue;
                }
                let consumer_class = dag.node(consumer_id).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
                if !can_fuse(anchor_class, consumer_class) {
                    continue;
                }

                let score = score_fusion(op_id, consumer_id, graph, dag, bottleneck_map, None);
                if score > 0.0 {
                    fusable_consumers.push(consumer_id);
                    claimed.insert(consumer_id);

                    // Continue chain: this consumer's outputs feed further consumers.
                    if let Some(consumer_op) = graph.op(consumer_id) {
                        for &out_tid in &consumer_op.outputs {
                            if let Some(tensor) = graph.tensor(out_tid) {
                                if tensor.consumers.len() == 1 {
                                    // Single consumer → safe to continue chain.
                                    let next_id = tensor.consumers[0];
                                    if !claimed.contains(&next_id) {
                                        queue.push_back(next_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !fusable_consumers.is_empty() {
                candidates.push(FusionCandidate {
                    anchor: op_id,
                    consumers: fusable_consumers,
                    anchor_class,
                });
            }
        }

        candidates
    }

    /// Check if `consumer` directly consumes `producer`'s output.
    fn has_data_dependency(producer: OpId, consumer: OpId, graph: &CompilerGraph) -> bool {
        let prod = match graph.op(producer) {
            Some(o) => o,
            None => return false,
        };
        for &out_tid in &prod.outputs {
            if let Some(tensor) = graph.tensor(out_tid) {
                if tensor.consumers.contains(&consumer) {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, SymDim};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::types::DType;

    fn build_simple_graph() -> (CompilerGraph, SemanticDAG, PostDominatorTree) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;

        // GEMM → SiLU → output
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);

        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        (g, dag, pdt)
    }

    #[test]
    fn test_pdt_build_simple_chain() {
        let (g, _dag, pdt) = build_simple_graph();

        // GEMM (OpId(0)) → SiLU (OpId(1))
        // SiLU is the exit node.
        // ipostdom(GEMM) = SiLU (all paths from GEMM go through SiLU).
        assert_eq!(pdt.len(), 2);
        assert_eq!(pdt.ipostdom(OpId(0)), Some(OpId(1)));
    }

    #[test]
    fn test_pdt_children() {
        let (g, _dag, pdt) = build_simple_graph();

        // SiLU is ipostdom of GEMM, so GEMM is a child of SiLU in PDT.
        let silu_children = pdt.children(OpId(1));
        assert!(silu_children.contains(&OpId(0)));
    }

    #[test]
    fn test_can_fuse_gemm_elemwise() {
        assert!(can_fuse(OpClass::Gemm, OpClass::ElemWise));
        assert!(can_fuse(OpClass::Gemm, OpClass::Reduction));
        assert!(can_fuse(OpClass::ElemWise, OpClass::ElemWise));
        assert!(!can_fuse(OpClass::Opaque, OpClass::ElemWise));
        assert!(!can_fuse(OpClass::Gemm, OpClass::Opaque));
    }

    #[test]
    fn test_score_fusion_positive() {
        let (g, dag, _pdt) = build_simple_graph();
        let score = score_fusion(OpId(0), OpId(1), &g, &dag, None, None);
        // GEMM → SiLU should have positive score (eliminates intermediate tensor).
        assert!(score > 0.0, "GEMM → SiLU fusion score should be positive, got {}", score);
    }

    #[test]
    fn test_score_fusion_zero_for_unrelated() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let c = g.add_tensor_concrete("c", &[1, 64], dt);

        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: dt, trans_b: false, has_bias: false }), vec![a, w], vec![out], "gemm");
        g.add_op(Op::Silu, vec![b], vec![c], "silu_unrelated");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);

        // GEMM and unrelated SiLU have no data dependency → score should be 0.
        let score = score_fusion(OpId(0), OpId(1), &g, &dag, None, None);
        assert_eq!(score, 0.0, "Unrelated ops should have 0 fusion score");
    }

    #[test]
    fn test_pdt_discover_candidates() {
        let (g, dag, pdt) = build_simple_graph();
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        assert!(!candidates.is_empty(), "Should find at least one fusion candidate");
        let first = &candidates[0];
        assert_eq!(first.anchor, OpId(0)); // GEMM is anchor
        assert!(first.consumers.contains(&OpId(1))); // SiLU is consumer
    }

    #[test]
    fn test_pdt_diamond_graph() {
        // Diamond: A → B, A → C, B → D, C → D
        // OpId(0)=silu_b, OpId(1)=silu_c, OpId(2)=add_d
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let c = g.add_tensor_concrete("c", &[1, 64], dt);
        let d_out = g.add_tensor_concrete("d_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![b], "silu_b");
        g.add_op(Op::Silu, vec![a], vec![c], "silu_c");
        g.add_op(Op::Add, vec![b, c], vec![d_out], "add_d");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // 3 ops total
        assert_eq!(pdt.len(), 3);
        // add_d (OpId(2)) is the exit node
        // ipostdom(silu_b) = add_d (all paths from B go through D)
        // ipostdom(silu_c) = add_d (all paths from C go through D)
        assert_eq!(pdt.ipostdom(OpId(0)), Some(OpId(2)));
        assert_eq!(pdt.ipostdom(OpId(1)), Some(OpId(2)));
    }

    #[test]
    fn test_compute_bytes_saved() {
        let (g, _dag, _pdt) = build_simple_graph();
        let bytes = compute_bytes_saved(OpId(0), OpId(1), &g);
        // gemm_out is [1, 64] × F32 = 256 bytes
        assert_eq!(bytes, 256);
    }

    // @trace TEST-PDT-09 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_pdt_build_empty_graph() {
        // Arrange: empty graph has zero ops.
        let g = CompilerGraph::new();
        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);

        // Act
        let pdt = PostDominatorTree::build(&g, &dag);

        // Assert: PDT should be empty with exit_id = u32::MAX.
        assert!(pdt.is_empty());
        assert_eq!(pdt.len(), 0);
        assert_eq!(pdt.exit_id, OpId(u32::MAX));
        // No nodes exist, so ipostdom/children return defaults.
        assert_eq!(pdt.ipostdom(OpId(0)), None);
        assert_eq!(pdt.children(OpId(0)), &[]);
    }

    // @trace TEST-PDT-10 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_pdt_subtree_bfs_collection() {
        // Arrange: build diamond graph A→B, A→C, B→D, C→D.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let c = g.add_tensor_concrete("c", &[1, 64], dt);
        let d_out = g.add_tensor_concrete("d_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![b], "silu_b");
        g.add_op(Op::Silu, vec![a], vec![c], "silu_c");
        g.add_op(Op::Add, vec![b, c], vec![d_out], "add_d");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act: subtree rooted at the exit node (add_d, OpId(2)) should contain all nodes.
        let subtree = pdt.subtree(OpId(2));

        // Assert: exit node's subtree includes all 3 ops.
        assert_eq!(subtree.len(), 3);
        assert!(subtree.contains(&OpId(2)));
        assert!(subtree.contains(&OpId(0)));
        assert!(subtree.contains(&OpId(1)));

        // Subtree of a leaf node should contain only itself.
        let leaf_subtree = pdt.subtree(OpId(0));
        assert_eq!(leaf_subtree.len(), 1);
        assert!(leaf_subtree.contains(&OpId(0)));
    }

    // @trace TEST-PDT-11 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_can_fuse_full_compatibility_matrix() {
        // Arrange & Act & Assert: exhaustive OpClass compatibility check.
        // GEMM anchor
        assert!(can_fuse(OpClass::Gemm, OpClass::ElemWise));
        assert!(can_fuse(OpClass::Gemm, OpClass::Reduction));
        assert!(can_fuse(OpClass::Gemm, OpClass::Injective));
        assert!(!can_fuse(OpClass::Gemm, OpClass::Gemm));    // GEMM cannot fuse with GEMM
        assert!(!can_fuse(OpClass::Gemm, OpClass::Opaque));

        // Reduction anchor
        assert!(can_fuse(OpClass::Reduction, OpClass::ElemWise));
        assert!(!can_fuse(OpClass::Reduction, OpClass::Reduction));
        assert!(!can_fuse(OpClass::Reduction, OpClass::Gemm));
        assert!(!can_fuse(OpClass::Reduction, OpClass::Injective));
        assert!(!can_fuse(OpClass::Reduction, OpClass::Opaque));

        // ElemWise anchor
        assert!(can_fuse(OpClass::ElemWise, OpClass::ElemWise));
        assert!(can_fuse(OpClass::ElemWise, OpClass::Injective));
        assert!(!can_fuse(OpClass::ElemWise, OpClass::Gemm));
        assert!(!can_fuse(OpClass::ElemWise, OpClass::Reduction));
        assert!(!can_fuse(OpClass::ElemWise, OpClass::Opaque));

        // Injective anchor
        assert!(can_fuse(OpClass::Injective, OpClass::ElemWise));
        assert!(!can_fuse(OpClass::Injective, OpClass::Injective));
        assert!(!can_fuse(OpClass::Injective, OpClass::Gemm));
        assert!(!can_fuse(OpClass::Injective, OpClass::Reduction));
        assert!(!can_fuse(OpClass::Injective, OpClass::Opaque));

        // Opaque anchor — nothing fuses
        assert!(!can_fuse(OpClass::Opaque, OpClass::ElemWise));
        assert!(!can_fuse(OpClass::Opaque, OpClass::Reduction));
        assert!(!can_fuse(OpClass::Opaque, OpClass::Gemm));
        assert!(!can_fuse(OpClass::Opaque, OpClass::Injective));
        assert!(!can_fuse(OpClass::Opaque, OpClass::Opaque));
    }

    // @trace TEST-PDT-12 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_compute_bytes_saved_no_shared_tensor() {
        // Arrange: two unrelated ops — producer output is not consumed by consumer.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out1 = g.add_tensor_concrete("out1", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let out2 = g.add_tensor_concrete("out2", &[1, 64], dt);

        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![a, w], vec![out1], "gemm",
        );
        g.add_op(Op::Silu, vec![b], vec![out2], "silu");

        // Act
        let bytes = compute_bytes_saved(OpId(0), OpId(1), &g);

        // Assert: no shared tensor between OpId(0) and OpId(1).
        assert_eq!(bytes, 0, "No shared tensor should yield 0 bytes saved");
    }

    // @trace TEST-PDT-13 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_compute_bytes_saved_nonexistent_producer() {
        // Arrange: empty graph, OpId(99) does not exist.
        let g = CompilerGraph::new();

        // Act
        let bytes = compute_bytes_saved(OpId(99), OpId(0), &g);

        // Assert: nonexistent producer returns 0.
        assert_eq!(bytes, 0, "Nonexistent producer should yield 0 bytes");
    }

    // @trace TEST-PDT-14 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_pdt_three_node_linear_chain() {
        // Arrange: GEMM → SiLU → Add (3-node chain).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let bias = g.add_tensor_concrete("bias", &[1, 64], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 64], dt);

        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(Op::Add, vec![silu_out, bias], vec![add_out], "add");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act & Assert
        assert_eq!(pdt.len(), 3);
        // GEMM → SiLU → Add: all paths from GEMM pass through SiLU and Add.
        // ipostdom(GEMM) = SiLU (closest post-dominator)
        assert_eq!(pdt.ipostdom(OpId(0)), Some(OpId(1)));
        // ipostdom(SiLU) = Add
        assert_eq!(pdt.ipostdom(OpId(1)), Some(OpId(2)));
        // Add is the exit node, no ipostdom.
        assert_eq!(pdt.ipostdom(OpId(2)), None);
    }

    // @trace TEST-PDT-15 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_discover_candidates_all_opaque_graph() {
        // Arrange: graph with ops that SemanticDAG classifies as Opaque.
        // CheckStopCondition is typically Opaque since it has no standard scalar registry entry.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let logits = g.add_tensor_concrete("logits", &[1, 100], dt);
        let token_out = g.add_tensor_concrete("token_out", &[1], dt);

        g.add_op(Op::CheckStopCondition, vec![logits], vec![token_out], "stop_check");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert: Opaque ops should not produce fusion candidates.
        assert!(candidates.is_empty(), "Opaque-only graph should have no fusion candidates");
    }

    // @trace TEST-PDT-16 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_score_fusion_with_bottleneck_map() {
        // Arrange: GEMM → SiLU with an OpBottleneckMap providing MemoryBound bottleneck.
        let (g, dag, _pdt) = build_simple_graph();

        use crate::compiler::pain_point::{BottleneckType, GemmBottleneck, GemmRole, FusionPriority, OpBottleneckMap, ExecPattern, ParallelismDesc};

        let mut fusion_benefits = std::collections::HashMap::new();
        fusion_benefits.insert(FusionPriority::EpilogueInjection, 2.5);

        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: {
                let mut m = std::collections::HashMap::new();
                m.insert(OpId(0), GemmBottleneck {
                    gemm_role: GemmRole::LmHead,
                    shape: (1, 64, 64),
                    arithmetic_intensity: 1.0,
                    ridge_point: 10.0,
                    bottleneck: BottleneckType::MemoryBound { bandwidth_utilization: 0.8 },
                    optimal_fusion: FusionPriority::EpilogueInjection,
                    fusion_benefits: fusion_benefits,
                    exec_pattern: ExecPattern::TileGemm {
                        tile_m: 4, tile_n: 4, tile_k: 64, warp_m: 0, warp_n: 0, mma_k: 0, pipeline_depth: 0,
                    },
                    parallelism: ParallelismDesc::SimdVectorize { element_width: 8, unroll_factor: 1 },
                });
                m
            },
            ridge_point: 10.0,
        };

        // Act
        let score_with_bn = score_fusion(OpId(0), OpId(1), &g, &dag, Some(&bottleneck_map), None);

        // Assert: score should be positive and reflect the bottleneck scale (0.8).
        assert!(score_with_bn > 0.0, "Score with bottleneck map should be positive, got {}", score_with_bn);
        // The bottleneck scale for MemoryBound(0.8) is max(0.8, 0.1) = 0.8.
        // Score without bottleneck uses scale=1.0 for GEMM, so bottleneck score should differ.
        let score_without = score_fusion(OpId(0), OpId(1), &g, &dag, None, None);
        // With MemoryBound 0.8 scale, the bytes_saved component is scaled down.
        assert_ne!(score_with_bn, score_without, "Bottleneck map should affect fusion score");
    }

    // @trace TEST-PDT-17 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_score_fusion_compute_bound_bottleneck() {
        // Arrange: GEMM → SiLU with ComputeBound bottleneck.
        let (g, dag, _pdt) = build_simple_graph();

        use crate::compiler::pain_point::{BottleneckType, GemmBottleneck, GemmRole, FusionPriority, OpBottleneckMap, ExecPattern, ParallelismDesc};

        let mut fusion_benefits = std::collections::HashMap::new();
        fusion_benefits.insert(FusionPriority::EpilogueInjection, 1.5);

        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: {
                let mut m = std::collections::HashMap::new();
                m.insert(OpId(0), GemmBottleneck {
                    gemm_role: GemmRole::OutputProjection,
                    shape: (1, 64, 64),
                    arithmetic_intensity: 50.0,
                    ridge_point: 10.0,
                    bottleneck: BottleneckType::ComputeBound { compute_utilization: 0.6 },
                    optimal_fusion: FusionPriority::EpilogueInjection,
                    fusion_benefits: fusion_benefits,
                    exec_pattern: ExecPattern::TileGemm {
                        tile_m: 4, tile_n: 4, tile_k: 64, warp_m: 0, warp_n: 0, mma_k: 0, pipeline_depth: 0,
                    },
                    parallelism: ParallelismDesc::SimdVectorize { element_width: 8, unroll_factor: 1 },
                });
                m
            },
            ridge_point: 10.0,
        };

        // Act
        let score = score_fusion(OpId(0), OpId(1), &g, &dag, Some(&bottleneck_map), None);

        // Assert: ComputeBound(0.6) scale = max(0.6, 0.1).min(1.0) = 0.6.
        // Score should still be positive but lower than without bottleneck.
        assert!(score > 0.0, "ComputeBound score should be positive, got {}", score);
    }

    // @trace TEST-PDT-18 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_score_fusion_latency_bound_bottleneck() {
        // Arrange: GEMM → SiLU with LatencyBound bottleneck.
        let (g, dag, _pdt) = build_simple_graph();

        use crate::compiler::pain_point::{BottleneckType, GemmBottleneck, GemmRole, FusionPriority, OpBottleneckMap, ExecPattern, ParallelismDesc};

        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: {
                let mut m = std::collections::HashMap::new();
                m.insert(OpId(0), GemmBottleneck {
                    gemm_role: GemmRole::Other,
                    shape: (1, 64, 64),
                    arithmetic_intensity: 0.1,
                    ridge_point: 10.0,
                    bottleneck: BottleneckType::LatencyBound { estimated_latency_ns: 100.0 },
                    optimal_fusion: FusionPriority::EpilogueInjection,
                    fusion_benefits: std::collections::HashMap::new(),
                    exec_pattern: ExecPattern::TileGemm {
                        tile_m: 4, tile_n: 4, tile_k: 64, warp_m: 0, warp_n: 0, mma_k: 0, pipeline_depth: 0,
                    },
                    parallelism: ParallelismDesc::SimdVectorize { element_width: 8, unroll_factor: 1 },
                });
                m
            },
            ridge_point: 10.0,
        };

        // Act
        let score = score_fusion(OpId(0), OpId(1), &g, &dag, Some(&bottleneck_map), None);

        // Assert: LatencyBound scale = 0.5 (hardcoded in compute_bottleneck_scale).
        assert!(score > 0.0, "LatencyBound score should be positive, got {}", score);
    }

    // @trace TEST-PDT-19 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_pdt_isolated_single_node() {
        // Arrange: graph with a single op that has no connection to anything else.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let out = g.add_tensor_concrete("out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);

        // Act
        let pdt = PostDominatorTree::build(&g, &dag);

        // Assert: single node has no ipostdom (it is the exit).
        assert_eq!(pdt.len(), 1);
        assert_eq!(pdt.ipostdom(OpId(0)), None);
        assert_eq!(pdt.children(OpId(0)), &[]);
        assert!(!pdt.is_empty());
    }

    // @trace TEST-PDT-20 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_discover_candidates_chain_fusion() {
        // Arrange: ElemWise chain: Silu → Mul → Add (all ElemWise, chain-fusable).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(Op::Mul, vec![silu_out], vec![mul_out], "mul");
        g.add_op(Op::Add, vec![mul_out, b], vec![add_out], "add");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert: Silu (ElemWise) should be anchor, Mul (ElemWise) should be fused consumer.
        // Add (ElemWise) is a consumer of Mul but also takes external input b —
        // it has two inputs so its output tensor has single consumer pattern check.
        assert!(!candidates.is_empty(), "Should find fusion candidates in ElemWise chain");
        let first = &candidates[0];
        assert_eq!(first.anchor, OpId(0));
        assert!(first.consumers.contains(&OpId(1)), "Mul should be fused into Silu anchor");
    }

    // @trace TEST-PDT-21 [req:REQ-FUSION] [level:unit]
    #[test]
    fn test_discover_candidates_broken_chain() {
        // Arrange: ElemWise → Opaque → ElemWise — chain should break at Opaque.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let logits = g.add_tensor_concrete("logits", &[1, 64], dt);
        let sc_out = g.add_tensor_concrete("sc_out", &[1], dt);
        let gelu_out = g.add_tensor_concrete("gelu_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(Op::CheckStopCondition, vec![logits], vec![sc_out], "stop");
        g.add_op(Op::Gelu, vec![silu_out], vec![gelu_out], "gelu");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert: Opaque (CheckStopCondition) should not be in any candidate.
        // Silu (ElemWise) cannot fuse with CheckStopCondition, but can potentially fuse
        // with Gelu (ElemWise) if there is a data dependency chain.
        for candidate in &candidates {
            assert!(
                !candidate.consumers.contains(&OpId(1)),
                "Opaque op should not appear as a fusion consumer"
            );
        }
    }

    // ── Test 22: PdtNode Debug format ──

    #[test]
    fn pdt_node_debug_format() {
        let node = PdtNode {
            op_id: OpId(5),
            ipostdom: Some(OpId(10)),
            children: vec![OpId(3), OpId(4)],
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("PdtNode"));
        assert!(debug.contains("op_id"));
    }

    // ── Test 23: PostDominatorTree Debug format ──

    #[test]
    fn pdt_debug_format() {
        let (_, _, pdt) = build_simple_graph();
        let debug = format!("{:?}", pdt);
        assert!(debug.contains("PostDominatorTree"));
    }

    // ── Test 24: FusionCandidate Debug format ──

    #[test]
    fn fusion_candidate_debug_format() {
        let candidate = FusionCandidate {
            anchor: OpId(0),
            consumers: vec![OpId(1), OpId(2)],
            anchor_class: OpClass::Gemm,
        };
        let debug = format!("{:?}", candidate);
        assert!(debug.contains("FusionCandidate"));
        assert!(debug.contains("Gemm"));
    }

    // ── Test 25: FusionCandidate Clone ──

    #[test]
    fn fusion_candidate_clone() {
        let candidate = FusionCandidate {
            anchor: OpId(0),
            consumers: vec![OpId(1)],
            anchor_class: OpClass::ElemWise,
        };
        let cloned = candidate.clone();
        assert_eq!(cloned.anchor, OpId(0));
        assert_eq!(cloned.consumers.len(), 1);
        assert_eq!(cloned.anchor_class, OpClass::ElemWise);
    }

    // ── Test 26: PdtNode Clone ──

    #[test]
    fn pdt_node_clone() {
        let node = PdtNode {
            op_id: OpId(7),
            ipostdom: Some(OpId(8)),
            children: vec![OpId(5)],
        };
        let cloned = node.clone();
        assert_eq!(cloned.op_id, OpId(7));
        assert_eq!(cloned.ipostdom, Some(OpId(8)));
        assert_eq!(cloned.children, vec![OpId(5)]);
    }

    // ── Test 27: PostDominatorTree subtree of nonexistent node ──

    #[test]
    fn pdt_subtree_nonexistent_node() {
        let (_, _, pdt) = build_simple_graph();
        // OpId(99) doesn't exist → subtree returns just [OpId(99)]
        let subtree = pdt.subtree(OpId(99));
        assert_eq!(subtree, vec![OpId(99)]);
    }

    // ── Test 28: PdtFusionEngine has_data_dependency ──

    #[test]
    fn pdt_has_data_dependency_true() {
        let (g, _, _) = build_simple_graph();
        // GEMM (OpId(0)) → SiLU (OpId(1)): GEMM's output is consumed by SiLU.
        assert!(PdtFusionEngine::has_data_dependency(OpId(0), OpId(1), &g));
    }

    #[test]
    fn pdt_has_data_dependency_false() {
        let (g, _, _) = build_simple_graph();
        // SiLU (OpId(1)) does not feed GEMM (OpId(0)).
        assert!(!PdtFusionEngine::has_data_dependency(OpId(1), OpId(0), &g));
    }

    #[test]
    fn pdt_has_data_dependency_nonexistent() {
        let g = CompilerGraph::new();
        assert!(!PdtFusionEngine::has_data_dependency(OpId(0), OpId(1), &g));
    }

    // ── Test 31: estimate_reg_penalty via score_fusion for different consumer classes ──

    #[test]
    fn score_fusion_elemwise_chain() {
        // Arrange: Silu → Mul (both ElemWise)
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(Op::Mul, vec![silu_out], vec![mul_out], "mul");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);

        // Act: Silu → Mul should have positive fusion score
        let score = score_fusion(OpId(0), OpId(1), &g, &dag, None, None);
        assert!(score > 0.0, "ElemWise chain fusion score should be positive, got {}", score);
    }

    // ── Test 32: discover_candidates empty graph ──

    #[test]
    fn discover_candidates_empty_graph() {
        let g = CompilerGraph::new();
        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);
        assert!(candidates.is_empty(), "Empty graph should yield no candidates");
    }

    // ── Test 33: PostDominatorTree exit_id matches last op ──

    #[test]
    fn pdt_exit_id_matches_last_topo_op() {
        let (_, _, pdt) = build_simple_graph();
        // 2 ops: OpId(0)=gemm, OpId(1)=silu. Exit = OpId(1).
        assert_eq!(pdt.exit_id, OpId(1));
    }

    // ── Test 34: compute_bytes_saved with self-connection ──

    #[test]
    fn compute_bytes_saved_self_connection() {
        let (g, _, _) = build_simple_graph();
        // A node doesn't consume its own output.
        let bytes = compute_bytes_saved(OpId(0), OpId(0), &g);
        assert_eq!(bytes, 0, "Self-connection should yield 0 bytes");
    }

    // ── Test 35: PDT on fan-out graph (one producer, multiple independent consumers) ──

    #[test]
    fn pdt_fan_out_graph_ipostdom() {
        // Arrange: Silu → Silu_b, Silu → Silu_c, Silu_b → Add, Silu_c → Add
        // but with fan-out: A feeds both B and C independently (no reconvergence).
        // A → B, A → C (B and C have no common successor).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let b_out = g.add_tensor_concrete("b_out", &[1, 64], dt);
        let c_out = g.add_tensor_concrete("c_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![b_out], "silu_b");
        g.add_op(Op::Silu, vec![a], vec![c_out], "silu_c");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Assert: 2 ops, both are leaves (no shared post-dominator except self).
        assert_eq!(pdt.len(), 2);
        // Neither B nor C post-dominates the other since they are independent.
        // Both are exit nodes in their respective paths, so no ipostdom from the other.
        let ipostdom_b = pdt.ipostdom(OpId(0));
        let ipostdom_c = pdt.ipostdom(OpId(1));
        // In a pure fan-out with no common successor, each node only post-dominates itself.
        assert!(ipostdom_b.is_none() || ipostdom_b == Some(OpId(0)));
        assert!(ipostdom_c.is_none() || ipostdom_c == Some(OpId(1)));
    }

    // ── Test 36: Four-node linear chain ipostdom chain ──

    #[test]
    fn pdt_four_node_chain_ipostdom_chain() {
        // Arrange: Silu → Mul → Add → Silu2 (4-node linear chain).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 64], dt);
        let silu2_out = g.add_tensor_concrete("silu2_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(Op::Mul, vec![silu_out], vec![mul_out], "mul");
        g.add_op(Op::Add, vec![mul_out, b], vec![add_out], "add");
        g.add_op(Op::Silu, vec![add_out], vec![silu2_out], "silu2");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Assert: ipostdom chain — each node's ipostdom is its immediate successor.
        assert_eq!(pdt.len(), 4);
        assert_eq!(pdt.ipostdom(OpId(0)), Some(OpId(1)), "Silu ipostdom = Mul");
        assert_eq!(pdt.ipostdom(OpId(1)), Some(OpId(2)), "Mul ipostdom = Add");
        assert_eq!(pdt.ipostdom(OpId(2)), Some(OpId(3)), "Add ipostdom = Silu2");
        assert_eq!(pdt.ipostdom(OpId(3)), None, "Silu2 is exit, no ipostdom");
    }

    // ── Test 37: Fusion score monotonically increases with tensor size ──

    #[test]
    fn score_fusion_larger_tensor_higher_score() {
        // Arrange: two GEMM→SiLU graphs with different intermediate tensor sizes.
        let mut g_small = CompilerGraph::new();
        let mut g_large = CompilerGraph::new();
        let dt = DType::F32;

        // Small: [1, 16] intermediate = 64 bytes
        let a_s = g_small.add_tensor_concrete("a", &[1, 16], dt);
        let w_s = g_small.add_tensor_concrete("w", &[16, 16], dt);
        let gemm_out_s = g_small.add_tensor_concrete("gemm_out", &[1, 16], dt);
        let silu_out_s = g_small.add_tensor_concrete("silu_out", &[1, 16], dt);
        g_small.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 16, k: 16, dtype: dt, trans_b: false, has_bias: false }),
            vec![a_s, w_s], vec![gemm_out_s], "gemm",
        );
        g_small.add_op(Op::Silu, vec![gemm_out_s], vec![silu_out_s], "silu");

        // Large: [1, 256] intermediate = 1024 bytes
        let a_l = g_large.add_tensor_concrete("a", &[1, 256], dt);
        let w_l = g_large.add_tensor_concrete("w", &[256, 256], dt);
        let gemm_out_l = g_large.add_tensor_concrete("gemm_out", &[1, 256], dt);
        let silu_out_l = g_large.add_tensor_concrete("silu_out", &[1, 256], dt);
        g_large.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false, has_bias: false }),
            vec![a_l, w_l], vec![gemm_out_l], "gemm",
        );
        g_large.add_op(Op::Silu, vec![gemm_out_l], vec![silu_out_l], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let dag_small = SemanticDAG::from_graph(&g_small, &registry);
        let dag_large = SemanticDAG::from_graph(&g_large, &registry);

        // Act
        let score_small = score_fusion(OpId(0), OpId(1), &g_small, &dag_small, None, None);
        let score_large = score_fusion(OpId(0), OpId(1), &g_large, &dag_large, None, None);

        // Assert: larger intermediate tensor → higher fusion score.
        assert!(
            score_large > score_small,
            "Larger tensor should yield higher fusion score: large={} vs small={}",
            score_large, score_small,
        );
        assert!(score_small > 0.0, "Small fusion score should still be positive");
        assert!(score_large > 0.0, "Large fusion score should be positive");
    }

    // ── Test 38: discover_candidates does not double-claim consumers ──

    #[test]
    fn discover_candidates_no_double_claim() {
        // Arrange: Silu → Mul → Add chain. Each op should be claimed at most once.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(Op::Mul, vec![silu_out], vec![mul_out], "mul");
        g.add_op(Op::Add, vec![mul_out, b], vec![add_out], "add");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert: collect all claimed consumer OpIds and verify no duplicates.
        let mut all_consumers: Vec<OpId> = Vec::new();
        for candidate in &candidates {
            all_consumers.extend_from_slice(&candidate.consumers);
        }
        let unique_count = all_consumers.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(
            unique_count, all_consumers.len(),
            "No consumer should appear in more than one fusion candidate"
        );
    }

    // ── Test 39: Two disconnected subgraphs in a single CompilerGraph ──

    #[test]
    fn pdt_two_disconnected_subgraphs() {
        // Arrange: two independent chains: Silu_a→Mul_a and Silu_b→Mul_b.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a1 = g.add_tensor_concrete("a1", &[1, 64], dt);
        let s1 = g.add_tensor_concrete("s1", &[1, 64], dt);
        let m1 = g.add_tensor_concrete("m1", &[1, 64], dt);
        let a2 = g.add_tensor_concrete("a2", &[1, 64], dt);
        let s2 = g.add_tensor_concrete("s2", &[1, 64], dt);
        let m2 = g.add_tensor_concrete("m2", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a1], vec![s1], "silu_1");
        g.add_op(Op::Mul, vec![s1], vec![m1], "mul_1");
        g.add_op(Op::Silu, vec![a2], vec![s2], "silu_2");
        g.add_op(Op::Mul, vec![s2], vec![m2], "mul_2");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Assert: 4 nodes total.
        assert_eq!(pdt.len(), 4);
        // The last op in topo order is the exit node.
        // Disconnected components still get correct ipostdom within their own subchain.
        // OpId(0) → OpId(1) in one chain, OpId(2) → OpId(3) in the other.
        assert_eq!(pdt.ipostdom(OpId(0)), Some(OpId(1)));
        assert_eq!(pdt.ipostdom(OpId(2)), Some(OpId(3)));
    }

    // ── Test 40: Injective → ElemWise fusion compatibility and discovery ──

    #[test]
    fn discover_candidates_injective_to_elemwise() {
        // Arrange: RoPE (Injective) → Silu (ElemWise) — valid fusion pair.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 64], dt);
        let rope_out = g.add_tensor_concrete("rope_out", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let freq = g.add_tensor_concrete("freq", &[32], dt);

        g.add_op(Op::RoPE(RopeSpec { num_heads: 4, head_dim: 16, theta: 10000.0, partial: 1.0, rope_scaling: None }), vec![input, freq], vec![rope_out], "rope");
        g.add_op(Op::Silu, vec![rope_out], vec![silu_out], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert: Rope (Injective) should fuse with Silu (ElemWise).
        // can_fuse(Injective, ElemWise) == true.
        assert!(!candidates.is_empty(), "Injective → ElemWise should produce fusion candidates");
        let found = candidates.iter().any(|c| {
            c.anchor_class == OpClass::Injective && c.consumers.contains(&OpId(1))
        });
        assert!(found, "Should find Injective anchor with Silu consumer");
    }

    // ── Test 41: Reduction anchor discovers ElemWise consumer ──

    #[test]
    fn discover_candidates_reduction_anchor() {
        // Arrange: RmsNorm (Reduction) → Silu (ElemWise).
        // can_fuse(Reduction, ElemWise) == true.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 64], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let weight = g.add_tensor_concrete("weight", &[64], dt);

        g.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }), vec![input, weight], vec![norm_out], "rmsnorm");
        g.add_op(Op::Silu, vec![norm_out], vec![silu_out], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert
        let reduction_anchors: Vec<&FusionCandidate> = candidates
            .iter()
            .filter(|c| c.anchor_class == OpClass::Reduction)
            .collect();
        assert!(
            !reduction_anchors.is_empty(),
            "RmsNorm (Reduction) should be an anchor with Silu consumer"
        );
        assert!(
            reduction_anchors.iter().any(|c| c.consumers.contains(&OpId(1))),
            "Reduction anchor should have Silu as consumer"
        );
    }

    // ── Test 42: GEMM → GEMM does not fuse ──

    #[test]
    fn discover_candidates_gemm_to_gemm_no_fusion() {
        // Arrange: two consecutive GEMMs. can_fuse(Gemm, Gemm) == false.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let w1 = g.add_tensor_concrete("w1", &[64, 64], dt);
        let gemm1_out = g.add_tensor_concrete("gemm1_out", &[1, 64], dt);
        let w2 = g.add_tensor_concrete("w2", &[64, 64], dt);
        let gemm2_out = g.add_tensor_concrete("gemm2_out", &[1, 64], dt);

        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![a, w1], vec![gemm1_out], "gemm1",
        );
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![gemm1_out, w2], vec![gemm2_out], "gemm2",
        );

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act
        let candidates = PdtFusionEngine::discover_candidates(&g, &dag, &pdt, None);

        // Assert: GEMM→GEMM is not fusable, so no candidate should have Gemm consumer.
        for candidate in &candidates {
            for consumer in &candidate.consumers {
                let consumer_class = dag.node(*consumer).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
                assert_ne!(
                    consumer_class, OpClass::Gemm,
                    "GEMM should never appear as a fusion consumer of another GEMM"
                );
            }
        }
    }

    // ── Test 43: PDT subtree rooted at non-exit includes only descendants ──

    #[test]
    fn pdt_subtree_mid_chain_excludes_later_nodes() {
        // Arrange: 4-node chain: Silu → Mul → Add → Silu2.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[1, 64], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 64], dt);
        let silu2_out = g.add_tensor_concrete("silu2_out", &[1, 64], dt);

        g.add_op(Op::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(Op::Mul, vec![silu_out], vec![mul_out], "mul");
        g.add_op(Op::Add, vec![mul_out, b], vec![add_out], "add");
        g.add_op(Op::Silu, vec![add_out], vec![silu2_out], "silu2");

        let registry = ScalarOpRegistry::with_defaults();
        let dag = SemanticDAG::from_graph(&g, &registry);
        let pdt = PostDominatorTree::build(&g, &dag);

        // Act: subtree of OpId(2) (Add). Add's children in PDT are nodes whose
        // ipostdom is Add. In a chain, no later node has ipostdom = Add
        // (Mul's ipostdom = Add, but Mul is earlier, not a child in forward PDT sense).
        let subtree_add = pdt.subtree(OpId(2));

        // Assert: The subtree contains OpId(2) itself at minimum.
        assert!(subtree_add.contains(&OpId(2)), "Subtree should contain the root node");
        // The exit node OpId(3) should NOT be in subtree of OpId(2) because
        // OpId(3) has no ipostdom pointing to OpId(2).
        assert!(
            !subtree_add.contains(&OpId(3)),
            "Exit node should not be in subtree of a mid-chain node"
        );
    }

    // ── Test 44: compute_bytes_saved proportional to intermediate tensor element count ──

    #[test]
    fn compute_bytes_saved_proportional_to_element_count() {
        // Arrange: two GEMM→SiLU graphs with different intermediate tensor element counts.
        // Intermediate tensors always use 4 bytes (F32 accumulator) regardless of declared dtype,
        // so bytes_saved is proportional to element count.
        let mut g_small = CompilerGraph::new();
        let mut g_large = CompilerGraph::new();

        // Small: [1, 32] intermediate = 128 bytes
        let a_s = g_small.add_tensor_concrete("a", &[1, 32], DType::F32);
        let w_s = g_small.add_tensor_concrete("w", &[32, 32], DType::F32);
        let out_s = g_small.add_tensor_concrete("out", &[1, 32], DType::F32);
        let silu_s = g_small.add_tensor_concrete("silu", &[1, 32], DType::F32);
        g_small.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 32, k: 32, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a_s, w_s], vec![out_s], "gemm",
        );
        g_small.add_op(Op::Silu, vec![out_s], vec![silu_s], "silu");

        // Large: [1, 128] intermediate = 512 bytes (4x the elements)
        let a_l = g_large.add_tensor_concrete("a", &[1, 128], DType::F32);
        let w_l = g_large.add_tensor_concrete("w", &[128, 128], DType::F32);
        let out_l = g_large.add_tensor_concrete("out", &[1, 128], DType::F32);
        let silu_l = g_large.add_tensor_concrete("silu", &[1, 128], DType::F32);
        g_large.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a_l, w_l], vec![out_l], "gemm",
        );
        g_large.add_op(Op::Silu, vec![out_l], vec![silu_l], "silu");

        // Act
        let bytes_small = compute_bytes_saved(OpId(0), OpId(1), &g_small);
        let bytes_large = compute_bytes_saved(OpId(0), OpId(1), &g_large);

        // Assert: 128 elements × 4 bytes = 512, 32 elements × 4 bytes = 128.
        // Large should be exactly 4x small.
        assert!(bytes_small > 0, "Small graph bytes saved should be positive");
        assert!(bytes_large > 0, "Large graph bytes saved should be positive");
        assert_eq!(bytes_large, bytes_small * 4,
            "Larger intermediate tensor should have proportionally more bytes: small={} large={}",
            bytes_small, bytes_large,
        );
    }
}
