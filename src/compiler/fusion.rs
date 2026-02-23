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
    /// The fusion pattern that was applied.
    pub pattern: FusionPattern,
    /// All op IDs in this group (anchor + epilogue), in execution order.
    pub ops: Vec<OpId>,
}

/// Named fusion patterns recognized by the pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionPattern {
    /// Single op, no fusion applied.
    Standalone,
    /// GEMM with fused elementwise epilogue (e.g., GEMM + SiLU, GEMM + Add).
    GemmEpilogue,
    /// Chain of elementwise ops collapsed into a single loop.
    ElementwiseChain,
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
            .filter(|g| g.pattern != FusionPattern::Standalone)
            .map(|g| g.ops.len())
            .sum()
    }
}

/// Run the fusion pass on a CompilerGraph.
///
/// Returns a `FusionPlan` describing which ops are grouped together.
/// The plan respects data dependencies (topological order) and only
/// fuses ops that `semantics::can_fuse()` approves.
pub fn fuse(graph: &CompilerGraph) -> FusionPlan {
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

                    let pattern = if norm_prefix.is_some() && !epilogue.is_empty() {
                        // Both norm prefix and epilogue
                        FusionPattern::GemmEpilogue
                    } else if norm_prefix.is_some() {
                        FusionPattern::NormIntoGemm
                    } else {
                        FusionPattern::GemmEpilogue
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
                        pattern,
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
                        pattern: FusionPattern::Standalone,
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

                let pattern = if chain_ids.is_empty() {
                    FusionPattern::Standalone
                } else {
                    FusionPattern::ElementwiseChain
                };

                for &oid in &all_ops {
                    op_to_group.insert(oid, gid);
                    claimed.insert(oid);
                }

                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: chain_ids,
                    pattern,
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
                    pattern: FusionPattern::Standalone,
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
                    pattern: FusionPattern::QkvSharedInput,
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
pub fn fuse_with_dag(graph: &CompilerGraph, registry: &ScalarOpRegistry) -> FusionPlan {
    let dag = SemanticDAG::from_graph(graph, registry);
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
                    let pattern = if norm_prefix.is_some() && !epilogue.is_empty() {
                        FusionPattern::GemmEpilogue
                    } else if norm_prefix.is_some() {
                        FusionPattern::NormIntoGemm
                    } else {
                        FusionPattern::GemmEpilogue
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
                        pattern,
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
                        pattern: FusionPattern::Standalone,
                        ops: vec![op_id],
                    });
                }
            }
            OpClass::ElemWise | OpClass::Injective => {
                // Try to chain with downstream elementwise/injective ops
                let chain = collect_elementwise_chain_dag(graph, op, &claimed, &dag);
                let gid = groups.len();
                let mut all_ops = vec![op_id];
                let chain_ids: Vec<OpId> = chain.iter().map(|o| o.id).collect();
                all_ops.extend_from_slice(&chain_ids);

                let pattern = if chain_ids.is_empty() {
                    FusionPattern::Standalone
                } else {
                    FusionPattern::ElementwiseChain
                };

                for &oid in &all_ops {
                    op_to_group.insert(oid, gid);
                    claimed.insert(oid);
                }

                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: chain_ids,
                    pattern,
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
                    pattern: FusionPattern::Standalone,
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
                g.pattern,
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
        let plan = fuse(&graph);

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
            .any(|g| g.pattern == FusionPattern::QkvSharedInput);
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

        let plan = fuse(&g);
        eprintln!("{plan}");

        // Should fuse into one group
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::GemmEpilogue);
        assert_eq!(plan.groups[0].ops.len(), 2);
    }

    #[test]
    fn test_standalone_reduction() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);

        g.add_op(OpKind::Softmax, vec![a], vec![b], "softmax");

        let plan = fuse(&g);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::Standalone);
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

        let plan = fuse(&g);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::ElementwiseChain);
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

        let plan = fuse(&g);
        // Silu and Softmax should be in separate groups
        assert_eq!(plan.num_groups(), 2);
    }

    #[test]
    fn test_gemma_geglu_fusion() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let plan = fuse(&graph);

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

        let plan = fuse_with_dag(&graph, &registry);

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
            .any(|g| g.pattern == FusionPattern::QkvSharedInput);
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

        let old_plan = fuse(&graph);
        let new_plan = fuse_with_dag(&graph, &registry);

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
        let plan = fuse_with_dag(&g, &registry);

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::GemmEpilogue);
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
        let plan = fuse_with_dag(&g, &registry);

        // RoPE + Silu should fuse (both fusable in new path: Injective + ElemWise)
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::ElementwiseChain);
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

        let plan = fuse(&g);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::GemmEpilogue);
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

        let plan = fuse(&g);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].pattern, FusionPattern::Standalone);
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

        let plan = fuse(&g);
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
        let plan = fuse_with_dag(&g, &registry);

        // Softmax and GEMM must be in separate groups — no NormIntoGemm fusion
        assert_eq!(plan.num_groups(), 2, "Softmax should not fuse as norm prefix");
        for group in &plan.groups {
            assert_ne!(
                group.pattern,
                FusionPattern::NormIntoGemm,
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

        let plan = fuse(&g);
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
        let plan = fuse_with_dag(&g, &registry);

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
}
