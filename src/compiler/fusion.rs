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

use crate::compiler::graph::{CompilerGraph, CompilerOp, OpKind, OpId, TensorId};
use crate::compiler::semantics::{self, OpSemantics};

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
        .filter(|op| matches!(op.kind, OpKind::Gemm { .. } | OpKind::GemmBias { .. }))
        .collect();

    // Group by first input tensor
    let mut by_input: std::collections::HashMap<TensorId, Vec<&CompilerOp>> =
        std::collections::HashMap::new();
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

        // Consumer must be elementwise and fusable
        if !semantics::can_fuse(&anchor.kind, &consumer.kind)
            && !matches!(semantics::classify(&consumer.kind), OpSemantics::Elementwise)
        {
            break;
        }

        if semantics::classify(&consumer.kind) != OpSemantics::Elementwise {
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
}
