//! Parallel strategy — thread assignment per fusion group.
//!
//! Each fusion group gets a `ParallelStrategy` that determines how many
//! threads to use and along which dimension to parallelize. The decision
//! is based on the group's computation pattern, problem size, and the
//! hardware's parallel threshold from `DeviceProfile`.

use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass};
use crate::dispatch::DeviceProfile;

/// Thread assignment for a fusion group.
#[derive(Debug, Clone)]
pub struct ParallelStrategy {
    /// Which group this strategy applies to.
    pub group_id: usize,
    /// Number of threads to use.
    pub num_threads: usize,
    /// Parallelization dimension.
    pub parallel_dim: ParallelDim,
    /// Whether to use thread-local accumulators (needed for reductions).
    pub thread_local_acc: bool,
}

/// The dimension along which a fusion group is parallelized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelDim {
    /// Parallelize over M dimension (row-parallel GEMM).
    M,
    /// Parallelize over N dimension (column-parallel GEMM).
    N,
    /// Parallelize over elements (elementwise / reduction).
    Elements,
    /// Sequential execution (problem too small to parallelize).
    Sequential,
}

/// Compute parallel strategy for each group in a fusion plan.
pub fn plan_parallelism(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    dag: &SemanticDAG,
    profile: &DeviceProfile,
) -> Vec<ParallelStrategy> {
    plan.groups
        .iter()
        .map(|group| plan_group(group, graph, dag, profile))
        .collect()
}

/// Compute parallel strategy for a single fusion group.
fn plan_group(
    group: &FusionGroup,
    graph: &CompilerGraph,
    dag: &SemanticDAG,
    profile: &DeviceProfile,
) -> ParallelStrategy {
    let (num_threads, parallel_dim, thread_local_acc) = match group.mode {
        FusionMode::EpilogueInjection
        | FusionMode::NormIntoGemm
        | FusionMode::QkvSharedInput => {
            plan_gemm_group(group, graph, profile)
        }
        FusionMode::Standalone => {
            // Standalone: check if anchor is a GEMM
            let is_gemm = graph
                .op(group.anchor)
                .map(|op| matches!(op.kind, OpKind::Gemm { .. } | OpKind::GemmBias { .. } | OpKind::QuantGemm { .. }))
                .unwrap_or(false);
            if is_gemm {
                plan_gemm_group(group, graph, profile)
            } else {
                plan_elementwise_group(group, graph, dag, profile)
            }
        }
        FusionMode::LoopFusion => {
            plan_elementwise_group(group, graph, dag, profile)
        }
        FusionMode::TileLevelFusion | FusionMode::ComputeRoot => {
            // Future: dedicated strategies; fall back to GEMM-like for now
            plan_gemm_group(group, graph, profile)
        }
    };

    ParallelStrategy {
        group_id: group.id,
        num_threads,
        parallel_dim,
        thread_local_acc,
    }
}

/// Parallel strategy for GEMM-anchored groups.
///
/// Prefers M-parallel when there are enough rows; falls back to N-parallel
/// or sequential for small problems.
fn plan_gemm_group(
    group: &FusionGroup,
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> (usize, ParallelDim, bool) {
    let anchor_op = match graph.op(group.anchor) {
        Some(op) => op,
        None => return (1, ParallelDim::Sequential, false),
    };

    match &anchor_op.kind {
        OpKind::Gemm { m, n, .. } | OpKind::GemmBias { m, n, .. } | OpKind::QuantGemm { m, n, .. } => {
            let total_work = m * n;
            let threshold = profile.parallel_threshold();

            if total_work < threshold / 4 {
                (1, ParallelDim::Sequential, false)
            } else if *m >= profile.physical_cores * 2 {
                let threads = profile.physical_cores.min(*m);
                (threads, ParallelDim::M, false)
            } else {
                let (_, nr) = profile.microkernel_mr_nr();
                let threads = profile.physical_cores.min(*n / nr.max(1));
                (threads.max(1), ParallelDim::N, false)
            }
        }
        _ => (1, ParallelDim::Sequential, false),
    }
}

/// Parallel strategy for elementwise / standalone groups.
///
/// Parallelizes over elements when the problem is large enough.
/// Enables thread-local accumulators for reduction ops (softmax, norms).
fn plan_elementwise_group(
    group: &FusionGroup,
    graph: &CompilerGraph,
    dag: &SemanticDAG,
    profile: &DeviceProfile,
) -> (usize, ParallelDim, bool) {
    let total_elems = estimate_group_elements(group, graph);
    let threshold = profile.parallel_threshold();

    if total_elems < threshold {
        return (1, ParallelDim::Sequential, false);
    }

    let has_reduction = dag
        .node(group.anchor)
        .map(|n| n.op_class == OpClass::Reduction)
        .unwrap_or(false);

    (profile.physical_cores, ParallelDim::Elements, has_reduction)
}

/// Estimate total elements processed by a group's anchor op.
fn estimate_group_elements(group: &FusionGroup, graph: &CompilerGraph) -> usize {
    if let Some(op) = graph.op(group.anchor) {
        match &op.kind {
            OpKind::Gemm { m, n, .. } | OpKind::GemmBias { m, n, .. } | OpKind::QuantGemm { m, n, .. } => m * n,
            _ => {
                op.outputs
                    .iter()
                    .filter_map(|&tid| graph.tensor(tid))
                    .map(|t| t.shape.iter().product::<usize>())
                    .sum::<usize>()
                    .max(1)
            }
        }
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::CompilerGraph;
    use crate::compiler::ir::LayerIR;
    use crate::compiler::fusion;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::semantic_dag::SemanticDAG;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::{DType, ModelConfig};

    #[test]
    fn test_gemm_parallel_strategy() {
        // Large GEMM should be multi-threaded
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![512, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let out = g.add_tensor("out", vec![512, 4096], dt);
        g.add_op(
            OpKind::Gemm { m: 512, n: 4096, k: 4096 },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry);
        let dag = SemanticDAG::from_graph(&g, &registry);
        let profile = DeviceProfile::detect();

        let strategies = plan_parallelism(&plan, &g, &dag, &profile);
        assert_eq!(strategies.len(), 1);
        assert!(
            strategies[0].num_threads > 1,
            "Large GEMM should use multiple threads, got {}",
            strategies[0].num_threads
        );
        assert!(
            matches!(strategies[0].parallel_dim, ParallelDim::M | ParallelDim::N),
            "GEMM should parallelize over M or N, got {:?}",
            strategies[0].parallel_dim
        );
    }

    #[test]
    fn test_small_elementwise_sequential() {
        // Small elementwise should be sequential
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 64], dt);
        let b = g.add_tensor("b", vec![1, 64], dt);
        g.add_op(OpKind::Silu, vec![a], vec![b], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry);
        let dag = SemanticDAG::from_graph(&g, &registry);
        let profile = DeviceProfile::detect();

        let strategies = plan_parallelism(&plan, &g, &dag, &profile);
        assert_eq!(strategies.len(), 1);
        assert_eq!(strategies[0].num_threads, 1);
        assert_eq!(strategies[0].parallel_dim, ParallelDim::Sequential);
    }

    #[test]
    fn test_llama_parallel_strategies() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&graph, &registry);
        let dag = SemanticDAG::from_graph(&graph, &registry);

        let strategies = plan_parallelism(&plan, &graph, &dag, &profile);

        assert_eq!(strategies.len(), plan.num_groups());

        for s in &strategies {
            // Every group should have at least 1 thread
            assert!(s.num_threads >= 1, "Group {} has 0 threads", s.group_id);
        }

        eprintln!("LLaMA parallel strategies:");
        for s in &strategies {
            eprintln!(
                "  group {} → {} threads, {:?}, tl_acc={}",
                s.group_id, s.num_threads, s.parallel_dim, s.thread_local_acc
            );
        }
    }
}
