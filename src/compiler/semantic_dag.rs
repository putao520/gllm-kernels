//! SemanticDAG — Phase 1 output: CompilerGraph enriched with OpTrace and OpClass.
//!
//! Each node carries its OpTrace (from Phase 0) and auto-derived OpClass
//! (from ComputePattern). This replaces the old hand-maintained OpSemantics mapping.

use crate::compiler::graph::{CompilerGraph, CompilerOp, OpKind, OpId, TensorId};
use crate::compiler::trace::{OpTrace, ComputePattern};
use crate::compiler::registry::ScalarOpRegistry;

/// TVM-style operator classification.
/// Auto-derived from ComputePattern — no manual mapping table needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// Elementwise: vec_add, silu, gelu, relu, vec_mul
    ElemWise,
    /// Injective: rope, reshape, transpose (multi-input/output elementwise)
    Injective,
    /// Reduction: softmax, rms_norm, layer_norm
    Reduction,
    /// Matrix multiply: gemm, gemv
    Gemm,
    /// Opaque: quantized matmul, etc. (does not participate in fusion)
    Opaque,
}

/// Bottleneck classification for roofline analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Mixed,
}

/// A semantic-annotated node in the DAG.
#[derive(Debug, Clone)]
pub struct SemanticNode {
    /// Node ID (matches CompilerGraph op index)
    pub node_id: OpId,
    /// Operator kind (from CompilerGraph)
    pub op_kind: OpKind,
    /// Computation structure (from Phase 0 / registry)
    pub op_trace: Option<OpTrace>,
    /// TVM operator class (auto-derived from op_trace.pattern)
    pub op_class: OpClass,
    /// Bottleneck type
    pub bottleneck: Bottleneck,
    /// Arithmetic intensity (FLOPs / Bytes)
    pub arithmetic_intensity: f32,
    /// Input tensor IDs
    pub inputs: Vec<TensorId>,
    /// Output tensor IDs
    pub outputs: Vec<TensorId>,
    /// Label for debugging
    pub label: String,
}

/// Tensor edge annotation for def-use analysis.
#[derive(Debug, Clone)]
pub struct TensorEdge {
    pub tensor_id: TensorId,
    /// Data size in bytes
    pub data_bytes: usize,
    /// Number of consumer nodes
    pub num_consumers: usize,
    /// Can this tensor be passed via registers (single consumer + elemwise)?
    pub can_register_pass: bool,
}

/// SemanticDAG — CompilerGraph + semantic annotations.
pub struct SemanticDAG {
    /// Semantic nodes (topological order)
    pub nodes: Vec<SemanticNode>,
    /// Tensor edge annotations
    pub edges: Vec<TensorEdge>,
    /// Graph input tensor IDs
    pub graph_inputs: Vec<TensorId>,
    /// Graph output tensor IDs
    pub graph_outputs: Vec<TensorId>,
}

impl SemanticDAG {
    /// Build a SemanticDAG from a CompilerGraph using the given registry.
    pub fn from_graph(graph: &CompilerGraph, registry: &ScalarOpRegistry) -> Self {
        let topo_order = graph.topological_sort();

        let mut nodes = Vec::with_capacity(topo_order.len());
        for &op_id in &topo_order {
            let op = graph.op(op_id).expect("topological_sort returned invalid OpId");
            let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
            let op_trace = registry.get_trace(&key).cloned();
            let op_class = op_trace
                .as_ref()
                .map(|t| Self::derive_op_class(&t.pattern))
                .unwrap_or_else(|| Self::fallback_op_class(&op.kind));

            let (ai, bottleneck) = Self::compute_arithmetic_intensity(op, graph);

            nodes.push(SemanticNode {
                node_id: op.id,
                op_kind: op.kind.clone(),
                op_trace,
                op_class,
                bottleneck,
                arithmetic_intensity: ai,
                inputs: op.inputs.clone(),
                outputs: op.outputs.clone(),
                label: op.label.clone(),
            });
        }

        // Build tensor edges
        let mut edges = Vec::new();
        for tensor in &graph.tensors {
            let data_bytes = tensor.shape.iter().product::<usize>() * tensor.dtype.size_bytes();
            let num_consumers = tensor.consumers.len();
            let can_register_pass = num_consumers == 1; // simplified heuristic
            edges.push(TensorEdge {
                tensor_id: tensor.id,
                data_bytes,
                num_consumers,
                can_register_pass,
            });
        }

        SemanticDAG {
            nodes,
            edges,
            graph_inputs: graph.inputs.clone(),
            graph_outputs: graph.outputs.clone(),
        }
    }

    /// ComputePattern → OpClass auto-derivation (SPEC §12.2 table)
    pub fn derive_op_class(pattern: &ComputePattern) -> OpClass {
        match pattern {
            ComputePattern::Elementwise { .. } => OpClass::ElemWise,
            ComputePattern::BinaryElementwise { .. } => OpClass::ElemWise,
            ComputePattern::Injective { .. } => OpClass::Injective,
            ComputePattern::Reduction { .. } => OpClass::Reduction,
            ComputePattern::NormLike { .. } => OpClass::Reduction,
            ComputePattern::Gemm => OpClass::Gemm,
            ComputePattern::QuantDecode { .. } => OpClass::Opaque,
        }
    }

    /// Fallback classification when no OpTrace is available.
    fn fallback_op_class(kind: &OpKind) -> OpClass {
        match kind {
            OpKind::Silu | OpKind::Gelu | OpKind::Add | OpKind::Mul | OpKind::Residual => {
                OpClass::ElemWise
            }
            OpKind::SwiGlu | OpKind::GeGlu => OpClass::ElemWise,
            OpKind::RoPE { .. } | OpKind::Transpose { .. } | OpKind::Reshape { .. } => {
                OpClass::Injective
            }
            OpKind::Softmax | OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => {
                OpClass::Reduction
            }
            OpKind::Gemm { .. } | OpKind::GemmBias { .. } | OpKind::QuantGemm { .. } => {
                OpClass::Gemm
            }
            OpKind::Dequantize { .. } => OpClass::ElemWise,
        }
    }

    /// Compute arithmetic intensity and bottleneck classification.
    fn compute_arithmetic_intensity(
        op: &CompilerOp,
        graph: &CompilerGraph,
    ) -> (f32, Bottleneck) {
        // Total input + output bytes
        let mut total_bytes = 0usize;
        for &tid in op.inputs.iter().chain(op.outputs.iter()) {
            if let Some(t) = graph.tensor(tid) {
                total_bytes += t.shape.iter().product::<usize>() * t.dtype.size_bytes();
            }
        }
        if total_bytes == 0 {
            return (0.0, Bottleneck::Memory);
        }

        // Estimate FLOPs
        let flops: usize = match &op.kind {
            OpKind::Gemm { m, n, k } => 2 * m * n * k,
            OpKind::GemmBias { m, n, k } => 2 * m * n * k + m * n,
            OpKind::QuantGemm { m, n, k, .. } => 2 * m * n * k,
            OpKind::Silu | OpKind::Gelu => {
                // ~10 FLOPs per element (exp + div + mul)
                op.outputs
                    .iter()
                    .filter_map(|&tid| graph.tensor(tid))
                    .map(|t| t.shape.iter().product::<usize>() * 10)
                    .sum()
            }
            OpKind::RmsNorm { .. } => {
                // 2 passes: sum_sq (2 flops/elem) + scale (3 flops/elem) + sqrt
                op.outputs
                    .iter()
                    .filter_map(|&tid| graph.tensor(tid))
                    .map(|t| t.shape.iter().product::<usize>() * 5)
                    .sum()
            }
            _ => {
                // Default: 1-2 FLOPs per element
                op.outputs
                    .iter()
                    .filter_map(|&tid| graph.tensor(tid))
                    .map(|t| t.shape.iter().product::<usize>() * 2)
                    .sum()
            }
        };

        let ai = flops as f32 / total_bytes as f32;
        let bottleneck = if ai > 8.0 {
            Bottleneck::Compute
        } else if ai < 2.0 {
            Bottleneck::Memory
        } else {
            Bottleneck::Mixed
        };

        (ai, bottleneck)
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get node by OpId.
    pub fn node(&self, id: OpId) -> Option<&SemanticNode> {
        self.nodes.iter().find(|n| n.node_id == id)
    }

    /// Get all nodes with a given OpClass.
    pub fn nodes_by_class(&self, class: OpClass) -> Vec<&SemanticNode> {
        self.nodes.iter().filter(|n| n.op_class == class).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::CompilerGraph;
    use crate::compiler::ir::LayerIR;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::trace::{ComputePattern, TraceOp};
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_derive_op_class_from_pattern() {
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::Elementwise {
                body: vec![TraceOp::Input(0)]
            }),
            OpClass::ElemWise
        );
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::BinaryElementwise { body: vec![] }),
            OpClass::ElemWise
        );
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::Injective {
                body: vec![],
                num_inputs: 4,
                num_outputs: 2
            }),
            OpClass::Injective
        );
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::Reduction {
                identity: 0.0,
                combine: vec![]
            }),
            OpClass::Reduction
        );
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::NormLike {
                reduce: vec![],
                finalize: vec![],
                transform: vec![]
            }),
            OpClass::Reduction
        );
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::Gemm),
            OpClass::Gemm
        );
        assert_eq!(
            SemanticDAG::derive_op_class(&ComputePattern::QuantDecode {
                block_size: 32,
                decode: vec![]
            }),
            OpClass::Opaque
        );
    }

    #[test]
    fn test_semantic_dag_from_llama_graph() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();

        let dag = SemanticDAG::from_graph(&graph, &registry);

        // Should have same number of nodes as ops
        assert_eq!(dag.num_nodes(), graph.num_ops());

        // Should have GEMM nodes
        let gemm_nodes = dag.nodes_by_class(OpClass::Gemm);
        assert!(
            gemm_nodes.len() >= 7,
            "LLaMA should have >=7 GEMM ops, got {}",
            gemm_nodes.len()
        );

        // Should have Reduction nodes (RmsNorm, Softmax)
        let reduction_nodes = dag.nodes_by_class(OpClass::Reduction);
        assert!(
            reduction_nodes.len() >= 2,
            "Should have >=2 reduction ops, got {}",
            reduction_nodes.len()
        );

        // Should have ElemWise nodes (SwiGLU, Residual, etc.)
        let elemwise_nodes = dag.nodes_by_class(OpClass::ElemWise);
        assert!(!elemwise_nodes.is_empty());

        // GEMM nodes should have positive arithmetic intensity
        // (batch=1 GEMMs are GEMV-like with AI < 1.0, so we just check > 0)
        for node in &gemm_nodes {
            assert!(
                node.arithmetic_intensity > 0.0,
                "GEMM AI should be positive, got {}",
                node.arithmetic_intensity
            );
        }

        eprintln!("SemanticDAG: {} nodes", dag.num_nodes());
        for node in &dag.nodes {
            eprintln!(
                "  [{:>2}] {:?} {:?} AI={:.1} {:?}",
                node.node_id.0, node.op_class, node.bottleneck,
                node.arithmetic_intensity, node.label
            );
        }
    }

    #[test]
    fn test_semantic_dag_all_nodes_have_trace() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();

        let dag = SemanticDAG::from_graph(&graph, &registry);

        // All nodes should have an OpTrace (since registry has defaults for all ops)
        for node in &dag.nodes {
            assert!(
                node.op_trace.is_some(),
                "Node {} ({}) missing OpTrace",
                node.node_id.0, node.label
            );
        }
    }

    #[test]
    fn test_tensor_edges() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();

        let dag = SemanticDAG::from_graph(&graph, &registry);

        assert_eq!(dag.edges.len(), graph.num_tensors());

        // All edges should have positive data_bytes
        for edge in &dag.edges {
            assert!(edge.data_bytes > 0, "Edge {:?} has 0 bytes", edge.tensor_id);
        }
    }
}
