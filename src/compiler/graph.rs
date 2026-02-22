//! CompilerGraph — DAG representation for the JIT inference compiler.
//!
//! The graph captures the computation of a single transformer layer as a
//! directed acyclic graph of typed operations. Each operation reads from
//! input tensors and produces output tensors. Tensors carry shape metadata
//! and def-use chains (single producer, multiple consumers).
//!
//! Pipeline: LayerIR → CompilerGraph → (Phase 2: fusion) → (Phase 3: codegen)

use std::collections::HashMap;
use crate::compiler::ir::LayerIR;
use crate::dispatch::device_profile::DeviceProfile;
use crate::inference::types::DType;
use crate::traits::Activation;

// ── Identifiers ────────────────────────────────────────────────────

/// Unique tensor identifier within a CompilerGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

/// Unique operation identifier within a CompilerGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub u32);

// ── Tensor metadata ────────────────────────────────────────────────

/// Shape and type metadata for a tensor in the graph.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub id: TensorId,
    /// Symbolic shape dimensions (0 = batch, dynamic at runtime).
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// The op that produces this tensor (None for graph inputs).
    pub producer: Option<OpId>,
    /// Ops that consume this tensor.
    pub consumers: Vec<OpId>,
    /// Human-readable name for debugging.
    pub name: String,
}

// ── Operation kinds ────────────────────────────────────────────────

/// The set of operations the compiler graph can represent.
#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    // ── Normalization ──
    RmsNorm { eps: f32 },
    LayerNorm { eps: f32 },

    // ── Linear algebra ──
    /// C = A × B  (row-major, A is [M,K], B is [K,N], C is [M,N])
    Gemm { m: usize, n: usize, k: usize },
    /// C = A × B + bias
    GemmBias { m: usize, n: usize, k: usize },

    // ── Activations ──
    Silu,
    Gelu,
    /// SwiGLU: silu(gate) * up
    SwiGlu,
    /// GeGLU: gelu(gate) * up
    GeGlu,

    // ── Attention ──
    Softmax,
    /// Rotary position embedding (non-interleaved).
    RoPE { head_dim: usize, theta: f64 },

    // ── Elementwise ──
    Add,
    Mul,
    /// Residual connection: out = x + residual
    Residual,

    // ── Layout ──
    Transpose { perm: Vec<usize> },
    Reshape { target_shape: Vec<usize> },
}

// ── Compiler operation (graph node) ────────────────────────────────

/// A single operation in the compiler graph.
#[derive(Debug, Clone)]
pub struct CompilerOp {
    pub id: OpId,
    pub kind: OpKind,
    /// Input tensor IDs (order matters: matches OpKind semantics).
    pub inputs: Vec<TensorId>,
    /// Output tensor IDs.
    pub outputs: Vec<TensorId>,
    /// Optional label for debugging / visualization.
    pub label: String,
}

// ── CompilerGraph ──────────────────────────────────────────────────

/// DAG of operations for a single transformer layer.
///
/// Tensors are SSA-like: each tensor has exactly one producer (or is a
/// graph input) and zero or more consumers. This makes def-use analysis
/// trivial and enables clean fusion decisions.
#[derive(Debug, Clone)]
pub struct CompilerGraph {
    pub ops: Vec<CompilerOp>,
    pub tensors: Vec<TensorMeta>,
    /// Graph input tensor IDs (layer input, weights, etc.)
    pub inputs: Vec<TensorId>,
    /// Graph output tensor IDs (layer output)
    pub outputs: Vec<TensorId>,
    next_tensor_id: u32,
    next_op_id: u32,
}

impl CompilerGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        CompilerGraph {
            ops: Vec::new(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_tensor_id: 0,
            next_op_id: 0,
        }
    }

    /// Allocate a new tensor with the given shape and dtype.
    pub fn add_tensor(&mut self, name: &str, shape: Vec<usize>, dtype: DType) -> TensorId {
        let id = TensorId(self.next_tensor_id);
        self.next_tensor_id += 1;
        self.tensors.push(TensorMeta {
            id,
            shape,
            dtype,
            producer: None,
            consumers: Vec::new(),
            name: name.to_string(),
        });
        id
    }

    /// Add an operation to the graph. Updates def-use chains automatically.
    pub fn add_op(
        &mut self,
        kind: OpKind,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        label: &str,
    ) -> OpId {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;

        // Update def-use: mark this op as producer of its outputs
        for &tid in &outputs {
            if let Some(t) = self.tensor_mut(tid) {
                t.producer = Some(id);
            }
        }
        // Update def-use: mark this op as consumer of its inputs
        for &tid in &inputs {
            if let Some(t) = self.tensor_mut(tid) {
                t.consumers.push(id);
            }
        }

        self.ops.push(CompilerOp {
            id,
            kind,
            inputs,
            outputs,
            label: label.to_string(),
        });
        id
    }

    /// Get tensor metadata by ID.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.id == id)
    }

    /// Get mutable tensor metadata by ID.
    fn tensor_mut(&mut self, id: TensorId) -> Option<&mut TensorMeta> {
        self.tensors.iter_mut().find(|t| t.id == id)
    }

    /// Get operation by ID.
    pub fn op(&self, id: OpId) -> Option<&CompilerOp> {
        self.ops.iter().find(|o| o.id == id)
    }

    /// Number of operations.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Build def-use chains: TensorId → (producer OpId, consumer OpIds).
    pub fn def_use_chains(&self) -> HashMap<TensorId, (Option<OpId>, Vec<OpId>)> {
        let mut chains = HashMap::new();
        for t in &self.tensors {
            chains.insert(t.id, (t.producer, t.consumers.clone()));
        }
        chains
    }

    /// Topological sort of operations (Kahn's algorithm).
    ///
    /// Returns ops in dependency order: an op appears only after all ops
    /// that produce its input tensors. Panics if the graph has cycles.
    pub fn topological_sort(&self) -> Vec<OpId> {
        let n = self.ops.len();
        if n == 0 {
            return Vec::new();
        }

        // Build in-degree map: for each op, count how many of its input
        // tensors are produced by other ops in the graph.
        let mut in_degree: HashMap<OpId, usize> = HashMap::new();
        let mut adj: HashMap<OpId, Vec<OpId>> = HashMap::new();

        for op in &self.ops {
            in_degree.entry(op.id).or_insert(0);
            adj.entry(op.id).or_insert_with(Vec::new);
        }

        for op in &self.ops {
            for &input_tid in &op.inputs {
                if let Some(t) = self.tensor(input_tid) {
                    if let Some(producer_id) = t.producer {
                        // producer_id → op.id edge
                        adj.entry(producer_id).or_default().push(op.id);
                        *in_degree.entry(op.id).or_insert(0) += 1;
                    }
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<OpId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        // Sort for deterministic output
        queue.sort_by_key(|id| id.0);

        let mut result = Vec::with_capacity(n);
        let mut head = 0;

        while head < queue.len() {
            let current = queue[head];
            head += 1;
            result.push(current);

            if let Some(neighbors) = adj.get(&current) {
                for &next in neighbors {
                    let deg = in_degree.get_mut(&next).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(next);
                    }
                }
            }
        }

        assert_eq!(
            result.len(),
            n,
            "CompilerGraph has a cycle! sorted {} of {} ops",
            result.len(),
            n
        );
        result
    }

    /// Lower a `LayerIR` (Decoder architecture) into a CompilerGraph.
    ///
    /// Produces the standard LLaMA-style decoder graph:
    /// ```text
    /// input → RmsNorm₁ → Q/K/V GEMMs → RoPE → Attention → O GEMM → Residual₁
    ///       → RmsNorm₂ → Gate GEMM → SwiGLU(↑Up GEMM) → Down GEMM → Residual₂
    /// ```
    pub fn from_layer_ir(ir: &LayerIR, _profile: &DeviceProfile) -> Self {
        let mut g = CompilerGraph::new();
        let dt = ir.dtype;
        let b = ir.max_batch;
        let h = ir.hidden;
        let q_dim = ir.q_dim();
        let kv_dim = ir.kv_dim();
        let inter = ir.intermediate;

        // ── Graph inputs ──
        let input = g.add_tensor("input", vec![b, h], dt);
        let w_norm1 = g.add_tensor("w_rms_norm1", vec![h], dt);
        let w_q = g.add_tensor("w_q", vec![h, q_dim], dt);
        let w_k = g.add_tensor("w_k", vec![h, kv_dim], dt);
        let w_v = g.add_tensor("w_v", vec![h, kv_dim], dt);
        let w_o = g.add_tensor("w_o", vec![q_dim, h], dt);
        let w_norm2 = g.add_tensor("w_rms_norm2", vec![h], dt);
        let w_gate = g.add_tensor("w_gate", vec![h, inter], dt);
        let w_up = g.add_tensor("w_up", vec![h, inter], dt);
        let w_down = g.add_tensor("w_down", vec![inter, h], dt);
        let cos_sin = g.add_tensor("cos_sin", vec![ir.head_dim / 2], dt);

        g.inputs = vec![input, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down, cos_sin];

        // ── Phase 1: Attention ──

        // RmsNorm₁
        let normed1 = g.add_tensor("normed1", vec![b, h], dt);
        g.add_op(
            OpKind::RmsNorm { eps: ir.rms_eps },
            vec![input, w_norm1],
            vec![normed1],
            "rms_norm_1",
        );

        // Q projection: [B, H] × [H, Q] → [B, Q]
        let q_out = g.add_tensor("q", vec![b, q_dim], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: q_dim, k: h },
            vec![normed1, w_q],
            vec![q_out],
            "gemm_q",
        );

        // K projection: [B, H] × [H, KV] → [B, KV]
        let k_out = g.add_tensor("k", vec![b, kv_dim], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: kv_dim, k: h },
            vec![normed1, w_k],
            vec![k_out],
            "gemm_k",
        );

        // V projection: [B, H] × [H, KV] → [B, KV]
        let v_out = g.add_tensor("v", vec![b, kv_dim], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: kv_dim, k: h },
            vec![normed1, w_v],
            vec![v_out],
            "gemm_v",
        );

        // RoPE on Q
        let q_rope = g.add_tensor("q_rope", vec![b, q_dim], dt);
        g.add_op(
            OpKind::RoPE { head_dim: ir.head_dim, theta: ir.rope_theta },
            vec![q_out, cos_sin],
            vec![q_rope],
            "rope_q",
        );

        // RoPE on K
        let k_rope = g.add_tensor("k_rope", vec![b, kv_dim], dt);
        g.add_op(
            OpKind::RoPE { head_dim: ir.head_dim, theta: ir.rope_theta },
            vec![k_out, cos_sin],
            vec![k_rope],
            "rope_k",
        );

        // Attention: softmax(Q·K^T / √d) · V → [B, Q]
        // Represented as a single opaque Softmax node for now.
        // Phase 2 fusion will expand this into FlashAttention tiling.
        let attn_scores = g.add_tensor("attn_scores", vec![b, ir.num_heads, b], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: b, k: ir.head_dim },
            vec![q_rope, k_rope],
            vec![attn_scores],
            "attn_qk",
        );

        let attn_probs = g.add_tensor("attn_probs", vec![b, ir.num_heads, b], dt);
        g.add_op(
            OpKind::Softmax,
            vec![attn_scores],
            vec![attn_probs],
            "attn_softmax",
        );

        let attn_out = g.add_tensor("attn_out", vec![b, q_dim], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: ir.head_dim, k: b },
            vec![attn_probs, v_out],
            vec![attn_out],
            "attn_v",
        );

        // O projection: [B, Q] × [Q, H] → [B, H]
        let o_out = g.add_tensor("o_proj", vec![b, h], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: h, k: q_dim },
            vec![attn_out, w_o],
            vec![o_out],
            "gemm_o",
        );

        // Residual₁: input + o_out
        let resid1 = g.add_tensor("residual1", vec![b, h], dt);
        g.add_op(
            OpKind::Residual,
            vec![input, o_out],
            vec![resid1],
            "residual_1",
        );

        // ── Phase 2: FFN ──

        // RmsNorm₂
        let normed2 = g.add_tensor("normed2", vec![b, h], dt);
        g.add_op(
            OpKind::RmsNorm { eps: ir.rms_eps },
            vec![resid1, w_norm2],
            vec![normed2],
            "rms_norm_2",
        );

        // Gate GEMM: [B, H] × [H, Inter] → [B, Inter]
        let gate_out = g.add_tensor("gate", vec![b, inter], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: inter, k: h },
            vec![normed2, w_gate],
            vec![gate_out],
            "gemm_gate",
        );

        // Up GEMM: [B, H] × [H, Inter] → [B, Inter]
        let up_out = g.add_tensor("up", vec![b, inter], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: inter, k: h },
            vec![normed2, w_up],
            vec![up_out],
            "gemm_up",
        );

        // SwiGLU / GeGLU fusion
        let ffn_act = g.add_tensor("ffn_act", vec![b, inter], dt);
        let act_kind = match ir.activation {
            Activation::GeGlu => OpKind::GeGlu,
            _ => OpKind::SwiGlu,
        };
        g.add_op(
            act_kind,
            vec![gate_out, up_out],
            vec![ffn_act],
            "swiglu",
        );

        // Down GEMM: [B, Inter] × [Inter, H] → [B, H]
        let down_out = g.add_tensor("down", vec![b, h], dt);
        g.add_op(
            OpKind::Gemm { m: b, n: h, k: inter },
            vec![ffn_act, w_down],
            vec![down_out],
            "gemm_down",
        );

        // Residual₂: resid1 + down_out
        let output = g.add_tensor("output", vec![b, h], dt);
        g.add_op(
            OpKind::Residual,
            vec![resid1, down_out],
            vec![output],
            "residual_2",
        );

        g.outputs = vec![output];
        g
    }
}

impl Default for CompilerGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CompilerGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CompilerGraph: {} ops, {} tensors", self.ops.len(), self.tensors.len())?;
        for op in &self.ops {
            let ins: Vec<String> = op.inputs.iter().map(|t| {
                self.tensor(*t).map(|m| m.name.as_str()).unwrap_or("?").to_string()
            }).collect();
            let outs: Vec<String> = op.outputs.iter().map(|t| {
                self.tensor(*t).map(|m| m.name.as_str()).unwrap_or("?").to_string()
            }).collect();
            writeln!(f, "  [{:>2}] {} : ({}) → ({})",
                op.id.0, op.label, ins.join(", "), outs.join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::LayerIR;
    use crate::inference::types::ModelConfig;
    use crate::dispatch::DeviceProfile;

    #[test]
    fn test_empty_graph() {
        let g = CompilerGraph::new();
        assert_eq!(g.num_ops(), 0);
        assert_eq!(g.num_tensors(), 0);
        assert!(g.topological_sort().is_empty());
    }

    #[test]
    fn test_simple_chain() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let c = g.add_tensor("c", vec![1, 4096], dt);

        let op0 = g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        let op1 = g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");

        let sorted = g.topological_sort();
        assert_eq!(sorted, vec![op0, op1]);
    }

    #[test]
    fn test_diamond_dag() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor("in", vec![1, 4096], dt);
        let left = g.add_tensor("left", vec![1, 4096], dt);
        let right = g.add_tensor("right", vec![1, 4096], dt);
        let out = g.add_tensor("out", vec![1, 4096], dt);

        let op_l = g.add_op(OpKind::Silu, vec![input], vec![left], "left");
        let op_r = g.add_op(OpKind::Gelu, vec![input], vec![right], "right");
        let op_add = g.add_op(OpKind::Add, vec![left, right], vec![out], "add");

        let sorted = g.topological_sort();
        // op_l and op_r must come before op_add
        let pos_l = sorted.iter().position(|&x| x == op_l).unwrap();
        let pos_r = sorted.iter().position(|&x| x == op_r).unwrap();
        let pos_add = sorted.iter().position(|&x| x == op_add).unwrap();
        assert!(pos_l < pos_add);
        assert!(pos_r < pos_add);
    }

    #[test]
    fn test_def_use_chains() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 4096], dt);
        let b = g.add_tensor("b", vec![1, 4096], dt);
        let c = g.add_tensor("c", vec![1, 4096], dt);

        let op0 = g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        let op1 = g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");

        let chains = g.def_use_chains();
        // 'a' is a graph input (no producer), consumed by op0
        assert_eq!(chains[&a].0, None);
        assert_eq!(chains[&a].1, vec![op0]);
        // 'b' produced by op0, consumed by op1
        assert_eq!(chains[&b].0, Some(op0));
        assert_eq!(chains[&b].1, vec![op1]);
        // 'c' produced by op1, no consumers
        assert_eq!(chains[&c].0, Some(op1));
        assert!(chains[&c].1.is_empty());
    }

    #[test]
    fn test_from_layer_ir_decoder() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let g = CompilerGraph::from_layer_ir(&ir, &profile);

        eprintln!("{g}");

        // Should have a reasonable number of ops
        assert!(g.num_ops() >= 14, "expected ≥14 ops, got {}", g.num_ops());
        // Should have inputs and outputs
        assert!(!g.inputs.is_empty());
        assert!(!g.outputs.is_empty());
        // Topological sort should succeed (no cycles)
        let sorted = g.topological_sort();
        assert_eq!(sorted.len(), g.num_ops());
    }

    #[test]
    fn test_from_layer_ir_gemma_geglu() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let g = CompilerGraph::from_layer_ir(&ir, &profile);

        // Should contain a GeGlu op
        let has_geglu = g.ops.iter().any(|op| matches!(op.kind, OpKind::GeGlu));
        assert!(has_geglu, "Gemma graph should have GeGlu op");
    }
}
