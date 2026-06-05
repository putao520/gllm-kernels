impl CompilerGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        CompilerGraph {
            ops: Vec::new(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            telemetry: EpilogueTelemetryConfig::default(),
            layer_loop_config: None,
            hetero_layer_loop_config: None,
            embedding_scale: None,
            custom_weight_layout: None,
            quant_weight_bytes: HashMap::new(),
            max_seq_len: 2048,
            kv_load_mode: None,
            next_tensor_id: 0,
            next_op_id: 0,
        }
    }

    /// Register physical byte size for a quantized weight tensor.
    /// Called by graph builders when QuantGemm ops are created — the tensor's
    /// dtype is F32 but its physical data is smaller (e.g., Q8_0: N*(K/32)*34).
    pub fn set_quant_weight_bytes(&mut self, tid: TensorId, bytes: usize) {
        self.quant_weight_bytes.insert(tid, bytes);
    }

    /// Allocate a new tensor with a symbolic shape (`Vec<SymDim>`).
    ///
    /// Use this when one or more dimensions are dynamic (e.g. `total_seq`).
    /// For fully-concrete shapes prefer `add_tensor_concrete`.
    pub fn add_tensor(&mut self, name: &str, shape: Vec<SymDim>, dtype: DType) -> TensorId {
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

    /// Allocate a new tensor with a fully-concrete `&[usize]` shape.
    ///
    /// Convenience wrapper over `add_tensor` — converts each `usize` to
    /// `SymDim::Concrete`. Use this for all static dimensions.
    pub fn add_tensor_concrete(&mut self, name: &str, shape: &[usize], dtype: DType) -> TensorId {
        let sym_shape: Vec<SymDim> = shape.iter().map(|&d| SymDim::Concrete(d)).collect();
        self.add_tensor(name, sym_shape, dtype)
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
            guard: LayerCondition::Always,
        });
        id
    }

    /// Add an operation with a layer execution guard.
    /// Ops inside a layer template can be conditionally skipped at runtime
    /// based on `layer_loop_counter`. See `LayerCondition` for semantics.
    pub fn add_op_guarded(
        &mut self,
        kind: OpKind,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        label: &str,
        guard: LayerCondition,
    ) -> OpId {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;

        for &tid in &outputs {
            if let Some(t) = self.tensor_mut(tid) {
                t.producer = Some(id);
            }
        }
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
            guard,
        });
        id
    }

    /// Get tensor metadata by ID.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.id == id)
    }

    /// Total number of elements in a tensor (product of concrete shape dimensions).
    ///
    /// Symbolic dimensions are treated as 0 (unknown at compile time), so the
    /// product will be 0 for tensors with any symbolic dimension. Callers that
    /// need the true element count at runtime must resolve via `ShapeBinding`.
    pub fn tensor_numel(&self, id: TensorId) -> Option<usize> {
        self.tensor(id).map(|t| {
            let product: usize = t.shape.iter().map(|d| match d {
                SymDim::Concrete(v) => *v,
                SymDim::Symbolic { .. } => 0,
            }).product();
            product.max(1)
        })
    }

    /// Total number of elements for allocation purposes.
    ///
    /// Unlike `tensor_numel` which treats symbolic dims as 0, this uses
    /// `max_for_allocation` so symbolic dims contribute their `max_value`
    /// (or `default` if no max is specified). Used by `weight_layout` to
    /// allocate buffers large enough for the compiled loop bounds.
    pub fn tensor_numel_for_alloc(&self, id: TensorId, default_max: usize) -> Option<usize> {
        self.tensor(id).map(|t| {
            let product: usize = t.shape.iter()
                .map(|d| d.max_for_allocation(default_max))
                .product();
            product.max(1)
        })
    }

    /// Total number of elements with symbolic dims resolved via `binding`.
    pub fn tensor_numel_resolved(&self, id: TensorId, binding: &ShapeBinding) -> Option<Result<usize, CompilerError>> {
        self.tensor(id).map(|t| {
            let mut product = 1usize;
            for d in &t.shape {
                product *= d.resolve(binding)?;
            }
            Ok(product.max(1))
        })
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

    /// Set telemetry configuration
    pub fn set_telemetry(&mut self, config: EpilogueTelemetryConfig) {
        self.telemetry = config;
    }

    /// Infer the computation dtype from the graph.
    ///
    /// Scans Gemm/GemmBias ops for an explicit dtype field. Falls back to
    /// the dtype of the first input tensor if no GEMM ops exist.
    /// ARCH-DTYPE-FULLCHAIN-ORCH: replaces `unwrap_or(DType::F32)` pattern.
    pub fn infer_computation_dtype(&self) -> DType {
        // Priority 1: explicit dtype from GEMM ops
        if let Some(dt) = self.ops.iter().find_map(|op| match &op.kind {
            OpKind::Gemm { dtype, .. } | OpKind::GemmBias { dtype, .. } => Some(*dtype),
            _ => None,
        }) {
            return dt;
        }
        // Priority 2: dtype of first input tensor
        if let Some(&tid) = self.inputs.first() {
            if let Some(t) = self.tensor(tid) {
                return t.dtype;
            }
        }
        // Last resort (should not happen for well-formed graphs)
        DType::F32
    }

    /// Compute weight blob layout: skip inputs[0] (activation input),
    /// remaining inputs packed contiguously in order.
    /// Uses `storage_dtype` for byte size calculations (matches `pack_weights()` format).
    ///
    /// Symbolic dimensions use their `max_value` for allocation sizing so the
    /// weight blob is large enough for the compiled loop bounds.
    ///
    /// When `custom_weight_layout` is set (mega-kernel case), returns that instead.
    pub fn weight_layout(&self) -> WeightLayout {
        if let Some(ref custom) = self.custom_weight_layout {
            return custom.clone();
        }
        let mut offsets = Vec::new();
        let mut cursor = 0usize;
        for &tid in self.inputs.iter().skip(1) {
            offsets.push((tid, cursor));
            if let Some(&qb) = self.quant_weight_bytes.get(&tid) {
                cursor += qb;
            } else {
                let numel = self.tensor_numel_for_alloc(tid, self.max_seq_len).unwrap_or(0);
                let dtype_size = self.tensor(tid)
                    .map(|t| t.dtype.size_bytes()).unwrap_or(DType::F32.size_bytes());
                cursor += numel * dtype_size;
            }
        }
        WeightLayout { offsets, total_bytes: cursor }
    }

    /// Set a custom weight layout override (used by mega-kernel).
    pub fn set_custom_weight_layout(&mut self, layout: WeightLayout) {
        self.custom_weight_layout = Some(layout);
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
            adj.entry(op.id).or_default();
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
                    // SAFETY: all nodes were inserted into in_degree during the initial loop
                    let deg = in_degree.get_mut(&next)
                        .expect("SAFETY: all graph nodes are in in_degree map");
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

    /// Lower a `LayerIR` into a CompilerGraph.
    ///
    /// Dispatches to architecture-specific builders:
    /// - Decoder: RmsNorm → Attn(QKV+RoPE+GQA+O) → Residual → RmsNorm → FFN(gate+up+SiLU+down) → Residual
    /// - Encoder: LayerNorm → Attn(QKV+O) → Residual → LayerNorm → FFN(up+GELU+down) → Residual → MeanPool → L2Normalize
    pub fn from_layer_ir(ir: &LayerIR, _profile: &DeviceProfile) -> Result<Self, CompilerError> {
        match &ir.arch {
            LayerArch::Encoder => return Self::from_layer_ir_encoder(ir),
            LayerArch::DecoderMoE { .. } => {
                return Err("MoE decoder not yet supported in from_layer_ir".into());
            }
            LayerArch::Decoder => {} // fall through to existing decoder logic
        }
        let mut g = CompilerGraph::new();
        let dt = ir.dtype;
        let b = ir.max_batch;
        let h = ir.hidden;
        let q_dim = ir.q_dim();
        let kv_dim = ir.kv_dim();
        let inter = ir.intermediate;

        // ── Graph inputs ──
        let input = g.add_tensor_concrete("input", &[b, h], dt);
        let w_norm1 = g.add_tensor_concrete("w_rms_norm1", &[h], dt);
        let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
        let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
        let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
        let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
        let w_norm2 = g.add_tensor_concrete("w_rms_norm2", &[h], dt);
        let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
        let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
        let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);
        let cos_sin = g.add_tensor_concrete("cos_sin", &[ir.head_dim / 2], dt);

        g.inputs = vec![input, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down, cos_sin];

        // ── Phase 1: Attention ──

        // RmsNorm₁
        let normed1 = g.add_tensor_concrete("normed1", &[b, h], dt);
        g.add_op(
            OpKind::RmsNorm { eps: ir.rms_eps },
            vec![input, w_norm1],
            vec![normed1],
            "rms_norm_1",
        );

        // Q projection: [B, H] × [H, Q] → [B, Q]
        let q_out = g.add_tensor_concrete("q", &[b, q_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: q_dim, k: h, dtype: dt, trans_b: false },
            vec![normed1, w_q],
            vec![q_out],
            "gemm_q",
        );

        // K projection: [B, H] × [H, KV] → [B, KV]
        let k_out = g.add_tensor_concrete("k", &[b, kv_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: kv_dim, k: h, dtype: dt, trans_b: false },
            vec![normed1, w_k],
            vec![k_out],
            "gemm_k",
        );

        // V projection: [B, H] × [H, KV] → [B, KV]
        let v_out = g.add_tensor_concrete("v", &[b, kv_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: kv_dim, k: h, dtype: dt, trans_b: false },
            vec![normed1, w_v],
            vec![v_out],
            "gemm_v",
        );

        // RoPE on Q
        let q_rope = g.add_tensor_concrete("q_rope", &[b, q_dim], dt);
        g.add_op(
            OpKind::RoPE { num_heads: ir.num_heads, head_dim: ir.head_dim, theta: ir.rope_theta, partial: ir.partial_rotary_factor, rope_scaling: None },
            vec![q_out, cos_sin],
            vec![q_rope],
            "rope_q",
        );

        // RoPE on K
        let k_rope = g.add_tensor_concrete("k_rope", &[b, kv_dim], dt);
        g.add_op(
            OpKind::RoPE { num_heads: ir.num_kv_heads, head_dim: ir.head_dim, theta: ir.rope_theta, partial: ir.partial_rotary_factor, rope_scaling: None },
            vec![k_out, cos_sin],
            vec![k_rope],
            "rope_k",
        );

        // Attention: softmax(Q·K^T / √d) · V
        //
        // Simplified per-head modeling: `b` here represents seq_len (the
        // token dimension), not the batch dimension. The GEMMs below model
        // a single attention head's computation:
        //   Q·K^T : [seq_len, head_dim] × [head_dim, seq_len] → [seq_len, seq_len]
        //   A·V   : [seq_len, seq_len] × [seq_len, head_dim] → [seq_len, head_dim]
        // Multi-head parallelism is implicit (num_heads independent heads).
        // Phase 2 fusion will expand this into FlashAttention tiling.
        let attn_scores = g.add_tensor_concrete("attn_scores", &[b, ir.num_heads, b], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: b, k: ir.head_dim, dtype: dt, trans_b: false },
            vec![q_rope, k_rope],
            vec![attn_scores],
            "attn_qk",
        );

        let attn_probs = g.add_tensor_concrete("attn_probs", &[b, ir.num_heads, b], dt);
        g.add_op(
            OpKind::Softmax,
            vec![attn_scores],
            vec![attn_probs],
            "attn_softmax",
        );

        let attn_out = g.add_tensor_concrete("attn_out", &[b, q_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: ir.head_dim, k: b, dtype: dt, trans_b: false },
            vec![attn_probs, v_out],
            vec![attn_out],
            "attn_v",
        );

        // O projection: [B, Q] × [Q, H] → [B, H]
        let o_out = g.add_tensor_concrete("o_proj", &[b, h], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: h, k: q_dim, dtype: dt, trans_b: false },
            vec![attn_out, w_o],
            vec![o_out],
            "gemm_o",
        );

        // Residual₁: input + o_out
        let resid1 = g.add_tensor_concrete("residual1", &[b, h], dt);
        g.add_op(
            OpKind::Residual,
            vec![input, o_out],
            vec![resid1],
            "residual_1",
        );

        // ── Phase 2: FFN ──

        // RmsNorm₂
        let normed2 = g.add_tensor_concrete("normed2", &[b, h], dt);
        g.add_op(
            OpKind::RmsNorm { eps: ir.rms_eps },
            vec![resid1, w_norm2],
            vec![normed2],
            "rms_norm_2",
        );

        // Gate GEMM: [B, H] × [H, Inter] → [B, Inter]
        let gate_out = g.add_tensor_concrete("gate", &[b, inter], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: inter, k: h, dtype: dt, trans_b: false },
            vec![normed2, w_gate],
            vec![gate_out],
            "gemm_gate",
        );

        // Up GEMM: [B, H] × [H, Inter] → [B, Inter]
        let up_out = g.add_tensor_concrete("up", &[b, inter], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: inter, k: h, dtype: dt, trans_b: false },
            vec![normed2, w_up],
            vec![up_out],
            "gemm_up",
        );

        // Gated activation fusion: activation(gate) * up
        let ffn_act = g.add_tensor_concrete("ffn_act", &[b, inter], dt);
        let (act_kind, act_label) = match ir.activation {
            Activation::Silu => (OpKind::SwiGlu, "swiglu"),
            Activation::Gelu => (OpKind::GeGlu, "geglu"),
            Activation::GeGlu => (OpKind::GeGlu, "geglu"),
            Activation::None | Activation::Relu => {
                return Err(format!(
                    "Unsupported gated activation {:?} in FFN — \
                     only Silu (SwiGLU), Gelu (GeGLU) are supported for gate*up fusion",
                    ir.activation
                ).into());
            }
        };
        g.add_op(
            act_kind,
            vec![gate_out, up_out],
            vec![ffn_act],
            act_label,
        );

        // Down GEMM: [B, Inter] × [Inter, H] → [B, H]
        let down_out = g.add_tensor_concrete("down", &[b, h], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(b), n: h, k: inter, dtype: dt, trans_b: false },
            vec![ffn_act, w_down],
            vec![down_out],
            "gemm_down",
        );

        // Residual₂: resid1 + down_out
        let output = g.add_tensor_concrete("output", &[b, h], dt);
        g.add_op(
            OpKind::Residual,
            vec![resid1, down_out],
            vec![output],
            "residual_2",
        );

        g.outputs = vec![output];
        Ok(g)
    }

    /// Build an Encoder graph from LayerIR.
    ///
    /// ```text
    /// input → LayerNorm₁ → Q/K/V GEMMs → MHA → O GEMM → Residual₁
    ///       → LayerNorm₂ → Up GEMM → GELU → Down GEMM → Residual₂
    ///       → MeanPool → L2Normalize
    /// ```
    fn from_layer_ir_encoder(ir: &LayerIR) -> Result<Self, CompilerError> {
        let mut g = CompilerGraph::new();
        let dt = ir.dtype;
        let seq = ir.max_batch; // seq_len dimension
        let h = ir.hidden;
        let q_dim = ir.q_dim();
        let kv_dim = ir.kv_dim();
        let inter = ir.intermediate;

        // ── Graph inputs ──
        let input = g.add_tensor_concrete("input", &[seq, h], dt);
        let w_norm1 = g.add_tensor_concrete("w_ln1_gamma", &[h], dt);
        let w_norm1_b = g.add_tensor_concrete("w_ln1_beta", &[h], dt);
        let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
        let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
        let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
        let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
        let w_norm2 = g.add_tensor_concrete("w_ln2_gamma", &[h], dt);
        let w_norm2_b = g.add_tensor_concrete("w_ln2_beta", &[h], dt);
        let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
        let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);

        g.inputs = vec![
            input, w_norm1, w_norm1_b, w_q, w_k, w_v, w_o,
            w_norm2, w_norm2_b, w_up, w_down,
        ];

        // ── Phase 1: Attention ──

        // LayerNorm₁ (with bias, unlike RmsNorm)
        let normed1 = g.add_tensor_concrete("normed1", &[seq, h], dt);
        g.add_op(
            OpKind::LayerNorm { eps: ir.rms_eps },
            vec![input, w_norm1, w_norm1_b],
            vec![normed1],
            "layer_norm_1",
        );

        // Q projection: [seq, H] × [H, Q] → [seq, Q]
        let q_out = g.add_tensor_concrete("q", &[seq, q_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: q_dim, k: h, dtype: dt, trans_b: false },
            vec![normed1, w_q],
            vec![q_out],
            "gemm_q",
        );

        // K projection: [seq, H] × [H, KV] → [seq, KV]
        let k_out = g.add_tensor_concrete("k", &[seq, kv_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: kv_dim, k: h, dtype: dt, trans_b: false },
            vec![normed1, w_k],
            vec![k_out],
            "gemm_k",
        );

        // V projection: [seq, H] × [H, KV] → [seq, KV]
        let v_out = g.add_tensor_concrete("v", &[seq, kv_dim], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: kv_dim, k: h, dtype: dt, trans_b: false },
            vec![normed1, w_v],
            vec![v_out],
            "gemm_v",
        );

        // Multi-head attention (no RoPE for encoder, bidirectional)
        let attn_out = g.add_tensor_concrete("attn_out", &[seq, q_dim], dt);
        g.add_op(
            OpKind::MultiHeadAttention {
                seq_len: SymDim::Concrete(seq),
                num_heads: ir.num_heads,
                num_kv_heads: ir.num_heads, // encoder: num_kv_heads == num_heads
                head_dim: ir.head_dim,
                causal: false, // encoder: bidirectional attention
                attention_sinks: false,
            },
            vec![q_out, k_out, v_out],
            vec![attn_out],
            "mha",
        );

        // O projection: [seq, Q] × [Q, H] → [seq, H]
        let o_out = g.add_tensor_concrete("o_proj", &[seq, h], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: h, k: q_dim, dtype: dt, trans_b: false },
            vec![attn_out, w_o],
            vec![o_out],
            "gemm_o",
        );

        // Residual₁: input + o_out
        let resid1 = g.add_tensor_concrete("residual1", &[seq, h], dt);
        g.add_op(
            OpKind::Residual,
            vec![input, o_out],
            vec![resid1],
            "residual_1",
        );

        // ── Phase 2: FFN ──

        // LayerNorm₂
        let normed2 = g.add_tensor_concrete("normed2", &[seq, h], dt);
        g.add_op(
            OpKind::LayerNorm { eps: ir.rms_eps },
            vec![resid1, w_norm2, w_norm2_b],
            vec![normed2],
            "layer_norm_2",
        );

        // Up GEMM: [seq, H] × [H, Inter] → [seq, Inter]
        let up_out = g.add_tensor_concrete("up", &[seq, inter], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: inter, k: h, dtype: dt, trans_b: false },
            vec![normed2, w_up],
            vec![up_out],
            "gemm_up",
        );

        // GELU activation
        let gelu_out = g.add_tensor_concrete("gelu", &[seq, inter], dt);
        g.add_op(
            OpKind::Gelu,
            vec![up_out],
            vec![gelu_out],
            "gelu",
        );

        // Down GEMM: [seq, Inter] × [Inter, H] → [seq, H]
        let down_out = g.add_tensor_concrete("down", &[seq, h], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: h, k: inter, dtype: dt, trans_b: false },
            vec![gelu_out, w_down],
            vec![down_out],
            "gemm_down",
        );

        // Residual₂: resid1 + down_out
        let resid2 = g.add_tensor_concrete("residual2", &[seq, h], dt);
        g.add_op(
            OpKind::Residual,
            vec![resid1, down_out],
            vec![resid2],
            "residual_2",
        );

        // ── Phase 3: Pooling + Normalization ──

        // Mean pooling across sequence dimension → [1, H]
        let pooled = g.add_tensor_concrete("pooled", &[1, h], dt);
        g.add_op(
            OpKind::MeanPool { seq_len: seq, hidden: h, cls_mode: false },
            vec![resid2],
            vec![pooled],
            "mean_pool",
        );

        // L2 normalize the embedding
        let normalized = g.add_tensor_concrete("normalized", &[1, h], dt);
        g.add_op(
            OpKind::L2Normalize { hidden: h },
            vec![pooled],
            vec![normalized],
            "l2_normalize",
        );

        g.outputs = vec![normalized];
        Ok(g)
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

