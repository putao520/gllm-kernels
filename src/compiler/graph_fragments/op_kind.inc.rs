/// KV cache 来源 — op-level 自描述，替代 graph-level topology 推导。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvSource {
    /// K/V 来自当前图的 tensor（self-attention prefill、conformer encoder、
    /// vision attention、intent tracker 等无持久化 KV cache 的场景）。
    /// 不读写持久化 KV cache，zero cache copy overhead。
    FromTensor,
    /// K/V 来自持久化 KV cache（LLM decoder generate loop）。
    /// 层索引用运行时 layer_loop_counter（mega-kernel ABI 提供），
    /// 而非编译时常量——因为层循环中每层 layer_idx 不同。
    FromCache,
}

/// The set of operations the compiler graph can represent.
#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    // ── Normalization ──
    RmsNorm { feature_dim: usize, eps: f32 },
    LayerNorm { feature_dim: usize, eps: f32 },
    /// Value-Normalization: RMSNorm without learned scale (no weight multiplication).
    /// Used by Gemma 4 for Value vector normalization.
    /// `V_out = V / sqrt(mean(V^2) + eps)`
    ValueNorm { feature_dim: usize, eps: f32 },

    // ── Linear algebra ──
    /// C = A × B  (row-major, A is [M,K], B is [K,N], C is [M,N])
    Gemm { m: SymDim, n: usize, k: usize, dtype: DType, trans_b: bool },
    /// C = A × B + bias
    GemmBias { m: SymDim, n: usize, k: usize, dtype: DType, trans_b: bool },

    // ── Activations ──
    Silu,
    Gelu,
    /// tanh(x) — RobertaClassificationHead 的中间激活。
    Tanh,
    /// SwiGLU: silu(gate) * up
    SwiGlu,
    /// Clipped SwiGLU (OpenAI gpt-oss-20b style):
    /// `gate' = clamp(gate, -limit, +limit)`,
    /// `up'   = clamp(up,   -limit, +limit)`,
    /// `out   = silu(gate') * up'`.
    ///
    /// `limit` is the symmetric magnitude clipping threshold
    /// (e.g. `swiglu_limit = 7.0` in gpt-oss-20b).
    SwiGluClipped { limit: f32 },
    /// GeGLU: gelu(gate) * up
    GeGlu,

    // ── Attention ──
    Softmax,
    /// Multi-head attention: Q[s,q_dim] × K[s,kv_dim] → softmax → × V[s,kv_dim] → [s,q_dim]
    /// Q/K/V are seq-major: Q[pos * num_heads * head_dim + head * head_dim + d]
    /// K/V use num_kv_heads (GQA: num_kv_heads <= num_heads).
    ///
    /// `attention_sinks = true` enables **learnable per-head attention sink** (OpenAI
    /// gpt-oss-20b / StreamingLLM style): a learnable scalar `sinks[h]` is concatenated
    /// into the softmax denominator as a virtual `ki = -1` position that absorbs
    /// probability mass without contributing to the output. When `attention_sinks = true`,
    /// `op.inputs[3]` **must** be the sinks tensor of shape `[num_heads]` (f32).
    ///
    /// Modified online softmax (per (qi, h)):
    ///   running_max ← sinks[h]
    ///   running_sum ← 1.0   (= exp(sinks[h] - sinks[h]))
    ///   for each ki:
    ///     score     ← dot(Q[qi,h,:], K[ki,h,:]) * scale
    ///     new_max   ← max(running_max, score)
    ///     corr      ← exp(running_max - new_max)
    ///     weight    ← exp(score - new_max)
    ///     o_acc[d]  ← o_acc[d] * corr + weight * V[ki,h,d]
    ///     running_sum ← running_sum * corr + weight
    ///     running_max ← new_max
    ///   o[qi,h,d] ← o_acc[d] / running_sum          (sink does **not** contribute to output)
    MultiHeadAttention { seq_len: SymDim, num_heads: usize, num_kv_heads: usize, head_dim: usize, causal: bool, attention_sinks: bool, kv_source: KvSource },
    /// Rotary position embedding (non-interleaved).
    /// `partial` controls what fraction of head_dim dimensions are rotated (0.0~1.0).
    /// 1.0 = standard full RoPE. Gemma 4 global layers use 0.25 (p-RoPE).
    /// `rope_scaling` selects the frequency / temperature scaling family:
    /// `None` ⇒ standard RoPE (no scaling), `Some(Yarn { .. })` ⇒ NTK-aware
    /// YaRN long-context scaling (OpenAI gpt-oss-20b / DeepSeek V2 style),
    /// `Some(Linear { .. })` ⇒ position-interpolation linear scaling.
    /// The frequency table (inv_freq) and `attention_scaling` (mscale) derived
    /// from `rope_scaling` are propagated through `RopeCacheRequirement` to the
    /// scratchpad filler — `lower_rope_full` consumes the resulting cos/sin
    /// table unchanged. See [`crate::compiler::rope_scaling`] for the math.
    RoPE {
        num_heads: usize,
        head_dim: usize,
        theta: f64,
        partial: f32,
        rope_scaling: Option<RopeScaling>,
    },
    /// Dual RoPE for models with per-layer different theta/partial (e.g., Gemma 4).
    /// At runtime, selects between sliding and global RoPE parameters based on
    /// the layer index: if `(layer_idx + offset) % divisor == remainder` → global,
    /// otherwise → sliding. The cos/sin tables for both parameter sets are
    /// precomputed into primary and secondary scratchpad caches.
    DualRoPE {
        num_heads: usize,
        head_dim: usize,
        /// Sliding layer theta (e.g., 10000 for Gemma 4)
        sliding_theta: f64,
        /// Sliding layer partial ratio (e.g., 1.0 for full rotation)
        sliding_partial: f32,
        /// Global layer theta (e.g., 1000000 for Gemma 4)
        global_theta: f64,
        /// Global layer partial ratio (e.g., 0.25 for p-RoPE)
        global_partial: f32,
        rope_scaling: Option<RopeScaling>,
        /// Layer index condition: global when `(layer_idx + offset) % divisor == remainder`
        layer_offset: usize,
        layer_divisor: usize,
        layer_remainder: usize,
    },

    // ── Elementwise ──
    Add,
    Mul,
    /// Multiply by a compile-time constant: out = x * value.
    /// Used for LAuReL √2 residual scaling and similar constant-folding scenarios.
    ScaleConst { value: f32 },
    /// Residual connection: out = x + residual
    Residual,

    /// Pooling over seq dimension.
    /// cls_mode=true: extract position 0 (BERT CLS token), no averaging.
    /// cls_mode=false: average all positions.
    /// Input: [seq_len, hidden], Output: [hidden]
    MeanPool { seq_len: usize, hidden: usize, cls_mode: bool },

    /// L2 normalize: out[i] = x[i] / ||x||₂
    /// Input: [hidden], Output: [hidden]
    L2Normalize { hidden: usize },

    /// QK-Normalization (Gemma 4): L2 normalize per-head then scale by √head_dim.
    /// Replaces Softcap in Gemma 2.
    /// Input: [seq_len * num_heads * head_dim], Output: same shape.
    /// Each head vector is independently normalized.
    QkNorm { head_dim: usize, eps: f32 },

    /// Head-wise RMSNorm with learned weight (Qwen3 q_norm/k_norm pattern).
    /// `out[h, d] = x[h, d] * weight[d] / sqrt(mean(x[h, :]^2) + eps)`
    /// 等价于 standard RMSNorm 应用在 reshape 后的 [..., head_dim] 维度。
    /// Input: [seq * num_heads * head_dim] flat, Output: same shape。
    /// inputs[0] = x, inputs[1] = weight ([head_dim]).
    /// 与 QkNorm 区别:QkNorm 是无 weight 的 L2 normalize × √head_dim (Gemma 4),
    /// HeadRmsNorm 是 mean-based RMSNorm × learned weight (Qwen3)。
    HeadRmsNorm { head_dim: usize, eps: f32 },

    // ── Quantization ──
    /// Quantized GEMM: dequantize weights on-the-fly during matmul.
    ///
    /// Weight blob 布局（ARCH-QUANT-LAYOUT）:
    /// - 量化权重数据起点: `0`（相对于 weight_ptr）
    /// - Scale 数据起点: `scale_offset` 字节（由 WeightLayout 元数据提供）
    /// Quantized GEMM: dequantize-on-the-fly + FMA fusion.
    /// The `quant_type` enum carries all block layout metadata (block_size, bits, block_bytes, etc.)
    /// — JIT codegen uses it to select format-specific dequantization paths.
    QuantGemm {
        m: SymDim,
        n: usize,
        k: usize,
        quant_type: crate::quant::QuantType,
    },
    /// Standalone dequantization: convert quantized block to f32.
    Dequantize {
        /// Number of elements to dequantize
        num_elements: usize,
        /// Block size
        block_size: usize,
        /// Bits per element
        bits: usize,
    },

    // ── Mega-Kernel Generate Loop Ops (GRAPH-SHAPE-DRIVEN-MEGA-KERNEL §2.2) ──

    /// Find the index of the maximum value in a logits vector.
    ///
    /// Logit softcapping: `cap * tanh(logits / cap)` (Gemma 4, Grok).
    /// Applied element-wise before Argmax.
    LogitSoftcap { cap: f32 },

    /// Argmax over logits to find the token with highest logit.
    /// OpClass: Reduction — scans entire vector. May fuse with preceding logits-producer GEMM
    /// as EpilogueInjection (max computed in GEMM accumulator registers).
    ///
    /// Input: `logits[1, vocab_size]` (f32 row-major)
    /// Output: `token_id[1]` (scalar index as f32 bits)
    Argmax { vocab_size: usize },

    /// Write the generated token ID to the output token buffer at the current
    /// generate-loop iteration position.
    ///
    /// OpClass: Opaque — side-effect write. Always Standalone.
    /// Uses `AbiPtrs.gen_loop_counter` to compute write position.
    ///
    /// Input: `token_id[1]` (from Argmax output)
    /// Output: sentinel tensor (DAG SSA consistency)
    StoreToken,

    /// Check whether generation should stop (EOS token hit or max tokens reached).
    /// Emits conditional exit from the generate loop.
    ///
    /// OpClass: Opaque — control flow. Always Standalone.
    /// Uses `AbiPtrs.gen_loop_counter` + MegaKernelFn ABI stack offsets.
    ///
    /// Input: `token_id[1]` (from Argmax output)
    /// Output: sentinel tensor (DAG SSA consistency)
    CheckStopCondition,

    // ── Business Feature Ops (§1.5) — conditionally inserted at graph build time ──

    /// Write selected token logits to output buffer (classify mode).
    /// From GEMM accumulator, gather logits at target_indices and store to output.
    ///
    /// OpClass: Opaque. Always Standalone.
    WriteLogits { target_indices: Vec<u32> },

    /// Early exit from layer loop at anchor layer (encode mode).
    /// Compiles to: CMP layer_counter, anchor → JE .early_exit_path.
    ///
    /// OpClass: Opaque. Always Standalone.
    EarlyExit { anchor_layer: usize },

    /// Guardrail veto probe: read shared memory veto flag → conditional JMP.
    /// Inserted after each GEMM in layer loop when guardrail_enabled.
    ///
    /// OpClass: Opaque. Always Standalone.
    GuardrailCheck { probe_offset: usize },

    /// Semantic Gatekeeper knowledge injection: ADD residual vector to hidden state.
    /// Inserted after embedding when SG is enabled.
    ///
    /// OpClass: Opaque. Always Standalone.
    SgInject { knowledge_offset: usize, dim: usize },

    /// Semantic Gatekeeper detection: extract hidden state to shared memory.
    /// Inserted at detect_layer GEMM output when SG is enabled.
    ///
    /// OpClass: Opaque. Always Standalone.
    SgDetect { detect_offset: usize, hidden_dim: usize },

    /// CoT Step Hook: read step control flag from shared memory → conditional JMP.
    /// Inserted at end of layer loop when CoT step hook is enabled.
    ///
    /// OpClass: Opaque. Always Standalone.
    CotStepCheck { shared_mem_offset: usize },

    /// Session KV Cache restore: check session_position > 0 → skip processed tokens.
    /// Inserted after embed_gather when session_enabled=true.
    /// Compiles to: CMP session_position, 0 → JE .skip → pointer arithmetic.
    /// session_enabled=false → not inserted → zero instruction overhead.
    ///
    /// OpClass: Opaque. Always Standalone.
    SessionKvRestore,

    /// Multimodal fused hidden injection: ADD precomputed hidden to token embedding.
    /// Inserted after embed_gather (and after SessionKvRestore if present) when
    /// multimodal_enabled=true.
    /// Compiles to: load fused_hidden_ptr + num_mm_tokens from ABI → vectorized ADD loop.
    /// multimodal_enabled=false → not inserted → zero instruction overhead.
    ///
    /// OpClass: Opaque. Always Standalone.
    MmHiddenInject { hidden_dim: usize },

    // ── Mega-Kernel / MoE / Cached Attention ──
    /// Mega-Kernel in-kernel dispatch (SPEC §9.1).
    /// Single kernel launch with block-level routing based on RequestStateTable.
    /// Backend-specific codegen:
    /// - x86_64: indirect jump via function pointer table
    /// - PTX/HIP/MSL: branch table with predicated execution
    MegaKernelDispatch {
        num_requests: usize,
        rst_ptr: u64,
        prefill_fn: u64,   // Function pointer to prefill kernel
        decode_fn: u64,    // Function pointer to decode kernel
        chunked_fn: u64,   // Function pointer to chunked prefill kernel
    },
    /// Cached GQA attention: Q[seq_len, q_dim] × K_cache[total_seq, kv_dim]
    /// → softmax(causal) → × V_cache. Supports GQA and sparsity stats.
    CachedGQA {
        seq_len: usize,
        total_seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        strategy: AttentionStrategy,
        kv_dtype: DType,
        kv_source: KvSource,
    },
    /// MoE gate: hidden[seq_len, hidden] @ router_w[hidden, num_experts] → softmax → top_k gate selection
    MoEGate {
        seq_len: usize,
        num_experts: usize,
        hidden: usize,
        top_k: usize,
    },
    /// Top-K selection: gate_probs[seq_len, num_experts] → indices[seq_len, top_k] + weights[seq_len, top_k]
    TopK {
        seq_len: usize,
        num_experts: usize,
        top_k: usize,
    },
    /// Weighted sum of expert outputs: output[s] = Σ_k weight[s,k] * expert_out[k][s]
    WeightedSum {
        seq_len: usize,
        hidden: usize,
        top_k: usize,
    },

    /// MoE Router: hidden @ weight.T + bias → softmax → top-k → (weights, indices).
    ///
    /// Inputs:
    /// 0. `hidden`    `[seq_len, hidden_size]` (f32)
    /// 1. `weight`    `[num_experts, hidden_size]` (f32)
    /// 2. `bias`      `[num_experts]` (f32, optional)
    ///
    /// Outputs:
    /// - `router_weights`  `[seq_len, top_k]` (f32)
    /// - `router_indices`  `[seq_len, top_k]` (u32 in f32 bits)
    ///
    /// OpClass: Opaque. Used by gpt-oss and deepseek MoE YAML templates.
    MoERouter {
        num_experts: usize,
        top_k: usize,
        hidden: usize,
        seq_len: SymDim,
    },

    /// Packed-expert MoE dispatch + mxfp4 dequant + clipped SwiGLU + down projection.
    ///
    /// 针对 OpenAI gpt-oss-20b 的 packed-expert MoE 存储布局:
    /// - `gate_up_blocks`  : mxfp4 packed `[num_experts, 2·intermediate_size, hidden / block_size, bytes_per_block]`
    /// - `gate_up_scales`  : e8m0 scales `[num_experts, 2·intermediate_size, hidden / block_size]`
    /// - `gate_up_bias`    : f32       `[num_experts, 2·intermediate_size]`
    /// - `down_blocks`     : mxfp4 packed `[num_experts, hidden, intermediate_size / block_size, bytes_per_block]`
    /// - `down_scales`     : e8m0 scales `[num_experts, hidden, intermediate_size / block_size]`
    /// - `down_bias`       : f32       `[num_experts, hidden]`
    ///
    /// 语义 (per token `s`):
    /// ```text
    /// for k in 0..top_k:
    ///     e   = router_indices[s, k]     (u32, stored in f32 bits)
    ///     w   = router_weights[s, k]     (f32)
    ///     gu  = mxfp4_dequant(gate_up_blocks[e], gate_up_scales[e]) + gate_up_bias[e]
    ///              ∈ ℝ^{2·intermediate_size}  (row-major over intermediate·2)
    ///     gate, up = split(gu, first=intermediate, second=intermediate)
    ///     activ    = clipped_swiglu(gate, up, limit=swiglu_limit)
    ///     dn       = mxfp4_dequant(down_blocks[e], down_scales[e]) @ activ + down_bias[e]
    ///              ∈ ℝ^{hidden}
    ///     output[s] += w · dn
    /// ```
    ///
    /// 输入顺序 (严格对齐 lower):
    /// 0. `hidden_input`        `[seq_len, hidden]`
    /// 1. `router_weights`      `[seq_len, top_k]` (f32)
    /// 2. `router_indices`      `[seq_len, top_k]` (u32 stored in f32 bits)
    /// 3. `gate_up_blocks`      see layout above
    /// 4. `gate_up_scales`      see layout above
    /// 5. `gate_up_bias`        `[num_experts, 2·intermediate_size]` (f32)
    /// 6. `down_blocks`         see layout above
    /// 7. `down_scales`         see layout above
    /// 8. `down_bias`           `[num_experts, hidden]` (f32)
    ///
    /// OpClass: Opaque (组合算子，包含索引分发 + mxfp4 dequant + SwiGLU +
    /// GEMV + 累加)。走 plan_lower 专用 dispatch 分支 `lower_moe_dispatch_packed`。
    MoEDispatchPacked {
        num_experts: usize,
        top_k: usize,
        mxfp4_block_size: usize,
        swiglu_limit: f32,
        intermediate_size: usize,
        hidden: usize,
        seq_len: SymDim,
    },

    // ── Embedding / View (ARCH-FULL-JIT §4.3/§4.4) ──
    /// Embedding lookup: output[i] = table[indices[i]]
    /// JIT compiles to indexed load loop. OpClass::Injective.
    Gather {
        /// Embedding table row count (vocab_size)
        table_rows: usize,
        /// Embedding dimension per row
        embed_dim: usize,
        /// Number of indices (seq_len, may be Symbolic)
        index_dim: SymDim,
        /// How the lookup indices are obtained.
        indices_kind: GatherIndicesKind,
        /// Scaling factor applied after gather: out = scale * table[indices[i]].
        /// Gemma models use sqrt(hidden_size); None = no scaling (default).
        /// SPEC/39: per-op topology-driven, replaces graph.embedding_scale config.
        scale: Option<f32>,
    },
    /// Quantized embedding lookup: output[i] = dequantize(table_quant[indices[i]]).
    ///
    /// Replaces the "Rust dequantize entire embed table → Gather" path (ARCH-RUST-IS-CODEGEN).
    /// JIT reads each quantized block for the requested token_id and decodes it on-the-fly,
    /// producing F32 output rows without touching the rest of the table.
    ///
    /// Weight blob layout: same as QuantGemm rows — consecutive quantized blocks per row.
    /// Input:  token_ids[seq_len] (i32 flat array)
    /// Weight: quantized embed table [vocab_size rows × row_blocks blocks × block_bytes each]
    /// Output: F32 embeddings [seq_len, hidden_dim]
    ///
    /// OpClass: Injective (indexed memory → decode → store, no arithmetic reduction).
    QuantGather {
        /// Quantization format (carries block_size, block_bytes, scale layout metadata).
        quant_type: crate::quant::QuantType,
        /// Embedding table row count (vocab_size).
        vocab_size: usize,
        /// Embedding dimension per row (hidden_dim); must be divisible by block_size.
        hidden_dim: usize,
        /// Number of indices to look up (seq_len, may be Symbolic).
        index_dim: SymDim,
        /// Scaling factor applied after dequantize: out = scale * dequant(table[indices[i]]).
        /// SPEC/39: per-op topology-driven, replaces graph.embedding_scale config.
        scale: Option<f32>,
    },
    /// Zero-copy sub-tensor view (pointer offset, no data movement).
    /// JIT codegen: NOP (authorized SPEC exception, same as Reshape/Transpose).
    SliceView {
        axis: usize,
        start: usize,
        end: usize,
    },

    /// Row-major column slice (cache-friendly copy, row_stride changes).
    ///
    /// 语义: `output[s, j] = input[s, start + j]`，`s ∈ [0, seq_len)`, `j ∈ [0, slice_dim)`。
    ///
    /// - Input  shape: `[seq_len, input_inner]`（row-major, row_stride = input_inner × elem）
    /// - Output shape: `[seq_len, slice_dim]`（row-major, row_stride = slice_dim × elem）
    /// - Copy 语义: row_stride 发生变化，**无法 zero-copy view**（下游 GEMM/elementwise
    ///   默认 contiguous 布局），因此必须真实拷贝。
    ///
    /// 典型用法（Gemma 4 PLE）:
    ///   `ple_full [seq, num_layers * dim_per_layer]` → 按 `layer_idx` 切出
    ///   `ple_slice [seq, dim_per_layer]`，`start = layer_idx * dim_per_layer`。
    ///
    /// OpClass: Injective（内存约束，无规约 / GEMM）。
    /// JIT 实现: `lower_column_slice`（双层 loop_begin / loop_end，禁止 Rust 循环展开）。
    ColumnSlice {
        /// 外层循环维度（seq_len 或其他可 Symbolic 的外层）。
        seq_len: SymDim,
        /// Input 张量最内层维度（= num_layers × slice_dim 的意思，但不假设结构）。
        input_inner: usize,
        /// 起始列（元素单位，非字节）；输出第 0 列对应 input 第 start 列。
        start: usize,
        /// Output 张量最内层维度（= slice_dim，逐行拷贝的元素数）。
        slice_dim: usize,
    },

    // ── Layout ──
    Transpose { perm: Vec<usize> },
    Reshape { target_shape: Vec<usize> },

    // ── KV Cache ──
    /// KV cache scatter write: 将 [seq_len, kv_dim] interleaved 布局的 K/V
    /// 散写到 per-head [max_seq, head_dim] 的 KV cache 布局。
    /// 一次 kernel launch 替代 O(num_kv_heads × seq_len) 次 DtoD 调用。
    /// Inputs: [k_src, v_src, kv_cache]
    /// Grid: (num_kv_heads, seq_len, 1), Block: (min(head_dim, 1024), 1, 1)
    KvScatterWrite {
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        write_start: usize,
        /// bytes: layer * num_kv_heads * head_stride
        layer_offset: usize,
        /// bytes: K half size (for V offset)
        half_offset: usize,
        /// bytes: max_seq * head_dim * dtype_size
        head_stride: usize,
        dtype_size: usize,
    },

    /// CPU-side KV cache write: store K/V projections to paged KV cache.
    /// Used by generate graphs where MHA writes K/V to the KV cache for future decoding.
    /// Inputs: [k_proj, v_proj]
    KvCacheWrite { num_kv_heads: usize, head_dim: usize, seq_len: SymDim },

    // ── P4/P5 features ──
    /// Variable-length batching for ragged inputs.
    VariableLengthBatch,

    /// Attention skip mask based on entropy threshold.
    AttentionSkipMask { seq_len: SymDim, threshold: f32 },

    /// Fused RmsNorm + GEMM.
    FusedRmsNormGemm { m: SymDim, n: usize, k: usize, eps: f32, dtype: DType, trans_b: bool },

    /// Residual connection with telemetry output.
    ResidualWithTelemetry { hidden: usize },

    /// Entropy-based gate for KV cache write decisions.
    EntropyGate { seq_len: SymDim, vocab_size: usize, entropy_threshold: f32 },

    /// Value-range quantization.
    VRangeQuant { seq_len: SymDim, kv_dim: usize, block_size: usize, range_threshold: f32 },

    /// KV centroid prefetch.
    KvCentroidPrefetch { seq_len: SymDim, num_heads: usize, head_dim: usize, prefetch_distance: usize },

    /// Layer bypass for early exit.
    LayerBypass { threshold: f32 },

    /// Gate mask for MoE routing.
    GateMask { hidden: usize },

    /// Masked GEMM.
    MaskedGemm { m: SymDim, n: usize, k: usize, dtype: DType, trans_b: bool },

    /// MoE conditional add.
    MoEConditionalAdd { seq_len: SymDim, hidden: usize, num_experts: usize, expert_idx: usize },

    /// Softmax with entropy output.
    SoftmaxWithEntropy { vocab_size: usize },

    /// Per-Layer Embedding (Gemma 4 E2B/E4B).
    ///
    /// AltUp Predict: 对 [P,S,H] 胖激活做 altup 维混合预测 (Gemma 4 E2B/E4B)。
    ///
    /// predictions[p][s][h] = stacked[p][s][h] + Σ_q coefs[p][q] · stacked[q][s][h]
    ///
    /// 输入顺序:
    /// 0. stacked    [P, seq_len, hidden]  — 上层输出的胖激活
    /// 1. coefs      [seq_len, P²]        — 预测系数 (由外部 GEMM + reshape 产出)
    ///
    /// 输出:
    /// 0. predictions [P, seq_len, hidden] — 预测后的胖激活
    ///
    /// P = num_preds (编译期常量, ≤4, 可 Const 展开)。
    /// OpClass: Injective (多输入多输出逐元素)。
    /// SPEC: SPEC/DOCS/architecture/gemma4-altup.md §3.1
    AltUpPredict {
        seq_len: SymDim,
        num_preds: usize,
        hidden: usize,
    },

    /// AltUp Correct: 用 active 路的 innovation 修正所有 P 路预测 (Gemma 4 E2B/E4B)。
    ///
    /// innovation[s][h] = activated[s][h] - predictions[0][s][h]
    /// corrected[p][s][h] = predictions[p][s][h] + corrected_coefs[s][p] × innovation[s][h]
    /// corrected[0] = activated  (active 路直接覆盖)
    ///
    /// 输入顺序:
    /// 0. predictions     [P, seq_len, hidden]  — AltUpPredict 的输出
    /// 1. corrected_coefs [seq_len, P]          — 修正系数 (由外部 GEMM + bias+1 产出)
    /// 2. activated       [seq_len, hidden]     — active 路经 attention/FFN 后的结果
    ///
    /// 输出:
    /// 0. corrected [P, seq_len, hidden] — 修正后的胖激活
    ///
    /// SPEC: SPEC/DOCS/architecture/gemma4-altup.md §3.2
    AltUpCorrect {
        seq_len: SymDim,
        num_preds: usize,
        hidden: usize,
    },

    /// AltUp Inject: 将 PLE 门控结果注入到非 active 预测 (Gemma 4 E2B/E4B)。
    ///
    /// corrected[p][s][h] += ple_projected[s][h]   for p = 1..P-1
    ///
    /// 输入顺序:
    /// 0. corrected      [P, seq_len, hidden]  — AltUpCorrect 的输出 (in-place 修改)
    /// 1. ple_projected  [seq_len, hidden]     — PLE 门控投影结果
    ///
    /// 输出:
    /// 0. corrected [P, seq_len, hidden] — 注入后的胖激活 (与输入 0 共享 buffer)
    ///
    /// SPEC: SPEC/DOCS/architecture/gemma4-altup.md §3.3
    AltUpInject {
        seq_len: SymDim,
        num_preds: usize,
        hidden: usize,
    },

    /// Depthwise 1D convolution (USM Conformer convolution module 核心 op).
    ///
    /// 每个 channel 独立 1D 卷积,输入 `[seq_len, channels]` + 权重 `[channels, kernel_size]`
    /// → 输出 `[seq_len, channels]`。`causal = true` 时 seq 前 zero-pad `kernel_size - 1`
    /// 个元素,保证 `output[t, c]` 只依赖 `input[0..=t, c]` (Conformer 推理标准配置);
    /// `causal = false` 时为 "SAME" padding,奇数 kernel_size 前后各 pad
    /// `(kernel_size - 1) / 2`。
    ///
    /// 输入顺序:
    /// 0. x       [seq_len, channels]     — 输入张量 (row-major)
    /// 1. weight  [channels, kernel_size] — per-channel 卷积核
    ///
    /// OpClass: Opaque (组合算子,JIT lower 后续任务补齐;当前 scalar 参考实现
    /// 见 `scalar-ops/src/depthwise_conv1d.rs`)。
    DepthwiseConv1D {
        channels: usize,
        kernel_size: usize,
        causal: bool,
    },

    /// SigLIP / ViT vision tower patch embedding (T44).
    ///
    /// 图像 `[in_channels, image_size, image_size]` 通过 Conv2D (kernel_size =
    /// patch_size, stride = patch_size) 打成 `num_patches = (image_size /
    /// patch_size)^2` 个 patch token, 每个 token 为 `embed_dim` 维。
    ///
    /// 输入顺序:
    /// 0. image   [in_channels, image_size, image_size]
    /// 1. kernel  [embed_dim, in_channels, patch_size, patch_size]
    ///
    /// 输出: `[num_patches, embed_dim]` (row-major flatten, p = p_row * side + p_col)。
    ///
    /// OpClass: Opaque (Conv2D 视为不可融合组合算子; 当前 scalar 参考实现见
    /// `scalar-ops/src/patch_embed.rs`, JIT lower 后续任务补齐)。
    PatchEmbed {
        patch_size: usize,
        embed_dim: usize,
        in_channels: usize,
        image_size: usize,
    },

    /// SigLIP / ViT learned 2D positional embedding (T44).
    ///
    /// 输入:
    /// 0. patches   [num_patches, embed_dim]
    /// 1. pos_table [num_patches, embed_dim]
    ///
    /// 输出: `out[p, d] = patches[p, d] + pos_table[p, d]` (pure elementwise add)。
    ///
    /// OpClass: Elementwise (binary add; 可融合到前驱 PatchEmbed / 后继 LayerNorm)。
    /// 当前 scalar 参考实现见 `scalar-ops/src/learned_pos_2d.rs`。
    LearnedPos2D {
        num_patches: usize,
        embed_dim: usize,
    },

    /// Semantic Gatekeeper Q-Tap STG (ARCH-SG-QTAP).
    ///
    /// Pure side-effect op that copies the Q vector (the q_proj output) into a
    /// pre-allocated host/device ring buffer and bumps an atomic `step_index`
    /// counter with release memory ordering. The main computation path
    /// (q_proj → RoPE → Attention) is untouched — downstream consumers still
    /// read from `inputs[0]`.
    ///
    /// SPEC refs:
    /// - `SPEC/SEMANTIC-GATEKEEPER.md §4` Q 截获协议
    /// - `SPEC/08-EXECUTOR.md §4.2.1` FusedAttentionLayer Q-Tap 扩展
    ///
    /// Layout of the ring buffer at `sink_ptr`:
    /// ```text
    ///   [u8; num_slots * q_dim * dtype.size_bytes()]
    /// ```
    ///
    /// The atomic `step_index` (absolute host address `step_index_ptr`) is
    /// read-acquire / bumped-release by the callback vs. the JIT, giving the
    /// reader a consistent way to tell whether the currently visible slot is
    /// fresh.
    ///
    /// Inputs:
    ///   - `inputs[0]`: Q tensor to tap (shape `[seq_len, q_dim]` row-major)
    ///
    /// Outputs:
    ///   - `outputs[0]`: 1-element sentinel tensor (for DAG SSA consistency;
    ///                   not consumed by any downstream op).
    QTapSTG {
        /// Gatekeeper ring buffer base pointer — absolute host virtual address
        /// (compile-time constant, baked into the machine code as `mov reg, imm64`).
        sink_ptr: u64,
        /// Atomic step_index absolute host virtual address (u64 atomic).
        /// JIT issues release-bump after the STG write; reader loads with acquire.
        step_index_ptr: u64,
        /// Data type written to the ring buffer (equal to the dtype of the
        /// q_proj output tensor).
        dtype: DType,
        /// Q vector dimension (`num_heads × head_dim`). May be `SymDim::Symbolic`
        /// if the executor binds q_dim at runtime.
        q_dim: SymDim,
        /// Whether to tap every token or only the last (decode) position.
        position: QTapPosition,
        /// Ring buffer slot count (≥ 2 to avoid reader/writer races).
        num_slots: usize,
    },

    /// Multi-Token Prediction draft candidate generation (MTP-001).
    ///
    /// Generates K additional candidate tokens per decode step by projecting
    /// the hidden state through depth dedicated weight matrices.
    /// Each depth computes: GEMV (hidden @ weight) → argmax → store candidate.
    ///
    /// OpClass: Opaque (composite structural op, no fusion with surrounding ops).
    MtpDraft {
        depth: usize,
        hidden_size: usize,
        vocab_size: usize,
    },

    // ── MLA (Multi-head Latent Attention) — DeepSeek V3/R1, Kimi-K2 ──

    /// MLA KV compression: X · W_DKV → c_KV
    /// Semantically a GEMM but tagged for KV cache latent-dim write path.
    /// ComputePattern: Gemm — standard lowering.
    MlaKvCompress {
        m: SymDim,
        d_c: usize,
        hidden: usize,
    },

    /// MLA Q absorption: Q · W_UK^T → Q_absorbed (per head)
    /// Per-head GEMM, W_UK partitioned as [d_c × d] per head.
    /// ComputePattern: Gemm (batched per-head).
    MlaQAbsorb {
        seq_len: SymDim,
        num_heads: usize,
        head_dim: usize,
        d_c: usize,
    },

    /// MLA V restoration: c_KV · W_UV → V (per head)
    /// Per-head GEMM, W_UV partitioned as [d_c × d] per head.
    /// ComputePattern: Gemm (batched per-head).
    MlaVRestore {
        seq_len: SymDim,
        num_heads: usize,
        head_dim: usize,
        d_c: usize,
    },

    /// MLA Attention: Q_absorbed × concat(c_KV_no_rope, RoPE(k_pe))^T → scores → × V
    /// Score computed in compressed d_c space (not full d), then V restored per-head.
    /// ComputePattern: Structural — TraceOp extension required.
    MlaAttention {
        seq_len: SymDim,
        num_heads: usize,
        head_dim: usize,
        d_c: usize,
        d_rope: usize,
        causal: bool,
        kv_source: KvSource,
    },

    /// MLA decoupled RoPE merge: replace c_KV[d_c-d_rope..d_c] with RoPE(k_pe).
    /// Injective: reads c_KV + k_pe, writes merged key.
    MlaRopeMerge {
        seq_len: SymDim,
        d_c: usize,
        d_rope: usize,
    },
}

/// Which position(s) of the Q tensor the Q-Tap STG writes per invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QTapPosition {
    /// Write only the last token's Q vector (decode step scenario).
    /// Ring buffer slot size = `q_dim * dtype.size_bytes()`.
    LastToken,
    /// Write the entire sequence's Q (prefill scenario).
    /// Ring buffer slot size = `seq_len * q_dim * dtype.size_bytes()`.
    AllTokens,
}

/// Configuration for inserting a Q-Tap STG node into a FusedAttentionLayer graph.
///
/// When `Some`, the graph builder inserts an `OpKind::QTapSTG` node after the
/// q_proj GEMM output, copying the Q vector into the ring buffer for the
/// Semantic Gatekeeper callback to read.
///
/// SPEC refs:
/// - `SPEC/SEMANTIC-GATEKEEPER.md §4` Q 截获协议
/// - `SPEC/08-EXECUTOR.md §4.2.1` FusedAttentionLayer Q-Tap 扩展
#[derive(Debug, Clone)]
pub struct QTapGraphConfig {
    /// Gatekeeper ring buffer base pointer — absolute host virtual address.
    pub sink_ptr: u64,
    /// Atomic step_index absolute host virtual address (AtomicU64).
    pub step_index_ptr: u64,
    /// Data type written to the ring buffer (must match q_proj output dtype).
    pub dtype: DType,
    /// Whether to tap every token or only the last (decode) position.
    pub position: QTapPosition,
    /// Ring buffer slot count (≥ 2, must be power of two).
    pub num_slots: usize,
}

// ── Telemetry offsets ──────────────────────────────────────────────

/// Telemetry buffer offsets for epilogue probes (SPEC §9.5).
pub mod telemetry_offsets {
    pub const SILU_DEAD_NEURON_COUNT: usize = 0;
    pub const SILU_DEAD_NEURON_MASK_OFFSET: usize = 8;
    pub const EXPERT_HIT_COUNTS_OFFSET: usize = 64;
    pub const MAX_EXPERTS: usize = 64;
    pub const RESIDUAL_DELTA_OFFSET: usize = 128;
    pub const COSINE_SIMILARITY_OFFSET: usize = 136;
    pub const CHANNEL_SCALE_PTR_OFFSET: usize = 144;
    pub const CENTROID_TOKEN_IDX_OFFSET: usize = 324;
    pub const SOFTMAX_SHARPNESS_OFFSET: usize = 332;
    pub const SOFTMAX_MAX_OFFSET: usize = 340;
    pub const EFFECTIVE_CONTEXT_LEN_OFFSET: usize = 348;
    pub const IS_ATTENTION_SINK_OFFSET: usize = 356;
    pub const GEMM_ROW_NORM_L1_OFFSET: usize = 364;
    pub const GEMM_ROW_MAX_OFFSET: usize = 372;
    pub const GEMM_ROW_MIN_OFFSET: usize = 380;
    /// §13.10 Embedding L2 norm (RaBitQ): ‖embed‖₂ per row, f32.
    pub const EMBED_L2_NORM_OFFSET: usize = 388;
    pub const TELEMETRY_BUFFER_MIN_BYTES: usize = 512;
}

pub const RESIDUAL_NORM_EPSILON: f32 = 1e-6;
pub const SOFTMAX_SINK_THRESHOLD: f32 = 0.5;
pub const SILU_DEAD_NEURON_THRESHOLD: f32 = 0.01;

/// Epilogue telemetry configuration (SPEC §9.5).
#[derive(Debug, Clone, Default)]
pub struct EpilogueTelemetryConfig {
    pub silu_dead_neuron: bool,
    pub moe_hit_counter: bool,
    pub rmsnorm_channel_scale: bool,
    pub softmax_sharpness: bool,
    pub residual_cosine_sim: bool,
    pub gemm_row_stats: bool,
    /// §13.10 Embedding L2 norm (RaBitQ): ‖embed‖₂ per row.
    pub embed_l2_norm: bool,
}

// ── Layer execution guard (NO_LAYER_EXPAND) ──────────────────────

/// Per-op execution guard within the layer loop.
/// A single template is compiled once; the guard is materialized as a
/// runtime `GprCondAction` comparing `layer_loop_counter` against the
/// threshold. Only meaningful inside a layer loop; ops outside the loop
/// must use `Always` (the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerCondition {
    /// Every layer executes this op (default — no guard emitted).
    #[default]
    Always,
    /// Execute only when `layer_idx < threshold` (donor layers).
    LayerIdxLt(usize),
    /// Execute only when `layer_idx >= threshold` (consumer layers).
    LayerIdxGe(usize),
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
    /// Layer loop execution guard. `Always` = no guard (zero overhead).
    pub guard: LayerCondition,
    /// FAT-OPCODE-ARCHITECTURE-V2 §Phase 9: 胖 opcode 缓存。
    ///
    /// add_op 时一次性从 OpKind + graph context 翻译并缓存。
    /// 所有 lowering 路径直接读 op_v2，不调用 Op::from_op_kind（消除重复翻译）。
    /// None 表示尚未翻译（如测试 fixture 或部分构造路径）。
    pub op_v2: Option<crate::compiler::graph::Op>,
}

impl CompilerOp {
    /// 获取已缓存的 Op v2，或在未缓存时按 graph 上下文即时翻译并返回。
    ///
    /// 正常生产路径下，add_op 已缓存 op_v2，本方法零翻译开销。
    /// 测试 fixture / 部分构造路径下，op_v2 为 None，本方法按需翻译。
    pub fn op_v2_resolved(&self, graph: &CompilerGraph) -> Option<crate::compiler::graph::Op> {
        self.op_v2.clone().or_else(|| crate::compiler::graph::Op::from_op_kind(self, graph))
    }

    /// 检查 Op v2 是否为 GEMM 类（胖 opcode 自描述）。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::Gemm{..} | OpKind::GemmBias{..})`。
    pub fn op_v2_is_gemm_like(&self, graph: &CompilerGraph) -> bool {
        self.op_v2_resolved(graph).map(|o| o.is_gemm_like()).unwrap_or(false)
    }

    /// 检查 Op v2 是否为 GemmBias（带 bias 的 GEMM）。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::GemmBias{..})`。
    pub fn op_v2_is_gemm_with_bias(&self, graph: &CompilerGraph) -> bool {
        self.op_v2_resolved(graph).map(|o| o.is_gemm_with_bias()).unwrap_or(false)
    }

    /// 检查 Op v2 是否为 QuantGemm。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::QuantGemm{..})`。
    pub fn op_v2_is_quant_gemm(&self, graph: &CompilerGraph) -> bool {
        self.op_v2_resolved(graph).map(|o| o.is_quant_gemm()).unwrap_or(false)
    }

    /// 检查 Op v2 是否为 Norm 类（RmsNorm/LayerNorm/ValueNorm/HeadRmsNorm）。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::RmsNorm{..} | ...)`。
    pub fn op_v2_is_norm_like(&self, graph: &CompilerGraph) -> bool {
        self.op_v2_resolved(graph).map(|o| o.is_norm_like()).unwrap_or(false)
    }

    /// 提取 GEMM trans_b 参数（胖 opcode 自描述）。
    /// 替代 fusion pass 中的
    /// `match op.kind { OpKind::Gemm{trans_b,..} | OpKind::GemmBias{trans_b,..} => *trans_b, _ => false }`。
    pub fn op_v2_gemm_trans_b(&self, graph: &CompilerGraph) -> bool {
        self.op_v2_resolved(graph).and_then(|o| o.gemm_trans_b()).unwrap_or(false)
    }

    /// 提取 GEMM 维度（胖 opcode 自描述）。
    /// 替代 `extract_gemm_dims_sym` 中
    /// `match op.kind { OpKind::Gemm{m,n,k,..} | ... => (m.clone(),*n,*k), _ => Err(...) }`。
    pub fn op_v2_gemm_dims(&self, graph: &CompilerGraph) -> Option<(crate::compiler::graph::SymDim, usize, usize)> {
        self.op_v2_resolved(graph).and_then(|o| o.gemm_dims())
    }

    /// 输出别名到输入的 IR 元数据（胖 opcode 自描述）。
    /// 替代 `op.kind.output_aliases_input()`。
    pub fn op_v2_output_aliases_input(&self, graph: &CompilerGraph) -> Option<usize> {
        self.op_v2_resolved(graph).and_then(|o| o.output_aliases_input())
    }
}

// ── CompilerGraph ──────────────────────────────────────────────────

/// DAG of operations for a single transformer layer (or full model for mega-kernel).
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
    /// Epilogue telemetry configuration
    pub telemetry: EpilogueTelemetryConfig,
    /// Layer loop configuration for mega-kernel compilation.
    /// When present, layer ops (name prefix "layer.") are wrapped in emit_loop.
    pub layer_loop_config: Option<LayerLoopConfig>,
    /// Heterogeneous layer loop config for models with alternating layer types
    /// (e.g., Gemma-4 E2B). When present, takes precedence over layer_loop_config.
    pub hetero_layer_loop_config: Option<HeteroLayerLoopConfig>,
    /// SPEC/39 NOTE: 编译器不再读取此字段。embedding_scale 已迁移到 OpKind::Gather::scale
    /// 和 OpKind::QuantGather::scale（per-op 拓扑驱动）。此字段仅保留供 gllm 侧
    /// build_graph 构建 OpKind 时读取，编译器路径不使用。
    pub embedding_scale: Option<f32>,
    /// Custom weight layout override. When present, weight_layout() returns this
    /// instead of computing from inputs. Used by mega-kernel where layer weights
    /// have offsets relative to layer start (not absolute from blob start).
    custom_weight_layout: Option<WeightLayout>,
    /// Physical byte sizes for quantized weight tensors. When a tensor ID is
    /// present here, weight_layout() uses this size instead of numel * dtype_size.
    /// This is necessary because GGUF quantized formats (Q4_0, Q8_0, etc.) have
    /// physical sizes much smaller than numel * 4 (F32).
    quant_weight_bytes: HashMap<TensorId, usize>,
    /// Maximum sequence length for this model (from `max_position_embeddings`).
    /// Used as the default `max_value` for `SymDim::Symbolic` dimensions and
    /// for buffer allocation sizing in `weight_layout()` / `buffer_alloc`.
    /// Set by graph builders from model config; defaults to 2048 for backward compat.
    pub max_seq_len: usize,
    /// KV cache load mode for attention variant (KV-OPT-009).
    /// When set, the MHA lowering generates K/V load instructions matching the
    /// specified precision tier (e.g., KiviDequantLoad for KIVI4/KIVI2).
    /// None = Direct (standard VecLoad, default).
    pub kv_load_mode: Option<crate::compiler::codegen::vm::instr::KvLoadMode>,
    next_tensor_id: u32,
    next_op_id: u32,
}

