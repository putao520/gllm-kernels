/// Hashable key for `OpKind` (OpKind contains f32/f64 fields that prevent `Hash`).
///
/// For norm ops, the key is eps-agnostic: the trace pattern is identical
/// regardless of epsilon, so a single cached trace serves all eps values.
/// The actual eps is read from `OpKind` at codegen time.
/// values produce distinct cache keys and compiled kernels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpKindKey {
    /// RmsNorm (eps-agnostic; actual eps comes from OpKind at codegen time).
    RmsNorm,
    /// LayerNorm (eps-agnostic; actual eps comes from OpKind at codegen time).
    LayerNorm,
    Gemm,
    GemmBias,
    Silu,
    Gelu,
    Tanh,
    SwiGlu,
    /// Clipped SwiGLU (OpenAI gpt-oss-20b).
    /// The `limit` value lives on `Op::SwiGluClipped` itself; the
    /// registry stores a single template trace and
    /// `plan_lower::extract_op_trace` rewrites the clamp constants per-op.
    SwiGluClipped,
    GeGlu,
    Softmax,
    RoPE,
    DualRoPE,
    Add,
    Mul,
    Residual,
    /// LogitSoftcap: cap * tanh(x / cap). cap is a compile-time constant from OpKind.
    LogitSoftcap,
    Reshape,
    Transpose,
    QuantGemm,
    Dequantize,
    MultiHeadAttention,
    MeanPool,
    L2Normalize,
    /// QkNorm (head_dim-agnostic; actual head_dim comes from OpKind at codegen time).
    QkNorm,
    /// HeadRmsNorm (Qwen3 q_norm/k_norm): head-wise RMSNorm with learned weight.
    HeadRmsNorm,
    /// ValueNorm (eps-agnostic; actual eps comes from OpKind at codegen time).
    /// RMSNorm without weight multiplication — Gemma 4 Value vector normalization.
    ValueNorm,
    CachedGQA,
    MoEGate,
    TopK,
    WeightedSum,
    KvScatterWrite,
    KvCacheWrite,
    // P4/P5 stub variants
    VariableLengthBatch,
    AttentionSkipMask,
    FusedRmsNormGemm,
    ResidualWithTelemetry,
    EntropyGate,
    VRangeQuant,
    KvCentroidPrefetch,
    LayerBypass,
    GateMask,
    MaskedGemm,
    MoEConditionalAdd,
    SoftmaxWithEntropy,
    MegaKernelDispatch,
    Gather,
    /// Quantized embedding lookup (ARCH-RUST-IS-CODEGEN): dequantize-on-the-fly per token.
    QuantGather,
    SliceView,
    /// Row-major column slice (真实 copy, row_stride 变化, 详见 Op::ColumnSlice).
    ColumnSlice,
    /// AltUp Predict (Gemma 4 E2B/E4B): multi-path prediction.
    AltUpPredict,
    /// AltUp Correct (Gemma 4 E2B/E4B): innovation correction.
    AltUpCorrect,
    /// AltUp Inject (Gemma 4 E2B/E4B): PLE gate injection.
    AltUpInject,
    /// Depthwise 1D convolution (USM Conformer convolution module).
    /// Per-channel 1D conv with causal or SAME padding; scalar reference at
    /// `scalar-ops/src/depthwise_conv1d.rs::scalar_depthwise_conv1d`.
    DepthwiseConv1D,
    /// Patch embedding Conv2D (SigLIP / ViT vision tower).
    /// `[in_channels, H, W] -> [num_patches, embed_dim]` via stride=patch_size
    /// sliding window; scalar reference at
    /// `scalar-ops/src/patch_embed.rs::scalar_patch_embed`.
    PatchEmbed,
    /// Learned 2D positional embedding (SigLIP / ViT).
    /// Pure binary elementwise add `patches + pos_table`; scalar reference at
    /// `scalar-ops/src/learned_pos_2d.rs::scalar_learned_pos_2d`.
    LearnedPos2D,
    /// MXFP4 dequantization (OCP Microscaling FP4).
    /// Decodes 4-bit e2m1 nibbles + e8m0 power-of-2 scale → f32.
    /// Distinct from `Dequantize` because the e2m1 codeword → value mapping is a
    /// 16-entry LUT, not a linear `quant * scale`. SymExec must capture the LUT
    /// behaviour separately to avoid emitting incorrect linear-decode fast paths.
    DequantizeMxfp4,
    /// Packed-expert MoE dispatch (OpenAI gpt-oss-20b).
    /// Composite op: index dispatch + mxfp4 dequant + clipped SwiGLU + down GEMV
    /// + weighted accumulate. Opaque (无法 SymExec 追踪), 走 plan_lower 专用
    /// 分支 `lower_moe_dispatch_packed`, registry 只挂标量参考函数指针。
    MoEDispatchPacked,
    /// MoE router: hidden @ weight.T + bias → softmax → top-k selection.
    /// Composite op producing router_weights [seq_len, top_k] and router_indices
    /// [seq_len, top_k] (u32 in f32 bits). Opaque, specialized dispatch in plan_lower.
    MoERouter,
    /// Semantic Gatekeeper Q-Tap STG (ARCH-SG-QTAP).
    /// Pure side-effect op: copy q_proj output to ring buffer + bump atomic step_index.
    /// No scalar reference trace (dispatch goes through `lower_qtap_stg`).
    QTapSTG,
    /// Argmax: find max element index in a vector (GRAPH-SHAPE-DRIVEN-MEGA-KERNEL §2.2).
    /// Reduction op, can be fused as GEMM epilogue (e.g., logits GEMM + argmax).
    Argmax,
    /// StoreToken: write generated token to output buffer (side-effect, Opaque).
    StoreToken,
    /// CheckStopCondition: check EOS / max_tokens for generate loop exit (Opaque).
    CheckStopCondition,
    // Business config ops (§1.5 conditional graph construction)
    /// WriteLogits: write selected logits to output buffer for classification/encoding.
    WriteLogits,
    /// EarlyExit: conditional early exit at anchor layer (Intent Recall / EncodeToLayer).
    EarlyExit,
    /// GuardrailCheck: post-node veto probe (in-flight safety check).
    GuardrailCheck,
    /// SgInject: Semantic Gatekeeper knowledge residual vector injection.
    SgInject,
    /// SgDetect: Semantic Gatekeeper hidden state extraction for detection.
    SgDetect,
    /// CotStepCheck: CoT Step Hook step control flags check.
    CotStepCheck,
    /// SessionKvRestore: session KV cache cross-turn restore.
    SessionKvRestore,
    /// MmHiddenInject: multimodal fused hidden state injection.
    MmHiddenInject,
    /// MtpDraft: Multi-Token Prediction draft candidate generation (MTP-001).
    MtpDraft,
    // MLA (Multi-head Latent Attention) — DeepSeek V3/R1, Kimi-K2
    /// MLA KV compression: X · W_DKV → c_KV (GEMM path).
    MlaKvCompress,
    /// MLA Q absorption: Q · W_UK^T → Q_absorbed (per-head GEMM).
    MlaQAbsorb,
    /// MLA V restoration: c_KV · W_UV → V (per-head GEMM).
    MlaVRestore,
    /// MLA Attention: Q_absorbed × key^T → scores → × V (Structural, TraceOp extension).
    MlaAttention,
    /// MLA decoupled RoPE merge: replace c_KV tail with RoPE(k_pe) (Injective).
    MlaRopeMerge,
    /// Multiply by compile-time constant (Elementwise).
    ScaleConst,
}

#[derive(Debug)]
pub enum RegistryError {
    /// Symbolic execution failed to produce a trace.
    SymExec(SymExecError),
    /// The extracted trace was empty or invalid.
    EmptyTrace,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::SymExec(e) => write!(f, "symexec error: {e}"),
            RegistryError::EmptyTrace => write!(f, "extracted trace is empty"),
        }
    }
}

impl From<SymExecError> for RegistryError {
    fn from(e: SymExecError) -> Self {
        RegistryError::SymExec(e)
    }
}

