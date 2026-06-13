// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 Op v2 — 胖 opcode 自描述架构 (FAT-OPCODE-ARCHITECTURE-V2)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// 与 OpKind（贫血 enum）共存。Phase 4-7 按类别迁移 lowering 到 Op。
// Phase 8 后 OpKind 删除。
//
// include! 模式：本文件被 include 到 graph.rs，与 op_kind.inc.rs / types.inc.rs
// 共享作用域。KvSource/RopeScaling/AttentionStrategy/GatherIndicesKind/SymDim/DType
// 等类型直接可见，无需 use。

// ── Attention Specs ──

/// MHA 几何参数（Q/K/V 维度）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionGeometry {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Attention mask 类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    /// 因果掩码（decoder self-attention）
    Causal,
    /// 全注意力（encoder/conformer self-attention）
    Full,
    /// 滑动窗口（Mistral-style）
    Sliding { window: usize },
}

/// Attention sinks 配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinksSpec {
    None,
    /// Learnable per-head sinks（inputs[3] = sinks tensor）
    Learnable,
}

/// MultiHeadAttention 完整自描述 spec。
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSpec {
    pub geometry: AttentionGeometry,
    pub mask: AttentionMask,
    pub kv_source: KvSource,
    pub sinks: SinksSpec,
    pub seq_len: SymDim,
    pub dtype: DType,
}

/// CachedGQA spec（GQA + KV cache）。
#[derive(Debug, Clone, PartialEq)]
pub struct CachedGqaSpec {
    pub geometry: AttentionGeometry,
    pub mask: AttentionMask,
    pub kv_source: KvSource,
    pub seq_len: SymDim,
    pub total_seq: usize,
    pub kv_dtype: DType,
    pub strategy: AttentionStrategy,
}

/// MLA attention spec（DeepSeek-style）。
#[derive(Debug, Clone, PartialEq)]
pub struct MlaSpec {
    pub seq_len: SymDim,
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_c: usize,
    pub d_rope: usize,
    pub causal: bool,
    pub kv_source: KvSource,
}

// ── Norm Specs ──

/// Norm 类 op 自描述 spec（RmsNorm/LayerNorm/ValueNorm/HeadRmsNorm 共用）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormSpec {
    pub feature_dim: usize,
    pub eps: f32,
    pub dtype: DType,
    pub has_weight: bool,
}

// ── Gemm Specs ──

/// Gemm 自描述 spec。
#[derive(Debug, Clone, PartialEq)]
pub struct GemmSpec {
    pub m: SymDim,
    pub n: usize,
    pub k: usize,
    pub dtype: DType,
    pub trans_b: bool,
    pub has_bias: bool,
}

/// QuantGemm spec（量化矩阵乘）。
#[derive(Debug, Clone, PartialEq)]
pub struct QuantGemmSpec {
    pub m: SymDim,
    pub n: usize,
    pub k: usize,
    pub quant_type: crate::quant::QuantType,
}

// ── RoPE Specs ──

#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    pub theta: f64,
    pub partial: f32,
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DualRopeSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    pub sliding_theta: f64,
    pub sliding_partial: f32,
    pub global_theta: f64,
    pub global_partial: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub layer_offset: usize,
    pub layer_divisor: usize,
    pub layer_remainder: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 Op enum — 扁平变体（与 OpKind 1:1 对应）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 胖 opcode 自描述架构 — 取代 OpKind 的下一代算子枚举。
///
/// 每个变体携带完整语义，lowering 签名目标：`(prog, op, sess) -> Result`。
/// 无参数变体保持 unit；复杂变体用 Spec struct。
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // ── Normalization ──
    RmsNorm(NormSpec),
    LayerNorm(NormSpec),
    ValueNorm(NormSpec),
    HeadRmsNorm { head_dim: usize, eps: f32, dtype: DType },

    // ── Linear algebra ──
    Gemm(GemmSpec),
    GemmBias(GemmSpec),
    QuantGemm(QuantGemmSpec),

    // ── Activations ──
    Silu,
    Gelu,
    Tanh,
    SwiGlu,
    SwiGluClipped { limit: f32 },
    GeGlu,

    // ── Attention ──
    Softmax,
    MultiHeadAttention(AttentionSpec),
    CachedGqa(CachedGqaSpec),
    MlaAttention(MlaSpec),
    MlaKvCompress { seq_len: SymDim, num_heads: usize, head_dim: usize, d_c: usize },
    MlaQAbsorb { seq_len: SymDim, num_heads: usize, d_c: usize, head_dim: usize },
    MlaVRestore { seq_len: SymDim, num_heads: usize, d_c: usize, head_dim: usize },
    MlaRopeMerge { seq_len: SymDim, num_heads: usize, head_dim: usize, d_rope: usize },

    // ── Position encoding ──
    RoPE(RopeSpec),
    DualRoPE(DualRopeSpec),

    // ── Elementwise binary ──
    Add,
    Mul,
    ScaleConst { value: f32 },
    Residual,

    // ── Pooling / normalize ──
    MeanPool { seq_len: usize, hidden: usize, cls_mode: bool },
    QkNorm { head_dim: usize, eps: f32 },
    L2Normalize,

    // ── Quantization ──
    Dequantize { num_elements: usize, block_size: usize, bits: usize },
    LogitSoftcap { cap: f32 },

    // ── Generation control ──
    Argmax { vocab_size: usize },
    StoreToken,
    CheckStopCondition,
    WriteLogits { target_indices: Vec<u32> },
    EarlyExit { anchor_layer: usize },

    // ── Business config ops ──
    GuardrailCheck { probe_offset: usize },
    SgInject { knowledge_offset: usize, dim: usize },
    SgDetect { detect_offset: usize, hidden_dim: usize },
    CotStepCheck { shared_mem_offset: usize },
    SessionKvRestore,
    MmHiddenInject { hidden_dim: usize },

    // ── Dispatch ──
    MegaKernelDispatch,

    // ── MoE ──
    MoEGate { seq_len: SymDim, num_experts: usize, hidden: usize, top_k: usize },
    TopK { seq_len: SymDim, num_experts: usize, top_k: usize },
    WeightedSum { seq_len: SymDim, num_experts: usize, top_k: usize, hidden: usize },
    MoERouter { num_experts: usize, top_k: usize, hidden: usize, seq_len: SymDim },
    MoEDispatchPacked {
        seq_len: SymDim, num_experts: usize, top_k: usize,
        mxfp4_block_size: usize, swiglu_limit: f32, hidden: usize, intermediate: usize,
    },
    MoEConditionalAdd { seq_len: SymDim, hidden: usize, num_experts: usize, expert_idx: usize },

    // ── Structural ──
    Gather { seq_len: SymDim, index_len: usize, index_kind: GatherIndicesKind },
    QuantGather { seq_len: SymDim, index_len: usize, quant_type: crate::quant::QuantType },
    SliceView { start: usize, end: usize, stride: usize },
    ColumnSlice { start: usize, end: usize },
    Transpose { perm: Vec<usize> },
    Reshape { target_shape: Vec<usize> },

    // ── KV cache ──
    KvScatterWrite { seq_len: SymDim, num_kv_heads: usize, head_dim: usize },
    KvCacheWrite { num_kv_heads: usize, head_dim: usize, seq_len: SymDim },

    // ── P4/P5 advanced ──
    VariableLengthBatch,
    AttentionSkipMask { seq_len: SymDim, threshold: f32 },
    FusedRmsNormGemm { m: SymDim, n: usize, k: usize, eps: f32, dtype: DType, trans_b: bool },
    ResidualWithTelemetry { hidden: usize },
    EntropyGate { seq_len: SymDim, vocab_size: usize, entropy_threshold: f32 },
    VRangeQuant { seq_len: SymDim, kv_dim: usize, block_size: usize, range_threshold: f32 },
    KvCentroidPrefetch { seq_len: SymDim, num_heads: usize, head_dim: usize, prefetch_distance: usize },
    LayerBypass { threshold: f32 },
    GateMask { hidden: usize },
    MaskedGemm { m: SymDim, n: usize, k: usize, dtype: DType, trans_b: bool },
    SoftmaxWithEntropy { vocab_size: usize },

    // ── Gemma 4 AltUp+PLE ──
    AltUpPredict { num_preds: usize, hidden: usize },
    AltUpCorrect { num_preds: usize, hidden: usize },
    AltUpInject { num_preds: usize, hidden: usize },

    // ── Vision/Audio ──
    DepthwiseConv1D { channels: usize, kernel_size: usize, causal: bool },
    PatchEmbed { patch_size: usize, embed_dim: usize, in_channels: usize, image_size: usize },
    LearnedPos2D { num_patches: usize, embed_dim: usize },

    // ── Quant pipeline ──
    QTapSTG {
        sink_ptr: u64, step_index_ptr: u64, dtype: DType,
        q_dim: SymDim, position: QTapPosition, num_slots: usize,
    },

    // ── MTP ──
    MtpDraft { depth: usize, hidden_size: usize, vocab_size: usize },

    // ── Last token / all tokens (sampling) ──
    LastToken,
    AllTokens,
}

impl Op {
    /// 返回 op 的类别名称（用于 JIT 日志、调试、profiling）。
    pub fn category(&self) -> &'static str {
        match self {
            Op::RmsNorm(_) | Op::LayerNorm(_) | Op::ValueNorm(_) | Op::HeadRmsNorm { .. }
            | Op::QkNorm { .. } | Op::L2Normalize => "norm",
            Op::Gemm(_) | Op::GemmBias(_) | Op::QuantGemm(_)
            | Op::FusedRmsNormGemm { .. } | Op::MaskedGemm { .. } => "gemm",
            Op::Silu | Op::Gelu | Op::Tanh | Op::SwiGlu | Op::SwiGluClipped { .. } | Op::GeGlu
            | Op::Add | Op::Mul | Op::ScaleConst { .. } | Op::Residual => "activation",
            Op::MultiHeadAttention(_) | Op::CachedGqa(_) | Op::MlaAttention(_)
            | Op::Softmax | Op::MlaKvCompress { .. } | Op::MlaQAbsorb { .. }
            | Op::MlaVRestore { .. } | Op::MlaRopeMerge { .. } => "attention",
            Op::RoPE(_) | Op::DualRoPE(_) => "rope",
            Op::MoEGate { .. } | Op::TopK { .. } | Op::WeightedSum { .. }
            | Op::MoERouter { .. } | Op::MoEDispatchPacked { .. }
            | Op::MoEConditionalAdd { .. } => "moe",
            Op::DepthwiseConv1D { .. } | Op::PatchEmbed { .. } | Op::LearnedPos2D { .. } => "vision_audio",
            Op::AltUpPredict { .. } | Op::AltUpCorrect { .. } | Op::AltUpInject { .. } => "altup",
            Op::Argmax { .. } | Op::StoreToken | Op::CheckStopCondition
            | Op::WriteLogits { .. } | Op::EarlyExit { .. } | Op::LastToken | Op::AllTokens => "sampling",
            Op::GuardrailCheck { .. } | Op::SgInject { .. } | Op::SgDetect { .. }
            | Op::CotStepCheck { .. } | Op::SessionKvRestore | Op::MmHiddenInject { .. } => "business",
            Op::Gather { .. } | Op::QuantGather { .. } | Op::SliceView { .. }
            | Op::ColumnSlice { .. } | Op::Transpose { .. } | Op::Reshape { .. }
            | Op::MegaKernelDispatch => "structural",
            Op::Dequantize { .. } | Op::QTapSTG { .. } | Op::VRangeQuant { .. } => "quant",
            Op::MeanPool { .. } | Op::LogitSoftcap { .. } => "misc",
            Op::KvScatterWrite { .. } | Op::KvCacheWrite { .. } => "kv_cache",
            Op::VariableLengthBatch | Op::AttentionSkipMask { .. }
            | Op::ResidualWithTelemetry { .. } | Op::EntropyGate { .. }
            | Op::KvCentroidPrefetch { .. } | Op::LayerBypass { .. }
            | Op::GateMask { .. } | Op::SoftmaxWithEntropy { .. } => "advanced",
            Op::MtpDraft { .. } => "mtp",
        }
    }

    /// Phase 4: Norm/Activation 类别的 from_op_kind 转换。
    ///
    /// dtype 从 op.inputs[0] 的 tensor 推导（ARCH-DTYPE-JIT-TYPED）。
    /// 其余类别（Gemm/Attention/MoE/Structural）在 Phase 5-7 实现。
    pub fn from_op_kind_norm_activation(
        op: &CompilerOp,
        graph: &CompilerGraph,
    ) -> Option<Self> {
        // 从 op 第一个输入 tensor 推导 dtype
        let input_dtype = op.inputs.first()
            .and_then(|&tid| graph.tensor(tid))
            .map(|t| t.dtype)
            .unwrap_or(DType::F32);

        match &op.kind {
            // ── Activation（无参数 + 简单参数）──
            OpKind::Silu => Some(Op::Silu),
            OpKind::Gelu => Some(Op::Gelu),
            OpKind::Tanh => Some(Op::Tanh),
            OpKind::SwiGlu => Some(Op::SwiGlu),
            OpKind::GeGlu => Some(Op::GeGlu),
            OpKind::SwiGluClipped { limit } => Some(Op::SwiGluClipped { limit: *limit }),

            // ── Elementwise binary ──
            OpKind::Add => Some(Op::Add),
            OpKind::Mul => Some(Op::Mul),
            OpKind::ScaleConst { value } => Some(Op::ScaleConst { value: *value }),
            OpKind::Residual => Some(Op::Residual),

            // ── Norm（dtype 自描述）──
            OpKind::RmsNorm { feature_dim, eps } => Some(Op::RmsNorm(NormSpec {
                feature_dim: *feature_dim, eps: *eps, dtype: input_dtype, has_weight: true,
            })),
            OpKind::LayerNorm { feature_dim, eps } => Some(Op::LayerNorm(NormSpec {
                feature_dim: *feature_dim, eps: *eps, dtype: input_dtype, has_weight: true,
            })),
            OpKind::ValueNorm { feature_dim, eps } => Some(Op::ValueNorm(NormSpec {
                feature_dim: *feature_dim, eps: *eps, dtype: input_dtype, has_weight: false,
            })),
            OpKind::HeadRmsNorm { head_dim, eps } => Some(Op::HeadRmsNorm {
                head_dim: *head_dim, eps: *eps, dtype: input_dtype,
            }),

            _ => None, // 非 Norm/Activation 类别，Phase 5-7 处理
        }
    }

    /// Phase 5: Gemm/Quant 类别的 from_op_kind 转换。
    ///
    /// dtype 从 op.inputs[0] tensor 推导（Gemm 的 activation dtype）。
    pub fn from_op_kind_gemm_quant(
        op: &CompilerOp,
        graph: &CompilerGraph,
    ) -> Option<Self> {
        let input_dtype = op.inputs.first()
            .and_then(|&tid| graph.tensor(tid))
            .map(|t| t.dtype)
            .unwrap_or(DType::F32);

        match &op.kind {
            // Gemm
            OpKind::Gemm { m, n, k, dtype, trans_b } => Some(Op::Gemm(GemmSpec {
                m: m.clone(), n: *n, k: *k, dtype: *dtype, trans_b: *trans_b, has_bias: false,
            })),
            OpKind::GemmBias { m, n, k, dtype, trans_b } => Some(Op::GemmBias(GemmSpec {
                m: m.clone(), n: *n, k: *k, dtype: *dtype, trans_b: *trans_b, has_bias: true,
            })),
            OpKind::QuantGemm { m, n, k, quant_type } => Some(Op::QuantGemm(QuantGemmSpec {
                m: m.clone(), n: *n, k: *k, quant_type: *quant_type,
            })),
            OpKind::FusedRmsNormGemm { m, n, k, eps, dtype, trans_b } => Some(Op::FusedRmsNormGemm {
                m: m.clone(), n: *n, k: *k, eps: *eps, dtype: *dtype, trans_b: *trans_b,
            }),
            OpKind::MaskedGemm { m, n, k, dtype, trans_b } => Some(Op::MaskedGemm {
                m: m.clone(), n: *n, k: *k, dtype: *dtype, trans_b: *trans_b,
            }),

            // Quant
            OpKind::Dequantize { num_elements, block_size, bits } => Some(Op::Dequantize {
                num_elements: *num_elements, block_size: *block_size, bits: *bits,
            }),
            OpKind::VRangeQuant { seq_len, kv_dim, block_size, range_threshold } => Some(Op::VRangeQuant {
                seq_len: seq_len.clone(), kv_dim: *kv_dim,
                block_size: *block_size, range_threshold: *range_threshold,
            }),
            OpKind::QTapSTG { sink_ptr, step_index_ptr, dtype, q_dim, position, num_slots } => Some(Op::QTapSTG {
                sink_ptr: *sink_ptr, step_index_ptr: *step_index_ptr, dtype: *dtype,
                q_dim: q_dim.clone(), position: *position, num_slots: *num_slots,
            }),

            _ => None,
        }
    }
}
