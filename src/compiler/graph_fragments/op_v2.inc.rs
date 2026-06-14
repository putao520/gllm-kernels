// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 Op v2 — 胖 opcode 自描述架构 (FAT-OPCODE-ARCHITECTURE-V2)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// 与 OpKind（前端 IR）永久共存——类比 LLVM IR（前端）与 SelectionDAG（后端）。
// from_op_kind = IR Translator（compile pipeline 入口一次性翻译）。
// OpKind 服务于 graph 构建层（fusion/semantic_dag/registry），Op 服务于 lowering 层。
// 两者均为 SSOT 在各自层级，不是"待删除"代码。
//
// include! 模式：本文件被 include 到 graph.rs，与 op_kind.inc.rs / types.inc.rs
// 共享作用域。KvSource/RopeScaling/AttentionStrategy/GatherIndicesKind/SymDim/DType
// 等类型直接可见，无需 use。

/// Op v2 架构版本号 — 用于 JIT cache hash 失效管理。
///
/// 当 Op enum 结构变更（新增/删除变体、Spec 字段变更）时 bump 此版本号。
/// graph_content_hash 包含此常量，版本变更自动失效旧 JIT cache。
///
/// v1: 初始版本（Phase 0-7 完成，41 类别 from_op_kind 100% 覆盖）
/// v2: Phase 8 — VmInstr 复合指令加 dtype/guard/effect 字段（Predicate + MemEffect 新类型）
///     VecLoad/VecStore +predicate；GatherLoad/ScatterStore +dtype +predicate；
///     MemCopy +dtype +guard +effect。JIT 输出 VmInstr 结构变化，缓存必须失效。
pub const OPCODE_VERSION: u32 = 2;
// @trace REQ-FATOP-003 [entity:Op] OPCODE_VERSION JIT cache 版本管理

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
// @trace REQ-FATOP-004 [entity:AttentionSpec] AttentionSpec kv_source 自描述
// @trace REQ-FATOP-005 [entity:AttentionSpec] AttentionSpec sinks 配置
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
// @trace REQ-FATOP-006 [entity:NormSpec] NormSpec dtype 自描述
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormSpec {
    pub feature_dim: usize,
    pub eps: f32,
    pub dtype: DType,
    pub has_weight: bool,
}

// ── Gemm Specs ──

/// Gemm 自描述 spec。
// @trace REQ-FATOP-007 [entity:GemmSpec] GemmSpec 完整自描述
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
// @trace REQ-FATOP-001 [entity:Op] Op 枚举扁平结构（81 变体）
// @trace REQ-FATOP-002 [entity:AttentionSpec] [entity:NormSpec] [entity:GemmSpec] Spec struct 携带完整自描述元数据
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
    MlaKvCompress { m: SymDim, d_c: usize, hidden: usize },
    MlaQAbsorb { seq_len: SymDim, num_heads: usize, d_c: usize, head_dim: usize },
    MlaVRestore { seq_len: SymDim, num_heads: usize, d_c: usize, head_dim: usize },
    MlaRopeMerge { seq_len: SymDim, d_c: usize, d_rope: usize },

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
    L2Normalize { hidden: usize },

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
    MegaKernelDispatch { num_requests: usize, rst_ptr: u64, prefill_fn: u64, decode_fn: u64, chunked_fn: u64 },

    // ── MoE ──
    MoEGate { seq_len: SymDim, num_experts: usize, hidden: usize, top_k: usize },
    TopK { seq_len: SymDim, num_experts: usize, top_k: usize },
    WeightedSum { seq_len: SymDim, hidden: usize, top_k: usize },
    MoERouter { num_experts: usize, top_k: usize, hidden: usize, seq_len: SymDim },
    MoEDispatchPacked {
        seq_len: SymDim, num_experts: usize, top_k: usize,
        mxfp4_block_size: usize, swiglu_limit: f32, hidden: usize, intermediate_size: usize,
    },
    MoEConditionalAdd { seq_len: SymDim, hidden: usize, num_experts: usize, expert_idx: usize },

    // ── Structural ──
    Gather { table_rows: usize, embed_dim: usize, index_dim: SymDim, indices_kind: GatherIndicesKind, scale: Option<f32> },
    QuantGather { vocab_size: usize, hidden_dim: usize, index_dim: SymDim, quant_type: crate::quant::QuantType, scale: Option<f32> },
    SliceView { axis: usize, start: usize, end: usize },
    ColumnSlice { seq_len: SymDim, input_inner: usize, start: usize, slice_dim: usize },
    Transpose { perm: Vec<usize> },
    Reshape { target_shape: Vec<usize> },

    // ── KV cache ──
    KvScatterWrite {
        seq_len: usize, num_kv_heads: usize, head_dim: usize, kv_dim: usize,
        write_start: usize, layer_offset: usize, half_offset: usize,
        head_stride: usize, dtype_size: usize,
    },
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
    AltUpPredict { seq_len: SymDim, num_preds: usize, hidden: usize },
    AltUpCorrect { seq_len: SymDim, num_preds: usize, hidden: usize },
    AltUpInject { seq_len: SymDim, num_preds: usize, hidden: usize },

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

}

impl Op {
    /// 统一 from_op_kind 入口 — 组合 Phase 4-7 的 4 个类别方法。
    ///
    /// 调用方无需知道 op 类别，自动路由到对应的转换方法。
    /// 返回 None 表示无匹配（理论上不会发生，因为 4 个方法覆盖所有 OpKind 变体）。
    // @trace REQ-FATOP-013 [entity:Op] [entity:OpKind] from_op_kind IR Translator
    ///
    /// FAT-OPCODE §Phase 9: pub(crate) 限制 — 仅 add_op 内部和 lowering fallback 使用。
    /// 外部代码必须通过 CompilerOp.op_v2 缓存访问 Op（add_op 时已翻译）。
    pub(crate) fn from_op_kind(op: &CompilerOp, graph: &CompilerGraph) -> Option<Self> {
        Self::from_op_kind_norm_activation(op, graph)
            .or_else(|| Self::from_op_kind_gemm_quant(op, graph))
            .or_else(|| Self::from_op_kind_attention_moe(op, graph))
            .or_else(|| Self::from_op_kind_structural(op, graph))
    }

    /// 返回 op 的类别名称（用于 JIT 日志、调试、profiling）。
    pub fn category(&self) -> &'static str {
        match self {
            Op::RmsNorm(_) | Op::LayerNorm(_) | Op::ValueNorm(_) | Op::HeadRmsNorm { .. }
            | Op::QkNorm { .. } | Op::L2Normalize { .. } => "norm",
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
            | Op::WriteLogits { .. } | Op::EarlyExit { .. } => "sampling",
            Op::GuardrailCheck { .. } | Op::SgInject { .. } | Op::SgDetect { .. }
            | Op::CotStepCheck { .. } | Op::SessionKvRestore | Op::MmHiddenInject { .. } => "business",
            Op::Gather { .. } | Op::QuantGather { .. } | Op::SliceView { .. }
            | Op::ColumnSlice { .. } | Op::Transpose { .. } | Op::Reshape { .. }
            | Op::MegaKernelDispatch { .. } => "structural",
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

    /// Phase 6: Attention/MoE/MLA 类别的 from_op_kind 转换。
    pub fn from_op_kind_attention_moe(
        op: &CompilerOp,
        _graph: &CompilerGraph,
    ) -> Option<Self> {
        match &op.kind {
            // Attention
            OpKind::MultiHeadAttention { seq_len, num_heads, num_kv_heads, head_dim, causal, attention_sinks, kv_source } => {
                Some(Op::MultiHeadAttention(AttentionSpec {
                    geometry: AttentionGeometry {
                        num_q_heads: *num_heads, num_kv_heads: *num_kv_heads, head_dim: *head_dim,
                    },
                    mask: if *causal { AttentionMask::Causal } else { AttentionMask::Full },
                    kv_source: *kv_source,
                    sinks: if *attention_sinks { SinksSpec::Learnable } else { SinksSpec::None },
                    seq_len: seq_len.clone(),
                    dtype: DType::F32,
                }))
            }
            OpKind::CachedGQA { seq_len, total_seq, num_heads, num_kv_heads, head_dim, strategy, kv_dtype, kv_source } => {
                Some(Op::CachedGqa(CachedGqaSpec {
                    geometry: AttentionGeometry {
                        num_q_heads: *num_heads, num_kv_heads: *num_kv_heads, head_dim: *head_dim,
                    },
                    mask: AttentionMask::Causal,
                    kv_source: *kv_source,
                    seq_len: SymDim::Concrete(*seq_len),
                    total_seq: *total_seq,
                    kv_dtype: *kv_dtype,
                    strategy: strategy.clone(),
                }))
            }
            OpKind::MlaAttention { seq_len, num_heads, head_dim, d_c, d_rope, causal, kv_source } => {
                Some(Op::MlaAttention(MlaSpec {
                    seq_len: seq_len.clone(), num_heads: *num_heads, head_dim: *head_dim,
                    d_c: *d_c, d_rope: *d_rope, causal: *causal, kv_source: *kv_source,
                }))
            }

            // MLA sub-ops
            OpKind::MlaKvCompress { m, d_c, hidden } => Some(Op::MlaKvCompress {
                m: m.clone(), d_c: *d_c, hidden: *hidden,
            }),
            OpKind::MlaQAbsorb { seq_len, num_heads, head_dim, d_c } => Some(Op::MlaQAbsorb {
                seq_len: seq_len.clone(), num_heads: *num_heads, head_dim: *head_dim, d_c: *d_c,
            }),
            OpKind::MlaVRestore { seq_len, num_heads, head_dim, d_c } => Some(Op::MlaVRestore {
                seq_len: seq_len.clone(), num_heads: *num_heads, head_dim: *head_dim, d_c: *d_c,
            }),
            OpKind::MlaRopeMerge { seq_len, d_c, d_rope } => Some(Op::MlaRopeMerge {
                seq_len: seq_len.clone(), d_c: *d_c, d_rope: *d_rope,
            }),

            // MoE
            OpKind::MoEGate { seq_len, num_experts, hidden, top_k } => Some(Op::MoEGate {
                seq_len: SymDim::Concrete(*seq_len), num_experts: *num_experts, hidden: *hidden, top_k: *top_k,
            }),
            OpKind::TopK { seq_len, num_experts, top_k } => Some(Op::TopK {
                seq_len: SymDim::Concrete(*seq_len), num_experts: *num_experts, top_k: *top_k,
            }),
            OpKind::WeightedSum { seq_len, hidden, top_k } => Some(Op::WeightedSum {
                seq_len: SymDim::Concrete(*seq_len), hidden: *hidden, top_k: *top_k,
            }),
            OpKind::MoERouter { num_experts, top_k, hidden, seq_len } => Some(Op::MoERouter {
                num_experts: *num_experts, top_k: *top_k, hidden: *hidden, seq_len: seq_len.clone(),
            }),
            OpKind::MoEDispatchPacked {
                seq_len, num_experts, top_k, mxfp4_block_size, swiglu_limit, intermediate_size, hidden,
            } => Some(Op::MoEDispatchPacked {
                seq_len: seq_len.clone(), num_experts: *num_experts, top_k: *top_k,
                mxfp4_block_size: *mxfp4_block_size, swiglu_limit: *swiglu_limit,
                hidden: *hidden, intermediate_size: *intermediate_size,
            }),
            OpKind::MoEConditionalAdd { seq_len, hidden, num_experts, expert_idx } => Some(Op::MoEConditionalAdd {
                seq_len: seq_len.clone(), hidden: *hidden, num_experts: *num_experts, expert_idx: *expert_idx,
            }),

            _ => None,
        }
    }

    /// Phase 7: Structural/Embed/Vision-Audio/Generation/Business 类别的 from_op_kind 转换。
    pub fn from_op_kind_structural(
        op: &CompilerOp,
        _graph: &CompilerGraph,
    ) -> Option<Self> {
        match &op.kind {
            OpKind::RoPE { num_heads, head_dim, theta, partial, rope_scaling } => Some(Op::RoPE(RopeSpec {
                num_heads: *num_heads, head_dim: *head_dim, theta: *theta,
                partial: *partial, rope_scaling: rope_scaling.clone(),
            })),
            OpKind::DualRoPE { num_heads, head_dim, sliding_theta, sliding_partial, global_theta, global_partial, rope_scaling, layer_offset, layer_divisor, layer_remainder } => Some(Op::DualRoPE(DualRopeSpec {
                num_heads: *num_heads, head_dim: *head_dim,
                sliding_theta: *sliding_theta, sliding_partial: *sliding_partial,
                global_theta: *global_theta, global_partial: *global_partial,
                rope_scaling: rope_scaling.clone(),
                layer_offset: *layer_offset, layer_divisor: *layer_divisor, layer_remainder: *layer_remainder,
            })),
            OpKind::MeanPool { seq_len, hidden, cls_mode } => Some(Op::MeanPool { seq_len: *seq_len, hidden: *hidden, cls_mode: *cls_mode }),
            OpKind::QkNorm { head_dim, eps } => Some(Op::QkNorm { head_dim: *head_dim, eps: *eps }),
            OpKind::L2Normalize { hidden } => Some(Op::L2Normalize { hidden: *hidden }),
            OpKind::Argmax { vocab_size } => Some(Op::Argmax { vocab_size: *vocab_size }),
            OpKind::StoreToken => Some(Op::StoreToken),
            OpKind::CheckStopCondition => Some(Op::CheckStopCondition),
            OpKind::WriteLogits { target_indices } => Some(Op::WriteLogits { target_indices: target_indices.clone() }),
            OpKind::EarlyExit { anchor_layer } => Some(Op::EarlyExit { anchor_layer: *anchor_layer }),
            OpKind::GuardrailCheck { probe_offset } => Some(Op::GuardrailCheck { probe_offset: *probe_offset }),
            OpKind::SgInject { knowledge_offset, dim } => Some(Op::SgInject { knowledge_offset: *knowledge_offset, dim: *dim }),
            OpKind::SgDetect { detect_offset, hidden_dim } => Some(Op::SgDetect { detect_offset: *detect_offset, hidden_dim: *hidden_dim }),
            OpKind::CotStepCheck { shared_mem_offset } => Some(Op::CotStepCheck { shared_mem_offset: *shared_mem_offset }),
            OpKind::SessionKvRestore => Some(Op::SessionKvRestore),
            OpKind::MmHiddenInject { hidden_dim } => Some(Op::MmHiddenInject { hidden_dim: *hidden_dim }),
            OpKind::MegaKernelDispatch { num_requests, rst_ptr, prefill_fn, decode_fn, chunked_fn } => Some(Op::MegaKernelDispatch {
                num_requests: *num_requests, rst_ptr: *rst_ptr, prefill_fn: *prefill_fn, decode_fn: *decode_fn, chunked_fn: *chunked_fn,
            }),
            OpKind::Gather { table_rows, embed_dim, index_dim, indices_kind, scale } => Some(Op::Gather {
                table_rows: *table_rows, embed_dim: *embed_dim, index_dim: index_dim.clone(), indices_kind: indices_kind.clone(), scale: *scale,
            }),
            OpKind::QuantGather { vocab_size, hidden_dim, index_dim, quant_type, scale } => Some(Op::QuantGather {
                vocab_size: *vocab_size, hidden_dim: *hidden_dim, index_dim: index_dim.clone(), quant_type: *quant_type, scale: *scale,
            }),
            OpKind::SliceView { axis, start, end } => Some(Op::SliceView { axis: *axis, start: *start, end: *end }),
            OpKind::ColumnSlice { seq_len, input_inner, start, slice_dim } => Some(Op::ColumnSlice { seq_len: seq_len.clone(), input_inner: *input_inner, start: *start, slice_dim: *slice_dim }),
            OpKind::Transpose { perm } => Some(Op::Transpose { perm: perm.clone() }),
            OpKind::Reshape { target_shape } => Some(Op::Reshape { target_shape: target_shape.clone() }),
            OpKind::KvScatterWrite { seq_len, num_kv_heads, head_dim, kv_dim, write_start, layer_offset, half_offset, head_stride, dtype_size } => Some(Op::KvScatterWrite {
                seq_len: *seq_len, num_kv_heads: *num_kv_heads, head_dim: *head_dim, kv_dim: *kv_dim,
                write_start: *write_start, layer_offset: *layer_offset, half_offset: *half_offset,
                head_stride: *head_stride, dtype_size: *dtype_size,
            }),
            OpKind::KvCacheWrite { num_kv_heads, head_dim, seq_len } => Some(Op::KvCacheWrite {
                num_kv_heads: *num_kv_heads, head_dim: *head_dim, seq_len: seq_len.clone(),
            }),
            OpKind::VariableLengthBatch => Some(Op::VariableLengthBatch),
            OpKind::AttentionSkipMask { seq_len, threshold } => Some(Op::AttentionSkipMask { seq_len: seq_len.clone(), threshold: *threshold }),
            OpKind::ResidualWithTelemetry { hidden } => Some(Op::ResidualWithTelemetry { hidden: *hidden }),
            OpKind::EntropyGate { seq_len, vocab_size, entropy_threshold } => Some(Op::EntropyGate { seq_len: seq_len.clone(), vocab_size: *vocab_size, entropy_threshold: *entropy_threshold }),
            OpKind::KvCentroidPrefetch { seq_len, num_heads, head_dim, prefetch_distance } => Some(Op::KvCentroidPrefetch { seq_len: seq_len.clone(), num_heads: *num_heads, head_dim: *head_dim, prefetch_distance: *prefetch_distance }),
            OpKind::LayerBypass { threshold } => Some(Op::LayerBypass { threshold: *threshold }),
            OpKind::GateMask { hidden } => Some(Op::GateMask { hidden: *hidden }),
            OpKind::SoftmaxWithEntropy { vocab_size } => Some(Op::SoftmaxWithEntropy { vocab_size: *vocab_size }),
            OpKind::AltUpPredict { seq_len, num_preds, hidden } => Some(Op::AltUpPredict { seq_len: seq_len.clone(), num_preds: *num_preds, hidden: *hidden }),
            OpKind::AltUpCorrect { seq_len, num_preds, hidden } => Some(Op::AltUpCorrect { seq_len: seq_len.clone(), num_preds: *num_preds, hidden: *hidden }),
            OpKind::AltUpInject { seq_len, num_preds, hidden } => Some(Op::AltUpInject { seq_len: seq_len.clone(), num_preds: *num_preds, hidden: *hidden }),
            OpKind::DepthwiseConv1D { channels, kernel_size, causal } => Some(Op::DepthwiseConv1D { channels: *channels, kernel_size: *kernel_size, causal: *causal }),
            OpKind::PatchEmbed { patch_size, embed_dim, in_channels, image_size } => Some(Op::PatchEmbed { patch_size: *patch_size, embed_dim: *embed_dim, in_channels: *in_channels, image_size: *image_size }),
            OpKind::LearnedPos2D { num_patches, embed_dim } => Some(Op::LearnedPos2D { num_patches: *num_patches, embed_dim: *embed_dim }),
            OpKind::MtpDraft { depth, hidden_size, vocab_size } => Some(Op::MtpDraft { depth: *depth, hidden_size: *hidden_size, vocab_size: *vocab_size }),
            _ => None,
        }
    }
}
