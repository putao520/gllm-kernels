// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 Op — 单 IR (终点：OpKind enum 已物理删除)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// 收敛完成：Op 是唯一 IR，服务 graph 构建 + lowering + registry 全链路。
// 历史的 from_op_kind Translator（IR Translator）已随 OpKind enum 一起删除。
//
// include! 模式：本文件被 include 到 graph.rs，与 op_kind.inc.rs / types.inc.rs
// 共享作用域。KvSource/RopeScaling/AttentionStrategy/GatherIndicesKind/SymDim/DType
// 等类型直接可见，无需 use。

/// Op 架构版本号 — 用于 JIT cache hash 失效管理。
///
/// 当 Op enum 结构变更（新增/删除变体、Spec 字段变更）时 bump 此版本号。
/// graph_content_hash 包含此常量，版本变更自动失效旧 JIT cache。
///
/// v1: 初始版本（Phase 0-7 完成，41 类别 from_op_kind 100% 覆盖）
/// v2: Phase 8 — VmInstr 复合指令加 dtype/guard/effect 字段（Predicate + MemEffect 新类型）
///     VecLoad/VecStore +predicate；GatherLoad/ScatterStore +dtype +predicate；
///     MemCopy +dtype +guard +effect。JIT 输出 VmInstr 结构变化，缓存必须失效。
/// v3: — OpKind enum 物理删除，Op 成为唯一 IR。from_op_kind Translator 删除。
pub const OPCODE_VERSION: u32 = 3;
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
// §2 Op enum — 单 IR (终点)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 单 IR — 编译器全链路（graph 构建 + fusion + registry + lowering）的唯一算子枚举。
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
    Sigmoid,
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
    // @trace REQ-FATOP-025: KvCacheWrite 已物理删除 — attention FromCache lowering 内部覆盖 KV 写入

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
    /// 返回 op 的类别名称（用于 JIT 日志、调试、profiling）。
    pub fn category(&self) -> &'static str {
        match self {
            Op::RmsNorm(_) | Op::LayerNorm(_) | Op::ValueNorm(_) | Op::HeadRmsNorm { .. }
            | Op::QkNorm { .. } | Op::L2Normalize { .. } => "norm",
            Op::Gemm(_) | Op::GemmBias(_) | Op::QuantGemm(_)
            | Op::FusedRmsNormGemm { .. } | Op::MaskedGemm { .. } => "gemm",
            Op::Silu | Op::Gelu | Op::Tanh | Op::Sigmoid | Op::SwiGlu | Op::SwiGluClipped { .. } | Op::GeGlu
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
            Op::KvScatterWrite { .. } => "kv_cache",
            Op::VariableLengthBatch | Op::AttentionSkipMask { .. }
            | Op::ResidualWithTelemetry { .. } | Op::EntropyGate { .. }
            | Op::KvCentroidPrefetch { .. } | Op::LayerBypass { .. }
            | Op::GateMask { .. } | Op::SoftmaxWithEntropy { .. } => "advanced",
            Op::MtpDraft { .. } => "mtp",
        }
    }

    /// 是否为 GEMM 类 op（Gemm/GemmBias/FusedRmsNormGemm/MaskedGemm，不含 QuantGemm）。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::Gemm{..} | OpKind::GemmBias{..})`。
    pub fn is_gemm_like(&self) -> bool {
        matches!(self,
            Op::Gemm(_) | Op::GemmBias(_) | Op::FusedRmsNormGemm { .. } | Op::MaskedGemm { .. })
    }

    /// GEMM trans_b 参数（胖 opcode 自描述，无需 op.kind 反查）。
    /// None 表示非 GEMM 类 op。替代 fusion pass 中的
    /// `match op.kind { OpKind::Gemm{trans_b,..} | OpKind::GemmBias{trans_b,..} => *trans_b, _ => false }`。
    pub fn gemm_trans_b(&self) -> Option<bool> {
        match self {
            Op::Gemm(spec) | Op::GemmBias(spec) => Some(spec.trans_b),
            Op::FusedRmsNormGemm { trans_b, .. } | Op::MaskedGemm { trans_b, .. } => Some(*trans_b),
            _ => None,
        }
    }

    /// 是否为带 bias 的 GEMM（GemmBias）。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::GemmBias{..})`。
    pub fn is_gemm_with_bias(&self) -> bool {
        matches!(self, Op::GemmBias(_))
    }

    /// 是否为 Norm 类 op（RmsNorm/LayerNorm/ValueNorm/HeadRmsNorm）。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::RmsNorm{..} | ...)`。
    pub fn is_norm_like(&self) -> bool {
        matches!(self,
            Op::RmsNorm(_) | Op::LayerNorm(_) | Op::ValueNorm(_) | Op::HeadRmsNorm { .. })
    }

    /// 是否为 QuantGemm。
    /// 替代 fusion pass 中的 `matches!(op.kind, OpKind::QuantGemm{..})`。
    pub fn is_quant_gemm(&self) -> bool {
        matches!(self, Op::QuantGemm(_))
    }

    /// 输出别名到输入的 IR 元数据（胖 opcode 自描述）。
    ///
    /// 替代 OpKind::output_aliases_input。返回 Some(input_idx) 表示 output[0] 与
    /// input[input_idx] 共享同一物理 slot（side-channel/in-place op），返回 None
    /// 表示 output 需要独立分配。
    ///
    /// - SgDetect/SgInject：side-channel op，main-path data 透传，输出别名到 input[0]
    /// - 其他 op：输出独立 scratchpad slot
    pub fn output_aliases_input(&self) -> Option<usize> {
        match self {
            Op::SgDetect { .. } | Op::SgInject { .. } => Some(0),
            _ => None,
        }
    }

    /// 提取 GEMM 维度（胖 opcode 自描述）。
    /// 替代 `extract_gemm_dims_sym` 中
    /// `match op.kind { OpKind::Gemm{m,n,k,..} | OpKind::GemmBias{m,n,k,..} | ... => (m,n,k) }`。
    /// 返回 None 表示非 GEMM 类 op。
    pub fn gemm_dims(&self) -> Option<(SymDim, usize, usize)> {
        match self {
            Op::Gemm(spec) | Op::GemmBias(spec) => Some((spec.m.clone(), spec.n, spec.k)),
            Op::QuantGemm(spec) => Some((spec.m.clone(), spec.n, spec.k)),
            Op::FusedRmsNormGemm { m, n, k, .. } | Op::MaskedGemm { m, n, k, .. } => Some((m.clone(), *n, *k)),
            _ => None,
        }
    }

    /// 提取 Attention head_dim（胖 opcode 自描述）。
    /// 替代 `match op.kind { OpKind::MultiHeadAttention{head_dim,..} | OpKind::RoPE{head_dim,..} => *head_dim }`。
    pub fn attention_head_dim(&self) -> Option<usize> {
        match self {
            Op::MultiHeadAttention(spec) => Some(spec.geometry.head_dim),
            Op::CachedGqa(spec) => Some(spec.geometry.head_dim),
            Op::MlaAttention(spec) => Some(spec.head_dim),
            Op::RoPE(spec) => Some(spec.head_dim),
            Op::DualRoPE(spec) => Some(spec.head_dim),
            _ => None,
        }
    }

    /// 计算用于 JIT cache hash 的内容指纹（胖 opcode 自描述）。
    /// 替代 mod.rs::graph_content_hash 中
    /// `std::mem::discriminant(&op.kind).hash() + match OpKind { 参数 hash }`。
    ///
    /// 包含：Op discriminant + 关键参数（eps/维度/dtype/trans_b/limit 等）。
    /// 不同参数组合产生不同 hash，确保 JIT cache 正确失效。
    pub fn content_hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Op::RmsNorm(spec) | Op::LayerNorm(spec) | Op::ValueNorm(spec) => {
                spec.eps.to_bits().hash(state);
                spec.feature_dim.hash(state);
            }
            Op::HeadRmsNorm { head_dim, eps, .. } => {
                head_dim.hash(state);
                eps.to_bits().hash(state);
            }
            Op::QkNorm { head_dim, eps } => {
                head_dim.hash(state);
                eps.to_bits().hash(state);
            }
            Op::L2Normalize { hidden } => { hidden.hash(state); }
            Op::Gemm(spec) | Op::GemmBias(spec) => {
                spec.m.hash(state);
                spec.n.hash(state);
                spec.k.hash(state);
                std::mem::discriminant(&spec.dtype).hash(state);
                spec.trans_b.hash(state);
            }
            Op::QuantGemm(spec) => {
                spec.m.hash(state);
                spec.n.hash(state);
                spec.k.hash(state);
                std::mem::discriminant(&spec.quant_type).hash(state);
            }
            Op::Dequantize { num_elements, block_size, bits } => {
                num_elements.hash(state);
                block_size.hash(state);
                bits.hash(state);
            }
            Op::RoPE(spec) => {
                spec.num_heads.hash(state);
                spec.head_dim.hash(state);
                spec.theta.to_bits().hash(state);
                spec.partial.to_bits().hash(state);
                match &spec.rope_scaling {
                    None => 0u8.hash(state),
                    Some(s) => {
                        1u8.hash(state);
                        s.fingerprint_bytes().hash(state);
                    }
                }
            }
            Op::DualRoPE(spec) => {
                spec.num_heads.hash(state);
                spec.head_dim.hash(state);
                spec.sliding_theta.to_bits().hash(state);
                spec.sliding_partial.to_bits().hash(state);
                match &spec.rope_scaling {
                    None => 0u8.hash(state),
                    Some(s) => {
                        1u8.hash(state);
                        s.fingerprint_bytes().hash(state);
                    }
                }
            }
            Op::Transpose { perm } => { perm.hash(state); }
            Op::Reshape { target_shape } => { target_shape.hash(state); }
            Op::SliceView { axis, start, end } => {
                axis.hash(state); start.hash(state); end.hash(state);
            }
            Op::MeanPool { seq_len, hidden, .. } => {
                seq_len.hash(state);
                hidden.hash(state);
            }
            Op::SwiGluClipped { limit } => { limit.to_bits().hash(state); }
            Op::MoEDispatchPacked { num_experts, top_k, mxfp4_block_size, swiglu_limit,
                                    intermediate_size, hidden, seq_len } => {
                num_experts.hash(state);
                top_k.hash(state);
                mxfp4_block_size.hash(state);
                swiglu_limit.to_bits().hash(state);
                intermediate_size.hash(state);
                hidden.hash(state);
                seq_len.hash(state);
            }
            // 其他 op discriminant 已经足够（unit 或参数不影响 JIT cache）
            _ => {}
        }
    }


    /// 仅 Gemm/GemmBias/FusedRmsNormGemm/MaskedGemm 携带 dtype；
    /// QuantGemm 返回 None（调用方通过 graph.infer_computation_dtype 推导）。
    pub fn gemm_dtype(&self) -> Option<DType> {
        match self {
            Op::Gemm(spec) | Op::GemmBias(spec) => Some(spec.dtype),
            Op::FusedRmsNormGemm { dtype, .. } | Op::MaskedGemm { dtype, .. } => Some(*dtype),
            _ => None,
        }
    }
}

/// Reduction 类 op 几何参数（胖 opcode 自描述）。
/// 替代 fusion/lowering 路径中
/// `match op.kind { OpKind::MeanPool{seq_len,hidden,cls_mode} | OpKind::L2Normalize{hidden} => ... }`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReductionGeometry {
    pub seq_len: usize,
    pub hidden: usize,
    pub cls_mode: bool,
}

/// Norm variant tag（graph 层不依赖 codegen 层 NormKind，调用方做映射）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormVariant {
    RmsNorm,
    ValueNorm,
    LayerNorm,
    HeadRmsNorm,
}

/// NormLike op 元数据（胖 opcode 自描述）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormMeta {
    pub variant: NormVariant,
    pub eps: f32,
    pub has_weight: bool,
}

impl Op {
    /// Reduction 类 op 几何参数（胖 opcode 自描述）。
    /// 替代 `try_dispatch_reduction` 中
    /// `match op.kind { OpKind::MeanPool{seq_len,hidden,cls_mode} | OpKind::L2Normalize{hidden} => (seq_bound, hidden) }`。
    pub fn reduction_geometry(&self) -> Option<ReductionGeometry> {
        match self {
            Op::MeanPool { seq_len, hidden, cls_mode } => Some(ReductionGeometry {
                seq_len: *seq_len,
                hidden: *hidden,
                cls_mode: *cls_mode,
            }),
            Op::L2Normalize { hidden } => Some(ReductionGeometry {
                seq_len: 1,
                hidden: *hidden,
                cls_mode: false,
            }),
            _ => None,
        }
    }

    /// 提取 NormLike op 的 eps + has_weight + variant name（胖 opcode 自描述）。
    /// variant name 用于调用方映射到对应 NormKind（避免 graph 层依赖 codegen 层）。
    /// 替代 `build_norm_pattern` / `emit_normlike_inline` 中的
    /// `match op.kind { OpKind::RmsNorm{eps,..} | OpKind::ValueNorm{eps,..} => eps, ... }`。
    pub fn norm_meta(&self) -> Option<NormMeta> {
        match self {
            Op::RmsNorm(spec) => Some(NormMeta { variant: NormVariant::RmsNorm, eps: spec.eps, has_weight: true }),
            Op::ValueNorm(spec) => Some(NormMeta { variant: NormVariant::ValueNorm, eps: spec.eps, has_weight: false }),
            Op::LayerNorm(spec) => Some(NormMeta { variant: NormVariant::LayerNorm, eps: spec.eps, has_weight: true }),
            Op::HeadRmsNorm { eps, .. } => Some(NormMeta { variant: NormVariant::HeadRmsNorm, eps: *eps, has_weight: true }),
            _ => None,
        }
    }

}
