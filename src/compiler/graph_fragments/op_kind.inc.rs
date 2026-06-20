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
///
/// 终点：OpKind enum 已物理删除，单 IR (Op) 直接存储为 `op`。
/// 所有 lowering / fusion / registry 路径直接读 `op`。
#[derive(Debug, Clone)]
pub struct CompilerOp {
    pub id: OpId,
    /// Input tensor IDs (order matters: matches Op semantics).
    pub inputs: Vec<TensorId>,
    /// Output tensor IDs.
    pub outputs: Vec<TensorId>,
    /// Optional label for debugging / visualization.
    pub label: String,
    /// Layer loop execution guard. `Always` = no guard (zero overhead).
    pub guard: LayerCondition,
    /// 唯一 IR (Op)。胖 opcode 自描述，携带完整语义元数据（dtype/Spec struct）。
    pub op: crate::compiler::graph::Op,
}

impl CompilerOp {
    /// 直接返回已缓存的 Op（必填，无需 Option 解包）。
    ///
    /// 保留 `_graph` 参数仅为向后兼容调用点签名，方法体零开销直接返回缓存。
    pub fn op_resolved(&self, _graph: &CompilerGraph) -> Option<crate::compiler::graph::Op> {
        Some(self.op.clone())
    }

    /// 检查 Op 是否为 GEMM 类（胖 opcode 自描述）。
    pub fn op_is_gemm_like(&self, _graph: &CompilerGraph) -> bool {
        self.op.is_gemm_like()
    }

    /// 检查 Op 是否为 GemmBias（带 bias 的 GEMM）。
    pub fn op_is_gemm_with_bias(&self, _graph: &CompilerGraph) -> bool {
        self.op.is_gemm_with_bias()
    }

    /// 检查 Op 是否为 QuantGemm。
    pub fn op_is_quant_gemm(&self, _graph: &CompilerGraph) -> bool {
        self.op.is_quant_gemm()
    }

    /// 检查 Op 是否为 Norm 类（RmsNorm/LayerNorm/ValueNorm/HeadRmsNorm）。
    pub fn op_is_norm_like(&self, _graph: &CompilerGraph) -> bool {
        self.op.is_norm_like()
    }

    /// 提取 GEMM trans_b 参数（胖 opcode 自描述）。
    pub fn op_gemm_trans_b(&self, _graph: &CompilerGraph) -> bool {
        self.op.gemm_trans_b().unwrap_or(false)
    }

    /// 提取 GEMM 维度（胖 opcode 自描述）。
    pub fn op_gemm_dims(&self, _graph: &CompilerGraph) -> Option<(crate::compiler::graph::SymDim, usize, usize)> {
        self.op.gemm_dims()
    }

    /// 提取 GEMM dtype（胖 opcode 自描述）。
    pub fn op_gemm_dtype(&self, _graph: &CompilerGraph) -> Option<crate::types::DType> {
        self.op.gemm_dtype()
    }

    /// 提取 Attention head_dim（胖 opcode 自描述）。
    pub fn op_attention_head_dim(&self, _graph: &CompilerGraph) -> Option<usize> {
        self.op.attention_head_dim()
    }

    /// 输出别名到输入的 IR 元数据（胖 opcode 自描述）。
    pub fn op_output_aliases_input(&self, _graph: &CompilerGraph) -> Option<usize> {
        self.op.output_aliases_input()
    }

    /// 测试 fixture / 手工构造路径的 Op-first 构造器（单 IR）。
    ///
    /// 直接接受 `Op`（携带 dtype/Spec 自描述元数据），无需 kind_fallback。
    /// 用途：unit-test fixture 和 fusion pass 中手工创建独立 CompilerOp 的场景。
    pub fn new_from_op(
        id: OpId,
        op: crate::compiler::graph::Op,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        label: impl Into<String>,
        guard: LayerCondition,
    ) -> Self {
        CompilerOp {
            id,
            inputs,
            outputs,
            label: label.into(),
            guard,
            op,
        }
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

