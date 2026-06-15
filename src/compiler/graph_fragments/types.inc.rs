// CompilerGraph — DAG representation for the JIT inference compiler.
//
// The graph captures the computation of a single transformer layer as a
// directed acyclic graph of typed operations. Each operation reads from
// input tensors and produces output tensors. Tensors carry shape metadata
// and def-use chains (single producer, multiple consumers).
//
// Pipeline: LayerIR → CompilerGraph → (Fusion) → (ISA Lowering: codegen)

use std::collections::HashMap;
use crate::compiler::ir::LayerIR;
use crate::dispatch::device_profile::DeviceProfile;
use crate::types::DType;
use crate::traits::Activation;
use crate::types::CompilerError;

// SymDim::Symbolic 维度的编译时上界通过 `CompilerGraph.max_seq_len` 字段传递,
// 该字段从模型配置 `max_position_embeddings` 正向传播(REQ-SYMDIM-PAGED-KV §2.1)。
// 历史 `SYMDIM_MAX_SEQ_LEN = 2048` 硬编码常量已物理删除。

// ── Symbolic shape ─────────────────────────────────────────────────

/// A tensor dimension that is either a concrete compile-time value or a
/// symbolic name resolved at runtime via `ShapeBinding`.
///
/// - `Concrete(n)`: known at graph-build time; participates in loop-unrolling,
///   tile-size selection, and register allocation during JIT codegen.
/// - `Symbolic { name, max_value }`: unknown at build time; the generated kernel uses a
///   runtime-loaded stride/bound. The graph is compiled **once** and reused
///   across steps with different bindings (e.g. `total_seq` growing each step).
///   `max_value` is used for compile-time buffer allocation (scratchpad, pack buffers).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    /// Compile-time concrete dimension.
    Concrete(usize),
    /// Runtime symbolic dimension; resolved via `ShapeBinding` before execution.
    /// `max_value`: upper bound for buffer allocation at compile time.
    Symbolic { name: String, max_value: Option<usize> },
}

impl SymDim {
    /// Resolve to a concrete `usize`, looking up symbolic names in `binding`.
    pub fn resolve(&self, binding: &ShapeBinding) -> Result<usize, CompilerError> {
        match self {
            SymDim::Concrete(v) => Ok(*v),
            SymDim::Symbolic { name, .. } => binding
                .get(name)
                .copied()
                .ok_or_else(|| CompilerError::Internal(format!("unresolved symbolic dim: {name}"))),
        }
    }

    /// Return the concrete value if this is `Concrete`, else `None`.
    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            SymDim::Concrete(v) => Some(*v),
            SymDim::Symbolic { .. } => None,
        }
    }

    /// Return the maximum value for buffer allocation at compile time.
    /// For `Concrete`, returns the value itself.
    /// For `Symbolic`, returns `max_value` if set, otherwise the provided default.
    ///
    /// ARCH-SYMDIM-NO-UNWRAP: `default` 仅供 buffer 分配时的防御性兜底，
    /// 生产路径（成本模型、并行规划、融合决策）应使用 `max_for_allocation_strict()` 要求显式 max_value。
    pub fn max_for_allocation(&self, default: usize) -> usize {
        match self {
            SymDim::Concrete(v) => *v,
            SymDim::Symbolic { max_value, .. } => max_value.unwrap_or(default),
        }
    }

    /// 严格版本: Symbolic 无 max_value 时返回 Err，禁止静默兜底。
    /// 用于成本模型、并行规划、融合决策等影响行为的场景。
    pub fn max_for_allocation_strict(&self) -> Result<usize, CompilerError> {
        match self {
            SymDim::Concrete(v) => Ok(*v),
            SymDim::Symbolic { name, max_value } => max_value.ok_or_else(|| {
                CompilerError::Internal(format!(
                    "SymDim::Symbolic('{}').max_value 缺失 — ARCH-SYMDIM 要求构造时提供编译时上界",
                    name
                ))
            }),
        }
    }

    /// Returns `true` if this dimension is symbolic (runtime-resolved).
    pub fn is_symbolic(&self) -> bool {
        matches!(self, SymDim::Symbolic { .. })
    }
}

impl From<usize> for SymDim {
    fn from(v: usize) -> Self {
        SymDim::Concrete(v)
    }
}

/// Runtime shape binding: maps symbolic dimension names to concrete `usize` values.
///
/// Pass a `ShapeBinding` to `CompiledLayer::execute` (or any function that
/// needs to resolve symbolic dimensions) so that `SymDim::Symbolic` entries
/// can be resolved to actual sizes.
///
/// # Example
/// ```
/// let binding = ShapeBinding::new()
///     .bind("total_seq", 42)
///     .bind("batch_size", 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ShapeBinding {
    bindings: HashMap<String, usize>,
}

impl ShapeBinding {
    /// Create an empty binding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a name→value mapping (builder style).
    pub fn bind(mut self, name: impl Into<String>, value: usize) -> Self {
        self.bindings.insert(name.into(), value);
        self
    }

    /// Insert a name→value mapping in place.
    pub fn insert(&mut self, name: impl Into<String>, value: usize) {
        self.bindings.insert(name.into(), value);
    }

    /// Look up a symbolic name.
    pub fn get(&self, name: &str) -> Option<&usize> {
        self.bindings.get(name)
    }

    /// Resolve a `SymDim` using this binding.
    pub fn resolve(&self, dim: &SymDim) -> Result<usize, CompilerError> {
        dim.resolve(self)
    }
}

impl<const N: usize> From<[(&'static str, usize); N]> for ShapeBinding {
    fn from(arr: [(&'static str, usize); N]) -> Self {
        let mut b = ShapeBinding::new();
        for (k, v) in arr {
            b.insert(k, v);
        }
        b
    }
}


// ── Weight blob layout ────────────────────────────────────────────

/// Weight blob layout: maps graph input tensors to byte offsets.
/// Platform-independent — used by all backends.
#[derive(Debug, Clone)]
pub struct WeightLayout {
    /// (TensorId, byte_offset) for each weight tensor in blob order.
    pub offsets: Vec<(TensorId, usize)>,
    /// Total weight blob size in bytes.
    pub total_bytes: usize,
}

impl WeightLayout {
    /// Look up the byte offset of a weight tensor in the blob.
    pub fn offset_of(&self, tid: TensorId) -> Option<usize> {
        self.offsets.iter().find(|(t, _)| *t == tid).map(|(_, off)| *off)
    }
}

// ── Layer loop config (mega-kernel) ──────────────────────────────────

/// Configuration for the layer loop in mega-kernel VM IR lowering.
///
/// When present on a CompilerGraph, `lower_fusion_plan_inner()` wraps
/// layer ops (identified by "layer." name prefix) in `LoopBegin/LoopEnd`
/// and adjusts weight pointer addressing to stride across N layers.
///
/// Weight addressing in the loop:
///   layer_weight_base = weight_ptr + layer_blob_base_offset + byte_offset
///   actual_address = layer_weight_base + per_layer_relative_offset
///
/// Outside the loop, global weights use absolute offsets from weight_ptr.
#[derive(Debug, Clone)]
pub struct LayerLoopConfig {
    /// Number of layers (loop iteration count).
    pub num_layers: usize,
    /// Byte stride between consecutive layers in the weight blob.
    pub weight_stride: usize,
    /// Absolute byte offset of layer_0's weights in the weight blob
    /// (= embed_bytes for typical generative models).
    pub layer_blob_base_offset: usize,
    /// Indices into graph.inputs that are per-layer weights.
    /// These weights use offsets relative to layer start in WeightLayout.
    pub layer_weight_input_indices: Vec<usize>,
    /// Activation alias: the layer loop's final residual output tensor must
    /// share the same physical scratch buffer as the activation input tensor.
    /// This enables in-place update — residual writes `input + delta` back to
    /// the activation buffer, so the next iteration reads the updated value.
    pub activation_alias: Option<(TensorId, TensorId)>,
    /// Per-layer input stride (bytes) for AltUp PLE gate (Gemma 4 E2B/E4B).
    /// The per_layer_input buffer [L, S, hpl] is precomputed once, and each
    /// layer reads a different [S, hpl] slice. This stride equals S * hpl * 4.
    /// Zero when AltUp is not active (no per-layer input stepping needed).
    pub per_layer_input_stride: usize,
}

/// Heterogeneous layer loop configuration for models with alternating layer types.
///
/// For Gemma-4 E2B: 7 segments of [4 sliding + 1 full], with different
/// GEMM dimensions (head_dim) per layer type AND different FFN intermediate
/// per segment group (segments 0-2: small, segments 3-6: large).
#[derive(Debug, Clone)]
pub struct HeteroLayerLoopConfig {
    /// Number of segments (7 for Gemma-4 E2B).
    pub num_segments: usize,
    /// Sliding layers per segment (4 for Gemma-4 E2B).
    pub sliding_per_segment: usize,
    /// Byte stride for sliding layers in small-FFN segments.
    pub sliding_small_stride: usize,
    /// Byte stride for full layers in small-FFN segments.
    pub full_small_stride: usize,
    /// Byte stride for sliding layers in large-FFN segments.
    pub sliding_large_stride: usize,
    /// Byte stride for full layers in large-FFN segments.
    pub full_large_stride: usize,
    /// Segment stride for small-FFN segments.
    pub small_segment_stride: usize,
    /// Segment stride for large-FFN segments.
    pub large_segment_stride: usize,
    /// First segment index using large FFN.
    pub large_ffn_start_segment: usize,
    /// Absolute byte offset of the first layer's weights in the weight blob.
    pub layer_blob_base_offset: usize,
    /// Weight input indices for sliding_small layer ops.
    pub sliding_small_weight_input_indices: Vec<usize>,
    /// Weight input indices for full_small layer ops.
    pub full_small_weight_input_indices: Vec<usize>,
    /// Weight input indices for sliding_large layer ops.
    pub sliding_large_weight_input_indices: Vec<usize>,
    /// Weight input indices for full_large layer ops.
    pub full_large_weight_input_indices: Vec<usize>,
    /// Activation aliases for the combined loop.
    /// Each (input_tid, output_tid) pair causes output_tid to share the same
    /// physical scratch buffer as input_tid, enabling in-place update.
    /// For 4-type heterogeneous models, all 4 template outputs alias to
    /// the same activation buffer so that the layer loop can iterate in-place.
    pub activation_aliases: Vec<(TensorId, TensorId)>,
}

/// Platform-independent pointer source for a fusion group.
#[derive(Debug, Clone)]
pub enum PtrSource {
    /// Activation input (graph.inputs[0]) — passed via ABI arg 0
    ActivationInput,
    /// Weight blob at byte offset — passed via ABI arg 1 + offset
    WeightBlob(usize),
    /// Intermediate tensor in scratchpad at byte offset
    Scratchpad(usize),
    /// Graph output — passed via ABI output ptr
    GraphOutput,
    /// Graph output with byte offset for multi-output groups
    GraphOutputWithOffset(usize),
}

/// Platform-independent pointer binding for a fusion group.
#[derive(Debug, Clone, Default)]
pub struct GroupPointerMap {
    /// A matrix (GEMM input / elementwise input)
    pub a_ptr: Option<PtrSource>,
    /// B matrix (GEMM weight)
    pub b_ptr: Option<PtrSource>,
    /// Bias vector (GemmBias)
    pub bias_ptr: Option<PtrSource>,
    /// Norm weight
    pub norm_weight_ptr: Option<PtrSource>,
    /// Norm bias (LayerNorm)
    pub norm_bias_ptr: Option<PtrSource>,
    /// Output destination
    pub output_ptr: Option<PtrSource>,
}
// ── Multi-output ABI ──────────────────────────────────────────────

/// Multi-output kernel configuration (DEEP-001).
///
/// When `num_outputs > 1`, the JIT ABI changes: the `output` parameter becomes
/// `output_ptrs: *const *mut f32` — a pointer to an array of output pointers.
/// The codegen prologue loads each output pointer into a callee-saved register.
///
/// When `num_outputs == 1` (the default), the original single-output ABI is
/// preserved with zero overhead.
#[derive(Debug, Clone, Default)]
pub struct MultiOutputConfig {
    /// Output count (1 = traditional single-output ABI).
    pub num_outputs: usize,
    /// TensorIds of each output (empty for single-output).
    pub output_tensors: Vec<TensorId>,
}

impl MultiOutputConfig {
    /// Single-output (legacy ABI, zero overhead).
    pub fn single() -> Self {
        Self { num_outputs: 1, output_tensors: Vec::new() }
    }

    /// Multi-output: N output tensors written by one kernel.
    pub fn multi(tensors: Vec<TensorId>) -> Self {
        Self { num_outputs: tensors.len(), output_tensors: tensors }
    }

    /// Returns `true` when the kernel writes more than one output tensor.
    pub fn is_multi_output(&self) -> bool {
        self.num_outputs > 1
    }
}

// ── Identifiers ────────────────────────────────────────────────────

/// Unique tensor identifier within a CompilerGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

/// Unique operation identifier within a CompilerGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub u32);

// ── Tensor metadata ────────────────────────────────────────────────

/// Shape and type metadata for a tensor in the graph.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub id: TensorId,
    /// Shape dimensions — may mix `Concrete` and `Symbolic` entries.
    /// Symbolic entries are resolved at execution time via `ShapeBinding`.
    pub shape: Vec<SymDim>,
    pub dtype: DType,
    /// The op that produces this tensor (None for graph inputs).
    pub producer: Option<OpId>,
    /// Ops that consume this tensor.
    pub consumers: Vec<OpId>,
    /// Human-readable name for debugging.
    pub name: String,
}

impl TensorMeta {
    /// Product of all **concrete** dimensions (symbolic dims contribute 1).
    pub fn concrete_numel(&self) -> usize {
        self.shape.iter().map(|d| match d {
            SymDim::Concrete(v) => *v,
            SymDim::Symbolic { .. } => 1,
        }).product::<usize>().max(1)
    }

    /// Byte size of this tensor using concrete dimensions only.
    ///
    /// Weight tensors (graph inputs) use their declared dtype for byte size.
    /// Intermediate tensors (with a producer op) always use F32 (4 bytes) because
    /// JIT codegen accumulates in f32 regardless of source dtype.
    pub fn concrete_bytes(&self) -> usize {
        let elem_bytes = if self.producer.is_some() {
            4 // intermediate tensors: always f32 accumulator
        } else {
            self.dtype.size_bytes() // weight/input tensors: use declared dtype
        };
        self.concrete_numel() * elem_bytes
    }
}

// ── Attention strategy ─────────────────────────────────────────────

/// Attention algorithm strategy — auto-selected by hardware + sequence length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionStrategy {
    /// Standard QK^T → softmax → V, suitable for short sequences (seq_len < 1024)
    Naive,
    /// Tiled FlashAttention-2: O(N) memory, suitable for long sequences
    FlashV2 { block_m: usize, block_n: usize },
    /// GPU PagedAttention: paged KV cache for high-concurrency decode
    Paged { page_size: usize },
    /// Sliding Window: fixed window attention (Mistral-style)
    SlidingWindow { window_size: usize },
}

// ── RoPE scaling ───────────────────────────────────────────────────

/// RoPE frequency / temperature scaling family.
///
/// All variants are pure metadata — the actual `inv_freq` table and
/// `attention_scaling` (mscale) are derived in
/// [`crate::compiler::rope_scaling`]. The cos/sin table consumer
/// (`lower_rope_full`) stays oblivious: scaling is folded into the
/// pre-computed cos/sin scratchpad by the caller.
///
/// ## Variants
/// - `Yarn`: NTK-aware "by-parts" interpolation/extrapolation per dim
///   (https://arxiv.org/abs/2309.00071). Used by gpt-oss-20b long-context,
///   DeepSeek V2/V3, Qwen2.5-Long-Context. Requires `original_max_position`
///   (the model's pre-extension max context, e.g. 4096 for Llama-2 7B).
/// - `Linear`: Position interpolation (PI). `inv_freq[i] /= factor`.
///   Used by RedPajama / SuperHOT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// NTK-aware YaRN scaling.
    ///
    /// - `factor`: context-extension factor (e.g. 32.0 for 32× extension).
    /// - `beta_fast`: high-rotation cutoff (default 32.0). Dims rotating ≥
    ///   `beta_fast` cycles over `original_max_position` are extrapolated.
    /// - `beta_slow`: low-rotation cutoff (default 1.0). Dims rotating ≤
    ///   `beta_slow` cycles are interpolated. Mid range is linearly ramped.
    /// - `original_max_position`: the model's original (pre-extension) max
    ///   sequence length (e.g. 4096 for Llama-2, 8192 for gpt-oss).
    Yarn {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_position: usize,
    },
    /// Linear (Position Interpolation) scaling.
    /// `inv_freq[i] = base_inv_freq[i] / factor`.
    Linear { factor: f32 },
}

impl RopeScaling {
    /// Stable bit-pattern fingerprint for cache hashing (avoids `f32`/`usize`
    /// `Hash` discrepancies — same shape ⇒ identical bytes ⇒ identical hash).
    pub fn fingerprint_bytes(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        match self {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                buf[0] = 1;
                buf[4..8].copy_from_slice(&factor.to_bits().to_le_bytes());
                buf[8..12].copy_from_slice(&beta_fast.to_bits().to_le_bytes());
                buf[12..16].copy_from_slice(&beta_slow.to_bits().to_le_bytes());
                buf[16..24].copy_from_slice(&(*original_max_position as u64).to_le_bytes());
            }
            RopeScaling::Linear { factor } => {
                buf[0] = 2;
                buf[4..8].copy_from_slice(&factor.to_bits().to_le_bytes());
            }
        }
        buf
    }
}

// ── Operation kinds ────────────────────────────────────────────────

/// How a Gather op obtains its lookup indices.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
#[derive(Default)]
pub enum GatherIndicesKind {
    /// Indices come from a tensor (input_ids, etc.) — loaded from memory at runtime.
    #[default]
    Tensor,
    /// Indices are the loop counter itself: [0, 1, 2, ..., seq_len-1].
    /// Used for position embedding lookup where position_ids = arange(seq_len).
    Arange,
    /// All indices are zero: broadcast the first table row.
    /// Used for token_type embedding lookup where token_type_ids = zeros.
    Zeros,
}

