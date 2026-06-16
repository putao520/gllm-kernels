//! Mega-Kernel ABI — whole-model single-entry-point function signature.
//!
//! Replaces the per-node `CompiledLayerFn` (10 params) with a single mega-kernel
//! that does: embedding → N layer loops → logits producer → sampling → generate loop.
//!
//! See SPEC 08-EXECUTOR.md §4.1.3 for the canonical ABI definition.

// ── Business configuration (§1.5) ─────────────────────────────────────

/// Output mode — multiple output heads for the same model (HEAD-ROUTING.md).
///
/// Runtime switching via `output_mode_selector` ABI parameter = zero recompilation.
///
/// DELETED: OutputModeDispatch JMP table mechanism removed (SPEC/39 §7).
/// Head Routing now compiles multiple MegaKernel functions instead.
/// OutputMode enum remains for BUILD-stage strategy selection only.
#[derive(Debug, Clone)]
pub enum OutputMode {
    /// Autoregressive generation: logits producer → argmax → store → check → loop
    Generate { max_new_tokens: usize, eos_token_id: u32 },
    /// Binary classification: logits producer → write positive/negative token logits
    ClassifyBinary { positive_token_id: u32, negative_token_id: u32 },
    /// Multi-way classification: logits producer → write N label token logits
    ClassifyMultiway { label_token_ids: Vec<u32> },
    /// Mid-layer encoding: truncate at anchor layer → pool hidden state
    EncodeToLayer { anchor_layer: usize, pool_mode: PoolMode },
}

#[derive(Debug, Clone)]
pub enum PoolMode {
    LastToken,
    MeanPool,
    ClsToken,
}

/// Semantic Gatekeeper injection configuration.
#[derive(Debug, Clone)]
pub struct SgConfig {
    /// Layer index where hidden state is extracted for detection.
    pub detect_layer: usize,
    /// Shared memory offset for knowledge residual vector injection (after embed).
    pub inject_offset: usize,
    /// Shared memory offset for hidden state extraction (at detect layer).
    pub detect_offset: usize,
    /// Q-Tap ring buffer configuration for extracting Q vectors at detection layer.
    /// When Some, a QTapSTG op is inserted after q_proj GEMM at the detect_layer.
    pub q_tap: Option<super::graph::QTapGraphConfig>,
}

/// CoT Step Hook configuration.
#[derive(Debug, Clone)]
pub struct CotStepConfig {
    /// Shared memory offset for step control flags.
    pub shared_mem_offset: usize,
}

/// MTP (Multi-Token Prediction) configuration for JIT codegen.
///
/// When present, the mega-kernel generates K candidate tokens per decode step
/// by running depth additional projection passes after the main logits producer.
/// Each depth projects the current hidden state to vocab logits via a dedicated
/// weight matrix packed in the weight blob.
#[derive(Debug, Clone)]
pub struct MtpKernelConfig {
    /// Number of prediction depths (K in MTP-K).
    /// Each depth requires one projection matrix of shape [hidden_size, vocab_size].
    pub depth: usize,
    /// Hidden dimension size (same as model hidden_size).
    pub hidden_size: usize,
    /// Vocabulary size (same as model vocab_size).
    pub vocab_size: usize,
}

/// Business configuration for mega-kernel compilation (§1.5).
///
/// Declared at model load time → affects CompilerGraph construction →
/// compiled as conditional branches embedded in the mega-kernel.
/// Disabled features produce zero instruction overhead (not even runtime if-else).
#[derive(Debug, Clone)]
pub struct BusinessConfig {
    /// Output modes to compile (can be multiple for zero-recompilation switching).
    /// SPEC/39 NOTE: 编译器不直接读取此字段。output_modes 通过 GraphTopologyAnalysis
    /// 间接使用（has_classify_binary/has_classify_multiway/has_encode_to_layer/max_new_tokens/eos_token_id）。
    pub output_modes: Vec<OutputMode>,
    /// Whether to compile post_node veto probes (GuardrailCheck ops).
    /// SPEC/39 NOTE: 编译器不读取此字段。GuardrailCheck op 存在性由图拓扑推导。
    pub guardrail_enabled: bool,
    /// Semantic Gatekeeper configuration (knowledge injection + hidden extraction).
    /// SPEC/39 NOTE: 编译器不读取此字段。SgDetect/SgInject op 存在性由图拓扑推导。
    pub semantic_gatekeeper: Option<SgConfig>,
    /// Intent Recall anchor layer (enables EarlyExit op at this layer).
    /// SPEC/39 NOTE: 编译器不读取此字段。EarlyExit op 存在性由图拓扑推导。
    pub intent_anchor_layer: Option<usize>,
    /// CoT Step Hook configuration (step control after each inference step).
    /// SPEC/39 NOTE: 编译器不读取此字段。CotStepHook op 存在性由图拓扑推导。
    pub cot_step_hook: Option<CotStepConfig>,
    /// Whether model uses HeadRmsNorm on Q/K after projection (Qwen3 q_norm/k_norm).
    // REMOVED (SPEC/39): has_head_rms_norm — now derived from Op::HeadRmsNorm topology.
    // REMOVED (SPEC/39): head_rms_norm_eps — now derived from Op::HeadRmsNorm { eps: , dtype: DType::F32 }.
    // REMOVED (SPEC/39): ffn_activation — now derived from Op::SwiGlu/GeGlu/Gelu topology.
    // REMOVED (SPEC/39): has_qk_norm — now derived from Op::QkNorm topology.
    // REMOVED (SPEC/39): has_value_norm — now derived from Op::ValueNorm topology.
    // REMOVED (SPEC/39): value_norm_eps — now derived from Op::ValueNorm(NormSpec { feature_dim: feature_dim, eps: , dtype: DType::F32, has_weight: false }).
    // REMOVED (SPEC/39): logit_softcapping — now derived from Op::LogitSoftcap { cap:  } topology.
    // REMOVED (SPEC/39): embedding_scale — now derived from OpKind::Gather::scale / OpKind::QuantGather::scale.
    /// Whether to compile session KV cache restore (cross-turn KV reuse).
    /// When enabled, embed phase checks session_position > 0 → skip processed tokens.
    /// SPEC/39 NOTE: 编译器不读取此字段。SessionSkipProcessed op 存在性由图拓扑推导。
    pub session_enabled: bool,
    /// Whether to compile multimodal fused hidden injection.
    /// When enabled, embed phase reads fused_hidden_ptr → ADD to token embedding.
    /// SPEC/39 NOTE: 编译器不读取此字段。MultimodalInject op 存在性由图拓扑推导。
    pub multimodal_enabled: bool,
    /// Whether to compile JIT debug instrumentation (INT3 breakpoints + source map).
    /// When disabled (default): zero instruction overhead, no source map generated.
    /// When enabled: DebugBreakpoint/DebugMarker VmInstr inserted at key points,
    ///   INT3 instructions in machine code, JitSourceMap attached to compile output.
    /// SPEC/39 NOTE: 此字段是唯一合法的编译器直接读取参数——debug_jit 影响全局
    /// codegen 行为（INT3 插入、source map 生成），不是 OpKind 可表达的。
    pub debug_jit: bool,
    // REMOVED (SPEC/39): mtp_config — now derived from topology.mtp_config (Op::MtpDraft).
}

impl Default for BusinessConfig {
    fn default() -> Self {
        Self {
            output_modes: vec![OutputMode::Generate {
                max_new_tokens: 512,
                eos_token_id: 2,
            }],
            guardrail_enabled: false,
            semantic_gatekeeper: None,
            intent_anchor_layer: None,
            cot_step_hook: None,
            session_enabled: false,
            multimodal_enabled: false,
            debug_jit: false,
        }
    }
}

// ── ABI function signature ─────────────────────────────────────────

/// Signature of a compiled mega-kernel function.
///
/// All stack parameters use integer types to ensure they land on the stack
/// under x86-64 SysV ABI. Floats (temperature, top_p) are passed as `u32`
/// via `f32::to_bits()` — the callee reconstructs with `f32::from_bits()`.
///
/// ```text
/// fn(input_ids_ptr, weight_blob_ptr, kv_cache_ptr, positions_ptr,
///    aux_ptr, batch_size,    ← register params (rdi..r9)
///    prompt_len, scratchpad_ptr, output_tokens_ptr,
///    temperature_u32, top_k, top_p_u32, max_new_tokens, eos_token_id,
///    hook_ctx_ptr, telemetry_ptr)
///    ← stack params ([rbp+16]..[rbp+88])
///    → rax: generated token count (generate) | 0 (classify/encode)
/// ```
pub type MegaKernelFn = unsafe extern "C" fn(
    *const u32,   // arg 0: input_ids_ptr         → rdi  (prompt token ID array)
    *const u8,    // arg 1: weight_blob_ptr        → rsi  (all weights contiguous)
    *mut u8,      // arg 2: kv_cache_ptr           → rdx
    *const u32,   // arg 3: positions_ptr          → rcx
    *const u8,    // arg 4: aux_ptr                → r8   (KV-V half pointers)
    usize,        // arg 5: batch_size             → r9
    // Stack parameters (8-byte aligned, all usize to guarantee 8-byte passing):
    usize,        // arg 6: prompt_len             → [rbp+16]  (SymDim runtime binding)
    *mut u8,      // arg 7: scratchpad_ptr         → [rbp+24]
    *mut u32,     // arg 8: output_tokens_ptr      → [rbp+32]
    usize,        // arg 9: temperature_u32        → [rbp+40] (f32::to_bits() as usize)
    usize,        // arg 10: top_k                 → [rbp+48]
    usize,        // arg 11: top_p_u32             → [rbp+56] (f32::to_bits() as usize)
    usize,        // arg 12: max_new_tokens        → [rbp+64]
    usize,        // arg 13: eos_token_id          → [rbp+72]
    *const u8,    // arg 14: hook_ctx_ptr          → [rbp+80]
    *mut u8,      // arg 15: telemetry_ptr         → [rbp+88]
    usize,        // arg 16: session_position      → [rbp+96] (0 = new session, >0 = resume position)
    *const u8,    // arg 17: fused_hidden_ptr      → [rbp+104] (NULL = no multimodal injection)
    usize,        // arg 18: num_mm_tokens          → [rbp+112] (number of multimodal tokens to inject)
    *const u8,    // arg 19: callback_table_ptr    → [rbp+120] (NULL = no callbacks, C-style fn_ptr array)
    *const u32,   // arg 20: page_table_ptr        → [rbp+128] (NULL = contiguous KV, u32[] = paged KV)
    *const u8,    // arg 21: batch_ctx_ptr         → [rbp+136] (NULL = single-seq legacy, non-NULL = batch mode)
) -> usize;      // → rax: generated token count (generate) | 0 (classify/encode)

/// ABI parameter names in order. Used by SymDimSlotMap to resolve
/// symbolic dimensions and runtime values.
pub const MEGA_KERNEL_PARAMS: &[&str] = &[
    "input_ids_ptr",        // arg 0  → rdi
    "weight_blob_ptr",      // arg 1  → rsi
    "kv_cache_ptr",         // arg 2  → rdx
    "positions_ptr",        // arg 3  → rcx
    "aux_ptr",              // arg 4  → r8
    "batch_size",           // arg 5  → r9
    "prompt_len",           // arg 6  → [rbp+16]
    "scratchpad_ptr",       // arg 7  → [rbp+24]
    "output_tokens_ptr",    // arg 8  → [rbp+32]
    "temperature_u32",      // arg 9  → [rbp+40] (f32::to_bits())
    "top_k",                // arg 10 → [rbp+48]
    "top_p_u32",            // arg 11 → [rbp+56] (f32::to_bits())
    "max_new_tokens",       // arg 12 → [rbp+64]
    "eos_token_id",         // arg 13 → [rbp+72]
    "hook_ctx_ptr",         // arg 14 → [rbp+80]
    "telemetry_ptr",        // arg 15 → [rbp+88]
    "session_position",     // arg 16 → [rbp+96] (0 = new session, >0 = resume position)
    "fused_hidden_ptr",     // arg 17 → [rbp+104] (NULL = no multimodal injection)
    "num_mm_tokens",        // arg 18 → [rbp+112] (number of multimodal tokens to inject)
    "callback_table_ptr",   // arg 19 → [rbp+120] (NULL = no callbacks)
    "page_table_ptr",       // arg 20 → [rbp+128] (NULL = contiguous KV, u32[] = paged KV)
    "batch_ctx_ptr",        // arg 21 → [rbp+136] (NULL = single-seq legacy mode, non-NULL = batch mode)
];

/// Stack parameter byte offsets (relative to [rbp+16] after standard prologue).
/// First 6 params in registers (rdi..r9), remaining on stack.
/// All stack params are 8-byte aligned (usize/u32/pointer promoted to 8 bytes by SysV ABI).
pub const MEGA_KERNEL_STACK_OFFSETS: &[i32] = &[
    16,  // prompt_len             → [rbp+16]
    24,  // scratchpad_ptr         → [rbp+24]
    32,  // output_tokens_ptr      → [rbp+32]
    40,  // temperature_u32        → [rbp+40]
    48,  // top_k                  → [rbp+48]
    56,  // top_p_u32              → [rbp+56]
    64,  // max_new_tokens         → [rbp+64]
    72,  // eos_token_id           → [rbp+72]
    80,  // hook_ctx_ptr           → [rbp+80]
    88,  // telemetry_ptr          → [rbp+88]
    96,  // session_position       → [rbp+96]
    104, // fused_hidden_ptr       → [rbp+104]
    112, // num_mm_tokens          → [rbp+112]
    120, // callback_table_ptr     → [rbp+120]
    128, // page_table_ptr         → [rbp+128]
    136, // batch_ctx_ptr          → [rbp+136]
];

// ── Model geometry config ──────────────────────────────────────────

/// Model geometry for mega-kernel compilation.
///
/// All fields are determined at model load time from `ModelGeometry`.
/// Slim runtime configuration for graph-driven mega-kernel compilation.
/// Geometry (hidden, num_heads, etc.) is derived from the CompilerGraph.
/// Only non-derivable fields remain here.
///
/// SPEC/39: BusinessConfig has been removed from CompileConfig — the compiler
/// reads no business parameters except `debug_jit`, which is the sole field
/// that affects global codegen behavior (INT3 insertion, source map generation)
/// and cannot be expressed as an OpKind. All other business fields (output_modes,
/// guardrail_enabled, etc.) affect BUILD-stage graph construction only, and are
/// not read by the compiler at all.
#[derive(Debug, Clone)]
pub struct CompileConfig {
    /// Maximum sequence length (buffer allocation, NOT in tensor shapes).
    pub max_seq_len: usize,
    /// Enable JIT debug instrumentation (INT3 breakpoints + source map).
    /// This is the only "business" parameter the compiler reads directly —
    /// debug_jit affects global codegen behavior, not something OpKind can express.
    /// Promoted from BusinessConfig to CompileConfig top level (SPEC/39).
    pub debug_jit: bool,
    /// Heterogeneous layer configuration.
    pub hetero: Option<HeteroLayerConfig>,
    /// SPEC REQ-UMK-001: Compilation target — selects between CPU (native JIT
    /// machine code) and GPU (PTX/HIP/MSL source) codegen paths inside
    /// `InferenceCompiler::compile()`. Defaults to `Cpu` when constructed via
    /// `Default::default()`.
    pub target: CompileTarget,
}

impl Default for CompileConfig {
    fn default() -> Self {
        CompileConfig {
            max_seq_len: 0,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::default(),
        }
    }
}

/// SPEC REQ-UMK-001: Target backend for the unified `compile()` entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileTarget {
    /// CPU native JIT — produces `CompileOutput::Cpu(MegaKernelCompileOutput)`.
    Cpu,
    /// GPU PTX/HIP/MSL — produces `CompileOutput::Gpu(GpuMegaKernelOutput)`.
    /// `sm_version` selects the PTX target SM version (e.g. 600, 800, 1200).
    /// Ignored for HIP backend (uses MI300 defaults internally).
    Gpu { sm_version: u32 },
}

impl Default for CompileTarget {
    fn default() -> Self {
        CompileTarget::Cpu
    }
}

/// Configuration for models with heterogeneous layer types.
///
/// Gemma-4 E2B has 35 layers in a repeating pattern of [4 sliding + 1 full] × 7 segments.
/// Additionally, segments 0-2 use small FFN, segments 3-6 use large FFN.
/// The mega-kernel compiles 7 segments with per-type weight strides.
#[derive(Debug, Clone)]
pub struct HeteroLayerConfig {
    /// Number of segments (7 for Gemma-4 E2B).
    pub num_segments: usize,
    /// Sliding layers per segment (4 for Gemma-4 E2B).
    pub sliding_per_segment: usize,
    /// Head dimension for sliding attention layers.
    pub sliding_head_dim: usize,
    /// Number of Q heads for sliding layers (Gemma 4 E2B: 4, derived from q_dim / head_dim).
    pub sliding_num_q_heads: usize,
    /// Number of KV heads for sliding layers.
    pub sliding_num_kv_heads: usize,
    /// Head dimension for full attention layers.
    pub full_head_dim: usize,
    /// Number of Q heads for full layers (Gemma 4 E2B: 8, derived from q_dim / head_dim).
    pub full_num_q_heads: usize,
    /// Number of KV heads for full layers (same as sliding for Gemma-4).
    pub full_num_kv_heads: usize,
    /// Original layer indices for full attention layers (for weight packing / KV cache mapping).
    pub full_layer_indices: Vec<usize>,
    /// FFN intermediate size for "small FFN" segments (segments 0..large_ffn_start_segment).
    pub small_intermediate: usize,
    /// FFN intermediate size for "large FFN" segments (segments large_ffn_start_segment..num_segments).
    pub large_intermediate: usize,
    /// First segment index that uses large_intermediate (3 for Gemma-4 E2B: segments 3-6).
    pub large_ffn_start_segment: usize,
}

// ── Weight layout ──────────────────────────────────────────────────

/// Layout of the contiguous weight blob for mega-kernel execution.
///
/// Weights are packed in order: embed → layer_0 → layer_1 → ... → logits_producer.
/// Each layer's weights follow a fixed internal order.
/// Byte offsets are baked as immediates in JIT code.
#[derive(Debug, Clone)]
pub struct KernelWeightLayout {
    /// Byte offset of the embedding weight matrix.
    pub embed_offset: usize,
    /// Byte size of the embedding weight matrix.
    pub embed_bytes: usize,
    /// Byte offset of the first layer's weights.
    pub layer_0_offset: usize,
    /// Byte stride between consecutive layers (all layers have identical weight shapes).
    pub layer_stride: usize,
    /// Per-layer weight breakdown (offsets relative to layer base).
    pub per_layer: PerLayerWeightLayout,
    /// Byte offset of the final RMSNorm weight (between last layer and logits producer).
    pub final_norm_offset: usize,
    /// Byte size of the final RMSNorm weight.
    pub final_norm_bytes: usize,
    /// Byte offset of the logits producer weight matrix.
    pub logits_producer_offset: usize,
    /// Byte size of the logits producer weight matrix.
    pub logits_producer_bytes: usize,
    /// Total weight blob size in bytes.
    pub total_bytes: usize,
}

/// Per-layer weight offsets (relative to layer base pointer).
#[derive(Debug, Clone)]
pub struct PerLayerWeightLayout {
    pub attn_norm_offset: usize,
    pub attn_norm_bytes: usize,
    pub w_q_offset: usize,
    pub w_q_bytes: usize,
    pub w_k_offset: usize,
    pub w_k_bytes: usize,
    pub w_v_offset: usize,
    pub w_v_bytes: usize,
    pub w_o_offset: usize,
    pub w_o_bytes: usize,
    /// Qwen3 HeadRmsNorm weight for Q (shape=[head_dim]). Zero for non-Qwen3 models.
    pub w_q_norm_offset: usize,
    pub w_q_norm_bytes: usize,
    /// Qwen3 HeadRmsNorm weight for K (shape=[head_dim]). Zero for non-Qwen3 models.
    pub w_k_norm_offset: usize,
    pub w_k_norm_bytes: usize,
    pub ffn_norm_offset: usize,
    pub ffn_norm_bytes: usize,
    pub w_gate_offset: usize,
    pub w_gate_bytes: usize,
    pub w_up_offset: usize,
    pub w_up_bytes: usize,
    pub w_down_offset: usize,
    pub w_down_bytes: usize,
}

impl KernelWeightLayout {
    /// Compute weight layout from graph-derived geometry.
    pub fn from_graph_geometry(geo: &super::graph_geometry::GraphDerivedGeometry) -> Self {
        Self::build(
            geo.storage_dtype.size_bytes(),
            geo.hidden,
            geo.intermediate,
            geo.num_layers,
            geo.num_heads,
            geo.num_kv_heads,
            geo.head_dim,
            geo.vocab_size,
        )
    }

    fn build(
        elem_bytes: usize,
        hidden: usize,
        intermediate: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vocab_size: usize,
    ) -> Self {
        let embed_bytes = vocab_size * hidden * elem_bytes;
        let embed_offset = 0;

        let attn_norm_bytes = hidden * elem_bytes;
        let w_q_bytes = num_heads * head_dim * hidden * elem_bytes;
        let w_k_bytes = num_kv_heads * head_dim * hidden * elem_bytes;
        let w_v_bytes = num_kv_heads * head_dim * hidden * elem_bytes;
        let w_o_bytes = hidden * (num_heads * head_dim) * elem_bytes;
        let w_q_norm_bytes = head_dim * elem_bytes;
        let w_k_norm_bytes = head_dim * elem_bytes;
        let ffn_norm_bytes = hidden * elem_bytes;
        let w_gate_bytes = intermediate * hidden * elem_bytes;
        let w_up_bytes = intermediate * hidden * elem_bytes;
        let w_down_bytes = hidden * intermediate * elem_bytes;

        let layer_stride = attn_norm_bytes + w_q_bytes + w_k_bytes + w_v_bytes + w_o_bytes
            + w_q_norm_bytes + w_k_norm_bytes + ffn_norm_bytes
            + w_gate_bytes + w_up_bytes + w_down_bytes;
        let layer_0_offset = embed_offset + embed_bytes;

        let final_norm_bytes = hidden * elem_bytes;
        let final_norm_offset = layer_0_offset + num_layers * layer_stride;

        let logits_producer_offset = final_norm_offset + final_norm_bytes;
        let logits_producer_bytes = vocab_size * hidden * elem_bytes;
        let total_bytes = logits_producer_offset + logits_producer_bytes;

        let o0 = 0;
        let o1 = o0 + attn_norm_bytes;
        let o2 = o1 + w_q_bytes;
        let o3 = o2 + w_k_bytes;
        let o4 = o3 + w_v_bytes;
        let o5 = o4 + w_o_bytes;
        let o6 = o5 + w_q_norm_bytes;
        let o7 = o6 + w_k_norm_bytes;
        let o8 = o7 + ffn_norm_bytes;
        let o9 = o8 + w_gate_bytes;
        let o10 = o9 + w_up_bytes;

        let per_layer = PerLayerWeightLayout {
            attn_norm_offset: o0,
            attn_norm_bytes,
            w_q_offset: o1,
            w_q_bytes,
            w_k_offset: o2,
            w_k_bytes,
            w_v_offset: o3,
            w_v_bytes,
            w_o_offset: o4,
            w_o_bytes,
            w_q_norm_offset: o5,
            w_q_norm_bytes,
            w_k_norm_offset: o6,
            w_k_norm_bytes,
            ffn_norm_offset: o7,
            ffn_norm_bytes,
            w_gate_offset: o8,
            w_gate_bytes,
            w_up_offset: o9,
            w_up_bytes,
            w_down_offset: o10,
            w_down_bytes,
        };

        Self {
            embed_offset,
            embed_bytes,
            layer_0_offset,
            layer_stride,
            per_layer,
            final_norm_offset,
            final_norm_bytes,
            logits_producer_offset,
            logits_producer_bytes,
            total_bytes,
        }
    }

    /// Get the byte offset of a specific layer's weight base.
    pub fn layer_base_offset(&self, layer_idx: usize) -> usize {
        self.layer_0_offset + layer_idx * self.layer_stride
    }

}

/// Heterogeneous weight layout with per-segment-type information.
///
/// Supports 4 layer types (Gemma-4 E2B): sliding_small, full_small, sliding_large, full_large.
/// "small" vs "large" refers to FFN intermediate size, which differs by segment.
#[derive(Debug, Clone)]
pub struct HeteroKernelWeightLayout {
    pub embed_offset: usize,
    pub embed_bytes: usize,
    pub layer_0_offset: usize,
    /// Segment stride for small-FFN segments (segments 0..large_ffn_start_segment).
    pub small_segment_stride: usize,
    /// Segment stride for large-FFN segments (segments large_ffn_start_segment..num_segments).
    pub large_segment_stride: usize,
    /// Byte stride for sliding layers in small-FFN segments.
    pub sliding_small_stride: usize,
    /// Byte stride for full layers in small-FFN segments.
    pub full_small_stride: usize,
    /// Byte stride for sliding layers in large-FFN segments.
    pub sliding_large_stride: usize,
    /// Byte stride for full layers in large-FFN segments.
    pub full_large_stride: usize,
    /// Per-layer layout for sliding attention + small FFN layers.
    pub sliding_small_per_layer: PerLayerWeightLayout,
    /// Per-layer layout for full attention + small FFN layers.
    pub full_small_per_layer: PerLayerWeightLayout,
    /// Per-layer layout for sliding attention + large FFN layers.
    pub sliding_large_per_layer: PerLayerWeightLayout,
    /// Per-layer layout for full attention + large FFN layers.
    pub full_large_per_layer: PerLayerWeightLayout,
    /// First segment index using large FFN (segments 0..this use small).
    pub large_ffn_start_segment: usize,
    pub final_norm_offset: usize,
    pub final_norm_bytes: usize,
    pub logits_producer_offset: usize,
    pub logits_producer_bytes: usize,
    pub total_bytes: usize,
}

impl HeteroKernelWeightLayout {
    /// Build from graph-derived geometry instead of ModelMegaConfig.
    pub fn from_geometry_and_config(
        geo: &super::graph_geometry::GraphDerivedGeometry,
        hetero: &HeteroLayerConfig,
    ) -> Self {
        Self::build(geo.storage_dtype.size_bytes(), geo.hidden, geo.num_heads, geo.vocab_size, hetero)
    }

    fn build(elem_bytes: usize, h: usize, num_heads: usize, vocab_size: usize, hetero: &HeteroLayerConfig) -> Self {

        let embed_bytes = vocab_size * h * elem_bytes;
        let layer_0_offset = embed_bytes;

        // 4 per-type layouts: (attention_type × ffn_size)
        // Derived from geometry-sourced dimensions (REQ-D4-WL-TEMPLATE).
        // HeteroWeightLayout computes weight byte offsets from HeteroLayerConfig
        // parameters, which themselves originate from GraphDerivedGeometry.
        let compute_wl = |hidden: usize,
                          q_dim: usize,
                          kv_dim: usize,
                          o_in_dim: usize,
                          head_dim: usize,
                          intermediate: usize,
                          eb: usize|
         -> (PerLayerWeightLayout, usize) {
            let attn_norm_bytes = hidden * eb;
            let w_q_bytes = q_dim * hidden * eb;
            let w_k_bytes = kv_dim * hidden * eb;
            let w_v_bytes = kv_dim * hidden * eb;
            let w_o_bytes = hidden * o_in_dim * eb;
            let w_q_norm_bytes = head_dim * eb;
            let w_k_norm_bytes = head_dim * eb;
            let ffn_norm_bytes = hidden * eb;
            let w_gate_bytes = intermediate * hidden * eb;
            let w_up_bytes = intermediate * hidden * eb;
            let w_down_bytes = hidden * intermediate * eb;

            let total = attn_norm_bytes + w_q_bytes + w_k_bytes + w_v_bytes + w_o_bytes
                + w_q_norm_bytes + w_k_norm_bytes + ffn_norm_bytes
                + w_gate_bytes + w_up_bytes + w_down_bytes;

            let o0 = 0usize;
            let o1 = o0 + attn_norm_bytes;
            let o2 = o1 + w_q_bytes;
            let o3 = o2 + w_k_bytes;
            let o4 = o3 + w_v_bytes;
            let o5 = o4 + w_o_bytes;
            let o6 = o5 + w_q_norm_bytes;
            let o7 = o6 + w_k_norm_bytes;
            let o8 = o7 + ffn_norm_bytes;
            let o9 = o8 + w_gate_bytes;
            let o10 = o9 + w_up_bytes;

            (PerLayerWeightLayout {
                attn_norm_offset: o0, attn_norm_bytes,
                w_q_offset: o1, w_q_bytes,
                w_k_offset: o2, w_k_bytes,
                w_v_offset: o3, w_v_bytes,
                w_o_offset: o4, w_o_bytes,
                w_q_norm_offset: o5, w_q_norm_bytes,
                w_k_norm_offset: o6, w_k_norm_bytes,
                ffn_norm_offset: o7, ffn_norm_bytes,
                w_gate_offset: o8, w_gate_bytes,
                w_up_offset: o9, w_up_bytes,
                w_down_offset: o10, w_down_bytes,
            }, total)
        };

        let (sliding_small_pl, sliding_small_total) = compute_wl(
            h, hetero.sliding_num_q_heads * hetero.sliding_head_dim,
            hetero.sliding_num_kv_heads * hetero.sliding_head_dim,
            hetero.sliding_num_q_heads * hetero.sliding_head_dim,
            hetero.sliding_head_dim, hetero.small_intermediate, elem_bytes,
        );
        let (full_small_pl, full_small_total) = compute_wl(
            h, hetero.full_num_q_heads * hetero.full_head_dim,
            hetero.full_num_kv_heads * hetero.full_head_dim,
            hetero.full_num_q_heads * hetero.full_head_dim,
            hetero.full_head_dim, hetero.small_intermediate, elem_bytes,
        );
        let (sliding_large_pl, sliding_large_total) = compute_wl(
            h, hetero.sliding_num_q_heads * hetero.sliding_head_dim,
            hetero.sliding_num_kv_heads * hetero.sliding_head_dim,
            hetero.sliding_num_q_heads * hetero.sliding_head_dim,
            hetero.sliding_head_dim, hetero.large_intermediate, elem_bytes,
        );
        let (full_large_pl, full_large_total) = compute_wl(
            h, hetero.full_num_q_heads * hetero.full_head_dim,
            hetero.full_num_kv_heads * hetero.full_head_dim,
            hetero.full_num_q_heads * hetero.full_head_dim,
            hetero.full_head_dim, hetero.large_intermediate, elem_bytes,
        );

        let small_seg_stride = hetero.sliding_per_segment * sliding_small_total + full_small_total;
        let large_seg_stride = hetero.sliding_per_segment * sliding_large_total + full_large_total;

        let num_small_segs = hetero.large_ffn_start_segment;
        let num_large_segs = hetero.num_segments - num_small_segs;
        let layers_blob_bytes = num_small_segs * small_seg_stride + num_large_segs * large_seg_stride;

        let final_norm_bytes = h * elem_bytes;
        let final_norm_offset = layer_0_offset + layers_blob_bytes;
        let logits_producer_bytes = vocab_size * h * elem_bytes;
        let logits_producer_offset = final_norm_offset + final_norm_bytes;

        Self {
            embed_offset: 0,
            embed_bytes,
            layer_0_offset,
            small_segment_stride: small_seg_stride,
            large_segment_stride: large_seg_stride,
            sliding_small_stride: sliding_small_total,
            full_small_stride: full_small_total,
            sliding_large_stride: sliding_large_total,
            full_large_stride: full_large_total,
            sliding_small_per_layer: sliding_small_pl,
            full_small_per_layer: full_small_pl,
            sliding_large_per_layer: sliding_large_pl,
            full_large_per_layer: full_large_pl,
            large_ffn_start_segment: hetero.large_ffn_start_segment,
            final_norm_offset,
            final_norm_bytes,
            logits_producer_offset,
            logits_producer_bytes,
            total_bytes: logits_producer_offset + logits_producer_bytes,
        }
    }
}

// ── Buffer layout ──────────────────────────────────────────────────

/// Layout of runtime buffers for mega-kernel execution.
///
/// All offsets are relative to the scratchpad pointer (arg 7).
/// Activation ping/pong buffers alternate between layers.
#[derive(Debug, Clone)]
pub struct BufferLayout {
    /// Activation buffer A (ping): max_seq_len × hidden × elem_bytes
    pub activation_a_offset: usize,
    /// Activation buffer B (pong): same size
    pub activation_b_offset: usize,
    /// Size of each activation buffer in bytes.
    pub activation_bytes: usize,
    /// Logits buffer: vocab_size × elem_bytes (for sampling)
    pub logits_offset: usize,
    /// Logits buffer size in bytes.
    pub logits_bytes: usize,
    /// Sampling workspace offset (temperature scaling, top-k sort, etc.)
    pub sampling_workspace_offset: usize,
    /// Sampling workspace size in bytes.
    pub sampling_workspace_bytes: usize,
    /// SG detect buffer offset (hidden × elem_bytes, only when sg_enabled)
    pub sg_detect_offset: usize,
    /// SG knowledge vector offset (hidden × elem_bytes, only when sg_enabled)
    pub sg_knowledge_offset: usize,
    /// SG data total bytes (0 when sg disabled)
    pub sg_data_bytes: usize,
    /// Total scratchpad size required.
    pub total_scratchpad_bytes: usize,
}

impl BufferLayout {
    /// Compute buffer layout from graph-derived geometry + max_seq_len.
    ///
    /// Single-sequence mode: activation_dim == max_seq_len (M = 1 × seq_len).
    pub fn from_graph_geometry(
        geo: &super::graph_geometry::GraphDerivedGeometry,
        max_seq_len: usize,
    ) -> Self {
        Self::build(geo.storage_dtype.size_bytes(), max_seq_len, geo.hidden, geo.vocab_size)
    }

    /// Compute buffer layout for batched inference (SPEC/20 REQ-BCI-010).
    ///
    /// - `max_M`: upper bound on M dimension (total_prefill_tokens or num_active_seqs),
    ///   driving activation and logits buffer sizing.
    /// - `max_seq_len`: upper bound on single-sequence length, driving RoPE cache sizing.
    ///   Note: RoPE cache offset is currently managed by plan_lower's `compute_rope_requirement()`.
    ///   This parameter is reserved for future unification into this layout.
    #[allow(non_snake_case)] // SPEC/20 REQ-BCI-010: parameter name matches SPEC notation
    pub fn from_graph_geometry_batched(
        geo: &super::graph_geometry::GraphDerivedGeometry,
        max_M: usize,
        max_seq_len: usize,
    ) -> Self {
        let _ = max_seq_len; // Reserved: will drive RoPE cache sizing when unified from plan_lower
        Self::build(geo.storage_dtype.size_bytes(), max_M, geo.hidden, geo.vocab_size)
    }

    /// Build buffer layout from raw dimensions.
    ///
    /// `activation_dim` is the M dimension (max_seq_len for single-seq, max_M for batched).
    /// It drives activation ping/pong and logits buffer sizing.
    fn build(elem_bytes: usize, activation_dim: usize, hidden: usize, vocab_size: usize) -> Self {
        let activation_bytes = activation_dim * hidden * elem_bytes;
        let logits_bytes = activation_dim * vocab_size * elem_bytes;
        let sampling_workspace_bytes = vocab_size * elem_bytes * 4;
        let sg_hidden_bytes = hidden * elem_bytes;

        let mut off = 0;

        let activation_a_offset = off;
        off += activation_bytes;

        let activation_b_offset = off;
        off += activation_bytes;

        let logits_offset = off;
        off += logits_bytes;

        let sampling_workspace_offset = off;
        off += sampling_workspace_bytes;

        let sg_detect_offset = off;
        off += sg_hidden_bytes;
        let sg_knowledge_offset = off;
        off += sg_hidden_bytes;
        let sg_data_bytes = sg_hidden_bytes * 2;

        Self {
            activation_a_offset,
            activation_b_offset,
            activation_bytes,
            logits_offset,
            logits_bytes,
            sampling_workspace_offset,
            sampling_workspace_bytes,
            sg_detect_offset,
            sg_knowledge_offset,
            sg_data_bytes,
            total_scratchpad_bytes: off,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OutputMode & PoolMode ──────────────────────────────────────────

    #[test]
    fn output_mode_generate_fields() {
        let mode = OutputMode::Generate { max_new_tokens: 256, eos_token_id: 2 };
        match mode {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(max_new_tokens, 256);
                assert_eq!(eos_token_id, 2);
            }
            _ => panic!("expected Generate variant"),
        }
    }

    #[test]
    fn output_mode_classify_binary() {
        let mode = OutputMode::ClassifyBinary { positive_token_id: 1, negative_token_id: 0 };
        match mode {
            OutputMode::ClassifyBinary { positive_token_id, negative_token_id } => {
                assert_eq!(positive_token_id, 1);
                assert_eq!(negative_token_id, 0);
            }
            _ => panic!("expected ClassifyBinary"),
        }
    }

    #[test]
    fn output_mode_classify_multiway() {
        let ids = vec![10, 20, 30];
        let mode = OutputMode::ClassifyMultiway { label_token_ids: ids.clone() };
        match mode {
            OutputMode::ClassifyMultiway { label_token_ids } => {
                assert_eq!(label_token_ids, ids);
            }
            _ => panic!("expected ClassifyMultiway"),
        }
    }

    #[test]
    fn output_mode_encode_to_layer() {
        let mode = OutputMode::EncodeToLayer { anchor_layer: 12, pool_mode: PoolMode::MeanPool };
        match mode {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode } => {
                assert_eq!(anchor_layer, 12);
                assert!(matches!(pool_mode, PoolMode::MeanPool));
            }
            _ => panic!("expected EncodeToLayer"),
        }
    }

    #[test]
    fn pool_mode_variants() {
        let modes = [PoolMode::LastToken, PoolMode::MeanPool, PoolMode::ClsToken];
        assert_eq!(modes.len(), 3);
    }

    // ── BusinessConfig default ──────────────────────────────

    #[test]
    fn business_config_default_output_mode() {
        let cfg = BusinessConfig::default();
        assert_eq!(cfg.output_modes.len(), 1);
        match &cfg.output_modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 512);
                assert_eq!(*eos_token_id, 2);
            }
            _ => panic!("default should be Generate"),
        }
    }

    #[test]
    fn business_config_default_flags_off() {
        let cfg = BusinessConfig::default();
        assert!(!cfg.guardrail_enabled);
        assert!(cfg.semantic_gatekeeper.is_none());
        assert!(cfg.intent_anchor_layer.is_none());
        assert!(cfg.cot_step_hook.is_none());
        assert!(!cfg.session_enabled);
        assert!(!cfg.multimodal_enabled);
        assert!(!cfg.debug_jit);
    }

    // ── ABI constants ─────────────────────────────────────────────────

    #[test]
    fn mega_kernel_params_count() {
        assert_eq!(MEGA_KERNEL_PARAMS.len(), 22);
    }

    #[test]
    fn mega_kernel_params_first_six() {
        assert_eq!(MEGA_KERNEL_PARAMS[0], "input_ids_ptr");
        assert_eq!(MEGA_KERNEL_PARAMS[1], "weight_blob_ptr");
        assert_eq!(MEGA_KERNEL_PARAMS[2], "kv_cache_ptr");
        assert_eq!(MEGA_KERNEL_PARAMS[3], "positions_ptr");
        assert_eq!(MEGA_KERNEL_PARAMS[4], "aux_ptr");
        assert_eq!(MEGA_KERNEL_PARAMS[5], "batch_size");
    }

    #[test]
    fn mega_kernel_params_prompt_len_is_arg6() {
        assert_eq!(MEGA_KERNEL_PARAMS[6], "prompt_len");
    }

    #[test]
    fn mega_kernel_params_last() {
        assert_eq!(MEGA_KERNEL_PARAMS[21], "batch_ctx_ptr");
    }

    #[test]
    fn mega_kernel_stack_offsets_count() {
        assert_eq!(MEGA_KERNEL_STACK_OFFSETS.len(), 16);
    }

    #[test]
    fn mega_kernel_stack_offsets_start_at_16() {
        assert_eq!(MEGA_KERNEL_STACK_OFFSETS[0], 16);
    }

    #[test]
    fn mega_kernel_stack_offsets_8_byte_aligned() {
        for &off in MEGA_KERNEL_STACK_OFFSETS {
            assert_eq!(off % 8, 0, "offset {} not 8-byte aligned", off);
        }
    }

    #[test]
    fn mega_kernel_stack_offsets_end_at_136() {
        assert_eq!(*MEGA_KERNEL_STACK_OFFSETS.last().unwrap(), 136);
    }

    #[test]
    fn mega_kernel_stack_offsets_strictly_increasing() {
        for w in MEGA_KERNEL_STACK_OFFSETS.windows(2) {
            assert!(w[0] < w[1], "offsets not strictly increasing: {} >= {}", w[0], w[1]);
        }
    }

    // ── WeightLayout ────────────────────────────────────────

    #[test]
    fn weight_layout_build_symmetric_qkv() {
        let wl = KernelWeightLayout::build(
            2,    // elem_bytes (BF16)
            4096, // hidden
            11008, // intermediate
            32,   // num_layers
            32,   // num_heads
            32,   // num_kv_heads (GQA disabled)
            128,  // head_dim
            32000, // vocab_size
        );

        assert_eq!(wl.embed_offset, 0);
        let expected_embed = 32000 * 4096 * 2;
        assert_eq!(wl.embed_bytes, expected_embed);
        assert_eq!(wl.layer_0_offset, expected_embed);

        assert_eq!(wl.per_layer.w_q_bytes, 32 * 128 * 4096 * 2);
        assert_eq!(wl.per_layer.w_k_bytes, 32 * 128 * 4096 * 2);
        assert_eq!(wl.per_layer.w_v_bytes, 32 * 128 * 4096 * 2);
        assert_eq!(wl.per_layer.w_o_bytes, 4096 * (32 * 128) * 2);
    }

    #[test]
    fn weight_layout_gqa_smaller_kv() {
        let wl = KernelWeightLayout::build(
            2, 4096, 11008, 32, 32, 8, 128, 32000,
        );
        let q_bytes = wl.per_layer.w_q_bytes;
        let k_bytes = wl.per_layer.w_k_bytes;
        let v_bytes = wl.per_layer.w_v_bytes;

        assert!(k_bytes < q_bytes, "GQA K projection should be smaller than Q");
        assert_eq!(k_bytes, v_bytes, "K and V should be same size");
        assert_eq!(k_bytes, 8 * 128 * 4096 * 2);
    }

    #[test]
    fn weight_layout_layer_base_offset() {
        let wl = KernelWeightLayout::build(2, 4096, 11008, 2, 32, 32, 128, 32000);

        assert_eq!(wl.layer_base_offset(0), wl.layer_0_offset);
        assert_eq!(wl.layer_base_offset(1), wl.layer_0_offset + wl.layer_stride);
    }

    #[test]
    fn weight_layout_per_layer_offsets_contiguous() {
        let wl = KernelWeightLayout::build(2, 4096, 11008, 1, 32, 8, 128, 32000);
        let pl = &wl.per_layer;

        assert_eq!(pl.w_q_offset, pl.attn_norm_offset + pl.attn_norm_bytes);
        assert_eq!(pl.w_k_offset, pl.w_q_offset + pl.w_q_bytes);
        assert_eq!(pl.w_v_offset, pl.w_k_offset + pl.w_k_bytes);
        assert_eq!(pl.w_o_offset, pl.w_v_offset + pl.w_v_bytes);
        assert_eq!(pl.w_q_norm_offset, pl.w_o_offset + pl.w_o_bytes);
        assert_eq!(pl.w_k_norm_offset, pl.w_q_norm_offset + pl.w_q_norm_bytes);
        assert_eq!(pl.ffn_norm_offset, pl.w_k_norm_offset + pl.w_k_norm_bytes);
        assert_eq!(pl.w_gate_offset, pl.ffn_norm_offset + pl.ffn_norm_bytes);
        assert_eq!(pl.w_up_offset, pl.w_gate_offset + pl.w_gate_bytes);
        assert_eq!(pl.w_down_offset, pl.w_up_offset + pl.w_up_bytes);
    }

    #[test]
    fn weight_layout_total_bytes_consistent() {
        let wl = KernelWeightLayout::build(2, 4096, 11008, 4, 32, 8, 128, 32000);
        let expected = wl.logits_producer_offset + wl.logits_producer_bytes;
        assert_eq!(wl.total_bytes, expected);
        assert!(wl.total_bytes > 0);
    }

    // ── BufferLayout ────────────────────────────────────────

    #[test]
    fn buffer_layout_activation_ping_pong() {
        let bl = BufferLayout::build(2, 512, 4096, 32000);
        assert_eq!(bl.activation_a_offset, 0);
        assert_eq!(bl.activation_b_offset, bl.activation_bytes);
        assert_eq!(bl.activation_bytes, 512 * 4096 * 2);
    }

    #[test]
    fn buffer_layout_logits_after_activations() {
        let bl = BufferLayout::build(2, 512, 4096, 32000);
        assert_eq!(bl.logits_offset, bl.activation_b_offset + bl.activation_bytes);
        assert_eq!(bl.logits_bytes, 512 * 32000 * 2);
    }

    #[test]
    fn buffer_layout_sampling_after_logits() {
        let bl = BufferLayout::build(2, 512, 4096, 32000);
        assert_eq!(bl.sampling_workspace_offset, bl.logits_offset + bl.logits_bytes);
        assert_eq!(bl.sampling_workspace_bytes, 32000 * 2 * 4);
    }

    #[test]
    fn buffer_layout_sg_data() {
        let bl = BufferLayout::build(2, 128, 768, 50000);
        let sg_hidden = 768 * 2;
        assert_eq!(bl.sg_detect_offset, bl.sampling_workspace_offset + bl.sampling_workspace_bytes);
        assert_eq!(bl.sg_knowledge_offset, bl.sg_detect_offset + sg_hidden);
        assert_eq!(bl.sg_data_bytes, sg_hidden * 2);
    }

    #[test]
    fn buffer_layout_total_scratchpad() {
        let bl = BufferLayout::build(4, 256, 1024, 5000);
        assert_eq!(bl.total_scratchpad_bytes, bl.sg_knowledge_offset + 1024 * 4);
    }

    // ── HeteroLayerConfig ─────────────────────────────────────────────

    #[test]
    fn hetero_layer_config_gemma4_e2b() {
        let cfg = HeteroLayerConfig {
            num_segments: 7,
            sliding_per_segment: 4,
            sliding_head_dim: 256,
            sliding_num_q_heads: 4,
            sliding_num_kv_heads: 1,
            full_head_dim: 256,
            full_num_q_heads: 8,
            full_num_kv_heads: 1,
            full_layer_indices: vec![4, 9, 14, 19, 24, 29, 34],
            small_intermediate: 4096,
            large_intermediate: 8192,
            large_ffn_start_segment: 3,
        };
        assert_eq!(cfg.num_segments, 7);
        assert_eq!(cfg.full_layer_indices.len(), 7);
        assert_eq!(cfg.large_ffn_start_segment, 3);
        assert!(cfg.large_intermediate > cfg.small_intermediate);
    }

    // ── SgConfig ──────────────────────────────────────────────────────

    #[test]
    fn sg_config_fields() {
        let sg = SgConfig {
            detect_layer: 16,
            inject_offset: 0,
            detect_offset: 4096,
            q_tap: None,
        };
        assert_eq!(sg.detect_layer, 16);
        assert!(sg.q_tap.is_none());
    }

    // ── CotStepConfig ─────────────────────────────────────────────────

    #[test]
    fn cot_step_config() {
        let cfg = CotStepConfig { shared_mem_offset: 8192 };
        assert_eq!(cfg.shared_mem_offset, 8192);
    }

    // ── MtpKernelConfig ───────────────────────────────────────────────

    #[test]
    fn mtp_kernel_config() {
        let cfg = MtpKernelConfig { depth: 2, hidden_size: 4096, vocab_size: 32000 };
        assert_eq!(cfg.depth, 2);
        assert_eq!(cfg.hidden_size, 4096);
    }

    // ── CompileConfig ──────────────────────────────────────────────

    #[test]
    fn mega_kernel_config_fields() {
        let cfg = CompileConfig {
            max_seq_len: 4096,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        assert_eq!(cfg.max_seq_len, 4096);
        assert!(!cfg.debug_jit);
        assert!(cfg.hetero.is_none());
        assert_eq!(cfg.target, CompileTarget::Cpu);
    }
}

