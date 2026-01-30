use crate::kernel_types::{
    // L2 Block-level configs (ARCH-GRANULARITY-001)
    AttentionBlockConfig, FFNBlockConfig, EmbeddingConfig, LMHeadConfig,
    KVCacheUpdateConfig, MeanPoolingConfig, ClsPoolingConfig, NormalizeConfig,
    DequantizeConfig, EngramFuseConfig,
    // L3 High-level inference configs (ARCH-ADR-003)
    TransformerLayerWeights, MoETransformerLayerWeights, KVCacheState,
    GeneratorForwardConfig, MoEGeneratorForwardConfig, EmbeddingForwardConfig,
    RerankForwardConfig,
    // GPU-pure weights (ARCH-GPU-001: zero-copy forward)
    EmbeddingModelWeightsGpu,
    // GPU-native types (ARCH-ADR-010: complete zero-copy inference)
    RerankerModelWeightsGpu, GeneratorModelWeightsGpu, KVCacheGpu,
    TransformerLayerWeightsGpu, TransformerLayerConfigGpu,
    MoEGeneratorModelWeightsGpu,
    // Zero-copy logits (ARCH-PERF-001)
    LogitsTensor,
};
use crate::gpu_types::GpuTensor;
use crate::ops::sampling::SamplingConfig;
use crate::runtime_detection::BackendType;

pub enum TensorSlice<'a> {
    F32(&'a [f32]),
    F16(&'a [half::f16]),
    BF16(&'a [half::bf16]),
}

pub enum TensorSliceMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [half::f16]),
    BF16(&'a mut [half::bf16]),
}

/// Backend trait for L2 block-level operators (ARCH-GRANULARITY-001).
///
/// L0 atomic operators have been removed from the public API and are now
/// internal implementation details in the `ops::*` modules. This trait
/// only exposes high-level block operators that fuse multiple atomic
/// operations for better GPU utilization.
///
/// For low-level operations, use the crate-level function exports:
/// - `gllm_kernels::ops::attention::cpu_flash_attention`
/// - `gllm_kernels::ops::matmul::cpu_matmul`
/// - `gllm_kernels::ops::quantized::q4_matmul_cpu`
/// - `gllm_kernels::ops::sampling::argmax`
/// - `gllm_kernels::ops::moe_routing::moe_route`
/// - etc.
pub trait Backend: Send + Sync {
    // =========================================================================
    // L2 Block-Level Operators (ARCH-GRANULARITY-001)
    // =========================================================================
    //
    // These methods provide higher-level abstractions that fuse multiple
    // atomic operations for better GPU utilization. Each backend can override
    // with optimized implementations.

    /// Complete attention block (L2 block-level).
    ///
    /// Fuses: Input norm + QKV projection + RoPE + Attention + Output projection
    /// with optional Engram integration point.
    ///
    /// # Arguments
    /// * `hidden` - Input hidden states [batch, seq_len, hidden_size]
    /// * `q_weight` - Query projection weight [num_q_heads * head_dim, hidden_size]
    /// * `k_weight` - Key projection weight [num_kv_heads * head_dim, hidden_size]
    /// * `v_weight` - Value projection weight [num_kv_heads * head_dim, hidden_size]
    /// * `o_weight` - Output projection weight [hidden_size, num_q_heads * head_dim]
    /// * `norm_weight` - RMS norm weight [hidden_size]
    /// * `cos_cache` - RoPE cosine cache [max_seq_len, head_dim/2]
    /// * `sin_cache` - RoPE sine cache [max_seq_len, head_dim/2]
    /// * `kv_cache_k` - Optional KV cache for keys
    /// * `kv_cache_v` - Optional KV cache for values
    /// * `config` - Attention block configuration
    ///
    /// # Returns
    /// Output hidden states [batch, seq_len, hidden_size]
    fn attention_block(
        &self,
        hidden: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        v_weight: &[f32],
        o_weight: &[f32],
        norm_weight: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache_k: Option<&mut [f32]>,
        kv_cache_v: Option<&mut [f32]>,
        config: &AttentionBlockConfig,
    ) -> Result<Vec<f32>, String>;

    /// Complete FFN block (L2 block-level).
    ///
    /// Fuses: Input norm + Gate projection + Up projection + Activation + Down projection
    ///
    /// For LLaMA-style FFN: output = down(silu(gate(x)) * up(x))
    /// For GPT-style FFN: output = down(gelu(up(x)))
    ///
    /// # Arguments
    /// * `hidden` - Input hidden states [batch, seq_len, hidden_size]
    /// * `gate_weight` - Gate projection weight (for LLaMA-style)
    /// * `up_weight` - Up projection weight
    /// * `down_weight` - Down projection weight
    /// * `norm_weight` - RMS norm weight [hidden_size]
    /// * `config` - FFN block configuration
    ///
    /// # Returns
    /// Output hidden states [batch, seq_len, hidden_size]
    fn ffn_block(
        &self,
        hidden: &[f32],
        gate_weight: Option<&[f32]>,
        up_weight: &[f32],
        down_weight: &[f32],
        norm_weight: &[f32],
        config: &FFNBlockConfig,
    ) -> Result<Vec<f32>, String>;

    /// Embedding layer (L2 block-level).
    ///
    /// Handles: Token lookup + optional position encoding
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs [batch, seq_len]
    /// * `embed_weight` - Embedding weight [vocab_size, hidden_size]
    /// * `position_weight` - Optional position embedding weight
    /// * `config` - Embedding configuration
    ///
    /// # Returns
    /// Embedded representations [batch, seq_len, hidden_size]
    fn embedding(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        position_weight: Option<&[f32]>,
        config: &EmbeddingConfig,
    ) -> Result<Vec<f32>, String>;

    /// Language model head (L2 block-level).
    ///
    /// Projects hidden states to vocabulary logits.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states [batch, seq_len, hidden_size]
    /// * `lm_weight` - LM head weight [vocab_size, hidden_size]
    /// * `norm_weight` - Final norm weight [hidden_size]
    /// * `config` - LM head configuration
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    fn lm_head(
        &self,
        hidden: &[f32],
        lm_weight: &[f32],
        norm_weight: &[f32],
        config: &LMHeadConfig,
    ) -> Result<Vec<f32>, String>;

    /// Engram hash lookup (L2 block-level, CPU only).
    ///
    /// Performs O(1) hash-based lookup in DRAM embedding table.
    /// Always executes on CPU regardless of backend.
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs [batch, seq_len]
    /// * `engram_table` - Engram embedding table (memory-mapped)
    /// * `ngram_size` - Size of n-gram for hashing
    ///
    /// # Returns
    /// Tuple of (embeddings [batch, seq_len, hidden], bucket_indices)
    fn engram_lookup(
        &self,
        tokens: &[u32],
        engram_table: &[f32],
        hidden_size: usize,
        ngram_size: usize,
        num_buckets: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), String>;

    /// Engram-Attention fusion (L2 block-level).
    ///
    /// Merges standard attention output with Engram lookup results.
    ///
    /// # Arguments
    /// * `attention_output` - Standard attention output [batch, seq_len, hidden]
    /// * `engram_output` - Engram lookup output [batch, seq_len, hidden]
    /// * `config` - Fusion configuration
    ///
    /// # Returns
    /// Fused output [batch, seq_len, hidden_size]
    fn engram_fuse(
        &self,
        attention_output: &[f32],
        engram_output: &[f32],
        config: &EngramFuseConfig,
    ) -> Result<Vec<f32>, String>;

    /// KV cache update (L2 block-level).
    ///
    /// Updates key-value cache for autoregressive generation.
    ///
    /// # Arguments
    /// * `k_cache` - Key cache [batch, num_kv_heads, max_len, head_dim]
    /// * `v_cache` - Value cache [batch, num_kv_heads, max_len, head_dim]
    /// * `new_k` - New keys [batch, num_kv_heads, new_len, head_dim]
    /// * `new_v` - New values [batch, num_kv_heads, new_len, head_dim]
    /// * `config` - KV cache update configuration
    fn kv_cache_update(
        &self,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        new_k: &[f32],
        new_v: &[f32],
        config: &KVCacheUpdateConfig,
    ) -> Result<(), String>;

    /// Token sampling (L2 block-level, CPU).
    ///
    /// Performs temperature scaling, top-k, top-p filtering and sampling.
    /// Always executes on CPU due to random number generation requirements.
    ///
    /// # Arguments
    /// * `logits` - Input logits [batch, vocab_size]
    /// * `vocab_size` - Vocabulary size
    /// * `config` - Sampling configuration
    ///
    /// # Returns
    /// Sampled token IDs [batch]
    fn sample(
        &self,
        logits: &[f32],
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String>;

    /// Mean pooling (L2 block-level).
    ///
    /// Computes mean of hidden states over sequence dimension.
    /// Used for sentence embedding models.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states [batch, seq_len, hidden_size]
    /// * `attention_mask` - Optional mask [batch, seq_len]
    /// * `config` - Mean pooling configuration
    ///
    /// # Returns
    /// Pooled output [batch, hidden_size]
    fn mean_pooling(
        &self,
        hidden: &[f32],
        attention_mask: Option<&[f32]>,
        config: &MeanPoolingConfig,
    ) -> Result<Vec<f32>, String>;

    /// CLS token pooling (L2 block-level).
    ///
    /// Extracts the CLS token representation.
    /// Used for reranker and classification models.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states [batch, seq_len, hidden_size]
    /// * `config` - CLS pooling configuration
    ///
    /// # Returns
    /// CLS representation [batch, hidden_size]
    fn cls_pooling(
        &self,
        hidden: &[f32],
        config: &ClsPoolingConfig,
    ) -> Result<Vec<f32>, String>;

    /// L2 normalization (L2 block-level).
    ///
    /// Normalizes vectors to unit length.
    /// Used for embedding similarity computation.
    ///
    /// # Arguments
    /// * `input` - Input vectors [batch, dim]
    /// * `config` - Normalize configuration
    ///
    /// # Returns
    /// Normalized vectors [batch, dim]
    fn normalize(
        &self,
        input: &[f32],
        config: &NormalizeConfig,
    ) -> Result<Vec<f32>, String>;

    /// Generic dequantization (L2 block-level).
    ///
    /// Converts quantized weights to floating point.
    /// Supports Q4_0, Q4_K, Q8_0, and AWQ formats.
    ///
    /// # Arguments
    /// * `quantized` - Quantized weight data
    /// * `scales` - Quantization scales
    /// * `zeros` - Optional zero points (for AWQ)
    /// * `config` - Dequantization configuration
    ///
    /// # Returns
    /// Dequantized weights [n, k]
    fn dequantize(
        &self,
        quantized: &[u8],
        scales: &[half::f16],
        zeros: Option<&[u32]>,
        config: &DequantizeConfig,
    ) -> Result<Vec<f32>, String>;

    fn backend_type(&self) -> BackendType;

    // =========================================================================
    // Memory Tiering Primitives (ARCH-MEM-TIERING)
    // =========================================================================

    /// Swap a KV cache block from GPU (hot tier) to CPU memory (cold tier).
    ///
    /// # Arguments
    /// * `block_id` - Logical block identifier
    /// * `cpu_buffer` - Destination buffer for the swapped-out bytes
    fn swap_out(&self, block_id: u64, cpu_buffer: &mut [u8]) -> Result<(), String>;

    /// Swap a KV cache block from CPU memory (cold tier) back to GPU (hot tier).
    ///
    /// # Arguments
    /// * `cpu_buffer` - Source buffer containing the block bytes
    /// * `block_id` - Logical block identifier
    fn swap_in(&self, cpu_buffer: &[u8], block_id: u64) -> Result<(), String>;

    // =========================================================================
    // L3 High-Level Inference API (ARCH-ADR-003)
    // =========================================================================
    //
    // These methods provide complete model forward passes by composing L2 block
    // operators. gllm should call these instead of managing individual layers.
    // Default implementations use L2 methods; backends can override with fused kernels.

    /// Complete dense generator forward pass (L3 high-level).
    ///
    /// Performs: embedding → [attention_block → ffn_block] × n_layers → lm_head
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs [seq_len]
    /// * `embed_weight` - Embedding weights [vocab_size, hidden_size]
    /// * `layers` - Transformer layer weights for all layers
    /// * `final_norm` - Final layer norm weights [hidden_size]
    /// * `lm_head_weight` - LM head weights [vocab_size, hidden_size]
    /// * `cos_cache` - RoPE cosine cache [max_seq_len, head_dim/2]
    /// * `sin_cache` - RoPE sine cache [max_seq_len, head_dim/2]
    /// * `kv_cache` - KV cache state (mutable)
    /// * `config` - Generator forward configuration
    ///
    /// # Returns
    /// Logits for the last token [vocab_size]
    fn generator_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        lm_head_weight: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache: &mut KVCacheState<'_>,
        config: &GeneratorForwardConfig,
    ) -> Result<LogitsTensor, String>;

    /// Complete MoE generator forward pass (L3 high-level).
    ///
    /// Performs: embedding → [attention_block → moe_ffn_block] × n_layers → lm_head
    /// where moe_ffn_block handles expert routing and sparse computation.
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs [seq_len]
    /// * `embed_weight` - Embedding weights [vocab_size, hidden_size]
    /// * `layers` - MoE transformer layer weights for all layers
    /// * `final_norm` - Final layer norm weights [hidden_size]
    /// * `lm_head_weight` - LM head weights [vocab_size, hidden_size]
    /// * `cos_cache` - RoPE cosine cache [max_seq_len, head_dim/2]
    /// * `sin_cache` - RoPE sine cache [max_seq_len, head_dim/2]
    /// * `kv_cache` - KV cache state (mutable)
    /// * `config` - MoE generator forward configuration
    ///
    /// # Returns
    /// Logits for the last token [vocab_size]
    fn moe_generator_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[MoETransformerLayerWeights<'_>],
        final_norm: &[f32],
        lm_head_weight: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache: &mut KVCacheState<'_>,
        config: &MoEGeneratorForwardConfig,
    ) -> Result<LogitsTensor, String>;

    /// Complete embedding model forward pass (L3 high-level).
    ///
    /// Performs: embedding → [attention_block → ffn_block] × n_layers → pooling → normalize
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs [seq_len]
    /// * `embed_weight` - Embedding weights [vocab_size, hidden_size]
    /// * `layers` - Transformer layer weights for all layers
    /// * `final_norm` - Final layer norm weights [hidden_size] (optional)
    /// * `config` - Embedding forward configuration
    ///
    /// # Returns
    /// Normalized embedding vector [hidden_size]
    fn embedding_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: Option<&[f32]>,
        config: &EmbeddingForwardConfig,
    ) -> Result<Vec<f32>, String>;

    /// Upload CPU weights to GPU, returning GPU-resident weights (ARCH-GPU-001).
    ///
    /// This should be called once at model load time. The returned weights
    /// can be reused for all forward passes without re-uploading.
    fn upload_embedding_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: Option<&[f32]>,
        config: &EmbeddingForwardConfig,
    ) -> Result<EmbeddingModelWeightsGpu, String>;

    /// GPU-pure embedding forward pass (ARCH-GPU-001: zero-copy).
    ///
    /// Uses pre-uploaded GPU weights, avoiding per-forward-pass data transfer.
    /// Only tokens are uploaded (small), and only final output is downloaded.
    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &EmbeddingModelWeightsGpu,
        config: &EmbeddingForwardConfig,
    ) -> Result<Vec<f32>, String>;

    /// Complete reranker model forward pass (L3 high-level).
    ///
    /// Performs: embedding → [attention_block → ffn_block] × n_layers → cls_pooling → score_head
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs [seq_len] (query + document concatenated)
    /// * `embed_weight` - Embedding weights [vocab_size, hidden_size]
    /// * `layers` - Transformer layer weights for all layers
    /// * `final_norm` - Final layer norm weights [hidden_size]
    /// * `score_weight` - Score head weights [1, hidden_size]
    /// * `config` - Rerank forward configuration
    ///
    /// # Returns
    /// Relevance score [1]
    fn rerank_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        score_weight: &[f32],
        config: &RerankForwardConfig,
    ) -> Result<Vec<f32>, String>;

    // =========================================================================
    // GPU-Native Kernels (ARCH-ADR-010: Complete Zero-Copy Inference)
    // All 4 backends (CUDA/ROCm/Metal/CPU) must implement these methods.
    // =========================================================================

    /// GPU-native embedding lookup (tokens → hidden states).
    ///
    /// Performs gather operation on GPU: output[b][s][:] = embed_weight[tokens[b][s]][:]
    ///
    /// # Arguments
    /// * `tokens` - Token IDs, U32 tensor [batch, seq_len]
    /// * `embed_weight` - Embedding weights, F32 tensor [vocab_size, hidden_dim]
    /// * `output` - Output hidden states, F32 tensor [batch, seq_len, hidden_dim]
    fn embedding_lookup_gpu(
        &self,
        tokens: &GpuTensor,
        embed_weight: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String>;

    /// GPU-native transformer layer forward.
    ///
    /// Performs: input_norm → attention → residual → post_attn_norm → ffn → residual
    /// All computation stays on GPU, no intermediate readback.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states (in-place update), F32 tensor [batch, seq_len, hidden_dim]
    /// * `layer_weights` - GPU-resident layer weights
    /// * `kv_cache` - Optional KV cache for generation (GPU-resident)
    /// * `config` - Layer configuration
    fn transformer_layer_gpu(
        &self,
        hidden: &mut GpuTensor,
        layer_weights: &TransformerLayerWeightsGpu,
        kv_cache: Option<&mut KVCacheGpu>,
        config: &TransformerLayerConfigGpu,
    ) -> Result<(), String>;

    /// GPU-native RMS normalization (in-place).
    ///
    /// # Arguments
    /// * `hidden` - Hidden states (in-place update), F32 tensor [batch, seq_len, hidden_dim]
    /// * `weight` - Norm weights, F32 tensor [hidden_dim]
    /// * `eps` - Epsilon for numerical stability
    fn rms_norm_gpu(
        &self,
        hidden: &mut GpuTensor,
        weight: &GpuTensor,
        eps: f32,
    ) -> Result<(), String>;

    /// GPU-native mean pooling.
    ///
    /// Averages hidden states across sequence dimension with attention mask.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states, F32 tensor [batch, seq_len, hidden_dim]
    /// * `attention_mask` - Mask, F32 tensor [batch, seq_len]
    /// * `output` - Pooled output, F32 tensor [batch, hidden_dim]
    fn mean_pooling_gpu(
        &self,
        hidden: &GpuTensor,
        attention_mask: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String>;

    /// GPU-native L2 normalization (in-place).
    ///
    /// Normalizes embeddings to unit length.
    ///
    /// # Arguments
    /// * `embeddings` - Embeddings (in-place update), F32 tensor [batch, hidden_dim]
    fn normalize_gpu(
        &self,
        embeddings: &mut GpuTensor,
    ) -> Result<(), String>;

    /// GPU-native CLS pooling (extract first token).
    ///
    /// # Arguments
    /// * `hidden` - Hidden states, F32 tensor [batch, seq_len, hidden_dim]
    /// * `output` - CLS output, F32 tensor [batch, hidden_dim]
    fn cls_pooling_gpu(
        &self,
        hidden: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String>;

    /// GPU-native classifier (linear layer for scoring).
    ///
    /// # Arguments
    /// * `hidden` - Hidden states, F32 tensor [batch, hidden_dim]
    /// * `weight` - Classifier weights, F32 tensor [num_classes, hidden_dim]
    /// * `bias` - Optional bias, F32 tensor [num_classes]
    /// * `output` - Scores, F32 tensor [batch, num_classes]
    fn classifier_gpu(
        &self,
        hidden: &GpuTensor,
        weight: &GpuTensor,
        bias: Option<&GpuTensor>,
        output: &mut GpuTensor,
    ) -> Result<(), String>;

    /// GPU-native LM head (project to vocabulary logits).
    ///
    /// # Arguments
    /// * `hidden` - Hidden states, F32 tensor [batch, hidden_dim]
    /// * `weight` - LM head weights, F32 tensor [vocab_size, hidden_dim]
    /// * `output` - Logits, F32 tensor [batch, vocab_size]
    fn lm_head_gpu(
        &self,
        hidden: &GpuTensor,
        weight: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String>;

    // =========================================================================
    // GPU-Native High-Level Forward Methods (ARCH-ADR-010)
    // =========================================================================

    /// Upload reranker model weights to GPU (ARCH-ADR-010).
    fn upload_reranker_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        classifier_weight: &[f32],
        classifier_bias: Option<&[f32]>,
        config: &RerankForwardConfig,
    ) -> Result<RerankerModelWeightsGpu, String>;

    /// GPU-pure reranker forward pass (ARCH-ADR-010: zero-copy).
    ///
    /// Uses pre-uploaded GPU weights. Only tokens uploaded, only scores downloaded.
    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &RerankerModelWeightsGpu,
        config: &RerankForwardConfig,
    ) -> Result<Vec<f32>, String>;

    /// Upload generator/LLM model weights to GPU (ARCH-ADR-010).
    fn upload_generator_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        lm_head: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        config: &GeneratorForwardConfig,
    ) -> Result<GeneratorModelWeightsGpu, String>;

    /// Allocate KV cache on GPU (ARCH-ADR-010: GPU-resident KV cache).
    fn alloc_kv_cache_gpu(
        &self,
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        max_len: usize,
        head_dim: usize,
    ) -> Result<KVCacheGpu, String> {
        // Default implementation: fall back to CPU
        Err("KV cache GPU allocation not implemented for this backend".to_string())
    }

    /// Upload MoE generator/LLM model weights to GPU (ARCH-ADR-010).
    fn upload_moe_generator_weights(
        &self,
        embed_weight: &[f32],
        layers: &[MoETransformerLayerWeights<'_>],
        final_norm: &[f32],
        lm_head: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        config: &MoEGeneratorForwardConfig,
    ) -> Result<MoEGeneratorModelWeightsGpu, String> {
        // Default implementation: fall back to CPU
        Err("MoE weight upload not implemented for this backend".to_string())
    }

    /// GPU-pure MoE generator forward pass (ARCH-ADR-010: zero-copy).
    fn moe_generator_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &MoEGeneratorModelWeightsGpu,
        kv_cache: &mut KVCacheGpu,
        config: &MoEGeneratorForwardConfig,
    ) -> Result<LogitsTensor, String> {
        // Default implementation: fall back to CPU
        Err("MoE GPU forward not implemented for this backend".to_string())
    }

    /// GPU-pure generator forward pass (ARCH-ADR-010: zero-copy).
    ///
    /// Uses pre-uploaded GPU weights and GPU-resident KV cache.
    /// Only tokens uploaded, only logits downloaded.
    fn generator_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &GeneratorModelWeightsGpu,
        kv_cache: &mut KVCacheGpu,
        config: &GeneratorForwardConfig,
    ) -> Result<LogitsTensor, String>;

    // =========================================================================
    // Zero-Copy Sampling API (ARCH-PERF-001)
    // =========================================================================

    /// GPU-pure generator forward that keeps logits on GPU (ARCH-PERF-001).
    ///
    /// Returns LogitsTensor which may remain on GPU, avoiding the expensive
    /// GPU→CPU transfer of full vocabulary logits (128KB+ per token).
    ///
    /// Use with `sample_from_tensor` for complete zero-copy generation.
    fn generator_forward_gpu_pure_logits(
        &self,
        tokens: &[u32],
        weights: &GeneratorModelWeightsGpu,
        kv_cache: &mut KVCacheGpu,
        config: &GeneratorForwardConfig,
    ) -> Result<LogitsTensor, String> {
        // Default implementation: redirect to generator_forward_gpu_pure
        self.generator_forward_gpu_pure(tokens, weights, kv_cache, config)
    }

    /// Sample from LogitsTensor (CPU or GPU) (ARCH-PERF-001).
    ///
    /// For GPU tensors, this uses GPU-accelerated sampling:
    /// - Greedy (temp=0): GPU argmax kernel → 1 u32 transfer
    /// - Sampling: GPU top-k (k=64) → 64 candidates transfer → CPU sampling
    ///
    /// This reduces per-token transfer by >99.8% (32k floats → 64 floats).
    fn sample_from_tensor(
        &self,
        logits: &LogitsTensor,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        // Default implementation: extract CPU data and use regular sample
        match logits {
            LogitsTensor::Cpu(data) => self.sample(data, vocab_size, config),
            LogitsTensor::Gpu(_) => {
                Err("GPU sampling not implemented for this backend".to_string())
            }
        }
    }
}
