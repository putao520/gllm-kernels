//! Backend-centric zero-cost dispatch with method-level generics.
//!
//! Design principles:
//! - Method-level generics: `fn flash_attention<T: KernelFloat>(...)` for mixed precision
//! - Manual enum dispatch: `#[inline(always)]` + match eliminates vtable overhead
//! - Const branch elimination: `T::TYPE_ID` is const, match branches eliminated at compile time
//!
//! Mixed precision example:
//! ```ignore
//! backend.linear::<f16>(input, weight, &mut hidden);      // weights in f16
//! backend.flash_attention::<f32>(&q, &k, &v, &mut out);   // attention in f32
//! backend.softmax::<f16>(&logits, &mut probs);            // output in f16
//! ```

use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;

use cudarc::driver::{CudaContext, CudaStream, CudaSlice, DeviceRepr};

use crate::gpu_types::{GpuBuffer, GpuKVCache, GpuTensor, TensorDtype};
use crate::runtime_detection::BackendType;
use crate::wgpu_kernels::LinearParams;
use crate::{
    FlashAttentionConfig, PagedAttentionConfig, SoftmaxConfig, MatmulConfig,
    Eagle3Config, SpecEEConfig, FlashTreeAttentionConfig, Int2QuantResult,
    EvicPressCompressionResult, EvicPressEvictResult, MedusaForwardResult,
    MedusaVerifyResult, PromptCacheLookupResult, ChunkedPrefillResult,
    MedusaConfig, MedusaVerifyConfig, PromptCacheLookupConfig,
    PromptCacheBlendConfig, ChunkedPrefillConfig, GpuRerankConfig,
    GpuRerankStageResult, Int2QuantConfig, EvicPressCompressConfig,
    EvicPressEvictConfig,
};

use crate::types::*;
use crate::ops;
use crate::cuda_kernels::{
    FlashAttentionKernel as CudaFlashAttentionKernel,
    PagedAttentionKernel as CudaPagedAttentionKernel,
    Eagle3Kernel as CudaEagle3Kernel,
    SpecEEKernel as CudaSpecEEKernel,
    FlashTreeAttnKernel as CudaFlashTreeAttnKernel,
    Int2QuantizerKernel as CudaInt2QuantizerKernel,
    EvicPressKernel as CudaEvicPressKernel,
    MedusaKernel as CudaMedusaKernel,
    PromptCacheKernel as CudaPromptCacheKernel,
    ChunkedPrefillKernel as CudaChunkedPrefillKernel,
    CudaLinear,
    CudaEmbeddingOpsKernel,
    OnlineSoftmaxKernel as CudaOnlineSoftmaxKernel,
    RmsNormKernel as CudaRmsNormKernel,
};
use crate::hip_kernels::{
    HsaFlashAttentionKernel, HsaPagedAttentionKernel, HsaEagle3Kernel,
    HsaSpecEEKernel, HsaFlashTreeAttnKernel, HsaInt2QuantizerKernel,
    HsaEvicPressKernel, HsaMedusaKernel, HsaPromptCacheKernel,
    HsaChunkedPrefillKernel, HsaLinear, HsaEmbeddingOpsKernel,
    HsaRmsNormKernel,
    HsaEagle3Config, HsaSpecEEConfig, HsaFlashTreeAttnConfig,
    HsaInt2QuantizerConfig, HsaEvicPressConfig, HsaMedusaConfig,
    HsaPromptCacheConfig, HsaChunkedPrefillConfig,
    HsaQueueWrapper, HsaBuffer, GpuAgent, find_gpu_agents,
};

#[cfg(target_os = "macos")]
use crate::metal_kernels::{
    FlashAttentionKernel as MetalFlashAttentionKernel,
    PagedAttentionKernel as MetalPagedAttentionKernel,
    Eagle3Kernel as MetalEagle3Kernel,
    SpecEEKernel as MetalSpecEEKernel,
    FlashTreeAttnKernel as MetalFlashTreeAttnKernel,
    Int2QuantizerKernel as MetalInt2QuantizerKernel,
    EvicPressKernel as MetalEvicPressKernel,
    MedusaKernel as MetalMedusaKernel,
    PromptCacheKernel as MetalPromptCacheKernel,
    ChunkedPrefillKernel as MetalChunkedPrefillKernel,
    MetalLinear,
    MetalRmsNorm,
    EmbeddingOpsKernel as MetalEmbeddingOpsKernel,
    get_metal_device,
};
use crate::wgpu_kernels::{
    FlashAttentionKernel as WgpuFlashAttentionKernel,
    PagedAttentionKernel as WgpuPagedAttentionKernel,
    EmbeddingOpsKernel as WgpuEmbeddingOpsKernel,
    Eagle3Kernel as WgpuEagle3Kernel,
    SpecEEKernel as WgpuSpecEEKernel,
    FlashTreeAttn as WgpuFlashTreeAttn,
    WgpuInt2Quantizer,
    WgpuEvicPress,
    WgpuMedusa,
    WgpuPromptCache,
    WgpuChunkedPrefill,
    WgpuLinear,
    WgpuRmsNorm,
    WgpuTensorOps,
    WgpuMoeFfn,
    MoEFfnParams,
    WgpuMoERouting,
    MoERoutingGpuParams,
    RmsNormParams,
    TreeAttnParams,
};

// =============================================================================
// Unified Backend Trait with Method-Level Generics
// =============================================================================
//
// Design: Method-level generics for mixed precision inference support.
//
// Why NOT object-level generics (`Backend<T>`)?
// - Same backend must handle different precisions in one inference pass
// - Example: weights in f16, attention in f32, output in f16
//
// Why NOT enum_dispatch?
// - enum_dispatch doesn't support generic trait methods
// - Manual match + #[inline(always)] achieves same zero-cost dispatch
//
// Zero-cost mechanism:
// 1. DispatchedBackend::flash_attention<T>() -> #[inline(always)] match
// 2. Each backend's flash_attention<T>() uses T::TYPE_ID const branch
// 3. Compiler eliminates all unused branches at monomorphization
// =============================================================================

/// Backend trait with method-level generics for mixed precision support.
///
/// All compute-intensive methods are generic over `T: KernelFloat`, allowing:
/// - Same backend instance to handle f32, f16, bf16 in one inference
/// - Zero-cost dispatch via T::TYPE_ID const branch elimination
/// - Full mixed precision: weights, activations, outputs can differ
pub trait Backend: Send + Sync {
    // =========================================================================
    // Core Attention Operations (Generic)
    // =========================================================================

    /// Flash attention with O(1) memory scaling.
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig);

    /// Paged attention for KV cache with virtual memory.
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig);

    /// Softmax with numerical stability.
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig);

    /// Matrix multiplication C = A @ B.
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig);

    // =========================================================================
    // GPU Tensor Operations (Type-erased, dtype in GpuTensor)
    // =========================================================================

    /// Linear forward on GPU tensors.
    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String>;

    /// Linear forward on GPU tensors with accumulation to output (fused add).
    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String>;

    /// Linear forward with host input/output (upload, compute, readback).
    fn linear_forward_host_io<T: KernelFloat>(&self, input: &[T], weight: &GpuTensor, output: &mut [T], params: LinearParams) -> Result<(), String>;

    /// Readback GPU tensor to host slice (convenience wrapper).
    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String>;

    /// FFN (gate-up-down) forward on GPU tensors.
    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String>;

    /// Attention forward on GPU tensors with KV cache.
    fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String>;

    /// RMS normalization on GPU tensors.
    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String>;

    /// RMS normalization inplace on GPU tensor (input == output).
    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String>;

    /// Allocate raw GPU buffer.
    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String>;

    /// Allocate and upload weights to GPU.
    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String>;

    /// Read GPU tensor back to host (generic).
    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String>;

    /// Read GPU u32 tensor back to host (type-safe).
    fn readback_u32(&self, gpu: &GpuTensor, host: &mut [u32]) -> Result<(), String>;

    /// Upload host data to GPU tensor (generic).
    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String>;

    /// Update KV cache on GPU.
    fn update_kv_cache_gpu(&self, cache: &mut GpuKVCache, layer_idx: usize, new_k: &GpuTensor, new_v: &GpuTensor) -> Result<(), String>;

    // =========================================================================
    // MoE GPU Operations
    // =========================================================================

    /// Zero GPU tensor (set all elements to 0).
    fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String>;

    /// Add two GPU tensors: output += input.
    fn tensor_add_gpu(&self, output: &mut GpuTensor, input: &GpuTensor) -> Result<(), String>;

    /// Extract slice from GPU tensor: output = input[offset..offset+len].
    fn tensor_slice_gpu(&self, input: &GpuTensor, offset: usize, len: usize, output: &mut GpuTensor) -> Result<(), String>;

    /// Scale input and add to output at offset: output[offset..] += input * scale.
    fn tensor_scale_add_gpu(&self, input: &GpuTensor, output: &mut GpuTensor, offset: usize, scale: f32) -> Result<(), String>;

    /// MoE routing on GPU: compute top-k experts and weights without host readback.
    fn moe_route_gpu(
        &self,
        hidden_states: &GpuTensor,
        gate_weights: &GpuTensor,
        expert_indices_out: &mut GpuTensor,
        expert_weights_out: &mut GpuTensor,
        config: MoERoutingGpuConfig,
    ) -> Result<(), String>;

    /// Fused MoE forward with routing tensors already on GPU.
    ///
    /// This is the pure GPU path required by ARCH-MOE-001/002.
    fn moe_forward_gpu_pure(
        &self,
        input: &GpuTensor,
        expert_indices: &GpuTensor,
        expert_weights: &GpuTensor,
        all_gate_weights: &GpuTensor,
        all_up_weights: &GpuTensor,
        all_down_weights: &GpuTensor,
        output: &mut GpuTensor,
        config: MoEForwardConfig,
    ) -> Result<(), String>;

    // =========================================================================
    // Speculative Decoding Operations (Generic where applicable)
    // =========================================================================

    /// EAGLE-3 confidence prediction.
    fn eagle3_confidence<T: KernelFloat>(&self, layer_hidden_states: &[&[T]], confidence_weights: &[T], confidence_bias: T, config: &Eagle3Config) -> Option<Vec<T>>;

    /// Spec-EE early exit confidence.
    fn spec_ee_confidence<T: KernelFloat>(&self, hidden_states: &[T], classifier_weight: &[T], classifier_bias: T, config: &SpecEEConfig) -> Option<Vec<T>>;

    /// Flash tree attention for speculative decoding.
    fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool;

    /// Medusa forward pass.
    fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult>;

    /// Medusa verification.
    fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult>;

    // =========================================================================
    // Quantization Operations
    // =========================================================================

    /// Int2 quantization.
    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult>;

    /// Int2 dequantization.
    fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>>;

    // =========================================================================
    // KV Cache Compression (EvicPress)
    // =========================================================================

    /// EvicPress KV cache compression.
    fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult>;

    /// EvicPress eviction decision.
    fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult>;

    // =========================================================================
    // Prompt Caching
    // =========================================================================

    /// Prompt cache lookup by hash.
    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult>;

    /// Blend cached and fresh KV.
    fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>>;

    // =========================================================================
    // Chunked Prefill
    // =========================================================================

    /// Chunked prefill attention for long contexts.
    fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>>;

    // =========================================================================
    // Embedding Operations (Fixed precision for now)
    // =========================================================================

    /// Three-stage rerank pipeline.
    fn rerank_pipeline(&self, binary_query: &[u32], binary_database: &[u32], int8_query: &[u32], int8_database: &[u32], num_vectors: usize, config: &GpuRerankConfig, int8_scale: f32) -> Result<GpuRerankStageResult, String>;

    /// Binary inner product with Hamming distance.
    fn binary_ip_hamming(&self, queries: &[u64], database: &[u64], scores: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig);

    /// Asymmetric binary inner product.
    fn binary_ip_asymmetric(&self, queries: &[f32], database: &[u64], scores: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig);

    /// Int8 dot product.
    fn int8_dot_product(&self, queries: &[i8], database: &[i8], scores: &mut [f32], config: &crate::ops::embedding::Int8DotConfig);

    /// Int4 packed dot product.
    fn int4_packed_dot_product(&self, queries: &[u8], database: &[u8], scores: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig);

    // =========================================================================
    // Backend Info
    // =========================================================================

    /// Get backend type.
    fn backend_type(&self) -> BackendType;
}

// =============================================================================
// DispatchedBackend with Manual Zero-Cost Dispatch
// =============================================================================
//
// Manual dispatch instead of enum_dispatch because:
// 1. enum_dispatch doesn't support generic trait methods
// 2. #[inline(always)] + match achieves same zero-cost dispatch
// 3. Compiler inlines the match and eliminates unused branches
// =============================================================================

#[derive(Clone)]
pub enum DispatchedBackend {
    Cpu(CpuBackend),
    Wgpu(WgpuBackend),
    Cuda(CudaBackend),
    Rocm(RocmBackend),
    #[cfg(target_os = "macos")]
    Metal(MetalBackend),
}

impl DispatchedBackend {
    pub fn new(backend_type: BackendType) -> Self {
        match backend_type {
            BackendType::Wgpu => DispatchedBackend::Wgpu(WgpuBackend),
            BackendType::Cuda => DispatchedBackend::Cuda(CudaBackend),
            BackendType::Rocm => DispatchedBackend::Rocm(RocmBackend),
            #[cfg(target_os = "macos")]
            BackendType::Metal => DispatchedBackend::Metal(MetalBackend),
            _ => DispatchedBackend::Cpu(CpuBackend),
        }
    }

    // =========================================================================
    // Manual Dispatch Methods with #[inline(always)]
    // =========================================================================

    #[inline(always)]
    pub fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        match self {
            Self::Cpu(b) => b.flash_attention(q, k, v, output, config),
            Self::Wgpu(b) => b.flash_attention(q, k, v, output, config),
            Self::Cuda(b) => b.flash_attention(q, k, v, output, config),
            Self::Rocm(b) => b.flash_attention(q, k, v, output, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.flash_attention(q, k, v, output, config),
        }
    }

    #[inline(always)]
    pub fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        match self {
            Self::Cpu(b) => b.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
            Self::Wgpu(b) => b.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
            Self::Cuda(b) => b.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
            Self::Rocm(b) => b.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
        }
    }

    #[inline(always)]
    pub fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        match self {
            Self::Cpu(b) => b.softmax(input, output, config),
            Self::Wgpu(b) => b.softmax(input, output, config),
            Self::Cuda(b) => b.softmax(input, output, config),
            Self::Rocm(b) => b.softmax(input, output, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.softmax(input, output, config),
        }
    }

    #[inline(always)]
    pub fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        match self {
            Self::Cpu(b_impl) => b_impl.matmul(a, b, c, config),
            Self::Wgpu(b_impl) => b_impl.matmul(a, b, c, config),
            Self::Cuda(b_impl) => b_impl.matmul(a, b, c, config),
            Self::Rocm(b_impl) => b_impl.matmul(a, b, c, config),
            #[cfg(target_os = "macos")]
            Self::Metal(b_impl) => b_impl.matmul(a, b, c, config),
        }
    }

    #[inline(always)]
    pub fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.linear_forward_gpu(input, weight, output, params),
            Self::Wgpu(b) => b.linear_forward_gpu(input, weight, output, params),
            Self::Cuda(b) => b.linear_forward_gpu(input, weight, output, params),
            Self::Rocm(b) => b.linear_forward_gpu(input, weight, output, params),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.linear_forward_gpu(input, weight, output, params),
        }
    }

    #[inline(always)]
    pub fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.linear_forward_gpu_add(input, weight, output, params),
            Self::Wgpu(b) => b.linear_forward_gpu_add(input, weight, output, params),
            Self::Cuda(b) => b.linear_forward_gpu_add(input, weight, output, params),
            Self::Rocm(b) => b.linear_forward_gpu_add(input, weight, output, params),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.linear_forward_gpu_add(input, weight, output, params),
        }
    }

    #[inline(always)]
    pub fn linear_forward_host_io<T: KernelFloat>(&self, input: &[T], weight: &GpuTensor, output: &mut [T], params: LinearParams) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.linear_forward_host_io(input, weight, output, params),
            Self::Wgpu(b) => b.linear_forward_host_io(input, weight, output, params),
            Self::Cuda(b) => b.linear_forward_host_io(input, weight, output, params),
            Self::Rocm(b) => b.linear_forward_host_io(input, weight, output, params),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.linear_forward_host_io(input, weight, output, params),
        }
    }

    #[inline(always)]
    pub fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.linear_forward_host_io_readback(gpu_tensor, output),
            Self::Wgpu(b) => b.linear_forward_host_io_readback(gpu_tensor, output),
            Self::Cuda(b) => b.linear_forward_host_io_readback(gpu_tensor, output),
            Self::Rocm(b) => b.linear_forward_host_io_readback(gpu_tensor, output),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.linear_forward_host_io_readback(gpu_tensor, output),
        }
    }

    #[inline(always)]
    pub fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.ffn_forward_gpu(input, gate, up, down, intermediate, output, gate_up_params, down_params),
            Self::Wgpu(b) => b.ffn_forward_gpu(input, gate, up, down, intermediate, output, gate_up_params, down_params),
            Self::Cuda(b) => b.ffn_forward_gpu(input, gate, up, down, intermediate, output, gate_up_params, down_params),
            Self::Rocm(b) => b.ffn_forward_gpu(input, gate, up, down, intermediate, output, gate_up_params, down_params),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.ffn_forward_gpu(input, gate, up, down, intermediate, output, gate_up_params, down_params),
        }
    }

    #[inline(always)]
    pub fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.attention_forward_gpu(q, k_cache, v_cache, output, config),
            Self::Wgpu(b) => b.attention_forward_gpu(q, k_cache, v_cache, output, config),
            Self::Cuda(b) => b.attention_forward_gpu(q, k_cache, v_cache, output, config),
            Self::Rocm(b) => b.attention_forward_gpu(q, k_cache, v_cache, output, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.attention_forward_gpu(q, k_cache, v_cache, output, config),
        }
    }

    #[inline(always)]
    pub fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.rms_norm_gpu(input, weight, output, eps),
            Self::Wgpu(b) => b.rms_norm_gpu(input, weight, output, eps),
            Self::Cuda(b) => b.rms_norm_gpu(input, weight, output, eps),
            Self::Rocm(b) => b.rms_norm_gpu(input, weight, output, eps),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.rms_norm_gpu(input, weight, output, eps),
        }
    }

    #[inline(always)]
    pub fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.rms_norm_gpu_inplace(data, weight, eps),
            Self::Wgpu(b) => b.rms_norm_gpu_inplace(data, weight, eps),
            Self::Cuda(b) => b.rms_norm_gpu_inplace(data, weight, eps),
            Self::Rocm(b) => b.rms_norm_gpu_inplace(data, weight, eps),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.rms_norm_gpu_inplace(data, weight, eps),
        }
    }

    #[inline(always)]
    pub fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        match self {
            Self::Cpu(b) => b.allocate_buffer(size_bytes),
            Self::Wgpu(b) => b.allocate_buffer(size_bytes),
            Self::Cuda(b) => b.allocate_buffer(size_bytes),
            Self::Rocm(b) => b.allocate_buffer(size_bytes),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.allocate_buffer(size_bytes),
        }
    }

    #[inline(always)]
    pub fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        match self {
            Self::Cpu(b) => b.allocate_weights(data, shape, dtype),
            Self::Wgpu(b) => b.allocate_weights(data, shape, dtype),
            Self::Cuda(b) => b.allocate_weights(data, shape, dtype),
            Self::Rocm(b) => b.allocate_weights(data, shape, dtype),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.allocate_weights(data, shape, dtype),
        }
    }

    #[inline(always)]
    pub fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.readback(gpu, host),
            Self::Wgpu(b) => b.readback(gpu, host),
            Self::Cuda(b) => b.readback(gpu, host),
            Self::Rocm(b) => b.readback(gpu, host),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.readback(gpu, host),
        }
    }

    /// Read GPU u32 tensor back to host (type-safe).
    #[inline(always)]
    pub fn readback_u32(&self, gpu: &GpuTensor, host: &mut [u32]) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.readback_u32(gpu, host),
            Self::Wgpu(b) => b.readback_u32(gpu, host),
            Self::Cuda(b) => b.readback_u32(gpu, host),
            Self::Rocm(b) => b.readback_u32(gpu, host),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.readback_u32(gpu, host),
        }
    }

    #[inline(always)]
    pub fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.upload(host, gpu),
            Self::Wgpu(b) => b.upload(host, gpu),
            Self::Cuda(b) => b.upload(host, gpu),
            Self::Rocm(b) => b.upload(host, gpu),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.upload(host, gpu),
        }
    }

    #[inline(always)]
    pub fn update_kv_cache_gpu(&self, cache: &mut GpuKVCache, layer_idx: usize, new_k: &GpuTensor, new_v: &GpuTensor) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.update_kv_cache_gpu(cache, layer_idx, new_k, new_v),
            Self::Wgpu(b) => b.update_kv_cache_gpu(cache, layer_idx, new_k, new_v),
            Self::Cuda(b) => b.update_kv_cache_gpu(cache, layer_idx, new_k, new_v),
            Self::Rocm(b) => b.update_kv_cache_gpu(cache, layer_idx, new_k, new_v),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.update_kv_cache_gpu(cache, layer_idx, new_k, new_v),
        }
    }

    // MoE GPU Operations

    #[inline(always)]
    pub fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.tensor_zero_gpu(tensor),
            Self::Wgpu(b) => b.tensor_zero_gpu(tensor),
            Self::Cuda(b) => b.tensor_zero_gpu(tensor),
            Self::Rocm(b) => b.tensor_zero_gpu(tensor),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.tensor_zero_gpu(tensor),
        }
    }

    #[inline(always)]
    pub fn tensor_add_gpu(&self, output: &mut GpuTensor, input: &GpuTensor) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.tensor_add_gpu(output, input),
            Self::Wgpu(b) => b.tensor_add_gpu(output, input),
            Self::Cuda(b) => b.tensor_add_gpu(output, input),
            Self::Rocm(b) => b.tensor_add_gpu(output, input),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.tensor_add_gpu(output, input),
        }
    }

    #[inline(always)]
    pub fn tensor_slice_gpu(&self, input: &GpuTensor, offset: usize, len: usize, output: &mut GpuTensor) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.tensor_slice_gpu(input, offset, len, output),
            Self::Wgpu(b) => b.tensor_slice_gpu(input, offset, len, output),
            Self::Cuda(b) => b.tensor_slice_gpu(input, offset, len, output),
            Self::Rocm(b) => b.tensor_slice_gpu(input, offset, len, output),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.tensor_slice_gpu(input, offset, len, output),
        }
    }

    #[inline(always)]
    pub fn tensor_scale_add_gpu(&self, input: &GpuTensor, output: &mut GpuTensor, offset: usize, scale: f32) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.tensor_scale_add_gpu(input, output, offset, scale),
            Self::Wgpu(b) => b.tensor_scale_add_gpu(input, output, offset, scale),
            Self::Cuda(b) => b.tensor_scale_add_gpu(input, output, offset, scale),
            Self::Rocm(b) => b.tensor_scale_add_gpu(input, output, offset, scale),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.tensor_scale_add_gpu(input, output, offset, scale),
        }
    }

    #[inline(always)]
    pub fn moe_route_gpu(
        &self,
        hidden_states: &GpuTensor,
        gate_weights: &GpuTensor,
        expert_indices_out: &mut GpuTensor,
        expert_weights_out: &mut GpuTensor,
        config: MoERoutingGpuConfig,
    ) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.moe_route_gpu(hidden_states, gate_weights, expert_indices_out, expert_weights_out, config),
            Self::Wgpu(b) => b.moe_route_gpu(hidden_states, gate_weights, expert_indices_out, expert_weights_out, config),
            Self::Cuda(b) => b.moe_route_gpu(hidden_states, gate_weights, expert_indices_out, expert_weights_out, config),
            Self::Rocm(b) => b.moe_route_gpu(hidden_states, gate_weights, expert_indices_out, expert_weights_out, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.moe_route_gpu(hidden_states, gate_weights, expert_indices_out, expert_weights_out, config),
        }
    }

    /// Fused MoE forward with routing tensors on GPU.
    #[inline(always)]
    pub fn moe_forward_gpu_pure(
        &self,
        input: &GpuTensor,
        expert_indices: &GpuTensor,
        expert_weights: &GpuTensor,
        all_gate_weights: &GpuTensor,
        all_up_weights: &GpuTensor,
        all_down_weights: &GpuTensor,
        output: &mut GpuTensor,
        config: MoEForwardConfig,
    ) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.moe_forward_gpu_pure(input, expert_indices, expert_weights, all_gate_weights, all_up_weights, all_down_weights, output, config),
            Self::Wgpu(b) => b.moe_forward_gpu_pure(input, expert_indices, expert_weights, all_gate_weights, all_up_weights, all_down_weights, output, config),
            Self::Cuda(b) => b.moe_forward_gpu_pure(input, expert_indices, expert_weights, all_gate_weights, all_up_weights, all_down_weights, output, config),
            Self::Rocm(b) => b.moe_forward_gpu_pure(input, expert_indices, expert_weights, all_gate_weights, all_up_weights, all_down_weights, output, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.moe_forward_gpu_pure(input, expert_indices, expert_weights, all_gate_weights, all_up_weights, all_down_weights, output, config),
        }
    }

    #[inline(always)]
    pub fn eagle3_confidence<T: KernelFloat>(&self, layer_hidden_states: &[&[T]], confidence_weights: &[T], confidence_bias: T, config: &Eagle3Config) -> Option<Vec<T>> {
        match self {
            Self::Cpu(b) => b.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config),
            Self::Wgpu(b) => b.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config),
            Self::Cuda(b) => b.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config),
            Self::Rocm(b) => b.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config),
        }
    }

    #[inline(always)]
    pub fn spec_ee_confidence<T: KernelFloat>(&self, hidden_states: &[T], classifier_weight: &[T], classifier_bias: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        match self {
            Self::Cpu(b) => b.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config),
            Self::Wgpu(b) => b.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config),
            Self::Cuda(b) => b.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config),
            Self::Rocm(b) => b.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config),
        }
    }

    #[inline(always)]
    pub fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        match self {
            Self::Cpu(b) => b.flash_tree_attention(query, key, value, tree_mask, output, config),
            Self::Wgpu(b) => b.flash_tree_attention(query, key, value, tree_mask, output, config),
            Self::Cuda(b) => b.flash_tree_attention(query, key, value, tree_mask, output, config),
            Self::Rocm(b) => b.flash_tree_attention(query, key, value, tree_mask, output, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.flash_tree_attention(query, key, value, tree_mask, output, config),
        }
    }

    #[inline(always)]
    pub fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        match self {
            Self::Cpu(b) => b.medusa_forward(head_logits, config),
            Self::Wgpu(b) => b.medusa_forward(head_logits, config),
            Self::Cuda(b) => b.medusa_forward(head_logits, config),
            Self::Rocm(b) => b.medusa_forward(head_logits, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.medusa_forward(head_logits, config),
        }
    }

    #[inline(always)]
    pub fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        match self {
            Self::Cpu(b) => b.medusa_verify(candidate_tokens, target_logits, config),
            Self::Wgpu(b) => b.medusa_verify(candidate_tokens, target_logits, config),
            Self::Cuda(b) => b.medusa_verify(candidate_tokens, target_logits, config),
            Self::Rocm(b) => b.medusa_verify(candidate_tokens, target_logits, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.medusa_verify(candidate_tokens, target_logits, config),
        }
    }

    #[inline(always)]
    pub fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        match self {
            Self::Cpu(b) => b.int2_quantize(input, config),
            Self::Wgpu(b) => b.int2_quantize(input, config),
            Self::Cuda(b) => b.int2_quantize(input, config),
            Self::Rocm(b) => b.int2_quantize(input, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.int2_quantize(input, config),
        }
    }

    #[inline(always)]
    pub fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        match self {
            Self::Cpu(b) => b.int2_dequantize(quantized, scales, zeros, config),
            Self::Wgpu(b) => b.int2_dequantize(quantized, scales, zeros, config),
            Self::Cuda(b) => b.int2_dequantize(quantized, scales, zeros, config),
            Self::Rocm(b) => b.int2_dequantize(quantized, scales, zeros, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.int2_dequantize(quantized, scales, zeros, config),
        }
    }

    #[inline(always)]
    pub fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        match self {
            Self::Cpu(b) => b.evic_press_compress(kv_cache, config),
            Self::Wgpu(b) => b.evic_press_compress(kv_cache, config),
            Self::Cuda(b) => b.evic_press_compress(kv_cache, config),
            Self::Rocm(b) => b.evic_press_compress(kv_cache, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.evic_press_compress(kv_cache, config),
        }
    }

    #[inline(always)]
    pub fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        match self {
            Self::Cpu(b) => b.evic_press_evict(attention_weights, token_ages, current_zones, config),
            Self::Wgpu(b) => b.evic_press_evict(attention_weights, token_ages, current_zones, config),
            Self::Cuda(b) => b.evic_press_evict(attention_weights, token_ages, current_zones, config),
            Self::Rocm(b) => b.evic_press_evict(attention_weights, token_ages, current_zones, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.evic_press_evict(attention_weights, token_ages, current_zones, config),
        }
    }

    #[inline(always)]
    pub fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        match self {
            Self::Cpu(b) => b.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config),
            Self::Wgpu(b) => b.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config),
            Self::Cuda(b) => b.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config),
            Self::Rocm(b) => b.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config),
        }
    }

    #[inline(always)]
    pub fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        match self {
            Self::Cpu(b) => b.prompt_cache_blend(cached_kv, fresh_kv, config),
            Self::Wgpu(b) => b.prompt_cache_blend(cached_kv, fresh_kv, config),
            Self::Cuda(b) => b.prompt_cache_blend(cached_kv, fresh_kv, config),
            Self::Rocm(b) => b.prompt_cache_blend(cached_kv, fresh_kv, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.prompt_cache_blend(cached_kv, fresh_kv, config),
        }
    }

    #[inline(always)]
    pub fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        match self {
            Self::Cpu(b) => b.chunked_prefill_attention(query, key, value, config),
            Self::Wgpu(b) => b.chunked_prefill_attention(query, key, value, config),
            Self::Cuda(b) => b.chunked_prefill_attention(query, key, value, config),
            Self::Rocm(b) => b.chunked_prefill_attention(query, key, value, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.chunked_prefill_attention(query, key, value, config),
        }
    }

    #[inline(always)]
    pub fn rerank_pipeline(&self, binary_query: &[u32], binary_database: &[u32], int8_query: &[u32], int8_database: &[u32], num_vectors: usize, config: &GpuRerankConfig, int8_scale: f32) -> Result<GpuRerankStageResult, String> {
        match self {
            Self::Cpu(b) => b.rerank_pipeline(binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale),
            Self::Wgpu(b) => b.rerank_pipeline(binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale),
            Self::Cuda(b) => b.rerank_pipeline(binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale),
            Self::Rocm(b) => b.rerank_pipeline(binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.rerank_pipeline(binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale),
        }
    }

    #[inline(always)]
    pub fn binary_ip_hamming(&self, queries: &[u64], database: &[u64], scores: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        match self {
            Self::Cpu(b) => b.binary_ip_hamming(queries, database, scores, config),
            Self::Wgpu(b) => b.binary_ip_hamming(queries, database, scores, config),
            Self::Cuda(b) => b.binary_ip_hamming(queries, database, scores, config),
            Self::Rocm(b) => b.binary_ip_hamming(queries, database, scores, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.binary_ip_hamming(queries, database, scores, config),
        }
    }

    #[inline(always)]
    pub fn binary_ip_asymmetric(&self, queries: &[f32], database: &[u64], scores: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        match self {
            Self::Cpu(b) => b.binary_ip_asymmetric(queries, database, scores, config),
            Self::Wgpu(b) => b.binary_ip_asymmetric(queries, database, scores, config),
            Self::Cuda(b) => b.binary_ip_asymmetric(queries, database, scores, config),
            Self::Rocm(b) => b.binary_ip_asymmetric(queries, database, scores, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.binary_ip_asymmetric(queries, database, scores, config),
        }
    }

    #[inline(always)]
    pub fn int8_dot_product(&self, queries: &[i8], database: &[i8], scores: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        match self {
            Self::Cpu(b) => b.int8_dot_product(queries, database, scores, config),
            Self::Wgpu(b) => b.int8_dot_product(queries, database, scores, config),
            Self::Cuda(b) => b.int8_dot_product(queries, database, scores, config),
            Self::Rocm(b) => b.int8_dot_product(queries, database, scores, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.int8_dot_product(queries, database, scores, config),
        }
    }

    #[inline(always)]
    pub fn int4_packed_dot_product(&self, queries: &[u8], database: &[u8], scores: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        match self {
            Self::Cpu(b) => b.int4_packed_dot_product(queries, database, scores, config),
            Self::Wgpu(b) => b.int4_packed_dot_product(queries, database, scores, config),
            Self::Cuda(b) => b.int4_packed_dot_product(queries, database, scores, config),
            Self::Rocm(b) => b.int4_packed_dot_product(queries, database, scores, config),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.int4_packed_dot_product(queries, database, scores, config),
        }
    }

    #[inline(always)]
    pub fn backend_type(&self) -> BackendType {
        match self {
            Self::Cpu(b) => b.backend_type(),
            Self::Wgpu(b) => b.backend_type(),
            Self::Cuda(b) => b.backend_type(),
            Self::Rocm(b) => b.backend_type(),
#[cfg(target_os = "macos")]
            Self::Metal(b) => b.backend_type(),
        }
    }
}

// =============================================================================
// Device Static Globals
// =============================================================================

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
static WGPU_CONTEXT: OnceLock<Option<WgpuContext>> = OnceLock::new();
static WGPU_LINEAR_KERNEL: OnceLock<Option<WgpuLinear>> = OnceLock::new();
static WGPU_RMS_NORM_KERNEL: OnceLock<Option<WgpuRmsNorm>> = OnceLock::new();
static WGPU_KERNEL: OnceLock<Option<WgpuFlashAttentionKernel>> = OnceLock::new();
static WGPU_PAGED_KERNEL: OnceLock<Option<WgpuPagedAttentionKernel>> = OnceLock::new();
static WGPU_EMBEDDING_KERNEL: OnceLock<Option<WgpuEmbeddingOpsKernel>> = OnceLock::new();
static WGPU_EAGLE3_KERNEL: OnceLock<Option<WgpuEagle3Kernel>> = OnceLock::new();
static WGPU_SPEC_EE_KERNEL: OnceLock<Option<WgpuSpecEEKernel>> = OnceLock::new();
static WGPU_FLASH_TREE_KERNEL: OnceLock<Option<WgpuFlashTreeAttn>> = OnceLock::new();
static WGPU_INT2_KERNEL: OnceLock<Option<WgpuInt2Quantizer>> = OnceLock::new();
static WGPU_EVICT_PRESS_KERNEL: OnceLock<Option<WgpuEvicPress>> = OnceLock::new();
static WGPU_MEDUSA_KERNEL: OnceLock<Option<WgpuMedusa>> = OnceLock::new();
static WGPU_PROMPT_CACHE_KERNEL: OnceLock<Option<WgpuPromptCache>> = OnceLock::new();
static WGPU_CHUNKED_PREFILL_KERNEL: OnceLock<Option<WgpuChunkedPrefill>> = OnceLock::new();
static WGPU_TENSOR_OPS: OnceLock<Option<WgpuTensorOps>> = OnceLock::new();
static WGPU_MOE_ROUTING_KERNEL: OnceLock<Option<WgpuMoERouting>> = OnceLock::new();

static CUDA_CONTEXT: OnceLock<Option<Arc<CudaContext>>> = OnceLock::new();
static CUDA_STREAM: OnceLock<Option<Arc<CudaStream>>> = OnceLock::new();
static CUDA_FLASH_ATTN: OnceLock<Option<CudaFlashAttentionKernel>> = OnceLock::new();
static CUDA_PAGED_ATTN: OnceLock<Option<CudaPagedAttentionKernel>> = OnceLock::new();
static CUDA_EAGLE3_KERNEL: OnceLock<Option<CudaEagle3Kernel>> = OnceLock::new();
static CUDA_SPEC_EE_KERNEL: OnceLock<Option<CudaSpecEEKernel>> = OnceLock::new();
static CUDA_FLASH_TREE_KERNEL: OnceLock<Option<CudaFlashTreeAttnKernel>> = OnceLock::new();
static CUDA_INT2_KERNEL: OnceLock<Option<CudaInt2QuantizerKernel>> = OnceLock::new();
static CUDA_EVICT_PRESS_KERNEL: OnceLock<Option<CudaEvicPressKernel>> = OnceLock::new();
static CUDA_MEDUSA_KERNEL: OnceLock<Option<CudaMedusaKernel>> = OnceLock::new();
static CUDA_PROMPT_CACHE_KERNEL: OnceLock<Option<CudaPromptCacheKernel>> = OnceLock::new();
static CUDA_CHUNKED_PREFILL_KERNEL: OnceLock<Option<CudaChunkedPrefillKernel>> = OnceLock::new();
static CUDA_LINEAR_KERNEL: OnceLock<Option<CudaLinear>> = OnceLock::new();
static CUDA_RMS_NORM_KERNEL: OnceLock<Option<CudaRmsNormKernel>> = OnceLock::new();
static CUDA_EMBEDDING_KERNEL: OnceLock<Option<CudaEmbeddingOpsKernel>> = OnceLock::new();
static CUDA_SOFTMAX_KERNEL: OnceLock<Option<CudaOnlineSoftmaxKernel>> = OnceLock::new();

// ROCm static variables - Fat Binary: always compiled, runtime detection
static ROCM_AGENT: OnceLock<Option<GpuAgent>> = OnceLock::new();
static ROCM_QUEUE: OnceLock<Option<HsaQueueWrapper>> = OnceLock::new();
static ROCM_FLASH_ATTN: OnceLock<Option<HsaFlashAttentionKernel>> = OnceLock::new();
static ROCM_PAGED_ATTN: OnceLock<Option<HsaPagedAttentionKernel>> = OnceLock::new();
static ROCM_RMS_NORM_KERNEL: OnceLock<Option<HsaRmsNormKernel>> = OnceLock::new();
static ROCM_EAGLE3_KERNEL: OnceLock<Option<HsaEagle3Kernel>> = OnceLock::new();
static ROCM_SPEC_EE_KERNEL: OnceLock<Option<HsaSpecEEKernel>> = OnceLock::new();
static ROCM_FLASH_TREE_KERNEL: OnceLock<Option<HsaFlashTreeAttnKernel>> = OnceLock::new();
static ROCM_INT2_KERNEL: OnceLock<Option<HsaInt2QuantizerKernel>> = OnceLock::new();
static ROCM_EVICT_PRESS_KERNEL: OnceLock<Option<HsaEvicPressKernel>> = OnceLock::new();
static ROCM_MEDUSA_KERNEL: OnceLock<Option<HsaMedusaKernel>> = OnceLock::new();
static ROCM_PROMPT_CACHE_KERNEL: OnceLock<Option<HsaPromptCacheKernel>> = OnceLock::new();
static ROCM_CHUNKED_PREFILL_KERNEL: OnceLock<Option<HsaChunkedPrefillKernel>> = OnceLock::new();
static ROCM_LINEAR_KERNEL: OnceLock<Option<HsaLinear>> = OnceLock::new();
static ROCM_EMBEDDING_KERNEL: OnceLock<Option<HsaEmbeddingOpsKernel>> = OnceLock::new();

// Metal static variables (macOS only)
#[cfg(target_os = "macos")]
static METAL_FLASH_ATTN: OnceLock<Option<MetalFlashAttentionKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_PAGED_ATTN: OnceLock<Option<MetalPagedAttentionKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_EAGLE3_KERNEL: OnceLock<Option<MetalEagle3Kernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_SPEC_EE_KERNEL: OnceLock<Option<MetalSpecEEKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_FLASH_TREE_KERNEL: OnceLock<Option<MetalFlashTreeAttnKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_INT2_KERNEL: OnceLock<Option<MetalInt2QuantizerKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_EVICT_PRESS_KERNEL: OnceLock<Option<MetalEvicPressKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_MEDUSA_KERNEL: OnceLock<Option<MetalMedusaKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_PROMPT_CACHE_KERNEL: OnceLock<Option<MetalPromptCacheKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_CHUNKED_PREFILL_KERNEL: OnceLock<Option<MetalChunkedPrefillKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_LINEAR_KERNEL: OnceLock<Option<MetalLinear>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_RMS_NORM_KERNEL: OnceLock<Option<MetalRmsNorm>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_EMBEDDING_KERNEL: OnceLock<Option<MetalEmbeddingOpsKernel>> = OnceLock::new();

fn get_cuda_context() -> Option<&'static Arc<CudaContext>> {
    // CudaContext::new(0) already returns Result<Arc<CudaContext>, _>
    CUDA_CONTEXT.get_or_init(|| CudaContext::new(0).ok()).as_ref()
}

fn get_cuda_stream() -> Option<&'static Arc<CudaStream>> {
    let ctx = get_cuda_context()?;
    CUDA_STREAM.get_or_init(|| Some(ctx.default_stream())).as_ref()
}
pub(crate) fn get_wgpu_context() -> Option<&'static WgpuContext> {
    WGPU_CONTEXT.get_or_init(|| {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.ok()?;
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.ok()?;
            Some(WgpuContext { device, queue })
        })
    }).as_ref()
}

// ROCm helper functions - Fat Binary: always compiled, runtime detection
fn get_rocm_agent() -> Option<&'static GpuAgent> {
    ROCM_AGENT
        .get_or_init(|| find_gpu_agents().ok().and_then(|agents| agents.into_iter().next()))
        .as_ref()
}

fn get_rocm_queue() -> Option<&'static HsaQueueWrapper> {
    let agent = get_rocm_agent()?;
    ROCM_QUEUE.get_or_init(|| HsaQueueWrapper::new(agent).ok()).as_ref()
}

fn init_cuda_kernel<T, E, F>(lock: &'static OnceLock<Option<T>>, init: F) -> Option<&'static T>
where
    F: FnOnce(&Arc<CudaContext>) -> Result<T, E>,
{
    let ctx = get_cuda_context()?;
    lock.get_or_init(|| init(ctx).ok()).as_ref()
}

fn init_rocm_kernel<T, E, F>(lock: &'static OnceLock<Option<T>>, init: F) -> Option<&'static T>
where
    F: FnOnce(i32) -> Result<T, E>,
{
    // Use device 0 by default (similar to CUDA)
    lock.get_or_init(|| init(0).ok()).as_ref()
}

fn init_rocm_config_kernel<T, E, F>(lock: &'static OnceLock<Option<T>>, init: F) -> Option<&'static T>
where
    F: FnOnce() -> Result<T, E>,
{
    lock.get_or_init(|| init().ok()).as_ref()
}

#[cfg(target_os = "macos")]
fn init_metal_kernel<T, E, F>(lock: &'static OnceLock<Option<T>>, init: F) -> Option<&'static T>
where
    F: FnOnce(&metal::Device) -> Result<T, E>,
{
    let device = get_metal_device()?;
    lock.get_or_init(|| init(device).ok()).as_ref()
}

fn init_wgpu_kernel<T, E, F>(lock: &'static OnceLock<Option<T>>, init: F) -> Option<&'static T>
where
    F: FnOnce(&wgpu::Device, &wgpu::Queue) -> Result<T, E>,
{
    let ctx = get_wgpu_context()?;
    lock.get_or_init(|| init(&ctx.device, &ctx.queue).ok()).as_ref()
}

fn rms_norm_dims(
    input: &GpuTensor,
    weight: &GpuTensor,
    output: Option<&GpuTensor>,
) -> Result<(usize, usize), String> {
    if weight.shape.len() != 1 {
        return Err("RMSNorm weight must be 1D".into());
    }
    let hidden = weight.shape[0];
    if hidden == 0 {
        return Err("RMSNorm hidden dimension is zero".into());
    }
    let input_elems: usize = input.shape.iter().product();
    if input_elems % hidden != 0 {
        return Err("RMSNorm input size not divisible by hidden".into());
    }
    if let Some(out) = output {
        let output_elems: usize = out.shape.iter().product();
        if output_elems != input_elems {
            return Err("RMSNorm output shape mismatch".into());
        }
    }
    Ok((input_elems / hidden, hidden))
}

fn build_paged_tables_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    page_table: &[u32],
    seq_lens: &[u32],
    output_len: usize,
    config: &PagedAttentionConfig,
) -> Option<(ops::paged_attn::PagedAttentionLayout, Vec<i32>, Vec<i32>)> {
    let layout = ops::paged_attn::build_paged_layout(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        output_len,
        config,
    )?;
    let block_tables: Vec<i32> = page_table.iter().map(|&v| v as i32).collect();
    let seq_len_i32 = layout.seq_len as i32;
    let block_offsets: Vec<i32> = seq_lens
        .iter()
        .map(|&len| len as i32 - seq_len_i32)
        .collect();
    Some((layout, block_tables, block_offsets))
}

fn split_u64_to_u32(words: &[u64]) -> Vec<u32> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for &word in words {
        let bytes = word.to_le_bytes();
        out.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        out.push(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]));
    }
    out
}

fn pack_i8_to_u32(values: &[i8]) -> Vec<u32> {
    let mut out = Vec::with_capacity((values.len() + 3) / 4);
    for chunk in values.chunks(4) {
        let mut bytes = [0u8; 4];
        for (idx, &val) in chunk.iter().enumerate() {
            bytes[idx] = val as u8;
        }
        out.push(u32::from_le_bytes(bytes));
    }
    out
}

fn pack_u8_to_u32(values: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity((values.len() + 3) / 4);
    for chunk in values.chunks(4) {
        let mut bytes = [0u8; 4];
        for (idx, &val) in chunk.iter().enumerate() {
            bytes[idx] = val;
        }
        out.push(u32::from_le_bytes(bytes));
    }
    out
}

fn cuda_upload<T: DeviceRepr>(stream: &Arc<CudaStream>, data: &[T]) -> Result<CudaSlice<T>, String> {
    stream
        .clone_htod(data)
        .map_err(|err| format!("CUDA H2D copy failed: {err}"))
}

fn cuda_download<T: DeviceRepr>(stream: &Arc<CudaStream>, data: &CudaSlice<T>) -> Result<Vec<T>, String> {
    stream
        .clone_dtoh(data)
        .map_err(|err| format!("CUDA D2H copy failed: {err}"))
}

// MoE GPU helper functions for CUDA
// Note: These are placeholder implementations. For full performance, need custom CUDA kernels.

fn cuda_alloc_zeros(stream: &Arc<CudaStream>, size_bytes: usize) -> Result<CudaSlice<u8>, String> {
    // Allocate by uploading zeros (cudarc doesn't have memset API)
    let data = vec![0u8; size_bytes];
    stream.clone_htod(&data).map_err(|e| format!("CUDA alloc_zeros: {e}"))
}

fn cuda_tensor_add(_stream: &Arc<CudaStream>, _output: &CudaSlice<u8>, _input: &CudaSlice<u8>, _len: usize) -> Result<(), String> {
    // TODO: Implement as CUDA kernel (element-wise add)
    Err("CUDA tensor_add_gpu not yet implemented - requires custom kernel".into())
}

fn cuda_tensor_slice(_stream: &Arc<CudaStream>, _src: &CudaSlice<u8>, _dst: &CudaSlice<u8>, _src_offset: usize, _len: usize) -> Result<(), String> {
    // TODO: Implement as CUDA kernel or use cuMemcpy with offset
    Err("CUDA tensor_slice_gpu not yet implemented - requires custom kernel".into())
}

fn cuda_tensor_scale_add(_stream: &Arc<CudaStream>, _output: &CudaSlice<u8>, _input: &CudaSlice<u8>, _offset: usize, _len: usize, _scale: f32) -> Result<(), String> {
    // TODO: Implement as CUDA kernel (scale and add)
    Err("CUDA tensor_scale_add_gpu not yet implemented - requires custom kernel".into())
}

fn hsa_upload<T: Copy>(agent: &GpuAgent, data: &[T]) -> Result<HsaBuffer<T>, String> {
    HsaBuffer::from_slice(agent, data).map_err(|err| format!("HSA H2D copy failed: {err}"))
}

fn hsa_download<T: Default + Clone>(buffer: &HsaBuffer<T>) -> Result<Vec<T>, String> {
    buffer
        .to_vec()
        .map_err(|err| format!("HSA D2H copy failed: {err}"))
}

#[cfg(target_os = "macos")]
fn metal_buffer_from_slice<T: Copy>(device: &metal::Device, data: &[T]) -> Result<metal::Buffer, String> {
    let byte_len = data.len() * std::mem::size_of::<T>();
    if byte_len == 0 {
        return Ok(device.new_buffer(0, metal::MTLResourceOptions::StorageModeShared));
    }
    Ok(device.new_buffer_with_data(
        data.as_ptr() as *const _,
        byte_len as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ))
}

#[cfg(target_os = "macos")]
fn metal_download<T: Default + Copy>(buffer: &metal::Buffer, len: usize) -> Result<Vec<T>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let ptr = buffer.contents() as *const T;
    if ptr.is_null() {
        return Err("Metal buffer has null contents".into());
    }
    let mut out = vec![T::default(); len];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    }
    Ok(out)
}

fn cast_slice<T, U>(input: &[T]) -> &[U] {
    unsafe { std::slice::from_raw_parts(input.as_ptr() as *const U, input.len()) }
}

fn cast_slice_mut<T, U>(input: &mut [T]) -> &mut [U] {
    unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut U, input.len()) }
}

fn vec_from_f32<T: KernelFloat>(values: Vec<f32>) -> Vec<T> {
    values.into_iter().map(T::from_f32).collect()
}

fn chunked_prefill_from_f32<T: KernelFloat>(result: ChunkedPrefillResult<f32>) -> ChunkedPrefillResult<T> {
    ChunkedPrefillResult {
        output: vec_from_f32(result.output),
        log_sum_exp: result.log_sum_exp,
    }
}

fn f16_to_u16(values: &[half::f16]) -> Vec<u16> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn u16_to_f16(values: &[u16]) -> Vec<half::f16> {
    values.iter().map(|v| half::f16::from_bits(*v)).collect()
}

// =============================================================================
// CPU Backend Implementation
// =============================================================================
//
// CPU backend serves as:
// 1. Reference implementation for correctness verification
// 2. Fallback when GPU backends fail
// 3. SIMD-optimized implementation (AVX2/AVX512/NEON)
// =============================================================================

#[derive(Clone, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    // =========================================================================
    // Core Attention Operations (Generic)
    // =========================================================================

    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        // ops::attention::flash_attention is already generic over KernelFloat
        ops::attention::flash_attention(q, k, v, output, config);
    }

    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        ops::paged_attn::paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        ops::softmax::softmax(input, output, config);
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        ops::matmul(a, b, c, config);
    }

    // =========================================================================
    // GPU Tensor Operations (CPU returns error)
    // =========================================================================

    fn linear_forward_gpu(&self, _: &GpuTensor, _: &GpuTensor, _: &mut GpuTensor, _: LinearParams) -> Result<(), String> {
        Err("CPU backend does not support GPU tensor operations".into())
    }

    fn linear_forward_gpu_add(&self, _: &GpuTensor, _: &GpuTensor, _: &mut GpuTensor, _: LinearParams) -> Result<(), String> {
        Err("CPU backend does not support GPU tensor operations".into())
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, _: &[T], _: &GpuTensor, _: &mut [T], _: LinearParams) -> Result<(), String> {
        Err("CPU backend does not support GPU tensor operations".into())
    }

    fn linear_forward_host_io_readback(&self, _: &GpuTensor, _: &mut [f32]) -> Result<(), String> {
        Err("CPU backend does not support GPU tensor operations".into())
    }

    fn ffn_forward_gpu(&self, _: &GpuTensor, _: &GpuTensor, _: &GpuTensor, _: &GpuTensor, _: &mut GpuTensor, _: &mut GpuTensor, _: LinearParams, _: LinearParams) -> Result<(), String> {
        Err("CPU backend does not support GPU tensor operations".into())
    }

    fn attention_forward_gpu(&self, _: &GpuTensor, _: &GpuBuffer, _: &GpuBuffer, _: &mut GpuTensor, _: FlashAttentionConfig) -> Result<(), String> {
        Err("CPU backend does not support GPU tensor operations".into())
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        if input.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("CPU rms_norm_gpu only supports f32 tensors".into());
        }

        let (rows, hidden) = rms_norm_dims(input, weight, Some(output))?;
        let input_bytes = match &input.buffer {
            GpuBuffer::Cpu(buf) => buf,
            _ => return Err("CPU rms_norm_gpu expects CPU input buffer".into()),
        };
        let weight_bytes = match &weight.buffer {
            GpuBuffer::Cpu(buf) => buf,
            _ => return Err("CPU rms_norm_gpu expects CPU weight buffer".into()),
        };

        let input_f32 = bytemuck::try_cast_slice::<u8, f32>(input_bytes.as_ref())
            .map_err(|_| "CPU rms_norm_gpu input alignment error".to_string())?;
        let weight_f32 = bytemuck::try_cast_slice::<u8, f32>(weight_bytes.as_ref())
            .map_err(|_| "CPU rms_norm_gpu weight alignment error".to_string())?;

        let expected_len = rows * hidden;
        if input_f32.len() < expected_len || weight_f32.len() < hidden {
            return Err("CPU rms_norm_gpu buffer length mismatch".into());
        }

        let mut output_f32 = vec![0.0f32; expected_len];
        ops::rms_norm::rms_norm_forward(
            &input_f32[..expected_len],
            &weight_f32[..hidden],
            &mut output_f32,
            rows,
            hidden,
            eps,
        );

        let out_bytes = bytemuck::cast_slice(&output_f32).to_vec();
        if out_bytes.len() != output.size_in_bytes {
            return Err("CPU rms_norm_gpu output size mismatch".into());
        }
        output.buffer = GpuBuffer::Cpu(Arc::new(out_bytes));
        Ok(())
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        if data.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 {
            return Err("CPU rms_norm_gpu_inplace only supports f32 tensors".into());
        }

        let (rows, hidden) = rms_norm_dims(data, weight, None)?;
        let data_bytes = match &data.buffer {
            GpuBuffer::Cpu(buf) => buf,
            _ => return Err("CPU rms_norm_gpu_inplace expects CPU data buffer".into()),
        };
        let weight_bytes = match &weight.buffer {
            GpuBuffer::Cpu(buf) => buf,
            _ => return Err("CPU rms_norm_gpu_inplace expects CPU weight buffer".into()),
        };

        let data_f32 = bytemuck::try_cast_slice::<u8, f32>(data_bytes.as_ref())
            .map_err(|_| "CPU rms_norm_gpu_inplace data alignment error".to_string())?;
        let weight_f32 = bytemuck::try_cast_slice::<u8, f32>(weight_bytes.as_ref())
            .map_err(|_| "CPU rms_norm_gpu_inplace weight alignment error".to_string())?;

        let expected_len = rows * hidden;
        if data_f32.len() < expected_len || weight_f32.len() < hidden {
            return Err("CPU rms_norm_gpu_inplace buffer length mismatch".into());
        }

        let mut data_vec = data_f32[..expected_len].to_vec();
        ops::rms_norm::rms_norm_inplace(&mut data_vec, &weight_f32[..hidden], rows, hidden, eps);

        let out_bytes = bytemuck::cast_slice(&data_vec).to_vec();
        if out_bytes.len() != data.size_in_bytes {
            return Err("CPU rms_norm_gpu_inplace output size mismatch".into());
        }
        data.buffer = GpuBuffer::Cpu(Arc::new(out_bytes));
        Ok(())
    }

    fn allocate_buffer(&self, size: usize) -> Result<GpuBuffer, String> {
        // CPU can allocate a Vec as fallback
        Ok(GpuBuffer::Cpu(Arc::new(vec![0u8; size])))
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        Ok(GpuTensor::new(
            GpuBuffer::Cpu(Arc::new(data.to_vec())),
            shape,
            dtype,
            BackendType::Cpu,
        ))
    }

    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        match &gpu.buffer {
            GpuBuffer::Cpu(buf) => {
                // Reinterpret bytes as T
                let src = unsafe {
                    std::slice::from_raw_parts(
                        buf.as_ptr() as *const T,
                        buf.len() / std::mem::size_of::<T>(),
                    )
                };
                if src.len() >= host.len() {
                    host.copy_from_slice(&src[..host.len()]);
                    Ok(())
                } else {
                    Err("CPU readback: insufficient data".into())
                }
            }
            _ => Err("CPU readback: not a CPU buffer".into()),
        }
    }

    fn readback_u32(&self, _: &GpuTensor, _: &mut [u32]) -> Result<(), String> {
        Err("CPU readback_u32 not implemented".into())
    }

    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        match &gpu.buffer {
            GpuBuffer::Cpu(buf) => {
                // For CPU, we need to create a new buffer since Arc is immutable
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        host.as_ptr() as *const u8,
                        host.len() * std::mem::size_of::<T>(),
                    )
                };
                gpu.buffer = GpuBuffer::Cpu(Arc::new(bytes.to_vec()));
                Ok(())
            }
            _ => Err("CPU upload: not a CPU buffer".into()),
        }
    }

    fn update_kv_cache_gpu(&self, _: &mut GpuKVCache, _: usize, _: &GpuTensor, _: &GpuTensor) -> Result<(), String> {
        Err("CPU backend does not support GPU KV cache".into())
    }

    // =========================================================================
    // Speculative Decoding Operations
    // =========================================================================

    fn eagle3_confidence<T: KernelFloat>(&self, layers: &[&[T]], w: &[T], b: T, config: &Eagle3Config) -> Option<Vec<T>> {
        // Convert to f32 for computation, then back
        // ops::eagle3 works with f32 internally
        let layers_f32: Vec<Vec<f32>> = layers.iter().map(|l| l.iter().map(|v| v.to_f32()).collect()).collect();
        let layers_refs: Vec<&[f32]> = layers_f32.iter().map(|v| v.as_slice()).collect();
        let w_f32: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
        let result = ops::eagle3::fused_confidence_predict(&layers_refs, &w_f32, b.to_f32(), config);
        Some(result.into_iter().map(T::from_f32).collect())
    }

    fn spec_ee_confidence<T: KernelFloat>(&self, hs: &[T], w: &[T], b: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        let hs_f32: Vec<f32> = hs.iter().map(|v| v.to_f32()).collect();
        let w_f32: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
        let result = ops::spec_ee::compute_confidence_stateless(&hs_f32, &w_f32, b.to_f32(), config);
        Some(result.into_iter().map(T::from_f32).collect())
    }

    fn flash_tree_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], m: &[i32], o: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        ops::flash_tree_attn::flash_tree_attention(q, k, v, m, o, config);
        true
    }

    fn medusa_forward<T: KernelFloat>(&self, hl: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        let hl_f32: Vec<f32> = hl.iter().map(|v| v.to_f32()).collect();
        ops::medusa::medusa_forward_stateless(&hl_f32, config).ok()
    }

    fn medusa_verify<T: KernelFloat>(&self, ct: &[i32], tl: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        let tl_f32: Vec<f32> = tl.iter().map(|v| v.to_f32()).collect();
        ops::medusa::medusa_verify_stateless(ct, &tl_f32, config).ok()
    }

    // =========================================================================
    // Quantization Operations
    // =========================================================================

    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        ops::int2_quantizer::int2_quantize(&input_f32, config).ok()
    }

    fn int2_dequantize<T: KernelFloat>(&self, q: &[i8], s: &[T], z: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        let s_f32: Vec<f32> = s.iter().map(|v| v.to_f32()).collect();
        let z_f32: Vec<f32> = z.iter().map(|v| v.to_f32()).collect();
        let result = ops::int2_quantizer::int2_dequantize(q, &s_f32, &z_f32, config).ok()?;
        Some(result.into_iter().map(T::from_f32).collect())
    }

    // =========================================================================
    // KV Cache Compression
    // =========================================================================

    fn evic_press_compress<T: KernelFloat>(&self, _kv: &[T], _config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        // TODO: Implement CPU version
        None
    }

    fn evic_press_evict<T: KernelFloat>(&self, _w: &[T], _ages: &[i32], _zones: &[i32], _config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        // TODO: Implement CPU version
        None
    }

    // =========================================================================
    // Prompt Caching
    // =========================================================================

    fn prompt_cache_lookup(&self, t: &[i32], h: &[u64], l: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        ops::prompt_cache::prompt_cache_lookup(t, h, l, config).ok()
    }

    fn prompt_cache_blend<T: KernelFloat>(&self, c: &[T], f: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        let c_f32: Vec<f32> = c.iter().map(|v| v.to_f32()).collect();
        let f_f32: Vec<f32> = f.iter().map(|v| v.to_f32()).collect();
        let result = ops::prompt_cache::prompt_cache_blend(&c_f32, &f_f32, config).ok()?;
        Some(result.into_iter().map(T::from_f32).collect())
    }

    // =========================================================================
    // Chunked Prefill
    // =========================================================================

    fn chunked_prefill_attention<T: KernelFloat>(&self, _q: &[T], _k: &[T], _v: &[T], _config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        // TODO: Implement CPU version
        None
    }

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    fn rerank_pipeline(&self, bq: &[u32], bd: &[u32], iq: &[u32], id: &[u32], nv: usize, config: &GpuRerankConfig, is: f32) -> Result<GpuRerankStageResult, String> {
        ops::embedding::rerank_pipeline(bq, bd, iq, id, nv, config, is)
    }

    fn binary_ip_hamming(&self, q: &[u64], d: &[u64], s: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        crate::ops::embedding::binary_ip_hamming_simd(q, d, s, config);
    }

    fn binary_ip_asymmetric(&self, q: &[f32], d: &[u64], s: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        crate::ops::embedding::binary_ip_asymmetric(q, d, s, config);
    }

    fn int8_dot_product(&self, q: &[i8], d: &[i8], s: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        crate::ops::embedding::int8_dot_product_unrolled(q, d, s, config);
    }

    fn int4_packed_dot_product(&self, q: &[u8], d: &[u8], s: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        crate::ops::embedding::int4_packed_dot_product(q, d, s, config);
    }

    // =========================================================================
    // MoE GPU Operations (CPU fallback)
    // =========================================================================

    fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String> {
        match &mut tensor.buffer {
            GpuBuffer::Cpu(buf) => {
                let data = Arc::make_mut(buf);
                data.fill(0);
                Ok(())
            }
            _ => Err("CPU backend: tensor_zero_gpu requires CPU buffer".into()),
        }
    }

    fn tensor_add_gpu(&self, output: &mut GpuTensor, input: &GpuTensor) -> Result<(), String> {
        let input_slice = match &input.buffer {
            GpuBuffer::Cpu(buf) => bytemuck::try_cast_slice::<u8, f32>(buf.as_ref())
                .map_err(|_| "CPU tensor_add_gpu input alignment error".to_string())?,
            _ => return Err("CPU backend: tensor_add_gpu requires CPU buffer".into()),
        };
        match &mut output.buffer {
            GpuBuffer::Cpu(buf) => {
                let data = Arc::make_mut(buf);
                let output_slice = bytemuck::try_cast_slice_mut::<u8, f32>(data)
                    .map_err(|_| "CPU tensor_add_gpu output alignment error".to_string())?;
                for (o, i) in output_slice.iter_mut().zip(input_slice.iter()) {
                    *o += i;
                }
                Ok(())
            }
            _ => Err("CPU backend: tensor_add_gpu requires CPU buffer".into()),
        }
    }

    fn tensor_slice_gpu(&self, input: &GpuTensor, offset: usize, len: usize, output: &mut GpuTensor) -> Result<(), String> {
        let input_slice = match &input.buffer {
            GpuBuffer::Cpu(buf) => bytemuck::try_cast_slice::<u8, f32>(buf.as_ref())
                .map_err(|_| "CPU tensor_slice_gpu input alignment error".to_string())?,
            _ => return Err("CPU backend: tensor_slice_gpu requires CPU buffer".into()),
        };
        if offset + len > input_slice.len() {
            return Err(format!("CPU tensor_slice_gpu: offset {} + len {} > input len {}", offset, len, input_slice.len()));
        }
        match &mut output.buffer {
            GpuBuffer::Cpu(buf) => {
                let data = Arc::make_mut(buf);
                let output_slice = bytemuck::try_cast_slice_mut::<u8, f32>(data)
                    .map_err(|_| "CPU tensor_slice_gpu output alignment error".to_string())?;
                if output_slice.len() < len {
                    return Err(format!("CPU tensor_slice_gpu: output len {} < requested len {}", output_slice.len(), len));
                }
                output_slice[..len].copy_from_slice(&input_slice[offset..offset + len]);
                Ok(())
            }
            _ => Err("CPU backend: tensor_slice_gpu requires CPU buffer".into()),
        }
    }

    fn tensor_scale_add_gpu(&self, input: &GpuTensor, output: &mut GpuTensor, offset: usize, scale: f32) -> Result<(), String> {
        let input_slice = match &input.buffer {
            GpuBuffer::Cpu(buf) => bytemuck::try_cast_slice::<u8, f32>(buf.as_ref())
                .map_err(|_| "CPU tensor_scale_add_gpu input alignment error".to_string())?,
            _ => return Err("CPU backend: tensor_scale_add_gpu requires CPU buffer".into()),
        };
        match &mut output.buffer {
            GpuBuffer::Cpu(buf) => {
                let data = Arc::make_mut(buf);
                let output_slice = bytemuck::try_cast_slice_mut::<u8, f32>(data)
                    .map_err(|_| "CPU tensor_scale_add_gpu output alignment error".to_string())?;
                if offset + input_slice.len() > output_slice.len() {
                    return Err(format!("CPU tensor_scale_add_gpu: offset {} + input len {} > output len {}", offset, input_slice.len(), output_slice.len()));
                }
                for (i, &v) in input_slice.iter().enumerate() {
                    output_slice[offset + i] += v * scale;
                }
                Ok(())
            }
            _ => Err("CPU backend: tensor_scale_add_gpu requires CPU buffer".into()),
        }
    }

    fn moe_route_gpu(
        &self,
        _: &GpuTensor,
        _: &GpuTensor,
        _: &mut GpuTensor,
        _: &mut GpuTensor,
        _: MoERoutingGpuConfig,
    ) -> Result<(), String> {
        Err("CPU backend does not support GPU MoE routing".into())
    }

    fn moe_forward_gpu_pure(
        &self,
        _: &GpuTensor,
        _: &GpuTensor,
        _: &GpuTensor,
        _: &GpuTensor,
        _: &GpuTensor,
        _: &GpuTensor,
        _: &mut GpuTensor,
        _: MoEForwardConfig,
    ) -> Result<(), String> {
        Err("CPU moe_forward_gpu_pure not implemented".into())
    }

    // =========================================================================
    // Backend Info
    // =========================================================================

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }
}

// =============================================================================
// WGPU Backend
// =============================================================================

#[derive(Clone, Default)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    // =========================================================================
    // Core Attention Operations (Generic) - Uses WGPU kernels with CPU fallback
    // =========================================================================

    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        let kernel = match WGPU_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            WgpuFlashAttentionKernel::new(&ctx.device, &ctx.queue).ok()
        }) {
            Some(k) => k,
            None => {
                log::warn!("WGPU flash attention kernel unavailable; using CPU");
                return CpuBackend.flash_attention(q, k, v, output, config);
            }
        };

        let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
        let batch_size = q.len() / (config.num_heads * config.seq_len_q * config.head_dim);

        match T::TYPE_ID {
            FloatType::F32 => {
                if let Ok(result) = kernel.forward_f32(
                    cast_slice(q),
                    cast_slice(k),
                    cast_slice(v),
                    batch_size,
                    config.num_heads,
                    config.seq_len_q,
                    config.seq_len_kv,
                    config.head_dim,
                    Some(config.block_size_q as u32),
                    scale,
                ) {
                    let out_slice = cast_slice_mut::<T, f32>(output);
                    out_slice.copy_from_slice(&result);
                } else {
                    CpuBackend.flash_attention(q, k, v, output, config);
                }
            }
            FloatType::F16 => {
                if let Ok(result) = kernel.forward_f16(
                    cast_slice(q),
                    cast_slice(k),
                    cast_slice(v),
                    batch_size,
                    config.num_heads,
                    config.seq_len_q,
                    config.seq_len_kv,
                    config.head_dim,
                    Some(config.block_size_q as u32),
                    scale,
                ) {
                    let out_slice = cast_slice_mut::<T, half::f16>(output);
                    out_slice.copy_from_slice(&result);
                } else {
                    CpuBackend.flash_attention(q, k, v, output, config);
                }
            }
            _ => CpuBackend.flash_attention(q, k, v, output, config),
        }
    }

    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        let kernel = match WGPU_PAGED_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            WgpuPagedAttentionKernel::new(&ctx.device, &ctx.queue).ok()
        }) {
            Some(k) => k,
            None => return CpuBackend.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
        };

        // Convert page_table (u32) to block_tables (i32)
        let block_tables: Vec<i32> = page_table.iter().map(|&x| x as i32).collect();

        // Compute block_offsets from seq_lens (cumulative sum)
        let mut block_offsets: Vec<i32> = Vec::with_capacity(seq_lens.len() + 1);
        block_offsets.push(0);
        let mut offset = 0i32;
        for &len in seq_lens {
            offset += ((len as usize + config.block_size - 1) / config.block_size) as i32;
            block_offsets.push(offset);
        }

        let batch_size = seq_lens.len();
        let max_seq_len = seq_lens.iter().copied().max().unwrap_or(1) as usize;

        match T::TYPE_ID {
            FloatType::F32 => {
                match kernel.forward_f32(
                    cast_slice(q),
                    cast_slice(k_cache),
                    cast_slice(v_cache),
                    &block_tables,
                    &block_offsets,
                    batch_size,
                    config.num_kv_heads,
                    config.head_dim,
                    config.block_size,
                    max_seq_len,
                ) {
                    Ok(result) => {
                        let out_slice = cast_slice_mut::<T, f32>(output);
                        out_slice.copy_from_slice(&result);
                    }
                    Err(_) => CpuBackend.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
                }
            }
            FloatType::F16 => {
                match kernel.forward_f16(
                    cast_slice(q),
                    cast_slice(k_cache),
                    cast_slice(v_cache),
                    &block_tables,
                    &block_offsets,
                    batch_size,
                    config.num_kv_heads,
                    config.head_dim,
                    config.block_size,
                    max_seq_len,
                ) {
                    Ok(result) => {
                        let out_slice = cast_slice_mut::<T, half::f16>(output);
                        out_slice.copy_from_slice(&result);
                    }
                    Err(_) => CpuBackend.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
                }
            }
            _ => CpuBackend.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
        }
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        // Softmax uses ops implementation (compute-bound, not memory-bound)
        CpuBackend.softmax(input, output, config);
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        // Matmul uses linear kernel path for GPU acceleration
        CpuBackend.matmul(a, b, c, config);
    }

    // =========================================================================
    // GPU Tensor Operations
    // =========================================================================

    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        let kernel_lock = WGPU_LINEAR_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuLinear::new(ctx.device.clone(), ctx.queue.clone()))
        });
        let kernel = kernel_lock.as_ref().ok_or("WGPU context failed")?;

        let in_buf = match &input.buffer { GpuBuffer::Wgpu(b) => b, _ => return Err("Input not WGPU".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Wgpu(b) => b, _ => return Err("Weight not WGPU".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Wgpu(b) => b, _ => return Err("Output not WGPU".into()) };

        kernel.forward(params, in_buf, w_buf, None, out_buf);
        Ok(())
    }

    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        // For WGPU, linear + add is not fused; do linear then add in-place
        // First compute linear to a temp buffer, then add
        // For now, just do regular linear (fused add requires shader modification)
        self.linear_forward_gpu(input, weight, output, params)
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, input: &[T], weight: &GpuTensor, output: &mut [T], params: LinearParams) -> Result<(), String> {
        // Upload input, compute, readback
        let ctx = get_wgpu_context().ok_or("WGPU not init")?;
        let kernel_lock = WGPU_LINEAR_KERNEL.get_or_init(|| {
            Some(WgpuLinear::new(ctx.device.clone(), ctx.queue.clone()))
        });
        let kernel = kernel_lock.as_ref().ok_or("WGPU context failed")?;

        // Create input buffer
        let input_f32: Vec<f32> = input.iter().map(|&x| x.to_f32()).collect();
        let in_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("host_io_input"),
            contents: bytemuck::cast_slice(&input_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let w_buf = match &weight.buffer { GpuBuffer::Wgpu(b) => b, _ => return Err("Weight not WGPU".into()) };

        // Create output buffer
        let out_size = output.len() * std::mem::size_of::<f32>();
        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("host_io_output"),
            size: out_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        kernel.forward(params, &in_buf, w_buf, None, &out_buf);

        // Readback
        let mut output_f32 = vec![0.0f32; output.len()];
        kernel.readback_to_slice(&out_buf, &mut output_f32);
        for (i, &v) in output_f32.iter().enumerate() {
            output[i] = T::from_f32(v);
        }
        Ok(())
    }

    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        let kernel_lock = WGPU_LINEAR_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuLinear::new(ctx.device.clone(), ctx.queue.clone()))
        });
        let kernel = kernel_lock.as_ref().ok_or("WGPU context failed")?;

        let buf = match &gpu_tensor.buffer { GpuBuffer::Wgpu(b) => b, _ => return Err("Tensor not WGPU".into()) };
        kernel.readback_to_slice(buf, output);
        Ok(())
    }

    fn ffn_forward_gpu(&self, _: &GpuTensor, _: &GpuTensor, _: &GpuTensor, _: &GpuTensor, _: &mut GpuTensor, _: &mut GpuTensor, _: LinearParams, _: LinearParams) -> Result<(), String> {
        Err("FFN GPU not yet implemented for WGPU".into())
    }

    fn attention_forward_gpu(&self, _: &GpuTensor, _: &GpuBuffer, _: &GpuBuffer, _: &mut GpuTensor, _: FlashAttentionConfig) -> Result<(), String> {
        Err("Attention GPU not yet implemented for WGPU".into())
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        if input.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("WGPU rms_norm_gpu only supports f32 tensors".into());
        }

        let (rows, hidden) = rms_norm_dims(input, weight, Some(output))?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "RMSNorm rows exceeds u32".to_string())?;
        let hidden_u32 = u32::try_from(hidden).map_err(|_| "RMSNorm hidden exceeds u32".to_string())?;

        let kernel_lock = WGPU_RMS_NORM_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuRmsNorm::new(ctx.device.clone(), ctx.queue.clone()))
        });
        let kernel = kernel_lock.as_ref().ok_or("WGPU RMSNorm kernel unavailable")?;

        let in_buf = match &input.buffer { GpuBuffer::Wgpu(buf) => buf, _ => return Err("Input not WGPU".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Wgpu(buf) => buf, _ => return Err("Weight not WGPU".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Wgpu(buf) => buf, _ => return Err("Output not WGPU".into()) };

        let params = RmsNormParams {
            rows: rows_u32,
            hidden: hidden_u32,
            _pad0: 0,
            _pad1: 0,
            eps,
            _pad2: [0.0; 3],
        };

        kernel.forward(params, in_buf, w_buf, out_buf);
        Ok(())
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        if data.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 {
            return Err("WGPU rms_norm_gpu_inplace only supports f32 tensors".into());
        }

        let (rows, hidden) = rms_norm_dims(data, weight, None)?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "RMSNorm rows exceeds u32".to_string())?;
        let hidden_u32 = u32::try_from(hidden).map_err(|_| "RMSNorm hidden exceeds u32".to_string())?;

        let kernel_lock = WGPU_RMS_NORM_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuRmsNorm::new(ctx.device.clone(), ctx.queue.clone()))
        });
        let kernel = kernel_lock.as_ref().ok_or("WGPU RMSNorm kernel unavailable")?;

        let data_buf = match &data.buffer { GpuBuffer::Wgpu(buf) => buf, _ => return Err("Data not WGPU".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Wgpu(buf) => buf, _ => return Err("Weight not WGPU".into()) };

        let params = RmsNormParams {
            rows: rows_u32,
            hidden: hidden_u32,
            _pad0: 0,
            _pad1: 0,
            eps,
            _pad2: [0.0; 3],
        };

        kernel.forward_inplace(params, data_buf, w_buf);
        Ok(())
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        let ctx = get_wgpu_context().ok_or("WGPU not init")?;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Allocated Buffer"),
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(GpuBuffer::Wgpu(Arc::new(buffer)))
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        let ctx = get_wgpu_context().ok_or("WGPU not init")?;
        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pinned Weight"),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        Ok(GpuTensor::new(GpuBuffer::Wgpu(Arc::new(buffer)), shape, dtype, BackendType::Wgpu))
    }

    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        if let GpuBuffer::Wgpu(src) = &gpu.buffer {
            let ctx = get_wgpu_context().ok_or("WGPU not init")?;
            let size = (host.len() * std::mem::size_of::<T>()) as u64;
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
            ctx.queue.submit(Some(enc.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            ctx.device.poll(wgpu::PollType::Wait);

            if let Ok(Ok(_)) = rx.recv() {
                let view = slice.get_mapped_range();
                let data = unsafe { std::slice::from_raw_parts(view.as_ptr() as *const T, host.len()) };
                host.copy_from_slice(data);
                drop(view);
                staging.unmap();
                Ok(())
            } else {
                Err("Map async failed".into())
            }
        } else {
            Err("Not WGPU buffer".into())
        }
    }

    fn readback_u32(&self, gpu: &GpuTensor, host: &mut [u32]) -> Result<(), String> {
        if gpu.dtype != TensorDtype::U32 {
            return Err("WGPU readback_u32 requires u32 tensor".into());
        }
        let src = match &gpu.buffer {
            GpuBuffer::Wgpu(buf) => buf,
            _ => return Err("WGPU readback_u32: not WGPU buffer".into()),
        };
        let ctx = get_wgpu_context().ok_or("WGPU not init")?;
        let size = host.len() * std::mem::size_of::<u32>();
        if gpu.size_in_bytes < size {
            return Err(format!(
                "WGPU readback_u32: buffer bytes {} < required {}",
                gpu.size_in_bytes,
                size
            ));
        }
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging U32"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size as u64);
        ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        ctx.device.poll(wgpu::PollType::Wait);

        if let Ok(Ok(_)) = rx.recv() {
            let view = slice.get_mapped_range();
            let data = unsafe { std::slice::from_raw_parts(view.as_ptr() as *const u32, host.len()) };
            host.copy_from_slice(data);
            drop(view);
            staging.unmap();
            Ok(())
        } else {
            Err("Map async failed".into())
        }
    }

    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        if let GpuBuffer::Wgpu(dst) = &gpu.buffer {
            let ctx = get_wgpu_context().ok_or("WGPU not init")?;
            let bytes = unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * std::mem::size_of::<T>())
            };
            ctx.queue.write_buffer(dst, 0, bytes);
            Ok(())
        } else {
            Err("Not WGPU buffer".into())
        }
    }

    fn update_kv_cache_gpu(&self, cache: &mut GpuKVCache, layer_idx: usize, new_k: &GpuTensor, new_v: &GpuTensor) -> Result<(), String> {
        let ctx = get_wgpu_context().ok_or("WGPU not init")?;
        let stride = cache.batch_size * cache.num_heads * cache.head_dim;
        let seq_len = new_k.size_in_bytes / (stride * 4);
        let start_offset = (cache.current_len * stride * 4) as u64;
        let copy_size = new_k.size_in_bytes as u64;

        if let (GpuBuffer::Wgpu(dst_k), GpuBuffer::Wgpu(src_k)) = (&cache.keys[layer_idx], &new_k.buffer) {
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Update KV") });
            encoder.copy_buffer_to_buffer(src_k, 0, dst_k, start_offset, copy_size);
            if let (GpuBuffer::Wgpu(dst_v), GpuBuffer::Wgpu(src_v)) = (&cache.values[layer_idx], &new_v.buffer) {
                encoder.copy_buffer_to_buffer(src_v, 0, dst_v, start_offset, copy_size);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }
        if layer_idx == cache.num_layers - 1 { cache.current_len += seq_len; }
        Ok(())
    }

    // =========================================================================
    // Speculative Decoding (WGPU kernels with CPU fallback)
    // =========================================================================

    fn eagle3_confidence<T: KernelFloat>(&self, layers: &[&[T]], w: &[T], b: T, config: &Eagle3Config) -> Option<Vec<T>> {
        let kernel = match WGPU_EAGLE3_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            WgpuEagle3Kernel::new(&ctx.device, &ctx.queue).ok()
        }) {
            Some(k) => k,
            None => return CpuBackend.eagle3_confidence(layers, w, b, config),
        };

        if layers.len() < config.fusion_layers {
            return CpuBackend.eagle3_confidence(layers, w, b, config);
        }

        match T::TYPE_ID {
            FloatType::F32 => {
                // Flatten layers into single buffer
                let start = layers.len() - config.fusion_layers;
                let layer_data: Vec<f32> = layers[start..].iter()
                    .flat_map(|l| cast_slice::<T, f32>(*l).iter().copied())
                    .collect();

                let fused = match kernel.fuse_layers_f32(
                    &layer_data,
                    config.batch_size,
                    config.seq_len,
                    config.hidden_dim,
                    config.fusion_layers,
                ) {
                    Ok(f) => f,
                    Err(_) => return CpuBackend.eagle3_confidence(layers, w, b, config),
                };

                let w_f32 = cast_slice::<T, f32>(w);
                match kernel.predict_confidence_f32(
                    &fused,
                    w_f32,
                    config.batch_size,
                    config.seq_len,
                    config.fused_dim(),
                    b.to_f32(),
                ) {
                    Ok(conf) => Some(conf.into_iter().map(T::from_f32).collect()),
                    Err(_) => CpuBackend.eagle3_confidence(layers, w, b, config),
                }
            }
            FloatType::F16 => {
                let start = layers.len() - config.fusion_layers;
                let layer_data: Vec<half::f16> = layers[start..].iter()
                    .flat_map(|l| cast_slice::<T, half::f16>(*l).iter().copied())
                    .collect();

                let fused = match kernel.fuse_layers_f16(
                    &layer_data,
                    config.batch_size,
                    config.seq_len,
                    config.hidden_dim,
                    config.fusion_layers,
                ) {
                    Ok(f) => f,
                    Err(_) => return CpuBackend.eagle3_confidence(layers, w, b, config),
                };

                let w_f16 = cast_slice::<T, half::f16>(w);
                match kernel.predict_confidence_f16(
                    &fused,
                    w_f16,
                    config.batch_size,
                    config.seq_len,
                    config.fused_dim(),
                    b.to_f32(),
                ) {
                    Ok(conf) => Some(conf.into_iter().map(|x| T::from_f32(x.to_f32())).collect()),
                    Err(_) => CpuBackend.eagle3_confidence(layers, w, b, config),
                }
            }
            _ => CpuBackend.eagle3_confidence(layers, w, b, config),
        }
    }

    fn spec_ee_confidence<T: KernelFloat>(&self, hs: &[T], w: &[T], b: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        let kernel = match WGPU_SPEC_EE_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            WgpuSpecEEKernel::new(&ctx.device, &ctx.queue).ok()
        }) {
            Some(k) => k,
            None => return CpuBackend.spec_ee_confidence(hs, w, b, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                match kernel.compute_confidence_f32(
                    cast_slice(hs),
                    cast_slice(w),
                    config.batch_size,
                    config.seq_len,
                    config.hidden_dim,
                    config.current_layer,
                    config.exit_threshold,
                ) {
                    Ok(result) => Some(result.confidence.into_iter().map(T::from_f32).collect()),
                    Err(_) => CpuBackend.spec_ee_confidence(hs, w, b, config),
                }
            }
            _ => CpuBackend.spec_ee_confidence(hs, w, b, config),
        }
    }

    fn flash_tree_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], m: &[i32], o: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        let kernel = match WGPU_FLASH_TREE_KERNEL.get_or_init(|| {
            WgpuFlashTreeAttn::new_sync().ok()
        }) {
            Some(k) => k,
            None => return CpuBackend.flash_tree_attention(q, k, v, m, o, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                let tree_mask_f32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
                let params = crate::wgpu_kernels::flash_tree_attn::TreeAttnParams::new(
                    config.batch_size as u32,
                    config.num_heads as u32,
                    config.prefix_len as u32,
                    config.tree_size as u32,
                    config.head_dim as u32,
                    config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
                );
                let result = kernel.forward_f32(cast_slice(q), cast_slice(k), cast_slice(v), &tree_mask_f32, &params);
                let out_slice = cast_slice_mut::<T, f32>(o);
                out_slice.copy_from_slice(&result);
                true
            }
            _ => CpuBackend.flash_tree_attention(q, k, v, m, o, config),
        }
    }

    fn medusa_forward<T: KernelFloat>(&self, hl: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        let kernel = match WGPU_MEDUSA_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuMedusa::new(ctx.device.clone(), ctx.queue.clone()))
        }) {
            Some(k) => k,
            None => return CpuBackend.medusa_forward(hl, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                // head_logits shape: [batch_size, num_heads, vocab_size]
                let batch_size = config.batch_size as u32;
                let num_heads = config.num_heads as u32;
                let vocab_size = config.vocab_size as u32;
                let k = config.top_k as u32;
                let max_candidates = config.max_candidates as u32;

                // Process each head's logits to get top-K
                let head_size = vocab_size as usize;
                let logits_f32 = cast_slice::<T, f32>(hl);

                // Collect all top-K indices from all heads
                let mut all_top_indices: Vec<u32> = Vec::with_capacity(
                    config.batch_size * config.num_heads * config.top_k
                );
                let mut all_top_probs: Vec<f32> = Vec::with_capacity(
                    config.batch_size * config.num_heads * config.top_k
                );

                for head_idx in 0..config.num_heads {
                    // Get logits for this head: [batch_size, vocab_size]
                    let head_start = head_idx * config.batch_size * head_size;
                    let head_end = head_start + config.batch_size * head_size;
                    let head_logits = if head_end <= logits_f32.len() {
                        &logits_f32[head_start..head_end]
                    } else {
                        // Fallback for misaligned data
                        return CpuBackend.medusa_forward(hl, config);
                    };

                    // Apply temperature if needed
                    let scaled_logits: Vec<f32> = if config.temperature != 1.0 {
                        head_logits.iter().map(|&x| x / config.temperature).collect()
                    } else {
                        head_logits.to_vec()
                    };

                    // Get top-K for this head
                    let topk_result = kernel.top_k_f32(
                        &scaled_logits,
                        batch_size,
                        1,  // seq_len = 1 (last token position)
                        vocab_size,
                        k,
                    );

                    all_top_indices.extend(&topk_result.indices);
                    all_top_probs.extend(&topk_result.values);
                }

                // Build candidate tree from all head predictions
                let candidates = kernel.build_candidates(
                    &all_top_indices,
                    batch_size,
                    num_heads,
                    k,
                    max_candidates,
                );

                // Convert u32 tokens to i32 and counts to i32
                let candidate_tokens: Vec<i32> = candidates.candidates
                    .into_iter()
                    .map(|t| t as i32)
                    .collect();
                let num_candidates: Vec<i32> = candidates.counts
                    .into_iter()
                    .map(|c| c as i32)
                    .collect();

                // Candidate probabilities (use mean of top-K probs for each batch)
                let probs_per_batch = all_top_probs.chunks(config.num_heads * config.top_k)
                    .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                    .collect();

                Some(MedusaForwardResult {
                    candidate_tokens,
                    candidate_probs: probs_per_batch,
                    num_candidates,
                })
            }
            _ => CpuBackend.medusa_forward(hl, config),
        }
    }

    fn medusa_verify<T: KernelFloat>(&self, ct: &[i32], tl: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        CpuBackend.medusa_verify(ct, tl, config)
    }

    // =========================================================================
    // Quantization (WGPU kernels with CPU fallback)
    // =========================================================================

    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        let kernel = match WGPU_INT2_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuInt2Quantizer::new(ctx.device.clone(), ctx.queue.clone()))
        }) {
            Some(k) => k,
            None => return CpuBackend.int2_quantize(input, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                let (quantized, scales, zeros) = kernel.quantize_f32(cast_slice(input), config.group_size as u32);
                // Convert u32 packed to i8 representation
                let quantized_i8: Vec<i8> = quantized.iter()
                    .flat_map(|&v| [(v & 0xFF) as i8, ((v >> 8) & 0xFF) as i8, ((v >> 16) & 0xFF) as i8, ((v >> 24) & 0xFF) as i8])
                    .take(input.len())
                    .collect();
                Some(Int2QuantResult {
                    quantized: quantized_i8,
                    scales,
                    zeros,
                })
            }
            _ => CpuBackend.int2_quantize(input, config),
        }
    }

    fn int2_dequantize<T: KernelFloat>(&self, q: &[i8], s: &[T], z: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        let kernel = match WGPU_INT2_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuInt2Quantizer::new(ctx.device.clone(), ctx.queue.clone()))
        }) {
            Some(k) => k,
            None => return CpuBackend.int2_dequantize(q, s, z, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                // Pack i8 to u32
                let quantized_u32: Vec<u32> = q.chunks(4)
                    .map(|chunk| {
                        let mut v = 0u32;
                        for (i, &b) in chunk.iter().enumerate() {
                            v |= ((b as u8) as u32) << (i * 8);
                        }
                        v
                    })
                    .collect();
                let result = kernel.dequantize_f32(&quantized_u32, cast_slice(s), cast_slice(z), config.group_size as u32);
                Some(result.into_iter().map(T::from_f32).collect())
            }
            _ => CpuBackend.int2_dequantize(q, s, z, config),
        }
    }

    // =========================================================================
    // KV Cache Compression (WGPU kernels with CPU fallback)
    // =========================================================================

    fn evic_press_compress<T: KernelFloat>(&self, kv: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        // EvicPress compression requires complex zone management
        // TODO: Full WGPU implementation
        CpuBackend.evic_press_compress(kv, config)
    }

    fn evic_press_evict<T: KernelFloat>(&self, w: &[T], ages: &[i32], zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        let kernel = match WGPU_EVICT_PRESS_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuEvicPress::new(ctx.device.clone(), ctx.queue.clone()))
        }) {
            Some(k) => k,
            None => return CpuBackend.evic_press_evict(w, ages, zones, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                // Compute positions from ages (older = higher position index)
                let positions: Vec<u32> = ages.iter().map(|&a| a.max(0) as u32).collect();

                // Derive head_dim from attention weights shape:
                // w is [batch_size * num_heads * seq_len] attention weights
                let total_elements = w.len();
                let expected_per_head = config.batch_size * config.seq_len;
                let head_dim = if expected_per_head > 0 && total_elements >= expected_per_head {
                    total_elements / (config.batch_size * config.num_heads * config.seq_len)
                } else {
                    64 // default head_dim
                };

                // semantic_weight defaults to balance between attention and recency
                let semantic_weight = 1.0 - config.attention_weight - config.recency_weight;

                let result = kernel.compute_importance_f32(
                    cast_slice(w),
                    &positions,
                    config.batch_size as u32,
                    config.num_heads as u32,
                    config.seq_len as u32,
                    head_dim as u32,
                    (config.attention_weight, semantic_weight.max(0.0), config.recency_weight),
                );

                // Compute new zones based on importance scores and thresholds
                let new_zones: Vec<i32> = result.scores.iter().zip(zones.iter())
                    .map(|(&score, &current_zone)| {
                        if score >= config.hot_threshold {
                            0 // hot zone
                        } else if score >= config.warm_threshold {
                            1 // warm zone
                        } else if config.cache_pressure > 0.8 && current_zone >= 1 {
                            2 // cold zone (eligible for eviction under pressure)
                        } else {
                            current_zone.max(1) // stay in current zone or warm
                        }
                    })
                    .collect();

                Some(EvicPressEvictResult {
                    importance: result.scores,
                    new_zones,
                })
            }
            _ => CpuBackend.evic_press_evict(w, ages, zones, config),
        }
    }

    // =========================================================================
    // Prompt Caching (WGPU kernels with CPU fallback)
    // =========================================================================

    fn prompt_cache_lookup(&self, t: &[i32], h: &[u64], l: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        // Prompt cache lookup uses hash matching - CPU is efficient
        CpuBackend.prompt_cache_lookup(t, h, l, config)
    }

    fn prompt_cache_blend<T: KernelFloat>(&self, c: &[T], f: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        let kernel = match WGPU_PROMPT_CACHE_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuPromptCache::new(ctx.device.clone(), ctx.queue.clone()))
        }) {
            Some(k) => k,
            None => return CpuBackend.prompt_cache_blend(c, f, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                // batch_size = 1 for prompt cache blend (per-sequence operation)
                let blended = kernel.blend_kv_f32(
                    cast_slice(c),
                    cast_slice(f),
                    1,  // batch_size
                    config.num_heads as u32,
                    config.head_dim as u32,
                    config.match_len as u32,
                    config.fresh_len as u32,
                );
                Some(blended.into_iter().map(T::from_f32).collect())
            }
            _ => CpuBackend.prompt_cache_blend(c, f, config),
        }
    }

    // =========================================================================
    // Chunked Prefill (WGPU kernels with CPU fallback)
    // =========================================================================

    fn chunked_prefill_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        let kernel = match WGPU_CHUNKED_PREFILL_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuChunkedPrefill::new(ctx.device.clone(), ctx.queue.clone()))
        }) {
            Some(k) => k,
            None => return CpuBackend.chunked_prefill_attention(q, k, v, config),
        };

        match T::TYPE_ID {
            FloatType::F32 => {
                let result = kernel.chunked_attention_f32(
                    cast_slice(q),
                    cast_slice(k),
                    cast_slice(v),
                    config.batch_size as u32,
                    config.num_heads as u32,
                    config.head_dim as u32,
                    config.chunk_len as u32,
                    config.query_len as u32,
                    (config.chunk_start / config.chunk_len) as u32,
                );
                Some(ChunkedPrefillResult {
                    output: result.output.into_iter().map(T::from_f32).collect(),
                    log_sum_exp: result.lse,
                })
            }
            _ => CpuBackend.chunked_prefill_attention(q, k, v, config),
        }
    }

    // =========================================================================
    // Embedding Operations (WGPU kernels with CPU fallback)
    // =========================================================================

    fn rerank_pipeline(&self, bq: &[u32], bd: &[u32], iq: &[u32], id: &[u32], nv: usize, config: &GpuRerankConfig, is: f32) -> Result<GpuRerankStageResult, String> {
        let kernel = match WGPU_EMBEDDING_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            WgpuEmbeddingOpsKernel::new(&ctx.device, &ctx.queue).ok()
        }) {
            Some(k) => k,
            None => return CpuBackend.rerank_pipeline(bq, bd, iq, id, nv, config, is),
        };

        kernel.rerank_pipeline(bq, bd, iq, id, nv, config, is)
            .map_err(|e| format!("WGPU rerank error: {:?}", e))
    }

    fn binary_ip_hamming(&self, q: &[u64], d: &[u64], s: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        // WGPU kernel uses u32, but trait uses u64 - use CPU for type compatibility
        // (u64 -> u32 conversion would lose data)
        CpuBackend.binary_ip_hamming(q, d, s, config);
    }

    fn binary_ip_asymmetric(&self, q: &[f32], d: &[u64], s: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        // WGPU kernel uses u32 for binary data - use CPU for type compatibility
        CpuBackend.binary_ip_asymmetric(q, d, s, config);
    }

    fn int8_dot_product(&self, q: &[i8], d: &[i8], s: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        // WGPU kernel uses packed u32 (i8x4) - use CPU for type compatibility
        CpuBackend.int8_dot_product(q, d, s, config);
    }

    fn int4_packed_dot_product(&self, q: &[u8], d: &[u8], s: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        // WGPU kernel has different packed format - use CPU for type compatibility
        CpuBackend.int4_packed_dot_product(q, d, s, config);
    }

    // =========================================================================
    // MoE GPU Operations (Pure GPU compute shader - no readback)
    // =========================================================================

    fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String> {
        let ctx = get_wgpu_context().ok_or("WGPU not init")?;
        let buf = match &tensor.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_zero_gpu: expected WGPU buffer".into()),
        };
        let ops = WGPU_TENSOR_OPS.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuTensorOps::new(ctx.device.clone(), ctx.queue.clone()))
        }).as_ref().ok_or("WGPU TensorOps init failed")?;
        ops.tensor_zero(buf, tensor.len());
        Ok(())
    }

    fn tensor_add_gpu(&self, output: &mut GpuTensor, input: &GpuTensor) -> Result<(), String> {
        let in_buf = match &input.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_add_gpu: input not WGPU buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_add_gpu: output not WGPU buffer".into()),
        };
        let ops = WGPU_TENSOR_OPS.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuTensorOps::new(ctx.device.clone(), ctx.queue.clone()))
        }).as_ref().ok_or("WGPU TensorOps init failed")?;
        let size = input.len().min(output.len());
        ops.tensor_add(out_buf, in_buf, size);
        Ok(())
    }

    fn tensor_slice_gpu(&self, input: &GpuTensor, offset: usize, len: usize, output: &mut GpuTensor) -> Result<(), String> {
        let in_buf = match &input.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_slice_gpu: input not WGPU buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_slice_gpu: output not WGPU buffer".into()),
        };
        if offset + len > input.len() {
            return Err(format!("WGPU tensor_slice_gpu: offset {} + len {} > input len {}", offset, len, input.len()));
        }
        let ops = WGPU_TENSOR_OPS.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuTensorOps::new(ctx.device.clone(), ctx.queue.clone()))
        }).as_ref().ok_or("WGPU TensorOps init failed")?;
        ops.tensor_slice(out_buf, in_buf, offset, len);
        Ok(())
    }

    fn tensor_scale_add_gpu(&self, input: &GpuTensor, output: &mut GpuTensor, offset: usize, scale: f32) -> Result<(), String> {
        let in_buf = match &input.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_scale_add_gpu: input not WGPU buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU tensor_scale_add_gpu: output not WGPU buffer".into()),
        };
        if offset + input.len() > output.len() {
            return Err(format!("WGPU tensor_scale_add_gpu: offset {} + input len {} > output len {}", offset, input.len(), output.len()));
        }
        let ops = WGPU_TENSOR_OPS.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuTensorOps::new(ctx.device.clone(), ctx.queue.clone()))
        }).as_ref().ok_or("WGPU TensorOps init failed")?;
        ops.tensor_scale_add(out_buf, in_buf, offset, input.len(), scale);
        Ok(())
    }

    fn moe_route_gpu(
        &self,
        hidden_states: &GpuTensor,
        gate_weights: &GpuTensor,
        expert_indices_out: &mut GpuTensor,
        expert_weights_out: &mut GpuTensor,
        config: MoERoutingGpuConfig,
    ) -> Result<(), String> {
        let ctx = get_wgpu_context()
            .ok_or("WGPU context unavailable for moe_route_gpu")?;

        if config.num_tokens == 0 || config.hidden_size == 0 || config.num_experts == 0 || config.top_k == 0 {
            return Err("WGPU moe_route_gpu: config values must be > 0".into());
        }
        if config.top_k > config.num_experts {
            return Err(format!(
                "WGPU moe_route_gpu: top_k {} exceeds num_experts {}",
                config.top_k, config.num_experts
            ));
        }
        if config.top_k > crate::wgpu_kernels::moe_routing_gpu::MOE_ROUTING_MAX_TOPK as usize {
            return Err(format!(
                "WGPU moe_route_gpu: top_k {} exceeds max supported {}",
                config.top_k,
                crate::wgpu_kernels::moe_routing_gpu::MOE_ROUTING_MAX_TOPK
            ));
        }
        if hidden_states.dtype != TensorDtype::F32 || gate_weights.dtype != TensorDtype::F32 {
            return Err("WGPU moe_route_gpu: hidden_states and gate_weights must be f32 tensors".into());
        }
        if expert_indices_out.dtype != TensorDtype::U32 {
            return Err("WGPU moe_route_gpu: expert_indices_out must be u32 tensor".into());
        }
        if expert_weights_out.dtype != TensorDtype::F32 {
            return Err("WGPU moe_route_gpu: expert_weights_out must be f32 tensor".into());
        }

        let expected_hidden = config.num_tokens * config.hidden_size;
        if hidden_states.len() < expected_hidden {
            return Err(format!(
                "WGPU moe_route_gpu: hidden_states len {} < expected {}",
                hidden_states.len(),
                expected_hidden
            ));
        }
        let expected_gate = config.hidden_size * config.num_experts;
        if gate_weights.len() < expected_gate {
            return Err(format!(
                "WGPU moe_route_gpu: gate_weights len {} < expected {}",
                gate_weights.len(),
                expected_gate
            ));
        }
        let expected_out = config.num_tokens * config.top_k;
        let expected_bytes = expected_out * 4;
        if expert_indices_out.size_in_bytes < expected_bytes {
            return Err(format!(
                "WGPU moe_route_gpu: expert_indices_out bytes {} < expected {}",
                expert_indices_out.size_in_bytes,
                expected_bytes
            ));
        }
        if expert_weights_out.size_in_bytes < expected_bytes {
            return Err(format!(
                "WGPU moe_route_gpu: expert_weights_out bytes {} < expected {}",
                expert_weights_out.size_in_bytes,
                expected_bytes
            ));
        }

        let hidden_buf = match &hidden_states.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_route_gpu: hidden_states not WGPU buffer".into()),
        };
        let gate_buf = match &gate_weights.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_route_gpu: gate_weights not WGPU buffer".into()),
        };
        let indices_buf = match &expert_indices_out.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_route_gpu: expert_indices_out not WGPU buffer".into()),
        };
        let weights_buf = match &expert_weights_out.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_route_gpu: expert_weights_out not WGPU buffer".into()),
        };

        let params = MoERoutingGpuParams {
            num_tokens: u32::try_from(config.num_tokens)
                .map_err(|_| "WGPU moe_route_gpu: num_tokens exceeds u32".to_string())?,
            hidden_size: u32::try_from(config.hidden_size)
                .map_err(|_| "WGPU moe_route_gpu: hidden_size exceeds u32".to_string())?,
            num_experts: u32::try_from(config.num_experts)
                .map_err(|_| "WGPU moe_route_gpu: num_experts exceeds u32".to_string())?,
            top_k: u32::try_from(config.top_k)
                .map_err(|_| "WGPU moe_route_gpu: top_k exceeds u32".to_string())?,
        };

        let kernel = WGPU_MOE_ROUTING_KERNEL.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuMoERouting::new(ctx.device.clone(), ctx.queue.clone()))
        }).as_ref().ok_or("WGPU MoE routing kernel init failed")?;

        kernel.forward(params, hidden_buf, gate_buf, indices_buf, weights_buf);
        Ok(())
    }

    fn moe_forward_gpu_pure(
        &self,
        input: &GpuTensor,
        expert_indices: &GpuTensor,
        expert_weights: &GpuTensor,
        all_gate_weights: &GpuTensor,
        all_up_weights: &GpuTensor,
        all_down_weights: &GpuTensor,
        output: &mut GpuTensor,
        config: MoEForwardConfig,
    ) -> Result<(), String> {
        let ctx = get_wgpu_context()
            .ok_or("WGPU context unavailable for moe_forward_gpu_pure")?;

        if expert_indices.dtype != TensorDtype::U32 {
            return Err("WGPU moe_forward_gpu_pure: expert_indices must be u32 tensor".into());
        }
        if expert_weights.dtype != TensorDtype::F32 {
            return Err("WGPU moe_forward_gpu_pure: expert_weights must be f32 tensor".into());
        }

        let in_buf = match &input.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: input not WGPU buffer".into()),
        };
        let indices_buf = match &expert_indices.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: expert_indices not WGPU buffer".into()),
        };
        let weights_buf = match &expert_weights.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: expert_weights not WGPU buffer".into()),
        };
        let gate_buf = match &all_gate_weights.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: gate_weights not WGPU buffer".into()),
        };
        let up_buf = match &all_up_weights.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: up_weights not WGPU buffer".into()),
        };
        let down_buf = match &all_down_weights.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: down_weights not WGPU buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Wgpu(b) => b,
            _ => return Err("WGPU moe_forward_gpu_pure: output not WGPU buffer".into()),
        };

        let expected_routing = config.num_tokens * config.top_k;
        let expected_index_bytes = expected_routing * std::mem::size_of::<u32>();
        let expected_weight_bytes = expected_routing * std::mem::size_of::<f32>();
        if expert_indices.size_in_bytes < expected_index_bytes {
            return Err(format!(
                "WGPU moe_forward_gpu_pure: expert_indices bytes {} < expected {}",
                expert_indices.size_in_bytes,
                expected_index_bytes
            ));
        }
        if expert_weights.size_in_bytes < expected_weight_bytes {
            return Err(format!(
                "WGPU moe_forward_gpu_pure: expert_weights bytes {} < expected {}",
                expert_weights.size_in_bytes,
                expected_weight_bytes
            ));
        }

        // Scratch space for intermediate computations
        let scratch_size = config.num_tokens * config.top_k * config.intermediate_size * 2 * 4;
        let scratch_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MoE FFN Scratch"),
            size: scratch_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Initialize MoE FFN kernel (lazy)
        static MOE_FFN: OnceLock<Option<WgpuMoeFfn>> = OnceLock::new();
        let moe_kernel = MOE_FFN.get_or_init(|| {
            let ctx = get_wgpu_context()?;
            Some(WgpuMoeFfn::new(ctx.device.clone(), ctx.queue.clone()))
        }).as_ref().ok_or("WGPU MoE FFN kernel init failed")?;

        let params = MoEFfnParams {
            hidden_size: config.hidden_size as u32,
            intermediate_size: config.intermediate_size as u32,
            num_tokens: config.num_tokens as u32,
            top_k: config.top_k as u32,
            num_experts: config.num_experts as u32,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
        };

        moe_kernel.forward(
            in_buf,
            indices_buf,
            weights_buf,
            gate_buf,
            up_buf,
            down_buf,
            out_buf,
            &scratch_buf,
            params,
        );

        Ok(())
    }

    fn backend_type(&self) -> BackendType { BackendType::Wgpu }
}

// =============================================================================
// CUDA Backend Implementation
// =============================================================================
//
// CUDA backend with method-level generics.
// Uses T::TYPE_ID const branch to dispatch to appropriate kernel variant.
// Compiler eliminates unused branches at monomorphization.
// =============================================================================

#[derive(Clone, Default)]
pub struct CudaBackend;

impl CudaBackend {
    // =========================================================================
    // Core Attention Operations (Generic with const dispatch)
    // =========================================================================

    /// Host-slice based flash attention - falls back to CPU implementation.
    /// For actual CUDA acceleration, use `attention_forward_gpu` with GpuTensor.
    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        // Host slice operations fall back to CPU (CUDA acceleration requires device memory)
        // Real GPU acceleration goes through attention_forward_gpu with GpuTensor
        CpuBackend.flash_attention(q, k, v, output, config);
    }

    /// Host-slice based paged attention - falls back to CPU implementation.
    /// For actual CUDA acceleration, use the GPU tensor path.
    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        // Host slice operations fall back to CPU (CUDA acceleration requires device memory)
        CpuBackend.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        ops::softmax::softmax(input, output, config);
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        ops::matmul(a, b, c, config);
    }

    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let kernel = init_cuda_kernel(&CUDA_LINEAR_KERNEL, CudaLinear::new)
            .ok_or("CUDA linear kernel unavailable")?;
        let in_buf = match &input.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA linear input buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA linear weight buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA linear output buffer mismatch".into()) };
        kernel
            .forward(stream.as_ref(), params, in_buf.as_ref(), w_buf.as_ref(), None, out_buf.as_ref())
            .map_err(|err| format!("CUDA linear forward failed: {err}"))
    }

    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        // For now, just do linear (fused add not implemented)
        self.linear_forward_gpu(input, weight, output, params)
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, _input: &[T], _weight: &GpuTensor, _output: &mut [T], _params: LinearParams) -> Result<(), String> {
        Err("CUDA linear_forward_host_io not yet implemented".into())
    }

    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        let buf = match &gpu_tensor.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("Tensor not CUDA".into()) };
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let host_bytes: Vec<u8> = cuda_download(stream, buf.as_ref())?;
        // Convert bytes to f32
        let host_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                host_bytes.as_ptr() as *const f32,
                host_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        if host_data.len() != output.len() {
            return Err(format!("CUDA readback size mismatch: {} vs {}", host_data.len(), output.len()));
        }
        output.copy_from_slice(host_data);
        Ok(())
    }

    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let kernel = init_cuda_kernel(&CUDA_LINEAR_KERNEL, CudaLinear::new)
            .ok_or("CUDA linear kernel unavailable")?;
        let batch_size = input.shape.first().copied().unwrap_or(1);
        let in_buf = match &input.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA FFN input buffer mismatch".into()) };
        let gate_buf = match &gate.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA FFN gate buffer mismatch".into()) };
        let up_buf = match &up.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA FFN up buffer mismatch".into()) };
        let down_buf = match &down.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA FFN down buffer mismatch".into()) };
        let inter_buf = match &intermediate.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA FFN intermediate buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA FFN output buffer mismatch".into()) };
        kernel
            .fused_gate_up_silu(
                stream.as_ref(),
                gate_up_params,
                in_buf.as_ref(),
                gate_buf.as_ref(),
                up_buf.as_ref(),
                inter_buf.as_ref(),
                batch_size,
            )
            .map_err(|err| format!("CUDA FFN gate/up failed: {err}"))?;
        kernel
            .forward(stream.as_ref(), down_params, inter_buf.as_ref(), down_buf.as_ref(), None, out_buf.as_ref())
            .map_err(|err| format!("CUDA FFN down projection failed: {err}"))
    }

    fn attention_forward_gpu(&self, _: &GpuTensor, _: &GpuBuffer, _: &GpuBuffer, _: &mut GpuTensor, _: FlashAttentionConfig) -> Result<(), String> {
        Err("CUDA attention_forward_gpu not wired in backend.rs".into())
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        if input.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("CUDA rms_norm_gpu only supports f32 tensors".into());
        }
        let (rows, hidden) = rms_norm_dims(input, weight, Some(output))?;
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let kernel = init_cuda_kernel(&CUDA_RMS_NORM_KERNEL, CudaRmsNormKernel::new)
            .ok_or("CUDA RMSNorm kernel unavailable")?;
        let in_buf = match &input.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA rms_norm_gpu input buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA rms_norm_gpu weight buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA rms_norm_gpu output buffer mismatch".into()) };
        kernel
            .forward(stream, in_buf.as_ref(), w_buf.as_ref(), out_buf.as_ref(), rows, hidden, eps)
            .map_err(|err| format!("CUDA RMSNorm failed: {err}"))
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        if data.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 {
            return Err("CUDA rms_norm_gpu_inplace only supports f32 tensors".into());
        }
        let (rows, hidden) = rms_norm_dims(data, weight, None)?;
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let kernel = init_cuda_kernel(&CUDA_RMS_NORM_KERNEL, CudaRmsNormKernel::new)
            .ok_or("CUDA RMSNorm kernel unavailable")?;
        let data_buf = match &data.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA rms_norm_gpu_inplace buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA rms_norm_gpu_inplace weight buffer mismatch".into()) };
        kernel
            .forward(stream, data_buf.as_ref(), w_buf.as_ref(), data_buf.as_ref(), rows, hidden, eps)
            .map_err(|err| format!("CUDA RMSNorm inplace failed: {err}"))
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let buffer: CudaSlice<u8> = stream
            .alloc_zeros(size_bytes)
            .map_err(|err| format!("CUDA buffer allocation failed: {err}"))?;
        Ok(GpuBuffer::Cuda(Arc::new(buffer)))
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let buffer = cuda_upload(stream, data)?;
        Ok(GpuTensor::new(GpuBuffer::Cuda(Arc::new(buffer)), shape, dtype, BackendType::Cuda))
    }

    #[inline(always)]
    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let buf = match &gpu.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA readback buffer mismatch".into()) };
        let bytes = cuda_download(stream, buf.as_ref())?;
        match T::TYPE_ID {
            FloatType::F32 => {
                let values: &[f32] = bytemuck::cast_slice(&bytes);
                let host_f32 = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut f32, host.len()) };
                if values.len() != host_f32.len() {
                    return Err("CUDA readback length mismatch".into());
                }
                host_f32.copy_from_slice(values);
            }
            FloatType::F16 => {
                let values: &[u16] = bytemuck::cast_slice(&bytes);
                let host_f16 = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut half::f16, host.len()) };
                if values.len() != host_f16.len() {
                    return Err("CUDA readback f16 length mismatch".into());
                }
                for (h, v) in host_f16.iter_mut().zip(values.iter()) {
                    *h = half::f16::from_bits(*v);
                }
            }
            FloatType::BF16 => {
                let values: &[f32] = bytemuck::cast_slice(&bytes);
                if values.len() != host.len() {
                    return Err("CUDA readback bf16 length mismatch".into());
                }
                for (h, v) in host.iter_mut().zip(values.iter()) {
                    *h = T::from_f32(*v);
                }
            }
        }
        Ok(())
    }

    fn readback_u32(&self, _: &GpuTensor, _: &mut [u32]) -> Result<(), String> {
        Err("CUDA readback_u32 not implemented".into())
    }

    #[inline(always)]
    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        match T::TYPE_ID {
            FloatType::F32 => {
                let host_f32 = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const f32, host.len()) };
                let bytes: &[u8] = bytemuck::cast_slice(host_f32);
                let buffer = cuda_upload(stream, bytes)?;
                gpu.buffer = GpuBuffer::Cuda(Arc::new(buffer));
            }
            FloatType::F16 => {
                let host_f16 = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const half::f16, host.len()) };
                let u16_vec: Vec<u16> = host_f16.iter().map(|x| x.to_bits()).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&u16_vec);
                let buffer = cuda_upload(stream, bytes)?;
                gpu.buffer = GpuBuffer::Cuda(Arc::new(buffer));
            }
            FloatType::BF16 => {
                let f32_vec: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&f32_vec);
                let buffer = cuda_upload(stream, bytes)?;
                gpu.buffer = GpuBuffer::Cuda(Arc::new(buffer));
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn eagle3_confidence<T: KernelFloat>(&self, layer_hidden_states: &[&[T]], confidence_weights: &[T], confidence_bias: T, config: &Eagle3Config) -> Option<Vec<T>> {
        // Convert to f32 for GPU kernel, then convert back
        let layer_f32: Vec<Vec<f32>> = layer_hidden_states.iter()
            .map(|layer| layer.iter().map(|x| x.to_f32()).collect())
            .collect();
        let layer_refs: Vec<&[f32]> = layer_f32.iter().map(|v| v.as_slice()).collect();
        let weights_f32: Vec<f32> = confidence_weights.iter().map(|x| x.to_f32()).collect();
        let bias_f32 = confidence_bias.to_f32();

        let result = (|| -> Result<Vec<f32>, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EAGLE3_KERNEL, CudaEagle3Kernel::new)
                .ok_or("CUDA eagle3 kernel unavailable")?;
            if layer_refs.len() < config.fusion_layers {
                return Err("CUDA eagle3 layer count mismatch".into());
            }
            let start = layer_refs.len() - config.fusion_layers;
            let mut layer_buffers = Vec::with_capacity(config.fusion_layers);
            for layer in &layer_refs[start..] {
                layer_buffers.push(cuda_upload(stream, *layer)?);
            }
            let buffer_refs: Vec<&CudaSlice<f32>> = layer_buffers.iter().collect();
            let fused = kernel
                .fuse_hidden_states_f32(
                    stream,
                    &buffer_refs,
                    config.batch_size,
                    config.seq_len,
                    config.hidden_dim,
                    config.fusion_layers,
                )
                .map_err(|err| format!("CUDA eagle3 fuse failed: {err}"))?;
            let weights_dev = cuda_upload(stream, &weights_f32)?;
            let confidence = kernel
                .predict_confidence_f32(
                    stream,
                    &fused,
                    &weights_dev,
                    bias_f32,
                    config.batch_size,
                    config.seq_len,
                    config.fused_dim(),
                )
                .map_err(|err| format!("CUDA eagle3 confidence failed: {err}"))?;
            cuda_download(stream, &confidence)
        })();
        match result {
            Ok(conf) => Some(conf.into_iter().map(|x| T::from_f32(x)).collect()),
            Err(err) => {
                log::warn!("CUDA eagle3_confidence fallback: {err}");
                Some(ops::eagle3::fused_confidence_predict(&layer_refs, &weights_f32, bias_f32, config)
                    .into_iter().map(|x| T::from_f32(x)).collect())
            }
        }
    }

    #[inline(always)]
    fn spec_ee_confidence<T: KernelFloat>(&self, hidden_states: &[T], classifier_weight: &[T], classifier_bias: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        let hidden_f32: Vec<f32> = hidden_states.iter().map(|x| x.to_f32()).collect();
        let weight_f32: Vec<f32> = classifier_weight.iter().map(|x| x.to_f32()).collect();
        let bias_f32 = classifier_bias.to_f32();

        let result = (|| -> Result<Vec<f32>, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_SPEC_EE_KERNEL, CudaSpecEEKernel::new)
                .ok_or("CUDA spec-ee kernel unavailable")?;
            let hidden_dev = cuda_upload(stream, &hidden_f32)?;
            let weight_dev = cuda_upload(stream, &weight_f32)?;
            let confidence = kernel
                .compute_confidence_f32(
                    stream,
                    &hidden_dev,
                    &weight_dev,
                    bias_f32,
                    config.batch_size,
                    config.seq_len,
                    config.hidden_dim,
                )
                .map_err(|err| format!("CUDA spec-ee confidence failed: {err}"))?;
            cuda_download(stream, &confidence)
        })();
        match result {
            Ok(conf) => Some(conf.into_iter().map(|x| T::from_f32(x)).collect()),
            Err(err) => {
                log::warn!("CUDA spec_ee_confidence fallback: {err}");
                Some(ops::spec_ee::compute_confidence_stateless(&hidden_f32, &weight_f32, bias_f32, config)
                    .into_iter().map(|x| T::from_f32(x)).collect())
            }
        }
    }

    #[inline(always)]
    fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        let q_f32: Vec<f32> = query.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = key.iter().map(|x| x.to_f32()).collect();
        let v_f32: Vec<f32> = value.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<Vec<f32>, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_FLASH_TREE_KERNEL, CudaFlashTreeAttnKernel::new)
                .ok_or("CUDA flash-tree kernel unavailable")?;
            let q_dev = cuda_upload(stream, &q_f32)?;
            let k_dev = cuda_upload(stream, &k_f32)?;
            let v_dev = cuda_upload(stream, &v_f32)?;
            let mask_dev = cuda_upload(stream, tree_mask)?;
            let out_dev = kernel
                .tree_attention_f32(
                    stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mask_dev,
                    config.batch_size,
                    config.num_heads,
                    config.tree_size,
                    config.prefix_len,
                    config.head_dim,
                )
                .map_err(|err| format!("CUDA flash-tree attention failed: {err}"))?;
            cuda_download(stream, &out_dev)
        })();
        match result {
            Ok(out_f32) => {
                for (o, v) in output.iter_mut().zip(out_f32.iter()) {
                    *o = T::from_f32(*v);
                }
                true
            }
            Err(err) => {
                log::warn!("CUDA flash_tree_attention fallback: {err}");
                let mut out_f32 = vec![0.0f32; output.len()];
                ops::flash_tree_attn::flash_tree_attention(&q_f32, &k_f32, &v_f32, tree_mask, &mut out_f32, config);
                for (o, v) in output.iter_mut().zip(out_f32.iter()) {
                    *o = T::from_f32(*v);
                }
                true
            }
        }
    }

    #[inline(always)]
    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        let input_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<Int2QuantResult, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_INT2_KERNEL, CudaInt2QuantizerKernel::new)
                .ok_or("CUDA int2 kernel unavailable")?;
            let input_dev = cuda_upload(stream, &input_f32)?;
            let (quantized, scales, zeros) = kernel
                .quantize_f32(stream, &input_dev, config.group_size, input_f32.len())
                .map_err(|err| format!("CUDA int2 quantize failed: {err}"))?;
            let quantized_host = cuda_download(stream, &quantized)?;
            let scales_host = cuda_download(stream, &scales)?;
            let zeros_host = cuda_download(stream, &zeros)?;
            Ok(Int2QuantResult {
                quantized: quantized_host,
                scales: scales_host,
                zeros: zeros_host,
            })
        })();
        match result {
            Ok(res) => Some(res),
            Err(err) => {
                log::warn!("CUDA int2_quantize fallback: {err}");
                ops::int2_quantizer::int2_quantize(&input_f32, config).ok()
            }
        }
    }

    #[inline(always)]
    fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        let scales_f32: Vec<f32> = scales.iter().map(|x| x.to_f32()).collect();
        let zeros_f32: Vec<f32> = zeros.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<Vec<f32>, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_INT2_KERNEL, CudaInt2QuantizerKernel::new)
                .ok_or("CUDA int2 kernel unavailable")?;
            let quant_dev = cuda_upload(stream, quantized)?;
            let scales_dev = cuda_upload(stream, &scales_f32)?;
            let zeros_dev = cuda_upload(stream, &zeros_f32)?;
            let output = kernel
                .dequantize_f32(stream, &quant_dev, &scales_dev, &zeros_dev, quantized.len(), config.group_size)
                .map_err(|err| format!("CUDA int2 dequantize failed: {err}"))?;
            cuda_download(stream, &output)
        })();
        match result {
            Ok(res) => Some(res.into_iter().map(|x| T::from_f32(x)).collect()),
            Err(err) => {
                log::warn!("CUDA int2_dequantize fallback: {err}");
                ops::int2_quantizer::int2_dequantize(quantized, &scales_f32, &zeros_f32, config)
                    .ok()
                    .map(|v| v.into_iter().map(|x| T::from_f32(x)).collect())
            }
        }
    }

    #[inline(always)]
    fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        let kv_f32: Vec<f32> = kv_cache.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<EvicPressCompressionResult, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EVICT_PRESS_KERNEL, CudaEvicPressKernel::new)
                .ok_or("CUDA evic-press kernel unavailable")?;
            let kv_dev = cuda_upload(stream, &kv_f32)?;
            match config.compression {
                EvicPressCompression::Int8 => {
                    let (compressed, scales) = kernel
                        .compress_to_int8_f32(stream, &kv_dev, config.seq_len, config.head_dim)
                        .map_err(|err| format!("CUDA evic-press int8 compress failed: {err}"))?;
                    let data = cuda_download(stream, &compressed)?;
                    let scales = cuda_download(stream, &scales)?;
                    Ok(EvicPressCompressionResult::Int8 { data, scales })
                }
                EvicPressCompression::Int2 => {
                    let (int8_cache, _int8_scales) = kernel
                        .compress_to_int8_f32(stream, &kv_dev, config.seq_len, config.head_dim)
                        .map_err(|err| format!("CUDA evic-press int8 step failed: {err}"))?;
                    let (compressed, scales) = kernel
                        .compress_to_int2_f32(stream, &int8_cache, config.seq_len, config.head_dim)
                        .map_err(|err| format!("CUDA evic-press int2 compress failed: {err}"))?;
                    let data = cuda_download(stream, &compressed)?;
                    let scales = cuda_download(stream, &scales)?;
                    Ok(EvicPressCompressionResult::Int2 { data, scales })
                }
            }
        })();
        match result {
            Ok(res) => Some(res),
            Err(err) => {
                log::warn!("CUDA evic_press_compress failed: {err}");
                None
            }
        }
    }

    #[inline(always)]
    fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        let attn_f32: Vec<f32> = attention_weights.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<EvicPressEvictResult, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EVICT_PRESS_KERNEL, CudaEvicPressKernel::new)
                .ok_or("CUDA evic-press kernel unavailable")?;
            let attn_dev = cuda_upload(stream, &attn_f32)?;
            let ages_dev = cuda_upload(stream, token_ages)?;
            let zones_dev = cuda_upload(stream, current_zones)?;
            let importance = kernel
                .compute_importance_f32(
                    stream,
                    &attn_dev,
                    &ages_dev,
                    config.batch_size,
                    config.num_heads,
                    config.seq_len,
                    config.recency_weight,
                    config.attention_weight,
                )
                .map_err(|err| format!("CUDA evic-press importance failed: {err}"))?;
            let new_zones = kernel
                .zone_transition(
                    stream,
                    &zones_dev,
                    &importance,
                    config.seq_len,
                    config.hot_threshold,
                    config.warm_threshold,
                    config.cache_pressure,
                )
                .map_err(|err| format!("CUDA evic-press zone transition failed: {err}"))?;
            let importance_host = cuda_download(stream, &importance)?;
            let new_zones_host = cuda_download(stream, &new_zones)?;
            Ok(EvicPressEvictResult {
                importance: importance_host,
                new_zones: new_zones_host,
            })
        })();
        match result {
            Ok(res) => Some(res),
            Err(err) => {
                log::warn!("CUDA evic_press_evict failed: {err}");
                None
            }
        }
    }

    #[inline(always)]
    fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        let logits_f32: Vec<f32> = head_logits.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<MedusaForwardResult, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_MEDUSA_KERNEL, CudaMedusaKernel::new)
                .ok_or("CUDA medusa kernel unavailable")?;
            let logits_dev = cuda_upload(stream, &logits_f32)?;
            let (top_k_tokens, top_k_probs) = kernel
                .top_k_sample_f32(
                    stream,
                    &logits_dev,
                    config.batch_size,
                    config.num_heads,
                    config.vocab_size,
                    config.top_k,
                    config.temperature,
                )
                .map_err(|err| format!("CUDA medusa top-k failed: {err}"))?;
            let (candidates, probs, counts) = kernel
                .build_candidates_f32(
                    stream,
                    &top_k_tokens,
                    &top_k_probs,
                    config.batch_size,
                    config.num_heads,
                    config.top_k,
                    config.max_candidates,
                )
                .map_err(|err| format!("CUDA medusa build candidates failed: {err}"))?;
            let candidate_tokens = cuda_download(stream, &candidates)?;
            let candidate_probs = cuda_download(stream, &probs)?;
            let num_candidates = cuda_download(stream, &counts)?;
            Ok(MedusaForwardResult {
                candidate_tokens,
                candidate_probs,
                num_candidates,
            })
        })();
        match result {
            Ok(res) => Some(res),
            Err(err) => {
                log::warn!("CUDA medusa_forward fallback: {err}");
                ops::medusa::medusa_forward_stateless(&logits_f32, config).ok()
            }
        }
    }

    #[inline(always)]
    fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        let logits_f32: Vec<f32> = target_logits.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<MedusaVerifyResult, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_MEDUSA_KERNEL, CudaMedusaKernel::new)
                .ok_or("CUDA medusa kernel unavailable")?;
            let tokens_dev = cuda_upload(stream, candidate_tokens)?;
            let logits_dev = cuda_upload(stream, &logits_f32)?;
            let (accepted, best) = kernel
                .verify_candidates_f32(
                    stream,
                    &tokens_dev,
                    &logits_dev,
                    config.batch_size,
                    config.num_candidates,
                    config.seq_len,
                    config.vocab_size,
                )
                .map_err(|err| format!("CUDA medusa verify failed: {err}"))?;
            let accepted_lengths = cuda_download(stream, &accepted)?;
            let best_candidate = cuda_download(stream, &best)?;
            Ok(MedusaVerifyResult {
                accepted_lengths,
                best_candidate,
            })
        })();
        match result {
            Ok(res) => Some(res),
            Err(err) => {
                log::warn!("CUDA medusa_verify fallback: {err}");
                ops::medusa::medusa_verify_stateless(candidate_tokens, &logits_f32, config).ok()
            }
        }
    }

    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        let result = (|| -> Result<PromptCacheLookupResult, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_PROMPT_CACHE_KERNEL, CudaPromptCacheKernel::new)
                .ok_or("CUDA prompt cache kernel unavailable")?;
            let tokens_dev = cuda_upload(stream, tokens)?;
            let cache_hashes_dev = cuda_upload(stream, cache_hashes)?;
            let cache_lengths_i32: Vec<i32> = cache_lengths.iter().map(|&v| v as i32).collect();
            let cache_lengths_dev = cuda_upload(stream, &cache_lengths_i32)?;
            let query_hashes = kernel
                .compute_hash_f32(stream, &tokens_dev, tokens.len(), config.hash_seed)
                .map_err(|err| format!("CUDA prompt hash failed: {err}"))?;
            let (best_entry, match_length) = kernel
                .find_prefix_match(
                    stream,
                    &query_hashes,
                    &cache_hashes_dev,
                    &cache_lengths_dev,
                    tokens.len().min(config.max_cache_len),
                    config.num_entries,
                    config.max_cache_len,
                )
                .map_err(|err| format!("CUDA prompt match failed: {err}"))?;
            let query_hashes_host = cuda_download(stream, &query_hashes)?;
            let mut best_entry_host = cuda_download(stream, &best_entry)?;
            let mut match_length_host = cuda_download(stream, &match_length)?;
            let mut best_entry_value = best_entry_host.pop().unwrap_or(-1);
            let mut match_len_value = match_length_host.pop().unwrap_or(0) as usize;
            if match_len_value < config.min_match_len {
                best_entry_value = -1;
                match_len_value = 0;
            }
            Ok(PromptCacheLookupResult {
                best_entry: best_entry_value,
                match_length: match_len_value,
                query_hashes: query_hashes_host,
            })
        })();
        match result {
            Ok(res) => Some(res),
            Err(err) => {
                log::warn!("CUDA prompt_cache_lookup fallback: {err}");
                ops::prompt_cache::prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config).ok()
            }
        }
    }

    #[inline(always)]
    fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        // Convert to f32 for GPU kernel (internal f32 kernel)
        let cached_f32: Vec<f32> = cached_kv.iter().map(|x| x.to_f32()).collect();
        let fresh_f32: Vec<f32> = fresh_kv.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<Vec<f32>, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_PROMPT_CACHE_KERNEL, CudaPromptCacheKernel::new)
                .ok_or("CUDA prompt cache kernel unavailable")?;
            let cached_dev = cuda_upload(stream, &cached_f32)?;
            let fresh_dev = cuda_upload(stream, &fresh_f32)?;
            let blended = kernel
                .cache_blend_f32(
                    stream,
                    &cached_dev,
                    &fresh_dev,
                    config.match_len,
                    config.fresh_len,
                    config.num_heads,
                    config.head_dim,
                    config.blend_window,
                )
                .map_err(|err| format!("CUDA prompt cache blend failed: {err}"))?;
            cuda_download(stream, &blended)
        })();
        match result {
            Ok(res) => Some(res.into_iter().map(T::from_f32).collect()),
            Err(err) => {
                log::warn!("CUDA prompt_cache_blend fallback: {err}");
                ops::prompt_cache::prompt_cache_blend(&cached_f32, &fresh_f32, config)
                    .ok()
                    .map(|v| v.into_iter().map(T::from_f32).collect())
            }
        }
    }

    #[inline(always)]
    fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        // Convert to f32 for GPU kernel
        let q_f32: Vec<f32> = query.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = key.iter().map(|x| x.to_f32()).collect();
        let v_f32: Vec<f32> = value.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<ChunkedPrefillResult<f32>, String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_CHUNKED_PREFILL_KERNEL, CudaChunkedPrefillKernel::new)
                .ok_or("CUDA chunked prefill kernel unavailable")?;
            let q_dev = cuda_upload(stream, &q_f32)?;
            let k_dev = cuda_upload(stream, &k_f32)?;
            let v_dev = cuda_upload(stream, &v_f32)?;
            let (output_dev, lse_dev) = kernel
                .chunked_attention_f32(
                    stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    config.batch_size,
                    config.num_heads,
                    config.query_len,
                    config.chunk_len,
                    config.head_dim,
                    config.chunk_start,
                    config.causal,
                )
                .map_err(|err| format!("CUDA chunked prefill failed: {err}"))?;
            let output = cuda_download(stream, &output_dev)?;
            let log_sum_exp = cuda_download(stream, &lse_dev)?;
            Ok(ChunkedPrefillResult { output, log_sum_exp })
        })();
        match result {
            Ok(res) => Some(ChunkedPrefillResult {
                output: res.output.into_iter().map(T::from_f32).collect(),
                log_sum_exp: res.log_sum_exp,  // log_sum_exp is already Vec<f32>
            }),
            Err(err) => {
                log::warn!("CUDA chunked_prefill_attention fallback: {err}");
                ops::chunked_prefill::chunked_prefill_attention(&q_f32, &k_f32, &v_f32, config)
                    .ok()
                    .map(|r| ChunkedPrefillResult {
                        output: r.output.into_iter().map(T::from_f32).collect(),
                        log_sum_exp: r.log_sum_exp,  // log_sum_exp is already Vec<f32>
                    })
            }
        }
    }

    fn update_kv_cache_gpu(&self, _: &mut GpuKVCache, _: usize, _: &GpuTensor, _: &GpuTensor) -> Result<(), String> {
        Err("CUDA update_kv_cache_gpu not wired in backend.rs".into())
    }

    fn rerank_pipeline(&self, bq: &[u32], bd: &[u32], iq: &[u32], id: &[u32], nv: usize, config: &GpuRerankConfig, is: f32) -> Result<GpuRerankStageResult, String> {
        ops::embedding::rerank_pipeline(bq, bd, iq, id, nv, config, is)
    }

    fn binary_ip_hamming(&self, queries: &[u64], database: &[u64], scores: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        let result = (|| -> Result<(), String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EMBEDDING_KERNEL, CudaEmbeddingOpsKernel::new)
                .ok_or("CUDA embedding kernel unavailable")?;
            let q_u32 = split_u64_to_u32(queries);
            let d_u32 = split_u64_to_u32(database);
            let q_dev = cuda_upload(stream, &q_u32)?;
            let d_dev = cuda_upload(stream, &d_u32)?;
            let out_dev = kernel
                .binary_ip_hamming(stream, &q_dev, &d_dev, config.dim, config.num_queries, config.num_vectors)
                .map_err(|err| format!("CUDA binary ip hamming failed: {err}"))?;
            let out_host = cuda_download(stream, &out_dev)?;
            if out_host.len() != scores.len() {
                return Err("CUDA binary ip hamming output length mismatch".into());
            }
            scores.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("CUDA binary_ip_hamming fallback: {err}");
            ops::embedding::binary_ip_hamming_simd(queries, database, scores, config);
        }
    }

    fn binary_ip_asymmetric(&self, queries: &[f32], database: &[u64], scores: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        let result = (|| -> Result<(), String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EMBEDDING_KERNEL, CudaEmbeddingOpsKernel::new)
                .ok_or("CUDA embedding kernel unavailable")?;
            let d_u32 = split_u64_to_u32(database);
            let q_dev = cuda_upload(stream, queries)?;
            let d_dev = cuda_upload(stream, &d_u32)?;
            let out_dev = kernel
                .binary_ip_asymmetric(stream, &q_dev, &d_dev, config.dim, config.num_queries, config.num_vectors)
                .map_err(|err| format!("CUDA binary ip asymmetric failed: {err}"))?;
            let out_host = cuda_download(stream, &out_dev)?;
            if out_host.len() != scores.len() {
                return Err("CUDA binary ip asymmetric output length mismatch".into());
            }
            scores.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("CUDA binary_ip_asymmetric fallback: {err}");
            ops::embedding::binary_ip_asymmetric(queries, database, scores, config);
        }
    }

    fn int8_dot_product(&self, queries: &[i8], database: &[i8], scores: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        let result = (|| -> Result<(), String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EMBEDDING_KERNEL, CudaEmbeddingOpsKernel::new)
                .ok_or("CUDA embedding kernel unavailable")?;
            let q_u32 = pack_i8_to_u32(queries);
            let d_u32 = pack_i8_to_u32(database);
            let q_dev = cuda_upload(stream, &q_u32)?;
            let d_dev = cuda_upload(stream, &d_u32)?;
            let out_dev = kernel
                .int8_dot_product(stream, &q_dev, &d_dev, config.dim, config.num_queries, config.num_vectors, config.scale)
                .map_err(|err| format!("CUDA int8 dot failed: {err}"))?;
            let out_host = cuda_download(stream, &out_dev)?;
            if out_host.len() != scores.len() {
                return Err("CUDA int8 dot output length mismatch".into());
            }
            scores.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("CUDA int8_dot_product fallback: {err}");
            ops::embedding::int8_dot_product_unrolled(queries, database, scores, config);
        }
    }

    fn int4_packed_dot_product(&self, queries: &[u8], database: &[u8], scores: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        let result = (|| -> Result<(), String> {
            let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
            let kernel = init_cuda_kernel(&CUDA_EMBEDDING_KERNEL, CudaEmbeddingOpsKernel::new)
                .ok_or("CUDA embedding kernel unavailable")?;
            let q_u32 = pack_u8_to_u32(queries);
            let d_u32 = pack_u8_to_u32(database);
            let q_dev = cuda_upload(stream, &q_u32)?;
            let d_dev = cuda_upload(stream, &d_u32)?;
            let out_dev = kernel
                .int4_dot_product(
                    stream,
                    &q_dev,
                    &d_dev,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                    config.scale,
                    config.zero_point as i32,
                )
                .map_err(|err| format!("CUDA int4 dot failed: {err}"))?;
            let out_host = cuda_download(stream, &out_dev)?;
            if out_host.len() != scores.len() {
                return Err("CUDA int4 dot output length mismatch".into());
            }
            scores.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("CUDA int4_packed_dot_product fallback: {err}");
            ops::embedding::int4_packed_dot_product(queries, database, scores, config);
        }
    }

    fn backend_type(&self) -> BackendType { BackendType::Cuda }
}

impl Backend for CudaBackend {
    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        // Delegate to inherent method which falls back to CPU
        CudaBackend::flash_attention(self, q, k, v, output, config);
    }

    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        // Delegate to inherent method which falls back to CPU
        CudaBackend::paged_attention(self, q, k_cache, v_cache, page_table, seq_lens, output, config);
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        // Delegate to inherent method
        CudaBackend::softmax(self, input, output, config);
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        // Delegate to inherent method
        CudaBackend::matmul(self, a, b, c, config);
    }

    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        CudaBackend::linear_forward_gpu(self, input, weight, output, params)
    }

    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        CudaBackend::linear_forward_gpu_add(self, input, weight, output, params)
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, input: &[T], weight: &GpuTensor, output: &mut [T], params: LinearParams) -> Result<(), String> {
        CudaBackend::linear_forward_host_io(self, input, weight, output, params)
    }

    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        CudaBackend::linear_forward_host_io_readback(self, gpu_tensor, output)
    }

    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        CudaBackend::ffn_forward_gpu(self, input, gate, up, down, intermediate, output, gate_up_params, down_params)
    }

    fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String> {
        CudaBackend::attention_forward_gpu(self, q, k_cache, v_cache, output, config)
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        CudaBackend::rms_norm_gpu(self, input, weight, output, eps)
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        CudaBackend::rms_norm_gpu_inplace(self, data, weight, eps)
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        CudaBackend::allocate_buffer(self, size_bytes)
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        CudaBackend::allocate_weights(self, data, shape, dtype)
    }

    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let buf = match &gpu.buffer { GpuBuffer::Cuda(buf) => buf, _ => return Err("CUDA readback buffer mismatch".into()) };
        let bytes = cuda_download(stream, buf.as_ref())?;
        let values: &[T] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const T,
                bytes.len() / std::mem::size_of::<T>(),
            )
        };
        if values.len() != host.len() {
            return Err("CUDA readback length mismatch".into());
        }
        host.copy_from_slice(values);
        Ok(())
    }

    fn readback_u32(&self, _: &GpuTensor, _: &mut [u32]) -> Result<(), String> {
        Err("CUDA readback_u32 not implemented".into())
    }

    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        let expected_bytes = host.len() * std::mem::size_of::<T>();
        if gpu.size_in_bytes != expected_bytes {
            return Err("CUDA upload size mismatch".into());
        }
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, expected_bytes) };
        let buffer = cuda_upload(stream, bytes)?;
        gpu.buffer = GpuBuffer::Cuda(Arc::new(buffer));
        Ok(())
    }

    fn update_kv_cache_gpu(&self, cache: &mut GpuKVCache, layer_idx: usize, new_k: &GpuTensor, new_v: &GpuTensor) -> Result<(), String> {
        CudaBackend::update_kv_cache_gpu(self, cache, layer_idx, new_k, new_v)
    }

    #[inline(always)]
    fn eagle3_confidence<T: KernelFloat>(&self, layer_hidden_states: &[&[T]], confidence_weights: &[T], confidence_bias: T, config: &Eagle3Config) -> Option<Vec<T>> {
        CudaBackend::eagle3_confidence(self, layer_hidden_states, confidence_weights, confidence_bias, config)
    }

    #[inline(always)]
    fn spec_ee_confidence<T: KernelFloat>(&self, hidden_states: &[T], classifier_weight: &[T], classifier_bias: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        CudaBackend::spec_ee_confidence(self, hidden_states, classifier_weight, classifier_bias, config)
    }

    #[inline(always)]
    fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        CudaBackend::flash_tree_attention(self, query, key, value, tree_mask, output, config)
    }

    #[inline(always)]
    fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        CudaBackend::medusa_forward(self, head_logits, config)
    }

    #[inline(always)]
    fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        CudaBackend::medusa_verify(self, candidate_tokens, target_logits, config)
    }

    #[inline(always)]
    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        CudaBackend::int2_quantize(self, input, config)
    }

    #[inline(always)]
    fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        CudaBackend::int2_dequantize(self, quantized, scales, zeros, config)
    }

    #[inline(always)]
    fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        CudaBackend::evic_press_compress(self, kv_cache, config)
    }

    #[inline(always)]
    fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        CudaBackend::evic_press_evict(self, attention_weights, token_ages, current_zones, config)
    }

    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        CudaBackend::prompt_cache_lookup(self, tokens, cache_hashes, cache_lengths, config)
    }

    #[inline(always)]
    fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        CudaBackend::prompt_cache_blend(self, cached_kv, fresh_kv, config)
    }

    #[inline(always)]
    fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        CudaBackend::chunked_prefill_attention(self, query, key, value, config)
    }

    fn rerank_pipeline(&self, binary_query: &[u32], binary_database: &[u32], int8_query: &[u32], int8_database: &[u32], num_vectors: usize, config: &GpuRerankConfig, int8_scale: f32) -> Result<GpuRerankStageResult, String> {
        CudaBackend::rerank_pipeline(self, binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale)
    }

    fn binary_ip_hamming(&self, queries: &[u64], database: &[u64], scores: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        CudaBackend::binary_ip_hamming(self, queries, database, scores, config);
    }

    fn binary_ip_asymmetric(&self, queries: &[f32], database: &[u64], scores: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        CudaBackend::binary_ip_asymmetric(self, queries, database, scores, config);
    }

    fn int8_dot_product(&self, queries: &[i8], database: &[i8], scores: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        CudaBackend::int8_dot_product(self, queries, database, scores, config);
    }

    fn int4_packed_dot_product(&self, queries: &[u8], database: &[u8], scores: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        CudaBackend::int4_packed_dot_product(self, queries, database, scores, config);
    }

    // =========================================================================
    // MoE GPU Operations (CUDA)
    // =========================================================================

    fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String> {
        // CUDA: Replace buffer with a new zeroed allocation
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let size_bytes = tensor.size_in_bytes;
        let new_buf = cuda_alloc_zeros(&stream, size_bytes)?;
        tensor.buffer = GpuBuffer::Cuda(Arc::new(new_buf));
        Ok(())
    }

    fn tensor_add_gpu(&self, output: &mut GpuTensor, input: &GpuTensor) -> Result<(), String> {
        let in_buf = match &input.buffer {
            GpuBuffer::Cuda(b) => b,
            _ => return Err("CUDA tensor_add_gpu: input not CUDA buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Cuda(b) => b,
            _ => return Err("CUDA tensor_add_gpu: output not CUDA buffer".into()),
        };
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        let size = input.len().min(output.len());
        cuda_tensor_add(&stream, out_buf.as_ref(), in_buf.as_ref(), size)
            .map_err(|e| format!("CUDA tensor_add_gpu failed: {e}"))
    }

    fn tensor_slice_gpu(&self, input: &GpuTensor, offset: usize, len: usize, output: &mut GpuTensor) -> Result<(), String> {
        let in_buf = match &input.buffer {
            GpuBuffer::Cuda(b) => b,
            _ => return Err("CUDA tensor_slice_gpu: input not CUDA buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Cuda(b) => b,
            _ => return Err("CUDA tensor_slice_gpu: output not CUDA buffer".into()),
        };
        if offset + len > input.len() {
            return Err(format!("CUDA tensor_slice_gpu: offset {} + len {} > input len {}", offset, len, input.len()));
        }
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        cuda_tensor_slice(&stream, in_buf.as_ref(), out_buf.as_ref(), offset * std::mem::size_of::<f32>(), len * std::mem::size_of::<f32>())
            .map_err(|e| format!("CUDA tensor_slice_gpu failed: {e}"))
    }

    fn tensor_scale_add_gpu(&self, input: &GpuTensor, output: &mut GpuTensor, offset: usize, scale: f32) -> Result<(), String> {
        let in_buf = match &input.buffer {
            GpuBuffer::Cuda(b) => b,
            _ => return Err("CUDA tensor_scale_add_gpu: input not CUDA buffer".into()),
        };
        let out_buf = match &output.buffer {
            GpuBuffer::Cuda(b) => b,
            _ => return Err("CUDA tensor_scale_add_gpu: output not CUDA buffer".into()),
        };
        if offset + input.len() > output.len() {
            return Err(format!("CUDA tensor_scale_add_gpu: offset {} + input len {} > output len {}", offset, input.len(), output.len()));
        }
        let stream = get_cuda_stream().ok_or("CUDA stream unavailable")?;
        cuda_tensor_scale_add(&stream, out_buf.as_ref(), in_buf.as_ref(), offset, input.len(), scale)
            .map_err(|e| format!("CUDA tensor_scale_add_gpu failed: {e}"))
    }

    fn moe_route_gpu(
        &self,
        _hidden_states: &GpuTensor,
        _gate_weights: &GpuTensor,
        _expert_indices_out: &mut GpuTensor,
        _expert_weights_out: &mut GpuTensor,
        _config: MoERoutingGpuConfig,
    ) -> Result<(), String> {
        Err("CUDA moe_route_gpu not yet implemented - requires custom CUDA kernel".into())
    }

    fn moe_forward_gpu_pure(
        &self,
        _input: &GpuTensor,
        _expert_indices: &GpuTensor,
        _expert_weights: &GpuTensor,
        _all_gate_weights: &GpuTensor,
        _all_up_weights: &GpuTensor,
        _all_down_weights: &GpuTensor,
        _output: &mut GpuTensor,
        _config: MoEForwardConfig,
    ) -> Result<(), String> {
        Err("CUDA moe_forward_gpu_pure not yet implemented - requires custom CUDA kernel".into())
    }

    fn backend_type(&self) -> BackendType {
        CudaBackend::backend_type(self)
    }
}


// =============================================================================
// ROCm Backend Implementation
// =============================================================================

// ROCm Backend - Fat Binary: always compiled, runtime detection
#[derive(Clone, Default)]
pub struct RocmBackend;

impl RocmBackend {
    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        // Convert to f32 for GPU kernel
        let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
        let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<Vec<f32>, String> {
            let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
            let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
            let kernel = init_rocm_kernel(&ROCM_FLASH_ATTN, HsaFlashAttentionKernel::new)
                .ok_or("ROCm flash attention kernel unavailable")?;
            let q_dev = hsa_upload(agent, &q_f32)?;
            let k_dev = hsa_upload(agent, &k_f32)?;
            let v_dev = hsa_upload(agent, &v_f32)?;
            let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
            let out_dev = kernel
                .forward_f32(
                    queue,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    config.batch_size,
                    config.num_heads,
                    config.seq_len_q,
                    config.seq_len_kv,
                    config.head_dim,
                    config.causal,
                    scale,
                    0,
                )
                .map_err(|err| format!("ROCm flash attention failed: {err}"))?;
            hsa_download(&out_dev)
        })();
        match result {
            Ok(out_f32) => {
                for (o, v) in output.iter_mut().zip(out_f32.iter()) {
                    *o = T::from_f32(*v);
                }
            }
            Err(err) => {
                log::warn!("ROCm flash_attention fallback: {err}");
                ops::attention::flash_attention(q, k, v, output, config);
            }
        }
    }

    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        // Convert to f32 for GPU kernel
        let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = k_cache.iter().map(|x| x.to_f32()).collect();
        let v_f32: Vec<f32> = v_cache.iter().map(|x| x.to_f32()).collect();

        let result = (|| -> Result<Vec<f32>, String> {
            let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
            let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
            let kernel = init_rocm_kernel(&ROCM_PAGED_ATTN, HsaPagedAttentionKernel::new)
                .ok_or("ROCm paged attention kernel unavailable")?;
            let (layout, block_tables, block_offsets) = build_paged_tables_f32(
                &q_f32,
                &k_f32,
                &v_f32,
                page_table,
                seq_lens,
                output.len(),
                &config,
            )
            .ok_or("ROCm paged attention layout invalid")?;
            let q_dev = hsa_upload(agent, &q_f32)?;
            let k_dev = hsa_upload(agent, &k_f32)?;
            let v_dev = hsa_upload(agent, &v_f32)?;
            let block_tables_dev = hsa_upload(agent, &block_tables)?;
            let block_offsets_dev = hsa_upload(agent, &block_offsets)?;
            let out_dev = kernel
                .forward_f32(
                    queue,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &block_tables_dev,
                    &block_offsets_dev,
                    layout.batch_size,
                    layout.num_heads,
                    layout.head_dim,
                    layout.page_size,
                    layout.seq_len,
                )
                .map_err(|err| format!("ROCm paged attention failed: {err}"))?;
            hsa_download(&out_dev)
        })();
        match result {
            Ok(out_f32) => {
                for (o, v) in output.iter_mut().zip(out_f32.iter()) {
                    *o = T::from_f32(*v);
                }
            }
            Err(err) => {
                log::warn!("ROCm paged_attention fallback: {err}");
                ops::paged_attn::paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
        }
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        // ROCm softmax - fallback to CPU
        CpuBackend.softmax(input, output, config);
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        CpuBackend.matmul(a, b, c, config);
    }
    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let queue = get_rocm_queue().ok_or("ROCm queue not available")?;
        // HsaLinear::new takes &GpuAgent, initialize directly
        let kernel = ROCM_LINEAR_KERNEL.get_or_init(|| HsaLinear::new(agent).ok()).as_ref()
            .ok_or("ROCm linear kernel init failed")?;
        let in_buf = match &input.buffer { GpuBuffer::Rocm(b) => b, _ => return Err("Input not ROCm buffer".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Rocm(b) => b, _ => return Err("Weight not ROCm buffer".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Rocm(b) => b, _ => return Err("Output not ROCm buffer".into()) };
        // Get raw pointers from HsaBuffer
        let batch_size = input.shape.first().copied().unwrap_or(1);
        kernel.forward(queue, params, in_buf.as_ptr(), w_buf.as_ptr(), None, out_buf.as_ptr() as *mut _, batch_size)
    }

    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        // For now, just do linear (fused add not implemented)
        self.linear_forward_gpu(input, weight, output, params)
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, _input: &[T], _weight: &GpuTensor, _output: &mut [T], _params: LinearParams) -> Result<(), String> {
        Err("ROCm linear_forward_host_io not yet implemented".into())
    }

    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        let buf = match &gpu_tensor.buffer { GpuBuffer::Rocm(b) => b, _ => return Err("Tensor not ROCm".into()) };
        // HSA memory copy to host
        let size = output.len() * std::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr() as *const f32,
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(())
    }

    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let queue = get_rocm_queue().ok_or("ROCm queue not available")?;
        let kernel = ROCM_LINEAR_KERNEL
            .get_or_init(|| HsaLinear::new(agent).ok())
            .as_ref()
            .ok_or("ROCm linear kernel init failed")?;
        let batch_size = input.shape.first().copied().unwrap_or(1);
        let in_buf = match &input.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm FFN input buffer mismatch".into()) };
        let gate_buf = match &gate.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm FFN gate buffer mismatch".into()) };
        let up_buf = match &up.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm FFN up buffer mismatch".into()) };
        let down_buf = match &down.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm FFN down buffer mismatch".into()) };
        let inter_buf = match &intermediate.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm FFN intermediate buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm FFN output buffer mismatch".into()) };
        kernel
            .fused_gate_up_silu(
                queue,
                gate_up_params,
                in_buf.as_ptr(),
                gate_buf.as_ptr(),
                up_buf.as_ptr(),
                inter_buf.as_ptr(),
                batch_size,
            )
            .map_err(|err| format!("ROCm FFN gate/up failed: {err}"))?;
        kernel
            .forward(
                queue,
                down_params,
                inter_buf.as_ptr(),
                down_buf.as_ptr(),
                None,
                out_buf.as_ptr(),
                batch_size,
            )
            .map_err(|err| format!("ROCm FFN down projection failed: {err}"))
    }
    fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String> {
        let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
        let kernel = init_rocm_kernel(&ROCM_FLASH_ATTN, HsaFlashAttentionKernel::new)
            .ok_or("ROCm flash attention kernel unavailable")?;
        if q.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("ROCm attention_forward_gpu only supports f32 tensors".into());
        }
        let q_buf = match &q.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm attention input buffer mismatch".into()) };
        let k_buf = match k_cache { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm attention key cache buffer mismatch".into()) };
        let v_buf = match v_cache { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm attention value cache buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm attention output buffer mismatch".into()) };
        let q_f32: &HsaBuffer<f32> = unsafe { std::mem::transmute(q_buf.as_ref()) };
        let k_f32: &HsaBuffer<f32> = unsafe { std::mem::transmute(k_buf.as_ref()) };
        let v_f32: &HsaBuffer<f32> = unsafe { std::mem::transmute(v_buf.as_ref()) };
        let out_f32: &HsaBuffer<f32> = unsafe { std::mem::transmute(out_buf.as_ref()) };
        let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
        kernel
            .forward_gpu_f32(
                queue,
                q_f32,
                k_f32,
                v_f32,
                out_f32,
                config.batch_size,
                config.num_heads,
                config.seq_len_q,
                config.seq_len_kv,
                config.head_dim,
                config.causal,
                scale,
                0,
            )
            .map_err(|err| format!("ROCm attention_forward_gpu failed: {err}"))
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        if input.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("ROCm rms_norm_gpu only supports f32 tensors".into());
        }
        let (rows, hidden) = rms_norm_dims(input, weight, Some(output))?;
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let queue = get_rocm_queue().ok_or("ROCm queue not available")?;
        let kernel = ROCM_RMS_NORM_KERNEL
            .get_or_init(|| HsaRmsNormKernel::new(agent).ok())
            .as_ref()
            .ok_or("ROCm RMSNorm kernel init failed")?;
        let in_buf = match &input.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm RMSNorm input buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm RMSNorm weight buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm RMSNorm output buffer mismatch".into()) };
        kernel.forward(queue, in_buf.as_ptr(), w_buf.as_ptr(), out_buf.as_ptr() as *mut _, rows, hidden, eps)
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        if data.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 {
            return Err("ROCm rms_norm_gpu_inplace only supports f32 tensors".into());
        }
        let (rows, hidden) = rms_norm_dims(data, weight, None)?;
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let queue = get_rocm_queue().ok_or("ROCm queue not available")?;
        let kernel = ROCM_RMS_NORM_KERNEL
            .get_or_init(|| HsaRmsNormKernel::new(agent).ok())
            .as_ref()
            .ok_or("ROCm RMSNorm kernel init failed")?;
        let data_buf = match &data.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm RMSNorm data buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm RMSNorm weight buffer mismatch".into()) };
        kernel.forward(queue, data_buf.as_ptr(), w_buf.as_ptr(), data_buf.as_ptr() as *mut _, rows, hidden, eps)
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let buffer = HsaBuffer::<u8>::alloc_zeros(agent, size_bytes).map_err(|e| format!("ROCm alloc failed: {e}"))?;
        Ok(GpuBuffer::Rocm(Arc::new(buffer)))
    }
    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let buffer = hsa_upload(agent, data)?;
        Ok(GpuTensor::new(GpuBuffer::Rocm(Arc::new(buffer)), shape, dtype, BackendType::Rocm))
    }
    #[inline(always)]
    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        if let GpuBuffer::Rocm(buffer) = &gpu.buffer {
            let f32_buffer: &HsaBuffer<f32> = unsafe { std::mem::transmute(buffer.as_ref()) };
            let data = hsa_download(f32_buffer)?;
            if data.len() != host.len() {
                return Err("ROCm readback length mismatch".into());
            }
            for (h, d) in host.iter_mut().zip(data.iter()) {
                *h = T::from_f32(*d);
            }
            Ok(())
        } else {
            Err("Not ROCm buffer".into())
        }
    }

    fn readback_u32(&self, _: &GpuTensor, _: &mut [u32]) -> Result<(), String> {
        Err("ROCm readback_u32 not implemented".into())
    }

    #[inline(always)]
    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        let element_size = match T::TYPE_ID {
            FloatType::F32 => 4,
            FloatType::F16 | FloatType::BF16 => 2,
        };
        let expected_bytes = host.len() * element_size;
        if gpu.size_in_bytes != expected_bytes {
            return Err("ROCm upload size mismatch".into());
        }
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        // Convert to f32, then to bytes for upload
        let host_f32: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&host_f32);
        let buffer = HsaBuffer::<u8>::from_slice(agent, bytes)
            .map_err(|e| format!("ROCm upload failed: {e}"))?;
        gpu.buffer = GpuBuffer::Rocm(Arc::new(buffer));
        Ok(())
    }

    #[inline(always)]
    fn eagle3_confidence<T: KernelFloat>(&self, layers: &[&[T]], w: &[T], b: T, config: &Eagle3Config) -> Option<Vec<T>> {
        if init_rocm_config_kernel(&ROCM_EAGLE3_KERNEL, || HsaEagle3Kernel::new(HsaEagle3Config::default())).is_none() {
            log::warn!("ROCm eagle3 kernel unavailable; using CPU");
        }
        CpuBackend.eagle3_confidence(layers, w, b, config)
    }

    #[inline(always)]
    fn spec_ee_confidence<T: KernelFloat>(&self, hs: &[T], w: &[T], b: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        if init_rocm_config_kernel(&ROCM_SPEC_EE_KERNEL, || HsaSpecEEKernel::new(HsaSpecEEConfig::default())).is_none() {
            log::warn!("ROCm spec_ee kernel unavailable; using CPU");
        }
        CpuBackend.spec_ee_confidence(hs, w, b, config)
    }

    #[inline(always)]
    fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        if init_rocm_config_kernel(&ROCM_FLASH_TREE_KERNEL, || HsaFlashTreeAttnKernel::new(HsaFlashTreeAttnConfig::default())).is_none() {
            log::warn!("ROCm flash_tree kernel unavailable; using CPU");
        }
        CpuBackend.flash_tree_attention(query, key, value, tree_mask, output, config)
    }

    #[inline(always)]
    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        let rocm_config = HsaInt2QuantizerConfig {
            group_size: config.group_size,
            symmetric: false,
        };
        if init_rocm_config_kernel(&ROCM_INT2_KERNEL, || HsaInt2QuantizerKernel::new(rocm_config)).is_none() {
            log::warn!("ROCm int2 quantizer unavailable; using CPU");
        }
        CpuBackend.int2_quantize(input, config)
    }

    #[inline(always)]
    fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        let rocm_config = HsaInt2QuantizerConfig {
            group_size: config.group_size,
            symmetric: false,
        };
        if init_rocm_config_kernel(&ROCM_INT2_KERNEL, || HsaInt2QuantizerKernel::new(rocm_config)).is_none() {
            log::warn!("ROCm int2 quantizer unavailable; using CPU");
        }
        CpuBackend.int2_dequantize(quantized, scales, zeros, config)
    }

    #[inline(always)]
    fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        if init_rocm_config_kernel(&ROCM_EVICT_PRESS_KERNEL, || HsaEvicPressKernel::new(HsaEvicPressConfig::default())).is_none() {
            log::warn!("ROCm evic_press kernel unavailable; using CPU");
        }
        CpuBackend.evic_press_compress(kv_cache, config)
    }

    #[inline(always)]
    fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        if init_rocm_config_kernel(&ROCM_EVICT_PRESS_KERNEL, || HsaEvicPressKernel::new(HsaEvicPressConfig::default())).is_none() {
            log::warn!("ROCm evic_press kernel unavailable; using CPU");
        }
        CpuBackend.evic_press_evict(attention_weights, token_ages, current_zones, config)
    }

    #[inline(always)]
    fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        if init_rocm_config_kernel(&ROCM_MEDUSA_KERNEL, || HsaMedusaKernel::new(HsaMedusaConfig::default())).is_none() {
            log::warn!("ROCm medusa kernel unavailable; using CPU");
        }
        CpuBackend.medusa_forward(head_logits, config)
    }

    #[inline(always)]
    fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        if init_rocm_config_kernel(&ROCM_MEDUSA_KERNEL, || HsaMedusaKernel::new(HsaMedusaConfig::default())).is_none() {
            log::warn!("ROCm medusa kernel unavailable; using CPU");
        }
        CpuBackend.medusa_verify(candidate_tokens, target_logits, config)
    }
    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        if init_rocm_config_kernel(&ROCM_PROMPT_CACHE_KERNEL, || HsaPromptCacheKernel::new(HsaPromptCacheConfig::default())).is_none() {
            log::warn!("ROCm prompt_cache kernel unavailable; using CPU");
        }
        CpuBackend.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config)
    }
    #[inline(always)]
    fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        if init_rocm_config_kernel(&ROCM_PROMPT_CACHE_KERNEL, || HsaPromptCacheKernel::new(HsaPromptCacheConfig::default())).is_none() {
            log::warn!("ROCm prompt_cache kernel unavailable; using CPU");
        }
        // Convert to f32 for internal kernel, then convert back
        let cached_f32: Vec<f32> = cached_kv.iter().map(|x| x.to_f32()).collect();
        let fresh_f32: Vec<f32> = fresh_kv.iter().map(|x| x.to_f32()).collect();
        CpuBackend.prompt_cache_blend(&cached_f32, &fresh_f32, config)
            .map(|v| v.into_iter().map(T::from_f32).collect())
    }
    #[inline(always)]
    fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        if init_rocm_config_kernel(&ROCM_CHUNKED_PREFILL_KERNEL, || HsaChunkedPrefillKernel::new(HsaChunkedPrefillConfig::default())).is_none() {
            log::warn!("ROCm chunked_prefill kernel unavailable; using CPU");
        }
        // Convert to f32 for internal kernel
        let q_f32: Vec<f32> = query.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = key.iter().map(|x| x.to_f32()).collect();
        let v_f32: Vec<f32> = value.iter().map(|x| x.to_f32()).collect();
        CpuBackend.chunked_prefill_attention(&q_f32, &k_f32, &v_f32, config)
            .map(|res| ChunkedPrefillResult {
                output: res.output.into_iter().map(T::from_f32).collect(),
                log_sum_exp: res.log_sum_exp,  // log_sum_exp is already Vec<f32>
            })
    }
    fn update_kv_cache_gpu(&self, _: &mut GpuKVCache, _: usize, _: &GpuTensor, _: &GpuTensor) -> Result<(), String> {
        Err("ROCm KV cache update not yet implemented".into())
    }
    fn rerank_pipeline(&self, bq: &[u32], bd: &[u32], iq: &[u32], id: &[u32], nv: usize, config: &GpuRerankConfig, is: f32) -> Result<GpuRerankStageResult, String> {
        if init_rocm_kernel(&ROCM_EMBEDDING_KERNEL, HsaEmbeddingOpsKernel::new).is_none() {
            log::warn!("ROCm embedding kernel unavailable; using CPU");
        }
        CpuBackend.rerank_pipeline(bq, bd, iq, id, nv, config, is)
    }
    fn binary_ip_hamming(&self, q: &[u64], d: &[u64], s: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        let result = (|| -> Result<(), String> {
            let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
            let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
            let kernel = init_rocm_kernel(&ROCM_EMBEDDING_KERNEL, HsaEmbeddingOpsKernel::new)
                .ok_or("ROCm embedding kernel unavailable")?;
            let q_u32 = split_u64_to_u32(q);
            let d_u32 = split_u64_to_u32(d);
            let q_dev = hsa_upload(agent, &q_u32)?;
            let d_dev = hsa_upload(agent, &d_u32)?;
            let out_dev = kernel
                .binary_ip_hamming(
                    queue,
                    &q_dev,
                    &d_dev,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                )
                .map_err(|err| format!("ROCm binary_ip_hamming failed: {err}"))?;
            let out_host = hsa_download(&out_dev)?;
            if out_host.len() != s.len() {
                return Err("ROCm binary_ip_hamming output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("ROCm binary_ip_hamming fallback: {err}");
            ops::embedding::binary_ip_hamming(q, d, s, config);
        }
    }
    fn binary_ip_asymmetric(&self, q: &[f32], d: &[u64], s: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        let result = (|| -> Result<(), String> {
            let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
            let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
            let kernel = init_rocm_kernel(&ROCM_EMBEDDING_KERNEL, HsaEmbeddingOpsKernel::new)
                .ok_or("ROCm embedding kernel unavailable")?;
            let d_u32 = split_u64_to_u32(d);
            let q_dev = hsa_upload(agent, q)?;
            let d_dev = hsa_upload(agent, &d_u32)?;
            let out_dev = kernel
                .binary_ip_asymmetric(
                    queue,
                    &q_dev,
                    &d_dev,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                )
                .map_err(|err| format!("ROCm binary_ip_asymmetric failed: {err}"))?;
            let out_host = hsa_download(&out_dev)?;
            if out_host.len() != s.len() {
                return Err("ROCm binary_ip_asymmetric output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("ROCm binary_ip_asymmetric fallback: {err}");
            ops::embedding::binary_ip_asymmetric(q, d, s, config);
        }
    }
    fn int8_dot_product(&self, q: &[i8], d: &[i8], s: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        let result = (|| -> Result<(), String> {
            let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
            let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
            let kernel = init_rocm_kernel(&ROCM_EMBEDDING_KERNEL, HsaEmbeddingOpsKernel::new)
                .ok_or("ROCm embedding kernel unavailable")?;
            let q_u32 = pack_i8_to_u32(q);
            let d_u32 = pack_i8_to_u32(d);
            let q_dev = hsa_upload(agent, &q_u32)?;
            let d_dev = hsa_upload(agent, &d_u32)?;
            let out_dev = kernel
                .int8_dot_product(
                    queue,
                    &q_dev,
                    &d_dev,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                    config.scale,
                )
                .map_err(|err| format!("ROCm int8_dot_product failed: {err}"))?;
            let out_host = hsa_download(&out_dev)?;
            if out_host.len() != s.len() {
                return Err("ROCm int8_dot_product output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("ROCm int8_dot_product fallback: {err}");
            ops::embedding::int8_dot_product(q, d, s, config);
        }
    }
    fn int4_packed_dot_product(&self, q: &[u8], d: &[u8], s: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        let result = (|| -> Result<(), String> {
            let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
            let queue = get_rocm_queue().ok_or("ROCm queue unavailable")?;
            let kernel = init_rocm_kernel(&ROCM_EMBEDDING_KERNEL, HsaEmbeddingOpsKernel::new)
                .ok_or("ROCm embedding kernel unavailable")?;
            let q_u32 = pack_u8_to_u32(q);
            let d_u32 = pack_u8_to_u32(d);
            let q_dev = hsa_upload(agent, &q_u32)?;
            let d_dev = hsa_upload(agent, &d_u32)?;
            let out_dev = kernel
                .int4_dot_product(
                    queue,
                    &q_dev,
                    &d_dev,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                    config.scale,
                    config.zero_point as i32,
                )
                .map_err(|err| format!("ROCm int4_dot_product failed: {err}"))?;
            let out_host = hsa_download(&out_dev)?;
            if out_host.len() != s.len() {
                return Err("ROCm int4_dot_product output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("ROCm int4_packed_dot_product fallback: {err}");
            ops::embedding::int4_packed_dot_product(q, d, s, config);
        }
    }
    fn backend_type(&self) -> BackendType { BackendType::Rocm }
}

impl Backend for RocmBackend {
    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        // Delegate to inherent method (already handles GPU kernel with f32 conversion)
        RocmBackend::flash_attention(self, q, k, v, output, config);
    }

    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        // Delegate to inherent method (already handles GPU kernel with f32 conversion)
        RocmBackend::paged_attention(self, q, k_cache, v_cache, page_table, seq_lens, output, config);
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        // Delegate to inherent method
        RocmBackend::softmax(self, input, output, config);
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        // Delegate to inherent method
        RocmBackend::matmul(self, a, b, c, config);
    }

    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        RocmBackend::linear_forward_gpu(self, input, weight, output, params)
    }

    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        RocmBackend::linear_forward_gpu_add(self, input, weight, output, params)
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, input: &[T], weight: &GpuTensor, output: &mut [T], params: LinearParams) -> Result<(), String> {
        RocmBackend::linear_forward_host_io(self, input, weight, output, params)
    }

    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        RocmBackend::linear_forward_host_io_readback(self, gpu_tensor, output)
    }

    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        RocmBackend::ffn_forward_gpu(self, input, gate, up, down, intermediate, output, gate_up_params, down_params)
    }

    fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String> {
        RocmBackend::attention_forward_gpu(self, q, k_cache, v_cache, output, config)
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        RocmBackend::rms_norm_gpu(self, input, weight, output, eps)
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        RocmBackend::rms_norm_gpu_inplace(self, data, weight, eps)
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        RocmBackend::allocate_buffer(self, size_bytes)
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        RocmBackend::allocate_weights(self, data, shape, dtype)
    }

    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        let buffer = match &gpu.buffer { GpuBuffer::Rocm(buf) => buf, _ => return Err("ROCm readback buffer mismatch".into()) };
        let typed: &HsaBuffer<T> = unsafe { std::mem::transmute(buffer.as_ref()) };
        let data = hsa_download(typed)?;
        if data.len() != host.len() {
            return Err("ROCm readback length mismatch".into());
        }
        host.copy_from_slice(&data);
        Ok(())
    }

    fn readback_u32(&self, _: &GpuTensor, _: &mut [u32]) -> Result<(), String> {
        Err("ROCm readback_u32 not implemented".into())
    }

    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        let expected_bytes = host.len() * std::mem::size_of::<T>();
        if gpu.size_in_bytes != expected_bytes {
            return Err("ROCm upload size mismatch".into());
        }
        let agent = get_rocm_agent().ok_or("ROCm agent not available")?;
        let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, expected_bytes) };
        let buffer = HsaBuffer::<u8>::from_slice(agent, bytes)
            .map_err(|e| format!("ROCm upload failed: {e}"))?;
        gpu.buffer = GpuBuffer::Rocm(Arc::new(buffer));
        Ok(())
    }

    fn update_kv_cache_gpu(&self, cache: &mut GpuKVCache, layer_idx: usize, new_k: &GpuTensor, new_v: &GpuTensor) -> Result<(), String> {
        RocmBackend::update_kv_cache_gpu(self, cache, layer_idx, new_k, new_v)
    }

    #[inline(always)]
    fn eagle3_confidence<T: KernelFloat>(&self, layer_hidden_states: &[&[T]], confidence_weights: &[T], confidence_bias: T, config: &Eagle3Config) -> Option<Vec<T>> {
        RocmBackend::eagle3_confidence(self, layer_hidden_states, confidence_weights, confidence_bias, config)
    }

    #[inline(always)]
    fn spec_ee_confidence<T: KernelFloat>(&self, hidden_states: &[T], classifier_weight: &[T], classifier_bias: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        RocmBackend::spec_ee_confidence(self, hidden_states, classifier_weight, classifier_bias, config)
    }

    #[inline(always)]
    fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        RocmBackend::flash_tree_attention(self, query, key, value, tree_mask, output, config)
    }

    #[inline(always)]
    fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        RocmBackend::medusa_forward(self, head_logits, config)
    }

    #[inline(always)]
    fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        RocmBackend::medusa_verify(self, candidate_tokens, target_logits, config)
    }

    #[inline(always)]
    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        RocmBackend::int2_quantize(self, input, config)
    }

    #[inline(always)]
    fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        RocmBackend::int2_dequantize(self, quantized, scales, zeros, config)
    }

    #[inline(always)]
    fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        RocmBackend::evic_press_compress(self, kv_cache, config)
    }

    #[inline(always)]
    fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        RocmBackend::evic_press_evict(self, attention_weights, token_ages, current_zones, config)
    }

    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        RocmBackend::prompt_cache_lookup(self, tokens, cache_hashes, cache_lengths, config)
    }

    #[inline(always)]
    fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        RocmBackend::prompt_cache_blend(self, cached_kv, fresh_kv, config)
    }

    #[inline(always)]
    fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        RocmBackend::chunked_prefill_attention(self, query, key, value, config)
    }

    fn rerank_pipeline(&self, binary_query: &[u32], binary_database: &[u32], int8_query: &[u32], int8_database: &[u32], num_vectors: usize, config: &GpuRerankConfig, int8_scale: f32) -> Result<GpuRerankStageResult, String> {
        RocmBackend::rerank_pipeline(self, binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale)
    }

    fn binary_ip_hamming(&self, queries: &[u64], database: &[u64], scores: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        RocmBackend::binary_ip_hamming(self, queries, database, scores, config);
    }

    fn binary_ip_asymmetric(&self, queries: &[f32], database: &[u64], scores: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        RocmBackend::binary_ip_asymmetric(self, queries, database, scores, config);
    }

    fn int8_dot_product(&self, queries: &[i8], database: &[i8], scores: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        RocmBackend::int8_dot_product(self, queries, database, scores, config);
    }

    fn int4_packed_dot_product(&self, queries: &[u8], database: &[u8], scores: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        RocmBackend::int4_packed_dot_product(self, queries, database, scores, config);
    }

    // =========================================================================
    // MoE GPU Operations (ROCm/HSA)
    // =========================================================================

    fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String> {
        // ROCm: Replace buffer with a new zeroed allocation
        // HsaBuffer doesn't have memset, so we allocate fresh zeros
        let agent = get_rocm_agent().ok_or("ROCm agent unavailable")?;
        let size_bytes = tensor.size_in_bytes;
        let new_buf = HsaBuffer::<u8>::alloc_zeros(agent, size_bytes)
            .map_err(|e| format!("ROCm tensor_zero_gpu: {e}"))?;
        tensor.buffer = GpuBuffer::Rocm(Arc::new(new_buf));
        Ok(())
    }

    fn tensor_add_gpu(&self, _output: &mut GpuTensor, _input: &GpuTensor) -> Result<(), String> {
        // TODO: Implement as HIP kernel
        Err("ROCm tensor_add_gpu not yet implemented - requires custom kernel".into())
    }

    fn tensor_slice_gpu(&self, _input: &GpuTensor, _offset: usize, _len: usize, _output: &mut GpuTensor) -> Result<(), String> {
        // TODO: Implement as HSA memcpy with offset
        Err("ROCm tensor_slice_gpu not yet implemented".into())
    }

    fn tensor_scale_add_gpu(&self, _input: &GpuTensor, _output: &mut GpuTensor, _offset: usize, _scale: f32) -> Result<(), String> {
        // TODO: Implement as HIP kernel
        Err("ROCm tensor_scale_add_gpu not yet implemented - requires custom kernel".into())
    }

    fn moe_route_gpu(
        &self,
        _hidden_states: &GpuTensor,
        _gate_weights: &GpuTensor,
        _expert_indices_out: &mut GpuTensor,
        _expert_weights_out: &mut GpuTensor,
        _config: MoERoutingGpuConfig,
    ) -> Result<(), String> {
        Err("ROCm moe_route_gpu not yet implemented - requires custom HIP kernel".into())
    }

    fn moe_forward_gpu_pure(
        &self,
        _input: &GpuTensor,
        _expert_indices: &GpuTensor,
        _expert_weights: &GpuTensor,
        _all_gate_weights: &GpuTensor,
        _all_up_weights: &GpuTensor,
        _all_down_weights: &GpuTensor,
        _output: &mut GpuTensor,
        _config: MoEForwardConfig,
    ) -> Result<(), String> {
        Err("ROCm moe_forward_gpu_pure not yet implemented - requires custom HIP kernel".into())
    }

    fn backend_type(&self) -> BackendType {
        RocmBackend::backend_type(self)
    }
}


// =============================================================================
// Metal Backend Implementation (macOS only)
// =============================================================================

#[cfg(target_os = "macos")]
#[derive(Clone, Default)]
pub struct MetalBackend;

#[cfg(target_os = "macos")]
impl MetalBackend {
    fn flash_attention_f32(&self, q: &[f32], k: &[f32], v: &[f32], output: &mut [f32], config: FlashAttentionConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_FLASH_ATTN, MetalFlashAttentionKernel::new)
                .ok_or("Metal flash attention kernel unavailable")?;
            let q_buf = metal_buffer_from_slice(device, q)?;
            let k_buf = metal_buffer_from_slice(device, k)?;
            let v_buf = metal_buffer_from_slice(device, v)?;
            let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
            let out_buf = kernel
                .forward_f32(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    config.batch_size,
                    config.num_heads,
                    config.seq_len_q,
                    config.seq_len_kv,
                    config.head_dim,
                    config.causal,
                    scale,
                    0,
                )
                .map_err(|err| format!("Metal flash attention failed: {err}"))?;
            let out_host = metal_download::<f32>(&out_buf, output.len())?;
            if out_host.len() != output.len() {
                return Err("Metal flash attention output length mismatch".into());
            }
            output.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal flash_attention_f32 fallback: {err}");
            ops::attention::flash_attention(q, k, v, output, config);
        }
    }

    fn flash_attention_f16(&self, q: &[half::f16], k: &[half::f16], v: &[half::f16], output: &mut [half::f16], config: FlashAttentionConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_FLASH_ATTN, MetalFlashAttentionKernel::new)
                .ok_or("Metal flash attention kernel unavailable")?;
            let q_u16 = f16_to_u16(q);
            let k_u16 = f16_to_u16(k);
            let v_u16 = f16_to_u16(v);
            let q_buf = metal_buffer_from_slice(device, &q_u16)?;
            let k_buf = metal_buffer_from_slice(device, &k_u16)?;
            let v_buf = metal_buffer_from_slice(device, &v_u16)?;
            let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
            let out_buf = kernel
                .forward_f16(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    config.batch_size,
                    config.num_heads,
                    config.seq_len_q,
                    config.seq_len_kv,
                    config.head_dim,
                    config.causal,
                    scale,
                    0,
                )
                .map_err(|err| format!("Metal flash attention f16 failed: {err}"))?;
            let out_u16 = metal_download::<u16>(&out_buf, output.len())?;
            let out_f16 = u16_to_f16(&out_u16);
            if out_f16.len() != output.len() {
                return Err("Metal flash attention f16 output length mismatch".into());
            }
            output.copy_from_slice(&out_f16);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal flash_attention_f16 fallback: {err}");
            ops::attention::flash_attention(q, k, v, output, config);
        }
    }

    fn paged_attention_f32(&self, q: &[f32], k_cache: &[f32], v_cache: &[f32], page_table: &[u32], seq_lens: &[u32], output: &mut [f32], config: PagedAttentionConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_PAGED_ATTN, MetalPagedAttentionKernel::new)
                .ok_or("Metal paged attention kernel unavailable")?;
            let (layout, block_tables, block_offsets) = build_paged_tables_f32(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                output.len(),
                &config,
            )
            .ok_or("Metal paged attention layout invalid")?;
            let q_buf = metal_buffer_from_slice(device, q)?;
            let k_buf = metal_buffer_from_slice(device, k_cache)?;
            let v_buf = metal_buffer_from_slice(device, v_cache)?;
            let block_tables_buf = metal_buffer_from_slice(device, &block_tables)?;
            let block_offsets_buf = metal_buffer_from_slice(device, &block_offsets)?;
            let out_buf = kernel
                .forward_f32(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &block_tables_buf,
                    &block_offsets_buf,
                    layout.batch_size,
                    layout.num_heads,
                    layout.head_dim,
                    layout.page_size,
                    layout.seq_len,
                )
                .map_err(|err| format!("Metal paged attention failed: {err}"))?;
            let out_host = metal_download::<f32>(&out_buf, output.len())?;
            if out_host.len() != output.len() {
                return Err("Metal paged attention output length mismatch".into());
            }
            output.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal paged_attention_f32 fallback: {err}");
            ops::paged_attn::paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
        }
    }

    fn softmax_f32(&self, input: &[f32], output: &mut [f32], config: SoftmaxConfig) {
        ops::softmax::softmax(input, output, config);
    }

    fn matmul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], config: MatmulConfig) {
        ops::matmul(a, b, c, config);
    }

    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        let kernel = init_metal_kernel(&METAL_LINEAR_KERNEL, MetalLinear::new)
            .ok_or("Metal linear kernel unavailable")?;
        let in_buf = match &input.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal linear input buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal linear weight buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal linear output buffer mismatch".into()) };
        let batch_size = input.shape.first().copied().unwrap_or(1);
        kernel.forward(params, in_buf, w_buf, None, out_buf, batch_size)
    }

    fn linear_forward_gpu_add(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        // For now, just do linear (fused add not implemented)
        self.linear_forward_gpu(input, weight, output, params)
    }

    fn linear_forward_host_io<T: KernelFloat>(&self, _input: &[T], _weight: &GpuTensor, _output: &mut [T], _params: LinearParams) -> Result<(), String> {
        Err("Metal linear_forward_host_io not yet implemented".into())
    }

    fn linear_forward_host_io_readback(&self, gpu_tensor: &GpuTensor, output: &mut [f32]) -> Result<(), String> {
        let buf = match &gpu_tensor.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Tensor not Metal".into()) };
        metal_download::<f32>(buf, output.len())
            .map(|data| output.copy_from_slice(&data))
            .map_err(|e| format!("Metal readback failed: {}", e))
    }

    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        let kernel = init_metal_kernel(&METAL_LINEAR_KERNEL, MetalLinear::new)
            .ok_or("Metal linear kernel unavailable")?;
        let batch_size = input.shape.first().copied().unwrap_or(1);
        let in_buf = match &input.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal FFN input buffer mismatch".into()) };
        let gate_buf = match &gate.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal FFN gate buffer mismatch".into()) };
        let up_buf = match &up.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal FFN up buffer mismatch".into()) };
        let down_buf = match &down.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal FFN down buffer mismatch".into()) };
        let inter_buf = match &intermediate.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal FFN intermediate buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal FFN output buffer mismatch".into()) };
        kernel
            .fused_gate_up_silu(
                gate_up_params,
                in_buf,
                gate_buf,
                up_buf,
                inter_buf,
                batch_size,
            )
            .map_err(|err| format!("Metal FFN gate/up failed: {err}"))?;
        kernel
            .forward(
                down_params,
                inter_buf,
                down_buf,
                None,
                out_buf,
                batch_size,
            )
            .map_err(|err| format!("Metal FFN down projection failed: {err}"))
    }

    fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String> {
        let kernel = init_metal_kernel(&METAL_FLASH_ATTN, MetalFlashAttentionKernel::new)
            .ok_or("Metal flash attention kernel unavailable")?;
        if q.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("Metal attention_forward_gpu only supports f32 tensors".into());
        }
        let q_buf = match &q.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal attention input buffer mismatch".into()) };
        let k_buf = match k_cache { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal attention key cache buffer mismatch".into()) };
        let v_buf = match v_cache { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal attention value cache buffer mismatch".into()) };
        let out_dst = match &output.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal attention output buffer mismatch".into()) };
        let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
        let out_buf = kernel
            .forward_f32(
                q_buf,
                k_buf,
                v_buf,
                config.batch_size,
                config.num_heads,
                config.seq_len_q,
                config.seq_len_kv,
                config.head_dim,
                config.causal,
                scale,
                0,
            )
            .map_err(|err| format!("Metal attention_forward_gpu failed: {err}"))?;
        let byte_len = output.size_in_bytes;
        if out_buf.length() < byte_len as u64 {
            return Err("Metal attention output buffer too small".into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents() as *const u8,
                out_dst.contents() as *mut u8,
                byte_len,
            );
        }
        Ok(())
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        if input.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 || output.dtype != TensorDtype::F32 {
            return Err("Metal rms_norm_gpu only supports f32 tensors".into());
        }
        let (rows, hidden) = rms_norm_dims(input, weight, Some(output))?;
        let kernel = init_metal_kernel(&METAL_RMS_NORM_KERNEL, MetalRmsNorm::new)
            .ok_or("Metal RMSNorm kernel unavailable")?;
        let in_buf = match &input.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal RMSNorm input buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal RMSNorm weight buffer mismatch".into()) };
        let out_buf = match &output.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal RMSNorm output buffer mismatch".into()) };
        kernel.forward(in_buf, w_buf, out_buf, rows, hidden, eps)
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        if data.dtype != TensorDtype::F32 || weight.dtype != TensorDtype::F32 {
            return Err("Metal rms_norm_gpu_inplace only supports f32 tensors".into());
        }
        let (rows, hidden) = rms_norm_dims(data, weight, None)?;
        let kernel = init_metal_kernel(&METAL_RMS_NORM_KERNEL, MetalRmsNorm::new)
            .ok_or("Metal RMSNorm kernel unavailable")?;
        let data_buf = match &data.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal RMSNorm data buffer mismatch".into()) };
        let w_buf = match &weight.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal RMSNorm weight buffer mismatch".into()) };
        kernel.forward(data_buf, w_buf, data_buf, rows, hidden, eps)
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        let device = get_metal_device().ok_or("Metal device unavailable")?;
        let buffer = device.new_buffer(size_bytes as u64, metal::MTLResourceOptions::StorageModeShared);
        Ok(GpuBuffer::Metal(Arc::new(buffer)))
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        let device = get_metal_device().ok_or("Metal device unavailable")?;
        let buffer = metal_buffer_from_slice(device, data)?;
        Ok(GpuTensor::new(GpuBuffer::Metal(Arc::new(buffer)), shape, dtype, BackendType::Metal))
    }

    fn readback_f32(&self, gpu: &GpuTensor, host: &mut [f32]) -> Result<(), String> {
        let buffer = match &gpu.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Not Metal buffer".into()) };
        let data = metal_download::<f32>(buffer, host.len())?;
        if data.len() != host.len() {
            return Err("Metal readback length mismatch".into());
        }
        host.copy_from_slice(&data);
        Ok(())
    }

    fn upload_f32(&self, host: &[f32], gpu: &mut GpuTensor) -> Result<(), String> {
        let expected_bytes = host.len() * std::mem::size_of::<f32>();
        if gpu.size_in_bytes != expected_bytes {
            return Err("Metal upload size mismatch".into());
        }
        let device = get_metal_device().ok_or("Metal device unavailable")?;
        let buffer = metal_buffer_from_slice(device, host)?;
        gpu.buffer = GpuBuffer::Metal(Arc::new(buffer));
        Ok(())
    }

    fn eagle3_confidence_f32(&self, layer_hidden_states: &[&[f32]], confidence_weights: &[f32], confidence_bias: f32, config: &Eagle3Config) -> Option<Vec<f32>> {
        let result = (|| -> Result<Vec<f32>, String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_EAGLE3_KERNEL, MetalEagle3Kernel::new)
                .ok_or("Metal eagle3 kernel unavailable")?;
            if layer_hidden_states.len() < config.fusion_layers {
                return Err("Metal eagle3 layer count mismatch".into());
            }
            let start = layer_hidden_states.len() - config.fusion_layers;
            let mut layer_buffers = Vec::with_capacity(config.fusion_layers);
            for layer in &layer_hidden_states[start..] {
                layer_buffers.push(metal_buffer_from_slice(device, layer)?);
            }
            let layer_refs: Vec<&metal::Buffer> = layer_buffers.iter().collect();
            let fused = kernel
                .fuse_hidden_f32(
                    &layer_refs,
                    config.batch_size,
                    config.seq_len,
                    config.hidden_dim,
                    config.fusion_layers,
                )
                .map_err(|err| format!("Metal eagle3 fuse failed: {err}"))?;
            let weight_buf = metal_buffer_from_slice(device, confidence_weights)?;
            let out_buf = kernel
                .predict_confidence_f32(
                    &fused,
                    &weight_buf,
                    confidence_bias,
                    config.batch_size,
                    config.seq_len,
                    config.fused_dim(),
                )
                .map_err(|err| format!("Metal eagle3 predict failed: {err}"))?;
            let out_host = metal_download::<f32>(&out_buf, config.batch_size * config.seq_len)?;
            Ok(out_host)
        })();
        match result {
            Ok(values) => Some(values),
            Err(err) => {
                log::warn!("Metal eagle3_confidence_f32 fallback: {err}");
                CpuBackend.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config)
            }
        }
    }

    fn spec_ee_confidence_f32(&self, hidden_states: &[f32], classifier_weight: &[f32], classifier_bias: f32, config: &SpecEEConfig) -> Option<Vec<f32>> {
        if init_metal_kernel(&METAL_SPEC_EE_KERNEL, MetalSpecEEKernel::new).is_none() {
            log::warn!("Metal spec_ee kernel unavailable; using CPU");
        }
        CpuBackend.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config)
    }

    fn flash_tree_attention_f32(&self, query: &[f32], key: &[f32], value: &[f32], tree_mask: &[i32], output: &mut [f32], config: &FlashTreeAttentionConfig) -> bool {
        if init_metal_kernel(&METAL_FLASH_TREE_KERNEL, MetalFlashTreeAttnKernel::new).is_none() {
            log::warn!("Metal flash_tree kernel unavailable; using CPU");
        }
        CpuBackend.flash_tree_attention(query, key, value, tree_mask, output, config)
    }

    fn int2_quantize_f32(&self, input: &[f32], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        if init_metal_kernel(&METAL_INT2_KERNEL, MetalInt2QuantizerKernel::new).is_none() {
            log::warn!("Metal int2 quantizer unavailable; using CPU");
        }
        CpuBackend.int2_quantize(input, config)
    }

    fn int2_dequantize_f32(&self, quantized: &[i8], scales: &[f32], zeros: &[f32], config: &Int2QuantConfig) -> Option<Vec<f32>> {
        if init_metal_kernel(&METAL_INT2_KERNEL, MetalInt2QuantizerKernel::new).is_none() {
            log::warn!("Metal int2 quantizer unavailable; using CPU");
        }
        CpuBackend.int2_dequantize(quantized, scales, zeros, config)
    }

    fn evic_press_compress_f32(&self, kv_cache: &[f32], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        if init_metal_kernel(&METAL_EVICT_PRESS_KERNEL, MetalEvicPressKernel::new).is_none() {
            log::warn!("Metal evic_press kernel unavailable; using CPU");
        }
        CpuBackend.evic_press_compress(kv_cache, config)
    }

    fn evic_press_evict_f32(&self, attention_weights: &[f32], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        if init_metal_kernel(&METAL_EVICT_PRESS_KERNEL, MetalEvicPressKernel::new).is_none() {
            log::warn!("Metal evic_press kernel unavailable; using CPU");
        }
        CpuBackend.evic_press_evict(attention_weights, token_ages, current_zones, config)
    }

    fn medusa_forward(&self, head_logits: &[f32], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        if init_metal_kernel(&METAL_MEDUSA_KERNEL, MetalMedusaKernel::new).is_none() {
            log::warn!("Metal medusa kernel unavailable; using CPU");
        }
        CpuBackend.medusa_forward(head_logits, config)
    }

    fn medusa_verify(&self, candidate_tokens: &[i32], target_logits: &[f32], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        if init_metal_kernel(&METAL_MEDUSA_KERNEL, MetalMedusaKernel::new).is_none() {
            log::warn!("Metal medusa kernel unavailable; using CPU");
        }
        CpuBackend.medusa_verify(candidate_tokens, target_logits, config)
    }

    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        if init_metal_kernel(&METAL_PROMPT_CACHE_KERNEL, MetalPromptCacheKernel::new).is_none() {
            log::warn!("Metal prompt_cache kernel unavailable; using CPU");
        }
        CpuBackend.prompt_cache_lookup(tokens, cache_hashes, cache_lengths, config)
    }

    fn prompt_cache_blend_f32(&self, cached_kv: &[f32], fresh_kv: &[f32], config: &PromptCacheBlendConfig) -> Option<Vec<f32>> {
        if init_metal_kernel(&METAL_PROMPT_CACHE_KERNEL, MetalPromptCacheKernel::new).is_none() {
            log::warn!("Metal prompt_cache kernel unavailable; using CPU");
        }
        CpuBackend.prompt_cache_blend(cached_kv, fresh_kv, config)
    }

    fn chunked_prefill_attention_f32(&self, query: &[f32], key: &[f32], value: &[f32], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<f32>> {
        if init_metal_kernel(&METAL_CHUNKED_PREFILL_KERNEL, MetalChunkedPrefillKernel::new).is_none() {
            log::warn!("Metal chunked_prefill kernel unavailable; using CPU");
        }
        CpuBackend.chunked_prefill_attention(query, key, value, config)
    }

    fn update_kv_cache_gpu(&self, _: &mut GpuKVCache, _: usize, _: &GpuTensor, _: &GpuTensor) -> Result<(), String> {
        Err("Metal KV cache update not implemented".into())
    }

    fn rerank_pipeline(&self, bq: &[u32], bd: &[u32], iq: &[u32], id: &[u32], nv: usize, config: &GpuRerankConfig, is: f32) -> Result<GpuRerankStageResult, String> {
        if init_metal_kernel(&METAL_EMBEDDING_KERNEL, MetalEmbeddingOpsKernel::new).is_none() {
            log::warn!("Metal embedding kernel unavailable; using CPU");
        }
        CpuBackend.rerank_pipeline(bq, bd, iq, id, nv, config, is)
    }

    fn binary_ip_hamming(&self, q: &[u64], d: &[u64], s: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_EMBEDDING_KERNEL, MetalEmbeddingOpsKernel::new)
                .ok_or("Metal embedding kernel unavailable")?;
            let q_u32 = split_u64_to_u32(q);
            let d_u32 = split_u64_to_u32(d);
            let q_buf = metal_buffer_from_slice(device, &q_u32)?;
            let d_buf = metal_buffer_from_slice(device, &d_u32)?;
            let out_buf = kernel
                .binary_ip_hamming(
                    &q_buf,
                    &d_buf,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                )
                .map_err(|err| format!("Metal binary_ip_hamming failed: {err}"))?;
            let out_host = metal_download::<i32>(&out_buf, s.len())?;
            if out_host.len() != s.len() {
                return Err("Metal binary_ip_hamming output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal binary_ip_hamming fallback: {err}");
            ops::embedding::binary_ip_hamming(q, d, s, config);
        }
    }

    fn binary_ip_asymmetric(&self, q: &[f32], d: &[u64], s: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_EMBEDDING_KERNEL, MetalEmbeddingOpsKernel::new)
                .ok_or("Metal embedding kernel unavailable")?;
            let d_u32 = split_u64_to_u32(d);
            let q_buf = metal_buffer_from_slice(device, q)?;
            let d_buf = metal_buffer_from_slice(device, &d_u32)?;
            let out_buf = kernel
                .binary_ip_asymmetric(
                    &q_buf,
                    &d_buf,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                )
                .map_err(|err| format!("Metal binary_ip_asymmetric failed: {err}"))?;
            let out_host = metal_download::<f32>(&out_buf, s.len())?;
            if out_host.len() != s.len() {
                return Err("Metal binary_ip_asymmetric output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal binary_ip_asymmetric fallback: {err}");
            ops::embedding::binary_ip_asymmetric(q, d, s, config);
        }
    }

    fn int8_dot_product(&self, q: &[i8], d: &[i8], s: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_EMBEDDING_KERNEL, MetalEmbeddingOpsKernel::new)
                .ok_or("Metal embedding kernel unavailable")?;
            let q_u32 = pack_i8_to_u32(q);
            let d_u32 = pack_i8_to_u32(d);
            let q_buf = metal_buffer_from_slice(device, &q_u32)?;
            let d_buf = metal_buffer_from_slice(device, &d_u32)?;
            let out_buf = kernel
                .int8_dot_product(
                    &q_buf,
                    &d_buf,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                    config.scale,
                )
                .map_err(|err| format!("Metal int8_dot_product failed: {err}"))?;
            let out_host = metal_download::<f32>(&out_buf, s.len())?;
            if out_host.len() != s.len() {
                return Err("Metal int8_dot_product output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal int8_dot_product fallback: {err}");
            ops::embedding::int8_dot_product(q, d, s, config);
        }
    }

    fn int4_packed_dot_product(&self, q: &[u8], d: &[u8], s: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        let result = (|| -> Result<(), String> {
            let device = get_metal_device().ok_or("Metal device unavailable")?;
            let kernel = init_metal_kernel(&METAL_EMBEDDING_KERNEL, MetalEmbeddingOpsKernel::new)
                .ok_or("Metal embedding kernel unavailable")?;
            let q_u32 = pack_u8_to_u32(q);
            let d_u32 = pack_u8_to_u32(d);
            let q_buf = metal_buffer_from_slice(device, &q_u32)?;
            let d_buf = metal_buffer_from_slice(device, &d_u32)?;
            let out_buf = kernel
                .int4_dot_product(
                    &q_buf,
                    &d_buf,
                    config.dim,
                    config.num_queries,
                    config.num_vectors,
                    config.scale,
                    config.zero_point as i32,
                )
                .map_err(|err| format!("Metal int4_dot_product failed: {err}"))?;
            let out_host = metal_download::<f32>(&out_buf, s.len())?;
            if out_host.len() != s.len() {
                return Err("Metal int4_dot_product output length mismatch".into());
            }
            s.copy_from_slice(&out_host);
            Ok(())
        })();
        if let Err(err) = result {
            log::warn!("Metal int4_packed_dot_product fallback: {err}");
            ops::embedding::int4_packed_dot_product(q, d, s, config);
        }
    }

    fn backend_type(&self) -> BackendType { BackendType::Metal }
}

#[cfg(target_os = "macos")]
impl Backend for MetalBackend {
    #[inline(always)]
    fn flash_attention<T: KernelFloat>(&self, q: &[T], k: &[T], v: &[T], output: &mut [T], config: FlashAttentionConfig) {
        match T::TYPE_ID {
            FloatType::F32 => self.flash_attention_f32(
                cast_slice(q),
                cast_slice(k),
                cast_slice(v),
                cast_slice_mut(output),
                config,
            ),
            FloatType::F16 => self.flash_attention_f16(
                cast_slice(q),
                cast_slice(k),
                cast_slice(v),
                cast_slice_mut(output),
                config,
            ),
            FloatType::BF16 => CpuBackend.flash_attention(q, k, v, output, config),
        }
    }

    #[inline(always)]
    fn paged_attention<T: KernelFloat>(&self, q: &[T], k_cache: &[T], v_cache: &[T], page_table: &[u32], seq_lens: &[u32], output: &mut [T], config: PagedAttentionConfig) {
        match T::TYPE_ID {
            FloatType::F32 => self.paged_attention_f32(
                cast_slice(q),
                cast_slice(k_cache),
                cast_slice(v_cache),
                page_table,
                seq_lens,
                cast_slice_mut(output),
                config,
            ),
            _ => CpuBackend.paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config),
        }
    }

    #[inline(always)]
    fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        match T::TYPE_ID {
            FloatType::F32 => self.softmax_f32(cast_slice(input), cast_slice_mut(output), config),
            _ => CpuBackend.softmax(input, output, config),
        }
    }

    #[inline(always)]
    fn matmul<T: KernelFloat>(&self, a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
        match T::TYPE_ID {
            FloatType::F32 => self.matmul_f32(cast_slice(a), cast_slice(b), cast_slice_mut(c), config),
            _ => CpuBackend.matmul(a, b, c, config),
        }
    }

    fn linear_forward_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, params: LinearParams) -> Result<(), String> {
        MetalBackend::linear_forward_gpu(self, input, weight, output, params)
    }

    fn ffn_forward_gpu(&self, input: &GpuTensor, gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, intermediate: &mut GpuTensor, output: &mut GpuTensor, gate_up_params: LinearParams, down_params: LinearParams) -> Result<(), String> {
        MetalBackend::ffn_forward_gpu(self, input, gate, up, down, intermediate, output, gate_up_params, down_params)
    }

    fn attention_forward_gpu(&self, q: &GpuTensor, k_cache: &GpuBuffer, v_cache: &GpuBuffer, output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String> {
        MetalBackend::attention_forward_gpu(self, q, k_cache, v_cache, output, config)
    }

    fn rms_norm_gpu(&self, input: &GpuTensor, weight: &GpuTensor, output: &mut GpuTensor, eps: f32) -> Result<(), String> {
        MetalBackend::rms_norm_gpu(self, input, weight, output, eps)
    }

    fn rms_norm_gpu_inplace(&self, data: &mut GpuTensor, weight: &GpuTensor, eps: f32) -> Result<(), String> {
        MetalBackend::rms_norm_gpu_inplace(self, data, weight, eps)
    }

    fn allocate_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, String> {
        MetalBackend::allocate_buffer(self, size_bytes)
    }

    fn allocate_weights(&self, data: &[u8], shape: Vec<usize>, dtype: TensorDtype) -> Result<GpuTensor, String> {
        MetalBackend::allocate_weights(self, data, shape, dtype)
    }

    fn readback<T: KernelFloat>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String> {
        let buffer = match &gpu.buffer { GpuBuffer::Metal(buf) => buf, _ => return Err("Metal readback buffer mismatch".into()) };
        let data = metal_download::<T>(buffer, host.len())?;
        if data.len() != host.len() {
            return Err("Metal readback length mismatch".into());
        }
        host.copy_from_slice(&data);
        Ok(())
    }

    fn readback_u32(&self, _: &GpuTensor, _: &mut [u32]) -> Result<(), String> {
        Err("Metal readback_u32 not implemented".into())
    }

    fn upload<T: KernelFloat>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String> {
        let expected_bytes = host.len() * std::mem::size_of::<T>();
        if gpu.size_in_bytes != expected_bytes {
            return Err("Metal upload size mismatch".into());
        }
        let device = get_metal_device().ok_or("Metal device unavailable")?;
        let buffer = metal_buffer_from_slice(device, host)?;
        gpu.buffer = GpuBuffer::Metal(Arc::new(buffer));
        Ok(())
    }

    fn update_kv_cache_gpu(&self, cache: &mut GpuKVCache, layer_idx: usize, new_k: &GpuTensor, new_v: &GpuTensor) -> Result<(), String> {
        MetalBackend::update_kv_cache_gpu(self, cache, layer_idx, new_k, new_v)
    }

    fn eagle3_confidence<T: KernelFloat>(&self, layer_hidden_states: &[&[T]], confidence_weights: &[T], confidence_bias: T, config: &Eagle3Config) -> Option<Vec<T>> {
        match T::TYPE_ID {
            FloatType::F32 => {
                let layers_f32: Vec<&[f32]> = layer_hidden_states
                    .iter()
                    .map(|layer| cast_slice::<T, f32>(*layer))
                    .collect();
                let weights_f32 = cast_slice::<T, f32>(confidence_weights);
                let bias_f32 = confidence_bias.to_f32();
                self.eagle3_confidence_f32(&layers_f32, weights_f32, bias_f32, config)
                    .map(vec_from_f32::<T>)
            }
            _ => CpuBackend.eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, config),
        }
    }

    fn spec_ee_confidence<T: KernelFloat>(&self, hidden_states: &[T], classifier_weight: &[T], classifier_bias: T, config: &SpecEEConfig) -> Option<Vec<T>> {
        match T::TYPE_ID {
            FloatType::F32 => {
                let hs_f32 = cast_slice::<T, f32>(hidden_states);
                let w_f32 = cast_slice::<T, f32>(classifier_weight);
                let b_f32 = classifier_bias.to_f32();
                self.spec_ee_confidence_f32(hs_f32, w_f32, b_f32, config)
                    .map(vec_from_f32::<T>)
            }
            _ => CpuBackend.spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, config),
        }
    }

    fn flash_tree_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], tree_mask: &[i32], output: &mut [T], config: &FlashTreeAttentionConfig) -> bool {
        match T::TYPE_ID {
            FloatType::F32 => self.flash_tree_attention_f32(
                cast_slice(query),
                cast_slice(key),
                cast_slice(value),
                tree_mask,
                cast_slice_mut(output),
                config,
            ),
            _ => CpuBackend.flash_tree_attention(query, key, value, tree_mask, output, config),
        }
    }

    fn medusa_forward<T: KernelFloat>(&self, head_logits: &[T], config: &MedusaConfig) -> Option<MedusaForwardResult> {
        match T::TYPE_ID {
            FloatType::F32 => self.medusa_forward(cast_slice(head_logits), config),
            _ => CpuBackend.medusa_forward(head_logits, config),
        }
    }

    fn medusa_verify<T: KernelFloat>(&self, candidate_tokens: &[i32], target_logits: &[T], config: &MedusaVerifyConfig) -> Option<MedusaVerifyResult> {
        match T::TYPE_ID {
            FloatType::F32 => self.medusa_verify(candidate_tokens, cast_slice(target_logits), config),
            _ => CpuBackend.medusa_verify(candidate_tokens, target_logits, config),
        }
    }

    fn int2_quantize<T: KernelFloat>(&self, input: &[T], config: &Int2QuantConfig) -> Option<Int2QuantResult> {
        match T::TYPE_ID {
            FloatType::F32 => self.int2_quantize_f32(cast_slice(input), config),
            _ => CpuBackend.int2_quantize(input, config),
        }
    }

    fn int2_dequantize<T: KernelFloat>(&self, quantized: &[i8], scales: &[T], zeros: &[T], config: &Int2QuantConfig) -> Option<Vec<T>> {
        match T::TYPE_ID {
            FloatType::F32 => self
                .int2_dequantize_f32(quantized, cast_slice(scales), cast_slice(zeros), config)
                .map(vec_from_f32::<T>),
            _ => CpuBackend.int2_dequantize(quantized, scales, zeros, config),
        }
    }

    fn evic_press_compress<T: KernelFloat>(&self, kv_cache: &[T], config: &EvicPressCompressConfig) -> Option<EvicPressCompressionResult> {
        match T::TYPE_ID {
            FloatType::F32 => self.evic_press_compress_f32(cast_slice(kv_cache), config),
            _ => CpuBackend.evic_press_compress(kv_cache, config),
        }
    }

    fn evic_press_evict<T: KernelFloat>(&self, attention_weights: &[T], token_ages: &[i32], current_zones: &[i32], config: &EvicPressEvictConfig) -> Option<EvicPressEvictResult> {
        match T::TYPE_ID {
            FloatType::F32 => self.evic_press_evict_f32(cast_slice(attention_weights), token_ages, current_zones, config),
            _ => CpuBackend.evic_press_evict(attention_weights, token_ages, current_zones, config),
        }
    }

    fn prompt_cache_lookup(&self, tokens: &[i32], cache_hashes: &[u64], cache_lengths: &[u32], config: &PromptCacheLookupConfig) -> Option<PromptCacheLookupResult> {
        MetalBackend::prompt_cache_lookup(self, tokens, cache_hashes, cache_lengths, config)
    }

    fn prompt_cache_blend<T: KernelFloat>(&self, cached_kv: &[T], fresh_kv: &[T], config: &PromptCacheBlendConfig) -> Option<Vec<T>> {
        match T::TYPE_ID {
            FloatType::F32 => self
                .prompt_cache_blend_f32(cast_slice(cached_kv), cast_slice(fresh_kv), config)
                .map(vec_from_f32::<T>),
            _ => CpuBackend.prompt_cache_blend(cached_kv, fresh_kv, config),
        }
    }

    fn chunked_prefill_attention<T: KernelFloat>(&self, query: &[T], key: &[T], value: &[T], config: &ChunkedPrefillConfig) -> Option<ChunkedPrefillResult<T>> {
        match T::TYPE_ID {
            FloatType::F32 => self
                .chunked_prefill_attention_f32(cast_slice(query), cast_slice(key), cast_slice(value), config)
                .map(chunked_prefill_from_f32::<T>),
            _ => CpuBackend.chunked_prefill_attention(query, key, value, config),
        }
    }

    fn rerank_pipeline(&self, binary_query: &[u32], binary_database: &[u32], int8_query: &[u32], int8_database: &[u32], num_vectors: usize, config: &GpuRerankConfig, int8_scale: f32) -> Result<GpuRerankStageResult, String> {
        MetalBackend::rerank_pipeline(self, binary_query, binary_database, int8_query, int8_database, num_vectors, config, int8_scale)
    }

    fn binary_ip_hamming(&self, queries: &[u64], database: &[u64], scores: &mut [i32], config: &crate::ops::embedding::BinaryIpConfig) {
        MetalBackend::binary_ip_hamming(self, queries, database, scores, config);
    }

    fn binary_ip_asymmetric(&self, queries: &[f32], database: &[u64], scores: &mut [f32], config: &crate::ops::embedding::BinaryIpConfig) {
        MetalBackend::binary_ip_asymmetric(self, queries, database, scores, config);
    }

    fn int8_dot_product(&self, queries: &[i8], database: &[i8], scores: &mut [f32], config: &crate::ops::embedding::Int8DotConfig) {
        MetalBackend::int8_dot_product(self, queries, database, scores, config);
    }

    fn int4_packed_dot_product(&self, queries: &[u8], database: &[u8], scores: &mut [f32], config: &crate::ops::embedding::Int4PackedConfig) {
        MetalBackend::int4_packed_dot_product(self, queries, database, scores, config);
    }

    // =========================================================================
    // MoE GPU Operations (Metal)
    // =========================================================================

    #[cfg(target_os = "macos")]
    fn tensor_zero_gpu(&self, tensor: &mut GpuTensor) -> Result<(), String> {
        let buf = match &tensor.buffer {
            GpuBuffer::Metal(b) => b,
            _ => return Err("Metal tensor_zero_gpu: expected Metal buffer".into()),
        };
        // Metal: fill buffer with zeros using blit encoder
        let device = get_metal_device().ok_or("Metal device unavailable")?;
        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.fill_buffer(buf, metal::NSRange::new(0, buf.length()), 0);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    fn tensor_zero_gpu(&self, _tensor: &mut GpuTensor) -> Result<(), String> {
        Err("Metal backend only available on macOS".into())
    }

    fn tensor_add_gpu(&self, _output: &mut GpuTensor, _input: &GpuTensor) -> Result<(), String> {
        // TODO: Implement as Metal compute shader
        Err("Metal tensor_add_gpu not yet implemented - requires compute shader".into())
    }

    fn tensor_slice_gpu(&self, _input: &GpuTensor, _offset: usize, _len: usize, _output: &mut GpuTensor) -> Result<(), String> {
        // TODO: Implement using Metal blit encoder copy
        Err("Metal tensor_slice_gpu not yet implemented".into())
    }

    fn tensor_scale_add_gpu(&self, _input: &GpuTensor, _output: &mut GpuTensor, _offset: usize, _scale: f32) -> Result<(), String> {
        // TODO: Implement as Metal compute shader
        Err("Metal tensor_scale_add_gpu not yet implemented - requires compute shader".into())
    }

    fn moe_route_gpu(
        &self,
        _hidden_states: &GpuTensor,
        _gate_weights: &GpuTensor,
        _expert_indices_out: &mut GpuTensor,
        _expert_weights_out: &mut GpuTensor,
        _config: MoERoutingGpuConfig,
    ) -> Result<(), String> {
        Err("Metal moe_route_gpu not yet implemented - requires compute shader".into())
    }

    fn moe_forward_gpu_pure(
        &self,
        _input: &GpuTensor,
        _expert_indices: &GpuTensor,
        _expert_weights: &GpuTensor,
        _all_gate_weights: &GpuTensor,
        _all_up_weights: &GpuTensor,
        _all_down_weights: &GpuTensor,
        _output: &mut GpuTensor,
        _config: MoEForwardConfig,
    ) -> Result<(), String> {
        Err("Metal moe_forward_gpu_pure not yet implemented - requires compute shader".into())
    }

    fn backend_type(&self) -> BackendType {
        MetalBackend::backend_type(self)
    }
}
