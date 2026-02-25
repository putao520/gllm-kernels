//! Layer 2: Inference Backend — high-level inference interface.
//!
//! `InferenceBackend` is the primary trait for running transformer inference.
//! `CpuInferenceBackend` provides the reference implementation using Layer 1
//! atomic operators as fallback, with JIT-compiled fused layers when available.

pub mod types;
pub mod tensor;
pub mod weights;
pub mod kv_cache;
pub mod cpu_backend;
pub mod flash_attn;

pub use types::{DType, ModelArch, ModelConfig, InferenceError};
pub use tensor::{DeviceTensor, DeviceKind};
pub use weights::{ModelWeights, LayerWeights};
pub use kv_cache::KvCache;
pub use cpu_backend::CpuInferenceBackend;
pub use flash_attn::{FlashAttnConfig, flash_attn_single_head, flash_attn_multi_head};

/// The primary inference backend trait.
///
/// Implementations handle hardware detection, optional JIT compilation,
/// memory management, and forward pass execution.
///
/// The CPU implementation (`CpuInferenceBackend`) supports two paths:
/// - **JIT path**: entire transformer layer compiled to a single function
/// - **Fallback path**: composed from Layer 1 atomic operators (`Kernels<E>`)
pub trait InferenceBackend: Send + Sync {
    /// Initialize the backend: hardware detection → autotune → JIT compile.
    fn init(config: &ModelConfig) -> Result<Self, InferenceError>
    where
        Self: Sized;

    /// Device kind this backend runs on.
    fn device_kind(&self) -> DeviceKind;

    // ── Memory ──

    /// Allocate a tensor on this device.
    fn alloc(&self, num_elements: usize, dtype: DType) -> Result<DeviceTensor, InferenceError>;

    /// Upload from host slice to device tensor.
    fn upload_f32(&self, src: &[f32], dst: &mut DeviceTensor) -> Result<(), InferenceError>;

    /// Download from device tensor to host slice.
    fn download_f32(&self, src: &DeviceTensor, dst: &mut [f32]) -> Result<(), InferenceError>;

    // ── KV Cache ──

    /// Allocate KV cache for the given batch and sequence configuration.
    fn alloc_kv_cache(
        &self,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<KvCache, InferenceError>;

    // ── Forward pass ──

    /// Run decoder forward pass (all layers).
    ///
    /// `input`: [batch_size, seq_len, hidden_size] — input hidden states
    /// `positions`: [batch_size, seq_len] — position indices for RoPE
    /// `kv_cache`: mutable KV cache
    /// `weights`: model weights on device
    /// `seq_lens`: per-sequence lengths in the batch
    /// `output`: [batch_size, seq_len, vocab_size] — output logits
    fn decoder_forward(
        &self,
        input: &DeviceTensor,
        positions: &DeviceTensor,
        kv_cache: &mut KvCache,
        weights: &ModelWeights,
        seq_lens: &[usize],
        output: &mut DeviceTensor,
    ) -> Result<(), InferenceError>;

    /// Run encoder forward pass (all layers).
    fn encoder_forward(
        &self,
        input: &DeviceTensor,
        positions: &DeviceTensor,
        attention_mask: &DeviceTensor,
        weights: &ModelWeights,
        output: &mut DeviceTensor,
    ) -> Result<(), InferenceError>;

    /// Sample output token IDs from logits.
    fn sample(
        &self,
        logits: &DeviceTensor,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        output_ids: &mut [u32],
    ) -> Result<(), InferenceError>;

    /// Synchronize all pending operations.
    fn sync(&self) -> Result<(), InferenceError>;
}
