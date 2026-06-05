//! Layer 2: Inference Backend — high-level inference interface.
//!
//! `InferenceBackend` is the primary trait for running transformer inference.
//! `CpuInferenceBackend` provides the reference implementation using Layer 1
//! atomic operators as fallback, with JIT-compiled fused layers when available.

pub mod tensor;
pub mod weights;
pub mod kv_cache;
pub mod cpu_backend;
pub mod flash_attn;

pub use crate::types::{DType, ModelArch, ModelConfig, InferenceError};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelArch;

    /// Helper: minimal valid model config for testing.
    fn test_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            intermediate_size: 64,
            num_layers: 1,
            vocab_size: 50,
            max_seq_len: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    // ── CpuInferenceBackend::init + device_kind ────────────────────────

    #[test]
    fn test_cpu_backend_init_returns_cpu_device_kind() {
        // Arrange
        let config = test_config();

        // Act
        let backend = CpuInferenceBackend::init(&config).unwrap();

        // Assert
        assert_eq!(backend.device_kind(), DeviceKind::Cpu);
    }

    // ── CpuInferenceBackend::alloc ─────────────────────────────────────

    #[test]
    fn test_cpu_backend_alloc_creates_correct_tensor() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();

        // Act
        let tensor = backend.alloc(128, DType::F32).unwrap();

        // Assert
        assert_eq!(tensor.num_elements(), 128);
        assert_eq!(tensor.len_bytes(), 128 * 4);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_cpu());
    }

    // ── CpuInferenceBackend upload/download round-trip ─────────────────

    #[test]
    fn test_cpu_backend_upload_download_roundtrip() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();
        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut tensor = backend.alloc(16, DType::F32).unwrap();

        // Act: upload
        backend.upload_f32(&original, &mut tensor).unwrap();

        // Act: download
        let mut downloaded = vec![0.0f32; 16];
        backend.download_f32(&tensor, &mut downloaded).unwrap();

        // Assert
        assert_eq!(downloaded, original);
    }

    // ── upload_f32 rejects wrong dtype ─────────────────────────────────

    #[test]
    fn test_upload_f32_rejects_non_f32_tensor() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();
        let mut tensor = backend.alloc(4, DType::F16).unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        // Act
        let result = backend.upload_f32(&data, &mut tensor);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }
    }

    // ── upload_f32 rejects undersized tensor ───────────────────────────

    #[test]
    fn test_upload_f32_rejects_undersized_tensor() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();
        let mut tensor = backend.alloc(2, DType::F32).unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 elements, tensor only has 2

        // Act
        let result = backend.upload_f32(&data, &mut tensor);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }
    }

    // ── download_f32 rejects wrong dtype ───────────────────────────────

    #[test]
    fn test_download_f32_rejects_non_f32_tensor() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();
        let tensor = backend.alloc(4, DType::F16).unwrap();
        let mut buf = vec![0.0f32; 4];

        // Act
        let result = backend.download_f32(&tensor, &mut buf);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }
    }

    // ── download_f32 rejects undersized buffer ─────────────────────────

    #[test]
    fn test_download_f32_rejects_undersized_buffer() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();
        let tensor = backend.alloc(8, DType::F32).unwrap();
        let mut buf = vec![0.0f32; 4]; // buffer too small for 8 elements

        // Act
        let result = backend.download_f32(&tensor, &mut buf);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }
    }

    // ── CpuInferenceBackend::alloc_kv_cache ────────────────────────────

    #[test]
    fn test_cpu_backend_alloc_kv_cache() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();

        // Act
        let cache = backend.alloc_kv_cache(2, 32).unwrap();

        // Assert
        assert!(cache.total_pages() > 0);
        assert_eq!(cache.free_page_count(), cache.total_pages());
        assert_eq!(cache.num_kv_heads(), config.num_kv_heads);
        assert_eq!(cache.head_dim(), config.head_dim);
    }

    // ── ModelWeights::alloc_cpu produces correct layer count ───────────

    #[test]
    fn test_model_weights_alloc_cpu_layer_count() {
        // Arrange
        let config = test_config(); // num_layers=1

        // Act
        let weights = ModelWeights::alloc_cpu(&config).unwrap();

        // Assert
        assert_eq!(weights.num_layers(), 1);
        assert_eq!(weights.layers.len(), 1);
    }

    // ── ModelWeights::alloc_cpu embedding and lm_head sizes ────────────

    #[test]
    fn test_model_weights_alloc_cpu_tensor_sizes() {
        // Arrange
        let config = test_config(); // hidden=32, vocab=50, heads=4, kv=4, head_dim=8, inter=64

        // Act
        let weights = ModelWeights::alloc_cpu(&config).unwrap();

        // Assert: embedding = [vocab_size, hidden_size]
        assert_eq!(weights.embedding.num_elements(), config.vocab_size * config.hidden_size);
        // Assert: lm_head = [hidden_size, vocab_size]
        assert_eq!(weights.lm_head.num_elements(), config.hidden_size * config.vocab_size);
        // Assert: final_norm = [hidden_size]
        assert_eq!(weights.final_norm.num_elements(), config.hidden_size);
    }

    // ── ModelWeights::alloc_cpu per-layer weight sizes ─────────────────

    #[test]
    fn test_model_weights_alloc_cpu_layer_weight_sizes() {
        // Arrange
        let config = test_config(); // hidden=32, heads=4, kv=4, head_dim=8, inter=64, no qkv bias

        // Act
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let layer = &weights.layers[0];

        // Assert
        assert_eq!(layer.attn_norm.num_elements(), config.hidden_size);
        assert_eq!(layer.wq.num_elements(), config.hidden_size * config.num_heads * config.head_dim);
        assert_eq!(layer.wk.num_elements(), config.hidden_size * config.num_kv_heads * config.head_dim);
        assert_eq!(layer.wv.num_elements(), config.hidden_size * config.num_kv_heads * config.head_dim);
        assert_eq!(layer.wo.num_elements(), config.num_heads * config.head_dim * config.hidden_size);
        assert_eq!(layer.w_gate.num_elements(), config.hidden_size * config.intermediate_size);
        assert_eq!(layer.w_up.num_elements(), config.hidden_size * config.intermediate_size);
        assert_eq!(layer.w_down.num_elements(), config.intermediate_size * config.hidden_size);
        // No QKV bias for Llama
        assert!(layer.qkv_bias.is_none());
        // No norm bias for Llama (RMSNorm)
        assert_eq!(layer.attn_norm_bias.num_elements(), 0);
        assert_eq!(layer.ffn_norm_bias.num_elements(), 0);
    }

    // ── DeviceKind equality and Copy ───────────────────────────────────

    #[test]
    fn test_device_kind_equality_and_variants() {
        // Arrange & Act
        let cpu = DeviceKind::Cpu;
        let cuda0 = DeviceKind::Cuda(0);
        let cuda1 = DeviceKind::Cuda(1);
        let metal0 = DeviceKind::Metal(0);

        // Assert
        assert_eq!(cpu, DeviceKind::Cpu);
        assert_ne!(cpu, cuda0);
        assert_ne!(cuda0, cuda1);
        assert_ne!(cuda0, metal0);
        // DeviceKind is Copy
        let copied = cpu;
        assert_eq!(copied, DeviceKind::Cpu);
    }

    // ── InferenceBackend trait object safety smoke test ────────────────

    #[test]
    fn test_inference_backend_dyn_dispatch() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();

        // Act: use as a trait object to verify object safety
        let backend_ref: &dyn InferenceBackend = &backend;

        // Assert
        assert_eq!(backend_ref.device_kind(), DeviceKind::Cpu);
        // Verify sync works through trait object
        backend_ref.sync().unwrap();
    }

    // ── Re-exported types are accessible from inference module ──────────

    #[test]
    fn test_reexports_are_accessible() {
        // This test verifies that all re-exported types from mod.rs are usable.
        // If any re-export is missing, this will fail to compile.

        // Arrange
        let config = test_config();

        // Act & Assert: DType
        let _dtype: DType = DType::F32;

        // Act & Assert: ModelConfig (already used above)
        let _config: &ModelConfig = &config;

        // Act & Assert: DeviceKind
        let _kind: DeviceKind = DeviceKind::Cpu;

        // Act & Assert: DeviceTensor
        let _tensor: DeviceTensor = DeviceTensor::alloc_cpu(10, DType::F32).unwrap();

        // Act & Assert: KvCache
        let _cache: KvCache = KvCache::new(&config, 1, 16).unwrap();

        // Act & Assert: FlashAttnConfig
        let _attn_config: FlashAttnConfig = FlashAttnConfig::new(8, 4, 4);
    }

    // ── FlashAttnConfig::new computes correct scale ────────────────────

    #[test]
    fn test_flash_attn_config_new_scale() {
        // Arrange: head_dim=64
        let head_dim = 64;

        // Act
        let cfg = FlashAttnConfig::new(head_dim, 8, 2);

        // Assert: scale = 1/sqrt(head_dim) = 1/8 = 0.125
        let expected_scale = 1.0 / (head_dim as f32).sqrt();
        assert!((cfg.scale - expected_scale).abs() < 1e-6);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 2);
        // Default tile_kv is 256
        assert_eq!(cfg.tile_kv, 256);
    }

    // ── FlashAttnConfig::with_cache_hint derives tile_kv from L1 ───────

    #[test]
    fn test_flash_attn_config_with_cache_hint_tile_kv() {
        // Arrange: 32 KiB L1, head_dim=64
        // bytes_per_kv_row = 64 * 4 * 2 = 512 bytes (K+V per row)
        // max_tile = (32 KiB / 2) / 512 = 32
        // next_power_of_two(32) = 32, >> 1 = 16
        // clamp [16, 512] => 16
        let l1_bytes = 32 * 1024;

        // Act
        let cfg = FlashAttnConfig::with_cache_hint(64, 8, 4, l1_bytes);

        // Assert
        assert_eq!(cfg.tile_kv, 16);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 4);
    }

    // ── CpuInferenceBackend allocates F16 tensor ───────────────────────

    #[test]
    fn test_cpu_backend_alloc_f16_tensor() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();

        // Act
        let tensor = backend.alloc(64, DType::F16).unwrap();

        // Assert
        assert_eq!(tensor.num_elements(), 64);
        assert_eq!(tensor.len_bytes(), 128); // 64 * 2 bytes
        assert_eq!(tensor.dtype(), DType::F16);
        assert!(tensor.is_cpu());
    }

    // ── CpuInferenceBackend sync succeeds ──────────────────────────────

    #[test]
    fn test_cpu_backend_sync_succeeds() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();

        // Act
        let result = backend.sync();

        // Assert: CPU sync is a no-op that always succeeds
        assert!(result.is_ok());
    }

    // ── ModelWeights with Gpt2 arch has norm bias vectors ──────────────

    #[test]
    fn test_model_weights_gpt2_arch_norm_bias_sizes() {
        // Arrange: GPT-2 arch triggers LayerNorm bias allocation
        let mut config = test_config();
        config.arch = ModelArch::Gpt2;

        // Act
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let layer = &weights.layers[0];

        // Assert: norm bias vectors have hidden_size elements
        assert_eq!(layer.attn_norm_bias.num_elements(), config.hidden_size);
        assert_eq!(layer.ffn_norm_bias.num_elements(), config.hidden_size);
    }

    // ── ModelWeights with has_qkv_bias allocates bias tensor ───────────

    #[test]
    fn test_model_weights_qkv_bias_allocates_correct_size() {
        // Arrange: Qwen-style model with QKV bias
        let mut config = test_config(); // hidden=32, heads=4, kv_heads=4, head_dim=8
        config.has_qkv_bias = true;
        let q_dim = config.num_heads * config.head_dim;   // 32
        let kv_dim = config.num_kv_heads * config.head_dim; // 32
        let expected_bias_elems = q_dim + 2 * kv_dim;     // 96

        // Act
        let weights = ModelWeights::alloc_cpu(&config).unwrap();

        // Assert
        let bias = weights.layers[0].qkv_bias.as_ref().expect("qkv_bias should exist");
        assert_eq!(bias.num_elements(), expected_bias_elems);
    }

    // ── InferenceError Display formats correctly ───────────────────────

    #[test]
    fn test_inference_error_display_variants() {
        // Arrange & Act
        let oom = InferenceError::OutOfMemory { requested: 1024, available: 512 };
        let shape = InferenceError::ShapeMismatch {
            expected: "32 bytes".to_string(),
            got: "16 bytes".to_string(),
        };
        let runtime = InferenceError::RuntimeError("bad thing".to_string());

        // Assert
        let oom_str = format!("{oom}");
        assert!(oom_str.contains("1024") && oom_str.contains("512"), "OOM display: {oom_str}");

        let shape_str = format!("{shape}");
        assert!(shape_str.contains("32 bytes") && shape_str.contains("16 bytes"), "ShapeMismatch display: {shape_str}");

        let rt_str = format!("{runtime}");
        assert!(rt_str.contains("bad thing"), "RuntimeError display: {rt_str}");
    }

    // ── CpuInferenceBackend upload/download with zero-length data ──────

    #[test]
    fn test_cpu_backend_upload_download_zero_length() {
        // Arrange
        let config = test_config();
        let backend = CpuInferenceBackend::init(&config).unwrap();
        let mut tensor = backend.alloc(0, DType::F32).unwrap();

        // Act: upload empty slice
        let src: Vec<f32> = vec![];
        backend.upload_f32(&src, &mut tensor).unwrap();

        // Act: download into empty buffer
        let mut dst: Vec<f32> = vec![];
        backend.download_f32(&tensor, &mut dst).unwrap();

        // Assert
        assert!(dst.is_empty());
    }

    // ── DeviceKind Debug output contains variant name ──────────────────

    #[test]
    fn test_device_kind_debug_output() {
        // Arrange & Act
        let cpu = format!("{:?}", DeviceKind::Cpu);
        let cuda3 = format!("{:?}", DeviceKind::Cuda(3));
        let metal7 = format!("{:?}", DeviceKind::Metal(7));

        // Assert: debug output includes variant name and payload
        assert!(cpu.contains("Cpu"), "Cpu debug: {cpu}");
        assert!(cuda3.contains("Cuda"), "Cuda debug: {cuda3}");
        assert!(cuda3.contains("3"), "Cuda(3) should contain '3': {cuda3}");
        assert!(metal7.contains("Metal"), "Metal debug: {metal7}");
        assert!(metal7.contains("7"), "Metal(7) should contain '7': {metal7}");
    }

    // ── CpuInferenceBackend alloc_kv_cache with batch_size > 1 ─────────

    #[test]
    fn test_cpu_backend_alloc_kv_cache_batch_size() {
        // Arrange
        let config = test_config(); // hidden=32, heads=4, kv_heads=4, head_dim=8, layers=1

        // Act: batch_size=3
        let cache = CpuInferenceBackend::init(&config)
            .unwrap()
            .alloc_kv_cache(3, 16)
            .unwrap();

        // Assert: cache has pages allocated for all 3 sequences
        assert!(cache.total_pages() > 0);
        assert_eq!(cache.free_page_count(), cache.total_pages());
        assert_eq!(cache.num_kv_heads(), config.num_kv_heads);
        assert_eq!(cache.head_dim(), config.head_dim);
    }
}
