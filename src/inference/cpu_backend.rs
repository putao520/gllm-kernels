//! CPU inference backend — fallback path using Layer 1 atomic operators.
//!
//! This is the reference implementation of `InferenceBackend`. It composes
//! the existing `Kernels<E>` operators to implement full transformer layers.
//!
//! When the JIT compiler is available, `decoder_forward` delegates to the
//! compiled layer function. Otherwise it falls back to operator-by-operator
//! execution through this module.

use crate::cpu_kernels::CpuKernels;
use crate::dispatch::{DeviceProfile, device_profile};
use crate::inference::types::{DType, InferenceError, ModelConfig};
use crate::inference::tensor::{DeviceKind, DeviceTensor};
use crate::inference::weights::ModelWeights;
use crate::inference::kv_cache::KvCache;
use crate::inference::InferenceBackend;
use crate::traits::Kernels;

/// CPU inference backend with fallback operator composition.
pub struct CpuInferenceBackend {
    config: ModelConfig,
    profile: DeviceProfile,
    kernels: CpuKernels<f32>,
    /// Pre-allocated scratchpad for intermediate results
    scratchpad: Vec<f32>,
}

impl InferenceBackend for CpuInferenceBackend {
    fn init(config: &ModelConfig) -> Result<Self, InferenceError> {
        let profile = device_profile().clone();

        // Scratchpad: enough for the largest intermediate tensor in a single layer
        // QKV projection output: batch=1, seq=1 → 3 * hidden_size
        // FFN intermediate: 2 * intermediate_size (gate + up)
        // Attention scores: num_heads * max_seq_len
        let scratch_size = (3 * config.hidden_size)
            .max(2 * config.intermediate_size)
            .max(config.num_heads * config.max_seq_len)
            * 4; // generous padding
        let scratchpad = vec![0.0f32; scratch_size];

        Ok(CpuInferenceBackend {
            config: config.clone(),
            profile,
            kernels: CpuKernels::new(),
            scratchpad,
        })
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, num_elements: usize, dtype: DType) -> Result<DeviceTensor, InferenceError> {
        DeviceTensor::alloc_cpu(num_elements, dtype)
    }

    fn upload_f32(&self, src: &[f32], dst: &mut DeviceTensor) -> Result<(), InferenceError> {
        if dst.num_elements() < src.len() {
            return Err(InferenceError::ShapeMismatch {
                expected: format!("{} elements", src.len()),
                got: format!("{} elements", dst.num_elements()),
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *const u8,
                dst.as_mut_ptr(),
                src.len() * 4,
            );
        }
        Ok(())
    }

    fn download_f32(&self, src: &DeviceTensor, dst: &mut [f32]) -> Result<(), InferenceError> {
        if dst.len() < src.num_elements() {
            return Err(InferenceError::ShapeMismatch {
                expected: format!("{} elements", src.num_elements()),
                got: format!("{} elements", dst.len()),
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr() as *mut u8,
                src.num_elements() * 4,
            );
        }
        Ok(())
    }

    fn alloc_kv_cache(
        &self,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<KvCache, InferenceError> {
        KvCache::new(&self.config, batch_size, max_seq_len)
    }

    fn decoder_forward(
        &self,
        input: &DeviceTensor,
        _positions: &DeviceTensor,
        _kv_cache: &mut KvCache,
        weights: &ModelWeights,
        _seq_lens: &[usize],
        output: &mut DeviceTensor,
    ) -> Result<(), InferenceError> {
        // Fallback path: operator-by-operator execution
        // For now, implement a simplified single-token, single-batch path
        let h = self.config.hidden_size;
        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let inter = self.config.intermediate_size;

        let input_slice: &[f32] = unsafe { input.as_slice() };

        // Working buffers
        let mut hidden = input_slice.to_vec();
        let mut normed = vec![0.0f32; h];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut o_proj = vec![0.0f32; h];
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut act = vec![0.0f32; inter];
        let mut down = vec![0.0f32; h];

        for layer_idx in 0..self.config.num_layers {
            let lw = &weights.layers[layer_idx];

            // 1. RMSNorm
            let norm_w: &[f32] = unsafe { lw.attn_norm.as_slice() };
            self.kernels.rms_norm(&hidden, norm_w, &mut normed, self.config.norm_eps);

            // 2. QKV projections
            let wq: &[f32] = unsafe { lw.wq.as_slice() };
            let wk: &[f32] = unsafe { lw.wk.as_slice() };
            let wv: &[f32] = unsafe { lw.wv.as_slice() };
            self.kernels.gemm(&normed, wq, &mut q, 1, q_dim, h);
            self.kernels.gemm(&normed, wk, &mut k, 1, kv_dim, h);
            self.kernels.gemm(&normed, wv, &mut v, 1, kv_dim, h);

            // 3. RoPE would be applied here (skipped in fallback for now)
            // 4. Attention computation (simplified: skip KV cache for initial impl)

            // 5. Output projection
            let wo: &[f32] = unsafe { lw.wo.as_slice() };
            // For simplified path: attn_out = q (placeholder)
            attn_out.copy_from_slice(&q[..q_dim]);
            self.kernels.gemm(&attn_out, wo, &mut o_proj, 1, h, q_dim);

            // 6. Residual add
            self.kernels.vec_add(&hidden, &o_proj, &mut hidden.clone());
            let residual = hidden.clone();

            // 7. FFN RMSNorm
            let ffn_norm_w: &[f32] = unsafe { lw.ffn_norm.as_slice() };
            self.kernels.rms_norm(&hidden, ffn_norm_w, &mut normed, self.config.norm_eps);

            // 8. FFN: gate + up + SiLU + down
            let wg: &[f32] = unsafe { lw.w_gate.as_slice() };
            let wu: &[f32] = unsafe { lw.w_up.as_slice() };
            let wd: &[f32] = unsafe { lw.w_down.as_slice() };
            self.kernels.gemm(&normed, wg, &mut gate, 1, inter, h);
            self.kernels.gemm(&normed, wu, &mut up, 1, inter, h);
            self.kernels.silu(&gate, &mut act);
            self.kernels.vec_mul(&act, &up, &mut gate);
            self.kernels.gemm(&gate, wd, &mut down, 1, h, inter);

            // 9. Residual add
            self.kernels.vec_add(&residual, &down, &mut hidden);
        }

        // Copy to output
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr() as *const u8,
                output.as_mut_ptr() as *mut u8,
                h * 4,
            );
        }

        Ok(())
    }

    fn encoder_forward(
        &self,
        _input: &DeviceTensor,
        _positions: &DeviceTensor,
        _attention_mask: &DeviceTensor,
        _weights: &ModelWeights,
        _output: &mut DeviceTensor,
    ) -> Result<(), InferenceError> {
        Err(InferenceError::Unsupported(
            "encoder forward not yet implemented in CPU fallback".into(),
        ))
    }

    fn sample(
        &self,
        logits: &DeviceTensor,
        temperature: f32,
        _top_k: usize,
        _top_p: f32,
        output_ids: &mut [u32],
    ) -> Result<(), InferenceError> {
        // Simple argmax sampling (temperature=0) or greedy
        let logits_slice: &[f32] = unsafe { logits.as_slice() };
        let vocab_size = self.config.vocab_size;

        for (batch_idx, out_id) in output_ids.iter_mut().enumerate() {
            let start = batch_idx * vocab_size;
            let end = start + vocab_size;
            if end > logits_slice.len() {
                return Err(InferenceError::ShapeMismatch {
                    expected: format!("logits for batch {batch_idx}"),
                    got: "insufficient logits".into(),
                });
            }
            let batch_logits = &logits_slice[start..end];

            if temperature <= 0.0 || temperature < 1e-6 {
                // Argmax
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0u32;
                for (i, &v) in batch_logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i as u32;
                    }
                }
                *out_id = max_idx;
            } else {
                // Temperature-scaled argmax (full sampling requires RNG)
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0u32;
                for (i, &v) in batch_logits.iter().enumerate() {
                    let scaled = v / temperature;
                    if scaled > max_val {
                        max_val = scaled;
                        max_idx = i as u32;
                    }
                }
                *out_id = max_idx;
            }
        }

        Ok(())
    }

    fn sync(&self) -> Result<(), InferenceError> {
        // CPU is synchronous — no-op
        Ok(())
    }
}

/// Helper: unsafe mutable slice access without borrow checker issues.
/// Only valid for CPU tensors.
unsafe fn tensor_as_mut_f32(t: &mut DeviceTensor) -> &mut [f32] {
    std::slice::from_raw_parts_mut(t.as_mut_ptr() as *mut f32, t.num_elements())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::{ModelArch, ModelConfig};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 2,
            vocab_size: 100,
            max_seq_len: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
        }
    }

    #[test]
    fn test_cpu_backend_init() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        assert_eq!(backend.device_kind(), DeviceKind::Cpu);
    }

    #[test]
    fn test_cpu_backend_alloc() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let t = backend.alloc(256, DType::F32).unwrap();
        assert_eq!(t.num_elements(), 256);
    }

    #[test]
    fn test_cpu_backend_upload_download() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut t = backend.alloc(4, DType::F32).unwrap();
        backend.upload_f32(&src, &mut t).unwrap();
        let mut dst = vec![0.0f32; 4];
        backend.download_f32(&t, &mut dst).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn test_cpu_backend_sample_argmax() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut logits_data = vec![0.0f32; 100];
        logits_data[42] = 10.0; // token 42 has highest logit
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 1];
        backend.sample(&logits, 0.0, 0, 0.0, &mut ids).unwrap();
        assert_eq!(ids[0], 42);
    }
}
