//! CPU inference backend — fallback path using Layer 1 atomic operators.
//!
//! This is the reference implementation of `InferenceBackend`. It composes
//! the existing `Kernels<E>` operators to implement full transformer layers.
//!
//! When the JIT compiler is available, `decoder_forward` delegates to the
//! compiled layer function. Otherwise it falls back to operator-by-operator
//! execution through this module.

use crate::compiler::InferenceCompiler;
use crate::cpu_kernels::CpuKernels;
use crate::dispatch::{DeviceProfile, device_profile};
use crate::inference::types::{DType, InferenceError, ModelConfig};
use crate::inference::tensor::{DeviceKind, DeviceTensor};
use crate::inference::weights::ModelWeights;
use crate::inference::kv_cache::{KvCache, PAGE_SIZE};
use crate::inference::InferenceBackend;
use crate::traits::Kernels;

/// CPU inference backend with fallback operator composition.
///
/// When the JIT compiler is available, `decoder_forward` attempts to use
/// compiled fused layers. Falls back to operator-by-operator execution
/// through Layer 1 kernels when JIT is unavailable or compilation fails.
pub struct CpuInferenceBackend {
    config: ModelConfig,
    profile: DeviceProfile,
    kernels: CpuKernels<f32>,
    /// Optional JIT compiler for fused layer execution
    compiler: Option<InferenceCompiler>,
}

impl InferenceBackend for CpuInferenceBackend {
    fn init(config: &ModelConfig) -> Result<Self, InferenceError> {
        let profile = device_profile().clone();

        let compiler = Some(InferenceCompiler::with_profile(profile.clone()));

        Ok(CpuInferenceBackend {
            config: config.clone(),
            profile,
            kernels: CpuKernels::new(),
            compiler,
        })
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, num_elements: usize, dtype: DType) -> Result<DeviceTensor, InferenceError> {
        DeviceTensor::alloc_cpu(num_elements, dtype)
    }

    fn upload_f32(&self, src: &[f32], dst: &mut DeviceTensor) -> Result<(), InferenceError> {
        if dst.dtype() != DType::F32 {
            return Err(InferenceError::ShapeMismatch {
                expected: "F32 tensor".to_string(),
                got: format!("{:?} tensor", dst.dtype()),
            });
        }
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
        if src.dtype() != DType::F32 {
            return Err(InferenceError::ShapeMismatch {
                expected: "F32 tensor".to_string(),
                got: format!("{:?} tensor", src.dtype()),
            });
        }
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
        positions: &DeviceTensor,
        kv_cache: &mut KvCache,
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

            // 2b. Apply QKV bias if present (Qwen has bias, Llama does not)
            if let Some(ref bias) = lw.qkv_bias {
                let bias_slice: &[f32] = unsafe { bias.as_slice() };
                for i in 0..q_dim { q[i] += bias_slice[i]; }
                for i in 0..kv_dim { k[i] += bias_slice[q_dim + i]; }
                for i in 0..kv_dim { v[i] += bias_slice[q_dim + kv_dim + i]; }
            }

            // 3. RoPE (with partial rotary support for Phi)
            let head_dim = self.config.head_dim;
            let rotary_dim = ((head_dim as f32 * self.config.partial_rotary_factor) as usize) & !1;
            let half = rotary_dim / 2;
            let mut cos_table = vec![0.0f32; half];
            let mut sin_table = vec![0.0f32; half];
            let positions_slice: &[f32] = unsafe { positions.as_slice() };
            let token_pos = positions_slice.get(0).copied().unwrap_or(0.0) as usize;
            let pos = token_pos as f64;
            for i in 0..half {
                let freq = 1.0 / (self.config.rope_theta).powf(2.0 * i as f64 / rotary_dim as f64);
                let angle = pos * freq;
                cos_table[i] = angle.cos() as f32;
                sin_table[i] = angle.sin() as f32;
            }
            apply_rope_inplace(&mut q, &cos_table, &sin_table, self.config.num_heads, head_dim, self.config.partial_rotary_factor);
            apply_rope_inplace(&mut k, &cos_table, &sin_table, self.config.num_kv_heads, head_dim, self.config.partial_rotary_factor);

            // 4. Attention computation with KV cache + causal mask
            let num_heads = self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let heads_per_kv = num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Append current K/V into the paged KV cache
            let kv_positions = kv_cache.append(layer_idx, 0, 1)?;
            let (page_id, offset_in_page) = kv_positions[0];

            // Write K/V into cache page
            // Page layout: [2 (K+V), num_kv_heads, PAGE_SIZE, head_dim] as f32
            let v_base = num_kv_heads * PAGE_SIZE * head_dim;
            let page_ptr = kv_cache.page_mut_ptr(page_id) as *mut f32;
            for kv_h in 0..num_kv_heads {
                let head_base = kv_h * PAGE_SIZE * head_dim + offset_in_page * head_dim;
                for d in 0..head_dim {
                    unsafe {
                        *page_ptr.add(head_base + d) = k[kv_h * head_dim + d];
                        *page_ptr.add(v_base + head_base + d) = v[kv_h * head_dim + d];
                    }
                }
            }

            // Multi-head attention with causal mask
            let cached_len = kv_cache.seq_len(layer_idx, 0);
            let seq_pages = kv_cache.seq_pages(layer_idx, 0);

            for ah in 0..num_heads {
                let kv_h = ah / heads_per_kv;
                let q_off = ah * head_dim;

                // Compute scores: Q[h] · K[kv_h][t] for all cached positions
                let mut scores = vec![0.0f32; cached_len];
                for t in 0..cached_len {
                    let pid = seq_pages[t / PAGE_SIZE];
                    let off = t % PAGE_SIZE;
                    let kp = kv_cache.page_ptr(pid) as *const f32;
                    let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * unsafe { *kp.add(k_base + d) };
                    }
                    scores[t] = dot * scale;
                }

                // Causal mask: zero out future positions
                // + sliding window: zero out positions older than W
                for t in 0..cached_len {
                    if t > token_pos {
                        scores[t] = f32::NEG_INFINITY;
                    } else if let Some(w) = self.config.sliding_window {
                        // Sliding window: only attend to positions in [q - W + 1, q]
                        if token_pos >= w && t < token_pos - w + 1 {
                            scores[t] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for t in 0..cached_len {
                        let pid = seq_pages[t / PAGE_SIZE];
                        let off = t % PAGE_SIZE;
                        let vp = kv_cache.page_ptr(pid) as *const f32;
                        let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim + d;
                        val += scores[t] * unsafe { *vp.add(v_off) };
                    }
                    attn_out[q_off + d] = val;
                }
            }

            // 5. Output projection
            let wo: &[f32] = unsafe { lw.wo.as_slice() };
            self.kernels.gemm(&attn_out, wo, &mut o_proj, 1, h, q_dim);

            // 6. Residual add
            let mut residual = vec![0.0f32; h];
            self.kernels.vec_add(&hidden, &o_proj, &mut residual);
            hidden.copy_from_slice(&residual);

            // 7. FFN RMSNorm
            let ffn_norm_w: &[f32] = unsafe { lw.ffn_norm.as_slice() };
            self.kernels.rms_norm(&hidden, ffn_norm_w, &mut normed, self.config.norm_eps);

            // 8. FFN: gate + up + activation + down
            let wg: &[f32] = unsafe { lw.w_gate.as_slice() };
            let wu: &[f32] = unsafe { lw.w_up.as_slice() };
            let wd: &[f32] = unsafe { lw.w_down.as_slice() };
            self.kernels.gemm(&normed, wg, &mut gate, 1, inter, h);
            self.kernels.gemm(&normed, wu, &mut up, 1, inter, h);
            match self.config.arch {
                crate::inference::types::ModelArch::Gemma => self.kernels.gelu(&gate, &mut act),
                _ => self.kernels.silu(&gate, &mut act),
            }
            self.kernels.vec_mul(&act, &up, &mut gate);
            self.kernels.gemm(&gate, wd, &mut down, 1, h, inter);

            // 9. Residual add
            self.kernels.vec_add(&residual, &down, &mut hidden);
        }

        // 10. Final RMSNorm
        let final_norm_w: &[f32] = unsafe { weights.final_norm.as_slice() };
        let mut final_normed = vec![0.0f32; h];
        self.kernels.rms_norm(&hidden, final_norm_w, &mut final_normed, self.config.norm_eps);

        // 11. LM head projection (hidden_size -> vocab_size)
        let lm_head_w: &[f32] = unsafe { weights.lm_head.as_slice() };
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        self.kernels.gemm(&final_normed, lm_head_w, &mut logits, 1, vocab, h);

        // Copy logits to output
        let out_elems = output.num_elements().min(vocab);
        unsafe {
            std::ptr::copy_nonoverlapping(
                logits.as_ptr() as *const u8,
                output.as_mut_ptr() as *mut u8,
                out_elems * 4,
            );
        }

        Ok(())
    }

    fn encoder_forward(
        &self,
        input: &DeviceTensor,
        _positions: &DeviceTensor,
        attention_mask: &DeviceTensor,
        weights: &ModelWeights,
        output: &mut DeviceTensor,
    ) -> Result<(), InferenceError> {
        let h = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter = self.config.intermediate_size;
        let eps = self.config.norm_eps;
        let seq_len = input.num_elements() / h;

        if seq_len == 0 {
            return Ok(());
        }

        let input_slice: &[f32] = unsafe { input.as_slice() };
        let mask_slice: &[f32] = unsafe { attention_mask.as_slice() };

        // Working buffers
        let mut hidden = input_slice.to_vec();
        let mut normed = vec![0.0f32; seq_len * h];
        let mut q = vec![0.0f32; seq_len * q_dim];
        let mut k = vec![0.0f32; seq_len * kv_dim];
        let mut v = vec![0.0f32; seq_len * kv_dim];
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        let mut proj = vec![0.0f32; seq_len * h];
        let mut up_buf = vec![0.0f32; seq_len * inter];
        let mut gelu_buf = vec![0.0f32; seq_len * inter];
        let mut down_buf = vec![0.0f32; seq_len * h];
        let mut residual = vec![0.0f32; seq_len * h];

        // Per-head attention scratch
        let mut scores = vec![0.0f32; seq_len * seq_len];
        let mut attn_w = vec![0.0f32; seq_len * seq_len];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_group_size = num_heads / num_kv_heads;

        for layer_idx in 0..self.config.num_layers {
            let lw = &weights.layers[layer_idx];

            // --- 1. Pre-attention LayerNorm ---
            let gamma: &[f32] = unsafe { lw.attn_norm.as_slice() };
            let beta: &[f32] = unsafe { lw.attn_norm_bias.as_slice() };
            for s in 0..seq_len {
                let src = &hidden[s * h..(s + 1) * h];
                let dst = &mut normed[s * h..(s + 1) * h];
                self.kernels.layer_norm(src, gamma, beta, dst, eps);
            }

            // --- 2. QKV projections: [seq_len, h] @ W → [seq_len, dim] ---
            let wq: &[f32] = unsafe { lw.wq.as_slice() };
            let wk: &[f32] = unsafe { lw.wk.as_slice() };
            let wv: &[f32] = unsafe { lw.wv.as_slice() };
            self.kernels.gemm(&normed, wq, &mut q, seq_len, q_dim, h);
            self.kernels.gemm(&normed, wk, &mut k, seq_len, kv_dim, h);
            self.kernels.gemm(&normed, wv, &mut v, seq_len, kv_dim, h);

            // --- 3. Multi-head attention (no RoPE, no KV cache) ---
            for head in 0..num_heads {
                let kv_head = head / kv_group_size;
                let q_off = head * head_dim;
                let kv_off = kv_head * head_dim;

                // scores[si, sj] = sum_d Q_h[si,d] * K_h[sj,d] * scale
                for si in 0..seq_len {
                    for sj in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[si * q_dim + q_off + d]
                                 * k[sj * kv_dim + kv_off + d];
                        }
                        scores[si * seq_len + sj] = dot * scale;
                    }
                }

                // Apply attention mask: 0.0 → -inf
                for i in 0..seq_len * seq_len {
                    if mask_slice[i] == 0.0 {
                        scores[i] = f32::NEG_INFINITY;
                    }
                }

                // Softmax per query position
                for si in 0..seq_len {
                    let row = &scores[si * seq_len..(si + 1) * seq_len];
                    let out = &mut attn_w[si * seq_len..(si + 1) * seq_len];
                    self.kernels.softmax(row, out);
                }

                // attn_out_h[si, d] = sum_sj attn_w[si, sj] * V_h[sj, d]
                for si in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for sj in 0..seq_len {
                            sum += attn_w[si * seq_len + sj]
                                 * v[sj * kv_dim + kv_off + d];
                        }
                        attn_out[si * q_dim + q_off + d] = sum;
                    }
                }
            }

            // --- 4. Output projection + residual ---
            let wo: &[f32] = unsafe { lw.wo.as_slice() };
            self.kernels.gemm(&attn_out, wo, &mut proj, seq_len, h, q_dim);
            self.kernels.vec_add(&hidden, &proj, &mut residual);
            hidden.copy_from_slice(&residual);

            // --- 5. Pre-FFN LayerNorm ---
            let ffn_gamma: &[f32] = unsafe { lw.ffn_norm.as_slice() };
            let ffn_beta: &[f32] = unsafe { lw.ffn_norm_bias.as_slice() };
            for s in 0..seq_len {
                let src = &hidden[s * h..(s + 1) * h];
                let dst = &mut normed[s * h..(s + 1) * h];
                self.kernels.layer_norm(src, ffn_gamma, ffn_beta, dst, eps);
            }

            // --- 6. FFN: up → GELU → down + residual ---
            let wu: &[f32] = unsafe { lw.w_up.as_slice() };
            let wd: &[f32] = unsafe { lw.w_down.as_slice() };
            self.kernels.gemm(&normed, wu, &mut up_buf, seq_len, inter, h);
            self.kernels.gelu(&up_buf, &mut gelu_buf);
            self.kernels.gemm(&gelu_buf, wd, &mut down_buf, seq_len, h, inter);
            self.kernels.vec_add(&hidden, &down_buf, &mut residual);
            hidden.copy_from_slice(&residual);
        }

        // Copy result to output tensor
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr() as *const u8,
                output.as_mut_ptr(),
                seq_len * h * 4,
            );
        }

        Ok(())
    }

    /// Sample token IDs from logits.
    ///
    /// Current implementation: greedy argmax only. When temperature > 0,
    /// logits are temperature-scaled but selection is still argmax.
    /// `top_k` and `top_p` parameters are accepted but not yet implemented.
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

/// Apply rotary position embedding in-place.
///
/// Supports partial rotary (Phi-style): only the first `rotary_dim` dimensions
/// of each head are rotated; the rest pass through unchanged.
///
/// `qk` layout: `[num_heads, head_dim]` (single token).
fn apply_rope_inplace(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    num_heads: usize,
    head_dim: usize,
    partial_rotary_factor: f32,
) {
    let rotary_dim = ((head_dim as f32 * partial_rotary_factor) as usize) & !1;
    let half = rotary_dim / 2;

    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = qk[base + i];
            let x1 = qk[base + half + i];
            let c = cos[i];
            let s = sin[i];
            qk[base + i] = x0 * c - x1 * s;
            qk[base + half + i] = x0 * s + x1 * c;
        }
        // Dimensions [rotary_dim..head_dim] are untouched.
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
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
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

    fn tiny_qwen_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Qwen,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 2,
            vocab_size: 100,
            max_seq_len: 32,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: true,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    #[test]
    fn test_qwen_config_creation() {
        let cfg = ModelConfig::qwen_7b();
        assert_eq!(cfg.arch, ModelArch::Qwen);
        assert_eq!(cfg.vocab_size, 151936);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-6);
        assert!(cfg.has_qkv_bias);
    }

    #[test]
    fn test_decoder_forward_qwen_tiny() {
        use crate::inference::weights::ModelWeights;
        use crate::inference::InferenceBackend;

        let cfg = tiny_qwen_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        // Verify bias tensors were allocated
        for lw in &weights.layers {
            assert!(lw.qkv_bias.is_some());
            let bias = lw.qkv_bias.as_ref().unwrap();
            let q_dim = cfg.num_heads * cfg.head_dim;
            let kv_dim = cfg.num_kv_heads * cfg.head_dim;
            assert_eq!(bias.num_elements(), q_dim + 2 * kv_dim);
        }

        // Fill norm weights with 1.0 for stable RMSNorm
        let ones = vec![1.0f32; cfg.hidden_size];
        for lw in weights.layers.iter_mut() {
            unsafe {
                let norm: &mut [f32] = lw.attn_norm.as_mut_slice();
                norm.copy_from_slice(&ones);
                let ffn: &mut [f32] = lw.ffn_norm.as_mut_slice();
                ffn.copy_from_slice(&ones);
            }
        }

        let input_data = vec![0.1f32; cfg.hidden_size];
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions_data = vec![0.0f32; 1];
        let positions = unsafe { DeviceTensor::from_slice(&positions_data) };
        let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let mut output = backend.alloc(cfg.hidden_size, DType::F32).unwrap();

        let result = backend.decoder_forward(
            &input, &positions, &mut kv_cache, &weights, &[1], &mut output,
        );
        assert!(result.is_ok(), "decoder_forward failed: {:?}", result.err());
        assert_eq!(output.num_elements(), cfg.hidden_size);
    }

    #[test]
    fn test_apply_rope_partial() {
        // 2 heads, head_dim=8, partial_rotary_factor=0.5 → rotary_dim=4, half=2
        let num_heads = 2;
        let head_dim = 8;
        let partial_rotary_factor = 0.5f32;

        let cos = vec![0.5f32, 0.8];
        let sin = vec![0.6f32, 0.3];

        let mut qk: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let original = qk.clone();

        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, partial_rotary_factor);

        // Head 0: rotated part (dims 0..4)
        assert!((qk[0] - (1.0 * 0.5 - 3.0 * 0.6)).abs() < 1e-5);
        assert!((qk[1] - (2.0 * 0.8 - 4.0 * 0.3)).abs() < 1e-5);
        assert!((qk[2] - (1.0 * 0.6 + 3.0 * 0.5)).abs() < 1e-5);
        assert!((qk[3] - (2.0 * 0.3 + 4.0 * 0.8)).abs() < 1e-5);

        // Head 0: non-rotated part (dims 4..8) must be unchanged
        assert_eq!(qk[4], original[4]);
        assert_eq!(qk[5], original[5]);
        assert_eq!(qk[6], original[6]);
        assert_eq!(qk[7], original[7]);

        // Head 1: non-rotated part must be unchanged
        assert_eq!(qk[12], original[12]);
        assert_eq!(qk[13], original[13]);
        assert_eq!(qk[14], original[14]);
        assert_eq!(qk[15], original[15]);
    }

    #[test]
    fn test_apply_rope_full() {
        // 2 heads, head_dim=4, partial_rotary_factor=1.0 → all dims rotated
        let num_heads = 2;
        let head_dim = 4;

        let cos = vec![0.5f32, 0.8];
        let sin = vec![0.6f32, 0.3];

        let mut qk: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let original = qk.clone();

        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);

        // Head 0: all 4 dims rotated
        assert!((qk[0] - (1.0 * 0.5 - 3.0 * 0.6)).abs() < 1e-5);
        assert!((qk[1] - (2.0 * 0.8 - 4.0 * 0.3)).abs() < 1e-5);
        assert!((qk[2] - (1.0 * 0.6 + 3.0 * 0.5)).abs() < 1e-5);
        assert!((qk[3] - (2.0 * 0.3 + 4.0 * 0.8)).abs() < 1e-5);

        // No dimension should remain unchanged
        assert_ne!(qk[0], original[0]);
        assert_ne!(qk[2], original[2]);

        // Head 1: all 4 dims rotated
        assert!((qk[4] - (5.0 * 0.5 - 7.0 * 0.6)).abs() < 1e-5);
        assert!((qk[5] - (6.0 * 0.8 - 8.0 * 0.3)).abs() < 1e-5);
        assert!((qk[6] - (5.0 * 0.6 + 7.0 * 0.5)).abs() < 1e-5);
        assert!((qk[7] - (6.0 * 0.3 + 8.0 * 0.8)).abs() < 1e-5);
    }

    /// Test causal mask in attention: token 0 sees only itself,
    /// token 1 sees token 0 and itself, future positions are masked.
    #[test]
    fn test_causal_mask_attention() {
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // All-ones Q/K so raw dot = head_dim, scaled = sqrt(head_dim)
        let q = vec![1.0f32; head_dim];
        let k0 = vec![1.0f32; head_dim];
        let k1 = vec![1.0f32; head_dim];
        let v0 = vec![1.0f32; head_dim]; // value for position 0
        let v1 = vec![2.0f32; head_dim]; // value for position 1 (distinct)

        // Helper: compute softmax-weighted output with causal mask
        let attend = |q: &[f32],
                      keys: &[&[f32]],
                      vals: &[&[f32]],
                      token_pos: usize|
         -> (Vec<f32>, Vec<f32>) {
            let cached_len = keys.len();
            let mut scores = vec![0.0f32; cached_len];
            for t in 0..cached_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * keys[t][d];
                }
                scores[t] = dot * scale;
            }

            // Causal mask: zero out future positions
            for t in 0..cached_len {
                if t > token_pos {
                    scores[t] = f32::NEG_INFINITY;
                }
            }
            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            // Weighted V
            let mut out = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                for t in 0..cached_len {
                    out[d] += scores[t] * vals[t][d];
                }
            }
            (scores, out)
        };

        // --- Token 0 at position 0, cache has [pos0] ---
        let (w0, out0) = attend(&q, &[&k0], &[&v0], 0);
        // Token 0 attends 100% to itself
        assert!(
            (w0[0] - 1.0).abs() < 1e-6,
            "token 0 should attend fully to pos 0, got {}",
            w0[0]
        );
        for d in 0..head_dim {
            assert!((out0[d] - 1.0).abs() < 1e-6);
        }

        // --- Token 1 at position 1, cache has [pos0, pos1] ---
        let (w1, out1) = attend(&q, &[&k0, &k1], &[&v0, &v1], 1);
        // Equal Q·K scores → 50/50 split
        assert!(
            (w1[0] - 0.5).abs() < 1e-6,
            "token 1 should attend 50% to pos 0, got {}",
            w1[0]
        );
        assert!(
            (w1[1] - 0.5).abs() < 1e-6,
            "token 1 should attend 50% to pos 1, got {}",
            w1[1]
        );
        // Output = 0.5*v0 + 0.5*v1 = 0.5*1 + 0.5*2 = 1.5
        for d in 0..head_dim {
            assert!((out1[d] - 1.5).abs() < 1e-6);
        }

        // --- Token 0 at position 0, but cache has [pos0, pos1, pos2] ---
        // Future positions 1 and 2 must be masked out
        let k2 = vec![1.0f32; head_dim];
        let v2 = vec![3.0f32; head_dim];
        let (w_masked, out_masked) =
            attend(&q, &[&k0, &k1, &k2], &[&v0, &v1, &v2], 0);
        assert!(
            (w_masked[0] - 1.0).abs() < 1e-6,
            "masked: pos 0 should get all weight, got {}",
            w_masked[0]
        );
        assert!(w_masked[1] < 1e-6, "masked: pos 1 should be zero");
        assert!(w_masked[2] < 1e-6, "masked: pos 2 should be zero");
        for d in 0..head_dim {
            assert!((out_masked[d] - 1.0).abs() < 1e-6);
        }
    }

    /// Test sliding window attention masking (Mistral-style).
    ///
    /// With window W=3, a query at position 5 should only attend to
    /// positions 3, 4, 5 — positions 0, 1, 2 must get -inf scores.
    #[test]
    fn test_sliding_window_attention_mask() {
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let window: usize = 3;

        // All-ones Q/K so raw dot products are equal across positions
        let q = vec![1.0f32; head_dim];
        // 6 cached KV positions (0..=5), distinct values so we can verify output
        let keys: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0f32; head_dim]).collect();
        let vals: Vec<Vec<f32>> = (0..6).map(|t| vec![t as f32; head_dim]).collect();

        // Helper: compute attention output with causal + sliding window mask
        let attend_sw = |token_pos: usize, cached_len: usize| -> (Vec<f32>, Vec<f32>) {
            let mut scores = vec![0.0f32; cached_len];
            for t in 0..cached_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * keys[t][d];
                }
                scores[t] = dot * scale;
            }

            // Causal mask + sliding window
            for t in 0..cached_len {
                if t > token_pos {
                    scores[t] = f32::NEG_INFINITY;
                } else if token_pos >= window && t < token_pos - window + 1 {
                    scores[t] = f32::NEG_INFINITY;
                }
            }

            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }

            // Weighted V
            let mut out = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                for t in 0..cached_len {
                    out[d] += scores[t] * vals[t][d];
                }
            }
            (scores, out)
        };

        // --- Query at position 1 (within window, no clipping) ---
        // Window=3, pos=1: attend to [0, 1] (pos < W, so no left clipping)
        let (w1, _) = attend_sw(1, 2);
        assert!((w1[0] - 0.5).abs() < 1e-6, "pos 1: should attend 50% to pos 0, got {}", w1[0]);
        assert!((w1[1] - 0.5).abs() < 1e-6, "pos 1: should attend 50% to pos 1, got {}", w1[1]);

        // --- Query at position 5, cache has 6 entries (0..=5) ---
        // Window=3: attend to [3, 4, 5], positions 0, 1, 2 must be masked
        let (w5, out5) = attend_sw(5, 6);

        // Positions 0, 1, 2 must have zero weight (were -inf before softmax)
        for t in 0..3 {
            assert!(
                w5[t] < 1e-6,
                "pos 5 window=3: position {} should be masked, got weight {}",
                t, w5[t]
            );
        }

        // Positions 3, 4, 5 should each get 1/3 weight (equal Q·K scores)
        for t in 3..6 {
            assert!(
                (w5[t] - 1.0 / 3.0).abs() < 1e-5,
                "pos 5 window=3: position {} should get 1/3 weight, got {}",
                t, w5[t]
            );
        }

        // Output = (1/3)*3.0 + (1/3)*4.0 + (1/3)*5.0 = 4.0
        for d in 0..head_dim {
            assert!(
                (out5[d] - 4.0).abs() < 1e-4,
                "pos 5 window=3: output[{d}] should be 4.0, got {}",
                out5[d]
            );
        }

        // --- Query at position 3, cache has 4 entries (0..=3) ---
        // Window=3: attend to [1, 2, 3], position 0 must be masked
        let (w3, out3) = attend_sw(3, 4);
        assert!(
            w3[0] < 1e-6,
            "pos 3 window=3: position 0 should be masked, got weight {}",
            w3[0]
        );
        for t in 1..4 {
            assert!(
                (w3[t] - 1.0 / 3.0).abs() < 1e-5,
                "pos 3 window=3: position {} should get 1/3 weight, got {}",
                t, w3[t]
            );
        }
        // Output = (1/3)*1.0 + (1/3)*2.0 + (1/3)*3.0 = 2.0
        for d in 0..head_dim {
            assert!(
                (out3[d] - 2.0).abs() < 1e-4,
                "pos 3 window=3: output[{d}] should be 2.0, got {}",
                out3[d]
            );
        }
    }

    /// Verify that Mistral config constructor has sliding_window set.
    #[test]
    fn test_mistral_config_creation() {
        let cfg = ModelConfig::mistral_7b();
        assert_eq!(cfg.arch, ModelArch::Mistral);
        assert_eq!(cfg.sliding_window, Some(4096));
    }

    fn tiny_encoder_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gpt2,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 64,
            num_layers: 2,
            vocab_size: 50,
            max_seq_len: 16,
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

    /// Fill a DeviceTensor with a simple deterministic pattern for testing.
    unsafe fn fill_tensor_pattern(t: &mut DeviceTensor, scale: f32) {
        let s: &mut [f32] = t.as_mut_slice();
        for (i, v) in s.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * scale;
        }
    }

    #[test]
    fn test_encoder_forward_tiny_model() {
        let cfg = tiny_encoder_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let h = cfg.hidden_size;
        let seq_len = 4usize;

        // Fill weights with small deterministic values
        for lw in weights.layers.iter_mut() {
            unsafe {
                let gamma: &mut [f32] = lw.attn_norm.as_mut_slice();
                for v in gamma.iter_mut() { *v = 1.0; }
                let beta: &mut [f32] = lw.attn_norm_bias.as_mut_slice();
                for v in beta.iter_mut() { *v = 0.0; }
                let ffn_gamma: &mut [f32] = lw.ffn_norm.as_mut_slice();
                for v in ffn_gamma.iter_mut() { *v = 1.0; }
                let ffn_beta: &mut [f32] = lw.ffn_norm_bias.as_mut_slice();
                for v in ffn_beta.iter_mut() { *v = 0.0; }
                fill_tensor_pattern(&mut lw.wq, 0.01);
                fill_tensor_pattern(&mut lw.wk, 0.01);
                fill_tensor_pattern(&mut lw.wv, 0.01);
                fill_tensor_pattern(&mut lw.wo, 0.01);
                fill_tensor_pattern(&mut lw.w_up, 0.01);
                fill_tensor_pattern(&mut lw.w_down, 0.01);
            }
        }

        let mut input_data = vec![0.0f32; seq_len * h];
        for (i, v) in input_data.iter_mut().enumerate() {
            *v = ((i % 11) as f32 - 5.0) * 0.1;
        }
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions_data = vec![0.0f32; seq_len];
        let positions = unsafe { DeviceTensor::from_slice(&positions_data) };
        let mask_data = vec![1.0f32; seq_len * seq_len];
        let mask = unsafe { DeviceTensor::from_slice(&mask_data) };
        let mut output = DeviceTensor::alloc_cpu(seq_len * h, DType::F32).unwrap();

        backend
            .encoder_forward(&input, &positions, &mask, &weights, &mut output)
            .unwrap();

        let out_slice: &[f32] = unsafe { output.as_slice() };
        assert_eq!(out_slice.len(), seq_len * h);
        for &v in out_slice {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
        let any_nonzero = out_slice.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "output is all zeros");
    }

    #[test]
    fn test_encoder_forward_masked_attention() {
        let cfg = tiny_encoder_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let h = cfg.hidden_size;
        let seq_len = 3usize;

        for lw in weights.layers.iter_mut() {
            unsafe {
                let gamma: &mut [f32] = lw.attn_norm.as_mut_slice();
                for v in gamma.iter_mut() { *v = 1.0; }
                let beta: &mut [f32] = lw.attn_norm_bias.as_mut_slice();
                for v in beta.iter_mut() { *v = 0.0; }
                let fg: &mut [f32] = lw.ffn_norm.as_mut_slice();
                for v in fg.iter_mut() { *v = 1.0; }
                let fb: &mut [f32] = lw.ffn_norm_bias.as_mut_slice();
                for v in fb.iter_mut() { *v = 0.0; }
                fill_tensor_pattern(&mut lw.wq, 0.02);
                fill_tensor_pattern(&mut lw.wk, 0.02);
                fill_tensor_pattern(&mut lw.wv, 0.02);
                fill_tensor_pattern(&mut lw.wo, 0.02);
                fill_tensor_pattern(&mut lw.w_up, 0.02);
                fill_tensor_pattern(&mut lw.w_down, 0.02);
            }
        }

        let mut input_data = vec![0.1f32; seq_len * h];
        for (i, v) in input_data.iter_mut().enumerate() {
            *v = ((i % 5) as f32 - 2.0) * 0.1;
        }
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions_data = vec![0.0f32; seq_len];
        let positions = unsafe { DeviceTensor::from_slice(&positions_data) };

        // Full attention mask
        let full_mask = vec![1.0f32; seq_len * seq_len];
        let mask_full = unsafe { DeviceTensor::from_slice(&full_mask) };
        let mut out_full = DeviceTensor::alloc_cpu(seq_len * h, DType::F32).unwrap();
        backend
            .encoder_forward(&input, &positions, &mask_full, &weights, &mut out_full)
            .unwrap();

        // Partial mask: block position 2 from attending to position 0
        let mut partial_mask = vec![1.0f32; seq_len * seq_len];
        partial_mask[2 * seq_len + 0] = 0.0;
        let mask_partial = unsafe { DeviceTensor::from_slice(&partial_mask) };
        let mut out_partial = DeviceTensor::alloc_cpu(seq_len * h, DType::F32).unwrap();
        backend
            .encoder_forward(&input, &positions, &mask_partial, &weights, &mut out_partial)
            .unwrap();

        let s_full: &[f32] = unsafe { out_full.as_slice() };
        let s_partial: &[f32] = unsafe { out_partial.as_slice() };
        let diff: f32 = s_full.iter().zip(s_partial.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "masking had no effect on output (diff={diff})");
    }

    /// End-to-end numerical correctness test for decoder_forward.
    ///
    /// Uses a tiny model (hidden=8, heads=2, head_dim=4, inter=16, 1 layer)
    /// with deterministic weights. Computes the expected output step-by-step
    /// in plain Rust (no kernel calls) and asserts the kernel output matches
    /// within 1e-4 relative error.
    ///
    /// At position 0 with a single token:
    ///   - RoPE is identity (all angles = 0)
    ///   - Attention has cached_len=1, so softmax = [1.0], output = V
    /// This simplifies the reference to:
    ///   normed = rms_norm(input) → V = normed@Wv → o = V@Wo →
    ///   h1 = input+o → normed2 = rms_norm(h1) → gate/up = normed2@Wg/Wu →
    ///   act = silu(gate)*up → down = act@Wd → output = h1+down
    #[test]
    fn test_decoder_forward_numerical_correctness() {
        let cfg = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let h = cfg.hidden_size;       // 8
        let q_dim = cfg.num_heads * cfg.head_dim;   // 8
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 8
        let inter = cfg.intermediate_size; // 16

        // Deterministic weight generator with per-matrix seed
        let weight_pattern = |size: usize, seed: usize| -> Vec<f32> {
            (0..size).map(|i| ((i + seed) as f32 * 0.01).sin() * 0.1).collect()
        };

        let norm_w = vec![1.0f32; h];
        let wq_data = weight_pattern(h * q_dim, 0);
        let wk_data = weight_pattern(h * kv_dim, 100);
        let wv_data = weight_pattern(h * kv_dim, 200);
        let wo_data = weight_pattern(q_dim * h, 300);
        let wg_data = weight_pattern(h * inter, 400);
        let wu_data = weight_pattern(h * inter, 500);
        let wd_data = weight_pattern(inter * h, 600);

        // Fill weights into the model
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();
        unsafe {
            let lw = &mut weights.layers[0];
            let copy = |dst: &mut DeviceTensor, src: &[f32]| {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const u8,
                    dst.as_mut_ptr(),
                    src.len() * 4,
                );
            };
            copy(&mut lw.attn_norm, &norm_w);
            copy(&mut lw.ffn_norm, &norm_w);
            copy(&mut lw.wq, &wq_data);
            copy(&mut lw.wk, &wk_data);
            copy(&mut lw.wv, &wv_data);
            copy(&mut lw.wo, &wo_data);
            copy(&mut lw.w_gate, &wg_data);
            copy(&mut lw.w_up, &wu_data);
            copy(&mut lw.w_down, &wd_data);
            copy(&mut weights.final_norm, &norm_w);
        }
        let lm_head_data = weight_pattern(h * cfg.vocab_size, 700);
        unsafe {
            let copy = |dst: &mut DeviceTensor, src: &[f32]| {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const u8,
                    dst.as_mut_ptr(),
                    src.len() * 4,
                );
            };
            copy(&mut weights.lm_head, &lm_head_data);
        }

        // Input vector
        let input_data: Vec<f32> = (0..h)
            .map(|i| ((i as f32 * 0.1 + 0.5).sin() * 0.5))
            .collect();
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions_data = vec![0.0f32; 1];
        let positions = unsafe { DeviceTensor::from_slice(&positions_data) };
        let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

        // Run the kernel path
        backend
            .decoder_forward(&input, &positions, &mut kv_cache, &weights, &[1], &mut output)
            .unwrap();
        let kernel_out: Vec<f32> = unsafe { output.as_slice::<f32>() }.to_vec();

        // ---- Reference computation in plain f32 math ----

        // RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
        let ref_rms_norm = |x: &[f32], w: &[f32], eps: f32| -> Vec<f32> {
            let n = x.len();
            let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let scale = 1.0 / (ss + eps).sqrt();
            x.iter().zip(w).map(|(&xi, &wi)| xi * scale * wi).collect()
        };

        // Matmul: C[1×n] = A[1×k] @ B[k×n]  (B row-major)
        let ref_matmul = |a: &[f32], b: &[f32], n: usize, k: usize| -> Vec<f32> {
            let mut c = vec![0.0f32; n];
            for j in 0..n {
                for p in 0..k {
                    c[j] += a[p] * b[p * n + j];
                }
            }
            c
        };

        // SiLU: x / (1 + exp(-x))
        let ref_silu = |x: &[f32]| -> Vec<f32> {
            x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
        };

        // vec elementwise: a op b
        let ref_add = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(&x, &y)| x + y).collect()
        };
        let ref_mul = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(&x, &y)| x * y).collect()
        };

        // Step 1: Attention RMSNorm
        let normed = ref_rms_norm(&input_data, &norm_w, cfg.norm_eps);

        // Step 2: QKV projections
        // At position 0, RoPE is identity (cos=1, sin=0), so Q/K are unchanged.
        // With cached_len=1, softmax([score]) = [1.0], so attn_out = V.
        let _q = ref_matmul(&normed, &wq_data, q_dim, h);
        let _k = ref_matmul(&normed, &wk_data, kv_dim, h);
        let v = ref_matmul(&normed, &wv_data, kv_dim, h);

        // Step 3: Attention output = V (single token at pos 0)
        // attn_out layout matches V since num_heads == num_kv_heads
        let attn_out = v;

        // Step 4: Output projection
        let o_proj = ref_matmul(&attn_out, &wo_data, h, q_dim);

        // Step 5: First residual connection
        let hidden = ref_add(&input_data, &o_proj);

        // Step 6: FFN RMSNorm
        let normed2 = ref_rms_norm(&hidden, &norm_w, cfg.norm_eps);

        // Step 7: FFN gate/up projections
        let gate = ref_matmul(&normed2, &wg_data, inter, h);
        let up = ref_matmul(&normed2, &wu_data, inter, h);

        // Step 8: SiLU activation + elementwise multiply
        let act = ref_silu(&gate);
        let gate_up = ref_mul(&act, &up);

        // Step 9: Down projection
        let down = ref_matmul(&gate_up, &wd_data, h, inter);

        // Step 10: Second residual connection
        let hidden_final = ref_add(&hidden, &down);

        // Step 11: Final RMSNorm
        let final_normed = ref_rms_norm(&hidden_final, &norm_w, cfg.norm_eps);

        // Step 12: LM head projection (hidden_size -> vocab_size)
        let expected = ref_matmul(&final_normed, &lm_head_data, cfg.vocab_size, h);

        // ---- Compare kernel output vs reference ----
        let max_rel_err = kernel_out
            .iter()
            .zip(expected.iter())
            .map(|(&got, &exp)| {
                let denom = exp.abs().max(1e-7);
                (got - exp).abs() / denom
            })
            .fold(0.0f32, f32::max);

        assert!(
            max_rel_err < 1e-4,
            "decoder_forward numerical mismatch: max relative error = {max_rel_err:.6e}\n\
             kernel:   {kernel_out:?}\n\
             expected: {expected:?}"
        );

        // Sanity: output should be finite and non-trivial
        assert!(kernel_out.iter().all(|v| v.is_finite()));
        assert!(kernel_out.iter().any(|v| v.abs() > 1e-6));
    }
}
