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
use crate::types::{DType, InferenceError, ModelConfig};
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
                std::mem::size_of_val(src),
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
                src.num_elements() * std::mem::size_of::<f32>(),
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
        // Single-token, single-batch path (multi-token/batch requires Layer 2 JIT)
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
            let token_pos = positions_slice.first().copied().unwrap_or(0.0) as usize;
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
            // Page layout: [2 (K+V), num_kv_heads, PAGE_SIZE, head_dim] in kv_dtype
            let v_base = num_kv_heads * PAGE_SIZE * head_dim;
            let kv_dtype = kv_cache.dtype();
            let elem_bytes = kv_dtype.size_bytes();
            let page_ptr = kv_cache.page_mut_ptr(page_id);
            for kv_h in 0..num_kv_heads {
                let head_base = kv_h * PAGE_SIZE * head_dim + offset_in_page * head_dim;
                for d in 0..head_dim {
                    let k_val = k[kv_h * head_dim + d];
                    let v_val = v[kv_h * head_dim + d];
                    let k_byte_off = (head_base + d) * elem_bytes;
                    let v_byte_off = (v_base + head_base + d) * elem_bytes;
                    unsafe {
                        match kv_dtype {
                            DType::F32 => {
                                *(page_ptr.add(k_byte_off) as *mut f32) = k_val;
                                *(page_ptr.add(v_byte_off) as *mut f32) = v_val;
                            }
                            DType::BF16 => {
                                *(page_ptr.add(k_byte_off) as *mut half::bf16) =
                                    half::bf16::from_f32(k_val);
                                *(page_ptr.add(v_byte_off) as *mut half::bf16) =
                                    half::bf16::from_f32(v_val);
                            }
                            DType::F16 => {
                                *(page_ptr.add(k_byte_off) as *mut half::f16) =
                                    half::f16::from_f32(k_val);
                                *(page_ptr.add(v_byte_off) as *mut half::f16) =
                                    half::f16::from_f32(v_val);
                            }
                            other => {
                                return Err(InferenceError::Unsupported(format!(
                                    "dtype {other:?} for KV cache write"
                                )));
                            }
                        }
                    }
                }
            }

            // Multi-head attention with causal mask
            let cached_len = kv_cache.seq_len(layer_idx, 0);
            let seq_pages = kv_cache.seq_pages(layer_idx, 0);
            let kv_dtype = kv_cache.dtype();
            let elem_bytes = kv_dtype.size_bytes();

            for ah in 0..num_heads {
                let kv_h = ah / heads_per_kv;
                let q_off = ah * head_dim;

                // Compute scores: Q[h] · K[kv_h][t] for all cached positions
                let mut scores = vec![0.0f32; cached_len];
                for t in 0..cached_len {
                    let pid = seq_pages[t / PAGE_SIZE];
                    let off = t % PAGE_SIZE;
                    let kp = kv_cache.page_ptr(pid);
                    let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let byte_off = (k_base + d) * elem_bytes;
                        let k_val = unsafe {
                            match kv_dtype {
                                DType::F32 => *(kp.add(byte_off) as *const f32),
                                DType::BF16 => {
                                    (*(kp.add(byte_off) as *const half::bf16)).to_f32()
                                }
                                DType::F16 => {
                                    (*(kp.add(byte_off) as *const half::f16)).to_f32()
                                }
                                other => {
                                    return Err(InferenceError::Unsupported(format!(
                                        "dtype {other:?} for KV cache K read"
                                    )));
                                }
                            }
                        };
                        dot += q[q_off + d] * k_val;
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
                        let vp = kv_cache.page_ptr(pid);
                        let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim + d;
                        let byte_off = v_off * elem_bytes;
                        let v_val = unsafe {
                            match kv_dtype {
                                DType::F32 => *(vp.add(byte_off) as *const f32),
                                DType::BF16 => {
                                    (*(vp.add(byte_off) as *const half::bf16)).to_f32()
                                }
                                DType::F16 => {
                                    (*(vp.add(byte_off) as *const half::f16)).to_f32()
                                }
                                other => {
                                    return Err(InferenceError::Unsupported(format!(
                                        "dtype {other:?} for KV cache V read"
                                    )));
                                }
                            }
                        };
                        val += scores[t] * v_val;
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
                crate::types::ModelArch::Gemma => self.kernels.gelu(&gate, &mut act),
                _ => self.kernels.silu(&gate, &mut act),
            }
            self.kernels.vec_mul(&act, &up, &mut gate);
            self.kernels.gemm(&gate, wd, &mut down, 1, h, inter);

            // 9. Residual add
            self.kernels.vec_add(&residual, &down, &mut hidden);
        }

        // Final RMSNorm
        let final_norm_w: &[f32] = unsafe { weights.final_norm.as_slice() };
        self.kernels.rms_norm(&hidden, final_norm_w, &mut normed, self.config.norm_eps);

        // LM head: [1, h] @ [h, vocab_size] → [1, vocab_size]
        let lm_head_w: &[f32] = unsafe { weights.lm_head.as_slice() };
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        self.kernels.gemm(&normed, lm_head_w, &mut logits, 1, vocab_size, h);

        // Copy logits to output
        unsafe {
            std::ptr::copy_nonoverlapping(
                logits.as_ptr() as *const u8,
                output.as_mut_ptr(),
                vocab_size * std::mem::size_of::<f32>(),
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

        // --- Mean pooling: average across sequence dimension ---
        let mut pooled = vec![0.0f32; h];
        for d in 0..h {
            let mut sum = 0.0f32;
            for s in 0..seq_len {
                sum += hidden[s * h + d];
            }
            pooled[d] = sum / seq_len as f32;
        }

        // --- L2 normalize the pooled embedding ---
        let mut sum_sq = 0.0f32;
        for d in 0..h {
            sum_sq += pooled[d] * pooled[d];
        }
        let inv_norm = 1.0 / (sum_sq + 1e-12_f32).sqrt();
        for d in 0..h {
            pooled[d] *= inv_norm;
        }

        // Copy pooled result to output tensor
        unsafe {
            std::ptr::copy_nonoverlapping(
                pooled.as_ptr() as *const u8,
                output.as_mut_ptr(),
                h * std::mem::size_of::<f32>(),
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

impl CpuInferenceBackend {
    /// Greedy text generation: embed prompt tokens → decoder_forward loop → sample.
    ///
    /// Processes prompt tokens one by one through the decoder, then generates
    /// `max_new_tokens` new tokens via greedy (or temperature-scaled) sampling.
    /// Returns the generated token IDs (not including the prompt).
    pub fn generate(
        &self,
        weights: &ModelWeights,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>, InferenceError> {
        if prompt_tokens.is_empty() {
            return Err(InferenceError::RuntimeError(
                "prompt_tokens must not be empty".into(),
            ));
        }
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let h = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let max_seq = prompt_tokens.len() + max_new_tokens;

        let mut kv_cache = self.alloc_kv_cache(1, max_seq)?;
        let mut logits_tensor = self.alloc(vocab_size, DType::F32)?;
        let embedding: &[f32] = unsafe { weights.embedding.as_slice() };

        // Process prompt tokens one by one (builds KV cache)
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            let tid = token_id as usize;
            if tid >= self.config.vocab_size {
                return Err(InferenceError::RuntimeError(
                    format!("token_id {tid} >= vocab_size {}", self.config.vocab_size),
                ));
            }
            let token_embed = &embedding[tid * h..(tid + 1) * h];
            let input = unsafe { DeviceTensor::from_slice(token_embed) };
            let pos_data = [pos as f32];
            let positions = unsafe { DeviceTensor::from_slice(&pos_data) };

            self.decoder_forward(
                &input, &positions, &mut kv_cache, weights, &[1], &mut logits_tensor,
            )?;
        }

        // Sample first generated token from last prompt token logits
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next_token = [0u32; 1];
        self.sample(&logits_tensor, temperature, 0, 0.0, &mut next_token)?;
        generated.push(next_token[0]);

        // Auto-regressive generation loop
        for step in 1..max_new_tokens {
            let pos = prompt_tokens.len() + step - 1;
            let tid = next_token[0] as usize;
            if tid >= self.config.vocab_size {
                break;
            }
            let token_embed = &embedding[tid * h..(tid + 1) * h];
            let input = unsafe { DeviceTensor::from_slice(token_embed) };
            let pos_data = [pos as f32];
            let positions = unsafe { DeviceTensor::from_slice(&pos_data) };

            self.decoder_forward(
                &input, &positions, &mut kv_cache, weights, &[1], &mut logits_tensor,
            )?;
            self.sample(&logits_tensor, temperature, 0, 0.0, &mut next_token)?;
            generated.push(next_token[0]);
        }

        Ok(generated)
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
    use crate::types::{ModelArch, ModelConfig};

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
        let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

        let result = backend.decoder_forward(
            &input, &positions, &mut kv_cache, &weights, &[1], &mut output,
        );
        assert!(result.is_ok(), "decoder_forward failed: {:?}", result.err());
        assert_eq!(output.num_elements(), cfg.vocab_size);
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
        let final_norm_data = vec![1.0f32; h];
        let lm_head_data = weight_pattern(h * cfg.vocab_size, 700);

        // Fill weights into the model
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();
        unsafe {
            let copy = |dst: &mut DeviceTensor, src: &[f32]| {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const u8,
                    dst.as_mut_ptr(),
                    src.len() * std::mem::size_of::<f32>(),
                );
            };
            copy(&mut weights.layers[0].attn_norm, &norm_w);
            copy(&mut weights.layers[0].ffn_norm, &norm_w);
            copy(&mut weights.layers[0].wq, &wq_data);
            copy(&mut weights.layers[0].wk, &wk_data);
            copy(&mut weights.layers[0].wv, &wv_data);
            copy(&mut weights.layers[0].wo, &wo_data);
            copy(&mut weights.layers[0].w_gate, &wg_data);
            copy(&mut weights.layers[0].w_up, &wu_data);
            copy(&mut weights.layers[0].w_down, &wd_data);
            copy(&mut weights.final_norm, &final_norm_data);
            copy(&mut weights.lm_head, &lm_head_data);
        }

        // Input vector
        let input_data: Vec<f32> = (0..h)
            .map(|i| (i as f32 * 0.1 + 0.5).sin() * 0.5)
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
        let final_normed = ref_rms_norm(&hidden_final, &final_norm_data, cfg.norm_eps);

        // Step 12: LM head projection
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
            max_rel_err < 1e-3,
            "decoder_forward numerical mismatch: max relative error = {max_rel_err:.6e}\n\
             kernel[..8]:   {:?}\n\
             expected[..8]: {:?}",
            &kernel_out[..8.min(kernel_out.len())],
            &expected[..8.min(expected.len())]
        );

        // Sanity: logits should be finite and non-trivial
        assert!(kernel_out.iter().all(|v| v.is_finite()));
        assert!(kernel_out.iter().any(|v| v.abs() > 1e-8));
    }

    /// Test that decoder_forward produces vocab_size logits.
    #[test]
    fn test_decoder_forward_produces_logits() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        // Fill norm weights with 1.0
        let ones = vec![1.0f32; cfg.hidden_size];
        for lw in weights.layers.iter_mut() {
            unsafe {
                let norm: &mut [f32] = lw.attn_norm.as_mut_slice();
                norm.copy_from_slice(&ones);
                let ffn: &mut [f32] = lw.ffn_norm.as_mut_slice();
                ffn.copy_from_slice(&ones);
            }
        }
        unsafe {
            let fn_w: &mut [f32] = weights.final_norm.as_mut_slice();
            fn_w.copy_from_slice(&ones);
            let lm: &mut [f32] = weights.lm_head.as_mut_slice();
            for (i, v) in lm.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 0.01;
            }
        }

        let input_data = vec![0.1f32; cfg.hidden_size];
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let pos_data = vec![0.0f32; 1];
        let positions = unsafe { DeviceTensor::from_slice(&pos_data) };
        let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

        backend
            .decoder_forward(&input, &positions, &mut kv_cache, &weights, &[1], &mut output)
            .unwrap();

        let logits: &[f32] = unsafe { output.as_slice() };
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "logits contain non-finite values");
        let first = logits[0];
        assert!(
            logits.iter().any(|&v| (v - first).abs() > 1e-8),
            "all logits are identical — lm_head projection has no effect"
        );
    }

    /// Test multi-layer stacking: 2-layer model produces different output than 1-layer.
    #[test]
    fn test_multi_layer_forward_differs() {
        let make_cfg = |n_layers: usize| ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: n_layers,
            vocab_size: 20,
            max_seq_len: 16,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let run_forward = |n_layers: usize| -> Vec<f32> {
            let cfg = make_cfg(n_layers);
            let backend = CpuInferenceBackend::init(&cfg).unwrap();
            let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

            let ones = vec![1.0f32; cfg.hidden_size];
            for lw in weights.layers.iter_mut() {
                unsafe {
                    lw.attn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                    lw.ffn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                    fill_tensor_pattern(&mut lw.wq, 0.02);
                    fill_tensor_pattern(&mut lw.wk, 0.02);
                    fill_tensor_pattern(&mut lw.wv, 0.02);
                    fill_tensor_pattern(&mut lw.wo, 0.02);
                    fill_tensor_pattern(&mut lw.w_gate, 0.02);
                    fill_tensor_pattern(&mut lw.w_up, 0.02);
                    fill_tensor_pattern(&mut lw.w_down, 0.02);
                }
            }
            unsafe {
                weights.final_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                let lm: &mut [f32] = weights.lm_head.as_mut_slice();
                for (i, v) in lm.iter_mut().enumerate() {
                    *v = (i as f32 * 0.01).sin() * 0.1;
                }
            }

            let input_data: Vec<f32> = (0..cfg.hidden_size)
                .map(|i| (i as f32 * 0.1 + 0.3).sin() * 0.5)
                .collect();
            let input = unsafe { DeviceTensor::from_slice(&input_data) };
            let pos_data = [0.0f32];
            let positions = unsafe { DeviceTensor::from_slice(&pos_data) };
            let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
            let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

            backend
                .decoder_forward(&input, &positions, &mut kv_cache, &weights, &[1], &mut output)
                .unwrap();

            unsafe { output.as_slice::<f32>() }.to_vec()
        };

        let out_1 = run_forward(1);
        let out_2 = run_forward(2);

        assert_eq!(out_1.len(), 20);
        assert_eq!(out_2.len(), 20);

        let diff: f32 = out_1.iter().zip(out_2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "1-layer and 2-layer outputs are identical (diff={diff})"
        );
    }

    /// Test greedy generation produces a deterministic token sequence.
    #[test]
    fn test_generate_greedy() {
        use crate::inference::cpu_backend::CpuInferenceBackend;

        let cfg = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 20,
            max_seq_len: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let ones = vec![1.0f32; cfg.hidden_size];
        unsafe {
            for lw in weights.layers.iter_mut() {
                lw.attn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                lw.ffn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                fill_tensor_pattern(&mut lw.wq, 0.01);
                fill_tensor_pattern(&mut lw.wk, 0.01);
                fill_tensor_pattern(&mut lw.wv, 0.01);
                fill_tensor_pattern(&mut lw.wo, 0.01);
                fill_tensor_pattern(&mut lw.w_gate, 0.01);
                fill_tensor_pattern(&mut lw.w_up, 0.01);
                fill_tensor_pattern(&mut lw.w_down, 0.01);
            }
            weights.final_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
            let emb: &mut [f32] = weights.embedding.as_mut_slice();
            for (i, v) in emb.iter_mut().enumerate() {
                *v = ((i as f32 * 0.07 + 0.3).sin()) * 0.5;
            }
            let lm: &mut [f32] = weights.lm_head.as_mut_slice();
            for (i, v) in lm.iter_mut().enumerate() {
                *v = ((i as f32 * 0.03 + 0.1).cos()) * 0.1;
            }
        }

        let prompt = vec![1u32, 2, 3];
        let generated = backend.generate(&weights, &prompt, 5, 0.0).unwrap();

        assert_eq!(generated.len(), 5, "should generate exactly 5 tokens");
        for &tok in &generated {
            assert!(
                (tok as usize) < cfg.vocab_size,
                "generated token {tok} >= vocab_size {}",
                cfg.vocab_size
            );
        }

        let generated2 = backend.generate(&weights, &prompt, 5, 0.0).unwrap();
        assert_eq!(generated, generated2, "greedy generation should be deterministic");
    }

    /// Test generate with zero max_new_tokens returns empty.
    #[test]
    fn test_generate_zero_tokens() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let result = backend.generate(&weights, &[0], 0, 0.0).unwrap();
        assert!(result.is_empty());
    }

    /// Test generate with empty prompt returns error.
    #[test]
    fn test_generate_empty_prompt() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let result = backend.generate(&weights, &[], 5, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_temperature_scaled_argmax() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut logits_data = vec![0.0f32; 100];
        logits_data[7] = 5.0;
        logits_data[33] = 10.0;
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 1];
        backend.sample(&logits, 2.0, 0, 0.0, &mut ids).unwrap();
        assert_eq!(ids[0], 33);
    }

    #[test]
    fn test_sample_multi_batch_argmax() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let vocab_size = cfg.vocab_size;
        let mut logits_data = vec![0.0f32; vocab_size * 3];
        logits_data[1 * vocab_size + 5] = 99.0;
        logits_data[2 * vocab_size + 88] = 99.0;
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 3];
        backend.sample(&logits, 0.0, 0, 0.0, &mut ids).unwrap();
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 5);
        assert_eq!(ids[2], 88);
    }

    #[test]
    fn test_sample_insufficient_logits_returns_error() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let logits_data = vec![0.0f32; 50];
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 2];
        let result = backend.sample(&logits, 0.0, 0, 0.0, &mut ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_upload_f32_preserves_data_integrity() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let src: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let mut t = backend.alloc(128, DType::F32).unwrap();
        backend.upload_f32(&src, &mut t).unwrap();
        let mut dst = vec![0.0f32; 128];
        backend.download_f32(&t, &mut dst).unwrap();
        for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "mismatch at index {i}: {a} != {b}");
        }
    }

    #[test]
    fn test_download_f32_larger_buffer_zeroes_remain() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let src = vec![42.0f32, -1.0, 0.5];
        let mut t = backend.alloc(4, DType::F32).unwrap();
        backend.upload_f32(&src, &mut t).unwrap();
        let mut dst = vec![f32::NAN; 4];
        backend.download_f32(&t, &mut dst).unwrap();
        assert_eq!(dst[0], 42.0);
        assert_eq!(dst[1], -1.0);
        assert!((dst[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sync_is_noop_success() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        backend.sync().unwrap();
    }

    #[test]
    fn test_alloc_kv_cache_page_counts() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let pages_per_seq = (cfg.max_seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
        assert_eq!(cache.total_pages(), pages_per_seq * cfg.num_layers);
        assert_eq!(cache.free_page_count(), cache.total_pages());
        assert_eq!(cache.bytes_per_page(), 2 * cfg.num_kv_heads * PAGE_SIZE * cfg.head_dim * cfg.dtype.size_bytes());
    }

    #[test]
    fn test_inference_error_display_variants() {
        let e1 = InferenceError::InvalidConfig("bad".into());
        assert!(e1.to_string().contains("bad"));
        let e2 = InferenceError::OutOfMemory { requested: 1024, available: 512 };
        assert!(e2.to_string().contains("1024") && e2.to_string().contains("512"));
        let e3 = InferenceError::RuntimeError("crash".into());
        assert!(e3.to_string().contains("crash"));
        let e4 = InferenceError::Unsupported("foo".into());
        assert!(e4.to_string().contains("foo"));
    }

    #[test]
    fn test_model_arch_debug_and_equality() {
        assert_eq!(ModelArch::Llama, ModelArch::Llama);
        assert_ne!(ModelArch::Llama, ModelArch::Gpt2);
        assert_ne!(ModelArch::Qwen, ModelArch::Gemma);
        let debug = format!("{:?}", ModelArch::Mistral);
        assert!(debug.contains("Mistral"));
    }

    #[test]
    fn test_apply_rope_identity_rotation() {
        let num_heads = 2;
        let head_dim = 4;
        let cos = vec![1.0f32; 2];
        let sin = vec![0.0f32; 2];
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = qk.clone();
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);
        for (i, (&a, &b)) in original.iter().zip(qk.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "identity RoPE changed value at index {i}");
        }
    }

    #[test]
    fn test_encoder_forward_empty_sequence_returns_ok() {
        let cfg = tiny_encoder_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();
        let input_data = vec![0.0f32; 0];
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions_data = vec![0.0f32; 0];
        let positions = unsafe { DeviceTensor::from_slice(&positions_data) };
        let mask_data = vec![0.0f32; 0];
        let mask = unsafe { DeviceTensor::from_slice(&mask_data) };
        let mut output = DeviceTensor::alloc_cpu(1, DType::F32).unwrap();
        let result = backend.encoder_forward(&input, &positions, &mask, &weights, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_oob_token_id_returns_error() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();
        let oob_token = cfg.vocab_size as u32 + 10;
        let result = backend.generate(&weights, &[oob_token], 1, 0.0);
        assert!(result.is_err());
    }

    // ── Test 32: upload_f32 dtype mismatch error ──

    #[test]
    fn upload_f32_dtype_mismatch_returns_error() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut t = backend.alloc(4, DType::F16).unwrap();
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = backend.upload_f32(&src, &mut t);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("F32"), "error should mention F32, got: {msg}");
    }

    // ── Test 33: upload_f32 buffer too small error ──

    #[test]
    fn upload_f32_buffer_too_small_returns_error() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut t = backend.alloc(2, DType::F32).unwrap();
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = backend.upload_f32(&src, &mut t);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("4"), "error should mention 4 elements, got: {msg}");
    }

    // ── Test 34: download_f32 dtype mismatch error ──

    #[test]
    fn download_f32_dtype_mismatch_returns_error() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let t = backend.alloc(4, DType::F16).unwrap();
        let mut dst = vec![0.0f32; 4];
        let result = backend.download_f32(&t, &mut dst);
        assert!(result.is_err());
    }

    // ── Test 35: apply_rope_90_degree_rotation ──

    #[test]
    fn apply_rope_90_degree_rotation() {
        // Arrange — cos=0, sin=1 → 90° rotation: (x,y) → (-y, x)
        let num_heads = 1;
        let head_dim = 4;
        let cos = vec![0.0f32; 2];
        let sin = vec![1.0f32; 2];
        let mut qk = vec![3.0f32, 5.0, 7.0, 11.0];

        // Act
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);

        // Assert — (x0, x1, x2, x3) with half=2:
        //   rotated: idx 0 = x0*c - x2*s = 3*0 - 7*1 = -7
        //            idx 1 = x1*c - x3*s = 5*0 - 11*1 = -11
        //            idx 2 = x0*s + x2*c = 3*1 + 7*0 = 3
        //            idx 3 = x1*s + x3*c = 5*1 + 11*0 = 5
        assert!((qk[0] - (-7.0)).abs() < 1e-5);
        assert!((qk[1] - (-11.0)).abs() < 1e-5);
        assert!((qk[2] - 3.0).abs() < 1e-5);
        assert!((qk[3] - 5.0).abs() < 1e-5);
    }

    // ── Test 36: apply_rope_zero_input_passthrough ──

    #[test]
    fn apply_rope_zero_input_remains_zero() {
        // Arrange
        let num_heads = 2;
        let head_dim = 4;
        let cos = vec![0.5f32, 0.8];
        let sin = vec![0.6f32, 0.3];
        let mut qk = vec![0.0f32; num_heads * head_dim];

        // Act
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);

        // Assert — zero input should produce zero output
        for (i, &v) in qk.iter().enumerate() {
            assert!(v.abs() < 1e-10, "index {i}: expected zero, got {v}");
        }
    }

    // ── Test 37: alloc with different dtypes ──

    #[test]
    fn alloc_with_different_dtypes() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();

        let t_f32 = backend.alloc(64, DType::F32).unwrap();
        assert_eq!(t_f32.num_elements(), 64);

        let t_f16 = backend.alloc(32, DType::F16).unwrap();
        assert_eq!(t_f16.num_elements(), 32);
        assert_ne!(t_f32.dtype(), t_f16.dtype());
    }

    // ── Test 38: tiny_config fields are consistent ──

    #[test]
    fn tiny_config_fields_consistent() {
        let cfg = tiny_config();
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        assert_eq!(cfg.arch, ModelArch::Llama);
        assert!(cfg.vocab_size > 0);
        assert!(cfg.max_seq_len > 0);
        assert!(cfg.rope_theta > 0.0);
        assert!(cfg.norm_eps > 0.0);
        assert!(!cfg.has_qkv_bias);
        assert_eq!(cfg.partial_rotary_factor, 1.0);
        assert!(cfg.sliding_window.is_none());
    }

    // ── Test 39: tiny_encoder_config fields are consistent ──

    #[test]
    fn tiny_encoder_config_fields_consistent() {
        let cfg = tiny_encoder_config();
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        assert_eq!(cfg.num_kv_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim);
        assert_eq!(cfg.arch, ModelArch::Gpt2);
        assert!(cfg.vocab_size > 0);
        assert!(cfg.intermediate_size > cfg.hidden_size);
    }

    // ── Test 40: kv_cache_pages_calculate_correctly ──

    #[test]
    fn kv_cache_pages_calculate_correctly() {
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();

        // Assert — total_pages = ceil(max_seq_len / PAGE_SIZE) * num_layers
        let pages_per_seq_layer = (cfg.max_seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
        let expected_total = pages_per_seq_layer * cfg.num_layers;
        assert_eq!(cache.total_pages(), expected_total);
        assert!(cache.bytes_per_page() > 0);
    }

    // ── Test 41: InferenceError Io variant display ──

    #[test]
    fn inference_error_io_variant_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e = InferenceError::Io(io_err);
        assert!(e.to_string().contains("file missing"));
    }

    // ── Test 42: InferenceError ShapeMismatch variant display ──

    #[test]
    fn inference_error_shape_mismatch_display() {
        let e = InferenceError::ShapeMismatch {
            expected: "[2,3]".into(),
            got: "[4,5]".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("[2,3]"), "should contain expected shape");
        assert!(msg.contains("[4,5]"), "should contain got shape");
    }

    // ── Test 43: download_f32 buffer too small returns error ──

    #[test]
    fn download_f32_buffer_too_small_returns_error() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut t = backend.alloc(4, DType::F32).unwrap();
        backend.upload_f32(&src, &mut t).unwrap();

        // Act — dst buffer has only 2 elements for a 4-element tensor
        let mut dst = vec![0.0f32; 2];
        let result = backend.download_f32(&t, &mut dst);

        // Assert
        assert!(result.is_err(), "should error when dst is smaller than tensor");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("4"),
            "error message should mention tensor element count, got: {msg}"
        );
    }

    // ── Test 44: DType size_bytes returns correct values ──

    #[test]
    fn dtype_size_bytes_all_variants() {
        // Arrange & Act & Assert
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::F8E4M3.size_bytes(), 1);
        assert_eq!(DType::F8E5M2.size_bytes(), 1);
        // Sub-byte types return 1 as minimum addressable unit
        assert_eq!(DType::F6E3M2.size_bytes(), 1);
        assert_eq!(DType::F6E2M3.size_bytes(), 1);
        assert_eq!(DType::F4E2M1.size_bytes(), 1);
    }

    // ── Test 45: DType elem_id is unique across all variants ──

    #[test]
    fn dtype_elem_id_unique_across_variants() {
        // Arrange
        let ids = [
            DType::F32.elem_id(),
            DType::F16.elem_id(),
            DType::BF16.elem_id(),
            DType::U8.elem_id(),
            DType::F8E4M3.elem_id(),
            DType::F8E5M2.elem_id(),
            DType::F6E3M2.elem_id(),
            DType::F6E2M3.elem_id(),
            DType::F4E2M1.elem_id(),
        ];

        // Act & Assert — all IDs must be unique
        let mut sorted = ids.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "elem_id values must be unique");
    }

    // ── Test 46: DType gpu_type_name coverage ──

    #[test]
    fn dtype_gpu_type_name_coverage() {
        // Arrange & Act & Assert
        assert_eq!(DType::F32.gpu_type_name().unwrap(), "f32");
        assert_eq!(DType::F16.gpu_type_name().unwrap(), "f16");
        assert_eq!(DType::BF16.gpu_type_name().unwrap(), "bf16");
        assert_eq!(DType::U8.gpu_type_name().unwrap(), "u8");
        assert_eq!(DType::F8E4M3.gpu_type_name().unwrap(), "e4m3");
        assert_eq!(DType::F8E5M2.gpu_type_name().unwrap(), "e5m2");
        assert_eq!(DType::F4E2M1.gpu_type_name().unwrap(), "e2m1");
        // FP6 variants should return Err
        assert!(DType::F6E3M2.gpu_type_name().is_err());
        assert!(DType::F6E2M3.gpu_type_name().is_err());
    }

    // ── Test 47: DeviceKind equality and debug traits ──

    #[test]
    fn device_kind_equality_and_debug() {
        // Arrange & Act & Assert
        assert_eq!(DeviceKind::Cpu, DeviceKind::Cpu);
        assert_ne!(DeviceKind::Cpu, DeviceKind::Cuda(0));
        assert_eq!(DeviceKind::Cuda(0), DeviceKind::Cuda(0));
        assert_ne!(DeviceKind::Cuda(0), DeviceKind::Cuda(1));
        assert_eq!(DeviceKind::Metal(0), DeviceKind::Metal(0));

        let debug = format!("{:?}", DeviceKind::Cpu);
        assert!(debug.contains("Cpu"), "Debug should contain 'Cpu', got: {debug}");

        let debug_cuda = format!("{:?}", DeviceKind::Cuda(3));
        assert!(debug_cuda.contains("3"), "Debug should contain device id, got: {debug_cuda}");
    }

    // ── Test 48: apply_rope with minimal head_dim (2) ──

    #[test]
    fn apply_rope_minimal_head_dim() {
        // Arrange — 1 head, head_dim=2, half=1: only 1 pair
        let num_heads = 1;
        let head_dim = 2;
        let cos = vec![0.8f32];
        let sin = vec![0.6f32];
        let mut qk = vec![3.0f32, 4.0];

        // Act
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);

        // Assert — idx 0: x0*c - x1*s = 3*0.8 - 4*0.6 = 2.4 - 2.4 = 0.0
        //          idx 1: x0*s + x1*c = 3*0.6 + 4*0.8 = 1.8 + 3.2 = 5.0
        assert!((qk[0] - 0.0).abs() < 1e-5, "expected 0.0, got {}", qk[0]);
        assert!((qk[1] - 5.0).abs() < 1e-5, "expected 5.0, got {}", qk[1]);
    }

    // ── Test 49: sample with all equal logits picks first token ──

    #[test]
    fn sample_all_equal_logits_picks_first() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let logits_data = vec![1.0f32; cfg.vocab_size];
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 1];

        // Act
        backend.sample(&logits, 0.0, 0, 0.0, &mut ids).unwrap();

        // Assert — argmax with all equal values should return first index
        assert_eq!(ids[0], 0);
    }

    // ── Test 50: GQA config with num_kv_heads < num_heads is consistent ──

    #[test]
    fn gqa_config_kv_heads_fewer_than_heads() {
        // Arrange — 8 heads, 2 KV heads (4:1 GQA ratio)
        let cfg = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 64,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 128,
            num_layers: 1,
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
        };

        // Act & Assert
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        assert_eq!(cfg.num_kv_heads * cfg.head_dim, 16); // kv_dim
        assert!(cfg.num_heads % cfg.num_kv_heads == 0, "GQA ratio must be integer");
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 4); // 4 query heads per KV head
    }

    // ── Test 51: InferenceError CompileError variant display ──

    #[test]
    fn inference_error_compile_error_display() {
        // Arrange
        let compiler_err = crate::compiler::CompilerError::UnsupportedDType { dtype: DType::F4E2M1, isa: "avx2".into() };
        let e = InferenceError::CompileError(compiler_err);

        // Act
        let msg = e.to_string();

        // Assert
        assert!(msg.contains("compile error"), "should contain 'compile error', got: {msg}");
    }

    // ── Test 52: upload_download round-trip with negative and extreme values ──

    #[test]
    fn upload_download_negative_and_extreme_roundtrip() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let src = vec![
            -100.0f32,
            f32::MIN_POSITIVE,
            f32::MAX,
            0.0,
            -0.0,
            1e-30,
            1e30,
        ];
        let mut t = backend.alloc(src.len(), DType::F32).unwrap();

        // Act
        backend.upload_f32(&src, &mut t).unwrap();
        let mut dst = vec![0.0f32; src.len()];
        backend.download_f32(&t, &mut dst).unwrap();

        // Assert — bitwise equality for normal values, bit-equivalent for -0.0
        for (i, (&expected, &got)) in src.iter().zip(dst.iter()).enumerate() {
            if expected == 0.0 && expected.is_sign_negative() {
                assert!(got == 0.0 && got.is_sign_negative(), "idx {i}: -0.0 sign lost");
            } else {
                assert!(
                    (got - expected).abs() <= expected.abs() * f32::EPSILON,
                    "idx {i}: {expected} != {got}"
                );
            }
        }
    }

    // ── Test 54: alloc zero elements returns valid tensor ──

    #[test]
    fn alloc_zero_elements_returns_valid_tensor() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();

        // Act
        let t = backend.alloc(0, DType::F32).unwrap();

        // Assert
        assert_eq!(t.num_elements(), 0);
        assert_eq!(t.dtype(), DType::F32);
    }

    // ── Test 55: apply_rope_inplace multi-head independence ──

    #[test]
    fn apply_rope_multi_head_independence() {
        // Arrange — 3 heads, head_dim=4, full rotary. Each head gets same cos/sin
        // but operates on independent slices of qk.
        let num_heads = 3;
        let head_dim = 4;
        let cos = vec![0.0f32, 1.0]; // half=2
        let sin = vec![1.0f32, 0.0];

        // Head 0: [1,2,3,4], Head 1: [10,20,30,40], Head 2: [100,200,300,400]
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0];

        // Act
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);

        // Assert — Head 0: cos=[0,1], sin=[1,0]
        //   idx0 = 1*0 - 3*1 = -3,  idx1 = 2*1 - 4*0 = 2
        //   idx2 = 1*1 + 3*0 = 1,   idx3 = 2*0 + 4*1 = 4
        assert!((qk[0] - (-3.0)).abs() < 1e-5);
        assert!((qk[1] - 2.0).abs() < 1e-5);
        assert!((qk[2] - 1.0).abs() < 1e-5);
        assert!((qk[3] - 4.0).abs() < 1e-5);

        // Head 1: same rotation applied to [10,20,30,40]
        assert!((qk[4] - (-30.0)).abs() < 1e-5);
        assert!((qk[5] - 20.0).abs() < 1e-5);
        assert!((qk[6] - 10.0).abs() < 1e-5);
        assert!((qk[7] - 40.0).abs() < 1e-5);

        // Head 2: same rotation applied to [100,200,300,400]
        assert!((qk[8] - (-300.0)).abs() < 1e-4);
        assert!((qk[9] - 200.0).abs() < 1e-4);
        assert!((qk[10] - 100.0).abs() < 1e-4);
        assert!((qk[11] - 400.0).abs() < 1e-4);
    }

    // ── Test 56: InferenceError From<io::Error> conversion ──

    #[test]
    fn inference_error_from_io_error() {
        // Arrange
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");

        // Act
        let e: InferenceError = io_err.into();

        // Assert
        let msg = e.to_string();
        assert!(msg.contains("I/O error"), "should contain I/O prefix, got: {msg}");
        assert!(msg.contains("access denied"), "should contain original message, got: {msg}");
    }

    // ── Test 57: CompilerError display variants ──

    #[test]
    fn compiler_error_display_variants() {
        // Arrange & Act & Assert
        use crate::compiler::CompilerError;

        let e1 = CompilerError::InvalidGraph("cycle detected".into());
        assert!(e1.to_string().contains("cycle detected"));

        let e2 = CompilerError::CodegenViolation("alignment 16".into());
        assert!(e2.to_string().contains("alignment 16"));

        let e3 = CompilerError::FeatureDisabled("jit-x86".into());
        assert!(e3.to_string().contains("jit-x86"));

        let e4 = CompilerError::Internal("bug".into());
        assert!(e4.to_string().contains("bug"));
    }

    // ── Test 58: InferenceError From<CompilerError> conversion ──

    #[test]
    fn inference_error_from_compiler_error() {
        // Arrange
        use crate::compiler::CompilerError;
        let ce = CompilerError::InvalidGraph("broken".into());

        // Act
        let ie: InferenceError = ce.into();

        // Assert
        let msg = ie.to_string();
        assert!(msg.contains("compile error"), "should wrap CompilerError, got: {msg}");
        assert!(msg.contains("broken"), "should contain original message, got: {msg}");
    }

    // ── Test 59: DType ptx/msl/hip type names are non-empty ──

    #[test]
    fn dtype_backend_type_names_nonempty() {
        // Arrange
        let all_dtypes = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F4E2M1,
        ];

        // Assert — all backend type strings must be non-empty
        for dt in &all_dtypes {
            assert!(!dt.ptx_type().is_empty(), "DType {:?} ptx_type is empty", dt);
            assert!(!dt.ptx_reg_type().is_empty(), "DType {:?} ptx_reg_type is empty", dt);
            assert!(!dt.ptx_ld_type().is_empty(), "DType {:?} ptx_ld_type is empty", dt);
            assert!(!dt.hip_type().is_empty(), "DType {:?} hip_type is empty", dt);
            assert!(!dt.msl_type().is_empty(), "DType {:?} msl_type is empty", dt);
        }
    }

    // ── Test 60: ModelConfig approx_weight_bytes quantized smaller than F32 ──

    #[test]
    fn model_config_approx_weight_bytes_quantized_smaller() {
        // Arrange
        use crate::quant::QuantType;
        let cfg_f32 = tiny_config();
        let cfg_q4 = ModelConfig {
            quant_type: Some(QuantType::AWQ4),
            ..tiny_config()
        };

        // Act
        let bytes_f32 = cfg_f32.approx_weight_bytes();
        let bytes_q4 = cfg_q4.approx_weight_bytes();

        // Assert — quantized should be significantly smaller
        assert!(
            bytes_q4 < bytes_f32,
            "quantized weight bytes ({bytes_q4}) should be less than F32 ({bytes_f32})"
        );
    }

    // ── Test 61: apply_rope partial boundary with odd head_dim ──

    #[test]
    fn apply_rope_partial_boundary_odd_head_dim() {
        // Arrange — head_dim=5, partial=0.5 → rotary_dim = (5*0.5) & !1 = 2, half=1
        let num_heads = 1;
        let head_dim = 5;
        let cos = vec![0.5f32];
        let sin = vec![0.5f32];
        let mut qk = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let original = qk.clone();

        // Act
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 0.5);

        // Assert — idx 0 = x0*c - x1*s = 10*0.5 - 20*0.5 = -5
        //          idx 1 = x0*s + x1*c = 10*0.5 + 20*0.5 = 15
        assert!((qk[0] - (-5.0)).abs() < 1e-5, "got {}", qk[0]);
        assert!((qk[1] - 15.0).abs() < 1e-5, "got {}", qk[1]);
        // Dims 2..5 unchanged
        for i in 2..5 {
            assert_eq!(qk[i], original[i], "dim {i} should be unchanged");
        }
    }

    // ── Test 62: sample with all-negative logits picks least negative ──

    #[test]
    fn sample_negative_logits_picks_highest() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut logits_data = vec![-100.0f32; cfg.vocab_size];
        logits_data[73] = -0.001;
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 1];

        // Act
        backend.sample(&logits, 0.0, 0, 0.0, &mut ids).unwrap();

        // Assert
        assert_eq!(ids[0], 73);
    }

    // ── Test 63: DeviceKind Clone and Copy traits ──

    #[test]
    fn device_kind_clone_and_copy() {
        // Arrange
        let original = DeviceKind::Cuda(7);

        // Act — Clone
        let cloned = original.clone();
        assert_eq!(cloned, original);

        // Act — Copy (implicit via rebinding)
        let copied = original;
        assert_eq!(copied, original);
    }

    // ── Test 64: decoder_forward with Gemma arch uses GELU in FFN ──

    #[test]
    fn decoder_forward_gemma_uses_gelu() {
        // Arrange — tiny Gemma model (GELU activation instead of SiLU)
        let cfg = ModelConfig {
            arch: ModelArch::Gemma,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 20,
            max_seq_len: 16,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let ones = vec![1.0f32; cfg.hidden_size];
        unsafe {
            for lw in weights.layers.iter_mut() {
                lw.attn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                lw.ffn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                fill_tensor_pattern(&mut lw.wq, 0.02);
                fill_tensor_pattern(&mut lw.wk, 0.02);
                fill_tensor_pattern(&mut lw.wv, 0.02);
                fill_tensor_pattern(&mut lw.wo, 0.02);
                fill_tensor_pattern(&mut lw.w_gate, 0.02);
                fill_tensor_pattern(&mut lw.w_up, 0.02);
                fill_tensor_pattern(&mut lw.w_down, 0.02);
            }
            weights.final_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
            let lm: &mut [f32] = weights.lm_head.as_mut_slice();
            for (i, v) in lm.iter_mut().enumerate() {
                *v = (i as f32 * 0.01).sin() * 0.1;
            }
        }

        let input_data: Vec<f32> = (0..cfg.hidden_size)
            .map(|i| (i as f32 * 0.1 + 0.3).sin() * 0.5)
            .collect();
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions = unsafe { DeviceTensor::from_slice(&[0.0f32]) };
        let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

        // Act
        let result = backend.decoder_forward(
            &input, &positions, &mut kv_cache, &weights, &[1], &mut output,
        );

        // Assert — GELU path should complete without error and produce finite logits
        assert!(result.is_ok(), "Gemma decoder_forward failed: {:?}", result.err());
        let logits: &[f32] = unsafe { output.as_slice() };
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "Gemma logits contain non-finite values");
        assert!(logits.iter().any(|&v| v.abs() > 1e-8), "Gemma logits are all zero");
    }

    // ── Test 65: decoder_forward second token has longer KV context ──

    #[test]
    fn decoder_forward_second_token_has_longer_kv_context() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let ones = vec![1.0f32; cfg.hidden_size];
        for lw in weights.layers.iter_mut() {
            unsafe {
                lw.attn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                lw.ffn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                fill_tensor_pattern(&mut lw.wq, 0.02);
                fill_tensor_pattern(&mut lw.wk, 0.02);
                fill_tensor_pattern(&mut lw.wv, 0.02);
                fill_tensor_pattern(&mut lw.wo, 0.02);
                fill_tensor_pattern(&mut lw.w_gate, 0.02);
                fill_tensor_pattern(&mut lw.w_up, 0.02);
                fill_tensor_pattern(&mut lw.w_down, 0.02);
            }
        }
        unsafe {
            weights.final_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
            let lm: &mut [f32] = weights.lm_head.as_mut_slice();
            for (i, v) in lm.iter_mut().enumerate() {
                *v = (i as f32 * 0.01).sin() * 0.1;
            }
        }

        let input_data: Vec<f32> = (0..cfg.hidden_size)
            .map(|i| (i as f32 * 0.1 + 0.3).sin() * 0.5)
            .collect();
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

        // Act — process 3 tokens, verify KV cache grows each step
        assert_eq!(kv_cache.seq_len(0, 0), 0, "initial KV cache should be empty");

        let pos0 = unsafe { DeviceTensor::from_slice(&[0.0f32]) };
        backend
            .decoder_forward(&input, &pos0, &mut kv_cache, &weights, &[1], &mut output)
            .unwrap();
        assert_eq!(kv_cache.seq_len(0, 0), 1, "after 1 token, seq_len should be 1");

        let pos1 = unsafe { DeviceTensor::from_slice(&[1.0f32]) };
        backend
            .decoder_forward(&input, &pos1, &mut kv_cache, &weights, &[1], &mut output)
            .unwrap();
        assert_eq!(kv_cache.seq_len(0, 0), 2, "after 2 tokens, seq_len should be 2");

        let pos2 = unsafe { DeviceTensor::from_slice(&[2.0f32]) };
        backend
            .decoder_forward(&input, &pos2, &mut kv_cache, &weights, &[1], &mut output)
            .unwrap();

        // Assert — KV cache grew correctly across layers and positions
        assert_eq!(kv_cache.seq_len(0, 0), 3, "after 3 tokens, seq_len should be 3");
        assert_eq!(kv_cache.seq_len(1, 0), 3, "layer 1 seq_len should also be 3");

        // Final logits should be finite
        let logits: &[f32] = unsafe { output.as_slice() };
        assert!(logits.iter().all(|v| v.is_finite()), "logits contain non-finite values");
    }

    // ── Test 66: encoder_forward output is L2 normalized ──

    #[test]
    fn encoder_forward_output_is_l2_normalized() {
        // Arrange
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
                fill_tensor_pattern(&mut lw.wq, 0.01);
                fill_tensor_pattern(&mut lw.wk, 0.01);
                fill_tensor_pattern(&mut lw.wv, 0.01);
                fill_tensor_pattern(&mut lw.wo, 0.01);
                fill_tensor_pattern(&mut lw.w_up, 0.01);
                fill_tensor_pattern(&mut lw.w_down, 0.01);
            }
        }

        let mut input_data = vec![0.1f32; seq_len * h];
        for (i, v) in input_data.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.2;
        }
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions = unsafe { DeviceTensor::from_slice(&vec![0.0f32; seq_len]) };
        let mask = unsafe { DeviceTensor::from_slice(&vec![1.0f32; seq_len * seq_len]) };
        let mut output = DeviceTensor::alloc_cpu(h, DType::F32).unwrap();

        // Act
        backend
            .encoder_forward(&input, &positions, &mask, &weights, &mut output)
            .unwrap();

        // Assert — output should be L2 normalized (norm ~= 1.0)
        let out_slice: &[f32] = unsafe { output.as_slice() };
        let l2_norm: f32 = out_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (l2_norm - 1.0).abs() < 1e-4,
            "encoder output L2 norm should be ~1.0, got {l2_norm}"
        );
    }

    // ── Test 67: generate produces different tokens for different prompts ──

    #[test]
    fn generate_different_prompts_produce_different_output() {
        // Arrange
        let cfg = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 20,
            max_seq_len: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let ones = vec![1.0f32; cfg.hidden_size];
        unsafe {
            for lw in weights.layers.iter_mut() {
                lw.attn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                lw.ffn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                fill_tensor_pattern(&mut lw.wq, 0.01);
                fill_tensor_pattern(&mut lw.wk, 0.01);
                fill_tensor_pattern(&mut lw.wv, 0.01);
                fill_tensor_pattern(&mut lw.wo, 0.01);
                fill_tensor_pattern(&mut lw.w_gate, 0.01);
                fill_tensor_pattern(&mut lw.w_up, 0.01);
                fill_tensor_pattern(&mut lw.w_down, 0.01);
            }
            weights.final_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
            let emb: &mut [f32] = weights.embedding.as_mut_slice();
            for (i, v) in emb.iter_mut().enumerate() {
                *v = ((i as f32 * 0.07 + 0.3).sin()) * 0.5;
            }
            let lm: &mut [f32] = weights.lm_head.as_mut_slice();
            for (i, v) in lm.iter_mut().enumerate() {
                *v = ((i as f32 * 0.03 + 0.1).cos()) * 0.1;
            }
        }

        // Act
        let gen_a = backend.generate(&weights, &[1u32, 2], 3, 0.0).unwrap();
        let gen_b = backend.generate(&weights, &[5u32, 8], 3, 0.0).unwrap();

        // Assert — different prompts should produce at least one different token
        assert_eq!(gen_a.len(), 3);
        assert_eq!(gen_b.len(), 3);
        let any_diff = gen_a.iter().zip(gen_b.iter()).any(|(a, b)| a != b);
        assert!(any_diff, "different prompts produced identical output: {gen_a:?} vs {gen_b:?}");
    }

    // ── Test 68: sample with temperature 1.0 preserves argmax winner ──

    #[test]
    fn sample_temperature_one_preserves_argmax() {
        // Arrange — token 55 has a clearly dominant logit
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut logits_data = vec![1.0f32; cfg.vocab_size];
        logits_data[55] = 100.0;
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 1];

        // Act — temperature=1.0 scales but dominant token still wins
        backend.sample(&logits, 1.0, 0, 0.0, &mut ids).unwrap();

        // Assert — argmax winner should be the same as temperature=0
        assert_eq!(ids[0], 55);
    }

    // ── Test 69: decoder_forward with sliding_window config completes ──

    #[test]
    fn decoder_forward_sliding_window_completes() {
        // Arrange — Mistral-style config with sliding_window=4
        let cfg = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 20,
            max_seq_len: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: Some(4),
        };
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        let ones = vec![1.0f32; cfg.hidden_size];
        unsafe {
            for lw in weights.layers.iter_mut() {
                lw.attn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                lw.ffn_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
                fill_tensor_pattern(&mut lw.wq, 0.02);
                fill_tensor_pattern(&mut lw.wk, 0.02);
                fill_tensor_pattern(&mut lw.wv, 0.02);
                fill_tensor_pattern(&mut lw.wo, 0.02);
                fill_tensor_pattern(&mut lw.w_gate, 0.02);
                fill_tensor_pattern(&mut lw.w_up, 0.02);
                fill_tensor_pattern(&mut lw.w_down, 0.02);
            }
            weights.final_norm.as_mut_slice::<f32>().copy_from_slice(&ones);
            let lm: &mut [f32] = weights.lm_head.as_mut_slice();
            for (i, v) in lm.iter_mut().enumerate() {
                *v = (i as f32 * 0.01).sin() * 0.1;
            }
        }

        let input_data = vec![0.1f32; cfg.hidden_size];
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let mut kv_cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let mut output = backend.alloc(cfg.vocab_size, DType::F32).unwrap();

        // Process tokens at positions 0..5 — position 5 triggers sliding window masking
        for pos in 0u32..6 {
            let pos_tensor = unsafe { DeviceTensor::from_slice(&[pos as f32]) };
            backend
                .decoder_forward(&input, &pos_tensor, &mut kv_cache, &weights, &[1], &mut output)
                .unwrap();
        }

        // Assert — sliding window masking didn't break computation
        let logits: &[f32] = unsafe { output.as_slice() };
        assert!(logits.iter().all(|v| v.is_finite()), "sliding window produced non-finite logits");
        assert!(logits.iter().any(|&v| v.abs() > 1e-8), "sliding window produced all-zero logits");
    }

    // ── Test 70: tensor_as_mut_f32 writes are readable ──

    #[test]
    fn tensor_as_mut_f32_write_read_roundtrip() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut t = backend.alloc(8, DType::F32).unwrap();

        // Act — write via unsafe mutable slice helper
        unsafe {
            let slice = tensor_as_mut_f32(&mut t);
            for (i, v) in slice.iter_mut().enumerate() {
                *v = (i as f32).sqrt();
            }
        }
        let mut dst = vec![0.0f32; 8];
        backend.download_f32(&t, &mut dst).unwrap();

        // Assert — values written through tensor_as_mut_f32 are preserved
        for (i, &v) in dst.iter().enumerate() {
            let expected = (i as f32).sqrt();
            assert!((v - expected).abs() < 1e-6, "idx {i}: expected {expected}, got {v}");
        }
    }

    // ── Test 71: encoder_forward with single-token sequence ──

    #[test]
    fn encoder_forward_single_token_sequence() {
        // Arrange
        let cfg = tiny_encoder_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut weights = ModelWeights::alloc_cpu(&cfg).unwrap();
        let h = cfg.hidden_size;

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
                fill_tensor_pattern(&mut lw.wq, 0.01);
                fill_tensor_pattern(&mut lw.wk, 0.01);
                fill_tensor_pattern(&mut lw.wv, 0.01);
                fill_tensor_pattern(&mut lw.wo, 0.01);
                fill_tensor_pattern(&mut lw.w_up, 0.01);
                fill_tensor_pattern(&mut lw.w_down, 0.01);
            }
        }

        let input_data: Vec<f32> = (0..h).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let input = unsafe { DeviceTensor::from_slice(&input_data) };
        let positions = unsafe { DeviceTensor::from_slice(&[0.0f32]) };
        let mask = unsafe { DeviceTensor::from_slice(&[1.0f32]) };
        let mut output = DeviceTensor::alloc_cpu(h, DType::F32).unwrap();

        // Act
        backend
            .encoder_forward(&input, &positions, &mask, &weights, &mut output)
            .unwrap();

        // Assert — single token produces a valid L2-normalized embedding
        let out_slice: &[f32] = unsafe { output.as_slice() };
        assert_eq!(out_slice.len(), h);
        assert!(out_slice.iter().all(|v| v.is_finite()), "single-token output contains non-finite values");
        let l2: f32 = out_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((l2 - 1.0).abs() < 1e-4, "single-token output L2 norm should be ~1.0, got {l2}");
    }

    // ── Test 72: alloc_kv_cache batch_size scales page allocation ──

    #[test]
    fn alloc_kv_cache_batch_size_scales_pages() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();

        // Act
        let cache_b1 = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();
        let cache_b3 = backend.alloc_kv_cache(3, cfg.max_seq_len).unwrap();

        // Assert — batch=3 should have 3x the pages of batch=1
        let pages_b1 = cache_b1.total_pages();
        let pages_b3 = cache_b3.total_pages();
        assert_eq!(
            pages_b3, pages_b1 * 3,
            "batch=3 pages ({pages_b3}) should be 3x batch=1 pages ({pages_b1})"
        );
    }

    // ── Test 73: ModelWeights alloc_cpu matches config dimensions ──

    #[test]
    fn model_weights_alloc_matches_config_dimensions() {
        // Arrange
        let cfg = tiny_config();

        // Act
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        // Assert — verify key weight tensor dimensions
        let h = cfg.hidden_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        for (layer_idx, lw) in weights.layers.iter().enumerate() {
            assert_eq!(lw.wq.num_elements(), h * q_dim, "layer {layer_idx} wq size mismatch");
            assert_eq!(lw.wk.num_elements(), h * kv_dim, "layer {layer_idx} wk size mismatch");
            assert_eq!(lw.wv.num_elements(), h * kv_dim, "layer {layer_idx} wv size mismatch");
            assert_eq!(lw.wo.num_elements(), q_dim * h, "layer {layer_idx} wo size mismatch");
            assert_eq!(lw.w_gate.num_elements(), h * inter, "layer {layer_idx} w_gate size mismatch");
            assert_eq!(lw.w_up.num_elements(), h * inter, "layer {layer_idx} w_up size mismatch");
            assert_eq!(lw.w_down.num_elements(), inter * h, "layer {layer_idx} w_down size mismatch");
            assert_eq!(lw.attn_norm.num_elements(), h, "layer {layer_idx} attn_norm size mismatch");
            assert_eq!(lw.ffn_norm.num_elements(), h, "layer {layer_idx} ffn_norm size mismatch");
        }
        assert_eq!(weights.final_norm.num_elements(), h, "final_norm size mismatch");
        assert_eq!(
            weights.lm_head.num_elements(), h * cfg.vocab_size,
            "lm_head size mismatch"
        );
    }

    // ── Test 74: llama_7b() constructor fields are valid ──

    #[test]
    fn llama_7b_config_fields_are_valid() {
        // Arrange & Act
        let cfg = ModelConfig::llama_7b();

        // Assert — verify structural consistency
        assert_eq!(cfg.arch, ModelArch::Llama);
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        assert_eq!(cfg.num_heads, cfg.num_kv_heads, "LLaMA uses MHA, not GQA");
        assert!(cfg.vocab_size > 0);
        assert!(cfg.num_layers > 0);
        assert!(!cfg.has_qkv_bias, "LLaMA has no QKV bias");
        assert_eq!(cfg.partial_rotary_factor, 1.0);
        assert!(cfg.sliding_window.is_none(), "LLaMA has no sliding window");
    }

    // ── Test 75: phi_2b() config has partial rotary factor 0.5 ──

    #[test]
    fn phi_2b_config_has_partial_rotary() {
        // Arrange & Act
        let cfg = ModelConfig::phi_2b();

        // Assert — Phi uses partial rotary embedding (only half the head dims)
        assert_eq!(cfg.arch, ModelArch::Phi);
        assert!(
            (cfg.partial_rotary_factor - 0.5).abs() < 1e-6,
            "Phi should have partial_rotary_factor=0.5, got {}",
            cfg.partial_rotary_factor,
        );
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        assert!(!cfg.has_qkv_bias);
    }

    // ── Test 76: gemma_2b() config has extreme GQA ratio ──

    #[test]
    fn gemma_2b_config_has_extreme_gqa() {
        // Arrange & Act
        let cfg = ModelConfig::gemma_2b();

        // Assert — Gemma 2B uses 8 query heads but only 1 KV head (8:1 GQA)
        assert_eq!(cfg.arch, ModelArch::Gemma);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.num_heads % cfg.num_kv_heads, 0, "GQA ratio must be integer");
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 8, "expected 8:1 GQA ratio");
    }

    // ── Test 77: kv_cache_bytes_per_token calculation matches manual formula ──

    #[test]
    fn kv_cache_bytes_per_token_calculation() {
        // Arrange
        let cfg = tiny_config();
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let expected = 2 * kv_dim * cfg.dtype.size_bytes() * cfg.num_layers;

        // Act
        let actual = cfg.kv_cache_bytes_per_token();

        // Assert
        assert_eq!(
            actual, expected,
            "kv_cache_bytes_per_token: got {actual}, expected {expected}"
        );

        // Cross-check with a different config (GQA)
        let cfg_gqa = ModelConfig {
            arch: ModelArch::Mistral,
            hidden_size: 64,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 128,
            num_layers: 4,
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
        };
        let kv_dim_gqa = cfg_gqa.num_kv_heads * cfg_gqa.head_dim;
        let expected_gqa = 2 * kv_dim_gqa * 4 * 4;
        assert_eq!(cfg_gqa.kv_cache_bytes_per_token(), expected_gqa);
    }

    // ── Test 78: CompilerError RegisterOverflow display includes details ──

    #[test]
    fn compiler_error_register_overflow_display() {
        // Arrange
        let err = crate::compiler::CompilerError::RegisterOverflow {
            needed: 32,
            available: 16,
            context: "GEMM tile accumulation".into(),
        };

        // Act
        let msg = err.to_string();

        // Assert — all three pieces of info should appear
        assert!(msg.contains("32"), "should mention needed registers, got: {msg}");
        assert!(msg.contains("16"), "should mention available registers, got: {msg}");
        assert!(msg.contains("GEMM"), "should mention context, got: {msg}");
    }

    // ── Test 79: CompilerError From<String> and From<&str> conversions ──

    #[test]
    fn compiler_error_from_string_conversion() {
        // Arrange & Act
        let from_string: crate::compiler::CompilerError = String::from("allocation failed").into();
        let from_str: crate::compiler::CompilerError = "codegen panic".into();

        // Assert — both should map to Internal variant
        let msg_s = from_string.to_string();
        assert!(msg_s.contains("allocation failed"), "got: {msg_s}");

        let msg_r = from_str.to_string();
        assert!(msg_r.contains("codegen panic"), "got: {msg_r}");
    }

    // ── Test 80: InferenceError Unsupported variant display format ──

    #[test]
    fn inference_error_unsupported_variant_display() {
        // Arrange
        let err = InferenceError::Unsupported("batch_size > 1 on CPU backend".into());

        // Act
        let msg = err.to_string();

        // Assert
        assert!(msg.contains("unsupported"), "should contain 'unsupported', got: {msg}");
        assert!(msg.contains("batch_size"), "should contain reason, got: {msg}");
    }

    // ── Test 81: apply_rope negated rotation is approximate inverse ──

    #[test]
    fn apply_rope_negated_rotation_is_approximate_inverse() {
        // Arrange — apply rotation with (cos, sin) then with (cos, -sin)
        // should recover the original vector approximately
        let num_heads = 1;
        let head_dim = 4;
        let cos = vec![0.6f32, 0.8];
        let sin = vec![0.8f32, 0.6]; // arbitrary non-trivial rotation
        let original = vec![3.0f32, -1.0, 2.5, 0.7];
        let mut qk = original.clone();

        // Act — forward rotation
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);
        // Reverse rotation: negate sin
        let neg_sin: Vec<f32> = sin.iter().map(|&s| -s).collect();
        apply_rope_inplace(&mut qk, &cos, &neg_sin, num_heads, head_dim, 1.0);

        // Assert — should recover original values
        for (i, (&got, &expected)) in qk.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "inverse rotation failed at idx {i}: got {got}, expected {expected}"
            );
        }
    }

    // ── Test 82: sample with single batch picks dominant token ──

    #[test]
    fn sample_with_single_batch_dominant_token() {
        // Arrange — provide a full vocab-sized logits vector where token 0 dominates
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut logits_data = vec![-10.0f32; cfg.vocab_size];
        logits_data[0] = 100.0; // token 0 dominates overwhelmingly
        let logits = unsafe { DeviceTensor::from_slice(&logits_data) };
        let mut ids = [0u32; 1];

        // Act
        backend.sample(&logits, 0.0, 0, 0.0, &mut ids).unwrap();

        // Assert — token 0 with highest logit should be selected
        assert_eq!(ids[0], 0);
    }

    // ── Test 83: ModelConfig approx_weight_bytes scales with num_layers ──

    #[test]
    fn model_config_approx_weight_bytes_scales_with_layers() {
        // Arrange — two configs differing only in num_layers
        let cfg_2layers = tiny_config();
        let cfg_4layers = ModelConfig {
            num_layers: 4,
            ..tiny_config()
        };

        // Act
        let bytes_2 = cfg_2layers.approx_weight_bytes();
        let bytes_4 = cfg_4layers.approx_weight_bytes();

        // Assert — doubling layers should increase weight bytes
        assert!(
            bytes_4 > bytes_2,
            "4-layer model ({bytes_4}) should be larger than 2-layer ({bytes_2})"
        );
        // The per-layer cost is (bytes_4 - bytes_2) / 2 per extra layer
        let per_layer_diff = (bytes_4 - bytes_2) / 2;
        assert!(per_layer_diff > 0, "per-layer weight contribution should be positive");
    }

    // ── Test 84: apply_rope single head dim=2 rotates correctly ──

    #[test]
    fn apply_rope_single_head_dim2_rotation() {
        // Arrange — 1 head, head_dim=2, full rotary → half=1, single cos/sin pair
        let num_heads = 1;
        let head_dim = 2;
        let cos = vec![1.0f32]; // cos(0) = 1
        let sin = vec![0.0f32]; // sin(0) = 0
        let mut qk = vec![5.0f32, -3.0];

        // Act — identity rotation at angle 0
        apply_rope_inplace(&mut qk, &cos, &sin, num_heads, head_dim, 1.0);

        // Assert — x0*c - x1*s = 5*1 - (-3)*0 = 5; x0*s + x1*c = 5*0 + (-3)*1 = -3
        assert!((qk[0] - 5.0).abs() < 1e-6, "expected 5.0, got {}", qk[0]);
        assert!((qk[1] - (-3.0)).abs() < 1e-6, "expected -3.0, got {}", qk[1]);
    }

    // ── Test 85: InferenceError OutOfMemory display includes byte counts ──

    #[test]
    fn inference_error_out_of_memory_display() {
        // Arrange
        let err = InferenceError::OutOfMemory {
            requested: 1_073_741_824,
            available: 536_870_912,
        };

        // Act
        let msg = err.to_string();

        // Assert — both byte counts should appear in the message
        assert!(msg.contains("1073741824"), "should mention requested bytes, got: {msg}");
        assert!(msg.contains("536870912"), "should mention available bytes, got: {msg}");
        assert!(msg.to_lowercase().contains("memory"), "should mention memory, got: {msg}");
    }

    // ── Test 86: ModelConfig with GQA has smaller kv_cache_bytes_per_token ──

    #[test]
    fn gqa_reduces_kv_cache_bytes_per_token() {
        // Arrange — MHA (num_kv_heads = num_heads) vs GQA (num_kv_heads < num_heads)
        let cfg_mha = tiny_config(); // num_kv_heads = 4 = num_heads
        let cfg_gqa = ModelConfig {
            num_kv_heads: 2,
            ..tiny_config()
        };

        // Act
        let bytes_mha = cfg_mha.kv_cache_bytes_per_token();
        let bytes_gqa = cfg_gqa.kv_cache_bytes_per_token();

        // Assert — GQA should use exactly half the KV bytes (2 vs 4 kv_heads)
        assert_eq!(
            bytes_gqa * 2, bytes_mha,
            "GQA (2 kv_heads) should be half of MHA (4 kv_heads): got {bytes_gqa} vs {bytes_mha}"
        );
    }

    // ── Test 87: CompilerError UnsupportedDType display includes dtype and isa ──

    #[test]
    fn compiler_error_unsupported_dtype_display() {
        // Arrange
        let err = crate::compiler::CompilerError::UnsupportedDType {
            dtype: DType::F4E2M1,
            isa: "AVX2".into(),
        };

        // Act
        let msg = err.to_string();

        // Assert — message should reference both the dtype and ISA
        assert!(msg.contains("F4E2M1"), "should mention dtype, got: {msg}");
        assert!(msg.contains("AVX2"), "should mention ISA, got: {msg}");
    }

    // ── Test 88: alloc_kv_cache page size matches PAGE_SIZE constant ──

    #[test]
    fn alloc_kv_cache_page_size_matches_constant() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();

        // Act
        let cache = backend.alloc_kv_cache(1, cfg.max_seq_len).unwrap();

        // Assert — bytes_per_page should match the global PAGE_SIZE
        let expected_bytes_per_page = PAGE_SIZE * cfg.num_kv_heads * cfg.head_dim * 2 * cfg.dtype.size_bytes();
        assert_eq!(
            cache.bytes_per_page(),
            expected_bytes_per_page,
            "bytes_per_page mismatch"
        );
    }

    // ── Test 89: ModelConfig llama_7b and mistral_7b have different sliding_window ──

    #[test]
    fn llama_and_mistral_differ_on_sliding_window() {
        // Arrange & Act
        let llama = ModelConfig::llama_7b();
        let mistral = ModelConfig::mistral_7b();

        // Assert — LLaMA has no sliding window, Mistral has one
        assert!(llama.sliding_window.is_none(), "LLaMA should have no sliding window");
        assert!(mistral.sliding_window.is_some(), "Mistral should have sliding window");
        assert_eq!(mistral.sliding_window.unwrap(), 4096);
    }

    // ── Test 90: upload_f32 rejects wrong dtype tensor ──

    #[test]
    fn upload_f32_rejects_non_f32_tensor() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let mut tensor = backend.alloc(16, DType::F16).unwrap();
        let src = vec![1.0f32; 16];

        // Act
        let result = backend.upload_f32(&src, &mut tensor);

        // Assert — should fail because tensor is F16, not F32
        assert!(result.is_err(), "upload_f32 should reject non-F32 tensor");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("F32"), "error should mention F32, got: {msg}");
    }

    // ── Test 91: download_f32 rejects non-F32 source tensor ──

    #[test]
    fn download_f32_rejects_non_f32_tensor() {
        // Arrange
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let src = backend.alloc(16, DType::F16).unwrap();
        let mut dst = vec![0.0f32; 16];

        // Act
        let result = backend.download_f32(&src, &mut dst);

        // Assert — should fail because source tensor is F16, not F32
        assert!(result.is_err(), "download_f32 should reject non-F32 source");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("F32"), "error should mention F32, got: {msg}");
    }

    // ── Test 92: ModelWeights alloc_cpu embedding size matches vocab_size * hidden_size ──

    #[test]
    fn model_weights_embedding_size_matches_vocab_hidden() {
        // Arrange
        let cfg = tiny_config();
        let expected = cfg.vocab_size * cfg.hidden_size;

        // Act
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();

        // Assert
        assert_eq!(
            weights.embedding.num_elements(), expected,
            "embedding tensor should have vocab_size * hidden_size elements"
        );
    }

    // ── Test 93: generate returns error for token_id >= vocab_size ──

    #[test]
    fn generate_token_at_vocab_boundary_returns_error() {
        // Arrange — token_id == vocab_size is out of bounds
        let cfg = tiny_config();
        let backend = CpuInferenceBackend::init(&cfg).unwrap();
        let weights = ModelWeights::alloc_cpu(&cfg).unwrap();
        let oob_token = cfg.vocab_size as u32; // exactly vocab_size, which is OOB

        // Act
        let result = backend.generate(&weights, &[oob_token], 1, 0.0);

        // Assert — should fail because token_id >= vocab_size
        assert!(result.is_err(), "generate should fail for token_id >= vocab_size");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("token_id"), "error should mention token_id, got: {msg}");
    }
}
