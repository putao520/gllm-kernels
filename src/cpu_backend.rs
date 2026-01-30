use crate::backend_trait::Backend;
use crate::kernel_types::{
    FlashAttentionConfig,
    // L2 Block-level configs
    AttentionBlockConfig, FFNBlockConfig, EmbeddingConfig, LMHeadConfig,
    KVCacheUpdateConfig, MeanPoolingConfig, ClsPoolingConfig, NormalizeConfig,
    DequantizeConfig, EngramFuseConfig, QuantFormat, Activation,
    // L3 High-level inference configs (ARCH-ADR-003)
    TransformerLayerWeights, MoETransformerLayerWeights, KVCacheState,
    GeneratorForwardConfig, MoEGeneratorForwardConfig, EmbeddingForwardConfig,
    RerankForwardConfig, PoolingType,
    // GPU-pure weights (ARCH-GPU-001 / ARCH-ADR-010)
    EmbeddingModelWeightsGpu, TransformerLayerWeightsGpu,
    RerankerModelWeightsGpu, GeneratorModelWeightsGpu, KVCacheGpu,
    TransformerLayerConfigGpu,
};
use crate::gpu_types::{GpuBuffer, GpuTensor, TensorDtype};
use crate::ops::sampling::SamplingConfig;
use crate::runtime_detection::BackendType;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct CpuBackend {}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    // =========================================================================
    // L2 Block-Level Operators (ARCH-GRANULARITY-001)
    // =========================================================================

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
    ) -> Result<Vec<f32>, String> {
        let batch = config.batch_size;
        let seq_len = config.seq_len;
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_q_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;

        // 1. RMS Norm on input
        let mut normed = vec![0.0f32; batch * seq_len * hidden_size];
        crate::ops::rms_norm::rms_norm_forward(
            hidden,
            norm_weight,
            &mut normed,
            batch * seq_len,
            hidden_size,
            config.rms_norm_eps,
        );

        // 2. QKV projections
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut q = vec![0.0f32; batch * seq_len * q_dim];
        let mut k = vec![0.0f32; batch * seq_len * kv_dim];
        let mut v = vec![0.0f32; batch * seq_len * kv_dim];

        crate::ops::linear::linear_forward(
            &normed, q_weight, None, &mut q,
            batch * seq_len, hidden_size, q_dim,
        );
        crate::ops::linear::linear_forward(
            &normed, k_weight, None, &mut k,
            batch * seq_len, hidden_size, kv_dim,
        );
        crate::ops::linear::linear_forward(
            &normed, v_weight, None, &mut v,
            batch * seq_len, hidden_size, kv_dim,
        );

        // 3. Apply RoPE if enabled
        if config.use_rope {
            let mut q_rope = vec![0.0f32; q.len()];
            let mut k_rope = vec![0.0f32; k.len()];
            crate::ops::rope::rope_apply(
                &q, &k,
                cos_cache, sin_cache,
                &mut q_rope, &mut k_rope,
                batch, seq_len, num_q_heads, num_kv_heads, head_dim,
                config.position_offset,
            );
            q = q_rope;
            k = k_rope;
        }

        // 4. Update KV cache if provided
        if let (Some(k_cache), Some(v_cache)) = (kv_cache_k, kv_cache_v) {
            let cache_offset = config.position_offset * num_kv_heads * head_dim;
            let new_len = seq_len * num_kv_heads * head_dim;
            for b in 0..batch {
                let src_start = b * new_len;
                let dst_start = b * k_cache.len() / batch + cache_offset;
                k_cache[dst_start..dst_start + new_len].copy_from_slice(&k[src_start..src_start + new_len]);
                v_cache[dst_start..dst_start + new_len].copy_from_slice(&v[src_start..src_start + new_len]);
            }
        }

        // 5. Flash attention
        let flash_config = FlashAttentionConfig {
            batch_size: batch,
            num_heads: num_q_heads,
            num_kv_heads,
            head_dim,
            seq_len_q: seq_len,
            seq_len_kv: seq_len,
            causal: config.causal,
            scale: config.scale,
            dropout_prob: config.dropout_prob,
            ..Default::default()
        };

        let mut attn_out = vec![0.0f32; batch * seq_len * q_dim];
        crate::ops::attention::cpu_flash_attention(&q, &k, &v, &mut attn_out, flash_config);

        // 6. Output projection
        let mut output = vec![0.0f32; batch * seq_len * hidden_size];
        crate::ops::linear::linear_forward(
            &attn_out, o_weight, None, &mut output,
            batch * seq_len, q_dim, hidden_size,
        );

        // 7. Residual connection
        for (out, inp) in output.iter_mut().zip(hidden.iter()) {
            *out += inp;
        }

        Ok(output)
    }

    fn ffn_block(
        &self,
        hidden: &[f32],
        gate_weight: Option<&[f32]>,
        up_weight: &[f32],
        down_weight: &[f32],
        norm_weight: &[f32],
        config: &FFNBlockConfig,
    ) -> Result<Vec<f32>, String> {
        let batch = config.batch_size;
        let seq_len = config.seq_len;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        // 1. RMS Norm
        let mut normed = vec![0.0f32; batch * seq_len * hidden_size];
        crate::ops::rms_norm::rms_norm_forward(
            hidden,
            norm_weight,
            &mut normed,
            batch * seq_len,
            hidden_size,
            config.rms_norm_eps,
        );

        // 2. Up projection
        let mut up_out = vec![0.0f32; batch * seq_len * intermediate_size];
        crate::ops::linear::linear_forward(
            &normed, up_weight, None, &mut up_out,
            batch * seq_len, hidden_size, intermediate_size,
        );

        // 3. Gate projection and activation (LLaMA-style)
        if config.use_gate {
            if let Some(gate_w) = gate_weight {
                let mut gate_out = vec![0.0f32; batch * seq_len * intermediate_size];
                crate::ops::linear::linear_forward(
                    &normed, gate_w, None, &mut gate_out,
                    batch * seq_len, hidden_size, intermediate_size,
                );

                // Apply activation to gate and multiply with up
                match config.activation {
                    Activation::SiLU => {
                        crate::ops::activations::silu_mul_inplace(&mut gate_out, &up_out);
                    }
                    Activation::GELU | Activation::GELUExact => {
                        crate::ops::activations::gelu_inplace(&mut gate_out);
                        crate::ops::activations::mul_inplace(&mut gate_out, &up_out);
                    }
                    Activation::ReLU => {
                        crate::ops::activations::relu_inplace(&mut gate_out);
                        crate::ops::activations::mul_inplace(&mut gate_out, &up_out);
                    }
                    Activation::None => {
                        crate::ops::activations::mul_inplace(&mut gate_out, &up_out);
                    }
                }
                up_out = gate_out;
            }
        } else {
            // GPT-style: just activation on up projection
            match config.activation {
                Activation::SiLU => crate::ops::activations::silu_inplace(&mut up_out),
                Activation::GELU => crate::ops::activations::gelu_inplace(&mut up_out),
                Activation::GELUExact => crate::ops::activations::gelu_exact_inplace(&mut up_out),
                Activation::ReLU => crate::ops::activations::relu_inplace(&mut up_out),
                Activation::None => {}
            }
        }

        // 4. Down projection
        let mut output = vec![0.0f32; batch * seq_len * hidden_size];
        crate::ops::linear::linear_forward(
            &up_out, down_weight, None, &mut output,
            batch * seq_len, intermediate_size, hidden_size,
        );

        // 5. Residual connection
        for (out, inp) in output.iter_mut().zip(hidden.iter()) {
            *out += inp;
        }

        Ok(output)
    }

    fn embedding(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        position_weight: Option<&[f32]>,
        config: &EmbeddingConfig,
    ) -> Result<Vec<f32>, String> {
        let num_tokens = tokens.len();
        let hidden_size = config.hidden_size;

        let mut output = vec![0.0f32; num_tokens * hidden_size];

        // Token embedding lookup
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = token as usize;
            if token_idx >= config.vocab_size {
                return Err(format!("Token {} out of vocabulary range {}", token_idx, config.vocab_size));
            }
            let src_start = token_idx * hidden_size;
            let dst_start = i * hidden_size;
            output[dst_start..dst_start + hidden_size]
                .copy_from_slice(&embed_weight[src_start..src_start + hidden_size]);
        }

        // Add position embeddings if provided
        if config.add_position_embedding {
            if let Some(pos_weight) = position_weight {
                for (i, out_chunk) in output.chunks_mut(hidden_size).enumerate() {
                    let pos = i % config.max_seq_len;
                    let pos_start = pos * hidden_size;
                    for (j, val) in out_chunk.iter_mut().enumerate() {
                        *val += pos_weight[pos_start + j];
                    }
                }
            }
        }

        Ok(output)
    }

    fn lm_head(
        &self,
        hidden: &[f32],
        lm_weight: &[f32],
        norm_weight: &[f32],
        config: &LMHeadConfig,
    ) -> Result<Vec<f32>, String> {
        let batch = config.batch_size;
        let seq_len = config.seq_len;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;

        // 1. Final RMS norm
        let mut normed = vec![0.0f32; batch * seq_len * hidden_size];
        crate::ops::rms_norm::rms_norm_forward(
            hidden,
            norm_weight,
            &mut normed,
            batch * seq_len,
            hidden_size,
            config.rms_norm_eps,
        );

        // 2. Project to vocabulary
        let mut logits = vec![0.0f32; batch * seq_len * vocab_size];
        crate::ops::linear::linear_forward(
            &normed, lm_weight, None, &mut logits,
            batch * seq_len, hidden_size, vocab_size,
        );

        Ok(logits)
    }

    fn engram_lookup(
        &self,
        tokens: &[u32],
        engram_table: &[f32],
        hidden_size: usize,
        ngram_size: usize,
        num_buckets: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), String> {
        let num_tokens = tokens.len();
        let mut embeddings = vec![0.0f32; num_tokens * hidden_size];
        let mut bucket_indices = vec![0u64; num_tokens];

        // Use Engram hash function for lookup
        let hasher = crate::ops::engram_hash::EngramHasher::new(crate::ops::engram_hash::EngramHashConfig {
            ngram_size,
            num_buckets,
            ..Default::default()
        });

        for i in 0..num_tokens {
            let start = i.saturating_sub(ngram_size - 1);
            let ngram: Vec<u32> = tokens[start..=i].to_vec();
            let bucket = hasher.hash_ngram(&ngram);
            bucket_indices[i] = bucket;

            let bucket_idx = (bucket as usize) % num_buckets;
            if bucket_idx * hidden_size + hidden_size <= engram_table.len() {
                let src_start = bucket_idx * hidden_size;
                let dst_start = i * hidden_size;
                embeddings[dst_start..dst_start + hidden_size]
                    .copy_from_slice(&engram_table[src_start..src_start + hidden_size]);
            }
        }

        Ok((embeddings, bucket_indices))
    }

    fn engram_fuse(
        &self,
        attention_output: &[f32],
        engram_output: &[f32],
        config: &EngramFuseConfig,
    ) -> Result<Vec<f32>, String> {
        if attention_output.len() != engram_output.len() {
            return Err("Attention and Engram outputs must have same shape".to_string());
        }

        let mut output = vec![0.0f32; attention_output.len()];
        for i in 0..output.len() {
            output[i] = config.attention_scale * attention_output[i]
                + config.engram_scale * engram_output[i];
        }

        Ok(output)
    }

    fn kv_cache_update(
        &self,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        new_k: &[f32],
        new_v: &[f32],
        config: &KVCacheUpdateConfig,
    ) -> Result<(), String> {
        let batch = config.batch_size;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let num_new = config.num_new_tokens;
        let pos = config.position;

        let stride_per_batch = config.max_cache_len * num_kv_heads * head_dim;
        let new_stride = num_new * num_kv_heads * head_dim;

        for b in 0..batch {
            let cache_offset = b * stride_per_batch + pos * num_kv_heads * head_dim;
            let new_offset = b * new_stride;

            k_cache[cache_offset..cache_offset + new_stride]
                .copy_from_slice(&new_k[new_offset..new_offset + new_stride]);
            v_cache[cache_offset..cache_offset + new_stride]
                .copy_from_slice(&new_v[new_offset..new_offset + new_stride]);
        }

        Ok(())
    }

    fn sample(
        &self,
        logits: &[f32],
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        let batch_size = logits.len() / vocab_size;
        Ok(crate::ops::sampling::sample_tokens(logits, batch_size, vocab_size, config))
    }

    fn mean_pooling(
        &self,
        hidden: &[f32],
        attention_mask: Option<&[f32]>,
        config: &MeanPoolingConfig,
    ) -> Result<Vec<f32>, String> {
        let batch = config.batch_size;
        let seq_len = config.seq_len;
        let hidden_size = config.hidden_size;

        let mut output = vec![0.0f32; batch * hidden_size];

        for b in 0..batch {
            let mut sum = vec![0.0f32; hidden_size];
            let mut count = 0.0f32;

            for s in 0..seq_len {
                let mask_val = if let Some(mask) = attention_mask {
                    mask[b * seq_len + s]
                } else {
                    1.0
                };

                if mask_val > 0.0 {
                    let idx = (b * seq_len + s) * hidden_size;
                    for h in 0..hidden_size {
                        sum[h] += hidden[idx + h] * mask_val;
                    }
                    count += mask_val;
                }
            }

            if count > 0.0 {
                let out_idx = b * hidden_size;
                for h in 0..hidden_size {
                    output[out_idx + h] = sum[h] / count;
                }
            }
        }

        Ok(output)
    }

    fn cls_pooling(
        &self,
        hidden: &[f32],
        config: &ClsPoolingConfig,
    ) -> Result<Vec<f32>, String> {
        let batch = config.batch_size;
        let hidden_size = config.hidden_size;
        let cls_pos = config.cls_position;

        let seq_len = hidden.len() / (batch * hidden_size);
        let mut output = vec![0.0f32; batch * hidden_size];

        for b in 0..batch {
            let src_idx = (b * seq_len + cls_pos) * hidden_size;
            let dst_idx = b * hidden_size;
            output[dst_idx..dst_idx + hidden_size]
                .copy_from_slice(&hidden[src_idx..src_idx + hidden_size]);
        }

        Ok(output)
    }

    fn normalize(
        &self,
        input: &[f32],
        config: &NormalizeConfig,
    ) -> Result<Vec<f32>, String> {
        let batch = config.batch_size;
        let dim = config.dim;

        let mut output = vec![0.0f32; batch * dim];

        for b in 0..batch {
            let start = b * dim;
            let end = start + dim;

            // Compute L2 norm
            let mut norm_sq = 0.0f32;
            for &val in &input[start..end] {
                norm_sq += val * val;
            }
            let norm = (norm_sq + config.eps).sqrt();

            // Normalize
            for (i, &val) in input[start..end].iter().enumerate() {
                output[start + i] = val / norm;
            }
        }

        Ok(output)
    }

    fn dequantize(
        &self,
        quantized: &[u8],
        scales: &[half::f16],
        zeros: Option<&[u32]>,
        config: &DequantizeConfig,
    ) -> Result<Vec<f32>, String> {
        match config.format {
            QuantFormat::Q4_0 | QuantFormat::Q4_K => {
                crate::ops::quantized::q4_dequantize_cpu(quantized, scales, config.n, config.k)
            }
            QuantFormat::Q8_0 => {
                // Q8 uses i8, need to reinterpret
                let q_weight: &[i8] = bytemuck::cast_slice(quantized);
                // For Q8, we can use the matmul with identity to get dequantized
                let identity = vec![1.0f32; config.k];
                crate::ops::quantized::q8_matmul_cpu(&identity, q_weight, scales, 1, config.n, config.k)
            }
            QuantFormat::AWQ => {
                let qweight: &[u32] = bytemuck::cast_slice(quantized);
                let qzeros = zeros.ok_or("AWQ requires zero points")?;
                crate::ops::quantized::awq_dequantize_cpu(
                    qweight, qzeros, scales, config.n, config.k, config.group_size,
                )
            }
        }
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    // =========================================================================
    // Memory Tiering Primitives (ARCH-MEM-TIERING)
    // CPU backend does not support memory tiering
    // =========================================================================

    fn swap_out(&self, _block_id: u64, _cpu_buffer: &mut [u8]) -> Result<(), String> {
        Err("CpuBackend does not support memory tiering swap_out".to_string())
    }

    fn swap_in(&self, _cpu_buffer: &[u8], _block_id: u64) -> Result<(), String> {
        Err("CpuBackend does not support memory tiering swap_in".to_string())
    }

    // =========================================================================
    // L3 High-Level Inference API (ARCH-ADR-003)
    // =========================================================================

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
    ) -> Result<crate::kernel_types::LogitsTensor, String> {
        let seq_len = tokens.len();
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_q_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;

        // 1. Embedding lookup
        let embed_config = EmbeddingConfig {
            vocab_size: config.vocab_size,
            hidden_size,
            max_seq_len: config.max_seq_len,
            add_position_embedding: false,
            padding_idx: None,
        };
        let mut hidden = self.embedding(tokens, embed_weight, None, &embed_config)?;

        // 2. Process each layer
        for (layer_idx, layer) in layers.iter().enumerate() {
            // Get mutable references to KV cache for this layer
            let (k_cache_slice, v_cache_slice) = {
                let cache_offset = layer_idx * num_kv_heads * config.max_seq_len * head_dim;
                let cache_size = num_kv_heads * config.max_seq_len * head_dim;
                (
                    Some(&mut kv_cache.k_cache[cache_offset..cache_offset + cache_size]),
                    Some(&mut kv_cache.v_cache[cache_offset..cache_offset + cache_size]),
                )
            };

            // Attention block
            let attn_config = AttentionBlockConfig {
                batch_size: 1,
                seq_len,
                hidden_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                causal: true,
                use_rope: config.use_rope,
                rope_theta: config.rope_theta,
                position_offset: kv_cache.seq_len,
                scale: None,
                rms_norm_eps: config.rms_norm_eps,
                engram_hook: None,
                use_flash_attention: true,
                dropout_prob: 0.0,
            };
            let attn_output = self.attention_block(
                &hidden,
                layer.q_weight,
                layer.k_weight,
                layer.v_weight,
                layer.o_weight,
                layer.input_norm,
                cos_cache,
                sin_cache,
                k_cache_slice,
                v_cache_slice,
                &attn_config,
            )?;

            // Residual connection
            for (h, a) in hidden.iter_mut().zip(attn_output.iter()) {
                *h += a;
            }

            // FFN block
            let ffn_config = FFNBlockConfig {
                batch_size: 1,
                seq_len,
                hidden_size,
                intermediate_size,
                activation: config.activation,
                use_gate: true,
                use_bias: false,
                rms_norm_eps: config.rms_norm_eps,
            };
            let ffn_output = self.ffn_block(
                &hidden,
                layer.gate_weight,
                layer.up_weight,
                layer.down_weight,
                layer.post_attn_norm,
                &ffn_config,
            )?;

            // Residual connection
            for (h, f) in hidden.iter_mut().zip(ffn_output.iter()) {
                *h += f;
            }
        }

        // Update KV cache sequence length
        kv_cache.seq_len += seq_len;

        // 3. Final norm + LM head (only last token)
        let last_hidden = hidden[(seq_len - 1) * hidden_size..].to_vec();
        let lm_config = LMHeadConfig {
            vocab_size: config.vocab_size,
            hidden_size,
            batch_size: 1,
            seq_len: 1,
            tie_word_embeddings: false,
            rms_norm_eps: config.rms_norm_eps,
        };
        Ok(crate::kernel_types::LogitsTensor::Cpu(self.lm_head(&last_hidden, lm_head_weight, final_norm, &lm_config)?))
    }

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
    ) -> Result<crate::kernel_types::LogitsTensor, String> {
        let seq_len = tokens.len();
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_q_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;

        // 1. Embedding lookup
        let embed_config = EmbeddingConfig {
            vocab_size: config.vocab_size,
            hidden_size,
            max_seq_len: config.max_seq_len,
            add_position_embedding: false,
            padding_idx: None,
        };
        let mut hidden = self.embedding(tokens, embed_weight, None, &embed_config)?;

        // 2. Process each layer
        for (layer_idx, layer) in layers.iter().enumerate() {
            // Get mutable references to KV cache for this layer
            let (k_cache_slice, v_cache_slice) = {
                let cache_offset = layer_idx * num_kv_heads * config.max_seq_len * head_dim;
                let cache_size = num_kv_heads * config.max_seq_len * head_dim;
                (
                    Some(&mut kv_cache.k_cache[cache_offset..cache_offset + cache_size]),
                    Some(&mut kv_cache.v_cache[cache_offset..cache_offset + cache_size]),
                )
            };

            // Attention block
            let attn_config = AttentionBlockConfig {
                batch_size: 1,
                seq_len,
                hidden_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                causal: true,
                use_rope: config.use_rope,
                rope_theta: config.rope_theta,
                position_offset: kv_cache.seq_len,
                scale: None,
                rms_norm_eps: config.rms_norm_eps,
                engram_hook: None,
                use_flash_attention: true,
                dropout_prob: 0.0,
            };
            let attn_output = self.attention_block(
                &hidden,
                layer.q_weight,
                layer.k_weight,
                layer.v_weight,
                layer.o_weight,
                layer.input_norm,
                cos_cache,
                sin_cache,
                k_cache_slice,
                v_cache_slice,
                &attn_config,
            )?;

            // Residual connection
            for (h, a) in hidden.iter_mut().zip(attn_output.iter()) {
                *h += a;
            }

            // MoE FFN: route tokens to experts and combine outputs
            // Apply post_attn_norm first
            let mut normed = vec![0.0f32; hidden.len()];
            crate::ops::rms_norm::rms_norm_forward(&hidden, layer.post_attn_norm, &mut normed, 1, hidden_size, config.rms_norm_eps);

            // Route and compute MoE
            let moe_config = crate::MoERoutingConfig {
                num_experts: config.num_experts,
                num_experts_per_tok: config.num_experts_per_tok,
                hidden_size,
            };
            let routing = crate::moe_route(&normed, layer.router_weight, 1, seq_len, &moe_config);

            // Process each token through its assigned experts
            let mut moe_output = vec![0.0f32; seq_len * hidden_size];
            for token_idx in 0..seq_len {
                let token_input = &normed[token_idx * hidden_size..(token_idx + 1) * hidden_size];
                let mut token_output = vec![0.0f32; hidden_size];
                let route_base = token_idx * config.num_experts_per_tok;

                for k in 0..config.num_experts_per_tok {
                    let route_idx = route_base + k;
                    let expert_idx = routing.expert_indices[route_idx] as usize;
                    let weight = routing.expert_weights[route_idx];

                    if expert_idx < config.num_experts {
                        let expert = &layer.experts[expert_idx];
                        // Gate projection
                        let mut gate_out = vec![0.0f32; intermediate_size];
                        crate::linear_forward(token_input, expert.gate, None, &mut gate_out, 1, hidden_size, intermediate_size);
                        // Up projection
                        let mut up = vec![0.0f32; intermediate_size];
                        crate::linear_forward(token_input, expert.up, None, &mut up, 1, hidden_size, intermediate_size);
                        // SiLU + element-wise multiply
                        crate::silu_inplace(&mut gate_out);
                        for (g, u) in gate_out.iter_mut().zip(up.iter()) {
                            *g *= u;
                        }
                        // Down projection
                        let mut expert_out = vec![0.0f32; hidden_size];
                        crate::linear_forward(&gate_out, expert.down, None, &mut expert_out, 1, intermediate_size, hidden_size);
                        // Weighted sum
                        for (o, e) in token_output.iter_mut().zip(expert_out.iter()) {
                            *o += weight * e;
                        }
                    }
                }
                moe_output[token_idx * hidden_size..(token_idx + 1) * hidden_size]
                    .copy_from_slice(&token_output);
            }

            // Residual connection
            for (h, m) in hidden.iter_mut().zip(moe_output.iter()) {
                *h += m;
            }
        }

        // Update KV cache sequence length
        kv_cache.seq_len += seq_len;

        // 3. Final norm + LM head (only last token)
        let last_hidden = hidden[(seq_len - 1) * hidden_size..].to_vec();
        let lm_config = LMHeadConfig {
            vocab_size: config.vocab_size,
            hidden_size,
            batch_size: 1,
            seq_len: 1,
            tie_word_embeddings: false,
            rms_norm_eps: config.rms_norm_eps,
        };
        Ok(crate::kernel_types::LogitsTensor::Cpu(self.lm_head(&last_hidden, lm_head_weight, final_norm, &lm_config)?))
    }

    fn embedding_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: Option<&[f32]>,
        config: &EmbeddingForwardConfig,
    ) -> Result<Vec<f32>, String> {
        let batch_size = config.batch_size;
        let seq_len = config.seq_len;
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_q_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;

        // 1. Embedding lookup
        let embed_config = EmbeddingConfig {
            vocab_size: config.vocab_size,
            hidden_size,
            max_seq_len: seq_len,
            add_position_embedding: false,
            padding_idx: None,
        };
        let mut hidden = self.embedding(tokens, embed_weight, None, &embed_config)?;

        // 2. Process each layer (no KV cache for embedding models)
        for layer in layers.iter() {
            // Attention block (bidirectional, no KV cache)
            let attn_config = AttentionBlockConfig {
                batch_size,
                seq_len,
                hidden_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                position_offset: 0,
                causal: false,  // Bidirectional for embedding models
                use_rope: false,  // No RoPE for BERT-style
                rope_theta: 10000.0,
                scale: None,
                rms_norm_eps: config.rms_norm_eps,
                engram_hook: None,
                use_flash_attention: true,
                dropout_prob: 0.0,
            };
            let attn_output = self.attention_block(
                &hidden,
                layer.q_weight,
                layer.k_weight,
                layer.v_weight,
                layer.o_weight,
                layer.input_norm,
                &[],  // No RoPE for BERT-style
                &[],
                None, None,  // No KV cache
                &attn_config,
            )?;

            // Residual connection
            for (h, a) in hidden.iter_mut().zip(attn_output.iter()) {
                *h += a;
            }

            // FFN block
            let ffn_config = FFNBlockConfig {
                batch_size,
                seq_len,
                hidden_size,
                intermediate_size,
                activation: config.activation,
                use_gate: false,  // BERT-style typically doesn't use gate
                use_bias: true,
                rms_norm_eps: config.rms_norm_eps,
            };
            let ffn_output = self.ffn_block(
                &hidden,
                layer.gate_weight,
                layer.up_weight,
                layer.down_weight,
                layer.post_attn_norm,
                &ffn_config,
            )?;

            // Residual connection
            for (h, f) in hidden.iter_mut().zip(ffn_output.iter()) {
                *h += f;
            }
        }

        // 3. Optional final norm
        if let Some(norm_weight) = final_norm {
            let mut normalized = vec![0.0f32; hidden.len()];
            crate::ops::rms_norm::rms_norm_forward(&hidden, norm_weight, &mut normalized, batch_size * seq_len, hidden_size, config.rms_norm_eps);
            hidden = normalized;
        }

        // 4. Pooling
        let pooled = match config.pooling {
            PoolingType::Mean => {
                let pool_config = MeanPoolingConfig {
                    batch_size,
                    seq_len,
                    hidden_size,
                    use_attention_mask: false,
                };
                self.mean_pooling(&hidden, None, &pool_config)?
            }
            PoolingType::Cls => {
                let pool_config = ClsPoolingConfig {
                    batch_size,
                    hidden_size,
                    cls_position: 0,
                };
                self.cls_pooling(&hidden, &pool_config)?
            }
            PoolingType::Last => {
                // Get the last token of each sequence in the batch
                let mut result = Vec::with_capacity(batch_size * hidden_size);
                for b in 0..batch_size {
                    let last_pos = (b * seq_len + seq_len - 1) * hidden_size;
                    result.extend_from_slice(&hidden[last_pos..last_pos + hidden_size]);
                }
                result
            }
        };

        // 5. L2 normalize if requested
        if config.normalize {
            let norm_config = NormalizeConfig {
                batch_size,
                dim: hidden_size,
                eps: 1e-12,
            };
            self.normalize(&pooled, &norm_config)
        } else {
            Ok(pooled)
        }
    }

    fn rerank_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        score_weight: &[f32],
        config: &RerankForwardConfig,
    ) -> Result<Vec<f32>, String> {
        let batch_size = config.batch_size;
        let seq_len = config.seq_len;
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_q_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;

        // 1. Embedding lookup
        let embed_config = EmbeddingConfig {
            vocab_size: config.vocab_size,
            hidden_size,
            max_seq_len: seq_len,
            add_position_embedding: false,
            padding_idx: None,
        };
        let mut hidden = self.embedding(tokens, embed_weight, None, &embed_config)?;

        // 2. Process each layer (bidirectional, no KV cache)
        for layer in layers.iter() {
            let attn_config = AttentionBlockConfig {
                batch_size,
                seq_len,
                hidden_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                position_offset: 0,
                causal: false,  // Bidirectional for reranker
                use_rope: false,  // No RoPE for BERT-style
                rope_theta: 10000.0,
                scale: None,
                rms_norm_eps: config.rms_norm_eps,
                engram_hook: None,
                use_flash_attention: true,
                dropout_prob: 0.0,
            };
            let attn_output = self.attention_block(
                &hidden,
                layer.q_weight,
                layer.k_weight,
                layer.v_weight,
                layer.o_weight,
                layer.input_norm,
                &[],  // No RoPE for BERT-style
                &[],
                None, None,
                &attn_config,
            )?;

            for (h, a) in hidden.iter_mut().zip(attn_output.iter()) {
                *h += a;
            }

            let ffn_config = FFNBlockConfig {
                batch_size,
                seq_len,
                hidden_size,
                intermediate_size,
                activation: config.activation,
                use_gate: false,  // BERT-style typically doesn't use gate
                use_bias: true,
                rms_norm_eps: config.rms_norm_eps,
            };
            let ffn_output = self.ffn_block(
                &hidden,
                layer.gate_weight,
                layer.up_weight,
                layer.down_weight,
                layer.post_attn_norm,
                &ffn_config,
            )?;

            for (h, f) in hidden.iter_mut().zip(ffn_output.iter()) {
                *h += f;
            }
        }

        // 3. Final norm
        let mut normalized = vec![0.0f32; hidden.len()];
        crate::ops::rms_norm::rms_norm_forward(&hidden, final_norm, &mut normalized, batch_size * seq_len, hidden_size, config.rms_norm_eps);
        hidden = normalized;

        // 4. CLS pooling (first token of each sequence)
        let cls_pooling_config = ClsPoolingConfig {
            batch_size,
            hidden_size,
            cls_position: 0,
        };
        let cls_output = self.cls_pooling(&hidden, &cls_pooling_config)?;

        // 5. Score head (linear projection to scalar for each batch)
        let mut scores = vec![0.0f32; batch_size];
        crate::linear_forward(&cls_output, score_weight, None, &mut scores, batch_size, hidden_size, 1);

        Ok(scores)
    }

    // =========================================================================
    // GPU-pure methods (ARCH-GPU-001)
    // For CPU backend, these use the Cpu variant of GpuBuffer.
    // =========================================================================

    fn upload_embedding_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: Option<&[f32]>,
        _config: &EmbeddingForwardConfig,
    ) -> Result<EmbeddingModelWeightsGpu, String> {
        // Helper to create a CPU-backed GpuTensor from f32 slice
        fn f32_to_gpu_tensor(data: &[f32], shape: Vec<usize>) -> GpuTensor {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            GpuTensor::new(
                GpuBuffer::Cpu(Arc::new(bytes)),
                shape,
                TensorDtype::F32,
                BackendType::Cpu,
            )
        }

        // Upload embedding weights
        let embedding = f32_to_gpu_tensor(embed_weight, vec![embed_weight.len()]);

        // Upload layer weights
        let gpu_layers: Vec<TransformerLayerWeightsGpu> = layers.iter().map(|layer| {
            TransformerLayerWeightsGpu {
                input_norm: f32_to_gpu_tensor(layer.input_norm, vec![layer.input_norm.len()]),
                q_weight: f32_to_gpu_tensor(layer.q_weight, vec![layer.q_weight.len()]),
                k_weight: f32_to_gpu_tensor(layer.k_weight, vec![layer.k_weight.len()]),
                v_weight: f32_to_gpu_tensor(layer.v_weight, vec![layer.v_weight.len()]),
                o_weight: f32_to_gpu_tensor(layer.o_weight, vec![layer.o_weight.len()]),
                post_attn_norm: f32_to_gpu_tensor(layer.post_attn_norm, vec![layer.post_attn_norm.len()]),
                gate_weight: layer.gate_weight.map(|g| f32_to_gpu_tensor(g, vec![g.len()])),
                up_weight: f32_to_gpu_tensor(layer.up_weight, vec![layer.up_weight.len()]),
                down_weight: f32_to_gpu_tensor(layer.down_weight, vec![layer.down_weight.len()]),
                cos_cache: None,
                sin_cache: None,
            }
        }).collect();

        // Upload final norm
        let final_norm_gpu = match final_norm {
            Some(norm) => f32_to_gpu_tensor(norm, vec![norm.len()]),
            None => f32_to_gpu_tensor(&[], vec![0]),
        };

        Ok(EmbeddingModelWeightsGpu {
            embedding,
            layers: gpu_layers,
            final_norm: final_norm_gpu,
        })
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &EmbeddingModelWeightsGpu,
        config: &EmbeddingForwardConfig,
    ) -> Result<Vec<f32>, String> {
        // Helper to extract f32 slice from CPU-backed GpuTensor
        fn gpu_tensor_to_f32(tensor: &GpuTensor) -> Result<&[f32], String> {
            match &tensor.buffer {
                GpuBuffer::Cpu(bytes) => {
                    let ptr = bytes.as_ptr() as *const f32;
                    let len = bytes.len() / 4;
                    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
                }
                #[allow(unreachable_patterns)]
                _ => Err("Expected CPU-backed GpuTensor".to_string()),
            }
        }

        // Extract weights from GpuTensor
        let embed_weight = gpu_tensor_to_f32(&weights.embedding)?;
        let final_norm = gpu_tensor_to_f32(&weights.final_norm)?;
        let final_norm_opt = if final_norm.is_empty() { None } else { Some(final_norm) };

        // Extract layer weights
        let layer_weights: Result<Vec<TransformerLayerWeights<'_>>, String> = weights.layers.iter()
            .map(|layer| {
                Ok(TransformerLayerWeights {
                    input_norm: gpu_tensor_to_f32(&layer.input_norm)?,
                    q_weight: gpu_tensor_to_f32(&layer.q_weight)?,
                    k_weight: gpu_tensor_to_f32(&layer.k_weight)?,
                    v_weight: gpu_tensor_to_f32(&layer.v_weight)?,
                    o_weight: gpu_tensor_to_f32(&layer.o_weight)?,
                    post_attn_norm: gpu_tensor_to_f32(&layer.post_attn_norm)?,
                    gate_weight: match &layer.gate_weight {
                        Some(g) => Some(gpu_tensor_to_f32(g)?),
                        None => None,
                    },
                    up_weight: gpu_tensor_to_f32(&layer.up_weight)?,
                    down_weight: gpu_tensor_to_f32(&layer.down_weight)?,
                })
            })
            .collect();
        let layers = layer_weights?;

        // Call existing embedding_forward
        self.embedding_forward(tokens, embed_weight, &layers, final_norm_opt, config)
    }

    // =========================================================================
    // GPU-Native Kernel Methods (ARCH-ADR-010)
    // CPU backend implements these using CPU-backed GpuBuffer::Cpu
    // =========================================================================

    fn embedding_lookup_gpu(
        &self,
        tokens: &GpuTensor,
        embed_weight: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        // Extract data from CPU-backed tensors
        let token_bytes = match &tokens.buffer {
            GpuBuffer::Cpu(b) => b,
            _ => return Err("CpuBackend: expected CPU-backed tensor for tokens".into()),
        };
        let embed_bytes = match &embed_weight.buffer {
            GpuBuffer::Cpu(b) => b,
            _ => return Err("CpuBackend: expected CPU-backed tensor for embed_weight".into()),
        };

        let token_ids: &[u32] = bytemuck::cast_slice(token_bytes);
        let embed_data: &[f32] = bytemuck::cast_slice(embed_bytes);

        // embed_weight shape: [vocab_size, hidden_dim]
        let hidden_dim = embed_weight.shape.last().copied().unwrap_or(0);

        // Perform gather: output[i] = embed_data[token_ids[i] * hidden_dim..]
        let mut result = Vec::with_capacity(token_ids.len() * hidden_dim);
        for &tid in token_ids {
            let start = (tid as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= embed_data.len() {
                result.extend_from_slice(&embed_data[start..end]);
            } else {
                return Err(format!("Token ID {} out of bounds", tid));
            }
        }

        // Write result to output
        output.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&result).to_vec().into());
        Ok(())
    }

    fn transformer_layer_gpu(
        &self,
        hidden: &mut GpuTensor,
        layer_weights: &TransformerLayerWeightsGpu,
        kv_cache: Option<&mut KVCacheGpu>,
        config: &TransformerLayerConfigGpu,
    ) -> Result<(), String> {
        // Helper to extract f32 slice from CPU-backed GpuTensor
        fn gpu_tensor_to_f32(tensor: &GpuTensor) -> Result<&[f32], String> {
            match &tensor.buffer {
                GpuBuffer::Cpu(bytes) => Ok(bytemuck::cast_slice(bytes)),
                _ => Err("CpuBackend: expected CPU-backed tensor".into()),
            }
        }

        let hidden_data: Vec<f32> = match &hidden.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
            _ => return Err("CpuBackend: expected CPU-backed tensor for hidden".into()),
        };

        let cos_cache = layer_weights
            .cos_cache
            .as_ref()
            .map(|c| gpu_tensor_to_f32(c))
            .transpose()?;
        let sin_cache = layer_weights
            .sin_cache
            .as_ref()
            .map(|s| gpu_tensor_to_f32(s))
            .transpose()?;

        // Build config for existing attention_block
        let attn_config = AttentionBlockConfig {
            batch_size: config.batch_size,
            seq_len: config.seq_len,
            hidden_size: config.hidden_size,
            num_q_heads: config.num_q_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            position_offset: config.position,
            rms_norm_eps: config.rms_norm_eps,
            use_rope: true,
            causal: true,
            ..Default::default()
        };

        // Extract KV cache if provided
        let (mut kv_k, mut kv_v) = if let Some(kv) = kv_cache.as_ref() {
            let k_data: Vec<f32> = match &kv.k_cache.buffer {
                GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
                _ => return Err("CpuBackend: expected CPU-backed KV cache".into()),
            };
            let v_data: Vec<f32> = match &kv.v_cache.buffer {
                GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
                _ => return Err("CpuBackend: expected CPU-backed KV cache".into()),
            };
            (k_data, v_data)
        } else {
            (vec![], vec![])
        };

        let kv_k_opt = if kv_k.is_empty() { None } else { Some(kv_k.as_mut_slice()) };
        let kv_v_opt = if kv_v.is_empty() { None } else { Some(kv_v.as_mut_slice()) };

        // Run attention block
        let attn_out = self.attention_block(
            &hidden_data,
            gpu_tensor_to_f32(&layer_weights.q_weight)?,
            gpu_tensor_to_f32(&layer_weights.k_weight)?,
            gpu_tensor_to_f32(&layer_weights.v_weight)?,
            gpu_tensor_to_f32(&layer_weights.o_weight)?,
            gpu_tensor_to_f32(&layer_weights.input_norm)?,
            cos_cache.unwrap_or(&[]),
            sin_cache.unwrap_or(&[]),
            kv_k_opt,
            kv_v_opt,
            &attn_config,
        )?;

        // Build FFN config
        let ffn_config = FFNBlockConfig {
            batch_size: config.batch_size,
            seq_len: config.seq_len,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            activation: if config.use_silu { Activation::SiLU } else { Activation::GELU },
            use_gate: layer_weights.gate_weight.is_some(),
            use_bias: false,
            rms_norm_eps: config.rms_norm_eps,
        };

        // Run FFN block
        let gate_weight = layer_weights.gate_weight.as_ref().map(|g| gpu_tensor_to_f32(g)).transpose()?;
        let ffn_out = self.ffn_block(
            &attn_out,
            gate_weight,
            gpu_tensor_to_f32(&layer_weights.up_weight)?,
            gpu_tensor_to_f32(&layer_weights.down_weight)?,
            gpu_tensor_to_f32(&layer_weights.post_attn_norm)?,
            &ffn_config,
        )?;

        // Update hidden
        hidden.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&ffn_out).to_vec().into());

        // Update KV cache if provided
        if let Some(kv) = kv_cache {
            if !kv_k.is_empty() {
                kv.k_cache.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&kv_k).to_vec().into());
            }
            if !kv_v.is_empty() {
                kv.v_cache.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&kv_v).to_vec().into());
            }
            kv.seq_len = config.position + config.seq_len;
        }

        Ok(())
    }

    fn rms_norm_gpu(
        &self,
        hidden: &mut GpuTensor,
        weight: &GpuTensor,
        eps: f32,
    ) -> Result<(), String> {
        let hidden_data: Vec<f32> = match &hidden.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };
        let weight_data: &[f32] = match &weight.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };

        let hidden_dim = weight_data.len();
        let num_tokens = hidden_data.len() / hidden_dim;

        let mut result = vec![0.0f32; hidden_data.len()];
        for i in 0..num_tokens {
            let start = i * hidden_dim;
            let slice = &hidden_data[start..start + hidden_dim];

            // Compute RMS
            let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

            // Normalize and scale
            for j in 0..hidden_dim {
                result[start + j] = (slice[j] / rms) * weight_data[j];
            }
        }

        hidden.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&result).to_vec().into());
        Ok(())
    }

    fn mean_pooling_gpu(
        &self,
        hidden: &GpuTensor,
        attention_mask: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let hidden_data: &[f32] = match &hidden.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };
        let mask_data: &[f32] = match &attention_mask.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };

        // hidden: [batch, seq_len, hidden_dim]
        // mask: [batch, seq_len]
        let batch_size = hidden.shape.first().copied().unwrap_or(1);
        let seq_len = hidden.shape.get(1).copied().unwrap_or(1);
        let hidden_dim = hidden.shape.last().copied().unwrap_or(0);

        let mut result = vec![0.0f32; batch_size * hidden_dim];

        for b in 0..batch_size {
            let mut sum = vec![0.0f32; hidden_dim];
            let mut count = 0.0f32;

            for s in 0..seq_len {
                let mask_val = mask_data[b * seq_len + s];
                if mask_val > 0.0 {
                    let offset = b * seq_len * hidden_dim + s * hidden_dim;
                    for d in 0..hidden_dim {
                        sum[d] += hidden_data[offset + d] * mask_val;
                    }
                    count += mask_val;
                }
            }

            if count > 0.0 {
                for d in 0..hidden_dim {
                    result[b * hidden_dim + d] = sum[d] / count;
                }
            }
        }

        output.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&result).to_vec().into());
        Ok(())
    }

    fn normalize_gpu(
        &self,
        embeddings: &mut GpuTensor,
    ) -> Result<(), String> {
        let data: Vec<f32> = match &embeddings.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };

        let hidden_dim = embeddings.shape.last().copied().unwrap_or(0);
        let batch_size = data.len() / hidden_dim;

        let mut result = vec![0.0f32; data.len()];

        for b in 0..batch_size {
            let start = b * hidden_dim;
            let slice = &data[start..start + hidden_dim];

            // Compute L2 norm
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Normalize
            if norm > 1e-12 {
                for d in 0..hidden_dim {
                    result[start + d] = slice[d] / norm;
                }
            }
        }

        embeddings.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&result).to_vec().into());
        Ok(())
    }

    fn cls_pooling_gpu(
        &self,
        hidden: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let hidden_data: &[f32] = match &hidden.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };

        // hidden: [batch, seq_len, hidden_dim]
        let batch_size = hidden.shape.first().copied().unwrap_or(1);
        let seq_len = hidden.shape.get(1).copied().unwrap_or(1);
        let hidden_dim = hidden.shape.last().copied().unwrap_or(0);

        // Extract first token for each batch
        let mut result = Vec::with_capacity(batch_size * hidden_dim);
        for b in 0..batch_size {
            let offset = b * seq_len * hidden_dim;
            result.extend_from_slice(&hidden_data[offset..offset + hidden_dim]);
        }

        output.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&result).to_vec().into());
        Ok(())
    }

    fn classifier_gpu(
        &self,
        hidden: &GpuTensor,
        weight: &GpuTensor,
        bias: Option<&GpuTensor>,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let hidden_data: &[f32] = match &hidden.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };
        let weight_data: &[f32] = match &weight.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => return Err("CpuBackend: expected CPU-backed tensor".into()),
        };
        let bias_data: Option<&[f32]> = bias.map(|b| match &b.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes),
            _ => &[] as &[f32],
        });

        // hidden: [batch, hidden_dim]
        // weight: [num_classes, hidden_dim]
        let batch_size = hidden.shape.first().copied().unwrap_or(1);
        let hidden_dim = hidden.shape.last().copied().unwrap_or(0);
        let num_classes = weight.shape.first().copied().unwrap_or(0);

        let mut result = vec![0.0f32; batch_size * num_classes];

        // Linear: output = hidden @ weight.T + bias
        for b in 0..batch_size {
            for c in 0..num_classes {
                let mut sum = 0.0f32;
                for d in 0..hidden_dim {
                    sum += hidden_data[b * hidden_dim + d] * weight_data[c * hidden_dim + d];
                }
                if let Some(bd) = bias_data {
                    if c < bd.len() {
                        sum += bd[c];
                    }
                }
                result[b * num_classes + c] = sum;
            }
        }

        output.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&result).to_vec().into());
        Ok(())
    }

    fn lm_head_gpu(
        &self,
        hidden: &GpuTensor,
        weight: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        // LM head is just a linear layer without bias
        self.classifier_gpu(hidden, weight, None, output)
    }

    // =========================================================================
    // GPU-Native High-Level Forward Methods (ARCH-ADR-010)
    // =========================================================================

    fn upload_reranker_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        classifier_weight: &[f32],
        classifier_bias: Option<&[f32]>,
        config: &RerankForwardConfig,
    ) -> Result<RerankerModelWeightsGpu, String> {
        // Helper to create CPU-backed GpuTensor
        fn f32_to_gpu_tensor(data: &[f32], shape: Vec<usize>) -> GpuTensor {
            let size_in_bytes = data.len() * std::mem::size_of::<f32>();
            GpuTensor {
                buffer: GpuBuffer::Cpu(bytemuck::cast_slice(data).to_vec().into()),
                shape,
                dtype: TensorDtype::F32,
                size_in_bytes,
                backend: BackendType::Cpu,
            }
        }

        let hidden_dim = config.hidden_size;
        let vocab_size = embed_weight.len() / hidden_dim;

        let embedding = f32_to_gpu_tensor(embed_weight, vec![vocab_size, hidden_dim]);
        let final_norm_gpu = f32_to_gpu_tensor(final_norm, vec![hidden_dim]);

        let num_classes = classifier_weight.len() / hidden_dim;
        let classifier_weight_gpu = f32_to_gpu_tensor(classifier_weight, vec![num_classes, hidden_dim]);
        let classifier_bias_gpu = classifier_bias.map(|b| f32_to_gpu_tensor(b, vec![num_classes]));

        // Convert layers
        let gpu_layers: Vec<TransformerLayerWeightsGpu> = layers.iter().map(|layer| {
            TransformerLayerWeightsGpu {
                input_norm: f32_to_gpu_tensor(layer.input_norm, vec![hidden_dim]),
                q_weight: f32_to_gpu_tensor(layer.q_weight, vec![layer.q_weight.len()]),
                k_weight: f32_to_gpu_tensor(layer.k_weight, vec![layer.k_weight.len()]),
                v_weight: f32_to_gpu_tensor(layer.v_weight, vec![layer.v_weight.len()]),
                o_weight: f32_to_gpu_tensor(layer.o_weight, vec![layer.o_weight.len()]),
                post_attn_norm: f32_to_gpu_tensor(layer.post_attn_norm, vec![hidden_dim]),
                gate_weight: layer.gate_weight.map(|g| f32_to_gpu_tensor(g, vec![g.len()])),
                up_weight: f32_to_gpu_tensor(layer.up_weight, vec![layer.up_weight.len()]),
                down_weight: f32_to_gpu_tensor(layer.down_weight, vec![layer.down_weight.len()]),
                cos_cache: None,
                sin_cache: None,
            }
        }).collect();

        Ok(RerankerModelWeightsGpu {
            embedding,
            layers: gpu_layers,
            final_norm: final_norm_gpu,
            classifier_weight: classifier_weight_gpu,
            classifier_bias: classifier_bias_gpu,
        })
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &RerankerModelWeightsGpu,
        config: &RerankForwardConfig,
    ) -> Result<Vec<f32>, String> {
        // Helper to extract f32 slice from CPU-backed GpuTensor
        fn gpu_tensor_to_f32(tensor: &GpuTensor) -> Result<&[f32], String> {
            match &tensor.buffer {
                GpuBuffer::Cpu(bytes) => Ok(bytemuck::cast_slice(bytes)),
                _ => Err("CpuBackend: expected CPU-backed tensor".into()),
            }
        }

        // Convert GPU weights back to slices for existing rerank_forward
        let embed_weight = gpu_tensor_to_f32(&weights.embedding)?;
        let final_norm = gpu_tensor_to_f32(&weights.final_norm)?;
        let classifier_weight = gpu_tensor_to_f32(&weights.classifier_weight)?;
        let classifier_bias = weights.classifier_bias.as_ref().map(|b| gpu_tensor_to_f32(b)).transpose()?;

        let layer_weights: Result<Vec<TransformerLayerWeights<'_>>, String> = weights.layers.iter()
            .map(|layer| {
                Ok(TransformerLayerWeights {
                    input_norm: gpu_tensor_to_f32(&layer.input_norm)?,
                    q_weight: gpu_tensor_to_f32(&layer.q_weight)?,
                    k_weight: gpu_tensor_to_f32(&layer.k_weight)?,
                    v_weight: gpu_tensor_to_f32(&layer.v_weight)?,
                    o_weight: gpu_tensor_to_f32(&layer.o_weight)?,
                    post_attn_norm: gpu_tensor_to_f32(&layer.post_attn_norm)?,
                    gate_weight: layer.gate_weight.as_ref().map(|g| gpu_tensor_to_f32(g)).transpose()?,
                    up_weight: gpu_tensor_to_f32(&layer.up_weight)?,
                    down_weight: gpu_tensor_to_f32(&layer.down_weight)?,
                })
            })
            .collect();
        let layers = layer_weights?;

        // Call existing rerank_forward (classifier_bias handled separately if needed)
        self.rerank_forward(
            tokens,
            embed_weight,
            &layers,
            final_norm,
            classifier_weight,
            config,
        )
    }

    fn upload_generator_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: &[f32],
        lm_head: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        config: &GeneratorForwardConfig,
    ) -> Result<GeneratorModelWeightsGpu, String> {
        // Helper to create CPU-backed GpuTensor
        fn f32_to_gpu_tensor(data: &[f32], shape: Vec<usize>) -> GpuTensor {
            let size_in_bytes = data.len() * std::mem::size_of::<f32>();
            GpuTensor {
                buffer: GpuBuffer::Cpu(bytemuck::cast_slice(data).to_vec().into()),
                shape,
                dtype: TensorDtype::F32,
                size_in_bytes,
                backend: BackendType::Cpu,
            }
        }

        let hidden_dim = config.hidden_size;
        let vocab_size = embed_weight.len() / hidden_dim;

        let embedding = f32_to_gpu_tensor(embed_weight, vec![vocab_size, hidden_dim]);
        let final_norm_gpu = f32_to_gpu_tensor(final_norm, vec![hidden_dim]);
        let lm_head_gpu = f32_to_gpu_tensor(lm_head, vec![vocab_size, hidden_dim]);
        let cos_cache_gpu = f32_to_gpu_tensor(cos_cache, vec![cos_cache.len()]);
        let sin_cache_gpu = f32_to_gpu_tensor(sin_cache, vec![sin_cache.len()]);

        // Convert layers
        let gpu_layers: Vec<TransformerLayerWeightsGpu> = layers.iter().map(|layer| {
            TransformerLayerWeightsGpu {
                input_norm: f32_to_gpu_tensor(layer.input_norm, vec![hidden_dim]),
                q_weight: f32_to_gpu_tensor(layer.q_weight, vec![layer.q_weight.len()]),
                k_weight: f32_to_gpu_tensor(layer.k_weight, vec![layer.k_weight.len()]),
                v_weight: f32_to_gpu_tensor(layer.v_weight, vec![layer.v_weight.len()]),
                o_weight: f32_to_gpu_tensor(layer.o_weight, vec![layer.o_weight.len()]),
                post_attn_norm: f32_to_gpu_tensor(layer.post_attn_norm, vec![hidden_dim]),
                gate_weight: layer.gate_weight.map(|g| f32_to_gpu_tensor(g, vec![g.len()])),
                up_weight: f32_to_gpu_tensor(layer.up_weight, vec![layer.up_weight.len()]),
                down_weight: f32_to_gpu_tensor(layer.down_weight, vec![layer.down_weight.len()]),
                cos_cache: Some(cos_cache_gpu.clone()),
                sin_cache: Some(sin_cache_gpu.clone()),
            }
        }).collect();

        Ok(GeneratorModelWeightsGpu {
            embedding,
            layers: gpu_layers,
            final_norm: final_norm_gpu,
            lm_head: lm_head_gpu,
            cos_cache: cos_cache_gpu,
            sin_cache: sin_cache_gpu,
        })
    }

    fn alloc_kv_cache_gpu(
        &self,
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        max_len: usize,
        head_dim: usize,
    ) -> Result<KVCacheGpu, String> {
        // Allocate CPU-backed KV cache
        let cache_size = num_layers * batch_size * num_kv_heads * max_len * head_dim;
        let size_in_bytes = cache_size * std::mem::size_of::<f32>();
        let k_cache = GpuTensor {
            buffer: GpuBuffer::Cpu(vec![0u8; size_in_bytes].into()),
            shape: vec![num_layers, batch_size, num_kv_heads, max_len, head_dim],
            dtype: TensorDtype::F32,
            size_in_bytes,
            backend: BackendType::Cpu,
        };
        let v_cache = GpuTensor {
            buffer: GpuBuffer::Cpu(vec![0u8; size_in_bytes].into()),
            shape: vec![num_layers, batch_size, num_kv_heads, max_len, head_dim],
            dtype: TensorDtype::F32,
            size_in_bytes,
            backend: BackendType::Cpu,
        };

        Ok(KVCacheGpu {
            k_cache,
            v_cache,
            seq_len: 0,
            max_len,
            num_layers,
            num_kv_heads,
            head_dim,
        })
    }

    fn generator_forward_gpu_pure(
        &self,
        tokens: &[u32],
        weights: &GeneratorModelWeightsGpu,
        kv_cache: &mut KVCacheGpu,
        config: &GeneratorForwardConfig,
    ) -> Result<crate::kernel_types::LogitsTensor, String> {
        // Helper to extract f32 slice from CPU-backed GpuTensor
        fn gpu_tensor_to_f32(tensor: &GpuTensor) -> Result<&[f32], String> {
            match &tensor.buffer {
                GpuBuffer::Cpu(bytes) => Ok(bytemuck::cast_slice(bytes)),
                _ => Err("CpuBackend: expected CPU-backed tensor".into()),
            }
        }

        // Convert GPU weights back to slices for existing generator_forward
        let embed_weight = gpu_tensor_to_f32(&weights.embedding)?;
        let final_norm = gpu_tensor_to_f32(&weights.final_norm)?;
        let lm_head = gpu_tensor_to_f32(&weights.lm_head)?;
        let cos_cache = gpu_tensor_to_f32(&weights.cos_cache)?;
        let sin_cache = gpu_tensor_to_f32(&weights.sin_cache)?;

        let layer_weights: Result<Vec<TransformerLayerWeights<'_>>, String> = weights.layers.iter()
            .map(|layer| {
                Ok(TransformerLayerWeights {
                    input_norm: gpu_tensor_to_f32(&layer.input_norm)?,
                    q_weight: gpu_tensor_to_f32(&layer.q_weight)?,
                    k_weight: gpu_tensor_to_f32(&layer.k_weight)?,
                    v_weight: gpu_tensor_to_f32(&layer.v_weight)?,
                    o_weight: gpu_tensor_to_f32(&layer.o_weight)?,
                    post_attn_norm: gpu_tensor_to_f32(&layer.post_attn_norm)?,
                    gate_weight: layer.gate_weight.as_ref().map(|g| gpu_tensor_to_f32(g)).transpose()?,
                    up_weight: gpu_tensor_to_f32(&layer.up_weight)?,
                    down_weight: gpu_tensor_to_f32(&layer.down_weight)?,
                })
            })
            .collect();
        let layers = layer_weights?;

        // Extract KV cache slices
        let mut k_cache_data: Vec<f32> = match &kv_cache.k_cache.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
            _ => return Err("CpuBackend: expected CPU-backed KV cache".into()),
        };
        let mut v_cache_data: Vec<f32> = match &kv_cache.v_cache.buffer {
            GpuBuffer::Cpu(bytes) => bytemuck::cast_slice(bytes).to_vec(),
            _ => return Err("CpuBackend: expected CPU-backed KV cache".into()),
        };

        let mut kv_state = KVCacheState {
            k_cache: k_cache_data.as_mut_slice(),
            v_cache: v_cache_data.as_mut_slice(),
            seq_len: kv_cache.seq_len,
            max_len: kv_cache.max_len,
        };

        // Call existing generator_forward
        let result = self.generator_forward(
            tokens,
            embed_weight,
            &layers,
            final_norm,
            lm_head,
            cos_cache,
            sin_cache,
            &mut kv_state,
            config,
        )?;

        // Update KV cache
        kv_cache.seq_len = kv_state.seq_len;
        kv_cache.k_cache.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&k_cache_data).to_vec().into());
        kv_cache.v_cache.buffer = GpuBuffer::Cpu(bytemuck::cast_slice(&v_cache_data).to_vec().into());

        Ok(result)
    }
}
