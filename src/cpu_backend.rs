use crate::backend_match::{
    match_float1, match_float1_mut, match_float1_out, match_float2_out, match_float2_out2,
    match_float3_out,
};
use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
use crate::kernel_types::{
    FlashAttentionConfig, MatmulConfig, PagedAttentionConfig, SoftmaxConfig,
};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};
use crate::ops::rope::RoPEConfig;
use crate::ops::sampling::{SamplingConfig, TopKResult};
use crate::runtime_detection::BackendType;

pub struct CpuBackend {
}

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
    fn flash_attention(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        v: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: FlashAttentionConfig,
    ) -> Result<(), String> {
        match_float3_out(
            "flash_attention",
            q,
            k,
            v,
            output,
            |q, k, v, out| crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone()),
            |q, k, v, out| crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone()),
            |q, k, v, out| crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone()),
        )
    }

    fn paged_attention(
        &self,
        q: TensorSlice<'_>,
        k_cache: TensorSlice<'_>,
        v_cache: TensorSlice<'_>,
        page_table: &[u32],
        seq_lens: &[u32],
        output: TensorSliceMut<'_>,
        config: PagedAttentionConfig,
    ) -> Result<(), String> {
        match_float3_out(
            "paged_attention",
            q,
            k_cache,
            v_cache,
            output,
            |q, k, v, out| {
                crate::ops::attention::cpu_paged_attention(
                    q,
                    k,
                    v,
                    page_table,
                    seq_lens,
                    out,
                    config.clone(),
                );
            },
            |q, k, v, out| {
                crate::ops::attention::cpu_paged_attention(
                    q,
                    k,
                    v,
                    page_table,
                    seq_lens,
                    out,
                    config.clone(),
                );
            },
            |q, k, v, out| {
                crate::ops::attention::cpu_paged_attention(
                    q,
                    k,
                    v,
                    page_table,
                    seq_lens,
                    out,
                    config.clone(),
                );
            },
        )
    }

    fn softmax(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: SoftmaxConfig,
    ) -> Result<(), String> {
        match_float1_out(
            "softmax",
            input,
            output,
            |input, out| crate::ops::softmax::softmax(input, out, config.clone()),
            |input, out| crate::ops::softmax::softmax(input, out, config.clone()),
            |input, out| crate::ops::softmax::softmax(input, out, config.clone()),
        )
    }

    fn matmul(
        &self,
        a: TensorSlice<'_>,
        b: TensorSlice<'_>,
        c: TensorSliceMut<'_>,
        config: MatmulConfig,
    ) -> Result<(), String> {
        match_float2_out(
            "matmul",
            a,
            b,
            c,
            |a, b, c| crate::ops::matmul::cpu_matmul(a, b, c, config.clone()),
            |a, b, c| crate::ops::matmul::cpu_matmul(a, b, c, config.clone()),
            |a, b, c| crate::ops::matmul::cpu_matmul(a, b, c, config.clone()),
        )
    }

    fn rope_precompute(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
        config: RoPEConfig,
    ) -> Result<(), String> {
        crate::ops::rope::rope_precompute(cos_out, sin_out, &config);
        Ok(())
    }

    fn rope_apply(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        cos_cache: &[f32],
        sin_cache: &[f32],
        q_out: TensorSliceMut<'_>,
        k_out: TensorSliceMut<'_>,
        batch_size: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), String> {
        match_float2_out2(
            "rope_apply",
            q,
            k,
            q_out,
            k_out,
            |q, k, q_out, k_out| {
                crate::ops::rope::rope_apply(
                    q,
                    k,
                    cos_cache,
                    sin_cache,
                    q_out,
                    k_out,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    position_offset,
                );
            },
            |q, k, q_out, k_out| {
                crate::ops::rope::rope_apply(
                    q,
                    k,
                    cos_cache,
                    sin_cache,
                    q_out,
                    k_out,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    position_offset,
                );
            },
            |q, k, q_out, k_out| {
                crate::ops::rope::rope_apply(
                    q,
                    k,
                    cos_cache,
                    sin_cache,
                    q_out,
                    k_out,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    position_offset,
                );
            },
        )
    }

    fn rope_apply_inplace(
        &self,
        x: TensorSliceMut<'_>,
        cos_cache: &[f32],
        sin_cache: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), String> {
        match_float1_mut(
            x,
            |x| {
                crate::ops::rope::rope_apply_inplace(
                    x,
                    cos_cache,
                    sin_cache,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    position_offset,
                );
            },
            |x| {
                crate::ops::rope::rope_apply_inplace(
                    x,
                    cos_cache,
                    sin_cache,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    position_offset,
                );
            },
            |x| {
                crate::ops::rope::rope_apply_inplace(
                    x,
                    cos_cache,
                    sin_cache,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    position_offset,
                );
            },
        )
    }

    fn topk(
        &self,
        logits: TensorSlice<'_>,
        k: usize,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<TopKResult, String> {
        match_float1(
            logits,
            |logits| crate::ops::sampling::topk(logits, k, batch_size, vocab_size),
            |logits| crate::ops::sampling::topk(logits, k, batch_size, vocab_size),
            |logits| crate::ops::sampling::topk(logits, k, batch_size, vocab_size),
        )
    }

    fn apply_temperature(
        &self,
        logits: TensorSliceMut<'_>,
        temperature: f32,
    ) -> Result<(), String> {
        match_float1_mut(
            logits,
            |logits| crate::ops::sampling::apply_temperature(logits, temperature),
            |logits| crate::ops::sampling::apply_temperature(logits, temperature),
            |logits| crate::ops::sampling::apply_temperature(logits, temperature),
        )
    }

    fn sample_tokens(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        match_float1(
            logits,
            |logits| crate::ops::sampling::sample_tokens(logits, batch_size, vocab_size, config),
            |logits| crate::ops::sampling::sample_tokens(logits, batch_size, vocab_size, config),
            |logits| crate::ops::sampling::sample_tokens(logits, batch_size, vocab_size, config),
        )
    }

    fn argmax(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, String> {
        match_float1(
            logits,
            |logits| crate::ops::sampling::argmax(logits, batch_size, vocab_size),
            |logits| crate::ops::sampling::argmax(logits, batch_size, vocab_size),
            |logits| crate::ops::sampling::argmax(logits, batch_size, vocab_size),
        )
    }

    fn moe_route(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<MoERoutingResult, String> {
        match_float1(
            hidden_states,
            |hidden_states| {
                crate::ops::moe_routing::moe_route(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                crate::ops::moe_routing::moe_route(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                crate::ops::moe_routing::moe_route(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
        )
    }

    fn compute_routing_logits(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<Vec<f32>, String> {
        match_float1(
            hidden_states,
            |hidden_states| {
                crate::ops::moe_routing::compute_routing_logits(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                crate::ops::moe_routing::compute_routing_logits(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                crate::ops::moe_routing::compute_routing_logits(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
        )
    }

    fn add_bias(
        &self,
        output: TensorSliceMut<'_>,
        bias: TensorSlice<'_>,
        batch: usize,
        features: usize,
    ) -> Result<(), String> {
        match_float1_out(
            "add_bias",
            bias,
            output,
            |bias, output| crate::ops::linear::add_bias(output, bias, batch, features),
            |bias, output| crate::ops::linear::add_bias(output, bias, batch, features),
            |bias, output| crate::ops::linear::add_bias(output, bias, batch, features),
        )
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }
}
