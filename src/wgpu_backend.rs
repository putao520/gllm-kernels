use crate::backend_core::BackendCore;
use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
use crate::kernel_dispatcher::{
    FlashAttentionConfig, MatmulConfig, PagedAttentionConfig, SoftmaxConfig,
};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};
use crate::ops::rope::RoPEConfig;
use crate::ops::sampling::{SamplingConfig, TopKResult};
use crate::runtime_detection::BackendType;

pub struct WgpuBackend {
    core: BackendCore,
}

impl WgpuBackend {
    pub fn new() -> Self {
        Self {
            core: BackendCore::new(BackendType::Wgpu),
        }
    }
}

impl Default for WgpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for WgpuBackend {
    fn flash_attention(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        v: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: FlashAttentionConfig,
    ) -> Result<(), String> {
        self.core.flash_attention(q, k, v, output, config)
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
        self.core
            .paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config)
    }

    fn softmax(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: SoftmaxConfig,
    ) -> Result<(), String> {
        self.core.softmax(input, output, config)
    }

    fn matmul(
        &self,
        a: TensorSlice<'_>,
        b: TensorSlice<'_>,
        c: TensorSliceMut<'_>,
        config: MatmulConfig,
    ) -> Result<(), String> {
        self.core.matmul(a, b, c, config)
    }

    fn rope_precompute(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
        config: RoPEConfig,
    ) -> Result<(), String> {
        self.core.rope_precompute(cos_out, sin_out, config)
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
        self.core.rope_apply(
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
        self.core.rope_apply_inplace(
            x,
            cos_cache,
            sin_cache,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            position_offset,
        )
    }

    fn topk(
        &self,
        logits: TensorSlice<'_>,
        k: usize,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<TopKResult, String> {
        self.core.topk(logits, k, batch_size, vocab_size)
    }

    fn apply_temperature(
        &self,
        logits: TensorSliceMut<'_>,
        temperature: f32,
    ) -> Result<(), String> {
        self.core.apply_temperature(logits, temperature)
    }

    fn sample_tokens(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        self.core.sample_tokens(logits, batch_size, vocab_size, config)
    }

    fn argmax(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, String> {
        self.core.argmax(logits, batch_size, vocab_size)
    }

    fn moe_route(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<MoERoutingResult, String> {
        self.core
            .moe_route(hidden_states, gate_weights, batch_size, seq_len, config)
    }

    fn compute_routing_logits(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<Vec<f32>, String> {
        self.core
            .compute_routing_logits(hidden_states, gate_weights, batch_size, seq_len, config)
    }

    fn add_bias(
        &self,
        output: TensorSliceMut<'_>,
        bias: TensorSlice<'_>,
        batch: usize,
        features: usize,
    ) -> Result<(), String> {
        self.core.add_bias(output, bias, batch, features)
    }

    fn backend_type(&self) -> BackendType {
        self.core.backend_type()
    }
}
