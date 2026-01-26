use crate::kernel_types::{
    FlashAttentionConfig, MatmulConfig, PagedAttentionConfig, SoftmaxConfig,
};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};
use crate::ops::rope::RoPEConfig;
use crate::ops::sampling::{SamplingConfig, TopKResult};
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

pub trait Backend: Send + Sync {
    fn flash_attention(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        v: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: FlashAttentionConfig,
    ) -> Result<(), String>;

    fn paged_attention(
        &self,
        q: TensorSlice<'_>,
        k_cache: TensorSlice<'_>,
        v_cache: TensorSlice<'_>,
        page_table: &[u32],
        seq_lens: &[u32],
        output: TensorSliceMut<'_>,
        config: PagedAttentionConfig,
    ) -> Result<(), String>;

    fn softmax(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: SoftmaxConfig,
    ) -> Result<(), String>;

    fn matmul(
        &self,
        a: TensorSlice<'_>,
        b: TensorSlice<'_>,
        c: TensorSliceMut<'_>,
        config: MatmulConfig,
    ) -> Result<(), String>;

    /// Q4_0/Q4_K quantized matrix multiplication.
    fn q4_matmul(
        &self,
        input: &[f32],
        q_weight: &[u8],
        scales: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String>;

    /// Q8_0 quantized matrix multiplication.
    fn q8_matmul(
        &self,
        input: &[f32],
        q_weight: &[i8],
        scales: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String>;

    /// Q4_0 dequantization (packed INT4 -> F32).
    fn q4_dequantize(
        &self,
        q_weight: &[u8],
        scales: &[half::f16],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String>;

    /// AWQ INT4 dequantization (packed INT4 -> F32).
    fn awq_dequantize(
        &self,
        qweight: &[u32],
        qzeros: &[u32],
        scales: &[half::f16],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<f32>, String>;

    /// AWQ INT4 quantized matrix multiplication.
    fn awq_matmul(
        &self,
        input: &[f32],
        qweight: &[u32],
        qzeros: &[u32],
        scales: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<f32>, String>;

    fn rope_precompute(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
        config: RoPEConfig,
    ) -> Result<(), String>;

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
    ) -> Result<(), String>;

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
    ) -> Result<(), String>;

    fn topk(
        &self,
        logits: TensorSlice<'_>,
        k: usize,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<TopKResult, String>;

    fn apply_temperature(
        &self,
        logits: TensorSliceMut<'_>,
        temperature: f32,
    ) -> Result<(), String>;

    fn sample_tokens(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String>;

    fn argmax(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, String>;

    fn moe_route(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<MoERoutingResult, String>;

    fn compute_routing_logits(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<Vec<f32>, String>;

    fn rms_norm(
        &self,
        input: TensorSlice<'_>,
        weight: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String>;

    fn rms_norm_inplace(
        &self,
        data: TensorSliceMut<'_>,
        weight: TensorSlice<'_>,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String>;

    fn silu_inplace(
        &self,
        data: TensorSliceMut<'_>,
    ) -> Result<(), String>;

    fn silu(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
    ) -> Result<(), String>;

    fn add_bias(
        &self,
        output: TensorSliceMut<'_>,
        bias: TensorSlice<'_>,
        batch: usize,
        features: usize,
    ) -> Result<(), String>;

    fn backend_type(&self) -> BackendType;
}
