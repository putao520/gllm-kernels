use std::sync::Arc;

use crate::runtime_detection::{detect_backend, BackendType};

pub use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
pub use crate::cpu_backend::CpuBackend;
pub use crate::cuda_backend::CudaBackend;
pub use crate::metal_backend::MetalBackend;
pub use crate::rocm_backend::RocmBackend;
pub use crate::wgpu_backend::WgpuBackend;

#[derive(Clone)]
pub enum BackendImpl {
    Cpu(CpuBackend),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "metal")]
    Metal(MetalBackend),
    #[cfg(feature = "rocm")]
    Rocm(RocmBackend),
}

macro_rules! dispatch_backend {
    ($self:expr, $method:ident $(, $args:expr)* $(,)?) => {
        match $self {
            BackendImpl::Cpu(b) => b.$method($($args),*),
            #[cfg(feature = "wgpu")]
            BackendImpl::Wgpu(b) => b.$method($($args),*),
            #[cfg(feature = "cuda")]
            BackendImpl::Cuda(b) => b.$method($($args),*),
            #[cfg(feature = "metal")]
            BackendImpl::Metal(b) => b.$method($($args),*),
            #[cfg(feature = "rocm")]
            BackendImpl::Rocm(b) => b.$method($($args),*),
        }
    };
}

impl Backend for BackendImpl {
    fn flash_attention(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        v: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: crate::kernel_types::FlashAttentionConfig,
    ) -> Result<(), String> {
        dispatch_backend!(self, flash_attention, q, k, v, output, config)
    }

    fn paged_attention(
        &self,
        q: TensorSlice<'_>,
        k_cache: TensorSlice<'_>,
        v_cache: TensorSlice<'_>,
        page_table: &[u32],
        seq_lens: &[u32],
        output: TensorSliceMut<'_>,
        config: crate::kernel_types::PagedAttentionConfig,
    ) -> Result<(), String> {
        dispatch_backend!(
            self,
            paged_attention,
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            output,
            config
        )
    }

    fn softmax(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: crate::kernel_types::SoftmaxConfig,
    ) -> Result<(), String> {
        dispatch_backend!(self, softmax, input, output, config)
    }

    fn matmul(
        &self,
        a: TensorSlice<'_>,
        b: TensorSlice<'_>,
        c: TensorSliceMut<'_>,
        config: crate::kernel_types::MatmulConfig,
    ) -> Result<(), String> {
        dispatch_backend!(self, matmul, a, b, c, config)
    }

    fn q4_matmul(
        &self,
        input: &[f32],
        q_weight: &[u8],
        scales: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        dispatch_backend!(self, q4_matmul, input, q_weight, scales, m, n, k)
    }

    fn q8_matmul(
        &self,
        input: &[f32],
        q_weight: &[i8],
        scales: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        dispatch_backend!(self, q8_matmul, input, q_weight, scales, m, n, k)
    }

    fn q4_dequantize(
        &self,
        q_weight: &[u8],
        scales: &[half::f16],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        dispatch_backend!(self, q4_dequantize, q_weight, scales, n, k)
    }

    fn awq_dequantize(
        &self,
        qweight: &[u32],
        qzeros: &[u32],
        scales: &[half::f16],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<f32>, String> {
        dispatch_backend!(self, awq_dequantize, qweight, qzeros, scales, n, k, group_size)
    }

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
    ) -> Result<Vec<f32>, String> {
        dispatch_backend!(
            self,
            awq_matmul,
            input,
            qweight,
            qzeros,
            scales,
            m,
            n,
            k,
            group_size
        )
    }

    fn rope_precompute(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
        config: crate::ops::rope::RoPEConfig,
    ) -> Result<(), String> {
        dispatch_backend!(self, rope_precompute, cos_out, sin_out, config)
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
        dispatch_backend!(
            self,
            rope_apply,
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
            position_offset
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
        dispatch_backend!(
            self,
            rope_apply_inplace,
            x,
            cos_cache,
            sin_cache,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            position_offset
        )
    }

    fn topk(
        &self,
        logits: TensorSlice<'_>,
        k: usize,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<crate::ops::sampling::TopKResult, String> {
        dispatch_backend!(self, topk, logits, k, batch_size, vocab_size)
    }

    fn apply_temperature(
        &self,
        logits: TensorSliceMut<'_>,
        temperature: f32,
    ) -> Result<(), String> {
        dispatch_backend!(self, apply_temperature, logits, temperature)
    }

    fn sample_tokens(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
        config: &crate::ops::sampling::SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        dispatch_backend!(self, sample_tokens, logits, batch_size, vocab_size, config)
    }

    fn argmax(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, String> {
        dispatch_backend!(self, argmax, logits, batch_size, vocab_size)
    }

    fn moe_route(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &crate::ops::moe_routing::MoERoutingConfig,
    ) -> Result<crate::ops::moe_routing::MoERoutingResult, String> {
        dispatch_backend!(
            self,
            moe_route,
            hidden_states,
            gate_weights,
            batch_size,
            seq_len,
            config
        )
    }

    fn compute_routing_logits(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &crate::ops::moe_routing::MoERoutingConfig,
    ) -> Result<Vec<f32>, String> {
        dispatch_backend!(
            self,
            compute_routing_logits,
            hidden_states,
            gate_weights,
            batch_size,
            seq_len,
            config
        )
    }

    fn rms_norm(
        &self,
        input: TensorSlice<'_>,
        weight: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        dispatch_backend!(self, rms_norm, input, weight, output, batch, hidden, eps)
    }

    fn rms_norm_inplace(
        &self,
        data: TensorSliceMut<'_>,
        weight: TensorSlice<'_>,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        dispatch_backend!(self, rms_norm_inplace, data, weight, batch, hidden, eps)
    }

    fn silu_inplace(
        &self,
        data: TensorSliceMut<'_>,
    ) -> Result<(), String> {
        dispatch_backend!(self, silu_inplace, data)
    }

    fn silu(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
    ) -> Result<(), String> {
        dispatch_backend!(self, silu, input, output)
    }

    fn add_bias(
        &self,
        output: TensorSliceMut<'_>,
        bias: TensorSlice<'_>,
        batch: usize,
        features: usize,
    ) -> Result<(), String> {
        dispatch_backend!(self, add_bias, output, bias, batch, features)
    }

    fn backend_type(&self) -> BackendType {
        dispatch_backend!(self, backend_type)
    }
}

pub fn auto_select_static() -> BackendImpl {
    #[cfg(feature = "cuda")]
    if crate::runtime_detection::try_cuda() {
        return BackendImpl::Cuda(CudaBackend::new());
    }
    #[cfg(feature = "rocm")]
    if crate::runtime_detection::try_rocm() {
        return BackendImpl::Rocm(RocmBackend::new());
    }
    #[cfg(feature = "metal")]
    if crate::runtime_detection::try_metal() {
        return BackendImpl::Metal(MetalBackend::new());
    }
    #[cfg(feature = "wgpu")]
    if crate::runtime_detection::try_wgpu() {
        return BackendImpl::Wgpu(WgpuBackend::new());
    }
    BackendImpl::Cpu(CpuBackend::new())
}

pub fn auto_select_backend() -> Arc<dyn Backend> {
    match detect_backend() {
        BackendType::Cuda => Arc::new(CudaBackend::new()),
        BackendType::Rocm => Arc::new(RocmBackend::new()),
        BackendType::Metal => Arc::new(MetalBackend::new()),
        BackendType::Wgpu => Arc::new(WgpuBackend::new()),
        BackendType::Cpu => Arc::new(CpuBackend::new()),
    }
}
