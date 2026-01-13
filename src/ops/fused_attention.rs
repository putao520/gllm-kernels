//! Fused QKV projection + attention operator with CUDA kernel fallback.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
#[cfg(feature = "fused-kernel")]
use burn::tensor::TensorData;

use crate::ops::flash_attention::HierarchicalFlashAttention;

#[cfg(feature = "fused-kernel")]
use std::any::{Any, TypeId};
#[cfg(feature = "fused-kernel")]
use crate::cuda_kernels::FusedQKVAttentionKernel as CudaFusedQKVAttentionKernel;
#[cfg(feature = "fused-kernel")]
use cudarc::driver::CudaContext;
#[cfg(feature = "fused-kernel")]
use half::f16;
#[cfg(feature = "fused-kernel")]
use std::sync::Arc;

#[cfg(feature = "fused-kernel")]
const MAX_HEAD_DIM: usize = 256;

/// Fused QKV projection + attention module.
#[derive(Debug, Clone)]
pub struct FusedQKVAttention<B: Backend> {
    w_qkv: Tensor<B, 4>,
    b_qkv: Option<Tensor<B, 3>>,
    attention: HierarchicalFlashAttention,
    enable_kernel: bool,
    min_seq_len: usize,
    causal: bool,
    position_offset: usize,
}

impl<B: Backend> FusedQKVAttention<B> {
    /// Create a new fused QKV attention module.
    pub fn new(w_qkv: Tensor<B, 4>, b_qkv: Option<Tensor<B, 3>>) -> Self {
        Self {
            w_qkv,
            b_qkv,
            attention: HierarchicalFlashAttention::default_config(),
            enable_kernel: true,
            min_seq_len: 1,
            causal: false,
            position_offset: 0,
        }
    }

    /// Override the attention implementation used for unfused fallback.
    pub fn with_attention(mut self, attention: HierarchicalFlashAttention) -> Self {
        self.attention = attention;
        self
    }

    /// Enable or disable the CUDA fused kernel path.
    pub fn with_kernel(mut self, enable: bool) -> Self {
        self.enable_kernel = enable;
        self
    }

    /// Minimum sequence length before attempting the fused kernel.
    pub fn with_min_seq_len(mut self, min_seq_len: usize) -> Self {
        self.min_seq_len = min_seq_len.max(1);
        self
    }

    /// Configure causal behavior for the unfused fallback.
    pub fn with_causal(mut self, causal: bool, position_offset: usize) -> Self {
        self.causal = causal;
        self.position_offset = position_offset;
        self
    }

    /// Forward pass with optional CUDA kernel acceleration.
    pub fn forward(&self, input: &Tensor<B, 3>) -> Tensor<B, 4>
    where
        B::FloatElem: 'static,
    {
        #[cfg(feature = "fused-kernel")]
        if self.should_use_fused::<B>(input) {
            if let Some(output) = self.try_fused_kernel(input) {
                return output;
            }
        }

        self.forward_unfused(input)
    }

    fn forward_unfused(&self, input: &Tensor<B, 3>) -> Tensor<B, 4> {
        let device = input.device();
        let [batch_size, seq_len, hidden_dim] = input.dims();
        let [qkv_dim, num_heads, head_dim, hidden_dim_w] = self.w_qkv.dims();

        if batch_size == 0 || seq_len == 0 {
            return Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
        }

        if qkv_dim != 3 || hidden_dim != hidden_dim_w || head_dim == 0 || num_heads == 0 {
            log::warn!("Fused attention unfused fallback: invalid QKV weight shape");
            return Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
        }
        if let Some(bias) = self.b_qkv.as_ref() {
            if bias.dims() != [3, num_heads, head_dim] {
                log::warn!("Fused attention unfused fallback: invalid bias shape");
                return Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
            }
        }

        let w_q = self
            .w_qkv
            .clone()
            .slice([0..1, 0..num_heads, 0..head_dim, 0..hidden_dim])
            .reshape([num_heads, head_dim, hidden_dim]);
        let w_k = self
            .w_qkv
            .clone()
            .slice([1..2, 0..num_heads, 0..head_dim, 0..hidden_dim])
            .reshape([num_heads, head_dim, hidden_dim]);
        let w_v = self
            .w_qkv
            .clone()
            .slice([2..3, 0..num_heads, 0..head_dim, 0..hidden_dim])
            .reshape([num_heads, head_dim, hidden_dim]);

        let b_q = self
            .b_qkv
            .as_ref()
            .map(|bias| bias.clone().slice([0..1, 0..num_heads, 0..head_dim]).reshape([num_heads, head_dim]));
        let b_k = self
            .b_qkv
            .as_ref()
            .map(|bias| bias.clone().slice([1..2, 0..num_heads, 0..head_dim]).reshape([num_heads, head_dim]));
        let b_v = self
            .b_qkv
            .as_ref()
            .map(|bias| bias.clone().slice([2..3, 0..num_heads, 0..head_dim]).reshape([num_heads, head_dim]));

        let q = project_qkv(input, w_q, b_q);
        let k = project_qkv(input, w_k, b_k);
        let v = project_qkv(input, w_v, b_v);

        self.attention.forward(q, k, v, self.causal, self.position_offset)
    }

    #[cfg(feature = "fused-kernel")]
    fn should_use_fused<T: Backend + 'static>(&self, input: &Tensor<T, 3>) -> bool {
        let seq_len = input.dims()[1];
        self.enable_kernel
            && !self.causal
            && self.position_offset == 0
            && seq_len >= self.min_seq_len
            && is_cuda_backend::<T>()
    }

    #[cfg(feature = "fused-kernel")]
    fn try_fused_kernel<T: Backend + 'static>(&self, input: &Tensor<T, 3>) -> Option<Tensor<T, 4>>
    where
        T::FloatElem: 'static,
    {
        let [batch_size, seq_len, hidden_dim] = input.dims();
        let [qkv_dim, num_heads, head_dim, hidden_dim_w] = self.w_qkv.dims();

        if self.causal || self.position_offset != 0 {
            log::warn!("CUDA fused attention fallback: causal attention not supported");
            return None;
        }

        if qkv_dim != 3 || hidden_dim != hidden_dim_w || head_dim == 0 || num_heads == 0 {
            log::warn!("CUDA fused attention fallback: invalid QKV weight shape");
            return None;
        }
        if let Some(bias) = self.b_qkv.as_ref() {
            if bias.dims() != [3, num_heads, head_dim] {
                log::warn!("CUDA fused attention fallback: invalid bias shape");
                return None;
            }
        }
        if head_dim > MAX_HEAD_DIM {
            log::warn!("CUDA fused attention fallback: head_dim {} exceeds MAX_HEAD_DIM", head_dim);
            return None;
        }

        let device = input.device();
        let cuda_index = {
            #[cfg(feature = "cuda")]
            {
                let device_any = &device as &dyn Any;
                if let Some(cuda_device) = device_any.downcast_ref::<burn_cuda::CudaDevice>() {
                    cuda_device.index
                } else {
                    0
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                0
            }
        };

        let cuda_ctx = match CudaContext::new(cuda_index) {
            Ok(ctx) => Arc::new(ctx),
            Err(err) => {
                log::warn!("CUDA fused attention fallback: device init failed: {err}");
                return None;
            }
        };

        let stream = cuda_ctx.default_stream();
        let kernel = match CudaFusedQKVAttentionKernel::new(&cuda_ctx) {
            Ok(kernel) => kernel,
            Err(err) => {
                log::warn!("CUDA fused attention fallback: kernel load failed: {err}");
                return None;
            }
        };

        let elem_type = TypeId::of::<T::FloatElem>();
        if elem_type == TypeId::of::<f32>() {
            let input_host = input.clone().into_data().into_vec::<f32>().ok()?;
            let w_host = self.w_qkv.clone().into_data().into_vec::<f32>().ok()?;

            let bias_host = match &self.b_qkv {
                Some(bias) => bias.clone().into_data().into_vec::<f32>().ok()?,
                None => vec![0.0f32; 3 * num_heads * head_dim],
            };

            let input_dev = stream.clone_htod(&input_host).ok()?;
            let w_dev = stream.clone_htod(&w_host).ok()?;
            let b_dev = stream.clone_htod(&bias_host).ok()?;

            let output = kernel
                .forward_f32(
                    &stream,
                    &input_dev,
                    &w_dev,
                    &b_dev,
                    batch_size,
                    seq_len,
                    hidden_dim,
                    num_heads,
                    head_dim,
                )
                .ok()?;

            let out_host = stream.clone_dtoh(&output).ok()?;
            return Some(Tensor::<T, 4>::from_data(
                TensorData::new(out_host, [batch_size, num_heads, seq_len, head_dim]),
                &device,
            ));
        }

        if elem_type == TypeId::of::<f16>() {
            let input_host = input.clone().into_data().into_vec::<f16>().ok()?;
            let w_host = self.w_qkv.clone().into_data().into_vec::<f16>().ok()?;

            let bias_host = match &self.b_qkv {
                Some(bias) => bias.clone().into_data().into_vec::<f16>().ok()?,
                None => vec![f16::from_f32(0.0); 3 * num_heads * head_dim],
            };

            let input_dev = stream.clone_htod(&input_host).ok()?;
            let w_dev = stream.clone_htod(&w_host).ok()?;
            let b_dev = stream.clone_htod(&bias_host).ok()?;

            let output = kernel
                .forward_f16(
                    &stream,
                    &input_dev,
                    &w_dev,
                    &b_dev,
                    batch_size,
                    seq_len,
                    hidden_dim,
                    num_heads,
                    head_dim,
                )
                .ok()?;

            let out_host = stream.clone_dtoh(&output).ok()?;
            return Some(Tensor::<T, 4>::from_data(
                TensorData::new(out_host, [batch_size, num_heads, seq_len, head_dim]),
                &device,
            ));
        }

        log::warn!("CUDA fused attention fallback: unsupported dtype");
        None
    }
}

fn project_qkv<B: Backend>(
    input: &Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 2>>,
) -> Tensor<B, 4> {
    let device = input.device();
    let [batch_size, seq_len, hidden_dim] = input.dims();
    let [num_heads, head_dim, hidden_dim_w] = weight.dims();

    if batch_size == 0 || seq_len == 0 {
        return Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
    }

    if hidden_dim != hidden_dim_w {
        log::warn!("Fused attention fallback: hidden_dim mismatch");
        return Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
    }

    let tokens = batch_size.saturating_mul(seq_len);
    let input_flat = input.clone().reshape([tokens, hidden_dim]);
    let weight_flat = weight.reshape([num_heads * head_dim, hidden_dim]);
    let weight_t = weight_flat.transpose();

    let projected = input_flat.matmul(weight_t);
    let projected = projected
        .reshape([batch_size, seq_len, num_heads, head_dim])
        .swap_dims(1, 2);

    if let Some(bias) = bias {
        let bias = bias.reshape([1, num_heads, 1, head_dim]);
        projected + bias
    } else {
        projected
    }
}

#[cfg(feature = "fused-kernel")]
fn is_cuda_backend<B: Backend + 'static>() -> bool {
    #[cfg(feature = "cuda")]
    {
        let type_id = TypeId::of::<B>();
        if type_id == TypeId::of::<burn_cuda::Cuda>() {
            return true;
        }
        #[cfg(feature = "fusion")]
        {
            if type_id == TypeId::of::<burn_fusion::Fusion<burn_cuda::Cuda>>() {
                return true;
            }
        }
        false
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
