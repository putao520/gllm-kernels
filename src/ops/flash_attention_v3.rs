//! FlashAttention-3 style optimizations with CUDA-aware fallback.

use std::any::Any;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::ops::flash_attention::{FlashAttentionConfig, HierarchicalFlashAttention};
#[cfg(feature = "cuda-kernel")]
use crate::cuda_kernels::FlashAttentionKernel;
#[cfg(feature = "cuda-kernel")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda-kernel")]
use half::f16;
#[cfg(feature = "cuda-kernel")]
use std::any::TypeId;
#[cfg(feature = "cuda-kernel")]
use std::sync::Arc;

/// Configuration for FlashAttention-3 optimizations.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttention3Config {
    /// Hopper WGMMA instruction flag.
    pub use_wgmma: bool,
    /// Enable TMA async pipeline for producer/consumer warps.
    pub async_pipeline: bool,
    /// Enable FP8 quantization.
    pub fp8_enabled: bool,
    /// Enable block-wise FP8 quantization.
    pub block_quantization: bool,
}

impl Default for FlashAttention3Config {
    fn default() -> Self {
        Self {
            use_wgmma: false,
            async_pipeline: false,
            fp8_enabled: false,
            block_quantization: false,
        }
    }
}

impl FlashAttention3Config {
    /// Mask configuration with compile-time feature flags.
    pub fn effective(&self) -> Self {
        Self {
            use_wgmma: self.use_wgmma && cfg!(feature = "flash-attention-v3-wgmma"),
            async_pipeline: self.async_pipeline && cfg!(feature = "flash-attention-v3-async"),
            fp8_enabled: self.fp8_enabled && cfg!(feature = "flash-attention-v3-fp8"),
            block_quantization: self.block_quantization
                && cfg!(feature = "flash-attention-v3-block-quant"),
        }
    }

    /// Check whether any optimizations are enabled.
    pub fn any_enabled(&self) -> bool {
        self.use_wgmma || self.async_pipeline || self.fp8_enabled || self.block_quantization
    }
}

/// FlashAttention-3 wrapper that falls back to the hierarchical implementation.
#[derive(Debug, Clone)]
pub struct FlashAttention3 {
    base: HierarchicalFlashAttention,
    config: FlashAttention3Config,
}

impl FlashAttention3 {
    /// Create a new FlashAttention-3 wrapper.
    pub fn new(base: FlashAttentionConfig, config: FlashAttention3Config) -> Self {
        Self {
            base: HierarchicalFlashAttention::new(base),
            config,
        }
    }

    /// Access the FlashAttention-3 configuration.
    pub fn config(&self) -> &FlashAttention3Config {
        &self.config
    }

    /// Access the base FlashAttention configuration.
    pub fn base_config(&self) -> &FlashAttentionConfig {
        self.base.config()
    }

    /// Forward pass with FlashAttention-3 optimizations when available.
    pub fn forward<B: Backend + 'static>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        if self.should_use_v3::<B>() {
            self.forward_v3(q, k, v, causal, position_offset)
        } else {
            self.base.forward(q, k, v, causal, position_offset)
        }
    }

    /// Forward pass using the CUDA kernel when available, otherwise fall back.
    pub fn forward_cuda_kernel<B: Backend + 'static>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4>
    where
        B::FloatElem: 'static,
    {
        #[cfg(feature = "cuda-kernel")]
        {
            if is_cuda_backend::<B>() {
                if let Some(output) = self.try_forward_cuda_kernel(&q, &k, &v, causal, position_offset)
                {
                    return output;
                }
            }
        }

        self.forward(q, k, v, causal, position_offset)
    }

    fn should_use_v3<B: Backend + 'static>(&self) -> bool {
        let config = self.config.effective();
        cfg!(feature = "flash-attention-v3") && config.any_enabled() && is_cuda_backend::<B>()
    }

    fn forward_v3<B: Backend + 'static>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let config = self.config.effective();

        let (k, v) = if config.fp8_enabled {
            self.quantize_kv(k, v, config)
        } else {
            (k, v)
        };

        if config.async_pipeline {
            self.forward_async_pipeline(q, k, v, causal, position_offset)
        } else {
            self.forward_interleaved(q, k, v, causal, position_offset)
        }
    }

    fn forward_interleaved<B: Backend + 'static>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        // The hierarchical implementation already interleaves matmul and softmax updates.
        self.base.forward(q, k, v, causal, position_offset)
    }

    fn forward_async_pipeline<B: Backend + 'static>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        // Placeholder for producer/consumer warp specialization. CPU/WGPU fall back to the base path.
        self.base.forward(q, k, v, causal, position_offset)
    }

    fn quantize_kv<B: Backend + 'static>(
        &self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        config: FlashAttention3Config,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if config.block_quantization {
            let block_size = self.base.config().block_kv.max(1);
            (
                Self::quantize_fp8_blocked(k, block_size),
                Self::quantize_fp8_blocked(v, block_size),
            )
        } else {
            (Self::quantize_fp8(k), Self::quantize_fp8(v))
        }
    }

    fn quantize_fp8<B: Backend + 'static>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        const FP8_MAX: f32 = 240.0;
        let device = tensor.device();
        let dims = tensor.dims();
        let mut data = tensor
            .into_data()
            .into_vec::<f32>()
            .expect("fp8 quantization expects f32 data");

        let mut max_abs = 0.0f32;
        for value in &data {
            max_abs = max_abs.max(value.abs());
        }
        let scale = if max_abs > 0.0 { max_abs / FP8_MAX } else { 1.0 };

        for value in &mut data {
            let q = (*value / scale).round().max(-FP8_MAX).min(FP8_MAX);
            *value = q * scale;
        }

        Tensor::<B, 4>::from_data(TensorData::new(data, dims), &device)
    }

    #[cfg(feature = "cuda-kernel")]
    fn try_forward_cuda_kernel<B: Backend + 'static>(
        &self,
        q: &Tensor<B, 4>,
        k: &Tensor<B, 4>,
        v: &Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Option<Tensor<B, 4>>
    where
        B::FloatElem: 'static,
    {
        let q_dims = q.dims();
        if k.dims() != q_dims || v.dims() != q_dims {
            log::warn!("CUDA kernel fallback: Q/K/V shape mismatch");
            return None;
        }

        let [batch_size, num_heads, seq_len, head_dim] = q_dims;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device = q.device();

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
                log::warn!("CUDA kernel fallback: device init failed: {err}");
                return None;
            }
        };

        let stream = cuda_ctx.default_stream();
        let kernel = match FlashAttentionKernel::new(&cuda_ctx) {
            Ok(kernel) => kernel,
            Err(err) => {
                log::warn!("CUDA kernel fallback: kernel load failed: {err}");
                return None;
            }
        };

        let elem_type = TypeId::of::<B::FloatElem>();
        if elem_type == TypeId::of::<f32>() {
            let q_host = q
                .clone()
                .into_data()
                .into_vec::<f32>()
                .ok()?;
            let k_host = k
                .clone()
                .into_data()
                .into_vec::<f32>()
                .ok()?;
            let v_host = v
                .clone()
                .into_data()
                .into_vec::<f32>()
                .ok()?;

            let q_dev = stream.clone_htod(&q_host).ok()?;
            let k_dev = stream.clone_htod(&k_host).ok()?;
            let v_dev = stream.clone_htod(&v_host).ok()?;

            let output = kernel
                .forward_f32(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    causal,
                    scale,
                    position_offset,
                )
                .ok()?;

            let out_host = stream.clone_dtoh(&output).ok()?;
            return Some(Tensor::<B, 4>::from_data(TensorData::new(out_host, q_dims), &device));
        }

        if elem_type == TypeId::of::<f16>() {
            let q_host = q
                .clone()
                .into_data()
                .into_vec::<f16>()
                .ok()?;
            let k_host = k
                .clone()
                .into_data()
                .into_vec::<f16>()
                .ok()?;
            let v_host = v
                .clone()
                .into_data()
                .into_vec::<f16>()
                .ok()?;

            let q_dev = stream.clone_htod(&q_host).ok()?;
            let k_dev = stream.clone_htod(&k_host).ok()?;
            let v_dev = stream.clone_htod(&v_host).ok()?;

            let output = kernel
                .forward_f16(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    causal,
                    scale,
                    position_offset,
                )
                .ok()?;

            let out_host = stream.clone_dtoh(&output).ok()?;
            return Some(Tensor::<B, 4>::from_data(TensorData::new(out_host, q_dims), &device));
        }

        log::warn!("CUDA kernel fallback: unsupported dtype");
        None
    }

    fn quantize_fp8_blocked<B: Backend + 'static>(
        tensor: Tensor<B, 4>,
        block_size: usize,
    ) -> Tensor<B, 4> {
        const FP8_MAX: f32 = 240.0;
        let device = tensor.device();
        let dims = tensor.dims();
        let [batch, num_heads, seq_len, head_dim] = dims;
        let block_size = block_size.max(1);
        let mut data = tensor
            .into_data()
            .into_vec::<f32>()
            .expect("fp8 quantization expects f32 data");

        let stride_seq = head_dim;
        let stride_head = seq_len * stride_seq;
        let stride_batch = num_heads * stride_head;

        for batch_idx in 0..batch {
            let base_batch = batch_idx * stride_batch;
            for head_idx in 0..num_heads {
                let base_head = base_batch + head_idx * stride_head;
                let mut start = 0usize;
                while start < seq_len {
                    let end = (start + block_size).min(seq_len);
                    let mut max_abs = 0.0f32;
                    for seq_idx in start..end {
                        let row = base_head + seq_idx * stride_seq;
                        for dim_idx in 0..head_dim {
                            max_abs = max_abs.max(data[row + dim_idx].abs());
                        }
                    }
                    let scale = if max_abs > 0.0 { max_abs / FP8_MAX } else { 1.0 };
                    for seq_idx in start..end {
                        let row = base_head + seq_idx * stride_seq;
                        for dim_idx in 0..head_dim {
                            let idx = row + dim_idx;
                            let q = (data[idx] / scale).round().max(-FP8_MAX).min(FP8_MAX);
                            data[idx] = q * scale;
                        }
                    }
                    start = end;
                }
            }
        }

        Tensor::<B, 4>::from_data(TensorData::new(data, dims), &device)
    }
}

fn is_cuda_backend<B: Backend + 'static>() -> bool {
    #[cfg(feature = "cuda")]
    {
        use std::any::TypeId;

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

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_flash_attention_v3_effective_config() {
        let config = FlashAttention3Config {
            use_wgmma: true,
            async_pipeline: true,
            fp8_enabled: true,
            block_quantization: true,
        };
        let effective = config.effective();

        assert_eq!(
            effective.use_wgmma,
            config.use_wgmma && cfg!(feature = "flash-attention-v3-wgmma")
        );
        assert_eq!(
            effective.async_pipeline,
            config.async_pipeline && cfg!(feature = "flash-attention-v3-async")
        );
        assert_eq!(
            effective.fp8_enabled,
            config.fp8_enabled && cfg!(feature = "flash-attention-v3-fp8")
        );
        assert_eq!(
            effective.block_quantization,
            config.block_quantization && cfg!(feature = "flash-attention-v3-block-quant")
        );
    }

    #[test]
    fn test_flash_attention_v3_fallback_matches_base() {
        let device = <TestBackend as Backend>::Device::default();
        let base_config = FlashAttentionConfig {
            block_q: 4,
            block_kv: 4,
            use_log_space: false,
            ..Default::default()
        };
        let base = HierarchicalFlashAttention::new(base_config.clone());
        let v3 = FlashAttention3::new(
            base_config,
            FlashAttention3Config {
                use_wgmma: true,
                async_pipeline: true,
                fp8_enabled: true,
                block_quantization: true,
            },
        );

        let q = Tensor::<TestBackend, 4>::random([1, 2, 4, 4], Distribution::Normal(0.0, 0.5), &device);
        let k = Tensor::<TestBackend, 4>::random([1, 2, 4, 4], Distribution::Normal(0.0, 0.5), &device);
        let v = Tensor::<TestBackend, 4>::random([1, 2, 4, 4], Distribution::Normal(0.0, 0.5), &device);

        let out_base = base.forward(q.clone(), k.clone(), v.clone(), false, 0);
        let out_v3 = v3.forward(q, k, v, false, 0);

        let base_data = out_base
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        let v3_data = out_v3
            .into_data()
            .into_vec::<f32>()
            .expect("output data");

        for (idx, (base_val, v3_val)) in base_data.iter().zip(v3_data.iter()).enumerate() {
            let diff = (base_val - v3_val).abs();
            assert!(diff < 1e-3, "Mismatch at {}: base={}, v3={}", idx, base_val, v3_val);
        }
    }
}
