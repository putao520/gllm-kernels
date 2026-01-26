use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};

use crate::backend_match::{
    apply_f32_binary_out, apply_f32_inplace_weight, apply_f32_unary_inplace, apply_f32_unary_out,
    match_float1, match_float1_mut, match_float1_mut_weight, match_float1_out,
    match_float2_out, match_float2_out2, match_float3_out,
};
use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
use crate::cuda_kernels::{
    FlashAttentionKernel as CudaFlashAttentionKernel,
    PagedAttentionKernel as CudaPagedAttentionKernel,
    RmsNormKernel,
    CudaSilu,
};
use crate::kernel_types::{
    FlashAttentionConfig, KernelFloat, MatmulConfig, PagedAttentionConfig, SoftmaxConfig,
};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};
use crate::ops::rope::RoPEConfig;
use crate::ops::sampling::{SamplingConfig, TopKResult};
use crate::runtime_detection::BackendType;

/// Global CUDA context (lazy initialized).
static CUDA_CONTEXT: OnceLock<Option<Arc<CudaContext>>> = OnceLock::new();
/// Global CUDA stream (lazy initialized).
static CUDA_STREAM: OnceLock<Option<Arc<CudaStream>>> = OnceLock::new();
/// Global CUDA flash attention kernel (lazy initialized).
static CUDA_FLASH_KERNEL: OnceLock<Option<CudaFlashAttentionKernel>> = OnceLock::new();
/// Global CUDA paged attention kernel (lazy initialized).
static CUDA_PAGED_KERNEL: OnceLock<Option<CudaPagedAttentionKernel>> = OnceLock::new();
/// Global CUDA RMSNorm kernel (lazy initialized).
static CUDA_RMSNORM_KERNEL: OnceLock<Option<RmsNormKernel>> = OnceLock::new();
/// Global CUDA SiLU kernel (lazy initialized).
static CUDA_SILU_KERNEL: OnceLock<Option<CudaSilu>> = OnceLock::new();

fn get_cuda_context() -> Option<&'static Arc<CudaContext>> {
    CUDA_CONTEXT
        .get_or_init(|| {
            match CudaContext::new(0) {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    log::warn!("Failed to create CUDA context: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_stream() -> Option<&'static Arc<CudaStream>> {
    CUDA_STREAM
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            Some(ctx.default_stream())
        })
        .as_ref()
}

fn get_cuda_flash_kernel() -> Option<&'static CudaFlashAttentionKernel> {
    CUDA_FLASH_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaFlashAttentionKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA flash attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_paged_kernel() -> Option<&'static CudaPagedAttentionKernel> {
    CUDA_PAGED_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaPagedAttentionKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA paged attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_rmsnorm_kernel() -> Option<&'static RmsNormKernel> {
    CUDA_RMSNORM_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match RmsNormKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA RMSNorm kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_silu_kernel() -> Option<&'static CudaSilu> {
    CUDA_SILU_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaSilu::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA SiLU kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

/// CUDA flash attention dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn cuda_flash_attention<T: KernelFloat>(
    kernel: &CudaFlashAttentionKernel,
    stream: &Arc<CudaStream>,
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: &FlashAttentionConfig,
) -> bool {
    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    let q_buf = match stream.clone_htod(&q_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy Q to GPU: {}", e);
            return false;
        }
    };
    let k_buf = match stream.clone_htod(&k_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy K to GPU: {}", e);
            return false;
        }
    };
    let v_buf = match stream.clone_htod(&v_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy V to GPU: {}", e);
            return false;
        }
    };

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let seq_len = config.seq_len_q;

    let result = kernel.forward_f32(
        stream,
        &q_buf,
        &k_buf,
        &v_buf,
        config.batch_size,
        config.num_heads,
        seq_len,
        config.head_dim,
        config.causal,
        scale,
        0,
    );

    match result {
        Ok(out_buf) => match stream.clone_dtoh(&out_buf) {
            Ok(out_data) => {
                for (i, val) in out_data.into_iter().enumerate() {
                    if i < output.len() {
                        output[i] = T::from_f32(val);
                    }
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("CUDA kernel execution failed: {}", e);
            false
        }
    }
}

/// CUDA paged attention dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn cuda_paged_attention<T: KernelFloat>(
    kernel: &CudaPagedAttentionKernel,
    stream: &Arc<CudaStream>,
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [T],
    config: &PagedAttentionConfig,
) -> bool {
    let inputs = match crate::ops::paged_attn::build_paged_gpu_inputs(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        output.len(),
        config,
    ) {
        Some(inputs) => inputs,
        None => return false,
    };

    let q_buf = match stream.clone_htod(&inputs.q_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy Q to GPU: {}", e);
            return false;
        }
    };
    let k_buf = match stream.clone_htod(&inputs.k_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy K to GPU: {}", e);
            return false;
        }
    };
    let v_buf = match stream.clone_htod(&inputs.v_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy V to GPU: {}", e);
            return false;
        }
    };
    let table_buf = match stream.clone_htod(&inputs.block_tables) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy block_tables to GPU: {}", e);
            return false;
        }
    };
    let offsets_buf = match stream.clone_htod(&inputs.block_offsets) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy block_offsets to GPU: {}", e);
            return false;
        }
    };

    let result = kernel.forward_f32(
        stream,
        &q_buf,
        &k_buf,
        &v_buf,
        &table_buf,
        &offsets_buf,
        inputs.layout.batch_size,
        inputs.layout.num_heads,
        inputs.layout.head_dim,
        inputs.layout.page_size,
        inputs.layout.seq_len,
    );

    match result {
        Ok(out_buf) => match stream.clone_dtoh(&out_buf) {
            Ok(out_data) => {
                for (i, value) in out_data.into_iter().enumerate() {
                    if i < output.len() {
                        output[i] = T::from_f32(value);
                    }
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("CUDA paged kernel execution failed: {}", e);
            false
        }
    }
}

/// CUDA RMSNorm dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn cuda_rms_norm<T: KernelFloat>(
    kernel: &RmsNormKernel,
    stream: &Arc<CudaStream>,
    input: &[T],
    weight: &[T],
    output: &mut [T],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> bool {
    let expected = batch.saturating_mul(hidden);
    if input.len() != expected || output.len() != expected || weight.len() != hidden {
        log::debug!("CUDA rms_norm dispatch skipped: length mismatch");
        return false;
    }

    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();

    let input_buf = match stream.clone_htod(&input_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy rms_norm input to GPU: {}", e);
            return false;
        }
    };
    let weight_buf = match stream.clone_htod(&weight_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy rms_norm weight to GPU: {}", e);
            return false;
        }
    };
    let mut output_buf = match stream.alloc_zeros(output.len()) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm output on GPU: {}", e);
            return false;
        }
    };

    if let Err(e) = kernel.forward(stream, &input_buf, &weight_buf, &mut output_buf, batch, hidden, eps) {
        log::debug!("CUDA rms_norm kernel execution failed: {}", e);
        return false;
    }

    let output_f32 = match stream.clone_dtoh(&output_buf) {
        Ok(out_data) => out_data,
        Err(e) => {
            log::debug!("Failed to copy rms_norm output from GPU: {}", e);
            return false;
        }
    };

    for (dst, val) in output.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }

    true
}

/// CUDA RMSNorm inplace dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn cuda_rms_norm_inplace<T: KernelFloat>(
    kernel: &RmsNormKernel,
    stream: &Arc<CudaStream>,
    data: &mut [T],
    weight: &[T],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> bool {
    let expected = batch.saturating_mul(hidden);
    if data.len() != expected || weight.len() != hidden {
        log::debug!("CUDA rms_norm_inplace dispatch skipped: length mismatch");
        return false;
    }

    let data_f32: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();

    let data_buf = match stream.clone_htod(&data_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy rms_norm data to GPU: {}", e);
            return false;
        }
    };
    let weight_buf = match stream.clone_htod(&weight_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy rms_norm weight to GPU: {}", e);
            return false;
        }
    };
    let mut output_buf = match stream.alloc_zeros(data.len()) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm output on GPU: {}", e);
            return false;
        }
    };

    if let Err(e) = kernel.forward(stream, &data_buf, &weight_buf, &mut output_buf, batch, hidden, eps) {
        log::debug!("CUDA rms_norm kernel execution failed: {}", e);
        return false;
    }

    let output_f32 = match stream.clone_dtoh(&output_buf) {
        Ok(out_data) => out_data,
        Err(e) => {
            log::debug!("Failed to copy rms_norm output from GPU: {}", e);
            return false;
        }
    };

    for (dst, val) in data.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }

    true
}

/// CUDA SiLU dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn cuda_silu<T: KernelFloat>(
    kernel: &CudaSilu,
    stream: &Arc<CudaStream>,
    input: &[T],
    output: &mut [T],
) -> bool {
    if input.len() != output.len() {
        log::debug!("CUDA silu dispatch skipped: length mismatch");
        return false;
    }

    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();

    let input_buf = match stream.clone_htod(&input_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy silu input to GPU: {}", e);
            return false;
        }
    };
    let mut output_buf = match stream.alloc_zeros(output.len()) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate silu output on GPU: {}", e);
            return false;
        }
    };

    if let Err(e) = kernel.forward(stream, &input_buf, &mut output_buf, output.len()) {
        log::debug!("CUDA silu kernel execution failed: {}", e);
        return false;
    }

    let output_f32 = match stream.clone_dtoh(&output_buf) {
        Ok(out_data) => out_data,
        Err(e) => {
            log::debug!("Failed to copy silu output from GPU: {}", e);
            return false;
        }
    };

    for (dst, val) in output.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }

    true
}

/// CUDA SiLU inplace dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn cuda_silu_inplace<T: KernelFloat>(
    kernel: &CudaSilu,
    stream: &Arc<CudaStream>,
    data: &mut [T],
) -> bool {
    let data_f32: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();

    let mut data_buf = match stream.clone_htod(&data_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy silu data to GPU: {}", e);
            return false;
        }
    };

    if let Err(e) = kernel.forward_inplace(stream, &mut data_buf, data.len()) {
        log::debug!("CUDA silu inplace kernel execution failed: {}", e);
        return false;
    }

    let output_f32 = match stream.clone_dtoh(&data_buf) {
        Ok(out_data) => out_data,
        Err(e) => {
            log::debug!("Failed to copy silu inplace output from GPU: {}", e);
            return false;
        }
    };

    for (dst, val) in data.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }

    true
}

pub struct CudaBackend {}

impl CudaBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CudaBackend {
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
            |q, k, v, out| {
                if let (Some(kernel), Some(stream)) = (get_cuda_flash_kernel(), get_cuda_stream()) {
                    if cuda_flash_attention(kernel, stream, q, k, v, out, &config) {
                        return;
                    }
                    log::debug!("CUDA kernel dispatch failed, falling back to CPU");
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                if let (Some(kernel), Some(stream)) = (get_cuda_flash_kernel(), get_cuda_stream()) {
                    if cuda_flash_attention(kernel, stream, q, k, v, out, &config) {
                        return;
                    }
                    log::debug!("CUDA kernel dispatch failed, falling back to CPU");
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                if let (Some(kernel), Some(stream)) = (get_cuda_flash_kernel(), get_cuda_stream()) {
                    if cuda_flash_attention(kernel, stream, q, k, v, out, &config) {
                        return;
                    }
                    log::debug!("CUDA kernel dispatch failed, falling back to CPU");
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
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
                if let (Some(kernel), Some(stream)) = (get_cuda_paged_kernel(), get_cuda_stream()) {
                    if cuda_paged_attention(kernel, stream, q, k, v, page_table, seq_lens, out, &config) {
                        return;
                    }
                    log::debug!("CUDA paged attention dispatch failed, falling back to CPU");
                }
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
                if let (Some(kernel), Some(stream)) = (get_cuda_paged_kernel(), get_cuda_stream()) {
                    if cuda_paged_attention(kernel, stream, q, k, v, page_table, seq_lens, out, &config) {
                        return;
                    }
                    log::debug!("CUDA paged attention dispatch failed, falling back to CPU");
                }
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
                if let (Some(kernel), Some(stream)) = (get_cuda_paged_kernel(), get_cuda_stream()) {
                    if cuda_paged_attention(kernel, stream, q, k, v, page_table, seq_lens, out, &config) {
                        return;
                    }
                    log::debug!("CUDA paged attention dispatch failed, falling back to CPU");
                }
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

    fn rms_norm(
        &self,
        input: TensorSlice<'_>,
        weight: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        match_float2_out(
            "rms_norm",
            input,
            weight,
            output,
            |input, weight, output| {
                if let (Some(kernel), Some(stream)) = (get_cuda_rmsnorm_kernel(), get_cuda_stream()) {
                    if cuda_rms_norm(kernel, stream, input, weight, output, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("CUDA rms_norm dispatch failed, falling back to CPU");
                }
                crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
            },
            |input, weight, output| {
                if let (Some(kernel), Some(stream)) = (get_cuda_rmsnorm_kernel(), get_cuda_stream()) {
                    if cuda_rms_norm(kernel, stream, input, weight, output, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("CUDA rms_norm dispatch failed, falling back to CPU");
                }
                apply_f32_binary_out(input, weight, output, |input, weight, output| {
                    crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
                });
            },
            |input, weight, output| {
                if let (Some(kernel), Some(stream)) = (get_cuda_rmsnorm_kernel(), get_cuda_stream()) {
                    if cuda_rms_norm(kernel, stream, input, weight, output, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("CUDA rms_norm dispatch failed, falling back to CPU");
                }
                apply_f32_binary_out(input, weight, output, |input, weight, output| {
                    crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
                });
            },
        )
    }

    fn rms_norm_inplace(
        &self,
        data: TensorSliceMut<'_>,
        weight: TensorSlice<'_>,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        match_float1_mut_weight(
            "rms_norm_inplace",
            data,
            weight,
            |data, weight| {
                if let (Some(kernel), Some(stream)) = (get_cuda_rmsnorm_kernel(), get_cuda_stream()) {
                    if cuda_rms_norm_inplace(kernel, stream, data, weight, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("CUDA rms_norm_inplace dispatch failed, falling back to CPU");
                }
                crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
            },
            |data, weight| {
                if let (Some(kernel), Some(stream)) = (get_cuda_rmsnorm_kernel(), get_cuda_stream()) {
                    if cuda_rms_norm_inplace(kernel, stream, data, weight, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("CUDA rms_norm_inplace dispatch failed, falling back to CPU");
                }
                apply_f32_inplace_weight(data, weight, |data, weight| {
                    crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
                });
            },
            |data, weight| {
                if let (Some(kernel), Some(stream)) = (get_cuda_rmsnorm_kernel(), get_cuda_stream()) {
                    if cuda_rms_norm_inplace(kernel, stream, data, weight, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("CUDA rms_norm_inplace dispatch failed, falling back to CPU");
                }
                apply_f32_inplace_weight(data, weight, |data, weight| {
                    crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
                });
            },
        )
    }

    fn silu_inplace(
        &self,
        data: TensorSliceMut<'_>,
    ) -> Result<(), String> {
        match_float1_mut(
            data,
            |data| {
                if let (Some(kernel), Some(stream)) = (get_cuda_silu_kernel(), get_cuda_stream()) {
                    if cuda_silu_inplace(kernel, stream, data) {
                        return;
                    }
                    log::debug!("CUDA silu_inplace dispatch failed, falling back to CPU");
                }
                crate::ops::activations::silu_inplace(data);
            },
            |data| {
                if let (Some(kernel), Some(stream)) = (get_cuda_silu_kernel(), get_cuda_stream()) {
                    if cuda_silu_inplace(kernel, stream, data) {
                        return;
                    }
                    log::debug!("CUDA silu_inplace dispatch failed, falling back to CPU");
                }
                apply_f32_unary_inplace(data, |data| {
                    crate::ops::activations::silu_inplace(data);
                });
            },
            |data| {
                if let (Some(kernel), Some(stream)) = (get_cuda_silu_kernel(), get_cuda_stream()) {
                    if cuda_silu_inplace(kernel, stream, data) {
                        return;
                    }
                    log::debug!("CUDA silu_inplace dispatch failed, falling back to CPU");
                }
                apply_f32_unary_inplace(data, |data| {
                    crate::ops::activations::silu_inplace(data);
                });
            },
        )
    }

    fn silu(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
    ) -> Result<(), String> {
        match_float1_out(
            "silu",
            input,
            output,
            |input, output| {
                if let (Some(kernel), Some(stream)) = (get_cuda_silu_kernel(), get_cuda_stream()) {
                    if cuda_silu(kernel, stream, input, output) {
                        return;
                    }
                    log::debug!("CUDA silu dispatch failed, falling back to CPU");
                }
                crate::ops::activations::silu(input, output);
            },
            |input, output| {
                if let (Some(kernel), Some(stream)) = (get_cuda_silu_kernel(), get_cuda_stream()) {
                    if cuda_silu(kernel, stream, input, output) {
                        return;
                    }
                    log::debug!("CUDA silu dispatch failed, falling back to CPU");
                }
                apply_f32_unary_out(input, output, |input, output| {
                    crate::ops::activations::silu(input, output);
                });
            },
            |input, output| {
                if let (Some(kernel), Some(stream)) = (get_cuda_silu_kernel(), get_cuda_stream()) {
                    if cuda_silu(kernel, stream, input, output) {
                        return;
                    }
                    log::debug!("CUDA silu dispatch failed, falling back to CPU");
                }
                apply_f32_unary_out(input, output, |input, output| {
                    crate::ops::activations::silu(input, output);
                });
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
        BackendType::Cuda
    }
}
