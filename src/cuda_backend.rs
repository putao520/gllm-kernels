use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};

use crate::backend_trait::Backend;
use crate::cuda_kernels::{
    FlashAttentionKernel as CudaFlashAttentionKernel,
    PagedAttentionKernel as CudaPagedAttentionKernel,
    QuantizedDequantKernel,
    RmsNormKernel,
    CudaRoPEKernel,
    CudaSilu,
    CudaLinear,
    CudaElementwiseKernel,
    CudaPoolingKernel,
};
use crate::kernel_types::{
    AttentionBlockConfig, FFNBlockConfig, EmbeddingConfig, LMHeadConfig,
    KVCacheUpdateConfig, MeanPoolingConfig, ClsPoolingConfig, NormalizeConfig,
    DequantizeConfig, EngramFuseConfig, FlashAttentionConfig,
    KernelFloat, PagedAttentionConfig, MatmulConfig, LinearParams,
    // L3 High-level inference configs (ARCH-ADR-003)
    TransformerLayerWeights, MoETransformerLayerWeights, KVCacheState,
    GeneratorForwardConfig, MoEGeneratorForwardConfig, EmbeddingForwardConfig,
    RerankForwardConfig,
    // GPU-pure weights (ARCH-GPU-001 / ARCH-ADR-010)
    EmbeddingModelWeightsGpu, TransformerLayerWeightsGpu,
    RerankerModelWeightsGpu, GeneratorModelWeightsGpu, KVCacheGpu,
    TransformerLayerConfigGpu,
};
use crate::gpu_types::{GpuBuffer, GpuTensor, TensorDtype};
use crate::ops::sampling::SamplingConfig;
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
/// Global CUDA RoPE kernel (lazy initialized).
static CUDA_ROPE_KERNEL: OnceLock<Option<CudaRoPEKernel>> = OnceLock::new();
/// Global CUDA SiLU kernel (lazy initialized).
static CUDA_SILU_KERNEL: OnceLock<Option<CudaSilu>> = OnceLock::new();
/// Global CUDA quantized dequantization kernel (lazy initialized).
static CUDA_QUANTIZED_KERNEL: OnceLock<Option<QuantizedDequantKernel>> = OnceLock::new();
/// Global CUDA linear kernel (lazy initialized).
static CUDA_LINEAR_KERNEL: OnceLock<Option<CudaLinear>> = OnceLock::new();
/// Global CUDA elementwise kernel (lazy initialized).
static CUDA_ELEMENTWISE_KERNEL: OnceLock<Option<CudaElementwiseKernel>> = OnceLock::new();
/// Global CUDA pooling kernel (lazy initialized).
static CUDA_POOLING_KERNEL: OnceLock<Option<CudaPoolingKernel>> = OnceLock::new();

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

fn get_cuda_rope_kernel() -> Option<&'static CudaRoPEKernel> {
    CUDA_ROPE_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaRoPEKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA RoPE kernel: {}", e);
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

fn get_cuda_quantized_kernel() -> Option<&'static QuantizedDequantKernel> {
    CUDA_QUANTIZED_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match QuantizedDequantKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA quantized dequant kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_linear_kernel() -> Option<&'static CudaLinear> {
    CUDA_LINEAR_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaLinear::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA linear kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_elementwise_kernel() -> Option<&'static CudaElementwiseKernel> {
    CUDA_ELEMENTWISE_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaElementwiseKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA elementwise kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_cuda_pooling_kernel() -> Option<&'static CudaPoolingKernel> {
    CUDA_POOLING_KERNEL
        .get_or_init(|| {
            let ctx = get_cuda_context()?;
            match CudaPoolingKernel::new(ctx) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize CUDA pooling kernel: {}", e);
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

fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize, String> {
    a.checked_mul(b).ok_or_else(|| format!("{name} overflow"))
}

fn cuda_q4_dequantize(
    kernel: &QuantizedDequantKernel,
    stream: &Arc<CudaStream>,
    q_weight: &[u8],
    scales: &[half::f16],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if n == 0 || k == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if k % 32 != 0 {
        return Err("k must be multiple of 32 for Q4".into());
    }
    let blocks = k / 32;
    let num_blocks = checked_mul(n, blocks, "q4 blocks")?;
    if num_blocks > i32::MAX as usize {
        return Err("q4 blocks exceed addressable range".into());
    }
    let expected_q_weight = checked_mul(num_blocks, 16, "q4 weights")?;
    if q_weight.len() != expected_q_weight {
        return Err(format!(
            "q_weight length mismatch: expected {expected_q_weight}, got {}",
            q_weight.len()
        ));
    }
    if scales.len() != num_blocks {
        return Err(format!(
            "scales length mismatch: expected {num_blocks}, got {}",
            scales.len()
        ));
    }
    if num_blocks
        .checked_mul(32)
        .ok_or_else(|| "output overflow".to_string())?
        > u32::MAX as usize
    {
        return Err("output exceeds addressable range".into());
    }

    let q_weight_buf = stream.clone_htod(q_weight).map_err(|e| {
        format!("CUDA Q4 dequantize upload failed: {e}")
    })?;
    let scales_buf = stream.clone_htod(scales).map_err(|e| {
        format!("CUDA Q4 dequantize scales upload failed: {e}")
    })?;
    let output_buf = kernel
        .dequantize_q4(stream, &q_weight_buf, &scales_buf, num_blocks)
        .map_err(|e| format!("CUDA Q4 dequantize failed: {e}"))?;
    stream
        .clone_dtoh(&output_buf)
        .map_err(|e| format!("CUDA Q4 dequantize readback failed: {e}"))
}

fn cuda_awq_dequantize(
    kernel: &QuantizedDequantKernel,
    stream: &Arc<CudaStream>,
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[half::f16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    if n == 0 || k == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    if n % 8 != 0 {
        return Err("n must be multiple of 8 for AWQ packing".into());
    }
    if k % group_size != 0 {
        return Err("k must be multiple of group_size for AWQ".into());
    }
    let groups = k / group_size;
    if groups > i32::MAX as usize {
        return Err("group count exceeds addressable range".into());
    }
    let packed_out = n / 8;
    let expected_qweight = checked_mul(packed_out, k, "qweight")?;
    let expected_qzeros = checked_mul(packed_out, groups, "qzeros")?;
    let expected_scales = checked_mul(n, groups, "scales")?;
    if qweight.len() != expected_qweight {
        return Err(format!(
            "qweight length mismatch: expected {expected_qweight}, got {}",
            qweight.len()
        ));
    }
    if qzeros.len() != expected_qzeros {
        return Err(format!(
            "qzeros length mismatch: expected {expected_qzeros}, got {}",
            qzeros.len()
        ));
    }
    if scales.len() != expected_scales {
        return Err(format!(
            "scales length mismatch: expected {expected_scales}, got {}",
            scales.len()
        ));
    }
    if n > i32::MAX as usize || k > i32::MAX as usize || group_size > i32::MAX as usize {
        return Err("dimensions exceed addressable range".into());
    }
    if n
        .checked_mul(k)
        .ok_or_else(|| "output overflow".to_string())?
        > u32::MAX as usize
    {
        return Err("output exceeds addressable range".into());
    }

    let qweight_buf = stream.clone_htod(qweight).map_err(|e| {
        format!("CUDA AWQ dequantize upload failed: {e}")
    })?;
    let qzeros_buf = stream.clone_htod(qzeros).map_err(|e| {
        format!("CUDA AWQ dequantize zeros upload failed: {e}")
    })?;
    let scales_buf = stream.clone_htod(scales).map_err(|e| {
        format!("CUDA AWQ dequantize scales upload failed: {e}")
    })?;
    let output_buf = kernel
        .dequantize_awq(stream, &qweight_buf, &qzeros_buf, &scales_buf, n, k, group_size)
        .map_err(|e| format!("CUDA AWQ dequantize failed: {e}"))?;
    stream
        .clone_dtoh(&output_buf)
        .map_err(|e| format!("CUDA AWQ dequantize readback failed: {e}"))
}

#[derive(Clone, Copy)]
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
    // =========================================================================
    // L2 Block-Level Operators (ARCH-GRANULARITY-001)
    // Fused GPU pipeline: data stays on GPU throughout entire L2 operation
    // Only one upload (CPU→GPU) at start and one download (GPU→CPU) at end
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
        _kv_cache_k: Option<&mut [f32]>,
        _kv_cache_v: Option<&mut [f32]>,
        config: &AttentionBlockConfig,
    ) -> Result<Vec<f32>, String> {
        // NOTE: This is a CPU API (accepts &[f32] host slices).
        // Per ARCH-ADR-010, GPU optimization should happen at higher level
        // (generator_forward_gpu_pure) with GpuTensor inputs.
        // Block-level CPU API should use CPU backend directly.
        crate::cpu_backend::CpuBackend::new().attention_block(
            hidden, q_weight, k_weight, v_weight, o_weight,
            norm_weight, cos_cache, sin_cache, _kv_cache_k, _kv_cache_v, config,
        )
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
        // NOTE: This is a CPU API (accepts &[f32] host slices).
        // Per ARCH-ADR-010, GPU optimization should happen at higher level
        // (generator_forward_gpu_pure) with GpuTensor inputs.
        // Block-level CPU API should use CPU backend directly.
        crate::cpu_backend::CpuBackend::new().ffn_block(
            hidden, gate_weight, up_weight, down_weight, norm_weight, config,
        )
    }

    fn embedding(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        position_weight: Option<&[f32]>,
        config: &EmbeddingConfig,
    ) -> Result<Vec<f32>, String> {
        // Embedding is efficient on CPU (simple lookup), no GPU benefit
        crate::cpu_backend::CpuBackend::new().embedding(
            tokens, embed_weight, position_weight, config,
        )
    }

    fn lm_head(
        &self,
        hidden: &[f32],
        lm_weight: &[f32],
        norm_weight: &[f32],
        config: &LMHeadConfig,
    ) -> Result<Vec<f32>, String> {
        // NOTE: This is a CPU API (accepts &[f32] host slices).
        // Per ARCH-ADR-010, GPU optimization should happen at higher level
        // (generator_forward_gpu_pure) with GpuTensor inputs.
        // Block-level CPU API should use CPU backend directly.
        crate::cpu_backend::CpuBackend::new().lm_head(hidden, lm_weight, norm_weight, config)
    }

    fn engram_lookup(
        &self,
        tokens: &[u32],
        engram_table: &[f32],
        hidden_size: usize,
        ngram_size: usize,
        num_buckets: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), String> {
        // Engram lookup is always CPU (DRAM-based hash table)
        crate::cpu_backend::CpuBackend::new().engram_lookup(
            tokens, engram_table, hidden_size, ngram_size, num_buckets,
        )
    }

    fn engram_fuse(
        &self,
        attention_output: &[f32],
        engram_output: &[f32],
        config: &EngramFuseConfig,
    ) -> Result<Vec<f32>, String> {
        // Simple element-wise fusion, CPU is efficient
        crate::cpu_backend::CpuBackend::new().engram_fuse(attention_output, engram_output, config)
    }

    fn kv_cache_update(
        &self,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        new_k: &[f32],
        new_v: &[f32],
        config: &KVCacheUpdateConfig,
    ) -> Result<(), String> {
        // KV cache update is memory-bound, CPU efficient
        crate::cpu_backend::CpuBackend::new().kv_cache_update(k_cache, v_cache, new_k, new_v, config)
    }

    fn sample(
        &self,
        logits: &[f32],
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        // Sampling requires RNG, always CPU
        crate::cpu_backend::CpuBackend::new().sample(logits, vocab_size, config)
    }

    fn mean_pooling(
        &self,
        hidden: &[f32],
        attention_mask: Option<&[f32]>,
        config: &MeanPoolingConfig,
    ) -> Result<Vec<f32>, String> {
        // Simple reduction, CPU efficient for typical batch sizes
        crate::cpu_backend::CpuBackend::new().mean_pooling(hidden, attention_mask, config)
    }

    fn cls_pooling(
        &self,
        hidden: &[f32],
        config: &ClsPoolingConfig,
    ) -> Result<Vec<f32>, String> {
        // CLS extraction is just memory copy
        crate::cpu_backend::CpuBackend::new().cls_pooling(hidden, config)
    }

    fn normalize(
        &self,
        input: &[f32],
        config: &NormalizeConfig,
    ) -> Result<Vec<f32>, String> {
        // L2 normalization, CPU efficient for embedding vectors
        crate::cpu_backend::CpuBackend::new().normalize(input, config)
    }

    fn dequantize(
        &self,
        quantized: &[u8],
        scales: &[half::f16],
        zeros: Option<&[u32]>,
        config: &DequantizeConfig,
    ) -> Result<Vec<f32>, String> {
        // Try GPU dequantization for large tensors
        if let (Some(kernel), Some(stream)) = (get_cuda_quantized_kernel(), get_cuda_stream()) {
            let total_elements = config.n * config.k;
            // Use GPU for large tensors (> 1M elements)
            if total_elements > 1_000_000 {
                // Upload quantized data
                if let Ok(quant_buf) = stream.clone_htod(quantized) {
                    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
                    if let Ok(scales_buf) = stream.clone_htod(&scales_f32) {
                        if let Ok(mut output_buf) = stream.alloc_zeros::<f32>(total_elements) {
                            // Call GPU dequantize kernel based on format
                            let success = match config.format {
                                crate::kernel_types::QuantFormat::Q4_0 |
                                crate::kernel_types::QuantFormat::Q4_K => {
                                    // Convert scales from f32 back to f16 for kernel
                                    let scales_f16: Vec<half::f16> = scales.to_vec();
                                    if let Ok(scales_f16_buf) = stream.clone_htod(&scales_f16) {
                                        let num_blocks = (config.n * config.k) / 32;
                                        kernel.dequantize_q4(stream, &quant_buf, &scales_f16_buf, num_blocks).is_ok()
                                    } else { false }
                                }
                                crate::kernel_types::QuantFormat::AWQ => {
                                    if let Some(z) = zeros {
                                        // AWQ uses u32 qweight and qzeros
                                        let qweight: Vec<u32> = quantized.chunks(4).map(|chunk| {
                                            u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0),
                                                               chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)])
                                        }).collect();
                                        if let Ok(qweight_buf) = stream.clone_htod(&qweight) {
                                            if let Ok(qzeros_buf) = stream.clone_htod(z) {
                                                let scales_f16: Vec<half::f16> = scales.to_vec();
                                                if let Ok(scales_f16_buf) = stream.clone_htod(&scales_f16) {
                                                    kernel.dequantize_awq(stream, &qweight_buf, &qzeros_buf, &scales_f16_buf, config.n, config.k, config.group_size).is_ok()
                                                } else { false }
                                            } else { false }
                                        } else { false }
                                    } else { false }
                                }
                                _ => false,
                            };
                            if success {
                                if let Ok(result) = stream.clone_dtoh(&output_buf) {
                                    return Ok(result);
                                }
                            }
                        }
                    }
                }
                log::debug!("CUDA dequantize failed, falling back to CPU");
            }
        }

        // CPU fallback
        crate::cpu_backend::CpuBackend::new().dequantize(quantized, scales, zeros, config)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    // =========================================================================
    // L3 High-Level Inference API (ARCH-ADR-003)
    // CPU fallback for now; GPU-optimized versions can be added later
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
    ) -> Result<Vec<f32>, String> {
        // TODO: Implement CUDA-optimized generator forward
        // For now, delegate to CPU implementation
        crate::cpu_backend::CpuBackend::new().generator_forward(
            tokens, embed_weight, layers, final_norm, lm_head_weight,
            cos_cache, sin_cache, kv_cache, config,
        )
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
    ) -> Result<Vec<f32>, String> {
        // TODO: Implement CUDA-optimized MoE generator forward
        crate::cpu_backend::CpuBackend::new().moe_generator_forward(
            tokens, embed_weight, layers, final_norm, lm_head_weight,
            cos_cache, sin_cache, kv_cache, config,
        )
    }

    fn embedding_forward(
        &self,
        tokens: &[u32],
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: Option<&[f32]>,
        config: &EmbeddingForwardConfig,
    ) -> Result<Vec<f32>, String> {
        // TODO: Implement CUDA-optimized embedding forward
        crate::cpu_backend::CpuBackend::new().embedding_forward(
            tokens, embed_weight, layers, final_norm, config,
        )
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
        // TODO: Implement CUDA-optimized rerank forward
        crate::cpu_backend::CpuBackend::new().rerank_forward(
            tokens, embed_weight, layers, final_norm, score_weight, config,
        )
    }

    // =========================================================================
    // GPU-pure methods (ARCH-GPU-001)
    // Upload weights to GPU ONCE at load time, reuse for all forward passes.
    // NO repeated GPU<->CPU transfers during inference.
    // =========================================================================

    fn upload_embedding_weights(
        &self,
        embed_weight: &[f32],
        layers: &[TransformerLayerWeights<'_>],
        final_norm: Option<&[f32]>,
        _config: &EmbeddingForwardConfig,
    ) -> Result<EmbeddingModelWeightsGpu, String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;

        // Helper to upload f32 slice to CUDA GPU memory
        // Returns GpuTensor with CUDA buffer
        fn upload_f32_to_cuda(
            stream: &Arc<CudaStream>,
            data: &[f32],
            shape: Vec<usize>,
        ) -> Result<GpuTensor, String> {
            if data.is_empty() {
                // Empty tensor - use CPU buffer
                return Ok(GpuTensor::new(
                    GpuBuffer::Cpu(Arc::new(Vec::new())),
                    shape,
                    TensorDtype::F32,
                    BackendType::Cuda,
                ));
            }

            // Convert f32 slice to u8 slice (safe, no allocation)
            let bytes: &[u8] = bytemuck::cast_slice(data);

            // Upload bytes to CUDA GPU
            let cuda_slice = stream.clone_htod(bytes)
                .map_err(|e| format!("CUDA upload failed: {}", e))?;

            Ok(GpuTensor::new(
                GpuBuffer::Cuda(Arc::new(cuda_slice)),
                shape,
                TensorDtype::F32,
                BackendType::Cuda,
            ))
        }

        // Upload embedding weights to GPU
        let embedding = upload_f32_to_cuda(stream, embed_weight, vec![embed_weight.len()])?;

        // Upload all layer weights to GPU
        let gpu_layers: Result<Vec<TransformerLayerWeightsGpu>, String> = layers.iter()
            .map(|layer| {
                Ok(TransformerLayerWeightsGpu {
                    input_norm: upload_f32_to_cuda(stream, layer.input_norm, vec![layer.input_norm.len()])?,
                    q_weight: upload_f32_to_cuda(stream, layer.q_weight, vec![layer.q_weight.len()])?,
                    k_weight: upload_f32_to_cuda(stream, layer.k_weight, vec![layer.k_weight.len()])?,
                    v_weight: upload_f32_to_cuda(stream, layer.v_weight, vec![layer.v_weight.len()])?,
                    o_weight: upload_f32_to_cuda(stream, layer.o_weight, vec![layer.o_weight.len()])?,
                    post_attn_norm: upload_f32_to_cuda(stream, layer.post_attn_norm, vec![layer.post_attn_norm.len()])?,
                    gate_weight: match layer.gate_weight {
                        Some(g) => Some(upload_f32_to_cuda(stream, g, vec![g.len()])?),
                        None => None,
                    },
                    up_weight: upload_f32_to_cuda(stream, layer.up_weight, vec![layer.up_weight.len()])?,
                    down_weight: upload_f32_to_cuda(stream, layer.down_weight, vec![layer.down_weight.len()])?,
                    cos_cache: None,
                    sin_cache: None,
                })
            })
            .collect();
        let gpu_layers = gpu_layers?;

        // Upload final norm to GPU
        let final_norm_data = final_norm.unwrap_or(&[]);
        let final_norm_gpu = upload_f32_to_cuda(stream, final_norm_data, vec![final_norm_data.len()])?;

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
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;

        // Helper to download f32 data from CUDA GPU memory
        fn download_f32_from_cuda(
            stream: &Arc<CudaStream>,
            tensor: &GpuTensor,
        ) -> Result<Vec<f32>, String> {
            match &tensor.buffer {
                GpuBuffer::Cuda(cuda_slice) => {
                    // Download bytes from GPU
                    let bytes: Vec<u8> = stream.clone_dtoh(cuda_slice.as_ref())
                        .map_err(|e| format!("CUDA download failed: {}", e))?;

                    // Convert bytes to f32 (safe copy)
                    let f32_data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                    Ok(f32_data)
                }
                GpuBuffer::Cpu(bytes) => {
                    // CPU buffer fallback (for empty tensors)
                    if bytes.is_empty() {
                        return Ok(Vec::new());
                    }
                    let f32_data: Vec<f32> = bytemuck::cast_slice(bytes.as_slice()).to_vec();
                    Ok(f32_data)
                }
                #[allow(unreachable_patterns)]
                _ => Err("Unexpected GPU buffer type for CUDA backend".to_string()),
            }
        }

        // Download weights from GPU (only once per forward pass)
        // TODO: Implement true GPU-native forward using CUDA kernels to avoid this download
        let embed_weight = download_f32_from_cuda(stream, &weights.embedding)?;
        let final_norm = download_f32_from_cuda(stream, &weights.final_norm)?;
        let final_norm_opt = if final_norm.is_empty() { None } else { Some(final_norm.as_slice()) };

        let layer_weights: Result<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>, Vec<f32>, Vec<f32>)>, String> =
            weights.layers.iter()
                .map(|layer| {
                    Ok((
                        download_f32_from_cuda(stream, &layer.input_norm)?,
                        download_f32_from_cuda(stream, &layer.q_weight)?,
                        download_f32_from_cuda(stream, &layer.k_weight)?,
                        download_f32_from_cuda(stream, &layer.v_weight)?,
                        download_f32_from_cuda(stream, &layer.o_weight)?,
                        download_f32_from_cuda(stream, &layer.post_attn_norm)?,
                        layer.gate_weight.as_ref().map(|g| download_f32_from_cuda(stream, g)).transpose()?,
                        download_f32_from_cuda(stream, &layer.up_weight)?,
                        download_f32_from_cuda(stream, &layer.down_weight)?,
                    ))
                })
                .collect();
        let layer_data = layer_weights?;

        let layers: Vec<TransformerLayerWeights<'_>> = layer_data.iter()
            .map(|(input_norm, q, k, v, o, post_norm, gate, up, down)| {
                TransformerLayerWeights {
                    input_norm: input_norm.as_slice(),
                    q_weight: q.as_slice(),
                    k_weight: k.as_slice(),
                    v_weight: v.as_slice(),
                    o_weight: o.as_slice(),
                    post_attn_norm: post_norm.as_slice(),
                    gate_weight: gate.as_ref().map(|g| g.as_slice()),
                    up_weight: up.as_slice(),
                    down_weight: down.as_slice(),
                }
            })
            .collect();

        // Call existing embedding_forward
        // Note: This still downloads from GPU for computation. True GPU-native forward
        // will be implemented when CUDA embedding/attention/FFN kernels are integrated.
        self.embedding_forward(tokens, &embed_weight, &layers, final_norm_opt, config)
    }

    // =========================================================================
    // GPU-Native Kernel Methods (ARCH-ADR-010)
    // CUDA backend: True GPU-resident computation
    //
    // IMPORTANT: These methods MUST keep all data on GPU throughout computation.
    // Pattern: GpuTensor in → CUDA kernel dispatch → GpuTensor out
    // FORBIDDEN: download → CPU compute → upload (violates ARCH-GPU-001-B)
    // =========================================================================

    fn embedding_lookup_gpu(
        &self,
        tokens: &GpuTensor,
        embed_weight: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let kernel = get_cuda_pooling_kernel()
            .ok_or("CUDA pooling kernel not available")?;

        if tokens.dtype != TensorDtype::U32 {
            return Err("embedding_lookup_gpu: tokens must be U32 tensor".into());
        }
        if embed_weight.dtype != TensorDtype::F32 {
            return Err("embedding_lookup_gpu: embed_weight must be F32 tensor".into());
        }
        if embed_weight.shape.len() < 2 {
            return Err("embedding_lookup_gpu: embed_weight must be 2D [vocab, hidden]".into());
        }

        let num_tokens: usize = tokens.shape.iter().product();
        let hidden_dim = *embed_weight.shape.last().unwrap();
        let embed_elements: usize = embed_weight.shape.iter().product();
        let output_elements = num_tokens
            .checked_mul(hidden_dim)
            .ok_or("embedding_lookup_gpu: output size overflow")?;

        let GpuBuffer::Cuda(tokens_slice) = &tokens.buffer else {
            return Err("embedding_lookup_gpu: tokens must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(embed_slice) = &embed_weight.buffer else {
            return Err("embedding_lookup_gpu: embed_weight must be CUDA buffer".into());
        };

        let token_view = unsafe {
            tokens_slice
                .transmute::<u32>(num_tokens)
                .ok_or("embedding_lookup_gpu: failed to transmute tokens to u32 view")?
        };
        let table_view = unsafe {
            embed_slice
                .transmute::<f32>(embed_elements)
                .ok_or("embedding_lookup_gpu: failed to transmute embedding to f32 view")?
        };

        let output_bytes = output_elements * std::mem::size_of::<f32>();
        let mut output_slice: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(output_bytes)
            .map_err(|e| format!("embedding_lookup_gpu: alloc failed: {}", e))?;
        let mut output_view = unsafe {
            output_slice
                .transmute_mut::<f32>(output_elements)
                .ok_or("embedding_lookup_gpu: failed to transmute output to f32 view")?
        };

        kernel.embedding_gather_view(
            stream,
            &token_view,
            &table_view,
            &mut output_view,
            num_tokens,
            hidden_dim,
        )
        .map_err(|e| format!("embedding_lookup_gpu kernel error: {}", e))?;

        let mut out_shape = tokens.shape.clone();
        out_shape.push(hidden_dim);
        output.buffer = GpuBuffer::Cuda(Arc::new(output_slice));
        output.shape = out_shape;
        output.dtype = TensorDtype::F32;
        output.size_in_bytes = output_bytes;
        output.backend = BackendType::Cuda;

        Ok(())
    }

    fn transformer_layer_gpu(
        &self,
        hidden: &mut GpuTensor,
        layer_weights: &TransformerLayerWeightsGpu,
        kv_cache: Option<&mut KVCacheGpu>,
        config: &TransformerLayerConfigGpu,
    ) -> Result<(), String> {
        if hidden.dtype != TensorDtype::F32 {
            return Err("transformer_layer_gpu: hidden must be F32 tensor".into());
        }
        if hidden.shape.len() < 2 {
            return Err("transformer_layer_gpu: hidden must be at least 2D".into());
        }

        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let rms_kernel = get_cuda_rmsnorm_kernel()
            .ok_or("CUDA RMSNorm kernel not available")?;
        let linear_kernel = get_cuda_linear_kernel()
            .ok_or("CUDA linear kernel not available")?;
        let rope_kernel = get_cuda_rope_kernel()
            .ok_or("CUDA RoPE kernel not available")?;
        let silu_kernel = get_cuda_silu_kernel()
            .ok_or("CUDA SiLU kernel not available")?;
        let flash_kernel = get_cuda_flash_kernel()
            .ok_or("CUDA flash attention kernel not available")?;
        let elem_kernel = get_cuda_elementwise_kernel()
            .ok_or("CUDA elementwise kernel not available")?;

        let batch = config.batch_size;
        let seq = config.seq_len;
        let hidden_dim = config.hidden_size;
        let total_tokens = batch
            .checked_mul(seq)
            .ok_or("transformer_layer_gpu: batch*seq overflow")?;

        let hidden_elements: usize = hidden.shape.iter().product();
        let expected_hidden = total_tokens
            .checked_mul(hidden_dim)
            .ok_or("transformer_layer_gpu: hidden size overflow")?;
        if hidden_elements != expected_hidden {
            return Err(format!(
                "transformer_layer_gpu: hidden shape mismatch (got {}, expected {})",
                hidden_elements, expected_hidden
            ));
        }

        let q_out = config.num_q_heads
            .checked_mul(config.head_dim)
            .ok_or("transformer_layer_gpu: q_out overflow")?;
        let kv_out = config.num_kv_heads
            .checked_mul(config.head_dim)
            .ok_or("transformer_layer_gpu: kv_out overflow")?;

        // Extract hidden buffer
        let GpuBuffer::Cuda(hidden_slice) = &hidden.buffer else {
            return Err("transformer_layer_gpu: hidden must be CUDA buffer".into());
        };
        let hidden_view = unsafe {
            hidden_slice
                .transmute::<f32>(hidden_elements)
                .ok_or("transformer_layer_gpu: failed to transmute hidden to f32 view")?
        };

        // Load norm weights
        let GpuBuffer::Cuda(input_norm_slice) = &layer_weights.input_norm.buffer else {
            return Err("transformer_layer_gpu: input_norm must be CUDA buffer".into());
        };
        let input_norm_view = unsafe {
            input_norm_slice
                .transmute::<f32>(hidden_dim)
                .ok_or("transformer_layer_gpu: failed to transmute input_norm to f32 view")?
        };

        let mut normed: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(hidden_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc normed failed: {}", e))?;
        {
            let mut normed_view = normed.as_view_mut();
            rms_kernel
                .forward_view(
                    stream,
                    &hidden_view,
                    &input_norm_view,
                    &mut normed_view,
                    total_tokens,
                    hidden_dim,
                    config.rms_norm_eps,
                )
                .map_err(|e| format!("transformer_layer_gpu rms_norm error: {}", e))?;
        }

        // QKV projections
        let q_weight_elems = q_out
            .checked_mul(hidden_dim)
            .ok_or("transformer_layer_gpu: q_weight size overflow")?;
        let kv_weight_elems = kv_out
            .checked_mul(hidden_dim)
            .ok_or("transformer_layer_gpu: kv_weight size overflow")?;

        let GpuBuffer::Cuda(q_weight_slice) = &layer_weights.q_weight.buffer else {
            return Err("transformer_layer_gpu: q_weight must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(k_weight_slice) = &layer_weights.k_weight.buffer else {
            return Err("transformer_layer_gpu: k_weight must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(v_weight_slice) = &layer_weights.v_weight.buffer else {
            return Err("transformer_layer_gpu: v_weight must be CUDA buffer".into());
        };

        let q_weight_view = unsafe {
            q_weight_slice
                .transmute::<f32>(q_weight_elems)
                .ok_or("transformer_layer_gpu: failed to transmute q_weight")?
        };
        let k_weight_view = unsafe {
            k_weight_slice
                .transmute::<f32>(kv_weight_elems)
                .ok_or("transformer_layer_gpu: failed to transmute k_weight")?
        };
        let v_weight_view = unsafe {
            v_weight_slice
                .transmute::<f32>(kv_weight_elems)
                .ok_or("transformer_layer_gpu: failed to transmute v_weight")?
        };

        let q_elements = total_tokens
            .checked_mul(q_out)
            .ok_or("transformer_layer_gpu: q_elements overflow")?;
        let kv_elements = total_tokens
            .checked_mul(kv_out)
            .ok_or("transformer_layer_gpu: kv_elements overflow")?;

        let mut q_linear: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(q_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc q_linear failed: {}", e))?;
        let mut k_linear: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(kv_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc k_linear failed: {}", e))?;
        let mut v_linear: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(kv_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc v_linear failed: {}", e))?;

        let params_q = LinearParams {
            in_features: hidden_dim as u32,
            out_features: q_out as u32,
            has_bias: 0,
            padding: 0,
        };
        let params_kv = LinearParams {
            in_features: hidden_dim as u32,
            out_features: kv_out as u32,
            has_bias: 0,
            padding: 0,
        };

        {
            let normed_view = normed.as_view();
            let mut q_out_view = q_linear.as_view_mut();
            linear_kernel.forward_view(
                stream,
                params_q,
                &normed_view,
                &q_weight_view,
                None,
                &mut q_out_view,
                total_tokens,
            )?;
        }
        {
            let normed_view = normed.as_view();
            let mut k_out_view = k_linear.as_view_mut();
            linear_kernel.forward_view(
                stream,
                params_kv,
                &normed_view,
                &k_weight_view,
                None,
                &mut k_out_view,
                total_tokens,
            )?;
        }
        {
            let normed_view = normed.as_view();
            let mut v_out_view = v_linear.as_view_mut();
            linear_kernel.forward_view(
                stream,
                params_kv,
                &normed_view,
                &v_weight_view,
                None,
                &mut v_out_view,
                total_tokens,
            )?;
        }

        // Apply RoPE to Q and K (in-place, layout: [batch, seq, heads, head_dim])
        let (cos_cache, sin_cache) = match (layer_weights.cos_cache.as_ref(), layer_weights.sin_cache.as_ref()) {
            (Some(cos), Some(sin)) => (cos, sin),
            _ => return Err("transformer_layer_gpu: RoPE cache missing (cos/sin)".into()),
        };
        if cos_cache.dtype != TensorDtype::F32 || sin_cache.dtype != TensorDtype::F32 {
            return Err("transformer_layer_gpu: RoPE cache must be F32 tensor".into());
        }
        let GpuBuffer::Cuda(cos_cache_slice) = &cos_cache.buffer else {
            return Err("transformer_layer_gpu: cos_cache must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(sin_cache_slice) = &sin_cache.buffer else {
            return Err("transformer_layer_gpu: sin_cache must be CUDA buffer".into());
        };
        if config.head_dim % 2 != 0 {
            return Err("transformer_layer_gpu: head_dim must be even for RoPE".into());
        }
        let pos_end = config.position
            .checked_add(seq)
            .ok_or("transformer_layer_gpu: RoPE position overflow")?;
        let needed = pos_end
            .checked_mul(config.head_dim / 2)
            .ok_or("transformer_layer_gpu: RoPE cache size overflow")?;
        let cos_len = cos_cache.size_in_bytes / std::mem::size_of::<f32>();
        let sin_len = sin_cache.size_in_bytes / std::mem::size_of::<f32>();
        if cos_len < needed || sin_len < needed {
            return Err("transformer_layer_gpu: RoPE cache length is insufficient".into());
        }
        let cos_view = unsafe {
            cos_cache_slice
                .transmute::<f32>(cos_len)
                .ok_or("transformer_layer_gpu: failed to transmute cos_cache")?
        };
        let sin_view = unsafe {
            sin_cache_slice
                .transmute::<f32>(sin_len)
                .ok_or("transformer_layer_gpu: failed to transmute sin_cache")?
        };
        {
            let mut q_view = q_linear.as_view_mut();
            rope_kernel
                .apply_inplace_f32(
                    stream,
                    &mut q_view,
                    &cos_view,
                    &sin_view,
                    batch,
                    seq,
                    config.num_q_heads,
                    config.head_dim,
                    config.position,
                )
                .map_err(|e| format!("transformer_layer_gpu RoPE q error: {}", e))?;
        }
        {
            let mut k_view = k_linear.as_view_mut();
            rope_kernel
                .apply_inplace_f32(
                    stream,
                    &mut k_view,
                    &cos_view,
                    &sin_view,
                    batch,
                    seq,
                    config.num_kv_heads,
                    config.head_dim,
                    config.position,
                )
                .map_err(|e| format!("transformer_layer_gpu RoPE k error: {}", e))?;
        }

        // Permute to [batch, heads, seq, head_dim]
        let mut q_perm: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(q_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc q_perm failed: {}", e))?;
        let mut k_perm: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(kv_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc k_perm failed: {}", e))?;
        let mut v_perm: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(kv_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc v_perm failed: {}", e))?;

        {
            let q_in = q_linear.as_view();
            let mut q_out_view = q_perm.as_view_mut();
            elem_kernel
                .permute_qkv_view(
                    stream,
                    &q_in,
                    &mut q_out_view,
                    batch,
                    seq,
                    config.num_q_heads,
                    config.head_dim,
                )
                .map_err(|e| format!("transformer_layer_gpu permute q error: {}", e))?;
        }
        {
            let k_in = k_linear.as_view();
            let mut k_out_view = k_perm.as_view_mut();
            elem_kernel
                .permute_qkv_view(
                    stream,
                    &k_in,
                    &mut k_out_view,
                    batch,
                    seq,
                    config.num_kv_heads,
                    config.head_dim,
                )
                .map_err(|e| format!("transformer_layer_gpu permute k error: {}", e))?;
        }
        {
            let v_in = v_linear.as_view();
            let mut v_out_view = v_perm.as_view_mut();
            elem_kernel
                .permute_qkv_view(
                    stream,
                    &v_in,
                    &mut v_out_view,
                    batch,
                    seq,
                    config.num_kv_heads,
                    config.head_dim,
                )
                .map_err(|e| format!("transformer_layer_gpu permute v error: {}", e))?;
        }

        // KV cache update (GPU-to-GPU copy)
        if let Some(kv_cache) = kv_cache {
            if kv_cache.num_kv_heads != config.num_kv_heads || kv_cache.head_dim != config.head_dim {
                return Err("transformer_layer_gpu: KV cache shape mismatch".into());
            }
            if kv_cache.num_layers != 1 {
                return Err("transformer_layer_gpu: KV cache must be per-layer (num_layers == 1)".into());
            }
            let pos_end = config.position
                .checked_add(seq)
                .ok_or("transformer_layer_gpu: KV cache position overflow")?;
            if pos_end > kv_cache.max_len {
                return Err("transformer_layer_gpu: KV cache position exceeds max_len".into());
            }
            if kv_cache.k_cache.dtype != TensorDtype::F32 || kv_cache.v_cache.dtype != TensorDtype::F32 {
                return Err("transformer_layer_gpu: KV cache must be F32 tensors".into());
            }
            let GpuBuffer::Cuda(k_cache_arc) = &mut kv_cache.k_cache.buffer else {
                return Err("transformer_layer_gpu: KV cache k must be CUDA buffer".into());
            };
            let GpuBuffer::Cuda(v_cache_arc) = &mut kv_cache.v_cache.buffer else {
                return Err("transformer_layer_gpu: KV cache v must be CUDA buffer".into());
            };
            let k_cache_slice = Arc::get_mut(k_cache_arc)
                .ok_or("transformer_layer_gpu: KV cache k buffer is shared")?;
            let v_cache_slice = Arc::get_mut(v_cache_arc)
                .ok_or("transformer_layer_gpu: KV cache v buffer is shared")?;
            let k_cache_elems = kv_cache.k_cache.size_in_bytes / std::mem::size_of::<f32>();
            let v_cache_elems = kv_cache.v_cache.size_in_bytes / std::mem::size_of::<f32>();
            let mut k_cache_view = unsafe {
                k_cache_slice
                    .transmute_mut::<f32>(k_cache_elems)
                    .ok_or("transformer_layer_gpu: failed to transmute KV k cache")?
            };
            let mut v_cache_view = unsafe {
                v_cache_slice
                    .transmute_mut::<f32>(v_cache_elems)
                    .ok_or("transformer_layer_gpu: failed to transmute KV v cache")?
            };
            let k_src_view = k_perm.as_view();
            let v_src_view = v_perm.as_view();
            let copy_len = seq
                .checked_mul(config.head_dim)
                .ok_or("transformer_layer_gpu: KV copy len overflow")?;

            for b in 0..batch {
                for h in 0..config.num_kv_heads {
                    let src_start = (b * config.num_kv_heads + h)
                        .checked_mul(seq)
                        .and_then(|v| v.checked_mul(config.head_dim))
                        .ok_or("transformer_layer_gpu: KV src offset overflow")?;
                    let src_end = src_start + copy_len;
                    let src_k = k_src_view
                        .try_slice(src_start..src_end)
                        .ok_or("transformer_layer_gpu: KV k src slice OOB")?;
                    let src_v = v_src_view
                        .try_slice(src_start..src_end)
                        .ok_or("transformer_layer_gpu: KV v src slice OOB")?;

                    let dst_start = (b * config.num_kv_heads + h)
                        .checked_mul(kv_cache.max_len)
                        .and_then(|v| v.checked_add(config.position))
                        .and_then(|v| v.checked_mul(config.head_dim))
                        .ok_or("transformer_layer_gpu: KV dst offset overflow")?;
                    let dst_end = dst_start + copy_len;
                    let mut dst_k = k_cache_view
                        .try_slice_mut(dst_start..dst_end)
                        .ok_or("transformer_layer_gpu: KV k dst slice OOB")?;
                    let mut dst_v = v_cache_view
                        .try_slice_mut(dst_start..dst_end)
                        .ok_or("transformer_layer_gpu: KV v dst slice OOB")?;

                    stream
                        .memcpy_dtod(&src_k, &mut dst_k)
                        .map_err(|e| format!("transformer_layer_gpu KV k copy error: {}", e))?;
                    stream
                        .memcpy_dtod(&src_v, &mut dst_v)
                        .map_err(|e| format!("transformer_layer_gpu KV v copy error: {}", e))?;
                }
            }

            kv_cache.seq_len = pos_end;
        }

        // Expand K/V heads for GQA/MQA if needed
        let k_attn: &cudarc::driver::CudaSlice<f32>;
        let v_attn: &cudarc::driver::CudaSlice<f32>;
        let mut k_gqa: Option<cudarc::driver::CudaSlice<f32>> = None;
        let mut v_gqa: Option<cudarc::driver::CudaSlice<f32>> = None;
        if config.num_q_heads == config.num_kv_heads {
            k_attn = &k_perm;
            v_attn = &v_perm;
        } else {
            if config.num_q_heads % config.num_kv_heads != 0 {
                return Err("transformer_layer_gpu: num_q_heads must be multiple of num_kv_heads".into());
            }
            let heads_per_kv = config.num_q_heads / config.num_kv_heads;
            let gqa_elements = total_tokens
                .checked_mul(config.num_q_heads)
                .and_then(|v| v.checked_mul(config.head_dim))
                .ok_or("transformer_layer_gpu: gqa elements overflow")?;
            let mut k_buf: cudarc::driver::CudaSlice<f32> = stream
                .alloc_zeros(gqa_elements)
                .map_err(|e| format!("transformer_layer_gpu: alloc k_gqa failed: {}", e))?;
            let mut v_buf: cudarc::driver::CudaSlice<f32> = stream
                .alloc_zeros(gqa_elements)
                .map_err(|e| format!("transformer_layer_gpu: alloc v_gqa failed: {}", e))?;

            let k_src_view = k_perm.as_view();
            let v_src_view = v_perm.as_view();
            let copy_len = seq
                .checked_mul(config.head_dim)
                .ok_or("transformer_layer_gpu: gqa copy len overflow")?;

            for b in 0..batch {
                for kv_h in 0..config.num_kv_heads {
                    let src_start = (b * config.num_kv_heads + kv_h)
                        .checked_mul(seq)
                        .and_then(|v| v.checked_mul(config.head_dim))
                        .ok_or("transformer_layer_gpu: gqa src offset overflow")?;
                    let src_end = src_start + copy_len;
                    let src_k = k_src_view
                        .try_slice(src_start..src_end)
                        .ok_or("transformer_layer_gpu: gqa k src slice OOB")?;
                    let src_v = v_src_view
                        .try_slice(src_start..src_end)
                        .ok_or("transformer_layer_gpu: gqa v src slice OOB")?;

                    for rep in 0..heads_per_kv {
                        let q_h = kv_h * heads_per_kv + rep;
                        let dst_start = (b * config.num_q_heads + q_h)
                            .checked_mul(seq)
                            .and_then(|v| v.checked_mul(config.head_dim))
                            .ok_or("transformer_layer_gpu: gqa dst offset overflow")?;
                        let dst_end = dst_start + copy_len;
                        let mut dst_k = k_buf
                            .try_slice_mut(dst_start..dst_end)
                            .ok_or("transformer_layer_gpu: gqa k dst slice OOB")?;
                        let mut dst_v = v_buf
                            .try_slice_mut(dst_start..dst_end)
                            .ok_or("transformer_layer_gpu: gqa v dst slice OOB")?;

                        stream
                            .memcpy_dtod(&src_k, &mut dst_k)
                            .map_err(|e| format!("transformer_layer_gpu gqa k copy error: {}", e))?;
                        stream
                            .memcpy_dtod(&src_v, &mut dst_v)
                            .map_err(|e| format!("transformer_layer_gpu gqa v copy error: {}", e))?;
                    }
                }
            }

            k_gqa = Some(k_buf);
            v_gqa = Some(v_buf);
            k_attn = k_gqa.as_ref().ok_or("transformer_layer_gpu: gqa k missing")?;
            v_attn = v_gqa.as_ref().ok_or("transformer_layer_gpu: gqa v missing")?;
        }

        let scale = 1.0f32 / (config.head_dim as f32).sqrt();
        let attn_out = flash_kernel
            .forward_f32(
                stream,
                &q_perm,
                k_attn,
                v_attn,
                batch,
                config.num_q_heads,
                seq,
                config.head_dim,
                true,
                scale,
                config.position,
            )
            .map_err(|e| format!("transformer_layer_gpu flash attention error: {}", e))?;

        let mut attn_back: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(q_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc attn_back failed: {}", e))?;
        {
            let attn_in = attn_out.as_view();
            let mut attn_out_view = attn_back.as_view_mut();
            elem_kernel
                .permute_qkv_back_view(
                    stream,
                    &attn_in,
                    &mut attn_out_view,
                    batch,
                    seq,
                    config.num_q_heads,
                    config.head_dim,
                )
                .map_err(|e| format!("transformer_layer_gpu permute back error: {}", e))?;
        }

        let o_weight_elems = hidden_dim
            .checked_mul(q_out)
            .ok_or("transformer_layer_gpu: o_weight size overflow")?;
        let GpuBuffer::Cuda(o_weight_slice) = &layer_weights.o_weight.buffer else {
            return Err("transformer_layer_gpu: o_weight must be CUDA buffer".into());
        };
        let o_weight_view = unsafe {
            o_weight_slice
                .transmute::<f32>(o_weight_elems)
                .ok_or("transformer_layer_gpu: failed to transmute o_weight")?
        };

        let mut attn_proj: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(hidden_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc attn_proj failed: {}", e))?;
        {
            let attn_in = attn_back.as_view();
            let mut attn_out_view = attn_proj.as_view_mut();
            let params_o = LinearParams {
                in_features: q_out as u32,
                out_features: hidden_dim as u32,
                has_bias: 0,
                padding: 0,
            };
            linear_kernel.forward_view(
                stream,
                params_o,
                &attn_in,
                &o_weight_view,
                None,
                &mut attn_out_view,
                total_tokens,
            )?;
        }

        // Residual add: hidden + attn_proj
        let mut attn_residual: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(hidden_elements * std::mem::size_of::<f32>())
            .map_err(|e| format!("transformer_layer_gpu: alloc attn_residual failed: {}", e))?;
        {
            let hidden_in = hidden_view;
            let attn_in = attn_proj.as_view();
            let mut out_view = unsafe {
                attn_residual
                    .transmute_mut::<f32>(hidden_elements)
                    .ok_or("transformer_layer_gpu: failed to transmute attn_residual")?
            };
            elem_kernel
                .add_view(stream, &hidden_in, &attn_in, &mut out_view, hidden_elements)
                .map_err(|e| format!("transformer_layer_gpu residual add error: {}", e))?;
        }

        hidden.buffer = GpuBuffer::Cuda(Arc::new(attn_residual));
        hidden.dtype = TensorDtype::F32;
        hidden.size_in_bytes = hidden_elements * std::mem::size_of::<f32>();
        hidden.backend = BackendType::Cuda;

        // Post-attention RMSNorm
        let GpuBuffer::Cuda(post_norm_slice) = &layer_weights.post_attn_norm.buffer else {
            return Err("transformer_layer_gpu: post_attn_norm must be CUDA buffer".into());
        };
        let post_norm_view = unsafe {
            post_norm_slice
                .transmute::<f32>(hidden_dim)
                .ok_or("transformer_layer_gpu: failed to transmute post_attn_norm")?
        };

        let GpuBuffer::Cuda(hidden_slice) = &hidden.buffer else {
            return Err("transformer_layer_gpu: hidden must be CUDA buffer after residual".into());
        };
        let hidden_view = unsafe {
            hidden_slice
                .transmute::<f32>(hidden_elements)
                .ok_or("transformer_layer_gpu: failed to transmute hidden for post norm")?
        };

        let mut post_normed: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(hidden_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc post_normed failed: {}", e))?;
        {
            let mut post_normed_view = post_normed.as_view_mut();
            rms_kernel
                .forward_view(
                    stream,
                    &hidden_view,
                    &post_norm_view,
                    &mut post_normed_view,
                    total_tokens,
                    hidden_dim,
                    config.rms_norm_eps,
                )
                .map_err(|e| format!("transformer_layer_gpu post rms_norm error: {}", e))?;
        }

        // FFN: gate * silu(up) or up + activation
        let inter_size = config.intermediate_size;
        let inter_elements = total_tokens
            .checked_mul(inter_size)
            .ok_or("transformer_layer_gpu: inter size overflow")?;

        let mut ffn_intermediate: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(inter_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc ffn intermediate failed: {}", e))?;

        if let Some(gate_weight) = layer_weights.gate_weight.as_ref() {
            let GpuBuffer::Cuda(gate_slice) = &gate_weight.buffer else {
                return Err("transformer_layer_gpu: gate_weight must be CUDA buffer".into());
            };
            let gate_elems = inter_size
                .checked_mul(hidden_dim)
                .ok_or("transformer_layer_gpu: gate_weight size overflow")?;
            let gate_view = unsafe {
                gate_slice
                    .transmute::<f32>(gate_elems)
                    .ok_or("transformer_layer_gpu: failed to transmute gate_weight")?
            };

            let GpuBuffer::Cuda(up_slice) = &layer_weights.up_weight.buffer else {
                return Err("transformer_layer_gpu: up_weight must be CUDA buffer".into());
            };
            let up_elems = inter_size
                .checked_mul(hidden_dim)
                .ok_or("transformer_layer_gpu: up_weight size overflow")?;
            let up_view = unsafe {
                up_slice
                    .transmute::<f32>(up_elems)
                    .ok_or("transformer_layer_gpu: failed to transmute up_weight")?
            };

            let params_ffn = LinearParams {
                in_features: hidden_dim as u32,
                out_features: inter_size as u32,
                has_bias: 0,
                padding: 0,
            };
            let mut out_view = ffn_intermediate.as_view_mut();
            linear_kernel.fused_gate_up_silu_view(
                stream,
                params_ffn,
                &post_normed.as_view(),
                &gate_view,
                &up_view,
                &mut out_view,
                total_tokens,
            )?;
        } else {
            let GpuBuffer::Cuda(up_slice) = &layer_weights.up_weight.buffer else {
                return Err("transformer_layer_gpu: up_weight must be CUDA buffer".into());
            };
            let up_elems = inter_size
                .checked_mul(hidden_dim)
                .ok_or("transformer_layer_gpu: up_weight size overflow")?;
            let up_view = unsafe {
                up_slice
                    .transmute::<f32>(up_elems)
                    .ok_or("transformer_layer_gpu: failed to transmute up_weight")?
            };

            let params_up = LinearParams {
                in_features: hidden_dim as u32,
                out_features: inter_size as u32,
                has_bias: 0,
                padding: 0,
            };
            {
                let mut out_view = ffn_intermediate.as_view_mut();
                linear_kernel.forward_view(
                    stream,
                    params_up,
                    &post_normed.as_view(),
                    &up_view,
                    None,
                    &mut out_view,
                    total_tokens,
                )?;
            }

            if config.use_silu {
                silu_kernel
                    .forward_inplace(stream, &mut ffn_intermediate, inter_elements)
                    .map_err(|e| format!("transformer_layer_gpu silu error: {}", e))?;
            } else {
                return Err("transformer_layer_gpu: GELU activation not supported for CUDA".into());
            }
        }

        let GpuBuffer::Cuda(down_slice) = &layer_weights.down_weight.buffer else {
            return Err("transformer_layer_gpu: down_weight must be CUDA buffer".into());
        };
        let down_elems = hidden_dim
            .checked_mul(inter_size)
            .ok_or("transformer_layer_gpu: down_weight size overflow")?;
        let down_view = unsafe {
            down_slice
                .transmute::<f32>(down_elems)
                .ok_or("transformer_layer_gpu: failed to transmute down_weight")?
        };

        let mut ffn_out: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros(hidden_elements)
            .map_err(|e| format!("transformer_layer_gpu: alloc ffn_out failed: {}", e))?;
        {
            let mut out_view = ffn_out.as_view_mut();
            let params_down = LinearParams {
                in_features: inter_size as u32,
                out_features: hidden_dim as u32,
                has_bias: 0,
                padding: 0,
            };
            linear_kernel.forward_view(
                stream,
                params_down,
                &ffn_intermediate.as_view(),
                &down_view,
                None,
                &mut out_view,
                total_tokens,
            )?;
        }

        // Residual add: hidden + ffn_out
        let GpuBuffer::Cuda(hidden_slice) = &hidden.buffer else {
            return Err("transformer_layer_gpu: hidden must be CUDA buffer before ffn residual".into());
        };
        let hidden_view = unsafe {
            hidden_slice
                .transmute::<f32>(hidden_elements)
                .ok_or("transformer_layer_gpu: failed to transmute hidden for ffn residual")?
        };

        let mut ffn_residual: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(hidden_elements * std::mem::size_of::<f32>())
            .map_err(|e| format!("transformer_layer_gpu: alloc ffn_residual failed: {}", e))?;
        {
            let ffn_in = ffn_out.as_view();
            let mut out_view = unsafe {
                ffn_residual
                    .transmute_mut::<f32>(hidden_elements)
                    .ok_or("transformer_layer_gpu: failed to transmute ffn_residual")?
            };
            elem_kernel
                .add_view(stream, &hidden_view, &ffn_in, &mut out_view, hidden_elements)
                .map_err(|e| format!("transformer_layer_gpu ffn residual add error: {}", e))?;
        }

        hidden.buffer = GpuBuffer::Cuda(Arc::new(ffn_residual));
        hidden.dtype = TensorDtype::F32;
        hidden.size_in_bytes = hidden_elements * std::mem::size_of::<f32>();
        hidden.backend = BackendType::Cuda;

        Ok(())
    }

    fn rms_norm_gpu(
        &self,
        hidden: &mut GpuTensor,
        weight: &GpuTensor,
        eps: f32,
    ) -> Result<(), String> {
        // GPU-native RMSNorm using CUDA kernel (ARCH-GPU-001)
        // Pattern: All computation stays on GPU, no CPU roundtrip

        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let kernel = get_cuda_rmsnorm_kernel()
            .ok_or("CUDA RMSNorm kernel not available")?;

        // Validate shapes
        if hidden.shape.len() < 2 {
            return Err("rms_norm_gpu: hidden must be at least 2D (batch × hidden_dim)".into());
        }
        let hidden_dim = *hidden.shape.last().unwrap();
        let rows: usize = hidden.shape.iter().take(hidden.shape.len() - 1).product();
        let num_elements = rows * hidden_dim;

        // Extract CUDA slices and transmute to f32 views
        let GpuBuffer::Cuda(input_slice) = &hidden.buffer else {
            return Err("rms_norm_gpu: hidden must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(weight_slice) = &weight.buffer else {
            return Err("rms_norm_gpu: weight must be CUDA buffer".into());
        };

        // Create typed views from u8 slices (zero-copy reinterpretation)
        let input_view = unsafe {
            input_slice.transmute::<f32>(num_elements)
                .ok_or("rms_norm_gpu: failed to transmute input to f32 view")?
        };
        let weight_view = unsafe {
            weight_slice.transmute::<f32>(hidden_dim)
                .ok_or("rms_norm_gpu: failed to transmute weight to f32 view")?
        };

        // Allocate output buffer on GPU (same size as input)
        let output_bytes = num_elements * std::mem::size_of::<f32>();
        let mut output_slice: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(output_bytes)
            .map_err(|e| format!("rms_norm_gpu: failed to allocate output: {}", e))?;

        // Create mutable output view
        let mut output_view = unsafe {
            output_slice.transmute_mut::<f32>(num_elements)
                .ok_or("rms_norm_gpu: failed to transmute output to f32 view")?
        };

        // Execute kernel: all data stays on GPU
        kernel.forward_view(stream, &input_view, &weight_view, &mut output_view, rows, hidden_dim, eps)
            .map_err(|e| format!("rms_norm_gpu kernel error: {}", e))?;

        // Replace hidden's buffer with the output (swap, not copy)
        hidden.buffer = GpuBuffer::Cuda(Arc::new(output_slice));

        Ok(())
    }

    fn mean_pooling_gpu(
        &self,
        hidden: &GpuTensor,
        attention_mask: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let kernel = get_cuda_pooling_kernel()
            .ok_or("CUDA pooling kernel not available")?;

        if hidden.dtype != TensorDtype::F32 {
            return Err("mean_pooling_gpu: hidden must be F32 tensor".into());
        }
        if attention_mask.dtype != TensorDtype::F32 {
            return Err("mean_pooling_gpu: attention_mask must be F32 tensor".into());
        }
        if hidden.shape.len() < 2 {
            return Err("mean_pooling_gpu: hidden must be at least 2D".into());
        }

        let batch = hidden.shape[0];
        let seq = if hidden.shape.len() > 2 { hidden.shape[1] } else { 1 };
        let hidden_dim = *hidden.shape.last().unwrap();
        let hidden_elements: usize = hidden.shape.iter().product();
        let expected_hidden = batch
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(hidden_dim))
            .ok_or("mean_pooling_gpu: hidden size overflow")?;
        if hidden_elements != expected_hidden {
            return Err("mean_pooling_gpu: hidden shape mismatch".into());
        }

        let mask_elements: usize = attention_mask.shape.iter().product();
        let expected_mask = batch
            .checked_mul(seq)
            .ok_or("mean_pooling_gpu: mask size overflow")?;
        if mask_elements != expected_mask {
            return Err("mean_pooling_gpu: attention_mask shape mismatch".into());
        }

        let GpuBuffer::Cuda(hidden_slice) = &hidden.buffer else {
            return Err("mean_pooling_gpu: hidden must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(mask_slice) = &attention_mask.buffer else {
            return Err("mean_pooling_gpu: attention_mask must be CUDA buffer".into());
        };

        let hidden_view = unsafe {
            hidden_slice
                .transmute::<f32>(hidden_elements)
                .ok_or("mean_pooling_gpu: failed to transmute hidden")?
        };
        let mask_view = unsafe {
            mask_slice
                .transmute::<f32>(mask_elements)
                .ok_or("mean_pooling_gpu: failed to transmute mask")?
        };

        let output_elements = batch
            .checked_mul(hidden_dim)
            .ok_or("mean_pooling_gpu: output size overflow")?;
        let output_bytes = output_elements * std::mem::size_of::<f32>();
        let mut output_slice: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(output_bytes)
            .map_err(|e| format!("mean_pooling_gpu: alloc failed: {}", e))?;
        let mut output_view = unsafe {
            output_slice
                .transmute_mut::<f32>(output_elements)
                .ok_or("mean_pooling_gpu: failed to transmute output")?
        };

        kernel
            .mean_pooling_view(
                stream,
                &hidden_view,
                Some(&mask_view),
                &mut output_view,
                batch,
                seq,
                hidden_dim,
            )
            .map_err(|e| format!("mean_pooling_gpu kernel error: {}", e))?;

        output.buffer = GpuBuffer::Cuda(Arc::new(output_slice));
        output.shape = vec![batch, hidden_dim];
        output.dtype = TensorDtype::F32;
        output.size_in_bytes = output_bytes;
        output.backend = BackendType::Cuda;

        Ok(())
    }

    fn normalize_gpu(
        &self,
        embeddings: &mut GpuTensor,
    ) -> Result<(), String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let kernel = get_cuda_pooling_kernel()
            .ok_or("CUDA pooling kernel not available")?;

        if embeddings.dtype != TensorDtype::F32 {
            return Err("normalize_gpu: embeddings must be F32 tensor".into());
        }
        if embeddings.shape.is_empty() {
            return Ok(());
        }

        let hidden_dim = *embeddings.shape.last().unwrap();
        if hidden_dim == 0 {
            return Ok(());
        }

        let total_elements: usize = embeddings.shape.iter().product();
        let batch = total_elements / hidden_dim;

        let GpuBuffer::Cuda(emb_slice) = &embeddings.buffer else {
            return Err("normalize_gpu: embeddings must be CUDA buffer".into());
        };

        let mut out_slice: cudarc::driver::CudaSlice<u8> = stream
            .clone_dtod(emb_slice.as_ref())
            .map_err(|e| format!("normalize_gpu: dtod copy failed: {}", e))?;
        let mut emb_view = unsafe {
            out_slice
                .transmute_mut::<f32>(total_elements)
                .ok_or("normalize_gpu: failed to transmute embeddings")?
        };

        let eps = 1e-12f32;
        kernel
            .l2_normalize_view(stream, &mut emb_view, batch, hidden_dim, eps)
            .map_err(|e| format!("normalize_gpu kernel error: {}", e))?;

        embeddings.buffer = GpuBuffer::Cuda(Arc::new(out_slice));
        embeddings.dtype = TensorDtype::F32;
        embeddings.size_in_bytes = total_elements * std::mem::size_of::<f32>();
        embeddings.backend = BackendType::Cuda;

        Ok(())
    }

    fn cls_pooling_gpu(
        &self,
        hidden: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let kernel = get_cuda_pooling_kernel()
            .ok_or("CUDA pooling kernel not available")?;

        if hidden.dtype != TensorDtype::F32 {
            return Err("cls_pooling_gpu: hidden must be F32 tensor".into());
        }
        if hidden.shape.len() < 2 {
            return Err("cls_pooling_gpu: hidden must be at least 2D".into());
        }

        let batch = hidden.shape[0];
        let seq = if hidden.shape.len() > 2 { hidden.shape[1] } else { 1 };
        let hidden_dim = *hidden.shape.last().unwrap();
        let hidden_elements: usize = hidden.shape.iter().product();
        let expected_hidden = batch
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(hidden_dim))
            .ok_or("cls_pooling_gpu: hidden size overflow")?;
        if hidden_elements != expected_hidden {
            return Err("cls_pooling_gpu: hidden shape mismatch".into());
        }

        let GpuBuffer::Cuda(hidden_slice) = &hidden.buffer else {
            return Err("cls_pooling_gpu: hidden must be CUDA buffer".into());
        };
        let hidden_view = unsafe {
            hidden_slice
                .transmute::<f32>(hidden_elements)
                .ok_or("cls_pooling_gpu: failed to transmute hidden")?
        };

        let output_elements = batch
            .checked_mul(hidden_dim)
            .ok_or("cls_pooling_gpu: output size overflow")?;
        let output_bytes = output_elements * std::mem::size_of::<f32>();
        let mut output_slice: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(output_bytes)
            .map_err(|e| format!("cls_pooling_gpu: alloc failed: {}", e))?;
        let mut output_view = unsafe {
            output_slice
                .transmute_mut::<f32>(output_elements)
                .ok_or("cls_pooling_gpu: failed to transmute output")?
        };

        kernel
            .cls_extract_view(
                stream,
                &hidden_view,
                &mut output_view,
                batch,
                seq,
                hidden_dim,
                0,
            )
            .map_err(|e| format!("cls_pooling_gpu kernel error: {}", e))?;

        output.buffer = GpuBuffer::Cuda(Arc::new(output_slice));
        output.shape = vec![batch, hidden_dim];
        output.dtype = TensorDtype::F32;
        output.size_in_bytes = output_bytes;
        output.backend = BackendType::Cuda;

        Ok(())
    }

    fn classifier_gpu(
        &self,
        hidden: &GpuTensor,
        weight: &GpuTensor,
        bias: Option<&GpuTensor>,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;
        let kernel = get_cuda_linear_kernel()
            .ok_or("CUDA linear kernel not available")?;

        // hidden shape: [batch, hidden_dim] or [batch, seq_len, hidden_dim]
        // weight shape: [out_dim, hidden_dim]
        // output shape: [batch, out_dim] or [batch, seq_len, out_dim]
        if hidden.shape.is_empty() || weight.shape.len() != 2 {
            return Err("classifier_gpu: invalid tensor shapes".into());
        }

        let hidden_dim = *hidden.shape.last().unwrap();
        let out_dim = weight.shape[0];
        let weight_hidden_dim = weight.shape[1];

        if hidden_dim != weight_hidden_dim {
            return Err(format!(
                "classifier_gpu: hidden_dim mismatch: hidden={}, weight={}",
                hidden_dim, weight_hidden_dim
            ));
        }

        // Compute batch_size (product of all dimensions except last)
        let batch_size: usize = hidden.shape.iter().take(hidden.shape.len() - 1).product();
        let hidden_elements = batch_size * hidden_dim;
        let weight_elements = out_dim * hidden_dim;
        let output_elements = batch_size * out_dim;

        // Extract CUDA buffers
        let GpuBuffer::Cuda(hidden_slice) = &hidden.buffer else {
            return Err("classifier_gpu: hidden must be CUDA buffer".into());
        };
        let GpuBuffer::Cuda(weight_slice) = &weight.buffer else {
            return Err("classifier_gpu: weight must be CUDA buffer".into());
        };

        // Transmute to f32 views
        let hidden_view = unsafe {
            hidden_slice.transmute::<f32>(hidden_elements)
                .ok_or("classifier_gpu: failed to transmute hidden to f32 view")?
        };
        let weight_view = unsafe {
            weight_slice.transmute::<f32>(weight_elements)
                .ok_or("classifier_gpu: failed to transmute weight to f32 view")?
        };

        // Handle optional bias
        let bias_view = if let Some(bias_tensor) = bias {
            let GpuBuffer::Cuda(bias_slice) = &bias_tensor.buffer else {
                return Err("classifier_gpu: bias must be CUDA buffer".into());
            };
            let bias_elements = bias_tensor.shape.iter().product::<usize>();
            Some(unsafe {
                bias_slice.transmute::<f32>(bias_elements)
                    .ok_or("classifier_gpu: failed to transmute bias to f32 view")?
            })
        } else {
            None
        };

        // Allocate output buffer on GPU
        let output_bytes = output_elements * std::mem::size_of::<f32>();
        let mut output_slice: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros(output_bytes)
            .map_err(|e| format!("classifier_gpu: failed to allocate output: {}", e))?;

        let mut output_view = unsafe {
            output_slice.transmute_mut::<f32>(output_elements)
                .ok_or("classifier_gpu: failed to transmute output to f32 view")?
        };

        // Setup linear params
        let params = LinearParams {
            in_features: hidden_dim as u32,
            out_features: out_dim as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
            padding: 0,
        };

        // Call linear kernel
        kernel.forward_view(
            stream,
            params,
            &hidden_view,
            &weight_view,
            bias_view.as_ref(),
            &mut output_view,
            batch_size,
        ).map_err(|e| format!("classifier_gpu kernel error: {}", e))?;

        // Update output tensor buffer
        output.buffer = GpuBuffer::Cuda(Arc::new(output_slice));
        output.shape = if hidden.shape.len() > 1 {
            let mut shape = hidden.shape[..hidden.shape.len()-1].to_vec();
            shape.push(out_dim);
            shape
        } else {
            vec![out_dim]
        };
        output.dtype = TensorDtype::F32;

        Ok(())
    }

    fn lm_head_gpu(
        &self,
        hidden: &GpuTensor,
        weight: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), String> {
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
        let stream = get_cuda_stream().ok_or("CUDA stream not available")?;

        // Helper to upload f32 slice to GPU
        fn upload_f32(stream: &Arc<CudaStream>, data: &[f32], shape: Vec<usize>) -> Result<GpuTensor, String> {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            let buf = stream.clone_htod(bytes).map_err(|e| e.to_string())?;
            Ok(GpuTensor {
                buffer: GpuBuffer::Cuda(Arc::new(buf)),
                shape,
                dtype: TensorDtype::F32,
                size_in_bytes: bytes.len(),
                backend: BackendType::Cuda,
            })
        }

        let hidden_dim = config.hidden_size;
        let vocab_size = embed_weight.len() / hidden_dim;

        let embedding = upload_f32(&stream, embed_weight, vec![vocab_size, hidden_dim])?;
        let final_norm_gpu = upload_f32(&stream, final_norm, vec![hidden_dim])?;

        let num_classes = classifier_weight.len() / hidden_dim;
        let classifier_weight_gpu = upload_f32(&stream, classifier_weight, vec![num_classes, hidden_dim])?;
        let classifier_bias_gpu = classifier_bias
            .map(|b| upload_f32(&stream, b, vec![num_classes]))
            .transpose()?;

        // Convert layers
        let gpu_layers: Result<Vec<TransformerLayerWeightsGpu>, String> = layers.iter().map(|layer| {
            Ok(TransformerLayerWeightsGpu {
                input_norm: upload_f32(&stream, layer.input_norm, vec![hidden_dim])?,
                q_weight: upload_f32(&stream, layer.q_weight, vec![layer.q_weight.len()])?,
                k_weight: upload_f32(&stream, layer.k_weight, vec![layer.k_weight.len()])?,
                v_weight: upload_f32(&stream, layer.v_weight, vec![layer.v_weight.len()])?,
                o_weight: upload_f32(&stream, layer.o_weight, vec![layer.o_weight.len()])?,
                post_attn_norm: upload_f32(&stream, layer.post_attn_norm, vec![hidden_dim])?,
                gate_weight: layer.gate_weight.map(|g| upload_f32(&stream, g, vec![g.len()])).transpose()?,
                up_weight: upload_f32(&stream, layer.up_weight, vec![layer.up_weight.len()])?,
                down_weight: upload_f32(&stream, layer.down_weight, vec![layer.down_weight.len()])?,
                cos_cache: None,
                sin_cache: None,
            })
        }).collect();

        Ok(RerankerModelWeightsGpu {
            embedding,
            layers: gpu_layers?,
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
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;

        let expected_tokens = config
            .batch_size
            .checked_mul(config.seq_len)
            .ok_or("CUDA rerank_forward_gpu_pure: token size overflow")?;
        if tokens.len() != expected_tokens {
            return Err(format!(
                "CUDA rerank_forward_gpu_pure: token length mismatch (got {}, expected {})",
                tokens.len(),
                expected_tokens
            ));
        }

        // Upload tokens to GPU (U32)
        let token_bytes: &[u8] = bytemuck::cast_slice(tokens);
        let token_buf = stream
            .clone_htod(token_bytes)
            .map_err(|e| format!("CUDA token upload failed: {}", e))?;
        let tokens_gpu = GpuTensor::new(
            GpuBuffer::Cuda(Arc::new(token_buf)),
            vec![config.batch_size, config.seq_len],
            TensorDtype::U32,
            BackendType::Cuda,
        );

        // Embedding lookup
        let mut hidden = GpuTensor {
            buffer: GpuBuffer::Cpu(Arc::new(Vec::new())),
            shape: Vec::new(),
            dtype: TensorDtype::F32,
            size_in_bytes: 0,
            backend: BackendType::Cuda,
        };
        self.embedding_lookup_gpu(&tokens_gpu, &weights.embedding, &mut hidden)?;

        // Prepare identity RoPE cache if missing (cos=1, sin=0) to keep Q/K unchanged
        let mut identity_rope: Option<(GpuTensor, GpuTensor)> = None;
            if config.head_dim % 2 != 0 {
                return Err("CUDA rerank_forward_gpu_pure: head_dim must be even for RoPE".into());
            }
        if weights.layers.iter().any(|layer| layer.cos_cache.is_none() || layer.sin_cache.is_none()) {
            let rope_len = config
                .seq_len
                .checked_mul(config.head_dim / 2)
                .ok_or("CUDA rerank_forward_gpu_pure: RoPE cache size overflow")?;
            let cos = vec![1.0f32; rope_len];
            let sin = vec![0.0f32; rope_len];
            let cos_bytes: &[u8] = bytemuck::cast_slice(&cos);
            let sin_bytes: &[u8] = bytemuck::cast_slice(&sin);
            let cos_buf = stream
                .clone_htod(cos_bytes)
                .map_err(|e| format!("CUDA RoPE cos upload failed: {}", e))?;
            let sin_buf = stream
                .clone_htod(sin_bytes)
                .map_err(|e| format!("CUDA RoPE sin upload failed: {}", e))?;
            let cos_gpu = GpuTensor {
                buffer: GpuBuffer::Cuda(Arc::new(cos_buf)),
                shape: vec![rope_len],
                dtype: TensorDtype::F32,
                size_in_bytes: cos_bytes.len(),
                backend: BackendType::Cuda,
            };
            let sin_gpu = GpuTensor {
                buffer: GpuBuffer::Cuda(Arc::new(sin_buf)),
                shape: vec![rope_len],
                dtype: TensorDtype::F32,
                size_in_bytes: sin_bytes.len(),
                backend: BackendType::Cuda,
            };
            identity_rope = Some((cos_gpu, sin_gpu));
        }

        let use_silu = matches!(config.activation, crate::kernel_types::Activation::SiLU);
        let layer_cfg = TransformerLayerConfigGpu {
            batch_size: config.batch_size,
            seq_len: config.seq_len,
            hidden_size: config.hidden_size,
            num_q_heads: config.num_q_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            rms_norm_eps: config.rms_norm_eps,
            position: 0,
            use_silu,
        };

        for layer in weights.layers.iter() {
            let mut layer_weights = layer.clone();
            if layer_weights.cos_cache.is_none() || layer_weights.sin_cache.is_none() {
                if let Some((cos_gpu, sin_gpu)) = identity_rope.as_ref() {
                    layer_weights.cos_cache = Some(cos_gpu.clone());
                    layer_weights.sin_cache = Some(sin_gpu.clone());
                }
            }
            self.transformer_layer_gpu(&mut hidden, &layer_weights, None, &layer_cfg)?;
        }

        // Final RMS norm
        self.rms_norm_gpu(&mut hidden, &weights.final_norm, config.rms_norm_eps)?;

        // Mean pooling (no attention mask provided, so use all-ones mask)
        let mask = vec![1.0f32; expected_tokens];
        let mask_bytes: &[u8] = bytemuck::cast_slice(&mask);
        let mask_buf = stream
            .clone_htod(mask_bytes)
            .map_err(|e| format!("CUDA mask upload failed: {}", e))?;
        let mask_gpu = GpuTensor {
            buffer: GpuBuffer::Cuda(Arc::new(mask_buf)),
            shape: vec![config.batch_size, config.seq_len],
            dtype: TensorDtype::F32,
            size_in_bytes: mask_bytes.len(),
            backend: BackendType::Cuda,
        };

        let mut pooled = GpuTensor {
            buffer: GpuBuffer::Cpu(Arc::new(Vec::new())),
            shape: Vec::new(),
            dtype: TensorDtype::F32,
            size_in_bytes: 0,
            backend: BackendType::Cuda,
        };
        self.mean_pooling_gpu(&hidden, &mask_gpu, &mut pooled)?;

        // Classifier head
        let mut scores_gpu = GpuTensor {
            buffer: GpuBuffer::Cpu(Arc::new(Vec::new())),
            shape: Vec::new(),
            dtype: TensorDtype::F32,
            size_in_bytes: 0,
            backend: BackendType::Cuda,
        };
        self.classifier_gpu(
            &pooled,
            &weights.classifier_weight,
            weights.classifier_bias.as_ref(),
            &mut scores_gpu,
        )?;

        // Download scores to host
        match &scores_gpu.buffer {
            GpuBuffer::Cuda(cuda_slice) => {
                let bytes: Vec<u8> = stream
                    .clone_dtoh(cuda_slice.as_ref())
                    .map_err(|e| format!("CUDA scores download failed: {}", e))?;
                Ok(bytemuck::cast_slice(&bytes).to_vec())
            }
            GpuBuffer::Cpu(bytes) => Ok(bytemuck::cast_slice(bytes.as_slice()).to_vec()),
            #[allow(unreachable_patterns)]
            _ => Err("Unexpected GPU buffer type for CUDA backend".into()),
        }
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
        let stream = get_cuda_stream().ok_or("CUDA stream not available")?;

        fn upload_f32(stream: &Arc<CudaStream>, data: &[f32], shape: Vec<usize>) -> Result<GpuTensor, String> {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            let buf = stream.clone_htod(bytes).map_err(|e| e.to_string())?;
            Ok(GpuTensor {
                buffer: GpuBuffer::Cuda(Arc::new(buf)),
                shape,
                dtype: TensorDtype::F32,
                size_in_bytes: bytes.len(),
                backend: BackendType::Cuda,
            })
        }

        let hidden_dim = config.hidden_size;
        let vocab_size = embed_weight.len() / hidden_dim;

        let embedding = upload_f32(&stream, embed_weight, vec![vocab_size, hidden_dim])?;
        let final_norm_gpu = upload_f32(&stream, final_norm, vec![hidden_dim])?;
        let lm_head_gpu = upload_f32(&stream, lm_head, vec![vocab_size, hidden_dim])?;
        let cos_cache_gpu = upload_f32(&stream, cos_cache, vec![cos_cache.len()])?;
        let sin_cache_gpu = upload_f32(&stream, sin_cache, vec![sin_cache.len()])?;

        let gpu_layers: Result<Vec<TransformerLayerWeightsGpu>, String> = layers.iter().map(|layer| {
            Ok(TransformerLayerWeightsGpu {
                input_norm: upload_f32(&stream, layer.input_norm, vec![hidden_dim])?,
                q_weight: upload_f32(&stream, layer.q_weight, vec![layer.q_weight.len()])?,
                k_weight: upload_f32(&stream, layer.k_weight, vec![layer.k_weight.len()])?,
                v_weight: upload_f32(&stream, layer.v_weight, vec![layer.v_weight.len()])?,
                o_weight: upload_f32(&stream, layer.o_weight, vec![layer.o_weight.len()])?,
                post_attn_norm: upload_f32(&stream, layer.post_attn_norm, vec![hidden_dim])?,
                gate_weight: layer.gate_weight.map(|g| upload_f32(&stream, g, vec![g.len()])).transpose()?,
                up_weight: upload_f32(&stream, layer.up_weight, vec![layer.up_weight.len()])?,
                down_weight: upload_f32(&stream, layer.down_weight, vec![layer.down_weight.len()])?,
                cos_cache: Some(cos_cache_gpu.clone()),
                sin_cache: Some(sin_cache_gpu.clone()),
            })
        }).collect();

        Ok(GeneratorModelWeightsGpu {
            embedding,
            layers: gpu_layers?,
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
        let stream = get_cuda_stream().ok_or("CUDA stream not available")?;

        let cache_size = num_layers * batch_size * num_kv_heads * max_len * head_dim;
        let size_in_bytes = cache_size * std::mem::size_of::<f32>();
        let zeros_bytes = vec![0u8; size_in_bytes];

        let k_buf = stream.clone_htod(&zeros_bytes).map_err(|e| e.to_string())?;
        let v_buf = stream.clone_htod(&zeros_bytes).map_err(|e| e.to_string())?;

        let k_cache = GpuTensor {
            buffer: GpuBuffer::Cuda(Arc::new(k_buf)),
            shape: vec![num_layers, batch_size, num_kv_heads, max_len, head_dim],
            dtype: TensorDtype::F32,
            size_in_bytes,
            backend: BackendType::Cuda,
        };
        let v_cache = GpuTensor {
            buffer: GpuBuffer::Cuda(Arc::new(v_buf)),
            shape: vec![num_layers, batch_size, num_kv_heads, max_len, head_dim],
            dtype: TensorDtype::F32,
            size_in_bytes,
            backend: BackendType::Cuda,
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
    ) -> Result<Vec<f32>, String> {
        let stream = get_cuda_stream()
            .ok_or("CUDA stream not available")?;

        let expected_tokens = config
            .batch_size
            .checked_mul(config.seq_len)
            .ok_or("CUDA generator_forward_gpu_pure: token size overflow")?;
        if tokens.len() != expected_tokens {
            return Err(format!(
                "CUDA generator_forward_gpu_pure: token length mismatch (got {}, expected {})",
                tokens.len(),
                expected_tokens
            ));
        }

        if kv_cache.num_layers != weights.layers.len() {
            return Err(format!(
                "CUDA generator_forward_gpu_pure: KV cache layers mismatch (cache={}, weights={})",
                kv_cache.num_layers,
                weights.layers.len()
            ));
        }
        if kv_cache.num_kv_heads != config.num_kv_heads || kv_cache.head_dim != config.head_dim {
            return Err("CUDA generator_forward_gpu_pure: KV cache shape mismatch".into());
        }

        let layer_elems = config
            .batch_size
            .checked_mul(config.num_kv_heads)
            .and_then(|v| v.checked_mul(kv_cache.max_len))
            .and_then(|v| v.checked_mul(config.head_dim))
            .ok_or("CUDA generator_forward_gpu_pure: KV cache size overflow")?;
        let expected_cache_elems = layer_elems
            .checked_mul(kv_cache.num_layers)
            .ok_or("CUDA generator_forward_gpu_pure: KV cache total size overflow")?;

        let cache_elems = kv_cache.k_cache.size_in_bytes / std::mem::size_of::<f32>();
        if cache_elems != expected_cache_elems {
            return Err("CUDA generator_forward_gpu_pure: KV cache buffer size mismatch".into());
        }

        // Upload tokens to GPU (U32)
        let token_bytes: &[u8] = bytemuck::cast_slice(tokens);
        let token_buf = stream
            .clone_htod(token_bytes)
            .map_err(|e| format!("CUDA token upload failed: {}", e))?;
        let tokens_gpu = GpuTensor::new(
            GpuBuffer::Cuda(Arc::new(token_buf)),
            vec![config.batch_size, config.seq_len],
            TensorDtype::U32,
            BackendType::Cuda,
        );

        // Embedding lookup
        let mut hidden = GpuTensor {
            buffer: GpuBuffer::Cpu(Arc::new(Vec::new())),
            shape: Vec::new(),
            dtype: TensorDtype::F32,
            size_in_bytes: 0,
            backend: BackendType::Cuda,
        };
        self.embedding_lookup_gpu(&tokens_gpu, &weights.embedding, &mut hidden)?;

        let use_silu = matches!(config.activation, crate::kernel_types::Activation::SiLU);
        let layer_cfg = TransformerLayerConfigGpu {
            batch_size: config.batch_size,
            seq_len: config.seq_len,
            hidden_size: config.hidden_size,
            num_q_heads: config.num_q_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            rms_norm_eps: config.rms_norm_eps,
            position: kv_cache.seq_len,
            use_silu,
        };

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            // Extract per-layer KV cache into a temporary GPU buffer
            let (layer_k_buf, layer_v_buf) = {
                let GpuBuffer::Cuda(k_cache_slice) = &kv_cache.k_cache.buffer else {
                    return Err("CUDA generator_forward_gpu_pure: KV cache k must be CUDA buffer".into());
                };
                let GpuBuffer::Cuda(v_cache_slice) = &kv_cache.v_cache.buffer else {
                    return Err("CUDA generator_forward_gpu_pure: KV cache v must be CUDA buffer".into());
                };
                let total_elems = expected_cache_elems;
                let k_view = unsafe {
                    k_cache_slice
                        .transmute::<f32>(total_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute KV k cache")?
                };
                let v_view = unsafe {
                    v_cache_slice
                        .transmute::<f32>(total_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute KV v cache")?
                };
                let start = layer_idx
                    .checked_mul(layer_elems)
                    .ok_or("CUDA generator_forward_gpu_pure: KV layer offset overflow")?;
                let end = start + layer_elems;
                let k_src = k_view
                    .try_slice(start..end)
                    .ok_or("CUDA generator_forward_gpu_pure: KV k slice OOB")?;
                let v_src = v_view
                    .try_slice(start..end)
                    .ok_or("CUDA generator_forward_gpu_pure: KV v slice OOB")?;

                let mut k_buf: cudarc::driver::CudaSlice<u8> = stream
                    .alloc_zeros(layer_elems * std::mem::size_of::<f32>())
                    .map_err(|e| format!("CUDA generator_forward_gpu_pure: KV k alloc failed: {}", e))?;
                let mut v_buf: cudarc::driver::CudaSlice<u8> = stream
                    .alloc_zeros(layer_elems * std::mem::size_of::<f32>())
                    .map_err(|e| format!("CUDA generator_forward_gpu_pure: KV v alloc failed: {}", e))?;
                let mut k_dst = unsafe {
                    k_buf
                        .transmute_mut::<f32>(layer_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute KV k layer")?
                };
                let mut v_dst = unsafe {
                    v_buf
                        .transmute_mut::<f32>(layer_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute KV v layer")?
                };
                stream
                    .memcpy_dtod(&k_src, &mut k_dst)
                    .map_err(|e| format!("CUDA generator_forward_gpu_pure: KV k dtod failed: {}", e))?;
                stream
                    .memcpy_dtod(&v_src, &mut v_dst)
                    .map_err(|e| format!("CUDA generator_forward_gpu_pure: KV v dtod failed: {}", e))?;
                (k_buf, v_buf)
            };

            let mut layer_kv = KVCacheGpu {
                k_cache: GpuTensor {
                    buffer: GpuBuffer::Cuda(Arc::new(layer_k_buf)),
                    shape: vec![1, config.batch_size, config.num_kv_heads, kv_cache.max_len, config.head_dim],
                    dtype: TensorDtype::F32,
                    size_in_bytes: layer_elems * std::mem::size_of::<f32>(),
                    backend: BackendType::Cuda,
                },
                v_cache: GpuTensor {
                    buffer: GpuBuffer::Cuda(Arc::new(layer_v_buf)),
                    shape: vec![1, config.batch_size, config.num_kv_heads, kv_cache.max_len, config.head_dim],
                    dtype: TensorDtype::F32,
                    size_in_bytes: layer_elems * std::mem::size_of::<f32>(),
                    backend: BackendType::Cuda,
                },
                seq_len: kv_cache.seq_len,
                max_len: kv_cache.max_len,
                num_layers: 1,
                num_kv_heads: kv_cache.num_kv_heads,
                head_dim: kv_cache.head_dim,
            };

            self.transformer_layer_gpu(&mut hidden, layer, Some(&mut layer_kv), &layer_cfg)?;

            // Copy per-layer KV cache back into the global cache
            {
                let GpuBuffer::Cuda(k_cache_arc) = &mut kv_cache.k_cache.buffer else {
                    return Err("CUDA generator_forward_gpu_pure: KV cache k must be CUDA buffer".into());
                };
                let GpuBuffer::Cuda(v_cache_arc) = &mut kv_cache.v_cache.buffer else {
                    return Err("CUDA generator_forward_gpu_pure: KV cache v must be CUDA buffer".into());
                };
                let k_cache_slice = Arc::get_mut(k_cache_arc)
                    .ok_or("CUDA generator_forward_gpu_pure: KV cache k buffer is shared")?;
                let v_cache_slice = Arc::get_mut(v_cache_arc)
                    .ok_or("CUDA generator_forward_gpu_pure: KV cache v buffer is shared")?;
                let mut k_view = unsafe {
                    k_cache_slice
                        .transmute_mut::<f32>(expected_cache_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute KV k cache")?
                };
                let mut v_view = unsafe {
                    v_cache_slice
                        .transmute_mut::<f32>(expected_cache_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute KV v cache")?
                };
                let start = layer_idx
                    .checked_mul(layer_elems)
                    .ok_or("CUDA generator_forward_gpu_pure: KV layer offset overflow")?;
                let end = start + layer_elems;
                let mut k_dst = k_view
                    .try_slice_mut(start..end)
                    .ok_or("CUDA generator_forward_gpu_pure: KV k dst slice OOB")?;
                let mut v_dst = v_view
                    .try_slice_mut(start..end)
                    .ok_or("CUDA generator_forward_gpu_pure: KV v dst slice OOB")?;

                let GpuBuffer::Cuda(layer_k_buf) = &layer_kv.k_cache.buffer else {
                    return Err("CUDA generator_forward_gpu_pure: layer KV k must be CUDA buffer".into());
                };
                let GpuBuffer::Cuda(layer_v_buf) = &layer_kv.v_cache.buffer else {
                    return Err("CUDA generator_forward_gpu_pure: layer KV v must be CUDA buffer".into());
                };
                let k_src = unsafe {
                    layer_k_buf
                        .transmute::<f32>(layer_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute layer KV k")?
                };
                let v_src = unsafe {
                    layer_v_buf
                        .transmute::<f32>(layer_elems)
                        .ok_or("CUDA generator_forward_gpu_pure: failed to transmute layer KV v")?
                };

                stream
                    .memcpy_dtod(&k_src, &mut k_dst)
                    .map_err(|e| format!("CUDA generator_forward_gpu_pure: KV k writeback failed: {}", e))?;
                stream
                    .memcpy_dtod(&v_src, &mut v_dst)
                    .map_err(|e| format!("CUDA generator_forward_gpu_pure: KV v writeback failed: {}", e))?;
            }
        }

        let pos_end = kv_cache
            .seq_len
            .checked_add(config.seq_len)
            .ok_or("CUDA generator_forward_gpu_pure: KV cache position overflow")?;
        kv_cache.seq_len = pos_end;

        // Final RMS norm
        self.rms_norm_gpu(&mut hidden, &weights.final_norm, config.rms_norm_eps)?;

        // LM head
        let mut logits_gpu = GpuTensor {
            buffer: GpuBuffer::Cpu(Arc::new(Vec::new())),
            shape: Vec::new(),
            dtype: TensorDtype::F32,
            size_in_bytes: 0,
            backend: BackendType::Cuda,
        };
        self.lm_head_gpu(&hidden, &weights.lm_head, &mut logits_gpu)?;

        // Download logits to host
        match &logits_gpu.buffer {
            GpuBuffer::Cuda(cuda_slice) => {
                let bytes: Vec<u8> = stream
                    .clone_dtoh(cuda_slice.as_ref())
                    .map_err(|e| format!("CUDA logits download failed: {}", e))?;
                Ok(bytemuck::cast_slice(&bytes).to_vec())
            }
            GpuBuffer::Cpu(bytes) => Ok(bytemuck::cast_slice(bytes.as_slice()).to_vec()),
            #[allow(unreachable_patterns)]
            _ => Err("Unexpected GPU buffer type for CUDA backend".into()),
        }
    }
}

// =========================================================================
// CUDA Kernel Helper Functions (Internal)
// NOTE: Real GPU kernels need to be implemented here.
// These should use cuBLAS, cuDNN, or custom CUDA PTX kernels.
// =========================================================================

// TODO: When implementing real CUDA kernels, add functions like:
//
// fn cuda_gemm(
//     cublas_handle: &CublasHandle,
//     a: &CudaSlice<f32>, // GPU pointer
//     b: &CudaSlice<f32>, // GPU pointer
//     c: &mut CudaSlice<f32>, // GPU pointer
//     m: usize, n: usize, k: usize,
// ) -> Result<(), String>
//
// fn cuda_rms_norm(
//     stream: &CudaStream,
//     input: &CudaSlice<f32>,  // GPU pointer
//     weight: &CudaSlice<f32>, // GPU pointer
//     output: &mut CudaSlice<f32>, // GPU pointer
//     hidden_dim: usize,
//     eps: f32,
// ) -> Result<(), String>
//
// These functions MUST:
// - Take GPU pointers directly (no host data)
// - Launch CUDA kernels
// - Keep all intermediate results on GPU
// - Never call clone_dtoh/clone_htod internally
