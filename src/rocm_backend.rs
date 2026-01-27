use crate::backend_match::{
    apply_f32_binary_out, apply_f32_inplace_weight, apply_f32_unary_inplace, apply_f32_unary_out,
    match_float1, match_float1_mut, match_float1_mut_weight, match_float1_out,
    match_float2_out, match_float2_out2, match_float3_out,
};
use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
use crate::kernel_types::{
    FlashAttentionConfig, KernelFloat, LinearParams, MatmulConfig, PagedAttentionConfig,
    SoftmaxConfig,
};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};
use crate::ops::rope::RoPEConfig;
use crate::ops::sampling::{SamplingConfig, TopKResult};
use crate::runtime_detection::BackendType;

#[cfg(target_os = "linux")]
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
use crate::kernel_types::FloatType;

#[cfg(target_os = "linux")]
use crate::hip_kernels::{
    find_gpu_agents, is_hsa_available, GpuAgent, HsaBuffer, HsaFlashAttentionKernel,
    HsaPagedAttentionKernel, HsaQueueWrapper,
    HsaSoftmaxKernel, HsaSamplingKernel, HsaMoeRouteKernel, HsaRoPEKernel, HsaQuantizedKernel,
    HsaRmsNormKernel, HsaLinearKernel, HsaSiluKernel,
};

#[cfg(target_os = "linux")]
static HSA_FLASH_KERNEL: OnceLock<Option<HsaFlashAttentionKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_PAGED_KERNEL: OnceLock<Option<HsaPagedAttentionKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_QUEUE: OnceLock<Option<HsaQueueWrapper>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_GPU_AGENT: OnceLock<Option<GpuAgent>> = OnceLock::new();

// Core operator kernels
#[cfg(target_os = "linux")]
static HSA_SOFTMAX_KERNEL: OnceLock<Option<HsaSoftmaxKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_SAMPLING_KERNEL: OnceLock<Option<HsaSamplingKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_MOE_KERNEL: OnceLock<Option<HsaMoeRouteKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_ROPE_KERNEL: OnceLock<Option<HsaRoPEKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_QUANTIZED_KERNEL: OnceLock<Option<HsaQuantizedKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_RMS_NORM_KERNEL: OnceLock<Option<HsaRmsNormKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_LINEAR_KERNEL: OnceLock<Option<HsaLinearKernel>> = OnceLock::new();
#[cfg(target_os = "linux")]
static HSA_SILU_KERNEL: OnceLock<Option<HsaSiluKernel>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn get_hsa_flash_kernel() -> Option<&'static HsaFlashAttentionKernel> {
    HSA_FLASH_KERNEL
        .get_or_init(|| {
            if !is_hsa_available() {
                return None;
            }
            match HsaFlashAttentionKernel::new(0) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize HSA flash attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_paged_kernel() -> Option<&'static HsaPagedAttentionKernel> {
    HSA_PAGED_KERNEL
        .get_or_init(|| {
            if !is_hsa_available() {
                return None;
            }
            match HsaPagedAttentionKernel::new(0) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize HSA paged attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_queue() -> Option<&'static HsaQueueWrapper> {
    HSA_QUEUE
        .get_or_init(|| {
            if !is_hsa_available() {
                return None;
            }
            let agents = match find_gpu_agents() {
                Ok(agents) => agents,
                Err(e) => {
                    log::warn!("Failed to find GPU agents: {}", e);
                    return None;
                }
            };
            if agents.is_empty() {
                return None;
            }
            match HsaQueueWrapper::new(&agents[0]) {
                Ok(queue) => Some(queue),
                Err(e) => {
                    log::warn!("Failed to create HSA queue: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_gpu_agent() -> Option<&'static GpuAgent> {
    HSA_GPU_AGENT
        .get_or_init(|| {
            if !is_hsa_available() {
                return None;
            }
            let agents = match find_gpu_agents() {
                Ok(agents) => agents,
                Err(e) => {
                    log::warn!("Failed to find GPU agents: {}", e);
                    return None;
                }
            };
            if agents.is_empty() {
                return None;
            }
            Some(agents[0].clone())
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_softmax_kernel() -> Option<&'static HsaSoftmaxKernel> {
    HSA_SOFTMAX_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaSoftmaxKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA softmax kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_sampling_kernel() -> Option<&'static HsaSamplingKernel> {
    HSA_SAMPLING_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaSamplingKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA sampling kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_moe_kernel() -> Option<&'static HsaMoeRouteKernel> {
    HSA_MOE_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaMoeRouteKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA MoE routing kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_rope_kernel() -> Option<&'static HsaRoPEKernel> {
    HSA_ROPE_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaRoPEKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA RoPE kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_quantized_kernel() -> Option<&'static HsaQuantizedKernel> {
    HSA_QUANTIZED_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaQuantizedKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA quantized kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_rms_norm_kernel() -> Option<&'static HsaRmsNormKernel> {
    HSA_RMS_NORM_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaRmsNormKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA RMSNorm kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_linear_kernel() -> Option<&'static HsaLinearKernel> {
    HSA_LINEAR_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaLinearKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA linear kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_silu_kernel() -> Option<&'static HsaSiluKernel> {
    HSA_SILU_KERNEL
        .get_or_init(|| {
            let agent = get_hsa_gpu_agent()?;
            match HsaSiluKernel::new(agent) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::debug!("Failed to initialize HSA SiLU kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

/// ROCm/HSA flash attention dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
#[cfg(target_os = "linux")]
fn rocm_flash_attention<T: KernelFloat>(
    kernel: &HsaFlashAttentionKernel,
    queue: &HsaQueueWrapper,
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: &FlashAttentionConfig,
) -> bool {
    let agent = kernel.agent();

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let seq_len = config.seq_len_q;

    if T::TYPE_ID == FloatType::F16 {
        let q_bits: &[u16] = unsafe {
            std::slice::from_raw_parts(q.as_ptr() as *const u16, q.len())
        };
        let k_bits: &[u16] = unsafe {
            std::slice::from_raw_parts(k.as_ptr() as *const u16, k.len())
        };
        let v_bits: &[u16] = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u16, v.len())
        };
        let output_bits: &mut [u16] = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u16, output.len())
        };

        let q_buf = match HsaBuffer::from_slice(agent, q_bits) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 Q buffer: {}", e);
                return false;
            }
        };
        let k_buf = match HsaBuffer::from_slice(agent, k_bits) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 K buffer: {}", e);
                return false;
            }
        };
        let v_buf = match HsaBuffer::from_slice(agent, v_bits) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 V buffer: {}", e);
                return false;
            }
        };

        let result = kernel.forward_f16(
            queue,
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
            Ok(out_buf) => match out_buf.to_vec() {
                Ok(out_data) => {
                    let copy_len = output_bits.len().min(out_data.len());
                    output_bits[..copy_len].copy_from_slice(&out_data[..copy_len]);
                    true
                }
                Err(e) => {
                    log::debug!("Failed to copy f16 output from GPU: {}", e);
                    false
                }
            },
            Err(e) => {
                log::debug!("HSA f16 kernel execution failed: {}", e);
                false
            }
        }
    } else {
        let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
        let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

        let q_buf = match HsaBuffer::from_slice(agent, &q_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate Q buffer: {}", e);
                return false;
            }
        };
        let k_buf = match HsaBuffer::from_slice(agent, &k_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate K buffer: {}", e);
                return false;
            }
        };
        let v_buf = match HsaBuffer::from_slice(agent, &v_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate V buffer: {}", e);
                return false;
            }
        };

        let result = kernel.forward_f32(
            queue,
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
            Ok(out_buf) => match out_buf.to_vec() {
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
                log::debug!("HSA kernel execution failed: {}", e);
                false
            }
        }
    }
}

/// Build paged attention layout and block metadata without dtype conversion.
#[cfg(target_os = "linux")]
fn rocm_build_paged_metadata<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output_len: usize,
    config: &PagedAttentionConfig,
) -> Option<(crate::ops::paged_attn::PagedAttentionLayout, Vec<i32>, Vec<i32>)> {
    let layout = crate::ops::paged_attn::build_paged_layout(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        output_len,
        config,
    )?;

    let kv_len = seq_lens[0] as usize;
    if seq_lens.iter().any(|&len| len as usize != kv_len) {
        log::debug!("Paged attention: GPU kernels require uniform seq_lens");
        return None;
    }
    if kv_len != layout.max_kv_len {
        log::debug!("Paged attention: GPU kernels require packed page_table");
        return None;
    }
    if kv_len < layout.seq_len {
        log::warn!("Paged attention: kv_len shorter than seq_len");
        return None;
    }

    let offset = kv_len - layout.seq_len;
    let offset_i32 = match i32::try_from(offset) {
        Ok(value) => value,
        Err(_) => {
            log::warn!("Paged attention: block offset exceeds i32");
            return None;
        }
    };
    let block_offsets = vec![offset_i32; layout.batch_size];

    let max_block_id = page_table.iter().copied().max().unwrap_or(0) as usize;
    if max_block_id >= layout.num_blocks {
        log::warn!("Paged attention: page_table references invalid block id");
        return None;
    }

    let block_tables: Vec<i32> = match page_table
        .iter()
        .map(|&value| i32::try_from(value).ok())
        .collect::<Option<Vec<_>>>()
    {
        Some(values) => values,
        None => {
            log::warn!("Paged attention: page_table value exceeds i32");
            return None;
        }
    };

    Some((layout, block_tables, block_offsets))
}

/// ROCm/HSA paged attention dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
#[cfg(target_os = "linux")]
fn rocm_paged_attention<T: KernelFloat>(
    kernel: &HsaPagedAttentionKernel,
    queue: &HsaQueueWrapper,
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [T],
    config: &PagedAttentionConfig,
) -> bool {
    let agent = kernel.agent();

    if T::TYPE_ID == FloatType::F16 {
        let (layout, block_tables, block_offsets) = match rocm_build_paged_metadata(
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            output.len(),
            config,
        ) {
            Some(values) => values,
            None => return false,
        };

        let q_bits: &[u16] = unsafe {
            std::slice::from_raw_parts(q.as_ptr() as *const u16, q.len())
        };
        let k_bits: &[u16] = unsafe {
            std::slice::from_raw_parts(k_cache.as_ptr() as *const u16, k_cache.len())
        };
        let v_bits: &[u16] = unsafe {
            std::slice::from_raw_parts(v_cache.as_ptr() as *const u16, v_cache.len())
        };
        let output_bits: &mut [u16] = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u16, output.len())
        };

        let q_buf = match HsaBuffer::from_slice(agent, q_bits) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 Q buffer: {}", e);
                return false;
            }
        };
        let k_buf = match HsaBuffer::from_slice(agent, k_bits) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 K buffer: {}", e);
                return false;
            }
        };
        let v_buf = match HsaBuffer::from_slice(agent, v_bits) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 V buffer: {}", e);
                return false;
            }
        };
        let table_buf = match HsaBuffer::from_slice(agent, &block_tables) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate block_tables buffer: {}", e);
                return false;
            }
        };
        let offsets_buf = match HsaBuffer::from_slice(agent, &block_offsets) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate block_offsets buffer: {}", e);
                return false;
            }
        };

        let result = kernel.forward_f16(
            queue,
            &q_buf,
            &k_buf,
            &v_buf,
            &table_buf,
            &offsets_buf,
            layout.batch_size,
            layout.num_heads,
            layout.head_dim,
            layout.page_size,
            layout.seq_len,
        );

        match result {
            Ok(out_buf) => match out_buf.to_vec() {
                Ok(out_data) => {
                    let copy_len = output_bits.len().min(out_data.len());
                    output_bits[..copy_len].copy_from_slice(&out_data[..copy_len]);
                    true
                }
                Err(e) => {
                    log::debug!("Failed to copy f16 output from GPU: {}", e);
                    false
                }
            },
            Err(e) => {
                log::debug!("HSA f16 paged kernel execution failed: {}", e);
                false
            }
        }
    } else {
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

        let q_buf = match HsaBuffer::from_slice(agent, &inputs.q_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate Q buffer: {}", e);
                return false;
            }
        };
        let k_buf = match HsaBuffer::from_slice(agent, &inputs.k_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate K buffer: {}", e);
                return false;
            }
        };
        let v_buf = match HsaBuffer::from_slice(agent, &inputs.v_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate V buffer: {}", e);
                return false;
            }
        };
        let table_buf = match HsaBuffer::from_slice(agent, &inputs.block_tables) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate block_tables buffer: {}", e);
                return false;
            }
        };
        let offsets_buf = match HsaBuffer::from_slice(agent, &inputs.block_offsets) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate block_offsets buffer: {}", e);
                return false;
            }
        };

        let result = kernel.forward_f32(
            queue,
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
            Ok(out_buf) => match out_buf.to_vec() {
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
                log::debug!("HSA paged kernel execution failed: {}", e);
                false
            }
        }
    }
}

/// ROCm/HSA softmax dispatch.
#[cfg(target_os = "linux")]
fn rocm_softmax<T: KernelFloat>(
    kernel: &HsaSoftmaxKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    input: &[T],
    output: &mut [T],
    num_rows: usize,
    row_size: usize,
) -> bool {
    use std::ffi::c_void;

    if kernel.has_f16() && T::TYPE_ID == FloatType::F16 {
        let input_f16: &[half::f16] = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const half::f16, input.len())
        };
        let output_f16: &mut [half::f16] = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut half::f16, output.len())
        };

        let input_buf = match HsaBuffer::from_slice(agent, input_f16) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 input buffer: {}", e);
                return false;
            }
        };

        let output_init = vec![half::f16::ZERO; output.len()];
        let mut output_buf = match HsaBuffer::from_slice(agent, &output_init) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 output buffer: {}", e);
                return false;
            }
        };

        let result = kernel.softmax_f16(
            queue,
            input_buf.as_ptr() as *const c_void,
            output_buf.as_mut_ptr() as *mut c_void,
            num_rows,
            row_size,
        );

        match result {
            Ok(()) => match output_buf.to_vec() {
                Ok(out_data) => {
                    let copy_len = output_f16.len().min(out_data.len());
                    output_f16[..copy_len].copy_from_slice(&out_data[..copy_len]);
                    true
                }
                Err(e) => {
                    log::debug!("Failed to copy f16 softmax output from GPU: {}", e);
                    false
                }
            },
            Err(e) => {
                log::debug!("HSA f16 softmax execution failed: {}", e);
                false
            }
        }
    } else {
        let input_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();

        let input_buf = match HsaBuffer::from_slice(agent, &input_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate input buffer: {}", e);
                return false;
            }
        };

        let output_f32 = vec![0.0f32; output.len()];
        let mut output_buf = match HsaBuffer::from_slice(agent, &output_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate output buffer: {}", e);
                return false;
            }
        };

        let result = kernel.softmax_f32(
            queue,
            input_buf.as_ptr() as *const c_void,
            output_buf.as_mut_ptr() as *mut c_void,
            num_rows,
            row_size,
        );

        match result {
            Ok(()) => match output_buf.to_vec() {
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
                log::debug!("HSA softmax execution failed: {}", e);
                false
            }
        }
    }
}

/// ROCm/HSA RMSNorm dispatch for f32 data.
#[cfg(target_os = "linux")]
fn rocm_rms_norm_f32(
    kernel: &HsaRmsNormKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> bool {
    use std::ffi::c_void;

    let expected = batch.saturating_mul(hidden);
    if input.len() != expected || output.len() != expected || weight.len() != hidden {
        log::debug!("ROCm rms_norm dispatch skipped: length mismatch");
        return false;
    }

    let input_buf = match HsaBuffer::from_slice(agent, input) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm input buffer: {}", e);
            return false;
        }
    };
    let weight_buf = match HsaBuffer::from_slice(agent, weight) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm weight buffer: {}", e);
            return false;
        }
    };

    let output_init = vec![0.0f32; output.len()];
    let mut output_buf = match HsaBuffer::from_slice(agent, &output_init) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm output buffer: {}", e);
            return false;
        }
    };

    let result = kernel.forward(
        queue,
        input_buf.as_ptr() as *const c_void,
        weight_buf.as_ptr() as *const c_void,
        output_buf.as_mut_ptr() as *mut c_void,
        batch,
        hidden,
        eps,
    );

    match result {
        Ok(()) => match output_buf.to_vec() {
            Ok(out_data) => {
                for (dst, val) in output.iter_mut().zip(out_data.iter()) {
                    *dst = *val;
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy rms_norm output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("HSA rms_norm execution failed: {}", e);
            false
        }
    }
}

/// ROCm/HSA RMSNorm inplace dispatch for f32 data.
#[cfg(target_os = "linux")]
fn rocm_rms_norm_inplace_f32(
    kernel: &HsaRmsNormKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    data: &mut [f32],
    weight: &[f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> bool {
    use std::ffi::c_void;

    let expected = batch.saturating_mul(hidden);
    if data.len() != expected || weight.len() != hidden {
        log::debug!("ROCm rms_norm_inplace dispatch skipped: length mismatch");
        return false;
    }

    let data_buf = match HsaBuffer::from_slice(agent, data) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm_inplace data buffer: {}", e);
            return false;
        }
    };
    let weight_buf = match HsaBuffer::from_slice(agent, weight) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm_inplace weight buffer: {}", e);
            return false;
        }
    };

    let output_init = vec![0.0f32; data.len()];
    let mut output_buf = match HsaBuffer::from_slice(agent, &output_init) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate rms_norm_inplace output buffer: {}", e);
            return false;
        }
    };

    let result = kernel.forward(
        queue,
        data_buf.as_ptr() as *const c_void,
        weight_buf.as_ptr() as *const c_void,
        output_buf.as_mut_ptr() as *mut c_void,
        batch,
        hidden,
        eps,
    );

    match result {
        Ok(()) => match output_buf.to_vec() {
            Ok(out_data) => {
                for (dst, val) in data.iter_mut().zip(out_data.iter()) {
                    *dst = *val;
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy rms_norm_inplace output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("HSA rms_norm_inplace execution failed: {}", e);
            false
        }
    }
}

/// ROCm/HSA SiLU dispatch for f32 data.
#[cfg(target_os = "linux")]
fn rocm_silu_f32(
    kernel: &HsaSiluKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    input: &[f32],
    output: &mut [f32],
) -> bool {
    use std::ffi::c_void;

    if input.len() != output.len() {
        log::debug!("ROCm silu dispatch skipped: length mismatch");
        return false;
    }

    let input_buf = match HsaBuffer::from_slice(agent, input) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate silu input buffer: {}", e);
            return false;
        }
    };

    let output_init = vec![0.0f32; output.len()];
    let mut output_buf = match HsaBuffer::from_slice(agent, &output_init) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate silu output buffer: {}", e);
            return false;
        }
    };

    let result = kernel.forward(
        queue,
        input_buf.as_ptr() as *const c_void,
        output_buf.as_mut_ptr() as *mut c_void,
        input.len(),
    );

    match result {
        Ok(()) => match output_buf.to_vec() {
            Ok(out_data) => {
                for (dst, val) in output.iter_mut().zip(out_data.iter()) {
                    *dst = *val;
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy silu output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("HSA silu execution failed: {}", e);
            false
        }
    }
}

/// ROCm/HSA SiLU inplace dispatch for f32 data.
#[cfg(target_os = "linux")]
fn rocm_silu_inplace_f32(
    kernel: &HsaSiluKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    data: &mut [f32],
) -> bool {
    let mut data_buf = match HsaBuffer::from_slice(agent, data) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate silu_inplace data buffer: {}", e);
            return false;
        }
    };

    let result = kernel.forward_inplace(queue, data_buf.as_mut_ptr(), data.len());

    match result {
        Ok(()) => match data_buf.to_vec() {
            Ok(out_data) => {
                for (dst, val) in data.iter_mut().zip(out_data.iter()) {
                    *dst = *val;
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy silu_inplace output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("HSA silu_inplace execution failed: {}", e);
            false
        }
    }
}

/// ROCm/HSA matmul dispatch via the linear kernel for f32 data.
///
/// This kernel currently supports only the most common inference shape:
/// - transpose_a = false
/// - transpose_b = false
/// - alpha = 1.0
/// - beta = 0.0
#[cfg(target_os = "linux")]
fn rocm_matmul_supported(config: &MatmulConfig) -> bool {
    if config.transpose_a || config.transpose_b {
        log::debug!("ROCm matmul dispatch skipped: transpose not supported by linear kernel");
        return false;
    }
    if config.alpha != 1.0 || config.beta != 0.0 {
        log::debug!("ROCm matmul dispatch skipped: alpha/beta not supported by linear kernel");
        return false;
    }
    true
}

#[cfg(target_os = "linux")]
fn rocm_matmul_lengths_ok(a: &[f32], b: &[f32], c: &[f32], config: &MatmulConfig) -> bool {
    let expected_a = config.m.saturating_mul(config.k);
    let expected_b = config.k.saturating_mul(config.n);
    let expected_c = config.m.saturating_mul(config.n);
    if a.len() != expected_a || b.len() != expected_b || c.len() != expected_c {
        log::debug!("ROCm matmul dispatch skipped: length mismatch");
        return false;
    }
    true
}

#[cfg(target_os = "linux")]
fn rocm_matmul_params(config: &MatmulConfig) -> Option<LinearParams> {
    let in_features = match u32::try_from(config.k) {
        Ok(v) => v,
        Err(_) => {
            log::debug!("ROCm matmul dispatch skipped: k exceeds u32");
            return None;
        }
    };
    let out_features = match u32::try_from(config.n) {
        Ok(v) => v,
        Err(_) => {
            log::debug!("ROCm matmul dispatch skipped: n exceeds u32");
            return None;
        }
    };
    Some(LinearParams {
        in_features,
        out_features,
        has_bias: 0,
        padding: 0,
    })
}

#[cfg(target_os = "linux")]
fn rocm_matmul_f32(
    kernel: &HsaLinearKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    config: &MatmulConfig,
) -> bool {
    use std::ffi::c_void;

    if !rocm_matmul_supported(config) {
        return false;
    }
    if !rocm_matmul_lengths_ok(a, b, c, config) {
        return false;
    }

    let params = match rocm_matmul_params(config) {
        Some(p) => p,
        None => return false,
    };

    let a_buf = match HsaBuffer::from_slice(agent, a) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate matmul input buffer: {}", e);
            return false;
        }
    };
    let b_buf = match HsaBuffer::from_slice(agent, b) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate matmul weight buffer: {}", e);
            return false;
        }
    };

    let output_init = vec![0.0f32; c.len()];
    let mut c_buf = match HsaBuffer::from_slice(agent, &output_init) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate matmul output buffer: {}", e);
            return false;
        }
    };

    let result = kernel.forward(
        queue,
        params,
        a_buf.as_ptr() as *const c_void,
        b_buf.as_ptr() as *const c_void,
        None,
        c_buf.as_mut_ptr() as *mut c_void,
        config.m,
    );

    match result {
        Ok(()) => match c_buf.to_vec() {
            Ok(out_data) => {
                for (dst, val) in c.iter_mut().zip(out_data.iter()) {
                    *dst = *val;
                }
                true
            }
            Err(e) => {
                log::debug!("Failed to copy matmul output from GPU: {}", e);
                false
            }
        },
        Err(e) => {
            log::debug!("HSA matmul execution failed: {}", e);
            false
        }
    }
}

/// ROCm/HSA argmax dispatch.
#[cfg(target_os = "linux")]
fn rocm_argmax<T: KernelFloat>(
    kernel: &HsaSamplingKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    logits: &[T],
    batch_size: usize,
    vocab_size: usize,
) -> Option<Vec<u32>> {
    use std::ffi::c_void;

    if kernel.has_f16() && T::TYPE_ID == FloatType::F16 {
        let logits_f16: &[half::f16] = unsafe {
            std::slice::from_raw_parts(logits.as_ptr() as *const half::f16, logits.len())
        };

        let logits_buf = match HsaBuffer::from_slice(agent, logits_f16) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 logits buffer: {}", e);
                return None;
            }
        };

        let indices = vec![0u32; batch_size];
        let mut indices_buf = match HsaBuffer::from_slice(agent, &indices) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate indices buffer: {}", e);
                return None;
            }
        };

        let result = kernel.argmax_f16(
            queue,
            logits_buf.as_ptr() as *const c_void,
            indices_buf.as_mut_ptr() as *mut c_void,
            batch_size,
            vocab_size,
        );

        return match result {
            Ok(()) => match indices_buf.to_vec() {
                Ok(out_data) => Some(out_data),
                Err(e) => {
                    log::debug!("Failed to copy indices from GPU: {}", e);
                    None
                }
            },
            Err(e) => {
                log::debug!("HSA argmax f16 execution failed: {}", e);
                None
            }
        };
    }

    let logits_f32: Vec<f32> = logits.iter().map(|x| x.to_f32()).collect();

    let logits_buf = match HsaBuffer::from_slice(agent, &logits_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate logits buffer: {}", e);
            return None;
        }
    };

    let indices = vec![0u32; batch_size];
    let mut indices_buf = match HsaBuffer::from_slice(agent, &indices) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate indices buffer: {}", e);
            return None;
        }
    };

    let result = kernel.argmax_f32(
        queue,
        logits_buf.as_ptr() as *const c_void,
        indices_buf.as_mut_ptr() as *mut c_void,
        batch_size,
        vocab_size,
    );

    match result {
        Ok(()) => match indices_buf.to_vec() {
            Ok(out_data) => Some(out_data),
            Err(e) => {
                log::debug!("Failed to copy indices from GPU: {}", e);
                None
            }
        },
        Err(e) => {
            log::debug!("HSA argmax execution failed: {}", e);
            None
        }
    }
}

/// ROCm/HSA topk dispatch.
#[cfg(target_os = "linux")]
fn rocm_topk<T: KernelFloat>(
    kernel: &HsaSamplingKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    logits: &[T],
    k: usize,
    batch_size: usize,
    vocab_size: usize,
) -> Option<TopKResult> {
    use std::ffi::c_void;

    if !kernel.has_topk() {
        return None;
    }

    if kernel.has_f16() && T::TYPE_ID == FloatType::F16 {
        let logits_f16: &[half::f16] = unsafe {
            std::slice::from_raw_parts(logits.as_ptr() as *const half::f16, logits.len())
        };

        let logits_buf = match HsaBuffer::from_slice(agent, logits_f16) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 logits buffer: {}", e);
                return None;
            }
        };

        let indices = vec![0u32; batch_size * k];
        let mut indices_buf = match HsaBuffer::from_slice(agent, &indices) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate indices buffer: {}", e);
                return None;
            }
        };

        // TopK values are accumulated as f32 even for f16 logits.
        let values = vec![0.0f32; batch_size * k];
        let mut values_buf = match HsaBuffer::from_slice(agent, &values) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate values buffer: {}", e);
                return None;
            }
        };

        let result = kernel.topk_f16(
            queue,
            logits_buf.as_ptr() as *const c_void,
            indices_buf.as_mut_ptr() as *mut c_void,
            values_buf.as_mut_ptr() as *mut c_void,
            batch_size,
            vocab_size,
            k,
        );

        return match result {
            Ok(()) => {
                let out_indices = match indices_buf.to_vec() {
                    Ok(data) => data,
                    Err(e) => {
                        log::debug!("Failed to copy indices from GPU: {}", e);
                        return None;
                    }
                };
                let out_values = match values_buf.to_vec() {
                    Ok(data) => data,
                    Err(e) => {
                        log::debug!("Failed to copy values from GPU: {}", e);
                        return None;
                    }
                };
                Some(TopKResult {
                    indices: out_indices,
                    values: out_values,
                })
            }
            Err(e) => {
                log::debug!("HSA topk f16 execution failed: {}", e);
                None
            }
        };
    }

    let logits_f32: Vec<f32> = logits.iter().map(|x| x.to_f32()).collect();

    let logits_buf = match HsaBuffer::from_slice(agent, &logits_f32) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate logits buffer: {}", e);
            return None;
        }
    };

    let indices = vec![0u32; batch_size * k];
    let mut indices_buf = match HsaBuffer::from_slice(agent, &indices) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate indices buffer: {}", e);
            return None;
        }
    };

    let values = vec![0.0f32; batch_size * k];
    let mut values_buf = match HsaBuffer::from_slice(agent, &values) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate values buffer: {}", e);
            return None;
        }
    };

    let result = kernel.topk_f32(
        queue,
        logits_buf.as_ptr() as *const c_void,
        indices_buf.as_mut_ptr() as *mut c_void,
        values_buf.as_mut_ptr() as *mut c_void,
        batch_size,
        vocab_size,
        k,
    );

    match result {
        Ok(()) => {
            let out_indices = match indices_buf.to_vec() {
                Ok(data) => data,
                Err(e) => {
                    log::debug!("Failed to copy indices from GPU: {}", e);
                    return None;
                }
            };
            let out_values = match values_buf.to_vec() {
                Ok(data) => data,
                Err(e) => {
                    log::debug!("Failed to copy values from GPU: {}", e);
                    return None;
                }
            };
            Some(TopKResult {
                indices: out_indices,
                values: out_values,
            })
        }
        Err(e) => {
            log::debug!("HSA topk execution failed: {}", e);
            None
        }
    }
}

/// ROCm/HSA RoPE apply dispatch.
#[cfg(target_os = "linux")]
fn rocm_rope_apply<T: KernelFloat>(
    kernel: &HsaRoPEKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    q: &[T],
    k: &[T],
    cos_cache: &[f32],
    sin_cache: &[f32],
    q_out: &mut [T],
    k_out: &mut [T],
    batch_size: usize,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    position_offset: usize,
) -> bool {
    use std::ffi::c_void;

    let cos_buf = match HsaBuffer::from_slice(agent, cos_cache) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate cos cache buffer: {}", e);
            return false;
        }
    };
    let sin_buf = match HsaBuffer::from_slice(agent, sin_cache) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to allocate sin cache buffer: {}", e);
            return false;
        }
    };

    if kernel.has_f16() && T::TYPE_ID == FloatType::F16 {
        let q_f16: &[half::f16] = unsafe {
            std::slice::from_raw_parts(q.as_ptr() as *const half::f16, q.len())
        };
        let k_f16: &[half::f16] = unsafe {
            std::slice::from_raw_parts(k.as_ptr() as *const half::f16, k.len())
        };
        let q_out_f16: &mut [half::f16] = unsafe {
            std::slice::from_raw_parts_mut(q_out.as_mut_ptr() as *mut half::f16, q_out.len())
        };
        let k_out_f16: &mut [half::f16] = unsafe {
            std::slice::from_raw_parts_mut(k_out.as_mut_ptr() as *mut half::f16, k_out.len())
        };

        let q_buf = match HsaBuffer::from_slice(agent, q_f16) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 Q buffer: {}", e);
                return false;
            }
        };
        let k_buf = match HsaBuffer::from_slice(agent, k_f16) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 K buffer: {}", e);
                return false;
            }
        };

        let q_out_init = vec![half::f16::ZERO; q_out.len()];
        let mut q_out_buf = match HsaBuffer::from_slice(agent, &q_out_init) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 Q_out buffer: {}", e);
                return false;
            }
        };
        let k_out_init = vec![half::f16::ZERO; k_out.len()];
        let mut k_out_buf = match HsaBuffer::from_slice(agent, &k_out_init) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate f16 K_out buffer: {}", e);
                return false;
            }
        };

        let result = kernel.rope_apply_f16(
            queue,
            q_buf.as_ptr() as *const c_void,
            k_buf.as_ptr() as *const c_void,
            cos_buf.as_ptr() as *const c_void,
            sin_buf.as_ptr() as *const c_void,
            q_out_buf.as_mut_ptr() as *mut c_void,
            k_out_buf.as_mut_ptr() as *mut c_void,
            batch_size,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            position_offset,
        );

        match result {
            Ok(()) => {
                match (q_out_buf.to_vec(), k_out_buf.to_vec()) {
                    (Ok(q_data), Ok(k_data)) => {
                        let q_copy_len = q_out_f16.len().min(q_data.len());
                        q_out_f16[..q_copy_len].copy_from_slice(&q_data[..q_copy_len]);
                        let k_copy_len = k_out_f16.len().min(k_data.len());
                        k_out_f16[..k_copy_len].copy_from_slice(&k_data[..k_copy_len]);
                        true
                    }
                    (Err(e), _) => {
                        log::debug!("Failed to copy f16 RoPE Q output from GPU: {}", e);
                        false
                    }
                    (_, Err(e)) => {
                        log::debug!("Failed to copy f16 RoPE K output from GPU: {}", e);
                        false
                    }
                }
            }
            Err(e) => {
                log::debug!("HSA f16 RoPE apply failed: {}", e);
                false
            }
        }
    } else {
        let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
        let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();

        let q_buf = match HsaBuffer::from_slice(agent, &q_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate Q buffer: {}", e);
                return false;
            }
        };
        let k_buf = match HsaBuffer::from_slice(agent, &k_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate K buffer: {}", e);
                return false;
            }
        };

        let q_out_f32 = vec![0.0f32; q_out.len()];
        let mut q_out_buf = match HsaBuffer::from_slice(agent, &q_out_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate Q_out buffer: {}", e);
                return false;
            }
        };
        let k_out_f32 = vec![0.0f32; k_out.len()];
        let mut k_out_buf = match HsaBuffer::from_slice(agent, &k_out_f32) {
            Ok(buf) => buf,
            Err(e) => {
                log::debug!("Failed to allocate K_out buffer: {}", e);
                return false;
            }
        };

        let result = kernel.rope_apply_f32(
            queue,
            q_buf.as_ptr() as *const c_void,
            k_buf.as_ptr() as *const c_void,
            cos_buf.as_ptr() as *const c_void,
            sin_buf.as_ptr() as *const c_void,
            q_out_buf.as_mut_ptr() as *mut c_void,
            k_out_buf.as_mut_ptr() as *mut c_void,
            batch_size,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            position_offset,
        );

        match result {
            Ok(()) => match (q_out_buf.to_vec(), k_out_buf.to_vec()) {
                (Ok(q_data), Ok(k_data)) => {
                    for (i, val) in q_data.into_iter().enumerate() {
                        if i < q_out.len() {
                            q_out[i] = T::from_f32(val);
                        }
                    }
                    for (i, val) in k_data.into_iter().enumerate() {
                        if i < k_out.len() {
                            k_out[i] = T::from_f32(val);
                        }
                    }
                    true
                }
                (Err(e), _) => {
                    log::debug!("Failed to copy RoPE Q output from GPU: {}", e);
                    false
                }
                (_, Err(e)) => {
                    log::debug!("Failed to copy RoPE K output from GPU: {}", e);
                    false
                }
            },
            Err(e) => {
                log::debug!("HSA RoPE apply failed: {}", e);
                false
            }
        }
    }
}

/// ROCm/HSA RoPE apply inplace dispatch.
#[cfg(target_os = "linux")]
fn rocm_rope_apply_inplace<T: KernelFloat>(
    kernel: &HsaRoPEKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    x: &mut [T],
    cos_cache: &[f32],
    sin_cache: &[f32],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: usize,
) -> bool {
    use std::ffi::c_void;

    if !kernel.has_inplace() {
        return false;
    }

    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();

    let mut x_buf = match HsaBuffer::from_slice(agent, &x_f32) {
        Ok(buf) => buf,
        Err(_) => return false,
    };
    let cos_buf = match HsaBuffer::from_slice(agent, cos_cache) {
        Ok(buf) => buf,
        Err(_) => return false,
    };
    let sin_buf = match HsaBuffer::from_slice(agent, sin_cache) {
        Ok(buf) => buf,
        Err(_) => return false,
    };

    let result = kernel.rope_apply_inplace_f32(
        queue,
        x_buf.as_mut_ptr() as *mut c_void,
        cos_buf.as_ptr() as *const c_void,
        sin_buf.as_ptr() as *const c_void,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        position_offset,
    );

    match result {
        Ok(()) => {
            if let Ok(out_data) = x_buf.to_vec() {
                for (i, val) in out_data.into_iter().enumerate() {
                    if i < x.len() {
                        x[i] = T::from_f32(val);
                    }
                }
                true
            } else {
                false
            }
        },
        Err(_) => false,
    }
}

/// ROCm/HSA Q4 matmul dispatch.
#[cfg(target_os = "linux")]
fn rocm_q4_matmul(
    kernel: &HsaQuantizedKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    input: &[f32],
    q_weight: &[u8],
    scales: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    use std::ffi::c_void;

    let input_buf = match HsaBuffer::from_slice(agent, input) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let weight_buf = match HsaBuffer::from_slice(agent, q_weight) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
    let scales_buf = match HsaBuffer::from_slice(agent, &scales_f32) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    let output = vec![0.0f32; m * n];
    let mut output_buf = match HsaBuffer::from_slice(agent, &output) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    // For symmetric quantization, zeros are not needed (pass null)
    // Group size 32 is typical for Q4 block-based quantization
    let result = kernel.q4_matmul_f32(
        queue,
        input_buf.as_ptr() as *const c_void,
        weight_buf.as_ptr() as *const c_void,
        scales_buf.as_ptr() as *const c_void,
        std::ptr::null(), // zeros_ptr - symmetric quantization
        output_buf.as_mut_ptr() as *mut c_void,
        m, k, n,
        32, // group_size
    );

    match result {
        Ok(()) => output_buf.to_vec().ok(),
        Err(_) => None,
    }
}

/// ROCm/HSA Q8 matmul dispatch.
#[cfg(target_os = "linux")]
fn rocm_q8_matmul(
    kernel: &HsaQuantizedKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    input: &[f32],
    q_weight: &[i8],
    scales: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    use std::ffi::c_void;

    if !kernel.has_q8() {
        return None;
    }

    let input_buf = match HsaBuffer::from_slice(agent, input) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    // Cast i8 to u8 for buffer allocation
    let weight_u8: Vec<u8> = q_weight.iter().map(|&w| w as u8).collect();
    let weight_buf = match HsaBuffer::from_slice(agent, &weight_u8) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
    let scales_buf = match HsaBuffer::from_slice(agent, &scales_f32) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    let output = vec![0.0f32; m * n];
    let mut output_buf = match HsaBuffer::from_slice(agent, &output) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    // For symmetric quantization, zeros are not needed (pass null)
    // Group size 32 is typical for Q8 block-based quantization
    let result = kernel.q8_matmul_f32(
        queue,
        input_buf.as_ptr() as *const c_void,
        weight_buf.as_ptr() as *const c_void,
        scales_buf.as_ptr() as *const c_void,
        std::ptr::null(), // zeros_ptr - symmetric quantization
        output_buf.as_mut_ptr() as *mut c_void,
        m, k, n,
        32, // group_size
    );

    match result {
        Ok(()) => output_buf.to_vec().ok(),
        Err(_) => None,
    }
}

#[cfg(target_os = "linux")]
fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize, String> {
    a.checked_mul(b).ok_or_else(|| format!("{name} overflow"))
}

/// ROCm/HSA Q4 dequantize dispatch.
#[cfg(target_os = "linux")]
fn rocm_q4_dequantize(
    kernel: &HsaQuantizedKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    q_weight: &[u8],
    scales: &[half::f16],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    use std::ffi::c_void;

    if !kernel.has_q4_dequantize() {
        return None;
    }
    if n == 0 || k == 0 {
        log::debug!("ROCm q4_dequantize invalid dims: n={n}, k={k}");
        return None;
    }
    if k % 32 != 0 {
        log::debug!("ROCm q4_dequantize requires k multiple of 32, got {k}");
        return None;
    }

    let blocks = k / 32;
    let num_blocks = match checked_mul(n, blocks, "q4 blocks") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm q4_dequantize num_blocks overflow: {e}");
            return None;
        }
    };
    if num_blocks == 0 || num_blocks > i32::MAX as usize {
        log::debug!("ROCm q4_dequantize num_blocks out of range: {num_blocks}");
        return None;
    }

    let expected_q_weight = match checked_mul(num_blocks, 16, "q4 weights") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm q4_dequantize q_weight overflow: {e}");
            return None;
        }
    };
    if q_weight.len() != expected_q_weight {
        log::debug!(
            "ROCm q4_dequantize q_weight length mismatch: expected {expected_q_weight}, got {}",
            q_weight.len()
        );
        return None;
    }
    if scales.len() != num_blocks {
        log::debug!(
            "ROCm q4_dequantize scales length mismatch: expected {num_blocks}, got {}",
            scales.len()
        );
        return None;
    }

    let total_values = match checked_mul(num_blocks, 32, "q4 output") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm q4_dequantize output overflow: {e}");
            return None;
        }
    };
    if total_values > u32::MAX as usize {
        log::debug!("ROCm q4_dequantize output exceeds addressable range: {total_values}");
        return None;
    }

    let q_weight_buf = match HsaBuffer::from_slice(agent, q_weight) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm q4_dequantize upload q_weight failed: {e}");
            return None;
        }
    };
    let scales_buf = match HsaBuffer::from_slice(agent, scales) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm q4_dequantize upload scales failed: {e}");
            return None;
        }
    };
    let mut output_buf = match HsaBuffer::<f32>::alloc_zeros(agent, total_values) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm q4_dequantize alloc output failed: {e}");
            return None;
        }
    };

    let result = kernel.q4_dequantize_f32(
        queue,
        q_weight_buf.as_ptr() as *const c_void,
        scales_buf.as_ptr() as *const c_void,
        output_buf.as_mut_ptr() as *mut c_void,
        num_blocks,
    );

    match result {
        Ok(()) => match output_buf.to_vec() {
            Ok(v) => Some(v),
            Err(e) => {
                log::debug!("ROCm q4_dequantize readback failed: {e}");
                None
            }
        },
        Err(e) => {
            log::debug!("ROCm q4_dequantize kernel execution failed: {e}");
            None
        }
    }
}

/// ROCm/HSA AWQ dequantize dispatch.
#[cfg(target_os = "linux")]
fn rocm_awq_dequantize(
    kernel: &HsaQuantizedKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[half::f16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Option<Vec<f32>> {
    use std::ffi::c_void;

    if !kernel.has_awq_dequantize() {
        return None;
    }
    if n == 0 || k == 0 {
        log::debug!("ROCm awq_dequantize invalid dims: n={n}, k={k}");
        return None;
    }
    if group_size == 0 {
        log::debug!("ROCm awq_dequantize requires group_size > 0");
        return None;
    }
    if n % 8 != 0 {
        log::debug!("ROCm awq_dequantize requires n multiple of 8, got {n}");
        return None;
    }
    if k % group_size != 0 {
        log::debug!(
            "ROCm awq_dequantize requires k multiple of group_size: k={k}, group_size={group_size}"
        );
        return None;
    }

    let groups = k / group_size;
    if groups == 0 || groups > i32::MAX as usize {
        log::debug!("ROCm awq_dequantize groups out of range: {groups}");
        return None;
    }
    let packed_out = n / 8;

    let expected_qweight = match checked_mul(packed_out, k, "awq qweight") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm awq_dequantize qweight overflow: {e}");
            return None;
        }
    };
    let expected_qzeros = match checked_mul(packed_out, groups, "awq qzeros") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm awq_dequantize qzeros overflow: {e}");
            return None;
        }
    };
    let expected_scales = match checked_mul(n, groups, "awq scales") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm awq_dequantize scales overflow: {e}");
            return None;
        }
    };

    if qweight.len() != expected_qweight {
        log::debug!(
            "ROCm awq_dequantize qweight length mismatch: expected {expected_qweight}, got {}",
            qweight.len()
        );
        return None;
    }
    if qzeros.len() != expected_qzeros {
        log::debug!(
            "ROCm awq_dequantize qzeros length mismatch: expected {expected_qzeros}, got {}",
            qzeros.len()
        );
        return None;
    }
    if scales.len() != expected_scales {
        log::debug!(
            "ROCm awq_dequantize scales length mismatch: expected {expected_scales}, got {}",
            scales.len()
        );
        return None;
    }
    if n > i32::MAX as usize || k > i32::MAX as usize || group_size > i32::MAX as usize {
        log::debug!("ROCm awq_dequantize dimensions exceed addressable range");
        return None;
    }

    let total_values = match checked_mul(n, k, "awq output") {
        Ok(v) => v,
        Err(e) => {
            log::debug!("ROCm awq_dequantize output overflow: {e}");
            return None;
        }
    };
    if total_values > u32::MAX as usize {
        log::debug!("ROCm awq_dequantize output exceeds addressable range: {total_values}");
        return None;
    }

    let qweight_buf = match HsaBuffer::from_slice(agent, qweight) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm awq_dequantize upload qweight failed: {e}");
            return None;
        }
    };
    let qzeros_buf = match HsaBuffer::from_slice(agent, qzeros) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm awq_dequantize upload qzeros failed: {e}");
            return None;
        }
    };
    let scales_buf = match HsaBuffer::from_slice(agent, scales) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm awq_dequantize upload scales failed: {e}");
            return None;
        }
    };
    let mut output_buf = match HsaBuffer::<f32>::alloc_zeros(agent, total_values) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("ROCm awq_dequantize alloc output failed: {e}");
            return None;
        }
    };

    let result = kernel.awq_dequantize_f32(
        queue,
        qweight_buf.as_ptr() as *const c_void,
        qzeros_buf.as_ptr() as *const c_void,
        scales_buf.as_ptr() as *const c_void,
        output_buf.as_mut_ptr() as *mut c_void,
        n,
        k,
        group_size,
        groups,
    );

    match result {
        Ok(()) => match output_buf.to_vec() {
            Ok(v) => Some(v),
            Err(e) => {
                log::debug!("ROCm awq_dequantize readback failed: {e}");
                None
            }
        },
        Err(e) => {
            log::debug!("ROCm awq_dequantize kernel execution failed: {e}");
            None
        }
    }
}

/// ROCm/HSA AWQ matmul dispatch.
#[cfg(target_os = "linux")]
fn rocm_awq_matmul(
    kernel: &HsaQuantizedKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    input: &[f32],
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Option<Vec<f32>> {
    use std::ffi::c_void;

    if !kernel.has_awq() {
        return None;
    }

    let input_buf = match HsaBuffer::from_slice(agent, input) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let weight_buf = match HsaBuffer::from_slice(agent, qweight) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let zeros_buf = match HsaBuffer::from_slice(agent, qzeros) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
    let scales_buf = match HsaBuffer::from_slice(agent, &scales_f32) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    let output = vec![0.0f32; m * n];
    let mut output_buf = match HsaBuffer::from_slice(agent, &output) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    // Note: AWQ kernel takes scales but not separate zeros
    // (zeros are packed with weights in AWQ format)
    let _ = &zeros_buf; // Keep allocation but unused in current kernel
    let result = kernel.awq_matmul_f32(
        queue,
        input_buf.as_ptr() as *const c_void,
        weight_buf.as_ptr() as *const c_void,
        scales_buf.as_ptr() as *const c_void,
        output_buf.as_mut_ptr() as *mut c_void,
        m, k, n, group_size,
    );

    match result {
        Ok(()) => output_buf.to_vec().ok(),
        Err(_) => None,
    }
}

/// ROCm/HSA MoE routing dispatch.
#[cfg(target_os = "linux")]
fn rocm_moe_route<T: KernelFloat>(
    kernel: &HsaMoeRouteKernel,
    queue: &HsaQueueWrapper,
    agent: &GpuAgent,
    hidden_states: &[T],
    gate_weights: &[f32],
    batch_size: usize,
    seq_len: usize,
    config: &MoERoutingConfig,
) -> Option<MoERoutingResult> {
    use std::ffi::c_void;

    let hidden_f32: Vec<f32> = hidden_states.iter().map(|x| x.to_f32()).collect();
    let num_tokens = batch_size * seq_len;

    let hidden_buf = match HsaBuffer::from_slice(agent, &hidden_f32) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let gate_buf = match HsaBuffer::from_slice(agent, gate_weights) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    let expert_indices = vec![0u32; num_tokens * config.num_experts_per_tok];
    let expert_weights = vec![0.0f32; num_tokens * config.num_experts_per_tok];

    let mut indices_buf = match HsaBuffer::from_slice(agent, &expert_indices) {
        Ok(buf) => buf,
        Err(_) => return None,
    };
    let mut weights_buf = match HsaBuffer::from_slice(agent, &expert_weights) {
        Ok(buf) => buf,
        Err(_) => return None,
    };

    let result = kernel.moe_route_f32(
        queue,
        hidden_buf.as_ptr() as *const c_void,
        gate_buf.as_ptr() as *const c_void,
        indices_buf.as_mut_ptr() as *mut c_void,
        weights_buf.as_mut_ptr() as *mut c_void,
        num_tokens,
        config.hidden_size,
        config.num_experts,
        config.num_experts_per_tok,
    );

    match result {
        Ok(()) => {
            let out_indices = indices_buf.to_vec().ok()?;
            let out_weights = weights_buf.to_vec().ok()?;

            Some(MoERoutingResult {
                expert_indices: out_indices,
                expert_weights: out_weights,
                num_tokens,
                top_k: config.num_experts_per_tok,
            })
        },
        Err(_) => None,
    }
}

#[derive(Clone, Copy)]
pub struct RocmBackend {}

impl RocmBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for RocmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for RocmBackend {
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue)) = (get_hsa_flash_kernel(), get_hsa_queue()) {
                        if rocm_flash_attention(kernel, queue, q, k, v, out, &config) {
                            return;
                        }
                        log::debug!("HSA kernel dispatch failed, falling back to CPU");
                    }
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue)) = (get_hsa_flash_kernel(), get_hsa_queue()) {
                        if rocm_flash_attention(kernel, queue, q, k, v, out, &config) {
                            return;
                        }
                        log::debug!("HSA kernel dispatch failed, falling back to CPU");
                    }
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue)) = (get_hsa_flash_kernel(), get_hsa_queue()) {
                        if rocm_flash_attention(kernel, queue, q, k, v, out, &config) {
                            return;
                        }
                        log::debug!("HSA kernel dispatch failed, falling back to CPU");
                    }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue)) = (get_hsa_paged_kernel(), get_hsa_queue()) {
                        if rocm_paged_attention(kernel, queue, q, k, v, page_table, seq_lens, out, &config) {
                            return;
                        }
                        log::debug!("HSA paged attention dispatch failed, falling back to CPU");
                    }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue)) = (get_hsa_paged_kernel(), get_hsa_queue()) {
                        if rocm_paged_attention(kernel, queue, q, k, v, page_table, seq_lens, out, &config) {
                            return;
                        }
                        log::debug!("HSA paged attention dispatch failed, falling back to CPU");
                    }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue)) = (get_hsa_paged_kernel(), get_hsa_queue()) {
                        if rocm_paged_attention(kernel, queue, q, k, v, page_table, seq_lens, out, &config) {
                            return;
                        }
                        log::debug!("HSA paged attention dispatch failed, falling back to CPU");
                    }
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
            |input, out| {
                let num_rows = config.effective_num_rows(input.len());
                let row_size = config.effective_row_size(input.len());
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_softmax_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_softmax(kernel, queue, agent, input, out, num_rows, row_size) {
                            return;
                        }
                    }
                }
                crate::ops::softmax::softmax(input, out, config.clone());
            },
            |input, out| {
                let num_rows = config.effective_num_rows(input.len());
                let row_size = config.effective_row_size(input.len());
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_softmax_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_softmax(kernel, queue, agent, input, out, num_rows, row_size) {
                            return;
                        }
                    }
                }
                crate::ops::softmax::softmax(input, out, config.clone());
            },
            |input, out| {
                let num_rows = config.effective_num_rows(input.len());
                let row_size = config.effective_row_size(input.len());
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_softmax_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_softmax(kernel, queue, agent, input, out, num_rows, row_size) {
                            return;
                        }
                    }
                }
                crate::ops::softmax::softmax(input, out, config.clone());
            },
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
            |a, b, c| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) = (
                        get_hsa_linear_kernel(),
                        get_hsa_queue(),
                        get_hsa_gpu_agent(),
                    ) {
                        if rocm_matmul_f32(kernel, queue, agent, a, b, c, &config) {
                            return;
                        }
                    }
                }
                crate::ops::matmul::cpu_matmul(a, b, c, config.clone());
            },
            |a, b, c| crate::ops::matmul::cpu_matmul(a, b, c, config.clone()),
            |a, b, c| crate::ops::matmul::cpu_matmul(a, b, c, config.clone()),
        )
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
        #[cfg(target_os = "linux")]
        {
            if let (Some(kernel), Some(queue), Some(agent)) =
                (get_hsa_quantized_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
            {
                if let Some(result) = rocm_q4_matmul(kernel, queue, agent, input, q_weight, scales, m, n, k) {
                    return Ok(result);
                }
            }
        }
        crate::ops::quantized::q4_matmul_cpu(input, q_weight, scales, m, n, k)
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
        #[cfg(target_os = "linux")]
        {
            if let (Some(kernel), Some(queue), Some(agent)) =
                (get_hsa_quantized_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
            {
                if let Some(result) = rocm_q8_matmul(kernel, queue, agent, input, q_weight, scales, m, n, k) {
                    return Ok(result);
                }
            }
        }
        crate::ops::quantized::q8_matmul_cpu(input, q_weight, scales, m, n, k)
    }

    fn q4_dequantize(
        &self,
        q_weight: &[u8],
        scales: &[half::f16],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        #[cfg(target_os = "linux")]
        {
            if let (Some(kernel), Some(queue), Some(agent)) =
                (get_hsa_quantized_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
            {
                if let Some(result) = rocm_q4_dequantize(kernel, queue, agent, q_weight, scales, n, k) {
                    return Ok(result);
                }
                log::debug!("ROCm q4_dequantize dispatch failed, falling back to CPU");
            }
        }
        crate::ops::quantized::q4_dequantize_cpu(q_weight, scales, n, k)
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
        #[cfg(target_os = "linux")]
        {
            if let (Some(kernel), Some(queue), Some(agent)) =
                (get_hsa_quantized_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
            {
                if let Some(result) =
                    rocm_awq_dequantize(kernel, queue, agent, qweight, qzeros, scales, n, k, group_size)
                {
                    return Ok(result);
                }
                log::debug!("ROCm awq_dequantize dispatch failed, falling back to CPU");
            }
        }
        crate::ops::quantized::awq_dequantize_cpu(qweight, qzeros, scales, n, k, group_size)
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
        #[cfg(target_os = "linux")]
        {
            if let (Some(kernel), Some(queue), Some(agent)) =
                (get_hsa_quantized_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
            {
                if let Some(result) = rocm_awq_matmul(
                    kernel, queue, agent, input, qweight, qzeros, scales, m, n, k, group_size,
                ) {
                    return Ok(result);
                }
            }
        }
        crate::ops::quantized::awq_matmul_cpu(
            input,
            qweight,
            qzeros,
            scales,
            m,
            n,
            k,
            group_size,
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_rope_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_rope_apply(
                            kernel, queue, agent, q, k, cos_cache, sin_cache, q_out, k_out,
                            batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset,
                        ) {
                            return;
                        }
                    }
                }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_rope_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_rope_apply(
                            kernel, queue, agent, q, k, cos_cache, sin_cache, q_out, k_out,
                            batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset,
                        ) {
                            return;
                        }
                    }
                }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_rope_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_rope_apply(
                            kernel, queue, agent, q, k, cos_cache, sin_cache, q_out, k_out,
                            batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset,
                        ) {
                            return;
                        }
                    }
                }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_rope_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_rope_apply_inplace(
                            kernel, queue, agent, x, cos_cache, sin_cache,
                            batch_size, seq_len, num_heads, head_dim, position_offset,
                        ) {
                            return;
                        }
                    }
                }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_rope_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_rope_apply_inplace(
                            kernel, queue, agent, x, cos_cache, sin_cache,
                            batch_size, seq_len, num_heads, head_dim, position_offset,
                        ) {
                            return;
                        }
                    }
                }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_rope_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if rocm_rope_apply_inplace(
                            kernel, queue, agent, x, cos_cache, sin_cache,
                            batch_size, seq_len, num_heads, head_dim, position_offset,
                        ) {
                            return;
                        }
                    }
                }
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
            |logits| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_sampling_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_topk(kernel, queue, agent, logits, k, batch_size, vocab_size) {
                            return result;
                        }
                    }
                }
                crate::ops::sampling::topk(logits, k, batch_size, vocab_size)
            },
            |logits| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_sampling_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_topk(kernel, queue, agent, logits, k, batch_size, vocab_size) {
                            return result;
                        }
                    }
                }
                crate::ops::sampling::topk(logits, k, batch_size, vocab_size)
            },
            |logits| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_sampling_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_topk(kernel, queue, agent, logits, k, batch_size, vocab_size) {
                            return result;
                        }
                    }
                }
                crate::ops::sampling::topk(logits, k, batch_size, vocab_size)
            },
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
            |logits| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_sampling_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_argmax(kernel, queue, agent, logits, batch_size, vocab_size) {
                            return result;
                        }
                    }
                }
                crate::ops::sampling::argmax(logits, batch_size, vocab_size)
            },
            |logits| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_sampling_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_argmax(kernel, queue, agent, logits, batch_size, vocab_size) {
                            return result;
                        }
                    }
                }
                crate::ops::sampling::argmax(logits, batch_size, vocab_size)
            },
            |logits| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_sampling_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_argmax(kernel, queue, agent, logits, batch_size, vocab_size) {
                            return result;
                        }
                    }
                }
                crate::ops::sampling::argmax(logits, batch_size, vocab_size)
            },
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_moe_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_moe_route(
                            kernel, queue, agent, hidden_states, gate_weights, batch_size, seq_len, config,
                        ) {
                            return result;
                        }
                    }
                }
                crate::ops::moe_routing::moe_route(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_moe_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_moe_route(
                            kernel, queue, agent, hidden_states, gate_weights, batch_size, seq_len, config,
                        ) {
                            return result;
                        }
                    }
                }
                crate::ops::moe_routing::moe_route(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) =
                        (get_hsa_moe_kernel(), get_hsa_queue(), get_hsa_gpu_agent())
                    {
                        if let Some(result) = rocm_moe_route(
                            kernel, queue, agent, hidden_states, gate_weights, batch_size, seq_len, config,
                        ) {
                            return result;
                        }
                    }
                }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) = (
                        get_hsa_rms_norm_kernel(),
                        get_hsa_queue(),
                        get_hsa_gpu_agent(),
                    ) {
                        if rocm_rms_norm_f32(kernel, queue, agent, input, weight, output, batch, hidden, eps) {
                            return;
                        }
                    }
                }
                crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
            },
            |input, weight, output| {
                apply_f32_binary_out(input, weight, output, |input, weight, output| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_rms_norm_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_rms_norm_f32(
                                kernel,
                                queue,
                                agent,
                                input,
                                weight,
                                output,
                                batch,
                                hidden,
                                eps,
                            ) {
                                return;
                            }
                        }
                    }
                    crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
                });
            },
            |input, weight, output| {
                apply_f32_binary_out(input, weight, output, |input, weight, output| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_rms_norm_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_rms_norm_f32(
                                kernel,
                                queue,
                                agent,
                                input,
                                weight,
                                output,
                                batch,
                                hidden,
                                eps,
                            ) {
                                return;
                            }
                        }
                    }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) = (
                        get_hsa_rms_norm_kernel(),
                        get_hsa_queue(),
                        get_hsa_gpu_agent(),
                    ) {
                        if rocm_rms_norm_inplace_f32(kernel, queue, agent, data, weight, batch, hidden, eps) {
                            return;
                        }
                    }
                }
                crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
            },
            |data, weight| {
                apply_f32_inplace_weight(data, weight, |data, weight| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_rms_norm_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_rms_norm_inplace_f32(
                                kernel,
                                queue,
                                agent,
                                data,
                                weight,
                                batch,
                                hidden,
                                eps,
                            ) {
                                return;
                            }
                        }
                    }
                    crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
                });
            },
            |data, weight| {
                apply_f32_inplace_weight(data, weight, |data, weight| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_rms_norm_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_rms_norm_inplace_f32(
                                kernel,
                                queue,
                                agent,
                                data,
                                weight,
                                batch,
                                hidden,
                                eps,
                            ) {
                                return;
                            }
                        }
                    }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) = (
                        get_hsa_silu_kernel(),
                        get_hsa_queue(),
                        get_hsa_gpu_agent(),
                    ) {
                        if rocm_silu_inplace_f32(kernel, queue, agent, data) {
                            return;
                        }
                    }
                }
                crate::ops::activations::silu_inplace(data);
            },
            |data| {
                apply_f32_unary_inplace(data, |data| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_silu_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_silu_inplace_f32(kernel, queue, agent, data) {
                                return;
                            }
                        }
                    }
                    crate::ops::activations::silu_inplace(data);
                });
            },
            |data| {
                apply_f32_unary_inplace(data, |data| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_silu_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_silu_inplace_f32(kernel, queue, agent, data) {
                                return;
                            }
                        }
                    }
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
                #[cfg(target_os = "linux")]
                {
                    if let (Some(kernel), Some(queue), Some(agent)) = (
                        get_hsa_silu_kernel(),
                        get_hsa_queue(),
                        get_hsa_gpu_agent(),
                    ) {
                        if rocm_silu_f32(kernel, queue, agent, input, output) {
                            return;
                        }
                    }
                }
                crate::ops::activations::silu(input, output);
            },
            |input, output| {
                apply_f32_unary_out(input, output, |input, output| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_silu_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_silu_f32(kernel, queue, agent, input, output) {
                                return;
                            }
                        }
                    }
                    crate::ops::activations::silu(input, output);
                });
            },
            |input, output| {
                apply_f32_unary_out(input, output, |input, output| {
                    #[cfg(target_os = "linux")]
                    {
                        if let (Some(kernel), Some(queue), Some(agent)) = (
                            get_hsa_silu_kernel(),
                            get_hsa_queue(),
                            get_hsa_gpu_agent(),
                        ) {
                            if rocm_silu_f32(kernel, queue, agent, input, output) {
                                return;
                            }
                        }
                    }
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
        BackendType::Rocm
    }
}
