use crate::backend_match::{
    apply_f32_binary_out, apply_f32_inplace_weight, apply_f32_unary_inplace, apply_f32_unary_out,
    match_float1, match_float1_mut, match_float1_mut_weight, match_float1_out,
    match_float2_out, match_float2_out2, match_float3_out,
};
use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
use crate::kernel_types::{
    FlashAttentionConfig, KernelFloat, MatmulConfig, PagedAttentionConfig, SoftmaxConfig,
};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};
use crate::ops::rope::RoPEConfig;
use crate::ops::sampling::{SamplingConfig, TopKResult};
use crate::runtime_detection::BackendType;

#[cfg(target_os = "macos")]
use std::sync::OnceLock;

#[cfg(target_os = "macos")]
use crate::metal_kernels::{
    FlashAttentionKernel as MetalFlashAttentionKernel,
    PagedAttentionKernel as MetalPagedAttentionKernel,
};

#[cfg(target_os = "macos")]
static METAL_FLASH_KERNEL: OnceLock<Option<MetalFlashAttentionKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_PAGED_KERNEL: OnceLock<Option<MetalPagedAttentionKernel>> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL_DEVICE: OnceLock<Option<metal::Device>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn get_metal_device() -> Option<&'static metal::Device> {
    METAL_DEVICE
        .get_or_init(|| metal::Device::system_default())
        .as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_flash_kernel() -> Option<&'static MetalFlashAttentionKernel> {
    METAL_FLASH_KERNEL
        .get_or_init(|| {
            let device = get_metal_device()?;
            match MetalFlashAttentionKernel::new(device) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize Metal flash attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_paged_kernel() -> Option<&'static MetalPagedAttentionKernel> {
    METAL_PAGED_KERNEL
        .get_or_init(|| {
            let device = get_metal_device()?;
            match MetalPagedAttentionKernel::new(device) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize Metal paged attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

/// Metal flash attention dispatch (macOS only).
/// Returns true if GPU execution succeeded, false to fallback to CPU.
#[cfg(target_os = "macos")]
fn metal_flash_attention<T: KernelFloat>(
    kernel: &MetalFlashAttentionKernel,
    device: &metal::Device,
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: &FlashAttentionConfig,
) -> bool {
    use metal::MTLResourceOptions;

    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    let buffer_size = (q_f32.len() * std::mem::size_of::<f32>()) as u64;

    let q_buf = device.new_buffer_with_data(
        q_f32.as_ptr() as *const _,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_f32.as_ptr() as *const _,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_f32.as_ptr() as *const _,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let seq_len = config.seq_len_q;

    let result = kernel.forward_f32(
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
        Ok(out_buf) => {
            let out_ptr = out_buf.contents() as *const f32;
            let out_len = output.len();
            unsafe {
                let out_slice = std::slice::from_raw_parts(out_ptr, out_len);
                for (i, &val) in out_slice.iter().enumerate() {
                    if i < output.len() {
                        output[i] = T::from_f32(val);
                    }
                }
            }
            true
        }
        Err(e) => {
            log::debug!("Metal kernel execution failed: {}", e);
            false
        }
    }
}

/// Metal paged attention dispatch (macOS only).
/// Returns true if GPU execution succeeded, false to fallback to CPU.
#[cfg(target_os = "macos")]
fn metal_paged_attention<T: KernelFloat>(
    kernel: &MetalPagedAttentionKernel,
    device: &metal::Device,
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [T],
    config: &PagedAttentionConfig,
) -> bool {
    use metal::MTLResourceOptions;

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

    let q_buf = device.new_buffer_with_data(
        inputs.q_f32.as_ptr() as *const _,
        (inputs.q_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        inputs.k_f32.as_ptr() as *const _,
        (inputs.k_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        inputs.v_f32.as_ptr() as *const _,
        (inputs.v_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let table_buf = device.new_buffer_with_data(
        inputs.block_tables.as_ptr() as *const _,
        (inputs.block_tables.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = device.new_buffer_with_data(
        inputs.block_offsets.as_ptr() as *const _,
        (inputs.block_offsets.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let result = kernel.forward_f32(
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
        Ok(out_buf) => {
            let out_ptr = out_buf.contents() as *const f32;
            let out_len = output.len();
            unsafe {
                let out_slice = std::slice::from_raw_parts(out_ptr, out_len);
                for (i, &val) in out_slice.iter().enumerate() {
                    if i < output.len() {
                        output[i] = T::from_f32(val);
                    }
                }
            }
            true
        }
        Err(e) => {
            log::debug!("Metal paged kernel execution failed: {}", e);
            false
        }
    }
}

pub struct MetalBackend {}

impl MetalBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MetalBackend {
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
                #[cfg(target_os = "macos")]
                {
                    if let (Some(kernel), Some(device)) = (get_metal_flash_kernel(), get_metal_device()) {
                        if metal_flash_attention(kernel, device, q, k, v, out, &config) {
                            return;
                        }
                        log::debug!("Metal kernel dispatch failed, falling back to CPU");
                    }
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                #[cfg(target_os = "macos")]
                {
                    if let (Some(kernel), Some(device)) = (get_metal_flash_kernel(), get_metal_device()) {
                        if metal_flash_attention(kernel, device, q, k, v, out, &config) {
                            return;
                        }
                        log::debug!("Metal kernel dispatch failed, falling back to CPU");
                    }
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                #[cfg(target_os = "macos")]
                {
                    if let (Some(kernel), Some(device)) = (get_metal_flash_kernel(), get_metal_device()) {
                        if metal_flash_attention(kernel, device, q, k, v, out, &config) {
                            return;
                        }
                        log::debug!("Metal kernel dispatch failed, falling back to CPU");
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
                #[cfg(target_os = "macos")]
                {
                    if let (Some(kernel), Some(device)) = (get_metal_paged_kernel(), get_metal_device()) {
                        if metal_paged_attention(kernel, device, q, k, v, page_table, seq_lens, out, &config) {
                            return;
                        }
                        log::debug!("Metal paged attention dispatch failed, falling back to CPU");
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
                #[cfg(target_os = "macos")]
                {
                    if let (Some(kernel), Some(device)) = (get_metal_paged_kernel(), get_metal_device()) {
                        if metal_paged_attention(kernel, device, q, k, v, page_table, seq_lens, out, &config) {
                            return;
                        }
                        log::debug!("Metal paged attention dispatch failed, falling back to CPU");
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
                #[cfg(target_os = "macos")]
                {
                    if let (Some(kernel), Some(device)) = (get_metal_paged_kernel(), get_metal_device()) {
                        if metal_paged_attention(kernel, device, q, k, v, page_table, seq_lens, out, &config) {
                            return;
                        }
                        log::debug!("Metal paged attention dispatch failed, falling back to CPU");
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
                crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
            },
            |input, weight, output| {
                apply_f32_binary_out(input, weight, output, |input, weight, output| {
                    crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
                });
            },
            |input, weight, output| {
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
                crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
            },
            |data, weight| {
                apply_f32_inplace_weight(data, weight, |data, weight| {
                    crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
                });
            },
            |data, weight| {
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
            |data| crate::ops::activations::silu_inplace(data),
            |data| {
                apply_f32_unary_inplace(data, |data| {
                    crate::ops::activations::silu_inplace(data);
                });
            },
            |data| {
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
            |input, output| crate::ops::activations::silu(input, output),
            |input, output| {
                apply_f32_unary_out(input, output, |input, output| {
                    crate::ops::activations::silu(input, output);
                });
            },
            |input, output| {
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
        BackendType::Metal
    }
}
