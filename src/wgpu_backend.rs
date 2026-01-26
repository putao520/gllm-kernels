use std::collections::HashMap;
use std::mem;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

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
use crate::wgpu_kernels::{
    FlashAttentionKernel as WgpuFlashAttentionKernel,
    PagedAttentionKernel as WgpuPagedAttentionKernel,
    RmsNormParams, WgpuRmsNorm,
    SiluParams, WgpuSilu,
};
use wgpu::util::DeviceExt;

struct WgpuDeviceQueue {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Global WGPU device/queue (lazy initialized).
static WGPU_DEVICE_QUEUE: OnceLock<Option<WgpuDeviceQueue>> = OnceLock::new();
/// Global WGPU flash attention kernel (lazy initialized).
static WGPU_FLASH_KERNEL: OnceLock<Option<WgpuFlashAttentionKernel>> = OnceLock::new();
/// Global WGPU paged attention kernel (lazy initialized).
static WGPU_PAGED_KERNEL: OnceLock<Option<WgpuPagedAttentionKernel>> = OnceLock::new();
/// Global WGPU RMSNorm kernel (lazy initialized).
static WGPU_RMSNORM_KERNEL: OnceLock<Option<WgpuRmsNorm>> = OnceLock::new();
/// Global WGPU SiLU kernel (lazy initialized).
static WGPU_SILU_KERNEL: OnceLock<Option<WgpuSilu>> = OnceLock::new();

/// Global buffer pool for GPU buffer reuse (P0-MEM-1).
static BUFFER_POOL: OnceLock<Mutex<BufferPool>> = OnceLock::new();

/// Global staging buffer pool for readback reuse (P0-MEM-2).
static STAGING_POOL: OnceLock<Mutex<StagingBufferPool>> = OnceLock::new();

/// Buffer pool configuration constants.
const MAX_POOL_SIZE_PER_BUCKET: usize = 8;
const MAX_TOTAL_POOL_BYTES: usize = 512 * 1024 * 1024; // 512 MB
const BUFFER_EXPIRY_SECS: u64 = 60;

/// Size bucket for buffer pooling (rounded to power of 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferSizeBucket(u64);

impl BufferSizeBucket {
    /// Round up to next power of 2, minimum 256 bytes.
    fn from_size(size: u64) -> Self {
        let min_size = 256u64;
        let size = size.max(min_size);
        let bucket = size.next_power_of_two();
        BufferSizeBucket(bucket)
    }

    fn size(&self) -> u64 {
        self.0
    }
}

/// Cached buffer entry with timestamp.
struct PooledBuffer {
    buffer: wgpu::Buffer,
    last_used: Instant,
}

/// Buffer pool for GPU buffer reuse (P0-MEM-1).
struct BufferPool {
    /// Buckets of buffers grouped by size.
    storage_buckets: HashMap<BufferSizeBucket, Vec<PooledBuffer>>,
    /// Current total bytes in pool.
    total_bytes: usize,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            storage_buckets: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Get or create a storage buffer with the given size and usage.
    fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> wgpu::Buffer {
        let bucket = BufferSizeBucket::from_size(size);

        // Try to get from pool
        if let Some(buffers) = self.storage_buckets.get_mut(&bucket) {
            if let Some(entry) = buffers.pop() {
                self.total_bytes = self.total_bytes.saturating_sub(bucket.size() as usize);
                log::trace!("BufferPool: reused buffer size={} label={}", bucket.size(), label);
                return entry.buffer;
            }
        }

        // Create new buffer
        log::trace!("BufferPool: created new buffer size={} label={}", bucket.size(), label);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bucket.size(),
            usage,
            mapped_at_creation: false,
        })
    }

    /// Release a buffer back to the pool.
    fn release(&mut self, buffer: wgpu::Buffer, original_size: u64) {
        let bucket = BufferSizeBucket::from_size(original_size);
        let bucket_size = bucket.size() as usize;

        // Check pool limits
        if self.total_bytes + bucket_size > MAX_TOTAL_POOL_BYTES {
            log::trace!("BufferPool: dropping buffer (pool full)");
            return; // Drop buffer instead of pooling
        }

        let buffers = self.storage_buckets.entry(bucket).or_insert_with(Vec::new);
        if buffers.len() >= MAX_POOL_SIZE_PER_BUCKET {
            log::trace!("BufferPool: dropping buffer (bucket full)");
            return; // Drop buffer instead of pooling
        }

        buffers.push(PooledBuffer {
            buffer,
            last_used: Instant::now(),
        });
        self.total_bytes += bucket_size;
    }

    /// Cleanup expired buffers.
    fn cleanup_expired(&mut self) {
        let expiry = Duration::from_secs(BUFFER_EXPIRY_SECS);
        let now = Instant::now();

        for (bucket, buffers) in self.storage_buckets.iter_mut() {
            let before_len = buffers.len();
            buffers.retain(|entry| now.duration_since(entry.last_used) < expiry);
            let removed = before_len - buffers.len();
            if removed > 0 {
                self.total_bytes = self.total_bytes.saturating_sub(removed * bucket.size() as usize);
            }
        }
    }
}

/// Staging buffer pool for readback operations (P0-MEM-2).
/// Uses size buckets to store reusable staging buffers.
struct StagingBufferPool {
    /// Buckets of staging buffers grouped by size.
    buckets: HashMap<BufferSizeBucket, Vec<wgpu::Buffer>>,
    /// Current total bytes in pool.
    total_bytes: usize,
}

impl StagingBufferPool {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Get or create a staging buffer of at least the given size.
    fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        min_size: u64,
        label: &'static str,
    ) -> wgpu::Buffer {
        let bucket = BufferSizeBucket::from_size(min_size);

        // Try to get from pool
        if let Some(buffers) = self.buckets.get_mut(&bucket) {
            if let Some(buffer) = buffers.pop() {
                self.total_bytes = self.total_bytes.saturating_sub(bucket.size() as usize);
                log::trace!("StagingBufferPool: reused buffer size={} label={}", bucket.size(), label);
                return buffer;
            }
        }

        // Create new buffer
        log::trace!("StagingBufferPool: created new buffer size={} label={}", bucket.size(), label);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bucket.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Release a staging buffer back to the pool.
    fn release(&mut self, buffer: wgpu::Buffer, original_size: u64) {
        let bucket = BufferSizeBucket::from_size(original_size);
        let bucket_size = bucket.size() as usize;

        // Check pool limits (use same limits as BufferPool)
        if self.total_bytes + bucket_size > MAX_TOTAL_POOL_BYTES / 4 {
            log::trace!("StagingBufferPool: dropping buffer (pool full)");
            return;
        }

        let buffers = self.buckets.entry(bucket).or_insert_with(Vec::new);
        if buffers.len() >= MAX_POOL_SIZE_PER_BUCKET {
            log::trace!("StagingBufferPool: dropping buffer (bucket full)");
            return;
        }

        buffers.push(buffer);
        self.total_bytes += bucket_size;
    }
}

fn get_buffer_pool() -> &'static Mutex<BufferPool> {
    BUFFER_POOL.get_or_init(|| Mutex::new(BufferPool::new()))
}

fn get_staging_pool() -> &'static Mutex<StagingBufferPool> {
    STAGING_POOL.get_or_init(|| Mutex::new(StagingBufferPool::new()))
}

fn get_wgpu_device_queue() -> Option<&'static WgpuDeviceQueue> {
    WGPU_DEVICE_QUEUE
        .get_or_init(|| {
            let instance = wgpu::Instance::default();
            let adapter = match pollster::block_on(instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                },
            )) {
                Ok(adapter) => adapter,
                Err(e) => {
                    log::warn!("Failed to request WGPU adapter: {}", e);
                    return None;
                }
            };

            let (device, queue) = match pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gllm-wgpu-utils"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                },
            )) {
                Ok(result) => result,
                Err(e) => {
                    log::warn!("Failed to request WGPU device: {}", e);
                    return None;
                }
            };

            Some(WgpuDeviceQueue { device, queue })
        })
        .as_ref()
}

fn get_wgpu_flash_kernel() -> Option<&'static WgpuFlashAttentionKernel> {
    WGPU_FLASH_KERNEL
        .get_or_init(|| {
            match WgpuFlashAttentionKernel::create_default(false) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize WGPU flash attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_wgpu_paged_kernel() -> Option<&'static WgpuPagedAttentionKernel> {
    WGPU_PAGED_KERNEL
        .get_or_init(|| {
            match WgpuPagedAttentionKernel::create_default(false) {
                Ok(kernel) => Some(kernel),
                Err(e) => {
                    log::warn!("Failed to initialize WGPU paged attention kernel: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

fn get_wgpu_rmsnorm_kernel() -> Option<&'static WgpuRmsNorm> {
    WGPU_RMSNORM_KERNEL
        .get_or_init(|| {
            let device_queue = get_wgpu_device_queue()?;
            Some(WgpuRmsNorm::new(
                device_queue.device.clone(),
                device_queue.queue.clone(),
            ))
        })
        .as_ref()
}

fn get_wgpu_silu_kernel() -> Option<&'static WgpuSilu> {
    WGPU_SILU_KERNEL
        .get_or_init(|| {
            let device_queue = get_wgpu_device_queue()?;
            Some(WgpuSilu::new(
                device_queue.device.clone(),
                device_queue.queue.clone(),
            ))
        })
        .as_ref()
}

/// WGPU flash attention dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn wgpu_flash_attention<T: KernelFloat>(
    kernel: &WgpuFlashAttentionKernel,
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: &FlashAttentionConfig,
) -> bool {
    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let seq_len = config.seq_len_q;

    let result = kernel.forward_f32(
        &q_f32,
        &k_f32,
        &v_f32,
        config.batch_size,
        config.num_heads,
        seq_len,
        config.head_dim,
        None,
        scale,
    );

    match result {
        Ok(out_data) => {
            for (i, val) in out_data.into_iter().enumerate() {
                if i < output.len() {
                    output[i] = T::from_f32(val);
                }
            }
            true
        }
        Err(e) => {
            log::debug!("WGPU kernel execution failed: {}", e);
            false
        }
    }
}

/// WGPU paged attention dispatch.
/// Returns true if GPU execution succeeded, false to fallback to CPU.
fn wgpu_paged_attention<T: KernelFloat>(
    kernel: &WgpuPagedAttentionKernel,
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

    let result = kernel.forward_f32(
        &inputs.q_f32,
        &inputs.k_f32,
        &inputs.v_f32,
        &inputs.block_tables,
        &inputs.block_offsets,
        inputs.layout.batch_size,
        inputs.layout.num_heads,
        inputs.layout.head_dim,
        inputs.layout.page_size,
        inputs.layout.seq_len,
    );

    match result {
        Ok(out_data) => {
            for (i, value) in out_data.into_iter().enumerate() {
                if i < output.len() {
                    output[i] = T::from_f32(value);
                }
            }
            true
        }
        Err(e) => {
            log::debug!("WGPU paged kernel execution failed: {}", e);
            false
        }
    }
}

fn readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
    label: &'static str,
) -> Result<Vec<f32>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }

    let size_bytes = len
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| format!("{label}: readback size overflow"))?
        as wgpu::BufferAddress;

    // P0-MEM-2: Get staging buffer from pool
    let staging = {
        let mut pool = get_staging_pool()
            .lock()
            .map_err(|_| format!("{label}: staging pool lock poisoned"))?;
        pool.get_or_create(device, size_bytes, label)
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = receiver
        .recv()
        .map_err(|_| format!("{label}: map_async channel closed"))?;
    map_result.map_err(|err| format!("{label}: map_async failed: {err}"))?;

    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    if values.len() < len {
        // P0-MEM-2: Release staging buffer back to pool on error
        drop(data);
        staging.unmap();
        if let Ok(mut pool) = get_staging_pool().lock() {
            pool.release(staging, size_bytes);
        }
        return Err(format!("{label}: readback length mismatch"));
    }

    let mut output = vec![0.0f32; len];
    output.copy_from_slice(&values[..len]);
    drop(data);
    staging.unmap();

    // P0-MEM-2: Release staging buffer back to pool
    if let Ok(mut pool) = get_staging_pool().lock() {
        pool.release(staging, size_bytes);
    }

    Ok(output)
}

fn wgpu_rms_norm<T: KernelFloat>(
    kernel: &WgpuRmsNorm,
    input: &[T],
    weight: &[T],
    output: &mut [T],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> bool {
    if input.len() != batch.saturating_mul(hidden)
        || output.len() != batch.saturating_mul(hidden)
        || weight.len() != hidden
    {
        log::debug!("WGPU rms_norm dispatch skipped: length mismatch");
        return false;
    }

    let rows = match u32::try_from(batch) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let hidden_u32 = match u32::try_from(hidden) {
        Ok(v) => v,
        Err(_) => return false,
    };

    let device = kernel.device();
    let queue = kernel.queue();

    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rms_norm_input"),
        contents: bytemuck::cast_slice(&input_f32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rms_norm_weight"),
        contents: bytemuck::cast_slice(&weight_f32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_bytes = match output.len().checked_mul(mem::size_of::<f32>()) {
        Some(bytes) => bytes as wgpu::BufferAddress,
        None => return false,
    };
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms_norm_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = RmsNormParams {
        rows,
        hidden: hidden_u32,
        _pad0: 0,
        _pad1: 0,
        eps,
        _pad2: [0.0; 3],
    };

    kernel.forward(params, &input_buf, &weight_buf, &output_buf);
    let output_f32 = match readback_f32(device, queue, &output_buf, output.len(), "rms_norm_readback") {
        Ok(values) => values,
        Err(err) => {
            log::debug!("WGPU rms_norm readback failed: {}", err);
            return false;
        }
    };

    for (dst, val) in output.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }
    true
}

fn wgpu_rms_norm_inplace<T: KernelFloat>(
    kernel: &WgpuRmsNorm,
    data: &mut [T],
    weight: &[T],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> bool {
    if data.len() != batch.saturating_mul(hidden) || weight.len() != hidden {
        log::debug!("WGPU rms_norm_inplace dispatch skipped: length mismatch");
        return false;
    }

    let rows = match u32::try_from(batch) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let hidden_u32 = match u32::try_from(hidden) {
        Ok(v) => v,
        Err(_) => return false,
    };

    let device = kernel.device();
    let queue = kernel.queue();

    let data_f32: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();

    let data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rms_norm_data"),
        contents: bytemuck::cast_slice(&data_f32),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rms_norm_weight_inplace"),
        contents: bytemuck::cast_slice(&weight_f32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let params = RmsNormParams {
        rows,
        hidden: hidden_u32,
        _pad0: 0,
        _pad1: 0,
        eps,
        _pad2: [0.0; 3],
    };

    kernel.forward_inplace(params, &data_buf, &weight_buf);
    let output_f32 = match readback_f32(device, queue, &data_buf, data.len(), "rms_norm_inplace_readback") {
        Ok(values) => values,
        Err(err) => {
            log::debug!("WGPU rms_norm_inplace readback failed: {}", err);
            return false;
        }
    };

    for (dst, val) in data.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }
    true
}

fn wgpu_silu<T: KernelFloat>(
    kernel: &WgpuSilu,
    input: &[T],
    output: &mut [T],
) -> bool {
    if input.len() != output.len() {
        log::debug!("WGPU silu dispatch skipped: length mismatch");
        return false;
    }

    let len = match u32::try_from(input.len()) {
        Ok(v) => v,
        Err(_) => return false,
    };

    let device = kernel.device();
    let queue = kernel.queue();

    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("silu_input"),
        contents: bytemuck::cast_slice(&input_f32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_bytes = match output.len().checked_mul(mem::size_of::<f32>()) {
        Some(bytes) => bytes as wgpu::BufferAddress,
        None => return false,
    };
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = SiluParams {
        len,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    kernel.forward(params, &input_buf, &output_buf);
    let output_f32 = match readback_f32(device, queue, &output_buf, output.len(), "silu_readback") {
        Ok(values) => values,
        Err(err) => {
            log::debug!("WGPU silu readback failed: {}", err);
            return false;
        }
    };

    for (dst, val) in output.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }
    true
}

fn wgpu_silu_inplace<T: KernelFloat>(
    kernel: &WgpuSilu,
    data: &mut [T],
) -> bool {
    let len = match u32::try_from(data.len()) {
        Ok(v) => v,
        Err(_) => return false,
    };

    let device = kernel.device();
    let queue = kernel.queue();

    let data_f32: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("silu_data"),
        contents: bytemuck::cast_slice(&data_f32),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let params = SiluParams {
        len,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    kernel.forward_inplace(params, &data_buf);
    let output_f32 = match readback_f32(device, queue, &data_buf, data.len(), "silu_inplace_readback") {
        Ok(values) => values,
        Err(err) => {
            log::debug!("WGPU silu_inplace readback failed: {}", err);
            return false;
        }
    };

    for (dst, val) in data.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }
    true
}

pub struct WgpuBackend {}

impl WgpuBackend {
    pub fn new() -> Self {
        Self {}
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
        match_float3_out(
            "flash_attention",
            q,
            k,
            v,
            output,
            |q, k, v, out| {
                if let Some(kernel) = get_wgpu_flash_kernel() {
                    if wgpu_flash_attention(kernel, q, k, v, out, &config) {
                        return;
                    }
                    log::debug!("WGPU kernel dispatch failed, falling back to CPU");
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                if let Some(kernel) = get_wgpu_flash_kernel() {
                    if wgpu_flash_attention(kernel, q, k, v, out, &config) {
                        return;
                    }
                    log::debug!("WGPU kernel dispatch failed, falling back to CPU");
                }
                crate::ops::attention::cpu_flash_attention(q, k, v, out, config.clone());
            },
            |q, k, v, out| {
                if let Some(kernel) = get_wgpu_flash_kernel() {
                    if wgpu_flash_attention(kernel, q, k, v, out, &config) {
                        return;
                    }
                    log::debug!("WGPU kernel dispatch failed, falling back to CPU");
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
                if let Some(kernel) = get_wgpu_paged_kernel() {
                    if wgpu_paged_attention(kernel, q, k, v, page_table, seq_lens, out, &config) {
                        return;
                    }
                    log::debug!("WGPU paged attention dispatch failed, falling back to CPU");
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
                if let Some(kernel) = get_wgpu_paged_kernel() {
                    if wgpu_paged_attention(kernel, q, k, v, page_table, seq_lens, out, &config) {
                        return;
                    }
                    log::debug!("WGPU paged attention dispatch failed, falling back to CPU");
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
                if let Some(kernel) = get_wgpu_paged_kernel() {
                    if wgpu_paged_attention(kernel, q, k, v, page_table, seq_lens, out, &config) {
                        return;
                    }
                    log::debug!("WGPU paged attention dispatch failed, falling back to CPU");
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

    fn q4_matmul(
        &self,
        input: &[f32],
        q_weight: &[u8],
        scales: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
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
        crate::ops::quantized::q8_matmul_cpu(input, q_weight, scales, m, n, k)
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
                if let Some(kernel) = get_wgpu_rmsnorm_kernel() {
                    if wgpu_rms_norm(kernel, input, weight, output, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("WGPU rms_norm dispatch failed, falling back to CPU");
                }
                crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
            },
            |input, weight, output| {
                if let Some(kernel) = get_wgpu_rmsnorm_kernel() {
                    if wgpu_rms_norm(kernel, input, weight, output, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("WGPU rms_norm dispatch failed, falling back to CPU");
                }
                apply_f32_binary_out(input, weight, output, |input, weight, output| {
                    crate::ops::rms_norm::rms_norm_forward(input, weight, output, batch, hidden, eps);
                });
            },
            |input, weight, output| {
                if let Some(kernel) = get_wgpu_rmsnorm_kernel() {
                    if wgpu_rms_norm(kernel, input, weight, output, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("WGPU rms_norm dispatch failed, falling back to CPU");
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
                if let Some(kernel) = get_wgpu_rmsnorm_kernel() {
                    if wgpu_rms_norm_inplace(kernel, data, weight, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("WGPU rms_norm_inplace dispatch failed, falling back to CPU");
                }
                crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
            },
            |data, weight| {
                if let Some(kernel) = get_wgpu_rmsnorm_kernel() {
                    if wgpu_rms_norm_inplace(kernel, data, weight, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("WGPU rms_norm_inplace dispatch failed, falling back to CPU");
                }
                apply_f32_inplace_weight(data, weight, |data, weight| {
                    crate::ops::rms_norm::rms_norm_inplace(data, weight, batch, hidden, eps);
                });
            },
            |data, weight| {
                if let Some(kernel) = get_wgpu_rmsnorm_kernel() {
                    if wgpu_rms_norm_inplace(kernel, data, weight, batch, hidden, eps) {
                        return;
                    }
                    log::debug!("WGPU rms_norm_inplace dispatch failed, falling back to CPU");
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
                if let Some(kernel) = get_wgpu_silu_kernel() {
                    if wgpu_silu_inplace(kernel, data) {
                        return;
                    }
                    log::debug!("WGPU silu_inplace dispatch failed, falling back to CPU");
                }
                crate::ops::activations::silu_inplace(data);
            },
            |data| {
                if let Some(kernel) = get_wgpu_silu_kernel() {
                    if wgpu_silu_inplace(kernel, data) {
                        return;
                    }
                    log::debug!("WGPU silu_inplace dispatch failed, falling back to CPU");
                }
                apply_f32_unary_inplace(data, |data| {
                    crate::ops::activations::silu_inplace(data);
                });
            },
            |data| {
                if let Some(kernel) = get_wgpu_silu_kernel() {
                    if wgpu_silu_inplace(kernel, data) {
                        return;
                    }
                    log::debug!("WGPU silu_inplace dispatch failed, falling back to CPU");
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
                if let Some(kernel) = get_wgpu_silu_kernel() {
                    if wgpu_silu(kernel, input, output) {
                        return;
                    }
                    log::debug!("WGPU silu dispatch failed, falling back to CPU");
                }
                crate::ops::activations::silu(input, output);
            },
            |input, output| {
                if let Some(kernel) = get_wgpu_silu_kernel() {
                    if wgpu_silu(kernel, input, output) {
                        return;
                    }
                    log::debug!("WGPU silu dispatch failed, falling back to CPU");
                }
                apply_f32_unary_out(input, output, |input, output| {
                    crate::ops::activations::silu(input, output);
                });
            },
            |input, output| {
                if let Some(kernel) = get_wgpu_silu_kernel() {
                    if wgpu_silu(kernel, input, output) {
                        return;
                    }
                    log::debug!("WGPU silu dispatch failed, falling back to CPU");
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
        BackendType::Wgpu
    }
}
