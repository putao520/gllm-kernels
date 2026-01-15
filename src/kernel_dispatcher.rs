//! Zero-cost kernel dispatcher with runtime backend selection.
//!
//! Uses generics for f16/f32 support - compile-time monomorphization = zero cost.
//!
//! ## GPU Backend Integration
//!
//! - **CUDA**: Uses cudarc for dynamic loading (PTX at runtime)
//! - **ROCm**: Uses HSA Runtime via `hsa_flash_attn` module (HSACO at runtime)
//! - **Metal**: Uses metal-rs for metallib loading
//! - **WGPU**: Uses wgpu for SPIRV execution

use crate::runtime_detection::{BackendType, detect_backend};
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::hip_kernels::{
    is_hsa_available, find_gpu_agents,
    HsaFlashAttentionKernel, HsaBuffer, HsaQueueWrapper,
};

use crate::cuda_kernels::FlashAttentionKernel as CudaFlashAttentionKernel;
use crate::wgpu_kernels::FlashAttentionKernel as WgpuFlashAttentionKernel;
#[cfg(target_os = "macos")]
use crate::metal_kernels::FlashAttentionKernel as MetalFlashAttentionKernel;
use cudarc::driver::{CudaContext, CudaStream};

use std::sync::OnceLock;

/// Global HSA kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_KERNEL: OnceLock<Option<HsaFlashAttentionKernel>> = OnceLock::new();

/// Global HSA queue (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_QUEUE: OnceLock<Option<HsaQueueWrapper>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn get_hsa_kernel() -> Option<&'static HsaFlashAttentionKernel> {
    HSA_KERNEL.get_or_init(|| {
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
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_queue() -> Option<&'static HsaQueueWrapper> {
    HSA_QUEUE.get_or_init(|| {
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
    }).as_ref()
}

// =============================================================================
// CUDA Lazy Initialization
// =============================================================================

/// Global CUDA context (lazy initialized)
static CUDA_CONTEXT: OnceLock<Option<Arc<CudaContext>>> = OnceLock::new();

/// Global CUDA stream (lazy initialized)
static CUDA_STREAM: OnceLock<Option<Arc<CudaStream>>> = OnceLock::new();

/// Global CUDA flash attention kernel (lazy initialized)
static CUDA_KERNEL: OnceLock<Option<CudaFlashAttentionKernel>> = OnceLock::new();

fn get_cuda_context() -> Option<&'static Arc<CudaContext>> {
    CUDA_CONTEXT.get_or_init(|| {
        match CudaContext::new(0) {
            // cudarc 0.18: new() returns Result<Arc<CudaContext>>
            Ok(ctx) => Some(ctx),
            Err(e) => {
                log::warn!("Failed to create CUDA context: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_stream() -> Option<&'static Arc<CudaStream>> {
    CUDA_STREAM.get_or_init(|| {
        let ctx = get_cuda_context()?;
        // cudarc 0.18: default_stream() returns Arc<CudaStream> directly
        Some(ctx.default_stream())
    }).as_ref()
}

fn get_cuda_kernel() -> Option<&'static CudaFlashAttentionKernel> {
    CUDA_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaFlashAttentionKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA flash attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

// =============================================================================
// WGPU Lazy Initialization
// =============================================================================

/// Global WGPU flash attention kernel (lazy initialized)
static WGPU_KERNEL: OnceLock<Option<WgpuFlashAttentionKernel>> = OnceLock::new();

fn get_wgpu_kernel() -> Option<&'static WgpuFlashAttentionKernel> {
    WGPU_KERNEL.get_or_init(|| {
        // Initialize WGPU kernel (does not require f16 support)
        match WgpuFlashAttentionKernel::create_default(false) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize WGPU flash attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

// =============================================================================
// Metal Lazy Initialization (macOS only)
// =============================================================================

#[cfg(target_os = "macos")]
static METAL_KERNEL: OnceLock<Option<MetalFlashAttentionKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_DEVICE: OnceLock<Option<metal::Device>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn get_metal_device() -> Option<&'static metal::Device> {
    METAL_DEVICE.get_or_init(|| {
        metal::Device::system_default()
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_kernel() -> Option<&'static MetalFlashAttentionKernel> {
    METAL_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalFlashAttentionKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal flash attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

/// Trait for kernel-compatible floating point types.
/// Implemented for f32 and half::f16. Zero-cost via monomorphization.
pub trait KernelFloat: Copy + Default + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn zero() -> Self;
}

impl KernelFloat for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 { self }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { v }
    #[inline(always)]
    fn zero() -> Self { 0.0 }
}

impl KernelFloat for half::f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 { half::f16::to_f32(self) }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { half::f16::from_f32(v) }
    #[inline(always)]
    fn zero() -> Self { half::f16::ZERO }
}

/// Configuration for Flash Attention kernel.
#[derive(Clone, Debug)]
pub struct FlashAttentionConfig {
    /// Query block size for tiling.
    pub block_size_q: usize,
    /// Key/Value block size for tiling.
    pub block_size_kv: usize,
    /// Whether to apply causal mask.
    pub causal: bool,
    /// Use log-space softmax for 2M+ context stability.
    pub use_log_space_softmax: bool,
    /// Use Kahan compensated summation for numerical stability.
    pub use_kahan_accumulator: bool,
    /// Dropout probability (0.0 = no dropout).
    pub dropout_prob: f32,
    /// Optional scale factor (default: 1/sqrt(head_dim)).
    pub scale: Option<f32>,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Sequence length for query.
    pub seq_len_q: usize,
    /// Sequence length for key/value.
    pub seq_len_kv: usize,
    /// Batch size.
    pub batch_size: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 128,
            block_size_kv: 128,
            causal: true,
            use_log_space_softmax: false,
            use_kahan_accumulator: false,
            dropout_prob: 0.0,
            scale: None,
            num_heads: 1,
            head_dim: 64,
            seq_len_q: 1,
            seq_len_kv: 1,
            batch_size: 1,
        }
    }
}

/// Configuration for Paged Attention kernel.
#[derive(Clone, Debug)]
pub struct PagedAttentionConfig {
    /// Number of tokens per page.
    pub page_size: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Block size for computation.
    pub block_size: usize,
    /// Use log-space softmax for 2M+ context stability.
    pub use_log_space_softmax: bool,
    /// Use Kahan compensated summation.
    pub use_kahan_accumulator: bool,
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            num_kv_heads: 1,
            head_dim: 64,
            block_size: 128,
            use_log_space_softmax: true, // Enable by default for safety
            use_kahan_accumulator: true,
        }
    }
}

/// Configuration for Softmax kernel.
#[derive(Clone, Debug)]
pub struct SoftmaxConfig {
    /// Use log-space computation for numerical stability.
    pub use_log_space: bool,
    /// Use Kahan compensated summation.
    pub use_kahan: bool,
    /// Axis along which to compute softmax.
    pub axis: i32,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self {
            use_log_space: true,
            use_kahan: true,
            axis: -1,
        }
    }
}

/// Zero-cost kernel dispatcher with runtime backend selection.
///
/// This struct holds the detected backend type and provides methods
/// to dispatch kernels to the appropriate implementation.
///
/// # Example
///
/// ```ignore
/// use gllm_kernels::{KernelDispatcher, FlashAttentionConfig};
/// use half::f16;
///
/// let dispatcher = KernelDispatcher::new();
///
/// // Get slices from your tensor library
/// let q: &[f16] = &[/* query data */];
/// let k: &[f16] = &[/* key data */];
/// let v: &[f16] = &[/* value data */];
/// let mut output = vec![f16::ZERO; output_len];
///
/// dispatcher.flash_attention(
///     q, k, v,
///     &mut output,
///     FlashAttentionConfig {
///         use_log_space_softmax: true,
///         use_kahan_accumulator: true,
///         ..Default::default()
///     },
/// );
/// ```
pub struct KernelDispatcher {
    backend: BackendType,
}

impl KernelDispatcher {
    /// Create a new dispatcher with auto-detected backend.
    ///
    /// Detection priority: CUDA > ROCm > Metal > WGPU > CPU
    /// Zero configuration - automatically selects the best available backend.
    pub fn new() -> Self {
        Self {
            backend: detect_backend(),
        }
    }

    /// Create a dispatcher with a specific backend.
    pub fn with_backend(backend: BackendType) -> Self {
        Self { backend }
    }

    /// Get the current backend type.
    #[inline]
    pub fn backend(&self) -> BackendType {
        self.backend
    }

    /// Flash Attention kernel dispatch (generic over f16/f32).
    /// Zero-cost via monomorphization.
    #[inline(always)]
    pub fn flash_attention<T: KernelFloat>(
        &self,
        q: &[T],
        k: &[T],
        v: &[T],
        output: &mut [T],
        config: FlashAttentionConfig,
    ) {
        match self.backend {
            #[cfg(target_os = "linux")]
            BackendType::Rocm => {
                // Try HSA flash attention on AMD GPUs
                if let (Some(kernel), Some(queue)) = (get_hsa_kernel(), get_hsa_queue()) {
                    if rocm_flash_attention(kernel, queue, q, k, v, output, &config) {
                        return;
                    }
                    log::debug!("HSA kernel dispatch failed, falling back to CPU");
                }
                cpu_flash_attention(q, k, v, output, config);
            }
            #[cfg(not(target_os = "linux"))]
            BackendType::Rocm => {
                cpu_flash_attention(q, k, v, output, config);
            }
            BackendType::Cuda => {
                // Try CUDA flash attention via cudarc
                if let (Some(kernel), Some(stream)) = (get_cuda_kernel(), get_cuda_stream()) {
                    if cuda_flash_attention(kernel, stream, q, k, v, output, &config) {
                        return;
                    }
                    log::debug!("CUDA kernel dispatch failed, falling back to CPU");
                }
                cpu_flash_attention(q, k, v, output, config);
            }
            BackendType::Wgpu => {
                // Try WGPU flash attention via wgpu
                if let Some(kernel) = get_wgpu_kernel() {
                    if wgpu_flash_attention(kernel, q, k, v, output, &config) {
                        return;
                    }
                    log::debug!("WGPU kernel dispatch failed, falling back to CPU");
                }
                cpu_flash_attention(q, k, v, output, config);
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                // Try Metal flash attention on macOS
                if let (Some(kernel), Some(device)) = (get_metal_kernel(), get_metal_device()) {
                    if metal_flash_attention(kernel, device, q, k, v, output, &config) {
                        return;
                    }
                    log::debug!("Metal kernel dispatch failed, falling back to CPU");
                }
                cpu_flash_attention(q, k, v, output, config);
            }
            #[cfg(not(target_os = "macos"))]
            BackendType::Metal => {
                // Metal not available on non-macOS
                cpu_flash_attention(q, k, v, output, config);
            }
            BackendType::Cpu => {
                cpu_flash_attention(q, k, v, output, config);
            }
        }
    }

    /// Paged Attention kernel dispatch.
    ///
    /// # Arguments
    /// * `q` - Query tensor as flat slice
    /// * `k_cache` - Key cache pages
    /// * `v_cache` - Value cache pages
    /// * `page_table` - Logical to physical page mapping
    /// * `output` - Output buffer
    /// * `config` - Kernel configuration
    ///
    /// # Note
    /// GPU kernels are loaded at runtime via cudarc/metal/wgpu.
    /// Currently falls back to CPU reference implementation.
    /// GPU dispatch will be added when kernels are ready.
    #[inline(always)]
    pub fn paged_attention(
        &self,
        q: &[half::f16],
        k_cache: &[half::f16],
        v_cache: &[half::f16],
        page_table: &[u32],
        seq_lens: &[u32],
        output: &mut [half::f16],
        config: PagedAttentionConfig,
    ) {
        match self.backend {
            // GPU backends will use runtime-loaded kernels when available
            // For now, fall back to CPU reference implementation
            BackendType::Cuda | BackendType::Rocm | BackendType::Metal | BackendType::Wgpu => {
                // TODO: Implement GPU paged attention dispatch via cuda_kernels::PagedAttentionKernel
                // The CUDA kernel is available in cuda_kernels::paged_attn module
                // Need to add CudaContext and stream management
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
            BackendType::Cpu => {
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
        }
    }

    /// Softmax kernel dispatch with 2M context stability.
    ///
    /// Softmax kernel dispatch (generic over f16/f32).
    #[inline(always)]
    pub fn softmax<T: KernelFloat>(&self, input: &[T], output: &mut [T], config: SoftmaxConfig) {
        // All backends use the generic CPU implementation for now
        cpu_softmax(input, output, config);
    }
}

impl Default for KernelDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CPU Reference Implementations (always available)
// =============================================================================

use crate::ops::stable_accumulator::{AccumulatorConfig, KahanAccumulator, StableAccumulator};

/// CPU reference implementation of Flash Attention (generic).
#[inline(always)]
fn cpu_flash_attention<T: KernelFloat>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: FlashAttentionConfig,
) {
    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let batch = config.batch_size;
    let heads = config.num_heads;
    let seq_q = config.seq_len_q;
    let seq_kv = config.seq_len_kv;
    let head_dim = config.head_dim;

    let acc_config = if config.use_log_space_softmax || config.use_kahan_accumulator {
        AccumulatorConfig::max_precision()
    } else {
        AccumulatorConfig::short_context()
    };

    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                let mut stable = StableAccumulator::new(acc_config.clone());
                let max_j = if config.causal { i + 1 } else { seq_kv };
                let mut scores = Vec::with_capacity(max_j.min(seq_kv));
                let mut block_max = f64::NEG_INFINITY;

                // First pass: compute scores
                for j in 0..max_j.min(seq_kv) {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = b * heads * seq_q * head_dim + h * seq_q * head_dim + i * head_dim + d;
                        let k_idx = b * heads * seq_kv * head_dim + h * seq_kv * head_dim + j * head_dim + d;
                        score += q[q_idx].to_f32() * k[k_idx].to_f32();
                    }
                    score *= scale;
                    block_max = block_max.max(score as f64);
                    scores.push(score);
                }

                if !scores.is_empty() {
                    let block_sum_exp: f64 = scores.iter().map(|&s| ((s as f64) - block_max).exp()).sum();
                    stable.update(block_max, block_sum_exp);
                }

                let m = stable.max();
                let l = stable.sum();

                // Second pass: weighted sum
                let mut weighted_sum: Vec<KahanAccumulator<f32>> = vec![KahanAccumulator::new(); head_dim];

                for (j, &score) in scores.iter().enumerate() {
                    let attn_weight = if l > 0.0 { (((score as f64) - m).exp() / l) as f32 } else { 0.0 };
                    for d in 0..head_dim {
                        let v_idx = b * heads * seq_kv * head_dim + h * seq_kv * head_dim + j * head_dim + d;
                        weighted_sum[d].add(attn_weight * v[v_idx].to_f32());
                    }
                }

                // Write output
                for d in 0..head_dim {
                    let out_idx = b * heads * seq_q * head_dim + h * seq_q * head_dim + i * head_dim + d;
                    output[out_idx] = T::from_f32(weighted_sum[d].value());
                }
            }
        }
    }
}

/// CPU reference implementation of Paged Attention.
fn cpu_paged_attention(
    _q: &[half::f16],
    _k_cache: &[half::f16],
    _v_cache: &[half::f16],
    _page_table: &[u32],
    _seq_lens: &[u32],
    _output: &mut [half::f16],
    _config: PagedAttentionConfig,
) {
    // TODO: Implement CPU paged attention
    log::warn!("CPU paged attention not yet implemented");
}

/// CPU reference implementation of Softmax (generic).
#[inline(always)]
fn cpu_softmax<T: KernelFloat>(input: &[T], output: &mut [T], config: SoftmaxConfig) {
    if input.is_empty() {
        return;
    }

    let acc_config = if config.use_log_space || config.use_kahan {
        AccumulatorConfig::max_precision()
    } else {
        AccumulatorConfig::short_context()
    };

    let mut max_val = f64::NEG_INFINITY;
    for &x in input {
        max_val = max_val.max(x.to_f32() as f64);
    }

    if config.use_log_space || config.use_kahan {
        let mut stable = StableAccumulator::new(acc_config);
        let sum_exp: f64 = input.iter().map(|&x| ((x.to_f32() as f64) - max_val).exp()).sum();
        stable.update(max_val, sum_exp);

        let m = stable.max();
        let l = stable.sum();

        for (i, &x) in input.iter().enumerate() {
            let prob = if l > 0.0 { (((x.to_f32() as f64) - m).exp() / l) as f32 } else { 0.0 };
            output[i] = T::from_f32(prob);
        }
    } else {
        let max_f32 = input.iter().map(|x| x.to_f32()).fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = input.iter().map(|x| (x.to_f32() - max_f32).exp()).sum();
        for (i, &x) in input.iter().enumerate() {
            output[i] = T::from_f32((x.to_f32() - max_f32).exp() / sum);
        }
    }
}

// =============================================================================
// GPU Kernel Dispatch Functions
// =============================================================================

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

    // Convert input to f32 for HSA kernel
    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    // Allocate HSA buffers
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

    // Calculate scale factor (default: 1/sqrt(head_dim))
    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());

    // HSA kernel expects same seq_len for Q and KV (square attention)
    // Use seq_len_q for self-attention scenarios
    let seq_len = config.seq_len_q;

    // Execute kernel
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
        0, // position_offset (for RoPE, not used in basic flash attn)
    );

    match result {
        Ok(out_buf) => {
            // Copy result back to host
            match out_buf.to_vec() {
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
            }
        }
        Err(e) => {
            log::debug!("HSA kernel execution failed: {}", e);
            false
        }
    }
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
    // Convert input to f32 for CUDA kernel
    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    // Copy data to device (host-to-device)
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

    // Calculate scale factor (default: 1/sqrt(head_dim))
    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());

    // CUDA kernel expects same seq_len for Q and KV
    let seq_len = config.seq_len_q;

    // Execute kernel
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
        0, // position_offset (for RoPE, not used in basic flash attn)
    );

    match result {
        Ok(out_buf) => {
            // Copy result back to host (device-to-host)
            match stream.clone_dtoh(&out_buf) {
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
            }
        }
        Err(e) => {
            log::debug!("CUDA kernel execution failed: {}", e);
            false
        }
    }
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
    // Convert input to f32 for WGPU kernel
    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    // Calculate scale factor (default: 1/sqrt(head_dim))
    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());

    // WGPU kernel expects same seq_len for Q and KV
    let seq_len = config.seq_len_q;

    // Execute kernel
    let result = kernel.forward_f32(
        &q_f32,
        &k_f32,
        &v_f32,
        config.batch_size,
        config.num_heads,
        seq_len,
        config.head_dim,
        None, // Use default block size
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

    // Convert input to f32 for Metal kernel
    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();

    // Calculate buffer size in bytes
    let buffer_size = (q_f32.len() * std::mem::size_of::<f32>()) as u64;

    // Create Metal buffers
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

    // Calculate scale factor (default: 1/sqrt(head_dim))
    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());

    // Metal kernel expects same seq_len for Q and KV
    let seq_len = config.seq_len_q;

    // Execute kernel
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
        0, // position_offset
    );

    match result {
        Ok(out_buf) => {
            // Copy result back to host
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatcher_creation() {
        let dispatcher = KernelDispatcher::new();
        // Should default to CPU on most test environments
        println!("Detected backend: {:?}", dispatcher.backend());
    }

    #[test]
    fn test_flash_attention_cpu() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        // Simple 1x1x2x4 test case
        let q = vec![half::f16::from_f32(1.0); 8];
        let k = vec![half::f16::from_f32(1.0); 8];
        let v = vec![half::f16::from_f32(1.0); 8];
        let mut output = vec![half::f16::ZERO; 8];

        dispatcher.flash_attention(
            &q,
            &k,
            &v,
            &mut output,
            FlashAttentionConfig {
                batch_size: 1,
                num_heads: 1,
                seq_len_q: 2,
                seq_len_kv: 2,
                head_dim: 4,
                causal: false,
                use_log_space_softmax: true,
                use_kahan_accumulator: true,
                ..Default::default()
            },
        );

        // Output should be all 1.0 (average of all-1 values)
        for &o in &output {
            let val = o.to_f32();
            assert!(
                (val - 1.0).abs() < 0.01,
                "Expected ~1.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_softmax_cpu() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        let input: Vec<half::f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let mut output = vec![half::f16::ZERO; 4];

        dispatcher.softmax(
            &input,
            &mut output,
            SoftmaxConfig {
                use_log_space: true,
                use_kahan: true,
                axis: -1,
            },
        );

        // Verify sum is approximately 1.0
        let sum: f32 = output.iter().map(|x| x.to_f32()).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax sum should be ~1.0, got {}",
            sum
        );

        // Verify monotonicity (larger input -> larger output)
        for i in 0..3 {
            assert!(
                output[i + 1].to_f32() > output[i].to_f32(),
                "Softmax should preserve order"
            );
        }
    }
}
