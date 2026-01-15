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
    find_gpu_agents, is_hsa_available, HsaBuffer, HsaFlashAttentionKernel,
    HsaPagedAttentionKernel, HsaQueueWrapper,
};

use crate::cuda_kernels::{
    FlashAttentionKernel as CudaFlashAttentionKernel,
    PagedAttentionKernel as CudaPagedAttentionKernel,
};
use crate::wgpu_kernels::{
    FlashAttentionKernel as WgpuFlashAttentionKernel,
    PagedAttentionKernel as WgpuPagedAttentionKernel,
};
#[cfg(target_os = "macos")]
use crate::metal_kernels::{
    FlashAttentionKernel as MetalFlashAttentionKernel,
    PagedAttentionKernel as MetalPagedAttentionKernel,
};
use cudarc::driver::{CudaContext, CudaStream};

use std::sync::OnceLock;

/// Global HSA kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_KERNEL: OnceLock<Option<HsaFlashAttentionKernel>> = OnceLock::new();

/// Global HSA paged attention kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_PAGED_KERNEL: OnceLock<Option<HsaPagedAttentionKernel>> = OnceLock::new();

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
fn get_hsa_paged_kernel() -> Option<&'static HsaPagedAttentionKernel> {
    HSA_PAGED_KERNEL.get_or_init(|| {
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

/// Global CUDA paged attention kernel (lazy initialized)
static CUDA_PAGED_KERNEL: OnceLock<Option<CudaPagedAttentionKernel>> = OnceLock::new();

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

fn get_cuda_paged_kernel() -> Option<&'static CudaPagedAttentionKernel> {
    CUDA_PAGED_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaPagedAttentionKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA paged attention kernel: {}", e);
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

/// Global WGPU paged attention kernel (lazy initialized)
static WGPU_PAGED_KERNEL: OnceLock<Option<WgpuPagedAttentionKernel>> = OnceLock::new();

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

fn get_wgpu_paged_kernel() -> Option<&'static WgpuPagedAttentionKernel> {
    WGPU_PAGED_KERNEL.get_or_init(|| {
        match WgpuPagedAttentionKernel::create_default(false) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize WGPU paged attention kernel: {}", e);
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
static METAL_PAGED_KERNEL: OnceLock<Option<MetalPagedAttentionKernel>> = OnceLock::new();

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

#[cfg(target_os = "macos")]
fn get_metal_paged_kernel() -> Option<&'static MetalPagedAttentionKernel> {
    METAL_PAGED_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalPagedAttentionKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal paged attention kernel: {}", e);
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
    pub fn paged_attention<T: KernelFloat>(
        &self,
        q: &[T],
        k_cache: &[T],
        v_cache: &[T],
        page_table: &[u32],
        seq_lens: &[u32],
        output: &mut [T],
        config: PagedAttentionConfig,
    ) {
        match self.backend {
            #[cfg(target_os = "linux")]
            BackendType::Rocm => {
                if let (Some(kernel), Some(queue)) = (get_hsa_paged_kernel(), get_hsa_queue()) {
                    if rocm_paged_attention(
                        kernel,
                        queue,
                        q,
                        k_cache,
                        v_cache,
                        page_table,
                        seq_lens,
                        output,
                        &config,
                    ) {
                        return;
                    }
                    log::debug!("HSA paged attention dispatch failed, falling back to CPU");
                }
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
            #[cfg(not(target_os = "linux"))]
            BackendType::Rocm => {
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_paged_kernel(), get_cuda_stream()) {
                    if cuda_paged_attention(
                        kernel,
                        stream,
                        q,
                        k_cache,
                        v_cache,
                        page_table,
                        seq_lens,
                        output,
                        &config,
                    ) {
                        return;
                    }
                    log::debug!("CUDA paged attention dispatch failed, falling back to CPU");
                }
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_paged_kernel() {
                    if wgpu_paged_attention(
                        kernel,
                        q,
                        k_cache,
                        v_cache,
                        page_table,
                        seq_lens,
                        output,
                        &config,
                    ) {
                        return;
                    }
                    log::debug!("WGPU paged attention dispatch failed, falling back to CPU");
                }
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_paged_kernel(), get_metal_device()) {
                    if metal_paged_attention(
                        kernel,
                        device,
                        q,
                        k_cache,
                        v_cache,
                        page_table,
                        seq_lens,
                        output,
                        &config,
                    ) {
                        return;
                    }
                    log::debug!("Metal paged attention dispatch failed, falling back to CPU");
                }
                cpu_paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
            }
            #[cfg(not(target_os = "macos"))]
            BackendType::Metal => {
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

#[derive(Clone, Copy)]
struct PagedAttentionLayout {
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_kv_len: usize,
    page_size: usize,
    num_blocks: usize,
    scale: f32,
}

struct PagedGpuInputs {
    layout: PagedAttentionLayout,
    q_f32: Vec<f32>,
    k_f32: Vec<f32>,
    v_f32: Vec<f32>,
    block_tables: Vec<i32>,
    block_offsets: Vec<i32>,
}

fn build_paged_layout<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output_len: usize,
    config: &PagedAttentionConfig,
) -> Option<PagedAttentionLayout> {
    if output_len == 0 {
        return None;
    }
    if q.is_empty() {
        log::warn!("Paged attention: empty query input");
        return None;
    }
    if q.len() != output_len {
        log::warn!("Paged attention: output length mismatch");
        return None;
    }
    let batch_size = seq_lens.len();
    if batch_size == 0 {
        log::warn!("Paged attention: seq_lens is empty");
        return None;
    }
    let num_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let page_size = config.page_size;
    if num_heads == 0 || head_dim == 0 || page_size == 0 {
        log::warn!("Paged attention: invalid config (zeros in heads/head_dim/page_size)");
        return None;
    }
    let tokens_per_query = match num_heads
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(batch_size))
    {
        Some(value) => value,
        None => {
            log::warn!("Paged attention: dimension overflow");
            return None;
        }
    };
    if q.len() % tokens_per_query != 0 {
        log::warn!("Paged attention: q length mismatch");
        return None;
    }
    let seq_len = q.len() / tokens_per_query;
    if seq_len == 0 {
        log::warn!("Paged attention: seq_len is zero");
        return None;
    }
    if k_cache.len() != v_cache.len() {
        log::warn!("Paged attention: K/V cache length mismatch");
        return None;
    }
    let block_stride = match page_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(head_dim))
    {
        Some(value) => value,
        None => {
            log::warn!("Paged attention: block stride overflow");
            return None;
        }
    };
    if block_stride == 0 || k_cache.len() % block_stride != 0 {
        log::warn!("Paged attention: KV cache stride mismatch");
        return None;
    }
    let num_blocks = k_cache.len() / block_stride;
    if num_blocks == 0 {
        log::warn!("Paged attention: KV cache is empty");
        return None;
    }
    if page_table.len() % batch_size != 0 {
        log::warn!("Paged attention: page_table length mismatch");
        return None;
    }
    let max_kv_len = page_table.len() / batch_size;
    if max_kv_len == 0 {
        log::warn!("Paged attention: page_table is empty");
        return None;
    }
    if seq_lens
        .iter()
        .any(|&kv_len| kv_len as usize > max_kv_len)
    {
        log::warn!("Paged attention: seq_lens exceed page_table length");
        return None;
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    Some(PagedAttentionLayout {
        batch_size,
        num_heads,
        head_dim,
        seq_len,
        max_kv_len,
        page_size,
        num_blocks,
        scale,
    })
}

fn build_paged_gpu_inputs<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output_len: usize,
    config: &PagedAttentionConfig,
) -> Option<PagedGpuInputs> {
    let layout = build_paged_layout(q, k_cache, v_cache, page_table, seq_lens, output_len, config)?;
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

    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_cache.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_cache.iter().map(|x| x.to_f32()).collect();

    Some(PagedGpuInputs {
        layout,
        q_f32,
        k_f32,
        v_f32,
        block_tables,
        block_offsets,
    })
}

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
fn cpu_paged_attention<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [T],
    config: PagedAttentionConfig,
) {
    let layout = match build_paged_layout(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        output.len(),
        &config,
    ) {
        Some(layout) => layout,
        None => {
            output.iter_mut().for_each(|v| *v = T::zero());
            return;
        }
    };

    let acc_config = if config.use_log_space_softmax || config.use_kahan_accumulator {
        AccumulatorConfig::max_precision()
    } else {
        AccumulatorConfig::short_context()
    };

    let batch_stride = layout.num_heads * layout.seq_len * layout.head_dim;
    let mut invalid_block_logged = false;

    for b in 0..layout.batch_size {
        let kv_len = seq_lens[b] as usize;
        if kv_len == 0 {
            let base = b * batch_stride;
            output[base..base + batch_stride].iter_mut().for_each(|v| *v = T::zero());
            continue;
        }
        if kv_len > layout.max_kv_len || kv_len < layout.seq_len {
            if !invalid_block_logged {
                log::warn!("Paged attention: invalid kv length for batch {}", b);
                invalid_block_logged = true;
            }
            let base = b * batch_stride;
            output[base..base + batch_stride].iter_mut().for_each(|v| *v = T::zero());
            continue;
        }

        let position_offset = kv_len - layout.seq_len;
        let table_base = b * layout.max_kv_len;
        let num_blocks = (kv_len + layout.page_size - 1) / layout.page_size;

        for h in 0..layout.num_heads {
            for q_pos in 0..layout.seq_len {
                let q_base = ((b * layout.num_heads + h) * layout.seq_len + q_pos) * layout.head_dim;
                let mut q_local = vec![0.0f32; layout.head_dim];
                for d in 0..layout.head_dim {
                    q_local[d] = q[q_base + d].to_f32();
                }

                let q_abs = position_offset + q_pos;
                let mut stable = StableAccumulator::new(acc_config.clone());
                let mut block_scores = Vec::with_capacity(layout.page_size);

                for block_idx in 0..num_blocks {
                    let token_base = block_idx * layout.page_size;
                    if token_base >= kv_len {
                        break;
                    }
                    let block_id = page_table[table_base + token_base] as usize;
                    if block_id >= layout.num_blocks {
                        if !invalid_block_logged {
                            log::warn!("Paged attention: page_table references invalid block id");
                            invalid_block_logged = true;
                        }
                        continue;
                    }

                    block_scores.clear();
                    let tokens_in_block = (kv_len - token_base).min(layout.page_size);
                    for t in 0..tokens_in_block {
                        let k_idx = token_base + t;
                        if k_idx > q_abs {
                            continue;
                        }
                        let kv_base =
                            ((block_id * layout.page_size + t) * layout.num_heads + h) * layout.head_dim;
                        let mut score = 0.0f32;
                        for d in 0..layout.head_dim {
                            score += q_local[d] * k_cache[kv_base + d].to_f32();
                        }
                        score *= layout.scale;
                        block_scores.push(score);
                    }

                    if block_scores.is_empty() {
                        continue;
                    }
                    let mut block_max = f64::NEG_INFINITY;
                    for &score in &block_scores {
                        block_max = block_max.max(score as f64);
                    }
                    let mut block_sum_exp = 0.0f64;
                    for &score in &block_scores {
                        block_sum_exp += ((score as f64) - block_max).exp();
                    }
                    stable.update(block_max, block_sum_exp);
                }

                let m = stable.max();
                let l = stable.sum();
                let mut weighted_sum: Vec<KahanAccumulator<f32>> =
                    vec![KahanAccumulator::new(); layout.head_dim];

                for block_idx in 0..num_blocks {
                    let token_base = block_idx * layout.page_size;
                    if token_base >= kv_len {
                        break;
                    }
                    let block_id = page_table[table_base + token_base] as usize;
                    if block_id >= layout.num_blocks {
                        continue;
                    }

                    let tokens_in_block = (kv_len - token_base).min(layout.page_size);
                    for t in 0..tokens_in_block {
                        let k_idx = token_base + t;
                        if k_idx > q_abs {
                            continue;
                        }
                        let kv_base =
                            ((block_id * layout.page_size + t) * layout.num_heads + h) * layout.head_dim;
                        let mut score = 0.0f32;
                        for d in 0..layout.head_dim {
                            score += q_local[d] * k_cache[kv_base + d].to_f32();
                        }
                        score *= layout.scale;

                        let attn_weight = if l > 0.0 {
                            (((score as f64) - m).exp() / l) as f32
                        } else {
                            0.0
                        };
                        for d in 0..layout.head_dim {
                            weighted_sum[d].add(attn_weight * v_cache[kv_base + d].to_f32());
                        }
                    }
                }

                for d in 0..layout.head_dim {
                    output[q_base + d] = T::from_f32(weighted_sum[d].value());
                }
            }
        }
    }
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

// =============================================================================
// Paged Attention GPU Dispatch Functions
// =============================================================================

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
    let inputs = match build_paged_gpu_inputs(
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

    let agent = kernel.agent();

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
    let inputs = match build_paged_gpu_inputs(
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
    let inputs = match build_paged_gpu_inputs(
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

    let inputs = match build_paged_gpu_inputs(
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
            let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            for (i, &value) in out_slice.iter().enumerate() {
                if i < output.len() {
                    output[i] = T::from_f32(value);
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
    fn test_paged_attention_cpu() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        let q = vec![half::f16::from_f32(1.0); 4];
        let k_cache = vec![half::f16::from_f32(1.0); 4];
        let v_cache = vec![half::f16::from_f32(1.0); 4];
        let page_table = vec![0u32; 2];
        let seq_lens = vec![2u32];
        let mut output = vec![half::f16::ZERO; 4];

        dispatcher.paged_attention(
            &q,
            &k_cache,
            &v_cache,
            &page_table,
            &seq_lens,
            &mut output,
            PagedAttentionConfig {
                page_size: 2,
                num_kv_heads: 1,
                head_dim: 2,
                ..Default::default()
            },
        );

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
