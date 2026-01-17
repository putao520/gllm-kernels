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
    HsaPagedAttentionKernel, HsaQueueWrapper, HsaEmbeddingOpsKernel,
    HsaEagle3Kernel, HsaEagle3Config, HsaSpecEEKernel, HsaSpecEEConfig,
    HsaFlashTreeAttnKernel, HsaFlashTreeAttnConfig, HsaInt2QuantizerKernel,
    HsaInt2QuantizerConfig, HsaEvicPressKernel, HsaEvicPressConfig,
    HsaMedusaKernel, HsaMedusaConfig, HsaPromptCacheKernel, HsaPromptCacheConfig,
    HsaChunkedPrefillKernel, HsaChunkedPrefillConfig,
};

use crate::cuda_kernels::{
    FlashAttentionKernel as CudaFlashAttentionKernel,
    PagedAttentionKernel as CudaPagedAttentionKernel,
    CudaEmbeddingOpsKernel,
    Eagle3Kernel as CudaEagle3Kernel,
    SpecEEKernel as CudaSpecEEKernel,
    FlashTreeAttnKernel as CudaFlashTreeAttnKernel,
    Int2QuantizerKernel as CudaInt2QuantizerKernel,
    EvicPressKernel as CudaEvicPressKernel,
    MedusaKernel as CudaMedusaKernel,
    PromptCacheKernel as CudaPromptCacheKernel,
    ChunkedPrefillKernel as CudaChunkedPrefillKernel,
};
use crate::wgpu_kernels::{
    FlashAttentionKernel as WgpuFlashAttentionKernel,
    PagedAttentionKernel as WgpuPagedAttentionKernel,
    EmbeddingOpsKernel as WgpuEmbeddingOpsKernel,
    Eagle3Kernel as WgpuEagle3Kernel,
    SpecEEKernel as WgpuSpecEEKernel,
    FlashTreeAttn as WgpuFlashTreeAttn,
    WgpuInt2Quantizer,
    WgpuEvicPress,
    WgpuMedusa,
    WgpuPromptCache,
    WgpuChunkedPrefill,
};

// Re-export rerank types for public API
pub use crate::wgpu_kernels::{GpuRerankConfig, GpuRerankStageResult};

#[cfg(target_os = "macos")]
use crate::metal_kernels::{
    FlashAttentionKernel as MetalFlashAttentionKernel,
    PagedAttentionKernel as MetalPagedAttentionKernel,
    MetalEmbeddingOpsKernel,
    Eagle3Kernel as MetalEagle3Kernel,
    SpecEEKernel as MetalSpecEEKernel,
    FlashTreeAttnKernel as MetalFlashTreeAttnKernel,
    Int2QuantizerKernel as MetalInt2QuantizerKernel,
    EvicPressKernel as MetalEvicPressKernel,
    MedusaKernel as MetalMedusaKernel,
    PromptCacheKernel as MetalPromptCacheKernel,
    ChunkedPrefillKernel as MetalChunkedPrefillKernel,
};
use cudarc::driver::{CudaContext, CudaStream};

use std::sync::OnceLock;

/// Global HSA kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_KERNEL: OnceLock<Option<HsaFlashAttentionKernel>> = OnceLock::new();

/// Global HSA paged attention kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_PAGED_KERNEL: OnceLock<Option<HsaPagedAttentionKernel>> = OnceLock::new();

/// Global HSA embedding ops kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_EMBEDDING_KERNEL: OnceLock<Option<HsaEmbeddingOpsKernel>> = OnceLock::new();

/// Global HSA EAGLE-3 kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_EAGLE3_KERNEL: OnceLock<Option<HsaEagle3Kernel>> = OnceLock::new();

/// Global HSA SpecEE kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_SPEC_EE_KERNEL: OnceLock<Option<HsaSpecEEKernel>> = OnceLock::new();

/// Global HSA Flash Tree-attention kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_FLASH_TREE_KERNEL: OnceLock<Option<HsaFlashTreeAttnKernel>> = OnceLock::new();

/// Global HSA INT2 quantizer kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_INT2_KERNEL: OnceLock<Option<HsaInt2QuantizerKernel>> = OnceLock::new();

/// Global HSA EvicPress kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_EVICT_PRESS_KERNEL: OnceLock<Option<HsaEvicPressKernel>> = OnceLock::new();

/// Global HSA Medusa kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_MEDUSA_KERNEL: OnceLock<Option<HsaMedusaKernel>> = OnceLock::new();

/// Global HSA Prompt Cache kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_PROMPT_CACHE_KERNEL: OnceLock<Option<HsaPromptCacheKernel>> = OnceLock::new();

/// Global HSA Chunked Prefill kernel instance (lazy initialized)
#[cfg(target_os = "linux")]
static HSA_CHUNKED_PREFILL_KERNEL: OnceLock<Option<HsaChunkedPrefillKernel>> = OnceLock::new();

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
fn get_hsa_embedding_kernel() -> Option<&'static HsaEmbeddingOpsKernel> {
    HSA_EMBEDDING_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaEmbeddingOpsKernel::new(0) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA embedding ops kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_eagle3_kernel() -> Option<&'static HsaEagle3Kernel> {
    HSA_EAGLE3_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaEagle3Kernel::new(HsaEagle3Config::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA EAGLE-3 kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_spec_ee_kernel() -> Option<&'static HsaSpecEEKernel> {
    HSA_SPEC_EE_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaSpecEEKernel::new(HsaSpecEEConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA SpecEE kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_flash_tree_kernel() -> Option<&'static HsaFlashTreeAttnKernel> {
    HSA_FLASH_TREE_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaFlashTreeAttnKernel::new(HsaFlashTreeAttnConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA flash tree attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_int2_kernel() -> Option<&'static HsaInt2QuantizerKernel> {
    HSA_INT2_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaInt2QuantizerKernel::new(HsaInt2QuantizerConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA INT2 quantizer kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_evic_press_kernel() -> Option<&'static HsaEvicPressKernel> {
    HSA_EVICT_PRESS_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaEvicPressKernel::new(HsaEvicPressConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA EvicPress kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_medusa_kernel() -> Option<&'static HsaMedusaKernel> {
    HSA_MEDUSA_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaMedusaKernel::new(HsaMedusaConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA Medusa kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_prompt_cache_kernel() -> Option<&'static HsaPromptCacheKernel> {
    HSA_PROMPT_CACHE_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaPromptCacheKernel::new(HsaPromptCacheConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA prompt cache kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "linux")]
fn get_hsa_chunked_prefill_kernel() -> Option<&'static HsaChunkedPrefillKernel> {
    HSA_CHUNKED_PREFILL_KERNEL.get_or_init(|| {
        if !is_hsa_available() {
            return None;
        }
        match HsaChunkedPrefillKernel::new(HsaChunkedPrefillConfig::default()) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize HSA chunked prefill kernel: {}", e);
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

/// Global CUDA embedding ops kernel (lazy initialized)
static CUDA_EMBEDDING_KERNEL: OnceLock<Option<CudaEmbeddingOpsKernel>> = OnceLock::new();

/// Global CUDA EAGLE-3 kernel (lazy initialized)
static CUDA_EAGLE3_KERNEL: OnceLock<Option<CudaEagle3Kernel>> = OnceLock::new();

/// Global CUDA SpecEE kernel (lazy initialized)
static CUDA_SPEC_EE_KERNEL: OnceLock<Option<CudaSpecEEKernel>> = OnceLock::new();

/// Global CUDA Flash Tree-attention kernel (lazy initialized)
static CUDA_FLASH_TREE_KERNEL: OnceLock<Option<CudaFlashTreeAttnKernel>> = OnceLock::new();

/// Global CUDA INT2 quantizer kernel (lazy initialized)
static CUDA_INT2_KERNEL: OnceLock<Option<CudaInt2QuantizerKernel>> = OnceLock::new();

/// Global CUDA EvicPress kernel (lazy initialized)
static CUDA_EVICT_PRESS_KERNEL: OnceLock<Option<CudaEvicPressKernel>> = OnceLock::new();

/// Global CUDA Medusa kernel (lazy initialized)
static CUDA_MEDUSA_KERNEL: OnceLock<Option<CudaMedusaKernel>> = OnceLock::new();

/// Global CUDA Prompt Cache kernel (lazy initialized)
static CUDA_PROMPT_CACHE_KERNEL: OnceLock<Option<CudaPromptCacheKernel>> = OnceLock::new();

/// Global CUDA Chunked Prefill kernel (lazy initialized)
static CUDA_CHUNKED_PREFILL_KERNEL: OnceLock<Option<CudaChunkedPrefillKernel>> = OnceLock::new();

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

fn get_cuda_embedding_kernel() -> Option<&'static CudaEmbeddingOpsKernel> {
    CUDA_EMBEDDING_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaEmbeddingOpsKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA embedding ops kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_eagle3_kernel() -> Option<&'static CudaEagle3Kernel> {
    CUDA_EAGLE3_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaEagle3Kernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA EAGLE-3 kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_spec_ee_kernel() -> Option<&'static CudaSpecEEKernel> {
    CUDA_SPEC_EE_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaSpecEEKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA SpecEE kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_flash_tree_kernel() -> Option<&'static CudaFlashTreeAttnKernel> {
    CUDA_FLASH_TREE_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaFlashTreeAttnKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA flash tree attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_int2_kernel() -> Option<&'static CudaInt2QuantizerKernel> {
    CUDA_INT2_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaInt2QuantizerKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA INT2 quantizer kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_evic_press_kernel() -> Option<&'static CudaEvicPressKernel> {
    CUDA_EVICT_PRESS_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaEvicPressKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA EvicPress kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_medusa_kernel() -> Option<&'static CudaMedusaKernel> {
    CUDA_MEDUSA_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaMedusaKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA Medusa kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_prompt_cache_kernel() -> Option<&'static CudaPromptCacheKernel> {
    CUDA_PROMPT_CACHE_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaPromptCacheKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA prompt cache kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_cuda_chunked_prefill_kernel() -> Option<&'static CudaChunkedPrefillKernel> {
    CUDA_CHUNKED_PREFILL_KERNEL.get_or_init(|| {
        let ctx = get_cuda_context()?;
        match CudaChunkedPrefillKernel::new(ctx) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize CUDA chunked prefill kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

// =============================================================================
// WGPU Lazy Initialization
// =============================================================================

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

static WGPU_CONTEXT: OnceLock<Option<WgpuContext>> = OnceLock::new();

fn get_wgpu_context() -> Option<&'static WgpuContext> {
    WGPU_CONTEXT.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;

        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }

        let limits = wgpu::Limits::default();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gllm-wgpu-kernels"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        )).ok()?;

        Some(WgpuContext { device, queue })
    }).as_ref()
}

/// Global WGPU flash attention kernel (lazy initialized)
static WGPU_KERNEL: OnceLock<Option<WgpuFlashAttentionKernel>> = OnceLock::new();

/// Global WGPU paged attention kernel (lazy initialized)
static WGPU_PAGED_KERNEL: OnceLock<Option<WgpuPagedAttentionKernel>> = OnceLock::new();

/// Global WGPU embedding ops kernel (lazy initialized)
static WGPU_EMBEDDING_KERNEL: OnceLock<Option<WgpuEmbeddingOpsKernel>> = OnceLock::new();

/// Global WGPU EAGLE-3 kernel (lazy initialized)
static WGPU_EAGLE3_KERNEL: OnceLock<Option<WgpuEagle3Kernel>> = OnceLock::new();

/// Global WGPU SpecEE kernel (lazy initialized)
static WGPU_SPEC_EE_KERNEL: OnceLock<Option<WgpuSpecEEKernel>> = OnceLock::new();

/// Global WGPU Flash Tree-attention kernel (lazy initialized)
static WGPU_FLASH_TREE_KERNEL: OnceLock<Option<WgpuFlashTreeAttn>> = OnceLock::new();

/// Global WGPU INT2 quantizer kernel (lazy initialized)
static WGPU_INT2_KERNEL: OnceLock<Option<WgpuInt2Quantizer>> = OnceLock::new();

/// Global WGPU EvicPress kernel (lazy initialized)
static WGPU_EVICT_PRESS_KERNEL: OnceLock<Option<WgpuEvicPress>> = OnceLock::new();

/// Global WGPU Medusa kernel (lazy initialized)
static WGPU_MEDUSA_KERNEL: OnceLock<Option<WgpuMedusa>> = OnceLock::new();

/// Global WGPU Prompt Cache kernel (lazy initialized)
static WGPU_PROMPT_CACHE_KERNEL: OnceLock<Option<WgpuPromptCache>> = OnceLock::new();

/// Global WGPU Chunked Prefill kernel (lazy initialized)
static WGPU_CHUNKED_PREFILL_KERNEL: OnceLock<Option<WgpuChunkedPrefill>> = OnceLock::new();

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

fn get_wgpu_embedding_kernel() -> Option<&'static WgpuEmbeddingOpsKernel> {
    WGPU_EMBEDDING_KERNEL.get_or_init(|| {
        match WgpuEmbeddingOpsKernel::create_default() {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize WGPU embedding ops kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_wgpu_eagle3_kernel() -> Option<&'static WgpuEagle3Kernel> {
    WGPU_EAGLE3_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        match WgpuEagle3Kernel::new(&ctx.device, &ctx.queue) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize WGPU EAGLE-3 kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_wgpu_spec_ee_kernel() -> Option<&'static WgpuSpecEEKernel> {
    WGPU_SPEC_EE_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        match WgpuSpecEEKernel::new(&ctx.device, &ctx.queue) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize WGPU SpecEE kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_wgpu_flash_tree_kernel() -> Option<&'static WgpuFlashTreeAttn> {
    WGPU_FLASH_TREE_KERNEL.get_or_init(|| {
        match WgpuFlashTreeAttn::new_sync() {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize WGPU flash tree attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

fn get_wgpu_int2_kernel() -> Option<&'static WgpuInt2Quantizer> {
    WGPU_INT2_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        Some(WgpuInt2Quantizer::new(ctx.device.clone(), ctx.queue.clone()))
    }).as_ref()
}

fn get_wgpu_evic_press_kernel() -> Option<&'static WgpuEvicPress> {
    WGPU_EVICT_PRESS_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        Some(WgpuEvicPress::new(ctx.device.clone(), ctx.queue.clone()))
    }).as_ref()
}

fn get_wgpu_medusa_kernel() -> Option<&'static WgpuMedusa> {
    WGPU_MEDUSA_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        Some(WgpuMedusa::new(ctx.device.clone(), ctx.queue.clone()))
    }).as_ref()
}

fn get_wgpu_prompt_cache_kernel() -> Option<&'static WgpuPromptCache> {
    WGPU_PROMPT_CACHE_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        Some(WgpuPromptCache::new(ctx.device.clone(), ctx.queue.clone()))
    }).as_ref()
}

fn get_wgpu_chunked_prefill_kernel() -> Option<&'static WgpuChunkedPrefill> {
    WGPU_CHUNKED_PREFILL_KERNEL.get_or_init(|| {
        let ctx = get_wgpu_context()?;
        Some(WgpuChunkedPrefill::new(ctx.device.clone(), ctx.queue.clone()))
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
static METAL_EMBEDDING_KERNEL: OnceLock<Option<MetalEmbeddingOpsKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_EAGLE3_KERNEL: OnceLock<Option<MetalEagle3Kernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_SPEC_EE_KERNEL: OnceLock<Option<MetalSpecEEKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_FLASH_TREE_KERNEL: OnceLock<Option<MetalFlashTreeAttnKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_INT2_KERNEL: OnceLock<Option<MetalInt2QuantizerKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_EVICT_PRESS_KERNEL: OnceLock<Option<MetalEvicPressKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_MEDUSA_KERNEL: OnceLock<Option<MetalMedusaKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_PROMPT_CACHE_KERNEL: OnceLock<Option<MetalPromptCacheKernel>> = OnceLock::new();

#[cfg(target_os = "macos")]
static METAL_CHUNKED_PREFILL_KERNEL: OnceLock<Option<MetalChunkedPrefillKernel>> = OnceLock::new();

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

#[cfg(target_os = "macos")]
fn get_metal_embedding_kernel() -> Option<&'static MetalEmbeddingOpsKernel> {
    METAL_EMBEDDING_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalEmbeddingOpsKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal embedding ops kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_eagle3_kernel() -> Option<&'static MetalEagle3Kernel> {
    METAL_EAGLE3_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalEagle3Kernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal EAGLE-3 kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_spec_ee_kernel() -> Option<&'static MetalSpecEEKernel> {
    METAL_SPEC_EE_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalSpecEEKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal SpecEE kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_flash_tree_kernel() -> Option<&'static MetalFlashTreeAttnKernel> {
    METAL_FLASH_TREE_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalFlashTreeAttnKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal flash tree attention kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_int2_kernel() -> Option<&'static MetalInt2QuantizerKernel> {
    METAL_INT2_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalInt2QuantizerKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal INT2 quantizer kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_evic_press_kernel() -> Option<&'static MetalEvicPressKernel> {
    METAL_EVICT_PRESS_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalEvicPressKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal EvicPress kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_medusa_kernel() -> Option<&'static MetalMedusaKernel> {
    METAL_MEDUSA_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalMedusaKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal Medusa kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_prompt_cache_kernel() -> Option<&'static MetalPromptCacheKernel> {
    METAL_PROMPT_CACHE_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalPromptCacheKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal prompt cache kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

#[cfg(target_os = "macos")]
fn get_metal_chunked_prefill_kernel() -> Option<&'static MetalChunkedPrefillKernel> {
    METAL_CHUNKED_PREFILL_KERNEL.get_or_init(|| {
        let device = get_metal_device()?;
        match MetalChunkedPrefillKernel::new(device) {
            Ok(kernel) => Some(kernel),
            Err(e) => {
                log::warn!("Failed to initialize Metal chunked prefill kernel: {}", e);
                None
            }
        }
    }).as_ref()
}

/// Float type identifier for const-time kernel selection (ADR-001).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatType {
    F32,
    F16,
    BF16,
}

impl FloatType {
    /// Convert to u8 for serialization (zero-cost).
    #[inline(always)]
    pub const fn as_u8(self) -> u8 {
        match self {
            FloatType::F32 => 0,
            FloatType::F16 => 1,
            FloatType::BF16 => 2,
        }
    }

    /// Convert from u8 for deserialization.
    #[inline(always)]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(FloatType::F32),
            1 => Some(FloatType::F16),
            2 => Some(FloatType::BF16),
            _ => None,
        }
    }
}

/// Trait for kernel-compatible floating point types.
/// Implemented for f32, half::f16, and half::bf16. Zero-cost via monomorphization.
pub trait KernelFloat: Copy + Default + Send + Sync + 'static {
    /// Compile-time type identifier for zero-cost kernel selection.
    const TYPE_ID: FloatType;

    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl KernelFloat for f32 {
    const TYPE_ID: FloatType = FloatType::F32;

    #[inline(always)]
    fn to_f32(self) -> f32 { self }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { v }
    #[inline(always)]
    fn zero() -> Self { 0.0 }
    #[inline(always)]
    fn one() -> Self { 1.0 }
    #[inline(always)]
    fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline(always)]
    fn exp(self) -> Self { f32::exp(self) }
    #[inline(always)]
    fn max(self, other: Self) -> Self { f32::max(self, other) }
}

impl KernelFloat for half::f16 {
    const TYPE_ID: FloatType = FloatType::F16;

    #[inline(always)]
    fn to_f32(self) -> f32 { half::f16::to_f32(self) }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { half::f16::from_f32(v) }
    #[inline(always)]
    fn zero() -> Self { half::f16::ZERO }
    #[inline(always)]
    fn one() -> Self { half::f16::ONE }
    #[inline(always)]
    fn sqrt(self) -> Self { half::f16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)]
    fn exp(self) -> Self { half::f16::from_f32(self.to_f32().exp()) }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }
}

impl KernelFloat for half::bf16 {
    const TYPE_ID: FloatType = FloatType::BF16;

    #[inline(always)]
    fn to_f32(self) -> f32 { half::bf16::to_f32(self) }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { half::bf16::from_f32(v) }
    #[inline(always)]
    fn zero() -> Self { half::bf16::ZERO }
    #[inline(always)]
    fn one() -> Self { half::bf16::ONE }
    #[inline(always)]
    fn sqrt(self) -> Self { half::bf16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)]
    fn exp(self) -> Self { half::bf16::from_f32(self.to_f32().exp()) }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }
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

/// Configuration for EAGLE-3 draft/verify dispatch.
#[derive(Clone, Debug)]
pub struct Eagle3Config {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub fusion_layers: usize,
    pub min_draft_len: usize,
    pub max_draft_len: usize,
    pub confidence_threshold: f32,
}

impl Default for Eagle3Config {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_dim: 768,
            fusion_layers: 4,
            min_draft_len: 1,
            max_draft_len: 8,
            confidence_threshold: 0.5,
        }
    }
}

impl Eagle3Config {
    fn fused_dim(&self) -> usize {
        self.hidden_dim * self.fusion_layers
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err("batch_size and seq_len must be > 0");
        }
        if self.hidden_dim == 0 || self.fusion_layers == 0 {
            return Err("hidden_dim and fusion_layers must be > 0");
        }
        if self.min_draft_len == 0 || self.max_draft_len < self.min_draft_len {
            return Err("invalid draft length range");
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err("confidence_threshold must be in [0, 1]");
        }
        Ok(())
    }
}

/// Result of EAGLE-3 draft generation.
#[derive(Clone, Debug)]
pub struct Eagle3DraftResult {
    pub draft_lengths: Vec<usize>,
    pub draft_tokens: Vec<i32>,
    pub max_draft_len: usize,
}

/// Configuration for EAGLE-3 verification.
#[derive(Clone, Debug)]
pub struct Eagle3VerifyConfig {
    pub batch_size: usize,
    pub draft_len: usize,
    pub vocab_size: usize,
}

/// Result of EAGLE-3 verification.
#[derive(Clone, Debug)]
pub struct Eagle3VerifyResult {
    pub accepted_lengths: Vec<usize>,
    pub acceptance_probs: Vec<f32>,
}

/// Configuration for SpecEE dispatch.
#[derive(Clone, Debug)]
pub struct SpecEEConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub skip_threshold: f32,
    pub exit_threshold: f32,
    pub current_layer: usize,
}

impl Default for SpecEEConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_dim: 768,
            skip_threshold: 0.9,
            exit_threshold: 0.95,
            current_layer: 0,
        }
    }
}

impl SpecEEConfig {
    fn validate(&self) -> Result<(), &'static str> {
        if self.batch_size == 0 || self.seq_len == 0 || self.hidden_dim == 0 {
            return Err("batch_size, seq_len, hidden_dim must be > 0");
        }
        if !(0.0..=1.0).contains(&self.skip_threshold) {
            return Err("skip_threshold must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.exit_threshold) {
            return Err("exit_threshold must be in [0, 1]");
        }
        Ok(())
    }
}

/// Result of SpecEE forward dispatch.
#[derive(Clone, Debug)]
pub struct SpecEEForwardResult {
    pub confidence: Vec<f32>,
    pub skip_decisions: Vec<i32>,
    pub should_exit: Vec<i32>,
    pub exit_layer: Vec<i32>,
}

/// Configuration for Flash Tree-attention dispatch.
#[derive(Clone, Debug)]
pub struct FlashTreeAttentionConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub tree_size: usize,
    pub prefix_len: usize,
    pub head_dim: usize,
    pub scale: Option<f32>,
}

impl Default for FlashTreeAttentionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 1,
            tree_size: 1,
            prefix_len: 0,
            head_dim: 64,
            scale: None,
        }
    }
}

/// Configuration for INT2 quantization dispatch.
#[derive(Clone, Debug)]
pub struct Int2QuantConfig {
    pub group_size: usize,
}

impl Default for Int2QuantConfig {
    fn default() -> Self {
        Self { group_size: 128 }
    }
}

/// Result of INT2 quantization.
#[derive(Clone, Debug)]
pub struct Int2QuantResult {
    pub quantized: Vec<i8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

/// Compression mode for EvicPress.
#[derive(Clone, Copy, Debug)]
pub enum EvicPressCompression {
    Int8,
    Int2,
}

/// Configuration for EvicPress compression.
#[derive(Clone, Debug)]
pub struct EvicPressCompressConfig {
    pub seq_len: usize,
    pub head_dim: usize,
    pub compression: EvicPressCompression,
}

/// Result of EvicPress compression.
#[derive(Clone, Debug)]
pub enum EvicPressCompressionResult {
    Int8 { data: Vec<i8>, scales: Vec<f32> },
    Int2 { data: Vec<u8>, scales: Vec<f32> },
}

/// Configuration for EvicPress eviction.
#[derive(Clone, Debug)]
pub struct EvicPressEvictConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub recency_weight: f32,
    pub attention_weight: f32,
    pub hot_threshold: f32,
    pub warm_threshold: f32,
    pub cache_pressure: f32,
}

impl Default for EvicPressEvictConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            recency_weight: 0.3,
            attention_weight: 0.7,
            hot_threshold: 0.8,
            warm_threshold: 0.4,
            cache_pressure: 0.0,
        }
    }
}

/// Result of EvicPress eviction decision.
#[derive(Clone, Debug)]
pub struct EvicPressEvictResult {
    pub importance: Vec<f32>,
    pub new_zones: Vec<i32>,
}

/// Configuration for Medusa forward dispatch.
#[derive(Clone, Debug)]
pub struct MedusaConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub top_k: usize,
    pub max_candidates: usize,
    pub temperature: f32,
}

impl Default for MedusaConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 4,
            vocab_size: 32000,
            top_k: 10,
            max_candidates: 64,
            temperature: 1.0,
        }
    }
}

/// Result of Medusa forward dispatch.
#[derive(Clone, Debug)]
pub struct MedusaForwardResult {
    pub candidate_tokens: Vec<i32>,
    pub candidate_probs: Vec<f32>,
    pub num_candidates: Vec<i32>,
}

/// Configuration for Medusa verification.
#[derive(Clone, Debug)]
pub struct MedusaVerifyConfig {
    pub batch_size: usize,
    pub num_candidates: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}

/// Result of Medusa verification.
#[derive(Clone, Debug)]
pub struct MedusaVerifyResult {
    pub accepted_lengths: Vec<i32>,
    pub best_candidate: Vec<i32>,
}

/// Configuration for prompt cache lookup.
#[derive(Clone, Debug)]
pub struct PromptCacheLookupConfig {
    pub num_entries: usize,
    pub max_cache_len: usize,
    pub hash_seed: u64,
    pub min_match_len: usize,
}

impl Default for PromptCacheLookupConfig {
    fn default() -> Self {
        Self {
            num_entries: 1024,
            max_cache_len: 4096,
            hash_seed: 0x9e3779b97f4a7c15,
            min_match_len: 32,
        }
    }
}

/// Result of prompt cache lookup.
#[derive(Clone, Debug)]
pub struct PromptCacheLookupResult {
    pub best_entry: i32,
    pub match_length: usize,
    pub query_hashes: Vec<u64>,
}

/// Configuration for prompt cache blending.
#[derive(Clone, Debug)]
pub struct PromptCacheBlendConfig {
    pub match_len: usize,
    pub fresh_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub blend_window: usize,
}

impl Default for PromptCacheBlendConfig {
    fn default() -> Self {
        Self {
            match_len: 0,
            fresh_len: 0,
            num_heads: 1,
            head_dim: 64,
            blend_window: 16,
        }
    }
}

/// Configuration for chunked prefill attention dispatch.
#[derive(Clone, Debug)]
pub struct ChunkedPrefillConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub query_len: usize,
    pub chunk_len: usize,
    pub head_dim: usize,
    pub chunk_start: usize,
    pub causal: bool,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 1,
            query_len: 1,
            chunk_len: 1,
            head_dim: 64,
            chunk_start: 0,
            causal: true,
        }
    }
}

/// Result of chunked prefill attention dispatch.
#[derive(Clone, Debug)]
pub struct ChunkedPrefillResult<T: KernelFloat> {
    pub output: Vec<T>,
    pub log_sum_exp: Vec<f32>,
}

/// Matrix multiplication configuration.
///
/// Supports C = A * B^T (transposed B is common for weight matrices).
#[derive(Clone, Debug)]
pub struct MatmulConfig {
    /// M dimension (rows of A, rows of C)
    pub m: usize,
    /// K dimension (cols of A, cols of B^T = rows of B)
    pub k: usize,
    /// N dimension (cols of B^T = cols of B, cols of C)
    pub n: usize,
    /// Whether B is stored transposed (common for weight matrices)
    pub transpose_b: bool,
    /// Alpha scalar multiplier (C = alpha * A * B + beta * C)
    pub alpha: f32,
    /// Beta scalar multiplier (0.0 means C is overwritten)
    pub beta: f32,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            m: 1,
            k: 1,
            n: 1,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
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

    // =========================================================================
    // Position Encoding Kernels (RoPE)
    // =========================================================================

    /// Precompute cos/sin frequency tables for RoPE.
    ///
    /// # Arguments
    /// * `cos_out` - Output buffer for cosine values: [max_seq_len, dim/2]
    /// * `sin_out` - Output buffer for sine values: [max_seq_len, dim/2]
    /// * `config` - RoPE configuration
    #[inline(always)]
    pub fn rope_precompute(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
        config: crate::ops::rope::RoPEConfig,
    ) {
        // All backends use the CPU implementation (precompute is typically done once)
        crate::ops::rope::rope_precompute(cos_out, sin_out, &config);
    }

    /// Apply RoPE to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor: [batch, seq, heads, head_dim]
    /// * `k` - Key tensor: [batch, seq, kv_heads, head_dim]
    /// * `cos_cache` - Precomputed cosine values: [max_seq, dim/2]
    /// * `sin_cache` - Precomputed sine values: [max_seq, dim/2]
    /// * `q_out` - Output query tensor (same shape as q)
    /// * `k_out` - Output key tensor (same shape as k)
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `num_q_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads
    /// * `head_dim` - Head dimension
    /// * `position_offset` - Starting position offset (for incremental decoding)
    #[inline(always)]
    pub fn rope_apply<T: KernelFloat>(
        &self,
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
    ) {
        // GPU backends can override with optimized kernels
        // For now, all backends use the CPU implementation
        crate::ops::rope::rope_apply(
            q, k, cos_cache, sin_cache, q_out, k_out,
            batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset,
        );
    }

    /// Apply RoPE to a single tensor in-place.
    ///
    /// # Arguments
    /// * `x` - Input/output tensor: [batch, seq, heads, head_dim]
    /// * `cos_cache` - Precomputed cosine values: [max_seq, dim/2]
    /// * `sin_cache` - Precomputed sine values: [max_seq, dim/2]
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of heads
    /// * `head_dim` - Head dimension
    /// * `position_offset` - Starting position offset
    #[inline(always)]
    pub fn rope_apply_inplace<T: KernelFloat>(
        &self,
        x: &mut [T],
        cos_cache: &[f32],
        sin_cache: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) {
        crate::ops::rope::rope_apply_inplace(
            x, cos_cache, sin_cache,
            batch_size, seq_len, num_heads, head_dim, position_offset,
        );
    }

    // =========================================================================
    // Sampling Kernels
    // =========================================================================

    /// Top-K selection: find k largest values and their indices.
    ///
    /// # Arguments
    /// * `logits` - Input logits: [batch_size, vocab_size]
    /// * `k` - Number of top elements to select
    /// * `batch_size` - Batch size
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// TopKResult containing indices and values for each batch element.
    #[inline(always)]
    pub fn topk<T: KernelFloat>(
        &self,
        logits: &[T],
        k: usize,
        batch_size: usize,
        vocab_size: usize,
    ) -> crate::ops::sampling::TopKResult {
        crate::ops::sampling::topk(logits, k, batch_size, vocab_size)
    }

    /// Apply temperature scaling to logits in-place.
    ///
    /// # Arguments
    /// * `logits` - Input/output logits to scale
    /// * `temperature` - Temperature value (>0, 1.0 = no change)
    #[inline(always)]
    pub fn apply_temperature<T: KernelFloat>(
        &self,
        logits: &mut [T],
        temperature: f32,
    ) {
        crate::ops::sampling::apply_temperature(logits, temperature);
    }

    /// Sample tokens from logits using configured sampling strategy.
    ///
    /// Supports temperature scaling, top-k filtering, and nucleus (top-p) sampling.
    ///
    /// # Arguments
    /// * `logits` - Input logits: [batch_size, vocab_size]
    /// * `batch_size` - Batch size
    /// * `vocab_size` - Vocabulary size
    /// * `config` - Sampling configuration
    ///
    /// # Returns
    /// Vector of sampled token indices, one per batch element.
    #[inline(always)]
    pub fn sample_tokens<T: KernelFloat>(
        &self,
        logits: &[T],
        batch_size: usize,
        vocab_size: usize,
        config: &crate::ops::sampling::SamplingConfig,
    ) -> Vec<u32> {
        crate::ops::sampling::sample_tokens(logits, batch_size, vocab_size, config)
    }

    /// Greedy decoding: select the token with highest logit.
    ///
    /// # Arguments
    /// * `logits` - Input logits: [batch_size, vocab_size]
    /// * `batch_size` - Batch size
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Vector of token indices with maximum logit, one per batch element.
    #[inline(always)]
    pub fn argmax<T: KernelFloat>(
        &self,
        logits: &[T],
        batch_size: usize,
        vocab_size: usize,
    ) -> Vec<u32> {
        crate::ops::sampling::argmax(logits, batch_size, vocab_size)
    }

    // =========================================================================
    // MoE (Mixture-of-Experts) Routing Kernels
    // =========================================================================

    /// Compute MoE routing: select top-k experts for each token.
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states: [batch * seq, hidden_size]
    /// * `gate_weights` - Router gate weights: [hidden_size, num_experts]
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `config` - Routing configuration
    ///
    /// # Returns
    /// MoERoutingResult with expert indices and normalized weights.
    #[inline(always)]
    pub fn moe_route<T: KernelFloat>(
        &self,
        hidden_states: &[T],
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &crate::ops::moe_routing::MoERoutingConfig,
    ) -> crate::ops::moe_routing::MoERoutingResult {
        crate::ops::moe_routing::moe_route(hidden_states, gate_weights, batch_size, seq_len, config)
    }

    /// Compute routing logits without selecting experts.
    ///
    /// Useful for routing analysis or custom expert selection.
    #[inline(always)]
    pub fn compute_routing_logits<T: KernelFloat>(
        &self,
        hidden_states: &[T],
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &crate::ops::moe_routing::MoERoutingConfig,
    ) -> Vec<f32> {
        crate::ops::moe_routing::compute_routing_logits(
            hidden_states, gate_weights, batch_size, seq_len, config
        )
    }

    /// Get expert load statistics (tokens routed to each expert).
    #[inline(always)]
    pub fn compute_expert_load(
        &self,
        routing_result: &crate::ops::moe_routing::MoERoutingResult,
        num_experts: usize,
    ) -> Vec<usize> {
        crate::ops::moe_routing::compute_expert_load(routing_result, num_experts)
    }

    /// Compute load balancing auxiliary loss.
    #[inline(always)]
    pub fn compute_load_balance_loss(
        &self,
        routing_logits: &[f32],
        num_tokens: usize,
        num_experts: usize,
    ) -> f32 {
        crate::ops::moe_routing::compute_load_balance_loss(routing_logits, num_tokens, num_experts)
    }

    // =========================================================================
    // Inference Optimization Kernels (REQ-OP-008 to REQ-OP-015)
    // =========================================================================

    /// EAGLE-3 draft generation: compute confidence + draft length + tokens.
    pub fn eagle3_draft<T: KernelFloat>(
        &self,
        draft_logits: &[f32],
        layer_hidden_states: &[&[T]],
        confidence_weights: &[f32],
        confidence_bias: f32,
        config: Eagle3Config,
    ) -> Result<Eagle3DraftResult, String> {
        config
            .validate()
            .map_err(|e| format!("EAGLE-3 config invalid: {e}"))?;

        if layer_hidden_states.len() < config.fusion_layers {
            return Err("insufficient hidden state layers for fusion".into());
        }

        let vocab_size = match draft_logits.len()
            .checked_div(config.batch_size * config.seq_len)
        {
            Some(v) if v > 0 && v * config.batch_size * config.seq_len == draft_logits.len() => v,
            _ => return Err("draft_logits length mismatch".into()),
        };

        let confidence = match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_eagle3_kernel(), get_cuda_stream()) {
                    match cuda_eagle3_confidence(
                        kernel,
                        stream,
                        layer_hidden_states,
                        confidence_weights,
                        confidence_bias,
                        &config,
                    ) {
                        Ok(confidence) => Some(confidence),
                        Err(e) => {
                            log::debug!("CUDA EAGLE-3 confidence failed: {e}, falling back to CPU");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_eagle3_kernel(), get_metal_device()) {
                    match metal_eagle3_confidence(
                        kernel,
                        device,
                        layer_hidden_states,
                        confidence_weights,
                        confidence_bias,
                        &config,
                    ) {
                        Ok(confidence) => Some(confidence),
                        Err(e) => {
                            log::debug!("Metal EAGLE-3 confidence failed: {e}, falling back to CPU");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_eagle3_kernel() {
                    match wgpu_eagle3_confidence(
                        kernel,
                        layer_hidden_states,
                        confidence_weights,
                        confidence_bias,
                        &config,
                    ) {
                        Ok(confidence) => Some(confidence),
                        Err(e) => {
                            log::debug!("WGPU EAGLE-3 confidence failed: {e}, falling back to CPU");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        let confidence = confidence.unwrap_or_else(|| {
            cpu_eagle3_confidence(layer_hidden_states, confidence_weights, confidence_bias, &config)
        });

        let draft_lengths = cpu_eagle3_draft_lengths(&confidence, &config);
        let draft_tokens = cpu_eagle3_select_tokens(
            draft_logits,
            config.batch_size,
            config.seq_len,
            vocab_size,
            config.max_draft_len,
        )?;

        Ok(Eagle3DraftResult {
            draft_lengths,
            draft_tokens,
            max_draft_len: config.max_draft_len,
        })
    }

    /// EAGLE-3 verify draft tokens against target logits.
    pub fn eagle3_verify(
        &self,
        draft_logits: &[f32],
        target_logits: &[f32],
        draft_tokens: &[i32],
        config: Eagle3VerifyConfig,
    ) -> Result<Eagle3VerifyResult, String> {
        cpu_eagle3_verify(draft_logits, target_logits, draft_tokens, &config)
    }

    /// SpecEE forward: compute confidence + skip/exit decisions.
    pub fn spec_ee_forward<T: KernelFloat>(
        &self,
        hidden_states: &[T],
        classifier_weight: &[T],
        classifier_bias: f32,
        config: SpecEEConfig,
    ) -> Result<SpecEEForwardResult, String> {
        config
            .validate()
            .map_err(|e| format!("SpecEE config invalid: {e}"))?;

        let confidence = match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_spec_ee_kernel(), get_cuda_stream()) {
                    match cuda_spec_ee_confidence(
                        kernel,
                        stream,
                        hidden_states,
                        classifier_weight,
                        classifier_bias,
                        &config,
                    ) {
                        Ok(confidence) => Some(confidence),
                        Err(e) => {
                            log::debug!("CUDA SpecEE confidence failed: {e}, falling back to CPU");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_spec_ee_kernel(), get_metal_device()) {
                    match metal_spec_ee_confidence(
                        kernel,
                        device,
                        hidden_states,
                        classifier_weight,
                        classifier_bias,
                        &config,
                    ) {
                        Ok(confidence) => Some(confidence),
                        Err(e) => {
                            log::debug!("Metal SpecEE confidence failed: {e}, falling back to CPU");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_spec_ee_kernel() {
                    match wgpu_spec_ee_confidence(
                        kernel,
                        hidden_states,
                        classifier_weight,
                        classifier_bias,
                        &config,
                    ) {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            log::debug!("WGPU SpecEE confidence failed: {e}, falling back to CPU");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        let confidence = confidence.unwrap_or_else(|| {
            cpu_spec_ee_confidence(hidden_states, classifier_weight, classifier_bias, &config)
        });

        Ok(cpu_spec_ee_decisions(confidence, &config))
    }

    /// Flash Tree-attention dispatch.
    pub fn flash_tree_attention<T: KernelFloat>(
        &self,
        query: &[T],
        key: &[T],
        value: &[T],
        tree_mask: &[i32],
        output: &mut [T],
        config: FlashTreeAttentionConfig,
    ) {
        let ran_gpu = match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_flash_tree_kernel(), get_cuda_stream()) {
                    cuda_flash_tree_attention(
                        kernel,
                        stream,
                        query,
                        key,
                        value,
                        tree_mask,
                        output,
                        &config,
                    )
                } else {
                    false
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_flash_tree_kernel(), get_metal_device()) {
                    metal_flash_tree_attention(
                        kernel,
                        device,
                        query,
                        key,
                        value,
                        tree_mask,
                        output,
                        &config,
                    )
                } else {
                    false
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_flash_tree_kernel() {
                    wgpu_flash_tree_attention(
                        kernel,
                        query,
                        key,
                        value,
                        tree_mask,
                        output,
                        &config,
                    )
                } else {
                    false
                }
            }
            _ => false,
        };

        if !ran_gpu {
            cpu_flash_tree_attention(query, key, value, tree_mask, output, &config);
        }
    }

    /// INT2 quantize.
    pub fn int2_quantize<T: KernelFloat>(
        &self,
        input: &[T],
        config: Int2QuantConfig,
    ) -> Result<Int2QuantResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_int2_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_int2_quantize(kernel, stream, input, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA INT2 quantize failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_int2_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_int2_quantize(kernel, device, input, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal INT2 quantize failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_int2_kernel() {
                    if let Ok(result) = wgpu_int2_quantize(kernel, input, &config) {
                        return Ok(result);
                    }
                    log::debug!("WGPU INT2 quantize failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_int2_quantize(input, &config)
    }

    /// INT2 dequantize.
    pub fn int2_dequantize<T: KernelFloat>(
        &self,
        quantized: &[i8],
        scales: &[f32],
        zeros: &[f32],
        config: Int2QuantConfig,
    ) -> Result<Vec<T>, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_int2_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_int2_dequantize(kernel, stream, quantized, scales, zeros, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA INT2 dequantize failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_int2_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_int2_dequantize(kernel, device, quantized, scales, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal INT2 dequantize failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_int2_kernel() {
                    if let Ok(result) = wgpu_int2_dequantize(kernel, quantized, scales, zeros, &config) {
                        return Ok(result);
                    }
                    log::debug!("WGPU INT2 dequantize failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_int2_dequantize(quantized, scales, zeros, &config)
    }

    /// EvicPress compression (INT8/INT2).
    pub fn evic_press_compress<T: KernelFloat>(
        &self,
        kv_cache: &[T],
        config: EvicPressCompressConfig,
    ) -> Result<EvicPressCompressionResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_evic_press_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_evic_press_compress(kernel, stream, kv_cache, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA EvicPress compress failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_evic_press_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_evic_press_compress(kernel, device, kv_cache, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal EvicPress compress failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_evic_press_kernel() {
                    if let Ok(result) = wgpu_evic_press_compress(kernel, kv_cache, &config) {
                        return Ok(result);
                    }
                    log::debug!("WGPU EvicPress compress failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_evic_press_compress(kv_cache, &config)
    }

    /// EvicPress eviction decision.
    pub fn evic_press_evict<T: KernelFloat>(
        &self,
        attention_weights: &[T],
        token_ages: &[i32],
        current_zones: &[i32],
        config: EvicPressEvictConfig,
    ) -> Result<EvicPressEvictResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_evic_press_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_evic_press_evict(
                        kernel,
                        stream,
                        attention_weights,
                        token_ages,
                        current_zones,
                        &config,
                    ) {
                        return Ok(result);
                    }
                    log::debug!("CUDA EvicPress evict failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_evic_press_evict(attention_weights, token_ages, current_zones, &config)
    }

    /// Medusa forward: top-k sampling + candidate tree.
    pub fn medusa_forward(
        &self,
        head_logits: &[f32],
        config: MedusaConfig,
    ) -> Result<MedusaForwardResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_medusa_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_medusa_forward(kernel, stream, head_logits, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA Medusa forward failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_medusa_kernel() {
                    if let Ok(result) = wgpu_medusa_forward(kernel, head_logits, &config) {
                        return Ok(result);
                    }
                    log::debug!("WGPU Medusa forward failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_medusa_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_medusa_forward(kernel, device, head_logits, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal Medusa forward failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_medusa_forward(head_logits, &config)
    }

    /// Medusa verify candidates against target logits.
    pub fn medusa_verify(
        &self,
        candidate_tokens: &[i32],
        target_logits: &[f32],
        config: MedusaVerifyConfig,
    ) -> Result<MedusaVerifyResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_medusa_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_medusa_verify(kernel, stream, candidate_tokens, target_logits, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA Medusa verify failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_medusa_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_medusa_verify(kernel, device, candidate_tokens, target_logits, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal Medusa verify failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_medusa_verify(candidate_tokens, target_logits, &config)
    }

    /// Prompt cache lookup: hash + prefix match.
    pub fn prompt_cache_lookup(
        &self,
        tokens: &[i32],
        cache_hashes: &[u64],
        cache_lengths: &[u32],
        config: PromptCacheLookupConfig,
    ) -> Result<PromptCacheLookupResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_prompt_cache_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_prompt_cache_lookup(
                        kernel,
                        stream,
                        tokens,
                        cache_hashes,
                        cache_lengths,
                        &config,
                    ) {
                        return Ok(result);
                    }
                    log::debug!("CUDA prompt cache lookup failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_prompt_cache_kernel() {
                    if let Ok(result) = wgpu_prompt_cache_lookup(
                        kernel,
                        tokens,
                        cache_hashes,
                        cache_lengths,
                        &config,
                    ) {
                        return Ok(result);
                    }
                    log::debug!("WGPU prompt cache lookup failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_prompt_cache_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_prompt_cache_lookup(
                        kernel,
                        device,
                        tokens,
                        cache_hashes,
                        cache_lengths,
                        &config,
                    ) {
                        return Ok(result);
                    }
                    log::debug!("Metal prompt cache lookup failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_prompt_cache_lookup(tokens, cache_hashes, cache_lengths, &config)
    }

    /// Prompt cache blend: merge cached KV with fresh KV.
    pub fn prompt_cache_blend<T: KernelFloat>(
        &self,
        cached_kv: &[T],
        fresh_kv: &[T],
        config: PromptCacheBlendConfig,
    ) -> Result<Vec<T>, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_prompt_cache_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_prompt_cache_blend(kernel, stream, cached_kv, fresh_kv, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA prompt cache blend failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_prompt_cache_kernel() {
                    if let Ok(result) = wgpu_prompt_cache_blend(kernel, cached_kv, fresh_kv, &config) {
                        return Ok(result);
                    }
                    log::debug!("WGPU prompt cache blend failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_prompt_cache_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_prompt_cache_blend(kernel, device, cached_kv, fresh_kv, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal prompt cache blend failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_prompt_cache_blend(cached_kv, fresh_kv, &config)
    }

    /// Chunked prefill attention.
    pub fn chunked_prefill_attention<T: KernelFloat>(
        &self,
        query: &[T],
        key: &[T],
        value: &[T],
        config: ChunkedPrefillConfig,
    ) -> Result<ChunkedPrefillResult<T>, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_chunked_prefill_kernel(), get_cuda_stream()) {
                    if let Ok(result) = cuda_chunked_prefill_attention(kernel, stream, query, key, value, &config) {
                        return Ok(result);
                    }
                    log::debug!("CUDA chunked prefill attention failed, falling back to CPU");
                }
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_chunked_prefill_kernel() {
                    if let Ok(result) = wgpu_chunked_prefill_attention(kernel, query, key, value, &config) {
                        return Ok(result);
                    }
                    log::debug!("WGPU chunked prefill attention failed, falling back to CPU");
                }
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                if let (Some(kernel), Some(device)) = (get_metal_chunked_prefill_kernel(), get_metal_device()) {
                    if let Ok(result) = metal_chunked_prefill_attention(kernel, device, query, key, value, &config) {
                        return Ok(result);
                    }
                    log::debug!("Metal chunked prefill attention failed, falling back to CPU");
                }
            }
            _ => {}
        }

        cpu_chunked_prefill_attention(query, key, value, &config)
    }

    // =========================================================================
    // Embedding Operations (Binary/Int8/Int4 quantization for vector search)
    // =========================================================================

    /// Binary Inner Product (Hamming distance) for vector similarity.
    ///
    /// Computes Hamming distance between binary-quantized vectors.
    /// Lower score = more similar.
    ///
    /// # Arguments
    /// * `queries` - Packed binary query vectors [num_queries, dim/64]
    /// * `database` - Packed binary database vectors [num_vectors, dim/64]
    /// * `scores` - Output Hamming distances [num_queries, num_vectors]
    /// * `config` - Configuration with dim, num_queries, num_vectors
    #[inline(always)]
    pub fn binary_ip_hamming(
        &self,
        queries: &[u64],
        database: &[u64],
        scores: &mut [i32],
        config: crate::ops::embedding::BinaryIpConfig,
    ) {
        // Reinterpret u64 as u32 pairs (GPU uses 32-bit packing)
        let queries_u32: &[u32] = bytemuck::cast_slice(queries);
        let database_u32: &[u32] = bytemuck::cast_slice(database);

        match self.backend {
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_embedding_kernel() {
                    match kernel.binary_ip_hamming(
                        queries_u32,
                        database_u32,
                        config.dim,
                        config.num_queries,
                        config.num_vectors,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("WGPU binary_ip_hamming failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_embedding_kernel(), get_cuda_stream()) {
                    match cuda_binary_ip_hamming(
                        kernel, stream, queries_u32, database_u32,
                        config.dim, config.num_queries, config.num_vectors,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("CUDA binary_ip_hamming failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
            #[cfg(target_os = "linux")]
            BackendType::Rocm => {
                if let (Some(kernel), Some(queue)) = (get_hsa_embedding_kernel(), get_hsa_queue()) {
                    match rocm_binary_ip_hamming(
                        kernel, queue, queries_u32, database_u32,
                        config.dim, config.num_queries, config.num_vectors,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("HSA binary_ip_hamming failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
            #[cfg(not(target_os = "linux"))]
            BackendType::Rocm => {
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
            #[cfg(target_os = "macos")]
            BackendType::Metal => {
                // Metal requires GPU buffer allocation - fall back to CPU for now
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
            #[cfg(not(target_os = "macos"))]
            BackendType::Metal => {
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
            BackendType::Cpu => {
                crate::ops::embedding::binary_ip_hamming_simd(queries, database, scores, &config);
            }
        }
    }

    /// Asymmetric Binary Inner Product (f32 query vs binary database).
    ///
    /// More accurate than symmetric binary, query stays at full precision.
    /// Higher score = more similar.
    #[inline(always)]
    pub fn binary_ip_asymmetric(
        &self,
        queries: &[f32],
        database: &[u64],
        scores: &mut [f32],
        config: crate::ops::embedding::BinaryIpConfig,
    ) {
        match self.backend {
            BackendType::Wgpu => {
                // Reinterpret u64 as u32 pairs (GPU uses 32-bit packing)
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let Some(kernel) = get_wgpu_embedding_kernel() {
                    match kernel.binary_ip_asymmetric(
                        queries,
                        database_u32,
                        config.dim,
                        config.num_queries,
                        config.num_vectors,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("WGPU binary_ip_asymmetric failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::binary_ip_asymmetric(queries, database, scores, &config);
            }
            BackendType::Cuda => {
                // Reinterpret u64 as u32 pairs (GPU uses 32-bit packing)
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let (Some(kernel), Some(stream)) = (get_cuda_embedding_kernel(), get_cuda_stream()) {
                    match cuda_binary_ip_asymmetric(
                        kernel, stream, queries, database_u32,
                        config.dim, config.num_queries, config.num_vectors,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("CUDA binary_ip_asymmetric failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::binary_ip_asymmetric(queries, database, scores, &config);
            }
            #[cfg(target_os = "linux")]
            BackendType::Rocm => {
                // Reinterpret u64 as u32 pairs (GPU uses 32-bit packing)
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let (Some(kernel), Some(queue)) = (get_hsa_embedding_kernel(), get_hsa_queue()) {
                    match rocm_binary_ip_asymmetric(
                        kernel, queue, queries, database_u32,
                        config.dim, config.num_queries, config.num_vectors,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("HSA binary_ip_asymmetric failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::binary_ip_asymmetric(queries, database, scores, &config);
            }
            #[cfg(not(target_os = "linux"))]
            BackendType::Rocm => {
                crate::ops::embedding::binary_ip_asymmetric(queries, database, scores, &config);
            }
            BackendType::Metal | BackendType::Cpu => {
                crate::ops::embedding::binary_ip_asymmetric(queries, database, scores, &config);
            }
        }
    }

    /// Int8 Dot Product for vector similarity.
    ///
    /// 4x throughput improvement over f32 with minimal accuracy loss.
    /// Higher score = more similar.
    #[inline(always)]
    pub fn int8_dot_product(
        &self,
        queries: &[i8],
        database: &[i8],
        scores: &mut [f32],
        config: crate::ops::embedding::Int8DotConfig,
    ) {
        match self.backend {
            BackendType::Wgpu => {
                // Reinterpret i8 as u32 (packed i8x4)
                let queries_u32: &[u32] = bytemuck::cast_slice(queries);
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let Some(kernel) = get_wgpu_embedding_kernel() {
                    match kernel.int8_dot_product(
                        queries_u32,
                        database_u32,
                        config.dim,
                        config.num_queries,
                        config.num_vectors,
                        config.scale,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("WGPU int8_dot_product failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::int8_dot_product_unrolled(queries, database, scores, &config);
            }
            BackendType::Cuda => {
                // Reinterpret i8 as u32 (packed i8x4)
                let queries_u32: &[u32] = bytemuck::cast_slice(queries);
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let (Some(kernel), Some(stream)) = (get_cuda_embedding_kernel(), get_cuda_stream()) {
                    match cuda_int8_dot_product(
                        kernel, stream, queries_u32, database_u32,
                        config.dim, config.num_queries, config.num_vectors, config.scale,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("CUDA int8_dot_product failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::int8_dot_product_unrolled(queries, database, scores, &config);
            }
            #[cfg(target_os = "linux")]
            BackendType::Rocm => {
                // Reinterpret i8 as u32 (packed i8x4)
                let queries_u32: &[u32] = bytemuck::cast_slice(queries);
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let (Some(kernel), Some(queue)) = (get_hsa_embedding_kernel(), get_hsa_queue()) {
                    match rocm_int8_dot_product(
                        kernel, queue, queries_u32, database_u32,
                        config.dim, config.num_queries, config.num_vectors, config.scale,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("HSA int8_dot_product failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::int8_dot_product_unrolled(queries, database, scores, &config);
            }
            #[cfg(not(target_os = "linux"))]
            BackendType::Rocm => {
                crate::ops::embedding::int8_dot_product_unrolled(queries, database, scores, &config);
            }
            BackendType::Metal | BackendType::Cpu => {
                crate::ops::embedding::int8_dot_product_unrolled(queries, database, scores, &config);
            }
        }
    }

    /// Int4 Packed Dot Product for maximum memory efficiency.
    ///
    /// 8x memory bandwidth improvement, 2 values packed per byte.
    #[inline(always)]
    pub fn int4_packed_dot_product(
        &self,
        queries: &[u8],
        database: &[u8],
        scores: &mut [f32],
        config: crate::ops::embedding::Int4PackedConfig,
    ) {
        match self.backend {
            BackendType::Wgpu => {
                // Reinterpret u8 as u32 (packed i4x8)
                let queries_u32: &[u32] = bytemuck::cast_slice(queries);
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let Some(kernel) = get_wgpu_embedding_kernel() {
                    match kernel.int4_dot_product(
                        queries_u32,
                        database_u32,
                        config.dim,
                        config.num_queries,
                        config.num_vectors,
                        config.scale,
                        config.zero_point as i32,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("WGPU int4_dot_product failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::int4_packed_dot_product(queries, database, scores, &config);
            }
            BackendType::Cuda => {
                // Reinterpret u8 as u32 (packed i4x8)
                let queries_u32: &[u32] = bytemuck::cast_slice(queries);
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let (Some(kernel), Some(stream)) = (get_cuda_embedding_kernel(), get_cuda_stream()) {
                    match cuda_int4_dot_product(
                        kernel, stream, queries_u32, database_u32,
                        config.dim, config.num_queries, config.num_vectors,
                        config.scale, config.zero_point as i32,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("CUDA int4_dot_product failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::int4_packed_dot_product(queries, database, scores, &config);
            }
            #[cfg(target_os = "linux")]
            BackendType::Rocm => {
                // Reinterpret u8 as u32 (packed i4x8)
                let queries_u32: &[u32] = bytemuck::cast_slice(queries);
                let database_u32: &[u32] = bytemuck::cast_slice(database);

                if let (Some(kernel), Some(queue)) = (get_hsa_embedding_kernel(), get_hsa_queue()) {
                    match rocm_int4_dot_product(
                        kernel, queue, queries_u32, database_u32,
                        config.dim, config.num_queries, config.num_vectors,
                        config.scale, config.zero_point as i32,
                    ) {
                        Ok(result) => {
                            scores.copy_from_slice(&result);
                            return;
                        }
                        Err(e) => {
                            log::debug!("HSA int4_dot_product failed: {}, falling back to CPU", e);
                        }
                    }
                }
                crate::ops::embedding::int4_packed_dot_product(queries, database, scores, &config);
            }
            #[cfg(not(target_os = "linux"))]
            BackendType::Rocm => {
                crate::ops::embedding::int4_packed_dot_product(queries, database, scores, &config);
            }
            BackendType::Metal | BackendType::Cpu => {
                crate::ops::embedding::int4_packed_dot_product(queries, database, scores, &config);
            }
        }
    }

    /// Matryoshka dimension truncation with optional normalization.
    ///
    /// Enables runtime dimension selection (e.g., 1024→256) for speed/accuracy tradeoff.
    ///
    /// **Performance Note**: This operation is ALWAYS executed on CPU regardless of
    /// selected backend. The operation is essentially a memory copy (dimension truncation)
    /// with optional L2 normalization. CPU execution avoids GPU kernel launch overhead
    /// (~5-50μs) and PCIe data transfer overhead (~100-400μs roundtrip), making CPU
    /// consistently 10-100x faster for this specific operation.
    #[inline(always)]
    pub fn matryoshka_truncate(
        &self,
        embeddings: &[f32],
        output: &mut [f32],
        config: crate::ops::embedding::MatryoshkaConfig,
    ) {
        // ALWAYS use CPU implementation - GPU overhead exceeds computation time
        // for this simple memory-bound operation (dimension truncation + optional normalize)
        crate::ops::embedding::matryoshka_truncate(embeddings, output, &config);
    }

    /// GPU-accelerated three-stage rerank pipeline.
    ///
    /// Efficiently reduces candidate set through progressive refinement:
    /// - Stage 1 (Binary): Hamming distance on binary embeddings (millions → binary_k)
    /// - Stage 2 (Int8): Dot product on int8 embeddings (binary_k → int8_k)
    /// - Stage 3: Returns final candidates for cross-encoder scoring
    ///
    /// **Backend Selection**:
    /// - WGPU: Full GPU acceleration with Top-K selection kernels
    /// - Others: Falls back to CPU implementation
    ///
    /// # Arguments
    /// * `binary_query` - Binary query embedding (packed u32, dim/32 elements)
    /// * `binary_database` - Binary database embeddings (packed u32, num_vectors * dim/32 elements)
    /// * `int8_query` - Int8 query embedding (packed u32, dim/4 elements)
    /// * `int8_database` - Int8 database embeddings (packed u32, num_vectors * dim/4 elements)
    /// * `num_vectors` - Number of vectors in database
    /// * `config` - Pipeline configuration (binary_k, int8_k, dim)
    /// * `int8_scale` - Scale factor for int8 dequantization
    ///
    /// # Returns
    /// `Ok(GpuRerankStageResult)` with final candidate indices and scores, or error message.
    pub fn rerank_pipeline(
        &self,
        binary_query: &[u32],
        binary_database: &[u32],
        int8_query: &[u32],
        int8_database: &[u32],
        num_vectors: usize,
        config: &GpuRerankConfig,
        int8_scale: f32,
    ) -> Result<GpuRerankStageResult, String> {
        match self.backend {
            BackendType::Cuda => {
                if let (Some(kernel), Some(stream)) = (get_cuda_embedding_kernel(), get_cuda_stream()) {
                    match cuda_rerank_pipeline(
                        kernel, stream,
                        binary_query, binary_database,
                        int8_query, int8_database,
                        num_vectors, config, int8_scale,
                    ) {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            log::debug!("CUDA rerank_pipeline failed: {}, falling back to CPU", e);
                        }
                    }
                }
                self.cpu_rerank_pipeline(
                    binary_query, binary_database,
                    int8_query, int8_database,
                    num_vectors, config, int8_scale,
                )
            }
            BackendType::Wgpu => {
                if let Some(kernel) = get_wgpu_embedding_kernel() {
                    kernel.rerank_pipeline(
                        binary_query,
                        binary_database,
                        int8_query,
                        int8_database,
                        num_vectors,
                        config,
                        int8_scale,
                    ).map_err(|e| format!("WGPU rerank_pipeline failed: {}", e))
                } else {
                    self.cpu_rerank_pipeline(
                        binary_query, binary_database,
                        int8_query, int8_database,
                        num_vectors, config, int8_scale,
                    )
                }
            }
            // Other backends fall back to CPU implementation
            _ => {
                self.cpu_rerank_pipeline(
                    binary_query, binary_database,
                    int8_query, int8_database,
                    num_vectors, config, int8_scale,
                )
            }
        }
    }

    /// CPU fallback for rerank pipeline.
    fn cpu_rerank_pipeline(
        &self,
        binary_query: &[u32],
        binary_database: &[u32],
        int8_query: &[u32],
        int8_database: &[u32],
        num_vectors: usize,
        config: &GpuRerankConfig,
        int8_scale: f32,
    ) -> Result<GpuRerankStageResult, String> {
        use crate::ops::embedding::{
            binary_ip_hamming_simd, int8_dot_product_unrolled,
            BinaryIpConfig, Int8DotConfig,
        };

        // Cast u32 slices to u64 for CPU binary operations
        // CPU uses u64 (2x u32), GPU uses u32 packed
        let binary_query_u64: &[u64] = bytemuck::cast_slice(binary_query);
        let binary_database_u64: &[u64] = bytemuck::cast_slice(binary_database);

        // Stage 1: Binary Hamming distance
        let binary_config = BinaryIpConfig {
            dim: config.dim,
            num_queries: 1,
            num_vectors,
        };
        let mut binary_scores = vec![0i32; num_vectors];
        binary_ip_hamming_simd(binary_query_u64, binary_database_u64, &mut binary_scores, &binary_config);

        // Top-K selection (ascending - lower hamming = better)
        let mut indexed_scores: Vec<(usize, i32)> = binary_scores.iter().copied().enumerate().collect();
        indexed_scores.sort_by_key(|(_, score)| *score);
        indexed_scores.truncate(config.binary_k);

        let stage1_indices: Vec<u32> = indexed_scores.iter().map(|(i, _)| *i as u32).collect();

        // Stage 2: Int8 dot product on candidates
        // CPU uses i8, GPU uses packed u32 (4x i8 per u32)
        let int8_query_i8: &[i8] = bytemuck::cast_slice(int8_query);
        let int8_database_i8: &[i8] = bytemuck::cast_slice(int8_database);

        let int8_bytes_per_vec = config.dim;
        let mut candidate_database = Vec::with_capacity(stage1_indices.len() * int8_bytes_per_vec);
        for &idx in &stage1_indices {
            let start = idx as usize * int8_bytes_per_vec;
            let end = start + int8_bytes_per_vec;
            if end <= int8_database_i8.len() {
                candidate_database.extend_from_slice(&int8_database_i8[start..end]);
            }
        }

        let int8_config = Int8DotConfig {
            dim: config.dim,
            num_queries: 1,
            num_vectors: stage1_indices.len(),
            scale: int8_scale,
        };
        let mut int8_scores = vec![0f32; stage1_indices.len()];
        int8_dot_product_unrolled(int8_query_i8, &candidate_database, &mut int8_scores, &int8_config);

        // Top-K selection (descending - higher score = better)
        let mut indexed_int8: Vec<(usize, f32)> = int8_scores.iter().copied().enumerate().collect();
        indexed_int8.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed_int8.truncate(config.int8_k);

        // Map back to original indices
        let final_indices: Vec<u32> = indexed_int8.iter()
            .map(|(i, _)| stage1_indices[*i])
            .collect();
        let final_scores: Vec<f32> = indexed_int8.iter()
            .map(|(_, s)| *s)
            .collect();

        Ok(GpuRerankStageResult {
            indices: final_indices,
            scores: final_scores,
        })
    }

    // =========================================================================
    // Matrix Multiplication Kernels (GEMM)
    // =========================================================================

    /// General matrix multiplication: C = alpha * A * B + beta * C
    ///
    /// # Arguments
    /// * `a` - Input matrix A: [M, K]
    /// * `b` - Input matrix B: [K, N] or [N, K] if transpose_b=true
    /// * `c` - Output matrix C: [M, N] (pre-allocated)
    /// * `config` - GEMM configuration
    ///
    /// # Note
    /// Currently falls back to CPU for all backends.
    /// GPU dispatch will be added when cuBLAS/rocBLAS/Metal Performance Shaders
    /// integration is complete.
    #[inline(always)]
    pub fn matmul<T: KernelFloat>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        config: MatmulConfig,
    ) {
        // All backends use the CPU implementation for now
        // GPU backends (CUDA/cuBLAS, ROCm/rocBLAS, Metal/MPS) will be added later
        cpu_matmul(a, b, c, config);
    }

    /// Add bias to output tensor in-place.
    ///
    /// # Arguments
    /// * `output` - Output tensor: [batch, features]
    /// * `bias` - Bias vector: [features]
    /// * `batch` - Batch size
    /// * `features` - Number of features
    #[inline(always)]
    pub fn add_bias<T: KernelFloat>(
        &self,
        output: &mut [T],
        bias: &[T],
        batch: usize,
        features: usize,
    ) {
        debug_assert_eq!(output.len(), batch * features);
        debug_assert_eq!(bias.len(), features);

        for b in 0..batch {
            let row = &mut output[b * features..(b + 1) * features];
            for (i, x) in row.iter_mut().enumerate() {
                *x = T::from_f32(x.to_f32() + bias[i].to_f32());
            }
        }
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

/// CPU reference implementation of matrix multiplication (GEMM).
///
/// C = alpha * A * B + beta * C
///
/// Where:
/// - A is [M, K]
/// - B is [K, N] (or [N, K] if transpose_b=true)
/// - C is [M, N]
#[inline(always)]
fn cpu_matmul<T: KernelFloat>(a: &[T], b: &[T], c: &mut [T], config: MatmulConfig) {
    let MatmulConfig { m, k, n, transpose_b, alpha, beta } = config;

    debug_assert_eq!(a.len(), m * k, "A matrix size mismatch");
    debug_assert_eq!(c.len(), m * n, "C matrix size mismatch");

    if transpose_b {
        // B is stored as [N, K], we want B^T which is [K, N]
        debug_assert_eq!(b.len(), n * k, "B matrix size mismatch (transposed)");

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    // A[i, l] = a[i * k + l]
                    // B^T[l, j] = B[j, l] = b[j * k + l]
                    let a_val = a[i * k + l].to_f32() as f64;
                    let b_val = b[j * k + l].to_f32() as f64;
                    sum += a_val * b_val;
                }
                let c_idx = i * n + j;
                let c_val = if beta != 0.0 {
                    beta as f64 * c[c_idx].to_f32() as f64
                } else {
                    0.0
                };
                c[c_idx] = T::from_f32((alpha as f64 * sum + c_val) as f32);
            }
        }
    } else {
        // B is stored as [K, N]
        debug_assert_eq!(b.len(), k * n, "B matrix size mismatch");

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    // A[i, l] = a[i * k + l]
                    // B[l, j] = b[l * n + j]
                    let a_val = a[i * k + l].to_f32() as f64;
                    let b_val = b[l * n + j].to_f32() as f64;
                    sum += a_val * b_val;
                }
                let c_idx = i * n + j;
                let c_val = if beta != 0.0 {
                    beta as f64 * c[c_idx].to_f32() as f64
                } else {
                    0.0
                };
                c[c_idx] = T::from_f32((alpha as f64 * sum + c_val) as f32);
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

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn to_f32_vec<T: KernelFloat>(data: &[T]) -> Vec<f32> {
    data.iter().map(|v| v.to_f32()).collect()
}

#[inline]
fn from_f32_vec<T: KernelFloat>(data: Vec<f32>) -> Vec<T> {
    data.into_iter().map(T::from_f32).collect()
}

fn reorder_bhld_to_blhd(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * seq * heads * dim];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let src_base = ((b * heads + h) * seq + s) * dim;
                let dst_base = ((b * seq + s) * heads + h) * dim;
                output[dst_base..dst_base + dim]
                    .copy_from_slice(&data[src_base..src_base + dim]);
            }
        }
    }
    output
}

fn reorder_blhd_to_bhld(
    data: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * heads * seq * dim];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let src_base = ((b * seq + s) * heads + h) * dim;
                let dst_base = ((b * heads + h) * seq + s) * dim;
                output[dst_base..dst_base + dim]
                    .copy_from_slice(&data[src_base..src_base + dim]);
            }
        }
    }
    output
}

fn reorder_bld_to_bhd(
    data: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * heads * seq];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let src = (b * seq + s) * heads + h;
                let dst = (b * heads + h) * seq + s;
                output[dst] = data[src];
            }
        }
    }
    output
}

fn softmax_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_temp = 1.0 / temperature;
    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        max_val = max_val.max(v * inv_temp);
    }
    let mut sum = 0.0f32;
    let mut exps = Vec::with_capacity(logits.len());
    for &v in logits {
        let exp_v = (v * inv_temp - max_val).exp();
        sum += exp_v;
        exps.push(exp_v);
    }
    if sum > 0.0 {
        for v in exps.iter_mut() {
            *v /= sum;
        }
    }
    exps
}

fn softmax_token_prob(logits: &[f32], token_idx: usize) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        max_val = max_val.max(v);
    }
    let mut sum = 0.0f32;
    let mut token_exp = 0.0f32;
    for (idx, &v) in logits.iter().enumerate() {
        let exp_v = (v - max_val).exp();
        sum += exp_v;
        if idx == token_idx {
            token_exp = exp_v;
        }
    }
    if sum > 0.0 { token_exp / sum } else { 0.0 }
}

fn cpu_eagle3_confidence<T: KernelFloat>(
    layer_hidden_states: &[&[T]],
    confidence_weights: &[f32],
    confidence_bias: f32,
    config: &Eagle3Config,
) -> Vec<f32> {
    let fused_dim = config.fused_dim();
    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if confidence_weights.len() != fused_dim {
        return vec![0.0f32; config.batch_size * config.seq_len];
    }
    if layer_hidden_states.len() < config.fusion_layers {
        return vec![0.0f32; config.batch_size * config.seq_len];
    }
    for layer in layer_hidden_states {
        if layer.len() < hidden_len {
            return vec![0.0f32; config.batch_size * config.seq_len];
        }
    }

    let start_layer = layer_hidden_states.len() - config.fusion_layers;
    let mut confidence = vec![0.0f32; config.batch_size * config.seq_len];

    for b in 0..config.batch_size {
        for s in 0..config.seq_len {
            let pos = b * config.seq_len + s;
            let base = pos * config.hidden_dim;
            let mut logit = confidence_bias;

            for (layer_idx, layer) in layer_hidden_states[start_layer..].iter().enumerate() {
                let weight_base = layer_idx * config.hidden_dim;
                for d in 0..config.hidden_dim {
                    logit += layer[base + d].to_f32() * confidence_weights[weight_base + d];
                }
            }

            confidence[pos] = sigmoid(logit);
        }
    }

    confidence
}

fn cpu_eagle3_draft_lengths(confidence: &[f32], config: &Eagle3Config) -> Vec<usize> {
    let mut lengths = vec![config.min_draft_len; config.batch_size];
    if confidence.len() < config.batch_size * config.seq_len {
        return lengths;
    }

    for b in 0..config.batch_size {
        let base = b * config.seq_len;
        let mut draft_len = config.max_draft_len.min(config.seq_len);
        for s in 0..draft_len {
            if confidence[base + s] < config.confidence_threshold {
                draft_len = if s > 0 { s } else { config.min_draft_len };
                break;
            }
        }
        if draft_len < config.min_draft_len {
            draft_len = config.min_draft_len;
        }
        if draft_len > config.max_draft_len {
            draft_len = config.max_draft_len;
        }
        lengths[b] = draft_len;
    }

    lengths
}

fn cpu_eagle3_select_tokens(
    draft_logits: &[f32],
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    max_draft_len: usize,
) -> Result<Vec<i32>, String> {
    let expected = batch_size * seq_len * vocab_size;
    if draft_logits.len() != expected {
        return Err("draft_logits length mismatch".into());
    }
    if seq_len < max_draft_len {
        return Err("seq_len < max_draft_len".into());
    }

    let mut tokens = vec![0i32; batch_size * max_draft_len];
    for b in 0..batch_size {
        for s in 0..max_draft_len {
            let base = (b * seq_len + s) * vocab_size;
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = draft_logits[base + v];
                if val > best_val {
                    best_val = val;
                    best_idx = v;
                }
            }
            tokens[b * max_draft_len + s] = best_idx as i32;
        }
    }

    Ok(tokens)
}

fn cpu_eagle3_verify(
    draft_logits: &[f32],
    target_logits: &[f32],
    draft_tokens: &[i32],
    config: &Eagle3VerifyConfig,
) -> Result<Eagle3VerifyResult, String> {
    let expected_logits = config.batch_size * config.draft_len * config.vocab_size;
    let expected_tokens = config.batch_size * config.draft_len;

    if draft_logits.len() != expected_logits || target_logits.len() != expected_logits {
        return Err("draft/target logits length mismatch".into());
    }
    if draft_tokens.len() != expected_tokens {
        return Err("draft_tokens length mismatch".into());
    }

    let mut acceptance_probs = vec![0.0f32; expected_tokens];
    let mut accepted_lengths = vec![0usize; config.batch_size];

    for b in 0..config.batch_size {
        let mut accepted = 0usize;
        for s in 0..config.draft_len {
            let token = draft_tokens[b * config.draft_len + s] as usize;
            let base = (b * config.draft_len + s) * config.vocab_size;
            let draft_slice = &draft_logits[base..base + config.vocab_size];
            let target_slice = &target_logits[base..base + config.vocab_size];

            let p_draft = softmax_token_prob(draft_slice, token);
            let p_target = softmax_token_prob(target_slice, token);
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else {
                0.0
            };
            acceptance_probs[b * config.draft_len + s] = accept_prob;

            if p_target >= p_draft {
                accepted += 1;
            } else {
                break;
            }
        }
        accepted_lengths[b] = accepted;
    }

    Ok(Eagle3VerifyResult {
        accepted_lengths,
        acceptance_probs,
    })
}

fn cpu_spec_ee_confidence<T: KernelFloat>(
    hidden_states: &[T],
    classifier_weight: &[T],
    classifier_bias: f32,
    config: &SpecEEConfig,
) -> Vec<f32> {
    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if hidden_states.len() != hidden_len || classifier_weight.len() != config.hidden_dim {
        return vec![0.0f32; config.batch_size * config.seq_len];
    }

    let mut confidence = vec![0.0f32; config.batch_size * config.seq_len];
    for b in 0..config.batch_size {
        for s in 0..config.seq_len {
            let base = (b * config.seq_len + s) * config.hidden_dim;
            let mut logit = classifier_bias;
            for d in 0..config.hidden_dim {
                logit += hidden_states[base + d].to_f32() * classifier_weight[d].to_f32();
            }
            confidence[b * config.seq_len + s] = sigmoid(logit);
        }
    }

    confidence
}

fn cpu_spec_ee_decisions(confidence: Vec<f32>, config: &SpecEEConfig) -> SpecEEForwardResult {
    let mut skip = vec![0i32; confidence.len()];
    let mut exit = vec![0i32; confidence.len()];
    let mut exit_layer = vec![-1i32; confidence.len()];

    for (idx, &conf) in confidence.iter().enumerate() {
        if conf >= config.skip_threshold {
            skip[idx] = 1;
        }
        if conf >= config.exit_threshold {
            exit[idx] = 1;
            exit_layer[idx] = config.current_layer as i32;
        }
    }

    SpecEEForwardResult {
        confidence,
        skip_decisions: skip,
        should_exit: exit,
        exit_layer,
    }
}

fn cpu_flash_tree_attention<T: KernelFloat>(
    query: &[T],
    key: &[T],
    value: &[T],
    tree_mask: &[i32],
    output: &mut [T],
    config: &FlashTreeAttentionConfig,
) {
    let ctx_len = config.prefix_len + config.tree_size;
    let q_len = config.batch_size * config.num_heads * config.tree_size * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * ctx_len * config.head_dim;
    let mask_len = config.tree_size * config.tree_size;

    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        output.iter_mut().for_each(|v| *v = T::zero());
        return;
    }
    if tree_mask.len() != mask_len {
        output.iter_mut().for_each(|v| *v = T::zero());
        return;
    }

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let tree_stride = config.head_dim;

    for b in 0..config.batch_size {
        for h in 0..config.num_heads {
            for t in 0..config.tree_size {
                let q_base = ((b * config.num_heads + h) * config.tree_size + t) * tree_stride;
                let mut max_score = f32::NEG_INFINITY;

                for j in 0..ctx_len {
                    if j >= config.prefix_len {
                        let tree_j = j - config.prefix_len;
                        if tree_mask[t * config.tree_size + tree_j] == 0 {
                            continue;
                        }
                    }
                    let k_base = ((b * config.num_heads + h) * ctx_len + j) * tree_stride;
                    let mut score = 0.0f32;
                    for d in 0..config.head_dim {
                        score += query[q_base + d].to_f32() * key[k_base + d].to_f32();
                    }
                    score *= scale;
                    if score > max_score {
                        max_score = score;
                    }
                }

                if max_score == f32::NEG_INFINITY {
                    for d in 0..config.head_dim {
                        output[q_base + d] = T::zero();
                    }
                    continue;
                }

                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; config.head_dim];
                for j in 0..ctx_len {
                    if j >= config.prefix_len {
                        let tree_j = j - config.prefix_len;
                        if tree_mask[t * config.tree_size + tree_j] == 0 {
                            continue;
                        }
                    }
                    let k_base = ((b * config.num_heads + h) * ctx_len + j) * tree_stride;
                    let mut score = 0.0f32;
                    for d in 0..config.head_dim {
                        score += query[q_base + d].to_f32() * key[k_base + d].to_f32();
                    }
                    score *= scale;
                    let weight = (score - max_score).exp();
                    sum_exp += weight;
                    let v_base = k_base;
                    for d in 0..config.head_dim {
                        acc[d] += weight * value[v_base + d].to_f32();
                    }
                }

                if sum_exp > 0.0 {
                    for d in 0..config.head_dim {
                        output[q_base + d] = T::from_f32(acc[d] / sum_exp);
                    }
                } else {
                    for d in 0..config.head_dim {
                        output[q_base + d] = T::zero();
                    }
                }
            }
        }
    }
}

fn cpu_int2_quantize<T: KernelFloat>(
    input: &[T],
    config: &Int2QuantConfig,
) -> Result<Int2QuantResult, String> {
    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }

    let values = to_f32_vec(input);
    let num_elements = values.len();
    let num_groups = (num_elements + config.group_size - 1) / config.group_size;
    let mut quantized = vec![0i8; num_elements];
    let mut scales = vec![0.0f32; num_groups];
    let zeros = vec![0.0f32; num_groups];

    for g in 0..num_groups {
        let start = g * config.group_size;
        let end = (start + config.group_size).min(num_elements);
        let mut max_abs = 0.0f32;
        for &v in &values[start..end] {
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 1.5 } else { 1.0 };
        scales[g] = scale;

        for (idx, &v) in values[start..end].iter().enumerate() {
            let normalized = (v / scale).clamp(-1.5, 1.5);
            let q = ((normalized + 1.5) + 0.5).floor() as i8;
            quantized[start + idx] = q.clamp(0, 3);
        }
    }

    Ok(Int2QuantResult {
        quantized,
        scales,
        zeros,
    })
}

fn cpu_int2_dequantize<T: KernelFloat>(
    quantized: &[i8],
    scales: &[f32],
    _zeros: &[f32],
    config: &Int2QuantConfig,
) -> Result<Vec<T>, String> {
    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    let num_groups = (quantized.len() + config.group_size - 1) / config.group_size;
    if scales.len() < num_groups {
        return Err("scales length mismatch".into());
    }

    let mut output = Vec::with_capacity(quantized.len());
    for (idx, &q) in quantized.iter().enumerate() {
        let group = idx / config.group_size;
        let scale = scales[group];
        let level = match (q & 0x03) {
            0 => -1.5,
            1 => -0.5,
            2 => 0.5,
            _ => 1.5,
        };
        output.push(T::from_f32(level * scale));
    }

    Ok(output)
}

fn cpu_evic_press_compress<T: KernelFloat>(
    kv_cache: &[T],
    config: &EvicPressCompressConfig,
) -> Result<EvicPressCompressionResult, String> {
    if config.seq_len == 0 || config.head_dim == 0 {
        return Err("seq_len and head_dim must be > 0".into());
    }
    let total = config.seq_len * config.head_dim;
    if kv_cache.len() != total {
        return Err("kv_cache length mismatch".into());
    }

    let values = to_f32_vec(kv_cache);
    match config.compression {
        EvicPressCompression::Int8 => {
            let mut data = vec![0i8; total];
            let mut scales = vec![0.0f32; config.seq_len];
            for t in 0..config.seq_len {
                let start = t * config.head_dim;
                let end = start + config.head_dim;
                let mut max_abs = 0.0f32;
                for &v in &values[start..end] {
                    max_abs = max_abs.max(v.abs());
                }
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                scales[t] = scale;
                for (idx, &v) in values[start..end].iter().enumerate() {
                    let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
                    data[start + idx] = q;
                }
            }
            Ok(EvicPressCompressionResult::Int8 { data, scales })
        }
        EvicPressCompression::Int2 => {
            let mut data = vec![0u8; (total + 3) / 4];
            let mut scales = vec![0.0f32; config.seq_len];
            for t in 0..config.seq_len {
                let start = t * config.head_dim;
                let end = start + config.head_dim;
                let mut max_abs = 0.0f32;
                for &v in &values[start..end] {
                    max_abs = max_abs.max(v.abs());
                }
                let scale = if max_abs > 0.0 { max_abs / 1.5 } else { 1.0 };
                scales[t] = scale;
                for (idx, &v) in values[start..end].iter().enumerate() {
                    let normalized = (v / scale).clamp(-1.5, 1.5);
                    let q = ((normalized + 1.5) + 0.5).floor() as u8;
                    let q = q.min(3);
                    let elem_idx = start + idx;
                    let byte_idx = elem_idx / 4;
                    let shift = (3 - (elem_idx % 4)) * 2;
                    data[byte_idx] |= (q & 0x03) << shift;
                }
            }
            Ok(EvicPressCompressionResult::Int2 { data, scales })
        }
    }
}

fn cpu_evic_press_evict<T: KernelFloat>(
    attention_weights: &[T],
    token_ages: &[i32],
    _current_zones: &[i32],
    config: &EvicPressEvictConfig,
) -> Result<EvicPressEvictResult, String> {
    let expected = config.batch_size * config.num_heads * config.seq_len;
    if attention_weights.len() != expected {
        return Err("attention_weights length mismatch".into());
    }
    if token_ages.len() != config.seq_len {
        return Err("token_ages length mismatch".into());
    }

    let mut importance = vec![0.0f32; config.seq_len];
    let mut new_zones = vec![0i32; config.seq_len];
    let denom = (config.batch_size * config.num_heads).max(1) as f32;
    let cache_pressure = config.cache_pressure.clamp(0.0, 1.0);
    let hot_threshold = config.hot_threshold * (1.0 - cache_pressure);
    let warm_threshold = config.warm_threshold * (1.0 - cache_pressure);

    for t in 0..config.seq_len {
        let mut attn_sum = 0.0f32;
        for b in 0..config.batch_size {
            for h in 0..config.num_heads {
                let idx = (b * config.num_heads + h) * config.seq_len + t;
                attn_sum += attention_weights[idx].to_f32();
            }
        }
        let avg_attn = attn_sum / denom;
        let recency = 1.0 / (1.0 + token_ages[t].max(0) as f32);
        let score = config.attention_weight * avg_attn + config.recency_weight * recency;
        importance[t] = score;

        new_zones[t] = if score >= hot_threshold {
            0
        } else if score >= warm_threshold {
            1
        } else {
            2
        };
    }

    Ok(EvicPressEvictResult {
        importance,
        new_zones,
    })
}

fn cpu_medusa_forward(
    head_logits: &[f32],
    config: &MedusaConfig,
) -> Result<MedusaForwardResult, String> {
    if config.temperature <= 0.0 {
        return Err("temperature must be > 0".into());
    }
    let expected = config.batch_size * config.num_heads * config.vocab_size;
    if head_logits.len() != expected {
        return Err("head_logits length mismatch".into());
    }
    if config.top_k == 0 || config.top_k > config.vocab_size {
        return Err("top_k must be in (0, vocab_size]".into());
    }

    let mut top_k_tokens = vec![0i32; config.batch_size * config.num_heads * config.top_k];
    let mut top_k_probs = vec![0.0f32; config.batch_size * config.num_heads * config.top_k];

    for b in 0..config.batch_size {
        for h in 0..config.num_heads {
            let base = (b * config.num_heads + h) * config.vocab_size;
            let probs = softmax_probs(
                &head_logits[base..base + config.vocab_size],
                config.temperature,
            );
            let mut indices: Vec<usize> = (0..config.vocab_size).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            for k in 0..config.top_k {
                let idx = indices[k];
                top_k_tokens[(b * config.num_heads + h) * config.top_k + k] = idx as i32;
                top_k_probs[(b * config.num_heads + h) * config.top_k + k] = probs[idx];
            }
        }
    }

    let total_possible = config.top_k.pow(config.num_heads as u32);
    let max_iter = total_possible.min(config.max_candidates);
    let mut candidate_tokens =
        vec![0i32; config.batch_size * config.max_candidates * config.num_heads];
    let mut candidate_probs = vec![0.0f32; config.batch_size * config.max_candidates];
    let mut num_candidates = vec![0i32; config.batch_size];

    for b in 0..config.batch_size {
        let mut count = 0usize;
        for c in 0..max_iter {
            let mut temp = c;
            let mut prob = 1.0f32;
            let mut tokens = vec![0i32; config.num_heads];
            for h in 0..config.num_heads {
                let k_idx = temp % config.top_k;
                temp /= config.top_k;
                let base = (b * config.num_heads + h) * config.top_k + k_idx;
                let token = top_k_tokens[base];
                let p = top_k_probs[base];
                tokens[h] = token;
                prob *= p;
            }
            if prob > 0.001f32 {
                if count >= config.max_candidates {
                    break;
                }
                let offset = (b * config.max_candidates + count) * config.num_heads;
                for h in 0..config.num_heads {
                    candidate_tokens[offset + h] = tokens[h];
                }
                candidate_probs[b * config.max_candidates + count] = prob;
                count += 1;
            }
        }
        num_candidates[b] = count as i32;
    }

    Ok(MedusaForwardResult {
        candidate_tokens,
        candidate_probs,
        num_candidates,
    })
}

fn cpu_medusa_verify(
    candidate_tokens: &[i32],
    target_logits: &[f32],
    config: &MedusaVerifyConfig,
) -> Result<MedusaVerifyResult, String> {
    let expected_candidates = config.batch_size * config.num_candidates * config.seq_len;
    let expected_logits =
        config.batch_size * config.num_candidates * config.seq_len * config.vocab_size;
    if candidate_tokens.len() < expected_candidates {
        return Err("candidate_tokens length mismatch".into());
    }
    if target_logits.len() != expected_logits {
        return Err("target_logits length mismatch".into());
    }

    let mut accepted_lengths = vec![0i32; config.batch_size * config.num_candidates];
    let mut best_candidate = vec![0i32; config.batch_size];

    for b in 0..config.batch_size {
        let mut best_len = -1i32;
        let mut best_idx = 0i32;
        for c in 0..config.num_candidates {
            let cand_base = (b * config.num_candidates + c) * config.seq_len;
            let cand_token = candidate_tokens[cand_base] as usize;
            let logits_base = ((b * config.num_candidates + c) * config.seq_len) * config.vocab_size;
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..config.vocab_size {
                let val = target_logits[logits_base + v];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }
            let accepted = if max_idx == cand_token { 1 } else { 0 };
            accepted_lengths[b * config.num_candidates + c] = accepted;
            if accepted > best_len {
                best_len = accepted;
                best_idx = c as i32;
            }
        }
        best_candidate[b] = best_idx;
    }

    Ok(MedusaVerifyResult {
        accepted_lengths,
        best_candidate,
    })
}

fn rolling_hash(tokens: &[i32], seed: u64) -> Vec<u64> {
    let mut hash = seed ^ 0xcbf29ce484222325;
    let mut output = Vec::with_capacity(tokens.len());
    for &token in tokens {
        hash ^= token as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        output.push(hash);
    }
    output
}

fn cpu_prompt_cache_lookup(
    tokens: &[i32],
    cache_hashes: &[u64],
    cache_lengths: &[u32],
    config: &PromptCacheLookupConfig,
) -> Result<PromptCacheLookupResult, String> {
    if cache_lengths.len() < config.num_entries {
        return Err("cache_lengths length mismatch".into());
    }
    let expected_hashes = config.num_entries * config.max_cache_len;
    if cache_hashes.len() < expected_hashes {
        return Err("cache_hashes length mismatch".into());
    }

    let query_hashes = rolling_hash(tokens, config.hash_seed);
    let query_len = tokens.len().min(config.max_cache_len);

    let mut best_entry = -1i32;
    let mut best_len = 0usize;
    for entry in 0..config.num_entries {
        let cache_len = cache_lengths[entry] as usize;
        if cache_len == 0 {
            continue;
        }
        let max_compare = query_len.min(cache_len).min(config.max_cache_len);
        let mut match_len = 0usize;
        let base = entry * config.max_cache_len;
        for i in 0..max_compare {
            if query_hashes[i] == cache_hashes[base + i] {
                match_len = i + 1;
            } else {
                break;
            }
        }
        if match_len > best_len {
            best_len = match_len;
            best_entry = entry as i32;
        }
    }

    if best_len < config.min_match_len {
        best_entry = -1;
        best_len = 0;
    }

    Ok(PromptCacheLookupResult {
        best_entry,
        match_length: best_len,
        query_hashes,
    })
}

fn cpu_prompt_cache_blend<T: KernelFloat>(
    cached_kv: &[T],
    fresh_kv: &[T],
    config: &PromptCacheBlendConfig,
) -> Result<Vec<T>, String> {
    let kv_size = config.num_heads * config.head_dim;
    let cached_len = config.match_len * kv_size;
    let fresh_len = config.fresh_len * kv_size;
    if cached_kv.len() < cached_len || fresh_kv.len() < fresh_len {
        return Err("cached/fresh kv length mismatch".into());
    }

    let mut output = Vec::with_capacity(cached_len + fresh_len);
    output.extend_from_slice(&cached_kv[..cached_len]);
    output.extend_from_slice(&fresh_kv[..fresh_len]);
    Ok(output)
}

fn cpu_chunked_prefill_attention<T: KernelFloat>(
    query: &[T],
    key: &[T],
    value: &[T],
    config: &ChunkedPrefillConfig,
) -> Result<ChunkedPrefillResult<T>, String> {
    if config.query_len == 0 || config.chunk_len == 0 || config.head_dim == 0 {
        return Err("query_len, chunk_len, head_dim must be > 0".into());
    }

    let q_len = config.batch_size * config.num_heads * config.query_len * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * config.chunk_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return Err("query/key/value length mismatch".into());
    }

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut output = vec![T::zero(); q_len];
    let mut log_sum_exp = vec![0.0f32; config.batch_size * config.num_heads * config.query_len];

    for b in 0..config.batch_size {
        for h in 0..config.num_heads {
            for q in 0..config.query_len {
                let q_base = ((b * config.num_heads + h) * config.query_len + q) * config.head_dim;
                let global_q = config.chunk_start + q;
                if global_q >= config.chunk_len {
                    continue;
                }
                let mut max_score = f32::NEG_INFINITY;
                let mut max_k = config.chunk_len;
                if config.causal {
                    max_k = (global_q + 1).min(config.chunk_len);
                }
                for k_idx in 0..max_k {
                    let k_base = ((b * config.num_heads + h) * config.chunk_len + k_idx) * config.head_dim;
                    let mut score = 0.0f32;
                    for d in 0..config.head_dim {
                        score += query[q_base + d].to_f32() * key[k_base + d].to_f32();
                    }
                    score *= scale;
                    if score > max_score {
                        max_score = score;
                    }
                }

                if max_score == f32::NEG_INFINITY {
                    continue;
                }

                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; config.head_dim];
                for k_idx in 0..max_k {
                    let k_base = ((b * config.num_heads + h) * config.chunk_len + k_idx) * config.head_dim;
                    let mut score = 0.0f32;
                    for d in 0..config.head_dim {
                        score += query[q_base + d].to_f32() * key[k_base + d].to_f32();
                    }
                    score *= scale;
                    let weight = (score - max_score).exp();
                    sum_exp += weight;
                    let v_base = k_base;
                    for d in 0..config.head_dim {
                        acc[d] += weight * value[v_base + d].to_f32();
                    }
                }

                if sum_exp > 0.0 {
                    for d in 0..config.head_dim {
                        output[q_base + d] = T::from_f32(acc[d] / sum_exp);
                    }
                }
                log_sum_exp[(b * config.num_heads + h) * config.query_len + q] =
                    max_score + sum_exp.ln();
            }
        }
    }

    Ok(ChunkedPrefillResult { output, log_sum_exp })
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

fn cuda_eagle3_confidence<T: KernelFloat>(
    kernel: &CudaEagle3Kernel,
    stream: &Arc<CudaStream>,
    layer_hidden_states: &[&[T]],
    confidence_weights: &[f32],
    confidence_bias: f32,
    config: &Eagle3Config,
) -> Result<Vec<f32>, String> {
    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if confidence_weights.len() != config.fused_dim() {
        return Err("confidence_weights length mismatch".into());
    }
    if layer_hidden_states.len() < config.fusion_layers {
        return Err("insufficient hidden state layers".into());
    }
    for layer in layer_hidden_states {
        if layer.len() < hidden_len {
            return Err("hidden state length mismatch".into());
        }
    }

    let start_layer = layer_hidden_states.len() - config.fusion_layers;
    let mut layer_bufs = Vec::with_capacity(config.fusion_layers);
    for layer in &layer_hidden_states[start_layer..] {
        let layer_f32 = to_f32_vec(layer);
        let buf = stream.clone_htod(&layer_f32)
            .map_err(|e| format!("Failed to copy hidden states to GPU: {}", e))?;
        layer_bufs.push(buf);
    }
    let layer_refs: Vec<&cudarc::driver::CudaSlice<f32>> = layer_bufs.iter().collect();

    let fused = kernel.fuse_hidden_states_f32(
        stream,
        &layer_refs,
        config.batch_size,
        config.seq_len,
        config.hidden_dim,
        config.fusion_layers,
    ).map_err(|e| format!("CUDA EAGLE-3 fuse failed: {}", e))?;

    let weight_buf = stream.clone_htod(confidence_weights)
        .map_err(|e| format!("Failed to copy confidence weights to GPU: {}", e))?;

    let confidence = kernel.predict_confidence_f32(
        stream,
        &fused,
        &weight_buf,
        confidence_bias,
        config.batch_size,
        config.seq_len,
        config.fused_dim(),
    ).map_err(|e| format!("CUDA EAGLE-3 confidence failed: {}", e))?;

    stream.clone_dtoh(&confidence)
        .map_err(|e| format!("Failed to copy confidence from GPU: {}", e))
}

#[cfg(target_os = "macos")]
fn metal_eagle3_confidence<T: KernelFloat>(
    kernel: &MetalEagle3Kernel,
    device: &metal::Device,
    layer_hidden_states: &[&[T]],
    confidence_weights: &[f32],
    confidence_bias: f32,
    config: &Eagle3Config,
) -> Result<Vec<f32>, String> {
    use metal::MTLResourceOptions;

    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if confidence_weights.len() != config.fused_dim() {
        return Err("confidence_weights length mismatch".into());
    }
    if layer_hidden_states.len() < config.fusion_layers {
        return Err("insufficient hidden state layers".into());
    }
    for layer in layer_hidden_states {
        if layer.len() < hidden_len {
            return Err("hidden state length mismatch".into());
        }
    }

    let start_layer = layer_hidden_states.len() - config.fusion_layers;
    let mut buffers = Vec::with_capacity(config.fusion_layers);
    for layer in &layer_hidden_states[start_layer..] {
        let layer_f32 = to_f32_vec(layer);
        let buf = device.new_buffer_with_data(
            layer_f32.as_ptr() as *const _,
            (layer_f32.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        buffers.push(buf);
    }
    let buffer_refs: Vec<&metal::Buffer> = buffers.iter().collect();

    let fused = kernel.fuse_hidden_f32(
        &buffer_refs,
        config.batch_size,
        config.seq_len,
        config.hidden_dim,
        config.fusion_layers,
    ).map_err(|e| format!("Metal EAGLE-3 fuse failed: {}", e))?;

    let weight_buf = device.new_buffer_with_data(
        confidence_weights.as_ptr() as *const _,
        (confidence_weights.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output = kernel.predict_confidence_f32(
        &fused,
        &weight_buf,
        confidence_bias,
        config.batch_size,
        config.seq_len,
        config.fused_dim(),
    ).map_err(|e| format!("Metal EAGLE-3 confidence failed: {}", e))?;

    let out_ptr = output.contents() as *const f32;
    let out_len = config.batch_size * config.seq_len;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
    Ok(out_slice.to_vec())
}

fn wgpu_eagle3_confidence<T: KernelFloat>(
    kernel: &WgpuEagle3Kernel,
    layer_hidden_states: &[&[T]],
    confidence_weights: &[f32],
    confidence_bias: f32,
    config: &Eagle3Config,
) -> Result<Vec<f32>, String> {
    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if confidence_weights.len() != config.fused_dim() {
        return Err("confidence_weights length mismatch".into());
    }
    if layer_hidden_states.len() < config.fusion_layers {
        return Err("insufficient hidden state layers".into());
    }
    for layer in layer_hidden_states {
        if layer.len() < hidden_len {
            return Err("hidden state length mismatch".into());
        }
    }

    let start_layer = layer_hidden_states.len() - config.fusion_layers;
    let mut fused_input = Vec::with_capacity(config.fusion_layers * hidden_len);
    for layer in &layer_hidden_states[start_layer..] {
        fused_input.extend(layer.iter().map(|v| v.to_f32()));
    }

    let fused = kernel.fuse_layers_f32(
        &fused_input,
        config.batch_size,
        config.seq_len,
        config.hidden_dim,
        config.fusion_layers,
    ).map_err(|e| format!("WGPU EAGLE-3 fuse failed: {}", e))?;

    kernel.predict_confidence_f32(
        &fused,
        confidence_weights,
        config.batch_size,
        config.seq_len,
        config.fused_dim(),
        confidence_bias,
    ).map_err(|e| format!("WGPU EAGLE-3 confidence failed: {}", e))
}

fn cuda_spec_ee_confidence<T: KernelFloat>(
    kernel: &CudaSpecEEKernel,
    stream: &Arc<CudaStream>,
    hidden_states: &[T],
    classifier_weight: &[T],
    classifier_bias: f32,
    config: &SpecEEConfig,
) -> Result<Vec<f32>, String> {
    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if hidden_states.len() != hidden_len || classifier_weight.len() != config.hidden_dim {
        return Err("hidden/classifier length mismatch".into());
    }

    let hidden_f32 = to_f32_vec(hidden_states);
    let weight_f32 = to_f32_vec(classifier_weight);
    let hidden_buf = stream.clone_htod(&hidden_f32)
        .map_err(|e| format!("Failed to copy hidden states to GPU: {}", e))?;
    let weight_buf = stream.clone_htod(&weight_f32)
        .map_err(|e| format!("Failed to copy classifier weights to GPU: {}", e))?;

    let confidence = kernel.compute_confidence_f32(
        stream,
        &hidden_buf,
        &weight_buf,
        classifier_bias,
        config.batch_size,
        config.seq_len,
        config.hidden_dim,
    ).map_err(|e| format!("CUDA SpecEE confidence failed: {}", e))?;

    stream.clone_dtoh(&confidence)
        .map_err(|e| format!("Failed to copy confidence from GPU: {}", e))
}

#[cfg(target_os = "macos")]
fn metal_spec_ee_confidence<T: KernelFloat>(
    kernel: &MetalSpecEEKernel,
    device: &metal::Device,
    hidden_states: &[T],
    classifier_weight: &[T],
    classifier_bias: f32,
    config: &SpecEEConfig,
) -> Result<Vec<f32>, String> {
    use metal::MTLResourceOptions;

    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if hidden_states.len() != hidden_len || classifier_weight.len() != config.hidden_dim {
        return Err("hidden/classifier length mismatch".into());
    }

    let hidden_f32 = to_f32_vec(hidden_states);
    let weight_f32 = to_f32_vec(classifier_weight);
    let hidden_buf = device.new_buffer_with_data(
        hidden_f32.as_ptr() as *const _,
        (hidden_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weight_buf = device.new_buffer_with_data(
        weight_f32.as_ptr() as *const _,
        (weight_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output = kernel.compute_confidence_f32(
        &hidden_buf,
        &weight_buf,
        config.batch_size,
        config.seq_len,
        config.hidden_dim,
        1,
        1.0,
    ).map_err(|e| format!("Metal SpecEE confidence failed: {}", e))?;

    let out_ptr = output.contents() as *const f32;
    let out_len = config.batch_size * config.seq_len;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
    let mut confidence = out_slice.to_vec();
    if classifier_bias != 0.0 {
        for conf in confidence.iter_mut() {
            let clamped = conf.clamp(1e-6, 1.0 - 1e-6);
            let logit = (clamped / (1.0 - clamped)).ln();
            *conf = sigmoid(logit + classifier_bias);
        }
    }

    Ok(confidence)
}

fn wgpu_spec_ee_confidence<T: KernelFloat>(
    kernel: &WgpuSpecEEKernel,
    hidden_states: &[T],
    classifier_weight: &[T],
    classifier_bias: f32,
    config: &SpecEEConfig,
) -> Result<SpecEEForwardResult, String> {
    let hidden_len = config.batch_size * config.seq_len * config.hidden_dim;
    if hidden_states.len() != hidden_len || classifier_weight.len() != config.hidden_dim {
        return Err("hidden/classifier length mismatch".into());
    }
    let hidden_f32 = to_f32_vec(hidden_states);
    let weight_f32 = to_f32_vec(classifier_weight);

    let result = kernel.compute_confidence_f32(
        &hidden_f32,
        &weight_f32,
        config.batch_size,
        config.seq_len,
        config.hidden_dim,
        config.current_layer,
        config.exit_threshold,
    ).map_err(|e| format!("WGPU SpecEE confidence failed: {}", e))?;

    let mut confidence = result.confidence;
    if classifier_bias != 0.0 {
        for conf in confidence.iter_mut() {
            let clamped = conf.clamp(1e-6, 1.0 - 1e-6);
            let logit = (clamped / (1.0 - clamped)).ln();
            *conf = sigmoid(logit + classifier_bias);
        }
    }

    let mut skip_decisions = vec![0i32; confidence.len()];
    let mut should_exit = vec![0i32; confidence.len()];
    let mut exit_layer = vec![-1i32; confidence.len()];
    for (idx, &conf) in confidence.iter().enumerate() {
        if conf >= config.skip_threshold {
            skip_decisions[idx] = 1;
        }
        if conf >= config.exit_threshold {
            should_exit[idx] = 1;
            exit_layer[idx] = config.current_layer as i32;
        }
    }

    Ok(SpecEEForwardResult {
        confidence,
        skip_decisions,
        should_exit,
        exit_layer,
    })
}

fn cuda_flash_tree_attention<T: KernelFloat>(
    kernel: &CudaFlashTreeAttnKernel,
    stream: &Arc<CudaStream>,
    query: &[T],
    key: &[T],
    value: &[T],
    tree_mask: &[i32],
    output: &mut [T],
    config: &FlashTreeAttentionConfig,
) -> bool {
    let ctx_len = config.prefix_len + config.tree_size;
    let q_len = config.batch_size * config.num_heads * config.tree_size * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * ctx_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return false;
    }
    if tree_mask.len() != config.tree_size * config.tree_size {
        return false;
    }

    let q_f32 = to_f32_vec(query);
    let k_f32 = to_f32_vec(key);
    let v_f32 = to_f32_vec(value);

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
    let mask_buf = match stream.clone_htod(tree_mask) {
        Ok(buf) => buf,
        Err(e) => {
            log::debug!("Failed to copy tree mask to GPU: {}", e);
            return false;
        }
    };

    let result = kernel.tree_attention_f32(
        stream,
        &q_buf,
        &k_buf,
        &v_buf,
        &mask_buf,
        config.batch_size,
        config.num_heads,
        config.tree_size,
        config.prefix_len,
        config.head_dim,
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
            log::debug!("CUDA flash tree attention failed: {}", e);
            false
        }
    }
}

#[cfg(target_os = "macos")]
fn metal_flash_tree_attention<T: KernelFloat>(
    kernel: &MetalFlashTreeAttnKernel,
    device: &metal::Device,
    query: &[T],
    key: &[T],
    value: &[T],
    tree_mask: &[i32],
    output: &mut [T],
    config: &FlashTreeAttentionConfig,
) -> bool {
    use metal::MTLResourceOptions;

    let ctx_len = config.prefix_len + config.tree_size;
    let q_len = config.batch_size * config.num_heads * config.tree_size * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * ctx_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return false;
    }
    if tree_mask.len() != config.tree_size * config.tree_size {
        return false;
    }

    let q_f32 = to_f32_vec(query);
    let k_f32 = to_f32_vec(key);
    let v_f32 = to_f32_vec(value);

    let q_buf = device.new_buffer_with_data(
        q_f32.as_ptr() as *const _,
        (q_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_f32.as_ptr() as *const _,
        (k_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_f32.as_ptr() as *const _,
        (v_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mask_buf = device.new_buffer_with_data(
        tree_mask.as_ptr() as *const _,
        (tree_mask.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut parent_indices = vec![0i32; config.tree_size];
    for i in 1..config.tree_size {
        let mut parent = 0i32;
        for j in (0..i).rev() {
            if tree_mask[i * config.tree_size + j] != 0 {
                parent = j as i32;
                break;
            }
        }
        parent_indices[i] = parent;
    }
    let parent_buf = device.new_buffer_with_data(
        parent_indices.as_ptr() as *const _,
        (parent_indices.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let result = kernel.tree_attention_f32(
        &q_buf,
        &k_buf,
        &v_buf,
        &mask_buf,
        &parent_buf,
        config.batch_size,
        config.num_heads,
        config.tree_size,
        ctx_len,
        config.head_dim,
        scale,
    );

    match result {
        Ok(out_buf) => {
            let out_ptr = out_buf.contents() as *const f32;
            let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, output.len()) };
            for (i, &value) in out_slice.iter().enumerate() {
                output[i] = T::from_f32(value);
            }
            true
        }
        Err(e) => {
            log::debug!("Metal flash tree attention failed: {}", e);
            false
        }
    }
}

fn wgpu_flash_tree_attention<T: KernelFloat>(
    kernel: &WgpuFlashTreeAttn,
    query: &[T],
    key: &[T],
    value: &[T],
    tree_mask: &[i32],
    output: &mut [T],
    config: &FlashTreeAttentionConfig,
) -> bool {
    let ctx_len = config.prefix_len + config.tree_size;
    let q_len = config.batch_size * config.num_heads * config.tree_size * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * ctx_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return false;
    }
    if tree_mask.len() != config.tree_size * config.tree_size {
        return false;
    }

    let q_f32 = to_f32_vec(query);
    let k_f32 = to_f32_vec(key);
    let v_f32 = to_f32_vec(value);
    let q_reordered = reorder_bhld_to_blhd(
        &q_f32,
        config.batch_size,
        config.num_heads,
        config.tree_size,
        config.head_dim,
    );
    let k_reordered = reorder_bhld_to_blhd(
        &k_f32,
        config.batch_size,
        config.num_heads,
        ctx_len,
        config.head_dim,
    );
    let v_reordered = reorder_bhld_to_blhd(
        &v_f32,
        config.batch_size,
        config.num_heads,
        ctx_len,
        config.head_dim,
    );
    let mask_f32: Vec<f32> = tree_mask
        .iter()
        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
        .collect();

    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let params = crate::wgpu_kernels::flash_tree_attn::TreeAttnParams::new(
        config.batch_size as u32,
        config.num_heads as u32,
        config.prefix_len as u32,
        config.tree_size as u32,
        config.head_dim as u32,
        scale,
    );
    let out = kernel.forward_f32(&q_reordered, &k_reordered, &v_reordered, &mask_f32, &params);
    let out_reordered = reorder_blhd_to_bhld(
        &out,
        config.batch_size,
        config.tree_size,
        config.num_heads,
        config.head_dim,
    );
    for (i, value) in out_reordered.into_iter().enumerate() {
        if i < output.len() {
            output[i] = T::from_f32(value);
        }
    }
    true
}

fn cuda_int2_quantize<T: KernelFloat>(
    kernel: &CudaInt2QuantizerKernel,
    stream: &Arc<CudaStream>,
    input: &[T],
    config: &Int2QuantConfig,
) -> Result<Int2QuantResult, String> {
    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    let input_f32 = to_f32_vec(input);
    let input_buf = stream.clone_htod(&input_f32)
        .map_err(|e| format!("Failed to copy input to GPU: {}", e))?;

    let (quant_buf, scales_buf, zeros_buf) = kernel.quantize_f32(
        stream,
        &input_buf,
        config.group_size,
        input_f32.len(),
    ).map_err(|e| format!("CUDA INT2 quantize failed: {}", e))?;

    let quantized = stream.clone_dtoh(&quant_buf)
        .map_err(|e| format!("Failed to copy quantized from GPU: {}", e))?;
    let scales = stream.clone_dtoh(&scales_buf)
        .map_err(|e| format!("Failed to copy scales from GPU: {}", e))?;
    let zeros = stream.clone_dtoh(&zeros_buf)
        .map_err(|e| format!("Failed to copy zeros from GPU: {}", e))?;

    Ok(Int2QuantResult {
        quantized,
        scales,
        zeros,
    })
}

#[cfg(target_os = "macos")]
fn metal_int2_quantize<T: KernelFloat>(
    kernel: &MetalInt2QuantizerKernel,
    device: &metal::Device,
    input: &[T],
    config: &Int2QuantConfig,
) -> Result<Int2QuantResult, String> {
    use metal::MTLResourceOptions;

    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    let input_f32 = to_f32_vec(input);
    if input_f32.len() % config.group_size != 0 || input_f32.len() % 4 != 0 {
        return Err("input length must be divisible by group_size and 4".into());
    }

    let input_buf = device.new_buffer_with_data(
        input_f32.as_ptr() as *const _,
        (input_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let (packed, scales_buf) = kernel.quantize_f32(
        &input_buf,
        input_f32.len(),
        config.group_size,
    ).map_err(|e| format!("Metal INT2 quantize failed: {}", e))?;

    let unpacked = kernel.unpack_int2(&packed, input_f32.len())
        .map_err(|e| format!("Metal INT2 unpack failed: {}", e))?;

    let quant_ptr = unpacked.contents() as *const u8;
    let quant_slice = unsafe { std::slice::from_raw_parts(quant_ptr, input_f32.len()) };
    let quantized: Vec<i8> = quant_slice.iter().map(|&v| v as i8).collect();

    let scales_ptr = scales_buf.contents() as *const f32;
    let num_groups = input_f32.len() / config.group_size;
    let scales_slice = unsafe { std::slice::from_raw_parts(scales_ptr, num_groups) };
    let scales = scales_slice.to_vec();
    let zeros = vec![0.0f32; num_groups];

    Ok(Int2QuantResult {
        quantized,
        scales,
        zeros,
    })
}

fn wgpu_int2_quantize<T: KernelFloat>(
    kernel: &WgpuInt2Quantizer,
    input: &[T],
    config: &Int2QuantConfig,
) -> Result<Int2QuantResult, String> {
    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    let input_f32 = to_f32_vec(input);
    let (quantized_u32, scales, zeros) =
        kernel.quantize_f32(&input_f32, config.group_size as u32);
    let quantized: Vec<i8> = quantized_u32.into_iter().map(|v| (v & 0x03) as i8).collect();

    Ok(Int2QuantResult {
        quantized,
        scales,
        zeros,
    })
}

fn cuda_int2_dequantize<T: KernelFloat>(
    kernel: &CudaInt2QuantizerKernel,
    stream: &Arc<CudaStream>,
    quantized: &[i8],
    scales: &[f32],
    zeros: &[f32],
    config: &Int2QuantConfig,
) -> Result<Vec<T>, String> {
    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    let quant_buf = stream.clone_htod(quantized)
        .map_err(|e| format!("Failed to copy quantized to GPU: {}", e))?;
    let scales_buf = stream.clone_htod(scales)
        .map_err(|e| format!("Failed to copy scales to GPU: {}", e))?;
    let zeros_buf = stream.clone_htod(zeros)
        .map_err(|e| format!("Failed to copy zeros to GPU: {}", e))?;

    let output = kernel.dequantize_f32(
        stream,
        &quant_buf,
        &scales_buf,
        &zeros_buf,
        config.group_size,
        quantized.len(),
    ).map_err(|e| format!("CUDA INT2 dequantize failed: {}", e))?;

    let output_f32 = stream.clone_dtoh(&output)
        .map_err(|e| format!("Failed to copy output from GPU: {}", e))?;
    Ok(from_f32_vec(output_f32))
}

#[cfg(target_os = "macos")]
fn metal_int2_dequantize<T: KernelFloat>(
    kernel: &MetalInt2QuantizerKernel,
    device: &metal::Device,
    quantized: &[i8],
    scales: &[f32],
    config: &Int2QuantConfig,
) -> Result<Vec<T>, String> {
    use metal::MTLResourceOptions;

    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    if quantized.len() % config.group_size != 0 || quantized.len() % 4 != 0 {
        return Err("quantized length must be divisible by group_size and 4".into());
    }
    let num_groups = quantized.len() / config.group_size;
    if scales.len() != num_groups {
        return Err("scales length mismatch".into());
    }

    let quant_u8: Vec<u8> = quantized.iter().map(|&v| v as u8).collect();
    let quant_buf = device.new_buffer_with_data(
        quant_u8.as_ptr() as *const _,
        quant_u8.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let packed = kernel.pack_int2(&quant_buf, quant_u8.len())
        .map_err(|e| format!("Metal INT2 pack failed: {}", e))?;
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const _,
        (scales.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = kernel.dequantize_f32(&packed, &scales_buf, quant_u8.len(), config.group_size)
        .map_err(|e| format!("Metal INT2 dequantize failed: {}", e))?;

    let out_ptr = output_buf.contents() as *const f32;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, quant_u8.len()) };
    Ok(from_f32_vec(out_slice.to_vec()))
}

fn wgpu_int2_dequantize<T: KernelFloat>(
    kernel: &WgpuInt2Quantizer,
    quantized: &[i8],
    scales: &[f32],
    zeros: &[f32],
    config: &Int2QuantConfig,
) -> Result<Vec<T>, String> {
    if config.group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    let quant_u32: Vec<u32> = quantized.iter().map(|&v| (v as u8) as u32).collect();
    let output = kernel.dequantize_f32(&quant_u32, scales, zeros, config.group_size as u32);
    Ok(from_f32_vec(output))
}

fn cuda_evic_press_compress<T: KernelFloat>(
    kernel: &CudaEvicPressKernel,
    stream: &Arc<CudaStream>,
    kv_cache: &[T],
    config: &EvicPressCompressConfig,
) -> Result<EvicPressCompressionResult, String> {
    let total = config.seq_len * config.head_dim;
    if kv_cache.len() != total {
        return Err("kv_cache length mismatch".into());
    }
    let kv_f32 = to_f32_vec(kv_cache);
    let kv_buf = stream.clone_htod(&kv_f32)
        .map_err(|e| format!("Failed to copy kv_cache to GPU: {}", e))?;

    match config.compression {
        EvicPressCompression::Int8 => {
            let (data_buf, scales_buf) = kernel.compress_to_int8_f32(
                stream,
                &kv_buf,
                config.seq_len,
                config.head_dim,
            ).map_err(|e| format!("CUDA EvicPress INT8 compress failed: {}", e))?;
            let data = stream.clone_dtoh(&data_buf)
                .map_err(|e| format!("Failed to copy INT8 data from GPU: {}", e))?;
            let scales = stream.clone_dtoh(&scales_buf)
                .map_err(|e| format!("Failed to copy INT8 scales from GPU: {}", e))?;
            Ok(EvicPressCompressionResult::Int8 { data, scales })
        }
        EvicPressCompression::Int2 => {
            let (int8_buf, _) = kernel.compress_to_int8_f32(
                stream,
                &kv_buf,
                config.seq_len,
                config.head_dim,
            ).map_err(|e| format!("CUDA EvicPress INT8 precompress failed: {}", e))?;
            let (data_buf, scales_buf) = kernel.compress_to_int2_f32(
                stream,
                &int8_buf,
                config.seq_len,
                config.head_dim,
            ).map_err(|e| format!("CUDA EvicPress INT2 compress failed: {}", e))?;
            let data = stream.clone_dtoh(&data_buf)
                .map_err(|e| format!("Failed to copy INT2 data from GPU: {}", e))?;
            let scales = stream.clone_dtoh(&scales_buf)
                .map_err(|e| format!("Failed to copy INT2 scales from GPU: {}", e))?;
            Ok(EvicPressCompressionResult::Int2 { data, scales })
        }
    }
}

#[cfg(target_os = "macos")]
fn metal_evic_press_compress<T: KernelFloat>(
    kernel: &MetalEvicPressKernel,
    device: &metal::Device,
    kv_cache: &[T],
    config: &EvicPressCompressConfig,
) -> Result<EvicPressCompressionResult, String> {
    use metal::MTLResourceOptions;

    let total = config.seq_len * config.head_dim;
    if kv_cache.len() != total {
        return Err("kv_cache length mismatch".into());
    }
    let kv_f32 = to_f32_vec(kv_cache);
    let kv_buf = device.new_buffer_with_data(
        kv_f32.as_ptr() as *const _,
        (kv_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    match config.compression {
        EvicPressCompression::Int8 => {
            let (data_buf, scales_buf) = kernel.compress_to_int8_f32(
                &kv_buf,
                kv_f32.len(),
                config.head_dim,
            ).map_err(|e| format!("Metal EvicPress INT8 compress failed: {}", e))?;
            let data_ptr = data_buf.contents() as *const i8;
            let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, kv_f32.len()) };
            let scales_ptr = scales_buf.contents() as *const f32;
            let scales_slice = unsafe { std::slice::from_raw_parts(scales_ptr, config.seq_len) };
            Ok(EvicPressCompressionResult::Int8 {
                data: data_slice.to_vec(),
                scales: scales_slice.to_vec(),
            })
        }
        EvicPressCompression::Int2 => {
            let (data_buf, scales_buf) = kernel.compress_to_int2_f32(
                &kv_buf,
                kv_f32.len(),
                config.head_dim,
            ).map_err(|e| format!("Metal EvicPress INT2 compress failed: {}", e))?;
            let packed_len = (kv_f32.len() + 3) / 4;
            let data_ptr = data_buf.contents() as *const u8;
            let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, packed_len) };
            let scales_ptr = scales_buf.contents() as *const f32;
            let scales_slice = unsafe { std::slice::from_raw_parts(scales_ptr, config.seq_len) };
            Ok(EvicPressCompressionResult::Int2 {
                data: data_slice.to_vec(),
                scales: scales_slice.to_vec(),
            })
        }
    }
}

fn wgpu_evic_press_compress<T: KernelFloat>(
    kernel: &WgpuEvicPress,
    kv_cache: &[T],
    config: &EvicPressCompressConfig,
) -> Result<EvicPressCompressionResult, String> {
    let total = config.seq_len * config.head_dim;
    if kv_cache.len() != total {
        return Err("kv_cache length mismatch".into());
    }
    if matches!(config.compression, EvicPressCompression::Int2) {
        return Err("WGPU EvicPress INT2 compression not supported".into());
    }
    let kv_f32 = to_f32_vec(kv_cache);
    let (data_i32, scales) = kernel.hot_to_warm_f32(&kv_f32, config.head_dim as u32);
    let data: Vec<i8> = data_i32.into_iter().map(|v| v as i8).collect();
    Ok(EvicPressCompressionResult::Int8 { data, scales })
}

fn cuda_evic_press_evict<T: KernelFloat>(
    kernel: &CudaEvicPressKernel,
    stream: &Arc<CudaStream>,
    attention_weights: &[T],
    token_ages: &[i32],
    current_zones: &[i32],
    config: &EvicPressEvictConfig,
) -> Result<EvicPressEvictResult, String> {
    let expected = config.batch_size * config.num_heads * config.seq_len;
    if attention_weights.len() != expected {
        return Err("attention_weights length mismatch".into());
    }
    if token_ages.len() != config.seq_len || current_zones.len() != config.seq_len {
        return Err("token_ages/current_zones length mismatch".into());
    }

    let attn_f32 = to_f32_vec(attention_weights);
    let attn_buf = stream.clone_htod(&attn_f32)
        .map_err(|e| format!("Failed to copy attention weights to GPU: {}", e))?;
    let age_buf = stream.clone_htod(token_ages)
        .map_err(|e| format!("Failed to copy token ages to GPU: {}", e))?;
    let zones_buf = stream.clone_htod(current_zones)
        .map_err(|e| format!("Failed to copy current zones to GPU: {}", e))?;

    let importance = kernel.compute_importance_f32(
        stream,
        &attn_buf,
        &age_buf,
        config.batch_size,
        config.num_heads,
        config.seq_len,
        config.recency_weight,
        config.attention_weight,
    ).map_err(|e| format!("CUDA EvicPress importance failed: {}", e))?;

    let new_zones = kernel.zone_transition(
        stream,
        &zones_buf,
        &importance,
        config.seq_len,
        config.hot_threshold,
        config.warm_threshold,
        config.cache_pressure,
    ).map_err(|e| format!("CUDA EvicPress zone transition failed: {}", e))?;

    let importance_host = stream.clone_dtoh(&importance)
        .map_err(|e| format!("Failed to copy importance from GPU: {}", e))?;
    let new_zones_host = stream.clone_dtoh(&new_zones)
        .map_err(|e| format!("Failed to copy zones from GPU: {}", e))?;

    Ok(EvicPressEvictResult {
        importance: importance_host,
        new_zones: new_zones_host,
    })
}

fn cuda_medusa_forward(
    kernel: &CudaMedusaKernel,
    stream: &Arc<CudaStream>,
    head_logits: &[f32],
    config: &MedusaConfig,
) -> Result<MedusaForwardResult, String> {
    let expected = config.batch_size * config.num_heads * config.vocab_size;
    if head_logits.len() != expected {
        return Err("head_logits length mismatch".into());
    }
    if config.top_k == 0 || config.top_k > config.vocab_size {
        return Err("top_k must be in (0, vocab_size]".into());
    }

    let logits_buf = stream.clone_htod(head_logits)
        .map_err(|e| format!("Failed to copy head logits to GPU: {}", e))?;

    let (top_k_tokens, top_k_probs) = kernel.top_k_sample_f32(
        stream,
        &logits_buf,
        config.batch_size,
        config.num_heads,
        config.vocab_size,
        config.top_k,
        config.temperature,
    ).map_err(|e| format!("CUDA Medusa top-k failed: {}", e))?;

    let (candidates, candidate_probs, num_candidates) = kernel.build_candidates_f32(
        stream,
        &top_k_tokens,
        &top_k_probs,
        config.batch_size,
        config.num_heads,
        config.top_k,
        config.max_candidates,
    ).map_err(|e| format!("CUDA Medusa build candidates failed: {}", e))?;

    let candidate_tokens = stream.clone_dtoh(&candidates)
        .map_err(|e| format!("Failed to copy candidates from GPU: {}", e))?;
    let candidate_probs = stream.clone_dtoh(&candidate_probs)
        .map_err(|e| format!("Failed to copy candidate probs from GPU: {}", e))?;
    let num_candidates = stream.clone_dtoh(&num_candidates)
        .map_err(|e| format!("Failed to copy num_candidates from GPU: {}", e))?;

    Ok(MedusaForwardResult {
        candidate_tokens,
        candidate_probs,
        num_candidates,
    })
}

#[cfg(target_os = "macos")]
fn metal_medusa_forward(
    kernel: &MetalMedusaKernel,
    device: &metal::Device,
    head_logits: &[f32],
    config: &MedusaConfig,
) -> Result<MedusaForwardResult, String> {
    use metal::MTLResourceOptions;

    let expected = config.batch_size * config.num_heads * config.vocab_size;
    if head_logits.len() != expected {
        return Err("head_logits length mismatch".into());
    }
    if config.top_k == 0 || config.top_k > config.vocab_size {
        return Err("top_k must be in (0, vocab_size]".into());
    }
    if config.temperature <= 0.0 {
        return Err("temperature must be > 0".into());
    }

    let logits_buf = device.new_buffer_with_data(
        head_logits.as_ptr() as *const _,
        (head_logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let (top_k_tokens, top_k_probs) = kernel.top_k_sample_f32(
        &logits_buf,
        config.batch_size,
        config.vocab_size,
        config.num_heads,
        config.top_k,
        config.temperature,
    ).map_err(|e| format!("Metal Medusa top-k failed: {}", e))?;

    let (candidates, _tree_indices) = kernel.build_candidates_f32(
        &top_k_tokens,
        &top_k_probs,
        config.batch_size,
        config.num_heads,
        config.top_k,
        config.max_candidates,
    ).map_err(|e| format!("Metal Medusa build candidates failed: {}", e))?;

    let topk_len = config.batch_size * config.num_heads * config.top_k;
    let topk_tokens_ptr = top_k_tokens.contents() as *const u32;
    let topk_tokens = unsafe { std::slice::from_raw_parts(topk_tokens_ptr, topk_len) };
    let topk_probs_ptr = top_k_probs.contents() as *const f32;
    let topk_probs = unsafe { std::slice::from_raw_parts(topk_probs_ptr, topk_len) };

    let cand_len = config.batch_size * config.max_candidates * config.num_heads;
    let cand_ptr = candidates.contents() as *const u32;
    let cand_slice = unsafe { std::slice::from_raw_parts(cand_ptr, cand_len) };
    let mut candidate_tokens: Vec<i32> = cand_slice.iter().map(|&v| v as i32).collect();
    let mut candidate_probs = vec![0.0f32; config.batch_size * config.max_candidates];
    let mut num_candidates = vec![0i32; config.batch_size];

    for b in 0..config.batch_size {
        let mut count = 0usize;
        for c in 0..config.max_candidates {
            let base = (b * config.max_candidates + c) * config.num_heads;
            let mut prob = 1.0f32;
            let mut valid = true;
            for h in 0..config.num_heads {
                let token = candidate_tokens[base + h];
                let mut found = None;
                let topk_base = (b * config.num_heads + h) * config.top_k;
                for k in 0..config.top_k {
                    let idx = topk_base + k;
                    if topk_tokens[idx] as i32 == token {
                        found = Some(topk_probs[idx]);
                        break;
                    }
                }
                match found {
                    Some(p) => prob *= p,
                    None => {
                        prob = 0.0;
                        valid = false;
                        break;
                    }
                }
            }
            if valid && prob > 0.0 {
                candidate_probs[b * config.max_candidates + c] = prob;
                count += 1;
            } else {
                for h in 0..config.num_heads {
                    candidate_tokens[base + h] = -1;
                }
            }
        }
        num_candidates[b] = count as i32;
    }

    Ok(MedusaForwardResult {
        candidate_tokens,
        candidate_probs,
        num_candidates,
    })
}

fn wgpu_medusa_forward(
    kernel: &WgpuMedusa,
    head_logits: &[f32],
    config: &MedusaConfig,
) -> Result<MedusaForwardResult, String> {
    let expected = config.batch_size * config.num_heads * config.vocab_size;
    if head_logits.len() != expected {
        return Err("head_logits length mismatch".into());
    }
    if config.top_k == 0 || config.top_k > config.vocab_size {
        return Err("top_k must be in (0, vocab_size]".into());
    }

    let topk = kernel.top_k_f32(
        head_logits,
        config.batch_size as u32,
        config.num_heads as u32,
        config.vocab_size as u32,
        config.top_k as u32,
    );
    let tree = kernel.build_candidates(
        &topk.indices,
        config.batch_size as u32,
        config.num_heads as u32,
        config.top_k as u32,
        config.max_candidates as u32,
    );

    let mut candidate_tokens =
        vec![-1i32; config.batch_size * config.max_candidates * config.num_heads];
    let mut candidate_probs = vec![0.0f32; config.batch_size * config.max_candidates];
    let mut num_candidates = vec![0i32; config.batch_size];

    for b in 0..config.batch_size {
        let count = tree.counts[b] as usize;
        num_candidates[b] = count as i32;
        for c in 0..count.min(config.max_candidates) {
            let token = tree.candidates[b * config.max_candidates + c] as i32;
            let base = (b * config.max_candidates + c) * config.num_heads;
            candidate_tokens[base] = token;
            for h in 1..config.num_heads {
                candidate_tokens[base + h] = -1;
            }
            candidate_probs[b * config.max_candidates + c] = 1.0;
        }
    }

    Ok(MedusaForwardResult {
        candidate_tokens,
        candidate_probs,
        num_candidates,
    })
}

fn cuda_medusa_verify(
    kernel: &CudaMedusaKernel,
    stream: &Arc<CudaStream>,
    candidate_tokens: &[i32],
    target_logits: &[f32],
    config: &MedusaVerifyConfig,
) -> Result<MedusaVerifyResult, String> {
    let expected_candidates = config.batch_size * config.num_candidates * config.seq_len;
    let expected_logits = config.batch_size * config.num_candidates * config.seq_len * config.vocab_size;
    if candidate_tokens.len() < expected_candidates {
        return Err("candidate_tokens length mismatch".into());
    }
    if target_logits.len() != expected_logits {
        return Err("target_logits length mismatch".into());
    }

    let mut probs = Vec::with_capacity(target_logits.len());
    let block = config.vocab_size;
    for chunk in target_logits.chunks(block) {
        probs.extend(softmax_probs(chunk, 1.0));
    }

    let candidate_buf = stream.clone_htod(candidate_tokens)
        .map_err(|e| format!("Failed to copy candidates to GPU: {}", e))?;
    let probs_buf = stream.clone_htod(&probs)
        .map_err(|e| format!("Failed to copy target probs to GPU: {}", e))?;

    let (accepted, best) = kernel.verify_candidates_f32(
        stream,
        &candidate_buf,
        &probs_buf,
        config.batch_size,
        config.num_candidates,
        config.seq_len,
        config.vocab_size,
    ).map_err(|e| format!("CUDA Medusa verify failed: {}", e))?;

    let accepted_lengths = stream.clone_dtoh(&accepted)
        .map_err(|e| format!("Failed to copy accepted lengths from GPU: {}", e))?;
    let best_candidate = stream.clone_dtoh(&best)
        .map_err(|e| format!("Failed to copy best candidate from GPU: {}", e))?;

    Ok(MedusaVerifyResult {
        accepted_lengths,
        best_candidate,
    })
}

#[cfg(target_os = "macos")]
fn metal_medusa_verify(
    kernel: &MetalMedusaKernel,
    device: &metal::Device,
    candidate_tokens: &[i32],
    target_logits: &[f32],
    config: &MedusaVerifyConfig,
) -> Result<MedusaVerifyResult, String> {
    use metal::MTLResourceOptions;

    let expected_candidates = config.batch_size * config.num_candidates * config.seq_len;
    let expected_logits =
        config.batch_size * config.num_candidates * config.seq_len * config.vocab_size;
    if candidate_tokens.len() < expected_candidates {
        return Err("candidate_tokens length mismatch".into());
    }
    if target_logits.len() != expected_logits {
        return Err("target_logits length mismatch".into());
    }

    let candidates_u32: Vec<u32> = candidate_tokens
        .iter()
        .take(expected_candidates)
        .map(|&v| if v < 0 { 0 } else { v as u32 })
        .collect();

    let candidates_buf = device.new_buffer_with_data(
        candidates_u32.as_ptr() as *const _,
        (candidates_u32.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let logits_buf = device.new_buffer_with_data(
        target_logits.as_ptr() as *const _,
        (target_logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let (accepted_mask, _accepted_count) = kernel.verify_candidates_f32(
        &candidates_buf,
        &logits_buf,
        config.batch_size,
        config.num_candidates,
        config.seq_len,
        config.vocab_size,
    ).map_err(|e| format!("Metal Medusa verify failed: {}", e))?;

    let accepted_len = config.batch_size * config.num_candidates;
    let accepted_ptr = accepted_mask.contents() as *const u32;
    let accepted_slice = unsafe { std::slice::from_raw_parts(accepted_ptr, accepted_len) };
    let accepted_lengths: Vec<i32> = accepted_slice.iter().map(|&v| v as i32).collect();

    let mut best_candidate = vec![0i32; config.batch_size];
    for b in 0..config.batch_size {
        let base = b * config.num_candidates;
        let mut best_len = -1i32;
        let mut best_idx = 0i32;
        for c in 0..config.num_candidates {
            let len = accepted_lengths[base + c];
            if len > best_len {
                best_len = len;
                best_idx = c as i32;
            }
        }
        best_candidate[b] = best_idx;
    }

    Ok(MedusaVerifyResult {
        accepted_lengths,
        best_candidate,
    })
}

fn cuda_prompt_cache_lookup(
    kernel: &CudaPromptCacheKernel,
    stream: &Arc<CudaStream>,
    tokens: &[i32],
    cache_hashes: &[u64],
    cache_lengths: &[u32],
    config: &PromptCacheLookupConfig,
) -> Result<PromptCacheLookupResult, String> {
    if config.num_entries == 0 {
        return Err("num_entries must be > 0".into());
    }
    let query_len = tokens.len().min(config.max_cache_len);
    if query_len == 0 {
        return Err("tokens must not be empty".into());
    }
    if cache_lengths.len() < config.num_entries {
        return Err("cache_lengths length mismatch".into());
    }
    let expected_hashes = config.num_entries * config.max_cache_len;
    if cache_hashes.len() < expected_hashes {
        return Err("cache_hashes length mismatch".into());
    }

    let tokens_buf = stream.clone_htod(&tokens[..query_len])
        .map_err(|e| format!("Failed to copy tokens to GPU: {}", e))?;
    let query_hashes_buf = kernel.compute_hash_f32(
        stream,
        &tokens_buf,
        query_len,
        config.hash_seed,
    ).map_err(|e| format!("CUDA prompt cache hash failed: {}", e))?;

    let cache_hashes_buf = stream.clone_htod(cache_hashes)
        .map_err(|e| format!("Failed to copy cache hashes to GPU: {}", e))?;
    let cache_lengths_i32: Vec<i32> = cache_lengths
        .iter()
        .take(config.num_entries)
        .map(|&v| v as i32)
        .collect();
    let cache_lengths_buf = stream.clone_htod(&cache_lengths_i32)
        .map_err(|e| format!("Failed to copy cache lengths to GPU: {}", e))?;

    let (best_entry_buf, match_len_buf) = kernel.find_prefix_match(
        stream,
        &query_hashes_buf,
        &cache_hashes_buf,
        &cache_lengths_buf,
        query_len,
        config.num_entries,
        config.max_cache_len,
    ).map_err(|e| format!("CUDA prompt cache prefix match failed: {}", e))?;

    let best_entry_host = stream.clone_dtoh(&best_entry_buf)
        .map_err(|e| format!("Failed to copy best entry from GPU: {}", e))?;
    let match_len_host = stream.clone_dtoh(&match_len_buf)
        .map_err(|e| format!("Failed to copy match length from GPU: {}", e))?;
    let mut best_entry = best_entry_host.get(0).copied().unwrap_or(-1);
    let mut match_length = match_len_host.get(0).copied().unwrap_or(0).max(0) as usize;
    if match_length < config.min_match_len {
        best_entry = -1;
        match_length = 0;
    }

    let query_hashes = stream.clone_dtoh(&query_hashes_buf)
        .map_err(|e| format!("Failed to copy query hashes from GPU: {}", e))?;

    Ok(PromptCacheLookupResult {
        best_entry,
        match_length,
        query_hashes,
    })
}

fn wgpu_prompt_cache_lookup(
    kernel: &WgpuPromptCache,
    tokens: &[i32],
    cache_hashes: &[u64],
    cache_lengths: &[u32],
    config: &PromptCacheLookupConfig,
) -> Result<PromptCacheLookupResult, String> {
    if config.num_entries == 0 {
        return Err("num_entries must be > 0".into());
    }
    let query_len = tokens.len().min(config.max_cache_len);
    if query_len == 0 {
        return Err("tokens must not be empty".into());
    }
    if cache_lengths.len() < config.num_entries {
        return Err("cache_lengths length mismatch".into());
    }
    let expected_hashes = config.num_entries * config.max_cache_len;
    if cache_hashes.len() < expected_hashes {
        return Err("cache_hashes length mismatch".into());
    }

    let mut tokens_u32 = Vec::with_capacity(query_len);
    for &token in tokens.iter().take(query_len) {
        if token < 0 {
            return Err("tokens must be non-negative".into());
        }
        tokens_u32.push(token as u32);
    }

    let query_hashes_u32 = kernel.compute_hash(
        &tokens_u32,
        1,
        query_len as u32,
        1,
    );

    let mut cache_hashes_u32 = vec![0u32; expected_hashes];
    for entry in 0..config.num_entries {
        let cache_len = cache_lengths[entry].min(config.max_cache_len as u32) as usize;
        let base = entry * config.max_cache_len;
        for i in 0..config.max_cache_len {
            let src = cache_hashes[base + i] as u32;
            cache_hashes_u32[base + i] = if i < cache_len { src } else { u32::MAX };
        }
    }

    let result = kernel.find_prefix_match(
        &query_hashes_u32,
        &cache_hashes_u32,
        1,
        query_len as u32,
        config.num_entries as u32,
        config.max_cache_len as u32,
    );

    let match_entry = result.match_entries.get(0).copied().unwrap_or(u32::MAX);
    let mut best_entry = if match_entry == u32::MAX { -1 } else { match_entry as i32 };
    let mut match_length = result.match_lengths.get(0).copied().unwrap_or(0) as usize;
    if match_length < config.min_match_len {
        best_entry = -1;
        match_length = 0;
    }

    let query_hashes: Vec<u64> = query_hashes_u32.into_iter().map(|v| v as u64).collect();

    Ok(PromptCacheLookupResult {
        best_entry,
        match_length,
        query_hashes,
    })
}

#[cfg(target_os = "macos")]
fn metal_prompt_cache_lookup(
    kernel: &MetalPromptCacheKernel,
    device: &metal::Device,
    tokens: &[i32],
    cache_hashes: &[u64],
    cache_lengths: &[u32],
    config: &PromptCacheLookupConfig,
) -> Result<PromptCacheLookupResult, String> {
    use metal::MTLResourceOptions;

    if config.num_entries == 0 {
        return Err("num_entries must be > 0".into());
    }
    let query_len = tokens.len().min(config.max_cache_len);
    if query_len == 0 {
        return Err("tokens must not be empty".into());
    }
    if cache_lengths.len() < config.num_entries {
        return Err("cache_lengths length mismatch".into());
    }
    let expected_hashes = config.num_entries * config.max_cache_len;
    if cache_hashes.len() < expected_hashes {
        return Err("cache_hashes length mismatch".into());
    }

    let tokens_f32: Vec<f32> = tokens.iter().take(query_len).map(|&v| v as f32).collect();
    let tokens_buf = device.new_buffer_with_data(
        tokens_f32.as_ptr() as *const _,
        (tokens_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let query_hashes_buf = kernel.compute_hash_f32(
        &tokens_buf,
        1,
        query_len,
        1,
        1,
    ).map_err(|e| format!("Metal prompt cache hash failed: {}", e))?;

    let mut cache_trimmed = vec![u64::MAX; config.num_entries * query_len];
    for entry in 0..config.num_entries {
        let cache_len = cache_lengths[entry].min(config.max_cache_len as u32) as usize;
        let base = entry * config.max_cache_len;
        let dst_base = entry * query_len;
        for i in 0..query_len {
            if i < cache_len {
                cache_trimmed[dst_base + i] = cache_hashes[base + i];
            }
        }
    }

    let cache_buf = device.new_buffer_with_data(
        cache_trimmed.as_ptr() as *const _,
        (cache_trimmed.len() * std::mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let (match_indices, match_lengths) = kernel.find_prefix_match(
        &query_hashes_buf,
        &cache_buf,
        1,
        config.num_entries,
        query_len,
        config.min_match_len,
    ).map_err(|e| format!("Metal prompt cache prefix match failed: {}", e))?;

    let idx_ptr = match_indices.contents() as *const i32;
    let len_ptr = match_lengths.contents() as *const u32;
    let best_entry = unsafe { *idx_ptr };
    let mut match_length = unsafe { *len_ptr } as usize;

    let query_ptr = query_hashes_buf.contents() as *const u64;
    let query_slice = unsafe { std::slice::from_raw_parts(query_ptr, query_len) };
    let query_hashes = query_slice.to_vec();

    let mut best_entry = best_entry;
    if match_length < config.min_match_len {
        best_entry = -1;
        match_length = 0;
    }

    Ok(PromptCacheLookupResult {
        best_entry,
        match_length,
        query_hashes,
    })
}

fn cuda_prompt_cache_blend<T: KernelFloat>(
    kernel: &CudaPromptCacheKernel,
    stream: &Arc<CudaStream>,
    cached_kv: &[T],
    fresh_kv: &[T],
    config: &PromptCacheBlendConfig,
) -> Result<Vec<T>, String> {
    let kv_size = config.num_heads * config.head_dim;
    let cached_len = config.match_len * kv_size;
    let fresh_len = config.fresh_len * kv_size;
    if cached_kv.len() < cached_len || fresh_kv.len() < fresh_len {
        return Err("cached/fresh kv length mismatch".into());
    }

    let cached_f32 = to_f32_vec(&cached_kv[..cached_len]);
    let fresh_f32 = to_f32_vec(&fresh_kv[..fresh_len]);
    let cached_buf = stream.clone_htod(&cached_f32)
        .map_err(|e| format!("Failed to copy cached kv to GPU: {}", e))?;
    let fresh_buf = stream.clone_htod(&fresh_f32)
        .map_err(|e| format!("Failed to copy fresh kv to GPU: {}", e))?;

    let output = kernel.cache_blend_f32(
        stream,
        &cached_buf,
        &fresh_buf,
        config.match_len,
        config.fresh_len,
        config.num_heads,
        config.head_dim,
        config.blend_window,
    ).map_err(|e| format!("CUDA prompt cache blend failed: {}", e))?;

    let output_f32 = stream.clone_dtoh(&output)
        .map_err(|e| format!("Failed to copy blended kv from GPU: {}", e))?;
    Ok(from_f32_vec(output_f32))
}

fn wgpu_prompt_cache_blend<T: KernelFloat>(
    kernel: &WgpuPromptCache,
    cached_kv: &[T],
    fresh_kv: &[T],
    config: &PromptCacheBlendConfig,
) -> Result<Vec<T>, String> {
    let kv_size = config.num_heads * config.head_dim;
    let cached_len = config.match_len * kv_size;
    let fresh_len = config.fresh_len * kv_size;
    if cached_kv.len() < cached_len || fresh_kv.len() < fresh_len {
        return Err("cached/fresh kv length mismatch".into());
    }

    let cached_f32 = to_f32_vec(&cached_kv[..cached_len]);
    let fresh_f32 = to_f32_vec(&fresh_kv[..fresh_len]);
    let output = kernel.blend_kv_f32(
        &cached_f32,
        &fresh_f32,
        1,
        config.num_heads as u32,
        config.head_dim as u32,
        config.match_len as u32,
        config.fresh_len as u32,
    );
    Ok(from_f32_vec(output))
}

#[cfg(target_os = "macos")]
fn metal_prompt_cache_blend<T: KernelFloat>(
    kernel: &MetalPromptCacheKernel,
    device: &metal::Device,
    cached_kv: &[T],
    fresh_kv: &[T],
    config: &PromptCacheBlendConfig,
) -> Result<Vec<T>, String> {
    use metal::MTLResourceOptions;

    let kv_size = config.num_heads * config.head_dim;
    let cached_len = config.match_len * kv_size;
    let fresh_len = config.fresh_len * kv_size;
    if cached_kv.len() < cached_len || fresh_kv.len() < fresh_len {
        return Err("cached/fresh kv length mismatch".into());
    }

    let cached_f32 = to_f32_vec(&cached_kv[..cached_len]);
    let fresh_f32 = to_f32_vec(&fresh_kv[..fresh_len]);
    let cached_buf = device.new_buffer_with_data(
        cached_f32.as_ptr() as *const _,
        (cached_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let fresh_buf = device.new_buffer_with_data(
        fresh_f32.as_ptr() as *const _,
        (fresh_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output = kernel.cache_blend_f32(
        &cached_buf,
        &fresh_buf,
        1,
        config.match_len,
        config.fresh_len,
        config.num_heads,
        config.head_dim,
        0.0,
    ).map_err(|e| format!("Metal prompt cache blend failed: {}", e))?;

    let out_ptr = output.contents() as *const f32;
    let out_len = (config.match_len + config.fresh_len) * kv_size;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
    Ok(from_f32_vec(out_slice.to_vec()))
}

fn cuda_chunked_prefill_attention<T: KernelFloat>(
    kernel: &CudaChunkedPrefillKernel,
    stream: &Arc<CudaStream>,
    query: &[T],
    key: &[T],
    value: &[T],
    config: &ChunkedPrefillConfig,
) -> Result<ChunkedPrefillResult<T>, String> {
    if config.query_len == 0 || config.chunk_len == 0 || config.head_dim == 0 {
        return Err("query_len, chunk_len, head_dim must be > 0".into());
    }

    let q_len = config.batch_size * config.num_heads * config.query_len * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * config.chunk_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return Err("query/key/value length mismatch".into());
    }

    let q_f32 = to_f32_vec(query);
    let k_f32 = to_f32_vec(key);
    let v_f32 = to_f32_vec(value);
    let q_buf = stream.clone_htod(&q_f32)
        .map_err(|e| format!("Failed to copy query to GPU: {}", e))?;
    let k_buf = stream.clone_htod(&k_f32)
        .map_err(|e| format!("Failed to copy key to GPU: {}", e))?;
    let v_buf = stream.clone_htod(&v_f32)
        .map_err(|e| format!("Failed to copy value to GPU: {}", e))?;

    let (output_buf, lse_buf) = kernel.chunked_attention_f32(
        stream,
        &q_buf,
        &k_buf,
        &v_buf,
        config.batch_size,
        config.num_heads,
        config.query_len,
        config.chunk_len,
        config.head_dim,
        config.chunk_start,
        config.causal,
    ).map_err(|e| format!("CUDA chunked prefill attention failed: {}", e))?;

    let output_f32 = stream.clone_dtoh(&output_buf)
        .map_err(|e| format!("Failed to copy chunked output from GPU: {}", e))?;
    let lse = stream.clone_dtoh(&lse_buf)
        .map_err(|e| format!("Failed to copy chunked lse from GPU: {}", e))?;

    Ok(ChunkedPrefillResult {
        output: from_f32_vec(output_f32),
        log_sum_exp: lse,
    })
}

fn wgpu_chunked_prefill_attention<T: KernelFloat>(
    kernel: &WgpuChunkedPrefill,
    query: &[T],
    key: &[T],
    value: &[T],
    config: &ChunkedPrefillConfig,
) -> Result<ChunkedPrefillResult<T>, String> {
    if config.query_len == 0 || config.chunk_len == 0 || config.head_dim == 0 {
        return Err("query_len, chunk_len, head_dim must be > 0".into());
    }
    if !config.causal {
        return Err("WGPU chunked prefill only supports causal attention".into());
    }

    let q_len = config.batch_size * config.num_heads * config.query_len * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * config.chunk_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return Err("query/key/value length mismatch".into());
    }
    if config.query_len == 0 || config.chunk_start % config.query_len != 0 {
        return Err("chunk_start must align to query_len".into());
    }

    let chunk_idx = config.chunk_start / config.query_len;
    let q_f32 = to_f32_vec(query);
    let k_f32 = to_f32_vec(key);
    let v_f32 = to_f32_vec(value);
    let q_reordered = reorder_bhld_to_blhd(
        &q_f32,
        config.batch_size,
        config.num_heads,
        config.query_len,
        config.head_dim,
    );
    let k_reordered = reorder_bhld_to_blhd(
        &k_f32,
        config.batch_size,
        config.num_heads,
        config.chunk_len,
        config.head_dim,
    );
    let v_reordered = reorder_bhld_to_blhd(
        &v_f32,
        config.batch_size,
        config.num_heads,
        config.chunk_len,
        config.head_dim,
    );

    let result = kernel.chunked_attention_f32(
        &q_reordered,
        &k_reordered,
        &v_reordered,
        config.batch_size as u32,
        config.num_heads as u32,
        config.head_dim as u32,
        config.query_len as u32,
        config.chunk_len as u32,
        chunk_idx as u32,
    );

    let output = reorder_blhd_to_bhld(
        &result.output,
        config.batch_size,
        config.query_len,
        config.num_heads,
        config.head_dim,
    );
    let lse = reorder_bld_to_bhd(
        &result.lse,
        config.batch_size,
        config.query_len,
        config.num_heads,
    );

    Ok(ChunkedPrefillResult {
        output: from_f32_vec(output),
        log_sum_exp: lse,
    })
}

#[cfg(target_os = "macos")]
fn metal_chunked_prefill_attention<T: KernelFloat>(
    kernel: &MetalChunkedPrefillKernel,
    device: &metal::Device,
    query: &[T],
    key: &[T],
    value: &[T],
    config: &ChunkedPrefillConfig,
) -> Result<ChunkedPrefillResult<T>, String> {
    use metal::MTLResourceOptions;

    if config.query_len == 0 || config.chunk_len == 0 || config.head_dim == 0 {
        return Err("query_len, chunk_len, head_dim must be > 0".into());
    }

    let q_len = config.batch_size * config.num_heads * config.query_len * config.head_dim;
    let kv_len = config.batch_size * config.num_heads * config.chunk_len * config.head_dim;
    if query.len() != q_len || key.len() != kv_len || value.len() != kv_len {
        return Err("query/key/value length mismatch".into());
    }
    if config.query_len == 0 || config.chunk_start % config.query_len != 0 {
        return Err("chunk_start must align to query_len".into());
    }

    let chunk_idx = config.chunk_start / config.query_len;
    let q_f32 = to_f32_vec(query);
    let k_f32 = to_f32_vec(key);
    let v_f32 = to_f32_vec(value);
    let q_buf = device.new_buffer_with_data(
        q_f32.as_ptr() as *const _,
        (q_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_f32.as_ptr() as *const _,
        (k_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_f32.as_ptr() as *const _,
        (v_f32.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let (output_buf, lse_buf) = kernel.chunked_attention_f32(
        &q_buf,
        &k_buf,
        &v_buf,
        config.batch_size,
        config.num_heads,
        config.query_len,
        config.chunk_len,
        config.head_dim,
        chunk_idx,
        scale,
        config.causal,
    ).map_err(|e| format!("Metal chunked prefill attention failed: {}", e))?;

    let output_ptr = output_buf.contents() as *const f32;
    let output_len = q_len;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, output_len) };
    let lse_ptr = lse_buf.contents() as *const f32;
    let lse_len = config.batch_size * config.num_heads * config.query_len;
    let lse_slice = unsafe { std::slice::from_raw_parts(lse_ptr, lse_len) };

    Ok(ChunkedPrefillResult {
        output: from_f32_vec(output_slice.to_vec()),
        log_sum_exp: lse_slice.to_vec(),
    })
}

// =============================================================================
// Embedding Ops GPU Dispatch Functions
// =============================================================================

/// ROCm/HSA binary IP Hamming dispatch.
/// Returns Ok(scores) if GPU execution succeeded, Err to fallback to CPU.
#[cfg(target_os = "linux")]
fn rocm_binary_ip_hamming(
    kernel: &HsaEmbeddingOpsKernel,
    queue: &HsaQueueWrapper,
    queries: &[u32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> Result<Vec<i32>, String> {
    let agent = kernel.agent();

    let q_buf = HsaBuffer::from_slice(agent, queries)
        .map_err(|e| format!("Failed to allocate Q buffer: {}", e))?;
    let db_buf = HsaBuffer::from_slice(agent, database)
        .map_err(|e| format!("Failed to allocate DB buffer: {}", e))?;

    let result = kernel.binary_ip_hamming(queue, &q_buf, &db_buf, dim, num_queries, num_vectors)
        .map_err(|e| format!("HSA binary_ip_hamming failed: {}", e))?;

    result.to_vec()
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// ROCm/HSA binary IP asymmetric dispatch.
#[cfg(target_os = "linux")]
fn rocm_binary_ip_asymmetric(
    kernel: &HsaEmbeddingOpsKernel,
    queue: &HsaQueueWrapper,
    queries: &[f32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> Result<Vec<f32>, String> {
    let agent = kernel.agent();

    let q_buf = HsaBuffer::from_slice(agent, queries)
        .map_err(|e| format!("Failed to allocate Q buffer: {}", e))?;
    let db_buf = HsaBuffer::from_slice(agent, database)
        .map_err(|e| format!("Failed to allocate DB buffer: {}", e))?;

    let result = kernel.binary_ip_asymmetric(queue, &q_buf, &db_buf, dim, num_queries, num_vectors)
        .map_err(|e| format!("HSA binary_ip_asymmetric failed: {}", e))?;

    result.to_vec()
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// ROCm/HSA int8 dot product dispatch.
#[cfg(target_os = "linux")]
fn rocm_int8_dot_product(
    kernel: &HsaEmbeddingOpsKernel,
    queue: &HsaQueueWrapper,
    queries: &[u32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    let agent = kernel.agent();

    let q_buf = HsaBuffer::from_slice(agent, queries)
        .map_err(|e| format!("Failed to allocate Q buffer: {}", e))?;
    let db_buf = HsaBuffer::from_slice(agent, database)
        .map_err(|e| format!("Failed to allocate DB buffer: {}", e))?;

    let result = kernel.int8_dot_product(queue, &q_buf, &db_buf, dim, num_queries, num_vectors, scale)
        .map_err(|e| format!("HSA int8_dot_product failed: {}", e))?;

    result.to_vec()
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// ROCm/HSA int4 dot product dispatch.
#[cfg(target_os = "linux")]
fn rocm_int4_dot_product(
    kernel: &HsaEmbeddingOpsKernel,
    queue: &HsaQueueWrapper,
    queries: &[u32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
    scale: f32,
    zero_point: i32,
) -> Result<Vec<f32>, String> {
    let agent = kernel.agent();

    let q_buf = HsaBuffer::from_slice(agent, queries)
        .map_err(|e| format!("Failed to allocate Q buffer: {}", e))?;
    let db_buf = HsaBuffer::from_slice(agent, database)
        .map_err(|e| format!("Failed to allocate DB buffer: {}", e))?;

    let result = kernel.int4_dot_product(queue, &q_buf, &db_buf, dim, num_queries, num_vectors, scale, zero_point)
        .map_err(|e| format!("HSA int4_dot_product failed: {}", e))?;

    result.to_vec()
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// CUDA binary IP Hamming dispatch.
fn cuda_binary_ip_hamming(
    kernel: &CudaEmbeddingOpsKernel,
    stream: &Arc<CudaStream>,
    queries: &[u32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> Result<Vec<i32>, String> {
    let q_buf = stream.clone_htod(queries)
        .map_err(|e| format!("Failed to copy Q to GPU: {}", e))?;
    let db_buf = stream.clone_htod(database)
        .map_err(|e| format!("Failed to copy DB to GPU: {}", e))?;

    let result = kernel.binary_ip_hamming(stream, &q_buf, &db_buf, dim, num_queries, num_vectors)
        .map_err(|e| format!("CUDA binary_ip_hamming failed: {}", e))?;

    stream.clone_dtoh(&result)
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// CUDA binary IP asymmetric dispatch.
fn cuda_binary_ip_asymmetric(
    kernel: &CudaEmbeddingOpsKernel,
    stream: &Arc<CudaStream>,
    queries: &[f32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> Result<Vec<f32>, String> {
    let q_buf = stream.clone_htod(queries)
        .map_err(|e| format!("Failed to copy Q to GPU: {}", e))?;
    let db_buf = stream.clone_htod(database)
        .map_err(|e| format!("Failed to copy DB to GPU: {}", e))?;

    let result = kernel.binary_ip_asymmetric(stream, &q_buf, &db_buf, dim, num_queries, num_vectors)
        .map_err(|e| format!("CUDA binary_ip_asymmetric failed: {}", e))?;

    stream.clone_dtoh(&result)
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// CUDA int8 dot product dispatch.
fn cuda_int8_dot_product(
    kernel: &CudaEmbeddingOpsKernel,
    stream: &Arc<CudaStream>,
    queries: &[u32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    let q_buf = stream.clone_htod(queries)
        .map_err(|e| format!("Failed to copy Q to GPU: {}", e))?;
    let db_buf = stream.clone_htod(database)
        .map_err(|e| format!("Failed to copy DB to GPU: {}", e))?;

    let result = kernel.int8_dot_product(stream, &q_buf, &db_buf, dim, num_queries, num_vectors, scale)
        .map_err(|e| format!("CUDA int8_dot_product failed: {}", e))?;

    stream.clone_dtoh(&result)
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// CUDA int4 dot product dispatch.
fn cuda_int4_dot_product(
    kernel: &CudaEmbeddingOpsKernel,
    stream: &Arc<CudaStream>,
    queries: &[u32],
    database: &[u32],
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
    scale: f32,
    zero_point: i32,
) -> Result<Vec<f32>, String> {
    let q_buf = stream.clone_htod(queries)
        .map_err(|e| format!("Failed to copy Q to GPU: {}", e))?;
    let db_buf = stream.clone_htod(database)
        .map_err(|e| format!("Failed to copy DB to GPU: {}", e))?;

    let result = kernel.int4_dot_product(stream, &q_buf, &db_buf, dim, num_queries, num_vectors, scale, zero_point)
        .map_err(|e| format!("CUDA int4_dot_product failed: {}", e))?;

    stream.clone_dtoh(&result)
        .map_err(|e| format!("Failed to copy result from GPU: {}", e))
}

/// CUDA rerank pipeline: Binary Hamming → Int8 Dot Product → Top-K
fn cuda_rerank_pipeline(
    kernel: &CudaEmbeddingOpsKernel,
    stream: &Arc<CudaStream>,
    binary_query: &[u32],
    binary_database: &[u32],
    int8_query: &[u32],
    int8_database: &[u32],
    num_vectors: usize,
    config: &GpuRerankConfig,
    int8_scale: f32,
) -> Result<GpuRerankStageResult, String> {
    // Stage 1: Binary Hamming distance on GPU
    let binary_scores = cuda_binary_ip_hamming(
        kernel, stream, binary_query, binary_database,
        config.dim, 1, num_vectors,
    )?;

    // CPU Top-K selection (ascending - lower hamming = better)
    let mut indexed_scores: Vec<(usize, i32)> = binary_scores.iter().copied().enumerate().collect();
    indexed_scores.sort_by_key(|(_, score)| *score);
    indexed_scores.truncate(config.binary_k);

    let stage1_indices: Vec<u32> = indexed_scores.iter().map(|(i, _)| *i as u32).collect();

    // Stage 2: Gather int8 candidates and compute on GPU
    // int8 uses packed u32 (4x i8 per u32), so elements_per_vec = dim / 4
    let int8_packed_per_vec = config.dim / 4;
    let mut candidate_database = Vec::with_capacity(stage1_indices.len() * int8_packed_per_vec);
    for &idx in &stage1_indices {
        let start = idx as usize * int8_packed_per_vec;
        let end = start + int8_packed_per_vec;
        if end <= int8_database.len() {
            candidate_database.extend_from_slice(&int8_database[start..end]);
        }
    }

    // Int8 dot product on GPU
    let int8_scores = cuda_int8_dot_product(
        kernel, stream, int8_query, &candidate_database,
        config.dim, 1, stage1_indices.len(), int8_scale,
    )?;

    // Final CPU Top-K selection (descending - higher score = better)
    let mut final_indexed: Vec<(usize, f32)> = int8_scores.iter().copied().enumerate().collect();
    final_indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    final_indexed.truncate(config.int8_k);

    // Map back to original indices
    let indices: Vec<u32> = final_indexed.iter()
        .map(|(i, _)| stage1_indices[*i])
        .collect();
    let scores: Vec<f32> = final_indexed.iter()
        .map(|(_, s)| *s)
        .collect();

    Ok(GpuRerankStageResult { indices, scores })
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

    #[test]
    fn test_matmul_basic() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A * B = [[19, 22], [43, 50]]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        dispatcher.matmul(
            &a,
            &b,
            &mut c,
            MatmulConfig {
                m: 2,
                k: 2,
                n: 2,
                transpose_b: false,
                alpha: 1.0,
                beta: 0.0,
            },
        );

        assert!((c[0] - 19.0).abs() < 1e-5, "Expected 19.0, got {}", c[0]);
        assert!((c[1] - 22.0).abs() < 1e-5, "Expected 22.0, got {}", c[1]);
        assert!((c[2] - 43.0).abs() < 1e-5, "Expected 43.0, got {}", c[2]);
        assert!((c[3] - 50.0).abs() < 1e-5, "Expected 50.0, got {}", c[3]);
    }

    #[test]
    fn test_matmul_transposed() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        // A = [[1, 2], [3, 4]]  (2x2)
        // B^T = [[5, 7], [6, 8]]  (stored as B = [[5, 6], [7, 8]] i.e. [N=2, K=2])
        // C = A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17, 23], [39, 53]]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0]; // stored as [N, K]
        let mut c = vec![0.0f32; 4];

        dispatcher.matmul(
            &a,
            &b,
            &mut c,
            MatmulConfig {
                m: 2,
                k: 2,
                n: 2,
                transpose_b: true,
                alpha: 1.0,
                beta: 0.0,
            },
        );

        assert!((c[0] - 17.0).abs() < 1e-5, "Expected 17.0, got {}", c[0]);
        assert!((c[1] - 23.0).abs() < 1e-5, "Expected 23.0, got {}", c[1]);
        assert!((c[2] - 39.0).abs() < 1e-5, "Expected 39.0, got {}", c[2]);
        assert!((c[3] - 53.0).abs() < 1e-5, "Expected 53.0, got {}", c[3]);
    }

    #[test]
    fn test_matmul_alpha_beta() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C_init = [[1, 1], [1, 1]]
        // C = 2 * A * B + 0.5 * C_init = 2 * [[19, 22], [43, 50]] + [[0.5, 0.5], [0.5, 0.5]]
        //   = [[38.5, 44.5], [86.5, 100.5]]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![1.0f32; 4];

        dispatcher.matmul(
            &a,
            &b,
            &mut c,
            MatmulConfig {
                m: 2,
                k: 2,
                n: 2,
                transpose_b: false,
                alpha: 2.0,
                beta: 0.5,
            },
        );

        assert!((c[0] - 38.5).abs() < 1e-5, "Expected 38.5, got {}", c[0]);
        assert!((c[1] - 44.5).abs() < 1e-5, "Expected 44.5, got {}", c[1]);
        assert!((c[2] - 86.5).abs() < 1e-5, "Expected 86.5, got {}", c[2]);
        assert!((c[3] - 100.5).abs() < 1e-5, "Expected 100.5, got {}", c[3]);
    }

    #[test]
    fn test_matmul_f16() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        let a: Vec<half::f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let b: Vec<half::f16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let mut c = vec![half::f16::ZERO; 4];

        dispatcher.matmul(
            &a,
            &b,
            &mut c,
            MatmulConfig {
                m: 2,
                k: 2,
                n: 2,
                transpose_b: false,
                alpha: 1.0,
                beta: 0.0,
            },
        );

        // f16 has lower precision, use larger epsilon
        assert!((c[0].to_f32() - 19.0).abs() < 0.1, "Expected ~19.0, got {}", c[0].to_f32());
        assert!((c[1].to_f32() - 22.0).abs() < 0.1, "Expected ~22.0, got {}", c[1].to_f32());
        assert!((c[2].to_f32() - 43.0).abs() < 0.1, "Expected ~43.0, got {}", c[2].to_f32());
        assert!((c[3].to_f32() - 50.0).abs() < 0.1, "Expected ~50.0, got {}", c[3].to_f32());
    }

    #[test]
    fn test_add_bias() {
        let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

        // output = [[1, 2], [3, 4]] + bias = [10, 20]
        // result = [[11, 22], [13, 24]]
        let mut output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bias: Vec<f32> = vec![10.0, 20.0];

        dispatcher.add_bias(&mut output, &bias, 2, 2);

        assert!((output[0] - 11.0).abs() < 1e-5, "Expected 11.0, got {}", output[0]);
        assert!((output[1] - 22.0).abs() < 1e-5, "Expected 22.0, got {}", output[1]);
        assert!((output[2] - 13.0).abs() < 1e-5, "Expected 13.0, got {}", output[2]);
        assert!((output[3] - 24.0).abs() < 1e-5, "Expected 24.0, got {}", output[3]);
    }
}
