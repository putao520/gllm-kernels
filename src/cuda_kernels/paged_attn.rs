//! CUDA paged attention kernels using cudarc.
//!
//! The kernel implements paged KV cache access with online softmax accumulation.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::f16;

const KERNEL_F32: &str = "paged_attention_forward_f32";
const KERNEL_F16: &str = "paged_attention_forward_f16";
const MAX_HEAD_DIM: usize = 256;
const DEFAULT_BLOCK: u32 = 128;

const PRECOMPILED_PTX: &str = include_str!("kernels/paged_attention.ptx");
const KERNEL_SOURCE: &str = include_str!("kernels/paged_attention.cu");

/// Errors surfaced by the CUDA paged attention kernels.
#[derive(Debug)]
pub enum PagedAttentionError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
}

impl fmt::Display for PagedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
        }
    }
}

impl std::error::Error for PagedAttentionError {}

impl From<DriverError> for PagedAttentionError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

/// Paged attention CUDA kernel wrapper.
pub struct PagedAttentionKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
    kernel_f16: CudaFunction,
}

impl PagedAttentionKernel {
    /// Load a paged attention kernel module on the given device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, PagedAttentionError> {
        let ptx = load_ptx()?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| PagedAttentionError::KernelMissing(KERNEL_F32))?;
        let kernel_f16 = module
            .load_function(KERNEL_F16)
            .map_err(|_| PagedAttentionError::KernelMissing(KERNEL_F16))?;

        Ok(Self {
            module,
            kernel_f32,
            kernel_f16,
        })
    }

    /// Paged attention forward for f16 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f16>,
        k_cache: &CudaSlice<f16>,
        v_cache: &CudaSlice<f16>,
        block_tables: &CudaSlice<i32>,
        block_offsets: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f16>, PagedAttentionError> {
        self.forward_f16(
            stream,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
        )
    }

    /// Paged attention forward for f32 inputs.
    pub fn forward_f32(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        block_tables: &CudaSlice<i32>,
        block_offsets: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, PagedAttentionError> {
        self.forward_f32_with_block(
            stream,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
            DEFAULT_BLOCK,
        )
    }

    /// Paged attention forward for f16 inputs.
    pub fn forward_f16(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f16>,
        k_cache: &CudaSlice<f16>,
        v_cache: &CudaSlice<f16>,
        block_tables: &CudaSlice<i32>,
        block_offsets: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f16>, PagedAttentionError> {
        self.forward_f16_with_block(
            stream,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
            DEFAULT_BLOCK,
        )
    }

    fn forward_f32_with_block(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        block_tables: &CudaSlice<i32>,
        block_offsets: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
        block_size: u32,
    ) -> Result<CudaSlice<f32>, PagedAttentionError> {
        let (output_len, cfg) = build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let page_block_i32 = page_block_size as i32;
        let seq_len_i32 = seq_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(q);
            builder.arg(k_cache);
            builder.arg(v_cache);
            builder.arg(block_tables);
            builder.arg(block_offsets);
            builder.arg(&mut output);
            builder.arg(&batch_size_i32);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&page_block_i32);
            builder.arg(&seq_len_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    fn forward_f16_with_block(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f16>,
        k_cache: &CudaSlice<f16>,
        v_cache: &CudaSlice<f16>,
        block_tables: &CudaSlice<i32>,
        block_offsets: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
        block_size: u32,
    ) -> Result<CudaSlice<f16>, PagedAttentionError> {
        let (output_len, cfg) = build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_len)?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let page_block_i32 = page_block_size as i32;
        let seq_len_i32 = seq_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f16);
            builder.arg(q);
            builder.arg(k_cache);
            builder.arg(v_cache);
            builder.arg(block_tables);
            builder.arg(block_offsets);
            builder.arg(&mut output);
            builder.arg(&batch_size_i32);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&page_block_i32);
            builder.arg(&seq_len_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}

fn load_ptx() -> Result<Ptx, PagedAttentionError> {
    if let Ok(path) = std::env::var("GLLM_PAGED_ATTN_PTX") {
        return Ok(Ptx::from_file(path));
    }

    if !PRECOMPILED_PTX.contains("Placeholder") {
        return Ok(Ptx::from_src(PRECOMPILED_PTX));
    }

    #[cfg(feature = "nvrtc")]
    {
        use cudarc::nvrtc::compile_ptx;
        return compile_ptx(KERNEL_SOURCE).map_err(|e| {
            PagedAttentionError::InvalidConfig(format!("NVRTC compilation failed: {}", e))
        });
    }

    #[cfg(not(feature = "nvrtc"))]
    Err(PagedAttentionError::InvalidConfig(
        "PTX is a placeholder. Either: \n\
         1. Compile with: nvcc -ptx -arch=sm_61 paged_attention.cu -o paged_attention.ptx\n\
         2. Set GLLM_PAGED_ATTN_PTX=/path/to/compiled.ptx\n\
         3. Enable 'nvrtc' feature and ensure CUDA toolkit is installed".into(),
    ))
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: u32,
) -> Result<(usize, LaunchConfig), PagedAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(PagedAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(PagedAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }
    if batch_size > i32::MAX as usize
        || num_heads > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || head_dim > i32::MAX as usize
    {
        return Err(PagedAttentionError::InvalidConfig(
            "Dimensions exceed i32::MAX".into(),
        ));
    }

    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| PagedAttentionError::InvalidConfig("num_queries overflow".into()))?;
    let output_len = num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| PagedAttentionError::InvalidConfig("output_len overflow".into()))?;

    let block_dim = block_size.clamp(1, 1024) as usize;
    let grid_dim = (num_queries + block_dim - 1) / block_dim;
    let grid_dim = u32::try_from(grid_dim)
        .map_err(|_| PagedAttentionError::InvalidConfig("grid_dim exceeds u32::MAX".into()))?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    Ok((output_len, cfg))
}
