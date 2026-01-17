//! CUDA INT2 extreme quantization kernel.
//!
//! Implements 2-bit quantization for KV cache compression:
//! - Scale/zero-point quantization to INT2
//! - Group-wise quantization with configurable group size
//! - Efficient bit packing (16 values per u32)
//!
//! # SM-Aware PTX Loading
//!
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 80 (Ampere): A100, RTX 30 series and higher
//!
//! 🚨 **Fat Binary Only**: NO runtime compilation fallback.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

// Kernel function names
const KERNEL_QUANTIZE_F32: &str = "int2_quantize_f32";
const KERNEL_QUANTIZE_F16: &str = "int2_quantize_f16";
const KERNEL_DEQUANTIZE_F32: &str = "int2_dequantize_f32";
const KERNEL_DEQUANTIZE_F16: &str = "int2_dequantize_f16";
const KERNEL_PACK_INT2: &str = "int2_pack";
const KERNEL_UNPACK_INT2: &str = "int2_unpack";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for INT2 quantizer kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static INT2_QUANTIZER_PTX: PtxCollection = PtxCollection {
    kernel_name: "int2_quantizer",
    ptx_versions: &[
        (61, include_str!("kernels/int2_quantizer_sm61.ptx")),
        (80, include_str!("kernels/int2_quantizer.ptx")),
    ],
};

/// Errors surfaced by the CUDA INT2 quantizer kernels.
#[derive(Debug)]
pub enum Int2QuantizerError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Dimension mismatch.
    DimensionMismatch(String),
}

impl fmt::Display for Int2QuantizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
        }
    }
}

impl std::error::Error for Int2QuantizerError {}

impl From<DriverError> for Int2QuantizerError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for Int2QuantizerError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// INT2 quantizer CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - FP32/FP16 to INT2 quantization
/// - INT2 to FP32/FP16 dequantization
/// - Efficient bit packing/unpacking
pub struct Int2QuantizerKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Quantization kernels
    kernel_quantize_f32: CudaFunction,
    kernel_quantize_f16: CudaFunction,
    // Dequantization kernels
    kernel_dequantize_f32: CudaFunction,
    kernel_dequantize_f16: CudaFunction,
    // Packing kernels
    kernel_pack: CudaFunction,
    kernel_unpack: CudaFunction,
}

impl Int2QuantizerKernel {
    /// Load INT2 quantizer kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Int2QuantizerError> {
        let ptx = INT2_QUANTIZER_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_quantize_f32 = module
            .load_function(KERNEL_QUANTIZE_F32)
            .map_err(|_| Int2QuantizerError::KernelMissing(KERNEL_QUANTIZE_F32))?;
        let kernel_quantize_f16 = module
            .load_function(KERNEL_QUANTIZE_F16)
            .map_err(|_| Int2QuantizerError::KernelMissing(KERNEL_QUANTIZE_F16))?;
        let kernel_dequantize_f32 = module
            .load_function(KERNEL_DEQUANTIZE_F32)
            .map_err(|_| Int2QuantizerError::KernelMissing(KERNEL_DEQUANTIZE_F32))?;
        let kernel_dequantize_f16 = module
            .load_function(KERNEL_DEQUANTIZE_F16)
            .map_err(|_| Int2QuantizerError::KernelMissing(KERNEL_DEQUANTIZE_F16))?;
        let kernel_pack = module
            .load_function(KERNEL_PACK_INT2)
            .map_err(|_| Int2QuantizerError::KernelMissing(KERNEL_PACK_INT2))?;
        let kernel_unpack = module
            .load_function(KERNEL_UNPACK_INT2)
            .map_err(|_| Int2QuantizerError::KernelMissing(KERNEL_UNPACK_INT2))?;

        Ok(Self {
            module,
            kernel_quantize_f32,
            kernel_quantize_f16,
            kernel_dequantize_f32,
            kernel_dequantize_f16,
            kernel_pack,
            kernel_unpack,
        })
    }

    /// Quantize FP32 tensor to INT2 with group-wise scaling.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `input` - Input tensor [num_elements]
    /// * `group_size` - Number of elements per quantization group
    /// * `num_elements` - Total number of elements
    ///
    /// # Returns
    /// Tuple of (quantized [num_elements], scales [num_groups], zeros [num_groups])
    pub fn quantize_f32(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        group_size: usize,
        num_elements: usize,
    ) -> Result<(CudaSlice<i8>, CudaSlice<f32>, CudaSlice<f32>), Int2QuantizerError> {
        let num_groups = (num_elements + group_size - 1) / group_size;

        let mut quantized: CudaSlice<i8> = stream.alloc_zeros(num_elements)?;
        let mut scales: CudaSlice<f32> = stream.alloc_zeros(num_groups)?;
        let mut zeros: CudaSlice<f32> = stream.alloc_zeros(num_groups)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (num_groups + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (group_size * std::mem::size_of::<f32>()) as u32,
        };

        let num_elements_i32 = num_elements as i32;
        let group_size_i32 = group_size as i32;
        let num_groups_i32 = num_groups as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_quantize_f32);
            builder.arg(input);
            builder.arg(&mut quantized);
            builder.arg(&mut scales);
            builder.arg(&mut zeros);
            builder.arg(&num_elements_i32);
            builder.arg(&group_size_i32);
            builder.arg(&num_groups_i32);
            builder.launch(cfg)?;
        }

        Ok((quantized, scales, zeros))
    }

    /// Quantize FP16 tensor to INT2 with group-wise scaling.
    pub fn quantize_f16(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f16>,
        group_size: usize,
        num_elements: usize,
    ) -> Result<(CudaSlice<i8>, CudaSlice<f16>, CudaSlice<f16>), Int2QuantizerError> {
        let num_groups = (num_elements + group_size - 1) / group_size;

        let mut quantized: CudaSlice<i8> = stream.alloc_zeros(num_elements)?;
        let mut scales: CudaSlice<f16> = stream.alloc_zeros(num_groups)?;
        let mut zeros: CudaSlice<f16> = stream.alloc_zeros(num_groups)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (num_groups + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (group_size * std::mem::size_of::<f16>()) as u32,
        };

        let num_elements_i32 = num_elements as i32;
        let group_size_i32 = group_size as i32;
        let num_groups_i32 = num_groups as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_quantize_f16);
            builder.arg(input);
            builder.arg(&mut quantized);
            builder.arg(&mut scales);
            builder.arg(&mut zeros);
            builder.arg(&num_elements_i32);
            builder.arg(&group_size_i32);
            builder.arg(&num_groups_i32);
            builder.launch(cfg)?;
        }

        Ok((quantized, scales, zeros))
    }

    /// Dequantize INT2 tensor back to FP32.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `quantized` - Quantized tensor [num_elements]
    /// * `scales` - Scale factors [num_groups]
    /// * `zeros` - Zero points [num_groups]
    /// * `group_size` - Number of elements per quantization group
    /// * `num_elements` - Total number of elements
    ///
    /// # Returns
    /// Dequantized tensor [num_elements]
    pub fn dequantize_f32(
        &self,
        stream: &Arc<CudaStream>,
        quantized: &CudaSlice<i8>,
        scales: &CudaSlice<f32>,
        zeros: &CudaSlice<f32>,
        group_size: usize,
        num_elements: usize,
    ) -> Result<CudaSlice<f32>, Int2QuantizerError> {
        let mut output: CudaSlice<f32> = stream.alloc_zeros(num_elements)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (num_elements + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_elements_i32 = num_elements as i32;
        let group_size_i32 = group_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_dequantize_f32);
            builder.arg(quantized);
            builder.arg(scales);
            builder.arg(zeros);
            builder.arg(&mut output);
            builder.arg(&num_elements_i32);
            builder.arg(&group_size_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Dequantize INT2 tensor back to FP16.
    pub fn dequantize_f16(
        &self,
        stream: &Arc<CudaStream>,
        quantized: &CudaSlice<i8>,
        scales: &CudaSlice<f16>,
        zeros: &CudaSlice<f16>,
        group_size: usize,
        num_elements: usize,
    ) -> Result<CudaSlice<f16>, Int2QuantizerError> {
        let mut output: CudaSlice<f16> = stream.alloc_zeros(num_elements)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (num_elements + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_elements_i32 = num_elements as i32;
        let group_size_i32 = group_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_dequantize_f16);
            builder.arg(quantized);
            builder.arg(scales);
            builder.arg(zeros);
            builder.arg(&mut output);
            builder.arg(&num_elements_i32);
            builder.arg(&group_size_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Pack INT2 values into u32 (16 values per u32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `input` - Unpacked INT2 values [num_elements]
    /// * `num_elements` - Number of elements (must be multiple of 16)
    ///
    /// # Returns
    /// Packed tensor [num_elements / 16]
    pub fn pack_int2(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<i8>,
        num_elements: usize,
    ) -> Result<CudaSlice<u32>, Int2QuantizerError> {
        if num_elements % 16 != 0 {
            return Err(Int2QuantizerError::DimensionMismatch(
                "num_elements must be multiple of 16 for INT2 packing".into(),
            ));
        }

        let packed_size = num_elements / 16;
        let mut output: CudaSlice<u32> = stream.alloc_zeros(packed_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (packed_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let packed_size_i32 = packed_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_pack);
            builder.arg(input);
            builder.arg(&mut output);
            builder.arg(&packed_size_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Unpack u32 to INT2 values (16 values per u32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `packed` - Packed tensor [packed_size]
    /// * `num_elements` - Target number of elements (packed_size * 16)
    ///
    /// # Returns
    /// Unpacked INT2 values [num_elements]
    pub fn unpack_int2(
        &self,
        stream: &Arc<CudaStream>,
        packed: &CudaSlice<u32>,
        num_elements: usize,
    ) -> Result<CudaSlice<i8>, Int2QuantizerError> {
        if num_elements % 16 != 0 {
            return Err(Int2QuantizerError::DimensionMismatch(
                "num_elements must be multiple of 16 for INT2 unpacking".into(),
            ));
        }

        let mut output: CudaSlice<i8> = stream.alloc_zeros(num_elements)?;
        let packed_size = num_elements / 16;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (packed_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let packed_size_i32 = packed_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_unpack);
            builder.arg(packed);
            builder.arg(&mut output);
            builder.arg(&packed_size_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}

/// Configuration for INT2 quantizer CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct Int2QuantizerCudaConfig {
    /// Group size for quantization.
    pub group_size: usize,
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
    /// Clipping percentile for outliers (0.0 = no clipping).
    pub clip_percentile: f32,
}

impl Default for Int2QuantizerCudaConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            symmetric: false,
            clip_percentile: 0.0,
        }
    }
}

impl Int2QuantizerCudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), Int2QuantizerError> {
        if self.group_size == 0 {
            return Err(Int2QuantizerError::InvalidConfig(
                "group_size must be > 0".into(),
            ));
        }
        if self.clip_percentile < 0.0 || self.clip_percentile > 1.0 {
            return Err(Int2QuantizerError::InvalidConfig(
                "clip_percentile must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Compute number of groups for given number of elements.
    pub fn num_groups(&self, num_elements: usize) -> usize {
        (num_elements + self.group_size - 1) / self.group_size
    }

    /// Compute packed size (number of u32s needed).
    pub fn packed_size(&self, num_elements: usize) -> usize {
        (num_elements + 15) / 16
    }
}
