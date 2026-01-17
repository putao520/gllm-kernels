//! Metal INT2 extreme quantization kernels.
//!
//! This module provides Metal GPU-accelerated kernels for INT2 quantization:
//! - Group-wise INT2 quantization with scale factors
//! - INT2 dequantization back to floating point
//! - Efficient bit packing/unpacking
//!
//! ## Precompiled metallib (Required)
//!
//! metallib must be precompiled before use:
//! ```bash
//! ./scripts/compile_metal_kernels.sh
//! ```
//!
//! metallib is Metal's intermediate format (like PTX/HSACO).
//! NO runtime compilation fallback - metallib must be precompiled and embedded.

use std::fmt;
use std::mem;
use std::os::raw::c_void;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

use crate::metal_kernels::metallib_loader::{MetallibCollection, MetallibLoadError};

const KERNEL_QUANTIZE_F32: &str = "int2_quantize_f32";
const KERNEL_QUANTIZE_F16: &str = "int2_quantize_f16";
const KERNEL_DEQUANTIZE_F32: &str = "int2_dequantize_f32";
const KERNEL_DEQUANTIZE_F16: &str = "int2_dequantize_f16";
const KERNEL_PACK_INT2: &str = "int2_pack";
const KERNEL_UNPACK_INT2: &str = "int2_unpack";

/// Metallib collection for INT2 quantizer kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static INT2_QUANTIZER_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "int2_quantizer",
    metallib_data: include_bytes!("kernels/int2_quantizer.metallib"),
};

/// Parameters for quantization kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct QuantizeParams {
    num_elements: u32,
    group_size: u32,
    num_groups: u32,
    _pad: u32,
}

/// Parameters for dequantization kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct DequantizeParams {
    num_elements: u32,
    group_size: u32,
    num_groups: u32,
    _pad: u32,
}

/// Parameters for pack/unpack kernels.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PackParams {
    num_int2_values: u32,
    _pad: [u32; 3],
}

/// Errors surfaced by the Metal INT2 quantizer kernels.
#[derive(Debug)]
pub enum Int2QuantizerError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for Int2QuantizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for Int2QuantizerError {}

impl From<MetallibLoadError> for Int2QuantizerError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for INT2 quantization operations.
#[derive(Clone, Debug)]
pub struct Int2QuantizerConfig {
    /// Group size for quantization (elements per scale factor).
    pub group_size: usize,
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
}

impl Default for Int2QuantizerConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            symmetric: true,
        }
    }
}

impl Int2QuantizerConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), Int2QuantizerError> {
        if self.group_size == 0 {
            return Err(Int2QuantizerError::InvalidConfig("group_size must be positive".into()));
        }
        if self.group_size % 4 != 0 {
            return Err(Int2QuantizerError::InvalidConfig("group_size must be multiple of 4".into()));
        }
        Ok(())
    }
}

/// INT2 Quantizer Metal kernel wrapper.
pub struct Int2QuantizerKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_quantize_f32: ComputePipelineState,
    pipeline_quantize_f16: ComputePipelineState,
    pipeline_dequantize_f32: ComputePipelineState,
    pipeline_dequantize_f16: ComputePipelineState,
    pipeline_pack: ComputePipelineState,
    pipeline_unpack: ComputePipelineState,
}

impl Int2QuantizerKernel {
    /// Load INT2 quantizer kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, Int2QuantizerError> {
        let library = load_library(device)?;

        let pipeline_quantize_f32 = build_pipeline(device, &library, KERNEL_QUANTIZE_F32)?;
        let pipeline_quantize_f16 = build_pipeline(device, &library, KERNEL_QUANTIZE_F16)?;
        let pipeline_dequantize_f32 = build_pipeline(device, &library, KERNEL_DEQUANTIZE_F32)?;
        let pipeline_dequantize_f16 = build_pipeline(device, &library, KERNEL_DEQUANTIZE_F16)?;
        let pipeline_pack = build_pipeline(device, &library, KERNEL_PACK_INT2)?;
        let pipeline_unpack = build_pipeline(device, &library, KERNEL_UNPACK_INT2)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_quantize_f32,
            pipeline_quantize_f16,
            pipeline_dequantize_f32,
            pipeline_dequantize_f16,
            pipeline_pack,
            pipeline_unpack,
        })
    }

    /// Quantize f32 tensor to INT2 with group-wise scaling.
    ///
    /// # Returns
    /// (quantized_packed, scales): Packed INT2 values and scale factors per group
    pub fn quantize_f32(
        &self,
        input: &Buffer,
        num_elements: usize,
        group_size: usize,
    ) -> Result<(Buffer, Buffer), Int2QuantizerError> {
        self.quantize_impl(
            input,
            num_elements,
            group_size,
            &self.pipeline_quantize_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Quantize f16 tensor to INT2 with group-wise scaling.
    pub fn quantize_f16(
        &self,
        input: &Buffer,
        num_elements: usize,
        group_size: usize,
    ) -> Result<(Buffer, Buffer), Int2QuantizerError> {
        self.quantize_impl(
            input,
            num_elements,
            group_size,
            &self.pipeline_quantize_f16,
            mem::size_of::<u16>(),
        )
    }

    fn quantize_impl(
        &self,
        input: &Buffer,
        num_elements: usize,
        group_size: usize,
        pipeline: &ComputePipelineState,
        scale_element_size: usize,
    ) -> Result<(Buffer, Buffer), Int2QuantizerError> {
        if num_elements % group_size != 0 {
            return Err(Int2QuantizerError::InvalidConfig(
                "num_elements must be divisible by group_size".into(),
            ));
        }

        let num_groups = num_elements / group_size;

        let params = QuantizeParams {
            num_elements: num_elements as u32,
            group_size: group_size as u32,
            num_groups: num_groups as u32,
            _pad: 0,
        };

        // Output: packed INT2 (4 values per byte) and scales
        let packed_bytes = (num_elements / 4) as u64;
        let scales_bytes = (num_groups * scale_element_size) as u64;

        let quantized = self.device.new_buffer(packed_bytes, MTLResourceOptions::StorageModeShared);
        let scales = self.device.new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(&quantized), 0);
        encoder.set_buffer(2, Some(&scales), 0);

        let params_size = mem::size_of::<QuantizeParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_groups as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((quantized, scales))
    }

    /// Dequantize INT2 tensor back to f32.
    pub fn dequantize_f32(
        &self,
        quantized: &Buffer,
        scales: &Buffer,
        num_elements: usize,
        group_size: usize,
    ) -> Result<Buffer, Int2QuantizerError> {
        self.dequantize_impl(
            quantized,
            scales,
            num_elements,
            group_size,
            &self.pipeline_dequantize_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Dequantize INT2 tensor back to f16.
    pub fn dequantize_f16(
        &self,
        quantized: &Buffer,
        scales: &Buffer,
        num_elements: usize,
        group_size: usize,
    ) -> Result<Buffer, Int2QuantizerError> {
        self.dequantize_impl(
            quantized,
            scales,
            num_elements,
            group_size,
            &self.pipeline_dequantize_f16,
            mem::size_of::<u16>(),
        )
    }

    fn dequantize_impl(
        &self,
        quantized: &Buffer,
        scales: &Buffer,
        num_elements: usize,
        group_size: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, Int2QuantizerError> {
        if num_elements % group_size != 0 {
            return Err(Int2QuantizerError::InvalidConfig(
                "num_elements must be divisible by group_size".into(),
            ));
        }

        let num_groups = num_elements / group_size;

        let params = DequantizeParams {
            num_elements: num_elements as u32,
            group_size: group_size as u32,
            num_groups: num_groups as u32,
            _pad: 0,
        };

        let output_bytes = (num_elements * element_size) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(quantized), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let params_size = mem::size_of::<DequantizeParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_groups as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Pack INT2 values (4 values per byte).
    pub fn pack_int2(
        &self,
        unpacked: &Buffer,
        num_values: usize,
    ) -> Result<Buffer, Int2QuantizerError> {
        if num_values % 4 != 0 {
            return Err(Int2QuantizerError::InvalidConfig(
                "num_values must be divisible by 4".into(),
            ));
        }

        let params = PackParams {
            num_int2_values: num_values as u32,
            _pad: [0; 3],
        };

        let packed_bytes = (num_values / 4) as u64;
        let packed = self.device.new_buffer(packed_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_pack);
        encoder.set_buffer(0, Some(unpacked), 0);
        encoder.set_buffer(1, Some(&packed), 0);

        let params_size = mem::size_of::<PackParams>() as u64;
        encoder.set_bytes(2, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((num_values / 4) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_pack);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(packed)
    }

    /// Unpack INT2 values (4 values per byte to individual bytes).
    pub fn unpack_int2(
        &self,
        packed: &Buffer,
        num_values: usize,
    ) -> Result<Buffer, Int2QuantizerError> {
        if num_values % 4 != 0 {
            return Err(Int2QuantizerError::InvalidConfig(
                "num_values must be divisible by 4".into(),
            ));
        }

        let params = PackParams {
            num_int2_values: num_values as u32,
            _pad: [0; 3],
        };

        let unpacked_bytes = num_values as u64;
        let unpacked = self.device.new_buffer(unpacked_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_unpack);
        encoder.set_buffer(0, Some(packed), 0);
        encoder.set_buffer(1, Some(&unpacked), 0);

        let params_size = mem::size_of::<PackParams>() as u64;
        encoder.set_bytes(2, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((num_values / 4) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_unpack);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unpacked)
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, Int2QuantizerError> {
    INT2_QUANTIZER_METALLIB.load(device).map_err(Int2QuantizerError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, Int2QuantizerError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| Int2QuantizerError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(Int2QuantizerError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
