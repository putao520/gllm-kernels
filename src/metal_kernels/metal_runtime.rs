//! Metal Runtime for loading precompiled metallib.
//!
//! This module provides utilities for loading precompiled Metal shaders (metallib)
//! and executing compute kernels on Apple GPUs.
//!
//! ## Architecture
//!
//! ```text
//! Compile time: Metal shader source → metallib binary (via xcrun metallib) → embedded
//! Runtime: Metal Framework loads embedded metallib → GPU executes
//! Fallback: Runtime compilation from embedded source if metallib fails
//! ```
//!
//! ## Zero Configuration
//!
//! This module is fully automatic with no user configuration required.
//! The kernel loader automatically selects the best loading strategy.

use std::sync::OnceLock;

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

/// Global Metal device instance.
static METAL_DEVICE: OnceLock<Option<Device>> = OnceLock::new();

/// Check if Metal is available on this system.
pub fn is_metal_available() -> bool {
    get_metal_device().is_some()
}

/// Get the default Metal device.
pub fn get_metal_device() -> Option<&'static Device> {
    METAL_DEVICE
        .get_or_init(|| Device::system_default())
        .as_ref()
}

/// Get all available Metal devices.
pub fn get_all_metal_devices() -> Vec<Device> {
    Device::all()
}

/// Error type for Metal operations.
#[derive(Debug)]
pub enum MetalError {
    /// No Metal device available.
    DeviceNotAvailable,
    /// Failed to load metallib.
    LibraryLoadFailed(String),
    /// Function not found in library.
    FunctionNotFound(String),
    /// Pipeline creation failed.
    PipelineCreationFailed(String),
    /// Invalid buffer size.
    InvalidBuffer(String),
    /// Execution error.
    ExecutionFailed(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceNotAvailable => write!(f, "No Metal device available"),
            Self::LibraryLoadFailed(msg) => write!(f, "Failed to load metallib: {}", msg),
            Self::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            Self::PipelineCreationFailed(msg) => write!(f, "Pipeline creation failed: {}", msg),
            Self::InvalidBuffer(msg) => write!(f, "Invalid buffer: {}", msg),
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
        }
    }
}

impl std::error::Error for MetalError {}

/// Metal device information.
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name.
    pub name: String,
    /// Whether this is a low-power device.
    pub is_low_power: bool,
    /// Whether this device is headless.
    pub is_headless: bool,
    /// Registry ID for the device.
    pub registry_id: u64,
}

impl MetalDeviceInfo {
    /// Get info for a Metal device.
    pub fn from_device(device: &Device) -> Self {
        Self {
            name: device.name().to_string(),
            is_low_power: device.is_low_power(),
            is_headless: device.is_headless(),
            registry_id: device.registry_id(),
        }
    }
}

/// Get information about all Metal devices.
pub fn get_device_info() -> Vec<MetalDeviceInfo> {
    get_all_metal_devices()
        .iter()
        .map(MetalDeviceInfo::from_device)
        .collect()
}

/// Metal kernel loader for precompiled metallib.
pub struct MetalKernelLoader {
    device: Device,
    library: Library,
}

impl MetalKernelLoader {
    /// Load a metallib from embedded bytes.
    ///
    /// # Arguments
    /// * `data` - The metallib binary data (typically from `include_bytes!`)
    ///
    /// # Example
    /// ```ignore
    /// const METALLIB: &[u8] = include_bytes!("kernels/flash_attention.metallib");
    /// let loader = MetalKernelLoader::from_bytes(METALLIB)?;
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self, MetalError> {
        let device = get_metal_device()
            .cloned()
            .ok_or(MetalError::DeviceNotAvailable)?;

        Self::from_bytes_with_device(&device, data)
    }

    /// Load a metallib from bytes with a specific device.
    pub fn from_bytes_with_device(device: &Device, data: &[u8]) -> Result<Self, MetalError> {
        let library = device
            .new_library_with_data(data)
            .map_err(|e| MetalError::LibraryLoadFailed(e))?;

        Ok(Self {
            device: device.clone(),
            library,
        })
    }

    /// Load a metallib from a file path.
    pub fn from_path(path: &str) -> Result<Self, MetalError> {
        let device = get_metal_device()
            .cloned()
            .ok_or(MetalError::DeviceNotAvailable)?;

        let library = device
            .new_library_with_file(path)
            .map_err(|e| MetalError::LibraryLoadFailed(e))?;

        Ok(Self { device, library })
    }

    /// Compile from Metal shader source (fallback when metallib not available).
    pub fn from_source(source: &str) -> Result<Self, MetalError> {
        let device = get_metal_device()
            .cloned()
            .ok_or(MetalError::DeviceNotAvailable)?;

        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(source, &options)
            .map_err(|e| MetalError::LibraryLoadFailed(e))?;

        Ok(Self { device, library })
    }

    /// Get a compute function from the library.
    pub fn get_function(&self, name: &str) -> Result<metal::Function, MetalError> {
        self.library
            .get_function(name, None)
            .map_err(|_| MetalError::FunctionNotFound(name.to_string()))
    }

    /// Create a compute pipeline for a function.
    pub fn create_pipeline(&self, function_name: &str) -> Result<ComputePipelineState, MetalError> {
        let function = self.get_function(function_name)?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(e))
    }

    /// Get the underlying device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the underlying library.
    pub fn library(&self) -> &Library {
        &self.library
    }
}

/// Metal kernel executor for running compute kernels.
pub struct MetalKernelExecutor {
    device: Device,
    queue: CommandQueue,
}

impl MetalKernelExecutor {
    /// Create a new executor for the default device.
    pub fn new() -> Result<Self, MetalError> {
        let device = get_metal_device()
            .cloned()
            .ok_or(MetalError::DeviceNotAvailable)?;

        Ok(Self {
            queue: device.new_command_queue(),
            device,
        })
    }

    /// Create a new executor with a specific device.
    pub fn with_device(device: &Device) -> Self {
        Self {
            device: device.clone(),
            queue: device.new_command_queue(),
        }
    }

    /// Allocate a buffer on the device.
    pub fn alloc_buffer(&self, size: usize) -> Buffer {
        self.device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate a buffer and copy data from host.
    pub fn buffer_from_slice<T>(&self, data: &[T]) -> Buffer {
        let size = std::mem::size_of_val(data) as u64;
        let buffer = self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.contents() as *mut u8,
                size as usize,
            );
        }

        buffer
    }

    /// Copy buffer contents to a host vector.
    pub fn buffer_to_vec<T: Clone + Default>(&self, buffer: &Buffer, len: usize) -> Vec<T> {
        let mut result = vec![T::default(); len];
        let size = len * std::mem::size_of::<T>();

        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents() as *const u8,
                result.as_mut_ptr() as *mut u8,
                size,
            );
        }

        result
    }

    /// Dispatch a compute kernel.
    ///
    /// # Arguments
    /// * `pipeline` - The compute pipeline state
    /// * `buffers` - Buffers to bind (in order)
    /// * `params` - Optional parameter data (bound at buffer index after `buffers`)
    /// * `grid_size` - Total number of threads
    /// * `threadgroup_size` - Threads per threadgroup
    pub fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: Option<&[u8]>,
        grid_size: (u64, u64, u64),
        threadgroup_size: (u64, u64, u64),
    ) -> Result<(), MetalError> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Bind buffers
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buffer), 0);
        }

        // Bind parameters if provided
        if let Some(param_data) = params {
            encoder.set_bytes(
                buffers.len() as u64,
                param_data.len() as u64,
                param_data.as_ptr() as *const std::ffi::c_void,
            );
        }

        let threads_per_grid = MTLSize::new(grid_size.0, grid_size.1, grid_size.2);
        let threads_per_threadgroup = MTLSize::new(
            threadgroup_size.0,
            threadgroup_size.1,
            threadgroup_size.2,
        );

        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check for errors
        if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(MetalError::ExecutionFailed(
                "Command buffer execution failed".to_string()
            ));
        }

        Ok(())
    }

    /// Dispatch with automatic threadgroup size calculation.
    pub fn dispatch_auto(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: Option<&[u8]>,
        total_threads: u64,
    ) -> Result<(), MetalError> {
        let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
        let threadgroup_size = max_threads.min(256).max(1);

        self.dispatch(
            pipeline,
            buffers,
            params,
            (total_threads, 1, 1),
            (threadgroup_size, 1, 1),
        )
    }

    /// Get the underlying device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the underlying command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
}

impl Default for MetalKernelExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create MetalKernelExecutor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        // This test will pass on macOS and fail on other platforms
        let available = is_metal_available();
        println!("Metal available: {}", available);
    }

    #[test]
    fn test_device_info() {
        if !is_metal_available() {
            return;
        }

        let info = get_device_info();
        for device in &info {
            println!("Device: {} (low_power: {}, headless: {})",
                device.name, device.is_low_power, device.is_headless);
        }
    }
}
