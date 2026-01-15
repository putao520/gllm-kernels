//! Metal metallib loader with automatic runtime compilation fallback.
//!
//! This module provides `MetallibCollection` for loading precompiled Metal libraries
//! with automatic fallback to runtime compilation from source.
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
//! The kernel loader automatically selects the best loading strategy:
//! 1. Precompiled metallib (fastest, production use)
//! 2. Runtime compilation from source (fallback)

use std::fmt;

use metal::{Device, Library};

/// Error type for metallib loading operations.
#[derive(Debug)]
pub enum MetallibLoadError {
    /// No Metal device available.
    DeviceNotAvailable,
    /// Failed to load metallib from data.
    LoadFailed(String),
    /// Failed to compile from source.
    CompileFailed(String),
    /// No kernel available (both embedded and compile failed).
    NoKernelAvailable(String),
}

impl fmt::Display for MetallibLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotAvailable => write!(f, "No Metal device available"),
            Self::LoadFailed(msg) => write!(f, "Failed to load metallib: {}", msg),
            Self::CompileFailed(msg) => write!(f, "Failed to compile Metal source: {}", msg),
            Self::NoKernelAvailable(name) => {
                write!(f, "No kernel available for '{}': embedded load and runtime compile both failed", name)
            }
        }
    }
}

impl std::error::Error for MetallibLoadError {}

/// Collection of Metal shader data for automatic loading.
///
/// This struct holds both precompiled metallib binary and source code,
/// allowing automatic fallback from embedded metallib to runtime compilation.
///
/// # Example
///
/// ```ignore
/// static FLASH_ATTENTION_METALLIB: MetallibCollection = MetallibCollection {
///     kernel_name: "flash_attention",
///     metallib_data: include_bytes!("kernels/flash_attention.metallib"),
///     source: include_str!("kernels/flash_attention.metal"),
/// };
///
/// let library = FLASH_ATTENTION_METALLIB.load(&device)?;
/// ```
pub struct MetallibCollection {
    /// Kernel name for logging purposes.
    pub kernel_name: &'static str,
    /// Precompiled metallib binary data (from `include_bytes!`).
    pub metallib_data: &'static [u8],
    /// Metal shader source code for runtime compilation fallback.
    pub source: &'static str,
}

impl MetallibCollection {
    /// Load Metal library using zero-config automatic fallback.
    ///
    /// # Loading Priority
    ///
    /// 1. **Embedded metallib** (fastest) - Load precompiled binary
    /// 2. **Runtime compilation** (fallback) - Compile from source
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal device to load the library on
    ///
    /// # Returns
    ///
    /// A Metal Library containing the compiled kernels.
    pub fn load(&self, device: &Device) -> Result<Library, MetallibLoadError> {
        // Priority 1: Try embedded precompiled metallib
        if !self.metallib_data.is_empty() {
            log::debug!("[{}] Trying to load embedded metallib ({} bytes)",
                self.kernel_name, self.metallib_data.len());

            match device.new_library_with_data(self.metallib_data) {
                Ok(lib) => {
                    log::info!("[{}] Loaded embedded metallib successfully", self.kernel_name);
                    return Ok(lib);
                }
                Err(e) => {
                    log::warn!("[{}] Failed to load embedded metallib: {}, trying runtime compile",
                        self.kernel_name, e);
                }
            }
        } else {
            log::debug!("[{}] No embedded metallib, trying runtime compile", self.kernel_name);
        }

        // Priority 2: Runtime compilation from source
        if !self.source.is_empty() {
            log::debug!("[{}] Compiling Metal shader from source at runtime", self.kernel_name);

            let options = metal::CompileOptions::new();
            match device.new_library_with_source(self.source, &options) {
                Ok(lib) => {
                    log::info!("[{}] Compiled from source successfully", self.kernel_name);
                    return Ok(lib);
                }
                Err(e) => {
                    log::error!("[{}] Runtime compilation failed: {}", self.kernel_name, e);
                    return Err(MetallibLoadError::CompileFailed(e));
                }
            }
        }

        Err(MetallibLoadError::NoKernelAvailable(self.kernel_name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metallib_collection_empty() {
        let collection = MetallibCollection {
            kernel_name: "test",
            metallib_data: &[],
            source: "",
        };

        // Should fail with NoKernelAvailable when both are empty
        if let Some(device) = Device::system_default() {
            let result = collection.load(&device);
            assert!(result.is_err());
        }
    }
}
