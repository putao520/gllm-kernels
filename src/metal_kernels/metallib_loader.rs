//! Metal metallib loader - Fat Binary approach.
//!
//! This module provides `MetallibCollection` for loading precompiled Metal libraries.
//! metallib is Metal's intermediate format (like PTX for CUDA, HSACO for ROCm).
//!
//! ## Architecture
//!
//! ```text
//! Compile time: Metal shader source → metallib binary (via xcrun metallib) → embedded
//! Runtime: Metal Framework loads embedded metallib → GPU executes
//! ```
//!
//! ## Design Principle
//!
//! metallib is already a platform-independent intermediate representation.
//! NO runtime compilation fallback - if metallib fails to load, it's an error.
//! This matches CUDA/ROCm behavior where PTX/HSACO are directly loaded.

use std::fmt;

use metal::{Device, Library};

/// Error type for metallib loading operations.
#[derive(Debug)]
pub enum MetallibLoadError {
    /// No Metal device available.
    DeviceNotAvailable,
    /// Failed to load metallib from embedded data.
    LoadFailed(String),
    /// No metallib data embedded.
    NoMetallibEmbedded(String),
}

impl fmt::Display for MetallibLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotAvailable => write!(f, "No Metal device available"),
            Self::LoadFailed(msg) => write!(f, "Failed to load embedded metallib: {}", msg),
            Self::NoMetallibEmbedded(name) => {
                write!(f, "No metallib embedded for kernel '{}'. Build with: xcrun metallib", name)
            }
        }
    }
}

impl std::error::Error for MetallibLoadError {}

/// Collection of precompiled Metal library data.
///
/// This struct holds precompiled metallib binary (Metal's intermediate format).
/// NO source code or runtime compilation - metallib must be precompiled.
///
/// # Example
///
/// ```ignore
/// static FLASH_ATTENTION_METALLIB: MetallibCollection = MetallibCollection {
///     kernel_name: "flash_attention",
///     metallib_data: include_bytes!("kernels/flash_attention.metallib"),
/// };
///
/// let library = FLASH_ATTENTION_METALLIB.load(&device)?;
/// ```
pub struct MetallibCollection {
    /// Kernel name for logging and error messages.
    pub kernel_name: &'static str,
    /// Precompiled metallib binary data (from `include_bytes!`).
    pub metallib_data: &'static [u8],
}

impl MetallibCollection {
    /// Load Metal library from embedded metallib data.
    ///
    /// metallib is Metal's intermediate format (like PTX/HSACO).
    /// NO runtime compilation fallback - if metallib fails, returns error.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal device to load the library on
    ///
    /// # Returns
    ///
    /// A Metal Library containing the compiled kernels.
    ///
    /// # Errors
    ///
    /// - `NoMetallibEmbedded` - No metallib data was embedded
    /// - `LoadFailed` - Metal framework failed to load the metallib
    pub fn load(&self, device: &Device) -> Result<Library, MetallibLoadError> {
        if self.metallib_data.is_empty() {
            log::error!("[{}] No metallib embedded. Build with: xcrun metallib", self.kernel_name);
            return Err(MetallibLoadError::NoMetallibEmbedded(self.kernel_name.to_string()));
        }

        log::debug!("[{}] Loading embedded metallib ({} bytes)",
            self.kernel_name, self.metallib_data.len());

        match device.new_library_with_data(self.metallib_data) {
            Ok(lib) => {
                log::info!("[{}] Loaded metallib successfully", self.kernel_name);
                Ok(lib)
            }
            Err(e) => {
                log::error!("[{}] Failed to load metallib: {}", self.kernel_name, e);
                Err(MetallibLoadError::LoadFailed(e))
            }
        }
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
        };

        // Should fail with NoMetallibEmbedded when metallib is empty
        if let Some(device) = Device::system_default() {
            let result = collection.load(&device);
            assert!(matches!(result, Err(MetallibLoadError::NoMetallibEmbedded(_))));
        }
    }
}
