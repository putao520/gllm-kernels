//! SM-aware PTX loader for CUDA kernels.
//!
//! This module provides automatic PTX selection based on the GPU's compute capability.
//! It supports:
//! - Multiple precompiled PTX versions for different SM architectures
//! - Automatic SM detection and best-match selection
//! - Fat Binary Only: NO runtime compilation fallback
//!
//! # Design Philosophy
//!
//! 🚨 **Fat Binary Only Architecture**:
//! - All PTX must be precompiled and embedded at compile time
//! - NO NVRTC runtime compilation
//! - If no matching PTX is found, return error (not fallback to compilation)
//!
//! # Supported Architectures (2019-2026)
//!
//! | SM | Architecture | GPUs |
//! |----|--------------|------|
//! | 61 | Pascal | GTX 1060/1070/1080 |
//! | 75 | Turing | RTX 2060/2070/2080, GTX 1650/1660 |
//! | 80 | Ampere | A100, A30 |
//! | 86 | Ampere | RTX 3060/3070/3080/3090, A10, A40 |
//! | 89 | Ada Lovelace | RTX 4060/4070/4080/4090, L4, L40 |
//! | 90 | Hopper | H100, H200 |
//! | 100 | Blackwell | B100, B200 (CUDA 12.8+) |
//! | 120 | Blackwell Consumer | RTX 5070/5080/5090 (CUDA 12.8+) |

use std::sync::Arc;
use cudarc::driver::{CudaContext, DriverError, sys};
use cudarc::nvrtc::Ptx;

/// Supported SM architectures for PTX selection.
/// Listed in descending order for best-match selection.
pub const SUPPORTED_SM_VERSIONS: &[u32] = &[
    120, // Blackwell Consumer (RTX 50 series)
    100, // Blackwell Data Center (B100, B200)
    90,  // Hopper (H100, H200)
    89,  // Ada Lovelace (RTX 40 series)
    86,  // Ampere Consumer (RTX 30 series)
    80,  // Ampere Data Center (A100)
    75,  // Turing (RTX 20 series)
    61,  // Pascal (GTX 10 series)
];

/// Minimum SM version we support.
pub const MIN_SM_VERSION: u32 = 61;

/// Error type for PTX loading.
#[derive(Debug)]
pub enum PtxLoadError {
    /// GPU SM version is too old.
    UnsupportedSm(u32),
    /// Failed to detect SM version.
    SmDetectionFailed(String),
    /// No matching precompiled PTX found (Fat Binary Only - no runtime compilation).
    NoPtxAvailable(String),
    /// CUDA driver error.
    Driver(DriverError),
}

impl std::fmt::Display for PtxLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSm(sm) => write!(
                f,
                "GPU compute capability sm_{} is not supported. Minimum required: sm_{}",
                sm, MIN_SM_VERSION
            ),
            Self::SmDetectionFailed(msg) => write!(f, "Failed to detect SM version: {}", msg),
            Self::NoPtxAvailable(msg) => write!(f, "No PTX available: {}", msg),
            Self::Driver(e) => write!(f, "CUDA driver error: {}", e),
        }
    }
}

impl std::error::Error for PtxLoadError {}

impl From<DriverError> for PtxLoadError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Detect the compute capability (SM version) of the given CUDA context's device.
pub fn detect_sm_version(ctx: &Arc<CudaContext>) -> Result<u32, PtxLoadError> {
    // Query compute capability using device attributes
    let major = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| PtxLoadError::SmDetectionFailed(format!("Failed to query compute capability major: {}", e)))?;
    let minor = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| PtxLoadError::SmDetectionFailed(format!("Failed to query compute capability minor: {}", e)))?;

    let sm_version = (major * 10 + minor) as u32;
    log::debug!("Detected GPU compute capability: sm_{} (major={}, minor={})", sm_version, major, minor);

    Ok(sm_version)
}

/// Find the best matching SM version for the given GPU.
///
/// Returns the highest supported SM version that is <= the GPU's SM version.
/// PTX compiled for a lower SM version is forward-compatible with higher SM GPUs.
pub fn find_best_sm_match(gpu_sm: u32, available_sms: &[u32]) -> Option<u32> {
    // Find the highest available SM that is <= gpu_sm
    available_sms
        .iter()
        .filter(|&&sm| sm <= gpu_sm)
        .max()
        .copied()
}

/// PTX collection for a kernel.
/// Contains precompiled PTX for multiple SM versions.
///
/// 🚨 **Fat Binary Only**: No runtime compilation support.
/// All PTX must be precompiled and embedded at compile time.
pub struct PtxCollection {
    /// Kernel name for logging.
    pub kernel_name: &'static str,
    /// Available PTX versions: (sm_version, ptx_content).
    /// PTX compiled for a lower SM version is forward-compatible with higher SM GPUs.
    pub ptx_versions: &'static [(u32, &'static str)],
}

impl PtxCollection {
    /// Load the best matching PTX for the given CUDA context.
    ///
    /// 🚨 **Fat Binary Only**: This method only loads precompiled PTX.
    /// If no matching PTX is found, it returns an error (NO runtime compilation).
    pub fn load(&self, ctx: &Arc<CudaContext>) -> Result<Ptx, PtxLoadError> {
        // Step 1: Detect GPU SM version
        let gpu_sm = detect_sm_version(ctx)?;

        if gpu_sm < MIN_SM_VERSION {
            return Err(PtxLoadError::UnsupportedSm(gpu_sm));
        }

        // Step 2: Find available SM versions from our PTX collection
        let available_sms: Vec<u32> = self.ptx_versions.iter().map(|(sm, _)| *sm).collect();

        // Step 3: Find best matching PTX
        if let Some(best_sm) = find_best_sm_match(gpu_sm, &available_sms) {
            // Find the PTX content for the best SM
            if let Some((_, ptx_content)) = self.ptx_versions.iter().find(|(sm, _)| *sm == best_sm) {
                // Verify PTX is not a placeholder
                if !ptx_content.contains("Placeholder") && !ptx_content.is_empty() {
                    log::info!(
                        "Loading {} PTX for sm_{} (GPU is sm_{})",
                        self.kernel_name, best_sm, gpu_sm
                    );
                    return Ok(Ptx::from_src(*ptx_content));
                }
            }
        }

        // 🚨 Fat Binary Only: NO runtime compilation fallback
        // Return error if no matching precompiled PTX is found
        Err(PtxLoadError::NoPtxAvailable(format!(
            "No precompiled PTX for {} kernel (GPU sm_{}). Available SM versions: {:?}. \
             Fat Binary Only architecture: runtime compilation is disabled.",
            self.kernel_name, gpu_sm, available_sms
        )))
    }
}

/// Macro to define a PTX collection with embedded PTX files.
///
/// 🚨 **Fat Binary Only**: All PTX must be precompiled and embedded.
/// NO runtime compilation support - source code parameter removed.
///
/// Usage:
/// ```ignore
/// define_ptx_collection!(
///     TILED_ATTENTION_PTX,
///     "tiled_attention",
///     [
///         (61, include_str!("kernels/tiled_attention_sm61.ptx")),
///         (75, include_str!("kernels/tiled_attention_sm75.ptx")),
///         (80, include_str!("kernels/tiled_attention_sm80.ptx")),
///     ]
/// );
/// ```
#[macro_export]
macro_rules! define_ptx_collection {
    ($name:ident, $kernel_name:literal, [ $(($sm:expr, $ptx:expr)),* $(,)? ]) => {
        pub static $name: $crate::cuda_kernels::ptx_loader::PtxCollection =
            $crate::cuda_kernels::ptx_loader::PtxCollection {
                kernel_name: $kernel_name,
                ptx_versions: &[ $(($sm, $ptx)),* ],
            };
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_best_sm_match() {
        let available = vec![61, 75, 80, 86, 89, 90];

        // Exact match
        assert_eq!(find_best_sm_match(80, &available), Some(80));

        // GPU is newer than any available PTX
        assert_eq!(find_best_sm_match(100, &available), Some(90));

        // GPU is between available versions
        assert_eq!(find_best_sm_match(87, &available), Some(86));

        // GPU is older than minimum
        assert_eq!(find_best_sm_match(50, &available), None);

        // GPU matches minimum
        assert_eq!(find_best_sm_match(61, &available), Some(61));
    }

    #[test]
    fn test_sm_version_ordering() {
        // Verify SUPPORTED_SM_VERSIONS is in descending order
        let mut prev = u32::MAX;
        for &sm in SUPPORTED_SM_VERSIONS {
            assert!(sm < prev, "SUPPORTED_SM_VERSIONS must be in descending order");
            prev = sm;
        }
    }
}
