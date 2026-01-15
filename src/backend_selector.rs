//! Dynamic backend selector with fully automatic runtime detection.
//!
//! Fat Binary approach: all backends compiled, runtime selection based on hardware.
//! Zero configuration - no environment variables required.

use std::sync::OnceLock;

use crate::backend_trait::BackendSelector;
use crate::runtime_detection::{BackendType, detect_backend};

/// Global backend selector instance.
static SELECTOR: OnceLock<Box<dyn BackendSelector>> = OnceLock::new();

/// Initialize the backend selector.
///
/// This function automatically detects the best available backend
/// by checking hardware capabilities in priority order:
/// CUDA > ROCm > Metal > WGPU > CPU
///
/// The selected backend is cached for subsequent calls.
pub fn init_backend_selector() {
    SELECTOR.get_or_init(|| {
        // Automatic detection (zero configuration)
        let detected = detect_backend();
        log::info!("Auto-detected backend: {}", detected.name());

        match detected {
            BackendType::Cuda => create_cuda_backend(),
            BackendType::Metal => create_metal_backend(),
            BackendType::Rocm => create_rocm_backend(),
            BackendType::Wgpu => create_wgpu_backend(),
            BackendType::Cpu => create_cpu_backend(),
        }
    });
}

/// Get the current backend selector instance.
///
/// Returns `None` if `init_backend_selector()` has not been called.
pub fn get_backend_selector() -> Option<&'static dyn BackendSelector> {
    SELECTOR.get().map(|b| b.as_ref())
}

/// Force a specific backend (advanced usage).
///
/// This bypasses both environment variables and auto-detection.
pub fn force_backend<B: BackendSelector + 'static>(backend: B) {
    let _ = SELECTOR.set(Box::new(backend));
}

// Backend factory functions (unconditional - runtime detection handles availability)

fn create_cuda_backend() -> Box<dyn BackendSelector> {
    Box::new(CudaBackendSelector)
}

fn create_metal_backend() -> Box<dyn BackendSelector> {
    Box::new(MetalBackendSelector)
}

fn create_rocm_backend() -> Box<dyn BackendSelector> {
    Box::new(RocmBackendSelector)
}

fn create_wgpu_backend() -> Box<dyn BackendSelector> {
    Box::new(WgpuBackendSelector)
}

fn create_cpu_backend() -> Box<dyn BackendSelector> {
    Box::new(CpuBackendSelector)
}

// Backend selector implementations

struct CudaBackendSelector;

impl BackendSelector for CudaBackendSelector {
    fn name(&self) -> &'static str {
        "CUDA"
    }
}

struct MetalBackendSelector;

impl BackendSelector for MetalBackendSelector {
    fn name(&self) -> &'static str {
        "Metal"
    }
}

struct RocmBackendSelector;

impl BackendSelector for RocmBackendSelector {
    fn name(&self) -> &'static str {
        "ROCm"
    }
}

struct WgpuBackendSelector;

impl BackendSelector for WgpuBackendSelector {
    fn name(&self) -> &'static str {
        "WGPU"
    }
}

struct CpuBackendSelector;

impl BackendSelector for CpuBackendSelector {
    fn name(&self) -> &'static str {
        "CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automatic_detection() {
        init_backend_selector();
        let selector = get_backend_selector().unwrap();
        // Backend should be one of the valid types
        let name = selector.name();
        assert!(
            name == "CUDA" || name == "ROCm" || name == "Metal" || name == "WGPU" || name == "CPU",
            "Unexpected backend name: {}", name
        );
    }

    #[test]
    fn test_backend_names() {
        assert_eq!(CudaBackendSelector.name(), "CUDA");
        assert_eq!(MetalBackendSelector.name(), "Metal");
        assert_eq!(RocmBackendSelector.name(), "ROCm");
        assert_eq!(WgpuBackendSelector.name(), "WGPU");
        assert_eq!(CpuBackendSelector.name(), "CPU");
    }
}
