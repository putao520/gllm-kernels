//! Dynamic backend selector with runtime detection and environment variable override.

use std::sync::OnceLock;

use crate::backend_trait::BackendSelector;
use crate::runtime_detection::{BackendType, detect_backend};

/// Global backend selector instance.
static SELECTOR: OnceLock<Box<dyn BackendSelector>> = OnceLock::new();

/// Initialize the backend selector.
///
/// This function:
/// 1. Checks the `GLLM_BACKEND` environment variable for manual override
/// 2. Falls back to automatic detection if no override is specified
/// 3. Caches the selected backend for subsequent calls
pub fn init_backend_selector() {
    SELECTOR.get_or_init(|| {
        // Check environment variable override
        if let Ok(backend_str) = std::env::var("GLLM_BACKEND") {
            match backend_str.to_lowercase().as_str() {
                "cuda" => {
                    #[cfg(feature = "cuda")]
                    {
                        log::info!("Backend overridden to CUDA via GLLM_BACKEND");
                        return create_cuda_backend();
                    }
                    #[cfg(not(feature = "cuda"))]
                    log::warn!("GLLM_BACKEND=cuda requested but CUDA feature not enabled, falling back to detection");
                }
                "rocm" => {
                    #[cfg(feature = "rocm")]
                    {
                        log::info!("Backend overridden to ROCm via GLLM_BACKEND");
                        return create_rocm_backend();
                    }
                    #[cfg(not(feature = "rocm"))]
                    log::warn!("GLLM_BACKEND=rocm requested but ROCm feature not enabled, falling back to detection");
                }
                "metal" => {
                    #[cfg(feature = "metal")]
                    {
                        log::info!("Backend overridden to Metal via GLLM_BACKEND");
                        return create_metal_backend();
                    }
                    #[cfg(not(feature = "metal"))]
                    log::warn!("GLLM_BACKEND=metal requested but Metal feature not enabled, falling back to detection");
                }
                "wgpu" => {
                    #[cfg(feature = "wgpu")]
                    {
                        log::info!("Backend overridden to WGPU via GLLM_BACKEND");
                        return create_wgpu_backend();
                    }
                    #[cfg(not(feature = "wgpu"))]
                    log::warn!("GLLM_BACKEND=wgpu requested but WGPU feature not enabled, falling back to detection");
                }
                "cpu" => {
                    log::info!("Backend overridden to CPU via GLLM_BACKEND");
                    return create_cpu_backend();
                }
                _ => {
                    log::warn!("Invalid GLLM_BACKEND value: {}, falling back to detection", backend_str);
                }
            }
        }

        // Automatic detection
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

// Backend factory functions

#[cfg(feature = "cuda")]
fn create_cuda_backend() -> Box<dyn BackendSelector> {
    Box::new(CudaBackendSelector)
}

#[cfg(not(feature = "cuda"))]
fn create_cuda_backend() -> Box<dyn BackendSelector> {
    log::error!("CUDA backend requested but not compiled in, falling back to CPU");
    create_cpu_backend()
}

#[cfg(feature = "metal")]
fn create_metal_backend() -> Box<dyn BackendSelector> {
    Box::new(MetalBackendSelector)
}

#[cfg(not(feature = "metal"))]
fn create_metal_backend() -> Box<dyn BackendSelector> {
    log::error!("Metal backend requested but not compiled in, falling back to CPU");
    create_cpu_backend()
}

#[cfg(feature = "rocm")]
fn create_rocm_backend() -> Box<dyn BackendSelector> {
    Box::new(RocmBackendSelector)
}

#[cfg(not(feature = "rocm"))]
fn create_rocm_backend() -> Box<dyn BackendSelector> {
    log::error!("ROCm backend requested but not compiled in, falling back to CPU");
    create_cpu_backend()
}

#[cfg(feature = "wgpu")]
fn create_wgpu_backend() -> Box<dyn BackendSelector> {
    Box::new(WgpuBackendSelector)
}

#[cfg(not(feature = "wgpu"))]
fn create_wgpu_backend() -> Box<dyn BackendSelector> {
    log::error!("WGPU backend requested but not compiled in, falling back to CPU");
    create_cpu_backend()
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
    fn test_environment_override() {
        std::env::set_var("GLLM_BACKEND", "cpu");
        init_backend_selector();

        let selector = get_backend_selector().unwrap();
        assert_eq!(selector.name(), "CPU");

        // Clean up
        std::env::remove_var("GLLM_BACKEND");
    }

    #[test]
    fn test_force_backend() {
        force_backend(CpuBackendSelector);

        let selector = get_backend_selector().unwrap();
        assert_eq!(selector.name(), "CPU");
    }
}
