//! Runtime backend detection and selection.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cuda,
    Rocm,
    Metal,
    Wgpu,
    Cpu,
}

impl BackendType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::Metal => "Metal",
            Self::Wgpu => "WGPU",
            Self::Cpu => "CPU",
        }
    }
}

static DETECTED_BACKEND: OnceLock<BackendType> = OnceLock::new();

pub fn detect_backend() -> BackendType {
    *DETECTED_BACKEND.get_or_init(|| {
        if try_cuda() {
            return BackendType::Cuda;
        }
        if try_metal() {
            return BackendType::Metal;
        }
        if try_rocm() {
            return BackendType::Rocm;
        }
        if try_wgpu() {
            return BackendType::Wgpu;
        }
        BackendType::Cpu
    })
}

fn try_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::result;
        result::init().is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    false
}

fn try_metal() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::Device;
        Device::system_default().is_some()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    false
}

fn try_rocm() -> bool {
    #[cfg(all(feature = "rocm", target_os = "linux"))]
    {
        std::path::Path::new("/dev/kfd").exists()
    }
    #[cfg(not(all(feature = "rocm", target_os = "linux")))]
    false
}

fn try_wgpu() -> bool {
    #[cfg(feature = "wgpu")]
    {
        true
    }
    #[cfg(not(feature = "wgpu"))]
    false
}
