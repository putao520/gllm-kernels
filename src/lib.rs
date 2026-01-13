//! gllm-kernels: low-level attention kernels built on Burn.

pub mod backend;
pub mod comm;
#[cfg(feature = "cuda-kernel")]
pub mod cuda_kernels;
#[cfg(feature = "rocm-kernel")]
pub mod hip_kernels;
#[cfg(feature = "metal-kernel")]
pub mod metal_kernels;
pub mod device;
pub mod ops;
pub mod types;

// Runtime backend detection
pub mod runtime_detection;
pub mod backend_trait;
pub mod backend_selector;
pub mod kernel_cache;

pub use backend::{select_device, DefaultBackend};
pub use runtime_detection::{
    BackendType, BackendDetectionResult, detect_backend, redetect_backend,
};
pub use kernel_cache::{
    kernel_cache_dir, kernel_cache_path, load_cached_kernel, save_kernel_to_cache,
    clear_kernel_cache, kernel_cache_size,
};
