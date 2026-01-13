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
#[cfg(feature = "kernel-downloader")]
pub mod kernel_downloader;

#[cfg(any(feature = "cpu", feature = "cuda", feature = "rocm", feature = "metal", feature = "wgpu"))]
pub use backend::DefaultBackend;
pub use backend::select_device;
pub use runtime_detection::{
    BackendType, BackendDetectionResult, DeviceInfo, detect_backend, redetect_backend, ensure_kernels,
};
pub use kernel_cache::{
    kernel_cache_dir, kernel_cache_path, load_cached_kernel, save_kernel_to_cache,
    clear_kernel_cache, kernel_cache_size,
};
#[cfg(feature = "kernel-downloader")]
pub use kernel_downloader::{KernelDownloader, KernelDownloadError};
