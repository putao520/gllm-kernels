use burn::tensor::backend::Backend;

/// CUDA backend (NVIDIA GPU)
#[cfg(all(feature = "cuda", feature = "fusion"))]
pub type DefaultBackend = burn_fusion::Fusion<burn_cuda::Cuda>;

/// CUDA backend (NVIDIA GPU)
#[cfg(all(feature = "cuda", not(feature = "fusion")))]
pub type DefaultBackend = burn_cuda::Cuda;

/// ROCm backend (AMD GPU)
#[cfg(all(
    feature = "rocm",
    feature = "fusion",
    not(feature = "cuda"),
    not(feature = "wgpu")
))]
pub type DefaultBackend = burn_fusion::Fusion<burn_rocm::Rocm>;

/// ROCm backend (AMD GPU)
#[cfg(all(
    feature = "rocm",
    not(feature = "fusion"),
    not(feature = "cuda"),
    not(feature = "wgpu")
))]
pub type DefaultBackend = burn_rocm::Rocm;

/// WGPU backend (WebGPU/Vulkan - cross-platform GPU)
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type DefaultBackend = burn_wgpu::Wgpu;

/// Metal backend (Apple Silicon GPU)
#[cfg(all(
    feature = "metal",
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "wgpu")
))]
pub type DefaultBackend = burn_mlx::Mlx;

/// CPU backend (fallback)
#[cfg(all(
    feature = "cpu",
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "metal"),
    not(feature = "wgpu")
))]
pub type DefaultBackend = burn_ndarray::NdArray<f32>;

/// Select the default device for a backend
pub fn select_device<B: Backend>() -> B::Device {
    B::Device::default()
}
