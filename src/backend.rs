use burn::tensor::backend::Backend;

/// CUDA backend (NVIDIA GPU)
#[cfg(all(feature = "cuda", feature = "fusion"))]
pub type DefaultBackend = burn_fusion::Fusion<burn_cuda::Cuda>;

/// CUDA backend (NVIDIA GPU)
#[cfg(all(feature = "cuda", not(feature = "fusion")))]
pub type DefaultBackend = burn_cuda::Cuda;

/// WGPU backend (WebGPU/Vulkan - cross-platform GPU)
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type DefaultBackend = burn_wgpu::Wgpu;

/// CPU backend (fallback)
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
pub type DefaultBackend = burn_ndarray::NdArray<f32>;

/// Select the default device for a backend
pub fn select_device<B: Backend>() -> B::Device {
    B::Device::default()
}
