//! Backend type definitions for burn tensor operations.
//!
//! This module provides the DefaultBackend type alias. In the runtime-detection
//! architecture, we always use NdArray as the default backend for type definitions,
//! while actual kernel execution happens through KernelDispatcher which routes
//! to the appropriate GPU backend at runtime.

use burn::tensor::backend::Backend;

/// Default backend for burn tensor operations.
///
/// In the runtime-detection architecture, we use NdArray (CPU) as the default
/// backend type. GPU acceleration is achieved through KernelDispatcher, which
/// detects available backends at runtime and dispatches to:
/// - CUDA (NVIDIA GPUs via cudarc dynamic loading)
/// - ROCm (AMD GPUs on Linux)
/// - Metal (Apple Silicon on macOS)
/// - WGPU (Cross-platform WebGPU/Vulkan)
///
/// # Example
///
/// ```ignore
/// use gllm_kernels::{DefaultBackend, KernelDispatcher};
/// use burn::tensor::Tensor;
///
/// // Create tensors using the default backend
/// let tensor: Tensor<DefaultBackend, 2> = Tensor::zeros([4, 4]);
///
/// // For optimized operations, use KernelDispatcher
/// let dispatcher = KernelDispatcher::new();
/// // dispatcher routes to GPU kernels if available
/// ```
pub type DefaultBackend = burn_ndarray::NdArray<f32>;

/// Select the default device for a backend.
pub fn select_device<B: Backend>() -> B::Device {
    B::Device::default()
}
