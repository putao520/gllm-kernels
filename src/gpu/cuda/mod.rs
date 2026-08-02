//! CUDA GPU backend — driver API FFI + device management.
//!
//! Gated behind `#[cfg(feature = "jit-cuda")]`. All interaction with the NVIDIA
//! driver happens through runtime `dlopen` of `libcuda.so.1` — zero build-time
//! dependency on the CUDA SDK.

pub mod device;
pub mod driver;

pub use device::{CudaBuffer, CudaDevice, CudaModule, CudaStream};
pub use driver::{CUevent, CudaDriver};
