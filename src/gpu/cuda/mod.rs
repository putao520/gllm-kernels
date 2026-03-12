//! CUDA GPU backend — driver API FFI + device management.
//!
//! Gated behind `#[cfg(feature = "jit-cuda")]`. All interaction with the NVIDIA
//! driver happens through runtime `dlopen` of `libcuda.so.1` — zero build-time
//! dependency on the CUDA SDK.

pub mod driver;
pub mod device;

pub use driver::CudaDriver;
pub use device::CudaDevice;
