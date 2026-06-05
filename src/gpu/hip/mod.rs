//! HIP GPU backend — driver API FFI + device management.
//!
//! Gated behind `#[cfg(feature = "jit-hip")]`. All interaction with the AMD
//! driver happens through runtime `dlopen` of `libamdhip64.so` — zero build-time
//! dependency on the ROCm SDK.

pub mod driver;
pub mod device;

pub use driver::HipDriver;
pub use device::HipDevice;
