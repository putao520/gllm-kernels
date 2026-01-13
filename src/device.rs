//! Device helpers using Burn backends.

use burn::tensor::backend::Backend;

use crate::backend::select_device;

/// Select the default device for a specific backend.
pub fn device_for<B: Backend>() -> B::Device {
    select_device::<B>()
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "rocm", feature = "metal", feature = "wgpu"))]
mod default_device_impl {
    use super::*;

    use crate::backend::DefaultBackend;

    /// Default device for the configured backend.
    pub type DefaultDevice = <DefaultBackend as Backend>::Device;

    /// Select the default device for the configured backend.
    pub fn default_device() -> DefaultDevice {
        select_device::<DefaultBackend>()
    }
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "rocm", feature = "metal", feature = "wgpu"))]
pub use default_device_impl::{DefaultDevice, default_device};
