//! Device helpers using Burn backends.

use burn::tensor::backend::Backend;

use crate::backend::{select_device, DefaultBackend};

/// Default device for the configured backend.
pub type DefaultDevice = <DefaultBackend as Backend>::Device;

/// Select the default device for the configured backend.
pub fn default_device() -> DefaultDevice {
    select_device::<DefaultBackend>()
}

/// Select the default device for a specific backend.
pub fn device_for<B: Backend>() -> B::Device {
    select_device::<B>()
}
