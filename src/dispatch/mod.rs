//! Hardware profile and dispatch table for the inference compiler.
//!
//! Provides a unified `DeviceProfile` that integrates microarchitecture
//! detection, cache hierarchy, NUMA topology, and peak performance estimates.

pub mod device_profile;

pub use device_profile::{DeviceProfile, GemmBlocking, IsaLevel};

use std::sync::OnceLock;

static PROFILE: OnceLock<DeviceProfile> = OnceLock::new();

/// Get the global device profile (detected once, cached for process lifetime).
pub fn device_profile() -> &'static DeviceProfile {
    PROFILE.get_or_init(DeviceProfile::detect)
}
