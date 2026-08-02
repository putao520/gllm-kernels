//! Metal GPU backend (macOS / Apple Silicon).
//!
//! Provides `MetalDevice` implementing `GpuDevice` via Objective-C runtime FFI
//! to Metal.framework. Zero external dependencies — uses `dlopen` + `objc_msgSend`.
//!
//! Gated behind `#[cfg(feature = "jit-metal")]`.

pub mod device;
pub mod objc_runtime;

pub use device::MetalDevice;
