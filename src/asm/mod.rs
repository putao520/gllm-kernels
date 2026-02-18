//! Hand-written assembly microkernels for performance-critical hot paths.
//!
//! These replace the macro-generated baseline implementations when
//! benchmarks prove they outperform the compiler-generated code.

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
