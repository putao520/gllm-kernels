//! Scalar operator implementations — `extern "C"` pure scalar functions.
//!
//! These serve as:
//! 1. Golden reference for correctness testing.
//! 2. Phase 0 binary analysis targets (future: symbolic execution extracts OpTrace).
//!
//! Every function here is `#[no_mangle] pub extern "C"` so the JIT compiler
//! can locate them by symbol name in the binary.

pub mod activations;
pub mod norms;
pub mod blas;
pub mod rope;
