//! Scalar operator implementations — `extern "C"` pure scalar functions.
//!
//! These serve as:
//! 1. Golden reference for correctness testing.
//! 2. Phase 0 binary analysis targets (symbolic execution extracts OpTrace).
//!
//! Every function here is `#[no_mangle] pub extern "C"` so the JIT compiler
//! can locate them by symbol name in the binary.
//!
//! This crate is compiled with `opt-level = 1` (configured in the workspace
//! root Cargo.toml) to preserve loop structure for symexec analysis while
//! eliminating trivial redundancy. This prevents the compiler from vectorizing
//! or unrolling loops, which would make binary analysis intractable.

pub mod activations;
pub mod argmax;
pub mod attention;
pub mod blas;
pub mod control_flow;
pub mod depthwise_conv1d;
pub mod gather;
pub mod learned_pos_2d;
pub mod moe;
pub mod moe_dispatch_packed;
pub mod norms;
pub mod p4_p5;
pub mod patch_embed;
pub mod ple;
pub mod pooling;
pub mod rope;
