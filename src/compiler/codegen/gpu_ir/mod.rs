//! GPU IR — unified abstraction layer for GPU code generation backends.
//!
//! This module provides the `GpuDialect` trait and concrete implementations
//! for PTX (NVIDIA), HIP (AMD), and MSL (Apple Metal). The generic
//! `emit_trace_body<D>` function replaces the three duplicated trace body
//! emitters that previously lived in `ptx.rs`, `hip.rs`, and `air.rs`.
//!
//! See `SPEC/PLAN-gpu-codegen-unify.md` for the full design.

pub mod primitives;
pub mod trace_emitter;
pub mod kernel_builder;
pub mod plan_emitter;

pub use primitives::{GpuCapabilities, KernelParam, ParamQualifier, ParamType};
pub use trace_emitter::{emit_trace_body, GpuDialect};
#[cfg(feature = "jit-cuda")]
pub use trace_emitter::PtxDialect;
#[cfg(feature = "jit-hip")]
pub use trace_emitter::HipDialect;
#[cfg(feature = "jit-metal")]
pub use trace_emitter::MslDialect;
pub use kernel_builder::{
    build_elementwise_kernel, build_binary_elementwise_kernel,
    build_softmax_kernel, build_normlike_kernel,
    build_meanpool_kernel, build_dequantize_kernel,
};
pub use plan_emitter::gpu_emit_plan;
