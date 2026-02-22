//! Code generation — JIT compiler backend.
//!
//! This module hosts the JIT compilation system. The architecture per SPEC §8:
//!
//!   CompilerGraph → Phase 1 (DAG construction)
//!                 → Phase 2 (fusion decisions)
//!                 → Phase 3 (MachineCodeEmitter via iced-x86 / dynasm-rs)
//!                 → CompiledLayer
//!
//! Phase 3 should programmatically generate new machine code for each fused
//! kernel — complete GEMM microkernels with epilogue injection, fused
//! elementwise loops, and tile-level fusion. NOT trampoline calls.
//!
//! TODO: Implement `PlatformBackend` and `MachineCodeEmitter` traits per SPEC §8.6.

pub mod x86_64;
pub mod aarch64;
pub mod emitter;

/// Output of code generation: raw machine code bytes.
pub struct CodegenOutput {
    /// Raw machine code
    pub code: Vec<u8>,
    /// Required scratchpad size in bytes
    pub scratchpad_bytes: usize,
}
