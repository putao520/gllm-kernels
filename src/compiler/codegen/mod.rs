//! Code generation — Phase 3 of the JIT compiler pipeline.
//!
//! This module hosts the JIT compilation backends. The full pipeline per SPEC §8:
//!
//!   Phase 0 (ScalarOpRegistry + OpTrace via SymbolicExecutor)
//!   → Phase 1 (SemanticDAG: OpClass auto-derivation)
//!   → Phase 2 (Fusion + HW constraints + Parallel strategy + Buffer alloc)
//!   → Phase 3 (this module: native code generation via iced-x86 / dynasm-rs)
//!   → CompiledLayer
//!
//! Phase 3 programmatically generates new machine code for each fused kernel —
//! complete GEMM microkernels with epilogue injection, fused elementwise loops,
//! and tile-level fusion. NOT trampoline calls.
//!
//! Current status: x86_64 MVP implemented under `jit-x86` feature flag
//! (`x86_64::jit::X86CodeGen`). aarch64 backend uses dynasm-rs (BLIS 5-level loop nesting, NEON tanh/log/exp real implementations).

pub mod x86_64;
pub mod aarch64;
pub mod emitter;

pub use emitter::{MachineCodeEmitter, PlatformBackend, Platform};

/// Output of code generation: raw machine code bytes.
pub struct CodegenOutput {
    /// Raw machine code
    pub code: Vec<u8>,
    /// Required scratchpad size in bytes
    pub scratchpad_bytes: usize,
}
