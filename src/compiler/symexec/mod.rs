//! Phase 0: Binary symbolic execution engine.
//!
//! Analyzes `extern "C"` scalar functions by disassembling their compiled
//! binary and symbolically executing each instruction to extract the
//! computational structure (OpTrace).

pub mod engine;
pub mod sym_value;
#[cfg(feature = "jit-x86")]
pub mod decoder;
#[cfg(feature = "jit-x86")]
pub mod cfg;
#[cfg(feature = "jit-x86")]
pub mod loop_analyzer;
#[cfg(feature = "jit-x86")]
pub mod branch_merger;

pub use engine::{SymbolicExecutor, SymExecError};
pub use sym_value::{SymValue, LibmFn, SelectKind};
