//! Binary symbolic execution engine (Scalar + SymExec phase).
//!
//! Analyzes `extern "C"` scalar functions by disassembling their compiled
//! binary and symbolically executing each instruction to extract the
//! computational structure (OpTrace).

pub mod engine;
pub mod sym_value;

#[cfg(any(feature = "jit-x86", feature = "jit-aarch64"))]
pub mod decoder;
#[cfg(any(feature = "jit-x86", feature = "jit-aarch64"))]
pub mod cfg;
#[cfg(any(feature = "jit-x86", feature = "jit-aarch64"))]
pub mod loop_analyzer;
#[cfg(any(feature = "jit-x86", feature = "jit-aarch64"))]
pub mod branch_merger;

#[cfg(feature = "jit-aarch64")]
pub mod decoder_aarch64;

pub use engine::{SymbolicExecutor, SymExecError};
pub use sym_value::{SymValue, LibmFn, SelectKind};
