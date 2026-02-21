//! Code generation trait and dispatch.
//!
//! The `LayerCodegen` trait defines the interface for architecture-specific
//! code generators. Each target (x86_64, aarch64) implements this trait
//! to emit machine code for a compiled transformer layer.

pub mod x86_64;
pub mod aarch64;

use crate::compiler::ir::LayerIR;
use crate::compiler::planner::ExecutionPlan;
use crate::dispatch::device_profile::IsaLevel;
use crate::inference::types::InferenceError;

/// Output of code generation: raw machine code bytes.
pub struct CodegenOutput {
    /// Raw machine code
    pub code: Vec<u8>,
    /// Required scratchpad size in bytes
    pub scratchpad_bytes: usize,
}

/// Trait for architecture-specific code generators.
pub trait LayerCodegen {
    /// Generate machine code for a transformer layer.
    fn generate(
        &self,
        ir: &LayerIR,
        plan: &ExecutionPlan,
    ) -> Result<CodegenOutput, InferenceError>;

    /// Target ISA level.
    fn isa_level(&self) -> IsaLevel;
}

/// Select the appropriate codegen backend for the current platform.
pub fn select_codegen(isa: IsaLevel) -> Result<Box<dyn LayerCodegen>, InferenceError> {
    match isa {
        #[cfg(target_arch = "x86_64")]
        IsaLevel::Avx2 | IsaLevel::Avx512 => {
            Ok(Box::new(x86_64::X86_64Codegen::new(isa)))
        }
        #[cfg(target_arch = "aarch64")]
        IsaLevel::Neon => {
            Ok(Box::new(aarch64::Aarch64Codegen::new()))
        }
        _ => Err(InferenceError::Unsupported(format!(
            "no codegen backend for ISA level {:?} on this platform",
            isa
        ))),
    }
}
