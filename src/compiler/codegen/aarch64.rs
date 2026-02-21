//! aarch64 code generator — NEON machine code emission.
//!
//! Phase 1: emits a minimal stub (trampoline).
//! Phase 2 (future): full JIT with fused GEMM + activation + norm.

use crate::compiler::ir::LayerIR;
use crate::compiler::planner::ExecutionPlan;
use crate::compiler::codegen::{CodegenOutput, LayerCodegen};
use crate::dispatch::device_profile::IsaLevel;
use crate::inference::types::InferenceError;

/// aarch64 code generator.
pub struct Aarch64Codegen;

impl Aarch64Codegen {
    pub fn new() -> Self {
        Aarch64Codegen
    }
}

impl LayerCodegen for Aarch64Codegen {
    fn generate(
        &self,
        _ir: &LayerIR,
        plan: &ExecutionPlan,
    ) -> Result<CodegenOutput, InferenceError> {
        // Phase 1: emit a minimal stub that just returns.
        let mut code = Vec::with_capacity(16);

        // aarch64: `ret` = 0xD65F03C0
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

        Ok(CodegenOutput {
            code,
            scratchpad_bytes: plan.scratchpad_bytes,
        })
    }

    fn isa_level(&self) -> IsaLevel {
        IsaLevel::Neon
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use crate::compiler::ir::LayerIR;
    use crate::compiler::planner::ExecutionPlan;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_aarch64_codegen_stub() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile);

        let codegen = Aarch64Codegen::new();
        let output = codegen.generate(&ir, &plan).unwrap();

        assert!(!output.code.is_empty());

        use crate::compiler::executable::CompiledLayer;
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap();
        unsafe {
            let f = layer.entry_point();
            f(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
        }
    }
}
