//! x86_64 code generator — AVX2 / AVX-512 machine code emission.
//!
//! Phase 1: emits a minimal stub that calls back into Rust (trampoline).
//! Phase 2 (future): full JIT with fused GEMM + activation + norm.

use crate::compiler::ir::LayerIR;
use crate::compiler::planner::ExecutionPlan;
use crate::compiler::codegen::{CodegenOutput, LayerCodegen};
use crate::dispatch::device_profile::IsaLevel;
use crate::inference::types::InferenceError;

/// x86_64 code generator.
pub struct X86_64Codegen {
    isa: IsaLevel,
}

impl X86_64Codegen {
    pub fn new(isa: IsaLevel) -> Self {
        X86_64Codegen { isa }
    }
}

impl LayerCodegen for X86_64Codegen {
    fn generate(
        &self,
        ir: &LayerIR,
        plan: &ExecutionPlan,
    ) -> Result<CodegenOutput, InferenceError> {
        // Phase 1: emit a trampoline stub.
        // The stub saves callee-saved registers, sets up the stack frame,
        // and calls back into a Rust function that executes the fallback path.
        //
        // This allows the compilation cache and dispatch infrastructure to
        // work end-to-end while the actual JIT codegen is developed.

        let mut code = Vec::with_capacity(256);

        // x86_64 System V ABI:
        // Args: rdi, rsi, rdx, rcx, r8, r9 (matches CompiledLayerFn signature)
        //
        // Emit: push rbp; mov rbp, rsp; ... ; pop rbp; ret
        // For now, just emit a `ret` — the caller handles fallback.

        // push rbp
        code.push(0x55);
        // mov rbp, rsp
        code.extend_from_slice(&[0x48, 0x89, 0xE5]);
        // nop sled (placeholder for future codegen)
        for _ in 0..8 {
            code.push(0x90); // nop
        }
        // pop rbp
        code.push(0x5D);
        // ret
        code.push(0xC3);

        Ok(CodegenOutput {
            code,
            scratchpad_bytes: plan.scratchpad_bytes,
        })
    }

    fn isa_level(&self) -> IsaLevel {
        self.isa
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::compiler::ir::LayerIR;
    use crate::compiler::planner::ExecutionPlan;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_x86_codegen_stub() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile);

        let codegen = X86_64Codegen::new(profile.isa);
        let output = codegen.generate(&ir, &plan).unwrap();

        assert!(!output.code.is_empty());
        assert!(output.scratchpad_bytes > 0);

        // Verify the stub is callable (it just returns)
        use crate::compiler::executable::CompiledLayer;
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap();
        unsafe {
            let f = layer.entry_point();
            // Call with null pointers — the stub just returns
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
