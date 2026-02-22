//! x86_64 JIT code generation via iced-x86 CodeAssembler.
//!
//! TODO: Rewrite per SPEC §8.5 Phase 3 — programmatic code generation.
//!
//! The correct implementation should:
//! - Generate complete GEMM microkernels (K-loop, FMA sequences) via iced-x86
//! - Inject epilogue ops (SiLU/GELU/BiasAdd) into accumulator registers before store
//! - Generate fused elementwise loops (data flows through registers, no intermediate memory)
//! - Generate tile-level fusion (predecessor tile computation embedded in GEMM MC loop)
//!
//! The previous implementation was a trampoline dispatcher that emitted `call`
//! instructions to pre-compiled Rust functions. This violated the SPEC's core
//! principle: "融合 = 全新代码生成".

use super::CodegenOutput;

/// Emit a minimal x86_64 stub (push rbp; mov rbp,rsp; nops; pop rbp; ret).
///
/// Used as fallback until the real Phase 3 code generator is implemented.
pub fn emit_stub() -> CodegenOutput {
    let mut code = Vec::with_capacity(16);
    code.push(0x55);                          // push rbp
    code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
    for _ in 0..8 {
        code.push(0x90);                      // nop
    }
    code.push(0x5D);                          // pop rbp
    code.push(0xC3);                          // ret
    CodegenOutput { code, scratchpad_bytes: 0 }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::compiler::executable::CompiledLayer;

    #[test]
    fn test_x86_stub_callable() {
        let output = emit_stub();
        assert!(!output.code.is_empty());
        let layer = CompiledLayer::from_code(&output.code, 0, 0).unwrap();
        unsafe {
            let f = layer.entry_point();
            f(
                std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(),
                0, 0,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
    }
}
