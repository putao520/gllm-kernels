//! aarch64 code generation — stub until native aarch64 JIT backend is implemented.

use super::CodegenOutput;

/// Emit a minimal aarch64 stub (`ret` = 0xD65F03C0).
pub fn emit_stub() -> CodegenOutput {
    let mut code = Vec::with_capacity(4);
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes()); // ret
    CodegenOutput { code, scratchpad_bytes: 0 }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use crate::compiler::executable::CompiledLayer;

    #[test]
    fn test_aarch64_stub_callable() {
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
