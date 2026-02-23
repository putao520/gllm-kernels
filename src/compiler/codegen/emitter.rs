//! Emitter — legacy scratchpad layout + stub code generation.
//!
//! Phase 3 code generation is now implemented in `codegen::x86_64::jit::X86CodeGen`
//! (under the `jit-x86` feature flag), which uses `FusionPlan` + `BufferAllocation`
//! to emit native x86_64 machine code via iced-x86.
//!
//! What remains here:
//! - `ScratchpadLayout` / `compute_layout()`: legacy bump allocator for scratchpad
//!   planning. New code should use `buffer_alloc::allocate_buffers()` which implements
//!   interval-graph coloring for buffer reuse (SPEC Phase 2 Step 4).
//! - `emit_stub_code()`: fallback stub used when the `jit-x86` feature is not enabled.

use std::collections::HashMap;
use crate::compiler::graph::{CompilerGraph, TensorId};
use crate::compiler::codegen::CodegenOutput;

/// A scratchpad memory layout: maps each intermediate tensor to an offset.
#[derive(Debug, Clone)]
pub struct ScratchpadLayout {
    /// TensorId → byte offset within the scratchpad buffer.
    pub offsets: HashMap<TensorId, usize>,
    /// Total scratchpad size in bytes.
    pub total_bytes: usize,
}

/// Compute the scratchpad memory layout for all intermediate tensors.
///
/// Graph inputs (weights, activations) are passed via function arguments,
/// not the scratchpad. Only intermediate tensors (produced by ops) need
/// scratchpad space.
///
/// NOTE: This is a legacy bump allocator with no buffer sharing. The new path
/// uses `buffer_alloc::allocate_buffers()` which implements interval-graph
/// coloring for optimal buffer reuse. Retained for backward compatibility
/// with `emit_stub_code()`.
pub fn compute_layout(graph: &CompilerGraph) -> ScratchpadLayout {
    let mut offsets = HashMap::new();
    let mut current_offset: usize = 0;

    // Align to 64 bytes (cache line) for each tensor
    const ALIGN: usize = 64;

    for tensor in &graph.tensors {
        // Skip graph inputs (no producer = function argument)
        if tensor.producer.is_none() {
            continue;
        }

        let num_elements: usize = tensor.shape.iter().product();
        let elem_size = tensor.dtype.size_bytes();
        let byte_size = num_elements * elem_size;

        // Align up
        current_offset = (current_offset + ALIGN - 1) & !(ALIGN - 1);
        offsets.insert(tensor.id, current_offset);
        current_offset += byte_size;
    }

    // Final alignment
    let total_bytes = (current_offset + ALIGN - 1) & !(ALIGN - 1);

    ScratchpadLayout {
        offsets,
        total_bytes,
    }
}

/// Temporary: generate a stub CodegenOutput.
///
/// This is a placeholder until the real Phase 3 code generator is implemented.
/// It produces a valid but no-op function.
pub fn emit_stub_code(graph: &CompilerGraph) -> CodegenOutput {
    let layout = compute_layout(graph);
    let scratchpad_bytes = layout.total_bytes;

    #[cfg(target_arch = "x86_64")]
    let code = super::x86_64::emit_stub().code;

    #[cfg(target_arch = "aarch64")]
    let code = super::aarch64::emit_stub().code;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let code = vec![0xC3]; // ret

    CodegenOutput {
        code,
        scratchpad_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::CompilerGraph;
    use crate::compiler::ir::LayerIR;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_compute_layout() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let layout = compute_layout(&graph);

        // Should have offsets for intermediate tensors (not graph inputs)
        assert!(!layout.offsets.is_empty());
        assert!(layout.total_bytes > 0);

        // All offsets should be 64-byte aligned
        for (_, &offset) in &layout.offsets {
            assert_eq!(offset % 64, 0, "offset {} not 64-byte aligned", offset);
        }

        eprintln!(
            "Layout: {} tensors mapped, {} total bytes",
            layout.offsets.len(),
            layout.total_bytes
        );
    }

    #[test]
    fn test_emit_stub_code() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let output = emit_stub_code(&graph);

        assert!(!output.code.is_empty());
        assert!(output.scratchpad_bytes > 0);
    }
}
