//! Emitter — Phase 3 trait abstraction + legacy scratchpad layout.
//!
//! ## Trait Architecture (SPEC §8.6)
//!
//! Platform differences in Phase 3 code generation are encapsulated via two traits:
//!
//! - `MachineCodeEmitter`: the code-generation interface (emit_plan, simd_width).
//!   Implemented by `X86CodeGen` (jit-x86) and `AArch64CodeGen` (jit-aarch64).
//!
//! - `PlatformBackend`: factory + platform metadata. Implemented by `X86Backend`
//!   and `Arm64Backend`. The compiler pipeline depends only on `PlatformBackend`,
//!   not on the concrete emitter type.
//!
//! ## Legacy
//!
//! `ScratchpadLayout` / `compute_layout()` and `emit_stub_code()` are retained for
//! backward compatibility with the non-JIT fallback path and existing tests.
//! New code should use `buffer_alloc::allocate_buffers()` for buffer planning.

use std::collections::HashMap;
use crate::compiler::graph::{CompilerGraph, TensorId};
use crate::compiler::codegen::CodegenOutput;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::dispatch::DeviceProfile;

// ── Trait definitions ─────────────────────────────────────────────────────────

/// Platform identifier — carries capability flags used by the compiler pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// x86_64 with optional AVX-512 support.
    X86_64 { avx512: bool },
    /// AArch64 with optional SVE support.
    Aarch64 { sve: bool },
}

impl Platform {
    /// Detect the current host platform.
    pub fn detect(profile: &DeviceProfile) -> Self {
        use crate::dispatch::IsaLevel;

        #[cfg(target_arch = "x86_64")]
        return Platform::X86_64 {
            avx512: matches!(profile.isa, IsaLevel::Avx512),
        };

        #[cfg(target_arch = "aarch64")]
        return Platform::Aarch64 { sve: false };

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return Platform::X86_64 { avx512: false };
    }
}

/// Phase 3 code-generation interface.
///
/// Implemented by each platform's JIT backend. The compiler pipeline calls
/// `emit_plan` to produce native machine code for a complete fusion plan.
///
/// The `registry` parameter carries scalar-op metadata used by x86_64 for
/// epilogue injection; aarch64 implementations may ignore it.
pub trait MachineCodeEmitter {
    /// Generate native machine code for a complete fusion plan.
    fn emit_plan(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
        alloc: &BufferAllocation,
        profile: &DeviceProfile,
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String>;

    /// Number of f32 elements per SIMD register on this backend.
    fn simd_width(&self) -> usize;
}

/// Platform backend factory — provides platform metadata and constructs emitters.
///
/// The compiler pipeline depends only on this trait, not on the concrete emitter
/// type, keeping Phase 1-2 (DAG construction, fusion decisions) platform-agnostic.
pub trait PlatformBackend {
    /// The concrete emitter type produced by this backend.
    type Emitter: MachineCodeEmitter;

    /// Construct a new emitter for the given device profile.
    fn new_emitter(&self, profile: &DeviceProfile) -> Self::Emitter;

    /// The platform this backend targets.
    fn platform(&self) -> Platform;

    /// Number of SIMD registers available on this platform.
    fn num_simd_regs(&self) -> usize;
}

// ── Legacy scratchpad layout (retained for backward compatibility) ─────────────

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

    #[test]
    fn test_platform_detect() {
        let profile = DeviceProfile::detect();
        let platform = Platform::detect(&profile);
        // Just verify it returns a valid variant without panicking
        match platform {
            Platform::X86_64 { .. } => {},
            Platform::Aarch64 { .. } => {},
        }
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_backend_trait_roundtrip() {
        use crate::compiler::codegen::x86_64::X86Backend;

        let backend = X86Backend;
        let profile = DeviceProfile::detect();
        let emitter = backend.new_emitter(&profile);

        let plat = backend.platform();
        assert!(matches!(plat, Platform::X86_64 { .. }));
        assert!(backend.num_simd_regs() == 16 || backend.num_simd_regs() == 32);
        assert!(emitter.simd_width() == 8 || emitter.simd_width() == 16);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emitter_emit_plan_standalone_silu() {
        use crate::compiler::codegen::x86_64::jit::X86CodeGen;
        use crate::compiler::graph::{CompilerGraph, OpKind};
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::inference::types::DType;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![16], DType::F32);
        let output = g.add_tensor("output", vec![16], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        };
        let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
        let profile = DeviceProfile::detect();

        let mut emitter: Box<dyn MachineCodeEmitter> = Box::new(X86CodeGen::new(&profile));
        let output = emitter.emit_plan(&plan, &g, &alloc, &profile, None).unwrap();
        assert!(!output.code.is_empty(), "emit_plan via trait should produce code");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emitter_emit_plan_with_registry() {
        use crate::compiler::codegen::x86_64::jit::X86CodeGen;
        use crate::compiler::graph::{CompilerGraph, OpKind};
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::inference::types::DType;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![16], DType::F32);
        let output = g.add_tensor("output", vec![16], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        };
        let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();

        let mut emitter = X86CodeGen::new(&profile);
        let output = emitter.emit_plan(&plan, &g, &alloc, &profile, Some(&registry)).unwrap();
        assert!(!output.code.is_empty(), "registry-driven emit_plan should produce code");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_platform_detect_consistency() {
        use crate::compiler::codegen::x86_64::X86Backend;

        let profile = DeviceProfile::detect();
        let detected = Platform::detect(&profile);
        let backend = X86Backend;
        let backend_plat = backend.platform();

        // Both should report X86_64 on x86_64 hosts
        match (detected, backend_plat) {
            (Platform::X86_64 { avx512: a }, Platform::X86_64 { avx512: b }) => {
                assert_eq!(a, b, "Platform::detect and X86Backend::platform disagree on avx512");
            }
            _ => panic!("platform mismatch between detect and backend"),
        }
    }
}
