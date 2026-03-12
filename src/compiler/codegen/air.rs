//! AirCodeGen — Metal AIR / MSL code generation backend.
//!
//! Phase 3 code generator targeting Apple Metal. Initial implementation uses
//! MSL (Metal Shading Language) text compilation via `MTLDevice.newLibrary(source:)`.
//! Future iterations will emit AIR bitcode directly for faster compilation.
//!
//! Gated behind `#[cfg(feature = "jit-metal")]`.

use crate::compiler::codegen::emitter::{MachineCodeEmitter, Platform, PlatformBackend};
use crate::compiler::codegen::CodegenOutput;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::CompilerGraph;
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::dispatch::DeviceProfile;

// ── AirCodeGen ──────────────────────────────────────────────────────────────

/// Metal AIR / MSL code generator.
///
/// Generates Metal Shading Language compute kernels from a `FusionPlan`.
/// The MSL text is compiled at runtime via `MTLDevice.newLibrary(source:)`.
pub struct AirCodeGen {
    /// Metal GPU family (e.g. 7 = Apple7, 9 = Apple9).
    gpu_family: u32,
    /// Accumulated MSL source buffer.
    msl_buffer: String,
}

impl AirCodeGen {
    /// Create a new `AirCodeGen` for the given GPU family.
    pub fn new(gpu_family: u32) -> Self {
        Self {
            gpu_family,
            msl_buffer: String::new(),
        }
    }

    /// Emit the standard MSL header (includes, using namespace).
    pub fn emit_msl_header(&mut self) -> &str {
        self.msl_buffer.clear();
        self.msl_buffer.push_str("#include <metal_stdlib>\n");
        self.msl_buffer.push_str("using namespace metal;\n\n");
        &self.msl_buffer
    }

    /// Emit a simple elementwise compute kernel.
    ///
    /// Generates a kernel that applies `body_expr` to each element:
    /// ```msl
    /// kernel void <name>(device float* input [[buffer(0)]],
    ///                     device float* output [[buffer(1)]],
    ///                     uint tid [[thread_position_in_grid]]) {
    ///     output[tid] = <body_expr>;
    /// }
    /// ```
    ///
    /// `body_expr` should reference `input[tid]` for the input value.
    pub fn emit_elementwise_kernel(&mut self, name: &str, body_expr: &str) -> &str {
        use std::fmt::Write;

        write!(
            self.msl_buffer,
            "kernel void {name}(\n\
             \x20   device const float* input [[buffer(0)]],\n\
             \x20   device float* output [[buffer(1)]],\n\
             \x20   uint tid [[thread_position_in_grid]]\n\
             ) {{\n\
             \x20   output[tid] = {body_expr};\n\
             }}\n\n"
        )
        .expect("write to String cannot fail");

        &self.msl_buffer
    }

    /// Get the current accumulated MSL source.
    pub fn msl_source(&self) -> &str {
        &self.msl_buffer
    }

    /// SIMD width for this GPU family.
    ///
    /// Apple GPUs execute in SIMD groups of 32 threads.
    fn metal_simd_width(&self) -> usize {
        32
    }
}

impl MachineCodeEmitter for AirCodeGen {
    fn emit_plan(
        &mut self,
        _plan: &FusionPlan,
        _graph: &CompilerGraph,
        _alloc: &BufferAllocation,
        _profile: &DeviceProfile,
        _registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        // TODO: Implement MSL generation from FusionPlan.
        // The pipeline is: FusionPlan -> MSL text -> MTLLibrary (via MetalDevice::compile_msl).
        // For now, return an explicit error per project policy (no silent fallback).
        Err(format!(
            "AirCodeGen: emit_plan not yet implemented (gpu_family={})",
            self.gpu_family
        ))
    }

    fn simd_width(&self) -> usize {
        self.metal_simd_width()
    }
}

// ── MetalBackend ────────────────────────────────────────────────────────────

/// Metal platform backend — factory for `AirCodeGen` emitters.
pub struct MetalBackend {
    gpu_family: u32,
}

impl MetalBackend {
    /// Create a new `MetalBackend` for the given GPU family.
    pub fn new(gpu_family: u32) -> Self {
        Self { gpu_family }
    }
}

impl PlatformBackend for MetalBackend {
    type Emitter = AirCodeGen;

    fn new_emitter(&self, _profile: &DeviceProfile) -> Self::Emitter {
        AirCodeGen::new(self.gpu_family)
    }

    fn platform(&self) -> Platform {
        Platform::Metal { gpu_family: self.gpu_family }
    }

    /// Apple GPUs don't have a traditional "SIMD register file" like x86/ARM.
    /// They use a 32-wide SIMD group with a large register file per thread.
    /// We report 32 as a reasonable analogy (SIMD group width).
    fn num_simd_regs(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_codegen_msl_header() {
        let mut cg = AirCodeGen::new(9);
        let header = cg.emit_msl_header();
        assert!(header.contains("#include <metal_stdlib>"));
        assert!(header.contains("using namespace metal;"));
    }

    #[test]
    fn test_air_codegen_elementwise_kernel() {
        let mut cg = AirCodeGen::new(9);
        cg.emit_msl_header();
        let src = cg.emit_elementwise_kernel("silu", "input[tid] / (1.0 + exp(-input[tid]))");
        assert!(src.contains("kernel void silu("));
        assert!(src.contains("thread_position_in_grid"));
        assert!(src.contains("input[tid] / (1.0 + exp(-input[tid]))"));
    }

    #[test]
    fn test_air_codegen_simd_width() {
        let cg = AirCodeGen::new(7);
        assert_eq!(cg.simd_width(), 32);
    }

    #[test]
    fn test_air_codegen_emit_plan_returns_err() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::FusionPlan;
        use crate::compiler::graph::CompilerGraph;
        use crate::dispatch::DeviceProfile;

        let mut cg = AirCodeGen::new(9);
        let plan = FusionPlan {
            groups: vec![],
            op_to_group: Default::default(),
        };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation {
            slots: vec![],
            total_bytes: 0,
            num_tensors: 0,
            bytes_saved: 0,
        };
        let profile = DeviceProfile::detect();

        let result = cg.emit_plan(&plan, &graph, &alloc, &profile, None);
        match result {
            Err(msg) => assert!(msg.contains("not yet implemented")),
            Ok(_) => panic!("expected Err from unimplemented emit_plan"),
        }
    }

    #[test]
    fn test_metal_backend_platform() {
        let backend = MetalBackend::new(9);
        let plat = backend.platform();
        assert!(matches!(plat, Platform::Metal { gpu_family: 9 }));
        assert_eq!(backend.num_simd_regs(), 32);
    }
}
