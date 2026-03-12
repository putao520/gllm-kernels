//! PtxCodeGen — Phase 3 PTX code generation for NVIDIA GPUs.
//!
//! Implements `MachineCodeEmitter` to produce PTX text from a `FusionPlan`.
//! The PTX is later loaded via `cuModuleLoadData` where the NVIDIA driver
//! compiles it to SASS machine code for the target SM.
//!
//! This is the GPU counterpart of `X86CodeGen` / `DynasmAArch64CodeGen`.

use std::fmt::Write;

use crate::compiler::codegen::emitter::{MachineCodeEmitter, PlatformBackend, Platform};
use crate::compiler::codegen::CodegenOutput;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::CompilerGraph;
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::dispatch::DeviceProfile;

// ── PtxCodeGen ──────────────────────────────────────────────────────

/// PTX code generator targeting NVIDIA GPUs.
///
/// Generates PTX virtual ISA text that the NVIDIA driver compiles to native
/// SASS at module load time (`cuModuleLoadData`).
pub struct PtxCodeGen {
    /// Target SM version (e.g. 70, 80, 89, 90).
    sm_version: u32,
    /// Accumulated PTX text buffer.
    ptx_buffer: String,
}

impl PtxCodeGen {
    /// Create a new PtxCodeGen targeting the given SM version.
    pub fn new(sm_version: u32) -> Self {
        Self {
            sm_version,
            ptx_buffer: String::with_capacity(4096),
        }
    }

    /// Target SM version.
    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    /// Return the accumulated PTX text.
    pub fn ptx_text(&self) -> &str {
        &self.ptx_buffer
    }

    // ── PTX generation helpers ──────────────────────────────────────

    /// Emit the PTX module header (`.version`, `.target`, `.address_size`).
    ///
    /// PTX version is chosen based on SM:
    /// - sm_70+: PTX 6.0
    /// - sm_80+: PTX 7.0
    /// - sm_90+: PTX 8.0
    pub fn emit_header(&mut self) {
        let ptx_version = match self.sm_version {
            90.. => "8.0",
            80..=89 => "7.0",
            _ => "6.0",
        };

        self.ptx_buffer.clear();
        writeln!(self.ptx_buffer, ".version {ptx_version}").unwrap();
        writeln!(self.ptx_buffer, ".target sm_{}", self.sm_version).unwrap();
        writeln!(self.ptx_buffer, ".address_size 64").unwrap();
        writeln!(self.ptx_buffer).unwrap();
    }

    /// Emit a simple elementwise kernel that applies `op` to each element.
    ///
    /// Supported ops: "add", "mul", "silu", "relu", "neg".
    ///
    /// Generated kernel signature:
    /// ```ptx
    /// .visible .entry kernel_<op>(
    ///     .param .u64 input,
    ///     .param .u64 output,
    ///     .param .u32 n
    /// )
    /// ```
    pub fn emit_elementwise_kernel(&mut self, op: &str) -> Result<(), String> {
        let kernel_name = format!("kernel_{op}");

        // Kernel entry
        writeln!(self.ptx_buffer, ".visible .entry {kernel_name}(").unwrap();
        writeln!(self.ptx_buffer, "    .param .u64 param_input,").unwrap();
        writeln!(self.ptx_buffer, "    .param .u64 param_output,").unwrap();
        writeln!(self.ptx_buffer, "    .param .u32 param_n").unwrap();
        writeln!(self.ptx_buffer, ")").unwrap();
        writeln!(self.ptx_buffer, "{{").unwrap();

        // Register declarations
        writeln!(self.ptx_buffer, "    .reg .u64 %rd<4>;").unwrap();
        writeln!(self.ptx_buffer, "    .reg .u32 %r<4>;").unwrap();
        writeln!(self.ptx_buffer, "    .reg .f32 %f<4>;").unwrap();
        writeln!(self.ptx_buffer, "    .reg .pred %p<2>;").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Load parameters
        writeln!(self.ptx_buffer, "    ld.param.u64 %rd0, [param_input];").unwrap();
        writeln!(self.ptx_buffer, "    ld.param.u64 %rd1, [param_output];").unwrap();
        writeln!(self.ptx_buffer, "    ld.param.u32 %r0, [param_n];").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Compute global thread index: tid = blockIdx.x * blockDim.x + threadIdx.x
        writeln!(self.ptx_buffer, "    mov.u32 %r1, %ctaid.x;").unwrap();
        writeln!(self.ptx_buffer, "    mov.u32 %r2, %ntid.x;").unwrap();
        writeln!(self.ptx_buffer, "    mad.lo.u32 %r1, %r1, %r2, %tid.x;").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Bounds check: if (tid >= n) return
        writeln!(self.ptx_buffer, "    setp.ge.u32 %p0, %r1, %r0;").unwrap();
        writeln!(self.ptx_buffer, "    @%p0 bra DONE;").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Compute byte offset: offset = tid * 4 (f32)
        writeln!(self.ptx_buffer, "    mul.wide.u32 %rd2, %r1, 4;").unwrap();
        writeln!(self.ptx_buffer, "    add.u64 %rd2, %rd0, %rd2;").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Load input element
        writeln!(self.ptx_buffer, "    ld.global.f32 %f0, [%rd2];").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Apply operation
        match op {
            "add" => {
                // Binary add: output[i] = input[i] + input[i] (self-add for demo)
                writeln!(self.ptx_buffer, "    add.f32 %f1, %f0, %f0;").unwrap();
            }
            "mul" => {
                // Binary mul: output[i] = input[i] * input[i] (square for demo)
                writeln!(self.ptx_buffer, "    mul.f32 %f1, %f0, %f0;").unwrap();
            }
            "neg" => {
                writeln!(self.ptx_buffer, "    neg.f32 %f1, %f0;").unwrap();
            }
            "relu" => {
                writeln!(self.ptx_buffer, "    max.f32 %f1, %f0, 0f00000000;").unwrap();
            }
            "silu" => {
                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                writeln!(self.ptx_buffer, "    neg.f32 %f1, %f0;").unwrap();
                // exp(-x) via exp2(x * log2(e)): log2(e) ≈ 1.4426950408889634
                writeln!(self.ptx_buffer, "    mul.f32 %f1, %f1, 0f3FB8AA3B;").unwrap();
                writeln!(self.ptx_buffer, "    ex2.approx.f32 %f1, %f1;").unwrap();
                // 1 + exp(-x)
                writeln!(self.ptx_buffer, "    add.f32 %f1, %f1, 0f3F800000;").unwrap();
                // x / (1 + exp(-x))
                writeln!(self.ptx_buffer, "    div.approx.f32 %f1, %f0, %f1;").unwrap();
            }
            _ => {
                return Err(format!("PtxCodeGen: unsupported elementwise op '{op}'"));
            }
        }
        writeln!(self.ptx_buffer).unwrap();

        // Compute output byte offset
        writeln!(self.ptx_buffer, "    mul.wide.u32 %rd3, %r1, 4;").unwrap();
        writeln!(self.ptx_buffer, "    add.u64 %rd3, %rd1, %rd3;").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Store result
        writeln!(self.ptx_buffer, "    st.global.f32 [%rd3], %f1;").unwrap();
        writeln!(self.ptx_buffer).unwrap();

        // Done label and return
        writeln!(self.ptx_buffer, "DONE:").unwrap();
        writeln!(self.ptx_buffer, "    ret;").unwrap();
        writeln!(self.ptx_buffer, "}}").unwrap();

        Ok(())
    }
}

impl MachineCodeEmitter for PtxCodeGen {
    fn emit_plan(
        &mut self,
        _plan: &FusionPlan,
        _graph: &CompilerGraph,
        _alloc: &BufferAllocation,
        _profile: &DeviceProfile,
        _registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        Err("PtxCodeGen: not yet implemented".into())
    }

    /// PTX threads process one element each — "SIMD width" is effectively 1
    /// from the code generator's perspective (parallelism is expressed via
    /// grid/block dimensions, not vector registers).
    fn simd_width(&self) -> usize {
        1
    }
}

// ── PtxBackend ──────────────────────────────────────────────────────

/// Platform backend factory for CUDA PTX code generation.
pub struct PtxBackend {
    sm_version: u32,
}

impl PtxBackend {
    pub fn new(sm_version: u32) -> Self {
        Self { sm_version }
    }
}

impl PlatformBackend for PtxBackend {
    type Emitter = PtxCodeGen;

    fn new_emitter(&self, _profile: &DeviceProfile) -> PtxCodeGen {
        PtxCodeGen::new(self.sm_version)
    }

    fn platform(&self) -> Platform {
        Platform::Cuda { sm_version: self.sm_version }
    }

    /// CUDA GPUs have a large register file but it's managed per-thread by the
    /// hardware scheduler, not explicitly by the code generator. Return 255 as
    /// the per-thread register budget (CUDA default max).
    fn num_simd_regs(&self) -> usize {
        255
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_header_sm80() {
        let mut cg = PtxCodeGen::new(80);
        cg.emit_header();
        let ptx = cg.ptx_text();
        assert!(ptx.contains(".version 7.0"), "sm_80 should use PTX 7.0");
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_ptx_header_sm90() {
        let mut cg = PtxCodeGen::new(90);
        cg.emit_header();
        let ptx = cg.ptx_text();
        assert!(ptx.contains(".version 8.0"), "sm_90 should use PTX 8.0");
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn test_ptx_header_sm70() {
        let mut cg = PtxCodeGen::new(70);
        cg.emit_header();
        let ptx = cg.ptx_text();
        assert!(ptx.contains(".version 6.0"), "sm_70 should use PTX 6.0");
    }

    #[test]
    fn test_ptx_elementwise_silu() {
        let mut cg = PtxCodeGen::new(80);
        cg.emit_header();
        cg.emit_elementwise_kernel("silu").expect("silu should be supported");
        let ptx = cg.ptx_text();
        assert!(ptx.contains(".entry kernel_silu"));
        assert!(ptx.contains("ex2.approx.f32"), "SiLU needs exp2");
        assert!(ptx.contains("div.approx.f32"), "SiLU needs division");
    }

    #[test]
    fn test_ptx_elementwise_relu() {
        let mut cg = PtxCodeGen::new(80);
        cg.emit_header();
        cg.emit_elementwise_kernel("relu").expect("relu should be supported");
        let ptx = cg.ptx_text();
        assert!(ptx.contains(".entry kernel_relu"));
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn test_ptx_elementwise_neg() {
        let mut cg = PtxCodeGen::new(80);
        cg.emit_header();
        cg.emit_elementwise_kernel("neg").expect("neg should be supported");
        let ptx = cg.ptx_text();
        assert!(ptx.contains("neg.f32"));
    }

    #[test]
    fn test_ptx_unsupported_op() {
        let mut cg = PtxCodeGen::new(80);
        cg.emit_header();
        let result = cg.emit_elementwise_kernel("unknown_op");
        assert!(result.is_err());
    }

    #[test]
    fn test_ptx_emit_plan_not_yet_implemented() {
        use crate::compiler::fusion::FusionPlan;
        use crate::compiler::graph::CompilerGraph;
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::dispatch::DeviceProfile;

        let mut cg = PtxCodeGen::new(80);
        let plan = FusionPlan { groups: vec![], op_to_group: Default::default() };
        let graph = CompilerGraph::new();
        let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
        let profile = DeviceProfile::detect();

        let result = cg.emit_plan(&plan, &graph, &alloc, &profile, None);
        assert!(result.is_err());
        match result { Err(e) => assert!(e.contains("not yet implemented")), Ok(_) => panic!("expected Err"), }
    }

    #[test]
    fn test_ptx_backend_platform() {
        let backend = PtxBackend::new(80);
        assert!(matches!(backend.platform(), Platform::Cuda { sm_version: 80 }));
        assert_eq!(backend.num_simd_regs(), 255);
    }

    #[test]
    fn test_ptx_backend_emitter() {
        let backend = PtxBackend::new(89);
        let profile = DeviceProfile::detect();
        let emitter = backend.new_emitter(&profile);
        assert_eq!(emitter.sm_version(), 89);
        assert_eq!(emitter.simd_width(), 1);
    }
}
