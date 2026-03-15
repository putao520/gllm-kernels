//! AirCodeGen — Metal AIR / MSL code generation backend.
//!
//! Phase 3 code generator targeting Apple Metal. Generates MSL (Metal Shading
//! Language) compute kernels from a `FusionPlan`. The MSL text is compiled at
//! runtime via `MTLDevice.newLibrary(source:)`.
//!
//! Gated behind `#[cfg(feature = "jit-metal")]`.

use std::fmt::Write;
use crate::compiler::codegen::emitter::{MachineCodeEmitter, Platform, PlatformBackend};
use crate::compiler::codegen::CodegenOutput;
use crate::compiler::codegen::gpu_ir::trace_emitter::{GpuDialect, MslDialect};
use crate::compiler::codegen::gpu_ir::plan_emitter::gpu_emit_plan;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::CompilerGraph;
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::TraceOp;
use crate::dispatch::DeviceProfile;

// Re-export GEMM emitters from air_gemm submodule.
pub(crate) use super::air_gemm::*;

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
    /// Generates a kernel that applies `body_expr` to each element.
    /// `body_expr` should reference `input[tid]` for the input value.
    pub fn emit_elementwise_kernel(&mut self, name: &str, body_expr: &str) -> &str {
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

// ── MSL TraceOp → expression codegen ────────────────────────────────────────

/// Emit an MSL expression for a single `TraceOp`, given the SSA variable names
/// of all prior ops. Returns the expression string for this op.
fn trace_op_to_msl(op: &TraceOp, vars: &[String]) -> String {
    match op {
        TraceOp::Input(idx) => format!("v_input{idx}"),
        TraceOp::Const(val) => {
            if *val == f64::NEG_INFINITY {
                "(-INFINITY)".to_string()
            } else {
                format!("float({val:e})")
            }
        }
        TraceOp::Add(a, b) => format!("({} + {})", vars[*a as usize], vars[*b as usize]),
        TraceOp::Sub(a, b) => format!("({} - {})", vars[*a as usize], vars[*b as usize]),
        TraceOp::Mul(a, b) => format!("({} * {})", vars[*a as usize], vars[*b as usize]),
        TraceOp::Div(a, b) => format!("({} / {})", vars[*a as usize], vars[*b as usize]),
        TraceOp::Fma(a, b, c) => format!(
            "fma({}, {}, {})",
            vars[*a as usize], vars[*b as usize], vars[*c as usize]
        ),
        TraceOp::Neg(a) => format!("(-{})", vars[*a as usize]),
        TraceOp::Abs(a) => format!("abs({})", vars[*a as usize]),
        TraceOp::Exp(a) => format!("exp({})", vars[*a as usize]),
        TraceOp::Sqrt(a) => format!("sqrt({})", vars[*a as usize]),
        TraceOp::Rsqrt(a) => format!("rsqrt({})", vars[*a as usize]),
        TraceOp::Tanh(a) => format!("tanh({})", vars[*a as usize]),
        TraceOp::Recip(a) => format!("(1.0f / {})", vars[*a as usize]),
        TraceOp::Log(a) => format!("log({})", vars[*a as usize]),
        TraceOp::Max(a, b) => format!("max({}, {})", vars[*a as usize], vars[*b as usize]),
        TraceOp::Min(a, b) => format!("min({}, {})", vars[*a as usize], vars[*b as usize]),
    }
}

/// Emit a sequence of `TraceOp`s as MSL statements, binding each to `t{base}_{i}`.
/// Returns the variable name of the last op (the result).
fn emit_trace_body(
    out: &mut String,
    ops: &[TraceOp],
    base: usize,
    input_bindings: &[String],
) -> String {
    let mut vars: Vec<String> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        let var_name = format!("t{}_{}", base, i);
        let expr = if let TraceOp::Input(idx) = op {
            input_bindings
                .get(*idx as usize)
                .cloned()
                .unwrap_or_else(|| format!("v_input{idx}"))
        } else {
            trace_op_to_msl(op, &vars)
        };
        writeln!(out, "    float {var_name} = {expr};").unwrap();
        vars.push(var_name);
    }
    vars.last()
        .cloned()
        .unwrap_or_else(|| "0.0f".to_string())
}

// ── Kernel generators per ComputePattern ────────────────────────────────────

/// Emit a generic reduction kernel in MSL.
///
/// Grid-stride accumulation + threadgroup shared memory tree reduction.
/// Block size = 256 (8 SIMD-groups).
pub(crate) fn emit_reduction_kernel_msl(
    out: &mut String,
    kernel_name: &str,
    identity: f64,
    combine: &[TraceOp],
) {
    let block_size: usize = 256;
    let id_bits = (identity as f32).to_bits();

    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input [[buffer(0)]],").unwrap();
    writeln!(out, "    device float* output [[buffer(1)]],").unwrap();
    writeln!(out, "    constant uint& n [[buffer(2)]],").unwrap();
    writeln!(out, "    uint lid [[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint grid_size [[threadgroups_per_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    threadgroup float sdata[{block_size}];").unwrap();
    writeln!(out, "    float acc = as_type<float>(0x{id_bits:08X}u);").unwrap();
    writeln!(out).unwrap();

    // Grid-stride accumulation.
    writeln!(out, "    for (uint i = gid * {block_size} + lid; i < n; i += grid_size * {block_size}) {{").unwrap();
    writeln!(out, "        float in0 = acc;").unwrap();
    writeln!(out, "        float in1 = input[i];").unwrap();
    let bindings = vec!["in0".to_string(), "in1".to_string()];
    let result = emit_trace_body(out, combine, 1, &bindings);
    writeln!(out, "        acc = {result};").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Shared memory tree reduction.
    writeln!(out, "    sdata[lid] = acc;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();

    let mut s = block_size / 2;
    while s > 0 {
        writeln!(out, "    if (lid < {s}) {{").unwrap();
        writeln!(out, "        float in0 = sdata[lid];").unwrap();
        writeln!(out, "        float in1 = sdata[lid + {s}];").unwrap();
        let bindings2 = vec!["in0".to_string(), "in1".to_string()];
        let r2 = emit_trace_body(out, combine, 2, &bindings2);
        writeln!(out, "        sdata[lid] = {r2};").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        s /= 2;
    }

    writeln!(out, "    if (lid == 0) output[gid] = sdata[0];").unwrap();
    writeln!(out, "}}\n").unwrap();
}

/// Emit a multi-input/multi-output injective kernel in MSL.
///
/// Uses `thread_position_in_grid` for global thread ID with bounds check.
pub(crate) fn emit_injective_kernel_msl(
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
    num_inputs: usize,
    num_outputs: usize,
) {
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    let mut buf_idx: usize = 0;
    for i in 0..num_inputs {
        writeln!(out, "    device const float* input{i} [[buffer({buf_idx})]],").unwrap();
        buf_idx += 1;
    }
    for i in 0..num_outputs {
        writeln!(out, "    device float* output{i} [[buffer({buf_idx})]],").unwrap();
        buf_idx += 1;
    }
    writeln!(out, "    constant uint& n [[buffer({buf_idx})]],").unwrap();
    writeln!(out, "    uint tid [[thread_position_in_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    if (tid >= n) return;").unwrap();

    let mut bindings = Vec::new();
    for i in 0..num_inputs {
        writeln!(out, "    float in{i} = input{i}[tid];").unwrap();
        bindings.push(format!("in{i}"));
    }

    let result = emit_trace_body(out, body, 0, &bindings);

    if num_outputs == 1 {
        writeln!(out, "    output0[tid] = {result};").unwrap();
    } else {
        let base = body.len().saturating_sub(num_outputs);
        for i in 0..num_outputs {
            let var = format!("t0_{}", base + i);
            writeln!(out, "    output{i}[tid] = {var};").unwrap();
        }
    }

    writeln!(out, "}}\n").unwrap();
}

// GEMM kernels (emit_gemm_kernel_msl, emit_gemm_simdgroup_msl) moved to air_gemm.rs

pub(crate) fn emit_mha_kernel_msl(
    out: &mut String,
    kernel_name: &str,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let block_size = head_dim.next_power_of_two().min(256);
    let scale_f32 = 1.0_f32 / (head_dim as f32).sqrt();
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* Q [[buffer(0)]],").unwrap();
    writeln!(out, "    device const float* K [[buffer(1)]],").unwrap();
    writeln!(out, "    device const float* V [[buffer(2)]],").unwrap();
    writeln!(out, "    device float* out [[buffer(3)]],").unwrap();
    writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint tid [[thread_index_in_threadgroup]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const uint SEQ = {seq_len}u;").unwrap();
    writeln!(out, "    const uint DIM = {head_dim}u;").unwrap();
    writeln!(out, "    const float scale = {scale_f32};").unwrap();
    writeln!(out, "    uint query_row = gid.x;").unwrap();
    writeln!(out, "    uint head = gid.y;").unwrap();
    writeln!(out, "    uint base = head * SEQ * DIM;").unwrap();
    writeln!(out, "    threadgroup float smem_scores[{block_size}];").unwrap();
    writeln!(out, "    threadgroup float smem_max[1];").unwrap();
    writeln!(out, "    threadgroup float smem_sum[1];").unwrap();
    writeln!(out).unwrap();
    // Compute scores: score[j] = dot(Q[query_row], K[j]) * scale
    writeln!(out, "    for (uint j = tid; j < SEQ; j += {block_size}u) {{").unwrap();
    writeln!(out, "        float dot = 0.0f;").unwrap();
    writeln!(out, "        for (uint d = 0; d < DIM; d++) {{").unwrap();
    writeln!(out, "            dot += Q[base + query_row * DIM + d] * K[base + j * DIM + d];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        smem_scores[j] = dot * scale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    // Softmax: find max
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float mx = smem_scores[0];").unwrap();
    writeln!(out, "        for (uint j = 1; j < SEQ; j++) mx = max(mx, smem_scores[j]);").unwrap();
    writeln!(out, "        smem_max[0] = mx;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // Softmax: exp and sum
    writeln!(out, "    for (uint j = tid; j < SEQ; j += {block_size}u) {{").unwrap();
    writeln!(out, "        smem_scores[j] = exp(smem_scores[j] - smem_max[0]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float s = 0.0f;").unwrap();
    writeln!(out, "        for (uint j = 0; j < SEQ; j++) s += smem_scores[j];").unwrap();
    writeln!(out, "        smem_sum[0] = s;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (uint j = tid; j < SEQ; j += {block_size}u) {{").unwrap();
    writeln!(out, "        smem_scores[j] /= smem_sum[0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    // Weighted sum: out[query_row, d] = sum_j scores[j] * V[j, d]
    writeln!(out, "    for (uint d = tid; d < DIM; d += {block_size}u) {{").unwrap();
    writeln!(out, "        float acc = 0.0f;").unwrap();
    writeln!(out, "        for (uint j = 0; j < SEQ; j++) {{").unwrap();
    writeln!(out, "            acc += smem_scores[j] * V[base + j * DIM + d];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        out[base + query_row * DIM + d] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

pub(crate) fn emit_rope_kernel_msl(
    out: &mut String,
    kernel_name: &str,
    head_dim: usize,
    theta: f64,
) {
    let half_dim = head_dim / 2;
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input [[buffer(0)]],").unwrap();
    writeln!(out, "    device float* output [[buffer(1)]],").unwrap();
    writeln!(out, "    constant uint& seq_len [[buffer(2)]],").unwrap();
    writeln!(out, "    constant uint& num_heads [[buffer(3)]],").unwrap();
    writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint tid [[thread_index_in_threadgroup]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const uint HEAD_DIM = {head_dim}u;").unwrap();
    writeln!(out, "    const uint HALF_DIM = {half_dim}u;").unwrap();
    writeln!(out, "    const float THETA = float({theta:e});").unwrap();
    writeln!(out, "    uint pos = gid.x;").unwrap();
    writeln!(out, "    uint head = gid.y;").unwrap();
    writeln!(out, "    uint base = (pos * num_heads + head) * HEAD_DIM;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = tid; i < HALF_DIM; i += 32) {{").unwrap();
    writeln!(out, "        float freq = 1.0f / pow(THETA, float(2 * i) / float(HEAD_DIM));").unwrap();
    writeln!(out, "        float angle = float(pos) * freq;").unwrap();
    writeln!(out, "        float cos_a = cos(angle);").unwrap();
    writeln!(out, "        float sin_a = sin(angle);").unwrap();
    writeln!(out, "        float x0 = input[base + i];").unwrap();
    writeln!(out, "        float x1 = input[base + i + HALF_DIM];").unwrap();
    writeln!(out, "        output[base + i] = x0 * cos_a - x1 * sin_a;").unwrap();
    writeln!(out, "        output[base + i + HALF_DIM] = x1 * cos_a + x0 * sin_a;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// GEMM bias kernels (emit_gemm_bias_simdgroup_msl, emit_gemm_bias_kernel_msl) moved to air_gemm.rs

// ── MachineCodeEmitter impl ─────────────────────────────────────────────────

impl MachineCodeEmitter for AirCodeGen {
    fn emit_plan(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
        _alloc: &BufferAllocation,
        _profile: &DeviceProfile,
        registry: Option<&ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        let dialect = MslDialect { gpu_family: self.gpu_family };
        let mut msl = String::new();
        dialect.emit_header(&mut msl);

        if plan.groups.is_empty() {
            return Ok(CodegenOutput {
                code: msl.into_bytes(),
                scratchpad_bytes: 0,
            });
        }

        gpu_emit_plan(&dialect, &mut msl, plan, graph, registry, None)?;

        Ok(CodegenOutput {
            code: msl.into_bytes(),
            scratchpad_bytes: 0,
        })
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
        Platform::Metal {
            gpu_family: self.gpu_family,
        }
    }

    /// Apple GPUs use a 32-wide SIMD group with a large register file per thread.
    /// We report 32 as a reasonable analogy (SIMD group width).
    fn num_simd_regs(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::OpKind;

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
    fn test_air_codegen_emit_plan_empty() {
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
            Ok(output) => {
                let msl = String::from_utf8(output.code).unwrap();
                assert!(msl.contains("#include <metal_stdlib>"));
                assert!(msl.contains("using namespace metal;"));
            }
            Err(e) => panic!("expected Ok for empty plan, got Err: {e}"),
        }
    }

    #[test]
    fn test_air_codegen_trace_op_to_msl() {
        let vars = vec!["a".to_string(), "b".to_string()];
        assert_eq!(trace_op_to_msl(&TraceOp::Add(0, 1), &vars), "(a + b)");
        assert_eq!(trace_op_to_msl(&TraceOp::Mul(0, 1), &vars), "(a * b)");
        assert_eq!(trace_op_to_msl(&TraceOp::Exp(0), &vars), "exp(a)");
        assert_eq!(trace_op_to_msl(&TraceOp::Rsqrt(0), &vars), "rsqrt(a)");
        assert_eq!(
            trace_op_to_msl(&TraceOp::Const(f64::NEG_INFINITY), &vars),
            "(-INFINITY)"
        );
    }


    #[test]
    fn test_air_emit_normlike_l2normalize() {
        use crate::compiler::codegen::gpu_ir::trace_emitter::{GpuDialect, MslDialect};
        let reduce = vec![
            TraceOp::Input(0),
            TraceOp::Mul(0, 0),
        ];
        let finalize = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Div(0, 1),
            TraceOp::Const(1e-5),
            TraceOp::Add(2, 3),
            TraceOp::Rsqrt(4),
        ];
        let transform = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Mul(0, 1),
        ];
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        dialect.emit_normlike_kernel(
            &mut out, "test_l2norm",
            &reduce, &finalize, &transform,
            false, false, Some(1e-5),
        );
        assert!(out.contains("kernel void test_l2norm("), "missing kernel sig:\n{out}");
        assert!(out.contains("rsqrt"), "missing rsqrt:\n{out}");
        assert!(!out.contains("weight"), "unexpected weight:\n{out}");
        assert!(!out.contains("bias"), "unexpected bias:\n{out}");
    }
    #[test]
    fn test_metal_backend_platform() {
        let backend = MetalBackend::new(9);
        let plat = backend.platform();
        assert!(matches!(plat, Platform::Metal { gpu_family: 9 }));
        assert_eq!(backend.num_simd_regs(), 32);
    }

    #[test]
    fn test_air_emit_gemm_kernel() {
        let mut out = String::new();
        emit_gemm_kernel_msl(&mut out, "test_gemm", 64, 64, 32);
        assert!(out.contains("kernel void test_gemm("));
        assert!(out.contains("device const float* A"));
        assert!(out.contains("device const float* B"));
        assert!(out.contains("device float* C"));
        assert!(out.contains("threadgroup float smA[256]"));
        assert!(out.contains("threadgroup_barrier"));
        assert!(out.contains("C[row * N + col] = acc"));
    }

    #[test]
    fn test_air_emit_mha_kernel() {
        let mut out = String::new();
        emit_mha_kernel_msl(&mut out, "test_mha", 4, 2, 8);
        assert!(out.contains("kernel void test_mha("));
        assert!(out.contains("device const float* Q"));
        assert!(out.contains("device const float* K"));
        assert!(out.contains("device const float* V"));
        assert!(out.contains("device float* out"));
        assert!(out.contains("smem_scores"));
        assert!(out.contains("exp(smem_scores"));
        assert!(out.contains("smem_sum[0]"));
    }

    #[test]
    fn test_air_emit_plan_gemm() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::graph::CompilerGraph;
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::inference::types::DType;
        use crate::dispatch::DeviceProfile;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![32, 16], DType::F32);
        let b = g.add_tensor("B", vec![16, 32], DType::F32);
        let c = g.add_tensor("C", vec![32, 32], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op_id = g.add_op(OpKind::Gemm { m: 32, n: 32, k: 16 }, vec![a, b], vec![c], "gemm");

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

        let mut cg = AirCodeGen::new(9);
        let result = cg.emit_plan(&plan, &g, &alloc, &profile, Some(&registry));
        let msl = match result {
            Ok(o) => String::from_utf8(o.code).unwrap(),
            Err(e) => panic!("emit_plan GEMM failed: {e}"),
        };
        assert!(msl.contains("kernel void group_0("), "missing kernel entry");
        // gpu_family 9 → simdgroup path uses half* inputs
        assert!(
            msl.contains("device const half* A") || msl.contains("device const float* A"),
            "missing A param"
        );
    }

    #[test]
    fn test_air_emit_plan_mha() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::graph::CompilerGraph;
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::inference::types::DType;
        use crate::dispatch::DeviceProfile;
        use std::collections::HashMap;

        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;

        let mut g = CompilerGraph::new();
        let q = g.add_tensor("Q", vec![seq_len, hidden], DType::F32);
        let k = g.add_tensor("K", vec![seq_len, hidden], DType::F32);
        let v = g.add_tensor("V", vec![seq_len, hidden], DType::F32);
        let out = g.add_tensor("out", vec![seq_len, hidden], DType::F32);
        g.inputs = vec![q, k, v];
        g.outputs = vec![out];
        let op_id = g.add_op(
            OpKind::MultiHeadAttention { seq_len, num_heads, head_dim },
            vec![q, k, v],
            vec![out],
            "mha",
        );

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

        let mut cg = AirCodeGen::new(9);
        let result = cg.emit_plan(&plan, &g, &alloc, &profile, Some(&registry));
        let msl = match result {
            Ok(o) => String::from_utf8(o.code).unwrap(),
            Err(e) => panic!("emit_plan MHA failed: {e}"),
        };
        assert!(msl.contains("kernel void group_0("), "missing kernel entry");
        assert!(msl.contains("smem_scores"), "missing shared memory");
    }

    #[test]
    fn test_air_emit_rope_kernel() {
        let mut out = String::new();
        emit_rope_kernel_msl(&mut out, "test_rope", 128, 10000.0);
        assert!(out.contains("kernel void test_rope("));
        assert!(out.contains("const uint HEAD_DIM = 128u"));
        assert!(out.contains("const uint HALF_DIM = 64u"));
        assert!(out.contains("cos(angle)"));
        assert!(out.contains("sin(angle)"));
        assert!(out.contains("x0 * cos_a - x1 * sin_a"));
        assert!(out.contains("x1 * cos_a + x0 * sin_a"));
    }

    #[test]
    fn test_air_emit_gemm_bias_kernel() {
        let mut out = String::new();
        emit_gemm_bias_kernel_msl(&mut out, "test_gemm_bias", 64, 64, 32);
        assert!(out.contains("kernel void test_gemm_bias("));
        assert!(out.contains("device const float* bias"));
        assert!(out.contains("acc + bias[col]"));
    }

    #[test]
    fn test_air_emit_gemm_bias_simdgroup() {
        let mut out = String::new();
        emit_gemm_bias_simdgroup_msl(&mut out, "test_gemm_bias_sg", 64, 64, 32);
        assert!(out.contains("kernel void test_gemm_bias_sg("));
        assert!(out.contains("simdgroup_float8x8"));
        assert!(out.contains("simdgroup_multiply_accumulate"));
        assert!(out.contains("device const float* bias"));
        assert!(out.contains("+ bias[out_col]"));
    }

    #[test]
    fn test_air_emit_dequantize_kernel() {
        use crate::compiler::codegen::gpu_ir::trace_emitter::{GpuDialect, MslDialect};
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        dialect.emit_dequantize_kernel(&mut out, "test_dequant", 4096, 32, 4);
        assert!(out.contains("kernel void test_dequant("), "missing kernel sig:\n{out}");
        assert!(out.contains("MASK = 15u"), "missing mask for 4-bit:\n{out}");
        assert!(out.contains("ELEMS_PER_U32 = 8u"), "missing elems_per_u32:\n{out}");
    }

    #[test]
    fn test_air_emit_plan_gemm_bias_family7() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::graph::CompilerGraph;
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::inference::types::DType;
        use crate::dispatch::DeviceProfile;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![32, 16], DType::F32);
        let b = g.add_tensor("B", vec![16, 32], DType::F32);
        let c = g.add_tensor("C", vec![32, 32], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op_id = g.add_op(OpKind::GemmBias { m: 32, n: 32, k: 16 }, vec![a, b], vec![c], "gemm_bias");

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

        // gpu_family 9 → simdgroup path with bias
        let mut cg = AirCodeGen::new(9);
        let result = cg.emit_plan(&plan, &g, &alloc, &profile, Some(&registry));
        let msl = match result {
            Ok(o) => String::from_utf8(o.code).unwrap(),
            Err(e) => panic!("emit_plan GemmBias failed: {e}"),
        };
        assert!(msl.contains("kernel void group_0("), "missing kernel entry");
        assert!(msl.contains("simdgroup_multiply_accumulate"), "missing simdgroup MAC");
        assert!(msl.contains("bias"), "missing bias");
    }

    #[test]
    fn test_air_emit_plan_gemm_family5_scalar() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::graph::CompilerGraph;
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::inference::types::DType;
        use crate::dispatch::DeviceProfile;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![32, 16], DType::F32);
        let b = g.add_tensor("B", vec![16, 32], DType::F32);
        let c = g.add_tensor("C", vec![32, 32], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op_id = g.add_op(OpKind::Gemm { m: 32, n: 32, k: 16 }, vec![a, b], vec![c], "gemm");

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

        // gpu_family 5 → scalar tiled path
        let mut cg = AirCodeGen::new(5);
        let result = cg.emit_plan(&plan, &g, &alloc, &profile, Some(&registry));
        let msl = match result {
            Ok(o) => String::from_utf8(o.code).unwrap(),
            Err(e) => panic!("emit_plan Gemm scalar failed: {e}"),
        };
        assert!(msl.contains("kernel void group_0("), "missing kernel entry");
        assert!(msl.contains("threadgroup float smA"), "missing shared memory tiling");
        assert!(!msl.contains("simdgroup"), "should not use simdgroup on family 5");
    }
}
