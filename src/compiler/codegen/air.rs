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
use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::{ComputePattern, TraceOp};
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

fn emit_elementwise_kernel_from_trace(
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
) {
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input [[buffer(0)]],").unwrap();
    writeln!(out, "    device float* output [[buffer(1)]],").unwrap();
    writeln!(out, "    uint tid [[thread_position_in_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    let bindings = vec!["input[tid]".to_string()];
    let result = emit_trace_body(out, body, 0, &bindings);
    writeln!(out, "    output[tid] = {result};").unwrap();
    writeln!(out, "}}\n").unwrap();
}

fn emit_binary_elementwise_kernel_from_trace(
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
) {
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input0 [[buffer(0)]],").unwrap();
    writeln!(out, "    device const float* input1 [[buffer(1)]],").unwrap();
    writeln!(out, "    device float* output [[buffer(2)]],").unwrap();
    writeln!(out, "    uint tid [[thread_position_in_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    let bindings = vec!["input0[tid]".to_string(), "input1[tid]".to_string()];
    let result = emit_trace_body(out, body, 0, &bindings);
    writeln!(out, "    output[tid] = {result};").unwrap();
    writeln!(out, "}}\n").unwrap();
}

fn emit_normlike_kernel(
    out: &mut String,
    kernel_name: &str,
    reduce: &[TraceOp],
    finalize: &[TraceOp],
    transform: &[TraceOp],
    has_weight: bool,
    has_bias: bool,
    eps_override: Option<f32>,
) {
    let tg_size: usize = 256;
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input [[buffer(0)]],").unwrap();
    let mut buf_idx: usize = 1;
    if has_weight {
        writeln!(out, "    device const float* weight [[buffer({buf_idx})]],").unwrap();
        buf_idx += 1;
    }
    if has_bias {
        writeln!(out, "    device const float* bias [[buffer({buf_idx})]],").unwrap();
        buf_idx += 1;
    }
    writeln!(out, "    device float* output [[buffer({buf_idx})]],").unwrap();
    buf_idx += 1;
    writeln!(out, "    constant uint& N [[buffer({buf_idx})]],").unwrap();
    writeln!(out, "    uint lid [[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint gid [[threadgroup_position_in_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    threadgroup float shared[{tg_size}];").unwrap();
    writeln!(out).unwrap();

    // Phase 1: partial reduction
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    for (uint i = lid; i < N; i += {tg_size}) {{").unwrap();
    let reduce_bindings = vec!["input[gid * N + i]".to_string()];
    let reduce_result = emit_trace_body(out, reduce, 1, &reduce_bindings);
    writeln!(out, "        acc += {reduce_result};").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    shared[lid] = acc;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();

    // Tree reduction in shared memory
    writeln!(out, "    for (uint s = {tg_size} / 2; s > 0; s >>= 1) {{").unwrap();
    writeln!(out, "        if (lid < s) {{ shared[lid] += shared[lid + s]; }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Phase 2: finalize
    writeln!(out, "    float sum_val = shared[0];").unwrap();
    let finalize_bindings = vec!["sum_val".to_string(), "float(N)".to_string()];
    let mut patched_finalize: Vec<TraceOp>;
    let finalize_ops: &[TraceOp] = if let Some(eps) = eps_override {
        patched_finalize = finalize.to_vec();
        for op in &mut patched_finalize {
            if let TraceOp::Const(v) = op {
                if (*v - 1e-5f64).abs() < 1e-10 || (*v - 1e-12f64).abs() < 1e-15 {
                    *v = eps as f64;
                }
            }
        }
        &patched_finalize[..]
    } else {
        finalize
    };
    let scale_var = emit_trace_body(out, finalize_ops, 2, &finalize_bindings);
    writeln!(out).unwrap();

    // Phase 3: per-element transform
    writeln!(out, "    for (uint i = lid; i < N; i += {tg_size}) {{").unwrap();
    let mut xform_bindings = vec![
        "input[gid * N + i]".to_string(),
        scale_var.clone(),
    ];
    if has_weight {
        xform_bindings.push("weight[i]".to_string());
    }
    if has_bias {
        xform_bindings.push("bias[i]".to_string());
    }
    let xform_result = emit_trace_body(out, transform, 3, &xform_bindings);
    writeln!(out, "        output[gid * N + i] = {xform_result};").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

fn emit_softmax_kernel(out: &mut String, kernel_name: &str) {
    let tg_size: usize = 256;
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input [[buffer(0)]],").unwrap();
    writeln!(out, "    device float* output [[buffer(1)]],").unwrap();
    writeln!(out, "    constant uint& N [[buffer(2)]],").unwrap();
    writeln!(out, "    uint lid [[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint gid [[threadgroup_position_in_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    threadgroup float shared[{tg_size}];").unwrap();
    writeln!(out).unwrap();

    // Pass 1: find max
    writeln!(out, "    float max_val = -INFINITY;").unwrap();
    writeln!(out, "    for (uint i = lid; i < N; i += {tg_size}) {{").unwrap();
    writeln!(out, "        max_val = max(max_val, input[gid * N + i]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    shared[lid] = max_val;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (uint s = {tg_size} / 2; s > 0; s >>= 1) {{").unwrap();
    writeln!(out, "        if (lid < s) {{ shared[lid] = max(shared[lid], shared[lid + s]); }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float row_max = shared[0];").unwrap();
    writeln!(out).unwrap();

    // Pass 2: exp and sum
    writeln!(out, "    float sum_exp = 0.0f;").unwrap();
    writeln!(out, "    for (uint i = lid; i < N; i += {tg_size}) {{").unwrap();
    writeln!(out, "        sum_exp += exp(input[gid * N + i] - row_max);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    shared[lid] = sum_exp;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (uint s = {tg_size} / 2; s > 0; s >>= 1) {{").unwrap();
    writeln!(out, "        if (lid < s) {{ shared[lid] += shared[lid + s]; }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float inv_sum = 1.0f / shared[0];").unwrap();
    writeln!(out).unwrap();

    // Pass 3: normalize
    writeln!(out, "    for (uint i = lid; i < N; i += {tg_size}) {{").unwrap();
    writeln!(out, "        output[gid * N + i] = exp(input[gid * N + i] - row_max) * inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

fn emit_meanpool_kernel(
    out: &mut String,
    kernel_name: &str,
    seq_len: usize,
    hidden: usize,
) {
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* input [[buffer(0)]],").unwrap();
    writeln!(out, "    device float* output [[buffer(1)]],").unwrap();
    writeln!(out, "    uint tid [[thread_position_in_grid]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    uint h = tid;").unwrap();
    writeln!(out, "    if (h >= {hidden}u) return;").unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    for (uint s = 0; s < {seq_len}u; s++) {{").unwrap();
    writeln!(out, "        acc += input[s * {hidden}u + h];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    output[h] = acc / float({seq_len}u);").unwrap();
    writeln!(out, "}}\n").unwrap();
}

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
        let mut msl = String::new();
        msl.push_str("#include <metal_stdlib>\n");
        msl.push_str("using namespace metal;\n\n");

        if plan.groups.is_empty() {
            return Ok(CodegenOutput {
                code: msl.into_bytes(),
                scratchpad_bytes: 0,
            });
        }

        for group in &plan.groups {
            let anchor_op = graph.op(group.anchor).ok_or_else(|| {
                format!("AirCodeGen: anchor op {:?} not found in graph", group.anchor)
            })?;

            let kernel_name = format!("group_{}", group.id);
            let op_kind = &anchor_op.kind;

            // Reshape/Transpose are metadata-only — NOP on GPU
            if matches!(op_kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
                continue;
            }

            let registry = registry.ok_or_else(|| {
                format!("AirCodeGen: emit_plan requires a ScalarOpRegistry for {:?}", op_kind)
            })?;

            let key = ScalarOpRegistry::key_from_op_kind(op_kind);
            let trace = registry.get_trace(&key).ok_or_else(|| {
                format!("AirCodeGen: no OpTrace for {:?}", op_kind)
            })?;

            match &trace.pattern {
                ComputePattern::Elementwise { body } => {
                    emit_elementwise_kernel_from_trace(&mut msl, &kernel_name, body);
                }
                ComputePattern::BinaryElementwise { body } => {
                    emit_binary_elementwise_kernel_from_trace(&mut msl, &kernel_name, body);
                }
                ComputePattern::NormLike { reduce, finalize, transform } => {
                    let eps_override = match op_kind {
                        OpKind::RmsNorm { eps } => Some(*eps),
                        OpKind::LayerNorm { eps } => Some(*eps),
                        _ => None,
                    };
                    let has_weight = matches!(
                        op_kind,
                        OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }
                    );
                    let has_bias = matches!(op_kind, OpKind::LayerNorm { .. });
                    emit_normlike_kernel(
                        &mut msl,
                        &kernel_name,
                        reduce,
                        finalize,
                        transform,
                        has_weight,
                        has_bias,
                        eps_override,
                    );
                }
                ComputePattern::Reduction { .. } => {
                    match op_kind {
                        OpKind::Softmax => {
                            emit_softmax_kernel(&mut msl, &kernel_name);
                        }
                        OpKind::MeanPool { seq_len, hidden } => {
                            emit_meanpool_kernel(
                                &mut msl, &kernel_name, *seq_len, *hidden,
                            );
                        }
                        _ => {
                            return Err(format!(
                                "AirCodeGen: unsupported Reduction op {:?}",
                                op_kind
                            ));
                        }
                    }
                }
                ComputePattern::Gemm => {
                    return Err(format!(
                        "AirCodeGen: GEMM codegen not yet implemented for Metal (op {:?})",
                        op_kind
                    ));
                }
                ComputePattern::Injective { body, .. } => {
                    if body.is_empty() {
                        continue;
                    }
                    return Err(format!(
                        "AirCodeGen: non-trivial Injective codegen not yet implemented (op {:?})",
                        op_kind
                    ));
                }
                ComputePattern::QuantDecode { .. } => {
                    return Err(format!(
                        "AirCodeGen: QuantDecode codegen not yet implemented (op {:?})",
                        op_kind
                    ));
                }
            }
        }

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
        let mut out = String::new();
        emit_normlike_kernel(
            &mut out, "test_l2norm",
            &reduce, &finalize, &transform,
            false, false, Some(1e-5),
        );
        assert!(out.contains("kernel void test_l2norm("));
        assert!(out.contains("rsqrt"));
        assert!(!out.contains("weight"));
        assert!(!out.contains("bias"));
    }
    #[test]
    fn test_metal_backend_platform() {
        let backend = MetalBackend::new(9);
        let plat = backend.platform();
        assert!(matches!(plat, Platform::Metal { gpu_family: 9 }));
        assert_eq!(backend.num_simd_regs(), 32);
    }
}
