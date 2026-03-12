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
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::{ComputePattern, TraceOp};
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

    /// Emit the PTX module header (, , ).
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

    /// Emit a simple elementwise kernel that applies  to each element.
    ///
    /// Supported ops: "add", "mul", "silu", "relu", "neg".
    ///
    /// Generated kernel signature:
    /// 
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

// ── PTX TraceOp → instruction codegen ───────────────────────────────────────

/// Emit a PTX instruction sequence for a single `TraceOp`.
/// `vars` contains the PTX register name for each prior SSA op.
/// Returns the register name holding this op's result.
fn trace_op_to_ptx(
    out: &mut String,
    op: &TraceOp,
    idx: usize,
    vars: &[String],
    base: usize,
) -> String {
    let reg = format!("%t{}_{}", base, idx);
    match op {
        TraceOp::Input(i) => {
            // Input bindings are handled by the caller — just return a placeholder.
            // The caller replaces Input ops with pre-bound register names.
            format!("%input{i}")
        }
        TraceOp::Const(val) => {
            let bits = (*val as f32).to_bits();
            writeln!(out, "    mov.f32 {reg}, 0f{bits:08X};").unwrap();
            reg
        }
        TraceOp::Add(a, b) => {
            writeln!(out, "    add.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            reg
        }
        TraceOp::Sub(a, b) => {
            writeln!(out, "    sub.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            reg
        }
        TraceOp::Mul(a, b) => {
            writeln!(out, "    mul.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            reg
        }
        TraceOp::Div(a, b) => {
            writeln!(out, "    div.approx.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            reg
        }
        TraceOp::Fma(a, b, c) => {
            writeln!(out, "    fma.rn.f32 {reg}, {}, {}, {};", vars[*a as usize], vars[*b as usize], vars[*c as usize]).unwrap();
            reg
        }
        TraceOp::Neg(a) => {
            writeln!(out, "    neg.f32 {reg}, {};", vars[*a as usize]).unwrap();
            reg
        }
        TraceOp::Abs(a) => {
            writeln!(out, "    abs.f32 {reg}, {};", vars[*a as usize]).unwrap();
            reg
        }
        TraceOp::Exp(a) => {
            // exp(x) = exp2(x * log2(e)), log2(e) ≈ 1.4426950408889634 = 0x3FB8AA3B
            writeln!(out, "    mul.f32 {reg}, {}, 0f3FB8AA3B;", vars[*a as usize]).unwrap();
            writeln!(out, "    ex2.approx.f32 {reg}, {reg};").unwrap();
            reg
        }
        TraceOp::Sqrt(a) => {
            writeln!(out, "    sqrt.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            reg
        }
        TraceOp::Rsqrt(a) => {
            writeln!(out, "    rsqrt.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            reg
        }
        TraceOp::Tanh(a) => {
            writeln!(out, "    tanh.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            reg
        }
        TraceOp::Recip(a) => {
            writeln!(out, "    rcp.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            reg
        }
        TraceOp::Log(a) => {
            // ln(x) = log2(x) / log2(e) = log2(x) * ln(2)
            // ln(2) ≈ 0.6931471805599453 = 0x3F317218
            writeln!(out, "    lg2.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            writeln!(out, "    mul.f32 {reg}, {reg}, 0f3F317218;").unwrap();
            reg
        }
        TraceOp::Max(a, b) => {
            writeln!(out, "    max.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            reg
        }
        TraceOp::Min(a, b) => {
            writeln!(out, "    min.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            reg
        }
    }
}

/// Emit a sequence of `TraceOp`s as PTX instructions.
/// `input_bindings` maps Input(i) → register name.
/// Returns the register name of the last (result) op.
fn emit_trace_body_ptx(
    out: &mut String,
    ops: &[TraceOp],
    base: usize,
    input_bindings: &[String],
) -> String {
    let mut vars: Vec<String> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        let var = if let TraceOp::Input(idx) = op {
            input_bindings
                .get(*idx as usize)
                .cloned()
                .unwrap_or_else(|| format!("%input{idx}"))
        } else {
            trace_op_to_ptx(out, op, i, &vars, base)
        };
        vars.push(var);
    }
    vars.last()
        .cloned()
        .unwrap_or_else(|| "%zero".to_string())
}

// ── Kernel generators per ComputePattern ────────────────────────────────────

/// Count the maximum register index needed for a trace body.
fn max_regs_needed(ops: &[TraceOp], base: usize) -> usize {
    // Each non-Input op gets a register t{base}_{i}
    ops.len() + 8 // padding for temporaries
}

fn emit_elementwise_kernel_ptx(
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
) {
    let nregs = max_regs_needed(body, 0);
    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_n").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<4>;").unwrap();
    writeln!(out, "    .reg .u32 %r<4>;").unwrap();
    writeln!(out, "    .reg .f32 %t0_<{nregs}>;").unwrap();
    writeln!(out, "    .reg .f32 %f<4>;").unwrap();
    writeln!(out, "    .reg .pred %p<2>;").unwrap();
    writeln!(out).unwrap();

    // Load params + compute tid
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_output];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_n];").unwrap();
    writeln!(out, "    mov.u32 %r1, %ctaid.x;").unwrap();
    writeln!(out, "    mov.u32 %r2, %ntid.x;").unwrap();
    writeln!(out, "    mad.lo.u32 %r1, %r1, %r2, %tid.x;").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r1, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_DONE;").unwrap();
    writeln!(out).unwrap();

    // Load input element
    writeln!(out, "    mul.wide.u32 %rd2, %r1, 4;").unwrap();
    writeln!(out, "    add.u64 %rd2, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f0, [%rd2];").unwrap();
    writeln!(out).unwrap();

    // Emit trace body
    let bindings = vec!["%f0".to_string()];
    let result = emit_trace_body_ptx(out, body, 0, &bindings);
    writeln!(out).unwrap();

    // Store result
    writeln!(out, "    mul.wide.u32 %rd3, %r1, 4;").unwrap();
    writeln!(out, "    add.u64 %rd3, %rd1, %rd3;").unwrap();
    writeln!(out, "    st.global.f32 [%rd3], {result};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "{kernel_name}_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_binary_elementwise_kernel_ptx(
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
) {
    let nregs = max_regs_needed(body, 0);
    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input0,").unwrap();
    writeln!(out, "    .param .u64 param_input1,").unwrap();
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_n").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<6>;").unwrap();
    writeln!(out, "    .reg .u32 %r<4>;").unwrap();
    writeln!(out, "    .reg .f32 %t0_<{nregs}>;").unwrap();
    writeln!(out, "    .reg .f32 %f<4>;").unwrap();
    writeln!(out, "    .reg .pred %p<2>;").unwrap();
    writeln!(out).unwrap();

    // Load params + compute tid
    writeln!(out, "    ld.param.u64 %rd0, [param_input0];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_input1];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_output];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_n];").unwrap();
    writeln!(out, "    mov.u32 %r1, %ctaid.x;").unwrap();
    writeln!(out, "    mov.u32 %r2, %ntid.x;").unwrap();
    writeln!(out, "    mad.lo.u32 %r1, %r1, %r2, %tid.x;").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r1, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_DONE;").unwrap();
    writeln!(out).unwrap();

    // Load both input elements
    writeln!(out, "    mul.wide.u32 %rd3, %r1, 4;").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd0, %rd3;").unwrap();
    writeln!(out, "    ld.global.f32 %f0, [%rd4];").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd1, %rd3;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd4];").unwrap();
    writeln!(out).unwrap();

    // Emit trace body
    let bindings = vec!["%f0".to_string(), "%f1".to_string()];
    let result = emit_trace_body_ptx(out, body, 0, &bindings);
    writeln!(out).unwrap();

    // Store result
    writeln!(out, "    add.u64 %rd5, %rd2, %rd3;").unwrap();
    writeln!(out, "    st.global.f32 [%rd5], {result};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "{kernel_name}_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_normlike_kernel_ptx(
    out: &mut String,
    kernel_name: &str,
    reduce: &[TraceOp],
    finalize: &[TraceOp],
    transform: &[TraceOp],
    has_weight: bool,
    has_bias: bool,
    eps_override: Option<f32>,
) {
    let block_size: u32 = 256;
    let nregs_r = max_regs_needed(reduce, 1);
    let nregs_f = max_regs_needed(finalize, 2);
    let nregs_t = max_regs_needed(transform, 3);
    let _max_nregs = nregs_r.max(nregs_f).max(nregs_t);

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    if has_weight {
        writeln!(out, "    .param .u64 param_weight,").unwrap();
    }
    if has_bias {
        writeln!(out, "    .param .u64 param_bias,").unwrap();
    }
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_N").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();

    // Shared memory for reduction
    writeln!(out, "    .shared .f32 smem[{block_size}];").unwrap();
    writeln!(out).unwrap();

    // Registers
    writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
    writeln!(out, "    .reg .u32 %r<16>;").unwrap();
    writeln!(out, "    .reg .f32 %t1_<{nregs_r}>;").unwrap();
    writeln!(out, "    .reg .f32 %t2_<{nregs_f}>;").unwrap();
    writeln!(out, "    .reg .f32 %t3_<{nregs_t}>;").unwrap();
    writeln!(out, "    .reg .f32 %f<16>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    if has_weight {
        writeln!(out, "    ld.param.u64 %rd1, [param_weight];").unwrap();
    }
    if has_bias {
        writeln!(out, "    ld.param.u64 %rd2, [param_bias];").unwrap();
    }
    writeln!(out, "    ld.param.u64 %rd3, [param_output];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_N];").unwrap();
    writeln!(out).unwrap();

    // tid = threadIdx.x, gid = blockIdx.x
    writeln!(out, "    mov.u32 %r1, %tid.x;").unwrap();   // lid
    writeln!(out, "    mov.u32 %r2, %ctaid.x;").unwrap();  // gid
    writeln!(out).unwrap();

    // row_base = gid * N
    writeln!(out, "    mul.lo.u32 %r3, %r2, %r0;").unwrap();
    writeln!(out).unwrap();

    // ── Phase 1: partial reduction ──
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap(); // acc = 0.0
    writeln!(out, "    mov.u32 %r4, %r1;").unwrap();        // i = lid
    writeln!(out, "{kernel_name}_REDUCE_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r4, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_REDUCE_DONE;").unwrap();

    // Load input[row_base + i]
    writeln!(out, "    add.u32 %r5, %r3, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd4, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd0, %rd4;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd5];").unwrap();

    // Apply reduce body (e.g. x*x for RmsNorm)
    let reduce_bindings = vec!["%f1".to_string()];
    let reduce_result = emit_trace_body_ptx(out, reduce, 1, &reduce_bindings);
    writeln!(out, "    add.f32 %f0, %f0, {reduce_result};").unwrap();

    writeln!(out, "    add.u32 %r4, %r4, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_REDUCE_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_REDUCE_DONE:").unwrap();
    writeln!(out).unwrap();

    // Store partial sum to shared memory
    writeln!(out, "    mul.wide.u32 %rd6, %r1, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd7, smem;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd7, %rd6;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd6], %f0;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out).unwrap();

    // Tree reduction in shared memory
    let mut stride = block_size / 2;
    while stride > 0 {
        writeln!(out, "    setp.lt.u32 %p1, %r1, {stride};").unwrap();
        writeln!(out, "    @!%p1 bra {kernel_name}_SKIP_{stride};").unwrap();
        writeln!(out, "    add.u32 %r6, %r1, {stride};").unwrap();
        writeln!(out, "    mul.wide.u32 %rd8, %r6, 4;").unwrap();
        writeln!(out, "    add.u64 %rd8, %rd7, %rd8;").unwrap();
        writeln!(out, "    ld.shared.f32 %f2, [%rd8];").unwrap();
        writeln!(out, "    ld.shared.f32 %f3, [%rd6];").unwrap();
        writeln!(out, "    add.f32 %f3, %f3, %f2;").unwrap();
        writeln!(out, "    st.shared.f32 [%rd6], %f3;").unwrap();
        writeln!(out, "{kernel_name}_SKIP_{stride}:").unwrap();
        writeln!(out, "    bar.sync 0;").unwrap();
        stride /= 2;
    }
    writeln!(out).unwrap();

    // ── Phase 2: finalize (compute scale from reduction result) ──
    writeln!(out, "    ld.shared.f32 %f4, [smem];").unwrap(); // sum_val
    // Convert N to float for finalize
    writeln!(out, "    cvt.rn.f32.u32 %f5, %r0;").unwrap();  // float(N)

    let finalize_bindings = vec!["%f4".to_string(), "%f5".to_string()];
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
    let scale_var = emit_trace_body_ptx(out, finalize_ops, 2, &finalize_bindings);
    writeln!(out).unwrap();

    // ── Phase 3: per-element transform ──
    writeln!(out, "    mov.u32 %r4, %r1;").unwrap(); // i = lid
    writeln!(out, "{kernel_name}_XFORM_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p2, %r4, %r0;").unwrap();
    writeln!(out, "    @%p2 bra {kernel_name}_XFORM_DONE;").unwrap();

    // Load input[row_base + i]
    writeln!(out, "    add.u32 %r5, %r3, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd9, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd0, %rd9;").unwrap();
    writeln!(out, "    ld.global.f32 %f6, [%rd10];").unwrap();

    let mut xform_bindings = vec![
        "%f6".to_string(),
        scale_var.clone(),
    ];

    if has_weight {
        // Load weight[i]
        writeln!(out, "    mul.wide.u32 %rd11, %r4, 4;").unwrap();
        writeln!(out, "    add.u64 %rd12, %rd1, %rd11;").unwrap();
        writeln!(out, "    ld.global.f32 %f7, [%rd12];").unwrap();
        xform_bindings.push("%f7".to_string());
    }
    if has_bias {
        // Load bias[i]
        writeln!(out, "    mul.wide.u32 %rd13, %r4, 4;").unwrap();
        writeln!(out, "    add.u64 %rd14, %rd2, %rd13;").unwrap();
        writeln!(out, "    ld.global.f32 %f8, [%rd14];").unwrap();
        xform_bindings.push("%f8".to_string());
    }

    let xform_result = emit_trace_body_ptx(out, transform, 3, &xform_bindings);

    // Store output[row_base + i]
    writeln!(out, "    add.u64 %rd15, %rd3, %rd9;").unwrap();
    writeln!(out, "    st.global.f32 [%rd15], {xform_result};").unwrap();

    writeln!(out, "    add.u32 %r4, %r4, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_XFORM_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_XFORM_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_softmax_kernel_ptx(out: &mut String, kernel_name: &str) {
    let block_size: u32 = 256;
    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_N").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .shared .f32 smem[{block_size}];").unwrap();
    writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
    writeln!(out, "    .reg .u32 %r<16>;").unwrap();
    writeln!(out, "    .reg .f32 %f<16>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_output];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_N];").unwrap();
    writeln!(out, "    mov.u32 %r1, %tid.x;").unwrap();
    writeln!(out, "    mov.u32 %r2, %ctaid.x;").unwrap();
    writeln!(out, "    mul.lo.u32 %r3, %r2, %r0;").unwrap();
    writeln!(out).unwrap();

    // Pass 1: find max
    writeln!(out, "    mov.f32 %f0, 0fFF800000;").unwrap(); // -INF
    writeln!(out, "    mov.u32 %r4, %r1;").unwrap();
    writeln!(out, "{kernel_name}_MAX_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r4, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_MAX_DONE;").unwrap();
    writeln!(out, "    add.u32 %r5, %r3, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd2, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd3, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd3];").unwrap();
    writeln!(out, "    max.f32 %f0, %f0, %f1;").unwrap();
    writeln!(out, "    add.u32 %r4, %r4, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_MAX_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_MAX_DONE:").unwrap();

    // Reduce max in shared memory
    writeln!(out, "    mov.u64 %rd4, smem;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd5, %r1, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd4, %rd5;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd5], %f0;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();

    let mut stride = block_size / 2;
    while stride > 0 {
        writeln!(out, "    setp.lt.u32 %p1, %r1, {stride};").unwrap();
        writeln!(out, "    @!%p1 bra {kernel_name}_MAXR_{stride};").unwrap();
        writeln!(out, "    add.u32 %r6, %r1, {stride};").unwrap();
        writeln!(out, "    mul.wide.u32 %rd6, %r6, 4;").unwrap();
        writeln!(out, "    add.u64 %rd6, %rd4, %rd6;").unwrap();
        writeln!(out, "    ld.shared.f32 %f2, [%rd6];").unwrap();
        writeln!(out, "    ld.shared.f32 %f3, [%rd5];").unwrap();
        writeln!(out, "    max.f32 %f3, %f3, %f2;").unwrap();
        writeln!(out, "    st.shared.f32 [%rd5], %f3;").unwrap();
        writeln!(out, "{kernel_name}_MAXR_{stride}:").unwrap();
        writeln!(out, "    bar.sync 0;").unwrap();
        stride /= 2;
    }
    writeln!(out, "    ld.shared.f32 %f4, [smem];").unwrap(); // row_max
    writeln!(out).unwrap();

    // Pass 2: exp(x - max) and sum
    writeln!(out, "    mov.f32 %f5, 0f00000000;").unwrap(); // sum = 0
    writeln!(out, "    mov.u32 %r4, %r1;").unwrap();
    writeln!(out, "{kernel_name}_SUM_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r4, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_SUM_DONE;").unwrap();
    writeln!(out, "    add.u32 %r5, %r3, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd2, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd3, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd3];").unwrap();
    writeln!(out, "    sub.f32 %f1, %f1, %f4;").unwrap();
    // exp via exp2
    writeln!(out, "    mul.f32 %f1, %f1, 0f3FB8AA3B;").unwrap();
    writeln!(out, "    ex2.approx.f32 %f1, %f1;").unwrap();
    writeln!(out, "    add.f32 %f5, %f5, %f1;").unwrap();
    writeln!(out, "    add.u32 %r4, %r4, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_SUM_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_SUM_DONE:").unwrap();

    // Reduce sum in shared memory
    writeln!(out, "    st.shared.f32 [%rd5], %f5;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    stride = block_size / 2;
    while stride > 0 {
        writeln!(out, "    setp.lt.u32 %p1, %r1, {stride};").unwrap();
        writeln!(out, "    @!%p1 bra {kernel_name}_SUMR_{stride};").unwrap();
        writeln!(out, "    add.u32 %r6, %r1, {stride};").unwrap();
        writeln!(out, "    mul.wide.u32 %rd6, %r6, 4;").unwrap();
        writeln!(out, "    add.u64 %rd6, %rd4, %rd6;").unwrap();
        writeln!(out, "    ld.shared.f32 %f2, [%rd6];").unwrap();
        writeln!(out, "    ld.shared.f32 %f3, [%rd5];").unwrap();
        writeln!(out, "    add.f32 %f3, %f3, %f2;").unwrap();
        writeln!(out, "    st.shared.f32 [%rd5], %f3;").unwrap();
        writeln!(out, "{kernel_name}_SUMR_{stride}:").unwrap();
        writeln!(out, "    bar.sync 0;").unwrap();
        stride /= 2;
    }
    writeln!(out, "    ld.shared.f32 %f6, [smem];").unwrap(); // sum
    writeln!(out, "    rcp.approx.f32 %f6, %f6;").unwrap();   // inv_sum
    writeln!(out).unwrap();

    // Pass 3: normalize
    writeln!(out, "    mov.u32 %r4, %r1;").unwrap();
    writeln!(out, "{kernel_name}_NORM_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r4, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_NORM_DONE;").unwrap();
    writeln!(out, "    add.u32 %r5, %r3, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd2, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd3, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd3];").unwrap();
    writeln!(out, "    sub.f32 %f1, %f1, %f4;").unwrap();
    writeln!(out, "    mul.f32 %f1, %f1, 0f3FB8AA3B;").unwrap();
    writeln!(out, "    ex2.approx.f32 %f1, %f1;").unwrap();
    writeln!(out, "    mul.f32 %f1, %f1, %f6;").unwrap();
    writeln!(out, "    add.u64 %rd7, %rd1, %rd2;").unwrap();
    writeln!(out, "    st.global.f32 [%rd7], %f1;").unwrap();
    writeln!(out, "    add.u32 %r4, %r4, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_NORM_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_NORM_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_meanpool_kernel_ptx(
    out: &mut String,
    kernel_name: &str,
    seq_len: usize,
    hidden: usize,
) {
    let block_size: u32 = 256;
    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    writeln!(out, "    .param .u64 param_output").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<8>;").unwrap();
    writeln!(out, "    .reg .u32 %r<8>;").unwrap();
    writeln!(out, "    .reg .f32 %f<4>;").unwrap();
    writeln!(out, "    .reg .pred %p<2>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_output];").unwrap();
    writeln!(out).unwrap();

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    writeln!(out, "    mov.u32 %r0, %ctaid.x;").unwrap();
    writeln!(out, "    mov.u32 %r1, %tid.x;").unwrap();
    writeln!(out, "    mad.lo.u32 %r2, %r0, {block_size}, %r1;").unwrap();
    writeln!(out).unwrap();

    // Guard: if tid >= hidden {{ return; }}
    writeln!(out, "    setp.ge.u32 %p0, %r2, {hidden};").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_DONE;").unwrap();
    writeln!(out).unwrap();

    // acc = 0.0
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
    writeln!(out, "    mov.u32 %r3, 0;").unwrap(); // s = 0
    writeln!(out, "{kernel_name}_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p1, %r3, {seq_len};").unwrap();
    writeln!(out, "    @%p1 bra {kernel_name}_LOOP_DONE;").unwrap();

    // offset = s * hidden + tid
    writeln!(out, "    mad.lo.u32 %r4, %r3, {hidden}, %r2;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd2, %r4, 4;").unwrap();
    writeln!(out, "    add.u64 %rd3, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd3];").unwrap();
    writeln!(out, "    add.f32 %f0, %f0, %f1;").unwrap();

    writeln!(out, "    add.u32 %r3, %r3, 1;").unwrap();
    writeln!(out, "    bra {kernel_name}_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_LOOP_DONE:").unwrap();
    writeln!(out).unwrap();

    // output[tid] = acc * (1.0 / seq_len) using rcp.approx
    writeln!(out, "    mov.u32 %r5, {seq_len};").unwrap();
    writeln!(out, "    cvt.rn.f32.u32 %f2, %r5;").unwrap();
    writeln!(out, "    rcp.approx.f32 %f2, %f2;").unwrap();
    writeln!(out, "    mul.f32 %f0, %f0, %f2;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd4, %r2, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd1, %rd4;").unwrap();
    writeln!(out, "    st.global.f32 [%rd5], %f0;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "{kernel_name}_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}


fn emit_gemm_kernel_ptx(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
    // 16x16 tiled GEMM: C[m,n] = A[m,k] * B[k,n]
    // Grid: (ceil(n/16), ceil(m/16))  Block: (16, 16)
    let tile = 16usize;
    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_A,").unwrap();
    writeln!(out, "    .param .u64 param_B,").unwrap();
    writeln!(out, "    .param .u64 param_C,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_N,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    .shared .f32 smA[{t2}];", t2 = tile * tile).unwrap();
    writeln!(out, "    .shared .f32 smB[{t2}];", t2 = tile * tile).unwrap();
    writeln!(out, "    .reg .u64 %rd<12>;").unwrap();
    writeln!(out, "    .reg .u32 %r<20>;").unwrap();
    writeln!(out, "    .reg .f32 %f<8>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();
    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_A];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_B];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_C];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_K];").unwrap();
    // Thread/block indices
    writeln!(out, "    mov.u32 %r3, %tid.x;").unwrap();   // tx
    writeln!(out, "    mov.u32 %r4, %tid.y;").unwrap();   // ty
    writeln!(out, "    mov.u32 %r5, %ctaid.x;").unwrap(); // bx
    writeln!(out, "    mov.u32 %r6, %ctaid.y;").unwrap(); // by
    // row = by*16 + ty,  col = bx*16 + tx
    writeln!(out, "    mad.lo.u32 %r7, %r6, {tile}, %r4;").unwrap(); // row
    writeln!(out, "    mad.lo.u32 %r8, %r5, {tile}, %r3;").unwrap(); // col
    // acc = 0
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
    // smem base pointers
    writeln!(out, "    mov.u64 %rd3, smA;").unwrap();
    writeln!(out, "    mov.u64 %rd4, smB;").unwrap();
    // tile loop: t = 0..ceil(K/16)
    writeln!(out, "    mov.u32 %r9, 0;").unwrap(); // t
    writeln!(out, "{kernel_name}_TILE_LOOP:").unwrap();
    writeln!(out, "    // check t < ceil(K/16)").unwrap();
    writeln!(out, "    mul.lo.u32 %r10, %r9, {tile};").unwrap(); // t*16
    writeln!(out, "    setp.ge.u32 %p0, %r10, %r2;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_TILE_DONE;").unwrap();
    // Load A tile: A[row, t*16+tx]
    writeln!(out, "    add.u32 %r11, %r10, %r3;").unwrap(); // t*16+tx
    writeln!(out, "    setp.lt.u32 %p1, %r7, %r0;").unwrap();
    writeln!(out, "    setp.lt.u32 %p2, %r11, %r2;").unwrap();
    writeln!(out, "    and.pred %p1, %p1, %p2;").unwrap();
    writeln!(out, "    mov.f32 %f1, 0f00000000;").unwrap();
    writeln!(out, "    @%p1 {{").unwrap();
    writeln!(out, "        mad.lo.u32 %r12, %r7, %r2, %r11;").unwrap();
    writeln!(out, "        mul.wide.u32 %rd5, %r12, 4;").unwrap();
    writeln!(out, "        add.u64 %rd5, %rd0, %rd5;").unwrap();
    writeln!(out, "        ld.global.f32 %f1, [%rd5];").unwrap();
    writeln!(out, "    }}").unwrap();
    // store to smA[ty*16+tx]
    writeln!(out, "    mad.lo.u32 %r13, %r4, {tile}, %r3;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd6, %r13, 4;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd3, %rd6;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd6], %f1;").unwrap();
    // Load B tile: B[t*16+ty, col]
    writeln!(out, "    add.u32 %r11, %r10, %r4;").unwrap(); // t*16+ty
    writeln!(out, "    setp.lt.u32 %p1, %r11, %r2;").unwrap();
    writeln!(out, "    setp.lt.u32 %p2, %r8, %r1;").unwrap();
    writeln!(out, "    and.pred %p1, %p1, %p2;").unwrap();
    writeln!(out, "    mov.f32 %f2, 0f00000000;").unwrap();
    writeln!(out, "    @%p1 {{").unwrap();
    writeln!(out, "        mad.lo.u32 %r12, %r11, %r1, %r8;").unwrap();
    writeln!(out, "        mul.wide.u32 %rd5, %r12, 4;").unwrap();
    writeln!(out, "        add.u64 %rd5, %rd1, %rd5;").unwrap();
    writeln!(out, "        ld.global.f32 %f2, [%rd5];").unwrap();
    writeln!(out, "    }}").unwrap();
    // store to smB[ty*16+tx]
    writeln!(out, "    mul.wide.u32 %rd7, %r13, 4;").unwrap();
    writeln!(out, "    add.u64 %rd7, %rd4, %rd7;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd7], %f2;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    // Compute partial dot product
    writeln!(out, "    mov.u32 %r14, 0;").unwrap(); // i
    writeln!(out, "{kernel_name}_DOT_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p3, %r14, {tile};").unwrap();
    writeln!(out, "    @%p3 bra {kernel_name}_DOT_DONE;").unwrap();
    // smA[ty*16+i]
    writeln!(out, "    mad.lo.u32 %r15, %r4, {tile}, %r14;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd8, %r15, 4;").unwrap();
    writeln!(out, "    add.u64 %rd8, %rd3, %rd8;").unwrap();
    writeln!(out, "    ld.shared.f32 %f3, [%rd8];").unwrap();
    // smB[i*16+tx]
    writeln!(out, "    mad.lo.u32 %r15, %r14, {tile}, %r3;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd9, %r15, 4;").unwrap();
    writeln!(out, "    add.u64 %rd9, %rd4, %rd9;").unwrap();
    writeln!(out, "    ld.shared.f32 %f4, [%rd9];").unwrap();
    writeln!(out, "    fma.rn.f32 %f0, %f3, %f4, %f0;").unwrap();
    writeln!(out, "    add.u32 %r14, %r14, 1;").unwrap();
    writeln!(out, "    bra {kernel_name}_DOT_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_DOT_DONE:").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out, "    add.u32 %r9, %r9, 1;").unwrap();
    writeln!(out, "    bra {kernel_name}_TILE_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_TILE_DONE:").unwrap();
    // Write C[row, col] if in bounds
    writeln!(out, "    setp.lt.u32 %p1, %r7, %r0;").unwrap();
    writeln!(out, "    setp.lt.u32 %p2, %r8, %r1;").unwrap();
    writeln!(out, "    and.pred %p1, %p1, %p2;").unwrap();
    writeln!(out, "    @!%p1 bra {kernel_name}_END;").unwrap();
    writeln!(out, "    mad.lo.u32 %r16, %r7, %r1, %r8;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r16, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd2, %rd10;").unwrap();
    writeln!(out, "    st.global.f32 [%rd10], %f0;").unwrap();
    writeln!(out, "{kernel_name}_END:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}


fn emit_mha_kernel_ptx(out: &mut String, kernel_name: &str, seq_len: usize, num_heads: usize, head_dim: usize) {
    // Multi-Head Attention: output[b,h,i,j] = softmax(Q*K^T/sqrt(d)) * V
    // Simplified: one block per (head, query_row), block_size = head_dim (capped at 256)
    let block_size = head_dim.next_power_of_two().min(256);
    let scale_bits = format!("{:.8e}", 1.0_f64 / (head_dim as f64).sqrt())
        .replace("e", "e+").replace("e+-", "e-");
    // We encode scale as hex float literal for PTX
    let scale_f32 = 1.0_f32 / (head_dim as f32).sqrt();
    let scale_hex = format!("0f{:08X}", scale_f32.to_bits());

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_Q,").unwrap();
    writeln!(out, "    .param .u64 param_K,").unwrap();
    writeln!(out, "    .param .u64 param_V,").unwrap();
    writeln!(out, "    .param .u64 param_out,").unwrap();
    writeln!(out, "    .param .u32 param_seq,").unwrap();
    writeln!(out, "    .param .u32 param_heads,").unwrap();
    writeln!(out, "    .param .u32 param_dim").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    .shared .f32 smem_scores[{block_size}];").unwrap();
    writeln!(out, "    .shared .f32 smem_reduce[{block_size}];").unwrap();
    writeln!(out, "    .reg .u64 %rd<20>;").unwrap();
    writeln!(out, "    .reg .u32 %r<24>;").unwrap();
    writeln!(out, "    .reg .f32 %f<16>;").unwrap();
    writeln!(out, "    .reg .pred %p<6>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_Q];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_K];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_V];").unwrap();
    writeln!(out, "    ld.param.u64 %rd3, [param_out];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_seq];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_heads];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_dim];").unwrap();
    writeln!(out, "    mov.u32 %r3, %tid.x;").unwrap();   // thread in block
    writeln!(out, "    mov.u32 %r4, %ctaid.x;").unwrap(); // query row index
    writeln!(out, "    mov.u32 %r5, %ctaid.y;").unwrap(); // head index
    writeln!(out).unwrap();

    // Compute Q row base: Q + (head * seq * dim + query_row * dim) * 4
    writeln!(out, "    mul.lo.u32 %r6, %r5, %r0;").unwrap();       // head * seq
    writeln!(out, "    add.u32 %r6, %r6, %r4;").unwrap();           // + query_row
    writeln!(out, "    mul.lo.u32 %r6, %r6, %r2;").unwrap();        // * dim
    writeln!(out, "    mul.wide.u32 %rd4, %r6, 4;").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd0, %rd4;").unwrap();        // Q row ptr
    writeln!(out).unwrap();

    // K base for this head: K + head * seq * dim * 4
    writeln!(out, "    mul.lo.u32 %r7, %r5, %r0;").unwrap();
    writeln!(out, "    mul.lo.u32 %r7, %r7, %r2;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd5, %r7, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd1, %rd5;").unwrap();        // K head ptr
    writeln!(out).unwrap();

    // V base for this head
    writeln!(out, "    add.u64 %rd6, %rd2, %rd5;").unwrap();        // reuse offset (V same layout as K)
    writeln!(out, "    sub.u64 %rd6, %rd6, %rd1;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd2, %rd4;").unwrap();
    writeln!(out, "    sub.u64 %rd6, %rd6, %rd0;").unwrap();        // V row ptr = V + same offset as Q
    writeln!(out).unwrap();

    // --- Pass 1: compute attention scores scores[j] = dot(Q[i], K[j]) * scale ---
    writeln!(out, "    mov.u32 %r8, %r3;").unwrap(); // j = tid
    writeln!(out, "{kernel_name}_SCORE_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r8, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_SCORE_DONE;").unwrap();
    // dot(Q[i], K[j]) over dim
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
    writeln!(out, "    mov.u32 %r9, 0;").unwrap(); // d
    writeln!(out, "{kernel_name}_DOT_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p1, %r9, %r2;").unwrap();
    writeln!(out, "    @%p1 bra {kernel_name}_DOT_DONE;").unwrap();
    // Q[i][d]
    writeln!(out, "    mul.wide.u32 %rd7, %r9, 4;").unwrap();
    writeln!(out, "    add.u64 %rd7, %rd4, %rd7;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd7];").unwrap();
    // K[j][d]
    writeln!(out, "    mul.lo.u32 %r10, %r8, %r2;").unwrap();
    writeln!(out, "    add.u32 %r10, %r10, %r9;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd8, %r10, 4;").unwrap();
    writeln!(out, "    add.u64 %rd8, %rd5, %rd8;").unwrap();
    writeln!(out, "    ld.global.f32 %f2, [%rd8];").unwrap();
    writeln!(out, "    fma.rn.f32 %f0, %f1, %f2, %f0;").unwrap();
    writeln!(out, "    add.u32 %r9, %r9, 1;").unwrap();
    writeln!(out, "    bra {kernel_name}_DOT_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_DOT_DONE:").unwrap();
    // scale
    writeln!(out, "    mul.f32 %f0, %f0, {scale_hex};").unwrap();
    // store to smem_scores[j % block_size]
    writeln!(out, "    rem.u32 %r11, %r8, {block_size};").unwrap();
    writeln!(out, "    mov.u64 %rd9, smem_scores;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r11, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd9, %rd10;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd10], %f0;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_SCORE_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_SCORE_DONE:").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out).unwrap();

    // --- Pass 2: softmax over scores (reuse emit_softmax logic inline) ---
    // find max
    writeln!(out, "    mov.f32 %f3, 0fFF800000;").unwrap(); // -INF
    writeln!(out, "    mov.u32 %r8, %r3;").unwrap();
    writeln!(out, "{kernel_name}_SMAX_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r8, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_SMAX_DONE;").unwrap();
    writeln!(out, "    rem.u32 %r11, %r8, {block_size};").unwrap();
    writeln!(out, "    mov.u64 %rd9, smem_scores;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r11, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd9, %rd10;").unwrap();
    writeln!(out, "    ld.shared.f32 %f4, [%rd10];").unwrap();
    writeln!(out, "    max.f32 %f3, %f3, %f4;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_SMAX_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_SMAX_DONE:").unwrap();
    // reduce max across threads
    writeln!(out, "    mov.u64 %rd11, smem_reduce;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd12, %r3, 4;").unwrap();
    writeln!(out, "    add.u64 %rd12, %rd11, %rd12;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd12], %f3;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    let mut stride = block_size / 2;
    while stride > 0 {
        writeln!(out, "    setp.lt.u32 %p1, %r3, {stride};").unwrap();
        writeln!(out, "    @!%p1 bra {kernel_name}_MAXR_{stride};").unwrap();
        writeln!(out, "    add.u32 %r12, %r3, {stride};").unwrap();
        writeln!(out, "    mul.wide.u32 %rd13, %r12, 4;").unwrap();
        writeln!(out, "    add.u64 %rd13, %rd11, %rd13;").unwrap();
        writeln!(out, "    ld.shared.f32 %f5, [%rd13];").unwrap();
        writeln!(out, "    ld.shared.f32 %f6, [%rd12];").unwrap();
        writeln!(out, "    max.f32 %f6, %f6, %f5;").unwrap();
        writeln!(out, "    st.shared.f32 [%rd12], %f6;").unwrap();
        writeln!(out, "{kernel_name}_MAXR_{stride}:").unwrap();
        writeln!(out, "    bar.sync 0;").unwrap();
        stride /= 2;
    }
    writeln!(out, "    ld.shared.f32 %f7, [%rd11];").unwrap(); // row_max
    writeln!(out).unwrap();

    // exp(score - max) and sum
    writeln!(out, "    mov.f32 %f8, 0f00000000;").unwrap();
    writeln!(out, "    mov.u32 %r8, %r3;").unwrap();
    writeln!(out, "{kernel_name}_EXP_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r8, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_EXP_DONE;").unwrap();
    writeln!(out, "    rem.u32 %r11, %r8, {block_size};").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r11, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd9, %rd10;").unwrap();
    writeln!(out, "    ld.shared.f32 %f4, [%rd10];").unwrap();
    writeln!(out, "    sub.f32 %f4, %f4, %f7;").unwrap();
    writeln!(out, "    mul.f32 %f4, %f4, 0f3FB8AA3B;").unwrap();
    writeln!(out, "    ex2.approx.f32 %f4, %f4;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd10], %f4;").unwrap();
    writeln!(out, "    add.f32 %f8, %f8, %f4;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_EXP_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_EXP_DONE:").unwrap();
    // reduce sum
    writeln!(out, "    st.shared.f32 [%rd12], %f8;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    stride = block_size / 2;
    while stride > 0 {
        writeln!(out, "    setp.lt.u32 %p1, %r3, {stride};").unwrap();
        writeln!(out, "    @!%p1 bra {kernel_name}_SUMR_{stride};").unwrap();
        writeln!(out, "    add.u32 %r12, %r3, {stride};").unwrap();
        writeln!(out, "    mul.wide.u32 %rd13, %r12, 4;").unwrap();
        writeln!(out, "    add.u64 %rd13, %rd11, %rd13;").unwrap();
        writeln!(out, "    ld.shared.f32 %f5, [%rd13];").unwrap();
        writeln!(out, "    ld.shared.f32 %f6, [%rd12];").unwrap();
        writeln!(out, "    add.f32 %f6, %f6, %f5;").unwrap();
        writeln!(out, "    st.shared.f32 [%rd12], %f6;").unwrap();
        writeln!(out, "{kernel_name}_SUMR_{stride}:").unwrap();
        writeln!(out, "    bar.sync 0;").unwrap();
        stride /= 2;
    }
    writeln!(out, "    ld.shared.f32 %f9, [%rd11];").unwrap();
    writeln!(out, "    rcp.approx.f32 %f9, %f9;").unwrap(); // inv_sum
    writeln!(out).unwrap();

    // --- Pass 3: weighted sum over V ---
    // out[i][d] = sum_j attn[j] * V[j][d]
    writeln!(out, "    mov.u32 %r8, %r3;").unwrap(); // d = tid
    writeln!(out, "{kernel_name}_OUT_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r8, %r2;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_OUT_DONE;").unwrap();
    writeln!(out, "    mov.f32 %f10, 0f00000000;").unwrap();
    writeln!(out, "    mov.u32 %r9, 0;").unwrap(); // j
    writeln!(out, "{kernel_name}_WSUM_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p1, %r9, %r0;").unwrap();
    writeln!(out, "    @%p1 bra {kernel_name}_WSUM_DONE;").unwrap();
    // attn[j] from smem_scores
    writeln!(out, "    rem.u32 %r11, %r9, {block_size};").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r11, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd9, %rd10;").unwrap();
    writeln!(out, "    ld.shared.f32 %f11, [%rd10];").unwrap();
    writeln!(out, "    mul.f32 %f11, %f11, %f9;").unwrap(); // normalize
    // V[j][d]
    writeln!(out, "    mul.lo.u32 %r13, %r9, %r2;").unwrap();
    writeln!(out, "    add.u32 %r13, %r13, %r8;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd14, %r13, 4;").unwrap();
    writeln!(out, "    add.u64 %rd14, %rd6, %rd14;").unwrap();
    writeln!(out, "    ld.global.f32 %f12, [%rd14];").unwrap();
    writeln!(out, "    fma.rn.f32 %f10, %f11, %f12, %f10;").unwrap();
    writeln!(out, "    add.u32 %r9, %r9, 1;").unwrap();
    writeln!(out, "    bra {kernel_name}_WSUM_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_WSUM_DONE:").unwrap();
    // store out[i][d]
    writeln!(out, "    mul.lo.u32 %r14, %r5, %r0;").unwrap();
    writeln!(out, "    add.u32 %r14, %r14, %r4;").unwrap();
    writeln!(out, "    mul.lo.u32 %r14, %r14, %r2;").unwrap();
    writeln!(out, "    add.u32 %r14, %r14, %r8;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd15, %r14, 4;").unwrap();
    writeln!(out, "    add.u64 %rd15, %rd3, %rd15;").unwrap();
    writeln!(out, "    st.global.f32 [%rd15], %f10;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, {block_size};").unwrap();
    writeln!(out, "    bra {kernel_name}_OUT_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_OUT_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

// ── MachineCodeEmitter impl ─────────────────────────────────────────────────

impl MachineCodeEmitter for PtxCodeGen {
    fn emit_plan(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
        _alloc: &BufferAllocation,
        _profile: &DeviceProfile,
        registry: Option<&ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        let mut ptx = String::new();

        // PTX header
        let ptx_version = match self.sm_version {
            90.. => "8.0",
            80..=89 => "7.0",
            _ => "6.0",
        };
        writeln!(ptx, ".version {ptx_version}").unwrap();
        writeln!(ptx, ".target sm_{}", self.sm_version).unwrap();
        writeln!(ptx, ".address_size 64").unwrap();
        writeln!(ptx).unwrap();

        if plan.groups.is_empty() {
            return Ok(CodegenOutput {
                code: ptx.into_bytes(),
                scratchpad_bytes: 0,
            });
        }

        for group in &plan.groups {
            let anchor_op = graph.op(group.anchor).ok_or_else(|| {
                format!("PtxCodeGen: anchor op {:?} not found in graph", group.anchor)
            })?;

            let kernel_name = format!("group_{}", group.id);
            let op_kind = &anchor_op.kind;

            // Reshape/Transpose are metadata-only — NOP on GPU
            if matches!(op_kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
                continue;
            }


            let registry = registry.ok_or_else(|| {
                format!("PtxCodeGen: emit_plan requires a ScalarOpRegistry for {:?}", op_kind)
            })?;
            let key = ScalarOpRegistry::key_from_op_kind(op_kind);
            let trace = registry.get_trace(&key).ok_or_else(|| {
                format!("PtxCodeGen: no OpTrace for {:?}", op_kind)
            })?;

            match &trace.pattern {
                ComputePattern::Elementwise { body } => {
                    emit_elementwise_kernel_ptx(&mut ptx, &kernel_name, body);
                }
                ComputePattern::BinaryElementwise { body } => {
                    emit_binary_elementwise_kernel_ptx(&mut ptx, &kernel_name, body);
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
                    emit_normlike_kernel_ptx(
                        &mut ptx,
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
                            emit_softmax_kernel_ptx(&mut ptx, &kernel_name);
                        }
                        OpKind::MeanPool { seq_len, hidden } => {
                            emit_meanpool_kernel_ptx(
                                &mut ptx, &kernel_name, *seq_len, *hidden,
                            );
                        }
                        _ => {
                            return Err(format!(
                                "PtxCodeGen: unsupported Reduction op {:?}",
                                op_kind
                            ));
                        }
                    }
                }
                ComputePattern::Gemm => {
                    match op_kind {
                        OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => {
                            emit_gemm_kernel_ptx(&mut ptx, &kernel_name, *m, *n, *k);
                        }
                        OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                            emit_mha_kernel_ptx(&mut ptx, &kernel_name, *seq_len, *num_heads, *head_dim);
                        }
                        _ => {
                            return Err(format!(
                                "PtxCodeGen: unsupported Gemm-pattern op {:?}",
                                op_kind
                            ));
                        }
                    }
                }
                ComputePattern::Injective { body, .. } => {
                    if body.is_empty() {
                        continue;
                    }
                    return Err(format!(
                        "PtxCodeGen: non-trivial Injective codegen not yet implemented (op {:?})",
                        op_kind
                    ));
                }
                ComputePattern::QuantDecode { .. } => {
                    return Err(format!(
                        "PtxCodeGen: QuantDecode codegen not yet implemented (op {:?})",
                        op_kind
                    ));
                }
            }
        }

        Ok(CodegenOutput {
            code: ptx.into_bytes(),
            scratchpad_bytes: 0,
        })
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
    fn test_ptx_trace_op_to_ptx_basic() {
        let mut out = String::new();
        let vars = vec!["%f0".to_string(), "%f1".to_string()];

        let reg = trace_op_to_ptx(&mut out, &TraceOp::Add(0, 1), 0, &vars, 99);
        assert!(out.contains("add.f32"));
        assert_eq!(reg, "%t99_0");

        out.clear();
        let reg = trace_op_to_ptx(&mut out, &TraceOp::Mul(0, 1), 1, &vars, 99);
        assert!(out.contains("mul.f32"));

        out.clear();
        let reg = trace_op_to_ptx(&mut out, &TraceOp::Exp(0), 2, &vars, 99);
        assert!(out.contains("ex2.approx.f32"));

        out.clear();
        let reg = trace_op_to_ptx(&mut out, &TraceOp::Rsqrt(0), 3, &vars, 99);
        assert!(out.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_emit_trace_body() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
        ];
        let mut out = String::new();
        let bindings = vec!["%f0".to_string()];
        let result = emit_trace_body_ptx(&mut out, &body, 0, &bindings);
        assert!(out.contains("neg.f32"));
        assert_eq!(result, "%t0_1");
    }

    #[test]
    fn test_ptx_emit_elementwise_kernel_from_trace() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
        ];
        let mut out = String::new();
        emit_elementwise_kernel_ptx(&mut out, "test_neg", &body);
        assert!(out.contains(".visible .entry test_neg("));
        assert!(out.contains("neg.f32"));
        assert!(out.contains("st.global.f32"));
        assert!(out.contains("ret;"));
    }

    #[test]
    fn test_ptx_emit_binary_elementwise_kernel_from_trace() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(0, 1),
        ];
        let mut out = String::new();
        emit_binary_elementwise_kernel_ptx(&mut out, "test_add", &body);
        assert!(out.contains(".visible .entry test_add("));
        assert!(out.contains("param_input0"));
        assert!(out.contains("param_input1"));
        assert!(out.contains("add.f32"));
    }

    #[test]
    fn test_ptx_emit_normlike_kernel() {
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
            TraceOp::Input(2),
            TraceOp::Mul(0, 1),
            TraceOp::Mul(3, 2),
        ];
        let mut out = String::new();
        emit_normlike_kernel_ptx(
            &mut out, "test_rmsnorm",
            &reduce, &finalize, &transform,
            true, false, Some(1e-5),
        );
        assert!(out.contains(".visible .entry test_rmsnorm("));
        assert!(out.contains(".shared .f32 smem[256]"));
        assert!(out.contains("bar.sync 0"));
        assert!(out.contains("rsqrt.approx.f32"));
        assert!(out.contains("param_weight"));
        assert!(!out.contains("param_bias"));
    }


    #[test]
    fn test_ptx_emit_normlike_l2normalize() {
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
        emit_normlike_kernel_ptx(
            &mut out, "test_l2norm",
            &reduce, &finalize, &transform,
            false, false, Some(1e-5),
        );
        assert!(out.contains(".visible .entry test_l2norm("));
        assert!(out.contains("rsqrt.approx.f32"));
        assert!(!out.contains("param_weight"));
        assert!(!out.contains("param_bias"));
    }
    #[test]
    fn test_ptx_emit_softmax_kernel() {
        let mut out = String::new();
        emit_softmax_kernel_ptx(&mut out, "test_softmax");
        assert!(out.contains(".visible .entry test_softmax("));
        assert!(out.contains("max.f32"));
        assert!(out.contains("ex2.approx.f32"));
        assert!(out.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_emit_plan_empty() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::FusionPlan;
        use crate::compiler::graph::CompilerGraph;
        use crate::dispatch::DeviceProfile;

        let mut cg = PtxCodeGen::new(80);
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
                let ptx = String::from_utf8(output.code).unwrap();
                assert!(ptx.contains(".version 7.0"));
                assert!(ptx.contains(".target sm_80"));
                assert!(ptx.contains(".address_size 64"));
            }
            Err(e) => panic!("expected Ok for empty plan, got Err: {e}"),
        }
    }

    #[test]
    fn test_ptx_emit_plan_requires_registry() {
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::graph::{CompilerGraph, OpKind};
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::inference::types::DType;
        use crate::dispatch::DeviceProfile;
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

        let mut cg = PtxCodeGen::new(80);
        // Without registry should fail
        let result = cg.emit_plan(&plan, &g, &alloc, &profile, None);
        match result {
            Err(e) => assert!(e.contains("requires a ScalarOpRegistry"), "unexpected error: {e}"),
            Ok(_) => panic!("expected Err when registry is None"),
        }

        // With registry should succeed
        let registry = ScalarOpRegistry::with_defaults();
        let result = cg.emit_plan(&plan, &g, &alloc, &profile, Some(&registry));
        let ptx = match result {
            Ok(output) => String::from_utf8(output.code).unwrap(),
            Err(e) => panic!("expected Ok with registry, got Err: {e}"),
        };
        assert!(ptx.contains(".visible .entry group_0("));
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

    #[test]
    fn test_ptx_emit_meanpool_kernel() {
        let mut out = String::new();
        emit_meanpool_kernel_ptx(&mut out, "test_meanpool", 128, 768);
        assert!(out.contains(".visible .entry test_meanpool("));
        assert!(out.contains("param_input"));
        assert!(out.contains("param_output"));
        assert!(out.contains("rcp.approx.f32"));
        assert!(out.contains("st.global.f32"));
    }
}
