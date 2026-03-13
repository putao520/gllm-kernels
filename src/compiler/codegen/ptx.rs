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
use crate::compiler::codegen::gpu_ir::trace_emitter::{GpuDialect, PtxDialect};
use crate::compiler::codegen::gpu_ir::plan_emitter::gpu_emit_plan;
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

// GEMM kernels (emit_gemm_kernel_ptx, emit_gemm_tc_sm70/80/89_ptx) moved to ptx_gemm.rs

// Re-export GEMM emitters from ptx_gemm submodule.
pub(crate) use super::ptx_gemm::*;

/// Emit RoPE (Rotary Position Embedding) kernel.
///
/// RoPE applies rotation to pairs of elements: for each pair (x[2i], x[2i+1]),
///   x'[2i]   = x[2i]   * cos(theta_i) - x[2i+1] * sin(theta_i)
///   x'[2i+1] = x[2i]   * sin(theta_i) + x[2i+1] * cos(theta_i)
/// where theta_i = pos / (theta_base ^ (2i / head_dim))
///
/// Kernel signature: (input, output, pos, N, head_dim, theta_base_inv_hex)
pub(crate) fn emit_rope_kernel_ptx(out: &mut String, kernel_name: &str, head_dim: usize, theta: f64) {
    let block_size: u32 = 256;
    let half_dim = head_dim / 2;
    // Precompute 1/theta as f32 hex for PTX
    let inv_theta_f32 = (1.0 / theta) as f32;
    let inv_theta_hex = format!("0f{:08X}", inv_theta_f32.to_bits());

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_pos,").unwrap();
    writeln!(out, "    .param .u32 param_N").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<12>;").unwrap();
    writeln!(out, "    .reg .u32 %r<12>;").unwrap();
    writeln!(out, "    .reg .f32 %f<12>;").unwrap();
    writeln!(out, "    .reg .pred %p<2>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_output];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_pos];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out).unwrap();

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    writeln!(out, "    mov.u32 %r2, %ctaid.x;").unwrap();
    writeln!(out, "    mov.u32 %r3, %ntid.x;").unwrap();
    writeln!(out, "    mad.lo.u32 %r2, %r2, %r3, %tid.x;").unwrap();
    writeln!(out).unwrap();

    // Each thread handles one pair index i (processes elements 2*i and 2*i+1)
    // Guard: if tid >= N/2 return
    writeln!(out, "    shr.u32 %r4, %r1, 1;").unwrap(); // N/2
    writeln!(out, "    setp.ge.u32 %p0, %r2, %r4;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_DONE;").unwrap();
    writeln!(out).unwrap();

    // Compute i_mod = tid % half_dim (position within head)
    writeln!(out, "    rem.u32 %r5, %r2, {half_dim};").unwrap();
    writeln!(out).unwrap();

    // Compute freq = pos * (1/theta)^(2*i_mod/head_dim)
    // = pos * exp(-2*i_mod/head_dim * ln(theta))
    // We compute: freq_exp = 2*i_mod / head_dim (as float)
    // Then: base_freq = pow(1/theta, freq_exp) via exp2(freq_exp * log2(1/theta))
    // Then: angle = pos * base_freq
    writeln!(out, "    shl.b32 %r6, %r5, 1;").unwrap();         // 2 * i_mod
    writeln!(out, "    cvt.rn.f32.u32 %f0, %r6;").unwrap();     // float(2*i_mod)
    writeln!(out, "    mov.f32 %f1, 0f{:08X};", (1.0f32 / head_dim as f32).to_bits()).unwrap(); // 1/head_dim
    writeln!(out, "    mul.f32 %f0, %f0, %f1;").unwrap();       // 2*i_mod/head_dim
    // log2(1/theta) = -log2(theta)
    writeln!(out, "    mov.f32 %f2, {inv_theta_hex};").unwrap(); // 1/theta
    writeln!(out, "    lg2.approx.f32 %f2, %f2;").unwrap();     // log2(1/theta)
    writeln!(out, "    mul.f32 %f0, %f0, %f2;").unwrap();       // freq_exp * log2(1/theta)
    writeln!(out, "    ex2.approx.f32 %f0, %f0;").unwrap();     // (1/theta)^(2i/d)
    // angle = pos * base_freq
    writeln!(out, "    cvt.rn.f32.u32 %f3, %r0;").unwrap();     // float(pos)
    writeln!(out, "    mul.f32 %f0, %f3, %f0;").unwrap();       // angle
    writeln!(out).unwrap();

    // Compute sin(angle) and cos(angle) via PTX approx
    writeln!(out, "    sin.approx.f32 %f4, %f0;").unwrap();     // sin(angle)
    writeln!(out, "    cos.approx.f32 %f5, %f0;").unwrap();     // cos(angle)
    writeln!(out).unwrap();

    // Load input pair: x0 = input[2*tid], x1 = input[2*tid+1]
    writeln!(out, "    shl.b32 %r7, %r2, 1;").unwrap();         // 2*tid
    writeln!(out, "    mul.wide.u32 %rd2, %r7, 4;").unwrap();   // byte offset for x0
    writeln!(out, "    add.u64 %rd3, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f6, [%rd3];").unwrap();    // x0
    writeln!(out, "    ld.global.f32 %f7, [%rd3+4];").unwrap();  // x1
    writeln!(out).unwrap();

    // Apply rotation:
    //   out0 = x0 * cos - x1 * sin
    //   out1 = x0 * sin + x1 * cos
    writeln!(out, "    mul.f32 %f8, %f6, %f5;").unwrap();       // x0 * cos
    writeln!(out, "    mul.f32 %f9, %f7, %f4;").unwrap();       // x1 * sin
    writeln!(out, "    sub.f32 %f8, %f8, %f9;").unwrap();       // out0 = x0*cos - x1*sin
    writeln!(out, "    mul.f32 %f10, %f6, %f4;").unwrap();      // x0 * sin
    writeln!(out, "    fma.rn.f32 %f11, %f7, %f5, %f10;").unwrap(); // out1 = x1*cos + x0*sin
    writeln!(out).unwrap();

    // Store output pair
    writeln!(out, "    add.u64 %rd4, %rd1, %rd2;").unwrap();
    writeln!(out, "    st.global.f32 [%rd4], %f8;").unwrap();
    writeln!(out, "    st.global.f32 [%rd4+4], %f11;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "{kernel_name}_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

pub(crate) fn emit_mha_kernel_ptx(out: &mut String, kernel_name: &str, seq_len: usize, num_heads: usize, head_dim: usize) {
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
        let dialect = PtxDialect { sm_version: self.sm_version };
        let mut ptx = String::new();
        dialect.emit_header(&mut ptx);

        if plan.groups.is_empty() {
            return Ok(CodegenOutput {
                code: ptx.into_bytes(),
                scratchpad_bytes: 0,
            });
        }

        gpu_emit_plan(&dialect, &mut ptx, plan, graph, registry)?;

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

// ── Reduction kernel ────────────────────────────────────────────────

/// Emit a reduction kernel in PTX assembly.
///
/// Uses shared memory + tree reduction. Block size = 128 (32 warps * 4).
/// Grid-stride accumulation into per-thread accumulators, then shared memory
/// tree reduction within each block.
pub(crate) fn emit_reduction_kernel_ptx(
    out: &mut String,
    kernel_name: &str,
    identity: f64,
    combine: &[TraceOp],
) {
    let block_size: u32 = 128; // 32 (warp) * 4
    let id_bits = (identity as f32).to_bits();
    let combine_regs = max_regs_needed(combine, 1);
    let combine_regs2 = max_regs_needed(combine, 2);
    let total_regs = combine_regs.max(combine_regs2) + 4;

    // Kernel signature: input, output, n
    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_n").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();

    // Shared memory + registers
    writeln!(out, "    .shared .f32 sdata[{block_size}];").unwrap();
    writeln!(out, "    .reg .u64 %rd<8>;").unwrap();
    writeln!(out, "    .reg .u32 %r<16>;").unwrap();
    writeln!(out, "    .reg .f32 %f<{total_regs}>;").unwrap();
    writeln!(out, "    .reg .f32 %t1_<{combine_regs}>;").unwrap();
    writeln!(out, "    .reg .f32 %t2_<{combine_regs2}>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_output];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_n];").unwrap();
    writeln!(out).unwrap();

    // tid = threadIdx.x
    writeln!(out, "    mov.u32 %r1, %tid.x;").unwrap();
    // gid = blockIdx.x * block_size + tid
    writeln!(out, "    mov.u32 %r2, %ctaid.x;").unwrap();
    writeln!(out, "    mad.lo.u32 %r3, %r2, {block_size}, %r1;").unwrap();
    // grid_stride = gridDim.x * block_size
    writeln!(out, "    mov.u32 %r4, %nctaid.x;").unwrap();
    writeln!(out, "    mul.lo.u32 %r5, %r4, {block_size};").unwrap();
    writeln!(out).unwrap();

    // acc = identity
    writeln!(out, "    mov.f32 %f0, 0f{id_bits:08X};").unwrap();
    writeln!(out).unwrap();

    // Grid-stride loop: for (i = gid; i < n; i += grid_stride)
    writeln!(out, "    mov.u32 %r6, %r3;").unwrap();
    writeln!(out, "{kernel_name}_LOOP:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r6, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_LOOP_DONE;").unwrap();
    writeln!(out).unwrap();

    // Load input[i]
    writeln!(out, "    mul.wide.u32 %rd2, %r6, 4;").unwrap();
    writeln!(out, "    add.u64 %rd3, %rd0, %rd2;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd3];").unwrap();
    writeln!(out).unwrap();

    // acc = combine(acc, input[i])
    let bindings = vec!["%f0".to_string(), "%f1".to_string()];
    let result = emit_trace_body_ptx(out, combine, 1, &bindings);
    writeln!(out, "    mov.f32 %f0, {result};").unwrap();
    writeln!(out).unwrap();

    // i += grid_stride
    writeln!(out, "    add.u32 %r6, %r6, %r5;").unwrap();
    writeln!(out, "    bra {kernel_name}_LOOP;").unwrap();
    writeln!(out, "{kernel_name}_LOOP_DONE:").unwrap();
    writeln!(out).unwrap();

    // sdata[tid] = acc
    writeln!(out, "    mul.wide.u32 %rd4, %r1, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd5, sdata;").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd5, %rd4;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd4], %f0;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out).unwrap();

    // Tree reduction in shared memory
    let mut s = block_size / 2;
    let mut step = 0u32;
    while s > 0 {
        writeln!(out, "    setp.lt.u32 %p1, %r1, {s};").unwrap();
        writeln!(out, "    @!%p1 bra {kernel_name}_REDUCE_{step}_SKIP;").unwrap();

        // in0 = sdata[tid]
        writeln!(out, "    mul.wide.u32 %rd4, %r1, 4;").unwrap();
        writeln!(out, "    add.u64 %rd4, %rd5, %rd4;").unwrap();
        writeln!(out, "    ld.shared.f32 %f0, [%rd4];").unwrap();

        // in1 = sdata[tid + s]
        writeln!(out, "    add.u32 %r7, %r1, {s};").unwrap();
        writeln!(out, "    mul.wide.u32 %rd6, %r7, 4;").unwrap();
        writeln!(out, "    add.u64 %rd6, %rd5, %rd6;").unwrap();
        writeln!(out, "    ld.shared.f32 %f1, [%rd6];").unwrap();

        // combine
        let bindings2 = vec!["%f0".to_string(), "%f1".to_string()];
        let r2 = emit_trace_body_ptx(out, combine, 2, &bindings2);
        // sdata[tid] = result
        writeln!(out, "    st.shared.f32 [%rd4], {r2};").unwrap();

        writeln!(out, "{kernel_name}_REDUCE_{step}_SKIP:").unwrap();
        writeln!(out, "    bar.sync 0;").unwrap();
        s /= 2;
        step += 1;
    }
    writeln!(out).unwrap();

    // if (tid == 0) output[blockIdx.x] = sdata[0]
    writeln!(out, "    setp.ne.u32 %p2, %r1, 0;").unwrap();
    writeln!(out, "    @%p2 bra {kernel_name}_DONE;").unwrap();
    writeln!(out, "    ld.shared.f32 %f0, [%rd5];").unwrap();
    writeln!(out, "    mul.wide.u32 %rd6, %r2, 4;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd1, %rd6;").unwrap();
    writeln!(out, "    st.global.f32 [%rd6], %f0;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "{kernel_name}_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

// ── Injective kernel ────────────────────────────────────────────────

/// Emit a multi-input/multi-output injective kernel in PTX assembly.
///
/// Grid-stride loop with bounds check. Each thread processes one element.
pub(crate) fn emit_injective_kernel_ptx(
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
    num_inputs: usize,
    num_outputs: usize,
) {
    let nregs = max_regs_needed(body, 0);

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    for i in 0..num_inputs {
        writeln!(out, "    .param .u64 param_input{i},").unwrap();
    }
    for i in 0..num_outputs {
        writeln!(out, "    .param .u64 param_output{i},").unwrap();
    }
    writeln!(out, "    .param .u32 param_n").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();

    // Registers
    let num_rd = 4 + num_inputs + num_outputs;
    writeln!(out, "    .reg .u64 %rd<{num_rd}>;").unwrap();
    writeln!(out, "    .reg .u32 %r<8>;").unwrap();
    writeln!(out, "    .reg .f32 %t0_<{nregs}>;").unwrap();
    writeln!(out, "    .reg .f32 %f<{}>;", num_inputs + 4).unwrap();
    writeln!(out, "    .reg .pred %p<2>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    for i in 0..num_inputs {
        writeln!(out, "    ld.param.u64 %rd{i}, [param_input{i}];").unwrap();
    }
    let out_base = num_inputs;
    for i in 0..num_outputs {
        writeln!(out, "    ld.param.u64 %rd{}, [param_output{i}];", out_base + i).unwrap();
    }
    let n_reg_rd = out_base + num_outputs;
    writeln!(out, "    ld.param.u32 %r0, [param_n];").unwrap();
    writeln!(out).unwrap();

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    writeln!(out, "    mov.u32 %r1, %ctaid.x;").unwrap();
    writeln!(out, "    mov.u32 %r2, %ntid.x;").unwrap();
    writeln!(out, "    mad.lo.u32 %r1, %r1, %r2, %tid.x;").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r1, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {kernel_name}_DONE;").unwrap();
    writeln!(out).unwrap();

    // Load inputs
    let mut bindings = Vec::new();
    for i in 0..num_inputs {
        let freg = format!("%f{i}");
        writeln!(out, "    mul.wide.u32 %rd{n_reg_rd}, %r1, 4;").unwrap();
        writeln!(out, "    add.u64 %rd{n_reg_rd}, %rd{i}, %rd{n_reg_rd};").unwrap();
        writeln!(out, "    ld.global.f32 {freg}, [%rd{n_reg_rd}];").unwrap();
        bindings.push(freg);
    }
    writeln!(out).unwrap();

    // Emit trace body
    let result = emit_trace_body_ptx(out, body, 0, &bindings);
    writeln!(out).unwrap();

    // Store outputs
    if num_outputs == 1 {
        writeln!(out, "    mul.wide.u32 %rd{n_reg_rd}, %r1, 4;").unwrap();
        writeln!(out, "    add.u64 %rd{n_reg_rd}, %rd{out_base}, %rd{n_reg_rd};").unwrap();
        writeln!(out, "    st.global.f32 [%rd{n_reg_rd}], {result};").unwrap();
    } else {
        let base_op = body.len().saturating_sub(num_outputs);
        for i in 0..num_outputs {
            let var = format!("%t0_{}", base_op + i);
            writeln!(out, "    mul.wide.u32 %rd{n_reg_rd}, %r1, 4;").unwrap();
            writeln!(out, "    add.u64 %rd{n_reg_rd}, %rd{}, %rd{n_reg_rd};", out_base + i).unwrap();
            writeln!(out, "    st.global.f32 [%rd{n_reg_rd}], {var};").unwrap();
        }
    }
    writeln!(out).unwrap();

    writeln!(out, "{kernel_name}_DONE:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
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
            Err(e) => assert!(e.contains("ScalarOpRegistry required"), "unexpected error: {e}"),
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
    fn test_ptx_emit_gemm_tc_sm89_fp8() {
        let mut out = String::new();
        emit_gemm_tc_sm89_ptx(&mut out, "test_gemm_fp8", 64, 64, 64);
        assert!(out.contains(".visible .entry test_gemm_fp8("));
        assert!(out.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"));
        assert!(out.contains("cp.async.cg.shared.global"));
        assert!(out.contains("cp.async.commit_group"));
        assert!(out.contains("st.global.f32"));
    }

    #[test]
    fn test_ptx_emit_rope_kernel() {
        let mut out = String::new();
        emit_rope_kernel_ptx(&mut out, "test_rope", 128, 10000.0);
        assert!(out.contains(".visible .entry test_rope("));
        assert!(out.contains("param_input"));
        assert!(out.contains("param_output"));
        assert!(out.contains("param_pos"));
        assert!(out.contains("sin.approx.f32"));
        assert!(out.contains("cos.approx.f32"));
        assert!(out.contains("st.global.f32"));
    }

    #[test]
    fn test_ptx_gemm_sm_dispatch() {
        // Verify that different SM versions select different GEMM paths
        let mut out_70 = String::new();
        emit_gemm_tc_sm70_ptx(&mut out_70, "gemm_70", 64, 64, 64);
        assert!(out_70.contains("wmma.load.a.sync.aligned"));
        assert!(out_70.contains("wmma.mma.sync.aligned"));

        let mut out_80 = String::new();
        emit_gemm_tc_sm80_ptx(&mut out_80, "gemm_80", 64, 64, 64);
        assert!(out_80.contains("mma.sync.aligned.m16n8k16"));
        assert!(!out_80.contains("wmma.load"));

        let mut out_89 = String::new();
        emit_gemm_tc_sm89_ptx(&mut out_89, "gemm_89", 64, 64, 64);
        assert!(out_89.contains("mma.sync.aligned.m16n8k32"));
        assert!(out_89.contains("e4m3"));
    }
}
