//! GpuDialect trait + unified `emit_trace_body<D>`.
//!
//! Phase 1 of the GPU codegen unification (SPEC PLAN-gpu-codegen-unify §2.2).
//! Provides a single trait that PTX, HIP, and MSL backends implement to translate
//! `TraceOp` into their respective instruction formats. The generic
//! `emit_trace_body<D>` replaces the three duplicated `emit_trace_body_ptx/hip/msl`
//! functions.

#[cfg(any(feature = "jit-cuda", feature = "jit-hip", feature = "jit-metal"))]
use std::fmt::Write;
use crate::compiler::trace::TraceOp;
use super::primitives::{GpuCapabilities, KernelParam};
#[cfg(any(feature = "jit-cuda", feature = "jit-hip", feature = "jit-metal"))]
use super::primitives::{ParamType, ParamQualifier};
use crate::compiler::graph::OpKind;

// ── GpuDialect trait ────────────────────────────────────────────────────────

/// Abstraction over GPU code-generation backends (PTX / HIP C++ / MSL).
///
/// Each method emits text into an `&mut String` buffer. The trait is object-safe
/// for the subset used by `emit_trace_body`, but most callers will use it via
/// generics (`<D: GpuDialect>`).
pub trait GpuDialect {
    // ── File-level ──

    /// Emit the file/module header (includes, version directives, etc.).
    fn emit_header(&self, out: &mut String);

    // ── TraceOp translation ──

    /// Emit instructions for a single `TraceOp` and return the variable/register
    /// name holding the result.
    ///
    /// - `idx`: position in the trace body (used for naming: `t{tier}_{idx}`)
    /// - `vars`: variable names of all prior ops in this body
    /// - `tier`: namespace tier to avoid name collisions across phases
    fn emit_trace_op(
        &self,
        out: &mut String,
        op: &TraceOp,
        idx: usize,
        vars: &[String],
        tier: usize,
    ) -> String;

    // ── GPU metadata ──

    /// Warp/wavefront/SIMD-group width.
    fn warp_size(&self) -> u32;

    /// Backend capability flags.
    fn capabilities(&self) -> GpuCapabilities;

    // ── Kernel-level structural emission (Phase 2) ──

    /// Emit a kernel function signature opening + body brace.
    /// `params` is a list of `(name, ParamType, ParamQualifier)`.
    fn emit_kernel_start(
        &self,
        out: &mut String,
        name: &str,
        params: &[KernelParam],
        shared_mem_bytes: usize,
    );

    /// Emit the kernel closing brace / return.
    fn emit_kernel_end(&self, out: &mut String);

    /// Return an expression for the global thread ID (1D).
    fn global_tid_expr(&self) -> &'static str;

    /// Return an expression for the local thread ID within a threadgroup.
    fn local_tid_expr(&self) -> &'static str;

    /// Return an expression for the threadgroup/block ID.
    fn group_id_expr(&self) -> &'static str;

    /// Emit a shared memory declaration.
    fn emit_shared_decl(&self, out: &mut String, name: &str, count: usize);

    /// Emit a barrier / syncthreads.
    fn emit_barrier(&self, out: &mut String);

    /// Emit the start of a for-loop: `for (uint i = start; i < limit; i += stride)`.
    /// Returns the loop variable name.
    /// For PTX, emits label + branch structure.
    fn emit_for_start(
        &self,
        out: &mut String,
        var: &str,
        start: &str,
        limit: &str,
        stride: &str,
        label: &str,
    );

    /// Emit the end of a for-loop.
    fn emit_for_end(&self, out: &mut String, var: &str, stride: &str, label: &str);

    /// Emit an if-guard: `if (cond) { ... }` or PTX predicate branch.
    fn emit_if_start(&self, out: &mut String, cond: &str, label: &str);

    /// Emit the end of an if-guard.
    fn emit_if_end(&self, out: &mut String, label: &str);

    /// Emit a float variable declaration + assignment.
    fn emit_float_decl(&self, out: &mut String, name: &str, expr: &str);

    /// Emit a float assignment (no declaration).
    fn emit_float_assign(&self, out: &mut String, name: &str, expr: &str);

    /// Emit a global memory load: `name = ptr[index]`.
    fn emit_global_load(&self, out: &mut String, dst: &str, ptr: &str, index: &str);

    /// Emit a global memory store: `ptr[index] = value`.
    fn emit_global_store(&self, out: &mut String, ptr: &str, index: &str, value: &str);

    /// Emit a shared memory load.
    fn emit_shared_load(&self, out: &mut String, dst: &str, array: &str, index: &str);

    /// Emit a shared memory store.
    fn emit_shared_store(&self, out: &mut String, array: &str, index: &str, value: &str);

    /// Emit a bounds-check return: `if (tid >= n) return;`
    fn emit_bounds_check_return(&self, out: &mut String, var: &str, limit: &str, label: &str);

    /// Default block size for this backend.
    fn default_block_size(&self) -> u32 { 256 }

    // ── Backend-specific kernel dispatch (Phase 3) ──

    /// Emit a 1-input, 1-output elementwise kernel from a trace body.
    fn emit_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]);

    /// Emit a 2-input, 1-output binary elementwise kernel from a trace body.
    fn emit_binary_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]);

    /// Emit a GEMM kernel. Each backend uses its own strategy (tensor cores,
    /// simdgroup matrix, MFMA, or tiled shared-memory).
    fn emit_gemm_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
    ) -> Result<(), String>;

    /// Emit a multi-head attention kernel.
    fn emit_mha_kernel(
        &self,
        out: &mut String,
        name: &str,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    );

    /// Emit a RoPE (rotary positional embedding) kernel.
    fn emit_rope_kernel(
        &self,
        out: &mut String,
        name: &str,
        head_dim: usize,
        theta: f64,
    );

    /// Emit a generic reduction kernel with identity + combine trace.
    /// Returns `Err` if the backend doesn't support generic reductions.
    fn emit_reduction_kernel(
        &self,
        out: &mut String,
        name: &str,
        identity: f64,
        combine: &[TraceOp],
    ) -> Result<(), String>;

    /// Emit a multi-input/multi-output injective kernel.
    /// Returns `Err` if the backend doesn't support generic injective codegen.
    fn emit_injective_kernel(
        &self,
        out: &mut String,
        name: &str,
        body: &[TraceOp],
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<(), String>;

    /// Emit a NormLike kernel (RmsNorm, LayerNorm, etc.).
    fn emit_normlike_kernel(
        &self,
        out: &mut String,
        name: &str,
        reduce: &[TraceOp],
        finalize: &[TraceOp],
        transform: &[TraceOp],
        has_weight: bool,
        has_bias: bool,
        eps_override: Option<f32>,
    );

    /// Emit a Softmax kernel.
    fn emit_softmax_kernel(&self, out: &mut String, name: &str);

    /// Emit a MeanPool kernel.
    fn emit_meanpool_kernel(&self, out: &mut String, name: &str, seq_len: usize, hidden: usize);

    /// Emit a Dequantize kernel.
    fn emit_dequantize_kernel(
        &self,
        out: &mut String,
        name: &str,
        num_elements: usize,
        block_size: usize,
        bits: u8,
    );

    /// Emit a fused GEMM + epilogue kernel (EpilogueInjection).
    ///
    /// The GEMM computes C = A * B (+ bias for GemmBias), then each output
    /// element has the epilogue trace bodies applied in sequence before the
    /// final store. This avoids a round-trip through global memory between
    /// the GEMM and the activation/elementwise ops.
    ///
    /// `epilogue_bodies` contains the `TraceOp` SSA traces for each epilogue
    /// op (e.g. GELU, SiLU, residual add), in execution order.
    fn emit_gemm_epilogue_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
        epilogue_bodies: &[&[TraceOp]],
    ) -> Result<(), String>;

    /// Emit a fused TileLevelFusion kernel: norm tiled into GEMM thread-block
    /// loop via shared memory.
    ///
    /// The predecessor (NormLike) output is computed per tile strip in shared
    /// memory, then the GEMM reads from shared memory instead of global memory.
    /// This avoids a full global memory round-trip for the norm output.
    ///
    /// Default implementation returns `Err` — backends implement as needed.
    fn emit_tile_level_fusion_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        tile_rows: usize,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        shared_mem_budget: usize,
    ) -> Result<(), String> {
        let _ = (out, name, norm_op, gemm_op, tile_rows, norm_reduce, norm_finalize, norm_transform, shared_mem_budget);
        Err("TileLevelFusion not implemented for this GPU dialect".into())
    }

    /// Emit a ComputeRoot kernel: full norm into shared/global memory, then GEMM.
    ///
    /// The predecessor (NormLike) is computed fully before the GEMM. If the norm
    /// output fits in shared memory (≤ budget), it stays there; otherwise a
    /// two-kernel approach is used with global memory as intermediate.
    ///
    /// Default implementation returns `Err` — backends implement as needed.
    fn emit_compute_root_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        shared_mem_budget: usize,
    ) -> Result<(), String> {
        let _ = (out, name, norm_op, gemm_op, norm_reduce, norm_finalize, norm_transform, shared_mem_budget);
        Err("ComputeRoot not implemented for this GPU dialect".into())
    }
}

// ── Unified emit_trace_body ─────────────────────────────────────────────────

/// Emit a sequence of `TraceOp`s using the given dialect, returning the
/// variable name of the last (result) op.
///
/// `input_bindings` maps `Input(i)` to a pre-bound variable/register name.
/// Ops that are `TraceOp::Input(idx)` use the binding directly; all others
/// are emitted via `dialect.emit_trace_op()`.
pub fn emit_trace_body<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    ops: &[TraceOp],
    tier: usize,
    input_bindings: &[String],
) -> String {
    let mut vars: Vec<String> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        let var = if let TraceOp::Input(idx) = op {
            input_bindings
                .get(*idx as usize)
                .cloned()
                .unwrap_or_else(|| format!("input{idx}"))
        } else {
            dialect.emit_trace_op(out, op, i, &vars, tier)
        };
        vars.push(var);
    }
    vars.last()
        .cloned()
        .unwrap_or_else(|| "0.0f".to_string())
}

// ── PtxDialect ──────────────────────────────────────────────────────────────

/// PTX dialect — emits PTX virtual ISA assembly.
#[cfg(feature = "jit-cuda")]
pub struct PtxDialect {
    /// Target SM version (e.g. 70, 80, 89, 90).
    pub sm_version: u32,
}

#[cfg(feature = "jit-cuda")]
impl PtxDialect {
    pub fn new(sm_version: u32) -> Self {
        Self { sm_version }
    }
}

#[cfg(feature = "jit-cuda")]
impl GpuDialect for PtxDialect {
    fn emit_header(&self, out: &mut String) {
        let ptx_version = match self.sm_version {
            90.. => "8.0",
            80..=89 => "7.0",
            _ => "6.0",
        };
        writeln!(out, ".version {ptx_version}").unwrap();
        writeln!(out, ".target sm_{}", self.sm_version).unwrap();
        writeln!(out, ".address_size 64").unwrap();
        writeln!(out).unwrap();
    }

    fn emit_trace_op(
        &self,
        out: &mut String,
        op: &TraceOp,
        idx: usize,
        vars: &[String],
        tier: usize,
    ) -> String {
        let reg = format!("%t{}_{}", tier, idx);
        match op {
            TraceOp::Input(i) => {
                // Input bindings handled by caller — return placeholder.
                return format!("%input{i}");
            }
            TraceOp::Const(val) => {
                let bits = (*val as f32).to_bits();
                writeln!(out, "    mov.f32 {reg}, 0f{bits:08X};").unwrap();
            }
            TraceOp::Add(a, b) => {
                writeln!(out, "    add.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Sub(a, b) => {
                writeln!(out, "    sub.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Mul(a, b) => {
                writeln!(out, "    mul.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Div(a, b) => {
                writeln!(out, "    div.approx.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Fma(a, b, c) => {
                writeln!(out, "    fma.rn.f32 {reg}, {}, {}, {};", vars[*a as usize], vars[*b as usize], vars[*c as usize]).unwrap();
            }
            TraceOp::Neg(a) => {
                writeln!(out, "    neg.f32 {reg}, {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Abs(a) => {
                writeln!(out, "    abs.f32 {reg}, {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Exp(a) => {
                // exp(x) = exp2(x * log2(e)), log2(e) ~ 1.4426950408889634 = 0x3FB8AA3B
                writeln!(out, "    mul.f32 {reg}, {}, 0f3FB8AA3B;", vars[*a as usize]).unwrap();
                writeln!(out, "    ex2.approx.f32 {reg}, {reg};").unwrap();
            }
            TraceOp::Sqrt(a) => {
                writeln!(out, "    sqrt.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Rsqrt(a) => {
                writeln!(out, "    rsqrt.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Tanh(a) => {
                writeln!(out, "    tanh.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Recip(a) => {
                writeln!(out, "    rcp.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Log(a) => {
                // ln(x) = log2(x) * ln(2), ln(2) ~ 0.6931471805599453 = 0x3F317218
                writeln!(out, "    lg2.approx.f32 {reg}, {};", vars[*a as usize]).unwrap();
                writeln!(out, "    mul.f32 {reg}, {reg}, 0f3F317218;").unwrap();
            }
            TraceOp::Max(a, b) => {
                writeln!(out, "    max.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Min(a, b) => {
                writeln!(out, "    min.f32 {reg}, {}, {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
        }
        reg
    }

    fn warp_size(&self) -> u32 {
        32
    }

    fn capabilities(&self) -> GpuCapabilities {
        GpuCapabilities {
            has_matrix_unit: self.sm_version >= 70,
            has_injective_codegen: true,
        }
    }

    fn emit_kernel_start(
        &self,
        out: &mut String,
        name: &str,
        params: &[KernelParam],
        shared_mem_bytes: usize,
    ) {
        writeln!(out, ".visible .entry {name}(").unwrap();
        for (i, p) in params.iter().enumerate() {
            let ptx_ty = match p.ty {
                ParamType::FloatPtr => ".u64",
                ParamType::Uint => ".u32",
                ParamType::Float => ".f32",
            };
            let comma = if i + 1 < params.len() { "," } else { "" };
            writeln!(out, "    .param {ptx_ty} param_{}{comma}", p.name).unwrap();
        }
        writeln!(out, ")").unwrap();
        writeln!(out, "{{").unwrap();
        if shared_mem_bytes > 0 {
            let count = shared_mem_bytes / 4;
            writeln!(out, "    .shared .f32 smem[{count}];").unwrap();
        }
        writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
        writeln!(out, "    .reg .u32 %r<32>;").unwrap();
        writeln!(out, "    .reg .f32 %f<32>;").unwrap();
        writeln!(out, "    .reg .pred %p<8>;").unwrap();
        writeln!(out).unwrap();
        // Load params into registers
        for (i, p) in params.iter().enumerate() {
            match p.ty {
                ParamType::FloatPtr => {
                    writeln!(out, "    ld.param.u64 %rd{i}, [param_{}];", p.name).unwrap();
                }
                ParamType::Uint => {
                    writeln!(out, "    ld.param.u32 %r{i}, [param_{}];", p.name).unwrap();
                }
                ParamType::Float => {
                    writeln!(out, "    ld.param.f32 %f{i}, [param_{}];", p.name).unwrap();
                }
            }
        }
        writeln!(out).unwrap();
    }

    fn emit_kernel_end(&self, out: &mut String) {
        writeln!(out, "    ret;").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    fn global_tid_expr(&self) -> &'static str {
        // Caller must compute: mad.lo.u32 %r_tid, %ctaid.x, %ntid.x, %tid.x
        "%r_tid"
    }

    fn local_tid_expr(&self) -> &'static str { "%r_lid" }
    fn group_id_expr(&self) -> &'static str { "%r_gid" }

    fn emit_shared_decl(&self, out: &mut String, name: &str, count: usize) {
        writeln!(out, "    .shared .f32 {name}[{count}];").unwrap();
    }

    fn emit_barrier(&self, out: &mut String) {
        writeln!(out, "    bar.sync 0;").unwrap();
    }

    fn emit_for_start(
        &self,
        out: &mut String,
        var: &str,
        start: &str,
        limit: &str,
        _stride: &str,
        label: &str,
    ) {
        writeln!(out, "    mov.u32 {var}, {start};").unwrap();
        writeln!(out, "{label}_LOOP:").unwrap();
        writeln!(out, "    setp.ge.u32 %p0, {var}, {limit};").unwrap();
        writeln!(out, "    @%p0 bra {label}_DONE;").unwrap();
    }

    fn emit_for_end(&self, out: &mut String, var: &str, stride: &str, label: &str) {
        writeln!(out, "    add.u32 {var}, {var}, {stride};").unwrap();
        writeln!(out, "    bra {label}_LOOP;").unwrap();
        writeln!(out, "{label}_DONE:").unwrap();
    }

    fn emit_if_start(&self, out: &mut String, cond: &str, label: &str) {
        // cond is a PTX predicate expression like "setp.lt.u32 %p1, %r1, 128"
        writeln!(out, "    {cond};").unwrap();
        writeln!(out, "    @!%p1 bra {label}_SKIP;").unwrap();
    }

    fn emit_if_end(&self, out: &mut String, label: &str) {
        writeln!(out, "{label}_SKIP:").unwrap();
    }

    fn emit_float_decl(&self, out: &mut String, name: &str, expr: &str) {
        writeln!(out, "    mov.f32 {name}, {expr};").unwrap();
    }

    fn emit_float_assign(&self, out: &mut String, name: &str, expr: &str) {
        writeln!(out, "    mov.f32 {name}, {expr};").unwrap();
    }

    fn emit_global_load(&self, out: &mut String, dst: &str, ptr: &str, index: &str) {
        writeln!(out, "    mul.wide.u32 %rd15, {index}, 4;").unwrap();
        writeln!(out, "    add.u64 %rd15, {ptr}, %rd15;").unwrap();
        writeln!(out, "    ld.global.f32 {dst}, [%rd15];").unwrap();
    }

    fn emit_global_store(&self, out: &mut String, ptr: &str, index: &str, value: &str) {
        writeln!(out, "    mul.wide.u32 %rd15, {index}, 4;").unwrap();
        writeln!(out, "    add.u64 %rd15, {ptr}, %rd15;").unwrap();
        writeln!(out, "    st.global.f32 [%rd15], {value};").unwrap();
    }

    fn emit_shared_load(&self, out: &mut String, dst: &str, array: &str, index: &str) {
        writeln!(out, "    mul.wide.u32 %rd15, {index}, 4;").unwrap();
        writeln!(out, "    mov.u64 %rd14, {array};").unwrap();
        writeln!(out, "    add.u64 %rd15, %rd14, %rd15;").unwrap();
        writeln!(out, "    ld.shared.f32 {dst}, [%rd15];").unwrap();
    }

    fn emit_shared_store(&self, out: &mut String, array: &str, index: &str, value: &str) {
        writeln!(out, "    mul.wide.u32 %rd15, {index}, 4;").unwrap();
        writeln!(out, "    mov.u64 %rd14, {array};").unwrap();
        writeln!(out, "    add.u64 %rd15, %rd14, %rd15;").unwrap();
        writeln!(out, "    st.shared.f32 [%rd15], {value};").unwrap();
    }

    fn emit_bounds_check_return(&self, out: &mut String, var: &str, limit: &str, label: &str) {
        writeln!(out, "    setp.ge.u32 %p0, {var}, {limit};").unwrap();
        writeln!(out, "    @%p0 bra {label}_DONE;").unwrap();
    }

    fn emit_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]) {
        super::kernel_builder::build_elementwise_kernel(self, out, name, body);
    }

    fn emit_binary_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]) {
        super::kernel_builder::build_binary_elementwise_kernel(self, out, name, body);
    }

    fn emit_gemm_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
    ) -> Result<(), String> {
        use crate::compiler::codegen::ptx::*;
        match op_kind {
            OpKind::QuantGemm { m, n, k, .. } => {
                Err(format!(
                    "PtxDialect: fused QuantGemm not yet implemented ({}x{}x{}), use Dequantize + Gemm",
                    m, n, k
                ))
            }
            OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => {
                if self.sm_version >= 89 {
                    emit_gemm_tc_sm89_ptx(out, name, *m, *n, *k);
                } else if self.sm_version >= 80 {
                    emit_gemm_tc_sm80_ptx(out, name, *m, *n, *k);
                } else if self.sm_version >= 70 {
                    emit_gemm_tc_sm70_ptx(out, name, *m, *n, *k);
                } else {
                    // Fallback to unified tiled GEMM builder for older GPUs
                    let has_bias = matches!(op_kind, OpKind::GemmBias { .. });
                    return super::kernel_builder::build_gemm_tiled_kernel(
                        self, out, name, has_bias, &[],
                    );
                }
                Ok(())
            }
            OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                super::kernel_builder::build_mha_kernel(self, out, name, *seq_len, *num_heads, *head_dim);
                Ok(())
            }
            _ => Err(format!("PtxDialect: unsupported Gemm-pattern op {:?}", op_kind)),
        }
    }

    fn emit_mha_kernel(
        &self,
        out: &mut String,
        name: &str,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        super::kernel_builder::build_mha_kernel(self, out, name, seq_len, num_heads, head_dim);
    }

    fn emit_rope_kernel(
        &self,
        out: &mut String,
        name: &str,
        head_dim: usize,
        theta: f64,
    ) {
        crate::compiler::codegen::ptx::emit_rope_kernel_ptx(out, name, head_dim, theta);
    }

    fn emit_reduction_kernel(
        &self,
        out: &mut String,
        name: &str,
        identity: f64,
        combine: &[TraceOp],
    ) -> Result<(), String> {
        crate::compiler::codegen::ptx::emit_reduction_kernel_ptx(
            out, name, identity, combine,
        );
        Ok(())
    }

    fn emit_injective_kernel(
        &self,
        out: &mut String,
        name: &str,
        body: &[TraceOp],
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<(), String> {
        crate::compiler::codegen::ptx::emit_injective_kernel_ptx(
            out, name, body, num_inputs, num_outputs,
        );
        Ok(())
    }

    fn emit_normlike_kernel(
        &self,
        out: &mut String,
        name: &str,
        reduce: &[TraceOp],
        finalize: &[TraceOp],
        transform: &[TraceOp],
        has_weight: bool,
        has_bias: bool,
        eps_override: Option<f32>,
    ) {
        super::kernel_builder::build_normlike_kernel(
            self, out, name, reduce, finalize, transform,
            has_weight, has_bias, eps_override.map(|e| e as f64), self.default_block_size(),
        );
    }

    fn emit_softmax_kernel(&self, out: &mut String, name: &str) {
        super::kernel_builder::build_softmax_kernel(self, out, name, self.default_block_size());
    }

    fn emit_meanpool_kernel(&self, out: &mut String, name: &str, seq_len: usize, hidden: usize) {
        super::kernel_builder::build_meanpool_kernel(self, out, name, seq_len, hidden);
    }

    fn emit_dequantize_kernel(
        &self,
        out: &mut String,
        name: &str,
        num_elements: usize,
        block_size: usize,
        bits: u8,
    ) {
        super::kernel_builder::build_dequantize_kernel(
            self, out, name, num_elements, block_size, bits.into(),
        );
    }

    fn emit_gemm_epilogue_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
        epilogue_bodies: &[&[TraceOp]],
    ) -> Result<(), String> {
        super::kernel_builder::build_gemm_epilogue_kernel(
            self, out, name, op_kind, epilogue_bodies,
        )
    }

    fn emit_tile_level_fusion_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        tile_rows: usize,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        shared_mem_budget: usize,
    ) -> Result<(), String> {
        ptx_tile_level_fusion(
            self, out, name, norm_op, gemm_op, tile_rows,
            norm_reduce, norm_finalize, norm_transform, shared_mem_budget,
        )
    }

    fn emit_compute_root_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        shared_mem_budget: usize,
    ) -> Result<(), String> {
        ptx_compute_root(
            self, out, name, norm_op, gemm_op,
            norm_reduce, norm_finalize, norm_transform, shared_mem_budget,
        )
    }
}

/// PTX: extract norm eps and weight flag from OpKind.
#[cfg(feature = "jit-cuda")]
fn ptx_norm_params(norm_op: &OpKind) -> (f32, bool) {
    let eps = match norm_op {
        OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
        _ => 1e-6,
    };
    let has_w = matches!(norm_op, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. });
    (eps, has_w)
}

// ── PTX TileLevelFusion ─────────────────────────────────────────────────────

/// Single fused kernel: Phase 1 cooperative norm → shared memory, Phase 2
/// tiled GEMM reads from shared memory. Mirrors HipDialect logic in PTX ISA.
#[cfg(feature = "jit-cuda")]
#[allow(clippy::too_many_arguments)]
fn ptx_tile_level_fusion(
    dialect: &PtxDialect,
    out: &mut String,
    name: &str,
    norm_op: &OpKind,
    gemm_op: &OpKind,
    tile_rows: usize,
    norm_reduce: &[TraceOp],
    norm_finalize: &[TraceOp],
    norm_transform: &[TraceOp],
    shared_mem_budget: usize,
) -> Result<(), String> {
    let (_m, _n, k) = match gemm_op {
        OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
        _ => return Err(format!(
            "TileLevelFusion anchor must be Gemm/GemmBias, got {:?}", gemm_op
        )),
    };
    let (eps, has_nw) = ptx_norm_params(norm_op);
    let bs = 256usize;
    let ntf = tile_rows * k;
    if ntf * 4 > shared_mem_budget {
        return Err(format!(
            "TileLevelFusion: norm tile ({} B) > budget ({} B)", ntf * 4, shared_mem_budget
        ));
    }
    let eps_bits = eps.to_bits();
    let tile_k = 16usize;
    let btf = tile_k * 16;

    dialect.emit_header(out);
    // -- kernel signature --
    writeln!(out, ".visible .entry {name}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    if has_nw { writeln!(out, "    .param .u64 param_nw,").unwrap(); }
    writeln!(out, "    .param .u64 param_weight,").unwrap();
    writeln!(out, "    .param .u64 param_output,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_N,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<32>;").unwrap();
    writeln!(out, "    .reg .u32 %r<64>;").unwrap();
    writeln!(out, "    .reg .f32 %f<64>;").unwrap();
    writeln!(out, "    .reg .pred %p<16>;").unwrap();
    writeln!(out, "    .shared .f32 norm_tile[{ntf}];").unwrap();
    writeln!(out, "    .shared .f32 b_tile[{btf}];").unwrap();
    writeln!(out).unwrap();
    // -- load params --
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    let (nw_rd, w_rd, o_rd) = if has_nw {
        writeln!(out, "    ld.param.u64 %rd1, [param_nw];").unwrap();
        writeln!(out, "    ld.param.u64 %rd2, [param_weight];").unwrap();
        writeln!(out, "    ld.param.u64 %rd3, [param_output];").unwrap();
        (1u32, 2u32, 3u32)
    } else {
        writeln!(out, "    ld.param.u64 %rd1, [param_weight];").unwrap();
        writeln!(out, "    ld.param.u64 %rd2, [param_output];").unwrap();
        (0u32, 1u32, 2u32)
    };
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_K];").unwrap();
    writeln!(out).unwrap();

    // block_row = blockIdx.x * TILE_ROWS; tid; mc_cur = min(TILE_ROWS, M-block_row)
    writeln!(out, "    mov.u32 %r3, %ctaid.x;").unwrap();
    writeln!(out, "    mul.lo.u32 %r3, %r3, {tile_rows};").unwrap();
    writeln!(out, "    mov.u32 %r4, %tid.x;").unwrap();
    writeln!(out, "    sub.u32 %r5, %r0, %r3;").unwrap();
    writeln!(out, "    min.u32 %r5, {tile_rows}, %r5;").unwrap();
    writeln!(out).unwrap();

    // ── Phase 1: Cooperative norm ──
    writeln!(out, "    mov.u32 %r6, %r4;").unwrap();
    writeln!(out, "{name}_P1:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r6, %r5;").unwrap();
    writeln!(out, "    @%p0 bra {name}_P1D;").unwrap();
    // reduce: sum_sq(%f0) over c(%r7)
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
    writeln!(out, "    mov.u32 %r7, 0;").unwrap();
    writeln!(out, "{name}_RL:").unwrap();
    writeln!(out, "    setp.ge.u32 %p1, %r7, %r2;").unwrap();
    writeln!(out, "    @%p1 bra {name}_RD;").unwrap();
    // x = input[(block_row+r)*K+c]
    writeln!(out, "    add.u32 %r8, %r3, %r6;").unwrap();
    writeln!(out, "    mul.lo.u32 %r8, %r8, %r2;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, %r7;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r8, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd0, %rd10;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd10];").unwrap();
    let rb = vec!["%f1".into(), "%f0".into()];
    let rr = emit_trace_body(dialect, out, norm_reduce, 10, &rb);
    writeln!(out, "    add.f32 %f0, %f0, {rr};").unwrap();
    writeln!(out, "    add.u32 %r7, %r7, 1;").unwrap();
    writeln!(out, "    bra {name}_RL;").unwrap();
    writeln!(out, "{name}_RD:").unwrap();
    // finalize: norm_mean = sum_sq/K
    writeln!(out, "    cvt.rn.f32.u32 %f2, %r2;").unwrap();
    writeln!(out, "    div.approx.f32 %f2, %f0, %f2;").unwrap();
    let fb = vec!["%f2".into()];
    let inv = emit_trace_body(dialect, out, norm_finalize, 11, &fb);
    // transform loop: norm_tile[r*K+c] = transform(x, inv[, w])
    writeln!(out, "    mov.u32 %r7, 0;").unwrap();
    writeln!(out, "{name}_XL:").unwrap();
    writeln!(out, "    setp.ge.u32 %p2, %r7, %r2;").unwrap();
    writeln!(out, "    @%p2 bra {name}_XD;").unwrap();
    writeln!(out, "    add.u32 %r8, %r3, %r6;").unwrap();
    writeln!(out, "    mul.lo.u32 %r8, %r8, %r2;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, %r7;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd10, %r8, 4;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd0, %rd10;").unwrap();
    writeln!(out, "    ld.global.f32 %f3, [%rd10];").unwrap();
    let normed = if has_nw {
        writeln!(out, "    mul.wide.u32 %rd11, %r7, 4;").unwrap();
        writeln!(out, "    add.u64 %rd11, %rd{nw_rd}, %rd11;").unwrap();
        writeln!(out, "    ld.global.f32 %f4, [%rd11];").unwrap();
        emit_trace_body(dialect, out, norm_transform, 12,
            &["%f3".into(), inv.clone(), "%f4".into()])
    } else {
        emit_trace_body(dialect, out, norm_transform, 12,
            &["%f3".into(), inv.clone()])
    };
    // st.shared norm_tile[r*K+c]
    writeln!(out, "    mul.lo.u32 %r9, %r6, %r2;").unwrap();
    writeln!(out, "    add.u32 %r9, %r9, %r7;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd12, %r9, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd13, norm_tile;").unwrap();
    writeln!(out, "    add.u64 %rd12, %rd13, %rd12;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd12], {normed};").unwrap();
    writeln!(out, "    add.u32 %r7, %r7, 1;").unwrap();
    writeln!(out, "    bra {name}_XL;").unwrap();
    writeln!(out, "{name}_XD:").unwrap();
    writeln!(out, "    add.u32 %r6, %r6, {bs};").unwrap();
    writeln!(out, "    bra {name}_P1;").unwrap();
    writeln!(out, "{name}_P1D:").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out).unwrap();

    // ── Phase 2: Tiled GEMM from shared memory ──
    // ty = tid / 16; tx = tid % 16
    writeln!(out, "    shr.u32 %r10, %r4, 4;").unwrap();
    writeln!(out, "    and.b32 %r11, %r4, 15;").unwrap();
    // col_block loop (%r12)
    writeln!(out, "    mov.u32 %r12, 0;").unwrap();
    writeln!(out, "{name}_CB:").unwrap();
    writeln!(out, "    setp.ge.u32 %p3, %r12, %r1;").unwrap();
    writeln!(out, "    @%p3 bra {name}_CBD;").unwrap();
    // row loop: row = ty; row < mc_cur; row += bs/16
    let rows_per_iter = bs / 16;
    writeln!(out, "    mov.u32 %r13, %r10;").unwrap();
    writeln!(out, "{name}_RW:").unwrap();
    writeln!(out, "    setp.ge.u32 %p4, %r13, %r5;").unwrap();
    writeln!(out, "    @%p4 bra {name}_RWD;").unwrap();
    // acc = 0
    writeln!(out, "    mov.f32 %f10, 0f00000000;").unwrap();
    // pc loop (%r14): tile over K
    writeln!(out, "    mov.u32 %r14, 0;").unwrap();
    writeln!(out, "{name}_PC:").unwrap();
    writeln!(out, "    setp.ge.u32 %p5, %r14, %r2;").unwrap();
    writeln!(out, "    @%p5 bra {name}_PCD;").unwrap();
    // Load B tile: if tid < TILE_K*16
    writeln!(out, "    setp.ge.u32 %p6, %r4, {btf};").unwrap();
    writeln!(out, "    @%p6 bra {name}_BLD;").unwrap();
    // bk = tid/16, bn = tid%16
    writeln!(out, "    shr.u32 %r15, %r4, 4;").unwrap();
    writeln!(out, "    and.b32 %r16, %r4, 15;").unwrap();
    // gk = pc + bk, gn = col_block + bn
    writeln!(out, "    add.u32 %r17, %r14, %r15;").unwrap();
    writeln!(out, "    add.u32 %r18, %r12, %r16;").unwrap();
    // bounds check: gk < K && gn < N
    writeln!(out, "    setp.lt.u32 %p7, %r17, %r2;").unwrap();
    writeln!(out, "    setp.lt.u32 %p8, %r18, %r1;").unwrap();
    writeln!(out, "    and.pred %p7, %p7, %p8;").unwrap();
    writeln!(out, "    @!%p7 bra {name}_BZ;").unwrap();
    // b_tile[bk*16+bn] = weight[gk*N+gn]
    writeln!(out, "    mul.lo.u32 %r19, %r17, %r1;").unwrap();
    writeln!(out, "    add.u32 %r19, %r19, %r18;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd14, %r19, 4;").unwrap();
    writeln!(out, "    add.u64 %rd14, %rd{w_rd}, %rd14;").unwrap();
    writeln!(out, "    ld.global.f32 %f11, [%rd14];").unwrap();
    writeln!(out, "    bra {name}_BST;").unwrap();
    writeln!(out, "{name}_BZ:").unwrap();
    writeln!(out, "    mov.f32 %f11, 0f00000000;").unwrap();
    writeln!(out, "{name}_BST:").unwrap();
    // store to b_tile shared
    writeln!(out, "    shl.b32 %r20, %r15, 4;").unwrap();
    writeln!(out, "    add.u32 %r20, %r20, %r16;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd15, %r20, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd16, b_tile;").unwrap();
    writeln!(out, "    add.u64 %rd15, %rd16, %rd15;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd15], %f11;").unwrap();
    writeln!(out, "{name}_BLD:").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();

    // inner product: kk = 0..TILE_K
    writeln!(out, "    mov.u32 %r21, 0;").unwrap();
    writeln!(out, "{name}_KK:").unwrap();
    writeln!(out, "    setp.ge.u32 %p9, %r21, {tile_k};").unwrap();
    writeln!(out, "    @%p9 bra {name}_KKD;").unwrap();
    // gk = pc + kk; bounds check
    writeln!(out, "    add.u32 %r22, %r14, %r21;").unwrap();
    writeln!(out, "    setp.ge.u32 %p10, %r22, %r2;").unwrap();
    writeln!(out, "    @%p10 bra {name}_AZ;").unwrap();
    // a_val = norm_tile[row*K+gk]
    writeln!(out, "    mul.lo.u32 %r23, %r13, %r2;").unwrap();
    writeln!(out, "    add.u32 %r23, %r23, %r22;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd17, %r23, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd18, norm_tile;").unwrap();
    writeln!(out, "    add.u64 %rd17, %rd18, %rd17;").unwrap();
    writeln!(out, "    ld.shared.f32 %f12, [%rd17];").unwrap();
    writeln!(out, "    bra {name}_AFMA;").unwrap();
    writeln!(out, "{name}_AZ:").unwrap();
    writeln!(out, "    mov.f32 %f12, 0f00000000;").unwrap();
    writeln!(out, "{name}_AFMA:").unwrap();
    // b_val = b_tile[kk*16+tx]
    writeln!(out, "    shl.b32 %r24, %r21, 4;").unwrap();
    writeln!(out, "    add.u32 %r24, %r24, %r11;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd19, %r24, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd20, b_tile;").unwrap();
    writeln!(out, "    add.u64 %rd19, %rd20, %rd19;").unwrap();
    writeln!(out, "    ld.shared.f32 %f13, [%rd19];").unwrap();
    writeln!(out, "    fma.rn.f32 %f10, %f12, %f13, %f10;").unwrap();
    writeln!(out, "    add.u32 %r21, %r21, 1;").unwrap();
    writeln!(out, "    bra {name}_KK;").unwrap();
    writeln!(out, "{name}_KKD:").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    // pc += TILE_K
    writeln!(out, "    add.u32 %r14, %r14, {tile_k};").unwrap();
    writeln!(out, "    bra {name}_PC;").unwrap();
    writeln!(out, "{name}_PCD:").unwrap();

    // Store output[out_row*N+out_col]
    writeln!(out, "    add.u32 %r25, %r3, %r13;").unwrap();
    writeln!(out, "    add.u32 %r26, %r12, %r11;").unwrap();
    writeln!(out, "    setp.lt.u32 %p11, %r25, %r0;").unwrap();
    writeln!(out, "    setp.lt.u32 %p12, %r26, %r1;").unwrap();
    writeln!(out, "    and.pred %p11, %p11, %p12;").unwrap();
    writeln!(out, "    @!%p11 bra {name}_NST;").unwrap();
    writeln!(out, "    mul.lo.u32 %r27, %r25, %r1;").unwrap();
    writeln!(out, "    add.u32 %r27, %r27, %r26;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd21, %r27, 4;").unwrap();
    writeln!(out, "    add.u64 %rd21, %rd{o_rd}, %rd21;").unwrap();
    writeln!(out, "    st.global.f32 [%rd21], %f10;").unwrap();
    writeln!(out, "{name}_NST:").unwrap();
    // row += bs/16
    writeln!(out, "    add.u32 %r13, %r13, {rows_per_iter};").unwrap();
    writeln!(out, "    bra {name}_RW;").unwrap();
    writeln!(out, "{name}_RWD:").unwrap();
    // col_block += 16
    writeln!(out, "    add.u32 %r12, %r12, 16;").unwrap();
    writeln!(out, "    bra {name}_CB;").unwrap();
    writeln!(out, "{name}_CBD:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    Ok(())
}

// ── PTX ComputeRoot ─────────────────────────────────────────────────────────

/// Two-kernel approach: Kernel 1 computes full norm to global memory,
/// Kernel 2 runs tiled GEMM reading from that buffer.
#[cfg(feature = "jit-cuda")]
#[allow(clippy::too_many_arguments)]
fn ptx_compute_root(
    dialect: &PtxDialect,
    out: &mut String,
    name: &str,
    norm_op: &OpKind,
    gemm_op: &OpKind,
    norm_reduce: &[TraceOp],
    norm_finalize: &[TraceOp],
    norm_transform: &[TraceOp],
    _shared_mem_budget: usize,
) -> Result<(), String> {
    let (_m, _n, _k) = match gemm_op {
        OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
        _ => return Err(format!(
            "ComputeRoot anchor must be Gemm/GemmBias, got {:?}", gemm_op
        )),
    };
    let (eps, has_nw) = ptx_norm_params(norm_op);
    let _eps_bits = eps.to_bits();

    // ── Kernel 1: Full norm ──
    dialect.emit_header(out);
    let nn = format!("{name}_norm");
    writeln!(out, ".visible .entry {nn}(").unwrap();
    writeln!(out, "    .param .u64 param_input,").unwrap();
    if has_nw { writeln!(out, "    .param .u64 param_nw,").unwrap(); }
    writeln!(out, "    .param .u64 param_normed,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
    writeln!(out, "    .reg .u32 %r<32>;").unwrap();
    writeln!(out, "    .reg .f32 %f<32>;").unwrap();
    writeln!(out, "    .reg .pred %p<8>;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    ld.param.u64 %rd0, [param_input];").unwrap();
    let (nw_rd, no_rd) = if has_nw {
        writeln!(out, "    ld.param.u64 %rd1, [param_nw];").unwrap();
        writeln!(out, "    ld.param.u64 %rd2, [param_normed];").unwrap();
        (1u32, 2u32)
    } else {
        writeln!(out, "    ld.param.u64 %rd1, [param_normed];").unwrap();
        (0u32, 1u32)
    };
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_K];").unwrap();
    // row = blockIdx.x * blockDim.x + threadIdx.x
    writeln!(out, "    mov.u32 %r2, %ctaid.x;").unwrap();
    writeln!(out, "    mov.u32 %r3, %ntid.x;").unwrap();
    writeln!(out, "    mul.lo.u32 %r2, %r2, %r3;").unwrap();
    writeln!(out, "    mov.u32 %r3, %tid.x;").unwrap();
    writeln!(out, "    add.u32 %r2, %r2, %r3;").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r2, %r0;").unwrap();
    writeln!(out, "    @%p0 bra {nn}_RET;").unwrap();
    writeln!(out).unwrap();
    // Reduce
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
    writeln!(out, "    mov.u32 %r4, 0;").unwrap();
    writeln!(out, "{nn}_RL:").unwrap();
    writeln!(out, "    setp.ge.u32 %p1, %r4, %r1;").unwrap();
    writeln!(out, "    @%p1 bra {nn}_RD;").unwrap();
    writeln!(out, "    mul.lo.u32 %r5, %r2, %r1;").unwrap();
    writeln!(out, "    add.u32 %r5, %r5, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd5, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd0, %rd5;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd5];").unwrap();
    let rb = vec!["%f1".into(), "%f0".into()];
    let rr = emit_trace_body(dialect, out, norm_reduce, 10, &rb);
    writeln!(out, "    add.f32 %f0, %f0, {rr};").unwrap();
    writeln!(out, "    add.u32 %r4, %r4, 1;").unwrap();
    writeln!(out, "    bra {nn}_RL;").unwrap();
    writeln!(out, "{nn}_RD:").unwrap();
    // Finalize
    writeln!(out, "    cvt.rn.f32.u32 %f2, %r1;").unwrap();
    writeln!(out, "    div.approx.f32 %f2, %f0, %f2;").unwrap();
    let fb = vec!["%f2".into()];
    let inv = emit_trace_body(dialect, out, norm_finalize, 11, &fb);
    // Transform
    writeln!(out, "    mov.u32 %r4, 0;").unwrap();
    writeln!(out, "{nn}_XL:").unwrap();
    writeln!(out, "    setp.ge.u32 %p2, %r4, %r1;").unwrap();
    writeln!(out, "    @%p2 bra {nn}_RET;").unwrap();
    writeln!(out, "    mul.lo.u32 %r5, %r2, %r1;").unwrap();
    writeln!(out, "    add.u32 %r5, %r5, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd5, %r5, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd0, %rd5;").unwrap();
    writeln!(out, "    ld.global.f32 %f3, [%rd5];").unwrap();
    let normed = if has_nw {
        writeln!(out, "    mul.wide.u32 %rd6, %r4, 4;").unwrap();
        writeln!(out, "    add.u64 %rd6, %rd{nw_rd}, %rd6;").unwrap();
        writeln!(out, "    ld.global.f32 %f4, [%rd6];").unwrap();
        emit_trace_body(dialect, out, norm_transform, 12,
            &["%f3".into(), inv.clone(), "%f4".into()])
    } else {
        emit_trace_body(dialect, out, norm_transform, 12,
            &["%f3".into(), inv.clone()])
    };
    writeln!(out, "    mul.lo.u32 %r6, %r2, %r1;").unwrap();
    writeln!(out, "    add.u32 %r6, %r6, %r4;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd7, %r6, 4;").unwrap();
    writeln!(out, "    add.u64 %rd7, %rd{no_rd}, %rd7;").unwrap();
    writeln!(out, "    st.global.f32 [%rd7], {normed};").unwrap();
    writeln!(out, "    add.u32 %r4, %r4, 1;").unwrap();
    writeln!(out, "    bra {nn}_XL;").unwrap();
    writeln!(out, "{nn}_RET:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // ── Kernel 2: Tiled GEMM (A=normed, B=weight, C=output) ──
    ptx_compute_root_gemm(out, name);

    Ok(())
}

/// Emit the GEMM kernel for ComputeRoot (standard 16x16 tiled GEMM in PTX).
#[cfg(feature = "jit-cuda")]
fn ptx_compute_root_gemm(out: &mut String, name: &str) {
    let gn = format!("{name}_gemm");
    let ts = 16usize;
    writeln!(out, ".visible .entry {gn}(").unwrap();
    writeln!(out, "    .param .u64 param_A,").unwrap();
    writeln!(out, "    .param .u64 param_B,").unwrap();
    writeln!(out, "    .param .u64 param_C,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_N,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
    writeln!(out, "    .reg .u32 %r<32>;").unwrap();
    writeln!(out, "    .reg .f32 %f<16>;").unwrap();
    writeln!(out, "    .reg .pred %p<8>;").unwrap();
    writeln!(out, "    .shared .f32 As[{ts} * {ts}];").unwrap();
    writeln!(out, "    .shared .f32 Bs[{ts} * {ts}];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    ld.param.u64 %rd0, [param_A];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_B];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_C];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_K];").unwrap();
    // row = blockIdx.y*TS + threadIdx.y; col = blockIdx.x*TS + threadIdx.x
    writeln!(out, "    mov.u32 %r3, %ctaid.y;").unwrap();
    writeln!(out, "    shl.b32 %r3, %r3, 4;").unwrap();
    writeln!(out, "    mov.u32 %r4, %tid.y;").unwrap();
    writeln!(out, "    add.u32 %r3, %r3, %r4;").unwrap();
    writeln!(out, "    mov.u32 %r5, %ctaid.x;").unwrap();
    writeln!(out, "    shl.b32 %r5, %r5, 4;").unwrap();
    writeln!(out, "    mov.u32 %r6, %tid.x;").unwrap();
    writeln!(out, "    add.u32 %r5, %r5, %r6;").unwrap();
    writeln!(out, "    mov.f32 %f0, 0f00000000;").unwrap();
    // num_tiles
    writeln!(out, "    add.u32 %r7, %r2, {};", ts - 1).unwrap();
    writeln!(out, "    shr.u32 %r7, %r7, 4;").unwrap();
    writeln!(out, "    mov.u32 %r8, 0;").unwrap();
    writeln!(out, "{gn}_TL:").unwrap();
    writeln!(out, "    setp.ge.u32 %p0, %r8, %r7;").unwrap();
    writeln!(out, "    @%p0 bra {gn}_TD;").unwrap();
    // ak = t*TS+tx, bk = t*TS+ty
    writeln!(out, "    shl.b32 %r9, %r8, 4;").unwrap();
    writeln!(out, "    add.u32 %r10, %r9, %r6;").unwrap();
    writeln!(out, "    add.u32 %r11, %r9, %r4;").unwrap();
    // Load As
    writeln!(out, "    setp.lt.u32 %p1, %r3, %r0;").unwrap();
    writeln!(out, "    setp.lt.u32 %p2, %r10, %r2;").unwrap();
    writeln!(out, "    and.pred %p1, %p1, %p2;").unwrap();
    writeln!(out, "    @!%p1 bra {gn}_AZ;").unwrap();
    writeln!(out, "    mul.lo.u32 %r12, %r3, %r2;").unwrap();
    writeln!(out, "    add.u32 %r12, %r12, %r10;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd5, %r12, 4;").unwrap();
    writeln!(out, "    add.u64 %rd5, %rd0, %rd5;").unwrap();
    writeln!(out, "    ld.global.f32 %f1, [%rd5];").unwrap();
    writeln!(out, "    bra {gn}_AST;").unwrap();
    writeln!(out, "{gn}_AZ:").unwrap();
    writeln!(out, "    mov.f32 %f1, 0f00000000;").unwrap();
    writeln!(out, "{gn}_AST:").unwrap();
    // ty*TS+tx index for shared
    writeln!(out, "    shl.b32 %r13, %r4, 4;").unwrap();
    writeln!(out, "    add.u32 %r13, %r13, %r6;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd6, %r13, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd7, As;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd7, %rd6;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd6], %f1;").unwrap();
    // Load Bs
    writeln!(out, "    setp.lt.u32 %p3, %r11, %r2;").unwrap();
    writeln!(out, "    setp.lt.u32 %p4, %r5, %r1;").unwrap();
    writeln!(out, "    and.pred %p3, %p3, %p4;").unwrap();
    writeln!(out, "    @!%p3 bra {gn}_BZ;").unwrap();
    writeln!(out, "    mul.lo.u32 %r14, %r11, %r1;").unwrap();
    writeln!(out, "    add.u32 %r14, %r14, %r5;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd8, %r14, 4;").unwrap();
    writeln!(out, "    add.u64 %rd8, %rd1, %rd8;").unwrap();
    writeln!(out, "    ld.global.f32 %f2, [%rd8];").unwrap();
    writeln!(out, "    bra {gn}_BST;").unwrap();
    writeln!(out, "{gn}_BZ:").unwrap();
    writeln!(out, "    mov.f32 %f2, 0f00000000;").unwrap();
    writeln!(out, "{gn}_BST:").unwrap();
    writeln!(out, "    mul.wide.u32 %rd9, %r13, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd10, Bs;").unwrap();
    writeln!(out, "    add.u64 %rd9, %rd10, %rd9;").unwrap();
    writeln!(out, "    st.shared.f32 [%rd9], %f2;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    // Inner product kk=0..TS
    writeln!(out, "    mov.u32 %r15, 0;").unwrap();
    writeln!(out, "{gn}_KK:").unwrap();
    writeln!(out, "    setp.ge.u32 %p5, %r15, {ts};").unwrap();
    writeln!(out, "    @%p5 bra {gn}_KKD;").unwrap();
    writeln!(out, "    shl.b32 %r16, %r4, 4;").unwrap();
    writeln!(out, "    add.u32 %r16, %r16, %r15;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd11, %r16, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd12, As;").unwrap();
    writeln!(out, "    add.u64 %rd11, %rd12, %rd11;").unwrap();
    writeln!(out, "    ld.shared.f32 %f3, [%rd11];").unwrap();
    writeln!(out, "    shl.b32 %r17, %r15, 4;").unwrap();
    writeln!(out, "    add.u32 %r17, %r17, %r6;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd13, %r17, 4;").unwrap();
    writeln!(out, "    mov.u64 %rd14, Bs;").unwrap();
    writeln!(out, "    add.u64 %rd13, %rd14, %rd13;").unwrap();
    writeln!(out, "    ld.shared.f32 %f4, [%rd13];").unwrap();
    writeln!(out, "    fma.rn.f32 %f0, %f3, %f4, %f0;").unwrap();
    writeln!(out, "    add.u32 %r15, %r15, 1;").unwrap();
    writeln!(out, "    bra {gn}_KK;").unwrap();
    writeln!(out, "{gn}_KKD:").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out, "    add.u32 %r8, %r8, 1;").unwrap();
    writeln!(out, "    bra {gn}_TL;").unwrap();
    writeln!(out, "{gn}_TD:").unwrap();
    // Store C
    writeln!(out, "    setp.lt.u32 %p6, %r3, %r0;").unwrap();
    writeln!(out, "    setp.lt.u32 %p7, %r5, %r1;").unwrap();
    writeln!(out, "    and.pred %p6, %p6, %p7;").unwrap();
    writeln!(out, "    @!%p6 bra {gn}_RET;").unwrap();
    writeln!(out, "    mul.lo.u32 %r18, %r3, %r1;").unwrap();
    writeln!(out, "    add.u32 %r18, %r18, %r5;").unwrap();
    writeln!(out, "    mul.wide.u32 %rd15, %r18, 4;").unwrap();
    writeln!(out, "    add.u64 %rd15, %rd2, %rd15;").unwrap();
    writeln!(out, "    st.global.f32 [%rd15], %f0;").unwrap();
    writeln!(out, "{gn}_RET:").unwrap();
    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

// ── HipDialect ──────────────────────────────────────────────────────────────

/// HIP C++ dialect — emits HIP C++ source code for AMD GPUs.
#[cfg(feature = "jit-hip")]
pub struct HipDialect {
    /// Target GFX architecture (e.g. 908 = gfx908, 1100 = gfx1100).
    pub gfx_arch: u32,
}

#[cfg(feature = "jit-hip")]
impl HipDialect {
    pub fn new(gfx_arch: u32) -> Self {
        Self { gfx_arch }
    }
}

#[cfg(feature = "jit-hip")]
impl GpuDialect for HipDialect {
    fn emit_header(&self, out: &mut String) {
        writeln!(out, "#include <hip/hip_runtime.h>").unwrap();
        writeln!(out).unwrap();
    }

    fn emit_trace_op(
        &self,
        out: &mut String,
        op: &TraceOp,
        idx: usize,
        vars: &[String],
        tier: usize,
    ) -> String {
        let dst = format!("t{}_{}", tier, idx);
        match op {
            TraceOp::Input(i) => {
                let src = &vars[*i as usize];
                writeln!(out, "    float {dst} = {src};").unwrap();
            }
            TraceOp::Const(v) => {
                let bits = (*v as f32).to_bits();
                writeln!(out, "    float {dst} = __uint_as_float(0x{bits:08X}u);").unwrap();
            }
            TraceOp::Add(a, b) => {
                writeln!(out, "    float {dst} = {} + {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Sub(a, b) => {
                writeln!(out, "    float {dst} = {} - {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Mul(a, b) => {
                writeln!(out, "    float {dst} = {} * {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Div(a, b) => {
                writeln!(out, "    float {dst} = {} / {};", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Fma(a, b, c) => {
                writeln!(out, "    float {dst} = fmaf({}, {}, {});", vars[*a as usize], vars[*b as usize], vars[*c as usize]).unwrap();
            }
            TraceOp::Neg(a) => {
                writeln!(out, "    float {dst} = -{};", vars[*a as usize]).unwrap();
            }
            TraceOp::Abs(a) => {
                writeln!(out, "    float {dst} = fabsf({});", vars[*a as usize]).unwrap();
            }
            TraceOp::Exp(a) => {
                writeln!(out, "    float {dst} = expf({});", vars[*a as usize]).unwrap();
            }
            TraceOp::Sqrt(a) => {
                writeln!(out, "    float {dst} = sqrtf({});", vars[*a as usize]).unwrap();
            }
            TraceOp::Rsqrt(a) => {
                writeln!(out, "    float {dst} = rsqrtf({});", vars[*a as usize]).unwrap();
            }
            TraceOp::Tanh(a) => {
                writeln!(out, "    float {dst} = tanhf({});", vars[*a as usize]).unwrap();
            }
            TraceOp::Recip(a) => {
                writeln!(out, "    float {dst} = 1.0f / {};", vars[*a as usize]).unwrap();
            }
            TraceOp::Log(a) => {
                writeln!(out, "    float {dst} = logf({});", vars[*a as usize]).unwrap();
            }
            TraceOp::Max(a, b) => {
                writeln!(out, "    float {dst} = fmaxf({}, {});", vars[*a as usize], vars[*b as usize]).unwrap();
            }
            TraceOp::Min(a, b) => {
                writeln!(out, "    float {dst} = fminf({}, {});", vars[*a as usize], vars[*b as usize]).unwrap();
            }
        }
        dst
    }

    fn warp_size(&self) -> u32 {
        // RDNA (gfx10+) uses wave32, CDNA (gfx9xx) uses wave64
        if self.gfx_arch >= 1000 { 32 } else { 64 }
    }

    fn capabilities(&self) -> GpuCapabilities {
        GpuCapabilities {
            has_matrix_unit: self.gfx_arch >= 908,
            has_injective_codegen: false,
        }
    }

    fn emit_kernel_start(
        &self,
        out: &mut String,
        name: &str,
        params: &[KernelParam],
        shared_mem_bytes: usize,
    ) {
        writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
        for (i, p) in params.iter().enumerate() {
            let comma = if i + 1 < params.len() { "," } else { "" };
            let ty_str = match (&p.ty, &p.qualifier) {
                (ParamType::FloatPtr, ParamQualifier::Input) => "const float* __restrict__",
                (ParamType::FloatPtr, _) => "float* __restrict__",
                (ParamType::Uint, _) => "const unsigned int",
                (ParamType::Float, _) => "const float",
            };
            writeln!(out, "    {ty_str} {}{comma}", p.name).unwrap();
        }
        writeln!(out, ") {{").unwrap();
        if shared_mem_bytes > 0 {
            let count = shared_mem_bytes / 4;
            writeln!(out, "    __shared__ float smem[{count}];").unwrap();
        }
    }

    fn emit_kernel_end(&self, out: &mut String) {
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    fn global_tid_expr(&self) -> &'static str {
        "hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x"
    }

    fn local_tid_expr(&self) -> &'static str { "hipThreadIdx_x" }
    fn group_id_expr(&self) -> &'static str { "hipBlockIdx_x" }

    fn emit_shared_decl(&self, out: &mut String, name: &str, count: usize) {
        writeln!(out, "    __shared__ float {name}[{count}];").unwrap();
    }

    fn emit_barrier(&self, out: &mut String) {
        writeln!(out, "    __syncthreads();").unwrap();
    }

    fn emit_for_start(
        &self,
        out: &mut String,
        var: &str,
        start: &str,
        limit: &str,
        stride: &str,
        _label: &str,
    ) {
        writeln!(out, "    for (unsigned int {var} = {start}; {var} < {limit}; {var} += {stride}) {{").unwrap();
    }

    fn emit_for_end(&self, out: &mut String, _var: &str, _stride: &str, _label: &str) {
        writeln!(out, "    }}").unwrap();
    }

    fn emit_if_start(&self, out: &mut String, cond: &str, _label: &str) {
        writeln!(out, "    if ({cond}) {{").unwrap();
    }

    fn emit_if_end(&self, out: &mut String, _label: &str) {
        writeln!(out, "    }}").unwrap();
    }

    fn emit_float_decl(&self, out: &mut String, name: &str, expr: &str) {
        writeln!(out, "    float {name} = {expr};").unwrap();
    }

    fn emit_float_assign(&self, out: &mut String, name: &str, expr: &str) {
        writeln!(out, "    {name} = {expr};").unwrap();
    }

    fn emit_global_load(&self, out: &mut String, dst: &str, ptr: &str, index: &str) {
        writeln!(out, "    float {dst} = {ptr}[{index}];").unwrap();
    }

    fn emit_global_store(&self, out: &mut String, ptr: &str, index: &str, value: &str) {
        writeln!(out, "    {ptr}[{index}] = {value};").unwrap();
    }

    fn emit_shared_load(&self, out: &mut String, dst: &str, array: &str, index: &str) {
        writeln!(out, "    float {dst} = {array}[{index}];").unwrap();
    }

    fn emit_shared_store(&self, out: &mut String, array: &str, index: &str, value: &str) {
        writeln!(out, "    {array}[{index}] = {value};").unwrap();
    }

    fn emit_bounds_check_return(&self, out: &mut String, var: &str, limit: &str, _label: &str) {
        writeln!(out, "    if ({var} >= {limit}) return;").unwrap();
    }

    fn emit_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]) {
        super::kernel_builder::build_elementwise_kernel(self, out, name, body);
    }

    fn emit_binary_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]) {
        super::kernel_builder::build_binary_elementwise_kernel(self, out, name, body);
    }

    fn emit_gemm_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
    ) -> Result<(), String> {
        use crate::compiler::codegen::hip::*;
        match op_kind {
            OpKind::Gemm { .. } | OpKind::GemmBias { .. } => {
                if self.gfx_arch >= 908 {
                    emit_gemm_mfma_kernel_hip(out, name);
                } else {
                    // Use unified tiled GEMM builder for non-MFMA hardware
                    let has_bias = matches!(op_kind, OpKind::GemmBias { .. });
                    return super::kernel_builder::build_gemm_tiled_kernel(
                        self, out, name, has_bias, &[],
                    );
                }
                Ok(())
            }
            OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                super::kernel_builder::build_mha_kernel(self, out, name, *seq_len, *num_heads, *head_dim);
                Ok(())
            }
            OpKind::QuantGemm { m, n, k, .. } => {
                Err(format!(
                    "HipDialect: fused QuantGemm not yet implemented ({}x{}x{}), use Dequantize + Gemm",
                    m, n, k
                ))
            }
            _ => Err(format!("HipDialect: unsupported Gemm-pattern op {:?}", op_kind)),
        }
    }

    fn emit_mha_kernel(
        &self,
        out: &mut String,
        name: &str,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        super::kernel_builder::build_mha_kernel(self, out, name, seq_len, num_heads, head_dim);
    }

    fn emit_rope_kernel(
        &self,
        out: &mut String,
        name: &str,
        head_dim: usize,
        theta: f64,
    ) {
        crate::compiler::codegen::hip::emit_rope_kernel_hip(out, name, head_dim, theta);
    }

    fn emit_reduction_kernel(
        &self,
        out: &mut String,
        name: &str,
        identity: f64,
        combine: &[TraceOp],
    ) -> Result<(), String> {
        crate::compiler::codegen::hip::emit_reduction_kernel_hip(
            out, name, identity, combine, self.gfx_arch,
        );
        Ok(())
    }

    fn emit_injective_kernel(
        &self,
        out: &mut String,
        name: &str,
        body: &[TraceOp],
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<(), String> {
        crate::compiler::codegen::hip::emit_injective_kernel_hip(
            out, name, body, num_inputs, num_outputs,
        );
        Ok(())
    }

    fn emit_normlike_kernel(
        &self,
        out: &mut String,
        name: &str,
        reduce: &[TraceOp],
        finalize: &[TraceOp],
        transform: &[TraceOp],
        has_weight: bool,
        has_bias: bool,
        eps_override: Option<f32>,
    ) {
        super::kernel_builder::build_normlike_kernel(
            self, out, name, reduce, finalize, transform,
            has_weight, has_bias, eps_override.map(|e| e as f64), self.default_block_size(),
        );
    }

    fn emit_softmax_kernel(&self, out: &mut String, name: &str) {
        super::kernel_builder::build_softmax_kernel(self, out, name, self.default_block_size());
    }

    fn emit_meanpool_kernel(&self, out: &mut String, name: &str, seq_len: usize, hidden: usize) {
        super::kernel_builder::build_meanpool_kernel(self, out, name, seq_len, hidden);
    }

    fn emit_dequantize_kernel(
        &self,
        out: &mut String,
        name: &str,
        num_elements: usize,
        block_size: usize,
        bits: u8,
    ) {
        super::kernel_builder::build_dequantize_kernel(
            self, out, name, num_elements, block_size, bits.into(),
        );
    }

    fn emit_gemm_epilogue_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
        epilogue_bodies: &[&[TraceOp]],
    ) -> Result<(), String> {
        super::kernel_builder::build_gemm_epilogue_kernel(
            self, out, name, op_kind, epilogue_bodies,
        )
    }

    fn emit_tile_level_fusion_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        tile_rows: usize,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        shared_mem_budget: usize,
    ) -> Result<(), String> {
        // Extract GEMM dimensions from anchor op
        let (m, n, k) = match gemm_op {
            OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
            _ => return Err(format!(
                "TileLevelFusion anchor must be Gemm/GemmBias, got {:?}", gemm_op
            )),
        };

        // Extract eps from norm op (RmsNorm/LayerNorm)
        let eps = match norm_op {
            OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
            _ => 1e-6,
        };
        let has_norm_weight = matches!(
            norm_op, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }
        );

        let block_size = 256usize;
        // shared memory: norm_tile[tile_rows * K] for norm output
        let norm_tile_floats = tile_rows * k;
        let norm_tile_bytes = norm_tile_floats * 4;
        if norm_tile_bytes > shared_mem_budget {
            return Err(format!(
                "TileLevelFusion: norm tile ({} bytes) exceeds shared_mem budget ({} bytes)",
                norm_tile_bytes, shared_mem_budget
            ));
        }

        let eps_bits = eps.to_bits();

        // Kernel signature
        writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
        writeln!(out, "    const float* __restrict__ input,").unwrap();
        if has_norm_weight {
            writeln!(out, "    const float* __restrict__ norm_weight,").unwrap();
        }
        writeln!(out, "    const float* __restrict__ weight,").unwrap();
        writeln!(out, "    float* __restrict__ output,").unwrap();
        writeln!(out, "    const int M,").unwrap();
        writeln!(out, "    const int N,").unwrap();
        writeln!(out, "    const int K").unwrap();
        writeln!(out, ") {{").unwrap();

        // Constants and shared memory
        writeln!(out, "    const int TILE_ROWS = {tile_rows};").unwrap();
        writeln!(out, "    const float eps = __uint_as_float(0x{eps_bits:08X}u);").unwrap();
        writeln!(out, "    __shared__ float norm_tile[{norm_tile_floats}];").unwrap();
        writeln!(out).unwrap();

        // Block-level row tiling
        writeln!(out, "    int block_row = blockIdx.x * TILE_ROWS;").unwrap();
        writeln!(out, "    int tid = threadIdx.x;").unwrap();
        writeln!(out, "    int mc_cur = min(TILE_ROWS, M - block_row);").unwrap();
        writeln!(out).unwrap();

        // Phase 1: Cooperative RmsNorm/LayerNorm for tile_rows into shared memory
        writeln!(out, "    // Phase 1: Cooperative norm for tile rows").unwrap();
        writeln!(out, "    for (int r = tid; r < mc_cur; r += {block_size}) {{").unwrap();

        // Reduce phase: compute sum of squares
        writeln!(out, "        float sum_sq = 0.0f;").unwrap();
        writeln!(out, "        for (int c = 0; c < K; c++) {{").unwrap();
        writeln!(out, "            float x = input[(block_row + r) * K + c];").unwrap();

        // Emit reduce trace inline (sum_sq += x*x for RmsNorm)
        let reduce_bindings = vec!["x".to_string(), "sum_sq".to_string()];
        let reduce_result = emit_trace_body(self, out, norm_reduce, 10, &reduce_bindings);
        writeln!(out, "            sum_sq += {reduce_result};").unwrap();
        writeln!(out, "        }}").unwrap();

        // Finalize phase: compute normalization factor
        writeln!(out, "        float norm_mean = sum_sq / (float)K;").unwrap();
        let finalize_bindings = vec!["norm_mean".to_string()];
        let inv_rms = emit_trace_body(self, out, norm_finalize, 11, &finalize_bindings);
        writeln!(out).unwrap();

        // Transform phase: apply norm and write to shared memory
        writeln!(out, "        for (int c = 0; c < K; c++) {{").unwrap();
        writeln!(out, "            float x = input[(block_row + r) * K + c];").unwrap();
        if has_norm_weight {
            let transform_bindings = vec![
                "x".to_string(), inv_rms.clone(), "norm_weight[c]".to_string(),
            ];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "            norm_tile[r * K + c] = {normed};").unwrap();
        } else {
            let transform_bindings = vec!["x".to_string(), inv_rms.clone()];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "            norm_tile[r * K + c] = {normed};").unwrap();
        }
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    __syncthreads();").unwrap();
        writeln!(out).unwrap();

        // Phase 2: Tiled GEMM from shared memory
        let tile_k = 16usize;
        writeln!(out, "    // Phase 2: Tiled GEMM from shared memory").unwrap();
        writeln!(out, "    const int TILE_N = 16;").unwrap();
        writeln!(out, "    const int TILE_K = {tile_k};").unwrap();
        writeln!(out, "    __shared__ float b_tile[{tile_k} * 16];").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    int ty = tid / TILE_N;").unwrap();
        writeln!(out, "    int tx = tid % TILE_N;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    for (int col_block = 0; col_block < N; col_block += TILE_N) {{").unwrap();
        writeln!(out, "        for (int row = ty; row < mc_cur; row += ({block_size} / TILE_N)) {{").unwrap();
        writeln!(out, "            float acc = 0.0f;").unwrap();
        writeln!(out, "            for (int pc = 0; pc < K; pc += TILE_K) {{").unwrap();
        writeln!(out, "                // Load B tile to shared").unwrap();
        writeln!(out, "                if (tid < TILE_K * TILE_N) {{").unwrap();
        writeln!(out, "                    int bk = tid / TILE_N;").unwrap();
        writeln!(out, "                    int bn = tid % TILE_N;").unwrap();
        writeln!(out, "                    int gk = pc + bk;").unwrap();
        writeln!(out, "                    int gn = col_block + bn;").unwrap();
        writeln!(out, "                    b_tile[bk * TILE_N + bn] = (gk < K && gn < N) ? weight[gk * N + gn] : 0.0f;").unwrap();
        writeln!(out, "                }}").unwrap();
        writeln!(out, "                __syncthreads();").unwrap();
        writeln!(out, "                for (int kk = 0; kk < TILE_K; kk++) {{").unwrap();
        writeln!(out, "                    int gk = pc + kk;").unwrap();
        writeln!(out, "                    float a_val = (gk < K) ? norm_tile[row * K + gk] : 0.0f;").unwrap();
        writeln!(out, "                    acc = fmaf(a_val, b_tile[kk * TILE_N + tx], acc);").unwrap();
        writeln!(out, "                }}").unwrap();
        writeln!(out, "                __syncthreads();").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "            int out_row = block_row + row;").unwrap();
        writeln!(out, "            int out_col = col_block + tx;").unwrap();
        writeln!(out, "            if (out_row < M && out_col < N) {{").unwrap();
        writeln!(out, "                output[out_row * N + out_col] = acc;").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}\n").unwrap();

        Ok(())
    }

    fn emit_compute_root_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        _shared_mem_budget: usize,
    ) -> Result<(), String> {
        // ComputeRoot: two-kernel approach — full norm to global buffer, then GEMM.
        // Emitted as a single kernel with two phases using global memory intermediate.
        let (_m, _n, k) = match gemm_op {
            OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
            _ => return Err(format!(
                "ComputeRoot anchor must be Gemm/GemmBias, got {:?}", gemm_op
            )),
        };

        let eps = match norm_op {
            OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
            _ => 1e-6,
        };
        let has_norm_weight = matches!(
            norm_op, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }
        );

        let block_size = 256usize;
        let eps_bits = eps.to_bits();

        // Kernel 1: Full norm
        let norm_name = format!("{name}_norm");
        writeln!(out, "extern \"C\" __global__ void {norm_name}(").unwrap();
        writeln!(out, "    const float* __restrict__ input,").unwrap();
        if has_norm_weight {
            writeln!(out, "    const float* __restrict__ norm_weight,").unwrap();
        }
        writeln!(out, "    float* __restrict__ normed,").unwrap();
        writeln!(out, "    const int M,").unwrap();
        writeln!(out, "    const int K").unwrap();
        writeln!(out, ") {{").unwrap();
        writeln!(out, "    const float eps = __uint_as_float(0x{eps_bits:08X}u);").unwrap();
        writeln!(out, "    int row = blockIdx.x * blockDim.x + threadIdx.x;").unwrap();
        writeln!(out, "    if (row >= M) return;").unwrap();
        writeln!(out).unwrap();

        // Reduce
        writeln!(out, "    float sum_sq = 0.0f;").unwrap();
        writeln!(out, "    for (int c = 0; c < K; c++) {{").unwrap();
        writeln!(out, "        float x = input[row * K + c];").unwrap();
        let reduce_bindings = vec!["x".to_string(), "sum_sq".to_string()];
        let reduce_result = emit_trace_body(self, out, norm_reduce, 10, &reduce_bindings);
        writeln!(out, "        sum_sq += {reduce_result};").unwrap();
        writeln!(out, "    }}").unwrap();

        // Finalize
        writeln!(out, "    float norm_mean = sum_sq / (float)K;").unwrap();
        let finalize_bindings = vec!["norm_mean".to_string()];
        let inv_rms = emit_trace_body(self, out, norm_finalize, 11, &finalize_bindings);
        writeln!(out).unwrap();

        // Transform
        writeln!(out, "    for (int c = 0; c < K; c++) {{").unwrap();
        writeln!(out, "        float x = input[row * K + c];").unwrap();
        if has_norm_weight {
            let transform_bindings = vec![
                "x".to_string(), inv_rms.clone(), "norm_weight[c]".to_string(),
            ];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "        normed[row * K + c] = {normed};").unwrap();
        } else {
            let transform_bindings = vec!["x".to_string(), inv_rms.clone()];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "        normed[row * K + c] = {normed};").unwrap();
        }
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}\n").unwrap();

        // Kernel 2: Standard tiled GEMM reading from normed buffer
        let gemm_name = format!("{name}_gemm");
        let tile_size = 16usize;
        writeln!(out, "extern \"C\" __global__ void {gemm_name}(").unwrap();
        writeln!(out, "    const float* __restrict__ A,").unwrap();
        writeln!(out, "    const float* __restrict__ B,").unwrap();
        writeln!(out, "    float* __restrict__ C,").unwrap();
        writeln!(out, "    const int M,").unwrap();
        writeln!(out, "    const int N,").unwrap();
        writeln!(out, "    const int K").unwrap();
        writeln!(out, ") {{").unwrap();
        writeln!(out, "    const int TILE_SIZE = {tile_size};").unwrap();
        writeln!(out, "    __shared__ float As[{tile_size} * {tile_size}];").unwrap();
        writeln!(out, "    __shared__ float Bs[{tile_size} * {tile_size}];").unwrap();
        writeln!(out, "    int row = blockIdx.y * TILE_SIZE + threadIdx.y;").unwrap();
        writeln!(out, "    int col = blockIdx.x * TILE_SIZE + threadIdx.x;").unwrap();
        writeln!(out, "    float acc = 0.0f;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {{").unwrap();
        writeln!(out, "        int ak = t * TILE_SIZE + threadIdx.x;").unwrap();
        writeln!(out, "        int bk = t * TILE_SIZE + threadIdx.y;").unwrap();
        writeln!(out, "        As[threadIdx.y * TILE_SIZE + threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;").unwrap();
        writeln!(out, "        Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;").unwrap();
        writeln!(out, "        __syncthreads();").unwrap();
        writeln!(out, "        for (int kk = 0; kk < TILE_SIZE; kk++) {{").unwrap();
        writeln!(out, "            acc = fmaf(As[threadIdx.y * TILE_SIZE + kk], Bs[kk * TILE_SIZE + threadIdx.x], acc);").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        __syncthreads();").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    if (row < M && col < N) {{").unwrap();
        writeln!(out, "        C[row * N + col] = acc;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}\n").unwrap();

        Ok(())
    }
}

// ── MslDialect ──────────────────────────────────────────────────────────────

/// MSL dialect — emits Metal Shading Language source code for Apple GPUs.
#[cfg(feature = "jit-metal")]
pub struct MslDialect {
    /// Metal GPU family (e.g. 7 = Apple7, 9 = Apple9).
    pub gpu_family: u32,
}

#[cfg(feature = "jit-metal")]
impl MslDialect {
    pub fn new(gpu_family: u32) -> Self {
        Self { gpu_family }
    }
}

#[cfg(feature = "jit-metal")]
impl GpuDialect for MslDialect {
    fn emit_header(&self, out: &mut String) {
        writeln!(out, "#include <metal_stdlib>").unwrap();
        writeln!(out, "using namespace metal;").unwrap();
        writeln!(out).unwrap();
    }

    fn emit_trace_op(
        &self,
        out: &mut String,
        op: &TraceOp,
        idx: usize,
        vars: &[String],
        tier: usize,
    ) -> String {
        let var_name = format!("t{}_{}", tier, idx);
        // MSL uses inline expressions assigned to local variables.
        let expr = match op {
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
        };
        writeln!(out, "    float {var_name} = {expr};").unwrap();
        var_name
    }

    fn warp_size(&self) -> u32 {
        32 // Apple GPU SIMD-group width
    }

    fn capabilities(&self) -> GpuCapabilities {
        GpuCapabilities {
            has_matrix_unit: self.gpu_family >= 7,
            has_injective_codegen: true,
        }
    }

    fn emit_kernel_start(
        &self,
        out: &mut String,
        name: &str,
        params: &[KernelParam],
        shared_mem_bytes: usize,
    ) {
        writeln!(out, "kernel void {name}(").unwrap();
        let mut buf_idx: usize = 0;
        for (i, p) in params.iter().enumerate() {
            let comma = if i + 1 < params.len() { "," } else { "," };
            match (&p.ty, &p.qualifier) {
                (ParamType::FloatPtr, ParamQualifier::Input) => {
                    writeln!(out, "    device const float* {} [[buffer({buf_idx})]]{comma}", p.name).unwrap();
                    buf_idx += 1;
                }
                (ParamType::FloatPtr, _) => {
                    writeln!(out, "    device float* {} [[buffer({buf_idx})]]{comma}", p.name).unwrap();
                    buf_idx += 1;
                }
                (ParamType::Uint, _) => {
                    writeln!(out, "    constant uint& {} [[buffer({buf_idx})]]{comma}", p.name).unwrap();
                    buf_idx += 1;
                }
                (ParamType::Float, _) => {
                    writeln!(out, "    constant float& {} [[buffer({buf_idx})]]{comma}", p.name).unwrap();
                    buf_idx += 1;
                }
            }
        }
        writeln!(out, "    uint lid [[thread_position_in_threadgroup]],").unwrap();
        writeln!(out, "    uint gid [[threadgroup_position_in_grid]],").unwrap();
        writeln!(out, "    uint tid [[thread_position_in_grid]]").unwrap();
        writeln!(out, ") {{").unwrap();
        if shared_mem_bytes > 0 {
            let count = shared_mem_bytes / 4;
            writeln!(out, "    threadgroup float smem[{count}];").unwrap();
        }
    }

    fn emit_kernel_end(&self, out: &mut String) {
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    fn global_tid_expr(&self) -> &'static str { "tid" }
    fn local_tid_expr(&self) -> &'static str { "lid" }
    fn group_id_expr(&self) -> &'static str { "gid" }

    fn emit_shared_decl(&self, out: &mut String, name: &str, count: usize) {
        writeln!(out, "    threadgroup float {name}[{count}];").unwrap();
    }

    fn emit_barrier(&self, out: &mut String) {
        writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    }

    fn emit_for_start(
        &self,
        out: &mut String,
        var: &str,
        start: &str,
        limit: &str,
        stride: &str,
        _label: &str,
    ) {
        writeln!(out, "    for (uint {var} = {start}; {var} < {limit}; {var} += {stride}) {{").unwrap();
    }

    fn emit_for_end(&self, out: &mut String, _var: &str, _stride: &str, _label: &str) {
        writeln!(out, "    }}").unwrap();
    }

    fn emit_if_start(&self, out: &mut String, cond: &str, _label: &str) {
        writeln!(out, "    if ({cond}) {{").unwrap();
    }

    fn emit_if_end(&self, out: &mut String, _label: &str) {
        writeln!(out, "    }}").unwrap();
    }

    fn emit_float_decl(&self, out: &mut String, name: &str, expr: &str) {
        writeln!(out, "    float {name} = {expr};").unwrap();
    }

    fn emit_float_assign(&self, out: &mut String, name: &str, expr: &str) {
        writeln!(out, "    {name} = {expr};").unwrap();
    }

    fn emit_global_load(&self, out: &mut String, dst: &str, ptr: &str, index: &str) {
        writeln!(out, "    float {dst} = {ptr}[{index}];").unwrap();
    }

    fn emit_global_store(&self, out: &mut String, ptr: &str, index: &str, value: &str) {
        writeln!(out, "    {ptr}[{index}] = {value};").unwrap();
    }

    fn emit_shared_load(&self, out: &mut String, dst: &str, array: &str, index: &str) {
        writeln!(out, "    float {dst} = {array}[{index}];").unwrap();
    }

    fn emit_shared_store(&self, out: &mut String, array: &str, index: &str, value: &str) {
        writeln!(out, "    {array}[{index}] = {value};").unwrap();
    }

    fn emit_bounds_check_return(&self, out: &mut String, var: &str, limit: &str, _label: &str) {
        writeln!(out, "    if ({var} >= {limit}) return;").unwrap();
    }

    fn emit_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]) {
        super::kernel_builder::build_elementwise_kernel(self, out, name, body);
    }

    fn emit_binary_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]) {
        super::kernel_builder::build_binary_elementwise_kernel(self, out, name, body);
    }

    fn emit_gemm_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
    ) -> Result<(), String> {
        use crate::compiler::codegen::air::*;
        match op_kind {
            OpKind::Gemm { m, n, k } => {
                if self.gpu_family >= 7 {
                    emit_gemm_simdgroup_msl(out, name, *m, *n, *k);
                } else {
                    emit_gemm_kernel_msl(out, name, *m, *n, *k);
                }
                Ok(())
            }
            OpKind::GemmBias { m, n, k } => {
                if self.gpu_family >= 7 {
                    emit_gemm_bias_simdgroup_msl(out, name, *m, *n, *k);
                } else {
                    emit_gemm_bias_kernel_msl(out, name, *m, *n, *k);
                }
                Ok(())
            }
            OpKind::QuantGemm { m, n, k, .. } => {
                if self.gpu_family >= 7 {
                    emit_gemm_simdgroup_msl(out, name, *m, *n, *k);
                } else {
                    emit_gemm_kernel_msl(out, name, *m, *n, *k);
                }
                Ok(())
            }
            OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                emit_mha_kernel_msl(out, name, *seq_len, *num_heads, *head_dim);
                Ok(())
            }
            _ => Err(format!("MslDialect: unsupported Gemm-pattern op {:?}", op_kind)),
        }
    }

    fn emit_mha_kernel(
        &self,
        out: &mut String,
        name: &str,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        crate::compiler::codegen::air::emit_mha_kernel_msl(out, name, seq_len, num_heads, head_dim);
    }

    fn emit_rope_kernel(
        &self,
        out: &mut String,
        name: &str,
        head_dim: usize,
        theta: f64,
    ) {
        crate::compiler::codegen::air::emit_rope_kernel_msl(out, name, head_dim, theta);
    }

    fn emit_reduction_kernel(
        &self,
        out: &mut String,
        name: &str,
        identity: f64,
        combine: &[TraceOp],
    ) -> Result<(), String> {
        crate::compiler::codegen::air::emit_reduction_kernel_msl(
            out, name, identity, combine,
        );
        Ok(())
    }

    fn emit_injective_kernel(
        &self,
        out: &mut String,
        name: &str,
        body: &[TraceOp],
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<(), String> {
        crate::compiler::codegen::air::emit_injective_kernel_msl(
            out, name, body, num_inputs, num_outputs,
        );
        Ok(())
    }

    fn emit_normlike_kernel(
        &self,
        out: &mut String,
        name: &str,
        reduce: &[TraceOp],
        finalize: &[TraceOp],
        transform: &[TraceOp],
        has_weight: bool,
        has_bias: bool,
        eps_override: Option<f32>,
    ) {
        super::kernel_builder::build_normlike_kernel(
            self, out, name, reduce, finalize, transform,
            has_weight, has_bias, eps_override.map(|e| e as f64), self.default_block_size(),
        );
    }

    fn emit_softmax_kernel(&self, out: &mut String, name: &str) {
        super::kernel_builder::build_softmax_kernel(self, out, name, self.default_block_size());
    }

    fn emit_meanpool_kernel(&self, out: &mut String, name: &str, seq_len: usize, hidden: usize) {
        super::kernel_builder::build_meanpool_kernel(self, out, name, seq_len, hidden);
    }

    fn emit_dequantize_kernel(
        &self,
        out: &mut String,
        name: &str,
        num_elements: usize,
        block_size: usize,
        bits: u8,
    ) {
        super::kernel_builder::build_dequantize_kernel(
            self, out, name, num_elements, block_size, bits.into(),
        );
    }

    fn emit_gemm_epilogue_kernel(
        &self,
        out: &mut String,
        name: &str,
        op_kind: &OpKind,
        epilogue_bodies: &[&[TraceOp]],
    ) -> Result<(), String> {
        let (_m, _n, _k) = match op_kind {
            OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
            _ => return Err(format!(
                "MslDialect emit_gemm_epilogue_kernel: expected Gemm/GemmBias, got {:?}", op_kind
            )),
        };
        let has_bias = matches!(op_kind, OpKind::GemmBias { .. });
        let tile_size = 16usize;
        let mut buf_idx = 0u32;
        writeln!(out, "kernel void {name}(").unwrap();
        writeln!(out, "    device const float* A [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    device const float* B [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        if has_bias {
            writeln!(out, "    device const float* bias [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        }
        writeln!(out, "    device float* C [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& M [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& N [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& K_dim [[buffer({buf_idx})]],").unwrap();
        writeln!(out, "    uint2 lid [[thread_position_in_threadgroup]],").unwrap();
        writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]]").unwrap();
        writeln!(out, ") {{").unwrap();
        writeln!(out, "    const uint TILE = {tile_size};").unwrap();
        writeln!(out, "    threadgroup float As[{tile_size} * {tile_size}];").unwrap();
        writeln!(out, "    threadgroup float Bs[{tile_size} * {tile_size}];").unwrap();
        writeln!(out, "    uint row = gid.y * TILE + lid.y;").unwrap();
        writeln!(out, "    uint col = gid.x * TILE + lid.x;").unwrap();
        writeln!(out, "    float acc = 0.0f;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    for (uint t = 0; t < (K_dim + TILE - 1) / TILE; t++) {{").unwrap();
        writeln!(out, "        uint ak = t * TILE + lid.x;").unwrap();
        writeln!(out, "        uint bk = t * TILE + lid.y;").unwrap();
        writeln!(out, "        As[lid.y * TILE + lid.x] = (row < M && ak < K_dim) ? A[row * K_dim + ak] : 0.0f;").unwrap();
        writeln!(out, "        Bs[lid.y * TILE + lid.x] = (bk < K_dim && col < N) ? B[bk * N + col] : 0.0f;").unwrap();
        writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out, "        for (uint kk = 0; kk < TILE; kk++) {{").unwrap();
        writeln!(out, "            acc = fma(As[lid.y * TILE + kk], Bs[kk * TILE + lid.x], acc);").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out, "    }}").unwrap();
        if has_bias { writeln!(out, "    if (col < N) acc += bias[col];").unwrap(); }
        let mut prev = "acc".to_string();
        for (ei, body) in epilogue_bodies.iter().enumerate() {
            let b = vec![prev.clone()];
            prev = emit_trace_body(self, out, body, 20 + ei, &b);
        }
        writeln!(out, "    if (row < M && col < N) C[row * N + col] = {prev};").unwrap();
        writeln!(out, "}}\n").unwrap();
        Ok(())
    }

    fn emit_tile_level_fusion_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        tile_rows: usize,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        shared_mem_budget: usize,
    ) -> Result<(), String> {
        let (_m, _n, k) = match gemm_op {
            OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
            _ => return Err(format!(
                "TileLevelFusion anchor must be Gemm/GemmBias, got {:?}", gemm_op
            )),
        };
        let eps = match norm_op {
            OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
            _ => 1e-6,
        };
        let has_norm_weight = matches!(
            norm_op, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }
        );
        let block_size = 256usize;
        let norm_tile_floats = tile_rows * k;
        let norm_tile_bytes = norm_tile_floats * 4;
        if norm_tile_bytes > shared_mem_budget {
            return Err(format!(
                "TileLevelFusion: norm tile ({} bytes) exceeds shared_mem budget ({} bytes)",
                norm_tile_bytes, shared_mem_budget
            ));
        }
        let eps_bits = eps.to_bits();
        let mut buf_idx = 0u32;
        writeln!(out, "kernel void {name}(").unwrap();
        writeln!(out, "    device const float* input [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        if has_norm_weight {
            writeln!(out, "    device const float* norm_weight [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        }
        writeln!(out, "    device const float* weight [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    device float* output [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& M [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& N [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& K_dim [[buffer({buf_idx})]],").unwrap();
        writeln!(out, "    uint lid [[thread_position_in_threadgroup]],").unwrap();
        writeln!(out, "    uint gid [[threadgroup_position_in_grid]]").unwrap();
        writeln!(out, ") {{").unwrap();
        writeln!(out, "    const uint TILE_ROWS = {tile_rows};").unwrap();
        writeln!(out, "    const float eps = as_type<float>(0x{eps_bits:08X}u);").unwrap();
        writeln!(out, "    threadgroup float norm_tile[{norm_tile_floats}];").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    uint block_row = gid * TILE_ROWS;").unwrap();
        writeln!(out, "    uint mc_cur = min(TILE_ROWS, M - block_row);").unwrap();
        writeln!(out).unwrap();
        // Phase 1: Cooperative norm
        writeln!(out, "    // Phase 1: Cooperative norm for tile rows").unwrap();
        writeln!(out, "    for (uint r = lid; r < mc_cur; r += {block_size}) {{").unwrap();
        writeln!(out, "        float sum_sq = 0.0f;").unwrap();
        writeln!(out, "        for (uint c = 0; c < K_dim; c++) {{").unwrap();
        writeln!(out, "            float x = input[(block_row + r) * K_dim + c];").unwrap();
        let reduce_bindings = vec!["x".to_string(), "sum_sq".to_string()];
        let reduce_result = emit_trace_body(self, out, norm_reduce, 10, &reduce_bindings);
        writeln!(out, "            sum_sq += {reduce_result};").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        float norm_mean = sum_sq / float(K_dim);").unwrap();
        let finalize_bindings = vec!["norm_mean".to_string()];
        let inv_rms = emit_trace_body(self, out, norm_finalize, 11, &finalize_bindings);
        writeln!(out).unwrap();
        writeln!(out, "        for (uint c = 0; c < K_dim; c++) {{").unwrap();
        writeln!(out, "            float x = input[(block_row + r) * K_dim + c];").unwrap();
        if has_norm_weight {
            let transform_bindings = vec![
                "x".to_string(), inv_rms.clone(), "norm_weight[c]".to_string(),
            ];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "            norm_tile[r * K_dim + c] = {normed};").unwrap();
        } else {
            let transform_bindings = vec!["x".to_string(), inv_rms.clone()];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "            norm_tile[r * K_dim + c] = {normed};").unwrap();
        }
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out).unwrap();
        // Phase 2: Tiled GEMM
        let tile_k = 16usize;
        writeln!(out, "    // Phase 2: Tiled GEMM from threadgroup memory").unwrap();
        writeln!(out, "    const uint TILE_N = 16;").unwrap();
        writeln!(out, "    const uint TILE_K = {tile_k};").unwrap();
        writeln!(out, "    threadgroup float b_tile[{tile_k} * 16];").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    uint ty = lid / TILE_N;").unwrap();
        writeln!(out, "    uint tx = lid % TILE_N;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    for (uint col_block = 0; col_block < N; col_block += TILE_N) {{").unwrap();
        writeln!(out, "        for (uint row = ty; row < mc_cur; row += ({block_size} / TILE_N)) {{").unwrap();
        writeln!(out, "            float acc = 0.0f;").unwrap();
        writeln!(out, "            for (uint pc = 0; pc < K_dim; pc += TILE_K) {{").unwrap();
        writeln!(out, "                if (lid < TILE_K * TILE_N) {{").unwrap();
        writeln!(out, "                    uint bk = lid / TILE_N;").unwrap();
        writeln!(out, "                    uint bn = lid % TILE_N;").unwrap();
        writeln!(out, "                    uint gk = pc + bk;").unwrap();
        writeln!(out, "                    uint gn = col_block + bn;").unwrap();
        writeln!(out, "                    b_tile[bk * TILE_N + bn] = (gk < K_dim && gn < N) ? weight[gk * N + gn] : 0.0f;").unwrap();
        writeln!(out, "                }}").unwrap();
        writeln!(out, "                threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out, "                for (uint kk = 0; kk < TILE_K; kk++) {{").unwrap();
        writeln!(out, "                    uint gk = pc + kk;").unwrap();
        writeln!(out, "                    float a_val = (gk < K_dim) ? norm_tile[row * K_dim + gk] : 0.0f;").unwrap();
        writeln!(out, "                    acc = fma(a_val, b_tile[kk * TILE_N + tx], acc);").unwrap();
        writeln!(out, "                }}").unwrap();
        writeln!(out, "                threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "            uint out_row = block_row + row;").unwrap();
        writeln!(out, "            uint out_col = col_block + tx;").unwrap();
        writeln!(out, "            if (out_row < M && out_col < N) {{").unwrap();
        writeln!(out, "                output[out_row * N + out_col] = acc;").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}\n").unwrap();
        Ok(())
    }

    fn emit_compute_root_kernel(
        &self,
        out: &mut String,
        name: &str,
        norm_op: &OpKind,
        gemm_op: &OpKind,
        norm_reduce: &[TraceOp],
        norm_finalize: &[TraceOp],
        norm_transform: &[TraceOp],
        _shared_mem_budget: usize,
    ) -> Result<(), String> {
        let (_m, _n, _k) = match gemm_op {
            OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => (*m, *n, *k),
            _ => return Err(format!(
                "ComputeRoot anchor must be Gemm/GemmBias, got {:?}", gemm_op
            )),
        };
        let eps = match norm_op {
            OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
            _ => 1e-6,
        };
        let has_norm_weight = matches!(
            norm_op, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }
        );
        let eps_bits = eps.to_bits();
        // Kernel 1: Full norm to device memory
        let norm_name = format!("{name}_norm");
        let mut buf_idx = 0u32;
        writeln!(out, "kernel void {norm_name}(").unwrap();
        writeln!(out, "    device const float* input [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        if has_norm_weight {
            writeln!(out, "    device const float* norm_weight [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        }
        writeln!(out, "    device float* normed [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& M [[buffer({buf_idx})]],").unwrap(); buf_idx += 1;
        writeln!(out, "    constant uint& K_dim [[buffer({buf_idx})]],").unwrap();
        writeln!(out, "    uint tid [[thread_position_in_grid]]").unwrap();
        writeln!(out, ") {{").unwrap();
        writeln!(out, "    const float eps = as_type<float>(0x{eps_bits:08X}u);").unwrap();
        writeln!(out, "    uint row = tid;").unwrap();
        writeln!(out, "    if (row >= M) return;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    float sum_sq = 0.0f;").unwrap();
        writeln!(out, "    for (uint c = 0; c < K_dim; c++) {{").unwrap();
        writeln!(out, "        float x = input[row * K_dim + c];").unwrap();
        let reduce_bindings = vec!["x".to_string(), "sum_sq".to_string()];
        let reduce_result = emit_trace_body(self, out, norm_reduce, 10, &reduce_bindings);
        writeln!(out, "        sum_sq += {reduce_result};").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    float norm_mean = sum_sq / float(K_dim);").unwrap();
        let finalize_bindings = vec!["norm_mean".to_string()];
        let inv_rms = emit_trace_body(self, out, norm_finalize, 11, &finalize_bindings);
        writeln!(out).unwrap();
        writeln!(out, "    for (uint c = 0; c < K_dim; c++) {{").unwrap();
        writeln!(out, "        float x = input[row * K_dim + c];").unwrap();
        if has_norm_weight {
            let transform_bindings = vec![
                "x".to_string(), inv_rms.clone(), "norm_weight[c]".to_string(),
            ];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "        normed[row * K_dim + c] = {normed};").unwrap();
        } else {
            let transform_bindings = vec!["x".to_string(), inv_rms.clone()];
            let normed = emit_trace_body(self, out, norm_transform, 12, &transform_bindings);
            writeln!(out, "        normed[row * K_dim + c] = {normed};").unwrap();
        }
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}\n").unwrap();
        // Kernel 2: Standard tiled GEMM
        let gemm_name = format!("{name}_gemm");
        let tile_size = 16usize;
        writeln!(out, "kernel void {gemm_name}(").unwrap();
        writeln!(out, "    device const float* A [[buffer(0)]],").unwrap();
        writeln!(out, "    device const float* B [[buffer(1)]],").unwrap();
        writeln!(out, "    device float* C [[buffer(2)]],").unwrap();
        writeln!(out, "    constant uint& M [[buffer(3)]],").unwrap();
        writeln!(out, "    constant uint& N [[buffer(4)]],").unwrap();
        writeln!(out, "    constant uint& K_dim [[buffer(5)]],").unwrap();
        writeln!(out, "    uint2 lid [[thread_position_in_threadgroup]],").unwrap();
        writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]]").unwrap();
        writeln!(out, ") {{").unwrap();
        writeln!(out, "    const uint TILE_SIZE = {tile_size};").unwrap();
        writeln!(out, "    threadgroup float As[{tile_size} * {tile_size}];").unwrap();
        writeln!(out, "    threadgroup float Bs[{tile_size} * {tile_size}];").unwrap();
        writeln!(out, "    uint row = gid.y * TILE_SIZE + lid.y;").unwrap();
        writeln!(out, "    uint col = gid.x * TILE_SIZE + lid.x;").unwrap();
        writeln!(out, "    float acc = 0.0f;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    for (uint t = 0; t < (K_dim + TILE_SIZE - 1) / TILE_SIZE; t++) {{").unwrap();
        writeln!(out, "        uint ak = t * TILE_SIZE + lid.x;").unwrap();
        writeln!(out, "        uint bk = t * TILE_SIZE + lid.y;").unwrap();
        writeln!(out, "        As[lid.y * TILE_SIZE + lid.x] = (row < M && ak < K_dim) ? A[row * K_dim + ak] : 0.0f;").unwrap();
        writeln!(out, "        Bs[lid.y * TILE_SIZE + lid.x] = (bk < K_dim && col < N) ? B[bk * N + col] : 0.0f;").unwrap();
        writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out, "        for (uint kk = 0; kk < TILE_SIZE; kk++) {{").unwrap();
        writeln!(out, "            acc = fma(As[lid.y * TILE_SIZE + kk], Bs[kk * TILE_SIZE + lid.x], acc);").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    if (row < M && col < N) C[row * N + col] = acc;").unwrap();
        writeln!(out, "}}\n").unwrap();
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::TraceOp;

    /// SiLU body: x / (1 + exp(-x))
    fn silu_body() -> Vec<TraceOp> {
        vec![
            TraceOp::Input(0),   // [0] x
            TraceOp::Neg(0),     // [1] -x
            TraceOp::Exp(1),     // [2] exp(-x)
            TraceOp::Const(1.0), // [3] 1.0
            TraceOp::Add(2, 3),  // [4] 1 + exp(-x)
            TraceOp::Div(0, 4),  // [5] x / (1 + exp(-x))
        ]
    }

    #[test]
    #[cfg(feature = "jit-cuda")]
    fn ptx_dialect_silu_matches_legacy() {
        let dialect = PtxDialect::new(80);
        let mut out = String::new();
        let bindings = vec!["%f0".to_string()];
        let result = emit_trace_body(&dialect, &mut out, &silu_body(), 0, &bindings);

        // Should produce PTX instructions with %t0_N registers
        assert!(out.contains("neg.f32 %t0_1, %f0;"), "missing neg: {out}");
        assert!(out.contains("mul.f32 %t0_2, %t0_1, 0f3FB8AA3B;"), "missing exp mul: {out}");
        assert!(out.contains("ex2.approx.f32 %t0_2, %t0_2;"), "missing ex2: {out}");
        assert!(out.contains("mov.f32 %t0_3, 0f3F800000;"), "missing const 1.0: {out}");
        assert!(out.contains("add.f32 %t0_4, %t0_2, %t0_3;"), "missing add: {out}");
        assert!(out.contains("div.approx.f32 %t0_5, %f0, %t0_4;"), "missing div: {out}");
        assert_eq!(result, "%t0_5");
    }

    #[test]
    #[cfg(feature = "jit-hip")]
    fn hip_dialect_silu_matches_legacy() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        let bindings = vec!["x".to_string()];
        let result = emit_trace_body(&dialect, &mut out, &silu_body(), 0, &bindings);

        assert!(out.contains("float t0_1 = -x;"), "missing neg: {out}");
        assert!(out.contains("float t0_2 = expf(t0_1);"), "missing exp: {out}");
        assert!(out.contains("float t0_5 = x / t0_4;"), "missing div: {out}");
        assert_eq!(result, "t0_5");
    }

    #[test]
    #[cfg(feature = "jit-metal")]
    fn msl_dialect_silu_matches_legacy() {
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        let bindings = vec!["input[tid]".to_string()];
        let result = emit_trace_body(&dialect, &mut out, &silu_body(), 0, &bindings);

        assert!(out.contains("float t0_1 = (-input[tid]);"), "missing neg: {out}");
        assert!(out.contains("float t0_2 = exp(t0_1);"), "missing exp: {out}");
        assert!(out.contains("float t0_5 = (input[tid] / t0_4);"), "missing div: {out}");
        assert_eq!(result, "t0_5");
    }

    #[test]
    #[cfg(all(feature = "jit-cuda", feature = "jit-hip", feature = "jit-metal"))]
    fn all_trace_ops_covered() {
        // Ensure every TraceOp variant produces output for all three dialects.
        let ops = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Const(3.14),
            TraceOp::Add(0, 1),
            TraceOp::Sub(0, 1),
            TraceOp::Mul(0, 1),
            TraceOp::Div(0, 1),
            TraceOp::Fma(0, 1, 2),
            TraceOp::Neg(0),
            TraceOp::Abs(0),
            TraceOp::Exp(0),
            TraceOp::Sqrt(0),
            TraceOp::Rsqrt(0),
            TraceOp::Tanh(0),
            TraceOp::Recip(0),
            TraceOp::Log(0),
            TraceOp::Max(0, 1),
            TraceOp::Min(0, 1),
        ];
        let bindings = vec!["a".to_string(), "b".to_string()];

        for dialect_name in &["ptx", "hip", "msl"] {
            let mut out = String::new();
            let result = match *dialect_name {
                "ptx" => emit_trace_body(&PtxDialect::new(80), &mut out, &ops, 0, &bindings),
                "hip" => emit_trace_body(&HipDialect::new(908), &mut out, &ops, 0, &bindings),
                "msl" => emit_trace_body(&MslDialect::new(9), &mut out, &ops, 0, &bindings),
                _ => unreachable!(),
            };
            assert!(!result.is_empty(), "{dialect_name} produced empty result");
            assert!(!out.is_empty(), "{dialect_name} produced no output");
        }
    }

    #[test]
    #[cfg(all(feature = "jit-cuda", feature = "jit-hip", feature = "jit-metal"))]
    fn warp_sizes_correct() {
        assert_eq!(PtxDialect::new(80).warp_size(), 32);
        assert_eq!(HipDialect::new(908).warp_size(), 64);  // CDNA wave64
        assert_eq!(HipDialect::new(1100).warp_size(), 32); // RDNA wave32
        assert_eq!(MslDialect::new(9).warp_size(), 32);
    }

    #[test]
    #[cfg(feature = "jit-cuda")]
    fn empty_body_returns_default() {
        let dialect = PtxDialect::new(80);
        let mut out = String::new();
        let result = emit_trace_body(&dialect, &mut out, &[], 0, &[]);
        assert_eq!(result, "0.0f");
        assert!(out.is_empty());
    }

    #[test]
    #[cfg(feature = "jit-hip")]
    fn tier_namespacing() {
        // Different tiers should produce different variable names.
        let ops = vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
        ];
        let bindings = vec!["x".to_string()];

        let dialect = HipDialect::new(908);
        let mut out0 = String::new();
        let r0 = emit_trace_body(&dialect, &mut out0, &ops, 0, &bindings);
        let mut out1 = String::new();
        let r1 = emit_trace_body(&dialect, &mut out1, &ops, 1, &bindings);

        assert_eq!(r0, "t0_1");
        assert_eq!(r1, "t1_1");
        assert!(out0.contains("t0_1"));
        assert!(out1.contains("t1_1"));
    }
}
