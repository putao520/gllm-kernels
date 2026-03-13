//! Unified kernel builders for GPU codegen.
//!
//! Phase 2 of the GPU codegen unification (SPEC PLAN-gpu-codegen-unify §2.3).
//! Provides generic `build_*_kernel<D: GpuDialect>` functions that replace the
//! duplicated kernel emitters in ptx.rs, hip.rs, and air.rs.
//!
//! Each builder emits a complete kernel using only `GpuDialect` trait methods,
//! making the kernel structure backend-agnostic while preserving backend-specific
//! instruction emission.

use std::fmt::Write;
use crate::compiler::trace::TraceOp;
use super::primitives::{KernelParam, ParamType, ParamQualifier};
use super::trace_emitter::{GpuDialect, emit_trace_body};

// ── Helper: param constructors ──────────────────────────────────────────────

fn param_in(name: &str) -> KernelParam {
    KernelParam {
        name: name.to_string(),
        ty: ParamType::FloatPtr,
        qualifier: ParamQualifier::Input,
    }
}

fn param_out(name: &str) -> KernelParam {
    KernelParam {
        name: name.to_string(),
        ty: ParamType::FloatPtr,
        qualifier: ParamQualifier::Output,
    }
}

fn param_uint(name: &str) -> KernelParam {
    KernelParam {
        name: name.to_string(),
        ty: ParamType::Uint,
        qualifier: ParamQualifier::Value,
    }
}

#[allow(dead_code)]
fn param_float(name: &str) -> KernelParam {
    KernelParam {
        name: name.to_string(),
        ty: ParamType::Float,
        qualifier: ParamQualifier::Value,
    }
}

// ── Elementwise kernel ──────────────────────────────────────────────────────

/// Emit a 1-input, 1-output elementwise kernel.
///
/// Layout: `output[tid] = f(input[tid])` with bounds check on `N`.
pub fn build_elementwise_kernel<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
) {
    let params = vec![
        param_in("input"),
        param_out("output"),
        param_uint("N"),
    ];
    dialect.emit_kernel_start(out, kernel_name, &params, 0);

    let tid = dialect.global_tid_expr();
    dialect.emit_bounds_check_return(out, tid, "N", &format!("{kernel_name}_BC"));

    // Load input element
    dialect.emit_global_load(out, "val", "input", tid);

    // Emit trace body with input binding
    let bindings = vec!["val".to_string()];
    let result = emit_trace_body(dialect, out, body, 0, &bindings);

    // Store result
    dialect.emit_global_store(out, "output", tid, &result);

    dialect.emit_kernel_end(out);
}

// ── Binary elementwise kernel ───────────────────────────────────────────────

/// Emit a 2-input, 1-output elementwise kernel.
///
/// Layout: `output[tid] = f(A[tid], B[tid])` with bounds check on `N`.
pub fn build_binary_elementwise_kernel<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    body: &[TraceOp],
) {
    let params = vec![
        param_in("A"),
        param_in("B"),
        param_out("output"),
        param_uint("N"),
    ];
    dialect.emit_kernel_start(out, kernel_name, &params, 0);

    let tid = dialect.global_tid_expr();
    dialect.emit_bounds_check_return(out, tid, "N", &format!("{kernel_name}_BC"));

    // Load both inputs
    dialect.emit_global_load(out, "a_val", "A", tid);
    dialect.emit_global_load(out, "b_val", "B", tid);

    // Emit trace body with two input bindings
    let bindings = vec!["a_val".to_string(), "b_val".to_string()];
    let result = emit_trace_body(dialect, out, body, 0, &bindings);

    // Store result
    dialect.emit_global_store(out, "output", tid, &result);

    dialect.emit_kernel_end(out);
}

// ── Softmax kernel ──────────────────────────────────────────────────────────

/// Emit a row-wise softmax kernel using shared memory.
///
/// Layout: one threadgroup per row. Each thread handles strided elements.
/// Three passes: (1) find max, (2) exp + sum, (3) normalize.
pub fn build_softmax_kernel<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    block_size: u32,
) {
    let params = vec![
        param_in("input"),
        param_out("output"),
        param_uint("N"),
    ];
    let smem_bytes = block_size as usize * 4;
    dialect.emit_kernel_start(out, kernel_name, &params, smem_bytes);

    let lid = dialect.local_tid_expr();
    let gid = dialect.group_id_expr();
    let bs = format!("{block_size}");

    // Row offset
    dialect.emit_float_decl(out, "row_off", &format!("{gid} * N"));

    // ── Pass 1: find max ──
    dialect.emit_float_decl(out, "local_max", "-INFINITY");
    dialect.emit_for_start(out, "i", lid, "N", &bs, &format!("{kernel_name}_MAX"));
    dialect.emit_global_load(out, "v", "input", &format!("row_off + i"));
    writeln!(out, "    local_max = fmaxf(local_max, v);").unwrap();
    dialect.emit_for_end(out, "i", &bs, &format!("{kernel_name}_MAX"));

    dialect.emit_shared_store(out, "smem", lid, "local_max");
    emit_tree_reduce(dialect, out, block_size, "max", kernel_name, "MAX");
    dialect.emit_shared_load(out, "row_max", "smem", "0");

    // ── Pass 2: exp(x - max) and sum ──
    dialect.emit_float_decl(out, "local_sum", "0.0f");
    dialect.emit_for_start(out, "i", lid, "N", &bs, &format!("{kernel_name}_EXP"));
    dialect.emit_global_load(out, "v", "input", &format!("row_off + i"));
    writeln!(out, "    v = expf(v - row_max);").unwrap();
    dialect.emit_global_store(out, "output", &format!("row_off + i"), "v");
    writeln!(out, "    local_sum += v;").unwrap();
    dialect.emit_for_end(out, "i", &bs, &format!("{kernel_name}_EXP"));

    dialect.emit_shared_store(out, "smem", lid, "local_sum");
    emit_tree_reduce(dialect, out, block_size, "add", kernel_name, "SUM");
    dialect.emit_shared_load(out, "row_sum", "smem", "0");

    // ── Pass 3: normalize ──
    dialect.emit_for_start(out, "i", lid, "N", &bs, &format!("{kernel_name}_NORM"));
    dialect.emit_global_load(out, "v", "output", &format!("row_off + i"));
    writeln!(out, "    v = v / row_sum;").unwrap();
    dialect.emit_global_store(out, "output", &format!("row_off + i"), "v");
    dialect.emit_for_end(out, "i", &bs, &format!("{kernel_name}_NORM"));

    dialect.emit_kernel_end(out);
}

// ── NormLike kernel (RMSNorm / LayerNorm) ───────────────────────────────────

/// Emit a norm-like kernel (RMSNorm, LayerNorm).
///
/// Three-phase pattern:
///   1. Reduce: accumulate per-element values (e.g. x*x for RMS)
///   2. Finalize: compute normalization factor from reduced value
///   3. Transform: apply per-element transformation
pub fn build_normlike_kernel<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    reduce_body: &[TraceOp],
    finalize_body: &[TraceOp],
    transform_body: &[TraceOp],
    has_weight: bool,
    has_bias: bool,
    eps: Option<f64>,
    block_size: u32,
) {
    let mut params = vec![
        param_in("input"),
        param_out("output"),
        param_uint("N"),
    ];
    if has_weight {
        params.push(param_in("weight"));
    }
    if has_bias {
        params.push(param_in("bias"));
    }

    let smem_bytes = block_size as usize * 4;
    let bs = format!("{block_size}");
    dialect.emit_kernel_start(out, kernel_name, &params, smem_bytes);

    let lid = dialect.local_tid_expr();
    let gid = dialect.group_id_expr();

    // Row offset
    dialect.emit_float_decl(out, "row_off", &format!("{gid} * N"));

    // Phase 1: Reduce
    dialect.emit_float_decl(out, "partial", "0.0f");
    dialect.emit_for_start(out, "i", lid, "N", &bs, &format!("{kernel_name}_RED"));
    dialect.emit_global_load(out, "x", "input", &format!("row_off + i"));

    let reduce_bindings = vec!["x".to_string()];
    let reduce_result = emit_trace_body(dialect, out, reduce_body, 0, &reduce_bindings);
    writeln!(out, "    partial += {reduce_result};").unwrap();

    dialect.emit_for_end(out, "i", &bs, &format!("{kernel_name}_RED"));

    // Shared-memory tree reduction for sum
    dialect.emit_shared_store(out, "smem", lid, "partial");
    emit_tree_reduce(dialect, out, block_size, "add", kernel_name, "RED");

    // Phase 2: Finalize — compute normalization factor
    dialect.emit_shared_load(out, "reduced", "smem", "0");
    writeln!(out, "    reduced = reduced / float(N);").unwrap();

    if let Some(eps_val) = eps {
        writeln!(out, "    reduced = reduced + {eps_val:e}f;").unwrap();
    }

    let finalize_bindings = vec!["reduced".to_string()];
    let norm_factor = emit_trace_body(dialect, out, finalize_body, 1, &finalize_bindings);

    // Phase 3: Transform — apply normalization per element
    dialect.emit_for_start(out, "i", lid, "N", &bs, &format!("{kernel_name}_XFORM"));
    dialect.emit_global_load(out, "x", "input", &format!("row_off + i"));

    let mut xform_bindings = vec!["x".to_string(), norm_factor.clone()];
    if has_weight {
        dialect.emit_global_load(out, "w", "weight", "i");
        xform_bindings.push("w".to_string());
    }
    if has_bias {
        dialect.emit_global_load(out, "bi", "bias", "i");
        xform_bindings.push("bi".to_string());
    }

    let xform_result = emit_trace_body(dialect, out, transform_body, 2, &xform_bindings);
    dialect.emit_global_store(out, "output", &format!("row_off + i"), &xform_result);

    dialect.emit_for_end(out, "i", &bs, &format!("{kernel_name}_XFORM"));

    dialect.emit_kernel_end(out);
}

// ── MeanPool kernel ─────────────────────────────────────────────────────────

/// Emit a mean-pooling kernel: average over seq_len dimension.
///
/// Layout: one thread per hidden dimension. Each thread sums over seq_len
/// and divides by seq_len.
pub fn build_meanpool_kernel<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    seq_len: usize,
    hidden: usize,
) {
    let params = vec![
        param_in("input"),
        param_out("output"),
    ];
    dialect.emit_kernel_start(out, kernel_name, &params, 0);

    let tid = dialect.global_tid_expr();
    dialect.emit_bounds_check_return(out, tid, &format!("{hidden}u"), &format!("{kernel_name}_BC"));

    dialect.emit_float_decl(out, "acc", "0.0f");
    dialect.emit_for_start(out, "s", "0", &format!("{seq_len}u"), "1", &format!("{kernel_name}_SUM"));
    dialect.emit_global_load(out, "v", "input", &format!("s * {hidden}u + {tid}"));
    writeln!(out, "    acc += v;").unwrap();
    dialect.emit_for_end(out, "s", "1", &format!("{kernel_name}_SUM"));

    let inv = 1.0f32 / seq_len as f32;
    writeln!(out, "    acc *= {inv:e}f;").unwrap();
    dialect.emit_global_store(out, "output", tid, "acc");

    dialect.emit_kernel_end(out);
}

// ── Dequantize kernel ───────────────────────────────────────────────────────

/// Emit a dequantization kernel (packed int -> float).
pub fn build_dequantize_kernel<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    kernel_name: &str,
    num_elements: usize,
    group_size: usize,
    bits: usize,
) {
    let mask = (1u64 << bits) - 1;
    let elems_per_u32 = 32 / bits;

    let params = vec![
        param_in("packed"),
        param_in("scales"),
        param_out("output"),
    ];
    dialect.emit_kernel_start(out, kernel_name, &params, 0);

    let tid = dialect.global_tid_expr();
    dialect.emit_bounds_check_return(out, tid, &format!("{num_elements}u"), &format!("{kernel_name}_BC"));

    writeln!(out, "    uint GROUP_SIZE = {group_size}u;").unwrap();
    writeln!(out, "    uint BITS = {bits}u;").unwrap();
    writeln!(out, "    uint MASK = {mask}u;").unwrap();
    writeln!(out, "    uint ELEMS_PER_U32 = {elems_per_u32}u;").unwrap();
    writeln!(out, "    uint block_idx = {tid} / GROUP_SIZE;").unwrap();
    writeln!(out, "    uint in_block = {tid} % GROUP_SIZE;").unwrap();
    writeln!(out, "    float scale = scales[block_idx];").unwrap();
    writeln!(out, "    uint packed_idx = (block_idx * GROUP_SIZE + in_block) / ELEMS_PER_U32;").unwrap();
    writeln!(out, "    uint bit_offset = (in_block % ELEMS_PER_U32) * BITS;").unwrap();
    writeln!(out, "    uint raw = (packed[packed_idx] >> bit_offset) & MASK;").unwrap();
    writeln!(out, "    float zero_point = float(1u << (BITS - 1u));").unwrap();
    writeln!(out, "    output[{tid}] = (float(raw) - zero_point) * scale;").unwrap();

    dialect.emit_kernel_end(out);
}

// ── Helper: shared-memory tree reduction ────────────────────────────────────

/// Emit a shared-memory tree reduction.
///
/// Precondition: `smem[lid]` is populated with per-thread partial values.
/// Postcondition: `smem[0]` holds the fully reduced result.
fn emit_tree_reduce<D: GpuDialect>(
    dialect: &D,
    out: &mut String,
    block_size: u32,
    combine: &str,
    kernel_name: &str,
    phase: &str,
) {
    dialect.emit_barrier(out);
    let lid = dialect.local_tid_expr();
    let mut s = block_size / 2;
    while s > 0 {
        let label = format!("{kernel_name}_R{phase}_{s}");
        dialect.emit_if_start(out, &format!("{lid} < {s}"), &label);

        dialect.emit_shared_load(out, "r_a", "smem", lid);
        dialect.emit_shared_load(out, "r_b", "smem", &format!("{lid} + {s}"));
        match combine {
            "max" => writeln!(out, "    r_a = fmaxf(r_a, r_b);").unwrap(),
            _ => writeln!(out, "    r_a = r_a + r_b;").unwrap(),
        }
        dialect.emit_shared_store(out, "smem", lid, "r_a");

        dialect.emit_if_end(out, &label);
        dialect.emit_barrier(out);
        s /= 2;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trace_emitter::{HipDialect, MslDialect};
    use crate::compiler::trace::TraceOp;

    /// SiLU body: x / (1 + exp(-x))
    fn silu_body() -> Vec<TraceOp> {
        vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
            TraceOp::Exp(1),
            TraceOp::Const(1.0),
            TraceOp::Add(2, 3),
            TraceOp::Div(0, 4),
        ]
    }

    /// Simple add body: a + b
    fn add_body() -> Vec<TraceOp> {
        vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(0, 1),
        ]
    }

    #[test]
    fn hip_elementwise_kernel_compiles() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_elementwise_kernel(&dialect, &mut out, "silu_kernel", &silu_body());

        assert!(out.contains("extern \"C\" __global__ void silu_kernel("), "missing kernel sig:\n{out}");
        assert!(out.contains("const float* __restrict__ input"), "missing input param:\n{out}");
        assert!(out.contains("float* __restrict__ output"), "missing output param:\n{out}");
        assert!(out.contains("expf("), "missing expf:\n{out}");
        assert!(out.contains("output["), "missing store:\n{out}");
    }

    #[test]
    fn msl_elementwise_kernel_compiles() {
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_elementwise_kernel(&dialect, &mut out, "silu_kernel", &silu_body());

        assert!(out.contains("kernel void silu_kernel("), "missing kernel sig:\n{out}");
        assert!(out.contains("device const float* input"), "missing input param:\n{out}");
        assert!(out.contains("device float* output"), "missing output param:\n{out}");
        assert!(out.contains("exp("), "missing exp:\n{out}");
    }

    #[test]
    fn hip_binary_elementwise_kernel_compiles() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_binary_elementwise_kernel(&dialect, &mut out, "add_kernel", &add_body());

        assert!(out.contains("extern \"C\" __global__ void add_kernel("), "missing kernel sig:\n{out}");
        assert!(out.contains("const float* __restrict__ A"), "missing A param:\n{out}");
        assert!(out.contains("const float* __restrict__ B"), "missing B param:\n{out}");
    }

    #[test]
    fn msl_binary_elementwise_kernel_compiles() {
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_binary_elementwise_kernel(&dialect, &mut out, "add_kernel", &add_body());

        assert!(out.contains("kernel void add_kernel("), "missing kernel sig:\n{out}");
        assert!(out.contains("device const float* A"), "missing A param:\n{out}");
    }

    #[test]
    fn hip_softmax_kernel_structure() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_softmax_kernel(&dialect, &mut out, "softmax_k", 256);

        assert!(out.contains("fmaxf("), "missing max reduction:\n{out}");
        assert!(out.contains("expf("), "missing exp:\n{out}");
        assert!(out.contains("__syncthreads()"), "missing barrier:\n{out}");
        assert!(out.contains("__shared__ float smem["), "missing shared mem:\n{out}");
    }

    #[test]
    fn msl_softmax_kernel_structure() {
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_softmax_kernel(&dialect, &mut out, "softmax_k", 256);

        assert!(out.contains("threadgroup_barrier"), "missing barrier:\n{out}");
        assert!(out.contains("threadgroup float smem["), "missing shared mem:\n{out}");
    }

    #[test]
    fn hip_meanpool_kernel_structure() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_meanpool_kernel(&dialect, &mut out, "meanpool_k", 128, 768);

        assert!(out.contains("extern \"C\" __global__ void meanpool_k("), "missing sig:\n{out}");
        assert!(out.contains("768u"), "missing hidden dim:\n{out}");
        assert!(out.contains("128u"), "missing seq_len:\n{out}");
    }

    #[test]
    fn msl_meanpool_kernel_structure() {
        let dialect = MslDialect::new(9);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_meanpool_kernel(&dialect, &mut out, "meanpool_k", 128, 768);

        assert!(out.contains("kernel void meanpool_k("), "missing sig:\n{out}");
    }

    #[test]
    fn hip_dequantize_kernel_structure() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);
        build_dequantize_kernel(&dialect, &mut out, "deq_k", 1024, 32, 4);

        assert!(out.contains("extern \"C\" __global__ void deq_k("), "missing sig:\n{out}");
        assert!(out.contains("MASK = 15u"), "missing mask for 4-bit:\n{out}");
        assert!(out.contains("ELEMS_PER_U32 = 8u"), "missing elems_per_u32:\n{out}");
    }

    #[test]
    fn normlike_kernel_rmsnorm_structure() {
        let dialect = HipDialect::new(908);
        let mut out = String::new();
        dialect.emit_header(&mut out);

        // RMSNorm reduce: x * x
        let reduce = vec![
            TraceOp::Input(0),
            TraceOp::Mul(0, 0),
        ];
        // Finalize: rsqrt
        let finalize = vec![
            TraceOp::Input(0),
            TraceOp::Rsqrt(0),
        ];
        // Transform: x * norm_factor * weight
        let transform = vec![
            TraceOp::Input(0),  // x
            TraceOp::Input(1),  // norm_factor
            TraceOp::Mul(0, 1),
            TraceOp::Input(2),  // weight
            TraceOp::Mul(2, 3),
        ];

        build_normlike_kernel(
            &dialect, &mut out, "rmsnorm_k",
            &reduce, &finalize, &transform,
            true, false, Some(1e-6), 256,
        );

        assert!(out.contains("extern \"C\" __global__ void rmsnorm_k("), "missing sig:\n{out}");
        assert!(out.contains("__shared__ float smem["), "missing shared mem:\n{out}");
        assert!(out.contains("rsqrtf("), "missing rsqrt:\n{out}");
        assert!(out.contains("weight"), "missing weight param:\n{out}");
    }
}
