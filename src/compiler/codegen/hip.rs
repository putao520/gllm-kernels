//! HipCodeGen — Phase 3 HIP C++ code generation for AMD GPUs.
//!
//! Implements `MachineCodeEmitter` to produce HIP C++ kernel source from a `FusionPlan`.
//! The HIP source is later compiled via `hiprtcCompileProgram` or `hipcc` to native
//! AMDGPU ISA for the target GFX architecture.
//!
//! This is the AMD counterpart of `PtxCodeGen` (NVIDIA CUDA).
//!
//! Gated behind `#[cfg(feature = "jit-hip")]`.

use std::fmt::Write;

use crate::compiler::codegen::emitter::{MachineCodeEmitter, PlatformBackend, Platform};
use crate::compiler::codegen::CodegenOutput;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::{ComputePattern, TraceOp};
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::dispatch::DeviceProfile;

// ── HipCodeGen ──────────────────────────────────────────────────────

/// HIP C++ code generator targeting AMD GPUs.
///
/// Generates HIP C++ compute kernel source that is compiled to native AMDGPU
/// ISA via hiprtc or hipcc for the target GFX architecture.
pub struct HipCodeGen {
    /// Target GFX architecture (e.g. 908 = gfx908 / MI100, 1100 = gfx1100 / RX 7900).
    gfx_arch: u32,
    /// Accumulated HIP C++ source buffer.
    hip_buffer: String,
}

impl HipCodeGen {
    /// Create a new HipCodeGen targeting the given GFX architecture.
    pub fn new(gfx_arch: u32) -> Self {
        Self {
            gfx_arch,
            hip_buffer: String::with_capacity(4096),
        }
    }

    /// Target GFX architecture.
    pub fn gfx_arch(&self) -> u32 {
        self.gfx_arch
    }

    /// Return the accumulated HIP C++ source text.
    pub fn hip_source(&self) -> &str {
        &self.hip_buffer
    }

    // ── HIP generation helpers ──────────────────────────────────────

    /// Emit the HIP C++ header (includes).
    pub fn emit_header(&mut self) {
        self.hip_buffer.clear();
        writeln!(self.hip_buffer, "#include <hip/hip_runtime.h>").unwrap();
        writeln!(self.hip_buffer, "#include <hip/hip_fp16.h>").unwrap();
        writeln!(self.hip_buffer).unwrap();
    }

    /// Emit a simple elementwise kernel that applies `op` to each element.
    ///
    /// Supported ops: "add", "mul", "silu", "relu", "neg".
    pub fn emit_elementwise_kernel(&mut self, op: &str) -> Result<(), String> {
        let kernel_name = format!("kernel_{op}");

        writeln!(self.hip_buffer, "extern \"C\" __global__ void {kernel_name}(").unwrap();
        writeln!(self.hip_buffer, "    const float* __restrict__ input,").unwrap();
        writeln!(self.hip_buffer, "    float* __restrict__ output,").unwrap();
        writeln!(self.hip_buffer, "    const unsigned int n").unwrap();
        writeln!(self.hip_buffer, ") {{").unwrap();
        writeln!(self.hip_buffer, "    unsigned int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;").unwrap();
        writeln!(self.hip_buffer, "    if (tid >= n) return;").unwrap();
        writeln!(self.hip_buffer, "    float x = input[tid];").unwrap();
        writeln!(self.hip_buffer, "    float result;").unwrap();

        match op {
            "add" => {
                writeln!(self.hip_buffer, "    result = x + x;").unwrap();
            }
            "mul" => {
                writeln!(self.hip_buffer, "    result = x * x;").unwrap();
            }
            "neg" => {
                writeln!(self.hip_buffer, "    result = -x;").unwrap();
            }
            "relu" => {
                writeln!(self.hip_buffer, "    result = fmaxf(x, 0.0f);").unwrap();
            }
            "silu" => {
                writeln!(self.hip_buffer, "    result = x / (1.0f + expf(-x));").unwrap();
            }
            _ => {
                return Err(format!("HipCodeGen: unsupported elementwise op '{op}'"));
            }
        }

        writeln!(self.hip_buffer, "    output[tid] = result;").unwrap();
        writeln!(self.hip_buffer, "}}").unwrap();
        writeln!(self.hip_buffer).unwrap();

        Ok(())
    }
}

// ── Trace-based HIP C++ emission ────────────────────────────────────

/// Emit a single TraceOp as HIP C++ and return the variable name holding the result.
fn trace_op_to_hip(
    out: &mut String,
    op: &TraceOp,
    idx: usize,
    vars: &[String],
    tier: u32,
) -> String {
    let dst = format!("t{tier}_{idx}");
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
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            writeln!(out, "    float {dst} = {va} + {vb};").unwrap();
        }
        TraceOp::Sub(a, b) => {
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            writeln!(out, "    float {dst} = {va} - {vb};").unwrap();
        }
        TraceOp::Mul(a, b) => {
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            writeln!(out, "    float {dst} = {va} * {vb};").unwrap();
        }
        TraceOp::Div(a, b) => {
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            writeln!(out, "    float {dst} = {va} / {vb};").unwrap();
        }
        TraceOp::Fma(a, b, c) => {
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            let vc = &vars[*c as usize];
            writeln!(out, "    float {dst} = fmaf({va}, {vb}, {vc});").unwrap();
        }
        TraceOp::Neg(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = -{va};").unwrap();
        }
        TraceOp::Abs(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = fabsf({va});").unwrap();
        }
        TraceOp::Exp(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = expf({va});").unwrap();
        }
        TraceOp::Sqrt(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = sqrtf({va});").unwrap();
        }
        TraceOp::Rsqrt(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = rsqrtf({va});").unwrap();
        }
        TraceOp::Tanh(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = tanhf({va});").unwrap();
        }
        TraceOp::Recip(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = 1.0f / {va};").unwrap();
        }
        TraceOp::Log(a) => {
            let va = &vars[*a as usize];
            writeln!(out, "    float {dst} = logf({va});").unwrap();
        }
        TraceOp::Max(a, b) => {
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            writeln!(out, "    float {dst} = fmaxf({va}, {vb});").unwrap();
        }
        TraceOp::Min(a, b) => {
            let va = &vars[*a as usize];
            let vb = &vars[*b as usize];
            writeln!(out, "    float {dst} = fminf({va}, {vb});").unwrap();
        }
    }
    dst
}

/// Emit a trace body (sequence of TraceOps) as HIP C++ and return the last variable name.
fn emit_trace_body_hip(
    out: &mut String,
    body: &[TraceOp],
    tier: u32,
    bindings: &[String],
) -> String {
    let mut vars: Vec<String> = bindings.to_vec();
    let mut last = String::new();
    for (i, op) in body.iter().enumerate() {
        let v = trace_op_to_hip(out, op, i, &vars, tier);
        last = v.clone();
        vars.push(v);
    }
    last
}

/// Compute the max register tier needed for a trace body.
fn max_regs_needed(body: &[TraceOp], base: u32) -> u32 {
    body.len() as u32 + base
}

// ── Kernel emitters ─────────────────────────────────────────────────

/// Emit a unary elementwise kernel from a trace body.
fn emit_elementwise_kernel_hip(out: &mut String, name: &str, body: &[TraceOp]) {
    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    writeln!(out, "    const float* __restrict__ input,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const unsigned int n").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    unsigned int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;").unwrap();
    writeln!(out, "    if (tid >= n) return;").unwrap();
    writeln!(out, "    float in0 = input[tid];").unwrap();

    let bindings = vec!["in0".to_string()];
    let result = emit_trace_body_hip(out, body, 0, &bindings);

    writeln!(out, "    output[tid] = {result};").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// Emit a binary elementwise kernel from a trace body.
fn emit_binary_elementwise_kernel_hip(out: &mut String, name: &str, body: &[TraceOp]) {
    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    writeln!(out, "    const float* __restrict__ input0,").unwrap();
    writeln!(out, "    const float* __restrict__ input1,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const unsigned int n").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    unsigned int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;").unwrap();
    writeln!(out, "    if (tid >= n) return;").unwrap();
    writeln!(out, "    float in0 = input0[tid];").unwrap();
    writeln!(out, "    float in1 = input1[tid];").unwrap();

    let bindings = vec!["in0".to_string(), "in1".to_string()];
    let result = emit_trace_body_hip(out, body, 0, &bindings);

    writeln!(out, "    output[tid] = {result};").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// Emit a multi-input/multi-output injective kernel from a trace body.
fn emit_injective_kernel_hip(
    out: &mut String,
    name: &str,
    body: &[TraceOp],
    num_inputs: usize,
    num_outputs: usize,
) {
    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    for i in 0..num_inputs {
        writeln!(out, "    const float* __restrict__ input{i},").unwrap();
    }
    for i in 0..num_outputs {
        let comma = if i + 1 < num_outputs { "," } else { "," };
        writeln!(out, "    float* __restrict__ output{i}{comma}").unwrap();
    }
    writeln!(out, "    const unsigned int n").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    unsigned int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;").unwrap();
    writeln!(out, "    if (tid >= n) return;").unwrap();

    let mut bindings = Vec::new();
    for i in 0..num_inputs {
        writeln!(out, "    float in{i} = input{i}[tid];").unwrap();
        bindings.push(format!("in{i}"));
    }

    let result = emit_trace_body_hip(out, body, 0, &bindings);

    // For single output, write the last trace result.
    // For multi-output, the trace body should produce enough values.
    if num_outputs == 1 {
        writeln!(out, "    output0[tid] = {result};").unwrap();
    } else {
        // Multi-output: last `num_outputs` trace ops are the outputs.
        let base = body.len().saturating_sub(num_outputs);
        for i in 0..num_outputs {
            let var = format!("t0_{}", base + i);
            writeln!(out, "    output{i}[tid] = {var};").unwrap();
        }
    }

    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// Wavefront size for the target GFX architecture.
///
/// RDNA (gfx10xx, gfx11xx) uses wave32; CDNA (gfx9xx) uses wave64.
fn wavefront_size(gfx_arch: u32) -> u32 {
    if gfx_arch >= 1000 {
        32 // RDNA wave32
    } else {
        64 // CDNA / GCN wave64
    }
}

/// Emit a reduction kernel (single-pass or multi-pass).
///
/// Uses shared memory + warp-shuffle reduction. The block size is chosen
/// based on the wavefront size of the target architecture.
fn emit_reduction_kernel_hip(
    out: &mut String,
    name: &str,
    identity: f64,
    combine: &[TraceOp],
    gfx_arch: u32,
) {
    let wf = wavefront_size(gfx_arch);
    let block_size = wf * 4; // 4 wavefronts per block
    let id_bits = (identity as f32).to_bits();

    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    writeln!(out, "    const float* __restrict__ input,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const unsigned int n").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    __shared__ float sdata[{block_size}];").unwrap();
    writeln!(out, "    unsigned int tid = hipThreadIdx_x;").unwrap();
    writeln!(out, "    unsigned int gid = hipBlockIdx_x * {block_size} + tid;").unwrap();
    writeln!(out, "    float acc = __uint_as_float(0x{id_bits:08X}u);").unwrap();
    writeln!(out).unwrap();

    // Grid-stride loop to accumulate into acc.
    writeln!(out, "    for (unsigned int i = gid; i < n; i += hipGridDim_x * {block_size}) {{").unwrap();
    writeln!(out, "        float in0 = acc;").unwrap();
    writeln!(out, "        float in1 = input[i];").unwrap();

    let bindings = vec!["in0".to_string(), "in1".to_string()];
    let result = emit_trace_body_hip(out, combine, 1, &bindings);
    writeln!(out, "        acc = {result};").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Shared memory reduction.
    writeln!(out, "    sdata[tid] = acc;").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    // Tree reduction in shared memory.
    let mut s = block_size / 2;
    while s > 0 {
        writeln!(out, "    if (tid < {s}) {{").unwrap();
        writeln!(out, "        float in0 = sdata[tid];").unwrap();
        writeln!(out, "        float in1 = sdata[tid + {s}];").unwrap();
        let bindings2 = vec!["in0".to_string(), "in1".to_string()];
        let r2 = emit_trace_body_hip(out, combine, 2, &bindings2);
        writeln!(out, "        sdata[tid] = {r2};").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    __syncthreads();").unwrap();
        s /= 2;
    }

    writeln!(out, "    if (tid == 0) output[hipBlockIdx_x] = sdata[0];").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

// ── GEMM kernel ─────────────────────────────────────────────────────

/// Emit a tiled GEMM kernel using shared memory.
///
/// C = alpha * A * B + beta * C
/// Uses a TILE_SIZE x TILE_SIZE blocking strategy.
fn emit_gemm_kernel_hip(
    out: &mut String,
    name: &str,
    tile_size: u32,
) {
    let ts = tile_size;
    writeln!(out, "#define TILE_SIZE {ts}").unwrap();
    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    writeln!(out, "    const float* __restrict__ A,").unwrap();
    writeln!(out, "    const float* __restrict__ B,").unwrap();
    writeln!(out, "    float* __restrict__ C,").unwrap();
    writeln!(out, "    const unsigned int M,").unwrap();
    writeln!(out, "    const unsigned int N,").unwrap();
    writeln!(out, "    const unsigned int K,").unwrap();
    writeln!(out, "    const float alpha,").unwrap();
    writeln!(out, "    const float beta").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    __shared__ float As[TILE_SIZE][TILE_SIZE];").unwrap();
    writeln!(out, "    __shared__ float Bs[TILE_SIZE][TILE_SIZE];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    unsigned int row = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;").unwrap();
    writeln!(out, "    unsigned int col = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;").unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (unsigned int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {{").unwrap();
    writeln!(out, "        unsigned int a_col = t * TILE_SIZE + hipThreadIdx_x;").unwrap();
    writeln!(out, "        unsigned int b_row = t * TILE_SIZE + hipThreadIdx_y;").unwrap();
    writeln!(out, "        As[hipThreadIdx_y][hipThreadIdx_x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;").unwrap();
    writeln!(out, "        Bs[hipThreadIdx_y][hipThreadIdx_x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;").unwrap();
    writeln!(out, "        __syncthreads();").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (unsigned int k = 0; k < TILE_SIZE; ++k) {{").unwrap();
    writeln!(out, "            acc = fmaf(As[hipThreadIdx_y][k], Bs[k][hipThreadIdx_x], acc);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        __syncthreads();").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (row < M && col < N) {{").unwrap();
    writeln!(out, "        unsigned int idx = row * N + col;").unwrap();
    writeln!(out, "        C[idx] = fmaf(alpha, acc, beta * C[idx]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "#undef TILE_SIZE").unwrap();
    writeln!(out).unwrap();
}

// ── Softmax kernel ──────────────────────────────────────────────────

/// Emit a row-wise softmax kernel using shared memory reduction.
fn emit_softmax_kernel_hip(out: &mut String, name: &str, gfx_arch: u32) {
    let wf = wavefront_size(gfx_arch);
    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    writeln!(out, "    const float* __restrict__ input,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const unsigned int rows,").unwrap();
    writeln!(out, "    const unsigned int cols").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    extern __shared__ float sdata[];").unwrap();
    writeln!(out, "    unsigned int row = hipBlockIdx_x;").unwrap();
    writeln!(out, "    unsigned int tid = hipThreadIdx_x;").unwrap();
    writeln!(out, "    if (row >= rows) return;").unwrap();
    writeln!(out).unwrap();

    // Phase 1: find row max
    writeln!(out, "    float local_max = -3.402823466e+38f;").unwrap();
    writeln!(out, "    for (unsigned int i = tid; i < cols; i += hipBlockDim_x) {{").unwrap();
    writeln!(out, "        local_max = fmaxf(local_max, input[row * cols + i]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sdata[tid] = local_max;").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out, "    for (unsigned int s = hipBlockDim_x / 2; s > 0; s >>= 1) {{").unwrap();
    writeln!(out, "        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);").unwrap();
    writeln!(out, "        __syncthreads();").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float row_max = sdata[0];").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    // Phase 2: compute exp(x - max) and sum
    writeln!(out, "    float local_sum = 0.0f;").unwrap();
    writeln!(out, "    for (unsigned int i = tid; i < cols; i += hipBlockDim_x) {{").unwrap();
    writeln!(out, "        float val = expf(input[row * cols + i] - row_max);").unwrap();
    writeln!(out, "        output[row * cols + i] = val;").unwrap();
    writeln!(out, "        local_sum += val;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sdata[tid] = local_sum;").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out, "    for (unsigned int s = hipBlockDim_x / 2; s > 0; s >>= 1) {{").unwrap();
    writeln!(out, "        if (tid < s) sdata[tid] += sdata[tid + s];").unwrap();
    writeln!(out, "        __syncthreads();").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float row_sum = sdata[0];").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    // Phase 3: normalize
    writeln!(out, "    float inv_sum = 1.0f / row_sum;").unwrap();
    writeln!(out, "    for (unsigned int i = tid; i < cols; i += hipBlockDim_x) {{").unwrap();
    writeln!(out, "        output[row * cols + i] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

// ── MachineCodeEmitter implementation ───────────────────────────────

impl MachineCodeEmitter for HipCodeGen {
    fn platform(&self) -> Platform {
        Platform::Hip
    }

    fn backend(&self) -> PlatformBackend {
        PlatformBackend::Hip {
            gfx_arch: self.gfx_arch,
        }
    }

    fn emit_plan(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
        registry: &ScalarOpRegistry,
        alloc: &BufferAllocation,
        device: &DeviceProfile,
    ) -> Result<CodegenOutput, String> {
        self.emit_header();

        let kernel_name = plan.name().unwrap_or("fused_kernel");
        let pattern = plan.compute_pattern();

        match pattern {
            ComputePattern::Elementwise { body, .. } => {
                let num_inputs = plan.num_inputs();
                let num_outputs = plan.num_outputs();
                if num_inputs == 1 {
                    emit_elementwise_kernel_hip(&mut self.hip_buffer, kernel_name, body);
                } else {
                    emit_binary_elementwise_kernel_hip(&mut self.hip_buffer, kernel_name, body);
                }
            }
            ComputePattern::Injective { body, num_inputs, num_outputs, .. } => {
                emit_injective_kernel_hip(
                    &mut self.hip_buffer,
                    kernel_name,
                    body,
                    *num_inputs,
                    *num_outputs,
                );
            }
            ComputePattern::Reduction { body, combine, identity, .. } => {
                emit_reduction_kernel_hip(
                    &mut self.hip_buffer,
                    kernel_name,
                    *identity,
                    combine,
                    self.gfx_arch,
                );
            }
            ComputePattern::Gemm { m, n, k, .. } => {
                let tile = if self.gfx_arch >= 1000 { 16 } else { 32 };
                emit_gemm_kernel_hip(&mut self.hip_buffer, kernel_name, tile);
            }
            ComputePattern::Softmax { .. } => {
                emit_softmax_kernel_hip(&mut self.hip_buffer, kernel_name, self.gfx_arch);
            }
            _ => {
                return Err(format!(
                    "HipCodeGen: unsupported compute pattern {:?}",
                    pattern
                ));
            }
        }

        Ok(CodegenOutput {
            source: self.hip_buffer.clone(),
            kernel_name: kernel_name.to_string(),
            platform: Platform::Hip,
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_header() {
        let mut cg = HipCodeGen::new(908);
        cg.emit_header();
        let src = cg.hip_source();
        assert!(src.contains("#include <hip/hip_runtime.h>"));
        assert!(src.contains("#include <hip/hip_fp16.h>"));
    }

    #[test]
    fn test_hip_elementwise_relu() {
        let mut cg = HipCodeGen::new(908);
        cg.emit_header();
        cg.emit_elementwise_kernel("relu").unwrap();
        let src = cg.hip_source();
        assert!(src.contains("kernel_relu"));
        assert!(src.contains("fmaxf(x, 0.0f)"));
        assert!(src.contains("hipBlockIdx_x"));
    }

    #[test]
    fn test_hip_elementwise_silu() {
        let mut cg = HipCodeGen::new(1100);
        cg.emit_header();
        cg.emit_elementwise_kernel("silu").unwrap();
        let src = cg.hip_source();
        assert!(src.contains("kernel_silu"));
        assert!(src.contains("expf(-x)"));
    }

    #[test]
    fn test_hip_unsupported_op() {
        let mut cg = HipCodeGen::new(908);
        cg.emit_header();
        let result = cg.emit_elementwise_kernel("unknown_op");
        assert!(result.is_err());
    }

    #[test]
    fn test_wavefront_size_cdna() {
        assert_eq!(wavefront_size(908), 64);
        assert_eq!(wavefront_size(942), 64);
    }

    #[test]
    fn test_wavefront_size_rdna() {
        assert_eq!(wavefront_size(1030), 32);
        assert_eq!(wavefront_size(1100), 32);
    }

    #[test]
    fn test_gfx_arch_accessor() {
        let cg = HipCodeGen::new(1100);
        assert_eq!(cg.gfx_arch(), 1100);
    }

    #[test]
    fn test_trace_op_add() {
        let mut out = String::new();
        let vars = vec!["a".to_string(), "b".to_string()];
        let dst = trace_op_to_hip(&mut out, &TraceOp::Add(0, 1), 0, &vars, 0);
        assert_eq!(dst, "t0_0");
        assert!(out.contains("a + b"));
    }

    #[test]
    fn test_trace_op_fma() {
        let mut out = String::new();
        let vars = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let dst = trace_op_to_hip(&mut out, &TraceOp::Fma(0, 1, 2), 0, &vars, 0);
        assert!(out.contains("fmaf(x, y, z)"));
    }

    #[test]
    fn test_emit_reduction_kernel() {
        let mut out = String::new();
        let combine = vec![TraceOp::Add(0, 1)];
        emit_reduction_kernel_hip(&mut out, "reduce_sum", 0.0, &combine, 908);
        assert!(out.contains("reduce_sum"));
        assert!(out.contains("__shared__"));
        assert!(out.contains("__syncthreads"));
    }

    #[test]
    fn test_emit_gemm_kernel() {
        let mut out = String::new();
        emit_gemm_kernel_hip(&mut out, "sgemm", 16);
        assert!(out.contains("sgemm"));
        assert!(out.contains("TILE_SIZE"));
        assert!(out.contains("fmaf("));
    }

    #[test]
    fn test_emit_softmax_kernel() {
        let mut out = String::new();
        emit_softmax_kernel_hip(&mut out, "softmax", 1100);
        assert!(out.contains("softmax"));
        assert!(out.contains("expf("));
        assert!(out.contains("inv_sum"));
    }
}
