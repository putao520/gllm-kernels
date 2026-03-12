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

/// Emit gfx908+ MFMA-based GEMM kernel using `__builtin_amdgcn_mfma_f32_16x16x16f16`.
///
/// Each wavefront (64 lanes) computes a 16×16 output tile via MFMA intrinsics.
/// Grid: (ceil(N/16), ceil(M/16)), Block: (64, 1).
fn emit_gemm_mfma_kernel_hip(out: &mut String, name: &str) {
    writeln!(out, "typedef _Float16 half8 __attribute__((ext_vector_type(8)));").unwrap();
    writeln!(out, "typedef float float4 __attribute__((ext_vector_type(4)));").unwrap();
    writeln!(out, "extern \"C\" float4 __builtin_amdgcn_mfma_f32_16x16x16f16(half8, half8, float4, int, int, int);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "extern \"C\" __global__ void {name}(").unwrap();
    writeln!(out, "    const _Float16* __restrict__ A,").unwrap();
    writeln!(out, "    const _Float16* __restrict__ B,").unwrap();
    writeln!(out, "    float* __restrict__ C,").unwrap();
    writeln!(out, "    const unsigned int M,").unwrap();
    writeln!(out, "    const unsigned int N,").unwrap();
    writeln!(out, "    const unsigned int K").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    unsigned int tile_row = hipBlockIdx_y;").unwrap();
    writeln!(out, "    unsigned int tile_col = hipBlockIdx_x;").unwrap();
    writeln!(out, "    unsigned int lane = hipThreadIdx_x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 acc = {{0.0f, 0.0f, 0.0f, 0.0f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (unsigned int kt = 0; kt < (K + 15) / 16; ++kt) {{").unwrap();
    writeln!(out, "        // Load A tile fragment (each lane loads 8 f16 values)").unwrap();
    writeln!(out, "        unsigned int a_row = tile_row * 16 + (lane / 16);").unwrap();
    writeln!(out, "        unsigned int a_col = kt * 16;").unwrap();
    writeln!(out, "        half8 fragA;").unwrap();
    writeln!(out, "        for (int i = 0; i < 8; ++i) {{").unwrap();
    writeln!(out, "            unsigned int c = a_col + (lane % 16 < 8 ? lane % 16 : lane % 16 - 8) + i;").unwrap();
    writeln!(out, "            fragA[i] = (a_row < M && c < K) ? A[a_row * K + c] : (_Float16)0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // Load B tile fragment").unwrap();
    writeln!(out, "        unsigned int b_row = kt * 16 + (lane / 16);").unwrap();
    writeln!(out, "        unsigned int b_col = tile_col * 16;").unwrap();
    writeln!(out, "        half8 fragB;").unwrap();
    writeln!(out, "        for (int i = 0; i < 8; ++i) {{").unwrap();
    writeln!(out, "            unsigned int c = b_col + (lane % 16 < 8 ? lane % 16 : lane % 16 - 8) + i;").unwrap();
    writeln!(out, "            fragB[i] = (b_row < K && c < N) ? B[b_row * N + c] : (_Float16)0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(fragA, fragB, acc, 0, 0, 0);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Store 4 output values per lane").unwrap();
    writeln!(out, "    unsigned int out_row = tile_row * 16 + (lane / 4);").unwrap();
    writeln!(out, "    unsigned int out_col = tile_col * 16 + (lane % 4) * 4;").unwrap();
    writeln!(out, "    if (out_row < M) {{").unwrap();
    writeln!(out, "        for (int i = 0; i < 4; ++i) {{").unwrap();
    writeln!(out, "            if (out_col + i < N) {{").unwrap();
    writeln!(out, "                C[out_row * N + out_col + i] = acc[i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// ── NormLike kernel ─────────────────────────────────────────────────

/// Emit a NormLike (RMSNorm / LayerNorm) kernel using 3 phases:
/// Phase 1: partial reduction (sum of squares), Phase 2: finalize (rsqrt),
/// Phase 3: per-element transform (scale * weight [+ bias]).
fn emit_normlike_kernel_hip(
    out: &mut String,
    kernel_name: &str,
    _reduce: &[TraceOp],
    _finalize: &[TraceOp],
    _transform: &[TraceOp],
    has_weight: bool,
    has_bias: bool,
    eps_override: Option<f32>,
) {
    let block_size = 256;

    write!(out, "extern \"C\" __global__ void {kernel_name}(\n").unwrap();
    writeln!(out, "    const float* __restrict__ input,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    if has_weight {
        writeln!(out, "    const float* __restrict__ weight,").unwrap();
    }
    if has_bias {
        writeln!(out, "    const float* __restrict__ bias,").unwrap();
    }
    writeln!(out, "    const int N").unwrap();
    writeln!(out, ") {{").unwrap();

    writeln!(out, "    __shared__ float smem[{block_size}];").unwrap();
    writeln!(out, "    int row = blockIdx.x;").unwrap();
    writeln!(out, "    int tid = threadIdx.x;").unwrap();
    writeln!(out, "    const float* row_in = input + row * N;").unwrap();
    writeln!(out, "    float* row_out = output + row * N;").unwrap();
    writeln!(out).unwrap();

    // Phase 1: Partial reduction
    writeln!(out, "    // Phase 1: reduction").unwrap();
    writeln!(out, "    float partial = 0.0f;").unwrap();
    writeln!(out, "    for (int i = tid; i < N; i += {block_size}) {{").unwrap();
    writeln!(out, "        float x = row_in[i];").unwrap();
    writeln!(out, "        partial += x * x;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    smem[tid] = partial;").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    // Tree reduction
    writeln!(out, "    for (int s = {block_size} / 2; s > 0; s >>= 1) {{").unwrap();
    writeln!(out, "        if (tid < s) smem[tid] += smem[tid + s];").unwrap();
    writeln!(out, "        __syncthreads();").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Phase 2: Finalize
    writeln!(out, "    // Phase 2: finalize").unwrap();
    let eps = eps_override.unwrap_or(1e-5);
    writeln!(out, "    float variance = smem[0] / float(N);").unwrap();
    writeln!(out, "    float scale = rsqrtf(variance + {eps:.8e}f);").unwrap();
    writeln!(out).unwrap();

    // Phase 3: Per-element transform
    writeln!(out, "    // Phase 3: transform").unwrap();
    writeln!(out, "    for (int i = tid; i < N; i += {block_size}) {{").unwrap();
    writeln!(out, "        float x = row_in[i];").unwrap();
    writeln!(out, "        float y = x * scale;").unwrap();
    if has_weight {
        writeln!(out, "        y *= weight[i];").unwrap();
    }
    if has_bias {
        writeln!(out, "        y += bias[i];").unwrap();
    }
    writeln!(out, "        row_out[i] = y;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// ── MeanPool kernel ────────────────────────────────────────────────

/// Emit a MeanPool kernel: average over the sequence dimension.
/// Each thread handles one hidden dimension element.
fn emit_meanpool_kernel_hip(
    out: &mut String,
    kernel_name: &str,
    seq_len: usize,
    hidden: usize,
) {
    let block_size = 256;
    writeln!(out, "extern \"C\" __global__ void {kernel_name}(").unwrap();
    writeln!(out, "    const float* __restrict__ input,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const int seq_len,").unwrap();
    writeln!(out, "    const int hidden").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    int h = blockIdx.x * {block_size} + threadIdx.x;").unwrap();
    writeln!(out, "    if (h >= {hidden}) return;").unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    for (int s = 0; s < {seq_len}; s++) {{").unwrap();
    writeln!(out, "        acc += input[s * {hidden} + h];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    output[h] = acc / float({seq_len});").unwrap();
    writeln!(out, "}}\n").unwrap();
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

// ── RoPE kernel ─────────────────────────────────────────────────────

/// Emit a Rotary Position Embedding (RoPE) kernel for HIP.
///
/// Applies rotary embeddings to input tensor with the given head dimension
/// and frequency base theta. Grid: (seq_len, num_heads), Block: (256).
fn emit_rope_kernel_hip(
    out: &mut String,
    kernel_name: &str,
    head_dim: usize,
    theta: f64,
) {
    let half_dim = head_dim / 2;
    let block_size = 256;
    writeln!(out, "extern \"C\" __global__ void {kernel_name}(").unwrap();
    writeln!(out, "    const float* __restrict__ input,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const int num_heads,").unwrap();
    writeln!(out, "    const int seq_len").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const int HEAD_DIM = {head_dim};").unwrap();
    writeln!(out, "    const int HALF_DIM = {half_dim};").unwrap();
    writeln!(out, "    const float THETA = {theta:e}f;").unwrap();
    writeln!(out, "    int pos = blockIdx.x;").unwrap();
    writeln!(out, "    int head = blockIdx.y;").unwrap();
    writeln!(out, "    int tid = threadIdx.x;").unwrap();
    writeln!(out, "    int base = (pos * num_heads + head) * HEAD_DIM;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i = tid; i < HALF_DIM; i += {block_size}) {{").unwrap();
    writeln!(out, "        float freq = 1.0f / powf(THETA, float(2 * i) / float(HEAD_DIM));").unwrap();
    writeln!(out, "        float angle = float(pos) * freq;").unwrap();
    writeln!(out, "        float cos_a = cosf(angle);").unwrap();
    writeln!(out, "        float sin_a = sinf(angle);").unwrap();
    writeln!(out, "        float x0 = input[base + i];").unwrap();
    writeln!(out, "        float x1 = input[base + i + HALF_DIM];").unwrap();
    writeln!(out, "        output[base + i] = x0 * cos_a - x1 * sin_a;").unwrap();
    writeln!(out, "        output[base + i + HALF_DIM] = x1 * cos_a + x0 * sin_a;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// ── MHA kernel ──────────────────────────────────────────────────────

/// Emit a Multi-Head Attention kernel for HIP.
///
/// 3-pass structure: (1) Q·K^T scores, (2) softmax, (3) weighted sum with V.
/// Grid: (seq_len, num_heads), Block: (head_dim.next_power_of_two().min(256)).
fn emit_mha_kernel_hip(
    out: &mut String,
    kernel_name: &str,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let block_size = head_dim.next_power_of_two().min(256);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    writeln!(out, "extern \"C\" __global__ void {kernel_name}(").unwrap();
    writeln!(out, "    const float* __restrict__ Q,").unwrap();
    writeln!(out, "    const float* __restrict__ K,").unwrap();
    writeln!(out, "    const float* __restrict__ V,").unwrap();
    writeln!(out, "    float* __restrict__ output,").unwrap();
    writeln!(out, "    const int seq_len,").unwrap();
    writeln!(out, "    const int num_heads,").unwrap();
    writeln!(out, "    const int head_dim").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const float scale = {scale:.8e}f;").unwrap();
    writeln!(out, "    __shared__ float smem_scores[{seq_len}];").unwrap();
    writeln!(out, "    __shared__ float smem_max[1];").unwrap();
    writeln!(out, "    __shared__ float smem_sum[1];").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    int query_idx = blockIdx.x;").unwrap();
    writeln!(out, "    int head_idx = blockIdx.y;").unwrap();
    writeln!(out, "    int tid = threadIdx.x;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    int q_offset = (query_idx * {num_heads} + head_idx) * {head_dim};").unwrap();
    writeln!(out).unwrap();

    // Pass 1: Q·K^T scores
    writeln!(out, "    // Pass 1: compute attention scores").unwrap();
    writeln!(out, "    for (int j = tid; j < {seq_len}; j += {block_size}) {{").unwrap();
    writeln!(out, "        int k_offset = (j * {num_heads} + head_idx) * {head_dim};").unwrap();
    writeln!(out, "        float dot = 0.0f;").unwrap();
    writeln!(out, "        for (int d = 0; d < {head_dim}; d++) {{").unwrap();
    writeln!(out, "            dot += Q[q_offset + d] * K[k_offset + d];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        smem_scores[j] = dot * scale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    // Pass 2: Softmax
    writeln!(out, "    // Pass 2: softmax").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float max_val = smem_scores[0];").unwrap();
    writeln!(out, "        for (int j = 1; j < {seq_len}; j++) {{").unwrap();
    writeln!(out, "            max_val = fmaxf(max_val, smem_scores[j]);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        smem_max[0] = max_val;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    for (int j = tid; j < {seq_len}; j += {block_size}) {{").unwrap();
    writeln!(out, "        smem_scores[j] = expf(smem_scores[j] - smem_max[0]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float sum = 0.0f;").unwrap();
    writeln!(out, "        for (int j = 0; j < {seq_len}; j++) {{").unwrap();
    writeln!(out, "            sum += smem_scores[j];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        float inv_sum = 1.0f / sum;").unwrap();
    writeln!(out, "        for (int j = 0; j < {seq_len}; j++) {{").unwrap();
    writeln!(out, "            smem_scores[j] *= inv_sum;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    __syncthreads();").unwrap();
    writeln!(out).unwrap();

    // Pass 3: Weighted sum
    writeln!(out, "    // Pass 3: weighted sum").unwrap();
    writeln!(out, "    int out_offset = (query_idx * {num_heads} + head_idx) * {head_dim};").unwrap();
    writeln!(out, "    for (int d = tid; d < {head_dim}; d += {block_size}) {{").unwrap();
    writeln!(out, "        float acc = 0.0f;").unwrap();
    writeln!(out, "        for (int j = 0; j < {seq_len}; j++) {{").unwrap();
    writeln!(out, "            int v_offset = (j * {num_heads} + head_idx) * {head_dim};").unwrap();
    writeln!(out, "            acc += smem_scores[j] * V[v_offset + d];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        output[out_offset + d] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// ── QuantDecode kernel ──────────────────────────────────────────────

/// Emit a dequantization kernel for HIP.
///
/// Unpacks N-bit quantized values from packed u32 words, applies scale and
/// zero-point offset to produce float output.
fn emit_dequantize_kernel_hip(
    out: &mut String,
    kernel_name: &str,
    num_elements: usize,
    block_size: usize,
    bits: usize,
) {
    let mask = (1u64 << bits) - 1;
    let elems_per_u32 = 32 / bits;
    let hip_block: usize = 256;
    let zero_point = 1u32 << (bits - 1);

    writeln!(out, "extern \"C\" __global__ void {kernel_name}(").unwrap();
    writeln!(out, "    const unsigned int* __restrict__ packed,").unwrap();
    writeln!(out, "    const float* __restrict__ scales,").unwrap();
    writeln!(out, "    float* __restrict__ output").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const unsigned int NUM_ELEMENTS = {num_elements}u;").unwrap();
    writeln!(out, "    const unsigned int BLOCK_SIZE = {block_size}u;").unwrap();
    writeln!(out, "    const unsigned int BITS = {bits}u;").unwrap();
    writeln!(out, "    const unsigned int MASK = {mask}u;").unwrap();
    writeln!(out, "    const unsigned int ELEMS_PER_U32 = {elems_per_u32}u;").unwrap();
    writeln!(out, "    unsigned int tid = blockIdx.x * {hip_block}u + threadIdx.x;").unwrap();
    writeln!(out, "    if (tid >= NUM_ELEMENTS) return;").unwrap();
    writeln!(out, "    unsigned int block_idx = tid / BLOCK_SIZE;").unwrap();
    writeln!(out, "    unsigned int in_block = tid % BLOCK_SIZE;").unwrap();
    writeln!(out, "    float scale = scales[block_idx];").unwrap();
    writeln!(out, "    unsigned int packed_idx = tid / ELEMS_PER_U32;").unwrap();
    writeln!(out, "    unsigned int bit_offset = (in_block % ELEMS_PER_U32) * BITS;").unwrap();
    writeln!(out, "    unsigned int raw = (packed[packed_idx] >> bit_offset) & MASK;").unwrap();
    writeln!(out, "    float zero_point = float({zero_point}u);").unwrap();
    writeln!(out, "    output[tid] = (float(raw) - zero_point) * scale;").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// ── MachineCodeEmitter implementation ───────────────────────────────

impl MachineCodeEmitter for HipCodeGen {
    fn emit_plan(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
        _alloc: &BufferAllocation,
        _profile: &DeviceProfile,
        registry: Option<&ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        self.emit_header();

        if plan.groups.is_empty() {
            return Ok(CodegenOutput {
                code: self.hip_buffer.clone().into_bytes(),
                scratchpad_bytes: 0,
            });
        }

        for group in &plan.groups {
            let anchor_op = graph.op(group.anchor).ok_or_else(|| {
                format!("HipCodeGen: anchor op {:?} not found in graph", group.anchor)
            })?;

            let kernel_name = format!("group_{}", group.id);
            let op_kind = &anchor_op.kind;

            // Reshape/Transpose are metadata-only — NOP on GPU
            if matches!(op_kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
                continue;
            }

            let registry = registry.ok_or_else(|| {
                format!("HipCodeGen: emit_plan requires a ScalarOpRegistry for {:?}", op_kind)
            })?;
            let key = ScalarOpRegistry::key_from_op_kind(op_kind);
            let trace = registry.get_trace(&key).ok_or_else(|| {
                format!("HipCodeGen: no OpTrace for {:?}", op_kind)
            })?;

            match &trace.pattern {
                ComputePattern::Elementwise { body } => {
                    emit_elementwise_kernel_hip(&mut self.hip_buffer, &kernel_name, body);
                }
                ComputePattern::BinaryElementwise { body } => {
                    emit_binary_elementwise_kernel_hip(&mut self.hip_buffer, &kernel_name, body);
                }
                ComputePattern::Injective { body, num_inputs, num_outputs } => {
                    match op_kind {
                        OpKind::RoPE { head_dim, theta } => {
                            emit_rope_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                *head_dim,
                                *theta,
                            );
                        }
                        _ => {
                            if body.is_empty() {
                                continue;
                            }
                            emit_injective_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                body,
                                *num_inputs,
                                *num_outputs,
                            );
                        }
                    }
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
                    emit_normlike_kernel_hip(
                        &mut self.hip_buffer,
                        &kernel_name,
                        reduce,
                        finalize,
                        transform,
                        has_weight,
                        has_bias,
                        eps_override,
                    );
                }
                ComputePattern::Reduction { combine, identity, .. } => {
                    match op_kind {
                        OpKind::Softmax => {
                            emit_softmax_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                self.gfx_arch,
                            );
                        }
                        OpKind::MeanPool { seq_len, hidden } => {
                            emit_meanpool_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                *seq_len,
                                *hidden,
                            );
                        }
                        _ => {
                            emit_reduction_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                *identity,
                                combine,
                                self.gfx_arch,
                            );
                        }
                    }
                }
                ComputePattern::Gemm => {
                    match op_kind {
                        OpKind::Gemm { .. } | OpKind::GemmBias { .. } => {
                            if self.gfx_arch >= 908 {
                                emit_gemm_mfma_kernel_hip(&mut self.hip_buffer, &kernel_name);
                            } else {
                                let tile = if self.gfx_arch >= 1000 { 16 } else { 32 };
                                emit_gemm_kernel_hip(&mut self.hip_buffer, &kernel_name, tile);
                            }
                        }
                        OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                            emit_mha_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                *seq_len,
                                *num_heads,
                                *head_dim,
                            );
                        }
                        _ => {
                            return Err(format!(
                                "HipCodeGen: unsupported Gemm-pattern op {:?}",
                                op_kind
                            ));
                        }
                    }
                }
                ComputePattern::QuantDecode { block_size, decode: _ } => {
                    match op_kind {
                        OpKind::Dequantize { num_elements, block_size: bs, bits } => {
                            emit_dequantize_kernel_hip(
                                &mut self.hip_buffer,
                                &kernel_name,
                                *num_elements,
                                *bs,
                                *bits,
                            );
                        }
                        _ => {
                            return Err(format!(
                                "HipCodeGen: unsupported QuantDecode op {:?}",
                                op_kind
                            ));
                        }
                    }
                }
            }
        }

        Ok(CodegenOutput {
            code: self.hip_buffer.clone().into_bytes(),
            scratchpad_bytes: 0,
        })
    }

    fn simd_width(&self) -> usize {
        // HIP wavefront width in f32 elements
        wavefront_size(self.gfx_arch) as usize
    }
}

// ── HipBackend ──────────────────────────────────────────────────────

/// HIP platform backend factory.
pub struct HipBackend {
    gfx_arch: u32,
}

impl HipBackend {
    pub fn new(gfx_arch: u32) -> Self {
        Self { gfx_arch }
    }
}

impl PlatformBackend for HipBackend {
    type Emitter = HipCodeGen;

    fn new_emitter(&self, _profile: &DeviceProfile) -> HipCodeGen {
        HipCodeGen::new(self.gfx_arch)
    }

    fn platform(&self) -> Platform {
        Platform::Hip { gfx_arch: self.gfx_arch }
    }

    fn num_simd_regs(&self) -> usize {
        256
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

    #[test]
    fn test_emit_normlike_kernel_rmsnorm() {
        let mut out = String::new();
        emit_normlike_kernel_hip(
            &mut out, "rmsnorm", &[], &[], &[],
            true,  // has_weight
            false, // no bias
            Some(1e-5),
        );
        assert!(out.contains("rmsnorm"));
        assert!(out.contains("rsqrtf("));
        assert!(out.contains("weight[i]"));
        assert!(!out.contains("bias[i]"));
        assert!(out.contains("__shared__"));
    }

    #[test]
    fn test_emit_normlike_kernel_layernorm() {
        let mut out = String::new();
        emit_normlike_kernel_hip(
            &mut out, "layernorm", &[], &[], &[],
            true, // has_weight
            true, // has_bias
            Some(1e-6),
        );
        assert!(out.contains("layernorm"));
        assert!(out.contains("weight[i]"));
        assert!(out.contains("bias[i]"));
    }

    #[test]
    fn test_emit_meanpool_kernel() {
        let mut out = String::new();
        emit_meanpool_kernel_hip(&mut out, "meanpool", 128, 768);
        assert!(out.contains("meanpool"));
        assert!(out.contains("128"));
        assert!(out.contains("768"));
        assert!(out.contains("float(128)"));
    }

    #[test]
    fn test_emit_binary_elementwise_kernel() {
        let mut out = String::new();
        let body = vec![TraceOp::Add(0, 1)];
        emit_binary_elementwise_kernel_hip(&mut out, "binary_add", &body);
        assert!(out.contains("binary_add"));
        assert!(out.contains("input0"));
        assert!(out.contains("input1"));
        assert!(out.contains("in0"));
        assert!(out.contains("in1"));
    }

    #[test]
    fn test_emit_gemm_mfma_kernel() {
        let mut out = String::new();
        emit_gemm_mfma_kernel_hip(&mut out, "mfma_gemm");
        assert!(out.contains("mfma_gemm"));
        assert!(out.contains("__builtin_amdgcn_mfma_f32_16x16x16f16"));
    }

    #[test]
    fn test_emit_rope_kernel() {
        let mut out = String::new();
        emit_rope_kernel_hip(&mut out, "rope", 128, 10000.0);
        assert!(out.contains("rope"));
        assert!(out.contains("cosf("));
        assert!(out.contains("sinf("));
    }

    #[test]
    fn test_emit_mha_kernel() {
        let mut out = String::new();
        emit_mha_kernel_hip(&mut out, "mha", 32, 8, 64);
        assert!(out.contains("mha"));
        assert!(out.contains("smem_scores"));
        assert!(out.contains("expf("));
    }

    #[test]
    fn test_emit_dequantize_kernel() {
        let mut out = String::new();
        emit_dequantize_kernel_hip(&mut out, "dequant", 1024, 128, 4);
        assert!(out.contains("dequant"));
        assert!(out.contains("MASK"));
        assert!(out.contains("zero_point"));
    }

    #[test]
    fn test_hip_backend() {
        let backend = HipBackend::new(908);
        assert!(matches!(backend.platform(), Platform::Hip { gfx_arch: 908 }));
        assert_eq!(backend.num_simd_regs(), 256);
        let profile = DeviceProfile::detect();
        let emitter = backend.new_emitter(&profile);
        assert_eq!(emitter.gfx_arch(), 908);
        assert_eq!(emitter.simd_width(), 64); // CDNA wave64
    }

    #[test]
    fn test_hip_backend_rdna() {
        let backend = HipBackend::new(1100);
        assert!(matches!(backend.platform(), Platform::Hip { gfx_arch: 1100 }));
        let profile = DeviceProfile::detect();
        let emitter = backend.new_emitter(&profile);
        assert_eq!(emitter.simd_width(), 32); // RDNA wave32
    }
}
