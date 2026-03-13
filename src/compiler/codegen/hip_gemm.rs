//! HIP GEMM kernel emitters — extracted from `hip.rs`.
//!
//! Contains tiled shared-memory GEMM and gfx908+ MFMA-based GEMM.

use std::fmt::Write;

/// Emit a tiled GEMM kernel using shared memory.
///
/// C = alpha * A * B + beta * C
/// Uses a TILE_SIZE x TILE_SIZE blocking strategy.
pub(crate) fn emit_gemm_kernel_hip(
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
pub(crate) fn emit_gemm_mfma_kernel_hip(out: &mut String, name: &str) {
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
