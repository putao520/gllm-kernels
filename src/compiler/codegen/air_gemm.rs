//! MSL GEMM kernel emitters — extracted from `air.rs`.
//!
//! Contains tiled shared-memory GEMM and Apple simdgroup_matrix GEMM variants,
//! with and without bias.

use std::fmt::Write;

pub(crate) fn emit_gemm_kernel_msl(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
    let tile = 16usize;
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* A [[buffer(0)]],").unwrap();
    writeln!(out, "    device const float* B [[buffer(1)]],").unwrap();
    writeln!(out, "    device float* C [[buffer(2)]],").unwrap();
    writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint2 tid [[thread_position_in_threadgroup]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const uint M = {m}u, N = {n}u, K = {k}u;").unwrap();
    writeln!(out, "    const uint TILE = {tile}u;").unwrap();
    writeln!(out, "    threadgroup float smA[{t2}];", t2 = tile * tile).unwrap();
    writeln!(out, "    threadgroup float smB[{t2}];", t2 = tile * tile).unwrap();
    writeln!(out, "    uint row = gid.y * TILE + tid.y;").unwrap();
    writeln!(out, "    uint col = gid.x * TILE + tid.x;").unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {{").unwrap();
    writeln!(out, "        uint aCol = t * TILE + tid.x;").unwrap();
    writeln!(out, "        uint bRow = t * TILE + tid.y;").unwrap();
    writeln!(out, "        smA[tid.y * TILE + tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;").unwrap();
    writeln!(out, "        smB[tid.y * TILE + tid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "        for (uint i = 0; i < TILE; i++) {{").unwrap();
    writeln!(out, "            acc += smA[tid.y * TILE + i] * smB[i * TILE + tid.x];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    if (row < M && col < N) {{").unwrap();
    writeln!(out, "        C[row * N + col] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

/// Emit Apple GPU family 7+ GEMM using `simdgroup_matrix` (8×8 tiles).
///
/// Uses `simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`
/// for hardware-accelerated matrix multiply on Apple Silicon (A14+, M1+).
/// Grid: (ceil(N/8), ceil(M/8)), Block: (32, 1) — one simdgroup per threadgroup.
pub(crate) fn emit_gemm_simdgroup_msl(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
    let ktiles = (k + 7) / 8;
    writeln!(out, "#include <metal_simdgroup_matrix>").unwrap();
    writeln!(out, "using namespace metal;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const half* A [[buffer(0)]],").unwrap();
    writeln!(out, "    device const half* B [[buffer(1)]],").unwrap();
    writeln!(out, "    device float* C [[buffer(2)]],").unwrap();
    writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint simd_lane [[thread_index_in_simdgroup]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const uint M = {m}u, N = {n}u, K = {k}u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    simdgroup_float8x8 acc;").unwrap();
    writeln!(out, "    simdgroup_half8x8 fragA, fragB;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Zero accumulator").unwrap();
    writeln!(out, "    acc = simdgroup_float8x8(0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint tile_row = gid.y * 8;").unwrap();
    writeln!(out, "    uint tile_col = gid.x * 8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint kt = 0; kt < {ktiles}u; ++kt) {{").unwrap();
    writeln!(out, "        uint k_off = kt * 8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // Load A tile: A[tile_row..+8][k_off..+8], stride=K").unwrap();
    writeln!(out, "        simdgroup_load(fragA, A + tile_row * K + k_off, K);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // Load B tile: B[k_off..+8][tile_col..+8], stride=N").unwrap();
    writeln!(out, "        simdgroup_load(fragB, B + k_off * N + tile_col, N);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // Multiply-accumulate").unwrap();
    writeln!(out, "        simdgroup_multiply_accumulate(acc, fragA, fragB, acc);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Store result: C[tile_row..+8][tile_col..+8], stride=N").unwrap();
    writeln!(out, "    simdgroup_store(acc, C + tile_row * N + tile_col, N);").unwrap();
    writeln!(out, "}}\n").unwrap();
}

pub(crate) fn emit_gemm_bias_simdgroup_msl(
    out: &mut String,
    kernel_name: &str,
    m: usize,
    n: usize,
    k: usize,
) {
    let ktiles = (k + 7) / 8;
    writeln!(out, "#include <metal_simdgroup_matrix>").unwrap();
    writeln!(out, "using namespace metal;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const half* A [[buffer(0)]],").unwrap();
    writeln!(out, "    device const half* B [[buffer(1)]],").unwrap();
    writeln!(out, "    device const float* bias [[buffer(2)]],").unwrap();
    writeln!(out, "    device float* C [[buffer(3)]],").unwrap();
    writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint simd_lane [[thread_index_in_simdgroup]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const uint M = {m}u, N = {n}u, K = {k}u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    simdgroup_float8x8 acc;").unwrap();
    writeln!(out, "    simdgroup_half8x8 fragA, fragB;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    acc = simdgroup_float8x8(0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint tile_row = gid.y * 8;").unwrap();
    writeln!(out, "    uint tile_col = gid.x * 8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint kt = 0; kt < {ktiles}u; ++kt) {{").unwrap();
    writeln!(out, "        uint k_off = kt * 8;").unwrap();
    writeln!(out, "        simdgroup_load(fragA, A + tile_row * K + k_off, K);").unwrap();
    writeln!(out, "        simdgroup_load(fragB, B + k_off * N + tile_col, N);").unwrap();
    writeln!(out, "        simdgroup_multiply_accumulate(acc, fragA, fragB, acc);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Store to threadgroup, add bias, write to C").unwrap();
    writeln!(out, "    threadgroup float tile[64];").unwrap();
    writeln!(out, "    simdgroup_store(acc, tile, 8);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // First 32 lanes cover rows 0..3 of the 8x8 tile").unwrap();
    writeln!(out, "    uint lane_row = simd_lane / 8;").unwrap();
    writeln!(out, "    uint lane_col = simd_lane % 8;").unwrap();
    writeln!(out, "    uint out_row = tile_row + lane_row;").unwrap();
    writeln!(out, "    uint out_col = tile_col + lane_col;").unwrap();
    writeln!(out, "    if (out_row < M && out_col < N) {{").unwrap();
    writeln!(out, "        C[out_row * N + out_col] = tile[lane_row * 8 + lane_col] + bias[out_col];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    // Remaining rows 4..7").unwrap();
    writeln!(out, "    uint lane_row2 = lane_row + 4;").unwrap();
    writeln!(out, "    uint out_row2 = tile_row + lane_row2;").unwrap();
    writeln!(out, "    if (out_row2 < M && out_col < N) {{").unwrap();
    writeln!(out, "        C[out_row2 * N + out_col] = tile[lane_row2 * 8 + lane_col] + bias[out_col];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

pub(crate) fn emit_gemm_bias_kernel_msl(
    out: &mut String,
    kernel_name: &str,
    m: usize,
    n: usize,
    k: usize,
) {
    let tile = 16usize;
    writeln!(out, "kernel void {kernel_name}(").unwrap();
    writeln!(out, "    device const float* A [[buffer(0)]],").unwrap();
    writeln!(out, "    device const float* B [[buffer(1)]],").unwrap();
    writeln!(out, "    device const float* bias [[buffer(2)]],").unwrap();
    writeln!(out, "    device float* C [[buffer(3)]],").unwrap();
    writeln!(out, "    uint2 gid [[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint2 tid [[thread_position_in_threadgroup]]").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "    const uint M = {m}u, N = {n}u, K = {k}u;").unwrap();
    writeln!(out, "    const uint TILE = {tile}u;").unwrap();
    writeln!(out, "    threadgroup float smA[{t2}];", t2 = tile * tile).unwrap();
    writeln!(out, "    threadgroup float smB[{t2}];", t2 = tile * tile).unwrap();
    writeln!(out, "    uint row = gid.y * TILE + tid.y;").unwrap();
    writeln!(out, "    uint col = gid.x * TILE + tid.x;").unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {{").unwrap();
    writeln!(out, "        uint aCol = t * TILE + tid.x;").unwrap();
    writeln!(out, "        uint bRow = t * TILE + tid.y;").unwrap();
    writeln!(out, "        smA[tid.y * TILE + tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;").unwrap();
    writeln!(out, "        smB[tid.y * TILE + tid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "        for (uint i = 0; i < TILE; i++) {{").unwrap();
    writeln!(out, "            acc += smA[tid.y * TILE + i] * smB[i * TILE + tid.x];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    if (row < M && col < N) {{").unwrap();
    writeln!(out, "        C[row * N + col] = acc + bias[col];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}
