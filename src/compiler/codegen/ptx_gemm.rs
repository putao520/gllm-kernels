//! PTX GEMM kernel emitters — extracted from `ptx.rs`.
//!
//! Contains tiled GEMM (scalar), Tensor Core sm_70 (wmma), sm_80 (mma.sync),
//! and sm_89 (FP8 mma.sync) variants.

use std::fmt::Write;

pub(crate) fn emit_gemm_kernel_ptx(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
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



/// Emit sm_80+ Tensor Core GEMM using mma.sync.aligned.m16n8k16.f32.f16.f16.f32.
///
/// Uses cp.async for global→shared loads, then shared→register packing for
/// mma operands. Each warp computes a 16×8 output tile per mma instruction.
/// Grid: (ceil(N/16), ceil(M/16)), Block: (32, 1) — one warp per block.
pub(crate) fn emit_gemm_tc_sm80_ptx(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
    let _ = (m, n); // dimensions used for grid launch, not baked into kernel
    let ktiles = (k + 15) / 16;

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_A,").unwrap();
    writeln!(out, "    .param .u64 param_B,").unwrap();
    writeln!(out, "    .param .u64 param_C,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_N,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ") {{").unwrap();

    // Shared memory for A tile (16×16 f16) and B tile (16×8 f16)
    writeln!(out, "    .shared .align 16 .b8 smA[512];").unwrap();  // 16*16*2 bytes
    writeln!(out, "    .shared .align 16 .b8 smB[256];").unwrap();  // 16*8*2 bytes

    // Registers: mma needs 4×f32 accumulators, 4×b32 for A, 2×b32 for B
    writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
    writeln!(out, "    .reg .u32 %r<32>;").unwrap();
    writeln!(out, "    .reg .f32 %acc<4>;").unwrap();
    writeln!(out, "    .reg .b32 %fragA<4>;").unwrap();
    writeln!(out, "    .reg .b32 %fragB<2>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_A];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_B];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_C];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_K];").unwrap();
    writeln!(out).unwrap();

    // Block indices → tile row/col
    writeln!(out, "    mov.u32 %r3, %ctaid.y;").unwrap();  // tile row
    writeln!(out, "    mov.u32 %r4, %ctaid.x;").unwrap();  // tile col
    writeln!(out, "    mov.u32 %r5, %tid.x;").unwrap();    // lane id (0..31)
    writeln!(out).unwrap();

    // Zero accumulators
    writeln!(out, "    mov.f32 %acc0, 0f00000000;").unwrap();
    writeln!(out, "    mov.f32 %acc1, 0f00000000;").unwrap();
    writeln!(out, "    mov.f32 %acc2, 0f00000000;").unwrap();
    writeln!(out, "    mov.f32 %acc3, 0f00000000;").unwrap();
    writeln!(out).unwrap();

    // K-tile loop
    writeln!(out, "    mov.u32 %r6, 0;").unwrap();  // k_iter
    writeln!(out, "LOOP_K_{kernel_name}:").unwrap();

    // cp.async: global → shared for A tile (16×16 f16)
    writeln!(out, "    // -- cp.async load A tile --").unwrap();
    writeln!(out, "    shl.b32 %r7, %r3, 4;").unwrap();         // tile_row * 16
    writeln!(out, "    add.u32 %r8, %r7, %r5;").unwrap();       // row offset by lane
    writeln!(out, "    shl.b32 %r9, %r6, 4;").unwrap();         // k_iter * 16
    writeln!(out, "    mad.lo.u32 %r10, %r8, %r2, %r9;").unwrap(); // row*K + k_off
    writeln!(out, "    shl.b32 %r10, %r10, 1;").unwrap();       // byte offset (f16)
    writeln!(out, "    cvt.u64.u32 %rd3, %r10;").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd0, %rd3;").unwrap();    // &A[row][k_off]
    writeln!(out, "    shl.b32 %r11, %r5, 5;").unwrap();        // lane*32 shared offset
    writeln!(out, "    mov.u32 %r12, smA;").unwrap();
    writeln!(out, "    add.u32 %r12, %r12, %r11;").unwrap();
    writeln!(out, "    cp.async.cg.shared.global [%r12], [%rd4], 16;").unwrap();
    writeln!(out).unwrap();

    // cp.async: global → shared for B tile (16×8 f16)
    writeln!(out, "    // -- cp.async load B tile --").unwrap();
    writeln!(out, "    shl.b32 %r13, %r4, 3;").unwrap();        // tile_col * 8
    writeln!(out, "    mad.lo.u32 %r14, %r9, %r1, %r13;").unwrap(); // k_off*N + col_off (B is K×N)
    writeln!(out, "    add.u32 %r14, %r14, %r5;").unwrap();     // + lane
    writeln!(out, "    shl.b32 %r14, %r14, 1;").unwrap();       // byte offset
    writeln!(out, "    cvt.u64.u32 %rd5, %r14;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd1, %rd5;").unwrap();
    writeln!(out, "    shl.b32 %r15, %r5, 4;").unwrap();        // lane*16 shared offset
    writeln!(out, "    mov.u32 %r16, smB;").unwrap();
    writeln!(out, "    add.u32 %r16, %r16, %r15;").unwrap();
    writeln!(out, "    cp.async.cg.shared.global [%r16], [%rd6], 16;").unwrap();
    writeln!(out).unwrap();

    // Commit and wait
    writeln!(out, "    cp.async.commit_group;").unwrap();
    writeln!(out, "    cp.async.wait_group 0;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out).unwrap();

    // Load fragments from shared memory
    writeln!(out, "    // -- load mma fragments from shared --").unwrap();
    writeln!(out, "    mov.u32 %r17, smA;").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA0, [%r17];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA1, [%r17+4];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA2, [%r17+8];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA3, [%r17+12];").unwrap();
    writeln!(out, "    mov.u32 %r18, smB;").unwrap();
    writeln!(out, "    ld.shared.b32 %fragB0, [%r18];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragB1, [%r18+4];").unwrap();
    writeln!(out).unwrap();

    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    writeln!(out, "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").unwrap();
    writeln!(out, "        {{%acc0, %acc1, %acc2, %acc3}},").unwrap();
    writeln!(out, "        {{%fragA0, %fragA1, %fragA2, %fragA3}},").unwrap();
    writeln!(out, "        {{%fragB0, %fragB1}},").unwrap();
    writeln!(out, "        {{%acc0, %acc1, %acc2, %acc3}};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    bar.sync 0;").unwrap();

    // Loop increment
    writeln!(out, "    add.u32 %r6, %r6, 1;").unwrap();
    writeln!(out, "    setp.lt.u32 %p0, %r6, {ktiles};").unwrap();
    writeln!(out, "    @%p0 bra LOOP_K_{kernel_name};").unwrap();
    writeln!(out).unwrap();

    // Store accumulators to C
    writeln!(out, "    // -- store C tile --").unwrap();
    writeln!(out, "    shl.b32 %r19, %r3, 4;").unwrap();        // tile_row*16
    writeln!(out, "    add.u32 %r20, %r19, %r5;").unwrap();     // + lane
    writeln!(out, "    shl.b32 %r21, %r4, 3;").unwrap();        // tile_col*8
    writeln!(out, "    mad.lo.u32 %r22, %r20, %r1, %r21;").unwrap(); // row*N + col
    writeln!(out, "    shl.b32 %r22, %r22, 2;").unwrap();       // byte offset (f32)
    writeln!(out, "    cvt.u64.u32 %rd7, %r22;").unwrap();
    writeln!(out, "    add.u64 %rd8, %rd2, %rd7;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8], %acc0;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8+4], %acc1;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8+8], %acc2;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8+12], %acc3;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
}

/// Emit sm_70 Tensor Core GEMM using wmma.load/wmma.mma/wmma.store (16×16×16).
///
/// Volta-era wmma path: simpler than mma.sync but still uses Tensor Cores.
/// Grid: (ceil(N/16), ceil(M/16)), Block: (32, 1) — one warp per block.
pub(crate) fn emit_gemm_tc_sm70_ptx(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
    let _ = (m, n);
    let ktiles = (k + 15) / 16;

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_A,").unwrap();
    writeln!(out, "    .param .u64 param_B,").unwrap();
    writeln!(out, "    .param .u64 param_C,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_N,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ") {{").unwrap();

    // wmma fragment registers: a(8 .b32), b(8 .b32), c(8 .f32), d(8 .f32)
    writeln!(out, "    .reg .u64 %rd<12>;").unwrap();
    writeln!(out, "    .reg .u32 %r<24>;").unwrap();
    writeln!(out, "    .reg .b32 %fragA<8>;").unwrap();
    writeln!(out, "    .reg .b32 %fragB<8>;").unwrap();
    writeln!(out, "    .reg .f32 %fragC<8>;").unwrap();
    writeln!(out, "    .reg .f32 %fragD<8>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_A];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_B];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_C];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_K];").unwrap();
    writeln!(out).unwrap();

    // Block indices
    writeln!(out, "    mov.u32 %r3, %ctaid.y;").unwrap();
    writeln!(out, "    mov.u32 %r4, %ctaid.x;").unwrap();
    writeln!(out).unwrap();

    // Zero accumulator fragments
    for i in 0..8 {
        writeln!(out, "    mov.f32 %fragC{i}, 0f00000000;").unwrap();
    }
    writeln!(out).unwrap();

    // Compute base pointers for this tile
    writeln!(out, "    shl.b32 %r5, %r3, 4;").unwrap();   // tile_row * 16
    writeln!(out, "    shl.b32 %r6, %r4, 4;").unwrap();   // tile_col * 16
    writeln!(out).unwrap();

    // K-tile loop
    writeln!(out, "    mov.u32 %r7, 0;").unwrap();
    writeln!(out, "LOOP_K_{kernel_name}:").unwrap();
    writeln!(out, "    shl.b32 %r8, %r7, 4;").unwrap();   // k_iter * 16
    writeln!(out).unwrap();

    // Compute A tile address: A + (tile_row*16)*K + k_iter*16, stride=K
    writeln!(out, "    mad.lo.u32 %r9, %r5, %r2, %r8;").unwrap();
    writeln!(out, "    shl.b32 %r9, %r9, 1;").unwrap();   // f16 byte offset
    writeln!(out, "    cvt.u64.u32 %rd3, %r9;").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd0, %rd3;").unwrap();
    writeln!(out, "    cvt.u64.u32 %rd5, %r2;").unwrap();  // stride = K
    writeln!(out, "    shl.b64 %rd5, %rd5, 1;").unwrap();  // stride in bytes
    writeln!(out).unwrap();

    // wmma.load.a.sync.aligned.row.m16n16k16.f16
    writeln!(out, "    wmma.load.a.sync.aligned.row.m16n16k16.f16").unwrap();
    writeln!(out, "        {{%fragA0, %fragA1, %fragA2, %fragA3, %fragA4, %fragA5, %fragA6, %fragA7}},").unwrap();
    writeln!(out, "        [%rd4], %rd5;").unwrap();
    writeln!(out).unwrap();

    // Compute B tile address: B + (k_iter*16)*N + tile_col*16, stride=N
    writeln!(out, "    mad.lo.u32 %r10, %r8, %r1, %r6;").unwrap();
    writeln!(out, "    shl.b32 %r10, %r10, 1;").unwrap();
    writeln!(out, "    cvt.u64.u32 %rd6, %r10;").unwrap();
    writeln!(out, "    add.u64 %rd7, %rd1, %rd6;").unwrap();
    writeln!(out, "    cvt.u64.u32 %rd8, %r1;").unwrap();  // stride = N
    writeln!(out, "    shl.b64 %rd8, %rd8, 1;").unwrap();
    writeln!(out).unwrap();

    // wmma.load.b.sync.aligned.row.m16n16k16.f16
    writeln!(out, "    wmma.load.b.sync.aligned.row.m16n16k16.f16").unwrap();
    writeln!(out, "        {{%fragB0, %fragB1, %fragB2, %fragB3, %fragB4, %fragB5, %fragB6, %fragB7}},").unwrap();
    writeln!(out, "        [%rd7], %rd8;").unwrap();
    writeln!(out).unwrap();

    // wmma.mma.sync.aligned.row.row.m16n16k16.f32.f16.f16.f32
    writeln!(out, "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f16.f16.f32").unwrap();
    writeln!(out, "        {{%fragD0, %fragD1, %fragD2, %fragD3, %fragD4, %fragD5, %fragD6, %fragD7}},").unwrap();
    writeln!(out, "        {{%fragA0, %fragA1, %fragA2, %fragA3, %fragA4, %fragA5, %fragA6, %fragA7}},").unwrap();
    writeln!(out, "        {{%fragB0, %fragB1, %fragB2, %fragB3, %fragB4, %fragB5, %fragB6, %fragB7}},").unwrap();
    writeln!(out, "        {{%fragC0, %fragC1, %fragC2, %fragC3, %fragC4, %fragC5, %fragC6, %fragC7}};").unwrap();
    writeln!(out).unwrap();

    // Copy D → C for next accumulation
    for i in 0..8 {
        writeln!(out, "    mov.f32 %fragC{i}, %fragD{i};").unwrap();
    }
    writeln!(out).unwrap();

    // Loop
    writeln!(out, "    add.u32 %r7, %r7, 1;").unwrap();
    writeln!(out, "    setp.lt.u32 %p0, %r7, {ktiles};").unwrap();
    writeln!(out, "    @%p0 bra LOOP_K_{kernel_name};").unwrap();
    writeln!(out).unwrap();

    // Store C tile: C + (tile_row*16)*N + tile_col*16, stride=N
    writeln!(out, "    mad.lo.u32 %r11, %r5, %r1, %r6;").unwrap();
    writeln!(out, "    shl.b32 %r11, %r11, 2;").unwrap();  // f32 byte offset
    writeln!(out, "    cvt.u64.u32 %rd9, %r11;").unwrap();
    writeln!(out, "    add.u64 %rd10, %rd2, %rd9;").unwrap();
    writeln!(out, "    cvt.u64.u32 %rd11, %r1;").unwrap();
    writeln!(out, "    shl.b64 %rd11, %rd11, 2;").unwrap(); // stride in bytes (f32)
    writeln!(out).unwrap();

    // wmma.store.d.sync.aligned.row.m16n16k16.f32
    writeln!(out, "    wmma.store.d.sync.aligned.row.m16n16k16.f32").unwrap();
    writeln!(out, "        [%rd10], {{%fragC0, %fragC1, %fragC2, %fragC3, %fragC4, %fragC5, %fragC6, %fragC7}},").unwrap();
    writeln!(out, "        %rd11;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
}


/// Emit sm_89 Tensor Core GEMM using FP8 mma.sync variant (e4m3/e5m2).
///
/// Ada Lovelace (sm_89) supports FP8 Tensor Core via mma.sync.aligned.m16n8k32.
/// A operands are .e4m3, B operands are .e4m3, accumulators are .f32.
/// Grid: (ceil(N/16), ceil(M/16)), Block: (32, 1) — one warp per block.
pub(crate) fn emit_gemm_tc_sm89_ptx(out: &mut String, kernel_name: &str, m: usize, n: usize, k: usize) {
    let _ = (m, n);
    let ktiles = (k + 31) / 32; // FP8 k-tile is 32

    writeln!(out, ".visible .entry {kernel_name}(").unwrap();
    writeln!(out, "    .param .u64 param_A,").unwrap();
    writeln!(out, "    .param .u64 param_B,").unwrap();
    writeln!(out, "    .param .u64 param_C,").unwrap();
    writeln!(out, "    .param .u32 param_M,").unwrap();
    writeln!(out, "    .param .u32 param_N,").unwrap();
    writeln!(out, "    .param .u32 param_K").unwrap();
    writeln!(out, ") {{").unwrap();

    // Shared memory for A tile (16×32 FP8 = 512 bytes) and B tile (32×8 FP8 = 256 bytes)
    writeln!(out, "    .shared .align 16 .b8 smA[512];").unwrap();
    writeln!(out, "    .shared .align 16 .b8 smB[256];").unwrap();

    // Registers: mma needs 4×f32 accumulators, 4×b32 for A (FP8 packed), 2×b32 for B
    writeln!(out, "    .reg .u64 %rd<16>;").unwrap();
    writeln!(out, "    .reg .u32 %r<32>;").unwrap();
    writeln!(out, "    .reg .f32 %acc<4>;").unwrap();
    writeln!(out, "    .reg .b32 %fragA<4>;").unwrap();
    writeln!(out, "    .reg .b32 %fragB<2>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out).unwrap();

    // Load params
    writeln!(out, "    ld.param.u64 %rd0, [param_A];").unwrap();
    writeln!(out, "    ld.param.u64 %rd1, [param_B];").unwrap();
    writeln!(out, "    ld.param.u64 %rd2, [param_C];").unwrap();
    writeln!(out, "    ld.param.u32 %r0, [param_M];").unwrap();
    writeln!(out, "    ld.param.u32 %r1, [param_N];").unwrap();
    writeln!(out, "    ld.param.u32 %r2, [param_K];").unwrap();
    writeln!(out).unwrap();

    // Block indices → tile row/col
    writeln!(out, "    mov.u32 %r3, %ctaid.y;").unwrap();  // tile row
    writeln!(out, "    mov.u32 %r4, %ctaid.x;").unwrap();  // tile col
    writeln!(out, "    mov.u32 %r5, %tid.x;").unwrap();    // lane id (0..31)
    writeln!(out).unwrap();

    // Zero accumulators
    writeln!(out, "    mov.f32 %acc0, 0f00000000;").unwrap();
    writeln!(out, "    mov.f32 %acc1, 0f00000000;").unwrap();
    writeln!(out, "    mov.f32 %acc2, 0f00000000;").unwrap();
    writeln!(out, "    mov.f32 %acc3, 0f00000000;").unwrap();
    writeln!(out).unwrap();

    // K-tile loop (k-tile = 32 for FP8)
    writeln!(out, "    mov.u32 %r6, 0;").unwrap();
    writeln!(out, "LOOP_K_{kernel_name}:").unwrap();

    // cp.async: global → shared for A tile (16×32 FP8 = 512 bytes)
    writeln!(out, "    // -- cp.async load A tile (FP8) --").unwrap();
    writeln!(out, "    shl.b32 %r7, %r3, 4;").unwrap();         // tile_row * 16
    writeln!(out, "    add.u32 %r8, %r7, %r5;").unwrap();       // row offset by lane
    writeln!(out, "    shl.b32 %r9, %r6, 5;").unwrap();         // k_iter * 32
    writeln!(out, "    mad.lo.u32 %r10, %r8, %r2, %r9;").unwrap(); // row*K + k_off
    // FP8: 1 byte per element, no shift needed
    writeln!(out, "    cvt.u64.u32 %rd3, %r10;").unwrap();
    writeln!(out, "    add.u64 %rd4, %rd0, %rd3;").unwrap();
    writeln!(out, "    shl.b32 %r11, %r5, 5;").unwrap();        // lane*32 shared offset
    writeln!(out, "    mov.u32 %r12, smA;").unwrap();
    writeln!(out, "    add.u32 %r12, %r12, %r11;").unwrap();
    writeln!(out, "    cp.async.cg.shared.global [%r12], [%rd4], 16;").unwrap();
    writeln!(out).unwrap();

    // cp.async: global → shared for B tile (32×8 FP8 = 256 bytes)
    writeln!(out, "    // -- cp.async load B tile (FP8) --").unwrap();
    writeln!(out, "    shl.b32 %r13, %r4, 3;").unwrap();        // tile_col * 8
    writeln!(out, "    mad.lo.u32 %r14, %r9, %r1, %r13;").unwrap();
    writeln!(out, "    add.u32 %r14, %r14, %r5;").unwrap();
    // FP8: 1 byte per element
    writeln!(out, "    cvt.u64.u32 %rd5, %r14;").unwrap();
    writeln!(out, "    add.u64 %rd6, %rd1, %rd5;").unwrap();
    writeln!(out, "    shl.b32 %r15, %r5, 4;").unwrap();
    writeln!(out, "    mov.u32 %r16, smB;").unwrap();
    writeln!(out, "    add.u32 %r16, %r16, %r15;").unwrap();
    writeln!(out, "    cp.async.cg.shared.global [%r16], [%rd6], 16;").unwrap();
    writeln!(out).unwrap();

    // Commit and wait
    writeln!(out, "    cp.async.commit_group;").unwrap();
    writeln!(out, "    cp.async.wait_group 0;").unwrap();
    writeln!(out, "    bar.sync 0;").unwrap();
    writeln!(out).unwrap();

    // Load fragments from shared memory
    writeln!(out, "    // -- load mma fragments from shared (FP8 packed into b32) --").unwrap();
    writeln!(out, "    mov.u32 %r17, smA;").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA0, [%r17];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA1, [%r17+4];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA2, [%r17+8];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragA3, [%r17+12];").unwrap();
    writeln!(out, "    mov.u32 %r18, smB;").unwrap();
    writeln!(out, "    ld.shared.b32 %fragB0, [%r18];").unwrap();
    writeln!(out, "    ld.shared.b32 %fragB1, [%r18+4];").unwrap();
    writeln!(out).unwrap();

    // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    writeln!(out, "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32").unwrap();
    writeln!(out, "        {{%acc0, %acc1, %acc2, %acc3}},").unwrap();
    writeln!(out, "        {{%fragA0, %fragA1, %fragA2, %fragA3}},").unwrap();
    writeln!(out, "        {{%fragB0, %fragB1}},").unwrap();
    writeln!(out, "        {{%acc0, %acc1, %acc2, %acc3}};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    bar.sync 0;").unwrap();

    // Loop increment
    writeln!(out, "    add.u32 %r6, %r6, 1;").unwrap();
    writeln!(out, "    setp.lt.u32 %p0, %r6, {ktiles};").unwrap();
    writeln!(out, "    @%p0 bra LOOP_K_{kernel_name};").unwrap();
    writeln!(out).unwrap();

    // Store accumulators to C (f32)
    writeln!(out, "    // -- store C tile --").unwrap();
    writeln!(out, "    shl.b32 %r19, %r3, 4;").unwrap();
    writeln!(out, "    add.u32 %r20, %r19, %r5;").unwrap();
    writeln!(out, "    shl.b32 %r21, %r4, 3;").unwrap();
    writeln!(out, "    mad.lo.u32 %r22, %r20, %r1, %r21;").unwrap();
    writeln!(out, "    shl.b32 %r22, %r22, 2;").unwrap();
    writeln!(out, "    cvt.u64.u32 %rd7, %r22;").unwrap();
    writeln!(out, "    add.u64 %rd8, %rd2, %rd7;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8], %acc0;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8+4], %acc1;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8+8], %acc2;").unwrap();
    writeln!(out, "    st.global.f32 [%rd8+12], %acc3;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    ret;").unwrap();
    writeln!(out, "}}").unwrap();
}
