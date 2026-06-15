//! GEMM inline lowering — naive, BLIS tiled, GPU tiled/pipelined, trans-B, epilogue fusion.

use super::instr::*;
use super::lower;
use super::plan_lower::LoweringContext;
use super::plan_lower::SymDimSlotMap;
use super::telemetry_emit::emit_gemm_row_stats_telemetry;
use crate::compiler::trace::{QuantPrecision, TraceOp, ValueId};
use crate::compiler::graph::SymDim;
use crate::compiler::pain_point::ExecPattern;
use crate::types::CompilerError;

pub(crate) fn emit_tile_gemm(
    prog: &mut VmProgram, width: SimdWidth,
    rows: usize, cols: usize, kd: usize, k: usize, dt: crate::types::DType,
) -> Result<(), CompilerError> {
    let tile_c = prog.alloc_vreg(VRegKind::Tile, width);
    let tile_a = prog.alloc_vreg(VRegKind::Tile, width);
    let tile_b = prog.alloc_vreg(VRegKind::Tile, width);
    prog.emit(VmInstr::TileConfig { rows, cols, dtype: dt });
    let k_tiles = (k + kd - 1) / kd;
    prog.emit_loop(BoundExpr::Const(k_tiles), kd * dt.size_bytes(), |prog, _ctr, _off| {
        prog.emit(VmInstr::TileMma { c: tile_c, a: tile_a, b: tile_b });
    });
    prog.emit(VmInstr::TileRelease);
    Ok(())
}

pub(crate) fn emit_gemm_inline_with_hook<'a>(
    prog: &mut VmProgram,
    m_dim: &SymDim, n: usize, k: usize,
    ctx: &LoweringContext<'a>,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    seq_bound_override: Option<&BoundExpr>,
    gemm_op_id: Option<crate::compiler::graph::OpId>,
    pack_map: Option<&'a crate::compiler::pack_map::PackMap>,
    trans_b: bool,
) -> Result<(), CompilerError> {
    use super::isa_hook::FmaStrategy;
    let width = ctx.session.width;
    let sym_map = &ctx.session.sym_map;
    let budget = ctx.session.budget.as_ref();
    // §0.2.10: per-op ParallelismDesc — decode (M=1) unroll=1, prefill unroll=k_unroll
    let k_unroll = gemm_op_id
        .and_then(|id| ctx.parallelism_for_op(id))
        .map(|pd| match pd {
            super::super::super::pain_point::ParallelismDesc::SimdVectorize { unroll_factor, .. } => unroll_factor,
            _ => 1,
        })
        .unwrap_or_else(|| {
            ctx.parallelism
                .as_ref()
                .map(|pd| match pd {
                    super::super::super::pain_point::ParallelismDesc::SimdVectorize { unroll_factor, .. } => *unroll_factor,
                    _ => 1,
                })
                .unwrap_or(1)
        });
    let hook = ctx.session.hook.ok_or_else(|| CompilerError::CodegenViolation(
        "emit_gemm_inline_with_hook: IsaHook is mandatory (ARCH-ISA-HOOK-MANDATORY)".into(),
    ))?;
    let m = match m_dim {
        SymDim::Concrete(v) => *v,
        SymDim::Symbolic { max_value, .. } => max_value
            .expect("ARCH-SYMDIM: GEMM M Symbolic must have max_value"),
    };

    // §0.2.9 ExecPattern 感知策略选择:
    // 从 R0 PainPointAnalyzer 的瓶颈映射中查找当前 op 的执行模式。
    let effective_exec_pattern = gemm_op_id
        .and_then(|id| ctx.exec_pattern_for_op(id))
        .or(ctx.exec_pattern);

    match &effective_exec_pattern {
        Some(ExecPattern::ScalarLoop) => {
            return emit_gemm_inline_with_epilogue(prog, m_dim, n, k, width, a_ptr, b_ptr, c_ptr, &[], sym_map, false, seq_bound_override, ctx.dtype, trans_b, super::isa_hook::EpiloguePlace::OnAccumulators);
        }
        Some(ExecPattern::SharedMemTile { .. }) => {
            let (mr, nr) = hook.gemm_microkernel_shape();
            let lanes = width.f32_lanes().max(1);
            let can_blis = !trans_b && !m_dim.is_symbolic() && m >= mr && n >= nr * lanes && k >= 16;
            if can_blis {
                return emit_gemm_blis_inline(prog, m, n, k, width, a_ptr, b_ptr, c_ptr, mr, nr, pack_map, k_unroll, ctx.dtype, trans_b);
            }
        }
        Some(ExecPattern::TileGemm { tile_m, tile_n, tile_k, warp_m, warp_n, mma_k, pipeline_depth }) => {
            // §0.2.9: ExecPattern::TileGemm 指导微核形状 — 优先使用 ExecPattern 的 tile 参数
            let is_gpu = matches!(width, SimdWidth::Warp(_));
            if is_gpu {
                // GPU path: 从 HardwareProfile 获取三级分块 + 流水线参数
                // ExecPattern 由 CPU-centric derive_exec_pattern 生成，warp 参数可能为 0
                // → 使用 HardwareProfile.gpu_gemm_tiles() 覆盖
                use crate::compiler::hardware_profile::HardwareProfile;
                use crate::dispatch::DeviceProfile;
                let dp = DeviceProfile::detect();
                let hw = HardwareProfile::detect(&dp);
                let (hw_cta_m, hw_cta_n, hw_cta_k, hw_warp_m, hw_warp_n, hw_mma_k) = hw.gpu_gemm_tiles();
                let hw_pipe = hw.gpu_pipeline_depth();

                // 如果 ExecPattern 已有 GPU 参数 (warp_m > 0)，优先使用；否则用 HardwareProfile 默认值
                let (cta_m, cta_n, cta_k) = if *warp_m > 0 || *warp_n > 0 {
                    (*tile_m, *tile_n, *tile_k)
                } else if hw_warp_m > 0 {
                    (hw_cta_m, hw_cta_n, hw_cta_k)
                } else {
                    // HardwareProfile 也没有 GPU 参数 → 不是 GPU 设备，fall through 到 CPU
                    (*tile_m, *tile_n, *tile_k)
                };
                let (w_m, w_n, mk) = if *warp_m > 0 { (*warp_m, *warp_n, *mma_k) } else { (hw_warp_m, hw_warp_n, hw_mma_k) };
                let pipe_depth = if *pipeline_depth > 0 { *pipeline_depth } else { hw_pipe };

                if w_m > 0 || w_n > 0 {
                    if pipe_depth >= 2 {
                        return emit_gemm_gpu_pipelined(prog, m, n, k, width, a_ptr, b_ptr, c_ptr,
                            cta_m, cta_n, cta_k, w_m, w_n, mk, pipe_depth, ctx.dtype, trans_b,
                            hw.has_tma());
                    }
                    return emit_gemm_gpu_tiled_inline(prog, m, n, k, width, a_ptr, b_ptr, c_ptr,
                        cta_m, cta_n, cta_k, w_m, w_n, mk, ctx.dtype, trans_b);
                }
            }
            let (default_mr, default_nr) = hook.gemm_microkernel_shape();
            let mr = *tile_m;
            let nr = *tile_n;
            let lanes = width.f32_lanes().max(1);
            let can_blis = !trans_b && !m_dim.is_symbolic() && m >= mr && n >= nr * lanes && k >= 16;
            if can_blis {
                return emit_gemm_blis_inline(prog, m, n, k, width, a_ptr, b_ptr, c_ptr, mr, nr, pack_map, k_unroll, ctx.dtype, trans_b);
            }
            // TileGemm 参数不兼容 BLIS → fallback 到默认微核形状
            let can_blis_default = !trans_b && !m_dim.is_symbolic() && m >= default_mr && n >= default_nr * lanes && k >= 16;
            if can_blis_default {
                return emit_gemm_blis_inline(prog, m, n, k, width, a_ptr, b_ptr, c_ptr, default_mr, default_nr, pack_map, k_unroll, ctx.dtype, trans_b);
            }
        }
        Some(ExecPattern::AsyncPipeline) => {
            // §0.2.9: AsyncPipeline → BLIS + prefetch hint（减少缓存未命中）
            let (mr, nr) = hook.gemm_microkernel_shape();
            let lanes = width.f32_lanes().max(1);
            let can_blis = !trans_b && !m_dim.is_symbolic() && m >= mr && n >= nr * lanes && k >= 16;
            if can_blis {
                return emit_gemm_blis_inline(prog, m, n, k, width, a_ptr, b_ptr, c_ptr, mr, nr, pack_map, k_unroll, ctx.dtype, trans_b);
            }
        }
        None => {}
    }

    let strategy = match budget {
        Some(b) => super::isa_hook::select_fma_best(hook, m, n, k, ctx.dtype.to_dtype(), b),
        None => hook.select_fma(m, n, k),
    };

    match strategy {
        FmaStrategy::TileMma(ref cfg) => {
            let (rows, cols, kd, dt) = (cfg.rows, cfg.cols, cfg.k_depth, cfg.dtype);
            emit_tile_gemm(prog, width, rows, cols, kd, k, dt)
        }
        FmaStrategy::Wgmma(ref cfg) => {
            let (rows, cols, kd, dt) = (cfg.m, cfg.n, cfg.k, cfg.input_dtype);
            emit_tile_gemm(prog, width, rows, cols, kd, k, dt)
        }
        FmaStrategy::Tcgen05(ref cfg) => {
            let (rows, cols, kd, dt) = (cfg.m, cfg.n, cfg.k, cfg.input_dtype);
            emit_tile_gemm(prog, width, rows, cols, kd, k, dt)
        }
        FmaStrategy::Mfma(ref cfg) => {
            let (rows, cols, kd, dt) = (cfg.m, cfg.n, cfg.k, cfg.input_dtype);
            emit_tile_gemm(prog, width, rows, cols, kd, k, dt)
        }
        FmaStrategy::Fma3 | FmaStrategy::MulAdd => {
            let (mr, nr) = hook.gemm_microkernel_shape();
            let lanes = width.f32_lanes().max(1);
            // trans_b=true: BLIS tiled access assumes contiguous B rows; non-contiguous
            // transposed layout requires element-wise addressing → use naive path.
            let can_blis = !trans_b && !m_dim.is_symbolic() && m >= mr && n >= nr * lanes && k >= 16;
            if can_blis {
                emit_gemm_blis_inline(prog, m, n, k, width, a_ptr, b_ptr, c_ptr, mr, nr, pack_map, k_unroll, ctx.dtype, trans_b)
            } else {
                emit_gemm_inline_with_epilogue(prog, m_dim, n, k, width, a_ptr, b_ptr, c_ptr, &[], sym_map, false, seq_bound_override, ctx.dtype, trans_b, super::isa_hook::EpiloguePlace::OnAccumulators)
            }
        }
    }
}

/// GEMM BLIS 内联 (mr×nr 微内核)。
///
/// 仅在 m 为 Concrete (非 Symbolic) 时调用。M 维度编译时展开。
/// `pack_map`: 预留参数，未来物理重打包 (QuantGather) 实现后用于 B-matrix stride。
/// 当前 B-matrix 始终按 row-major stride (n * elem_bytes) 寻址。
pub(crate) fn emit_gemm_blis_inline(
    prog: &mut VmProgram,
    m: usize, n: usize, k: usize,
    width: SimdWidth,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    mr: usize, nr: usize,
    pack_map: Option<&crate::compiler::pack_map::PackMap>,
    unroll_factor: usize,
    dtype: QuantPrecision,
    trans_b: bool,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let acc_dtype = dtype.accumulator_dtype();
    let needs_narrow = dtype.needs_narrowing_from(acc_dtype);

    // mr×nr 累加器
    let num_acc = mr * nr;
    let accs: Vec<VRegId> = (0..num_acc)
        .map(|_| prog.alloc_vreg(VRegKind::Vec, width))
        .collect();
    let a_broadcast = prog.alloc_vreg(VRegKind::Vec, width);
    let b_vec = prog.alloc_vreg(VRegKind::Vec, width);

    // IR loop: i in 0..m step mr (M is Concrete, compile-time unroll)
    for i_block in (0..m).step_by(mr) {
        let mr_actual = mr.min(m - i_block);
        for j_block in (0..n).step_by(nr * lanes) {
            let nr_actual = nr.min((n - j_block + lanes - 1) / lanes);

            for a in &accs[..mr_actual * nr_actual] {
                prog.emit(VmInstr::Broadcast { dst: *a, src: ScalarExpr::Const(0.0), width, dtype: acc_dtype });
            }

            let k_unroll = unroll_factor.max(1).min(k);
            let k_step = elem * k_unroll;
            let k_iters = (k + k_unroll - 1) / k_unroll;
            // B-matrix row stride in bytes. Row-major: n * elem_bytes.
            // pack_map stride (e.g. PanelPack nr*elem) applies only when B has been
            // physically repacked — which no runtime step currently does.
            // Until a QuantGather-style repack is implemented, always use n * elem.
            let b_row_stride: usize = if trans_b { elem } else { n * elem };
            let b_col_stride: usize = if trans_b { k * elem } else { elem };
            prog.emit_loop(BoundExpr::Const(k_iters), k_step, |prog, k_ctr, k_off| {
                for u in 0..k_unroll {
                    let u_byte_off = u * elem;
                    let b_k_off = u * b_row_stride;
                    for r in 0..mr_actual {
                        let a_off = (i_block + r) * k * dtype.elem_bytes() + u_byte_off;
                        prog.emit(VmInstr::Broadcast {
                            dst: a_broadcast,
                            src: ScalarExpr::MemLoad(a_ptr, OffsetExpr::loop_plus_const(k_off, a_off)),
                            width, dtype,
                        });
                        for c in 0..nr_actual {
                            let b_off = j_block * b_col_stride + c * lanes * b_col_stride;
                            let acc_idx = r * nr_actual + c;
                            if acc_idx < accs.len() {
                                prog.emit(VmInstr::VecLoad {
                                    dst: b_vec, base: b_ptr,
                                    offset: OffsetExpr::Add(
                                        Box::new(OffsetExpr::Add(
                                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(k_ctr)), b_row_stride * k_unroll)),
                                            Box::new(OffsetExpr::Const(b_k_off)),
                                        )),
                                        Box::new(OffsetExpr::Const(b_off)),
                                    ),
                                    width, dtype, predicate: None,
                                });
                                // Direct FMA: dst = acc + a * b (in-place accumulate)
                                prog.emit(VmInstr::Fma {
                                    dst: accs[acc_idx],
                                    acc: accs[acc_idx],
                                    a: a_broadcast,
                                    b: b_vec,
                                    dtype,
                                });
                            }
                        }
                    }
                }
            });

            // Store C (REQ-DTYPE-006: 窄化写回如果 acc_dtype != dtype)
            for r in 0..mr_actual {
                for c in 0..nr_actual {
                    let acc_idx = r * nr_actual + c;
                    if acc_idx < accs.len() {
                        let c_off = (i_block + r) * n * dtype.elem_bytes() + (j_block + c * lanes) * dtype.elem_bytes();
                        let store_src = if needs_narrow {
                            let narrowed = prog.alloc_vreg(VRegKind::Vec, width);
                            prog.emit(VmInstr::VecNarrow { dst: narrowed, src: accs[acc_idx], dst_dtype: dtype, src_dtype: acc_dtype, width });
                            narrowed
                        } else {
                            accs[acc_idx]
                        };
                        prog.emit(VmInstr::VecStore {
                            base: c_ptr, offset: OffsetExpr::Const(c_off), src: store_src, width,
                            dtype, predicate: None,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// GPU 三级分块 GEMM: CTA × Warp × MMA.
/// 生成 VmInstr 序列实现 CTA 级 (cta_m × cta_n × cta_k) 分块，
/// 每个 CTA 内按 warp (warp_m × warp_n) 子分块，
/// 最内层 MMA 指令维度 mma_k.
pub(crate) fn emit_gemm_gpu_tiled_inline(
    prog: &mut VmProgram,
    m: usize, n: usize, k: usize,
    width: SimdWidth,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    cta_m: usize, cta_n: usize, cta_k: usize,
    warp_m: usize, warp_n: usize, mma_k: usize,
    dtype: QuantPrecision,
    _trans_b: bool,
) -> Result<(), CompilerError> {
    let elem = dtype.elem_bytes();
    let acc_dtype = dtype.gpu_accumulator_dtype(); // REQ-DTYPE-005: GPU path
    let mma_k = mma_k.max(16);

    // GPU GEMM 三级分块:
    // Level 1: CTA 循环 — 每个 CTA 处理 cta_m × cta_n × cta_k 的 tile
    // Level 2: Warp 循环 — CTA 内每个 warp 处理 warp_m × warp_n
    // Level 3: MMA 微核 — warp 内 MMA 指令处理 mma_k 深度

    // 累加器: warp_m × warp_n f32 寄存器
    let acc_count = warp_m * warp_n;
    let accs: Vec<VRegId> = (0..acc_count.min(16))
        .map(|_| prog.alloc_vreg(VRegKind::Vec, width))
        .collect();

    // CTA 级循环: 遍历 M/N 维度
    for i_cta in (0..m).step_by(cta_m) {
        let mi = cta_m.min(m - i_cta);
        for j_cta in (0..n).step_by(cta_n) {
            let nj = cta_n.min(n - j_cta);

            // 清零累加器
            for acc in &accs {
                prog.emit(VmInstr::Broadcast { dst: *acc, src: ScalarExpr::Const(0.0), width, dtype: acc_dtype });
            }

            // K 维度循环
            for k_tile in (0..k).step_by(cta_k) {
                let kk = cta_k.min(k - k_tile);

                // Warp 级循环: 在 CTA tile 内分配 warp 工作
                for i_warp in (0..mi).step_by(warp_m) {
                    let wi = warp_m.min(mi - i_warp);
                    for j_warp in (0..nj).step_by(warp_n) {
                        let wj = warp_n.min(nj - j_warp);

                        // MMA 内层 K 循环: 加载 A/B tile → FMA → 累加
                        for k_inner in (0..kk).step_by(mma_k) {
                            // Load A tile: a_ptr + (i_cta + i_warp) * k * elem + (k_tile + k_inner) * elem
                            let a_off = ((i_cta + i_warp) * k + k_tile + k_inner) * elem;
                            // Load B tile: b_ptr + (k_tile + k_inner) * n * elem + (j_cta + j_warp) * elem
                            let b_off = ((k_tile + k_inner) * n + j_cta + j_warp) * elem;

                            for row in 0..wi {
                                let a_row_off = a_off + row * k * elem;
                                let a_vec = prog.alloc_vreg(VRegKind::Vec, width);
                                prog.emit(VmInstr::VecLoad {
                                    dst: a_vec, base: a_ptr,
                                    offset: OffsetExpr::Const(a_row_off),
                                    width, dtype, predicate: None,
                                });

                                for col in 0..wj {
                                    let b_col_off = b_off + col * elem;
                                    let b_vec = prog.alloc_vreg(VRegKind::Vec, width);
                                    prog.emit(VmInstr::VecLoad {
                                        dst: b_vec, base: b_ptr,
                                        offset: OffsetExpr::Const(b_col_off),
                                        width, dtype, predicate: None,
                                    });

                                    // FMA: acc[row * wj + col] += a_vec * b_vec
                                    let acc_idx = row * wj + col;
                                    if acc_idx < accs.len() {
                                        prog.emit(VmInstr::Fma {
                                            dst: accs[acc_idx],
                                            acc: accs[acc_idx],
                                            a: a_vec,
                                            b: b_vec,
                                            dtype: acc_dtype,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 写回 C tile: c_ptr + (i_cta) * n * elem + (j_cta) * elem
            for row in 0..mi.min(warp_m) {
                for col in 0..nj.min(warp_n) {
                    let acc_idx = row * nj.min(warp_n) + col;
                    if acc_idx < accs.len() {
                        let c_off = ((i_cta + row) * n + j_cta + col) * elem;
                        let store_src = if dtype.needs_narrowing_from(acc_dtype) {
                            let narrowed = prog.alloc_vreg(VRegKind::Vec, width);
                            prog.emit(VmInstr::VecNarrow { dst: narrowed, src: accs[acc_idx], dst_dtype: dtype, src_dtype: acc_dtype, width });
                            narrowed
                        } else {
                            accs[acc_idx]
                        };
                        prog.emit(VmInstr::VecStore {
                            base: c_ptr,
                            offset: OffsetExpr::Const(c_off),
                            src: store_src,
                            width,
                            dtype, predicate: None,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// GPU 三级分块 GEMM + 双缓冲流水线: CTA × Warp × MMA + cp.async/TMA 与 MMA 重叠.
///
/// 流水线结构 (pipeline_depth=2 ping-pong):
/// ```
/// smem = [A_ping(cta_m × cta_k), A_pong, B_ping(cta_k × cta_n), B_pong]
///
/// Stage 0: AsyncLoad(A_ping ← global[k=0]) + AsyncLoad(B_ping ← global[k=0]) + Wait
/// Stage 1..K-1:
///   MMA(A_cur, B_cur) → acc   ‖   AsyncLoad(A_next ← global[k=i+1]) + AsyncLoad(B_next)
///   Wait (ensure next load done)
/// Last stage: MMA(A_last, B_last) → acc
/// ```
///
/// SM80: cp.async.ca.shared.global + cp.async.wait_group
/// SM90+: TMA cp.async.bulk + mbarrier.try_wait
pub(crate) fn emit_gemm_gpu_pipelined(
    prog: &mut VmProgram,
    m: usize, n: usize, k: usize,
    width: SimdWidth,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    cta_m: usize, cta_n: usize, cta_k: usize,
    warp_m: usize, warp_n: usize, mma_k: usize,
    pipeline_depth: usize,
    dtype: QuantPrecision,
    _trans_b: bool,
    use_tma: bool,
) -> Result<(), CompilerError> {
    let elem = dtype.elem_bytes();
    let acc_dtype = dtype.gpu_accumulator_dtype(); // REQ-DTYPE-005: GPU path
    let mma_k = mma_k.max(16);
    let num_stages = pipeline_depth.min(3);

    // Shared memory names for ping-pong buffers
    let smem_a_names: Vec<String> = (0..num_stages).map(|s| format!("smem_a_{}", s)).collect();
    let smem_b_names: Vec<String> = (0..num_stages).map(|s| format!("smem_b_{}", s)).collect();

    // Each smem tile: A = cta_m × cta_k elements, B = cta_k × cta_n elements
    // Pad rows to 128B alignment (32 banks × 4B) to eliminate bank conflicts.
    // Swizzle-free approach: pad each row to next multiple of 128 bytes.
    let smem_a_row_bytes = ((cta_k * elem + 127) / 128) * 128;
    let smem_b_row_bytes = ((cta_n * elem + 127) / 128) * 128;
    let smem_a_bytes = cta_m * smem_a_row_bytes;
    let smem_b_bytes = cta_k * smem_b_row_bytes;

    // Allocate shared memory for all pipeline stages
    for i in 0..num_stages {
        prog.emit(VmInstr::SharedMemAlloc {
            name: smem_a_names[i].clone(),
            bytes: smem_a_bytes,
        });
        prog.emit(VmInstr::SharedMemAlloc {
            name: smem_b_names[i].clone(),
            bytes: smem_b_bytes,
        });
    }

    // TMA prologue: initialize tensor descriptors and barrier (SM90+ only)
    if use_tma {
        // TMA descriptor for A matrix: shape (m, k), tile (cta_m, cta_k)
        prog.emit(VmInstr::TmaDescriptorInit {
            desc_name: "tma_desc_a".to_string(),
            global_dim: [m, k],
            global_stride: [k * elem, elem],
            box_dim: [cta_m, cta_k],
            swizzle: TmaSwizzle::Swizzle128,
            dtype,
        });
        // TMA descriptor for B matrix: shape (k, n), tile (cta_k, cta_n)
        prog.emit(VmInstr::TmaDescriptorInit {
            desc_name: "tma_desc_b".to_string(),
            global_dim: [k, n],
            global_stride: [n * elem, elem],
            box_dim: [cta_k, cta_n],
            swizzle: TmaSwizzle::Swizzle128,
            dtype,
        });
        // mbarrier for TMA completion signal
        prog.emit(VmInstr::BarrierInit {
            name: "tma_bar".to_string(),
            thread_count: 1,
        });
    }

    // Accumulator registers: warp_m × warp_n
    let acc_count = warp_m * warp_n;
    let accs: Vec<VRegId> = (0..acc_count.min(16))
        .map(|_| prog.alloc_vreg(VRegKind::Vec, width))
        .collect();

    // CTA-level M/N loops
    for i_cta in (0..m).step_by(cta_m) {
        let mi = cta_m.min(m - i_cta);
        for j_cta in (0..n).step_by(cta_n) {
            let nj = cta_n.min(n - j_cta);

            // Zero accumulators
            for acc in &accs {
                prog.emit(VmInstr::Broadcast { dst: *acc, src: ScalarExpr::Const(0.0), width, dtype: acc_dtype });
            }

            // K tiles
            let k_tiles: Vec<usize> = (0..k).step_by(cta_k).collect();
            let num_k_tiles = k_tiles.len();

            if num_k_tiles == 0 {
                // No K iterations — skip to writeback
            } else {
                // ─── Prologue: async load stage 0 ───
                let k0 = k_tiles[0];
                let kk0 = cta_k.min(k - k0);
                emit_async_load_tile(prog, &smem_a_names[0], &smem_b_names[0],
                    a_ptr, b_ptr, i_cta, j_cta, k0, mi, nj, kk0,
                    cta_m, cta_n, cta_k, n, k, elem, dtype, width,
                    smem_a_row_bytes, smem_b_row_bytes, use_tma);

                if num_k_tiles == 1 {
                    // Single K tile: wait + compute + done
                    if use_tma {
                        prog.emit(VmInstr::WarpBarrierWait {
                            barrier_name: "tma_bar".to_string(),
                            parity: 0,
                        });
                    } else {
                        prog.emit(VmInstr::AsyncWait { handle: 0 });
                    }
                    emit_mma_on_smem(prog, &accs, &smem_a_names[0], &smem_b_names[0],
                        i_cta, j_cta, k0, mi, nj, kk0.min(cta_k),
                        cta_m, cta_n, cta_k, warp_m, warp_n, mma_k,
                        n, k, elem, dtype, acc_dtype, width,
                        smem_a_row_bytes, smem_b_row_bytes);
                } else {
                    // ─── Steady state: compute tile[i] ‖ async load tile[i+1] ───
                    let mut cur_stage = 0usize;
                    let mut parity = 0u32;

                    for tile_idx in 0..num_k_tiles {
                        let next_stage = (cur_stage + 1) % num_stages;

                        if tile_idx > 0 {
                            // Wait for previous async load of current tile to complete
                            if use_tma {
                                prog.emit(VmInstr::WarpBarrierWait {
                                    barrier_name: "tma_bar".to_string(),
                                    parity,
                                });
                            } else {
                                prog.emit(VmInstr::AsyncWait { handle: 0 });
                            }
                        }

                        let ki = k_tiles[tile_idx];
                        let kki = cta_k.min(k - ki);

                        if tile_idx < num_k_tiles - 1 {
                            // Compute current tile from smem[cur_stage] ‖ async load next tile into smem[next_stage]
                            emit_mma_on_smem(prog, &accs, &smem_a_names[cur_stage], &smem_b_names[cur_stage],
                                i_cta, j_cta, ki, mi, nj, kki,
                                cta_m, cta_n, cta_k, warp_m, warp_n, mma_k,
                                n, k, elem, dtype, acc_dtype, width,
                                smem_a_row_bytes, smem_b_row_bytes);

                            // Async load next K tile
                            let ki_next = k_tiles[tile_idx + 1];
                            let kki_next = cta_k.min(k - ki_next);
                            emit_async_load_tile(prog, &smem_a_names[next_stage], &smem_b_names[next_stage],
                                a_ptr, b_ptr, i_cta, j_cta, ki_next, mi, nj, kki_next,
                                cta_m, cta_n, cta_k, n, k, elem, dtype, width,
                                smem_a_row_bytes, smem_b_row_bytes, use_tma);
                        } else {
                            // Epilogue: compute last tile (no more async loads)
                            emit_mma_on_smem(prog, &accs, &smem_a_names[cur_stage], &smem_b_names[cur_stage],
                                i_cta, j_cta, ki, mi, nj, kki,
                                cta_m, cta_n, cta_k, warp_m, warp_n, mma_k,
                                n, k, elem, dtype, acc_dtype, width,
                                smem_a_row_bytes, smem_b_row_bytes);
                        }

                        parity ^= 1;
                        cur_stage = next_stage;
                    }
                }
            }

            // ─── Writeback C tile ───
            for row in 0..mi.min(warp_m) {
                for col in 0..nj.min(warp_n) {
                    let acc_idx = row * nj.min(warp_n) + col;
                    if acc_idx < accs.len() {
                        let c_off = ((i_cta + row) * n + j_cta + col) * elem;
                        let store_src = if dtype.needs_narrowing_from(acc_dtype) {
                            let narrowed = prog.alloc_vreg(VRegKind::Vec, width);
                            prog.emit(VmInstr::VecNarrow { dst: narrowed, src: accs[acc_idx], dst_dtype: dtype, src_dtype: acc_dtype, width });
                            narrowed
                        } else {
                            accs[acc_idx]
                        };
                        prog.emit(VmInstr::VecStore {
                            base: c_ptr,
                            offset: OffsetExpr::Const(c_off),
                            src: store_src,
                            width,
                            dtype, predicate: None,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Async load A/B tiles from global memory to shared memory.
/// A tile: rows [i_cta..i_cta+mi), cols [k_off..k_off+kk) → smem_a
/// B tile: rows [k_off..k_off+kk), cols [j_cta..j_cta+nj) → smem_b
///
/// When `use_tma` is true (SM90+), uses TMA 2D tensor copy — a single instruction
/// per tile that offloads the entire transfer to hardware. Coord registers hold
/// the tile's starting row/column in the global tensor; the TMA descriptor carries
/// box dimensions so the hardware knows the transfer extent.
/// When `use_tma` is false (SM80), falls back to per-row VecLoad + SharedMemAsyncStore.
pub(crate) fn emit_async_load_tile(
    prog: &mut VmProgram,
    smem_a_name: &str,
    smem_b_name: &str,
    _a_ptr: VRegId,
    _b_ptr: VRegId,
    i_cta: usize,
    j_cta: usize,
    k_off: usize,
    mi: usize,
    _nj: usize,
    kk: usize,
    cta_m: usize,
    cta_n: usize,
    cta_k: usize,
    n: usize,
    k: usize,
    elem: usize,
    dtype: QuantPrecision,
    width: SimdWidth,
    smem_a_row_stride: usize,
    smem_b_row_stride: usize,
    use_tma: bool,
) {
    if use_tma {
        // TMA path (SM90+): one Tma2DCopy per tile — hardware handles the entire transfer.
        // coord_x / coord_y are GPR registers holding the tile's global starting coordinates.
        let coord_row_a = prog.alloc_vreg(VRegKind::Scalar, width);
        prog.emit(VmInstr::GprLoadImm { dst: coord_row_a, value: i_cta });
        let coord_col_a = prog.alloc_vreg(VRegKind::Scalar, width);
        prog.emit(VmInstr::GprLoadImm { dst: coord_col_a, value: k_off });

        prog.emit(VmInstr::Tma2DCopy {
            desc_name: "tma_desc_a".to_string(),
            smem_name: smem_a_name.to_string(),
            coord_x: coord_row_a,
            coord_y: coord_col_a,
            barrier_name: "tma_bar".to_string(),
        });

        let coord_row_b = prog.alloc_vreg(VRegKind::Scalar, width);
        prog.emit(VmInstr::GprLoadImm { dst: coord_row_b, value: k_off });
        let coord_col_b = prog.alloc_vreg(VRegKind::Scalar, width);
        prog.emit(VmInstr::GprLoadImm { dst: coord_col_b, value: j_cta });

        prog.emit(VmInstr::Tma2DCopy {
            desc_name: "tma_desc_b".to_string(),
            smem_name: smem_b_name.to_string(),
            coord_x: coord_row_b,
            coord_y: coord_col_b,
            barrier_name: "tma_bar".to_string(),
        });
    } else {
        // SM80 fallback path: per-row VecLoad + SharedMemAsyncStore

        // Load A tile (row-major): cta_m rows × cta_k cols from global → smem_a
        // Each row uses padded row stride for bank conflict elimination.
        for row in 0..mi {
            let global_offset = ((i_cta + row) * k + k_off) * elem;
            let smem_offset = row * smem_a_row_stride;
            let tmp = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: tmp,
                base: _a_ptr,
                offset: OffsetExpr::Const(global_offset),
                width,
                dtype, predicate: None,
            });
            prog.emit(VmInstr::SharedMemAsyncStore {
                name: smem_a_name.to_string(),
                dst_offset: OffsetExpr::Const(smem_offset),
                src: tmp,
                width,
                dtype,
            });
        }

        // Load B tile (row-major): cta_k rows × cta_n cols from global → smem_b
        for row in 0..kk {
            let global_offset = ((k_off + row) * n + j_cta) * elem;
            let smem_offset = row * smem_b_row_stride;
            let tmp = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: tmp,
                base: _b_ptr,
                offset: OffsetExpr::Const(global_offset),
                width,
                dtype, predicate: None,
            });
            prog.emit(VmInstr::SharedMemAsyncStore {
                name: smem_b_name.to_string(),
                dst_offset: OffsetExpr::Const(smem_offset),
                src: tmp,
                width,
                dtype,
            });
        }
    }
}

/// Compute MMA on shared memory tiles, accumulating into acc registers.
/// Warp-level: iterate warp_m × warp_n sub-tiles within the CTA tile.
/// Inner K: iterate mma_k depth within cta_k.
pub(crate) fn emit_mma_on_smem(
    prog: &mut VmProgram,
    accs: &[VRegId],
    smem_a_name: &str,
    smem_b_name: &str,
    _i_cta: usize,
    _j_cta: usize,
    _k_off: usize,
    mi: usize,
    nj: usize,
    kk: usize,
    cta_m: usize,
    cta_n: usize,
    cta_k: usize,
    warp_m: usize,
    warp_n: usize,
    mma_k: usize,
    _n: usize,
    _k: usize,
    _elem: usize,
    _dtype: QuantPrecision,
    acc_dtype: QuantPrecision,
    width: SimdWidth,
    smem_a_row_stride: usize,
    smem_b_row_stride: usize,
) {
    // Padded row stride in f32 elements (divide byte stride by 4)
    let smem_a_row_elems = smem_a_row_stride / 4;
    let smem_b_row_elems = smem_b_row_stride / 4;

    // Warp-level loop within CTA tile
    for i_warp in (0..mi).step_by(warp_m) {
        let wi = warp_m.min(mi - i_warp);
        for j_warp in (0..nj).step_by(warp_n) {
            let wj = warp_n.min(nj - j_warp);

            // Inner K loop: mma_k depth
            for k_inner in (0..kk).step_by(mma_k) {
                // Load A fragment from smem_a: row (i_warp + row), col k_inner
                for row in 0..wi {
                    let smem_a_off = (i_warp + row) * smem_a_row_elems + k_inner;
                    let a_vec = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::SharedMemLoad {
                        dst: a_vec,
                        name: smem_a_name.to_string(),
                        src_offset: OffsetExpr::Const(smem_a_off * 4), // f32 bytes
                        width,
                        dtype: acc_dtype,
                    });

                    for col in 0..wj {
                        let smem_b_off = k_inner * smem_b_row_elems + j_warp + col;
                        let b_vec = prog.alloc_vreg(VRegKind::Vec, width);
                        prog.emit(VmInstr::SharedMemLoad {
                            dst: b_vec,
                            name: smem_b_name.to_string(),
                            src_offset: OffsetExpr::Const(smem_b_off * 4),
                            width,
                            dtype: acc_dtype,
                        });

                        let acc_idx = row * wj + col;
                        if acc_idx < accs.len() {
                            prog.emit(VmInstr::Fma {
                                dst: accs[acc_idx],
                                acc: accs[acc_idx],
                                a: a_vec,
                                b: b_vec,
                                dtype: acc_dtype,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// GEMM 内联 (原始 naive 路径)。
/// Build B-matrix byte offset for GEMM inner K loop.
/// k_counter: loop counter VRegId (iteration index 0..k)
/// k_off: byte offset VRegId (= counter * step_bytes)
/// j_off: byte offset VRegId for J dimension
pub(crate) fn b_offset_expr(
    k_counter: VRegId, k_off: VRegId,
    j_off: VRegId,
    b_row_stride: usize, _elem: usize, _n: usize, k: usize, trans_b: bool,
    _lanes: usize,
) -> OffsetExpr {
    if trans_b {
        // B[j,p]: offset = j * k * elem + p * elem
        // j_off accumulates j_iter * elem, so j = j_off / elem
        // j * k * elem = j_off * k
        OffsetExpr::Add(
            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(j_off)), k)),
            Box::new(OffsetExpr::LoopOffset(k_off)),
        )
    } else {
        // B[p,j]: offset = p * n * elem + j_byte_off
        // p = k_counter (iteration index), b_row_stride = n * elem
        OffsetExpr::Add(
            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::ScalarVReg(k_counter)), b_row_stride)),
            Box::new(OffsetExpr::LoopOffset(j_off)),
        )
    }
}

/// `m_dim`: M 维度的 SymDim。Symbolic → sym_map.to_bound() 生成运行时循环;
/// Concrete → emit_loop(BoundExpr::Const)。禁止硬编码 StackArg 偏移。

/// Transposed-B GEMM with K-dimension vectorization.
///
/// When trans_b=true, B is [N,K] row-major. B[j][p] and B[j][p+1] are contiguous
/// (same row), but B[j][p] and B[j+1][p] are NOT contiguous (stride=K*elem).
/// Standard J-dimension vectorization fails — we vectorize along K instead.
pub(crate) fn emit_gemm_trans_b_inline(
    prog: &mut VmProgram,
    m_dim: &SymDim, n: usize, k: usize,
    width: SimdWidth,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    epilogue: &[TraceOp],
    sym_map: &SymDimSlotMap,
    seq_bound_override: Option<&BoundExpr>,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let m = match m_dim {
        SymDim::Concrete(v) => *v,
        SymDim::Symbolic { max_value, .. } => max_value
            .expect("ARCH-SYMDIM: GEMM M Symbolic must have max_value"),
    };
    if m == 0 || n == 0 || k == 0 {
        return Err(CompilerError::CodegenViolation(format!("zero dim ({m},{n},{k})")));
    }
    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let acc_dtype = dtype.accumulator_dtype();
    let needs_narrow = dtype.needs_narrowing_from(acc_dtype);
    let k_vecs = k / lanes;
    let k_tail = k - k_vecs * lanes;
    let a_row_stride = k * elem;
    let c_row_stride = n * elem;
    let b_row_stride = k * elem;
    let k_step = lanes * elem;

    let acc_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let a_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let b_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let s_width = SimdWidth::Scalar;
    let s_acc = prog.alloc_vreg(VRegKind::Vec, s_width);
    let s_tail = prog.alloc_vreg(VRegKind::Vec, s_width);


    // j loop uses emit_loop to avoid compile-time unrolling (n can be 576/1536).
    let emit_j_loop = |prog: &mut VmProgram, m_off: OffsetExpr| {
        prog.emit_loop(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| {
            // j_byte_off = j * b_row_stride = j * k * elem
            // But j_off accumulates j_iter * elem, so j = j_off / elem
            // B[j][p] offset = j * k * elem + p * elem = j_off * k + p_off
            // Similarly for C[i][j] offset = m_off * c_row_stride + j_off

            prog.emit(VmInstr::Broadcast {
                dst: acc_vec, src: ScalarExpr::Const(0.0), width, dtype: acc_dtype,
            });

            if k_vecs > 0 {
                prog.emit_loop(BoundExpr::Const(k_vecs), k_step, |prog, _p_ctr, p_off| {
                    // Load A[i][p*lanes .. (p+1)*lanes]
                    prog.emit(VmInstr::VecLoad {
                        dst: a_vec, base: a_ptr,
                        offset: OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(
                                Box::new(m_off.clone()), a_row_stride,
                            )),
                            Box::new(OffsetExpr::LoopOffset(p_off)),
                        ),
                        width, dtype, predicate: None,
                    });
                    // Load B[j][p*lanes .. (p+1)*lanes]
                    // B row offset = j * k * elem = j_off * k (since j_off is in bytes, j_off / elem * k * elem = j_off * k)
                    // But B[j][p_start] offset = j * k * elem + p_start * elem
                    // = (j_off / elem) * k * elem + p_start * elem
                    // = j_off * k + p_start (where p_start is LoopOffset in bytes)
                    prog.emit(VmInstr::VecLoad {
                        dst: b_vec, base: b_ptr,
                        offset: OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(j_off)), k)),
                            Box::new(OffsetExpr::LoopOffset(p_off)),
                        ),
                        width, dtype, predicate: None,
                    });
                    // Direct FMA: dst = acc + a * b (in-place accumulate)
                    prog.emit(VmInstr::Fma { dst: acc_vec, acc: acc_vec, a: a_vec, b: b_vec, dtype });
                });
            }

            if k_tail > 0 {
                let tail_base = k_vecs * lanes * elem;
                prog.emit(VmInstr::Broadcast {
                    dst: s_tail, src: ScalarExpr::Const(0.0), width: s_width, dtype: acc_dtype,
                });
                for t in 0..k_tail {
                    let p_byte = tail_base + t * elem;
                    prog.emit(VmInstr::Broadcast {
                        dst: s_acc,
                        src: ScalarExpr::MemLoad(a_ptr, OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(m_off.clone()), a_row_stride)),
                            Box::new(OffsetExpr::Const(p_byte)),
                        )),
                        width: s_width, dtype,
                    });
                    let s_b = prog.alloc_vreg(VRegKind::Vec, s_width);
                    prog.emit(VmInstr::Broadcast {
                        dst: s_b,
                        src: ScalarExpr::MemLoad(b_ptr, OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(j_off)), k)),
                            Box::new(OffsetExpr::Const(p_byte)),
                        )),
                        width: s_width, dtype,
                    });
                    // Direct FMA: s_tail = s_tail + s_acc * s_b
                    prog.emit(VmInstr::Fma { dst: s_tail, acc: s_tail, a: s_acc, b: s_b, dtype });
                }
            }

            let reduced = prog.alloc_vreg(VRegKind::Vec, s_width);
            prog.emit(VmInstr::HReduce { dst: reduced, src: acc_vec, op: ReduceOp::Sum });

            let result = if k_tail > 0 {
                let sum = prog.alloc_vreg(VRegKind::Vec, s_width);
                let add_body: Vec<TraceOp> = vec![
                    TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(ValueId(0), ValueId(1)),
                ];
                super::auto_select::auto_lower_trace_into(
                    prog, &add_body, &[reduced, s_tail], sum, s_width, dtype,
                ).expect("trans_b GEMM add tail");
                sum
            } else {
                reduced
            };

            if !epilogue.is_empty() {
                lower::lower_trace_body_compat(prog, epilogue, result, None, s_width, dtype)
                    .expect("lower_trace_body: OpTrace invariant violation");
            }

            let store_src = if needs_narrow {
                let narrowed = prog.alloc_vreg(VRegKind::Vec, s_width);
                prog.emit(VmInstr::VecNarrow {
                    dst: narrowed, src: result, dst_dtype: dtype, src_dtype: acc_dtype, width: s_width,
                });
                narrowed
            } else {
                result
            };

            prog.emit(VmInstr::VecStore {
                base: c_ptr,
                offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(m_off.clone()), c_row_stride)),
                    Box::new(OffsetExpr::LoopOffset(j_off)),
                ),
                src: store_src, width: s_width, dtype, predicate: None,
            });
        });
    };

    let m_bound = if m_dim.is_symbolic() {
        seq_bound_override.cloned().unwrap_or_else(|| sym_map.to_bound(m_dim))
    } else {
        BoundExpr::Const(m)
    };
    prog.emit_loop(m_bound, 1, |prog, _m_ctr, m_off| {
        emit_j_loop(prog, OffsetExpr::LoopOffset(m_off));
    });

    Ok(())
}

pub(crate) fn emit_gemm_inline_with_epilogue(
    prog: &mut VmProgram,
    m_dim: &SymDim, n: usize, k: usize,
    width: SimdWidth,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    epilogue: &[TraceOp],
    sym_map: &SymDimSlotMap,
    enable_row_stats: bool,
    seq_bound_override: Option<&BoundExpr>,
    dtype: QuantPrecision,
    trans_b: bool,
    epi_place: super::isa_hook::EpiloguePlace,
) -> Result<(), CompilerError> {
    let m = match m_dim {
        SymDim::Concrete(v) => *v,
        SymDim::Symbolic { max_value, .. } => max_value
            .expect("ARCH-SYMDIM: GEMM M Symbolic must have max_value"),
    };
    if m == 0 || n == 0 || k == 0 {
        return Err(CompilerError::CodegenViolation(format!("zero dim ({m},{n},{k})")));
    }
    // Route trans_b to K-dimension vectorized path
    if trans_b {
        return emit_gemm_trans_b_inline(
            prog, m_dim, n, k, width, a_ptr, b_ptr, c_ptr,
            epilogue, sym_map, seq_bound_override, dtype,
        );
    }
    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let acc_dtype = dtype.accumulator_dtype();
    let needs_narrow = dtype.needs_narrowing_from(acc_dtype);
    let n_vecs = n / lanes;

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let a_broadcast = prog.alloc_vreg(VRegKind::Vec, width);
    let b_vec = prog.alloc_vreg(VRegKind::Vec, width);

    // ARCH-GEMM-NTAIL: 当 n < lanes 或 n % lanes != 0 时, vectorized j 循环
    // 只覆盖前 n_vecs*lanes 列, 剩下的 [n_vecs*lanes, n) 必须用 scalar tail
    // 覆盖, 否则 output 的那些列永远不会被写 → 读到内存垃圾 (ARCH-DATA-FLOW-CONTRACT §6)。
    let n_tail = n - n_vecs * lanes;
    let tail_base_bytes = n_vecs * lanes * elem;

    // ARCH-EPILOGUE-PLACE: 根据 EpiloguePlace 策略决定 epilogue 执行位置
    let do_epilogue_inline = matches!(epi_place, super::isa_hook::EpiloguePlace::OnAccumulators);

    let a_row_stride = k * dtype.elem_bytes();
    let c_row_stride = n * dtype.elem_bytes();
    let b_row_stride = n * dtype.elem_bytes();
    let j_step = lanes * elem;

    // Scalar tail VReg (仅 n_tail>0 时使用)。在外层分配一次避免在 m_loop closure
    // 内反复 alloc (VReg 生命周期覆盖整个 m 循环, 由 RegAlloc Pass 3 延展)。
    let (s_acc, s_a, s_b) = if n_tail > 0 {
        let s_width = SimdWidth::Scalar;
        (
            prog.alloc_vreg(VRegKind::Vec, s_width),
            prog.alloc_vreg(VRegKind::Vec, s_width),
            prog.alloc_vreg(VRegKind::Vec, s_width),
        )
    } else {
        (acc, a_broadcast, b_vec) // 未使用时复用,避免 alloc
    };

    // Unified M-loop body shared by both Symbolic and Concrete M dimensions.
    // Only the outer loop bound differs; inner j/k loops + tail handling identical.
    let m_bound = if m_dim.is_symbolic() {
        seq_bound_override.cloned().unwrap_or_else(|| sym_map.to_bound(m_dim))
    } else {
        BoundExpr::Const(m)
    };
    prog.emit_loop(m_bound, 1, |prog, _m_ctr, m_off| {
        if n_vecs > 0 {
            prog.emit_loop(BoundExpr::Const(n_vecs), j_step, |prog, _j_ctr, j_off| {
                prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype: acc_dtype });
                prog.emit_loop(BoundExpr::Const(k), elem, |prog, k_ctr, k_off| {
                    prog.emit(VmInstr::Broadcast {
                        dst: a_broadcast,
                        src: ScalarExpr::MemLoad(a_ptr, OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(m_off)), a_row_stride)),
                            Box::new(OffsetExpr::LoopOffset(k_off)),
                        )),
                        width, dtype,
                    });
                    prog.emit(VmInstr::VecLoad {
                        dst: b_vec, base: b_ptr,
                        offset: b_offset_expr(k_ctr, k_off, j_off, b_row_stride, elem, n, k, trans_b, lanes),
                        width, dtype, predicate: None,
                    });
                    // Direct FMA: dst = acc + a * b (in-place accumulate)
                    prog.emit(VmInstr::Fma { dst: acc, acc, a: a_broadcast, b: b_vec, dtype });
                });
                if !epilogue.is_empty() && do_epilogue_inline {
                    lower::lower_trace_body_compat(prog, epilogue, acc, None, width, dtype)
                        .expect("lower_trace_body: OpTrace invariant violation");
                }
                let store_src = if needs_narrow {
                    let narrowed = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecNarrow { dst: narrowed, src: acc, dst_dtype: dtype, src_dtype: acc_dtype, width });
                    narrowed
                } else { acc };
                prog.emit(VmInstr::VecStore {
                    base: c_ptr,
                    offset: OffsetExpr::Add(
                        Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(m_off)), c_row_stride)),
                        Box::new(OffsetExpr::LoopOffset(j_off)),
                    ),
                    src: store_src, width, dtype, predicate: None,
                });
            });
        }
        if n_tail > 0 {
            let s_width = SimdWidth::Scalar;
            for t in 0..n_tail {
                let j_off_const = tail_base_bytes + t * elem;
                prog.emit(VmInstr::Broadcast { dst: s_acc, src: ScalarExpr::Const(0.0), width: s_width, dtype: acc_dtype });
                prog.emit_loop(BoundExpr::Const(k), elem, |prog, k_ctr, k_off| {
                    prog.emit(VmInstr::Broadcast {
                        dst: s_a,
                        src: ScalarExpr::MemLoad(a_ptr, OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(m_off)), a_row_stride)),
                            Box::new(OffsetExpr::LoopOffset(k_off)),
                        )),
                        width: s_width, dtype,
                    });
                    prog.emit(VmInstr::VecLoad {
                        dst: s_b, base: b_ptr,
                        offset: if trans_b {
                            OffsetExpr::Add(
                                Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::Const(j_off_const)), k)),
                                Box::new(OffsetExpr::LoopOffset(k_off)),
                            )
                        } else {
                            OffsetExpr::Add(
                                Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(k_ctr)), b_row_stride)),
                                Box::new(OffsetExpr::Const(j_off_const)),
                            )
                        },
                        width: s_width, dtype, predicate: None,
                    });
                    // Direct FMA: dst = s_acc + s_a * s_b (in-place accumulate)
                    prog.emit(VmInstr::Fma { dst: s_acc, acc: s_acc, a: s_a, b: s_b, dtype });
                });
                if !epilogue.is_empty() && do_epilogue_inline {
                    lower::lower_trace_body_compat(prog, epilogue, s_acc, None, s_width, dtype)
                        .expect("lower_trace_body: OpTrace invariant violation");
                }
                // REQ-DTYPE-006: 窄化写回 (scalar tail path)
                let s_store_src = if needs_narrow {
                    let s_narrowed = prog.alloc_vreg(VRegKind::Vec, s_width);
                    prog.emit(VmInstr::VecNarrow { dst: s_narrowed, src: s_acc, dst_dtype: dtype, src_dtype: acc_dtype, width: s_width });
                    s_narrowed
                } else { s_acc };
                prog.emit(VmInstr::VecStore {
                    base: c_ptr,
                    offset: OffsetExpr::Add(
                        Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(m_off)), c_row_stride)),
                        Box::new(OffsetExpr::Const(j_off_const)),
                    ),
                    src: s_store_src, width: s_width, dtype, predicate: None,
                });
            }
        }
    });

    // §13.7 GEMM row-level activation stats telemetry:
    // After the m×j loops complete, emit L1 norm / max / min statistics
    // from the last row's first j-vector tile (representative sample).
    // We load from c_ptr at the last row offset to avoid needing the accumulator
    // register after the loop (accumulator VReg lifetime may not extend past loop end).
    // Only emit when telemetry is enabled (graph.telemetry.gemm_row_stats == true).
    if enable_row_stats && n_vecs > 0 {
        let last_row_offset = (m.saturating_sub(1)) * c_row_stride;
        let row_stats_vec = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecLoad {
            dst: row_stats_vec, base: c_ptr,
            offset: OffsetExpr::Const(last_row_offset),
            width,
            dtype, predicate: None,
        });
        emit_gemm_row_stats_telemetry(prog, row_stats_vec, width, sym_map, dtype)?;
    }

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 27 REQ-AT-008: 模板驱动 GEMM emit 桥接
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 模板驱动 GEMM emit — SPEC 27 归一管线
///
/// 通过 `select_template()` 选择算法模板 → `TemplateInterpreter::instantiate()`
/// 产出 `Vec<TraceOp>` → `auto_lower_trace_raw()` 编译为 VmInstr。
///
/// 返回 `(template_name, trace_ops)` 用于调用方验证和调试。
/// 当模板不可用或参数不足时返回 `None`。
pub(crate) fn emit_gemm_template_driven(
    prog: &mut VmProgram,
    m: usize, n: usize, k: usize,
    width: SimdWidth,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,
    dtype: QuantPrecision,
    resource_plan: Option<&super::resource_planner::GraphResourcePlan>,
) -> Option<(String, Vec<TraceOp>)> {
    use super::algo_registry;
    use super::algo_interpreter::{TemplateInterpreter, ParamTable, TemplateInputs};
    use crate::dispatch::device_profile::DeviceProfile;

    // 策略选择: 根据 DeviceProfile 选择最优模板
    let profile = DeviceProfile::detect();
    let (strategy, template) = {
        // 尝试 BLIS (CPU optimized)
        if let Some(tmpl) = algo_registry::select_template(
            &crate::compiler::codegen::vm::algo_template::AlgoStrategy::GemmBlis,
            &profile,
        ) {
            (crate::compiler::codegen::vm::algo_template::AlgoStrategy::GemmBlis, tmpl)
        } else {
            let tmpl = algo_registry::select_template(
            &crate::compiler::codegen::vm::algo_template::AlgoStrategy::GemmNaive,
            &profile,
        )?;
            (crate::compiler::codegen::vm::algo_template::AlgoStrategy::GemmNaive, tmpl)
        }
    };

    // 填充参数表 — 优先使用 GraphResourcePlan 的 suggested_blocking
    let lanes = width.f32_lanes().max(1);
    let mut params = ParamTable::new();
    params.set("m", m);
    params.set("n", n);
    params.set("k", k);

    let (mr, nr, mc, nc, kc) = if let Some(plan) = resource_plan {
        if let Some(blocking) = plan.gemm_blocking_for_group(0) {
            (blocking.mr, blocking.nr, blocking.mc, blocking.nc, blocking.kc)
        } else {
            (4, lanes, m.min(64), n.min(64), k.min(32))
        }
    } else {
        (4, lanes, m.min(64), n.min(64), k.min(32))
    };

    params.set("mr", mr);
    params.set("nr", nr);
    params.set("mc", mc);
    params.set("nc", nc);
    params.set("kc", kc);
    params.set("k_step", 1);

    // 实例化模板
    let inputs = TemplateInputs::gemm();
    let mut interp = TemplateInterpreter::new(params);
    let trace_ops = interp.instantiate(template, &inputs);

    // 编译 TraceOp → VmInstr
    // TemplateInputs::gemm() 需要 6 个输入: a_ptr, b_ptr, c_ptr, a_offset, b_offset, c_offset
    let a_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    let b_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    let c_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);

    super::auto_select::auto_lower_trace_raw(
        prog, &trace_ops, &[a_ptr, b_ptr, c_ptr, a_offset, b_offset, c_offset], width, dtype,
    ).ok()?;

    Some((template.name.to_string(), trace_ops))
}

#[cfg(test)]
mod template_tests {
    use super::*;

    #[test]
    fn test_template_driven_gemm_produces_instrs() {
        use crate::compiler::codegen::vm::algo_registry;
        use crate::compiler::codegen::vm::algo_interpreter::{TemplateInterpreter, ParamTable, TemplateInputs};
        use crate::dispatch::device_profile::DeviceProfile;
        use crate::compiler::codegen::vm::algo_template::AlgoStrategy;

        let profile = DeviceProfile::detect();
        let tmpl = algo_registry::select_template(&AlgoStrategy::GemmNaive, &profile)
            .expect("GEMM_NAIVE should always match");

        let mut params = ParamTable::new();
        params.set("m", 4);
        params.set("n", 8);
        params.set("k", 16);
        params.set("mr", 4);
        params.set("nr", 8);
        let inputs = TemplateInputs::gemm();
        let mut interp = TemplateInterpreter::new(params);
        let ops = interp.instantiate(tmpl, &inputs);

        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let a_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let b_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let c_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);

        let lower_result = super::super::auto_select::auto_lower_trace_raw(
            &mut prog, &ops, &[a_ptr, b_ptr, c_ptr, a_offset, b_offset, c_offset], width, QuantPrecision::F32,
        );
        assert!(lower_result.is_ok(), "auto_lower_trace_raw should succeed: {:?}", lower_result);
        assert!(!prog.instrs.is_empty(), "should produce VmInstrs");
    }

    #[test]
    fn test_template_driven_gemm_large() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, width);

        let result = emit_gemm_template_driven(
            &mut prog, 32, 64, 128,
            width, a_ptr, b_ptr, c_ptr,
            QuantPrecision::F32,
            None,
        );

        assert!(result.is_some(), "template-driven GEMM large should succeed");
        let (name, _trace_ops) = result.unwrap();
        assert!(name == "GEMM_NAIVE" || name == "GEMM_BLIS",
            "template name should be GEMM_NAIVE or GEMM_BLIS, got {}", name);
        assert!(prog.instrs.len() > 10,
            "large GEMM should produce significant instructions, got {}", prog.instrs.len());
    }

    #[test]
    fn test_b_offset_expr_normal_layout() {
        let prog = &mut VmProgram::new();
        let k_ctr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let j_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let b_row_stride = 256 * 4; // n=256, f32
        let n = 256;
        let k = 128;

        let expr = b_offset_expr(k_ctr, k_off, j_off, b_row_stride, 4, n, k, false, 8);

        // Normal (non-trans) B: offset = k_ctr * b_row_stride + j_off
        match expr {
            OffsetExpr::Add(inner, _) => {
                assert!(matches!(*inner, OffsetExpr::Mul(..)));
            }
            other => panic!("expected Add for normal layout, got {:?}", other),
        }
    }

    #[test]
    fn test_b_offset_expr_trans_b_layout() {
        let prog = &mut VmProgram::new();
        let k_ctr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let j_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let n = 256;
        let k = 128;

        let expr = b_offset_expr(k_ctr, k_off, j_off, 0, 4, n, k, true, 8);

        // Transposed B: offset = j_off * k + k_off
        match expr {
            OffsetExpr::Add(inner, _) => {
                assert!(matches!(*inner, OffsetExpr::Mul(..)), "trans-B first term should be Mul(j_off, k)");
            }
            other => panic!("expected Add for trans-B layout, got {:?}", other),
        }
    }

    #[test]
    fn test_gemm_inline_with_epilogue_rejects_zero_dim() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(0), 4, 4, width, a, b, c, &[], &sym_map, false, None, dtype, false, crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );
        assert!(result.is_err(), "zero M should be rejected");
    }

    #[test]
    fn test_gemm_inline_with_epilogue_rejects_zero_n() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(4), 0, 4, width, a, b, c, &[], &sym_map, false, None, dtype, false, crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );
        assert!(result.is_err(), "zero N should be rejected");
    }

    #[test]
    fn test_gemm_inline_with_epilogue_rejects_zero_k() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(4), 4, 0, width, a, b, c, &[], &sym_map, false, None, dtype, false, crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );
        assert!(result.is_err(), "zero K should be rejected");
    }

    #[test]
    fn test_gemm_blis_inline_produces_instrs() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        let result = emit_gemm_blis_inline(
            &mut prog, 4, 8, 16, width, a, b, c, 4, 2, None, 1, dtype, false,
        );
        assert!(result.is_ok(), "BLIS GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "BLIS GEMM should produce instructions");
    }

    #[test]
    fn test_gemm_gpu_tiled_inline_produces_instrs() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 16, 16, 16, width, a, b, c,
            16, 16, 16, 16, 16, 16, dtype, false,
        );
        assert!(result.is_ok(), "GPU tiled GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "GPU tiled GEMM should produce instructions");
    }

    #[test]
    fn test_gemm_inline_with_epilogue_concrete_m() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 8, 8, width, a, b, c, &[], &sym_map, false, None, dtype, false, crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );
        assert!(result.is_ok(), "small concrete GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +13 new tests: struct constructors, Debug/Clone, enum variants,
    // Default impls, field boundary values, float precision, overflow safety,
    // struct update syntax
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_simd_width_scalar_f32_lanes_and_bytes() {
        // Arrange: Scalar width represents a single f32 element
        let scalar = SimdWidth::Scalar;

        // Act
        let lanes = scalar.f32_lanes();
        let bytes = scalar.bytes();

        // Assert: scalar has exactly 1 lane and 4 bytes (one f32)
        assert_eq!(lanes, 1, "Scalar SIMD width should have 1 f32 lane");
        assert_eq!(bytes, 4, "Scalar SIMD width should be 4 bytes");
    }

    #[test]
    fn test_simd_width_warp_variant_carries_thread_count() {
        // Arrange: Warp(32) = NVIDIA warp, Warp(64) = AMD wavefront
        let nv = SimdWidth::Warp(32);
        let amd = SimdWidth::Warp(64);

        // Act
        let nv_lanes = nv.f32_lanes();
        let amd_lanes = amd.f32_lanes();

        // Assert: warp size propagates through as lane count
        assert_eq!(nv_lanes, 32, "NVIDIA warp should report 32 lanes");
        assert_eq!(amd_lanes, 64, "AMD wavefront should report 64 lanes");
    }

    #[test]
    fn test_vreg_id_copy_and_equality() {
        // Arrange: VRegId is Copy + PartialEq
        let id_a = VRegId(42);
        let id_b = id_a; // Copy, not move
        let id_c = VRegId(42);
        let id_d = VRegId(99);

        // Assert: Copy semantics and equality
        assert_eq!(id_a, id_b, "VRegId should support Copy");
        assert_eq!(id_a, id_c, "VRegId with same inner value should be equal");
        assert_ne!(id_a, id_d, "VRegId with different inner value should not be equal");
    }

    #[test]
    fn test_vreg_kind_all_variants_are_distinct() {
        // Arrange: all VRegKind variants
        let kinds = [
            VRegKind::Ptr, VRegKind::Vec, VRegKind::Scalar,
            VRegKind::Counter, VRegKind::ByteOffset, VRegKind::Tile, VRegKind::Mask,
        ];

        // Act & Assert: every pair is distinct
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(kinds[i], kinds[j],
                    "VRegKind variants at index {i} and {j} should be distinct");
            }
        }
    }

    #[test]
    fn test_offset_expr_loop_plus_const_zero_offset() {
        // Arrange: a LoopOffset with no additive constant
        let vreg = VRegId(7);

        // Act: when constant is 0, should return plain LoopOffset (no Add wrapper)
        let expr = OffsetExpr::loop_plus_const(vreg, 0);

        // Assert
        assert_eq!(expr, OffsetExpr::LoopOffset(vreg),
            "loop_plus_const with 0 should yield bare LoopOffset");
    }

    #[test]
    fn test_offset_expr_loop_plus_const_nonzero_offset() {
        // Arrange: a LoopOffset with a nonzero additive constant
        let vreg = VRegId(3);

        // Act
        let expr = OffsetExpr::loop_plus_const(vreg, 64);

        // Assert: should be Add(LoopOffset(vreg), Const(64))
        match expr {
            OffsetExpr::Add(left, right) => {
                assert_eq!(*left, OffsetExpr::LoopOffset(vreg));
                assert_eq!(*right, OffsetExpr::Const(64));
            }
            other => panic!("expected Add, got {:?}", other),
        }
    }

    #[test]
    fn test_offset_expr_substitute_loop_offset_nested() {
        // Arrange: nested expression Mul(Add(LoopOffset(5), Const(10)), 4)
        let target = VRegId(5);
        let inner = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(target)),
            Box::new(OffsetExpr::Const(10)),
        );
        let expr = OffsetExpr::Mul(Box::new(inner), 4);

        // Act: substitute LoopOffset(5) -> Const(100)
        let result = expr.substitute_loop_offset(target, 100);

        // Assert: LoopOffset(5) replaced but structure preserved
        let expected = OffsetExpr::Mul(
            Box::new(OffsetExpr::Add(
                Box::new(OffsetExpr::Const(100)),
                Box::new(OffsetExpr::Const(10)),
            )),
            4,
        );
        assert_eq!(result, expected, "substitute should replace only the target LoopOffset");
    }

    #[test]
    fn test_scalar_expr_const_float_precision() {
        // Arrange: special float values
        let zero = ScalarExpr::Const(0.0);
        let neg = ScalarExpr::Const(-1.5);
        let tiny = ScalarExpr::Const(f32::MIN_POSITIVE);

        // Act & Assert: values survive round-trip through pattern match
        match (zero, neg, tiny) {
            (ScalarExpr::Const(z), ScalarExpr::Const(n), ScalarExpr::Const(t)) => {
                assert_eq!(z, 0.0);
                assert_eq!(n, -1.5);
                assert!(t > 0.0, "MIN_POSITIVE should be positive");
                assert!(t < 1e-37, "MIN_POSITIVE should be very small");
            }
            other => panic!("expected all Const, got {:?}", other),
        }
    }

    #[test]
    fn test_fp8_kind_variants_distinct_and_copy() {
        // Arrange: both FP8 format variants
        let e4m3 = Fp8Kind::E4M3;
        let e5m2 = Fp8Kind::E5M2;
        let copy = e4m3; // Copy trait test

        // Assert
        assert_eq!(e4m3, copy, "Fp8Kind should be Copy");
        assert_ne!(e4m3, e5m2, "E4M3 and E5M2 should be distinct variants");
    }

    #[test]
    fn test_tma_swizzle_debug_format() {
        // Arrange: all TmaSwizzle variants
        let variants = [
            TmaSwizzle::None, TmaSwizzle::Swizzle32,
            TmaSwizzle::Swizzle64, TmaSwizzle::Swizzle128,
        ];

        // Act: format via Debug
        let debug_strs: Vec<String> = variants.iter().map(|v| format!("{:?}", v)).collect();

        // Assert: each debug string is unique and non-empty
        for s in &debug_strs {
            assert!(!s.is_empty(), "Debug output should not be empty");
        }
        for i in 0..debug_strs.len() {
            for j in (i + 1)..debug_strs.len() {
                assert_ne!(debug_strs[i], debug_strs[j],
                    "TmaSwizzle Debug outputs should be unique for different variants");
            }
        }
    }

    #[test]
    fn test_epilogue_place_variants_and_equality() {
        // Arrange: both EpiloguePlace variants
        let on_acc = crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators;
        let after_store = crate::compiler::codegen::vm::isa_hook::EpiloguePlace::AfterStore;

        // Act & Assert: inequality and Copy
        assert_ne!(on_acc, after_store);
        let copied = on_acc;
        assert_eq!(on_acc, copied, "EpiloguePlace should be Copy");
    }

    #[test]
    fn test_quant_precision_f32_accumulator_no_narrowing() {
        // Arrange: F32 precision — accumulator is also F32
        let dtype = QuantPrecision::F32;

        // Act
        let acc = dtype.accumulator_dtype();
        let needs_narrow = dtype.needs_narrowing_from(acc);

        // Assert: F32 accumulate -> F32 output needs no narrowing
        assert_eq!(acc, QuantPrecision::F32, "F32 accumulator should be F32");
        assert!(!needs_narrow, "F32->F32 should not need narrowing");
    }

    #[test]
    fn test_quant_precision_bf16_accumulator_widens() {
        // Arrange: BF16 precision — accumulator widens to F32
        let dtype = QuantPrecision::BF16;

        // Act
        let acc = dtype.accumulator_dtype();
        let needs_narrow = dtype.needs_narrowing_from(acc);

        // Assert: BF16 -> F32 accumulator, then narrowing back to BF16
        assert_eq!(acc, QuantPrecision::F32, "BF16 accumulator should widen to F32");
        assert!(needs_narrow, "BF16 should need narrowing from F32 accumulator");
    }

    #[test]
    fn test_simd_width_w128_and_w512_lane_counts() {
        let w128 = SimdWidth::W128;
        let w512 = SimdWidth::W512;

        assert_eq!(w128.f32_lanes(), 4);
        assert_eq!(w128.bytes(), 16);
        assert_eq!(w512.f32_lanes(), 16);
        assert_eq!(w512.bytes(), 64);
    }

    #[test]
    fn test_simd_width_scalable_reports_zero_runtime_lanes() {
        let sve = SimdWidth::Scalable;

        assert_eq!(sve.f32_lanes(), 0);
        assert_eq!(sve.bytes(), 0);
    }

    #[test]
    fn test_sym_dim_concrete_as_concrete_returns_value() {
        let dim = SymDim::Concrete(512);

        assert_eq!(dim.as_concrete(), Some(512));
        assert!(!dim.is_symbolic());
    }

    #[test]
    fn test_sym_dim_symbolic_is_symbolic_true() {
        let dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(2048) };

        assert!(dim.is_symbolic());
        assert_eq!(dim.as_concrete(), None);
    }

    #[test]
    fn test_offset_expr_scalar_vreg_not_substituted_by_loop_offset() {
        let vreg = VRegId(10);
        let loop_vreg = VRegId(20);
        let expr = OffsetExpr::Add(
            Box::new(OffsetExpr::ScalarVReg(vreg)),
            Box::new(OffsetExpr::LoopOffset(loop_vreg)),
        );

        let result = expr.substitute_loop_offset(loop_vreg, 999);

        match &result {
            OffsetExpr::Add(left, right) => {
                assert_eq!(**left, OffsetExpr::ScalarVReg(vreg), "ScalarVReg should pass through unchanged");
                assert_eq!(**right, OffsetExpr::Const(999), "LoopOffset should be substituted");
            }
            other => panic!("expected Add, got {:?}", other),
        }
    }

    #[test]
    fn test_vm_program_alloc_vreg_monotonically_increasing() {
        let mut prog = VmProgram::new();

        let a = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let c = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);

        assert!(a.0 < b.0, "VRegId should increase: {} < {}", a.0, b.0);
        assert!(b.0 < c.0, "VRegId should increase: {} < {}", b.0, c.0);
        assert_eq!(prog.vreg_count(), 3);
    }

    #[test]
    fn test_emit_tile_gemm_produces_tile_lifecycle() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dt = crate::types::DType::F32;

        let result = emit_tile_gemm(&mut prog, width, 4, 4, 4, 16, dt);

        assert!(result.is_ok());
        let instrs: Vec<&VmInstr> = prog.instrs.iter().collect();
        let has_tile_config = instrs.iter().any(|i| matches!(i, VmInstr::TileConfig { .. }));
        let has_tile_release = instrs.iter().any(|i| matches!(i, VmInstr::TileRelease));
        assert!(has_tile_config, "should emit TileConfig");
        assert!(has_tile_release, "should emit TileRelease");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_rejects_zero_k() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(4), 8, 0, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );
        assert!(result.is_err(), "trans_b GEMM with zero K should be rejected");
    }

    #[test]
    fn test_emit_gemm_gpu_pipelined_single_k_tile() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        let result = emit_gemm_gpu_pipelined(
            &mut prog, 16, 16, 16, width, a, b, c,
            16, 16, 16, 16, 16, 16, 2, dtype, false, false,
        );

        assert!(result.is_ok(), "pipelined GPU GEMM with single K tile should succeed: {:?}", result.err());
        let has_smem_alloc = prog.instrs.iter().any(|i| matches!(i, VmInstr::SharedMemAlloc { .. }));
        assert!(has_smem_alloc, "pipelined GEMM should allocate shared memory");
    }

    #[test]
    fn test_emit_gemm_blis_inline_trans_b_produces_instrs() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        let result = emit_gemm_blis_inline(
            &mut prog, 4, 8, 16, width, a, b, c, 4, 2, None, 1, dtype, true,
        );
        assert!(result.is_ok(), "BLIS GEMM with trans_b should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn test_bound_expr_const_equality_and_clone() {
        let a = BoundExpr::Const(42);
        let b = a.clone();
        let c = BoundExpr::Const(42);
        let d = BoundExpr::Const(99);

        assert_eq!(a, b, "cloned BoundExpr::Const should be equal");
        assert_eq!(a, c, "same-value BoundExpr::Const should be equal");
        assert_ne!(a, d, "different-value BoundExpr::Const should not be equal");
    }

    #[test]
    fn test_quant_precision_fp8e4m3_elem_bytes_is_one() {
        let dtype = QuantPrecision::FP8E4M3;

        assert_eq!(dtype.elem_bytes(), 1, "FP8 E4M3 should be 1 byte per element");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: GEMM boundary validation, dtype precision, offset
    // expression substitution, symbolic dimensions, BLIS small-M fallback,
    // GPU tiled BF16 narrowing, BoundExpr variants, SymDim symbolic max
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_emit_gemm_trans_b_inline_rejects_zero_m() {
        // Arrange: trans-B GEMM with zero M dimension
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(0), 8, 16, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: zero M is a CodegenViolation
        assert!(result.is_err(), "trans_b GEMM with zero M should be rejected");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_rejects_zero_n() {
        // Arrange: trans-B GEMM with zero N dimension
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(4), 0, 16, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: zero N is a CodegenViolation
        assert!(result.is_err(), "trans_b GEMM with zero N should be rejected");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_concrete_dims_produces_instrs() {
        // Arrange: valid concrete dimensions for trans-B GEMM
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: M=2, N=8, K=16 with trans_b path
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(2), 8, 16, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: succeeds and produces instructions (VecStore at minimum)
        assert!(result.is_ok(), "trans_b GEMM with valid dims should succeed: {:?}", result.err());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "trans_b GEMM should emit VecStore instructions");
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_bf16_produces_narrowing() {
        // Arrange: GPU tiled GEMM with BF16 — accumulator widens to F32, needs narrowing
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let dtype = QuantPrecision::BF16;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 16, 16, 16, width, a, b, c,
            16, 16, 16, 16, 16, 16, dtype, false,
        );

        // Assert: BF16 should produce VecNarrow to convert F32 accumulator back to BF16
        assert!(result.is_ok(), "GPU tiled GEMM with BF16 should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BF16 GPU tiled GEMM should emit VecNarrow for accumulator writeback");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_symbolic_m_succeeds() {
        // Arrange: symbolic M dimension (common in decode where seq_len varies)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(512) };

        // Act: symbolic M with concrete N=8, K=16
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &m_dim, 8, 16, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: symbolic M should succeed (loop bound from sym_map)
        assert!(result.is_ok(), "symbolic M GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "symbolic M GEMM should produce instructions");
    }

    #[test]
    fn test_bound_expr_symbolic_constructs_and_compares() {
        // Arrange: two SymBound instances with same name
        let sb1 = SymBound { name: "seq_len".into(), max_alloc: 2048 };
        let sb2 = SymBound { name: "seq_len".into(), max_alloc: 2048 };
        let sb3 = SymBound { name: "batch_size".into(), max_alloc: 32 };

        // Act: wrap in BoundExpr
        let be1 = BoundExpr::Symbolic(sb1.clone());
        let be2 = BoundExpr::Symbolic(sb2);
        let be3 = BoundExpr::Symbolic(sb3);

        // Assert: equality semantics
        assert_eq!(be1, be2, "Symbolic with same name and max_alloc should be equal");
        assert_ne!(be1, be3, "Symbolic with different name should not be equal");

        // Assert: Debug is non-empty
        let debug = format!("{:?}", be1);
        assert!(debug.contains("seq_len"), "Debug should contain the name");
    }

    #[test]
    fn test_b_offset_expr_trans_b_structure_is_mul_add() {
        // Arrange: trans-B offset expression parameters
        let prog = &mut VmProgram::new();
        let k_ctr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let j_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k = 64;

        // Act
        let expr = b_offset_expr(k_ctr, k_off, j_off, 0, 4, 32, k, true, 8);

        // Assert: trans_b => Add(Mul(LoopOffset(j_off), k), LoopOffset(k_off))
        match &expr {
            OffsetExpr::Add(left, right) => {
                assert!(
                    matches!(left.as_ref(), OffsetExpr::Mul(..)),
                    "trans-B first term should be Mul(j_off, k)"
                );
                assert!(
                    matches!(right.as_ref(), OffsetExpr::LoopOffset(..)),
                    "trans-B second term should be LoopOffset(k_off)"
                );
            }
            other => panic!("expected Add for trans-B, got {:?}", other),
        }
    }

    #[test]
    fn test_quant_precision_fp8e5m2_elem_bytes_and_accumulator() {
        // Arrange: FP8 E5M2 (alternate FP8 format)
        let dtype = QuantPrecision::FP8E5M2;

        // Act
        let elem_bytes = dtype.elem_bytes();
        let acc = dtype.accumulator_dtype();
        let needs_narrow = dtype.needs_narrowing_from(acc);

        // Assert: FP8 is 1 byte per element, accumulator widens to F32, needs narrowing
        assert_eq!(elem_bytes, 1, "FP8 E5M2 should be 1 byte per element");
        assert_eq!(acc, QuantPrecision::F32, "FP8 E5M2 accumulator should widen to F32");
        assert!(needs_narrow, "FP8 E5M2 should need narrowing from F32 accumulator");
    }

    #[test]
    fn test_emit_gemm_blis_inline_small_m_clamps_mr() {
        // Arrange: BLIS GEMM where M < mr — should clamp mr_actual down
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=2, mr=4 — mr_actual should clamp to 2, not panic
        let result = emit_gemm_blis_inline(
            &mut prog, 2, 16, 16, width, a, b, c, 4, 2, None, 1, QuantPrecision::F32, false,
        );

        // Assert: should succeed despite M < mr
        assert!(result.is_ok(), "BLIS GEMM with M < mr should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should still produce instructions");
    }

    #[test]
    fn test_offset_expr_substitute_non_target_loop_offset_unchanged() {
        // Arrange: two different LoopOffset VRegs — only one should be substituted
        let target = VRegId(5);
        let bystander = VRegId(99);
        let expr = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(target)),
            Box::new(OffsetExpr::LoopOffset(bystander)),
        );

        // Act: substitute only target
        let result = expr.substitute_loop_offset(target, 200);

        // Assert: target replaced, bystander unchanged
        match result {
            OffsetExpr::Add(left, right) => {
                assert_eq!(*left, OffsetExpr::Const(200), "target LoopOffset should be substituted");
                assert_eq!(*right, OffsetExpr::LoopOffset(bystander), "non-target LoopOffset should pass through");
            }
            other => panic!("expected Add, got {:?}", other),
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: GPU pipelined TMA, MMA on smem, async load SM80,
    // BLIS unroll factor, GPU partial tiles, OffsetExpr deep nesting,
    // ReduceOp variants, SymDimSlotMap default, epilogue telemetry flag,
    // trans-B symbolic M
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_emit_gemm_gpu_pipelined_tma_emits_barrier_init() {
        // Arrange: GPU pipelined GEMM with TMA enabled (SM90+ path)
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: use_tma=true triggers TmaDescriptorInit + BarrierInit
        let result = emit_gemm_gpu_pipelined(
            &mut prog, 16, 16, 32, width, a, b, c,
            16, 16, 16, 16, 16, 16, 2, QuantPrecision::F32, false, true,
        );

        // Assert
        assert!(result.is_ok(), "TMA pipelined GEMM should succeed: {:?}", result.err());
        let has_tma_desc = prog.instrs.iter().any(|i| matches!(i, VmInstr::TmaDescriptorInit { .. }));
        let has_barrier = prog.instrs.iter().any(|i| matches!(i, VmInstr::BarrierInit { .. }));
        assert!(has_tma_desc, "TMA path should emit TmaDescriptorInit");
        assert!(has_barrier, "TMA path should emit BarrierInit");
    }

    #[test]
    fn test_emit_mma_on_smem_produces_fma_instructions() {
        // Arrange: MMA on shared memory accumulates into registers
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let accs: Vec<VRegId> = (0..4).map(|_| prog.alloc_vreg(VRegKind::Vec, width)).collect();

        // Act: mi=2, nj=2, kk=16, warp_m=2, warp_n=2, mma_k=16
        emit_mma_on_smem(
            &mut prog, &accs, "smem_a", "smem_b",
            0, 0, 0, 2, 2, 16,
            16, 16, 16, 2, 2, 16,
            16, 16, 4, QuantPrecision::F32, QuantPrecision::F32, width,
            128, 128,
        );

        // Assert: should produce SharedMemLoad + Fma instructions
        let has_smem_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::SharedMemLoad { .. }));
        let has_fma = prog.instrs.iter().any(|i| matches!(i, VmInstr::Fma { .. }));
        assert!(has_smem_load, "MMA on smem should emit SharedMemLoad for A/B fragments");
        assert!(has_fma, "MMA on smem should emit Fma to accumulate");
    }

    #[test]
    fn test_emit_async_load_tile_sm80_produces_shared_mem_async_store() {
        // Arrange: SM80 path (use_tma=false) for async tile loading
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: load a 2x2 tile via SM80 per-row path
        emit_async_load_tile(
            &mut prog, "smem_a", "smem_b",
            a_ptr, b_ptr,
            0, 0, 0, // i_cta, j_cta, k_off
            2, 2, 2, // mi, nj, kk
            4, 4, 4, // cta_m, cta_n, cta_k
            8, 8, 4, // n, k, elem
            QuantPrecision::F32, width,
            32, 32, // smem_a_row_stride, smem_b_row_stride
            false, // use_tma=false -> SM80 path
        );

        // Assert: SM80 path should emit VecLoad + SharedMemAsyncStore per row
        let has_async_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::SharedMemAsyncStore { .. }));
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_vec_load, "SM80 async load should emit VecLoad from global");
        assert!(has_async_store, "SM80 async load should emit SharedMemAsyncStore to smem");
    }

    #[test]
    fn test_emit_gemm_blis_inline_high_unroll_factor() {
        // Arrange: BLIS GEMM with unroll_factor=4 (K-dimension unrolling)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: K=32 with unroll_factor=4 -> 8 k iterations instead of 32
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 8, 32, width, a, b, c, 4, 2, None, 4, QuantPrecision::F32, false,
        );

        // Assert: should succeed with fewer loop iterations due to unrolling
        assert!(result.is_ok(), "BLIS GEMM with unroll_factor=4 should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_partial_last_tiles() {
        // Arrange: non-divisible dimensions — last CTA/warp tiles are partial
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=20 with cta_m=16 -> last tile has mi=4 (partial)
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 20, 20, 20, width, a, b, c,
            16, 16, 16, 16, 16, 16, QuantPrecision::F32, false,
        );

        // Assert: partial tiles should be handled correctly (clamped mi/nj)
        assert!(result.is_ok(), "GPU tiled GEMM with partial tiles should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn test_offset_expr_deeply_nested_substitute() {
        // Arrange: 3-level nested expression
        let target = VRegId(1);
        let inner = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(target)),
            Box::new(OffsetExpr::Const(10)),
        );
        let mid = OffsetExpr::Mul(Box::new(inner), 8);
        let outer = OffsetExpr::Add(
            Box::new(mid),
            Box::new(OffsetExpr::Const(256)),
        );

        // Act: substitute LoopOffset(1) -> Const(50)
        let result = outer.substitute_loop_offset(target, 50);

        // Assert: substitution recurses through all nesting levels
        match &result {
            OffsetExpr::Add(left, right) => {
                assert_eq!(**right, OffsetExpr::Const(256), "outer Const(256) should be unchanged");
                match left.as_ref() {
                    OffsetExpr::Mul(inner_add, scale) => {
                        assert_eq!(*scale, 8);
                        match inner_add.as_ref() {
                            OffsetExpr::Add(substituted, c) => {
                                assert_eq!(**substituted, OffsetExpr::Const(50));
                                assert_eq!(**c, OffsetExpr::Const(10));
                            }
                            other => panic!("expected inner Add, got {:?}", other),
                        }
                    }
                    other => panic!("expected Mul, got {:?}", other),
                }
            }
            other => panic!("expected outer Add, got {:?}", other),
        }
    }

    #[test]
    fn test_reduce_op_variants_distinct_and_copy() {
        // Arrange: ReduceOp enum variants
        let sum = ReduceOp::Sum;
        let max = ReduceOp::Max;
        let min = ReduceOp::Min;

        // Act & Assert: all distinct
        assert_ne!(sum, max, "Sum and Max should be distinct");
        assert_ne!(sum, min, "Sum and Min should be distinct");
        assert_ne!(max, min, "Max and Min should be distinct");

        // Copy trait
        let sum_copy = sum;
        assert_eq!(sum, sum_copy, "ReduceOp should be Copy");
    }

    #[test]
    fn test_sym_dim_slot_map_mega_kernel_abi_has_seq_len_slot() {
        // Arrange: default ABI slot map (standard calling convention slots)
        let map = SymDimSlotMap::mega_kernel_abi();

        // Act: look up "seq_len" — the most fundamental symbolic dimension
        let dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(512) };
        let bound = map.to_bound(&dim);

        // Assert: should produce a valid bound (any variant is acceptable)
        match &bound {
            BoundExpr::Symbolic(sb) => {
                assert_eq!(sb.name, "seq_len", "bound name should match dimension name");
                assert!(sb.max_alloc > 0, "max_alloc should be positive");
            }
            BoundExpr::Const(v) => {
                assert_eq!(*v, 512, "fallback Const should use max_value");
            }
            _ => {
                // Runtime / DynamicVReg / DynamicVRegPlusOne — valid for symbolic dims
            }
        }
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_telemetry_produces_hreduce() {
        // Arrange: GEMM with telemetry enabled (enable_row_stats=true)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 8, 8, width, a, b, c, &[], &sym_map,
            true, // enable_row_stats = true
            None, QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: telemetry should emit HReduce for row statistics
        assert!(result.is_ok(), "GEMM with telemetry should succeed: {:?}", result.err());
        let has_hreduce = prog.instrs.iter().any(|i| matches!(i, VmInstr::HReduce { .. }));
        assert!(has_hreduce, "GEMM with enable_row_stats should emit HReduce");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_symbolic_m_succeeds() {
        // Arrange: trans-B GEMM with symbolic M (decode path)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(128) };

        // Act
        let result = emit_gemm_trans_b_inline(
            &mut prog, &m_dim, 8, 16, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: symbolic M should work for trans-B path too
        assert!(result.is_ok(), "trans-B GEMM with symbolic M should succeed: {:?}", result.err());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "trans-B GEMM with symbolic M should emit VecStore");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: trans-B K-tail, trans-B epilogue, pipelined BF16,
    // epilogue trans-B routing, N<lanes tail-only, BLIS with pack_map,
    // pipelined multi-K-tile, AfterStore epilogue, TMA async load path,
    // GPU tiled BF16 partial tiles narrowing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_emit_gemm_trans_b_inline_k_tail_emits_hreduce() {
        // Arrange: K=13 with W256 (lanes=8) -> k_vecs=1, k_tail=5
        // The k_tail loop produces scalar elements that must be HReduce'd with the
        // vector result before store.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: K=13 not divisible by lanes=8, so k_tail=5
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(1), 4, 13, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: k_tail path should emit HReduce to collapse vector accumulator
        assert!(result.is_ok(), "trans-B GEMM with K-tail should succeed: {:?}", result.err());
        let has_hreduce = prog.instrs.iter().any(|i| matches!(i, VmInstr::HReduce { .. }));
        assert!(has_hreduce, "trans-B GEMM with k_tail>0 should emit HReduce");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_with_epilogue_lowering() {
        // Arrange: trans-B GEMM with a non-empty epilogue (AddConst trace)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let epilogue: Vec<TraceOp> = vec![
            TraceOp::Input(0),
            TraceOp::Const(1.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];

        // Act
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(1), 4, 16, width, a, b, c, &epilogue, &sym_map, None, QuantPrecision::F32,
        );

        // Assert: epilogue should be lowered without error
        assert!(result.is_ok(), "trans-B GEMM with epilogue should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should produce instructions with epilogue");
    }

    #[test]
    fn test_emit_gemm_gpu_pipelined_bf16_produces_narrowing() {
        // Arrange: GPU pipelined GEMM with BF16 — accumulator is F32, writeback needs narrowing
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: BF16 with 2-stage pipeline, single K-tile (K=cta_k=16)
        let result = emit_gemm_gpu_pipelined(
            &mut prog, 16, 16, 16, width, a, b, c,
            16, 16, 16, 16, 16, 16, 2, QuantPrecision::BF16, false, false,
        );

        // Assert: BF16 writeback should emit VecNarrow (F32 acc -> BF16 store)
        assert!(result.is_ok(), "pipelined GPU GEMM BF16 should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BF16 pipelined GEMM should emit VecNarrow for writeback");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_routes_trans_b() {
        // Arrange: call emit_gemm_inline_with_epilogue with trans_b=true,
        // which internally routes to emit_gemm_trans_b_inline.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: trans_b=true triggers the trans-B routing path
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 8, 16, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::F32, true, // trans_b = true
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: trans-B routing should succeed
        assert!(result.is_ok(), "epilogue GEMM with trans_b should route and succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should produce instructions via trans-B path");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_n_smaller_than_lanes() {
        // Arrange: N=3 with W256 (lanes=8) -> n_vecs=0, n_tail=3
        // All output is via scalar tail path; no vectorized j-loop.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: N=3 < lanes=8, only scalar tail path runs
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(1), 3, 4, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: tail-only path should succeed and emit VecStore
        assert!(result.is_ok(), "GEMM with N < lanes should succeed: {:?}", result.err());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "tail-only GEMM should emit VecStore for scalar elements");
    }

    #[test]
    fn test_emit_gemm_blis_inline_with_pack_map() {
        // Arrange: BLIS GEMM with a PackMap that provides a custom k-stride for B
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let pack_map = crate::compiler::pack_map::PackMap::Identity;

        // Act: BLIS with PackMap (non-None) — should use pack_map.blis_k_stride_bytes
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 8, 16, width, a, b, c, 4, 2, Some(&pack_map), 1, QuantPrecision::F32, false,
        );

        // Assert: should succeed; PackMap path is exercised
        assert!(result.is_ok(), "BLIS GEMM with PackMap should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "BLIS GEMM with PackMap should produce instructions");
    }

    #[test]
    fn test_emit_gemm_gpu_pipelined_multiple_k_tiles() {
        // Arrange: K=32 with cta_k=16 -> 2 K-tiles, steady-state pipeline engages
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: K=32 > cta_k=16, triggers multi-K-tile steady-state loop
        let result = emit_gemm_gpu_pipelined(
            &mut prog, 16, 16, 32, width, a, b, c,
            16, 16, 16, 16, 16, 16, 2, QuantPrecision::F32, false, false,
        );

        // Assert: multi-tile pipeline should succeed with AsyncWait for overlap
        assert!(result.is_ok(), "multi-K-tile pipelined GEMM should succeed: {:?}", result.err());
        let has_async_wait = prog.instrs.iter().any(|i| matches!(i, VmInstr::AsyncWait { .. }));
        assert!(has_async_wait, "multi-K-tile pipeline should emit AsyncWait for stage synchronization");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_after_store_skips_inline_epi() {
        // Arrange: EpiloguePlace::AfterStore — epilogue NOT applied on accumulators
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let epilogue: Vec<TraceOp> = vec![
            TraceOp::Input(0),
            TraceOp::Const(1.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];

        // Act: AfterStore — do_epilogue_inline is false
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 8, 8, width, a, b, c, &epilogue, &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::AfterStore,
        );

        // Assert: should succeed; epilogue is deferred to AfterStore phase
        assert!(result.is_ok(), "AfterStore GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn test_emit_async_load_tile_tma_path_emits_tma_2d_copy() {
        // Arrange: TMA path (use_tma=true) for async tile loading
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: TMA path — should emit GprLoadImm for coords + Tma2DCopy per tile
        emit_async_load_tile(
            &mut prog, "smem_a", "smem_b",
            a_ptr, b_ptr,
            0, 0, 0, // i_cta, j_cta, k_off
            2, 2, 2, // mi, nj, kk
            4, 4, 4, // cta_m, cta_n, cta_k
            8, 8, 4, // n, k, elem
            QuantPrecision::F32, width,
            32, 32, // smem_a_row_stride, smem_b_row_stride
            true, // use_tma=true -> TMA path
        );

        // Assert: TMA path emits Tma2DCopy (2 for A and B) and GprLoadImm for coordinates
        let has_tma_copy = prog.instrs.iter().any(|i| matches!(i, VmInstr::Tma2DCopy { .. }));
        let has_gpr_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::GprLoadImm { .. }));
        assert!(has_gpr_load, "TMA path should emit GprLoadImm for tile coordinates");
        assert!(has_tma_copy, "TMA path should emit Tma2DCopy for async tile load");
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_bf16_partial_tiles_narrowing() {
        // Arrange: non-divisible M/N with BF16 — partial tiles + narrowing on writeback
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=10 with cta_m=8 -> last tile has mi=2; BF16 needs narrowing
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 10, 10, 16, width, a, b, c,
            8, 8, 16, 4, 4, 16, QuantPrecision::BF16, false,
        );

        // Assert: partial tiles with BF16 should produce VecNarrow on writeback
        assert!(result.is_ok(), "GPU tiled BF16 partial tiles should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BF16 GPU tiled with partial tiles should emit VecNarrow");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: BLIS N<nr*lanes clamp, trans-B K-exact-divide,
    // decode M=1 GEMM, GPU tiled exact tiles, MMA partial warp,
    // async load mi=0 edge, b_offset trans-B k=1, pipelined depth=3,
    // BF16 n_vecs tail, BLIS K<unroll_factor clamp
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_emit_gemm_blis_inline_n_smaller_than_nr_times_lanes() {
        // Arrange: N=4 with nr=2, lanes=8 -> nr_actual clamps to 1 (min of 2 and ceil(4/8)=1)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256; // lanes=8
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: N=4 < nr*lanes=16 -> nr_actual clamped down
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 4, 16, width, a, b, c, 4, 2, None, 1, QuantPrecision::F32, false,
        );

        // Assert: should succeed with clamped nr_actual, no out-of-bounds access
        assert!(result.is_ok(), "BLIS GEMM with N < nr*lanes should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should produce instructions with clamped nr");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_k_exact_divide_by_lanes_no_tail() {
        // Arrange: K=16 with W256 (lanes=8) -> k_vecs=2, k_tail=0
        // No scalar tail loop needed — only vectorized K loop runs.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: K=16 / lanes=8 = 2 exactly, no k_tail
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(1), 4, 16, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: vector-only path should emit VecLoad (not scalar Broadcast for tail)
        assert!(result.is_ok(), "trans-B GEMM with exact K/lanes should succeed: {:?}", result.err());
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_vec_load, "should emit VecLoad for vectorized K loop");
        // k_tail=0: no scalar Add needed between vector and tail accumulator
        let broadcast_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Broadcast { .. })).count();
        // k_tail=0 means we still have Broadcast for zero-init, but no per-tail-element Broadcast
        assert!(broadcast_count > 0, "should have at least the accumulator zero-init Broadcast");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_decode_m1_succeeds() {
        // Arrange: decode-phase GEMM with M=1 (single token, most common decode case)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: M=1 single-row decode GEMM
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(1), 16, 32, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: single-row GEMM should succeed with valid instruction stream
        assert!(result.is_ok(), "M=1 decode GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "M=1 GEMM should emit VecStore for result writeback");
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_inline_exact_divisible_tiles() {
        // Arrange: M/N/K exactly divisible by CTA/warp dimensions — no partial tiles
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=32, cta_m=16, warp_m=8, mma_k=16 — all divide evenly
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 32, 32, 32, width, a, b, c,
            16, 16, 16, 8, 8, 16, QuantPrecision::F32, false,
        );

        // Assert: exact division means mi/wi/wj always equal full tile size
        assert!(result.is_ok(), "exact-divisible GPU tiled GEMM should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
        // With 2x2 CTAs and 2x2 warps per CTA, there should be multiple FMA ops
        let fma_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Fma { .. })).count();
        assert!(fma_count > 1, "exact-divisible GPU GEMM should have multiple FMA instructions, got {}", fma_count);
    }

    #[test]
    fn test_emit_mma_on_smem_partial_warp_tile() {
        // Arrange: CTA tile mi=3, nj=3 but warp_m=4, warp_n=4
        // warp tiles get clamped: wi=min(4,3)=3, wj=min(4,3)=3
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let accs: Vec<VRegId> = (0..16).map(|_| prog.alloc_vreg(VRegKind::Vec, width)).collect();

        // Act: mi=3, nj=3 with warp_m=4, warp_n=4 -> clamped to 3x3
        emit_mma_on_smem(
            &mut prog, &accs, "smem_a", "smem_b",
            0, 0, 0, 3, 3, 16,
            4, 4, 16, 4, 4, 16,
            4, 16, 4, QuantPrecision::F32, QuantPrecision::F32, width,
            64, 64,
        );

        // Assert: partial warp should still emit SharedMemLoad + Fma
        let smem_load_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::SharedMemLoad { .. })).count();
        let fma_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Fma { .. })).count();
        assert!(smem_load_count > 0, "partial warp MMA should emit SharedMemLoad");
        assert!(fma_count > 0, "partial warp MMA should emit Fma");
        // 3x3 = 9 accumulator entries, each needing a pair of loads + FMA
        assert!(fma_count >= 9, "partial 3x3 warp should have at least 9 FMAs, got {}", fma_count);
    }

    #[test]
    fn test_emit_async_load_tile_sm80_mi_zero_skips_a_loads() {
        // Arrange: SM80 async load with mi=0 (degenerate CTA tile — no A rows)
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: mi=0 means no A rows to load; kk=2 means B loads still happen
        emit_async_load_tile(
            &mut prog, "smem_a", "smem_b",
            a_ptr, b_ptr,
            0, 0, 0, // i_cta, j_cta, k_off
            0, 2, 2, // mi=0, nj=2, kk=2
            4, 4, 4, // cta_m, cta_n, cta_k
            8, 8, 4, // n, k, elem
            QuantPrecision::F32, width,
            32, 32,
            false, // use_tma=false
        );

        // Assert: no SharedMemAsyncStore for smem_a (mi=0), but B loads present
        let smem_a_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::SharedMemAsyncStore { name, .. } if name == "smem_a")
        }).count();
        let smem_b_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::SharedMemAsyncStore { name, .. } if name == "smem_b")
        }).count();
        assert_eq!(smem_a_stores, 0, "mi=0 should produce zero smem_a async stores");
        assert!(smem_b_stores > 0, "kk=2 should produce smem_b async stores");
    }

    #[test]
    fn test_b_offset_expr_trans_b_with_k_equals_one() {
        // Arrange: trans-B with minimal K=1 (edge case for inner dimension)
        let prog = &mut VmProgram::new();
        let k_ctr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let j_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: trans_B with k=1
        let expr = b_offset_expr(k_ctr, k_off, j_off, 0, 4, 8, 1, true, 8);

        // Assert: offset = j_off * 1 + k_off = j_off + k_off
        // The Mul factor should be 1 (k=1), which is still Mul(LoopOffset(j_off), 1)
        match &expr {
            OffsetExpr::Add(mul_term, loop_term) => {
                match mul_term.as_ref() {
                    OffsetExpr::Mul(inner, factor) => {
                        assert_eq!(*factor, 1, "trans-B with k=1 should have Mul factor=1");
                        assert!(matches!(inner.as_ref(), OffsetExpr::LoopOffset(_)));
                    }
                    other => panic!("expected Mul for trans-B first term, got {:?}", other),
                }
                assert!(matches!(loop_term.as_ref(), OffsetExpr::LoopOffset(_)));
            }
            other => panic!("expected Add for trans-B k=1, got {:?}", other),
        }
    }

    #[test]
    fn test_emit_gemm_gpu_pipelined_depth_three_stages() {
        // Arrange: 3-stage pipeline (pipeline_depth=3) — triple-buffered smem
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: pipeline_depth=3 with K=48, cta_k=16 -> 3 K-tiles across 3 stages
        let result = emit_gemm_gpu_pipelined(
            &mut prog, 16, 16, 48, width, a, b, c,
            16, 16, 16, 16, 16, 16, 3, QuantPrecision::F32, false, false,
        );

        // Assert: 3 stages should allocate 6 SharedMemAlloc (3 for A + 3 for B)
        assert!(result.is_ok(), "3-stage pipeline GEMM should succeed: {:?}", result.err());
        let smem_alloc_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::SharedMemAlloc { .. })).count();
        assert_eq!(smem_alloc_count, 6, "3-stage pipeline should allocate 6 shared memory buffers (3A+3B), got {}", smem_alloc_count);
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_bf16_n_vecs_with_narrowing() {
        // Arrange: BF16 GEMM with n_vecs > 0 — vectorized path emits VecNarrow for writeback
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256; // lanes=8
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: BF16 with N=16 -> n_vecs=2, n_tail=0; BF16 accumulator widens to F32
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 16, 8, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::BF16, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: BF16 with vectorized output should emit VecNarrow
        assert!(result.is_ok(), "BF16 vectorized GEMM should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BF16 vectorized GEMM should emit VecNarrow for F32->BF16 writeback");
    }

    #[test]
    fn test_emit_gemm_blis_inline_k_smaller_than_unroll_factor_clamps() {
        // Arrange: BLIS GEMM where K=3 < unroll_factor=8
        // k_unroll should clamp to k (=3), and k_iters=1
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: K=3 with unroll_factor=8 -> k_unroll clamped to min(8,3)=3
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 8, 3, width, a, b, c, 4, 2, None, 8, QuantPrecision::F32, false,
        );

        // Assert: K < unroll_factor should not panic; single K iteration with 3 micro-steps
        assert!(result.is_ok(), "BLIS GEMM with K < unroll_factor should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should produce instructions with clamped k_unroll");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: OffsetExpr::Const substitute identity, non-square
    // tile GEMM, BLIS mr=1, BF16 scalar tail, TMA multi-K pipeline,
    // MMA single K-step, SM80 nonzero CTA offsets, minimal trans-B GEMV,
    // GPU single-CTA tile, OffsetExpr Clone deep equality
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_offset_expr_const_substitute_returns_self() {
        // Arrange: pure Const expression — no LoopOffset to substitute
        let expr = OffsetExpr::Const(256);
        let target = VRegId(42);

        // Act: substitute on a Const is a no-op
        let result = expr.substitute_loop_offset(target, 999);

        // Assert: Const has no LoopOffset children, returns identical value
        assert_eq!(result, OffsetExpr::Const(256),
            "Const substitute should return the same Const");
    }

    #[test]
    fn test_emit_tile_gemm_non_square_tiles_produces_lifecycle() {
        // Arrange: non-square tile dimensions (rows=2, cols=8)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dt = crate::types::DType::F32;

        // Act
        let result = emit_tile_gemm(&mut prog, width, 2, 8, 4, 32, dt);

        // Assert: should still emit TileConfig + TileRelease lifecycle
        assert!(result.is_ok());
        let has_config = prog.instrs.iter().any(|i| matches!(i, VmInstr::TileConfig { .. }));
        let has_release = prog.instrs.iter().any(|i| matches!(i, VmInstr::TileRelease));
        assert!(has_config, "non-square tiles should emit TileConfig");
        assert!(has_release, "non-square tiles should emit TileRelease");
    }

    #[test]
    fn test_emit_gemm_blis_inline_mr_equals_one_succeeds() {
        // Arrange: BLIS with mr=1 (single-row microkernel)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: mr=1 means each M-block processes exactly 1 row
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 16, 8, width, a, b, c, 1, 2, None, 1, QuantPrecision::F32, false,
        );

        // Assert: mr=1 should succeed with 1×nr accumulators per block
        assert!(result.is_ok(), "BLIS mr=1 should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_bf16_with_scalar_tail() {
        // Arrange: BF16 GEMM with N=12, W256 (lanes=8) -> n_vecs=1, n_tail=4
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: BF16 with N=12 produces both vectorized + scalar tail paths
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 12, 8, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::BF16, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: BF16 tail path should emit VecNarrow for both vectorized and scalar stores
        assert!(result.is_ok(), "BF16 GEMM with tail should succeed: {:?}", result.err());
        let narrow_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecNarrow { .. })).count();
        assert!(narrow_count >= 1,
            "BF16 GEMM with tail should emit at least 1 VecNarrow, got {}", narrow_count);
    }

    #[test]
    fn test_emit_gemm_gpu_pipelined_tma_with_multi_k_tiles() {
        // Arrange: TMA pipelined with K=32, cta_k=16 -> 2 K-tiles in steady state
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: TMA + multi-K-tile triggers steady-state compute/load overlap
        let result = emit_gemm_gpu_pipelined(
            &mut prog, 16, 16, 32, width, a, b, c,
            16, 16, 16, 16, 16, 16, 2, QuantPrecision::F32, false, true,
        );

        // Assert: TMA steady-state should emit WarpBarrierWait for synchronization
        assert!(result.is_ok(), "TMA multi-K pipelined should succeed: {:?}", result.err());
        let has_barrier_wait = prog.instrs.iter().any(|i| matches!(i, VmInstr::WarpBarrierWait { .. }));
        assert!(has_barrier_wait, "TMA multi-K pipeline should emit WarpBarrierWait");
    }

    #[test]
    fn test_emit_mma_on_smem_single_k_step() {
        // Arrange: MMA where kk=mma_k=16 — exactly one inner K iteration
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let accs: Vec<VRegId> = (0..4).map(|_| prog.alloc_vreg(VRegKind::Vec, width)).collect();

        // Act: mi=2, nj=2, kk=16, mma_k=16 -> single inner K step
        emit_mma_on_smem(
            &mut prog, &accs, "smem_a", "smem_b",
            0, 0, 0, 2, 2, 16,
            2, 2, 16, 2, 2, 16,
            2, 16, 4, QuantPrecision::F32, QuantPrecision::F32, width,
            64, 64,
        );

        // Assert: exactly 2*2=4 FMA (one per accumulator per single K step)
        let fma_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Fma { .. })).count();
        assert_eq!(fma_count, 4,
            "single K-step MMA should have exactly 4 FMAs (2×2), got {}", fma_count);
    }

    #[test]
    fn test_emit_async_load_tile_sm80_nonzero_cta_offsets() {
        // Arrange: SM80 async load with nonzero CTA offsets
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: i_cta=4, j_cta=8, k_off=16 with mi=2, kk=2
        emit_async_load_tile(
            &mut prog, "smem_a", "smem_b",
            a_ptr, b_ptr,
            4, 8, 16, // i_cta=4, j_cta=8, k_off=16
            2, 2, 2, // mi=2, nj=2, kk=2
            8, 8, 8, // cta_m, cta_n, cta_k
            16, 32, 4, // n, k, elem
            QuantPrecision::F32, width,
            128, 128,
            false, // SM80 path
        );

        // Assert: nonzero offsets should not panic; VecLoad offsets incorporate i_cta/k_off
        let vec_load_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecLoad { .. })).count();
        assert!(vec_load_count > 0, "SM80 with nonzero offsets should emit VecLoad instructions");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_m1_n1_k1_minimal() {
        // Arrange: absolute minimum trans-B GEMV: M=1, N=1, K=1
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: 1×1×1 GEMV (single dot product)
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(1), 1, 1, width, a, b, c, &[], &sym_map, None, QuantPrecision::F32,
        );

        // Assert: minimal GEMV should succeed with at least a store
        assert!(result.is_ok(), "1×1×1 trans-B GEMV should succeed: {:?}", result.err());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "minimal trans-B GEMV should emit VecStore");
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_single_cta_tile() {
        // Arrange: M=cta_m, N=cta_n — exactly one CTA tile, no M/N loop iterations
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=16, cta_m=16, N=16, cta_n=16 — single CTA covers entire output
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 16, 16, 32, width, a, b, c,
            16, 16, 16, 8, 8, 16, QuantPrecision::F32, false,
        );

        // Assert: single-CTA should succeed with no partial CTA tiles
        assert!(result.is_ok(), "single-CTA GPU tiled should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
    }

    #[test]
    fn test_offset_expr_clone_preserves_deeply_nested_structure() {
        // Arrange: deeply nested OffsetExpr
        let vreg = VRegId(7);
        let inner = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(vreg)),
            Box::new(OffsetExpr::Const(32)),
        );
        let mid = OffsetExpr::Mul(Box::new(inner), 4);
        let outer = OffsetExpr::Add(
            Box::new(mid),
            Box::new(OffsetExpr::ScalarVReg(VRegId(99))),
        );

        // Act: clone the entire structure
        let cloned = outer.clone();

        // Assert: cloned structure is deeply equal
        assert_eq!(outer, cloned, "deeply nested OffsetExpr clone should be equal");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: SIMD width instruction count scaling, BF16 vs F32
    // instruction count, b_offset_expr normal Mul factor, BLIS trans_B
    // BF16 narrowing, symbolic M with seq_bound_override, tile GEMM BF16,
    // GPU decode M=1, epilogue N equals lanes, BLIS large M multi-block,
    // pipelined K smaller than cta_k
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_emit_gemm_inline_with_epilogue_w128_vs_w256_different_lane_widths() {
        // Arrange: same GEMM dimensions with W128 vs W256
        let mut prog_w128 = VmProgram::new();
        let mut prog_w256 = VmProgram::new();
        let width_128 = SimdWidth::W128;
        let width_256 = SimdWidth::W256;
        let a1 = prog_w128.alloc_vreg(VRegKind::Ptr, width_128);
        let b1 = prog_w128.alloc_vreg(VRegKind::Ptr, width_128);
        let c1 = prog_w128.alloc_vreg(VRegKind::Ptr, width_128);
        let a2 = prog_w256.alloc_vreg(VRegKind::Ptr, width_256);
        let b2 = prog_w256.alloc_vreg(VRegKind::Ptr, width_256);
        let c2 = prog_w256.alloc_vreg(VRegKind::Ptr, width_256);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: M=2, N=16, K=8 with both widths
        let r128 = emit_gemm_inline_with_epilogue(
            &mut prog_w128, &SymDim::Concrete(2), 16, 8, width_128, a1, b1, c1, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );
        let r256 = emit_gemm_inline_with_epilogue(
            &mut prog_w256, &SymDim::Concrete(2), 16, 8, width_256, a2, b2, c2, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: both succeed; W128 VecStore instructions carry W128 width, W256 carry W256
        assert!(r128.is_ok(), "W128 GEMM should succeed: {:?}", r128.err());
        assert!(r256.is_ok(), "W256 GEMM should succeed: {:?}", r256.err());
        let w128_stores = prog_w128.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W128, .. })
        }).count();
        let w256_stores = prog_w256.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W256, .. })
        }).count();
        assert!(w128_stores > 0, "W128 GEMM should emit VecStore with W128 width");
        assert!(w256_stores > 0, "W256 GEMM should emit VecStore with W256 width");
        // Verify no cross-width contamination: W128 program has no W256 stores, vice versa
        let w128_cross = prog_w128.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W256, .. })
        }).count();
        let w256_cross = prog_w256.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W128, .. })
        }).count();
        assert_eq!(w128_cross, 0, "W128 GEMM should not emit W256 VecStore");
        assert_eq!(w256_cross, 0, "W256 GEMM should not emit W128 VecStore");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_bf16_produces_more_instrs_than_f32() {
        // Arrange: same GEMM with BF16 vs F32 — BF16 needs VecNarrow for writeback
        let mut prog_f32 = VmProgram::new();
        let mut prog_bf16 = VmProgram::new();
        let width = SimdWidth::W256;
        let a1 = prog_f32.alloc_vreg(VRegKind::Ptr, width);
        let b1 = prog_f32.alloc_vreg(VRegKind::Ptr, width);
        let c1 = prog_f32.alloc_vreg(VRegKind::Ptr, width);
        let a2 = prog_bf16.alloc_vreg(VRegKind::Ptr, width);
        let b2 = prog_bf16.alloc_vreg(VRegKind::Ptr, width);
        let c2 = prog_bf16.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: M=2, N=8, K=8
        let r_f32 = emit_gemm_inline_with_epilogue(
            &mut prog_f32, &SymDim::Concrete(2), 8, 8, width, a1, b1, c1, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );
        let r_bf16 = emit_gemm_inline_with_epilogue(
            &mut prog_bf16, &SymDim::Concrete(2), 8, 8, width, a2, b2, c2, &[], &sym_map, false, None,
            QuantPrecision::BF16, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: both succeed; BF16 has extra VecNarrow instructions
        assert!(r_f32.is_ok(), "F32 GEMM should succeed: {:?}", r_f32.err());
        assert!(r_bf16.is_ok(), "BF16 GEMM should succeed: {:?}", r_bf16.err());
        let bf16_narrow_count = prog_bf16.instrs.iter().filter(|i| matches!(i, VmInstr::VecNarrow { .. })).count();
        assert!(bf16_narrow_count > 0, "BF16 GEMM should emit VecNarrow instructions");
        let f32_narrow_count = prog_f32.instrs.iter().filter(|i| matches!(i, VmInstr::VecNarrow { .. })).count();
        assert_eq!(f32_narrow_count, 0, "F32 GEMM should not emit VecNarrow");
    }

    #[test]
    fn test_b_offset_expr_normal_layout_mul_factor_equals_row_stride() {
        // Arrange: normal (non-trans) B layout with specific b_row_stride
        let prog = &mut VmProgram::new();
        let k_ctr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let j_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let b_row_stride = 512 * 4; // n=512, f32 -> 2048 bytes per row
        let n = 512;
        let k = 256;

        // Act: normal layout (trans_b=false)
        let expr = b_offset_expr(k_ctr, k_off, j_off, b_row_stride, 4, n, k, false, 8);

        // Assert: offset = ScalarVReg(k_ctr) * b_row_stride + LoopOffset(j_off)
        // The Mul factor should be exactly b_row_stride
        match &expr {
            OffsetExpr::Add(mul_term, loop_term) => {
                match mul_term.as_ref() {
                    OffsetExpr::Mul(inner, factor) => {
                        assert_eq!(*factor, b_row_stride,
                            "normal layout Mul factor should equal b_row_stride");
                        assert!(matches!(inner.as_ref(), OffsetExpr::ScalarVReg(_)),
                            "normal layout Mul inner should be ScalarVReg(k_ctr)");
                    }
                    other => panic!("expected Mul for normal layout first term, got {:?}", other),
                }
                assert!(matches!(loop_term.as_ref(), OffsetExpr::LoopOffset(_)),
                    "normal layout second term should be LoopOffset(j_off)");
            }
            other => panic!("expected Add for normal layout, got {:?}", other),
        }
    }

    #[test]
    fn test_emit_gemm_blis_inline_trans_b_bf16_produces_narrowing() {
        // Arrange: BLIS GEMM with trans_b=true and BF16 — accumulator widens, writeback narrows
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: trans_b=true with BF16
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 8, 16, width, a, b, c, 4, 2, None, 1, QuantPrecision::BF16, true,
        );

        // Assert: should succeed and emit VecNarrow for BF16 writeback
        assert!(result.is_ok(), "BLIS trans_b BF16 should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BLIS trans_b BF16 should emit VecNarrow for accumulator writeback");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_symbolic_m_with_seq_bound_override() {
        // Arrange: symbolic M with explicit seq_bound_override
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(1024) };
        let override_bound = BoundExpr::Const(64);

        // Act: symbolic M with explicit override (e.g., batch scheduler knows actual seq_len=64)
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &m_dim, 8, 16, width, a, b, c, &[], &sym_map, false,
            Some(&override_bound),
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: should succeed using the override bound instead of sym_map
        assert!(result.is_ok(), "symbolic M with seq_bound_override should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should produce instructions with override bound");
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "GEMM with override bound should emit VecStore");
    }

    #[test]
    fn test_emit_tile_gemm_bf16_dtype_produces_tile_config() {
        // Arrange: tile GEMM with BF16 dtype
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dt = crate::types::DType::BF16;

        // Act: tile GEMM with BF16
        let result = emit_tile_gemm(&mut prog, width, 4, 4, 4, 16, dt);

        // Assert: should succeed and emit TileConfig with BF16 dtype
        assert!(result.is_ok());
        let tile_config = prog.instrs.iter().find(|i| matches!(i, VmInstr::TileConfig { .. }));
        assert!(tile_config.is_some(), "BF16 tile GEMM should emit TileConfig");
        if let Some(VmInstr::TileConfig { dtype, .. }) = tile_config {
            assert_eq!(*dtype, crate::types::DType::BF16, "TileConfig dtype should be BF16");
        }
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_inline_decode_m1_succeeds() {
        // Arrange: GPU decode path with M=1 (single-row GEMV on GPU)
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=1 with cta_m=16 -> mi=1 (partial first CTA tile)
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 1, 16, 32, width, a, b, c,
            16, 16, 16, 1, 1, 16, QuantPrecision::F32, false,
        );

        // Assert: M=1 GPU GEMV should succeed with partial tile handling
        assert!(result.is_ok(), "GPU M=1 decode GEMV should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty(), "should produce instructions");
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "GPU M=1 GEMV should emit VecStore for result writeback");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_n_exactly_equals_lanes_no_tail() {
        // Arrange: N exactly equals SIMD lanes — no scalar tail path
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256; // lanes=8
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: N=8 = lanes=8 -> n_vecs=1, n_tail=0
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &SymDim::Concrete(2), 8, 8, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: should succeed with only vectorized path, no scalar tail
        assert!(result.is_ok(), "N=lanes GEMM should succeed: {:?}", result.err());
        let has_vec_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { width: SimdWidth::W256, .. }));
        assert!(has_vec_store, "N=lanes GEMM should emit W256 VecStore (vectorized path)");
        // No scalar VecStore should appear since n_tail=0
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert_eq!(scalar_stores, 0, "N=lanes should not emit scalar VecStore (n_tail=0)");
    }

    #[test]
    fn test_emit_gemm_blis_inline_large_m_produces_multiple_m_blocks() {
        // Arrange: BLIS GEMM where M > mr -> multiple M-block iterations
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=16, mr=4 -> 4 M-blocks (0..4, 4..8, 8..12, 12..16)
        let result = emit_gemm_blis_inline(
            &mut prog, 16, 16, 16, width, a, b, c, 4, 2, None, 1, QuantPrecision::F32, false,
        );

        // Assert: should succeed with multiple M-blocks
        assert!(result.is_ok(), "BLIS GEMM with M > mr should succeed: {:?}", result.err());
        // Each M-block writes its C rows via VecStore; 4 blocks should produce multiple stores
        let store_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(store_count >= 4, "M=16 with mr=4 should have at least 4 VecStore groups, got {}", store_count);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // +10 new tests: BLIS nr=1 microkernel, trans-B BF16 epilogue+tail,
    // GPU tiled partial K steps, BLIS M=mr exact boundary, naive GEMM
    // symbolic M loop bound, GPU tiled partial BF16 writeback, b_offset
    // normal vs trans_b structural difference, BLIS accumulator zero-init
    // per block, MMA partial inner K step, trans-B symbolic M override
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_emit_gemm_blis_inline_nr_equals_one_succeeds() {
        // Arrange: BLIS with nr=1 (single-column microkernel) — minimal column blocking
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: nr=1 means each J-block processes 1*lanes=8 columns per iteration
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 16, 8, width, a, b, c, 4, 1, None, 1, QuantPrecision::F32, false,
        );

        // Assert: nr=1 should succeed with 1-column accumulator per microkernel
        assert!(result.is_ok(), "BLIS GEMM with nr=1 should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "BLIS nr=1 should emit VecStore for C writeback");
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_bf16_with_k_tail_and_epilogue() {
        // Arrange: trans-B GEMM with BF16, K not divisible by lanes, and an epilogue
        // This exercises: BF16 narrowing + k_tail scalar path + epilogue lowering
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let epilogue: Vec<TraceOp> = vec![
            TraceOp::Input(0),
            TraceOp::Const(0.5),
            TraceOp::Mul(ValueId(0), ValueId(1)),
        ];

        // Act: BF16, K=10 (k_vecs=1, k_tail=2), with epilogue
        let result = emit_gemm_trans_b_inline(
            &mut prog, &SymDim::Concrete(1), 4, 10, width, a, b, c, &epilogue, &sym_map, None,
            QuantPrecision::BF16,
        );

        // Assert: BF16 trans-B with k_tail + epilogue should succeed
        assert!(result.is_ok(), "trans-B BF16 with k_tail+epilogue should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BF16 trans-B should emit VecNarrow for F32->BF16 writeback");
        let has_hreduce = prog.instrs.iter().any(|i| matches!(i, VmInstr::HReduce { .. }));
        assert!(has_hreduce, "trans-B with k_tail should emit HReduce to collapse vector acc");
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_inline_k_not_divisible_by_mma_k() {
        // Arrange: GPU tiled GEMM where K is not divisible by mma_k
        // K=24 with mma_k=16 -> 2 inner K iterations: first with 16, second with 8
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: K=24, mma_k=16 -> partial last inner K step
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 16, 16, 24, width, a, b, c,
            16, 16, 24, 4, 4, 16, QuantPrecision::F32, false,
        );

        // Assert: partial K steps should be handled correctly
        assert!(result.is_ok(), "GPU tiled GEMM with K not divisible by mma_k should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
        let fma_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Fma { .. })).count();
        assert!(fma_count > 0, "should emit FMA instructions for partial K steps");
    }

    #[test]
    fn test_emit_gemm_blis_inline_m_exactly_equals_mr_no_clamp() {
        // Arrange: M = mr exactly — no clamping needed, single M-block
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=4, mr=4 -> exactly one M-block, mr_actual=mr=4
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 16, 16, width, a, b, c, 4, 2, None, 1, QuantPrecision::F32, false,
        );

        // Assert: exact boundary should succeed without clamping
        assert!(result.is_ok(), "BLIS GEMM with M=mr should succeed: {:?}", result.err());
        let broadcast_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Broadcast { .. })).count();
        // With exactly one M-block, there should be exactly one set of accumulator zero-inits
        assert!(broadcast_count > 0, "should have accumulator zero-init Broadcast instructions");
    }

    #[test]
    fn test_emit_gemm_inline_with_epilogue_symbolic_m_uses_sym_map_bound() {
        // Arrange: symbolic M dimension — loop bound comes from sym_map, not hardcoded
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(256) };

        // Act: no seq_bound_override — must use sym_map.to_bound()
        let result = emit_gemm_inline_with_epilogue(
            &mut prog, &m_dim, 8, 16, width, a, b, c, &[], &sym_map, false, None,
            QuantPrecision::F32, false,
            crate::compiler::codegen::vm::isa_hook::EpiloguePlace::OnAccumulators,
        );

        // Assert: symbolic M should produce a valid instruction stream
        assert!(result.is_ok(), "symbolic M via sym_map should succeed: {:?}", result.err());
        assert!(!prog.instrs.is_empty());
        // Verify the M loop produces VecStore (not just zero-init)
        let store_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(store_count > 0, "symbolic M GEMM should emit VecStore instructions");
    }

    #[test]
    fn test_emit_gemm_gpu_tiled_inline_bf16_with_partial_tiles() {
        // Arrange: GPU tiled GEMM with BF16 and non-divisible M/N dimensions
        // This exercises both partial CTA/warp tiles AND BF16 VecNarrow writeback
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: M=10, N=10 with cta_m=8, cta_n=8, warp_m=4, warp_n=4
        // First CTA tile: mi=8, nj=8 (full). Second CTA tile: mi=2, nj=2 (partial)
        let result = emit_gemm_gpu_tiled_inline(
            &mut prog, 10, 10, 16, width, a, b, c,
            8, 8, 16, 4, 4, 16, QuantPrecision::BF16, false,
        );

        // Assert: partial tiles with BF16 should produce VecNarrow on writeback
        assert!(result.is_ok(), "GPU tiled BF16 partial should succeed: {:?}", result.err());
        let has_narrow = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecNarrow { .. }));
        assert!(has_narrow, "BF16 GPU tiled with partial tiles should emit VecNarrow");
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "should emit VecStore for result writeback");
    }

    #[test]
    fn test_b_offset_expr_normal_and_trans_b_produce_structurally_different_expressions() {
        // Arrange: same parameters, only trans_b differs
        let prog = &mut VmProgram::new();
        let k_ctr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let k_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let j_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let n = 64;
        let k = 32;
        let b_row_stride = n * 4;

        // Act: compute both expressions
        let normal = b_offset_expr(k_ctr, k_off, j_off, b_row_stride, 4, n, k, false, 8);
        let trans = b_offset_expr(k_ctr, k_off, j_off, 0, 4, n, k, true, 8);

        // Assert: structurally different — normal uses ScalarVReg(k_ctr) * row_stride,
        // trans uses LoopOffset(j_off) * k
        assert_ne!(normal, trans, "normal and trans-B offset expressions should differ");
        // Normal: first term is Mul(ScalarVReg(k_ctr), b_row_stride)
        if let OffsetExpr::Add(mul_term, _) = &normal {
            if let OffsetExpr::Mul(inner, factor) = mul_term.as_ref() {
                assert!(matches!(inner.as_ref(), OffsetExpr::ScalarVReg(_)));
                assert_eq!(*factor, b_row_stride);
            } else {
                panic!("normal first term should be Mul, got {:?}", mul_term);
            }
        } else {
            panic!("normal should be Add, got {:?}", normal);
        }
        // Trans: first term is Mul(LoopOffset(j_off), k)
        if let OffsetExpr::Add(mul_term, _) = &trans {
            if let OffsetExpr::Mul(inner, factor) = mul_term.as_ref() {
                assert!(matches!(inner.as_ref(), OffsetExpr::LoopOffset(_)));
                assert_eq!(*factor, k);
            } else {
                panic!("trans first term should be Mul, got {:?}", mul_term);
            }
        } else {
            panic!("trans should be Add, got {:?}", trans);
        }
    }

    #[test]
    fn test_emit_gemm_blis_inline_accumulator_zero_init_per_j_block() {
        // Arrange: BLIS GEMM with multiple J-blocks — each J-block should zero its accumulators
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256; // lanes=8
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);

        // Act: N=32, nr=2, lanes=8 -> nr*lanes=16 -> 2 J-blocks per M-block
        // Each J-block must zero its mr*nr_actual accumulator registers
        let result = emit_gemm_blis_inline(
            &mut prog, 4, 32, 8, width, a, b, c, 4, 2, None, 1, QuantPrecision::F32, false,
        );

        // Assert: accumulator zero-init Broadcast(0.0) should appear multiple times
        assert!(result.is_ok(), "BLIS GEMM should succeed: {:?}", result.err());
        let zero_broadcasts = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. })
        }).count();
        // With 2 J-blocks and mr=4, nr=2 -> 4*2=8 acc per block -> at least 2*8=16 zero-inits
        assert!(zero_broadcasts >= 16,
            "should have at least 16 accumulator zero-inits (2 J-blocks × 8 accs), got {}", zero_broadcasts);
    }

    #[test]
    fn test_emit_mma_on_smem_kk_smaller_than_mma_k_single_partial_step() {
        // Arrange: MMA where kk < mma_k — single partial inner K iteration
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let accs: Vec<VRegId> = (0..4).map(|_| prog.alloc_vreg(VRegKind::Vec, width)).collect();

        // Act: mi=2, nj=2, kk=8, mma_k=16 -> single inner step with 8 elements
        emit_mma_on_smem(
            &mut prog, &accs, "smem_a", "smem_b",
            0, 0, 0, 2, 2, 8, // kk=8 < mma_k=16
            2, 2, 16, 2, 2, 16,
            16, 16, 4, QuantPrecision::F32, QuantPrecision::F32, width,
            64, 64,
        );

        // Assert: single partial K step should emit SharedMemLoad + Fma for 2x2=4 accs
        let fma_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Fma { .. })).count();
        let smem_load_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::SharedMemLoad { .. })).count();
        assert_eq!(fma_count, 4,
            "partial kk MMA should have exactly 4 FMAs (2×2), got {}", fma_count);
        // A fragment is shared across columns: 2 A loads (per row) + 4 B loads (per row×col) = 6 SharedMemLoad
        assert_eq!(smem_load_count, 6,
            "partial kk MMA should have 6 SharedMemLoads (2 A + 4 B), got {}", smem_load_count);
    }

    #[test]
    fn test_emit_gemm_trans_b_inline_symbolic_m_with_seq_bound_override() {
        // Arrange: trans-B GEMM with symbolic M and explicit seq_bound_override
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let a = prog.alloc_vreg(VRegKind::Ptr, width);
        let b = prog.alloc_vreg(VRegKind::Ptr, width);
        let c = prog.alloc_vreg(VRegKind::Ptr, width);
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(2048) };
        let override_bound = BoundExpr::Const(1); // Override to M=1 (decode path)

        // Act: symbolic M with override=1 for trans-B path
        let result = emit_gemm_trans_b_inline(
            &mut prog, &m_dim, 8, 16, width, a, b, c, &[], &sym_map,
            Some(&override_bound), QuantPrecision::F32,
        );

        // Assert: override should work for trans-B path just like normal path
        assert!(result.is_ok(), "trans-B symbolic M with override should succeed: {:?}", result.err());
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "trans-B GEMM with symbolic M override should emit VecStore");
    }
}

