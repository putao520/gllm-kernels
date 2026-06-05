//! Vision and audio operator lowering — DepthwiseConv1D, PatchEmbed.

use super::instr::*;
use super::plan_lower::{SymDimSlotMap, TensorPtrResolver};
use super::vm_state::AbiPtrs;
use crate::compiler::codegen::DwcScratchRequirement;
use crate::compiler::graph::CompilerGraph;
use crate::compiler::trace::{QuantPrecision, TraceOp, ValueId};
use crate::types::CompilerError;


// PerLayerEmbed lower functions removed — replaced by AltUpPredict/AltUpCorrect/AltUpInject
// (Injective ops dispatched via emit_injective_inline, no dedicated lower needed)

pub(crate) fn lower_depthwise_conv1d(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    width: SimdWidth,
    channels: usize,
    kernel_size: usize,
    _causal: bool,
    sym_map: &SymDimSlotMap,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
    req: &DwcScratchRequirement,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if op.inputs.len() != 2 {
        return Err(CompilerError::CodegenViolation(format!(
            "DepthwiseConv1D: 需要 2 输入 [x, weight], 实际 {}", op.inputs.len(),
        )));
    }
    if op.outputs.len() != 1 {
        return Err(CompilerError::CodegenViolation(format!(
            "DepthwiseConv1D: 需要 1 输出, 实际 {}", op.outputs.len(),
        )));
    }

    // ── 物化指针 ──
    let x_ptr = resolver.materialize(prog, op.inputs[0], abi)
        .ok_or_else(|| CompilerError::CodegenViolation(
            "DepthwiseConv1D: inputs[0] (x) 无法定位".into()))?;
    let w_ptr = resolver.materialize(prog, op.inputs[1], abi)
        .ok_or_else(|| CompilerError::CodegenViolation(
            "DepthwiseConv1D: inputs[1] (weight) 无法定位".into()))?;
    let out_ptr = resolver.materialize(prog, op.outputs[0], abi)
        .ok_or_else(|| CompilerError::CodegenViolation(
            "DepthwiseConv1D: outputs[0] 无法定位".into()))?;
    let scratch_base = abi.scratch_ptr.ok_or_else(|| CompilerError::CodegenViolation(
        "DepthwiseConv1D: scratchpad 未初始化 (needs_scratch 应在 dwc_req 存在时为 true)".into()))?;

    // Padded buffer base = scratch_base + req.padded_offset
    let padded_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: padded_ptr,
        src: PtrExpr::VRegPlusConst(scratch_base, req.padded_offset),
    });

    let elem = dtype.elem_bytes();
    let channel_row_bytes = channels * dtype.elem_bytes(); // channels * 4
    let left_pad = req.left_pad;

    // ── 推导运行时 seq_bound (Symbolic 或 Concrete) ──
    let out_tid = op.outputs[0];
    let out_tensor = graph.tensor(out_tid).ok_or_else(|| CompilerError::CodegenViolation(
        "DepthwiseConv1D: 输出张量不存在".into()))?;
    let seq_dim = out_tensor.shape.first().cloned().ok_or_else(|| {
        CompilerError::CodegenViolation("DepthwiseConv1D: 输出 shape 为空".into())
    })?;
    let seq_bound = sym_map.to_bound(&seq_dim);

    // ── Stage 1: Zero-fill padded buffer [max_seq + total_pad, channels] ──
    // 按 padded_rows × channel_row_bytes 字节总量 (max_seq 使用 req.max_seq_len,
    // 编译时上界)。Symbolic seq < max_seq 时, 末尾 padding 多余, 不影响语义
    // (主卷积循环只读 0..seq+total_pad 范围)。
    let padded_rows_max = req.max_seq_len + left_pad + (req.kernel_size - 1 - left_pad);
    let total_padded_bytes = padded_rows_max * channel_row_bytes;
    emit_zero_fill_bytes(prog, padded_ptr, total_padded_bytes, width, dtype)?;

    // ── Stage 2: Copy input x[0..seq, :] → padded[left_pad..left_pad+seq, :] ──
    // 起点偏移: left_pad * channel_row_bytes
    let copy_dst_ptr = if left_pad > 0 {
        let p = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: p,
            src: PtrExpr::VRegPlusConst(padded_ptr, left_pad * channel_row_bytes),
        });
        p
    } else {
        padded_ptr
    };
    emit_row_copy(prog, x_ptr, copy_dst_ptr, seq_bound.clone(), channels, width, dtype)?;

    // ── Stage 3: Conv 主循环: output[t, c] = Σ_k padded[t+k, c] * w[c, k] ──
    // 循环结构 (ARCH-NO-LOOP-UNROLL 合规):
    //   外层 t:    emit_loop(seq_bound,     step = channels * 4)
    //   中层 c:    emit_loop(Const(channels), step = 4)     — channels Concrete
    //   内层 k:    Rust-unroll 0..kernel_size               — K 小常量 ≤ 15
    //
    // 每个 (t, c, k): acc += padded[t+k, c] * w[c, k]
    // padded[t+k, c] 偏移: (t_off) + k * channel_row_bytes + c_off
    //   (t_off 来自 t-loop byte_offset, c_off 来自 c-loop byte_offset)
    // w[c, k] 偏移: c * kernel_size * 4 + k * 4 = (c_off / 4) * kernel_size * 4 + k * 4
    //   由于 c_off 步进是 4B, c_off / 4 就是 c 计数, 但 c_off 本身已是 byte 量:
    //   row-major w[c, k] = w_ptr + (c * K + k) * 4 = w_ptr + c_off * K + k * 4
    //   OffsetExpr 支持 Mul(LoopOffset, K) + Const(k*4)。
    let s_width = SimdWidth::Scalar;
    let acc = prog.alloc_vreg(VRegKind::Vec, s_width);
    let x_val = prog.alloc_vreg(VRegKind::Vec, s_width);
    let w_val = prog.alloc_vreg(VRegKind::Vec, s_width);

    prog.emit_loop(seq_bound, channel_row_bytes, |prog, _t_ctr, t_off| {
        // 外层每次: 行基址 = padded_ptr + t_off (含 left_pad 偏移), output_row = out_ptr + t_off
        prog.emit_loop(BoundExpr::Const(channels), elem, |prog, _c_ctr, c_off| {
            // acc = 0
            prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width: s_width, dtype, });

            // Rust-unroll k — kernel_size ≤ 15 小常量, 合规 UNROLL_THRESHOLD 合法 Const 展开场景。
            for k in 0..kernel_size {
                // x_val = padded[t + k, c] = *(padded_ptr + t_off + k*channel_row_bytes + c_off)
                let padded_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(t_off)),
                    Box::new(OffsetExpr::Add(
                        Box::new(OffsetExpr::Const(k * channel_row_bytes)),
                        Box::new(OffsetExpr::LoopOffset(c_off)),
                    )),
                );
                prog.emit(VmInstr::VecLoad {
                    dst: x_val, base: padded_ptr, offset: padded_off, width: s_width,
                    dtype,
                });
                // w_val = broadcast w[c, k] = broadcast *(w_ptr + c_off * kernel_size + k * 4)
                // w[c, k] byte offset = c * K * 4 + k * 4 = c_off * K + k * 4
                // (c_off 是 byte_offset, 每次迭代加 4B → c_off / 4 == c, 所以 c_off * K = c * K * 4)
                let w_byte_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(c_off)), kernel_size)),
                    Box::new(OffsetExpr::Const(k * elem)),
                );
                prog.emit(VmInstr::Broadcast {
                    dst: w_val,
                    src: ScalarExpr::MemLoad(w_ptr, w_byte_off),
                    width: s_width,
                    dtype,
                });
                // acc = acc + x_val * w_val via auto_lower_trace_into
                {
                    let dwc_fma_body: Vec<TraceOp> = vec![
                        TraceOp::Input(0),  // [0] x_val
                        TraceOp::Input(1),  // [1] w_val
                        TraceOp::Input(2),  // [2] acc
                        TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)), // [3] new_acc
                    ];
                    super::auto_select::auto_lower_trace_into(prog, &dwc_fma_body, &[x_val, w_val, acc], acc, s_width, QuantPrecision::F32)
                        .expect("lower_depthwise_conv1d: FMA auto_lower invariant violation");
                }
            }

            // output[t, c] = acc (lane 0 only, SimdWidth::Scalar)
            let out_off = OffsetExpr::Add(
                Box::new(OffsetExpr::LoopOffset(t_off)),
                Box::new(OffsetExpr::LoopOffset(c_off)),
            );
            prog.emit(VmInstr::VecStore {
                base: out_ptr, offset: out_off, src: acc, width: s_width,
                dtype,
            });
        });
    });

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 PatchEmbed lower (SigLIP / ViT vision tower, T65)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// 参考实现: scalar-ops/src/patch_embed.rs::scalar_patch_embed (Phase 0 ground
// truth, 禁止运行时调用 — NO_SCALAR 铁律)。
//
// 算法:
//   num_patches_side = image_size / patch_size
//   for p_row in 0..num_patches_side:
//     for p_col in 0..num_patches_side:
//       for e in 0..embed_dim:
//         acc = 0
//         for c in 0..in_channels:
//           for kr in 0..patch_size:
//             for kc in 0..patch_size:
//               acc += image[c, p_row·P + kr, p_col·P + kc]
//                    * kernel[e, c, kr, kc]
//         patches[p_row·side + p_col, e] = acc  (row-major)
//
// 所有维度在 OpKind 中是 Concrete (ViT 模型的 image_size / patch_size /
// embed_dim / in_channels 在模型结构里硬固定, 不因 seq 动态变化)。
//
// Byte offset 策略 (每层 emit_loop 的 step_bytes 取 `elem=4`, 即 LoopOffset
// 等价于 `counter × 4`; 所有物理 byte 偏移通过 `OffsetExpr::Mul(LoopOffset, K)`
// 得到 `counter × 4 × K`):
//
//   image_off =   Mul(p_row_off, P·image_size)       // p_row · patch_size · image_size · elem
//               + Mul(p_col_off, P)                  // p_col · patch_size · elem
//               + Mul(c_off,     image_size·image_size) // c · image_size² · elem
//               + Mul(kr_off,    image_size)         // kr · image_size · elem
//               + Const(kc · elem)                   // kc · elem
//
//   kernel_off = Mul(e_off, in_channels · P · P)     // e · in_channels · P² · elem
//               + Mul(c_off, P · P)                  // c · P² · elem
//               + Mul(kr_off, P)                     // kr · P · elem
//               + Const(kc · elem)                   // kc · elem
//
//   output_off = Mul(p_row_off, num_patches_side · embed_dim)  // p_row · side · embed_dim · elem
//               + Mul(p_col_off, embed_dim)         // p_col · embed_dim · elem
//               + LoopOffset(e_off)                 // e · elem (scale 1, step=4 给 LoopOffset = e · 4)
//
// ARCH-SYMDIM-NO-CONST-DEGRADE: OpKind 字段全部 Concrete, 无 Symbolic 降级问题。
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_patch_embed(
    prog: &mut VmProgram,
    patch_size: usize,
    embed_dim: usize,
    in_channels: usize,
    image_size: usize,
    image_ptr: VRegId,
    kernel_ptr: VRegId,
    patches_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if patch_size == 0 || in_channels == 0 || image_size == 0 || embed_dim == 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "PatchEmbed: invalid dims (patch_size={patch_size}, embed_dim={embed_dim}, \
             in_channels={in_channels}, image_size={image_size})"
        )));
    }
    if image_size % patch_size != 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "PatchEmbed: image_size ({image_size}) 必须被 patch_size ({patch_size}) 整除 \
             (ViT 语义要求)"
        )));
    }
    let num_patches_side = image_size / patch_size;
    let elem = dtype.elem_bytes(); // 4 bytes
    let s_width = SimdWidth::Scalar;

    // ── 寄存器规划 (最小化 live 集, 所有 SIMD vreg 均标量 lane 0) ──
    let acc = prog.alloc_vreg(VRegKind::Vec, s_width);     // scalar f32 accumulator
    let x_val = prog.alloc_vreg(VRegKind::Vec, s_width);   // image[c, r, kc] 临时
    let w_val = prog.alloc_vreg(VRegKind::Vec, s_width);   // kernel[e, c, kr, kc] 临时

    // 常量缩放因子 (与上方公式对齐, 均为编译时常量):
    let scale_image_row   = patch_size * image_size;                    // p_row_off × scale
    let scale_image_col   = patch_size;                                 // p_col_off × scale
    let scale_image_chan  = image_size * image_size;                    // c_off × scale
    let scale_image_kr    = image_size;                                 // kr_off × scale
    let scale_kernel_e    = in_channels * patch_size * patch_size;      // e_off × scale
    let scale_kernel_c    = patch_size * patch_size;                    // c_off × scale
    let scale_kernel_kr   = patch_size;                                 // kr_off × scale
    let scale_out_p_row   = num_patches_side * embed_dim;               // p_row_off × scale
    let scale_out_p_col   = embed_dim;                                  // p_col_off × scale

    // ── 5 层 emit_loop (p_row → p_col → e → c → kr) + Rust unroll kc ──
    prog.emit_loop(BoundExpr::Const(num_patches_side), elem, |prog, _p_row_ctr, p_row_off| {
        prog.emit_loop(BoundExpr::Const(num_patches_side), elem, |prog, _p_col_ctr, p_col_off| {
            prog.emit_loop(BoundExpr::Const(embed_dim), elem, |prog, _e_ctr, e_off| {
                // acc = 0
                prog.emit(VmInstr::Broadcast {
                    dst: acc, src: ScalarExpr::Const(0.0), width: s_width,
                    dtype,
                });

                prog.emit_loop(BoundExpr::Const(in_channels), elem, |prog, _c_ctr, c_off| {
                    prog.emit_loop(BoundExpr::Const(patch_size), elem, |prog, _kr_ctr, kr_off| {
                        // Rust-unroll kc: patch_size 是编译时常量 (ViT 典型 14/16 ≤ 16,
                        // 合规 UNROLL_THRESHOLD 的"内层微维度 + 编译时确定且极小的维度"
                        // 例外)。展开后消除循环开销, vfmadd231ps 可级联。
                        for kc in 0..patch_size {
                            // image[c, r, col] byte offset:
                            //   Mul(p_row_off, P·IS)  (p_row · P · IS · elem)
                            // + Mul(p_col_off, P)     (p_col · P · elem)
                            // + Mul(c_off,     IS·IS) (c · IS² · elem)
                            // + Mul(kr_off,    IS)    (kr · IS · elem)
                            // + Const(kc · elem)
                            let image_off = add_offsets(vec![
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(p_row_off)), scale_image_row),
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(p_col_off)), scale_image_col),
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(c_off)),     scale_image_chan),
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(kr_off)),    scale_image_kr),
                                OffsetExpr::Const(kc * elem),
                            ]);
                            prog.emit(VmInstr::VecLoad {
                                dst: x_val, base: image_ptr, offset: image_off, width: s_width,
                                dtype,
                            });

                            // kernel[e, c, kr, kc] byte offset:
                            //   Mul(e_off,  IC·P²)
                            // + Mul(c_off,  P²)
                            // + Mul(kr_off, P)
                            // + Const(kc · elem)
                            let kernel_off = add_offsets(vec![
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(e_off)),  scale_kernel_e),
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(c_off)),  scale_kernel_c),
                                OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(kr_off)), scale_kernel_kr),
                                OffsetExpr::Const(kc * elem),
                            ]);
                            // kernel 值必须 broadcast 到 w_val (scalar width → lane 0)。
                            // 采用 VecLoad scalar (width=Scalar → 单 f32 load) 与 x_val 对称,
                            // vfmadd231ps 对 xmm 下位 f32 lane 运算即可 (等价 vfmadd231ss)。
                            prog.emit(VmInstr::VecLoad {
                                dst: w_val, base: kernel_ptr, offset: kernel_off, width: s_width,
                                dtype,
                            });
                            // acc = acc + x_val · w_val via auto_lower_trace_into
                            {
                                let pe_fma_body: Vec<TraceOp> = vec![
                                    TraceOp::Input(0),  // [0] x_val
                                    TraceOp::Input(1),  // [1] w_val
                                    TraceOp::Input(2),  // [2] acc
                                    TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)), // [3] new_acc
                                ];
                                super::auto_select::auto_lower_trace_into(prog, &pe_fma_body, &[x_val, w_val, acc], acc, s_width, QuantPrecision::F32)
                                    .expect("lower_patch_embed: FMA auto_lower invariant violation");
                            }
                        }
                    });
                });

                // patches[p, e] = acc:
                //   Mul(p_row_off, side·ED) + Mul(p_col_off, ED) + LoopOffset(e_off)
                let out_off = add_offsets(vec![
                    OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(p_row_off)), scale_out_p_row),
                    OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(p_col_off)), scale_out_p_col),
                    OffsetExpr::LoopOffset(e_off),
                ]);
                prog.emit(VmInstr::VecStore {
                    base: patches_ptr, offset: out_off, src: acc, width: s_width,
                    dtype,
                });
            });
        });
    });

    Ok(())
}

/// 构造多项相加的 OffsetExpr (左结合 Add), 过滤 `Const(0)` 和空 Mul (scale=0)。
/// 保持语义与 `a + b + c + ...` 一致, 便于 ISA lower 按 eval_offset_to_rax 处理。
pub(crate) fn add_offsets(parts: Vec<OffsetExpr>) -> OffsetExpr {
    let mut iter = parts.into_iter().filter(|p| !is_zero_offset(p));
    let mut acc = iter.next().unwrap_or(OffsetExpr::Const(0));
    for next in iter {
        acc = OffsetExpr::Add(Box::new(acc), Box::new(next));
    }
    acc
}

pub(crate) fn is_zero_offset(e: &OffsetExpr) -> bool {
    match e {
        OffsetExpr::Const(0) => true,
        OffsetExpr::Mul(_, 0) => true,
        _ => false,
    }
}

/// 零填充 `[ptr, ptr + total_bytes)` — 用于 DWC padded scratchpad 初始化。
///
/// 按最大 SIMD 宽度主循环 + scalar tail,避免越界写。total_bytes 是编译时常量。
pub(crate) fn emit_zero_fill_bytes(
    prog: &mut VmProgram,
    ptr: VRegId,
    total_bytes: usize,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    if total_bytes == 0 { return Ok(()); }
    let step_bytes = width.bytes();
    let vec_count = total_bytes / step_bytes;
    let tail_bytes = total_bytes - vec_count * step_bytes;

    let zero_vec = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: zero_vec, src: ScalarExpr::Const(0.0), width, dtype, });

    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _ctr, off| {
            prog.emit(VmInstr::VecStore {
                base: ptr, offset: OffsetExpr::LoopOffset(off), src: zero_vec, width,
                dtype,
            });
        });
    }
    // Scalar tail: tail_bytes / elem 次 4B 写
    let elem = dtype.elem_bytes();
    if tail_bytes > 0 {
        let tail_elems = tail_bytes / elem;
        let s_width = SimdWidth::Scalar;
        let zero_scalar = prog.alloc_vreg(VRegKind::Vec, s_width);
        prog.emit(VmInstr::Broadcast { dst: zero_scalar, src: ScalarExpr::Const(0.0), width: s_width, dtype, });
        let tail_base = vec_count * step_bytes;
        if tail_elems > 0 {
            prog.emit_loop(BoundExpr::Const(tail_elems), elem, |prog, _ctr, off| {
                let full_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(off)),
                    Box::new(OffsetExpr::Const(tail_base)),
                );
                prog.emit(VmInstr::VecStore {
                    base: ptr, offset: full_off, src: zero_scalar, width: s_width,
                    dtype,
                });
            });
        }
    }
    Ok(())
}

/// 行式复制 `src[0..rows, :inner]` → `dst[0..rows, :inner]` (row-major,
/// inner 连续, 每行 inner * 4 字节), 外层 seq_bound 可为 Symbolic。
pub(crate) fn emit_row_copy(
    prog: &mut VmProgram,
    src_ptr: VRegId,
    dst_ptr: VRegId,
    seq_bound: BoundExpr,
    inner: usize,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let elem = dtype.elem_bytes();
    let lanes = width.f32_lanes().max(1);
    let step_bytes = width.bytes();
    let row_bytes = inner * dtype.elem_bytes();
    let vec_count = inner / lanes;
    let tail = inner - vec_count * lanes;

    let tmp = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit_loop(seq_bound, row_bytes, |prog, _r_ctr, r_off| {
        let row_src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let row_dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: row_src, src: PtrExpr::VRegPlusVReg(src_ptr, r_off) });
        prog.emit(VmInstr::LoadPtr { dst: row_dst, src: PtrExpr::VRegPlusVReg(dst_ptr, r_off) });

        if vec_count > 0 {
            prog.emit_loop(BoundExpr::Const(vec_count), step_bytes, |prog, _c_ctr, c_off| {
                prog.emit(VmInstr::VecLoad {
                    dst: tmp, base: row_src, offset: OffsetExpr::LoopOffset(c_off), width,
                    dtype,
                });
                prog.emit(VmInstr::VecStore {
                    base: row_dst, offset: OffsetExpr::LoopOffset(c_off), src: tmp, width,
                    dtype,
                });
            });
        }
        if tail > 0 {
            let s_width = SimdWidth::Scalar;
            let tail_base = vec_count * step_bytes;
            let s_tmp = prog.alloc_vreg(VRegKind::Vec, s_width);
            prog.emit_loop(BoundExpr::Const(tail), elem, |prog, _t_ctr, t_off| {
                let off = OffsetExpr::Add(
                    Box::new(OffsetExpr::LoopOffset(t_off)),
                    Box::new(OffsetExpr::Const(tail_base)),
                );
                prog.emit(VmInstr::VecLoad {
                    dst: s_tmp, base: row_src, offset: off.clone(), width: s_width,
                    dtype,
                });
                prog.emit(VmInstr::VecStore {
                    base: row_dst, offset: off, src: s_tmp, width: s_width,
                    dtype,
                });
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::vm::instr::*;
    use crate::compiler::codegen::PleScratchRequirement;

    // ── add_offsets ──

    #[test]
    fn add_offsets_single_const_returns_same_const() {
        let result = add_offsets(vec![OffsetExpr::Const(42)]);
        assert_eq!(result, OffsetExpr::Const(42));
    }

    #[test]
    fn add_offsets_multiple_consts_produces_nested_add() {
        let result = add_offsets(vec![OffsetExpr::Const(10), OffsetExpr::Const(20), OffsetExpr::Const(30)]);
        let expected = OffsetExpr::Add(
            Box::new(OffsetExpr::Add(
                Box::new(OffsetExpr::Const(10)),
                Box::new(OffsetExpr::Const(20)),
            )),
            Box::new(OffsetExpr::Const(30)),
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn add_offsets_filters_zero_const() {
        let result = add_offsets(vec![OffsetExpr::Const(0), OffsetExpr::Const(100), OffsetExpr::Const(0)]);
        assert_eq!(result, OffsetExpr::Const(100));
    }

    #[test]
    fn add_offsets_filters_zero_scale_mul() {
        let vreg = VRegId(5);
        let result = add_offsets(vec![OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(vreg)), 0), OffsetExpr::Const(77)]);
        assert_eq!(result, OffsetExpr::Const(77));
    }

    #[test]
    fn add_offsets_empty_vec_produces_zero_const() {
        let result = add_offsets(vec![]);
        assert_eq!(result, OffsetExpr::Const(0));
    }

    #[test]
    fn add_offsets_preserves_non_zero_mul() {
        let vreg = VRegId(3);
        let result = add_offsets(vec![OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(vreg)), 64), OffsetExpr::Const(8)]);
        let expected = OffsetExpr::Add(
            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(vreg)), 64)),
            Box::new(OffsetExpr::Const(8)),
        );
        assert_eq!(result, expected);
    }

    // ── is_zero_offset ──

    #[test]
    fn is_zero_offset_const_zero() {
        assert!(is_zero_offset(&OffsetExpr::Const(0)));
    }

    #[test]
    fn is_zero_offset_const_nonzero() {
        assert!(!is_zero_offset(&OffsetExpr::Const(1)));
    }

    #[test]
    fn is_zero_offset_mul_zero_scale() {
        let vreg = VRegId(0);
        assert!(is_zero_offset(&OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(vreg)), 0)));
    }

    // ── emit_zero_fill_bytes ──

    #[test]
    fn emit_zero_fill_bytes_zero_total_is_noop() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let before = prog.instrs.len();
        emit_zero_fill_bytes(&mut prog, ptr, 0, SimdWidth::W256, QuantPrecision::F32).unwrap();
        assert_eq!(prog.instrs.len(), before);
    }

    #[test]
    fn emit_zero_fill_bytes_emits_broadcast_and_stores() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // 64 bytes / 32 bytes per W256 vector = 2 vec stores, no tail
        emit_zero_fill_bytes(&mut prog, ptr, 64, SimdWidth::W256, QuantPrecision::F32).unwrap();
        let has_broadcast = prog.instrs.iter().any(|i| matches!(i, VmInstr::Broadcast { .. }));
        let has_vec_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_broadcast, "expected Broadcast for zero value");
        assert!(has_vec_store, "expected VecStore instructions");
    }

    // ── lower_patch_embed validation ──

    #[test]
    fn lower_patch_embed_rejects_zero_patch_size() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_patch_embed(&mut prog, 0, 64, 3, 224, img, ker, out, QuantPrecision::F32);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid dims"), "expected 'invalid dims' in error, got: {msg}");
    }

    #[test]
    fn lower_patch_embed_rejects_indivisible_image_size() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_patch_embed(&mut prog, 14, 64, 3, 225, img, ker, out, QuantPrecision::F32);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("patch_size"), "expected patch_size mention in error, got: {msg}");
    }

    // ── lower_patch_embed: zero embed_dim ──

    // @trace TEST-VAE-01 [req:REQ-CG] [level:unit]
    #[test]
    fn lower_patch_embed_rejects_zero_embed_dim() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_patch_embed(&mut prog, 14, 0, 3, 224, img, ker, out, QuantPrecision::F32);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid dims"), "expected 'invalid dims' for zero embed_dim, got: {msg}");
    }

    // ── lower_patch_embed: zero in_channels ──

    // @trace TEST-VAE-02 [req:REQ-CG] [level:unit]
    #[test]
    fn lower_patch_embed_rejects_zero_in_channels() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_patch_embed(&mut prog, 14, 64, 0, 224, img, ker, out, QuantPrecision::F32);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid dims"), "expected 'invalid dims' for zero in_channels, got: {msg}");
    }

    // ── lower_patch_embed: zero image_size ──

    // @trace TEST-VAE-03 [req:REQ-CG] [level:unit]
    #[test]
    fn lower_patch_embed_rejects_zero_image_size() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_patch_embed(&mut prog, 14, 64, 3, 0, img, ker, out, QuantPrecision::F32);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid dims"), "expected 'invalid dims' for zero image_size, got: {msg}");
    }

    // ── lower_patch_embed: valid smallest case produces well-formed program ──

    // @trace TEST-VAE-04 [req:REQ-CG] [level:unit]
    #[test]
    fn lower_patch_embed_smallest_valid_produces_valid_program() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // patch_size=2, embed_dim=4, in_channels=1, image_size=2 → 1 patch, 4 output elements
        lower_patch_embed(&mut prog, 2, 4, 1, 2, img, ker, out, QuantPrecision::F32).unwrap();
        prog.validate_structure().expect("structure should be valid");
        prog.validate_provenance().expect("provenance should be valid");
        assert!(!prog.instrs.is_empty(), "expected instructions to be emitted");
    }

    // ── lower_patch_embed: patch_size == image_size (single patch) ──

    // @trace TEST-VAE-05 [req:REQ-CG] [level:unit]
    #[test]
    fn lower_patch_embed_patch_equals_image_single_patch() {
        let mut prog = VmProgram::new();
        let img = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ker = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // patch_size=4, image_size=4 → num_patches_side=1, single patch
        lower_patch_embed(&mut prog, 4, 8, 3, 4, img, ker, out, QuantPrecision::F32).unwrap();
        prog.validate_structure().expect("structure should be valid");
        // Should have 5 nested loops (p_row, p_col, e, c, kr)
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_begins, 5, "expected 5 nested loops for p_row/p_col/e/c/kr");
    }

    // ── emit_zero_fill_bytes: non-aligned total produces tail ──

    // @trace TEST-VAE-06 [req:REQ-CG] [level:unit]
    #[test]
    fn emit_zero_fill_bytes_non_aligned_produces_vec_and_scalar_tail() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // 100 bytes / 32 bytes per W256 = 3 vec stores + 4 bytes tail (1 scalar store)
        emit_zero_fill_bytes(&mut prog, ptr, 100, SimdWidth::W256, QuantPrecision::F32).unwrap();
        let vec_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::W256, .. })
        }).count();
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(vec_stores >= 1, "expected at least one W256 VecStore");
        assert!(scalar_stores >= 1, "expected at least one Scalar VecStore for tail");
    }

    // ── emit_zero_fill_bytes: exact SIMD-aligned total has no tail ──

    // @trace TEST-VAE-07 [req:REQ-CG] [level:unit]
    #[test]
    fn emit_zero_fill_bytes_aligned_total_has_no_scalar_tail() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // 32 bytes exactly = 1 × W256, no tail
        emit_zero_fill_bytes(&mut prog, ptr, 32, SimdWidth::W256, QuantPrecision::F32).unwrap();
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert_eq!(scalar_stores, 0, "expected no scalar tail stores for aligned total");
    }

    // ── emit_zero_fill_bytes: Scalar width path ──

    // @trace TEST-VAE-08 [req:REQ-CG] [level:unit]
    #[test]
    fn emit_zero_fill_bytes_scalar_width() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // 12 bytes with Scalar (4B per element) = 3 scalar stores, no vec path
        emit_zero_fill_bytes(&mut prog, ptr, 12, SimdWidth::Scalar, QuantPrecision::F32).unwrap();
        let stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(stores > 0, "expected VecStore instructions for scalar fill");
    }

    // ── emit_row_copy: basic row copy emits correct structure ──

    // @trace TEST-VAE-09 [req:REQ-CG] [level:unit]
    #[test]
    fn emit_row_copy_emits_loop_and_copies() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // 2 rows × 8 channels (W256-aligned), no tail
        emit_row_copy(&mut prog, src, dst, BoundExpr::Const(2), 8, SimdWidth::W256, QuantPrecision::F32).unwrap();
        prog.validate_structure().expect("structure should be valid");
        let loads = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecLoad { .. })).count();
        let stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(loads > 0, "expected VecLoad instructions in row copy");
        assert!(stores > 0, "expected VecStore instructions in row copy");
        assert_eq!(loads, stores, "VecLoad and VecStore count should match");
    }

    // ── emit_row_copy: inner not SIMD-aligned produces tail ──

    // @trace TEST-VAE-10 [req:REQ-CG] [level:unit]
    #[test]
    fn emit_row_copy_non_aligned_inner_produces_scalar_tail() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // 5 channels with W256 (8 lanes): 0 vec stores, 5 scalar tail stores per row
        emit_row_copy(&mut prog, src, dst, BoundExpr::Const(1), 5, SimdWidth::W256, QuantPrecision::F32).unwrap();
        prog.validate_structure().expect("structure should be valid");
        let scalar_stores = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_stores > 0, "expected scalar tail stores for non-aligned inner");
    }

    // ── is_zero_offset: Mul with non-zero scale returns false ──

    // @trace TEST-VAE-11 [req:REQ-CG] [level:unit]
    #[test]
    fn is_zero_offset_mul_nonzero_scale_returns_false() {
        let vreg = VRegId(7);
        assert!(!is_zero_offset(&OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(vreg)), 16)));
    }

    // ── is_zero_offset: Add branch returns false (non-zero, non-Mul-zero) ──

    // @trace TEST-VAE-12 [req:REQ-CG] [level:unit]
    #[test]
    fn is_zero_offset_add_returns_false() {
        let expr = OffsetExpr::Add(Box::new(OffsetExpr::Const(1)), Box::new(OffsetExpr::Const(2)));
        assert!(!is_zero_offset(&expr));
    }

    // ── add_offsets: mixed Mul and Const produces correct left-nested Add ──

    // @trace TEST-VAE-13 [req:REQ-CG] [level:unit]
    #[test]
    fn add_offsets_mixed_mul_and_const_produces_left_nested_add() {
        let v1 = VRegId(1);
        let v2 = VRegId(2);
        let parts = vec![
            OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(v1)), 56),
            OffsetExpr::Const(8),
            OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(v2)), 7),
        ];
        let result = add_offsets(parts);
        // Expected: ((Mul(v1, 56) + Const(8)) + Mul(v2, 7)) — left-nested
        let expected = OffsetExpr::Add(
            Box::new(OffsetExpr::Add(
                Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(v1)), 56)),
                Box::new(OffsetExpr::Const(8)),
            )),
            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(v2)), 7)),
        );
        assert_eq!(result, expected);
    }

    // ── PleScratchRequirement: construction and trait derivation ──

    // @trace TEST-VAE-14 [req:REQ-CG] [level:unit]
    #[test]
    fn ple_scratch_requirement_construction_and_equality() {
        let a = PleScratchRequirement {
            ctx_offset: 1024,
            post_mlp_offset: 8192,
            total_bytes: 16384,
            max_seq_len: 512,
            dim_per_layer: 256,
            hidden: 768,
        };
        assert_eq!(a.ctx_offset, 1024);
        assert_eq!(a.post_mlp_offset, 8192);
        assert_eq!(a.total_bytes, 16384);
        assert_eq!(a.max_seq_len, 512);
        assert_eq!(a.dim_per_layer, 256);
        assert_eq!(a.hidden, 768);

        let b = a.clone();
        assert_eq!(a, b, "PleScratchRequirement PartialEq via derived Clone");
    }

    // @trace TEST-VAE-15 [req:REQ-CG] [level:unit]
    #[test]
    fn ple_scratch_requirement_debug_format() {
        let req = PleScratchRequirement {
            ctx_offset: 0,
            post_mlp_offset: 4096,
            total_bytes: 8192,
            max_seq_len: 256,
            dim_per_layer: 128,
            hidden: 512,
        };
        let debug_str = format!("{:?}", req);
        assert!(debug_str.contains("ctx_offset"), "Debug should contain ctx_offset");
        assert!(debug_str.contains("post_mlp_offset"), "Debug should contain post_mlp_offset");
        assert!(debug_str.contains("total_bytes"), "Debug should contain total_bytes");
    }

    // ── DwcScratchRequirement: construction and trait derivation ──

    // @trace TEST-VAE-16 [req:REQ-CG] [level:unit]
    #[test]
    fn dwc_scratch_requirement_construction_and_equality() {
        let a = DwcScratchRequirement {
            padded_offset: 2048,
            total_bytes: 32768,
            max_seq_len: 512,
            channels: 64,
            kernel_size: 7,
            causal: true,
            left_pad: 6,
        };
        assert_eq!(a.padded_offset, 2048);
        assert_eq!(a.total_bytes, 32768);
        assert_eq!(a.max_seq_len, 512);
        assert_eq!(a.channels, 64);
        assert_eq!(a.kernel_size, 7);
        assert!(a.causal);
        assert_eq!(a.left_pad, 6);

        let b = a.clone();
        assert_eq!(a, b, "DwcScratchRequirement PartialEq via derived Clone");
    }

    // @trace TEST-VAE-17 [req:REQ-CG] [level:unit]
    #[test]
    fn dwc_scratch_requirement_debug_format() {
        let req = DwcScratchRequirement {
            padded_offset: 0,
            total_bytes: 1024,
            max_seq_len: 128,
            channels: 32,
            kernel_size: 3,
            causal: false,
            left_pad: 1,
        };
        let debug_str = format!("{:?}", req);
        assert!(debug_str.contains("padded_offset"), "Debug should contain padded_offset");
        assert!(debug_str.contains("kernel_size"), "Debug should contain kernel_size");
        assert!(debug_str.contains("causal"), "Debug should contain causal");
    }

}
