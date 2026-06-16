/// 统一参数化 quant GEMM — 所有参数从 QuantGemmPlan 自动推导。
pub(crate) fn emit_quant_gemm_tiled(
    prog: &mut VmProgram,
    m_bound: BoundExpr,
    n: usize,
    k: usize,
    desc: &crate::quant_format::QuantFormatDescriptor,
    width: SimdWidth,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
    dot_cap: DotProductCap,
) -> Result<(), CompilerError> {
    let plan = QuantGemmPlan::derive(m_bound, n, k, desc, width, dtype, dot_cap)?;

    match (&plan.mode, &plan.kernel) {
        (GemmMode::Float, GemmKernel::Float) => {
            emit_gemm_float_from_plan(prog, &plan, input_ptr, weight_ptr, output_ptr, desc)
        }
        (_, GemmKernel::Int8Native { scale_offset, data_offset }) => {
            emit_gemm_int8_from_plan(prog, &plan, input_ptr, weight_ptr, output_ptr,
                *scale_offset, *data_offset)
        }
        (_, GemmKernel::Assisted { scale_offset, data_offset }) => {
            emit_gemm_assisted_from_plan(prog, &plan, input_ptr, weight_ptr, output_ptr, desc,
                *scale_offset, *data_offset)
        }
        (_, GemmKernel::DequantFma) => {
            emit_gemm_dequant_from_plan(prog, &plan, input_ptr, weight_ptr, output_ptr, desc)
        }
        (_, GemmKernel::HighBitMerge { scale_offset, low_offset, high_offset, bias, high_bits }) => {
            emit_gemm_highbit_from_plan(prog, &plan, input_ptr, weight_ptr, output_ptr, desc,
                *scale_offset, *low_offset, *high_offset, *bias, *high_bits)
        }
        _ => Err(CompilerError::CodegenViolation("quant_gemm: inconsistent mode/kernel".into())),
    }
}

/// 浮点路径 GEMM — 无量化块结构，直接 k/lanes 迭代。
pub(crate) fn emit_gemm_float_from_plan(
    prog: &mut VmProgram,
    plan: &QuantGemmPlan,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    desc: &crate::quant_format::QuantFormatDescriptor,
) -> Result<(), CompilerError> {
    let QuantGemmPlan { lanes, elem, n, k, width, dtype, a_row_stride, c_row_stride, .. } = *plan;
    let bytes_per_elem = desc.block_bytes;
    let num_k_iters = k / lanes;
    if k % lanes != 0 {
        return Err(CompilerError::CodegenViolation(
            format!("gemm_float: k={} not divisible by lanes={}", k, lanes)
        ));
    }

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let a_val = prog.alloc_vreg(VRegKind::Vec, width);
    let b_val = prog.alloc_vreg(VRegKind::Vec, width);
    let hreduce_dst = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });

    let w_row_stride_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let k_block_stride_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let act_k_stride_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: w_row_stride_gpr, value: k * bytes_per_elem });
    prog.emit(VmInstr::GprLoadImm { dst: k_block_stride_gpr, value: lanes * bytes_per_elem });
    prog.emit(VmInstr::GprLoadImm { dst: act_k_stride_gpr, value: lanes * elem });

    prog.emit_loop_try(plan.m_bound.clone(), 1, |prog, _i_ctr, i_cnt| -> Result<(), CompilerError> {
        let w_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: w_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

        prog.emit_loop_try(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| -> Result<(), CompilerError> {
            prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
            let w_col_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: w_col_ptr, a: w_row_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });
            let k_act_off = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: k_act_off, value: 0 });

            prog.emit_loop_try(BoundExpr::Const(num_k_iters), lanes * bytes_per_elem, |prog, _kk_ctr, kk_off| -> Result<(), CompilerError> {
                let act_off = OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), a_row_stride)),
                    Box::new(OffsetExpr::ScalarVReg(k_act_off)),
                );
                prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: act_off, width, dtype , predicate: None });
                prog.emit(VmInstr::VecLoad { dst: b_val, base: w_col_ptr, offset: OffsetExpr::LoopOffset(kk_off), width, dtype , predicate: None });

                match desc.data_kind {
                    crate::quant_format::QuantDataKind::Bfloat16 => {
                        prog.emit(VmInstr::DotProduct { acc, a: a_val, b: b_val, input_dtype: DotDtype::Bf16, width });
                    }
                    crate::quant_format::QuantDataKind::Float16 => {
                        prog.emit(VmInstr::DotProduct { acc, a: a_val, b: b_val, input_dtype: DotDtype::Fp16, width });
                    }
                    _ => {
                        prog.emit(VmInstr::Fma { dst: acc, acc, a: a_val, b: b_val, dtype });
                    }
                }

                prog.emit(VmInstr::GprBinOp { dst: k_act_off, a: k_act_off, b: GprOperand::VReg(act_k_stride_gpr ), op: GprOp::Add });
                prog.emit(VmInstr::GprBinOp { dst: w_col_ptr, a: w_col_ptr, b: GprOperand::VReg(k_block_stride_gpr ), op: GprOp::Add });
                Ok(())
            })?;

            prog.emit(VmInstr::HReduce { dst: hreduce_dst, src: acc, op: super::instr::ReduceOp::Sum });
            prog.emit(VmInstr::VecScalarStore {
                base: output_ptr, src: hreduce_dst,
                offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), c_row_stride)),
                    Box::new(OffsetExpr::LoopOffset(j_off)),
                ),
            });
            prog.emit(VmInstr::GprBinOp { dst: w_row_ptr, a: w_row_ptr, b: GprOperand::VReg(w_row_stride_gpr ), op: GprOp::Add });
            Ok(())
        })
    })
}

/// INT8 硬件原生路径 — Int8Load + DotProduct(Int8) + ScaleApply。
pub(crate) fn emit_gemm_int8_from_plan(
    prog: &mut VmProgram,
    plan: &QuantGemmPlan,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    scale_offset: usize,
    data_offset: usize,
) -> Result<(), CompilerError> {
    let QuantGemmPlan {
        n, lanes, elem, width, dtype,
        block_bytes, gguf_num_blocks, iters_per_block,
        a_row_stride, c_row_stride, ..
    } = *plan;
    let quant_row_stride = plan.quant_row_stride;

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let a_val = prog.alloc_vreg(VRegKind::Vec, width);
    let w_raw = prog.alloc_vreg(VRegKind::Vec, width);
    let scale_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let hreduce_dst = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });
    prog.emit(VmInstr::Broadcast { dst: zero_vec, src: ScalarExpr::Const(0.0), width, dtype });

    let blk_ptr_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let j_weight_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let act_stride_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let data_step_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: blk_ptr_stride, value: block_bytes });
    prog.emit(VmInstr::GprLoadImm { dst: j_weight_stride, value: quant_row_stride });
    prog.emit(VmInstr::GprLoadImm { dst: act_stride_reg, value: plan.block_size * elem });
    prog.emit(VmInstr::GprLoadImm { dst: data_step_reg, value: plan.data_step.max(1) });

    let k_act_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    prog.emit_loop_try(plan.m_bound.clone(), 1, |prog, _i_ctr, i_cnt| -> Result<(), CompilerError> {
        let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

        prog.emit_loop_try(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| -> Result<(), CompilerError> {
            prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
            prog.emit(VmInstr::Broadcast { dst: zero_vec, src: ScalarExpr::Const(0.0), width, dtype });
            prog.emit(VmInstr::GprLoadImm { dst: k_act_base, value: 0 });

            let blk_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: weight_row_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

            prog.emit_loop_try(BoundExpr::Const(gguf_num_blocks), block_bytes,
                |prog, _blk_ctr, _blk_off| -> Result<(), CompilerError> {
                    prog.emit(VmInstr::QuantBlockLoad {
                        dst: scale_vec, base: blk_ptr,
                        offset: OffsetExpr::Const(scale_offset), unpack: BlockUnpackMode::F16Broadcast, width,
                    });
                    let data_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    if data_offset > 0 {
                        let off_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprLoadImm { dst: off_reg, value: data_offset });
                        prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: blk_ptr, b: GprOperand::VReg(off_reg ), op: GprOp::Add });
                    } else {
                        prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: blk_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });
                    }

                    prog.emit_loop_try(BoundExpr::Const(iters_per_block), lanes * elem,
                        |prog, _ei_ctr, ei_off| -> Result<(), CompilerError> {
                            let act_off = OffsetExpr::Add(
                                Box::new(OffsetExpr::Add(
                                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), a_row_stride)),
                                    Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                )),
                                Box::new(OffsetExpr::LoopOffset(ei_off)),
                            );
                            prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: act_off, width, dtype , predicate: None });
                            prog.emit(VmInstr::QuantBlockLoad { dst: w_raw, base: data_ptr, offset: OffsetExpr::Const(0), unpack: BlockUnpackMode::Int8, width });
                            prog.emit(VmInstr::DotProduct { acc, a: a_val, b: w_raw, input_dtype: DotDtype::Int8, width });
                            prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: data_ptr, b: GprOperand::VReg(data_step_reg ), op: GprOp::Add });
                            Ok(())
                        },
                    )?;

                    prog.emit(VmInstr::ScaleApply { dst: acc, acc, scale: scale_vec, zero: zero_vec, input_dtype: QuantPrecision::INT8, width });
                    prog.emit(VmInstr::GprBinOp { dst: k_act_base, a: k_act_base, b: GprOperand::VReg(act_stride_reg ), op: GprOp::Add });
                    prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: blk_ptr, b: GprOperand::VReg(blk_ptr_stride ), op: GprOp::Add });
                    Ok(())
                },
            )?;

            prog.emit(VmInstr::HReduce { dst: hreduce_dst, src: acc, op: super::instr::ReduceOp::Sum });
            prog.emit(VmInstr::VecScalarStore {
                base: output_ptr, src: hreduce_dst,
                offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), c_row_stride)),
                    Box::new(OffsetExpr::LoopOffset(j_off)),
                ),
            });
            prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_row_ptr, b: GprOperand::VReg(j_weight_stride ), op: GprOp::Add });
            Ok(())
        })
    })
}

/// Assisted 半硬件辅助 GEMM — 寄存器内 INT4 nibble unpack + dequant + FMA (REQ-QCG5)
///
/// 适用于 SimdAssisted/SimdBasic dot_cap + INT4 weight-only:
///   1. QuantBlockLoad(SignedNibbleLow/UnsignedNibbleLow) → lo nibbles dequant
///   2. FMA(act_lo, weight_lo * scale, acc)
///   3. QuantBlockLoad(SignedNibbleHigh/UnsignedNibbleHigh) → hi nibbles dequant
///   4. FMA(act_hi, weight_hi * scale, acc)
///
/// 每个 GGUF 字节存储两个值: lo nibble = block_pos[i], hi nibble = block_pos[i+16].
/// 两次独立 FMA 分别处理 lo/hi nibble 位置，激活偏移相差 block_size/2.
///
/// 相比 DequantFma: 避免 DecodeTraceBuilder 生成的完整反量化 trace，
/// 直接在寄存器层面完成 nibble 提取 + 反量化，延迟更低。
pub(crate) fn emit_gemm_assisted_from_plan(
    prog: &mut VmProgram,
    plan: &QuantGemmPlan,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    desc: &crate::quant_format::QuantFormatDescriptor,
    scale_offset: usize,
    data_offset: usize,
) -> Result<(), CompilerError> {
    let QuantGemmPlan {
        n, lanes, elem, width, dtype,
        block_bytes, block_size,
        gguf_num_blocks, iters_per_block,
        a_row_stride, c_row_stride,
        mode, ..
    } = *plan;
    let quant_row_stride = plan.quant_row_stride;
    let use_signed = matches!(desc.data_kind, crate::quant_format::QuantDataKind::SignedPackedInt4);

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let a_val = prog.alloc_vreg(VRegKind::Vec, width);
    let b_lo = prog.alloc_vreg(VRegKind::Vec, width);
    let b_hi = prog.alloc_vreg(VRegKind::Vec, width);
    let scale_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let hreduce_dst = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });
    prog.emit(VmInstr::Broadcast { dst: zero_vec, src: ScalarExpr::Const(0.0), width, dtype });

    let blk_ptr_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let j_weight_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let act_stride_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let data_step_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: blk_ptr_stride, value: block_bytes });
    prog.emit(VmInstr::GprLoadImm { dst: j_weight_stride, value: quant_row_stride });
    prog.emit(VmInstr::GprLoadImm { dst: act_stride_reg, value: block_size * elem });
    // Assisted nibble path: each iteration loads `lanes` bytes for lo+hi nibble decode
    prog.emit(VmInstr::GprLoadImm { dst: data_step_reg, value: lanes });

    let k_act_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // ── 通用循环模板 ─────────────────────────────────────────────────
    // 与 emit_gemm_int8_from_plan / emit_gemm_dequant_from_plan 保持一致的循环结构

    let do_m_block = |prog: &mut VmProgram, i_cnt, weight_row_ptr| -> Result<(), CompilerError> {
        prog.emit_loop_try(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| -> Result<(), CompilerError> {
            prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
            prog.emit(VmInstr::Broadcast { dst: zero_vec, src: ScalarExpr::Const(0.0), width, dtype });
            prog.emit(VmInstr::GprLoadImm { dst: k_act_base, value: 0 });

            let blk_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: weight_row_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

            prog.emit_loop_try(BoundExpr::Const(gguf_num_blocks), block_bytes,
                |prog, _blk_ctr, _blk_off| -> Result<(), CompilerError> {
                    // 1. 加载 scale (f16) → 广播到向量
                    prog.emit(VmInstr::QuantBlockLoad {
                        dst: scale_vec, base: blk_ptr,
                        offset: OffsetExpr::Const(scale_offset), unpack: BlockUnpackMode::F16Broadcast, width,
                    });
                    let data_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    if data_offset > 0 {
                        let off_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprLoadImm { dst: off_reg, value: data_offset });
                        prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: blk_ptr, b: GprOperand::VReg(off_reg ), op: GprOp::Add });
                    } else {
                        prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: blk_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });
                    }

                    // ei loop: each iteration processes `lanes` low-nibble positions +
                    // `lanes` high-nibble positions from the same packed bytes.
                    // GGUF Q4_0 layout: byte_i → lo_nibble = block_pos[i], hi_nibble = block_pos[16+i].
                    // We must do TWO separate FMA operations with DIFFERENT activation offsets.
                    // Nibble packing: each byte holds 2 values, so `lanes` values need only lanes/2 bytes.
                    // But SignedNibbleLow loads 8 bytes → 8 values. So we process lanes lo + lanes hi
                    // per iteration = 2*lanes block positions, needing only iters_per_block/2 iterations.
                    let nibble_iters = iters_per_block / 2;
                    let half_block_elem = block_size / 2 * elem; // offset to hi-nibble activations
                    prog.emit_loop_try(BoundExpr::Const(nibble_iters), lanes * elem,
                        |prog, _ei_ctr, ei_off| -> Result<(), CompilerError> {
                            // --- Low nibble FMA: lo nibbles = block positions [0..7] ---
                            let lo_act_off = OffsetExpr::Add(
                                Box::new(OffsetExpr::Add(
                                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), a_row_stride)),
                                    Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                )),
                                Box::new(OffsetExpr::LoopOffset(ei_off)),
                            );
                            prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: lo_act_off, width, dtype , predicate: None });

                            let low_unpack = if use_signed {
                                BlockUnpackMode::SignedNibbleLow
                            } else {
                                BlockUnpackMode::UnsignedNibbleLow
                            };
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: b_lo, base: data_ptr,
                                offset: OffsetExpr::Const(0), unpack: low_unpack, width,
                            });
                            prog.emit(VmInstr::VecBinOp { dst: b_lo, a: b_lo, b: scale_vec, op: VecOp::Mul, dtype: dtype });
                            prog.emit(VmInstr::Fma { dst: acc, acc, a: a_val, b: b_lo, dtype });

                            // --- High nibble FMA: hi nibbles = block positions [16..23] ---
                            let hi_act_off = OffsetExpr::Add(
                                Box::new(OffsetExpr::Add(
                                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), a_row_stride)),
                                    Box::new(OffsetExpr::Add(
                                        Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                        Box::new(OffsetExpr::Const(half_block_elem)),
                                    )),
                                )),
                                Box::new(OffsetExpr::LoopOffset(ei_off)),
                            );
                            prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: hi_act_off, width, dtype , predicate: None });

                            let high_unpack = if use_signed {
                                BlockUnpackMode::SignedNibbleHigh
                            } else {
                                BlockUnpackMode::UnsignedNibbleHigh
                            };
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: b_hi, base: data_ptr,
                                offset: OffsetExpr::Const(0), unpack: high_unpack, width,
                            });
                            prog.emit(VmInstr::VecBinOp { dst: b_hi, a: b_hi, b: scale_vec, op: VecOp::Mul, dtype: dtype });
                            prog.emit(VmInstr::Fma { dst: acc, acc, a: a_val, b: b_hi, dtype });

                            // Advance data_ptr
                            prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: data_ptr, b: GprOperand::VReg(data_step_reg ), op: GprOp::Add });
                            Ok(())
                        },
                    )?;

                    prog.emit(VmInstr::GprBinOp { dst: k_act_base, a: k_act_base, b: GprOperand::VReg(act_stride_reg ), op: GprOp::Add });
                    prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: blk_ptr, b: GprOperand::VReg(blk_ptr_stride ), op: GprOp::Add });
                    Ok(())
                },
            )?;

            prog.emit(VmInstr::HReduce { dst: hreduce_dst, src: acc, op: super::instr::ReduceOp::Sum });
            prog.emit(VmInstr::VecScalarStore {
                base: output_ptr, src: hreduce_dst,
                offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), c_row_stride)),
                    Box::new(OffsetExpr::LoopOffset(j_off)),
                ),
            });
            prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_row_ptr, b: GprOperand::VReg(j_weight_stride ), op: GprOp::Add });
            Ok(())
        })
    };

    match mode {
        GemmMode::Gemv => {
            let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let zero_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: zero_offset, value: 0 });
            prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });
            // M=1: i_cnt = zero_offset (0 × a_row_stride = 0)
            do_m_block(prog, zero_offset, weight_row_ptr)?;
        }
        GemmMode::General => {
            prog.emit_loop_try(plan.m_bound.clone(), 1, |prog, _i_ctr, i_cnt| -> Result<(), CompilerError> {
                let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });
                do_m_block(prog, i_cnt, weight_row_ptr)
            })?;
        }
        GemmMode::Float => unreachable!("Float mode uses emit_gemm_float_from_plan"),
    }

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §QCG-006 HighBitMerge — INT5/INT6 路径 (Q5_0/Q5_1/Q5_K/Q6_K)
//
// 使用 QuantBiPlaneLoad(Low5/Low6) 在 VmInstr 层面完成:
//   1. 加载低 4-bit nibbles (qs)
//   2. 加载高 bit 平面 (qh/hmask)
//   3. 合并为完整的 5-bit/6-bit 值
//   4. 减去静态偏置 (bias)
//   输出直接为 F32 向量。
//
// 然后乘以 scale + FMA 累加:
//   acc += activation * (biplane_value * scale)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// INT5/INT6 HighBitMerge GEMM — QuantBiPlaneLoad + scale + FMA (REQ-QCG-006).
///
/// 适用于 `PackedInt5` / `PackedInt6` data_kind (Q5_0, Q5_1, Q5_K, Q6_K):
///   1. QuantBiPlaneLoad(Low5/Low6) → 合并 nibble + high-bit → F32 (含 bias 减法)
///   2. 乘以 scale (VecBinOp::Mul)
///   3. FMA 累加: acc += act * scaled_weight
pub(crate) fn emit_gemm_highbit_from_plan(
    prog: &mut VmProgram,
    plan: &QuantGemmPlan,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    desc: &crate::quant_format::QuantFormatDescriptor,
    scale_offset: usize,
    low_offset: usize,
    high_offset: usize,
    bias: f32,
    high_bits: u8,
) -> Result<(), CompilerError> {
    let QuantGemmPlan {
        n, lanes, elem, width, dtype,
        block_bytes, block_size,
        gguf_num_blocks, iters_per_block,
        a_row_stride, c_row_stride,
        mode, ..
    } = *plan;
    let quant_row_stride = plan.quant_row_stride;

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let a_val = prog.alloc_vreg(VRegKind::Vec, width);
    let nibble_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let qh_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let w_merged = prog.alloc_vreg(VRegKind::Vec, width);
    let scale_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let bias_vec = prog.alloc_vreg(VRegKind::Vec, width);
    let hreduce_dst = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });
    prog.emit(VmInstr::Broadcast { dst: bias_vec, src: ScalarExpr::Const(bias), width, dtype });

    let blk_ptr_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let j_weight_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let act_stride_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: blk_ptr_stride, value: block_bytes });
    prog.emit(VmInstr::GprLoadImm { dst: j_weight_stride, value: quant_row_stride });
    prog.emit(VmInstr::GprLoadImm { dst: act_stride_reg, value: block_size * elem });

    let k_act_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // INT5/INT6 nibble packing: each byte stores 2 elements (low nibble = elem i, high nibble = elem i+half).
    // Two-phase approach (same as Assisted Q4_0):
    //   Phase A: low nibbles → sequential elements [0..7], activation at current offset
    //   Phase B: high nibbles → sequential elements [half..half+7], activation at +half_block offset
    // Each phase also merges high-bit-plane (qh) values.
    let nibble_iters = iters_per_block / 2;
    let half_block_elem = block_size / 2 * elem;

    // qh bit value: for INT5, bit_value=16 (bit 4); for INT6, bit_value=16 (bit 4) per bit
    let qh_bit_value = (1u32 << (high_bits as u32 * 4)) as f32; // 1<<4 = 16 for INT5

    let do_m_block = |prog: &mut VmProgram, i_cnt, weight_row_ptr| -> Result<(), CompilerError> {
        prog.emit_loop_try(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| -> Result<(), CompilerError> {
            prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
            prog.emit(VmInstr::GprLoadImm { dst: k_act_base, value: 0 });

            let blk_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: weight_row_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });

            prog.emit_loop_try(BoundExpr::Const(gguf_num_blocks), block_bytes,
                |prog, _blk_ctr, _blk_off| -> Result<(), CompilerError> {
                    // 1. Load scale
                    match &desc.scale_layout {
                        crate::quant_format::ScaleLayout::BlockScalar { .. } => {
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: scale_vec, base: blk_ptr,
                                offset: OffsetExpr::Const(scale_offset),
                                unpack: BlockUnpackMode::F16Broadcast, width,
                            });
                        }
                        crate::quant_format::ScaleLayout::BlockScalarWithMin { d_offset, m_offset, .. } => {
                            let d_vec = prog.alloc_vreg(VRegKind::Vec, width);
                            let m_vec = prog.alloc_vreg(VRegKind::Vec, width);
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: d_vec, base: blk_ptr,
                                offset: OffsetExpr::Const(*d_offset),
                                unpack: BlockUnpackMode::F16Broadcast, width,
                            });
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: m_vec, base: blk_ptr,
                                offset: OffsetExpr::Const(*m_offset),
                                unpack: BlockUnpackMode::F16Broadcast, width,
                            });
                            prog.emit(VmInstr::VecBinOp { dst: scale_vec, a: d_vec, b: m_vec, op: VecOp::Add, dtype: dtype });
                        }
                        _ => unreachable!(
                            "emit_gemm_highbit: Hierarchical/Q6KScales/ExternalArray/SubBlockScalars \
                             should use DequantFma path"
                        ),
                    }

                    // 2. Build qs_ptr and qh_ptr
                    let qs_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    if low_offset > 0 {
                        let off_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprLoadImm { dst: off_reg, value: low_offset });
                        prog.emit(VmInstr::GprBinOp { dst: qs_ptr, a: blk_ptr, b: GprOperand::VReg(off_reg), op: GprOp::Add });
                    } else {
                        prog.emit(VmInstr::GprBinOp { dst: qs_ptr, a: blk_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });
                    }

                    let qh_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    if high_offset > 0 {
                        let off_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprLoadImm { dst: off_reg, value: high_offset });
                        prog.emit(VmInstr::GprBinOp { dst: qh_ptr, a: blk_ptr, b: GprOperand::VReg(off_reg), op: GprOp::Add });
                    } else {
                        prog.emit(VmInstr::GprBinOp { dst: qh_ptr, a: blk_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });
                    }

                    // 3. Inner loop: two-phase FMA per iteration
                    //    Each iteration processes `lanes` low-nibble elements + `lanes` high-nibble elements
                    //    = 2*lanes block positions from `lanes` packed bytes.
                    prog.emit_loop_try(BoundExpr::Const(nibble_iters), lanes * elem,
                        |prog, _ei_ctr, ei_off| -> Result<(), CompilerError> {
                            // ── Phase A: Low nibbles ──
                            let lo_act_off = OffsetExpr::Add(
                                Box::new(OffsetExpr::Add(
                                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), a_row_stride)),
                                    Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                )),
                                Box::new(OffsetExpr::LoopOffset(ei_off)),
                            );
                            prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: lo_act_off, width, dtype , predicate: None });

                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: nibble_vec, base: qs_ptr,
                                offset: OffsetExpr::Const(0),
                                unpack: BlockUnpackMode::UnsignedNibbleLow, width,
                            });
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: qh_vec, base: qh_ptr,
                                offset: OffsetExpr::Const(0),
                                unpack: BlockUnpackMode::QhBitExpand { bit_value: qh_bit_value }, width,
                            });
                            // Merge: nibble + qh → INT5 value
                            prog.emit(VmInstr::VecBinOp { dst: w_merged, a: nibble_vec, b: qh_vec, op: VecOp::Add, dtype: dtype });
                            // Subtract bias
                            prog.emit(VmInstr::VecBinOp { dst: w_merged, a: w_merged, b: bias_vec, op: VecOp::Sub, dtype: dtype });
                            // Scale
                            prog.emit(VmInstr::VecBinOp { dst: w_merged, a: w_merged, b: scale_vec, op: VecOp::Mul, dtype: dtype });
                            // FMA
                            prog.emit(VmInstr::Fma { dst: acc, acc, a: a_val, b: w_merged, dtype });

                            // ── Phase B: High nibbles ──
                            let hi_act_off = OffsetExpr::Add(
                                Box::new(OffsetExpr::Add(
                                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), a_row_stride)),
                                    Box::new(OffsetExpr::Add(
                                        Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                        Box::new(OffsetExpr::Const(half_block_elem)),
                                    )),
                                )),
                                Box::new(OffsetExpr::LoopOffset(ei_off)),
                            );
                            prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: hi_act_off, width, dtype , predicate: None });

                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: nibble_vec, base: qs_ptr,
                                offset: OffsetExpr::Const(0),
                                unpack: BlockUnpackMode::UnsignedNibbleHigh, width,
                            });
                            // Load next qh byte for high-nibble elements
                            // Phase A reads qh byte N (for elements 0-7 of this iteration).
                            // Phase B reads qh byte N+2 (for elements 16-23 of this iteration).
                            // Q5_0 qh layout: byte 0=bits 0-7, byte 1=bits 8-15, byte 2=bits 16-23, byte 3=bits 24-31.
                            // Low nibble elements [i*8..i*8+7] use qh byte i; high nibble elements use qh byte i+2.
                            let qh_hi_off_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                            prog.emit(VmInstr::GprLoadImm { dst: qh_hi_off_reg, value: 2usize });
                            let qh_hi_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                            prog.emit(VmInstr::GprBinOp { dst: qh_hi_ptr, a: qh_ptr, b: GprOperand::VReg(qh_hi_off_reg), op: GprOp::Add });
                            prog.emit(VmInstr::QuantBlockLoad {
                                dst: qh_vec, base: qh_hi_ptr,
                                offset: OffsetExpr::Const(0),
                                unpack: BlockUnpackMode::QhBitExpand { bit_value: qh_bit_value }, width,
                            });
                            // Merge: nibble + qh → INT5 value
                            prog.emit(VmInstr::VecBinOp { dst: w_merged, a: nibble_vec, b: qh_vec, op: VecOp::Add, dtype: dtype });
                            // Subtract bias
                            prog.emit(VmInstr::VecBinOp { dst: w_merged, a: w_merged, b: bias_vec, op: VecOp::Sub, dtype: dtype });
                            // Scale
                            prog.emit(VmInstr::VecBinOp { dst: w_merged, a: w_merged, b: scale_vec, op: VecOp::Mul, dtype: dtype });
                            // FMA
                            prog.emit(VmInstr::Fma { dst: acc, acc, a: a_val, b: w_merged, dtype });

                            // Advance qs_ptr by `lanes` bytes (8 bytes per iteration, low+high from same bytes)
                            let qs_step_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                            prog.emit(VmInstr::GprLoadImm { dst: qs_step_reg, value: lanes });
                            prog.emit(VmInstr::GprBinOp { dst: qs_ptr, a: qs_ptr, b: GprOperand::VReg(qs_step_reg), op: GprOp::Add });

                            // Advance qh_ptr by 1 byte per iteration (Phase A reads qh[N], Phase B reads qh[N+2])
                            let qh_step_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                            prog.emit(VmInstr::GprLoadImm { dst: qh_step_reg, value: 1usize });
                            prog.emit(VmInstr::GprBinOp { dst: qh_ptr, a: qh_ptr, b: GprOperand::VReg(qh_step_reg), op: GprOp::Add });
                            Ok(())
                        },
                    )?;

                    prog.emit(VmInstr::GprBinOp { dst: k_act_base, a: k_act_base, b: GprOperand::VReg(act_stride_reg), op: GprOp::Add });
                    prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: blk_ptr, b: GprOperand::VReg(blk_ptr_stride), op: GprOp::Add });
                    Ok(())
                },
            )?;

            prog.emit(VmInstr::HReduce { dst: hreduce_dst, src: acc, op: super::instr::ReduceOp::Sum });
            prog.emit(VmInstr::VecScalarStore {
                base: output_ptr, src: hreduce_dst,
                offset: OffsetExpr::Add(
                    Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), c_row_stride)),
                    Box::new(OffsetExpr::LoopOffset(j_off)),
                ),
            });
            prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_row_ptr, b: GprOperand::VReg(j_weight_stride), op: GprOp::Add });
            Ok(())
        })
    };

    match mode {
        GemmMode::Gemv => {
            let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let zero_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: zero_offset, value: 0 });
            prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });
            do_m_block(prog, zero_offset, weight_row_ptr)?;
        }
        GemmMode::General => {
            prog.emit_loop_try(plan.m_bound.clone(), 1, |prog, _i_ctr, i_cnt| -> Result<(), CompilerError> {
                let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });
                do_m_block(prog, i_cnt, weight_row_ptr)
            })?;
        }
        GemmMode::Float => unreachable!("Float mode uses emit_gemm_float_from_plan"),
    }

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §QCG6 DequantFMAPath — 纯软件反量化路径 (Level 3 universal fallback)
//
// 公式: value = (qw - zp) × scale
// 指令: vfmadd231ps (x86) / FMLA (ARM) / FFMA (GPU)
//
// 流程:
//   1. DecodeTraceBuilder 生成反量化 trace:
//      BlockLoad → Unpack → Sub(zp) → Mul(scale) → QuantDequantFma
//   2. auto_lower_trace_raw 将 trace 展开为 VmInstr 序列
//   3. 累加: Fma(acc, activation, dequant_weight) → vfmadd231ps
//
// 适用硬件: 所有平台（无特殊硬件要求）。
// 与 Assisted (QCG5) 的区别:
//   - Assisted: 寄存器内 nibble unpack + 手动 scale 广播, 延迟更低
//   - DequantFMA: DecodeTraceBuilder 完整 trace, 覆盖所有格式, 通用性最强
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// DequantFMA path — pure software dequantize-then-FMA.
///
/// Level 3 universal fallback (REQ-QCG6). Decodes any quantized block to F32
/// via [`DecodeTraceBuilder`](super::quant_decode::DecodeTraceBuilder), then
/// accumulates with `vfmadd231ps` (or equivalent FMA on ARM/GPU).
///
/// # Formula
/// ```text
/// dequant_weight = (qw - zp) × scale
/// acc += activation × dequant_weight
/// ```
///
/// # When selected
/// Chosen by [`QuantGemmPlan::derive`] when the (quant_format, dot_product_cap)
/// combination does not qualify for Native (hardware dot-product) or Assisted
/// (register-inline nibble unpack).
pub(crate) struct DequantFMAPath;

impl DequantFMAPath {
    /// Emit the DequantFMA GEMM micro-kernel for the given plan.
    ///
    /// Delegates to [`emit_gemm_dequant_from_plan`] which orchestrates:
    /// - Block-level loops over GGUF blocks
    /// - `DecodeTraceBuilder` for per-element dequant trace
    /// - `auto_lower_trace_raw` + FMA for accumulation
    pub(crate) fn emit(
        prog: &mut VmProgram,
        plan: &QuantGemmPlan,
        input_ptr: VRegId,
        weight_ptr: VRegId,
        output_ptr: VRegId,
        desc: &crate::quant_format::QuantFormatDescriptor,
    ) -> Result<(), CompilerError> {
        emit_gemm_dequant_from_plan(prog, plan, input_ptr, weight_ptr, output_ptr, desc)
    }
}

/// 通用反量化 GEMM — DecodeTraceBuilder → auto_lower_trace_raw + FMA。
/// 支持 GEMV (M=1) 和 General (M>1)，由 plan.mode 自动决定循环层数。
pub(crate) fn emit_gemm_dequant_from_plan(
    prog: &mut VmProgram,
    plan: &QuantGemmPlan,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    desc: &crate::quant_format::QuantFormatDescriptor,
) -> Result<(), CompilerError> {
    let QuantGemmPlan {
        n, lanes, elem, width, dtype,
        block_bytes, block_size,
        gguf_num_blocks, iters_per_block,
        data_step, a_row_stride, c_row_stride,
        mode, ..
    } = *plan;
    let quant_row_stride = plan.quant_row_stride;

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let a_val = prog.alloc_vreg(VRegKind::Vec, width);
    let hreduce_dst = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });

    let blk_ptr_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let j_weight_stride = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let data_stride_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let act_stride_reg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: blk_ptr_stride, value: block_bytes });
    prog.emit(VmInstr::GprLoadImm { dst: j_weight_stride, value: quant_row_stride });
    prog.emit(VmInstr::GprLoadImm { dst: data_stride_reg, value: data_step.max(1) });
    prog.emit(VmInstr::GprLoadImm { dst: act_stride_reg, value: block_size * elem });

    let k_act_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    let builder = super::quant_decode::DecodeTraceBuilder::new(desc, lanes);
    let needs_lane_off = builder.needs_lane_offset();
    let needs_high_bits = builder.needs_high_bits_ptr();
    let high_bits_stride_val = builder.high_bits_stride();
    let mut decode_trace: Vec<TraceOp> = Vec::new();
    let decode_final_slot = builder.build(&mut decode_trace);

    let lane_offset_gpr = if needs_lane_off {
        let gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: gpr, value: 0 });
        Some(gpr)
    } else {
        None
    };
    let lanes_gpr = if needs_lane_off {
        let gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: gpr, value: lanes });
        Some(gpr)
    } else {
        None
    };

    // For NibbleWithHighBits (Q6_K, Q5_0, Q5_1): separate high-bit-plane pointer.
    let high_bits_stride_gpr = if needs_high_bits {
        let gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: gpr, value: high_bits_stride_val });
        Some(gpr)
    } else {
        None
    };
    // high_offset from DataLayout::NibbleWithHighBits — used to initialize high_bits_ptr
    let high_offset_val: usize = match &desc.data_layout {
        crate::quant_format::DataLayout::NibbleWithHighBits { high_offset, .. } => *high_offset,
        _ => 0,
    };

    let fma_trace: Vec<TraceOp> = vec![
        TraceOp::Input(0),
        TraceOp::Input(1),
        TraceOp::Input(2),
        TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),
    ];

    match mode {
        GemmMode::Gemv => {
            let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

            prog.emit_loop_try(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| -> Result<(), CompilerError> {
                prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
                prog.emit(VmInstr::GprLoadImm { dst: k_act_base, value: 0 });

                let blk_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: weight_row_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

                prog.emit_loop_try(BoundExpr::Const(gguf_num_blocks), block_bytes,
                    |prog, _blk_ctr, _blk_off| -> Result<(), CompilerError> {
                        if let Some(lo) = lane_offset_gpr {
                            prog.emit(VmInstr::GprLoadImm { dst: lo, value: 0 });
                        }
                        let data_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: blk_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

                        // high_bits_ptr: independently advanced pointer for NibbleWithHighBits high plane.
                        let high_bits_ptr = if needs_high_bits {
                            let hbp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                            prog.emit(VmInstr::AddPtr { dst: hbp, base: blk_ptr, offset: high_offset_val });
                            Some(hbp)
                        } else {
                            None
                        };

                        prog.emit_loop_try(BoundExpr::Const(iters_per_block), lanes * elem,
                            |prog, _ei_ctr, ei_off| -> Result<(), CompilerError> {
                                let mut decode_inputs: Vec<VRegId> = if let Some(lo) = lane_offset_gpr {
                                    vec![blk_ptr, data_ptr, lo]
                                } else {
                                    vec![blk_ptr, data_ptr]
                                };
                                if let Some(hbp) = high_bits_ptr {
                                    decode_inputs.push(hbp);
                                }
                                let decode_slots = super::auto_select::auto_lower_trace_raw(
                                    prog, &decode_trace, &decode_inputs, width, dtype)?;
                                let b_decoded = decode_slots[decode_final_slot.0 as usize];

                                let act_off = OffsetExpr::Add(
                                    Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                    Box::new(OffsetExpr::LoopOffset(ei_off)),
                                );
                                prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: act_off, width, dtype , predicate: None });

                                super::auto_select::auto_lower_trace_into(
                                    prog, &fma_trace, &[a_val, b_decoded, acc], acc, width, dtype,
                                )?;

                                prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: data_ptr, b: GprOperand::VReg(data_stride_reg ), op: GprOp::Add });
                                if let (Some(lo), Some(lg)) = (lane_offset_gpr, lanes_gpr) {
                                    prog.emit(VmInstr::GprBinOp { dst: lo, a: lo, b: GprOperand::VReg(lg ), op: GprOp::Add });
                                }
                                if let (Some(hbp), Some(hbs)) = (high_bits_ptr, high_bits_stride_gpr) {
                                    prog.emit(VmInstr::GprBinOp { dst: hbp, a: hbp, b: GprOperand::VReg(hbs ), op: GprOp::Add });
                                }
                                Ok(())
                            },
                        )?;

                        prog.emit(VmInstr::GprBinOp { dst: k_act_base, a: k_act_base, b: GprOperand::VReg(act_stride_reg ), op: GprOp::Add });
                        prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: blk_ptr, b: GprOperand::VReg(blk_ptr_stride ), op: GprOp::Add });
                        Ok(())
                    },
                )?;

                prog.emit(VmInstr::HReduce { dst: hreduce_dst, src: acc, op: super::instr::ReduceOp::Sum });
                prog.emit(VmInstr::VecScalarStore {
                    base: output_ptr, src: hreduce_dst,
                    offset: OffsetExpr::LoopOffset(j_off),
                });
                prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_row_ptr, b: GprOperand::VReg(j_weight_stride ), op: GprOp::Add });
                Ok(())
            })
        }
        GemmMode::General => {
            prog.emit_loop_try(plan.m_bound.clone(), 1, |prog, _i_ctr, i_cnt| -> Result<(), CompilerError> {
                let weight_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

                prog.emit_loop_try(BoundExpr::Const(n), elem, |prog, _j_ctr, j_off| -> Result<(), CompilerError> {
                    prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype });
                    prog.emit(VmInstr::GprLoadImm { dst: k_act_base, value: 0 });

                    let blk_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: weight_row_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

                    prog.emit_loop_try(BoundExpr::Const(gguf_num_blocks), block_bytes,
                        |prog, _blk_ctr, _blk_off| -> Result<(), CompilerError> {
                            if let Some(lo) = lane_offset_gpr {
                                prog.emit(VmInstr::GprLoadImm { dst: lo, value: 0 });
                            }
                            let data_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                            prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: blk_ptr, b: GprOperand::VReg(zero_gpr ), op: GprOp::Add });

                            let high_bits_ptr = if needs_high_bits {
                                let hbp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                                prog.emit(VmInstr::AddPtr { dst: hbp, base: blk_ptr, offset: high_offset_val });
                                Some(hbp)
                            } else {
                                None
                            };

                            prog.emit_loop_try(BoundExpr::Const(iters_per_block), lanes * elem,
                                |prog, _ei_ctr, ei_off| -> Result<(), CompilerError> {
                                    let mut decode_inputs: Vec<VRegId> = if let Some(lo) = lane_offset_gpr {
                                        vec![blk_ptr, data_ptr, lo]
                                    } else {
                                        vec![blk_ptr, data_ptr]
                                    };
                                    if let Some(hbp) = high_bits_ptr {
                                        decode_inputs.push(hbp);
                                    }
                                    let decode_slots = super::auto_select::auto_lower_trace_raw(
                                        prog, &decode_trace, &decode_inputs, width, dtype)?;
                                    let b_decoded = decode_slots[decode_final_slot.0 as usize];

                                    let act_off = OffsetExpr::Add(
                                        Box::new(OffsetExpr::Add(
                                            Box::new(OffsetExpr::Mul(
                                                Box::new(OffsetExpr::LoopOffset(i_cnt)),
                                                a_row_stride,
                                            )),
                                            Box::new(OffsetExpr::ScalarVReg(k_act_base)),
                                        )),
                                        Box::new(OffsetExpr::LoopOffset(ei_off)),
                                    );
                                    prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: act_off, width, dtype , predicate: None });

                                    super::auto_select::auto_lower_trace_into(
                                        prog, &fma_trace, &[a_val, b_decoded, acc], acc, width, dtype,
                                    )?;

                                    prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: data_ptr, b: GprOperand::VReg(data_stride_reg ), op: GprOp::Add });
                                    if let (Some(lo), Some(lg)) = (lane_offset_gpr, lanes_gpr) {
                                        prog.emit(VmInstr::GprBinOp { dst: lo, a: lo, b: GprOperand::VReg(lg ), op: GprOp::Add });
                                    }
                                    if let (Some(hbp), Some(hbs)) = (high_bits_ptr, high_bits_stride_gpr) {
                                        prog.emit(VmInstr::GprBinOp { dst: hbp, a: hbp, b: GprOperand::VReg(hbs), op: GprOp::Add });
                                    }
                                    Ok(())
                                },
                            )?;

                            prog.emit(VmInstr::GprBinOp { dst: k_act_base, a: k_act_base, b: GprOperand::VReg(act_stride_reg ), op: GprOp::Add });
                            prog.emit(VmInstr::GprBinOp { dst: blk_ptr, a: blk_ptr, b: GprOperand::VReg(blk_ptr_stride ), op: GprOp::Add });
                            Ok(())
                        },
                    )?;

                    prog.emit(VmInstr::HReduce { dst: hreduce_dst, src: acc, op: super::instr::ReduceOp::Sum });
                    prog.emit(VmInstr::VecScalarStore {
                        base: output_ptr, src: hreduce_dst,
                        offset: OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(i_cnt)), c_row_stride)),
                            Box::new(OffsetExpr::LoopOffset(j_off)),
                        ),
                    });
                    prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_row_ptr, b: GprOperand::VReg(j_weight_stride ), op: GprOp::Add });
                    Ok(())
                })
            })
        }
        GemmMode::Float => unreachable!("Float mode uses emit_gemm_float_from_plan"),
    }
}

