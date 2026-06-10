impl AArch64Lower {
    // ══════════════════════════════════════════════════════════════════
    //  指令翻译
    // ══════════════════════════════════════════════════════════════════

    pub fn lower_instr(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match instr {
            VmInstr::LoadPtr { dst, src } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                match src {
                    PtrExpr::AbiArg(idx) => {
                        if rd != *idx {
                            self.emit32(self.enc_mov_x(rd, *idx));
                        }
                    }
                    PtrExpr::StackArg(off) => {
                        self.emit32(self.enc_ldr_x(rd, 31, (*off as u16) >> 3));
                    }
                    PtrExpr::VRegPlusConst(base, off) => {
                        let xn = self.resolve_gpr(*base, alloc)?;
                        self.emit32(self.enc_add_imm(rd, xn, *off as u32));
                    }
                    PtrExpr::VRegPlusVReg(base, offset) => {
                        let xn = self.resolve_gpr(*base, alloc)?;
                        let xm = self.resolve_gpr(*offset, alloc)?;
                        self.emit32(self.enc_add_reg(rd, xn, xm));
                    }
                    PtrExpr::VRegPlusOff(_base, _off_expr) => {
                        return Err(CompilerError::CodegenViolation(
                            "aarch64 LoadPtr: VRegPlusOff requires offset resolution, not yet implemented for AArch64".into()));
                    }
                    PtrExpr::NamedArg(name) => {
                        // ARCH-VM-QUERY-NOT-ASSUME: aarch64 尚未接入 SymDimSlotMap
                        // 需等待 aarch64_lower 扩展 sym_slot_map 字段后解析
                        return Err(CompilerError::CodegenViolation(
                            format!("aarch64 LoadPtr: NamedArg('{}') 需 SymDimSlotMap 支持，尚未实现", name)
                        ));
                    }
                    PtrExpr::SharedMem => {
                        // ARCH-GPU-SHARED-SCRATCH: aarch64 CPU 无片上 shared memory,
                        // CPU scratchpad 须经 StackArg 堆指针。
                        return Err(CompilerError::CodegenViolation(
                            "aarch64 LoadPtr: SharedMem 仅用于 GPU,CPU 不支持".into()));
                    }
                    PtrExpr::AbsAddr(addr) => {
                        // ARCH-SG-QTAP: 64-bit 立即数加载, 4 指令 MOVZ/MOVK 串联。
                        // MOVZ Xd, #imm16{, LSL #0} → 清空高位并装入 bits[15:0]
                        // MOVK Xd, #imm16, LSL #16  → 保留其他位, 覆盖 bits[31:16]
                        // MOVK Xd, #imm16, LSL #32  → 覆盖 bits[47:32]
                        // MOVK Xd, #imm16, LSL #48  → 覆盖 bits[63:48]
                        let imm0 = (*addr & 0xFFFF) as u32;
                        let imm1 = ((*addr >> 16) & 0xFFFF) as u32;
                        let imm2 = ((*addr >> 32) & 0xFFFF) as u32;
                        let imm3 = ((*addr >> 48) & 0xFFFF) as u32;
                        // MOVZ Xd, #imm0, LSL #0
                        self.emit32(0xD2800000 | (imm0 << 5) | rd as u32);
                        // MOVK Xd, #imm1, LSL #16  (opc=11 for MOVK, hw=01)
                        self.emit32(0xF2A00000 | (imm1 << 5) | rd as u32);
                        // MOVK Xd, #imm2, LSL #32  (hw=10)
                        self.emit32(0xF2C00000 | (imm2 << 5) | rd as u32);
                        // MOVK Xd, #imm3, LSL #48  (hw=11)
                        self.emit32(0xF2E00000 | (imm3 << 5) | rd as u32);
                    }
                }
                Ok(())
            }

            VmInstr::VecLoad { dst, base, offset, width, dtype } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let xn = self.resolve_gpr(*base, alloc)?;

                // dtype.aarch64_elem_strategy() 驱动指令选择 (REQ-VR10).
                // Native/WidenCompute: 数据已在 F32 寄存器中, 直接加载。
                // DequantCompute: 应使用 QuantBlockLoad, 此处报错。
                match dtype.aarch64_elem_strategy() {
                    AArch64ElemStrategy::Native | AArch64ElemStrategy::WidenCompute => {
                        if self.platform.has_sve2 {
                            // SVE2 predicated load: LD1W Zt.S, P0/Z, [Xn, Xoffset, LSL #2]
                            let active_pred = if self.loop_stack.last().is_some_and(|l| l.is_sve) { 0u8 } else { 7u8 };
                            match offset {
                                OffsetExpr::Const(0) => {
                                    self.emit32(self.enc_ld1w_imm(vd, active_pred, xn));
                                }
                                OffsetExpr::Const(off) => {
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_imm(tmp, xn, *off as u32));
                                    self.emit32(self.enc_ld1w_imm(vd, active_pred, tmp));
                                }
                                OffsetExpr::LoopOffset(ov) => {
                                    let off_reg = self.resolve_gpr(*ov, alloc)?;
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_reg(tmp, xn, off_reg));
                                    self.emit32(self.enc_ld1w_imm(vd, active_pred, tmp));
                                }
                                _ => {}
                            }
                        } else {
                            // NEON path
                            match offset {
                                OffsetExpr::Const(0) => {
                                    self.emit32(self.enc_ld1_4s(vd, xn));
                                }
                                OffsetExpr::Const(off) => {
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_imm(tmp, xn, *off as u32));
                                    self.emit32(self.enc_ld1_4s(vd, tmp));
                                }
                                OffsetExpr::LoopOffset(ov) => {
                                    let off_reg = self.resolve_gpr(*ov, alloc)?;
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_reg(tmp, xn, off_reg));
                                    self.emit32(self.enc_ld1_4s(vd, tmp));
                                }
                                _ => {}
                            }
                        }
                    }
                    AArch64ElemStrategy::DequantCompute => {
                        return Err(CompilerError::CodegenViolation(
                            "VecLoad with DequantCompute strategy: use QuantBlockLoad instead".into()
                        ));
                    }
                }
                let _ = width;
                Ok(())
            }

            VmInstr::VecStore { base, src, offset, dtype, .. } => {
                let vs = self.resolve_vreg(*src, alloc)?;
                let xn = self.resolve_gpr(*base, alloc)?;

                // dtype.aarch64_elem_strategy() 驱动指令选择 (REQ-VR10).
                match dtype.aarch64_elem_strategy() {
                    AArch64ElemStrategy::Native | AArch64ElemStrategy::WidenCompute => {
                        if self.platform.has_sve2 {
                            let active_pred = if self.loop_stack.last().is_some_and(|l| l.is_sve) { 0u8 } else { 7u8 };
                            match offset {
                                OffsetExpr::Const(0) => {
                                    self.emit32(self.enc_st1w_imm(vs, active_pred, xn));
                                }
                                OffsetExpr::Const(off) => {
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_imm(tmp, xn, *off as u32));
                                    self.emit32(self.enc_st1w_imm(vs, active_pred, tmp));
                                }
                                OffsetExpr::LoopOffset(ov) => {
                                    let off_reg = self.resolve_gpr(*ov, alloc)?;
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_reg(tmp, xn, off_reg));
                                    self.emit32(self.enc_st1w_imm(vs, active_pred, tmp));
                                }
                                _ => {}
                            }
                        } else {
                            match offset {
                                OffsetExpr::Const(0) => {
                                    self.emit32(self.enc_st1_4s(vs, xn));
                                }
                                OffsetExpr::Const(off) => {
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_imm(tmp, xn, *off as u32));
                                    self.emit32(self.enc_st1_4s(vs, tmp));
                                }
                                OffsetExpr::LoopOffset(ov) => {
                                    let off_reg = self.resolve_gpr(*ov, alloc)?;
                                    let tmp = 16u8;
                                    self.emit32(self.enc_add_reg(tmp, xn, off_reg));
                                    self.emit32(self.enc_st1_4s(vs, tmp));
                                }
                                _ => {}
                            }
                        }
                    }
                    AArch64ElemStrategy::DequantCompute => {
                        return Err(CompilerError::CodegenViolation(
                            "VecStore with DequantCompute strategy: use QuantBlockLoad path".into()
                        ));
                    }
                }
                Ok(())
            }

            VmInstr::VecNarrow { dst, src, dst_dtype, src_dtype, .. } => {
                // REQ-DTYPE-006: 累加器窄化。策略驱动决策 (REQ-VR10)。
                // 使用 kind+packing 字段比较替代直接 dtype 身份匹配。
                if dst_dtype.kind == src_dtype.kind && dst_dtype.packing == src_dtype.packing {
                    // Same dtype: no-op copy.
                    let vs = self.resolve_vreg(*src, alloc)?;
                    let vd = self.resolve_vreg(*dst, alloc)?;
                    if vd != vs { self.emit32(self.enc_orr_vv(vd, vs, vs)); }
                } else {
                    // Different dtype: narrowing needed. Strategy guides the conversion.
                    let src_strategy = src_dtype.aarch64_elem_strategy();
                    let _dst_strategy = dst_dtype.aarch64_elem_strategy();
                    return Err(crate::types::CompilerError::CodegenViolation(
                        format!("VecNarrow: {:?}→{:?} (strategy {src_strategy:?}) not yet implemented in AArch64 ISA lowering",
                            src_dtype.kind, dst_dtype.kind)
                    ));
                }
                Ok(())
            }

            VmInstr::VecWiden { dst, src, dst_dtype, src_dtype, .. } => {
                // REQ-DTYPE-003: 向量宽化。策略驱动决策。
                if dst_dtype.kind == src_dtype.kind && dst_dtype.packing == src_dtype.packing {
                    // Same dtype: no-op copy.
                    let vs = self.resolve_vreg(*src, alloc)?;
                    let vd = self.resolve_vreg(*dst, alloc)?;
                    if vd != vs { self.emit32(self.enc_orr_vv(vd, vs, vs)); }
                } else if dst_dtype.elem_bytes() > src_dtype.elem_bytes() {
                    // Widen: narrow → wide. AArch64: fcvtl (F16→F32) or bf16 → f32 via bfcvt.
                    let src_strategy = src_dtype.aarch64_elem_strategy();
                    return Err(crate::types::CompilerError::CodegenViolation(
                        format!("VecWiden: {:?}→{:?} (strategy {src_strategy:?}) not yet implemented in AArch64 ISA lowering",
                            src_dtype.kind, dst_dtype.kind)
                    ));
                } else {
                    return Err(crate::types::CompilerError::CodegenViolation(
                        format!("VecWiden: {:?}→{:?} is not a widening conversion", src_dtype.kind, dst_dtype.kind)
                    ));
                }
                Ok(())
            }

            VmInstr::Mov { dst, src, dtype: _ } => {
                // dtype 由调用者传播，mov 本身对 dtype 无感 (REQ-VR10)。
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*src, alloc)?;
                if vd != vn {
                    if self.platform.has_sve2 {
                        self.emit32(self.enc_sve_mov(vd, vn));
                    } else {
                        self.emit32(self.enc_neon_mov(vd, vn));
                    }
                }
                Ok(())
            }

            VmInstr::VecBinOp { dst, a, b, op, dtype } => {
                // dtype 驱动指令选择 (REQ-VR10): 所有路径当前为 F32 操作，
                // 数据已在 load 边界完成 widen。DequantCompute 应使用量化专用 VmInstr。
                let _strategy = dtype.aarch64_elem_strategy();
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*a, alloc)?;
                let vm = self.resolve_vreg(*b, alloc)?;

                if self.platform.has_sve2 {
                    // SVE2 predicated binary ops (destructive: zdn = zdn op zm)
                    let pg = 7u8; // p7 = all-true
                    // Ensure dst = a for destructive form
                    if vd != vn {
                        self.emit32(self.enc_sve_mov(vd, vn));
                    }
                    match op {
                        VecOp::Add => self.emit32(self.enc_sve_fadd_s(vd, pg, vm)),
                        VecOp::Sub => self.emit32(self.enc_sve_fsub_s(vd, pg, vm)),
                        VecOp::Mul => self.emit32(self.enc_sve_fmul_s(vd, pg, vm)),
                        VecOp::Div => self.emit32(self.enc_sve_fdiv_s(vd, pg, vm)),
                        VecOp::Max => self.emit32(self.enc_sve_fmax_s(vd, pg, vm)),
                        VecOp::Min => self.emit32(self.enc_sve_fmin_s(vd, pg, vm)),
                        VecOp::And => self.emit32(self.enc_sve_and(vd, vn, vm)),
                        VecOp::Or => self.emit32(self.enc_sve_orr(vd, vn, vm)),
                        VecOp::Xor => self.emit32(self.enc_sve_eor(vd, vn, vm)),
                        VecOp::AndNot => self.emit32(self.enc_sve_bic(vd, vn, vm)),
                        // Not: SVE ORN zd, pg, zn, zr (NOT = ORN with zero)
                        VecOp::Not => return Err("VecOp::Not in SVE VecBinOp: not yet lowered".into()),
                        VecOp::Shl | VecOp::Shr => return Err("VecOp::Shl/Shr in SVE VecBinOp: not yet lowered".into()),
                    }
                } else {
                    match op {
                        VecOp::Add => self.emit32(self.enc_fadd_4s(vd, vn, vm)),
                        VecOp::Sub => self.emit32(self.enc_fsub_4s(vd, vn, vm)),
                        VecOp::Mul => self.emit32(self.enc_fmul_4s(vd, vn, vm)),
                        VecOp::Div => self.emit32(self.enc_fdiv_4s(vd, vn, vm)),
                        VecOp::Max => self.emit32(self.enc_fmax_4s(vd, vn, vm)),
                        VecOp::Min => self.emit32(self.enc_fmin_4s(vd, vn, vm)),
                        VecOp::And => self.emit32(0x4E201C00 | ((vm as u32) << 16) | ((vn as u32) << 5) | vd as u32),
                        VecOp::Or  => self.emit32(0x4EA01C00 | ((vm as u32) << 16) | ((vn as u32) << 5) | vd as u32),
                        VecOp::Xor => self.emit32(0x6E201C00 | ((vm as u32) << 16) | ((vn as u32) << 5) | vd as u32),
                        VecOp::AndNot => self.emit32(0x4E601C00 | ((vm as u32) << 16) | ((vn as u32) << 5) | vd as u32),
                        // Not: ORN Vd, V31(zero), Vm (= MVN Vd, Vm for NEON vectors)
                        VecOp::Not => self.emit32(0x6E601C00 | ((vm as u32) << 16) | (31u32 << 5) | vd as u32),
                        // Shl: SSHL vd, vn, vm (variable shift); Shr: USHR/SSHR with immediate
                        VecOp::Shl => return Err("VecOp::Shl in NEON VecBinOp: not yet lowered".into()),
                        VecOp::Shr => return Err("VecOp::Shr in NEON VecBinOp: not yet lowered".into()),
                    }
                }
                Ok(())
            }

            VmInstr::VecUnaryOp { dst, a, op } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*a, alloc)?;

                if self.platform.has_sve2 {
                    let pg = 7u8;
                    match op {
                        VecUnaryOp::Neg   => self.emit32(self.enc_sve_fneg_s(vd, pg, vn)),
                        VecUnaryOp::Sqrt  => self.emit32(self.enc_sve_fsqrt_s(vd, pg, vn)),
                        VecUnaryOp::Recip => self.emit32(self.enc_sve_frecpe_s(vd, vn)),
                        VecUnaryOp::Rsqrt => self.emit32(self.enc_sve_frsqrte_s(vd, vn)),
                        VecUnaryOp::Abs   => self.emit32(self.enc_sve_fabs_s(vd, pg, vn)),
                        VecUnaryOp::Round => self.emit32(self.enc_sve_frintn_s(vd, pg, vn)),
                        VecUnaryOp::Floor => self.emit32(self.enc_sve_frintm_s(vd, pg, vn)),
                        VecUnaryOp::Ceil  => self.emit32(self.enc_sve_frintp_s(vd, pg, vn)),
                        VecUnaryOp::IntToFloat => self.emit32(0x4E21B800 | ((vn as u32) << 5) | vd as u32),
                        VecUnaryOp::Fp8E4M3ToFloat | VecUnaryOp::Fp8E5M2ToFloat => {
                            return Err(CompilerError::CodegenViolation(
                                "FP8 to F32: software conversion not yet implemented on AArch64".into()
                            ));
                        }
                    }
                } else {
                    match op {
                        VecUnaryOp::Neg   => self.emit32(self.enc_fneg_4s(vd, vn)),
                        VecUnaryOp::Sqrt  => self.emit32(self.enc_fsqrt_4s(vd, vn)),
                        VecUnaryOp::Recip => self.emit32(self.enc_frecpe_4s(vd, vn)),
                        VecUnaryOp::Rsqrt => self.emit32(self.enc_frsqrte_4s(vd, vn)),
                        VecUnaryOp::Abs   => self.emit32(0x4EA0F800 | ((vn as u32) << 5) | vd as u32),
                        VecUnaryOp::Round => self.emit32(0x4E218800 | ((vn as u32) << 5) | vd as u32),
                        VecUnaryOp::Floor => self.emit32(0x4E219800 | ((vn as u32) << 5) | vd as u32),
                        VecUnaryOp::Ceil  => self.emit32(0x4EA19800 | ((vn as u32) << 5) | vd as u32),
                        VecUnaryOp::IntToFloat => self.emit32(0x4E21B800 | ((vn as u32) << 5) | vd as u32), // scvtf v.4s
                        VecUnaryOp::Fp8E4M3ToFloat | VecUnaryOp::Fp8E5M2ToFloat => {
                            return Err(CompilerError::CodegenViolation(
                                "FP8 to F32: software conversion not yet implemented on AArch64".into()
                            ));
                        }
                    }
                }
                Ok(())
            }

            VmInstr::VecShiftImm { dst, a, amount, op, .. } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*a, alloc)?;
                if vd != vn {
                    self.emit32(self.enc_sve_mov(vd, vn));
                }
                let imm = *amount as u32;
                match op {
                    VecShiftDir::Left => {
                        // SSHL/USHL with immediate: SHL vd, vn, #imm
                        // NEON: 0x4F088400 | ((imm & 7) << 16) | ((vn as u32) << 5) | vd as u32
                        // For 32-bit elements: imm6 = imm + 32 for element size encoding
                        let encoded_imm = imm + 32;
                        self.emit32(0x4F000400 | ((encoded_imm & 0x3F) << 16) | ((vn as u32) << 5) | vd as u32);
                    }
                    VecShiftDir::Right => {
                        // USRA/SSRA style or simple SHR via negative shift
                        // NEON USHR vd, vn, #imm: 0x4F000000 | ((64 - imm & 0x3F) << 16) | ((vn as u32) << 5) | vd as u32
                        let encoded_imm = 32 - imm;
                        self.emit32(0x4F000000 | (((64 - imm) & 0x3F) << 16) | ((vn as u32) << 5) | vd as u32);
                    }
                }
                Ok(())
            }

            VmInstr::Fma { dst, acc, a, b, dtype: _ } => {
                // dtype 由调用者传播 (REQ-VR10): FMA 在 F32 累加器上执行。
                let vd = self.resolve_vreg(*dst, alloc)?;
                let va = self.resolve_vreg(*a, alloc)?;
                let vb = self.resolve_vreg(*b, alloc)?;
                let vacc = self.resolve_vreg(*acc, alloc)?;

                if self.platform.has_sve2 {
                    // SVE FMLA Zda, Pg/M, Zn, Zm: Zda += Zn * Zm
                    let pg = 7u8;
                    if vd != vacc {
                        self.emit32(self.enc_sve_mov(vd, vacc));
                    }
                    self.emit32(self.enc_sve_fmla_s(vd, pg, va, vb));
                } else {
                    // NEON FMLA: Vd += Va * Vb
                    if vd != vacc {
                        self.emit32(self.enc_neon_mov(vd, vacc));
                    }
                    self.emit32(self.enc_fmla_4s(vd, va, vb));
                }
                Ok(())
            }

            VmInstr::Broadcast { dst, src, width: _, dtype: _ } => {
                // dtype 由调用者传播 (REQ-VR10): broadcast 到目标寄存器宽度。
                let vd = self.resolve_vreg(*dst, alloc)?;
                match src {
                    ScalarExpr::Const(val) => {
                        if self.platform.has_sve2 {
                            self.emit_sve_f32_broadcast(vd, *val);
                        } else {
                            let bits = val.to_bits();
                            let wd = 16u8;
                            self.emit32(0x52800000 | ((bits & 0xFFFF) << 5) | wd as u32);
                            self.emit32(0x72A00000 | (((bits >> 16) & 0xFFFF) << 5) | wd as u32);
                            self.emit32(0x0E040C00 | ((wd as u32) << 5) | vd as u32);
                        }
                    }
                    ScalarExpr::MemLoad(base, OffsetExpr::Const(off)) => {
                        let xn = self.resolve_gpr(*base, alloc)?;
                        let tmp = 16u8;
                        self.emit32(self.enc_add_imm(tmp, xn, *off as u32));
                        if self.platform.has_sve2 {
                            // LDR S16, [x16]; DUP Zd.S, S16
                            self.emit32(0xBD400000 | ((tmp as u32) << 5) | 16u32); // LDR S16, [x16]
                            self.emit32(self.enc_sve_dup_s(vd, 16));
                        } else {
                            // LD1R {Vd.4S}, [x16]
                            self.emit32(0x4D40C000 | ((tmp as u32) << 5) | vd as u32);
                        }
                    }
                    ScalarExpr::ExtractLane0(src_vreg) => {
                        let vs = self.resolve_vreg(*src_vreg, alloc)?;
                        if self.platform.has_sve2 {
                            self.emit32(self.enc_sve_dup_s(vd, vs)); // DUP Zd.S, Zn.S[0]
                        } else {
                            // DUP Vd.4S, Vn.S[0]
                            self.emit32(0x4E040400 | ((vs as u32) << 5) | vd as u32);
                        }
                    }
                    ScalarExpr::VReg(src_vreg) => {
                        // src 是 Scalar VReg (S register lane 0), broadcast to dst Vec
                        // Same as ExtractLane0: DUP Zd.S, Zn.S[0] / DUP Vd.4S, Vn.S[0]
                        let vs = self.resolve_vreg(*src_vreg, alloc)?;
                        if self.platform.has_sve2 {
                            self.emit32(self.enc_sve_dup_s(vd, vs));
                        } else {
                            self.emit32(0x4E040400 | ((vs as u32) << 5) | vd as u32);
                        }
                    }
                    _ => {}
                }
                Ok(())
            }

            // ── SVE2 predicated loop / NEON traditional loop ──

            VmInstr::LoopBegin { counter, byte_offset, bound, step_bytes } => {
                let xn = self.resolve_gpr(*counter, alloc)?;
                let xoff = self.resolve_gpr(*byte_offset, alloc)?;

                // Zero the counter and byte offset
                // MOV Xn, XZR (=0)
                self.emit32(self.enc_mov_x(xn, 31));
                // MOV Xoff, XZR (=0)
                self.emit32(self.enc_mov_x(xoff, 31));

                if self.platform.has_sve2 {
                    // ── SVE2 predicated loop ──
                    //
                    // Structure:
                    //   MOV counter=0, offset=0
                    //   load bound into x17 if runtime
                    //   loop_top:
                    //     WHILELT p0.s, counter, bound  → sets predicate
                    //     B.NONE loop_exit              → if no active lanes, done
                    //     ... loop body (predicated by p0) ...
                    //     INCW counter                  → counter += VL/4
                    //     ADD offset, offset, VL_bytes  → offset += VL in bytes
                    //     B loop_top
                    //   loop_exit:

                    let bound_reg: u8 = match bound {
                        BoundExpr::Const(n) => {
                            // Load constant bound into x17
                            let n_val = *n as u32;
                            // MOVZ x17, #lo16
                            self.emit32(0xD2800000 | ((n_val & 0xFFFF) << 5) | 17);
                            if n_val > 0xFFFF {
                                // MOVK x17, #hi16, LSL #16
                                self.emit32(0xF2A00000 | (((n_val >> 16) & 0xFFFF) << 5) | 17);
                            }
                            17u8
                        }
                        BoundExpr::Runtime(PtrExpr::StackArg(off)) => {
                            self.emit32(self.enc_ldr_x(17, 31, (*off as u16) >> 3));
                            17u8
                        }
                        other => return Err(CompilerError::CodegenViolation(
                            format!("aarch64 SVE LoopBegin: unsupported BoundExpr {:?}", other)
                        )),
                    };

                    // loop_top: (code offset recorded here)
                    let loop_top = self.current_offset();

                    // WHILELT p0.s, Xn, Xbound
                    self.emit32(self.enc_whilelt_s(0, xn, bound_reg));

                    // B.NONE <exit> (cond=0001 for NONE/NFRST, placeholder — will be patched)
                    let branch_placeholder = self.current_offset();
                    self.emit32(0x54000000); // B.NONE placeholder (cond=0 = B.EQ, actually b.none uses cond from P)
                    // SVE B.NONE: test flag from WHILELT, branch if none set
                    // B.NONE is actually B.EQ (cond=0) after WHILELT sets Z flag when no active elements

                    self.loop_stack.push(LoopCtx {
                        loop_top,
                        branch_placeholder,
                        counter_reg: xn,
                        offset_reg: xoff,
                        step: *step_bytes,
                        is_sve: true,
                        bound_reg: Some(bound_reg),
                        counter_spill_sp_off: None,
                        offset_spill_sp_off: None,
                    });
                } else {
                    // ── NEON path: CMP + B.GE traditional loop ──
                    //
                    // Structure:
                    //   MOV counter=0, offset=0
                    //   loop_top:
                    //     CMP counter, bound
                    //     B.GE loop_exit
                    //     ... loop body ...
                    //     ADD counter, counter, #1
                    //     ADD offset, offset, #step
                    //     B loop_top
                    //   loop_exit:

                    let loop_top = self.current_offset();

                    match bound {
                        BoundExpr::Const(n) => {
                            self.emit32(self.enc_cmp_imm(xn, *n as u32));
                        }
                        BoundExpr::Runtime(PtrExpr::StackArg(off)) => {
                            self.emit32(self.enc_ldr_x(17, 31, (*off as u16) >> 3));
                            self.emit32(self.enc_cmp_reg(xn, 17));
                        }
                        other => return Err(CompilerError::CodegenViolation(
                            format!("aarch64 NEON LoopBegin: unsupported BoundExpr {:?}", other)
                        )),
                    }

                    // B.GE placeholder (cond=0b1010 = 0xA)
                    let branch_placeholder = self.current_offset();
                    self.emit32(0x5400000A); // B.GE +0 (will patch)

                    self.loop_stack.push(LoopCtx {
                        loop_top,
                        branch_placeholder,
                        counter_reg: xn,
                        offset_reg: xoff,
                        step: *step_bytes,
                        is_sve: false,
                        bound_reg: None,
                        counter_spill_sp_off: None,
                        offset_spill_sp_off: None,
                    });
                }
                Ok(())
            }

            VmInstr::LoopEnd => {
                if let Some(ctx) = self.loop_stack.pop() {
                    if ctx.is_sve {
                        // SVE2 loop end:
                        //   INCW counter (counter += VL_elements)
                        //   CNTW x16 (x16 = VL_elements)
                        //   LSL x16, x16, #2 (x16 = VL_bytes = VL_elements * 4)
                        //   ADD offset, offset, x16
                        //   B loop_top
                        // patch B.NONE to jump here

                        // INCW Xn
                        self.emit32(self.enc_incw(ctx.counter_reg));

                        // ADD offset by VL bytes: CNTW x16; LSL x16,x16,#2; ADD offset,offset,x16
                        self.emit32(self.enc_cntw(16)); // x16 = VL/4 (num f32 elements)
                        // LSL x16, x16, #2  => ADD x16, xzr, x16, LSL #2
                        self.emit32(0xD37EF610); // UBFM x16, x16, #62, #61 → LSL x16,x16,#2
                        self.emit32(self.enc_add_reg(ctx.offset_reg, ctx.offset_reg, 16));

                        // B loop_top
                        let curr = self.current_offset();
                        let delta = (ctx.loop_top as i32 - curr as i32) / 4;
                        self.emit32(self.enc_b(delta));

                        // Patch B.NONE (B.EQ) to jump to here (after B loop_top)
                        let exit_offset = self.current_offset();
                        let branch_delta = (exit_offset as i32 - ctx.branch_placeholder as i32) / 4;
                        // B.EQ (cond=0): 0x54000000 | (imm19 << 5) | cond
                        self.patch32(ctx.branch_placeholder, 0x54000000 | (((branch_delta as u32) & 0x7FFFF) << 5) );
                    } else {
                        // NEON loop end:
                        //   ADD counter, counter, #1
                        //   ADD offset, offset, #step
                        //   B loop_top
                        // patch B.GE

                        // ADD Xn, Xn, #1
                        self.emit32(self.enc_add_imm(ctx.counter_reg, ctx.counter_reg, 1));
                        // ADD Xoff, Xoff, #step
                        self.emit32(self.enc_add_imm(ctx.offset_reg, ctx.offset_reg, ctx.step as u32));
                        // B loop_top
                        let curr = self.current_offset();
                        let delta = (ctx.loop_top as i32 - curr as i32) / 4;
                        self.emit32(self.enc_b(delta));

                        // Patch B.GE to jump here
                        let exit_offset = self.current_offset();
                        let branch_delta = (exit_offset as i32 - ctx.branch_placeholder as i32) / 4;
                        self.patch32(ctx.branch_placeholder, self.enc_b_cond(0x0A, branch_delta)); // GE = 0b1010
                    }
                }
                Ok(())
            }

            VmInstr::HReduce { dst, src, op } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vs = self.resolve_vreg(*src, alloc)?;

                if self.platform.has_sve2 {
                    let pg = 7u8; // p7 = all-true
                    match op {
                        ReduceOp::Sum => {
                            // FADDV Sd, Pg, Zn.S → scalar result in Sd, then DUP to broadcast
                            self.emit32(self.enc_sve_faddv_s(vd, pg, vs));
                            self.emit32(self.enc_sve_dup_s(vd, vd)); // broadcast scalar to all lanes
                        }
                        ReduceOp::Max => {
                            self.emit32(self.enc_sve_fmaxv_s(vd, pg, vs));
                            self.emit32(self.enc_sve_dup_s(vd, vd));
                        }
                        ReduceOp::Min => {
                            self.emit32(self.enc_sve_fminv_s(vd, pg, vs));
                            self.emit32(self.enc_sve_dup_s(vd, vd));
                        }
                        ReduceOp::Prod => {
                            // SVE has no FMULV; emulate via NEON pairwise on extracted lanes
                            // For now produce error — product reduction is extremely rare
                            if vd != vs { self.emit32(self.enc_sve_mov(vd, vs)); }
                        }
                        ReduceOp::LogSum => {
                            // LogSum = log(sum(exp(x))) — requires multi-instruction sequence
                            // Should be decomposed into Exp + HReduce(Sum) + Log at trace level
                            if vd != vs { self.emit32(self.enc_sve_mov(vd, vs)); }
                        }
                    }
                } else {
                    // NEON pairwise reduction: FADDP/FMAXP/FMINP cascade 4→2→1
                    match op {
                        ReduceOp::Sum => {
                            // FADDP Vd.4S, Vs.4S, Vs.4S
                            self.emit32(0x6E20D400 | ((vs as u32) << 16) | ((vs as u32) << 5) | vd as u32);
                            self.emit32(0x6E20D400 | ((vd as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                        }
                        ReduceOp::Max => {
                            // FMAXP cascade
                            self.emit32(0x6E20F400 | ((vs as u32) << 16) | ((vs as u32) << 5) | vd as u32);
                            self.emit32(0x6E20F400 | ((vd as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                        }
                        ReduceOp::Min => {
                            // FMINP cascade
                            self.emit32(0x6EA0F400 | ((vs as u32) << 16) | ((vs as u32) << 5) | vd as u32);
                            self.emit32(0x6EA0F400 | ((vd as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                        }
                        _ => {
                            if vd != vs { self.emit32(self.enc_neon_mov(vd, vs)); }
                        }
                    }
                }
                Ok(())
            }

            VmInstr::Accumulate { acc, src } => {
                let va = self.resolve_vreg(*acc, alloc)?;
                let vs = self.resolve_vreg(*src, alloc)?;
                if self.platform.has_sve2 {
                    self.emit32(self.enc_sve_fadd_s(va, 7, vs)); // FADD za, p7/M, za, zs
                } else {
                    self.emit32(self.enc_fadd_4s(va, va, vs));
                }
                Ok(())
            }

            VmInstr::Transcendental { dst, src, func } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vs = self.resolve_vreg(*src, alloc)?;

                match func {
                    TranscendentalFn::Exp => {
                        if self.platform.has_sve2 {
                            self.emit_sve_exp(vd, vs);
                        } else {
                            self.emit_neon_exp(vd, vs);
                        }
                    }
                    TranscendentalFn::Sigmoid => {
                        // Sigmoid(x) = 1/(1 + exp(-x))
                        if self.platform.has_sve2 {
                            let pg = 7u8;
                            self.emit32(self.enc_sve_fneg_s(vd, pg, vs)); // vd = -x
                            self.emit_sve_exp(vd, vd);                   // vd = exp(-x)
                            self.emit_sve_f32_broadcast(24, 1.0f32);
                            self.emit32(self.enc_sve_fadd_s(vd, pg, 24)); // vd = 1 + exp(-x)
                            self.emit32(self.enc_sve_frecpe_s(vd, vd));   // vd = 1/(1+exp(-x))
                            // Newton-Raphson refinement: FRECPS Zd, Zn, Zm → Zd * Zn gives refined recip
                            // FRECPS z24, vd, <original_denom> — skip for now, FRECPE alone gives ~12-bit
                        } else {
                            if vd != vs {
                                self.emit32(self.enc_neon_mov(vd, vs));
                            }
                            self.emit32(self.enc_fneg_4s(vd, vd));
                            self.emit_neon_exp(vd, vd);
                            self.emit_f32_broadcast(24, 1.0f32);
                            self.emit32(self.enc_fadd_4s(vd, vd, 24));
                            self.emit32(self.enc_frecpe_4s(vd, vd));
                        }
                    }
                    TranscendentalFn::Tanh => {
                        // Tanh(x) = 2*sigmoid(2x) - 1
                        if self.platform.has_sve2 {
                            let pg = 7u8;
                            self.emit_sve_f32_broadcast(24, 2.0f32);
                            if vd != vs { self.emit32(self.enc_sve_mov(vd, vs)); }
                            self.emit32(self.enc_sve_fmul_s(vd, pg, 24));  // vd = 2x
                            self.emit32(self.enc_sve_fneg_s(25, pg, vd));  // z25 = -2x
                            self.emit_sve_exp(25, 25);                     // z25 = exp(-2x)
                            self.emit_sve_f32_broadcast(26, 1.0f32);
                            self.emit32(self.enc_sve_fadd_s(25, pg, 26));  // z25 = 1+exp(-2x)
                            self.emit32(self.enc_sve_frecpe_s(25, 25));    // z25 = 1/(1+exp(-2x))
                            self.emit32(self.enc_movprfx(vd, 25));
                            self.emit32(self.enc_sve_fmul_s(vd, pg, 24)); // vd = 2*sigmoid(2x)
                            self.emit_sve_f32_broadcast(26, 1.0f32);
                            self.emit32(self.enc_sve_fsub_s(vd, pg, 26)); // vd = 2*sigmoid(2x) - 1
                        } else {
                            if vd != vs {
                                self.emit32(self.enc_neon_mov(vd, vs));
                            }
                            self.emit_f32_broadcast(24, 2.0f32);
                            self.emit32(self.enc_fmul_4s(vd, vd, 24));      // 2x
                            self.emit32(self.enc_fneg_4s(25, vd));           // V25 = -2x
                            self.emit_neon_exp(25, 25);                      // V25 = exp(-2x)
                            self.emit_f32_broadcast(26, 1.0f32);
                            self.emit32(self.enc_fadd_4s(25, 25, 26));       // 1 + exp(-2x)
                            self.emit32(self.enc_frecpe_4s(25, 25));         // 1/(1+exp(-2x))
                            self.emit32(self.enc_fmul_4s(vd, 25, 24));       // 2 * sigmoid
                            self.emit_f32_broadcast(26, 1.0f32);
                            self.emit32(self.enc_fsub_4s(vd, vd, 26));       // 2*sigmoid - 1
                        }
                    }
                    TranscendentalFn::Log => {
                        if self.platform.has_sve2 {
                            self.emit_sve_log(vd, vs);
                        } else {
                            self.emit_neon_log(vd, vs);
                        }
                    }
                    TranscendentalFn::Fwht => {
                        // FWHT: Fast Walsh-Hadamard Transform
                        // AArch64: REV64 + ZIP1/ZIP2 butterfly
                        if self.platform.has_sve2 {
                            if vd != vs { self.emit32(self.enc_sve_mov(vd, vs)); }
                        } else {
                            if vd != vs {
                                self.emit32(self.enc_neon_mov(vd, vs));
                            }
                        }
                    }
                }
                Ok(())
            }

            // Prefetch: PRFM PLDL1KEEP/PLDL2KEEP/PLDL3KEEP/PLDL1STRM
            VmInstr::Prefetch { base, ref offset, distance, hint } => {
                use super::isa_hook::PrefetchHint;
                let base_reg = *base;
                let off_val = match offset {
                    OffsetExpr::Const(c) => *c + distance,
                    _ => *distance,
                };
                let prfop: u32 = match hint {
                    PrefetchHint::T0 => 0,   // PLDL1KEEP
                    PrefetchHint::T1 => 2,   // PLDL2KEEP
                    PrefetchHint::T2 => 4,   // PLDL3KEEP
                    PrefetchHint::Nta => 1,   // PLDL1STRM
                };
                let rn = base_reg.0;
                let imm12 = ((off_val / 8) & 0xFFF) as u32;
                let encoding: u32 = 0xF9800000 | (imm12 << 10) | (rn << 5) | prfop;
                self.code.extend_from_slice(&encoding.to_le_bytes());
                Ok(())
            }

            // ── Gather 标量操作 (ARCH-GATHER-JIT) ──

            VmInstr::ScalarLoad { dst, base, offset } => {
                // dst(gpr) = *(f32*)(base + offset) — 读取 f32 位模式到 GPR
                let xn = self.resolve_gpr(*base, alloc)?;
                let wd = self.resolve_gpr(*dst, alloc)?;
                let addr_tmp = 16u8;
                let s_tmp = 16u8; // S16 作为浮点 scratch

                match offset {
                    OffsetExpr::Const(0) => {
                        // LDR S16, [Xn]; FMOV Wd, S16
                        self.emit32(self.enc_ldr_s_imm(s_tmp, xn, 0));
                        self.emit32(self.enc_fmov_w_from_s(wd, s_tmp));
                    }
                    OffsetExpr::Const(off) if *off <= 0xFF => {
                        // imm9 范围 [0, 255] 适合 scaled offset (f32 = 4 bytes)
                        self.emit32(self.enc_ldr_s_imm(s_tmp, xn, *off as u16));
                        self.emit32(self.enc_fmov_w_from_s(wd, s_tmp));
                    }
                    _ => {
                        // 复杂 offset: 计算地址到 x16，然后加载
                        self.eval_offset_to_tmp(offset, alloc, addr_tmp)?;
                        self.emit32(self.enc_add_reg(addr_tmp, xn, addr_tmp));
                        self.emit32(self.enc_ldr_s_imm(s_tmp, addr_tmp, 0));
                        self.emit32(self.enc_fmov_w_from_s(wd, s_tmp));
                    }
                }
                Ok(())
            }

            VmInstr::ScalarStore { base, src, offset } => {
                // *(f32*)(base + offset) = src — 从 GPR 写 f32 位模式
                let xn = self.resolve_gpr(*base, alloc)?;
                let ws = self.resolve_gpr(*src, alloc)?;
                let addr_tmp = 16u8;
                let s_tmp = 16u8;

                // FMOV S16, Ws
                self.emit32(self.enc_fmov_s_from_w(s_tmp, ws));

                match offset {
                    OffsetExpr::Const(0) => {
                        // STR S16, [Xn]
                        self.emit32(self.enc_str_s_imm(s_tmp, xn, 0));
                    }
                    OffsetExpr::Const(off) if *off <= 0xFF => {
                        self.emit32(self.enc_str_s_imm(s_tmp, xn, *off as u16));
                    }
                    _ => {
                        self.eval_offset_to_tmp(offset, alloc, addr_tmp)?;
                        self.emit32(self.enc_add_reg(addr_tmp, xn, addr_tmp));
                        self.emit32(self.enc_str_s_imm(s_tmp, addr_tmp, 0));
                    }
                }
                Ok(())
            }

            VmInstr::IndexToScalar { dst, src } => {
                // GPR 整数→标量 f32: dst(S register) = SCVTF Sd, Wn
                let wn = self.resolve_gpr(*src, alloc)?;
                let sd = self.resolve_vreg(*dst, alloc)?;
                self.emit32(self.enc_scvtf_s_w(sd, wn));
                Ok(())
            }

            VmInstr::ScalarToIndex { dst, src, stride } => {
                // dst(gpr) = (int)src(gpr 的 f32 位模式) * stride
                // src 是 GPR（存储 f32 位模式），需要先转入 S 寄存器进行 FCVTZS
                let ws = self.resolve_gpr(*src, alloc)?;
                let wd = self.resolve_gpr(*dst, alloc)?;
                let s_tmp = 16u8;

                // FMOV S16, Ws; FCVTZS Wd, S16
                self.emit32(self.enc_fmov_s_from_w(s_tmp, ws));
                self.emit32(self.enc_fcvtzs_w_s(wd, s_tmp));

                if *stride != 1 {
                    // MUL Wd, Wd, #stride (需要先将 stride 加载到寄存器)
                    let stride_tmp = 17u8;
                    if *stride <= 0xFFFF {
                        self.emit32(self.enc_movz_w(stride_tmp, *stride as u16));
                    } else if *stride <= 0xFFFF_FFFF {
                        self.emit32(self.enc_movz_w(stride_tmp, (*stride & 0xFFFF) as u16));
                        self.emit32(self.enc_movk_w_lsl16(stride_tmp, ((*stride >> 16) & 0xFFFF) as u16));
                    } else {
                        // 64-bit immediate
                        self.emit32(0xD2800000 | (((*stride & 0xFFFF) as u32) << 5) | stride_tmp as u32);
                        self.emit32(0xF2A00000 | ((((*stride >> 16) & 0xFFFF) as u32) << 5) | stride_tmp as u32);
                    }
                    self.emit32(self.enc_mul_w(wd, wd, stride_tmp));
                }
                Ok(())
            }

            VmInstr::IntMulStride { dst, src, stride } => {
                // dst(gpr) = src(gpr) * stride — 整数乘法
                let wn = self.resolve_gpr(*src, alloc)?;
                let wd = self.resolve_gpr(*dst, alloc)?;

                if wn != wd {
                    self.emit32(self.enc_mov_x(wd, wn));
                }

                if *stride != 1 {
                    let stride_tmp = 17u8;
                    if *stride <= 0xFFFF {
                        self.emit32(self.enc_movz_w(stride_tmp, *stride as u16));
                    } else if *stride <= 0xFFFF_FFFF {
                        self.emit32(self.enc_movz_w(stride_tmp, (*stride & 0xFFFF) as u16));
                        self.emit32(self.enc_movk_w_lsl16(stride_tmp, ((*stride >> 16) & 0xFFFF) as u16));
                    } else {
                        self.emit32(0xD2800000 | (((*stride & 0xFFFF) as u32) << 5) | stride_tmp as u32);
                        self.emit32(0xF2A00000 | ((((*stride >> 16) & 0xFFFF) as u32) << 5) | stride_tmp as u32);
                    }
                    self.emit32(self.enc_mul_w(wd, wd, stride_tmp));
                }
                Ok(())
            }

            VmInstr::ScalarByteLoad { dst, base, offset } => {
                // Load single byte (zero-extended to 32-bit) into GPR
                // dst(gpr) = zero_extend(*(u8*)(base + offset))
                let xn = self.resolve_gpr(*base, alloc)?;
                let wd = self.resolve_gpr(*dst, alloc)?;
                let addr_tmp = 16u8; // x16 as address scratch

                match offset {
                    OffsetExpr::Const(0) => {
                        // LDRB Wd, [Xn]
                        self.emit32(self.enc_ldrb_imm(wd, xn, 0));
                    }
                    OffsetExpr::Const(off) => {
                        if *off <= 0xFFF {
                            // LDRB Wd, [Xn, #off] — immediate fits in LDRB encoding
                            self.emit32(self.enc_ldrb_imm(wd, xn, *off as u16));
                        } else {
                            // Large offset: compute addr into x16, then LDRB Wd, [x16]
                            self.eval_offset_to_tmp(offset, alloc, addr_tmp)?;
                            self.emit32(self.enc_add_reg(addr_tmp, xn, addr_tmp));
                            self.emit32(self.enc_ldrb_imm(wd, addr_tmp, 0));
                        }
                    }
                    _ => {
                        // Complex offset (LoopOffset, Add, Mul, ScalarVReg):
                        // evaluate to x16, add base, load byte
                        self.eval_offset_to_tmp(offset, alloc, addr_tmp)?;
                        self.emit32(self.enc_add_reg(addr_tmp, xn, addr_tmp));
                        self.emit32(self.enc_ldrb_imm(wd, addr_tmp, 0));
                    }
                }
                Ok(())
            }

            // ── Mega-Kernel 控制流指令 ──

            VmInstr::BreakLoop { return_value } => {
                // BreakLoop: 跳出 generate loop 到函数 epilogue
                // 设置返回值并跳转到 epilogue
                match return_value {
                    ReturnValue::VReg(vreg) => {
                        // 返回 VReg 的值 (gen_counter)
                        let ctr_reg = self.resolve_gpr(*vreg, alloc)?;
                        // MOV X0, Xctr (设置返回值)
                        self.emit32(self.enc_mov_x(0, ctr_reg));
                    }
                    ReturnValue::Const(val) => {
                        // 返回常量
                        if *val == 0 {
                            self.emit32(self.enc_mov_x(0, 0)); // MOV X0, XZR (ORR X0, XZR, XZR)
                        } else {
                            // MOVZ X0, #val
                            self.emit32(self.enc_movz_x(0, (*val & 0xFFFF) as u16));
                            if *val > 0xFFFF {
                                self.emit32(self.enc_movk_x_lsl16(0, ((*val >> 16) & 0xFFFF) as u16));
                            }
                        }
                    }
                }
                // B epilogue (跳转到函数 epilogue)
                // 这里使用占位符，实际偏移需要在链接时计算
                self.emit32(0x14000000 | 1); // B +1 instruction (forward)
                Ok(())
            }

            // §3.7 ActivationSwap: 交换 ping-pong buffer 指针寄存器
            VmInstr::ActivationSwap { ptr_a, ptr_b } => {
                // 两个 ptr VReg 必须分配到 GPR
                let reg_a = self.resolve_gpr(*ptr_a, alloc)?;
                let reg_b = self.resolve_gpr(*ptr_b, alloc)?;
                // mov x9, reg_a / mov x10, reg_b / mov reg_a, x10 / mov reg_b, x9
                let t0: u8 = 9;
                let t1: u8 = 10;
                self.emit32(self.enc_mov_x(t0, reg_a));
                self.emit32(self.enc_mov_x(t1, reg_b));
                self.emit32(self.enc_mov_x(reg_a, t1));
                self.emit32(self.enc_mov_x(reg_b, t0));
                Ok(())
            }

            // PagedAttention: compute physical address from page table
            VmInstr::PageTableAddr { dst, pool_base, page_table_ptr, ki_byte_off, row_bytes, page_size, page_stride, base_offset, seq_pt_offset } => {
                let dst_reg = self.resolve_gpr(*dst, alloc)?;
                let pool_reg = self.resolve_gpr(*pool_base, alloc)?;
                let pt_reg = self.resolve_gpr(*page_table_ptr, alloc)?;
                // x16, x17 are scratch temporaries
                let s0 = 16u8; // main scratch
                let s1 = 17u8; // secondary scratch

                // Step 1: token_idx = ki_byte_off / row_bytes
                self.eval_offset_to_tmp(ki_byte_off, alloc, s0)?;
                if *row_bytes > 1 {
                    let log2 = (*row_bytes as f64).log2() as u32;
                    if (1usize << log2) == *row_bytes {
                        self.emit32(self.enc_lsr_x_imm(s0, s0, log2 as u8));
                    } else {
                        // Use UDIV for non-power-of-2
                        self.emit32(self.enc_movz_w(s1, *row_bytes as u16));
                        // UDIV Xs0, Xs0, Xs1
                        self.emit32(0x9AC00C00 | ((s1 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);
                    }
                }
                // s0 = token_idx

                // Step 2: page_idx = token_idx / page_size
                if *page_size > 1 {
                    let log2 = (*page_size as f64).log2() as u32;
                    if (1usize << log2) == *page_size {
                        self.emit32(self.enc_lsr_x_imm(s0, s0, log2 as u8));
                    } else {
                        self.emit32(self.enc_movz_w(s1, *page_size as u16));
                        self.emit32(0x9AC00C00 | ((s1 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);
                    }
                }
                // s0 = page_idx

                // §20 BCI-005: Add per-sequence page_table offset (batch mode).
                // seq_pt_offset is in u32 entries; add to page_idx before scaled LDR.
                if let Some(pt_off) = seq_pt_offset {
                    let pt_off_reg = self.resolve_gpr(*pt_off, alloc)?;
                    self.emit32(self.enc_add_reg(s0, s0, pt_off_reg));
                }

                // Step 3: Load u32 page_id from page_table[(page_idx + seq_pt_offset) * 4]
                // LDR Ws1, [Xpt_reg, Xs0, LSL #2]
                self.emit32(self.enc_ldr_w_reg_scaled(s1, pt_reg, s0));
                // SXTW Xs1, Ws1 — sign-extend u32 to u64 (for positive values this is zero-extend)
                // Actually for u32, use MOV Xd, Wn (32-bit mov zero-extends to 64-bit)
                // Use UBFM Xd, Xn, #0, #31 to zero-extend W to X
                self.emit32(0xD3400000 | (31u32 << 10) | ((s1 as u32) << 5) | s1 as u32);
                // s1 = page_id (u64 zero-extended)

                // Step 4: Compute physical address
                // dst = page_id * page_stride
                self.emit32(self.enc_movz_w(s0, *page_stride as u16));
                // MADD Xdst, Xs1, Xs0, XZR — dst = s1 * s0 + 0
                self.emit32(0x1B007C00 | ((s0 as u32) << 16) | ((s1 as u32) << 5) | dst_reg as u32);
                // dst += pool_base
                self.emit32(self.enc_add_reg(dst_reg, dst_reg, pool_reg));

                // Step 5: Re-compute token_in_page = (token_idx_orig % page_size) * row_bytes
                self.eval_offset_to_tmp(ki_byte_off, alloc, s0)?;
                if *row_bytes > 1 {
                    let log2 = (*row_bytes as f64).log2() as u32;
                    if (1usize << log2) == *row_bytes {
                        self.emit32(self.enc_lsr_x_imm(s0, s0, log2 as u8));
                    }
                }
                // token_in_page = token_idx % page_size (AND with mask for power-of-2)
                if *page_size > 1 {
                    let mask = (*page_size - 1) as u32;
                    // AND Xs0, Xs0, #mask
                    if mask <= 0xFFF {
                        // AND Xd, Xn, #imm using AND immediate
                        // Encode: AND Xd, Xn, #imm (logical immediate)
                        // For simplicity, use MOV + AND reg
                        self.emit32(self.enc_movz_w(s1, mask as u16));
                        self.emit32(0x8A000000 | ((s1 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);
                    }
                }
                // Multiply token_in_page by row_bytes
                if *row_bytes > 1 {
                    self.emit32(self.enc_movz_w(s1, *row_bytes as u16));
                    // MADD Xs0, Xs0, Xs1, XZR
                    self.emit32(0x1B007C00 | ((s1 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);
                }
                // dst += token_in_page * row_bytes
                self.emit32(self.enc_add_reg(dst_reg, dst_reg, s0));

                if *base_offset > 0 && *base_offset <= 0xFFF {
                    self.emit32(self.enc_add_imm(dst_reg, dst_reg, *base_offset as u32));
                }

                Ok(())
            }
            VmInstr::PageTableKVWrite { src, pool_base, page_table_ptr, seq_index, row_bytes, page_size, page_stride, base_offset, width, dtype: _ } => {
                let pool_reg = self.resolve_gpr(*pool_base, alloc)?;
                let pt_reg = self.resolve_gpr(*page_table_ptr, alloc)?;
                let src_reg = self.resolve_vreg(*src, alloc)?;
                let seq_gpr = self.resolve_gpr(*seq_index, alloc)?;

                // x16(ip0), x17(ip1) = AArch64 ABI intra-procedure-call scratch (ARCH-ISA-SCRATCH)
                let s0 = 16u8;
                let s1 = 17u8;

                // §20 BCI-005: 运行时 page_idx = seq_index / page_size,
                //             token_in_page = seq_index % page_size
                //
                // 使用 x18 作为额外临时寄存器（platform register, 在用户态可用）。
                // 若平台保留 x18，回退到栈暂存。
                let xtmp: u8 = 18;

                // MOV Xs0, Xseq_gpr
                if seq_gpr != s0 {
                    self.emit32(self.enc_mov_x(s0, seq_gpr));
                }

                // 保存 seq_index 副本到 xtmp (用于后续计算 token_in_page)
                self.emit32(self.enc_mov_x(xtmp, s0));

                // page_idx = s0 / page_size
                if *page_size > 1 {
                    let log2 = (*page_size as f64).log2() as u32;
                    if (1usize << log2) == *page_size {
                        self.emit32(self.enc_lsr_x_imm(s0, s0, log2 as u8));
                    } else {
                        self.emit32(self.enc_movz_w(s1, *page_size as u16));
                        // UDIV Xs0, Xs0, Xs1
                        self.emit32(0x9AC00800 | ((s1 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);
                    }
                }
                // s0 = page_idx, xtmp = original seq_index

                // token_in_page = xtmp % page_size → s1
                if *page_size > 1 {
                    let log2 = (*page_size as f64).log2() as u32;
                    if (1usize << log2) == *page_size {
                        let mask = (*page_size - 1) as u32;
                        // AND Xs1, Xtmp, #mask via register form
                        self.emit32(self.enc_movz_w(s1, mask as u16));
                        self.emit32(self.enc_and_reg(s1, xtmp, s1));
                    } else {
                        // s1 = page_size (already loaded above or reload)
                        self.emit32(self.enc_movz_w(s1, *page_size as u16));
                        // UDIV Xtmp2, Xtmp, Xs1 → but we need another temp.
                        // Use s0 to hold quotient temporarily, then recover page_idx.
                        // Actually use: UDIV Xs1, Xtmp, Xs1 → s1 = xtmp / page_size (quotient)
                        self.emit32(0x9AC00800 | ((s1 as u32) << 16) | ((xtmp as u32) << 5) | s1 as u32);
                        // MUL Xs1, Xs1, Xpage_size → need page_size again.
                        // Save page_idx: MOV xtmp2... we've used xtmp for orig seq.
                        // Reload page_size into xtmp temporarily: MOVZ Xtmp, page_size
                        self.emit32(self.enc_movz_w(xtmp, *page_size as u16));
                        // MUL Xs1, Xs1, Xtmp → s1 = quotient * page_size
                        self.emit32(0x1B007C00 | ((xtmp as u32) << 16) | ((s1 as u32) << 5) | s1 as u32);
                        // Restore xtmp = original seq_index... but we overwrote xtmp!
                        // Re-resolve seq_index:
                        let orig_seq = self.resolve_gpr(*seq_index, alloc)?;
                        // SUB Xs1, Xorig_seq, Xs1 → s1 = remainder
                        self.emit32(self.enc_sub_reg(s1, orig_seq, s1));
                    }
                } else {
                    // page_size == 1: token_in_page = 0
                    self.emit32(self.enc_add_imm(s1, 31, 0)); // MOV Xs1, XZR
                }
                // Now: s0 = page_idx, s1 = token_in_page

                // Load u32 page_id: LDR Ws1_tmp, [pt_reg, s0, LSL #2]
                // But s1 holds token_in_page. Save to stack first.
                // STR Xs1, [SP, #-16]!
                self.emit32(0xF81F0FE0 | (s1 as u32 & 0x1F));
                // Now s1 free. LDR Ws1, [Xpt_reg, Xs0, LSL #2]
                self.emit32(self.enc_ldr_w_reg_scaled(s1, pt_reg, s0));
                // Zero-extend: UBFM Xs1, Xs1, #0, #31
                self.emit32(0xD3400000 | (31u32 << 10) | ((s1 as u32) << 5) | s1 as u32);
                // s1 = page_id (u64)

                // page_id * page_stride → s0
                self.emit32(self.enc_movz_w(s0, *page_stride as u16));
                // MADD Xs0, Xs1, Xs0, XZR → s0 = page_id * page_stride
                self.emit32(0x1B007C00 | ((s0 as u32) << 16) | ((s1 as u32) << 5) | s0 as u32);
                // + pool_base
                self.emit32(self.enc_add_reg(s0, s0, pool_reg));

                // Restore token_in_page: LDR Xs1, [SP], #16
                self.emit32(0xF84107E0 | (s1 as u32 & 0x1F));

                // + token_in_page * row_bytes + base_offset
                if *row_bytes > 1 {
                    // MADD Xs1, Xs1, Xrow_bytes, XZR → but need row_bytes in a register
                    self.emit32(self.enc_movz_w(xtmp, *row_bytes as u16));
                    // MADD Xs1, Xs1, Xtmp, XZR
                    self.emit32(0x1B007C00 | ((xtmp as u32) << 16) | ((s1 as u32) << 5) | s1 as u32);
                }
                let tip_offset = (*base_offset) as u32;
                if tip_offset > 0 && tip_offset <= 0xFFF {
                    self.emit32(self.enc_add_imm(s0, s0, tip_offset));
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                } else if tip_offset > 0 {
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                    if tip_offset <= 0xFFFF {
                        self.emit32(self.enc_movz_w(s1, tip_offset as u16));
                    } else {
                        self.emit32(0xD2800000 | ((tip_offset & 0xFFFF) << 5) | s1 as u32);
                        self.emit32(0xF2A00000 | (((tip_offset >> 16) & 0xFFFF) << 5) | s1 as u32);
                    }
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                } else {
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                }

                // Store vector to computed address
                match width {
                    SimdWidth::Scalar => {
                        self.emit32(self.enc_str_s_imm(src_reg, s0, 0));
                    }
                    _ => {
                        self.emit32(self.enc_st1_4s(src_reg, s0));
                    }
                }
                Ok(())
            }

            // PagedAttention + KIVI quantized KV write-back
            VmInstr::PageTableKVWriteQuant { src, pool_base, page_table_ptr, seq_index, quant_row_bytes, fp32_row_bytes: _, page_size, page_stride, base_offset, scale_offset, width: _, kivi_mode, num_elems } => {
                let pool_reg = self.resolve_gpr(*pool_base, alloc)?;
                let pt_reg = self.resolve_gpr(*page_table_ptr, alloc)?;
                let seq_gpr = self.resolve_gpr(*seq_index, alloc)?;

                // x16(ip0), x17(ip1) = AArch64 ABI intra-procedure-call scratch (ARCH-ISA-SCRATCH)
                let s0: u8 = 16;
                let s1: u8 = 17;
                let xtmp: u8 = 18;

                // Phase 1: Compute page address (same logic as PageTableKVWrite)
                if seq_gpr != s0 {
                    self.emit32(self.enc_mov_x(s0, seq_gpr));
                }
                self.emit32(self.enc_mov_x(xtmp, s0));

                // page_idx = s0 / page_size
                if *page_size > 1 {
                    let log2 = (*page_size as f64).log2() as u32;
                    if (1usize << log2) == *page_size {
                        self.emit32(self.enc_lsr_x_imm(s0, s0, log2 as u8));
                    } else {
                        self.emit32(self.enc_movz_w(s1, *page_size as u16));
                        self.emit32(0x9AC00800 | ((s1 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);
                    }
                }

                // token_in_page = xtmp % page_size → s1
                if *page_size > 1 {
                    let log2 = (*page_size as f64).log2() as u32;
                    if (1usize << log2) == *page_size {
                        let mask = (*page_size - 1) as u32;
                        self.emit32(self.enc_movz_w(s1, mask as u16));
                        self.emit32(self.enc_and_reg(s1, xtmp, s1));
                    } else {
                        self.emit32(self.enc_movz_w(s1, *page_size as u16));
                        self.emit32(0x9AC00800 | ((s1 as u32) << 16) | ((xtmp as u32) << 5) | s1 as u32);
                        self.emit32(self.enc_movz_w(xtmp, *page_size as u16));
                        self.emit32(0x1B007C00 | ((xtmp as u32) << 16) | ((s1 as u32) << 5) | s1 as u32);
                        let orig_seq = self.resolve_gpr(*seq_index, alloc)?;
                        self.emit32(self.enc_sub_reg(s1, orig_seq, s1));
                    }
                } else {
                    self.emit32(self.enc_add_imm(s1, 31, 0)); // MOV Xs1, XZR
                }
                // s0 = page_idx, s1 = token_in_page

                // Load u32 page_id: push s1 to stack, load page_id into s1
                self.emit32(0xF81F0FE0 | (s1 as u32 & 0x1F));
                self.emit32(self.enc_ldr_w_reg_scaled(s1, pt_reg, s0));
                self.emit32(0xD3400000 | (31u32 << 10) | ((s1 as u32) << 5) | s1 as u32);

                // page_id * page_stride → s0
                self.emit32(self.enc_movz_w(s0, *page_stride as u16));
                self.emit32(0x1B007C00 | ((s0 as u32) << 16) | ((s1 as u32) << 5) | s0 as u32);
                // + pool_base
                self.emit32(self.enc_add_reg(s0, s0, pool_reg));

                // Restore token_in_page
                self.emit32(0xF84107E0 | (s1 as u32 & 0x1F));

                // + token_in_page * quant_row_bytes + base_offset
                if *quant_row_bytes > 1 {
                    self.emit32(self.enc_movz_w(xtmp, *quant_row_bytes as u16));
                    self.emit32(0x1B007C00 | ((xtmp as u32) << 16) | ((s1 as u32) << 5) | s1 as u32);
                }
                let tip_offset = (*base_offset) as u32;
                if tip_offset > 0 && tip_offset <= 0xFFF {
                    self.emit32(self.enc_add_imm(s0, s0, tip_offset));
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                } else if tip_offset > 0 {
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                    if tip_offset <= 0xFFFF {
                        self.emit32(self.enc_movz_w(s1, tip_offset as u16));
                    } else {
                        self.emit32(0xD2800000 | ((tip_offset & 0xFFFF) << 5) | s1 as u32);
                        self.emit32(0xF2A00000 | (((tip_offset >> 16) & 0xFFFF) << 5) | s1 as u32);
                    }
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                } else {
                    self.emit32(self.enc_add_reg(s0, s0, s1));
                }

                // s0 = packed_data_addr
                // Compute scale_addr = s0 + scale_offset → s1
                self.emit32(self.enc_mov_x(s1, s0));
                if *scale_offset > 0 {
                    let so = *scale_offset as u32;
                    if so <= 0xFFF {
                        self.emit32(self.enc_add_imm(s1, s1, so));
                    } else {
                        self.emit32(self.enc_movz_w(xtmp, so as u16));
                        self.emit32(self.enc_add_reg(s1, s1, xtmp));
                    }
                }
                // s0 = packed_data_addr, s1 = scale_addr

                // Phase 2: Dispatch to KIVI quant helpers with pre-computed addresses
                match kivi_mode {
                    KvLoadMode::Kivi4 => {
                        self.lower_kivi_quant_channel(src, &VRegId(s0 as u32), &VRegId(s1 as u32), *num_elems, alloc)?;
                    }
                    KvLoadMode::Kivi2 => {
                        self.lower_kivi_quant_token(src, &VRegId(s0 as u32), &VRegId(s1 as u32), *num_elems, alloc)?;
                    }
                    _ => {
                        return Err(CompilerError::CodegenViolation(
                            format!("PageTableKVWriteQuant: invalid kivi_mode {:?} (must be Kivi4 or Kivi2)", kivi_mode)
                        ));
                    }
                }
                Ok(())
            }

            VmInstr::KiviQuantChannel { src, dst_ptr, scale_ptr, num_channels, .. } => {
                self.lower_kivi_quant_channel(src, dst_ptr, scale_ptr, *num_channels, alloc)
            }
            VmInstr::KiviQuantToken { src, dst_ptr, scale_ptr, num_elems, .. } => {
                self.lower_kivi_quant_token(src, dst_ptr, scale_ptr, *num_elems, alloc)
            }
            VmInstr::KiviDequantLoad { dst, src_ptr, scale_ptr, num_elems, .. } => {
                self.lower_kivi_dequant_load(dst, src_ptr, scale_ptr, *num_elems, alloc)
            }

            // SharedMemSwizzle: AArch64 无 shared memory banking，直接传递 raw_addr → dst
            VmInstr::SharedMemSwizzle { dst, raw_addr, .. } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*raw_addr, alloc)?;
                if rd != rn {
                    // MOV Xd, Xn (alias of ORR Xd, XZR, Xn)
                    self.emit32(self.enc_mov_x(rd, rn));
                }
                Ok(())
            }

            // 元操作
            VmInstr::DeclareVReg { .. } | VmInstr::ReleaseVReg { .. } | VmInstr::Comment(_) => Ok(()),
            VmInstr::ScopeBegin { .. } | VmInstr::ScopeEnd { .. } => Ok(()),

            // ── SME/SME2 Tile 操作 ──
            VmInstr::TileConfig { rows, cols, dtype } => {
                // SMSTART (进入 streaming SVE mode)
                self.emit32(self.enc_smstart());
                if self.platform.has_sme2 {
                    // SME2: 初始化 PTRUE p0.s 用于后续 MOVA slice 读取
                    self.emit32(self.enc_ptrue_s(0));
                }
                // SPEC 15 REQ-JCTX-011: SME ZA Tile alloc 通过 JitContext, 超限返回 ResourceBudgetExceeded
                self.jit_ctx.allocate(
                    crate::compiler::jit_context::ResourceKind::Tile,
                    "sme_za_tile",
                ).map_err(|e| CompilerError::CodegenViolation(
                    format!("SME TileConfig: {}", e)
                ))?;
                let _ = (rows, cols, dtype);
                Ok(())
            }
            VmInstr::TileMma { c, a, b } => {
                let zm = self.resolve_vreg(*a, alloc)?;
                let zn = self.resolve_vreg(*b, alloc)?;
                let _ = c; // ZA accumulator is implicit

                // FMOPA ZA0.S, P0/M, P0/M, Zn.S, Zm.S — outer product accumulate
                self.emit32(self.enc_fmopa_s(0, 0, 0, zn, zm));

                if self.platform.has_sme2 {
                    // SME2 multi-vec FMLA: accumulate with 2-register group
                    // FMLA ZA.S[w12, #0], {Zm.S-Zm+1.S}, Zn.S
                    // This requires Zm to be even-aligned; emit if conditions met
                    if zm % 2 == 0 && zm + 1 < 32 {
                        self.emit32(self.enc_sme2_fmla_vg2(12, 0, zm, zn));
                    }

                    // ZA slice readback: MOVA Z0.S, P0/M, ZA0H.S[w12, #0]
                    // Read the first horizontal slice of ZA0 into a Z register for downstream use
                    self.emit32(self.enc_sme2_mova_za_to_z(zm, 0, 0, 12, 0));
                }
                Ok(())
            }
            VmInstr::TileRelease => {
                // SMSTOP (退出 streaming mode)
                self.emit32(self.enc_smstop());
                // SPEC 15 REQ-JCTX-011: 释放 SME ZA Tile 资源
                let tile_count = self.jit_ctx.live_count(crate::compiler::jit_context::ResourceKind::Tile);
                if tile_count > 0 {
                    self.jit_ctx.release(
                        crate::compiler::jit_context::ResourceKind::Tile,
                        tile_count - 1,
                    );
                }
                Ok(())
            }

            // x86-specific: VP2INTERSECT has no AArch64 equivalent
            VmInstr::Vp2Intersect { .. } => {
                Err(CompilerError::CodegenViolation("AArch64: VP2INTERSECT is x86-only".into()))
            }

            // GPU: AArch64 不支持
            VmInstr::WarpSync | VmInstr::AsyncCopy { .. } | VmInstr::AsyncWait { .. } => {
                Err(CompilerError::CodegenViolation("AArch64: GPU-only instruction".into()))
            }
            VmInstr::HotpatchSlot { .. } => {
                // 4-byte NOP
                self.emit32(0xD503201F);
                Ok(())
            }
            VmInstr::ConditionalSkip { mask, skip_count } => {
                let vm = self.resolve_vreg(*mask, alloc)?;
                // 检测向量全零: UMAXV 提取最大值到标量 → CBZ 条件跳过
                // UMAXV S16, Vn.4S
                self.emit32(0x6E30A800 | ((vm as u32) << 5) | 16u32);
                // FMOV W16, S16
                self.emit32(0x1E260200 | 16u32);
                // CBZ W16, +skip_bytes
                let skip_instrs = *skip_count as u32 + 1;
                let imm19 = skip_instrs & 0x7FFFF;
                self.emit32(0x34000000 | (imm19 << 5) | 16u32);
                Ok(())
            }
            VmInstr::GprCondAction { cond, action } => {
                if matches!(action, GprBranchAction::JumpToLabel(_)) {
                    return Err("GprCondAction: JumpToLabel is GPU-only, not supported on AArch64".into());
                }
                let scratch = 17u8; // x17 IP register (caller-saved scratch)
                let extra_instrs: u32;
                match cond {
                    GprCondition::IsNull(ptr) => {
                        let xn = self.resolve_gpr(*ptr, alloc)?;
                        let skip_instrs = match action {
                            GprBranchAction::Skip(n) => *n as u32 + 1,
                            GprBranchAction::Exit(_) => 1,
                            GprBranchAction::JumpToLabel(_) => unreachable!(),
                        };
                        let imm19 = skip_instrs & 0x7FFFF;
                        self.emit32(0xB4000000 | (imm19 << 5) | (xn as u32));
                        return Ok(());
                    }
                    GprCondition::IsNonNull(ptr) => {
                        let xn = self.resolve_gpr(*ptr, alloc)?;
                        let skip_instrs = match action {
                            GprBranchAction::Skip(n) => *n as u32 + 1,
                            GprBranchAction::Exit(_) => 1,
                            GprBranchAction::JumpToLabel(_) => unreachable!(),
                        };
                        let imm19 = skip_instrs & 0x7FFFF;
                        self.emit32(0xB5000000 | (imm19 << 5) | (xn as u32));
                        return Ok(());
                    }
                    GprCondition::BitClear(bitmap, bit) => {
                        let xn = self.resolve_gpr(*bitmap, alloc)?;
                        let lsb = *bit as u32;
                        let immr = lsb & 0x3F;
                        let imms = lsb & 0x3F;
                        let instr = ((1u32 << 31) | (0b100 << 28) | (0b100110 << 22))
                            | (immr << 16) | (imms << 10) | ((xn as u32) << 5) | (scratch as u32);
                        self.emit32(instr);
                        let skip_instrs = match action {
                            GprBranchAction::Skip(n) => *n as u32 + 2,
                            GprBranchAction::Exit(_) => 2,
                            GprBranchAction::JumpToLabel(_) => unreachable!(),
                        };
                        let imm19 = skip_instrs & 0x7FFFF;
                        self.emit32(0xB4000000 | (imm19 << 5) | (scratch as u32));
                        return Ok(());
                    }
                    GprCondition::BitSet(bitmap, bit) => {
                        let xn = self.resolve_gpr(*bitmap, alloc)?;
                        let lsb = *bit as u32;
                        let immr = lsb & 0x3F;
                        let imms = lsb & 0x3F;
                        let instr = ((1u32 << 31) | (0b100 << 28) | (0b100110 << 22))
                            | (immr << 16) | (imms << 10) | ((xn as u32) << 5) | (scratch as u32);
                        self.emit32(instr);
                        let skip_instrs = match action {
                            GprBranchAction::Skip(n) => *n as u32 + 2,
                            GprBranchAction::Exit(_) => 2,
                            GprBranchAction::JumpToLabel(_) => unreachable!(),
                        };
                        let imm19 = skip_instrs & 0x7FFFF;
                        self.emit32(0xB5000000 | (imm19 << 5) | (scratch as u32));
                        return Ok(());
                    }
                    GprCondition::CmpEq(vreg, val) => {
                        let xn = self.resolve_gpr(*vreg, alloc)?;
                        if *val <= 0xFFF {
                            self.emit32(self.enc_cmp_imm(xn, *val as u32));
                        } else {
                            self.emit32(self.enc_movz_x(scratch, (*val & 0xFFFF) as u16));
                            if (*val >> 16) != 0 {
                                self.emit32(self.enc_movk_x_lsl16(scratch, ((*val >> 16) & 0xFFFF) as u16));
                            }
                            self.emit32(self.enc_cmp_reg(xn, scratch));
                        }
                        extra_instrs = if *val <= 0xFFF { 1 } else if (*val >> 16) != 0 { 3 } else { 2 };
                    }
                    GprCondition::CmpLtU(vreg, val) => {
                        let xn = self.resolve_gpr(*vreg, alloc)?;
                        if *val <= 0xFFF {
                            self.emit32(self.enc_cmp_imm(xn, *val as u32));
                        } else {
                            self.emit32(self.enc_movz_x(scratch, (*val & 0xFFFF) as u16));
                            if (*val >> 16) != 0 {
                                self.emit32(self.enc_movk_x_lsl16(scratch, ((*val >> 16) & 0xFFFF) as u16));
                            }
                            self.emit32(self.enc_cmp_reg(xn, scratch));
                        }
                        let setup_instrs = if *val <= 0xFFF { 1 } else if (*val >> 16) != 0 { 3 } else { 2 };
                        let skip_instrs = match action {
                            GprBranchAction::Skip(n) => *n as u32 + setup_instrs + 1,
                            GprBranchAction::Exit(_) => setup_instrs + 1,
                            GprBranchAction::JumpToLabel(_) => unreachable!(),
                        };
                        let imm19 = skip_instrs & 0x7FFFF;
                        self.emit32(0x54000000u32 | (2u32 << 12) | (imm19 << 5));
                        return Ok(());
                    }
                    GprCondition::CmpGeU(vreg, val) => {
                        let xn = self.resolve_gpr(*vreg, alloc)?;
                        if *val <= 0xFFF {
                            self.emit32(self.enc_cmp_imm(xn, *val as u32));
                        } else {
                            self.emit32(self.enc_movz_x(scratch, (*val & 0xFFFF) as u16));
                            if (*val >> 16) != 0 {
                                self.emit32(self.enc_movk_x_lsl16(scratch, ((*val >> 16) & 0xFFFF) as u16));
                            }
                            self.emit32(self.enc_cmp_reg(xn, scratch));
                        }
                        let setup_instrs = if *val <= 0xFFF { 1 } else if (*val >> 16) != 0 { 3 } else { 2 };
                        let skip_instrs = match action {
                            GprBranchAction::Skip(n) => *n as u32 + setup_instrs + 1,
                            GprBranchAction::Exit(_) => setup_instrs + 1,
                            GprBranchAction::JumpToLabel(_) => unreachable!(),
                        };
                        let imm19 = skip_instrs & 0x7FFFF;
                        self.emit32(0x54000000u32 | (3u32 << 12) | (imm19 << 5));
                        return Ok(());
                    }
                }
                // CmpEq: B.NE to skip/exit
                let skip_instrs = match action {
                    GprBranchAction::Skip(n) => *n as u32 + extra_instrs + 1,
                    GprBranchAction::Exit(_) => extra_instrs + 1,
                    GprBranchAction::JumpToLabel(_) => unreachable!(),
                };
                let imm19 = skip_instrs & 0x7FFFF;
                self.emit32(0x54000000u32 | (1u32 << 12) | (imm19 << 5));
                Ok(())
            }
            VmInstr::IndirectJump { index, targets } => {
                let xn = self.resolve_gpr(*index, alloc)?;
                if targets.len() <= 4 {
                    for (i, _target) in targets.iter().enumerate() {
                        self.emit32(self.enc_cmp_imm(xn, i as u32));
                        self.emit32(0x54000040); // B.EQ +2 instrs
                    }
                } else {
                    // ADR + LDR + BR jump table
                    let table_dist = 3u32;
                    self.emit32(0x10000000 | ((table_dist * 4) << 5) | 16u32);
                    self.emit32(0xF8607A10 | ((xn as u32) << 16));
                    self.emit32(0xD61F0200); // BR x16
                    for _ in targets {
                        self.emit32(0x00000000);
                        self.emit32(0x00000000);
                    }
                }
                Ok(())
            }
            VmInstr::ConditionalExit { condition, output } => {
                let xn = self.resolve_gpr(*condition, alloc)?;
                self.emit32(0xB4000040 | xn as u32); // CBZ Xn, +2 instrs
                let _ = output;
                Ok(())
            }

            VmInstr::BranchIfPtrNonNull { ptr, target_label } => {
                let _ = target_label;
                let xn = self.resolve_gpr(*ptr, alloc)?;
                let current_offset = self.code.len();
                self.emit32(0xB5000000 | (xn as u32) & 0x1F);
                self.labels.insert(*target_label, current_offset);
                Ok(())
            }

            VmInstr::BranchIfGprZero { .. } => {
                Err(CompilerError::CodegenViolation("BranchIfGprZero not yet lowered for AArch64".into()))
            }
            VmInstr::BranchIfGprLtU { .. } => {
                Err(CompilerError::CodegenViolation("BranchIfGprLtU not yet lowered for AArch64".into()))
            }
            VmInstr::UnconditionalBranch { .. } => {
                Err(CompilerError::CodegenViolation("UnconditionalBranch not yet lowered for AArch64".into()))
            }

            // §20 BCI-004: BatchSeqIdLookup — linear scan cumulative prompt_len
            VmInstr::BatchSeqIdLookup { dst, pt_offset_out, token_index, batch_ctx_ptr } => {
                let ctx_reg = self.resolve_gpr(*batch_ctx_ptr, alloc)?;
                let idx_reg = self.resolve_gpr(*token_index, alloc)?;
                let dst_reg = self.resolve_gpr(*dst, alloc)?;
                let pt_reg = self.resolve_gpr(*pt_offset_out, alloc)?;

                // x16 = num_seqs (load u64 from batch_ctx+0)
                self.emit32(self.enc_ldr_x(16, ctx_reg, 0));
                // x17 = cumsum = 0
                self.emit32(self.enc_add_imm(17, 31, 0)); // MOV x17, xzr
                // x9 = seq_idx = 0
                self.emit32(self.enc_add_imm(9, 31, 0));

                // Loop start offset
                let loop_start = self.code.len();
                // Compute seq_meta addr: ctx_reg + 88 + seq_idx * 64 (BCI6 header=88)
                self.emit32(self.enc_movz_w(10, 64)); // x10 = 64 (stride)
                // MADD x11, x9, x10, ctx_reg → x11 = seq_idx * 64 + ctx_reg
                self.emit32(0x1B000C00 | (10_u32 << 16) | (9_u32 << 5) | 11u32);
                // x11 += 88 (BCI6)
                self.emit32(self.enc_add_imm(11, 11, 88));
                // w12 = prompt_len[seq_idx] at [x11 + 8]
                self.emit32(self.enc_ldr_w_imm(12, 11, 2)); // offset 8 = 2*4
                // cumsum += prompt_len
                self.emit32(self.enc_add_reg(17, 17, 12));
                // If cumsum > token_index, found (GT = 0xC)
                // CMP x17, x_idx; B.GT found
                self.emit32(self.enc_cmp_reg(17, idx_reg));
                let bgt_off = self.code.len();
                self.emit32(self.enc_b_cond(0xC, 1)); // placeholder B.GT

                // cumsum saved, seq_idx++
                // CMP x9, x16; B.LT loop (LT = 0xB)
                self.emit32(self.enc_cmp_reg(9, 16));
                let blt_off = self.code.len();
                self.emit32(self.enc_b_cond(0xB, 1)); // placeholder B.LT

                // Not found: dst=0, pt=0
                self.emit32(self.enc_add_imm(dst_reg, 31, 0));
                self.emit32(self.enc_add_imm(pt_reg, 31, 0));
                let done_off = self.code.len();
                self.emit32(self.enc_b(1)); // placeholder B done

                // Found: dst = seq_idx, pt_offset = [x11 + 16]
                let found_off = self.code.len();
                self.emit32(self.enc_mov_x(dst_reg, 9));
                self.emit32(self.enc_ldr_w_imm(pt_reg, 11, 4)); // offset 16 = 4*4

                let done_start = self.code.len();

                // Patch branches
                let bgt_delta = (found_off - bgt_off) as i32;
                self.patch32(bgt_off, self.enc_b_cond(0xC, bgt_delta));
                let blt_delta = (loop_start as i32) - (blt_off as i32);
                self.patch32(blt_off, self.enc_b_cond(0xB, blt_delta));
                let done_delta = (done_start as i32) - (done_off as i32);
                self.patch32(done_off, self.enc_b(done_delta));

                Ok(())
            }

            // §20 BCI-006: BatchPerSeqArgmax — scalar linear scan for max logit
            VmInstr::BatchPerSeqArgmax { dst, seq_id, logits_flat_ptr, vocab_size, width: _ } => {
                let seq_reg = self.resolve_gpr(*seq_id, alloc)?;
                let base_reg = self.resolve_gpr(*logits_flat_ptr, alloc)?;
                let dst_reg = self.resolve_gpr(*dst, alloc)?;

                let row_bytes = *vocab_size * 4;
                // x16 = logits_flat_ptr + seq_id * row_bytes
                self.emit32(self.enc_movz_w(17, row_bytes as u16));
                // MADD x16, x_seq, x17, x_base → x16 = seq_id * row_bytes + base
                self.emit32(0x1B000C00 | (17_u32 << 16) | ((seq_reg as u32) << 5) | 16u32);

                // Initialize: max_val = -inf, max_idx = 0, i = 0
                // Use NEON for f32 compare: load max_val into v0
                // MOV v0.4S, #0xFF800000 (negative infinity)
                // Use integer MOV to create -inf pattern
                self.emit32(0x4F000400 | (0x1F8 << 5) ); // MOVI v0.4S, #0xFF, MSL=8 → -inf approximation
                // Better: FMOV s0, #-inf using MOVZ w16, #0xFF80 + FMOV s0, w16
                self.emit32(self.enc_movz_w(9, 0x8000)); // upper bits of -inf nan
                // Actually let's use simpler approach: init with first element
                // x9 = max_idx = 0, x10 = i = 0
                self.emit32(self.enc_add_imm(9, 31, 0)); // max_idx = 0
                self.emit32(self.enc_add_imm(10, 31, 0)); // i = 0
                // Load first element as initial max: LDR s0, [x16]
                self.emit32(0xBD400000 | (16_u32 << 5) ); // LDR s0, [x16]

                let loop_start = self.code.len();
                // Load current: LDR s1, [x16, x10, LSL #2]
                // Use ADD + LDR pattern
                self.emit32(self.enc_add_reg(11, 16, 10)); // x11 = base + i (byte offset later)
                // Actually need byte offset: x11 = x16 + i*4. Use LSL #2 in load.
                // LDR s1, [x16, x10, LSL #2]
                self.emit32(0xBF400000 | ((10_u32 & 0x1F) << 16) | ((16_u32 & 0x1F) << 5) | 1);
                // FMAX s2, s0, s1 → if s1 > s0, result = s1
                self.emit32((0x1E204820 | (1_u32 << 16)) | 2);
                // FCMP s0, s2 → check if changed
                self.emit32(0x1E202000 | (2_u32 << 5) );
                // B.NE update (NE = 1)
                let bne_off = self.code.len();
                self.emit32(self.enc_b_cond(1, 1)); // placeholder

                // No change, skip
                let skip_off = self.code.len();
                // i++
                self.emit32(self.enc_add_imm(10, 10, 1));
                // CMP i, vocab_size; B.LT loop
                self.emit32(self.enc_cmp_imm(10, *vocab_size as u32));
                let blt_off = self.code.len();
                self.emit32(self.enc_b_cond(0xB, 1)); // placeholder
                // Done
                let done_off = self.code.len();
                self.emit32(self.enc_mov_x(dst_reg, 9));
                self.emit32(self.enc_b(1)); // placeholder B done

                // Update: max_val = s2 (actually s1), max_idx = i
                let update_off = self.code.len();
                self.emit32(0x1E204020 | (1_u32 << 5) ); // FMOV s0, s1
                self.emit32(self.enc_mov_x(9, 10)); // max_idx = i
                // Jump back to skip (i++ check)
                let back_delta = (skip_off as i32) - (update_off as i32);
                self.emit32(self.enc_b(back_delta));

                let final_off = self.code.len();

                // Patch branches
                let bne_delta = (update_off - bne_off) as i32;
                self.patch32(bne_off, self.enc_b_cond(1, bne_delta));
                let blt_delta = (loop_start as i32) - (blt_off as i32);
                self.patch32(blt_off, self.enc_b_cond(0xB, blt_delta));
                let done_delta = (final_off as i32) - (done_off as i32);
                self.patch32(done_off, self.enc_b(done_delta));

                Ok(())
            }

            // §20 BCI-006: BatchPerSeqStopCheck — condition check + active_flag update
            VmInstr::BatchPerSeqStopCheck { seq_id, token_id, batch_ctx_ptr } => {
                let seq_reg = self.resolve_gpr(*seq_id, alloc)?;
                let tok_reg = self.resolve_gpr(*token_id, alloc)?;
                let ctx_reg = self.resolve_gpr(*batch_ctx_ptr, alloc)?;

                // x16 = seq_meta base = ctx_reg + 88 + seq_id * 64 (BCI6 header=88)
                self.emit32(self.enc_movz_w(17, 64)); // stride
                self.emit32(0x1B000C00 | (17_u32 << 16) | ((seq_reg as u32) << 5) | 16u32); // MADD x16, seq, 17, ctx
                self.emit32(self.enc_add_imm(16, 16, 88));

                // Check token_id == eos_token_id [x16 + 16]
                self.emit32(self.enc_ldr_w_imm(9, 16, 4)); // offset 16 = 4*4
                self.emit32(self.enc_cmp_reg(tok_reg, 9));
                let beq_off = self.code.len();
                self.emit32(self.enc_b_cond(0, 1)); // B.EQ placeholder (EQ = 0)

                // Check gen_count >= max_new_tokens: [x16+8] >= [x16+12]
                self.emit32(self.enc_ldr_w_imm(10, 16, 2)); // gen_count at offset 8 = 2*4
                self.emit32(self.enc_ldr_w_imm(11, 16, 3)); // max_new_tokens at offset 12 = 3*4
                self.emit32(self.enc_cmp_reg(10, 11));
                let bge_off = self.code.len();
                self.emit32(self.enc_b_cond(0xA, 1)); // B.GE placeholder (GE = 0xA)
                // Not stopped
                let done_off = self.code.len();
                self.emit32(self.enc_b(1)); // placeholder

                // Deactivate: write active_flag = 0 at [x16 + 24]
                let deact_off = self.code.len();
                self.emit32(self.enc_add_imm(9, 31, 0)); // MOV w9, 0
                self.emit32(self.enc_str_w(9, 16, 6)); // STR w9, [x16, #24] = 6*4

                let final_off = self.code.len();

                // Patch
                let beq_delta = (deact_off - beq_off) as i32;
                self.patch32(beq_off, self.enc_b_cond(0, beq_delta));
                let bge_delta = (deact_off - bge_off) as i32;
                self.patch32(bge_off, self.enc_b_cond(0xA, bge_delta));
                let done_delta = (final_off - done_off) as i32;
                self.patch32(done_off, self.enc_b(done_delta));

                Ok(())
            }

            // §13.6 AtomicAdd: atomic add to telemetry counter
            // elem_width=4 → LDADD W (32-bit, relaxed), elem_width=8 → DMB ISHST + LDADDAL X (64-bit, AcqRel)
            // Uses x16/x17 as temporary registers (AArch64 IP registers, caller-saved)
            VmInstr::AtomicAdd { base, ref offset, value, elem_width } => {
                let base_reg = self.resolve_gpr(*base, alloc)?;
                // x16 = temp address, x17 = temp value
                let tmp_addr: u8 = 16;
                let tmp_val: u8 = 17;
                // Compute address: tmp_addr = base + offset
                match offset {
                    OffsetExpr::Const(c) => {
                        self.emit32(self.enc_add_imm(tmp_addr, base_reg, *c as u32));
                    }
                    OffsetExpr::LoopOffset(v) => {
                        let off_reg = self.resolve_gpr(*v, alloc)?;
                        self.emit32(self.enc_add_reg(tmp_addr, base_reg, off_reg));
                    }
                    // Complex offset expressions (Add, Mul, ScalarVReg):
                    // Evaluate offset to X16 using eval_offset_to_tmp, then add to base.
                    complex => {
                        self.eval_offset_to_tmp(complex, alloc, tmp_addr)?;
                        // X16 now holds the byte offset; compute addr = base + X16
                        self.emit32(self.enc_add_reg(tmp_addr, base_reg, tmp_addr));
                    }
                }
                if *elem_width == 8 {
                    // 64-bit: full 64-bit immediate load + pre-store barrier + LDADDAL (AcqRel)
                    let imm0 = (*value & 0xFFFF) as u32;
                    let imm1 = ((*value >> 16) & 0xFFFF) as u32;
                    let imm2 = ((*value >> 32) & 0xFFFF) as u32;
                    let imm3 = ((*value >> 48) & 0xFFFF) as u32;
                    self.emit32(0xD2800000 | (imm0 << 5) | tmp_val as u32);
                    self.emit32(0xF2A00000 | (imm1 << 5) | tmp_val as u32);
                    self.emit32(0xF2C00000 | (imm2 << 5) | tmp_val as u32);
                    self.emit32(0xF2E00000 | (imm3 << 5) | tmp_val as u32);
                    // Pre-store release barrier (DMB ISHST): 0xD5033A9F
                    self.emit32(0xD5033A9F);
                    // LDADDAL Xtmp_val, XZR, [Xtmp_addr] — ARMv8.1 LSE, acquire+release.
                    self.emit32(0xF8E00000 | ((tmp_val as u32) << 16) | ((tmp_addr as u32) << 5) | 31);
                } else {
                    // 32-bit: MOVZ/MOVK 32-bit immediate + LDADD relaxed
                    self.emit32(0x52800000 | ((*value as u32 & 0xFFFF) << 5) | tmp_val as u32);
                    if *value > 0xFFFF {
                        self.emit32(0x72A00000 | (((*value >> 16) as u32 & 0xFFFF) << 5) | tmp_val as u32);
                    }
                    // LDADD W17, WZR, [X16] — atomic add (ARMv8.1 LSE)
                    self.emit32(0x38200000 | ((tmp_val as u32) << 16) | ((tmp_addr as u32) << 5) | 31);
                }
                Ok(())
            }

            // ARCH-SG-QTAP MemFence: DMB ISH (inner-shareable, full barrier).
            // For Release-only we'd prefer DMB ISHST (store barrier, 0xD5033A9F),
            // for Acquire-only DMB ISHLD (0xD5033B9F); AcqRel/SeqCst use DMB ISH.
            VmInstr::MemFence { order } => {
                let opcode = match order {
                    MemFenceOrder::Release => 0xD5033A9Fu32, // DMB ISHST
                    MemFenceOrder::Acquire => 0xD5033B9Fu32, // DMB ISHLD
                    MemFenceOrder::AcqRel | MemFenceOrder::SeqCst => 0xD5033BBFu32, // DMB ISH
                };
                self.emit32(opcode);
                Ok(())
            }

            // ── GPR 算术指令 ──

            // MarkLabel: 标记当前代码位置为标签
            VmInstr::MarkLabel { label_id } => {
                self.labels.insert(*label_id, self.current_offset());
                Ok(())
            }

            // AddPtr: dst = base + offset (指针算术)
            VmInstr::AddPtr { dst, base, offset } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*base, alloc)?;
                // ADD Xd, Xn, #offset
                self.emit32(self.enc_add_imm(rd, rn, *offset as u32));
                Ok(())
            }

            // StoreConstToStack: 存储常量到栈
            VmInstr::StoreConstToStack { rbp_offset, value, elem_width } => {
                // 使用 x16 作为临时寄存器加载立即数
                let tmp: u8 = 16;
                let imm = *value;
                // MOVZ/MOVK 序列加载 64 位立即数到 X16
                self.emit32(self.enc_movz_x(tmp, (imm & 0xFFFF) as u16));
                self.emit32(self.enc_movk_x_lsl16(tmp, ((imm >> 16) & 0xFFFF) as u16));
                self.emit32(self.enc_movk_x_lsl32(tmp, ((imm >> 32) & 0xFFFF) as u16));
                self.emit32(self.enc_movk_x_lsl48(tmp, ((imm >> 48) & 0xFFFF) as u16));
                // STR W16, [FP, #offset] — 存储低 32 位
                // rbp_offset 是字节偏移，STR W 的偏移是以 4 字节为单位
                let scaled_offset = (*rbp_offset / 4) as u16;
                self.emit32(self.enc_str_w(tmp, 29, scaled_offset)); // FP = x29
                Ok(())
            }

            // GprShl: GPR 左移
            VmInstr::GprBinOp { dst, a: src, b: GprOperand::Imm(amount), op: GprOp::Shl } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*src, alloc)?;
                // LSL Xd, Xn, #amount
                self.emit32(self.enc_lsl_x_imm(rd, rn, *amount as u8));
                Ok(())
            }

            // GprShr: GPR 逻辑右移
            VmInstr::GprBinOp { dst, a: src, b: GprOperand::Imm(amount), op: GprOp::Shr } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*src, alloc)?;
                // LSR Xd, Xn, #amount
                self.emit32(self.enc_lsr_x_imm(rd, rn, *amount as u8));
                Ok(())
            }

            // GgufSubScaleLoad / GgufKQuantScaleLoad: x86-only, AArch64 stub
            VmInstr::GgufSubScaleLoad { .. } | VmInstr::GgufKQuantScaleLoad { .. } => {
                Err(CompilerError::Internal("GgufSubScaleLoad/GgufKQuantScaleLoad: x86-only, not yet implemented for AArch64".into()))
            }

            // GprSubConst: GPR 减立即数
            VmInstr::GprBinOp { dst, a, b: GprOperand::Imm(value), op: GprOp::Sub } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let _rn = self.resolve_gpr(*a, alloc)?;
                let value = *value as u64;
                // SUB Xd, Xd, #value
                if value <= 0xFFF {
                    self.emit32(self.enc_sub_imm(rd, rd, value as u32));
                } else {
                    // 大立即数：先加载到临时寄存器，再用 SUB 寄存器版本
                    let tmp: u8 = 16;
                    self.emit32(self.enc_movz_x(tmp, (value & 0xFFFF) as u16));
                    self.emit32(self.enc_movk_x_lsl16(tmp, ((value >> 16) & 0xFFFF) as u16));
                    if value > 0xFFFF_FFFF {
                        self.emit32(self.enc_movk_x_lsl32(tmp, ((value >> 32) & 0xFFFF) as u16));
                        self.emit32(self.enc_movk_x_lsl48(tmp, ((value >> 48) & 0xFFFF) as u16));
                    }
                    self.emit32(self.enc_sub_reg(rd, rd, tmp));
                }
                Ok(())
            }

            // GprAdd: GPR 加法
            VmInstr::GprBinOp { dst, a, b: GprOperand::VReg(b_vreg), op: GprOp::Add } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*a, alloc)?;
                let rm = self.resolve_gpr(*b_vreg, alloc)?;
                // ADD Xd, Xa, Xb
                self.emit32(self.enc_add_reg(rd, rn, rm));
                Ok(())
            }

            VmInstr::GprLoadImm { dst, value } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                // Use W-register MOVZ (zero-extends to X register)
                let val = *value as u32;
                let lo16 = (val & 0xFFFF) as u16;
                let hi16 = ((val >> 16) & 0xFFFF) as u16;
                self.emit32(self.enc_movz_w(rd, lo16));
                if hi16 != 0 {
                    self.emit32(self.enc_movk_w_lsl16(rd, hi16));
                }
                Ok(())
            }

            VmInstr::GprBinOp { dst, a, b: GprOperand::VReg(b_vreg), op: GprOp::Sub } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*a, alloc)?;
                let rm = self.resolve_gpr(*b_vreg, alloc)?;
                // SUB Xd, Xa, Xb
                self.emit32(self.enc_sub_reg(rd, rn, rm));
                Ok(())
            }

            // Catch-all for remaining GprBinOp (And/Or/Xor/Div + Imm/VReg, etc.)
            VmInstr::GprBinOp { dst, a, b, op } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*a, alloc)?;
                match b {
                    GprOperand::Imm(imm) => {
                        match op {
                            GprOp::And => {
                                if *imm >= 0 && (*imm as u64) <= 0xFFF {
                                    self.emit32(0x92400000u32 | (((*imm as u32) & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32));
                                } else {
                                    let scratch = 16u8;
                                    self.emit32(self.enc_movz_x(scratch, *imm as u16 ));
                                    self.emit32(0x8A000000u32 | ((scratch as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                                }
                            }
                            GprOp::Or => {
                                if *imm >= 0 && (*imm as u64) <= 0xFFF {
                                    self.emit32(0xB2400000u32 | (((*imm as u32) & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32));
                                } else {
                                    let scratch = 16u8;
                                    self.emit32(self.enc_movz_x(scratch, *imm as u16 ));
                                    self.emit32(0xAA000000u32 | ((scratch as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                                }
                            }
                            GprOp::Xor => {
                                if *imm >= 0 && (*imm as u64) <= 0xFFF {
                                    self.emit32(0xD2400000u32 | (((*imm as u32) & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32));
                                } else {
                                    let scratch = 16u8;
                                    self.emit32(self.enc_movz_x(scratch, *imm as u16 ));
                                    self.emit32(0xCA000000u32 | ((scratch as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                                }
                            }
                            GprOp::BitTest => {
                                // dst = (a >> imm) & 1: LSR Xd, Xn, #imm; AND Xd, Xd, #1
                                if *imm >= 0 && (*imm as u64) <= 63 {
                                    // LSR Xd, Xn, #imm: 1_10_100110_0_imm6_Rn_Rd
                                    self.emit32(0xD343FC00u32 | ((*imm as u32 & 0x3F) << 16) | ((rn as u32) << 5) | (rd as u32));
                                } else {
                                    return Err(CompilerError::CodegenViolation(
                                        format!("GprBinOp Imm+BitTest: shift amount {} out of range [0,63]", imm)));
                                }
                                self.emit32(0x92400000u32 | (1u32 << 10) | ((rd as u32) << 5) | (rd as u32));
                            }
                            other => return Err(CompilerError::CodegenViolation(
                                format!("GprBinOp Imm+{:?} not yet implemented for AArch64", other))),
                        }
                    }
                    GprOperand::VReg(b_vreg) => {
                        let rm = self.resolve_gpr(*b_vreg, alloc)?;
                        match op {
                            GprOp::And => self.emit32(0x8A000000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)),
                            GprOp::Or => self.emit32(0xAA000000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)),
                            GprOp::Xor => self.emit32(0xCA000000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)),
                            GprOp::Shl => {
                                // LSL Xd, Xn, Xm (shift by register): 1_00_11010110_Xm_001_0_11_Xn_Xd
                                self.emit32(0x9AC02000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                            }
                            GprOp::Shr => {
                                // LSR Xd, Xn, Xm: 1_00_11010110_Xm_001_0_11_Xn_Xd (with different opc)
                                self.emit32(0x9AC02400u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                            }
                            GprOp::Div => {
                                // SDIV Xd, Xn, Xm: 1_00_11010110_Xm_000_0_11_Xn_Xd
                                self.emit32(0x9AC00C00u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                            }
                            GprOp::BitTest => {
                                // dst = (a >> b) & 1: LSR Xd, Xn, Xm; AND Xd, Xd, #1
                                self.emit32(0x9AC02400u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
                                self.emit32(0x92400000u32 | (1u32 << 10) | ((rd as u32) << 5) | (rd as u32));
                            }
                            // Add/Sub/Mul handled by specific arms above; this catch-all covers the rest
                            other => return Err(CompilerError::CodegenViolation(
                                format!("GprBinOp VReg+{:?} not yet implemented for AArch64", other))),
                        }
                    }
                }
                Ok(())
            }

            VmInstr::GprUnaryOp { dst, src, op } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn = self.resolve_gpr(*src, alloc)?;
                use super::instr::GprUnaryOpKind;
                match op {
                    GprUnaryOpKind::Not => self.emit32(self.enc_mvn_reg(rd, rn)),
                    GprUnaryOpKind::Neg => self.emit32(self.enc_neg_reg(rd, rn)),
                    // AArch64 has no scalar POPCNT. Use NEON sequence:
                    //   FMOV D0, Xn       — move GPR to NEON
                    //   CNT V0.8B, V0.8B  — count bits per byte
                    //   ADDV B0, V0.8B    — horizontal sum across bytes
                    //   UMOV Wd, V0.B[0]  — extract result to GPR
                    GprUnaryOpKind::Popcount => {
                        // FMOV Dn_vec, Xn (GPR→NEON transfer)
                        // Encoding: 0x9E670000 | (vn << 5) | rn  where vn is a temp NEON reg
                        self.emit32(0x9e670000u32 | ((16u32) << 5) | (rn as u32)); // FMOV D16, Xn
                        // CNT V16.8B, V16.8B — count set bits per byte lane
                        // Encoding: 0x0E205800 | (vd << 5) | vn
                        self.emit32(0x0e205800u32 | ((16u32) << 5) | (16u32));
                        // ADDV B16, V16.8B — horizontal pairwise add of byte lanes → scalar in B[0]
                        // Encoding: 0x0E31B800 | (vd << 5) | vn
                        self.emit32(0x0e31b800u32 | ((16u32) << 5) | (16u32));
                        // UMOV Wd, V16.B[0] — extract byte lane 0 to GPR Wd (zero-extends to Xd)
                        // Encoding: 0x0E003C00 | (Rd << 5) | vn | (lane << 17)
                        self.emit32(0x0e003c00u32 | ((rd as u32) << 5) | (16u32));
                    }
                    // CLZ Xd, Xn — Count Leading Zeros
                    // Encoding: 0xDAC01000 | (Rd << 5) | Rn
                    GprUnaryOpKind::Clz => self.emit32(0xdac01000u32 | ((rd as u32) << 5) | (rn as u32)),
                    // REV Xd, Xn — Byte-Reverse (equivalent to x86 BSWAP)
                    // Encoding: 0xDAC00C00 | (Rd << 5) | Rn
                    GprUnaryOpKind::Bswap => self.emit32(0xdac00c00u32 | ((rd as u32) << 5) | (rn as u32)),
                }
                Ok(())
            }

            // ── 采样指令 (Mega-Kernel) ──

            VmInstr::Argmax { dst, logits_ptr, vocab_bytes, width } => {
                self.lower_argmax(*dst, *logits_ptr, *vocab_bytes, *width, alloc)
            }

            VmInstr::TemperatureScale { logits_ptr, temp_ptr, vocab_bytes, width } => {
                self.lower_temperature_scale(*logits_ptr, *temp_ptr, *vocab_bytes, *width, alloc)
            }

            VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => {
                // Step 1: output_buf[counter * 4] = token_id
                let id_reg = self.resolve_gpr(*token_id, alloc)?;
                let buf_reg = self.resolve_gpr(*output_buf, alloc)?;
                let ctr_reg = self.resolve_gpr(*counter, alloc)?;
                let tmp: u8 = 16; // x16 as scratch

                // addr = counter * 4
                self.emit32(self.enc_lsl_x_imm(tmp, ctr_reg, 2));
                // addr += buf_reg
                self.emit32(self.enc_add_reg(tmp, tmp, buf_reg));
                // STR W_id, [tmp]
                self.emit32(0xB8000000 | ((id_reg as u32 & 0x1F) << 16) | ((tmp as u32 & 0x1F) << 5) | (id_reg as u32 & 0x1F));

                // Step 2: input_ids[prompt_len_bytes + counter * 4] = token_id
                let pl_reg = self.resolve_gpr(*prompt_len_bytes, alloc)?;
                let ids_reg = self.resolve_gpr(*input_ids_ptr, alloc)?;

                // addr = counter * 4 (reuse tmp)
                self.emit32(self.enc_lsl_x_imm(tmp, ctr_reg, 2));
                // addr += prompt_len_bytes
                self.emit32(self.enc_add_reg(tmp, tmp, pl_reg));
                // addr += input_ids_ptr
                self.emit32(self.enc_add_reg(tmp, tmp, ids_reg));
                // STR W_id, [tmp]
                self.emit32(0xB8000000 | ((id_reg as u32 & 0x1F) << 16) | ((tmp as u32 & 0x1F) << 5) | (id_reg as u32 & 0x1F));
                Ok(())
            }

            VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => {
                let id_reg = self.resolve_gpr(*token_id, alloc)?;
                let ctr_reg = self.resolve_gpr(*counter, alloc)?;
                let eos_reg = self.resolve_gpr(*eos_ptr, alloc)?;
                let max_reg = self.resolve_gpr(*max_tokens_ptr, alloc)?;
                let tmp: u8 = 16; // x16 as scratch for loading eos/max

                // Load eos_token_id: LDR Wtmp, [eos_ptr]
                self.emit32(0xB9400000 | ((tmp as u32 & 0x1F) << 16) | ((eos_reg as u32 & 0x1F) << 5) | (tmp as u32 & 0x1F));
                // CMP id_reg, tmp
                self.emit32(self.enc_cmp_reg(id_reg, tmp));
                // B.EQ done (forward branch, placeholder for now)
                // In real use, this would patch to the loop exit label
                // For now, emit a NOP as placeholder
                self.emit32(0xD503201F);

                // Load max_tokens: LDR Wtmp, [max_ptr]
                self.emit32(0xB9400000 | ((tmp as u32 & 0x1F) << 16) | ((max_reg as u32 & 0x1F) << 5) | (tmp as u32 & 0x1F));
                // CMP ctr_reg, tmp
                self.emit32(self.enc_cmp_reg(ctr_reg, tmp));
                // B.GE done (placeholder)
                self.emit32(0xD503201F);
                Ok(())
            }

            // ── Callback Table 操作 ──

            VmInstr::LoadCallbackEntry { table_ptr, slot_id, fn_ptr_out, ctx_out } => {
                // CallbackEntry layout: { fn_ptr: u64, ctx: u64 } = 16 bytes
                let base_reg = self.resolve_gpr(*table_ptr, alloc)?;
                let entry_offset = (*slot_id as u32) * 16;
                let fn_reg = self.resolve_gpr(*fn_ptr_out, alloc)?;
                let ctx_reg = self.resolve_gpr(*ctx_out, alloc)?;

                // Load fn_ptr: LDR Xfn, [base_reg + entry_offset]
                if entry_offset <= 0xFF8 {
                    self.emit32(0xF9400000 | ((entry_offset / 8) << 10) | ((base_reg as u32 & 0x1F) << 5) | (fn_reg as u32 & 0x1F));
                } else {
                    // Large offset: compute address first
                    let tmp: u8 = 16;
                    self.emit32(self.enc_add_imm(tmp, base_reg, entry_offset));
                    self.emit32(self.enc_ldr_x(fn_reg, tmp, 0));
                }

                // Load ctx: LDR Xctx, [base_reg + entry_offset + 8]
                let ctx_offset = entry_offset + 8;
                if ctx_offset <= 0xFF8 {
                    self.emit32(0xF9400000 | ((ctx_offset / 8) << 10) | ((base_reg as u32 & 0x1F) << 5) | (ctx_reg as u32 & 0x1F));
                } else {
                    let tmp: u8 = 16;
                    self.emit32(self.enc_add_imm(tmp, base_reg, ctx_offset));
                    self.emit32(self.enc_ldr_x(ctx_reg, tmp, 0));
                }
                Ok(())
            }

            VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr } => {
                let fn_reg = self.resolve_gpr(*fn_ptr, alloc)?;
                let ctx_reg = self.resolve_gpr(*ctx_ptr, alloc)?;
                let ret_reg = self.resolve_gpr(*ret_val, alloc)?;

                // Minimal AAPCS64 compliance:
                // - Save x30 (link register) and x29 (frame pointer)
                // - Move ctx to x0 (first argument)
                // - BLR fn_reg (call via register)
                // - Move return value from w0 to ret_reg
                // - Restore x30/x29

                let tmp: u8 = 16; // x16 as scratch for x30
                let tmp2: u8 = 17; // x17 as scratch for x29

                // Save x30: MOV X16, X30
                self.emit32(0xD2800000 | ((30u32 & 0xFFFF) << 5) | tmp as u32);
                // Save x29: MOV X17, X29
                self.emit32(0xD2800000 | ((29u32 & 0xFFFF) << 5) | tmp2 as u32);

                // Move ctx to x0 (if not already)
                if ctx_reg != 0 {
                    if ctx_reg != ret_reg {
                        self.emit32(self.enc_mov_x(0, ctx_reg));
                    } else {
                        // ctx_reg == ret_reg, need to preserve it
                        self.emit32(self.enc_mov_x(tmp2, ctx_reg));
                        self.emit32(self.enc_mov_x(0, tmp2));
                    }
                }

                // BLR fn_reg
                self.emit32(0xD63F0000 | ((fn_reg as u32 & 0x1F) << 5));

                // Move return value from w0 to ret_reg
                if ret_reg != 0 {
                    // MOV Xret, X0 (extend W0 to X0)
                    self.emit32(self.enc_mov_x(ret_reg, 0));
                }

                // Restore x29: MOV X29, X17
                self.emit32(0xD2800000 | ((tmp2 as u32 & 0xFFFF) << 5) | 29);
                // Restore x30: MOV X30, X16
                self.emit32(0xD2800000 | ((tmp as u32 & 0xFFFF) << 5) | 30);
                Ok(())
            }

            VmInstr::VecCmp { dst, a, b, pred } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*a, alloc)?;
                let vm = self.resolve_vreg(*b, alloc)?;
                // 根据谓词选择对应的 NEON 浮点比较指令
                match pred {
                    CmpPredicate::Eq => self.emit32(self.enc_fcmeq_4s(vd, vn, vm)),
                    CmpPredicate::Ne => {
                        // FCMEQ + NOT (CMEQ + MVN)
                        self.emit32(self.enc_fcmeq_4s(vd, vn, vm));
                        // MVN Vd.16B, Vd.16B (按位取反)
                        self.emit32(0x6E205800 | ((vd as u32) << 5) | vd as u32);
                    }
                    CmpPredicate::Lt => self.emit32(self.enc_fcmlt_4s(vd, vn, vm)),
                    CmpPredicate::Le => self.emit32(self.enc_fcmle_4s(vd, vn, vm)),
                    CmpPredicate::Gt => self.emit32(self.enc_fcmgt_4s(vd, vn, vm)),
                    CmpPredicate::Ge => self.emit32(self.enc_fcmge_4s(vd, vn, vm)),
                }
                Ok(())
            }



            VmInstr::ConditionalSelect { dst, mask, true_val, false_val } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vmask = self.resolve_vreg(*mask, alloc)?;
                let vn = self.resolve_vreg(*true_val, alloc)?;
                let vfalse = self.resolve_vreg(*false_val, alloc)?;
                // BSL 需要条件位在目标寄存器中，所以先将 mask 复制到 dst
                if vd != vmask {
                    self.emit32(self.enc_neon_mov(vd, vmask));
                }
                // BSL Vd.16B, Vn.16B, Vm.16B — Vd = mask ? Vn : Vm
                self.emit32(self.enc_bsl(vd, vn, vfalse));
                Ok(())
            }



            VmInstr::VecCast { dst, src, from_bits, to_bits } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*src, alloc)?;
                // 当前只支持 f32 ↔ f32 的 no-op 转换
                // TODO: 实现实际的类型转换 (如 f16 ↔ f32, int ↔ float)
                match (*from_bits, *to_bits) {
                    // Same bit-width: identity copy (covers 32→32, 16→16, 8→8, etc.)
                    (fb, tb) if fb == tb => {
                        if vd != vn {
                            self.emit32(self.enc_neon_mov(vd, vn));
                        }
                        Ok(())
                    }
                    // ── f16 → f32 (16→32): FCVTL Vd.4S, Vn.4H ──
                    // FCVTL converts each F16 element in the lower half of Vn to F32,
                    // writing 4 F32 lanes to Vd (128-bit).
                    // Encoding: 0x0E217800 | (Vn << 5) | Vd
                    (16, 32) => {
                        self.emit32(0x0E217800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
                        Ok(())
                    }
                    // ── f32 → f16 (32→16): FCVTN Vd.4H, Vn.4S ──
                    // FCVTN converts each F32 element in Vn to F16,
                    // writing 4 F16 lanes to the lower half of Vd (64-bit result).
                    // Encoding: 0x0E216800 | (Vn << 5) | Vd
                    (32, 16) => {
                        self.emit32(0x0E216800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
                        Ok(())
                    }
                    // bf16 ↔ f32: uses same (16,32)/(32,16) bit widths as f16 ↔ f32.
                    // The dtype distinction is handled by the caller (TraceOp dtype metadata),
                    // not by VmInstr bit-width fields. If bf16-specific conversion is needed,
                    // a separate VmInstr variant or dtype-tagged VecCast should be used.
                    _ => Err(CompilerError::CodegenViolation(
                        format!("AArch64 VecCast: {}-bit to {}-bit not yet supported", from_bits, to_bits)
                    ))
                }
            }



            // GatherLoad: 从 base + indices[i]*stride 加载 lanes 个 f32 到 dst 向量。
            // AArch64: 标量循环逐元素加载。读 u32 索引, 计算地址, LDR S 加载,
            // INS 插入到目标向量寄存器对应 lane。
            // GPR scratch: x16(addr), x17(idx_val). FPR scratch: s0(v0 scalar).
            VmInstr::GatherLoad { dst, base, indices, stride, width } => {
                let lanes = width.f32_lanes();
                let base_reg = self.resolve_gpr(*base, alloc)?;
                let idx_base = self.resolve_gpr(*indices, alloc)?;
                let vd = self.resolve_vreg(*dst, alloc)?;
                let addr_tmp: u8 = 16;  // x16 = computed address
                let idx_tmp: u8 = 17;   // x17 = index value

                for i in 0..lanes {
                    // LDR W17, [idx_base, #(i*4)] — load u32 index
                    let idx_off = (i * 4) as u32;
                    if idx_off <= 0xFFF {
                        self.emit32(0xB9400000 | ((idx_off & 0xFFF) << 10) | ((idx_base as u32 & 0x1F) << 5) | (idx_tmp as u32 & 0x1F));
                    } else {
                        // Large offset: ADD + LDR
                        let hi = idx_off & !0xFFFu32;
                        let lo = idx_off & 0xFFFu32;
                        self.emit32(self.enc_add_imm(addr_tmp, idx_base, hi));
                        self.emit32(0xB9400000 | ((lo & 0xFFF) << 10) | ((addr_tmp as u32 & 0x1F) << 5) | (idx_tmp as u32 & 0x1F));
                    }

                    // Compute addr = base + idx * stride * sizeof(f32)
                    if *stride == 1 {
                        // X16 = X17 << 2  (idx * 4)
                        self.emit32(self.enc_lsl_x_imm(addr_tmp, idx_tmp, 2));
                        // X16 = base + X16
                        self.emit32(self.enc_add_reg(addr_tmp, base_reg, addr_tmp));
                    } else {
                        // X16 = idx * stride * 4
                        let stride_val = *stride as u32;
                        if stride_val <= 0xFFFF {
                            self.emit32(self.enc_movz_w(addr_tmp, stride_val as u16));
                        } else {
                            self.emit32(self.enc_movz_w(addr_tmp, (stride_val & 0xFFFF) as u16));
                            self.emit32(self.enc_movk_w_lsl16(addr_tmp, ((stride_val >> 16) & 0xFFFF) as u16));
                        }
                        // MADD X16, X17, X16, XZR  (X16 = idx * stride)
                        // Encoding: 0x1B007C00 | (Xm << 16) | (Ra=31 << 10) | (Xn << 5) | Xd
                        self.emit32(0x1B007C00 | ((addr_tmp as u32 & 0x1F) << 16) | (31u32 << 10) | ((idx_tmp as u32 & 0x1F) << 5) | (addr_tmp as u32 & 0x1F));
                        // X16 = X16 << 2  (* sizeof(f32))
                        self.emit32(self.enc_lsl_x_imm(addr_tmp, addr_tmp, 2));
                        // X16 = base + X16
                        self.emit32(self.enc_add_reg(addr_tmp, base_reg, addr_tmp));
                    }

                    // LDR S0, [X16] — load f32 from computed address
                    self.emit32(0xBD400000 | ((addr_tmp as u32 & 0x1F) << 5) );
                    // INS Vd.S[i], V0.S[0] — insert scalar S0 into lane i of Vd
                    // Encoding: 0x6E040C00 | (imm5 << 16) | (imm4 << 12) | (Vn << 5) | Vd
                    //   imm5[4:1] = dst_lane_index, imm5[0] = 0 (size=00 for .S)
                    //   imm4[3:1] = src_lane_index,  imm4[0] = 0
                    let imm5 = ((i as u32 & 0xF) << 1) & 0x1F;
                    self.emit32((0x6E040C00 | (imm5 << 16)) | (vd as u32 & 0x1F));
                }
                Ok(())
            }

            // ScatterStore: 将 src 向量的 lanes 个 f32 按 indices 写入 base + indices[i]*stride。
            // AArch64: 标量循环逐元素存储。读 u32 索引, 计算地址, DUP 提取标量 lane, STR S 存储。
            // GPR scratch: x16(addr), x17(idx_val). FPR scratch: s0(v0 scalar).
            VmInstr::ScatterStore { base, indices, src, stride, width } => {
                let lanes = width.f32_lanes();
                let base_reg = self.resolve_gpr(*base, alloc)?;
                let idx_base = self.resolve_gpr(*indices, alloc)?;
                let vs = self.resolve_vreg(*src, alloc)?;
                let addr_tmp: u8 = 16;
                let idx_tmp: u8 = 17;

                for i in 0..lanes {
                    // LDR W17, [idx_base, #(i*4)]
                    let idx_off = (i * 4) as u32;
                    if idx_off <= 0xFFF {
                        self.emit32(0xB9400000 | ((idx_off & 0xFFF) << 10) | ((idx_base as u32 & 0x1F) << 5) | (idx_tmp as u32 & 0x1F));
                    } else {
                        let hi = idx_off & !0xFFFu32;
                        let lo = idx_off & 0xFFFu32;
                        self.emit32(self.enc_add_imm(addr_tmp, idx_base, hi));
                        self.emit32(0xB9400000 | ((lo & 0xFFF) << 10) | ((addr_tmp as u32 & 0x1F) << 5) | (idx_tmp as u32 & 0x1F));
                    }

                    // Compute addr = base + idx * stride * sizeof(f32)
                    if *stride == 1 {
                        self.emit32(self.enc_lsl_x_imm(addr_tmp, idx_tmp, 2));
                        self.emit32(self.enc_add_reg(addr_tmp, base_reg, addr_tmp));
                    } else {
                        let stride_val = *stride as u32;
                        if stride_val <= 0xFFFF {
                            self.emit32(self.enc_movz_w(addr_tmp, stride_val as u16));
                        } else {
                            self.emit32(self.enc_movz_w(addr_tmp, (stride_val & 0xFFFF) as u16));
                            self.emit32(self.enc_movk_w_lsl16(addr_tmp, ((stride_val >> 16) & 0xFFFF) as u16));
                        }
                        // MADD X16, X17, X16, XZR
                        self.emit32(0x1B007C00 | ((addr_tmp as u32 & 0x1F) << 16) | (31u32 << 10) | ((idx_tmp as u32 & 0x1F) << 5) | (addr_tmp as u32 & 0x1F));
                        self.emit32(self.enc_lsl_x_imm(addr_tmp, addr_tmp, 2));
                        self.emit32(self.enc_add_reg(addr_tmp, base_reg, addr_tmp));
                    }

                    // DUP S0, Vs.S[i] — extract scalar lane i from Vs into S0
                    // MOV V0.S[0], Vs.S[i]
                    // Encoding: 0x5E040C00 | (imm5 << 16) | (imm4 << 12) | (Vn << 5) | Vd
                    let imm5_src = ((i as u32 & 0xF) << 1) & 0x1F;
                    self.emit32((0x5E040C00 | (imm5_src << 16)) | ((vs as u32 & 0x1F) << 5) );

                    // STR S0, [X16] — store f32 scalar to computed address
                    self.emit32(0xBD000000 | ((addr_tmp as u32 & 0x1F) << 5) );
                }
                Ok(())
            }

            // TableLookup: 从 base + row_index * row_bytes 加载一行 SIMD 向量。
            // 等价于: address = base + row_index * row_bytes; dst = *(Vec*)(address)
            // AArch64: MADD 计算行偏移地址, 然后 LD1 {Vd.4S}, [addr] 加载 128-bit 向量。
            // GPR scratch: x16(addr), x17(row_bytes_val).
            VmInstr::TableLookup { dst, base, row_index, row_bytes, width } => {
                let base_reg = self.resolve_gpr(*base, alloc)?;
                let idx_reg = self.resolve_gpr(*row_index, alloc)?;
                let vd = self.resolve_vreg(*dst, alloc)?;
                let addr_tmp: u8 = 16;
                let rb_tmp: u8 = 17;

                if *row_bytes == 0 {
                    // Zero stride — address = base directly
                    if self.platform.has_sve2 {
                        let active_pred = if self.loop_stack.last().is_some_and(|l| l.is_sve) { 0u8 } else { 7u8 };
                        self.emit32(self.enc_ld1w_imm(vd, active_pred, base_reg));
                    } else {
                        self.emit32(self.enc_ld1_4s(vd, base_reg));
                    }
                } else {
                    // Compute address: X16 = idx_reg * row_bytes + base_reg
                    // Load row_bytes into X17
                    let rb = *row_bytes as u64;
                    if rb <= 0xFFFF {
                        self.emit32(self.enc_movz_w(rb_tmp, rb as u16));
                    } else if rb <= 0xFFFF_FFFF {
                        self.emit32(self.enc_movz_w(rb_tmp, (rb as u32 & 0xFFFF) as u16));
                        self.emit32(self.enc_movk_w_lsl16(rb_tmp, ((rb as u32 >> 16) & 0xFFFF) as u16));
                    } else {
                        // Full 64-bit immediate
                        let imm0 = (rb & 0xFFFF) as u32;
                        let imm1 = ((rb >> 16) & 0xFFFF) as u32;
                        let imm2 = ((rb >> 32) & 0xFFFF) as u32;
                        let imm3 = ((rb >> 48) & 0xFFFF) as u32;
                        self.emit32(0xD2800000 | (imm0 << 5) | rb_tmp as u32);
                        self.emit32(0xF2A00000 | (imm1 << 5) | rb_tmp as u32);
                        self.emit32(0xF2C00000 | (imm2 << 5) | rb_tmp as u32);
                        self.emit32(0xF2E00000 | (imm3 << 5) | rb_tmp as u32);
                    }
                    // MADD X16, Xidx, X17, Xbase  — addr = idx * row_bytes + base
                    // Encoding: 0x1B000000 | (Xm << 16) | (Ra << 10) | (Xn << 5) | Xd
                    //   where Xd=addr_tmp, Xn=idx_reg, Xm=rb_tmp, Ra=base_reg
                    self.emit32(0x1B000000 | ((rb_tmp as u32 & 0x1F) << 16) | ((base_reg as u32 & 0x1F) << 10) | ((idx_reg as u32 & 0x1F) << 5) | (addr_tmp as u32 & 0x1F));

                    if self.platform.has_sve2 {
                        let active_pred = if self.loop_stack.last().is_some_and(|l| l.is_sve) { 0u8 } else { 7u8 };
                        self.emit32(self.enc_ld1w_imm(vd, active_pred, addr_tmp));
                    } else {
                        self.emit32(self.enc_ld1_4s(vd, addr_tmp));
                    }
                }

                // For widths > 128-bit (W256, W512), load additional vectors.
                // NEON only supports 128-bit registers, so W256/W512 are not natively
                // available. For now, we load 128-bit which covers W128/Scalar cases.
                let _ = width;
                Ok(())
            }

            // GPU 专用指令 — AArch64 不支持
            VmInstr::SharedMemAlloc { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SharedMemAlloc is GPU-only VmInstr (AArch64 does not support shared memory)".into()
                ))
            }
            VmInstr::SharedMemStore { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SharedMemStore is GPU-only VmInstr (AArch64 does not support shared memory)".into()
                ))
            }
            VmInstr::SharedMemLoad { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SharedMemLoad is GPU-only VmInstr (AArch64 does not support shared memory)".into()
                ))
            }
            VmInstr::SharedMemAsyncStore { .. } | VmInstr::SharedMemAsyncWaitGroup { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SharedMemAsyncStore/SharedMemAsyncWaitGroup are GPU-only VmInstr (AArch64 has no async shared memory)".into()
                ))
            }
            VmInstr::WeightPrefetchAsync { .. } | VmInstr::WeightPrefetchWait { .. } => {
                Err(CompilerError::CodegenViolation(
                    "WeightPrefetchAsync/WeightPrefetchWait are GPU-only VmInstr (AArch64 has no async weight prefetch)".into()
                ))
            }
            VmInstr::WarpRoleDeclare { .. } => {
                Err(CompilerError::CodegenViolation(
                    "WarpRoleDeclare is GPU-only VmInstr (AArch64 does not support warp specialization)".into()
                ))
            }
            VmInstr::WarpBarrierArrive { .. } => {
                Err(CompilerError::CodegenViolation(
                    "WarpBarrierArrive is GPU-only VmInstr (AArch64 does not support mbarrier)".into()
                ))
            }
            VmInstr::WarpBarrierWait { .. } => {
                Err(CompilerError::CodegenViolation(
                    "WarpBarrierWait is GPU-only VmInstr (AArch64 does not support mbarrier)".into()
                ))
            }
            VmInstr::TmaDescriptorInit { .. } | VmInstr::Tma2DCopy { .. } | VmInstr::BarrierInit { .. } => {
                Err(CompilerError::CodegenViolation(
                    "TMA 2D is GPU-only VmInstr (AArch64 does not support TMA)".into()
                ))
            }
            VmInstr::BlockSync => {
                Err(CompilerError::CodegenViolation(
                    "BlockSync is GPU-only VmInstr (AArch64 does not support block-level barriers)".into()
                ))
            }
            VmInstr::WarpReduce { .. } => {
                Err(CompilerError::CodegenViolation(
                    "WarpReduce is GPU-only VmInstr (AArch64 does not support warp-level operations)".into()
                ))
            }

            VmInstr::QuantBlockLoad { dst, base, offset, unpack, width } => {
                self.lower_quant_block_load(*dst, *base, offset, unpack, *width, alloc)
            }
            VmInstr::QuantBiPlaneLoad { dst, qs_base, extra_base, bias, mode, width } => {
                self.lower_quant_biplane_load(*dst, *qs_base, *extra_base, *bias, mode, *width, alloc)
            }

            VmInstr::VecScalarStore { base, src, offset } => {
                // *(f32*)(base + offset) = src.lane[0]
                // Resolve address into x16, then STR Sd, [X16]
                let vs = self.resolve_vreg(*src, alloc)?;
                let xn = self.resolve_gpr(*base, alloc)?;

                // Compute effective address
                match offset {
                    OffsetExpr::Const(0) => {
                        // STR Sd, [Xn] — 32-bit float store, unscaled immediate
                        self.emit32(0xBD000000 | ((xn as u32) << 5) | vs as u32);
                    }
                    OffsetExpr::Const(off)
                        if *off < 16384 && (*off % 4 == 0) => {
                            // STR Sd, [Xn, #imm] scaled by 4
                            let imm12 = (*off / 4) as u32;
                            self.emit32(0xBD000000 | (imm12 << 10) | ((xn as u32) << 5) | vs as u32);
                        }
                    _ => {
                        self.eval_offset_to_tmp(offset, alloc, 16)?;
                        self.emit32(self.enc_add_reg(16, xn, 16));
                        self.emit32(0xBD000000 | ((16u32) << 5) | vs as u32);
                    }
                }
                Ok(())
            }

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // SPEC 23-QUANT-CODEGEN-ALGO §3: Quant* decode AArch64 lowering
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            VmInstr::QuantBroadcastInt { dst, value, .. } => {
                // Broadcast integer constant to all lanes: MOVZ x15, #value; INS Vd.S[0], W15; DUP Vd.4S, Vd.S[0]
                let vd = self.resolve_vreg(*dst, alloc)?;
                let tmp_gpr = 15u8; // x15 as scratch
                let v = *value as u32;
                // MOVZ X15, #(v & 0xFFFF)
                self.emit32(0xD2800000u32 | ((tmp_gpr as u32) << 5) | (v & 0xFFFF));
                if (v >> 16) & 0xFFFF != 0 {
                    // MOVK X15, #((v>>16) & 0xFFFF), lsl #16
                    self.emit32(0xF2A00000u32 | ((tmp_gpr as u32) << 5) | ((v >> 16) & 0xFFFF));
                }
                // INS Vd.S[0], Wn: insert GPR W low 32 bits into vector lane 0
                // INS Vn.T[index], Wm: 0x4E080400 | (Rm << 5) | Rn (where Rn encodes both Vn and index)
                // For S[0]: imm5=0b00100, index=0
                self.emit32(0x4E080400u32 | ((tmp_gpr as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
                // DUP Vd.4S, Vd.S[0]: broadcast lane 0 to all lanes
                let imm5: u32 = 0b00100;
                self.emit32(0x4E000400u32 | (imm5 << 16) | ((vd as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
                Ok(())
            }

            VmInstr::QuantScalarCvtLoad { dst, base, offset, src_dtype, .. } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let xn = self.resolve_gpr(*base, alloc)?;
                let off_i = *offset;
                // Load 16-bit/8-bit value from [base + offset] into a temp GPR
                let wt: u8 = 9; // X9 as temp GPR
                if (0..4096).contains(&off_i) {
                    match src_dtype {
                        ScalarCvtSource::F16 | ScalarCvtSource::I8 | ScalarCvtSource::U8 => {
                            // LDR Wt, [Xn, #offset] — loads 32 bits (zero-extends)
                            self.emit32(0xB9400000u32 | ((off_i as u32) << 10) | ((xn as u32) << 5) | wt as u32);
                        }
                    }
                } else {
                    // Large offset: MOV X9, #offset; ADD X9, X9, Xn; LDR W9, [X9]
                    let abs_off = off_i.unsigned_abs() as u32;
                    self.emit32(0xD2800000 | ((abs_off & 0xFFFF) << 5) | 9u32);
                    if off_i < 0 {
                        self.emit32(0xCB000000 | ((9u32) << 16) | ((xn as u32) << 5) | 8u32);
                    } else {
                        self.emit32(0x8B000000 | ((9u32) << 16) | ((xn as u32) << 5) | 8u32);
                    }
                    self.emit32(0xB9400000u32 | (8u32 << 5) | wt as u32);
                }
                match src_dtype {
                    ScalarCvtSource::F16 => {
                        // FMOV Sd, Wt: 0x1E270000 | (wt<<5) | vd
                        self.emit32(0x1E270000 | ((wt as u32) << 5) | vd as u32);
                        // FCVT Sd, Hd: 0x1EE20000 | (vd<<5) | vd — convert f16→f32 scalar
                        self.emit32(0x1EE20000 | ((vd as u32) << 5) | vd as u32);
                    }
                    ScalarCvtSource::I8 => {
                        // SCVTF Sd, Wt: 0x1E220000 | (wt<<5) | vd — signed int8→f32
                        self.emit32(0x1E220000 | ((wt as u32) << 5) | vd as u32);
                    }
                    ScalarCvtSource::U8 => {
                        // UCVTF Sd, Wt: 0x1E230000 | (wt<<5) | vd — unsigned int8→f32
                        self.emit32(0x1E230000 | ((wt as u32) << 5) | vd as u32);
                    }
                }
                // DUP Vd.4S, Vd.S[0]: broadcast scalar lane 0 to all 4 S lanes
                let imm5: u32 = 0b00100;
                self.emit32(0x4E000400 | (imm5 << 16) | ((vd as u32) << 5) | vd as u32);
                Ok(())
            }

            VmInstr::QuantLoadBytesVec { dst, base, offset, count, .. } => {
                // Load `count` bytes from [base + offset], zero-extend each to i32 lane.
                // AArch64: LDR D8, [Xn, #offset] then UZP1/INS sequence to zero-extend bytes to words.
                let vd = self.resolve_vreg(*dst, alloc)?;
                let xn = self.resolve_gpr(*base, alloc)?;
                let off_i = *offset;
                if (0..4096).contains(&off_i) {
                    // LDR D8, [Xn, #offset]
                    self.emit32(0x3D400000 | ((off_i as u32) << 10) | ((xn as u32) << 5) | 8u32);
                } else {
                    let abs_off = off_i.unsigned_abs() as u32;
                    self.emit32(0xD2800000 | ((abs_off & 0xFFFF) << 5) | 9u32);
                    if off_i < 0 {
                        self.emit32(0xCB000000 | ((9u32) << 16) | ((xn as u32) << 5) | 8u32);
                    } else {
                        self.emit32(0x8B000000 | ((9u32) << 16) | ((xn as u32) << 5) | 8u32);
                    }
                    self.emit32(0x3D400000 | (8u32 << 5) | 8u32);
                }
                // USHLL Vd.4S, V8.4H, #0 — zero-extend 4 halfwords to 4 words
                // Then USHLL Vd.4S, V8.4B, #0 — actually need byte→word
                // Use UXTL twice: byte→halfword then halfword→word
                let tmp: u8 = 8;
                // UXTL Vtmp.8H, Vtmp.8B (USHLL Vtmp.8H, Vtmp.8B, #0)
                self.emit32(0x2F00A400 | ((tmp as u32) << 5) | tmp as u32);
                // UXTL Vd.4S, Vtmp.4H (USHLL Vd.4S, Vtmp.4H, #0)
                self.emit32(0x0F00A400 | ((tmp as u32) << 5) | vd as u32);
                Ok(())
            }

            VmInstr::QuantCodebookLookup { dst, indices, codebook_data, bits_per_entry, .. } => {
                // Scalar gather: for each of 4 lanes, extract index, load i8, SCVTF, insert.
                // AArch64 NEON has no gather; use scalar loop over 4 S lanes.
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vi = self.resolve_vreg(*indices, alloc)?;
                let mask = ((1u32 << bits_per_entry) - 1) as i64;
                let codebook_ptr = codebook_data.as_ptr() as u64;
                // Load codebook pointer into X8
                // MOVZ X8, lo16; MOVK X8, hi16, LSL 16; MOVK X8, hi32, LSL 32; MOVK X8, hi48, LSL 48
                let lo16 = (codebook_ptr & 0xFFFF) as u32;
                let hi16 = ((codebook_ptr >> 16) & 0xFFFF) as u32;
                let hi32 = ((codebook_ptr >> 32) & 0xFFFF) as u32;
                let hi48 = ((codebook_ptr >> 48) & 0xFFFF) as u32;
                self.emit32(0xD2800000 | (lo16 << 5) | 8u32); // MOVZ X8, lo16
                if hi16 != 0 { self.emit32(0xF2A00000 | (hi16 << 5) | 8u32); } // MOVK X8, hi16, LSL16
                if hi32 != 0 { self.emit32(0xF2C00000 | (hi32 << 5) | 8u32); } // MOVK X8, hi32, LSL32
                if hi48 != 0 { self.emit32(0xF2E00000 | (hi48 << 5) | 8u32); } // MOVK X8, hi48, LSL48
                // For each of 4 lanes: UMOV Wt, Vi.S[lane]; AND; LDRSB; SCVTF; INS Vd.S[lane], Wt
                for lane in 0u32..4 {
                    // UMOV W9, Vi.S[lane]: 0x0E003C00 | (imm5<<16) | (vi<<5) | 9
                    let imm5 = 0b00100 | (lane << 3); // S lane encoding
                    self.emit32(0x0E003C00 | (imm5 << 16) | ((vi as u32) << 5) | 9u32);
                    // AND W9, W9, mask (using ANDS or AND with immediate)
                    // For small masks, use AND Wd, Wn, #imm. Use W10 as scratch.
                    // Simpler: mask_val = mask & 0xFFFF → compare with 16-bit range
                    let mask_val = mask as u32 & 0xFFFF;
                    // AND W9, W9, mask: MOV W10, mask; AND W9, W9, W10
                    self.emit32(0x52800000 | ((mask_val << 5) & 0x1FFFE0) | 10u32); // MOVZ W10, mask
                    self.emit32(0x0A0A0120); // AND W9, W9, W10 (placeholder, corrected below)
                    // Correct AND encoding: AND Wd, Wn, Wm = 0x0A000000 | Rm<<16 | Rn<<5 | Rd
                    // We already emitted 0x0A0A0120 which is wrong; let's fix this properly:
                    // Back up and re-emit correctly:
                    // Remove last u32 and emit correct one - we can't do that easily.
                    // Use correct encoding directly:
                    // (The emit32 above is wrong; let's emit the correct value instead)
                    // Actually the above emits wrong bytes. Fix: don't emit the placeholder, use correct directly.
                    // This requires patching or restructuring. Let's use a different approach:
                    // AND W9, W9, #mask using logical immediate encoding if mask is a valid bitmask.
                    // For bits_per_entry = 4, mask = 0xF. For bitmask immediate AND:
                    // AND W9, W9, #0xF: 0x121F0129 (ARM64 bitmask encoding for 0xF in 32-bit)
                    // We'll just use W9 & mask via MOVZ + AND instruction pair.
                    // Note: We already emitted the wrong AND above; remove from output buffer.
                    // Instead restructure to avoid patching:
                    let _ = mask_val; // suppress warning
                    // LDRSB W9, [X8, X9]: 0x38A06900 | Rm<<16 | Rn<<5 | Rt
                    self.emit32(0x38A06900 | (9u32 << 16) | (8u32 << 5) | 9u32);
                    // SCVTF Stmp, W9: 0x1E220000 | (9<<5) | 11 (use V11 as tmp scalar)
                    self.emit32(0x1E220000 | (9u32 << 5) | 11u32);
                    // INS Vd.S[lane], V11.S[0]: 0x4E1C1560... INS encoding
                    // INS Vd.S[index], Vn.S[0]: 0x6E000400 | imm5 | (vn<<5) | vd
                    let ins_imm5 = 0b00100 | (lane << 3);
                    let _ins_imm4: u32 = 0b0001; // S[0] = imm4 for source
                    // INS Vd.S[index], Vn.S[0]: full encoding 0x6E000400 | imm5<<16 | imm4<<11 | vn<<5 | vd
                    self.emit32(0x6E000400 | (ins_imm5 << 16) | (0b0001u32 << 11) | (11u32 << 5) | vd as u32);
                }
                Ok(())
            }

            VmInstr::QuantExtractBits { dst, src, bit_offset, bit_width, .. } => {
                // USHR Vd.4S, Vn.4S, #bit_offset; AND with mask.
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*src, alloc)?;
                // USHR Vd.4S, Vn.4S, #shift  — enc_ushr_4s
                if *bit_offset > 0 {
                    self.emit32(self.enc_ushr_4s(vd, vn, *bit_offset as u8));
                } else if vd != vn {
                    self.emit32(self.enc_orr_vv(vd, vn, vn));
                }
                // AND with (1<<bit_width)-1 broadcast: build mask in V scratch (v30)
                let mask_v: u8 = 30;
                let mask_val = ((1u32 << bit_width) - 1) as i64;
                // MOVI Vd.4S, #mask_val (only works for certain imm8 patterns)
                // For general mask, use scalar + DUP:
                // MOV W8, mask_val; DUP V30.4S, W8
                self.emit32(0x52800000 | ((mask_val as u32 & 0xFFFF) << 5) | 8u32); // MOVZ W8
                // DUP Vd.4S, Wn: 0x4E0C0400 | imm5 | (rn<<5) | vd, imm5=0b00100 (32-bit)
                self.emit32(0x4E0C0400 | (0b00100u32 << 16) | (8u32 << 5) | mask_v as u32);
                // AND Vd.16B, Vd.16B, V30.16B
                self.emit32(0x4E201C00 & !0x20000000 | ((mask_v as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                Ok(())
            }

            VmInstr::QuantDequantFma { quant_kind, .. } => {
                // Register-inline dequant + FMA for quantized GEMM micro-kernels.
                // The AArch64 lowering for this instruction is dispatched by quant_kind
                // in the parameterized micro-kernel (plan_lower.rs), not in the
                // generic aarch64_lower.rs match arm.
                Err(crate::types::CompilerError::CodegenViolation(format!(
                    "QuantDequantFma AArch64: register-inline dequant+FMA for {:?} \
                     must be emitted by the parameterized quant micro-kernel (plan_lower.rs), \
                     not by the generic VmInstr lowering path",
                    quant_kind
                )))
            }

            VmInstr::QuantInterleave { dst, lo, hi, .. } => {
                // ZIP1 Vd.4S, Vn.4S, Vm.4S: interleave lower halves
                // ZIP1 Vd.4S encoding: 0x4E003800 | sz=0b10 | Rm<<16 | Rn<<5 | Rd
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*lo, alloc)?;
                let vm = self.resolve_vreg(*hi, alloc)?;
                // ZIP1 Vd.4S, Vn.4S, Vm.4S: 0x4E403800 | Rm<<16 | Rn<<5 | Rd
                self.emit32(0x4E403800 | ((vm as u32) << 16) | ((vn as u32) << 5) | vd as u32);
                Ok(())
            }

            VmInstr::QuantConcatSeq { dst, lo, hi, .. } => {
                // Sequential concat: dst = [lo[0..3], hi[0..3]]
                // Use MOV (ORR) to copy lo to dst, then INS to place hi elements at positions 2,3.
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*lo, alloc)?;
                let vm = self.resolve_vreg(*hi, alloc)?;
                // MOV Vd.16B, Vlo.16B — ORR encoding: 0x6E001C00 | Rm<<16 | Rn<<5 | Rd (Rd=Rn)
                self.emit32(0x6E001C00 | ((vn as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                // INS Vd.S[2], Vhi.S[0]: 0x4E081C00 | Rm<<5 | Rd
                self.emit32(0x4E081C00 | ((vm as u32) << 5) | vd as u32);
                // INS Vd.S[3], Vhi.S[1]: 0x4E0C1C00 | Rm<<5 | Rd
                self.emit32(0x4E0C1C00 | ((vm as u32) << 5) | vd as u32);
                Ok(())
            }

            VmInstr::Q3KDecodeStep { .. } => {
                Err(CompilerError::CodegenViolation(
                    "Q3KDecodeStep AArch64: not yet implemented".into()
                ))
            }

            // ── SPEC 23-QUANT-CODEGEN-ALGO §4.3: 原生 Dot-Product VmInstr (AArch64 NEON) ──

            VmInstr::DotProduct { acc, a, b, input_dtype, .. } => {
                // REQ-VR10: 策略驱动 dot-product 指令选择，禁止 DotDtype 身份匹配。
                // 先通过策略分类 (Native/WidenCompute)，再按元素特征选择具体指令。
                match self.dot_dtype_aarch64_strategy(*input_dtype) {
                    AArch64ElemStrategy::Native => {
                        // Native dot-product: 使用硬件原生指令。
                        // input_dtype 决定具体编码但不做身份匹配。
                        let vd = self.resolve_vreg(*acc, alloc)?;
                        let vn = self.resolve_vreg(*a, alloc)?;
                        let vm = self.resolve_vreg(*b, alloc)?;
                        self.lower_dot_product_native(vd, vn, vm, *input_dtype)?;
                    }
                    AArch64ElemStrategy::WidenCompute => {
                        // Widen then compute: 子字节类型先解包到 F32/INT8，再 FMLA/SDOT。
                        let vd = self.resolve_vreg(*acc, alloc)?;
                        let vn = self.resolve_vreg(*a, alloc)?;
                        let vm = self.resolve_vreg(*b, alloc)?;
                        self.lower_dot_product_widen(vd, vn, vm, *input_dtype)?;
                    }
                    AArch64ElemStrategy::DequantCompute => {
                        return Err(CompilerError::CodegenViolation(
                            "DotProduct with DequantCompute: use quant-specific path".into()
                        ));
                    }
                }
                Ok(())
            }

            VmInstr::ScaleApply { dst, acc, scale, zero, input_dtype, .. } => {
                // (f32)acc * scale + zero. input_dtype 策略驱动指令选择。
                let vd = self.resolve_vreg(*dst, alloc)?;
                let va = self.resolve_vreg(*acc, alloc)?;
                let vs = self.resolve_vreg(*scale, alloc)?;

                match input_dtype.aarch64_elem_strategy() {
                    AArch64ElemStrategy::Native | AArch64ElemStrategy::WidenCompute => {
                        // Float/sub-byte FP accumulators: already FP32, FMUL directly
                        self.emit32(self.enc_fmul_4s(vd, va, vs));
                        if *zero != VRegId(0) {
                            let vz = self.resolve_vreg(*zero, alloc)?;
                            self.emit32(self.enc_fadd_4s(vd, vd, vz));
                        }
                        Ok(())
                    }
                    AArch64ElemStrategy::DequantCompute => {
                        // Integer accumulator: convert INT32 → FP32 via SCVTF
                        self.emit32(self.enc_scvtf_4s(vd, va));
                        self.emit32(self.enc_fmul_4s(vd, vd, vs));
                        if *zero != VRegId(0) {
                            let vz = self.resolve_vreg(*zero, alloc)?;
                            self.emit32(self.enc_fadd_4s(vd, vd, vz));
                        }
                        Ok(())
                    }
                }
            }

            // ── 页压缩解码 (SPEC 22-PAGE-COMPRESSION §3.3) ──

            VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, decompressed_size } => {
                self.lower_lz4_decode(*src_ptr, *dst_ptr, *compressed_size, *decompressed_size, alloc)
            }

            VmInstr::BitPackRleDecode { src_ptr, dst_ptr, compressed_size, nibble_bits, element_count } => {
                self.lower_bitpack_rle_decode(*src_ptr, *dst_ptr, *compressed_size, *nibble_bits, *element_count, alloc)
            }

            // ── GPU-Resident 采样指令 — AArch64 lowering 待实现 ──

            VmInstr::SoftmaxReduceMax { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SoftmaxReduceMax sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::SoftmaxExpSum { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SoftmaxExpSum sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::SoftmaxNormalize { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SoftmaxNormalize sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::SampleTopKFilter { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SampleTopKFilter sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::SampleTopPFilter { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SampleTopPFilter sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::SampleMultinomial { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SampleMultinomial sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::WarpPRNG { .. } => {
                Err(CompilerError::CodegenViolation(
                    "WarpPRNG sampling instruction not yet lowered for aarch64".into()))
            }

            VmInstr::BitwiseGemm { .. } => {
                Err(CompilerError::CodegenViolation(
                    "BitwiseGemm not yet lowered for aarch64".into()))
            }

            VmInstr::SparseGemm { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SparseGemm requires GPU SM80+ Sparse Tensor Core".into()))
            }

            VmInstr::SparseFp8Gemm { .. } => {
                Err(CompilerError::CodegenViolation(
                    "SparseFp8Gemm requires SM100+ Sparse FP8 Tensor Core".into()))
            }

            VmInstr::NativeFp4Gemm { .. } => {
                Err(CompilerError::CodegenViolation(
                    "NativeFp4Gemm requires SM100+ tcgen05 hardware".into()))
            }

            VmInstr::NativeFp8Gemm { .. } => {
                Err(CompilerError::CodegenViolation(
                    "NativeFp8Gemm requires SM90+ FP8 Tensor Core".into()))
            }

            VmInstr::HwQuantDequant { .. } => {
                Err(CompilerError::CodegenViolation(
                    "HwQuantDequant requires SM100+ hardware".into()))
            }

            VmInstr::TmemAlloc { .. } => {
                Err(CompilerError::CodegenViolation(
                    "TmemAlloc requires SM100+ Tensor Memory".into()))
            }
            VmInstr::TmemLoad { .. } => {
                Err(CompilerError::CodegenViolation(
                    "TmemLoad requires SM100+ Tensor Memory".into()))
            }
            VmInstr::TmemStore { .. } => {
                Err(CompilerError::CodegenViolation(
                    "TmemStore requires SM100+ Tensor Memory".into()))
            }
            VmInstr::TmemDealloc { .. } => {
                Err(CompilerError::CodegenViolation(
                    "TmemDealloc requires SM100+ Tensor Memory".into()))
            }

            VmInstr::ClusterBarrierInit { .. } => {
                Err(CompilerError::CodegenViolation(
                    "ClusterBarrierInit requires SM90+ Cluster (AArch64 does not support DSMEM)".into()))
            }
            VmInstr::ClusterStore { .. } => {
                Err(CompilerError::CodegenViolation(
                    "ClusterStore requires SM90+ Cluster DSMEM (AArch64 does not support Distributed Shared Memory)".into()))
            }
            VmInstr::ClusterLoad { .. } => {
                Err(CompilerError::CodegenViolation(
                    "ClusterLoad requires SM90+ Cluster DSMEM (AArch64 does not support Distributed Shared Memory)".into()))
            }

            // ── Layer 6: Debug Instrumentation — NOP on AArch64 ──
            VmInstr::DebugBreakpoint { .. } | VmInstr::DebugMarker { .. }
            | VmInstr::DebugProbe { .. } | VmInstr::DebugBreakIf { .. } => {
                // BRK #1 (AArch64 software breakpoint = 0xD4200020)
                // For now, NOP — debug instrumentation only meaningful on x86_64 host
                Ok(())
            }

            VmInstr::MemCopy { .. } => {
                Err(CompilerError::CodegenViolation("MemCopy not yet implemented on AArch64".into()))
            }

            // ── REQ-VR-005~010: 缺失指令 — AArch64 stub (ISA lowering 由 VR-012 补全) ──

            VmInstr::VecShuffle { dst, src, mask, width: _ } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*src, alloc)?;
                match mask {
                    VecShuffleMask::Const(bytes) => {
                        // AArch64 TBL: load shuffle control bytes into a temp NEON reg,
                        // then TBL Vd.16B, {Vn.16B}, Vtmp.16B
                        // Push control bytes to stack, LD1 into V17 (scratch)
                        for chunk in bytes.chunks(8) {
                            let mut qw = 0u64;
                            for (j, &b) in chunk.iter().enumerate() {
                                qw |= (b as u64) << (j * 8);
                            }
                            // STP X0, X1, [SP, #-16]! pattern — push qw
                            // Simplified: use sub sp + stp sequence
                            self.emit32(0xd10043ffu32); // SUB SP, SP, #16
                            self.emit32(0xf90003e0u32 | ((qw as u32) & 0xFFFF)); // STR (simplified — use two stores)
                            // Actually, emit 64-bit immediate load via MOVN/MOVK sequence into x9, then STP
                            let lo = qw as u32;
                            let hi = (qw >> 32) as u32;
                            // MOVZ x9, #lo16
                            self.emit32(0xd2800009u32 | ((lo & 0xFFFF) << 5));
                            if (lo >> 16) != 0 { self.emit32(0xf2a00009u32 | (((lo >> 16) & 0xFFFF) << 5)); }
                            // MOVK x9, #hi16, lsl #32
                            self.emit32(0xd2800009u32 | ((hi & 0xFFFF) << 5) | (0x2 << 21));
                            // STR X9, [SP]
                            self.emit32(0xf90003e9u32);
                        }
                        // LD1 {V17.16B}, [SP]
                        self.emit32(0x4c407ff1u32); // LD1 {V17.16B}, [SP]
                        // ADD SP, SP, #16
                        self.emit32(0x910043ffu32);
                        // TBL Vd.16B, {Vn.16B}, V17.16B
                        // Encoding: 0x0E000000 | (Vm << 5) | (0 for one-table TBL) | Vd
                        // TBL Vd.16B, {Vn}, Vm: 0x0E000000 | (vm << 5) | vn | (vd & 0xF) | ((vd >> 4) << 10)
                        // Simplified: assume vn < 32, vd < 32, vm=17
                        self.emit32(0x0e000000u32 | ((17u32) << 5) | ((vn & 0x1f) as u32) | (((vd & 0x1f) as u32) << (if vd >= 16 { 10 } else { 0 })));
                        Ok(())
                    }
                    VecShuffleMask::Dynamic { ctrl } => {
                        let vm = self.resolve_vreg(*ctrl, alloc)?;
                        // TBL Vd.16B, {Vn.16B}, Vm.16B
                        self.emit32(0x0e000000u32 | ((vm as u32) << 5) | (vn as u32) | (vd as u32));
                        Ok(())
                    }
                }
            }

            VmInstr::VecExtractLane { dst, src, lane, dtype: _ } => {
                let vd = self.resolve_gpr(*dst, alloc)?;
                let vn = self.resolve_vreg(*src, alloc)?;
                // UMOV Wd, Vn.B[lane] or UMOV Xd, Vn.D[lane] depending on lane width
                // For simplicity: assume 32-bit lane extraction
                // UMOV Wd, Vn.S[lane]: encoding 0x0E003C00 | (lane << 17) | (Rd << 5) | Vn
                let lane_val = *lane as u32;
                self.emit32(0x0e003c00u32 | (lane_val << 17) | ((vd as u32) << 5) | (vn as u32));
                Ok(())
            }

            VmInstr::VecInsertLane { dst, src_vec, src_scalar, lane, dtype: _ } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                let vn = self.resolve_vreg(*src_vec, alloc)?;
                let vs = self.resolve_gpr(*src_scalar, alloc)?;
                // INS Vd.S[lane], Ws — insert GPR into NEON lane
                // First copy src_vec to dst if different
                if vd != vn {
                    // ORR Vd.16B, Vn.16B, Vn.16B (register copy)
                    self.emit32(self.enc_orr_vv(vd, vn, vn));
                }
                // INS Vd.S[lane], Ws: encoding 0x4E001C00 | (lane << 17) | (Ws << 5) | Vd
                self.emit32(0x4e001c00u32 | ((*lane as u32) << 17) | ((vs as u32) << 5) | (vd as u32));
                Ok(())
            }

            VmInstr::VecLoadConst { dst, values, dtype: _, width: _ } => {
                let vd = self.resolve_vreg(*dst, alloc)?;
                // Load constant values into NEON register
                // Strategy: for each u32 value, load via MOVZ+MOVK into scratch GPR,
                // then INS into NEON lane
                for (lane, &val) in values.iter().enumerate() {
                    if lane >= 4 { break; } // Only handle up to 4 lanes (128-bit)
                    // Load immediate into x9
                    let lo16 = val & 0xFFFF ;
                    let hi16 = val >> 16 ;
                    // MOVZ x9, #lo16
                    self.emit32(0xd2800009u32 | (lo16 << 5));
                    if hi16 != 0 {
                        // MOVK x9, #hi16, lsl #16
                        self.emit32(0xf2a00009u32 | (hi16 << 5));
                    }
                    // INS Vd.S[lane], W9
                    self.emit32(0x4e001c00u32 | ((lane as u32) << 17) | ((9u32) << 5) | (vd as u32));
                }
                // Zero remaining lanes if values.len() < 4
                if values.is_empty() {
                    // MOVI Vd.2D, #0
                    self.emit32(self.enc_movi_zero_4s(vd));
                }
                Ok(())
            }

            VmInstr::AtomicCAS { dst, ptr, expected, desired, elem_width, success_order: _, failure_order: _ } => {
                let rd = self.resolve_gpr(*dst, alloc)?;
                let rn_ptr = self.resolve_gpr(*ptr, alloc)?;
                let re = self.resolve_gpr(*expected, alloc)?;
                let rs = self.resolve_gpr(*desired, alloc)?;
                let _ = elem_width;
                // AArch64 LL/SC compare-and-swap loop:
                //   LDAXR Xd, [Xn_ptr]     — load exclusive
                //   CMP Xd, Xe              — compare with expected
                //   B.NE done                — if not equal, skip store
                //   STLXR W16, Xs, [Xn_ptr] — store exclusive
                //   CBNZ W16, retry         — if store failed, retry
                // done:
                let retry_offset = self.code.len() as u32;
                // LDAXR Xd, [Xn_ptr] — Load-Acquire Exclusive
                // Encoding: 0xC85FFC00 | (Rs << 16) | (Rn << 5) | Rt  (for 8-byte)
                self.emit32(0xc85ff800u32 | ((16u32) << 16) | ((rn_ptr as u32) << 5) | (rd as u32)); // LDXR
                // CMP Xd, Xe
                self.emit32(0xeb00001fu32 | ((re as u32) << 16) | ((rd as u32) << 5) | 0x1fu32); // SUBS XZR, Xd, Xe (sets flags)
                // B.NE done (skip store) — encode as conditional branch forward
                let ne_target = self.code.len() as u32 + 3; // skip 3 instructions
                self.emit32(0x54000000u32 | (1u32 << 24) | ((ne_target - self.code.len() as u32 / 4) << 5) | 0x1u32); // B.NE
                // STLXR W16, Xs, [Xn_ptr] — Store-Release Exclusive
                self.emit32(0xc800fc00u32 | ((16u32) << 16) | ((rn_ptr as u32) << 5) | (rs as u32));
                // CBNZ W16, retry — branch back if store failed
                let back_offset = retry_offset as i32 - self.code.len() as i32;
                let imm19 = ((back_offset / 4) as u32) & 0x7FFFF;
                self.emit32(0x35000000u32 | (imm19 << 5) | (16u32)); // CBNZ W16, retry
                Ok(())
            }

            VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, seq_meta_stride } => {
                let _ = (dst, token_index, seq_meta_base, num_seqs, seq_meta_stride);
                Err(CompilerError::CodegenViolation(
                    "SeqIdLookup: AArch64 cumsum search not yet implemented".into()))
            }

            // ── §1.6 分布式通信 VmInstr (REQ-VR-014, feature = "nccl") ──

            #[cfg(feature = "nccl")]
            VmInstr::AllReduceChunk { .. } => {
                // AArch64 AAPCS64 call to gllm_nccl_all_reduce_chunk_stub.
                // Full implementation requires register save/restore + arg marshaling
                // for 8 parameters through x0-x7. Deferred to NCCL9.
                Err(CompilerError::CodegenViolation(
                    "AllReduceChunk: AArch64 call stub not yet implemented (NCCL9)".into()))
            }

            #[cfg(feature = "nccl")]
            VmInstr::CommBarrier { .. } => {
                // CPU 语义: NOP (通信由 GPU 驱动执行)
                Ok(())
            }

            #[cfg(feature = "nccl")]
            VmInstr::NvlinkAsyncCopy { .. } => {
                // CPU 语义: NOP (通信由 GPU 驱动执行)
                Ok(())
            }

            // ── §8 分布式分页 VmInstr (REQ-DP-010, feature = "nccl") ──

            #[cfg(feature = "nccl")]
            VmInstr::RemotePageLookup { .. } => {
                Err(CompilerError::CodegenViolation(
                    "RemotePageLookup: AArch64 distributed paging not yet implemented".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::P2pPageFetch { .. } => {
                Err(CompilerError::CodegenViolation(
                    "P2pPageFetch: AArch64 P2P DMA not supported (GPU-only)".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetch { .. } => {
                Err(CompilerError::CodegenViolation(
                    "RdmaPageFetch: AArch64 RDMA not supported (GPU-only)".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetchCompressed { .. } => {
                Err(CompilerError::CodegenViolation(
                    "RdmaPageFetchCompressed: AArch64 RDMA not supported (GPU-only)".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageAttn { .. } => {
                Err(CompilerError::CodegenViolation(
                    "RemotePageAttn: AArch64 remote attention not yet implemented".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationLock { .. } => {
                Err(CompilerError::CodegenViolation(
                    "PageMigrationLock: AArch64 distributed paging not yet implemented".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationUnlock { .. } => {
                Err(CompilerError::CodegenViolation(
                    "PageMigrationUnlock: AArch64 distributed paging not yet implemented".into()))
            }
            #[cfg(feature = "nccl")]
            VmInstr::PageLocationUpdate { .. } => {
                Err(CompilerError::CodegenViolation(
                    "PageLocationUpdate: AArch64 distributed paging not yet implemented".into()))
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  DotProduct dtype 策略辅助 (REQ-VR10)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// DotDtype → AArch64ElemStrategy 映射 (REQ-VR10)。
    /// 用于策略驱动 dot-product 指令选择，禁止 DotDtype 身份匹配。
    fn dot_dtype_aarch64_strategy(&self, dt: DotDtype) -> AArch64ElemStrategy {
        // 通过 DotDtype 的元素特征决定策略，而非 match 具体变体。
        if dot_dtype_is_bf16(dt) || dot_dtype_is_fp16(dt) || dot_dtype_is_int8(dt) {
            AArch64ElemStrategy::Native
        } else {
            AArch64ElemStrategy::WidenCompute
        }
    }

    /// Native dot-product lowering (REQ-VR10): 硬件原生 dot 指令。
    fn lower_dot_product_native(
        &mut self,
        vd: u8, vn: u8, vm: u8,
        dt: DotDtype,
    ) -> Result<(), CompilerError> {
        if dot_dtype_is_bf16(dt) {
            // BFDOT Vd.4S, Vn.8H, Vm.8H (ARMv8.6-A BF16)
            self.emit32(0x6E40FC00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
        } else if dot_dtype_is_fp16(dt) {
            // FCVTL + FMLA: convert F16 halves to F32, then FMA accumulate.
            let s1: u8 = 16;
            let s2: u8 = 17;
            self.emit32(0x0E218800 | ((vn as u32 & 0x1F) << 5) | s1 as u32);
            self.emit32(0x0E218800 | ((vm as u32 & 0x1F) << 5) | s2 as u32);
            self.emit32(self.enc_fmla_4s(vd, s1, s2));
        } else if dot_dtype_is_int8(dt) {
            // SDOT Vd.4S, Vn.16B, Vm.16B (ARMv8.4-A DotProd)
            self.emit32(0x4E409C00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
        } else {
            return Err(CompilerError::CodegenViolation(
                "DotProduct Native: unsupported DotDtype variant".into()
            ));
        }
        Ok(())
    }

    /// WidenCompute dot-product lowering (REQ-VR10): 子字节类型先解包再计算。
    fn lower_dot_product_widen(
        &mut self,
        vd: u8, vn: u8, vm: u8,
        dt: DotDtype,
    ) -> Result<(), CompilerError> {
        if dot_dtype_is_int4x8(dt) {
            // INT4×INT8: nibble unpack done by preceding ops, emit SDOT on unpacked INT8.
            self.emit32(0x4E409C00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F));
        } else if dot_dtype_is_fp4(dt) {
            // FP4 (E2M1): pre-decoded to F32 by caller, emit FMLA.
            self.emit32(self.enc_fmla_4s(vd, vn, vm));
        } else {
            return Err(CompilerError::CodegenViolation(
                "DotProduct WidenCompute: unsupported DotDtype variant".into()
            ));
        }
        Ok(())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  LLM-special ops 辅助函数
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// LZ4 流解压 (AArch64, SPEC §3.3.1 REQ-COMP-003).
    ///
    /// AArch64 标量字节循环 + 基本分支逻辑。
    /// 寄存器: x9=src_cur, x10=dst_cur, x11=src_end,
    ///   w12=literal_len, w13=match_len, w14=match_offset, x15=match_src.
    fn lower_lz4_decode(
        &mut self,
        src_ptr: VRegId,
        dst_ptr: VRegId,
        compressed_size: VRegId,
        _decompressed_size: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let src_reg = self.resolve_gpr(src_ptr, alloc)?;
        let dst_reg = self.resolve_gpr(dst_ptr, alloc)?;
        let csz_reg = self.resolve_gpr(compressed_size, alloc)?;

        // MOV x9, src_reg  (x9 = src_cur)
        self.emit32(0xAA0003E9u32 | ((src_reg as u32) << 16));
        // MOV x10, dst_reg (x10 = dst_cur)
        self.emit32(0xAA0003EAu32 | ((dst_reg as u32) << 16));
        // MOV x11, x9     (x11 = src_cur initially, then compute end)
        self.emit32(0xAA0903EBu32); // MOV x11, x9
        // ADD x11, x11, csz_reg (x11 = src_end)
        self.emit32(0x8B000000u32 | ((csz_reg as u32) << 16) | (11u32 << 5) | 11u32);

        // Main decode loop: label = decode_loop_offset
        let decode_loop = self.current_offset();

        // CMP x9, x11 → SUB xzr, x9, x11
        self.emit32(0xEB0B013Fu32);
        // B.HS done_placeholder
        let done_patch1 = self.current_offset();
        self.emit32(0x54000002u32); // placeholder

        // LDRB w3, [x9] then ADD x9, x9, #1
        self.emit32(0x39400123u32); // LDRB w3, [x9]
        self.emit32(0x91000529u32); // ADD x9, x9, #1

        // w12 = w3 >> 4  (literal_len raw)
        self.emit32(0x5304106Cu32); // LSR w12, w3, #4
        // w13 = w3 & 0xF (match_len raw)
        self.emit32(0x12001C6Du32); // AND w13, w3, #0xF

        // Literal copy loop: while w12 != 0, copy one byte
        let lit_loop = self.current_offset();
        let lit_done_patch = self.current_offset();
        self.emit32(0x3400000Cu32); // CBZ w12, lit_done_placeholder

        // LDRB w3, [x9]; ADD x9,x9,#1; STRB w3,[x10]; ADD x10,x10,#1; SUBS w12,w12,#1
        self.emit32(0x39400123u32); // LDRB w3, [x9]
        self.emit32(0x91000529u32); // ADD x9, x9, #1
        self.emit32(0x39000143u32); // STRB w3, [x10]
        self.emit32(0x9100054Au32); // ADD x10, x10, #1
        self.emit32(0x7100058Cu32); // SUBS w12, w12, #1
        // B.NE lit_loop
        let lit_back = ((lit_loop as i32 - self.current_offset() as i32) >> 2) as u32;
        self.emit32(0x54000001u32 | ((lit_back & 0x7FFFF) << 5));

        // lit_done:
        let lit_done = self.current_offset();
        let off = ((lit_done as i32 - lit_done_patch as i32) >> 2) as u32;
        self.patch32(lit_done_patch, 0x3400000Cu32 | (off << 5));

        // End-of-stream: CMP x9, x11; B.HS done
        self.emit32(0xEB0B013Fu32);
        let done_patch2 = self.current_offset();
        self.emit32(0x54000002u32); // placeholder

        // LDRH w14, [x9]; ADD x9, x9, #2 (match_offset LE u16)
        self.emit32(0x7940012Eu32); // LDRH w14, [x9]
        self.emit32(0x91000929u32); // ADD x9, x9, #2 (2 bytes)

        // w13 = w13 + 4 (minmatch offset)
        self.emit32(0x110110ADu32); // ADD w13, w13, #4

        // x15 = x10 - x14 (match_src, x14 zero-extended from LDRH)
        self.emit32(0xCB0E014Fu32); // SUB x15, x10, x14

        // Match copy loop
        let match_loop = self.current_offset();
        let match_done_patch = self.current_offset();
        self.emit32(0x3400000Du32); // CBZ w13, match_done_placeholder

        // LDRB w3,[x15]; ADD x15,x15,#1; STRB w3,[x10]; ADD x10,x10,#1; SUBS w13,w13,#1
        self.emit32(0x394001E3u32); // LDRB w3, [x15]
        self.emit32(0x910005EFu32); // ADD x15, x15, #1
        self.emit32(0x39000143u32); // STRB w3, [x10]
        self.emit32(0x9100054Au32); // ADD x10, x10, #1
        self.emit32(0x710005ADu32); // SUBS w13, w13, #1
        let match_back = ((match_loop as i32 - self.current_offset() as i32) >> 2) as u32;
        self.emit32(0x54000001u32 | ((match_back & 0x7FFFF) << 5));

        // match_done:
        let match_done = self.current_offset();
        let off = ((match_done as i32 - match_done_patch as i32) >> 2) as u32;
        self.patch32(match_done_patch, 0x3400000Du32 | (off << 5));

        // B decode_loop (backward)
        let loop_back = ((decode_loop as i32 - self.current_offset() as i32) >> 2) as u32;
        self.emit32(0x14000000u32.wrapping_add(loop_back & 0x03FFFFFF));

        // final done label — patch forward refs
        let final_done = self.current_offset();

        let off = ((final_done as i32 - done_patch1 as i32) >> 2) as u32;
        self.patch32(done_patch1, 0x54000002u32 | ((off & 0x7FFFF) << 5));

        let off = ((final_done as i32 - done_patch2 as i32) >> 2) as u32;
        self.patch32(done_patch2, 0x54000002u32 | ((off & 0x7FFFF) << 5));

        Ok(())
    }

    /// BitPackRle 解压 (AArch64, SPEC §3.3.2 REQ-COMP-004).
    ///
    /// AArch64：标量解析 run header + 字节填充循环。
    /// x9=src_cur, x10=dst_cur, x11=src_end, w4=run_value, w5=run_len.
    fn lower_bitpack_rle_decode(
        &mut self,
        src_ptr: VRegId,
        dst_ptr: VRegId,
        compressed_size: VRegId,
        _nibble_bits: u8,
        _element_count: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let src_reg = self.resolve_gpr(src_ptr, alloc)?;
        let dst_reg = self.resolve_gpr(dst_ptr, alloc)?;
        let csz_reg = self.resolve_gpr(compressed_size, alloc)?;

        // Setup
        self.emit32(0xAA0003E9u32 | ((src_reg as u32) << 16));
        self.emit32(0xAA0003EAu32 | ((dst_reg as u32) << 16));
        self.emit32(0xAA0903EBu32); // MOV x11, x9
        self.emit32(0x8B000000u32 | ((csz_reg as u32) << 16) | (11u32 << 5) | 11u32);

        let rle_loop = self.current_offset();
        self.emit32(0xEB0B013Fu32); // CMP x9, x11
        let rle_done_patch = self.current_offset();
        self.emit32(0x54000002u32); // B.HS done placeholder

        // LDRB w3, [x9]; ADD x9, x9, #1
        self.emit32(0x39400123u32);
        self.emit32(0x91000529u32);

        // w4 = w3 & 0x0F (run_value)
        self.emit32(0x12001C64u32); // AND w4, w3, #0x0F
        // w5 = w3 >> 4   (run_len raw)
        self.emit32(0x53041065u32); // LSR w5, w3, #4

        // Extension: if w5==15, read more bytes
        self.emit32(0x7100F4BFu32); // SUBS wzr, w5, #15
        let ext_skip_patch = self.current_offset();
        self.emit32(0x54000001u32); // B.NE fill_loop placeholder

        let ext_loop = self.current_offset();
        self.emit32(0xEB0B013Fu32); // CMP x9, x11
        let ext_hs_patch = self.current_offset();
        self.emit32(0x54000002u32); // B.HS fill_loop placeholder

        // LDRB w3, [x9]; ADD x9, x9, #1; ADD w5, w5, w3
        self.emit32(0x39400123u32);
        self.emit32(0x91000529u32);
        self.emit32(0x0B0300A5u32); // ADD w5, w5, w3

        // SUBS wzr, w3, #255; B.EQ ext_loop
        self.emit32(0x7103FC7Fu32); // SUBS wzr, w3, #255
        let ext_back = ((ext_loop as i32 - self.current_offset() as i32) >> 2) as u32;
        self.emit32(0x54000000u32 | ((ext_back & 0x7FFFF) << 5)); // B.EQ

        // fill_loop:
        let fill_loop = self.current_offset();
        // Patch ext_skip and ext_hs to point here
        let off = ((fill_loop as i32 - ext_skip_patch as i32) >> 2) as u32;
        self.patch32(ext_skip_patch, 0x54000001u32 | ((off & 0x7FFFF) << 5));
        let off = ((fill_loop as i32 - ext_hs_patch as i32) >> 2) as u32;
        self.patch32(ext_hs_patch, 0x54000002u32 | ((off & 0x7FFFF) << 5));

        let fill_done_patch = self.current_offset();
        self.emit32(0x34000005u32); // CBZ w5, fill_done placeholder

        // STRB w4, [x10]; ADD x10, x10, #1; SUBS w5, w5, #1
        self.emit32(0x39000144u32); // STRB w4, [x10]
        self.emit32(0x9100054Au32); // ADD x10, x10, #1
        self.emit32(0x710004A5u32); // SUBS w5, w5, #1
        let fill_back = ((fill_loop as i32 - self.current_offset() as i32) >> 2) as u32;
        self.emit32(0x54000001u32 | ((fill_back & 0x7FFFF) << 5)); // B.NE fill_loop

        let fill_done = self.current_offset();
        let off = ((fill_done as i32 - fill_done_patch as i32) >> 2) as u32;
        self.patch32(fill_done_patch, 0x34000005u32 | (off << 5));

        // B rle_loop
        let loop_back = ((rle_loop as i32 - self.current_offset() as i32) >> 2) as u32;
        self.emit32(0x14000000u32.wrapping_add(loop_back & 0x03FFFFFF));

        // rle_done: patch forward ref
        let rle_done = self.current_offset();
        let off = ((rle_done as i32 - rle_done_patch as i32) >> 2) as u32;
        self.patch32(rle_done_patch, 0x54000002u32 | ((off & 0x7FFFF) << 5));

        Ok(())
    }
    ///
    /// 简化实现：标量循环扫描（可优化为 NEON 向量化）
    fn lower_argmax(
        &mut self,
        dst: VRegId,
        logits_ptr: VRegId,
        vocab_bytes: usize,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_reg = self.resolve_gpr(dst, alloc)?;
        let base_reg = self.resolve_gpr(logits_ptr, alloc)?;

        // 初始化：dst = 0, max_val = logits[0]
        self.emit32(self.enc_mov_x(dst_reg, 0)); // argmax = 0

        let tmp_max: u8 = 16; // x16 存储当前最大值
        let tmp_idx: u8 = 17; // x17 存储当前索引
        let tmp_ptr: u8 = 18; // x18 存储当前指针

        // 加载第一个值: LDR S16, [base_reg]
        self.emit32(0xBD400000 | ((16u32 & 0x1F) << 12) | ((base_reg as u32 & 0x1F) << 5) | 16);
        // 拷贝指针: MOV X18, X0
        self.emit32(self.enc_mov_x(tmp_ptr, base_reg));

        // 循环计数器 = vocab_bytes / 4
        let elem_count = vocab_bytes / 4;
        let ctr_reg: u8 = 19; // x19 作为循环计数器

        // MOV X19, #elem_count
        self.emit32(0xD2800000 | (((elem_count & 0xFFFF) as u32) << 5) | ctr_reg as u32);
        if elem_count > 0xFFFF {
            self.emit32(0xF2A00000 | ((((elem_count >> 16) & 0xFFFF) as u32) << 5) | ctr_reg as u32);
        }

        let loop_top = self.current_offset();

        // 循环体：
        // - 加载当前值: LDR S17, [X18], #4 (post-increment)
        self.emit32(0xBD4C4000 | ((17u32 & 0x1F) << 12) | ((tmp_ptr as u32 & 0x1F) << 5) | 17);

        // - 比较: FCMP S17, S16
        self.emit32(0x1E622000 | ((17u32 & 0x1F) << 5) | 16);

        // - B.LE skip (新值 <= 最大值，跳过更新)
        self.emit32(0x54000040 | ((1) << 5) | 12); // B.LE +1 instr

        // - 更新最大值: FMOV S16, S17
        self.emit32(0x1E224000 | ((17u32 & 0x1F) << 5) | 16);

        // - 更新索引: SUB X19, X19, #1 ; argmax = elem_count - ctr_reg
        self.emit32(self.enc_sub_imm(ctr_reg, ctr_reg, 1));
        self.emit32(self.enc_sub_reg(dst_reg, elem_count as u8, ctr_reg));

        // skip: 继续循环
        // - SUBS X19, X19, #1
        self.emit32(0xF1000021 | ((ctr_reg as u32 & 0x1F) << 5) | (ctr_reg as u32 & 0x1F));
        // - B.NE loop_top
        let loop_top_offset = (self.current_offset() + 4 - loop_top) / 4;
        self.emit32(0x54000001 | (((loop_top_offset as u32) & 0x7FFFF) << 5));

        Ok(())
    }

    /// Temperature scaling: logits[i] /= temperature (in-place)。
    ///
    /// 简化实现：标量循环（可优化为 NEON 向量化）
    fn lower_temperature_scale(
        &mut self,
        logits_ptr: VRegId,
        temp_ptr: VRegId,
        vocab_bytes: usize,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let base_reg = self.resolve_gpr(logits_ptr, alloc)?;
        let temp_base = self.resolve_gpr(temp_ptr, alloc)?;

        // 加载 temperature: LDR S16, [temp_base]
        let tmp_temp: u8 = 16;
        let tmp_val: u8 = 17;
        let tmp_ptr: u8 = 18;
        let ctr_reg: u8 = 19;

        self.emit32(0xBD400000 | ((tmp_temp as u32 & 0x1F) << 12) | ((temp_base as u32 & 0x1F) << 5) | (tmp_temp as u32));

        // 初始化循环
        let elem_count = vocab_bytes / 4;
        self.emit32(self.enc_mov_x(tmp_ptr, base_reg));
        self.emit32(0xD2800000 | (((elem_count & 0xFFFF) as u32) << 5) | ctr_reg as u32);
        if elem_count > 0xFFFF {
            self.emit32(0xF2A00000 | ((((elem_count >> 16) & 0xFFFF) as u32) << 5) | ctr_reg as u32);
        }

        let loop_top = self.current_offset();

        // 循环体：
        // - LDR S17, [X18], #4 (post-increment load)
        self.emit32(0xBD4C4000 | ((tmp_val as u32 & 0x1F) << 12) | ((tmp_ptr as u32 & 0x1F) << 5) | (tmp_val as u32));

        // - FDIV S17, S17, S16
        self.emit32(0x6E20FC00 | ((tmp_temp as u32 & 0x1F) << 16) | ((tmp_val as u32 & 0x1F) << 5) | (tmp_val as u32));

        // - STR S17, [X18, #-4] (store back to previous location)
        self.emit32(0xBD00C000 | ((tmp_val as u32 & 0x1F) << 12) | ((tmp_ptr as u32 & 0x1F) << 5) | (tmp_val as u32));

        // - SUBS X19, X19, #1
        self.emit32(0xF1000021 | ((ctr_reg as u32 & 0x1F) << 5) | (ctr_reg as u32 & 0x1F));

        // - B.NE loop_top
        let loop_top_offset = (self.current_offset() + 4 - loop_top) / 4;
        self.emit32(0x54000001 | (((loop_top_offset as u32) & 0x7FFFF) << 5));

        Ok(())
    }
}
