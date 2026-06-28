impl GpuLower {
    pub fn lower_instr(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match instr {
            // ═══ 内存操作 ═══

            VmInstr::VecLoad { dst, base, offset, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // ARCH-GPU-PTX-ADDR: PTX `[reg+reg]` 语法非法，必须先算出 64-bit 地址。
                        // offset 是 32-bit 整数，base 是 64-bit 指针。流程：
                        //   cvt.u64.u32 addr_scratch, off;  addr_scratch += b;  ld.global.f32 d, [addr_scratch]
                        let addr = self.scratch_gpr_names[0]; // %rs0 — 复用为 b64 临时（PTX 允许）
                        let addr64 = self.scratch_gpr_names[1]; // %rs1 — 实际 64-bit 存放
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("ld.global.f32 {d}, [%rd_addr];"));
                        let _ = (addr, addr64);
                    }
                    GpuDialect::Hip { .. } => self.emit_line(&format!("{d} = *(({b}) + ({off})/4);")),
                    GpuDialect::Metal { .. } => self.emit_line(&format!("{d} = {b}[({off})/4];")),
                }
                Ok(())
            }

            VmInstr::VecNarrow { dst, src, .. } => {
                // GPU 窄化：寄存器到寄存器拷贝（same-dtype no-op，cross-dtype 待实现）
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                self.emit_line(&format!("mov.v4.f32 {d}, {s};  // VecNarrow (no-op for same dtype)"));
                Ok(())
            }

            VmInstr::VecWiden { dst, src, dst_dtype, src_dtype, .. } => {
                // REQ-DTYPE-003: GPU 向量宽化。BF16→F32 使用 cvt.rn.f32.bf16。
                if dst_dtype.elem_bytes() > src_dtype.elem_bytes() {
                    let d = self.reg_name_with_kind(*dst, alloc);
                    let s = self.reg_name_with_kind(*src, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("cvt.rn.f32.bf16 {d}, {s};  // VecWiden BF16→F32"));
                        }
                        GpuDialect::Hip { .. } => {
                            self.emit_line(&format!("v_cvt_f32_bf16 {d}, {s};  // VecWiden BF16→F32"));
                        }
                        GpuDialect::Metal { .. } => {
                            self.emit_line(&format!("{d} = convert_float({s});  // VecWiden BF16→F32"));
                        }
                    }
                } else {
                    let d = self.reg_name_with_kind(*dst, alloc);
                    let s = self.reg_name_with_kind(*src, alloc);
                    self.emit_line(&format!("mov.v4.f32 {d}, {s};  // VecWiden (no-op for same dtype)"));
                }
                Ok(())
            }

            VmInstr::Mov { dst, src, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line(&format!("mov.f32 {d}, {s};")),
                    GpuDialect::Hip { .. } => self.emit_line(&format!("s_mov_b32 {d}, {s};")),
                    GpuDialect::Metal { .. } => self.emit_line(&format!("{d} = {s};")),
                }
                Ok(())
            }

            VmInstr::VecStore { base, src, offset, .. } => {
                let s = self.reg_name_with_kind(*src, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {s};"));
                    }
                    GpuDialect::Hip { .. } => self.emit_line(&format!("*(({b}) + ({off})/4) = {s};")),
                    GpuDialect::Metal { .. } => self.emit_line(&format!("{b}[({off})/4] = {s};")),
                }
                Ok(())
            }

            VmInstr::LoadPtr { dst, src } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                match src {
                    PtrExpr::AbiArg(idx) => {
                        // ARCH-GPU-PARAM-NAMES: 使用 prologue 声明的参数名，不用 param{idx}
                        let param = self.param_name(*idx)?;
                        match self.dialect {
                            GpuDialect::Ptx { .. } => self.emit_line(&format!("ld.param.u64 {d}, [{param}];")),
                            _ => self.emit_line(&format!("{d} = {param};")),
                        }
                    }
                    PtrExpr::VRegPlusConst(base, off) => {
                        let b = self.reg_name_with_kind(*base, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => self.emit_line(&format!("add.u64 {d}, {b}, {off};")),
                            _ => self.emit_line(&format!("{d} = {b} + {off};")),
                        }
                    }
                    PtrExpr::VRegPlusVReg(base, offset) => {
                        let b = self.reg_name_with_kind(*base, alloc);
                        let o = self.reg_name_with_kind(*offset, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => self.emit_line(&format!("add.u64 {d}, {b}, {o};")),
                            _ => self.emit_line(&format!("{d} = {b} + {o};")),
                        }
                    }
                    PtrExpr::VRegPlusOff(base, off_expr) => {
                        let b = self.reg_name_with_kind(*base, alloc);
                        let off_str = self.offset_to_string(off_expr, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => self.emit_line(&format!("add.u64 {d}, {b}, {off_str};")),
                            _ => self.emit_line(&format!("{d} = {b} + ({off_str});")),
                        }
                    }
                    PtrExpr::SharedMem => {
                        // ARCH-GPU-SHARED-SCRATCH: scratchpad 由 prologue 的
                        // `.shared`/`__shared__`/`threadgroup` 符号 `smem` 提供,
                        // LoadPtr 取该符号地址 (PTX: generic 地址; HIP/Metal: 指针转换)。
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                // PTX: shared 符号地址 = generic 地址空间 ptr
                                // `mov.u64 dst, smem;` 会取 generic alias,
                                // ld/st 需再 `cvta.to.shared` 或直接使用 state-space qualified
                                // ld.shared。当前 kernel 所有 ld 都是 ld.global,所以这里
                                // 用 cvta 把 shared 符号转 generic。
                                self.emit_line(&format!("cvta.shared.u64 {d}, smem;"));
                            }
                            GpuDialect::Hip { .. } => {
                                self.emit_line(&format!("{d} = (float*)smem;"));
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line(&format!("{d} = (device float*)smem;"));
                            }
                        }
                    }
                    PtrExpr::StackArg(_) | PtrExpr::NamedArg(_) => {
                        // GPU ABI 没有栈参数概念，所有参数通过 .param 传入
                        return Err(CompilerError::CodegenViolation(
                            format!("GPU LoadPtr: {:?} 不适用于 GPU ABI（GPU 用 AbiArg/.param）", src)
                        ));
                    }
                    PtrExpr::AbsAddr(_) => {
                        // ARCH-SG-QTAP: GPU 的 Q-tap 路径必须走设备可见指针 (通常是
                        // cudaHostAlloc 映射或 device pointer), 其地址由 host 在
                        // 运行时写入 .param 而非编译时嵌入。
                        //
                        // 本次 session 只实现 CPU (x86_64 主, AArch64 次) codegen;
                        // GPU 的 Q-tap 支持作为独立后续 session (需要 SPEC 定义
                        // 零拷贝 host↔device visibility 协议与 .param 绑定流程)。
                        return Err(CompilerError::CodegenViolation(
                            "GPU LoadPtr: PtrExpr::AbsAddr 尚未在 GPU 后端实现 \
                             (ARCH-SG-QTAP GPU 路径留作独立 session)".into()
                        ));
                    }
                }
                Ok(())
            }

            VmInstr::Broadcast { dst, src, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                match src {
                    ScalarExpr::Const(val) => match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            // ARCH-GPU-PTX-LITERAL: PTX f32 立即数必须是 IEEE 754 hex: 0f<8位十六进制>
                            self.emit_line(&format!("mov.f32 {d}, 0f{:08X};", val.to_bits()));
                        }
                        _ => self.emit_line(&format!("{d} = {val}f;")),
                    },
                    ScalarExpr::MemLoad(base, offset) => {
                        let b = self.reg_name_with_kind(*base, alloc);
                        let off = self.offset_to_string(offset, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                                self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                                self.emit_line(&format!("ld.global.f32 {d}, [%rd_addr];"));
                            }
                            GpuDialect::Hip { .. } => self.emit_line(&format!("{d} = *(({b}) + ({off})/4);")),
                            GpuDialect::Metal { .. } => self.emit_line(&format!("{d} = {b}[({off})/4];")),
                        }
                    }
                    ScalarExpr::ExtractLane0(src_vreg) => {
                        // GPU thread-per-element 模型：ExtractLane0 等价于 mov
                        let s = self.reg_name_with_kind(*src_vreg, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => self.emit_line(&format!("mov.f32 {d}, {s};")),
                            _ => self.emit_line(&format!("{d} = {s};")),
                        }
                    }
                    ScalarExpr::VReg(src_vreg) => {
                        // GPU thread-per-element: VReg is already scalar, just assign
                        let s = self.reg_name_with_kind(*src_vreg, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => self.emit_line(&format!("mov.f32 {d}, {s};")),
                            _ => self.emit_line(&format!("{d} = {s};")),
                        }
                    }
                }
                Ok(())
            }

            // ═══ 算术操作 ═══

            VmInstr::VecBinOp { dst, a, b, op, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let vb = self.reg_name_with_kind(*b, alloc);
                // ARCH-GPU-PTX-VECOP: 位运算必须用 b32 类型，浮点用 f32
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let line = match op {
                            VecOp::Add => format!("add.f32 {d}, {va}, {vb};"),
                            VecOp::Sub => format!("sub.f32 {d}, {va}, {vb};"),
                            VecOp::Mul => format!("mul.f32 {d}, {va}, {vb};"),
                            VecOp::Div => format!("div.approx.f32 {d}, {va}, {vb};"),
                            VecOp::Max => format!("max.f32 {d}, {va}, {vb};"),
                            VecOp::Min => format!("min.f32 {d}, {va}, {vb};"),
                            VecOp::And => format!("and.b32 {d}, {va}, {vb};"),
                            VecOp::Or  => format!("or.b32 {d}, {va}, {vb};"),
                            VecOp::Xor => format!("xor.b32 {d}, {va}, {vb};"),
                            VecOp::AndNot => {
                                self.emit_line(&format!("not.b32 {d}, {vb};"));
                                format!("and.b32 {d}, {va}, {d};")
                            }
                            VecOp::Not => format!("not.b32 {d}, {va};"),
                            VecOp::Shl => format!("shl.b32 {d}, {va}, {vb};"),
                            VecOp::Shr => format!("shr.b32 {d}, {va}, {vb};"),
                        };
                        self.emit_line(&line);
                    }
                    _ => {
                        let op_str = match op {
                            VecOp::Add => "+", VecOp::Sub => "-",
                            VecOp::Mul => "*", VecOp::Div => "/",
                            VecOp::Max => "", VecOp::Min => "",
                            VecOp::And => "&", VecOp::Or => "|",
                            VecOp::Xor => "^", VecOp::AndNot => "&~",
                            VecOp::Shl => "<<", VecOp::Shr => ">>",
                            VecOp::Not => "~",
                        };
                        if matches!(op, VecOp::Max) {
                            self.emit_line(&format!("{d} = max({va}, {vb});"));
                        } else if matches!(op, VecOp::Min) {
                            self.emit_line(&format!("{d} = min({va}, {vb});"));
                        } else if matches!(op, VecOp::Not) {
                            self.emit_line(&format!("{d} = ~{va};"));
                        } else {
                            self.emit_line(&format!("{d} = {va} {op_str} {vb};"));
                        }
                    }
                }
                Ok(())
            }

            VmInstr::VecShiftImm { dst, a, amount, op, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let imm = *amount;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let line = match op {
                            VecShiftDir::Left => format!("shl.b32 {d}, {va}, {imm};"),
                            VecShiftDir::Right => format!("shr.b32 {d}, {va}, {imm};"),
                        };
                        self.emit_line(&line);
                    }
                    _ => {
                        let op_str = match op {
                            VecShiftDir::Left => "<<",
                            VecShiftDir::Right => ">>",
                        };
                        self.emit_line(&format!("{d} = {va} {op_str} {imm};"));
                    }
                }
                Ok(())
            }

            VmInstr::VecUnaryOp { dst, a, op } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let line = match op {
                            VecUnaryOp::Neg => format!("neg.f32 {d}, {va};"),
                            VecUnaryOp::Abs => format!("abs.f32 {d}, {va};"),
                            VecUnaryOp::Sqrt => format!("sqrt.approx.f32 {d}, {va};"),
                            VecUnaryOp::Rsqrt => format!("rsqrt.approx.f32 {d}, {va};"),
                            VecUnaryOp::Recip => format!("rcp.approx.f32 {d}, {va};"),
                            // PTX 没有 round/floor/ceil for f32 — 使用 cvt.rni / cvt.rmi / cvt.rpi
                            VecUnaryOp::Round => format!("cvt.rni.f32.f32 {d}, {va};"),
                            VecUnaryOp::Floor => format!("cvt.rmi.f32.f32 {d}, {va};"),
                            VecUnaryOp::Ceil  => format!("cvt.rpi.f32.f32 {d}, {va};"),
                            VecUnaryOp::IntToFloat => format!("cvt.rn.f32.s32 {d}, {va};"),
                            VecUnaryOp::Fp8E4M3ToFloat => format!("cvt.rn.f32.f8_e4m3 {d}, {va};"),
                            VecUnaryOp::Fp8E5M2ToFloat => format!("cvt.rn.f32.f8_e5m2 {d}, {va};"),
                        };
                        self.emit_line(&line);
                    }
                    _ => {
                        let op_str = match op {
                            VecUnaryOp::Neg => "-", VecUnaryOp::Abs => "fabs",
                            VecUnaryOp::Sqrt => "sqrt", VecUnaryOp::Rsqrt => "rsqrt",
                            VecUnaryOp::Recip => "1.0/", VecUnaryOp::Round => "round",
                            VecUnaryOp::Floor => "floor", VecUnaryOp::Ceil => "ceil",
                            VecUnaryOp::IntToFloat => "int_as_float",
                            VecUnaryOp::Fp8E4M3ToFloat => "fp8_e4m3_to_float",
                            VecUnaryOp::Fp8E5M2ToFloat => "fp8_e5m2_to_float",
                        };
                        self.emit_line(&format!("{d} = {op_str}({va});"));
                    }
                }
                Ok(())
            }

            VmInstr::Fma { dst, acc, a, b, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let vacc = self.reg_name_with_kind(*acc, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let vb = self.reg_name_with_kind(*b, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line(&format!("fma.rn.f32 {d}, {va}, {vb}, {vacc};")),
                    _ => self.emit_line(&format!("{d} = fma({va}, {vb}, {vacc});")),
                }
                Ok(())
            }

            VmInstr::Accumulate { acc, src } => {
                let a = self.reg_name_with_kind(*acc, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line(&format!("add.f32 {a}, {a}, {s};")),
                    _ => self.emit_line(&format!("{a} += {s};")),
                }
                Ok(())
            }

            VmInstr::Transcendental { dst, src, func } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                let fn_str = match func {
                    TranscendentalFn::Exp => "ex2.approx",
                    TranscendentalFn::Log => "lg2.approx",
                    TranscendentalFn::Tanh => "tanh.approx",
                    _ => "mov",
                };
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line(&format!("{fn_str}.f32 {d}, {s};")),
                    _ => self.emit_line(&format!("{d} = {fn_str}({s});")),
                }
                Ok(())
            }

            // ═══ 归约 ═══

            VmInstr::HReduce { dst, src, op } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                let op_str = match op {
                    ReduceOp::Sum => "add", ReduceOp::Max => "max",
                    ReduceOp::Min => "min", ReduceOp::Prod => "mul",
                    ReduceOp::LogSum => "add", // LogSum: sum(exp(x)), final log applied by caller
                };
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // ARCH-GPU-SCRATCH: shfl 归约结果存 scratch_vec[0]（%fs0）
                        let fs0 = self.scratch_vec_names[0];
                        self.emit_line(&format!("mov.f32 {d}, {s};"));
                        for delta in [16, 8, 4, 2, 1] {
                            self.emit_line(&format!("shfl.sync.bfly.b32 {fs0}, {d}, {delta}, 0x1f, 0xffffffff;"));
                            self.emit_line(&format!("{op_str}.f32 {d}, {d}, {fs0};"));
                        }
                    }
                    GpuDialect::Hip { wave_size, .. } => {
                        // HIP 同样需要 scratch 保存 shuffle 结果，否则 op(d, d) 计算错误
                        self.emit_line(&format!("float hip_shfl_tmp; {d} = {s};"));
                        let max_delta = wave_size / 2;
                        let mut delta = max_delta;
                        while delta >= 1 {
                            self.emit_line(&format!("hip_shfl_tmp = __shfl_xor({d}, {delta});"));
                            self.emit_line(&format!("{d} = {op_str}({d}, hip_shfl_tmp);"));
                            delta /= 2;
                        }
                    }
                    _ => self.emit_line(&format!("{d} = warp_reduce_{op_str}({s});")),
                }
                Ok(())
            }

            // ═══ Tensor Core MMA (SM 版本特化) ═══

            VmInstr::TileConfig { rows, cols, dtype } => {
                let sm = self.sm_version();
                if let Some(v) = sm {
                    if v >= 100 {
                        // SM100+ Blackwell: TMEM 分配
                        let tmem_cols = (*cols).max(32);
                        self.emit_line(&format!("// §SM100 tcgen05: allocate TMEM ({rows}×{tmem_cols} {dtype:?})"));
                        self.emit_line(&format!("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%smem_tmem_addr], {tmem_cols};"));
                        self.emit_line("mov.b32 %tmem_addr, %smem_tmem_addr;");
                        self.tmem_allocated = true;
                        // SPEC 15 REQ-JCTX-012: TMEM alloc 通过 JitContext, 超限返回 ResourceBudgetExceeded
                        self.jit_ctx.allocate(
                            crate::compiler::jit_context::ResourceKind::Tile,
                            "tmem_sm100",
                        ).map_err(|e| CompilerError::CodegenViolation(
                            format!("GPU TileConfig TMEM: {}", e)
                        ))?;
                    } else if v >= 90 {
                        // SM90 Hopper: mbarrier 初始化
                        self.emit_line("// §SM90 WGMMA: init mbarrier for async pipeline");
                        self.emit_line("mbarrier.init.shared.b64 [mbar], 1;");
                        // SPEC 15 REQ-JCTX-012: Barrier + TileAccumulator alloc 通过 JitContext
                        self.jit_ctx.allocate(
                            crate::compiler::jit_context::ResourceKind::Barrier,
                            "mbarrier_wgmma",
                        ).map_err(|e| CompilerError::CodegenViolation(
                            format!("GPU TileConfig Barrier: {}", e)
                        ))?;
                        self.jit_ctx.allocate(
                            crate::compiler::jit_context::ResourceKind::TileAccumulator,
                            "wgmma_fragment",
                        ).map_err(|e| CompilerError::CodegenViolation(
                            format!("GPU TileConfig TileAccumulator: {}", e)
                        ))?;
                    } else if v >= 70 {
                        self.emit_line(&format!("// §SM{v} TileConfig {rows}×{cols} {dtype:?}"));
                    }
                } else if let Some(gfx) = self.gfx_arch() {
                    self.emit_line(&format!("// gfx{gfx} MFMA TileConfig {rows}×{cols} {dtype:?}"));
                }
                Ok(())
            }

            VmInstr::TileMma { c, a, b } => {
                let vc = self.reg_name_with_kind(*c, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let vb = self.reg_name_with_kind(*b, alloc);
                match self.dialect {
                    GpuDialect::Ptx { sm_version } => {
                        if sm_version >= 100 {
                            // ── SM100+ Blackwell: tcgen05.mma ──
                            self.emit_line("// §SM100 tcgen05.mma (block-scaled, TMEM-backed)");
                            self.emit_line(&format!("// A-desc in {va}, B-desc in {vb}, C-tmem at %tmem_addr"));
                            self.emit_line("tcgen05.mma.cta_group::1.kind::f16");
                            self.emit_line(&format!("  [%tmem_addr], {va}, {vb}, 0x0,"));
                            self.emit_line("  0, 1;");
                            self.emit_line("tcgen05.wait::ld.sync.aligned;");
                        } else if sm_version >= 90 {
                            // ── SM90 Hopper: wgmma.mma_async ──
                            self.emit_line("// §SM90 WGMMA async MMA (warpgroup 128 threads)");
                            self.emit_line("wgmma.fence.sync.aligned;");
                            self.emit_line(&format!("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {{{vc}}}, {va}, {vb};"));
                            self.emit_line("wgmma.commit_group.sync.aligned;");
                            self.emit_line("wgmma.wait_group.sync.aligned 0;");
                        } else if sm_version >= 80 {
                            // ── SM80-89 Ampere/Ada: mma.sync ──
                            self.emit_line(&format!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {vc}, {va}, {vb}, {vc};"));
                        } else {
                            // ── SM70-79 Volta/Turing: wmma ──
                            self.emit_line(&format!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 {vc}, {va}, {vb}, {vc};"));
                        }
                    }
                    GpuDialect::Hip { gfx_arch, .. } => {
                        if gfx_arch >= 950 {
                            // ── gfx950 CDNA4: MFMA v2 32×32×16 ──
                            self.emit_line(&format!("v_mfma_f32_32x32x16_f16 {vc}, {va}, {vb}, {vc};"));
                        } else if gfx_arch >= 908 {
                            // ── gfx908+ CDNA2/3: MFMA v1 16×16×16 ──
                            self.emit_line(&format!("v_mfma_f32_16x16x16_f16 {vc}, {va}, {vb}, {vc};"));
                        } else {
                            self.emit_line("// RDNA: no MFMA, scalar FMA fallback");
                            self.emit_line(&format!("// {vc} += {va} * {vb}"));
                        }
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("// Metal simdgroup_matrix_multiply: {vc} += {va} * {vb}"));
                    }
                }
                Ok(())
            }

            VmInstr::TileRelease => {
                if self.tmem_allocated {
                    self.emit_line("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tmem_addr, 32;");
                    self.tmem_allocated = false;
                    // SPEC 15 REQ-JCTX-012: 释放 TMEM Tile 资源
                    let tile_count = self.jit_ctx.live_count(
                        crate::compiler::jit_context::ResourceKind::Tile,
                    );
                    if tile_count > 0 {
                        self.jit_ctx.release(
                            crate::compiler::jit_context::ResourceKind::Tile,
                            tile_count - 1,
                        );
                    }
                }
                // SPEC 15 REQ-JCTX-012: 释放 Barrier + TileAccumulator (SM90 WGMMA)
                let barrier_count = self.jit_ctx.live_count(
                    crate::compiler::jit_context::ResourceKind::Barrier,
                );
                if barrier_count > 0 {
                    self.jit_ctx.release(
                        crate::compiler::jit_context::ResourceKind::Barrier,
                        barrier_count - 1,
                    );
                }
                let acc_count = self.jit_ctx.live_count(
                    crate::compiler::jit_context::ResourceKind::TileAccumulator,
                );
                if acc_count > 0 {
                    self.jit_ctx.release(
                        crate::compiler::jit_context::ResourceKind::TileAccumulator,
                        acc_count - 1,
                    );
                }
                Ok(())
            }

            // ═══ 异步内存 (SM 版本特化) ═══

            VmInstr::AsyncCopy { dst, src, size } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        // SM90+ Hopper/Blackwell: TMA + cp.async.bulk
                        self.emit_line(&format!("// §SM90 TMA async bulk copy ({size} bytes)"));
                        self.emit_line(&format!("cp.async.bulk.shared::cluster.global [{d}], [{s}], {size}, [mbar];"));
                        self.emit_line("mbarrier.arrive.expect_tx.shared::cta.b64 [mbar], {size};");
                    }
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        // SM80-89: cp.async
                        self.emit_line(&format!("cp.async.ca.shared.global [{d}], [{s}], {size};"));
                    }
                    GpuDialect::Hip { gfx_arch, .. } if gfx_arch >= 950 => {
                        // gfx950: GLOBAL_LOAD_LDS 128-bit
                        self.emit_line(&format!("// gfx950 GLOBAL_LOAD_LDS: {s} → {d} ({size} bytes)"));
                        self.emit_line(&format!("global_load_lds {d}, {s}, off;"));
                    }
                    _ => return Err(CompilerError::CodegenViolation(format!("AsyncCopy: GPU dialect {:?} does not support async copy", self.dialect))),
                }
                Ok(())
            }

            VmInstr::AsyncWait { .. } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        self.emit_line("// §SM90 mbarrier wait");
                        let ps1 = self.scratch_pred_names[1];
                        self.emit_line(&format!("mbarrier.try_wait.parity.shared::cta.b64 {ps1}, [mbar], 0;"));
                    }
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        self.emit_line("cp.async.wait_all;");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("s_waitcnt lgkmcnt(0);");
                    }
                    _ => return Err(CompilerError::CodegenViolation(format!("AsyncWait: GPU dialect {:?} does not support async wait", self.dialect))),
                }
                Ok(())
            }

            // ═══ 同步 ═══

            VmInstr::WarpSync => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        self.emit_line("bar.sync 0;"); // SM90: prefer cuda::barrier for fine-grained
                    }
                    GpuDialect::Ptx { .. } => self.emit_line("bar.sync 0;"),
                    GpuDialect::Hip { .. } => self.emit_line("__syncthreads();"),
                    GpuDialect::Metal { .. } => self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);"),
                }
                Ok(())
            }

            VmInstr::SharedMemAlloc { name, bytes } => {
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(".shared .align 4 .b8 {}[{}];", name, bytes));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("__shared__ float {}[{}];", name, bytes / 4));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("threadgroup float {}[{}];", name, bytes / 4));
                    }
                }
                Ok(())
            }

            VmInstr::SharedMemStore { name, dst_offset, src, width, dtype } => {
                let offset_str = self.offset_to_string(dst_offset, alloc);
                let src_reg = self.reg_name_with_kind(*src, alloc);
                let dtype_str = dtype.gpu_native_type_name().map_err(|_| CompilerError::CodegenViolation(format!("SharedMemStore: unsupported dtype {:?} for GPU native store", dtype)))?;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("st.shared.{} [{} + {}], {};", dtype_str, name, offset_str, src_reg));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{}[({}) / 4] = {};", name, offset_str, src_reg));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{}[({}) / 4] = {};", name, offset_str, src_reg));
                    }
                }
                Ok(())
            }

            VmInstr::SharedMemLoad { dst, name, src_offset, width, dtype } => {
                let offset_str = self.offset_to_string(src_offset, alloc);
                let dst_reg = self.reg_name_with_kind(*dst, alloc);
                let dtype_str = dtype.gpu_native_type_name().map_err(|_| CompilerError::CodegenViolation(format!("SharedMemLoad: unsupported dtype {:?} for GPU native load", dtype)))?;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("ld.shared.{} {}, [{} + {}];", dtype_str, dst_reg, name, offset_str));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{} = {}[({}) / 4];", dst_reg, name, offset_str));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{} = {}[({}) / 4];", dst_reg, name, offset_str));
                    }
                }
                Ok(())
            }

            VmInstr::SharedMemAsyncStore { name, dst_offset, src, width, dtype } => {
                let offset_str = self.offset_to_string(dst_offset, alloc);
                let src_reg = self.reg_name_with_kind(*src, alloc);
                let dtype_str = dtype.gpu_native_type_name().map_err(|_| CompilerError::CodegenViolation(format!("SharedMemAsyncStore: unsupported dtype {:?} for GPU async store", dtype)))?;
                let bytes_per_elem = dtype.gpu_compute_bytes();
                let copy_bytes = width.f32_lanes().max(1) * bytes_per_elem;
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        // SM80+: cp.async.ca.shared.global
                        self.emit_line(&format!("cp.async.ca.shared.global [{} + {}], [{}], {};", name, offset_str, src_reg, copy_bytes));
                    }
                    GpuDialect::Ptx { .. } => {
                        // SM<80: no cp.async, use st.shared as synchronous fallback
                        self.emit_line(&format!("st.shared.{} [{} + {}], {};  // async fallback for SM<80", dtype_str, name, offset_str, src_reg));
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP: synchronous shared store
                        self.emit_line(&format!("{}[({}) / 4] = {};  // async fallback", name, offset_str, src_reg));
                    }
                    GpuDialect::Metal { .. } => {
                        // Metal: synchronous shared store
                        self.emit_line(&format!("{}[({}) / 4] = {};  // async fallback", name, offset_str, src_reg));
                    }
                }
                Ok(())
            }

            VmInstr::SharedMemAsyncWaitGroup { n } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        self.emit_line(&format!("cp.async.wait_group {};", n));
                    }
                    GpuDialect::Ptx { .. } => {
                        // SM<80: no cp.async, use bar.sync as synchronous barrier
                        self.emit_line(&format!("bar.sync {};  // async wait fallback for SM<80", n));
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP: use workgroup barrier
                        self.emit_line("work_groupBarrier();");
                    }
                    GpuDialect::Metal { .. } => {
                        // Metal: use threadgroup barrier
                        self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
                    }
                }
                Ok(())
            }

            VmInstr::WarpRoleDeclare { role } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        match role {
                            0 => {
                                // Producer: increase register allocation for TMA-heavy warps
                                self.emit_line("// §SM90 Warp Specialization: Producer warp (TMA load)");
                                self.emit_line("setmaxnreg.inc.sync.allocating_group.u32 32;");
                            }
                            1 => {
                                // Consumer: decrease register allocation for WGMMA-heavy warps
                                self.emit_line("// §SM90 Warp Specialization: Consumer warp (WGMMA compute)");
                                self.emit_line("setmaxnreg.dec.sync.allocating_group.u32 32;");
                            }
                            _ => {
                                self.emit_line(&format!(
                                    "// WarpRoleDeclare: unknown role {} (NOP)", role
                                ));
                            }
                        }
                    }
                    _ => {
                        // SM80-/HIP/Metal: no warp specialization support
                    }
                }
                Ok(())
            }

            VmInstr::WarpBarrierArrive { barrier_name, tx_bytes } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        // SM90+: mbarrier.arrive.expect_tx — Producer signals data ready
                        let barrier = self.resolve_barrier_symbol(barrier_name);
                        self.emit_line(&format!(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 [{barrier}], {tx_bytes};"
                        ));
                    }
                    _ => {
                        // SM80-/HIP/Metal: NOP (cp.async.commit_group handled by SharedMemAsyncWaitGroup)
                    }
                }
                Ok(())
            }

            VmInstr::WarpBarrierWait { barrier_name, parity } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        // SM90+: mbarrier.try_wait.parity — Consumer waits for data
                        let barrier = self.resolve_barrier_symbol(barrier_name);
                        let ps0 = self.scratch_pred_names[0];
                        self.emit_line(&format!(
                            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 {ps0}, [{barrier}], {parity};"
                        ));
                    }
                    _ => {
                        // SM80-/HIP/Metal: NOP
                    }
                }
                Ok(())
            }

            VmInstr::TmaDescriptorInit { desc_name, global_dim, global_stride, box_dim, swizzle, dtype } => {
                // TMA descriptor is initialized on Host side (cuTensorMapEncodeTiled).
                // GPU kernel receives the descriptor as an ABI parameter; no PTX emission needed.
                // Emit a descriptive comment for PTX readability.
                self.emit_line(&format!(
                    "// TMA Descriptor: {} dim=[{},{}] stride=[{},{}] box=[{},{}] swizzle={:?} dtype={:?}",
                    desc_name, global_dim[0], global_dim[1], global_stride[0], global_stride[1],
                    box_dim[0], box_dim[1], swizzle, dtype
                ));
                Ok(())
            }

            VmInstr::Tma2DCopy { desc_name, smem_name, coord_x, coord_y, barrier_name } => {
                let cx = self.reg_name_with_kind(*coord_x, alloc);
                let cy = self.reg_name_with_kind(*coord_y, alloc);
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        // SM90+: TMA 2D tensor copy
                        // cp.async.bulk.tensor.2d.shared.global [%smem], [%desc], {%cx, %cy}, [%barrier];
                        let barrier = self.resolve_barrier_symbol(barrier_name);
                        self.emit_line(&format!(
                            "cp.async.bulk.tensor.2d.shared.global [{}], [{}], {{{}, {}}}, [{}];",
                            smem_name, desc_name, cx, cy, barrier
                        ));
                        // mbarrier arrive — signals to consumer that data is in flight.
                        // The TMA descriptor carries box dimensions, so hardware knows the transfer size.
                        // arrive.expect_tx with 0 bytes: the TX size is encoded in the cp.async.bulk op itself.
                        self.emit_line(&format!(
                            "mbarrier.arrive.expect_tx.shared::cta.b64 [{}], 0;",
                            barrier
                        ));
                    }
                    GpuDialect::Ptx { sm_version } => {
                        return Err(CompilerError::CodegenViolation(
                            format!("Tma2DCopy requires SM90+ (current SM{}), use 1D cp.async path for SM80", sm_version)
                        ));
                    }
                    GpuDialect::Hip { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "Tma2DCopy requires SM90+ PTX (HIP does not support TMA 2D)".into()
                        ));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "Tma2DCopy requires SM90+ PTX (Metal does not support TMA 2D)".into()
                        ));
                    }
                }
                Ok(())
            }

            VmInstr::BarrierInit { name, thread_count } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        let barrier = self.resolve_barrier_symbol(name);
                        self.emit_line(&format!(
                            "mbarrier.init.shared::cta.b64 [{barrier}], {thread_count};"
                        ));
                    }
                    _ => {
                        // SM80-/HIP/Metal: NOP (uses cp.async.wait_group instead)
                    }
                }
                Ok(())
            }

            VmInstr::BlockSync => {
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line("bar.sync 0;");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("__syncthreads();");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
                    }
                }
                Ok(())
            }

            VmInstr::WarpReduce { op, src, dst, width } => {
                let src_reg = self.reg_name_with_kind(*src, alloc);
                let dst_reg = self.reg_name_with_kind(*dst, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // 5-step tree reduction for 32 threads
                        match op {
                            ReduceOp::Sum => {
                                self.emit_line("// warp reduction sum");
                                self.emit_line(&format!("shfl.sync.down.b32 %tmp1, {}, 16, 0x1f;", src_reg));
                                self.emit_line(&format!("add.f32 %tmp2, {}, %tmp1;", src_reg));
                                self.emit_line("shfl.sync.down.b32 %tmp3, %tmp2, 8, 0x1f;");
                                self.emit_line("add.f32 %tmp4, %tmp2, %tmp3;");
                                self.emit_line("shfl.sync.down.b32 %tmp5, %tmp4, 4, 0x1f;");
                                self.emit_line("add.f32 %tmp6, %tmp4, %tmp5;");
                                self.emit_line("shfl.sync.down.b32 %tmp7, %tmp6, 2, 0x1f;");
                                self.emit_line("add.f32 %tmp8, %tmp6, %tmp7;");
                                self.emit_line("shfl.sync.down.b32 %tmp9, %tmp8, 1, 0x1f;");
                                self.emit_line(&format!("add.f32 {}, %tmp8, %tmp9;", dst_reg));
                            }
                            ReduceOp::Max => {
                                self.emit_line("// warp reduction max");
                                self.emit_line(&format!("shfl.sync.down.b32 %tmp1, {}, 16, 0x1f;", src_reg));
                                self.emit_line(&format!("max.f32 %tmp2, {}, %tmp1;", src_reg));
                                self.emit_line("shfl.sync.down.b32 %tmp3, %tmp2, 8, 0x1f;");
                                self.emit_line("max.f32 %tmp4, %tmp2, %tmp3;");
                                self.emit_line("shfl.sync.down.b32 %tmp5, %tmp4, 4, 0x1f;");
                                self.emit_line("max.f32 %tmp6, %tmp4, %tmp5;");
                                self.emit_line("shfl.sync.down.b32 %tmp7, %tmp6, 2, 0x1f;");
                                self.emit_line("max.f32 %tmp8, %tmp6, %tmp7;");
                                self.emit_line("shfl.sync.down.b32 %tmp9, %tmp8, 1, 0x1f;");
                                self.emit_line(&format!("max.f32 {}, %tmp8, %tmp9;", dst_reg));
                            }
                            ReduceOp::Min => {
                                self.emit_line("// warp reduction min");
                                self.emit_line(&format!("shfl.sync.down.b32 %tmp1, {}, 16, 0x1f;", src_reg));
                                self.emit_line(&format!("min.f32 %tmp2, {}, %tmp1;", src_reg));
                                self.emit_line("shfl.sync.down.b32 %tmp3, %tmp2, 8, 0x1f;");
                                self.emit_line("min.f32 %tmp4, %tmp2, %tmp3;");
                                self.emit_line("shfl.sync.down.b32 %tmp5, %tmp4, 4, 0x1f;");
                                self.emit_line("min.f32 %tmp6, %tmp4, %tmp5;");
                                self.emit_line("shfl.sync.down.b32 %tmp7, %tmp6, 2, 0x1f;");
                                self.emit_line("min.f32 %tmp8, %tmp6, %tmp7;");
                                self.emit_line("shfl.sync.down.b32 %tmp9, %tmp8, 1, 0x1f;");
                                self.emit_line(&format!("min.f32 {}, %tmp8, %tmp9;", dst_reg));
                            }
                            _ => {
                                self.emit_line("// unsupported warp reduction op");
                            }
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("// hip warp reduction");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("// metal warp reduction");
                    }
                }
                Ok(())
            }

            // ═══ 控制流（修复循环标签） ═══

            VmInstr::LoopBegin { counter, byte_offset, bound, step_bytes } => {
                let c = self.reg_name_with_kind(*counter, alloc);
                let off = self.reg_name_with_kind(*byte_offset, alloc);
                let label_id = self.next_loop_label();
                // ARCH-GPU-LOOP-TRACKING: 保存完整状态到栈，LoopEnd 从栈读
                self.loop_stack.push((label_id, c.clone(), off.clone(), *step_bytes));
                let ps0 = self.scratch_pred_names[0];
                let rs_bound = self.scratch_gpr_names[2]; // %rs_bound

                match bound {
                    BoundExpr::Const(n) => match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("mov.u32 {c}, 0;"));
                            self.emit_line(&format!("mov.u32 {off}, 0;"));
                            self.emit_line(&format!("LOOP_{label_id}:"));
                            self.emit_line(&format!("setp.ge.u32 {ps0}, {c}, {n};"));
                            self.emit_line(&format!("@{ps0} bra LOOP_END_{label_id};"));
                        }
                        _ => {
                            self.emit_line(&format!("for (int {c} = 0, {off} = 0; {c} < {n}; {c}++, {off} += {step_bytes}) {{"));
                            self.indent += 1;
                        }
                    },
                    BoundExpr::Runtime(PtrExpr::AbiArg(idx)) => {
                        let param = self.param_name(*idx)?;
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("ld.param.u32 {rs_bound}, [{param}];"));
                                self.emit_line(&format!("mov.u32 {c}, 0;"));
                                self.emit_line(&format!("mov.u32 {off}, 0;"));
                                self.emit_line(&format!("LOOP_{label_id}:"));
                                self.emit_line(&format!("setp.ge.u32 {ps0}, {c}, {rs_bound};"));
                                self.emit_line(&format!("@{ps0} bra LOOP_END_{label_id};"));
                            }
                            _ => {
                                self.emit_line(&format!("for (int {c} = 0, {off} = 0; {c} < {param}; {c}++, {off} += {step_bytes}) {{"));
                                self.indent += 1;
                            }
                        }
                    }
                    // ARCH-SYMDIM-NO-CONST-DEGRADE: 符号维度 (如 seq_len) 在 GPU 上
                    // 从 `.param` 读运行时值。当前 GPU ABI 只有 `seq_len` 暴露,
                    // 其他符号维度 (如 total_seq) 走别名也映射到 seq_len。
                    BoundExpr::Symbolic(sym) => {
                        // 在 abi_param_names 中查找名字对应的索引。
                        // sym_dim_aliases ("total_seq" → "seq_len") 在 SymDimSlotMap 层
                        // 已 resolve — 但此处 BoundExpr::Symbolic 仍带原始名字 (GEMM 维度
                        // 用 seq_len 本身), 直接匹配 abi_param_names。
                        let (idx, _) = self.abi_param_names.iter().enumerate()
                            .find(|(_, name)| **name == sym.name.as_str() || (sym.name == "total_seq" && **name == "seq_len"))
                            .ok_or_else(|| CompilerError::CodegenViolation(format!(
                                "GPU LoopBegin: Symbolic dim '{}' 未在 GPU ABI .param 中声明。\
                                 可用参数: {:?} (ARCH-GPU-ABI)",
                                sym.name, self.abi_param_names)))?;
                        let param = self.param_name(idx as u8)?;
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("ld.param.u32 {rs_bound}, [{param}];"));
                                self.emit_line(&format!("mov.u32 {c}, 0;"));
                                self.emit_line(&format!("mov.u32 {off}, 0;"));
                                self.emit_line(&format!("LOOP_{label_id}:"));
                                self.emit_line(&format!("setp.ge.u32 {ps0}, {c}, {rs_bound};"));
                                self.emit_line(&format!("@{ps0} bra LOOP_END_{label_id};"));
                            }
                            _ => {
                                self.emit_line(&format!("for (int {c} = 0, {off} = 0; {c} < {param}; {c}++, {off} += {step_bytes}) {{"));
                                self.indent += 1;
                            }
                        }
                    }
                    // ARCH-CAUSAL: counter < vreg_value — 动态上界由前置 VReg 提供。
                    BoundExpr::DynamicVReg(vreg_id) => {
                        let bound_reg = self.reg_name_with_kind(*vreg_id, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("mov.u32 {c}, 0;"));
                                self.emit_line(&format!("mov.u32 {off}, 0;"));
                                self.emit_line(&format!("LOOP_{label_id}:"));
                                self.emit_line(&format!("setp.ge.u32 {ps0}, {c}, {bound_reg};"));
                                self.emit_line(&format!("@{ps0} bra LOOP_END_{label_id};"));
                            }
                            _ => {
                                self.emit_line(&format!("for (int {c} = 0, {off} = 0; {c} < {bound_reg}; {c}++, {off} += {step_bytes}) {{"));
                                self.indent += 1;
                            }
                        }
                    }
                    // ARCH-CAUSAL: counter < vreg_value + 1 (即 counter ≤ vreg_value)
                    BoundExpr::DynamicVRegPlusOne(vreg_id) => {
                        let bound_reg = self.reg_name_with_kind(*vreg_id, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                // 临时用 rs_bound = bound_reg + 1
                                self.emit_line(&format!("add.u32 {rs_bound}, {bound_reg}, 1;"));
                                self.emit_line(&format!("mov.u32 {c}, 0;"));
                                self.emit_line(&format!("mov.u32 {off}, 0;"));
                                self.emit_line(&format!("LOOP_{label_id}:"));
                                self.emit_line(&format!("setp.ge.u32 {ps0}, {c}, {rs_bound};"));
                                self.emit_line(&format!("@{ps0} bra LOOP_END_{label_id};"));
                            }
                            _ => {
                                self.emit_line(&format!("for (int {c} = 0, {off} = 0; {c} <= {bound_reg}; {c}++, {off} += {step_bytes}) {{"));
                                self.indent += 1;
                            }
                        }
                    }
                    // ARCH-NO-SILENT-FALLBACK: 其他 BoundExpr 变体 (如 Runtime(StackArg)) 在 GPU
                    // ABI 下不适用, 必须显式报错而非静默无限循环。
                    BoundExpr::Runtime(other) => {
                        return Err(CompilerError::CodegenViolation(format!(
                            "GPU LoopBegin: BoundExpr::Runtime({:?}) 不适用于 GPU ABI (\
                             GPU 只支持 AbiArg — NO_SILENT_FALLBACK)",
                            other)));
                    }
                }
                Ok(())
            }

            VmInstr::LoopEnd => {
                // ARCH-GPU-LOOP-TRACKING: 从栈读取真实 counter/offset 名字，禁止硬编码 %r0
                let (label_id, c, off, step_bytes) = self.loop_stack.pop().ok_or_else(|| {
                    CompilerError::CodegenViolation("LoopEnd without matching LoopBegin".into())
                })?;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("add.u32 {c}, {c}, 1;"));
                        self.emit_line(&format!("add.u32 {off}, {off}, {step_bytes};"));
                        self.emit_line(&format!("bra LOOP_{label_id};"));
                        self.emit_line(&format!("LOOP_END_{label_id}:"));
                    }
                    _ => {
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // ═══ 条件控制流 ═══

            VmInstr::ConditionalSkip { mask, skip_count } => {
                let m = self.reg_name_with_kind(*mask, alloc);
                let skip_id = self.next_skip_label();
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("setp.eq.f32 {ps0}, {m}, 0.0;"));
                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                    }
                    _ => self.emit_line(&format!("if ({m} != 0.0) {{")),
                }
                let _ = skip_count;
                Ok(())
            }
            // REQ-VR-003: GprCondAction — 统一条件操作 (替代旧 GprSkipIfNull/GprBitTest/GprCmpExit)
            VmInstr::GprCondAction { cond, action } => {
                let ps0 = self.scratch_pred_names[0];
                match cond {
                    GprCondition::IsNull(ptr) => {
                        let p = self.reg_name_with_kind(*ptr, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("setp.eq.u64 {ps0}, {p}, 0;"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            _ => self.emit_line(&format!("if ({p} != 0UL) {{")),
                        }
                    }
                    GprCondition::IsNonNull(ptr) => {
                        let p = self.reg_name_with_kind(*ptr, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("setp.ne.u64 {ps0}, {p}, 0;"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            _ => self.emit_line(&format!("if ({p} == 0UL) {{")),
                        }
                    }
                    GprCondition::CmpEq(a, imm) => {
                        let va = self.reg_name_with_kind(*a, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("setp.eq.u32 {ps0}, {va}, {imm};"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            _ => {
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        self.emit_line(&format!("if ({va} == {imm}u) {{"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("if ({va} == {imm}u) goto {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("if ({va} == {imm}u) goto {label};"));
                                    }
                                }
                            }
                        }
                    }
                    GprCondition::CmpLtU(a, imm) => {
                        let va = self.reg_name_with_kind(*a, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("setp.lt.u32 {ps0}, {va}, {imm};"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            _ => {
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        self.emit_line(&format!("if ({va} < {imm}u) {{"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("if ({va} < {imm}u) goto {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("if ({va} < {imm}u) goto {label};"));
                                    }
                                }
                            }
                        }
                    }
                    GprCondition::CmpGeU(a, imm) => {
                        let va = self.reg_name_with_kind(*a, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("setp.ge.u32 {ps0}, {va}, {imm};"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            _ => {
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        self.emit_line(&format!("if ({va} >= {imm}u) {{"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("if ({va} >= {imm}u) goto {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("if ({va} >= {imm}u) goto {label};"));
                                    }
                                }
                            }
                        }
                    }
                    GprCondition::BitClear(bitmap, bit) => {
                        let bm = self.reg_name_with_kind(*bitmap, alloc);
                        let mask = 1u32 << *bit as u32;
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("and.b32 {bm}, {bm}, {mask};"));
                                self.emit_line(&format!("setp.eq.u32 {ps0}, {bm}, 0;"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                                match action {
                                    GprBranchAction::Skip(count) => {
                                        self.emit_line(&format!("if (({bm} & {mask}u) == 0u) {{"));
                                        for _ in 0..*count { self.emit_line("// skip"); }
                                        self.emit_line("}");
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("if (({bm} & {mask}u) == 0u) goto {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("if (({bm} & {mask}u) == 0u) goto {label};"));
                                    }
                                }
                            }
                        }
                    }
                    GprCondition::BitSet(bitmap, bit) => {
                        let bm = self.reg_name_with_kind(*bitmap, alloc);
                        let mask = 1u32 << *bit as u32;
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("and.b32 {bm}, {bm}, {mask};"));
                                self.emit_line(&format!("setp.ne.u32 {ps0}, {bm}, 0;"));
                                match action {
                                    GprBranchAction::Skip(_) => {
                                        let skip_id = self.next_skip_label();
                                        self.emit_line(&format!("@{ps0} bra SKIP_{skip_id};"));
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("@{ps0} bra {label};"));
                                    }
                                }
                            }
                            GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                                match action {
                                    GprBranchAction::Skip(count) => {
                                        self.emit_line(&format!("if (({bm} & {mask}u) != 0u) {{"));
                                        for _ in 0..*count { self.emit_line("// skip"); }
                                        self.emit_line("}");
                                    }
                                    GprBranchAction::Exit(_) => {
                                        self.emit_line(&format!("if (({bm} & {mask}u) != 0u) goto {};", self.epilogue_label));
                                    }
                                    GprBranchAction::JumpToLabel(label_id) => {
                                        let label = self.path_labels.get(label_id)
                                            .cloned()
                                            .unwrap_or_else(|| format!("LABEL_{label_id}"));
                                        self.emit_line(&format!("if (({bm} & {mask}u) != 0u) goto {label};"));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }

            VmInstr::ConditionalExit { condition, .. } => {
                let c = self.reg_name_with_kind(*condition, alloc);
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("setp.ne.u32 {ps0}, {c}, 0;"));
                        self.emit_line(&format!("@{ps0} ret;"));
                    }
                    _ => self.emit_line(&format!("if ({c}) return;")),
                }
                Ok(())
            }

            VmInstr::BranchIfPtrNonNull { ptr, target_label } => {
                let p = self.reg_name_with_kind(*ptr, alloc);
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("setp.ne.u64 {ps0}, {p}, 0;"));
                        self.emit_line(&format!("@{ps0} bra BATCH_MODE_{target_label};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("if ({p} != 0) goto batch_mode_{target_label};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("if ({p} != 0) {{ goto batch_mode_{target_label}; }}"));
                    }
                }
                Ok(())
            }

            VmInstr::BranchIfGprZero { value, target_label } => {
                let v = self.reg_name_with_kind(*value, alloc);
                let label_str = format!("LABEL_{target_label}");
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let ps0 = self.scratch_pred_names[0];
                        self.emit_line(&format!("setp.eq.u32 {ps0}, {v}, 0;"));
                        self.emit_line(&format!("@{ps0} bra {label_str};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("if ({v} == 0) goto {label_str};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("if ({v} == 0) {{ goto {label_str}; }}"));
                    }
                }
                Ok(())
            }

            VmInstr::BranchIfGprLtU { a, b, target_label } => {
                let va = self.reg_name_with_kind(*a, alloc);
                let vb = self.reg_name_with_kind(*b, alloc);
                let label_str = format!("LABEL_{target_label}");
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let ps0 = self.scratch_pred_names[0];
                        self.emit_line(&format!("setp.lt.u32 {ps0}, {va}, {vb};"));
                        self.emit_line(&format!("@{ps0} bra {label_str};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("if ((unsigned){va} < (unsigned){vb}) goto {label_str};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("if ({va} < {vb}) {{ goto {label_str}; }}"));
                    }
                }
                Ok(())
            }

            VmInstr::UnconditionalBranch { target_label } => {
                let label_str = format!("LABEL_{target_label}");
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("bra {label_str};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("goto {label_str};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("goto {label_str};"));
                    }
                }
                Ok(())
            }

            // §20 BCI-004: BatchSeqIdLookup — cumulative prompt_len scan
            VmInstr::BatchSeqIdLookup { dst, pt_offset_out, token_index, batch_ctx_ptr } => {
                let dst_name = self.reg_name_with_kind(*dst, alloc);
                let pt_name = self.reg_name_with_kind(*pt_offset_out, alloc);
                let idx_name = self.reg_name_with_kind(*token_index, alloc);
                let ctx_name = self.reg_name_with_kind(*batch_ctx_ptr, alloc);

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load num_seqs, then linear scan
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 _ctx_base;");
                        self.emit_line(&format!("  mov.u64 _ctx_base, {};", ctx_name));
                        self.emit_line("  .reg .u64 _num_seqs;");
                        self.emit_line("  ld.global.u64 _num_seqs, [_ctx_base];");
                        self.emit_line("  .reg .u32 _cumsum, _seq_idx, _prompt_len;");
                        self.emit_line("  mov.u32 _cumsum, 0;");
                        self.emit_line("  mov.u32 _seq_idx, 0;");
                        self.emit_line("  .reg .pred _found;");
                        self.emit_line("  setp.eq.u32 _found, 1, 0; // false");
                        // Loop header
                        let label = self.next_loop_label();
                        self.emit_line(&format!("$L_batch_scan_{}:", label));
                        // seq_meta_addr = ctx + 88 + seq_idx * 64 (BCI6 header=88)
                        self.emit_line("  .reg .u64 _meta;");
                        self.emit_line("  .reg .u32 _off;");
                        self.emit_line("  mad.lo.u32 _off, _seq_idx, 64, 88;");
                        self.emit_line("  add.u64 _meta, _ctx_base, _off;");
                        // prompt_len at meta+8
                        self.emit_line("  ld.global.u32 _prompt_len, [_meta+8];");
                        self.emit_line("  add.u32 _cumsum, _cumsum, _prompt_len;");
                        // if cumsum > token_index, found
                        self.emit_line(&format!("  setp.gt.u32 _found, _cumsum, {};", idx_name));
                        self.emit_line("  @!_found add.u32 _seq_idx, _seq_idx, 1;");
                        self.emit_line("  @!_found setp.lt.u32 _found, _seq_idx, _num_seqs;");
                        self.emit_line(&format!("  @!_found bra $L_batch_scan_{};", label));
                        // Store result
                        self.emit_line(&format!("  selp.u32 {}, _seq_idx, 0, _found;", dst_name));
                        // pt_offset at meta+16
                        self.emit_line("  .reg .u32 _pt_off;");
                        self.emit_line("  add.u64 _meta, _meta, 16;");
                        self.emit_line("  @ _found ld.global.u32 _pt_off, [_meta];");
                        self.emit_line(&format!("  selp.u32 {}, _pt_off, 0, _found;", pt_name));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  uint32_t _idx = {};", idx_name));
                        self.emit_line(&format!("  auto _ctx = (uint8_t*){};", ctx_name));
                        self.emit_line("  uint64_t _ns = *(uint64_t*)_ctx;");
                        self.emit_line("  uint32_t _cum = 0;");
                        self.emit_line(&format!("  uint32_t {} = 0, {} = 0;", dst_name, pt_name));
                        self.emit_line("  for (uint32_t _s = 0; _s < _ns; _s++) {");
                        self.emit_line("    auto _m = (uint32_t*)(_ctx + 88 + _s * 64); // BCI6");
                        self.emit_line("    _cum += _m[2]; // prompt_len at +8");
                        self.emit_line("    if (_cum > _idx) {");
                        self.emit_line(&format!("      {} = _s;", dst_name));
                        self.emit_line(&format!("      {} = _m[4]; // page_table_offset at +16", pt_name));
                        self.emit_line("      break;");
                        self.emit_line("    }");
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // §20 BCI-006: BatchPerSeqArgmax — scalar scan for max logit
            VmInstr::BatchPerSeqArgmax { dst, seq_id, logits_flat_ptr, vocab_size, width: _ } => {
                let dst_name = self.reg_name_with_kind(*dst, alloc);
                let seq_name = self.reg_name_with_kind(*seq_id, alloc);
                let base_name = self.reg_name_with_kind(*logits_flat_ptr, alloc);

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let row_bytes = *vocab_size * 4;
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 _row;");
                        self.emit_line(&format!("  mad.lo.u64 _row, {}, {}, {};", seq_name, row_bytes, base_name));
                        self.emit_line("  .reg .f32 _max_val;");
                        self.emit_line("  .reg .u32 _max_idx, _i;");
                        self.emit_line("  mov.u32 _max_idx, 0;");
                        self.emit_line("  mov.u32 _i, 0;");
                        self.emit_line("  ld.global.f32 _max_val, [_row];");
                        let label = self.next_loop_label();
                        self.emit_line(&format!("$L_batch_argmax_{}:", label));
                        self.emit_line("  add.u32 _i, _i, 1;");
                        self.emit_line(&format!("  setp.ge.u32 _done, _i, {};", *vocab_size));
                        self.emit_line(&format!("  @_done bra $L_batch_argmax_done_{};", label));
                        self.emit_line("  .reg .f32 _val;");
                        self.emit_line("  .reg .u32 _byte_off;");
                        self.emit_line("  shl.b32 _byte_off, _i, 2;");
                        self.emit_line("  .reg .u64 _addr;");
                        self.emit_line("  add.u64 _addr, _row, _byte_off;");
                        self.emit_line("  ld.global.f32 _val, [_addr];");
                        self.emit_line("  .reg .pred _gt;");
                        self.emit_line("  setp.gt.f32 _gt, _val, _max_val;");
                        self.emit_line("  @_gt mov.f32 _max_val, _val;");
                        self.emit_line("  @_gt mov.u32 _max_idx, _i;");
                        self.emit_line(&format!("  bra $L_batch_argmax_{};", label));
                        self.emit_line(&format!("$L_batch_argmax_done_{}:", label));
                        self.emit_line(&format!("  mov.u32 {}, _max_idx;", dst_name));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                        let row_bytes = *vocab_size * 4;
                        self.emit_line("{");
                        self.emit_line(&format!("  float* _row = (float*){} + (size_t){} * {};", base_name, seq_name, *vocab_size));
                        self.emit_line("  float _max = _row[0];");
                        self.emit_line(&format!("  uint32_t {} = 0;", dst_name));
                        self.emit_line(&format!("  for (uint32_t _i = 1; _i < {}; _i++) {{", *vocab_size));
                        self.emit_line("    if (_row[_i] > _max) { _max = _row[_i];");
                        self.emit_line(&format!("      {} = _i; }}", dst_name));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // §20 BCI-006: BatchPerSeqStopCheck — condition check + active_flag update
            VmInstr::BatchPerSeqStopCheck { seq_id, token_id, batch_ctx_ptr } => {
                let seq_name = self.reg_name_with_kind(*seq_id, alloc);
                let tok_name = self.reg_name_with_kind(*token_id, alloc);
                let ctx_name = self.reg_name_with_kind(*batch_ctx_ptr, alloc);

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 _meta;");
                        self.emit_line(&format!("  mad.lo.u32 _off, {}, 64, 88; // BCI6", seq_name));
                        self.emit_line(&format!("  add.u64 _meta, {}, _off;", ctx_name));
                        // Check eos: token_id == eos_token_id at meta+16
                        self.emit_line("  .reg .u32 _eos;");
                        self.emit_line("  ld.global.u32 _eos, [_meta + 16];");
                        self.emit_line("  .reg .pred _is_eos;");
                        self.emit_line(&format!("  setp.eq.u32 _is_eos, {}, _eos;", tok_name));
                        // Check gen_count >= max_new_tokens
                        self.emit_line("  .reg .u32 _gen, _max_tok;");
                        self.emit_line("  ld.global.u32 _gen, [_meta + 8];");
                        self.emit_line("  ld.global.u32 _max_tok, [_meta + 12];");
                        self.emit_line("  .reg .pred _over;");
                        self.emit_line("  setp.ge.u32 _over, _gen, _max_tok;");
                        // Deactivate if either
                        self.emit_line("  .reg .pred _deact;");
                        self.emit_line("  or.pred _deact, _is_eos, _over;");
                        self.emit_line("  @ _deact st.global.u32 [_meta + 24], 0;");
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  auto _ctx = (uint8_t*){};", ctx_name));
                        self.emit_line(&format!("  auto _m = (uint32_t*)(_ctx + 88 + (size_t){} * 64); // BCI6", seq_name));
                        self.emit_line(&format!("  bool _stop = ({} == _m[4]) || (_m[2] >= _m[3]);", tok_name)); // eos at +16=4*4, gen at +8=2*4, max at +12=3*4
                        self.emit_line("  if (_stop) _m[6] = 0; // active_flag at +24=6*4");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            VmInstr::IndirectJump { index, targets } => {
                let idx = self.reg_name_with_kind(*index, alloc);
                let ps0 = self.scratch_pred_names[0];
                for (i, _) in targets.iter().enumerate() {
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("setp.eq.u32 {ps0}, {idx}, {i};"));
                            self.emit_line(&format!("@{ps0} bra EXPERT_{i};"));
                        }
                        _ => self.emit_line(&format!("if ({idx} == {i}) goto expert_{i};")),
                    }
                }
                Ok(())
            }

            VmInstr::HotpatchSlot { .. } => {
                // GPU hotpatch: 8-byte NOP trampoline for runtime JMP patching.
                // Only PTX supports this mechanism.
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line("// hotpatch: 8-byte NOP trampoline"),
                    _ => return Err(CompilerError::CodegenViolation(format!("HotpatchSlot: GPU dialect {:?} does not support hotpatch", self.dialect))),
                }
                Ok(())
            }

            VmInstr::Prefetch { base, hint, .. } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        let d = self.reg_name_with_kind(*base, alloc);
                        let hint_str = match hint {
                            super::isa_hook::PrefetchHint::T0 => ".L1",
                            super::isa_hook::PrefetchHint::T1 => ".L2",
                            _ => ".L1",
                        };
                        self.emit_line(&format!("prefetch.global{hint_str} [{d}];"));
                    }
                    _ => {} // GPU prefetch 由 texture cache 自动管理
                }
                Ok(())
            }

            // x86-specific: SPARSE_MASK_INTERSECT has no GPU equivalent
            VmInstr::SparseMaskIntersect { .. } => {
                Err(CompilerError::CodegenViolation("GPU: SPARSE_MASK_INTERSECT is x86-only".into()))
            }

            // REQ-CG-008: Scalar ops — GPU lowering (PTX/HIP/Metal)

            // ScalarLoad: dst(gpr) = *(u32*)(base + offset)
            VmInstr::ScalarLoad { dst, base, offset } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("ld.global.u32 {d}, [%rd_addr];"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{d} = *((unsigned int*){b} + ({off})/4);"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = ((device unsigned int*){b})[({off})/4];"));
                    }
                }
                Ok(())
            }

            // ScalarStore: *(u32*)(base + offset) = src(gpr)
            VmInstr::ScalarStore { base, src, offset } => {
                let s = self.reg_name_with_kind(*src, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("st.global.u32 [%rd_addr], {s};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("*((unsigned int*){b} + ({off})/4) = {s};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("((device unsigned int*){b})[({off})/4] = {s};"));
                    }
                }
                Ok(())
            }

            // ScalarToIndex: dst(gpr) = (int)src(float) * stride
            VmInstr::ScalarToIndex { dst, src, stride } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("cvt.rni.s32.f32 {d}, {s};"));
                        if *stride != 1 {
                            self.emit_line(&format!("mul.lo.s32 {d}, {d}, {stride};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{d} = (int)({s}) * {stride};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = as_type<int>({s}) * {stride};"));
                    }
                }
                Ok(())
            }

            // IndexToScalar: dst(xmm/float) = (float)src(gpr/int)
            VmInstr::IndexToScalar { dst, src } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("cvt.rn.f32.s32 {d}, {s};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{d} = (float)({s});"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = float({s});"));
                    }
                }
                Ok(())
            }

            // IntMulStride: dst(gpr) = src(gpr) * stride
            VmInstr::IntMulStride { dst, src, stride } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        if *stride != 1 {
                            self.emit_line(&format!("mul.lo.s32 {d}, {s}, {stride};"));
                        } else {
                            self.emit_line(&format!("mov.s32 {d}, {s};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        if *stride != 1 {
                            self.emit_line(&format!("{d} = {s} * {stride};"));
                        } else {
                            self.emit_line(&format!("{d} = {s};"));
                        }
                    }
                    GpuDialect::Metal { .. } => {
                        if *stride != 1 {
                            self.emit_line(&format!("{d} = {s} * {stride};"));
                        } else {
                            self.emit_line(&format!("{d} = {s};"));
                        }
                    }
                }
                Ok(())
            }

            VmInstr::ScalarByteLoad { dst, base, offset } => {
                // 单字节加载: dst(gpr) = zero_extend(*(u8*)(base + offset))
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // PTX: 先算 64-bit 地址, 然后 ld.global.u8, 再 cvt.u32.u8 零扩展
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — u8 临时
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("ld.global.u8 {rs0}, [%rd_addr];"));
                        self.emit_line(&format!("cvt.u32.u8 {d}, {rs0};"));
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP (C++): 通过 unsigned char* 读取自然零扩展到 unsigned int
                        self.emit_line(&format!("{d} = (unsigned int)(*((unsigned char*){b} + {off}));"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = (unsigned int)(((device unsigned char*){b})[{off}]);"));
                    }
                }
                Ok(())
            }

            // AtomicAdd: elem_width 决定 u32 或 u64 原子加。
            VmInstr::AtomicAdd { base, ref offset, value, elem_width } => {
                let b = self.reg_name_with_kind(*base, alloc);
                let off_str = self.offset_to_string(offset, alloc);
                match (self.dialect, elem_width) {
                    (GpuDialect::Ptx { .. }, 4) => {
                        self.emit_line(&format!("atom.global.add.u32 [{b}+{off_str}], {value};"));
                    }
                    (GpuDialect::Ptx { .. }, 8) => {
                        self.emit_line(&format!("atom.global.add.u64 [{b}+{off_str}], {value};"));
                    }
                    (GpuDialect::Hip { .. }, 4) => {
                        self.emit_line(&format!("atomic_add(&({b}[{off_str}/4]), {value}u);"));
                    }
                    (GpuDialect::Hip { .. }, 8) => {
                        self.emit_line(&format!("atomic_add((unsigned long long*)({b}+{off_str}), {value}ull);"));
                    }
                    (GpuDialect::Metal { .. }, 4) => {
                        self.emit_line(&format!("atomic_fetch_add_explicit(&{b}[{off_str}/4], {value}, memory_order_relaxed);"));
                    }
                    (GpuDialect::Metal { .. }, 8) => {
                        self.emit_line(&format!(
                            "atomic_fetch_add_explicit((device atomic_ulong*)({b}+{off_str}), {value}UL, memory_order_relaxed);"
                        ));
                    }
                    _ => return Err(CompilerError::CodegenViolation(format!("AtomicAdd elem_width={elem_width} for dialect={:?}", self.dialect))),
                }
                Ok(())
            }

            // REQ-CG-014: GPU MemFence — memory barrier for Q-Tap callback boundaries.
            VmInstr::MemFence { order } => {
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // PTX: membar.gl for global scope (covers AcqRel/SeqCst),
                        // membar.cta for block scope (Release-only/Acquire-only).
                        match order {
                            MemFenceOrder::Release | MemFenceOrder::Acquire => {
                                self.emit_line("membar.cta;");
                            }
                            MemFenceOrder::AcqRel | MemFenceOrder::SeqCst => {
                                self.emit_line("membar.gl;");
                            }
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP: __threadfence() for global, __threadfence_block() for block.
                        match order {
                            MemFenceOrder::Release | MemFenceOrder::Acquire => {
                                self.emit_line("__threadfence_block();");
                            }
                            MemFenceOrder::AcqRel | MemFenceOrder::SeqCst => {
                                self.emit_line("__threadfence();");
                            }
                        }
                    }
                    GpuDialect::Metal { .. } => {
                        // Metal: threadgroup_barrier with appropriate memory scope.
                        match order {
                            MemFenceOrder::Release | MemFenceOrder::Acquire => {
                                self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
                            }
                            MemFenceOrder::AcqRel | MemFenceOrder::SeqCst => {
                                self.emit_line("threadgroup_barrier(mem_flags::mem_device);");
                            }
                        }
                    }
                }
                Ok(())
            }

            // REQ-CG-011: BreakLoop — exit generate loop, jump to epilogue
            VmInstr::BreakLoop { return_value } => {
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        match return_value {
                            ReturnValue::Const(val) => {
                                let rs0 = self.scratch_gpr_names[0];
                                self.emit_line(&format!("mov.u32 {rs0}, {val};"));
                                self.emit_line(&format!("st.global.u32 [%rd_output], {rs0};"));
                            }
                            ReturnValue::VReg(vreg) => {
                                let v = self.reg_name_with_kind(*vreg, alloc);
                                self.emit_line(&format!("st.global.u32 [%rd_output], {v};"));
                            }
                        }
                        self.emit_line(&format!("bra {};", self.epilogue_label));
                    }
                    _ => {
                        match return_value {
                            ReturnValue::Const(val) => {
                                self.emit_line(&format!("*((unsigned int*)rd_0) = {val}u;"));
                            }
                            ReturnValue::VReg(vreg) => {
                                let v = self.reg_name_with_kind(*vreg, alloc);
                                self.emit_line(&format!("*((unsigned int*)rd_0) = {v};"));
                            }
                        }
                        self.emit_line(&format!("goto {};", self.epilogue_label));
                    }
                }
                Ok(())
            }

            // REQ-CG-012: MarkLabel — emit label for JumpToLabel targets
            VmInstr::MarkLabel { label_id } => {
                let label = self.path_labels.get(label_id)
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{label_id}"));
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line(&format!("{label}:")),
                    _ => self.emit_line(&format!("{label}:;")),
                }
                Ok(())
            }

            // §3.7 ActivationSwap: GPU 不使用层循环栈帧 swap，直接 NOP
            VmInstr::ActivationSwap { .. } => Ok(()),

            // §0.2.8 WeightPrefetchAsync: 异步预取下一层权重到共享内存
            VmInstr::WeightPrefetchAsync { smem_name, weight_base, weight_offset, size } => {
                let base_reg = self.reg_name_with_kind(*weight_base, alloc);
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        // SM90+ Hopper/Blackwell: cp.async.bulk (TMA)
                        self.emit_line(&format!(
                            "// §SM90 TMA weight prefetch: {} bytes from [{} + {}] → {}",
                            size, base_reg, weight_offset, smem_name
                        ));
                        self.emit_line(&format!(
                            "cp.async.bulk.shared::cluster.global [{}], [{} + {}], {}, [mbar];",
                            smem_name, base_reg, weight_offset, size
                        ));
                        self.emit_line(&format!(
                            "mbarrier.arrive.expect_tx.shared::cta.b64 [mbar], {};", size
                        ));
                    }
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        // SM80-89: cp.async
                        self.emit_line(&format!(
                            "// §SM80 weight prefetch: {} bytes from [{} + {}] → {}",
                            size, base_reg, weight_offset, smem_name
                        ));
                        self.emit_line(&format!(
                            "cp.async.ca.shared.global [{}], [{} + {}], {};",
                            smem_name, base_reg, weight_offset, size
                        ));
                    }
                    GpuDialect::Hip { gfx_arch, .. } if gfx_arch >= 950 => {
                        // gfx950: GLOBAL_LOAD_LDS
                        self.emit_line(&format!(
                            "// gfx950 weight prefetch: {} bytes → {}", size, smem_name
                        ));
                        self.emit_line(&format!(
                            "global_load_lds {}, {} + {}, off;",
                            smem_name, base_reg, weight_offset
                        ));
                    }
                    _ => {
                        // No async prefetch support: synchronous load comment
                        self.emit_line(&format!(
                            "// weight prefetch (sync fallback): {} bytes → {}",
                            size, smem_name
                        ));
                    }
                }
                Ok(())
            }

            // §0.2.8 WeightPrefetchWait: 等待权重预取完成
            VmInstr::WeightPrefetchWait { group } => {
                match self.dialect {
                    GpuDialect::Ptx { sm_version } if sm_version >= 90 => {
                        self.emit_line("// §SM90 weight prefetch wait (mbarrier)");
                        let ps = self.scratch_pred_names[0];
                        self.emit_line(&format!(
                            "mbarrier.try_wait.parity.shared::cta.b64 {}, [mbar], 0;", ps
                        ));
                    }
                    GpuDialect::Ptx { sm_version } if sm_version >= 80 => {
                        self.emit_line(&format!("cp.async.wait_group {};", group));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("s_waitcnt lgkmcnt(0);");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
                    }
                    _ => {
                        self.emit_line("// weight prefetch wait (no-op)");
                    }
                }
                Ok(())
            }

            // PagedAttention: compute physical address from page table
            // PTX: shr.u32 + mad.wide.u32 + ld.global.u32 + mad.lo.u32
            // HIP: integer arithmetic + pointer offset
            VmInstr::PageTableAddr { dst, pool_base, page_table_ptr, ki_byte_off, row_bytes, page_size, page_stride, base_offset, seq_pt_offset } => {
                let dst_name = self.reg_name_with_kind(*dst, alloc);
                let pool_name = self.reg_name_with_kind(*pool_base, alloc);
                let pt_name = self.reg_name_with_kind(*page_table_ptr, alloc);
                let ki_str = self.offset_to_string(ki_byte_off, alloc);

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Step 1: token_idx = ki_byte_off / row_bytes
                        let log2_row = (*row_bytes as f64).log2() as u32;
                        let log2_page = (*page_size as f64).log2() as u32;

                        // token_idx = ki_byte_off >> log2(row_bytes)
                        self.emit_line(&format!("shr.u32 {}, {}, {};", dst_name, ki_str, log2_row));
                        // page_idx = token_idx >> log2(page_size)
                        self.emit_line(&format!("shr.u32 {}, {}, {};", dst_name, dst_name, log2_page));
                        // §20 BCI-005: add per-sequence page_table offset (u32 entries)
                        if let Some(pt_off) = seq_pt_offset {
                            let pt_off_name = self.reg_name_with_kind(*pt_off, alloc);
                            self.emit_line(&format!("add.u32 {}, {}, {};", dst_name, dst_name, pt_off_name));
                        }
                        // Load u32 page_id: byte_off = page_idx * 4
                        // mad.wide.u32 %r_tmp, page_idx, 4, %pt_ptr → then ld.global.u32
                        let tmp = self.reg_name_with_kind(VRegId(30), alloc); // use high vreg as temp
                        self.emit_line(&format!("mad.wide.u32 {}, {}, 4, {};", tmp, dst_name, pt_name));
                        self.emit_line(&format!("ld.global.u32 {}, [{}];", dst_name, tmp));
                        // Zero-extend page_id to 64-bit for address math
                        // addr = pool_base + page_id * page_stride + token_in_page * row_bytes + base_offset
                        self.emit_line(&format!("mad.lo.u32 {}, {}, {}, {};", dst_name, dst_name, *page_stride, pool_name));
                        // Re-compute token_in_page
                        self.emit_line(&format!("shr.u32 {}, {}, {};", tmp, ki_str, log2_row));
                        let mask = (*page_size - 1) as u32;
                        self.emit_line(&format!("and.b32 {}, {}, {};", tmp, tmp, mask));
                        self.emit_line(&format!("mad.lo.u32 {}, {}, {}, {};", tmp, tmp, *row_bytes, dst_name));
                        if *base_offset > 0 {
                            self.emit_line(&format!("add.u32 {}, {}, {};", dst_name, tmp, *base_offset));
                        } else {
                            self.emit_line(&format!("mov.u32 {}, {};", dst_name, tmp));
                        }
                    }
                    GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                        // HIP/Metal: C++ style integer arithmetic
                        let log2_row = (*row_bytes as f64).log2() as u32;
                        let log2_page = (*page_size as f64).log2() as u32;
                        let mask = *page_size - 1 ;

                        // token_idx = ki_byte_off >> log2(row_bytes)
                        self.emit_line(&format!("{{ uint32_t _tki = {} >> {};", ki_str, log2_row));
                        // page_idx + seq_pt_offset (BCI-005)
                        if let Some(pt_off) = seq_pt_offset {
                            let pt_off_name = self.reg_name_with_kind(*pt_off, alloc);
                            self.emit_line(&format!("  uint32_t _ppid = ((uint32_t*){})[(_tki >> {}) + {}];", pt_name, log2_page, pt_off_name));
                        } else {
                            self.emit_line(&format!("  uint32_t _ppid = ((uint32_t*){})[_tki >> {}];", pt_name, log2_page));
                        }
                        // addr = pool_base + ppid * page_stride + (_tki & mask) * row_bytes + base_offset
                        self.emit_line(&format!("  {} = (uint32_t)((uintptr_t){} + (size_t)_ppid * {} + ((_tki & {}) * {}){});", dst_name, pool_name, *page_stride, mask, *row_bytes, if *base_offset > 0 { format!(" + {}", *base_offset) } else { String::new() }));
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            VmInstr::PageTableKVWrite { src, pool_base, page_table_ptr, seq_index, row_bytes, page_size, page_stride, base_offset, width: _, dtype } => {
                let src_name = self.reg_name_with_kind(*src, alloc);
                let pool_name = self.reg_name_with_kind(*pool_base, alloc);
                let pt_name = self.reg_name_with_kind(*page_table_ptr, alloc);

                let page_idx = seq_index.0 as usize / page_size;
                let token_in_page = seq_index.0 as usize % page_size;
                let pt_byte_offset = page_idx * 4;
                let tip_offset = token_in_page * row_bytes + base_offset;

                let dt = dtype.gpu_type_name().map_err(|_| CompilerError::CodegenViolation(format!("PageTableKVWrite: unsupported dtype {:?} for GPU lowering", dtype)))?;

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load u32 page_id from page_table
                        self.emit_line(&format!("ld.global.u32 {}, [{} + {}];", src_name, pt_name, pt_byte_offset));
                        // addr = pool_base + page_id * page_stride + tip_offset
                        self.emit_line(&format!("mad.lo.u32 {}, {}, {}, {};", src_name, src_name, *page_stride, pool_name));
                        if tip_offset > 0 {
                            self.emit_line(&format!("add.u32 {}, {}, {};", src_name, src_name, tip_offset));
                        }
                    }
                    GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{{ uint32_t _ppid = ((uint32_t*){})[{}];", pt_name, page_idx));
                        self.emit_line(&format!("  auto _addr = ({}*){} + (size_t)_ppid * {} + {};", dt, pool_name, *page_stride, tip_offset));
                        self.emit_line(&format!("  *_addr = {};", src_name));
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // PagedAttention + KIVI quantized KV write-back
            VmInstr::PageTableKVWriteQuant { src, pool_base, page_table_ptr, seq_index, quant_row_bytes, fp32_row_bytes: _, page_size, page_stride, base_offset, scale_offset, width: _, kivi_mode, num_elems } => {
                let src_name = self.reg_name_with_kind(*src, alloc);
                let pool_name = self.reg_name_with_kind(*pool_base, alloc);
                let pt_name = self.reg_name_with_kind(*page_table_ptr, alloc);

                let page_idx = seq_index.0 as usize / page_size;
                let token_in_page = seq_index.0 as usize % page_size;
                let pt_byte_offset = page_idx * 4;
                let tip_offset = token_in_page * quant_row_bytes + base_offset;

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — packed_data_addr
                        let rs1 = self.scratch_gpr_names[1]; // %rs1 — temp
                        let rs2 = self.scratch_gpr_names[2]; // %rs_bound — temp

                        // Load u32 page_id → rs0
                        self.emit_line(&format!("ld.global.u32 {}, [{} + {}];", rs0, pt_name, pt_byte_offset));
                        // packed_addr = pool_base + page_id * page_stride + tip_offset → rs0
                        self.emit_line(&format!("mad.lo.u32 {}, {}, {}, {};", rs0, rs0, *page_stride, pool_name));
                        if tip_offset > 0 {
                            self.emit_line(&format!("add.u32 {}, {}, {};", rs0, rs0, tip_offset));
                        }
                        // rs0 = packed_data_addr

                        match kivi_mode {
                            KvLoadMode::Kivi4 => {
                                // Per-channel: each pair gets its own scale
                                let num_pairs = (*num_elems + 1) / 2;
                                for pair in 0..num_pairs {
                                    let lo_off = pair * 2;
                                    let hi_off = pair * 2 + 1;
                                    // scale = max(|src[lo]|, |src[hi]|)
                                    self.emit_line(&format!("ld.global.f32 {rs1}, [{src_name}+{}];", lo_off * 4));
                                    self.emit_line(&format!("abs.f32 {rs1}, {rs1};"));
                                    if hi_off < *num_elems {
                                        self.emit_line(&format!("ld.global.f32 {rs2}, [{src_name}+{}];", hi_off * 4));
                                        self.emit_line(&format!("abs.f32 {rs2}, {rs2};"));
                                        self.emit_line(&format!("max.f32 {rs1}, {rs1}, {rs2};"));
                                    }
                                    // store scale at rs0 + scale_offset + pair*4
                                    let scale_byte_off = *scale_offset + pair * 4;
                                    self.emit_line(&format!("st.global.f32 [{rs0}+{}], {rs1};", scale_byte_off));
                                    // Reload lo for quantization
                                    self.emit_line(&format!("ld.global.f32 {rs1}, [{src_name}+{}];", lo_off * 4));
                                    self.emit_line(&format!("mov.b32 {rs1}, {rs1};"));
                                    self.emit_line(&format!("shr.b32 {rs1}, {rs1}, 20;"));
                                    self.emit_line(&format!("and.b32 {rs1}, {rs1}, 0xF;"));
                                    if hi_off < *num_elems {
                                        self.emit_line(&format!("ld.global.f32 {rs2}, [{src_name}+{}];", hi_off * 4));
                                        self.emit_line(&format!("mov.b32 {rs2}, {rs2};"));
                                        self.emit_line(&format!("shr.b32 {rs2}, {rs2}, 20;"));
                                        self.emit_line(&format!("and.b32 {rs2}, {rs2}, 0xF;"));
                                        self.emit_line(&format!("shl.b32 {rs2}, {rs2}, 4;"));
                                        self.emit_line(&format!("or.b32 {rs1}, {rs1}, {rs2};"));
                                    }
                                    self.emit_line(&format!("st.global.u8 [{rs0}+{pair}], {rs1};"));
                                }
                            }
                            KvLoadMode::Kivi2 => {
                                // Per-token: single scale for all elements
                                // Pass 1: find max
                                self.emit_line(&format!("mov.f32 {rs1}, 0F00000000;")); // 0.0f
                                for i in 0..*num_elems {
                                    self.emit_line(&format!("ld.global.f32 {rs2}, [{src_name}+{}];", i * 4));
                                    self.emit_line(&format!("abs.f32 {rs2}, {rs2};"));
                                    self.emit_line(&format!("max.f32 {rs1}, {rs1}, {rs2};"));
                                }
                                // store scale
                                self.emit_line(&format!("st.global.f32 [{rs0}+{}], {rs1};", *scale_offset));
                                // Pass 2: pack nibbles
                                let num_pairs = (*num_elems + 1) / 2;
                                for pair in 0..num_pairs {
                                    let lo_off = pair * 2;
                                    let hi_off = pair * 2 + 1;
                                    self.emit_line(&format!("ld.global.f32 {rs1}, [{src_name}+{}];", lo_off * 4));
                                    self.emit_line(&format!("mov.b32 {rs1}, {rs1};"));
                                    self.emit_line(&format!("shr.b32 {rs1}, {rs1}, 20;"));
                                    self.emit_line(&format!("and.b32 {rs1}, {rs1}, 0xF;"));
                                    if hi_off < *num_elems {
                                        self.emit_line(&format!("ld.global.f32 {rs2}, [{src_name}+{}];", hi_off * 4));
                                        self.emit_line(&format!("mov.b32 {rs2}, {rs2};"));
                                        self.emit_line(&format!("shr.b32 {rs2}, {rs2}, 20;"));
                                        self.emit_line(&format!("and.b32 {rs2}, {rs2}, 0xF;"));
                                        self.emit_line(&format!("shl.b32 {rs2}, {rs2}, 4;"));
                                        self.emit_line(&format!("or.b32 {rs1}, {rs1}, {rs2};"));
                                    }
                                    self.emit_line(&format!("st.global.u8 [{rs0}+{pair}], {rs1};"));
                                }
                            }
                            _ => {
                                return Err(CompilerError::CodegenViolation(
                                    format!("PageTableKVWriteQuant: invalid kivi_mode {:?}", kivi_mode)
                                ));
                            }
                        }
                        Ok(())
                    }
                    GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  uint32_t _ppid = ((uint32_t*){})[{}];", pt_name, page_idx));
                        self.emit_line(&format!("  auto _packed_addr = (uint8_t*){} + (size_t)_ppid * {} + {};", pool_name, *page_stride, tip_offset));
                        self.emit_line(&format!("  auto _scale_addr = (float*)(_packed_addr + {});", *scale_offset));
                        match kivi_mode {
                            KvLoadMode::Kivi4 => {
                                let num_pairs = (*num_elems + 1) / 2;
                                for pair in 0..num_pairs {
                                    let lo_off = pair * 2;
                                    let hi_off = pair * 2 + 1;
                                    self.emit_line(&format!("  {{ float _lo = ((float*){})[{}];", src_name, lo_off));
                                    self.emit_line(&format!("    float _hi = ((float*){})[{}];", src_name, hi_off));
                                    self.emit_line("    float _sc = fmax(fabs(_lo), fabs(_hi));");
                                    self.emit_line(&format!("    _scale_addr[{}] = _sc;", pair));
                                    self.emit_line("    uint32_t _nlo = ((*((uint32_t*)&_lo)) >> 20) & 0xF;");
                                    self.emit_line("    uint32_t _nhi = ((*((uint32_t*)&_hi)) >> 20) & 0xF;");
                                    self.emit_line(&format!("    _packed_addr[{}] = (uint8_t)((_nhi << 4) | _nlo); }}", pair));
                                }
                            }
                            KvLoadMode::Kivi2 => {
                                self.emit_line("  float _maxv = 0.0f;");
                                self.emit_line(&format!("  for (int _i = 0; _i < {}; _i++) {{", num_elems));
                                self.emit_line(&format!("    float _v = fabs(((float*){})[_i]);", src_name));
                                self.emit_line("    _maxv = fmax(_maxv, _v);");
                                self.emit_line("  }");
                                self.emit_line("  _scale_addr[0] = _maxv;");
                                let num_pairs = (*num_elems + 1) / 2;
                                for pair in 0..num_pairs {
                                    let lo_off = pair * 2;
                                    let hi_off = pair * 2 + 1;
                                    self.emit_line(&format!("  {{ uint32_t _nlo = ((*((uint32_t*)&((float*){})[{}])) >> 20) & 0xF;", src_name, lo_off));
                                    self.emit_line(&format!("    uint32_t _nhi = ((*((uint32_t*)&((float*){})[{}])) >> 20) & 0xF;", src_name, hi_off));
                                    self.emit_line(&format!("    _packed_addr[{}] = (uint8_t)((_nhi << 4) | _nlo); }}", pair));
                                }
                            }
                            _ => {
                                return Err(CompilerError::CodegenViolation(
                                    format!("PageTableKVWriteQuant: invalid kivi_mode {:?}", kivi_mode)
                                ));
                            }
                        }
                        self.emit_line("}");
                        Ok(())
                    }
                }
            }

            VmInstr::KiviQuantChannel { src, dst_ptr, scale_ptr, num_channels, width: _ } => {
                // KIVI per-channel 4-bit quantization: f32 pairs → packed nibble + scale.
                self.emit_kivi_quant(src, dst_ptr, scale_ptr, *num_channels, alloc, "channel")
            }
            VmInstr::KiviQuantToken { src, dst_ptr, scale_ptr, num_elems, width: _ } => {
                // KIVI per-token 4-bit quantization: identical to per-channel but different semantic.
                self.emit_kivi_quant(src, dst_ptr, scale_ptr, *num_elems, alloc, "token")
            }
            VmInstr::KiviDequantLoad { dst, src_ptr, scale_ptr, num_elems, width: _ } => {
                // KIVI 4-bit dequant: packed byte → 2 × (nibble - 8) × scale → f32
                let d = self.reg_name_with_kind(*dst, alloc);
                let sp = self.reg_name_with_kind(*src_ptr, alloc);
                let sptr = self.reg_name_with_kind(*scale_ptr, alloc);
                let rs0 = self.scratch_gpr_names[0];
                let rs1 = self.scratch_gpr_names[1];
                let num_pairs = (*num_elems + 1) / 2;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        for pair in 0..num_pairs {
                            // Load packed byte
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{sp}+{pair}];"));
                            // Unpack low nibble
                            self.emit_line(&format!("mov.u32 {rs1}, {rs0};"));
                            self.emit_line(&format!("and.b32 {rs1}, {rs1}, 0xF;"));
                            // (nibble - 8) → f32 → × scale
                            self.emit_line(&format!("sub.s32 {rs1}, {rs1}, 8;"));
                            self.emit_line(&format!("cvt.rn.f32.s32 {rs0}, {rs1};"));
                            self.emit_line(&format!("ld.global.f32 {rs1}, [{sptr}+{}];", pair * 4));
                            self.emit_line(&format!("mul.rn.f32 {d}, {rs0}, {rs1};"));
                            if pair * 2 + 1 < *num_elems {
                                // High nibble
                                self.emit_line(&format!("ld.global.u8 {rs0}, [{sp}+{pair}];"));
                                self.emit_line(&format!("shr.b32 {rs0}, {rs0}, 4;"));
                                self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0xF;"));
                                self.emit_line(&format!("sub.s32 {rs0}, {rs0}, 8;"));
                                self.emit_line(&format!("cvt.rn.f32.s32 {rs1}, {rs0};"));
                                self.emit_line(&format!("ld.global.f32 {rs0}, [{sptr}+{}];", pair * 4));
                                self.emit_line(&format!("mul.rn.f32 {d}, {rs1}, {rs0};"));
                            }
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned char* _sp = (unsigned char*)({sp});"));
                        self.emit_line(&format!("  float* _scp = (float*)({sptr});"));
                        for pair in 0..num_pairs {
                            self.emit_line("  {");
                            self.emit_line(&format!("    unsigned char _byte = _sp[{pair}];"));
                            self.emit_line("    int _lo = (_byte & 0xF) - 8;");
                            self.emit_line(&format!("    {d} = (float)_lo * _scp[{pair}];"));
                            if pair * 2 + 1 < *num_elems {
                                self.emit_line("    int _hi = ((_byte >> 4) & 0xF) - 8;");
                                self.emit_line(&format!("    {d} = (float)_hi * _scp[{pair}];"));
                            }
                            self.emit_line("  }");
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  device unsigned char* _sp = (device unsigned char*)({sp});"));
                        self.emit_line(&format!("  device float* _scp = (device float*)({sptr});"));
                        for pair in 0..num_pairs {
                            self.emit_line("  {");
                            self.emit_line(&format!("    unsigned char _byte = _sp[{pair}];"));
                            self.emit_line("    int _lo = (_byte & 0xF) - 8;");
                            self.emit_line(&format!("    {d} = (float)_lo * _scp[{pair}];"));
                            if pair * 2 + 1 < *num_elems {
                                self.emit_line("    int _hi = ((_byte >> 4) & 0xF) - 8;");
                                self.emit_line(&format!("    {d} = (float)_hi * _scp[{pair}];"));
                            }
                            self.emit_line("  }");
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // SharedMemSwizzle: XOR swizzle 消除 shared memory bank conflict
            VmInstr::SharedMemSwizzle { dst, raw_addr, log2_banks, log2_bank_width } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*raw_addr, alloc);
                let shift_amount = (*log2_banks as u32) + (*log2_bank_width as u32);
                let mask = (1u32 << (*log2_banks as u32)) - 1;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // XOR swizzle: swizzled = raw ^ ((raw >> shift) & mask)
                        let tmp = self.scratch_gpr_names[0];
                        self.emit_line(&format!("shr.u32 {tmp}, {s}, {shift_amount};"));
                        self.emit_line(&format!("and.b32 {tmp}, {tmp}, {mask};"));
                        self.emit_line(&format!("xor.b32 {d}, {s}, {tmp};"));
                    }
                    GpuDialect::Hip { .. } => {
                        // AMD GPU: same XOR swizzle logic
                        self.emit_line("{");
                        self.emit_line(&format!("  uint tmp = ({s}) >> {shift_amount};"));
                        self.emit_line(&format!("  tmp &= {mask};"));
                        self.emit_line(&format!("  {d} = ({s}) ^ tmp;"));
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        // Metal: same XOR swizzle logic
                        self.emit_line("{");
                        self.emit_line(&format!("  uint tmp = ({s}) >> {shift_amount};"));
                        self.emit_line(&format!("  tmp &= {mask};"));
                        self.emit_line(&format!("  {d} = ({s}) ^ tmp;"));
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // ═══ 元操作 ═══
            VmInstr::DeclareVReg { .. } | VmInstr::ReleaseVReg { .. } | VmInstr::Comment(_) => Ok(()),
            VmInstr::VecScalarStore { base, src, offset } => {
                // Store single f32 from Vec register lane 0 to *(f32*)(base + offset).
                let s = self.reg_name_with_kind(*src, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Compute address: base is u64, off may be u32.
                        // Use %rd_addr scratch to hold the 64-bit address.
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {s};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("*((float*)({b}+({off}))) = {s};"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("*((device float*)({b}+({off}))) = {s};"));
                    }
                }
                Ok(())
            }

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // SPEC 23-QUANT-CODEGEN-ALGO §3: Quant* decode GPU/PTX/HIP/MSL lowering
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            VmInstr::QuantBroadcastInt { dst, value, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => self.emit_line(&format!("mov.u32 {d}, {};", *value as u32)),
                    _ => self.emit_line(&format!("{d} = {}u;", *value as u32)),
                }
                Ok(())
            }

            VmInstr::QuantScalarCvtLoad { dst, base, offset, src_dtype, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = *offset;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        match src_dtype {
                            ScalarCvtSource::F16 => {
                                self.emit_line(&format!("  {d} = __half2float(((device half*)({b}))[{}]));", off));
                            }
                            ScalarCvtSource::I8 => {
                                self.emit_line(&format!("  {d} = (float)(((device signed char*)({b}))[{}]);", off));
                            }
                            ScalarCvtSource::U8 => {
                                self.emit_line(&format!("  {d} = (float)(((device unsigned char*)({b}))[{}]);", off));
                            }
                        }
                    }
                    _ => {
                        match src_dtype {
                            ScalarCvtSource::F16 => {
                                self.emit_line(&format!("  {d} = float_from_half(((device half*)({b}))[{}]));", off));
                            }
                            ScalarCvtSource::I8 => {
                                self.emit_line(&format!("  {d} = float((({b})[{}]));", off));
                            }
                            ScalarCvtSource::U8 => {
                                self.emit_line(&format!("  {d} = float(((device uchar*)({b}))[{}]));", off));
                            }
                        }
                    }
                }
                Ok(())
            }

            VmInstr::QuantLoadBytesVec { dst, base, offset, count, .. } => {
                // Load `count` bytes from base+offset, zero-extend each to i32/f32 lane.
                // GPU: use vector load + type punning.
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let off = *offset;
                let cnt = *count;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line("{");
                        for i in 0..cnt {
                            self.emit_line(&format!(".reg .u8 %b{i}; .reg .u32 %w{i}; ld.global.u8 %b{i}, [{b}+{off}+{i}]; cvt.u32.u8 %w{i}, %b{i};"));
                        }
                        // Pack into vector register
                        self.emit_line(".reg .v4.u32 %v4;");
                        for i in 0..cnt.min(4) {
                            self.emit_line(&format!("mov.v4.u32.s{}, %w{};", i, i));
                        }
                        self.emit_line(&format!("cvt.rn.f32.u32 {d}, %w0;")); // Simplified
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{{ uint4 _v; for(int _i=0; _i<{cnt}; _i++) {{ _v.x = (unsigned)(*((unsigned char*)({b}+{off}+_i))); }} {d} = make_float4((float)_v.x, (float)_v.y, (float)_v.z, (float)_v.w); }}"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{{ float4 _v(0); for(int _i=0; _i<{cnt}; _i++) {{ _v.x = float(*((device uchar*)({b}+{off}+_i))); }} {d} = _v; }}"));
                    }
                }
                Ok(())
            }

            VmInstr::QuantCodebookLookup { dst, indices, codebook_data, bits_per_entry, .. } => {
                // GPU: embed codebook in .const section or use inline immediate table.
                // Each thread processes one element: dst = (float)codebook[indices & mask]
                let d = self.reg_name_with_kind(*dst, alloc);
                let idx = self.reg_name_with_kind(*indices, alloc);
                let mask = ((1u32 << bits_per_entry) - 1) as i32;
                // Generate inline PTX with embedded codebook entries via .const array.
                let cb_len = codebook_data.len();
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Build inline lookup via PTX branch-less array (register-file table).
                        // For small codebooks (≤16), use mov + setp + selp chain.
                        self.emit_line(&format!("{{ .reg .s32 %cb_idx; and.b32 %cb_idx, {idx}, {mask};"));
                        for (i, &val) in codebook_data.iter().enumerate().take(cb_len.min(256)) {
                            self.emit_line(&format!("  {{ .reg .pred %cb_p{i}; setp.eq.s32 %cb_p{i}, %cb_idx, {i}; @%cb_p{i} cvt.rn.f32.s32 {d}, {val}; }}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP: use device-side array
                        let entries: Vec<String> = codebook_data.iter().map(|v| v.to_string()).collect();
                        let table = entries.join(", ");
                        self.emit_line(&format!("{{ const int8_t cb[] = {{{table}}}; {d} = (float)cb[{idx} & {mask}]; }}"));
                    }
                    GpuDialect::Metal { .. } => {
                        let entries: Vec<String> = codebook_data.iter().map(|v| v.to_string()).collect();
                        let table = entries.join(", ");
                        self.emit_line(&format!("{{ constant int8_t cb[] = {{{table}}}; {d} = (float)cb[{idx} & {mask}]; }}"));
                    }
                }
                Ok(())
            }

            VmInstr::QuantExtractBits { dst, src, bit_offset, bit_width, .. } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                let mask = ((1u32 << bit_width) - 1) as i32;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("shr.u32 {d}, {s}, {bit_offset};"));
                        self.emit_line(&format!("and.b32 {d}, {d}, {mask};"));
                    }
                    _ => self.emit_line(&format!("{d} = ({s} >> {bit_offset}) & {mask};")),
                }
                Ok(())
            }

            VmInstr::QuantDequantFma { dst, weight, activation, scale, zero_point, quant_kind, dtype: _, width: _ } => {
                use crate::quant_format::QuantDataKind as DK;
                let d = self.reg_name_with_kind(*dst, alloc);
                let w = self.reg_name_with_kind(*weight, alloc);
                let act = self.reg_name_with_kind(*activation, alloc);
                let sc = self.reg_name_with_kind(*scale, alloc);
                let zp = self.reg_name_with_kind(*zero_point, alloc);
                let sm = self.sm_version().unwrap_or(0);

                match quant_kind {
                    // PackedInt4 (AWQ/GPTQ): AND+shift unpack 4-bit -> sub zp -> mul scale -> FMA
                    DK::PackedInt4 | DK::SignedPackedInt4 => {
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line("{");
                                self.emit_line("  .reg .f32 %qdf_lo, %qdf_hi;");
                                self.emit_line(&format!("  and.b32 %qdf_lo, {w}, 0xF;"));
                                self.emit_line(&format!("  shr.b32 %qdf_hi, {w}, 4;"));
                                self.emit_line("  and.b32 %qdf_hi, %qdf_hi, 0xF;");
                                self.emit_line("  cvt.rn.f32.s32 %qdf_lo, %qdf_lo;");
                                self.emit_line("  cvt.rn.f32.s32 %qdf_hi, %qdf_hi;");
                                self.emit_line(&format!("  sub.f32 %qdf_lo, %qdf_lo, {zp};"));
                                self.emit_line(&format!("  sub.f32 %qdf_hi, %qdf_hi, {zp};"));
                                self.emit_line(&format!("  mul.f32 %qdf_lo, %qdf_lo, {sc};"));
                                self.emit_line(&format!("  mul.f32 %qdf_hi, %qdf_hi, {sc};"));
                                self.emit_line(&format!("  fma.rn.f32 %qdf_lo, %qdf_lo, {act}, {d};"));
                                self.emit_line(&format!("  fma.rn.f32 {d}, %qdf_hi, {act}, %qdf_lo;"));
                                self.emit_line("}");
                            }
                            _ => {
                                self.emit_line("{");
                                self.emit_line(&format!("  float qdf_lo = (float)((int)({w} & 0xF));"));
                                self.emit_line(&format!("  float qdf_hi = (float)((int)(({w} >> 4) & 0xF));"));
                                self.emit_line(&format!("  float qdf_dlo = (qdf_lo - {zp}) * {sc};"));
                                self.emit_line(&format!("  float qdf_dhi = (qdf_hi - {zp}) * {sc};"));
                                self.emit_line(&format!("  {d} += qdf_dlo * {act} + qdf_dhi * {act};"));
                                self.emit_line("}");
                            }
                        }
                        Ok(())
                    }

                    // MXFP4 / Float4: E2M1 data + E8M0 scale -> f32 -> FMA
                    DK::Float4 => {
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line("{");
                                self.emit_line("  .reg .f32 %qdf_lo, %qdf_hi;");
                                self.emit_line("  .reg .s32 %qdf_qs_lo, %qdf_qs_hi;");
                                self.emit_line(&format!("  and.b32 %qdf_qs_lo, {w}, 0xF;"));
                                self.emit_line(&format!("  shr.b32 %qdf_qs_hi, {w}, 4;"));
                                self.emit_line("  and.b32 %qdf_qs_hi, %qdf_qs_hi, 0xF;");
                                self.emit_line("  cvt.rn.f32.s32 %qdf_lo, %qdf_qs_lo;");
                                self.emit_line("  cvt.rn.f32.s32 %qdf_hi, %qdf_qs_hi;");
                                // scale already contains E8M0 decoded f32 value
                                self.emit_line(&format!("  mul.f32 %qdf_lo, %qdf_lo, {sc};"));
                                self.emit_line(&format!("  mul.f32 %qdf_hi, %qdf_hi, {sc};"));
                                self.emit_line(&format!("  fma.rn.f32 %qdf_lo, %qdf_lo, {act}, {d};"));
                                self.emit_line(&format!("  fma.rn.f32 {d}, %qdf_hi, {act}, %qdf_lo;"));
                                self.emit_line("}");
                            }
                            _ => {
                                self.emit_line("{");
                                self.emit_line(&format!("  int qdf_qs_lo = (int)({w} & 0xF);"));
                                self.emit_line(&format!("  int qdf_qs_hi = (int)(({w} >> 4) & 0xF);"));
                                self.emit_line(&format!("  float qdf_v_lo = (float)qdf_qs_lo * {sc};"));
                                self.emit_line(&format!("  float qdf_v_hi = (float)qdf_qs_hi * {sc};"));
                                self.emit_line(&format!("  {d} += qdf_v_lo * {act} + qdf_v_hi * {act};"));
                                self.emit_line("}");
                            }
                        }
                        Ok(())
                    }

                    // Int8: hardware-native dot product path
                    DK::Int8 => {
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                // Scalar int8 -> f32 conversion + scale + FMA
                                self.emit_line(&format!("cvt.rn.f32.s32 {d}, {w};"));
                                self.emit_line(&format!("mul.f32 {d}, {d}, {sc};"));
                                self.emit_line(&format!("fma.rn.f32 {d}, {d}, {act}, {d};"));
                            }
                            _ => {
                                self.emit_line(&format!("{d} += (float)(int)({w}) * {sc} * {act};"));
                            }
                        }
                        Ok(())
                    }

                    // Nvfp4: tcgen05 hardware native (SM100+ only)
                    DK::Nvfp4 => {
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                if sm >= 100 {
                                    self.emit_line("{");
                                    self.emit_line("  .reg .f32 %qdf_e2m1_v;");
                                    self.emit_line("  .reg .s32 %qdf_idx;");
                                    self.emit_line(&format!("  and.b32 %qdf_idx, {w}, 0xF;"));
                                    self.emit_line("  cvt.rn.f32.s32 %qdf_e2m1_v, %qdf_idx;");
                                    self.emit_line(&format!("  mul.f32 %qdf_e2m1_v, %qdf_e2m1_v, {sc};"));
                                    self.emit_line(&format!("  fma.rn.f32 {d}, %qdf_e2m1_v, {act}, {d};"));
                                    self.emit_line("}");
                                } else {
                                    return Err(CompilerError::CodegenViolation(
                                        "QuantDequantFma Nvfp4: requires SM100+ for tcgen05 FP4 native".into()));
                                }
                            }
                            _ => {
                                return Err(CompilerError::CodegenViolation(
                                    "QuantDequantFma Nvfp4: only supported on PTX SM100+".into()));
                            }
                        }
                        Ok(())
                    }

                    // Native float formats: no dequant needed, direct FMA
                    DK::Bfloat16 | DK::Float16 | DK::Float32 => {
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                self.emit_line(&format!("fma.rn.f32 {d}, {w}, {act}, {d};"));
                            }
                            _ => {
                                self.emit_line(&format!("{d} += {w} * {act};"));
                            }
                        }
                        Ok(())
                    }

                    // Other formats: not yet supported in register-inline path
                    _ => Err(CompilerError::CodegenViolation(format!(
                        "QuantDequantFma: quant_kind {:?} not yet supported for GPU register-inline dequant+FMA",
                        quant_kind
                    ))),
                }
            }

            VmInstr::QuantInterleave { dst, lo, hi, .. } => {
                // GPU is scalar per-thread; interleave is handled by thread indexing.
                // Emit a no-op move (the caller structures thread indices to interleave).
                let d = self.reg_name_with_kind(*dst, alloc);
                let vlo = self.reg_name_with_kind(*lo, alloc);
                let vhi = self.reg_name_with_kind(*hi, alloc);
                // Use prmt for byte-level interleave (PTX) or conditional select.
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Thread even lanes → lo, odd lanes → hi: use threadIdx.x & 1
                        self.emit_line(&format!("{{ .reg .pred %intlv_p; .reg .u32 %intlv_tid; mov.u32 %intlv_tid, %tid.x; and.b32 %intlv_tid, %intlv_tid, 1; setp.eq.u32 %intlv_p, %intlv_tid, 1; selp.f32 {d}, {vhi}, {vlo}, %intlv_p; }}"));
                    }
                    _ => self.emit_line(&format!("{d} = (threadIdx.x & 1) ? {vhi} : {vlo};")),
                }
                Ok(())
            }

            VmInstr::QuantConcatSeq { dst, lo, hi, .. } => {
                // GPU scalar per-thread: sequential concat maps to thread index < half → lo, else → hi.
                let d = self.reg_name_with_kind(*dst, alloc);
                let vlo = self.reg_name_with_kind(*lo, alloc);
                let vhi = self.reg_name_with_kind(*hi, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("{{ .reg .pred %ccat_p; .reg .u32 %ccat_tid; mov.u32 %ccat_tid, %tid.x; setp.lt.u32 %ccat_p, %ccat_tid, 4; selp.f32 {d}, {vlo}, {vhi}, %ccat_p; }}"));
                    }
                    _ => self.emit_line(&format!("{d} = (threadIdx.x < 4) ? {vlo} : {vhi};")),
                }
                Ok(())
            }

            VmInstr::Q3KDecodeStep { .. } => {
                Err(CompilerError::CodegenViolation(
                    "Q3KDecodeStep GPU: not yet implemented".into()
                ))
            }

            VmInstr::ScopeBegin { .. } | VmInstr::ScopeEnd { .. } => Ok(()),

            // ── SPEC 23-QUANT-CODEGEN-ALGO §4.3 / REQ-VR-002: Unified Dot-Product VmInstr (GPU lowering) ──

            VmInstr::DotProduct { acc, a, b, input_dtype, width } => {
                let _ = width;
                let vc = self.reg_name_with_kind(*acc, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let vb = self.reg_name_with_kind(*b, alloc);
                match input_dtype {
                    DotDtype::Bf16 => {
                        // BF16 dot-product: fp32_acc += bf16_a · bf16_b (per-element).
                        // PTX: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (SM80+)
                        //      wgmma.mma_async (SM90+), tcgen05 (SM100+)
                        //      For SIMT scalar path: cvt bf16→f32 then fma.rn.f32
                        // HIP: v_mfma_f32_16x16x16_bf16 (CDNA) or scalar __half2float + fma
                        // Metal: simdgroup_matrix_multiply or scalar fma
                        match self.dialect {
                            GpuDialect::Ptx { sm_version } => {
                                if sm_version >= 100 {
                                    self.emit_line("// §SM100 DotProduct<Bf16> tcgen05");
                                    self.emit_line(&format!("tcgen05.mma.cta_group::1.kind::bf16 [%tmem_addr], {va}, {vb}, 0x0, 0, 1;"));
                                    self.emit_line("tcgen05.wait::ld.sync.aligned;");
                                } else if sm_version >= 90 {
                                    self.emit_line("wgmma.fence.sync.aligned;");
                                    self.emit_line(&format!("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {{{vc}}}, {va}, {vb};"));
                                    self.emit_line("wgmma.commit_group.sync.aligned;");
                                    self.emit_line("wgmma.wait_group.sync.aligned 0;");
                                } else if sm_version >= 80 {
                                    self.emit_line(&format!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{vc}}}, {{{va}}}, {{{vb}}}, {{{vc}}};"));
                                } else {
                                    let fs0 = self.scratch_vec_names[0];
                                    let fs1 = self.scratch_vec_names[1];
                                    self.emit_line(&format!("cvt.rn.f32.bf16 {fs0}, {va};"));
                                    self.emit_line(&format!("cvt.rn.f32.bf16 {fs1}, {vb};"));
                                    self.emit_line(&format!("fma.rn.f32 {vc}, {fs0}, {fs1}, {vc};"));
                                }
                            }
                            GpuDialect::Hip { gfx_arch, .. } => {
                                if gfx_arch >= 950 {
                                    self.emit_line(&format!("v_mfma_f32_32x32x16_bf16 {vc}, {va}, {vb}, {vc};"));
                                } else if gfx_arch >= 908 {
                                    self.emit_line(&format!("v_mfma_f32_16x16x16_bf16 {vc}, {va}, {vb}, {vc};"));
                                } else {
                                    let fs0 = self.scratch_vec_names[0];
                                    let fs1 = self.scratch_vec_names[1];
                                    self.emit_line(&format!("cvt.rn.f32.bf16 {fs0}, {va};  // HIP bf16->f32"));
                                    self.emit_line(&format!("cvt.rn.f32.bf16 {fs1}, {vb};  // HIP bf16->f32"));
                                    self.emit_line(&format!("fma.rn.f32 {vc}, {fs0}, {fs1}, {vc};"));
                                }
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line(&format!("{vc} = fma((float){va}, (float){vb}, {vc});"));
                            }
                        }
                    }
                    DotDtype::Fp16 => {
                        // FP16 dot-product: fp32_acc += fp16_a · fp16_b (per-element).
                        match self.dialect {
                            GpuDialect::Ptx { sm_version } => {
                                if sm_version >= 100 {
                                    self.emit_line("// §SM100 DotProduct<Fp16> tcgen05");
                                    self.emit_line(&format!("tcgen05.mma.cta_group::1.kind::f16 [%tmem_addr], {va}, {vb}, 0x0, 0, 1;"));
                                    self.emit_line("tcgen05.wait::ld.sync.aligned;");
                                } else if sm_version >= 90 {
                                    self.emit_line("wgmma.fence.sync.aligned;");
                                    self.emit_line(&format!("wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {{{vc}}}, {va}, {vb};"));
                                    self.emit_line("wgmma.commit_group.sync.aligned;");
                                    self.emit_line("wgmma.wait_group.sync.aligned 0;");
                                } else if sm_version >= 80 {
                                    self.emit_line(&format!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{vc}}}, {{{va}}}, {{{vb}}}, {{{vc}}};"));
                                } else if sm_version >= 70 {
                                    self.emit_line(&format!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 {{{vc}}}, {{{va}}}, {{{vb}}}, {{{vc}}};"));
                                } else {
                                    let fs0 = self.scratch_vec_names[0];
                                    let fs1 = self.scratch_vec_names[1];
                                    self.emit_line(&format!("cvt.rn.f32.f16 {fs0}, {va};"));
                                    self.emit_line(&format!("cvt.rn.f32.f16 {fs1}, {vb};"));
                                    self.emit_line(&format!("fma.rn.f32 {vc}, {fs0}, {fs1}, {vc};"));
                                }
                            }
                            GpuDialect::Hip { gfx_arch, .. } => {
                                if gfx_arch >= 950 {
                                    self.emit_line(&format!("v_mfma_f32_32x32x16_f16 {vc}, {va}, {vb}, {vc};"));
                                } else if gfx_arch >= 908 {
                                    self.emit_line(&format!("v_mfma_f32_16x16x16_f16 {vc}, {va}, {vb}, {vc};"));
                                } else {
                                    self.emit_line(&format!("{vc} = fma((float){va}, (float){vb}, {vc});"));
                                }
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line(&format!("{vc} = fma((float){va}, (float){vb}, {vc});"));
                            }
                        }
                    }
                    DotDtype::Int8 => {
                        // INT8 dot-product: int32_acc += int8_a · int8_b (per-element).
                        // PTX SM80+: mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 (IMMA)
                        // PTX SM<80: software mul.s32 + add.s32
                        match self.dialect {
                            GpuDialect::Ptx { sm_version } => {
                                if sm_version >= 90 {
                                    self.emit_line(&format!("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {{{vc}}}, {{{va}}}, {{{vb}}}, {{{vc}}};"));
                                } else if sm_version >= 80 {
                                    self.emit_line(&format!("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {{{vc}}}, {{{va}}}, {{{vb}}}, {{{vc}}};"));
                                } else {
                                    let rs0 = self.scratch_gpr_names[0];
                                    self.emit_line(&format!("mul.lo.s32 {rs0}, {va}, {vb};"));
                                    self.emit_line(&format!("add.s32 {vc}, {vc}, {rs0};"));
                                }
                            }
                            GpuDialect::Hip { gfx_arch, .. } => {
                                if gfx_arch >= 950 {
                                    self.emit_line(&format!("v_mfma_i32_32x32x16_i8 {vc}, {va}, {vb}, {vc};"));
                                } else if gfx_arch >= 908 {
                                    self.emit_line(&format!("v_mfma_i32_16x16x16_i8 {vc}, {va}, {vb}, {vc};"));
                                } else {
                                    self.emit_line(&format!("{vc} += (int)({va}) * (int)({vb});"));
                                }
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line(&format!("{vc} += (int)({va}) * (int)({vb});"));
                            }
                        }
                    }
                    DotDtype::Int4x8 => {
                        // INT4×INT8 dot-product: int32_acc += int4_a · int8_b.
                        // Software unpack 4-bit to 8-bit, then IMMA/integer MAC.
                        // The a register contains packed 4-bit values; unpack low/high nibbles,
                        // sign-extend to 8-bit, then use integer multiply-accumulate.
                        match self.dialect {
                            GpuDialect::Ptx { sm_version: _ } => {
                                // Software unpack: extract low nibble (& 0xF) and high nibble (>> 4),
                                // subtract 8 for Q4_0 symmetric zero-point, then integer MAC.
                                let rs0 = self.scratch_gpr_names[0];
                                let rs1 = self.scratch_gpr_names[1];
                                // Low nibble: a & 0xF → sub 8 (sign) → mul by b → add to acc
                                self.emit_line(&format!("and.b32 {rs0}, {va}, 0xF;"));
                                self.emit_line(&format!("sub.s32 {rs0}, {rs0}, 8;"));
                                self.emit_line(&format!("mul.lo.s32 {rs0}, {rs0}, {vb};"));
                                self.emit_line(&format!("add.s32 {vc}, {vc}, {rs0};"));
                                // High nibble: (a >> 4) & 0xF → sub 8 → mul by b → add to acc
                                self.emit_line(&format!("shr.b32 {rs1}, {va}, 4;"));
                                self.emit_line(&format!("and.b32 {rs1}, {rs1}, 0xF;"));
                                self.emit_line(&format!("sub.s32 {rs1}, {rs1}, 8;"));
                                self.emit_line(&format!("mul.lo.s32 {rs1}, {rs1}, {vb};"));
                                self.emit_line(&format!("add.s32 {vc}, {vc}, {rs1};"));
                            }
                            GpuDialect::Hip { .. } => {
                                self.emit_line("{");
                                self.emit_line(&format!("  int _lo = ({va} & 0xF) - 8;"));
                                self.emit_line(&format!("  int _hi = (({va} >> 4) & 0xF) - 8;"));
                                self.emit_line(&format!("  {vc} += _lo * (int)({vb}) + _hi * (int)({vb});"));
                                self.emit_line("}");
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line("{");
                                self.emit_line(&format!("  int _lo = ({va} & 0xF) - 8;"));
                                self.emit_line(&format!("  int _hi = (({va} >> 4) & 0xF) - 8;"));
                                self.emit_line(&format!("  {vc} += _lo * (int)({vb}) + _hi * (int)({vb});"));
                                self.emit_line("}");
                            }
                        }
                    }
                    DotDtype::Fp4 => {
                        // FP4 dot-product: fp32_acc += e2m1_a · e2m1_b.
                        // SM100+: tcgen05 FP4 tensor core native instruction.
                        // SM<100 / other GPUs: software e2m1 decode (16-entry LUT) → F32 → FMA.
                        //
                        // E2M1 format (4 bits per value):
                        //   sign = bit 3
                        //   exp  = bits[2:1] (2-bit, values 0-3)
                        //   mant = bit 0     (1-bit, values 0-1)
                        //   value = (-1)^sign × (1 + mant×0.5) × 2^(exp - 1)
                        //   Special: nibble 0 → 0.0
                        match self.dialect {
                            GpuDialect::Ptx { sm_version } => {
                                if sm_version >= 100 {
                                    self.emit_line("// §SM100 DotProduct<Fp4> tcgen05 FP4");
                                    self.emit_line(&format!("tcgen05.mma.cta_group::1.kind::fp4 [%tmem_addr], {va}, {vb}, 0x0, 0, 1;"));
                                    self.emit_line("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    // SM<100: software e2m1 decode → F32 → FMA
                                    let rs0 = self.scratch_gpr_names[0];
                                    let rs1 = self.scratch_gpr_names[1];
                                    let fs0 = self.scratch_vec_names[0]; // decoded a
                                    let fs1 = self.scratch_vec_names[1]; // decoded b
                                    let fs2 = self.scratch_vec_names[2]; // magnitude temp
                                    let ps0 = self.scratch_pred_names[0]; // zero predicate
                                    let ps1 = self.scratch_pred_names[1]; // sign predicate

                                    // ── Decode va (e2m1 → f32) ──
                                    self.emit_line(&format!("// Decode FP4 e2m1: {va}"));
                                    self.emit_line(&format!("setp.eq.u32 {ps0}, {va}, 0;"));
                                    let skip_a = self.next_skip_label();
                                    self.emit_line(&format!("@{ps0} bra Lskip_a_{skip_a};"));
                                    self.emit_line(&format!("and.b32 {rs0}, {va}, 1;"));         // mant
                                    self.emit_line(&format!("cvt.rn.f32.u32 {fs2}, {rs0};"));    // mant as f32
                                    self.emit_line(&format!("mul.rn.f32 {fs2}, {fs2}, 0f3F000000;")); // mant * 0.5
                                    self.emit_line(&format!("add.rn.f32 {fs2}, {fs2}, 0f3F800000;")); // 1.0 + mant*0.5
                                    self.emit_line(&format!("shr.b32 {rs0}, {va}, 1;"));
                                    self.emit_line(&format!("and.b32 {rs0}, {rs0}, 3;"));         // exp_field
                                    self.emit_line(&format!("cvt.rn.f32.u32 {fs0}, {rs0};"));     // exp as f32
                                    self.emit_line(&format!("sub.rn.f32 {fs0}, {fs0}, 0f3F800000;")); // exp - 1.0
                                    self.emit_line(&format!("ex2.approx.ftz.f32 {fs0}, {fs0};")); // 2^(exp-1)
                                    self.emit_line(&format!("mul.rn.f32 {fs0}, {fs2}, {fs0};"));  // magnitude
                                    // Apply sign
                                    self.emit_line(&format!("shr.b32 {rs0}, {va}, 3;"));
                                    self.emit_line(&format!("and.b32 {rs0}, {rs0}, 1;"));
                                    self.emit_line(&format!("setp.ne.u32 {ps1}, {rs0}, 0;"));
                                    self.emit_line(&format!("@{ps1} neg.f32 {fs0}, {fs0};"));
                                    self.emit_line(&format!("Lskip_a_{skip_a}:"));
                                    self.emit_line(&format!("@{ps0} mov.f32 {fs0}, 0f00000000;"));

                                    // ── Decode vb (e2m1 → f32) ──
                                    self.emit_line(&format!("// Decode FP4 e2m1: {vb}"));
                                    self.emit_line(&format!("setp.eq.u32 {ps0}, {vb}, 0;"));
                                    let skip_b = self.next_skip_label();
                                    self.emit_line(&format!("@{ps0} bra Lskip_b_{skip_b};"));
                                    self.emit_line(&format!("and.b32 {rs1}, {vb}, 1;"));
                                    self.emit_line(&format!("cvt.rn.f32.u32 {fs2}, {rs1};"));
                                    self.emit_line(&format!("mul.rn.f32 {fs2}, {fs2}, 0f3F000000;"));
                                    self.emit_line(&format!("add.rn.f32 {fs2}, {fs2}, 0f3F800000;"));
                                    self.emit_line(&format!("shr.b32 {rs1}, {vb}, 1;"));
                                    self.emit_line(&format!("and.b32 {rs1}, {rs1}, 3;"));
                                    self.emit_line(&format!("cvt.rn.f32.u32 {fs1}, {rs1};"));
                                    self.emit_line(&format!("sub.rn.f32 {fs1}, {fs1}, 0f3F800000;"));
                                    self.emit_line(&format!("ex2.approx.ftz.f32 {fs1}, {fs1};"));
                                    self.emit_line(&format!("mul.rn.f32 {fs1}, {fs2}, {fs1};"));
                                    self.emit_line(&format!("shr.b32 {rs1}, {vb}, 3;"));
                                    self.emit_line(&format!("and.b32 {rs1}, {rs1}, 1;"));
                                    self.emit_line(&format!("setp.ne.u32 {ps1}, {rs1}, 0;"));
                                    self.emit_line(&format!("@{ps1} neg.f32 {fs1}, {fs1};"));
                                    self.emit_line(&format!("Lskip_b_{skip_b}:"));
                                    self.emit_line(&format!("@{ps0} mov.f32 {fs1}, 0f00000000;"));

                                    // ── Accumulate: acc += decoded_a * decoded_b ──
                                    self.emit_line(&format!("fma.rn.f32 {vc}, {fs0}, {fs1}, {vc};"));
                                }
                            }
                            GpuDialect::Hip { .. } => {
                                self.emit_line("{");
                                self.emit_line("  auto _e2m1 = [](unsigned n) -> float {");
                                self.emit_line("    if (n == 0) return 0.0f;");
                                self.emit_line("    int sign = (n >> 3) & 1;");
                                self.emit_line("    int exp2 = (n >> 1) & 3;");
                                self.emit_line("    int mant = n & 1;");
                                self.emit_line("    float mag = (1.0f + mant * 0.5f) * exp2f((float)exp2 - 1.0f);");
                                self.emit_line("    return sign ? -mag : mag;");
                                self.emit_line("  };");
                                self.emit_line(&format!("  float _a_f32 = _e2m1((unsigned)({va}));"));
                                self.emit_line(&format!("  float _b_f32 = _e2m1((unsigned)({vb}));"));
                                self.emit_line(&format!("  {vc} = fma(_a_f32, _b_f32, {vc});"));
                                self.emit_line("}");
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line("{");
                                self.emit_line("  auto _e2m1 = [](unsigned n) -> float {");
                                self.emit_line("    if (n == 0) return 0.0f;");
                                self.emit_line("    int sign = (n >> 3) & 1;");
                                self.emit_line("    int exp2 = (n >> 1) & 3;");
                                self.emit_line("    int mant = n & 1;");
                                self.emit_line("    float mag = (1.0f + mant * 0.5f) * exp2((float)exp2 - 1.0f);");
                                self.emit_line("    return sign ? -mag : mag;");
                                self.emit_line("  };");
                                self.emit_line(&format!("  float _a_f32 = _e2m1((unsigned)({va}));"));
                                self.emit_line(&format!("  float _b_f32 = _e2m1((unsigned)({vb}));"));
                                self.emit_line(&format!("  {vc} = fma(_a_f32, _b_f32, {vc});"));
                                self.emit_line("}");
                            }
                        }
                    }
                }
                Ok(())
            }

            VmInstr::ScaleApply { dst, acc, scale, zero, input_dtype, .. } => {
                // acc → FP32: dst = (f32)acc * scale + zero. input_dtype 决定转换指令。
                let d = self.reg_name_with_kind(*dst, alloc);
                let acc_r = self.reg_name_with_kind(*acc, alloc);
                let sc = self.reg_name_with_kind(*scale, alloc);
                let z = self.reg_name_with_kind(*zero, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Convert int32 → f32, multiply by scale
                        let fs0 = self.scratch_vec_names[0];
                        self.emit_line(&format!("cvt.rn.f32.s32 {fs0}, {acc_r};"));
                        self.emit_line(&format!("mul.rn.f32 {d}, {fs0}, {sc};"));
                        // Add zero-point if not NONE_VREG (VRegId(0))
                        if *zero != VRegId(0) {
                            self.emit_line(&format!("add.rn.f32 {d}, {d}, {z};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        if *zero != VRegId(0) {
                            self.emit_line(&format!("{d} = fma((float)({acc_r}), {sc}, {z});"));
                        } else {
                            self.emit_line(&format!("{d} = (float)({acc_r}) * {sc};"));
                        }
                    }
                    GpuDialect::Metal { .. } => {
                        if *zero != VRegId(0) {
                            self.emit_line(&format!("{d} = fma((float)({acc_r}), {sc}, {z});"));
                        } else {
                            self.emit_line(&format!("{d} = (float)({acc_r}) * {sc};"));
                        }
                    }
                }
                Ok(())
            }

            // ── 页压缩解码 (SPEC 22-PAGE-COMPRESSION §3.3) ──

            VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, decompressed_size } => {
                self.lower_gpu_lz4_decode(*src_ptr, *dst_ptr, *compressed_size, *decompressed_size, alloc)
            }

            VmInstr::BitPackRleDecode { src_ptr, dst_ptr, compressed_size, nibble_bits, element_count } => {
                self.lower_gpu_bitpack_rle_decode(*src_ptr, *dst_ptr, *compressed_size, *nibble_bits, *element_count, alloc)
            }

            // REQ-CG-013: GPU LoadCallbackEntry — load fn_ptr + ctx from callback table.
            // CallbackEntry layout: { fn_ptr: u64, ctx: u64 } = 16 bytes per entry.
            VmInstr::LoadCallbackEntry { table_ptr, slot_id, fn_ptr_out, ctx_out } => {
                let tbl = self.reg_name_with_kind(*table_ptr, alloc);
                let fn_reg = self.reg_name_with_kind(*fn_ptr_out, alloc);
                let ctx_reg = self.reg_name_with_kind(*ctx_out, alloc);
                let entry_offset = (*slot_id as u64) * 16;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load fn_ptr: ld.global.u64 [table + slot_id * 16]
                        self.emit_line(&format!(
                            "ld.global.u64 {}, [{}+{}];",
                            fn_reg, tbl, entry_offset
                        ));
                        // Load ctx: ld.global.u64 [table + slot_id * 16 + 8]
                        self.emit_line(&format!(
                            "ld.global.u64 {}, [{}+{}];",
                            ctx_reg, tbl, entry_offset + 8
                        ));
                    }
                    GpuDialect::Hip { .. } => {
                        // Cast table to u64 pointer array, index by slot
                        self.emit_line(&format!(
                            "{} = ((unsigned long long*){})[{slot_id}];",
                            fn_reg, tbl
                        ));
                        self.emit_line(&format!(
                            "{} = ((unsigned long long*){})[{}];",
                            ctx_reg, tbl, *slot_id + 1
                        ));
                    }
                    GpuDialect::Metal { .. } => {
                        // Cast table to device u64 pointer, index by slot
                        self.emit_line(&format!(
                            "{} = ((device ulong*){})[{slot_id} * 2];",
                            fn_reg, tbl
                        ));
                        self.emit_line(&format!(
                            "{} = ((device ulong*){})[{slot_id} * 2 + 1];",
                            ctx_reg, tbl
                        ));
                    }
                }
                Ok(())
            }

            // REQ-CG-013: GPU NativeCall — call device function pointer.
            // Limited to __device__ function calls via function pointer.
            VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr } => {
                let fn_reg = self.reg_name_with_kind(*fn_ptr, alloc);
                let ctx_reg = self.reg_name_with_kind(*ctx_ptr, alloc);
                let ret_reg = self.reg_name_with_kind(*ret_val, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // PTX call: move ctx to param, call func, read return value
                        self.emit_line("{");
                        self.emit_line("    .param .b64 param_ctx;");
                        self.emit_line("    .param .b32 param_ret;");
                        self.emit_line(&format!(
                            "    st.param.b64 [param_ctx], {};",
                            ctx_reg
                        ));
                        self.emit_line(&format!(
                            "    call (param_ret), {}, (param_ctx);",
                            fn_reg
                        ));
                        self.emit_line(&format!(
                            "    ld.param.b32 {}, [param_ret];",
                            ret_reg
                        ));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP: C-style function pointer call
                        // Cast fn_ptr to appropriate function signature and call with ctx
                        self.emit_line(&format!(
                            "{} = ((unsigned int(*)(unsigned long long)){})({});",
                            ret_reg, fn_reg, ctx_reg
                        ));
                    }
                    GpuDialect::Metal { .. } => {
                        // Metal does not support dynamic function pointers
                        return Err(CompilerError::CodegenViolation(
                            "GPU lower: VmInstr::NativeCall not supported on Metal (no dynamic function pointers)".into()
                        ));
                    }
                }
                Ok(())
            }

            // ═══ REQ-CG-010: Vector control ═══

            // VecCast: float <-> half conversion
            VmInstr::VecCast { dst, src, from_bits, to_bits } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                if from_bits == to_bits {
                    // Same bit-width: no-op copy
                    match self.dialect {
                        GpuDialect::Ptx { .. } => self.emit_line(&format!("mov.f32 {d}, {s};")),
                        _ => self.emit_line(&format!("{d} = {s};")),
                    }
                } else {
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            if *from_bits == 16 && *to_bits == 32 {
                                self.emit_line(&format!("cvt.rn.f32.f16 {d}, {s};"));
                            } else if *from_bits == 32 && *to_bits == 16 {
                                self.emit_line(&format!("cvt.rn.f16.f32 {d}, {s};"));
                            } else if *from_bits == 16 && *to_bits == 16 {
                                self.emit_line(&format!("mov.b32 {d}, {s};"));
                            } else if *from_bits == 32 && *to_bits == 32 {
                                self.emit_line(&format!("mov.f32 {d}, {s};"));
                            } else {
                                return Err(CompilerError::CodegenViolation(
                                    format!("VecCast: unsupported GPU conversion {}->{} bits", from_bits, to_bits)
                                ));
                            }
                        }
                        GpuDialect::Hip { .. } => {
                            if *from_bits == 16 && *to_bits == 32 {
                                self.emit_line(&format!("{d} = __half2float(__half({s}));"));
                            } else if *from_bits == 32 && *to_bits == 16 {
                                self.emit_line(&format!("{d} = __float2half({s});"));
                            } else {
                                self.emit_line(&format!("{d} = {s};"));
                            }
                        }
                        GpuDialect::Metal { .. } => {
                            if *from_bits == 16 && *to_bits == 32 {
                                self.emit_line(&format!("{d} = float(half({s}));"));
                            } else if *from_bits == 32 && *to_bits == 16 {
                                self.emit_line(&format!("{d} = half({s});"));
                            } else {
                                self.emit_line(&format!("{d} = {s};"));
                            }
                        }
                    }
                }
                Ok(())
            }

            // VecCmp: vector comparison with predicate
            VmInstr::VecCmp { dst, a, b, pred } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let vb = self.reg_name_with_kind(*b, alloc);
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let cmp_op = match pred {
                            CmpPredicate::Eq => "setp.eq.f32",
                            CmpPredicate::Ne => "setp.ne.f32",
                            CmpPredicate::Lt => "setp.lt.f32",
                            CmpPredicate::Le => "setp.le.f32",
                            CmpPredicate::Gt => "setp.gt.f32",
                            CmpPredicate::Ge => "setp.ge.f32",
                        };
                        self.emit_line(&format!("{cmp_op} {ps0}, {va}, {vb};"));
                        self.emit_line(&format!("selp.u32 {d}, 0xFFFFFFFF, 0, {ps0};"));
                    }
                    _ => {
                        let cmp_expr = match pred {
                            CmpPredicate::Eq => "==",
                            CmpPredicate::Ne => "!=",
                            CmpPredicate::Lt => "<",
                            CmpPredicate::Le => "<=",
                            CmpPredicate::Gt => ">",
                            CmpPredicate::Ge => ">=",
                        };
                        self.emit_line(&format!("{d} = ({va} {cmp_expr} {vb}) ? 0xFFFFFFFFu : 0u;"));
                    }
                }
                Ok(())
            }

            // ConditionalSelect: dst = mask ? true_val : false_val
            VmInstr::ConditionalSelect { dst, mask, true_val, false_val } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let m = self.reg_name_with_kind(*mask, alloc);
                let tv = self.reg_name_with_kind(*true_val, alloc);
                let fv = self.reg_name_with_kind(*false_val, alloc);
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("setp.ne.u32 {ps0}, {m}, 0;"));
                        self.emit_line(&format!("selp.f32 {d}, {tv}, {fv}, {ps0};"));
                    }
                    _ => {
                        self.emit_line(&format!("{d} = ({m} != 0u) ? {tv} : {fv};"));
                    }
                }
                Ok(())
            }

            // ═══ REQ-CG-011: LLM ops ═══

            // Argmax: find max value index in logits vector
            VmInstr::Argmax { dst, logits_ptr, vocab_bytes, width: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let base = self.reg_name_with_kind(*logits_ptr, alloc);
                let label_id = self.next_loop_label();
                let elem_count = *vocab_bytes / 4; // f32 elements
                let rs0 = self.scratch_gpr_names[0]; // index counter
                let fs0 = self.scratch_vec_names[0]; // current max value
                let fs1 = self.scratch_vec_names[1]; // loaded value
                let ps0 = self.scratch_pred_names[0];
                let ps1 = self.scratch_pred_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load first element as initial max
                        self.emit_line("cvt.u64.u32 %rd_addr, 0;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs0}, [%rd_addr];"));
                        self.emit_line(&format!("mov.u32 {d}, 0;")); // best_idx
                        self.emit_line(&format!("mov.u32 {rs0}, 1;")); // start from 1
                        self.emit_line(&format!("ARGMAX_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra ARGMAX_DONE_{label_id};"));
                        // Load element
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;"); // *4
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr];"));
                        // Compare
                        self.emit_line(&format!("setp.gt.f32 {ps1}, {fs1}, {fs0};"));
                        // Conditional update: if better, update max and idx
                        self.emit_line(&format!("@{ps1} mov.f32 {fs0}, {fs1};"));
                        self.emit_line(&format!("@{ps1} mov.u32 {d}, {rs0};"));
                        // Increment
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra ARGMAX_LOOP_{label_id};"));
                        self.emit_line(&format!("ARGMAX_DONE_{label_id}:"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("float max_val = *(({base}) + 0);"));
                        self.emit_line(&format!("{d} = 0;"));
                        self.emit_line(&format!("for (unsigned int {rs0} = 1; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("float {fs1} = *(({base}) + {rs0});"));
                        self.emit_line(&format!("if ({fs1} > max_val) {{ max_val = {fs1}; {d} = {rs0}; }}"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // TemperatureScale: logits /= temperature (in-place)
            VmInstr::TemperatureScale { logits_ptr, temp_ptr, vocab_bytes, width: _ } => {
                let base = self.reg_name_with_kind(*logits_ptr, alloc);
                let tp = self.reg_name_with_kind(*temp_ptr, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0];
                let fs0 = self.scratch_vec_names[0]; // temperature value
                let fs1 = self.scratch_vec_names[1]; // loaded logit
                let fs2 = self.scratch_vec_names[2]; // scaled logit
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load temperature
                        self.emit_line(&format!("ld.global.f32 {fs0}, [{tp}];"));
                        // Loop over logits
                        self.emit_line(&format!("mov.u32 {rs0}, 0;"));
                        self.emit_line(&format!("TEMP_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra TEMP_DONE_{label_id};"));
                        // Load logit
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr];"));
                        // Divide by temperature
                        self.emit_line(&format!("div.rn.f32 {fs2}, {fs1}, {fs0};"));
                        // Store back
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {fs2};"));
                        // Increment
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra TEMP_LOOP_{label_id};"));
                        self.emit_line(&format!("TEMP_DONE_{label_id}:"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("float {fs0} = *({tp});"));
                        self.emit_line(&format!("for (unsigned int {rs0} = 0; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("({base})[{rs0}] /= {fs0};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // StoreToken: write token_id to output_buf[counter] and input_ids
            VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => {
                let tid = self.reg_name_with_kind(*token_id, alloc);
                let obuf = self.reg_name_with_kind(*output_buf, alloc);
                let cnt = self.reg_name_with_kind(*counter, alloc);
                let iids = self.reg_name_with_kind(*input_ids_ptr, alloc);
                let plb = self.reg_name_with_kind(*prompt_len_bytes, alloc);
                let rs0 = self.scratch_gpr_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // output_buf[counter] = token_id
                        self.emit_line(&format!("shl.b32 {rs0}, {cnt}, 2;")); // counter * 4
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {obuf};"));
                        self.emit_line(&format!("st.global.u32 [%rd_addr], {tid};"));
                        // input_ids[prompt_len_bytes/4 + counter + 1] = token_id
                        self.emit_line(&format!("shr.u32 {rs0}, {plb}, 2;")); // prompt_len_bytes / 4
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, {cnt};"));
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("shl.b32 {rs0}, {rs0}, 2;")); // * 4 bytes
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {iids};"));
                        self.emit_line(&format!("st.global.u32 [%rd_addr], {tid};"));
                    }
                    _ => {
                        self.emit_line(&format!("*(({obuf}) + {cnt}) = {tid};"));
                        self.emit_line(&format!("*(({iids}) + ({plb}/4u + {cnt} + 1u)) = {tid};"));
                    }
                }
                Ok(())
            }

            // CheckStopCondition: if token == eos or counter >= max_tokens, jump to epilogue
            VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => {
                let tid = self.reg_name_with_kind(*token_id, alloc);
                let cnt = self.reg_name_with_kind(*counter, alloc);
                let eptr = self.reg_name_with_kind(*eos_ptr, alloc);
                let mptr = self.reg_name_with_kind(*max_tokens_ptr, alloc);
                let rs0 = self.scratch_gpr_names[0];
                let ps0 = self.scratch_pred_names[0];
                let ps1 = self.scratch_pred_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load eos_token_id
                        self.emit_line(&format!("ld.global.u32 {rs0}, [{eptr}];"));
                        // Compare token_id == eos
                        self.emit_line(&format!("setp.eq.u32 {ps0}, {tid}, {rs0};"));
                        self.emit_line(&format!("@{ps0} bra {};", self.epilogue_label));
                        // Load max_new_tokens
                        self.emit_line(&format!("ld.global.u32 {rs0}, [{mptr}];"));
                        // Compare counter >= max_tokens
                        self.emit_line(&format!("setp.ge.u32 {ps1}, {cnt}, {rs0};"));
                        self.emit_line(&format!("@{ps1} bra {};", self.epilogue_label));
                    }
                    _ => {
                        self.emit_line(&format!("if ({tid} == *({eptr})) goto {};", self.epilogue_label));
                        self.emit_line(&format!("if ({cnt} >= *({mptr})) goto {};", self.epilogue_label));
                    }
                }
                Ok(())
            }

            // ═══ REQ-CG-012: GPR ops ═══

            // AddPtr: dst = base + offset (64-bit pointer add)
            VmInstr::AddPtr { dst, base, offset } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("add.u64 {d}, {b}, {offset};"));
                    }
                    _ => {
                        self.emit_line(&format!("{d} = {b} + {offset}UL;"));
                    }
                }
                Ok(())
            }

            // StoreConstToStack: write compile-time constant to local/shared memory slot
            VmInstr::StoreConstToStack { rbp_offset, value, elem_width } => {
                let slot_bytes = (*rbp_offset as usize) * (*elem_width);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("st.shared.u32 [smem+{slot_bytes}], {value};"));
                    }
                    _ => {
                        self.emit_line(&format!("((unsigned int*)smem)[{slot_bytes}/4] = {value}u;"));
                    }
                }
                Ok(())
            }

            // REQ-VR-003: GprBinOp — 统一 GPR 二元操作 (替代旧 GprShl/GprShr/GprSubConst/GprAdd/GprSub/GprMulConst)
            VmInstr::GprBinOp { dst, a, b, op } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let va = self.reg_name_with_kind(*a, alloc);
                let vb_str = match b {
                    GprOperand::VReg(v) => self.reg_name_with_kind(*v, alloc),
                    GprOperand::Imm(imm) => format!("{imm}"),
                };
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let ptx_op = match op {
                            GprOp::Add => "add.u32",
                            GprOp::Sub => "sub.u32",
                            GprOp::Mul => "mul.wide.u32",
                            GprOp::Div => "div.u32",
                            GprOp::Shl => "shl.b32",
                            GprOp::Shr => "shr.u32",
                            GprOp::And => "and.b32",
                            GprOp::Or  => "or.b32",
                            GprOp::Xor => "xor.b32",
                            GprOp::BitTest => {
                                // dst = (a >> b) & 1
                                self.emit_line(&format!("shr.b32 {d}, {va}, {vb_str};"));
                                self.emit_line(&format!("and.b32 {d}, {d}, 1;"));
                                return Ok(());
                            }
                        };
                        self.emit_line(&format!("{ptx_op} {d}, {va}, {vb_str};"));
                    }
                    _ => {
                        let metal_op = match op {
                            GprOp::Add => "+",
                            GprOp::Sub => "-",
                            GprOp::Mul => "*",
                            GprOp::Div => "/",
                            GprOp::Shl => "<<",
                            GprOp::Shr => ">>",
                            GprOp::And => "&",
                            GprOp::Or  => "|",
                            GprOp::Xor => "^",
                            GprOp::BitTest => {
                                // dst = (a >> b) & 1
                                self.emit_line(&format!("{d} = ({va} >> {vb_str}) & 1;"));
                                return Ok(());
                            }
                        };
                        self.emit_line(&format!("{d} = {va} {metal_op} {vb_str};"));
                    }
                }
                Ok(())
            }

            // GgufSubScaleLoad / GgufKQuantScaleLoad: x86-only, GPU stub
            VmInstr::GgufSubScaleLoad { .. } | VmInstr::GgufKQuantScaleLoad { .. } => {
                Err(CompilerError::Internal("GgufSubScaleLoad/GgufKQuantScaleLoad: x86-only, not yet implemented for GPU".into()))
            }

            VmInstr::QuantBlockLoad { dst, base, offset, unpack, width } => {
                self.lower_quant_block_load(*dst, *base, offset, unpack, *width, alloc)
            }
            VmInstr::QuantBiPlaneLoad { dst, qs_base, extra_base, bias, mode, width } => {
                self.lower_quant_biplane_load(*dst, *qs_base, *extra_base, *bias, mode, *width, alloc)
            }

            // REQ-VR-010: GprUnaryOp — GPR 一元操作 (Not/Popcount/Clz/Bswap/Neg)
            VmInstr::GprUnaryOp { dst, src, op } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let ptx_op = match op {
                            GprUnaryOpKind::Not => format!("not.b32 {d}, {s};"),
                            GprUnaryOpKind::Popcount => format!("popc.b32 {d}, {s};"),
                            GprUnaryOpKind::Clz => format!("clz.b32 {d}, {s};"),
                            GprUnaryOpKind::Bswap => format!("prmt.b32 {d}, {s}, {s}, 0x0123;"),
                            GprUnaryOpKind::Neg => format!("sub.u32 {d}, 0, {s};"),
                        };
                        self.emit_line(&ptx_op);
                    }
                    _ => {
                        let metal_expr = match op {
                            GprUnaryOpKind::Not => format!("~{s}"),
                            GprUnaryOpKind::Popcount => format!("popcount({s})"),
                            GprUnaryOpKind::Clz => format!("clz({s})"),
                            GprUnaryOpKind::Bswap => format!("(({s} << 24) | (({s} & 0xFF00) << 8) | (({s} >> 8) & 0xFF00) | ({s} >> 24))"),
                            GprUnaryOpKind::Neg => format!("-{s}"),
                        };
                        self.emit_line(&format!("{d} = {metal_expr};"));
                    }
                }
                Ok(())
            }

            VmInstr::GprLoadImm { dst, value } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("mov.u32 {d}, {value};"));
                    }
                    _ => {
                        self.emit_line(&format!("{d} = {value};"));
                    }
                }
                Ok(())
            }

            // REQ-CG-009: Gather ops — GPU lowering (PTX/HIP/Metal)

            // GatherLoad: 从 base + indices[i]*stride 加载 lanes 个 f32 到 dst 向量。
            // GPU SIMT 模型：每个 thread 处理一个 lane。
            VmInstr::GatherLoad { dst, base, indices, stride, width , dtype: _dtype, predicate: _predicate, } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let idx_base = self.reg_name_with_kind(*indices, alloc);
                let lanes = width.f32_lanes();
                let rs0 = self.scratch_gpr_names[0];
                let rs1 = self.scratch_gpr_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line("// GatherLoad: SIMT — each thread loads one element");
                        self.emit_line(&format!("ld.global.u32 {rs0}, [{idx_base}];"));
                        if *stride != 1 {
                            self.emit_line(&format!("mul.lo.s32 {rs1}, {rs0}, {stride};"));
                            self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs1};"));
                        } else {
                            self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        }
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("ld.global.f32 {d}, [%rd_addr];"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned int _idx = *{idx_base};"));
                        if *stride != 1 {
                            self.emit_line(&format!("  unsigned int _off = _idx * {stride};"));
                        } else {
                            self.emit_line("  unsigned int _off = _idx;");
                        }
                        self.emit_line(&format!("  {d} = *(({b}) + _off/4);"));
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned int _idx = *{idx_base};"));
                        if *stride != 1 {
                            self.emit_line(&format!("  unsigned int _off = _idx * {stride};"));
                        } else {
                            self.emit_line("  unsigned int _off = _idx;");
                        }
                        self.emit_line(&format!("  {d} = {b}[_off/4];"));
                        self.emit_line("}");
                    }
                }
                let _ = (lanes, rs0, rs1);
                Ok(())
            }

            // ScatterStore: 将 src 向量的 lanes 个 f32 按 indices 写入 base + indices[i]*stride。
            VmInstr::ScatterStore { base, indices, src, stride, width , dtype: _dtype, predicate: _predicate, } => {
                let s = self.reg_name_with_kind(*src, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let idx_base = self.reg_name_with_kind(*indices, alloc);
                let lanes = width.f32_lanes();
                let rs0 = self.scratch_gpr_names[0];
                let rs1 = self.scratch_gpr_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line("// ScatterStore: SIMT — each thread stores one element");
                        self.emit_line(&format!("ld.global.u32 {rs0}, [{idx_base}];"));
                        if *stride != 1 {
                            self.emit_line(&format!("mul.lo.s32 {rs1}, {rs0}, {stride};"));
                            self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs1};"));
                        } else {
                            self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        }
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {s};"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned int _idx = *{idx_base};"));
                        if *stride != 1 {
                            self.emit_line(&format!("  unsigned int _off = _idx * {stride};"));
                        } else {
                            self.emit_line("  unsigned int _off = _idx;");
                        }
                        self.emit_line(&format!("  *(({b}) + _off/4) = {s};"));
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned int _idx = *{idx_base};"));
                        if *stride != 1 {
                            self.emit_line(&format!("  unsigned int _off = _idx * {stride};"));
                        } else {
                            self.emit_line("  unsigned int _off = _idx;");
                        }
                        self.emit_line(&format!("  {b}[_off/4] = {s};"));
                        self.emit_line("}");
                    }
                }
                let _ = (lanes, rs0, rs1);
                Ok(())
            }

            // TableLookup: 从 base + row_index * row_bytes 加载一行 SIMD 向量。
            // 等价于 IntMulStride(row_index, row_bytes) + PtrAdd(base) + VecLoad。
            VmInstr::TableLookup { dst, base, row_index, row_bytes, width } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let b = self.reg_name_with_kind(*base, alloc);
                let ri = self.reg_name_with_kind(*row_index, alloc);
                let lanes = width.f32_lanes();
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        if *row_bytes != 0 {
                            self.emit_line(&format!("mul.wide.u32 %rd_addr, {ri}, {row_bytes};"));
                        } else {
                            self.emit_line("cvt.u64.u32 %rd_addr, 0;");
                        }
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("ld.global.f32 {d}, [%rd_addr];"));
                    }
                    GpuDialect::Hip { .. } => {
                        if *row_bytes != 0 {
                            self.emit_line(&format!("{d} = *(({b}) + (unsigned long long)({ri}) * {row_bytes} / 4);"));
                        } else {
                            self.emit_line(&format!("{d} = *{b};"));
                        }
                    }
                    GpuDialect::Metal { .. } => {
                        if *row_bytes != 0 {
                            self.emit_line(&format!("{d} = {b}[(unsigned long)({ri}) * {row_bytes} / 4];"));
                        } else {
                            self.emit_line(&format!("{d} = {b}[0];"));
                        }
                    }
                }
                let _ = lanes;
                Ok(())
            }

            // ── GPU-Resident 采样指令 (REQ-GPU-SAMPLE) ──

            VmInstr::SoftmaxReduceMax { dst, logits_ptr, vocab_bytes, width: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let base = self.reg_name_with_kind(*logits_ptr, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0];
                let fs0 = self.scratch_vec_names[0];
                let fs1 = self.scratch_vec_names[1];
                let fs2 = self.scratch_vec_names[2];
                let ps0 = self.scratch_pred_names[0];
                let ps1 = self.scratch_pred_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load first element as initial max
                        self.emit_line("cvt.u64.u32 %rd_addr, 0;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs0}, [%rd_addr];"));
                        // Warp-shuffle max across 32 threads for numerical stability
                        self.emit_line("// warp reduction max across threads");
                        self.emit_line(&format!("shfl.sync.down.b32 {fs1}, {fs0}, 16, 0x1f;"));
                        self.emit_line(&format!("max.f32 {fs0}, {fs0}, {fs1};"));
                        self.emit_line(&format!("shfl.sync.down.b32 {fs1}, {fs0}, 8, 0x1f;"));
                        self.emit_line(&format!("max.f32 {fs0}, {fs0}, {fs1};"));
                        self.emit_line(&format!("shfl.sync.down.b32 {fs1}, {fs0}, 4, 0x1f;"));
                        self.emit_line(&format!("max.f32 {fs0}, {fs0}, {fs1};"));
                        self.emit_line(&format!("shfl.sync.down.b32 {fs1}, {fs0}, 2, 0x1f;"));
                        self.emit_line(&format!("max.f32 {fs0}, {fs0}, {fs1};"));
                        self.emit_line(&format!("shfl.sync.down.b32 {fs1}, {fs0}, 1, 0x1f;"));
                        self.emit_line(&format!("max.f32 {fs0}, {fs0}, {fs1};"));
                        // Per-thread loop over vocab chunk (each thread scans its slice)
                        self.emit_line(&format!("mov.u32 {rs0}, 0;"));
                        self.emit_line(&format!("SMAX_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra SMAX_DONE_{label_id};"));
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr];"));
                        self.emit_line(&format!("setp.gt.f32 {ps1}, {fs1}, {fs0};"));
                        self.emit_line(&format!("@{ps1} mov.f32 {fs0}, {fs1};"));
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra SMAX_LOOP_{label_id};"));
                        self.emit_line(&format!("SMAX_DONE_{label_id}:"));
                        // Final warp broadcast (thread 0's result to all)
                        self.emit_line(&format!("shfl.sync.idx.b32 {fs2}, {fs0}, 0, 0x1f;"));
                        self.emit_line(&format!("mov.f32 {d}, {fs2};"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("float {fs0} = -3.402823466e38f;"));
                        self.emit_line(&format!("for (unsigned int {rs0} = 0; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("float {fs1} = (({base})[{rs0}]);"));
                        self.emit_line(&format!("if ({fs1} > {fs0}) {fs0} = {fs1};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.emit_line(&format!("{d} = {fs0};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                let _ = (fs2, ps1);
                Ok(())
            }

            VmInstr::SoftmaxExpSum { sum_dst, logits_ptr, max_val, vocab_bytes, width: _ } => {
                let sd = self.reg_name_with_kind(*sum_dst, alloc);
                let base = self.reg_name_with_kind(*logits_ptr, alloc);
                let mx = self.reg_name_with_kind(*max_val, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0];
                let fs0 = self.scratch_vec_names[0]; // accumulator
                let fs1 = self.scratch_vec_names[1]; // loaded value
                let fs2 = self.scratch_vec_names[2]; // exp result
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("mov.f32 {fs0}, 0.0;"));
                        self.emit_line(&format!("mov.u32 {rs0}, 0;"));
                        self.emit_line(&format!("SEXPSUM_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra SEXPSUM_DONE_{label_id};"));
                        // Load logit
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr];"));
                        // exp(x - max) using ex2.approx: exp2(log2e * (x - max))
                        self.emit_line(&format!("sub.f32 {fs2}, {fs1}, {mx};"));
                        self.emit_line(&format!("mul.f32 {fs2}, {fs2}, 1.4426950409;")); // log2(e)
                        self.emit_line(&format!("ex2.approx.ftz.f32 {fs2}, {fs2};"));
                        // Accumulate
                        self.emit_line(&format!("add.f32 {fs0}, {fs0}, {fs2};"));
                        // Store exp back for SoftmaxNormalize
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {fs2};"));
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra SEXPSUM_LOOP_{label_id};"));
                        self.emit_line(&format!("SEXPSUM_DONE_{label_id}:"));
                        self.emit_line(&format!("mov.f32 {sd}, {fs0};"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("float {fs0} = 0.0f;"));
                        self.emit_line(&format!("float {fs2} = {mx};"));
                        self.emit_line(&format!("for (unsigned int {rs0} = 0; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("float {fs1} = expf(({base})[{rs0}] - {fs2});"));
                        self.emit_line(&format!("({base})[{rs0}] = {fs1};"));
                        self.emit_line(&format!("{fs0} += {fs1};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.emit_line(&format!("{sd} = {fs0};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            VmInstr::SoftmaxNormalize { logits_ptr, sum_val, vocab_bytes, width: _ } => {
                let base = self.reg_name_with_kind(*logits_ptr, alloc);
                let sv = self.reg_name_with_kind(*sum_val, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0];
                let fs0 = self.scratch_vec_names[0]; // loaded exp
                let fs1 = self.scratch_vec_names[1]; // normalized
                let ps0 = self.scratch_pred_names[0];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("mov.u32 {rs0}, 0;"));
                        self.emit_line(&format!("SNORM_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra SNORM_DONE_{label_id};"));
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                        self.emit_line(&format!("ld.global.f32 {fs0}, [%rd_addr];"));
                        self.emit_line(&format!("div.rn.f32 {fs1}, {fs0}, {sv};"));
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {fs1};"));
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra SNORM_LOOP_{label_id};"));
                        self.emit_line(&format!("SNORM_DONE_{label_id}:"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("float {fs1} = {sv};"));
                        self.emit_line(&format!("for (unsigned int {rs0} = 0; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("({base})[{rs0}] /= {fs1};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            VmInstr::SampleTopKFilter { probs_ptr, indices_ptr, k_ptr, vocab_bytes, width: _ } => {
                let pbase = self.reg_name_with_kind(*probs_ptr, alloc);
                let ibase = self.reg_name_with_kind(*indices_ptr, alloc);
                let kp = self.reg_name_with_kind(*k_ptr, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0]; // outer loop i
                let rs1 = self.scratch_gpr_names[1]; // inner loop j / count of kept
                let fs0 = self.scratch_vec_names[0]; // threshold prob value
                let fs1 = self.scratch_vec_names[1]; // loaded prob
                let fs2 = self.scratch_vec_names[2]; // temp
                let ps0 = self.scratch_pred_names[0];
                let ps1 = self.scratch_pred_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Top-K filter: find K-th largest value via partial selection,
                        // then zero out everything below threshold.
                        // Strategy: bubble-selection outer loop runs K iterations,
                        // each iteration finds the current maximum and swaps it to front.
                        // Then zero everything from index K onward.
                        self.emit_line(&format!("ld.global.u32 {rs1}, [{kp}];")); // K
                        // Selection sort: move top-K elements to front
                        self.emit_line(&format!("mov.u32 {rs0}, 0;"));
                        self.emit_line(&format!("TOPK_OUTER_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {rs1};"));
                        self.emit_line(&format!("@{ps0} bra TOPK_OUTER_DONE_{label_id};"));
                        // Find max from [i..elem_count)
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {pbase};"));
                        self.emit_line(&format!("ld.global.f32 {fs0}, [%rd_addr];")); // best_val
                        self.emit_line(&format!("mov.u32 %rs_bound, {rs0};")); // best_idx
                        // Inner loop
                        let inner_label = self.next_loop_label();
                        self.emit_line(&format!("mov.u32 %rd_inner_j, {rs0};"));
                        self.emit_line("add.u32 %rd_inner_j, %rd_inner_j, 1;");
                        self.emit_line(&format!("TOPK_INNER_{inner_label}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps1}, %rd_inner_j, {elem_count};"));
                        self.emit_line(&format!("@{ps1} bra TOPK_INNER_DONE_{inner_label};"));
                        self.emit_line("cvt.u64.u32 %rd_addr2, %rd_inner_j;");
                        self.emit_line("shl.b32 %rd_addr2, %rd_addr2, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr2, %rd_addr2, {pbase};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr2];"));
                        self.emit_line(&format!("setp.gt.f32 {ps0}, {fs1}, {fs0};"));
                        self.emit_line(&format!("@{ps0} mov.f32 {fs0}, {fs1};"));
                        self.emit_line(&format!("@{ps0} mov.u32 %rs_bound, %rd_inner_j;"));
                        self.emit_line("add.u32 %rd_inner_j, %rd_inner_j, 1;");
                        self.emit_line(&format!("bra TOPK_INNER_{inner_label};"));
                        self.emit_line(&format!("TOPK_INNER_DONE_{inner_label}:"));
                        // Swap prob[i] <-> prob[best_idx] if needed
                        self.emit_line(&format!("setp.eq.u32 {ps0}, %rs_bound, {rs0};"));
                        self.emit_line(&format!("@{ps0} bra TOPK_NOSWAP_{label_id}_{inner_label};"));
                        // Swap probs
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {pbase};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr];")); // probs[i]
                        self.emit_line("cvt.u64.u32 %rd_addr2, %rs_bound;");
                        self.emit_line("shl.b32 %rd_addr2, %rd_addr2, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr2, %rd_addr2, {pbase};"));
                        self.emit_line(&format!("ld.global.f32 {fs2}, [%rd_addr2];")); // probs[best]
                        self.emit_line(&format!("st.global.f32 [%rd_addr], {fs2};"));
                        self.emit_line(&format!("st.global.f32 [%rd_addr2], {fs1};"));
                        // Swap indices similarly
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {ibase};"));
                        self.emit_line("ld.global.u32 %rd_tmp_u, [%rd_addr];");
                        self.emit_line("cvt.u64.u32 %rd_addr2, %rs_bound;");
                        self.emit_line("shl.b32 %rd_addr2, %rd_addr2, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr2, %rd_addr2, {ibase};"));
                        self.emit_line("ld.global.u32 %rd_tmp_u2, [%rd_addr2];");
                        self.emit_line("st.global.u32 [%rd_addr], %rd_tmp_u2;");
                        self.emit_line("st.global.u32 [%rd_addr2], %rd_tmp_u;");
                        self.emit_line(&format!("TOPK_NOSWAP_{label_id}_{inner_label}:"));
                        // Store index at position i
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {ibase};"));
                        self.emit_line(&format!("st.global.u32 [%rd_addr], {rs0};")); // original index
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra TOPK_OUTER_{label_id};"));
                        self.emit_line(&format!("TOPK_OUTER_DONE_{label_id}:"));
                        // Zero out probs[K..elem_count)
                        let zero_label = self.next_loop_label();
                        self.emit_line(&format!("mov.u32 {rs0}, {rs1};")); // start at K
                        self.emit_line(&format!("TOPK_ZERO_{zero_label}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra TOPK_ZERO_DONE_{zero_label};"));
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {pbase};"));
                        self.emit_line("st.global.f32 [%rd_addr], 0.0;");
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra TOPK_ZERO_{zero_label};"));
                        self.emit_line(&format!("TOPK_ZERO_DONE_{zero_label}:"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("unsigned int {rs1} = *({kp});"));
                        // Initialize indices
                        self.emit_line(&format!("for (unsigned int {rs0} = 0; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("({ibase})[{rs0}] = {rs0};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        // Selection sort top-K
                        self.emit_line(&format!("for (unsigned int {rs0} = 0; {rs0} < {rs1}; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("unsigned int best = {rs0};"));
                        self.emit_line(&format!("for (unsigned int j = {rs0}+1; j < {elem_count}u; j++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("if (({pbase})[j] > ({pbase})[best]) best = j;"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        // Swap
                        self.emit_line(&format!("if (best != {rs0}) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("float {fs0} = ({pbase})[{rs0}]; ({pbase})[{rs0}] = ({pbase})[best]; ({pbase})[best] = {fs0};"));
                        self.emit_line(&format!("unsigned int {fs1} = ({ibase})[{rs0}]; ({ibase})[{rs0}] = ({ibase})[best]; ({ibase})[best] = {fs1};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        // Zero rest
                        self.emit_line(&format!("for (unsigned int {rs0} = {rs1}; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("({pbase})[{rs0}] = 0.0f;"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                let _ = (fs2, ps1);
                Ok(())
            }

            VmInstr::SampleTopPFilter { probs_ptr, p_ptr, vocab_bytes, width: _ } => {
                let pbase = self.reg_name_with_kind(*probs_ptr, alloc);
                let pp = self.reg_name_with_kind(*p_ptr, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0];
                let fs0 = self.scratch_vec_names[0]; // cumulative sum
                let fs1 = self.scratch_vec_names[1]; // loaded prob
                let fs2 = self.scratch_vec_names[2]; // target p
                let ps0 = self.scratch_pred_names[0];
                let ps1 = self.scratch_pred_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("ld.global.f32 {fs2}, [{pp}];")); // target p
                        self.emit_line(&format!("mov.f32 {fs0}, 0.0;"));
                        self.emit_line(&format!("mov.u32 {rs0}, 0;"));
                        self.emit_line(&format!("TOPP_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra TOPP_CUTOFF_{label_id};"));
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {pbase};"));
                        self.emit_line(&format!("ld.global.f32 {fs1}, [%rd_addr];"));
                        self.emit_line(&format!("add.f32 {fs0}, {fs0}, {fs1};"));
                        // If cumulative > p, zero remaining and done
                        self.emit_line(&format!("setp.gt.f32 {ps1}, {fs0}, {fs2};"));
                        self.emit_line(&format!("@{ps1} bra TOPP_CUTOFF_{label_id};"));
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra TOPP_LOOP_{label_id};"));
                        self.emit_line(&format!("TOPP_CUTOFF_{label_id}:"));
                        // Zero out probs[i+1..elem_count)
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        let zero_label = self.next_loop_label();
                        self.emit_line(&format!("TOPP_ZERO_{zero_label}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs0}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra TOPP_ZERO_DONE_{zero_label};"));
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {pbase};"));
                        self.emit_line("st.global.f32 [%rd_addr], 0.0;");
                        self.emit_line(&format!("add.u32 {rs0}, {rs0}, 1;"));
                        self.emit_line(&format!("bra TOPP_ZERO_{zero_label};"));
                        self.emit_line(&format!("TOPP_ZERO_DONE_{zero_label}:"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("float {fs2} = *({pp});"));
                        self.emit_line(&format!("float {fs0} = 0.0f;"));
                        self.emit_line(&format!("unsigned int {rs0} = 0;"));
                        self.emit_line(&format!("for (; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("{fs0} += ({pbase})[{rs0}];"));
                        self.emit_line(&format!("if ({fs0} >= {fs2}) break;"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.emit_line(&format!("for ({rs0}++; {rs0} < {elem_count}u; {rs0}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("({pbase})[{rs0}] = 0.0f;"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                let _ = (fs1, ps1);
                Ok(())
            }

            VmInstr::SampleMultinomial { dst, probs_ptr, rng_state_ptr, vocab_bytes, width: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let pbase = self.reg_name_with_kind(*probs_ptr, alloc);
                let rng = self.reg_name_with_kind(*rng_state_ptr, alloc);
                let elem_count = *vocab_bytes / 4;
                let label_id = self.next_loop_label();
                let rs0 = self.scratch_gpr_names[0]; // loop counter
                let rs1 = self.scratch_gpr_names[1]; // candidate index
                let fs0 = self.scratch_vec_names[0]; // uniform random
                let fs1 = self.scratch_vec_names[1]; // cumulative sum
                let fs2 = self.scratch_vec_names[2]; // loaded prob
                let ps0 = self.scratch_pred_names[0];
                let ps1 = self.scratch_pred_names[1];
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Philox-based uniform random: load counter+key, multiply, xor
                        self.emit_line(&format!("ld.global.u64 %rd_rng0, [{rng}];")); // counter
                        self.emit_line(&format!("ld.global.u64 %rd_rng1, [{rng}+8];")); // key
                        // Philox round: state = state * 0x9E3779B97F4A7C15 + key
                        self.emit_line("mul.hi.u64 %rd_tmp, %rd_rng0, 0x9E3779B97F4A7C15;");
                        self.emit_line("add.u64 %rd_rng0, %rd_tmp, %rd_rng1;");
                        self.emit_line("xor.b64 %rd_rng0, %rd_rng0, %rd_rng1;");
                        // Store updated state
                        self.emit_line(&format!("st.global.u64 [{rng}], %rd_rng0;"));
                        // Convert to [0,1) f32: take low 24 bits and normalize
                        self.emit_line("cvt.u32.u64 %rd_tmp_u, %rd_rng0;");
                        self.emit_line("and.b32 %rd_tmp_u, %rd_tmp_u, 0x007FFFFF;");
                        self.emit_line("or.b32 %rd_tmp_u, %rd_tmp_u, 0x3F800000;"); // 1.0 + mantissa
                        self.emit_line(&format!("mov.b32 {fs0}, %rd_tmp_u;"));
                        self.emit_line(&format!("sub.f32 {fs0}, {fs0}, 1.0;")); // [0, 1)
                        // Binary search: cumulative sum until >= random
                        self.emit_line(&format!("mov.f32 {fs1}, 0.0;"));
                        self.emit_line(&format!("mov.u32 {rs1}, 0;"));
                        self.emit_line(&format!("SAMPLE_LOOP_{label_id}:"));
                        self.emit_line(&format!("setp.ge.u32 {ps0}, {rs1}, {elem_count};"));
                        self.emit_line(&format!("@{ps0} bra SAMPLE_DONE_{label_id};"));
                        self.emit_line(&format!("cvt.u64.u32 %rd_addr, {rs1};"));
                        self.emit_line("shl.b32 %rd_addr, %rd_addr, 2;");
                        self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {pbase};"));
                        self.emit_line(&format!("ld.global.f32 {fs2}, [%rd_addr];"));
                        self.emit_line(&format!("add.f32 {fs1}, {fs1}, {fs2};"));
                        self.emit_line(&format!("setp.ge.f32 {ps1}, {fs1}, {fs0};"));
                        self.emit_line(&format!("@{ps1} bra SAMPLE_DONE_{label_id};"));
                        self.emit_line(&format!("add.u32 {rs1}, {rs1}, 1;"));
                        self.emit_line(&format!("bra SAMPLE_LOOP_{label_id};"));
                        self.emit_line(&format!("SAMPLE_DONE_{label_id}:"));
                        self.emit_line(&format!("mov.u32 {d}, {rs1};"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        // Simple LCG random from state
                        self.emit_line(&format!("uint64_t _rng0 = *((uint64_t*)({rng}));"));
                        self.emit_line(&format!("uint64_t _rng1 = *((uint64_t*)({rng})+1);"));
                        self.emit_line("_rng0 = _rng0 * 0x9E3779B97F4A7C15ULL + _rng1;");
                        self.emit_line("_rng0 ^= _rng1;");
                        self.emit_line(&format!("*((uint64_t*)({rng})) = _rng0;"));
                        self.emit_line("uint32_t _mantissa = (uint32_t)(_rng0) & 0x007FFFFFu;");
                        self.emit_line("_mantissa |= 0x3F800000u;");
                        self.emit_line(&format!("float {fs0};"));
                        self.emit_line("*(uint32_t*)&{fs0} = _mantissa;");
                        self.emit_line(&format!("{fs0} -= 1.0f;"));
                        // Cumulative search
                        self.emit_line(&format!("float {fs1} = 0.0f;"));
                        self.emit_line(&format!("unsigned int {rs1} = 0;"));
                        self.emit_line(&format!("for (; {rs1} < {elem_count}u; {rs1}++) {{"));
                        self.indent += 1;
                        self.emit_line(&format!("{fs1} += ({pbase})[{rs1}];"));
                        self.emit_line(&format!("if ({fs1} >= {fs0}) break;"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                        self.emit_line(&format!("{d} = {rs1};"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                let _ = (rs0, ps0);
                Ok(())
            }

            VmInstr::WarpPRNG { dst, rng_state_ptr } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let rng = self.reg_name_with_kind(*rng_state_ptr, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Philox 2×32 single round:
                        // counter = state[0], key = state[1] + laneid
                        // result = hash(counter, key + laneid)
                        self.emit_line("// WarpPRNG: Philox 2x32 single round per lane");
                        // Load state
                        self.emit_line(&format!("ld.global.u32 %rd_tmp_u, [{rng}];")); // counter lo
                        self.emit_line(&format!("ld.global.u32 %rd_tmp_u2, [{rng}+4];")); // key lo
                        // Mix in lane ID for per-thread uniqueness
                        self.emit_line("mov.u32 %rd_laneid, %tid.x;");
                        self.emit_line("and.b32 %rd_laneid, %rd_laneid, 31;"); // lane within warp
                        self.emit_line("xor.b32 %rd_tmp_u2, %rd_tmp_u2, %rd_laneid;");
                        // Philox S-box: mul hi + xor (single round)
                        self.emit_line("mul.hi.u32 %rd_tmp_u3, %rd_tmp_u, 0x6C622960;");
                        self.emit_line("xor.b32 %rd_tmp_u, %rd_tmp_u, %rd_tmp_u2;");
                        self.emit_line("add.u32 %rd_tmp_u, %rd_tmp_u, %rd_tmp_u3;");
                        // Update counter in state
                        self.emit_line("add.u32 %rd_tmp_u2, %rd_tmp_u, 1;");
                        self.emit_line(&format!("st.global.u32 [{rng}], %rd_tmp_u2;"));
                        // Convert to [0, 1) f32: IEEE 754 trick
                        self.emit_line("and.b32 %rd_tmp_u, %rd_tmp_u, 0x007FFFFF;");
                        self.emit_line("or.b32 %rd_tmp_u, %rd_tmp_u, 0x3F800000;");
                        self.emit_line(&format!("mov.b32 {d}, %rd_tmp_u;"));
                        self.emit_line(&format!("sub.f32 {d}, {d}, 1.0;"));
                    }
                    _ => {
                        self.emit_line("{");
                        self.indent += 1;
                        self.emit_line(&format!("uint32_t _ctr = *((uint32_t*)({rng}));"));
                        self.emit_line(&format!("uint32_t _key = *((uint32_t*)({rng})+1);"));
                        self.emit_line("_key ^= (uint32_t)(threadIdx.x & 31);");
                        self.emit_line("_ctr = (_ctr * 0x6C622960u) ^ _key;");
                        self.emit_line(&format!("*((uint32_t*)({rng})) = _ctr + 1;"));
                        self.emit_line("_ctr &= 0x007FFFFFu;");
                        self.emit_line("_ctr |= 0x3F800000u;");
                        self.emit_line(&format!("float {d};"));
                        self.emit_line(&format!("*(uint32_t*)&{d} = _ctr;"));
                        self.emit_line(&format!("{d} -= 1.0f;"));
                        self.indent = self.indent.saturating_sub(1);
                        self.emit_line("}");
                    }
                }
                Ok(())
            }

            // ── Bitwise GEMM (TQ1_0 XOR + POPCNT) ──

            VmInstr::BitwiseGemm { dst, sign_bits, input_sign_bits, scale, .. } => {
                // TQ1_0 ternary XOR + POPCNT batch dot product.
                // dot = (32 - 2 * popcount(sign_bits XOR input_sign_bits)) * scale
                let d = self.reg_name_with_kind(*dst, alloc);
                let sb = self.reg_name_with_kind(*sign_bits, alloc);
                let ib = self.reg_name_with_kind(*input_sign_bits, alloc);
                let sc = self.reg_name_with_kind(*scale, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — XOR result / popcount
                        let rs1 = self.scratch_gpr_names[1]; // %rs1 — constant 32
                        let fs0 = self.scratch_vec_names[0]; // %fs0 — float intermediate
                        // xor → popcnt → compute (32 - 2*pop) → cvt to f32 → mul by scale
                        self.emit_line(&format!("xor.b32 {rs0}, {sb}, {ib};"));
                        self.emit_line(&format!("popc.b32 {rs0}, {rs0};"));
                        self.emit_line(&format!("shl.b32 {rs0}, {rs0}, 1;")); // 2 * pop
                        self.emit_line(&format!("mov.u32 {rs1}, 32;"));
                        self.emit_line(&format!("sub.s32 {rs0}, {rs1}, {rs0};")); // 32 - 2*pop
                        self.emit_line(&format!("cvt.rn.f32.s32 {fs0}, {rs0};"));
                        self.emit_line(&format!("mul.rn.f32 {d}, {fs0}, {sc};"));
                    }
                    _ => {
                        // HIP/Metal: __popc() / __builtin_popcount() intrinsic
                        self.emit_line(&format!(
                            "{d} = (32 - 2 * __popc({sb} ^ {ib})) * {sc};"
                        ));
                    }
                }
                Ok(())
            }

            // ── GPU Sparse Tensor Core 2:4 Structured-Sparsity GEMM (SM80+) ──
            VmInstr::SparseGemm { acc, a_sparse, b_dense, sparse_mask_ptr, m, n, k, .. } => {
                if self.sm_version().is_some_and(|sm| sm >= 80) {
                    let acc_r = self.reg_name_with_kind(*acc, alloc);
                    let a_r = self.reg_name_with_kind(*a_sparse, alloc);
                    let b_r = self.reg_name_with_kind(*b_dense, alloc);
                    let mask_r = self.reg_name_with_kind(*sparse_mask_ptr, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            // mma.sparse.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
                            // {:acc}, {:a}, {:b}, {:mask}
                            self.emit_line(&format!(
                                "mma.sparse.sync.aligned.m{}n{}k{}.row.col.f32.f16.f16.f32 {{{}, {}, {}, {}}};",
                                m, n, k, acc_r, a_r, b_r, mask_r
                            ));
                        }
                        _ => {
                            // HIP/Metal: no native sparse tensor core instruction
                            return Err(CompilerError::CodegenViolation(
                                "SparseGemm: only PTX SM80+ sparse tensor core supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "SparseGemm requires SM80+ Sparse Tensor Core".into()))
                }
            }

            // ── SM100 Native FP4 Matrix Multiply (tcgen05.mma .f4) ──
            VmInstr::NativeFp4Gemm { acc, a, b, scale_a, scale_b, m, n, k, .. } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    let acc_r = self.reg_name_with_kind(*acc, alloc);
                    let a_r = self.reg_name_with_kind(*a, alloc);
                    let b_r = self.reg_name_with_kind(*b, alloc);
                    let sa_r = self.reg_name_with_kind(*scale_a, alloc);
                    let sb_r = self.reg_name_with_kind(*scale_b, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            // tcgen05.mma.synched.cta_group::1.mMnNkK.f4.f4.f32
                            // {:acc}, {:a}, {:b}, {:scale_a}, {:scale_b}
                            self.emit_line(&format!(
                                "tcgen05.mma.synched.cta_group::1.m{}n{}k{}.f4.f4.f32 {{{}, {}, {}, {}, {}}};",
                                m, n, k, acc_r, a_r, b_r, sa_r, sb_r
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "NativeFp4Gemm: only PTX SM100+ tcgen05 supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "NativeFp4Gemm requires SM100+ tcgen05 hardware".into()))
                }
            }

            // ── SM100+ Sparse FP8 Tensor Core GEMM (2:4 稀疏 + FP8 同时生效) ──
            VmInstr::SparseFp8Gemm { acc, a_sparse, b_dense, sparse_mask_ptr, m, n, k, fp8_kind, .. } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    let acc_r = self.reg_name_with_kind(*acc, alloc);
                    let a_r = self.reg_name_with_kind(*a_sparse, alloc);
                    let b_r = self.reg_name_with_kind(*b_dense, alloc);
                    let mask_r = self.reg_name_with_kind(*sparse_mask_ptr, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            // mma.sparse.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
                            // {%acc}, {%a_sparse}, {%b_dense}, %meta
                            let fmt = match fp8_kind {
                                Fp8Kind::E4M3 => "e4m3",
                                Fp8Kind::E5M2 => "e5m2",
                            };
                            self.emit_line(&format!(
                                "mma.sparse.sync.aligned.m{}n{}k{}.row.col.f32.{}.{}.f32 {{{}, {}, {}, {}}};",
                                m, n, k, fmt, fmt, acc_r, a_r, b_r, mask_r
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "SparseFp8Gemm: only PTX SM100+ sparse FP8 tensor core supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "SparseFp8Gemm requires SM100+ Sparse FP8 Tensor Core".into()))
                }
            }

            // ── SM90 Native FP8 Tensor Core GEMM (E4M3/E5M2) ──
            VmInstr::NativeFp8Gemm { acc, a, b, m, n, k, fp8_kind, .. } => {
                if self.sm_version().is_some_and(|sm| sm >= 90) {
                    let acc_r = self.reg_name_with_kind(*acc, alloc);
                    let a_r = self.reg_name_with_kind(*a, alloc);
                    let b_r = self.reg_name_with_kind(*b, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
                            // mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
                            let fmt = match fp8_kind {
                                Fp8Kind::E4M3 => "e4m3",
                                Fp8Kind::E5M2 => "e5m2",
                            };
                            self.emit_line(&format!(
                                "mma.sync.aligned.m{}n{}k{}.row.col.f32.{}.{}.f32 {{{}, {}, {}}};",
                                m, n, k, fmt, fmt, acc_r, a_r, b_r
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "NativeFp8Gemm: only PTX SM90+ FP8 Tensor Core supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "NativeFp8Gemm requires SM90+ FP8 Tensor Core".into()))
                }
            }

            // ── SM100+ Hardware Quantization Dequant (Tensor Core 内建 4-bit decode) ──
            VmInstr::HwQuantDequant { dst, packed_weight, block_scale, global_scale, quant_kind, count, .. } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            let _dst_r = self.reg_name_with_kind(*dst, alloc);
                            let _pw_r = self.reg_name_with_kind(*packed_weight, alloc);
                            let _bs_r = self.reg_name_with_kind(*block_scale, alloc);
                            let _gs_r = self.reg_name_with_kind(*global_scale, alloc);
                            // tcgen05 指令在 Tensor Core 内完成 4-bit decode x scale -> F32
                            // 独立使用时，硬件解量化在 mma 指令内部隐式完成，不需要单独的解量化指令
                            // 主要用途是和 NativeFp4Gemm 配合使用
                            self.emit_line(&format!(
                                "// HwQuantDequant: {} elements, {:?} — hardware 4-bit decode (implicit in tcgen05.mma)",
                                count, quant_kind
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "HwQuantDequant: only PTX SM100+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "HwQuantDequant requires SM100+ hardware".into()))
                }
            }

            // ── SM100+ Tensor Memory (TMEM) Instructions ──

            VmInstr::TmemAlloc { name, bytes } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("tcgen05.alloc.tmem.shared::cta [{}], {};", name, bytes));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "TmemAlloc: only PTX SM100+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "TmemAlloc requires SM100+ Tensor Memory".into()))
                }
            }

            VmInstr::TmemLoad { dst, name, offset, width, dtype } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    let dst_r = self.reg_name_with_kind(*dst, alloc);
                    let off_str = self.offset_to_string(offset, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("tcgen05.ld.tmem {}, [{} + {}];", dst_r, name, off_str));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "TmemLoad: only PTX SM100+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "TmemLoad requires SM100+ Tensor Memory".into()))
                }
            }

            VmInstr::TmemStore { name, offset, src, width, dtype } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    let src_r = self.reg_name_with_kind(*src, alloc);
                    let off_str = self.offset_to_string(offset, alloc);
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("tcgen05.st.tmem [{} + {}], {};", name, off_str, src_r));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "TmemStore: only PTX SM100+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "TmemStore requires SM100+ Tensor Memory".into()))
                }
            }

            // TmemDealloc: SM100+ Tensor Memory deallocation
            VmInstr::TmemDealloc { name } => {
                if self.sm_version().is_some_and(|sm| sm >= 100) {
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!("tcgen05.dealloc_tmem [{}];", name));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "TmemDealloc: only PTX SM100+ supported".into()
                            ))
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "TmemDealloc requires SM100+ Tensor Memory".into()))
                }
            }

            // ── SM90+ Cluster Cooperation (Distributed Shared Memory) ──

            VmInstr::ClusterBarrierInit { name, thread_count } => {
                if self.sm_version().is_some_and(|sm| sm >= 90) {
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!(
                                "barrier.cluster.init _ [{}], {};",
                                name, thread_count
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "ClusterBarrierInit: only PTX SM90+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "ClusterBarrierInit requires SM90+ Cluster".into()))
                }
            }

            VmInstr::ClusterStore { name, offset, src, width: _, dtype } => {
                if self.sm_version().is_some_and(|sm| sm >= 90) {
                    let src_r = self.reg_name_with_kind(*src, alloc);
                    let off_str = self.offset_to_string(offset, alloc);
                    let dtype_str = dtype.gpu_native_type_name().map_err(|_| CompilerError::CodegenViolation(
                        format!("ClusterStore: unsupported dtype {:?} for GPU native store", dtype)
                    ))?;
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!(
                                "st.shared::cluster.{} [{} + {}], {};",
                                dtype_str, name, off_str, src_r
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "ClusterStore: only PTX SM90+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "ClusterStore requires SM90+ Cluster DSMEM".into()))
                }
            }

            VmInstr::ClusterLoad { dst, name, offset, width: _, dtype } => {
                if self.sm_version().is_some_and(|sm| sm >= 90) {
                    let dst_r = self.reg_name_with_kind(*dst, alloc);
                    let off_str = self.offset_to_string(offset, alloc);
                    let dtype_str = dtype.gpu_native_type_name().map_err(|_| CompilerError::CodegenViolation(
                        format!("ClusterLoad: unsupported dtype {:?} for GPU native load", dtype)
                    ))?;
                    match self.dialect {
                        GpuDialect::Ptx { .. } => {
                            self.emit_line(&format!(
                                "ld.shared::cluster.{} {}, [{} + {}];",
                                dtype_str, dst_r, name, off_str
                            ));
                        }
                        _ => {
                            return Err(CompilerError::CodegenViolation(
                                "ClusterLoad: only PTX SM90+ supported".into()
                            ));
                        }
                    }
                    Ok(())
                } else {
                    Err(CompilerError::CodegenViolation(
                        "ClusterLoad requires SM90+ Cluster DSMEM".into()))
                }
            }

            // ── Layer 6: Debug Instrumentation — NOP on GPU ──
            VmInstr::DebugBreakpoint { .. } | VmInstr::DebugMarker { .. }
            | VmInstr::DebugProbe { .. } | VmInstr::DebugBreakIf { .. } => {
                Ok(())
            }

            VmInstr::MemCopy { dst, src, bytes, dtype: _, guard: _, effect: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                for off in (0..*bytes).step_by(8) {
                    let remaining = *bytes - off;
                    if remaining >= 8 {
                        self.emit_line(&format!("ld.global.u64 %rd_tmp_mc, [{}+{}];", s, off));
                        self.emit_line(&format!("st.global.u64 [{}+{}], %rd_tmp_mc;", d, off));
                    } else if remaining >= 4 {
                        self.emit_line(&format!("ld.global.u32 %r_tmp_mc, [{}+{}];", s, off));
                        self.emit_line(&format!("st.global.u32 [{}+{}], %r_tmp_mc;", d, off));
                    } else {
                        for b in 0..remaining {
                            self.emit_line(&format!("ld.global.u8 %rb_tmp_mc, [{}+{}];", s, off + b));
                            self.emit_line(&format!("st.global.u8 [{}+{}], %rb_tmp_mc;", d, off + b));
                        }
                    }
                }
                Ok(())
            }

            // ── REQ-VR-005~010: 缺失指令 — GPU ISA lowering ──

            VmInstr::VecShuffle { dst, src, mask, width: _ } => {
                // REQ-VR11: GPU VecShuffle — byte-level permutation within 32-bit scalar.
                // GPU SIMT model: each VReg holds one f32 (4 bytes) per thread.
                // VecShuffle permutes the 4 bytes within that value.
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match mask {
                    VecShuffleMask::Const(bytes) => {
                        // Build prmt control byte from mask bytes.
                        // prmt.b32 control: each nibble (4 bits) selects a source byte:
                        //   bits[3:0] = dst_byte[0] source idx
                        //   bits[7:4] = dst_byte[1] source idx
                        //   bits[11:8] = dst_byte[2] source idx
                        //   bits[15:12] = dst_byte[3] source idx
                        let ctrl: u32 = bytes.iter().enumerate().take(4).map(|(i, &b)| {
                            ((b as u32) & 0x3) << (i * 4)
                        }).sum();
                        // Byte-select bits (4:0) set to 0x4 for prmt "copy from a,b zero-extend"
                        let prmt_val = ctrl | 0x4444;
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                let rs0 = self.scratch_gpr_names[0];
                                self.emit_line(&format!("mov.u32 {rs0}, {};", prmt_val));
                                self.emit_line(&format!("prmt.b32 {d}, {s}, {s}, {rs0};"));
                            }
                            GpuDialect::Hip { .. } => {
                                // HIP: rearrange bytes within uint32 using shifts and masks.
                                // Build a per-byte extraction and OR together.
                                self.emit_line("{");
                                self.emit_line(&format!("  unsigned int _src = __builtin_amdgcn_as_uint({s});"));
                                self.emit_line("  unsigned int _result = 0;");
                                for i in 0..4 {
                                    let b = bytes.get(i).copied().unwrap_or(i as u8);
                                    if b < 4 {
                                        self.emit_line(&format!(
                                            "  _result |= ((_src >> ({} * 8)) & 0xFF) << ({} * 8);",
                                            b, i
                                        ));
                                    }
                                }
                                self.emit_line(&format!("  {d} = __builtin_amdgcn_as_float(_result);"));
                                self.emit_line("}");
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line("{");
                                self.emit_line(&format!("  unsigned int _src = as_type<unsigned int>({s});"));
                                self.emit_line("  unsigned int _result = 0;");
                                for i in 0..4 {
                                    let b = bytes.get(i).copied().unwrap_or(i as u8);
                                    if b < 4 {
                                        self.emit_line(&format!(
                                            "  _result |= ((_src >> ({} * 8)) & 0xFF) << ({} * 8);",
                                            b, i
                                        ));
                                    }
                                }
                                self.emit_line(&format!("  {d} = as_type<float>(_result);"));
                                self.emit_line("}");
                            }
                        }
                        Ok(())
                    }
                    VecShuffleMask::Dynamic { ctrl } => {
                        let c = self.reg_name_with_kind(*ctrl, alloc);
                        match self.dialect {
                            GpuDialect::Ptx { .. } => {
                                // prmt.b32: each nibble of ctrl selects source byte from s (a=b=s).
                                self.emit_line(&format!("prmt.b32 {d}, {s}, {s}, {c};"));
                            }
                            GpuDialect::Hip { .. } => {
                                // HIP: extract per-byte using ctrl register bit pattern.
                                // ctrl encodes per-nibble source byte indices.
                                self.emit_line("{");
                                self.emit_line(&format!("  unsigned int _src = __builtin_amdgcn_as_uint({s});"));
                                self.emit_line("  unsigned int _result = 0;");
                                self.emit_line(&format!("  unsigned int _ctrl = {c};"));
                                self.emit_line("  _result |= ((_src >> (((_ctrl      ) & 3) * 8)) & 0xFF) << 0;");
                                self.emit_line("  _result |= ((_src >> (((_ctrl >>  4) & 3) * 8)) & 0xFF) << 8;");
                                self.emit_line("  _result |= ((_src >> (((_ctrl >>  8) & 3) * 8)) & 0xFF) << 16;");
                                self.emit_line("  _result |= ((_src >> (((_ctrl >> 12) & 3) * 8)) & 0xFF) << 24;");
                                self.emit_line(&format!("  {d} = __builtin_amdgcn_as_float(_result);"));
                                self.emit_line("}");
                            }
                            GpuDialect::Metal { .. } => {
                                self.emit_line("{");
                                self.emit_line(&format!("  unsigned int _src = as_type<unsigned int>({s});"));
                                self.emit_line("  unsigned int _result = 0;");
                                self.emit_line(&format!("  unsigned int _ctrl = {c};"));
                                self.emit_line("  _result |= ((_src >> (((_ctrl      ) & 3) * 8)) & 0xFF) << 0;");
                                self.emit_line("  _result |= ((_src >> (((_ctrl >>  4) & 3) * 8)) & 0xFF) << 8;");
                                self.emit_line("  _result |= ((_src >> (((_ctrl >>  8) & 3) * 8)) & 0xFF) << 16;");
                                self.emit_line("  _result |= ((_src >> (((_ctrl >> 12) & 3) * 8)) & 0xFF) << 24;");
                                self.emit_line(&format!("  {d} = as_type<float>(_result);"));
                                self.emit_line("}");
                            }
                        }
                        Ok(())
                    }
                }
            }

            VmInstr::VecExtractLane { dst, src, lane, dtype: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("mov.f32 {d}, {s}.x[{}];", lane));
                        Ok(())
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{d} = __builtin_amdgcn_ds_swizzle({s}, {lane});"));
                        Ok(())
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = {s}[{}];", lane));
                        Ok(())
                    }
                }
            }

            VmInstr::VecInsertLane { dst, src_vec, src_scalar, lane, dtype: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let sv = self.reg_name_with_kind(*src_vec, alloc);
                let ss = self.reg_name_with_kind(*src_scalar, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Copy src_vec to dst first, then overwrite the lane
                        let tmp = self.scratch_vec_names[0];
                        self.emit_line(&format!("mov.f32 {tmp}, {sv};"));
                        self.emit_line(&format!("mov.f32 {tmp}.x[{}], {ss};", lane));
                        self.emit_line(&format!("mov.f32 {d}, {tmp};"));
                        Ok(())
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{d} = {sv};"));
                        self.emit_line(&format!("((float*)&{d})[{lane}] = {ss};"));
                        Ok(())
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = {sv};"));
                        self.emit_line(&format!("{d}[{lane}] = {ss};"));
                        Ok(())
                    }
                }
            }

            VmInstr::VecLoadConst { dst, values, dtype: _, width: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0];
                        let fs0 = self.scratch_vec_names[0];
                        for (i, &bits) in values.iter().enumerate().take(4) {
                            self.emit_line(&format!("mov.u32 {rs0}, 0x{:08x};", bits));
                            self.emit_line(&format!("mov.f32 {fs0}, {rs0};"));
                            self.emit_line(&format!("mov.f32 {d}.x[{}], {fs0};", i));
                        }
                        Ok(())
                    }
                    GpuDialect::Hip { .. } => {
                        for (i, &bits) in values.iter().enumerate().take(4) {
                            self.emit_line(&format!("*((__attribute__((address_space(5))) uint*)&{d} + {}) = 0x{:08x}u;", i, bits));
                        }
                        Ok(())
                    }
                    GpuDialect::Metal { .. } => {
                        let vals: Vec<String> = values.iter().take(4).map(|&bits| {
                            let f = f32::from_bits(bits);
                            format!("{}", f)
                        }).collect();
                        let padded = if vals.len() < 4 {
                            let mut v = vals;
                            while v.len() < 4 { v.push("0.0".to_string()); }
                            v
                        } else { vals };
                        self.emit_line(&format!("float4 {d} = float4({}, {}, {}, {});", padded[0], padded[1], padded[2], padded[3]));
                        Ok(())
                    }
                }
            }

            VmInstr::AtomicCAS { dst, ptr, expected, desired, elem_width, success_order, failure_order: _ } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let p = self.reg_name_with_kind(*ptr, alloc);
                let e = self.reg_name_with_kind(*expected, alloc);
                let w = self.reg_name_with_kind(*desired, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let scope = match success_order {
                            MemOrdering::Relaxed => ".relaxed.gpu",
                            MemOrdering::Acquire => ".acquire.gpu",
                            MemOrdering::Release => ".release.gpu",
                            MemOrdering::AcqRel => ".acq_rel.gpu",
                            MemOrdering::SeqCst => ".sys",
                        };
                        let ty = if *elem_width == 8 { "u64" } else { "u32" };
                        self.emit_line(&format!("atom{scope}.cas.{ty} {d}, [{p}], {e}, {w};"));
                        Ok(())
                    }
                    _ => Err(CompilerError::CodegenViolation(
                        format!("AtomicCAS: {:?} not yet supported", self.dialect)))
                }
            }

            VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, seq_meta_stride } => {
                let _ = (dst, token_index, seq_meta_base, num_seqs, seq_meta_stride);
                Err(CompilerError::CodegenViolation(
                    "SeqIdLookup: GPU cumsum search not yet implemented".into()))
            }

            // ── §1.6 分布式通信 VmInstr (REQ-VR-014, feature = "nccl") ──

            #[cfg(feature = "nccl")]
            VmInstr::AllReduceChunk { sendbuf, recvbuf, count, dtype, op, rank, world_size, chunk_idx } => {
                // GPU PTX: call to all_reduce_chunk kernel
                // Parameters passed through PTX calling convention
                let sb = self.reg_name_with_kind(*sendbuf, alloc);
                let rb = self.reg_name_with_kind(*recvbuf, alloc);
                let cnt = self.reg_name_with_kind(*count, alloc);
                let rk = self.reg_name_with_kind(*rank, alloc);
                let ws = self.reg_name_with_kind(*world_size, alloc);
                let ci = self.reg_name_with_kind(*chunk_idx, alloc);

                let dtype_val = match dtype {
                    CommDType::Fp32 => 0u32, CommDType::Fp16 => 1,
                    CommDType::Bf16 => 2, CommDType::Fp8 => 3, CommDType::Int8 => 4,
                };
                let op_val = match op {
                    ReduceOp::Sum => 0u32, ReduceOp::Max => 1,
                    ReduceOp::Min => 2, ReduceOp::Prod => 3, ReduceOp::LogSum => 4,
                };

                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line("{");
                        self.emit_line("    .param .b64 param_sendbuf;");
                        self.emit_line("    .param .b64 param_recvbuf;");
                        self.emit_line("    .param .b64 param_count;");
                        self.emit_line("    .param .b32 param_dtype;");
                        self.emit_line("    .param .b32 param_op;");
                        self.emit_line("    .param .b32 param_rank;");
                        self.emit_line("    .param .b32 param_world_size;");
                        self.emit_line("    .param .b32 param_chunk_idx;");
                        self.emit_line(&format!("    st.param.b64 [param_sendbuf], {};", sb));
                        self.emit_line(&format!("    st.param.b64 [param_recvbuf], {};", rb));
                        self.emit_line(&format!("    st.param.b64 [param_count], {};", cnt));
                        self.emit_line(&format!("    st.param.b32 [param_dtype], {};", dtype_val));
                        self.emit_line(&format!("    st.param.b32 [param_op], {};", op_val));
                        self.emit_line(&format!("    st.param.b32 [param_rank], {};", rk));
                        self.emit_line(&format!("    st.param.b32 [param_world_size], {};", ws));
                        self.emit_line(&format!("    st.param.b32 [param_chunk_idx], {};", ci));
                        self.emit_line("    call.uni gllm_nccl_all_reduce_chunk, (param_sendbuf, param_recvbuf, param_count, param_dtype, param_op, param_rank, param_world_size, param_chunk_idx);");
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "gllm_nccl_all_reduce_chunk({}, {}, {}, {}, {}, {}, {}, {});",
                            sb, rb, cnt, dtype_val, op_val, rk, ws, ci));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "AllReduceChunk: Metal GPU call stub not yet implemented".into()));
                    }
                }
                Ok(())
            }

            #[cfg(feature = "nccl")]
            VmInstr::CommBarrier { barrier_id, thread_count } => {
                let tc = self.reg_name_with_kind(*thread_count, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("bar.sync {}, {};", barrier_id, tc));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("__syncthreads(); // barrier_id={}", barrier_id));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!(
                            "threadgroup_barrier(mem_flags::mem_threadgroup); // barrier_id={}",
                            barrier_id));
                    }
                }
                Ok(())
            }

            #[cfg(feature = "nccl")]
            VmInstr::NvlinkAsyncCopy { dst, src, len, lane } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let s = self.reg_name_with_kind(*src, alloc);
                let l = self.reg_name_with_kind(*len, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [{}], [{}], {}, [mbarrier_{}];",
                            d, s, l, lane));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "// NvlinkAsyncCopy: dst={} src={} len={} lane={}",
                            d, s, l, lane));
                        self.emit_line("__builtin_amdgcn_global_load_tr_b64_i32(");
                        self.emit_line(&format!("    (const ulong*){}", d));
                        self.emit_line(&format!("    (const ulong*){}", s));
                        self.emit_line(&format!("    {}", l));
                        self.emit_line(");");
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "NvlinkAsyncCopy: Metal GPU async copy not yet implemented".into()));
                    }
                }
                Ok(())
            }

            // ── §8 分布式分页 VmInstr (REQ-DP-010, feature = "nccl") ──

            #[cfg(feature = "nccl")]
            VmInstr::RemotePageLookup { dst, seq_id, page_index, routing_table_base } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let sid = self.reg_name_with_kind(*seq_id, alloc);
                let pi = self.reg_name_with_kind(*page_index, alloc);
                let rtb = self.reg_name_with_kind(*routing_table_base, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(
                            "call.uni _(gllm_dp_remote_page_lookup), ({d}, {sid}, {pi}, {rtb});"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "{d} = gllm_dp_remote_page_lookup({sid}, {pi}, {rtb});"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "RemotePageLookup: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::P2pPageFetch { local_buf, peer_buf, page_size, barrier } => {
                let lb = self.reg_name_with_kind(*local_buf, alloc);
                let pb = self.reg_name_with_kind(*peer_buf, alloc);
                let ps = self.reg_name_with_kind(*page_size, alloc);
                let br = self.reg_name_with_kind(*barrier, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(
                            "call.uni gllm_dp_p2p_page_fetch, ({lb}, {pb}, {ps}, {br});"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "gllm_dp_p2p_page_fetch({lb}, {pb}, {ps}, {br});"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "P2pPageFetch: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetch { local_buf, remote_addr, rkey, page_size, sq_desc, doorbell, cq_addr } => {
                let lb = self.reg_name_with_kind(*local_buf, alloc);
                let ra = self.reg_name_with_kind(*remote_addr, alloc);
                let rk = self.reg_name_with_kind(*rkey, alloc);
                let ps = self.reg_name_with_kind(*page_size, alloc);
                let sq = self.reg_name_with_kind(*sq_desc, alloc);
                let db = self.reg_name_with_kind(*doorbell, alloc);
                let cq = self.reg_name_with_kind(*cq_addr, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(
                            "call.uni gllm_dp_rdma_page_fetch, ({lb}, {ra}, {rk}, {ps}, {sq}, {db}, {cq});"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "gllm_dp_rdma_page_fetch({lb}, {ra}, {rk}, {ps}, {sq}, {db}, {cq});"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "RdmaPageFetch: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetchCompressed { local_buf, scratch_buf, page_size, remote_addr, rkey, sq_desc, doorbell, cq_addr, quant_scheme, compress_algorithm } => {
                let lb = self.reg_name_with_kind(*local_buf, alloc);
                let sb = self.reg_name_with_kind(*scratch_buf, alloc);
                let ps = self.reg_name_with_kind(*page_size, alloc);
                let ra = self.reg_name_with_kind(*remote_addr, alloc);
                let rk = self.reg_name_with_kind(*rkey, alloc);
                let sq = self.reg_name_with_kind(*sq_desc, alloc);
                let db = self.reg_name_with_kind(*doorbell, alloc);
                let cq = self.reg_name_with_kind(*cq_addr, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(
                            "call.uni gllm_dp_rdma_page_fetch_compressed, ({lb}, {sb}, {ps}, {ra}, {rk}, {sq}, {db}, {cq}, {quant_scheme}, {compress_algorithm});"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "gllm_dp_rdma_page_fetch_compressed({lb}, {sb}, {ps}, {ra}, {rk}, {sq}, {db}, {cq}, {quant_scheme}, {compress_algorithm});"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "RdmaPageFetchCompressed: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageAttn { q_buf, k_remote_buf, v_remote_buf, output_buf, shared_buf, barrier, tile_bytes } => {
                let qb = self.reg_name_with_kind(*q_buf, alloc);
                let kb = self.reg_name_with_kind(*k_remote_buf, alloc);
                let vb = self.reg_name_with_kind(*v_remote_buf, alloc);
                let ob = self.reg_name_with_kind(*output_buf, alloc);
                let shb = self.reg_name_with_kind(*shared_buf, alloc);
                let br = self.reg_name_with_kind(*barrier, alloc);
                let tb = self.reg_name_with_kind(*tile_bytes, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!(
                            "call.uni gllm_dp_remote_page_attn, ({qb}, {kb}, {vb}, {ob}, {shb}, {br}, {tb});"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!(
                            "gllm_dp_remote_page_attn({qb}, {kb}, {vb}, {ob}, {shb}, {br}, {tb});"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "RemotePageAttn: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationLock { dst, entry_addr } => {
                let d = self.reg_name_with_kind(*dst, alloc);
                let ea = self.reg_name_with_kind(*entry_addr, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("atom.global.cas.b64 {d}, [{ea}], 0, 1;"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("auto {d} = atomicCAS(reinterpret_cast<unsigned long long*>({ea}), 0ULL, 1ULL);"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "PageMigrationLock: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationUnlock { entry_addr } => {
                let ea = self.reg_name_with_kind(*entry_addr, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("st.global.u64 [{ea}], 0;"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("atomicExch(reinterpret_cast<unsigned long long*>({ea}), 0ULL);"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "PageMigrationUnlock: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
            #[cfg(feature = "nccl")]
            VmInstr::PageLocationUpdate { entry_addr, new_location, new_state } => {
                let ea = self.reg_name_with_kind(*entry_addr, alloc);
                let nl = self.reg_name_with_kind(*new_location, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        self.emit_line(&format!("st.global.u32 [{ea}], {{ {nl}, {new_state} }};  // PageLocationUpdate"));
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("atomicExch(reinterpret_cast<unsigned int*>({ea}), {nl});  // PageLocationUpdate state={new_state}"));
                    }
                    GpuDialect::Metal { .. } => {
                        return Err(CompilerError::CodegenViolation(
                            "PageLocationUpdate: Metal GPU not yet implemented".into()));
                    }
                }
                Ok(())
            }
        }
    }
}
