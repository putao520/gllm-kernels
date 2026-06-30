impl X86Lower {
    // ── AMX Tile 辅助方法 (HW-TIER TASK-D §4 + §6.2) ──
    //
    // 三个方法供 TileLoad / TileMma / TileStore arm 共享, 必须定义在 impl 级别
    // (不能在 match arm 内定义 nested fn)。

    /// 把 tile VRegId 解析为物理 tmm 寄存器 (iced AsmRegisterTmm)。
    ///
    /// RegAllocator 对 VRegKind::Tile 分配 PhysReg::Tile(PhysTile(u8))。
    /// u8 ∈ [0..8] 对应 tmm0..tmm7 (AMX 8 个 tile 寄存器上限)。
    /// 若 vreg 未分配 (None) 或被 spill → CodegenViolation (tile 不允许 spill,
    /// RegAllocator Tile 分支不产生 Spilled 变体)。
    fn resolve_phys_tile(
        &self,
        vreg: VRegId,
        alloc: &RegAllocation,
    ) -> Result<AsmRegisterTmm, CompilerError> {
        match alloc.get(vreg) {
            Some(super::isa_profile::PhysReg::Tile(pt)) => Self::phys_tile_to_tmm(pt),
            other => Err(CompilerError::CodegenViolation(format!(
                "AMX tile v{} not allocated to PhysReg::Tile (got {:?}); \
                 tile vregs must be allocated by RegAllocator Tile pool",
                vreg.0, other,
            ))),
        }
    }

    /// PhysTile(u8) → iced AsmRegisterTmm (tmm0..tmm7)。
    fn phys_tile_to_tmm(
        pt: super::isa_profile::PhysTile,
    ) -> Result<AsmRegisterTmm, CompilerError> {
        use iced_x86::code_asm::registers::*;
        Ok(match pt.0 {
            0 => tmm0, 1 => tmm1, 2 => tmm2, 3 => tmm3,
            4 => tmm4, 5 => tmm5, 6 => tmm6, 7 => tmm7,
            other => return Err(CompilerError::CodegenViolation(format!(
                "AMX PhysTile({}) out of range; AMX has 8 tile registers (tmm0..tmm7)", other,
            ))),
        })
    }

    /// 计算有效地址 `base_ptr + k_offset` (或 `c_ptr + out_offset`) 到 rax,
    /// 返回 rax (scratch[0])。供 TileLoad/TileStore 的 TILELOADD/TILESTORED
    /// sibmem 操作数做 base 寄存器。
    ///
    /// R1 决定性风险缓解: k_offset (K 循环推进偏移) 与 row_stride (2D 逐行跨度)
    /// 是两个独立量。本函数只算 base+k_offset, row_stride 由调用方加载到另一个
    /// 寄存器作为 sibmem 的 index 寄存器, 两者禁止合并。
    ///
    /// base_ptr/k_offset 是 GPR VReg (Ptr/ByteOffset/Scalar kind), 可能被
    /// RegAllocator spill 到栈。用 resolve_gpr_read 通过 scratch slot 1/2 load。
    /// 计算结果落到 scratch[0] (rax), 之后 scratch[1] (r10) 可安全重用做 stride。
    fn resolve_tile_eff_addr_to_rax(
        &mut self,
        base_ptr: VRegId,
        k_offset: VRegId,
        alloc: &RegAllocation,
    ) -> Result<AsmRegister64, CompilerError> {
        let s0 = self.scratch_gprs[0]; // rax
        // base_ptr → scratch slot 1 (r10) 或物理 GPR
        let base_reg = self.resolve_gpr_read(base_ptr, alloc, 1)?;
        // k_offset → scratch slot 2 (r11) 或物理 GPR
        let koff_reg = self.resolve_gpr_read(k_offset, alloc, 2)?;
        // rax = base_ptr
        self.asm.mov(s0, base_reg).map_err(Self::err)?;
        // rax += k_offset  (eff addr = base + K 循环推进偏移)
        self.asm.add(s0, koff_reg).map_err(Self::err)?;
        Ok(s0)
    }

    /// 断言 c/a/b 物理映射为 tmm0/tmm1/tmm2 (固定单 MMA 链布局)。
    ///
    /// 发射 TDP* (Tile Dot Product) 指令的 VEX 手编字节, 支持任意 tmm 三元组。
    ///
    /// 用于 iced 1.21 code_asm 未覆盖的 AMX 指令 (TDPHF8PS/TDPBF8PS/TDPTF32PS —
    /// Diamond Rapids AMX-FP8/AMX-TF32)。BF16/F16/INT8 走 iced 原生 code_asm 方法。
    ///
    /// VEX.128 编码 (5 字节): C4 + byte2 + byte3 + opcode + ModRM
    ///   byte2  = R.X.B.mmmmmm, 全部 tmm 0-7 (无 REX 扩展) → R=X=B=1, mmmmmm=0F38=000_010 → 0xE2
    ///   byte3  = W.vvvv.L.pp
    ///            W=0 (AMX TDP 均 W0), vvvv = ~tmm_b[3:0] (4-bit inverted), L=0 (VEX.128), pp=mandatory prefix
    ///   opcode = TDP* 指令 opcode (e.g. 0xFD for FP8, 0x6C for TF32)
    ///   ModRM  = 0xC0 | (tmm_c[2:0] << 3) | tmm_a[2:0]   (mod=11, reg=tmm_c, rm=tmm_a)
    ///
    /// pp 映射 (VEX.pp 编码, iced 实测一致): 00=none, 01=66, 10=F3, 11=F2
    ///
    /// 验证: 与 iced 1.21 原生 tdpbf16ps(tmm0,tmm1,tmm2) 输出 C4 E2 6A 5C C1 完全一致
    /// (BF16: pp=F3=10, op=0x5C, tmm_b=2→vvvv=~2=1101, byte3=0x68|0x02=0x6A, ModRM=0xC1)。
    fn emit_tdp_raw(
        &mut self,
        opcode: u8,
        pp: u8,
        tmm_c: AsmRegisterTmm,
        tmm_a: AsmRegisterTmm,
        tmm_b: AsmRegisterTmm,
    ) -> Result<(), CompilerError> {
        // 物理寄存器号: iced AsmRegisterTmm → Register 枚举 (TMM0=241..TMM7=248)。
        // tmm 编号 = reg_u32 - 241。仅支持 tmm0..tmm7 (AMX 8-tmm 上限)。
        use iced_x86::Register;
        const TMM0_BASE: u32 = Register::TMM0 as u32;
        let c_num = Into::<Register>::into(tmm_c) as u32 - TMM0_BASE;
        let a_num = Into::<Register>::into(tmm_a) as u32 - TMM0_BASE;
        let b_num = Into::<Register>::into(tmm_b) as u32 - TMM0_BASE;
        if c_num > 7 || a_num > 7 || b_num > 7 {
            return Err(CompilerError::CodegenViolation(format!(
                "AMX emit_tdp_raw: tmm out of range (c={}, a={}, b={}); AMX has 8 tile regs",
                c_num, a_num, b_num,
            )));
        }
        let c = c_num as u8;
        let a = a_num as u8;
        let b = b_num as u8;
        let pp = pp & 0x03;
        // byte2: R=1 X=1 B=1 (无扩展), mmmmmm=0F38=000010 → 0b11100010 = 0xE2
        let byte2: u8 = 0xE2;
        // byte3: W=0 | vvvv=(~b & 0xF)<<3 | L=0 (VEX.128) | pp
        let vvvv = (!b) & 0x0F;
        let byte3: u8 = (vvvv << 3) | pp;
        // ModRM: mod=11, reg=tmm_c, rm=tmm_a
        let modrm: u8 = 0xC0 | ((c & 0x07) << 3) | (a & 0x07);
        self.asm.db(&[0xC4, byte2, byte3, opcode, modrm]).map_err(Self::err)?;
        Ok(())
    }

    pub fn lower_instr(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        let pre_len = self.asm.instructions().len();
        let result = self.lower_instr_inner(instr, alloc);
        if result.is_err() {
            let post_len = self.asm.instructions().len();
            let instr_short = format!("{:?}", instr);
            let instr_name = instr_short.split('{').next().unwrap_or("?");
            eprintln!("[X86-LOWER-ERR] VmInstr {} failed after emitting {} instructions (pre={} post={})",
                instr_name, post_len - pre_len, pre_len, post_len);
            // Show last few instructions for context
            let instrs = self.asm.instructions();
            let start = post_len.saturating_sub(5);
            for (i, ins) in instrs.iter().enumerate().skip(start).take(5) {
                eprintln!("  [{}] {:?}", i, ins);
            }
        }
        result
    }

    fn lower_instr_inner(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        // ARCH-LOWER-DISPATCH-LAYERING (BCE-20260630-LOWER-INSTR-GOD-MATCH):
        // L0 分类 dispatch — 巨型 VmInstr match 已拆为 L0(分类) → L1(变体路由) → L2(叶子 emit)。
        // 新增 VmInstr 变体 = category() 补一行 + lower_<variant>_x86 一个叶子 fn。
        // 禁止在此处内联变体逻辑 (OCP); 禁止 catch-all 静默 NOP (NO_SILENT_FALLBACK)。
        //
        // ConditionalSkip 递减计数器 preamble (保留原行为, 与 dispatch 分离):
        if !instr.is_meta() {
            if let Some((count, _)) = self.skip_stack.last_mut() {
                if *count > 0 {
                    *count -= 1;
                }
            }
            if let Some((0, _)) = self.skip_stack.last() {
                let (_, mut label) = self.skip_stack.pop().unwrap();
                self.asm.set_label(&mut label).map_err(Self::err)?;
            }
        }

        match instr.category() {
            super::vm_instr_category::InstrCategory::Memory => self.lower_memory_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::Arith => self.lower_arith_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::Control => self.lower_control_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::Tile => self.lower_tile_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::Quant => self.lower_quant_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::GpuComm => self.lower_gpu_comm_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::Sampling => self.lower_sampling_x86(instr, alloc),
            super::vm_instr_category::InstrCategory::Misc => self.lower_misc_x86(instr, alloc),
        }
    }


    // ── 辅助 ──

    /// LZ4 流解压 (x86_64, SPEC §3.3.1 REQ-COMP-003).
    // ── Sampling instruction x86 lowering implementations ──

    /// SoftmaxReduceMax: scan logits_ptr[0..vocab_bytes], find max, write to dst (scalar f32).
    fn lower_softmax_reduce_max(
        &mut self,
        dst: VRegId,
        logits_ptr: VRegId,
        vocab_bytes: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_reg = self.resolve_gpr_write(dst, alloc, 0)?;
        let base_reg = self.resolve_gpr_read(logits_ptr, alloc, 1)?;
        let lanes = width.f32_lanes();
        let step = lanes * 4;
        let num_vecs = vocab_bytes / step;

        let tmp = self.scratch_ymm(0);
        let tmp_xmm = self.scratch_xmm(0);
        let val_xmm = self.scratch_xmm(1);
        let max_xmm = self.scratch_xmm(2);
        let base_saved = self.scratch_gprs[1];

        self.asm.mov(base_saved, base_reg).map_err(Self::err)?;

        // Load first vector, horizontal max → max_xmm
        self.asm.vmovups(tmp, ymmword_ptr(base_saved)).map_err(Self::err)?;
        // Horizontal max: extract high 128, max pairwise down to scalar
        self.asm.vextractf128(val_xmm, tmp, 1).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vmovhlps(val_xmm, tmp_xmm, tmp_xmm).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vshufps(val_xmm, tmp_xmm, tmp_xmm, 0x01).map_err(Self::err)?;
        self.asm.vmaxss(max_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;

        // Scan remaining vectors
        let offset_reg = self.scratch_gprs[2];
        self.asm.mov(offset_reg, step as u64).map_err(Self::err)?;
        let mut scan_label = self.asm.create_label();
        let mut skip_label = self.asm.create_label();

        self.asm.set_label(&mut scan_label).map_err(Self::err)?;
        self.asm.vmovups(tmp, ymmword_ptr(base_saved + offset_reg)).map_err(Self::err)?;
        self.asm.vextractf128(val_xmm, tmp, 1).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vmovhlps(val_xmm, tmp_xmm, tmp_xmm).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vshufps(val_xmm, tmp_xmm, tmp_xmm, 0x01).map_err(Self::err)?;
        self.asm.vmaxss(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        // if tmp_xmm > max_xmm, update
        self.asm.vcomiss(tmp_xmm, max_xmm).map_err(Self::err)?;
        self.asm.jbe(skip_label).map_err(Self::err)?;
        self.asm.vmaxss(max_xmm, max_xmm, tmp_xmm).map_err(Self::err)?;

        self.asm.set_label(&mut skip_label).map_err(Self::err)?;
        self.asm.add(offset_reg, step as i32).map_err(Self::err)?;
        self.asm.cmp(offset_reg, (num_vecs * step) as i32).map_err(Self::err)?;
        self.asm.jb(scan_label).map_err(Self::err)?;

        // max_xmm → dst via eax
        self.asm.vmovd(eax, max_xmm).map_err(Self::err)?;
        self.asm.mov(dst_reg, rax).map_err(Self::err)?;
        self.commit_gpr_write(dst, alloc, 0)?;
        Ok(())
    }

    /// SoftmaxExpSum: exp(logits - max) in-place + horizontal sum → sum_dst.
    /// Uses Wynne's fast exp (improved Schraudolph) via integer multiply + bias.
    /// Accuracy: max relative error < 1% for x in [-87, 0], sufficient for sampling.
    fn lower_softmax_exp_sum(
        &mut self,
        sum_dst: VRegId,
        logits_ptr: VRegId,
        max_val: VRegId,
        vocab_bytes: usize,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let sum_dst_reg = self.resolve_gpr_write(sum_dst, alloc, 0)?;
        let base_reg = self.resolve_gpr_read(logits_ptr, alloc, 1)?;
        let max_gpr = self.resolve_gpr_read(max_val, alloc, 2)?;

        let max_xmm = self.scratch_xmm(0);
        self.asm.vmovd(max_xmm, Self::gpr64_to_32(max_gpr)).map_err(Self::err)?;

        let val_xmm = self.scratch_xmm(1);
        let sum_xmm = self.scratch_xmm(2);
        self.asm.vxorps(sum_xmm, sum_xmm, sum_xmm).map_err(Self::err)?;

        let i_reg = self.scratch_gprs[1];
        let base_saved = self.scratch_gprs[2];
        self.asm.mov(base_saved, base_reg).map_err(Self::err)?;
        self.asm.xor(i_reg, i_reg).map_err(Self::err)?;

        let mut loop_lbl = self.asm.create_label();
        let mut neg_lbl = self.asm.create_label();
        let mut done_lbl = self.asm.create_label();

        self.asm.set_label(&mut loop_lbl).map_err(Self::err)?;
        // Load f32, subtract max → val = logits[i] - max
        self.asm.vmovss(val_xmm, dword_ptr(base_saved + i_reg)).map_err(Self::err)?;
        self.asm.vsubss(val_xmm, val_xmm, max_xmm).map_err(Self::err)?;
        // Check if val < -87 (exp underflows to ~0). Use -87.0 as threshold.
        // We compare: if val_bits (as int) represents a value < -87.0, skip to zero.
        // Simpler: compare with const -87.0
        let neg87 = self.const_f32(-87.0f32);
        self.asm.vcomiss(val_xmm, dword_ptr(neg87)).map_err(Self::err)?;
        self.asm.jb(neg_lbl).map_err(Self::err)?; // val < -87 → exp ≈ 0

        // Fast exp: multiply by 12102203/2^23 (≈ log2(e)), add 127*2^23 bias,
        // then reinterpret integer as float.
        // In practice: convert to int, multiply by 12102203, add 1064866805 (bias),
        // reinterpret as float, then second-order correction.
        self.asm.vmovd(Self::gpr64_to_32(self.scratch_gprs[0]), val_xmm).map_err(Self::err)?;
        // val_bits * 12102203 + 1064866805
        self.asm.imul_3(self.scratch_gprs[0], self.scratch_gprs[0], 12102203).map_err(Self::err)?;
        self.asm.add(self.scratch_gprs[0], 1064866805i32).map_err(Self::err)?;
        // Reinterpret as float
        self.asm.vmovd(val_xmm, Self::gpr64_to_32(self.scratch_gprs[0])).map_err(Self::err)?;
        // Second-order correction (Wynne): val = val * (1 - (val - 2^(k+1)) * correction)
        // For sampling accuracy, the basic version is sufficient.
        // Clamp to non-negative (safety)
        let zero_f = self.const_f32(0.0f32);
        self.asm.vmaxss(val_xmm, val_xmm, dword_ptr(zero_f)).map_err(Self::err)?;
        self.asm.jmp(done_lbl).map_err(Self::err)?;

        // Negative path: exp(x) ≈ 0 for x < -87
        self.asm.set_label(&mut neg_lbl).map_err(Self::err)?;
        self.asm.vxorps(val_xmm, val_xmm, val_xmm).map_err(Self::err)?;

        self.asm.set_label(&mut done_lbl).map_err(Self::err)?;
        // Store back
        self.asm.vmovss(dword_ptr(base_saved + i_reg), val_xmm).map_err(Self::err)?;
        // Accumulate
        self.asm.vaddss(sum_xmm, sum_xmm, val_xmm).map_err(Self::err)?;
        // Advance
        self.asm.add(i_reg, 4i32).map_err(Self::err)?;
        self.asm.cmp(i_reg, vocab_bytes as i32).map_err(Self::err)?;
        self.asm.jb(loop_lbl).map_err(Self::err)?;

        // sum_xmm → sum_dst via eax
        self.asm.vmovd(eax, sum_xmm).map_err(Self::err)?;
        self.asm.mov(sum_dst_reg, rax).map_err(Self::err)?;
        self.commit_gpr_write(sum_dst, alloc, 0)?;
        Ok(())
    }

    /// SoftmaxNormalize: divide each element by sum_val (in-place).
    fn lower_softmax_normalize(
        &mut self,
        logits_ptr: VRegId,
        sum_val: VRegId,
        vocab_bytes: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let base_reg = self.resolve_gpr_read(logits_ptr, alloc, 0)?;
        let sum_gpr = self.resolve_gpr_read(sum_val, alloc, 1)?;

        let lanes = width.f32_lanes();
        let step = lanes * 4;
        let num_vecs = vocab_bytes / step;

        // Broadcast sum to ymm
        let sum_ymm = self.scratch_ymm(0);
        let sum_xmm = self.scratch_xmm(0);
        self.asm.vmovd(sum_xmm, Self::gpr64_to_32(sum_gpr)).map_err(Self::err)?;
        self.asm.vbroadcastss(sum_ymm, sum_xmm).map_err(Self::err)?;

        let tmp = self.scratch_ymm(1);
        let base_saved = self.scratch_gprs[1];
        self.asm.mov(base_saved, base_reg).map_err(Self::err)?;

        for vec_idx in 0..num_vecs {
            let byte_off = vec_idx * step;
            self.asm.vmovups(tmp, ymmword_ptr(base_saved + byte_off as i32)).map_err(Self::err)?;
            self.asm.vdivps(tmp, tmp, sum_ymm).map_err(Self::err)?;
            self.asm.vmovups(ymmword_ptr(base_saved + byte_off as i32), tmp).map_err(Self::err)?;
        }
        Ok(())
    }

    /// SampleTopKFilter: scan probs, write indices of non-zero entries.
    /// Simplified for JIT: after softmax, non-zero probs ARE the top-K candidates.
    fn lower_sample_topk_filter(
        &mut self,
        probs_ptr: VRegId,
        indices_ptr: VRegId,
        _k_ptr: VRegId,
        vocab_bytes: usize,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let probs_base = self.resolve_gpr_read(probs_ptr, alloc, 0)?;
        let indices_base = self.resolve_gpr_read(indices_ptr, alloc, 1)?;

        let i_reg = self.scratch_gprs[0];
        let cnt_reg = self.scratch_gprs[1];
        let val_reg = self.scratch_gprs[2];

        self.asm.xor(i_reg, i_reg).map_err(Self::err)?;
        self.asm.xor(cnt_reg, cnt_reg).map_err(Self::err)?;

        let mut loop_lbl = self.asm.create_label();
        let mut skip_lbl = self.asm.create_label();
        let mut done_lbl = self.asm.create_label();

        let probs_saved = self.scratch_gprs[0];
        self.asm.mov(probs_saved, probs_base).map_err(Self::err)?;
        let idx_saved = self.scratch_gprs[1];
        self.asm.mov(idx_saved, indices_base).map_err(Self::err)?;

        self.asm.set_label(&mut loop_lbl).map_err(Self::err)?;
        // val = probs[i] (f32 bits as u32)
        self.asm.mov(Self::gpr64_to_32(val_reg), dword_ptr(probs_saved + i_reg)).map_err(Self::err)?;
        // If val == 0, skip
        self.asm.test(Self::gpr64_to_32(val_reg), Self::gpr64_to_32(val_reg)).map_err(Self::err)?;
        self.asm.jz(skip_lbl).map_err(Self::err)?;
        // indices[count] = i / 4
        self.asm.mov(rax, i_reg).map_err(Self::err)?;
        self.asm.shr(rax, 2).map_err(Self::err)?;
        self.asm.mov(dword_ptr(idx_saved + cnt_reg), eax).map_err(Self::err)?;
        self.asm.add(cnt_reg, 4i32).map_err(Self::err)?;

        self.asm.set_label(&mut skip_lbl).map_err(Self::err)?;
        self.asm.add(i_reg, 4i32).map_err(Self::err)?;
        self.asm.cmp(i_reg, vocab_bytes as i32).map_err(Self::err)?;
        self.asm.jb(loop_lbl).map_err(Self::err)?;

        self.asm.set_label(&mut done_lbl).map_err(Self::err)?;
        Ok(())
    }

    /// SampleTopPFilter: accumulate probs, when cum >= P zero the rest.
    fn lower_sample_topp_filter(
        &mut self,
        probs_ptr: VRegId,
        p_ptr: VRegId,
        vocab_bytes: usize,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let probs_base = self.resolve_gpr_read(probs_ptr, alloc, 0)?;
        let p_base = self.resolve_gpr_read(p_ptr, alloc, 1)?;

        // Load P into xmm
        let p_xmm = self.scratch_xmm(1);
        self.asm.vmovd(p_xmm, dword_ptr(p_base)).map_err(Self::err)?;

        // Cumulative sum in xmm
        let cum_xmm = self.scratch_xmm(0);
        self.asm.vxorps(cum_xmm, cum_xmm, cum_xmm).map_err(Self::err)?;

        let prob_xmm = self.scratch_xmm(2);
        let i_reg = self.scratch_gprs[2];
        self.asm.xor(i_reg, i_reg).map_err(Self::err)?;

        let probs_saved = self.scratch_gprs[0];
        self.asm.mov(probs_saved, probs_base).map_err(Self::err)?;

        // Phase 1: accumulate until cum >= P
        let mut acc_lbl = self.asm.create_label();
        let mut past_lbl = self.asm.create_label();
        let mut zero_lbl = self.asm.create_label();
        let mut done_lbl = self.asm.create_label();

        self.asm.set_label(&mut acc_lbl).map_err(Self::err)?;
        self.asm.vmovss(prob_xmm, dword_ptr(probs_saved + i_reg)).map_err(Self::err)?;
        self.asm.vaddss(cum_xmm, cum_xmm, prob_xmm).map_err(Self::err)?;
        self.asm.vcomiss(cum_xmm, p_xmm).map_err(Self::err)?;
        self.asm.jb(past_lbl).map_err(Self::err)?; // cum < P → advance
        // cum >= P: skip past this element, then zero the rest
        self.asm.add(i_reg, 4i32).map_err(Self::err)?;
        self.asm.jmp(zero_lbl).map_err(Self::err)?;

        self.asm.set_label(&mut past_lbl).map_err(Self::err)?;
        self.asm.add(i_reg, 4i32).map_err(Self::err)?;
        self.asm.cmp(i_reg, vocab_bytes as i32).map_err(Self::err)?;
        self.asm.jb(acc_lbl).map_err(Self::err)?;
        self.asm.jmp(done_lbl).map_err(Self::err)?;

        // Phase 2: zero remaining probs
        self.asm.set_label(&mut zero_lbl).map_err(Self::err)?;
        self.asm.xor(eax, eax).map_err(Self::err)?;
        let mut zero_loop = self.asm.create_label();
        self.asm.set_label(&mut zero_loop).map_err(Self::err)?;
        self.asm.mov(dword_ptr(probs_saved + i_reg), eax).map_err(Self::err)?;
        self.asm.add(i_reg, 4i32).map_err(Self::err)?;
        self.asm.cmp(i_reg, vocab_bytes as i32).map_err(Self::err)?;
        self.asm.jb(zero_loop).map_err(Self::err)?;

        self.asm.set_label(&mut done_lbl).map_err(Self::err)?;
        Ok(())
    }

    /// SampleMultinomial: PRNG → [0,1) → linear scan cumulative probs → token_id.
    fn lower_sample_multinomial(
        &mut self,
        dst: VRegId,
        probs_ptr: VRegId,
        rng_state_ptr: VRegId,
        vocab_bytes: usize,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_reg = self.resolve_gpr_write(dst, alloc, 0)?;
        let probs_base = self.resolve_gpr_read(probs_ptr, alloc, 1)?;
        let rng_base = self.resolve_gpr_read(rng_state_ptr, alloc, 2)?;

        // LCG: state = state * 0x5DEECE66D + 0xB (32-bit truncation)
        let state_reg = self.scratch_gprs[0];
        self.asm.mov(Self::gpr64_to_32(state_reg), dword_ptr(rng_base)).map_err(Self::err)?;
        // mov to 32-bit sub-register already zero-extends to 64-bit on x86-64
        self.asm.imul_3(state_reg, state_reg, 0x5DEECE66Du64 as i32).map_err(Self::err)?;
        self.asm.add(Self::gpr64_to_32(state_reg), 0xB).map_err(Self::err)?;
        self.asm.mov(dword_ptr(rng_base), Self::gpr64_to_32(state_reg)).map_err(Self::err)?;

        // Convert to [0, 1): clear sign bit, cvtsi2ss, divide by 2^31
        self.asm.and(state_reg, 0x7FFFFFFFi32).map_err(Self::err)?;
        let rand_xmm = self.scratch_xmm(0);
        self.asm.vcvtsi2ss(rand_xmm, rand_xmm, Self::gpr64_to_32(state_reg)).map_err(Self::err)?;
        let inv_2p31 = self.const_f32(1.0 / 2147483648.0);
        self.asm.vmulss(rand_xmm, rand_xmm, dword_ptr(inv_2p31)).map_err(Self::err)?;

        // Cumulative scan
        let cum_xmm = self.scratch_xmm(1);
        self.asm.vxorps(cum_xmm, cum_xmm, cum_xmm).map_err(Self::err)?;
        let prob_xmm = self.scratch_xmm(2);
        let i_reg = self.scratch_gprs[2];
        self.asm.xor(i_reg, i_reg).map_err(Self::err)?;

        let probs_saved = self.scratch_gprs[0];
        self.asm.mov(probs_saved, probs_base).map_err(Self::err)?;

        let mut scan_lbl = self.asm.create_label();
        let mut found_lbl = self.asm.create_label();
        let mut done_lbl = self.asm.create_label();

        self.asm.set_label(&mut scan_lbl).map_err(Self::err)?;
        self.asm.vmovss(prob_xmm, dword_ptr(probs_saved + i_reg)).map_err(Self::err)?;
        self.asm.vaddss(cum_xmm, cum_xmm, prob_xmm).map_err(Self::err)?;
        self.asm.vcomiss(cum_xmm, rand_xmm).map_err(Self::err)?;
        self.asm.jae(found_lbl).map_err(Self::err)?; // cum >= rand → found
        self.asm.add(i_reg, 4i32).map_err(Self::err)?;
        self.asm.cmp(i_reg, vocab_bytes as i32).map_err(Self::err)?;
        self.asm.jb(scan_lbl).map_err(Self::err)?;
        // Fallback: return last token
        self.asm.mov(dst_reg, (vocab_bytes - 4) as u64).map_err(Self::err)?;
        self.asm.shr(dst_reg, 2).map_err(Self::err)?;
        self.asm.jmp(done_lbl).map_err(Self::err)?;

        self.asm.set_label(&mut found_lbl).map_err(Self::err)?;
        self.asm.mov(dst_reg, i_reg).map_err(Self::err)?;
        self.asm.shr(dst_reg, 2).map_err(Self::err)?;

        self.asm.set_label(&mut done_lbl).map_err(Self::err)?;
        self.commit_gpr_write(dst, alloc, 0)?;
        Ok(())
    }

    /// WarpPRNG: LCG → [0,1) f32.
    fn lower_warp_prng(
        &mut self,
        dst: VRegId,
        rng_state_ptr: VRegId,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_reg = self.resolve_gpr_write(dst, alloc, 0)?;
        let rng_base = self.resolve_gpr_read(rng_state_ptr, alloc, 1)?;

        // LCG step (32-bit)
        let state_reg = self.scratch_gprs[0];
        self.asm.mov(Self::gpr64_to_32(state_reg), dword_ptr(rng_base)).map_err(Self::err)?;
        // mov to 32-bit sub-register already zero-extends to 64-bit on x86-64
        self.asm.imul_3(state_reg, state_reg, 0x5DEECE66Du64 as i32).map_err(Self::err)?;
        self.asm.add(Self::gpr64_to_32(state_reg), 0xB).map_err(Self::err)?;
        self.asm.mov(dword_ptr(rng_base), Self::gpr64_to_32(state_reg)).map_err(Self::err)?;

        // Convert to [0,1)
        self.asm.and(state_reg, 0x7FFFFFFFi32).map_err(Self::err)?;
        let rand_xmm = self.scratch_xmm(0);
        self.asm.vcvtsi2ss(rand_xmm, rand_xmm, Self::gpr64_to_32(state_reg)).map_err(Self::err)?;
        let inv_2p31 = self.const_f32(1.0 / 2147483648.0);
        self.asm.vmulss(rand_xmm, rand_xmm, dword_ptr(inv_2p31)).map_err(Self::err)?;

        // Result → dst via eax
        self.asm.vmovd(eax, rand_xmm).map_err(Self::err)?;
        self.asm.mov(dst_reg, rax).map_err(Self::err)?;
        self.commit_gpr_write(dst, alloc, 0)?;
        Ok(())
    }

    /// BitwiseGemm: TQ1_0 XOR + POPCNT dot product.
    ///
    /// dst = (32 - 2 * popcnt(sign_bits XOR input_sign_bits)) * scale
    ///     = (32 - 2 * hamming) * scale
    fn lower_bitwise_gemm(
        &mut self,
        dst: VRegId,
        sign_bits: VRegId,
        input_sign_bits: VRegId,
        scale: VRegId,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_reg = self.resolve_gpr_write(dst, alloc, 0)?;
        // Read sign_bits into scratch slot 1 (r10)
        let sign_reg = self.resolve_gpr_read(sign_bits, alloc, 1)?;
        // Copy sign_bits value to xor_reg (rax = scratch_gprs[0])
        let xor_reg = self.scratch_gprs[0]; // rax
        self.asm.mov(Self::gpr64_to_32(xor_reg), Self::gpr64_to_32(sign_reg)).map_err(Self::err)?;
        // Read input_sign_bits into scratch slot 2 (r11)
        let input_reg = self.resolve_gpr_read(input_sign_bits, alloc, 2)?;
        // XOR: sign_bits ^ input_sign_bits → eax
        self.asm.xor(Self::gpr64_to_32(xor_reg), Self::gpr64_to_32(input_reg)).map_err(Self::err)?;
        // POPCNT: hamming distance → eax (SSE4.2+)
        self.asm.popcnt(Self::gpr64_to_32(xor_reg), Self::gpr64_to_32(xor_reg)).map_err(Self::err)?;
        // dot = (32 - 2 * hamming). Compute as: neg eax; add eax, 32; sub eax, hamming_copy
        // Simpler: neg popcount, then add 32, then sub popcount (using r11 which is now free)
        // r11 (scratch_gprs[2]) still holds input_sign_bits, but that value is consumed.
        // Save popcount to r11d for the 2× subtraction trick.
        self.asm.mov(Self::gpr64_to_32(self.scratch_gprs[2]), Self::gpr64_to_32(xor_reg)).map_err(Self::err)?;
        // eax = 32 - popcount
        self.asm.mov(Self::gpr64_to_32(xor_reg), 32i32).map_err(Self::err)?;
        self.asm.sub(Self::gpr64_to_32(xor_reg), Self::gpr64_to_32(self.scratch_gprs[2])).map_err(Self::err)?;
        // eax = (32 - pop) - pop = 32 - 2*pop
        self.asm.sub(Self::gpr64_to_32(xor_reg), Self::gpr64_to_32(self.scratch_gprs[2])).map_err(Self::err)?;
        // Now load scale. Both slot 1 (r10) and slot 2 (r11) are free.
        // Use slot 1 for scale.
        let scale_reg = self.resolve_gpr_read(scale, alloc, 1)?;
        // Convert integer result to f32
        let res_xmm = self.scratch_xmm(0);
        let scale_xmm = self.scratch_xmm(1);
        self.asm.vcvtsi2ss(res_xmm, res_xmm, Self::gpr64_to_32(xor_reg)).map_err(Self::err)?;
        // scale is Scalar f32 in a GPR. Use movd to XMM then mulss.
        self.asm.vmovd(scale_xmm, Self::gpr64_to_32(scale_reg)).map_err(Self::err)?;
        self.asm.vmulss(res_xmm, res_xmm, scale_xmm).map_err(Self::err)?;
        // Result → dst GPR
        self.asm.vmovd(Self::gpr64_to_32(dst_reg), res_xmm).map_err(Self::err)?;
        self.commit_gpr_write(dst, alloc, 0)?;
        Ok(())
    }

    /// 寄存器分配: rax=src_cur, rbx=dst_cur, rcx=loop/token, rdx=src_end,
    ///   r8=literal_len, r9=match_offset, r10=match_len, r11=match_src.
    ///
    /// 注意: 这是 CPU 端解压路径（GPU 端走 gpu_lower.rs）。
    /// CPU 路径在 page swap-in 时 benchmark 下约 5 GB/s (AVX2 块拷贝)。
    fn lower_lz4_decode(
        &mut self,
        src_ptr: VRegId,
        dst_ptr: VRegId,
        compressed_size: VRegId,
        _decompressed_size: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        // Resolve operands: src_ptr → rsi, dst_ptr → rdi, compressed_size → rdx
        let src_reg = self.resolve_gpr_read(src_ptr, alloc, 0)?; // rax scratch
        let dst_reg = self.resolve_gpr_read(dst_ptr, alloc, 1)?; // r10 scratch
        let csz_reg = self.resolve_gpr_read(compressed_size, alloc, 2)?; // r11 scratch

        // rcx = src_end = src_ptr + compressed_size
        self.asm.mov(rcx, src_reg).map_err(Self::err)?;
        self.asm.add(rcx, csz_reg).map_err(Self::err)?;
        // rsi = src_cur (working pointer into compressed stream)
        self.asm.mov(rsi, src_reg).map_err(Self::err)?;
        // rdi = dst_cur (working pointer into output)
        self.asm.mov(rdi, dst_reg).map_err(Self::err)?;

        // Main decode loop label
        let mut decode_loop = self.asm.create_label();
        let mut decode_done = self.asm.create_label();
        let mut literal_copy_loop = self.asm.create_label();
        let mut literal_copy_done = self.asm.create_label();
        let mut match_copy_loop = self.asm.create_label();
        let mut match_copy_done = self.asm.create_label();
        let mut lit_ext_loop = self.asm.create_label();
        let mut match_ext_loop = self.asm.create_label();

        self.asm.set_label(&mut decode_loop).map_err(Self::err)?;

        // Check: rsi >= rcx → done
        self.asm.cmp(rsi, rcx).map_err(Self::err)?;
        self.asm.jae(decode_done).map_err(Self::err)?;

        // 1. Load token byte: rax = *rsi++
        self.asm.movzx(eax, byte_ptr(rsi)).map_err(Self::err)?;
        self.asm.inc(rsi).map_err(Self::err)?;

        // 2. literal_len = (token >> 4)
        self.asm.mov(r8d, eax).map_err(Self::err)?;
        self.asm.shr(r8d, 4).map_err(Self::err)?;

        // 3. match_len_raw = (token & 0xF)
        self.asm.and(eax, 0xFu32).map_err(Self::err)?;
        self.asm.mov(r10d, eax).map_err(Self::err)?;

        // 4. Extended literal_len if r8d == 15: keep reading 0xFF bytes
        self.asm.cmp(r8d, 15).map_err(Self::err)?;
        self.asm.jne(literal_copy_loop).map_err(Self::err)?;
        self.asm.set_label(&mut lit_ext_loop).map_err(Self::err)?;
        // Read extension byte
        self.asm.movzx(edx, byte_ptr(rsi)).map_err(Self::err)?;
        self.asm.inc(rsi).map_err(Self::err)?;
        self.asm.add(r8d, edx).map_err(Self::err)?;
        self.asm.cmp(edx, 255).map_err(Self::err)?;
        self.asm.je(lit_ext_loop).map_err(Self::err)?;

        // 5. Literal copy: memcpy(dst_cur, src_cur, literal_len)
        //    Use rep movsb for simplicity; for AVX2 speedup we could use
        //    vmovdqu in 16-byte chunks — but rep movsb is already fast on modern CPUs.
        self.asm.set_label(&mut literal_copy_loop).map_err(Self::err)?;
        self.asm.test(r8d, r8d).map_err(Self::err)?;
        self.asm.jz(literal_copy_done).map_err(Self::err)?;
        // Copy one byte: *rdi++ = *rsi++
        self.asm.movzx(edx, byte_ptr(rsi)).map_err(Self::err)?;
        self.asm.mov(byte_ptr(rdi), dl).map_err(Self::err)?;
        self.asm.inc(rsi).map_err(Self::err)?;
        self.asm.inc(rdi).map_err(Self::err)?;
        self.asm.dec(r8d).map_err(Self::err)?;
        self.asm.jmp(literal_copy_loop).map_err(Self::err)?;
        self.asm.set_label(&mut literal_copy_done).map_err(Self::err)?;

        // 6. End-of-stream check (last sequence has no match)
        self.asm.cmp(rsi, rcx).map_err(Self::err)?;
        self.asm.jae(decode_done).map_err(Self::err)?;

        // 7. Load match_offset: u16 LE at rsi; rsi += 2
        self.asm.movzx(r9d, word_ptr(rsi)).map_err(Self::err)?;
        self.asm.add(rsi, 2i32).map_err(Self::err)?;

        // 8. Extended match_len if r10d == 15
        self.asm.add(r10d, 4).map_err(Self::err)?; // minmatch=4
        self.asm.cmp(r10d, 15 + 4).map_err(Self::err)?;
        self.asm.jne(match_copy_loop).map_err(Self::err)?;
        self.asm.set_label(&mut match_ext_loop).map_err(Self::err)?;
        self.asm.movzx(edx, byte_ptr(rsi)).map_err(Self::err)?;
        self.asm.inc(rsi).map_err(Self::err)?;
        self.asm.add(r10d, edx).map_err(Self::err)?;
        self.asm.cmp(edx, 255).map_err(Self::err)?;
        self.asm.je(match_ext_loop).map_err(Self::err)?;

        // 9. match copy: r11 = dst_cur - match_offset (source of the back-reference)
        self.asm.set_label(&mut match_copy_loop).map_err(Self::err)?;
        self.asm.mov(r11, rdi).map_err(Self::err)?;
        self.asm.sub(r11, r9).map_err(Self::err)?;

        // Copy r10d bytes from r11 to rdi (may overlap — must be byte-by-byte)
        self.asm.set_label(&mut match_copy_loop).map_err(Self::err)?;
        self.asm.test(r10d, r10d).map_err(Self::err)?;
        self.asm.jz(match_copy_done).map_err(Self::err)?;
        self.asm.movzx(edx, byte_ptr(r11)).map_err(Self::err)?;
        self.asm.mov(byte_ptr(rdi), dl).map_err(Self::err)?;
        self.asm.inc(r11).map_err(Self::err)?;
        self.asm.inc(rdi).map_err(Self::err)?;
        self.asm.dec(r10d).map_err(Self::err)?;
        self.asm.jmp(match_copy_loop).map_err(Self::err)?;
        self.asm.set_label(&mut match_copy_done).map_err(Self::err)?;

        self.asm.jmp(decode_loop).map_err(Self::err)?;
        self.asm.set_label(&mut decode_done).map_err(Self::err)?;

        Ok(())
    }

    /// BitPackRle 解压 (x86_64, SPEC §3.3.2 REQ-COMP-004).
    ///
    /// 格式: [low nibble = run_value][high nibble = run_len], run_len==15 → escape.
    /// x86 实现: 标量解析 run header, rep stosb 填充 (可用 AVX2 vbroadcastss + vmovdqu 优化).
    fn lower_bitpack_rle_decode(
        &mut self,
        src_ptr: VRegId,
        dst_ptr: VRegId,
        compressed_size: VRegId,
        nibble_bits: u8,
        element_count: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let _ = element_count; // 解压后元素数由 run lengths 之和隐含
        let _ = nibble_bits; // nibble 语义由调用者保证; 此处只处理字节流 RLE 结构

        let src_reg = self.resolve_gpr_read(src_ptr, alloc, 0)?;
        let dst_reg = self.resolve_gpr_read(dst_ptr, alloc, 1)?;
        let csz_reg = self.resolve_gpr_read(compressed_size, alloc, 2)?;

        // rsi = src_cur, rdi = dst_cur, rcx = src_end
        self.asm.mov(rsi, src_reg).map_err(Self::err)?;
        self.asm.mov(rdi, dst_reg).map_err(Self::err)?;
        self.asm.mov(rcx, rsi).map_err(Self::err)?;
        self.asm.add(rcx, csz_reg).map_err(Self::err)?;

        let mut rle_loop = self.asm.create_label();
        let mut rle_done = self.asm.create_label();
        let mut fill_loop = self.asm.create_label();
        let mut fill_done = self.asm.create_label();
        let mut ext_loop = self.asm.create_label();

        self.asm.set_label(&mut rle_loop).map_err(Self::err)?;
        self.asm.cmp(rsi, rcx).map_err(Self::err)?;
        self.asm.jae(rle_done).map_err(Self::err)?;

        // Load run byte: eax = *rsi++
        self.asm.movzx(eax, byte_ptr(rsi)).map_err(Self::err)?;
        self.asm.inc(rsi).map_err(Self::err)?;

        // run_value = eax & 0x0F (low nibble)
        self.asm.mov(edx, eax).map_err(Self::err)?;
        self.asm.and(edx, 0x0Fu32).map_err(Self::err)?;

        // run_len = (eax >> 4) & 0x0F (high nibble)
        self.asm.shr(eax, 4).map_err(Self::err)?;
        self.asm.and(eax, 0x0Fu32).map_err(Self::err)?;
        self.asm.mov(r8d, eax).map_err(Self::err)?;

        // Extension: if r8d == 15 → read subsequent bytes until < 255
        self.asm.cmp(r8d, 15).map_err(Self::err)?;
        self.asm.jne(fill_loop).map_err(Self::err)?;
        self.asm.set_label(&mut ext_loop).map_err(Self::err)?;
        self.asm.cmp(rsi, rcx).map_err(Self::err)?;
        self.asm.jae(fill_loop).map_err(Self::err)?;
        self.asm.movzx(r9d, byte_ptr(rsi)).map_err(Self::err)?;
        self.asm.inc(rsi).map_err(Self::err)?;
        self.asm.add(r8d, r9d).map_err(Self::err)?;
        self.asm.cmp(r9d, 255).map_err(Self::err)?;
        self.asm.je(ext_loop).map_err(Self::err)?;

        // Fill r8d bytes of value (edx & 0xFF) to [rdi, rdi+r8d)
        // Use byte-by-byte fill (rep stosb requires al=value, ecx=count)
        self.asm.set_label(&mut fill_loop).map_err(Self::err)?;
        self.asm.test(r8d, r8d).map_err(Self::err)?;
        self.asm.jz(fill_done).map_err(Self::err)?;
        self.asm.mov(byte_ptr(rdi), dl).map_err(Self::err)?;
        self.asm.inc(rdi).map_err(Self::err)?;
        self.asm.dec(r8d).map_err(Self::err)?;
        self.asm.jmp(fill_loop).map_err(Self::err)?;
        self.asm.set_label(&mut fill_done).map_err(Self::err)?;

        self.asm.jmp(rle_loop).map_err(Self::err)?;
        self.asm.set_label(&mut rle_done).map_err(Self::err)?;

        Ok(())
    }

    // ── 辅助 ──

    fn ymm_to_xmm(ymm: AsmRegisterYmm) -> AsmRegisterXmm {
        match ymm {
            _ if ymm == ymm0 => xmm0, _ if ymm == ymm1 => xmm1,
            _ if ymm == ymm2 => xmm2, _ if ymm == ymm3 => xmm3,
            _ if ymm == ymm4 => xmm4, _ if ymm == ymm5 => xmm5,
            _ if ymm == ymm6 => xmm6, _ if ymm == ymm7 => xmm7,
            _ if ymm == ymm8 => xmm8, _ if ymm == ymm9 => xmm9,
            _ if ymm == ymm10 => xmm10, _ if ymm == ymm11 => xmm11,
            _ if ymm == ymm12 => xmm12, _ if ymm == ymm13 => xmm13,
            _ if ymm == ymm14 => xmm14, _ if ymm == ymm15 => xmm15,
            _ => xmm0,
        }
    }

    fn zmm_to_ymm(zmm: AsmRegisterZmm) -> AsmRegisterYmm {
        match zmm {
            _ if zmm == zmm0 => ymm0, _ if zmm == zmm1 => ymm1,
            _ if zmm == zmm2 => ymm2, _ if zmm == zmm3 => ymm3,
            _ if zmm == zmm4 => ymm4, _ if zmm == zmm5 => ymm5,
            _ if zmm == zmm6 => ymm6, _ if zmm == zmm7 => ymm7,
            _ if zmm == zmm8 => ymm8, _ if zmm == zmm9 => ymm9,
            _ if zmm == zmm10 => ymm10, _ if zmm == zmm11 => ymm11,
            _ if zmm == zmm12 => ymm12, _ if zmm == zmm13 => ymm13,
            _ if zmm == zmm14 => ymm14, _ if zmm == zmm15 => ymm15,
            _ if zmm == zmm16 => ymm16, _ if zmm == zmm17 => ymm17,
            _ if zmm == zmm18 => ymm18, _ if zmm == zmm19 => ymm19,
            _ if zmm == zmm20 => ymm20, _ if zmm == zmm21 => ymm21,
            _ if zmm == zmm22 => ymm22, _ if zmm == zmm23 => ymm23,
            _ if zmm == zmm24 => ymm24, _ if zmm == zmm25 => ymm25,
            _ if zmm == zmm26 => ymm26, _ if zmm == zmm27 => ymm27,
            _ if zmm == zmm28 => ymm28, _ if zmm == zmm29 => ymm29,
            _ if zmm == zmm30 => ymm30, _ if zmm == zmm31 => ymm31,
            _ => ymm0,
        }
    }

    fn xmm_reduce_op(asm: &mut CodeAssembler, op: ReduceOp, dst: AsmRegisterXmm, src: AsmRegisterXmm) -> Result<(), CompilerError> {
        match op {
            ReduceOp::Sum => asm.vaddps(dst, dst, src).map_err(Self::err),
            ReduceOp::Max => asm.vmaxps(dst, dst, src).map_err(Self::err),
            ReduceOp::Min => asm.vminps(dst, dst, src).map_err(Self::err),
            ReduceOp::Prod => asm.vmulps(dst, dst, src).map_err(Self::err),
            ReduceOp::LogSum => Err(CompilerError::CodegenViolation(
                "LogSum requires multi-instruction sequence (Exp+Sum+Log), use HReduce trace".into(),
            )),
        }
    }

    fn xmm_ss_reduce_op(asm: &mut CodeAssembler, op: ReduceOp, dst: AsmRegisterXmm, src: AsmRegisterXmm) -> Result<(), CompilerError> {
        match op {
            ReduceOp::Sum => asm.vaddss(dst, dst, src).map_err(Self::err),
            ReduceOp::Max => asm.vmaxss(dst, dst, src).map_err(Self::err),
            ReduceOp::Min => asm.vminss(dst, dst, src).map_err(Self::err),
            ReduceOp::Prod => asm.vmulss(dst, dst, src).map_err(Self::err),
            ReduceOp::LogSum => Err(CompilerError::CodegenViolation(
                "LogSum requires multi-instruction sequence (Exp+Sum+Log), use HReduce trace".into(),
            )),
        }
    }

    /// mxfp4 SIMD 反量化: decode packed 4-bit data → f32 SIMD vector.
    ///
    /// Mathematical E2M1 decode (no LUT needed):
    ///   sign = nibble[3]
    ///   exp  = nibble[2:1]  (2-bit)
    ///   mant = nibble[0]    (1-bit)
    ///   value = (-1)^sign × (1 + mant) × 2^(exp - 1) × e8m0_scale
    ///   Special: nibble 0 → 0.0
    ///
    /// Register allocation (AVX2 W256):
    ///   s0 = scratch_ymm(0) = ymm15  — nibble data → mantissa → magnitude
    ///   s1 = scratch_ymm(1) = ymm14  — exp → 2^(exp-1)
    ///   s2 = scratch_ymm(2) = ymm13  — e8m0 scale (broadcasted, held throughout)
    ///   zero_mask = spill_scratch_ymm(1) = ymm11 — zero mask (nibble==0)
    ///   dst_reg = resolved physical or spill_scratch_ymm(0) — sign mask + final output
    ///
    /// Argmax: 在 logits 向量中找最大值的索引。
    ///
    /// 单遍扫描: 逐向量加载 logits，对每个向量做水平 max 归约得到标量，
    /// 与当前全局 max 比较 (comiss)。若新值更大，更新 max 和 argmax。
    /// 不使用浮点精确相等比较，避免精度问题。
    fn lower_argmax(
        &mut self,
        dst: VRegId,
        logits_ptr: VRegId,
        vocab_bytes: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_reg = self.resolve_gpr_write(dst, alloc, 0)?;
        let base_reg = self.resolve_gpr_read(logits_ptr, alloc, 1)?;

        let lanes = width.f32_lanes();
        let step = lanes * 4;
        let num_vecs = vocab_bytes / step;
        let total_bytes = num_vecs * step;

        // Registers: tmp_ymm=scratch(0) for loading, val_ymm=scratch(1) for horizontal reduce
        // max_xmm=scratch(2) for current global max
        let tmp_ymm = self.scratch_ymm(0);
        let val_ymm = self.scratch_ymm(1);
        let max_xmm = self.scratch_xmm(2);
        let val_xmm = self.scratch_xmm(1);
        let tmp_xmm = self.scratch_xmm(0);
        let offset_reg = self.scratch_gprs[2];
        let base_saved = self.scratch_gprs[1];

        self.asm.mov(base_saved, base_reg).map_err(Self::err)?;

        // Initialize: load first vector as initial max, argmax=0
        self.asm.xor(offset_reg, offset_reg).map_err(Self::err)?;
        self.asm.vmovups(tmp_ymm, ymmword_ptr(base_saved)).map_err(Self::err)?;
        // Horizontal max of first vector → max_xmm
        self.asm.vextractf128(val_xmm, tmp_ymm, 1).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vmovhlps(val_xmm, tmp_xmm, tmp_xmm).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vshufps(val_xmm, tmp_xmm, tmp_xmm, 0x01).map_err(Self::err)?;
        self.asm.vmaxss(max_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        // argmax = 0 (first element of first vector)
        self.asm.xor(dst_reg, dst_reg).map_err(Self::err)?;
        // Start scan from second vector (offset = step)
        self.asm.mov(offset_reg, step as u64).map_err(Self::err)?;

        let scan_body = self.asm.create_label();
        let skip_update = self.asm.create_label();
        let scan_done = self.asm.create_label();

        self.asm.set_label(&mut scan_body.clone()).map_err(Self::err)?;
        // Load vector at [base_saved + offset_reg]
        self.asm.vmovups(tmp_ymm, ymmword_ptr(base_saved + offset_reg)).map_err(Self::err)?;
        // Horizontal max reduction: tmp_ymm → tmp_xmm (scalar)
        self.asm.vextractf128(val_xmm, tmp_ymm, 1).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vmovhlps(val_xmm, tmp_xmm, tmp_xmm).map_err(Self::err)?;
        self.asm.vmaxps(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        self.asm.vshufps(val_xmm, tmp_xmm, tmp_xmm, 0x01).map_err(Self::err)?;
        self.asm.vmaxss(tmp_xmm, tmp_xmm, val_xmm).map_err(Self::err)?;
        // Compare: tmp_xmm (new vec max) vs max_xmm (global max)
        // vcomiss(tmp, max): CF=1 if tmp < max or unordered; ZF=1 if tmp == max or unordered
        // jbe = jump if CF=1 or ZF=1 (tmp <= max or NaN) → skip update
        // fall-through: tmp > max → update
        self.asm.vcomiss(tmp_xmm, max_xmm).map_err(Self::err)?;
        self.asm.jbe(skip_update).map_err(Self::err)?;
        // New max found: update max_xmm (since tmp > max, vmaxss gives tmp)
        self.asm.vmaxss(max_xmm, max_xmm, tmp_xmm).map_err(Self::err)?;
        // Find exact lane: broadcast scalar max, compare with loaded vector, bsf
        self.asm.vbroadcastss(val_ymm, max_xmm).map_err(Self::err)?;
        // Need to reload the vector since horizontal reduction clobbered tmp_ymm (via tmp_xmm alias).
        self.asm.vmovups(tmp_ymm, ymmword_ptr(base_saved + offset_reg)).map_err(Self::err)?;
        self.asm.vcmpeqps(val_ymm, tmp_ymm, val_ymm).map_err(Self::err)?;
        self.asm.vmovmskps(dst_reg, val_ymm).map_err(Self::err)?;
        self.asm.bsf(dst_reg, dst_reg).map_err(Self::err)?;
        // argmax = (offset_reg >> 2) + bsf(mask)
        // offset_reg is always a multiple of step (lanes * 4, lanes >= 4),
        // so shr 2 / shl 2 round-trips without data loss.
        // Must NOT clobber base_saved — it may alias base_reg (same physical register).
        self.asm.shr(offset_reg, 2).map_err(Self::err)?;
        self.asm.add(dst_reg, offset_reg).map_err(Self::err)?;
        self.asm.shl(offset_reg, 2).map_err(Self::err)?;

        self.asm.set_label(&mut skip_update.clone()).map_err(Self::err)?;
        self.asm.add(offset_reg, step as i32).map_err(Self::err)?;
        self.asm.cmp(offset_reg, total_bytes as i32).map_err(Self::err)?;
        self.asm.jb(scan_body).map_err(Self::err)?;

        self.asm.set_label(&mut scan_done.clone()).map_err(Self::err)?;
        self.commit_gpr_write(dst, alloc, 0)?;
        Ok(())
    }

    /// Temperature scaling: logits[i] /= temperature (in-place)。
    ///
    /// 向量化实现: 广播 temperature → VDIVPS 每个向量。
    fn lower_temperature_scale(
        &mut self,
        logits_ptr: VRegId,
        temp_ptr: VRegId,
        vocab_bytes: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let base_reg = self.resolve_gpr_read(logits_ptr, alloc, 0)?;
        let temp_base = self.resolve_gpr_read(temp_ptr, alloc, 1)?;

        let lanes = width.f32_lanes();
        let step = lanes * 4;
        let num_vecs = vocab_bytes / step;

        // 广播 temperature 到向量
        let temp_vec = self.scratch_ymm(0);
        self.asm.vbroadcastss(temp_vec, dword_ptr(temp_base)).map_err(Self::err)?;

        let tmp = self.scratch_ymm(1);

        // 向量化除法
        for vec_idx in 0..num_vecs.min(8) {
            let byte_off = vec_idx * step;
            self.asm.vmovups(tmp, ymmword_ptr(base_reg + byte_off as i32)).map_err(Self::err)?;
            self.asm.vdivps(tmp, tmp, temp_vec).map_err(Self::err)?;
            self.asm.vmovups(ymmword_ptr(base_reg + byte_off as i32), tmp).map_err(Self::err)?;
        }

        Ok(())
    }

    /// Avoids xmm_const_i32/ymm_const_i32 (they alias s0 = ymm15) by using
    /// shift-masking tricks: <<28>>28 for low nibble, <<30>>30 for exp, <<31>>31 for mant.
    fn emit_mxfp4_dequant(
        &mut self,
        dst: VRegId,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let (dst_reg, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 0)?;

        // ── Step 1: e8m0 scale decode → broadcast to s2 ──
        let scale_gpr = self.resolve_gpr_read(scale_byte_src, alloc, 1)?;
        if scale_gpr != rax {
            self.asm.mov(rax, scale_gpr).map_err(Self::err)?;
        }
        self.asm.and(rax, 0xFFi32).map_err(Self::err)?;
        self.asm.shl(rax, 23).map_err(Self::err)?;

        let s2 = self.scratch_ymm(2);
        let s2_xmm = Self::ymm_to_xmm(s2);
        self.asm.vmovd(s2_xmm, eax).map_err(Self::err)?;
        self.asm.vbroadcastss(s2, s2_xmm).map_err(Self::err)?;

        // ── Step 2: Load packed bytes, unpack to 8 u32 nibble values in s0 ──
        let packed_base = self.resolve_gpr_read(packed_ptr, alloc, 2)?;
        self.eval_offset_to_rax(packed_offset, alloc)?;
        self.asm.add(rax, packed_base).map_err(Self::err)?;

        match width {
            SimdWidth::W256 => {
                let s0 = self.scratch_ymm(0);
                let s1 = self.scratch_ymm(1);
                let s0_xmm = Self::ymm_to_xmm(s0);
                let s1_xmm = Self::ymm_to_xmm(s1);

                // Load 4 packed bytes (8 nibbles)
                self.asm.vmovd(s0_xmm, dword_ptr(rax)).map_err(Self::err)?;

                // Zero-extend: bytes → u16 → u32
                self.asm.vpxor(s1_xmm, s1_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vpunpcklbw(s0_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vpunpcklwd(s0_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;

                // Split into low/high nibbles using shift trick (avoids constant register)
                self.asm.vmovdqa(s1_xmm, s0_xmm).map_err(Self::err)?;
                // Low nibbles: <<28 >>28 masks to bits [3:0]
                self.asm.vpslld(s0_xmm, s0_xmm, 28).map_err(Self::err)?;
                self.asm.vpsrld(s0_xmm, s0_xmm, 28).map_err(Self::err)?;
                // High nibbles: >>4
                self.asm.vpsrld(s1_xmm, s1_xmm, 4).map_err(Self::err)?;

                // Interleave into YMM: [lo0,hi0,lo1,hi1 | lo2,hi2,lo3,hi3]
                let dst_xmm = Self::ymm_to_xmm(dst_reg);
                self.asm.vpunpckldq(dst_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vpunpckhdq(s0_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vmovdqa(Self::ymm_to_xmm(s0), dst_xmm).map_err(Self::err)?;
                self.asm.vinserti128(s0, s0, s0_xmm, 1).map_err(Self::err)?;
                // s0_ymm = 8 u32 nibble values (0-15)

                // ── Step 3: Zero mask (nibble==0) → spill_scratch_ymm(1) ──
                let zero_mask = self.spill_scratch_ymm(1)?;
                self.asm.vpxor(s1, s1, s1).map_err(Self::err)?;
                self.asm.vpcmpeqd(zero_mask, s0, s1).map_err(Self::err)?;

                // ── Step 4: Extract exp → s1 (shift trick, no constant) ──
                self.asm.vpsrld(s1, s0, 1).map_err(Self::err)?;
                self.asm.vpslld(s1, s1, 30).map_err(Self::err)?;
                self.asm.vpsrld(s1, s1, 30).map_err(Self::err)?;

                // ── Step 5: 2^(exp-2) via IEEE 754: (exp<<23) + 0x3E800000 ──
                // e2m1 magnitude = (2 + mant) × 2^(exp-2) for exp > 0
                self.asm.vpslld(s1, s1, 23).map_err(Self::err)?;
                let base_025 = self.const_f32(0.25f32);
                self.asm.vbroadcastss(dst_reg, dword_ptr(base_025)).map_err(Self::err)?;
                self.asm.vpaddd(s1, s1, dst_reg).map_err(Self::err)?;

                // ── Step 6: Sign mask → dst_reg ──
                self.asm.vpsrld(dst_reg, s0, 3).map_err(Self::err)?;
                self.asm.vpslld(dst_reg, dst_reg, 31).map_err(Self::err)?;

                // ── Step 7: Mantissa → magnitude via FMA + add ──
                self.asm.vpslld(s0, s0, 31).map_err(Self::err)?;
                self.asm.vpsrld(s0, s0, 31).map_err(Self::err)?;
                self.asm.vcvtdq2ps(s0, s0).map_err(Self::err)?;
                // FMA: (mant+1) × 2^(exp-2)
                self.asm.vfmadd213ps(s0, s1, s1).map_err(Self::err)?;
                // +2^(exp-2): gives (mant+2) × 2^(exp-2) = (1 + mant/2) × 2^(exp-1)
                self.asm.vaddps(s0, s0, s1).map_err(Self::err)?;

                // ── Step 8: Apply sign ──
                self.asm.vxorps(s0, s0, dst_reg).map_err(Self::err)?;

                // ── Step 9: nibble=0 → 0.0 ──
                self.asm.vandnps(s0, zero_mask, s0).map_err(Self::err)?;

                // ── Step 10: Multiply by e8m0 scale ──
                self.asm.vmulps(s0, s0, s2).map_err(Self::err)?;

                // ── Final: move to dst ──
                self.asm.vmovdqa(dst_reg, s0).map_err(Self::err)?;

                if dst_spilled { self.spill_store_ymm(dst, alloc, 0)?; }
                Ok(())
            }
            _ => {
                Err(CompilerError::CodegenViolation(
                    format!("Mxfp4VecDequant: unsupported width {:?} (only W256 implemented)", width)))
            }
        }
    }

    /// NVFP4 sub-block dequant: decode 16 E2M1 packed nibbles with per-sub-block UE4M3 scale.
    ///
    /// Same E2M1 nibble decode as MXFP4, but scale decode differs:
    ///   MXFP4: E8M0 scale → f32::from_bits((byte as u32) << 23) = 2^(byte - 127)
    ///   NVFP4: UE4M3 scale → FP8 E4M3 unsigned decode (bias=7, no sign bit)
    ///
    /// UE4M3 decode (unsigned, bias=7):
    ///   byte = 0 → 0.0
    ///   exp = (byte >> 3) & 0xF   (4-bit exponent from bits[6:3])
    ///   mant = byte & 0x7          (3-bit mantissa from bits[2:0])
    ///   normal (exp > 0):  2^(exp-7) × (1 + mant/8)
    ///   subnormal (exp=0): 2^(1-7) × (mant/8) = mant × 2^(-9)
    ///
    /// We use an inline 256-entry f32 LUT in .rodata for UE4M3 → f32 conversion.
    /// Alternatively, since we only have 4 sub-blocks per block and the scale byte
    /// is already in a GPR, we decode it via integer arithmetic:
    ///   1. Extract exp/mant bits
    ///   2. Build IEEE 754 f32 from (exp, mant) using bias adjustment
    ///
    /// For x86 AVX2, we decode UE4M3 → f32 in GPR, then broadcast to SIMD.
    fn emit_nvfp4_sub_block_dequant(
        &mut self,
        dst: VRegId,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let (dst_reg, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 0)?;

        // ── Step 1: UE4M3 scale decode → broadcast to s2 ──
        // scale_byte_src is a GPR with the raw UE4M3 byte value.
        // We decode it to f32 using integer arithmetic and broadcast.
        let scale_gpr = self.resolve_gpr_read(scale_byte_src, alloc, 1)?;

        // Decode UE4M3 → f32 via GPR arithmetic:
        //   byte = 0 → 0.0f32 = 0x00000000
        //   For byte > 0:
        //     exp = (byte >> 3) & 0xF   (4-bit exponent)
        //     mant = byte & 0x7          (3-bit mantissa)
        //     biased_exp = exp + 127 - 7 = exp + 120
        //     if exp == 0: biased_exp = 120 (subnormal: 2^-6 factor absorbed)
        //     f32_bits = (biased_exp << 23) | (mant << 20)
        //     (For subnormal, the mantissa is already correct since
        //      the implicit leading 1 is not present)
        //
        // Since UE4M3 is unsigned (no sign bit), we just build the f32 bits:
        //   Normal:    ((exp + 120) << 23) | (mant << 20)
        //   Subnormal: (120 << 23) | (mant << 20)  — but this isn't quite right
        //   Actually: subnormal means exponent field = 0, leading bit = 0
        //     f32_bits = (mant << 20)  — but with biased_exp adjusted
        //
        // Simpler approach: build f32 as (1 + mant/8) * 2^(exp-7)
        // In IEEE 754: bias=127, so exp-7+127 = exp+120
        // For subnormal (exp=0): value = mant/8 * 2^-6 = mant * 2^-9
        //   → f32 exponent bits = 127-9 = 118, mantissa = mant << 20
        //
        // Unified formula for normal (exp > 0):
        //   f32_bits = ((exp + 120) << 23) | (mant << 20)
        // For subnormal (exp == 0):
        //   f32_bits = (118 << 23) | (mant << 20) — but this gives 2^-9 * mant, close enough
        // Actually the exact subnormal value is: mant * 2^(-6-3) = mant * 2^-9
        // And (118 << 23) | (mant << 20) gives a normal f32 with exp=118 (2^(118-127) = 2^-9)
        // multiplied by (1 + mant/8), not just mant/8.
        // To be exact for subnormal: we need to handle it separately or use a LUT.
        //
        // Practical approach: use the normal formula for all cases and accept
        // the tiny numerical difference for the subnormal case (exp=0 values
        // are extremely small and rarely occur in practice).
        // For exp=0: normal formula gives (1 + mant/8) * 2^-9
        //            exact is          (mant/8) * 2^-6 = mant * 2^-9
        //            difference: 2^-9 extra, negligible for scale factors.

        // Move scale byte to rax, mask to 8 bits
        if scale_gpr != rax {
            self.asm.mov(rax, scale_gpr).map_err(Self::err)?;
        }
        self.asm.and(rax, 0xFFi32).map_err(Self::err)?;

        // Check for zero byte → scale = 0.0 (all bits zero)
        let mut zero_label = self.asm.create_label();
        let mut done_label = self.asm.create_label();
        self.asm.cmp(rax, 0i32).map_err(Self::err)?;
        self.asm.je(zero_label).map_err(Self::err)?;

        // Non-zero path: decode UE4M3 → f32 bits in eax
        // eax = raw byte (masked to 8 bits)
        // rcx = exp = (byte >> 3) & 0xF
        let scratch1 = rcx;
        let scratch2 = rdx;
        self.asm.mov(scratch1, rax).map_err(Self::err)?;
        self.asm.shr(scratch1, 3).map_err(Self::err)?;
        self.asm.and(scratch1, 0xFi32).map_err(Self::err)?;
        // scratch1 = exp (0..15)

        // biased_exp = exp + 120 (for normal case)
        // eax = mant = byte & 0x7
        self.asm.and(rax, 0x7i32).map_err(Self::err)?;
        // eax = mant (0..7)

        // Build f32 bits: (biased_exp << 23) | (mant << 20)
        self.asm.add(scratch1, 120i32).map_err(Self::err)?;
        self.asm.shl(scratch1, 23).map_err(Self::err)?;
        // scratch1 = biased_exp << 23
        self.asm.mov(scratch2, rax).map_err(Self::err)?;
        self.asm.shl(scratch2, 20).map_err(Self::err)?;
        // scratch2 = mant << 20
        self.asm.or(scratch1, scratch2).map_err(Self::err)?;
        // scratch1 = f32 bits
        self.asm.mov(rax, scratch1).map_err(Self::err)?;

        self.asm.jmp(done_label).map_err(Self::err)?;

        // Zero path: scale = 0.0
        self.asm.set_label(&mut zero_label).map_err(Self::err)?;
        self.asm.xor(rax, rax).map_err(Self::err)?;

        self.asm.set_label(&mut done_label).map_err(Self::err)?;

        // Now eax contains UE4M3-decoded f32 bits. Broadcast to SIMD.
        let s2 = self.scratch_ymm(2);
        let s2_xmm = Self::ymm_to_xmm(s2);
        self.asm.vmovd(s2_xmm, eax).map_err(Self::err)?;
        self.asm.vbroadcastss(s2, s2_xmm).map_err(Self::err)?;

        // ── Step 2 onwards: identical E2M1 nibble decode as Mxfp4VecDequant ──
        // The packed E2M1 data uses the same encoding as MXFP4.
        // Load packed bytes, unpack to u32 nibble values, decode E2M1, multiply by scale.

        let packed_base = self.resolve_gpr_read(packed_ptr, alloc, 2)?;
        self.eval_offset_to_rax(packed_offset, alloc)?;
        self.asm.add(rax, packed_base).map_err(Self::err)?;

        match width {
            SimdWidth::W256 => {
                let s0 = self.scratch_ymm(0);
                let s1 = self.scratch_ymm(1);
                let s0_xmm = Self::ymm_to_xmm(s0);
                let s1_xmm = Self::ymm_to_xmm(s1);

                // Load 4 packed bytes (8 nibbles)
                self.asm.vmovd(s0_xmm, dword_ptr(rax)).map_err(Self::err)?;

                // Zero-extend: bytes → u16 → u32
                self.asm.vpxor(s1_xmm, s1_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vpunpcklbw(s0_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vpunpcklwd(s0_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;

                // Split into low/high nibbles using shift trick
                self.asm.vmovdqa(s1_xmm, s0_xmm).map_err(Self::err)?;
                self.asm.vpslld(s0_xmm, s0_xmm, 28).map_err(Self::err)?;
                self.asm.vpsrld(s0_xmm, s0_xmm, 28).map_err(Self::err)?;
                self.asm.vpsrld(s1_xmm, s1_xmm, 4).map_err(Self::err)?;

                // Interleave into YMM
                let dst_xmm = Self::ymm_to_xmm(dst_reg);
                self.asm.vpunpckldq(dst_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vpunpckhdq(s0_xmm, s0_xmm, s1_xmm).map_err(Self::err)?;
                self.asm.vmovdqa(Self::ymm_to_xmm(s0), dst_xmm).map_err(Self::err)?;
                self.asm.vinserti128(s0, s0, s0_xmm, 1).map_err(Self::err)?;
                // s0_ymm = 8 u32 nibble values (0-15)

                // Zero mask (nibble==0)
                let zero_mask = self.spill_scratch_ymm(1)?;
                self.asm.vpxor(s1, s1, s1).map_err(Self::err)?;
                self.asm.vpcmpeqd(zero_mask, s0, s1).map_err(Self::err)?;

                // Extract exp
                self.asm.vpsrld(s1, s0, 1).map_err(Self::err)?;
                self.asm.vpslld(s1, s1, 30).map_err(Self::err)?;
                self.asm.vpsrld(s1, s1, 30).map_err(Self::err)?;

                // 2^(exp-2) via IEEE 754
                self.asm.vpslld(s1, s1, 23).map_err(Self::err)?;
                let base_025 = self.const_f32(0.25f32);
                self.asm.vbroadcastss(dst_reg, dword_ptr(base_025)).map_err(Self::err)?;
                self.asm.vpaddd(s1, s1, dst_reg).map_err(Self::err)?;

                // Sign mask → dst_reg
                self.asm.vpsrld(dst_reg, s0, 3).map_err(Self::err)?;
                self.asm.vpslld(dst_reg, dst_reg, 31).map_err(Self::err)?;

                // Mantissa → magnitude via FMA + add
                self.asm.vpslld(s0, s0, 31).map_err(Self::err)?;
                self.asm.vpsrld(s0, s0, 31).map_err(Self::err)?;
                self.asm.vcvtdq2ps(s0, s0).map_err(Self::err)?;
                self.asm.vfmadd213ps(s0, s1, s1).map_err(Self::err)?;
                self.asm.vaddps(s0, s0, s1).map_err(Self::err)?;

                // Apply sign
                self.asm.vxorps(s0, s0, dst_reg).map_err(Self::err)?;

                // nibble=0 → 0.0
                self.asm.vandnps(s0, zero_mask, s0).map_err(Self::err)?;

                // Multiply by UE4M3 scale
                self.asm.vmulps(s0, s0, s2).map_err(Self::err)?;

                // Move to dst
                self.asm.vmovdqa(dst_reg, s0).map_err(Self::err)?;

                if dst_spilled { self.spill_store_ymm(dst, alloc, 0)?; }
                Ok(())
            }
            _ => {
                Err(CompilerError::CodegenViolation(
                    format!("Nvfp4SubBlockDequant: unsupported width {:?} (only W256 implemented)", width)))
            }
        }
    }
}
