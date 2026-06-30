impl AArch64Lower {
    // ══════════════════════════════════════════════════════════════════
    //  指令翻译
    // ══════════════════════════════════════════════════════════════════

    // ARCH-LOWER-DISPATCH-LAYERING (BCE-20260630-LOWER-INSTR-GOD-MATCH):
    // L0 分类 dispatch — 巨型 VmInstr match 已拆为 L0(分类) → L1(变体路由) → L2(叶子 emit)。
    // 新增 VmInstr 变体 = category() 补一行 + lower_<variant>_aarch64 一个叶子 fn。
    // 禁止在此处内联变体逻辑 (OCP); 禁止 catch-all 静默 NOP (NO_SILENT_FALLBACK)。
    // L1 变体路由 + L2 叶子 emit 见 lower_instr_dispatch.inc.rs。

    pub fn lower_instr(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match instr.category() {
            super::vm_instr_category::InstrCategory::Memory => self.lower_memory_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::Arith => self.lower_arith_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::Control => self.lower_control_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::Tile => self.lower_tile_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::Quant => self.lower_quant_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::GpuComm => self.lower_gpu_comm_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::Sampling => self.lower_sampling_aarch64(instr, alloc),
            super::vm_instr_category::InstrCategory::Misc => self.lower_misc_aarch64(instr, alloc),
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
