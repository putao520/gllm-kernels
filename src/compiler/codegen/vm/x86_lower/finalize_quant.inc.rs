impl X86Lower {
    pub fn take_source_map(&mut self) -> super::debug_map::JitSourceMap {
        std::mem::take(&mut self.source_map)
    }

    pub fn finalize(mut self) -> Result<Vec<u8>, CompilerError> {
        // SPEC 15 REQ-JCTX-022: 编译后资源泄漏检测
        let report = self.jit_ctx.usage_report();
        if !report.warnings.is_empty() {
            for w in &report.warnings {
                match w {
                    crate::compiler::jit_context::ResourceWarning::NearExhaustion { kind, peak, capacity } => {
                        eprintln!("[JitContext] WARNING: {:?} near exhaustion: {}/{}", kind, peak, capacity);
                    }
                    crate::compiler::jit_context::ResourceWarning::Leak { kind, instance, purpose } => {
                        eprintln!("[JitContext] WARNING: {:?} leak: instance {} ({}) not released", kind, instance, purpose);
                    }
                }
            }
        }
        eprintln!("[X86-LOWER] const_pool entries: {}, instructions before pool: ~{}", self.const_pool.len(), self.asm.instructions().len());
        // 常量池在 ret 之后。
        // iced set_label + db() 交互：ret() 生成 C3 (1 byte)，但 set_label
        // 可能将该 label 绑定到 ret 指令自身的地址而非 ret 之后的地址。
        // 插入 NOP 确保下一个 set_label 位置在 ret 之后。
        self.asm.nop().map_err(Self::err)?;
        for i in 0..self.const_pool.len() {
            self.asm.set_label(&mut self.const_pool[i].1).map_err(Self::err)?;
            for &val in &self.const_pool[i].0 {
                self.asm.db(&val.to_le_bytes()).map_err(Self::err)?;
            }
        }
        let code = self.asm.assemble(0x0).map_err(|e| CompilerError::Internal(format!("assemble: {e}")))?;
        Ok(code)
    }

    /// GatherLoad: 从 base + indices[i]*stride 加载 lanes 个 f32 到 dst 向量。
    /// 逐元素加载到 xmm0，存入 dst 的 spill slot，最后 VecLoad 到 dst。
    fn emit_gather_load(
        &mut self,
        dst: VRegId,
        base: VRegId,
        indices: VRegId,
        stride: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let lanes = width.f32_lanes();
        let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
        let idx_base = self.resolve_gpr_read(indices, alloc, 3)?;
        // 获取 dst 的 spill 区域 rbp 偏移（如果已 spill），
        // 否则使用 spill scratch slot 0 的区域（scratch_vec_ids[3] 对应 ymm 区域）
        let tmp_rbp: i32 = match self.spill_offset(dst, alloc) {
            Some(off) => off,
            None => {
                // dst 未 spill：使用 spill scratch slot 0 (ymm12) 作为临时存储
                // 但这里需要栈地址，所以分配一个临时栈槽
                // TODO: 未来可以优化为直接使用 ymm12 而非栈
                let scratch_stack_size = 32; // 8 floats × 4 bytes
                self.stack_layout.spill_rbp_offset(0, scratch_stack_size)
            }
        };
        for i in 0..lanes {
            let idx_byte_off = (i * 4) as i32;
            self.asm.mov(eax, dword_ptr(idx_base + idx_byte_off)).map_err(Self::err)?;
            if stride != 1 {
                self.asm.mov(r10d, stride as u32).map_err(Self::err)?;
                self.asm.imul_2(eax, r10d).map_err(Self::err)?;
            }
            self.asm.movsxd(rax, eax).map_err(Self::err)?;
            self.asm.shl(rax, 2).map_err(Self::err)?;
            self.asm.add(rax, base_reg).map_err(Self::err)?;
            self.asm.vmovss(xmm0, dword_ptr(rax)).map_err(Self::err)?;
            let store_off = tmp_rbp + (i * 4) as i32;
            self.asm.vmovss(dword_ptr(rbp + store_off), xmm0).map_err(Self::err)?;
        }
        let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 0)?;
        match width {
            SimdWidth::W512 => {
                let (dst_zmm, _) = self.resolve_zmm_or_spill_write(dst, alloc, 0)?;
                self.asm.vmovups(dst_zmm, zmmword_ptr(rbp + tmp_rbp)).map_err(Self::err)?;
            }
            SimdWidth::W256 => {
                self.asm.vmovups(dst_ymm, ymmword_ptr(rbp + tmp_rbp)).map_err(Self::err)?;
            }
            SimdWidth::Scalar => {
                self.asm.vmovss(Self::ymm_to_xmm(dst_ymm), dword_ptr(rbp + tmp_rbp)).map_err(Self::err)?;
            }
            _ => {
                self.asm.vmovups(dst_ymm, ymmword_ptr(rbp + tmp_rbp)).map_err(Self::err)?;
            }
        }
        if dst_spilled { self.spill_store_ymm(dst, alloc, 0)?; }
        Ok(())
    }

    /// ScatterStore: 将 src 向量的 lanes 个 f32 按 indices 写入 base + indices[i]*stride。
    fn emit_scatter_store(
        &mut self,
        base: VRegId,
        indices: VRegId,
        src: VRegId,
        stride: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let lanes = width.f32_lanes();
        let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
        let idx_base = self.resolve_gpr_read(indices, alloc, 3)?;
        let (src_ymm, _) = self.resolve_ymm_or_spill(src, alloc, 0)?;
        let src_xmm = Self::ymm_to_xmm(src_ymm);
        for i in 0..lanes {
            let idx_byte_off = (i * 4) as i32;
            self.asm.mov(eax, dword_ptr(idx_base + idx_byte_off)).map_err(Self::err)?;
            if stride != 1 {
                self.asm.mov(r10d, stride as u32).map_err(Self::err)?;
                self.asm.imul_2(eax, r10d).map_err(Self::err)?;
            }
            self.asm.movsxd(rax, eax).map_err(Self::err)?;
            self.asm.shl(rax, 2).map_err(Self::err)?;
            self.asm.add(rax, base_reg).map_err(Self::err)?;
            // vextractps 只接受 xmm 寄存器，从 ymm 低 128 位提取
            // 对于 i >= 4，需要先 vextractf128 到 xmm，再 vextractps
            if i < 4 {
                self.asm.vextractps(dword_ptr(rax), src_xmm, i as i32).map_err(Self::err)?;
            } else {
                // 提取高 128 位到临时 xmm0
                self.asm.vextractf128(xmm0, src_ymm, 1).map_err(Self::err)?;
                self.asm.vextractps(dword_ptr(rax), xmm0, (i - 4) as i32).map_err(Self::err)?;
            }
        }
        Ok(())
    }

    /// TableLookup: 从 base + row_index * row_bytes 加载一行 SIMD 向量。
    fn emit_table_lookup(
        &mut self,
        dst: VRegId,
        base: VRegId,
        row_index: VRegId,
        row_bytes: usize,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
        let idx_reg = self.resolve_gpr_read(row_index, alloc, 3)?;
        self.asm.mov(rax, idx_reg).map_err(Self::err)?;
        self.asm.mov(r10, row_bytes as u64).map_err(Self::err)?;
        self.asm.imul_2(rax, r10).map_err(Self::err)?;
        self.asm.add(rax, base_reg).map_err(Self::err)?;
        let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 0)?;
        match width {
            SimdWidth::W512 => {
                let (dst_zmm, _) = self.resolve_zmm_or_spill_write(dst, alloc, 0)?;
                self.asm.vmovups(dst_zmm, zmmword_ptr(rax)).map_err(Self::err)?;
            }
            _ => {
                self.asm.vmovups(dst_ymm, ymmword_ptr(rax)).map_err(Self::err)?;
            }
        }
        if dst_spilled { self.spill_store_ymm(dst, alloc, 0)?; }
        Ok(())
    }

    /// KIVI per-channel 4-bit 量化: 逐元素标量提取 → max → pack → store
    ///
    /// 算法:
    ///   scale[i] = max(|src[i]|) per channel-pair (2 channels share 1 scale)
    ///   packed[i/2] = (src[i] / scale[i/2]) * 7.0 + 8.0 (4-bit unsigned [0,15])
    ///   存 scale 为 f16 (2 bytes), packed data 为 nibble pairs
    ///
    /// 寄存器分配:
    ///   rax: 临时 (元素值读取)
    ///   rcx: 循环计数器
    ///   rdx: dst_ptr
    ///   rsi: scale_ptr
    ///   r8:  src 基址 (spill 或寄存器偏移)
    fn lower_kivi_quant_channel(
        &mut self,
        src: &VRegId,
        dst_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_channels: usize,
        _width: &SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_gpr = self.resolve_gpr_read(*dst_ptr, alloc, 0)?;
        let scale_gpr = self.resolve_gpr_read(*scale_ptr, alloc, 1)?;
        let src_offset = self.spill_offset(*src, alloc)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("KiviQuantChannel: src v{} not spilled (need stack-backed array)", src.0)
            ))?;

        let num_pairs = (num_channels + 1) / 2;

        self.asm.xor(ecx, ecx).map_err(Self::err)?;
        self.asm.mov(rdx, dst_gpr).map_err(Self::err)?;
        self.asm.mov(rsi, scale_gpr).map_err(Self::err)?;

        let mut loop_label = self.asm.create_label();
        let mut end_label = self.asm.create_label();

        self.asm.set_label(&mut loop_label).map_err(Self::err)?;
        self.asm.cmp(ecx, num_pairs as u32).map_err(Self::err)?;
        self.asm.jae(end_label).map_err(Self::err)?;

        // scale = max(|val_2i|, |val_2i+1|)
        self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
        self.asm.and(eax, 0x7FFF_FFFFu32).map_err(Self::err)?;
        self.asm.mov(r8d, eax).map_err(Self::err)?;

        self.asm.mov(eax, dword_ptr(rbp + (src_offset + 4))).map_err(Self::err)?;
        self.asm.and(eax, 0x7FFF_FFFFu32).map_err(Self::err)?;
        self.asm.cmp(r8d, eax).map_err(Self::err)?;
        self.asm.cmovb(r8d, eax).map_err(Self::err)?;

        // store f32 scale
        self.asm.mov(dword_ptr(rsi), r8d).map_err(Self::err)?;
        self.asm.add(rsi, 4i32).map_err(Self::err)?;

        // nibble_0 = (val_2i >> 20) & 0xF
        self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
        self.asm.shr(eax, 20).map_err(Self::err)?;
        self.asm.and(eax, 0xFu32).map_err(Self::err)?;
        self.asm.mov(r9d, eax).map_err(Self::err)?;

        // nibble_1 = (val_2i+1 >> 20) & 0xF
        self.asm.mov(eax, dword_ptr(rbp + (src_offset + 4))).map_err(Self::err)?;
        self.asm.shr(eax, 20).map_err(Self::err)?;
        self.asm.and(eax, 0xFu32).map_err(Self::err)?;
        self.asm.shl(eax, 4).map_err(Self::err)?;
        self.asm.or(eax, r9d).map_err(Self::err)?;

        self.asm.mov(byte_ptr(rdx), al).map_err(Self::err)?;
        self.asm.inc(rdx).map_err(Self::err)?;

        self.asm.inc(ecx).map_err(Self::err)?;
        self.asm.jmp(loop_label).map_err(Self::err)?;

        self.asm.set_label(&mut end_label).map_err(Self::err)?;
        Ok(())
    }

    /// KIVI per-token 4-bit 量化
    fn lower_kivi_quant_token(
        &mut self,
        src: &VRegId,
        dst_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_elems: usize,
        _width: &SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_gpr = self.resolve_gpr_read(*dst_ptr, alloc, 0)?;
        let scale_gpr = self.resolve_gpr_read(*scale_ptr, alloc, 1)?;
        let src_offset = self.spill_offset(*src, alloc)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("KiviQuantToken: src v{} not spilled", src.0)
            ))?;

        // Pass 1: find max(|elem[i]|)
        self.asm.xor(ecx, ecx).map_err(Self::err)?;
        self.asm.xor(r8d, r8d).map_err(Self::err)?;

        let mut max_loop = self.asm.create_label();
        let mut max_done = self.asm.create_label();

        self.asm.set_label(&mut max_loop).map_err(Self::err)?;
        self.asm.cmp(ecx, num_elems as u32).map_err(Self::err)?;
        self.asm.jae(max_done).map_err(Self::err)?;

        self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
        self.asm.and(eax, 0x7FFF_FFFFu32).map_err(Self::err)?;
        self.asm.cmp(r8d, eax).map_err(Self::err)?;
        self.asm.cmovb(r8d, eax).map_err(Self::err)?;

        self.asm.inc(ecx).map_err(Self::err)?;
        self.asm.jmp(max_loop).map_err(Self::err)?;

        self.asm.set_label(&mut max_done).map_err(Self::err)?;
        self.asm.mov(dword_ptr(scale_gpr), r8d).map_err(Self::err)?;

        // Pass 2: pack into nibble pairs
        self.asm.xor(ecx, ecx).map_err(Self::err)?;
        self.asm.mov(rdx, dst_gpr).map_err(Self::err)?;

        let mut pack_loop = self.asm.create_label();
        let mut pack_done = self.asm.create_label();
        let num_pairs = (num_elems + 1) / 2;

        self.asm.set_label(&mut pack_loop).map_err(Self::err)?;
        self.asm.cmp(ecx, num_pairs as u32).map_err(Self::err)?;
        self.asm.jae(pack_done).map_err(Self::err)?;

        self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
        self.asm.shr(eax, 20).map_err(Self::err)?;
        self.asm.and(eax, 0xFu32).map_err(Self::err)?;
        self.asm.mov(r9d, eax).map_err(Self::err)?;

        self.asm.mov(eax, dword_ptr(rbp + (src_offset + 4))).map_err(Self::err)?;
        self.asm.shr(eax, 20).map_err(Self::err)?;
        self.asm.and(eax, 0xFu32).map_err(Self::err)?;
        self.asm.shl(eax, 4).map_err(Self::err)?;
        self.asm.or(eax, r9d).map_err(Self::err)?;

        self.asm.mov(byte_ptr(rdx), al).map_err(Self::err)?;
        self.asm.inc(rdx).map_err(Self::err)?;
        self.asm.inc(ecx).map_err(Self::err)?;
        self.asm.jmp(pack_loop).map_err(Self::err)?;

        self.asm.set_label(&mut pack_done).map_err(Self::err)?;
        Ok(())
    }

    /// KIVI 4-bit 反量化 load: packed 4-bit + f32 scale → f32 向量
    fn lower_kivi_dequant_load(
        &mut self,
        dst: &VRegId,
        src_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_elems: usize,
        _width: &SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let src_gpr = self.resolve_gpr_read(*src_ptr, alloc, 0)?;
        let scale_gpr = self.resolve_gpr_read(*scale_ptr, alloc, 1)?;
        let dst_offset = self.spill_offset(*dst, alloc)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("KiviDequantLoad: dst v{} not spilled", dst.0)
            ))?;

        // Load scale (f32 bits)
        self.asm.mov(r8d, dword_ptr(scale_gpr)).map_err(Self::err)?;

        // Loop over pairs, unpack nibbles
        self.asm.xor(ecx, ecx).map_err(Self::err)?;
        self.asm.mov(rdx, src_gpr).map_err(Self::err)?;

        let mut unpack_loop = self.asm.create_label();
        let mut unpack_done = self.asm.create_label();
        let num_pairs = (num_elems + 1) / 2;

        self.asm.set_label(&mut unpack_loop).map_err(Self::err)?;
        self.asm.cmp(ecx, num_pairs as u32).map_err(Self::err)?;
        self.asm.jae(unpack_done).map_err(Self::err)?;

        // Load packed byte
        self.asm.movzx(eax, byte_ptr(rdx)).map_err(Self::err)?;

        // nibble_0 = eax & 0xF
        self.asm.mov(r9d, eax).map_err(Self::err)?;
        self.asm.and(r9d, 0xFu32).map_err(Self::err)?;

        // nibble_1 = (eax >> 4) & 0xF
        self.asm.shr(eax, 4).map_err(Self::err)?;
        self.asm.and(eax, 0xFu32).map_err(Self::err)?;

        // elem[0] = reconstruct(nibble_0, scale): approximate f32 via mantissa injection
        self.asm.mov(r10d, r9d).map_err(Self::err)?;
        self.asm.shl(r10d, 20).map_err(Self::err)?;
        self.asm.or(r10d, r8d).map_err(Self::err)?;
        self.asm.mov(dword_ptr(rbp + dst_offset), r10d).map_err(Self::err)?;

        // elem[1] = reconstruct(nibble_1, scale)
        self.asm.shl(eax, 20).map_err(Self::err)?;
        self.asm.or(eax, r8d).map_err(Self::err)?;
        self.asm.mov(dword_ptr(rbp + (dst_offset + 4)), eax).map_err(Self::err)?;

        self.asm.inc(rdx).map_err(Self::err)?;
        self.asm.inc(ecx).map_err(Self::err)?;
        self.asm.jmp(unpack_loop).map_err(Self::err)?;

        self.asm.set_label(&mut unpack_done).map_err(Self::err)?;
        Ok(())
    }

    /// KIVI quantized KV page write-back: quantize FP32 src → 4-bit packed + scale, write to page.
    ///
    /// packed_addr and scale_addr are pre-computed in GPRs (s0, s1).
    /// For Kivi4 (per-channel): each pair gets its own scale (same as KiviQuantChannel).
    /// For Kivi2 (per-token): one scale for all elements (same as KiviQuantToken).
    ///
    /// Register allocation:
    ///   rax (s0): packed_data_addr (preserved)
    ///   r10 (s1): scale_addr (preserved)
    ///   ecx: loop counter
    ///   edx: packed write pointer
    ///   r8d: scale accumulator / scale value
    ///   r9d, eax: temporaries
    fn lower_kivi_quant_page_write(
        &mut self,
        src: &VRegId,
        packed_addr: AsmRegister64,
        scale_addr: AsmRegister64,
        num_elems: usize,
        kivi_mode: KvLoadMode,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let src_offset = self.spill_offset(*src, alloc)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("PageTableKVWriteQuant: src v{} not spilled (need stack-backed array)", src.0)
            ))?;

        let s0 = packed_addr;
        let s1 = scale_addr;

        match kivi_mode {
            KvLoadMode::Kivi4 => {
                // Per-channel 4-bit: each element pair gets its own scale.
                // Identical logic to lower_kivi_quant_channel but writes to pre-computed addrs.
                let num_pairs = (num_elems + 1) / 2;

                // Save packed_addr to rdx for byte-store loop
                self.asm.mov(rdx, s0).map_err(Self::err)?;
                // rsi = scale_addr for scale-store loop
                self.asm.mov(rsi, s1).map_err(Self::err)?;
                self.asm.xor(ecx, ecx).map_err(Self::err)?;

                let mut loop_label = self.asm.create_label();
                let mut end_label = self.asm.create_label();

                self.asm.set_label(&mut loop_label).map_err(Self::err)?;
                self.asm.cmp(ecx, num_pairs as u32).map_err(Self::err)?;
                self.asm.jae(end_label).map_err(Self::err)?;

                // scale = max(|val_2i|, |val_2i+1|)
                self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
                self.asm.and(eax, 0x7FFF_FFFFu32).map_err(Self::err)?;
                self.asm.mov(r8d, eax).map_err(Self::err)?;

                self.asm.mov(eax, dword_ptr(rbp + (src_offset + 4))).map_err(Self::err)?;
                self.asm.and(eax, 0x7FFF_FFFFu32).map_err(Self::err)?;
                self.asm.cmp(r8d, eax).map_err(Self::err)?;
                self.asm.cmovb(r8d, eax).map_err(Self::err)?;

                // store f32 scale
                self.asm.mov(dword_ptr(rsi), r8d).map_err(Self::err)?;
                self.asm.add(rsi, 4i32).map_err(Self::err)?;

                // nibble_0 = (val_2i >> 20) & 0xF
                self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
                self.asm.shr(eax, 20).map_err(Self::err)?;
                self.asm.and(eax, 0xFu32).map_err(Self::err)?;
                self.asm.mov(r9d, eax).map_err(Self::err)?;

                // nibble_1 = (val_2i+1 >> 20) & 0xF
                self.asm.mov(eax, dword_ptr(rbp + (src_offset + 4))).map_err(Self::err)?;
                self.asm.shr(eax, 20).map_err(Self::err)?;
                self.asm.and(eax, 0xFu32).map_err(Self::err)?;
                self.asm.shl(eax, 4).map_err(Self::err)?;
                self.asm.or(eax, r9d).map_err(Self::err)?;

                self.asm.mov(byte_ptr(rdx), al).map_err(Self::err)?;
                self.asm.inc(rdx).map_err(Self::err)?;

                self.asm.inc(ecx).map_err(Self::err)?;
                self.asm.jmp(loop_label).map_err(Self::err)?;

                self.asm.set_label(&mut end_label).map_err(Self::err)?;
            }
            KvLoadMode::Kivi2 => {
                // Per-token: one scale for all elements.
                // Pass 1: find max(|elem[i]|)
                self.asm.xor(ecx, ecx).map_err(Self::err)?;
                self.asm.xor(r8d, r8d).map_err(Self::err)?;

                let mut max_loop = self.asm.create_label();
                let mut max_done = self.asm.create_label();

                self.asm.set_label(&mut max_loop).map_err(Self::err)?;
                self.asm.cmp(ecx, num_elems as u32).map_err(Self::err)?;
                self.asm.jae(max_done).map_err(Self::err)?;

                self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
                self.asm.and(eax, 0x7FFF_FFFFu32).map_err(Self::err)?;
                self.asm.cmp(r8d, eax).map_err(Self::err)?;
                self.asm.cmovb(r8d, eax).map_err(Self::err)?;

                self.asm.inc(ecx).map_err(Self::err)?;
                self.asm.jmp(max_loop).map_err(Self::err)?;

                self.asm.set_label(&mut max_done).map_err(Self::err)?;
                // Store single scale to scale_addr
                self.asm.mov(dword_ptr(s1), r8d).map_err(Self::err)?;

                // Pass 2: pack into nibble pairs
                self.asm.xor(ecx, ecx).map_err(Self::err)?;
                self.asm.mov(rdx, s0).map_err(Self::err)?;

                let mut pack_loop = self.asm.create_label();
                let mut pack_done = self.asm.create_label();
                let num_pairs = (num_elems + 1) / 2;

                self.asm.set_label(&mut pack_loop).map_err(Self::err)?;
                self.asm.cmp(ecx, num_pairs as u32).map_err(Self::err)?;
                self.asm.jae(pack_done).map_err(Self::err)?;

                self.asm.mov(eax, dword_ptr(rbp + src_offset)).map_err(Self::err)?;
                self.asm.shr(eax, 20).map_err(Self::err)?;
                self.asm.and(eax, 0xFu32).map_err(Self::err)?;
                self.asm.mov(r9d, eax).map_err(Self::err)?;

                self.asm.mov(eax, dword_ptr(rbp + (src_offset + 4))).map_err(Self::err)?;
                self.asm.shr(eax, 20).map_err(Self::err)?;
                self.asm.and(eax, 0xFu32).map_err(Self::err)?;
                self.asm.shl(eax, 4).map_err(Self::err)?;
                self.asm.or(eax, r9d).map_err(Self::err)?;

                self.asm.mov(byte_ptr(rdx), al).map_err(Self::err)?;
                self.asm.inc(rdx).map_err(Self::err)?;
                self.asm.inc(ecx).map_err(Self::err)?;
                self.asm.jmp(pack_loop).map_err(Self::err)?;

                self.asm.set_label(&mut pack_done).map_err(Self::err)?;
            }
            _ => {
                return Err(CompilerError::CodegenViolation(
                    format!("PageTableKVWriteQuant: invalid kivi_mode {:?}", kivi_mode)
                ));
            }
        }
        Ok(())
    }

    fn lower_quant_block_load(
        &mut self,
        dst: VRegId,
        base: VRegId,
        offset: &OffsetExpr,
        unpack: &BlockUnpackMode,
        width: &SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        match unpack {
            BlockUnpackMode::Int8 => {                // ARCH-ISA-SCRATCH: base 走 scratch slot 2 (r11), 避开
                // eval_offset_to_rax 使用的 slot 0/1 (rax/r10)。否则
                // offset 是嵌套 Add 时 `mov s1, s0` 会覆盖 base 值。
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;
                // vpmovsxbd ymm, m64: sign-extend 8 x i8 → 8 x i32
                self.asm.vpmovsxbd(dst_ymm, qword_ptr(rax)).map_err(Self::err)?;
                // vcvtdq2ps ymm, ymm: convert i32 → f32
                self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;
                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::F16Broadcast => {                // ARCH-ISA-SCRATCH: base 走 scratch slot 2 (r11), 避开
                // eval_offset_to_rax 使用的 slot 0/1 (rax/r10)。
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;
                let dst_xmm = Self::ymm_to_xmm(dst_ymm);
                // Load 4 bytes (contains f16 scale at offset 0) into xmm low bits
                self.asm.vmovd(dst_xmm, dword_ptr(rax)).map_err(Self::err)?;
                // vcvtph2ps xmm, xmm: convert low 2 x f16 → 2 x f32
                self.asm.vcvtph2ps(dst_xmm, dst_xmm).map_err(Self::err)?;
                // vbroadcastss ymm, xmm[0]: broadcast first f32 (the scale) to all lanes
                self.asm.vbroadcastss(dst_ymm, dst_xmm).map_err(Self::err)?;
                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::SignedNibbleLow => {                // GGUF PackedNibbles low-nibble load: load lanes bytes, extract LOW nibbles (& 0x0F),
                // subtract 8.0, produce lanes F32 values.
                // GGUF split layout: positions 0..block_size/2 use low nibbles of sequential bytes.
                // ARCH-ISA-SCRATCH: base → slot 2 (r11), scratch → slot 0/1 (rax/r10)
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;

                let lanes = width.f32_lanes().max(1);
                if lanes == 8 {
                    // AVX2 path: 8 bytes → 8 low nibbles → 8 × f32
                    let dst_xmm = Self::ymm_to_xmm(dst_ymm);
                    let scratch_xmm = self.scratch_xmm(0);
                    let mask_ymm = self.scratch_ymm(0);   // ymm15: AND mask / 8.0 constant

                    // 1. Load 8 bytes → 8 × i32 via vpmovzxbd (256-bit)
                    self.asm.vmovq(dst_xmm, qword_ptr(rax)).map_err(Self::err)?;
                    self.asm.vpmovzxbd(dst_ymm, dst_xmm).map_err(Self::err)?;

                    // 2. AND with 0x0F → low nibbles only
                    self.asm.mov(rax, 0x0F0F0F0Fi64 as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vpbroadcastd(mask_ymm, scratch_xmm).map_err(Self::err)?;
                    self.asm.vpand(dst_ymm, dst_ymm, mask_ymm).map_err(Self::err)?;

                    // 3. Convert i32 → f32
                    self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;

                    // 4. Subtract 8.0 (Q4_0/Q4_1 symmetric zero-point)
                    self.asm.mov(rax, f32::to_bits(8.0f32) as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vbroadcastss(mask_ymm, scratch_xmm).map_err(Self::err)?;
                    self.asm.vsubps(dst_ymm, dst_ymm, mask_ymm).map_err(Self::err)?;
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt4Load: only AVX2 (8 lanes) supported, got {} lanes", lanes)
                    ));
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::SignedNibbleHigh => {                // GGUF PackedNibbles high-nibble load: load lanes bytes, extract HIGH nibbles (>> 4),
                // subtract 8.0, produce lanes F32 values.
                // GGUF split layout: positions block_size/2..block_size use high nibbles.
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;

                let lanes = width.f32_lanes().max(1);
                if lanes == 8 {
                    let dst_xmm = Self::ymm_to_xmm(dst_ymm);
                    let scratch_xmm = self.scratch_xmm(0);
                    let mask_ymm = self.scratch_ymm(0);

                    // 1. Load 8 bytes → 8 × i32 via vpmovzxbd (256-bit)
                    self.asm.vmovq(dst_xmm, qword_ptr(rax)).map_err(Self::err)?;
                    self.asm.vpmovzxbd(dst_ymm, dst_xmm).map_err(Self::err)?;

                    // 2. Shift right by 4 → high nibbles only
                    self.asm.vpsrld(dst_ymm, dst_ymm, 4).map_err(Self::err)?;

                    // 3. Convert i32 → f32
                    self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;

                    // 4. Subtract 8.0 (symmetric zero-point)
                    self.asm.mov(rax, f32::to_bits(8.0f32) as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vbroadcastss(mask_ymm, scratch_xmm).map_err(Self::err)?;
                    self.asm.vsubps(dst_ymm, dst_ymm, mask_ymm).map_err(Self::err)?;
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt4HighLoad: only AVX2 (8 lanes) supported, got {} lanes", lanes)
                    ));
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::UnsignedNibbleLow => {                // GGUF Unsigned packed nibbles load: load lanes bytes, extract LOW nibbles (& 0x0F),
                // convert to f32 (no zero-point subtraction — unsigned range [0..15]).
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;

                let lanes = width.f32_lanes().max(1);
                if lanes == 8 {
                    let dst_xmm = Self::ymm_to_xmm(dst_ymm);
                    let scratch_xmm = self.scratch_xmm(0);
                    let mask_ymm = self.scratch_ymm(0);

                    // 1. Load 8 bytes → 8 × i32 via vpmovzxbd (256-bit)
                    self.asm.vmovq(dst_xmm, qword_ptr(rax)).map_err(Self::err)?;
                    self.asm.vpmovzxbd(dst_ymm, dst_xmm).map_err(Self::err)?;

                    // 2. AND with 0x0F → low nibbles only
                    self.asm.mov(rax, 0x0F0F0F0Fi64 as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vpbroadcastd(mask_ymm, scratch_xmm).map_err(Self::err)?;
                    self.asm.vpand(dst_ymm, dst_ymm, mask_ymm).map_err(Self::err)?;

                    // 3. Convert i32 → f32 (no subtract — unsigned)
                    self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufUInt4Load: only AVX2 (8 lanes) supported, got {} lanes", lanes)
                    ));
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::UnsignedNibbleHigh => {                // GGUF Unsigned packed nibbles high-nibble load: load lanes bytes, extract HIGH nibbles (>> 4),
                // convert to f32 (no zero-point subtraction — unsigned range [0..15]).
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;

                let lanes = width.f32_lanes().max(1);
                if lanes == 8 {
                    let dst_xmm = Self::ymm_to_xmm(dst_ymm);

                    // 1. Load 8 bytes → 8 × i32 via vpmovzxbd (256-bit)
                    self.asm.vmovq(dst_xmm, qword_ptr(rax)).map_err(Self::err)?;
                    self.asm.vpmovzxbd(dst_ymm, dst_xmm).map_err(Self::err)?;

                    // 2. Shift right by 4 → high nibbles only
                    self.asm.vpsrld(dst_ymm, dst_ymm, 4).map_err(Self::err)?;

                    // 3. Convert i32 → f32 (no subtract — unsigned)
                    self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufUInt4HighLoad: only AVX2 (8 lanes) supported, got {} lanes", lanes)
                    ));
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::Bitpack2 { bias } => {                // Q2K 2-bit packed → i32 → f32, subtract bias.
                // Each byte has 4 × 2-bit values. For 8 lanes: 2 bytes → 8 × i32 → f32.
                let lanes = width.f32_lanes().max(1);
                if lanes != 8 {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt2Load: only 8 lanes supported, got {}", lanes)
                    ));
                }

                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let qs_reg = self.resolve_gpr_read(base, alloc, 2)?;

                // Load 2 bytes (16 bits = 8 × 2-bit values) from base
                // Scalar extraction into stack, then load as ymm vector
                self.asm.sub(rsp, 32).map_err(Self::err)?;

                let tmp_gpr = r10;
                // Load 2 bytes into rax
                self.asm.movzx(rax, word_ptr(qs_reg)).map_err(Self::err)?;

                // Extract 8 × 2-bit values: byte[i/4] bits [(i%4)*2+1 : (i%4)*2]
                for i in 0..8 {
                    let byte_idx = i / 4;
                    let bit_shift = (i % 4) * 2 + byte_idx * 8;
                    self.asm.mov(tmp_gpr, rax).map_err(Self::err)?;
                    self.asm.shr(tmp_gpr, bit_shift as u32).map_err(Self::err)?;
                    self.asm.and(tmp_gpr, 3i32).map_err(Self::err)?;
                    // Store as i32 at rsp + i*4
                    self.asm.mov(dword_ptr(rsp + i * 4), tmp_gpr).map_err(Self::err)?;
                }

                // Load 8 × i32 from stack → ymm
                self.asm.vmovdqu(dst_ymm, ymmword_ptr(rsp)).map_err(Self::err)?;

                // Restore stack
                self.asm.add(rsp, 32).map_err(Self::err)?;

                // Convert i32 → f32
                self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;

                // Subtract bias
                if *bias != 0.0 {
                    let s_xmm = self.scratch_xmm(0);
                    let s_ymm = self.scratch_ymm(0);
                    self.asm.mov(rax, f32::to_bits(*bias) as u64).map_err(Self::err)?;
                    self.asm.vmovd(s_xmm, eax).map_err(Self::err)?;
                    self.asm.vbroadcastss(s_ymm, s_xmm).map_err(Self::err)?;
                    self.asm.vsubps(dst_ymm, dst_ymm, s_ymm).map_err(Self::err)?;
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BlockUnpackMode::Mxfp4 { scale_src } => {
                self.emit_mxfp4_dequant(dst, base, offset, *scale_src, *width, alloc)
            }

            BlockUnpackMode::Nvfp4 { scale_src } => {
                self.emit_nvfp4_sub_block_dequant(dst, base, offset, *scale_src, *width, alloc)
            }

            BlockUnpackMode::QhBitExpand { bit_value } => {
                let lanes = width.f32_lanes().max(1);
                if lanes != 8 {
                    return Err(CompilerError::CodegenViolation(
                        format!("QhBitExpand: only 8 lanes supported, got {}", lanes)
                    ));
                }
                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let base_reg = self.resolve_gpr_read(base, alloc, 2)?;
                self.eval_offset_to_rax(offset, alloc)?;
                self.asm.add(rax, base_reg).map_err(Self::err)?;

                // Save qh address in a scratch GPR before using rax for constants
                let addr_reg = self.scratch_gprs[1]; // r10
                self.asm.mov(addr_reg, rax).map_err(Self::err)?;

                let scratch_xmm = self.scratch_xmm(0);
                let scratch_ymm = self.scratch_ymm(0);
                let qh_xmm = self.scratch_xmm(2);
                let qh_ymm = self.scratch_ymm(2);

                // Bit-position mask: [1,2,4,8,16,32,64,128]
                self.asm.mov(rax, 0x8040201008040201u64).map_err(Self::err)?;
                self.asm.vmovq(scratch_xmm, rax).map_err(Self::err)?;
                self.asm.vpbroadcastq(scratch_ymm, scratch_xmm).map_err(Self::err)?;

                // Load qh byte from saved address → broadcast
                self.asm.movzx(rax, byte_ptr(addr_reg)).map_err(Self::err)?;
                self.asm.vmovd(qh_xmm, eax).map_err(Self::err)?;
                self.asm.vpbroadcastb(qh_ymm, qh_xmm).map_err(Self::err)?;

                // AND with bit mask
                self.asm.vpand(qh_ymm, qh_ymm, scratch_ymm).map_err(Self::err)?;

                // Non-zero → 0xFF per byte (unsigned-aware).
                // vpcmpgtb is signed: 0x80 = -128 > 0 = false — breaks bit 7.
                // Use pcmpeqb + invert instead: detect zero, then XOR with all-ones.
                let ones_ymm = self.scratch_ymm(1);
                self.asm.vpcmpeqb(ones_ymm, ones_ymm, ones_ymm).map_err(Self::err)?;
                self.asm.vpxor(scratch_ymm, scratch_ymm, scratch_ymm).map_err(Self::err)?;
                self.asm.vpcmpeqb(scratch_ymm, qh_ymm, scratch_ymm).map_err(Self::err)?;
                self.asm.vpxor(qh_ymm, ones_ymm, scratch_ymm).map_err(Self::err)?;

                // AND with bit_value as byte (e.g., 16 = 0x10 for INT5)
                let bv = *bit_value as i32;
                self.asm.mov(rax, ((bv as u64) * 0x0101010101010101u64)).map_err(Self::err)?;
                self.asm.vmovq(scratch_xmm, rax).map_err(Self::err)?;
                self.asm.vpbroadcastb(scratch_ymm, scratch_xmm).map_err(Self::err)?;
                self.asm.vpand(qh_ymm, qh_ymm, scratch_ymm).map_err(Self::err)?;

                // Expand 8 bytes → 8 × i32
                let qh_xmm2 = Self::ymm_to_xmm(qh_ymm);
                self.asm.vpmovzxbd(dst_ymm, qh_xmm2).map_err(Self::err)?;

                // Convert to f32
                self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }
        }
    }

    fn lower_quant_biplane_load(
        &mut self,
        dst: VRegId,
        qs_base: VRegId,
        extra_base: VRegId,
        bias: f32,
        mode: &BiPlaneMode,
        width: &SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        match mode {
            BiPlaneMode::Low5 => {                // INT5: load qs (nibbles) + qh (1-bit high plane) → merge → f32
                // For 8 lanes (AVX2): 4 qs bytes + 1 qh byte → 8 f32 values
                let lanes = width.f32_lanes().max(1);
                if lanes != 8 {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt5Load: only 8 lanes supported, got {}", lanes)
                    ));
                }

                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let qs_reg = self.resolve_gpr_read(qs_base, alloc, 2)?;
                let qh_reg = self.resolve_gpr_read(extra_base, alloc, 2)?;

                // ── Step 1: Nibble unpack from qs_base (same as GgufInt4Load) ──
                // Load 4 bytes → vpmovzxbd → AND 0x0F + shift right 4 → interleave → i32
                let dst_xmm = Self::ymm_to_xmm(dst_ymm);
                self.asm.vmovd(dst_xmm, dword_ptr(qs_reg)).map_err(Self::err)?;
                self.asm.vpmovzxbd(dst_ymm, dst_xmm).map_err(Self::err)?;

                // AND with 0x0F for low nibbles
                let scratch_xmm = self.scratch_xmm(0);
                let scratch_ymm_reg = self.scratch_ymm(0);
                self.asm.mov(rax, 0x0F0F0F0Fi64 as u64).map_err(Self::err)?;
                self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                self.asm.vpbroadcastd(scratch_ymm_reg, scratch_xmm).map_err(Self::err)?;
                self.asm.vpand(dst_ymm, dst_ymm, scratch_ymm_reg).map_err(Self::err)?;

                // Shift right by 4 for high nibbles
                let high_ymm = self.scratch_ymm(1);
                self.asm.vmovdqa(high_ymm, dst_ymm).map_err(Self::err)?;
                self.asm.mov(rax, 4u64).map_err(Self::err)?;
                self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                self.asm.vpsrld(dst_ymm, dst_ymm, scratch_xmm).map_err(Self::err)?;

                // Interleave: [low0, high0, low1, high1, ...]
                self.asm.vpunpckldq(dst_ymm, high_ymm, dst_ymm).map_err(Self::err)?;
                // dst_ymm has 8 × i32 nibble values (0-15)

                // ── Step 2: Expand qh bits → 8 × i32 (0 or 16) ──
                // Load 1 qh byte, broadcast, AND with bit-position mask, compare > 0, AND 0x10
                // Build bit-position mask: [1,2,4,8,16,32,64,128, ...] as bytes
                self.asm.mov(rax, 0x8040201008040201u64).map_err(Self::err)?;
                self.asm.vmovq(scratch_xmm, rax).map_err(Self::err)?;
                self.asm.vpbroadcastq(scratch_ymm_reg, scratch_xmm).map_err(Self::err)?;

                // Load qh byte → broadcast — use scratch(2) for qh work
                let qh_xmm = self.scratch_xmm(2);
                let qh_ymm = self.scratch_ymm(2);
                // Load byte via movzx → vmovd
                self.asm.movzx(rax, byte_ptr(qh_reg)).map_err(Self::err)?;
                self.asm.vmovd(qh_xmm, eax).map_err(Self::err)?;
                self.asm.vpbroadcastb(qh_ymm, qh_xmm).map_err(Self::err)?;

                // AND with bit mask
                self.asm.vpand(qh_ymm, qh_ymm, scratch_ymm_reg).map_err(Self::err)?;

                // Compare > 0: use scratch_ymm(0) as zero register (already done with bit mask)
                // Reuse scratch_ymm(0) for zero — xor it first
                self.asm.vpxor(scratch_ymm_reg, scratch_ymm_reg, scratch_ymm_reg).map_err(Self::err)?;
                self.asm.vpcmpgtb(qh_ymm, qh_ymm, scratch_ymm_reg).map_err(Self::err)?;
                // qh_ymm = [0xFF or 0x00 per byte]

                // AND with 0x10
                self.asm.mov(rax, 0x1010101010101010u64).map_err(Self::err)?;
                self.asm.vmovq(scratch_xmm, rax).map_err(Self::err)?;
                self.asm.vpbroadcastb(scratch_ymm_reg, scratch_xmm).map_err(Self::err)?;
                self.asm.vpand(qh_ymm, qh_ymm, scratch_ymm_reg).map_err(Self::err)?;
                // qh_ymm = [16 or 0 per byte] for first 8 bytes

                // Expand bytes to i32: vpmovzxbd — use high_ymm (scratch(1)) for result
                let qh_xmm2 = Self::ymm_to_xmm(qh_ymm);
                self.asm.vpmovzxbd(high_ymm, qh_xmm2).map_err(Self::err)?;
                // high_ymm = 8 × i32 (0 or 16)

                // ── Step 3: Merge nibble + qh → INT5 value ──
                self.asm.vpaddd(dst_ymm, dst_ymm, high_ymm).map_err(Self::err)?;
                // dst_ymm = 8 × i32 (0-31)

                // ── Step 4: Subtract bias, convert to f32 ──
                self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;
                if bias != 0.0 {
                    self.asm.mov(rax, f32::to_bits(bias) as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vbroadcastss(scratch_ymm_reg, scratch_xmm).map_err(Self::err)?;
                    self.asm.vsubps(dst_ymm, dst_ymm, scratch_ymm_reg).map_err(Self::err)?;
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BiPlaneMode::Low6 => {                // INT6: load qs (nibbles) + qh (2-bit high plane) → merge → f32
                // For 8 lanes (AVX2): 4 qs bytes + 2 qh bytes → 8 f32 values
                let lanes = width.f32_lanes().max(1);
                if lanes != 8 {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt6Load: only 8 lanes supported, got {}", lanes)
                    ));
                }

                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let qs_reg = self.resolve_gpr_read(qs_base, alloc, 2)?;
                let qh_reg = self.resolve_gpr_read(extra_base, alloc, 2)?;

                // ── Step 1: Nibble unpack (same as INT5) ──
                let dst_xmm = Self::ymm_to_xmm(dst_ymm);
                self.asm.vmovd(dst_xmm, dword_ptr(qs_reg)).map_err(Self::err)?;
                self.asm.vpmovzxbd(dst_ymm, dst_xmm).map_err(Self::err)?;

                let scratch_xmm = self.scratch_xmm(0);
                let scratch_ymm_reg = self.scratch_ymm(0);
                self.asm.mov(rax, 0x0F0F0F0Fi64 as u64).map_err(Self::err)?;
                self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                self.asm.vpbroadcastd(scratch_ymm_reg, scratch_xmm).map_err(Self::err)?;
                self.asm.vpand(dst_ymm, dst_ymm, scratch_ymm_reg).map_err(Self::err)?;

                let high_ymm = self.scratch_ymm(1);
                self.asm.vmovdqa(high_ymm, dst_ymm).map_err(Self::err)?;
                self.asm.mov(rax, 4u64).map_err(Self::err)?;
                self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                self.asm.vpsrld(dst_ymm, dst_ymm, scratch_xmm).map_err(Self::err)?;
                self.asm.vpunpckldq(dst_ymm, high_ymm, dst_ymm).map_err(Self::err)?;

                // ── Step 2: Expand qh 2-bit values → 8 × i32 (0, 16, 32, 48) ──
                // qh has 4 × 2-bit values per byte, 8 elements = 2 qh bytes
                // Load 2 qh bytes
                // For Q6K: qh[byte0] has 2-bit values for elements 0-3, qh[byte1] for 4-7
                // Element i: ((qh[i/4] >> (2*(i%4))) & 3) << 4

                // Strategy: load 2 bytes, expand each 2-bit field to i32, shift left by 4
                // Load qh word (2 bytes)
                let qh_ymm = self.scratch_ymm(2);
                let qh_xmm = self.scratch_xmm(2);
                self.asm.vmovd(qh_xmm, word_ptr(qh_reg)).map_err(Self::err)?;
                // qh_xmm low 16 bits = [qh_byte0, qh_byte1]

                // Expand 2 bytes → 8 i32 with 2-bit extraction
                // Use vpshufb to spread bytes, then mask and shift
                // Simpler: expand 2 bytes to 8 bytes via vpshufb, then AND+shift per 2-bit field
                // Build shuffle mask: [0,0,0,0, 1,1,1,1] to duplicate each byte 4 times
                self.asm.mov(rax, 0x0101010100000000u64).map_err(Self::err)?;
                self.asm.vmovq(scratch_xmm, rax).map_err(Self::err)?;
                self.asm.vpshufb(qh_xmm, qh_xmm, scratch_xmm).map_err(Self::err)?;
                // qh_xmm = [b0,b0,b0,b0, b1,b1,b1,b1, 0,0,...] as bytes

                // For each byte, extract 2-bit fields: (byte >> (2*i)) & 3
                // AND with 0x03 → gives 2-bit value for first position
                self.asm.mov(rax, 0x0303030303030303u64).map_err(Self::err)?;
                self.asm.vmovq(scratch_xmm, rax).map_err(Self::err)?;
                self.asm.vpand(qh_xmm, qh_xmm, scratch_xmm).map_err(Self::err)?;
                // qh_xmm bytes = [b0&3, b0&3, b0&3, b0&3, b1&3, ...]

                // But this only gives the first 2-bit field. We need per-lane shifts.
                // Use a different approach: build all 4 shifts explicitly.
                // Alternative: use vpsrlw with per-word shift amount
                // Better: use 4 AND+shift sequences:
                // Lane 0: byte & 3
                // Lane 1: (byte >> 2) & 3
                // Lane 2: (byte >> 4) & 3
                // Lane 3: (byte >> 6) & 3

                // Actually, let's use a mask + shift approach with 4 separate masks:
                // Mask [3, 0x0C, 0x30, 0xC0] → shift [0, 2, 4, 6]
                // But vpsrlb doesn't exist. Use vpsrlw (16-bit shift) as approximation:
                // Byte-level shift: AND mask then vpsrlw by [0,2,4,6] (as word shifts)
                // This is imprecise for byte values.

                // Most practical: use 4 separate extraction steps.
                // But that's 4 passes which is slow.

                // Alternative: use a vpshufb LUT approach.
                // The 2-bit value (0-3) maps to (0, 16, 32, 48) when shifted left by 4.
                // We can build a 4-entry LUT and use it.

                // Simplest correct approach: scalar extraction
                // Load 2 qh bytes into GPR, extract 8 × 2-bit values, build i32 vector
                self.asm.movzx(rax, word_ptr(qh_reg)).map_err(Self::err)?;
                // rax = qh_byte1 << 8 | qh_byte0

                // Extract 8 × 2-bit values and build as 8 × i32 packed in memory
                // Use stack space: push 8 i32 values
                let tmp_gpr = r10;
                // Build 8 i32 values:
                // From byte0: v0=(b0>>0)&3, v1=(b0>>2)&3, v2=(b0>>4)&3, v3=(b0>>6)&3
                // From byte1: v4=(b1>>0)&3, v5=(b1>>2)&3, v6=(b1>>4)&3, v7=(b1>>6)&3
                // Shift left by 4: each becomes 0, 16, 32, or 48
                // Use rsp-based temporary storage

                // Actually, let's use a simpler approach with the vector path.
                // Load qh bytes, broadcast each, AND with per-position mask, shift:

                // Build a packed constant for 2-bit extraction via multiplication.
                // trick: for each byte position, the 2-bit field can be extracted by:
                // 1. AND with appropriate mask
                // 2. Shift right
                // But we need per-lane different shifts.
                // Use vpmulhuw with a magic multiplier? Or use vpshufb LUT.

                // OK let me use the scalar approach via stack. It's the most straightforward.
                // Allocate 32 bytes on stack for 8 × i32 values.
                self.asm.sub(rsp, 32).map_err(Self::err)?;

                // Extract 8 × 2-bit values from the 2 qh bytes
                let qh_val = rax;  // already has the 2 bytes
                for i in 0..8 {
                    let byte_shift = (i % 4) * 2;
                    let byte_idx = i / 4;
                    let shift = byte_shift + byte_idx * 8; // total bit shift
                    self.asm.mov(tmp_gpr, qh_val).map_err(Self::err)?;
                    self.asm.shr(tmp_gpr, shift as u32).map_err(Self::err)?;
                    self.asm.and(tmp_gpr, 3i32).map_err(Self::err)?;
                    self.asm.shl(tmp_gpr, 4).map_err(Self::err)?; // shift left by 4 for <<4 merge
                    // Store as i32 at rsp + i*4
                    self.asm.mov(dword_ptr(rsp + i * 4), tmp_gpr).map_err(Self::err)?;
                }

                // Load 8 × i32 from stack → qh_ymm (scratch(2))
                self.asm.vmovdqu(qh_ymm, ymmword_ptr(rsp)).map_err(Self::err)?;

                // Restore stack
                self.asm.add(rsp, 32).map_err(Self::err)?;

                // ── Step 3: Merge nibble + qh → INT6 value ──
                self.asm.vpaddd(dst_ymm, dst_ymm, qh_ymm).map_err(Self::err)?;

                // ── Step 4: Subtract bias, convert to f32 ──
                self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;
                if bias != 0.0 {
                    self.asm.mov(rax, f32::to_bits(bias) as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vbroadcastss(scratch_ymm_reg, scratch_xmm).map_err(Self::err)?;
                    self.asm.vsubps(dst_ymm, dst_ymm, scratch_ymm_reg).map_err(Self::err)?;
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

            BiPlaneMode::Q3Merge => {                // Q3K 3-bit: qs(2-bit) + hmask(1-bit) → merged 3-bit → i32 → f32, subtract bias.
                // For 8 lanes: 2 qs bytes + 1 hmask byte → 8 × i32 → f32.
                let lanes = width.f32_lanes().max(1);
                if lanes != 8 {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt3Load: only 8 lanes supported, got {}", lanes)
                    ));
                }

                let (dst_ymm, dst_spilled) = self.resolve_ymm_or_spill_write(dst, alloc, 2)?;
                let qs_reg = self.resolve_gpr_read(qs_base, alloc, 0)?;
                let hmask_reg = self.resolve_gpr_read(extra_base, alloc, 1)?;

                // Layout: [rsp+0..+31] = 32B temp for element extraction
                //         [rsp+32..+39] = 8B temp for saving qs value
                // NOTE: qs_reg may be RAX (scratch_slot=0), hmask_reg may be R10 (scratch_slot=1).
                // We must not clobber R10 after loading hmask_reg.
                // Save qs value to stack slot at [rsp+32] to avoid register pressure.
                self.asm.sub(rsp, 40).map_err(Self::err)?;

                // Load 2 qs bytes and save to [rsp+32]
                self.asm.movzx(rax, word_ptr(qs_reg)).map_err(Self::err)?;
                self.asm.mov(qword_ptr(rsp + 32), rax).map_err(Self::err)?;

                // Load 1 hmask byte
                let hmask_tmp = r11;
                self.asm.movzx(hmask_tmp, byte_ptr(hmask_reg)).map_err(Self::err)?;

                // Extract 8 elements: qs_2bit | (hmask_bit << 2)
                for i in 0..8 {
                    // Extract qs 2-bit: byte[i/4] bits [(i%4)*2+1 : (i%4)*2]
                    let qs_byte_idx = i / 4;
                    let qs_bit_shift = (i % 4) * 2 + qs_byte_idx * 8;
                    self.asm.mov(rax, qword_ptr(rsp + 32)).map_err(Self::err)?;
                    self.asm.shr(rax, qs_bit_shift as u32).map_err(Self::err)?;
                    self.asm.and(rax, 3i32).map_err(Self::err)?;
                    // rax = qs_2bit (0-3)

                    // Extract hmask bit
                    let hmask_bit = i % 8;
                    // Shift hmask byte right by bit position, AND with 1
                    self.asm.mov(rcx, hmask_tmp).map_err(Self::err)?;
                    self.asm.shr(rcx, hmask_bit as u32).map_err(Self::err)?;
                    self.asm.and(rcx, 1i32).map_err(Self::err)?;
                    // rcx = hmask_bit (0 or 1)

                    // Merge: qs_2bit | (hmask_bit << 2)
                    self.asm.shl(rcx, 2).map_err(Self::err)?;
                    self.asm.or(rax, rcx).map_err(Self::err)?;
                    // rax = 3-bit value (0-7)

                    // Store as i32 at rsp + i*4
                    self.asm.mov(dword_ptr(rsp + i * 4), eax).map_err(Self::err)?;
                }

                // Load 8 × i32 from stack → ymm
                self.asm.vmovdqu(dst_ymm, ymmword_ptr(rsp)).map_err(Self::err)?;

                // Restore stack
                self.asm.add(rsp, 40).map_err(Self::err)?;

                // Convert i32 → f32
                self.asm.vcvtdq2ps(dst_ymm, dst_ymm).map_err(Self::err)?;

                // Subtract bias
                if bias != 0.0 {
                    let scratch_ymm = if dst_ymm != ymm15 { ymm15 } else { ymm14 };
                    let scratch_xmm = Self::ymm_to_xmm(scratch_ymm);
                    self.asm.mov(rax, f32::to_bits(bias) as u64).map_err(Self::err)?;
                    self.asm.vmovd(scratch_xmm, eax).map_err(Self::err)?;
                    self.asm.vbroadcastss(scratch_ymm, scratch_xmm).map_err(Self::err)?;
                    self.asm.vsubps(dst_ymm, dst_ymm, scratch_ymm).map_err(Self::err)?;
                }

                if dst_spilled { self.spill_store_ymm(dst, alloc, 2)?; }
                Ok(())
            }

        }
    }
}

