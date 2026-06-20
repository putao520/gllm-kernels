impl X86Lower {
    pub fn new() -> Self {
        Self::with_avx512(false)
    }

    pub fn with_avx512(use_avx512: bool) -> Self {
        use crate::dispatch::device_profile::DeviceProfile;
        let jit_ctx = crate::compiler::jit_context::JitContext::from_device_profile(&DeviceProfile::detect());
        Self {
            asm: CodeAssembler::new(64).expect("64-bit"),
            use_avx512,
            const_pool: Vec::new(),
            loop_stack: Vec::new(),
            scope_saves: Vec::new(),
            skip_stack: Vec::new(),
            stack_layout: StackLayout::default(),
            amx_tile_dtype: None,
            jit_ctx,
            sym_slot_map: super::plan_lower::SymDimSlotMap::mega_kernel_abi(),
            // 默认 scratch (x86 SysV): rax (eval_offset), r10/r11 (spill load/store)
            scratch_gprs: vec![rax, r10, r11],
            // 默认 vec scratch (与 IsaProfile 一致): ymm15..10 共 6 条
            // 前 3 = 内部 scratch (HReduce/FWHT/broadcast),后 3 = spill scratch (a/b/dst)
            scratch_vec_ids: vec![
                super::isa_profile::PhysVec(15),
                super::isa_profile::PhysVec(14),
                super::isa_profile::PhysVec(13),
                super::isa_profile::PhysVec(12),
                super::isa_profile::PhysVec(11),
                super::isa_profile::PhysVec(10),
            ],
            epilogue_label: None,
            dispatch_labels: HashMap::new(),
            source_map: super::debug_map::JitSourceMap::new(),
            zero_vregs: HashSet::new(),
            stack_arg_vregs: HashMap::new(),
        }
    }

    /// 带 SymDimSlotMap 构造（ARCH-SYMDIM-THREADING §5.2）。
    pub fn with_sym_map(use_avx512: bool, sym_slot_map: super::plan_lower::SymDimSlotMap) -> Self {
        let mut s = Self::with_avx512(use_avx512);
        s.sym_slot_map = sym_slot_map;
        s
    }

    /// 预扫描 VmProgram，收集所有 GprLoadImm { value: 0 } 的目标 VReg。
    /// 在 ISA lowering 循环前调用一次，供 GprBinOp::Add peephole 优化使用。
    pub fn precompute_zero_vregs(&mut self, program: &super::instr::VmProgram) {
        for instr in &program.instrs {
            if let VmInstr::GprLoadImm { dst, value } = instr {
                if *value == 0 {
                    self.zero_vregs.insert(*dst);
                }
            }
        }
    }

    /// 从 IsaProfile 注入 scratch GPR（ARCH-ISA-SCRATCH）。
    /// `scratch_phys` 必须与 IsaProfile.scratch_gprs 一致，否则 RegAllocator 和 Lower 不同步。
    /// scratch[0] = eval_offset 主 scratch，scratch[1..] 用于 spill load/store
    /// (ARCH-REGALLOC-GPR-SPILL 要求 ≥3 以支持二输入 VmInstr 双 spill)。
    pub fn set_scratch_gprs(&mut self, scratch_phys: &[super::isa_profile::PhysGpr]) -> Result<(), CompilerError> {
        if scratch_phys.len() < 3 {
            return Err(CompilerError::CodegenViolation(format!(
                "X86Lower 需要 ≥3 个 scratch GPR (ARCH-REGALLOC-GPR-SPILL)，IsaProfile 提供 {}", scratch_phys.len())));
        }
        self.scratch_gprs = scratch_phys.iter().map(|p| Self::gpr(*p)).collect();
        Ok(())
    }

    /// 从 IsaProfile 注入 scratch vec 寄存器（ARCH-ISA-SCRATCH-VEC）。
    /// 需要 ≥6 条:前 3 为内部 scratch (HReduce/FWHT/broadcast),后 3 为 spill scratch
    /// (a/b/dst 中转,支持一条 VmInstr 中三个 spilled VReg 同时使用)。
    pub fn set_scratch_vec_regs(&mut self, scratch_phys: &[super::isa_profile::PhysVec]) -> Result<(), CompilerError> {
        if scratch_phys.len() < 6 {
            return Err(CompilerError::CodegenViolation(format!(
                "X86Lower 需要 ≥6 个 scratch vec 寄存器 (3 内部 + 3 spill),IsaProfile 提供 {}", scratch_phys.len())));
        }
        self.scratch_vec_ids = scratch_phys.to_vec();
        Ok(())
    }

    /// 返回指定索引的 scratch YMM 寄存器。
    fn scratch_ymm(&self, idx: usize) -> AsmRegisterYmm { Self::ymm(self.scratch_vec_ids[idx]) }
    /// 返回指定索引的 scratch XMM 寄存器。
    fn scratch_xmm(&self, idx: usize) -> AsmRegisterXmm { Self::ymm_to_xmm(Self::ymm(self.scratch_vec_ids[idx])) }
    /// 返回指定索引的 scratch ZMM 寄存器。
    fn scratch_zmm(&self, idx: usize) -> AsmRegisterZmm { Self::zmm(self.scratch_vec_ids[idx]) }

    fn err(e: iced_x86::IcedError) -> CompilerError {
        CompilerError::Internal(e.to_string())
    }

    // ── 物理寄存器映射 ──

    fn gpr(phys: PhysGpr) -> AsmRegister64 {
        match phys.0 {
            0 => rax, 1 => rcx, 2 => rdx, 3 => rbx,
            6 => rsi, 7 => rdi,
            8 => r8, 9 => r9, 10 => r10, 11 => r11,
            12 => r12, 13 => r13, 14 => r14, 15 => r15,
            other => unreachable!("RegAllocator produced invalid PhysGpr({}); x86_64 GPR range [0..15]", other),
        }
    }

    fn gpr32(phys: PhysGpr) -> AsmRegister32 {
        match phys.0 {
            0 => eax, 1 => ecx, 2 => edx, 3 => ebx,
            6 => esi, 7 => edi,
            8 => r8d, 9 => r9d, 10 => r10d, 11 => r11d,
            12 => r12d, 13 => r13d, 14 => r14d, 15 => r15d,
            other => unreachable!("gpr32({}): x86_64 GPR range [0..15]", other),
        }
    }

    /// Convert a 64-bit GPR AsmRegister to its 32-bit counterpart.
    /// Required for DWORD stores (e.g., StoreToken u32 writes).
    #[allow(non_upper_case_globals)]
    fn gpr64_to_32(reg: AsmRegister64) -> AsmRegister32 {
        match reg {
            rax => eax, rcx => ecx, rdx => edx, rbx => ebx,
            rsp => esp, rbp => ebp, rsi => esi, rdi => edi,
            r8 => r8d, r9 => r9d, r10 => r10d, r11 => r11d,
            r12 => r12d, r13 => r13d, r14 => r14d, r15 => r15d,
            other => unreachable!("gpr64_to_32({:?}): not a GPR64", other),
        }
    }

    fn ymm(phys: PhysVec) -> AsmRegisterYmm {
        match phys.0 {
            0 => ymm0, 1 => ymm1, 2 => ymm2, 3 => ymm3,
            4 => ymm4, 5 => ymm5, 6 => ymm6, 7 => ymm7,
            8 => ymm8, 9 => ymm9, 10 => ymm10, 11 => ymm11,
            12 => ymm12, 13 => ymm13, 14 => ymm14, 15 => ymm15,
            other => unreachable!("RegAllocator produced invalid PhysVec({}) for YMM; AVX2 range [0..15]", other),
        }
    }

    fn zmm(phys: PhysVec) -> AsmRegisterZmm {
        match phys.0 {
            0 => zmm0, 1 => zmm1, 2 => zmm2, 3 => zmm3,
            4 => zmm4, 5 => zmm5, 6 => zmm6, 7 => zmm7,
            8 => zmm8, 9 => zmm9, 10 => zmm10, 11 => zmm11,
            12 => zmm12, 13 => zmm13, 14 => zmm14, 15 => zmm15,
            16 => zmm16, 17 => zmm17, 18 => zmm18, 19 => zmm19,
            20 => zmm20, 21 => zmm21, 22 => zmm22, 23 => zmm23,
            24 => zmm24, 25 => zmm25, 26 => zmm26, 27 => zmm27,
            28 => zmm28, 29 => zmm29, 30 => zmm30, 31 => zmm31,
            other => unreachable!("RegAllocator produced invalid PhysVec({}) for ZMM; AVX-512 range [0..31]", other),
        }
    }

    /// 非 spill 路径: 仅当 VReg 确实分到物理 GPR 时返回，否则报错。
    /// 保留给那些 "显然不能 spill" 的上下文 (如 ABI 入参寄存器 copy)。
    /// 多数调用应使用 resolve_gpr_read / resolve_gpr_write / commit_gpr_write。
    fn resolve_gpr(&self, vreg: VRegId, alloc: &RegAllocation) -> Result<AsmRegister64, CompilerError> {
        match alloc.get(vreg) {
            Some(super::isa_profile::PhysReg::Gpr(g)) => Ok(Self::gpr(g)),
            Some(super::isa_profile::PhysReg::Spilled(_)) => Err(CompilerError::CodegenViolation(
                format!("v{} was spilled but caller used non-spill resolve_gpr", vreg.0),
            )),
            _ => Err(CompilerError::CodegenViolation(format!("v{} not allocated to GPR", vreg.0))),
        }
    }

    /// 读取 spilled VReg 到指定 scratch 槽；如果已分到 GPR 则直接返回。
    /// `scratch_slot` 是 self.scratch_gprs 的索引 (0 保留给 eval_offset_to_rax)，
    /// 通常使用 slot 1, 2 (r10, r11)。同一条 VmInstr 多个 spilled 输入时，
    /// 调用方需选择互不冲突的 slot。
    fn resolve_gpr_read(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        scratch_slot: usize,
    ) -> Result<AsmRegister64, CompilerError> {
        match alloc.get(vreg) {
            Some(super::isa_profile::PhysReg::Gpr(g)) => Ok(Self::gpr(g)),
            Some(super::isa_profile::PhysReg::Spilled(slot_id)) => {
                let scratch = *self.scratch_gprs.get(scratch_slot)
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "resolve_gpr_read: scratch_slot {} out of range (scratch_gprs.len={})",
                        scratch_slot, self.scratch_gprs.len(),
                    )))?;
                // ARCH-SPILL-SAFE-ISA: Disabled. The root cause fix is in
                // ScopedSpillAllocator — it now assigns unique offsets to each
                // spill slot, preventing two VRegs from sharing the same stack
                // memory location. The SpillSafeRecipe approach was incorrect
                // because VRegs can be redefined (non-SSA), causing stale
                // recipes to load wrong values.
                // if let Some(recipe) = self.stack_arg_vregs.get(&vreg) { ... }
                let spill = alloc.spills.get(slot_id as usize)
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "resolve_gpr_read: spill slot {} missing for v{}", slot_id, vreg.0,
                    )))?;
                // spill_offset 来自 emit_prologue 里的栈布局：[rbp - frame_gpr_spill_base - offset - 8]
                let rbp_off = self.gpr_spill_rbp_offset(spill.offset);
                self.asm.mov(scratch, qword_ptr(rbp + rbp_off)).map_err(Self::err)?;
                Ok(scratch)
            }
            _ => Err(CompilerError::CodegenViolation(format!("v{} not allocated to GPR", vreg.0))),
        }
    }

    /// Resolve an OffsetExpr to a physical GPR by computing the offset value.
    fn resolve_offset_to_gpr(
        &mut self,
        off: &OffsetExpr,
        alloc: &RegAllocation,
        scratch_slot: usize,
    ) -> Result<AsmRegister64, CompilerError> {
        let scratch = *self.scratch_gprs.get(scratch_slot)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("resolve_offset_to_gpr: scratch_slot {} out of range", scratch_slot),
            ))?;
        match off {
            OffsetExpr::Const(c) => {
                self.asm.mov(scratch, *c as u64).map_err(Self::err)?;
                Ok(scratch)
            }
            OffsetExpr::ScalarVReg(v) => {
                let val = self.resolve_gpr_read(*v, alloc, scratch_slot)?;
                if val != scratch { self.asm.mov(scratch, val).map_err(Self::err)?; }
                Ok(scratch)
            }
            OffsetExpr::Add(a, b) => {
                let a_reg = self.resolve_offset_to_gpr(a, alloc, scratch_slot)?;
                // For the second operand, use next scratch slot
                let b_slot = scratch_slot + 1;
                let b_reg = self.resolve_offset_to_gpr(b, alloc, b_slot)?;
                self.asm.add(scratch, b_reg).map_err(Self::err)?;
                Ok(scratch)
            }
            OffsetExpr::Mul(inner, scale) => {
                let inner_reg = self.resolve_offset_to_gpr(inner, alloc, scratch_slot)?;
                if inner_reg != scratch { self.asm.mov(scratch, inner_reg).map_err(Self::err)?; }
                if *scale != 1 {
                    self.asm.imul_3(scratch, scratch, *scale as i32).map_err(Self::err)?;
                }
                Ok(scratch)
            }
            OffsetExpr::LoopOffset(v) => {
                let val = self.resolve_gpr_read(*v, alloc, scratch_slot)?;
                if val != scratch { self.asm.mov(scratch, val).map_err(Self::err)?; }
                Ok(scratch)
            }
        }
    }

    /// 自动分配 scratch slot 并读取 VReg。
    ///
    /// 维护内部 scratch slot 使用栈：每个 slot_alloc() 获取一个空闲 slot，
    /// slot_free() 释放。消除了手写 scratch_slot 参数导致的 aliasing bug。
    ///
    /// 如果 VReg 已在物理 GPR 中，不消耗 scratch slot。
    fn gpr_read_auto(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        slot_state: &mut ScratchSlotState,
    ) -> Result<(AsmRegister64, Option<usize>), CompilerError> {
        match alloc.get(vreg) {
            Some(super::isa_profile::PhysReg::Gpr(g)) => Ok((Self::gpr(g), None)),
            Some(super::isa_profile::PhysReg::Spilled(slot_id)) => {
                let slot = slot_state.alloc().ok_or_else(|| {
                    CompilerError::CodegenViolation(format!(
                        "gpr_read_auto: no free scratch slot for v{}", vreg.0,
                    ))
                })?;
                let scratch = *self.scratch_gprs.get(slot)
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "gpr_read_auto: slot {} out of range", slot,
                    )))?;
                // ARCH-SPILL-SAFE-ISA: Disabled (see resolve_gpr_read for explanation).
                // Root cause fix is in ScopedSpillAllocator — unique offsets per VReg.
                let spill = alloc.spills.get(slot_id as usize)
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "gpr_read_auto: spill slot {} missing for v{}", slot_id, vreg.0,
                    )))?;
                let rbp_off = self.gpr_spill_rbp_offset(spill.offset);
                self.asm.mov(scratch, qword_ptr(rbp + rbp_off)).map_err(Self::err)?;
                Ok((scratch, Some(slot)))
            }
            _ => Err(CompilerError::CodegenViolation(format!("v{} not allocated to GPR", vreg.0))),
        }
    }

    /// 返回 spilled VReg 的写目标寄存器（物理 GPR 或 scratch）。
    /// **调用方必须在写入后调用 commit_gpr_write(vreg, alloc, scratch_slot)**
    /// 把 scratch 的值 store 回栈，否则 spill 值丢失。
    fn resolve_gpr_write(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        scratch_slot: usize,
    ) -> Result<AsmRegister64, CompilerError> {
        match alloc.get(vreg) {
            Some(super::isa_profile::PhysReg::Gpr(g)) => Ok(Self::gpr(g)),
            Some(super::isa_profile::PhysReg::Spilled(_)) => {
                let scratch = *self.scratch_gprs.get(scratch_slot)
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "resolve_gpr_write: scratch_slot {} out of range", scratch_slot,
                    )))?;
                Ok(scratch)
            }
            _ => Err(CompilerError::CodegenViolation(format!("v{} not allocated to GPR", vreg.0))),
        }
    }

    /// 若 VReg 被 spilled，把 scratch_slot 的值 store 回栈槽。
    /// 对分到物理 GPR 的 VReg 此调用为 no-op，可无条件调用。
    fn commit_gpr_write(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        scratch_slot: usize,
    ) -> Result<(), CompilerError> {
        if let Some(super::isa_profile::PhysReg::Spilled(slot_id)) = alloc.get(vreg) {
            let scratch = *self.scratch_gprs.get(scratch_slot)
                .ok_or_else(|| CompilerError::CodegenViolation(format!(
                    "commit_gpr_write: scratch_slot {} out of range", scratch_slot,
                )))?;
            let spill = alloc.spills.get(slot_id as usize)
                .ok_or_else(|| CompilerError::CodegenViolation(format!(
                    "commit_gpr_write: spill slot {} missing for v{}", slot_id, vreg.0,
                )))?;
            let rbp_off = self.gpr_spill_rbp_offset(spill.offset);
            self.asm.mov(qword_ptr(rbp + rbp_off), scratch).map_err(Self::err)?;
        }
        Ok(())
    }

    /// GPR spill slot → [rbp + rbp_off] 转换（从 StackLayout 查询）。
    fn gpr_spill_rbp_offset(&self, spill_off: usize) -> i32 {
        self.stack_layout.spill_rbp_offset(spill_off, 8)
    }

    /// x86-64 System V ABI 入参寄存器映射 (SysV 规范，非随意硬编码)。
    /// 来源: System V AMD64 ABI Draft §3.2.3 表 3.
    fn sysv_arg_reg(idx: u8) -> Result<AsmRegister64, CompilerError> {
        match idx {
            0 => Ok(rdi), 1 => Ok(rsi), 2 => Ok(rdx),
            3 => Ok(rcx), 4 => Ok(r8), 5 => Ok(r9),
            _ => Err(CompilerError::CodegenViolation(
                format!("AbiArg({}) out of SysV register range [0..5]", idx))),
        }
    }

    /// ARCH-ABI-SAVE-AREA: AbiArg(idx) 对应 push 后的栈槽 rbp 相对偏移（从 StackLayout 查询）。
    ///
    /// emit_prologue 在 push 每个 ABI reg 时记录了精确偏移，无需公式推导。
    fn abi_arg_rbp_offset(&self, idx: u8) -> i32 {
        self.stack_layout.abi_arg_rbp_offset(idx)
            .unwrap_or_else(|| panic!("AbiArg({}) slot not recorded — emit_prologue not called?", idx))
    }

    /// 加载指定的 PtrExpr 到 dst_reg（PtrExpr 必须已解析，不含 NamedArg）。
    fn emit_load_ptr_from_resolved(
        &mut self,
        dst_reg: AsmRegister64,
        src: &PtrExpr,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        match src {
            PtrExpr::AbiArg(idx) => {
                // ARCH-ABI-SAVE-AREA (task #18 root cause fix):
                // 旧实现: `mov dst_reg, sysv_arg_reg(idx)` — 当多个 LoadPtr {AbiArg} 指令
                // 的 dst 物理寄存器与某个 SysV 输入寄存器形成循环 (e.g. v0→rsi←AbiArg(0)=rdi,
                // v1→rdi←AbiArg(1)=rsi),序列化的 mov 会先 `mov rsi,rdi` 把 rsi 上的入参 1
                // 覆盖掉,接着 `mov rdi,rsi` 读到的已经是入参 0 的副本而非原始入参 1,
                // 导致 weight_ptr 退化为 input_ptr 副本,JIT 后续按 weight 偏移寻址读到
                // 越界的堆/栈垃圾,产生跨调用不确定输出 (debug_embed_determinism)。
                //
                // 修复: 在 prologue 中将 6 个 SysV 入参寄存器一次性 push 到栈帧的固定槽位
                // (abi_save_area), 此处所有 AbiArg 读取一律从 rbp-相对槽 mov 加载,彻底
                // 消除循环依赖。代价: 6 条 push + 每个 LoadPtr 多一条 mov mem→reg
                // (相对原 mov reg→reg, 仅多 ~3-4 cycle, 不影响推理热路径性能)。
                let src_off = self.abi_arg_rbp_offset(*idx);
                self.asm.mov(dst_reg, qword_ptr(rbp + src_off)).map_err(Self::err)?;
                let _ = Self::sysv_arg_reg(*idx)?; // 保留 idx 范围检查
            }
            PtrExpr::StackArg(off) => {
                self.asm.mov(dst_reg, qword_ptr(rbp + *off)).map_err(Self::err)?;
            }
            PtrExpr::VRegPlusConst(base, off) => {
                let base_reg = self.resolve_gpr_read(*base, alloc, 1)?;
                if *off <= i32::MAX as usize {
                    self.asm.lea(dst_reg, qword_ptr(base_reg + *off as i32)).map_err(Self::err)?;
                } else {
    // Offset exceeds 32-bit LEA displacement: copy base to dst, add 64-bit imm
                    if dst_reg != base_reg {
                        self.asm.mov(dst_reg, base_reg).map_err(Self::err)?;
                    }
                    // iced-x86 add reg64, imm64 is not directly available;
                    // use mov scratch, imm64 + add reg, scratch.
                    // Pick a scratch that is NOT dst_reg to avoid clobber.
                    let scratch = self.scratch_gprs.iter()
                        .find(|&&s| s != dst_reg)
                        .copied()
                        .unwrap_or(self.scratch_gprs[0]);
                    self.asm.mov(scratch, *off as u64).map_err(Self::err)?;
                    self.asm.add(dst_reg, scratch).map_err(Self::err)?;
                }
            }
            PtrExpr::VRegPlusVReg(base, offset) => {
                let base_reg = self.resolve_gpr_read(*base, alloc, 1)?;
                let off_reg = self.resolve_gpr_read(*offset, alloc, 2)?;
                self.asm.lea(dst_reg, qword_ptr(base_reg + off_reg)).map_err(Self::err)?;
            }
            PtrExpr::VRegPlusOff(base, off_expr) => {
                let base_reg = self.resolve_gpr_read(*base, alloc, 1)?;
                let off_val = self.resolve_offset_to_gpr(off_expr, alloc, 2)?;
                self.asm.lea(dst_reg, qword_ptr(base_reg + off_val)).map_err(Self::err)?;
            }
            PtrExpr::NamedArg(name) => {
                return Err(CompilerError::CodegenViolation(format!(
                    "emit_load_ptr_from_resolved: NamedArg('{}') 未解析就到达 ISA Lower", name)));
            }
            PtrExpr::SharedMem => {
                // ARCH-GPU-SHARED-SCRATCH: CPU x86 无片上 shared memory,
                // CPU scratchpad 必须经 StackArg 读取堆指针。
                return Err(CompilerError::CodegenViolation(
                    "emit_load_ptr_from_resolved: SharedMem 仅用于 GPU,CPU 不支持".into()));
            }
            PtrExpr::AbsAddr(addr) => {
                // ARCH-SG-QTAP: 将 64-bit 主机虚拟地址作为立即数加载 (mov rd, imm64).
                // iced-x86 的 mov r64, imm64 被编码为 REX.W + B8+rd + imm64 (10 字节)。
                self.asm.mov(dst_reg, *addr).map_err(Self::err)?;
            }
        }
        Ok(())
    }

    /// 查找 spilled VReg 在栈帧中的偏移 (相对 rbp),返回 vmovups/movq 用的低地址。
    /// 从 StackLayout 查询，消除手写公式。
    fn spill_offset(&self, vreg: VRegId, alloc: &RegAllocation) -> Option<i32> {
        for spill in &alloc.spills {
            if spill.vreg == vreg {
                return Some(self.stack_layout.spill_rbp_offset(spill.offset, spill.size));
            }
        }
        None
    }

    /// Resolve Vec VReg：物理寄存器 → 直接返回，spilled → 从栈加载到指定 spill scratch 后返回。
    ///
    /// `spill_slot` 是 `scratch_vec_ids` 中专用 spill scratch 的索引 (0..=2,对应底层
    /// scratch_vec_ids[3..=5] = ymm12/11/10)。一条 VmInstr 中多个 spilled 输入需选用
    /// 互不冲突的 slot:
    /// - 二元 op (VecBinOp): a→0, b→1, dst→2
    /// - Fma: acc→2, a→0, b→1, dst→2 (复用 acc slot,因为 Fma 在 acc 上累加得到 dst)
    /// - 一元 op (VecUnaryOp/Transcendental/HReduce): src→0, dst→2
    ///
    /// 返回 (ymm_reg, is_spilled)。调用者在写入 spilled dst 后须调用 spill_store_ymm 写回。
    /// 使用 vmovups (非 vmovaps) 避免 spill 区 32B 对齐假设。
    fn resolve_ymm_or_spill(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        spill_slot: usize,
    ) -> Result<(AsmRegisterYmm, bool), CompilerError> {
        if let Some(phys) = alloc.get_vec(vreg) {
            return Ok((Self::ymm(phys), false));
        }
        if let Some(off) = self.spill_offset(vreg, alloc) {
            let scratch = self.spill_scratch_ymm(spill_slot)?;
            self.asm.vmovups(scratch, ymmword_ptr(rbp + off)).map_err(Self::err)?;
            Ok((scratch, true))
        } else {
            Err(CompilerError::CodegenViolation(format!("v{} not allocated and not spilled", vreg.0)))
        }
    }

    /// Resolve Vec VReg 作为**只写** dst:物理寄存器 → 直接返回,spilled → 返回指定 spill
    /// scratch 寄存器(不发 vmovups load,因为 dst 旧值不被读)。调用者在写入完成后须调用
    /// spill_store_ymm 把 scratch 写回栈 slot。
    fn resolve_ymm_or_spill_write(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        spill_slot: usize,
    ) -> Result<(AsmRegisterYmm, bool), CompilerError> {
        if let Some(phys) = alloc.get_vec(vreg) {
            return Ok((Self::ymm(phys), false));
        }
        if self.spill_offset(vreg, alloc).is_some() {
            let scratch = self.spill_scratch_ymm(spill_slot)?;
            Ok((scratch, true))
        } else {
            Err(CompilerError::CodegenViolation(format!("v{} not allocated and not spilled", vreg.0)))
        }
    }

    /// 将 spill scratch 寄存器写回栈 slot。`spill_slot` 必须与
    /// resolve_ymm_or_spill_write 时用的 slot 一致。
    fn spill_store_ymm(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        spill_slot: usize,
    ) -> Result<(), CompilerError> {
        if let Some(off) = self.spill_offset(vreg, alloc) {
            let scratch = self.spill_scratch_ymm(spill_slot)?;
            self.asm.vmovups(ymmword_ptr(rbp + off), scratch).map_err(Self::err)?;
        }
        Ok(())
    }

    /// 取专用 spill scratch ymm (slot ∈ 0..=2 → scratch_vec_ids[3+slot])。
    fn spill_scratch_ymm(&self, slot: usize) -> Result<AsmRegisterYmm, CompilerError> {
        let idx = 3usize.checked_add(slot).ok_or_else(|| {
            CompilerError::CodegenViolation(format!("spill_scratch_ymm slot overflow: {}", slot))
        })?;
        let phys = self.scratch_vec_ids.get(idx).ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "spill_scratch_ymm: slot {} (scratch_vec_ids[{}]) 不可用,共 {} 条 scratch",
                slot, idx, self.scratch_vec_ids.len(),
            ))
        })?;
        Ok(Self::ymm(*phys))
    }

    /// AVX-512 spill scratch zmm 版本。
    fn spill_scratch_zmm(&self, slot: usize) -> Result<AsmRegisterZmm, CompilerError> {
        let idx = 3usize.checked_add(slot).ok_or_else(|| {
            CompilerError::CodegenViolation(format!("spill_scratch_zmm slot overflow: {}", slot))
        })?;
        let phys = self.scratch_vec_ids.get(idx).ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "spill_scratch_zmm: slot {} (scratch_vec_ids[{}]) 不可用,共 {} 条 scratch",
                slot, idx, self.scratch_vec_ids.len(),
            ))
        })?;
        Ok(Self::zmm(*phys))
    }

    /// AVX-512 zmm 版 resolve_ymm_or_spill。spill 区按 ymm (32B) 对齐,zmm load
    /// 必须用 vmovups (非 vmovaps) 因为 zmm 需要 64B 对齐而 spill slot 仅 32B 对齐。
    /// **WARNING**: zmm spill 槽大小由 RegAllocator 设置 (max(width.bytes(), 32) = 64),
    /// 但 64B 对齐不保证 → 必须 vmovups。
    fn resolve_zmm_or_spill(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        spill_slot: usize,
    ) -> Result<(AsmRegisterZmm, bool), CompilerError> {
        if let Some(phys) = alloc.get_vec(vreg) {
            return Ok((Self::zmm(phys), false));
        }
        if let Some(off) = self.spill_offset(vreg, alloc) {
            let scratch = self.spill_scratch_zmm(spill_slot)?;
            self.asm.vmovups(scratch, zmmword_ptr(rbp + off)).map_err(Self::err)?;
            Ok((scratch, true))
        } else {
            Err(CompilerError::CodegenViolation(format!("v{} not allocated and not spilled", vreg.0)))
        }
    }

    /// AVX-512 zmm 版 resolve_ymm_or_spill_write。
    fn resolve_zmm_or_spill_write(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        spill_slot: usize,
    ) -> Result<(AsmRegisterZmm, bool), CompilerError> {
        if let Some(phys) = alloc.get_vec(vreg) {
            return Ok((Self::zmm(phys), false));
        }
        if self.spill_offset(vreg, alloc).is_some() {
            let scratch = self.spill_scratch_zmm(spill_slot)?;
            Ok((scratch, true))
        } else {
            Err(CompilerError::CodegenViolation(format!("v{} not allocated and not spilled", vreg.0)))
        }
    }

    /// 将 spill scratch zmm 寄存器写回栈 slot。
    fn spill_store_zmm(
        &mut self,
        vreg: VRegId,
        alloc: &RegAllocation,
        spill_slot: usize,
    ) -> Result<(), CompilerError> {
        if let Some(off) = self.spill_offset(vreg, alloc) {
            let scratch = self.spill_scratch_zmm(spill_slot)?;
            self.asm.vmovups(zmmword_ptr(rbp + off), scratch).map_err(Self::err)?;
        }
        Ok(())
    }

    /// 递归求值任意深度的 OffsetExpr → scratch[0] (字节偏移)。
    ///
    /// 支持 Const / LoopOffset / Mul(LoopOffset, scale) / Add(a, b) / ScalarVReg 的任意嵌套。
    /// ARCH-ISA-SCRATCH: scratch_gprs 在 IsaProfile.gpr_regs 中已排除，RegAllocator 不分配；
    /// 因此 mov scratch[1], scratch[0] 安全，不会破坏任何活跃 VReg。
    ///
    /// 嵌套深度处理：
    /// - 非嵌套 Add(simple, simple): 仅使用 scratch[0] + scratch[1]
    /// - 嵌套 Add(Add, _): 当前层用 push/pop 栈保护 scratch[1]（因内层递归也要用）
    ///   栈只在函数帧内临时使用，不破坏栈对齐（push/pop 必成对）
    fn eval_offset_to_rax(&mut self, offset: &OffsetExpr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        let s0 = self.scratch_gprs[0];
        let s1 = self.scratch_gprs[1];
        match offset {
            OffsetExpr::Const(c) => {
                self.asm.mov(s0, *c as u64).map_err(Self::err)?;
            }
            OffsetExpr::LoopOffset(ov) => {
                match alloc.get(*ov) {
                    Some(super::isa_profile::PhysReg::Gpr(g)) => {
                        let o = Self::gpr(g);
                        if o != s0 {
                            self.asm.mov(s0, o).map_err(Self::err)?;
                        }
                    }
                    Some(super::isa_profile::PhysReg::Spilled(slot_id)) => {
                        let slot = &alloc.spills[slot_id as usize];
                        let rbp_off = self.gpr_spill_rbp_offset(slot.offset);
                        self.asm.mov(s0, qword_ptr(rbp + (rbp_off as i64))).map_err(Self::err)?;
                    }
                    _ => {
                        return Err(CompilerError::CodegenViolation(
                            format!("eval_offset: LoopOffset v{} not allocated", ov.0),
                        ));
                    }
                }
            }
            OffsetExpr::Mul(inner, scale) => {
                self.eval_offset_to_rax(inner, alloc)?;
                self.asm.imul_3(s0, s0, *scale as i32).map_err(Self::err)?;
            }
            OffsetExpr::Add(a, b) => {
                // a → s0 → 保存, b → s0, s0 += 保存值
                // b 嵌套 Add 时内层会再用 s1，故用栈保护
                let need_stack_save = matches!(b.as_ref(), OffsetExpr::Add(..));
                self.eval_offset_to_rax(a, alloc)?;
                if need_stack_save {
                    self.asm.push(s0).map_err(Self::err)?;
                    self.eval_offset_to_rax(b, alloc)?;
                    self.asm.pop(s1).map_err(Self::err)?;
                    self.asm.add(s0, s1).map_err(Self::err)?;
                } else {
                    self.asm.mov(s1, s0).map_err(Self::err)?;
                    self.eval_offset_to_rax(b, alloc)?;
                    self.asm.add(s0, s1).map_err(Self::err)?;
                }
            }
            OffsetExpr::ScalarVReg(sv) => {
                if let Some(g) = alloc.get_gpr(*sv).map(Self::gpr) {
                    if g != s0 {
                        self.asm.mov(s0, g).map_err(Self::err)?;
                    }
                } else if let Some(off) = self.spill_offset(*sv, alloc) {
                    self.asm.mov(s0, qword_ptr(rbp + off)).map_err(Self::err)?;
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("eval_offset: ScalarVReg v{} not allocated to GPR or spill", sv.0),
                    ));
                }
            }
        }
        Ok(())
    }

    fn const_f32(&mut self, val: f32) -> CodeLabel {
        // 复用已有常量
        for (entry, label) in &self.const_pool {
            if entry[0].to_bits() == val.to_bits() {
                return *label;
            }
        }
        let label = self.asm.create_label();
        self.const_pool.push(([val; 8], label));
        label
    }

    // ── Prologue / Epilogue ──

    pub fn emit_prologue(&mut self, frame: &StackFrame, alloc: &RegAllocation) -> Result<(), CompilerError> {
        // 追踪 rbp 偏移：push rbp 占 [rbp-8]，rsp 每次减 8
        self.asm.push(rbp).map_err(Self::err)?;
        self.asm.mov(rbp, rsp).map_err(Self::err)?;
        // push rbp + mov rbp,rsp sets rbp = rsp. After this, [rbp] = old rbp.
        // Each subsequent push decrements rsp by 8 and stores at [rsp] = [rbp - cur_off - 8].
        // So cur_off starts at 0; after first push, cur_off = 8, slot = [rbp-8].
        let mut cur_off: usize = 0;

        // 初始化 StackLayout
        let mut layout = StackLayout::default();
        layout.frame_pointer_off = 8;

        // Push callee-saved 并记录到 StackLayout
        for &reg in &alloc.callee_saved_used {
            self.asm.push(Self::gpr(reg)).map_err(Self::err)?;
            cur_off += 8;
            layout.callee_save_slots.push((reg, -(cur_off as i32)));
        }

        // ARCH-ABI-SAVE-AREA (task #18 root cause fix):
        // 在 callee-save 之后立即按 SysV 入参顺序 push 6 个寄存器 (rdi, rsi, rdx,
        // rcx, r8, r9 = AbiArg 0..5)。所有后续 LoadPtr { src: AbiArg(i) } 从这些
        // 固定栈槽读取,彻底消除"AbiArg → 物理 GPR 序列拷贝"的循环依赖
        // (旧 bug: v0→rsi←AbiArg(0)=rdi 接 v1→rdi←AbiArg(1)=rsi 串行 mov,
        // 第二条 mov 读到已被第一条 clobber 的 rsi,导致 weight_ptr 退化为 Q ptr,
        // r9 = lea [rdi+0x300000] 越界,跨调用读到不同堆/栈垃圾 → 非确定输出)。
        //
        // 栈槽布局 (push 顺序 = idx 升序,对应 rbp-relative 由近到远):
        //   [rbp - callee_save_area - 8]  = AbiArg(0) = rdi (input)
        //   [rbp - callee_save_area - 16] = AbiArg(1) = rsi (weights)
        //   [rbp - callee_save_area - 24] = AbiArg(2) = rdx (kv_cache)
        //   [rbp - callee_save_area - 32] = AbiArg(3) = rcx (positions)
        //   [rbp - callee_save_area - 40] = AbiArg(4) = r8  (seq_lens)
        //   [rbp - callee_save_area - 48] = AbiArg(5) = r9  (batch_size)
        let abi_regs: [(AsmRegister64, usize); 6] = [
            (rdi, 0), (rsi, 1), (rdx, 2), (rcx, 3), (r8, 4), (r9, 5),
        ];
        for &(reg, idx) in &abi_regs {
            self.asm.push(reg).map_err(Self::err)?;
            cur_off += 8;
            layout.abi_arg_slots[idx] = Some(-(cur_off as i32));
        }

        // 记录 spill 区基偏移
        layout.spill_base_off = -(cur_off as i32);

        // ARCH-MXCSR-SAVE: JIT code may modify MXCSR (e.g. DAZ/FTZ flags).
        // Always save/restore MXCSR. For red-zone frames, we must allocate
        // extra stack space because the red zone is used by spill slots.
        // Use [rsp] after sub rsp for MXCSR in all cases.
        let mxcsr_extra = MXCSR_SLOT_BYTES;
        let alloc_bytes = frame.total_size as i32 + mxcsr_extra;
        layout.rsp_sub_bytes = alloc_bytes;
        layout.mxcsr_rsp_offset = 0;

        self.asm.sub(rsp, alloc_bytes).map_err(Self::err)?;
        self.asm.stmxcsr(dword_ptr(rsp)).map_err(Self::err)?;

        // ARCH-SPILL-ZERO-INIT (task #18 root cause fix):
        // Zero-initialize each spill slot to prevent use-of-uninitialized stack memory.
        // 即使 RegAllocator 在某条 VmInstr 的 spill VReg 上有 use-before-write bug
        // (e.g. MHA register-only 路径中 hd_vecs > 4 时某些累加器路径),也由 prologue
        // 兜底保证 spill 槽起始为 0,消除"首次推理 vs 后续推理"非确定性 (e1 ≠ e2 = e3
        // 的诊断现象,见 debug_embed_determinism)。
        //
        // 寻址用 rbp-relative，使用 StackLayout.spill_rbp_offset() 消除手写公式。
        //
        // 寄存器选择: rax 在此处不存活 (lower body 还未读 ABI 入参 rdi/rsi/...),
        // xor rax, rax 安全。
        if !alloc.spills.is_empty() && !frame.uses_red_zone {
            self.asm.xor(rax, rax).map_err(Self::err)?;
            for spill in &alloc.spills {
                // spill slot 的低地址 = spill_rbp_offset(offset, size)。
                // 逐 qword (8B) 清零，从低地址到高地址。
                let base_off = layout.spill_rbp_offset(spill.offset, spill.size);
                let qwords = spill.size / 8;
                for q in 0..qwords {
                    let rbp_off = base_off + (q * 8) as i32;
                    self.asm.mov(qword_ptr(rbp + rbp_off), rax).map_err(Self::err)?;
                }
            }
        }

        // 将构建好的 StackLayout 存入 self
        self.stack_layout = layout;

        // ARCH-VM-STATE-TRACKING: 不在 prologue 中硬编码保存 rcx (positions)。
        // 如果 lower 函数需要 positions，通过 VmInstr::LoadPtr + vm_state.arg_ptr_expr("positions")
        // 在使用时加载。rcx 是 caller-saved，RegAllocator 管理其生命周期。

        Ok(())
    }

    pub fn emit_epilogue(&mut self, _frame: &StackFrame, alloc: &RegAllocation) -> Result<(), CompilerError> {
        // 设置 epilogue label — BreakLoop 的 JMP 目标
        if let Some(mut label) = self.epilogue_label.take() {
            self.asm.set_label(&mut label).map_err(Self::err)?;
        }

        // ARCH-MXCSR-RESTORE: restore caller's MXCSR before returning.
        // 从 StackLayout 读取 rsp_sub_bytes，消除 magic number 重复。
        self.asm.ldmxcsr(dword_ptr(rsp)).map_err(Self::err)?;
        self.asm.add(rsp, self.stack_layout.rsp_sub_bytes).map_err(Self::err)?;

        // ABI save area: 反向 pop 释放 6 个 SysV 入参栈槽。
        // pop 顺序与 prologue push 顺序相反（rdi→r9，所以 pop r9→rdi）。
        // 从 StackLayout.abi_arg_slots 推导，不再硬编码。
        let abi_reg_order: [(AsmRegister64, usize); 6] = [
            (rdi, 0), (rsi, 1), (rdx, 2), (rcx, 3), (r8, 4), (r9, 5),
        ];
        for &(reg, _idx) in abi_reg_order.iter().rev() {
            self.asm.pop(reg).map_err(Self::err)?;
        }

        // Pop callee-saved (reverse order) — 从 StackLayout 读取 push 顺序
        for &(reg, _off) in self.stack_layout.callee_save_slots.iter().rev() {
            self.asm.pop(Self::gpr(reg)).map_err(Self::err)?;
        }

        self.asm.pop(rbp).map_err(Self::err)?;
        self.asm.ret().map_err(Self::err)?;
        Ok(())
    }

    // ── 指令翻译 ──
}

