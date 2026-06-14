
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6 VmProgram — 编译时状态追踪序列
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// VReg 按 Kind 分类统计（用于 GPU emit_prologue 动态声明）。
///
/// ARCH-GPU-REG-NAMESPACE: VRegId 全局递增，但 PTX 命名空间独立。
/// `vec_max_id` = Vec 类型 VReg 的最大 VRegId.0（不是数量）。
/// 声明 `.reg .f32 %f<vec_max_id+1>;` 覆盖 `%f0..%f{vec_max_id}`。
#[derive(Debug, Clone, Copy, Default)]
pub struct VRegKindCounts {
    /// Ptr/Scalar/Counter/ByteOffset 的最大 VRegId.0 (共享 %r/%rd 命名空间)
    pub gpr_max_id: Option<u32>,
    /// Vec 的最大 VRegId.0 (%f 命名空间)
    pub vec_max_id: Option<u32>,
    /// Mask 的最大 VRegId.0 (%p 命名空间)
    pub mask_max_id: Option<u32>,
    /// Tile 的最大 VRegId.0 (%t 命名空间)
    pub tile_max_id: Option<u32>,
}

impl VRegKindCounts {
    /// PTX %r/%rd 声明数量 = max_id + 1（覆盖 0..=max_id）。
    pub fn gpr_like(&self) -> u32 {
        self.gpr_max_id.map(|m| m + 1).unwrap_or(0)
    }
    pub fn vec_like(&self) -> u32 {
        self.vec_max_id.map(|m| m + 1).unwrap_or(0)
    }
    pub fn mask_like(&self) -> u32 {
        self.mask_max_id.map(|m| m + 1).unwrap_or(0)
    }
    pub fn tile_like(&self) -> u32 {
        self.tile_max_id.map(|m| m + 1).unwrap_or(0)
    }
}

/// 层模板 ABI 映射 — 模板内 VRegId → 主程序 VRegId 的对应关系。
#[derive(Debug, Clone)]
pub struct LayerAbiMap {
    pub input_ptr: VRegId,
    pub weight_ptr: VRegId,
    pub output_ptr: VRegId,
    pub scratch_base: VRegId,
}

/// 异构层模板 — 一组 fusion groups 编译后的 VmProgram + ABI 映射。
///
/// 用于第一层并行化：异构模型的 4 种层类型各自独立编译为 LayerTemplate，
/// 然后在主 VmProgram 中按顺序实例化。
#[derive(Debug, Clone)]
pub struct LayerTemplate {
    pub body: VmProgram,
    pub abi_map: LayerAbiMap,
}

/// 编译时状态追踪序列——记录完整的代码生成状态转移 + VReg 声明。
///
/// 不是虚拟机程序。这是半结构化 IR，供 RegAlloc/StackFrame/IsaLower 消费。
#[derive(Debug, Clone)]
pub struct VmProgram {
    /// 状态追踪记录序列
    pub instrs: Vec<VmInstr>,
    /// 下一个 VRegId 分配值
    next_vreg: u32,
    /// 下一个 ScopeId 分配值 (REQ-LC-005)
    next_scope_id: usize,
    /// 下一个 label ID 分配值 (MarkLabel/BranchIfGprZero/etc.)
    next_label_id: usize,
}

impl VmProgram {
    pub fn new() -> Self {
        Self { instrs: Vec::new(), next_vreg: 0, next_scope_id: 0, next_label_id: 1000 }
    }

    /// Allocate a unique label ID for MarkLabel / branch targets.
    /// Starts at 1000 to avoid collisions with hardcoded labels in mega_kernel_emit.
    pub fn alloc_label(&mut self) -> usize {
        let id = self.next_label_id;
        self.next_label_id += 1;
        id
    }

    /// 分配新的 VRegId 并发射 DeclareVReg。
    pub fn alloc_vreg(&mut self, kind: VRegKind, width: SimdWidth) -> VRegId {
        let id = VRegId(self.next_vreg);
        self.next_vreg += 1;
        self.instrs.push(VmInstr::DeclareVReg { id, kind, width });
        id
    }

    /// 已分配 VReg 总数（= next_vreg）。
    /// ARCH-GPU-REG-COUNT: GPU Lower 据此动态声明 PTX 寄存器数量。
    pub fn vreg_count(&self) -> u32 {
        self.next_vreg
    }

    /// 按 VRegKind 统计每类 VReg 的最大 VRegId.0。
    /// ARCH-GPU-REG-NAMESPACE: VRegId 全局递增，但 PTX %f/%r/%p 独立命名空间。
    /// 声明容量 = max_id + 1，确保 0..=max_id 都被覆盖。
    pub fn vreg_counts_by_kind(&self) -> VRegKindCounts {
        let mut counts = VRegKindCounts::default();
        fn update_max(slot: &mut Option<u32>, id: u32) {
            *slot = Some(slot.map_or(id, |cur| cur.max(id)));
        }
        for instr in &self.instrs {
            if let VmInstr::DeclareVReg { id, kind, .. } = instr {
                match kind {
                    VRegKind::Ptr | VRegKind::Scalar | VRegKind::Counter | VRegKind::ByteOffset => {
                        update_max(&mut counts.gpr_max_id, id.0);
                    }
                    VRegKind::Vec => update_max(&mut counts.vec_max_id, id.0),
                    VRegKind::Mask => update_max(&mut counts.mask_max_id, id.0),
                    VRegKind::Tile => update_max(&mut counts.tile_max_id, id.0),
                }
            }
        }
        counts
    }

    /// 发射一条指令。
    pub fn emit(&mut self, instr: VmInstr) {
        self.instrs.push(instr);
    }

    /// 发射循环: LoopBegin + body + LoopEnd。
    /// 自动分配 counter 和 byte_offset VReg。
    pub fn emit_loop(
        &mut self,
        bound: BoundExpr,
        step_bytes: usize,
        body: impl FnOnce(&mut Self, VRegId, VRegId),
    ) {
        let counter = self.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_offset = self.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        self.emit(VmInstr::LoopBegin { counter, byte_offset, bound, step_bytes });
        body(self, counter, byte_offset);
        self.emit(VmInstr::LoopEnd);
    }

    /// 发射作用域: ScopeBegin + body + ScopeEnd。
    /// 自动分配 scope_id，用于 ScopedSpillAllocator 的 scope-based 回收。
    pub fn emit_scope<E>(&mut self, body: impl FnOnce(&mut Self) -> Result<(), E>) -> Result<(), E> {
        let scope_id = self.next_scope_id;
        self.next_scope_id += 1;
        self.emit(VmInstr::ScopeBegin { scope_id });
        body(self)?;
        self.emit(VmInstr::ScopeEnd { scope_id });
        Ok(())
    }

    /// 发射循环 (Result 版本): LoopBegin + body + LoopEnd。
    /// 与 emit_loop 相同，但闭包返回 Result，支持 auto_lower_trace_raw 错误传播。
    pub fn emit_loop_try<E>(
        &mut self,
        bound: BoundExpr,
        step_bytes: usize,
        body: impl FnOnce(&mut Self, VRegId, VRegId) -> Result<(), E>,
    ) -> Result<(), E> {
        let counter = self.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_offset = self.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        self.emit(VmInstr::LoopBegin {
            counter,
            byte_offset,
            bound,
            step_bytes,
        });
        body(self, counter, byte_offset)?;
        self.emit(VmInstr::LoopEnd);
        Ok(())
    }

    /// 指令数量。
    pub fn len(&self) -> usize {
        self.instrs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instrs.is_empty()
    }

    /// 合并另一个 VmProgram 的指令。
    /// VRegId 需要重映射以避免与当前程序冲突。
    pub fn append(&mut self, other: VmProgram) {
        let offset = self.next_vreg;
        for instr in other.instrs {
            self.instrs.push(Self::remap_vreg_instr(instr, offset));
        }
        self.next_vreg += other.next_vreg;
    }

    /// 合并模板：VReg 重映射 + ABI 指针替换。
    ///
    /// 将模板 `other` 中的所有指令合并到 `self` 中，同时：
    /// - 按 `subst` 映射重写 VRegId（模板内部 VReg → 主程序 VReg）
    /// - DeclareVReg 指令跳过（已在主程序中声明）
    /// - 未在 subst 中的 VRegId 分配新的主程序 VRegId
    pub fn append_with_mapping(
        &mut self,
        other: VmProgram,
        subst: &[(VRegId, VRegId)],
    ) {
        // Collect VReg kinds/widths from template for allocating new VRegs in main program
        let template_kinds = other.collect_vreg_kinds();
        let template_widths = other.collect_vreg_widths();

        // Build mapping: template VRegId → main program VRegId
        let mut vreg_map: std::collections::HashMap<VRegId, VRegId> =
            std::collections::HashMap::from_iter(subst.iter().cloned());

        // Track next_vreg externally to avoid borrow conflicts with self.emit()
        let mut next_vreg = self.next_vreg;

        for instr in other.instrs {
            match &instr {
                VmInstr::DeclareVReg { .. } => continue, // skip declarations
                _ => {}
            }

            let remapped = Self::remap_vreg_with_map(
                instr,
                &mut vreg_map,
                &template_kinds,
                &template_widths,
                &mut next_vreg,
            );
            self.emit(remapped);
        }

        self.next_vreg = next_vreg;
    }

    /// Remap VRegIds in a single instruction using a substitution map.
    ///
    /// For VRegs already in `map`, uses the mapped value.
    /// For VRegs not yet in `map`, allocates a new VRegId via `next_vreg` and records it.
    fn remap_vreg_with_map(
        instr: VmInstr,
        map: &mut std::collections::HashMap<VRegId, VRegId>,
        _kinds: &std::collections::HashMap<VRegId, VRegKind>,
        _widths: &std::collections::HashMap<VRegId, SimdWidth>,
        next_vreg: &mut u32,
    ) -> VmInstr {
        let r = |vreg: VRegId, m: &mut std::collections::HashMap<VRegId, VRegId>, nv: &mut u32| -> VRegId {
            *m.entry(vreg).or_insert_with(|| {
                let id = VRegId(*nv);
                *nv += 1;
                id
            })
        };
        let remap_offset = |oe: OffsetExpr, m: &mut std::collections::HashMap<VRegId, VRegId>, nv: &mut u32| -> OffsetExpr {
            Self::remap_offset_with_map(oe, m, nv)
        };
        let remap_scalar = |se: ScalarExpr, m: &mut std::collections::HashMap<VRegId, VRegId>, nv: &mut u32| -> ScalarExpr {
            match se {
                ScalarExpr::Const(_) => se,
                ScalarExpr::MemLoad(base, off) => ScalarExpr::MemLoad(r(base, m, nv), remap_offset(off, m, nv)),
                ScalarExpr::ExtractLane0(v) => ScalarExpr::ExtractLane0(r(v, m, nv)),
                ScalarExpr::VReg(v) => ScalarExpr::VReg(r(v, m, nv)),
            }
        };
        let remap_ptr = |pe: PtrExpr, m: &mut std::collections::HashMap<VRegId, VRegId>, nv: &mut u32| -> PtrExpr {
            match pe {
                PtrExpr::VRegPlusVReg(a, b) => PtrExpr::VRegPlusVReg(r(a, m, nv), r(b, m, nv)),
                other => other,
            }
        };

        match instr {
            VmInstr::DeclareVReg { id, kind, width } => VmInstr::DeclareVReg { id: r(id, map, next_vreg), kind, width },
            VmInstr::ReleaseVReg { id } => VmInstr::ReleaseVReg { id: r(id, map, next_vreg) },
            VmInstr::VecLoad { dst, base, offset, width, dtype, predicate } => VmInstr::VecLoad { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset: remap_offset(offset, map, next_vreg), width, dtype, predicate },
            VmInstr::VecStore { base, offset, src, width, dtype, predicate } => VmInstr::VecStore { base: r(base, map, next_vreg), offset: remap_offset(offset, map, next_vreg), src: r(src, map, next_vreg), width, dtype, predicate },
            VmInstr::VecNarrow { dst, src, dst_dtype, src_dtype, width } => VmInstr::VecNarrow { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), dst_dtype, src_dtype, width },
            VmInstr::VecWiden { dst, src, dst_dtype, src_dtype, width } => VmInstr::VecWiden { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), dst_dtype, src_dtype, width },
            VmInstr::Mov { dst, src, dtype } => VmInstr::Mov { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), dtype },
            VmInstr::Broadcast { dst, src, width, dtype } => VmInstr::Broadcast { dst: r(dst, map, next_vreg), src: remap_scalar(src, map, next_vreg), width, dtype },
            VmInstr::LoadPtr { dst, src } => VmInstr::LoadPtr { dst: r(dst, map, next_vreg), src: remap_ptr(src, map, next_vreg) },
            VmInstr::VecBinOp { dst, a, b, op, dtype } => VmInstr::VecBinOp { dst: r(dst, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg), op, dtype },
            VmInstr::VecShiftImm { dst, a, amount, op, width } => VmInstr::VecShiftImm { dst: r(dst, map, next_vreg), a: r(a, map, next_vreg), amount, op, width },
            VmInstr::VecUnaryOp { dst, a, op } => VmInstr::VecUnaryOp { dst: r(dst, map, next_vreg), a: r(a, map, next_vreg), op },
            VmInstr::VecCmp { dst, a, b, pred } => VmInstr::VecCmp { dst: r(dst, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg), pred },
            VmInstr::VecCast { dst, src, from_bits, to_bits } => VmInstr::VecCast { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), from_bits, to_bits },
            VmInstr::ConditionalSelect { dst, mask, true_val, false_val } => VmInstr::ConditionalSelect { dst: r(dst, map, next_vreg), mask: r(mask, map, next_vreg), true_val: r(true_val, map, next_vreg), false_val: r(false_val, map, next_vreg) },
            VmInstr::Fma { dst, acc, a, b, dtype } => VmInstr::Fma { dst: r(dst, map, next_vreg), acc: r(acc, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg), dtype },
            VmInstr::HReduce { dst, src, op } => VmInstr::HReduce { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), op },
            VmInstr::Accumulate { acc, src } => VmInstr::Accumulate { acc: r(acc, map, next_vreg), src: r(src, map, next_vreg) },
            VmInstr::LoopBegin { counter, byte_offset, bound, step_bytes } => VmInstr::LoopBegin { counter: r(counter, map, next_vreg), byte_offset: r(byte_offset, map, next_vreg), bound, step_bytes },
            VmInstr::LoopEnd => VmInstr::LoopEnd,
            VmInstr::ScopeBegin { scope_id } => VmInstr::ScopeBegin { scope_id },
            VmInstr::ScopeEnd { scope_id } => VmInstr::ScopeEnd { scope_id },
            VmInstr::ConditionalSkip { mask, skip_count } => VmInstr::ConditionalSkip { mask: r(mask, map, next_vreg), skip_count },
            VmInstr::GprCondAction { cond, action } => VmInstr::GprCondAction {
                cond: match cond {
                    GprCondition::IsNull(v) => GprCondition::IsNull(r(v, map, next_vreg)),
                    GprCondition::IsNonNull(v) => GprCondition::IsNonNull(r(v, map, next_vreg)),
                    GprCondition::CmpEq(v, c) => GprCondition::CmpEq(r(v, map, next_vreg), c),
                    GprCondition::CmpLtU(v, c) => GprCondition::CmpLtU(r(v, map, next_vreg), c),
                    GprCondition::CmpGeU(v, c) => GprCondition::CmpGeU(r(v, map, next_vreg), c),
                    GprCondition::BitClear(v, b) => GprCondition::BitClear(r(v, map, next_vreg), b),
                    GprCondition::BitSet(v, b) => GprCondition::BitSet(r(v, map, next_vreg), b),
                },
                action: match action {
                    GprBranchAction::Skip(n) => GprBranchAction::Skip(n),
                    GprBranchAction::Exit(v) => GprBranchAction::Exit(r(v, map, next_vreg)),
                    GprBranchAction::JumpToLabel(label_id) => GprBranchAction::JumpToLabel(label_id),
                },
            },
            VmInstr::TileConfig { rows, cols, dtype } => VmInstr::TileConfig { rows, cols, dtype },
            VmInstr::TileMma { c, a, b } => VmInstr::TileMma { c: r(c, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg) },
            VmInstr::TileRelease => VmInstr::TileRelease,
            VmInstr::Vp2Intersect { dst_k0, dst_k1, a, b } => VmInstr::Vp2Intersect { dst_k0: r(dst_k0, map, next_vreg), dst_k1: r(dst_k1, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg) },
            VmInstr::WarpSync => VmInstr::WarpSync,
            VmInstr::AsyncCopy { dst, src, size } => VmInstr::AsyncCopy { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), size },
            VmInstr::AsyncWait { handle } => VmInstr::AsyncWait { handle },
            VmInstr::Transcendental { dst, src, func } => VmInstr::Transcendental { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), func },
            VmInstr::HotpatchSlot { slot_id, initial_target, alternatives } => VmInstr::HotpatchSlot { slot_id, initial_target, alternatives },
            VmInstr::IndirectJump { index, targets } => VmInstr::IndirectJump { index: r(index, map, next_vreg), targets },
            VmInstr::ConditionalExit { condition, output } => VmInstr::ConditionalExit { condition: r(condition, map, next_vreg), output: r(output, map, next_vreg) },
            VmInstr::BranchIfPtrNonNull { ptr, target_label } => VmInstr::BranchIfPtrNonNull { ptr: r(ptr, map, next_vreg), target_label },
            VmInstr::BranchIfGprZero { value, target_label } => VmInstr::BranchIfGprZero { value: r(value, map, next_vreg), target_label },
            VmInstr::BranchIfGprLtU { a, b, target_label } => VmInstr::BranchIfGprLtU { a: r(a, map, next_vreg), b: r(b, map, next_vreg), target_label },
            VmInstr::UnconditionalBranch { target_label } => VmInstr::UnconditionalBranch { target_label },
            VmInstr::BatchSeqIdLookup { dst, pt_offset_out, token_index, batch_ctx_ptr } => VmInstr::BatchSeqIdLookup { dst: r(dst, map, next_vreg), pt_offset_out: r(pt_offset_out, map, next_vreg), token_index: r(token_index, map, next_vreg), batch_ctx_ptr: r(batch_ctx_ptr, map, next_vreg) },
            VmInstr::BatchPerSeqArgmax { dst, seq_id, logits_flat_ptr, vocab_size, width } => VmInstr::BatchPerSeqArgmax { dst: r(dst, map, next_vreg), seq_id: r(seq_id, map, next_vreg), logits_flat_ptr: r(logits_flat_ptr, map, next_vreg), vocab_size, width },
            VmInstr::BatchPerSeqStopCheck { seq_id, token_id, batch_ctx_ptr } => VmInstr::BatchPerSeqStopCheck { seq_id: r(seq_id, map, next_vreg), token_id: r(token_id, map, next_vreg), batch_ctx_ptr: r(batch_ctx_ptr, map, next_vreg) },
            VmInstr::AtomicAdd { base, offset, value, elem_width } => VmInstr::AtomicAdd { base: r(base, map, next_vreg), offset: remap_offset(offset, map, next_vreg), value, elem_width },
            VmInstr::MemFence { order } => VmInstr::MemFence { order },
            VmInstr::Argmax { dst, logits_ptr, vocab_bytes, width } => VmInstr::Argmax { dst: r(dst, map, next_vreg), logits_ptr: r(logits_ptr, map, next_vreg), vocab_bytes, width },
            VmInstr::TemperatureScale { logits_ptr, temp_ptr, vocab_bytes, width } => VmInstr::TemperatureScale { logits_ptr: r(logits_ptr, map, next_vreg), temp_ptr: r(temp_ptr, map, next_vreg), vocab_bytes, width },
            VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => VmInstr::StoreToken { token_id: r(token_id, map, next_vreg), output_buf: r(output_buf, map, next_vreg), counter: r(counter, map, next_vreg), input_ids_ptr: r(input_ids_ptr, map, next_vreg), prompt_len_bytes: r(prompt_len_bytes, map, next_vreg) },
            VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => VmInstr::CheckStopCondition { token_id: r(token_id, map, next_vreg), counter: r(counter, map, next_vreg), eos_ptr: r(eos_ptr, map, next_vreg), max_tokens_ptr: r(max_tokens_ptr, map, next_vreg) },
            VmInstr::AddPtr { dst, base, offset } => VmInstr::AddPtr { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset },
            VmInstr::StoreConstToStack { rbp_offset, value, elem_width } => VmInstr::StoreConstToStack { rbp_offset, value, elem_width },
            VmInstr::BreakLoop { return_value } => VmInstr::BreakLoop {
                return_value: match return_value {
                    ReturnValue::Const(v) => ReturnValue::Const(v),
                    ReturnValue::VReg(v) => ReturnValue::VReg(r(v, map, next_vreg)),
                },
            },
            VmInstr::MarkLabel { label_id } => VmInstr::MarkLabel { label_id },
            VmInstr::GprBinOp { dst, a, b, op } => VmInstr::GprBinOp {
                dst: r(dst, map, next_vreg), a: r(a, map, next_vreg),
                b: match b {
                    GprOperand::VReg(v) => GprOperand::VReg(r(v, map, next_vreg)),
                    GprOperand::Imm(v) => GprOperand::Imm(v),
                }, op,
            },
            VmInstr::GprUnaryOp { dst, src, op } => VmInstr::GprUnaryOp { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), op },
            VmInstr::GprLoadImm { dst, value } => VmInstr::GprLoadImm { dst: r(dst, map, next_vreg), value },
            VmInstr::LoadCallbackEntry { table_ptr, slot_id, fn_ptr_out, ctx_out } => VmInstr::LoadCallbackEntry { table_ptr: r(table_ptr, map, next_vreg), slot_id, fn_ptr_out: r(fn_ptr_out, map, next_vreg), ctx_out: r(ctx_out, map, next_vreg) },
            VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr } => VmInstr::NativeCall { ret_val: r(ret_val, map, next_vreg), fn_ptr: r(fn_ptr, map, next_vreg), ctx_ptr: r(ctx_ptr, map, next_vreg) },
            VmInstr::ScalarLoad { dst, base, offset } => VmInstr::ScalarLoad { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset: remap_offset(offset, map, next_vreg) },
            VmInstr::ScalarStore { base, src, offset } => VmInstr::ScalarStore { base: r(base, map, next_vreg), src: r(src, map, next_vreg), offset: remap_offset(offset, map, next_vreg) },
            VmInstr::VecScalarStore { base, src, offset } => VmInstr::VecScalarStore { base: r(base, map, next_vreg), src: r(src, map, next_vreg), offset: remap_offset(offset, map, next_vreg) },
            VmInstr::ScalarToIndex { dst, src, stride } => VmInstr::ScalarToIndex { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), stride },
            VmInstr::IndexToScalar { dst, src } => VmInstr::IndexToScalar { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg) },
            VmInstr::IntMulStride { dst, src, stride } => VmInstr::IntMulStride { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), stride },
            VmInstr::ScalarByteLoad { dst, base, offset } => VmInstr::ScalarByteLoad { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset: remap_offset(offset, map, next_vreg) },
            VmInstr::QuantBlockLoad { dst, base, offset, unpack, width } => VmInstr::QuantBlockLoad { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset, unpack: unpack.remap_vregs(|v| r(v, map, next_vreg)), width },
            VmInstr::Prefetch { base, offset, distance, hint } => VmInstr::Prefetch { base: r(base, map, next_vreg), offset: remap_offset(offset, map, next_vreg), distance, hint },
            VmInstr::GatherLoad { dst, base, indices, stride, width, dtype, predicate } => VmInstr::GatherLoad { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), indices: r(indices, map, next_vreg), stride, width, dtype, predicate },
            VmInstr::ScatterStore { base, indices, src, stride, width, dtype, predicate } => VmInstr::ScatterStore { base: r(base, map, next_vreg), indices: r(indices, map, next_vreg), src: r(src, map, next_vreg), stride, width, dtype, predicate },
            VmInstr::TableLookup { dst, base, row_index, row_bytes, width } => VmInstr::TableLookup { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), row_index: r(row_index, map, next_vreg), row_bytes, width },
            VmInstr::ActivationSwap { ptr_a, ptr_b } => VmInstr::ActivationSwap { ptr_a: r(ptr_a, map, next_vreg), ptr_b: r(ptr_b, map, next_vreg) },
            VmInstr::PageTableAddr { dst, pool_base, page_table_ptr, ki_byte_off, row_bytes, page_size, page_stride, base_offset, seq_pt_offset } => VmInstr::PageTableAddr { dst: r(dst, map, next_vreg), pool_base: r(pool_base, map, next_vreg), page_table_ptr: r(page_table_ptr, map, next_vreg), ki_byte_off: remap_offset(ki_byte_off, map, next_vreg), row_bytes, page_size, page_stride, base_offset, seq_pt_offset: seq_pt_offset.map(|v| r(v, map, next_vreg)) },
            VmInstr::PageTableKVWrite { src, pool_base, page_table_ptr, seq_index, row_bytes, page_size, page_stride, base_offset, width, dtype } => VmInstr::PageTableKVWrite { src: r(src, map, next_vreg), pool_base: r(pool_base, map, next_vreg), page_table_ptr: r(page_table_ptr, map, next_vreg), seq_index: r(seq_index, map, next_vreg), row_bytes, page_size, page_stride, base_offset, width, dtype },
            VmInstr::PageTableKVWriteQuant { src, pool_base, page_table_ptr, seq_index, quant_row_bytes, fp32_row_bytes, page_size, page_stride, base_offset, scale_offset, width, kivi_mode, num_elems } => VmInstr::PageTableKVWriteQuant { src: r(src, map, next_vreg), pool_base: r(pool_base, map, next_vreg), page_table_ptr: r(page_table_ptr, map, next_vreg), seq_index: r(seq_index, map, next_vreg), quant_row_bytes, fp32_row_bytes, page_size, page_stride, base_offset, scale_offset, width, kivi_mode, num_elems },
            VmInstr::KiviQuantChannel { src, dst_ptr, scale_ptr, num_channels, width } => VmInstr::KiviQuantChannel { src: r(src, map, next_vreg), dst_ptr: r(dst_ptr, map, next_vreg), scale_ptr: r(scale_ptr, map, next_vreg), num_channels, width },
            VmInstr::KiviQuantToken { src, dst_ptr, scale_ptr, num_elems, width } => VmInstr::KiviQuantToken { src: r(src, map, next_vreg), dst_ptr: r(dst_ptr, map, next_vreg), scale_ptr: r(scale_ptr, map, next_vreg), num_elems, width },
            VmInstr::KiviDequantLoad { dst, src_ptr, scale_ptr, num_elems, width } => VmInstr::KiviDequantLoad { dst: r(dst, map, next_vreg), src_ptr: r(src_ptr, map, next_vreg), scale_ptr: r(scale_ptr, map, next_vreg), num_elems, width },
            VmInstr::SharedMemAlloc { name, bytes } => VmInstr::SharedMemAlloc { name, bytes },
            VmInstr::SharedMemStore { name, dst_offset, src, width, dtype } => VmInstr::SharedMemStore { name, dst_offset: remap_offset(dst_offset, map, next_vreg), src: r(src, map, next_vreg), width, dtype },
            VmInstr::SharedMemLoad { dst, name, src_offset, width, dtype } => VmInstr::SharedMemLoad { dst: r(dst, map, next_vreg), name, src_offset: remap_offset(src_offset, map, next_vreg), width, dtype },
            VmInstr::SharedMemAsyncStore { name, dst_offset, src, width, dtype } => VmInstr::SharedMemAsyncStore { name, dst_offset: remap_offset(dst_offset, map, next_vreg), src: r(src, map, next_vreg), width, dtype },
            VmInstr::SharedMemAsyncWaitGroup { n } => VmInstr::SharedMemAsyncWaitGroup { n },
            VmInstr::WeightPrefetchAsync { smem_name, weight_base, weight_offset, size } => VmInstr::WeightPrefetchAsync { smem_name, weight_base: r(weight_base, map, next_vreg), weight_offset, size },
            VmInstr::WeightPrefetchWait { group } => VmInstr::WeightPrefetchWait { group },
            VmInstr::WarpRoleDeclare { role } => VmInstr::WarpRoleDeclare { role },
            VmInstr::WarpBarrierArrive { barrier_name, tx_bytes } => VmInstr::WarpBarrierArrive { barrier_name, tx_bytes },
            VmInstr::WarpBarrierWait { barrier_name, parity } => VmInstr::WarpBarrierWait { barrier_name, parity },
            VmInstr::TmaDescriptorInit { desc_name, global_dim, global_stride, box_dim, swizzle, dtype } => VmInstr::TmaDescriptorInit { desc_name, global_dim, global_stride, box_dim, swizzle, dtype },
            VmInstr::Tma2DCopy { desc_name, smem_name, coord_x, coord_y, barrier_name } => VmInstr::Tma2DCopy { desc_name, smem_name, coord_x: r(coord_x, map, next_vreg), coord_y: r(coord_y, map, next_vreg), barrier_name },
            VmInstr::BarrierInit { name, thread_count } => VmInstr::BarrierInit { name, thread_count },
            VmInstr::BlockSync => VmInstr::BlockSync,
            VmInstr::WarpReduce { op, src, dst, width } => VmInstr::WarpReduce { op, src: r(src, map, next_vreg), dst: r(dst, map, next_vreg), width },
            // Quant* decode instrs
            VmInstr::QuantBroadcastInt { dst, value, width } => VmInstr::QuantBroadcastInt { dst: r(dst, map, next_vreg), value, width },
            VmInstr::QuantScalarCvtLoad { dst, base, offset, src_dtype, width } => VmInstr::QuantScalarCvtLoad { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset, src_dtype, width },
            VmInstr::QuantBiPlaneLoad { dst, qs_base, extra_base, bias, mode, width } => VmInstr::QuantBiPlaneLoad { dst: r(dst, map, next_vreg), qs_base: r(qs_base, map, next_vreg), extra_base: r(extra_base, map, next_vreg), bias, mode, width },
            VmInstr::QuantLoadBytesVec { dst, base, offset, count, signed, width } => VmInstr::QuantLoadBytesVec { dst: r(dst, map, next_vreg), base: r(base, map, next_vreg), offset, count, signed, width },
            VmInstr::QuantCodebookLookup { dst, indices, codebook_data, vector_size, bits_per_entry, width } => VmInstr::QuantCodebookLookup { dst: r(dst, map, next_vreg), indices: r(indices, map, next_vreg), codebook_data, vector_size, bits_per_entry, width },
            VmInstr::QuantExtractBits { dst, src, bit_offset, bit_width, width } => VmInstr::QuantExtractBits { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), bit_offset, bit_width, width },            VmInstr::QuantDequantFma { dst, weight, activation, scale, zero_point, quant_kind, dtype, width } => VmInstr::QuantDequantFma { dst: r(dst, map, next_vreg), weight: r(weight, map, next_vreg), activation: r(activation, map, next_vreg), scale: r(scale, map, next_vreg), zero_point: r(zero_point, map, next_vreg), quant_kind, dtype, width },            VmInstr::QuantInterleave { dst, lo, hi, width } => VmInstr::QuantInterleave { dst: r(dst, map, next_vreg), lo: r(lo, map, next_vreg), hi: r(hi, map, next_vreg), width },            VmInstr::QuantConcatSeq { dst, lo, hi, width } => VmInstr::QuantConcatSeq { dst: r(dst, map, next_vreg), lo: r(lo, map, next_vreg), hi: r(hi, map, next_vreg), width },            VmInstr::Q3KDecodeStep { dst, block_base, lane_offset, d_vreg, qs_offset, hmask_offset, lanes, width } => VmInstr::Q3KDecodeStep { dst: r(dst, map, next_vreg), block_base: r(block_base, map, next_vreg), lane_offset: r(lane_offset, map, next_vreg), d_vreg: r(d_vreg, map, next_vreg), qs_offset, hmask_offset, lanes, width },            VmInstr::DotProduct { acc, a, b, input_dtype, width } => VmInstr::DotProduct { acc: r(acc, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg), input_dtype, width },
            VmInstr::ScaleApply { dst, acc, scale, zero, input_dtype, width } => VmInstr::ScaleApply { dst: r(dst, map, next_vreg), acc: r(acc, map, next_vreg), scale: r(scale, map, next_vreg), zero: r(zero, map, next_vreg), input_dtype, width },
            VmInstr::BitwiseGemm { dst, sign_bits, input_sign_bits, scale, width } => VmInstr::BitwiseGemm { dst: r(dst, map, next_vreg), sign_bits: r(sign_bits, map, next_vreg), input_sign_bits: r(input_sign_bits, map, next_vreg), scale: r(scale, map, next_vreg), width },
            VmInstr::SparseGemm { acc, a_sparse, b_dense, sparse_mask_ptr, m, n, k, width, dtype } => VmInstr::SparseGemm { acc: r(acc, map, next_vreg), a_sparse: r(a_sparse, map, next_vreg), b_dense: r(b_dense, map, next_vreg), sparse_mask_ptr: r(sparse_mask_ptr, map, next_vreg), m, n, k, width, dtype },
            VmInstr::NativeFp4Gemm { acc, a, b, scale_a, scale_b, m, n, k, width } => VmInstr::NativeFp4Gemm { acc: r(acc, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg), scale_a: r(scale_a, map, next_vreg), scale_b: r(scale_b, map, next_vreg), m, n, k, width },
            VmInstr::SparseFp8Gemm { acc, a_sparse, b_dense, sparse_mask_ptr, m, n, k, width, fp8_kind } => VmInstr::SparseFp8Gemm { acc: r(acc, map, next_vreg), a_sparse: r(a_sparse, map, next_vreg), b_dense: r(b_dense, map, next_vreg), sparse_mask_ptr: r(sparse_mask_ptr, map, next_vreg), m, n, k, width, fp8_kind },
            VmInstr::NativeFp8Gemm { acc, a, b, m, n, k, width, fp8_kind } => VmInstr::NativeFp8Gemm { acc: r(acc, map, next_vreg), a: r(a, map, next_vreg), b: r(b, map, next_vreg), m, n, k, width, fp8_kind },
            VmInstr::HwQuantDequant { dst, packed_weight, block_scale, global_scale, quant_kind, count, width } => VmInstr::HwQuantDequant { dst: r(dst, map, next_vreg), packed_weight: r(packed_weight, map, next_vreg), block_scale: r(block_scale, map, next_vreg), global_scale: r(global_scale, map, next_vreg), quant_kind, count, width },
            VmInstr::TmemAlloc { name, bytes } => VmInstr::TmemAlloc { name, bytes },
            VmInstr::TmemLoad { dst, name, offset, width, dtype } => VmInstr::TmemLoad { dst: r(dst, map, next_vreg), name, offset: remap_offset(offset, map, next_vreg), width, dtype },
            VmInstr::TmemStore { name, offset, src, width, dtype } => VmInstr::TmemStore { name, offset: remap_offset(offset, map, next_vreg), src: r(src, map, next_vreg), width, dtype },
            VmInstr::TmemDealloc { name } => VmInstr::TmemDealloc { name },
            VmInstr::ClusterBarrierInit { name, thread_count } => VmInstr::ClusterBarrierInit { name, thread_count },
            VmInstr::ClusterStore { name, offset, src, width, dtype } => VmInstr::ClusterStore { name, offset: remap_offset(offset, map, next_vreg), src: r(src, map, next_vreg), width, dtype },
            VmInstr::ClusterLoad { dst, name, offset, width, dtype } => VmInstr::ClusterLoad { dst: r(dst, map, next_vreg), name, offset: remap_offset(offset, map, next_vreg), width, dtype },
            VmInstr::Comment(s) => VmInstr::Comment(s),
            VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, decompressed_size } => VmInstr::Lz4Decode { src_ptr: r(src_ptr, map, next_vreg), dst_ptr: r(dst_ptr, map, next_vreg), compressed_size: r(compressed_size, map, next_vreg), decompressed_size },
            VmInstr::BitPackRleDecode { src_ptr, dst_ptr, compressed_size, nibble_bits, element_count } => VmInstr::BitPackRleDecode { src_ptr: r(src_ptr, map, next_vreg), dst_ptr: r(dst_ptr, map, next_vreg), compressed_size: r(compressed_size, map, next_vreg), nibble_bits, element_count },
            VmInstr::GgufSubScaleLoad { dst, scales_base, sub_block_idx, width } => VmInstr::GgufSubScaleLoad { dst: r(dst, map, next_vreg), scales_base: r(scales_base, map, next_vreg), sub_block_idx: r(sub_block_idx, map, next_vreg), width },
            VmInstr::GgufKQuantScaleLoad { dst, scales_base, sub_block_idx, scales_count, is_q3k_extended, is_min, width } => VmInstr::GgufKQuantScaleLoad { dst: r(dst, map, next_vreg), scales_base: r(scales_base, map, next_vreg), sub_block_idx: r(sub_block_idx, map, next_vreg), scales_count, is_q3k_extended, is_min, width },
            // GPU sampling instrs
            VmInstr::SoftmaxReduceMax { dst, logits_ptr, vocab_bytes, width } => VmInstr::SoftmaxReduceMax { dst: r(dst, map, next_vreg), logits_ptr: r(logits_ptr, map, next_vreg), vocab_bytes, width },
            VmInstr::SoftmaxExpSum { sum_dst, logits_ptr, max_val, vocab_bytes, width } => VmInstr::SoftmaxExpSum { sum_dst: r(sum_dst, map, next_vreg), logits_ptr: r(logits_ptr, map, next_vreg), max_val: r(max_val, map, next_vreg), vocab_bytes, width },
            VmInstr::SoftmaxNormalize { logits_ptr, sum_val, vocab_bytes, width } => VmInstr::SoftmaxNormalize { logits_ptr: r(logits_ptr, map, next_vreg), sum_val: r(sum_val, map, next_vreg), vocab_bytes, width },
            VmInstr::SampleTopKFilter { probs_ptr, indices_ptr, k_ptr, vocab_bytes, width } => VmInstr::SampleTopKFilter { probs_ptr: r(probs_ptr, map, next_vreg), indices_ptr: r(indices_ptr, map, next_vreg), k_ptr: r(k_ptr, map, next_vreg), vocab_bytes, width },
            VmInstr::SampleTopPFilter { probs_ptr, p_ptr, vocab_bytes, width } => VmInstr::SampleTopPFilter { probs_ptr: r(probs_ptr, map, next_vreg), p_ptr: r(p_ptr, map, next_vreg), vocab_bytes, width },
            VmInstr::SampleMultinomial { dst, probs_ptr, rng_state_ptr, vocab_bytes, width } => VmInstr::SampleMultinomial { dst: r(dst, map, next_vreg), probs_ptr: r(probs_ptr, map, next_vreg), rng_state_ptr: r(rng_state_ptr, map, next_vreg), vocab_bytes, width },
            VmInstr::WarpPRNG { dst, rng_state_ptr } => VmInstr::WarpPRNG { dst: r(dst, map, next_vreg), rng_state_ptr: r(rng_state_ptr, map, next_vreg) },
            VmInstr::SharedMemSwizzle { dst, raw_addr, log2_banks, log2_bank_width } => VmInstr::SharedMemSwizzle { dst: r(dst, map, next_vreg), raw_addr: r(raw_addr, map, next_vreg), log2_banks, log2_bank_width },
            VmInstr::DebugBreakpoint { label } => VmInstr::DebugBreakpoint { label: label.clone() },
            VmInstr::DebugMarker { message } => VmInstr::DebugMarker { message: message.clone() },
            VmInstr::DebugProbe { vreg, probe_name, width } => VmInstr::DebugProbe { vreg: r(vreg, map, next_vreg), probe_name: probe_name.clone(), width },
            VmInstr::DebugBreakIf { label, cond_gpr } => VmInstr::DebugBreakIf { label: label.clone(), cond_gpr: r(cond_gpr, map, next_vreg) },
            VmInstr::MemCopy { dst, src, bytes, dtype, guard, effect } => VmInstr::MemCopy { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), bytes, dtype, guard, effect },
            VmInstr::VecShuffle { dst, src, mask, width } => {
                let remapped_mask = match mask {
                    VecShuffleMask::Const(v) => VecShuffleMask::Const(v),
                    VecShuffleMask::Dynamic { ctrl } => VecShuffleMask::Dynamic { ctrl: r(ctrl, map, next_vreg) },
                };
                VmInstr::VecShuffle { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), mask: remapped_mask, width }
            }
            VmInstr::VecExtractLane { dst, src, lane, dtype } => VmInstr::VecExtractLane { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), lane, dtype },
            VmInstr::VecInsertLane { dst, src_vec, src_scalar, lane, dtype } => VmInstr::VecInsertLane { dst: r(dst, map, next_vreg), src_vec: r(src_vec, map, next_vreg), src_scalar: r(src_scalar, map, next_vreg), lane, dtype },
            VmInstr::VecLoadConst { dst, values, dtype, width } => VmInstr::VecLoadConst { dst: r(dst, map, next_vreg), values, dtype, width },
            VmInstr::AtomicCAS { dst, ptr, expected, desired, elem_width, success_order, failure_order } => VmInstr::AtomicCAS { dst: r(dst, map, next_vreg), ptr: r(ptr, map, next_vreg), expected: r(expected, map, next_vreg), desired: r(desired, map, next_vreg), elem_width, success_order, failure_order },
            VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, seq_meta_stride } => VmInstr::SeqIdLookup { dst: r(dst, map, next_vreg), token_index: r(token_index, map, next_vreg), seq_meta_base: r(seq_meta_base, map, next_vreg), num_seqs: r(num_seqs, map, next_vreg), seq_meta_stride },
            #[cfg(feature = "nccl")]
            VmInstr::AllReduceChunk { sendbuf, recvbuf, count, dtype, op, rank, world_size, chunk_idx } => VmInstr::AllReduceChunk { sendbuf: r(sendbuf, map, next_vreg), recvbuf: r(recvbuf, map, next_vreg), count: r(count, map, next_vreg), dtype, op, rank: r(rank, map, next_vreg), world_size: r(world_size, map, next_vreg), chunk_idx: r(chunk_idx, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::CommBarrier { barrier_id, thread_count } => VmInstr::CommBarrier { barrier_id, thread_count: r(thread_count, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::NvlinkAsyncCopy { dst, src, len, lane } => VmInstr::NvlinkAsyncCopy { dst: r(dst, map, next_vreg), src: r(src, map, next_vreg), len: r(len, map, next_vreg), lane },
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageLookup { dst, seq_id, page_index, routing_table_base } => VmInstr::RemotePageLookup { dst: r(dst, map, next_vreg), seq_id: r(seq_id, map, next_vreg), page_index: r(page_index, map, next_vreg), routing_table_base: r(routing_table_base, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::P2pPageFetch { local_buf, peer_buf, page_size, barrier } => VmInstr::P2pPageFetch { local_buf: r(local_buf, map, next_vreg), peer_buf: r(peer_buf, map, next_vreg), page_size: r(page_size, map, next_vreg), barrier: r(barrier, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetch { local_buf, remote_addr, rkey, page_size, sq_desc, doorbell, cq_addr } => VmInstr::RdmaPageFetch { local_buf: r(local_buf, map, next_vreg), remote_addr: r(remote_addr, map, next_vreg), rkey: r(rkey, map, next_vreg), page_size: r(page_size, map, next_vreg), sq_desc: r(sq_desc, map, next_vreg), doorbell: r(doorbell, map, next_vreg), cq_addr: r(cq_addr, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetchCompressed { local_buf, scratch_buf, page_size, remote_addr, rkey, sq_desc, doorbell, cq_addr, quant_scheme, compress_algorithm } => VmInstr::RdmaPageFetchCompressed { local_buf: r(local_buf, map, next_vreg), scratch_buf: r(scratch_buf, map, next_vreg), page_size: r(page_size, map, next_vreg), remote_addr: r(remote_addr, map, next_vreg), rkey: r(rkey, map, next_vreg), sq_desc: r(sq_desc, map, next_vreg), doorbell: r(doorbell, map, next_vreg), cq_addr: r(cq_addr, map, next_vreg), quant_scheme, compress_algorithm },
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageAttn { q_buf, k_remote_buf, v_remote_buf, output_buf, shared_buf, barrier, tile_bytes } => VmInstr::RemotePageAttn { q_buf: r(q_buf, map, next_vreg), k_remote_buf: r(k_remote_buf, map, next_vreg), v_remote_buf: r(v_remote_buf, map, next_vreg), output_buf: r(output_buf, map, next_vreg), shared_buf: r(shared_buf, map, next_vreg), barrier: r(barrier, map, next_vreg), tile_bytes: r(tile_bytes, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationLock { dst, entry_addr } => VmInstr::PageMigrationLock { dst: r(dst, map, next_vreg), entry_addr: r(entry_addr, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationUnlock { entry_addr } => VmInstr::PageMigrationUnlock { entry_addr: r(entry_addr, map, next_vreg) },
            #[cfg(feature = "nccl")]
            VmInstr::PageLocationUpdate { entry_addr, new_location, new_state } => VmInstr::PageLocationUpdate { entry_addr: r(entry_addr, map, next_vreg), new_location: r(new_location, map, next_vreg), new_state },
        }
    }

    /// Remap VRegIds in an OffsetExpr using a substitution map.
    fn remap_offset_with_map(
        oe: OffsetExpr,
        map: &mut std::collections::HashMap<VRegId, VRegId>,
        next_vreg: &mut u32,
    ) -> OffsetExpr {
        let r = |vreg: VRegId, m: &mut std::collections::HashMap<VRegId, VRegId>, nv: &mut u32| -> VRegId {
            *m.entry(vreg).or_insert_with(|| {
                let id = VRegId(*nv);
                *nv += 1;
                id
            })
        };
        match oe {
            OffsetExpr::LoopOffset(id) => OffsetExpr::LoopOffset(r(id, map, next_vreg)),
            OffsetExpr::ScalarVReg(id) => OffsetExpr::ScalarVReg(r(id, map, next_vreg)),
            OffsetExpr::Add(a, b) => OffsetExpr::Add(
                Box::new(Self::remap_offset_with_map(*a, map, next_vreg)),
                Box::new(Self::remap_offset_with_map(*b, map, next_vreg)),
            ),
            OffsetExpr::Mul(a, scale) => OffsetExpr::Mul(
                Box::new(Self::remap_offset_with_map(*a, map, next_vreg)),
                scale,
            ),
            other => other,
        }
    }

    /// 重映射单条指令中所有 VRegId += offset。
    fn remap_vreg_instr(instr: VmInstr, offset: u32) -> VmInstr {
        let r = |id: VRegId| VRegId(id.0 + offset);
        let remap_offset = |oe: OffsetExpr| -> OffsetExpr {
            Self::remap_offset_expr(oe, offset)
        };
        let remap_scalar = |se: ScalarExpr| -> ScalarExpr {
            match se {
                ScalarExpr::Const(_) => se,
                ScalarExpr::MemLoad(base, off) => ScalarExpr::MemLoad(r(base), remap_offset(off)),
                ScalarExpr::ExtractLane0(v) => ScalarExpr::ExtractLane0(r(v)),
                ScalarExpr::VReg(v) => ScalarExpr::VReg(r(v)),
            }
        };
        let remap_ptr = |pe: PtrExpr| -> PtrExpr {
            match pe {
                PtrExpr::VRegPlusVReg(a, b) => PtrExpr::VRegPlusVReg(r(a), r(b)),
                other => other,
            }
        };
        match instr {
            VmInstr::DeclareVReg { id, kind, width } => VmInstr::DeclareVReg { id: r(id), kind, width },
            VmInstr::ReleaseVReg { id } => VmInstr::ReleaseVReg { id: r(id) },
            VmInstr::VecLoad { dst, base, offset, width, dtype, predicate } => VmInstr::VecLoad { dst: r(dst), base: r(base), offset: remap_offset(offset), width, dtype, predicate },
            VmInstr::VecStore { base, offset, src, width, dtype, predicate } => VmInstr::VecStore { base: r(base), offset: remap_offset(offset), src: r(src), width, dtype, predicate },
            VmInstr::VecNarrow { dst, src, dst_dtype, src_dtype, width } => VmInstr::VecNarrow { dst: r(dst), src: r(src), dst_dtype, src_dtype, width },
            VmInstr::VecWiden { dst, src, dst_dtype, src_dtype, width } => VmInstr::VecWiden { dst: r(dst), src: r(src), dst_dtype, src_dtype, width },
            VmInstr::Mov { dst, src, dtype } => VmInstr::Mov { dst: r(dst), src: r(src), dtype },
            VmInstr::Broadcast { dst, src, width, dtype } => VmInstr::Broadcast { dst: r(dst), src: remap_scalar(src), width, dtype },
            VmInstr::LoadPtr { dst, src } => VmInstr::LoadPtr { dst: r(dst), src: remap_ptr(src) },
            VmInstr::VecBinOp { dst, a, b, op, dtype } => VmInstr::VecBinOp { dst: r(dst), a: r(a), b: r(b), op, dtype },
            VmInstr::VecShiftImm { dst, a, amount, op, width } => VmInstr::VecShiftImm { dst: r(dst), a: r(a), amount, op, width },
            VmInstr::VecUnaryOp { dst, a, op } => VmInstr::VecUnaryOp { dst: r(dst), a: r(a), op },
            VmInstr::VecCmp { dst, a, b, pred } => VmInstr::VecCmp { dst: r(dst), a: r(a), b: r(b), pred },
            VmInstr::VecCast { dst, src, from_bits, to_bits } => VmInstr::VecCast { dst: r(dst), src: r(src), from_bits, to_bits },
            VmInstr::ConditionalSelect { dst, mask, true_val, false_val } => VmInstr::ConditionalSelect { dst: r(dst), mask: r(mask), true_val: r(true_val), false_val: r(false_val) },
            VmInstr::Fma { dst, acc, a, b, dtype } => VmInstr::Fma { dst: r(dst), acc: r(acc), a: r(a), b: r(b), dtype },
            VmInstr::HReduce { dst, src, op } => VmInstr::HReduce { dst: r(dst), src: r(src), op },
            VmInstr::Accumulate { acc, src } => VmInstr::Accumulate { acc: r(acc), src: r(src) },
            VmInstr::LoopBegin { counter, byte_offset, bound, step_bytes } => VmInstr::LoopBegin { counter: r(counter), byte_offset: r(byte_offset), bound, step_bytes },
            VmInstr::LoopEnd => VmInstr::LoopEnd,
            VmInstr::ScopeBegin { scope_id } => VmInstr::ScopeBegin { scope_id },
            VmInstr::ScopeEnd { scope_id } => VmInstr::ScopeEnd { scope_id },
            VmInstr::ConditionalSkip { mask, skip_count } => VmInstr::ConditionalSkip { mask: r(mask), skip_count },
            VmInstr::GprCondAction { cond, action } => VmInstr::GprCondAction {
                cond: cond.remap(r),
                action: action.remap(r),
            },
            VmInstr::TileConfig { rows, cols, dtype } => VmInstr::TileConfig { rows, cols, dtype },
            VmInstr::TileMma { c, a, b } => VmInstr::TileMma { c: r(c), a: r(a), b: r(b) },
            VmInstr::TileRelease => VmInstr::TileRelease,
            VmInstr::Vp2Intersect { dst_k0, dst_k1, a, b } => VmInstr::Vp2Intersect { dst_k0: r(dst_k0), dst_k1: r(dst_k1), a: r(a), b: r(b) },
            VmInstr::WarpSync => VmInstr::WarpSync,
            VmInstr::AsyncCopy { dst, src, size } => VmInstr::AsyncCopy { dst: r(dst), src: r(src), size },
            VmInstr::AsyncWait { handle } => VmInstr::AsyncWait { handle },
            VmInstr::Transcendental { dst, src, func } => VmInstr::Transcendental { dst: r(dst), src: r(src), func },
            VmInstr::HotpatchSlot { slot_id, initial_target, alternatives } => VmInstr::HotpatchSlot { slot_id, initial_target, alternatives },
            VmInstr::IndirectJump { index, targets } => VmInstr::IndirectJump { index: r(index), targets },
            VmInstr::ConditionalExit { condition, output } => VmInstr::ConditionalExit { condition: r(condition), output: r(output) },
            VmInstr::BranchIfPtrNonNull { ptr, target_label } => VmInstr::BranchIfPtrNonNull { ptr: r(ptr), target_label },
            VmInstr::BranchIfGprZero { value, target_label } => VmInstr::BranchIfGprZero { value: r(value), target_label },
            VmInstr::BranchIfGprLtU { a, b, target_label } => VmInstr::BranchIfGprLtU { a: r(a), b: r(b), target_label },
            VmInstr::UnconditionalBranch { target_label } => VmInstr::UnconditionalBranch { target_label },
            VmInstr::BatchSeqIdLookup { dst, pt_offset_out, token_index, batch_ctx_ptr } => VmInstr::BatchSeqIdLookup { dst: r(dst), pt_offset_out: r(pt_offset_out), token_index: r(token_index), batch_ctx_ptr: r(batch_ctx_ptr) },
            VmInstr::BatchPerSeqArgmax { dst, seq_id, logits_flat_ptr, vocab_size, width } => VmInstr::BatchPerSeqArgmax { dst: r(dst), seq_id: r(seq_id), logits_flat_ptr: r(logits_flat_ptr), vocab_size, width },
            VmInstr::BatchPerSeqStopCheck { seq_id, token_id, batch_ctx_ptr } => VmInstr::BatchPerSeqStopCheck { seq_id: r(seq_id), token_id: r(token_id), batch_ctx_ptr: r(batch_ctx_ptr) },
            VmInstr::AtomicAdd { base, offset, value, elem_width } => VmInstr::AtomicAdd { base: r(base), offset: remap_offset(offset), value, elem_width },
            VmInstr::MemFence { order } => VmInstr::MemFence { order },
            VmInstr::Argmax { dst, logits_ptr, vocab_bytes, width } => VmInstr::Argmax { dst: r(dst), logits_ptr: r(logits_ptr), vocab_bytes, width },
            VmInstr::TemperatureScale { logits_ptr, temp_ptr, vocab_bytes, width } => VmInstr::TemperatureScale { logits_ptr: r(logits_ptr), temp_ptr: r(temp_ptr), vocab_bytes, width },
            VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => VmInstr::StoreToken { token_id: r(token_id), output_buf: r(output_buf), counter: r(counter), input_ids_ptr: r(input_ids_ptr), prompt_len_bytes: r(prompt_len_bytes) },
            VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => VmInstr::CheckStopCondition { token_id: r(token_id), counter: r(counter), eos_ptr: r(eos_ptr), max_tokens_ptr: r(max_tokens_ptr) },
            VmInstr::AddPtr { dst, base, offset } => VmInstr::AddPtr { dst: r(dst), base: r(base), offset },
            VmInstr::StoreConstToStack { rbp_offset, value, elem_width } => VmInstr::StoreConstToStack { rbp_offset, value, elem_width },
            VmInstr::BreakLoop { return_value } => VmInstr::BreakLoop {
                return_value: match return_value {
                    ReturnValue::Const(v) => ReturnValue::Const(v),
                    ReturnValue::VReg(v) => ReturnValue::VReg(r(v)),
                },
            },
            VmInstr::MarkLabel { label_id } => VmInstr::MarkLabel { label_id },
            VmInstr::GprBinOp { dst, a, b, op } => VmInstr::GprBinOp {
                dst: r(dst), a: r(a),
                b: b.remap(r), op,
            },
            VmInstr::GprUnaryOp { dst, src, op } => VmInstr::GprUnaryOp { dst: r(dst), src: r(src), op },
            VmInstr::GprLoadImm { dst, value } => VmInstr::GprLoadImm { dst: r(dst), value },
            VmInstr::LoadCallbackEntry { table_ptr, slot_id, fn_ptr_out, ctx_out } => VmInstr::LoadCallbackEntry { table_ptr: r(table_ptr), slot_id, fn_ptr_out: r(fn_ptr_out), ctx_out: r(ctx_out) },
            VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr } => VmInstr::NativeCall { ret_val: r(ret_val), fn_ptr: r(fn_ptr), ctx_ptr: r(ctx_ptr) },
            VmInstr::ScalarLoad { dst, base, offset } => VmInstr::ScalarLoad { dst: r(dst), base: r(base), offset: remap_offset(offset) },
            VmInstr::ScalarStore { base, src, offset } => VmInstr::ScalarStore { base: r(base), src: r(src), offset: remap_offset(offset) },
            VmInstr::VecScalarStore { base, src, offset } => VmInstr::VecScalarStore { base: r(base), src: r(src), offset: remap_offset(offset) },
            VmInstr::ScalarToIndex { dst, src, stride } => VmInstr::ScalarToIndex { dst: r(dst), src: r(src), stride },
            VmInstr::IndexToScalar { dst, src } => VmInstr::IndexToScalar { dst: r(dst), src: r(src) },
            VmInstr::IntMulStride { dst, src, stride } => VmInstr::IntMulStride { dst: r(dst), src: r(src), stride },
            VmInstr::ScalarByteLoad { dst, base, offset } => VmInstr::ScalarByteLoad { dst: r(dst), base: r(base), offset: remap_offset(offset) },
            VmInstr::Prefetch { base, offset, distance, hint } => VmInstr::Prefetch { base: r(base), offset: remap_offset(offset), distance, hint },
            VmInstr::GatherLoad { dst, base, indices, stride, width, dtype, predicate } => VmInstr::GatherLoad { dst: r(dst), base: r(base), indices: r(indices), stride, width, dtype, predicate },
            VmInstr::ScatterStore { base, indices, src, stride, width, dtype, predicate } => VmInstr::ScatterStore { base: r(base), indices: r(indices), src: r(src), stride, width, dtype, predicate },
            VmInstr::TableLookup { dst, base, row_index, row_bytes, width } => VmInstr::TableLookup { dst: r(dst), base: r(base), row_index: r(row_index), row_bytes, width },
            VmInstr::ActivationSwap { ptr_a, ptr_b } => VmInstr::ActivationSwap { ptr_a: r(ptr_a), ptr_b: r(ptr_b) },
            VmInstr::PageTableAddr { dst, pool_base, page_table_ptr, ki_byte_off, row_bytes, page_size, page_stride, base_offset, seq_pt_offset } => VmInstr::PageTableAddr { dst: r(dst), pool_base: r(pool_base), page_table_ptr: r(page_table_ptr), ki_byte_off: remap_offset(ki_byte_off), row_bytes, page_size, page_stride, base_offset, seq_pt_offset },
            VmInstr::PageTableKVWrite { src, pool_base, page_table_ptr, seq_index, row_bytes, page_size, page_stride, base_offset, width, dtype } => VmInstr::PageTableKVWrite { src: r(src), pool_base: r(pool_base), page_table_ptr: r(page_table_ptr), seq_index: r(seq_index), row_bytes, page_size, page_stride, base_offset, width, dtype },
            VmInstr::PageTableKVWriteQuant { src, pool_base, page_table_ptr, seq_index, quant_row_bytes, fp32_row_bytes, page_size, page_stride, base_offset, scale_offset, width, kivi_mode, num_elems } => VmInstr::PageTableKVWriteQuant { src: r(src), pool_base: r(pool_base), page_table_ptr: r(page_table_ptr), seq_index: r(seq_index), quant_row_bytes, fp32_row_bytes, page_size, page_stride, base_offset, scale_offset, width, kivi_mode, num_elems },
            VmInstr::KiviQuantChannel { src, dst_ptr, scale_ptr, num_channels, width } => VmInstr::KiviQuantChannel { src: r(src), dst_ptr: r(dst_ptr), scale_ptr: r(scale_ptr), num_channels, width },
            VmInstr::KiviQuantToken { src, dst_ptr, scale_ptr, num_elems, width } => VmInstr::KiviQuantToken { src: r(src), dst_ptr: r(dst_ptr), scale_ptr: r(scale_ptr), num_elems, width },
            VmInstr::KiviDequantLoad { dst, src_ptr, scale_ptr, num_elems, width } => VmInstr::KiviDequantLoad { dst: r(dst), src_ptr: r(src_ptr), scale_ptr: r(scale_ptr), num_elems, width },
            VmInstr::SharedMemAlloc { name, bytes } => VmInstr::SharedMemAlloc { name, bytes },
            VmInstr::SharedMemStore { name, dst_offset, src, width, dtype } => VmInstr::SharedMemStore { name, dst_offset: remap_offset(dst_offset), src: r(src), width, dtype },
            VmInstr::SharedMemLoad { dst, name, src_offset, width, dtype } => VmInstr::SharedMemLoad { dst: r(dst), name, src_offset: remap_offset(src_offset), width, dtype },
            VmInstr::SharedMemAsyncStore { name, dst_offset, src, width, dtype } => VmInstr::SharedMemAsyncStore { name, dst_offset: remap_offset(dst_offset), src: r(src), width, dtype },
            VmInstr::SharedMemAsyncWaitGroup { n } => VmInstr::SharedMemAsyncWaitGroup { n },
            VmInstr::WeightPrefetchAsync { smem_name, weight_base, weight_offset, size } => VmInstr::WeightPrefetchAsync { smem_name, weight_base: r(weight_base), weight_offset, size },
            VmInstr::WeightPrefetchWait { group } => VmInstr::WeightPrefetchWait { group },
            VmInstr::WarpRoleDeclare { role } => VmInstr::WarpRoleDeclare { role },
            VmInstr::WarpBarrierArrive { barrier_name, tx_bytes } => VmInstr::WarpBarrierArrive { barrier_name, tx_bytes },
            VmInstr::WarpBarrierWait { barrier_name, parity } => VmInstr::WarpBarrierWait { barrier_name, parity },
            VmInstr::TmaDescriptorInit { desc_name, global_dim, global_stride, box_dim, swizzle, dtype } => VmInstr::TmaDescriptorInit { desc_name, global_dim, global_stride, box_dim, swizzle, dtype },
            VmInstr::Tma2DCopy { desc_name, smem_name, coord_x, coord_y, barrier_name } => VmInstr::Tma2DCopy { desc_name, smem_name, coord_x: r(coord_x), coord_y: r(coord_y), barrier_name },
            VmInstr::BarrierInit { name, thread_count } => VmInstr::BarrierInit { name, thread_count },
            VmInstr::BlockSync => VmInstr::BlockSync,
            VmInstr::WarpReduce { op, src, dst, width } => VmInstr::WarpReduce { op, src: r(src), dst: r(dst), width },
            // Quant* decode instrs
            VmInstr::QuantBroadcastInt { dst, value, width } => VmInstr::QuantBroadcastInt { dst: r(dst), value, width },
            VmInstr::QuantScalarCvtLoad { dst, base, offset, src_dtype, width } => VmInstr::QuantScalarCvtLoad { dst: r(dst), base: r(base), offset, src_dtype, width },
            VmInstr::QuantBlockLoad { dst, base, offset, unpack, width } => VmInstr::QuantBlockLoad { dst: r(dst), base: r(base), offset, unpack: unpack.remap_vregs(&r), width },
            VmInstr::QuantBiPlaneLoad { dst, qs_base, extra_base, bias, mode, width } => VmInstr::QuantBiPlaneLoad { dst: r(dst), qs_base: r(qs_base), extra_base: r(extra_base), bias, mode, width },
            VmInstr::QuantLoadBytesVec { dst, base, offset, count, signed, width } => VmInstr::QuantLoadBytesVec { dst: r(dst), base: r(base), offset, count, signed, width },
            VmInstr::QuantCodebookLookup { dst, indices, codebook_data, vector_size, bits_per_entry, width } => VmInstr::QuantCodebookLookup { dst: r(dst), indices: r(indices), codebook_data, vector_size, bits_per_entry, width },
            VmInstr::QuantExtractBits { dst, src, bit_offset, bit_width, width } => VmInstr::QuantExtractBits { dst: r(dst), src: r(src), bit_offset, bit_width, width },            VmInstr::QuantDequantFma { dst, weight, activation, scale, zero_point, quant_kind, dtype, width } => VmInstr::QuantDequantFma { dst: r(dst), weight: r(weight), activation: r(activation), scale: r(scale), zero_point: r(zero_point), quant_kind, dtype, width },            VmInstr::QuantInterleave { dst, lo, hi, width } => VmInstr::QuantInterleave { dst: r(dst), lo: r(lo), hi: r(hi), width },            VmInstr::QuantConcatSeq { dst, lo, hi, width } => VmInstr::QuantConcatSeq { dst: r(dst), lo: r(lo), hi: r(hi), width },            VmInstr::Q3KDecodeStep { dst, block_base, lane_offset, d_vreg, qs_offset, hmask_offset, lanes, width } => VmInstr::Q3KDecodeStep { dst: r(dst), block_base: r(block_base), lane_offset: r(lane_offset), d_vreg: r(d_vreg), qs_offset, hmask_offset, lanes, width },            VmInstr::DotProduct { acc, a, b, input_dtype, width } => VmInstr::DotProduct { acc: r(acc), a: r(a), b: r(b), input_dtype, width },
            VmInstr::ScaleApply { dst, acc, scale, zero, input_dtype, width } => VmInstr::ScaleApply { dst: r(dst), acc: r(acc), scale: r(scale), zero: r(zero), input_dtype, width },
            VmInstr::BitwiseGemm { dst, sign_bits, input_sign_bits, scale, width } => VmInstr::BitwiseGemm { dst: r(dst), sign_bits: r(sign_bits), input_sign_bits: r(input_sign_bits), scale: r(scale), width },
            VmInstr::SparseGemm { acc, a_sparse, b_dense, sparse_mask_ptr, m, n, k, width, dtype } => VmInstr::SparseGemm { acc: r(acc), a_sparse: r(a_sparse), b_dense: r(b_dense), sparse_mask_ptr: r(sparse_mask_ptr), m, n, k, width, dtype },
            VmInstr::NativeFp4Gemm { acc, a, b, scale_a, scale_b, m, n, k, width } => VmInstr::NativeFp4Gemm { acc: r(acc), a: r(a), b: r(b), scale_a: r(scale_a), scale_b: r(scale_b), m, n, k, width },
            VmInstr::SparseFp8Gemm { acc, a_sparse, b_dense, sparse_mask_ptr, m, n, k, width, fp8_kind } => VmInstr::SparseFp8Gemm { acc: r(acc), a_sparse: r(a_sparse), b_dense: r(b_dense), sparse_mask_ptr: r(sparse_mask_ptr), m, n, k, width, fp8_kind },
            VmInstr::NativeFp8Gemm { acc, a, b, m, n, k, width, fp8_kind } => VmInstr::NativeFp8Gemm { acc: r(acc), a: r(a), b: r(b), m, n, k, width, fp8_kind },
            VmInstr::HwQuantDequant { dst, packed_weight, block_scale, global_scale, quant_kind, count, width } => VmInstr::HwQuantDequant { dst: r(dst), packed_weight: r(packed_weight), block_scale: r(block_scale), global_scale: r(global_scale), quant_kind, count, width },
            VmInstr::TmemAlloc { name, bytes } => VmInstr::TmemAlloc { name, bytes },
            VmInstr::TmemLoad { dst, name, offset, width, dtype } => VmInstr::TmemLoad { dst: r(dst), name, offset: remap_offset(offset), width, dtype },
            VmInstr::TmemStore { name, offset, src, width, dtype } => VmInstr::TmemStore { name, offset: remap_offset(offset), src: r(src), width, dtype },
            VmInstr::TmemDealloc { name } => VmInstr::TmemDealloc { name },
            VmInstr::ClusterBarrierInit { name, thread_count } => VmInstr::ClusterBarrierInit { name, thread_count },
            VmInstr::ClusterStore { name, offset, src, width, dtype } => VmInstr::ClusterStore { name, offset: remap_offset(offset), src: r(src), width, dtype },
            VmInstr::ClusterLoad { dst, name, offset, width, dtype } => VmInstr::ClusterLoad { dst: r(dst), name, offset: remap_offset(offset), width, dtype },
            VmInstr::Comment(s) => VmInstr::Comment(s),
            VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, decompressed_size } => VmInstr::Lz4Decode { src_ptr: r(src_ptr), dst_ptr: r(dst_ptr), compressed_size: r(compressed_size), decompressed_size },
            VmInstr::BitPackRleDecode { src_ptr, dst_ptr, compressed_size, nibble_bits, element_count } => VmInstr::BitPackRleDecode { src_ptr: r(src_ptr), dst_ptr: r(dst_ptr), compressed_size: r(compressed_size), nibble_bits, element_count },
            VmInstr::GgufSubScaleLoad { dst, scales_base, sub_block_idx, width } => VmInstr::GgufSubScaleLoad { dst: r(dst), scales_base: r(scales_base), sub_block_idx: r(sub_block_idx), width },
            VmInstr::GgufKQuantScaleLoad { dst, scales_base, sub_block_idx, scales_count, is_q3k_extended, is_min, width } => VmInstr::GgufKQuantScaleLoad { dst: r(dst), scales_base: r(scales_base), sub_block_idx: r(sub_block_idx), scales_count, is_q3k_extended, is_min, width },
            // GPU sampling instrs
            VmInstr::SoftmaxReduceMax { dst, logits_ptr, vocab_bytes, width } => VmInstr::SoftmaxReduceMax { dst: r(dst), logits_ptr: r(logits_ptr), vocab_bytes, width },
            VmInstr::SoftmaxExpSum { sum_dst, logits_ptr, max_val, vocab_bytes, width } => VmInstr::SoftmaxExpSum { sum_dst: r(sum_dst), logits_ptr: r(logits_ptr), max_val: r(max_val), vocab_bytes, width },
            VmInstr::SoftmaxNormalize { logits_ptr, sum_val, vocab_bytes, width } => VmInstr::SoftmaxNormalize { logits_ptr: r(logits_ptr), sum_val: r(sum_val), vocab_bytes, width },
            VmInstr::SampleTopKFilter { probs_ptr, indices_ptr, k_ptr, vocab_bytes, width } => VmInstr::SampleTopKFilter { probs_ptr: r(probs_ptr), indices_ptr: r(indices_ptr), k_ptr: r(k_ptr), vocab_bytes, width },
            VmInstr::SampleTopPFilter { probs_ptr, p_ptr, vocab_bytes, width } => VmInstr::SampleTopPFilter { probs_ptr: r(probs_ptr), p_ptr: r(p_ptr), vocab_bytes, width },
            VmInstr::SampleMultinomial { dst, probs_ptr, rng_state_ptr, vocab_bytes, width } => VmInstr::SampleMultinomial { dst: r(dst), probs_ptr: r(probs_ptr), rng_state_ptr: r(rng_state_ptr), vocab_bytes, width },
            VmInstr::WarpPRNG { dst, rng_state_ptr } => VmInstr::WarpPRNG { dst: r(dst), rng_state_ptr: r(rng_state_ptr) },
            VmInstr::SharedMemSwizzle { dst, raw_addr, log2_banks, log2_bank_width } => VmInstr::SharedMemSwizzle { dst: r(dst), raw_addr: r(raw_addr), log2_banks, log2_bank_width },
            VmInstr::DebugBreakpoint { label } => VmInstr::DebugBreakpoint { label: label.clone() },
            VmInstr::DebugMarker { message } => VmInstr::DebugMarker { message: message.clone() },
            VmInstr::DebugProbe { vreg, probe_name, width } => VmInstr::DebugProbe { vreg: r(vreg), probe_name: probe_name.clone(), width },
            VmInstr::DebugBreakIf { label, cond_gpr } => VmInstr::DebugBreakIf { label: label.clone(), cond_gpr: r(cond_gpr) },
            VmInstr::MemCopy { dst, src, bytes, dtype, guard, effect } => VmInstr::MemCopy { dst: r(dst), src: r(src), bytes, dtype, guard, effect },
            VmInstr::VecShuffle { dst, src, mask, width } => VmInstr::VecShuffle { dst: r(dst), src: r(src), mask: mask.remap(&|v| r(v)), width },
            VmInstr::VecExtractLane { dst, src, lane, dtype } => VmInstr::VecExtractLane { dst: r(dst), src: r(src), lane, dtype },
            VmInstr::VecInsertLane { dst, src_vec, src_scalar, lane, dtype } => VmInstr::VecInsertLane { dst: r(dst), src_vec: r(src_vec), src_scalar: r(src_scalar), lane, dtype },
            VmInstr::VecLoadConst { dst, values, dtype, width } => VmInstr::VecLoadConst { dst: r(dst), values, dtype, width },
            VmInstr::AtomicCAS { dst, ptr, expected, desired, elem_width, success_order, failure_order } => VmInstr::AtomicCAS { dst: r(dst), ptr: r(ptr), expected: r(expected), desired: r(desired), elem_width, success_order, failure_order },
            VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, seq_meta_stride } => VmInstr::SeqIdLookup { dst: r(dst), token_index: r(token_index), seq_meta_base: r(seq_meta_base), num_seqs: r(num_seqs), seq_meta_stride },
            #[cfg(feature = "nccl")]
            VmInstr::AllReduceChunk { sendbuf, recvbuf, count, dtype, op, rank, world_size, chunk_idx } => VmInstr::AllReduceChunk { sendbuf: r(sendbuf), recvbuf: r(recvbuf), count: r(count), dtype, op, rank: r(rank), world_size: r(world_size), chunk_idx: r(chunk_idx) },
            #[cfg(feature = "nccl")]
            VmInstr::CommBarrier { barrier_id, thread_count } => VmInstr::CommBarrier { barrier_id, thread_count: r(thread_count) },
            #[cfg(feature = "nccl")]
            VmInstr::NvlinkAsyncCopy { dst, src, len, lane } => VmInstr::NvlinkAsyncCopy { dst: r(dst), src: r(src), len: r(len), lane },
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageLookup { dst, seq_id, page_index, routing_table_base } => VmInstr::RemotePageLookup { dst: r(dst), seq_id: r(seq_id), page_index: r(page_index), routing_table_base: r(routing_table_base) },
            #[cfg(feature = "nccl")]
            VmInstr::P2pPageFetch { local_buf, peer_buf, page_size, barrier } => VmInstr::P2pPageFetch { local_buf: r(local_buf), peer_buf: r(peer_buf), page_size: r(page_size), barrier: r(barrier) },
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetch { local_buf, remote_addr, rkey, page_size, sq_desc, doorbell, cq_addr } => VmInstr::RdmaPageFetch { local_buf: r(local_buf), remote_addr: r(remote_addr), rkey: r(rkey), page_size: r(page_size), sq_desc: r(sq_desc), doorbell: r(doorbell), cq_addr: r(cq_addr) },
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetchCompressed { local_buf, scratch_buf, page_size, remote_addr, rkey, sq_desc, doorbell, cq_addr, quant_scheme, compress_algorithm } => VmInstr::RdmaPageFetchCompressed { local_buf: r(local_buf), scratch_buf: r(scratch_buf), page_size: r(page_size), remote_addr: r(remote_addr), rkey: r(rkey), sq_desc: r(sq_desc), doorbell: r(doorbell), cq_addr: r(cq_addr), quant_scheme, compress_algorithm },
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageAttn { q_buf, k_remote_buf, v_remote_buf, output_buf, shared_buf, barrier, tile_bytes } => VmInstr::RemotePageAttn { q_buf: r(q_buf), k_remote_buf: r(k_remote_buf), v_remote_buf: r(v_remote_buf), output_buf: r(output_buf), shared_buf: r(shared_buf), barrier: r(barrier), tile_bytes: r(tile_bytes) },
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationLock { dst, entry_addr } => VmInstr::PageMigrationLock { dst: r(dst), entry_addr: r(entry_addr) },
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationUnlock { entry_addr } => VmInstr::PageMigrationUnlock { entry_addr: r(entry_addr) },
            #[cfg(feature = "nccl")]
            VmInstr::PageLocationUpdate { entry_addr, new_location, new_state } => VmInstr::PageLocationUpdate { entry_addr: r(entry_addr), new_location: r(new_location), new_state },
        }
    }

    fn remap_offset_expr(oe: OffsetExpr, offset: u32) -> OffsetExpr {
        let r = |id: VRegId| VRegId(id.0 + offset);
        match oe {
            OffsetExpr::LoopOffset(id) => OffsetExpr::LoopOffset(r(id)),
            OffsetExpr::ScalarVReg(id) => OffsetExpr::ScalarVReg(r(id)),
            OffsetExpr::Add(a, b) => OffsetExpr::Add(
                Box::new(Self::remap_offset_expr(*a, offset)),
                Box::new(Self::remap_offset_expr(*b, offset)),
            ),
            OffsetExpr::Mul(a, scale) => OffsetExpr::Mul(
                Box::new(Self::remap_offset_expr(*a, offset)),
                scale,
            ),
            other => other,
        }
    }

    /// §14.1 验证 provenance: 每个计算 VmInstr 必须有对应的 VReg 声明。
    /// 确保没有"凭空出现"的 VReg——所有寄存器都从 alloc_vreg 分配。
    pub fn validate_provenance(&self) -> Result<(), String> {
        let mut declared: std::collections::HashSet<VRegId> = std::collections::HashSet::new();
        for instr in &self.instrs {
            if let VmInstr::DeclareVReg { id, .. } = instr {
                declared.insert(*id);
            }
            // 检查所有引用的 VReg 都已声明
            for vreg in Self::referenced_vregs(instr) {
                if !declared.contains(&vreg) {
                    return Err(format!("VReg v{} used without DeclareVReg", vreg.0));
                }
            }
        }
        Ok(())
    }

    fn referenced_vregs(instr: &VmInstr) -> Vec<VRegId> {
        match instr {
            VmInstr::VecLoad { dst, base, .. } => vec![*dst, *base],
            VmInstr::VecStore { base, src, .. } => vec![*base, *src],
            VmInstr::VecNarrow { dst, src, .. } => vec![*dst, *src],
            VmInstr::VecWiden { dst, src, .. } => vec![*dst, *src],
            VmInstr::Mov { dst, src, .. } => vec![*dst, *src],
            VmInstr::VecBinOp { dst, a, b, .. } => vec![*dst, *a, *b],
            VmInstr::VecShiftImm { dst, a, .. } => vec![*dst, *a],
            VmInstr::VecUnaryOp { dst, a, .. } => vec![*dst, *a],
            VmInstr::VecCmp { dst, a, b, .. } => vec![*dst, *a, *b],
            VmInstr::VecCast { dst, src, .. } => vec![*dst, *src],
            VmInstr::ConditionalSelect { dst, mask, true_val, false_val } => vec![*dst, *mask, *true_val, *false_val],
            VmInstr::Fma { dst, acc, a, b, ..} => vec![*dst, *acc, *a, *b],
            VmInstr::HReduce { dst, src, .. } => vec![*dst, *src],
            VmInstr::Accumulate { acc, src } => vec![*acc, *src],
            VmInstr::Transcendental { dst, src, .. } => vec![*dst, *src],
            VmInstr::Prefetch { base, .. } => vec![*base],
            VmInstr::Vp2Intersect { dst_k0, dst_k1, a, b } => vec![*dst_k0, *dst_k1, *a, *b],
            VmInstr::LoadPtr { dst, .. } => vec![*dst],
            VmInstr::AtomicAdd { base, .. } => vec![*base],
            VmInstr::MemFence { .. } => vec![],
            VmInstr::Argmax { dst, logits_ptr, .. } => vec![*dst, *logits_ptr],
            VmInstr::TemperatureScale { logits_ptr, temp_ptr, .. } => vec![*logits_ptr, *temp_ptr],
            VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => vec![*token_id, *output_buf, *counter, *input_ids_ptr, *prompt_len_bytes],
            VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => vec![*token_id, *counter, *eos_ptr, *max_tokens_ptr],
            VmInstr::AddPtr { dst, base, .. } => vec![*dst, *base],
            VmInstr::StoreConstToStack { .. } => vec![],
            VmInstr::BreakLoop { return_value } => match return_value {
                ReturnValue::Const(_) => vec![],
                ReturnValue::VReg(v) => vec![*v],
            },
            VmInstr::MarkLabel { .. } => vec![],
            VmInstr::GprBinOp { dst, a, b, .. } => {
                let mut v = vec![*dst, *a];
                if let GprOperand::VReg(vr) = b { v.push(*vr); }
                v
            }
            VmInstr::GprUnaryOp { dst, src, .. } => vec![*dst, *src],
            VmInstr::GprLoadImm { dst, .. } => vec![*dst],
            VmInstr::GprCondAction { cond, action } => {
                let mut v = cond.vregs();
                v.extend(action.vregs());
                v
            }
            VmInstr::ScalarLoad { dst, base, .. } => vec![*dst, *base],
            VmInstr::ScalarStore { base, src, .. } => vec![*base, *src],
            VmInstr::VecScalarStore { base, src, .. } => vec![*base, *src],
            VmInstr::IndexToScalar { dst, src } => vec![*dst, *src],
            VmInstr::IntMulStride { dst, src, .. } => vec![*dst, *src],
            VmInstr::ScalarByteLoad { dst, base, .. } => vec![*dst, *base],
            VmInstr::Broadcast { dst, src, .. } => {
                let mut v = vec![*dst];
                if let ScalarExpr::MemLoad(b, _) = src { v.push(*b); }
                if let ScalarExpr::ExtractLane0(s) = src { v.push(*s); }
                if let ScalarExpr::VReg(s) = src { v.push(*s); }
                v
            }
            VmInstr::GatherLoad { dst, base, indices, .. } => vec![*dst, *base, *indices],
            VmInstr::ScatterStore { base, indices, src, .. } => vec![*base, *indices, *src],
            VmInstr::TableLookup { dst, base, row_index, .. } => vec![*dst, *base, *row_index],
            VmInstr::PageTableAddr { dst, pool_base, page_table_ptr, .. } => vec![*dst, *pool_base, *page_table_ptr],
            VmInstr::PageTableKVWrite { src, pool_base, page_table_ptr, seq_index, .. } => vec![*src, *pool_base, *page_table_ptr, *seq_index],
            VmInstr::PageTableKVWriteQuant { src, pool_base, page_table_ptr, seq_index, .. } => vec![*src, *pool_base, *page_table_ptr, *seq_index],
            VmInstr::KiviQuantChannel { src, dst_ptr, scale_ptr, .. } => vec![*src, *dst_ptr, *scale_ptr],
            VmInstr::KiviQuantToken { src, dst_ptr, scale_ptr, .. } => vec![*src, *dst_ptr, *scale_ptr],
            VmInstr::KiviDequantLoad { dst, src_ptr, scale_ptr, .. } => vec![*dst, *src_ptr, *scale_ptr],
            VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, .. } => vec![*src_ptr, *dst_ptr, *compressed_size],
            VmInstr::BitPackRleDecode { src_ptr, dst_ptr, compressed_size, .. } => vec![*src_ptr, *dst_ptr, *compressed_size],
            VmInstr::WeightPrefetchAsync { weight_base, .. } => vec![*weight_base],
            VmInstr::WeightPrefetchWait { .. } => vec![],
            VmInstr::TmaDescriptorInit { .. } => vec![],
            VmInstr::Tma2DCopy { coord_x, coord_y, .. } => vec![*coord_x, *coord_y],
            VmInstr::BarrierInit { .. } => vec![],
            VmInstr::QuantDequantFma { dst, weight, activation, scale, zero_point, .. } => vec![*dst, *weight, *activation, *scale, *zero_point],
            VmInstr::BitwiseGemm { dst, sign_bits, input_sign_bits, scale, .. } => vec![*dst, *sign_bits, *input_sign_bits, *scale],
            VmInstr::SparseGemm { acc, a_sparse, b_dense, sparse_mask_ptr, .. } => vec![*acc, *a_sparse, *b_dense, *sparse_mask_ptr],
            VmInstr::SparseFp8Gemm { acc, a_sparse, b_dense, sparse_mask_ptr, .. } => vec![*acc, *a_sparse, *b_dense, *sparse_mask_ptr],
            VmInstr::NativeFp4Gemm { acc, a, b, scale_a, scale_b, .. } => vec![*acc, *a, *b, *scale_a, *scale_b],
            VmInstr::NativeFp8Gemm { acc, a, b, .. } => vec![*acc, *a, *b],
            VmInstr::HwQuantDequant { dst, packed_weight, block_scale, global_scale, .. } => vec![*dst, *packed_weight, *block_scale, *global_scale],
            VmInstr::TmemAlloc { .. } => vec![],
            VmInstr::TmemLoad { dst, .. } => vec![*dst],
            VmInstr::TmemStore { src, .. } => vec![*src],
            VmInstr::TmemDealloc { .. } => vec![],
            VmInstr::ClusterBarrierInit { .. } => vec![],
            VmInstr::ClusterStore { src, .. } => vec![*src],
            VmInstr::ClusterLoad { dst, .. } => vec![*dst],
            VmInstr::DotProduct { acc, a, b, .. } => vec![*acc, *a, *b],
            VmInstr::DebugProbe { vreg, .. } => vec![*vreg],
            VmInstr::DebugBreakIf { cond_gpr, .. } => vec![*cond_gpr],
            VmInstr::DebugBreakpoint { .. } | VmInstr::DebugMarker { .. } => vec![],
            VmInstr::VecShuffle { dst, src, mask, .. } => {
                let mut v = vec![*dst, *src];
                if let VecShuffleMask::Dynamic { ctrl } = mask { v.push(*ctrl); }
                v
            }
            VmInstr::VecExtractLane { dst, src, .. } => vec![*dst, *src],
            VmInstr::VecInsertLane { dst, src_vec, src_scalar, .. } => vec![*dst, *src_vec, *src_scalar],
            VmInstr::VecLoadConst { dst, .. } => vec![*dst],
            VmInstr::AtomicCAS { dst, ptr, expected, desired, .. } => vec![*dst, *ptr, *expected, *desired],
            _ => vec![],
        }
    }

    /// §14.1 验证结构: 循环/作用域平衡。
    pub fn validate_structure(&self) -> Result<(), String> {
        let mut loop_depth = 0i32;
        let mut scope_depth = 0i32;

        for (i, instr) in self.instrs.iter().enumerate() {
            match instr {
                VmInstr::LoopBegin { .. } => loop_depth += 1,
                VmInstr::LoopEnd => {
                    loop_depth -= 1;
                    if loop_depth < 0 {
                        return Err(format!("instr[{i}]: LoopEnd without matching LoopBegin"));
                    }
                }
                VmInstr::ScopeBegin { .. } => scope_depth += 1,
                VmInstr::ScopeEnd { .. } => {
                    scope_depth -= 1;
                    if scope_depth < 0 {
                        return Err(format!("instr[{i}]: ScopeEnd without matching ScopeBegin"));
                    }
                }
                _ => {}
            }
        }

        if loop_depth != 0 {
            return Err(format!("unbalanced loops: depth={loop_depth} at end"));
        }
        if scope_depth != 0 {
            return Err(format!("unbalanced scopes: depth={scope_depth} at end"));
        }

        Ok(())
    }

    /// 验证每个 VRegId 的 DeclareVReg 出现在所有 use 之前。
    ///
    /// ARCH-VREG-DECLARE-BEFORE-USE (opt pass 不变式): `alloc_vreg` 把 DeclareVReg
    /// 追加到末尾, 但某些 opt pass 在指令流中段 insert 指令。若 pass 先 alloc_vreg
    /// 再 insert(mid_pos, <use of new vreg>), 则 declare 在 use 之后 → RegAllocator
    /// Pass 1 的 def_point 计算错误 (map.entry.or_insert 用 declare 位置覆盖), 导致
    /// interval 不覆盖中段 use → 物理寄存器冲突检测失败 → 灾难性误分配。
    ///
    /// 在 `compile_layer` 的 opt pass 之后调用此 validator 可立即捕获违反不变式
    /// 的 pass (不必等到运行时产生错误数值才发现)。
    pub fn validate_declares_before_uses(&self) -> Result<(), String> {
        use std::collections::HashSet;
        let mut declared: HashSet<VRegId> = HashSet::new();
        for (i, instr) in self.instrs.iter().enumerate() {
            let refs: Vec<VRegId> = super::reg_alloc::RegAllocator::referenced_vregs(instr);
            // DeclareVReg 自身: 本指令同时 declare 该 VReg, declare 生效于"本指令之前"。
            if let VmInstr::DeclareVReg { id, .. } = instr {
                declared.insert(*id);
                continue;
            }
            for vreg in refs {
                if !declared.contains(&vreg) {
                    return Err(format!(
                        "instr[{i}] ({:?}) uses v{} before it is declared (ARCH-VREG-DECLARE-BEFORE-USE)",
                        instr, vreg.0,
                    ));
                }
            }
        }
        Ok(())
    }

    /// D2: 操作数类型一致性验证 (ARCH-VM-TYPE-CHECK)。
    ///
    /// 检查每条 VmInstr 的操作数 VRegKind 是否匹配指令语义约束:
    /// - VecLoad/VecStore base 必须是 GPR 类 (Ptr/Scalar/Counter/ByteOffset)
    /// - VecLoad/VecStore dst/src 必须是 Vec 类
    /// - Fma 四个操作数都必须是 Vec 类
    /// - LoadPtr dst 必须是 GPR 类
    /// - Argmax logits_ptr 必须是 GPR 类, dst 必须是 Scalar
    /// - AddPtr dst/base 必须是 Ptr 类
    /// - LoopBegin counter 必须是 Counter, byte_offset 必须是 ByteOffset
    ///
    /// GPR 类 (Ptr/Scalar/Counter/ByteOffset) 之间互通但产生 WARNING;
    /// Vec 和 GPR 类之间绝不互通 → 编译错误。
    pub fn validate_type_consistency(&self) -> Result<(), String> {
        let kinds = self.collect_vreg_kinds();
        let instr_brief = |instr: &VmInstr| -> String {
            match instr {
                VmInstr::VecLoad { dst, base, .. } => format!("VecLoad dst=v{} base=v{}", dst.0, base.0),
                VmInstr::VecStore { base, src, .. } => format!("VecStore base=v{} src=v{}", base.0, src.0),
                VmInstr::VecNarrow { dst, src, .. } => format!("VecNarrow dst=v{} src=v{}", dst.0, src.0),
                VmInstr::VecWiden { dst, src, .. } => format!("VecWiden dst=v{} src=v{}", dst.0, src.0),
                VmInstr::Mov { dst, src, .. } => format!("Mov dst=v{} src=v{}", dst.0, src.0),
                VmInstr::Fma { dst, acc, a, b, .. } => format!("Fma dst=v{} acc=v{} a=v{} b=v{}", dst.0, acc.0, a.0, b.0),
                VmInstr::LoadPtr { dst, .. } => format!("LoadPtr dst=v{}", dst.0),
                VmInstr::Argmax { dst, logits_ptr, .. } => format!("Argmax dst=v{} logits_ptr=v{}", dst.0, logits_ptr.0),
                VmInstr::LoopBegin { counter, byte_offset, .. } => format!("LoopBegin counter=v{} byte_offset=v{}", counter.0, byte_offset.0),
                VmInstr::AddPtr { dst, base, .. } => format!("AddPtr dst=v{} base=v{}", dst.0, base.0),
                VmInstr::IndexToScalar { dst, src } => format!("IndexToScalar dst=v{} src=v{}", dst.0, src.0),
                VmInstr::IntMulStride { dst, src, .. } => format!("IntMulStride dst=v{} src=v{}", dst.0, src.0),
                VmInstr::ScalarToIndex { dst, src, .. } => format!("ScalarToIndex dst=v{} src=v{}", dst.0, src.0),
                VmInstr::Broadcast { dst, .. } => format!("Broadcast dst=v{}", dst.0),
                VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } =>
                    format!("StoreToken token_id=v{} output_buf=v{} counter=v{} input_ids_ptr=v{} prompt_len_bytes=v{}",
                        token_id.0, output_buf.0, counter.0, input_ids_ptr.0, prompt_len_bytes.0),
                _ => format!("{:?}", std::mem::discriminant(instr)),
            }
        };

        let is_gpr_kind = |k: &VRegKind| matches!(k, VRegKind::Ptr | VRegKind::Scalar | VRegKind::Counter | VRegKind::ByteOffset);
        let is_vec_kind = |k: &VRegKind| matches!(k, VRegKind::Vec);

        for (i, instr) in self.instrs.iter().enumerate() {
            let check = |vreg: VRegId, expect_gpr: bool, label: &str| -> Option<String> {
                if let Some(kind) = kinds.get(&vreg) {
                    if expect_gpr && !is_gpr_kind(kind) {
                        return Some(format!(
                            "instr[{i}] {}: {} v{} is {:?}, expected GPR class (Ptr/Scalar/Counter/ByteOffset)",
                            instr_brief(instr), label, vreg.0, kind));
                    }
                    if !expect_gpr && !is_vec_kind(kind) {
                        return Some(format!(
                            "instr[{i}] {}: {} v{} is {:?}, expected Vec class",
                            instr_brief(instr), label, vreg.0, kind));
                    }
                }
                None
            };

            match instr {
                VmInstr::VecLoad { dst, base, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                }
                VmInstr::VecStore { base, src, .. } => {
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::VecNarrow { dst, src, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::VecWiden { dst, src, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::Mov { dst, src, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::VecBinOp { dst, a, b, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*a, false, "a") { return Err(e); }
                    if let Some(e) = check(*b, false, "b") { return Err(e); }
                }
                VmInstr::VecShiftImm { dst, a, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*a, false, "a") { return Err(e); }
                }
                VmInstr::VecUnaryOp { dst, a, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*a, false, "a") { return Err(e); }
                }
                VmInstr::Fma { dst, acc, a, b, ..} => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*acc, false, "acc") { return Err(e); }
                    if let Some(e) = check(*a, false, "a") { return Err(e); }
                    if let Some(e) = check(*b, false, "b") { return Err(e); }
                }
                VmInstr::HReduce { dst, src, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::Accumulate { acc, src } => {
                    if let Some(e) = check(*acc, false, "acc") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::Transcendental { dst, src, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::LoadPtr { dst, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                }
                VmInstr::Argmax { dst, logits_ptr, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*logits_ptr, true, "logits_ptr") { return Err(e); }
                }
                VmInstr::AddPtr { dst, base, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                }
                VmInstr::ScalarLoad { dst, base, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                }
                VmInstr::ScalarStore { base, src, .. } => {
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                    if let Some(e) = check(*src, true, "src") { return Err(e); }
                }
                VmInstr::VecScalarStore { base, src, .. } => {
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::IntMulStride { dst, src, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*src, true, "src") { return Err(e); }
                }
                VmInstr::ScalarToIndex { dst, src, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*src, true, "src") { return Err(e); }
                }
                VmInstr::IndexToScalar { dst, src } => {
                    // dst holds f32 result (XMM register), src is GPR integer
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*src, true, "src") { return Err(e); }
                }
                VmInstr::ScalarByteLoad { dst, base, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                }
                VmInstr::Broadcast { dst, src, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let ScalarExpr::MemLoad(b, _) = src {
                        if let Some(e) = check(*b, true, "Broadcast.MemLoad.base") { return Err(e); }
                    }
                    if let ScalarExpr::ExtractLane0(s) = src {
                        if let Some(e) = check(*s, false, "Broadcast.ExtractLane0.src") { return Err(e); }
                    }
                    if let ScalarExpr::VReg(s) = src {
                        // VReg src can be GPR (legacy) or Vec (from IndexToScalar output)
                        // No strict class check — both are valid
                        let _ = s;
                    }
                }
                VmInstr::LoopBegin { counter, byte_offset, .. } => {
                    match kinds.get(counter) {
                        Some(VRegKind::Counter) => {}
                        Some(k) => return Err(format!(
                            "instr[{i}] LoopBegin: counter v{} is {:?}, expected Counter",
                            counter.0, k)),
                        None => {}
                    }
                    match kinds.get(byte_offset) {
                        Some(VRegKind::ByteOffset) => {}
                        Some(k) => return Err(format!(
                            "instr[{i}] LoopBegin: byte_offset v{} is {:?}, expected ByteOffset",
                            byte_offset.0, k)),
                        None => {}
                    }
                }
                VmInstr::TemperatureScale { logits_ptr, temp_ptr, .. } => {
                    if let Some(e) = check(*logits_ptr, true, "logits_ptr") { return Err(e); }
                    if let Some(e) = check(*temp_ptr, true, "temp_ptr") { return Err(e); }
                }
                VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => {
                    if let Some(e) = check(*token_id, true, "token_id") { return Err(e); }
                    if let Some(e) = check(*output_buf, true, "output_buf") { return Err(e); }
                    if let Some(e) = check(*counter, true, "counter") { return Err(e); }
                    if let Some(e) = check(*input_ids_ptr, true, "input_ids_ptr") { return Err(e); }
                    if let Some(e) = check(*prompt_len_bytes, true, "prompt_len_bytes") { return Err(e); }
                }
                VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => {
                    if let Some(e) = check(*token_id, true, "token_id") { return Err(e); }
                    if let Some(e) = check(*counter, true, "counter") { return Err(e); }
                    if let Some(e) = check(*eos_ptr, true, "eos_ptr") { return Err(e); }
                    if let Some(e) = check(*max_tokens_ptr, true, "max_tokens_ptr") { return Err(e); }
                }
                VmInstr::GprBinOp { dst, a, b, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*a, true, "a") { return Err(e); }
                    if let Some(v) = b.vreg() {
                        if let Some(e) = check(v, true, "b") { return Err(e); }
                    }
                }
                VmInstr::GprUnaryOp { dst, src, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                    if let Some(e) = check(*src, true, "src") { return Err(e); }
                }
                VmInstr::GprLoadImm { dst, .. } => {
                    if let Some(e) = check(*dst, true, "dst") { return Err(e); }
                }
                VmInstr::AtomicAdd { base, .. } => {
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                }
                VmInstr::GatherLoad { dst, base, indices, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                    if let Some(e) = check(*indices, true, "indices") { return Err(e); }
                }
                VmInstr::ScatterStore { base, indices, src, .. } => {
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                    if let Some(e) = check(*indices, true, "indices") { return Err(e); }
                    if let Some(e) = check(*src, false, "src") { return Err(e); }
                }
                VmInstr::TableLookup { dst, base, row_index, .. } => {
                    if let Some(e) = check(*dst, false, "dst") { return Err(e); }
                    if let Some(e) = check(*base, true, "base") { return Err(e); }
                    if let Some(e) = check(*row_index, true, "row_index") { return Err(e); }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// D3: SIMD 宽度一致性验证 (ARCH-VM-WIDTH-CHECK)。
    ///
    /// 检查每条 VmInstr 的操作数 SimdWidth 是否一致:
    /// - VecBinOp: dst.width == a.width == b.width
    /// - Fma: dst.width == acc.width == a.width == b.width
    /// - VecUnaryOp: dst.width == a.width
    /// - HReduce: dst.width == src.width
    /// - Accumulate: acc.width == src.width
    /// - Transcendental: dst.width == src.width
    /// - Broadcast: dst.width == 参数 width
    /// - VecLoad: dst.width == 参数 width
    /// - VecStore: src.width == 参数 width
    pub fn validate_width_consistency(&self) -> Result<(), String> {
        let widths = self.collect_vreg_widths();

        for (i, instr) in self.instrs.iter().enumerate() {
            match instr {
                // VecBinOp: dst/a/b 通常同宽，但 Scalar tail 模式下
                // b 可以比 a/dst 窄 (x86 lower 只用 lane 0)。
                VmInstr::VecBinOp { dst, a, b, .. } => {
                    Self::check_width_compatible_triple(i, "VecBinOp", &widths, *dst, *a, *b)?;
                }
                VmInstr::Fma { dst, acc, a, b, ..} => {
                    Self::check_width_match(i, "Fma", &widths, &[*dst, *acc, *a, *b])?;
                }
                VmInstr::VecUnaryOp { dst, a, .. } => {
                    Self::check_width_match(i, "VecUnaryOp", &widths, &[*dst, *a])?;
                }
                VmInstr::HReduce { dst, src, .. } => {
                    // HReduce: src is vector, dst is scalar (or same-width for in-place).
                    // Allow dst(Scalar) + src(W256+) — horizontal reduction narrows width.
                    if let (Some(dw), Some(sw)) = (widths.get(dst), widths.get(src)) {
                        if dw.f32_lanes() > sw.f32_lanes() {
                            return Err(format!(
                                "instr[{}] HReduce: dst {:?} wider than src {:?}",
                                i, dw, sw));
                        }
                    }
                }
                // Accumulate: 允许 src 比 acc 窄 (Scalar tail 元素累加到 W256 acc，
                // x86 lower 只影响 lane 0)。禁止 src 比 acc 宽。
                VmInstr::Accumulate { acc, src } => {
                    if let (Some(aw), Some(sw)) = (widths.get(acc), widths.get(src)) {
                        if sw.f32_lanes() > aw.f32_lanes() {
                            return Err(format!(
                                "instr[{i}] Accumulate: src v{} width {:?} wider than acc v{} width {:?}",
                                src.0, sw, acc.0, aw));
                        }
                    }
                }
                VmInstr::Transcendental { dst, src, .. } => {
                    Self::check_width_match(i, "Transcendental", &widths, &[*dst, *src])?;
                }
                VmInstr::Broadcast { dst, width, src, .. } => {
                    if let Some(dw) = widths.get(dst) {
                        if dw != width {
                            return Err(format!(
                                "instr[{i}] Broadcast: dst v{} width {:?} != instruction width {:?}",
                                dst.0, dw, width));
                        }
                    }
                    // ExtractLane0 的语义是取 src 的 lane 0 (标量值),
                    // 然后广播到 dst 的全宽度。src 可以是任意 SIMD 宽度,
                    // dst 由 instruction width 参数决定。无需额外宽度检查。
                    if let ScalarExpr::ExtractLane0(_s) = src {}
                }
                VmInstr::VecLoad { dst, width, .. } => {
                    if let Some(dw) = widths.get(dst) {
                        if dw != width {
                            return Err(format!(
                                "instr[{i}] VecLoad: dst v{} width {:?} != instruction width {:?}",
                                dst.0, dw, width));
                        }
                    }
                }
                VmInstr::VecStore { src, width, .. } => {
                    if let Some(sw) = widths.get(src) {
                        if sw != width {
                            return Err(format!(
                                "instr[{i}] VecStore: src v{} width {:?} != instruction width {:?}",
                                src.0, sw, width));
                        }
                    }
                }
                VmInstr::VecNarrow { dst, src, width, .. } => {
                    if let Some(sw) = widths.get(dst) {
                        if sw != width {
                            return Err(format!(
                                "instr[{i}] VecNarrow: dst v{} width {:?} != instruction width {:?}",
                                dst.0, sw, width));
                        }
                    }
                    if let Some(sw) = widths.get(src) {
                        if sw != width {
                            return Err(format!(
                                "instr[{i}] VecNarrow: src v{} width {:?} != instruction width {:?}",
                                src.0, sw, width));
                        }
                    }
                }
                VmInstr::VecWiden { dst, src, width, .. } => {
                    if let Some(sw) = widths.get(dst) {
                        if sw != width {
                            return Err(format!(
                                "instr[{i}] VecWiden: dst v{} width {:?} != instruction width {:?}",
                                dst.0, sw, width));
                        }
                    }
                    if let Some(sw) = widths.get(src) {
                        if sw != width {
                            return Err(format!(
                                "instr[{i}] VecWiden: src v{} width {:?} != instruction width {:?}",
                                src.0, sw, width));
                        }
                    }
                }
                VmInstr::Mov { .. } => {
                    // Mov is a same-width register-to-register copy, no width mismatch possible.
                }
                VmInstr::VecShiftImm { dst, a, width, .. } => {
                    Self::check_width_match(i, "VecShiftImm", &widths, &[*dst, *a])?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn check_width_match(
        instr_idx: usize,
        label: &str,
        widths: &std::collections::HashMap<VRegId, SimdWidth>,
        vregs: &[VRegId],
    ) -> Result<(), String> {
        let mut first: Option<(VRegId, SimdWidth)> = None;
        for &vreg in vregs {
            if let Some(&w) = widths.get(&vreg) {
                if let Some((fv, fw)) = first {
                    if fw != w {
                        return Err(format!(
                            "instr[{}] {}: width mismatch — v{} {:?} vs v{} {:?}",
                            instr_idx, label, fv.0, fw, vreg.0, w));
                    }
                } else {
                    first = Some((vreg, w));
                }
            }
        }
        Ok(())
    }

    /// VecBinOp 宽度兼容性检查: dst 和 a 必须同宽, b 宽度不限
    /// (Scalar tail 模式: `s_temp(Scalar) = s_temp(Scalar) * scale(W256)` 在
    /// x86 lower 中只乘 lane 0，语义正确。scale(W256) * s_temp(Scalar) 同理)。
    fn check_width_compatible_triple(
        instr_idx: usize,
        label: &str,
        widths: &std::collections::HashMap<VRegId, SimdWidth>,
        dst: VRegId,
        a: VRegId,
        b: VRegId,
    ) -> Result<(), String> {
        // dst and a must match
        if let (Some(dw), Some(aw)) = (widths.get(&dst), widths.get(&a)) {
            if dw != aw {
                return Err(format!(
                    "instr[{}] {}: dst v{} width {:?} != a v{} width {:?}",
                    instr_idx, label, dst.0, dw, a.0, aw));
            }
        }
        Ok(())
    }

    /// 收集所有 DeclareVReg 中的 VRegId → VRegKind 映射。
    fn collect_vreg_kinds(&self) -> std::collections::HashMap<VRegId, VRegKind> {
        let mut map = std::collections::HashMap::new();
        for instr in &self.instrs {
            if let VmInstr::DeclareVReg { id, kind, .. } = instr {
                map.insert(*id, *kind);
            }
        }
        map
    }

    /// 收集所有 DeclareVReg 中的 VRegId → SimdWidth 映射。
    fn collect_vreg_widths(&self) -> std::collections::HashMap<VRegId, SimdWidth> {
        let mut map = std::collections::HashMap::new();
        for instr in &self.instrs {
            if let VmInstr::DeclareVReg { id, width, .. } = instr {
                map.insert(*id, *width);
            }
        }
        map
    }

    /// D1: 值域验证 (ARCH-VM-VALUE-DOMAIN)。
    ///
    /// 编译时符号执行: 遍历 VmProgram，追踪每个 VRegId 的值域 (ValueDomain)。
    /// 当 VecLoad/VecStore 的 base 不是有效的 Ptr/ByteOffset 域时报错。
    /// 当 LoadPtr 的结果被当作 Vec 操作数时报错。
    ///
    /// 值域定义:
    /// - Ptr:      来自 ABI/栈参数的指针 (由 LoadPtr 加载)
    /// - ByteOff:  字节偏移 (由 LoopOffset/IntMulStride 产生)
    /// - Counter:  循环计数器值
    /// - VecData:  SIMD 向量数据 (由 VecLoad/Broadcast/Fma 产生)
    /// - Scalar:   标量值 (由 ScalarLoad/HReduce 产生)
    /// - Unknown:  无法推导
    pub fn validate_value_domains(&self) -> Result<(), String> {
        use std::collections::HashMap;

        /// 值的语义域
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Domain {
            /// 指针 — 来自 LoadPtr 或 AddPtr，可作为内存访问的 base
            Ptr,
            /// 字节偏移 — 来自 LoopOffset/IntMulStride，不能直接作 base
            ByteOff,
            /// 循环计数器
            Counter,
            /// 字节偏移 (LoopBegin 产生的关联偏移)
            LoopByteOffset,
            /// SIMD 向量数据
            VecData,
            /// 标量数据 (GPR 中的非指针值)
            ScalarData,
            /// 未知域
            Unknown,
        }

        let kinds = self.collect_vreg_kinds();
        let mut domains: HashMap<VRegId, Domain> = HashMap::new();

        // Initialize: Ptr kind → Unknown (需要 LoadPtr 才变成 Ptr),
        // Vec kind → Unknown, Counter → Counter, ByteOffset → LoopByteOffset.
        for (&id, &kind) in &kinds {
            let domain = match kind {
                VRegKind::Counter => Domain::Counter,
                VRegKind::ByteOffset => Domain::LoopByteOffset,
                _ => Domain::Unknown,
            };
            domains.insert(id, domain);
        }

        for (i, instr) in &mut self.instrs.iter().enumerate() {
            match instr {
                // LoadPtr: dst becomes Ptr domain
                VmInstr::LoadPtr { dst, src } => {
                    domains.insert(*dst, Domain::Ptr);
                }
                // AddPtr: dst = base + offset → if base is Ptr, dst is Ptr
                VmInstr::AddPtr { dst, base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if base_domain == Domain::Ptr {
                        domains.insert(*dst, Domain::Ptr);
                    } else {
                        // Adding offset to non-pointer: result is still non-pointer
                        domains.insert(*dst, base_domain);
                    }
                }
                // VecLoad: verify base is Ptr/ByteOffset domain
                VmInstr::VecLoad { dst, base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    match base_domain {
                        Domain::Ptr | Domain::Unknown => {} // Ptr is correct, Unknown may become Ptr later
                        Domain::ByteOff | Domain::LoopByteOffset | Domain::Counter => {
                            return Err(format!(
                                "instr[{i}] VecLoad: base v{} has domain {:?}, \
                                 expected Ptr. ByteOffset/Counter used as memory base → SIGSEGV",
                                base.0, base_domain));
                        }
                        Domain::VecData | Domain::ScalarData => {
                            return Err(format!(
                                "instr[{i}] VecLoad: base v{} has domain {:?}, \
                                 expected Ptr. Data value used as memory base → SIGSEGV",
                                base.0, base_domain));
                        }
                    }
                    domains.insert(*dst, Domain::VecData);
                }
                // VecStore: verify base is Ptr domain
                VmInstr::VecStore { base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    match base_domain {
                        Domain::Ptr | Domain::Unknown => {}
                        Domain::ByteOff | Domain::LoopByteOffset | Domain::Counter => {
                            return Err(format!(
                                "instr[{i}] VecStore: base v{} has domain {:?}, \
                                 expected Ptr. ByteOffset/Counter used as memory base → SIGSEGV",
                                base.0, base_domain));
                        }
                        Domain::VecData | Domain::ScalarData => {
                            return Err(format!(
                                "instr[{i}] VecStore: base v{} has domain {:?}, \
                                 expected Ptr. Data value used as memory base → SIGSEGV",
                                base.0, base_domain));
                        }
                    }
                }
                // Vec operations: propagate VecData domain
                VmInstr::Broadcast { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::VecNarrow { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::VecWiden { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::Mov { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::VecBinOp { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::VecUnaryOp { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::VecCmp { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::VecCast { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::ConditionalSelect { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::Fma { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::HReduce { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::Accumulate { .. } => {
                    // acc domain stays the same
                }
                VmInstr::Transcendental { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                // Scalar operations
                VmInstr::ScalarLoad { dst, base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] ScalarLoad: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                    // If dst VReg is registered as Ptr kind, propagate Ptr domain;
                    // otherwise the loaded value is a plain scalar.
                    let dst_domain = match kinds.get(dst) {
                        Some(VRegKind::Ptr) => Domain::Ptr,
                        _ => Domain::ScalarData,
                    };
                    domains.insert(*dst, dst_domain);
                }
                VmInstr::ScalarStore { base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] ScalarStore: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                }
                VmInstr::VecScalarStore { base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] VecScalarStore: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                }
                VmInstr::ScalarByteLoad { dst, base, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] ScalarByteLoad: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // IntMulStride: produces byte offset
                VmInstr::IntMulStride { dst, .. } => {
                    domains.insert(*dst, Domain::ByteOff);
                }
                VmInstr::ScalarToIndex { dst, .. } => {
                    domains.insert(*dst, Domain::ByteOff);
                }
                VmInstr::IndexToScalar { dst, .. } => {
                    domains.insert(*dst, Domain::ScalarData);
                }
                // GPR operations
                VmInstr::GprBinOp { dst, a, b, op } => {
                    let da = domains.get(a).copied().unwrap_or(Domain::Unknown);
                    let db = b.vreg().and_then(|v| domains.get(&v).copied()).unwrap_or(Domain::Unknown);
                    if matches!(op, GprOp::Mul) {
                        domains.insert(*dst, da);
                    } else if da == Domain::Ptr || db == Domain::Ptr {
                        domains.insert(*dst, Domain::Ptr);
                    } else if matches!(op, GprOp::Sub) && da == Domain::Unknown {
                        // Preserve existing domain for counter decrement
                    } else {
                        domains.insert(*dst, Domain::ScalarData);
                    }
                }
                VmInstr::GprUnaryOp { dst, .. } => {
                    domains.insert(*dst, Domain::ScalarData);
                }
                VmInstr::GprLoadImm { dst, .. } => {
                    domains.insert(*dst, Domain::ScalarData);
                }
                VmInstr::LoadCallbackEntry { table_ptr, fn_ptr_out, ctx_out, .. } => {
                    let tbl_domain = domains.get(table_ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(tbl_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] LoadCallbackEntry: table_ptr v{} has domain {:?}, expected Ptr",
                            table_ptr.0, tbl_domain));
                    }
                    domains.insert(*fn_ptr_out, Domain::Ptr);
                    domains.insert(*ctx_out, Domain::Ptr);
                }
                VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr } => {
                    for (&vreg, label) in [(fn_ptr, "fn_ptr"), (ctx_ptr, "ctx_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] NativeCall: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                    domains.insert(*ret_val, Domain::ScalarData);
                }
                // Sampling operations
                VmInstr::Argmax { dst, logits_ptr, .. } => {
                    let ptr_domain = domains.get(logits_ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ptr_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] Argmax: logits_ptr v{} has domain {:?}, expected Ptr",
                            logits_ptr.0, ptr_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                VmInstr::StoreToken { output_buf, input_ids_ptr, .. } => {
                    for (&vreg, label) in [(output_buf, "output_buf"), (input_ids_ptr, "input_ids_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] StoreToken: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                }
                VmInstr::CheckStopCondition { eos_ptr, max_tokens_ptr, .. } => {
                    for (&vreg, label) in [(eos_ptr, "eos_ptr"), (max_tokens_ptr, "max_tokens_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] CheckStopCondition: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                }
                VmInstr::TemperatureScale { logits_ptr, temp_ptr, .. } => {
                    for (&vreg, label) in [(logits_ptr, "logits_ptr"), (temp_ptr, "temp_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] TemperatureScale: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                }
                VmInstr::AtomicAdd { base, .. } => {
                    let d = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(d, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] AtomicAdd: base v{} has domain {:?}, expected Ptr",
                            base.0, d));
                    }
                }
                VmInstr::Prefetch { base, .. } => {
                    let d = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(d, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] Prefetch: base v{} has domain {:?}, expected Ptr",
                            base.0, d));
                    }
                }
                // GatherLoad: base must be Ptr, dst becomes VecData
                VmInstr::GatherLoad { dst, base, indices, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] GatherLoad: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                    let idx_domain = domains.get(indices).copied().unwrap_or(Domain::Unknown);
                    if !matches!(idx_domain, Domain::Ptr | Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] GatherLoad: indices v{} has domain {:?}, expected Ptr or ScalarData",
                            indices.0, idx_domain));
                    }
                    domains.insert(*dst, Domain::VecData);
                }
                // ScatterStore: base must be Ptr
                VmInstr::ScatterStore { base, indices, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] ScatterStore: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                    let idx_domain = domains.get(indices).copied().unwrap_or(Domain::Unknown);
                    if !matches!(idx_domain, Domain::Ptr | Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] ScatterStore: indices v{} has domain {:?}, expected Ptr or ScalarData",
                            indices.0, idx_domain));
                    }
                }
                // TableLookup: base must be Ptr, row_index must be GPR, dst becomes VecData
                VmInstr::TableLookup { dst, base, row_index, .. } => {
                    let base_domain = domains.get(base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] TableLookup: base v{} has domain {:?}, expected Ptr",
                            base.0, base_domain));
                    }
                    let idx_domain = domains.get(row_index).copied().unwrap_or(Domain::Unknown);
                    if !matches!(idx_domain, Domain::Ptr | Domain::ScalarData | Domain::ByteOff | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] TableLookup: row_index v{} has domain {:?}, expected Ptr/ScalarData/ByteOff",
                            row_index.0, idx_domain));
                    }
                    domains.insert(*dst, Domain::VecData);
                }
                // LoopBegin: counter and byte_offset domains set by DeclareVReg
                VmInstr::LoopBegin { .. } | VmInstr::LoopEnd => {}
                // Meta instructions: no domain changes
                VmInstr::DeclareVReg { .. } | VmInstr::ReleaseVReg { .. }
                | VmInstr::MarkLabel { .. } | VmInstr::Comment(_)
                | VmInstr::ScopeBegin { .. } | VmInstr::ScopeEnd { .. }
                | VmInstr::MemFence { .. } | VmInstr::StoreConstToStack { .. }
                | VmInstr::BreakLoop { .. } | VmInstr::ConditionalSkip { .. }
                | VmInstr::GprCondAction { .. }
                | VmInstr::WarpSync | VmInstr::AsyncCopy { .. } | VmInstr::AsyncWait { .. }
                | VmInstr::TileConfig { .. } | VmInstr::TileMma { .. } | VmInstr::TileRelease
                | VmInstr::Vp2Intersect { .. } | VmInstr::HotpatchSlot { .. }
                | VmInstr::IndirectJump { .. } | VmInstr::ConditionalExit { .. }
                | VmInstr::BranchIfPtrNonNull { .. }
                | VmInstr::BranchIfGprZero { .. }
                | VmInstr::BranchIfGprLtU { .. }
                | VmInstr::UnconditionalBranch { .. }
                | VmInstr::BatchSeqIdLookup { .. } | VmInstr::BatchPerSeqArgmax { .. }
                | VmInstr::BatchPerSeqStopCheck { .. }
                | VmInstr::ActivationSwap { .. }
                | VmInstr::PageTableAddr { .. }
                | VmInstr::PageTableKVWrite { .. }
                | VmInstr::PageTableKVWriteQuant { .. }
                | VmInstr::KiviQuantChannel { .. }
                | VmInstr::KiviQuantToken { .. }
                | VmInstr::KiviDequantLoad { .. }
                | VmInstr::SharedMemAlloc { .. }
                | VmInstr::SharedMemStore { .. }
                | VmInstr::SharedMemLoad { .. }
                | VmInstr::SharedMemAsyncStore { .. }
                | VmInstr::SharedMemAsyncWaitGroup { .. }
                | VmInstr::TmemAlloc { .. }
                | VmInstr::TmemStore { .. }
                | VmInstr::TmemDealloc { .. }
                | VmInstr::ClusterBarrierInit { .. }
                | VmInstr::ClusterStore { .. }
                | VmInstr::WeightPrefetchAsync { .. }
                | VmInstr::WeightPrefetchWait { .. }
                | VmInstr::WarpRoleDeclare { .. }
                | VmInstr::WarpBarrierArrive { .. }
                | VmInstr::WarpBarrierWait { .. }
                | VmInstr::TmaDescriptorInit { .. } | VmInstr::Tma2DCopy { .. } | VmInstr::BarrierInit { .. }
                | VmInstr::BlockSync
                | VmInstr::WarpReduce { .. }
                | VmInstr::Lz4Decode { .. }
                | VmInstr::BitPackRleDecode { .. }
                | VmInstr::DebugBreakpoint { .. }
                | VmInstr::DebugMarker { .. }
                | VmInstr::DebugProbe { .. }
                | VmInstr::DebugBreakIf { .. } => {}
                VmInstr::GgufSubScaleLoad { dst, scales_base, sub_block_idx, .. } => {
                    let sb_domain = domains.get(scales_base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(sb_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] GGUF SubScale load: scales_base v{} has domain {:?}, expected Ptr",
                            scales_base.0, sb_domain));
                    }
                    let idx_domain = domains.get(sub_block_idx).copied().unwrap_or(Domain::Unknown);
                    if idx_domain == Domain::VecData {
                        return Err(format!(
                            "instr[{i}] GGUF SubScale load: sub_block_idx v{} has domain VecData, expected Ptr/Scalar/Unknown",
                            sub_block_idx.0));
                    }
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::GgufKQuantScaleLoad { dst, scales_base, sub_block_idx, .. } => {
                    let sb_domain = domains.get(scales_base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(sb_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] GGUF KQuant scale load: scales_base v{} has domain {:?}, expected Ptr",
                            scales_base.0, sb_domain));
                    }
                    let idx_domain = domains.get(sub_block_idx).copied().unwrap_or(Domain::Unknown);
                    if idx_domain == Domain::VecData {
                        return Err(format!(
                            "instr[{i}] GGUF KQuant scale load: sub_block_idx v{} has domain VecData, expected Ptr/Scalar/Unknown",
                            sub_block_idx.0));
                    }
                    domains.insert(*dst, Domain::VecData);
                }
                // Quant* decode instrs: produce VecData
                VmInstr::QuantBroadcastInt { dst, .. }
                | VmInstr::QuantScalarCvtLoad { dst, .. }
                | VmInstr::QuantBlockLoad { dst, .. }
                | VmInstr::QuantBiPlaneLoad { dst, .. }
                | VmInstr::QuantLoadBytesVec { dst, .. }
                | VmInstr::QuantCodebookLookup { dst, .. }
                | VmInstr::QuantExtractBits { dst, .. }
                | VmInstr::QuantDequantFma { dst, .. }
                | VmInstr::QuantInterleave { dst, .. }
                | VmInstr::QuantConcatSeq { dst, .. }
                | VmInstr::Q3KDecodeStep { dst, .. }
                | VmInstr::DotProduct { acc: dst, .. }
                | VmInstr::ScaleApply { dst, .. }
                | VmInstr::BitwiseGemm { dst, .. }
                | VmInstr::SparseGemm { acc: dst, .. }
                | VmInstr::SparseFp8Gemm { acc: dst, .. }
                | VmInstr::NativeFp4Gemm { acc: dst, .. }
                | VmInstr::NativeFp8Gemm { acc: dst, .. }
                | VmInstr::HwQuantDequant { dst, .. }
                | VmInstr::TmemLoad { dst, .. }
                | VmInstr::ClusterLoad { dst, .. } => {
                    domains.insert(*dst, Domain::VecData);
                }
                VmInstr::MemCopy { dst, src, .. } => {
                    let src_domain = domains.get(src).copied().unwrap_or(Domain::Ptr);
                    domains.insert(*dst, src_domain);
                }
                // GPU sampling: SoftmaxReduceMax — logits_ptr must be Ptr, dst → ScalarData
                VmInstr::SoftmaxReduceMax { dst, logits_ptr, .. } => {
                    let ptr_domain = domains.get(logits_ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ptr_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SoftmaxReduceMax: logits_ptr v{} has domain {:?}, expected Ptr",
                            logits_ptr.0, ptr_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // GPU sampling: SoftmaxExpSum — logits_ptr Ptr, max_val ScalarData, sum_dst → ScalarData
                VmInstr::SoftmaxExpSum { sum_dst, logits_ptr, max_val, .. } => {
                    let ptr_domain = domains.get(logits_ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ptr_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SoftmaxExpSum: logits_ptr v{} has domain {:?}, expected Ptr",
                            logits_ptr.0, ptr_domain));
                    }
                    let mv_domain = domains.get(max_val).copied().unwrap_or(Domain::Unknown);
                    if !matches!(mv_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SoftmaxExpSum: max_val v{} has domain {:?}, expected ScalarData",
                            max_val.0, mv_domain));
                    }
                    domains.insert(*sum_dst, Domain::ScalarData);
                }
                // GPU sampling: SoftmaxNormalize — logits_ptr Ptr, sum_val ScalarData (in-place)
                VmInstr::SoftmaxNormalize { logits_ptr, sum_val, .. } => {
                    let ptr_domain = domains.get(logits_ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ptr_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SoftmaxNormalize: logits_ptr v{} has domain {:?}, expected Ptr",
                            logits_ptr.0, ptr_domain));
                    }
                    let sv_domain = domains.get(sum_val).copied().unwrap_or(Domain::Unknown);
                    if !matches!(sv_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SoftmaxNormalize: sum_val v{} has domain {:?}, expected ScalarData",
                            sum_val.0, sv_domain));
                    }
                }
                // GPU sampling: SampleTopKFilter — all Ptr (in-place filtering)
                VmInstr::SampleTopKFilter { probs_ptr, indices_ptr, k_ptr, .. } => {
                    for (&vreg, label) in [(probs_ptr, "probs_ptr"), (indices_ptr, "indices_ptr"), (k_ptr, "k_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] SampleTopKFilter: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                }
                // GPU sampling: SampleTopPFilter — all Ptr (in-place filtering)
                VmInstr::SampleTopPFilter { probs_ptr, p_ptr, .. } => {
                    for (&vreg, label) in [(probs_ptr, "probs_ptr"), (p_ptr, "p_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] SampleTopPFilter: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                }
                // GPU sampling: SampleMultinomial — probs_ptr/rng_state_ptr Ptr, dst → ScalarData
                VmInstr::SampleMultinomial { dst, probs_ptr, rng_state_ptr, .. } => {
                    for (&vreg, label) in [(probs_ptr, "probs_ptr"), (rng_state_ptr, "rng_state_ptr")] {
                        let d = domains.get(&vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] SampleMultinomial: {} v{} has domain {:?}, expected Ptr",
                                label, vreg.0, d));
                        }
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // GPU sampling: WarpPRNG — rng_state_ptr Ptr, dst → ScalarData
                VmInstr::WarpPRNG { dst, rng_state_ptr } => {
                    let ptr_domain = domains.get(rng_state_ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ptr_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] WarpPRNG: rng_state_ptr v{} has domain {:?}, expected Ptr",
                            rng_state_ptr.0, ptr_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // SharedMemSwizzle — raw_addr input (Ptr/ByteOffset), dst → Ptr
                VmInstr::SharedMemSwizzle { dst, raw_addr, .. } => {
                    let src_domain = domains.get(raw_addr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(src_domain, Domain::Ptr | Domain::ByteOff | Domain::LoopByteOffset | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SharedMemSwizzle: raw_addr v{} has domain {:?}, expected Ptr/ByteOffset",
                            raw_addr.0, src_domain));
                    }
                    domains.insert(*dst, Domain::Ptr);
                }
                // VecShuffle — src (Vec), dst → Vec (lane rearrangement)
                VmInstr::VecShuffle { dst, src, .. } => {
                    let src_domain = domains.get(src).copied().unwrap_or(Domain::Unknown);
                    if !matches!(src_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] VecShuffle: src v{} has domain {:?}, expected ScalarData (Vec)",
                            src.0, src_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // VecExtractLane — src (Vec), dst → ScalarData
                VmInstr::VecExtractLane { dst, src, .. } => {
                    let src_domain = domains.get(src).copied().unwrap_or(Domain::Unknown);
                    if !matches!(src_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] VecExtractLane: src v{} has domain {:?}, expected ScalarData (Vec)",
                            src.0, src_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // VecInsertLane — src_vec (Vec) + src_scalar (ScalarData), dst → Vec
                VmInstr::VecInsertLane { dst, src_vec, src_scalar, .. } => {
                    let vec_domain = domains.get(src_vec).copied().unwrap_or(Domain::Unknown);
                    let scalar_domain = domains.get(src_scalar).copied().unwrap_or(Domain::Unknown);
                    if !matches!(vec_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] VecInsertLane: src_vec v{} has domain {:?}, expected ScalarData (Vec)",
                            src_vec.0, vec_domain));
                    }
                    if !matches!(scalar_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] VecInsertLane: src_scalar v{} has domain {:?}, expected ScalarData",
                            src_scalar.0, scalar_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // VecLoadConst — dst → Vec (no src vregs)
                VmInstr::VecLoadConst { dst, .. } => {
                    domains.insert(*dst, Domain::ScalarData);
                }
                // AtomicCAS — ptr (Ptr), expected/desired (ScalarData), dst → ScalarData
                VmInstr::AtomicCAS { dst, ptr, expected, desired, .. } => {
                    let ptr_domain = domains.get(ptr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ptr_domain, Domain::Ptr | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] AtomicCAS: ptr v{} has domain {:?}, expected Ptr",
                            ptr.0, ptr_domain));
                    }
                    let exp_domain = domains.get(expected).copied().unwrap_or(Domain::Unknown);
                    if !matches!(exp_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] AtomicCAS: expected v{} has domain {:?}, expected ScalarData",
                            expected.0, exp_domain));
                    }
                    let des_domain = domains.get(desired).copied().unwrap_or(Domain::Unknown);
                    if !matches!(des_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] AtomicCAS: desired v{} has domain {:?}, expected ScalarData",
                            desired.0, des_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // SeqIdLookup — token_index (Counter), seq_meta_base (Ptr), num_seqs (ScalarData), dst → ScalarData
                VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, .. } => {
                    let idx_domain = domains.get(token_index).copied().unwrap_or(Domain::Unknown);
                    if !matches!(idx_domain, Domain::Counter | Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SeqIdLookup: token_index v{} has domain {:?}, expected Counter",
                            token_index.0, idx_domain));
                    }
                    let base_domain = domains.get(seq_meta_base).copied().unwrap_or(Domain::Unknown);
                    if !matches!(base_domain, Domain::Ptr | Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SeqIdLookup: seq_meta_base v{} has domain {:?}, expected Ptr",
                            seq_meta_base.0, base_domain));
                    }
                    let ns_domain = domains.get(num_seqs).copied().unwrap_or(Domain::Unknown);
                    if !matches!(ns_domain, Domain::ScalarData | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] SeqIdLookup: num_seqs v{} has domain {:?}, expected ScalarData",
                            num_seqs.0, ns_domain));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                // AllReduceChunk: sendbuf/recvbuf are Ptr, count/rank/world_size/chunk_idx are ScalarData
                #[cfg(feature = "nccl")]
                VmInstr::AllReduceChunk { sendbuf, recvbuf, count, rank, world_size, chunk_idx, .. } => {
                    for &ptr_vreg in &[sendbuf, recvbuf] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::ScalarData | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] AllReduceChunk: buf v{} has domain {:?}, expected Ptr",
                                ptr_vreg.0, d));
                        }
                    }
                    for &scalar_vreg in &[count, rank, world_size, chunk_idx] {
                        let d = domains.get(scalar_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::ScalarData | Domain::Counter | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] AllReduceChunk: param v{} has domain {:?}, expected ScalarData",
                                scalar_vreg.0, d));
                        }
                    }
                }
                // CommBarrier: thread_count is ScalarData
                #[cfg(feature = "nccl")]
                VmInstr::CommBarrier { thread_count, .. } => {
                    let tc_domain = domains.get(thread_count).copied().unwrap_or(Domain::Unknown);
                    if !matches!(tc_domain, Domain::ScalarData | Domain::Counter | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] CommBarrier: thread_count v{} has domain {:?}, expected ScalarData",
                            thread_count.0, tc_domain));
                    }
                }
                // NvlinkAsyncCopy: dst/src are Ptr, len is ScalarData
                #[cfg(feature = "nccl")]
                VmInstr::NvlinkAsyncCopy { dst, src, len, .. } => {
                    for &ptr_vreg in &[dst, src] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::ScalarData | Domain::Unknown) {
                            return Err(format!(
                                "instr[{i}] NvlinkAsyncCopy: ptr v{} has domain {:?}, expected Ptr",
                                ptr_vreg.0, d));
                        }
                    }
                    let len_domain = domains.get(len).copied().unwrap_or(Domain::Unknown);
                    if !matches!(len_domain, Domain::ScalarData | Domain::Counter | Domain::Unknown) {
                        return Err(format!(
                            "instr[{i}] NvlinkAsyncCopy: len v{} has domain {:?}, expected ScalarData",
                            len.0, len_domain));
                    }
                }
                // ── Distributed paging VmInstr (nccl) ──
                #[cfg(feature = "nccl")]
                VmInstr::RemotePageLookup { dst, seq_id, page_index, routing_table_base, .. } => {
                    for &ptr_vreg in &[routing_table_base] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!("instr[{i}] RemotePageLookup: ptr v{} has domain {:?}, expected Ptr", ptr_vreg.0, d));
                        }
                    }
                    domains.insert(*dst, Domain::Ptr);
                    let _ = (seq_id, page_index);
                }
                #[cfg(feature = "nccl")]
                VmInstr::P2pPageFetch { local_buf, peer_buf, page_size, barrier } => {
                    for &ptr_vreg in &[local_buf, peer_buf, barrier] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!("instr[{i}] P2pPageFetch: ptr v{} has domain {:?}, expected Ptr", ptr_vreg.0, d));
                        }
                    }
                    let _ = page_size;
                }
                #[cfg(feature = "nccl")]
                VmInstr::RdmaPageFetch { local_buf, remote_addr, rkey, page_size, sq_desc, doorbell, cq_addr } => {
                    for &ptr_vreg in &[local_buf, remote_addr, rkey, sq_desc, doorbell, cq_addr] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!("instr[{i}] RdmaPageFetch: ptr v{} has domain {:?}, expected Ptr", ptr_vreg.0, d));
                        }
                    }
                    let _ = page_size;
                }
                #[cfg(feature = "nccl")]
                VmInstr::RdmaPageFetchCompressed { local_buf, scratch_buf, remote_addr, rkey, sq_desc, doorbell, cq_addr, .. } => {
                    for &ptr_vreg in &[local_buf, scratch_buf, remote_addr, rkey, sq_desc, doorbell, cq_addr] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!("instr[{i}] RdmaPageFetchCompressed: ptr v{} has domain {:?}, expected Ptr", ptr_vreg.0, d));
                        }
                    }
                }
                #[cfg(feature = "nccl")]
                VmInstr::RemotePageAttn { q_buf, k_remote_buf, v_remote_buf, output_buf, shared_buf, barrier, tile_bytes } => {
                    for &ptr_vreg in &[q_buf, k_remote_buf, v_remote_buf, output_buf, shared_buf, barrier] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!("instr[{i}] RemotePageAttn: ptr v{} has domain {:?}, expected Ptr", ptr_vreg.0, d));
                        }
                    }
                    let _ = tile_bytes;
                }
                #[cfg(feature = "nccl")]
                VmInstr::PageMigrationLock { dst, entry_addr } => {
                    let d = domains.get(entry_addr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(d, Domain::Ptr | Domain::Unknown) {
                        return Err(format!("instr[{i}] PageMigrationLock: entry_addr v{} has domain {:?}, expected Ptr", entry_addr.0, d));
                    }
                    domains.insert(*dst, Domain::ScalarData);
                }
                #[cfg(feature = "nccl")]
                VmInstr::PageMigrationUnlock { entry_addr } => {
                    let d = domains.get(entry_addr).copied().unwrap_or(Domain::Unknown);
                    if !matches!(d, Domain::Ptr | Domain::Unknown) {
                        return Err(format!("instr[{i}] PageMigrationUnlock: entry_addr v{} has domain {:?}, expected Ptr", entry_addr.0, d));
                    }
                }
                #[cfg(feature = "nccl")]
                VmInstr::PageLocationUpdate { entry_addr, new_location, .. } => {
                    for &ptr_vreg in &[entry_addr, new_location] {
                        let d = domains.get(ptr_vreg).copied().unwrap_or(Domain::Unknown);
                        if !matches!(d, Domain::Ptr | Domain::Unknown) {
                            return Err(format!("instr[{i}] PageLocationUpdate: ptr v{} has domain {:?}, expected Ptr", ptr_vreg.0, d));
                        }
                    }
                }
                VmInstr::VecShiftImm { dst, a, .. } => {
                    let a_domain = domains.get(a).copied().unwrap_or(Domain::Unknown);
                    domains.insert(*dst, a_domain);
                }
            }
        }
        Ok(())
    }
}

