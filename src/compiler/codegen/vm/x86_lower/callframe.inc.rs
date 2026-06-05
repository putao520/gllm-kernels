
// ============================================================================
// SymbolicSaveFrame — 符号化栈帧追踪, 自动计算对齐, 生成 save/restore 序列
//
// 替代手工 push/pop/sub/add 计算, 消除 NativeCall 对齐错误根源。
// ============================================================================

/// 符号化地声明要保存的寄存器, 自动计算栈布局和对齐。
/// 所有 push/sub/vmovups 由 tracker 自动生成, 恢复序列由 `restore_all` 自动生成。
pub(crate) struct SymbolicSaveFrame<'a> {
    asm: &'a mut CodeAssembler,
    /// 累计栈调整 (字节), 用于对齐计算。
    total_adjustment: i32,
    /// 记录每个保存操作, 恢复时反向执行。前 4 项用于 debug 验证。
    op_log: Vec<&'static str>,
}

impl<'a> SymbolicSaveFrame<'a> {
    pub fn new(asm: &'a mut CodeAssembler) -> Self {
        Self { asm, total_adjustment: 0, op_log: Vec::new() }
    }

    /// 保存 GPR 列表。自动计算栈影响。
    pub fn push_gprs(&mut self, regs: &[AsmRegister64]) -> Result<(), CompilerError> {
        for &r in regs {
            self.asm.push(r).map_err(X86Lower::err)?;
        }
        let bytes = (regs.len() * 8) as i32;
        self.total_adjustment += bytes;
        self.op_log.push("push_gprs");
        Ok(())
    }

    /// 保存 RFLAGS。
    pub fn pushfq(&mut self) -> Result<(), CompilerError> {
        self.asm.pushfq().map_err(X86Lower::err)?;
        self.total_adjustment += 8;
        self.op_log.push("pushfq");
        Ok(())
    }

    /// 保存 YMM 寄存器块: sub rsp + N*32 + vmovups for each。
    /// 自动计算总空间和对齐。
    pub fn save_ymm_block(&mut self, regs: &[AsmRegisterYmm]) -> Result<(), CompilerError> {
        let stride: i32 = 32;
        let total = (regs.len() as i32) * stride;
        self.asm.sub(rsp, total).map_err(X86Lower::err)?;
        for (i, &r) in regs.iter().enumerate() {
            self.asm.vmovups(
                ymmword_ptr(rsp + i as i32 * stride), r,
            ).map_err(X86Lower::err)?;
        }
        self.total_adjustment += total;
        self.op_log.push("save_ymm");
        Ok(())
    }

    /// 执行 call rax, 验证 16B 栈对齐。
    /// call pushes 8B return addr, 所以 call 前 RSP 应为 8-mod-16,
    /// call 后 (callee 入口) RSP 为 16B-aligned。
    pub fn call_rax(&mut self, target: AsmRegister64) -> Result<(), CompilerError> {
        // call 前 RSP 必须 8-mod-16 (call +8B → callee 入口 0-mod-16)
        if self.total_adjustment % 16 != 8 {
            return Err(CompilerError::CodegenViolation(format!(
                "SymbolicSaveFrame::call_rax: bad alignment before call. \
                 total_adjustment={} (expected {}=8-mod-16). \
                 Caller must adjust save list to ensure callee gets 16B-aligned RSP.",
                self.total_adjustment,
                self.total_adjustment % 16,
            )));
        }
        self.asm.call(target).map_err(X86Lower::err)?;
        Ok(())
    }

    /// 自动生成恢复序列 (YMM load + add rsp + popfq + pop GPRs)。
    /// 调用者传入 save_gprs 和 save_ymms (与保存时相同顺序)。
    /// 不依赖 tracker 内部状态 — 可用于独立的恢复 frame 实例。
    pub fn restore_all(
        &mut self,
        gprs: &[AsmRegister64],
        ymms: &[AsmRegisterYmm],
    ) -> Result<(), CompilerError> {
        // Restore YMMs: load from stack then add rsp
        if !ymms.is_empty() {
            let stride: i32 = 32;
            for (i, &r) in ymms.iter().enumerate() {
                self.asm.vmovups(
                    r, ymmword_ptr(rsp + i as i32 * stride),
                ).map_err(X86Lower::err)?;
            }
            self.asm.add(rsp, (ymms.len() as i32) * stride).map_err(X86Lower::err)?;
        }
        // Restore RFLAGS + GPRs in reverse
        self.asm.popfq().map_err(X86Lower::err)?;
        for &r in gprs.iter().rev() {
            self.asm.pop(r).map_err(X86Lower::err)?;
        }
        Ok(())
    }

    /// 编译期断言：验证 save list 的栈对齐。
    /// 在构造 tracker 后、call_rax 前调用, catch 配置错误。
    pub fn verify_alignment(&self) -> Result<(), CompilerError> {
        // call rax 前应为 8-mod-16 (114 等等)
        if self.total_adjustment % 16 != 8 {
            return Err(CompilerError::CodegenViolation(format!(
                "SymbolicSaveFrame: total_adjustment={} ({} mod 16). \
                 Expected 8-mod-16 for NativeCall. \
                 Adjust GPR count, add/remove pushfq, or change YMM count.",
                self.total_adjustment,
                self.total_adjustment % 16,
            )));
        }
        Ok(())
    }
}

