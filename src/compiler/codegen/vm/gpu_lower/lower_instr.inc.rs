impl GpuLower {
    // ARCH-LOWER-DISPATCH-LAYERING (BCE-20260630-LOWER-INSTR-GOD-MATCH):
    // L0 分类 dispatch — 巨型 VmInstr match 已拆为 L0(分类) → L1(变体路由) → L2(叶子 emit)。
    // 新增 VmInstr 变体 = category() 补一行 + lower_<variant>_gpu 一个叶子 fn。
    // 禁止在此处内联变体逻辑 (OCP); 禁止 catch-all 静默 NOP (NO_SILENT_FALLBACK)。
    // L1 变体路由 + L2 叶子 emit 见 lower_instr_dispatch.inc.rs。

    pub fn lower_instr(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match instr.category() {
            super::vm_instr_category::InstrCategory::Memory => self.lower_memory_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::Arith => self.lower_arith_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::Control => self.lower_control_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::Tile => self.lower_tile_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::Quant => self.lower_quant_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::GpuComm => self.lower_gpu_comm_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::Sampling => self.lower_sampling_gpu(instr, alloc),
            super::vm_instr_category::InstrCategory::Misc => self.lower_misc_gpu(instr, alloc),
        }
    }
}
