//! x86_64 ISA Lower (REGISTER-VM SPEC §10)
//!
//! VmInstr → iced_x86 物理指令翻译。
//! 使用 RegAllocation 将 VRegId 映射到物理 ymm/GPR。
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 6 个片段):
//! - `x86_lower/helpers.inc.rs`         — 构造器 + resolve + spill helpers
//! - `x86_lower/lower_instr.inc.rs`     — lower_instr + lower_instr_inner (巨型 match)
//! - `x86_lower/emit_helpers.inc.rs`    — emit_fp4dot + emit_exp + fp8 + log
//! - `x86_lower/finalize_quant.inc.rs`  — finalize + gather/scatter + kivi + quant_load
//! - `x86_lower/callframe.inc.rs`       — SymbolicSaveFrame + CallFrame
//! - `x86_lower/tests.inc.rs`           — 测试模块

use std::collections::{HashMap, HashSet};
use iced_x86::code_asm::*;
use super::instr::*;
use super::isa_profile::*;
use super::reg_alloc::RegAllocation;
use super::stack_frame::StackFrame;
use crate::compiler::trace::DTypeKind;
use crate::compiler::trace::X86ElemStrategy;
use crate::types::CompilerError;

/// Scratch GPR slot 分配状态——追踪哪些 slot 正在使用。
#[derive(Debug)]
pub(crate) struct ScratchSlotState {
    in_use: Vec<bool>,
}

impl ScratchSlotState {
    pub fn new(num_slots: usize) -> Self {
        Self { in_use: vec![false; num_slots] }
    }
    pub fn alloc(&mut self) -> Option<usize> {
        for (i, used) in self.in_use.iter_mut().enumerate() {
            if !*used { *used = true; return Some(i); }
        }
        None
    }
    pub fn free(&mut self, slot: usize) {
        self.in_use[slot] = false;
    }
}

#[derive(Debug, Default)]
struct StackLayout {
    /// push rbp 后的 rbp 偏移（固定 = 8）
    frame_pointer_off: i32,
    /// callee-save 寄存器: [(物理寄存器, rbp偏移)]
    callee_save_slots: Vec<(PhysGpr, i32)>,
    /// ABI 入参: [(idx, rbp偏移)]
    abi_arg_slots: [Option<i32>; 6],
    /// Spill 区起始 rbp 偏移（相对 rbp 的负值）
    spill_base_off: i32,
    /// MXCSR rsp-relative offset (固定 = 0，即 [rsp])
    mxcsr_rsp_offset: i32,
    /// sub rsp 的总字节数（frame.total_size + MXCSR_SLOT_BYTES）
    rsp_sub_bytes: i32,
}

impl StackLayout {
    fn spill_rbp_offset(&self, spill_off: usize, spill_size: usize) -> i32 {
        self.spill_base_off - spill_off as i32 - spill_size as i32
    }
    fn abi_arg_rbp_offset(&self, idx: u8) -> Option<i32> {
        self.abi_arg_slots[idx as usize]
    }
}

const MXCSR_SLOT_BYTES: i32 = 8;

/// x86_64 ISA Lower。
pub struct X86Lower {
    pub asm: CodeAssembler,
    use_avx512: bool,
    const_pool: Vec<([f32; 8], CodeLabel)>,
    loop_stack: Vec<(CodeLabel, CodeLabel, AsmRegister64, Option<i32>, AsmRegister64, usize, Option<i32>)>,
    scope_saves: Vec<Vec<AsmRegister64>>,
    skip_stack: Vec<(usize, CodeLabel)>,
    stack_layout: StackLayout,
    amx_tile_dtype: Option<crate::types::DType>,
    jit_ctx: crate::compiler::jit_context::JitContext,
    sym_slot_map: super::plan_lower::SymDimSlotMap,
    scratch_gprs: Vec<AsmRegister64>,
    scratch_vec_ids: Vec<super::isa_profile::PhysVec>,
    epilogue_label: Option<CodeLabel>,
    dispatch_labels: HashMap<usize, CodeLabel>,
    source_map: super::debug_map::JitSourceMap,
    zero_vregs: HashSet<VRegId>,
}

include!("x86_lower/helpers.inc.rs");
include!("x86_lower/lower_instr.inc.rs");
include!("x86_lower/emit_helpers.inc.rs");
include!("x86_lower/finalize_quant.inc.rs");
include!("x86_lower/callframe.inc.rs");

#[cfg(test)]
include!("x86_lower/tests.inc.rs");
