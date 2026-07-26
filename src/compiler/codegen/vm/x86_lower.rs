//! x86_64 ISA Lower (REGISTER-VM SPEC §10)
//!
//! VmInstr → iced_x86 物理指令翻译。
//! 使用 RegAllocation 将 VRegId 映射到物理 ymm/GPR。
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 7 个片段):
//! - `x86_lower/helpers.inc.rs`         — 构造器 + resolve + spill helpers
//! - `x86_lower/lower_instr.inc.rs`     — lower_instr + lower_instr_inner (L0 分类 dispatch)
//! - `x86_lower/lower_instr_dispatch.inc.rs` — L1 变体路由 + L2 叶子 emit (ARCH-LOWER-DISPATCH-LAYERING)
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
use crate::compiler::trace::QuantPrecision;
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
    /// AVX-512 FP16 (vfmadd231ph) 探测结果 — NO-HW-DEGRADATION: 有 FP16 原生 FMA 必用之,
    /// 禁止软件 FMA 降级。由 Platform::X86_64 { has_avx512fp16, .. } 注入 (compile.inc.rs)。
    has_avx512fp16: bool,
    /// AVX-512 BF16 (VDPBF16PS / VCVTNEPS2BF16) 探测结果 — NO-SILENT-FALLBACK + NO-HW-DEGRADATION:
    /// BF16 是独立硬件特性 (Cooper Lake / Sapphire Rapids+), 非 AVX-512 子集。
    /// Ice Lake / Tiger Lake 客户端 CPU 有 AVX-512 但无 BF16 → 必须降级到 AVX2 软件序列,
    /// 禁止 emit vcvtneps2bf16/vdpbf16ps (会 SIGILL)。由 Platform::X86_64 { has_bf16, .. } 注入。
    has_bf16: bool,
    /// AVX-512 VNNI (VPDPBUSD) 探测结果 — NO-SILENT-FALLBACK: VNNI 是独立硬件特性,
    /// 非 AVX-512 子集。无 VNNI 时 INT8 dot product 无替代路径 → 必须 Err (NO-SILENT-FALLBACK)。
    /// 由 Platform::X86_64 { has_vnni, .. } 注入 (compile.inc.rs)。
    has_vnni: bool,
    const_pool: Vec<([f32; 8], CodeLabel)>,
    /// Arbitrary byte data tables (e.g. LoadLayerWeightOffset per-layer offset
    /// table) flushed AFTER `ret` in `finalize`, out of fall-through execution.
    ///
    /// ARCH-TABLE-OUT-OF-FALLTHROUGH: data tables baked into the code section
    /// must NOT be placed inline where execution would fall through into them.
    /// Unlike IndirectJump (whose `jmp [rax+idx*8]` never falls through), a
    /// `mov dst,[rax+idx*8]` load DOES fall through — baking the table right
    /// after the `mov` made the CPU execute the table bytes as instructions
    /// (BCE-20260715-SIGSEGV-TABLE-IN-FALLTHROUGH). Tables are emitted here and
    /// flushed after `ret` so they live at function end, unreachable by control
    /// flow. The label (forward reference) is bound in `finalize`.
    data_tables: Vec<(Vec<u8>, CodeLabel)>,
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
    /// VmInstr index 计数器 (每次 lower_instr 自增, 不改 lower_instr 签名 —
    /// 方案 (b), 最小侵入, 不波及 aarch64/gpu 对应方法)。
    vm_instr_counter: usize,
    /// VmInstr → iced 指令索引区间 (lower_instr 成功路径记录)。
    /// finalize 中 assemble_options(RETURN_NEW_INSTRUCTION_OFFSETS) 后,
    /// 用 new_instruction_offsets 把 [pre_iced_idx, post_iced_idx) 转成字节偏移。
    vm_instr_offsets: Vec<super::debug_map::VmInstrOffsetEntry>,
    zero_vregs: HashSet<VRegId>,
    /// ARCH-SPILL-SAFE-ISA: Maps VRegId → immutable reload recipe.
    /// Under extreme register pressure (4120+ spills), spill slots can be corrupted by other VReg
    /// writes due to ScopedSpillAllocator slot reuse. When `resolve_gpr_read` encounters a spilled
    /// VReg present in this map, it recomputes the value from the immutable ABI stack slot instead
    /// of reading the potentially-corrupted spill slot. This is the ISA-level fix — the VmInstr-level
    /// ARCH-SPILL-SAFE reloads in materialize()/load_op_scratch_ptr() are insufficient because
    /// regalloc also spills the freshly-allocated VReg.
    stack_arg_vregs: HashMap<VRegId, SpillSafeRecipe>,
}

/// Recipe for recomputing a VReg's value from immutable ABI stack slots.
/// ARCH-SPILL-SAFE-ISA: When a VReg is spilled and its spill slot may be corrupted,
/// we use this recipe to reload from the immutable source instead.
#[derive(Debug, Clone)]
enum SpillSafeRecipe {
    /// VReg = *(rbp + offset) — direct load from immutable stack slot.
    /// Used for LoadPtr { src: StackArg(off) } and LoadPtr { src: AbiArg(idx) }.
    StackLoad { rbp_offset: i32 },
    /// VReg = *(rbp + base_offset) + const_offset — load base from immutable slot, add offset.
    /// Used for LoadPtr { src: VRegPlusConst(base, off) } where base is StackArg-derived,
    /// and for AddPtr { base, offset } where base is StackArg-derived.
    StackLoadPlusConst { rbp_offset: i32, const_offset: usize },
}

include!("x86_lower/helpers.inc.rs");
include!("x86_lower/lower_instr.inc.rs");
include!("x86_lower/lower_instr_dispatch.inc.rs");
include!("x86_lower/emit_helpers.inc.rs");
include!("x86_lower/finalize_quant.inc.rs");
include!("x86_lower/callframe.inc.rs");

#[cfg(test)]
include!("x86_lower/tests.inc.rs");
