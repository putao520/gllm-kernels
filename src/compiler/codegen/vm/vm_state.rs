//! VM 状态机 (ARCH-VM-STATE-TRACKING)
//!
//! 半虚拟化核心：在 codegen 过程中实时跟踪物理布局。
//! 所有参数偏移、寄存器映射、栈帧位置由 VmState 计算。
//! 禁止在任何文件中硬编码栈偏移数字。
//!
//! §7: HeteroPhase + EmitState — plan_lower 跨迭代状态追踪
//! §7.3: HeteroPhasePlan — 异构层模板阶段预计算

use std::collections::HashMap;
use super::instr::PtrExpr;
use super::isa_profile::PhysGpr;
use crate::types::CompilerError;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 参数物理位置
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ABI 参数的物理位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgLocation {
    /// 在寄存器中 (x86 SysV: 前 6 个整数/指针参数)
    Register(u8),  // 寄存器 ABI 编号 (0=rdi, 1=rsi, ...)
    /// 在栈上 (第 7+ 个参数)
    /// offset 是相对 rbp 的正偏移 (prologue 后)
    Stack(i32),
    /// 片上共享内存 (GPU scratchpad 专用)。
    /// ARCH-GPU-SHARED-SCRATCH: GPU kernel 的 scratchpad 不经 `.param`,
    /// 由 prologue 的 `.shared`/`__shared__` 声明提供;codegen 通过符号名访问。
    SharedMem,
}

impl ArgLocation {
    /// 转换为 VM IR 的 PtrExpr。
    pub fn to_ptr_expr(self) -> PtrExpr {
        match self {
            ArgLocation::Register(idx) => PtrExpr::AbiArg(idx),
            ArgLocation::Stack(off) => PtrExpr::StackArg(off),
            ArgLocation::SharedMem => PtrExpr::SharedMem,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 VmState
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// VM 代码生成状态机。
///
/// 在 prologue → body → epilogue 的整个生成过程中持续更新。
/// 所有物理位置（参数偏移、寄存器映射、栈帧布局）从此结构查询。
///
/// ## 使用流程
/// ```text
/// let vm_state = VmState::init_mega_kernel_x86();     // ABI 初始化
/// lowerer.emit_prologue(&mut vm_state, ...);     // prologue 更新状态
/// let sym_map = SymDimSlotMap::from_vm_state(&vm_state);  // 构建 SlotMap
/// // lower 函数通过 sym_map 查询参数位置
/// ```
#[derive(Debug, Clone)]
pub struct VmState {
    /// ABI 参数位置表。
    /// key = 参数语义名称 ("input"/"output"/"seq_len"/...)
    /// value = 相对 rbp 的物理位置
    arg_locations: HashMap<String, ArgLocation>,

    /// 当前 rsp 相对 rbp 的偏移。
    /// prologue `push rbp; mov rbp, rsp` 后 = 0。
    /// 每次 push → -8，每次 sub rsp,N → -N。
    pub rsp_offset: i32,

    /// Callee-saved 寄存器保存位置 (push 顺序)
    pub callee_save_locations: Vec<(PhysGpr, i32)>,

    /// Spill 区域起始偏移（相对 rbp，负值）
    pub spill_base: i32,
}

impl VmState {
    /// 从 GPU kernel ABI 初始化（所有参数通过 `.param` 传递，无栈参数）。
    ///
    /// PTX/HIP/MSL 都将 kernel 参数放在设备专用的参数区（PTX: `.param`，HIP: __global__ args，
    /// MSL: `[[buffer(N)]]`），按声明顺序依次编号。ARCH-GPU-ABI 定义标准参数顺序。
    ///
    /// GPU kernel 不处理 kv_cache/positions/seq_lens/batch_size/scratchpad
    /// （这些是 CPU 推理调度器的概念）—— 仅暴露最小 5 参数集。
    pub fn init_gpu_kernel() -> Self {
        let mut arg_locations = HashMap::new();
        // 与 GpuLower::emit_prologue 声明的 .param 顺序严格一致
        arg_locations.insert("input".into(), ArgLocation::Register(0));    // input_ptr
        arg_locations.insert("weights".into(), ArgLocation::Register(1));  // weight_ptr
        arg_locations.insert("output".into(), ArgLocation::Register(2));   // output_ptr
        arg_locations.insert("seq_len".into(), ArgLocation::Register(3));
        arg_locations.insert("telemetry".into(), ArgLocation::Register(4));
        // ARCH-GPU-SHARED-SCRATCH: GPU scratchpad = 片上 shared memory,
        // 由 gpu_lower emit_prologue 的 `.shared`/`__shared__` 声明提供,
        // 非 `.param` 入参。LoadPtr 降低为对 `smem` 符号的 address-of。
        arg_locations.insert("scratchpad".into(), ArgLocation::SharedMem);
        Self {
            arg_locations,
            rsp_offset: 0,
            callee_save_locations: Vec::new(),
            spill_base: 0,
        }
    }

    /// 从 MegaKernelFn ABI 初始化 (16 参数: 6 寄存器 + 10 栈参数)。
    ///
    /// 栈参数严格 8 字节对齐 (SysV ABI: 每个参数占一个 eightbyte 槽位)。
    /// f32 已改为 u32 传递 (避免 SSE 寄存器传参导致 JIT 无法从栈读取)。
    pub fn init_mega_kernel_x86() -> Self {
        let param_names: &[&str] = &[
            "input_ids_ptr",        // arg 0 → AbiArg(0)
            "weight_blob_ptr",      // arg 1 → AbiArg(1)
            "kv_cache_ptr",         // arg 2 → AbiArg(2)
            "positions_ptr",        // arg 3 → AbiArg(3)
            "aux_ptr",              // arg 4 → AbiArg(4)
            "batch_size",           // arg 5 → AbiArg(5)
            "prompt_len",           // arg 6 → StackArg(16)
            "scratchpad_ptr",       // arg 7 → StackArg(24)
            "output_tokens_ptr",    // arg 8 → StackArg(32)
            "temperature_u32",      // arg 9 → StackArg(40)
            "top_k",                // arg 10 → StackArg(48)
            "top_p_u32",            // arg 11 → StackArg(56)
            "max_new_tokens",       // arg 12 → StackArg(64)
            "eos_token_id",         // arg 13 → StackArg(72)
            "hook_ctx_ptr",         // arg 14 → StackArg(80)
            "telemetry_ptr",        // arg 15 → StackArg(88)
            "session_position",     // arg 16 → StackArg(96)
            "fused_hidden_ptr",     // arg 17 → StackArg(104)
            "num_mm_tokens",        // arg 18 → StackArg(112)
            "callback_table_ptr",   // arg 19 → StackArg(120)
            "page_table_ptr",       // arg 20 → StackArg(128)
        ];

        let num_reg_args: usize = 6;
        let stack_arg_base: i32 = 16; // ret_addr(8) + saved_rbp(8)

        let mut arg_locations = HashMap::new();
        for (i, &name) in param_names.iter().enumerate() {
            let loc = if i < num_reg_args {
                ArgLocation::Register(i as u8)
            } else {
                let stack_index = (i - num_reg_args) as i32;
                ArgLocation::Stack(stack_arg_base + stack_index * 8)
            };
            arg_locations.insert(name.to_string(), loc);
        }

        Self {
            arg_locations,
            rsp_offset: 0,
            callee_save_locations: Vec::new(),
            spill_base: 0,
        }
    }

    /// 查询参数的物理位置 (ARCH-VM-QUERY-NOT-ASSUME)。
    ///
    /// 返回 PtrExpr 供 VmInstr::LoadPtr 使用。
    /// 查询失败返回 Err——禁止 fallback 到默认值。
    pub fn arg_ptr_expr(&self, name: &str) -> Result<PtrExpr, CompilerError> {
        let loc = self.arg_locations.get(name)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("VmState: unknown parameter '{}'. Available: {:?}",
                    name, self.arg_locations.keys().collect::<Vec<_>>())
            ))?;
        Ok(loc.to_ptr_expr())
    }

    /// 查询参数位置（返回 Option，不报错）。
    pub fn try_arg(&self, name: &str) -> Option<PtrExpr> {
        self.arg_locations.get(name).map(|loc| loc.to_ptr_expr())
    }

    // ── Prologue 状态跟踪 ──

    /// 记录一次 push 操作。rsp -= 8。
    pub fn track_push(&mut self, reg: PhysGpr) {
        self.rsp_offset -= 8;
        self.callee_save_locations.push((reg, self.rsp_offset));
    }

    /// 记录 sub rsp, N 操作。
    pub fn track_sub_rsp(&mut self, n: usize) {
        self.rsp_offset -= n as i32;
    }

    /// 设置 spill 区域基址。
    pub fn set_spill_base(&mut self, callee_save_count: usize) {
        self.spill_base = -(callee_save_count as i32 * 8);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 SymDim 别名映射
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl VmState {
    /// 符号维度名称的别名映射。
    /// "total_seq" 复用 "seq_len" 的位置（生成循环模式）。
    pub fn sym_dim_aliases() -> Vec<(&'static str, &'static str)> {
        vec![
            ("total_seq", "seq_len"),
        ]
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7 异构层模板状态追踪 (W7: 从 plan_lower.rs 提取)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Phase tracker for heterogeneous layer loop (Gemma-4 E2B: alternating sliding/full).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeteroPhase {
    BeforeLayers,
    SmallOuterLoop,
    InSlidingLoop,
    InFullBody,
    LargeOuterLoop,
    InLargeSlidingLoop,
    InLargeFullBody,
    Done,
}

/// §7.3 异构层模板阶段预计算 (W7: R5 输出)
///
/// plan_lower 在编译前预计算每个融合组的阶段转换点，
/// 运行时只读取预计算的状态序列。
#[derive(Debug, Clone)]
pub struct HeteroPhasePlan {
    /// (OpId, HeteroPhase) — 在哪个 op 处于哪个阶段
    pub transitions: Vec<(usize, HeteroPhase)>,
}

impl HeteroPhasePlan {
    /// 空计划 (同构模型)
    pub fn empty() -> Self {
        HeteroPhasePlan { transitions: Vec::new() }
    }

    /// 是否为空 (同构模型)
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7.2 EmitState — emit_fusion_groups 跨迭代可变状态 (W7: 从 plan_lower.rs 提取)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ABI 入参已 LoadPtr 后的 VReg 集合。
#[derive(Debug, Clone)]
pub struct AbiPtrs {
    pub input_ptr: super::instr::VRegId,
    pub weight_ptr: Option<super::instr::VRegId>,
    /// Weight ABI PtrExpr — always available regardless of weight_ptr VReg state.
    /// Used by materialize to emit fresh LoadPtr from ABI stack slot for Weight sources,
    /// avoiding stale physical register values after long loops.
    pub weight_abi_expr: Option<super::instr::PtrExpr>,
    pub output_ptr: super::instr::VRegId,
    pub scratch_ptr: Option<super::instr::VRegId>,
    /// Generate loop counter VReg (mega-kernel only).
    pub gen_loop_counter: Option<super::instr::VRegId>,
    /// Layer loop counter VReg (mega-kernel only).
    pub layer_loop_counter: Option<super::instr::VRegId>,
    /// Mega-kernel decode seq_len VReg: prompt_len + gen_counter.
    pub mega_decode_seq_len: Option<super::instr::VRegId>,
    /// Mega-kernel hook_ctx_ptr: SG shared memory base address.
    pub hook_ctx_ptr: Option<super::instr::VRegId>,
    /// SG detect scratchpad offset.
    pub sg_detect_scratch_offset: Option<usize>,
    /// SG knowledge scratchpad offset.
    pub sg_knowledge_scratch_offset: Option<usize>,
    /// Mega-kernel callback table pointer (ABI arg 20).
    pub callback_table_ptr: Option<super::instr::VRegId>,
    /// Mega-kernel page_table_ptr (ABI arg 21). NULL = contiguous KV, u32[] = paged KV.
    pub page_table_ptr: Option<super::instr::VRegId>,
    /// KV cache load mode for attention variant (KV-OPT-009).
    /// None = Direct (default). Some(mode) = use specified variant for K/V load.
    pub kv_load_mode: Option<super::instr::KvLoadMode>,
    /// Mega-kernel KV cache pool base pointer (ABI arg 2, kv_cache_ptr).
    /// Used by MHA contiguous path to read/write KV cache.
    pub kv_cache_ptr: Option<super::instr::VRegId>,
    /// §0.2.8 Ping-pong activation buffer ptrs (mega-kernel layer loop).
    /// ping_ptr: scratch_base + ping_offset, pong_ptr: scratch_base + pong_offset.
    /// ActivationSwap swaps these at each layer iteration boundary.
    pub activation_ping_ptr: Option<super::instr::VRegId>,
    pub activation_pong_ptr: Option<super::instr::VRegId>,
}

/// emit_fusion_groups 内部跨迭代可变状态。
///
/// 抽取自 emit_fusion_groups 的 5 个可变局部变量，使状态转移显式化，
/// 为层模板并行编译提供独立状态快照。
pub struct EmitState {
    /// ABI 指针集 — 层间 output→input 交换
    pub abi: AbiPtrs,
    /// 异构层模板阶段追踪
    pub hetero_phase: HeteroPhase,
    /// 是否已进入层循环
    pub in_layer_loop: bool,
    /// 异构段偏移 VReg
    pub hetero_seg_byte_offset: Option<super::instr::VRegId>,
    /// 异构段权重基址 VReg
    pub hetero_seg_weight_base: Option<super::instr::VRegId>,
    /// 全局 layer_idx VReg（异构层中用于 GprCondAction）
    pub hetero_global_layer_idx: Option<super::instr::VRegId>,
    /// 外层段循环 counter VReg（异构层 full body 需要从段索引推导全局 layer_idx）
    pub hetero_outer_seg_counter: Option<super::instr::VRegId>,
    /// Current active layer guard (for guard run merging).
    /// `Always` = no guard active. When consecutive ops share the same guard,
    /// only one GprCondAction is emitted covering all of them.
    pub active_guard: crate::compiler::graph::LayerCondition,
    /// Index of the pending `GprCondAction { Skip(0) }` instruction to patch-back
    /// once the guard run ends. `None` = no guard run in progress.
    pub guard_skip_patch: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_mega_kernel_x86_register_args() {
        let state = VmState::init_mega_kernel_x86();
        // 前 6 个参数在寄存器
        assert_eq!(state.arg_ptr_expr("input_ids_ptr").unwrap(), PtrExpr::AbiArg(0));
        assert_eq!(state.arg_ptr_expr("weight_blob_ptr").unwrap(), PtrExpr::AbiArg(1));
        assert_eq!(state.arg_ptr_expr("kv_cache_ptr").unwrap(), PtrExpr::AbiArg(2));
        assert_eq!(state.arg_ptr_expr("positions_ptr").unwrap(), PtrExpr::AbiArg(3));
        assert_eq!(state.arg_ptr_expr("aux_ptr").unwrap(), PtrExpr::AbiArg(4));
        assert_eq!(state.arg_ptr_expr("batch_size").unwrap(), PtrExpr::AbiArg(5));
    }

    #[test]
    fn test_unknown_param_returns_error() {
        let state = VmState::init_mega_kernel_x86();
        assert!(state.arg_ptr_expr("nonexistent").is_err());
    }

    #[test]
    fn test_prologue_tracking() {
        let mut state = VmState::init_mega_kernel_x86();
        assert_eq!(state.rsp_offset, 0);

        state.track_push(PhysGpr(3)); // rbx
        assert_eq!(state.rsp_offset, -8);
        assert_eq!(state.callee_save_locations.len(), 1);

        state.track_push(PhysGpr(12)); // r12
        assert_eq!(state.rsp_offset, -16);

        state.track_sub_rsp(64); // sub rsp, 64
        assert_eq!(state.rsp_offset, -80);

        // 参数位置不受 prologue 操作影响（相对 rbp）
        assert_eq!(state.arg_ptr_expr("scratchpad_ptr").unwrap(), PtrExpr::StackArg(24));
    }

    // ── New tests (TEST-VMS-06 .. TEST-VMS-18) ──

    /// @trace TEST-VMS-06 [req:REQ-VR] [level:unit]
    /// ArgLocation::to_ptr_expr maps all three variants correctly.
    #[test]
    fn test_arg_location_to_ptr_expr_all_variants() {
        // Arrange
        let reg = ArgLocation::Register(4);
        let stack = ArgLocation::Stack(56);
        let shared = ArgLocation::SharedMem;

        // Act & Assert
        assert_eq!(reg.to_ptr_expr(), PtrExpr::AbiArg(4));
        assert_eq!(stack.to_ptr_expr(), PtrExpr::StackArg(56));
        assert_eq!(shared.to_ptr_expr(), PtrExpr::SharedMem);
    }

    /// @trace TEST-VMS-07 [req:REQ-VR] [level:unit]
    /// init_gpu_kernel maps 5 Register args + scratchpad as SharedMem;
    /// CPU-only params (kv_cache, positions, seq_lens, batch_size) are absent.
    #[test]
    fn test_init_gpu_kernel_param_layout() {
        // Arrange
        let state = VmState::init_gpu_kernel();

        // Act & Assert — 5 params via register
        assert_eq!(state.arg_ptr_expr("input").unwrap(), PtrExpr::AbiArg(0));
        assert_eq!(state.arg_ptr_expr("weights").unwrap(), PtrExpr::AbiArg(1));
        assert_eq!(state.arg_ptr_expr("output").unwrap(), PtrExpr::AbiArg(2));
        assert_eq!(state.arg_ptr_expr("seq_len").unwrap(), PtrExpr::AbiArg(3));
        assert_eq!(state.arg_ptr_expr("telemetry").unwrap(), PtrExpr::AbiArg(4));

        // SharedMem scratchpad
        assert_eq!(state.arg_ptr_expr("scratchpad").unwrap(), PtrExpr::SharedMem);

        // CPU-only params absent
        assert!(state.arg_ptr_expr("kv_cache").is_err());
        assert!(state.arg_ptr_expr("positions").is_err());
        assert!(state.arg_ptr_expr("seq_lens").is_err());
        assert!(state.arg_ptr_expr("batch_size").is_err());

        // No prologue state mutated
        assert_eq!(state.rsp_offset, 0);
        assert!(state.callee_save_locations.is_empty());
        assert_eq!(state.spill_base, 0);
    }

    /// @trace TEST-VMS-08 [req:REQ-VR] [level:unit]
    /// init_mega_kernel_x86 maps 22 params: first 6 Register, rest StackArg.
    #[test]
    fn test_init_mega_kernel_x86_register_and_first_stack_args() {
        // Arrange
        let state = VmState::init_mega_kernel_x86();

        // Act & Assert — first 6 in registers
        assert_eq!(state.arg_ptr_expr("input_ids_ptr").unwrap(), PtrExpr::AbiArg(0));
        assert_eq!(state.arg_ptr_expr("weight_blob_ptr").unwrap(), PtrExpr::AbiArg(1));
        assert_eq!(state.arg_ptr_expr("kv_cache_ptr").unwrap(), PtrExpr::AbiArg(2));
        assert_eq!(state.arg_ptr_expr("positions_ptr").unwrap(), PtrExpr::AbiArg(3));
        assert_eq!(state.arg_ptr_expr("aux_ptr").unwrap(), PtrExpr::AbiArg(4));
        assert_eq!(state.arg_ptr_expr("batch_size").unwrap(), PtrExpr::AbiArg(5));

        // First stack args — stack_arg_base=16, stride=8
        assert_eq!(state.arg_ptr_expr("prompt_len").unwrap(), PtrExpr::StackArg(16));
        assert_eq!(state.arg_ptr_expr("scratchpad_ptr").unwrap(), PtrExpr::StackArg(24));
        assert_eq!(state.arg_ptr_expr("output_tokens_ptr").unwrap(), PtrExpr::StackArg(32));
    }

    /// @trace TEST-VMS-09 [req:REQ-VR] [level:unit]
    /// init_mega_kernel_x86 last stack args at correct offsets, including
    /// page_table_ptr at arg 21 → StackArg(136).
    #[test]
    fn test_init_mega_kernel_x86_last_stack_args() {
        // Arrange
        let state = VmState::init_mega_kernel_x86();

        // Act & Assert — later stack args
        assert_eq!(state.arg_ptr_expr("temperature_u32").unwrap(), PtrExpr::StackArg(40));
        assert_eq!(state.arg_ptr_expr("top_k").unwrap(), PtrExpr::StackArg(48));
        assert_eq!(state.arg_ptr_expr("top_p_u32").unwrap(), PtrExpr::StackArg(56));
        assert_eq!(state.arg_ptr_expr("max_new_tokens").unwrap(), PtrExpr::StackArg(64));
        assert_eq!(state.arg_ptr_expr("eos_token_id").unwrap(), PtrExpr::StackArg(72));
        assert_eq!(state.arg_ptr_expr("hook_ctx_ptr").unwrap(), PtrExpr::StackArg(80));
        assert_eq!(state.arg_ptr_expr("telemetry_ptr").unwrap(), PtrExpr::StackArg(88));
        assert_eq!(state.arg_ptr_expr("session_position").unwrap(), PtrExpr::StackArg(96));
        assert_eq!(state.arg_ptr_expr("fused_hidden_ptr").unwrap(), PtrExpr::StackArg(104));
        assert_eq!(state.arg_ptr_expr("num_mm_tokens").unwrap(), PtrExpr::StackArg(112));
        assert_eq!(state.arg_ptr_expr("callback_table_ptr").unwrap(), PtrExpr::StackArg(120));
        assert_eq!(state.arg_ptr_expr("page_table_ptr").unwrap(), PtrExpr::StackArg(128));
    }

    /// @trace TEST-VMS-10 [req:REQ-VR] [level:unit]
    /// arg_ptr_expr error message contains the unknown parameter name.
    #[test]
    fn test_arg_ptr_expr_error_contains_param_name() {
        // Arrange
        let state = VmState::init_mega_kernel_x86();

        // Act
        let result = state.arg_ptr_expr("bogus_param_xyz");

        // Assert
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("bogus_param_xyz"),
            "error should mention the unknown param name, got: {err_msg}"
        );
    }

    /// @trace TEST-VMS-11 [req:REQ-VR] [level:unit]
    /// try_arg returns Some for known params and None for unknown.
    #[test]
    fn test_try_arg_found_and_not_found() {
        // Arrange
        let state = VmState::init_mega_kernel_x86();

        // Act & Assert — known params
        assert_eq!(state.try_arg("input_ids_ptr"), Some(PtrExpr::AbiArg(0)));
        assert_eq!(state.try_arg("telemetry_ptr"), Some(PtrExpr::StackArg(88)));

        // Unknown param returns None (not an error)
        assert_eq!(state.try_arg("does_not_exist"), None);
    }

    /// @trace TEST-VMS-12 [req:REQ-VR] [level:unit]
    /// track_push records multiple callee-saved registers with correct offsets.
    #[test]
    fn test_track_push_multiple_callee_saves() {
        // Arrange
        let mut state = VmState::init_mega_kernel_x86();

        // Act — push rbx, r12, r13
        state.track_push(PhysGpr(3));  // rbx
        state.track_push(PhysGpr(12)); // r12
        state.track_push(PhysGpr(13)); // r13

        // Assert
        assert_eq!(state.rsp_offset, -24);
        assert_eq!(state.callee_save_locations.len(), 3);
        assert_eq!(state.callee_save_locations[0], (PhysGpr(3), -8));
        assert_eq!(state.callee_save_locations[1], (PhysGpr(12), -16));
        assert_eq!(state.callee_save_locations[2], (PhysGpr(13), -24));
    }

    /// @trace TEST-VMS-13 [req:REQ-VR] [level:unit]
    /// set_spill_base computes -(count * 8).
    #[test]
    fn test_set_spill_base_with_callee_save_count() {
        // Arrange
        let mut state = VmState::init_mega_kernel_x86();
        state.track_push(PhysGpr(3));
        state.track_push(PhysGpr(12));

        // Act — 2 callee-saved regs
        state.set_spill_base(2);

        // Assert
        assert_eq!(state.spill_base, -16);
    }

    /// @trace TEST-VMS-14 [req:REQ-VR] [level:unit]
    /// track_sub_rsp works independently and combined with track_push;
    /// arg queries remain stable relative to rbp.
    #[test]
    fn test_track_sub_rsp_combined_with_push() {
        // Arrange
        let mut state = VmState::init_mega_kernel_x86();
        state.track_push(PhysGpr(3));  // rsp = -8
        state.track_push(PhysGpr(12)); // rsp = -16

        // Act
        state.track_sub_rsp(256); // rsp = -16 - 256 = -272

        // Assert
        assert_eq!(state.rsp_offset, -272);

        // Arg locations are still relative to rbp, unaffected
        assert_eq!(state.arg_ptr_expr("input_ids_ptr").unwrap(), PtrExpr::AbiArg(0));
        assert_eq!(state.arg_ptr_expr("telemetry_ptr").unwrap(), PtrExpr::StackArg(88));
    }

    /// @trace TEST-VMS-15 [req:REQ-VR] [level:unit]
    /// sym_dim_aliases returns ("total_seq", "seq_len") alias pair.
    #[test]
    fn test_sym_dim_aliases_contains_total_seq_alias() {
        // Arrange
        let aliases = VmState::sym_dim_aliases();

        // Assert
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0], ("total_seq", "seq_len"));
    }

    /// @trace TEST-VMS-16 [req:REQ-VR] [level:unit]
    /// HeteroPhase variants are distinct and PartialEq works correctly.
    #[test]
    fn test_hetero_phase_equality_and_distinction() {
        // Arrange
        let phases = [
            HeteroPhase::BeforeLayers,
            HeteroPhase::SmallOuterLoop,
            HeteroPhase::InSlidingLoop,
            HeteroPhase::InFullBody,
            HeteroPhase::LargeOuterLoop,
            HeteroPhase::InLargeSlidingLoop,
            HeteroPhase::InLargeFullBody,
            HeteroPhase::Done,
        ];

        // Act & Assert — all 8 variants are distinct
        assert_eq!(phases.len(), 8);
        for i in 0..phases.len() {
            for j in 0..phases.len() {
                if i == j {
                    assert_eq!(phases[i], phases[j]);
                } else {
                    assert_ne!(phases[i], phases[j]);
                }
            }
        }
    }

    /// @trace TEST-VMS-17 [req:REQ-VR] [level:unit]
    /// HeteroPhasePlan::empty() has is_empty() == true; adding a transition
    /// makes is_empty() == false.
    #[test]
    fn test_hetero_phase_plan_empty_and_nonempty() {
        // Arrange
        let empty_plan = HeteroPhasePlan::empty();

        // Assert — empty
        assert!(empty_plan.is_empty());
        assert!(empty_plan.transitions.is_empty());

        // Arrange — non-empty
        let nonempty = HeteroPhasePlan {
            transitions: vec![
                (0, HeteroPhase::BeforeLayers),
                (3, HeteroPhase::InSlidingLoop),
                (7, HeteroPhase::Done),
            ],
        };

        // Assert — not empty
        assert!(!nonempty.is_empty());
        assert_eq!(nonempty.transitions.len(), 3);
        assert_eq!(nonempty.transitions[1].0, 3);
        assert_eq!(nonempty.transitions[1].1, HeteroPhase::InSlidingLoop);
    }

    /// @trace TEST-VMS-18 [req:REQ-VR] [level:unit]
    /// VmState::clone produces an independent snapshot; modifying the clone
    /// does not affect the original.
    #[test]
    fn test_vm_state_clone_independence() {
        // Arrange
        let mut original = VmState::init_mega_kernel_x86();
        original.track_push(PhysGpr(3));
        original.track_sub_rsp(64);
        original.set_spill_base(1);

        // Act — clone and modify
        let mut cloned = original.clone();
        cloned.track_push(PhysGpr(12));
        cloned.track_sub_rsp(128);

        // Assert — original untouched
        assert_eq!(original.rsp_offset, -8 - 64); // -72
        assert_eq!(original.callee_save_locations.len(), 1);

        // Assert — clone has extra state
        assert_eq!(cloned.rsp_offset, -8 - 64 - 8 - 128); // -208
        assert_eq!(cloned.callee_save_locations.len(), 2);
    }

    // ── Wave-12kgd tests (TEST-VMS-19 .. TEST-VMS-28) ──

    /// @trace TEST-VMS-19 [req:REQ-VR] [level:unit]
    /// track_sub_rsp(0) is a no-op on rsp_offset; arg queries remain valid.
    #[test]
    fn test_track_sub_rsp_zero_is_noop() {
        // Arrange
        let mut state = VmState::init_mega_kernel_x86();
        assert_eq!(state.rsp_offset, 0);

        // Act — subtract zero
        state.track_sub_rsp(0);

        // Assert — unchanged
        assert_eq!(state.rsp_offset, 0);
        assert_eq!(state.arg_ptr_expr("input_ids_ptr").unwrap(), PtrExpr::AbiArg(0));
    }

    /// @trace TEST-VMS-20 [req:REQ-VR] [level:unit]
    /// set_spill_base(0) sets spill_base to 0, representing no callee saves.
    #[test]
    fn test_set_spill_base_zero_count() {
        // Arrange
        let mut state = VmState::init_mega_kernel_x86();

        // Act — zero callee-saved regs
        state.set_spill_base(0);

        // Assert
        assert_eq!(state.spill_base, 0);
    }

    /// @trace TEST-VMS-21 [req:REQ-VR] [level:unit]
    /// init_mega_kernel_x86 rejects unknown parameter names with an error.
    #[test]
    fn test_mega_kernel_unknown_param_returns_error() {
        // Arrange
        let state = VmState::init_mega_kernel_x86();

        // Act
        let result = state.arg_ptr_expr("nonexistent_mega_param");

        // Assert
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("nonexistent_mega_param"),
            "error should mention the unknown param, got: {err_msg}"
        );
    }

    /// @trace TEST-VMS-22 [req:REQ-VR] [level:unit]
    /// init_gpu_kernel try_arg returns Some for known params, None for unknown.
    #[test]
    fn test_gpu_kernel_try_arg_semantics() {
        // Arrange
        let state = VmState::init_gpu_kernel();

        // Act & Assert — known params via try_arg
        assert_eq!(state.try_arg("input"), Some(PtrExpr::AbiArg(0)));
        assert_eq!(state.try_arg("weights"), Some(PtrExpr::AbiArg(1)));
        assert_eq!(state.try_arg("output"), Some(PtrExpr::AbiArg(2)));
        assert_eq!(state.try_arg("seq_len"), Some(PtrExpr::AbiArg(3)));
        assert_eq!(state.try_arg("telemetry"), Some(PtrExpr::AbiArg(4)));
        assert_eq!(state.try_arg("scratchpad"), Some(PtrExpr::SharedMem));

        // CPU-only params absent via try_arg
        assert_eq!(state.try_arg("kv_cache"), None);
        assert_eq!(state.try_arg("positions"), None);
        assert_eq!(state.try_arg("batch_size"), None);

        // Fully unknown param
        assert_eq!(state.try_arg("bogus"), None);
    }

    /// @trace TEST-VMS-23 [req:REQ-VR] [level:unit]
    /// ArgLocation Copy trait allows duplicating values without mutation.
    #[test]
    fn test_arg_location_copy_trait() {
        // Arrange
        let original = ArgLocation::Stack(48);
        let copied = original; // Copy, not move

        // Act — both should be usable independently
        let expr1 = original.to_ptr_expr();
        let expr2 = copied.to_ptr_expr();

        // Assert
        assert_eq!(expr1, PtrExpr::StackArg(48));
        assert_eq!(expr2, PtrExpr::StackArg(48));
    }

    /// @trace TEST-VMS-24 [req:REQ-VR] [level:unit]
    /// track_push interleaved with track_sub_rsp produces correct cumulative offset.
    #[test]
    fn test_interleaved_push_and_sub_rsp() {
        // Arrange
        let mut state = VmState::init_mega_kernel_x86();

        // Act — push, sub, push, sub
        state.track_push(PhysGpr(3));   // rsp = -8
        state.track_sub_rsp(32);         // rsp = -40
        state.track_push(PhysGpr(12));   // rsp = -48
        state.track_sub_rsp(64);         // rsp = -112

        // Assert
        assert_eq!(state.rsp_offset, -112);
        assert_eq!(state.callee_save_locations.len(), 2);
        assert_eq!(state.callee_save_locations[0], (PhysGpr(3), -8));
        assert_eq!(state.callee_save_locations[1], (PhysGpr(12), -48));
    }

    /// @trace TEST-VMS-25 [req:REQ-VR] [level:unit]
    /// HeteroPhase Copy trait: assigning one variant to another variable copies it.
    #[test]
    fn test_hetero_phase_copy_trait() {
        // Arrange
        let phase = HeteroPhase::InFullBody;
        let copied = phase; // Copy, not move

        // Act & Assert — both usable independently
        assert_eq!(phase, HeteroPhase::InFullBody);
        assert_eq!(copied, HeteroPhase::InFullBody);
    }

    /// @trace TEST-VMS-26 [req:REQ-VR] [level:unit]
    /// HeteroPhasePlan Clone produces an independent copy; modifying transitions
    /// in the clone does not affect the original.
    #[test]
    fn test_hetero_phase_plan_clone_independence() {
        // Arrange
        let original = HeteroPhasePlan {
            transitions: vec![
                (0, HeteroPhase::BeforeLayers),
                (5, HeteroPhase::InSlidingLoop),
            ],
        };

        // Act — clone and modify
        let mut cloned = original.clone();
        cloned.transitions.push((10, HeteroPhase::Done));

        // Assert — original has 2 transitions, clone has 3
        assert_eq!(original.transitions.len(), 2);
        assert_eq!(cloned.transitions.len(), 3);
        assert_eq!(cloned.transitions[2], (10, HeteroPhase::Done));
    }

    /// @trace TEST-VMS-28 [req:REQ-VR] [level:unit]
    /// init_gpu_kernel arg_ptr_expr error message lists available parameter names.
    #[test]
    fn test_gpu_kernel_error_lists_available_params() {
        // Arrange
        let state = VmState::init_gpu_kernel();

        // Act
        let result = state.arg_ptr_expr("batch_size");

        // Assert — error should list the available GPU params
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("input"),
            "error should list 'input' as available, got: {err_msg}"
        );
        assert!(
            err_msg.contains("scratchpad"),
            "error should list 'scratchpad' as available, got: {err_msg}"
        );
        assert!(
            err_msg.contains("telemetry"),
            "error should list 'telemetry' as available, got: {err_msg}"
        );
    }
}
