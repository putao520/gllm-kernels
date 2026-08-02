//! Register VM — 半虚拟机统一代码生成架构 (REGISTER-VM SPEC)
//!
//! 全链路: TraceOp → lower_*() → VmProgram → VmOptPass → RegAlloc → IsaLower → 物理指令
//!
//! 模块结构:
//! - `instr` — VM 指令集 (VmInstr) + 虚拟寄存器 (VRegId) + 表达式 (OffsetExpr)
//! - `isa_profile` — 硬件规则加载 (IsaProfile + AbiConvention + IsaFeature)
//! - `isa_hook` — ISA 特化注入 (IsaHook trait + 各后端 Hook)
//! - `lower` — TraceOp → VmInstr lowering (算法语义 → VM 指令)
//! - `reg_alloc` — 寄存器分配 (线性扫描 + InterferenceGraph)
//! - `stack_frame` — 栈帧管理 (scope-based + alignment)
//! - `pressure` — 硬件压力模型 (Memory/Register/Parallel/Quant)
//! - `opt_pass` — VM 优化 Pass 注册架构
//! - `x86_lower` — x86_64 ISA Lower (VmInstr → iced_x86)
//! - `plan_lower` — FusionPlan → VmProgram 翻译

pub mod aarch64_lower;
pub mod algo_interpreter;
pub mod algo_registry;
pub mod algo_template;
pub mod algo_templates;
pub mod attention_emit;
pub mod auto_select;
pub mod compiler_constraints;
pub mod debug_map;
pub mod dispatch_emit;
#[cfg(test)]
mod e2e_tests;
pub mod fusion_group_emit;
pub mod gemm_emit;
pub mod gemm_impls;
pub mod gpu_lower;
/// GPU SIMT 线程分区原语 (thread tile 抽象, ARCH-GPU-SIMT-THREAD-MODEL)。
pub mod gpu_thread_tile;
pub mod hetero_emit;
pub mod instr;
pub mod isa_hook;
pub mod isa_profile;
pub mod lower;
pub mod mega_kernel_emit;
pub mod mla_emit;
pub mod moe_quant_emit;
pub mod norm_softmax_emit;
pub mod numerical_sim;
pub mod op_impl;
pub mod opt_pass;
pub mod page_decode;
pub mod plan_lower;
pub mod ptx_registry;
pub mod quant_decode;
pub mod quant_gather_emit;
pub mod quant_offset_dsl;
pub mod reg_alloc;
pub mod reg_conflict;
pub mod resource_planner;
pub mod stack_frame;
pub mod structural_builder;
pub mod structural_emit;
pub mod telemetry_emit;
pub mod topology;
pub mod trace;
pub mod trace_opt;
pub mod verify;
pub mod vision_audio_emit;
pub mod vm_instr_category;
pub mod vm_state;
pub mod x86_lower;

// Re-exports: vm_state 中的公共类型，供 plan_lower 等模块直接使用
pub use vm_state::{AbiPtrs, EmitState, HeteroPhase, HeteroPhasePlan};
