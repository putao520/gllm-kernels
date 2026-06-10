//! GPU ISA Lower (REGISTER-VM SPEC §10)
//!
//! VmInstr → PTX/HIP/MSL 文本 IR。
//! GPU 后端输出文本 IR (字符串)，由 driver 编译为设备二进制。
//!
//! ## SM 版本特化
//!
//! | SM | 核心 MMA | 数据搬运 | 特化 |
//! |----|---------|---------|------|
//! | SM70 | wmma 16×16×16 | global_load | 无异步 |
//! | SM80 | mma.sync 16×8×16 | cp.async 128B | BF16/TF32 |
//! | SM90 | wgmma.mma_async 64×N×K | TMA 2D | warp_spec, FP8 |
//! | SM100+ | tcgen05.mma | TMA + TMEM | block-scaled, FP4 |
//! | gfx908+ | v_mfma_f32_16x16x16 | LDS | CDNA MFMA v1 |
//! | gfx950 | v_mfma_f32_32x32x16 | LDS 160KB | CDNA4 MFMA v2, FP8/FP4 |
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 5 个片段):
//! - `gpu_lower/prologue.inc.rs`     — GpuLower struct + 构造 + prologue/epilogue
//! - `gpu_lower/lower_instr.inc.rs`  — lower_instr 巨型 match (PTX/HIP/MSL 分发)
//! - `gpu_lower/lower_gpu.inc.rs`    — GPU 特化解码 (LZ4/BitPackRle/MXFP4/NVFP4/KIVI) + finalize
//! - `gpu_lower/quant_load.inc.rs`   — 量化 block/biplane load lowering
//! - `gpu_lower/tests.inc.rs`        — 测试模块

use super::instr::*;
use super::isa_profile::*;
use super::reg_alloc::RegAllocation;
use super::stack_frame::StackFrame;
use crate::types::CompilerError;

/// GPU 方言——决定输出格式。
#[derive(Debug, Clone, Copy)]
pub enum GpuDialect {
    /// NVIDIA PTX (sm_version 决定可用指令)
    Ptx { sm_version: u32 },
    /// AMD HIP (gfx_arch 决定 MFMA 版本, wave_size 决定线程组织)
    Hip { gfx_arch: u32, wave_size: u32 },
    /// Apple Metal Shading Language
    Metal { gpu_family: u32 },
}

/// GPU lowering state (REGISTER-VM SPEC §10).
pub struct GpuLower {
    dialect: GpuDialect,
    ir: String,
    indent: usize,
    /// 唯一循环标签计数器（修复 PTX 循环回跳地址）
    loop_label_counter: u32,
    /// 循环栈：每个元素保存 (label_id, counter_vreg_name, offset_vreg_name, step_bytes)
    /// ARCH-GPU-LOOP-TRACKING: LoopEnd 从栈读取真实 counter 名字，不再硬编码 %r0
    loop_stack: Vec<(u32, String, String, usize)>,
    /// skip 标签计数器
    skip_label_counter: u32,
    /// TMEM 是否已分配 (SM100+)
    tmem_allocated: bool,
    /// ABI 参数名映射（emit_prologue 声明的 .param 名字，LoadPtr 必须用同样名字）
    /// ARCH-GPU-PARAM-NAMES: 避免 param0/param1 与声明名字不匹配导致 PTX 汇编失败
    abi_param_names: Vec<&'static str>,
    /// ISA Lower 独占 scratch 虚拟寄存器名（PTX/HIP 文本 IR 层）。
    /// 这些名字在 emit_prologue 中被显式声明，RegAllocator 用独立编号空间避免冲突。
    scratch_vec_names: Vec<&'static str>,  // 如 %fs0/%fs1 (f-scratch)
    scratch_gpr_names: Vec<&'static str>,  // 如 %rs0/%rs1 (r-scratch)
    scratch_pred_names: Vec<&'static str>, // 如 %ps0/%ps1 (p-scratch)
    /// VReg → Kind 映射 (ARCH-GPU-REG-KIND)。
    /// 由 set_vreg_kind_map() 从 VmProgram 提取，使 reg_name 可按 kind 选正确命名空间。
    vreg_kinds: Vec<Option<VRegKind>>,
    /// MarkLabel path labels — key = MarkLabel label_id, value = generated label string.
    path_labels: std::collections::HashMap<usize, String>,
    /// Function epilogue label — BreakLoop bra 到此标签.
    epilogue_label: String,
    /// Path label counter for MarkLabel label generation.
    path_label_counter: u32,
    /// 硬件资源生命周期追踪 (SPEC 15 REQ-JCTX-012)
    jit_ctx: crate::compiler::jit_context::JitContext,
}

include!("gpu_lower/prologue.inc.rs");
include!("gpu_lower/lower_instr.inc.rs");
include!("gpu_lower/lower_gpu.inc.rs");
include!("gpu_lower/quant_load.inc.rs");

#[cfg(test)]
include!("gpu_lower/tests.inc.rs");
