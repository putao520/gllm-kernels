//! AArch64 ISA Lower (REGISTER-VM SPEC §10)
//!
//! VmInstr → AArch64 物理指令。
//! 支持 NEON (W128) / SVE2 (Scalable predicated) / SME2 (Tile multi-vec)。
//!
//! 三级 ISA 层次:
//!   1. NEON 基线 — 固定 128-bit (4×f32), LD1/ST1/FADD/FMUL/FMLA
//!   2. SVE2 — 可伸缩谓词循环 WHILELT+LD1W/ST1W/FADD predicated, 自动 tail 处理
//!   3. SME2 — ZA tile outer-product FMOPA + multi-vec FMLA + MOVA slice 读取
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 5 个片段):
//! - `aarch64_lower/helpers.inc.rs`       — 构造器 + resolve + emit helpers
//! - `aarch64_lower/emit_math.inc.rs`     — emit_f32_broadcast + exp + fp8 + quant 数学
//! - `aarch64_lower/lower_instr.inc.rs`   — lower_instr mega-match
//! - `aarch64_lower/finalize_quant.inc.rs` — finalize + quant_load + biplane_load
//! - `aarch64_lower/tests.inc.rs`         — 测试模块

use super::instr::*;
use super::isa_profile::*;
use super::reg_alloc::RegAllocation;
use super::stack_frame::StackFrame;
use crate::compiler::trace::AArch64ElemStrategy;
use crate::types::CompilerError;

/// AArch64 ISA Lower。
///
/// 直接输出机器码字节 (encoding)，不依赖外部汇编器库。
/// 根据 `Platform::AArch64` 的 SVE2/SME2 特性标志选择最优路径。
pub struct AArch64Lower {
    code: Vec<u8>,
    const_pool: Vec<(f32, usize)>, // (value, offset_in_code)
    /// 循环控制: (loop_top_offset, branch_placeholder_offset, counter_reg, offset_reg, step, is_sve)
    loop_stack: Vec<LoopCtx>,
    /// 目标平台特性
    platform: AArch64Features,
    /// 标签表: label_id -> code_offset (用于 MarkLabel 和分支回填)
    labels: std::collections::HashMap<usize, usize>,
    /// 硬件资源生命周期追踪 (SPEC 15 REQ-JCTX-011)
    jit_ctx: crate::compiler::jit_context::JitContext,
}

/// 从 `Platform::AArch64` 提取的特性快照。
#[derive(Debug, Clone)]
#[derive(Default)]
struct AArch64Features {
    has_sve2: bool,
    has_sme2: bool,
    sve_vl: usize, // bytes, 0 if no SVE
}


/// 循环上下文 — NEON 或 SVE2 路径各自的状态。
#[derive(Debug, Clone)]
struct LoopCtx {
    /// 循环体顶部的代码偏移 (B 跳回的目标)
    loop_top: usize,
    /// NEON: B.GE placeholder 在 code 中的偏移; SVE2: B.NONE placeholder 偏移
    branch_placeholder: usize,
    counter_reg: u8,
    offset_reg: u8,
    step: usize,
    is_sve: bool,
    /// SVE2 路径: 存放 bound 的 GPR (x17 / 传入的 reg)
    bound_reg: Option<u8>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  DotDtype 特征判定辅助 (REQ-VR10)
//  禁止 DotDtype 身份匹配，通过谓词函数替代 match 模式。
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// DotDtype 是否为 BF16 (BFDOT 原生指令)。
#[inline]
fn dot_dtype_is_bf16(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Bf16)
}

/// DotDtype 是否为 FP16 (FCVTL+FMLA 路径)。
#[inline]
fn dot_dtype_is_fp16(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Fp16)
}

/// DotDtype 是否为 INT8 (SDOT 原生指令)。
#[inline]
fn dot_dtype_is_int8(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Int8)
}

/// DotDtype 是否为 INT4×INT8 (WidenCompute: 解包后 SDOT)。
#[inline]
fn dot_dtype_is_int4x8(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Int4x8)
}

/// DotDtype 是否为 FP4/E2M1 (WidenCompute: 解码后 FMLA)。
#[inline]
fn dot_dtype_is_fp4(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Fp4)
}

include!("aarch64_lower/helpers.inc.rs");
include!("aarch64_lower/emit_math.inc.rs");
include!("aarch64_lower/lower_instr.inc.rs");
include!("aarch64_lower/finalize_quant.inc.rs");

#[cfg(test)]
include!("aarch64_lower/tests.inc.rs");
