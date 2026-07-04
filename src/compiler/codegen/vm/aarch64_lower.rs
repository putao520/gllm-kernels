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
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 6 个片段):
//! - `aarch64_lower/helpers.inc.rs`       — 构造器 + resolve + emit helpers
//! - `aarch64_lower/emit_math.inc.rs`     — emit_f32_broadcast + exp + fp8 + quant 数学
//! - `aarch64_lower/lower_instr.inc.rs`   — lower_instr L0 分类 dispatch (ARCH-LOWER-DISPATCH-LAYERING)
//! - `aarch64_lower/lower_instr_dispatch.inc.rs` — L1 变体路由 + L2 叶子 emit
//! - `aarch64_lower/finalize_quant.inc.rs` — finalize + quant_load + biplane_load
//! - `aarch64_lower/tests.inc.rs`         — 测试模块

use super::instr::*;
use super::isa_profile::*;
use super::reg_alloc::RegAllocation;
use super::stack_frame::StackFrame;
use crate::compiler::trace::AArch64ElemStrategy;
use crate::compiler::trace::QuantPrecision;
use crate::compiler::trace::DTypeKind;
use crate::types::CompilerError;
use crate::types::DType;

/// AArch64 ISA Lower。
///
/// 直接输出机器码字节 (encoding)，不依赖外部汇编器库。
/// 根据 `Platform::AArch64` 的 SVE/SVE2/SME2/BF16/DotProd/I8MM 特性标志选择最优路径。
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
///
/// 保留 Platform::AArch64 的全部特性字段 (BCE-20260703-AARCH64-FEATURES-DROPPED):
/// 原实现只用 `..` 丢弃了 has_bf16/has_dotprod/has_i8mm/has_sve, 导致 lower 层无法
/// 感知这些特性 → BFDOT/SDOT 无条件发出 (无该特性的 CPU SIGILL) + SVE1-only CPU
/// 被降级为 NEON。扩展后 lower 层可按特性门控指令选择 (NO-SILENT-FALLBACK: 不支持返回 Err)。
#[derive(Debug, Clone)]
#[derive(Default)]
struct AArch64Features {
    has_sve: bool,       // SVE 基础 (LD1W/FADD Z/WHILELT/PTRUE — SVE1 指令, has_sve 即可用)
    has_sve2: bool,      // SVE2 (SMMLA/USDOT/BFDOT-Z 等 SVE2 专属指令)
    has_sme2: bool,      // SME2 (multi-vec FMLA, ZA tile)
    has_bf16: bool,      // FEAT_BF16 (BFDOT/BFMMLA — NEON dot 指令)
    has_dotprod: bool,   // FEAT_DotProd (SDOT/UDOT — NEON dot 指令)
    has_i8mm: bool,      // FEAT_I8MM (SMMLA/UMMLA — i8mm 矩阵乘指令)
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
    /// counter spill slot offset from sp (None = in physical GPR)
    counter_spill_sp_off: Option<i32>,
    /// byte_offset spill slot offset from sp (None = in physical GPR)
    offset_spill_sp_off: Option<i32>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  DotDtype 特征判定辅助 (REQ-VR10)
//  禁止 DotDtype 身份匹配，通过谓词函数替代 match 模式。
//  BCE-20260704-X86HW-002: 新增 Bf16xF32/Fp16xF32 混合精度谓词 (a=F32, b=BF16/FP16)。
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// DotDtype 是否为纯双 BF16 (BFDOT 原生指令, a=BF16 b=BF16)。
#[inline]
fn dot_dtype_is_bf16(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Bf16)
}

/// DotDtype 是否为混合精度 BF16xF32 (a=F32 激活, b=BF16 权重)。
/// 走 WidenCompute 路径: b 需 BF16→F32 widen (USHLL+SHL), a 已 F32, 然后 FMLA。
/// 不需要 FEAT_BF16 (USHLL/SHL/FMLA 是 NEON 基线指令)。
#[inline]
fn dot_dtype_is_bf16xf32(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Bf16xF32)
}

/// DotDtype 是否为纯双 FP16 (FCVTL+FMLA 路径, a=FP16 b=FP16)。
#[inline]
fn dot_dtype_is_fp16(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Fp16)
}

/// DotDtype 是否为混合精度 Fp16xF32 (a=F32 激活, b=FP16 权重)。
/// 走 WidenCompute 路径: b 需 FP16→F32 widen (FCVTL), a 已 F32, 然后 FMLA。
#[inline]
fn dot_dtype_is_fp16xf32(dt: DotDtype) -> bool {
    matches!(dt, DotDtype::Fp16xF32)
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
include!("aarch64_lower/lower_instr_dispatch.inc.rs");
include!("aarch64_lower/finalize_quant.inc.rs");

#[cfg(test)]
include!("aarch64_lower/tests.inc.rs");
