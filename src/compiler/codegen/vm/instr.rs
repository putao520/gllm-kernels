//! Register-VM 半结构化状态追踪 IR (REGISTER-VM SPEC §3-§5)
//!
//! **这不是虚拟机指令系统。** 这是编译时的半结构化中间表达，借鉴 VM 思想
//! 在代码生成过程中动态跟踪寄存器分配、内存布局、栈帧状态，确保 JIT 生成的
//! 机器码没有寄存器错误复用/重定义、堆栈不平衡、并行与串行冲突。
//!
//! 每条 VmInstr 是一个状态转移记录：
//! - RegAllocator 读取 VmProgram 计算虚拟寄存器 (VRegId) 的活跃区间 → 物理寄存器映射
//! - StackFrame 读取 VmProgram 计算栈帧布局（callee-save/spill 区大小）
//! - IsaLower 遍历 VmProgram，将每条状态记录翻译为物理机器码
//! - LoopBegin/LoopEnd 标记循环作用域，供寄存器分配器追踪跨迭代活跃性
//!
//! 参考: SPEC/DOCS/scheduling/vm-state-tracking.md (REGISTER-VM 半虚拟化)
//!
//! 设计约束 (§14):
//! - 所有 VmInstr 必须有 TraceOp provenance (validate_provenance)
//! - 不允许标量 fallback (NO_SCALAR)
//! - 不允许静默 NOP (NO_SILENT_FALLBACK)
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 4 个片段):
//! - `instr_fragments/types.inc.rs`    — 小型类型定义 (Fp8Kind..KvLoadMode)
//! - `instr_fragments/vminstr.inc.rs`  — VmInstr 枚举 + impl
//! - `instr_fragments/program.inc.rs`  — VmProgram 结构体 + impl
//! - `instr_fragments/tests.inc.rs`    — 测试模块

include!("instr_fragments/types.inc.rs");
include!("instr_fragments/vminstr.inc.rs");
include!("instr_fragments/program.inc.rs");

#[cfg(test)]
include!("instr_fragments/tests.inc.rs");
