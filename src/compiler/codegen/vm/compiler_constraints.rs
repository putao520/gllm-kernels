//! 编译约束与生命周期集成 (REGISTER-VM SPEC §7 + JIT-LIFECYCLE INFRASTRUCTURE)
//!
//! 此模块将 JIT 生命周期管理集成到编译管线，提供：
//! - 编译约束定义与检查
//! - LifecycleTag 在编译管线中的传播
//! - 编译管线的生命周期感知验证
//!
//! REQ-LC-012: JIT 生命周期集成到 compiler_constraints.rs

use std::collections::HashMap;
use super::instr::{VmProgram, VRegId, VmInstr};
use super::reg_alloc::{RegAllocation, LiveInterval, LifecycleTag};
use super::stack_frame::ScopedSpillAllocator;
use super::verify::{VerifyReport, post_hoc_verify};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 编译约束定义 (REQ-LC-012)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 编译约束配置
///
/// 定义编译过程中的各种约束条件，用于指导寄存器分配和优化。
#[derive(Debug, Clone)]
pub struct CompilerConstraints {
    /// 是否启用生命周期感知分配
    pub lifecycle_aware_alloc: bool,
    /// 是否启用 scope-based spill 分配
    pub scoped_spill_alloc: bool,
    /// 是否启用 post-hoc 验证
    pub enable_post_hoc_verify: bool,
    /// 最大 spill 槽数量（防止栈溢出）
    pub max_spill_slots: usize,
    /// 最大 spill 总字节数（防止栈溢出）
    pub max_spill_bytes: usize,
}

impl Default for CompilerConstraints {
    fn default() -> Self {
        Self {
            lifecycle_aware_alloc: true,
            scoped_spill_alloc: true,
            enable_post_hoc_verify: true,
            max_spill_slots: 256,
            max_spill_bytes: 64 * 1024, // 64KB
        }
    }
}

impl CompilerConstraints {
    /// 创建默认约束
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建宽松约束（用于测试或快速编译）
    pub fn relaxed() -> Self {
        Self {
            lifecycle_aware_alloc: false,
            scoped_spill_alloc: false,
            enable_post_hoc_verify: false,
            max_spill_slots: 512,
            max_spill_bytes: 128 * 1024,
        }
    }

    /// 创建严格约束（用于生产环境）
    pub fn strict() -> Self {
        Self {
            lifecycle_aware_alloc: true,
            scoped_spill_alloc: true,
            enable_post_hoc_verify: true,
            max_spill_slots: 128,
            max_spill_bytes: 32 * 1024,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 编译管线集成 (REQ-LC-012)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 编译管线上下文
///
/// 在编译管线中传播生命周期信息和约束。
#[derive(Debug)]
pub struct CompilationContext {
    /// 编译约束
    pub constraints: CompilerConstraints,
    /// Scope 位置表 (用于 ScopedSpillAllocator)
    pub scope_positions: Vec<(usize, usize, usize)>,
    /// 生命周期标签映射 (VRegId → LifecycleTag)
    pub lifecycle_tags: HashMap<VRegId, LifecycleTag>,
}

impl CompilationContext {
    /// 创建新的编译上下文
    pub fn new(constraints: CompilerConstraints) -> Self {
        Self {
            constraints,
            scope_positions: Vec::new(),
            lifecycle_tags: HashMap::new(),
        }
    }

    /// 从 VmProgram 构建 scope 位置表
    pub fn build_scope_positions(&mut self, program: &VmProgram) {
        let mut positions = Vec::new();
        let mut scope_stack: Vec<(usize, usize)> = Vec::new(); // (scope_id, begin_pos)

        for (i, instr) in program.instrs.iter().enumerate() {
            match instr {
                VmInstr::ScopeBegin { scope_id } => {
                    scope_stack.push((*scope_id, i));
                }
                VmInstr::ScopeEnd { scope_id } => {
                    if let Some((id, begin)) = scope_stack.pop() {
                        if id == *scope_id {
                            positions.push((id, begin, i));
                        }
                    }
                }
                _ => {}
            }
        }

        self.scope_positions = positions;
    }

    /// 获取指定位置所属的 scope
    pub fn scope_at_position(&self, pos: usize) -> Option<usize> {
        for &(scope_id, begin, end) in &self.scope_positions {
            if pos >= begin && pos <= end {
                return Some(scope_id);
            }
        }
        None
    }
}

/// 编译管线约束检查器
///
/// REQ-LC-012: 在编译管线的各个阶段检查约束
pub struct ConstraintChecker<'a> {
    /// 编译上下文
    ctx: &'a CompilationContext,
    /// 违规记录
    violations: Vec<ConstraintViolation>,
}

impl<'a> ConstraintChecker<'a> {
    /// 创建新的约束检查器
    pub fn new(ctx: &'a CompilationContext) -> Self {
        Self {
            ctx,
            violations: Vec::new(),
        }
    }

    /// 检查 spill 数量约束
    pub fn check_spill_count(&mut self, spills_len: usize) -> Result<(), ConstraintViolation> {
        if spills_len > self.ctx.constraints.max_spill_slots {
            return Err(ConstraintViolation::TooManySpills {
                actual: spills_len,
                limit: self.ctx.constraints.max_spill_slots,
            });
        }
        Ok(())
    }

    /// 检查 spill 字节数约束
    pub fn check_spill_bytes(&mut self, spill_bytes: usize) -> Result<(), ConstraintViolation> {
        if spill_bytes > self.ctx.constraints.max_spill_bytes {
            return Err(ConstraintViolation::SpillBytesExceeded {
                actual: spill_bytes,
                limit: self.ctx.constraints.max_spill_bytes,
            });
        }
        Ok(())
    }

    /// 检查生命周期标签传播
    pub fn check_lifecycle_propagation(&mut self, intervals: &[LiveInterval]) -> Result<(), ConstraintViolation> {
        for iv in intervals {
            if !self.ctx.lifecycle_tags.contains_key(&iv.vreg) {
                return Err(ConstraintViolation::MissingLifecycleTag { vreg: iv.vreg });
            }
            let expected_tag = self.ctx.lifecycle_tags[&iv.vreg];
            if iv.lifecycle != expected_tag {
                return Err(ConstraintViolation::LifecycleTagMismatch {
                    vreg: iv.vreg,
                    expected: expected_tag,
                    actual: iv.lifecycle,
                });
            }
        }
        Ok(())
    }

    /// 运行所有约束检查
    pub fn run_all_checks(
        &mut self,
        spills_len: usize,
        spill_bytes: usize,
        intervals: &[LiveInterval],
    ) -> Result<(), Vec<ConstraintViolation>> {
        self.violations.clear();

        if let Err(v) = self.check_spill_count(spills_len) {
            self.violations.push(v);
        }
        if let Err(v) = self.check_spill_bytes(spill_bytes) {
            self.violations.push(v);
        }
        if self.ctx.constraints.lifecycle_aware_alloc {
            if let Err(v) = self.check_lifecycle_propagation(intervals) {
                self.violations.push(v);
            }
        }

        if self.violations.is_empty() {
            Ok(())
        } else {
            Err(self.violations.clone())
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 约束违规类型 (REQ-LC-012)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 编译约束违规类型
#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    /// Spill 槽数量超过限制
    TooManySpills {
        actual: usize,
        limit: usize,
    },
    /// Spill 字节数超过限制
    SpillBytesExceeded {
        actual: usize,
        limit: usize,
    },
    /// 缺少生命周期标签
    MissingLifecycleTag {
        vreg: VRegId,
    },
    /// 生命周期标签不匹配
    LifecycleTagMismatch {
        vreg: VRegId,
        expected: LifecycleTag,
        actual: LifecycleTag,
    },
}

impl std::fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManySpills { actual, limit } => {
                write!(f, "Too many spill slots: {} (limit: {}) (REQ-LC-012)", actual, limit)
            }
            Self::SpillBytesExceeded { actual, limit } => {
                write!(f, "Spill bytes exceeded: {} (limit: {}) (REQ-LC-012)", actual, limit)
            }
            Self::MissingLifecycleTag { vreg } => {
                write!(f, "Missing lifecycle tag for VRegId({}) (REQ-LC-012)", vreg.0)
            }
            Self::LifecycleTagMismatch { vreg, expected, actual } => {
                write!(f, "Lifecycle tag mismatch for VRegId({}): expected {:?}, got {:?} (REQ-LC-012)",
                    vreg.0, expected, actual)
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 编译管线入口 (REQ-LC-012)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 运行带约束的编译管线
///
/// REQ-LC-012: 集成 LifecycleTag、ScopedSpillAllocator 和 post_hoc_verify
///
/// # 流程
/// 1. 创建 CompilationContext 并构建 scope 位置表
/// 2. 运行寄存器分配（使用 ScopedSpillAllocator）
/// 3. 运行约束检查
/// 4. 运行 post_hoc_verify
pub fn compile_with_constraints(
    program: &VmProgram,
    alloc: &RegAllocation,
    intervals: &[LiveInterval],
    constraints: &CompilerConstraints,
) -> Result<VerifyReport, CompilationError> {
    // 1. 创建编译上下文
    let mut ctx = CompilationContext::new(constraints.clone());
    ctx.build_scope_positions(program);

    // 2. 从 intervals 提取生命周期标签
    for iv in intervals {
        ctx.lifecycle_tags.insert(iv.vreg, iv.lifecycle);
    }

    // 2b. 使用 ScopedSpillAllocator 验证 scope-based spill 分配 (REQ-LC-012)
    if constraints.scoped_spill_alloc {
        let mut scoped = ScopedSpillAllocator::new();
        // 重放 VmProgram 中的 scope 操作，确保 ScopedSpillAllocator 的 scope 栈
        // 与程序的实际 scope 结构一致。
        for instr in &program.instrs {
            match instr {
                VmInstr::ScopeBegin { .. } => {
                    scoped.scope_begin();
                }
                VmInstr::ScopeEnd { .. } => {
                    scoped.scope_end();
                }
                _ => {}
            }
        }
        // 为每个被 spill 的 VReg 分配 scope-aware slot，验证分配总量。
        // 跳过 freed slot 占位符 (VRegId(u32::MAX))。
        for spill in &alloc.spills {
            if spill.vreg.0 == u32::MAX {
                continue;
            }
            let def_point = match intervals.iter().find(|iv| iv.vreg == spill.vreg) {
                Some(iv) => iv.def_point,
                None => continue,
            };
            let scope_id = ctx.scope_at_position(def_point);
            scoped.alloc(spill.vreg, spill.size, scope_id);
        }
        let scoped_total = scoped.total_allocated();
        if scoped_total > constraints.max_spill_bytes {
            return Err(CompilationError::ConstraintViolations {
                count: 1,
                details: format!(
                    "ScopedSpillAllocator: total allocated {} bytes exceeds max {} bytes (REQ-LC-012)",
                    scoped_total, constraints.max_spill_bytes
                ),
            });
        }
    }

    // 3. 运行约束检查
    let spill_bytes: usize = alloc.spills.iter().map(|s| s.size).sum();
    let mut checker = ConstraintChecker::new(&ctx);
    checker.run_all_checks(alloc.spills.len(), spill_bytes, intervals)
        .map_err(|violations| CompilationError::ConstraintViolations {
            count: violations.len(),
            details: violations.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("; "),
        })?;

    // 4. 运行 post_hoc_verify
    if constraints.enable_post_hoc_verify {
        let report = post_hoc_verify(program, alloc, intervals)
            .map_err(|e| CompilationError::VerificationFailed(e.to_string()))?;

        if report.has_violations {
            return Err(CompilationError::VerificationFailed(format!(
                "post_hoc_verify found {} violations (REQ-LC-012)",
                report.total_count()
            )));
        }

        Ok(report)
    } else {
        Ok(VerifyReport::empty())
    }
}

/// 编译错误类型
#[derive(Debug, Clone)]
pub enum CompilationError {
    /// 约束违规
    ConstraintViolations {
        count: usize,
        details: String,
    },
    /// 验证失败
    VerificationFailed(String),
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstraintViolations { count, details } => {
                write!(f, "Compilation failed: {} constraint violations: {} (REQ-LC-012)",
                    count, details)
            }
            Self::VerificationFailed(msg) => {
                write!(f, "Compilation failed: verification error: {} (REQ-LC-012)", msg)
            }
        }
    }
}

impl std::error::Error for CompilationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::instr::{SimdWidth, VRegKind};

    #[test]
    fn test_compiler_constraints_default() {
        let constraints = CompilerConstraints::default();
        assert!(constraints.lifecycle_aware_alloc);
        assert!(constraints.scoped_spill_alloc);
        assert!(constraints.enable_post_hoc_verify);
    }

    #[test]
    fn test_compiler_constraints_relaxed() {
        let constraints = CompilerConstraints::relaxed();
        assert!(!constraints.lifecycle_aware_alloc);
        assert!(!constraints.scoped_spill_alloc);
        assert!(!constraints.enable_post_hoc_verify);
    }

    #[test]
    fn test_constraint_checker_spill_count() {
        let ctx = CompilationContext::new(CompilerConstraints::default());
        let mut checker = ConstraintChecker::new(&ctx);

        // 应该通过
        assert!(checker.check_spill_count(100).is_ok());

        // 应该失败（超过限制 256）
        assert!(checker.check_spill_count(300).is_err());
    }

    #[test]
    fn test_constraint_checker_spill_bytes() {
        let ctx = CompilationContext::new(CompilerConstraints::default());
        let mut checker = ConstraintChecker::new(&ctx);

        // 应该通过
        assert!(checker.check_spill_bytes(32 * 1024).is_ok());

        // 应该失败（超过限制 64KB）
        assert!(checker.check_spill_bytes(100 * 1024).is_err());
    }

    #[test]
    fn test_constraint_violation_display() {
        let violation = ConstraintViolation::TooManySpills { actual: 300, limit: 256 };
        let msg = format!("{}", violation);
        assert!(msg.contains("Too many spill slots"));
        assert!(msg.contains("300"));
        assert!(msg.contains("256"));
        assert!(msg.contains("REQ-LC-012"));
    }

    #[test]
    fn test_compilation_context_build_scope_positions() {
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });

        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.build_scope_positions(&prog);

        assert_eq!(ctx.scope_positions.len(), 2);
        assert_eq!(ctx.scope_positions[0], (1, 1, 2)); // scope 1
        assert_eq!(ctx.scope_positions[1], (0, 0, 3)); // scope 0
    }

    #[test]
    fn test_compilation_context_scope_at_position() {
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });

        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.build_scope_positions(&prog);

        // 位置 1 应该在 scope 1 中
        assert_eq!(ctx.scope_at_position(1), Some(1));
        // 位置 2 在 scope 1 的范围内（ScopeEnd 位置是 inclusive 的）
        // scope 1 = (id=1, begin=1, end=2), scope 0 = (id=0, begin=0, end=3)
        // scope_at_position 按顺序匹配，scope 1 先被匹配
        assert_eq!(ctx.scope_at_position(2), Some(1));
        // 位置 10 应该不在任何 scope 中
        assert_eq!(ctx.scope_at_position(10), None);
    }

    // ── 新增测试 (TEST-CC-08 ~ TEST-CC-20) ──────────────────────────────

    // @trace TEST-CC-08 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compiler_constraints_new_matches_default() {
        // Arrange & Act
        let new_constraints = CompilerConstraints::new();
        let default_constraints = CompilerConstraints::default();

        // Assert: new() should produce identical values to default()
        assert_eq!(new_constraints.lifecycle_aware_alloc, default_constraints.lifecycle_aware_alloc);
        assert_eq!(new_constraints.scoped_spill_alloc, default_constraints.scoped_spill_alloc);
        assert_eq!(new_constraints.enable_post_hoc_verify, default_constraints.enable_post_hoc_verify);
        assert_eq!(new_constraints.max_spill_slots, default_constraints.max_spill_slots);
        assert_eq!(new_constraints.max_spill_bytes, default_constraints.max_spill_bytes);
    }

    // @trace TEST-CC-09 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compiler_constraints_strict_values() {
        // Arrange & Act
        let constraints = CompilerConstraints::strict();

        // Assert
        assert!(constraints.lifecycle_aware_alloc);
        assert!(constraints.scoped_spill_alloc);
        assert!(constraints.enable_post_hoc_verify);
        assert_eq!(constraints.max_spill_slots, 128);
        assert_eq!(constraints.max_spill_bytes, 32 * 1024);
    }

    // @trace TEST-CC-10 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compiler_constraints_relaxed_values() {
        // Arrange & Act
        let constraints = CompilerConstraints::relaxed();

        // Assert: all awareness flags off, higher limits than default
        assert!(!constraints.lifecycle_aware_alloc);
        assert!(!constraints.scoped_spill_alloc);
        assert!(!constraints.enable_post_hoc_verify);
        assert_eq!(constraints.max_spill_slots, 512);
        assert_eq!(constraints.max_spill_bytes, 128 * 1024);
    }

    // @trace TEST-CC-11 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compilation_context_new_initializes_empty() {
        // Arrange
        let constraints = CompilerConstraints::new();

        // Act
        let ctx = CompilationContext::new(constraints);

        // Assert
        assert!(ctx.scope_positions.is_empty());
        assert!(ctx.lifecycle_tags.is_empty());
    }

    // @trace TEST-CC-12 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_lifecycle_missing_tag() {
        // Arrange: context has no lifecycle tags for this vreg
        let ctx = CompilationContext::new(CompilerConstraints::default());
        let mut checker = ConstraintChecker::new(&ctx);
        let interval = LiveInterval {
            vreg: VRegId(42),
            kind: VRegKind::Scalar,
            width: SimdWidth::Scalar,
            def_point: 0,
            last_use: 10,
            lifecycle: LifecycleTag::BodyLocal,
        };

        // Act
        let result = checker.check_lifecycle_propagation(&[interval]);

        // Assert
        let err = result.expect_err("should fail for missing lifecycle tag");
        match err {
            ConstraintViolation::MissingLifecycleTag { vreg } => {
                assert_eq!(vreg, VRegId(42));
            }
            other => panic!("expected MissingLifecycleTag, got {:?}", other),
        }
    }

    // @trace TEST-CC-13 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_lifecycle_tag_mismatch() {
        // Arrange: context has LoopInvariant but interval says BodyLocal
        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.lifecycle_tags.insert(VRegId(7), LifecycleTag::LoopInvariant);
        let mut checker = ConstraintChecker::new(&ctx);
        let interval = LiveInterval {
            vreg: VRegId(7),
            kind: VRegKind::Ptr,
            width: SimdWidth::Scalar,
            def_point: 0,
            last_use: 50,
            lifecycle: LifecycleTag::BodyLocal,
        };

        // Act
        let result = checker.check_lifecycle_propagation(&[interval]);

        // Assert
        let err = result.expect_err("should fail for tag mismatch");
        match err {
            ConstraintViolation::LifecycleTagMismatch { vreg, expected, actual } => {
                assert_eq!(vreg, VRegId(7));
                assert_eq!(expected, LifecycleTag::LoopInvariant);
                assert_eq!(actual, LifecycleTag::BodyLocal);
            }
            other => panic!("expected LifecycleTagMismatch, got {:?}", other),
        }
    }

    // @trace TEST-CC-14 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_lifecycle_propagation_success() {
        // Arrange: context tags match interval tags
        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.lifecycle_tags.insert(VRegId(1), LifecycleTag::Global);
        ctx.lifecycle_tags.insert(VRegId(2), LifecycleTag::LoopCarried);
        let mut checker = ConstraintChecker::new(&ctx);
        let intervals = vec![
            LiveInterval {
                vreg: VRegId(1),
                kind: VRegKind::Ptr,
                width: SimdWidth::Scalar,
                def_point: 0,
                last_use: 99,
                lifecycle: LifecycleTag::Global,
            },
            LiveInterval {
                vreg: VRegId(2),
                kind: VRegKind::Counter,
                width: SimdWidth::Scalar,
                def_point: 5,
                last_use: 80,
                lifecycle: LifecycleTag::LoopCarried,
            },
        ];

        // Act & Assert
        assert!(checker.check_lifecycle_propagation(&intervals).is_ok());
    }

    // @trace TEST-CC-15 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_run_all_skips_lifecycle_when_disabled() {
        // Arrange: relaxed constraints disable lifecycle_aware_alloc
        let mut ctx = CompilationContext::new(CompilerConstraints::relaxed());
        // Intentionally do NOT insert lifecycle tags for VRegId(99)
        let mut checker = ConstraintChecker::new(&ctx);
        let intervals = vec![
            LiveInterval {
                vreg: VRegId(99),
                kind: VRegKind::Scalar,
                width: SimdWidth::Scalar,
                def_point: 0,
                last_use: 10,
                lifecycle: LifecycleTag::BodyLocal,
            },
        ];

        // Act: should NOT fail on lifecycle since it's disabled
        let result = checker.run_all_checks(10, 1024, &intervals);

        // Assert
        assert!(result.is_ok());
    }

    // @trace TEST-CC-16 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_run_all_collects_multiple_violations() {
        // Arrange: both spill count and spill bytes exceeded
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act: 200 spills > 128 limit, 100KB bytes > 32KB limit
        let result = checker.run_all_checks(200, 100 * 1024, &[]);

        // Assert: should collect both violations
        let violations = result.expect_err("should fail with multiple violations");
        assert!(violations.len() >= 2);
    }

    // @trace TEST-CC-17 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_violation_display_spill_bytes_exceeded() {
        // Arrange
        let violation = ConstraintViolation::SpillBytesExceeded {
            actual: 100 * 1024,
            limit: 64 * 1024,
        };

        // Act
        let msg = format!("{}", violation);

        // Assert
        assert!(msg.contains("Spill bytes exceeded"));
        assert!(msg.contains("102400"));
        assert!(msg.contains("65536"));
        assert!(msg.contains("REQ-LC-012"));
    }

    // @trace TEST-CC-18 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_violation_display_missing_lifecycle_tag() {
        // Arrange
        let violation = ConstraintViolation::MissingLifecycleTag {
            vreg: VRegId(55),
        };

        // Act
        let msg = format!("{}", violation);

        // Assert
        assert!(msg.contains("Missing lifecycle tag"));
        assert!(msg.contains("55"));
        assert!(msg.contains("REQ-LC-012"));
    }

    // @trace TEST-CC-19 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_violation_display_lifecycle_tag_mismatch() {
        // Arrange
        let violation = ConstraintViolation::LifecycleTagMismatch {
            vreg: VRegId(10),
            expected: LifecycleTag::LoopInvariant,
            actual: LifecycleTag::CrossScope,
        };

        // Act
        let msg = format!("{}", violation);

        // Assert
        assert!(msg.contains("Lifecycle tag mismatch"));
        assert!(msg.contains("10"));
        assert!(msg.contains("LoopInvariant"));
        assert!(msg.contains("CrossScope"));
        assert!(msg.contains("REQ-LC-012"));
    }

    // @trace TEST-CC-20 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compilation_error_display_constraint_violations() {
        // Arrange
        let error = CompilationError::ConstraintViolations {
            count: 3,
            details: "Too many spills; Spill bytes exceeded".to_string(),
        };

        // Act
        let msg = format!("{}", error);

        // Assert
        assert!(msg.contains("3 constraint violations"));
        assert!(msg.contains("Too many spills"));
        assert!(msg.contains("REQ-LC-012"));
    }

    // @trace TEST-CC-21 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_build_scope_positions_empty_program() {
        // Arrange: program with no scope instructions
        let prog = VmProgram::new();
        let mut ctx = CompilationContext::new(CompilerConstraints::default());

        // Act
        ctx.build_scope_positions(&prog);

        // Assert
        assert!(ctx.scope_positions.is_empty());
    }

    // @trace TEST-CC-22 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_build_scope_positions_mismatched_scope_end_ignored() {
        // Arrange: ScopeEnd with no matching ScopeBegin
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeEnd { scope_id: 99 }); // no matching begin
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });

        let mut ctx = CompilationContext::new(CompilerConstraints::default());

        // Act
        ctx.build_scope_positions(&prog);

        // Assert: only scope 0 is recorded; scope 99 end is ignored
        assert_eq!(ctx.scope_positions.len(), 1);
        assert_eq!(ctx.scope_positions[0], (0, 1, 2));
    }

    // @trace TEST-CC-23 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_scope_at_position_boundary_inclusive() {
        // Arrange: scope spans positions [1, 3]
        let filler = VmInstr::DeclareVReg { id: VRegId(0), kind: VRegKind::Scalar, width: SimdWidth::Scalar };
        let mut prog = VmProgram::new();
        prog.emit(filler.clone());                          // pos 0
        prog.emit(VmInstr::ScopeBegin { scope_id: 5 }); // pos 1
        prog.emit(filler.clone());                          // pos 2
        prog.emit(VmInstr::ScopeEnd { scope_id: 5 });   // pos 3

        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.build_scope_positions(&prog);

        // Act & Assert: both begin and end positions are inclusive
        assert_eq!(ctx.scope_at_position(0), None); // before scope
        assert_eq!(ctx.scope_at_position(1), Some(5)); // at ScopeBegin
        assert_eq!(ctx.scope_at_position(2), Some(5)); // inside scope
        assert_eq!(ctx.scope_at_position(3), Some(5)); // at ScopeEnd
        assert_eq!(ctx.scope_at_position(4), None); // after scope
    }

    // @trace TEST-CC-24 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_spill_count_exact_boundary() {
        // Arrange: strict limits with max_spill_slots = 128
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act & Assert: exactly at limit should pass
        assert!(checker.check_spill_count(128).is_ok());
        // One over limit should fail
        assert!(checker.check_spill_count(129).is_err());
    }

    // @trace TEST-CC-25 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_spill_bytes_exact_boundary() {
        // Arrange: strict limits with max_spill_bytes = 32*1024
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act & Assert: exactly at limit should pass
        assert!(checker.check_spill_bytes(32 * 1024).is_ok());
        // One byte over should fail
        assert!(checker.check_spill_bytes(32 * 1024 + 1).is_err());
    }

    // @trace TEST-CC-26 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compilation_error_display_verification_failed() {
        // Arrange
        let error = CompilationError::VerificationFailed("stack misaligned".to_string());

        // Act
        let msg = format!("{}", error);

        // Assert
        assert!(msg.contains("verification error"));
        assert!(msg.contains("stack misaligned"));
        assert!(msg.contains("REQ-LC-012"));
    }

    // @trace TEST-CC-27 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_compilation_error_is_std_error() {
        // Arrange
        let error: Box<dyn std::error::Error> =
            Box::new(CompilationError::VerificationFailed("test".to_string()));

        // Act & Assert: CompilationError implements std::error::Error
        assert!(!error.to_string().is_empty());
    }

    // @trace TEST-CC-28 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_build_scope_positions_nested_three_levels() {
        // Arrange: three nested scopes
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 }); // pos 0
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 }); // pos 1
        prog.emit(VmInstr::ScopeBegin { scope_id: 2 }); // pos 2
        prog.emit(VmInstr::ScopeEnd { scope_id: 2 });   // pos 3
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });   // pos 4
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });   // pos 5

        let mut ctx = CompilationContext::new(CompilerConstraints::default());

        // Act
        ctx.build_scope_positions(&prog);

        // Assert: three scopes recorded in LIFO pop order
        assert_eq!(ctx.scope_positions.len(), 3);
        assert_eq!(ctx.scope_positions[0], (2, 2, 3)); // innermost first
        assert_eq!(ctx.scope_positions[1], (1, 1, 4));
        assert_eq!(ctx.scope_positions[2], (0, 0, 5)); // outermost last
    }

    // @trace TEST-CC-29 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_run_all_checks_all_pass() {
        // Arrange: all checks within limits, lifecycle tags match
        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.lifecycle_tags.insert(VRegId(0), LifecycleTag::Global);
        let mut checker = ConstraintChecker::new(&ctx);
        let intervals = vec![
            LiveInterval {
                vreg: VRegId(0),
                kind: VRegKind::Ptr,
                width: SimdWidth::Scalar,
                def_point: 0,
                last_use: 100,
                lifecycle: LifecycleTag::Global,
            },
        ];

        // Act: well within limits
        let result = checker.run_all_checks(50, 16 * 1024, &intervals);

        // Assert
        assert!(result.is_ok());
    }

    // @trace TEST-CC-30 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_run_all_checks_spill_count_only_fails() {
        // Arrange: spill count exceeded but bytes within limit
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act: 200 spills > 128 limit, but 1KB < 32KB bytes limit
        let result = checker.run_all_checks(200, 1024, &[]);

        // Assert: exactly one violation
        let violations = result.expect_err("should fail");
        assert_eq!(violations.len(), 1);
        match &violations[0] {
            ConstraintViolation::TooManySpills { actual, limit } => {
                assert_eq!(*actual, 200);
                assert_eq!(*limit, 128);
            }
            other => panic!("expected TooManySpills, got {:?}", other),
        }
    }

    // @trace TEST-CC-31 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_violation_display_too_many_spills() {
        // Arrange
        let violation = ConstraintViolation::TooManySpills {
            actual: 500,
            limit: 256,
        };

        // Act
        let msg = format!("{}", violation);

        // Assert
        assert!(msg.contains("Too many spill slots"));
        assert!(msg.contains("500"));
        assert!(msg.contains("256"));
        assert!(msg.contains("REQ-LC-012"));
    }

    // @trace TEST-CC-32 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_run_all_checks_empty_intervals_no_lifecycle_violation() {
        // Arrange: default constraints (lifecycle_aware=true) but no intervals
        let ctx = CompilationContext::new(CompilerConstraints::default());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act: no intervals means no lifecycle violations
        let result = checker.run_all_checks(10, 1024, &[]);

        // Assert
        assert!(result.is_ok());
    }

    // @trace TEST-CC-33 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_build_scope_positions_unclosed_scope_ignored() {
        // Arrange: ScopeBegin with no ScopeEnd
        let filler = VmInstr::DeclareVReg { id: VRegId(0), kind: VRegKind::Scalar, width: SimdWidth::Scalar };
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeBegin { scope_id: 42 });
        prog.emit(filler);
        // scope 42 is never closed

        let mut ctx = CompilationContext::new(CompilerConstraints::default());

        // Act
        ctx.build_scope_positions(&prog);

        // Assert: unclosed scope is not recorded
        assert!(ctx.scope_positions.is_empty());
    }

    // @trace TEST-CC-34 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_spill_count_zero_passes() {
        // Arrange
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act & Assert: zero spills always passes
        assert!(checker.check_spill_count(0).is_ok());
    }

    // @trace TEST-CC-35 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_checker_spill_bytes_zero_passes() {
        // Arrange
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act & Assert: zero bytes always passes
        assert!(checker.check_spill_bytes(0).is_ok());
    }

    // @trace TEST-CC-36 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_scope_at_position_with_overlapping_scopes_returns_first_match() {
        // Arrange: two scopes where one is wider
        let filler = VmInstr::DeclareVReg { id: VRegId(0), kind: VRegKind::Scalar, width: SimdWidth::Scalar };
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeBegin { scope_id: 10 }); // pos 0
        prog.emit(VmInstr::ScopeBegin { scope_id: 20 }); // pos 1
        prog.emit(VmInstr::ScopeEnd { scope_id: 20 });   // pos 2
        prog.emit(filler);                                // pos 3
        prog.emit(VmInstr::ScopeEnd { scope_id: 10 });   // pos 4

        let mut ctx = CompilationContext::new(CompilerConstraints::default());
        ctx.build_scope_positions(&prog);

        // Act & Assert: position 3 is in scope 10 (scope 20 ended)
        assert_eq!(ctx.scope_at_position(3), Some(10));
        // Position 1 is in both scopes; returns whichever is encountered first
        let pos1_scope = ctx.scope_at_position(1);
        assert!(pos1_scope == Some(20) || pos1_scope == Some(10));
    }

    // @trace TEST-CC-37 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_run_all_checks_spill_bytes_only_fails() {
        // Arrange: bytes exceeded but count within limit
        let ctx = CompilationContext::new(CompilerConstraints::strict());
        let mut checker = ConstraintChecker::new(&ctx);

        // Act: 10 spills < 128 limit, but 100KB > 32KB bytes limit
        let result = checker.run_all_checks(10, 100 * 1024, &[]);

        // Assert: exactly one violation (bytes only)
        let violations = result.expect_err("should fail");
        assert_eq!(violations.len(), 1);
        match &violations[0] {
            ConstraintViolation::SpillBytesExceeded { actual, limit } => {
                assert_eq!(*actual, 100 * 1024);
                assert_eq!(*limit, 32 * 1024);
            }
            other => panic!("expected SpillBytesExceeded, got {:?}", other),
        }
    }

    // @trace TEST-CC-38 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraints_clone_preserves_values() {
        // Arrange
        let original = CompilerConstraints::strict();

        // Act
        let cloned = original.clone();

        // Assert: clone produces identical values
        assert_eq!(original.lifecycle_aware_alloc, cloned.lifecycle_aware_alloc);
        assert_eq!(original.scoped_spill_alloc, cloned.scoped_spill_alloc);
        assert_eq!(original.enable_post_hoc_verify, cloned.enable_post_hoc_verify);
        assert_eq!(original.max_spill_slots, cloned.max_spill_slots);
        assert_eq!(original.max_spill_bytes, cloned.max_spill_bytes);
    }

    // @trace TEST-CC-39 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_run_all_checks_lifecycle_fails_with_aware_enabled() {
        // Arrange: default constraints (lifecycle_aware=true), missing tag
        let ctx = CompilationContext::new(CompilerConstraints::default());
        let mut checker = ConstraintChecker::new(&ctx);
        let intervals = vec![
            LiveInterval {
                vreg: VRegId(33),
                kind: VRegKind::Vec,
                width: SimdWidth::W256,
                def_point: 5,
                last_use: 20,
                lifecycle: LifecycleTag::BodyLocal,
            },
        ];

        // Act: VRegId(33) not in lifecycle_tags
        let result = checker.run_all_checks(10, 1024, &intervals);

        // Assert
        let violations = result.expect_err("should fail for missing tag");
        assert_eq!(violations.len(), 1);
        match &violations[0] {
            ConstraintViolation::MissingLifecycleTag { vreg } => {
                assert_eq!(*vreg, VRegId(33));
            }
            other => panic!("expected MissingLifecycleTag, got {:?}", other),
        }
    }

    // @trace TEST-CC-40 [req:REQ-LC-012] [level:unit]
    #[test]
    fn test_constraint_violation_clone_preserves_data() {
        // Arrange
        let violation = ConstraintViolation::LifecycleTagMismatch {
            vreg: VRegId(5),
            expected: LifecycleTag::Global,
            actual: LifecycleTag::CrossScope,
        };

        // Act
        let cloned = violation.clone();

        // Assert
        match cloned {
            ConstraintViolation::LifecycleTagMismatch { vreg, expected, actual } => {
                assert_eq!(vreg, VRegId(5));
                assert_eq!(expected, LifecycleTag::Global);
                assert_eq!(actual, LifecycleTag::CrossScope);
            }
            other => panic!("expected LifecycleTagMismatch, got {:?}", other),
        }
    }
}
