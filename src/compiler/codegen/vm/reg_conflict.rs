//! 物理寄存器冲突检测器 (REQ-LC-006)
//!
//! 检测同一物理寄存器在同一程序点被两个活跃 VReg 占用的情况。
//! 提供精确的冲突位置和涉及 VReg 报告。
//!
//! # 使用场景
//!
//! - 寄存器分配后的验证阶段（集成了 `verify_after_alloc`）
//! - 调试分配器 bug 时独立调用
//! - 冲突格式化为可读错误信息，指向冲突的程序位置和寄存器

use std::collections::{HashMap, HashSet};

use super::instr::*;
use super::isa_profile::*;
use super::reg_alloc::{LiveInterval, RegAllocation};

/// 物理寄存器冲突描述 (REQ-LC-006)
#[derive(Debug, Clone)]
pub struct RegConflict {
    /// 冲突涉及的物理寄存器
    pub phys_reg: PhysReg,
    /// 冲突涉及的 VReg 列表（至少 2 个）
    pub vregs: Vec<VRegId>,
    /// 冲突发生的程序点（指令索引）
    pub program_point: usize,
}

impl std::fmt::Display for RegConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reg_desc = match self.phys_reg {
            PhysReg::Gpr(g) => format!("GPR({})", g.0),
            PhysReg::Vec(v) => format!("VEC({})", v.0),
            PhysReg::Tile(t) => format!("TILE({})", t.0),
            PhysReg::Mask(m) => format!("MASK({})", m.0),
            PhysReg::Spilled(slot) => format!("SPILLED(slot {})", slot),
        };
        let vreg_list: Vec<String> = self.vregs.iter().map(|v| format!("VRegId({})", v.0)).collect();
        write!(
            f,
            "reg conflict at instr[{}]: physical register {} is occupied by {}",
            self.program_point,
            reg_desc,
            vreg_list.join(", "),
        )
    }
}

/// 检测物理寄存器冲突 (REQ-LC-006)
///
/// 扫描 VmProgram 的每个程序点，检查同一物理寄存器是否被两个或更多
/// 同时活跃的 VReg 占用。
///
/// # 参数
///
/// - `prog`: VmProgram — 指令序列
/// - `alloc`: RegAllocation — 寄存器分配结果（VRegId → PhysReg）
/// - `intervals`: LiveInterval 切片 — 每个 VReg 的活跃区间
///
/// # 返回值
///
/// `Vec<RegConflict>` — 所有检测到的冲突列表。空 vec 表示无冲突。
///
/// # 算法
///
/// 1. 对每个程序点 i（指令索引），计算所有在该点活跃的 VReg
///    （活跃 = def_point <= i <= last_use）
/// 2. 对每组活跃 VReg，检查其物理寄存器映射
/// 3. 若同一物理寄存器被 >= 2 个活跃 VReg 映射，报告冲突
///
/// # 复杂度
///
/// O(P × V) 其中 P = 程序指令数，V = VReg 数。
/// 实际场景中 V 通常 < 500，P < 2000，性能可接受。
pub fn detect_reg_conflicts(
    prog: &VmProgram,
    alloc: &RegAllocation,
    intervals: &[LiveInterval],
) -> Vec<RegConflict> {
    let mut conflicts: Vec<RegConflict> = Vec::new();

    if intervals.is_empty() {
        return conflicts;
    }

    // 构建 VRegId → LiveInterval 的查找表
    let interval_map: HashMap<VRegId, &LiveInterval> = intervals.iter()
        .map(|iv| (iv.vreg, iv))
        .collect();

    // 对每个程序点（指令索引）检查活跃 VReg 的物理寄存器冲突
    for i in 0..prog.instrs.len() {
        // 找出在此程序点活跃的所有 VReg（def_point <= i <= last_use）
        let active_at_point: Vec<VRegId> = interval_map.values()
            .filter(|iv| iv.def_point <= i && i <= iv.last_use)
            .map(|iv| iv.vreg)
            .collect();

        if active_at_point.len() < 2 {
            continue;
        }

        // 按物理寄存器分组：PhysReg → Vec<VRegId>
        let mut phys_to_vregs: HashMap<PhysReg, Vec<VRegId>> = HashMap::new();

        for &vreg in &active_at_point {
            if let Some(phys_reg) = alloc.get(vreg) {
                phys_to_vregs.entry(phys_reg).or_default().push(vreg);
            }
        }

        // 检查每组是否有 >= 2 个 VReg 共享同一物理寄存器
        for (phys_reg, vregs) in phys_to_vregs {
            if vregs.len() >= 2 {
                conflicts.push(RegConflict {
                    phys_reg,
                    vregs,
                    program_point: i,
                });
            }
        }
    }

    // 去重：同一物理寄存器在同一程序点的冲突只报告一次
    deduplicate_conflicts(&mut conflicts);

    conflicts
}

/// 去重：移除完全相同的冲突（同一物理寄存器、同一程序点的重复报告）。
fn deduplicate_conflicts(conflicts: &mut Vec<RegConflict>) {
    let mut seen: HashSet<(PhysReg, usize)> = HashSet::new();
    conflicts.retain(|c| seen.insert((c.phys_reg, c.program_point)));
}

#[cfg(test)]
mod tests {
    use crate::compiler::trace::QuantPrecision;
    use super::super::instr::VecOp;

    use super::super::reg_alloc::RegAllocator;
    use super::*;

    /// 构造一个简单的 VmProgram 用于测试。
    /// prog: 持有两个 VReg (v0=Ptr, v1=Vec)
    /// instrs: [DeclareVReg v0, DeclareVReg v1, VecLoad v1 ← [v0, 0]]
    fn build_simple_prog() -> VmProgram {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);  // VRegId(0)
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);     // VRegId(1)
        prog.emit(VmInstr::VecLoad {
            dst: v1,
            base: v0,
            offset: OffsetExpr::Const(0),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        });
        prog
    }

    /// 对简单的无冲突程序，检测应返回空列表。
    #[test]
    fn test_no_conflict_simple_prog() {
        let prog = build_simple_prog();
        let intervals = RegAllocator::compute_intervals(&prog);

        // 手动构造分配：v0 → GPR(0), v1 → VEC(0)
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(0)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(conflicts.is_empty(), "expected no conflicts, got: {:?}", conflicts);
    }

    /// 两个 VReg 映射到同一物理寄存器且活跃区间重叠 → 应检出冲突。
    #[test]
    fn test_overlap_conflict_detected() {
        let prog = build_simple_prog();
        let intervals = RegAllocator::compute_intervals(&prog);

        // 恶意分配：v0 和 v1 都映射到 GPR(0)
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(0)));
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(!conflicts.is_empty(), "expected conflict, got empty");

        // 验证冲突包含 GPR(0) 和两个 VReg
        let c0 = &conflicts[0];
        assert_eq!(c0.phys_reg, PhysReg::Gpr(PhysGpr(0)));
        assert!(c0.vregs.contains(&VRegId(0)));
        assert!(c0.vregs.contains(&VRegId(1)));
    }

    /// 两个 VReg 不重叠的活跃区间，即使映射到同一物理寄存器也不应冲突。
    #[test]
    fn test_non_overlapping_no_conflict() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);  // VRegId(0)
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);  // VRegId(1)
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar); // VRegId(2)

        // v0 先使用再释放（通过最后一个 use 在 VecLoad 之后）
        prog.emit(VmInstr::VecLoad {
            dst: v0,
            base,
            offset: OffsetExpr::Const(0),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        });
        // v0 最后使用点 = 此 Mov
        prog.emit(VmInstr::Mov {
            dst: base,
            src: v0,
            dtype: QuantPrecision::F32,
        });

        // v1 在 v0 之后使用，区间不重叠
        prog.emit(VmInstr::VecLoad {
            dst: v1,
            base,
            offset: OffsetExpr::Const(32),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        });

        let intervals = RegAllocator::compute_intervals(&prog);

        // v0 和 v1 都映射到 VEC(0) — 但区间不重叠，不应冲突
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(2), PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        // v0: def=2, last_use=3; v1: def=4, last_use=4
        // 没有程序点同时活跃 v0 和 v1 → 无冲突
        assert!(conflicts.is_empty(), "expected no conflicts for non-overlapping intervals, got: {:?}", conflicts);
    }

    /// 冲突报告应包含正确的程序点索引。
    #[test]
    fn test_conflict_reports_correct_position() {
        let mut prog = VmProgram::new();
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar); // VRegId(0)
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);    // VRegId(1)
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);    // VRegId(2)

        // instr[2]: VecLoad v0 (v0 live starts here)
        prog.emit(VmInstr::VecLoad {
            dst: v0,
            base,
            offset: OffsetExpr::Const(0),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        });
        // instr[3]: VecLoad v1 (v1 live starts here, v0 still live)
        prog.emit(VmInstr::VecLoad {
            dst: v1,
            base,
            offset: OffsetExpr::Const(32),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        });
        // instr[4]: VecBinOp uses v0 — extends v0's last_use past v1's def,
        // creating a true overlap when both map to the same physical register.
        let v2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecBinOp {
            dst: v2,
            a: v0,
            b: v1,
            op: VecOp::Add,
            dtype: QuantPrecision::F32,
        });

        let intervals = RegAllocator::compute_intervals(&prog);

        // v0(VRegId(1)) 和 v1(VRegId(2)) 都映射到 VEC(0)
        // v2(VRegId(3)) maps to a different register to isolate the v0/v1 conflict
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(3), PhysReg::Vec(PhysVec(1)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(!conflicts.is_empty(), "expected conflict: v0 and v1 both mapped to Vec(0) with overlapping live ranges (v0 used at instr[4], v1 defined at instr[3])");
    }

    #[test]
    fn test_conflict_empty_prog() {
        let prog = VmProgram::new();
        let alloc = RegAllocation {
            mapping: HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        let conflicts = detect_reg_conflicts(&prog, &alloc, &[]);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_conflict_single_vreg_no_conflict() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256); // VRegId(0)
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar); // VRegId(1)

        prog.emit(VmInstr::VecLoad {
            dst: v0,
            base,
            offset: OffsetExpr::Const(0),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        });

        let intervals = RegAllocator::compute_intervals(&prog);

        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(conflicts.is_empty());
    }

    // ── 7. RegConflict Display format ────────────────────────────────

    #[test]
    fn reg_conflict_display_gpr_format() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Gpr(PhysGpr(5)),
            vregs: vec![VRegId(0), VRegId(3)],
            program_point: 42,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("instr[42]"), "should contain program point");
        assert!(s.contains("GPR(5)"), "should contain physical register");
        assert!(s.contains("VRegId(0)"), "should contain first vreg");
        assert!(s.contains("VRegId(3)"), "should contain second vreg");
    }

    // ── 8. RegConflict Display — Vec register ────────────────────────

    #[test]
    fn reg_conflict_display_vec_format() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Vec(PhysVec(2)),
            vregs: vec![VRegId(1), VRegId(7)],
            program_point: 10,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("VEC(2)"), "should contain VEC register");
    }

    // ── 9. RegConflict Display — Spilled ─────────────────────────────

    #[test]
    fn reg_conflict_display_spilled_format() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Spilled(3),
            vregs: vec![VRegId(2)],
            program_point: 7,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("SPILLED(slot 3)"), "should contain spilled slot");
    }

    // ── 10. RegConflict Display — Tile and Mask ──────────────────────

    #[test]
    fn reg_conflict_display_tile_and_mask() {
        let tile_c = RegConflict {
            phys_reg: PhysReg::Tile(PhysTile(0)),
            vregs: vec![VRegId(0), VRegId(1)],
            program_point: 0,
        };
        let mask_c = RegConflict {
            phys_reg: PhysReg::Mask(PhysMask(1)),
            vregs: vec![VRegId(2), VRegId(3)],
            program_point: 1,
        };
        assert!(format!("{}", tile_c).contains("TILE(0)"));
        assert!(format!("{}", mask_c).contains("MASK(1)"));
    }

    // ── 11. deduplicate_conflicts removes exact duplicates ───────────

    #[test]
    fn deduplicate_removes_duplicates() {
        let mut conflicts = vec![
            RegConflict { phys_reg: PhysReg::Gpr(PhysGpr(0)), vregs: vec![VRegId(0), VRegId(1)], program_point: 5 },
            RegConflict { phys_reg: PhysReg::Gpr(PhysGpr(0)), vregs: vec![VRegId(0), VRegId(1)], program_point: 5 },
            RegConflict { phys_reg: PhysReg::Vec(PhysVec(0)), vregs: vec![VRegId(2)], program_point: 5 },
        ];
        super::deduplicate_conflicts(&mut conflicts);
        assert_eq!(conflicts.len(), 2, "should remove exact duplicate (same phys_reg + program_point)");
    }

    // ── 12. detect_reg_conflicts — empty intervals ───────────────────

    #[test]
    fn detect_empty_intervals_returns_empty() {
        let prog = VmProgram::new();
        let alloc = RegAllocation { mapping: HashMap::new(), spills: vec![], callee_saved_used: vec![] };
        let result = detect_reg_conflicts(&prog, &alloc, &[]);
        assert!(result.is_empty());
    }

    // ── 13. detect — vreg unmapped in alloc is ignored ───────────────

    #[test]
    fn detect_unmapped_vregs_ignored() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::VecBinOp { dst: v0, a: v1, b: v1, op: VecOp::Add, dtype: QuantPrecision::F32 });

        let intervals = RegAllocator::compute_intervals(&prog);
        // v0 mapped but v1 not in mapping → v1 ignored, no conflict
        let mut mapping = HashMap::new();
        mapping.insert(v0, PhysReg::Vec(PhysVec(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(conflicts.is_empty(), "unmapped vregs should not cause conflicts");
    }

    // ── 14. RegConflict clone preserves fields ───────────────────────

    #[test]
    fn reg_conflict_clone_preserves_fields() {
        let original = RegConflict {
            phys_reg: PhysReg::Gpr(PhysGpr(7)),
            vregs: vec![VRegId(0), VRegId(1), VRegId(2)],
            program_point: 100,
        };
        let cloned = original.clone();
        assert_eq!(cloned.phys_reg, original.phys_reg);
        assert_eq!(cloned.vregs, original.vregs);
        assert_eq!(cloned.program_point, original.program_point);
    }

    // ── 15. Three vregs same phys_reg same point → one conflict ──────

    #[test]
    fn three_vregs_same_phys_reg_one_conflict() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // All three used in one op → overlap
        prog.emit(VmInstr::VecBinOp { dst: v2, a: v0, b: v1, op: VecOp::Mul, dtype: QuantPrecision::F32 });

        let intervals = RegAllocator::compute_intervals(&prog);
        let mut mapping = HashMap::new();
        mapping.insert(v0, PhysReg::Vec(PhysVec(0)));
        mapping.insert(v1, PhysReg::Vec(PhysVec(0)));
        mapping.insert(v2, PhysReg::Vec(PhysVec(1)));
        mapping.insert(base, PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        // v0 and v1 both on Vec(0) and active at the same point
        let vec0_conflicts: Vec<_> = conflicts.iter().filter(|c| c.phys_reg == PhysReg::Vec(PhysVec(0))).collect();
        assert!(!vec0_conflicts.is_empty(), "should detect v0/v1 conflict on Vec(0)");
    }

    // ── 16. PhysReg equality across variants ─────────────────────────

    #[test]
    fn phys_reg_different_variants_not_equal() {
        let gpr = PhysReg::Gpr(PhysGpr(0));
        let vec = PhysReg::Vec(PhysVec(0));
        let tile = PhysReg::Tile(PhysTile(0));
        let mask = PhysReg::Mask(PhysMask(0));
        let spilled = PhysReg::Spilled(0);

        assert_ne!(gpr, vec);
        assert_ne!(gpr, tile);
        assert_ne!(vec, mask);
        assert_ne!(mask, spilled);
    }

    // ── 17. detect_reg_conflicts — single vreg interval no conflict ──

    #[test]
    fn single_interval_never_conflicts() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::VecLoad { dst: v0, base, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });

        let intervals = RegAllocator::compute_intervals(&prog);
        // Only one vec vreg (v0) mapped to Vec(0) — can't conflict with itself
        let mut mapping = HashMap::new();
        mapping.insert(v0, PhysReg::Vec(PhysVec(0)));
        mapping.insert(base, PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(conflicts.is_empty(), "single vreg on its own phys_reg should not conflict");
    }

    // ── 18. RegConflict with zero program_point ──────────────────────

    #[test]
    fn reg_conflict_program_point_zero_display() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Gpr(PhysGpr(0)),
            vregs: vec![VRegId(0), VRegId(1)],
            program_point: 0,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("instr[0]"), "should display program_point 0");
    }

    // ── 19. RegConflict vregs order preserved ────────────────────────

    #[test]
    fn reg_conflict_vregs_order_preserved_in_display() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Vec(PhysVec(0)),
            vregs: vec![VRegId(10), VRegId(20), VRegId(30)],
            program_point: 5,
        };
        let s = format!("{}", conflict);
        let pos10 = s.find("VRegId(10)").unwrap();
        let pos20 = s.find("VRegId(20)").unwrap();
        let pos30 = s.find("VRegId(30)").unwrap();
        assert!(pos10 < pos20 && pos20 < pos30, "vregs should appear in order");
    }

    // ── 20. Additional tests ─────────────────────────────────────────

    #[test]
    fn reg_conflict_debug_format() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Gpr(PhysGpr(3)),
            vregs: vec![VRegId(1), VRegId(2)],
            program_point: 99,
        };
        let debug_str = format!("{:?}", conflict);
        assert!(debug_str.contains("RegConflict"), "Debug output should contain struct name");
        assert!(debug_str.contains("Gpr"), "Debug output should contain Gpr variant");
    }

    #[test]
    fn reg_conflict_display_single_vreg() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Mask(PhysMask(2)),
            vregs: vec![VRegId(42)],
            program_point: 15,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("MASK(2)"), "should contain MASK register");
        assert!(s.contains("VRegId(42)"), "should contain the single vreg");
    }

    #[test]
    fn phys_reg_spilled_equality() {
        let a = PhysReg::Spilled(5);
        let b = PhysReg::Spilled(5);
        let c = PhysReg::Spilled(6);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn phys_reg_gpr_vec_tile_not_equal_same_index() {
        // Different register classes with same numeric index are not equal
        let gpr = PhysReg::Gpr(PhysGpr(0));
        let vec = PhysReg::Vec(PhysVec(0));
        let tile = PhysReg::Tile(PhysTile(0));
        let mask = PhysReg::Mask(PhysMask(0));
        assert_ne!(gpr, vec);
        assert_ne!(vec, tile);
        assert_ne!(tile, mask);
        assert_ne!(gpr, mask);
    }

    #[test]
    fn detect_reg_conflicts_multiple_points() {
        // Two pairs of conflicts at different program points
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // instr[2]: VecLoad v0
        prog.emit(VmInstr::VecLoad {
            dst: v0, base, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        // instr[3]: VecLoad v1 (v0 still live)
        prog.emit(VmInstr::VecLoad {
            dst: v1, base, offset: OffsetExpr::Const(32), width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        // instr[4]: VecBinOp uses both v0 and v1 — they overlap here
        let v2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecBinOp { dst: v2, a: v0, b: v1, op: VecOp::Add, dtype: QuantPrecision::F32 });

        let intervals = RegAllocator::compute_intervals(&prog);
        let mut mapping = HashMap::new();
        mapping.insert(v0, PhysReg::Vec(PhysVec(0)));
        mapping.insert(v1, PhysReg::Vec(PhysVec(0))); // conflict: same phys reg
        mapping.insert(v2, PhysReg::Vec(PhysVec(1)));
        mapping.insert(base, PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        // Should detect at least one conflict at program point where both v0 and v1 are live
        assert!(!conflicts.is_empty(), "should detect conflicts");
        // All conflicts should involve Vec(0)
        for c in &conflicts {
            assert_eq!(c.phys_reg, PhysReg::Vec(PhysVec(0)));
        }
    }

    #[test]
    fn deduplicate_preserves_distinct_conflicts() {
        let mut conflicts = vec![
            RegConflict { phys_reg: PhysReg::Gpr(PhysGpr(0)), vregs: vec![VRegId(0), VRegId(1)], program_point: 5 },
            RegConflict { phys_reg: PhysReg::Vec(PhysVec(0)), vregs: vec![VRegId(2), VRegId(3)], program_point: 5 },
            RegConflict { phys_reg: PhysReg::Gpr(PhysGpr(0)), vregs: vec![VRegId(0), VRegId(1)], program_point: 10 },
        ];
        super::deduplicate_conflicts(&mut conflicts);
        // GPR(0)@5 is deduplicated to 1, VEC(0)@5 kept, GPR(0)@10 kept → 3
        assert_eq!(conflicts.len(), 3, "distinct conflicts should all be kept");
    }

    #[test]
    fn detect_no_conflict_when_vregs_on_different_phys_regs() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::VecBinOp { dst: v0, a: v1, b: v1, op: VecOp::Add, dtype: QuantPrecision::F32 });

        let intervals = RegAllocator::compute_intervals(&prog);
        let mut mapping = HashMap::new();
        mapping.insert(v0, PhysReg::Vec(PhysVec(0)));
        mapping.insert(v1, PhysReg::Vec(PhysVec(1))); // different phys reg
        mapping.insert(base, PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(conflicts.is_empty(), "different phys regs should not conflict");
    }

    #[test]
    fn reg_conflict_display_format_contains_occupied_by() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Tile(PhysTile(1)),
            vregs: vec![VRegId(5), VRegId(6)],
            program_point: 20,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("occupied by"), "display should mention 'occupied by'");
        assert!(s.contains("instr[20]"), "display should contain program point");
    }

    #[test]
    fn detect_spilled_reg_conflict() {
        // Two vregs both spilled to same slot — should be detected
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::VecBinOp { dst: v0, a: v1, b: v1, op: VecOp::Add, dtype: QuantPrecision::F32 });

        let intervals = RegAllocator::compute_intervals(&prog);
        let mut mapping = HashMap::new();
        mapping.insert(v0, PhysReg::Spilled(3));
        mapping.insert(v1, PhysReg::Spilled(3)); // same spill slot — conflict
        mapping.insert(base, PhysReg::Gpr(PhysGpr(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        let conflicts = detect_reg_conflicts(&prog, &alloc, &intervals);
        assert!(!conflicts.is_empty(), "same spill slot should be detected as conflict");
        assert_eq!(conflicts[0].phys_reg, PhysReg::Spilled(3));
    }

    #[test]
    fn detect_reg_conflicts_large_program_point() {
        let conflict = RegConflict {
            phys_reg: PhysReg::Gpr(PhysGpr(0)),
            vregs: vec![VRegId(0)],
            program_point: 99999,
        };
        let s = format!("{}", conflict);
        assert!(s.contains("99999"), "should display large program point");
    }
}
