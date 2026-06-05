//! Diamond-branch detection and Select-node merging.
//!
//! Phase 4 of the control-flow upgrade: detects "diamond" patterns in the CFG
//! (compare → conditional branch → two short paths → merge point) and merges
//! them into `SymValue::Select` nodes. The simplifier then reduces
//! `Select(a > b, a, b) → Max(a, b)` etc.

use std::collections::HashMap;
use super::cfg::{BasicBlock, BlockId, BranchKind, ControlFlowGraph, Terminator};
use super::engine::SymbolicExecutor;
use super::sym_value::{SelectKind, SymValue};

// ---------------------------------------------------------------------------
// Diamond pattern detection
// ---------------------------------------------------------------------------

/// A detected diamond: cond-branch splits into two paths that rejoin.
#[derive(Debug)]
pub struct Diamond {
    /// The block containing the comparison + conditional branch.
    pub cond_block: BlockId,
    /// The "taken" path block(s) — executed when condition is true.
    pub taken_block: BlockId,
    /// The "fallthrough" path block — executed when condition is false.
    pub fallthrough_block: BlockId,
    /// The merge point where both paths rejoin.
    pub merge_block: BlockId,
    /// The branch condition kind.
    pub kind: BranchKind,
}

/// Find all diamond patterns in a set of blocks (e.g., a loop body).
///
/// A diamond is: block B ends with CondBranch { taken: T, fallthrough: F },
/// and both T and F have exactly one successor, and that successor is the same
/// merge block M. Additionally, T and F must each be a single basic block
/// (no further branching within the diamond arms).
pub fn find_diamonds(
    cfg: &ControlFlowGraph,
    block_ids: &[BlockId],
) -> Vec<Diamond> {
    let block_set: std::collections::BTreeSet<BlockId> = block_ids.iter().copied().collect();
    let mut diamonds = Vec::new();

    for &bid in block_ids {
        let block = match cfg.blocks.get(&bid) {
            Some(b) => b,
            None => continue,
        };

        if let Terminator::CondBranch { kind, taken, fallthrough } = &block.terminator {
            // Both arms must be in our block set.
            if !block_set.contains(taken) || !block_set.contains(fallthrough) {
                continue;
            }

            // Each arm must have exactly one successor (the merge point).
            let taken_succs = cfg.successors.get(taken).map(|v| v.as_slice()).unwrap_or(&[]);
            let fall_succs = cfg.successors.get(fallthrough).map(|v| v.as_slice()).unwrap_or(&[]);

            if taken_succs.len() != 1 || fall_succs.len() != 1 {
                continue;
            }

            let merge_from_taken = taken_succs[0];
            let merge_from_fall = fall_succs[0];

            // Both arms must merge to the same block.
            if merge_from_taken != merge_from_fall {
                continue;
            }

            // The merge block must also be in our block set (or be an exit).
            let merge = merge_from_taken;

            diamonds.push(Diamond {
                cond_block: bid,
                taken_block: *taken,
                fallthrough_block: *fallthrough,
                merge_block: merge,
                kind: kind.clone(),
            });
        }
    }

    diamonds
}

// ---------------------------------------------------------------------------
// Branch merging: execute both paths and build Select nodes
// ---------------------------------------------------------------------------

/// Execute both arms of a diamond and merge the results into Select nodes.
///
/// Returns the merged executor state at the merge point.
pub fn merge_diamond(
    diamond: &Diamond,
    cfg: &ControlFlowGraph,
    exec: &SymbolicExecutor,
) -> SymbolicExecutor {
    let select_kind = branch_kind_to_select(&diamond.kind);

    // Get the comparison flags from the executor (set by ucomiss/comiss).
    let (cond_lhs, cond_rhs) = exec.get_flags()
        .unwrap_or_else(|| (
            SymValue::Unknown("cmp_lhs".into()),
            SymValue::Unknown("cmp_rhs".into()),
        ));

    // Execute the "taken" path.
    let mut taken_exec = exec.snapshot();
    if let Some(block) = cfg.blocks.get(&diamond.taken_block) {
        execute_block(&mut taken_exec, block);
    }

    // Execute the "fallthrough" path.
    let mut fall_exec = exec.snapshot();
    if let Some(block) = cfg.blocks.get(&diamond.fallthrough_block) {
        execute_block(&mut fall_exec, block);
    }

    // Merge: for each XMM register, if the two paths produced different
    // values, create a Select node.
    let taken_state = taken_exec.xmm_state();
    let fall_state = fall_exec.xmm_state();
    let pre_state = exec.xmm_state();

    let mut merged = exec.snapshot();

    // Collect all registers that appear in either path.
    let mut all_regs: Vec<String> = taken_state.keys()
        .chain(fall_state.keys())
        .chain(pre_state.keys())
        .cloned()
        .collect();
    all_regs.sort();
    all_regs.dedup();

    for reg in &all_regs {
        let taken_val = taken_state.get(reg)
            .or_else(|| pre_state.get(reg));
        let fall_val = fall_state.get(reg)
            .or_else(|| pre_state.get(reg));

        match (taken_val, fall_val) {
            (Some(tv), Some(fv)) => {
                let tv_s = format!("{tv}");
                let fv_s = format!("{fv}");
                if tv_s == fv_s {
                    // Same value on both paths — no select needed.
                    merged.set(reg, tv.clone());
                } else {
                    // Different values → Select node.
                    let select = SymValue::Select {
                        kind: select_kind,
                        cond_lhs: Box::new(cond_lhs.clone()),
                        cond_rhs: Box::new(cond_rhs.clone()),
                        true_val: Box::new(tv.clone()),
                        false_val: Box::new(fv.clone()),
                    };
                    merged.set(reg, select);
                }
            }
            (Some(v), None) | (None, Some(v)) => {
                merged.set(reg, v.clone());
            }
            (None, None) => {}
        }
    }

    // Also merge stack spill slots.
    let taken_stack = taken_exec.stack_state();
    let fall_stack = fall_exec.stack_state();
    let pre_stack = exec.stack_state();

    let mut all_offsets: Vec<i64> = taken_stack.keys()
        .chain(fall_stack.keys())
        .chain(pre_stack.keys())
        .copied()
        .collect();
    all_offsets.sort();
    all_offsets.dedup();

    for offset in &all_offsets {
        let taken_val = taken_stack.get(offset)
            .or_else(|| pre_stack.get(offset));
        let fall_val = fall_stack.get(offset)
            .or_else(|| pre_stack.get(offset));

        match (taken_val, fall_val) {
            (Some(tv), Some(fv)) => {
                let tv_s = format!("{tv}");
                let fv_s = format!("{fv}");
                if tv_s != fv_s {
                    let select = SymValue::Select {
                        kind: select_kind,
                        cond_lhs: Box::new(cond_lhs.clone()),
                        cond_rhs: Box::new(cond_rhs.clone()),
                        true_val: Box::new(tv.clone()),
                        false_val: Box::new(fv.clone()),
                    };
                    merged.set_stack(*offset, select);
                } else {
                    merged.set_stack(*offset, tv.clone());
                }
            }
            (Some(v), None) | (None, Some(v)) => {
                merged.set_stack(*offset, v.clone());
            }
            (None, None) => {}
        }
    }

    merged
}

/// Execute all instructions in a basic block (skipping branch mnemonics).
fn execute_block(exec: &mut SymbolicExecutor, block: &BasicBlock) {
    for insn in &block.instructions {
        let ops: Vec<&str> = insn.operands.iter().map(|s| s.as_str()).collect();
        if is_branch_mnemonic(&insn.mnemonic) {
            continue;
        }
        let _ = exec.step(&insn.mnemonic, &ops);
    }
}

fn is_branch_mnemonic(m: &str) -> bool {
    matches!(
        m,
        // x86_64
        "je" | "jne" | "jb" | "jbe" | "ja" | "jae"
            | "jl" | "jle" | "jg" | "jge"
            | "jmp" | "js" | "jns" | "jp" | "jnp"
            | "ret"
            // AArch64
            | "b" | "bl" | "b.eq" | "b.ne" | "b.gt" | "b.ge" | "b.lt" | "b.le"
            | "b.hi" | "b.hs" | "b.lo" | "b.ls" | "b.mi" | "b.pl"
            | "cbz" | "cbnz" | "tbz" | "tbnz"
    )
}

/// Map x86 BranchKind to SelectKind.
///
/// For float comparisons (ucomiss), the unsigned conditions map to:
///   JA/JNBE → "above" → `a > b`  → SelectKind::Gt
///   JAE/JNB → "above or equal" → `a >= b` → SelectKind::Ge
///   JB/JNAE → "below" → `a < b`  → SelectKind::Lt
///   JBE/JNA → "below or equal" → `a <= b` → SelectKind::Le
fn branch_kind_to_select(kind: &BranchKind) -> SelectKind {
    match kind {
        BranchKind::Above => SelectKind::Gt,
        BranchKind::AboveEqual => SelectKind::Ge,
        BranchKind::Below => SelectKind::Lt,
        BranchKind::BelowEqual => SelectKind::Le,
        BranchKind::Greater => SelectKind::Gt,
        BranchKind::GreaterEqual => SelectKind::Ge,
        BranchKind::Less => SelectKind::Lt,
        BranchKind::LessEqual => SelectKind::Le,
        BranchKind::Equal => SelectKind::Eq,
        BranchKind::NotEqual => SelectKind::Ne,
        // Parity/sign flags — rare for float select, map to Ne as fallback.
        BranchKind::Sign | BranchKind::NotSign
        | BranchKind::Parity | BranchKind::NotParity => SelectKind::Ne,
    }
}

// ---------------------------------------------------------------------------
// Integration: execute blocks with diamond merging
// ---------------------------------------------------------------------------

/// Execute a sequence of blocks, automatically detecting and merging diamonds.
///
/// This replaces the naive "execute all blocks linearly" approach in
/// `analyze_single_loop`. Blocks within diamond arms are not executed
/// linearly — instead, both paths are forked and merged via Select nodes.
pub fn execute_blocks_with_merging(
    block_ids: &[BlockId],
    cfg: &ControlFlowGraph,
    exec: &mut SymbolicExecutor,
) {
    let diamonds = find_diamonds(cfg, block_ids);

    // Build a set of blocks that are part of diamond arms (should not be
    // executed linearly).
    let mut diamond_blocks: HashMap<BlockId, usize> = HashMap::new();
    for (i, d) in diamonds.iter().enumerate() {
        diamond_blocks.insert(d.taken_block, i);
        diamond_blocks.insert(d.fallthrough_block, i);
    }

    // Map cond_block → diamond index for triggering merge.
    let mut cond_to_diamond: HashMap<BlockId, usize> = HashMap::new();
    for (i, d) in diamonds.iter().enumerate() {
        cond_to_diamond.insert(d.cond_block, i);
    }

    // Track which diamonds have been merged.
    let mut merged: Vec<bool> = vec![false; diamonds.len()];

    for &bid in block_ids {
        // Skip blocks that are diamond arms — they'll be handled by merge.
        if diamond_blocks.contains_key(&bid) {
            continue;
        }

        // Execute this block normally.
        if let Some(block) = cfg.blocks.get(&bid) {
            execute_block(exec, block);
        }

        // If this block is a diamond's cond_block, trigger the merge.
        if let Some(&di) = cond_to_diamond.get(&bid) {
            if !merged[di] {
                let result = merge_diamond(&diamonds[di], cfg, exec);
                exec.restore(&result);
                merged[di] = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::symexec::cfg::DecodedInsn;
    use crate::compiler::symexec::sym_value::SymValue;

    #[test]
    fn test_select_simplifies_to_max() {
        // Select(a > b, a, b) → Max(a, b)
        let a = SymValue::Param(0);
        let b = SymValue::Const(0.0);
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(a.clone()),
            false_val: Box::new(b.clone()),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Max(_, _)),
            "expected Max, got: {simplified}"
        );
    }

    #[test]
    fn test_select_simplifies_to_min() {
        // Select(a < b, a, b) → Min(a, b)
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Lt,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(a.clone()),
            false_val: Box::new(b.clone()),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Min(_, _)),
            "expected Min, got: {simplified}"
        );
    }

    #[test]
    fn test_select_swapped_gt_to_min() {
        // Select(a > b, b, a) → Min(a, b)
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(b.clone()),
            false_val: Box::new(a.clone()),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Min(_, _)),
            "expected Min, got: {simplified}"
        );
    }

    #[test]
    fn test_select_const_condition() {
        // Select(3.0 > 1.0, param(0), param(1)) → param(0)
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Const(3.0)),
            cond_rhs: Box::new(SymValue::Const(1.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Param(0)),
            "expected Param(0), got: {simplified}"
        );
    }

    #[test]
    fn test_select_same_branches_collapse() {
        // Select(a > b, x, x) → x
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(2)),
            false_val: Box::new(SymValue::Param(2)),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Param(2)),
            "expected Param(2), got: {simplified}"
        );
    }

    #[test]
    fn test_relu_diamond_pattern() {
        // Simulate ReLU as diamond: ucomiss xmm0, xmm1(=0); ja .taken
        // Taken: xmm0 stays as param(0)
        // Fallthrough: xmm0 = 0.0
        // Merge: Select(param(0) > 0.0, param(0), 0.0) → Max(param(0), 0.0)
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("xorps", &["xmm1", "xmm1"]).unwrap(); // xmm1 = 0.0
        exec.step("ucomiss", &["xmm0", "xmm1"]).unwrap(); // flags: param(0) vs 0.0

        let (cond_lhs, cond_rhs) = exec.get_flags().unwrap();

        // Build Select manually (as merge_diamond would).
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(cond_lhs),
            cond_rhs: Box::new(cond_rhs),
            true_val: Box::new(SymValue::Param(0)),       // taken: keep xmm0
            false_val: Box::new(SymValue::Const(0.0)),     // fall: xmm0 = 0
        };

        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Max(_, _)),
            "ReLU diamond should simplify to Max, got: {simplified}"
        );

        // Verify it's Max(param(0), 0.0)
        let s = format!("{simplified}");
        assert!(s.contains("param(0)"), "should contain param(0): {s}");
        assert!(s.contains("0.0"), "should contain 0.0: {s}");
    }

    // @trace TEST-BM-07 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_find_diamonds_detects_simple_diamond() {
        // Arrange: Build a CFG with a diamond pattern:
        //   B0 (cond branch) → B1 (taken), B2 (fallthrough)
        //   B1 → B3 (merge), B2 → B3 (merge)
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Above,
                taken: b1,
                fallthrough: b2,
            },
        };
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2,
            start_addr: 20,
            end_addr: 30,
            instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3,
            start_addr: 30,
            end_addr: 40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: b0,
            successors,
            predecessors: std::collections::BTreeMap::new(),
        };

        let block_ids = vec![b0, b1, b2, b3];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert
        assert_eq!(diamonds.len(), 1, "should find exactly one diamond");
        let d = &diamonds[0];
        assert_eq!(d.cond_block, b0);
        assert_eq!(d.taken_block, b1);
        assert_eq!(d.fallthrough_block, b2);
        assert_eq!(d.merge_block, b3);
        assert_eq!(d.kind, BranchKind::Above);
    }

    // @trace TEST-BM-08 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_find_diamonds_no_diamond_with_divergent_merge() {
        // Arrange: B0 branches to B1 and B2, but they merge to different blocks
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);
        let b4 = BlockId(4);

        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Greater,
                taken: b1,
                fallthrough: b2,
            },
        };
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2,
            start_addr: 20,
            end_addr: 30,
            instructions: vec![],
            terminator: Terminator::Jump(b4),
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b4]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: b0,
            successors,
            predecessors: std::collections::BTreeMap::new(),
        };

        let block_ids = vec![b0, b1, b2, b3, b4];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: no diamond because the merge points differ
        assert!(diamonds.is_empty(), "divergent merge should not form a diamond");
    }

    // @trace TEST-BM-09 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_find_diamonds_empty_and_missing_blocks() {
        // Arrange: empty block list
        let cfg = ControlFlowGraph {
            blocks: std::collections::BTreeMap::new(),
            entry: BlockId(0),
            successors: std::collections::BTreeMap::new(),
            predecessors: std::collections::BTreeMap::new(),
        };

        // Act & Assert: empty input
        let diamonds = find_diamonds(&cfg, &[]);
        assert!(diamonds.is_empty(), "empty block list should yield no diamonds");

        // Arrange: blocks with IDs not present in the CFG
        let block_ids = vec![BlockId(99), BlockId(100)];

        // Act & Assert: missing blocks should be skipped gracefully
        let diamonds = find_diamonds(&cfg, &block_ids);
        assert!(diamonds.is_empty(), "missing blocks should yield no diamonds");
    }

    // @trace TEST-BM-11 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_branch_kind_to_select_all_mappings() {
        // Arrange & Act & Assert: verify every BranchKind maps to the correct SelectKind
        assert_eq!(branch_kind_to_select(&BranchKind::Above), SelectKind::Gt);
        assert_eq!(branch_kind_to_select(&BranchKind::AboveEqual), SelectKind::Ge);
        assert_eq!(branch_kind_to_select(&BranchKind::Below), SelectKind::Lt);
        assert_eq!(branch_kind_to_select(&BranchKind::BelowEqual), SelectKind::Le);
        assert_eq!(branch_kind_to_select(&BranchKind::Greater), SelectKind::Gt);
        assert_eq!(branch_kind_to_select(&BranchKind::GreaterEqual), SelectKind::Ge);
        assert_eq!(branch_kind_to_select(&BranchKind::Less), SelectKind::Lt);
        assert_eq!(branch_kind_to_select(&BranchKind::LessEqual), SelectKind::Le);
        assert_eq!(branch_kind_to_select(&BranchKind::Equal), SelectKind::Eq);
        assert_eq!(branch_kind_to_select(&BranchKind::NotEqual), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::Sign), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::NotSign), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::Parity), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::NotParity), SelectKind::Ne);
    }

    // @trace TEST-BM-12 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_is_branch_mnemonic_x86_and_aarch64() {
        // Arrange & Act & Assert: x86_64 branch mnemonics
        assert!(is_branch_mnemonic("je"));
        assert!(is_branch_mnemonic("jne"));
        assert!(is_branch_mnemonic("jb"));
        assert!(is_branch_mnemonic("jbe"));
        assert!(is_branch_mnemonic("ja"));
        assert!(is_branch_mnemonic("jae"));
        assert!(is_branch_mnemonic("jl"));
        assert!(is_branch_mnemonic("jle"));
        assert!(is_branch_mnemonic("jg"));
        assert!(is_branch_mnemonic("jge"));
        assert!(is_branch_mnemonic("jmp"));
        assert!(is_branch_mnemonic("js"));
        assert!(is_branch_mnemonic("jns"));
        assert!(is_branch_mnemonic("jp"));
        assert!(is_branch_mnemonic("jnp"));
        assert!(is_branch_mnemonic("ret"));

        // AArch64 branch mnemonics
        assert!(is_branch_mnemonic("b"));
        assert!(is_branch_mnemonic("bl"));
        assert!(is_branch_mnemonic("b.eq"));
        assert!(is_branch_mnemonic("b.ne"));
        assert!(is_branch_mnemonic("b.gt"));
        assert!(is_branch_mnemonic("b.ge"));
        assert!(is_branch_mnemonic("b.lt"));
        assert!(is_branch_mnemonic("b.le"));
        assert!(is_branch_mnemonic("b.hi"));
        assert!(is_branch_mnemonic("b.hs"));
        assert!(is_branch_mnemonic("b.lo"));
        assert!(is_branch_mnemonic("b.ls"));
        assert!(is_branch_mnemonic("b.mi"));
        assert!(is_branch_mnemonic("b.pl"));
        assert!(is_branch_mnemonic("cbz"));
        assert!(is_branch_mnemonic("cbnz"));
        assert!(is_branch_mnemonic("tbz"));
        assert!(is_branch_mnemonic("tbnz"));

        // Non-branch mnemonics
        assert!(!is_branch_mnemonic("addss"));
        assert!(!is_branch_mnemonic("ucomiss"));
        assert!(!is_branch_mnemonic("movss"));
        assert!(!is_branch_mnemonic("xorps"));
        assert!(!is_branch_mnemonic("nop"));
    }

    // @trace TEST-BM-13 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_select_ge_simplifies_to_max() {
        // Arrange: Select(a >= b, a, b) should simplify to Max(a, b)
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(a.clone()),
            false_val: Box::new(b.clone()),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Max(_, _)),
            "Select(a >= b, a, b) should simplify to Max, got: {simplified}"
        );
    }

    // @trace TEST-BM-14 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_select_le_simplifies_to_min() {
        // Arrange: Select(a <= b, a, b) should simplify to Min(a, b)
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Le,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(a.clone()),
            false_val: Box::new(b.clone()),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Min(_, _)),
            "Select(a <= b, a, b) should simplify to Min, got: {simplified}"
        );
    }

    // @trace TEST-BM-15 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_select_eq_const_true_folds() {
        // Arrange: Select(5.0 == 5.0, param(0), param(1)) → param(0) (condition is true)
        let select = SymValue::Select {
            kind: SelectKind::Eq,
            cond_lhs: Box::new(SymValue::Const(5.0)),
            cond_rhs: Box::new(SymValue::Const(5.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(0)),
            "Select(5.0 == 5.0, a, b) should fold to a, got: {simplified}"
        );
    }

    // @trace TEST-BM-16 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_select_ne_const_false_folds_to_false_branch() {
        // Arrange: Select(3.0 != 3.0, param(0), param(1)) → param(1) (condition is false)
        let select = SymValue::Select {
            kind: SelectKind::Ne,
            cond_lhs: Box::new(SymValue::Const(3.0)),
            cond_rhs: Box::new(SymValue::Const(3.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(1)),
            "Select(3.0 != 3.0, a, b) should fold to b, got: {simplified}"
        );
    }

    // @trace TEST-BM-17 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_select_non_foldable_preserves_structure() {
        // Arrange: Select with Eq/Ne and symbolic condition that cannot be simplified
        let select = SymValue::Select {
            kind: SelectKind::Eq,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(2)),
            false_val: Box::new(SymValue::Param(3)),
        };

        // Act
        let simplified = select.simplify();

        // Assert: should remain a Select since Eq with symbolic args doesn't match Max/Min
        assert!(
            matches!(simplified, SymValue::Select { kind: SelectKind::Eq, .. }),
            "symbolic Eq Select should stay as Select, got: {simplified}"
        );
    }

    // @trace TEST-BM-18 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_merge_diamond_produces_select_for_diverging_xmm() {
        // Arrange: Build a real CFG diamond with instructions that diverge on xmm0
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Taken path: xorps xmm0, xmm0 → xmm0 = 0.0
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        // Fallthrough path: addss xmm0, xmm1 → xmm0 = param(0) + param(1)
        let block2 = BasicBlock {
            id: b2,
            start_addr: 20,
            end_addr: 30,
            instructions: vec![DecodedInsn {
                mnemonic: "addss".to_string(),
                operands: vec!["xmm0".to_string(), "xmm1".to_string()],
                addr: 20,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Above,
                taken: b1,
                fallthrough: b2,
            },
        };
        let block3 = BasicBlock {
            id: b3,
            start_addr: 30,
            end_addr: 40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: b0,
            successors,
            predecessors: std::collections::BTreeMap::new(),
        };

        let diamond = Diamond {
            cond_block: b0,
            taken_block: b1,
            fallthrough_block: b2,
            merge_block: b3,
            kind: BranchKind::Above,
        };

        // Executor with 2 float args: xmm0=Param(0), xmm1=Param(1)
        let exec = SymbolicExecutor::new(2, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 should be a Select (values differ on two paths)
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { .. }),
            "xmm0 should be a Select node after merging diverging paths, got: {xmm0}"
        );
        // xmm1 should remain unchanged on both paths
        let xmm1 = merged.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Param(1)),
            "xmm1 should remain Param(1) unchanged, got: {xmm1}"
        );
    }

    // @trace TEST-BM-19 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_merge_diamond_no_flags_uses_unknown_condition() {
        // Arrange: diamond with an executor that has no comparison flags set
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Greater,
                taken: b1,
                fallthrough: b2,
            },
        };
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2,
            start_addr: 20,
            end_addr: 30,
            instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3,
            start_addr: 30,
            end_addr: 40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: b0,
            successors,
            predecessors: std::collections::BTreeMap::new(),
        };

        let diamond = Diamond {
            cond_block: b0,
            taken_block: b1,
            fallthrough_block: b2,
            merge_block: b3,
            kind: BranchKind::Greater,
        };

        // Executor with no flags set (no ucomiss executed)
        let exec = SymbolicExecutor::new(1, 0);
        assert!(exec.get_flags().is_none(), "precondition: no flags should be set");

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: merge should still work, producing Select with Unknown condition
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { .. }),
            "xmm0 should be a Select even without flags, got: {xmm0}"
        );
        // The condition lhs should be Unknown since no flags were set
        if let SymValue::Select { cond_lhs, .. } = &xmm0 {
            assert!(
                matches!(cond_lhs.as_ref(), SymValue::Unknown(_)),
                "condition lhs should be Unknown when no flags set, got: {cond_lhs}"
            );
        }
    }

    // @trace TEST-BM-20 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_execute_blocks_with_merging_skips_diamond_arms() {
        // Arrange: 4-block diamond where only cond and merge blocks execute linearly
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // B1 (taken arm): set xmm0 = 0
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        // B2 (fallthrough arm): add xmm0, xmm1
        let block2 = BasicBlock {
            id: b2,
            start_addr: 20,
            end_addr: 30,
            instructions: vec![DecodedInsn {
                mnemonic: "addss".to_string(),
                operands: vec!["xmm0".to_string(), "xmm1".to_string()],
                addr: 20,
            }],
            terminator: Terminator::Jump(b3),
        };
        // B3 (merge): movss xmm2, xmm0
        let block3 = BasicBlock {
            id: b3,
            start_addr: 30,
            end_addr: 40,
            instructions: vec![DecodedInsn {
                mnemonic: "movss".to_string(),
                operands: vec!["xmm2".to_string(), "xmm0".to_string()],
                addr: 30,
            }],
            terminator: Terminator::Return,
        };
        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Above,
                taken: b1,
                fallthrough: b2,
            },
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: b0,
            successors,
            predecessors: std::collections::BTreeMap::new(),
        };

        let mut exec = SymbolicExecutor::new(2, 0);

        // Act
        execute_blocks_with_merging(&[b0, b1, b2, b3], &cfg, &mut exec);

        // Assert: xmm2 should have been set from the merged xmm0 via movss in B3
        let xmm2 = exec.get_value("xmm2");
        assert!(
            matches!(xmm2, SymValue::Select { .. }),
            "xmm2 should be a Select propagated from merged xmm0, got: {xmm2}"
        );
    }

    // ── Wave 12k31b: +13 additional tests ──

    #[test]
    fn diamond_debug_contains_all_fields() {
        let d = Diamond {
            cond_block: BlockId(0),
            taken_block: BlockId(1),
            fallthrough_block: BlockId(2),
            merge_block: BlockId(3),
            kind: BranchKind::Above,
        };
        let debug = format!("{d:?}");
        assert!(debug.contains("cond_block"));
        assert!(debug.contains("taken_block"));
        assert!(debug.contains("fallthrough_block"));
        assert!(debug.contains("merge_block"));
    }

    #[test]
    fn select_simplifies_swapped_lt_to_max() {
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Lt,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(b.clone()),
            false_val: Box::new(a.clone()),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Max(_, _)),
            "Select(a < b, b, a) should simplify to Max, got: {simplified}"
        );
    }

    #[test]
    fn select_simplifies_swapped_ge_to_min() {
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(b.clone()),
            false_val: Box::new(a.clone()),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Min(_, _)),
            "Select(a >= b, b, a) should simplify to Min, got: {simplified}"
        );
    }

    #[test]
    fn select_simplifies_swapped_le_to_max() {
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Le,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(b.clone()),
            false_val: Box::new(a.clone()),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Max(_, _)),
            "Select(a <= b, b, a) should simplify to Max, got: {simplified}"
        );
    }

    #[test]
    fn select_const_lt_false_branch() {
        let select = SymValue::Select {
            kind: SelectKind::Lt,
            cond_lhs: Box::new(SymValue::Const(5.0)),
            cond_rhs: Box::new(SymValue::Const(3.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Param(1)),
            "Select(5.0 < 3.0, a, b) should fold to b, got: {simplified}"
        );
    }

    #[test]
    fn select_const_ge_true_branch() {
        let select = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(SymValue::Const(5.0)),
            cond_rhs: Box::new(SymValue::Const(3.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Param(0)),
            "Select(5.0 >= 3.0, a, b) should fold to a, got: {simplified}"
        );
    }

    #[test]
    fn select_const_le_true_branch() {
        let select = SymValue::Select {
            kind: SelectKind::Le,
            cond_lhs: Box::new(SymValue::Const(3.0)),
            cond_rhs: Box::new(SymValue::Const(5.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Param(0)),
            "Select(3.0 <= 5.0, a, b) should fold to a, got: {simplified}"
        );
    }

    #[test]
    fn is_branch_mnemonic_rejects_common_ops() {
        assert!(!is_branch_mnemonic("vaddps"));
        assert!(!is_branch_mnemonic("vmulps"));
        assert!(!is_branch_mnemonic("fmadd"));
        assert!(!is_branch_mnemonic("ldp"));
        assert!(!is_branch_mnemonic("stp"));
    }

    #[test]
    fn find_diamonds_unconditional_jump_not_detected() {
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::Jump(b1),
        };
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![],
            terminator: Terminator::Return,
        };
        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1]);
        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamonds = find_diamonds(&cfg, &[b0, b1]);
        assert!(diamonds.is_empty(), "unconditional jump should not form diamond");
    }

    #[test]
    fn find_diamonds_multiple_successors_no_diamond() {
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);
        let b4 = BlockId(4);
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b4),
        };
        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b4]);
        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamonds = find_diamonds(&cfg, &[b0, b1, b2, b3, b4]);
        assert!(diamonds.is_empty(), "different merge points should not form diamond");
    }

    #[test]
    fn merge_diamond_identical_paths_no_select() {
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };
        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);
        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);
        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Above,
        };
        let exec = SymbolicExecutor::new(1, 0);
        let merged = merge_diamond(&diamond, &cfg, &exec);
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Param(0)),
            "identical paths should not produce Select, got: {xmm0}"
        );
    }

    #[test]
    fn branch_kind_to_select_sign_variants_map_to_ne() {
        assert_eq!(branch_kind_to_select(&BranchKind::Sign), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::NotSign), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::Parity), SelectKind::Ne);
        assert_eq!(branch_kind_to_select(&BranchKind::NotParity), SelectKind::Ne);
    }

    #[test]
    fn select_non_foldable_ne_preserves_structure() {
        let select = SymValue::Select {
            kind: SelectKind::Ne,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(2)),
            false_val: Box::new(SymValue::Param(3)),
        };
        let simplified = select.simplify();
        assert!(
            matches!(simplified, SymValue::Select { kind: SelectKind::Ne, .. }),
            "symbolic Ne Select should stay as Select, got: {simplified}"
        );
    }

    // ── Wave 12kfd: +10 additional tests ──

    #[test]
    fn merge_diamond_stack_spill_diverges_produces_select() {
        // Arrange: Diamond where both arms write different values to the same stack slot.
        // Taken arm writes 42.0, fallthrough arm writes 99.0 → Select at stack offset 0.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Taken arm: movss [rsp+0], xmm0 → stores Param(0) to stack offset 0
        let block1 = BasicBlock {
            id: b1,
            start_addr: 10,
            end_addr: 20,
            instructions: vec![
                DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 10 },
                DecodedInsn { mnemonic: "movss".to_string(), operands: vec!["[rsp+0]".to_string(), "xmm1".to_string()], addr: 14 },
            ],
            terminator: Terminator::Jump(b3),
        };
        // Fallthrough arm: movss [rsp+0], xmm0 → stores Param(0) (unchanged) to stack offset 0
        // We need the stack values to differ, so taken writes const 0.0 while fallthrough writes Param(0)
        let block2 = BasicBlock {
            id: b2,
            start_addr: 20,
            end_addr: 30,
            instructions: vec![
                DecodedInsn { mnemonic: "movss".to_string(), operands: vec!["[rsp+0]".to_string(), "xmm0".to_string()], addr: 20 },
            ],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0,
            start_addr: 0,
            end_addr: 10,
            instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3,
            start_addr: 30,
            end_addr: 40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Above,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: stack offset 0 should be a Select (xorps→0.0 vs original Param(0))
        let stack_val = merged.stack_state().get(&0);
        assert!(
            stack_val.is_some(),
            "stack offset 0 should have a value after merge"
        );
        assert!(
            matches!(stack_val.unwrap(), SymValue::Select { .. }),
            "stack offset 0 should be a Select when arms diverge, got: {:?}", stack_val
        );
    }

    #[test]
    fn merge_diamond_missing_block_in_cfg_still_merges() {
        // Arrange: Diamond where the taken block is missing from cfg.blocks.
        // merge_diamond should handle this gracefully — taken_exec won't execute any block.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Only insert b0, b2, b3 — b1 is missing from cfg.blocks
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Greater, taken: b1, fallthrough: b2 },
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 20,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);
        // Note: b1 not in blocks

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Greater,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act: should not panic
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 should differ between paths (taken=Param(0) since block missing,
        // fallthrough=xorps→0.0), so it should be a Select
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { .. }),
            "xmm0 should be a Select when taken block is missing, got: {xmm0}"
        );
    }

    #[test]
    fn find_diamonds_arm_not_in_block_set_rejected() {
        // Arrange: Diamond where taken arm is NOT in the provided block_ids set.
        // find_diamonds should skip this because the arm check fails.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        // Only provide b0, b2 — b1 is excluded from the block_ids list
        let block_ids = vec![b0, b2];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: taken arm b1 not in block_set → diamond rejected
        assert!(diamonds.is_empty(), "diamond with arm outside block_ids should be rejected");
    }

    #[test]
    fn find_diamonds_arm_with_multiple_successors_rejected() {
        // Arrange: Diamond where taken arm has 2 successors (not 1) → rejected.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);
        let b4 = BlockId(4);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Less, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3, b4]); // taken arm has 2 successors
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let block_ids = vec![b0, b1, b2, b3, b4];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: taken arm has multiple successors → diamond rejected
        assert!(diamonds.is_empty(), "diamond with multi-successor arm should be rejected");
    }

    #[test]
    fn execute_block_skips_branch_mnemonics() {
        // Arrange: A basic block containing a mix of regular and branch instructions.
        // Branch instructions should be skipped during execution.
        let mut exec = SymbolicExecutor::new(1, 0);
        let block = BasicBlock {
            id: BlockId(0),
            start_addr: 0,
            end_addr: 20,
            instructions: vec![
                DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 0 },
                DecodedInsn { mnemonic: "ja".to_string(), operands: vec!["label".to_string()], addr: 4 },
                DecodedInsn { mnemonic: "addss".to_string(), operands: vec!["xmm0".to_string(), "xmm1".to_string()], addr: 8 },
                DecodedInsn { mnemonic: "jmp".to_string(), operands: vec!["somewhere".to_string()], addr: 12 },
            ],
            terminator: Terminator::Return,
        };

        // Act
        execute_block(&mut exec, &block);

        // Assert: xorps (xmm1=0.0) and addss (xmm0=param(0)+0.0) should have executed,
        // but ja and jmp should have been skipped. addss produces Add(Param(0), Const(0.0))
        let xmm0 = exec.get_value("xmm0");
        let xmm0_s = format!("{xmm0}");
        assert!(
            xmm0_s.contains("param(0)"),
            "xmm0 should contain param(0) after xorps+addss, got: {xmm0}"
        );
        let xmm1 = exec.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Const(0.0)),
            "xmm1 should be 0.0 after xorps, got: {xmm1}"
        );
    }

    #[test]
    fn execute_blocks_with_merging_multiple_diamonds() {
        // Arrange: CFG with two independent diamonds:
        //   Diamond 1: B0(cond) → B1(taken), B2(fall) → B3(merge)
        //   Diamond 2: B4(cond) → B5(taken), B6(fall) → B7(merge)
        let b0 = BlockId(0); let b1 = BlockId(1); let b2 = BlockId(2); let b3 = BlockId(3);
        let b4 = BlockId(4); let b5 = BlockId(5); let b6 = BlockId(6); let b7 = BlockId(7);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        // Diamond 1 taken: xorps xmm0, xmm0
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm0".to_string(), "xmm0".to_string()], addr: 10 }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Greater, taken: b5, fallthrough: b6 },
        };
        // Diamond 2 taken: xorps xmm1, xmm1
        let block5 = BasicBlock {
            id: b5, start_addr: 50, end_addr: 60,
            instructions: vec![DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 50 }],
            terminator: Terminator::Jump(b7),
        };
        let block6 = BasicBlock {
            id: b6, start_addr: 60, end_addr: 70, instructions: vec![],
            terminator: Terminator::Jump(b7),
        };
        let block7 = BasicBlock {
            id: b7, start_addr: 70, end_addr: 80, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);
        blocks.insert(b5, block5);
        blocks.insert(b6, block6);
        blocks.insert(b7, block7);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);
        successors.insert(b3, vec![b5, b6]);
        successors.insert(b5, vec![b7]);
        successors.insert(b6, vec![b7]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let mut exec = SymbolicExecutor::new(2, 0);

        // Act
        execute_blocks_with_merging(&[b0, b1, b2, b3, b5, b6, b7], &cfg, &mut exec);

        // Assert: xmm0 should be a Select (diamond 1: xorps vs unchanged)
        let xmm0 = exec.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { .. }),
            "xmm0 should be a Select from diamond 1, got: {xmm0}"
        );
        // xmm1 should be a Select (diamond 2: xorps vs unchanged)
        let xmm1 = exec.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Select { .. }),
            "xmm1 should be a Select from diamond 2, got: {xmm1}"
        );
    }

    #[test]
    fn merge_diamond_stack_identical_values_no_select() {
        // Arrange: Both arms of the diamond write the same value to the same stack slot.
        // Merge should keep the value without creating a Select node.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Both arms: xorps xmm1,xmm1 then movss [rsp+0], xmm1 → both write 0.0 to stack
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![
                DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 10 },
                DecodedInsn { mnemonic: "movss".to_string(), operands: vec!["[rsp+0]".to_string(), "xmm1".to_string()], addr: 14 },
            ],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30,
            instructions: vec![
                DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 20 },
                DecodedInsn { mnemonic: "movss".to_string(), operands: vec!["[rsp+0]".to_string(), "xmm1".to_string()], addr: 24 },
            ],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Above,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: stack offset 0 should NOT be a Select (identical on both paths)
        let stack_val = merged.stack_state().get(&0);
        assert!(
            stack_val.is_some(),
            "stack offset 0 should have a value"
        );
        assert!(
            !matches!(stack_val.unwrap(), SymValue::Select { .. }),
            "identical stack values should not produce Select, got: {:?}", stack_val
        );
    }

    #[test]
    fn find_diamonds_arm_no_successors_entry_rejected() {
        // Arrange: A branch where one arm has no entry in the successors map at all.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::AboveEqual, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Return,
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        // b1 has NO successors entry → successors.get(&b1) returns None → len=0
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let block_ids = vec![b0, b1, b2, b3];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: taken arm b1 has 0 successors → not 1 → diamond rejected
        assert!(diamonds.is_empty(), "diamond with arm having no successors should be rejected");
    }

    #[test]
    fn execute_blocks_with_merging_non_diamond_blocks_execute_normally() {
        // Arrange: A CFG with a regular block followed by a diamond, then another regular block.
        // Non-diamond blocks should execute their instructions normally.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);
        let b4 = BlockId(4);

        // B0: regular block — movss xmm2, xmm0
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10,
            instructions: vec![DecodedInsn {
                mnemonic: "movss".to_string(),
                operands: vec!["xmm2".to_string(), "xmm0".to_string()],
                addr: 0,
            }],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        // B3 (merge): also a regular block — movss xmm3, xmm0
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40,
            instructions: vec![DecodedInsn {
                mnemonic: "movss".to_string(),
                operands: vec!["xmm3".to_string(), "xmm0".to_string()],
                addr: 30,
            }],
            terminator: Terminator::Jump(b4),
        };
        let block4 = BasicBlock {
            id: b4, start_addr: 40, end_addr: 50, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);
        blocks.insert(b4, block4);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);
        successors.insert(b3, vec![b4]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let mut exec = SymbolicExecutor::new(1, 0);

        // Act
        execute_blocks_with_merging(&[b0, b1, b2, b3, b4], &cfg, &mut exec);

        // Assert: xmm2 was set before the diamond (regular block execution)
        let xmm2 = exec.get_value("xmm2");
        assert!(
            matches!(xmm2, SymValue::Param(0)),
            "xmm2 should be Param(0) from pre-diamond block, got: {xmm2}"
        );
        // xmm3 was set from merged xmm0 in the merge block (post-diamond)
        let xmm3 = exec.get_value("xmm3");
        assert!(
            matches!(xmm3, SymValue::Select { .. }),
            "xmm3 should be a Select propagated from merged xmm0, got: {xmm3}"
        );
    }

    #[test]
    fn merge_diamond_register_in_only_one_path_takes_existing() {
        // Arrange: Diamond where only one arm modifies xmm2 (via movss from xmm0),
        // the other arm leaves it unchanged. The merged result should use the value
        // from the arm that wrote it (not a Select, since the other arm's pre_state
        // is the same Param(0) value).
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Taken arm: movss xmm2, xmm1 → xmm2 = Param(1)
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "movss".to_string(),
                operands: vec!["xmm2".to_string(), "xmm1".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        // Fallthrough arm: no instructions (xmm2 unchanged)
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Greater, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Greater,
        };

        // 2 float args: xmm0=Param(0), xmm1=Param(1)
        let exec = SymbolicExecutor::new(2, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm2 is set to Param(1) on taken path, and neither fallthrough nor
        // pre_state have xmm2. The (Some, None) match arm copies the taken value directly.
        let xmm2 = merged.get_value("xmm2");
        assert!(
            matches!(xmm2, SymValue::Param(1)),
            "xmm2 should be Param(1) (taken value, single-path write), got: {xmm2}"
        );
    }

    // ── Wave 12kia: +10 additional tests ──

    #[test]
    fn select_eq_swapped_branches_stays_as_select() {
        // Arrange: Select(a == b, b, a) — swapped with Eq — cannot simplify to Max/Min.
        let a = SymValue::Param(0);
        let b = SymValue::Param(1);
        let select = SymValue::Select {
            kind: SelectKind::Eq,
            cond_lhs: Box::new(a.clone()),
            cond_rhs: Box::new(b.clone()),
            true_val: Box::new(b.clone()),
            false_val: Box::new(a.clone()),
        };

        // Act
        let simplified = select.simplify();

        // Assert: should remain a Select (Eq with swapped branches does not collapse)
        assert!(
            matches!(simplified, SymValue::Select { kind: SelectKind::Eq, .. }),
            "Select(Eq, b, a) with symbolic condition should stay as Select, got: {simplified}"
        );
    }

    #[test]
    fn find_diamonds_multiple_diamonds_in_same_block_set() {
        // Arrange: Two independent diamonds in one block set:
        //   Diamond 1: B0(cond) → B1(taken), B2(fall) → B3(merge)
        //   Diamond 2: B4(cond) → B5(taken), B6(fall) → B7(merge)
        let b0 = BlockId(0); let b1 = BlockId(1); let b2 = BlockId(2); let b3 = BlockId(3);
        let b4 = BlockId(4); let b5 = BlockId(5); let b6 = BlockId(6); let b7 = BlockId(7);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };
        let block4 = BasicBlock {
            id: b4, start_addr: 40, end_addr: 50, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Less, taken: b5, fallthrough: b6 },
        };
        let block5 = BasicBlock {
            id: b5, start_addr: 50, end_addr: 60, instructions: vec![],
            terminator: Terminator::Jump(b7),
        };
        let block6 = BasicBlock {
            id: b6, start_addr: 60, end_addr: 70, instructions: vec![],
            terminator: Terminator::Jump(b7),
        };
        let block7 = BasicBlock {
            id: b7, start_addr: 70, end_addr: 80, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);
        blocks.insert(b4, block4);
        blocks.insert(b5, block5);
        blocks.insert(b6, block6);
        blocks.insert(b7, block7);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);
        successors.insert(b4, vec![b5, b6]);
        successors.insert(b5, vec![b7]);
        successors.insert(b6, vec![b7]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let block_ids = vec![b0, b1, b2, b3, b4, b5, b6, b7];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: should find exactly 2 diamonds
        assert_eq!(diamonds.len(), 2, "should find exactly two diamonds");
        assert_eq!(diamonds[0].cond_block, b0);
        assert_eq!(diamonds[0].kind, BranchKind::Above);
        assert_eq!(diamonds[1].cond_block, b4);
        assert_eq!(diamonds[1].kind, BranchKind::Less);
    }

    #[test]
    fn merge_diamond_with_less_kind_produces_correct_select() {
        // Arrange: Diamond with BranchKind::Less — should produce Select with SelectKind::Lt
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Less, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Less,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 should be a Select with kind Lt
        let xmm0 = merged.get_value("xmm0");
        if let SymValue::Select { kind, .. } = &xmm0 {
            assert_eq!(*kind, SelectKind::Lt, "Select kind should be Lt for BranchKind::Less");
        } else {
            panic!("xmm0 should be a Select, got: {xmm0}");
        }
    }

    #[test]
    fn execute_blocks_with_merging_linear_cfg_no_diamonds() {
        // Arrange: A purely linear CFG (no CondBranch, only Jump/Fallthrough/Return).
        // execute_blocks_with_merging should simply execute all blocks linearly.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm1".to_string(), "xmm1".to_string()],
                addr: 0,
            }],
            terminator: Terminator::Jump(b1),
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "addss".to_string(),
                operands: vec!["xmm0".to_string(), "xmm1".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b2),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1]);
        successors.insert(b1, vec![b2]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let mut exec = SymbolicExecutor::new(1, 0);

        // Act
        execute_blocks_with_merging(&[b0, b1, b2], &cfg, &mut exec);

        // Assert: all instructions executed linearly — xmm1=0.0, xmm0=param(0)+0.0
        let xmm1 = exec.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Const(0.0)),
            "xmm1 should be 0.0 after xorps, got: {xmm1}"
        );
        let xmm0 = exec.get_value("xmm0");
        let xmm0_s = format!("{xmm0}");
        assert!(
            xmm0_s.contains("param(0)"),
            "xmm0 should contain param(0) after linear execution, got: {xmm0}"
        );
        // No Select should have been created
        assert!(
            !matches!(xmm0, SymValue::Select { .. }),
            "linear CFG should not produce any Select, got: {xmm0}"
        );
    }

    #[test]
    fn find_diamonds_block_ids_with_duplicates_yields_duplicate_detections() {
        // Arrange: find_diamonds iterates over the block_ids list (not a deduped set),
        // so duplicate entries for the cond_block produce duplicate Diamond results.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Greater, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        // block_ids with b0 appearing twice
        let block_ids = vec![b0, b1, b2, b3, b0];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: b0 is iterated twice -> two diamond detections for the same structure
        assert_eq!(diamonds.len(), 2, "duplicate cond_block entry yields duplicate diamonds");
        // Both should reference the same structural diamond
        assert_eq!(diamonds[0].cond_block, diamonds[1].cond_block);
        assert_eq!(diamonds[0].merge_block, diamonds[1].merge_block);
        assert_eq!(diamonds[0].kind, BranchKind::Greater);
    }

    #[test]
    fn select_ne_const_different_values_folds_to_true_branch() {
        // Arrange: Select(3.0 != 5.0, param(0), param(1)) -> condition is true -> param(0)
        let select = SymValue::Select {
            kind: SelectKind::Ne,
            cond_lhs: Box::new(SymValue::Const(3.0)),
            cond_rhs: Box::new(SymValue::Const(5.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(0)),
            "Select(3.0 != 5.0, a, b) should fold to a (true branch), got: {simplified}"
        );
    }

    #[test]
    fn merge_diamond_with_equal_kind_produces_eq_select() {
        // Arrange: Diamond with BranchKind::Equal — should produce Select with SelectKind::Eq
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Equal, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Equal,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 should be a Select with kind Eq
        let xmm0 = merged.get_value("xmm0");
        if let SymValue::Select { kind, .. } = &xmm0 {
            assert_eq!(*kind, SelectKind::Eq, "Select kind should be Eq for BranchKind::Equal");
        } else {
            panic!("xmm0 should be a Select, got: {xmm0}");
        }
    }

    #[test]
    fn merge_diamond_pre_state_xmm1_preserved_when_arms_dont_touch_it() {
        // Arrange: Executor has 2 float args (xmm0=Param(0), xmm1=Param(1)).
        // Diamond arm only modifies xmm0 (via xorps). xmm1 should remain Param(1)
        // on both paths, so no Select is created for xmm1.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Taken arm: only touches xmm0
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        // Fallthrough arm: no instructions at all
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Above,
        };

        let exec = SymbolicExecutor::new(2, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm1 should still be Param(1) — no Select created
        let xmm1 = merged.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Param(1)),
            "xmm1 should remain Param(1) since neither arm modified it, got: {xmm1}"
        );
    }

    #[test]
    fn find_diamonds_self_loop_taken_equals_merge_detected() {
        // Arrange: A CondBranch where both arms merge back to the cond block itself (B0).
        // B0 branches to B1 (taken) and B2 (fallthrough). Both B1 and B2 jump back to B0.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b0),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b0),
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b0]);
        successors.insert(b2, vec![b0]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };

        let block_ids = vec![b0, b1, b2];

        // Act
        let diamonds = find_diamonds(&cfg, &block_ids);

        // Assert: this IS a valid diamond shape structurally — both arms merge to the same block.
        assert_eq!(diamonds.len(), 1, "self-loop diamond should be detected structurally");
        assert_eq!(diamonds[0].cond_block, b0);
        assert_eq!(diamonds[0].merge_block, b0, "merge block should be b0 (self-loop)");
    }

    #[test]
    fn select_const_gt_false_branch_folds_correctly() {
        // Arrange: Select(1.0 > 5.0, param(0), param(1)) -> condition false -> param(1)
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Const(1.0)),
            cond_rhs: Box::new(SymValue::Const(5.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(1)),
            "Select(1.0 > 5.0, a, b) should fold to b (false branch), got: {simplified}"
        );
    }

    // ── Wave 12kib: +10 additional tests ──

    #[test]
    fn select_const_eq_different_values_folds_to_false_branch() {
        // Arrange: Select(3.0 == 7.0, param(0), param(1)) -> condition false -> param(1)
        let select = SymValue::Select {
            kind: SelectKind::Eq,
            cond_lhs: Box::new(SymValue::Const(3.0)),
            cond_rhs: Box::new(SymValue::Const(7.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(1)),
            "Select(3.0 == 7.0, a, b) should fold to b, got: {simplified}"
        );
    }

    #[test]
    fn select_const_ge_false_branch_folds() {
        // Arrange: Select(2.0 >= 5.0, param(0), param(1)) -> condition false -> param(1)
        let select = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(SymValue::Const(2.0)),
            cond_rhs: Box::new(SymValue::Const(5.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(1)),
            "Select(2.0 >= 5.0, a, b) should fold to b, got: {simplified}"
        );
    }

    #[test]
    fn select_const_le_false_branch_folds() {
        // Arrange: Select(8.0 <= 4.0, param(0), param(1)) -> condition false -> param(1)
        let select = SymValue::Select {
            kind: SelectKind::Le,
            cond_lhs: Box::new(SymValue::Const(8.0)),
            cond_rhs: Box::new(SymValue::Const(4.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert
        assert!(
            matches!(simplified, SymValue::Param(1)),
            "Select(8.0 <= 4.0, a, b) should fold to b, got: {simplified}"
        );
    }

    #[test]
    fn find_diamonds_zero_arg_executor_flags_none() {
        // Arrange: SymbolicExecutor with 0 float args — no comparison executed.
        let exec = SymbolicExecutor::new(0, 0);

        // Act & Assert
        assert!(exec.get_flags().is_none(), "no flags should be set on fresh executor");
    }

    #[test]
    fn diamond_kind_field_preserves_branch_kind() {
        // Arrange: Construct Diamond with BranchKind::BelowEqual
        let d = Diamond {
            cond_block: BlockId(5),
            taken_block: BlockId(6),
            fallthrough_block: BlockId(7),
            merge_block: BlockId(8),
            kind: BranchKind::BelowEqual,
        };

        // Assert: kind is preserved
        assert_eq!(d.kind, BranchKind::BelowEqual);
        assert_eq!(d.cond_block, BlockId(5));
        assert_eq!(d.merge_block, BlockId(8));
    }

    #[test]
    fn merge_diamond_with_not_equal_kind_produces_ne_select() {
        // Arrange: Diamond with BranchKind::NotEqual — should produce Select with SelectKind::Ne
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::NotEqual, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::NotEqual,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert
        let xmm0 = merged.get_value("xmm0");
        if let SymValue::Select { kind, .. } = &xmm0 {
            assert_eq!(*kind, SelectKind::Ne, "Select kind should be Ne for BranchKind::NotEqual");
        } else {
            panic!("xmm0 should be a Select, got: {xmm0}");
        }
    }

    #[test]
    fn is_branch_mnemonic_rejects_floating_point_ops() {
        // Arrange & Act & Assert: floating-point arithmetic mnemonics are not branches
        assert!(!is_branch_mnemonic("addss"));
        assert!(!is_branch_mnemonic("subss"));
        assert!(!is_branch_mnemonic("mulss"));
        assert!(!is_branch_mnemonic("divss"));
        assert!(!is_branch_mnemonic("sqrtss"));
    }

    #[test]
    fn execute_blocks_with_merging_empty_block_list_noop() {
        // Arrange
        let cfg = ControlFlowGraph {
            blocks: std::collections::BTreeMap::new(),
            entry: BlockId(0),
            successors: std::collections::BTreeMap::new(),
            predecessors: std::collections::BTreeMap::new(),
        };
        let mut exec = SymbolicExecutor::new(1, 0);
        let xmm0_before = exec.get_value("xmm0");

        // Act
        execute_blocks_with_merging(&[], &cfg, &mut exec);

        // Assert: executor state unchanged
        let xmm0_after = exec.get_value("xmm0");
        assert_eq!(format!("{xmm0_before}"), format!("{xmm0_after}"),
            "empty block list should not modify executor state");
    }

    #[test]
    fn merge_diamond_both_arms_modify_same_reg_produces_select() {
        // Arrange: Both arms write different expressions to xmm0.
        // Taken: xorps xmm0, xmm0 → xmm0 = 0.0
        // Fallthrough: addss xmm0, xmm1 → xmm0 = Param(0) + Param(1)
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30,
            instructions: vec![DecodedInsn {
                mnemonic: "addss".to_string(),
                operands: vec!["xmm0".to_string(), "xmm1".to_string()],
                addr: 20,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::GreaterEqual, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::GreaterEqual,
        };
        let exec = SymbolicExecutor::new(2, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 is a Select because both arms produce different values
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { kind: SelectKind::Ge, .. }),
            "xmm0 should be Select(Ge, ..) for BranchKind::GreaterEqual, got: {xmm0}"
        );
        // true_val = Const(0.0) from xorps, false_val = Add(Param(0), Param(1)) from addss
        if let SymValue::Select { true_val, false_val, .. } = &xmm0 {
            assert!(matches!(true_val.as_ref(), SymValue::Const(0.0)),
                "true_val should be Const(0.0) from xorps, got: {true_val}");
            let fv_s = format!("{false_val}");
            assert!(fv_s.contains("param(0)"),
                "false_val should contain param(0) from addss, got: {false_val}");
        }
    }

    #[test]
    fn find_diamonds_single_return_block_no_diamond() {
        // Arrange: A single block with Return terminator — no CondBranch, no diamond possible.
        let b0 = BlockId(0);
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::Return,
        };
        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        let cfg = ControlFlowGraph {
            blocks, entry: b0,
            successors: std::collections::BTreeMap::new(),
            predecessors: std::collections::BTreeMap::new(),
        };

        // Act
        let diamonds = find_diamonds(&cfg, &[b0]);

        // Assert
        assert!(diamonds.is_empty(), "single Return block should not form a diamond");
    }

    // ── Wave 12x59: +10 additional tests ──

    #[test]
    fn merge_diamond_empty_arms_preserves_pre_state() {
        // Arrange: Diamond where both arms have zero instructions.
        // Merge should preserve the pre-merge executor state unchanged.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Above,
        };
        let exec = SymbolicExecutor::new(2, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: both arms empty → no divergence → pre-state preserved
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Param(0)),
            "xmm0 should remain Param(0) with empty arms, got: {xmm0}"
        );
        let xmm1 = merged.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Param(1)),
            "xmm1 should remain Param(1) with empty arms, got: {xmm1}"
        );
    }

    #[test]
    fn find_diamonds_single_cond_block_only_no_diamond() {
        // Arrange: Only the cond block exists in block_ids — arms are not present.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Less, taken: b1, fallthrough: b2 },
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);

        let cfg = ControlFlowGraph {
            blocks, entry: b0,
            successors: std::collections::BTreeMap::new(),
            predecessors: std::collections::BTreeMap::new(),
        };

        // Act: only b0 in block_ids — arms b1/b2 not in block_set
        let diamonds = find_diamonds(&cfg, &[b0]);

        // Assert
        assert!(diamonds.is_empty(), "cond block alone without arms should not form diamond");
    }

    #[test]
    fn merge_diamond_three_way_xmm_diverge_all_different() {
        // Arrange: Diamond where taken, fallthrough, and pre-state all have different xmm0.
        // Taken: xorps xmm0, xmm0 → 0.0
        // Fallthrough: addss xmm0, xmm1 → Param(0)+Param(1)
        // Pre-state: Param(0)
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30,
            instructions: vec![DecodedInsn {
                mnemonic: "addss".to_string(),
                operands: vec!["xmm0".to_string(), "xmm1".to_string()],
                addr: 20,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Below, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Below,
        };
        let exec = SymbolicExecutor::new(2, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 diverges → Select with kind Lt (Below maps to Lt)
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { kind: SelectKind::Lt, .. }),
            "xmm0 should be Select(Lt, ..) for BranchKind::Below, got: {xmm0}"
        );
    }

    #[test]
    fn merge_diamond_loop_header_merge_point() {
        // Arrange: Diamond where the merge block is also a loop header
        // (both arms jump back to the cond block, forming a loop).
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);

        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Greater, taken: b1, fallthrough: b2 },
        };
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b0),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b0),
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b0]);
        successors.insert(b2, vec![b0]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b0,
            kind: BranchKind::Greater,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act: merge should work even when merge_block == cond_block (loop header)
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: xmm0 diverges (xorps→0.0 vs unchanged Param(0)) → Select
        let xmm0 = merged.get_value("xmm0");
        assert!(
            matches!(xmm0, SymValue::Select { .. }),
            "xmm0 should be a Select for loop-header merge, got: {xmm0}"
        );
    }

    #[test]
    fn phi_value_merge_two_different_params() {
        // Arrange: Construct a Select where true_val=Param(0), false_val=Param(1)
        // with symbolic condition — simulates PHI node merging two distinct values.
        let select = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(2)),
            cond_rhs: Box::new(SymValue::Const(0.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = select.simplify();

        // Assert: cannot fold — stays as Select since condition is symbolic
        assert!(
            matches!(simplified, SymValue::Select { kind: SelectKind::Gt, .. }),
            "PHI merge of two different params should stay as Select, got: {simplified}"
        );
    }

    #[test]
    fn phi_value_merge_identical_values_collapses() {
        // Arrange: PHI-like Select where both branches produce the same value.
        let select = SymValue::Select {
            kind: SelectKind::Lt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(2)),
            false_val: Box::new(SymValue::Param(2)),
        };

        // Act
        let simplified = select.simplify();

        // Assert: same true/false → collapse to the value
        assert!(
            matches!(simplified, SymValue::Param(2)),
            "PHI with identical values should collapse, got: {simplified}"
        );
    }

    #[test]
    fn merge_diamond_below_equal_kind_produces_le_select() {
        // Arrange: Diamond with BranchKind::BelowEqual — should produce Select with SelectKind::Le
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![DecodedInsn {
                mnemonic: "xorps".to_string(),
                operands: vec!["xmm0".to_string(), "xmm0".to_string()],
                addr: 10,
            }],
            terminator: Terminator::Jump(b3),
        };
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::BelowEqual, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::BelowEqual,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: Select kind should be Le (BelowEqual maps to Le)
        let xmm0 = merged.get_value("xmm0");
        if let SymValue::Select { kind, .. } = &xmm0 {
            assert_eq!(*kind, SelectKind::Le, "BelowEqual should map to Le, got: {kind}");
        } else {
            panic!("xmm0 should be a Select, got: {xmm0}");
        }
    }

    #[test]
    fn execute_blocks_with_merging_single_block_no_cond_branch() {
        // Arrange: A single block with only arithmetic (no CondBranch).
        let b0 = BlockId(0);
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10,
            instructions: vec![
                DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 0 },
                DecodedInsn { mnemonic: "addss".to_string(), operands: vec!["xmm0".to_string(), "xmm1".to_string()], addr: 4 },
            ],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        let cfg = ControlFlowGraph {
            blocks, entry: b0,
            successors: std::collections::BTreeMap::new(),
            predecessors: std::collections::BTreeMap::new(),
        };

        let mut exec = SymbolicExecutor::new(1, 0);

        // Act
        execute_blocks_with_merging(&[b0], &cfg, &mut exec);

        // Assert: instructions executed normally, no diamond processing
        let xmm1 = exec.get_value("xmm1");
        assert!(
            matches!(xmm1, SymValue::Const(0.0)),
            "xmm1 should be 0.0 after xorps, got: {xmm1}"
        );
    }

    #[test]
    fn merge_diamond_stack_only_one_arm_writes() {
        // Arrange: Diamond where only the taken arm writes to stack, fallthrough does not.
        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b2 = BlockId(2);
        let b3 = BlockId(3);

        // Taken arm: xorps xmm1, xmm1 then movss [rsp+0], xmm1 → stack[0] = 0.0
        let block1 = BasicBlock {
            id: b1, start_addr: 10, end_addr: 20,
            instructions: vec![
                DecodedInsn { mnemonic: "xorps".to_string(), operands: vec!["xmm1".to_string(), "xmm1".to_string()], addr: 10 },
                DecodedInsn { mnemonic: "movss".to_string(), operands: vec!["[rsp+0]".to_string(), "xmm1".to_string()], addr: 14 },
            ],
            terminator: Terminator::Jump(b3),
        };
        // Fallthrough arm: no stack write
        let block2 = BasicBlock {
            id: b2, start_addr: 20, end_addr: 30, instructions: vec![],
            terminator: Terminator::Jump(b3),
        };
        let block0 = BasicBlock {
            id: b0, start_addr: 0, end_addr: 10, instructions: vec![],
            terminator: Terminator::CondBranch { kind: BranchKind::Above, taken: b1, fallthrough: b2 },
        };
        let block3 = BasicBlock {
            id: b3, start_addr: 30, end_addr: 40, instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = std::collections::BTreeMap::new();
        blocks.insert(b0, block0);
        blocks.insert(b1, block1);
        blocks.insert(b2, block2);
        blocks.insert(b3, block3);

        let mut successors = std::collections::BTreeMap::new();
        successors.insert(b0, vec![b1, b2]);
        successors.insert(b1, vec![b3]);
        successors.insert(b2, vec![b3]);

        let cfg = ControlFlowGraph {
            blocks, entry: b0, successors, predecessors: std::collections::BTreeMap::new(),
        };
        let diamond = Diamond {
            cond_block: b0, taken_block: b1, fallthrough_block: b2, merge_block: b3,
            kind: BranchKind::Above,
        };
        let exec = SymbolicExecutor::new(1, 0);

        // Act
        let merged = merge_diamond(&diamond, &cfg, &exec);

        // Assert: stack offset 0 should have a value (written by taken arm only)
        let stack_val = merged.stack_state().get(&0);
        assert!(
            stack_val.is_some(),
            "stack offset 0 should have a value from taken arm write"
        );
    }

    #[test]
    fn select_nested_simplify_inner_first() {
        // Arrange: Nested Select where inner Select can simplify to Max,
        // and outer Select uses the result.
        let inner = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Const(0.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Const(0.0)),
        };
        // Outer: Select(inner > Param(1), inner, Param(1))
        // After inner simplifies to Max(Param(0), 0.0), outer remains Select
        let outer = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(inner.clone()),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(inner),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let simplified = outer.simplify();

        // Assert: inner should have simplified to Max, outer remains Select
        let s = format!("{simplified}");
        assert!(
            s.contains("max"),
            "simplified result should contain max from inner Select, got: {s}"
        );
    }
}
