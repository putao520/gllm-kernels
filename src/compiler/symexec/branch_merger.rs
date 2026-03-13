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
        .map(|(l, r)| (l, r))
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
        "je" | "jne" | "jb" | "jbe" | "ja" | "jae"
            | "jl" | "jle" | "jg" | "jge"
            | "jmp" | "js" | "jns" | "jp" | "jnp"
            | "ret"
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
}
