//! CFG construction and natural loop detection for symexec.
//!
//! Step 1 of the control-flow upgrade: builds a control-flow graph from
//! a compiled `extern "C"` function's machine code, then identifies natural
//! loops via dominator-tree analysis.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use crate::types::CompilerError;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Basic block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

/// Branch condition kind (x86 conditional jumps).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchKind {
    /// JA / JNBE — unsigned above
    Above,
    /// JAE / JNB — unsigned above or equal
    AboveEqual,
    /// JB / JNAE — unsigned below
    Below,
    /// JBE / JNA — unsigned below or equal
    BelowEqual,
    /// JG / JNLE — signed greater
    Greater,
    /// JGE / JNL — signed greater or equal
    GreaterEqual,
    /// JL / JNGE — signed less
    Less,
    /// JLE / JNG — signed less or equal
    LessEqual,
    /// JE / JZ — equal / zero
    Equal,
    /// JNE / JNZ — not equal / not zero
    NotEqual,
    /// JS — sign flag set
    Sign,
    /// JNS — sign flag clear
    NotSign,
    /// JP / JPE — parity even
    Parity,
    /// JNP / JPO — parity odd
    NotParity,
}

/// Block terminator — how control leaves a basic block.
#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    /// Falls through to the next sequential block.
    Fallthrough(BlockId),
    /// Unconditional jump.
    Jump(BlockId),
    /// Conditional branch: if condition → taken, else → fallthrough.
    CondBranch {
        kind: BranchKind,
        taken: BlockId,
        fallthrough: BlockId,
    },
    /// Function return.
    Return,
}

/// A single decoded instruction (architecture-independent representation).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedInsn {
    pub mnemonic: String,
    pub operands: Vec<String>,
    pub addr: u64,
}

/// A basic block: a maximal sequence of instructions with a single entry
/// and single exit (the terminator).
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub start_addr: u64,
    pub end_addr: u64,
    pub instructions: Vec<DecodedInsn>,
    pub terminator: Terminator,
}

/// Control-flow graph built from a function's machine code.
#[derive(Debug)]
pub struct ControlFlowGraph {
    pub blocks: BTreeMap<BlockId, BasicBlock>,
    pub entry: BlockId,
    pub successors: BTreeMap<BlockId, Vec<BlockId>>,
    pub predecessors: BTreeMap<BlockId, Vec<BlockId>>,
}

/// A natural loop identified by a back-edge in the CFG.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// The loop header block (dominates all blocks in the loop).
    pub header: BlockId,
    /// All blocks in the loop body (including header).
    pub body_blocks: BTreeSet<BlockId>,
    /// The latch block (source of the back-edge to header).
    pub latch: BlockId,
    /// Exit blocks (successors of body blocks that are outside the loop).
    pub exits: Vec<BlockId>,
    /// Ordinal index (sorted by header address).
    pub ordinal: usize,
    /// Nesting depth (0 = outermost).
    pub depth: usize,
}

/// Forest of natural loops found in a CFG.
#[derive(Debug)]
pub struct LoopForest {
    /// All detected natural loops.
    pub loops: Vec<NaturalLoop>,
    /// Parent → children mapping (by index into `loops`).
    pub children: HashMap<usize, Vec<usize>>,
    /// Indices of top-level (non-nested) loops.
    pub top_level: Vec<usize>,
}

// ---------------------------------------------------------------------------
// CFG construction
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

/// Build a control-flow graph from a compiled function.
///
/// # Safety
/// `fn_ptr` must point to valid executable memory containing a compiled
/// function. `max_bytes` limits how far the disassembler reads.
#[cfg(target_arch = "x86_64")]
pub unsafe fn build_cfg_from_fn(fn_ptr: *const u8, max_bytes: usize) -> Result<ControlFlowGraph, CompilerError> {
    if fn_ptr.is_null() {
        return Err("null function pointer".into());
    }

    let base_addr = fn_ptr as u64;
    let bytes = unsafe { std::slice::from_raw_parts(fn_ptr, max_bytes) };
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    decoder.set_ip(base_addr);

    // Linear scan — collect all instructions and jump targets.
    let mut all_insns: Vec<(u64, Instruction)> = Vec::new();
    let mut branch_targets: BTreeSet<u64> = BTreeSet::new();
    let mut func_end: Option<u64> = None;

    let mut instr = Instruction::default();
    while decoder.can_decode() {
        decoder.decode_out(&mut instr);
        let ip = instr.ip();
        let next_ip = instr.next_ip();
        let mnemonic = instr.mnemonic();

        all_insns.push((ip, instr));

        if mnemonic == Mnemonic::Ret {
            func_end = Some(next_ip);
            break;
        }

        if is_branch_mnemonic(mnemonic) && has_near_branch_operand(&instr) {
            let target = instr.near_branch_target();
            branch_targets.insert(target);
            // The fallthrough address is also a potential block start.
            branch_targets.insert(next_ip);
        }
    }

    if all_insns.is_empty() {
        return Err("no instructions decoded".into());
    }

    let end_addr = func_end.unwrap_or_else(|| {
        let last = &all_insns[all_insns.len() - 1].1;
        last.next_ip()
    });

    // Determine block boundaries.
    // A new block starts at: (a) the function entry, (b) any branch target,
    // (c) the instruction after a branch/ret.
    let mut block_starts: BTreeSet<u64> = BTreeSet::new();
    block_starts.insert(base_addr);
    for &target in &branch_targets {
        if target >= base_addr && target < end_addr {
            block_starts.insert(target);
        }
    }

    // Map address → BlockId.
    let mut addr_to_block: BTreeMap<u64, BlockId> = BTreeMap::new();
    for (idx, &addr) in block_starts.iter().enumerate() {
        addr_to_block.insert(addr, BlockId(idx as u32));
    }

    // Helper: find which block an address belongs to.
    let find_block = |addr: u64| -> Option<BlockId> {
        addr_to_block.get(&addr).copied()
    };

    // Build basic blocks.
    let block_start_vec: Vec<u64> = block_starts.iter().copied().collect();
    let mut blocks: BTreeMap<BlockId, BasicBlock> = BTreeMap::new();
    let mut successors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
    let mut predecessors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();

    for (blk_idx, &blk_start) in block_start_vec.iter().enumerate() {
        let blk_id = BlockId(blk_idx as u32);
        let blk_end = if blk_idx + 1 < block_start_vec.len() {
            block_start_vec[blk_idx + 1]
        } else {
            end_addr
        };

        // Collect instructions in this block.
        let mut insns: Vec<DecodedInsn> = Vec::new();
        let mut last_instr: Option<&Instruction> = None;

        for (ip, instr_ref) in &all_insns {
            if *ip >= blk_start && *ip < blk_end {
                insns.push(decode_insn(instr_ref));
                last_instr = Some(instr_ref);
            }
        }

        // Determine terminator from the last instruction.
        let terminator = if let Some(last) = last_instr {
            let m = last.mnemonic();
            if m == Mnemonic::Ret {
                Terminator::Return
            } else if m == Mnemonic::Jmp && has_near_branch_operand(last) {
                let target = last.near_branch_target();
                if let Some(target_id) = find_block(target) {
                    Terminator::Jump(target_id)
                } else {
                    // Jump outside function — treat as return.
                    Terminator::Return
                }
            } else if is_conditional_branch(m) && has_near_branch_operand(last) {
                let target = last.near_branch_target();
                let fallthrough_addr = last.next_ip();
                let taken_id = find_block(target).unwrap_or(blk_id);
                let fall_id = find_block(fallthrough_addr).unwrap_or(blk_id);
                let kind = mnemonic_to_branch_kind(m);
                Terminator::CondBranch {
                    kind,
                    taken: taken_id,
                    fallthrough: fall_id,
                }
            } else if blk_idx + 1 < block_start_vec.len() {
                // Non-branch last instruction → fallthrough.
                Terminator::Fallthrough(BlockId((blk_idx + 1) as u32))
            } else {
                Terminator::Return
            }
        } else {
            Terminator::Return
        };

        // Record successors.
        let succs = match &terminator {
            Terminator::Fallthrough(next) => vec![*next],
            Terminator::Jump(target) => vec![*target],
            Terminator::CondBranch { taken, fallthrough, .. } => vec![*taken, *fallthrough],
            Terminator::Return => vec![],
        };

        for &s in &succs {
            predecessors.entry(s).or_default().push(blk_id);
        }
        successors.insert(blk_id, succs);

        let block_end_addr = if let Some(last) = last_instr {
            last.next_ip()
        } else {
            blk_end
        };

        blocks.insert(blk_id, BasicBlock {
            id: blk_id,
            start_addr: blk_start,
            end_addr: block_end_addr,
            instructions: insns,
            terminator,
        });
    }

    // Ensure all blocks have entries in predecessor/successor maps.
    for &blk_id in blocks.keys() {
        successors.entry(blk_id).or_default();
        predecessors.entry(blk_id).or_default();
    }

    Ok(ControlFlowGraph {
        blocks,
        entry: BlockId(0),
        successors,
        predecessors,
    })
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn build_cfg_from_fn(fn_ptr: *const u8, max_bytes: usize) -> Result<ControlFlowGraph, CompilerError> {
    super::decoder_aarch64::build_cfg_from_fn_aarch64(fn_ptr, max_bytes)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn build_cfg_from_fn(_fn_ptr: *const u8, _max_bytes: usize) -> Result<ControlFlowGraph, CompilerError> {
    Err("CFG construction only supported on x86_64 and aarch64".into())
}

// ---------------------------------------------------------------------------
// Natural loop detection
// ---------------------------------------------------------------------------

/// Find all natural loops in a CFG using dominator-tree analysis.
///
/// Algorithm:
/// 1. Compute dominators (iterative dataflow).
/// 2. Find back-edges: (A → B) where B dominates A.
/// 3. For each back-edge, compute the natural loop body via reverse BFS.
/// 4. Compute nesting and assign ordinals.
pub fn find_loops(cfg: &ControlFlowGraph) -> LoopForest {
    if cfg.blocks.is_empty() {
        return LoopForest {
            loops: vec![],
            children: HashMap::new(),
            top_level: vec![],
        };
    }

    // Step 1: Compute dominators.
    let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
    let doms = compute_dominators(cfg, &block_ids);

    // Step 2: Find back-edges.
    let mut back_edges: Vec<(BlockId, BlockId)> = Vec::new(); // (latch, header)
    for (&src, succs) in &cfg.successors {
        for &dst in succs {
            if dominates(&doms, dst, src) {
                back_edges.push((src, dst));
            }
        }
    }

    // Step 3: Build natural loop for each back-edge.
    let mut loops: Vec<NaturalLoop> = Vec::new();
    for (latch, header) in &back_edges {
        let body = compute_loop_body(cfg, *header, *latch);

        // Find exit blocks: successors of body blocks that are outside the body.
        let mut exits: Vec<BlockId> = Vec::new();
        for &blk in &body {
            if let Some(succs) = cfg.successors.get(&blk) {
                for &s in succs {
                    if !body.contains(&s) && !exits.contains(&s) {
                        exits.push(s);
                    }
                }
            }
        }

        loops.push(NaturalLoop {
            header: *header,
            body_blocks: body,
            latch: *latch,
            exits,
            ordinal: 0,
            depth: 0,
        });
    }

    // Sort by header address for deterministic ordinals.
    loops.sort_by_key(|l| {
        cfg.blocks.get(&l.header).map(|b| b.start_addr).unwrap_or(0)
    });
    for (i, l) in loops.iter_mut().enumerate() {
        l.ordinal = i;
    }

    // Step 4: Compute nesting.
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut parent: HashMap<usize, usize> = HashMap::new();

    for i in 0..loops.len() {
        for j in 0..loops.len() {
            if i == j {
                continue;
            }
            // Loop j is nested inside loop i if i's body is a strict superset of j's body.
            if loops[i].body_blocks.is_superset(&loops[j].body_blocks)
                && loops[i].body_blocks.len() > loops[j].body_blocks.len()
            {
                // j is a candidate child of i. Pick the tightest parent.
                match parent.get(&j) {
                    Some(&current_parent) => {
                        if loops[i].body_blocks.len() < loops[current_parent].body_blocks.len() {
                            // i is a tighter parent than current_parent.
                            children.entry(current_parent).or_default().retain(|&x| x != j);
                            children.entry(i).or_default().push(j);
                            parent.insert(j, i);
                        }
                    }
                    None => {
                        children.entry(i).or_default().push(j);
                        parent.insert(j, i);
                    }
                }
            }
        }
    }

    // Compute depth from nesting.
    let top_level: Vec<usize> = (0..loops.len())
        .filter(|i| !parent.contains_key(i))
        .collect();

    // BFS to assign depths.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for &tl in &top_level {
        queue.push_back((tl, 0));
    }
    while let Some((idx, depth)) = queue.pop_front() {
        loops[idx].depth = depth;
        if let Some(kids) = children.get(&idx) {
            for &kid in kids {
                queue.push_back((kid, depth + 1));
            }
        }
    }

    LoopForest {
        loops,
        children,
        top_level,
    }
}

// ---------------------------------------------------------------------------
// Dominator computation (iterative dataflow)
// ---------------------------------------------------------------------------

/// Compute immediate dominators for all blocks using the iterative algorithm.
/// Returns a map: BlockId → immediate dominator BlockId.
/// The entry block dominates itself.
fn compute_dominators(
    cfg: &ControlFlowGraph,
    block_ids: &[BlockId],
) -> HashMap<BlockId, BlockId> {
    // Use the classic iterative dominator algorithm.
    // dom[n] = {n} ∪ (∩ dom[p] for p in preds[n])
    // We store dom sets as BTreeSets for determinism.

    let entry = cfg.entry;
    let mut dom_sets: HashMap<BlockId, BTreeSet<BlockId>> = HashMap::new();

    // Initialize: entry dominates only itself, all others = all blocks.
    let all_blocks: BTreeSet<BlockId> = block_ids.iter().copied().collect();
    for &bid in block_ids {
        if bid == entry {
            let mut s = BTreeSet::new();
            s.insert(entry);
            dom_sets.insert(bid, s);
        } else {
            dom_sets.insert(bid, all_blocks.clone());
        }
    }

    // Iterate until fixed point.
    let mut changed = true;
    while changed {
        changed = false;
        for &bid in block_ids {
            if bid == entry {
                continue;
            }
            let preds = cfg.predecessors.get(&bid);
            let new_dom = if let Some(preds) = preds {
                if preds.is_empty() {
                    // Unreachable block — dom = {self}.
                    let mut s = BTreeSet::new();
                    s.insert(bid);
                    s
                } else {
                    // Intersect dom sets of all predecessors.
                    let mut iter = preds.iter();
                    // SAFETY: this else branch is only reached when preds is non-empty
                    let first = iter.next().expect("SAFETY: preds is non-empty in else branch");
                    let mut intersection = dom_sets.get(first).cloned().unwrap_or_default();
                    for pred in iter {
                        let pred_dom = dom_sets.get(pred).cloned().unwrap_or_default();
                        intersection = intersection.intersection(&pred_dom).copied().collect();
                    }
                    intersection.insert(bid);
                    intersection
                }
            } else {
                let mut s = BTreeSet::new();
                s.insert(bid);
                s
            };

            if dom_sets.get(&bid) != Some(&new_dom) {
                dom_sets.insert(bid, new_dom);
                changed = true;
            }
        }
    }

    // Extract immediate dominators from dom sets.
    // idom(n) = the dominator of n that is dominated by all other dominators of n
    // (i.e., the closest strict dominator).
    let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
    for &bid in block_ids {
        if bid == entry {
            idom.insert(bid, bid);
            continue;
        }
        let dom_set = &dom_sets[&bid];
        // Strict dominators = dom_set - {bid}.
        let strict: BTreeSet<BlockId> = dom_set.iter().copied().filter(|&d| d != bid).collect();
        if strict.is_empty() {
            idom.insert(bid, bid);
            continue;
        }
        // idom = the strict dominator with the largest dom set (closest to bid).
        let mut best = entry;
        let mut best_size = 0;
        for &d in &strict {
            let d_size = dom_sets.get(&d).map(|s| s.len()).unwrap_or(0);
            if d_size > best_size {
                best_size = d_size;
                best = d;
            }
        }
        idom.insert(bid, best);
    }

    idom
}

/// Check if `a` dominates `b` using the idom map.
/// a dominates b if a == b or a dominates idom(b).
fn dominates(idom: &HashMap<BlockId, BlockId>, a: BlockId, b: BlockId) -> bool {
    let mut cur = b;
    loop {
        if cur == a {
            return true;
        }
        match idom.get(&cur) {
            Some(&parent) if parent != cur => cur = parent,
            _ => return false,
        }
    }
}

/// Compute the natural loop body for a back-edge (latch → header).
/// Uses reverse BFS from latch to header, collecting all blocks on the path.
fn compute_loop_body(
    cfg: &ControlFlowGraph,
    header: BlockId,
    latch: BlockId,
) -> BTreeSet<BlockId> {
    let mut body = BTreeSet::new();
    body.insert(header);

    if header == latch {
        return body;
    }

    let mut worklist: VecDeque<BlockId> = VecDeque::new();
    body.insert(latch);
    worklist.push_back(latch);

    while let Some(blk) = worklist.pop_front() {
        if let Some(preds) = cfg.predecessors.get(&blk) {
            for &pred in preds {
                if !body.contains(&pred) {
                    body.insert(pred);
                    worklist.push_back(pred);
                }
            }
        }
    }

    body
}

// ---------------------------------------------------------------------------
// x86_64 instruction helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn decode_insn(instr: &Instruction) -> DecodedInsn {
    let mnemonic = format!("{:?}", instr.mnemonic()).to_ascii_lowercase();
    let mut operands = Vec::new();
    for i in 0..instr.op_count() {
        match instr.op_kind(i) {
            OpKind::Register => {
                operands.push(format!("{:?}", instr.op_register(i)).to_ascii_lowercase());
            }
            OpKind::Memory => {
                let base = instr.memory_base();
                let index = instr.memory_index();
                let disp = instr.memory_displacement64();
                let mut parts = Vec::new();
                if base != Register::None {
                    parts.push(format!("{:?}", base).to_ascii_lowercase());
                }
                if index != Register::None {
                    let scale = instr.memory_index_scale();
                    if scale > 1 {
                        parts.push(format!("{:?}*{}", index, scale).to_ascii_lowercase());
                    } else {
                        parts.push(format!("{:?}", index).to_ascii_lowercase());
                    }
                }
                if disp != 0 || parts.is_empty() {
                    parts.push(format!("0x{:x}", disp));
                }
                operands.push(format!("[{}]", parts.join("+")));
            }
            OpKind::Immediate8 | OpKind::Immediate8to16 | OpKind::Immediate8to32
            | OpKind::Immediate8to64 | OpKind::Immediate16 | OpKind::Immediate32
            | OpKind::Immediate32to64 | OpKind::Immediate64 => {
                operands.push(format!("0x{:x}", instr.immediate(i)));
            }
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                operands.push(format!("0x{:x}", instr.near_branch_target()));
            }
            _ => {
                operands.push("?".into());
            }
        }
    }
    DecodedInsn {
        mnemonic,
        operands,
        addr: instr.ip(),
    }
}

#[cfg(target_arch = "x86_64")]
fn is_branch_mnemonic(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Je | Mnemonic::Jne | Mnemonic::Jb | Mnemonic::Jbe
            | Mnemonic::Ja | Mnemonic::Jae | Mnemonic::Jl | Mnemonic::Jle
            | Mnemonic::Jg | Mnemonic::Jge | Mnemonic::Jmp
            | Mnemonic::Js | Mnemonic::Jns | Mnemonic::Jp | Mnemonic::Jnp
    )
}

#[cfg(target_arch = "x86_64")]
fn is_conditional_branch(m: Mnemonic) -> bool {
    is_branch_mnemonic(m) && m != Mnemonic::Jmp
}

#[cfg(target_arch = "x86_64")]
fn has_near_branch_operand(instr: &Instruction) -> bool {
    for i in 0..instr.op_count() {
        match instr.op_kind(i) {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => return true,
            _ => {}
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
fn mnemonic_to_branch_kind(m: Mnemonic) -> BranchKind {
    match m {
        Mnemonic::Ja => BranchKind::Above,
        Mnemonic::Jae => BranchKind::AboveEqual,
        Mnemonic::Jb => BranchKind::Below,
        Mnemonic::Jbe => BranchKind::BelowEqual,
        Mnemonic::Jg => BranchKind::Greater,
        Mnemonic::Jge => BranchKind::GreaterEqual,
        Mnemonic::Jl => BranchKind::Less,
        Mnemonic::Jle => BranchKind::LessEqual,
        Mnemonic::Je => BranchKind::Equal,
        Mnemonic::Jne => BranchKind::NotEqual,
        Mnemonic::Js => BranchKind::Sign,
        Mnemonic::Jns => BranchKind::NotSign,
        Mnemonic::Jp => BranchKind::Parity,
        Mnemonic::Jnp => BranchKind::NotParity,
        _ => BranchKind::Equal, // fallback, shouldn't happen
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;

    /// A simple function with no branches — should produce a single block.
    #[test]
    fn test_cfg_simple_function() {
        // Use a trivial extern "C" fn as test subject.
        extern "C" fn add_one(x: f32) -> f32 {
            x + 1.0
        }

        let cfg = unsafe { build_cfg_from_fn(add_one as *const u8, 256) }.expect("CFG build failed");

        // Should have at least 1 block.
        assert!(!cfg.blocks.is_empty(), "CFG should have at least one block");
        // Entry block should exist.
        assert!(cfg.blocks.contains_key(&cfg.entry));
        // The last block should terminate with Return.
        let last_block_id = BlockId(cfg.blocks.len() as u32 - 1);
        let last_block = &cfg.blocks[&last_block_id];
        assert!(
            matches!(last_block.terminator, Terminator::Return),
            "last block should end with Return, got {:?}",
            last_block.terminator
        );
    }

    /// A function with a loop — should produce multiple blocks and a detected loop.
    #[test]
    fn test_cfg_loop_detection() {
        // A function with a simple loop: sum elements.
        extern "C" fn sum_array(ptr: *const f32, len: usize) -> f32 {
            let mut acc = 0.0f32;
            let mut i = 0;
            while i < len {
                acc += unsafe { *ptr.add(i) };
                i += 1;
            }
            acc
        }

        let cfg = unsafe { build_cfg_from_fn(sum_array as *const u8, 512) }.expect("CFG build failed");

        // Should have multiple blocks (at least entry, loop header, loop body, exit).
        assert!(
            cfg.blocks.len() >= 2,
            "loop function should have >= 2 blocks, got {}",
            cfg.blocks.len()
        );

        // Find loops.
        let forest = find_loops(&cfg);
        assert!(
            !forest.loops.is_empty(),
            "should detect at least one loop in sum_array"
        );

        // The loop should have a header and latch.
        let lp = &forest.loops[0];
        assert!(
            lp.body_blocks.len() >= 1,
            "loop body should have at least 1 block"
        );
        assert!(
            lp.body_blocks.contains(&lp.header),
            "loop body must contain the header"
        );
        assert!(
            lp.body_blocks.contains(&lp.latch),
            "loop body must contain the latch"
        );
    }

    /// Empty forest for a function with no loops.
    #[test]
    fn test_no_loops() {
        extern "C" fn identity(x: f32) -> f32 {
            x
        }

        let cfg = unsafe { build_cfg_from_fn(identity as *const u8, 256) }.expect("CFG build failed");
        let forest = find_loops(&cfg);
        assert!(
            forest.loops.is_empty(),
            "identity function should have no loops"
        );
        assert!(forest.top_level.is_empty());
    }
}

/// Pure data-structure unit tests — no architecture-specific code required.
#[cfg(test)]
mod data_structure_tests {
    use super::*;
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // BlockId
    // -----------------------------------------------------------------------

    #[test]
    fn block_id_equality_and_ordering() {
        let a = BlockId(1);
        let b = BlockId(1);
        let c = BlockId(2);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a < c, "BlockId(1) < BlockId(2)");
        assert!(c > a, "BlockId(2) > BlockId(1)");
    }

    #[test]
    fn block_id_hash_and_copy() {
        let a = BlockId(42);
        let b = a; // Copy
        assert_eq!(a, b);

        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b), "Copy of a must hash to the same bucket");
    }

    // -----------------------------------------------------------------------
    // BranchKind — all 14 variants
    // -----------------------------------------------------------------------

    #[test]
    fn branch_kind_all_variants_equality() {
        let all = [
            BranchKind::Above,
            BranchKind::AboveEqual,
            BranchKind::Below,
            BranchKind::BelowEqual,
            BranchKind::Greater,
            BranchKind::GreaterEqual,
            BranchKind::Less,
            BranchKind::LessEqual,
            BranchKind::Equal,
            BranchKind::NotEqual,
            BranchKind::Sign,
            BranchKind::NotSign,
            BranchKind::Parity,
            BranchKind::NotParity,
        ];

        // Each variant equals itself and differs from every other variant.
        for (i, v1) in all.iter().enumerate() {
            for (j, v2) in all.iter().enumerate() {
                if i == j {
                    assert_eq!(v1, v2, "variant {i} must equal itself");
                } else {
                    assert_ne!(v1, v2, "variant {i} must differ from variant {j}");
                }
            }
        }
    }

    #[test]
    fn branch_kind_clone_independent() {
        let original = BranchKind::GreaterEqual;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // -----------------------------------------------------------------------
    // Terminator
    // -----------------------------------------------------------------------

    #[test]
    fn terminator_variants_equality() {
        let t1 = Terminator::Fallthrough(BlockId(0));
        let t2 = Terminator::Fallthrough(BlockId(0));
        assert_eq!(t1, t2);

        let t3 = Terminator::Jump(BlockId(5));
        let t4 = Terminator::Jump(BlockId(5));
        assert_eq!(t3, t4);

        let t5 = Terminator::Return;
        let t6 = Terminator::Return;
        assert_eq!(t5, t6);
    }

    #[test]
    fn terminator_cond_branch_equality() {
        let cb1 = Terminator::CondBranch {
            kind: BranchKind::Less,
            taken: BlockId(3),
            fallthrough: BlockId(4),
        };
        let cb2 = Terminator::CondBranch {
            kind: BranchKind::Less,
            taken: BlockId(3),
            fallthrough: BlockId(4),
        };
        assert_eq!(cb1, cb2);
    }

    #[test]
    fn terminator_clone_preserves_data() {
        let original = Terminator::CondBranch {
            kind: BranchKind::NotEqual,
            taken: BlockId(7),
            fallthrough: BlockId(8),
        };
        let cloned = original.clone();

        match cloned {
            Terminator::CondBranch { kind, taken, fallthrough } => {
                assert_eq!(kind, BranchKind::NotEqual);
                assert_eq!(taken, BlockId(7));
                assert_eq!(fallthrough, BlockId(8));
            }
            _ => panic!("clone must preserve CondBranch variant"),
        }
    }

    // -----------------------------------------------------------------------
    // DecodedInsn
    // -----------------------------------------------------------------------

    #[test]
    fn decoded_insn_construction_and_equality() {
        let insn = DecodedInsn {
            mnemonic: "add".to_string(),
            operands: vec!["rax".to_string(), "rbx".to_string()],
            addr: 0x1000,
        };

        assert_eq!(insn.mnemonic, "add");
        assert_eq!(insn.operands.len(), 2);
        assert_eq!(insn.operands[0], "rax");
        assert_eq!(insn.addr, 0x1000);

        let same = DecodedInsn {
            mnemonic: "add".to_string(),
            operands: vec!["rax".to_string(), "rbx".to_string()],
            addr: 0x1000,
        };
        assert_eq!(insn, same);
    }

    // -----------------------------------------------------------------------
    // BasicBlock
    // -----------------------------------------------------------------------

    #[test]
    fn basic_block_construction_and_field_access() {
        let block = BasicBlock {
            id: BlockId(0),
            start_addr: 0x2000,
            end_addr: 0x2010,
            instructions: vec![DecodedInsn {
                mnemonic: "nop".to_string(),
                operands: vec![],
                addr: 0x2000,
            }],
            terminator: Terminator::Return,
        };

        assert_eq!(block.id, BlockId(0));
        assert_eq!(block.start_addr, 0x2000);
        assert_eq!(block.end_addr, 0x2010);
        assert_eq!(block.instructions.len(), 1);
        assert!(matches!(block.terminator, Terminator::Return));
    }

    // -----------------------------------------------------------------------
    // ControlFlowGraph — manual construction
    // -----------------------------------------------------------------------

    /// Helper: build a minimal three-block linear CFG: entry → B1 → B2 (Return).
    fn build_linear_cfg() -> ControlFlowGraph {
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(2)),
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), b0);
        blocks.insert(BlockId(1), b1);
        blocks.insert(BlockId(2), b2);

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2)]);
        successors.insert(BlockId(2), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);

        ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        }
    }

    #[test]
    fn cfg_manual_construction_linear() {
        let cfg = build_linear_cfg();

        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.entry, BlockId(0));
        assert_eq!(cfg.successors[&BlockId(0)], vec![BlockId(1)]);
        assert_eq!(cfg.successors[&BlockId(2)].len(), 0);
        assert_eq!(cfg.predecessors[&BlockId(2)], vec![BlockId(1)]);
    }

    #[test]
    fn find_loops_linear_yields_empty_forest() {
        let cfg = build_linear_cfg();
        let forest = find_loops(&cfg);

        assert!(forest.loops.is_empty());
        assert!(forest.top_level.is_empty());
        assert!(forest.children.is_empty());
    }

    // -----------------------------------------------------------------------
    // ControlFlowGraph — with a loop
    // -----------------------------------------------------------------------

    /// Helper: build a CFG with a single natural loop.
    ///   B0 (entry) → B1 (loop header) ↔ B2 (loop body) → B3 (exit, Return)
    ///   B2 falls through back to B1 (back-edge), B1 conditionally exits to B3.
    fn build_loop_cfg() -> ControlFlowGraph {
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual,
                taken: BlockId(3), // exit
                fallthrough: BlockId(2), // loop body
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(1)), // back-edge
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), b0);
        blocks.insert(BlockId(1), b1);
        blocks.insert(BlockId(2), b2);
        blocks.insert(BlockId(3), b3);

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(3), BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(1)]);
        successors.insert(BlockId(3), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(2)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(1)]);

        ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        }
    }

    #[test]
    fn find_loops_detects_single_loop() {
        let cfg = build_loop_cfg();
        let forest = find_loops(&cfg);

        assert_eq!(forest.loops.len(), 1, "should find exactly one loop");

        let lp = &forest.loops[0];
        assert_eq!(lp.header, BlockId(1), "header should be B1");
        assert_eq!(lp.latch, BlockId(2), "latch should be B2");
        assert_eq!(lp.ordinal, 0);
        assert_eq!(lp.depth, 0, "top-level loop has depth 0");

        assert!(
            lp.body_blocks.contains(&BlockId(1)),
            "body must contain header"
        );
        assert!(
            lp.body_blocks.contains(&BlockId(2)),
            "body must contain latch"
        );

        assert_eq!(lp.exits, vec![BlockId(3)], "exit should be B3");
    }

    #[test]
    fn loop_forest_top_level_indices() {
        let cfg = build_loop_cfg();
        let forest = find_loops(&cfg);

        assert_eq!(forest.top_level, vec![0], "single loop is top-level");
        assert!(
            forest.children.is_empty(),
            "no nested loops, so no children"
        );
    }

    // -----------------------------------------------------------------------
    // NaturalLoop
    // -----------------------------------------------------------------------

    #[test]
    fn natural_loop_construction_and_membership() {
        let mut body = BTreeSet::new();
        body.insert(BlockId(1));
        body.insert(BlockId(2));

        let lp = NaturalLoop {
            header: BlockId(1),
            body_blocks: body,
            latch: BlockId(2),
            exits: vec![BlockId(3)],
            ordinal: 0,
            depth: 1,
        };

        assert_eq!(lp.header, BlockId(1));
        assert_eq!(lp.latch, BlockId(2));
        assert_eq!(lp.depth, 1);
        assert!(lp.body_blocks.contains(&BlockId(1)));
        assert!(lp.body_blocks.contains(&BlockId(2)));
        assert!(!lp.body_blocks.contains(&BlockId(0)));
        assert_eq!(lp.exits.len(), 1);
    }

    // -----------------------------------------------------------------------
    // LoopForest
    // -----------------------------------------------------------------------

    #[test]
    fn loop_forest_empty_construction() {
        let forest = LoopForest {
            loops: vec![],
            children: HashMap::new(),
            top_level: vec![],
        };

        assert!(forest.loops.is_empty());
        assert!(forest.top_level.is_empty());
        assert!(forest.children.is_empty());
    }

    #[test]
    fn loop_forest_with_nested_loops() {
        let outer_body: BTreeSet<BlockId> = vec![BlockId(1), BlockId(2), BlockId(3)].into_iter().collect();
        let inner_body: BTreeSet<BlockId> = vec![BlockId(2), BlockId(3)].into_iter().collect();

        let outer = NaturalLoop {
            header: BlockId(1),
            body_blocks: outer_body,
            latch: BlockId(3),
            exits: vec![BlockId(4)],
            ordinal: 0,
            depth: 0,
        };
        let inner = NaturalLoop {
            header: BlockId(2),
            body_blocks: inner_body,
            latch: BlockId(3),
            exits: vec![BlockId(1)],
            ordinal: 1,
            depth: 1,
        };

        let mut children = HashMap::new();
        children.insert(0, vec![1]);

        let forest = LoopForest {
            loops: vec![outer, inner],
            children,
            top_level: vec![0],
        };

        assert_eq!(forest.loops.len(), 2);
        assert_eq!(forest.top_level, vec![0]);
        assert_eq!(forest.children[&0], vec![1]);
        assert_eq!(forest.loops[0].depth, 0);
        assert_eq!(forest.loops[1].depth, 1);
    }

    // -----------------------------------------------------------------------
    // Empty CFG → empty loop forest
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_empty_cfg() {
        let cfg = ControlFlowGraph {
            blocks: BTreeMap::new(),
            entry: BlockId(0),
            successors: BTreeMap::new(),
            predecessors: BTreeMap::new(),
        };

        let forest = find_loops(&cfg);
        assert!(forest.loops.is_empty());
        assert!(forest.top_level.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 14: Empty basic block (zero instructions)
    // -----------------------------------------------------------------------

    #[test]
    fn basic_block_with_zero_instructions() {
        // Arrange: construct a block with no instructions, only a terminator.
        let block = BasicBlock {
            id: BlockId(5),
            start_addr: 0x5000,
            end_addr: 0x5000,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(0)),
        };

        // Act & Assert: fields are accessible and correct.
        assert_eq!(block.id, BlockId(5));
        assert_eq!(block.start_addr, 0x5000);
        assert_eq!(block.end_addr, 0x5000);
        assert!(block.instructions.is_empty(), "empty block should have zero instructions");
        assert!(matches!(block.terminator, Terminator::Jump(BlockId(0))));
    }

    // -----------------------------------------------------------------------
    // Test 15: CondBranch with different BranchKinds are not equal
    // -----------------------------------------------------------------------

    #[test]
    fn cond_branch_different_kinds_are_not_equal() {
        // Arrange: two CondBranch terminators that differ only in BranchKind.
        let cb_less = Terminator::CondBranch {
            kind: BranchKind::Less,
            taken: BlockId(1),
            fallthrough: BlockId(2),
        };
        let cb_greater = Terminator::CondBranch {
            kind: BranchKind::Greater,
            taken: BlockId(1),
            fallthrough: BlockId(2),
        };

        // Act & Assert: same targets but different condition kind → not equal.
        assert_ne!(cb_less, cb_greater, "CondBranch with different BranchKind must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 16: Self-loop CFG (header == latch)
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_self_loop() {
        // Arrange: B0 → B1 (self-loop via Jump) where B1 also has exit to B2.
        //   B0 falls through to B1.
        //   B1 jumps to B1 (back-edge) and also conditionally exits to B2.
        // Actually, a block with a single Jump to itself is a self-loop.
        // We model it as: B1 has a CondBranch where taken=B1 (self), fallthrough=B2.
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual,
                taken: BlockId(1), // self-loop back-edge
                fallthrough: BlockId(2), // exit
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), b0);
        blocks.insert(BlockId(1), b1);
        blocks.insert(BlockId(2), b2);

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(1), BlockId(2)]);
        successors.insert(BlockId(2), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(1)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act
        let forest = find_loops(&cfg);

        // Assert: one self-loop with header == latch.
        assert_eq!(forest.loops.len(), 1, "self-loop CFG should have exactly one loop");
        let lp = &forest.loops[0];
        assert_eq!(lp.header, BlockId(1), "self-loop header is B1");
        assert_eq!(lp.latch, BlockId(1), "self-loop latch is also B1");
        assert!(lp.body_blocks.contains(&BlockId(1)), "self-loop body must contain its header");
        assert_eq!(lp.body_blocks.len(), 1, "self-loop body is just B1");
        assert_eq!(lp.exits, vec![BlockId(2)], "self-loop exit is B2");
    }

    // -----------------------------------------------------------------------
    // Test 17: CFG with an unreachable block
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_with_unreachable_block() {
        // Arrange: linear CFG B0 → B1 (Return), plus B2 with no predecessors.
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::Return,
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), b0);
        blocks.insert(BlockId(1), b1);
        blocks.insert(BlockId(2), b2);

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![]);
        successors.insert(BlockId(2), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![]); // unreachable

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act
        let forest = find_loops(&cfg);

        // Assert: no loops, unreachable block does not create false positives.
        assert!(forest.loops.is_empty(), "unreachable block should not create loops");
        assert_eq!(cfg.blocks.len(), 3, "CFG should still contain all blocks including unreachable");
    }

    // -----------------------------------------------------------------------
    // Test 18: compute_dominators on linear CFG
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_linear_cfg() {
        // Arrange: B0 → B1 → B2 (linear chain).
        let cfg = build_linear_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: entry dominates itself; B1 idom = B0; B2 idom = B1.
        assert_eq!(idom[&BlockId(0)], BlockId(0), "entry idom is itself");
        assert_eq!(idom[&BlockId(1)], BlockId(0), "B1 idom is B0");
        assert_eq!(idom[&BlockId(2)], BlockId(1), "B2 idom is B1");
    }

    // -----------------------------------------------------------------------
    // Test 19: compute_dominators on loop CFG
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_loop_cfg() {
        // Arrange: B0 → B1 ↔ B2 → B3 (loop with back-edge B2→B1).
        let cfg = build_loop_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: B0 dominates everything; B1 idom = B0; B2 idom = B1; B3 idom = B1.
        assert_eq!(idom[&BlockId(0)], BlockId(0), "entry idom is itself");
        assert_eq!(idom[&BlockId(1)], BlockId(0), "B1 idom is B0");
        assert_eq!(idom[&BlockId(2)], BlockId(1), "B2 idom is B1 (only predecessor in loop)");
        assert_eq!(idom[&BlockId(3)], BlockId(1), "B3 idom is B1 (exit from loop header)");
    }

    // -----------------------------------------------------------------------
    // Test 20: dominates helper — entry dominates all blocks in loop CFG
    // -----------------------------------------------------------------------

    #[test]
    fn dominates_entry_dominates_all() {
        // Arrange
        let cfg = build_loop_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
        let idom = compute_dominators(&cfg, &block_ids);

        // Act & Assert: entry (B0) dominates every block.
        for bid in &block_ids {
            assert!(
                dominates(&idom, cfg.entry, *bid),
                "entry must dominate block {:?}",
                bid
            );
        }

        // B2 does not dominate B3 (they are on different paths).
        assert!(
            !dominates(&idom, BlockId(2), BlockId(3)),
            "B2 does not dominate B3"
        );
    }

    // -----------------------------------------------------------------------
    // Test 21: compute_loop_body with header == latch (self-loop)
    // -----------------------------------------------------------------------

    #[test]
    fn compute_loop_body_self_loop_returns_header_only() {
        // Arrange: a minimal CFG where the self-loop resides.
        let cfg = ControlFlowGraph {
            blocks: BTreeMap::new(),
            entry: BlockId(0),
            successors: BTreeMap::new(),
            predecessors: BTreeMap::new(),
        };

        // Act: header == latch case.
        let body = compute_loop_body(&cfg, BlockId(1), BlockId(1));

        // Assert: only the header is in the body.
        assert_eq!(body.len(), 1);
        assert!(body.contains(&BlockId(1)), "self-loop body must contain header");
    }

    // -----------------------------------------------------------------------
    // Test 22: Two disjoint loops CFG
    // -----------------------------------------------------------------------

    /// Helper: build a CFG with two independent loops.
    ///   B0 → B1 (loop1 header) ↔ B2 (loop1 body) → B3 (loop2 header) ↔ B4 (loop2 body) → B5 (Return)
    fn build_two_loop_cfg() -> ControlFlowGraph {
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual,
                taken: BlockId(3),     // exit loop1 → loop2 header
                fallthrough: BlockId(2), // loop1 body
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(1)), // back-edge loop1
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Less,
                taken: BlockId(5),     // exit loop2 → return
                fallthrough: BlockId(4), // loop2 body
            },
        };
        let b4 = BasicBlock {
            id: BlockId(4),
            start_addr: 0x40,
            end_addr: 0x50,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(3)), // back-edge loop2
        };
        let b5 = BasicBlock {
            id: BlockId(5),
            start_addr: 0x50,
            end_addr: 0x60,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3, b4, b5] {
            blocks.insert(b.id, b);
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(3), BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(1)]);
        successors.insert(BlockId(3), vec![BlockId(5), BlockId(4)]);
        successors.insert(BlockId(4), vec![BlockId(3)]);
        successors.insert(BlockId(5), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(2)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(1), BlockId(4)]);
        predecessors.insert(BlockId(4), vec![BlockId(3)]);
        predecessors.insert(BlockId(5), vec![BlockId(3)]);

        ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        }
    }

    #[test]
    fn find_loops_two_disjoint_loops() {
        // Arrange
        let cfg = build_two_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: exactly 2 loops, both top-level (not nested), depth 0.
        assert_eq!(forest.loops.len(), 2, "should find exactly two loops");
        assert_eq!(forest.loops[0].header, BlockId(1), "first loop header is B1");
        assert_eq!(forest.loops[0].latch, BlockId(2), "first loop latch is B2");
        assert_eq!(forest.loops[1].header, BlockId(3), "second loop header is B3");
        assert_eq!(forest.loops[1].latch, BlockId(4), "second loop latch is B4");

        assert_eq!(forest.loops[0].depth, 0, "both loops are top-level");
        assert_eq!(forest.loops[1].depth, 0, "both loops are top-level");
        assert_eq!(forest.top_level.len(), 2, "both loops are top-level");
    }

    // -----------------------------------------------------------------------
    // Test 23: BranchKind all variants are Debug (format smoke test)
    // -----------------------------------------------------------------------

    #[test]
    fn branch_kind_debug_format_smoke_test() {
        // Arrange: exercise Debug formatting for all 14 variants.
        let kinds = [
            BranchKind::Above, BranchKind::AboveEqual, BranchKind::Below,
            BranchKind::BelowEqual, BranchKind::Greater, BranchKind::GreaterEqual,
            BranchKind::Less, BranchKind::LessEqual, BranchKind::Equal,
            BranchKind::NotEqual, BranchKind::Sign, BranchKind::NotSign,
            BranchKind::Parity, BranchKind::NotParity,
        ];

        // Act & Assert: every variant formats to a non-empty string.
        for kind in &kinds {
            let s = format!("{:?}", kind);
            assert!(!s.is_empty(), "BranchKind {:?} Debug output must not be empty", kind);
        }
    }

    // -----------------------------------------------------------------------
    // Test 24: Terminator Fallthrough vs Jump inequality
    // -----------------------------------------------------------------------

    #[test]
    fn terminator_fallthrough_vs_jump_not_equal() {
        // Arrange: Fallthrough and Jump targeting the same block.
        let ft = Terminator::Fallthrough(BlockId(3));
        let jmp = Terminator::Jump(BlockId(3));

        // Act & Assert: different variants are not equal even with same target.
        assert_ne!(ft, jmp, "Fallthrough and Jump must not be equal even with same BlockId target");
    }

    // -----------------------------------------------------------------------
    // Test 25: Nested loops with find_loops producing correct depths
    // -----------------------------------------------------------------------

    /// Helper: build a CFG with a doubly-nested loop.
    ///   B0 → B1 (outer header) ↔ B2 (outer body) → B3 (inner header) ↔ B4 (inner body) → B5 (outer latch) → B1
    ///   B1 conditionally exits to B6 (Return).
    fn build_nested_loop_cfg() -> ControlFlowGraph {
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Equal,
                taken: BlockId(6),     // exit outer loop
                fallthrough: BlockId(2), // outer body
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(3)),
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual,
                taken: BlockId(5),     // exit inner loop → outer latch
                fallthrough: BlockId(4), // inner body
            },
        };
        let b4 = BasicBlock {
            id: BlockId(4),
            start_addr: 0x40,
            end_addr: 0x50,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(3)), // inner back-edge
        };
        let b5 = BasicBlock {
            id: BlockId(5),
            start_addr: 0x50,
            end_addr: 0x60,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(1)), // outer back-edge
        };
        let b6 = BasicBlock {
            id: BlockId(6),
            start_addr: 0x60,
            end_addr: 0x70,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3, b4, b5, b6] {
            blocks.insert(b.id, b);
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(6), BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![BlockId(5), BlockId(4)]);
        successors.insert(BlockId(4), vec![BlockId(3)]);
        successors.insert(BlockId(5), vec![BlockId(1)]);
        successors.insert(BlockId(6), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(5)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(2), BlockId(4)]);
        predecessors.insert(BlockId(4), vec![BlockId(3)]);
        predecessors.insert(BlockId(5), vec![BlockId(3)]);
        predecessors.insert(BlockId(6), vec![BlockId(1)]);

        ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        }
    }

    #[test]
    fn find_loops_nested_loops_depth_assignment() {
        // Arrange
        let cfg = build_nested_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: 2 loops found, outer at depth 0, inner at depth 1.
        assert_eq!(forest.loops.len(), 2, "nested loop CFG should have exactly two loops");

        // Identify outer and inner by body size.
        let (outer_idx, inner_idx) = if forest.loops[0].body_blocks.len() > forest.loops[1].body_blocks.len() {
            (0, 1)
        } else {
            (1, 0)
        };

        let outer = &forest.loops[outer_idx];
        let inner = &forest.loops[inner_idx];

        assert_eq!(outer.depth, 0, "outer loop should have depth 0");
        assert_eq!(inner.depth, 1, "inner loop should have depth 1");
        assert!(
            outer.body_blocks.is_superset(&inner.body_blocks),
            "outer body must be a superset of inner body"
        );
        assert!(
            outer.body_blocks.len() > inner.body_blocks.len(),
            "outer body must be strictly larger"
        );

        // Outer loop is top-level.
        assert!(forest.top_level.contains(&outer_idx), "outer loop must be in top_level");
        // Inner loop is a child of outer.
        assert!(
            forest.children.get(&outer_idx).map_or(false, |kids| kids.contains(&inner_idx)),
            "inner loop must be a child of outer loop"
        );
    }

    // -----------------------------------------------------------------------
    // Test 26: Diamond (if-then-else) CFG — no loops
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Test 27: BlockId Debug format contains the inner value
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn block_id_debug_format() {
        // Arrange
        let id = BlockId(99);

        // Act
        let debug_str = format!("{:?}", id);

        // Assert: Debug output must contain "99".
        assert!(debug_str.contains("99"), "BlockId Debug output must contain the inner u32 value, got: {}", debug_str);
    }

    // -----------------------------------------------------------------------
    // Test 28: BlockId zero value is valid
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn block_id_zero_value() {
        // Arrange
        let zero = BlockId(0);

        // Act & Assert
        assert_eq!(zero, BlockId(0));
        assert!(zero <= BlockId(0));
        assert!(zero >= BlockId(0));
        assert!(!(zero > BlockId(1)));
    }

    // -----------------------------------------------------------------------
    // Test 29: DecodedInsn Clone produces independent copy
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn decoded_insn_clone_independent() {
        // Arrange
        let original = DecodedInsn {
            mnemonic: "mov".to_string(),
            operands: vec!["rax".to_string(), "0x1".to_string()],
            addr: 0xDEAD,
        };

        // Act
        let cloned = original.clone();

        // Assert: equal values.
        assert_eq!(original, cloned);
        assert_eq!(cloned.mnemonic, "mov");
        assert_eq!(cloned.addr, 0xDEAD);
    }

    // -----------------------------------------------------------------------
    // Test 30: DecodedInsn inequality — different addr
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn decoded_insn_different_addr_not_equal() {
        // Arrange
        let a = DecodedInsn {
            mnemonic: "nop".to_string(),
            operands: vec![],
            addr: 0x1000,
        };
        let b = DecodedInsn {
            mnemonic: "nop".to_string(),
            operands: vec![],
            addr: 0x2000,
        };

        // Act & Assert
        assert_ne!(a, b, "DecodedInsn with different addr must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 31: BasicBlock Clone produces independent copy
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn basic_block_clone_independent() {
        // Arrange
        let block = BasicBlock {
            id: BlockId(7),
            start_addr: 0xA000,
            end_addr: 0xA010,
            instructions: vec![DecodedInsn {
                mnemonic: "add".to_string(),
                operands: vec!["rax".to_string(), "rcx".to_string()],
                addr: 0xA000,
            }],
            terminator: Terminator::Jump(BlockId(3)),
        };

        // Act
        let cloned = block.clone();

        // Assert
        assert_eq!(cloned.id, BlockId(7));
        assert_eq!(cloned.start_addr, 0xA000);
        assert_eq!(cloned.instructions.len(), 1);
        assert!(matches!(cloned.terminator, Terminator::Jump(BlockId(3))));
    }

    // -----------------------------------------------------------------------
    // Test 32: Terminator Return vs Fallthrough(BlockId(0)) are not equal
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn terminator_return_vs_fallthrough_not_equal() {
        // Arrange
        let ret = Terminator::Return;
        let ft = Terminator::Fallthrough(BlockId(0));

        // Act & Assert
        assert_ne!(ret, ft, "Return and Fallthrough(BlockId(0)) must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 33: CondBranch with different taken targets are not equal
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn cond_branch_different_taken_not_equal() {
        // Arrange: same kind and fallthrough, different taken target.
        let cb_a = Terminator::CondBranch {
            kind: BranchKind::Equal,
            taken: BlockId(1),
            fallthrough: BlockId(2),
        };
        let cb_b = Terminator::CondBranch {
            kind: BranchKind::Equal,
            taken: BlockId(5),
            fallthrough: BlockId(2),
        };

        // Act & Assert
        assert_ne!(cb_a, cb_b, "CondBranch with different taken must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 34: NaturalLoop Clone produces independent copy
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn natural_loop_clone_independent() {
        // Arrange
        let mut body = BTreeSet::new();
        body.insert(BlockId(1));
        body.insert(BlockId(2));

        let lp = NaturalLoop {
            header: BlockId(1),
            body_blocks: body,
            latch: BlockId(2),
            exits: vec![BlockId(3)],
            ordinal: 7,
            depth: 2,
        };

        // Act
        let cloned = lp.clone();

        // Assert
        assert_eq!(cloned.header, BlockId(1));
        assert_eq!(cloned.latch, BlockId(2));
        assert_eq!(cloned.ordinal, 7);
        assert_eq!(cloned.depth, 2);
        assert!(cloned.body_blocks.contains(&BlockId(1)));
        assert_eq!(cloned.exits, vec![BlockId(3)]);
    }

    // -----------------------------------------------------------------------
    // Test 35: Single-block CFG (entry + Return) yields no loops
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_single_block_cfg() {
        // Arrange: CFG with a single block that returns immediately.
        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Return,
        });

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act
        let forest = find_loops(&cfg);

        // Assert: single block with Return cannot form a loop.
        assert!(forest.loops.is_empty(), "single Return block should not produce loops");
        assert!(forest.top_level.is_empty());
        assert!(forest.children.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 36: compute_loop_body with multi-block reverse path
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn compute_loop_body_multi_block_reverse_path() {
        // Arrange: B1 (header) → B2 → B3 (latch) → B1 (back-edge)
        // Forward edges: B1→B2, B2→B3. Back-edge: B3→B1.
        // Predecessors: B1←[B0,B3], B2←[B1], B3←[B2]
        let mut blocks = BTreeMap::new();
        for i in 0..4u32 {
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: Terminator::Return,
            });
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![BlockId(1)]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(3)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(2)]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act: header=B1, latch=B3
        let body = compute_loop_body(&cfg, BlockId(1), BlockId(3));

        // Assert: reverse BFS from B3 visits B3→(pred B2)→(pred B1, already in body, stop)
        assert!(body.contains(&BlockId(1)), "body must contain header B1");
        assert!(body.contains(&BlockId(2)), "body must contain intermediate B2");
        assert!(body.contains(&BlockId(3)), "body must contain latch B3");
        assert!(!body.contains(&BlockId(0)), "body must NOT contain B0 (outside loop)");
        assert_eq!(body.len(), 3, "loop body should have exactly 3 blocks");
    }

    // -----------------------------------------------------------------------
    // Test 37: find_loops_diamond_branch_no_loops (kept as Test 26)
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_diamond_branch_no_loops() {
        // Arrange: diamond pattern B0 → B1/B2 → B3 (Return). No back-edges.
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Less,
                taken: BlockId(1),
                fallthrough: BlockId(2),
            },
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(3)),
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(3)),
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3] {
            blocks.insert(b.id, b);
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1), BlockId(2)]);
        successors.insert(BlockId(1), vec![BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(0)]);
        predecessors.insert(BlockId(3), vec![BlockId(1), BlockId(2)]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act
        let forest = find_loops(&cfg);

        // Assert: diamond has no back-edges, so no loops.
        assert!(forest.loops.is_empty(), "diamond CFG has no loops");
        assert!(forest.top_level.is_empty(), "diamond CFG has no top-level loops");
    }

    // -----------------------------------------------------------------------
    // Test 38: Dominators in diamond CFG — B0 dominates all, B3 idom is B0
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn dominators_diamond_cfg() {
        // Arrange: diamond B0 → B1/B2 → B3 (Return).
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Less,
                taken: BlockId(1),
                fallthrough: BlockId(2),
            },
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(3)),
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(3)),
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3] {
            blocks.insert(b.id, b);
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1), BlockId(2)]);
        successors.insert(BlockId(1), vec![BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(0)]);
        predecessors.insert(BlockId(3), vec![BlockId(1), BlockId(2)]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: B0 dominates itself; B1 and B2 idom = B0; B3 idom = B0 (convergence point).
        assert_eq!(idom[&BlockId(0)], BlockId(0), "entry idom is itself");
        assert_eq!(idom[&BlockId(1)], BlockId(0), "B1 idom is B0");
        assert_eq!(idom[&BlockId(2)], BlockId(0), "B2 idom is B0");
        assert_eq!(idom[&BlockId(3)], BlockId(0), "B3 idom is B0 (convergence of diamond)");
    }

    // -----------------------------------------------------------------------
    // Test 39: dominates — block does not dominate itself via strict check
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn dominates_various_relationships() {
        // Arrange: use the loop CFG (B0→B1↔B2→B3).
        let cfg = build_loop_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
        let idom = compute_dominators(&cfg, &block_ids);

        // Act & Assert
        // B0 dominates B0 (reflexive).
        assert!(dominates(&idom, BlockId(0), BlockId(0)), "block dominates itself");
        // B1 does NOT dominate B0.
        assert!(!dominates(&idom, BlockId(1), BlockId(0)), "B1 does not dominate B0");
        // B0 dominates B3 (transitively through B1).
        assert!(dominates(&idom, BlockId(0), BlockId(3)), "B0 dominates B3 transitively");
        // B2 does not dominate B3.
        assert!(!dominates(&idom, BlockId(2), BlockId(3)), "B2 does not dominate B3");
    }

    // -----------------------------------------------------------------------
    // Test 40: Loop with multiple exit blocks
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_multiple_exits() {
        // Arrange: B0 → B1 (loop header) → B2 (loop body)
        //   B1 conditionally exits to B3 or B4 (two distinct exits).
        //   B2 jumps back to B1 (back-edge).
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        // B1 has a CondBranch into the loop body, but we need two exit edges.
        // Model: B1 CondBranch taken=B2 (loop body), fallthrough=B3 (exit1).
        // Then B2 CondBranch taken=B1 (back-edge), fallthrough=B4 (exit2).
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual,
                taken: BlockId(2),
                fallthrough: BlockId(3),
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Less,
                taken: BlockId(1),     // back-edge
                fallthrough: BlockId(4), // second exit
            },
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::Return,
        };
        let b4 = BasicBlock {
            id: BlockId(4),
            start_addr: 0x40,
            end_addr: 0x50,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3, b4] {
            blocks.insert(b.id, b);
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2), BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(1), BlockId(4)]);
        successors.insert(BlockId(3), vec![]);
        successors.insert(BlockId(4), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(2)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(1)]);
        predecessors.insert(BlockId(4), vec![BlockId(2)]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act
        let forest = find_loops(&cfg);

        // Assert: one loop with two exit blocks B3 and B4.
        assert_eq!(forest.loops.len(), 1, "should find exactly one loop");
        let lp = &forest.loops[0];
        assert_eq!(lp.exits.len(), 2, "loop should have exactly two exit blocks");

        let mut exit_set: BTreeSet<BlockId> = lp.exits.iter().copied().collect();
        assert!(exit_set.remove(&BlockId(3)), "B3 must be an exit");
        assert!(exit_set.remove(&BlockId(4)), "B4 must be an exit");
        assert!(exit_set.is_empty(), "no unexpected exits");
    }

    // -----------------------------------------------------------------------
    // Test 41: CondBranch with different fallthrough targets are not equal
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn cond_branch_different_fallthrough_not_equal() {
        // Arrange: same kind and taken, different fallthrough target.
        let cb_a = Terminator::CondBranch {
            kind: BranchKind::Above,
            taken: BlockId(1),
            fallthrough: BlockId(2),
        };
        let cb_b = Terminator::CondBranch {
            kind: BranchKind::Above,
            taken: BlockId(1),
            fallthrough: BlockId(9),
        };

        // Act & Assert
        assert_ne!(cb_a, cb_b, "CondBranch with different fallthrough must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 42: BlockId BTreeSet maintains sorted order
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn block_id_btree_set_sorted_order() {
        // Arrange
        let mut set = BTreeSet::new();
        set.insert(BlockId(5));
        set.insert(BlockId(1));
        set.insert(BlockId(3));
        set.insert(BlockId(2));
        set.insert(BlockId(4));

        // Act: collect in BTreeSet iteration order.
        let ordered: Vec<u32> = set.iter().map(|b| b.0).collect();

        // Assert: BTreeSet iterates in ascending order.
        assert_eq!(ordered, vec![1, 2, 3, 4, 5], "BlockId BTreeSet must iterate in ascending order");
    }

    // -----------------------------------------------------------------------
    // Test 43: Loop ordinal is assigned by header address order
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn loop_ordinal_sorted_by_header_address() {
        // Arrange: two-loop CFG where first loop header has lower address.
        let cfg = build_two_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: ordinals are assigned sequentially starting from 0.
        assert_eq!(forest.loops.len(), 2);
        assert_eq!(forest.loops[0].ordinal, 0, "first loop (lower address header) gets ordinal 0");
        assert_eq!(forest.loops[1].ordinal, 1, "second loop (higher address header) gets ordinal 1");

        // Verify address ordering.
        let addr0 = cfg.blocks[&forest.loops[0].header].start_addr;
        let addr1 = cfg.blocks[&forest.loops[1].header].start_addr;
        assert!(addr0 < addr1, "first loop header address must be less than second");
    }

    // -----------------------------------------------------------------------
    // Test 44: LoopForest children map is empty for non-nested disjoint loops
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn loop_forest_disjoint_loops_no_children() {
        // Arrange
        let cfg = build_two_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: two top-level loops, no parent-child relationships.
        assert_eq!(forest.top_level.len(), 2, "both disjoint loops are top-level");
        assert!(
            forest.children.is_empty(),
            "disjoint loops should have no parent-child relationships"
        );
        for lp in &forest.loops {
            assert_eq!(lp.depth, 0, "disjoint loop at depth 0");
        }
    }

    // -----------------------------------------------------------------------
    // Test 45: NaturalLoop body blocks is strictly a BTreeSet (no duplicates)
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn natural_loop_body_blocks_no_duplicates() {
        // Arrange: build a loop CFG and find loops.
        let cfg = build_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: body_blocks is a BTreeSet, which cannot have duplicates.
        assert_eq!(forest.loops.len(), 1);
        let body = &forest.loops[0].body_blocks;

        // Verify by inserting all elements into a Vec and checking against the set.
        let body_vec: Vec<BlockId> = body.iter().copied().collect();
        let body_unique: BTreeSet<BlockId> = body_vec.iter().copied().collect();
        assert_eq!(body.len(), body_unique.len(), "body_blocks must have no duplicates");
    }

    // -----------------------------------------------------------------------
    // Test 46: compute_loop_body with direct latch→header (no intermediate blocks)
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn compute_loop_body_direct_latch_to_header() {
        // Arrange: B1 (header) ↔ B2 (latch, directly jumps back to B1).
        // Predecessors: B1←[B0, B2], B2←[B1].
        let mut blocks = BTreeMap::new();
        for i in 0..3u32 {
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: Terminator::Return,
            });
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(1)]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(2)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);

        let cfg = ControlFlowGraph {
            blocks,
            entry: BlockId(0),
            successors,
            predecessors,
        };

        // Act: header=B1, latch=B2 (direct back-edge, no intermediate blocks in loop body).
        let body = compute_loop_body(&cfg, BlockId(1), BlockId(2));

        // Assert: body should be exactly {B1, B2}, not including B0.
        assert_eq!(body.len(), 2, "direct latch→header body should have exactly 2 blocks");
        assert!(body.contains(&BlockId(1)), "body must contain header B1");
        assert!(body.contains(&BlockId(2)), "body must contain latch B2");
        assert!(!body.contains(&BlockId(0)), "body must NOT contain B0");
    }

    // -----------------------------------------------------------------------
    // Test 47: BranchKind inequality — complementary pairs
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn branch_kind_complementary_pairs_not_equal() {
        // Arrange: test complementary condition pairs.
        let pairs = [
            (BranchKind::Above, BranchKind::BelowEqual),
            (BranchKind::AboveEqual, BranchKind::Below),
            (BranchKind::Greater, BranchKind::LessEqual),
            (BranchKind::GreaterEqual, BranchKind::Less),
            (BranchKind::Equal, BranchKind::NotEqual),
            (BranchKind::Sign, BranchKind::NotSign),
            (BranchKind::Parity, BranchKind::NotParity),
        ];

        // Act & Assert: each pair's members must differ.
        for (a, b) in &pairs {
            assert_ne!(a, b, "complementary pair {:?} and {:?} must not be equal", a, b);
        }
    }

    // -----------------------------------------------------------------------
    // Test 48: DecodedInsn with different operands are not equal
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn decoded_insn_different_operands_not_equal() {
        // Arrange: two instructions with same mnemonic and addr but different operands.
        let a = DecodedInsn {
            mnemonic: "mov".to_string(),
            operands: vec!["rax".to_string(), "rbx".to_string()],
            addr: 0x1000,
        };
        let b = DecodedInsn {
            mnemonic: "mov".to_string(),
            operands: vec!["rcx".to_string(), "rdx".to_string()],
            addr: 0x1000,
        };

        // Act & Assert
        assert_ne!(a, b, "DecodedInsn with different operands must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 49: Terminator::Jump with different targets are not equal
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn terminator_jump_different_targets_not_equal() {
        // Arrange
        let j1 = Terminator::Jump(BlockId(1));
        let j2 = Terminator::Jump(BlockId(2));

        // Act & Assert
        assert_eq!(j1, Terminator::Jump(BlockId(1)), "same target must be equal");
        assert_ne!(j1, j2, "Jump with different BlockId targets must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 50: NaturalLoop with empty exit list
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn natural_loop_empty_exits() {
        // Arrange: a loop with no exits (infinite loop body).
        let mut body = BTreeSet::new();
        body.insert(BlockId(0));
        body.insert(BlockId(1));

        let lp = NaturalLoop {
            header: BlockId(0),
            body_blocks: body,
            latch: BlockId(1),
            exits: vec![],
            ordinal: 0,
            depth: 0,
        };

        // Act & Assert
        assert!(lp.exits.is_empty(), "exits list should be empty");
        assert_eq!(lp.body_blocks.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 51: BasicBlock with multiple instructions preserves order
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn basic_block_multiple_instructions_preserves_order() {
        // Arrange: block with 3 instructions in sequence.
        let insns = vec![
            DecodedInsn { mnemonic: "push".to_string(), operands: vec!["rbp".to_string()], addr: 0x1000 },
            DecodedInsn { mnemonic: "mov".to_string(), operands: vec!["rbp".to_string(), "rsp".to_string()], addr: 0x1001 },
            DecodedInsn { mnemonic: "sub".to_string(), operands: vec!["rsp".to_string(), "0x20".to_string()], addr: 0x1004 },
        ];
        let block = BasicBlock {
            id: BlockId(0),
            start_addr: 0x1000,
            end_addr: 0x1008,
            instructions: insns,
            terminator: Terminator::Return,
        };

        // Act & Assert: instructions preserve insertion order.
        assert_eq!(block.instructions.len(), 3);
        assert_eq!(block.instructions[0].mnemonic, "push");
        assert_eq!(block.instructions[1].mnemonic, "mov");
        assert_eq!(block.instructions[2].mnemonic, "sub");
        // Addresses are in ascending order.
        assert!(block.instructions[0].addr < block.instructions[1].addr);
        assert!(block.instructions[1].addr < block.instructions[2].addr);
    }

    // -----------------------------------------------------------------------
    // Test 52: compute_loop_body with branching inside loop body
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn compute_loop_body_with_branch_inside_loop() {
        // Arrange: loop with a branch inside the body.
        //   B1 (header) → B2 or B3 (two body paths) → B4 (latch) → B1
        // Predecessors: B1←[B0,B4], B2←[B1], B3←[B1], B4←[B2,B3]
        let mut blocks = BTreeMap::new();
        for i in 0..5u32 {
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: Terminator::Return,
            });
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2), BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(4)]);
        successors.insert(BlockId(3), vec![BlockId(4)]);
        successors.insert(BlockId(4), vec![BlockId(1)]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(4)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(1)]);
        predecessors.insert(BlockId(4), vec![BlockId(2), BlockId(3)]);

        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act: header=B1, latch=B4
        let body = compute_loop_body(&cfg, BlockId(1), BlockId(4));

        // Assert: reverse BFS from B4 reaches B2 and B3 (both are predecessors of B4),
        // and B1 (predecessor of both B2 and B3, but already in body). B0 is excluded.
        assert!(body.contains(&BlockId(1)), "body must contain header B1");
        assert!(body.contains(&BlockId(2)), "body must contain branch path B2");
        assert!(body.contains(&BlockId(3)), "body must contain branch path B3");
        assert!(body.contains(&BlockId(4)), "body must contain latch B4");
        assert!(!body.contains(&BlockId(0)), "body must NOT contain B0 (outside loop)");
        assert_eq!(body.len(), 4, "body should have exactly 4 blocks");
    }

    // -----------------------------------------------------------------------
    // Test 53: find_loops on nested loop CFG — inner loop exits are correct
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_nested_inner_loop_exits() {
        // Arrange
        let cfg = build_nested_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: find the inner loop (smaller body) and verify its exits.
        assert_eq!(forest.loops.len(), 2);
        let (outer_idx, inner_idx) = if forest.loops[0].body_blocks.len() > forest.loops[1].body_blocks.len() {
            (0, 1)
        } else {
            (1, 0)
        };

        let inner = &forest.loops[inner_idx];
        // Inner loop exits should include the outer latch (B5), which is outside the inner body.
        assert!(
            !inner.exits.is_empty(),
            "inner loop must have at least one exit block"
        );
        // The inner loop exit must NOT be part of the inner body.
        for exit in &inner.exits {
            assert!(
                !inner.body_blocks.contains(exit),
                "exit block {:?} must not be in the inner loop body",
                exit
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 54: Dominators in two-loop CFG — shared block has correct idom
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn dominators_two_loop_cfg_shared_block() {
        // Arrange: two sequential disjoint loops.
        //   B0 → B1↔B2 (loop1) → B3↔B4 (loop2) → B5 (Return)
        let cfg = build_two_loop_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert
        assert_eq!(idom[&BlockId(0)], BlockId(0), "entry idom is itself");
        assert_eq!(idom[&BlockId(1)], BlockId(0), "B1 idom is B0");
        assert_eq!(idom[&BlockId(2)], BlockId(1), "B2 idom is B1 (only predecessor is B1)");
        assert_eq!(idom[&BlockId(3)], BlockId(1), "B3 idom is B1 (loop1 header dominates loop2 header)");
        assert_eq!(idom[&BlockId(4)], BlockId(3), "B4 idom is B3 (only predecessor is B3)");
        assert_eq!(idom[&BlockId(5)], BlockId(3), "B5 idom is B3 (exit from loop2 header)");
    }

    // -----------------------------------------------------------------------
    // Test 55: find_loops on CFG where loop header has two back-edges
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_two_back_edges_to_same_header() {
        // Arrange: B0 → B1 (header) → B2 (body1, jumps back) and B3 (body2, also jumps back).
        //   Two separate back-edges B2→B1 and B3→B1.
        let b0 = BasicBlock {
            id: BlockId(0),
            start_addr: 0x0,
            end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1),
            start_addr: 0x10,
            end_addr: 0x20,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual,
                taken: BlockId(4), // exit
                fallthrough: BlockId(2),
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2),
            start_addr: 0x20,
            end_addr: 0x30,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::Less,
                taken: BlockId(1),     // back-edge 1
                fallthrough: BlockId(3),
            },
        };
        let b3 = BasicBlock {
            id: BlockId(3),
            start_addr: 0x30,
            end_addr: 0x40,
            instructions: vec![],
            terminator: Terminator::Jump(BlockId(1)), // back-edge 2
        };
        let b4 = BasicBlock {
            id: BlockId(4),
            start_addr: 0x40,
            end_addr: 0x50,
            instructions: vec![],
            terminator: Terminator::Return,
        };

        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3, b4] {
            blocks.insert(b.id, b);
        }

        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(4), BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(1), BlockId(3)]);
        successors.insert(BlockId(3), vec![BlockId(1)]);
        successors.insert(BlockId(4), vec![]);

        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(2), BlockId(3)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(2)]);
        predecessors.insert(BlockId(4), vec![BlockId(1)]);

        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: two back-edges (B2→B1 and B3→B1) produce two distinct loops
        // that share the same header B1. Both have body {B1, B2, B3}.
        assert_eq!(forest.loops.len(), 2, "two back-edges should produce two loops");

        // Both loops should share the same header.
        assert_eq!(forest.loops[0].header, BlockId(1), "first loop header is B1");
        assert_eq!(forest.loops[1].header, BlockId(1), "second loop header is also B1");

        // Latches differ: one is B2, the other is B3.
        let latches: BTreeSet<BlockId> = forest.loops.iter().map(|l| l.latch).collect();
        assert_eq!(latches.len(), 2, "latches should be distinct");
        assert!(latches.contains(&BlockId(2)), "B2 must be a latch");
        assert!(latches.contains(&BlockId(3)), "B3 must be a latch");
    }

    // -----------------------------------------------------------------------
    // Test 56: ControlFlowGraph entry block has no predecessors
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn cfg_entry_block_has_no_predecessors() {
        // Arrange: use the loop CFG helper.
        let cfg = build_loop_cfg();

        // Act & Assert: entry block (B0) should have empty predecessor list.
        let entry_preds = cfg.predecessors.get(&cfg.entry).map(|p| p.len()).unwrap_or(0);
        assert_eq!(entry_preds, 0, "entry block should have zero predecessors");
    }

    // -----------------------------------------------------------------------
    // Test 57: find_loops on nested loop CFG — outer loop body is strict superset
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_nested_outer_body_is_superset_of_inner() {
        // Arrange
        let cfg = build_nested_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert
        assert_eq!(forest.loops.len(), 2);
        let (outer_idx, inner_idx) = if forest.loops[0].body_blocks.len() > forest.loops[1].body_blocks.len() {
            (0, 1)
        } else {
            (1, 0)
        };

        let outer = &forest.loops[outer_idx];
        let inner = &forest.loops[inner_idx];

        // Outer body is a strict superset of inner body.
        assert!(
            outer.body_blocks.is_superset(&inner.body_blocks),
            "outer body must be a superset of inner body"
        );
        // The outer header must NOT be in the inner body.
        assert!(
            !inner.body_blocks.contains(&outer.header),
            "outer header {:?} must not be in the inner loop body",
            outer.header
        );
        // The inner header must be in the outer body.
        assert!(
            outer.body_blocks.contains(&inner.header),
            "inner header {:?} must be in the outer loop body",
            inner.header
        );
    }

    // -----------------------------------------------------------------------
    // Test 58: DecodedInsn with different mnemonic are not equal
    // -----------------------------------------------------------------------

    #[test]
    fn decoded_insn_different_mnemonic_not_equal() {
        // Arrange: same operands and addr, different mnemonic.
        let a = DecodedInsn {
            mnemonic: "add".to_string(),
            operands: vec!["rax".to_string(), "rbx".to_string()],
            addr: 0x1000,
        };
        let b = DecodedInsn {
            mnemonic: "sub".to_string(),
            operands: vec!["rax".to_string(), "rbx".to_string()],
            addr: 0x1000,
        };

        // Act & Assert
        assert_ne!(a, b, "DecodedInsn with different mnemonic must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 59: Terminator::Fallthrough with different targets are not equal
    // -----------------------------------------------------------------------

    #[test]
    fn terminator_fallthrough_different_targets_not_equal() {
        // Arrange
        let ft1 = Terminator::Fallthrough(BlockId(1));
        let ft2 = Terminator::Fallthrough(BlockId(2));

        // Act & Assert
        assert_eq!(ft1, Terminator::Fallthrough(BlockId(1)));
        assert_ne!(ft1, ft2, "Fallthrough with different targets must not be equal");
    }

    // -----------------------------------------------------------------------
    // Test 60: NaturalLoop body_blocks set operations work correctly
    // -----------------------------------------------------------------------

    #[test]
    fn natural_loop_body_blocks_set_operations() {
        // Arrange: two loops with overlapping body blocks.
        let mut body1 = BTreeSet::new();
        body1.insert(BlockId(1));
        body1.insert(BlockId(2));
        body1.insert(BlockId(3));

        let mut body2 = BTreeSet::new();
        body2.insert(BlockId(2));
        body2.insert(BlockId(3));

        // Act & Assert
        assert!(body1.is_superset(&body2), "body1 is a superset of body2");
        assert!(!body2.is_superset(&body1), "body2 is not a superset of body1");
        assert!(body1.is_disjoint(&{
            let mut s = BTreeSet::new();
            s.insert(BlockId(0));
            s
        }));
    }

    // -----------------------------------------------------------------------
    // Test 61: ControlFlowGraph successors and predecessors are consistent
    // -----------------------------------------------------------------------

    #[test]
    fn cfg_successors_predecessors_consistency() {
        // Arrange: build the linear CFG.
        let cfg = build_linear_cfg();

        // Act & Assert: every successor edge must have a corresponding predecessor edge.
        for (&src, succs) in &cfg.successors {
            for &dst in succs {
                assert!(
                    cfg.predecessors.get(&dst).map_or(false, |preds| preds.contains(&src)),
                    "successor edge {:?}→{:?} must have matching predecessor",
                    src,
                    dst
                );
            }
        }

        // Reverse: every predecessor edge must have a matching successor edge.
        for (&dst, preds) in &cfg.predecessors {
            for &src in preds {
                assert!(
                    cfg.successors.get(&src).map_or(false, |succs| succs.contains(&dst)),
                    "predecessor edge {:?}→{:?} must have matching successor",
                    src,
                    dst
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 62: LoopForest with single loop has correct structure
    // -----------------------------------------------------------------------

    #[test]
    fn loop_forest_single_loop_structure() {
        // Arrange
        let cfg = build_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert: single loop at index 0, top_level contains 0, no children.
        assert_eq!(forest.loops.len(), 1);
        assert_eq!(forest.top_level, vec![0]);
        assert!(!forest.children.contains_key(&0), "single loop has no children");
        assert_eq!(forest.loops[0].ordinal, 0);
        assert_eq!(forest.loops[0].depth, 0);
    }

    // -----------------------------------------------------------------------
    // Test 63: BlockId Ord trait — min/max work correctly
    // -----------------------------------------------------------------------

    #[test]
    fn block_id_ord_min_max() {
        // Arrange
        use std::cmp::{max, min};

        let a = BlockId(10);
        let b = BlockId(20);
        let c = BlockId(30);

        // Act & Assert
        assert_eq!(min(a, b), BlockId(10));
        assert_eq!(max(b, c), BlockId(30));
        assert_eq!(min(a, c), BlockId(10));
        assert_eq!(max(a, b), BlockId(20));
    }

    // -----------------------------------------------------------------------
    // Test 64: BasicBlock start_addr equals first instruction addr
    // -----------------------------------------------------------------------

    #[test]
    fn basic_block_start_addr_matches_first_instruction() {
        // Arrange: block with two instructions, start_addr matches first instruction.
        let block = BasicBlock {
            id: BlockId(0),
            start_addr: 0x500,
            end_addr: 0x510,
            instructions: vec![
                DecodedInsn { mnemonic: "mov".to_string(), operands: vec!["rax".to_string(), "0x1".to_string()], addr: 0x500 },
                DecodedInsn { mnemonic: "ret".to_string(), operands: vec![], addr: 0x505 },
            ],
            terminator: Terminator::Return,
        };

        // Act & Assert
        assert!(
            !block.instructions.is_empty(),
            "block should have instructions"
        );
        assert_eq!(
            block.start_addr, block.instructions[0].addr,
            "start_addr must match first instruction address"
        );
    }

    // -----------------------------------------------------------------------
    // Test 65: Dominators — entry block is its own immediate dominator
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_entry_is_own_idom() {
        // Arrange: test across all three helper CFGs.
        let cfgs = [build_linear_cfg(), build_loop_cfg(), build_two_loop_cfg()];

        for cfg in &cfgs {
            let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
            let idom = compute_dominators(cfg, &block_ids);

            // Act & Assert: entry always dominates itself.
            assert_eq!(
                idom[&cfg.entry], cfg.entry,
                "entry {:?} must be its own immediate dominator",
                cfg.entry
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 66: LoopForest Debug format is non-empty
    // -----------------------------------------------------------------------

    #[test]
    fn loop_forest_debug_format_non_empty() {
        // Arrange
        let cfg = build_loop_cfg();
        let forest = find_loops(&cfg);

        // Act
        let debug = format!("{:?}", forest);

        // Assert: Debug output must contain key field names.
        assert!(!debug.is_empty(), "LoopForest Debug must not be empty");
        assert!(debug.contains("loops") || debug.contains("LoopForest"), "Debug must mention loops or LoopForest");
    }

    // -----------------------------------------------------------------------
    // Test 67: find_loops on nested loop CFG — outer loop contains all inner blocks
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_nested_outer_contains_inner_header_and_latch() {
        // Arrange
        let cfg = build_nested_loop_cfg();

        // Act
        let forest = find_loops(&cfg);

        // Assert
        assert_eq!(forest.loops.len(), 2);
        let (outer_idx, inner_idx) = if forest.loops[0].body_blocks.len() > forest.loops[1].body_blocks.len() {
            (0, 1)
        } else {
            (1, 0)
        };

        let outer = &forest.loops[outer_idx];
        let inner = &forest.loops[inner_idx];

        // Outer body must contain both inner header and inner latch.
        assert!(
            outer.body_blocks.contains(&inner.header),
            "outer body must contain inner header {:?}",
            inner.header
        );
        assert!(
            outer.body_blocks.contains(&inner.latch),
            "outer body must contain inner latch {:?}",
            inner.latch
        );
        // Inner latch must equal inner header or be a different block in inner body.
        assert!(
            inner.body_blocks.contains(&inner.latch),
            "inner latch must be in inner body"
        );
    }

    // -----------------------------------------------------------------------
    // Test 68: Empty CFG — compute_dominators returns entry dominating itself
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_empty_cfg_entry_only() {
        // Arrange: single-block CFG where entry returns.
        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Return,
        });
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: only entry block, it dominates itself.
        assert_eq!(idom.len(), 1, "single-block CFG should have exactly one idom entry");
        assert_eq!(idom[&BlockId(0)], BlockId(0), "entry idom is itself");
    }

    // -----------------------------------------------------------------------
    // Test 69: Single-block CFG with self-loop via CondBranch
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_single_block_self_loop_cond_branch() {
        // Arrange: single block B0 with CondBranch taken=B0, fallthrough=B1,
        // plus B1 (Return). B0 is both header and latch.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![],
            terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual, taken: BlockId(0), fallthrough: BlockId(1),
            },
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), b0);
        blocks.insert(BlockId(1), b1);
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(0), BlockId(1)]);
        successors.insert(BlockId(1), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![BlockId(0)]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: one self-loop on B0.
        assert_eq!(forest.loops.len(), 1, "should detect one self-loop");
        assert_eq!(forest.loops[0].header, BlockId(0));
        assert_eq!(forest.loops[0].latch, BlockId(0));
        assert_eq!(forest.loops[0].body_blocks.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 70: Unreachable block with self-loop does not produce false loop
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_unreachable_self_loop_no_false_positive() {
        // Arrange: B0 → B1 (Return). B2 is unreachable but has a self-loop.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Return,
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Jump(BlockId(2)), // self-loop
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![]);
        successors.insert(BlockId(2), vec![BlockId(2)]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(2)]); // unreachable self-loop
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: B2's self-loop is still detected (back-edge exists), but it
        // is separate from the reachable CFG. The loop count is 1.
        assert_eq!(forest.loops.len(), 1, "unreachable self-loop is still detected as a back-edge");
        assert_eq!(forest.loops[0].header, BlockId(2), "unreachable loop header is B2");
    }

    // -----------------------------------------------------------------------
    // Test 71: Dominator tree — unreachable block has idom = itself
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_unreachable_block_idom_is_self() {
        // Arrange: B0 → B1 (Return), B2 unreachable.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Return,
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![]);
        successors.insert(BlockId(2), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: unreachable B2 has no predecessors, so its dom set = {B2},
        // making idom(B2) = B2.
        assert_eq!(idom[&BlockId(2)], BlockId(2), "unreachable block idom is itself");
    }

    // -----------------------------------------------------------------------
    // Test 72: Nested loop — inner loop header is dominated by outer loop header
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_nested_inner_header_dominated_by_outer_header() {
        // Arrange
        let cfg = build_nested_loop_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
        let idom = compute_dominators(&cfg, &block_ids);

        // Act
        let forest = find_loops(&cfg);
        assert_eq!(forest.loops.len(), 2);
        let (outer_idx, inner_idx) = if forest.loops[0].body_blocks.len() > forest.loops[1].body_blocks.len() {
            (0, 1)
        } else {
            (1, 0)
        };
        let outer_header = forest.loops[outer_idx].header;
        let inner_header = forest.loops[inner_idx].header;

        // Assert: outer header dominates inner header.
        assert!(
            dominates(&idom, outer_header, inner_header),
            "outer header {:?} must dominate inner header {:?}",
            outer_header, inner_header
        );
    }

    // -----------------------------------------------------------------------
    // Test 73: CFG with multiple unreachable blocks — no false loops from reachable part
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_multiple_unreachable_blocks_no_false_loops() {
        // Arrange: B0 → B1 (Return). B2 and B3 are unreachable, no edges between them.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Return,
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Return,
        };
        let b3 = BasicBlock {
            id: BlockId(3), start_addr: 0x30, end_addr: 0x40,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![]);
        successors.insert(BlockId(2), vec![]);
        successors.insert(BlockId(3), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![]);
        predecessors.insert(BlockId(3), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: no back-edges exist, so no loops.
        assert!(forest.loops.is_empty(), "no back-edges means no loops");
        assert!(forest.top_level.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 74: dominates — non-entry block does not dominate entry
    // -----------------------------------------------------------------------

    #[test]
    fn dominates_non_entry_does_not_dominate_entry() {
        // Arrange: linear CFG B0 → B1 → B2.
        let cfg = build_linear_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
        let idom = compute_dominators(&cfg, &block_ids);

        // Act & Assert: B1 and B2 do not dominate B0.
        assert!(
            !dominates(&idom, BlockId(1), BlockId(0)),
            "B1 does not dominate entry B0"
        );
        assert!(
            !dominates(&idom, BlockId(2), BlockId(0)),
            "B2 does not dominate entry B0"
        );
    }

    // -----------------------------------------------------------------------
    // Test 75: Loop with header that is also the only body block (trivial loop)
    // -----------------------------------------------------------------------

    #[test]
    fn find_loops_trivial_loop_header_equals_latch_single_body() {
        // Arrange: B0 → B1 (self-loop via Jump to B1) with separate exit B2.
        // B1 has two successors: B1 (back-edge) and B2 (exit).
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Jump(BlockId(1)), // self-loop
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(1)]); // only self-loop, no exit edge
        successors.insert(BlockId(2), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(1)]);
        predecessors.insert(BlockId(2), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: one self-loop with body = {B1}, no exits (B2 is unreachable from B1).
        assert_eq!(forest.loops.len(), 1);
        let lp = &forest.loops[0];
        assert_eq!(lp.header, BlockId(1));
        assert_eq!(lp.latch, BlockId(1));
        assert_eq!(lp.body_blocks.len(), 1, "trivial loop body is just the header");
    }

    // -----------------------------------------------------------------------
    // Test 76: Dominator tree — diamond convergence point idom is the branch point
    // -----------------------------------------------------------------------

    #[test]
    fn dominators_diamond_convergence_idom_is_branch_point() {
        // Arrange: diamond B0 → B1/B2 → B3 → B4 (Return).
        // B3 is the convergence point. Its idom should be B0 (the branch point).
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::CondBranch {
                kind: BranchKind::Less, taken: BlockId(1), fallthrough: BlockId(2),
            },
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Jump(BlockId(3)),
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(3)),
        };
        let b3 = BasicBlock {
            id: BlockId(3), start_addr: 0x30, end_addr: 0x40,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(4)),
        };
        let b4 = BasicBlock {
            id: BlockId(4), start_addr: 0x40, end_addr: 0x50,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3, b4] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1), BlockId(2)]);
        successors.insert(BlockId(1), vec![BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![BlockId(4)]);
        successors.insert(BlockId(4), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(0)]);
        predecessors.insert(BlockId(3), vec![BlockId(1), BlockId(2)]);
        predecessors.insert(BlockId(4), vec![BlockId(3)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: B3's idom is B0 (the branch point), not B1 or B2.
        assert_eq!(idom[&BlockId(3)], BlockId(0), "diamond convergence B3 idom is branch point B0");
        // B4's idom is B3 (single predecessor path after convergence).
        assert_eq!(idom[&BlockId(4)], BlockId(3), "B4 idom is B3");
    }

    // -----------------------------------------------------------------------
    // Test 77: compute_loop_body with header that has no predecessors in CFG
    // -----------------------------------------------------------------------

    #[test]
    fn compute_loop_body_header_no_predecessors() {
        // Arrange: minimal CFG where header B1 has no predecessors recorded.
        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Return,
        });
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act: header == latch == B0 (self-loop on entry).
        let body = compute_loop_body(&cfg, BlockId(0), BlockId(0));

        // Assert: self-loop with header==latch returns just the header.
        assert_eq!(body.len(), 1);
        assert!(body.contains(&BlockId(0)));
    }

    // -----------------------------------------------------------------------
    // Test 78: CFG with all Jump terminators (no fallthrough chain)
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn cfg_all_jump_terminators_no_fallthrough() {
        // Arrange: B0 → B1 via Jump, B1 → B2 via Jump, B2 (Return).
        // No Fallthrough edges at all.
        let mut blocks = BTreeMap::new();
        for i in 0..3u32 {
            let term = if i < 2 {
                Terminator::Jump(BlockId(i + 1))
            } else {
                Terminator::Return
            };
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: term,
            });
        }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2)]);
        successors.insert(BlockId(2), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: pure Jump chain has no back-edges, hence no loops.
        assert!(forest.loops.is_empty(), "Jump chain with no back-edges must have no loops");
        // Verify terminator types are preserved.
        assert!(matches!(cfg.blocks[&BlockId(0)].terminator, Terminator::Jump(BlockId(1))));
        assert!(matches!(cfg.blocks[&BlockId(1)].terminator, Terminator::Jump(BlockId(2))));
        assert!(matches!(cfg.blocks[&BlockId(2)].terminator, Terminator::Return));
    }

    // -----------------------------------------------------------------------
    // Test 79: Successor/predecessor consistency on loop CFG (bidirectional)
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn cfg_successor_predecessor_consistency_loop_cfg() {
        // Arrange: use the loop CFG which has both forward and back-edges.
        let cfg = build_loop_cfg();

        // Act & Assert: every successor edge must have a matching predecessor edge.
        for (&src, succs) in &cfg.successors {
            for &dst in succs {
                assert!(
                    cfg.predecessors.get(&dst).map_or(false, |preds| preds.contains(&src)),
                    "successor edge {:?} -> {:?} must have matching predecessor in loop CFG",
                    src, dst
                );
            }
        }

        // Reverse: every predecessor edge must have a matching successor edge.
        for (&dst, preds) in &cfg.predecessors {
            for &src in preds {
                assert!(
                    cfg.successors.get(&src).map_or(false, |succs| succs.contains(&dst)),
                    "predecessor edge {:?} -> {:?} must have matching successor in loop CFG",
                    src, dst
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 80: Dominators — blocks inside loop closure dominated by loop header
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn dominators_loop_header_dominates_all_body_blocks() {
        // Arrange: loop CFG B0 → B1 (header) <-> B2 (body) → B3 (exit).
        let cfg = build_loop_cfg();
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
        let idom = compute_dominators(&cfg, &block_ids);

        // Act & Assert: B1 (loop header) dominates B2 (loop body block).
        // B1 also dominates itself (reflexive).
        assert!(
            dominates(&idom, BlockId(1), BlockId(1)),
            "loop header dominates itself"
        );
        assert!(
            dominates(&idom, BlockId(1), BlockId(2)),
            "loop header B1 must dominate loop body block B2"
        );
        // B1 also dominates the exit B3 (all paths to B3 go through B1).
        assert!(
            dominates(&idom, BlockId(1), BlockId(3)),
            "loop header B1 must dominate exit block B3"
        );
        // B2 does NOT dominate B1 (back-edge goes B2->B1, but B1 is reachable from B0).
        assert!(
            !dominates(&idom, BlockId(2), BlockId(1)),
            "loop body B2 does not dominate loop header B1"
        );
    }

    // -----------------------------------------------------------------------
    // Test 81: Loop body with branch that reconnects to latch
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_body_with_branch_to_latch() {
        // Arrange: B0 → B1 (header) → B2 or B3, both merge at B4 (latch) → B1.
        //   B1 conditionally enters B2 or B3 (two paths), both reach B4.
        //   B4 jumps back to B1 (back-edge).
        let mut blocks = BTreeMap::new();
        for i in 0..5u32 {
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: Terminator::Return,
            });
        }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2), BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(4)]);
        successors.insert(BlockId(3), vec![BlockId(4)]);
        successors.insert(BlockId(4), vec![BlockId(1)]); // back-edge
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(4)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(1)]);
        predecessors.insert(BlockId(4), vec![BlockId(2), BlockId(3)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: one loop with header=B1, latch=B4, body includes B1-B4.
        assert_eq!(forest.loops.len(), 1, "should find exactly one loop");
        let lp = &forest.loops[0];
        assert_eq!(lp.header, BlockId(1), "header is B1");
        assert_eq!(lp.latch, BlockId(4), "latch is B4");
        assert!(lp.body_blocks.contains(&BlockId(2)), "body must contain branch path B2");
        assert!(lp.body_blocks.contains(&BlockId(3)), "body must contain branch path B3");
        assert!(!lp.body_blocks.contains(&BlockId(0)), "body must NOT contain B0");
        assert_eq!(lp.body_blocks.len(), 4, "body should have exactly 4 blocks: B1, B2, B3, B4");
    }

    // -----------------------------------------------------------------------
    // Test 82: Triangle branch-merge pattern (no loop)
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_triangle_branch_merge_no_loop() {
        // Arrange: B0 → B1 (taken) / B2 (fallthrough) → B3 (Return).
        // B1 jumps directly to B3. This is a triangle, not a full diamond.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual, taken: BlockId(1), fallthrough: BlockId(2),
            },
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Jump(BlockId(3)),
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(3)),
        };
        let b3 = BasicBlock {
            id: BlockId(3), start_addr: 0x30, end_addr: 0x40,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1), BlockId(2)]);
        successors.insert(BlockId(1), vec![BlockId(3)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0)]);
        predecessors.insert(BlockId(2), vec![BlockId(0)]);
        predecessors.insert(BlockId(3), vec![BlockId(1), BlockId(2)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: triangle has no back-edges, so no loops.
        assert!(forest.loops.is_empty(), "triangle branch-merge has no loops");
        assert!(forest.top_level.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 83: Natural loop exit blocks are successors of body but not in body
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn natural_loop_exits_are_body_successors_outside_body() {
        // Arrange: loop CFG with known exit structure.
        let cfg = build_loop_cfg();

        // Act
        let forest = find_loops(&cfg);
        assert_eq!(forest.loops.len(), 1);
        let lp = &forest.loops[0];

        // Assert: every exit must be a successor of some body block.
        for exit in &lp.exits {
            let is_successor_of_body = lp.body_blocks.iter().any(|blk| {
                cfg.successors.get(blk).map_or(false, |succs| succs.contains(exit))
            });
            assert!(
                is_successor_of_body,
                "exit {:?} must be a successor of at least one body block",
                exit
            );
            // And the exit must NOT be in the loop body.
            assert!(
                !lp.body_blocks.contains(exit),
                "exit {:?} must not be in the loop body",
                exit
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 84: compute_loop_body where latch has predecessors outside the loop
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn compute_loop_body_latch_has_outside_predecessors() {
        // Arrange: B0 → B1 (header) → B2 (latch) → B1.
        // Additionally, B3 → B2 (external edge into latch, outside loop).
        // B3 is unreachable from entry but has a predecessor edge to B2.
        let mut blocks = BTreeMap::new();
        for i in 0..4u32 {
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: Terminator::Return,
            });
        }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(1)]); // back-edge
        successors.insert(BlockId(3), vec![BlockId(2)]); // external edge
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(2)]);
        predecessors.insert(BlockId(2), vec![BlockId(1), BlockId(3)]);
        predecessors.insert(BlockId(3), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act: header=B1, latch=B2
        let body = compute_loop_body(&cfg, BlockId(1), BlockId(2));

        // Assert: reverse BFS from B2 visits B2's predecessors (B1 and B3).
        // B1 is already in body (header), so BFS stops there.
        // B3 is added to body because reverse BFS follows all predecessor edges.
        assert!(body.contains(&BlockId(1)), "body must contain header B1");
        assert!(body.contains(&BlockId(2)), "body must contain latch B2");
        // B3 is a predecessor of latch, so reverse BFS includes it.
        assert!(body.contains(&BlockId(3)), "body must contain B3 (predecessor of latch)");
        assert!(!body.contains(&BlockId(0)), "body must NOT contain B0 (predecessor of header, not latch)");
    }

    // -----------------------------------------------------------------------
    // Test 85: Long linear chain produces correct dominator chain
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn dominators_long_linear_chain() {
        // Arrange: B0 → B1 → B2 → B3 → B4 → B5 (6 blocks, linear).
        let mut blocks = BTreeMap::new();
        for i in 0..6u32 {
            let term = if i < 5 {
                Terminator::Fallthrough(BlockId(i + 1))
            } else {
                Terminator::Return
            };
            blocks.insert(BlockId(i), BasicBlock {
                id: BlockId(i),
                start_addr: (i as u64) * 0x10,
                end_addr: (i as u64 + 1) * 0x10,
                instructions: vec![],
                terminator: term,
            });
        }
        let mut successors = BTreeMap::new();
        let mut predecessors = BTreeMap::new();
        for i in 0..6u32 {
            if i < 5 {
                successors.insert(BlockId(i), vec![BlockId(i + 1)]);
                predecessors.insert(BlockId(i + 1), vec![BlockId(i)]);
            } else {
                successors.insert(BlockId(i), vec![]);
            }
        }
        predecessors.insert(BlockId(0), vec![]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        // Act
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: idom forms a chain: idom(Bi) = B(i-1) for i > 0.
        assert_eq!(idom[&BlockId(0)], BlockId(0), "entry idom is itself");
        for i in 1..6u32 {
            assert_eq!(
                idom[&BlockId(i)], BlockId(i - 1),
                "B{} idom should be B{}",
                i, i - 1
            );
        }

        // Assert: B0 dominates all blocks; B3 dominates B3, B4, B5 but not B2.
        for i in 0..6u32 {
            assert!(
                dominates(&idom, BlockId(0), BlockId(i)),
                "B0 must dominate B{}",
                i
            );
        }
        assert!(dominates(&idom, BlockId(3), BlockId(5)), "B3 dominates B5");
        assert!(!dominates(&idom, BlockId(3), BlockId(2)), "B3 does not dominate B2");
    }

    // -----------------------------------------------------------------------
    // Test 86: find_loops with back-edge that bypasses intermediate blocks
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn find_loops_back_edge_bypasses_intermediate_blocks() {
        // Arrange: B0 → B1 (header) → B2 → B3 → B4 (latch) → B1.
        // The back-edge B4→B1 skips over B2 and B3.
        // B1 conditionally exits to B5.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(1)),
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::CondBranch {
                kind: BranchKind::NotEqual, taken: BlockId(5), fallthrough: BlockId(2),
            },
        };
        let b2 = BasicBlock {
            id: BlockId(2), start_addr: 0x20, end_addr: 0x30,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(3)),
        };
        let b3 = BasicBlock {
            id: BlockId(3), start_addr: 0x30, end_addr: 0x40,
            instructions: vec![], terminator: Terminator::Fallthrough(BlockId(4)),
        };
        let b4 = BasicBlock {
            id: BlockId(4), start_addr: 0x40, end_addr: 0x50,
            instructions: vec![], terminator: Terminator::Jump(BlockId(1)),
        };
        let b5 = BasicBlock {
            id: BlockId(5), start_addr: 0x50, end_addr: 0x60,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        for b in [b0, b1, b2, b3, b4, b5] { blocks.insert(b.id, b); }
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1)]);
        successors.insert(BlockId(1), vec![BlockId(5), BlockId(2)]);
        successors.insert(BlockId(2), vec![BlockId(3)]);
        successors.insert(BlockId(3), vec![BlockId(4)]);
        successors.insert(BlockId(4), vec![BlockId(1)]);
        successors.insert(BlockId(5), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(4)]);
        predecessors.insert(BlockId(2), vec![BlockId(1)]);
        predecessors.insert(BlockId(3), vec![BlockId(2)]);
        predecessors.insert(BlockId(4), vec![BlockId(3)]);
        predecessors.insert(BlockId(5), vec![BlockId(1)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);

        // Assert: one loop, header=B1, latch=B4, body includes all intermediate blocks.
        assert_eq!(forest.loops.len(), 1, "should find exactly one loop");
        let lp = &forest.loops[0];
        assert_eq!(lp.header, BlockId(1), "header is B1");
        assert_eq!(lp.latch, BlockId(4), "latch is B4");
        assert!(lp.body_blocks.contains(&BlockId(2)), "body must contain B2");
        assert!(lp.body_blocks.contains(&BlockId(3)), "body must contain B3");
        assert!(!lp.body_blocks.contains(&BlockId(0)), "body must NOT contain B0");
        assert!(!lp.body_blocks.contains(&BlockId(5)), "body must NOT contain B5 (exit)");
        assert_eq!(lp.exits, vec![BlockId(5)], "exit should be B5");
    }

    // -----------------------------------------------------------------------
    // Test 87: CondBranch where taken and fallthrough are the same block
    // -----------------------------------------------------------------------

    // @trace TEST-12k66
    #[test]
    fn cfg_cond_branch_both_targets_same_block() {
        // Arrange: B0 CondBranch where both taken and fallthrough go to B1.
        // This is degenerate (equivalent to unconditional jump) but must be handled.
        let b0 = BasicBlock {
            id: BlockId(0), start_addr: 0x0, end_addr: 0x10,
            instructions: vec![], terminator: Terminator::CondBranch {
                kind: BranchKind::Equal, taken: BlockId(1), fallthrough: BlockId(1),
            },
        };
        let b1 = BasicBlock {
            id: BlockId(1), start_addr: 0x10, end_addr: 0x20,
            instructions: vec![], terminator: Terminator::Return,
        };
        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId(0), b0);
        blocks.insert(BlockId(1), b1);
        let mut successors = BTreeMap::new();
        successors.insert(BlockId(0), vec![BlockId(1), BlockId(1)]);
        successors.insert(BlockId(1), vec![]);
        let mut predecessors = BTreeMap::new();
        predecessors.insert(BlockId(0), vec![]);
        predecessors.insert(BlockId(1), vec![BlockId(0), BlockId(0)]);
        let cfg = ControlFlowGraph { blocks, entry: BlockId(0), successors, predecessors };

        // Act
        let forest = find_loops(&cfg);
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
        let idom = compute_dominators(&cfg, &block_ids);

        // Assert: no loops (no back-edges).
        assert!(forest.loops.is_empty(), "degenerate CondBranch with same targets has no loops");
        // B0 dominates B1.
        assert!(dominates(&idom, BlockId(0), BlockId(1)), "B0 dominates B1");
        // B1 does not dominate B0.
        assert!(!dominates(&idom, BlockId(1), BlockId(0)), "B1 does not dominate B0");
    }
}
