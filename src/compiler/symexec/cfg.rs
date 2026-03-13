//! CFG construction and natural loop detection for symexec.
//!
//! Phase 1 of the control-flow upgrade: builds a control-flow graph from
//! a compiled `extern "C"` function's machine code, then identifies natural
//! loops via dominator-tree analysis.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
pub fn build_cfg_from_fn(fn_ptr: *const u8, max_bytes: usize) -> Result<ControlFlowGraph, String> {
    if fn_ptr.is_null() {
        return Err("null function pointer".into());
    }

    let base_addr = fn_ptr as u64;
    let bytes = unsafe { std::slice::from_raw_parts(fn_ptr, max_bytes) };
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    decoder.set_ip(base_addr);

    // Phase 1: Linear scan — collect all instructions and jump targets.
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

    // Phase 2: Determine block boundaries.
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

    // Phase 3: Build basic blocks.
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

#[cfg(not(target_arch = "x86_64"))]
pub fn build_cfg_from_fn(_fn_ptr: *const u8, _max_bytes: usize) -> Result<ControlFlowGraph, String> {
    Err("CFG construction only supported on x86_64".into())
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
                    let first = iter.next().unwrap();
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

        let cfg = build_cfg_from_fn(add_one as *const u8, 256).expect("CFG build failed");

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

        let cfg = build_cfg_from_fn(sum_array as *const u8, 512).expect("CFG build failed");

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

        let cfg = build_cfg_from_fn(identity as *const u8, 256).expect("CFG build failed");
        let forest = find_loops(&cfg);
        assert!(
            forest.loops.is_empty(),
            "identity function should have no loops"
        );
        assert!(forest.top_level.is_empty());
    }
}
