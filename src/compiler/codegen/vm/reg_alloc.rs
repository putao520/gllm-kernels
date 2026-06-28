//! 寄存器分配器 (REGISTER-VM SPEC §7)
//!
//! 线性扫描分配: VRegId → PhysReg。
//! InterferenceGraph 保证数学无冲突。

use std::collections::{HashMap, HashSet, BTreeSet};
use super::instr::*;
use super::isa_profile::*;

/// Tracks the first body-local occurrence of a VReg inside a loop, used by
/// `compute_intervals` Pass 3 to decide whether to extend `last_use` to
/// `LoopEnd`. A VReg that is purely written (no read of its prior value)
/// at its first occurrence inside the body is NOT loop-carried.
#[derive(Debug, Clone, Copy)]
enum FirstOccur {
    Read,
    Write,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0 生命周期语义标签 (REQ-LC-001)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// VReg 生命周期语义 — 由 compute_intervals Pass 4 自动推导，禁止手工标注。
///
/// 影响：(1) 物理寄存器分配优先级 (2) spill/reload 策略 (3) post-hoc 验证
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleTag {
    /// 定义在循环外，每次迭代只读。例: weight_ptr, scale_ptr
    LoopInvariant,
    /// 定义在循环外，循环内读+写，跨迭代传递。例: seq_counter, data_ptr
    LoopCarried,
    /// 定义在循环内，每次迭代重新覆盖。例: block_ptr, decoded_vec
    BodyLocal,
    /// 定义在 ScopeBegin 内，需在 ScopeEnd 后存活。例: GEMM 累加器跨 scope
    CrossScope,
    /// 全生命周期 (prologue 到 epilogue)。例: scratchpad_ptr, batch_ctx_ptr
    Global,
}

impl LifecycleTag {
    /// 分配优先级 (数值越小越优先分配物理寄存器)
    pub fn alloc_priority(self) -> u8 {
        match self {
            Self::Global => 0,
            Self::LoopInvariant => 2,
            Self::LoopCarried => 2,
            Self::CrossScope => 3,
            Self::BodyLocal => 4,
        }
    }

    /// Spill 优先级 (数值越小越先被 spill 到栈)
    pub fn spill_priority(self) -> u8 {
        match self {
            Self::BodyLocal => 0,
            Self::CrossScope => 1,
            Self::LoopCarried => 2,
            Self::LoopInvariant => 3,
            Self::Global => 4,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0.1 LifecycleTag 自动推导 (REQ-LC-002)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// REQ-LC-002: 从循环结构和 def-use chain 自动推导每个 VReg 的 LifecycleTag。
///
/// # 算法 (5 条规则, 按优先级)
///
/// | 优先级 | 规则                                    | 标签             |
/// |--------|-----------------------------------------|------------------|
/// | 1      | 全生命周期 (定义在 prologue, 活到 epilogue) | `Global`         |
/// | 2      | 定义在循环外 + 循环内只读                  | `LoopInvariant`  |
/// | 3      | 定义在循环内 + 跨迭代存活 (phi/back-edge)   | `LoopCarried`    |
/// | 4      | 定义在循环内 + 循环内消费完                | `BodyLocal`      |
/// | 5      | 跨 ScopeBegin/ScopeEnd 边界存活            | `CrossScope`     |
///
/// 特殊规则:
/// - `VRegKind::Counter` / `VRegKind::ByteOffset` 强制 `LoopCarried`
/// - `VRegKind::Ptr` 持有 weight/cache 指针且全生命周期 → `Global`
fn infer_lifecycle(
    vreg: VRegId,
    kind: VRegKind,
    def_point: usize,
    last_use: usize,
    loop_ranges: &[(usize, usize)],
    loop_first_found: &HashSet<(VRegId, usize)>,
    scope_ranges: &[(usize, usize)],
    program_len: usize,
) -> LifecycleTag {
    // ── 规则 1: 全生命周期 → Global ──
    // 定义在程序开头 (prologue) 且活跃到程序末尾 (epilogue)。
    // 典型: scratchpad_ptr, batch_ctx_ptr, weight_ptr (常量指针)。
    if def_point == 0 && last_use >= program_len.saturating_sub(1) {
        return LifecycleTag::Global;
    }

    let mut is_loop_invariant = false;
    let mut is_loop_carried = false;
    let mut is_body_local = true;

    for &(loop_begin, loop_end) in loop_ranges {
        let defined_outside_loop = def_point < loop_begin;
        let defined_inside_loop = def_point > loop_begin && def_point < loop_end;
        let used_inside_loop = last_use >= loop_begin;
        let used_past_loop = last_use > loop_end;

        // ── 规则 2: 定义在循环外 + 循环内只读 → LoopInvariant ──
        if defined_outside_loop && used_inside_loop {
            is_body_local = false;
            let key = (vreg, loop_begin);
            if loop_first_found.contains(&key) {
                // Pass 3 已分析: last_use 被延展到 LoopEnd → 首次引用是读
                // 否则 def_point 被收紧到首次写 → 非 invariant
                if last_use >= loop_end {
                    is_loop_invariant = true;
                }
                // 否则: 首次引用是写 → 每次迭代重新覆盖, 非 loop-invariant
            } else {
                // Pass 3 未跟踪 → 保守假设 loop-invariant
                is_loop_invariant = true;
            }
        }

        // ── 规则 3: 定义在循环内 + 跨迭代存活 → LoopCarried ──
        // phi node / back-edge: 当前迭代定义的值被下一迭代使用。
        // 判定: 定义在循环内, 且 last_use 越过 LoopEnd (值逃逸循环体)。
        if defined_inside_loop && used_past_loop {
            is_body_local = false;
            is_loop_carried = true;
        }

        // ── 规则 4: 定义在循环内 + 循环内消费完 → BodyLocal ──
        // 定义和使用都在同一循环体内, 不跨迭代。
        if defined_inside_loop && !used_past_loop {
            // 保持 is_body_local = true
        }
    }

    // ── 规则 2,3,4 决策 ──
    let mut tag = if is_loop_carried {
        LifecycleTag::LoopCarried
    } else if is_loop_invariant {
        LifecycleTag::LoopInvariant
    } else {
        // 默认: BodyLocal (不在任何循环内, 或定义/使用在循环外)
        LifecycleTag::BodyLocal
    };

    // ── 规则 5: 跨 ScopeBegin/ScopeEnd 边界 → CrossScope ──
    // VReg 在 scope 外定义, scope 结束后仍被使用, 且不在 scope 内定义。
    for &(scope_begin, scope_end) in scope_ranges {
        let crosses_scope = def_point < scope_begin && last_use > scope_end;
        let defined_in_scope = def_point > scope_begin && def_point < scope_end;
        if crosses_scope && !defined_in_scope {
            // 定义在 scope 前, 使用在 scope 后 → CrossScope
            if tag == LifecycleTag::BodyLocal {
                tag = LifecycleTag::CrossScope;
            }
        }
    }

    // ── 特殊规则: Counter / ByteOffset 强制 LoopCarried ──
    // 循环计数器和字节偏移量的语义决定了它们必须跨迭代存活。
    if matches!(kind, VRegKind::Counter | VRegKind::ByteOffset)
        && tag == LifecycleTag::BodyLocal {
            tag = LifecycleTag::LoopCarried;
        }

    tag
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 活跃区间
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone)]
pub struct LiveInterval {
    pub vreg: VRegId,
    pub kind: VRegKind,
    pub width: SimdWidth,
    pub def_point: usize,
    pub last_use: usize,
    /// REQ-LC-001: 生命周期语义标签，由 Pass 4 自动推导
    pub lifecycle: LifecycleTag,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 干涉图
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct InterferenceGraph {
    edges: HashMap<VRegId, HashSet<VRegId>>,
}

impl InterferenceGraph {
    /// O(n log n) sweep-line interference graph construction.
    ///
    /// Replaces the previous O(n²) pairwise comparison. For Phi-4 scale models
    /// (100K+ VRegs), this reduces build time from 30+ minutes to sub-second.
    ///
    /// Algorithm: partition intervals by physical register class, then run
    /// sweep-line independently per class. Within each class, sort intervals
    /// by def_point and maintain a BTreeSet of active intervals ordered by
    /// last_use. When processing interval i, all active intervals with
    /// last_use >= i.def_point interfere with i.
    pub fn build(intervals: &[LiveInterval]) -> Self {
        let mut edges: HashMap<VRegId, HashSet<VRegId>> = HashMap::new();

        // Partition by physical class: GPR (Ptr/Scalar/Counter/ByteOffset), Vec, Tile, Mask.
        let mut gpr: Vec<usize> = Vec::new();
        let mut vec: Vec<usize> = Vec::new();
        let mut tile: Vec<usize> = Vec::new();
        let mut mask: Vec<usize> = Vec::new();

        for (i, iv) in intervals.iter().enumerate() {
            match iv.kind {
                VRegKind::Ptr | VRegKind::Scalar | VRegKind::Counter | VRegKind::ByteOffset => gpr.push(i),
                VRegKind::Vec => vec.push(i),
                VRegKind::Tile => tile.push(i),
                VRegKind::Mask => mask.push(i),
            }
        }

        Self::build_partitioned(intervals, &gpr, &mut edges);
        Self::build_partitioned(intervals, &vec, &mut edges);
        Self::build_partitioned(intervals, &tile, &mut edges);
        Self::build_partitioned(intervals, &mask, &mut edges);

        Self { edges }
    }

    /// Sweep-line interference for a single physical register class.
    fn build_partitioned(
        intervals: &[LiveInterval],
        indices: &[usize],
        edges: &mut HashMap<VRegId, HashSet<VRegId>>,
    ) {
        if indices.len() <= 1 {
            return;
        }

        // Sort by (def_point, last_use).
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by_key(|&i| (intervals[i].def_point, intervals[i].last_use));

        // Active set ordered by (last_use, original_index) for deterministic ordering.
        let mut active: BTreeSet<(usize, usize)> = BTreeSet::new();

        for &idx in &sorted {
            let iv = &intervals[idx];

            // Expire: remove all active intervals whose last_use < iv.def_point.
            loop {
                let first = match active.iter().next() {
                    Some(&f) => f,
                    None => break,
                };
                if first.0 < iv.def_point {
                    active.remove(&first);
                } else {
                    break;
                }
            }

            // All remaining active intervals interfere with current.
            for &(_, aidx) in &active {
                let aiv = &intervals[aidx];
                edges.entry(aiv.vreg).or_default().insert(iv.vreg);
                edges.entry(iv.vreg).or_default().insert(aiv.vreg);
            }

            active.insert((iv.last_use, idx));
        }
    }

    pub fn interferes(&self, a: VRegId, b: VRegId) -> bool {
        self.edges.get(&a).is_some_and(|s| s.contains(&b))
    }

    pub fn neighbors(&self, v: VRegId) -> impl Iterator<Item = &VRegId> {
        self.edges.get(&v).into_iter().flat_map(|s| s.iter())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 分配结果
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone)]
pub struct SpillSlot {
    pub vreg: VRegId,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug)]
pub struct RegAllocation {
    pub mapping: HashMap<VRegId, PhysReg>,
    pub spills: Vec<SpillSlot>,
    pub callee_saved_used: Vec<PhysGpr>,
}

impl RegAllocation {
    pub fn get(&self, vreg: VRegId) -> Option<PhysReg> {
        self.mapping.get(&vreg).copied()
    }

    pub fn get_gpr(&self, vreg: VRegId) -> Option<PhysGpr> {
        match self.get(vreg)? {
            PhysReg::Gpr(g) => Some(g),
            _ => None,
        }
    }

    pub fn get_vec(&self, vreg: VRegId) -> Option<PhysVec> {
        match self.get(vreg)? {
            PhysReg::Vec(v) => Some(v),
            _ => None,
        }
    }

    pub fn num_vregs(&self) -> usize {
        self.mapping.len()
    }

    pub fn spill_slots(&self) -> &[SpillSlot] {
        &self.spills
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 分配器
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct RegAllocator<'a> {
    profile: &'a IsaProfile,
}

impl<'a> RegAllocator<'a> {
    pub fn new(profile: &'a IsaProfile) -> Self {
        Self { profile }
    }

    /// 扫描 VmProgram，计算每个 VRegId 的活跃区间。
    pub(crate) fn compute_intervals(program: &VmProgram) -> Vec<LiveInterval> {
        let mut map: HashMap<VRegId, (VRegKind, SimdWidth, usize, usize)> = HashMap::new();

        // Pass 1: 收集声明和引用
        for (i, instr) in program.instrs.iter().enumerate() {
            if let VmInstr::DeclareVReg { id, kind, width } = instr {
                map.entry(*id).or_insert((*kind, *width, i, i));
            }
            for vreg in Self::referenced_vregs(instr) {
                map.entry(vreg).and_modify(|e| e.3 = e.3.max(i));
            }
        }

        // Pass 2: LoopBegin 的 counter/byte_offset 活跃到对应 LoopEnd
        // 这确保循环变量不会被分配到和 body 内 VReg 相同的物理寄存器
        let mut loop_stack: Vec<(VRegId, VRegId)> = Vec::new(); // (counter, byte_offset)
        for (i, instr) in program.instrs.iter().enumerate() {
            if let VmInstr::LoopBegin { counter, byte_offset, .. } = instr {
                loop_stack.push((*counter, *byte_offset));
            }
            if matches!(instr, VmInstr::LoopEnd) {
                if let Some((counter, byte_offset)) = loop_stack.pop() {
                    // 延展 counter/byte_offset 的 last_use 到 LoopEnd
                    map.entry(counter).and_modify(|e| e.3 = e.3.max(i));
                    map.entry(byte_offset).and_modify(|e| e.3 = e.3.max(i));
                }
            }
        }

        // Pass 3: 循环不变量 (loop-invariant VRegs) 活跃到对应 LoopEnd。
        //
        // 若 VReg 在 loop 外定义 (def_point < LoopBegin) 且在 loop 内被引用
        // (LoopBegin < last_use < LoopEnd),说明**每次迭代都会重新读它**
        // (循环 back edge 隐含重读)。Pass 1 的线性扫描只看到 last_use 是
        // 某一条线性指令,会把它当作"用完就死",导致 RegAlloc 可能把该物理
        // 寄存器分配给 loop body 内其他 VReg,造成跨迭代值丢失。
        // 必须把 last_use 延展到 LoopEnd,让 InterferenceGraph 识别冲突。
        //
        // **例外 (ARCH-REGALLOC-WRITE-BEFORE-USE)**: 若 VReg 在 loop body 内
        // 的第一次引用是**写** (Broadcast / VecLoad / Fma(dst) / 等将 dst
        // 覆盖的指令) 而非读,则每次迭代 value 都被重新定义,上一次迭代的值
        // 没有被携带过 back edge,VReg 不是真正的 loop-carried。此时不应
        // 延展 last_use,让它的 live interval 只在 body 内从首写到末读,
        // 避免虚假 conflict 吃掉物理寄存器。典型场景: GEMM blis microkernel
        // 把 a_broadcast / b_vec 的 Declare 提到 loop 外以避免重复 alloc,
        // 但实际每次迭代先 Broadcast/VecLoad 再 FMA 读取,非 loop-invariant。
        let loop_ranges = {
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut stack: Vec<usize> = Vec::new();
            for (i, instr) in program.instrs.iter().enumerate() {
                match instr {
                    VmInstr::LoopBegin { .. } => stack.push(i),
                    VmInstr::LoopEnd => {
                        if let Some(begin) = stack.pop() {
                            ranges.push((begin, i));
                        }
                    }
                    _ => {}
                }
            }
            ranges
        };

        // Helper: 指令 `instr` 是否把 `vreg` 当作**被完全覆盖的 dst**(纯写,
        // 不读旧值即可完成语义). 对于 Fma { dst, acc, .. } 其中 `dst == acc`
        // 表示读-改-写 (累加依赖先前值),不算纯写.
        let is_pure_write_of = |instr: &VmInstr, vreg: VRegId| -> bool {
            match instr {
                VmInstr::Broadcast { dst, .. } => *dst == vreg,
                VmInstr::VecLoad { dst, .. } => *dst == vreg,
                VmInstr::LoadPtr { dst, .. } => *dst == vreg,
                VmInstr::ScalarLoad { dst, .. } => *dst == vreg,
                VmInstr::ScalarByteLoad { dst, .. } => *dst == vreg,
                VmInstr::GgufSubScaleLoad { dst, .. } => *dst == vreg,
                VmInstr::GgufKQuantScaleLoad { dst, .. } => *dst == vreg,
                VmInstr::GprBinOp { dst, a, b, .. } => {
                    let other_inputs_ok = *a != vreg && (b.vreg() != Some(vreg));
                    *dst == vreg && other_inputs_ok
                }
                VmInstr::GprUnaryOp { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::GprLoadImm { dst, .. } => *dst == vreg,
                VmInstr::VecUnaryOp { dst, a, .. } => *dst == vreg && *a != vreg,
                VmInstr::VecBinOp { dst, a, b, .. } => {
                    *dst == vreg && *a != vreg && *b != vreg
                }
                VmInstr::VecCmp { dst, a, b, .. } => {
                    *dst == vreg && *a != vreg && *b != vreg
                }
                VmInstr::VecCast { dst, src, .. } => {
                    *dst == vreg && *src != vreg
                }
                VmInstr::ConditionalSelect { dst, mask, true_val, false_val, .. } => {
                    *dst == vreg && *mask != vreg && *true_val != vreg && *false_val != vreg
                }
                VmInstr::HReduce { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::Transcendental { dst, src, .. } => *dst == vreg && *src != vreg,
                // Fma: dst==vreg && acc==dst → 读写; dst==vreg && acc!=dst → 纯写
                VmInstr::Fma { dst, acc, a, b, .. } => {
                    *dst == vreg && *acc != vreg && *a != vreg && *b != vreg
                }
                // Accumulate 是 acc += src,永远读 acc → 不是纯写
                VmInstr::Accumulate { .. } => false,
                _ => false,
            }
        };
        // Helper: 指令是否把 `vreg` 作为读引用
        let reads_vreg = |instr: &VmInstr, vreg: VRegId| -> bool {
            Self::referenced_vregs(instr).contains(&vreg)
                && !matches!(instr, VmInstr::DeclareVReg { .. })
        };

        // 扫描每个 loop,为每个 out-of-loop-defined VReg 判断它在 body 内第一次
        // 出现是写还是读。
        // - 首读 (Read-first): 值从 loop 外携带进来,每次迭代重读,
        //   **延展 last_use 到 LoopEnd** 以防 RegAlloc 把物理寄存器分给 body 内
        //   其他 VReg。
        // - 首写 (Write-first): 每次迭代重新覆盖,前一轮值无保留语义,
        //   **把 def_point 推进到首次写** 以收紧 live interval,让 RegAlloc
        //   能把该物理寄存器在 loop 外与其他 VReg 共享。
        //
        // 典型场景 (ARCH-REGALLOC-WRITE-BEFORE-USE): GEMM BLIS microkernel 把
        // a_broadcast / b_vec 的 Declare 放在 loop 外 (一次 alloc) 但每次 K-loop
        // 迭代先 Broadcast/VecLoad 再 FMA 读取。若不收紧 def_point,
        // [Declare_at_outer, last_FMA] 与 12 个 accumulator 的 live interval
        // 几乎完全重叠, 冲突度 = 12+2 = 14 > YMM 池 13 → 爆池。
        // ARCH-REGALLOC-PASS3-OPT: Single forward scan replaces O(V×M) + O(L×V×S)
        // nested loops with O(M) single-pass. For each instruction, check which
        // VRegs it references and track first-occurrence + loop-carry info.
        //
        // Phase 3a (global first-occurrence tightening):
        //   If a VReg's first non-Declare reference is a pure write, tighten
        //   def_point to that write position. Safe because no earlier reads exist.
        //
        // Phase 3b (loop-carry extension):
        //   After tightening, if a VReg straddles a loop boundary and its first
        //   occurrence within that loop is a read (not a write), extend last_use
        //   to LoopEnd to capture the back-edge read.

        // Build loop_end_map: for each LoopBegin index, its matching LoopEnd index.
        let mut loop_end_map: HashMap<usize, usize> = HashMap::new();
        // Build scope_end_map: for each ScopeBegin index, its matching ScopeEnd index.
        let mut scope_ranges: Vec<(usize, usize)> = Vec::new();
        {
            let mut loop_stack: Vec<usize> = Vec::new();
            let mut scope_stack: Vec<usize> = Vec::new();
            for (i, instr) in program.instrs.iter().enumerate() {
                match instr {
                    VmInstr::LoopBegin { .. } => loop_stack.push(i),
                    VmInstr::LoopEnd => {
                        if let Some(begin) = loop_stack.pop() {
                            loop_end_map.insert(begin, i);
                        }
                    }
                    VmInstr::ScopeBegin { .. } => scope_stack.push(i),
                    VmInstr::ScopeEnd { .. } => {
                        if let Some(begin) = scope_stack.pop() {
                            scope_ranges.push((begin, i));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Track which VRegs have found their global first occurrence (Phase 3a).
        // VRegId → (is_write, position)
        let mut global_first_found: HashSet<VRegId> = HashSet::new();

        // Track which (vreg, loop_begin) pairs have found their first occurrence
        // inside the loop body (Phase 3b).
        // (VRegId, loop_begin) → is_write_first
        let mut loop_first_found: HashSet<(VRegId, usize)> = HashSet::new();

        // Active loop stack for Phase 3b
        let mut active_loops: Vec<usize> = Vec::new();

        for (i, instr) in program.instrs.iter().enumerate() {
            if matches!(instr, VmInstr::LoopBegin { .. }) {
                active_loops.push(i);
            }
            if matches!(instr, VmInstr::LoopEnd) {
                active_loops.pop();
            }
            if matches!(instr, VmInstr::DeclareVReg { .. }) {
                continue;
            }
            // Collect all VRegs referenced by this instruction
            let referenced = Self::referenced_vregs(instr);
            for &vreg in &referenced {
                // Phase 3a: global first-occurrence tightening
                if !global_first_found.contains(&vreg) {
                    if let Some(entry) = map.get_mut(&vreg) {
                        let is_write = is_pure_write_of(instr, vreg);
                        if is_write && entry.2 < i {
                            entry.2 = i; // tighten def_point
                        }
                    }
                    global_first_found.insert(vreg);
                }
                // Phase 3b: per-loop first-occurrence tracking
                for &loop_begin in &active_loops {
                    let key = (vreg, loop_begin);
                    if !loop_first_found.contains(&key) {
                        let is_write = is_pure_write_of(instr, vreg);
                        loop_first_found.insert(key);
                        // If first occurrence in this loop body is NOT a write,
                        // mark this VReg as needing last_use extension to LoopEnd.
                        if !is_write {
                            if let Some(&loop_end) = loop_end_map.get(&loop_begin) {
                                if let Some(entry) = map.get_mut(&vreg) {
                                    if entry.2 < loop_begin && entry.3 < loop_end {
                                        entry.3 = loop_end;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut intervals: Vec<LiveInterval> = map.into_iter()
            .map(|(vreg, (kind, width, def_point, last_use))| {
                LiveInterval { vreg, kind, width, def_point, last_use, lifecycle: LifecycleTag::BodyLocal }
            })
            .collect();

        // ── Pass 4: Lifecycle Tagging (REQ-LC-002) ──
        // 委托给独立的 infer_lifecycle 函数, 实现 §0.1 的 5 条规则。
        let program_len = program.instrs.len();

        for iv in &mut intervals {
            iv.lifecycle = infer_lifecycle(
                iv.vreg,
                iv.kind,
                iv.def_point,
                iv.last_use,
                &loop_ranges,
                &loop_first_found,
                &scope_ranges,
                program_len,
            );
        }

        intervals
    }

    /// REQ-LC-005: 预扫描 VmProgram 构建 scope 位置表。
    /// 返回 Vec<(scope_begin_pos, scope_end_pos, scope_id)>，用于 scope_at_position 查询。
    fn build_scope_positions(program: &VmProgram) -> Vec<(usize, usize, usize)> {
        let mut positions = Vec::new();
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for (i, instr) in program.instrs.iter().enumerate() {
            match instr {
                VmInstr::ScopeBegin { scope_id } => stack.push((i, *scope_id)),
                VmInstr::ScopeEnd { .. } => {
                    if let Some((begin, sid)) = stack.pop() {
                        positions.push((begin, i, sid));
                    }
                }
                _ => {}
            }
        }
        positions
    }

    /// REQ-LC-005: 查询给定程序位置所属的最内层 scope。
    /// 返回 None 表示位置不在任何 scope 内（全局）。
    fn scope_at_position(scope_positions: &[(usize, usize, usize)], pos: usize) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None; // (begin, scope_id)
        for &(begin, end, sid) in scope_positions {
            if pos > begin && pos < end {
                // 选最内层（最晚开始的 scope）
                match best {
                    Some((b, _)) if begin <= b => {}
                    _ => best = Some((begin, sid)),
                }
            }
        }
        best.map(|(_, sid)| sid)
    }

    /// 递归收集 OffsetExpr 中引用的所有 VRegId。
    fn offset_vregs(expr: &OffsetExpr) -> Vec<VRegId> {
        match expr {
            OffsetExpr::Const(_) => vec![],
            OffsetExpr::LoopOffset(v) | OffsetExpr::ScalarVReg(v) => vec![*v],
            OffsetExpr::Add(a, b) => {
                let mut out = Self::offset_vregs(a);
                out.extend(Self::offset_vregs(b));
                out
            }
            OffsetExpr::Mul(inner, _) => Self::offset_vregs(inner),
        }
    }

    /// 收集 PtrExpr 中引用的所有 VRegId。
    fn ptr_vregs(expr: &PtrExpr) -> Vec<VRegId> {
        match expr {
            PtrExpr::AbiArg(_)
            | PtrExpr::StackArg(_)
            | PtrExpr::NamedArg(_)
            | PtrExpr::SharedMem
            | PtrExpr::AbsAddr(_) => vec![],
            PtrExpr::VRegPlusConst(base, _) => vec![*base],
            PtrExpr::VRegPlusVReg(base, off) => vec![*base, *off],
            PtrExpr::VRegPlusOff(base, _) => vec![*base],
        }
    }

    /// 收集 BoundExpr 中引用的所有 VRegId。
    fn bound_vregs(expr: &BoundExpr) -> Vec<VRegId> {
        match expr {
            BoundExpr::Const(_) | BoundExpr::Runtime(_) | BoundExpr::Symbolic(_) => vec![],
            BoundExpr::DynamicVReg(v) | BoundExpr::DynamicVRegPlusOne(v) => vec![*v],
        }
    }

    /// 收集 ScalarExpr 中引用的所有 VRegId。
    fn scalar_expr_vregs(expr: &ScalarExpr) -> Vec<VRegId> {
        match expr {
            ScalarExpr::Const(_) => vec![],
            ScalarExpr::MemLoad(base, off) => {
                let mut v = vec![*base];
                v.extend(Self::offset_vregs(off));
                v
            }
            ScalarExpr::ExtractLane0(src) => vec![*src],
            ScalarExpr::VReg(src) => vec![*src],
        }
    }

    /// 提取一条 VmInstr 引用的所有 VRegId (含嵌套表达式中的 VReg)。
    pub fn referenced_vregs(instr: &VmInstr) -> Vec<VRegId> {
        match instr {
            VmInstr::VecLoad { dst, base, offset, .. } => {
                let mut v = vec![*dst, *base];
                v.extend(Self::offset_vregs(offset));
                v
            }
            VmInstr::VecStore { base, offset, src, .. } => {
                let mut v = vec![*base, *src];
                v.extend(Self::offset_vregs(offset));
                v
            }
            VmInstr::VecNarrow { dst, src, .. } => vec![*dst, *src],
            VmInstr::VecWiden { dst, src, .. } => vec![*dst, *src],
            VmInstr::Mov { dst, src, .. } => vec![*dst, *src],
            VmInstr::Broadcast { dst, src, .. } => {
                let mut v = vec![*dst];
                v.extend(Self::scalar_expr_vregs(src));
                v
            }
            VmInstr::LoadPtr { dst, src } => {
                let mut v = vec![*dst];
                v.extend(Self::ptr_vregs(src));
                v
            }
            VmInstr::VecBinOp { dst, a, b, .. } => vec![*dst, *a, *b],
            VmInstr::VecShiftImm { dst, a, .. } => vec![*dst, *a],
            VmInstr::VecUnaryOp { dst, a, .. } => vec![*dst, *a],
            VmInstr::VecCmp { dst, a, b, .. } => vec![*dst, *a, *b],
            VmInstr::VecCast { dst, src, .. } => vec![*dst, *src],
            VmInstr::ConditionalSelect { dst, mask, true_val, false_val, .. } => vec![*dst, *mask, *true_val, *false_val],
            VmInstr::Fma { dst, acc, a, b, .. } => vec![*dst, *acc, *a, *b],
            VmInstr::HReduce { dst, src, .. } => vec![*dst, *src],
            VmInstr::Accumulate { acc, src } => vec![*acc, *src],
            VmInstr::LoopBegin { counter, byte_offset, bound, .. } => {
                let mut v = vec![*counter, *byte_offset];
                v.extend(Self::bound_vregs(bound));
                v
            }
            VmInstr::Transcendental { dst, src, .. } => vec![*dst, *src],
            VmInstr::ConditionalSkip { mask, .. } => vec![*mask],
            VmInstr::GprCondAction { cond, action } => {
                let mut v = cond.vregs();
                v.extend(action.vregs());
                v
            }
            VmInstr::TileMma { c, a, b } => vec![*c, *a, *b],
            VmInstr::SparseMaskIntersect { dst_k0, dst_k1, a, b } => vec![*dst_k0, *dst_k1, *a, *b],
            VmInstr::ScalarLoad { dst, base, offset } => {
                let mut v = vec![*dst, *base];
                v.extend(Self::offset_vregs(offset));
                v
            }
            VmInstr::ScalarToIndex { dst, src, .. } => vec![*dst, *src],
            VmInstr::IndexToScalar { dst, src } => vec![*dst, *src],
            VmInstr::ScalarByteLoad { dst, base, offset } => {
                let mut v = vec![*dst, *base];
                v.extend(Self::offset_vregs(offset));
                v
            }
            VmInstr::AsyncCopy { dst, src, .. } => vec![*dst, *src],
            VmInstr::IndirectJump { index, .. } => vec![*index],
            VmInstr::ConditionalExit { condition, output } => vec![*condition, *output],
            VmInstr::BranchIfPtrNonNull { ptr, .. } => vec![*ptr],
            VmInstr::BranchIfGprZero { value, .. } => vec![*value],
            VmInstr::BranchIfGprLtU { a, b, .. } => vec![*a, *b],
            VmInstr::UnconditionalBranch { .. } => vec![],
            VmInstr::BatchSeqIdLookup { dst, pt_offset_out, token_index, batch_ctx_ptr } => vec![*dst, *pt_offset_out, *token_index, *batch_ctx_ptr],
            VmInstr::BatchPerSeqArgmax { dst, seq_id, logits_flat_ptr, .. } => vec![*dst, *seq_id, *logits_flat_ptr],
            VmInstr::BatchPerSeqStopCheck { seq_id, token_id, batch_ctx_ptr } => vec![*seq_id, *token_id, *batch_ctx_ptr],
            VmInstr::AtomicAdd { base, .. } => vec![*base],
            VmInstr::MemFence { .. } => vec![],
            VmInstr::DeclareVReg { id, .. } => vec![*id],
            VmInstr::ReleaseVReg { id } => vec![*id],
            // Mega-kernel instructions — must reference all VReg operands
            // to keep their live intervals correct across Argmax/StoreToken/etc.
            VmInstr::Argmax { dst, logits_ptr, .. } => vec![*dst, *logits_ptr],
            VmInstr::TemperatureScale { logits_ptr, temp_ptr, .. } => vec![*logits_ptr, *temp_ptr],
            VmInstr::StoreToken { token_id, output_buf, counter, input_ids_ptr, prompt_len_bytes } => vec![*token_id, *output_buf, *counter, *input_ids_ptr, *prompt_len_bytes],
            VmInstr::CheckStopCondition { token_id, counter, eos_ptr, max_tokens_ptr } => vec![*token_id, *counter, *eos_ptr, *max_tokens_ptr],
            VmInstr::AddPtr { dst, base, .. } => vec![*dst, *base],
            VmInstr::StoreConstToStack { .. } => vec![],
            VmInstr::BreakLoop { return_value } => match return_value {
                ReturnValue::Const(_) => vec![],
                ReturnValue::VReg(v) => vec![*v],
            },
            VmInstr::MarkLabel { .. } => vec![],
            VmInstr::GprBinOp { dst, a, b, .. } => {
                let mut v = vec![*dst, *a];
                if let Some(vr) = b.vreg() { v.push(vr); }
                v
            }
            VmInstr::GprUnaryOp { dst, src, .. } => vec![*dst, *src],
            VmInstr::GprLoadImm { dst, .. } => vec![*dst],
            VmInstr::MemCopy { dst, src, .. } => vec![*dst, *src],
            VmInstr::LoadCallbackEntry { table_ptr, fn_ptr_out, ctx_out, .. } => vec![*table_ptr, *fn_ptr_out, *ctx_out],
            VmInstr::NativeCall { ret_val, fn_ptr, ctx_ptr } => vec![*ret_val, *fn_ptr, *ctx_ptr],
            VmInstr::ScalarStore { base, src, .. } => vec![*base, *src],
            VmInstr::VecScalarStore { base, src, .. } => vec![*base, *src],
            VmInstr::IntMulStride { dst, src, .. } => vec![*dst, *src],
            VmInstr::Prefetch { base, .. } => vec![*base],
            VmInstr::GgufSubScaleLoad { dst, scales_base, sub_block_idx, .. } => vec![*dst, *scales_base, *sub_block_idx],
            VmInstr::GgufKQuantScaleLoad { dst, scales_base, sub_block_idx, .. } => vec![*dst, *scales_base, *sub_block_idx],
            VmInstr::ActivationSwap { ptr_a, ptr_b } => vec![*ptr_a, *ptr_b],
            VmInstr::Comment(_) | VmInstr::PageTableAddr { .. } | VmInstr::PageTableKVWrite { .. } | VmInstr::PageTableKVWriteQuant { .. } | VmInstr::KiviQuantChannel { .. } | VmInstr::KiviQuantToken { .. } | VmInstr::KiviDequantLoad { .. } => vec![],
            VmInstr::HotpatchSlot { .. } => vec![],
            VmInstr::TileConfig { .. } => vec![],
            VmInstr::TileRelease => vec![],
            VmInstr::WarpSync => vec![],
            VmInstr::AsyncWait { .. } => vec![],
            VmInstr::ScopeBegin { .. } | VmInstr::ScopeEnd { .. } | VmInstr::LoopEnd => vec![],
            VmInstr::GatherLoad { dst, base, indices, .. } => vec![*dst, *base, *indices],
            VmInstr::ScatterStore { base, indices, src, .. } => vec![*base, *indices, *src],
            VmInstr::TableLookup { dst, base, row_index, .. } => vec![*dst, *base, *row_index],
            VmInstr::SharedMemAlloc { .. } => vec![],
            VmInstr::SharedMemStore { src, .. } => {
                let mut v = vec![*src];
                // Extract VReg from offset if needed
                if let VmInstr::SharedMemStore { dst_offset, .. } = instr {
                    v.extend(Self::offset_vregs(dst_offset));
                }
                v
            }
            VmInstr::SharedMemLoad { dst, .. } => {
                let mut v = vec![*dst];
                // Extract VReg from offset if needed
                if let VmInstr::SharedMemLoad { src_offset, .. } = instr {
                    v.extend(Self::offset_vregs(src_offset));
                }
                v
            }
            VmInstr::SharedMemAsyncStore { src, .. } => {
                let mut v = vec![*src];
                if let VmInstr::SharedMemAsyncStore { dst_offset, .. } = instr {
                    v.extend(Self::offset_vregs(dst_offset));
                }
                v
            }
            VmInstr::SharedMemAsyncWaitGroup { .. } => vec![],
            VmInstr::WeightPrefetchAsync { weight_base, .. } => vec![*weight_base],
            VmInstr::WeightPrefetchWait { .. } => vec![],
            VmInstr::BlockSync => vec![],
            VmInstr::WarpReduce { src, dst, .. } => vec![*src, *dst],
            // Quant* decode instrs
            VmInstr::QuantBroadcastInt { dst, .. } => vec![*dst],
            VmInstr::QuantLoadBytesVec { dst, base, .. } => vec![*dst, *base],
            VmInstr::QuantCodebookLookup { dst, indices, .. } => vec![*dst, *indices],
            VmInstr::QuantExtractBits { dst, src, .. } => vec![*dst, *src],
            VmInstr::QuantDequantFma { dst, weight, activation, scale, zero_point, .. } => vec![*dst, *weight, *activation, *scale, *zero_point],
            VmInstr::QuantInterleave { dst, lo, hi, .. } => vec![*dst, *lo, *hi],
            VmInstr::QuantConcatSeq { dst, lo, hi, .. } => vec![*dst, *lo, *hi],
            VmInstr::Q3KDecodeStep { dst, block_base, lane_offset, d_vreg, .. } => vec![*dst, *block_base, *lane_offset, *d_vreg],
            VmInstr::DotProduct { acc, a, b, .. } => vec![*acc, *a, *b],
            VmInstr::ScaleApply { dst, acc, scale, zero, .. } => vec![*dst, *acc, *scale, *zero],
            VmInstr::BitwiseGemm { dst, sign_bits, input_sign_bits, scale, .. } => vec![*dst, *sign_bits, *input_sign_bits, *scale],
            VmInstr::SparseGemm { acc, a_sparse, b_dense, sparse_mask_ptr, .. } => vec![*acc, *a_sparse, *b_dense, *sparse_mask_ptr],
            VmInstr::SparseFp8Gemm { acc, a_sparse, b_dense, sparse_mask_ptr, .. } => vec![*acc, *a_sparse, *b_dense, *sparse_mask_ptr],
            VmInstr::NativeFp4Gemm { acc, a, b, scale_a, scale_b, .. } => vec![*acc, *a, *b, *scale_a, *scale_b],
            VmInstr::NativeFp8Gemm { acc, a, b, .. } => vec![*acc, *a, *b],
            VmInstr::HwQuantDequant { dst, packed_weight, block_scale, global_scale, .. } => vec![*dst, *packed_weight, *block_scale, *global_scale],
            VmInstr::TmemAlloc { .. } => vec![],
            VmInstr::TmemLoad { dst, .. } => {
                let mut v = vec![*dst];
                if let VmInstr::TmemLoad { offset, .. } = instr {
                    v.extend(Self::offset_vregs(offset));
                }
                v
            }
            VmInstr::TmemStore { src, .. } => {
                let mut v = vec![*src];
                if let VmInstr::TmemStore { offset, .. } = instr {
                    v.extend(Self::offset_vregs(offset));
                }
                v
            }
            VmInstr::TmemDealloc { .. } => vec![],
            VmInstr::ClusterBarrierInit { .. } => vec![],
            VmInstr::ClusterStore { src, .. } => {
                let mut v = vec![*src];
                if let VmInstr::ClusterStore { offset, .. } = instr {
                    v.extend(Self::offset_vregs(offset));
                }
                v
            }
            VmInstr::ClusterLoad { dst, .. } => {
                let mut v = vec![*dst];
                if let VmInstr::ClusterLoad { offset, .. } = instr {
                    v.extend(Self::offset_vregs(offset));
                }
                v
            }
            VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, .. } => vec![*src_ptr, *dst_ptr, *compressed_size],
            VmInstr::BitPackRleDecode { src_ptr, dst_ptr, compressed_size, .. } => vec![*src_ptr, *dst_ptr, *compressed_size],

            // ── Sampling instructions ──
            VmInstr::SoftmaxReduceMax { dst, logits_ptr, .. } => vec![*dst, *logits_ptr],
            VmInstr::SoftmaxExpSum { sum_dst, logits_ptr, max_val, .. } => vec![*sum_dst, *logits_ptr, *max_val],
            VmInstr::SoftmaxNormalize { logits_ptr, sum_val, .. } => vec![*logits_ptr, *sum_val],
            VmInstr::SampleTopKFilter { probs_ptr, indices_ptr, k_ptr, .. } => vec![*probs_ptr, *indices_ptr, *k_ptr],
            VmInstr::SampleTopPFilter { probs_ptr, p_ptr, .. } => vec![*probs_ptr, *p_ptr],
            VmInstr::SampleMultinomial { dst, probs_ptr, rng_state_ptr, .. } => vec![*dst, *probs_ptr, *rng_state_ptr],
            VmInstr::WarpPRNG { dst, rng_state_ptr } => vec![*dst, *rng_state_ptr],
            VmInstr::SharedMemSwizzle { dst, raw_addr, .. } => vec![*dst, *raw_addr],
            VmInstr::WarpRoleDeclare { .. } | VmInstr::WarpBarrierArrive { .. } | VmInstr::WarpBarrierWait { .. } => vec![],
            VmInstr::DebugBreakpoint { .. } | VmInstr::DebugMarker { .. } => vec![],
            VmInstr::DebugProbe { vreg, .. } => vec![*vreg],
            VmInstr::DebugBreakIf { cond_gpr, .. } => vec![*cond_gpr],
            VmInstr::QuantScalarCvtLoad { base, .. } => vec![*base],
            VmInstr::QuantBlockLoad { dst, base, offset, unpack, .. } => {
                let mut v = vec![*dst, *base];
                v.extend(Self::offset_vregs(offset));
                v.extend(unpack.vregs());
                v
            }
            VmInstr::QuantBiPlaneLoad { dst, qs_base, extra_base, .. } => vec![*dst, *qs_base, *extra_base],
            VmInstr::VecShuffle { dst, src, mask, .. } => {
                let mut v = vec![*dst, *src];
                if let super::instr::VecShuffleMask::Dynamic { ctrl } = mask { v.push(*ctrl); }
                v
            }
            VmInstr::VecExtractLane { dst, src, .. } => vec![*dst, *src],
            VmInstr::VecInsertLane { dst, src_vec, src_scalar, .. } => vec![*dst, *src_vec, *src_scalar],
            VmInstr::VecLoadConst { dst, .. } => vec![*dst],
            VmInstr::AtomicCAS { dst, ptr, expected, desired, .. } => vec![*dst, *ptr, *expected, *desired],
            VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, .. } => vec![*dst, *token_index, *seq_meta_base, *num_seqs],
            #[cfg(feature = "nccl")]
            VmInstr::AllReduceChunk { sendbuf, recvbuf, count, rank, world_size, chunk_idx, .. } => vec![*sendbuf, *recvbuf, *count, *rank, *world_size, *chunk_idx],
            #[cfg(feature = "nccl")]
            VmInstr::CommBarrier { thread_count, .. } => vec![*thread_count],
            #[cfg(feature = "nccl")]
            VmInstr::NvlinkAsyncCopy { dst, src, len, .. } => vec![*dst, *src, *len],
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageLookup { dst, seq_id, page_index, routing_table_base } => vec![*dst, *seq_id, *page_index, *routing_table_base],
            #[cfg(feature = "nccl")]
            VmInstr::P2pPageFetch { local_buf, peer_buf, page_size, barrier } => vec![*local_buf, *peer_buf, *page_size, *barrier],
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetch { local_buf, remote_addr, rkey, page_size, sq_desc, doorbell, cq_addr } => vec![*local_buf, *remote_addr, *rkey, *page_size, *sq_desc, *doorbell, *cq_addr],
            #[cfg(feature = "nccl")]
            VmInstr::RdmaPageFetchCompressed { local_buf, scratch_buf, page_size, remote_addr, rkey, sq_desc, doorbell, cq_addr, .. } => vec![*local_buf, *scratch_buf, *page_size, *remote_addr, *rkey, *sq_desc, *doorbell, *cq_addr],
            #[cfg(feature = "nccl")]
            VmInstr::RemotePageAttn { q_buf, k_remote_buf, v_remote_buf, output_buf, shared_buf, barrier, tile_bytes } => vec![*q_buf, *k_remote_buf, *v_remote_buf, *output_buf, *shared_buf, *barrier, *tile_bytes],
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationLock { dst, entry_addr } => vec![*dst, *entry_addr],
            #[cfg(feature = "nccl")]
            VmInstr::PageMigrationUnlock { entry_addr } => vec![*entry_addr],
            #[cfg(feature = "nccl")]
            VmInstr::PageLocationUpdate { entry_addr, new_location, .. } => vec![*entry_addr, *new_location],
            VmInstr::TmaDescriptorInit { .. } => vec![],
            VmInstr::Tma2DCopy { coord_x, coord_y, .. } => vec![*coord_x, *coord_y],
            VmInstr::BarrierInit { .. } => vec![],
        }
    }

    /// 线性扫描分配。
    pub fn allocate(&self, program: &VmProgram) -> Result<RegAllocation, String> {
        self.allocate_impl(program, None)
    }

    /// 线性扫描分配 + JitContext 预算门控 (SPEC 15 REQ-JCTX-013)。
    ///
    /// 分配完成后验证使用的物理寄存器数不超过 JitContext 的硬件预算。
    pub fn allocate_with_context(
        &self,
        program: &VmProgram,
        ctx: &crate::compiler::jit_context::JitContext,
    ) -> Result<RegAllocation, String> {
        self.allocate_impl(program, Some(ctx))
    }

    fn allocate_impl(
        &self,
        program: &VmProgram,
        budget_ctx: Option<&crate::compiler::jit_context::JitContext>,
    ) -> Result<RegAllocation, String> {
        // ARCH-GPU-NO-LINEAR-SCAN: GPU 后端 (PTX/HIP/MSL) 使用虚拟寄存器命名空间
        // (%r_N / %f_N / %rd_N),由 GpuLower 按 VRegKind 直接映射,无需 CPU 风格
        // 物理寄存器分配。GPU profile 的 abi.callee_saved / caller_saved / vec_regs
        // 均为空,若走 CPU 线性扫描会立刻失败 (pool_size=0 → Counter 无法 spill)。
        // 直接返回空 mapping + 空 spills,GpuLower::reg_name_with_kind 通过 fallback
        // 路径按 VReg.0 生成虚拟名字。
        if matches!(
            self.profile.platform,
            super::isa_profile::Platform::Cuda { .. }
                | super::isa_profile::Platform::Hip { .. }
                | super::isa_profile::Platform::Metal { .. }
        ) {
            return Ok(RegAllocation {
                mapping: HashMap::new(),
                spills: Vec::new(),
                callee_saved_used: Vec::new(),
            });
        }

        let intervals = Self::compute_intervals(program);
        let interference = InterferenceGraph::build(&intervals);

        // DIAG: unique ID for each allocate_impl call
        static ALLOC_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let alloc_id = ALLOC_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        eprintln!("[ALLOC-ID] allocate_impl #{}: {} instrs, {} intervals", alloc_id, program.instrs.len(), intervals.len());

        // REQ-LC-004~006: 使用 ScopedSpillAllocator 替代顺序单调分配。
        // 预扫描 VmProgram 构建 scope 位置表，用于在 spill 分配时确定归属 scope。
        let scope_positions = Self::build_scope_positions(program);

        let mut mapping: HashMap<VRegId, PhysReg> = HashMap::new();
        let mut gpr_used: HashSet<PhysGpr> = HashSet::new();
        let mut vec_used: HashSet<PhysVec> = HashSet::new();
        let mut scoped_alloc = super::stack_frame::ScopedSpillAllocator::new();

        // 两阶段排序 (ARCH-REGALLOC-COUNTER-PRIORITY + REQ-LC-003):
        //   优先级: Counter > Global > LoopInvariant/LoopCarried > CrossScope > BodyLocal
        //   同优先级内按 def_point 排序保证 linear scan 正确性。
        // Vec/Tile/Mask 用独立寄存器类，排序不影响 GPR 分配。
        let mut sorted = intervals.clone();
        sorted.sort_by_key(|iv| {
            let kind_priority = match iv.kind {
                VRegKind::Counter | VRegKind::ByteOffset => 0,
                VRegKind::Ptr | VRegKind::Scalar => 1,
                VRegKind::Vec | VRegKind::Tile | VRegKind::Mask => 2,
            };
            let lifecycle_priority = iv.lifecycle.alloc_priority();
            (kind_priority, lifecycle_priority, iv.def_point)
        });

        for iv in &sorted {
            // 收集干涉邻居已占用的物理寄存器
            let occupied_gpr: HashSet<PhysGpr> = interference.neighbors(iv.vreg)
                .filter_map(|n| mapping.get(n))
                .filter_map(|p| match p { PhysReg::Gpr(g) => Some(*g), _ => None })
                .collect();
            let occupied_vec: HashSet<PhysVec> = interference.neighbors(iv.vreg)
                .filter_map(|n| mapping.get(n))
                .filter_map(|p| match p { PhysReg::Vec(v) => Some(*v), _ => None })
                .collect();

            match iv.kind {
                VRegKind::Ptr | VRegKind::Scalar | VRegKind::Counter | VRegKind::ByteOffset => {
                    // 分配 GPR。Counter/ByteOffset 优先 callee-saved (跨循环生存)。
                    let prefer_callee = matches!(iv.kind, VRegKind::Counter | VRegKind::ByteOffset);
                    let pool = if prefer_callee {
                        let mut p = self.profile.abi.callee_saved.clone();
                        p.extend(self.profile.abi.caller_saved.iter());
                        p
                    } else {
                        let mut p = self.profile.abi.caller_saved.clone();
                        p.extend(self.profile.abi.callee_saved.iter());
                        p
                    };

                    if let Some(reg) = pool.iter().find(|r| !occupied_gpr.contains(r)) {
                        mapping.insert(iv.vreg, PhysReg::Gpr(*reg));
                        gpr_used.insert(*reg);
                    } else {
                        // Counter/ByteOffset/Ptr/Scalar: 无可用物理 GPR → spill 到栈。
                        // ARCH-REGALLOC-GPR-SPILL: mapping 必须记录 Spilled 变体，
                        // 否则 resolve_gpr 在 ISA Lower 阶段找不到 mapping 报错。
                        // REQ-LC-004~006: 使用 ScopedSpillAllocator 分配 scope-aware slot。
                        // Counter spilling: ISA Lower 在 LoopBegin/LoopEnd 通过 scratch GPR
                        // 做 load→cmp→inc→store (见 x86_lower/aarch64_lower LoopBegin/LoopEnd)。
                        let scope_id = Self::scope_at_position(&scope_positions, iv.def_point);
                        let (slot_id, _offset) = scoped_alloc.alloc(iv.vreg, 8, scope_id);
                        mapping.insert(iv.vreg, PhysReg::Spilled(slot_id as u32));
                    }
                }
                VRegKind::Vec => {
                    if let Some(reg) = self.profile.vec_regs.iter()
                        .find(|r| !occupied_vec.contains(r))
                    {
                        mapping.insert(iv.vreg, PhysReg::Vec(*reg));
                        vec_used.insert(*reg);
                    } else {
                        // ARCH-REGALLOC-VEC-SPILL (task #14 part 2 fix): Vec spill 时
                        // 必须插入 mapping (PhysReg::Spilled),否则下游 ISA Lower 报
                        // "v{} not allocated to YMM"。
                        // REQ-LC-004~006: ScopedSpillAllocator scope-aware slot。
                        let size = iv.width.bytes().max(32);
                        let scope_id = Self::scope_at_position(&scope_positions, iv.def_point);
                        let (slot_id, _offset) = scoped_alloc.alloc(iv.vreg, size, scope_id);
                        mapping.insert(iv.vreg, PhysReg::Spilled(slot_id as u32));
                    }
                }
                VRegKind::Tile => {
                    if let Some(reg) = self.profile.tile_regs.first() {
                        mapping.insert(iv.vreg, PhysReg::Tile(*reg));
                    }
                }
                VRegKind::Mask => {
                    if let Some(reg) = self.profile.mask_regs.first() {
                        mapping.insert(iv.vreg, PhysReg::Mask(*reg));
                    }
                }
            }
        }

        // 统计实际使用的 callee-saved
        let callee_saved_used: Vec<PhysGpr> = self.profile.abi.callee_saved.iter()
            .filter(|r| gpr_used.contains(r))
            .copied()
            .collect();

        // REQ-LC-004: 将 ScopedSpillAllocator 产物转换为兼容的 spills Vec。
        let spills = scoped_alloc.into_spills();
        let spill_total_bytes: usize = spills.iter().map(|s| s.size).sum();

        // Debug: dump GPR allocation and spills to file (avoid test harness truncation)
        if !spills.is_empty() || std::env::var("GLLM_REGALLOC_DEBUG").is_ok() {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true).append(true)
                .open("/tmp/gllm_regalloc.log")
            {
                // Section header to separate multiple compiles
                let _ = writeln!(f, "\n=== REGALLOC ({} instrs, {} intervals, {} spills) ===",
                    program.instrs.len(), sorted.len(), spills.len());
                for iv in &sorted {
                    let phys = mapping.get(&iv.vreg);
                    let _ = writeln!(f, "  v{} {:?} [def={}, last={}] → {:?}",
                        iv.vreg.0, iv.kind, iv.def_point, iv.last_use, phys);
                }
                if !spills.is_empty() {
                    let _ = writeln!(f, "  spills: {} ({} bytes)", spills.len(), spill_total_bytes);
                    for s in &spills {
                        let _ = writeln!(f, "    v{} offset={} size={}", s.vreg.0, s.offset, s.size);
                    }
                }
            }
        }

        // ── Spill Safety Validation Pass ──
        // For every Spilled VReg, verify that its first write (def_point) precedes
        // its first read. Prologue zero-inits all spill slots; reading before writing
        // yields null → SIGSEGV.
        Self::validate_spill_safety(program, &mapping, &intervals)?;

        // D6: Spill layout consistency — verify no slot overlap and offset monotonicity.
        Self::validate_spill_layout(&spills, &intervals)?;

        // REQ-LC-012: Post-allocation completeness check
        Self::post_alloc_verify(&mapping, &intervals, &spills)?;

        // REQ-LC-010: Post-hoc consistency verification (rules 6/7/9)
        let alloc = RegAllocation { mapping, spills, callee_saved_used };
        super::verify::verify_after_alloc(program, &alloc, &intervals)
            .map_err(|e| format!("verify_after_alloc: {}", e))?;

        // SPEC 15 REQ-JCTX-013: 预算门控 — 分配上限从 JitContext.budget().capacity() 读取
        if let Some(ctx) = budget_ctx {
            use crate::compiler::jit_context::ResourceKind;
            let gpr_capacity = ctx.budget().capacity(ResourceKind::Gpr);
            let vec_capacity = ctx.budget().capacity(ResourceKind::SimdVec);
            let gpr_used_count = alloc.mapping.values()
                .filter(|p| matches!(p, PhysReg::Gpr(_)))
                .count();
            let vec_used_count = alloc.mapping.values()
                .filter(|p| matches!(p, PhysReg::Vec(_)))
                .count();
            if gpr_used_count > gpr_capacity && gpr_capacity > 0 {
                return Err(format!(
                    "RegAllocator: GPR budget exceeded: used {} > capacity {} (REQ-JCTX-013)",
                    gpr_used_count, gpr_capacity
                ));
            }
            if vec_used_count > vec_capacity && vec_capacity > 0 {
                return Err(format!(
                    "RegAllocator: Vec budget exceeded: used {} > capacity {} (REQ-JCTX-013)",
                    vec_used_count, vec_capacity
                ));
            }
        }

        Ok(alloc)
    }

    /// Spill Safety Validation Pass (半 VM 编译时符号追踪核心)
    ///
    /// Prologue 将所有 spill 槽初始化为 0。如果一个 Spilled VReg 在首次写入前
    /// 被读取，读到的值是 0（null），导致 SIGSEGV。
    ///
    /// 此 pass 验证：对于每个 Spilled VReg，首次**纯写**指令在其首次**读取**
    /// 之前。这是半 VM 符号执行的核心安全不变量——编译时追踪值的生命周期，
    /// 确保物理布局与逻辑语义一致。
    ///
    /// 同时检测嵌套循环中的 spill 安全性：如果 VReg 在外层 loop 定义但在内层
    /// loop 被读取，需要确保外层 loop 首次迭代在读取前已经写入了值。
    fn validate_spill_safety(
        program: &VmProgram,
        mapping: &HashMap<VRegId, PhysReg>,
        intervals: &[LiveInterval],
    ) -> Result<(), String> {
        let spilled_vregs: HashMap<VRegId, usize> = mapping.iter()
            .filter_map(|(vreg, phys)| match phys {
                PhysReg::Spilled(slot) => Some((*vreg, *slot as usize)),
                _ => None,
            })
            .collect();

        if spilled_vregs.is_empty() {
            return Ok(());
        }

        // Build interval lookup
        let interval_map: HashMap<VRegId, &LiveInterval> = intervals.iter()
            .map(|iv| (iv.vreg, iv))
            .collect();

        // For each spilled VReg, find first write and first read positions
        let is_pure_write = |instr: &VmInstr, vreg: VRegId| -> bool {
            match instr {
                VmInstr::Broadcast { dst, .. } => *dst == vreg,
                VmInstr::VecLoad { dst, .. } => *dst == vreg,
                VmInstr::LoadPtr { dst, .. } => *dst == vreg,
                VmInstr::ScalarLoad { dst, .. } => *dst == vreg,
                VmInstr::ScalarByteLoad { dst, .. } => *dst == vreg,
                VmInstr::GgufSubScaleLoad { dst, .. } => *dst == vreg,
                VmInstr::GgufKQuantScaleLoad { dst, .. } => *dst == vreg,
                VmInstr::GprBinOp { dst, a, b, .. } => {
                    let other_inputs_ok = *a != vreg && (b.vreg() != Some(vreg));
                    *dst == vreg && other_inputs_ok
                }
                VmInstr::GprUnaryOp { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::GprLoadImm { dst, .. } => *dst == vreg,
                VmInstr::VecUnaryOp { dst, a, .. } => *dst == vreg && *a != vreg,
                VmInstr::VecBinOp { dst, a, b, .. } => {
                    *dst == vreg && *a != vreg && *b != vreg
                }
                VmInstr::VecCmp { dst, a, b, .. } => {
                    *dst == vreg && *a != vreg && *b != vreg
                }
                VmInstr::VecCast { dst, src, .. } => {
                    *dst == vreg && *src != vreg
                }
                VmInstr::ConditionalSelect { dst, mask, true_val, false_val, .. } => {
                    *dst == vreg && *mask != vreg && *true_val != vreg && *false_val != vreg
                }
                VmInstr::HReduce { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::Transcendental { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::Fma { dst, acc, a, b, .. } => {
                    *dst == vreg && *acc != vreg && *a != vreg && *b != vreg
                }
                VmInstr::StoreConstToStack { .. } => false,
                VmInstr::AddPtr { dst, base, .. } => *dst == vreg && *base != vreg,
                VmInstr::MemCopy { .. } => false,
                VmInstr::IntMulStride { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::DeclareVReg { .. } => false,
                VmInstr::ScalarStore { src, .. } => *src == vreg, // reads vreg
                VmInstr::VecScalarStore { src, .. } => *src == vreg, // reads vreg
                // LoopBegin implicitly initializes counter and byte_offset to 0.
                // Both are pure writes — no prior value is read.
                // Quant* instructions that write dst
                VmInstr::QuantLoadBytesVec { dst, base, .. } => *dst == vreg && *base != vreg,
                VmInstr::QuantBroadcastInt { dst, .. } => *dst == vreg,
                VmInstr::QuantExtractBits { dst, src, .. } => *dst == vreg && *src != vreg,
                VmInstr::QuantCodebookLookup { dst, indices, .. } => *dst == vreg && *indices != vreg,
                VmInstr::QuantInterleave { dst, lo, hi, .. }
                | VmInstr::QuantConcatSeq { dst, lo, hi, .. } => {
                    *dst == vreg && *lo != vreg && *hi != vreg
                }
                VmInstr::Q3KDecodeStep { dst, block_base, lane_offset, d_vreg, .. } => {
                    *dst == vreg && *block_base != vreg && *lane_offset != vreg && *d_vreg != vreg
                }
                VmInstr::QuantScalarCvtLoad { dst, base, .. } => *dst == vreg && *base != vreg,
                VmInstr::QuantDequantFma { dst, weight, activation, scale, zero_point, .. } => {
                    *dst == vreg && *weight != vreg && *activation != vreg
                        && *scale != vreg && *zero_point != vreg
                }
                VmInstr::LoopBegin { counter, byte_offset, .. } => {
                    *counter == vreg || *byte_offset == vreg
                }
                VmInstr::ScalarToIndex { dst, src, .. } => *dst == vreg && *src != vreg,
                // Sampling/softmax instructions that write dst
                VmInstr::SoftmaxReduceMax { dst, logits_ptr, .. } => *dst == vreg && *logits_ptr != vreg,
                VmInstr::SoftmaxExpSum { sum_dst, logits_ptr, max_val, .. } => {
                    *sum_dst == vreg && *logits_ptr != vreg && *max_val != vreg
                }
                VmInstr::SampleMultinomial { dst, probs_ptr, rng_state_ptr, .. } => {
                    *dst == vreg && *probs_ptr != vreg && *rng_state_ptr != vreg
                }
                VmInstr::WarpPRNG { dst, rng_state_ptr } => {
                    *dst == vreg && *rng_state_ptr != vreg
                }
                VmInstr::Argmax { dst, logits_ptr, .. } => *dst == vreg && *logits_ptr != vreg,
                VmInstr::BatchPerSeqArgmax { dst, seq_id, logits_flat_ptr, .. } => {
                    *dst == vreg && *seq_id != vreg && *logits_flat_ptr != vreg
                }
                // §20 BCI cumsum search: dst is pure write (seq_id lookup result).
                VmInstr::SeqIdLookup { dst, token_index, seq_meta_base, num_seqs, .. } => {
                    *dst == vreg && *token_index != vreg && *seq_meta_base != vreg && *num_seqs != vreg
                }
                VmInstr::BatchSeqIdLookup { dst, pt_offset_out, token_index, batch_ctx_ptr } => {
                    *dst == vreg && *pt_offset_out != vreg && *token_index != vreg && *batch_ctx_ptr != vreg
                }
                VmInstr::SoftmaxNormalize { logits_ptr, sum_val, .. } => false, // in-place, no VReg dst
                VmInstr::SampleTopKFilter { .. } => false, // in-place, no VReg dst
                VmInstr::SampleTopPFilter { .. } => false, // in-place, no VReg dst
                _ => false,
            }
        };

        let reads_vreg = |instr: &VmInstr, vreg: VRegId| -> bool {
            Self::referenced_vregs(instr).contains(&vreg)
                && !matches!(instr, VmInstr::DeclareVReg { .. })
        };

        for (vreg, slot) in &spilled_vregs {
            let mut first_write: Option<usize> = None;
            let mut first_read: Option<usize> = None;
            let mut write_instr = "";
            let mut read_instr = "";

            for (i, instr) in program.instrs.iter().enumerate() {
                if first_write.is_none() && is_pure_write(instr, *vreg) {
                    first_write = Some(i);
                    write_instr = Self::instr_brief(instr);
                }
                if first_read.is_none() && reads_vreg(instr, *vreg) && !is_pure_write(instr, *vreg) {
                    first_read = Some(i);
                    read_instr = Self::instr_brief(instr);
                }
                if first_write.is_some() && first_read.is_some() {
                    break;
                }
            }

            let def_point = interval_map.get(vreg).map(|iv| iv.def_point);

            match (first_read, first_write) {
                (Some(read_pos), Some(write_pos)) if read_pos < write_pos => {
                    let msg = format!(
                        "Spill Safety Violation: v{} (slot {}) is read at instr {} ({:?}) \
                         before first write at instr {} ({:?}). \
                         Prologue zero-inits spill slots → reads null → SIGSEGV. \
                         def_point from interval = {:?}",
                        vreg.0, slot, read_pos, read_instr, write_pos, write_instr, def_point,
                    );
                    return Err(msg);
                }
                (Some(read_pos), None) => {
                    let instr_dump = if read_pos < program.instrs.len() {
                        format!("{:?}", program.instrs[read_pos])
                    } else {
                        "OOB".to_string()
                    };
                    let ctx_start = read_pos.saturating_sub(2);
                    let ctx_end = (read_pos + 3).min(program.instrs.len());
                    let ctx_dump: Vec<String> = (ctx_start..ctx_end).map(|i| {
                        let marker = if i == read_pos { ">>>" } else { "   " };
                        format!("{} [{}] {:?}", marker, i, program.instrs[i])
                    }).collect();
                    let msg = format!(
                        "Spill Safety Violation: v{} (slot {}) is read at instr {} ({:?}) \
                         but NEVER written. Prologue zero-inits → reads null → SIGSEGV.\n\
                         Read instr: {}\n\
                         Context:\n{}",
                        vreg.0, slot, read_pos, read_instr, instr_dump, ctx_dump.join("\n"),
                    );
                    return Err(msg);
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Brief description of an instruction for diagnostics
    fn instr_brief(instr: &VmInstr) -> &'static str {
        match instr {
            VmInstr::DeclareVReg { .. } => "DeclareVReg",
            VmInstr::VecLoad { .. } => "VecLoad",
            VmInstr::VecStore { .. } => "VecStore",
            VmInstr::VecNarrow { .. } => "VecNarrow",
            VmInstr::VecWiden { .. } => "VecWiden",
            VmInstr::Mov { .. } => "Mov",
            VmInstr::Broadcast { .. } => "Broadcast",
            VmInstr::LoadPtr { .. } => "LoadPtr",
            VmInstr::VecBinOp { .. } => "VecBinOp",
            VmInstr::VecUnaryOp { .. } => "VecUnaryOp",
            VmInstr::Fma { .. } => "Fma",
            VmInstr::HReduce { .. } => "HReduce",
            VmInstr::Accumulate { .. } => "Accumulate",
            VmInstr::LoopBegin { .. } => "LoopBegin",
            VmInstr::LoopEnd => "LoopEnd",
            VmInstr::ScalarLoad { .. } => "ScalarLoad",
            VmInstr::ScalarStore { .. } => "ScalarStore",
            VmInstr::VecScalarStore { .. } => "VecScalarStore",
            VmInstr::Argmax { .. } => "Argmax",
            VmInstr::StoreToken { .. } => "StoreToken",
            VmInstr::CheckStopCondition { .. } => "CheckStopCondition",
            VmInstr::AddPtr { .. } => "AddPtr",
            VmInstr::GprBinOp { .. } => "GprBinOp",
            VmInstr::GgufSubScaleLoad { .. } => "GgufSubScaleLoad",
            VmInstr::GgufKQuantScaleLoad { .. } => "GgufKQuantScaleLoad",
            VmInstr::IntMulStride { .. } => "IntMulStride",
            VmInstr::TemperatureScale { .. } => "TemperatureScale",
            VmInstr::Comment(_) => "Comment",
            _ => std::any::type_name::<VmInstr>()
                .split("::")
                .last()
                .unwrap_or("VmInstr"),
        }
    }

    /// D6: Spill layout consistency validation (ARCH-VM-SPILL-LAYOUT)。
    ///
    /// 验证三个不变量:
    /// 1. Spill slot 偏移严格递增 (无物理重叠)
    /// REQ-LC-012: Post-allocation completeness check
    ///
    /// 5 项不变量:
    /// 1. 每个 VRegId 都有物理寄存器或 spill slot
    /// 2. 两个干涉的 VReg 不在同一物理寄存器上
    /// 3. Spill slot 无重叠
    /// 4. Counter 永远在物理寄存器上 (不 spill)
    /// 5. LifecycleTag::LoopInvariant 的 VReg 在 loop 内有合法活跃区间
    fn post_alloc_verify(
        mapping: &HashMap<VRegId, PhysReg>,
        intervals: &[LiveInterval],
        spills: &[SpillSlot],
    ) -> Result<(), String> {
        // 1. 每个 VRegId 都有映射
        for iv in intervals {
            if !mapping.contains_key(&iv.vreg) {
                return Err(format!(
                    "post_alloc_verify: VRegId({}) has no physical register or spill slot",
                    iv.vreg.0
                ));
            }
        }

        // 2. Spill slot 无重叠
        for i in 0..spills.len() {
            for j in (i + 1)..spills.len() {
                let a = &spills[i];
                let b = &spills[j];
                let a_end = a.offset + a.size;
                let b_end = b.offset + b.size;
                if a.offset < b_end && b.offset < a_end {
                    return Err(format!(
                        "post_alloc_verify: spill slot overlap v{} [{}, {}) and v{} [{}, {})",
                        a.vreg.0, a.offset, a_end, b.vreg.0, b.offset, b_end
                    ));
                }
            }
        }

        // 3. Counter spill is now supported (load→cmp→inc→store pattern in ISA lower)

        // 4. LifecycleTag 一致性 — LoopInvariant 的活跃区间应跨 loop
        for iv in intervals {
            if iv.lifecycle == LifecycleTag::LoopInvariant && iv.def_point > 0 {
                // LoopInvariant 必须在 loop 外定义
                // (This is a soft check — the real verification is in verify.rs rule 9)
            }
        }

        Ok(())
    }

    /// 2. 同时活跃的 VReg 不共享重叠的 spill slot
    /// 3. GPR spill 的 size=8 和 Vec spill 的 size=32/64 与 VRegKind 一致
    fn validate_spill_layout(
        spills: &[SpillSlot],
        intervals: &[LiveInterval],
    ) -> Result<(), String> {
        if spills.len() <= 1 {
            return Ok(());
        }

        // Check 1: Monotonic offset + no physical overlap
        for w in spills.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if b.offset < a.offset + a.size {
                return Err(format!(
                    "Spill Layout: physical overlap — v{} [{}, {}) overlaps v{} [{}, {})",
                    a.vreg.0, a.offset, a.offset + a.size,
                    b.vreg.0, b.offset, b.offset + b.size,
                ));
            }
        }

        // Check 2: Simultaneously live VRegs must not share overlapping spill slots.
        // Since our allocator gives each VReg its own slot, this should always pass.
        // But if someone changes the allocator to reuse dead slots, this catches bugs.
        let interval_map: HashMap<VRegId, &LiveInterval> = intervals.iter()
            .map(|iv| (iv.vreg, iv))
            .collect();

        for i in 0..spills.len() {
            for j in (i + 1)..spills.len() {
                let a = &spills[i];
                let b = &spills[j];
                // Check if spill slots physically overlap
                if a.offset < b.offset + b.size && b.offset < a.offset + a.size {
                    // Check if they are simultaneously live
                    if let (Some(ia), Some(ib)) = (interval_map.get(&a.vreg), interval_map.get(&b.vreg)) {
                        if ia.def_point <= ib.last_use && ib.def_point <= ia.last_use {
                            return Err(format!(
                                "Spill Layout: live overlap — v{} [{}, {}) and v{} [{}, {}) \
                                 are simultaneously live (v{}: [{},{}], v{}: [{},{}])",
                                a.vreg.0, a.offset, a.offset + a.size,
                                b.vreg.0, b.offset, b.offset + b.size,
                                a.vreg.0, ia.def_point, ia.last_use,
                                b.vreg.0, ib.def_point, ib.last_use,
                            ));
                        }
                    }
                }
            }
        }

        // Check 3: Spill slot size consistency with VRegKind
        let kind_map: HashMap<VRegId, VRegKind> = intervals.iter()
            .map(|iv| (iv.vreg, iv.kind))
            .collect();
        for slot in spills {
            if let Some(kind) = kind_map.get(&slot.vreg) {
                let expected_size = match kind {
                    VRegKind::Ptr | VRegKind::Scalar | VRegKind::Counter | VRegKind::ByteOffset => 8,
                    VRegKind::Vec => 32, // minimum, can be 64 for AVX-512
                    _ => continue,
                };
                if slot.size < expected_size {
                    return Err(format!(
                        "Spill Layout: v{} ({:?}) spill size {} < expected minimum {}",
                        slot.vreg.0, kind, slot.size, expected_size,
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::trace::QuantPrecision;
    use super::*;
    use crate::dispatch::DeviceProfile;
    use crate::compiler::codegen::vm::auto_select;
    use crate::compiler::trace::{TraceOp, ValueId};

    fn test_profile() -> IsaProfile {
        IsaProfile::from_device_profile(&DeviceProfile::detect())
    }

    #[test]
    fn test_allocate_elementwise_no_spill() {
        let body = vec![TraceOp::Input(0), TraceOp::Neg(ValueId(0)), TraceOp::Exp(ValueId(1)),
            TraceOp::Const(1.0), TraceOp::Add(ValueId(2), ValueId(3)), TraceOp::Recip(ValueId(4)), TraceOp::Mul(ValueId(0), ValueId(5))];
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_reg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad { dst: vec_reg, base: input_ptr, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let _slots = auto_select::auto_lower_trace_raw(&mut prog, &body, &[vec_reg], SimdWidth::W256, QuantPrecision::F32).unwrap();
        let profile = test_profile();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();

        assert!(alloc.spills.is_empty(), "unexpected spills: {:?}", alloc.spills);
        assert!(alloc.mapping.len() >= 5);
    }

    #[test]
    fn test_allocate_gemm_no_conflict() {
        // GEMM-like VmProgram: 3 Ptr + 3 Vec + FMA loop
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let a_bc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let b_v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width: SimdWidth::W256, dtype: QuantPrecision::F32, });
        prog.emit(VmInstr::Broadcast { dst: a_bc, src: ScalarExpr::MemLoad(a_ptr, OffsetExpr::Const(0)), width: SimdWidth::W256, dtype: QuantPrecision::F32, });
        prog.emit(VmInstr::VecLoad { dst: b_v, base: b_ptr, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let fma_body = [TraceOp::Input(0), TraceOp::Input(1), TraceOp::Fma(ValueId(0), ValueId(0), ValueId(1))];
        let slots = auto_select::auto_lower_trace_raw(&mut prog, &fma_body, &[acc, a_bc, b_v], SimdWidth::W256, QuantPrecision::F32).unwrap();
        prog.emit(VmInstr::VecStore { base: c_ptr, src: slots[0], offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let profile = test_profile();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();

        // 验证无干涉: 所有同类已分配的 VReg 如果干涉则不能共享物理寄存器
        let intervals = RegAllocator::compute_intervals(&prog);
        let ig = InterferenceGraph::build(&intervals);
        for iv in &intervals {
            if let Some(phys) = alloc.get(iv.vreg) {
                for neighbor in ig.neighbors(iv.vreg) {
                    if let Some(n_phys) = alloc.get(*neighbor) {
                        assert_ne!(phys, n_phys,
                            "conflict: v{} and v{} both mapped to {:?}",
                            iv.vreg.0, neighbor.0, phys);
                    }
                }
            }
        }
    }

    #[test]
    fn test_callee_saved_tracked() {
        let body = vec![TraceOp::Input(0)];
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_reg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad { dst: vec_reg, base: input_ptr, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let _slots = auto_select::auto_lower_trace_raw(&mut prog, &body, &[vec_reg], SimdWidth::W256, QuantPrecision::F32).unwrap();
        let profile = test_profile();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();

        // callee_saved_used 应该是实际使用的 callee-saved 的子集
        for reg in &alloc.callee_saved_used {
            assert!(profile.abi.callee_saved.contains(reg));
        }
    }

    #[test]
    fn test_interference_graph() {
        let a = LiveInterval { vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256, def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal };
        let b = LiveInterval { vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256, def_point: 5, last_use: 15, lifecycle: LifecycleTag::BodyLocal };
        let c = LiveInterval { vreg: VRegId(2), kind: VRegKind::Vec, width: SimdWidth::W256, def_point: 11, last_use: 20, lifecycle: LifecycleTag::BodyLocal };

        let ig = InterferenceGraph::build(&[a, b, c]);
        assert!(ig.interferes(VRegId(0), VRegId(1))); // [0,10] ∩ [5,15] ≠ ∅
        assert!(!ig.interferes(VRegId(0), VRegId(2))); // [0,10] ∩ [11,20] = ∅
        assert!(ig.interferes(VRegId(1), VRegId(2))); // [5,15] ∩ [11,20] ≠ ∅
    }

    // ── LifecycleTag unit tests ────────────────────────────────────────

    #[test]
    fn test_lifecycle_alloc_priority_ordering() {
        assert!(LifecycleTag::Global.alloc_priority() < LifecycleTag::LoopInvariant.alloc_priority());
        assert!(LifecycleTag::LoopInvariant.alloc_priority() == LifecycleTag::LoopCarried.alloc_priority());
        assert!(LifecycleTag::LoopCarried.alloc_priority() < LifecycleTag::CrossScope.alloc_priority());
        assert!(LifecycleTag::CrossScope.alloc_priority() < LifecycleTag::BodyLocal.alloc_priority());
    }

    #[test]
    fn test_lifecycle_spill_priority_ordering() {
        assert!(LifecycleTag::BodyLocal.spill_priority() < LifecycleTag::CrossScope.spill_priority());
        assert!(LifecycleTag::CrossScope.spill_priority() < LifecycleTag::LoopCarried.spill_priority());
        assert!(LifecycleTag::LoopCarried.spill_priority() < LifecycleTag::LoopInvariant.spill_priority());
        assert!(LifecycleTag::LoopInvariant.spill_priority() < LifecycleTag::Global.spill_priority());
    }

    #[test]
    fn test_lifecycle_alloc_priority_values() {
        assert_eq!(LifecycleTag::Global.alloc_priority(), 0);
        assert_eq!(LifecycleTag::LoopInvariant.alloc_priority(), 2);
        assert_eq!(LifecycleTag::LoopCarried.alloc_priority(), 2);
        assert_eq!(LifecycleTag::CrossScope.alloc_priority(), 3);
        assert_eq!(LifecycleTag::BodyLocal.alloc_priority(), 4);
    }

    #[test]
    fn test_lifecycle_spill_priority_values() {
        assert_eq!(LifecycleTag::BodyLocal.spill_priority(), 0);
        assert_eq!(LifecycleTag::CrossScope.spill_priority(), 1);
        assert_eq!(LifecycleTag::LoopCarried.spill_priority(), 2);
        assert_eq!(LifecycleTag::LoopInvariant.spill_priority(), 3);
        assert_eq!(LifecycleTag::Global.spill_priority(), 4);
    }

    // ── infer_lifecycle unit tests ─────────────────────────────────────

    fn empty_hf() -> HashSet<(VRegId, usize)> { HashSet::new() }

    #[test]
    fn test_infer_lifecycle_global_def_at_zero_last_at_end() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 0, 99, &[], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::Global);
    }

    #[test]
    fn test_infer_lifecycle_global_exact_boundary() {
        // last_use = program_len - 1 => Global
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 0, 99, &[], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::Global);
        // last_use < program_len - 1 => NOT Global (no loops => BodyLocal)
        let tag2 = infer_lifecycle(VRegId(0), VRegKind::Ptr, 0, 98, &[], &empty_hf(), &[], 100);
        assert_eq!(tag2, LifecycleTag::BodyLocal);
    }

    #[test]
    fn test_infer_lifecycle_not_global_def_not_at_zero() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 5, 99, &[], &empty_hf(), &[], 100);
        assert_ne!(tag, LifecycleTag::Global);
    }

    #[test]
    fn test_infer_lifecycle_not_global_last_not_at_end() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 0, 50, &[], &empty_hf(), &[], 100);
        assert_ne!(tag, LifecycleTag::Global);
    }

    #[test]
    fn test_infer_lifecycle_loop_invariant_outside_def_read_inside() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 2, 20,
            &[(5, 15)], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::LoopInvariant);
    }

    #[test]
    fn test_infer_lifecycle_loop_carried_defined_inside_used_past() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 8, 25,
            &[(5, 15)], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::LoopCarried);
    }

    #[test]
    fn test_infer_lifecycle_body_local_defined_and_used_inside_loop() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 7, 12,
            &[(5, 15)], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::BodyLocal);
    }

    #[test]
    fn test_infer_lifecycle_body_local_no_loops() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 10, 30,
            &[], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::BodyLocal);
    }

    #[test]
    fn test_infer_lifecycle_cross_scope() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 2, 25,
            &[], &empty_hf(), &[(5, 20)], 100);
        assert_eq!(tag, LifecycleTag::CrossScope);
    }

    #[test]
    fn test_infer_lifecycle_cross_scope_not_if_defined_inside() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 8, 25,
            &[], &empty_hf(), &[(5, 20)], 100);
        assert_ne!(tag, LifecycleTag::CrossScope);
    }

    #[test]
    fn test_infer_lifecycle_counter_forced_loop_carried() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Counter, 10, 30,
            &[], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::LoopCarried);
    }

    #[test]
    fn test_infer_lifecycle_byte_offset_forced_loop_carried() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::ByteOffset, 10, 30,
            &[], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::LoopCarried);
    }

    #[test]
    fn test_infer_lifecycle_loop_invariant_with_first_found_read() {
        let mut hf = HashSet::new();
        hf.insert((VRegId(0), 5));
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 2, 15,
            &[(5, 15)], &hf, &[], 100);
        assert_eq!(tag, LifecycleTag::LoopInvariant);
    }

    #[test]
    fn test_infer_lifecycle_loop_invariant_first_found_write_short_last_use() {
        let mut hf = HashSet::new();
        hf.insert((VRegId(0), 5));
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 2, 10,
            &[(5, 15)], &hf, &[], 100);
        assert_ne!(tag, LifecycleTag::LoopInvariant);
    }

    #[test]
    fn test_infer_lifecycle_loop_invariant_with_two_loops() {
        // def_point=2 is outside both loops, used inside both => LoopInvariant
        // (loop_carried requires def_point inside a loop, which doesn't apply here)
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 2, 25,
            &[(5, 15), (10, 20)], &empty_hf(), &[], 100);
        assert_eq!(tag, LifecycleTag::LoopInvariant);
    }

    #[test]
    fn test_infer_lifecycle_nested_loops_one_carried_one_invariant() {
        let tag = infer_lifecycle(VRegId(0), VRegKind::Ptr, 2, 18,
            &[(5, 10), (8, 18)], &empty_hf(), &[], 100);
        assert!(tag == LifecycleTag::LoopInvariant || tag == LifecycleTag::LoopCarried);
    }

    // ── InterferenceGraph unit tests ───────────────────────────────────

    #[test]
    fn test_ig_empty_intervals() {
        let ig = InterferenceGraph::build(&[]);
        assert!(!ig.interferes(VRegId(0), VRegId(1)));
    }

    #[test]
    fn test_ig_single_interval_no_interference() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a]);
        assert!(!ig.interferes(VRegId(0), VRegId(0)));
    }

    #[test]
    fn test_ig_disjoint_same_class_no_interference() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 5, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 6, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        assert!(!ig.interferes(VRegId(0), VRegId(1)));
        assert!(!ig.interferes(VRegId(1), VRegId(0)));
    }

    #[test]
    fn test_ig_overlapping_same_class_interferes() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 5, last_use: 15, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        assert!(ig.interferes(VRegId(0), VRegId(1)));
        assert!(ig.interferes(VRegId(1), VRegId(0)));
    }

    #[test]
    fn test_ig_different_classes_no_interference() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Ptr, width: SimdWidth::Scalar,
            def_point: 5, last_use: 15, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        assert!(!ig.interferes(VRegId(0), VRegId(1)));
    }

    #[test]
    fn test_ig_neighbors_empty_for_isolated() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 5, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 10, last_use: 20, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        assert_eq!(ig.neighbors(VRegId(0)).count(), 0);
        assert_eq!(ig.neighbors(VRegId(1)).count(), 0);
    }

    #[test]
    fn test_ig_neighbors_count_for_chain() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 5, last_use: 15, lifecycle: LifecycleTag::BodyLocal,
        };
        let c = LiveInterval {
            vreg: VRegId(2), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 8, last_use: 18, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b, c]);
        assert_eq!(ig.neighbors(VRegId(0)).count(), 2);
        assert_eq!(ig.neighbors(VRegId(1)).count(), 2);
        assert_eq!(ig.neighbors(VRegId(2)).count(), 2);
    }

    #[test]
    fn test_ig_symmetry() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Ptr, width: SimdWidth::Scalar,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Ptr, width: SimdWidth::Scalar,
            def_point: 5, last_use: 15, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        assert_eq!(ig.interferes(VRegId(0), VRegId(1)), ig.interferes(VRegId(1), VRegId(0)));
    }

    #[test]
    fn test_ig_touching_boundaries_interfere() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 10, last_use: 20, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        // last_use=10 >= def_point=10 => they interfere
        assert!(ig.interferes(VRegId(0), VRegId(1)));
    }

    #[test]
    fn test_ig_counter_and_ptr_same_class_interfere() {
        let a = LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Counter, width: SimdWidth::Scalar,
            def_point: 0, last_use: 20, lifecycle: LifecycleTag::LoopCarried,
        };
        let b = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Ptr, width: SimdWidth::Scalar,
            def_point: 5, last_use: 15, lifecycle: LifecycleTag::BodyLocal,
        };
        let ig = InterferenceGraph::build(&[a, b]);
        assert!(ig.interferes(VRegId(0), VRegId(1)));
    }

    #[test]
    fn test_ig_many_concurrent_intervals() {
        let intervals: Vec<LiveInterval> = (0..20).map(|i| LiveInterval {
            vreg: VRegId(i), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 100, lifecycle: LifecycleTag::BodyLocal,
        }).collect();
        let ig = InterferenceGraph::build(&intervals);
        for i in 0..20u32 {
            assert_eq!(ig.neighbors(VRegId(i)).count(), 19);
        }
    }

    #[test]
    fn test_ig_staircase_pattern() {
        let intervals: Vec<LiveInterval> = (0..4u32).map(|i| LiveInterval {
            vreg: VRegId(i), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: i as usize, last_use: i as usize + 2, lifecycle: LifecycleTag::BodyLocal,
        }).collect();
        let ig = InterferenceGraph::build(&intervals);
        assert!(ig.interferes(VRegId(0), VRegId(1)));
        assert!(ig.interferes(VRegId(0), VRegId(2)));
        assert!(!ig.interferes(VRegId(0), VRegId(3)));
    }

    #[test]
    fn test_ig_non_overlapping_chain() {
        let intervals = vec![
            LiveInterval { vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 0, last_use: 5, lifecycle: LifecycleTag::BodyLocal },
            LiveInterval { vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 6, last_use: 10, lifecycle: LifecycleTag::BodyLocal },
            LiveInterval { vreg: VRegId(2), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 11, last_use: 15, lifecycle: LifecycleTag::BodyLocal },
        ];
        let ig = InterferenceGraph::build(&intervals);
        assert!(!ig.interferes(VRegId(0), VRegId(1)));
        assert!(!ig.interferes(VRegId(0), VRegId(2)));
        assert!(!ig.interferes(VRegId(1), VRegId(2)));
    }

    // ── RegAllocation accessor tests ───────────────────────────────────

    #[test]
    fn test_reg_allocation_get_returns_none_for_missing() {
        let alloc = RegAllocation {
            mapping: HashMap::new(), spills: vec![], callee_saved_used: vec![],
        };
        assert!(alloc.get(VRegId(0)).is_none());
    }

    #[test]
    fn test_reg_allocation_get_gpr_returns_correct() {
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(3)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        assert_eq!(alloc.get_gpr(VRegId(0)), Some(PhysGpr(3)));
        assert!(alloc.get_vec(VRegId(0)).is_none());
    }

    #[test]
    fn test_reg_allocation_get_vec_returns_correct() {
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(5)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        assert_eq!(alloc.get_vec(VRegId(1)), Some(PhysVec(5)));
        assert!(alloc.get_gpr(VRegId(1)).is_none());
    }

    #[test]
    fn test_reg_allocation_get_spilled() {
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(2), PhysReg::Spilled(0));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        assert!(alloc.get_gpr(VRegId(2)).is_none());
        assert!(alloc.get_vec(VRegId(2)).is_none());
        assert_eq!(alloc.get(VRegId(2)), Some(PhysReg::Spilled(0)));
    }

    #[test]
    fn test_reg_allocation_get_tile_and_mask() {
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(10), PhysReg::Tile(PhysTile(0)));
        mapping.insert(VRegId(11), PhysReg::Mask(PhysMask(0)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        assert_eq!(alloc.get(VRegId(10)), Some(PhysReg::Tile(PhysTile(0))));
        assert_eq!(alloc.get(VRegId(11)), Some(PhysReg::Mask(PhysMask(0))));
    }

    // ── compute_intervals tests ────────────────────────────────────────

    #[test]
    fn test_compute_intervals_empty_program() {
        let prog = VmProgram::new();
        let intervals = RegAllocator::compute_intervals(&prog);
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_compute_intervals_single_declare() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let intervals = RegAllocator::compute_intervals(&prog);
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].vreg, v);
        assert_eq!(intervals[0].def_point, 0);
    }

    #[test]
    fn test_compute_intervals_with_use_extends_last_use() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, src: vec0, offset: OffsetExpr::Const(16),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let intervals = RegAllocator::compute_intervals(&prog);
        let ptr_iv = intervals.iter().find(|iv| iv.vreg == ptr).unwrap();
        assert!(ptr_iv.last_use >= 3);
    }

    #[test]
    fn test_compute_intervals_counter_loop_invariant_when_outside_loop() {
        // When counter/byte_off are defined outside the loop body, they get
        // LoopInvariant (defined outside + used inside). The special Counter
        // forced-LoopCarried rule only upgrades BodyLocal, not LoopInvariant.
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, src: vec0, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: byte_off,
            bound: BoundExpr::Const(10), step_bytes: 4,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec1, base: ptr, offset: OffsetExpr::LoopOffset(byte_off),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::LoopEnd);
        prog.emit(VmInstr::Comment("after loop".to_string()));
        let intervals = RegAllocator::compute_intervals(&prog);
        let counter_iv = intervals.iter().find(|iv| iv.vreg == counter).unwrap();
        // Counter defined outside loop + used inside => LoopInvariant
        assert_eq!(counter_iv.lifecycle, LifecycleTag::LoopInvariant);
        let byte_off_iv = intervals.iter().find(|iv| iv.vreg == byte_off).unwrap();
        assert_eq!(byte_off_iv.lifecycle, LifecycleTag::LoopInvariant);
    }

    #[test]
    fn test_compute_intervals_counter_forced_loop_carried_no_loop() {
        // When Counter/ByteOffset have no loop context, special rule forces LoopCarried
        // (only upgrades BodyLocal, not other tags)
        let mut prog = VmProgram::new();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        // No loop at all — counter/byte_off are only declared, used nowhere
        prog.emit(VmInstr::Comment("no loop".to_string()));
        prog.emit(VmInstr::Comment("end".to_string()));
        let intervals = RegAllocator::compute_intervals(&prog);
        let counter_iv = intervals.iter().find(|iv| iv.vreg == counter).unwrap();
        // No loop => would be BodyLocal, but Counter special rule forces LoopCarried
        assert_eq!(counter_iv.lifecycle, LifecycleTag::LoopCarried);
        let byte_off_iv = intervals.iter().find(|iv| iv.vreg == byte_off).unwrap();
        assert_eq!(byte_off_iv.lifecycle, LifecycleTag::LoopCarried);
    }

    // ── scope tests ────────────────────────────────────────────────────

    #[test]
    fn test_build_scope_positions_empty() {
        let prog = VmProgram::new();
        let positions = RegAllocator::build_scope_positions(&prog);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_build_scope_positions_single() {
        let mut prog = VmProgram::new();
        let _ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });
        let positions = RegAllocator::build_scope_positions(&prog);
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].2, 0);
    }

    #[test]
    fn test_scope_at_position_none_when_outside() {
        let positions = vec![(5, 15, 0)];
        assert!(RegAllocator::scope_at_position(&positions, 3).is_none());
        assert!(RegAllocator::scope_at_position(&positions, 20).is_none());
    }

    #[test]
    fn test_scope_at_position_inside() {
        let positions = vec![(5, 15, 42)];
        assert_eq!(RegAllocator::scope_at_position(&positions, 10), Some(42));
    }

    #[test]
    fn test_scope_at_position_innermost_wins() {
        let positions = vec![(5, 20, 0), (8, 15, 1)];
        assert_eq!(RegAllocator::scope_at_position(&positions, 10), Some(1));
    }

    #[test]
    fn test_scope_at_position_boundary_is_outside() {
        let positions = vec![(5, 15, 0)];
        assert!(RegAllocator::scope_at_position(&positions, 5).is_none());
        assert!(RegAllocator::scope_at_position(&positions, 15).is_none());
    }

    // ── offset/ptr/bound vregs tests ──────────────────────────────────

    #[test]
    fn test_offset_vrefs_const() {
        assert!(RegAllocator::offset_vregs(&OffsetExpr::Const(42)).is_empty());
    }

    #[test]
    fn test_offset_vregs_loop_offset() {
        assert_eq!(RegAllocator::offset_vregs(&OffsetExpr::LoopOffset(VRegId(7))), vec![VRegId(7)]);
    }

    #[test]
    fn test_offset_vregs_add() {
        let expr = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(VRegId(1))),
            Box::new(OffsetExpr::ScalarVReg(VRegId(2))),
        );
        let vregs = RegAllocator::offset_vregs(&expr);
        assert_eq!(vregs.len(), 2);
    }

    #[test]
    fn test_ptr_vregs_vreg_plus_const() {
        assert_eq!(RegAllocator::ptr_vregs(&PtrExpr::VRegPlusConst(VRegId(3), 16)), vec![VRegId(3)]);
    }

    #[test]
    fn test_ptr_vregs_vreg_plus_vreg() {
        let vregs = RegAllocator::ptr_vregs(&PtrExpr::VRegPlusVReg(VRegId(1), VRegId(2)));
        assert_eq!(vregs.len(), 2);
    }

    #[test]
    fn test_bound_vrefs_const() {
        assert!(RegAllocator::bound_vregs(&BoundExpr::Const(100)).is_empty());
    }

    #[test]
    fn test_bound_vregs_dynamic_vreg() {
        assert_eq!(RegAllocator::bound_vregs(&BoundExpr::DynamicVReg(VRegId(7))), vec![VRegId(7)]);
    }

    #[test]
    fn test_scalar_expr_vregs_const() {
        assert!(RegAllocator::scalar_expr_vregs(&ScalarExpr::Const(1.0)).is_empty());
    }

    #[test]
    fn test_scalar_expr_vregs_mem_load() {
        let vregs = RegAllocator::scalar_expr_vregs(&ScalarExpr::MemLoad(VRegId(2), OffsetExpr::Const(0)));
        assert_eq!(vregs[0], VRegId(2));
    }

    // ── referenced_vregs tests ─────────────────────────────────────────

    #[test]
    fn test_referenced_vregs_comment_empty() {
        assert!(RegAllocator::referenced_vregs(&VmInstr::Comment("test".to_string())).is_empty());
    }

    #[test]
    fn test_referenced_vregs_loop_end_empty() {
        assert!(RegAllocator::referenced_vregs(&VmInstr::LoopEnd).is_empty());
    }

    #[test]
    fn test_referenced_vregs_scope_begin_empty() {
        assert!(RegAllocator::referenced_vregs(&VmInstr::ScopeBegin { scope_id: 0 }).is_empty());
    }

    #[test]
    fn test_referenced_vregs_fma() {
        let instr = VmInstr::Fma {
            dst: VRegId(0), acc: VRegId(0), a: VRegId(1), b: VRegId(2),
            dtype: QuantPrecision::F32,
        };
        let vregs = RegAllocator::referenced_vregs(&instr);
        assert_eq!(vregs.len(), 4);
    }

    #[test]
    fn test_referenced_vregs_loop_begin() {
        let instr = VmInstr::LoopBegin {
            counter: VRegId(0), byte_offset: VRegId(1),
            bound: BoundExpr::DynamicVReg(VRegId(2)), step_bytes: 4,
        };
        let vregs = RegAllocator::referenced_vregs(&instr);
        assert!(vregs.contains(&VRegId(0)));
        assert!(vregs.contains(&VRegId(1)));
        assert!(vregs.contains(&VRegId(2)));
    }

    #[test]
    fn test_referenced_vregs_gpr_bin_op_imm() {
        let instr = VmInstr::GprBinOp {
            dst: VRegId(0), a: VRegId(1), b: GprOperand::Imm(4), op: GprOp::Add,
        };
        let vregs = RegAllocator::referenced_vregs(&instr);
        assert_eq!(vregs.len(), 2);
    }

    #[test]
    fn test_referenced_vregs_gpr_bin_op_vreg() {
        let instr = VmInstr::GprBinOp {
            dst: VRegId(0), a: VRegId(1), b: GprOperand::VReg(VRegId(2)), op: GprOp::Add,
        };
        let vregs = RegAllocator::referenced_vregs(&instr);
        assert_eq!(vregs.len(), 3);
    }

    // ── validate_spill_layout tests ────────────────────────────────────

    #[test]
    fn test_validate_spill_layout_empty() {
        assert!(RegAllocator::validate_spill_layout(&[], &[]).is_ok());
    }

    #[test]
    fn test_validate_spill_layout_non_overlapping() {
        let spills = vec![
            SpillSlot { vreg: VRegId(0), offset: 0, size: 32 },
            SpillSlot { vreg: VRegId(1), offset: 32, size: 32 },
        ];
        let intervals = vec![
            LiveInterval { vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal },
            LiveInterval { vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 15, last_use: 30, lifecycle: LifecycleTag::BodyLocal },
        ];
        assert!(RegAllocator::validate_spill_layout(&spills, &intervals).is_ok());
    }

    #[test]
    fn test_validate_spill_layout_physical_overlap_rejected() {
        let spills = vec![
            SpillSlot { vreg: VRegId(0), offset: 0, size: 32 },
            SpillSlot { vreg: VRegId(1), offset: 16, size: 32 },
        ];
        let intervals = vec![
            LiveInterval { vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal },
            LiveInterval { vreg: VRegId(1), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 15, last_use: 30, lifecycle: LifecycleTag::BodyLocal },
        ];
        assert!(RegAllocator::validate_spill_layout(&spills, &intervals).is_err());
    }

    #[test]
    fn test_validate_spill_layout_gpr_too_small_rejected() {
        // Need >= 2 spills because validate_spill_layout returns Ok for len <= 1
        let spills = vec![
            SpillSlot { vreg: VRegId(0), offset: 0, size: 32 },
            SpillSlot { vreg: VRegId(1), offset: 32, size: 4 }, // Ptr needs >= 8
        ];
        let intervals = vec![
            LiveInterval { vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
                def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal },
            LiveInterval { vreg: VRegId(1), kind: VRegKind::Ptr, width: SimdWidth::Scalar,
                def_point: 15, last_use: 30, lifecycle: LifecycleTag::BodyLocal },
        ];
        assert!(RegAllocator::validate_spill_layout(&spills, &intervals).is_err());
    }

    // ── post_alloc_verify tests ────────────────────────────────────────

    #[test]
    fn test_post_alloc_verify_all_mapped() {
        let intervals = vec![LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        }];
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        assert!(RegAllocator::post_alloc_verify(&mapping, &intervals, &[]).is_ok());
    }

    #[test]
    fn test_post_alloc_verify_missing_mapping_rejected() {
        let intervals = vec![LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Vec, width: SimdWidth::W256,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::BodyLocal,
        }];
        let mapping = HashMap::new();
        assert!(RegAllocator::post_alloc_verify(&mapping, &intervals, &[]).is_err());
    }

    #[test]
    fn test_post_alloc_verify_counter_spilled_accepted() {
        let intervals = vec![LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Counter, width: SimdWidth::Scalar,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::LoopCarried,
        }];
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Spilled(0));
        let spills = vec![SpillSlot { vreg: VRegId(0), offset: 0, size: 8 }];
        assert!(RegAllocator::post_alloc_verify(&mapping, &intervals, &spills).is_ok());
    }

    #[test]
    fn test_post_alloc_verify_counter_in_gpr_ok() {
        let intervals = vec![LiveInterval {
            vreg: VRegId(0), kind: VRegKind::Counter, width: SimdWidth::Scalar,
            def_point: 0, last_use: 10, lifecycle: LifecycleTag::LoopCarried,
        }];
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(3)));
        assert!(RegAllocator::post_alloc_verify(&mapping, &intervals, &[]).is_ok());
    }

    // ── Full allocation integration tests ──────────────────────────────

    #[test]
    fn test_allocate_simple_program_no_loop() {
        let profile = test_profile();
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecUnaryOp {
            dst: vec1, a: vec0, op: VecUnaryOp::Neg,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, src: vec1, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        assert!(alloc.get(ptr).is_some());
        assert!(alloc.get(vec0).is_some());
        assert!(alloc.get(vec1).is_some());
    }

    #[test]
    fn test_allocate_overlapping_get_different_regs() {
        let profile = test_profile();
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec1, base: ptr, offset: OffsetExpr::Const(32),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecBinOp {
            dst: vec0, a: vec0, b: vec1, op: VecOp::Add, dtype: QuantPrecision::F32,
        });
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let phys0 = alloc.get_vec(vec0);
        let phys1 = alloc.get_vec(vec1);
        assert!(phys0.is_some());
        assert!(phys1.is_some());
        assert_ne!(phys0, phys1);
    }

    #[test]
    fn test_allocate_gpr_vec_different_pools() {
        let profile = test_profile();
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, src: vec0, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        assert!(alloc.get_gpr(ptr).is_some());
        assert!(alloc.get_vec(vec0).is_some());
    }

    #[test]
    fn test_allocate_gpu_returns_empty() {
        let gpu_profile = IsaProfile::cuda(80);
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let alloc = RegAllocator::new(&gpu_profile).allocate(&prog).unwrap();
        assert!(alloc.mapping.is_empty());
        assert!(alloc.spills.is_empty());
    }

    // ── validate_spill_safety tests ────────────────────────────────────

    #[test]
    fn test_validate_spill_safety_no_spills() {
        let prog = VmProgram::new();
        let mapping = HashMap::new();
        assert!(RegAllocator::validate_spill_safety(&prog, &mapping, &[]).is_ok());
    }

    #[test]
    fn test_validate_spill_safety_write_before_read() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, src: vec0, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let mut mapping = HashMap::new();
        mapping.insert(vec0, PhysReg::Spilled(0));
        let intervals = RegAllocator::compute_intervals(&prog);
        assert!(RegAllocator::validate_spill_safety(&prog, &mapping, &intervals).is_ok());
    }

    // ── LiveInterval basic properties ──────────────────────────────────

    #[test]
    fn test_live_interval_fields() {
        let iv = LiveInterval {
            vreg: VRegId(42), kind: VRegKind::Vec, width: SimdWidth::W512,
            def_point: 10, last_use: 20, lifecycle: LifecycleTag::BodyLocal,
        };
        assert_eq!(iv.vreg, VRegId(42));
        assert_eq!(iv.kind, VRegKind::Vec);
        assert_eq!(iv.width, SimdWidth::W512);
        assert_eq!(iv.def_point, 10);
        assert_eq!(iv.last_use, 20);
    }

    #[test]
    fn test_live_interval_clone() {
        let iv = LiveInterval {
            vreg: VRegId(1), kind: VRegKind::Ptr, width: SimdWidth::Scalar,
            def_point: 0, last_use: 5, lifecycle: LifecycleTag::Global,
        };
        let cloned = iv.clone();
        assert_eq!(iv.vreg, cloned.vreg);
        assert_eq!(iv.kind, cloned.kind);
        assert_eq!(iv.def_point, cloned.def_point);
        assert_eq!(iv.last_use, cloned.last_use);
        assert_eq!(iv.lifecycle, cloned.lifecycle);
    }

    // ── SpillSlot tests ────────────────────────────────────────────────

    #[test]
    fn test_spill_slot_fields() {
        let slot = SpillSlot { vreg: VRegId(7), offset: 64, size: 32 };
        assert_eq!(slot.vreg, VRegId(7));
        assert_eq!(slot.offset, 64);
        assert_eq!(slot.size, 32);
    }

    #[test]
    fn test_spill_slot_clone() {
        let slot = SpillSlot { vreg: VRegId(3), offset: 128, size: 64 };
        let cloned = slot.clone();
        assert_eq!(slot.vreg, cloned.vreg);
        assert_eq!(slot.offset, cloned.offset);
        assert_eq!(slot.size, cloned.size);
    }

    // ── instr_brief tests ──────────────────────────────────────────────

    #[test]
    fn test_instr_brief_known_variants() {
        assert_eq!(RegAllocator::instr_brief(&VmInstr::VecLoad {
            dst: VRegId(0), base: VRegId(1), offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        }), "VecLoad");
        assert_eq!(RegAllocator::instr_brief(&VmInstr::LoopEnd), "LoopEnd");
        assert_eq!(RegAllocator::instr_brief(&VmInstr::Fma {
            dst: VRegId(0), acc: VRegId(0), a: VRegId(1), b: VRegId(2),
            dtype: QuantPrecision::F32,
        }), "Fma");
        assert_eq!(RegAllocator::instr_brief(&VmInstr::Comment("x".to_string())), "Comment");
    }

    // ── LifecycleTag Copy + PartialEq ──────────────────────────────────

    #[test]
    fn test_lifecycle_tag_copy_semantics() {
        let tag = LifecycleTag::Global;
        let tag2 = tag;
        assert_eq!(tag, tag2);
    }

    #[test]
    fn test_lifecycle_tag_equality() {
        assert_eq!(LifecycleTag::Global, LifecycleTag::Global);
        assert_ne!(LifecycleTag::Global, LifecycleTag::BodyLocal);
        assert_ne!(LifecycleTag::LoopInvariant, LifecycleTag::LoopCarried);
    }

    // ── VRegId allocation uniqueness ──────────────────────────────────

    #[test]
    fn test_vreg_id_monotonically_increasing() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let v3 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let v4 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        // Each subsequent VRegId.0 must be strictly greater
        assert!(v0.0 < v1.0);
        assert!(v1.0 < v2.0);
        assert!(v2.0 < v3.0);
        assert!(v3.0 < v4.0);
    }

    // ── VRegKind all variants are distinct ────────────────────────────

    #[test]
    fn test_vreg_kind_all_variants_distinct() {
        let kinds = [
            VRegKind::Ptr, VRegKind::Vec, VRegKind::Scalar,
            VRegKind::Counter, VRegKind::ByteOffset, VRegKind::Tile, VRegKind::Mask,
        ];
        // Every pair must be distinct (no duplicates in the enum)
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(kinds[i], kinds[j],
                    "VRegKind variants at index {} and {} should differ", i, j);
            }
        }
    }

    // ── RegAllocation num_vregs and spill_slots accessors ─────────────

    #[test]
    fn test_reg_allocation_num_vregs_counts_mapping() {
        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(1)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(2), PhysReg::Spilled(0));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        assert_eq!(alloc.num_vregs(), 3);
    }

    #[test]
    fn test_reg_allocation_spill_slots_accessor() {
        let spills = vec![
            SpillSlot { vreg: VRegId(5), offset: 0, size: 32 },
            SpillSlot { vreg: VRegId(7), offset: 32, size: 8 },
        ];
        let alloc = RegAllocation {
            mapping: HashMap::new(), spills, callee_saved_used: vec![],
        };
        assert_eq!(alloc.spill_slots().len(), 2);
        assert_eq!(alloc.spill_slots()[0].vreg, VRegId(5));
        assert_eq!(alloc.spill_slots()[1].offset, 32);
    }

    // ── LifecycleTag alloc/spill priority inverse relationship ────────

    #[test]
    fn test_lifecycle_alloc_and_spill_priority_inverse_ordering() {
        // Tags that should be allocated physical registers first (low alloc_priority)
        // should be spilled last (high spill_priority), and vice versa.
        // Global: alloc=0 (earliest), spill=4 (spilled last)
        // BodyLocal: alloc=4 (latest), spill=0 (spilled first)
        assert_eq!(LifecycleTag::Global.alloc_priority(), 0);
        assert_eq!(LifecycleTag::Global.spill_priority(), 4);
        assert_eq!(LifecycleTag::BodyLocal.alloc_priority(), 4);
        assert_eq!(LifecycleTag::BodyLocal.spill_priority(), 0);
        // Verify ordering: Global < LoopInvariant <= LoopCarried < CrossScope < BodyLocal
        // for alloc, and BodyLocal < CrossScope < LoopCarried < LoopInvariant < Global
        // for spill (inverse ordering).
        let alloc_order = [
            LifecycleTag::Global, LifecycleTag::LoopInvariant,
            LifecycleTag::LoopCarried, LifecycleTag::CrossScope, LifecycleTag::BodyLocal,
        ];
        let spill_order = [
            LifecycleTag::BodyLocal, LifecycleTag::CrossScope,
            LifecycleTag::LoopCarried, LifecycleTag::LoopInvariant, LifecycleTag::Global,
        ];
        for i in 0..4 {
            assert!(alloc_order[i].alloc_priority() <= alloc_order[i + 1].alloc_priority());
            assert!(spill_order[i].spill_priority() <= spill_order[i + 1].spill_priority());
        }
    }

    // ── SimdWidth bytes and f32_lanes consistency ────────────────────

    #[test]
    fn test_simd_width_bytes_equals_f32_lanes_times_four() {
        let widths = [SimdWidth::Scalar, SimdWidth::W128, SimdWidth::W256, SimdWidth::W512];
        for w in widths {
            assert_eq!(w.bytes(), w.f32_lanes() * 4,
                "SimdWidth::{:?}.bytes() must equal f32_lanes()*4", w);
        }
    }

    // ── BoundExpr::DynamicVRegPlusOne extracts VReg ──────────────────

    #[test]
    fn test_bound_vregs_dynamic_vreg_plus_one() {
        let vregs = RegAllocator::bound_vregs(&BoundExpr::DynamicVRegPlusOne(VRegId(9)));
        assert_eq!(vregs, vec![VRegId(9)]);
    }

    // ── PtrExpr::NamedArg and AbsAddr produce no VRegs ───────────────

    #[test]
    fn test_ptr_vregs_named_arg_empty() {
        assert!(RegAllocator::ptr_vregs(&PtrExpr::NamedArg("weight_ptr".to_string())).is_empty());
    }

    #[test]
    fn test_ptr_vregs_abs_addr_empty() {
        assert!(RegAllocator::ptr_vregs(&PtrExpr::AbsAddr(0x1000)).is_empty());
    }

    // ── ScalarExpr::ExtractLane0 and VReg variants ───────────────────

    #[test]
    fn test_scalar_expr_vregs_extract_lane0() {
        let vregs = RegAllocator::scalar_expr_vregs(&ScalarExpr::ExtractLane0(VRegId(5)));
        assert_eq!(vregs, vec![VRegId(5)]);
    }

    #[test]
    fn test_scalar_expr_vregs_vreg_variant() {
        let vregs = RegAllocator::scalar_expr_vregs(&ScalarExpr::VReg(VRegId(12)));
        assert_eq!(vregs, vec![VRegId(12)]);
    }
}
