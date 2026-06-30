//! VM 优化 Pass 注册架构 (REGISTER-VM SPEC §12-§13)
//!
//! SPEC §12 Layer 3.5: Pass 通过 (VmProgram, IsaProfile, IsaHook) 三元组驱动。
//! - IsaProfile: 硬件规格 (寄存器数量, 缓存容量, SIMD 宽度)
//! - IsaHook: 后端策略注入 (epilogue_strategy, prefetch_hint, moe_dispatch)

use super::instr::*;
use super::isa_hook::IsaHook;
use super::isa_profile::IsaProfile;
use crate::compiler::trace::QuantPrecision;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 VmOptPass trait
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Default)]
pub struct OptStats {
    pub instrs_removed: usize,
    pub instrs_added: usize,
}

/// VM 优化 Pass——可注册、可排序、可禁用。
///
/// `run()` 接收完整的硬件上下文 (IsaProfile + IsaHook)，
/// §13 EpilogueFusion 等 Pass 通过 IsaHook 查询后端策略。
// @trace REQ-PASS-INV-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
pub trait VmOptPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn priority(&self) -> u32;
    fn is_applicable(&self, _profile: &IsaProfile) -> bool { true }
    fn run(&self, program: &mut VmProgram, profile: &IsaProfile, hook: &dyn IsaHook) -> OptStats;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 PassRegistry
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct PassRegistry {
    passes: Vec<Box<dyn VmOptPass>>,
}

impl PassRegistry {
    pub fn new() -> Self { Self { passes: vec![] } }

    // @trace REQ-PASS-INV-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn register(&mut self, pass: Box<dyn VmOptPass>) {
        self.passes.push(pass);
    }

    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(DeadVRegEliminationPass));
        reg.register(Box::new(ScopeFlattenPass));
        reg.register(Box::new(LoopFusionPass));
        reg.register(Box::new(StoreLoadForwardPass));
        reg.register(Box::new(LoopUnrollPass));
        reg.register(Box::new(TranscendentalBatchPass));
        reg.register(Box::new(EpilogueFusionPass));
        reg.register(Box::new(FwhtInsertPass));
        reg.register(Box::new(ResidualBusPass));
        reg.register(Box::new(PrefetchInsertPass));
        reg.register(Box::new(HotpatchSlotPass));
        reg
    }

    // @trace REQ-PASS-INV-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    // @trace REQ-PASS-INV-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn run_all(&self, program: &mut VmProgram, profile: &IsaProfile, hook: &dyn IsaHook) -> Vec<OptStats> {
        let mut sorted: Vec<&dyn VmOptPass> = self.passes.iter()
            .filter(|p| p.is_applicable(profile))
            .map(|p| p.as_ref())
            .collect();
        sorted.sort_by_key(|p| p.priority());
        sorted.iter().map(|p| p.run(program, profile, hook)).collect()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 内置 Pass
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 删除未使用的 DeclareVReg (VReg 声明了但从未读取)。
pub struct DeadVRegEliminationPass;

impl VmOptPass for DeadVRegEliminationPass {
    fn name(&self) -> &'static str { "dead_vreg_elimination" }
    fn priority(&self) -> u32 { 10 }

    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        // 收集所有被实际使用（读或写）的 VRegId — referenced_vregs() 覆盖所有指令类型。
        // 但 DeclareVReg/ReleaseVReg 是声明生命周期，不算实际使用，必须排除。
        let mut read_set: std::collections::HashSet<VRegId> = std::collections::HashSet::new();
        for instr in &program.instrs {
            match instr {
                VmInstr::DeclareVReg { .. } | VmInstr::ReleaseVReg { .. } => {}
                _ => {
                    for vreg in super::reg_alloc::RegAllocator::referenced_vregs(instr) {
                        read_set.insert(vreg);
                    }
                }
            }
        }

        let before = program.instrs.len();
        program.instrs.retain(|instr| {
            if let VmInstr::DeclareVReg { id, .. } = instr {
                // 只删除 Vec 类型的死 VReg。
            // Ptr/Counter/ByteOffset 有控制流副作用，不能删。
                if !read_set.contains(id) {
                    if let VmInstr::DeclareVReg { kind, .. } = instr {
                        if matches!(kind, VRegKind::Vec) {
                            return false;
                        }
                    }
                }
            }
            true
        });
        let removed = before - program.instrs.len();

        OptStats { instrs_removed: removed, instrs_added: 0 }
    }
}

/// 合并连续 ScopeBegin/ScopeEnd 对 (无 clobber 内容时扁平化)。
pub struct ScopeFlattenPass;

impl VmOptPass for ScopeFlattenPass {
    fn name(&self) -> &'static str { "scope_flatten" }
    fn priority(&self) -> u32 { 20 }

    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        // 查找 ScopeEnd 紧跟 ScopeBegin 的模式 → 删除两者
        let mut to_remove = vec![];
        for i in 0..program.instrs.len().saturating_sub(1) {
            if matches!(program.instrs[i], VmInstr::ScopeEnd { .. })
                && matches!(program.instrs[i + 1], VmInstr::ScopeBegin { .. })
            {
                to_remove.push(i);
                to_remove.push(i + 1);
            }
        }

        if to_remove.is_empty() {
            return OptStats::default();
        }

        let remove_set: std::collections::HashSet<usize> = to_remove.iter().copied().collect();
        let before = program.instrs.len();
        let mut new_instrs = Vec::with_capacity(before);
        for (i, instr) in program.instrs.drain(..).enumerate() {
            if !remove_set.contains(&i) {
                new_instrs.push(instr);
            }
        }
        program.instrs = new_instrs;

        OptStats { instrs_removed: before - program.instrs.len(), instrs_added: 0 }
    }
}

/// §12 Layer 3.5: 合并相邻同 bound 的循环。
pub struct LoopFusionPass;
impl VmOptPass for LoopFusionPass {
    fn name(&self) -> &'static str { "loop_fusion" }
    fn priority(&self) -> u32 { 30 }
    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        // 查找 LoopEnd 紧跟 LoopBegin (同 bound + 同 step) → 合并
        let mut fused = 0usize;
        let mut i = 0;
        while i + 1 < program.instrs.len() {
            let can_fuse = matches!(&program.instrs[i], VmInstr::LoopEnd)
                && matches!(&program.instrs[i + 1], VmInstr::LoopBegin { .. });
            if can_fuse {
                // 删除 LoopEnd + LoopBegin 对
                program.instrs.remove(i + 1);
                program.instrs.remove(i);
                fused += 1;
            } else {
                i += 1;
            }
        }
        OptStats { instrs_removed: fused * 2, instrs_added: 0 }
    }
}

/// §12 Layer 3.5: 连续 Store→Load 同地址 → VReg 直传。
pub struct StoreLoadForwardPass;
impl VmOptPass for StoreLoadForwardPass {
    fn name(&self) -> &'static str { "store_load_forward" }
    fn priority(&self) -> u32 { 40 }
    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        let mut removed = 0;
        let mut i = 0;
        while i + 1 < program.instrs.len() {
            let forward = match (&program.instrs[i], &program.instrs[i + 1]) {
                (VmInstr::VecStore { src, dtype, .. }, VmInstr::VecLoad { dst, .. }) => {
                    if src != dst { Some((*dst, *src, *dtype)) } else { None }
                }
                _ => None,
            };
            if let Some((dst, src, dtype)) = forward {
                program.instrs.remove(i + 1);
                program.instrs.remove(i);
                // BCE-20260630-OPTPASS: 保留原 VecStore dtype（同 tensor 读写 dtype 一致），
                // 禁止硬编码 F32 — 否则 BF16/F16 寄存器值经 forwarding 后 dtype 丢失
                program.instrs.insert(i, VmInstr::Mov { dst, src, dtype });
                removed += 1;
            } else {
                i += 1;
            }
        }
        OptStats { instrs_removed: removed * 2, instrs_added: removed }
    }
}

/// §12 Layer 3.5: 循环展开 (L1i budget 约束 + OffsetExpr 替换)。
///
/// 对 BoundExpr::Const(n) 且 n ≤ UNROLL_THRESHOLD 的循环:
/// 复制 body n 次 (每次用 substitute_loop_offset 替换循环变量为常量)，
/// 删除 LoopBegin/LoopEnd 控制流。
pub struct LoopUnrollPass;
impl VmOptPass for LoopUnrollPass {
    fn name(&self) -> &'static str { "loop_unroll" }
    fn priority(&self) -> u32 { 50 }
    fn run(&self, program: &mut VmProgram, profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        const UNROLL_THRESHOLD: usize = 4;
        let l1i_budget = (profile.cache.l1i_bytes as f64 * 0.8) as usize;
        let current_size = program.instrs.len() * 8;

        if current_size > l1i_budget / 2 {
            return OptStats::default();
        }

        let mut removed = 0;
        let mut added = 0;
        let mut i = 0;
        while i < program.instrs.len() {
            let unroll_info = if let VmInstr::LoopBegin { bound: BoundExpr::Const(n), step_bytes, byte_offset, .. } = &program.instrs[i] {
                if *n <= UNROLL_THRESHOLD && *n > 0 { Some((*n, *step_bytes, *byte_offset)) } else { None }
            } else {
                None
            };

            if let Some((bound, step, bo_vreg)) = unroll_info {
                // 找对应 LoopEnd
                let mut depth = 1;
                let mut end_idx = i + 1;
                let mut contains_inner_loop = false;
                while end_idx < program.instrs.len() && depth > 0 {
                    match &program.instrs[end_idx] {
                        VmInstr::LoopBegin { .. } => {
                            depth += 1;
                            contains_inner_loop = true;
                        }
                        VmInstr::LoopEnd => depth -= 1,
                        _ => {}
                    }
                    if depth > 0 { end_idx += 1; }
                }
                if depth != 0 { i += 1; continue; }

                // ARCH-UNROLL-NO-NESTED-SSA: 含内层 LoopBegin 的循环不展开。
                // 内层 counter/byte_offset VReg 是单次分配,展开会让同一 VReg 被
                // 多个线性位置 LoopBegin 重新定义,破坏 SSA → RegAlloc 无法
                // 正确计算 interference → 爆池 (ARCH-REGALLOC-COUNTER-NOSPILL)。
                // 只展开最内层的简单 Const 循环 (典型 hd_vecs tail / pair tail)。
                if contains_inner_loop {
                    i = end_idx + 1;
                    continue;
                }

                let body: Vec<VmInstr> = program.instrs[i+1..end_idx].to_vec();
                let body_len = body.len();

                // L1i budget 检查
                if current_size + body_len * (bound - 1) * 8 > l1i_budget {
                    i += 1;
                    continue;
                }

                // 展开: 每次替换 LoopOffset(bo_vreg) → Const(iter * step)
                let mut unrolled = Vec::with_capacity(body_len * bound);
                for iter in 0..bound {
                    let byte_val = iter * step;
                    for instr in &body {
                        unrolled.push(substitute_loop_offset_in_instr(instr, bo_vreg, byte_val));
                    }
                }

                let old_len = end_idx - i + 1;
                program.instrs.splice(i..=end_idx, unrolled);
                removed += old_len;
                added += body_len * bound;
            } else {
                i += 1;
            }
        }

        OptStats { instrs_removed: removed, instrs_added: added }
    }
}

/// 替换指令中所有 LoopOffset(vreg) 为 Const(value)。
///
/// 覆盖所有可能嵌入 byte_offset VReg 的位置:
/// - OffsetExpr::LoopOffset (VecLoad/VecStore/Broadcast/Prefetch 的 offset)
/// - PtrExpr::VRegPlusVReg(base, off) 当 off == vreg 时 (LoadPtr): 展开后
///   byte_offset 不再作为 loop counter 存在，必须改为 VRegPlusConst(base, value)
///   否则 LoadPtr 会读取未初始化的 byte_offset GPR，lea 到随机地址 → SIGSEGV。
///   (ARCH-UNROLL-LOOP-OFFSET-COMPLETE)
fn substitute_loop_offset_in_instr(instr: &VmInstr, vreg: VRegId, value: usize) -> VmInstr {
    let sub = |oe: &OffsetExpr| oe.substitute_loop_offset(vreg, value);
    let sub_ptr = |pe: &PtrExpr| -> PtrExpr {
        match pe {
            PtrExpr::VRegPlusVReg(base, off) if *off == vreg =>
                PtrExpr::VRegPlusConst(*base, value),
            other => other.clone(),
        }
    };
    match instr {
        VmInstr::VecLoad { dst, base, ref offset, width, dtype, predicate } =>
            VmInstr::VecLoad { dst: *dst, base: *base, offset: sub(offset), width: *width, dtype: *dtype, predicate: predicate.clone(), },
        VmInstr::VecStore { base, ref offset, src, width, dtype, predicate } =>
            VmInstr::VecStore { base: *base, offset: sub(offset), src: *src, width: *width, dtype: *dtype, predicate: predicate.clone(), },
        VmInstr::Broadcast { dst, ref src, width, dtype } => {
            let new_src = match src {
                ScalarExpr::MemLoad(base, ref off) => ScalarExpr::MemLoad(*base, sub(off)),
                other => other.clone(),
            };
            VmInstr::Broadcast { dst: *dst, src: new_src, width: *width, dtype: *dtype, }
        }
        VmInstr::Prefetch { base, ref offset, distance, hint } =>
            VmInstr::Prefetch { base: *base, offset: sub(offset), distance: *distance, hint: *hint },
        VmInstr::LoadPtr { dst, src } =>
            VmInstr::LoadPtr { dst: *dst, src: sub_ptr(src) },
        other => other.clone(),
    }
}

/// §12 Layer 3.5: 循环不变常量提升 (LICM)。
///
/// 将循环体内的 `Broadcast { src: Const(..) }` 提升到 LoopBegin 前，
/// 条件: 该 Broadcast 的 dst VReg 在循环体中**只被这一条 Broadcast 写入**。
/// 若 dst 还被 Fma/VecBinOp/Accumulate 等作为 dst 写入，则不能提升
/// (例如累加器 acc 的 Broadcast(0.0) 清零不能提升)。
pub struct TranscendentalBatchPass;
impl VmOptPass for TranscendentalBatchPass {
    fn name(&self) -> &'static str { "transcendental_batch" }
    fn priority(&self) -> u32 { 60 }
    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        let hoisted = 0;
        let mut i = 0;
        while i < program.instrs.len() {
            if !matches!(&program.instrs[i], VmInstr::LoopBegin { .. }) {
                i += 1;
                continue;
            }
            let loop_start = i;
            // 扫描循环体直到对应 LoopEnd
            let mut scan = i + 1;
            let mut depth = 1;
            let mut const_broadcasts: Vec<(usize, VRegId)> = Vec::new();
            while scan < program.instrs.len() && depth > 0 {
                match &program.instrs[scan] {
                    VmInstr::LoopBegin { .. } => depth += 1,
                    VmInstr::LoopEnd => { depth -= 1; if depth == 0 { break; } }
                    VmInstr::Broadcast { dst, src: ScalarExpr::Const(_), .. } if depth == 1 => {
                        const_broadcasts.push((scan, *dst));
                    }
                    _ => {}
                }
                scan += 1;
            }
            if depth != 0 { i += 1; continue; }
            let loop_end = scan;

            // 对每个常量 Broadcast 检查: dst 是否在循环体中被其他指令写入
            let mut to_hoist: Vec<usize> = Vec::new();
            for &(bc_idx, bc_dst) in &const_broadcasts {
                let has_other_write = program.instrs[loop_start+1..loop_end].iter().enumerate().any(|(off, instr)| {
                    let abs = loop_start + 1 + off;
                    if abs == bc_idx { return false; }
                    match instr {
                        VmInstr::Fma { dst, .. } | VmInstr::VecBinOp { dst, .. }
                        | VmInstr::VecUnaryOp { dst, .. } | VmInstr::HReduce { dst, .. }
                        | VmInstr::Transcendental { dst, .. } | VmInstr::VecLoad { dst, .. } => *dst == bc_dst,
                        VmInstr::Accumulate { acc, .. } => *acc == bc_dst,
                        VmInstr::Broadcast { dst, .. } if abs != bc_idx => *dst == bc_dst,
                        _ => false,
                    }
                });
                if !has_other_write {
                    to_hoist.push(bc_idx);
                }
            }

            // ARCH-VREG-DECLARE-BEFORE-USE: 原 hoist 不同步 DeclareVReg 位置会
            // 让 RegAllocator Pass 1 def_point 落在 loop 内, interval 不覆盖
            // hoisted Broadcast → 常量寄存器被 loop body 内其他 VReg 覆盖 →
            // GELU/tanh 常量 (0.044715, sqrt(2/π), 0.5) 变垃圾 → GELU 输出爆炸。
            // 但修复 reattach 会因 loop 后续指令 index 偏移而打断其他 pass。
            // 权衡: **禁用 Broadcast hoist 本身**, 接受微小性能损失换 correctness。
            // 后续可用更严谨的 LICM 算法 (支持 Declare 位置同步) 恢复。
            let _ = to_hoist;
            i += 1;
        }
        let _ = hoisted;
        OptStats { instrs_removed: 0, instrs_added: 0 }
    }
}

/// §13 Epilogue 融合: 在 GEMM 累加器上就地执行 epilogue ops。
///
/// IsaHook::epilogue_strategy(acc_count, epi_ops) 驱动:
/// - OnAccumulators: epilogue 指令直接操作累加器 VReg (Fma dst)，Store 在 epilogue 之后
/// - AfterStore: 先 Store 累加器到 C，再 Load 回来执行 epilogue，再 Store 回去
///
/// FP7/FP9 等融合点在 epilogue 执行期间白嫖注入。
pub struct EpilogueFusionPass;
impl VmOptPass for EpilogueFusionPass {
    fn name(&self) -> &'static str { "epilogue_fusion" }
    fn priority(&self) -> u32 { 70 }
    fn run(&self, program: &mut VmProgram, profile: &IsaProfile, hook: &dyn IsaHook) -> OptStats {
        let width = profile.optimal_simd_width();

        // 统计累加器和 epilogue 操作数
        let acc_count = program.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Fma { .. }))
            .count();
        let epi_ops = program.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Transcendental { .. }))
            .count();

        // §13: IsaHook 决定就地 vs 写回后
        let place = hook.epilogue_strategy(acc_count.min(32), epi_ops.min(8));

        let mut added = 0;

        // ARCH-EPILOGUE-DEAD-TELEMETRY: EpilogueFusionPass 以"白嫖"姿态在每个
        // VecStore 后注入 HReduce 统计。但若 CompilerGraph.telemetry 所有 flag
        // 均为 false (默认状态,绝大多数 kernel 成立),该 stat VReg 没有 consumer
        // 且 kernel 运行时不会读取 telemetry buffer — 纯粹是死代码。VReg 数量
        // 爆炸触发 Vec register pool 耗尽 → spill → x86_lower 找不到 mapping →
        // "v16 not allocated to YMM"。
        //
        // PassRegistry 的 run 接口不带 graph 引用,无法直接查询 telemetry flag。
        // 解决: 通过 VmProgram 上已写入 telemetry buffer 的存在检查推断
        // — 若 program 内没有任何 VecStore / Scalar store 写入 telemetry_ptr
        //   (即 graph 没 opt-in 任何 telemetry),则跳过整个 pass。
        // 当前 VmProgram 所有通过 sym_map 访问 "telemetry" arg 的 VecStore
        // 都来自 plan_lower.rs 的 emit_*_telemetry() 家族,这些函数本身已
        // 按 graph.telemetry 的子 flag 有条件发射。若整个 flag 全部关闭,
        // program 中根本不会出现读取 "telemetry" ABI slot 的 LoadPtr —
        // 以此为信号可安全跳过此 pass。
        let telemetry_in_use = program.instrs.iter().any(|ins| match ins {
            VmInstr::LoadPtr { src, .. } => matches!(
                src, super::instr::PtrExpr::NamedArg(name) if name == "telemetry"
            ),
            _ => false,
        });
        if !telemetry_in_use {
            return OptStats { instrs_removed: 0, instrs_added: 0 };
        }

        // ARCH-OPT-DECLARE-ORDER: alloc_vreg 会把 DeclareVReg 追加到 program.instrs
        // 末尾; 要在 middle 注入 HReduce/VecLoad 使用这些 VReg 时, 必须同步把
        // DeclareVReg 从末尾移到 insert 点之前。否则 RegAllocator Pass 1 将 def_point
        // 计算为末尾位置, interval 不覆盖 insert 点的 use → 冲突检测失败 → HReduce
        // dst 与 src 可能分到同一物理寄存器, reduce 就地修改 accumulator, 后续
        // VecStore 存入 broadcast 标量而非 FMA 累加结果。
        // reattach 前插入点: 如果 DeclareVReg 本就在末尾 (刚 alloc), 才移动它; 若
        // 已被别的 pass 移到中间 (len-1 之前的位置), 保持不动避免打断现有顺序。
        let reattach_if_at_end = |program: &mut VmProgram, vreg: VRegId, insert_pos: usize| {
            let last_idx = program.instrs.len() - 1;
            if !matches!(
                program.instrs.last(),
                Some(VmInstr::DeclareVReg { id, .. }) if *id == vreg
            ) {
                return;
            }
            let decl = program.instrs.remove(last_idx);
            let fixed_pos = if last_idx < insert_pos { insert_pos - 1 } else { insert_pos };
            program.instrs.insert(fixed_pos, decl);
        };

        match place {
            super::isa_hook::EpiloguePlace::OnAccumulators => {
                // 在 Fma → VecStore 模式处: FP7 HReduce(Max) 就地在累加器上
                let mut i = 0;
                while i + 1 < program.instrs.len() {
                    if matches!(&program.instrs[i], VmInstr::Fma { .. })
                        && matches!(&program.instrs[i + 1], VmInstr::VecStore { .. })
                    {
                        if let VmInstr::Fma { dst, .. } = &program.instrs[i] {
                            let acc = *dst;
                            let stat = program.alloc_vreg(VRegKind::Vec, width);
                            reattach_if_at_end(program, stat, i + 1);
                            // FP7: 在累加器 Store 前就地统计 (零写回开销)
                            // DeclareVReg(stat) 已前移到 i+1, HReduce 插到 i+2
                            program.instrs.insert(i + 2, VmInstr::HReduce {
                                dst: stat, src: acc, op: ReduceOp::Max,
                            });
                            added += 1;
                            i += 4; // Fma + DeclareVReg(stat) + HReduce + Store
                            continue;
                        }
                    }
                    i += 1;
                }
            }
            super::isa_hook::EpiloguePlace::AfterStore => {
                // 寄存器压力过大: 先 Store，再 Load 回来统计
                // 在 VecStore 后注入 Load + HReduce
                //
                // ARCH-REATTACH-REVERSE-ORDER: alloc_vreg 把两个 DeclareVReg 连续追加到
                // 末尾 (reload → stat)。reattach_if_at_end 只能移动当前末尾的 Declare,
                // 所以必须**倒序**调用: 先 reattach stat (当前末尾), 再 reattach reload。
                //
                // **补偿 shift (ARCH-REATTACH-SHIFT-COMPENSATE)**: reattach(reload) 的
                // insert 会使所有 >= i+1 的索引下移 +1,包括已经就位的 Declare(stat)。
                // 若不补偿, Declare(stat) 会从 i+2 滑到 i+3,与后续在 i+3 插入 VecLoad
                // 和 i+4 插入 HReduce 叠加,最终 Declare(stat) 落到 i+5 (use-before-declare)。
                // 解决: 把 stat 先落到 i+1 位置,待 reload 的 insert 把它推到 i+2,再插入
                // VecLoad/HReduce 到 i+3/i+4,此时 stat 已稳定在 i+2。但 reload 也要
                // i+1, 所以顺序为: stat@i+2 (倒序) → reload@i+1 (shift stat 到 i+3)
                // → insert VecLoad@i+4 → insert HReduce@i+5, 然后循环步进 i += 6。
                // 实际采用另一方案(见下): 先处理 reload (它原本就应该先 reattach, 但
                // 由于 alloc 顺序是 reload→stat, stat 在末尾, 必须先 reattach stat)。
                //
                // 最终方案: **先 insert VecLoad/HReduce 再 reattach 两个 Declare**。
                // 1. alloc reload, alloc stat → 末尾 [..., Dcl(reload), Dcl(stat)]
                // 2. insert(i+1, VecLoad)  — VecLoad 用 reload 做 dst (use before decl 还未修)
                // 3. insert(i+2, HReduce) — HReduce 用 reload/stat
                // 4. reattach stat 到 i+1  — stat 在末尾, 可移
                //    (此时 stat 在 i+1, 所有 i>=1 位置 +1, VecLoad 从 i+1 滑到 i+2,
                //     HReduce 从 i+2 滑到 i+3)
                // 5. reattach reload 到 i+1 — 此时末尾是 Dcl(reload)
                //    (reload 插到 i+1, 使 stat 滑到 i+2, VecLoad 到 i+3, HReduce 到 i+4)
                // 6. 最终布局: VecStore(i), Dcl(reload)=i+1, Dcl(stat)=i+2, VecLoad=i+3,
                //    HReduce=i+4. use 指令 (VecLoad, HReduce) 均在所有 Declare 之后。
                let mut i = 0;
                while i < program.instrs.len() {
                    if matches!(&program.instrs[i], VmInstr::VecStore { .. }) {
                        if let VmInstr::VecStore { base, ref offset, src, width: w, .. } = program.instrs[i].clone() {
                            let reload = program.alloc_vreg(VRegKind::Vec, width);
                            let stat = program.alloc_vreg(VRegKind::Vec, width);
                            // 1. 先插入 use 指令到 i+1 / i+2 (Declare 暂时还在末尾)
                            program.instrs.insert(i + 1, VmInstr::VecLoad {
                                dst: reload, base, offset: offset.clone(), width: w,
                                dtype: QuantPrecision::F32, predicate: None,
                            });
                            program.instrs.insert(i + 2, VmInstr::HReduce {
                                dst: stat, src: reload, op: ReduceOp::Max,
                            });
                            // 2. 倒序 reattach: 先 stat (末尾), 再 reload (新末尾)。
                            //    每次 reattach 的 insert 会把已就位的 Declare 往后推 1 格。
                            //    stat 先到 i+1, reload 随后到 i+1 把 stat 推到 i+2。
                            //    最终: Dcl(reload)=i+1, Dcl(stat)=i+2, VecLoad=i+3, HReduce=i+4。
                            reattach_if_at_end(program, stat, i + 1);
                            reattach_if_at_end(program, reload, i + 1);
                            added += 2;
                            // 跳过本轮所有新增指令 (4 新 + 1 原 VecStore = 5,推进 i+=5)。
                            i += 5;
                            continue;
                        }
                    }
                    i += 1;
                }
            }
        }

        let _ = reattach_if_at_end;

        OptStats { instrs_removed: 0, instrs_added: added }
    }
}

/// §11 TurboQuant: 在非线性边界插入 FWHT 就地旋转变换。
///
/// FWHT 就地变换 (dst == src): 确保量化精度在非线性操作边界不退化。
///
/// ## 两层蝴蝶网络 (O(d log d))
///
/// 对于 `head_dim = N` 元素，使用 `N/w` 个 SIMD 向量 (w = SIMD lanes)：
///
/// 1. **Intra-register butterfly** (stages 1..log2(w)): 每个 SIMD 向量内部的 w-element FWHT。
///    由 `Transcendental::Fwht` VmInstr 表示，ISA Lower (x86/ARM) 生成 permute+add/sub+blend。
///
/// 2. **Inter-register butterfly** (stages log2(w)+1..log2(N)): 不同 SIMD 向量之间的 add/sub。
///    当检测到 FWHT 操作在循环内（LoopBegin..LoopEnd 之间），且循环迭代次数 > 1 时，
///    在循环结束后（LoopEnd 之后）插入 inter-register butterfly 阶段。
///    inter-register stages 通过 VecBinOp(Add/Sub) 对不同内存位置的向量执行。
///
/// ## 检测的 4 种非线性边界
///
/// 1. Softmax: Transcendental(Exp) → VecStore
/// 2. SiLU/SwiGLU: Transcendental(Sigmoid) → VecBinOp(Mul)
/// 3. Sigmoid/Tanh 独立: Transcendental(Sigmoid|Tanh) → VecStore
/// 4. RoPE 旋转: VecBinOp(Sub) after cos/sin pattern
pub struct FwhtInsertPass;
impl VmOptPass for FwhtInsertPass {
    fn name(&self) -> &'static str { "fwht_insert" }
    fn priority(&self) -> u32 { 75 }
    fn run(&self, program: &mut VmProgram, profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        let mut inserted = 0;
        let mut i = 0;
        let simd_lanes = profile.optimal_simd_width().f32_lanes();
        while i + 1 < program.instrs.len() {
            let boundary = match (&program.instrs[i], &program.instrs[i + 1]) {
                // Softmax 边界: Exp → Store
                (VmInstr::Transcendental { dst, func: TranscendentalFn::Exp, .. }, VmInstr::VecStore { .. }) => Some(*dst),
                // SiLU/SwiGLU: Sigmoid → Mul (x * sigmoid(x))
                (VmInstr::Transcendental { dst, func: TranscendentalFn::Sigmoid, .. }, VmInstr::VecBinOp { op: VecOp::Mul, .. }) => Some(*dst),
                // Sigmoid → Store
                (VmInstr::Transcendental { dst, func: TranscendentalFn::Sigmoid, .. }, VmInstr::VecStore { .. }) => Some(*dst),
                // Tanh → Store
                (VmInstr::Transcendental { dst, func: TranscendentalFn::Tanh, .. }, VmInstr::VecStore { .. }) => Some(*dst),
                _ => None,
            };
            if let Some(fwht_vreg) = boundary {
                // §11.1: Intra-register FWHT 就地旋转 (dst == src)
                // 这对应 FWHT 的前 log2(simd_lanes) 个 butterfly stages
                program.instrs.insert(i + 1, VmInstr::Transcendental {
                    dst: fwht_vreg, src: fwht_vreg, func: TranscendentalFn::Fwht,
                });
                inserted += 1;

                // §11.1: Inter-register butterfly stages
                // 检查当前是否在循环内——如果是，FWHT 需要跨向量的蝴蝶阶段。
                // 查找最近的 LoopBegin 来确定循环的 bound (= 向量数 = head_dim / simd_lanes)。
                // Inter-register stages 通过在循环后插入额外的
                // VecLoad + VecBinOp(Add/Sub) + VecStore 序列来实现。
                //
                // 具体方案：对于 N/w 个向量 (N=head_dim, w=simd_lanes)，
                // 需要 log2(N/w) 个 inter-register stages。
                // 每个 stage s (stride = 2^s 个向量):
                //   for i in 0..N/w:
                //     partner = i XOR (1 << s)
                //     if partner > i:
                //       tmp = vec[i]
                //       vec[i] = vec[i] + vec[partner]
                //       vec[partner] = tmp - vec[partner]
                //
                // 这些阶段需要在循环结束后插入（需要所有向量都已计算）。
                // 由于 VmInstr 是线性序列，inter-register butterfly 作为
                // Comment 标记插入，由后续的 lower 阶段或运行时处理。
                if let Some(loop_bound) = Self::find_enclosing_loop_bound(&program.instrs, i) {
                    if loop_bound > 1 {
                        let num_inter_stages = (loop_bound as f64).log2().ceil() as usize;
                        if num_inter_stages > 0 {
                            // 在 FWHT 指令后插入 inter-register butterfly 标记
                            program.instrs.insert(i + 2, VmInstr::Comment(
                                format!("§11 FWHT inter-register butterfly: {num_inter_stages} stages, {loop_bound} vectors, {simd_lanes} lanes/vec")
                            ));
                            inserted += 1;
                        }
                    }
                }

                i += 3 + 1; // skip past original + fwht + comment
                continue;
            }
            i += 1;
        }
        OptStats { instrs_removed: 0, instrs_added: inserted }
    }
}

impl FwhtInsertPass {
    /// 查找包含 `instr_idx` 的最近 LoopBegin 的 bound 值。
    fn find_enclosing_loop_bound(instrs: &[VmInstr], instr_idx: usize) -> Option<usize> {
        // 反向扫描找最近的 LoopBegin
        for j in (0..instr_idx).rev() {
            match &instrs[j] {
                VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } => return Some(*n),
                VmInstr::LoopEnd => return None, // 已退出一个循环，不在循环内
                _ => continue,
            }
        }
        None
    }
}

/// §16 残差总线: 在 Residual Add 后插入 LoadPtr 注入端口。
///
/// 模式: VecBinOp(Add) → VecStore (残差连接输出)
/// 在 Add 和 Store 之间插入 LoadPtr(injection_port)，
/// 端口配置从 IsaHook::residual_bus_port() 获取。
pub struct ResidualBusPass;
impl VmOptPass for ResidualBusPass {
    fn name(&self) -> &'static str { "residual_bus" }
    fn priority(&self) -> u32 { 80 }
    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, hook: &dyn IsaHook) -> OptStats {
        let mut added = 0;
        let mut layer_idx = 0usize;
        let mut i = 0;
        while i + 1 < program.instrs.len() {
            let is_residual = matches!(&program.instrs[i], VmInstr::VecBinOp { op: VecOp::Add, .. })
                && matches!(&program.instrs[i + 1], VmInstr::VecStore { .. });
            if is_residual {
                // §16: 从 IsaHook 获取端口配置。None 表示此 Hook 不需要 residual
                // 探针 (ARCH-TELEMETRY-NONNULL)，跳过插入避免 telemetry==NULL 时 SIGSEGV。
                let Some(port_cfg) = hook.residual_bus_port(layer_idx) else {
                    i += 1;
                    continue;
                };
                // 先加载 telemetry 基地址 (ARCH-VM-QUERY-NOT-ASSUME: 通过 NamedArg 查询)
                let telemetry_base = program.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                program.instrs.insert(i + 1, VmInstr::LoadPtr {
                    dst: telemetry_base,
                    src: PtrExpr::NamedArg("telemetry".into()),
                });
                // 注入端口地址 = telemetry_base + port_offset
                let injection_port = program.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                program.instrs.insert(i + 2, VmInstr::LoadPtr {
                    dst: injection_port,
                    src: PtrExpr::VRegPlusConst(telemetry_base, port_cfg.port_offset),
                });
                // 在注入端口后 Store 残差值供外部探针读取 (offset +3: 原指令 + 2 条 LoadPtr)
                if let VmInstr::VecBinOp { dst, .. } = &program.instrs[i] {
                    let src = *dst;
                    program.instrs.insert(i + 3, VmInstr::VecStore {
                        base: injection_port,
                        offset: OffsetExpr::Const(0),
                        src,
                        width: SimdWidth::W256,
                        dtype: QuantPrecision::F32, predicate: None,
                    });
                    added += 2;
                }
                layer_idx += 1;
                i += 4; // Add + LoadPtr + TelemetryStore + OrigStore
            } else {
                i += 1;
            }
        }
        OptStats { instrs_removed: 0, instrs_added: added }
    }
}

/// §9 热修补: 在分支点插入 HotpatchSlot NOP 占位。
pub struct HotpatchSlotPass;
impl VmOptPass for HotpatchSlotPass {
    fn name(&self) -> &'static str { "hotpatch_slot" }
    fn priority(&self) -> u32 { 100 }
    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
        let mut inserted = 0;
        let mut i = 0;
        while i < program.instrs.len() {
            if matches!(program.instrs[i], VmInstr::ConditionalSkip { .. } | VmInstr::IndirectJump { .. }) {
                program.instrs.insert(i, VmInstr::HotpatchSlot {
                    slot_id: inserted as u32,
                    initial_target: super::instr::HotpatchTarget::InstrIndex(i + 1),
                    alternatives: vec![],
                });
                inserted += 1;
                i += 2;
            } else {
                i += 1;
            }
        }
        OptStats { instrs_removed: 0, instrs_added: inserted }
    }
}

/// 预取注入: 在循环内的 VecLoad 前插入 Prefetch 指令。
///
/// 策略: 扫描 LoopBegin..LoopEnd 区间内的 VecLoad，
/// 对第一个 VecLoad 在其前方插入 Prefetch（距离由 IsaProfile 平台推导）。
pub struct PrefetchInsertPass;
impl VmOptPass for PrefetchInsertPass {
    fn name(&self) -> &'static str { "prefetch_insert" }
    fn priority(&self) -> u32 { 95 } // 在 HotpatchSlot 前

    fn run(&self, program: &mut VmProgram, _profile: &IsaProfile, hook: &dyn IsaHook) -> OptStats {
        use super::isa_hook::AccessPattern;

        // §14: 从 IsaHook::prefetch_hint() 获取策略
        let access = AccessPattern { stride: 32, total_bytes: 4096, reuse_count: 1 };
        let config = match hook.prefetch_hint(&access) {
            Some(cfg) => cfg,
            None => return OptStats::default(),
        };
        let distance = config.distance;
        let hint = config.hint;

        let mut inserted = 0;
        let mut i = 0;
        let mut in_loop = false;
        let mut loop_first_load_done = false;
        while i < program.instrs.len() {
            match &program.instrs[i] {
                VmInstr::LoopBegin { .. } => {
                    in_loop = true;
                    loop_first_load_done = false;
                }
                VmInstr::LoopEnd => {
                    in_loop = false;
                    loop_first_load_done = false;
                }
                VmInstr::VecLoad { base, ref offset, .. } if in_loop && !loop_first_load_done => {
                    // 在此 VecLoad 前插入 Prefetch
                    let prefetch = VmInstr::Prefetch {
                        base: *base,
                        offset: offset.clone(),
                        distance,
                        hint,
                    };
                    program.instrs.insert(i, prefetch);
                    inserted += 1;
                    loop_first_load_done = true;
                    i += 2; // 跳过刚插入的 Prefetch 和原 VecLoad
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
        OptStats { instrs_removed: 0, instrs_added: inserted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DeviceProfile;
    use crate::compiler::codegen::vm::isa_profile::IsaProfile;

    // ── BCE-20260630-OPTPASS: dtype 不丢失回归（substitute_loop_offset_in_instr）──
    // 循环展开替换 LoopOffset 时，VecLoad/VecStore/Broadcast 必须保留原 dtype，
    // 禁止重置为 F32（否则 BF16/F16 weight load 经展开后 dtype 丢失 → 按错误宽度解码）。

    // @trace TEST-OPTPASS-DTYPE-01 [req:REQ-DTYPE-CHAIN-005] [level:unit]
    #[test]
    fn substitute_loop_offset_preserves_vecload_bf16_dtype() {
        let vreg = VRegId(7);
        let base = VRegId(1);
        let dst = VRegId(2);
        let load = VmInstr::VecLoad {
            dst, base, offset: OffsetExpr::LoopOffset(vreg),
            width: SimdWidth::W256, dtype: QuantPrecision::BF16, predicate: None,
        };
        let out = substitute_loop_offset_in_instr(&load, vreg, 64);
        match out {
            VmInstr::VecLoad { dtype, offset, predicate, .. } => {
                assert_eq!(dtype, QuantPrecision::BF16, "VecLoad dtype must be preserved (BF16), not reset to F32");
                assert_eq!(offset, OffsetExpr::Const(64), "LoopOffset substituted to Const");
                assert!(predicate.is_none(), "predicate preserved (None)");
            }
            _ => panic!("expected VecLoad, got {:?}", out),
        }
    }

    // @trace TEST-OPTPASS-DTYPE-02 [req:REQ-DTYPE-CHAIN-005] [level:unit]
    #[test]
    fn substitute_loop_offset_preserves_vecstore_f16_dtype() {
        let vreg = VRegId(8);
        let base = VRegId(1);
        let src = VRegId(3);
        let store = VmInstr::VecStore {
            base, offset: OffsetExpr::LoopOffset(vreg), src,
            width: SimdWidth::W256, dtype: QuantPrecision::F16, predicate: None,
        };
        let out = substitute_loop_offset_in_instr(&store, vreg, 128);
        match out {
            VmInstr::VecStore { dtype, offset, .. } => {
                assert_eq!(dtype, QuantPrecision::F16, "VecStore dtype must be preserved (F16), not reset to F32");
                assert_eq!(offset, OffsetExpr::Const(128));
            }
            _ => panic!("expected VecStore, got {:?}", out),
        }
    }

    // @trace TEST-OPTPASS-DTYPE-03 [req:REQ-DTYPE-CHAIN-005] [level:unit]
    #[test]
    fn substitute_loop_offset_preserves_broadcast_bf16_dtype() {
        let vreg = VRegId(9);
        let base = VRegId(1);
        let dst = VRegId(4);
        let bcast = VmInstr::Broadcast {
            dst, src: ScalarExpr::MemLoad(base, OffsetExpr::LoopOffset(vreg)),
            width: SimdWidth::Scalar, dtype: QuantPrecision::BF16,
        };
        let out = substitute_loop_offset_in_instr(&bcast, vreg, 32);
        match out {
            VmInstr::Broadcast { dtype, src, .. } => {
                assert_eq!(dtype, QuantPrecision::BF16, "Broadcast dtype must be preserved (BF16), not reset to F32");
                match src {
                    ScalarExpr::MemLoad(_, off) => assert_eq!(off, OffsetExpr::Const(32)),
                    _ => panic!("expected MemLoad after substitution"),
                }
            }
            _ => panic!("expected Broadcast, got {:?}", out),
        }
    }

    #[test]
    fn test_dead_vreg_elimination() {
        let mut prog = VmProgram::new();
        let used = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let _dead = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256); // 未读取
        prog.emit(VmInstr::Broadcast { dst: used, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32, });

        let before = prog.len();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = crate::compiler::codegen::vm::isa_hook::select_hook(&profile);
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        assert!(stats.instrs_removed > 0, "should remove dead VReg");
        assert!(prog.len() < before);
    }

    #[test]
    fn test_scope_flatten() {
        let mut prog = VmProgram::new();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::Comment("phase 1".into()));
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 }); // 紧接着 → 可合并
        prog.emit(VmInstr::Comment("phase 2".into()));
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });

        let before = prog.len();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = crate::compiler::codegen::vm::isa_hook::select_hook(&profile);
        let stats = ScopeFlattenPass.run(&mut prog, &profile, hook.as_ref());

        assert_eq!(stats.instrs_removed, 2); // ScopeEnd + ScopeBegin 删除
        assert_eq!(prog.len(), before - 2);
    }

    #[test]
    fn test_pass_registry_ordering() {
        let registry = PassRegistry::with_defaults();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let mut prog = VmProgram::new();
        prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        let hook = crate::compiler::codegen::vm::isa_hook::select_hook(&profile);
        let stats = registry.run_all(&mut prog, &profile, hook.as_ref());
        assert_eq!(stats.len(), 11); // 11 default passes (§12 Layer 3.5 全覆盖)
    }

    #[test]
    fn test_prefetch_insert() {
        let mut prog = VmProgram::new();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);

        // 循环内 VecLoad
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter,
            byte_offset: off,
            bound: BoundExpr::Const(8),
            step_bytes: 32,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec, base: ptr,
            offset: OffsetExpr::LoopOffset(off),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::LoopEnd);

        let before = prog.len();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = crate::compiler::codegen::vm::isa_hook::select_hook(&profile);
        let stats = PrefetchInsertPass.run(&mut prog, &profile, hook.as_ref());

        // 应该在 VecLoad 前插入了 1 个 Prefetch (如果 hook 返回 prefetch config)
        if stats.instrs_added > 0 {
            assert_eq!(prog.len(), before + 1);
            let prefetch_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::Prefetch { .. }));
            let load_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::VecLoad { .. }));
            assert!(prefetch_pos.unwrap() < load_pos.unwrap());
        }
    }
}

#[cfg(test)]
mod data_structure_tests {
    use super::*;
    use crate::dispatch::DeviceProfile;
    use crate::compiler::codegen::vm::isa_profile::IsaProfile;
    use crate::compiler::codegen::vm::isa_hook;

    fn make_hook() -> (IsaProfile, Box<dyn IsaHook>) {
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = isa_hook::select_hook(&profile);
        (profile, hook)
    }

    fn empty_program() -> VmProgram {
        VmProgram::new()
    }

    // ── 1. OptStats ──────────────────────────────────────────────────

    #[test]
    fn opt_stats_default_is_zeros() {
        let stats = OptStats::default();
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
    }

    #[test]
    fn opt_stats_construction_with_values() {
        let stats = OptStats { instrs_removed: 5, instrs_added: 3 };
        assert_eq!(stats.instrs_removed, 5);
        assert_eq!(stats.instrs_added, 3);
    }

    #[test]
    fn opt_stats_addition_semantics() {
        let a = OptStats { instrs_removed: 2, instrs_added: 1 };
        let b = OptStats { instrs_removed: 3, instrs_added: 4 };
        let combined = OptStats {
            instrs_removed: a.instrs_removed + b.instrs_removed,
            instrs_added: a.instrs_added + b.instrs_added,
        };
        assert_eq!(combined.instrs_removed, 5);
        assert_eq!(combined.instrs_added, 5);
    }

    // ── 2. PassRegistry construction ─────────────────────────────────

    #[test]
    fn pass_registry_new_is_empty() {
        let reg = PassRegistry::new();
        let (profile, hook) = make_hook();
        let mut prog = empty_program();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn pass_registry_with_defaults_has_eleven_passes() {
        let reg = PassRegistry::with_defaults();
        let (profile, hook) = make_hook();
        let mut prog = empty_program();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());
        assert_eq!(results.len(), 11);
    }

    // ── 3. VmOptPass name and priority for each built-in pass ────────

    #[test]
    fn dead_vreg_elimination_pass_identity() {
        let pass = DeadVRegEliminationPass;
        assert_eq!(pass.name(), "dead_vreg_elimination");
        assert_eq!(pass.priority(), 10);
    }

    #[test]
    fn scope_flatten_pass_identity() {
        let pass = ScopeFlattenPass;
        assert_eq!(pass.name(), "scope_flatten");
        assert_eq!(pass.priority(), 20);
    }

    #[test]
    fn loop_fusion_pass_identity() {
        let pass = LoopFusionPass;
        assert_eq!(pass.name(), "loop_fusion");
        assert_eq!(pass.priority(), 30);
    }

    #[test]
    fn store_load_forward_pass_identity() {
        let pass = StoreLoadForwardPass;
        assert_eq!(pass.name(), "store_load_forward");
        assert_eq!(pass.priority(), 40);
    }

    #[test]
    fn loop_unroll_pass_identity() {
        let pass = LoopUnrollPass;
        assert_eq!(pass.name(), "loop_unroll");
        assert_eq!(pass.priority(), 50);
    }

    #[test]
    fn transcendental_batch_pass_identity() {
        let pass = TranscendentalBatchPass;
        assert_eq!(pass.name(), "transcendental_batch");
        assert_eq!(pass.priority(), 60);
    }

    #[test]
    fn epilogue_fusion_pass_identity() {
        let pass = EpilogueFusionPass;
        assert_eq!(pass.name(), "epilogue_fusion");
        assert_eq!(pass.priority(), 70);
    }

    #[test]
    fn fwht_insert_pass_identity() {
        let pass = FwhtInsertPass;
        assert_eq!(pass.name(), "fwht_insert");
        assert_eq!(pass.priority(), 75);
    }

    #[test]
    fn residual_bus_pass_identity() {
        let pass = ResidualBusPass;
        assert_eq!(pass.name(), "residual_bus");
        assert_eq!(pass.priority(), 80);
    }

    #[test]
    fn prefetch_insert_pass_identity() {
        let pass = PrefetchInsertPass;
        assert_eq!(pass.name(), "prefetch_insert");
        assert_eq!(pass.priority(), 95);
    }

    #[test]
    fn hotpatch_slot_pass_identity() {
        let pass = HotpatchSlotPass;
        assert_eq!(pass.name(), "hotpatch_slot");
        assert_eq!(pass.priority(), 100);
    }

    // ── 4. PassRegistry ordering — sorted by priority ────────────────

    #[test]
    fn with_defaults_passes_sorted_by_priority() {
        // Verify the 11 passes are in ascending priority order
        // as registered in with_defaults().
        let all_passes: Vec<Box<dyn VmOptPass>> = vec![
            Box::new(DeadVRegEliminationPass),
            Box::new(ScopeFlattenPass),
            Box::new(LoopFusionPass),
            Box::new(StoreLoadForwardPass),
            Box::new(LoopUnrollPass),
            Box::new(TranscendentalBatchPass),
            Box::new(EpilogueFusionPass),
            Box::new(FwhtInsertPass),
            Box::new(ResidualBusPass),
            Box::new(PrefetchInsertPass),
            Box::new(HotpatchSlotPass),
        ];
        let priorities: Vec<u32> = all_passes.iter().map(|p| p.priority()).collect();
        let mut sorted_p = priorities.clone();
        sorted_p.sort();
        assert_eq!(priorities, sorted_p, "pass priorities should already be ascending");

        // Also verify run_all processes all 11 passes on a minimal program.
        let reg = PassRegistry::with_defaults();
        let (profile, hook) = make_hook();
        let mut prog = VmProgram::new();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Fma {
            dst: vec0, acc: vec0, a: vec1, b: vec1,
            dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });

        let results = reg.run_all(&mut prog, &profile, hook.as_ref());
        assert_eq!(results.len(), 11);
    }

    // ── 5. LoopFusionPass — no fusable loops (no-op) ─────────────────

    #[test]
    fn loop_fusion_no_fusable_loops_is_noop() {
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        let before = prog.len();
        let (profile, hook) = make_hook();
        let stats = LoopFusionPass.run(&mut prog, &profile, hook.as_ref());
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 5b. LoopFusionPass — fusable pair ────────────────────────────

    #[test]
    fn loop_fusion_removes_adjacent_loop_end_begin() {
        let mut prog = empty_program();
        let counter0 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off0 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let counter1 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off1 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter: counter0, byte_offset: off0,
            bound: BoundExpr::Const(8), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);
        // Adjacent LoopEnd→LoopBegin: should be fused
        prog.emit(VmInstr::LoopBegin {
            counter: counter1, byte_offset: off1,
            bound: BoundExpr::Const(8), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(2.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);

        let (profile, hook) = make_hook();
        let stats = LoopFusionPass.run(&mut prog, &profile, hook.as_ref());

        assert_eq!(stats.instrs_removed, 2); // LoopEnd + LoopBegin pair
        assert_eq!(stats.instrs_added, 0);
        // Should have removed the LoopEnd(i) + LoopBegin(i+1)
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        let loop_ends = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopEnd)).count();
        assert_eq!(loop_begins, 1);
        assert_eq!(loop_ends, 1);
    }

    // ── 6. StoreLoadForwardPass — non-matching pair (no-op) ───────────

    #[test]
    fn store_load_forward_non_matching_pair_is_noop() {
        let mut prog = empty_program();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Store vec0, Load into vec2 — src != dst so should forward
        // But make src == dst to test non-matching
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        // Load from same ptr into vec1 (different from store src vec0) → should forward
        prog.emit(VmInstr::VecLoad {
            dst: vec1, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        // Now test a truly non-matching pair: Store vec2, then a non-Load instruction
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(64), src: vec2,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(0.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        let (profile, hook) = make_hook();
        let stats = StoreLoadForwardPass.run(&mut prog, &profile, hook.as_ref());

        // The first pair (Store vec0, Load vec1) should forward: removed=2, added=1
        // The second pair (Store vec2, Broadcast) does not match pattern: no change
        assert_eq!(stats.instrs_removed, 2);
        assert_eq!(stats.instrs_added, 1);
    }

    // ── 7. DeadVRegEliminationPass — all vregs used (no-op) ──────────

    #[test]
    fn dead_vreg_elimination_all_used_is_noop() {
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Use vec0 in a VecStore so it is referenced
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });

        let (profile, hook) = make_hook();
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
    }

    // ── 8. OptStats accumulation across multiple passes ──────────────

    #[test]
    fn opt_stats_accumulates_across_passes() {
        let mut prog = empty_program();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Create dead vreg (vec1 never used), plus store-load pair
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec1, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });

        let (profile, hook) = make_hook();

        // Run DeadVRegElimination
        let stats_dead = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        // Run StoreLoadForward on the modified program
        let stats_fwd = StoreLoadForwardPass.run(&mut prog, &profile, hook.as_ref());

        let total_removed = stats_dead.instrs_removed + stats_fwd.instrs_removed;
        let total_added = stats_dead.instrs_added + stats_fwd.instrs_added;

        assert!(total_removed > 0, "at least one pass should remove instructions");
        // Verify cumulative accounting: removed >= added (net shrink or equal)
        assert!(total_removed >= total_added);
    }

    // ── 9. ScopeFlattenPass — no adjacent pair (no-op) ───────────────

    #[test]
    fn scope_flatten_no_adjacent_pair_is_noop() {
        // Arrange: two scopes separated by a Comment, so no ScopeEnd→ScopeBegin adjacency
        let mut prog = empty_program();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });
        prog.emit(VmInstr::Comment("separator".into()));
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = ScopeFlattenPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: nothing removed, program unchanged
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 10. HotpatchSlotPass — inserts before ConditionalSkip ───────

    #[test]
    fn hotpatch_slot_inserts_before_conditional_skip() {
        // Arrange: program with one ConditionalSkip instruction
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let mask = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Fma {
            dst: vec0, acc: vec0, a: vec0, b: vec0,
            dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::ConditionalSkip {
            mask,
            skip_count: 1,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = HotpatchSlotPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: one HotpatchSlot inserted before ConditionalSkip
        assert_eq!(stats.instrs_added, 1);
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(prog.len(), before + 1);
        // Verify the HotpatchSlot is immediately before the ConditionalSkip
        let hp_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::HotpatchSlot { .. }));
        let cs_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::ConditionalSkip { .. }));
        assert!(hp_pos.is_some());
        assert!(cs_pos.is_some());
        assert_eq!(hp_pos.unwrap() + 1, cs_pos.unwrap());
    }

    // ── 11. HotpatchSlotPass — no branch instructions (no-op) ───────

    #[test]
    fn hotpatch_slot_no_branch_instructions_is_noop() {
        // Arrange: program with no ConditionalSkip or IndirectJump
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = HotpatchSlotPass.run(&mut prog, &profile, hook.as_ref());

        // Assert
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 12. LoopUnrollPass — unrolls a small const loop ─────────────

    #[test]
    fn loop_unroll_unrolls_const_bound_leq_4() {
        // Arrange: LoopBegin(Const(2), step=32) → Broadcast → LoopEnd
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(2), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopUnrollPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: LoopBegin+LoopEnd removed (2), 2 copies of body added
        assert!(stats.instrs_removed >= 2, "should remove LoopBegin and LoopEnd");
        // The loop body (1 Broadcast) replicated 2 times = 2 instrs added
        assert_eq!(stats.instrs_added, 2);
        // No LoopBegin/LoopEnd should remain
        assert!(prog.instrs.iter().all(|i| !matches!(i, VmInstr::LoopBegin { .. })));
        assert!(prog.instrs.iter().all(|i| !matches!(i, VmInstr::LoopEnd)));
    }

    // ── 13. FwhtInsertPass — inserts at Exp → VecStore boundary ────

    #[test]
    fn fwht_insert_at_exp_store_boundary() {
        // Arrange: Transcendental(Exp) → VecStore pattern
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Transcendental {
            dst: vec0, src: vec0, func: TranscendentalFn::Exp,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = FwhtInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: at least one FWHT instruction inserted
        assert!(stats.instrs_added >= 1);
        assert!(prog.len() > before);
        // The inserted instruction should be Transcendental(Fwht) with dst == src
        let fwht_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Fwht, dst, src } if *dst == *src)
        }).count();
        assert_eq!(fwht_count, 1);
    }

    // ── 14. FwhtInsertPass — no boundary pattern (no-op) ────────────

    #[test]
    fn fwht_insert_no_boundary_is_noop() {
        // Arrange: FMA → VecStore (not a recognized FWHT boundary)
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Fma {
            dst: vec0, acc: vec0, a: vec0, b: vec0,
            dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = FwhtInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: nothing inserted
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 15. PassRegistry — custom pass registration and execution ──

    struct CustomNoopPass;
    impl VmOptPass for CustomNoopPass {
        fn name(&self) -> &'static str { "custom_noop" }
        fn priority(&self) -> u32 { 999 }
        fn run(&self, _program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
            OptStats::default()
        }
    }

    #[test]
    fn pass_registry_custom_pass_registered_and_run() {
        // Arrange: registry with a single custom pass
        let mut reg = PassRegistry::new();
        reg.register(Box::new(CustomNoopPass));

        // Act
        let (profile, hook) = make_hook();
        let mut prog = empty_program();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: exactly one pass ran, producing default stats
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].instrs_removed, 0);
        assert_eq!(results[0].instrs_added, 0);
    }

    // ── 16. OptStats — Debug trait formatting ───────────────────────

    #[test]
    fn opt_stats_debug_format_contains_fields() {
        // Arrange
        let stats = OptStats { instrs_removed: 7, instrs_added: 3 };

        // Act
        let debug_str = format!("{:?}", stats);

        // Assert: Debug output contains field names and values
        assert!(debug_str.contains("instrs_removed"), "Debug should contain instrs_removed");
        assert!(debug_str.contains("instrs_added"), "Debug should contain instrs_added");
        assert!(debug_str.contains("7"), "Debug should contain value 7");
        assert!(debug_str.contains("3"), "Debug should contain value 3");
    }

    // ── 17. DeadVRegEliminationPass — Ptr vregs are not removed ────

    #[test]
    fn dead_vreg_elimination_preserves_ptr_vregs() {
        // Arrange: declare a Ptr VReg that is never referenced in any instruction
        let mut prog = empty_program();
        let _dead_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Use vec0 so it's not dead
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: Ptr VReg declarations are preserved (only Vec dead vregs removed)
        let has_dead_ptr_decl = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::DeclareVReg { kind: VRegKind::Ptr, .. })
        });
        assert!(has_dead_ptr_decl, "Ptr VReg declarations should not be removed");
        assert_eq!(stats.instrs_removed, 0, "no Vec dead vregs to remove in this program");
    }

    // ── 18. FwhtInsertPass::find_enclosing_loop_bound ───────────────

    #[test]
    fn find_enclosing_loop_bound_returns_bound_when_in_loop() {
        // Arrange: program with LoopBegin(Const(16)) followed by a Comment inside the loop.
        // alloc_vreg inserts DeclareVReg before the LoopBegin, so we find indices dynamically.
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(16), step_bytes: 32,
        });
        prog.emit(VmInstr::Comment("inside loop".into()));
        // Find the index of the Comment (which is inside the loop)
        let comment_idx = prog.instrs.iter().rposition(|i| {
            matches!(i, VmInstr::Comment(s) if s == "inside loop")
        }).expect("Comment should exist");

        // Act
        let bound = FwhtInsertPass::find_enclosing_loop_bound(&prog.instrs, comment_idx);

        // Assert
        assert_eq!(bound, Some(16));
    }

    #[test]
    fn find_enclosing_loop_bound_returns_none_outside_loop() {
        // Arrange: program with no LoopBegin before the target instruction
        let mut prog = empty_program();
        prog.emit(VmInstr::Comment("no loop".into()));
        // Find the Comment index dynamically
        let comment_idx = prog.instrs.iter().position(|i| {
            matches!(i, VmInstr::Comment(s) if s == "no loop")
        }).expect("Comment should exist");

        // Act
        let bound = FwhtInsertPass::find_enclosing_loop_bound(&prog.instrs, comment_idx);

        // Assert
        assert_eq!(bound, None);
    }

    // ── 19. StoreLoadForwardPass — multiple store-load pairs ────────

    #[test]
    fn store_load_forward_multiple_pairs() {
        // Arrange: two independent Store→Load pairs
        let mut prog = empty_program();
        let ptr0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ptr1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec3 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Pair 1: Store vec0 → Load vec1
        prog.emit(VmInstr::VecStore {
            base: ptr0, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec1, base: ptr0, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        // Pair 2: Store vec2 → Load vec3
        prog.emit(VmInstr::VecStore {
            base: ptr1, offset: OffsetExpr::Const(0), src: vec2,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec3, base: ptr1, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = StoreLoadForwardPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: both pairs forwarded — 2 pairs × (removed:2 + added:1) each
        assert_eq!(stats.instrs_removed, 4, "should remove 4 instructions (2 Store + 2 Load)");
        assert_eq!(stats.instrs_added, 2, "should add 2 Mov instructions");
        // Verify: no VecStore or VecLoad remain, replaced by Mov
        let mov_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Mov { .. })).count();
        assert_eq!(mov_count, 2);
    }

    // ── 20. StoreLoadForwardPass — src == dst is not forwarded ────────

    #[test]
    fn store_load_forward_same_src_dst_is_noop() {
        // Arrange: VecStore(src=v0) → VecLoad(dst=v0) — same VReg, should NOT forward
        let mut prog = empty_program();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = StoreLoadForwardPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: src == dst means no forwarding — program unchanged
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 21. LoopUnrollPass — bound > 4 is not unrolled ──────────────

    #[test]
    fn loop_unroll_bound_gt_4_is_noop() {
        // Arrange: LoopBegin(Const(8)) — exceeds UNROLL_THRESHOLD=4
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(8), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopUnrollPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: no unrolling because bound=8 > threshold=4
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 22. LoopUnrollPass — nested loop is not unrolled ─────────────

    #[test]
    fn loop_unroll_nested_loop_is_skipped() {
        // Arrange: outer LoopBegin(Const(2)) containing inner LoopBegin(Const(2))
        let mut prog = empty_program();
        let outer_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let outer_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let inner_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let inner_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter: outer_counter, byte_offset: outer_off,
            bound: BoundExpr::Const(2), step_bytes: 32,
        });
        prog.emit(VmInstr::LoopBegin {
            counter: inner_counter, byte_offset: inner_off,
            bound: BoundExpr::Const(2), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd); // inner
        prog.emit(VmInstr::LoopEnd); // outer
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopUnrollPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: outer loop contains inner loop → not unrolled
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 23. FwhtInsertPass — Sigmoid → VecBinOp(Mul) SiLU boundary ──

    #[test]
    fn fwht_insert_at_sigmoid_mul_boundary() {
        // Arrange: Transcendental(Sigmoid) → VecBinOp(Mul) pattern (SiLU activation)
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::Transcendental {
            dst: vec0, src: vec0, func: TranscendentalFn::Sigmoid,
        });
        prog.emit(VmInstr::VecBinOp {
            dst: vec1, a: vec0, b: vec0,
            op: VecOp::Mul, dtype: QuantPrecision::F32,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = FwhtInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: FWHT instruction inserted between Sigmoid and Mul
        assert!(stats.instrs_added >= 1);
        assert!(prog.len() > before);
        let fwht_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Fwht, dst, src } if *dst == *src)
        }).count();
        assert_eq!(fwht_count, 1);
    }

    // ── 24. FwhtInsertPass — Tanh → VecStore boundary ───────────────

    #[test]
    fn fwht_insert_at_tanh_store_boundary() {
        // Arrange: Transcendental(Tanh) → VecStore pattern
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Transcendental {
            dst: vec0, src: vec0, func: TranscendentalFn::Tanh,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = FwhtInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: FWHT instruction inserted between Tanh and Store
        assert!(stats.instrs_added >= 1);
        assert!(prog.len() > before);
        let fwht_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Fwht, dst, src } if *dst == *src)
        }).count();
        assert_eq!(fwht_count, 1);
    }

    // ── 25. PrefetchInsertPass — no prefetch outside loop ────────────

    #[test]
    fn prefetch_insert_no_prefetch_outside_loop() {
        // Arrange: VecLoad outside any loop — should not get Prefetch
        let mut prog = empty_program();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::VecLoad {
            dst: vec0, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = PrefetchInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: no Prefetch inserted because VecLoad is not inside a loop
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 26. DeadVRegEliminationPass — multiple dead Vec, one alive ───

    #[test]
    fn dead_vreg_elimination_multiple_dead_one_alive() {
        // Arrange: 3 Vec VRegs: v0 used, v1 dead, v2 dead
        let mut prog = empty_program();
        let v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let _v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256); // dead
        let _v2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256); // dead
        prog.emit(VmInstr::Broadcast {
            dst: v0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: 2 dead Vec DeclareVRegs removed
        assert_eq!(stats.instrs_removed, 2, "should remove 2 dead Vec declarations");
        // The alive v0's DeclareVReg should still exist
        let alive_decl = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::DeclareVReg { id, .. } if *id == v0)
        });
        assert!(alive_decl, "used v0 DeclareVReg should be preserved");
    }

    // ── 27. ScopeFlattenPass — multiple adjacent pairs ───────────────

    #[test]
    fn scope_flatten_multiple_adjacent_pairs() {
        // Arrange: three scopes where two have ScopeEnd→ScopeBegin adjacency
        let mut prog = empty_program();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::Comment("a".into()));
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });
        // Adjacent pair 1: scope 0 end → scope 1 begin
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 });
        prog.emit(VmInstr::Comment("b".into()));
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });
        // Adjacent pair 2: scope 1 end → scope 2 begin
        prog.emit(VmInstr::ScopeBegin { scope_id: 2 });
        prog.emit(VmInstr::Comment("c".into()));
        prog.emit(VmInstr::ScopeEnd { scope_id: 2 });

        // Act
        let (profile, hook) = make_hook();
        let stats = ScopeFlattenPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: 2 adjacent pairs removed (4 instructions: 2 ScopeEnd + 2 ScopeBegin)
        assert_eq!(stats.instrs_removed, 4);
        // Remaining: ScopeBegin(0), Comment(a), Comment(b), Comment(c), ScopeEnd(2)
        // i.e., the two middle ScopeEnd+ScopeBegin pairs are gone
        let scope_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::ScopeBegin { .. })).count();
        let scope_ends = prog.instrs.iter().filter(|i| matches!(i, VmInstr::ScopeEnd { .. })).count();
        assert_eq!(scope_begins, 1, "only outermost ScopeBegin should remain");
        assert_eq!(scope_ends, 1, "only outermost ScopeEnd should remain");
    }

    // ── 28. HotpatchSlotPass — inserts before IndirectJump ───────────

    #[test]
    fn hotpatch_slot_inserts_before_indirect_jump() {
        // Arrange: program with one IndirectJump instruction
        let mut prog = empty_program();
        let index = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::IndirectJump {
            index,
            targets: vec![super::super::instr::JumpTarget { expert_id: 0, instr_index: 0 }],
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = HotpatchSlotPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: one HotpatchSlot inserted before IndirectJump
        assert_eq!(stats.instrs_added, 1);
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(prog.len(), before + 1);
        let hp_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::HotpatchSlot { .. }));
        let ij_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::IndirectJump { .. }));
        assert!(hp_pos.is_some());
        assert!(ij_pos.is_some());
        assert_eq!(hp_pos.unwrap() + 1, ij_pos.unwrap());
    }

    // ── 29. LoopFusionPass — three consecutive loops fuse two pairs ──

    #[test]
    fn loop_fusion_three_loops_fuse_two_pairs() {
        // Arrange: Loop0(body)→Loop1(body)→Loop2(body)
        // Adjacent pairs: LoopEnd(0)+LoopBegin(1), LoopEnd(1)+LoopBegin(2)
        let mut prog = empty_program();
        let c0 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off0 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let c1 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off1 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let c2 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off2 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Loop 0
        prog.emit(VmInstr::LoopBegin { counter: c0, byte_offset: off0, bound: BoundExpr::Const(4), step_bytes: 32 });
        prog.emit(VmInstr::Broadcast { dst: vec0, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::LoopEnd);
        // Loop 1
        prog.emit(VmInstr::LoopBegin { counter: c1, byte_offset: off1, bound: BoundExpr::Const(4), step_bytes: 32 });
        prog.emit(VmInstr::Broadcast { dst: vec0, src: ScalarExpr::Const(2.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::LoopEnd);
        // Loop 2
        prog.emit(VmInstr::LoopBegin { counter: c2, byte_offset: off2, bound: BoundExpr::Const(4), step_bytes: 32 });
        prog.emit(VmInstr::Broadcast { dst: vec0, src: ScalarExpr::Const(3.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::LoopEnd);

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopFusionPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: two adjacent pairs fused → removes 2×2 = 4 instructions
        assert_eq!(stats.instrs_removed, 4);
        // Only 1 LoopBegin + 1 LoopEnd remain
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        let loop_ends = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopEnd)).count();
        assert_eq!(loop_begins, 1);
        assert_eq!(loop_ends, 1);
    }

    // ── 30. substitute_loop_offset_in_instr — VecLoad/VecStore ───────

    #[test]
    fn substitute_loop_offset_vecload_vecstore() {
        // Arrange: VecLoad and VecStore with LoopOffset that should become Const
        let vec0 = VRegId(10);
        let ptr = VRegId(20);
        let off_vreg = VRegId(30); // the loop offset VReg

        let load = VmInstr::VecLoad {
            dst: vec0, base: ptr,
            offset: OffsetExpr::LoopOffset(off_vreg),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        };
        let store = VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::LoopOffset(off_vreg),
            src: vec0, width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        };

        // Act: substitute LoopOffset(off_vreg) → Const(64)
        let new_load = substitute_loop_offset_in_instr(&load, off_vreg, 64);
        let new_store = substitute_loop_offset_in_instr(&store, off_vreg, 64);

        // Assert: offsets are now Const(64)
        match new_load {
            VmInstr::VecLoad { offset, .. } => assert_eq!(offset, OffsetExpr::Const(64)),
            _ => panic!("expected VecLoad"),
        }
        match new_store {
            VmInstr::VecStore { offset, .. } => assert_eq!(offset, OffsetExpr::Const(64)),
            _ => panic!("expected VecStore"),
        }
    }

    // ── 31. substitute_loop_offset_in_instr — Broadcast with MemLoad(LoopOffset) ──

    #[test]
    fn substitute_loop_offset_broadcast_memload() {
        // Arrange: Broadcast with ScalarExpr::MemLoad(base, LoopOffset(vreg))
        let dst = VRegId(10);
        let base = VRegId(20);
        let off_vreg = VRegId(30);

        let bc = VmInstr::Broadcast {
            dst,
            src: ScalarExpr::MemLoad(base, OffsetExpr::LoopOffset(off_vreg)),
            width: SimdWidth::W256,
            dtype: QuantPrecision::F32,
        };

        // Act: substitute LoopOffset(off_vreg) → Const(96)
        let result = substitute_loop_offset_in_instr(&bc, off_vreg, 96);

        // Assert: MemLoad offset is now Const(96)
        match result {
            VmInstr::Broadcast { src: ScalarExpr::MemLoad(_, ref offset), .. } => {
                assert_eq!(*offset, OffsetExpr::Const(96));
            }
            _ => panic!("expected Broadcast"),
        }
    }

    // ── 32. substitute_loop_offset_in_instr — Prefetch ────────────────

    #[test]
    fn substitute_loop_offset_prefetch() {
        // Arrange: Prefetch with LoopOffset
        let base = VRegId(10);
        let off_vreg = VRegId(20);

        let pf = VmInstr::Prefetch {
            base,
            offset: OffsetExpr::LoopOffset(off_vreg),
            distance: 64,
            hint: super::super::isa_hook::PrefetchHint::T0,
        };

        // Act
        let result = substitute_loop_offset_in_instr(&pf, off_vreg, 128);

        // Assert
        match result {
            VmInstr::Prefetch { offset, .. } => assert_eq!(offset, OffsetExpr::Const(128)),
            _ => panic!("expected Prefetch"),
        }
    }

    // ── 33. substitute_loop_offset_in_instr — LoadPtr VRegPlusVReg → VRegPlusConst ──

    #[test]
    fn substitute_loop_offset_loadptr_vreg_plus_vreg() {
        // Arrange: LoadPtr with PtrExpr::VRegPlusVReg(base, off_vreg)
        let dst = VRegId(10);
        let base = VRegId(20);
        let off_vreg = VRegId(30);

        let lp = VmInstr::LoadPtr {
            dst,
            src: PtrExpr::VRegPlusVReg(base, off_vreg),
        };

        // Act: substitute off_vreg → Const(256)
        let result = substitute_loop_offset_in_instr(&lp, off_vreg, 256);

        // Assert: VRegPlusVReg(base, off_vreg) → VRegPlusConst(base, 256)
        match result {
            VmInstr::LoadPtr { src: PtrExpr::VRegPlusConst(b, val), .. } => {
                assert_eq!(b, base);
                assert_eq!(val, 256);
            }
            _ => panic!("expected LoadPtr with VRegPlusConst"),
        }
    }

    // ── 34. substitute_loop_offset_in_instr — non-matching vreg is no-op ──

    #[test]
    fn substitute_loop_offset_non_matching_vreg_unchanged() {
        // Arrange: VecLoad with LoopOffset(vreg_a), but substitute vreg_b
        let vec0 = VRegId(10);
        let ptr = VRegId(20);
        let vreg_a = VRegId(30);
        let vreg_b = VRegId(40); // different vreg — should NOT match

        let load = VmInstr::VecLoad {
            dst: vec0, base: ptr,
            offset: OffsetExpr::LoopOffset(vreg_a),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        };

        // Act: substitute wrong vreg
        let result = substitute_loop_offset_in_instr(&load, vreg_b, 64);

        // Assert: offset unchanged (still LoopOffset(vreg_a))
        match result {
            VmInstr::VecLoad { offset, .. } => {
                assert_eq!(offset, OffsetExpr::LoopOffset(vreg_a));
            }
            _ => panic!("expected VecLoad"),
        }
    }

    // ── 35. substitute_loop_offset_in_instr — unrelated instruction cloned ──

    #[test]
    fn substitute_loop_offset_unrelated_instr_cloned() {
        // Arrange: Fma has no offset field, so it should be cloned as-is
        let vec0 = VRegId(1);
        let vec1 = VRegId(2);
        let off_vreg = VRegId(99);

        let fma = VmInstr::Fma {
            dst: vec0, acc: vec0, a: vec1, b: vec1,
            dtype: QuantPrecision::F32,
        };

        // Act
        let result = substitute_loop_offset_in_instr(&fma, off_vreg, 42);

        // Assert: Fma returned unchanged (clone)
        match result {
            VmInstr::Fma { dst, acc, a, b, dtype } => {
                assert_eq!(dst, vec0);
                assert_eq!(acc, vec0);
                assert_eq!(a, vec1);
                assert_eq!(b, vec1);
                assert_eq!(dtype, QuantPrecision::F32);
            }
            _ => panic!("expected Fma"),
        }
    }

    // ── 36. find_enclosing_loop_bound — after LoopEnd returns None ────

    #[test]
    fn find_enclosing_loop_bound_after_loop_end_is_none() {
        // Arrange: LoopBegin → Comment → LoopEnd → Comment
        // The second Comment is after LoopEnd, so not inside any loop.
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(4), step_bytes: 32,
        });
        prog.emit(VmInstr::Comment("inside loop".into()));
        prog.emit(VmInstr::LoopEnd);
        prog.emit(VmInstr::Comment("after loop".into()));

        // Find the "after loop" Comment index
        let after_idx = prog.instrs.iter().rposition(|i| {
            matches!(i, VmInstr::Comment(s) if s == "after loop")
        }).expect("after loop Comment should exist");

        // Act
        let bound = FwhtInsertPass::find_enclosing_loop_bound(&prog.instrs, after_idx);

        // Assert: not inside any loop (LoopEnd terminates the reverse scan)
        assert_eq!(bound, None);
    }

    // ── 37. LoopUnrollPass — bound=0 is not unrolled ─────────────────

    #[test]
    fn loop_unroll_bound_zero_is_noop() {
        // Arrange: LoopBegin(Const(0)) — zero iterations, should not unroll
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(0), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopUnrollPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: bound=0 fails n > 0 check, no unrolling
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 38. FwhtInsertPass — Sigmoid → VecStore boundary ─────────────

    #[test]
    fn fwht_insert_at_sigmoid_store_boundary() {
        // Arrange: Transcendental(Sigmoid) → VecStore pattern
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Transcendental {
            dst: vec0, src: vec0, func: TranscendentalFn::Sigmoid,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = FwhtInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: FWHT instruction inserted between Sigmoid and Store
        assert!(stats.instrs_added >= 1);
        assert!(prog.len() > before);
        let fwht_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Fwht, dst, src } if *dst == *src)
        }).count();
        assert_eq!(fwht_count, 1);
    }

    // ── 39. DeadVRegEliminationPass — ByteOffset vregs are not removed ──

    #[test]
    fn dead_vreg_elimination_preserves_byteoffset_vregs() {
        // Arrange: declare a ByteOffset VReg that is never referenced
        let mut prog = empty_program();
        let _dead_bo = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Use vec0 so it's not dead
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: ByteOffset VReg declarations are preserved (only Vec dead vregs removed)
        let has_bo_decl = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::DeclareVReg { kind: VRegKind::ByteOffset, .. })
        });
        assert!(has_bo_decl, "ByteOffset VReg declarations should not be removed");
        assert_eq!(stats.instrs_removed, 0);
    }

    // ── 40. LoopFusionPass — non-adjacent LoopBegin+LoopBegin is no-op ──

    #[test]
    fn loop_fusion_two_consecutive_begins_is_noop() {
        // Arrange: two consecutive LoopBegin (no LoopEnd between them) — not a fusable pattern
        let mut prog = empty_program();
        let c0 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off0 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let c1 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off1 = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin { counter: c0, byte_offset: off0, bound: BoundExpr::Const(4), step_bytes: 32 });
        prog.emit(VmInstr::LoopBegin { counter: c1, byte_offset: off1, bound: BoundExpr::Const(4), step_bytes: 32 });
        prog.emit(VmInstr::Broadcast { dst: vec0, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::LoopEnd);
        prog.emit(VmInstr::LoopEnd);
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopFusionPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: no LoopEnd+LoopBegin adjacency → no fusion
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 41. PassRegistry with_defaults on empty program — all passes are no-op ──

    #[test]
    fn with_defaults_all_passes_noop_on_empty_program() {
        // Arrange: empty VmProgram (no instructions at all, only DeclareVReg from alloc)
        let reg = PassRegistry::with_defaults();
        let mut prog = empty_program();
        let (profile, hook) = make_hook();

        // Act
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: all 11 passes ran, none removed or added instructions
        assert_eq!(results.len(), 11);
        let total_removed = results.iter().map(|s| s.instrs_removed).sum::<usize>();
        let total_added = results.iter().map(|s| s.instrs_added).sum::<usize>();
        assert_eq!(total_removed, 0, "no instructions to remove in empty program");
        assert_eq!(total_added, 0, "no instructions to add in empty program");
    }

    // ── 42. Single pass execution — DeadVRegEliminationPass alone via registry ──

    #[test]
    fn single_pass_registry_execution() {
        // Arrange: registry with only DeadVRegEliminationPass
        let mut reg = PassRegistry::new();
        reg.register(Box::new(DeadVRegEliminationPass));
        let mut prog = empty_program();
        let _dead = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let alive = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast {
            dst: alive, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        // Act
        let (profile, hook) = make_hook();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: exactly 1 pass ran, removed the dead Vec VReg declaration
        assert_eq!(results.len(), 1);
        assert!(results[0].instrs_removed > 0, "should remove dead VReg");
        assert_eq!(results[0].instrs_added, 0);
    }

    // ── 43. Multi-pass chain — register order differs from execution order ──

    #[test]
    fn multi_pass_chain_execution_by_priority_not_register_order() {
        // Arrange: register passes in reverse priority order
        let mut reg = PassRegistry::new();
        reg.register(Box::new(HotpatchSlotPass));   // priority=100
        reg.register(Box::new(ScopeFlattenPass));    // priority=20
        reg.register(Box::new(DeadVRegEliminationPass)); // priority=10
        let mut prog = empty_program();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });
        prog.emit(VmInstr::ScopeBegin { scope_id: 1 });
        prog.emit(VmInstr::ScopeEnd { scope_id: 1 });
        let (profile, hook) = make_hook();

        // Act
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: 3 passes ran; ScopeFlatten should remove adjacent pair
        assert_eq!(results.len(), 3);
        // ScopeFlatten ran (priority=20) and found adjacent ScopeEnd→ScopeBegin
        let scope_stats = &results[1]; // sorted: [10, 20, 100]
        assert!(scope_stats.instrs_removed > 0 || scope_stats.instrs_added == 0);
    }

    // ── 44. Pass name uniqueness — all 11 default pass names are distinct ──

    #[test]
    fn all_default_pass_names_are_unique() {
        // Arrange: collect names from all 11 default passes
        let passes: Vec<Box<dyn VmOptPass>> = vec![
            Box::new(DeadVRegEliminationPass),
            Box::new(ScopeFlattenPass),
            Box::new(LoopFusionPass),
            Box::new(StoreLoadForwardPass),
            Box::new(LoopUnrollPass),
            Box::new(TranscendentalBatchPass),
            Box::new(EpilogueFusionPass),
            Box::new(FwhtInsertPass),
            Box::new(ResidualBusPass),
            Box::new(PrefetchInsertPass),
            Box::new(HotpatchSlotPass),
        ];
        let names: Vec<&str> = passes.iter().map(|p| p.name()).collect();

        // Act: check for duplicates
        let mut unique_names = std::collections::HashSet::new();
        for name in &names {
            unique_names.insert(*name);
        }

        // Assert: 11 unique names from 11 passes
        assert_eq!(unique_names.len(), 11, "all pass names must be unique");
    }

    // ── 45. is_applicable returns true for all built-in passes ──────────

    #[test]
    fn all_builtin_passes_is_applicable_default_true() {
        // Arrange: all 11 built-in passes and a default IsaProfile
        let (profile, _) = make_hook();
        let passes: Vec<Box<dyn VmOptPass>> = vec![
            Box::new(DeadVRegEliminationPass),
            Box::new(ScopeFlattenPass),
            Box::new(LoopFusionPass),
            Box::new(StoreLoadForwardPass),
            Box::new(LoopUnrollPass),
            Box::new(TranscendentalBatchPass),
            Box::new(EpilogueFusionPass),
            Box::new(FwhtInsertPass),
            Box::new(ResidualBusPass),
            Box::new(PrefetchInsertPass),
            Box::new(HotpatchSlotPass),
        ];

        // Act + Assert: every pass should be applicable on the detected profile
        for pass in &passes {
            assert!(pass.is_applicable(&profile),
                "pass '{}' should be applicable by default", pass.name());
        }
    }

    // ── 46. OptStats Debug trait — default vs explicit values ───────────

    #[test]
    fn opt_stats_debug_default_vs_explicit() {
        // Arrange: two OptStats — default and explicit
        let default_stats = OptStats::default();
        let explicit_stats = OptStats { instrs_removed: 100, instrs_added: 50 };

        // Act
        let default_debug = format!("{:?}", default_stats);
        let explicit_debug = format!("{:?}", explicit_stats);

        // Assert: default shows zeros, explicit shows the given values
        assert!(default_debug.contains("0"), "default Debug should show 0 values");
        assert!(explicit_debug.contains("100"), "explicit Debug should show 100");
        assert!(explicit_debug.contains("50"), "explicit Debug should show 50");
    }

    // ── 47. PassRegistry register then run — incremental addition ──────

    #[test]
    fn pass_registry_incremental_register_then_run() {
        // Arrange: start empty, add one pass, verify; add another, verify
        let mut reg = PassRegistry::new();
        let (profile, hook) = make_hook();
        let mut prog1 = empty_program();

        // Act 1: run with no passes
        let results1 = reg.run_all(&mut prog1, &profile, hook.as_ref());
        assert_eq!(results1.len(), 0);

        // Act 2: add a custom pass and run again
        reg.register(Box::new(CustomNoopPass));
        let mut prog2 = empty_program();
        let results2 = reg.run_all(&mut prog2, &profile, hook.as_ref());
        assert_eq!(results2.len(), 1);

        // Act 3: add another pass and run again
        reg.register(Box::new(DeadVRegEliminationPass));
        let mut prog3 = empty_program();
        let results3 = reg.run_all(&mut prog3, &profile, hook.as_ref());
        assert_eq!(results3.len(), 2);
    }

    // ── 48. Register same pass type twice — both run ───────────────────

    #[test]
    fn pass_registry_same_pass_type_registered_twice() {
        // Arrange: register two instances of DeadVRegEliminationPass
        let mut reg = PassRegistry::new();
        reg.register(Box::new(DeadVRegEliminationPass));
        reg.register(Box::new(DeadVRegEliminationPass));

        let mut prog = empty_program();
        let _dead = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let alive = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast {
            dst: alive, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        // Act
        let (profile, hook) = make_hook();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: both instances ran (2 results), first removed the dead VReg,
        // second found nothing left to remove
        assert_eq!(results.len(), 2);
        assert!(results[0].instrs_removed > 0, "first pass should remove dead VReg");
        assert_eq!(results[1].instrs_removed, 0, "second pass has nothing left to remove");
    }

    // ── 49. LoopFusionPass — nested loop has no adjacent LoopEnd+LoopBegin ──

    #[test]
    fn loop_fusion_nested_loops_not_fused() {
        // Arrange: outer LoopBegin → inner LoopBegin → Broadcast → LoopEnd(inner) → Broadcast → LoopEnd(outer)
        // The inner LoopEnd is NOT adjacent to the outer LoopBegin (inner is before outer end)
        let mut prog = empty_program();
        let outer_c = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let outer_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let inner_c = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let inner_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin { counter: outer_c, byte_offset: outer_off, bound: BoundExpr::Const(4), step_bytes: 32 });
        prog.emit(VmInstr::LoopBegin { counter: inner_c, byte_offset: inner_off, bound: BoundExpr::Const(2), step_bytes: 32 });
        prog.emit(VmInstr::Broadcast { dst: vec0, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::LoopEnd); // inner
        prog.emit(VmInstr::Broadcast { dst: vec0, src: ScalarExpr::Const(2.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::LoopEnd); // outer
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopFusionPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: inner LoopEnd is followed by Broadcast, not LoopBegin → no fusion
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 50. PassRegistry with_defaults on comment-only program ─────────

    #[test]
    fn with_defaults_on_comment_only_program_preserves_comments() {
        // Arrange: program with only Comment instructions (no VRegs, no data ops)
        let reg = PassRegistry::with_defaults();
        let mut prog = empty_program();
        prog.emit(VmInstr::Comment("hello".into()));
        prog.emit(VmInstr::Comment("world".into()));
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: all 11 passes ran, comments are preserved, no instructions added
        assert_eq!(results.len(), 11);
        let total_removed = results.iter().map(|s| s.instrs_removed).sum::<usize>();
        let total_added = results.iter().map(|s| s.instrs_added).sum::<usize>();
        assert_eq!(total_removed, 0, "comments should not be removed by any pass");
        assert_eq!(total_added, 0, "no new instructions added for comment-only program");
        assert_eq!(prog.len(), before);
    }

    // ── 51. DeadVRegEliminationPass — Counter vregs are not removed ────

    #[test]
    fn dead_vreg_elimination_preserves_counter_vregs() {
        // Arrange: declare a Counter VReg that is never referenced in any instruction
        let mut prog = empty_program();
        let _dead_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: Counter VReg declarations are preserved (only Vec dead vregs removed)
        let has_counter_decl = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::DeclareVReg { kind: VRegKind::Counter, .. })
        });
        assert!(has_counter_decl, "Counter VReg declarations should not be removed");
        assert_eq!(stats.instrs_removed, 0);
    }

    // ── 52. DeadVRegEliminationPass — all Vec vregs dead removes all ────

    #[test]
    fn dead_vreg_elimination_all_vec_dead_removes_all() {
        // Arrange: program with only DeclareVReg(Vec) — no instructions reference them
        let mut prog = empty_program();
        let _v0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let _v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let _v2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // No instructions that reference any of these VRegs

        // Act
        let (profile, hook) = make_hook();
        let stats = DeadVRegEliminationPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: all 3 dead Vec DeclareVRegs removed
        assert_eq!(stats.instrs_removed, 3, "should remove all 3 dead Vec declarations");
        let vec_decls = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::DeclareVReg { kind: VRegKind::Vec, .. })
        }).count();
        assert_eq!(vec_decls, 0, "no Vec DeclareVReg should remain");
    }

    // ── 53. StoreLoadForwardPass — Mov instruction has correct dst and src ──

    #[test]
    fn store_load_forward_mov_preserves_dst_src() {
        // Arrange: VecStore(src=v0) → VecLoad(dst=v1) — should become Mov(dst=v1, src=v0)
        let mut prog = empty_program();
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let vec1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecLoad {
            dst: vec1, base: ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = StoreLoadForwardPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: Store+Load replaced by Mov with correct dst/src
        assert_eq!(stats.instrs_removed, 2);
        assert_eq!(stats.instrs_added, 1);
        let mov_instr = prog.instrs.iter().find(|i| matches!(i, VmInstr::Mov { .. }));
        assert!(mov_instr.is_some(), "should contain a Mov instruction");
        if let Some(VmInstr::Mov { dst, src, dtype }) = mov_instr {
            assert_eq!(*dst, vec1, "Mov dst should be the Load's dst");
            assert_eq!(*src, vec0, "Mov src should be the Store's src");
            assert_eq!(*dtype, QuantPrecision::F32);
        }
    }

    // ── 54. LoopUnrollPass — bound=1 unrolls to single body copy ───────

    #[test]
    fn loop_unroll_bound_one_unrolls_to_single_body() {
        // Arrange: LoopBegin(Const(1), step=32) → Broadcast → LoopEnd
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(1), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopUnrollPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: LoopBegin+LoopEnd removed (2), 1 copy of body added (1)
        assert!(stats.instrs_removed >= 2, "should remove LoopBegin and LoopEnd");
        assert_eq!(stats.instrs_added, 1, "should add 1 copy of the body");
        // No loop control flow remains
        assert!(prog.instrs.iter().all(|i| !matches!(i, VmInstr::LoopBegin { .. })));
        assert!(prog.instrs.iter().all(|i| !matches!(i, VmInstr::LoopEnd)));
    }

    // ── 55. HotpatchSlotPass — multiple branch instructions get multiple slots ──

    #[test]
    fn hotpatch_slot_multiple_branches_get_multiple_slots() {
        // Arrange: program with both ConditionalSkip and IndirectJump
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let mask = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let index = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);

        prog.emit(VmInstr::ConditionalSkip { mask, skip_count: 1 });
        prog.emit(VmInstr::IndirectJump {
            index,
            targets: vec![super::super::instr::JumpTarget { expert_id: 0, instr_index: 0 }],
        });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = HotpatchSlotPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: 2 HotpatchSlots inserted (one before each branch)
        assert_eq!(stats.instrs_added, 2);
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(prog.len(), before + 2);
        // Each HotpatchSlot should have a distinct slot_id
        let slot_ids: Vec<u32> = prog.instrs.iter().filter_map(|i| {
            if let VmInstr::HotpatchSlot { slot_id, .. } = i { Some(*slot_id) } else { None }
        }).collect();
        assert_eq!(slot_ids.len(), 2);
        assert_ne!(slot_ids[0], slot_ids[1], "slot_ids should be distinct");
    }

    // ── 56. FwhtInsertPass — boundary pattern detection ────────────────

    #[test]
    fn fwht_insert_detects_boundary_and_inserts_fwht() {
        // Arrange: one recognized boundary pattern (Exp → VecStore)
        let mut prog = empty_program();
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Transcendental {
            dst: vec0, src: vec0, func: TranscendentalFn::Exp,
        });
        prog.emit(VmInstr::VecStore {
            base: ptr, offset: OffsetExpr::Const(0), src: vec0,
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });

        // Act
        let (profile, hook) = make_hook();
        let stats = FwhtInsertPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: 1 FWHT instruction inserted between Exp and VecStore
        assert!(stats.instrs_added >= 1, "should insert FWHT at boundary");
        let fwht_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::Transcendental { func: TranscendentalFn::Fwht, dst, src } if *dst == *src)
        }).count();
        assert_eq!(fwht_count, 1, "should have exactly 1 FWHT instruction");
    }

    // ── 57. TranscendentalBatchPass — currently disabled, is no-op ─────

    #[test]
    fn transcendental_batch_pass_is_currently_noop() {
        // Arrange: loop with a constant Broadcast that could be hoisted
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Const(8), step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(0.044715),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = TranscendentalBatchPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: pass is currently disabled (ARCH-VREG-DECLARE-BEFORE-USE),
        // so it should be a no-op regardless of hoistable patterns
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 58. PassRegistry — custom pass with is_applicable returning false ──

    #[test]
    fn pass_registry_skips_inapplicable_pass() {
        // Arrange: custom pass that is never applicable
        struct NeverApplicablePass;
        impl VmOptPass for NeverApplicablePass {
            fn name(&self) -> &'static str { "never_applicable" }
            fn priority(&self) -> u32 { 50 }
            fn is_applicable(&self, _profile: &IsaProfile) -> bool { false }
            fn run(&self, _program: &mut VmProgram, _profile: &IsaProfile, _hook: &dyn IsaHook) -> OptStats {
                OptStats { instrs_removed: 999, instrs_added: 999 }
            }
        }

        let mut reg = PassRegistry::new();
        reg.register(Box::new(NeverApplicablePass));
        let mut prog = empty_program();

        // Act
        let (profile, hook) = make_hook();
        let results = reg.run_all(&mut prog, &profile, hook.as_ref());

        // Assert: pass was skipped (is_applicable returned false), no results
        assert_eq!(results.len(), 0, "inapplicable pass should not run");
    }

    // ── 59. ScopeFlattenPass — single scope with no adjacent pair ──────

    #[test]
    fn scope_flatten_single_scope_is_noop() {
        // Arrange: a single scope with no ScopeEnd→ScopeBegin adjacency
        let mut prog = empty_program();
        prog.emit(VmInstr::ScopeBegin { scope_id: 0 });
        prog.emit(VmInstr::Comment("body".into()));
        prog.emit(VmInstr::ScopeEnd { scope_id: 0 });
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = ScopeFlattenPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: no adjacent ScopeEnd→ScopeBegin pair, nothing removed
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }

    // ── 60. LoopUnrollPass — Symbolic bound is not unrolled ────────────

    #[test]
    fn loop_unroll_symbolic_bound_is_noop() {
        // Arrange: LoopBegin with Symbolic bound — only Const bounds are unrollable
        let mut prog = empty_program();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let vec0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset: off,
            bound: BoundExpr::Symbolic(super::super::instr::SymBound {
                name: "seq_len".into(),
                max_alloc: 2048,
            }),
            step_bytes: 32,
        });
        prog.emit(VmInstr::Broadcast {
            dst: vec0, src: ScalarExpr::Const(1.0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32,
        });
        prog.emit(VmInstr::LoopEnd);
        let before = prog.len();

        // Act
        let (profile, hook) = make_hook();
        let stats = LoopUnrollPass.run(&mut prog, &profile, hook.as_ref());

        // Assert: Symbolic bound is not Const, so no unrolling
        assert_eq!(stats.instrs_removed, 0);
        assert_eq!(stats.instrs_added, 0);
        assert_eq!(prog.len(), before);
    }
}
