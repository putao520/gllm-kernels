//! 统一栈帧管理 + 硬件压力模型 (REGISTER-VM SPEC §8 + §11.5)

use super::isa_profile::*;

use super::reg_alloc::RegAllocation;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 StackFrame — scope-based 栈帧
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// scope-based 栈帧——预计算总大小，一次性 sub rsp。
#[derive(Debug)]
pub struct StackFrame {
    /// 栈帧总大小 (已对齐)
    pub total_size: usize,
    /// 对齐要求
    pub alignment: usize,
    /// Callee-saved 寄存器保存区大小
    pub callee_save_area: usize,
    /// Spill 区大小
    pub spill_area: usize,
    /// Scratchpad 区大小 (buffer_alloc + dynamic_allocs)
    pub scratchpad_area: usize,
    /// 是否可用 red zone (leaf + 小栈帧)
    pub uses_red_zone: bool,
}

impl StackFrame {
    /// 从 RegAllocation + IsaProfile 计算栈帧布局。
    pub fn compute(
        alloc: &RegAllocation,
        profile: &IsaProfile,
        scratchpad_bytes: usize,
    ) -> Self {
        let callee_save_area = alloc.callee_saved_used.len() * 8; // 每个 GPR push 8 bytes
        let spill_area: usize = alloc.spills.iter().map(|s| s.size).sum();
        // ARCH-SCRATCH-NOT-ON-STACK: scratchpad 是外部 buffer (ABI arg 8 = [rbp+32]),
        // 由调用方分配传入,不应计入 CPU 栈帧的 sub rsp 大小。否则 352MB 分类器
        // 模型导致 sub rsp 立即 SIGSEGV (Linux 默认栈 8MB)。
        // scratchpad_area 字段保留给 GPU shared memory 声明使用。
        let scratchpad_area = scratchpad_bytes;

        // CPU 栈帧仅包含 callee_saves + spill slots,不含 scratchpad
        let raw_size = callee_save_area + spill_area;

        // 对齐到 ABI 要求
        let alignment = profile.abi.stack_alignment;
        let total_size = if alignment > 0 {
            (raw_size + alignment - 1) & !(alignment - 1)
        } else {
            raw_size
        };

        // Red zone: 如果总大小 ≤ red_zone_bytes 且无嵌套调用，可以不 sub rsp
        let uses_red_zone = total_size <= profile.abi.red_zone_bytes
            && scratchpad_area == 0; // scratchpad 需要外部可见地址，不能用 red zone

        Self {
            total_size,
            alignment,
            callee_save_area,
            spill_area,
            scratchpad_area,
            uses_red_zone,
        }
    }

    /// Spill slot 在栈帧中的偏移 (相对于 rbp)。
    /// 布局: [rbp] → callee_saves → spills → scratchpad → [rsp]
    pub fn spill_offset(&self, spill_index: usize, alloc: &RegAllocation) -> i32 {
        let base = self.callee_save_area;
        let offset = alloc.spills.iter().take(spill_index).map(|s| s.size).sum::<usize>();
        -((base + offset + 8) as i32) // 负偏移 (栈向下增长)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.5 ScopedSpillAllocator — 基于 Scope 的 Spill Slot 管理 (REQ-LC-004~006)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scope 标识符
pub type ScopeId = usize;

/// Spill slot 状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// 可分配
    Free,
    /// 被 VReg 占用
    Occupied,
}

/// 单个 spill slot 的元数据
#[derive(Debug, Clone)]
pub struct SpillSlotInfo {
    /// 栈帧偏移 (字节)
    pub offset: usize,
    /// 字节数
    pub size: usize,
    /// 所属 scope (None = global)
    pub owner: Option<ScopeId>,
    /// 当前占用者
    pub vreg: Option<super::instr::VRegId>,
    /// 状态
    pub state: SlotState,
}

/// 基于 Scope 的 Spill Slot 分配器，替代顺序单调分配。
///
/// 当 ScopeEnd 时释放该 scope 的所有 slot，free_list 支持复用。
/// 嵌套 scope：内层 scope 结束时只释放内层独有的 slot。
#[derive(Debug)]
pub struct ScopedSpillAllocator {
    slots: Vec<SpillSlotInfo>,
    free_list: Vec<usize>,
    scope_stack: Vec<ScopeId>,
    scope_slots: std::collections::HashMap<ScopeId, Vec<usize>>,
    next_scope_id: ScopeId,
    next_offset: usize,
    /// DIAG: unique allocator instance ID
    diag_id: usize,
}

impl ScopedSpillAllocator {
    pub fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            scope_stack: Vec::new(),
            scope_slots: std::collections::HashMap::new(),
            next_scope_id: 0,
            next_offset: 0,
            diag_id: id,
        }
    }

    /// 分配一个 spill slot。
    /// 优先从 free_list 找 size 完全匹配的 slot；否则分配新 slot。
    /// 返回 (slot_index, byte_offset) — slot_index 用于 regalloc 的 Spilled(slot_id),
    /// byte_offset 用于栈帧布局计算。
    pub fn alloc(
        &mut self,
        vreg: super::instr::VRegId,
        size: usize,
        scope_id: Option<ScopeId>,
    ) -> (usize, usize) {
        // 优先从 free_list 找 size 完全匹配的 slot
        if let Some(pos) = self.free_list.iter().position(|&idx| self.slots[idx].size == size) {
            let idx = self.free_list.remove(pos);
            // DIAG: track slot reuse at offset 3720 (0xe88)
            if self.slots[idx].offset == 3720 {
                eprintln!("[SPILL-REUSE] alloc={} offset=3720 slot_idx={}: old_vreg={:?} -> new_vreg={:?} scope={:?}",
                    self.diag_id, idx, self.slots[idx].vreg, vreg, scope_id);
            }
            self.slots[idx].state = SlotState::Occupied;
            self.slots[idx].vreg = Some(vreg);
            self.slots[idx].owner = scope_id;
            if let Some(sid) = scope_id {
                self.scope_slots.entry(sid).or_default().push(idx);
            }
            return (idx, self.slots[idx].offset);
        }

        // 无匹配：分配新 slot
        let offset = self.next_offset;
        self.next_offset += size;
        // DIAG: track new allocation at offset 3720
        if offset <= 3720 && offset + size > 3720 {
            eprintln!("[SPILL-NEW] alloc={} overlap with 3720: offset={} size={} vreg={:?} scope={:?} slots_len={} next_offset={}",
                self.diag_id, offset, size, vreg, scope_id, self.slots.len(), self.next_offset);
        }
        let info = SpillSlotInfo {
            offset,
            size,
            owner: scope_id,
            vreg: Some(vreg),
            state: SlotState::Occupied,
        };
        let idx = self.slots.len();
        self.slots.push(info);
        if let Some(sid) = scope_id {
            self.scope_slots.entry(sid).or_default().push(idx);
        }
        (idx, offset)
    }

    /// ScopeBegin: 推入新 scope，返回 ScopeId。
    pub fn scope_begin(&mut self) -> ScopeId {
        let id = self.next_scope_id;
        self.next_scope_id += 1;
        self.scope_stack.push(id);
        id
    }

    /// ScopeEnd: 弹出 scope，释放该 scope 拥有的所有 slot。
    /// CrossScope 的 slot 不释放 (owner != 当前 scope)。
    pub fn scope_end(&mut self) -> Option<ScopeId> {
        let scope_id = self.scope_stack.pop()?;
        if let Some(indices) = self.scope_slots.remove(&scope_id) {
            for idx in indices {
                self.slots[idx].state = SlotState::Free;
                self.slots[idx].vreg = None;
                self.free_list.push(idx);
            }
        }
        Some(scope_id)
    }

    /// 当前活跃 scope (栈顶)
    pub fn current_scope(&self) -> Option<ScopeId> {
        self.scope_stack.last().copied()
    }

    /// 已分配的总字节数 (包括已释放但未复用的 slot)
    pub fn total_allocated(&self) -> usize {
        self.next_offset
    }

    /// 当前活跃 (Occupied) 的 slot 数量
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.state == SlotState::Occupied).count()
    }

    /// 转换为 RegAllocation 兼容的 spill 列表。
    /// 返回所有 slot（包括 Freed 状态的），保持与 slot_index 的一一对应关系。
    /// Freed slot 用 (VRegId(u32::MAX), 0, 0) 占位。
    pub fn into_spills(self) -> Vec<super::reg_alloc::SpillSlot> {
        use super::reg_alloc::SpillSlot;
        use super::instr::VRegId;
        self.slots.into_iter()
            .map(|s| {
                if s.state == SlotState::Occupied {
                    SpillSlot {
                        vreg: s.vreg.unwrap_or(VRegId(u32::MAX)),
                        offset: s.offset,
                        size: s.size,
                    }
                } else {
                    // Freed slot 占位 — 保持索引对齐
                    SpillSlot {
                        vreg: VRegId(u32::MAX),
                        offset: s.offset,
                        size: s.size,
                    }
                }
            })
            .collect()
    }
}

impl Default for ScopedSpillAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 PressureModel — 硬件压力模型
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 运行模式 (§10)。
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    Prefill { seq_len: usize },
    Decode { batch_size: usize },
    ChunkedPrefill { chunk_size: usize, total_seq: usize },
}

/// §11.6 GEMM 模式提示。
#[derive(Debug, Clone, Copy)]
pub enum GemmModeHint {
    LargeTile,
    SmallBatch,
    Gemv,
    Adaptive,
}

/// §11.6 预取模式。
#[derive(Debug, Clone, Copy)]
pub enum PrefetchMode {
    Streaming,
    Aggressive,
    Adaptive,
}

impl ExecutionMode {
    /// §11.6 GEMM 策略推导。
    pub fn gemm_strategy(&self) -> GemmModeHint {
        match self {
            Self::Prefill { seq_len } if *seq_len >= 64 => GemmModeHint::LargeTile,
            Self::Decode { batch_size } if *batch_size == 1 => GemmModeHint::Gemv,
            Self::Decode { .. } => GemmModeHint::SmallBatch,
            _ => GemmModeHint::Adaptive,
        }
    }

    /// §11.6 预取模式推导。
    pub fn prefetch_mode(&self) -> PrefetchMode {
        match self {
            Self::Prefill { .. } => PrefetchMode::Streaming,
            Self::Decode { .. } => PrefetchMode::Aggressive,
            Self::ChunkedPrefill { .. } => PrefetchMode::Adaptive,
        }
    }
}

/// GEMM blocking 参数。
#[derive(Debug, Clone)]
pub struct GemmBlocking {
    pub mr: usize,
    pub nr: usize,
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
}

/// 数据区域 (§11.5)。
#[derive(Debug, Clone)]
pub struct DataRegion {
    pub label: &'static str,
    pub size_bytes: usize,
    pub access_pattern: AccessPattern,
}

/// 内存访问模式。
#[derive(Debug, Clone)]
pub enum AccessPattern {
    Streaming,
    Reuse { reuse_count: usize },
    Random,
}

/// 缓存层级。
#[derive(Debug, Clone, Copy)]
pub enum CacheLevel { Register, L1, L2, L3, Hbm }

/// 内存压力 (§11.5)。
#[derive(Debug, Clone)]
pub struct MemoryPressure {
    pub l1_budget: usize,
    pub l2_budget: usize,
    pub l3_budget: usize,
    pub hbm_utilization: f32,
    /// 活跃数据集 vs 缓存层级映射
    pub data_placement: Vec<(DataRegion, CacheLevel)>,
}

/// 寄存器压力 (§11.5)。
#[derive(Debug, Clone)]
pub struct RegisterPressure {
    /// 可用向量寄存器数
    pub vec_available: usize,
    /// 可用 GPR 数
    pub gpr_available: usize,
    /// 预估 spill 数量
    pub estimated_spills: usize,
}

/// 并行压力 (§11.5)。
#[derive(Debug, Clone)]
pub struct ParallelPressure {
    pub simd_lanes: usize,
    pub gpu_target_occupancy: f32,
    pub gpu_shared_per_wave: usize,
    /// §12 形状桶
    pub shape_buckets: Vec<ShapeBucket>,
}

/// §11 FWHT 插入点 (Softmax/SwiGLU/RoPE 非线性边界)。
#[derive(Debug, Clone, Copy)]
pub enum FwhtInsertPoint {
    SoftmaxEpilogue,
    SwigluMidpoint,
    RopePreRotation,
}

/// §11 KV 量化模式。
#[derive(Debug, Clone, Copy)]
pub enum KvQuantMode {
    PerChannel,  // Key: 离线校准
    PerToken,    // Value: 运行时 reduce_max
}

/// 量化压力 (§11 TurboQuant)。
#[derive(Debug, Clone)]
pub struct QuantPressure {
    pub weight_bits: u8,
    /// FWHT 插入点
    pub fwht_points: Vec<FwhtInsertPoint>,
    /// KV 量化模式
    pub kv_quant: KvQuantMode,
    /// Attention Sink 保护 token 数
    pub sink_tokens: usize,
}

/// §14.6 代码大小预算。
#[derive(Debug, Clone)]
pub struct CodeSizeBudget {
    pub l1i_bytes: usize,
    pub budget_ratio: f32,
    pub max_code_bytes: usize,
}

/// §14.6 NUMA 拓扑约束。
#[derive(Debug, Clone)]
pub struct NumaConstraint {
    pub node_count: usize,
    pub node_l3_bytes: Vec<usize>,
    pub affinity: NumaAffinity,
}

/// NUMA 亲和策略。
#[derive(Debug, Clone)]
pub enum NumaAffinity {
    /// pack buffer 绑定到当前 NUMA 节点
    Local,
    /// pack buffer 可跨 NUMA 节点 (性能较差但内存更灵活)
    Interleave,
}

/// Epilogue 融合预算 (§13)。
#[derive(Debug, Clone)]
pub struct EpilogueBudget {
    /// GEMM 累加器数量 (mr×nr)
    pub accumulators: usize,
    /// Epilogue 后剩余 temp 向量数
    pub temp_after_epilogue: usize,
}

/// §12 形状桶 (空间异构)。
#[derive(Debug, Clone)]
pub struct ShapeBucket {
    pub seq_range: (usize, usize),
    pub batch_size: usize,
    pub strategy: BucketStrategy,
}

#[derive(Debug, Clone)]
pub enum BucketStrategy {
    /// 标准 GEMM blocking
    Standard,
    /// GEMV (seq_len=1)
    Gemv,
    /// 大 seq 用 Chunked Prefill
    ChunkedPrefill { chunk_size: usize },
}

/// 硬件压力模型——汇聚所有维度约束，驱动 VmOptPass 和 IsaHook 决策。
#[derive(Debug)]
pub struct PressureModel {
    pub mode: ExecutionMode,
    pub blocking: GemmBlocking,
    pub memory: MemoryPressure,
    pub register: RegisterPressure,
    pub parallel: ParallelPressure,
    pub quant: QuantPressure,
    pub epilogue_budget: EpilogueBudget,
    /// L1i 代码大小预算 (§14.6)
    pub code_budget_bytes: usize,
}

impl PressureModel {
    /// 从 IsaProfile + ExecutionMode 推导全维度压力。
    pub fn analyze(profile: &IsaProfile, mode: ExecutionMode) -> Self {
        let cache = &profile.cache;
        // ARCH-DATA-FLOW-CONTRACT §2.2: 元素字节数来自单一来源
        let elem = super::lower::computation_elem_bytes();

        // GEMM blocking 从缓存推导（容量从 IsaProfile 读取，不硬编码）
        let l1 = cache.l1d_bytes.max(8192);
        let l2 = cache.l2_bytes.max(65536);
        let l3 = cache.l3_bytes.max(1048576);

        // BLIS 微内核形状从 IsaProfile 派生（avx2 16 vec regs → mr=3,nr=4; avx-512 32 regs → mr=6,nr=4）
        let (mr, nr) = if profile.vec_regs.len() >= 32 { (6, 4) } else { (3, 4) };

        // BLIS 分块（标准公式）:
        //   kc = L1 / (2 * mr * elem)       — A+B 两个矩阵驻留 L1，mr 为行高
        //   mc = L1 / (2 * kc * elem)       — kc 列 pack_a 驻留 L1
        //   nc = L2 / (2 * kc * elem)       — pack_b 驻留 L2
        let kc = (l1 / (2 * mr * elem)).min(512).max(16);
        let mc = (l1 / (2 * kc * elem)).max(4);
        let nc = (l2 / (2 * kc * elem)).max(8);

        let code_budget_bytes = (cache.l1i_bytes as f64 * 0.8) as usize;

        let simd_lanes = profile.optimal_simd_width().f32_lanes();
        let vec_available = profile.vec_regs.len();
        let gpr_available = profile.gpr_regs.len();

        // Epilogue budget: accumulators = mr*nr, temp = vec_available - mr*nr
        let acc_count = mr * nr;
        let temp_after = vec_available.saturating_sub(acc_count + 4); // 4 scratch

        Self {
            mode,
            blocking: GemmBlocking { mr, nr, mc, nc, kc },
            memory: MemoryPressure {
                l1_budget: l1 / 2,
                l2_budget: l2 / 2,
                l3_budget: l3 / 2,
                hbm_utilization: 0.0,
                data_placement: vec![], // 由 lower 阶段根据 BufferAllocation 填充
            },
            register: RegisterPressure {
                vec_available,
                gpr_available,
                estimated_spills: 0,
            },
            parallel: ParallelPressure {
                simd_lanes,
                gpu_target_occupancy: 0.75,
                gpu_shared_per_wave: 0,
                shape_buckets: vec![
                    ShapeBucket { seq_range: (0, 1), batch_size: 1, strategy: BucketStrategy::Gemv },
                    ShapeBucket { seq_range: (2, 64), batch_size: 1, strategy: BucketStrategy::Standard },
                    ShapeBucket { seq_range: (65, 8192), batch_size: 1, strategy: BucketStrategy::ChunkedPrefill { chunk_size: 128 } },
                ],
            },
            quant: QuantPressure {
                weight_bits: 32,
                fwht_points: vec![],
                kv_quant: KvQuantMode::PerToken,
                sink_tokens: 4,
            },
            epilogue_budget: EpilogueBudget {
                accumulators: acc_count,
                temp_after_epilogue: temp_after,
            },
            code_budget_bytes,
        }
    }

    /// Epilogue 放置策略 (由寄存器压力驱动)。
    pub fn epilogue_placement(&self, epi_ops: usize) -> super::isa_hook::EpiloguePlace {
        if self.epilogue_budget.temp_after_epilogue >= epi_ops * 2 {
            super::isa_hook::EpiloguePlace::OnAccumulators
        } else {
            super::isa_hook::EpiloguePlace::AfterStore
        }
    }

    /// GPU Occupancy 计算。
    /// 返回给定 regs_per_thread 下的最大 waves/warps per SM。
    /// GPU Occupancy 计算——从 IsaProfile 的实际硬件参数推导。
    pub fn gpu_occupancy(&self, regs_per_thread: usize, profile: &super::isa_profile::IsaProfile) -> f32 {
        use super::isa_profile::Platform;
        let (reg_file, threads_per_wave, max_waves) = match &profile.platform {
            Platform::Cuda { reg_file_per_sm, warp_size, .. } => {
                (*reg_file_per_sm, *warp_size as usize, 64usize)
            }
            Platform::Hip { vgpr_per_cu, wave_size, .. } => {
                (*vgpr_per_cu, *wave_size as usize, 40usize)
            }
            _ => (65536, 32, 64), // CPU 兜底
        };
        if regs_per_thread == 0 || threads_per_wave == 0 { return 0.0; }
        let waves = reg_file / (regs_per_thread * threads_per_wave);
        (waves.min(max_waves) as f32) / (max_waves as f32)
    }

    /// §11.6 GEMM 策略提示 (委托给 ExecutionMode)。
    pub fn gemm_strategy(&self) -> GemmModeHint {
        self.mode.gemm_strategy()
    }

    /// §11.6 预取模式 (委托给 ExecutionMode)。
    pub fn prefetch_mode(&self) -> PrefetchMode {
        self.mode.prefetch_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DeviceProfile;
    use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, VmInstr, OffsetExpr, SimdWidth, VRegId};
    use crate::compiler::codegen::vm::reg_alloc::*;
    use crate::compiler::codegen::vm::auto_select;
    use crate::compiler::trace::{TraceOp, ValueId};
    use crate::compiler::trace::QuantPrecision;

    #[test]
    fn test_stack_frame_alignment() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let body = vec![TraceOp::Input(0), TraceOp::Neg(ValueId(0))];
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_reg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad { dst: vec_reg, base: input_ptr, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let slots = auto_select::auto_lower_trace_raw(&mut prog, &body, &[vec_reg], SimdWidth::W256, QuantPrecision::F32).unwrap();
        let last = *slots.last().unwrap();
        prog.emit(VmInstr::VecStore { base: output_ptr, src: last, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = StackFrame::compute(&alloc, &profile, 0);

        assert_eq!(frame.total_size % frame.alignment, 0, "frame not aligned");
    }

    #[test]
    fn test_stack_frame_with_scratchpad() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![PhysGpr(3), PhysGpr(12)], // rbx, r12
        };
        let frame = StackFrame::compute(&alloc, &profile, 1024);

        // ARCH-SCRATCH-NOT-ON-STACK: scratchpad is an external buffer (ABI arg),
        // not allocated on the CPU stack. total_size only includes callee_saves + spills.
        assert!(frame.total_size >= 16, "frame must include callee saves"); // 2 callee saves * 8 bytes = 16
        assert!(frame.scratchpad_area == 1024, "scratchpad_area field preserved for GPU use");
        assert!(!frame.uses_red_zone); // scratchpad 禁用 red zone
    }

    #[test]
    fn test_pressure_model_blocking() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let model = PressureModel::analyze(&profile, ExecutionMode::Prefill { seq_len: 512 });

        assert!(model.blocking.kc >= 16);
        assert!(model.blocking.mc >= 4);
        assert!(model.blocking.nc >= 8);
        assert!(model.code_budget_bytes > 0);
    }

    // ── ScopedSpillAllocator ──

    #[test]
    fn scoped_spill_alloc_sequential() {
        let mut allocator = ScopedSpillAllocator::new();
        let (idx0, off0) = allocator.alloc(VRegId(0), 4, None);
        let (idx1, off1) = allocator.alloc(VRegId(1), 8, None);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(off0, 0);
        assert_eq!(off1, 4);
        assert_eq!(allocator.total_allocated(), 12);
        assert_eq!(allocator.active_count(), 2);
    }

    #[test]
    fn scoped_spill_scope_begin_end_frees_slots() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope = allocator.scope_begin();
        allocator.alloc(VRegId(0), 16, Some(scope));
        allocator.alloc(VRegId(1), 32, Some(scope));
        assert_eq!(allocator.active_count(), 2);

        allocator.scope_end();
        assert_eq!(allocator.active_count(), 0);
        assert!(allocator.current_scope().is_none());
    }

    #[test]
    fn scoped_spill_freed_slot_reuse() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope = allocator.scope_begin();
        allocator.alloc(VRegId(0), 8, Some(scope));
        allocator.scope_end();
        // Freed slot should be reused (same size)
        let (idx, _off) = allocator.alloc(VRegId(1), 8, None);
        assert_eq!(idx, 0);
    }

    #[test]
    fn scoped_spill_default_is_empty() {
        let allocator = ScopedSpillAllocator::default();
        assert_eq!(allocator.active_count(), 0);
        assert_eq!(allocator.total_allocated(), 0);
        assert!(allocator.current_scope().is_none());
    }

    #[test]
    fn scoped_spill_into_spills_preserves_freed() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope = allocator.scope_begin();
        allocator.alloc(VRegId(10), 4, Some(scope));
        allocator.scope_end();
        let spills = allocator.into_spills();
        assert_eq!(spills.len(), 1);
        // Freed slot gets VRegId(u32::MAX)
        assert_eq!(spills[0].vreg, VRegId(u32::MAX));
    }

    // ── ExecutionMode strategies ──

    #[test]
    fn execution_mode_prefill_large_tile() {
        let mode = ExecutionMode::Prefill { seq_len: 128 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::LargeTile));
        assert!(matches!(mode.prefetch_mode(), PrefetchMode::Streaming));
    }

    #[test]
    fn execution_mode_prefill_small_adaptive() {
        let mode = ExecutionMode::Prefill { seq_len: 8 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::Adaptive));
    }

    #[test]
    fn execution_mode_decode_single_gemv() {
        let mode = ExecutionMode::Decode { batch_size: 1 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::Gemv));
        assert!(matches!(mode.prefetch_mode(), PrefetchMode::Aggressive));
    }

    #[test]
    fn execution_mode_decode_batch_small_batch() {
        let mode = ExecutionMode::Decode { batch_size: 4 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::SmallBatch));
    }

    #[test]
    fn execution_mode_chunked_prefill_adaptive() {
        let mode = ExecutionMode::ChunkedPrefill { chunk_size: 32, total_seq: 512 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::Adaptive));
        assert!(matches!(mode.prefetch_mode(), PrefetchMode::Adaptive));
    }

    // ── 14. ScopedSpillAllocator nested scopes ───────────────────────

    #[test]
    fn scoped_spill_nested_scopes_inner_frees_first() {
        let mut allocator = ScopedSpillAllocator::new();
        let outer = allocator.scope_begin();
        allocator.alloc(VRegId(0), 8, Some(outer));
        let inner = allocator.scope_begin();
        allocator.alloc(VRegId(1), 16, Some(inner));
        assert_eq!(allocator.active_count(), 2);

        allocator.scope_end(); // inner
        assert_eq!(allocator.active_count(), 1);
        assert_eq!(allocator.current_scope(), Some(outer));

        allocator.scope_end(); // outer
        assert_eq!(allocator.active_count(), 0);
    }

    // ── 15. ScopedSpillAllocator no scope scope_end returns None ─────

    #[test]
    fn scoped_spill_scope_end_empty_returns_none() {
        let mut allocator = ScopedSpillAllocator::new();
        assert!(allocator.scope_end().is_none());
    }

    // ── 16. ScopedSpillAllocator different size no reuse ─────────────

    #[test]
    fn scoped_spill_different_size_no_reuse() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope = allocator.scope_begin();
        allocator.alloc(VRegId(0), 8, Some(scope));
        allocator.scope_end();
        // Freed slot is size=8, request size=16 → new slot
        let (idx, _off) = allocator.alloc(VRegId(1), 16, None);
        assert_eq!(idx, 1, "different size should allocate new slot, not reuse");
    }

    // ── 17. ScopedSpillAllocator into_spills occupied preserves vreg ─

    #[test]
    fn scoped_spill_into_spills_occupied_keeps_vreg() {
        let mut allocator = ScopedSpillAllocator::new();
        allocator.alloc(VRegId(42), 32, None);
        let spills = allocator.into_spills();
        assert_eq!(spills.len(), 1);
        assert_eq!(spills[0].vreg, VRegId(42));
        assert_eq!(spills[0].size, 32);
    }

    // ── 18. ExecutionMode prefill seq_len=63 adaptive ───────────────

    #[test]
    fn execution_mode_prefill_boundary_below_64() {
        let mode = ExecutionMode::Prefill { seq_len: 63 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::Adaptive));
    }

    // ── 19. ExecutionMode prefill seq_len=64 large_tile ──────────────

    #[test]
    fn execution_mode_prefill_boundary_at_64() {
        let mode = ExecutionMode::Prefill { seq_len: 64 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::LargeTile));
    }

    // ── 20. GemmBlocking construction ────────────────────────────────

    #[test]
    fn gemm_blocking_construction_and_access() {
        let blocking = GemmBlocking { mr: 6, nr: 4, mc: 24, nc: 32, kc: 128 };
        assert_eq!(blocking.mr, 6);
        assert_eq!(blocking.nr, 4);
        assert_eq!(blocking.mc, 24);
        assert_eq!(blocking.nc, 32);
        assert_eq!(blocking.kc, 128);
    }

    // ── 21. SlotState equality ───────────────────────────────────────

    #[test]
    fn slot_state_equality() {
        assert_eq!(SlotState::Free, SlotState::Free);
        assert_eq!(SlotState::Occupied, SlotState::Occupied);
        assert_ne!(SlotState::Free, SlotState::Occupied);
    }

    // ── 22. SpillSlotInfo construction with all fields ───────────────

    #[test]
    fn spill_slot_info_construction() {
        let info = SpillSlotInfo {
            offset: 64,
            size: 32,
            owner: Some(1),
            vreg: Some(VRegId(5)),
            state: SlotState::Occupied,
        };
        assert_eq!(info.offset, 64);
        assert_eq!(info.size, 32);
        assert_eq!(info.owner, Some(1));
        assert_eq!(info.vreg, Some(VRegId(5)));
        assert_eq!(info.state, SlotState::Occupied);
    }

    // ── 23. GemmModeHint is Copy ─────────────────────────────────────

    #[test]
    fn gemm_mode_hint_is_copy() {
        let a = GemmModeHint::Gemv;
        let b = a; // Copy
        assert!(matches!(a, GemmModeHint::Gemv));
        assert!(matches!(b, GemmModeHint::Gemv));
    }

    // ── 24. PrefetchMode is Copy ─────────────────────────────────────

    #[test]
    fn prefetch_mode_is_copy() {
        let a = PrefetchMode::Aggressive;
        let b = a;
        assert!(matches!(a, PrefetchMode::Aggressive));
        assert!(matches!(b, PrefetchMode::Aggressive));
    }

    // ── 25. StackFrame zero callee_saves small frame ─────────────────

    #[test]
    fn stack_frame_zero_spills_small_frame() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        let frame = StackFrame::compute(&alloc, &profile, 0);
        assert_eq!(frame.total_size, 0);
        assert_eq!(frame.callee_save_area, 0);
        assert_eq!(frame.spill_area, 0);
    }

    // ── 26. ScopedSpillAllocator multiple allocs cumulative offset ───

    #[test]
    fn scoped_spill_multiple_allocs_cumulative() {
        let mut allocator = ScopedSpillAllocator::new();
        let (_, off0) = allocator.alloc(VRegId(0), 4, None);
        let (_, off1) = allocator.alloc(VRegId(1), 8, None);
        let (_, off2) = allocator.alloc(VRegId(2), 16, None);
        assert_eq!(off0, 0);
        assert_eq!(off1, 4);
        assert_eq!(off2, 12);
        assert_eq!(allocator.total_allocated(), 28);
    }

    // ── 27. ScopedSpillAllocator multi-scope only frees own slots ────

    #[test]
    fn scoped_spill_multi_scope_only_frees_own_slots() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope_a = allocator.scope_begin();
        let scope_b = allocator.scope_begin();
        allocator.alloc(VRegId(0), 8, Some(scope_a));
        allocator.alloc(VRegId(1), 16, Some(scope_b));
        assert_eq!(allocator.active_count(), 2);

        // Ending scope_b should only free slot belonging to scope_b
        allocator.scope_end();
        assert_eq!(allocator.active_count(), 1);
        // scope_a's slot is still occupied
        assert_eq!(allocator.current_scope(), Some(scope_a));

        allocator.scope_end();
        assert_eq!(allocator.active_count(), 0);
    }

    // ── 28. ScopedSpillAllocator global slot survives scope_end ──────

    #[test]
    fn scoped_spill_global_slot_survives_scope_end() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope = allocator.scope_begin();
        // Allocate a global slot (no scope_id)
        allocator.alloc(VRegId(0), 8, None);
        allocator.alloc(VRegId(1), 16, Some(scope));
        assert_eq!(allocator.active_count(), 2);

        // Ending scope only frees scope-owned slot, global slot persists
        allocator.scope_end();
        assert_eq!(allocator.active_count(), 1);
    }

    // ── 29. ScopedSpillAllocator free_list prefers reuse over new ────

    #[test]
    fn scoped_spill_free_list_prefers_reuse_over_new() {
        let mut allocator = ScopedSpillAllocator::new();
        let scope = allocator.scope_begin();
        allocator.alloc(VRegId(0), 8, Some(scope));
        allocator.alloc(VRegId(1), 8, Some(scope));
        allocator.scope_end(); // both freed

        // Allocate 3 slots of size 8 — first two should reuse freed slots
        let (idx0, _) = allocator.alloc(VRegId(2), 8, None);
        let (idx1, _) = allocator.alloc(VRegId(3), 8, None);
        let (idx2, _) = allocator.alloc(VRegId(4), 8, None);

        // idx0 and idx1 reuse freed slots (0 and 1), idx2 is new (2)
        assert!(
            (idx0 == 0 || idx0 == 1),
            "first alloc should reuse a freed slot, got idx={idx0}"
        );
        assert!(
            (idx1 == 0 || idx1 == 1) && idx1 != idx0,
            "second alloc should reuse the other freed slot, got idx={idx1}"
        );
        assert_eq!(idx2, 2, "third alloc must be a new slot");
        assert_eq!(allocator.active_count(), 3);
    }

    // ── 30. ScopedSpillAllocator zero size allocation ────────────────

    #[test]
    fn scoped_spill_zero_size_allocation() {
        let mut allocator = ScopedSpillAllocator::new();
        let (idx, off) = allocator.alloc(VRegId(0), 0, None);
        assert_eq!(idx, 0);
        assert_eq!(off, 0);
        // Zero-size slot should not advance total_allocated
        assert_eq!(allocator.total_allocated(), 0);
        assert_eq!(allocator.active_count(), 1);
    }

    // ── 31. StackFrame spill_offset calculation ──────────────────────

    #[test]
    fn stack_frame_spill_offset_calculation() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![
                SpillSlot { vreg: VRegId(0), offset: 0, size: 8 },
                SpillSlot { vreg: VRegId(1), offset: 8, size: 16 },
                SpillSlot { vreg: VRegId(2), offset: 24, size: 4 },
            ],
            callee_saved_used: vec![PhysGpr(3)], // 1 callee save = 8 bytes
        };
        let frame = StackFrame::compute(&alloc, &profile, 0);

        // spill_offset(0): base=callee_save_area(8), offset=0, result = -(8+0+8) = -16
        let off0 = frame.spill_offset(0, &alloc);
        assert_eq!(off0, -16);

        // spill_offset(1): base=8, offset=spills[0].size=8, result = -(8+8+8) = -24
        let off1 = frame.spill_offset(1, &alloc);
        assert_eq!(off1, -24);

        // spill_offset(2): base=8, offset=spills[0].size + spills[1].size = 8+16=24,
        // result = -(8+24+8) = -40
        let off2 = frame.spill_offset(2, &alloc);
        assert_eq!(off2, -40);
    }

    // ── 32. StackFrame with callee_saves and zero spills ─────────────

    #[test]
    fn stack_frame_callee_saves_no_spills() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![PhysGpr(3), PhysGpr(12), PhysGpr(13)],
        };
        let frame = StackFrame::compute(&alloc, &profile, 0);

        // 3 callee saves * 8 bytes = 24
        assert_eq!(frame.callee_save_area, 24);
        assert_eq!(frame.spill_area, 0);
        assert!(frame.total_size >= 24);
        assert_eq!(frame.total_size % frame.alignment, 0, "frame must be aligned");
    }

    // ── 33. StackFrame alignment rounds up ───────────────────────────

    #[test]
    fn stack_frame_alignment_rounds_up() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alignment = profile.abi.stack_alignment;

        // Create a frame where raw_size is deliberately NOT aligned
        // Use 1 callee save (8 bytes) + 1 spill of 3 bytes = 11 bytes raw
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![SpillSlot { vreg: VRegId(0), offset: 0, size: 3 }],
            callee_saved_used: vec![PhysGpr(3)],
        };
        let frame = StackFrame::compute(&alloc, &profile, 0);

        if alignment > 0 {
            let raw_size = 8 + 3; // callee_save + spill
            assert!(
                frame.total_size >= raw_size,
                "total_size must be at least raw_size ({raw_size}), got {}",
                frame.total_size
            );
            assert_eq!(
                frame.total_size % alignment, 0,
                "total_size must be aligned to {alignment}"
            );
        }
    }

    // ── 34. ExecutionMode decode batch_size>1 always SmallBatch ──────

    #[test]
    fn execution_mode_decode_large_batch_still_small_batch() {
        // batch_size=128 should still be SmallBatch (not Gemv)
        let mode = ExecutionMode::Decode { batch_size: 128 };
        assert!(matches!(mode.gemm_strategy(), GemmModeHint::SmallBatch));
        assert!(matches!(mode.prefetch_mode(), PrefetchMode::Aggressive));
    }

    // ── 35. ScopedSpillAllocator cross_scope slot outlives inner scope

    #[test]
    fn scoped_spill_cross_scope_slot_outlives_inner() {
        let mut allocator = ScopedSpillAllocator::new();
        let outer = allocator.scope_begin();
        // Allocate in outer scope
        allocator.alloc(VRegId(0), 8, Some(outer));
        let inner = allocator.scope_begin();
        // Allocate in inner scope
        allocator.alloc(VRegId(1), 16, Some(inner));
        assert_eq!(allocator.active_count(), 2);

        // End inner scope — only inner's slot is freed
        allocator.scope_end();
        assert_eq!(allocator.active_count(), 1);

        // Outer slot still accessible: reuse inner's freed slot
        let (idx, _off) = allocator.alloc(VRegId(2), 16, None);
        // Should reuse the freed inner slot (size=16 match), not outer's (size=8)
        assert_eq!(idx, 1, "should reuse freed inner slot with matching size");
        assert_eq!(allocator.active_count(), 2);
    }

    // ── 36. PressureModel analyze with Decode verifies register fields

    #[test]
    fn pressure_model_decode_register_pressure() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let model = PressureModel::analyze(&profile, ExecutionMode::Decode { batch_size: 4 });

        // Register counts must come from IsaProfile (not hardcoded)
        assert!(
            model.register.vec_available > 0,
            "vec_available must be positive"
        );
        assert!(
            model.register.gpr_available > 0,
            "gpr_available must be positive"
        );
        // Decode mode must yield SmallBatch + Aggressive
        assert!(matches!(model.gemm_strategy(), GemmModeHint::SmallBatch));
        assert!(matches!(model.prefetch_mode(), PrefetchMode::Aggressive));
        // Blocking parameters must be derived (not zero)
        assert!(model.blocking.mr > 0);
        assert!(model.blocking.nr > 0);
        assert!(model.blocking.kc >= 16);
    }
}
