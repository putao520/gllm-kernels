//! Graph Resource Planner — 全图资源预规划 (SPEC 28, 实验性)
//!
//! **状态**: 实验性模块，未集成到编译管线。当前编译路径使用 reg_alloc + stack_frame。
//! 集成需要: plan_mega_kernel_resources() 入口 + algo_interpreter 查询 + 模板消费。
//!
//! 在 VmInstr 发射前预计算完整资源布局：
//! - BufferLayout: 内存区间图着色 + ping-pong 双 buffer
//! - LoopInvariant: 循环不变量推导
//! - GroupPressure: 融合组寄存器压力曲线 (Wave 3)
//! - StackBlueprint: 栈帧蓝图 (Wave 4)
//! - ConcurrencyPartition: 并发资源分区 (Wave 3)
//!
//! 数据流向: CompilerGraph + FusionPlan → GraphResourcePlan → plan_lower 查询

use crate::compiler::buffer_alloc::BufferAllocation;
use crate::compiler::codegen::vm::isa_profile::IsaProfile;
use crate::compiler::codegen::vm::stack_frame::GemmBlocking;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::graph::{CompilerGraph, OpKind};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.2 BufferLayout — 内存区间图着色 (REQ-GRP-002)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scratchpad 内存布局 — 扩展现有 BufferAllocation 增加生命周期感知和 ping-pong。
///
/// 由 Phase 2 (内存区间图着色) 生成，plan_lower 在发射代码时查询
/// tensor→offset 映射和 ping-pong buffer 位置。
#[derive(Debug, Clone)]
pub struct BufferLayout {
    /// 基础 buffer 分配结果 (来自现有 interval coloring)
    pub base: BufferAllocation,
    /// 张量 ID → 内存区间映射
    pub memory_map: Vec<MemoryRegion>,
    /// Ping-pong 双 buffer 布局 (层间 activation 交替读写)
    pub ping_pong: Option<PingPongLayout>,
    /// Scratchpad 总字节数 (含所有复用后的实际大小)
    pub total_bytes: usize,
}

/// Scratchpad 内存区域 — 一段连续内存及其占用者。
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// 该区域在 scratchpad 中的偏移
    pub offset: usize,
    /// 区域大小 (字节)
    pub size_bytes: usize,
    /// 占用此区域的张量 ID 列表 (生命周期不重叠，可复用同一区域)
    pub tenants: Vec<crate::compiler::graph::TensorId>,
}

/// Ping-pong 双 buffer 布局 — 层间 activation 交替读写。
///
/// 层 N 读 ping → 计算 → 写 pong；层 N+1 读 pong → 计算 → 写 ping。
/// 消除层间 activation 拷贝，只需交换指针。
#[derive(Debug, Clone, Copy)]
pub struct PingPongLayout {
    /// 输入 buffer offset (层 N 读取的 activation)
    pub ping_offset: usize,
    /// 输出 buffer offset (层 N 写入的 activation)
    pub pong_offset: usize,
    /// 每个 buffer 的大小 (字节)
    pub buffer_bytes: usize,
}

impl BufferLayout {
    /// 根据 tensor_id 查询 scratchpad 偏移
    pub fn offset_for(&self, tid: crate::compiler::graph::TensorId) -> Option<usize> {
        self.base.slots.iter()
            .find(|s| s.tensor_id == tid)
            .map(|s| s.offset)
    }

    /// 获取 ping-pong 布局的 buffer 大小
    pub fn activation_bytes(&self) -> usize {
        self.ping_pong.map(|pp| pp.buffer_bytes).unwrap_or(0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.5 LoopInvariant — 循环不变量 (REQ-GRP-005)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 循环不变量 — 在层循环外计算一次，循环内只读。
///
/// 由 Phase 5 (循环不变量推导) 从 CompilerGraph 推导：
/// - 哪些值在所有层都相同 (如 cos/sin 表)
/// - 哪些权重偏移是常量 (如 PackMap 索引)
/// - 哪些标量参数每层相同 (如 eps, hidden_dim)
#[derive(Debug, Clone)]
pub struct LoopInvariant {
    /// 不变量类型
    pub kind: InvariantKind,
    /// 存放位置 (寄存器 or 栈)
    pub location: InvariantLocation,
    /// 计算方式 (在层循环外用什么指令算)
    pub computation: InvariantComputation,
}

/// 不变量类型 — 描述循环不变量的语义
#[derive(Debug, Clone)]
pub enum InvariantKind {
    /// RoPE cos/sin 表指针 (所有层共享)
    RopeTablePtr,
    /// RMSNorm scale/gamma 指针 (按层偏移)
    NormGammaPtr { layer_stride: usize },
    /// 模型配置常量 (hidden_dim, num_heads, etc.)
    ModelConfig { name: String, value: usize },
    /// PackMap 索引基址
    PackMapBase,
}

/// 不变量存放位置
#[derive(Debug, Clone)]
pub enum InvariantLocation {
    /// 固定 GPR (整个层循环生命周期)
    Gpr(usize),
    /// 栈 slot (rbp 偏移)
    Stack(i32),
}

/// 不变量计算方式 — 层循环前的初始化指令
#[derive(Debug, Clone, PartialEq)]
pub enum InvariantComputation {
    /// 从 ABI arg 加载 (arg index)
    LoadAbiArg(u8),
    /// 立即数加载
    LoadImm(u64),
    /// 指针运算: base + layer_stride * counter
    PtrArithmetic { base: u8, stride: usize },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.3 GroupPressure — 融合组寄存器压力 (REQ-GRP-003, Wave 3 实现)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 融合组的寄存器压力估计 — 指导代码发射策略。
///
/// 在发射每个融合组的代码之前查询此结构，决定：
/// - GEMM 展开度 (高压力 → 降低 mr/nr)
/// - 是否需要额外的 spill/reload
/// - 微核选择 (高压力用更小微核)
#[derive(Debug, Clone)]
pub struct GroupPressure {
    /// 融合组 ID
    pub group_id: usize,
    /// 该组峰值向量寄存器需求
    pub peak_vec_regs: usize,
    /// 该组峰值 GPR 需求
    pub peak_gpr_regs: usize,
    /// 该组可用的向量寄存器 (从总可用减去跨组活跃的)
    pub available_vec_regs: usize,
    /// 该组可用的 GPR (从总可用减去跨组活跃的)
    pub available_gpr_regs: usize,
    /// 建议的 GEMM 展开策略
    pub suggested_blocking: Option<GemmBlocking>,
    /// 是否需要该组前后插入 spill/reload
    pub needs_spill_fence: bool,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.4 StackBlueprint — 栈帧蓝图 (REQ-GRP-004, Wave 4 实现)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 栈帧蓝图 — 在代码发射前预计算完整栈布局。
///
/// 替代现有事后计算的 StackLayout，提供发射阶段可查询的栈规划。
#[derive(Debug, Clone)]
pub struct StackBlueprint {
    /// 总栈帧大小 (sub rsp, N 中的 N)
    pub total_frame_bytes: usize,
    /// ABI 参数 slots: [rbp + offset]
    pub abi_arg_slots: [Option<i32>; 6],
    /// Callee-save 寄存器 slots: (寄存器, rbp_offset)
    pub callee_save_slots: Vec<(u8, i32)>,
    /// Spill 区域起始偏移
    pub spill_base_rbp_off: i32,
    /// Spill slot 分配
    pub spill_slots: Vec<SpillSlot>,
    /// MXCSR 保存位置
    pub mxcsr_rsp_off: i32,
    /// Debug 探针 buffer (可选)
    pub debug_probe_region: Option<DebugProbeRegion>,
}

/// Spill slot 描述
#[derive(Debug, Clone)]
pub struct SpillSlot {
    /// rbp 偏移
    pub rbp_offset: i32,
    /// 大小 (字节)
    pub size_bytes: usize,
}

/// Debug 探针 buffer 区域
#[derive(Debug, Clone)]
pub struct DebugProbeRegion {
    /// rbp 偏移
    pub rbp_offset: i32,
    /// 大小 (字节)
    pub size_bytes: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.6 ConcurrencyPartition — 并发资源隔离 (REQ-GRP-006, Wave 3 实现)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 并发资源分区 — batch_size > 1 时区分 per-sequence 和共享资源。
#[derive(Debug, Clone)]
pub struct ConcurrencyPartition {
    /// Per-sequence 资源 (每条序列独占)
    pub per_sequence: PerSequenceResources,
    /// 共享资源 (所有序列共享)
    pub shared: SharedResources,
}

/// Per-sequence 资源
#[derive(Debug, Clone)]
pub struct PerSequenceResources {
    /// 每 sequence 的 scratchpad 区域大小
    pub scratchpad_bytes_per_seq: usize,
    /// 每 sequence 的 spill slot 数量
    pub spill_slots_per_seq: usize,
    /// KV cache 区域
    pub kv_cache_region: MemoryRegion,
    /// BatchSeqId → per-sequence data 的偏移映射
    pub seq_offset_map: SeqOffsetMap,
}

/// 共享资源
#[derive(Debug, Clone)]
pub struct SharedResources {
    /// 权重指针位置
    pub weight_ptr: InvariantLocation,
    /// 共享常量 (cos/sin 表, config 等)
    pub constants: Vec<InvariantLocation>,
    /// 全局 barrier/counter 区域
    pub global_counters: Vec<MemoryRegion>,
}

/// Per-sequence buffer 偏移映射
#[derive(Debug, Clone)]
pub struct SeqOffsetMap {
    /// activation buffer 偏移
    pub activation_offset: usize,
    /// KV cache 偏移
    pub kv_cache_offset: usize,
    /// 临时 buffer 偏移
    pub temp_buffer_offset: usize,
}

impl ConcurrencyPartition {
    /// batch_size=1 的单序列分区 (所有资源全局唯一)
    pub fn single_sequence(layout: &BufferLayout, stack: &StackBlueprint) -> Self {
        Self {
            per_sequence: PerSequenceResources {
                scratchpad_bytes_per_seq: layout.total_bytes,
                spill_slots_per_seq: stack.spill_slots.len(),
                kv_cache_region: MemoryRegion {
                    offset: 0,
                    size_bytes: 0,
                    tenants: vec![],
                },
                seq_offset_map: SeqOffsetMap {
                    activation_offset: 0,
                    kv_cache_offset: 0,
                    temp_buffer_offset: 0,
                },
            },
            shared: SharedResources {
                weight_ptr: InvariantLocation::Gpr(0),
                constants: vec![],
                global_counters: vec![],
            },
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.7 ResourceSummary + GraphResourcePlan
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 资源统计摘要
#[derive(Debug, Clone)]
pub struct ResourceSummary {
    pub total_scratchpad_bytes: usize,
    pub total_stack_bytes: usize,
    pub peak_vec_regs: usize,
    pub peak_gpr_regs: usize,
    pub num_layers: usize,
    pub num_buffer_reuse_slots: usize,
    pub bytes_saved_by_reuse: usize,
    pub num_loop_invariants: usize,
    pub batch_size: usize,
}

/// 全图资源规划 — 由 7 阶段管线生成的不可变数据。
///
/// plan_lower 在发射 VmInstr 前查询此结构获取所有资源决策。
#[derive(Debug, Clone)]
pub struct GraphResourcePlan {
    /// Buffer 布局 (区间图着色 + ping-pong)
    pub buffers: BufferLayout,
    /// 寄存器压力曲线 (per fusion group)
    pub pressure: Vec<GroupPressure>,
    /// 栈帧蓝图
    pub stack: StackBlueprint,
    /// 循环不变量
    pub loop_invariants: Vec<LoopInvariant>,
    /// 并发资源分区
    pub concurrency: ConcurrencyPartition,
    /// 资源统计摘要
    pub summary: ResourceSummary,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2.5 Phase 4: StackBlueprint — 栈帧蓝图计算 (REQ-GRP-004)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 从寄存器压力曲线推导 spill slot 需求，统一规划栈帧。
///
/// 布局 (x86_64 SysV ABI):
/// [高地址]
///   return address  [rbp+16] (pushed by call)
///   saved rbp       [rbp+8]  (pushed by push rbp)
///   ← rbp points here
///   ABI arg saves   [rbp-8..rbp-48]  (up to 6 × 8 bytes)
///   callee saves    [rbp-48..rbp-48-N] (rbx, r12-r15 etc.)
///   spill region    [spill_base..spill_base-spill_total]
///   debug probe     (optional)
/// [低地址] ← rsp after sub rsp, total_frame_bytes
pub fn plan_stack_blueprint(
    pressure: &[GroupPressure],
    num_abi_args: usize,
    num_callee_saves: usize,
    has_debug_probe: bool,
) -> StackBlueprint {
    // 计算最大 spill 需求
    let max_spill_vec = pressure.iter()
        .map(|g| g.peak_vec_regs.saturating_sub(g.available_vec_regs))
        .max()
        .unwrap_or(0);

    let max_spill_gpr = pressure.iter()
        .map(|g| g.peak_gpr_regs.saturating_sub(g.available_gpr_regs))
        .max()
        .unwrap_or(0);

    // 每条 spill 的向量寄存器需要 32/64 字节 (YMM/ZMM)
    let vec_spill_bytes = max_spill_vec * 64; // 按 ZMM 64B 保守估计
    let gpr_spill_bytes = max_spill_gpr * 8;
    let spill_total = vec_spill_bytes + gpr_spill_bytes;

    // ABI arg slots: 最多 6 个，每个 8 字节
    let mut abi_arg_slots: [Option<i32>; 6] = [None; 6];
    for i in 0..num_abi_args.min(6) {
        abi_arg_slots[i] = Some(-((i as i32 + 1) * 8));
    }

    // Callee-save slots: 紧接 ABI args
    let callee_save_base = -((num_abi_args.min(6) as i32 + 1) * 8);
    let callee_save_slots: Vec<(u8, i32)> = (0..num_callee_saves)
        .map(|i| (i as u8, callee_save_base - (i as i32 * 8)))
        .collect();

    // Spill region: 紧接 callee saves
    let spill_base_rbp_off = callee_save_base - (num_callee_saves as i32 * 8);

    // Spill slots
    let spill_slots: Vec<SpillSlot> = (0..max_spill_vec)
        .map(|i| SpillSlot {
            rbp_offset: spill_base_rbp_off - (i as i32 * 64) - 64,
            size_bytes: 64,
        })
        .chain((0..max_spill_gpr).map(|i| SpillSlot {
            rbp_offset: spill_base_rbp_off - (vec_spill_bytes as i32) - (i as i32 * 8) - 8,
            size_bytes: 8,
        }))
        .collect();

    // Debug probe (可选): 紧接 spill
    let debug_probe_region = if has_debug_probe {
        let probe_bytes = 4096; // 4KB probe buffer
        Some(DebugProbeRegion {
            rbp_offset: spill_base_rbp_off - (spill_total as i32) - (probe_bytes as i32),
            size_bytes: probe_bytes,
        })
    } else {
        None
    };

    let total_frame_bytes = (num_abi_args.min(6) * 8
        + num_callee_saves * 8
        + spill_total
        + debug_probe_region.as_ref().map(|r| r.size_bytes).unwrap_or(0)
        + 15) & !15; // 16-byte align

    StackBlueprint {
        total_frame_bytes,
        abi_arg_slots,
        callee_save_slots,
        spill_base_rbp_off,
        spill_slots,
        mxcsr_rsp_off: 0, // 将在 prologue 中计算
        debug_probe_region,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2.4 Phase 3: GroupPressure — 融合组寄存器压力估计 (REQ-GRP-003)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 融合组内寄存器需求估计
struct InnerDemand {
    vec_regs: usize,
    gpr_regs: usize,
    has_gemm: bool,
}

/// 估计每个融合组的峰值寄存器需求。
///
/// 算法：
/// 1. 分析组内所有 Op 的输入/输出数量
/// 2. 估计 GEMM 累加器需求 (mr × nr)
/// 3. 估计临时向量寄存器 (中间值)
/// 4. 估计 GPR 需求 (指针、循环计数器)
/// 5. 减去跨组活跃的寄存器 (前组输出仍活)
pub fn estimate_register_pressure(
    num_groups: usize,
    ops_per_group: &[usize],
    gemm_groups: &[bool],
    total_vec_regs: usize,
    total_gpr_regs: usize,
) -> Vec<GroupPressure> {
    (0..num_groups).map(|gid| {
        let op_count = ops_per_group.get(gid).copied().unwrap_or(0);
        let is_gemm = gemm_groups.get(gid).copied().unwrap_or(false);

        // 粗粒度估计: 每个 op ~2 vec regs + ~3 GPR (输入/输出/临时)
        let inner = InnerDemand {
            vec_regs: op_count.saturating_mul(2).max(4),
            gpr_regs: op_count.saturating_mul(3).max(6),
            has_gemm: is_gemm,
        };

        // GEMM 额外需求: 累加器 + packed panel 寄存器
        let vec_regs = if is_gemm { inner.vec_regs + 8 } else { inner.vec_regs };
        let gpr_regs = if is_gemm { inner.gpr_regs + 4 } else { inner.gpr_regs };

        let available_vec = total_vec_regs.saturating_sub(4); // 预留 callee-save
        let available_gpr = total_gpr_regs.saturating_sub(4);

        let needs_spill_fence = vec_regs > available_vec || gpr_regs > available_gpr;

        let suggested_blocking = if is_gemm {
            // 简化: 从可用 vec regs 推导 blocking
            let mr = (available_vec / 4).min(8).max(1);
            let nr = (available_vec / 4).min(4).max(1);
            Some(GemmBlocking { mc: mr * 4, nc: nr * 4, kc: 64, mr, nr })
        } else {
            None
        };

        GroupPressure {
            group_id: gid,
            peak_vec_regs: vec_regs,
            peak_gpr_regs: gpr_regs,
            available_vec_regs: available_vec,
            available_gpr_regs: available_gpr,
            suggested_blocking,
            needs_spill_fence,
        }
    }).collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2.7 Phase 6: ConcurrencyPartition — 并发资源分区 (REQ-GRP-006)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 根据 batch_size 决定资源分区策略。
///
/// - batch=1: 所有资源全局唯一
/// - batch>1: 区分 per-sequence (activation, KV) vs shared (weights, config)
pub fn partition_concurrency(
    buffer_layout: &BufferLayout,
    stack: &StackBlueprint,
    batch_size: usize,
    activation_bytes: usize,
    kv_bytes: usize,
) -> ConcurrencyPartition {
    if batch_size <= 1 {
        return ConcurrencyPartition::single_sequence(buffer_layout, stack);
    }

    let per_seq_scratchpad = buffer_layout.ping_pong
        .map(|pp| pp.buffer_bytes * 2)
        .unwrap_or(buffer_layout.total_bytes);

    ConcurrencyPartition {
        per_sequence: PerSequenceResources {
            scratchpad_bytes_per_seq: per_seq_scratchpad,
            spill_slots_per_seq: stack.spill_slots.len(),
            kv_cache_region: MemoryRegion {
                offset: activation_bytes,
                size_bytes: kv_bytes,
                tenants: vec![],
            },
            seq_offset_map: SeqOffsetMap {
                activation_offset: 0,
                kv_cache_offset: activation_bytes,
                temp_buffer_offset: activation_bytes + kv_bytes,
            },
        },
        shared: SharedResources {
            weight_ptr: InvariantLocation::Gpr(0),
            constants: vec![],
            global_counters: vec![],
        },
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2.8 Orchestration — build_resource_plan 编排入口 (REQ-GRP-007+008)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 编排全图资源规划的输入参数。
pub struct ResourcePlanInput {
    /// 基础 buffer 分配结果
    pub buffer_alloc: BufferAllocation,
    /// 融合组数量
    pub num_groups: usize,
    /// 每组算子数量
    pub ops_per_group: Vec<usize>,
    /// 每组是否包含 GEMM
    pub gemm_groups: Vec<bool>,
    /// 总可用向量寄存器数
    pub total_vec_regs: usize,
    /// 总可用 GPR 寄存器数
    pub total_gpr_regs: usize,
    /// ABI 参数数量 (x86_64: 最多 6)
    pub num_abi_args: usize,
    /// callee-save 寄存器数量
    pub num_callee_saves: usize,
    /// 是否启用 debug probe
    pub has_debug_probe: bool,
    /// 层循环不变量
    pub loop_invariants: Vec<LoopInvariant>,
    /// batch_size (1=单序列, >1=并发)
    pub batch_size: usize,
    /// 激活 buffer 字节数 (per sequence)
    pub activation_bytes: usize,
    /// KV cache 字节数 (per sequence)
    pub kv_bytes: usize,
}

/// 编排入口: 从输入参数生成完整的 GraphResourcePlan。
///
/// 执行顺序:
/// 1. BufferLayout (interval coloring + ping-pong)
/// 2. GroupPressure (register pressure estimation)
/// 3. StackBlueprint (from pressure → spill → frame layout)
/// 4. ConcurrencyPartition (per-sequence vs shared)
/// 5. ResourceSummary (aggregated stats)
pub fn build_resource_plan(input: ResourcePlanInput) -> GraphResourcePlan {
    // Phase 1: BufferLayout
    let ping_pong = if input.batch_size <= 1 {
        Some(PingPongLayout {
            ping_offset: 0,
            pong_offset: input.activation_bytes,
            buffer_bytes: input.activation_bytes,
        })
    } else {
        None
    };
    let total_bytes = input.buffer_alloc.total_bytes
        + ping_pong.map(|pp| pp.buffer_bytes).unwrap_or(0);
    let buffers = BufferLayout {
        base: input.buffer_alloc,
        memory_map: vec![],
        ping_pong,
        total_bytes,
    };

    // Phase 2: GroupPressure
    let pressure = estimate_register_pressure(
        input.num_groups,
        &input.ops_per_group,
        &input.gemm_groups,
        input.total_vec_regs,
        input.total_gpr_regs,
    );

    // Phase 3: StackBlueprint
    let stack = plan_stack_blueprint(
        &pressure,
        input.num_abi_args,
        input.num_callee_saves,
        input.has_debug_probe,
    );

    // Phase 4: ConcurrencyPartition
    let concurrency = partition_concurrency(
        &buffers,
        &stack,
        input.batch_size,
        input.activation_bytes,
        input.kv_bytes,
    );

    // Phase 5: Summary
    let bytes_saved_by_reuse = buffers.memory_map.iter().map(|r| r.size_bytes).sum::<usize>();
    let summary = ResourceSummary {
        total_scratchpad_bytes: buffers.total_bytes,
        total_stack_bytes: stack.total_frame_bytes,
        peak_vec_regs: pressure.iter().map(|g| g.peak_vec_regs).max().unwrap_or(0),
        peak_gpr_regs: pressure.iter().map(|g| g.peak_gpr_regs).max().unwrap_or(0),
        num_layers: input.num_groups,
        num_buffer_reuse_slots: buffers.memory_map.len(),
        bytes_saved_by_reuse,
        num_loop_invariants: input.loop_invariants.len(),
        batch_size: input.batch_size,
    };

    GraphResourcePlan {
        buffers,
        pressure,
        stack,
        loop_invariants: input.loop_invariants,
        concurrency,
        summary,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// REQ-GRP-008: plan_mega_kernel_resources — 编译管线集成入口
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 从编译管线实际数据构建 GraphResourcePlan。
///
/// 在 `compile_mega_kernel_from_graph` 中调用，位于 BufferAllocation 之后、
/// compile_mega_kernel_vm 之前。产出的 GraphResourcePlan 传递给模板解释器
/// 和 plan_lower 用于资源感知的代码发射。
pub fn plan_mega_kernel_resources(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    profile: &IsaProfile,
    alloc: &BufferAllocation,
    hidden_dim: usize,
    activation_bytes: usize,
    kv_bytes: usize,
) -> GraphResourcePlan {
    let ops_per_group: Vec<usize> = plan.groups.iter()
        .map(|g| g.ops.len())
        .collect();

    let gemm_groups: Vec<bool> = plan.groups.iter()
        .map(|g| {
            g.ops.iter().any(|&op_id| {
                graph.op(op_id).is_some_and(|op| {
                    matches!(op.kind, OpKind::Gemm { .. }
                        | OpKind::GemmBias { .. }
                        | OpKind::QuantGemm { .. })
                })
            })
        })
        .collect();

    let loop_invariants = derive_loop_invariants_from_graph(graph, hidden_dim);

    let total_vec_regs = profile.vec_regs.len();
    let total_gpr_regs = profile.gpr_regs.len();

    build_resource_plan(ResourcePlanInput {
        buffer_alloc: alloc.clone(),
        num_groups: plan.groups.len(),
        ops_per_group,
        gemm_groups,
        total_vec_regs,
        total_gpr_regs,
        num_abi_args: 6,
        num_callee_saves: profile.abi.callee_saved.len(),
        has_debug_probe: false,
        loop_invariants,
        batch_size: 1,
        activation_bytes,
        kv_bytes,
    })
}

/// 从 CompilerGraph 推导层循环不变量 (REQ-GRP-005)。
///
/// 分析图中哪些值在层循环中不变：
/// - RoPE cos/sin 表 (所有层共享)
/// - 模型配置常量 (hidden_dim, num_heads)
fn derive_loop_invariants_from_graph(graph: &CompilerGraph, hidden_dim: usize) -> Vec<LoopInvariant> {
    let mut invariants = Vec::new();

    let has_rope = graph.ops.iter().any(|op| {
        matches!(op.kind, OpKind::RoPE { .. })
    });
    if has_rope {
        invariants.push(LoopInvariant {
            kind: InvariantKind::RopeTablePtr,
            location: InvariantLocation::Stack(-256),
            computation: InvariantComputation::LoadAbiArg(0),
        });
    }

    invariants.push(LoopInvariant {
        kind: InvariantKind::ModelConfig {
            name: "hidden_dim".into(),
            value: hidden_dim,
        },
        location: InvariantLocation::Stack(-264),
        computation: InvariantComputation::LoadImm(hidden_dim as u64),
    });

    let has_packed = graph.ops.iter().any(|op| {
        matches!(op.kind,
            OpKind::QuantGemm { .. }
            | OpKind::Gather { .. }
        )
    });
    if has_packed {
        invariants.push(LoopInvariant {
            kind: InvariantKind::PackMapBase,
            location: InvariantLocation::Gpr(0),
            computation: InvariantComputation::LoadAbiArg(1),
        });
    }

    invariants
}

impl GraphResourcePlan {
    /// 查询指定融合组的建议 GEMM blocking 策略。
    pub fn gemm_blocking_for_group(&self, group_id: usize) -> Option<&GemmBlocking> {
        self.pressure.get(group_id)
            .and_then(|g| g.suggested_blocking.as_ref())
    }

    /// 查询指定融合组是否需要 spill fence。
    pub fn group_needs_spill(&self, group_id: usize) -> bool {
        self.pressure.get(group_id)
            .map(|g| g.needs_spill_fence)
            .unwrap_or(false)
    }

    /// 查询循环不变量列表。
    pub fn loop_invariant_by_index(&self, idx: usize) -> Option<&LoopInvariant> {
        self.loop_invariants.get(idx)
    }

    /// 总 scratchpad 内存（含 ping-pong 双 buffer）。
    pub fn total_scratchpad_with_pingpong(&self) -> usize {
        self.buffers.total_bytes
            + self.buffers.ping_pong.map(|pp| pp.buffer_bytes).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_resource_plan_single_sequence() {
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 4,
            ops_per_group: vec![3, 2, 5, 1],
            gemm_groups: vec![true, false, true, false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 6,
            num_callee_saves: 5,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 4096,
            kv_bytes: 8192,
        });

        assert_eq!(plan.summary.num_layers, 4);
        assert_eq!(plan.summary.batch_size, 1);
        assert!(plan.summary.peak_vec_regs > 0);
        assert!(plan.buffers.ping_pong.is_some());
        assert!(plan.gemm_blocking_for_group(0).is_some());
        assert!(!plan.gemm_blocking_for_group(1).is_some());
    }

    #[test]
    fn test_loop_invariant_lookup() {
        let inv = LoopInvariant {
            kind: InvariantKind::RopeTablePtr,
            location: InvariantLocation::Stack(-256),
            computation: InvariantComputation::LoadAbiArg(0),
        };

        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![2],
            gemm_groups: vec![false],
            total_vec_regs: 16,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![inv],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        assert!(plan.loop_invariant_by_index(0).is_some());
        assert!(plan.loop_invariant_by_index(1).is_none());
        assert_eq!(plan.loop_invariant_by_index(0).unwrap().computation, InvariantComputation::LoadAbiArg(0));
    }

    #[test]
    fn test_plan_mega_kernel_resources_from_graph() {
        use crate::compiler::graph::{CompilerGraph, OpKind, SymDim};
        use crate::compiler::fusion::FusionPlan;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::dispatch::device_profile::DeviceProfile;
        use crate::types::DType;

        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());

        let mut graph = CompilerGraph::new();
        let hidden = 512;

        let ids_tok = graph.add_tensor_concrete("ids", &[1], DType::F32);
        let embed_w = graph.add_tensor_concrete("embed_w", &[32000, hidden], DType::F32);
        let embed_out = graph.add_tensor_concrete("embed_out", &[1, hidden], DType::F32);
        let q_w = graph.add_tensor_concrete("q_w", &[hidden, hidden], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[1, hidden], DType::F32);
        let rope_out = graph.add_tensor_concrete("rope_out", &[1, hidden], DType::F32);

        let embed = graph.add_op(
            OpKind::Gather {
                table_rows: 32000, embed_dim: hidden,
                index_dim: SymDim::Concrete(1),
                indices_kind: crate::compiler::graph::GatherIndicesKind::default(),
            },
            vec![ids_tok, embed_w], vec![embed_out], "embed",
        );
        let q_proj = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1), n: hidden, k: hidden,
                dtype: crate::types::DType::F32,
                trans_b: true,
            },
            vec![embed_out, q_w], vec![q_out], "q_proj",
        );
        let rope = graph.add_op(
            OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![q_out], vec![rope_out], "rope",
        );

        let plan = FusionPlan {
            groups: vec![
                crate::compiler::fusion::FusionGroup {
                    id: 0, anchor: embed, epilogue: vec![],
                    mode: crate::compiler::fusion::FusionMode::Standalone,
                    ops: vec![embed], multi_output: Default::default(), dominant_dtype: None,
                },
                crate::compiler::fusion::FusionGroup {
                    id: 1, anchor: q_proj, epilogue: vec![rope],
                    mode: crate::compiler::fusion::FusionMode::EpilogueInjection,
                    ops: vec![q_proj, rope], multi_output: Default::default(), dominant_dtype: None,
                },
            ],
            op_to_group: [(embed, 0), (q_proj, 1), (rope, 1)].into_iter().collect(),
        };

        let alloc = BufferAllocation::default();
        let activation_bytes = 512 * 4;
        let kv_bytes = 512 * 2;

        let resource_plan = plan_mega_kernel_resources(
            &graph, &plan, &profile, &alloc,
            hidden, activation_bytes, kv_bytes,
        );

        assert_eq!(resource_plan.summary.num_layers, 2);
        assert_eq!(resource_plan.pressure.len(), 2);
        assert!(resource_plan.summary.peak_vec_regs > 0);
        assert!(resource_plan.summary.peak_gpr_regs > 0);
        assert_eq!(resource_plan.summary.batch_size, 1);
        assert!(resource_plan.summary.total_scratchpad_bytes > 0);
        assert!(resource_plan.summary.total_stack_bytes > 0);

        assert!(!resource_plan.gemm_blocking_for_group(0).is_some());
        assert!(resource_plan.gemm_blocking_for_group(1).is_some());

        let has_rope = resource_plan.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::RopeTablePtr));
        assert!(has_rope, "RoPE op should produce RopeTablePtr invariant");

        let has_hidden = resource_plan.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::ModelConfig { .. }));
        assert!(has_hidden, "hidden_dim should produce ModelConfig invariant");
    }

    #[test]
    fn test_resource_plan_ping_pong_layout() {
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::dispatch::device_profile::DeviceProfile;

        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());

        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 2,
            ops_per_group: vec![3, 3],
            gemm_groups: vec![true, true],
            total_vec_regs: profile.vec_regs.len(),
            total_gpr_regs: profile.gpr_regs.len(),
            num_abi_args: 6,
            num_callee_saves: profile.abi.callee_saved.len(),
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 2048,
            kv_bytes: 4096,
        });

        assert!(plan.buffers.ping_pong.is_some());
        let pp = plan.buffers.ping_pong.unwrap();
        assert_eq!(pp.buffer_bytes, 2048);
        assert_ne!(pp.ping_offset, pp.pong_offset);

        assert!(plan.total_scratchpad_with_pingpong() > 0);
    }
}

#[cfg(test)]
mod data_structure_tests {
    use super::*;
    use crate::compiler::buffer_alloc::BufferSlot;
    use crate::compiler::TensorId;

    // ── 1. BufferLayout ───────────────────────────────────────────────

    #[test]
    fn buffer_layout_offset_for_returns_none_for_unknown_tensor() {
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 0,
        };

        assert!(layout.offset_for(TensorId(99)).is_none());
    }

    #[test]
    fn buffer_layout_offset_for_returns_some_for_known_tensor() {
        let tid = TensorId(0);
        let layout = BufferLayout {
            base: BufferAllocation {
                slots: vec![BufferSlot {
                    tensor_id: tid,
                    offset: 128,
                    size_bytes: 64,
                }],
                ..BufferAllocation::default()
            },
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 192,
        };

        assert_eq!(layout.offset_for(tid), Some(128));
    }

    #[test]
    fn buffer_layout_activation_bytes_returns_zero_without_ping_pong() {
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 1024,
        };

        assert_eq!(layout.activation_bytes(), 0);
    }

    #[test]
    fn buffer_layout_activation_bytes_returns_buffer_size_with_ping_pong() {
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: Some(PingPongLayout {
                ping_offset: 0,
                pong_offset: 4096,
                buffer_bytes: 4096,
            }),
            total_bytes: 8192,
        };

        assert_eq!(layout.activation_bytes(), 4096);
    }

    // ── 2. PingPongLayout ─────────────────────────────────────────────

    #[test]
    fn ping_pong_layout_construction_and_different_offsets() {
        let pp = PingPongLayout {
            ping_offset: 0,
            pong_offset: 8192,
            buffer_bytes: 8192,
        };

        assert_ne!(pp.ping_offset, pp.pong_offset);
        assert_eq!(pp.buffer_bytes, 8192);
        assert_eq!(pp.pong_offset - pp.ping_offset, pp.buffer_bytes);
    }

    // ── 3. MemoryRegion ───────────────────────────────────────────────

    #[test]
    fn memory_region_construction_and_tenants() {
        let region = MemoryRegion {
            offset: 256,
            size_bytes: 1024,
            tenants: vec![TensorId(1), TensorId(2)],
        };

        assert_eq!(region.offset, 256);
        assert_eq!(region.size_bytes, 1024);
        assert_eq!(region.tenants.len(), 2);
        assert_eq!(region.tenants[0], TensorId(1));
        assert_eq!(region.tenants[1], TensorId(2));
    }

    // ── 4. LoopInvariant + 5. InvariantKind ───────────────────────────

    #[test]
    fn invariant_kind_all_variants_construct_correctly() {
        let rope = InvariantKind::RopeTablePtr;
        let norm = InvariantKind::NormGammaPtr { layer_stride: 512 };
        let config = InvariantKind::ModelConfig {
            name: "hidden_dim".into(),
            value: 768,
        };
        let pack = InvariantKind::PackMapBase;

        assert!(matches!(rope, InvariantKind::RopeTablePtr));
        assert!(matches!(norm, InvariantKind::NormGammaPtr { layer_stride: 512 }));
        assert!(matches!(config, InvariantKind::ModelConfig { name: _, value: 768 }));
        assert!(matches!(pack, InvariantKind::PackMapBase));
    }

    #[test]
    fn loop_invariant_construction_with_rope_table_ptr() {
        let inv = LoopInvariant {
            kind: InvariantKind::RopeTablePtr,
            location: InvariantLocation::Stack(-256),
            computation: InvariantComputation::LoadAbiArg(0),
        };

        assert!(matches!(inv.kind, InvariantKind::RopeTablePtr));
        assert!(matches!(inv.location, InvariantLocation::Stack(-256)));
        assert_eq!(inv.computation, InvariantComputation::LoadAbiArg(0));
    }

    #[test]
    fn loop_invariant_construction_with_norm_gamma_ptr() {
        let inv = LoopInvariant {
            kind: InvariantKind::NormGammaPtr { layer_stride: 1024 },
            location: InvariantLocation::Gpr(12),
            computation: InvariantComputation::PtrArithmetic { base: 2, stride: 1024 },
        };

        assert!(matches!(inv.kind, InvariantKind::NormGammaPtr { layer_stride: 1024 }));
        assert!(matches!(inv.location, InvariantLocation::Gpr(12)));
        assert_eq!(inv.computation, InvariantComputation::PtrArithmetic { base: 2, stride: 1024 });
    }

    // ── 6. InvariantLocation ──────────────────────────────────────────

    #[test]
    fn invariant_location_gpr_and_stack_variants() {
        let gpr = InvariantLocation::Gpr(5);
        let stack = InvariantLocation::Stack(-128);

        assert!(matches!(gpr, InvariantLocation::Gpr(5)));
        assert!(matches!(stack, InvariantLocation::Stack(-128)));
    }

    // ── 7. InvariantComputation ───────────────────────────────────────

    #[test]
    fn invariant_computation_all_variants_and_equality() {
        let load_abi = InvariantComputation::LoadAbiArg(3);
        let load_imm = InvariantComputation::LoadImm(42);
        let ptr_arith = InvariantComputation::PtrArithmetic { base: 1, stride: 256 };

        assert_eq!(load_abi, InvariantComputation::LoadAbiArg(3));
        assert_ne!(load_abi, InvariantComputation::LoadAbiArg(2));
        assert_eq!(load_imm, InvariantComputation::LoadImm(42));
        assert_eq!(ptr_arith, InvariantComputation::PtrArithmetic { base: 1, stride: 256 });
        assert_ne!(ptr_arith, InvariantComputation::PtrArithmetic { base: 1, stride: 128 });
    }

    // ── 8. GroupPressure ──────────────────────────────────────────────

    #[test]
    fn group_pressure_construction_all_fields() {
        let blocking = GemmBlocking { mr: 4, nr: 4, mc: 16, nc: 16, kc: 64 };
        let gp = GroupPressure {
            group_id: 2,
            peak_vec_regs: 20,
            peak_gpr_regs: 10,
            available_vec_regs: 28,
            available_gpr_regs: 12,
            suggested_blocking: Some(blocking.clone()),
            needs_spill_fence: false,
        };

        assert_eq!(gp.group_id, 2);
        assert_eq!(gp.peak_vec_regs, 20);
        assert_eq!(gp.peak_gpr_regs, 10);
        assert_eq!(gp.available_vec_regs, 28);
        assert_eq!(gp.available_gpr_regs, 12);
        assert!(gp.suggested_blocking.is_some());
        assert_eq!(gp.suggested_blocking.as_ref().unwrap().mr, 4);
        assert!(!gp.needs_spill_fence);
    }

    // ── 9. StackBlueprint ─────────────────────────────────────────────

    #[test]
    fn stack_blueprint_construction_and_field_access() {
        let bp = StackBlueprint {
            total_frame_bytes: 512,
            abi_arg_slots: [Some(-8), Some(-16), None, None, None, None],
            callee_save_slots: vec![(3u8, -56), (12u8, -64)],
            spill_base_rbp_off: -72,
            spill_slots: vec![
                SpillSlot { rbp_offset: -136, size_bytes: 64 },
            ],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        assert_eq!(bp.total_frame_bytes, 512);
        assert_eq!(bp.abi_arg_slots[0], Some(-8));
        assert_eq!(bp.callee_save_slots.len(), 2);
        assert_eq!(bp.spill_base_rbp_off, -72);
        assert_eq!(bp.spill_slots.len(), 1);
        assert!(bp.debug_probe_region.is_none());
    }

    // ── 10. SpillSlot ─────────────────────────────────────────────────

    #[test]
    fn spill_slot_construction() {
        let slot = SpillSlot {
            rbp_offset: -200,
            size_bytes: 64,
        };

        assert_eq!(slot.rbp_offset, -200);
        assert_eq!(slot.size_bytes, 64);
    }

    // ── 11. DebugProbeRegion ──────────────────────────────────────────

    #[test]
    fn debug_probe_region_construction() {
        let probe = DebugProbeRegion {
            rbp_offset: -4096,
            size_bytes: 4096,
        };

        assert_eq!(probe.rbp_offset, -4096);
        assert_eq!(probe.size_bytes, 4096);
    }

    // ── 12. ConcurrencyPartition::single_sequence ─────────────────────

    #[test]
    fn concurrency_partition_single_sequence() {
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: Some(PingPongLayout {
                ping_offset: 0,
                pong_offset: 4096,
                buffer_bytes: 4096,
            }),
            total_bytes: 8192,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 256,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -48,
            spill_slots: vec![
                SpillSlot { rbp_offset: -112, size_bytes: 64 },
                SpillSlot { rbp_offset: -176, size_bytes: 64 },
            ],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        let partition = ConcurrencyPartition::single_sequence(&layout, &stack);

        assert_eq!(partition.per_sequence.scratchpad_bytes_per_seq, 8192);
        assert_eq!(partition.per_sequence.spill_slots_per_seq, 2);
        assert_eq!(partition.per_sequence.kv_cache_region.offset, 0);
        assert_eq!(partition.per_sequence.kv_cache_region.size_bytes, 0);
        assert_eq!(partition.per_sequence.seq_offset_map.activation_offset, 0);
        assert!(matches!(partition.shared.weight_ptr, InvariantLocation::Gpr(0)));
        assert!(partition.shared.constants.is_empty());
        assert!(partition.shared.global_counters.is_empty());
    }

    // ── 13. SeqOffsetMap ──────────────────────────────────────────────

    #[test]
    fn seq_offset_map_construction() {
        let map = SeqOffsetMap {
            activation_offset: 0,
            kv_cache_offset: 8192,
            temp_buffer_offset: 16384,
        };

        assert_eq!(map.activation_offset, 0);
        assert_eq!(map.kv_cache_offset, 8192);
        assert_eq!(map.temp_buffer_offset, 16384);
    }

    // ── 14. PerSequenceResources ───────────────────────────────────────

    #[test]
    fn per_sequence_resources_construction() {
        let res = PerSequenceResources {
            scratchpad_bytes_per_seq: 4096,
            spill_slots_per_seq: 3,
            kv_cache_region: MemoryRegion {
                offset: 4096,
                size_bytes: 8192,
                tenants: vec![TensorId(0)],
            },
            seq_offset_map: SeqOffsetMap {
                activation_offset: 0,
                kv_cache_offset: 4096,
                temp_buffer_offset: 12288,
            },
        };

        assert_eq!(res.scratchpad_bytes_per_seq, 4096);
        assert_eq!(res.spill_slots_per_seq, 3);
        assert_eq!(res.kv_cache_region.size_bytes, 8192);
        assert_eq!(res.seq_offset_map.kv_cache_offset, 4096);
    }

    // ── 15. SharedResources ───────────────────────────────────────────

    #[test]
    fn shared_resources_construction() {
        let shared = SharedResources {
            weight_ptr: InvariantLocation::Gpr(13),
            constants: vec![
                InvariantLocation::Stack(-256),
                InvariantLocation::Stack(-320),
            ],
            global_counters: vec![MemoryRegion {
                offset: 0,
                size_bytes: 64,
                tenants: vec![],
            }],
        };

        assert!(matches!(shared.weight_ptr, InvariantLocation::Gpr(13)));
        assert_eq!(shared.constants.len(), 2);
        assert_eq!(shared.global_counters.len(), 1);
    }

    // ── 16. ResourceSummary ───────────────────────────────────────────

    #[test]
    fn resource_summary_construction_all_fields() {
        let summary = ResourceSummary {
            total_scratchpad_bytes: 65536,
            total_stack_bytes: 512,
            peak_vec_regs: 24,
            peak_gpr_regs: 12,
            num_layers: 32,
            num_buffer_reuse_slots: 8,
            bytes_saved_by_reuse: 32768,
            num_loop_invariants: 5,
            batch_size: 1,
        };

        assert_eq!(summary.total_scratchpad_bytes, 65536);
        assert_eq!(summary.total_stack_bytes, 512);
        assert_eq!(summary.peak_vec_regs, 24);
        assert_eq!(summary.peak_gpr_regs, 12);
        assert_eq!(summary.num_layers, 32);
        assert_eq!(summary.num_buffer_reuse_slots, 8);
        assert_eq!(summary.bytes_saved_by_reuse, 32768);
        assert_eq!(summary.num_loop_invariants, 5);
        assert_eq!(summary.batch_size, 1);
    }

    // ── 17. estimate_register_pressure — zero groups ──────────────────

    #[test]
    fn estimate_pressure_zero_groups_returns_empty() {
        let result = estimate_register_pressure(0, &[], &[], 32, 16);
        assert!(result.is_empty());
    }

    // ── 18. estimate_register_pressure — GEMM group gets blocking ─────

    #[test]
    fn estimate_pressure_gemm_group_has_suggested_blocking() {
        let result = estimate_register_pressure(
            2,
            &[3, 4],
            &[false, true],
            32,
            16,
        );
        assert!(result[0].suggested_blocking.is_none(), "non-GEMM group should have no blocking");
        assert!(result[1].suggested_blocking.is_some(), "GEMM group should have blocking");
        let blocking = result[1].suggested_blocking.as_ref().unwrap();
        assert!(blocking.mr > 0);
        assert!(blocking.nr > 0);
        assert!(blocking.kc > 0);
    }

    // ── 19. estimate_register_pressure — spill fence detection ────────

    #[test]
    fn estimate_pressure_detects_spill_fence_when_oversubscribed() {
        // 10 ops => 20 vec regs min + 8 GEMM = 28; total_vec=16 => available=12
        let result = estimate_register_pressure(
            1,
            &[10],
            &[true],
            16,
            8,
        );
        assert!(result[0].needs_spill_fence, "oversubscribed group should need spill fence");
    }

    // ── 20. plan_stack_blueprint — no args, no debug probe ────────────

    #[test]
    fn plan_stack_no_args_no_probe() {
        let bp = plan_stack_blueprint(&[], 0, 0, false);
        assert_eq!(bp.abi_arg_slots, [None; 6]);
        assert!(bp.callee_save_slots.is_empty());
        assert!(bp.spill_slots.is_empty());
        assert!(bp.debug_probe_region.is_none());
        assert_eq!(bp.total_frame_bytes % 16, 0, "frame should be 16-byte aligned");
    }

    // ── 21. plan_stack_blueprint — with debug probe ───────────────────

    #[test]
    fn plan_stack_with_debug_probe() {
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 10,
            peak_gpr_regs: 5,
            available_vec_regs: 28,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: false,
        }];
        let bp = plan_stack_blueprint(&pressure, 6, 4, true);
        assert!(bp.debug_probe_region.is_some());
        let probe = bp.debug_probe_region.unwrap();
        assert_eq!(probe.size_bytes, 4096);
        assert!(probe.rbp_offset < bp.spill_base_rbp_off);
    }

    // ── 22. plan_stack_blueprint — ABI arg slots positioned correctly ─

    #[test]
    fn plan_stack_abi_arg_slots_correct_offsets() {
        let bp = plan_stack_blueprint(&[], 4, 2, false);
        assert_eq!(bp.abi_arg_slots[0], Some(-8));
        assert_eq!(bp.abi_arg_slots[1], Some(-16));
        assert_eq!(bp.abi_arg_slots[2], Some(-24));
        assert_eq!(bp.abi_arg_slots[3], Some(-32));
        assert_eq!(bp.abi_arg_slots[4], None);
        assert_eq!(bp.abi_arg_slots[5], None);
        assert_eq!(bp.callee_save_slots.len(), 2);
    }

    // ── 23. partition_concurrency — batch>1 uses per-sequence offsets ─

    #[test]
    fn partition_concurrency_batch_gt_1_has_offsets() {
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: Some(PingPongLayout {
                ping_offset: 0,
                pong_offset: 4096,
                buffer_bytes: 4096,
            }),
            total_bytes: 8192,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 256,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -48,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        let partition = partition_concurrency(&layout, &stack, 4, 4096, 8192);

        assert_eq!(partition.per_sequence.seq_offset_map.activation_offset, 0);
        assert_eq!(partition.per_sequence.seq_offset_map.kv_cache_offset, 4096);
        assert_eq!(partition.per_sequence.seq_offset_map.temp_buffer_offset, 4096 + 8192);
        assert_eq!(partition.per_sequence.kv_cache_region.offset, 4096);
        assert_eq!(partition.per_sequence.kv_cache_region.size_bytes, 8192);
    }

    // ── 24. partition_concurrency — batch=1 delegates to single_sequence ─

    #[test]
    fn partition_concurrency_batch_1_is_single_sequence() {
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 1024,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 128,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -32,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        let partition = partition_concurrency(&layout, &stack, 1, 512, 1024);
        assert_eq!(partition.per_sequence.scratchpad_bytes_per_seq, 1024);
        assert_eq!(partition.per_sequence.kv_cache_region.size_bytes, 0);
    }

    // ── 25. GraphResourcePlan query methods — out of range ────────────

    #[test]
    fn resource_plan_query_out_of_range_returns_none() {
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 2,
            ops_per_group: vec![1, 1],
            gemm_groups: vec![false, false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 6,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        assert!(plan.gemm_blocking_for_group(99).is_none());
        assert!(!plan.group_needs_spill(99));
        assert!(plan.loop_invariant_by_index(99).is_none());
    }

    // ── 26. build_resource_plan — batch>1 disables ping_pong ──────────

    #[test]
    fn build_plan_batch_gt_1_no_ping_pong() {
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![2],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 4,
            activation_bytes: 2048,
            kv_bytes: 4096,
        });

        assert!(plan.buffers.ping_pong.is_none(), "batch>1 should not have ping-pong");
        assert_eq!(plan.summary.batch_size, 4);
    }

    // ── 27. PingPongLayout is Copy ────────────────────────────────────

    #[test]
    fn ping_pong_layout_is_copy() {
        let pp = PingPongLayout {
            ping_offset: 0,
            pong_offset: 4096,
            buffer_bytes: 4096,
        };
        let pp2 = pp;
        assert_eq!(pp.ping_offset, pp2.ping_offset);
        assert_eq!(pp.pong_offset, pp2.pong_offset);
    }

    // ── 28. GraphResourcePlan total_scratchpad_with_pingpong ──────────

    #[test]
    fn total_scratchpad_includes_ping_pong_bytes() {
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 8192,
            kv_bytes: 16384,
        });

        assert!(plan.buffers.ping_pong.is_some());
        let pp_bytes = plan.buffers.ping_pong.unwrap().buffer_bytes;
        assert!(plan.total_scratchpad_with_pingpong() >= plan.buffers.total_bytes + pp_bytes);
    }

    // ── 29. InvariantComputation equality semantics ───────────────────

    #[test]
    fn invariant_computation_ptr_arithmetic_equality() {
        let a = InvariantComputation::PtrArithmetic { base: 3, stride: 512 };
        let b = InvariantComputation::PtrArithmetic { base: 3, stride: 512 };
        let c = InvariantComputation::PtrArithmetic { base: 3, stride: 256 };
        let d = InvariantComputation::PtrArithmetic { base: 1, stride: 512 };

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    // ── 30. plan_stack_blueprint — spill slots created from pressure ──

    #[test]
    fn plan_stack_creates_spill_slots_from_oversubscribed_pressure() {
        // Arrange: 2 groups, group 0 needs 20 vec / 8 gpr more than available
        let pressure = vec![
            GroupPressure {
                group_id: 0,
                peak_vec_regs: 36,
                peak_gpr_regs: 12,
                available_vec_regs: 16,
                available_gpr_regs: 8,
                suggested_blocking: None,
                needs_spill_fence: true,
            },
            GroupPressure {
                group_id: 1,
                peak_vec_regs: 10,
                peak_gpr_regs: 4,
                available_vec_regs: 16,
                available_gpr_regs: 8,
                suggested_blocking: None,
                needs_spill_fence: false,
            },
        ];

        // Act
        let bp = plan_stack_blueprint(&pressure, 6, 4, false);

        // Assert: max_spill_vec = max(36-16, 10-16 clamped) = 20, max_spill_gpr = max(12-8, 4-8 clamped) = 4
        let vec_spill_count = bp.spill_slots.iter().filter(|s| s.size_bytes == 64).count();
        let gpr_spill_count = bp.spill_slots.iter().filter(|s| s.size_bytes == 8).count();
        assert_eq!(vec_spill_count, 20, "should have 20 ZMM spill slots");
        assert_eq!(gpr_spill_count, 4, "should have 4 GPR spill slots");
        assert!(bp.total_frame_bytes > 0);
        assert_eq!(bp.total_frame_bytes % 16, 0, "frame must be 16-byte aligned");
    }

    // ── 31. estimate_register_pressure — asymmetric groups ────────────

    #[test]
    fn estimate_pressure_asymmetric_groups_correct_peak() {
        // Arrange: 3 groups with different sizes and GEMM flags
        let result = estimate_register_pressure(
            3,
            &[1, 8, 3],
            &[false, true, false],
            32,
            16,
        );

        // Assert: group 0 has 1 op → max(2,4)=4 vec; group 1 has 8 ops + GEMM → max(16,4)+8=24 vec
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].peak_vec_regs, 4, "1 op → max(2,4) = 4");
        assert_eq!(result[1].peak_vec_regs, 24, "8 ops → max(16,4)+8 GEMM = 24");
        assert_eq!(result[2].peak_vec_regs, 6, "3 ops → max(6,4) = 6");

        // group 1 is the only GEMM group
        assert!(result[0].suggested_blocking.is_none());
        assert!(result[1].suggested_blocking.is_some());
        assert!(result[2].suggested_blocking.is_none());
    }

    // ── 32. partition_concurrency — batch>1 without ping_pong ────────

    #[test]
    fn partition_concurrency_batch_gt_1_without_ping_pong_uses_total_bytes() {
        // Arrange: layout without ping_pong, batch_size=2
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 4096,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 128,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -32,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        // Act
        let partition = partition_concurrency(&layout, &stack, 2, 2048, 4096);

        // Assert: without ping_pong, per_seq_scratchpad = total_bytes (not buffer_bytes*2)
        assert_eq!(partition.per_sequence.scratchpad_bytes_per_seq, 4096);
        assert_eq!(partition.per_sequence.seq_offset_map.temp_buffer_offset, 2048 + 4096);
    }

    // ── 33. build_resource_plan — summary aggregates multiple invariants ─

    #[test]
    fn build_resource_plan_summary_counts_multiple_invariants() {
        // Arrange: 3 loop invariants
        let invariants = vec![
            LoopInvariant {
                kind: InvariantKind::RopeTablePtr,
                location: InvariantLocation::Stack(-256),
                computation: InvariantComputation::LoadAbiArg(0),
            },
            LoopInvariant {
                kind: InvariantKind::ModelConfig { name: "hidden_dim".into(), value: 768 },
                location: InvariantLocation::Stack(-264),
                computation: InvariantComputation::LoadImm(768),
            },
            LoopInvariant {
                kind: InvariantKind::PackMapBase,
                location: InvariantLocation::Gpr(3),
                computation: InvariantComputation::LoadAbiArg(1),
            },
        ];

        // Act
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![2],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: invariants,
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert
        assert_eq!(plan.summary.num_loop_invariants, 3);
        assert!(plan.loop_invariant_by_index(0).is_some());
        assert!(plan.loop_invariant_by_index(1).is_some());
        assert!(plan.loop_invariant_by_index(2).is_some());
        assert!(matches!(
            plan.loop_invariant_by_index(2).unwrap().kind,
            InvariantKind::PackMapBase,
        ));
    }

    // ── 34. plan_mega_kernel_resources — graph with QuantGemm produces PackMapBase ─

    #[test]
    fn plan_mega_kernel_quant_gemm_produces_pack_map_invariant() {
        use crate::compiler::graph::{CompilerGraph, OpKind, SymDim};
        use crate::compiler::fusion::{FusionGroup, FusionMode};
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::dispatch::device_profile::DeviceProfile;
        use crate::types::DType;
        use crate::quant::QuantType;

        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());

        let mut graph = CompilerGraph::new();
        let hidden = 512;

        let input_t = graph.add_tensor_concrete("input", &[1, hidden], DType::F32);
        let q_weight = graph.add_tensor_concrete("q_weight", &[hidden, hidden], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[1, hidden], DType::F32);

        let qgemm = graph.add_op(
            OpKind::QuantGemm {
                m: SymDim::Concrete(1),
                n: hidden,
                k: hidden,
                quant_type: QuantType::Q4K,
            },
            vec![input_t, q_weight],
            vec![q_out],
            "qgemm",
        );

        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: qgemm,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![qgemm],
                multi_output: Default::default(),
                dominant_dtype: None,
            }],
            op_to_group: [(qgemm, 0)].into_iter().collect(),
        };

        let alloc = BufferAllocation::default();

        // Act
        let rp = plan_mega_kernel_resources(
            &graph, &plan, &profile, &alloc,
            hidden, 2048, 4096,
        );

        // Assert: QuantGemm should produce PackMapBase invariant (and always hidden_dim ModelConfig)
        let has_pack_map = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::PackMapBase));
        assert!(has_pack_map, "QuantGemm graph should produce PackMapBase invariant");

        let has_hidden = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::ModelConfig { .. }));
        assert!(has_hidden);

        // No RoPE in graph, so no RopeTablePtr
        let has_rope = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::RopeTablePtr));
        assert!(!has_rope, "graph without RoPE should not produce RopeTablePtr");
    }

    // ── 35. group_needs_spill returns true for oversubscribed group ──

    #[test]
    fn group_needs_spill_returns_true_for_real_oversubscribed_group() {
        // Arrange: single group with 12 ops + GEMM → vec = max(24,4)+8=32, avail=16-4=12
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![12],
            gemm_groups: vec![true],
            total_vec_regs: 16,
            total_gpr_regs: 8,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Act & Assert
        assert!(plan.group_needs_spill(0), "oversubscribed group should need spill");
    }

    // ── 36. plan_stack_blueprint — callee save slots positioned correctly ─

    #[test]
    fn plan_stack_callee_saves_follow_abi_args() {
        // Arrange: 3 ABI args, 3 callee saves
        // ABI args at -8, -16, -24; callee_save_base = -(4)*8 = -32
        // callee saves at -32, -40, -48

        // Act
        let bp = plan_stack_blueprint(&[], 3, 3, false);

        // Assert
        assert_eq!(bp.callee_save_slots.len(), 3);
        assert_eq!(bp.callee_save_slots[0], (0u8, -32));
        assert_eq!(bp.callee_save_slots[1], (1u8, -40));
        assert_eq!(bp.callee_save_slots[2], (2u8, -48));

        // spill_base follows callee saves
        assert_eq!(bp.spill_base_rbp_off, -32 - 3 * 8);
        assert_eq!(bp.spill_base_rbp_off, -56);
    }

    // ── 37. build_resource_plan — memory_map reuse saved bytes ───────

    #[test]
    fn build_resource_plan_memory_map_reuse_computed_in_summary() {
        // Arrange: BufferAllocation with empty slots; we verify bytes_saved_by_reuse
        // is the sum of memory_map region sizes
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert: with empty memory_map, bytes_saved_by_reuse = 0
        assert_eq!(plan.summary.bytes_saved_by_reuse, 0);
        assert_eq!(plan.summary.num_buffer_reuse_slots, 0);
    }

    // ── 38. estimate_register_pressure — groups exceed slice length ──

    #[test]
    fn estimate_pressure_groups_exceed_ops_slice_get_defaults() {
        // Arrange: 3 groups declared but only 1 entry in ops_per_group and gemm_groups
        let result = estimate_register_pressure(
            3,
            &[5],
            &[true],
            32,
            16,
        );

        // Assert: group 0 gets ops=5 + GEMM; groups 1,2 get ops=0 (default), no GEMM
        assert_eq!(result.len(), 3);
        // group 0: 5 ops → max(10,4) = 10 vec + 8 GEMM = 18
        assert_eq!(result[0].peak_vec_regs, 18);
        assert!(result[0].suggested_blocking.is_some());

        // group 1: 0 ops → max(0,4) = 4 vec, no GEMM
        assert_eq!(result[1].peak_vec_regs, 4);
        assert!(result[1].suggested_blocking.is_none());

        // group 2: same as group 1
        assert_eq!(result[2].peak_vec_regs, 4);
        assert!(!result[2].needs_spill_fence);
    }

    // ── 39. derive_loop_invariants_from_graph — graph with Gather produces PackMapBase ─

    #[test]
    fn derive_invariants_gather_produces_pack_map_base() {
        use crate::compiler::graph::{CompilerGraph, OpKind, SymDim};
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::fusion::{FusionGroup, FusionMode};
        use crate::dispatch::device_profile::DeviceProfile;
        use crate::types::DType;

        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());

        let mut graph = CompilerGraph::new();
        let hidden = 256;

        let ids = graph.add_tensor_concrete("ids", &[1], DType::F32);
        let embed_w = graph.add_tensor_concrete("embed_w", &[32000, hidden], DType::F32);
        let embed_out = graph.add_tensor_concrete("embed_out", &[1, hidden], DType::F32);

        let gather = graph.add_op(
            OpKind::Gather {
                table_rows: 32000,
                embed_dim: hidden,
                index_dim: SymDim::Concrete(1),
                indices_kind: crate::compiler::graph::GatherIndicesKind::default(),
            },
            vec![ids, embed_w],
            vec![embed_out],
            "embed",
        );

        let fusion_plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gather,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![gather],
                multi_output: Default::default(),
                dominant_dtype: None,
            }],
            op_to_group: [(gather, 0)].into_iter().collect(),
        };

        // Act
        let rp = plan_mega_kernel_resources(
            &graph, &fusion_plan, &profile,
            &BufferAllocation::default(),
            hidden, 1024, 2048,
        );

        // Assert: Gather produces PackMapBase; no RoPE → no RopeTablePtr
        let has_pack = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::PackMapBase));
        assert!(has_pack, "Gather graph should produce PackMapBase invariant");

        let has_rope = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::RopeTablePtr));
        assert!(!has_rope, "Gather-only graph should not produce RopeTablePtr");

        let has_hidden = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::ModelConfig { .. }));
        assert!(has_hidden);
    }

    // ── 40. plan_stack_blueprint — num_abi_args > 6 clamped to 6 ──────────

    #[test]
    fn plan_stack_clamps_abi_args_to_six() {
        // Arrange: request 10 ABI args, but only 6 slots exist

        // Act
        let bp = plan_stack_blueprint(&[], 10, 2, false);

        // Assert: all 6 slots filled, callee saves follow 6 ABI slots (not 10)
        assert_eq!(bp.abi_arg_slots[0], Some(-8));
        assert_eq!(bp.abi_arg_slots[5], Some(-48));
        assert_eq!(bp.callee_save_slots.len(), 2);
        let callee_save_base = -((6 + 1) * 8);
        assert_eq!(bp.callee_save_slots[0].1, callee_save_base);
    }

    // ── 41. estimate_register_pressure — all groups are GEMM ──────────

    #[test]
    fn estimate_pressure_all_gemm_groups_get_blocking() {
        // Arrange: 4 groups, all GEMM
        let result = estimate_register_pressure(
            4,
            &[2, 3, 4, 5],
            &[true, true, true, true],
            32,
            16,
        );

        // Assert: every group has suggested_blocking
        assert_eq!(result.len(), 4);
        for gp in &result {
            assert!(gp.suggested_blocking.is_some(), "GEMM group {} should have blocking", gp.group_id);
            assert!(gp.peak_vec_regs >= 12, "GEMM group adds +8 vec regs on top of base");
        }
    }

    // ── 42. build_resource_plan with debug probe — stack includes probe region ─

    #[test]
    fn build_plan_with_debug_probe_includes_probe_in_stack() {
        // Arrange
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![2],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: true,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert
        assert!(plan.stack.debug_probe_region.is_some());
        let probe = plan.stack.debug_probe_region.unwrap();
        assert_eq!(probe.size_bytes, 4096);
        assert!(plan.stack.total_frame_bytes >= 4096, "frame must include debug probe bytes");
    }

    // ── 43. plan_mega_kernel_resources — graph with RoPE + Gather produces all three invariant types ─

    #[test]
    fn plan_mega_kernel_rope_and_gather_produces_all_invariant_types() {
        use crate::compiler::graph::{CompilerGraph, OpKind, SymDim};
        use crate::compiler::fusion::{FusionGroup, FusionMode};
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::dispatch::device_profile::DeviceProfile;
        use crate::types::DType;

        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());

        let mut graph = CompilerGraph::new();
        let hidden = 256;

        let ids = graph.add_tensor_concrete("ids", &[1], DType::F32);
        let embed_w = graph.add_tensor_concrete("embed_w", &[32000, hidden], DType::F32);
        let embed_out = graph.add_tensor_concrete("embed_out", &[1, hidden], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[1, hidden], DType::F32);
        let rope_out = graph.add_tensor_concrete("rope_out", &[1, hidden], DType::F32);

        let gather = graph.add_op(
            OpKind::Gather {
                table_rows: 32000,
                embed_dim: hidden,
                index_dim: SymDim::Concrete(1),
                indices_kind: crate::compiler::graph::GatherIndicesKind::default(),
            },
            vec![ids, embed_w],
            vec![embed_out],
            "embed",
        );
        let rope = graph.add_op(
            OpKind::RoPE { num_heads: 4, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![q_out],
            vec![rope_out],
            "rope",
        );

        let fusion_plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gather,
                epilogue: vec![rope],
                mode: FusionMode::Standalone,
                ops: vec![gather, rope],
                multi_output: Default::default(),
                dominant_dtype: None,
            }],
            op_to_group: [(gather, 0), (rope, 0)].into_iter().collect(),
        };

        // Act
        let rp = plan_mega_kernel_resources(
            &graph, &fusion_plan, &profile,
            &BufferAllocation::default(),
            hidden, 1024, 2048,
        );

        // Assert: all three invariant types present
        let has_rope = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::RopeTablePtr));
        let has_pack = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::PackMapBase));
        let has_hidden = rp.loop_invariants.iter()
            .any(|inv| matches!(inv.kind, InvariantKind::ModelConfig { .. }));

        assert!(has_rope, "graph with RoPE should produce RopeTablePtr");
        assert!(has_pack, "graph with Gather should produce PackMapBase");
        assert!(has_hidden, "graph always produces ModelConfig for hidden_dim");
        assert!(rp.loop_invariants.len() >= 3, "should have at least 3 invariants");
    }

    // ── 44. partition_concurrency — batch=0 delegates to single_sequence ──

    #[test]
    fn partition_concurrency_batch_zero_is_single_sequence() {
        // Arrange
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 2048,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 128,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -32,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        // Act: batch_size=0 should follow batch<=1 path
        let partition = partition_concurrency(&layout, &stack, 0, 512, 1024);

        // Assert: same as single_sequence (kv_cache_region is empty)
        assert_eq!(partition.per_sequence.scratchpad_bytes_per_seq, 2048);
        assert_eq!(partition.per_sequence.kv_cache_region.size_bytes, 0);
        assert!(matches!(partition.shared.weight_ptr, InvariantLocation::Gpr(0)));
    }

    // ── 45. plan_stack_blueprint — spill slots are laid out sequentially downward ──

    #[test]
    fn plan_stack_spill_slots_are_sequential_downward() {
        // Arrange: pressure that forces both vec and gpr spills
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 30,
            peak_gpr_regs: 14,
            available_vec_regs: 16,
            available_gpr_regs: 8,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 2, 2, false);

        // Assert: 14 vec spills + 6 gpr spills
        let vec_slots: Vec<_> = bp.spill_slots.iter()
            .filter(|s| s.size_bytes == 64)
            .collect();
        let gpr_slots: Vec<_> = bp.spill_slots.iter()
            .filter(|s| s.size_bytes == 8)
            .collect();
        assert_eq!(vec_slots.len(), 14);
        assert_eq!(gpr_slots.len(), 6);

        // All spill slots must be below (more negative than) spill_base_rbp_off
        for slot in &bp.spill_slots {
            assert!(slot.rbp_offset < bp.spill_base_rbp_off,
                "spill slot at {} must be below spill base {}",
                slot.rbp_offset, bp.spill_base_rbp_off);
        }

        // Vec spill slots are strictly decreasing in rbp_offset
        for i in 1..vec_slots.len() {
            assert!(vec_slots[i].rbp_offset < vec_slots[i - 1].rbp_offset,
                "vec spill slots should be decreasing");
        }

        // GPR spill slots are strictly decreasing in rbp_offset
        for i in 1..gpr_slots.len() {
            assert!(gpr_slots[i].rbp_offset < gpr_slots[i - 1].rbp_offset,
                "gpr spill slots should be decreasing");
        }
    }

    // ── 46. total_scratchpad_with_pingpong — batch>1 returns total_bytes only ──

    #[test]
    fn total_scratchpad_batch_gt_1_no_double_counting() {
        // Arrange: batch>1 disables ping_pong
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 2,
            activation_bytes: 4096,
            kv_bytes: 8192,
        });

        // Assert: no ping_pong, so total_scratchpad_with_pingpong == buffers.total_bytes
        assert!(plan.buffers.ping_pong.is_none());
        assert_eq!(plan.total_scratchpad_with_pingpong(), plan.buffers.total_bytes);
    }

    // ── 47. estimate_register_pressure — single group, zero ops, no GEMM ──

    #[test]
    fn estimate_pressure_zero_ops_no_gemm_uses_minimum_defaults() {
        // Arrange: 1 group, 0 ops, no GEMM
        let result = estimate_register_pressure(1, &[0], &[false], 32, 16);

        // Assert: 0 ops -> max(0,4) = 4 vec, max(0,6) = 6 gpr (minimum defaults)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].peak_vec_regs, 4);
        assert_eq!(result[0].peak_gpr_regs, 6);
        assert!(result[0].suggested_blocking.is_none());
        assert!(!result[0].needs_spill_fence, "minimum defaults should not trigger spill");
    }

    // ── 48. build_resource_plan — loop_invariant_by_index returns correct kinds ──

    #[test]
    fn build_plan_loop_invariant_by_index_preserves_order() {
        // Arrange: 4 invariants in a specific order
        let invariants = vec![
            LoopInvariant {
                kind: InvariantKind::RopeTablePtr,
                location: InvariantLocation::Stack(-256),
                computation: InvariantComputation::LoadAbiArg(0),
            },
            LoopInvariant {
                kind: InvariantKind::ModelConfig { name: "num_heads".into(), value: 32 },
                location: InvariantLocation::Stack(-264),
                computation: InvariantComputation::LoadImm(32),
            },
            LoopInvariant {
                kind: InvariantKind::NormGammaPtr { layer_stride: 4096 },
                location: InvariantLocation::Gpr(10),
                computation: InvariantComputation::PtrArithmetic { base: 3, stride: 4096 },
            },
            LoopInvariant {
                kind: InvariantKind::PackMapBase,
                location: InvariantLocation::Gpr(5),
                computation: InvariantComputation::LoadAbiArg(2),
            },
        ];

        // Act
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: invariants,
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert: order preserved
        assert!(matches!(plan.loop_invariant_by_index(0).unwrap().kind, InvariantKind::RopeTablePtr));
        let inv1_kind = &plan.loop_invariant_by_index(1).unwrap().kind;
        if let InvariantKind::ModelConfig { name, value } = inv1_kind {
            assert_eq!(name, "num_heads");
            assert_eq!(*value, 32);
        } else {
            panic!("expected ModelConfig invariant at index 1");
        }
        assert!(matches!(
            plan.loop_invariant_by_index(2).unwrap().kind,
            InvariantKind::NormGammaPtr { layer_stride: 4096 }
        ));
        assert!(matches!(plan.loop_invariant_by_index(3).unwrap().kind, InvariantKind::PackMapBase));
        assert!(plan.loop_invariant_by_index(4).is_none());
    }

    // ── 49. plan_stack_blueprint — frame alignment always 16-byte aligned ──

    #[test]
    fn plan_stack_frame_alignment_with_varied_configs() {
        // Arrange: several configurations that produce odd total sizes

        // Config 1: 5 ABI args, 3 callee saves, debug probe, some pressure
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 20,
            peak_gpr_regs: 6,
            available_vec_regs: 28,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: false,
        }];
        let bp1 = plan_stack_blueprint(&pressure, 5, 3, true);

        // Config 2: 1 ABI arg, 1 callee save, no debug, no pressure
        let bp2 = plan_stack_blueprint(&[], 1, 1, false);

        // Assert: both configs produce 16-byte aligned frames
        assert_eq!(bp1.total_frame_bytes % 16, 0, "config 1 frame must be 16-byte aligned");
        assert_eq!(bp2.total_frame_bytes % 16, 0, "config 2 frame must be 16-byte aligned");
    }

    // ── 50. estimate_register_pressure — GEMM group peak includes extra GPR ──

    #[test]
    fn estimate_pressure_gemm_group_adds_extra_gpr() {
        // Arrange: compare GEMM vs non-GEMM with same ops
        let result = estimate_register_pressure(
            2,
            &[4, 4],
            &[false, true],
            32,
            16,
        );

        // Assert: GEMM group needs +4 extra GPR beyond the non-GEMM baseline
        let non_gemm_gpr = result[0].peak_gpr_regs;
        let gemm_gpr = result[1].peak_gpr_regs;
        assert_eq!(gemm_gpr, non_gemm_gpr + 4, "GEMM group should need 4 extra GPR");
    }

    // ── 51. estimate_register_pressure — available regs subtract callee reserve ──

    #[test]
    fn estimate_pressure_available_regs_subtracts_reserve() {
        // Arrange: total_vec=32, total_gpr=16 → available = total - 4
        let result = estimate_register_pressure(1, &[2], &[false], 32, 16);

        // Assert
        assert_eq!(result[0].available_vec_regs, 28, "available vec = 32 - 4 callee reserve");
        assert_eq!(result[0].available_gpr_regs, 12, "available gpr = 16 - 4 callee reserve");
    }

    // ── 52. plan_stack_blueprint — no pressure means zero spill slots ──

    #[test]
    fn plan_stack_zero_pressure_produces_no_spill_slots() {
        // Arrange: pressure with peak ≤ available (no oversubscription)
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 10,
            peak_gpr_regs: 4,
            available_vec_regs: 28,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: false,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 2, 2, false);

        // Assert: peak - available = negative → saturating_sub → 0 spill
        assert!(bp.spill_slots.is_empty(), "no oversubscription should produce zero spill slots");
    }

    // ── 53. build_resource_plan — summary peak regs from multiple groups ──

    #[test]
    fn build_plan_summary_peak_regs_is_max_across_groups() {
        // Arrange: 3 groups with varying GEMM flags
        let result = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 3,
            ops_per_group: vec![2, 6, 3],
            gemm_groups: vec![false, true, false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert: summary peak is the max across all groups
        let max_vec = result.pressure.iter().map(|g| g.peak_vec_regs).max().unwrap();
        let max_gpr = result.pressure.iter().map(|g| g.peak_gpr_regs).max().unwrap();
        assert_eq!(result.summary.peak_vec_regs, max_vec);
        assert_eq!(result.summary.peak_gpr_regs, max_gpr);
    }

    // ── 54. partition_concurrency — batch>1 with ping_pong uses buffer_bytes*2 ──

    #[test]
    fn partition_concurrency_batch_gt_1_with_ping_pong_uses_doubled_bytes() {
        // Arrange: layout with ping_pong, batch=3
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: Some(PingPongLayout {
                ping_offset: 0,
                pong_offset: 4096,
                buffer_bytes: 4096,
            }),
            total_bytes: 8192,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 128,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -32,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        // Act
        let partition = partition_concurrency(&layout, &stack, 3, 4096, 8192);

        // Assert: ping_pong buffer_bytes * 2 = 8192 per seq
        assert_eq!(partition.per_sequence.scratchpad_bytes_per_seq, 8192);
    }

    // ── 55. plan_stack_blueprint — mxcsr_rsp_off defaults to zero ──

    #[test]
    fn plan_stack_mxcsr_offset_defaults_to_zero() {
        // Arrange & Act
        let bp = plan_stack_blueprint(&[], 6, 4, false);

        // Assert: mxcsr_rsp_off is initialized to 0 (filled later in prologue)
        assert_eq!(bp.mxcsr_rsp_off, 0);
    }

    // ── 56. build_resource_plan — num_layers equals num_groups ──

    #[test]
    fn build_plan_num_layers_equals_input_groups() {
        // Arrange: 7 groups
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 7,
            ops_per_group: vec![1, 2, 3, 1, 4, 2, 1],
            gemm_groups: vec![false; 7],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert
        assert_eq!(plan.summary.num_layers, 7);
        assert_eq!(plan.pressure.len(), 7);
    }

    // ── 57. plan_stack_blueprint — debug probe offset is below all spill slots ──

    #[test]
    fn plan_stack_debug_probe_below_all_spill_slots() {
        // Arrange: pressure that causes both vec and gpr spills
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 30,
            peak_gpr_regs: 14,
            available_vec_regs: 16,
            available_gpr_regs: 8,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 4, 4, true);

        // Assert: debug probe must exist and be below every spill slot
        let probe = bp.debug_probe_region.expect("should have debug probe");
        for slot in &bp.spill_slots {
            assert!(probe.rbp_offset < slot.rbp_offset,
                "probe at {} must be below spill slot at {}",
                probe.rbp_offset, slot.rbp_offset);
        }
    }

    // ── 58. build_resource_plan — batch=1 total_scratchpad includes activation ──

    #[test]
    fn build_plan_batch_1_total_scratchpad_includes_activation() {
        // Arrange
        let activation_bytes = 4096usize;
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes,
            kv_bytes: 8192,
        });

        // Assert: total_bytes = buffer_alloc (0) + ping_pong (activation_bytes)
        assert!(plan.buffers.ping_pong.is_some());
        assert_eq!(plan.buffers.ping_pong.unwrap().buffer_bytes, activation_bytes);
        assert!(plan.buffers.total_bytes >= activation_bytes);
    }

    // ── 59. estimate_register_pressure — large ops count scales linearly ──

    #[test]
    fn estimate_pressure_large_ops_count_scales_linearly() {
        // Arrange: single group with 50 ops, no GEMM
        let result = estimate_register_pressure(1, &[50], &[false], 32, 16);

        // Assert: 50 ops → 50*2=100 vec regs, 50*3=150 gpr regs (no minimum clamp needed)
        assert_eq!(result[0].peak_vec_regs, 100, "50 ops * 2 vec per op = 100");
        assert_eq!(result[0].peak_gpr_regs, 150, "50 ops * 3 gpr per op = 150");
        assert!(result[0].needs_spill_fence, "100 vec >> 28 available, needs spill");
    }

    // ── 60. build_resource_plan — empty graph (zero groups, zero ops) ──

    #[test]
    fn build_plan_empty_graph_zero_groups() {
        // Arrange: completely empty graph
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 0,
            ops_per_group: vec![],
            gemm_groups: vec![],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 0,
            num_callee_saves: 0,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 0,
            kv_bytes: 0,
        });

        // Assert
        assert_eq!(plan.summary.num_layers, 0);
        assert!(plan.pressure.is_empty());
        assert_eq!(plan.summary.peak_vec_regs, 0);
        assert_eq!(plan.summary.peak_gpr_regs, 0);
        assert!(plan.gemm_blocking_for_group(0).is_none());
        assert!(!plan.group_needs_spill(0));
    }

    // ── 61. single group pressure — non-GEMM minimum defaults ──

    #[test]
    fn estimate_pressure_single_group_non_gemm_minimum() {
        // Arrange: single group, 1 op, no GEMM
        let result = estimate_register_pressure(1, &[1], &[false], 16, 8);

        // Assert: 1 op -> max(2,4)=4 vec, max(3,6)=6 gpr; available = 16-4=12, 8-4=4
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].peak_vec_regs, 4);
        assert_eq!(result[0].peak_gpr_regs, 6);
        assert_eq!(result[0].available_vec_regs, 12);
        assert_eq!(result[0].available_gpr_regs, 4);
        assert!(result[0].suggested_blocking.is_none());
        // 4 vec <= 12 available and 6 gpr > 4 available => needs spill
        assert!(result[0].needs_spill_fence, "6 gpr > 4 available gpr => spill fence");
    }

    // ── 62. BufferLayout coloring — memory_map with multiple tenants ──

    #[test]
    fn buffer_layout_memory_map_multiple_tenants_share_region() {
        // Arrange: a MemoryRegion with 3 non-overlapping tenants sharing same offset
        let region = MemoryRegion {
            offset: 0,
            size_bytes: 4096,
            tenants: vec![TensorId(0), TensorId(1), TensorId(2)],
        };
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![region],
            ping_pong: None,
            total_bytes: 4096,
        };

        // Assert: region has 3 tenants, offset_for still checks base slots
        assert_eq!(layout.memory_map.len(), 1);
        assert_eq!(layout.memory_map[0].tenants.len(), 3);
        assert_eq!(layout.activation_bytes(), 0, "no ping_pong => 0 activation bytes");
    }

    // ── 63. concurrency partition boundary — batch=1 vs batch=2 offset divergence ──

    #[test]
    fn concurrency_partition_batch_1_vs_2_offset_divergence() {
        // Arrange: same layout/stack, compare batch=1 and batch=2
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 8192,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 128,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -32,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        // Act
        let p1 = partition_concurrency(&layout, &stack, 1, 4096, 8192);
        let p2 = partition_concurrency(&layout, &stack, 2, 4096, 8192);

        // Assert: batch=1 has empty kv_cache_region; batch=2 has populated offsets
        assert_eq!(p1.per_sequence.kv_cache_region.size_bytes, 0);
        assert_eq!(p2.per_sequence.kv_cache_region.size_bytes, 8192);
        assert_eq!(p2.per_sequence.seq_offset_map.kv_cache_offset, 4096);
        assert_eq!(p2.per_sequence.seq_offset_map.temp_buffer_offset, 4096 + 8192);
    }

    // ── 64. estimate_register_pressure — total regs saturating at zero ──

    #[test]
    fn estimate_pressure_total_regs_zero_means_available_zero() {
        // Arrange: total_vec_regs=0, total_gpr_regs=0 (extreme edge)
        let result = estimate_register_pressure(1, &[2], &[false], 0, 0);

        // Assert: available = 0-4 = saturating_sub => 0; peak > available => spill
        assert_eq!(result[0].available_vec_regs, 0);
        assert_eq!(result[0].available_gpr_regs, 0);
        assert!(result[0].needs_spill_fence, "any demand > 0 available => spill");
        assert_eq!(result[0].peak_vec_regs, 4);
        assert_eq!(result[0].peak_gpr_regs, 6);
    }

    // ── 65. plan_stack_blueprint — only GPR spills, no vec spills ──

    #[test]
    fn plan_stack_only_gpr_spills_no_vec_spills() {
        // Arrange: group with vec undersubscribed but gpr oversubscribed
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 8,
            peak_gpr_regs: 20,
            available_vec_regs: 28,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 2, 2, false);

        // Assert: no vec spills (8 <= 28), 8 gpr spills (20-12=8)
        let vec_spills = bp.spill_slots.iter().filter(|s| s.size_bytes == 64).count();
        let gpr_spills = bp.spill_slots.iter().filter(|s| s.size_bytes == 8).count();
        assert_eq!(vec_spills, 0, "vec not oversubscribed => zero vec spills");
        assert_eq!(gpr_spills, 8, "20-12=8 gpr oversubscription => 8 gpr spills");
    }

    // ── 66. build_resource_plan — zero activation and kv bytes ──

    #[test]
    fn build_plan_zero_activation_and_kv_bytes() {
        // Arrange: activation_bytes=0, kv_bytes=0 (minimal model)
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 0,
            kv_bytes: 0,
        });

        // Assert: ping_pong buffer_bytes=0, total_bytes = buffer_alloc(0) + 0
        assert!(plan.buffers.ping_pong.is_some());
        assert_eq!(plan.buffers.ping_pong.unwrap().buffer_bytes, 0);
        assert_eq!(plan.buffers.ping_pong.unwrap().ping_offset, 0);
        assert_eq!(plan.buffers.ping_pong.unwrap().pong_offset, 0);
        assert_eq!(plan.buffers.total_bytes, 0);
    }

    // ── 67. partition_concurrency — batch>1 kv_offset equals activation_bytes ──

    #[test]
    fn partition_concurrency_batch_gt1_kv_offset_equals_activation() {
        // Arrange: batch=3 with specific activation/kv sizes
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 16384,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 256,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -48,
            spill_slots: vec![],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        // Act: activation_bytes=8192, kv_bytes=16384
        let partition = partition_concurrency(&layout, &stack, 3, 8192, 16384);

        // Assert: kv_cache offset = activation_bytes, temp = activation + kv
        assert_eq!(partition.per_sequence.kv_cache_region.offset, 8192);
        assert_eq!(partition.per_sequence.kv_cache_region.size_bytes, 16384);
        assert_eq!(partition.per_sequence.seq_offset_map.kv_cache_offset, 8192);
        assert_eq!(partition.per_sequence.seq_offset_map.temp_buffer_offset, 8192 + 16384);
    }

    // ── 68. plan_stack_blueprint — only vec spills, no gpr spills ──

    #[test]
    fn plan_stack_only_vec_spills_no_gpr_spills() {
        // Arrange: group with vec oversubscribed but gpr undersubscribed
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 36,
            peak_gpr_regs: 6,
            available_vec_regs: 16,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 4, 3, false);

        // Assert: 20 vec spills (36-16=20), 0 gpr spills (6-12=0 via saturating_sub)
        let vec_spills = bp.spill_slots.iter().filter(|s| s.size_bytes == 64).count();
        let gpr_spills = bp.spill_slots.iter().filter(|s| s.size_bytes == 8).count();
        assert_eq!(vec_spills, 20, "36-16=20 vec oversubscription => 20 ZMM spills");
        assert_eq!(gpr_spills, 0, "6 <= 12 => zero gpr spills");
    }

    // ── 69. NormGammaPtr invariant — different stride values ──

    #[test]
    fn norm_gamma_ptr_invariant_different_stride_values() {
        // Arrange: two NormGammaPtr invariants with different strides
        let inv1 = LoopInvariant {
            kind: InvariantKind::NormGammaPtr { layer_stride: 256 },
            location: InvariantLocation::Gpr(10),
            computation: InvariantComputation::PtrArithmetic { base: 2, stride: 256 },
        };
        let inv2 = LoopInvariant {
            kind: InvariantKind::NormGammaPtr { layer_stride: 4096 },
            location: InvariantLocation::Gpr(11),
            computation: InvariantComputation::PtrArithmetic { base: 2, stride: 4096 },
        };

        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![inv1, inv2],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert: both invariants preserved, stride values distinct
        assert_eq!(plan.summary.num_loop_invariants, 2);
        let inv0 = plan.loop_invariant_by_index(0).unwrap();
        let inv1_ret = plan.loop_invariant_by_index(1).unwrap();
        assert!(matches!(inv0.kind, InvariantKind::NormGammaPtr { layer_stride: 256 }));
        assert!(matches!(inv1_ret.kind, InvariantKind::NormGammaPtr { layer_stride: 4096 }));
        assert_ne!(
            inv0.computation,
            inv1_ret.computation,
            "different strides must produce different computations"
        );
    }

    // ── 70. BufferLayout offset_for with multiple slots returns correct mapping ──

    #[test]
    fn buffer_layout_offset_for_multiple_slots_returns_correct_mapping() {
        // Arrange: 3 tensors at different offsets
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);
        let layout = BufferLayout {
            base: BufferAllocation {
                slots: vec![
                    BufferSlot { tensor_id: t0, offset: 0, size_bytes: 1024 },
                    BufferSlot { tensor_id: t1, offset: 1024, size_bytes: 2048 },
                    BufferSlot { tensor_id: t2, offset: 3072, size_bytes: 512 },
                ],
                ..BufferAllocation::default()
            },
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 3584,
        };

        // Act & Assert
        assert_eq!(layout.offset_for(t0), Some(0));
        assert_eq!(layout.offset_for(t1), Some(1024));
        assert_eq!(layout.offset_for(t2), Some(3072));
        assert!(layout.offset_for(TensorId(99)).is_none(), "unknown tensor returns None");
    }

    // ── 71. PingPongLayout contiguous memory region calculation ──

    #[test]
    fn ping_pong_layout_total_region_is_double_buffer_bytes() {
        // Arrange: ping at 0, pong at 16384, each buffer 16384 bytes
        let pp = PingPongLayout {
            ping_offset: 0,
            pong_offset: 16384,
            buffer_bytes: 16384,
        };

        // Assert: the total contiguous region = pong_offset + buffer_bytes
        let total_region = pp.pong_offset + pp.buffer_bytes;
        assert_eq!(total_region, 32768);
        assert_eq!(total_region, pp.buffer_bytes * 2, "two equal buffers");
        assert_eq!(pp.pong_offset - pp.ping_offset, pp.buffer_bytes, "buffers are contiguous");
    }

    // ── 72. MemoryRegion non-overlapping offset ranges ──

    #[test]
    fn memory_region_end_offset_equals_offset_plus_size() {
        // Arrange: multiple regions with distinct ranges
        let r0 = MemoryRegion {
            offset: 0,
            size_bytes: 4096,
            tenants: vec![TensorId(0)],
        };
        let r1 = MemoryRegion {
            offset: 4096,
            size_bytes: 8192,
            tenants: vec![TensorId(1), TensorId(2)],
        };
        let r2 = MemoryRegion {
            offset: 12288,
            size_bytes: 2048,
            tenants: vec![TensorId(3)],
        };

        // Assert: each region's end = offset + size_bytes; no overlap
        assert_eq!(r0.offset + r0.size_bytes, 4096);
        assert_eq!(r1.offset + r1.size_bytes, 12288);
        assert_eq!(r2.offset + r2.size_bytes, 14336);
        assert_eq!(r1.offset, r0.offset + r0.size_bytes, "r1 starts where r0 ends");
        assert_eq!(r2.offset, r1.offset + r1.size_bytes, "r2 starts where r1 ends");
    }

    // ── 73. GroupPressure spill fence true when vec_oversubscribed but gpr_ok ──

    #[test]
    fn group_pressure_spill_fence_true_vec_only_oversubscribed() {
        // Arrange: vec oversubscribed, gpr fine
        let gp = GroupPressure {
            group_id: 0,
            peak_vec_regs: 30,
            peak_gpr_regs: 6,
            available_vec_regs: 16,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: true,
        };

        // Assert: spill fence triggered by vec alone
        assert!(gp.needs_spill_fence);
        assert!(gp.peak_vec_regs > gp.available_vec_regs, "vec oversubscribed");
        assert!(gp.peak_gpr_regs <= gp.available_gpr_regs, "gpr not oversubscribed");
    }

    // ── 74. StackBlueprint total_frame_bytes accounts for all components ──

    #[test]
    fn stack_blueprint_total_frame_accounts_for_all_components() {
        // Arrange: 6 ABI args + 5 callee saves + vec spills + debug probe
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 24,
            peak_gpr_regs: 6,
            available_vec_regs: 16,
            available_gpr_regs: 12,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 6, 5, true);

        // Assert: frame must include ABI args + callee saves + spill + debug probe
        let abi_bytes = 6 * 8;
        let callee_bytes = 5 * 8;
        let vec_spill_count = pressure[0].peak_vec_regs.saturating_sub(pressure[0].available_vec_regs);
        let vec_spill_bytes = vec_spill_count * 64;
        let probe_bytes = 4096;
        let min_expected = abi_bytes + callee_bytes + vec_spill_bytes + probe_bytes;

        assert!(bp.total_frame_bytes >= min_expected,
            "frame {} must cover at least {} (abi + callee + spill + probe)",
            bp.total_frame_bytes, min_expected);
        assert_eq!(bp.total_frame_bytes % 16, 0, "must be 16-byte aligned");
    }

    // ── 75. ConcurrencyPartition single_sequence preserves spill_slot_count ──

    #[test]
    fn concurrency_single_sequence_preserves_spill_slot_count() {
        // Arrange: stack with 5 spill slots
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![],
            ping_pong: None,
            total_bytes: 2048,
        };
        let stack = StackBlueprint {
            total_frame_bytes: 256,
            abi_arg_slots: [None; 6],
            callee_save_slots: vec![],
            spill_base_rbp_off: -48,
            spill_slots: vec![
                SpillSlot { rbp_offset: -112, size_bytes: 64 },
                SpillSlot { rbp_offset: -176, size_bytes: 64 },
                SpillSlot { rbp_offset: -240, size_bytes: 64 },
                SpillSlot { rbp_offset: -248, size_bytes: 8 },
                SpillSlot { rbp_offset: -256, size_bytes: 8 },
            ],
            mxcsr_rsp_off: 0,
            debug_probe_region: None,
        };

        // Act
        let partition = ConcurrencyPartition::single_sequence(&layout, &stack);

        // Assert: spill_slots_per_seq matches stack.spill_slots.len()
        assert_eq!(partition.per_sequence.spill_slots_per_seq, 5);
        assert_eq!(partition.per_sequence.spill_slots_per_seq, stack.spill_slots.len());
    }

    // ── 76. GraphResourcePlan gemm_blocking_for_group returns correct blocking ──

    #[test]
    fn resource_plan_gemm_blocking_for_group_returns_correct_values() {
        // Arrange: 3 groups, only group 1 is GEMM
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 3,
            ops_per_group: vec![2, 4, 2],
            gemm_groups: vec![false, true, false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Act & Assert
        assert!(plan.gemm_blocking_for_group(0).is_none(), "group 0 is not GEMM");
        let blocking = plan.gemm_blocking_for_group(1).expect("group 1 is GEMM");
        assert!(blocking.mr > 0);
        assert!(blocking.nr > 0);
        assert!(blocking.mc > 0);
        assert!(blocking.nc > 0);
        assert_eq!(blocking.kc, 64, "kc is always 64 in current implementation");
        assert!(plan.gemm_blocking_for_group(2).is_none(), "group 2 is not GEMM");
    }

    // ── 77. estimate_register_pressure — GEMM blocking derived from available regs ──

    #[test]
    fn estimate_pressure_gemm_blocking_constrained_by_available_regs() {
        // Arrange: tight register budget, 12 vec regs total
        let result = estimate_register_pressure(1, &[3], &[true], 12, 8);

        // Assert: available_vec = 12 - 4 = 8; mr = min(8/4, 8).max(1) = 2; nr = min(8/4, 4).max(1) = 2
        let blocking = result[0].suggested_blocking.as_ref().expect("GEMM group needs blocking");
        assert_eq!(blocking.mr, 2, "mr = min(available/4, 8).max(1) = min(2, 8) = 2");
        assert_eq!(blocking.nr, 2, "nr = min(available/4, 4).max(1) = min(2, 4) = 2");
        assert_eq!(blocking.mc, 8, "mc = mr * 4 = 8");
        assert_eq!(blocking.nc, 8, "nc = nr * 4 = 8");
    }

    // ── 78. plan_stack_blueprint — callee save register numbers are sequential ──

    #[test]
    fn plan_stack_callee_save_register_numbers_sequential() {
        // Arrange: 4 callee saves
        // Act
        let bp = plan_stack_blueprint(&[], 2, 4, false);

        // Assert: register numbers are 0, 1, 2, 3
        assert_eq!(bp.callee_save_slots.len(), 4);
        for (i, &(reg, _off)) in bp.callee_save_slots.iter().enumerate() {
            assert_eq!(reg, i as u8, "callee save register number should be sequential");
        }

        // Offsets are 8 bytes apart
        for i in 1..bp.callee_save_slots.len() {
            let diff = bp.callee_save_slots[i - 1].1 - bp.callee_save_slots[i].1;
            assert_eq!(diff, 8, "callee save slots should be 8 bytes apart");
        }
    }

    // ── 79. build_resource_plan — summary total_scratchpad matches buffers.total_bytes ──

    #[test]
    fn build_plan_summary_scratchpad_matches_buffers_total() {
        // Arrange: batch=1 with activation_bytes=8192
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 2,
            ops_per_group: vec![3, 3],
            gemm_groups: vec![true, false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 4,
            num_callee_saves: 4,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 8192,
            kv_bytes: 16384,
        });

        // Assert: summary.total_scratchpad_bytes == buffers.total_bytes
        assert_eq!(plan.summary.total_scratchpad_bytes, plan.buffers.total_bytes);
        // And summary.total_stack_bytes == stack.total_frame_bytes
        assert_eq!(plan.summary.total_stack_bytes, plan.stack.total_frame_bytes);
    }

    // ── 80. estimate_register_pressure — maximum pressure with huge ops count ──

    #[test]
    fn estimate_pressure_maximum_pressure_with_huge_ops_count() {
        // Arrange: single group with 1000 ops, GEMM enabled
        let result = estimate_register_pressure(1, &[1000], &[true], 32, 16);

        // Assert: 1000 ops -> 2000 vec regs + 8 GEMM = 2008; 3000 gpr + 4 GEMM = 3004
        assert_eq!(result[0].peak_vec_regs, 2008, "1000 ops * 2 + 8 GEMM = 2008");
        assert_eq!(result[0].peak_gpr_regs, 3004, "1000 ops * 3 + 4 GEMM = 3004");
        assert!(result[0].needs_spill_fence, "massive oversubscription requires spill");
        assert!(result[0].suggested_blocking.is_some(), "GEMM group has blocking");
    }

    // ── 81. plan_stack_blueprint — zero pressure produces minimal frame ──

    #[test]
    fn plan_stack_zero_pressure_produces_minimal_aligned_frame() {
        // Arrange: no pressure, 2 ABI args, 2 callee saves, no debug probe
        let bp = plan_stack_blueprint(&[], 2, 2, false);

        // Assert: minimal frame = (2 + 2) * 8 = 32 bytes, aligned to 16
        let min_bytes = 2 * 8 + 2 * 8; // 32
        assert!(bp.total_frame_bytes >= min_bytes);
        assert_eq!(bp.total_frame_bytes % 16, 0, "frame must be 16-byte aligned");
        assert!(bp.spill_slots.is_empty(), "no pressure means no spill slots");
        assert!(bp.debug_probe_region.is_none());
    }

    // ── 82. BufferLayout with ping_pong and memory_map combined ──

    #[test]
    fn buffer_layout_ping_pong_and_memory_map_combined() {
        // Arrange: layout with both ping_pong and memory_map
        let region = MemoryRegion {
            offset: 0,
            size_bytes: 2048,
            tenants: vec![TensorId(0), TensorId(1)],
        };
        let layout = BufferLayout {
            base: BufferAllocation::default(),
            memory_map: vec![region],
            ping_pong: Some(PingPongLayout {
                ping_offset: 2048,
                pong_offset: 2048 + 4096,
                buffer_bytes: 4096,
            }),
            total_bytes: 2048 + 8192,
        };

        // Assert: both features coexist
        assert_eq!(layout.memory_map.len(), 1);
        assert!(layout.ping_pong.is_some());
        assert_eq!(layout.activation_bytes(), 4096);
        assert_eq!(layout.total_bytes, 10240);
    }

    // ── 83. GroupPressure available regs never exceeds total minus reserve ──

    #[test]
    fn group_pressure_available_regs_never_exceeds_total_minus_reserve() {
        // Arrange: various total register counts
        let test_cases = [(32, 16), (16, 8), (8, 4), (64, 32)];

        for (total_vec, total_gpr) in test_cases {
            let result = estimate_register_pressure(1, &[2], &[false], total_vec, total_gpr);

            // Assert: available = total - 4 (callee reserve)
            assert_eq!(result[0].available_vec_regs, total_vec.saturating_sub(4));
            assert_eq!(result[0].available_gpr_regs, total_gpr.saturating_sub(4));
        }
    }

    // ── 84. SpillSlot rbp_offset ordering — vec spills before gpr spills ──

    #[test]
    fn spill_slot_vec_spills_come_before_gpr_spills_in_memory() {
        // Arrange: pressure causing both vec and gpr spills
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 20,
            peak_gpr_regs: 16,
            available_vec_regs: 12,
            available_gpr_regs: 8,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];

        // Act
        let bp = plan_stack_blueprint(&pressure, 2, 2, false);

        // Assert: vec spill slots (64-byte) have less negative offsets than gpr slots (8-byte)
        let vec_slots: Vec<_> = bp.spill_slots.iter()
            .filter(|s| s.size_bytes == 64)
            .collect();
        let gpr_slots: Vec<_> = bp.spill_slots.iter()
            .filter(|s| s.size_bytes == 8)
            .collect();

        assert!(!vec_slots.is_empty(), "should have vec spills");
        assert!(!gpr_slots.is_empty(), "should have gpr spills");

        // All vec slots should be closer to rbp (less negative) than all gpr slots
        let max_vec_offset = vec_slots.iter().map(|s| s.rbp_offset).max().unwrap();
        let min_gpr_offset = gpr_slots.iter().map(|s| s.rbp_offset).min().unwrap();
        assert!(max_vec_offset > min_gpr_offset,
            "vec spills (max offset {}) should be above gpr spills (min offset {})",
            max_vec_offset, min_gpr_offset);
    }

    // ── 85. build_resource_plan — empty ops_per_group with non-zero num_groups ──

    #[test]
    fn build_plan_empty_ops_per_group_with_non_zero_groups() {
        // Arrange: 3 groups but ops_per_group is empty (defaults to 0 ops each)
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 3,
            ops_per_group: vec![],
            gemm_groups: vec![],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        // Assert: 3 groups created, each with minimum defaults (4 vec, 6 gpr)
        assert_eq!(plan.pressure.len(), 3);
        for gp in &plan.pressure {
            assert_eq!(gp.peak_vec_regs, 4, "empty ops -> minimum 4 vec regs");
            assert_eq!(gp.peak_gpr_regs, 6, "empty ops -> minimum 6 gpr regs");
            assert!(gp.suggested_blocking.is_none(), "no GEMM -> no blocking");
        }
        assert_eq!(plan.summary.num_layers, 3);
    }

    // ── 86. InvariantLocation GPR and Stack are mutually exclusive ──

    #[test]
    fn invariant_location_gpr_and_stack_are_distinct_variants() {
        // Arrange: create invariants with different locations
        let gpr_inv = LoopInvariant {
            kind: InvariantKind::PackMapBase,
            location: InvariantLocation::Gpr(5),
            computation: InvariantComputation::LoadAbiArg(1),
        };
        let stack_inv = LoopInvariant {
            kind: InvariantKind::RopeTablePtr,
            location: InvariantLocation::Stack(-512),
            computation: InvariantComputation::LoadAbiArg(0),
        };

        // Assert: locations are different variants
        assert!(matches!(gpr_inv.location, InvariantLocation::Gpr(5)));
        assert!(matches!(stack_inv.location, InvariantLocation::Stack(-512)));

        // Build plan and verify both are preserved
        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation::default(),
            num_groups: 1,
            ops_per_group: vec![1],
            gemm_groups: vec![false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 2,
            num_callee_saves: 2,
            has_debug_probe: false,
            loop_invariants: vec![gpr_inv, stack_inv],
            batch_size: 1,
            activation_bytes: 1024,
            kv_bytes: 2048,
        });

        assert_eq!(plan.summary.num_loop_invariants, 2);
    }

    // ── 87. plan_stack_blueprint — frame size increases with each component ──

    #[test]
    fn plan_stack_frame_size_monotonically_increases_with_components() {
        // Arrange: progressively add components and verify frame grows

        // Baseline: no args, no callee, no pressure, no probe
        let bp0 = plan_stack_blueprint(&[], 0, 0, false);

        // Add ABI args
        let bp1 = plan_stack_blueprint(&[], 4, 0, false);
        assert!(bp1.total_frame_bytes >= bp0.total_frame_bytes + 4 * 8);

        // Add callee saves
        let bp2 = plan_stack_blueprint(&[], 4, 3, false);
        assert!(bp2.total_frame_bytes >= bp1.total_frame_bytes + 3 * 8);

        // Add pressure (spills)
        let pressure = vec![GroupPressure {
            group_id: 0,
            peak_vec_regs: 20,
            peak_gpr_regs: 8,
            available_vec_regs: 12,
            available_gpr_regs: 8,
            suggested_blocking: None,
            needs_spill_fence: true,
        }];
        let bp3 = plan_stack_blueprint(&pressure, 4, 3, false);
        assert!(bp3.total_frame_bytes >= bp2.total_frame_bytes + 8 * 64, "8 vec spills * 64 bytes");

        // Add debug probe
        let bp4 = plan_stack_blueprint(&pressure, 4, 3, true);
        assert!(bp4.total_frame_bytes >= bp3.total_frame_bytes + 4096, "4KB debug probe");
    }

    // ── 88. ResourceSummary all fields populated from plan components ──

    #[test]
    fn resource_summary_all_fields_populated_correctly() {
        // Arrange: comprehensive plan with all features
        let invariants = vec![
            LoopInvariant {
                kind: InvariantKind::RopeTablePtr,
                location: InvariantLocation::Stack(-256),
                computation: InvariantComputation::LoadAbiArg(0),
            },
            LoopInvariant {
                kind: InvariantKind::ModelConfig { name: "hidden_dim".into(), value: 1024 },
                location: InvariantLocation::Stack(-264),
                computation: InvariantComputation::LoadImm(1024),
            },
        ];

        let plan = build_resource_plan(ResourcePlanInput {
            buffer_alloc: BufferAllocation {
                slots: vec![BufferSlot {
                    tensor_id: TensorId(0),
                    offset: 0,
                    size_bytes: 4096,
                }],
                total_bytes: 4096,
                ..BufferAllocation::default()
            },
            num_groups: 4,
            ops_per_group: vec![2, 4, 3, 2],
            gemm_groups: vec![false, true, true, false],
            total_vec_regs: 32,
            total_gpr_regs: 16,
            num_abi_args: 6,
            num_callee_saves: 5,
            has_debug_probe: true,
            loop_invariants: invariants,
            batch_size: 1,
            activation_bytes: 8192,
            kv_bytes: 16384,
        });

        // Assert: all summary fields are populated
        assert!(plan.summary.total_scratchpad_bytes > 0);
        assert!(plan.summary.total_stack_bytes > 0);
        assert!(plan.summary.peak_vec_regs > 0);
        assert!(plan.summary.peak_gpr_regs > 0);
        assert_eq!(plan.summary.num_layers, 4);
        assert_eq!(plan.summary.num_loop_invariants, 2);
        assert_eq!(plan.summary.batch_size, 1);
        // bytes_saved_by_reuse is sum of memory_map sizes (empty in this case)
        assert_eq!(plan.summary.bytes_saved_by_reuse, 0);
    }

    // ── 89. estimate_register_pressure — non-GEMM group never has suggested_blocking ──

    #[test]
    fn estimate_pressure_non_gemm_never_has_suggested_blocking() {
        // Arrange: multiple non-GEMM groups with varying ops
        let ops_counts = [1, 5, 10, 20, 50];
        let gemm_flags = [false; 5];

        let result = estimate_register_pressure(
            5,
            &ops_counts,
            &gemm_flags,
            32,
            16,
        );

        // Assert: no non-GEMM group has suggested_blocking
        for (i, gp) in result.iter().enumerate() {
            assert!(gp.suggested_blocking.is_none(),
                "non-GEMM group {} should not have blocking", i);
        }
    }
}
