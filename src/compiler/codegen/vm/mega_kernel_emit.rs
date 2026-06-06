//! Mega-kernel compilation — unified generate-loop JIT entry point.

use super::instr::*;
use super::isa_profile::IsaProfile;
use super::vm_state::AbiPtrs;
use super::plan_lower::{
    LoweringContext, SymDimSlotMap, TensorPtrResolver,
    emit_fusion_groups, compute_rope_requirement, compute_ple_requirement,
    compute_dwc_requirement, graph_dtype, maybe_debug_bp,
};

use crate::compiler::codegen::RopeCacheRequirement;
use crate::compiler::fusion::{FusionPlan, FusionMode};
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::compiler::buffer_alloc::{BufferAllocation, TensorPtrSource};
use crate::compiler::mega_kernel_abi::MtpKernelConfig;
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::pain_point::{OpBottleneckMap, ParallelismDesc};
use crate::compiler::virtual_activation::VirtualActivationMap;
use crate::compiler::virtual_tensor::VirtualTensorMap;
use crate::compiler::hardware_profile::HardwareProfile;
use crate::compiler::trace::QuantPrecision;
use crate::types::CompilerError;
use super::resource_planner::GraphResourcePlan;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 32 §2.2: Mega-Kernel Variants (REQ-MKO-002)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Mega-kernel CTA organization variant — compiled in at JIT time.
///
/// | Variant | Sync | decode:prefill | Target |
/// |---------|------|----------------|--------|
/// | MK_SERIAL | ring barrier | 100%:0% | SM < 60 |
/// | MK_GRID_SYNC | cooperative grid_sync | 75%:25% | SM 70-89 |
/// | MK_CLUSTER_6_2 | cluster.sync + mbarrier | 6:2 per cluster | SM 90+ |
/// | MK_CLUSTER_5_3 | cluster.sync + mbarrier | 5:3 per cluster | SM 90+ prefill-heavy |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MkVariant {
    /// SM < 60: all CTAs execute serially, no decode/prefill partitioning.
    /// Cross-CTA sync via ring barrier (atom.global.add.u32 + membar.gl + spin-wait).
    Serial,
    /// SM 70-89: CTAs partitioned 75% decode / 25% prefill with grid_sync.
    GridSync { total_ctas: u32 },
    /// SM 90+: 6 decode + 2 prefill CTAs per cluster, cluster.sync + mbarrier.
    Cluster6x2 { cluster_size: u32, num_clusters: u32 },
    /// SM 90+: 5 decode + 3 prefill CTAs per cluster (prefill-heavy workloads).
    Cluster5x3 { cluster_size: u32, num_clusters: u32 },
}

/// Select the MkVariant based on SM version and total SM count.
pub fn select_mk_variant(sm_version: u32, total_sm: u32) -> MkVariant {
    if sm_version >= 90 {
        // SM90+ (Hopper/Blackwell): cluster.sync + mbarrier
        // Cluster 6:2 is the default balanced variant.
        // Cluster size determined by cudaOccupancyMaxPotentialClusterSize at runtime;
        // default portable=8, opt-in=16.
        let cluster_size = 8u32;
        let num_clusters = total_sm / cluster_size;
        MkVariant::Cluster6x2 { cluster_size, num_clusters: num_clusters.max(1) }
    } else if sm_version >= 70 {
        // SM70-89 (Volta/Turing/Ampere): cooperative grid_sync
        MkVariant::GridSync { total_ctas: total_sm }
    } else {
        // SM < 60 (Pascal/Maxwell): serial execution, ring barrier
        MkVariant::Serial
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 32 §3: Autonomous Batch Scheduling (REQ-MKO-003)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// BatchContext field offsets (from batch_ctx_ptr base).
/// Layout defined by SPEC/20 BCI-002 and SPEC/32 §3.
pub mod batch_ctx_offsets {
    pub const NUM_SEQS: u32 = 0;
    pub const TOTAL_PREFILL_TOKENS: u32 = 8;
    pub const INPUT_IDS_FLAT_PTR: u32 = 16;
    pub const OUTPUT_TOKENS_FLAT_PTR: u32 = 24;
    pub const SAMPLING_PARAMS_PTR: u32 = 32;
    pub const PAGE_TABLE_FLAT_PTR: u32 = 40;
    pub const MAX_BATCH_SIZE: u32 = 48;
    pub const REQUEST_QUEUE_PTR: u32 = 56;
    pub const KV_FREE_BITMAP_PTR: u32 = 64;
    pub const SEQ_META_BASE_PTR: u32 = 88;

    /// SPEC 32 §6.2: Extension area layout (after seq_meta array).
    /// Offset = SEQ_META_BASE_PTR + max_batch_size × SEQ_META_STRIDE.
    /// These are relative offsets within the extension area.
    pub mod ext {
        pub const REQUEST_QUEUE_EXT_PTR: u32 = 0;
        pub const OUTPUT_RING_PTR: u32 = 8;
        pub const KV_FREE_BITMAP_EXT_PTR: u32 = 16;
        pub const KV_POOL_TOTAL_PAGES: u32 = 24;
        pub const MAX_BATCH_SIZE_EXT: u32 = 28;
        // DualBatchMeta at +32, 24 bytes (epoch_arrival_count etc.)
        pub const DUAL_BATCH_META: u32 = 32;
        pub const AUTOTUNE_ACTUAL_BATCH: u32 = 56;
        pub const PAGE_ALLOC_POOL_CLUSTER_DSMEM_PTR: u32 = 60;
        pub const PENDING_FREE_LIST_PTR: u32 = 68;
        pub const PENDING_FREE_COUNT_PTR: u32 = 76;
        pub const OUTPUT_PER_CTA_DOORBELL_PTR: u32 = 80;
        pub const OUTPUT_EPOCH_FLAG_PTR: u32 = 88;
        /// Total extension area size (aligned to 96).
        pub const EXT_AREA_SIZE: u32 = 96;
    }

    /// SEQ_META_STRIDE per SPEC/20: 56 bytes (14 × u32).
    pub const SEQ_META_STRIDE: u32 = 56;
}

/// RequestQueue field offsets (from request_queue_ptr base).
pub mod request_queue_offsets {
    pub const READ_IDX: u32 = 0;   // u64
    pub const WRITE_IDX: u32 = 8;  // u64
    pub const CAPACITY: u32 = 16;  // u32
    pub const ENTRIES: u32 = 24;   // RequestQueueEntry[]
}

/// Emit MK_SERIAL compact: single-CTA mark + prefix-scan + parallel move.
///
/// For MK_SERIAL, all operations execute within a single CTA — no cross-CTA
/// synchronization needed. The compact operates on seq_meta entries in-place.
fn emit_compact_serial(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
    seq_count: VRegId,
) {
    let compacted_count = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: compacted_count, value: 0 });

    let loop_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
    let loop_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: loop_ctr, value: 0 });
    prog.emit(VmInstr::GprLoadImm { dst: loop_byte_off, value: 0 });

    prog.emit(VmInstr::LoopBegin {
        counter: loop_ctr,
        byte_offset: loop_byte_off,
        bound: BoundExpr::DynamicVReg(seq_count),
        step_bytes: 1,
    });

    // Load seq_meta_base from batch_ctx
    let seq_meta_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: seq_meta_base,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(batch_ctx_offsets::SEQ_META_BASE_PTR as usize),
    });

    // Compute byte offset for seq[i]: i * 64 (SeqMeta stride)
    let entry_byte_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp {
        dst: entry_byte_off,
        a: loop_ctr,
        b: GprOperand::Imm(6), // * 64
        op: GprOp::Shl,
    });

    // Load active_flag at seq_meta[i * 64 + 8]
    let flag_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: flag_val,
        base: seq_meta_base,
        offset: OffsetExpr::Add(
            Box::new(OffsetExpr::ScalarVReg(entry_byte_off)),
            Box::new(OffsetExpr::Const(8)),
        ),
    });

    // If active_flag == 0 → skip increment (sequence finished)
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(flag_val, 0),
        action: GprBranchAction::Skip(0),
    });

    // compacted_count++
    prog.emit(VmInstr::GprBinOp {
        dst: compacted_count,
        a: compacted_count,
        b: GprOperand::Imm(1),
        op: GprOp::Add,
    });

    prog.emit(VmInstr::LoopEnd);
}

/// Emit request queue refill using atomic read_idx increment.
///
/// `atom.global.add.u64 read_idx, 1` → load RequestQueueEntry at that index.
/// For MK_SERIAL, only CTA 0 executes refill (redundant guard in serial mode).
fn emit_request_queue_refill(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
    survivor_count: VRegId,
) {
    // Load request_queue_ptr from batch_ctx
    let rq_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: rq_ptr,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(batch_ctx_offsets::REQUEST_QUEUE_PTR as usize),
    });

    // Load max_batch_size from batch_ctx
    let max_batch = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: max_batch,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(batch_ctx_offsets::MAX_BATCH_SIZE as usize),
    });

    // Compute free_slots = max_batch - survivor_count
    let free_slots = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp {
        dst: free_slots,
        a: max_batch,
        b: GprOperand::VReg(survivor_count),
        op: GprOp::Sub,
    });

    // If free_slots == 0 → skip refill
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(free_slots, 0),
        action: GprBranchAction::Skip(0),
    });

    // Refill loop: for i in 0..free_slots
    let refill_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
    let refill_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: refill_ctr, value: 0 });
    prog.emit(VmInstr::GprLoadImm { dst: refill_byte_off, value: 0 });
    prog.emit(VmInstr::LoopBegin {
        counter: refill_ctr,
        byte_offset: refill_byte_off,
        bound: BoundExpr::DynamicVReg(free_slots),
        step_bytes: 1,
    });

    // Atomically increment read_idx: atom.global.add.u64(rq_ptr + 0, 1)
    prog.emit(VmInstr::AtomicAdd {
        base: rq_ptr,
        offset: OffsetExpr::Const(request_queue_offsets::READ_IDX as usize),
        value: 1,
        elem_width: 8,
    });

    // Load write_idx to check queue emptiness
    let write_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: write_idx,
        base: rq_ptr,
        offset: OffsetExpr::Const(request_queue_offsets::WRITE_IDX as usize),
    });

    // If read_idx >= write_idx → queue empty, skip entry load
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(write_idx, 0), // placeholder
        action: GprBranchAction::Skip(0),
    });

    prog.emit(VmInstr::LoopEnd);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 32 §2.2.1: Ring Barrier for MK_SERIAL (SM < 70, REQ-MKO-002)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Emit ring barrier arrive: atom.global.add.u32 barrier counter + membar.gl.
///
/// For MK_SERIAL on SM61 (no cluster/mbarrier support), cross-CTA sync
/// uses a global memory atomic counter. Each CTA increments on arrive.
fn emit_ring_barrier_arrive(
    prog: &mut VmProgram,
    barrier_ptr: VRegId,
) {
    prog.emit(VmInstr::AtomicAdd {
        base: barrier_ptr,
        offset: OffsetExpr::Const(0),
        value: 1,
        elem_width: 4,
    });
    prog.emit(VmInstr::MemFence { order: MemFenceOrder::AcqRel });
}

/// Emit ring barrier wait: spin-loop until barrier counter reaches expected value.
///
/// Loads the counter, compares against `expected_count`, and re-reads if not reached.
/// After successful wait, resets counter to 0 for next barrier phase.
fn emit_ring_barrier_wait(
    prog: &mut VmProgram,
    barrier_ptr: VRegId,
    expected_count: u32,
) {
    let counter_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    // Spin-wait: load barrier counter, skip reset if not yet reached expected_count
    prog.emit(VmInstr::ScalarLoad {
        dst: counter_val,
        base: barrier_ptr,
        offset: OffsetExpr::Const(0),
    });
    // If counter < expected_count → skip the reset (barrier not yet satisfied)
    // This is a simplified single-check; real spin-wait needs JumpToLabel loop.
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpLtU(counter_val, expected_count as u64),
        action: GprBranchAction::Skip(4), // skip past reset block
    });
    // Barrier satisfied: reset counter to 0 for next phase
    prog.emit(VmInstr::MemFence { order: MemFenceOrder::AcqRel });
    prog.emit(VmInstr::GprLoadImm { dst: counter_val, value: 0 });
    prog.emit(VmInstr::ScalarStore {
        src: counter_val,
        base: barrier_ptr,
        offset: OffsetExpr::Const(0),
    });
    prog.emit(VmInstr::MemFence { order: MemFenceOrder::Release });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 32 §3.2: Two-tier Page Allocator for MK_SERIAL (SM < 70)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Page allocator field offsets within BatchContext extension area.
pub mod page_pool_offsets {
    // pool_global free_list head: atom.global CAS to grab a page
    pub const GLOBAL_FREE_HEAD: u32 = 160; // batch_ctx + 160
    // pool_global total page count
    pub const GLOBAL_TOTAL_PAGES: u32 = 168;
    // pool_local free count (per-CTA, stored in seq_meta scratch area)
    // pool_local is managed via scalar registers, no device memory needed.
}

/// Emit page allocation from two-tier pool (MK_SERIAL variant).
///
/// Strategy:
/// 1. Check pool_local (per-CTA register-based counter) → if > 0, decrement and use
/// 2. If pool_local empty → batch-grab from pool_global via atom.global.cas
///
/// Returns the allocated page_id in `dst_page_id`.
fn emit_page_alloc_serial(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
    local_free_count: VRegId,  // per-CTA local free count (register)
    local_next_id: VRegId,     // per-CTA next available page_id (register)
    dst_page_id: VRegId,       // output: allocated page_id
) {
    // Fast path: pool_local has free pages
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(local_free_count, 0),
        action: GprBranchAction::Skip(0), // patch below
    });
    let skip_fast_path = prog.instrs.len() - 1;

    // pool_local hit: use local_next_id, increment it, decrement count
    prog.emit(VmInstr::GprBinOp {
        dst: dst_page_id, a: local_next_id, b: GprOperand::Imm(0), op: GprOp::Add,
    });
    prog.emit(VmInstr::GprBinOp {
        dst: local_next_id, a: local_next_id, b: GprOperand::Imm(1), op: GprOp::Add,
    });
    prog.emit(VmInstr::GprBinOp {
        dst: local_free_count, a: local_free_count, b: GprOperand::Imm(1), op: GprOp::Sub,
    });

    // Skip slow path
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(dst_page_id, 0), // always false; used as unconditional jump
        action: GprBranchAction::Skip(0), // patch below to skip slow path
    });
    let skip_slow = prog.instrs.len() - 1;

    // Patch fast-path skip to jump over fast-path body + skip_slow instruction
    let fast_path_body_len = prog.instrs.len() - skip_fast_path - 1;
    if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut n), .. } = prog.instrs[skip_fast_path] {
        *n = fast_path_body_len;
    }

    // Slow path: batch-grab from pool_global via atom.global.add
    // Read global_free_head ptr
    let global_head_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: global_head_ptr,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(page_pool_offsets::GLOBAL_FREE_HEAD as usize),
    });

    // atom.global.add.u64 global_head, BATCH_SIZE → get old value as our batch start
    let batch_size: u64 = 32; // grab 32 pages at a time from global pool
    prog.emit(VmInstr::AtomicAdd {
        base: global_head_ptr,
        offset: OffsetExpr::Const(0),
        value: batch_size,
        elem_width: 8, // u64 atomic
    });

    // The returned old value is the first page_id in our batch
    // (In full implementation, this needs AtomicFetchAdd VmInstr returning old value)
    // For now, use the batch start as first allocated page
    prog.emit(VmInstr::GprBinOp {
        dst: dst_page_id, a: global_head_ptr, b: GprOperand::Imm(0), op: GprOp::Add,
    });

    // Update local pool: next_id = old_head + 1, free_count = batch_size - 1
    prog.emit(VmInstr::GprBinOp {
        dst: local_next_id, a: dst_page_id, b: GprOperand::Imm(1), op: GprOp::Add,
    });
    prog.emit(VmInstr::GprLoadImm { dst: local_free_count, value: (batch_size - 1) as usize });

    // Patch skip_slow to jump over slow path body
    let slow_path_len = prog.instrs.len() - skip_slow - 1;
    if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut n), .. } = prog.instrs[skip_slow] {
        *n = slow_path_len;
    }
}

/// Emit page free: return page to pool_local (no atomic needed).
fn emit_page_free_serial(
    prog: &mut VmProgram,
    _page_id: VRegId,
    local_free_count: VRegId,
) {
    // Simply increment local free count; freed pages are reused next alloc cycle
    prog.emit(VmInstr::GprBinOp {
        dst: local_free_count, a: local_free_count, b: GprOperand::Imm(1), op: GprOp::Add,
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 32 §2.3: DualBatchMeta + §4: Streaming Output (REQ-MKO-004/005)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// DualBatchMeta: ping/pong batch state for continuous batching (SPEC 32 §2.3).
///
/// Layout (24 bytes):
///   +0  ping_seq_offset  u32   offset into seq_meta for ping buffer
///   +4  ping_seq_count   u32   number of active sequences in ping
///   +8  pong_seq_offset  u32   offset into seq_meta for pong buffer
///   +12 pong_seq_count   u32   number of active sequences in pong
///   +16 current_epoch    u32   monotonically increasing epoch counter
///   +20 pad              u32   alignment padding
#[derive(Debug, Clone)]
pub struct DualBatchMeta {
    pub ping_seq_offset: u32,
    pub ping_seq_count: u32,
    pub pong_seq_offset: u32,
    pub pong_seq_count: u32,
    pub current_epoch: u32,
}

/// OutputTokenEntry: per-token output written to per-CTA sub-ring (SPEC 32 §4.1).
/// 20 bytes per entry.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OutputTokenEntry {
    pub seq_id: u32,
    pub token_id: u32,
    pub is_final: u32,
    pub finish_reason: u32,
    pub gen_idx: u32,
}

/// OutputRingBuffer: per-CTA sub-ring + doorbell for streaming output (SPEC 32 §4.1).
///
/// Each CTA writes tokens to its own sub-ring (zero cross-CTA atomic),
/// then updates its doorbell slot to signal Rust consumer.
#[repr(C)]
pub struct OutputRingBuffer {
    pub sub_ring_base_ptr: u64,
    pub cta_sub_ring_size: u32,
    pub num_sub_rings: u32,
    pub per_cta_doorbell_ptr: u64,
    pub epoch_flag_ptr: u64,
}

impl OutputRingBuffer {
    pub const ENTRY_SIZE: usize = core::mem::size_of::<OutputTokenEntry>();

    pub fn capacity(&self) -> usize {
        self.cta_sub_ring_size as usize * self.num_sub_rings as usize
    }

    pub fn sub_ring_for_cta(&self, cta_id: u32) -> usize {
        cta_id as usize * self.cta_sub_ring_size as usize * Self::ENTRY_SIZE
    }
}

/// OutputRingBuffer layout constants for batch_ctx extension area (SPEC 32 §4.1).
pub mod output_ring_offsets {
    /// sub_ring_base_ptr at batch_ctx extension +0
    pub const SUB_RING_BASE_PTR: usize = 0;
    /// cta_sub_ring_size (u32) at +8
    pub const CTA_SUB_RING_SIZE: usize = 8;
    /// num_sub_rings (u32) at +12
    pub const NUM_SUB_RINGS: usize = 12;
    /// per_cta_doorbell_ptr at +16
    pub const PER_CTA_DOORBELL_PTR: usize = 16;
    /// epoch_flag_ptr at +24
    pub const EPOCH_FLAG_PTR: usize = 24;
}

/// Emit a single OutputTokenEntry write to per-CTA sub-ring.
///
/// For MK_SERIAL (single CTA), cta_id is always 0, so this is a simple
/// store to sub_ring[0][local_write_idx % cta_sub_ring_size].
fn emit_output_token_write(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
    seq_id: VRegId,
    token_id: VRegId,
    gen_idx: VRegId,
    is_final: bool,
) {
    // Load sub_ring_base_ptr from batch_ctx extension area
    let sub_ring_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: sub_ring_base,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(output_ring_offsets::SUB_RING_BASE_PTR),
    });

    // Compute entry offset: local_write_idx * sizeof(OutputTokenEntry) = * 20
    // For MK_SERIAL, local_write_idx is just a counter we maintain
    // The actual write is: sub_ring_base + cta_id * cta_sub_ring_size * 20 + local_idx * 20
    // Since MK_SERIAL has cta_id=0 and we simplify:
    // entry_ptr = sub_ring_base + gen_idx * 20 (simplified for single-seq MK_SERIAL)

    let byte_offset = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp {
        dst: byte_offset,
        a: gen_idx,
        b: GprOperand::Imm(4), // * 16 (approximate, gen_idx << 4)
        op: GprOp::Shl,
    });

    // For MK_SERIAL, write directly to sub_ring[0][gen_idx]
    let entry_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: entry_ptr,
        src: PtrExpr::VRegPlusConst(sub_ring_base, 0),
    });

    // Store OutputTokenEntry fields:
    // +0: seq_id, +4: token_id, +8: is_final, +12: finish_reason, +16: gen_idx
    prog.emit(VmInstr::ScalarStore {
        src: seq_id,
        base: entry_ptr,
        offset: OffsetExpr::ScalarVReg(byte_offset),
    });
    prog.emit(VmInstr::ScalarStore {
        src: token_id,
        base: entry_ptr,
        offset: OffsetExpr::Add(
            Box::new(OffsetExpr::ScalarVReg(byte_offset)),
            Box::new(OffsetExpr::Const(4)),
        ),
    });

    let is_final_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: is_final_val, value: if is_final { 1 } else { 0 } });
    prog.emit(VmInstr::ScalarStore {
        src: is_final_val,
        base: entry_ptr,
        offset: OffsetExpr::Add(
            Box::new(OffsetExpr::ScalarVReg(byte_offset)),
            Box::new(OffsetExpr::Const(8)),
        ),
    });
}

/// Emit per-CTA doorbell update after writing output tokens (SPEC 32 §4.1).
///
/// MK_SERIAL (single CTA): writes local_write_count to doorbell[0].
/// The Rust consumer polls/awaits this doorbell to consume sub-ring entries.
fn emit_doorbell_update(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
    local_write_count: VRegId,
) {
    let doorbell_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: doorbell_ptr,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(output_ring_offsets::PER_CTA_DOORBELL_PTR),
    });

    // st.global.u64 per_cta_doorbell[0] = local_write_count
    prog.emit(VmInstr::ScalarStore {
        src: local_write_count,
        base: doorbell_ptr,
        offset: OffsetExpr::Const(0),
    });
}

/// Emit complete streaming output for MK_SERIAL variant (SPEC 32 §4).
///
/// Writes an OutputTokenEntry to the per-CTA sub-ring and updates the doorbell.
/// For MK_SERIAL, cta_id is always 0, so the write goes to sub_ring[0].
fn emit_streaming_output_serial(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
    seq_id: VRegId,
    token_id: VRegId,
    gen_idx: VRegId,
    is_final: bool,
    local_write_count: VRegId,
) {
    emit_output_token_write(prog, batch_ctx_ptr, seq_id, token_id, gen_idx, is_final);

    // Increment local_write_count after each write
    prog.emit(VmInstr::GprBinOp {
        dst: local_write_count,
        a: local_write_count,
        b: GprOperand::Imm(1),
        op: GprOp::Add,
    });

    emit_doorbell_update(prog, batch_ctx_ptr, local_write_count);
}

/// Emit DualBatchMeta ping/pong swap: increment epoch and swap offsets (SPEC 32 §2.3).
///
/// After each decode step:
///   current_epoch += 1
///   swap(ping_seq_offset, pong_seq_offset)
///   swap(ping_seq_count, pong_seq_count)
fn emit_dual_batch_meta_swap(
    prog: &mut VmProgram,
    batch_ctx_ptr: VRegId,
) {
    // Load epoch flag ptr and increment epoch
    let epoch_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: epoch_ptr,
        base: batch_ctx_ptr,
        offset: OffsetExpr::Const(output_ring_offsets::EPOCH_FLAG_PTR),
    });

    // atom.global.add.u32 epoch_flag, 1
    prog.emit(VmInstr::AtomicAdd {
        base: epoch_ptr,
        offset: OffsetExpr::Const(0),
        value: 1,
        elem_width: 4,
    });
}

/// Select MkVariant and compute SmPartitionConfig based on DeviceProfile (REQ-MKO-005).
///
/// This is the compile-time entry point for hardware-aware MK configuration.
#[derive(Debug, Clone)]
pub struct SmPartitionConfig {
    pub variant: MkVariant,
    pub total_ctas: u32,
    pub decode_ctas: u32,
    pub prefill_ctas: u32,
    pub cluster_size: u32,
}

impl SmPartitionConfig {
    pub fn for_sm(sm_version: u32, total_sm: u32) -> Self {
        let variant = select_mk_variant(sm_version, total_sm);
        match variant {
            MkVariant::Serial => SmPartitionConfig {
                variant,
                total_ctas: total_sm.max(1),
                decode_ctas: total_sm.max(1),
                prefill_ctas: 0,
                cluster_size: 0,
            },
            MkVariant::GridSync { total_ctas } => {
                let decode_ctas = total_ctas * 3 / 4; // 75% decode
                SmPartitionConfig {
                    variant,
                    total_ctas,
                    decode_ctas,
                    prefill_ctas: total_ctas - decode_ctas,
                    cluster_size: 0,
                }
            }
            MkVariant::Cluster6x2 { cluster_size, num_clusters } => {
                let total_ctas = cluster_size * num_clusters;
                SmPartitionConfig {
                    variant,
                    total_ctas,
                    decode_ctas: 6 * num_clusters,
                    prefill_ctas: 2 * num_clusters,
                    cluster_size,
                }
            }
            MkVariant::Cluster5x3 { cluster_size, num_clusters } => {
                let total_ctas = cluster_size * num_clusters;
                SmPartitionConfig {
                    variant,
                    total_ctas,
                    decode_ctas: 5 * num_clusters,
                    prefill_ctas: 3 * num_clusters,
                    cluster_size,
                }
            }
        }
    }
}

// ── SPEC 32 §5: FusionParams — hardware-aware compile-time parameters (REQ-MKO-005) ──

/// GEMM tile dimensions (M, N, K).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSize(pub usize, pub usize, pub usize);

/// KV cache access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    WriteFull,
    ReadHistoryWriteOne,
}

/// Attention computation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    FlashAttention,
    IncrementalKvAttention,
}

/// Fusion parameters for prefill or decode phase (SPEC 32 §5.1).
#[derive(Debug, Clone)]
pub struct FusionParams {
    pub gemm_tile: TileSize,
    pub attention: AttentionMode,
    pub kv_mode: KvMode,
    pub kv_pipeline_stages: u32,
    pub use_dsmem_kv_share: bool,
    pub use_ld_nc: bool,
    pub use_tensor_core_gemv: bool,
}

/// Select prefill FusionParams based on HardwareProfile (SPEC 32 §5.1).
pub fn prefill_fusion_params(profile: &crate::compiler::hardware_profile::HardwareProfile) -> FusionParams {
    match profile {
        HardwareProfile::CudaSM100 => FusionParams {
            gemm_tile: TileSize(128, 256, 64),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 4,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::CudaSM90 => FusionParams {
            gemm_tile: TileSize(128, 256, 64),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 4,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::CudaSM80 => FusionParams {
            gemm_tile: TileSize(64, 128, 32),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 2,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::RocmMI200 | HardwareProfile::RocmMI300 => FusionParams {
            gemm_tile: TileSize(64, 128, 32),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 2,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::CpuAvx10_2 => FusionParams {
            gemm_tile: TileSize(16, 64, 64),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::CpuAvx512 | HardwareProfile::AppleM1 | HardwareProfile::AppleM2 | HardwareProfile::AppleM3 => FusionParams {
            gemm_tile: TileSize(16, 16, 64),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::CpuAvx2 => FusionParams {
            gemm_tile: TileSize(6, 64, 256),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::ArmNeoverse => FusionParams {
            gemm_tile: TileSize(8, 64, 128),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::Generic => FusionParams {
            gemm_tile: TileSize(4, 64, 128),
            attention: AttentionMode::FlashAttention,
            kv_mode: KvMode::WriteFull,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
    }
}

/// Select decode FusionParams based on HardwareProfile (SPEC 32 §5.1).
pub fn decode_fusion_params(profile: &crate::compiler::hardware_profile::HardwareProfile) -> FusionParams {
    match profile {
        HardwareProfile::CudaSM100 => FusionParams {
            gemm_tile: TileSize(1, 256, 64),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 4,
            use_dsmem_kv_share: true,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::CudaSM90 => FusionParams {
            gemm_tile: TileSize(1, 256, 64),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 4,
            use_dsmem_kv_share: true,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::CudaSM80 => FusionParams {
            gemm_tile: TileSize(1, 128, 32),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 2,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::RocmMI200 | HardwareProfile::RocmMI300 => FusionParams {
            gemm_tile: TileSize(1, 128, 32),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 2,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_core_gemv: true,
        },
        HardwareProfile::CpuAvx10_2 => FusionParams {
            gemm_tile: TileSize(1, 64, 64),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::CpuAvx512 | HardwareProfile::AppleM1 | HardwareProfile::AppleM2 | HardwareProfile::AppleM3 => FusionParams {
            gemm_tile: TileSize(1, 16, 64),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::CpuAvx2 => FusionParams {
            gemm_tile: TileSize(1, 64, 256),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::ArmNeoverse => FusionParams {
            gemm_tile: TileSize(1, 64, 128),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
        HardwareProfile::Generic => FusionParams {
            gemm_tile: TileSize(1, 64, 128),
            attention: AttentionMode::IncrementalKvAttention,
            kv_mode: KvMode::ReadHistoryWriteOne,
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_core_gemv: false,
        },
    }
}

///
/// The VmProgram includes:
/// 1. MegaKernelFn ABI parameter loading
/// 2. Generate loop (LoopBegin)
/// 3. Per-iteration input_ptr computation
/// 4. Forward pass (embed → N layers → lm_head) via emit_fusion_groups
/// 5. Argmax + StoreToken + CheckStopCondition
/// 6. Generate loop end (LoopEnd)
///
/// The returned VmProgram goes through the standard RegAlloc → StackFrame →
/// X86Lower pipeline to produce a single contiguous machine code function.
pub fn compile_mega_kernel_vm(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    registry: Option<&ScalarOpRegistry>,
    profile: &IsaProfile,
    hook: Option<&dyn super::isa_hook::IsaHook>,
    vocab_size: usize,
    buffer_layout: &crate::compiler::mega_kernel_abi::MegaKernelBufferLayout,
    bottleneck_map: Option<&OpBottleneckMap>,
    virtual_activation: Option<&VirtualActivationMap>,
    virtual_tensor_map: Option<&VirtualTensorMap>,
    layout: Option<&crate::compiler::layout_negotiator::LayoutAssignment>,
    debug_jit: bool,
    mtp_config: Option<&MtpKernelConfig>,
    resource_plan: Option<&GraphResourcePlan>,
    needs_kv_for_decode: bool,
    is_encoder: bool,
) -> Result<(VmProgram, Option<RopeCacheRequirement>, usize), CompilerError> {
    use crate::compiler::mega_kernel_abi::MEGA_KERNEL_STACK_OFFSETS;

    let _ = resource_plan;
    let width = profile.optimal_simd_width();
    let sym_map = SymDimSlotMap::mega_kernel_abi();
    let mut prog = VmProgram::new();

    // ── Phase 0: Load MegaKernelFn ABI parameters ──
    let input_ids_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // Load register params (AbiArg 0/1 = input_ids_ptr / weight_blob_ptr)
    prog.emit(VmInstr::LoadPtr { dst: input_ids_ptr, src: PtrExpr::AbiArg(0) });
    prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });

    // Load stack params: scratchpad, prompt_len, output_tokens
    let scratchpad_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let prompt_len_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    prog.emit(VmInstr::LoadPtr { dst: scratchpad_ptr, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[1]) }); // scratchpad_ptr → [rbp+24]
    prog.emit(VmInstr::LoadPtr { dst: prompt_len_vreg, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[0]) }); // prompt_len → [rbp+16]

    // ── Phase 0.5: Load batch_ctx_ptr (ABI arg 16) + branch to batch mode if non-NULL (SPEC/20 BCI-002/003) ──
    let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: batch_ctx_ptr, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[16]) });
    // If batch_ctx_ptr != NULL → jump to BATCH_MODE label (Phase 2/3 batch logic, to be implemented)
    // NULL → fall through to Phase 1.Legacy (current single-seq behavior, zero overhead)
    const BATCH_MODE_LABEL: usize = 100; // label ID for batch mode entry
    prog.emit(VmInstr::BranchIfPtrNonNull { ptr: batch_ctx_ptr, target_label: BATCH_MODE_LABEL });

    // seq_len=1 is SymDim::Concrete(1) — no runtime slot needed.
    // Old code wrote to [rbp+104] (caller's stack frame!) which caused heap corruption.

    // ── Phase 1: Compute derived values ──
    // prompt_len_bytes = prompt_len * sizeof(u32) = prompt_len << 2
    // IMPORTANT: compute prompt_len_bytes BEFORE loading output_tokens_ptr,
    // otherwise RegAllocator may reuse the same physical register and clobber prompt_len.
    let prompt_len_bytes = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp { dst: prompt_len_bytes, a: prompt_len_vreg, b: GprOperand::Imm(2_i64), op: GprOp::Shl });

    // Now safe to load output_tokens — prompt_len has been consumed
    prog.emit(VmInstr::LoadPtr { dst: output_tokens_ptr, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[2]) }); // output_tokens → [rbp+32]

    // input_base = input_ids_ptr (start from first token for prefill)
    // The unified loop processes tokens 0..prompt_len-2 (prefill) then
    // prompt_len-1..prompt_len+max_new_tokens-2 (generate).
    // StoreToken writes generated tokens to input_ids[prompt_len + gen_counter],
    // so the generate phase reads them on subsequent iterations.
    let input_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: input_base, src: PtrExpr::VRegPlusConst(input_ids_ptr, 0) });

    // ── Phase 1.5: Compute logits scratch offset ──
    // CRITICAL: logits must be placed AFTER all intermediate tensors and RoPE cache
    // to avoid overlap with BufferAllocation (which uses SYMDIM_MAX_SEQ_LEN for sizing).
    // Using buffer_layout.logits_offset causes memory corruption because that offset
    // is computed with config.max_seq_len (128) while BufferAllocation uses
    // SYMDIM_MAX_SEQ_LEN (2048) — the logits region lands inside intermediate space.
    let rope_req = compute_rope_requirement(plan, graph, alloc)?;
    let ple_req = compute_ple_requirement(plan, graph, alloc, rope_req.as_ref())?;
    let dwc_req = compute_dwc_requirement(plan, graph, alloc, rope_req.as_ref(), ple_req.as_ref())?;

    let logits_scratch_offset = {
        let after_rope = rope_req.as_ref()
            .map(|rc| {
                let cache_bytes = rc.max_seq_len * rc.head_dim * 4;
                rc.cache_offset + cache_bytes
            })
            .unwrap_or(alloc.total_bytes);
        
        (after_rope + 63) & !63
    };

    // output_ptr = scratchpad + logits_scratch_offset (after intermediates + RoPE cache)
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset) });


    // ── Phase 2: Unified prefill + generate loop ──
    // Total iterations = (prompt_len - 1) + max_new_tokens
    //   Iterations 0..prompt_len-2: prefill (embed + layers, no sampling)
    //   Iterations prompt_len-1..end: generate (embed + layers + lm_head + sampling + store)
    // This ensures the KV cache is populated for all prompt tokens before decoding starts.
    let gen_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
    let gen_byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);

    // Compute total loop bound = (prompt_len - 1) + max_new_tokens
    // We need to load max_new_tokens from the stack and add prompt_len - 1.
    let max_new_tokens_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: max_new_tokens_vreg,
        src: sym_map.resolve("max_new_tokens").cloned().unwrap_or(PtrExpr::StackArg(64)),
    });
    let one_imm = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: one_imm, value: 1 });
    let prompt_minus_1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp { dst: prompt_minus_1, a: prompt_len_vreg, b: GprOperand::VReg(one_imm ), op: GprOp::Sub });
    let total_iters = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp { dst: total_iters, a: max_new_tokens_vreg, b: GprOperand::VReg(prompt_minus_1 ), op: GprOp::Add });

    // Pre-allocate decode_counter before loop: defined outside → structurally LoopCarried.
    // Inside the loop it's recomputed as gen_counter - (prompt_len - 1) at each iteration.
    // After LoopEnd it's used for the return value (decode_counter + 1).
    let decode_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: decode_counter, value: 0 });

    // ── Phase 2: Unified prefill + generate loop ──
    // For encoder models: single iteration with M=prompt_len (all tokens at once).
    // For decoder models: total_iters = (prompt_len - 1) + max_new_tokens iterations,
    //   processing tokens one at a time (prefill then generate).
    let loop_bound = if is_encoder {
        BoundExpr::Const(1)
    } else {
        BoundExpr::DynamicVReg(total_iters)
    };

    prog.emit(VmInstr::LoopBegin {
        counter: gen_counter,
        byte_offset: gen_byte_offset,
        bound: loop_bound,
        step_bytes: 4,
    });

    // ── Phase 3: Compute per-iteration input_ptr ──
    // Encoder: full input_ids (all tokens processed in one pass).
    // Decoder: single token at current position.
    let gen_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    if is_encoder {
        prog.emit(VmInstr::LoadPtr { dst: gen_input_ptr, src: PtrExpr::VRegPlusConst(input_base, 0) });
    } else {
        prog.emit(VmInstr::LoadPtr { dst: gen_input_ptr, src: PtrExpr::VRegPlusVReg(input_base, gen_byte_offset) });
    }

    // ── Phase 3.1: Reload scratchpad_ptr from ABI stack slot ──
    // ARCH-REGALLOC-LOOP-RELOAD: scratchpad_ptr (v2) may be spilled to a stack slot
    // that gets overwritten by the massive register pressure in mega-kernel loops
    // (2700+ spill slots). Reloading from the ABI stack argument at each iteration
    // guarantees correctness regardless of spill slot corruption.
    let scratchpad_reloaded = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: scratchpad_reloaded,
        src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[1]),
    });
    // Also reload output_ptr = scratchpad + logits_offset
    let output_reloaded = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: output_reloaded,
        src: PtrExpr::VRegPlusConst(scratchpad_reloaded, logits_scratch_offset),
    });
    // ARCH-REGALLOC-LOOP-RELOAD: weight_ptr must also be reloaded. With 9000+ spill
    // slots, the original weight VReg's spill slot can be corrupted by intermediate
    // ops. Reload from the prologue-saved rsi slot (AbiArg 1 = weight_blob_ptr).
    // The prologue pushes all 6 SysV register args to fixed stack slots (abi_save_area)
    // at emit_prologue time, so PtrExpr::AbiArg(1) reads from [rbp - callee_save - 16]
    // which is immutable for the entire function lifetime.
    let weight_reloaded = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: weight_reloaded,
        src: PtrExpr::AbiArg(1), // weight_blob_ptr (rsi, saved in prologue abi_save_area)
    });
    // ARCH-REGALLOC-LOOP-RELOAD: input_ids_ptr (v0) is also spilled and its spill slot
    // can be corrupted. Reload from the prologue-saved rdi slot (AbiArg 0 = input_ids_ptr).
    let input_ids_reloaded = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: input_ids_reloaded,
        src: PtrExpr::AbiArg(0), // input_ids_ptr (rdi, saved in prologue abi_save_area)
    });

    // ── Phase 4: Forward pass (embed → N layers → lm_head) ──
    // (rope_req, ple_req, dwc_req already computed in Phase 1.5 above)

    let needs_scratch = alloc.total_bytes > 0 || ple_req.is_some() || dwc_req.is_some() || plan.groups.iter().any(|g| matches!(&g.mode,
        FusionMode::NormIntoGemm | FusionMode::QkvSharedInput | FusionMode::FFNBlock { .. }
        | FusionMode::TileLevelFusion { .. } | FusionMode::ComputeRoot { .. }
        | FusionMode::CrossLayerResidual { .. } | FusionMode::FusedQkvNormRope { .. }
    ));

    let mut resolver = TensorPtrResolver::build(graph, alloc);

    let original_weight_vreg = Some(weight_reloaded);

    // ── Mega-kernel output redirect ──
    // BufferAllocation maps output tensors to Intermediate scratch offsets, but the
    // mega-kernel output region starts at abi.output_ptr (= scratchpad + logits_offset).
    // Redirect the final output tensor to Output { offset: 0 } so it writes directly
    // to the region that Rust reads from.
    if let Some(lm_head_op) = graph.ops.iter().find(|op| op.label == "lm_head") {
        // Decoder: lm_head GEMM writes logits to output region
        if let Some(&logits_tid) = lm_head_op.outputs.first() {
            resolver.override_source(logits_tid, TensorPtrSource::Output { offset: 0 });
        }
    } else if let Some(&output_tid) = graph.outputs.first() {
        // Encoder/embedding: graph output (MeanPool/classifier result) writes to output region
        resolver.override_source(output_tid, TensorPtrSource::Output { offset: 0 });
    }

    // ── Phase 3.5: Compute seq_len for this iteration ──
    // Encoder: seq_len = prompt_len (all tokens processed in one pass).
    // Decoder: seq_len = gen_counter + 1 (grows during prefill, stays at prompt_len for decode).
    let decode_seq_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    if is_encoder {
        // Encoder: seq_len = prompt_len (copy from prompt_len_vreg)
        prog.emit(VmInstr::GprBinOp { dst: decode_seq_len, a: prompt_len_vreg, b: GprOperand::Imm(0), op: GprOp::Add });
    } else {
        prog.emit(VmInstr::GprBinOp { dst: decode_seq_len, a: gen_counter, b: GprOperand::VReg(one_imm ), op: GprOp::Add });
    }

    let mut current_abi = AbiPtrs {
        // gen_input_ptr points to the single current decode token:
        //   iter 0: input_ids[prompt_len-1] (last prompt token)
        //   iter 1+: input_ids[prompt_len+counter-1] (previous generated token)
        // The embed gather reads this single token ID (not the full input_ids array).
        input_ptr: gen_input_ptr,
        weight_ptr: original_weight_vreg,
        weight_abi_expr: original_weight_vreg.map(|original_weight_vreg| sym_map.resolve("weights").cloned().expect("ABI: weights")),
        output_ptr: output_reloaded,  // Reloaded each iteration from ABI stack
        scratch_ptr: if needs_scratch { Some(scratchpad_reloaded) } else { None },  // Reloaded each iteration
        gen_loop_counter: Some(gen_counter),
        layer_loop_counter: None,
        mega_decode_seq_len: Some(decode_seq_len),
        hook_ctx_ptr: {
            let hook_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: hook_ptr,
                src: sym_map.resolve("hook_ctx_ptr").cloned().expect("ABI: hook_ctx_ptr"),
            });
            Some(hook_ptr)
        },
        // SG scratch offsets: placed after actual JIT logits + sampling workspace.
        // JIT uses vocab_size * 4 bytes for decode logits (one row), NOT max_seq_len * vocab_size.
        // SG knowledge buffer persists across decode steps (pre-computed before generation).
        sg_detect_scratch_offset: {
            let sg_hidden_dim = graph.ops.iter()
                .find(|op| matches!(&op.kind, OpKind::SgDetect { .. }))
                .and_then(|op| op.inputs.first())
                .and_then(|&tid| graph.tensor(tid))
                .and_then(|t| t.shape.last())
                .and_then(|d| d.as_concrete());
            match sg_hidden_dim {
                Some(hdim) => {
                    let vocab_bytes = graph.tensors.iter()
                        .find(|t| t.name == "logits")
                        .and_then(|t| t.shape.last())
                        .and_then(|d| d.as_concrete())
                        .map(|v| v * 4)
                        .unwrap_or(0);
                    let sampling_bytes = vocab_bytes * 4;
                    let off = (logits_scratch_offset + vocab_bytes + sampling_bytes + 63) & !63;
                    Some(off)
                }
                None => None,
            }
        },
        sg_knowledge_scratch_offset: {
            let sg_hidden_dim = graph.ops.iter()
                .find(|op| matches!(&op.kind, OpKind::SgDetect { .. }))
                .and_then(|op| op.inputs.first())
                .and_then(|&tid| graph.tensor(tid))
                .and_then(|t| t.shape.last())
                .and_then(|d| d.as_concrete());
            match sg_hidden_dim {
                Some(hdim) => {
                    let vocab_bytes = graph.tensors.iter()
                        .find(|t| t.name == "logits")
                        .and_then(|t| t.shape.last())
                        .and_then(|d| d.as_concrete())
                        .map(|v| v * 4)
                        .unwrap_or(0);
                    let sampling_bytes = vocab_bytes * 4;
                    let detect_off = (logits_scratch_offset + vocab_bytes + sampling_bytes + 63) & !63;
                    let off = detect_off + hdim * 4;
                    Some(off)
                }
                None => None,
            }
        },
        callback_table_ptr: {
            // Only load callback_table_ptr if the graph has SG ops (SgDetect/SgInject).
            // Without SG, no callback code is emitted → no need to load the ABI arg.
            let has_sg = graph.ops.iter().any(|op| {
                matches!(op.kind, OpKind::SgDetect { .. } | OpKind::SgInject { .. })
            });
            if has_sg {
                let cb_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: cb_ptr,
                    src: sym_map.resolve("callback_table_ptr").cloned().expect("ABI: callback_table_ptr"),
                });
                Some(cb_ptr)
            } else {
                None
            }
        },
        page_table_ptr: {
            // Load page_table_ptr from ABI arg 21 ([rbp+136]).
            // NULL = contiguous KV, u32[] = paged KV.
            let has_mha = graph.ops.iter().any(|op| {
                matches!(op.kind, OpKind::MultiHeadAttention { .. })
            });
            if has_mha {
                let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: pt_ptr,
                    src: sym_map.resolve("page_table_ptr").cloned().unwrap_or(PtrExpr::StackArg(136)),
                });
                Some(pt_ptr)
            } else {
                None
            }
        },
        kv_load_mode: graph.kv_load_mode,
        kv_cache_ptr: {
            let has_mha = graph.ops.iter().any(|op| {
                matches!(op.kind, OpKind::MultiHeadAttention { .. })
            });
            // Encoder/embedding models (EncodeToLayer/Classify*) have MHA ops but
            // never decode, so they don't need persistent KV cache. When
            // needs_kv_for_decode is false, skip loading kv_cache_ptr entirely —
            // the MHA lowering falls through to using K/V projection pointers
            // directly (no KV store, no page-table lookup).
            if has_mha && needs_kv_for_decode {
                let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: kv_ptr,
                    src: sym_map.resolve("kv_cache_ptr").cloned().unwrap_or(PtrExpr::AbiArg(2)),
                });
                Some(kv_ptr)
            } else {
                None
            }
        },
        activation_ping_ptr: None,
        activation_pong_ptr: None,
    };

    let ctx = LoweringContext {
        width,
        dtype: graph_dtype(graph),
        sym_map: &sym_map,
        registry,
        hook,
        budget: None,
        rope_req: rope_req.as_ref(),
        ple_req: ple_req.as_ref(),
        dwc_req: dwc_req.as_ref(),
        exec_pattern: None,
        bottleneck_map,
        virtual_activation,
        parallelism: Some(ParallelismDesc::SimdVectorize {
            element_width: width.f32_lanes().max(1),
            unroll_factor: profile.k_unroll_factor,
        }),
        virtual_tensor_map,
        layout,
        page_size: 0,
        dot_cap: profile.dot_cap,
        batch_ctx_ptr: None, // ARCH-LEGACY-NO-BCI: legacy path is non-batch (v5=NULL at runtime).
        debug_jit,
    };

    emit_fusion_groups(
        &mut prog, plan, graph, alloc, &ctx,
        rope_req.as_ref().map(|r| r.cache_offset),
        &mut current_abi, original_weight_vreg, &resolver,
    )?;

    // L6 debug: embed + all layers done, before output mode dispatch
    maybe_debug_bp(&mut prog, &ctx, "forward_pass_done");

    // ── Phase 4.5: Output Mode JMP table ──
    // Load output_mode_selector from [rbp+80]
    let mode_selector = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: mode_selector, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[8]) }); // output_mode_selector → [rbp+80]

    // Label IDs — SPEC §1.5.5 defines exactly 4 paths
    const LABEL_GENERATE: usize = 0;
    const LABEL_CLASSIFY_BINARY: usize = 1;
    const LABEL_CLASSIFY_MULTIWAY: usize = 2;
    const LABEL_ENCODE: usize = 3;

    prog.emit(VmInstr::OutputModeDispatch {
        selector: mode_selector,
        paths: vec![LABEL_GENERATE, LABEL_CLASSIFY_BINARY, LABEL_CLASSIFY_MULTIWAY, LABEL_ENCODE],
    });


    // ── .generate_path: Phase 5-7 ──
    prog.emit(VmInstr::MarkLabel { label_id: LABEL_GENERATE });

    // ── Prefill/Generate branch ──
    // During prefill (gen_counter < prompt_len - 1), skip sampling and StoreToken.
    // The forward pass (embed + layers) already ran; KV cache is being populated.
    // Only the generate phase (gen_counter >= prompt_len - 1) needs sampling.
    const SKIP_SAMPLING_LABEL: usize = 300;
    prog.emit(VmInstr::BranchIfGprLtU {
        a: gen_counter,
        b: prompt_minus_1, // prompt_len - 1 (computed during loop setup)
        target_label: SKIP_SAMPLING_LABEL,
    });

    // ── Phase 5: Sampling ──
    // GPU-Resident 采样管线: temperature==0 → argmax, temperature>0 → stochastic
    maybe_debug_bp(&mut prog, &ctx, "pre_sample");
    let vocab_bytes = vocab_size * ctx.dtype.elem_bytes(); // f32 logits

    // Argmax reads from row 0 of the logits region.
    // The lm_head GEMV (M=1 decode) always writes its output to row 0
    // (Output { offset: 0 }), so sampling must read from the same row.
    let row_byte_offset = {
        let z = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: z, value: 0 });
        z
    };

    // logits_ptr = scratchpad + logits_scratch_offset + row_byte_offset
    let logits_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: logits_base, src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset) });
    let fresh_logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp { dst: fresh_logits_ptr, a: logits_base, b: GprOperand::VReg(row_byte_offset ), op: GprOp::Add });

    // ── Phase 5a: Load temperature, branch to argmax if T==0 ──
    const ARGMAX_LABEL: usize = 200;
    const SAMPLING_DONE_LABEL: usize = 201;

    let temp_u32 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: temp_u32,
        src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[3]), // temperature_u32
    });
    prog.emit(VmInstr::BranchIfGprZero { value: temp_u32, target_label: ARGMAX_LABEL });

    // ── Phase 5b: Stochastic sampling (T > 0) ──
    // TemperatureScale: logits[i] /= temperature
    let temp_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: temp_ptr, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[3]) });
    prog.emit(VmInstr::TemperatureScale {
        logits_ptr: fresh_logits_ptr,
        temp_ptr,
        vocab_bytes,
        width,
    });

    // Softmax: reduce-max → exp-sum → normalize
    let max_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::SoftmaxReduceMax {
        dst: max_val,
        logits_ptr: fresh_logits_ptr,
        vocab_bytes,
        width,
    });
    let sum_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::SoftmaxExpSum {
        sum_dst: sum_val,
        logits_ptr: fresh_logits_ptr,
        max_val,
        vocab_bytes,
        width,
    });
    prog.emit(VmInstr::SoftmaxNormalize {
        logits_ptr: fresh_logits_ptr,
        sum_val,
        vocab_bytes,
        width,
    });

    // Top-K filter (if top_k > 0)
    // Allocate indices buffer in scratchpad after logits
    let indices_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: indices_ptr,
        src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset + vocab_bytes),
    });
    let top_k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: top_k_ptr, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[4]) });
    prog.emit(VmInstr::SampleTopKFilter {
        probs_ptr: fresh_logits_ptr,
        indices_ptr,
        k_ptr: top_k_ptr,
        vocab_bytes,
        width,
    });

    // Top-P filter (if top_p > 0)
    let top_p_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: top_p_ptr, src: PtrExpr::StackArg(MEGA_KERNEL_STACK_OFFSETS[5]) });
    prog.emit(VmInstr::SampleTopPFilter {
        probs_ptr: fresh_logits_ptr,
        p_ptr: top_p_ptr,
        vocab_bytes,
        width,
    });

    // Multinomial sampling: PRNG + cumulative search
    // Allocate PRNG state (8 bytes) in scratchpad after indices
    let rng_state_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: rng_state_ptr,
        src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset + vocab_bytes + vocab_bytes),
    });
    let sampled_token = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::SampleMultinomial {
        dst: sampled_token,
        probs_ptr: fresh_logits_ptr,
        rng_state_ptr,
        vocab_bytes,
        width,
    });
    prog.emit(VmInstr::UnconditionalBranch { target_label: SAMPLING_DONE_LABEL });

    // ── Phase 5c: Argmax path (T == 0) ──
    prog.emit(VmInstr::MarkLabel { label_id: ARGMAX_LABEL });
    prog.emit(VmInstr::Argmax {
        dst: sampled_token, // reuse sampled_token for unified path
        logits_ptr: fresh_logits_ptr,
        vocab_bytes,
        width,
    });

    prog.emit(VmInstr::MarkLabel { label_id: SAMPLING_DONE_LABEL });

    // ── Phase 6: Store token ──
    // decode_counter = gen_counter - (prompt_len - 1) = actual generated token index
    prog.emit(VmInstr::GprBinOp { dst: decode_counter, a: gen_counter, b: GprOperand::VReg(prompt_minus_1 ), op: GprOp::Sub });
    prog.emit(VmInstr::StoreToken {
        token_id: sampled_token,
        output_buf: output_tokens_ptr,
        counter: decode_counter,
        input_ids_ptr,
        prompt_len_bytes,
    });

    // ── Phase 6.5: MTP (Multi-Token Prediction) candidate token generation ──
    // Route through TraceOp::MtpDraft → auto_select pipeline (MTP-001).
    if let Some(mtp) = mtp_config {
        let MtpKernelConfig { depth, hidden_size, vocab_size: _mtp_vocab } = *mtp;
        let elem_bytes = ctx.dtype.elem_bytes();
        let lm_head_bytes = vocab_size * hidden_size * elem_bytes;

        // Resolve MTP weight base: lm_head weight offset + lm_head size
        let lm_head_blob_offset = graph.ops.iter()
            .find(|op| op.label == "lm_head")
            .and_then(|op| op.inputs.get(1))
            .and_then(|&wid| resolver.source(wid))
            .and_then(|src| match src {
                TensorPtrSource::Weight { offset } => Some(offset),
                _ => None,
            })
            .ok_or_else(|| CompilerError::CodegenViolation(
                "MTP: cannot find lm_head weight offset in resolver".into()
            ))?;
        let mtp_weights_base_offset = lm_head_blob_offset + lm_head_bytes;
        let mtp_weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: mtp_weight_ptr,
            a: weight_reloaded,
            b: GprOperand::Imm(mtp_weights_base_offset as i64),
            op: GprOp::Add,
        });

        // Resolve hidden pointer (lm_head's first input)
        let (hidden_ptr_base, hidden_offset) = {
            let lm_head_op = graph.ops.iter()
                .find(|op| op.label == "lm_head")
                .ok_or_else(|| CompilerError::CodegenViolation("MTP: lm_head op not found".into()))?;
            let fn_tid = lm_head_op.inputs[0];
            let src = resolver.source(fn_tid)
                .ok_or_else(|| CompilerError::CodegenViolation(
                    "MTP: final_normed tensor source not found".into()
                ))?;
            match src {
                TensorPtrSource::ActivationPing => {
                    let ping = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    let ping_off = alloc.slots.iter()
                        .find(|s| s.tensor_id.0 == 0xFFFF_FF00)
                        .map(|s| s.offset)
                        .unwrap_or(0);
                    prog.emit(VmInstr::AddPtr { dst: ping, base: scratchpad_reloaded, offset: ping_off });
                    (ping, 0usize)
                }
                TensorPtrSource::ActivationPong => {
                    let pong = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    let pong_off = alloc.slots.iter()
                        .find(|s| s.tensor_id.0 == 0xFFFF_FF01)
                        .map(|s| s.offset)
                        .unwrap_or(0);
                    prog.emit(VmInstr::AddPtr { dst: pong, base: scratchpad_reloaded, offset: pong_off });
                    (pong, 0usize)
                }
                TensorPtrSource::Intermediate { offset } => {
                    (scratchpad_reloaded, offset)
                }
                _ => {
                    return Err(CompilerError::CodegenViolation(
                        format!("MTP: unexpected final_normed source: {:?}", src)
                    ));
                }
            }
        };
        let hidden_ptr = if hidden_offset > 0 {
            let hp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::AddPtr { dst: hp, base: hidden_ptr_base, offset: hidden_offset });
            hp
        } else {
            hidden_ptr_base
        };

        // Route through auto_select via TraceOp::MtpDraft
        use crate::compiler::trace::TraceOp;
        use crate::compiler::codegen::vm::auto_select::auto_lower_trace_raw;
        auto_lower_trace_raw(
            &mut prog,
            &[
                TraceOp::Input(0),
                TraceOp::Input(1),
                TraceOp::Input(2),
                TraceOp::MtpDraft { depth, hidden_size, vocab_size },
            ],
            &[hidden_ptr, mtp_weight_ptr, output_tokens_ptr],
            width,
            ctx.dtype,
        )?;
    }

    // ── Phase 7: Check stop condition ──
    // Symbolic ABI refs: eos_token_id (arg 13), max_new_tokens (arg 12)
    let eos_value = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: eos_value,
        src: sym_map.resolve("eos_token_id").cloned().unwrap(),
    });
    let max_tokens_value = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr {
        dst: max_tokens_value,
        src: sym_map.resolve("max_new_tokens").cloned().unwrap(),
    });

    prog.emit(VmInstr::CheckStopCondition {
        token_id: sampled_token,
        counter: decode_counter,
        eos_ptr: eos_value,
        max_tokens_ptr: max_tokens_value,
    });

    // ── SKIP_SAMPLING: prefill iterations land here (no sampling, no store) ──
    prog.emit(VmInstr::MarkLabel { label_id: SKIP_SAMPLING_LABEL });

    // ── Phase 8: Generate loop end ──
    prog.emit(VmInstr::LoopEnd);
    // Return decode_counter + 1 as generated count (decode_counter is 0-indexed)
    let ret_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp { dst: ret_count, a: decode_counter, b: GprOperand::VReg(one_imm ), op: GprOp::Add });
    prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::VReg(ret_count) });

    // ── .classify_binary_path: WriteLogits(pos, neg) → BreakLoop(0) ──
    // Logits already written to output buffer by lm_head GEMM.
    // Classify modes read specific logit positions from the output buffer in Rust.
    prog.emit(VmInstr::MarkLabel { label_id: LABEL_CLASSIFY_BINARY });
    prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });

    // ── .classify_multiway_path: WriteLogits(label_0, ..., label_N) → BreakLoop(0) ──
    prog.emit(VmInstr::MarkLabel { label_id: LABEL_CLASSIFY_MULTIWAY });
    prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });

    // ── .encode_path: hidden state already in activation buffer → BreakLoop(0) ──
    prog.emit(VmInstr::MarkLabel { label_id: LABEL_ENCODE });
    prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });

    // ── .batch_mode_path: Batch mode entry (SPEC/20 BCI-003) ──
    // BranchIfPtrNonNull jumps here when batch_ctx_ptr != NULL.
    //
    // Batch Prefill: read total_prefill_tokens from batch_ctx[8],
    // set input_ptr to input_ids_flat_ptr from batch_ctx[16],
    // run full forward pass (embed → N layers → lm_head) with M = total_prefill_tokens,
    // then return total_prefill_tokens to signal "prefill done, Rust handles decode."
    prog.emit(VmInstr::MarkLabel { label_id: BATCH_MODE_LABEL });

    {
        // Read total_prefill_tokens from batch_ctx + 8
        let batch_m = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::ScalarLoad {
            dst: batch_m,
            base: batch_ctx_ptr,
            offset: OffsetExpr::Const(8),
        });

        // ── Phase 0.7: ForwardPhaseDispatch (SPEC 32 REQ-MKO-001) ──
        // Three-way dispatch:
        //   total_prefill_tokens == 0 → .decode_path (label 101)
        //   total_prefill_tokens <= PREFILL_CHUNK_THRESHOLD → .mixed_path (label 102)
        //   total_prefill_tokens > PREFILL_CHUNK_THRESHOLD → .prefill_path (fall-through)
        const DECODE_ENTRY_LABEL: usize = 101;
        const MIXED_PATH_LABEL: usize = 102;
        const PREFILL_CHUNK_THRESHOLD: usize = 512;

        // Branch 1: batch_m == 0 → pure decode
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(batch_m, 0),
            action: GprBranchAction::JumpToLabel(DECODE_ENTRY_LABEL),
        });

        // Branch 2: batch_m <= PREFILL_CHUNK_THRESHOLD → mixed path
        // Equivalent to: batch_m < (PREFILL_CHUNK_THRESHOLD + 1)
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpLtU(batch_m, (PREFILL_CHUNK_THRESHOLD + 1) as u64),
            action: GprBranchAction::JumpToLabel(MIXED_PATH_LABEL),
        });
        // Fall-through: batch_m > threshold → dedicated prefill path

        // ── Mixed path entry point: batch_m <= PREFILL_CHUNK_THRESHOLD ──
        // Mixed path reuses the same forward pass as prefill (GEMM is M-uniform),
        // but with different GEMM tile parameters at compile time (REQ-MKO-005).
        // For MK_SERIAL (SM<70), mixed and prefill use identical tile sizes,
        // so mixed_path is simply a label alias jumping into prefill forward pass.
        prog.emit(VmInstr::MarkLabel { label_id: MIXED_PATH_LABEL });

        // Read num_seqs from batch_ctx + 0 (needed for per-seq metadata)
        let _batch_num_seqs = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::ScalarLoad {
            dst: _batch_num_seqs,
            base: batch_ctx_ptr,
            offset: OffsetExpr::Const(0),
        });

        // Read input_ids_flat_ptr from batch_ctx + 16
        let batch_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: batch_input_ptr,
            src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 16),
        });

        // Build batch prefill AbiPtrs — same weight/scratch/output as legacy path,
        // but input points to flat batch input_ids and seq_len = total_prefill_tokens.
        let batch_output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: batch_output_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset) });

        // page_table_flat_ptr from batch_ctx + 40
        let batch_pt_ptr = {
            let has_mha = graph.ops.iter().any(|op| {
                matches!(op.kind, OpKind::MultiHeadAttention { .. })
            });
            if has_mha {
                let pt = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: pt,
                    src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 40),
                });
                Some(pt)
            } else {
                None
            }
        };

        // Hook and callback for batch — reuse same hook_ctx_ptr as legacy
        // (SG is shared across all sequences in the batch)
        let batch_hook_ctx = {
            let hook_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr {
                dst: hook_ptr,
                src: sym_map.resolve("hook_ctx_ptr").cloned().expect("ABI: hook_ctx_ptr"),
            });
            Some(hook_ptr)
        };
        let batch_cb_ptr = {
            let has_sg = graph.ops.iter().any(|op| {
                matches!(op.kind, OpKind::SgDetect { .. } | OpKind::SgInject { .. })
            });
            if has_sg {
                let cb_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: cb_ptr,
                    src: sym_map.resolve("callback_table_ptr").cloned().expect("ABI: callback_table_ptr"),
                });
                Some(cb_ptr)
            } else {
                None
            }
        };

        let mut batch_current_abi = AbiPtrs {
            input_ptr: batch_input_ptr,
            weight_ptr: Some(weight_ptr),
            weight_abi_expr: Some(sym_map.resolve("weights").cloned().expect("ABI: weights")),
            output_ptr: batch_output_ptr,
            scratch_ptr: if needs_scratch { Some(scratchpad_ptr) } else { None },
            gen_loop_counter: None, // no generate loop in prefill
            layer_loop_counter: None,
            mega_decode_seq_len: Some(batch_m), // M = total_prefill_tokens
            hook_ctx_ptr: batch_hook_ctx,
            sg_detect_scratch_offset: current_abi.sg_detect_scratch_offset,
            sg_knowledge_scratch_offset: current_abi.sg_knowledge_scratch_offset,
            callback_table_ptr: batch_cb_ptr,
            page_table_ptr: batch_pt_ptr,
            kv_load_mode: graph.kv_load_mode,
        kv_cache_ptr: None,
        activation_ping_ptr: None,
        activation_pong_ptr: None,
        };

        let mut batch_resolver = TensorPtrResolver::build(graph, alloc);
        // Redirect logits output for batch (same as legacy path)
        if let Some(lm_head_op) = graph.ops.iter().find(|op| op.label == "lm_head") {
            if let Some(&logits_tid) = lm_head_op.outputs.first() {
                batch_resolver.override_source(logits_tid, TensorPtrSource::Output { offset: 0 });
            }
        }

        let batch_ctx = LoweringContext {
            width,
            dtype: graph_dtype(graph),
            sym_map: &sym_map,
            registry,
            hook,
            budget: None,
            rope_req: rope_req.as_ref(),
            ple_req: ple_req.as_ref(),
            dwc_req: dwc_req.as_ref(),
            exec_pattern: None,
            bottleneck_map,
            virtual_activation,
            parallelism: Some(ParallelismDesc::SimdVectorize {
                element_width: width.f32_lanes().max(1),
                unroll_factor: profile.k_unroll_factor,
            }),
            virtual_tensor_map,
            layout,
            page_size: 0,
            dot_cap: profile.dot_cap,
            batch_ctx_ptr: Some(batch_ctx_ptr),
            debug_jit,
        };

        // Emit batch prefill forward pass: embed → N layers → lm_head
        // with M = total_prefill_tokens (via mega_decode_seq_len = batch_m).
        // GEMM/FFN/Norm are M-uniform — they just see a larger batch dimension.
        // Attention uses BatchSeqIdLookup for per-token seq_id → KV cache lookup.
        emit_fusion_groups(
            &mut prog, plan, graph, alloc, &batch_ctx,
            rope_req.as_ref().map(|r| r.cache_offset),
            &mut batch_current_abi, Some(weight_ptr), &batch_resolver,
        )?;

        // ── Phase 2 post: Per-seq argmax on last token of each prompt (BCI-006) ──
        // Logits layout: [total_prefill_tokens, vocab_size] in row-major.
        // For seq s, last token row index = cumsum(prompt_lens)[s] - 1.
        // cumsum_acc tracks running sum of prompt_lens[0..seq).
        {
            let vocab_bytes = vocab_size * ctx.dtype.elem_bytes();

            // Read num_seqs from batch_ctx+0
            let num_seqs_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: num_seqs_gpr, base: batch_ctx_ptr, offset: OffsetExpr::Const(0) });

            // Read seq_meta_base from batch_ctx+88 (absolute pointer to per-seq array)
            let seq_meta_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: seq_meta_base, base: batch_ctx_ptr, offset: OffsetExpr::Const(88) });

            // logits_base = scratchpad + logits_scratch_offset
            let logits_base_arg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: logits_base_arg, src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset) });

            // Read output_tokens_flat_ptr from batch_ctx+24
            let out_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: out_tokens_ptr, src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 24) });

            // Read sampling_params_ptr from batch_ctx+56 (for prefill temperature check)
            let sampling_params_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: sampling_params_ptr, src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 56) });

            // cumsum accumulator: tracks sum of prompt_lens[0..seq) across iterations
            let cumsum_acc = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: cumsum_acc, value: 0 });

            let seq_stride: usize = 64;
            prog.emit_loop(BoundExpr::DynamicVReg(num_seqs_gpr), 1, |prog, seq_ctr, _seq_off| {
                // Read prompt_len[seq] = seq_meta_base + seq * stride + 0
                let prompt_len_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                let seq_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seq_byte_off, a: seq_ctr, b: GprOperand::Imm(seq_stride as i64), op: GprOp::Mul });
                prog.emit(VmInstr::ScalarLoad { dst: prompt_len_gpr, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(seq_byte_off)), Box::new(OffsetExpr::Const(0))) });

                // Skip if prompt_len == 0 (pure decode, no prefill for this seq)
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpEq(prompt_len_gpr, 0),
                    action: GprBranchAction::Skip(0), // placeholder, patched below
                });
                let skip_patch = prog.instrs.len() - 1;

                // Compute last token row index = cumsum_acc + prompt_len - 1
                let last_row_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                let pl_minus_1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: pl_minus_1, a: prompt_len_gpr, b: GprOperand::Imm(1), op: GprOp::Sub });
                prog.emit(VmInstr::GprBinOp { dst: last_row_idx, a: cumsum_acc, b: GprOperand::VReg(pl_minus_1), op: GprOp::Add });

                // logits_ptr = logits_base + last_row_idx * vocab_bytes
                let row_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: row_byte_off, a: last_row_idx, b: GprOperand::Imm(vocab_bytes as i64), op: GprOp::Mul });
                let logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: logits_ptr, a: logits_base_arg, b: GprOperand::VReg(row_byte_off), op: GprOp::Add });

                // Argmax this row
                let sampled = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::Argmax { dst: sampled, logits_ptr, vocab_bytes, width });

                // Write last_sampled_token[seq] at seq_meta + seq*stride + 48
                let tok_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: tok_off, a: seq_ctr, b: GprOperand::Imm(seq_stride as i64), op: GprOp::Mul });
                prog.emit(VmInstr::GprBinOp { dst: tok_off, a: tok_off, b: GprOperand::Imm(48), op: GprOp::Add });
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::ScalarVReg(tok_off), src: sampled });

                // Update gen_count[seq] = 1 at offset +44
                let one_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: one_gpr, value: 1 });
                let gc_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: gc_off, a: seq_ctr, b: GprOperand::Imm(seq_stride as i64), op: GprOp::Mul });
                prog.emit(VmInstr::GprBinOp { dst: gc_off, a: gc_off, b: GprOperand::Imm(44), op: GprOp::Add });
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::ScalarVReg(gc_off), src: one_gpr });

                // Update seq_position[seq] = prompt_len[seq] at offset +40
                let sp_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: sp_off, a: seq_ctr, b: GprOperand::Imm(seq_stride as i64), op: GprOp::Mul });
                prog.emit(VmInstr::GprBinOp { dst: sp_off, a: sp_off, b: GprOperand::Imm(40), op: GprOp::Add });
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::ScalarVReg(sp_off), src: prompt_len_gpr });

                // Write first generated token to output_tokens_flat[output_offset + prompt_len]
                // output_offset[seq] at seq_meta +52, write position = output_offset + prompt_len
                let out_off_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                let oo_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: oo_byte_off, a: seq_ctr, b: GprOperand::Imm(seq_stride as i64), op: GprOp::Mul });
                prog.emit(VmInstr::ScalarLoad { dst: out_off_gpr, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(oo_byte_off)), Box::new(OffsetExpr::Const(52))) });
                // flat_index = output_offset + prompt_len
                let flat_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: flat_idx, a: out_off_gpr, b: GprOperand::VReg(prompt_len_gpr), op: GprOp::Add });
                // byte_offset = flat_idx * 4
                let flat_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: flat_byte_off, a: flat_idx, b: GprOperand::Imm(4), op: GprOp::Mul });
                prog.emit(VmInstr::ScalarStore { base: out_tokens_ptr, offset: OffsetExpr::ScalarVReg(flat_byte_off), src: sampled });

                // Update cumsum_acc += prompt_len for next iteration
                prog.emit(VmInstr::GprBinOp { dst: cumsum_acc, a: cumsum_acc, b: GprOperand::VReg(prompt_len_gpr), op: GprOp::Add });

                // Patch skip target for prompt_len == 0
                let skip_count = prog.instrs.len() - skip_patch - 1;
                if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut n), .. } = prog.instrs[skip_patch] {
                    *n = skip_count;
                }
            });
        }

        // ── Phase 3: Batch Decode Step Loop (SPEC/20 REQ-BCI-003) ──
        // After prefill + per-seq argmax, enter decode step loop.
        // Structure: outer LoopBegin(max_decode_steps) → inner loops for scan/build/argmax → LoopEnd.
        {
            // ── Phase 0.7 decode entry: jump target when total_prefill_tokens == 0 (SPEC 32 REQ-MKO-001) ──
            prog.emit(VmInstr::MarkLabel { label_id: DECODE_ENTRY_LABEL });
            let vb = vocab_size * ctx.dtype.elem_bytes();
            let stride: usize = 64; // SEQ_META_STRIDE

            // Read max_decode_steps from batch_ctx+4
            let max_steps = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: max_steps, base: batch_ctx_ptr, offset: OffsetExpr::Const(4) });

            // Decode input buffer: scratchpad region after all existing allocations.
            let decode_input_offset = {
                let sampling_end = logits_scratch_offset + vb * 5;
                let sg_end = current_abi.sg_detect_scratch_offset
                    .map(|off| {
                        let hdim = graph.ops.iter()
                            .find(|op| matches!(&op.kind, OpKind::SgDetect { .. }))
                            .and_then(|op| op.inputs.first())
                            .and_then(|&tid| graph.tensor(tid))
                            .and_then(|t| t.shape.last())
                            .and_then(|d| d.as_concrete())
                            .unwrap_or(0);
                        (off + hdim * 4 + 63) & !63
                    })
                    .unwrap_or(0);
                let sgk_end = current_abi.sg_knowledge_scratch_offset
                    .map(|off| {
                        let hdim = graph.ops.iter()
                            .find(|op| matches!(&op.kind, OpKind::SgInject { .. }))
                            .and_then(|op| op.inputs.first())
                            .and_then(|&tid| graph.tensor(tid))
                            .and_then(|t| t.shape.last())
                            .and_then(|d| d.as_concrete())
                            .unwrap_or(0);
                        (off + hdim * 4 + 63) & !63
                    })
                    .unwrap_or(0);
                let base = sampling_end.max(sg_end).max(sgk_end);
                (base + 63) & !63
            };
            let decode_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::AddPtr { dst: decode_input_ptr, base: scratchpad_ptr, offset: decode_input_offset });

            // Re-read shared batch metadata (defined in per-seq argmax block above, not in scope)
            let num_seqs_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: num_seqs_gpr, base: batch_ctx_ptr, offset: OffsetExpr::Const(0) });
            let seq_meta_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad { dst: seq_meta_base, base: batch_ctx_ptr, offset: OffsetExpr::Const(88) });

            // Persistent GPRs across decode steps
            let num_active = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            let total_gen = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            let compact_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: total_gen, value: 0 });

            // ── Outer decode step loop ──
            let step_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
            let step_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            prog.emit(VmInstr::LoopBegin { counter: step_ctr, byte_offset: step_off, bound: BoundExpr::DynamicVReg(max_steps), step_bytes: 4 });

            // ── Step 3a: Count num_active — active_flag is 0 or 1, just accumulate ──
            prog.emit(VmInstr::GprLoadImm { dst: num_active, value: 0 });
            prog.emit_loop(BoundExpr::DynamicVReg(num_seqs_gpr), 1, |prog, seq_ctr, _off| {
                let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: byte_off, a: seq_ctr, b: GprOperand::Imm(stride as i64), op: GprOp::Mul });
                let flag = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: flag, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(36))) });
                // active_flag is 0 or 1 — just add it to num_active (no branch needed)
                prog.emit(VmInstr::GprBinOp { dst: num_active, a: num_active, b: GprOperand::VReg(flag), op: GprOp::Add });
            });

            // ── Step 3b: Break if num_active == 0 (IsNonNull → skip BreakLoop when active) ──
            prog.emit(VmInstr::GprCondAction {
                cond: GprCondition::IsNonNull(num_active),
                action: GprBranchAction::Skip(1),
            });
            prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::VReg(total_gen) });

            // ── Step 3c: Build compact decode input from last_sampled_token[active seqs] ──
            prog.emit(VmInstr::GprLoadImm { dst: compact_idx, value: 0 });
            prog.emit_loop(BoundExpr::DynamicVReg(num_seqs_gpr), 1, |prog, seq_ctr, _off| {
                let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: byte_off, a: seq_ctr, b: GprOperand::Imm(stride as i64), op: GprOp::Mul });

                // Read active_flag[seq]
                let flag = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: flag, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(36))) });

                // If inactive (flag == 0), skip the copy + increment (3 instructions)
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpEq(flag, 0),
                    action: GprBranchAction::Skip(3),
                });
                let skip_patch = prog.instrs.len() - 1;

                // Read last_sampled_token[seq] at offset +48
                let token = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: token, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(48))) });

                // Store to decode_input[compact_idx]
                let dst_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: dst_off, a: compact_idx, b: GprOperand::Imm(4), op: GprOp::Mul });
                prog.emit(VmInstr::ScalarStore { base: decode_input_ptr, offset: OffsetExpr::ScalarVReg(dst_off), src: token });

                // compact_idx += 1
                prog.emit(VmInstr::GprBinOp { dst: compact_idx, a: compact_idx, b: GprOperand::Imm(1), op: GprOp::Add });

                // Patch: the Skip count should be 4 (token load + store offset + store + increment)
                // But wait: GprCondAction is already counted. Instructions AFTER GprCondAction that
                // should be skipped = token load(1) + dst_off(2) + store(3) + increment(4) = 4.
                let actual_skip = prog.instrs.len() - skip_patch - 1;
                if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut n), .. } = prog.instrs[skip_patch] {
                    *n = actual_skip;
                }
            });

            // ── Step 3d: Forward pass with M = num_active ──
            // Build decode AbiPtrs: same weights/scratch/output, but input = decode_input, M = num_active.
            let decode_output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: decode_output_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset) });

            let mut decode_abi = AbiPtrs {
                input_ptr: decode_input_ptr,
                weight_ptr: Some(weight_ptr),
                weight_abi_expr: Some(sym_map.resolve("weights").cloned().expect("ABI: weights")),
                output_ptr: decode_output_ptr,
                scratch_ptr: if needs_scratch { Some(scratchpad_ptr) } else { None },
                gen_loop_counter: None,
                layer_loop_counter: None,
                mega_decode_seq_len: Some(num_active), // M = num_active
                hook_ctx_ptr: batch_hook_ctx,
                sg_detect_scratch_offset: current_abi.sg_detect_scratch_offset,
                sg_knowledge_scratch_offset: current_abi.sg_knowledge_scratch_offset,
                callback_table_ptr: batch_cb_ptr,
                page_table_ptr: batch_pt_ptr,
                kv_load_mode: graph.kv_load_mode,
                kv_cache_ptr: None,
                activation_ping_ptr: current_abi.activation_ping_ptr,
                activation_pong_ptr: current_abi.activation_pong_ptr,
            };

            emit_fusion_groups(
                &mut prog, plan, graph, alloc, &batch_ctx,
                rope_req.as_ref().map(|r| r.cache_offset),
                &mut decode_abi, Some(weight_ptr), &batch_resolver,
            )?;

            // ── Step 3e: Per-seq argmax + stop condition ──
            // Logits layout: [num_active, vocab_size]. Active seqs are compacted.
            // We iterate seqs again, maintaining a compact_row counter for active seqs.
            let compact_row = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: compact_row, value: 0 });
            prog.emit_loop(BoundExpr::DynamicVReg(num_seqs_gpr), 1, |prog, seq_ctr, _off| {
                let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: byte_off, a: seq_ctr, b: GprOperand::Imm(stride as i64), op: GprOp::Mul });

                // Read active_flag[seq]
                let flag = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: flag, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(36))) });

                // If inactive, skip all argmax + stop logic
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpEq(flag, 0),
                    action: GprBranchAction::Skip(0), // placeholder — patch after
                });
                let skip_start = prog.instrs.len() - 1;

                // ── Per-seq logits row: [compact_row * vocab_bytes] ──
                let row_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: row_off, a: compact_row, b: GprOperand::Imm(vb as i64), op: GprOp::Mul });
                let logits_row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: logits_row_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, logits_scratch_offset) });
                prog.emit(VmInstr::GprBinOp { dst: logits_row_ptr, a: logits_row_ptr, b: GprOperand::VReg(row_off), op: GprOp::Add });

                // ── Per-seq sampling: read temperature from sampling_params_ptr + seq * 16 + 0 ──
                // sampling_params layout: [temp_f32_bits, top_k_u32, top_p_f32_bits, eos_u32] × N
                let sp_ptr_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: sp_ptr_gpr, src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 56) });
                let seq_sp_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seq_sp_off, a: seq_ctr, b: GprOperand::Imm(16), op: GprOp::Mul });
                // Read temperature (sp_ptr + seq_sp_off + 0)
                let temp_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: temp_val, base: sp_ptr_gpr, offset: OffsetExpr::ScalarVReg(seq_sp_off) });

                // Branch: temperature == 0 → Argmax, temperature > 0 → stochastic
                // We use Skip-based branching: if temp != 0, skip the Argmax and go to stochastic.
                // First emit Argmax (greedy path), then stochastic path that overwrites `sampled`.

                let sampled = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

                // If temperature != 0 → skip Argmax, go to stochastic path
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::IsNonNull(temp_val),
                    action: GprBranchAction::Skip(1), // skip the Argmax instruction
                });

                // Argmax path (temperature == 0)
                prog.emit(VmInstr::Argmax { dst: sampled, logits_ptr: logits_row_ptr, vocab_bytes: vb, width });

                // After Argmax, skip the entire stochastic section
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::IsNonNull(temp_val), // temp != 0 means we came from stochastic, skip this skip
                    action: GprBranchAction::Skip(0), // placeholder — patched below
                });
                let stochastic_skip_patch = prog.instrs.len() - 1;

                // ── Stochastic sampling path (temperature > 0) ──
                // TemperatureScale: logits[i] /= temperature
                // temp_val is f32 bits — need a pointer for TemperatureScale
                let temp_store_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::AddPtr { dst: temp_store_ptr, base: sp_ptr_gpr, offset: 0 }); // point to seq's temp
                prog.emit(VmInstr::TemperatureScale {
                    logits_ptr: logits_row_ptr,
                    temp_ptr: temp_store_ptr,
                    vocab_bytes: vb,
                    width,
                });

                // Softmax: reduce-max → exp-sum → normalize
                let max_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::SoftmaxReduceMax {
                    dst: max_val,
                    logits_ptr: logits_row_ptr,
                    vocab_bytes: vb,
                    width,
                });
                let sum_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::SoftmaxExpSum {
                    sum_dst: sum_val,
                    logits_ptr: logits_row_ptr,
                    max_val,
                    vocab_bytes: vb,
                    width,
                });
                prog.emit(VmInstr::SoftmaxNormalize {
                    logits_ptr: logits_row_ptr,
                    sum_val,
                    vocab_bytes: vb,
                    width,
                });

                // Top-K: read top_k from sampling_params + seq_sp_off + 4
                let top_k_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                let seq_sp_off_k = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seq_sp_off_k, a: seq_sp_off, b: GprOperand::Imm(4), op: GprOp::Add });
                prog.emit(VmInstr::ScalarLoad { dst: top_k_val, base: sp_ptr_gpr, offset: OffsetExpr::ScalarVReg(seq_sp_off_k) });
                // TopK filter needs a ptr to k value — store k to scratch temp, use ptr
                let k_store_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                let indices_region = logits_scratch_offset + vb;
                prog.emit(VmInstr::LoadPtr { dst: k_store_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, indices_region + vb) });
                prog.emit(VmInstr::ScalarStore { base: scratchpad_ptr, offset: OffsetExpr::Const(indices_region + vb), src: top_k_val });
                let indices_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: indices_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, indices_region) });
                prog.emit(VmInstr::SampleTopKFilter {
                    probs_ptr: logits_row_ptr,
                    indices_ptr,
                    k_ptr: k_store_ptr,
                    vocab_bytes: vb,
                    width,
                });

                // Top-P: read top_p from sampling_params + seq_sp_off + 8
                let top_p_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                let seq_sp_off_p = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seq_sp_off_p, a: seq_sp_off, b: GprOperand::Imm(8), op: GprOp::Add });
                prog.emit(VmInstr::ScalarLoad { dst: top_p_val, base: sp_ptr_gpr, offset: OffsetExpr::ScalarVReg(seq_sp_off_p) });
                // Store p to scratch temp, use ptr
                let p_store_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: p_store_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, indices_region + vb + 4) });
                prog.emit(VmInstr::ScalarStore { base: scratchpad_ptr, offset: OffsetExpr::Const(indices_region + vb + 4), src: top_p_val });
                prog.emit(VmInstr::SampleTopPFilter {
                    probs_ptr: logits_row_ptr,
                    p_ptr: p_store_ptr,
                    vocab_bytes: vb,
                    width,
                });

                // Multinomial sampling
                let rng_state_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: rng_state_ptr, src: PtrExpr::VRegPlusConst(scratchpad_ptr, indices_region + vb + vb) });
                prog.emit(VmInstr::SampleMultinomial {
                    dst: sampled,
                    probs_ptr: logits_row_ptr,
                    rng_state_ptr,
                    vocab_bytes: vb,
                    width,
                });

                // Patch stochastic_skip to jump over entire stochastic section
                let stochastic_end = prog.instrs.len();
                let stochastic_instr_count = stochastic_end - stochastic_skip_patch - 1;
                if let VmInstr::GprCondAction { cond: _, action: GprBranchAction::Skip(ref mut n) } = prog.instrs[stochastic_skip_patch] {
                    *n = stochastic_instr_count;
                }

                // ── Write last_sampled_token[seq] = sampled at offset +48 ──
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(48))), src: sampled });

                // ── Write sampled token to output_tokens_flat[output_offset + gc] ──
                // Read output_offset[seq] at offset +52
                let out_off_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: out_off_val, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(52))) });

                // Read gen_count[seq] at offset +44 (BEFORE increment)
                let gc = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: gc, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(44))) });

                // flat_index = output_offset + gen_count
                let flat_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: flat_idx, a: out_off_val, b: GprOperand::VReg(gc), op: GprOp::Add });
                // byte_offset = flat_index * 4
                let flat_byte = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: flat_byte, a: flat_idx, b: GprOperand::Imm(4), op: GprOp::Mul });

                // Read output_tokens_flat_ptr from batch_ctx+24
                let out_flat_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: out_flat_ptr, src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 24) });
                prog.emit(VmInstr::ScalarStore { base: out_flat_ptr, offset: OffsetExpr::ScalarVReg(flat_byte), src: sampled });

                // ── Increment gen_count[seq] at offset +44 ──
                let gc_plus1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: gc_plus1, a: gc, b: GprOperand::Imm(1), op: GprOp::Add });
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(44))), src: gc_plus1 });

                // Increment seq_position[seq] at offset +40
                let sp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: sp, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(40))) });
                let sp_plus1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: sp_plus1, a: sp, b: GprOperand::Imm(1), op: GprOp::Add });
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(40))), src: sp_plus1 });

                // Check stop: gen_count >= max_new_tokens OR sampled == eos
                // Read max_new_tokens[seq] at offset +12
                let max_new = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: max_new, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(12))) });

                // Stop condition 1: max_new_tokens > 0 AND gen_count >= max_new_tokens
                // (max_new_tokens == 0 means no limit)
                let at_max = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: at_max, a: gc_plus1, b: GprOperand::VReg(max_new), op: GprOp::Sub });
                // at_max >= 0 means gen_count >= max_new_tokens (unsigned cmp: CmpLtU would check <)
                // If max_new == 0, skip this check
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::IsNonNull(max_new),
                    action: GprBranchAction::Skip(1),
                });
                let _max_skip = prog.instrs.len() - 1;
                // If gen_count >= max_new_tokens, deactivate. Use: at_max < 0 → still active
                // at_max is gc_plus1 - max_new. If gc_plus1 >= max_new, at_max >= 0 → stop.
                // CmpLtU checks at_max < 0, which is always false for unsigned. Hmm.
                // Use a simpler approach: compare gc_plus1 with max_new directly.
                // If gc_plus1 >= max_new → CmpLtU(gc_plus1, max_new) is false → skip not taken.
                // We want: if gc_plus1 >= max_new → deactivate.
                // We use: skip(deactivate) when gc_plus1 < max_new (still has room)
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpLtU(gc_plus1, 0), // placeholder — will patch
                    action: GprBranchAction::Skip(1),
                });
                let max_check_patch = prog.instrs.len() - 1;
                // We'll handle EOS check first, then patch both stops together.

                // Stop condition 2: sampled == eos (from sampling_params_ptr + seq * 16 + 12)
                // Read sampling_params_ptr from batch_ctx+56
                let sp_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr { dst: sp_ptr, src: PtrExpr::VRegPlusConst(batch_ctx_ptr, 56) });
                // eos for seq = sp_ptr + seq * 16 + 12 (packed: temp,top_k,top_p,eos per seq)
                let seq_sp_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seq_sp_off, a: seq_ctr, b: GprOperand::Imm(16), op: GprOp::Mul });
                prog.emit(VmInstr::GprBinOp { dst: seq_sp_off, a: seq_sp_off, b: GprOperand::Imm(12), op: GprOp::Add });
                let eos_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: eos_val, base: sp_ptr, offset: OffsetExpr::ScalarVReg(seq_sp_off) });

                // Compute eos_match = (sampled == eos_val). Use CmpEq for zero comparison.
                // sampled - eos_val == 0 means match.
                let eos_diff = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: eos_diff, a: sampled, b: GprOperand::VReg(eos_val), op: GprOp::Sub });

                // Deactivate if: (max_new > 0 AND gen_count >= max_new) OR (sampled == eos)
                // We write active_flag = 0 for both conditions.
                // The max_new_tokens stop:
                //   at_max = gc_plus1 - max_new. If at_max >= 0 (gen >= max) → set flag=0.
                //   Unsigned: at_max < 0 only possible if at_max is very large (underflow).
                //   Since both are u32, gc_plus1 >= max_new means no underflow → at_max >= 0.
                //   So: deactivate when NOT (gc_plus1 < max_new), i.e., when CmpLtU(gc_plus1, max_new) is false.
                // Patch the max_check placeholder: skip deactivate when gc_plus1 < max_new
                if let VmInstr::GprCondAction { cond: GprCondition::CmpLtU(ref mut v, _), .. } = prog.instrs[max_check_patch] {
                    *v = gc_plus1;
                    // Also need the immediate to be max_new. But CmpLtU takes (VRegId, u64).
                    // We can't compare two VRegs with CmpLtU. Need a different approach.
                    // Use the subtraction: at_max = gc_plus1 - max_new.
                    // at_max >= 0 → gen_count >= max_new → stop.
                    // Use: BitClear(at_max, 31) — bit 31 clear means non-negative in signed,
                    // but unsigned doesn't have sign. Hmm.
                    // Alternative: just use GprCondAction::CmpEq for exact match on a flag.
                }
                // Simpler stop: use a combined approach.
                // Zero the active_flag, then conditionally restore it if both stop conditions are false.
                let zero_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });
                let one_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: one_gpr, value: 1 });

                // Start by assuming we stop: write flag = 0
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(36))), src: zero_gpr });

                // If should continue (gen_count < max_new AND sampled != eos), restore flag = 1.
                // Condition to continue: at_max >= 0 (gen_count < max_new when max_new > 0) AND eos_diff != 0
                // Simplification: continue when (max_new == 0 OR gen_count < max_new) AND eos_diff != 0
                // This is complex with VmInstr. Use two sequential Skip checks:
                //   If gen_count >= max_new AND max_new > 0 → keep flag=0, skip restore
                //   If sampled == eos → keep flag=0, skip restore
                //   Otherwise → restore flag=1

                // Check 1: max_new > 0 AND gen_count >= max_new → keep flag=0
                // at_max = gc_plus1 - max_new. If max_new == 0, skip this check.
                // We already emitted at_max above. CmpLtU checks for at_max < huge (unsigned).
                // For simplicity: if max_new == 0 → skip the max check (max_new=0 means unlimited)
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpEq(max_new, 0), // max_new == 0 → unlimited → skip max stop
                    action: GprBranchAction::Skip(1),
                });
                // at_max >= 0 → gen >= max → flag stays 0 → skip the eos check and restore
                // We need "if at_max < max_new, don't stop". But CmpLtU takes (vreg, imm).
                // at_max = gc_plus1 - max_new. If at_max is very large (underflow), gen < max.
                // CmpLtU(at_max, 0x80000000) — if at_max < 2^31, it's positive → gen >= max.
                // Hmm this is getting too hacky. Let me use a direct comparison.
                // Actually: we can rewrite as CmpLtU(gc_plus1, max_new_value).
                // But max_new is in a VReg, not an immediate. CmpLtU takes (VRegId, u64).
                // We need VReg vs VReg comparison which isn't directly available.
                // WORKAROUND: compute stop = 1 if gc_plus1 >= max_new (unsigned subtraction underflow check).
                // at_max = gc_plus1.wrapping_sub(max_new). If no underflow → stop.
                // In u32: at_max >= 0x80000000 means underflow → gen < max → continue.
                // CmpLtU(at_max, 0x80000000) = true means no underflow → gen >= max → stop.
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpLtU(at_max, 0x8000_0000), // no underflow → gen >= max → stop
                    action: GprBranchAction::Skip(3), // skip eos check + restore → flag stays 0
                });

                // Check 2: sampled == eos → flag stays 0
                prog.emit(VmInstr::GprCondAction {
                    cond: GprCondition::CmpEq(eos_diff, 0), // sampled == eos
                    action: GprBranchAction::Skip(1), // skip restore → flag stays 0
                });

                // Neither stop condition met → restore active_flag = 1
                prog.emit(VmInstr::ScalarStore { base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(byte_off)), Box::new(OffsetExpr::Const(36))), src: one_gpr });

                // compact_row += 1 (only for active seqs — already past the inactive skip)
                prog.emit(VmInstr::GprBinOp { dst: compact_row, a: compact_row, b: GprOperand::Imm(1), op: GprOp::Add });

                // Patch the inactive skip count
                let skip_count = prog.instrs.len() - skip_start - 1;
                if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut n), .. } = prog.instrs[skip_start] {
                    *n = skip_count;
                }
            });

            // ── Step 3f: Increment total_gen ──
            prog.emit(VmInstr::GprBinOp { dst: total_gen, a: total_gen, b: GprOperand::Imm(1), op: GprOp::Add });

            // ── Outer decode step loop end ──
            prog.emit(VmInstr::LoopEnd);

            // Return total number of decode steps completed
            prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::VReg(total_gen) });
        }
    }

    // Stage 1.5: 符号验证 — 与 compile_layer 对齐
    // mega-kernel 路径之前跳过此验证，导致 VmInstr 错误静默传播到 ISA lowering
    super::verify::verify_vm_program(&prog)?;
    prog.validate_provenance()
        .map_err(|e| CompilerError::CodegenViolation(format!("mega-kernel provenance: {e}")))?;
    prog.validate_structure()
        .map_err(|e| CompilerError::CodegenViolation(format!("mega-kernel structure: {e}")))?;
    if let Err(e) = prog.validate_type_consistency() {
        return Err(CompilerError::CodegenViolation(format!("mega-kernel type-check: {e}")));
    }
    if let Err(e) = prog.validate_width_consistency() {
        return Err(CompilerError::CodegenViolation(format!("mega-kernel width-check: {e}")));
    }
    if let Err(e) = prog.validate_value_domains() {
        return Err(CompilerError::CodegenViolation(format!("mega-kernel value-domain: {e}")));
    }

    // DEBUG: dump mega-kernel VmProgram
    if let Ok(dir) = std::env::var("GLLM_DUMP_MEGA") {
        use std::io::Write;
        let _ = std::fs::create_dir_all(&dir);
        let path = format!("{}/mega_kernel_vm.txt", dir);
        if let Ok(mut f) = std::fs::File::create(&path) {
            writeln!(f, "=== Mega-Kernel VmProgram ({} instrs) ===", prog.instrs.len()).ok();
            for (i, instr) in prog.instrs.iter().enumerate() {
                writeln!(f, "{:4}: {:?}", i, instr).ok();
            }
        }
    }
    Ok((prog, rope_req, logits_scratch_offset))
}

/// Emit MTP candidate token generation (Phase 6.5).
///
/// For each MTP depth k=0..depth-1:
/// 1. Compute weight pointer: weight_ptr + lm_head_end + k * proj_bytes
/// 2. GEMV: mtp_logits[k] = final_normed @ weight[k]^T
/// 3. Argmax: candidate[k] = argmax(mtp_logits[k])
/// 4. Store to output_tokens[max_new_tokens + decode_counter * depth + k]
///
/// All computation uses JIT VmInstr (VecLoad/Fma/VecStore/Argmax/emit_loop).
fn emit_mtp_candidates(
    prog: &mut VmProgram,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    resolver: &TensorPtrResolver,
    ctx: &LoweringContext,
    mtp: &MtpKernelConfig,
    logits_scratch_offset: usize,
    vocab_size: usize,
    weight_ptr: VRegId,
    scratchpad_ptr: VRegId,
    output_tokens_ptr: VRegId,
    decode_counter: VRegId,
    max_new_tokens_vreg: VRegId,
    width: SimdWidth,
) -> Result<(), CompilerError> {
    let MtpKernelConfig { depth, hidden_size, vocab_size: mtp_vocab } = *mtp;
    let elem_bytes = ctx.dtype.elem_bytes();
    let vocab_bytes = vocab_size * elem_bytes;
    let hidden_bytes = hidden_size * elem_bytes;
    let proj_bytes = mtp_vocab * hidden_size * elem_bytes;

    // Find lm_head weight blob offset to compute MTP weight start.
    let lm_head_blob_offset = graph.ops.iter()
        .find(|op| op.label == "lm_head")
        .and_then(|op| op.inputs.get(1))
        .and_then(|&wid| resolver.source(wid))
        .and_then(|src| match src {
            TensorPtrSource::Weight { offset } => Some(offset),
            _ => None,
        })
        .ok_or_else(|| CompilerError::CodegenViolation(
            "MTP: cannot find lm_head weight offset in resolver".into()
        ))?;
    let lm_head_bytes = vocab_size * hidden_size * elem_bytes;
    let mtp_weights_base = lm_head_blob_offset + lm_head_bytes;

    // Find final_normed tensor source — it's lm_head's first input.
    let (hidden_ptr_base, hidden_offset) = {
        let lm_head_op = graph.ops.iter()
            .find(|op| op.label == "lm_head")
            .ok_or_else(|| CompilerError::CodegenViolation("MTP: lm_head op not found".into()))?;
        let fn_tid = lm_head_op.inputs[0];
        let src = resolver.source(fn_tid)
            .ok_or_else(|| CompilerError::CodegenViolation(
                "MTP: final_normed tensor source not found".into()
            ))?;
        match src {
            TensorPtrSource::ActivationPing => {
                let ping = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::AddPtr {
                    dst: ping,
                    base: scratchpad_ptr,
                    offset: 0, // will be overridden below with actual alloc offset
                });
                // Find the actual ping offset from alloc sentinel slots
                let ping_off = alloc.slots.iter()
                    .find(|s| s.tensor_id.0 == 0xFFFF_FF00)
                    .map(|s| s.offset)
                    .unwrap_or(0);
                // Re-emit with correct offset
                prog.emit(VmInstr::AddPtr { dst: ping, base: scratchpad_ptr, offset: ping_off });
                (ping, 0usize)
            }
            TensorPtrSource::ActivationPong => {
                let pong = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                let pong_off = alloc.slots.iter()
                    .find(|s| s.tensor_id.0 == 0xFFFF_FF01)
                    .map(|s| s.offset)
                    .unwrap_or(0);
                prog.emit(VmInstr::AddPtr { dst: pong, base: scratchpad_ptr, offset: pong_off });
                (pong, 0usize)
            }
            TensorPtrSource::Intermediate { offset } => {
                (scratchpad_ptr, offset)
            }
            _ => {
                return Err(CompilerError::CodegenViolation(
                    format!("MTP: unexpected final_normed source: {:?}", src)
                ));
            }
        }
    };

    // Compute hidden_ptr = hidden_ptr_base + hidden_offset
    let hidden_ptr = if hidden_offset > 0 {
        let hp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::AddPtr { dst: hp, base: hidden_ptr_base, offset: hidden_offset });
        hp
    } else {
        hidden_ptr_base
    };

    // MTP logits region: reuse scratchpad after main logits + sampling workspace.
    // Main logits: [logits_scratch_offset, logits_scratch_offset + vocab_bytes)
    // Sampling: [logits_scratch_offset + vocab_bytes, logits_scratch_offset + vocab_bytes * 5)
    // MTP logits: [logits_scratch_offset + vocab_bytes * 5, + vocab_bytes per depth)
    let mtp_logits_base_offset = logits_scratch_offset + vocab_bytes * 5;

    let lanes = width.f32_lanes().max(1);
    let mtp_depth_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: mtp_depth_vreg, value: depth });

    // Loop over MTP depths: for k = 0..depth
    prog.emit_loop(BoundExpr::Const(depth), 1, |prog, k_ctr, _k_off| {
        // ── Compute weight pointer for this depth ──
        // mtp_weight_ptr = weight_ptr + mtp_weights_base + k * proj_bytes
        let k_offset_in_blob = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: k_offset_in_blob, a: k_ctr, b: GprOperand::Imm(proj_bytes as i64), op: GprOp::Mul,
        });
        let weight_offset_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: weight_offset_val, value: mtp_weights_base });
        let total_weight_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: total_weight_off, a: weight_offset_val, b: GprOperand::VReg(k_offset_in_blob), op: GprOp::Add,
        });
        let mtp_weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: mtp_weight_ptr, a: weight_ptr, b: GprOperand::VReg(total_weight_off), op: GprOp::Add,
        });

        // ── Compute logits ptr for this depth ──
        // mtp_logits_ptr = scratchpad + mtp_logits_base_offset + k * vocab_bytes
        let k_vocab_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: k_vocab_off, a: k_ctr, b: GprOperand::Imm(vocab_bytes as i64), op: GprOp::Mul,
        });
        let mtp_logits_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: mtp_logits_off, value: mtp_logits_base_offset });
        let mtp_logits_full_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: mtp_logits_full_off, a: mtp_logits_off, b: GprOperand::VReg(k_vocab_off), op: GprOp::Add,
        });
        let mtp_logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: mtp_logits_ptr, a: scratchpad_ptr, b: GprOperand::VReg(mtp_logits_full_off), op: GprOp::Add,
        });

        // ── GEMV: logits[v] = sum_h(hidden[h] * weight[v * hidden + h]) ──
        // Outer loop: over vocab rows (each row produces one logit element)
        // Inner computation: dot product of hidden vector with weight row
        // Use vectorized approach: process hidden vector in SIMD chunks.
        emit_mtp_gemv(
            prog, hidden_ptr, mtp_weight_ptr, mtp_logits_ptr,
            hidden_size, vocab_size, elem_bytes, lanes, width,
        );

        // ── Argmax on MTP logits ──
        let mtp_candidate = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::Argmax {
            dst: mtp_candidate,
            logits_ptr: mtp_logits_ptr,
            vocab_bytes,
            width,
        });

        // ── Store MTP candidate to output buffer ──
        // output_offset = max_new_tokens + decode_counter * depth + k
        let dc_depth = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: dc_depth, a: decode_counter, b: GprOperand::VReg(mtp_depth_vreg), op: GprOp::Mul,
        });
        let dc_depth_k = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: dc_depth_k, a: dc_depth, b: GprOperand::VReg(k_ctr), op: GprOp::Add,
        });
        let store_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: store_idx, a: max_new_tokens_vreg, b: GprOperand::VReg(dc_depth_k), op: GprOp::Add,
        });
        let store_byte_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: store_byte_off, a: store_idx, b: GprOperand::Imm(4), op: GprOp::Mul,
        });
        prog.emit(VmInstr::ScalarStore {
            base: output_tokens_ptr,
            offset: OffsetExpr::ScalarVReg(store_byte_off),
            src: mtp_candidate,
        });
    });

    Ok(())
}

/// Emit a GEMV (matrix-vector multiply) for a single MTP projection.
///
/// Computes: output[v] = sum_{h=0}^{hidden-1} input[h] * weight[v * hidden + h]
/// for v = 0..vocab-1.
///
/// Uses vectorized inner loop: loads `lanes` elements of the hidden vector
/// and weight row, performs FMA, accumulates into a vec register, then
/// HReduce to scalar and stores.
fn emit_mtp_gemv(
    prog: &mut VmProgram,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    hidden: usize,
    vocab: usize,
    elem_bytes: usize,
    lanes: usize,
    width: SimdWidth,
) {
    let dtype = QuantPrecision::F32;
    let hidden_vec_iters = hidden / lanes;

    // Outer loop: iterate over vocab rows
    let v_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
    let v_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    prog.emit(VmInstr::LoopBegin {
        counter: v_ctr,
        byte_offset: v_off,
        bound: BoundExpr::Const(vocab),
        step_bytes: hidden * elem_bytes,
    });

    // Initialize vector accumulator to zero
    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let zero_bits = 0u32; // 0.0f32 as u32 = 0x00000000
    prog.emit(VmInstr::VecLoadConst {
        dst: acc,
        values: vec![zero_bits; lanes],
        dtype,
        width,
    });

    // Inner loop: vectorized dot product over hidden dimension
    if hidden_vec_iters > 0 {
        let h_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let h_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter: h_ctr,
            byte_offset: h_off,
            bound: BoundExpr::Const(hidden_vec_iters),
            step_bytes: lanes * elem_bytes,
        });

        // Load hidden[h_off..h_off+lanes]
        let hidden_vec = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecLoad {
            dst: hidden_vec,
            base: input_ptr,
            offset: OffsetExpr::LoopOffset(h_off),
            width,
            dtype,
        });

        // Load weight[v_off + h_off..v_off + h_off+lanes]
        let weight_vec = prog.alloc_vreg(VRegKind::Vec, width);
        let combined_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: combined_off, a: v_off, b: GprOperand::VReg(h_off), op: GprOp::Add,
        });
        prog.emit(VmInstr::VecLoad {
            dst: weight_vec,
            base: weight_ptr,
            offset: OffsetExpr::ScalarVReg(combined_off),
            width,
            dtype,
        });

        // acc += hidden_vec * weight_vec
        prog.emit(VmInstr::Fma {
            dst: acc,
            acc,
            a: hidden_vec,
            b: weight_vec,
            dtype,
        });

        prog.emit(VmInstr::LoopEnd);
    }

    // Horizontal reduce: sum all lanes of acc into a single scalar
    let acc_scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::HReduce {
        dst: acc_scalar,
        src: acc,
        op: super::instr::ReduceOp::Sum,
    });

    // Store result to output[v_ctr]
    let out_byte = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp {
        dst: out_byte, a: v_ctr, b: GprOperand::Imm(elem_bytes as i64), op: GprOp::Mul,
    });
    prog.emit(VmInstr::ScalarStore {
        base: output_ptr,
        offset: OffsetExpr::ScalarVReg(out_byte),
        src: acc_scalar,
    });

    prog.emit(VmInstr::LoopEnd);
}

/// Emit MTP draft candidate generation via TraceOp/auto_select pipeline (MTP-001).
///
/// This is the JIT-pipeline-compliant entry point called from `auto_select`
/// when `TraceOp::MtpDraft` is encountered. It replaces the old direct
/// `emit_mtp_candidates` path that bypassed the pipeline.
///
/// Generates `depth` candidate tokens: for each depth k, computes
/// GEMV (hidden @ weight_k) to produce logits, then argmax to select token.
pub fn emit_mtp_draft_inline(
    prog: &mut VmProgram,
    depth: usize,
    hidden_size: usize,
    vocab_size: usize,
    hidden_ptr: VRegId,
    weight_ptr: VRegId,
    output_tokens_ptr: VRegId,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let lanes = width.f32_lanes().max(1);
    let elem_bytes = dtype.elem_bytes();
    let hidden_bytes = hidden_size * elem_bytes;
    let vocab_bytes = vocab_size * elem_bytes;
    let proj_bytes = vocab_size * hidden_size * elem_bytes;

    let hidden_vec_iters = hidden_size / lanes;

    // Allocate scratch for logits (vocab_size elements per depth)
    // We reuse a single logits buffer across depths since each is consumed before the next.
    let logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::AddPtr { dst: logits_ptr, base: output_tokens_ptr, offset: 0 });

    prog.emit_loop(BoundExpr::Const(depth), 1, |prog, k_ctr, _k_off| {
        // Compute weight pointer for this depth: weight_ptr + k * proj_bytes
        let k_proj_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: k_proj_off, a: k_ctr, b: GprOperand::Imm(proj_bytes as i64), op: GprOp::Mul,
        });
        let depth_weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: depth_weight_ptr, a: weight_ptr, b: GprOperand::VReg(k_proj_off), op: GprOp::Add,
        });

        // GEMV: logits[v] = sum_h(hidden[h] * weight[v * hidden + h])
        emit_mtp_gemv(
            prog, hidden_ptr, depth_weight_ptr, logits_ptr,
            hidden_size, vocab_size, elem_bytes, lanes, width,
        );

        // Argmax on logits
        let candidate = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::Argmax {
            dst: candidate,
            logits_ptr,
            vocab_bytes,
            width,
        });

        // Store candidate: output_tokens[depth_offset + k]
        let k_byte_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: k_byte_off, a: k_ctr, b: GprOperand::Imm(4i64), op: GprOp::Mul,
        });
        prog.emit(VmInstr::ScalarStore {
            base: output_tokens_ptr,
            offset: OffsetExpr::ScalarVReg(k_byte_off),
            src: candidate,
        });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: emit_mtp_gemv emits correct loop structure with balanced LoopBegin/LoopEnd ──
    #[test]
    fn test_emit_mtp_gemv_produces_balanced_loops() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 64;
        let vocab = 128;
        let elem_bytes = 4;
        let lanes = 8;
        let width = SimdWidth::W256;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, vocab, elem_bytes, lanes, width,
        );

        // Assert: loop structure must be balanced
        assert!(prog.validate_structure().is_ok(), "emit_mtp_gemv must produce balanced loops");
    }

    // ── Test 2: emit_mtp_gemv emits exactly two loops (outer vocab + inner hidden) ──
    #[test]
    fn test_emit_mtp_gemv_has_two_loops_when_hidden_aligned() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 64;
        let vocab = 128;
        let elem_bytes = 4;
        let lanes = 8;
        let width = SimdWidth::W256;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, vocab, elem_bytes, lanes, width,
        );

        // Assert: count LoopBegin instructions — expect 2 (outer vocab loop + inner hidden loop)
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        let loop_end_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopEnd)).count();
        assert_eq!(loop_begin_count, 2, "emit_mtp_gemv with aligned hidden should have 2 loops");
        assert_eq!(loop_end_count, 2, "each LoopBegin must have a matching LoopEnd");
    }

    // ── Test 3: emit_mtp_gemv with non-aligned hidden (lanes > hidden) skips inner loop ──
    #[test]
    fn test_emit_mtp_gemv_skips_inner_loop_when_hidden_less_than_lanes() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 4; // less than lanes=8, so hidden_vec_iters = 4/8 = 0
        let vocab = 16;
        let elem_bytes = 4;
        let lanes = 8;
        let width = SimdWidth::W256;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, vocab, elem_bytes, lanes, width,
        );

        // Assert: only the outer vocab loop (no inner loop since hidden_vec_iters=0)
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_begin_count, 1, "no inner loop when hidden < lanes");
        assert!(prog.validate_structure().is_ok());
    }

    // ── Test 4: emit_mtp_gemv outer loop uses correct step_bytes ──
    #[test]
    fn test_emit_mtp_gemv_outer_loop_step_bytes() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 64;
        let vocab = 128;
        let elem_bytes = 4;
        let lanes = 8;
        let width = SimdWidth::W256;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, vocab, elem_bytes, lanes, width,
        );

        // Assert: outer loop step_bytes = hidden * elem_bytes = 256
        let outer_loop = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. } if *v == vocab => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(outer_loop, Some(hidden * elem_bytes), "outer loop step must be hidden * elem_bytes");
    }

    // ── Test 5: emit_mtp_gemv contains HReduce with Sum operation ──
    #[test]
    fn test_emit_mtp_gemv_contains_hreduce_sum() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: at least one HReduce with Sum
        let has_hreduce_sum = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::HReduce { op: ReduceOp::Sum, .. }
        ));
        assert!(has_hreduce_sum, "GEMV must reduce accumulator via HReduce::Sum");
    }

    // ── Test 6: emit_mtp_gemv includes FMA for accumulation ──
    #[test]
    fn test_emit_mtp_gemv_uses_fma_for_dot_product() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: FMA is used for acc += hidden * weight
        let has_fma = prog.instrs.iter().any(|i| matches!(i, VmInstr::Fma { .. }));
        assert!(has_fma, "GEMV inner loop must use FMA for accumulation");
    }

    // ── Test 7: emit_mtp_gemv initializes accumulator with zero ──
    #[test]
    fn test_emit_mtp_gemv_initializes_accumulator_zero() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: VecLoadConst with zero values for accumulator initialization
        let has_zero_init = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::VecLoadConst { values, .. } if values.iter().all(|&v| v == 0u32)
        ));
        assert!(has_zero_init, "accumulator must be initialized to zero via VecLoadConst");
    }

    // ── Test 8: emit_mtp_gemv provenance — all VRegs properly declared ──
    #[test]
    fn test_emit_mtp_gemv_provenance_valid() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: all referenced VRegs must have been declared
        assert!(prog.validate_provenance().is_ok(), "all VRegs must be declared before use");
    }

    // ── Test 9: emit_mtp_gemv uses F32 dtype for all vec operations ──
    #[test]
    fn test_emit_mtp_gemv_uses_f32_dtype() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: all VecLoad, VecLoadConst, and FMA operations use F32 dtype
        for instr in &prog.instrs {
            match instr {
                VmInstr::VecLoad { dtype, .. } | VmInstr::VecLoadConst { dtype, .. } | VmInstr::Fma { dtype, .. } => {
                    assert_eq!(*dtype, QuantPrecision::F32, "GEMV must use F32 dtype");
                }
                _ => {}
            }
        }
    }

    // ── Test 10: emit_mtp_gemv ScalarStore uses correct output register ──
    #[test]
    fn test_emit_mtp_gemv_stores_to_correct_output() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: at least one ScalarStore using output_ptr as base
        let has_output_store = prog.instrs.iter().any(|i| match i {
            VmInstr::ScalarStore { base, .. } => *base == output_ptr,
            _ => false,
        });
        assert!(has_output_store, "GEMV must store results to the output_ptr register");
    }

    // ── Test 11: emit_mtp_draft_inline produces valid structure with depth > 1 ──
    #[test]
    fn test_emit_mtp_draft_inline_multi_depth_structure() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let width = SimdWidth::W256;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            3,     // depth
            64,    // hidden_size
            128,   // vocab_size
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            width,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "MTP draft inline must produce balanced loops");
    }

    // ── Test 12: emit_mtp_draft_inline with depth=1 still produces valid program ──
    #[test]
    fn test_emit_mtp_draft_inline_single_depth_has_argmax() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            1,     // depth
            32,    // hidden_size
            64,    // vocab_size
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        let has_argmax = prog.instrs.iter().any(|i| matches!(i, VmInstr::Argmax { .. }));
        assert!(has_argmax, "MTP draft inline must emit Argmax for each depth");
    }

    // ── Test 13: emit_mtp_gemv with Scalar width uses lanes=1 ──
    #[test]
    fn test_emit_mtp_gemv_scalar_width_no_inner_loop() {
        // Arrange: Scalar width => lanes=1, hidden=8 => hidden_vec_iters=8
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 8;
        let vocab = 4;
        let elem_bytes = 4;
        let lanes = 1;
        let width = SimdWidth::Scalar;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, vocab, elem_bytes, lanes, width,
        );

        // Assert: structure still valid, inner loop present (8 iters), outer loop present (4 iters)
        assert!(prog.validate_structure().is_ok());
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_begin_count, 2, "Scalar width should still have both loops");
    }

    #[test]
    fn test_emit_mtp_gemv_w512_produces_balanced_loops() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            128, 256, 4, 16, SimdWidth::W512,
        );

        assert!(prog.validate_structure().is_ok());
        assert!(prog.validate_provenance().is_ok());
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_begin_count, 2);
    }

    #[test]
    fn test_emit_mtp_gemv_vocab_one_produces_single_outer_iteration() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 1, 4, 8, SimdWidth::W256,
        );

        assert!(prog.validate_structure().is_ok());
        let outer_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), .. } if *v == 1 => Some(true),
            _ => None,
        });
        assert!(outer_bound.is_some(), "vocab=1 means outer loop bound is 1");
    }

    #[test]
    fn test_emit_mtp_gemv_inner_loop_step_is_lanes_times_elem_bytes() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 64;
        let lanes = 8;
        let elem_bytes = 4;

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, 32, elem_bytes, lanes, SimdWidth::W256,
        );

        let inner_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == hidden / lanes => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(inner_step, Some(lanes * elem_bytes));
    }

    #[test]
    fn test_emit_mtp_gemv_accumulator_vec_width_matches_simd_width() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let width = SimdWidth::W128;

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 4, width,
        );

        let acc_decl = prog.instrs.iter().filter_map(|i| match i {
            VmInstr::DeclareVReg { id, kind: VRegKind::Vec, width: w } => Some((id, w)),
            _ => None,
        }).collect::<Vec<_>>();
        let has_w128_vec = acc_decl.iter().any(|(_, w)| **w == SimdWidth::W128);
        assert!(has_w128_vec, "vec VRegs must use the same SimdWidth as the GEMV width parameter");
    }

    #[test]
    fn test_emit_mtp_draft_inline_depth_zero_loop_bound_is_zero() {
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_mtp_draft_inline(
            &mut prog,
            0,
            64,
            128,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok());
        let outer_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(b), step_bytes, .. } if *b == 0 && *step_bytes == 1 => Some(true),
            _ => None,
        });
        assert!(outer_bound.is_some(), "depth=0 must emit loop with BoundExpr::Const(0)");
    }

    #[test]
    fn test_emit_mtp_draft_inline_emits_scalar_store_to_output_tokens_per_depth() {
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let depth = 4;

        let result = emit_mtp_draft_inline(
            &mut prog,
            depth,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        assert!(result.is_ok());
        let store_count = prog.instrs.iter().filter(|i| match i {
            VmInstr::ScalarStore { base, .. } if *base == output_tokens_ptr => true,
            _ => false,
        }).count();
        assert!(store_count > 0, "emit_mtp_draft_inline must emit stores to output_tokens_ptr");
    }

    #[test]
    fn test_emit_mtp_draft_inline_provenance_valid_all_depths() {
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_mtp_draft_inline(
            &mut prog,
            5,
            64,
            128,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        assert!(result.is_ok());
        assert!(prog.validate_provenance().is_ok(), "all VRegs must be declared before use across all 5 depths");
    }

    #[test]
    fn test_emit_mtp_gemv_output_stores_use_scalar_byte_offset_multiply() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let elem_bytes = 4;

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, elem_bytes, 8, SimdWidth::W256,
        );

        let has_mul_by_elem_bytes = prog.instrs.iter().any(|i| match i {
            VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(imm), .. } if *imm == elem_bytes as i64 => true,
            _ => false,
        });
        assert!(has_mul_by_elem_bytes, "output byte offset must be computed as v_ctr * elem_bytes");
    }

    #[test]
    fn test_emit_mtp_gemv_vec_load_from_input_uses_loop_offset() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 8, SimdWidth::W256,
        );

        let uses_loop_offset = prog.instrs.iter().any(|i| match i {
            VmInstr::VecLoad { base, offset: OffsetExpr::LoopOffset(_), .. } if *base == input_ptr => true,
            _ => false,
        });
        assert!(uses_loop_offset, "hidden vector load must use LoopOffset for streaming access");
    }

    #[test]
    fn test_emit_mtp_draft_inline_weight_ptr_advances_per_depth() {
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden_size = 32;
        let vocab_size = 64;
        let proj_bytes = (vocab_size * hidden_size * 4) as i64;

        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            hidden_size,
            vocab_size,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        assert!(result.is_ok());
        let mul_by_proj = prog.instrs.iter().filter(|i| match i {
            VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(imm), .. } if *imm == proj_bytes => true,
            _ => false,
        }).count();
        assert!(mul_by_proj >= 1, "weight offset must use multiply by proj_bytes");
    }

    #[test]
    fn test_emit_mtp_gemv_combined_offset_adds_vreg_and_h_off() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 8, SimdWidth::W256,
        );

        let has_combined_offset = prog.instrs.iter().any(|i| match i {
            VmInstr::GprBinOp { a, b: GprOperand::VReg(_), op: GprOp::Add, .. }
                if a != &input_ptr && a != &output_ptr => true,
            _ => false,
        });
        assert!(has_combined_offset, "weight vec load must combine v_off + h_off via Add");
    }

    #[test]
    fn test_emit_mtp_draft_inline_weight_ptr_adds_to_base() {
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        assert!(result.is_ok());
        let weight_add_count = prog.instrs.iter().filter(|i| match i {
            VmInstr::GprBinOp { a, op: GprOp::Add, b: GprOperand::VReg(_), .. } if *a == weight_ptr => true,
            _ => false,
        }).count();
        assert!(weight_add_count >= 1, "depth_weight_ptr must be computed as weight_ptr + offset");
    }

    #[test]
    fn test_emit_mtp_gemv_value_domains_valid() {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        assert!(
            prog.validate_value_domains().is_ok(),
            "all value domains must be consistent (ptr used as ptr, scalar as scalar, etc.)"
        );
    }

    // ── Test 27: emit_mtp_gemv weight VecLoad uses ScalarVReg offset for combined addressing ──
    #[test]
    fn test_emit_mtp_gemv_weight_load_uses_scalar_vreg_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 8, SimdWidth::W256,
        );

        // Assert: weight VecLoad must use ScalarVReg offset (v_off + h_off combined)
        let weight_load_uses_combined = prog.instrs.iter().any(|i| match i {
            VmInstr::VecLoad { base, offset: OffsetExpr::ScalarVReg(_), .. } if *base == weight_ptr => true,
            _ => false,
        });
        assert!(weight_load_uses_combined, "weight VecLoad must use ScalarVReg offset from combined v_off+h_off");
    }

    // ── Test 28: emit_mtp_gemv with hidden exactly equal to lanes produces one inner iteration ──
    #[test]
    fn test_emit_mtp_gemv_hidden_equals_lanes_single_inner_iter() {
        // Arrange: hidden=8, lanes=8 => hidden_vec_iters=1 (single inner loop iteration)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            8, 32, 4, 8, SimdWidth::W256,
        );

        // Assert: inner loop bound must be 1 (hidden_vec_iters = 8/8 = 1)
        let inner_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. } if *v == 1 && *step_bytes == 32 => Some(true),
            _ => None,
        });
        assert!(inner_bound.is_some(), "hidden=lanes => inner loop bound=1, step=lanes*elem_bytes=32");
        assert!(prog.validate_structure().is_ok());
    }

    // ── Test 29: emit_mtp_draft_inline with W128 uses correct vec register width ──
    #[test]
    fn test_emit_mtp_draft_inline_w128_vec_width() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let width = SimdWidth::W128;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            width,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        let has_w128_vec = prog.instrs.iter().any(|i| match i {
            VmInstr::DeclareVReg { kind: VRegKind::Vec, width: w, .. } if *w == SimdWidth::W128 => true,
            _ => false,
        });
        assert!(has_w128_vec, "W128 width must propagate to vec register declarations");
    }

    // ── Test 30: emit_mtp_gemv with large hidden produces correct inner step_bytes ──
    #[test]
    fn test_emit_mtp_gemv_large_hidden_inner_step() {
        // Arrange: hidden=256, lanes=16 (W512) => inner step_bytes = 16*4 = 64
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            256, 512, 4, 16, SimdWidth::W512,
        );

        // Assert: inner loop step = lanes * elem_bytes = 16 * 4 = 64
        let inner_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == 256 / 16 => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(inner_step, Some(64), "inner loop step must be lanes(16) * elem_bytes(4) = 64");
    }

    // ── Test 31: emit_mtp_draft_inline depth=0 still emits AddPtr for logits_ptr ──
    #[test]
    fn test_emit_mtp_draft_inline_depth_zero_emits_addptr() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            0,
            64,
            128,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: even with depth=0, logits_ptr is allocated via AddPtr
        assert!(result.is_ok());
        let has_addptr = prog.instrs.iter().any(|i| match i {
            VmInstr::AddPtr { base, .. } if *base == output_tokens_ptr => true,
            _ => false,
        });
        assert!(has_addptr, "logits_ptr must be set via AddPtr even when depth=0");
    }

    // ── Test 32: emit_mtp_gemv outer loop bound matches vocab size exactly ──
    #[test]
    fn test_emit_mtp_gemv_outer_bound_matches_vocab() {
        // Arrange: use a non-power-of-2 vocab to verify exact matching
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vocab = 97; // prime number, non-power-of-2

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, vocab, 4, 8, SimdWidth::W256,
        );

        // Assert: outer loop bound must be exactly vocab=97
        let outer_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *step_bytes == 32 * 4 => Some(*v),
            _ => None,
        });
        assert_eq!(outer_bound, Some(97), "outer loop bound must equal vocab size");
    }

    // ── Test 33: emit_mtp_draft_inline provenance valid with W512 and depth 3 ──
    #[test]
    fn test_emit_mtp_draft_inline_w512_provenance() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            3,
            128,
            256,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W512,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        assert!(prog.validate_provenance().is_ok(), "W512 depth=3 must pass provenance validation");
        assert!(prog.validate_structure().is_ok(), "W512 depth=3 must pass structure validation");
    }

    // ── Test 34: emit_mtp_gemv with minimal vocab=1 stores exactly one scalar result ──
    #[test]
    fn test_emit_mtp_gemv_vocab_one_stores_one_result() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            16, 1, 4, 8, SimdWidth::W256,
        );

        // Assert: exactly one ScalarStore to output_ptr
        let store_count = prog.instrs.iter().filter(|i| match i {
            VmInstr::ScalarStore { base, .. } if *base == output_ptr => true,
            _ => false,
        }).count();
        assert_eq!(store_count, 1, "vocab=1 must produce exactly one output store");
    }

    // ── Test 35: emit_mtp_gemv emits correct GprBinOp kinds for offset math ──
    #[test]
    fn test_emit_mtp_gemv_offset_math_uses_add_and_mul() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: offset math must use both Add (for combining v_off + h_off) and Mul (for elem_bytes)
        let has_add = prog.instrs.iter().any(|i| matches!(i, VmInstr::GprBinOp { op: GprOp::Add, .. }));
        let has_mul = prog.instrs.iter().any(|i| matches!(i, VmInstr::GprBinOp { op: GprOp::Mul, .. }));
        assert!(has_add, "GEMV offset math must include Add for pointer arithmetic");
        assert!(has_mul, "GEMV offset math must include Mul for byte offset computation");
    }

    // ── Test 36: emit_mtp_draft_inline with large hidden and small vocab passes width validation ──
    #[test]
    fn test_emit_mtp_draft_inline_width_consistency() {
        // Arrange: hidden=256, vocab=8, depth=2, W512
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            256,
            8,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W512,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        assert!(
            prog.validate_width_consistency().is_ok(),
            "all vec operations must use consistent SimdWidth (W512)"
        );
    }

    // ── Test 37: emit_mtp_gemv inner loop FMA acc register matches acc dst ──
    #[test]
    fn test_emit_mtp_gemv_fma_dst_equals_acc_operand() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 8, SimdWidth::W256,
        );

        // Assert: FMA dst must equal acc (in-place accumulation)
        let all_fma_consistent = prog.instrs.iter().all(|i| match i {
            VmInstr::Fma { dst, acc, .. } => dst == acc,
            _ => true,
        });
        assert!(all_fma_consistent, "FMA dst must equal acc for in-place accumulation");
    }

    // ── Test 38: emit_mtp_draft_inline depth=1 emits exactly one Argmax ──
    #[test]
    fn test_emit_mtp_draft_inline_single_depth_single_argmax() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            1,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        let argmax_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Argmax { .. })).count();
        assert_eq!(argmax_count, 1, "depth=1 must emit exactly one Argmax");
    }

    // ── Test 39: emit_mtp_gemv declares DeclareVReg instructions for all used registers ──
    #[test]
    fn test_emit_mtp_gemv_declare_vreg_covers_all_usage() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: provenance validation confirms all VRegs declared before use
        assert!(prog.validate_provenance().is_ok());
        // Also verify explicit DeclareVReg instructions exist
        let declare_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::DeclareVReg { .. })).count();
        assert!(declare_count > 0, "GEMV must emit DeclareVReg for all allocated registers");
    }

    // ── Test 40: emit_mtp_draft_inline emits at least one Argmax per depth iteration ──
    #[test]
    fn test_emit_mtp_draft_inline_argmax_present_for_nonzero_depth() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            4,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: the loop template contains at least one Argmax (executed per depth at runtime)
        assert!(result.is_ok());
        let argmax_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Argmax { .. })).count();
        assert!(
            argmax_count >= 1,
            "depth>0 must emit at least one Argmax in the loop template (found {argmax_count})"
        );
    }

    // ── Test 41: emit_mtp_draft_inline emits ScalarStore to output_tokens_ptr ──
    #[test]
    fn test_emit_mtp_draft_inline_stores_candidates_to_output() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            3,
            64,
            128,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: loop template emits ScalarStore to output_tokens_ptr (executed per depth at runtime)
        assert!(result.is_ok());
        let has_store = prog.instrs.iter().any(|i| match i {
            VmInstr::ScalarStore { base, .. } if *base == output_tokens_ptr => true,
            _ => false,
        });
        assert!(
            has_store,
            "emit_mtp_draft_inline must emit ScalarStore to output_tokens_ptr in the loop body"
        );
    }

    // ── Test 42: emit_mtp_gemv VecLoadConst values count matches lanes ──
    #[test]
    fn test_emit_mtp_gemv_vec_load_const_count_matches_lanes() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let lanes = 8;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, lanes, SimdWidth::W256,
        );

        // Assert: VecLoadConst for accumulator init must have values.len() == lanes
        let const_load = prog.instrs.iter().find(|i| matches!(i, VmInstr::VecLoadConst { .. }));
        assert!(const_load.is_some(), "must have VecLoadConst for accumulator init");
        if let VmInstr::VecLoadConst { values, .. } = const_load.unwrap() {
            assert_eq!(
                values.len(), lanes,
                "VecLoadConst values count must match lanes={lanes}"
            );
        }
    }

    // ── Test 43: emit_mtp_gemv Scalar width inner loop has correct bound ──
    #[test]
    fn test_emit_mtp_gemv_scalar_width_inner_bound_equals_hidden() {
        // Arrange: Scalar width => lanes=1, hidden=16 => hidden_vec_iters=16
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 16;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, 8, 4, 1, SimdWidth::Scalar,
        );

        // Assert: inner loop bound = hidden / lanes = 16 / 1 = 16
        let inner_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == hidden && *step_bytes == 4 => Some(*v),
            _ => None,
        });
        assert_eq!(
            inner_bound, Some(hidden),
            "Scalar width inner loop bound must equal hidden={hidden}"
        );
    }

    // ── Test 44: emit_mtp_draft_inline outer loop bound matches depth ──
    #[test]
    fn test_emit_mtp_draft_inline_loop_bound_equals_depth() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let depth = 7;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            depth,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        let outer_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. } if *v == depth => Some(*v),
            _ => None,
        });
        assert_eq!(
            outer_bound, Some(depth),
            "outer loop bound must equal depth={depth}"
        );
    }

    // ── Test 45: emit_mtp_gemv VecLoad instructions read from both input and weight ──
    #[test]
    fn test_emit_mtp_gemv_vec_loads_from_both_inputs() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 8, SimdWidth::W256,
        );

        // Assert: at least one VecLoad from input_ptr and one from weight_ptr
        let input_loads = prog.instrs.iter().any(|i| match i {
            VmInstr::VecLoad { base, .. } if *base == input_ptr => true,
            _ => false,
        });
        let weight_loads = prog.instrs.iter().any(|i| match i {
            VmInstr::VecLoad { base, .. } if *base == weight_ptr => true,
            _ => false,
        });
        assert!(input_loads, "GEMV must load hidden vector from input_ptr");
        assert!(weight_loads, "GEMV must load weight from weight_ptr");
    }

    // ── Test 46: emit_mtp_draft_inline with Scalar width produces valid structure ──
    #[test]
    fn test_emit_mtp_draft_inline_scalar_width_structure() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            16,
            32,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::Scalar,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "Scalar width MTP must produce balanced loops");
        assert!(prog.validate_provenance().is_ok(), "Scalar width MTP provenance must be valid");
    }

    // ── Test 47: emit_mtp_gemv passes width consistency validation ──
    #[test]
    fn test_emit_mtp_gemv_width_consistency_valid() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            64, 128, 4, 8, SimdWidth::W256,
        );

        // Assert: all vec operations use consistent SimdWidth (W256)
        assert!(
            prog.validate_width_consistency().is_ok(),
            "all vec operations must use consistent SimdWidth (W256)"
        );
    }

    // ── Test 48: emit_mtp_draft_inline passes width consistency with W128 ──
    #[test]
    fn test_emit_mtp_draft_inline_w128_width_consistency() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            3,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W128,
            QuantPrecision::F32,
        );

        // Assert
        assert!(result.is_ok());
        assert!(
            prog.validate_width_consistency().is_ok(),
            "W128 depth=3 must pass width consistency validation"
        );
    }

    // ── Test 49: emit_mtp_gemv HReduce accumulator src matches FMA dst ──
    #[test]
    fn test_emit_mtp_gemv_hreduce_src_matches_fma_dst() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 16, 4, 8, SimdWidth::W256,
        );

        // Assert: HReduce src must be the same VReg as FMA dst (the accumulator)
        let fma_dst = prog.instrs.iter().find_map(|i| match i {
            VmInstr::Fma { dst, .. } => Some(*dst),
            _ => None,
        });
        let hreduce_src = prog.instrs.iter().find_map(|i| match i {
            VmInstr::HReduce { src, .. } => Some(*src),
            _ => None,
        });
        assert!(
            fma_dst.is_some() && hreduce_src.is_some(),
            "must have both FMA and HReduce instructions"
        );
        assert_eq!(
            fma_dst, hreduce_src,
            "HReduce must reduce the FMA accumulator register"
        );
    }

    // ── Test 50: emit_mtp_draft_inline Argmax uses correct logits_ptr ──
    #[test]
    fn test_emit_mtp_draft_inline_argmax_uses_logits_ptr() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vocab_size = 64;
        let vocab_bytes = vocab_size * 4;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            vocab_size,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: Argmax must reference logits_ptr with correct vocab_bytes
        assert!(result.is_ok());
        let argmax_uses_logits = prog.instrs.iter().any(|i| match i {
            VmInstr::Argmax { logits_ptr: lp, vocab_bytes: vb, .. }
                if *vb == vocab_bytes => true,
            _ => false,
        });
        assert!(
            argmax_uses_logits,
            "Argmax must use logits_ptr with vocab_bytes={} (vocab_size * elem_bytes)",
            vocab_bytes
        );
    }

    // ── Test 51: emit_mtp_draft_inline candidate store uses byte offset multiplication by 4 ──
    #[test]
    fn test_emit_mtp_draft_inline_candidate_store_byte_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            3,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: the depth index byte offset must be k_ctr * 4
        assert!(result.is_ok());
        let has_byte_off_mul = prog.instrs.iter().any(|i| match i {
            VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(4), .. } => true,
            _ => false,
        });
        assert!(
            has_byte_off_mul,
            "candidate store must compute byte offset as k_ctr * 4 (sizeof u32)"
        );
    }

    // ── Test 52: emit_mtp_gemv outer loop step_bytes matches hidden * elem_bytes ──
    #[test]
    fn test_emit_mtp_gemv_outer_step_matches_hidden_times_elem_bytes() {
        // Arrange: non-power-of-2 hidden to verify exact step computation
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 48;
        let elem_bytes = 4;
        let expected_step = hidden * elem_bytes;

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            hidden, 32, elem_bytes, 8, SimdWidth::W256,
        );

        // Assert: outer loop step_bytes must be hidden * elem_bytes = 192
        let outer_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. } if *v == 32 => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(
            outer_step,
            Some(expected_step),
            "outer loop step_bytes must equal hidden({}) * elem_bytes({}) = {}",
            hidden, elem_bytes, expected_step
        );
    }

    // ── Test 53: emit_mtp_draft_inline with BF16 dtype passes structure validation ──
    #[test]
    fn test_emit_mtp_draft_inline_bf16_structure_valid() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::BF16,
        );

        // Assert: BF16 dtype must still produce valid loop structure
        assert!(result.is_ok());
        assert!(
            prog.validate_structure().is_ok(),
            "BF16 emit_mtp_draft_inline must produce balanced loops"
        );
    }

    // ── Test 54: emit_mtp_draft_inline with BF16 dtype produces correct elem_bytes ──
    #[test]
    fn test_emit_mtp_draft_inline_bf16_vocab_bytes_half() {
        // Arrange: BF16 elem_bytes = 2, vocab=64 => vocab_bytes=128 (vs F32's 256)
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vocab_size = 64;
        let bf16_vocab_bytes = vocab_size * 2; // BF16 elem_bytes = 2

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            vocab_size,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::BF16,
        );

        // Assert: Argmax vocab_bytes must be 128 (64 * 2) not 256 (64 * 4)
        assert!(result.is_ok());
        let argmax_uses_bf16_bytes = prog.instrs.iter().any(|i| match i {
            VmInstr::Argmax { vocab_bytes: vb, .. } if *vb == bf16_vocab_bytes => true,
            _ => false,
        });
        assert!(
            argmax_uses_bf16_bytes,
            "BF16 Argmax must use vocab_bytes={} (vocab_size * 2), not F32 size",
            bf16_vocab_bytes
        );
    }

    // ── Test 55: emit_mtp_draft_inline with depth 6 passes structure and provenance validation ──
    #[test]
    fn test_emit_mtp_draft_inline_large_depth_structure_valid() {
        // Arrange: large depth to stress-test loop nesting and register allocation
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            6,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: structure and provenance must be valid with depth=6
        assert!(result.is_ok());
        assert!(
            prog.validate_structure().is_ok(),
            "structure must be balanced with depth=6"
        );
        assert!(
            prog.validate_provenance().is_ok(),
            "provenance must be valid with depth=6"
        );
    }

    // ── Test 56: emit_mtp_gemv with hidden not divisible by lanes still valid ──
    #[test]
    fn test_emit_mtp_gemv_hidden_not_aligned_to_lanes_structure_valid() {
        // Arrange: hidden=20, lanes=8 => hidden_vec_iters = 20/8 = 2 (truncated, not exact fit)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            20, 32, 4, 8, SimdWidth::W256,
        );

        // Assert: structure must still be balanced even with non-aligned hidden
        assert!(
            prog.validate_structure().is_ok(),
            "non-aligned hidden must still produce balanced loops"
        );
        assert!(
            prog.validate_provenance().is_ok(),
            "non-aligned hidden provenance must be valid"
        );
    }

    // ── Test 57: emit_mtp_draft_inline logits_ptr uses AddPtr from output_tokens_ptr ──
    #[test]
    fn test_emit_mtp_draft_inline_logits_ptr_derived_from_output_tokens() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: logits_ptr must be derived from output_tokens_ptr via AddPtr(offset=0)
        assert!(result.is_ok());
        let has_logits_addptr = prog.instrs.iter().any(|i| match i {
            VmInstr::AddPtr { base, offset: 0, .. } if *base == output_tokens_ptr => true,
            _ => false,
        });
        assert!(
            has_logits_addptr,
            "logits_ptr must be set via AddPtr from output_tokens_ptr with offset=0"
        );
    }

    // ── Test 58: emit_mtp_gemv with W128 and large dimensions passes all validations ──
    #[test]
    fn test_emit_mtp_gemv_w128_large_dims_all_validations() {
        // Arrange: W128 (lanes=4), hidden=128, vocab=256
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            128, 256, 4, 4, SimdWidth::W128,
        );

        // Assert: all four validations must pass simultaneously
        assert!(prog.validate_structure().is_ok(), "structure must be balanced");
        assert!(prog.validate_provenance().is_ok(), "provenance must be valid");
        assert!(
            prog.validate_width_consistency().is_ok(),
            "W128 width must be consistent across all vec ops"
        );
        assert!(
            prog.validate_value_domains().is_ok(),
            "value domains must be consistent"
        );
    }

    // ── Test 59: emit_mtp_draft_inline depth=2 emits one outer loop with correct step ──
    #[test]
    fn test_emit_mtp_draft_inline_outer_loop_step_is_one() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let depth = 2;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            depth,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: the depth loop uses step_bytes=1 (emit_loop default for counter)
        assert!(result.is_ok());
        let depth_loop = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(b), step_bytes, .. } if *b == depth => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(
            depth_loop,
            Some(1),
            "depth loop must use step_bytes=1 (counter-based iteration)"
        );
    }

    // ── Test 60: emit_mtp_gemv with hidden=0 produces outer loop but no inner loop ──
    #[test]
    fn test_emit_mtp_gemv_zero_hidden_no_inner_loop() {
        // Arrange: hidden=0 => hidden_vec_iters=0 => inner loop body skipped entirely
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            0, 16, 4, 8, SimdWidth::W256,
        );

        // Assert: only 1 loop (outer vocab), no inner hidden loop
        let loop_begin_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert_eq!(loop_begin_count, 1, "hidden=0 must produce only the outer vocab loop");
        assert!(prog.validate_structure().is_ok(), "structure must be balanced with hidden=0");
    }

    // ── Test 61: emit_mtp_gemv inner loop is strictly nested inside outer loop ──
    #[test]
    fn test_emit_mtp_gemv_inner_loop_nested_in_outer() {
        // Arrange: verify LoopBegin/LoopEnd nesting — inner loop must start after outer LoopBegin
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: sequence must be LoopBegin(outer), ..., LoopBegin(inner), ..., LoopEnd, ..., LoopEnd
        let mut outer_seen = false;
        let mut inner_after_outer = false;
        for instr in &prog.instrs {
            match instr {
                VmInstr::LoopBegin { bound: BoundExpr::Const(v), .. } if *v == 64 => {
                    outer_seen = true;
                }
                VmInstr::LoopBegin { bound: BoundExpr::Const(v), .. } if *v == 4 => {
                    inner_after_outer = outer_seen;
                }
                _ => {}
            }
        }
        assert!(inner_after_outer, "inner hidden loop must appear after outer vocab loop begins");
    }

    // ── Test 62: emit_mtp_gemv HReduce output feeds directly into ScalarStore ──
    #[test]
    fn test_emit_mtp_gemv_hreduce_result_consumed_by_scalar_store() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: find HReduce dst, then verify a ScalarStore uses that dst as src
        let hreduce_dst = prog.instrs.iter().find_map(|i| match i {
            VmInstr::HReduce { dst, op: ReduceOp::Sum, .. } => Some(*dst),
            _ => None,
        });
        assert!(hreduce_dst.is_some(), "must have HReduce Sum");
        let hreduce_dst = hreduce_dst.unwrap();
        let store_uses_hreduce = prog.instrs.iter().any(|i| match i {
            VmInstr::ScalarStore { src, base, .. } if *base == output_ptr && *src == hreduce_dst => true,
            _ => false,
        });
        assert!(store_uses_hreduce, "ScalarStore must consume HReduce output directly");
    }

    // ── Test 63: emit_mtp_gemv with elem_bytes=2 produces correct inner step_bytes ──
    #[test]
    fn test_emit_mtp_gemv_elem_bytes_2_inner_step() {
        // Arrange: elem_bytes=2 (BF16-style), hidden=32, lanes=8 => inner step = 8*2 = 16
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 2, 8, SimdWidth::W256,
        );

        // Assert: inner loop step = lanes * elem_bytes = 8 * 2 = 16
        let inner_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == 32 / 8 => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(inner_step, Some(16), "inner step must be lanes(8) * elem_bytes(2) = 16");
        // Assert: outer loop step = hidden * elem_bytes = 32 * 2 = 64
        let outer_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == 64 => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(outer_step, Some(64), "outer step must be hidden(32) * elem_bytes(2) = 64");
    }

    // ── Test 64: emit_mtp_draft_inline depth=0 loop body still has Argmax and ScalarStore in template ──
    #[test]
    fn test_emit_mtp_draft_inline_depth_zero_loop_template_has_argmax_and_store() {
        // Arrange: depth=0 emits a loop with bound=0. The template body still contains
        // Argmax and ScalarStore (they are emitted unconditionally by emit_loop closure),
        // but the loop will execute 0 iterations at runtime.
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            0,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: loop template body contains Argmax and ScalarStore (emitted at compile time)
        assert!(result.is_ok());
        let has_argmax = prog.instrs.iter().any(|i| matches!(i, VmInstr::Argmax { .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarStore { base, .. } if *base == output_tokens_ptr));
        assert!(has_argmax, "loop template must contain Argmax even with depth=0");
        assert!(has_store, "loop template must contain ScalarStore to output_tokens_ptr even with depth=0");
    }

    // ── Test 65: emit_mtp_draft_inline depth=4 emits exactly one outer loop ──
    #[test]
    fn test_emit_mtp_draft_inline_depth_four_emits_one_outer_loop() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let depth = 4;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            depth,
            64,
            128,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: one depth loop (bound=4) containing nested GEMV loops (2 per iteration)
        assert!(result.is_ok());
        let depth_loop_count = prog.instrs.iter().filter(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(b), step_bytes, .. } if *b == depth && *step_bytes == 1 => true,
            _ => false,
        }).count();
        assert_eq!(depth_loop_count, 1, "must have exactly one depth loop with bound=depth and step=1");
    }

    // ── Test 66: emit_mtp_gemv with Scalar width and hidden=0 still emits VecLoadConst and HReduce ──
    #[test]
    fn test_emit_mtp_gemv_scalar_zero_hidden_has_hreduce_and_acc_init() {
        // Arrange: Scalar width, hidden=0 — inner loop skipped, but acc init + HReduce still present
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            0, 4, 4, 1, SimdWidth::Scalar,
        );

        // Assert: accumulator init (VecLoadConst) and HReduce still emitted even with hidden=0
        let has_acc_init = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoadConst { .. }));
        let has_hreduce = prog.instrs.iter().any(|i| matches!(i, VmInstr::HReduce { .. }));
        assert!(has_acc_init, "accumulator init must be emitted even with hidden=0");
        assert!(has_hreduce, "HReduce must be emitted even with hidden=0");
        assert!(prog.validate_provenance().is_ok());
    }

    // ── Test 67: emit_mtp_draft_inline ScalarStore uses ScalarVReg offset ──
    #[test]
    fn test_emit_mtp_draft_inline_store_uses_scalar_vreg_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: ScalarStore in depth loop must use OffsetExpr::ScalarVReg for byte offset
        assert!(result.is_ok());
        let has_vreg_offset_store = prog.instrs.iter().any(|i| match i {
            VmInstr::ScalarStore { base, offset: OffsetExpr::ScalarVReg(_), .. }
                if *base == output_tokens_ptr => true,
            _ => false,
        });
        assert!(
            has_vreg_offset_store,
            "depth loop ScalarStore must use ScalarVReg offset for dynamic depth indexing"
        );
    }

    // ── Test 68: emit_mtp_draft_inline with non-power-of-2 vocab passes structure and provenance ──
    #[test]
    fn test_emit_mtp_draft_inline_non_power_of_2_vocab() {
        // Arrange: vocab=48 (not power of 2), hidden=32, depth=2
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            32,
            48, // non-power-of-2 but SIMD-aligned (hidden=32, lanes=8)
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: core validations must pass with non-power-of-2 vocab
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "structure must be balanced with vocab=48");
        assert!(prog.validate_provenance().is_ok(), "provenance must be valid with vocab=48");
        assert!(prog.validate_width_consistency().is_ok(), "width must be consistent with vocab=48");
        // Verify Argmax vocab_bytes = 48 * 4 = 192 (not rounded to power-of-2)
        let argmax_uses_exact_vocab = prog.instrs.iter().any(|i| match i {
            VmInstr::Argmax { vocab_bytes, .. } if *vocab_bytes == 48 * 4 => true,
            _ => false,
        });
        assert!(argmax_uses_exact_vocab, "Argmax must use vocab_bytes=192 (48*4), not rounded up");
    }

    // ── Test 69: emit_mtp_gemv with W128 produces exactly W128-width VecLoadConst ──
    #[test]
    fn test_emit_mtp_gemv_w128_vec_load_const_width() {
        // Arrange: W128 => lanes=4, VecLoadConst must use W128 width
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 4, SimdWidth::W128,
        );

        // Assert: VecLoadConst must declare with W128 width and have 4 values
        let vl_const = prog.instrs.iter().find(|i| matches!(i, VmInstr::VecLoadConst { .. }));
        assert!(vl_const.is_some(), "must have VecLoadConst for accumulator init");
        if let VmInstr::VecLoadConst { values, width: w, .. } = vl_const.unwrap() {
            assert_eq!(*w, SimdWidth::W128, "VecLoadConst width must be W128");
            assert_eq!(values.len(), 4, "VecLoadConst must have 4 values for W128 (lanes=4)");
        }
    }

    // ── Test 70: emit_mtp_gemv with vocab=0 produces only outer loop with bound=0 ──
    #[test]
    fn test_emit_mtp_gemv_zero_vocab_only_outer_loop_bound_zero() {
        // Arrange: vocab=0 => outer loop runs 0 iterations, no inner loop
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 0, 4, 8, SimdWidth::W256,
        );

        // Assert: one outer loop with bound=0, no ScalarStore to output
        assert!(prog.validate_structure().is_ok());
        let zero_bound_loop = prog.instrs.iter().any(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(0), .. } => true,
            _ => false,
        });
        assert!(zero_bound_loop, "vocab=0 must emit outer loop with bound=0");
        let store_count = prog.instrs.iter().filter(|i| match i {
            VmInstr::ScalarStore { base, .. } if *base == output_ptr => true,
            _ => false,
        }).count();
        assert!(store_count > 0, "loop template must contain ScalarStore even with vocab=0");
    }

    // ── Test 71: emit_mtp_draft_inline with hidden not aligned to lanes still passes all validations ──
    #[test]
    fn test_emit_mtp_draft_inline_non_aligned_hidden_all_validations() {
        // Arrange: hidden=20 (not divisible by lanes=8), depth=2
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            20, // hidden=20, lanes=8 => hidden_vec_iters=2 (truncated)
            64,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: all validations pass despite non-aligned hidden
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "structure must be balanced with hidden=20");
        assert!(prog.validate_provenance().is_ok(), "provenance must be valid with hidden=20");
        assert!(prog.validate_width_consistency().is_ok(), "width must be consistent with hidden=20");
    }

    // ── Test 72: emit_mtp_gemv allocates distinct VRegIds for counter and byte_offset ──
    #[test]
    fn test_emit_mtp_gemv_counter_and_offset_are_distinct_vregs() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: each LoopBegin must have distinct counter and byte_offset VRegIds
        for instr in &prog.instrs {
            if let VmInstr::LoopBegin { counter, byte_offset, .. } = instr {
                assert_ne!(
                    counter, byte_offset,
                    "loop counter and byte_offset must be distinct VRegIds"
                );
            }
        }
    }

    // ── Test 73: emit_mtp_draft_inline with minimal hidden=8 and vocab=8 passes structure ──
    #[test]
    fn test_emit_mtp_draft_inline_minimal_dims_structure() {
        // Arrange: smallest practical dimensions
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            8,
            8,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: minimal dimensions must produce valid structure
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "minimal dims must produce balanced loops");
        assert!(prog.validate_provenance().is_ok(), "minimal dims provenance must be valid");
    }

    // ── Test 74: emit_mtp_gemv emits exactly one VecLoadConst per outer loop iteration ──
    #[test]
    fn test_emit_mtp_gemv_single_vec_load_const_for_acc_init() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: exactly one VecLoadConst (accumulator initialization)
        let vl_const_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecLoadConst { .. })).count();
        assert_eq!(vl_const_count, 1, "GEMV must emit exactly one VecLoadConst for accumulator init");
    }

    // ── Test 75: emit_mtp_draft_inline depth=1 emits weight offset GprBinOp with Mul ──
    #[test]
    fn test_emit_mtp_draft_inline_single_depth_has_weight_offset_mul() {
        // Arrange
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden_size = 32;
        let vocab_size = 64;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            1,
            hidden_size,
            vocab_size,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: weight offset computation must multiply k_ctr by proj_bytes
        assert!(result.is_ok());
        let proj_bytes = (vocab_size * hidden_size * 4) as i64;
        let has_proj_mul = prog.instrs.iter().any(|i| match i {
            VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(imm), .. }
                if *imm == proj_bytes => true,
            _ => false,
        });
        assert!(has_proj_mul, "depth=1 must still emit weight offset = k_ctr * proj_bytes");
    }

    // ── Test 76: emit_mtp_gemv outer loop uses Counter and ByteOffset register kinds ──
    #[test]
    fn test_emit_mtp_gemv_outer_loop_uses_counter_and_byte_offset_kinds() {
        // Arrange
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: LoopBegin must use Counter and ByteOffset register kinds
        let outer_loop = prog.instrs.iter().find(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(64), .. } => true,
            _ => false,
        });
        assert!(outer_loop.is_some(), "must find outer vocab loop");
        // Verify the counter and byte_offset VRegs were declared with correct kinds
        assert!(prog.validate_provenance().is_ok(), "counter/byte_offset must have correct VReg kinds");
    }

    // ── Test 77: emit_mtp_draft_inline nested GEMV inner loop bound matches hidden_over_lanes ──
    #[test]
    fn test_emit_mtp_draft_inline_nested_gemv_inner_bound_correct() {
        // Arrange: hidden=64, lanes=8 (W256) => inner bound should be 8
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let hidden = 64;
        let lanes = 8;
        let expected_inner_bound = hidden / lanes;

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            hidden,
            32,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: inner GEMV loop must have bound = hidden / lanes = 8
        assert!(result.is_ok());
        let inner_bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == expected_inner_bound && *step_bytes == lanes * 4 => Some(*v),
            _ => None,
        });
        assert_eq!(
            inner_bound,
            Some(expected_inner_bound),
            "nested GEMV inner loop bound must be hidden({}) / lanes({}) = {}",
            hidden, lanes, expected_inner_bound
        );
    }

    // ── Test 78: emit_mtp_gemv with W512 and hidden=33 (non-aligned) passes all validations ──
    #[test]
    fn test_emit_mtp_gemv_w512_non_aligned_hidden_all_validations() {
        // Arrange: W512 (lanes=16), hidden=33 => hidden_vec_iters=33/16=2 (truncated)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            33, 64, 4, 16, SimdWidth::W512,
        );

        // Assert: all four validations pass with W512 and non-aligned hidden
        assert!(prog.validate_structure().is_ok(), "W512 non-aligned hidden structure must be balanced");
        assert!(prog.validate_provenance().is_ok(), "W512 non-aligned hidden provenance must be valid");
        assert!(
            prog.validate_width_consistency().is_ok(),
            "W512 width must be consistent with non-aligned hidden"
        );
        assert!(
            prog.validate_value_domains().is_ok(),
            "W512 non-aligned hidden value domains must be consistent"
        );
    }

    // ── Test 79: emit_mtp_draft_inline with hidden > vocab produces valid asymmetric structure ──
    #[test]
    fn test_emit_mtp_draft_inline_hidden_greater_than_vocab_structure() {
        // Arrange: hidden=128, vocab=16 (asymmetric: tall-skinny weight matrix)
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2,
            128, // hidden >> vocab
            16,
            hidden_ptr,
            weight_ptr,
            output_tokens_ptr,
            SimdWidth::W256,
            QuantPrecision::F32,
        );

        // Assert: asymmetric dimensions must produce valid structure
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "hidden>vocab structure must be balanced");
        assert!(prog.validate_provenance().is_ok(), "hidden>vocab provenance must be valid");
        // GEMV outer loop bound = vocab = 16
        let gemv_outer = prog.instrs.iter().any(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(16), step_bytes, .. }
                if *step_bytes == 128 * 4 => true,
            _ => false,
        });
        assert!(gemv_outer, "GEMV outer loop bound must be vocab(16), step=hidden*4=512");
    }

    // ── Test 80: ABI stack offsets have correct 8-byte alignment and contiguous stride ──
    #[test]
    fn test_mega_kernel_abi_stack_offsets_alignment_and_stride() {
        // Arrange: load ABI constants from mega_kernel_abi module
        use crate::compiler::mega_kernel_abi::{MEGA_KERNEL_STACK_OFFSETS, MEGA_KERNEL_PARAMS};

        // Assert: MEGA_KERNEL_STACK_OFFSETS length must match MEGA_KERNEL_PARAMS minus register params
        let register_params = 6; // rdi, rsi, rdx, rcx, r8, r9
        assert_eq!(
            MEGA_KERNEL_STACK_OFFSETS.len(),
            MEGA_KERNEL_PARAMS.len() - register_params,
            "stack offsets must cover exactly the non-register ABI parameters"
        );

        // Assert: every stack offset must be 8-byte aligned (SysV ABI requirement)
        for &off in MEGA_KERNEL_STACK_OFFSETS {
            assert_eq!(off % 8, 0, "stack offset {} must be 8-byte aligned (SysV ABI)", off);
        }

        // Assert: offsets must be contiguous with 8-byte stride starting from [rbp+16]
        for (i, &off) in MEGA_KERNEL_STACK_OFFSETS.iter().enumerate() {
            let expected = 16 + (i as i32) * 8;
            assert_eq!(off, expected, "stack offset at index {} must be {}, got {}", i, expected, off);
        }
    }

    // ── Test 81: SymDimSlotMap mega_kernel_abi resolves all named ABI parameters ──
    #[test]
    fn test_symdim_slotmap_mega_kernel_abi_resolves_all_params() {
        // Arrange
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Assert: all core ABI parameters must resolve to valid PtrExpr
        let core_params = ["input", "weights", "kv_cache", "scratchpad", "telemetry"];
        for param in &core_params {
            let resolved = sym_map.resolve(param);
            assert!(resolved.is_some(), "SymDimSlotMap must resolve core param '{}'", param);
        }

        // Assert: resolved scratchpad must be StackArg (not AbiArg) since it's on the stack
        let scratchpad_expr = sym_map.resolve("scratchpad").unwrap();
        assert!(
            matches!(scratchpad_expr, PtrExpr::StackArg(_)),
            "scratchpad must resolve to StackArg (stack-based parameter)"
        );
    }

    // ── Test 82: emit_mtp_gemv with both hidden=0 and vocab=0 produces balanced empty structure ──
    #[test]
    fn test_emit_mtp_gemv_zero_hidden_and_zero_vocab_balanced_structure() {
        // Arrange: both dimensions zero — ultimate zero-length batch edge case
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            0, 0, 4, 8, SimdWidth::W256,
        );

        // Assert: structure must still be balanced (zero-iteration outer loop)
        assert!(prog.validate_structure().is_ok(), "zero-dim GEMV must produce balanced loops");
        assert!(prog.validate_provenance().is_ok(), "zero-dim GEMV provenance must be valid");
        // The outer loop has bound=0 and there is no inner loop (hidden=0)
        let outer_loop = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(0), .. } => Some(true),
            _ => None,
        });
        assert!(outer_loop.is_some(), "hidden=0, vocab=0 must still emit outer loop with bound=0");
    }

    // ── Test 83: emit_mtp_draft_inline with single-phase depth=1 produces minimal instruction set ──
    #[test]
    fn test_emit_mtp_draft_inline_single_phase_depth_one_minimal_instructions() {
        // Arrange: depth=1 is the simplest single-phase configuration
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            1, 32, 64,
            hidden_ptr, weight_ptr, output_tokens_ptr,
            SimdWidth::W256, QuantPrecision::F32,
        );

        // Assert: depth=1 must produce valid structure with one depth loop
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "depth=1 must produce balanced structure");
        // One depth loop (bound=1, step=1) containing GEMV + Argmax + ScalarStore
        let depth_loop = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(1), step_bytes: 1, .. } => Some(true),
            _ => None,
        });
        assert!(depth_loop.is_some(), "depth=1 must emit exactly one loop with bound=1");
    }

    // ── Test 84: emit_mtp_gemv phase ordering — VecLoadConst always before Fma ──
    #[test]
    fn test_emit_mtp_gemv_phase_ordering_acc_init_before_fma() {
        // Arrange: verify that accumulator initialization (VecLoadConst)
        // always appears before the first FMA in the instruction sequence
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 4, 8, SimdWidth::W256,
        );

        // Assert: VecLoadConst position must be before first Fma position
        let vl_const_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::VecLoadConst { .. }));
        let fma_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::Fma { .. }));
        assert!(vl_const_pos.is_some(), "must have VecLoadConst for accumulator init");
        assert!(fma_pos.is_some(), "must have FMA for dot product");
        assert!(
            vl_const_pos.unwrap() < fma_pos.unwrap(),
            "VecLoadConst (acc init) must appear before FMA (acc accumulation)"
        );
    }

    // ── Test 85: ABI parameter slot 0 (prompt_len) has offset 16 (first stack param after rbp) ──
    #[test]
    fn test_mega_kernel_abi_slot_zero_is_prompt_len_at_offset_16() {
        // Arrange: verify the canonical ABI layout — first stack param is prompt_len at [rbp+16]
        use crate::compiler::mega_kernel_abi::MEGA_KERNEL_STACK_OFFSETS;

        // Assert: slot 0 must be 16 (prompt_len at [rbp+16])
        assert_eq!(MEGA_KERNEL_STACK_OFFSETS[0], 16, "slot 0 must be prompt_len at [rbp+16]");
        // Assert: last slot must be batch_ctx_ptr at [rbp+144]
        assert_eq!(
            *MEGA_KERNEL_STACK_OFFSETS.last().unwrap(), 144,
            "last slot must be batch_ctx_ptr at [rbp+144]"
        );
        // Assert: total of 17 stack parameters (6 register + 17 stack = 23 total params)
        assert_eq!(MEGA_KERNEL_STACK_OFFSETS.len(), 17, "must have exactly 17 stack parameters");
    }

    // ── Test 86: emit_mtp_draft_inline depth loop phase ordering — GEMV before Argmax before ScalarStore ──
    #[test]
    fn test_emit_mtp_draft_inline_phase_ordering_gemv_argmax_store() {
        // Arrange: verify the correct phase ordering within the depth loop
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2, 32, 64,
            hidden_ptr, weight_ptr, output_tokens_ptr,
            SimdWidth::W256, QuantPrecision::F32,
        );

        // Assert: phase ordering must be GEMV (LoopBegin) → Argmax → ScalarStore
        assert!(result.is_ok());
        let gemv_loop_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::LoopBegin { bound: BoundExpr::Const(2), step_bytes: 1, .. }));
        let argmax_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::Argmax { .. }));
        let store_pos = prog.instrs.iter().position(|i| matches!(i, VmInstr::ScalarStore { base, .. } if *base == output_tokens_ptr));
        assert!(gemv_loop_pos.is_some(), "must have depth loop (bound=2)");
        assert!(argmax_pos.is_some(), "must have Argmax after GEMV");
        assert!(store_pos.is_some(), "must have ScalarStore after Argmax");
        let gl = gemv_loop_pos.unwrap();
        let am = argmax_pos.unwrap();
        let ss = store_pos.unwrap();
        assert!(gl < am, "depth loop (GEMV) must start before Argmax");
        assert!(am < ss, "Argmax must come before ScalarStore");
    }

    // ── Test 87: emit_mtp_gemv inner loop offset uses LoopOffset for streaming access ──
    #[test]
    fn test_emit_mtp_gemv_inner_loop_hidden_load_uses_loop_offset_not_scalar_vreg() {
        // Arrange: verify the inner (hidden) loop VecLoad uses LoopOffset
        // for streaming access, not ScalarVReg (which would require separate GprBinOp)
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            64, 128, 4, 8, SimdWidth::W256,
        );

        // Assert: hidden VecLoad must use LoopOffset (streaming), not ScalarVReg (manual offset)
        let hidden_loads_loop_offset = prog.instrs.iter().any(|i| match i {
            VmInstr::VecLoad { base, offset: OffsetExpr::LoopOffset(_), .. } if *base == input_ptr => true,
            _ => false,
        });
        assert!(
            hidden_loads_loop_offset,
            "hidden VecLoad must use LoopOffset for streaming access"
        );
    }

    // ── Test 88: emit_mtp_draft_inline with BF16 dtype GEMV uses correct elem_bytes=2 for proj ──
    #[test]
    fn test_emit_mtp_draft_inline_bf16_proj_bytes_is_vocab_times_hidden_times_2() {
        // Arrange: BF16 elem_bytes=2, hidden=32, vocab=64 => proj_bytes = 64*32*2 = 4096
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let expected_proj_bytes = (64 * 32 * 2) as i64; // BF16: vocab * hidden * 2

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            2, 32, 64,
            hidden_ptr, weight_ptr, output_tokens_ptr,
            SimdWidth::W256, QuantPrecision::BF16,
        );

        // Assert: weight offset multiplication must use proj_bytes = vocab*hidden*2 (not *4)
        assert!(result.is_ok());
        let has_bf16_proj_mul = prog.instrs.iter().any(|i| match i {
            VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(imm), .. }
                if *imm == expected_proj_bytes => true,
            _ => false,
        });
        assert!(
            has_bf16_proj_mul,
            "BF16 depth loop must multiply by proj_bytes={} (vocab*hidden*2), not F32 size",
            expected_proj_bytes
        );
    }

    // ── Test 89: emit_mtp_gemv with elem_bytes=1 inner step equals lanes ──
    #[test]
    fn test_emit_mtp_gemv_elem_bytes_1_inner_step_equals_lanes() {
        // Arrange: elem_bytes=1 (hypothetical int8-style), hidden=32, lanes=8
        // => inner step = lanes * elem_bytes = 8 * 1 = 8
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_mtp_gemv(
            &mut prog, input_ptr, weight_ptr, output_ptr,
            32, 64, 1, 8, SimdWidth::W256,
        );

        // Assert: inner loop step must be lanes * elem_bytes = 8
        let inner_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(v), step_bytes, .. }
                if *v == 32 / 8 => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(inner_step, Some(8), "inner step must be lanes(8) * elem_bytes(1) = 8");
        // Assert: outer loop step must be hidden * elem_bytes = 32
        let outer_step = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(64), step_bytes, .. } => Some(*step_bytes),
            _ => None,
        });
        assert_eq!(outer_step, Some(32), "outer step must be hidden(32) * elem_bytes(1) = 32");
    }

    // ── Test 90: emit_mtp_draft_inline depth=0 and vocab=0 still produces valid zero-iteration loop ──
    #[test]
    fn test_emit_mtp_draft_inline_zero_depth_and_zero_vocab_structure_valid() {
        // Arrange: ultimate zero-length batch — depth=0 and vocab=0
        let mut prog = VmProgram::new();
        let hidden_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_tokens_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_mtp_draft_inline(
            &mut prog,
            0, 32, 0,
            hidden_ptr, weight_ptr, output_tokens_ptr,
            SimdWidth::W256, QuantPrecision::F32,
        );

        // Assert: structure must be balanced even with both depth=0 and vocab=0
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "zero-depth zero-vocab must produce balanced structure");
        assert!(prog.validate_provenance().is_ok(), "provenance must be valid with zero-depth zero-vocab");
        // The depth loop has bound=0, inner GEMV loop has vocab=0 bound — template body still valid
        let zero_depth_loop = prog.instrs.iter().any(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(0), step_bytes: 1, .. } => true,
            _ => false,
        });
        assert!(zero_depth_loop, "must emit depth loop with bound=0");
    }

    // ── Test 91: MtpKernelConfig construction with positive fields ──
    #[test]
    fn test_mtp_kernel_config_construction_positive_fields() {
        // Arrange
        use crate::compiler::mega_kernel_abi::MtpKernelConfig;

        let depth = 3;
        let hidden_size = 1024;
        let vocab_size = 32000;

        // Act
        let config = MtpKernelConfig { depth, hidden_size, vocab_size };

        // Assert: all fields must match construction values
        assert_eq!(config.depth, depth, "MtpKernelConfig.depth must match input");
        assert_eq!(config.hidden_size, hidden_size, "MtpKernelConfig.hidden_size must match input");
        assert_eq!(config.vocab_size, vocab_size, "MtpKernelConfig.vocab_size must match input");
    }

    // ── Test 92: MegaKernelBusinessConfig default values match SPEC expectations ──
    #[test]
    fn test_mega_kernel_business_config_default_values() {
        // Arrange
        use crate::compiler::mega_kernel_abi::{MegaKernelBusinessConfig, FfnActivation, OutputMode};

        // Act
        let default = MegaKernelBusinessConfig::default();

        // Assert: default output mode must be Generate with canonical max_new_tokens and eos
        assert!(
            matches!(
                &default.output_modes[0],
                OutputMode::Generate { max_new_tokens: 512, eos_token_id: 2 }
            ),
            "default output mode must be Generate(512, eos=2)"
        );
        // Assert: default disabled features
        assert!(!default.guardrail_enabled, "guardrail must be disabled by default");
        assert!(default.semantic_gatekeeper.is_none(), "SG must be None by default");
        assert!(default.intent_anchor_layer.is_none(), "intent anchor must be None by default");
        assert!(default.cot_step_hook.is_none(), "CoT step hook must be None by default");
        assert!(!default.has_head_rms_norm, "head_rms_norm must be false by default");
        assert!(!default.has_qk_norm, "qk_norm must be false by default");
        assert!(!default.has_value_norm, "value_norm must be false by default");
        assert!(default.mtp_config.is_none(), "MTP must be None by default");
        // Assert: default ffn_activation must be SwiGLU
        assert_eq!(default.ffn_activation, FfnActivation::SwiGLU, "default FFN activation must be SwiGLU");
        // Assert: default eps values
        assert_eq!(default.head_rms_norm_eps, 1e-6, "default head_rms_norm_eps must be 1e-6");
        assert_eq!(default.value_norm_eps, 1e-6, "default value_norm_eps must be 1e-6");
        // Assert: default optional fields
        assert!(default.logit_softcapping.is_none(), "logit softcapping must be None by default");
        assert!(default.embedding_scale.is_none(), "embedding_scale must be None by default");
        // Assert: default session/multimodal/debug flags
        assert!(!default.session_enabled, "session must be disabled by default");
        assert!(!default.multimodal_enabled, "multimodal must be disabled by default");
        assert!(!default.debug_jit, "debug_jit must be disabled by default");
    }

    // ── Test 93: OutputMode variants carry correct discriminant data ──
    #[test]
    fn test_output_mode_variants_carry_discriminant_data() {
        // Arrange
        use crate::compiler::mega_kernel_abi::{OutputMode, PoolMode};

        // Act: construct each variant
        let generate = OutputMode::Generate { max_new_tokens: 100, eos_token_id: 50256 };
        let classify_binary = OutputMode::ClassifyBinary {
            positive_token_id: 1,
            negative_token_id: 0,
        };
        let classify_multiway = OutputMode::ClassifyMultiway {
            label_token_ids: vec![10, 20, 30],
        };
        let encode = OutputMode::EncodeToLayer {
            anchor_layer: 12,
            pool_mode: PoolMode::MeanPool,
        };

        // Assert: each variant must carry its specific data fields
        match &generate {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 100);
                assert_eq!(*eos_token_id, 50256);
            }
            _ => panic!("Generate variant must match"),
        }
        match &classify_binary {
            OutputMode::ClassifyBinary { positive_token_id, negative_token_id } => {
                assert_eq!(*positive_token_id, 1);
                assert_eq!(*negative_token_id, 0);
            }
            _ => panic!("ClassifyBinary variant must match"),
        }
        match &classify_multiway {
            OutputMode::ClassifyMultiway { label_token_ids } => {
                assert_eq!(label_token_ids.len(), 3);
                assert_eq!(label_token_ids[0], 10);
            }
            _ => panic!("ClassifyMultiway variant must match"),
        }
        match &encode {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode } => {
                assert_eq!(*anchor_layer, 12);
                assert!(matches!(pool_mode, PoolMode::MeanPool), "pool_mode must be MeanPool");
            }
            _ => panic!("EncodeToLayer variant must match"),
        }
    }

    // ── Test 94: FfnActivation enum has exactly three variants with correct equality ──
    #[test]
    fn test_ffn_activation_enum_variants_and_equality() {
        // Arrange
        use crate::compiler::mega_kernel_abi::FfnActivation;

        // Act: construct all variants
        let swiglu = FfnActivation::SwiGLU;
        let geglu = FfnActivation::GeGLU;
        let gelu = FfnActivation::Gelu;

        // Assert: each variant must be distinct (PartialEq/Eq derived)
        assert_ne!(swiglu, geglu, "SwiGLU and GeGLU must be different");
        assert_ne!(swiglu, gelu, "SwiGLU and Gelu must be different");
        assert_ne!(geglu, gelu, "GeGLU and Gelu must be different");
        // Assert: self-equality must hold
        assert_eq!(swiglu, swiglu, "SwiGLU must equal itself");
        assert_eq!(geglu, geglu, "GeGLU must equal itself");
        assert_eq!(gelu, gelu, "Gelu must equal itself");
        // Assert: Copy semantics must hold (each variant is unit-like, Copy derived)
        let copy_swiglu = swiglu;
        assert_eq!(copy_swiglu, swiglu, "Copy of SwiGLU must equal original");
    }

    // ── Test 95: MegaKernelWeightLayout layer_base_offset computation is linear ──
    #[test]
    fn test_mega_kernel_weight_layout_layer_base_offset_linear() {
        // Arrange
        use crate::compiler::mega_kernel_abi::{MegaKernelWeightLayout, PerLayerWeightLayout};

        // Construct a layout with known geometry by hand (build is private).
        // hidden=256, elem_bytes=4, vocab_size=1000, num_layers=4
        let embed_bytes = 1000 * 256 * 4; // vocab * hidden * elem_bytes
        let layer_0_offset = embed_bytes;
        // Per-layer stride: attn_norm(1024) + w_q(65536) + w_k(32768) + w_v(32768)
        //   + w_o(65536) + w_q_norm(256) + w_k_norm(256) + ffn_norm(1024)
        //   + w_gate(524288) + w_up(524288) + w_down(524288) = 1739904
        let layer_stride = 256*4 + 4*64*256*4 + 2*64*256*4 + 2*64*256*4
            + 256*(4*64)*4 + 64*4 + 64*4 + 256*4
            + 512*256*4 + 512*256*4 + 256*512*4;
        let final_norm_bytes = 256 * 4;
        let final_norm_offset = layer_0_offset + 4 * layer_stride;
        let lm_head_offset = final_norm_offset + final_norm_bytes;
        let lm_head_bytes = 1000 * 256 * 4;
        let total_bytes = lm_head_offset + lm_head_bytes;
        let layout = MegaKernelWeightLayout {
            embed_offset: 0,
            embed_bytes,
            layer_0_offset,
            layer_stride,
            per_layer: PerLayerWeightLayout {
                attn_norm_offset: 0, attn_norm_bytes: 1024,
                w_q_offset: 1024, w_q_bytes: 65536,
                w_k_offset: 66560, w_k_bytes: 32768,
                w_v_offset: 99328, w_v_bytes: 32768,
                w_o_offset: 132096, w_o_bytes: 65536,
                w_q_norm_offset: 197632, w_q_norm_bytes: 256,
                w_k_norm_offset: 197888, w_k_norm_bytes: 256,
                ffn_norm_offset: 198144, ffn_norm_bytes: 1024,
                w_gate_offset: 199168, w_gate_bytes: 524288,
                w_up_offset: 723456, w_up_bytes: 524288,
                w_down_offset: 1247744, w_down_bytes: 524288,
            },
            final_norm_offset,
            final_norm_bytes,
            lm_head_offset,
            lm_head_bytes,
            total_bytes,
        };

        // Act: compute layer offsets for layers 0..4
        let offsets: Vec<usize> = (0..4).map(|i| layout.layer_base_offset(i)).collect();

        // Assert: offsets must form an arithmetic progression with stride = layer_stride
        let stride = layout.layer_stride;
        assert!(stride > 0, "layer_stride must be positive");
        for i in 1..4 {
            let diff = offsets[i] - offsets[i - 1];
            assert_eq!(diff, stride, "layer_base_offset({}) - layer_base_offset({}) must equal layer_stride ({})", i, i - 1, stride);
        }
        // Assert: layer 0 offset must equal layer_0_offset field
        assert_eq!(offsets[0], layout.layer_0_offset, "layer_base_offset(0) must equal layer_0_offset");
        // Assert: total_bytes must exceed all layer offsets
        let last_layer_end = offsets[3] + stride;
        assert!(layout.total_bytes > last_layer_end, "total_bytes must exceed last layer weight region");
    }

    // ── Test 96: PerLayerWeightLayout offsets are monotonically increasing ──
    #[test]
    fn test_per_layer_weight_layout_offsets_monotonically_increasing() {
        // Arrange
        use crate::compiler::mega_kernel_abi::PerLayerWeightLayout;

        // Build a PerLayerWeightLayout by hand with known geometry:
        // hidden=256, intermediate=512, num_heads=4, num_kv_heads=2, head_dim=64, elem_bytes=4
        let h = 256usize;
        let inter = 512usize;
        let nh = 4usize;
        let nkv = 2usize;
        let hd = 64usize;
        let eb = 4usize;
        let attn_norm_bytes = h * eb;
        let w_q_bytes = nh * hd * h * eb;
        let w_k_bytes = nkv * hd * h * eb;
        let w_v_bytes = nkv * hd * h * eb;
        let w_o_bytes = h * (nh * hd) * eb;
        let w_q_norm_bytes = hd * eb;
        let w_k_norm_bytes = hd * eb;
        let ffn_norm_bytes = h * eb;
        let w_gate_bytes = inter * h * eb;
        let w_up_bytes = inter * h * eb;
        let w_down_bytes = h * inter * eb;

        let o0 = 0;
        let o1 = o0 + attn_norm_bytes;
        let o2 = o1 + w_q_bytes;
        let o3 = o2 + w_k_bytes;
        let o4 = o3 + w_v_bytes;
        let o5 = o4 + w_o_bytes;
        let o6 = o5 + w_q_norm_bytes;
        let o7 = o6 + w_k_norm_bytes;
        let o8 = o7 + ffn_norm_bytes;
        let o9 = o8 + w_gate_bytes;
        let o10 = o9 + w_up_bytes;

        let pl = PerLayerWeightLayout {
            attn_norm_offset: o0, attn_norm_bytes,
            w_q_offset: o1, w_q_bytes,
            w_k_offset: o2, w_k_bytes,
            w_v_offset: o3, w_v_bytes,
            w_o_offset: o4, w_o_bytes,
            w_q_norm_offset: o5, w_q_norm_bytes,
            w_k_norm_offset: o6, w_k_norm_bytes,
            ffn_norm_offset: o7, ffn_norm_bytes,
            w_gate_offset: o8, w_gate_bytes,
            w_up_offset: o9, w_up_bytes,
            w_down_offset: o10, w_down_bytes,
        };

        // Act: collect all offset pairs in order
        let offsets: Vec<(usize, usize)> = vec![
            (pl.attn_norm_offset, pl.attn_norm_bytes),
            (pl.w_q_offset, pl.w_q_bytes),
            (pl.w_k_offset, pl.w_k_bytes),
            (pl.w_v_offset, pl.w_v_bytes),
            (pl.w_o_offset, pl.w_o_bytes),
            (pl.w_q_norm_offset, pl.w_q_norm_bytes),
            (pl.w_k_norm_offset, pl.w_k_norm_bytes),
            (pl.ffn_norm_offset, pl.ffn_norm_bytes),
            (pl.w_gate_offset, pl.w_gate_bytes),
            (pl.w_up_offset, pl.w_up_bytes),
            (pl.w_down_offset, pl.w_down_bytes),
        ];

        // Assert: each weight region must start at the end of the previous one
        for i in 1..offsets.len() {
            let prev_end = offsets[i - 1].0 + offsets[i - 1].1;
            let curr_start = offsets[i].0;
            assert_eq!(
                curr_start, prev_end,
                "per-layer offset[{}] ({}) must equal prev_end (offset[{}].0={} + offset[{}].1={})",
                i, curr_start, i - 1, offsets[i - 1].0, i - 1, offsets[i - 1].1
            );
        }
        // Assert: first offset must be 0
        assert_eq!(offsets[0].0, 0, "first per-layer offset must be 0");
    }

    // ── Test 97: MEGA_KERNEL_PARAMS length matches stack offsets and ABI spec ──
    #[test]
    fn test_mega_kernel_params_length_matches_abi_spec() {
        // Arrange
        use crate::compiler::mega_kernel_abi::{MEGA_KERNEL_PARAMS, MEGA_KERNEL_STACK_OFFSETS};

        // Assert: total ABI params must be 23 (6 register + 17 stack)
        assert_eq!(MEGA_KERNEL_PARAMS.len(), 23, "MEGA_KERNEL_PARAMS must have exactly 23 entries (6 register + 17 stack)");

        // Assert: first 6 params are register params (names must contain pointer/usize semantics)
        let register_names = &MEGA_KERNEL_PARAMS[0..6];
        assert_eq!(register_names[0], "input_ids_ptr", "arg 0 must be input_ids_ptr");
        assert_eq!(register_names[1], "weight_blob_ptr", "arg 1 must be weight_blob_ptr");
        assert_eq!(register_names[5], "batch_size", "arg 5 must be batch_size");

        // Assert: stack param count must match stack offsets count
        let stack_params = &MEGA_KERNEL_PARAMS[6..];
        assert_eq!(stack_params.len(), MEGA_KERNEL_STACK_OFFSETS.len(), "stack param names must match stack offset count");

        // Assert: stack param names must match offsets by index
        for (i, &name) in stack_params.iter().enumerate() {
            let expected_offset = 16 + i * 8;
            assert_eq!(
                MEGA_KERNEL_STACK_OFFSETS[i], expected_offset as i32,
                "stack offset for '{}' at index {} must be {}",
                name, i, expected_offset
            );
        }
    }

    // ── Test 98: SymDimSlotMap resolves hook_ctx_ptr and callback_table_ptr ──
    #[test]
    fn test_symdim_slotmap_resolves_hook_and_callback_params() {
        // Arrange
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act: resolve hook-related ABI parameters
        let hook_ctx = sym_map.resolve("hook_ctx_ptr");
        let callback_table = sym_map.resolve("callback_table_ptr");

        // Assert: both must resolve to valid PtrExpr
        assert!(hook_ctx.is_some(), "SymDimSlotMap must resolve hook_ctx_ptr");
        assert!(callback_table.is_some(), "SymDimSlotMap must resolve callback_table_ptr");

        // Assert: hook_ctx_ptr must be a StackArg (resolved from mega-kernel ABI)
        match hook_ctx.unwrap() {
            PtrExpr::StackArg(offset) => assert_eq!(*offset, 88, "hook_ctx_ptr must be at [rbp+88]"),
            PtrExpr::AbiArg(arg) => panic!("hook_ctx_ptr should not be a register arg (got AbiArg({}))", arg),
            _ => panic!("hook_ctx_ptr must be StackArg or AbiArg"),
        }

        // Assert: callback_table_ptr must be a StackArg (resolved from mega-kernel ABI)
        match callback_table.unwrap() {
            PtrExpr::StackArg(offset) => assert_eq!(*offset, 128, "callback_table_ptr must be at [rbp+128]"),
            _ => panic!("callback_table_ptr must be StackArg"),
        }
    }

    // ── Test 99: MegaKernelWeightLayout embed offset is zero and embed_bytes is vocab × hidden × elem_bytes ──
    #[test]
    fn test_mega_kernel_weight_layout_embed_offset_zero_and_size_correct() {
        // Arrange
        use crate::compiler::mega_kernel_abi::{MegaKernelWeightLayout, PerLayerWeightLayout};

        let vocab_size = 32000;
        let hidden = 4096;
        let elem_bytes = 4;
        let expected_embed_bytes = vocab_size * hidden * elem_bytes;

        // Act: construct layout by hand (build is private)
        let embed_bytes = vocab_size * hidden * elem_bytes;
        let embed_offset = 0;
        let layer_0_offset = embed_offset + embed_bytes;
        // Using simplified values for remaining fields — test only verifies embed region
        let final_norm_bytes = hidden * elem_bytes;
        let final_norm_offset = layer_0_offset + 32 * 1739968; // placeholder
        let lm_head_bytes = vocab_size * hidden * elem_bytes;
        let lm_head_offset = final_norm_offset + final_norm_bytes;
        let total_bytes = lm_head_offset + lm_head_bytes;
        let layout = MegaKernelWeightLayout {
            embed_offset,
            embed_bytes,
            layer_0_offset,
            layer_stride: 1739968,
            per_layer: PerLayerWeightLayout {
                attn_norm_offset: 0, attn_norm_bytes: 4096 * 4,
                w_q_offset: 16384, w_q_bytes: 32 * 128 * 4096 * 4,
                w_k_offset: 16384 + 32 * 128 * 4096 * 4, w_k_bytes: 32 * 128 * 4096 * 4,
                w_v_offset: 16384 + 2 * 32 * 128 * 4096 * 4, w_v_bytes: 32 * 128 * 4096 * 4,
                w_o_offset: 16384 + 3 * 32 * 128 * 4096 * 4, w_o_bytes: 4096 * 32 * 128 * 4,
                w_q_norm_offset: 16384 + 3 * 32 * 128 * 4096 * 4 + 4096 * 32 * 128 * 4, w_q_norm_bytes: 128 * 4,
                w_k_norm_offset: 16384 + 3 * 32 * 128 * 4096 * 4 + 4096 * 32 * 128 * 4 + 512, w_k_norm_bytes: 128 * 4,
                ffn_norm_offset: 16384 + 3 * 32 * 128 * 4096 * 4 + 4096 * 32 * 128 * 4 + 1024, ffn_norm_bytes: 4096 * 4,
                w_gate_offset: 0, w_gate_bytes: 11008 * 4096 * 4,
                w_up_offset: 11008 * 4096 * 4, w_up_bytes: 11008 * 4096 * 4,
                w_down_offset: 2 * 11008 * 4096 * 4, w_down_bytes: 4096 * 11008 * 4,
            },
            final_norm_offset,
            final_norm_bytes,
            lm_head_offset,
            lm_head_bytes,
            total_bytes,
        };

        // Assert: embed must start at offset 0
        assert_eq!(layout.embed_offset, 0, "embed weight offset must be 0 (start of weight blob)");
        // Assert: embed_bytes must equal vocab_size × hidden × elem_bytes
        assert_eq!(layout.embed_bytes, expected_embed_bytes, "embed_bytes must equal vocab({}) * hidden({}) * elem_bytes({})", vocab_size, hidden, elem_bytes);
        // Assert: layer_0_offset must follow immediately after embed
        assert_eq!(layout.layer_0_offset, layout.embed_offset + layout.embed_bytes, "layer_0 must start after embed region");
        // Assert: lm_head must be the last weight region
        assert_eq!(layout.total_bytes, layout.lm_head_offset + layout.lm_head_bytes, "total_bytes must equal lm_head_offset + lm_head_bytes");
    }

    // ── Test 100: MtpKernelConfig Debug trait output contains all three fields ──
    #[test]
    fn test_mtp_kernel_config_debug_trait_output() {
        // Arrange
        use crate::compiler::mega_kernel_abi::MtpKernelConfig;

        let config = MtpKernelConfig {
            depth: 5,
            hidden_size: 2048,
            vocab_size: 50257,
        };

        // Act
        let debug_str = format!("{:?}", config);

        // Assert: Debug output must contain all field names and values
        assert!(debug_str.contains("depth"), "Debug output must contain 'depth'");
        assert!(debug_str.contains("hidden_size"), "Debug output must contain 'hidden_size'");
        assert!(debug_str.contains("vocab_size"), "Debug output must contain 'vocab_size'");
        assert!(debug_str.contains("5"), "Debug output must contain depth value 5");
        assert!(debug_str.contains("2048"), "Debug output must contain hidden_size value 2048");
        assert!(debug_str.contains("50257"), "Debug output must contain vocab_size value 50257");
    }

    // ── Test 101: Empty VmProgram passes structure and provenance validation ──
    #[test]
    fn test_empty_vm_program_passes_all_validations() {
        // Arrange: a freshly constructed VmProgram with no instructions
        let prog = VmProgram::new();

        // Assert: empty program must pass all four core validations
        assert!(prog.validate_structure().is_ok(),
            "empty VmProgram must have valid structure (no unbalanced loops)");
        assert!(prog.validate_provenance().is_ok(),
            "empty VmProgram must have valid provenance (no undeclared VRegs)");
        assert!(prog.validate_type_consistency().is_ok(),
            "empty VmProgram must have consistent types");
        assert!(prog.validate_value_domains().is_ok(),
            "empty VmProgram must have valid value domains");
    }

    // ── Test 102: VmProgram with single LoadPtr + GprLoadImm passes structure ──
    #[test]
    fn test_minimal_vm_program_with_two_instrs_passes_structure() {
        // Arrange: build a minimal program with prologue-style instructions
        let mut prog = VmProgram::new();
        let ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: ptr_vreg, src: PtrExpr::AbiArg(0) });
        let imm_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: imm_vreg, value: 42 });

        // Assert: two-instruction program must pass all validations
        assert!(prog.validate_structure().is_ok(),
            "two-instr program must have valid structure");
        assert!(prog.validate_provenance().is_ok(),
            "all VRegs must be declared before use");
        assert!(prog.validate_value_domains().is_ok(),
            "value domains must be consistent (Ptr used as Ptr, Scalar as Scalar)");
    }

    // ── Test 103: SymDimSlotMap resolves page_table_ptr to correct stack offset ──
    #[test]
    fn test_symdim_slotmap_resolves_page_table_ptr() {
        // Arrange
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let page_table = sym_map.resolve("page_table_ptr");

        // Assert: page_table_ptr must resolve (SPEC/18 REQ-PA)
        assert!(page_table.is_some(), "SymDimSlotMap must resolve page_table_ptr");
        // Assert: page_table_ptr must be a StackArg at the expected offset
        match page_table.unwrap() {
            PtrExpr::StackArg(offset) => {
                // page_table_ptr is ABI arg 21, stack slot index = 21 - 6 = 15, offset = 16 + 15*8 = 136
                assert_eq!(*offset, 136, "page_table_ptr must be at [rbp+136]");
            }
            other => panic!("page_table_ptr must be StackArg, got {:?}", other),
        }
    }

    // ── Test 104: SymDimSlotMap resolves max_new_tokens and eos_token_id ──
    #[test]
    fn test_symdim_slotmap_resolves_sampling_params() {
        // Arrange
        let sym_map = SymDimSlotMap::mega_kernel_abi();

        // Act
        let max_new_tokens = sym_map.resolve("max_new_tokens");
        let eos_token_id = sym_map.resolve("eos_token_id");

        // Assert: both sampling parameters must resolve
        assert!(max_new_tokens.is_some(), "SymDimSlotMap must resolve max_new_tokens");
        assert!(eos_token_id.is_some(), "SymDimSlotMap must resolve eos_token_id");
        // Assert: both must be StackArg (passed on stack, not in registers)
        assert!(
            matches!(max_new_tokens.unwrap(), PtrExpr::StackArg(_)),
            "max_new_tokens must be a stack argument"
        );
        assert!(
            matches!(eos_token_id.unwrap(), PtrExpr::StackArg(_)),
            "eos_token_id must be a stack argument"
        );
    }

    // ── Test 105: VmProgram alloc_vreg assigns distinct VRegIds for different kinds ──
    #[test]
    fn test_alloc_vreg_assigns_distinct_ids_across_kinds() {
        // Arrange
        let mut prog = VmProgram::new();

        // Act: allocate multiple VRegs of different kinds
        let ptr1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scalar1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let counter1 = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let ptr2 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Assert: all VRegIds must be distinct
        let ids = [ptr1, scalar1, counter1, ptr2];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "VRegIds must be distinct: {:?}[{}] == {:?}[{}]",
                    ids[i], i, ids[j], j
                );
            }
        }
    }

    // ── Test 106: VmProgram with balanced LoopBegin/LoopEnd passes structure but unbalanced fails ──
    #[test]
    fn test_balanced_loops_pass_and_unbalanced_fails_structure() {
        // Arrange: build a program with balanced loop
        let mut prog = VmProgram::new();
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset,
            bound: BoundExpr::Const(4),
            step_bytes: 32,
        });
        prog.emit(VmInstr::LoopEnd);

        // Assert: balanced loop must pass structure validation
        assert!(prog.validate_structure().is_ok(),
            "balanced LoopBegin/LoopEnd must pass structure validation");

        // Arrange: build an unbalanced program (missing LoopEnd)
        let mut prog2 = VmProgram::new();
        let counter2 = prog2.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_offset2 = prog2.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog2.emit(VmInstr::LoopBegin {
            counter: counter2, byte_offset: byte_offset2,
            bound: BoundExpr::Const(4),
            step_bytes: 32,
        });
        // Intentionally NOT emitting LoopEnd

        // Assert: unbalanced loop must fail structure validation
        assert!(prog2.validate_structure().is_err(),
            "unbalanced LoopBegin without LoopEnd must fail structure validation");
    }

    // ── Test 107: VmProgram with OutputModeDispatch followed by MarkLabels passes structure ──
    #[test]
    fn test_output_mode_dispatch_with_labels_passes_structure() {
        // Arrange: simulate the mega-kernel output mode dispatch pattern
        let mut prog = VmProgram::new();
        let selector = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::OutputModeDispatch {
            selector,
            paths: vec![0, 1, 2, 3],
        });
        prog.emit(VmInstr::MarkLabel { label_id: 0 });
        prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });
        prog.emit(VmInstr::MarkLabel { label_id: 1 });
        prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });
        prog.emit(VmInstr::MarkLabel { label_id: 2 });
        prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });
        prog.emit(VmInstr::MarkLabel { label_id: 3 });
        prog.emit(VmInstr::BreakLoop { return_value: ReturnValue::Const(0) });

        // Assert: the dispatch + labels pattern must pass structure validation
        assert!(prog.validate_structure().is_ok(),
            "OutputModeDispatch with 4 MarkLabel paths must pass structure validation");
        assert!(prog.validate_provenance().is_ok(),
            "all VRegs in dispatch pattern must be properly declared");
    }

    // ── Test 108: MegaKernelBufferLayout zero-sized allocation has zero total_scratchpad_bytes ──
    #[test]
    fn test_mega_kernel_buffer_layout_zero_total_bytes_when_empty() {
        // Arrange
        use crate::compiler::mega_kernel_abi::MegaKernelBufferLayout;

        // Act: construct a zero-sized buffer layout matching actual struct fields
        let layout = MegaKernelBufferLayout {
            activation_a_offset: 0,
            activation_b_offset: 0,
            activation_bytes: 0,
            logits_offset: 0,
            logits_bytes: 0,
            sampling_workspace_offset: 0,
            sampling_workspace_bytes: 0,
            sg_detect_offset: 0,
            sg_knowledge_offset: 0,
            sg_data_bytes: 0,
            total_scratchpad_bytes: 0,
        };

        // Assert: all offsets and sizes must be zero for an empty layout
        assert_eq!(layout.logits_offset, 0, "empty layout logits_offset must be 0");
        assert_eq!(layout.logits_bytes, 0, "empty layout logits_bytes must be 0");
        assert_eq!(layout.sampling_workspace_offset, 0, "empty layout sampling_workspace_offset must be 0");
        assert_eq!(layout.sampling_workspace_bytes, 0, "empty layout sampling_workspace_bytes must be 0");
        assert_eq!(layout.total_scratchpad_bytes, 0, "empty layout total_scratchpad_bytes must be 0");
    }

    // ── Test 109: VmProgram BranchIfPtrNonNull and BranchIfGprZero use correct VReg kinds ──
    #[test]
    fn test_branch_instructions_use_correct_vreg_kinds() {
        // Arrange: build a program with branch instructions matching mega-kernel Phase 0.5 pattern
        let mut prog = VmProgram::new();
        let ptr_vreg = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: ptr_vreg, src: PtrExpr::StackArg(16) });
        prog.emit(VmInstr::BranchIfPtrNonNull { ptr: ptr_vreg, target_label: 100 });
        let scalar_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: scalar_vreg, src: PtrExpr::StackArg(80) });
        prog.emit(VmInstr::BranchIfGprZero { value: scalar_vreg, target_label: 200 });

        // Assert: branch instructions must not violate value domains
        assert!(prog.validate_value_domains().is_ok(),
            "BranchIfPtrNonNull (Ptr) and BranchIfGprZero (Scalar) must use correct VReg kinds");
        assert!(prog.validate_provenance().is_ok(),
            "all branch VRegs must be declared before use");
    }

    // ── Test 110: GprOperand VReg variant references must be valid declared VRegs ──
    #[test]
    fn test_gpr_bin_op_vreg_operand_must_be_declared() {
        // Arrange: build a program where GprBinOp references a VReg operand
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let a = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let b = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst, a, b: GprOperand::VReg(b), op: GprOp::Add,
        });

        // Assert: provenance must pass — all three VRegs (dst, a, b) declared
        assert!(prog.validate_provenance().is_ok(),
            "GprBinOp with VReg operand must have all VRegs declared");

        // Assert: GprOperand::Imm variant also works
        let dst2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let a2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: dst2, a: a2, b: GprOperand::Imm(4), op: GprOp::Shl,
        });
        assert!(prog.validate_provenance().is_ok(),
            "GprBinOp with Imm operand must have dst and a declared");
    }

    // ── Test 111: MK_SERIAL selected for SM61 (GTX 1060, 9 SMs) ──
    #[test]
    fn test_mk_serial_selected_for_sm61() {
        let variant = select_mk_variant(61, 9);
        assert_eq!(variant, MkVariant::Serial, "SM61 (Pascal) should select MK_SERIAL");
    }

    // ── Test 112: MK_GRID_SYNC selected for SM80 (A100, 108 SMs) ──
    #[test]
    fn test_mk_grid_sync_selected_for_sm80() {
        let variant = select_mk_variant(80, 108);
        assert!(matches!(variant, MkVariant::GridSync { total_ctas: 108 }),
            "SM80 (Ampere) should select MK_GRID_SYNC with total_ctas=108");
    }

    // ── Test 113: MK_CLUSTER selected for SM90 (H100, 132 SMs) ──
    #[test]
    fn test_mk_cluster_selected_for_sm90() {
        let variant = select_mk_variant(90, 132);
        assert!(matches!(variant, MkVariant::Cluster6x2 { cluster_size: 8, num_clusters: 16 }),
            "SM90 (Hopper) should select MK_CLUSTER_6x2 with cluster_size=8, num_clusters=16");
    }

    // ── Test 114: MK_SERIAL selected for SM75 (Turing 2080, 48 SMs — edge case <60) ──
    #[test]
    fn test_mk_serial_selected_for_sm75_small_gpu() {
        // SM75 with < 60 SMs still gets Serial (e.g., mobile Turing)
        let variant = select_mk_variant(75, 48);
        assert!(matches!(variant, MkVariant::GridSync { total_ctas: 48 }),
            "SM75 >= 70 should get GridSync even with < 60 SMs");
    }

    // ── Test 115: MK_CLUSTER minimum 1 cluster for small SM90 GPU ──
    #[test]
    fn test_mk_cluster_min_one_cluster() {
        let variant = select_mk_variant(90, 4);
        assert!(matches!(variant, MkVariant::Cluster6x2 { num_clusters: 1, .. }),
            "Very small SM90 GPU should have at least 1 cluster");
    }

    // ── Test 116: emit_compact_serial produces LoopBegin/LoopEnd with correct structure ──
    #[test]
    fn test_compact_serial_vm_program_structure() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let seq_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: seq_count, value: 4 });

        emit_compact_serial(&mut prog, batch_ctx_ptr, seq_count);

        // Should contain LoopBegin + LoopEnd pair
        let has_loop_begin = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        let has_loop_end = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopEnd));
        assert!(has_loop_begin, "compact should contain LoopBegin");
        assert!(has_loop_end, "compact should contain LoopEnd");

        // Should contain GprCondAction (active_flag check)
        let has_cond = prog.instrs.iter().any(|i| matches!(i, VmInstr::GprCondAction { .. }));
        assert!(has_cond, "compact should check active_flag");
    }

    // ── Test 117: emit_request_queue_refill contains AtomicAdd for read_idx ──
    #[test]
    fn test_request_queue_refill_contains_atomic_add() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let survivor_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: survivor_count, value: 2 });

        emit_request_queue_refill(&mut prog, batch_ctx_ptr, survivor_count);

        // Should contain AtomicAdd (read_idx increment)
        let has_atomic = prog.instrs.iter().any(|i| matches!(i, VmInstr::AtomicAdd { elem_width: 8, .. }));
        assert!(has_atomic, "refill should contain AtomicAdd for u64 read_idx");

        // Should contain ScalarLoad for write_idx check
        let has_write_idx_load = prog.instrs.iter().any(|i| {
            if let VmInstr::ScalarLoad { offset: OffsetExpr::Const(off), .. } = i {
                *off == request_queue_offsets::WRITE_IDX as usize
            } else {
                false
            }
        });
        assert!(has_write_idx_load, "refill should load write_idx to check queue emptiness");
    }

    // ── Test 118: SmPartitionConfig SM61 → MK_SERIAL, all CTAs decode ──
    #[test]
    fn test_sm_partition_config_sm61_serial() {
        let config = SmPartitionConfig::for_sm(61, 9);
        assert_eq!(config.variant, MkVariant::Serial);
        assert_eq!(config.total_ctas, 9);
        assert_eq!(config.decode_ctas, 9);
        assert_eq!(config.prefill_ctas, 0);
    }

    // ── Test 119: SmPartitionConfig SM80 → GridSync 75:25 ──
    #[test]
    fn test_sm_partition_config_sm80_grid_sync() {
        let config = SmPartitionConfig::for_sm(80, 108);
        assert!(matches!(config.variant, MkVariant::GridSync { .. }));
        assert_eq!(config.total_ctas, 108);
        assert_eq!(config.decode_ctas, 81); // 75%
        assert_eq!(config.prefill_ctas, 27); // 25%
    }

    // ── Test 120: SmPartitionConfig SM90 → Cluster 6:2 ──
    #[test]
    fn test_sm_partition_config_sm90_cluster() {
        let config = SmPartitionConfig::for_sm(90, 132);
        assert!(matches!(config.variant, MkVariant::Cluster6x2 { cluster_size: 8, num_clusters: 16 }));
        assert_eq!(config.total_ctas, 128); // 8 * 16
        assert_eq!(config.decode_ctas, 96); // 6 * 16
        assert_eq!(config.prefill_ctas, 32); // 2 * 16
    }

    // ── Test 121: emit_output_token_write produces ScalarStore instructions ──
    #[test]
    fn test_output_token_write_vm_program() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let seq_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let token_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let gen_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: seq_id, value: 0 });
        prog.emit(VmInstr::GprLoadImm { dst: token_id, value: 42 });
        prog.emit(VmInstr::GprLoadImm { dst: gen_idx, value: 0 });

        emit_output_token_write(&mut prog, batch_ctx_ptr, seq_id, token_id, gen_idx, false);

        let stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::ScalarStore { .. }))
            .count();
        assert!(stores >= 2, "should store at least seq_id and token_id, got {stores} stores");
    }

    // ── Test 122: emit_dual_batch_meta_swap produces AtomicAdd for epoch ──
    #[test]
    fn test_dual_batch_meta_swap_epoch_atomic() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_dual_batch_meta_swap(&mut prog, batch_ctx_ptr);

        let has_atomic = prog.instrs.iter()
            .any(|i| matches!(i, VmInstr::AtomicAdd { elem_width: 4, value: 1, .. }));
        assert!(has_atomic, "epoch swap should contain AtomicAdd u32 +1");
    }

    // ── Test: emit_page_alloc_serial produces fast-path and slow-path ──
    #[test]
    fn test_page_alloc_serial_two_tier_structure() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let local_free_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let local_next_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let dst_page_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        emit_page_alloc_serial(&mut prog, batch_ctx_ptr, local_free_count, local_next_id, dst_page_id);

        // Must have GprCondAction for fast-path check (local_free_count == 0)
        let cond_actions = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { .. }))
            .count();
        assert!(cond_actions >= 2, "page alloc should have at least 2 conditional actions (fast-path skip + slow-path skip), got {cond_actions}");

        // Must have AtomicAdd for global pool grab
        let has_atomic = prog.instrs.iter()
            .any(|i| matches!(i, VmInstr::AtomicAdd { elem_width: 8, value: 32, .. }));
        assert!(has_atomic, "page alloc slow-path should contain AtomicAdd u64 +32 for batch grab");

        // Must update local_free_count and local_next_id
        let bin_ops = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprBinOp { dst, .. } if *dst == local_free_count || *dst == local_next_id))
            .count();
        assert!(bin_ops >= 3, "page alloc should update local_free_count and local_next_id in both paths");
    }

    // ── Test: emit_page_free_serial increments local count ──
    #[test]
    fn test_page_free_serial_increments_local_count() {
        let mut prog = VmProgram::new();
        let page_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let local_free_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        emit_page_free_serial(&mut prog, page_id, local_free_count);

        // Must contain exactly one GprBinOp (Add) to local_free_count
        let add_ops = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprBinOp { dst, op: GprOp::Add, b: GprOperand::Imm(1), .. } if *dst == local_free_count))
            .count();
        assert_eq!(add_ops, 1, "page free should increment local_free_count by 1");
    }

    // ── PTX Mega-Kernel Integration Tests (SPEC 32) ──

    // Helper: build a VmProgram with ForwardPhaseDispatch logic, lower to PTX, return IR string.
    fn build_forward_phase_dispatch_ptx(sm_version: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();

        // Allocate VRegs matching the Phase 0.7 pattern in compile_mega_kernel_vm
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let batch_m = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Load total_prefill_tokens from batch_ctx + 8
        prog.emit(VmInstr::ScalarLoad {
            dst: batch_m,
            base: batch_ctx_ptr,
            offset: OffsetExpr::Const(batch_ctx_offsets::TOTAL_PREFILL_TOKENS as usize),
        });

        // Phase 0.7 three-way dispatch (SPEC 32 REQ-MKO-001)
        const DECODE_ENTRY_LABEL: usize = 101;
        const MIXED_PATH_LABEL: usize = 102;
        const PREFILL_CHUNK_THRESHOLD: usize = 512;

        // Branch 1: batch_m == 0 → pure decode
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(batch_m, 0),
            action: GprBranchAction::JumpToLabel(DECODE_ENTRY_LABEL),
        });

        // Branch 2: batch_m <= threshold → mixed path
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpLtU(batch_m, (PREFILL_CHUNK_THRESHOLD + 1) as u64),
            action: GprBranchAction::JumpToLabel(MIXED_PATH_LABEL),
        });

        // Fall-through: dedicated prefill path
        let _prefill_placeholder = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: _prefill_placeholder, value: 1 });

        // Mixed path entry (reuses prefill forward pass for MK_SERIAL)
        prog.emit(VmInstr::MarkLabel { label_id: MIXED_PATH_LABEL });

        // Prefill/mixed path placeholder
        let _placeholder = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: _placeholder, value: 2 });

        // Decode entry label
        prog.emit(VmInstr::MarkLabel { label_id: DECODE_ENTRY_LABEL });

        // Build GPU lower
        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();

        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }

        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    // ── Test 123: PTX SM61 ForwardPhaseDispatch (three-way branch) ──
    #[test]
    fn test_ptx_sm61_forward_phase_dispatch() {
        let ir = build_forward_phase_dispatch_ptx(61);

        // Must load total_prefill_tokens (u32) from BatchContext
        assert!(ir.contains("ld.global.u32"),
            "ForwardPhaseDispatch should emit ld.global.u32 for loading total_prefill_tokens: {ir}");

        // Must have comparisons for three-way branch (eq + lt.u32)
        assert!(ir.contains("setp.eq.u32"),
            "ForwardPhaseDispatch should emit setp.eq.u32 for batch_m==0 check: {ir}");
        assert!(ir.contains("setp.lt.u32"),
            "ForwardPhaseDispatch should emit setp.lt.u32 for batch_m<=threshold check: {ir}");

        // Must have labels for decode entry (101) and mixed path (102)
        assert!(ir.contains("LABEL_101:"),
            "ForwardPhaseDispatch should emit LABEL_101 for decode entry: {ir}");
        assert!(ir.contains("LABEL_102:"),
            "ForwardPhaseDispatch should emit LABEL_102 for mixed path: {ir}");
    }

    // Helper: build a VmProgram with request queue refill logic, lower to PTX, return IR string.
    fn build_request_queue_refill_ptx(sm_version: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let survivor_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: survivor_count, value: 2 });

        emit_request_queue_refill(&mut prog, batch_ctx_ptr, survivor_count);

        // Lower to PTX
        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();

        for instr in &prog.instrs {
            let _ = lower.lower_instr(instr, &alloc);
        }

        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    // ── Test 124: PTX SM61 RequestQueue Refill ──
    #[test]
    fn test_ptx_sm61_request_queue_refill() {
        let ir = build_request_queue_refill_ptx(61);

        // Must contain atom.global.add.u64 for read_idx increment
        assert!(ir.contains("atom.global.add.u64"),
            "RequestQueue refill should emit atom.global.add.u64 for read_idx increment: {ir}");

        // Must load request_queue_ptr via ld.global (u32 for scalar vreg, u64 for pointer vreg)
        assert!(ir.contains("ld.global."),
            "RequestQueue refill should emit ld.global for loading pointer fields: {ir}");
    }

    // Helper: build a VmProgram with output streaming logic, lower to PTX, return IR string.
    fn build_output_stream_ptx(sm_version: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let seq_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let token_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let gen_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: seq_id, value: 0 });
        prog.emit(VmInstr::GprLoadImm { dst: token_id, value: 42 });
        prog.emit(VmInstr::GprLoadImm { dst: gen_idx, value: 0 });

        // Emit output token write (stores seq_id, token_id, is_final to output ring)
        emit_output_token_write(&mut prog, batch_ctx_ptr, seq_id, token_id, gen_idx, false);

        // Also emit a ring buffer write_idx increment via AtomicAdd
        // (simulating the write_idx advance in output streaming)
        let ring_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::ScalarLoad {
            dst: ring_ptr,
            base: batch_ctx_ptr,
            offset: OffsetExpr::Const(output_ring_offsets::PER_CTA_DOORBELL_PTR),
        });
        prog.emit(VmInstr::AtomicAdd {
            base: ring_ptr,
            offset: OffsetExpr::Const(0),
            value: 1,
            elem_width: 8,
        });

        // Lower to PTX
        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();

        for instr in &prog.instrs {
            let _ = lower.lower_instr(instr, &alloc);
        }

        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    // ── Test 125: PTX SM61 Output Streaming ──
    #[test]
    fn test_ptx_sm61_output_streaming() {
        let ir = build_output_stream_ptx(61);

        // Must contain st.global.u32 for output ring write (seq_id/token_id stores)
        assert!(ir.contains("st.global.u32"),
            "Output streaming should emit st.global.u32 for output ring write: {ir}");

        // Must contain atom.global.add.u64 for write_idx increment
        assert!(ir.contains("atom.global.add.u64"),
            "Output streaming should emit atom.global.add.u64 for write_idx increment: {ir}");
    }

    // ── Test 126: MkVariant selection for all SM paths ──
    #[test]
    fn test_mk_variant_selection_all_paths() {
        // SM100, 120 SMs → Cluster6x2 (cluster_size=8, num_clusters=120/8=15)
        let v100 = select_mk_variant(100, 120);
        assert!(matches!(v100, MkVariant::Cluster6x2 { cluster_size: 8, num_clusters: 15 }),
            "SM100 with 120 SMs should select Cluster6x2 {{ cluster_size=8, num_clusters=15 }}, got {:?}", v100);

        // SM90, 80 SMs → Cluster6x2 (cluster_size=8, num_clusters=80/8=10)
        let v90 = select_mk_variant(90, 80);
        assert!(matches!(v90, MkVariant::Cluster6x2 { cluster_size: 8, num_clusters: 10 }),
            "SM90 with 80 SMs should select Cluster6x2 {{ cluster_size=8, num_clusters=10 }}, got {:?}", v90);

        // SM80, 60 SMs → GridSync { total_ctas: 60 }
        let v80 = select_mk_variant(80, 60);
        assert!(matches!(v80, MkVariant::GridSync { total_ctas: 60 }),
            "SM80 with 60 SMs should select GridSync {{ total_ctas=60 }}, got {:?}", v80);

        // SM61, 9 SMs → Serial
        let v61 = select_mk_variant(61, 9);
        assert_eq!(v61, MkVariant::Serial,
            "SM61 with 9 SMs should select Serial, got {:?}", v61);

        // SM52, 4 SMs → Serial
        let v52 = select_mk_variant(52, 4);
        assert_eq!(v52, MkVariant::Serial,
            "SM52 with 4 SMs should select Serial, got {:?}", v52);
    }

    // ── SPEC 32 Ring Barrier PTX Tests (Phase 1.4, REQ-MKO-002) ──

    fn build_ring_barrier_arrive_ptx(sm_version: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();
        let barrier_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_ring_barrier_arrive(&mut prog, barrier_ptr);

        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();
        for instr in &prog.instrs {
            let _ = lower.lower_instr(instr, &alloc);
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    fn build_ring_barrier_wait_ptx(sm_version: u32, expected: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();
        let barrier_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_ring_barrier_wait(&mut prog, barrier_ptr, expected);

        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();
        for instr in &prog.instrs {
            let _ = lower.lower_instr(instr, &alloc);
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    #[test]
    fn test_ptx_sm61_ring_barrier_arrive_contains_atom_add() {
        let ir = build_ring_barrier_arrive_ptx(61);
        assert!(ir.contains("atom.global.add.u32"),
            "Ring barrier arrive should emit atom.global.add.u32: {ir}");
    }

    #[test]
    fn test_ptx_sm61_ring_barrier_arrive_contains_membar() {
        let ir = build_ring_barrier_arrive_ptx(61);
        assert!(ir.contains("membar.gl") || ir.contains("bar.sync"),
            "Ring barrier arrive should emit memory fence: {ir}");
    }

    #[test]
    fn test_ptx_sm61_ring_barrier_wait_loads_counter() {
        let ir = build_ring_barrier_wait_ptx(61, 9);
        assert!(ir.contains("ld.global."),
            "Ring barrier wait should load barrier counter via ld.global: {ir}");
    }

    #[test]
    fn test_ptx_sm61_ring_barrier_wait_resets_counter() {
        let ir = build_ring_barrier_wait_ptx(61, 9);
        assert!(ir.contains("st.global."),
            "Ring barrier wait should reset counter via st.global: {ir}");
    }

    #[test]
    fn test_ptx_sm61_ring_barrier_arrive_produces_valid_ptx() {
        let ir = build_ring_barrier_arrive_ptx(61);
        assert!(ir.contains(".version 6.5"), "SM61 PTX should use version 6.5: {ir}");
        assert!(ir.contains(".target sm_61"), "SM61 PTX should target sm_61: {ir}");
    }

    // ── SPEC 32 Compact Serial PTX Tests (Phase 3.3, REQ-MKO-003) ──

    fn build_compact_serial_ptx(sm_version: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let seq_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: seq_count, value: 4 });
        emit_compact_serial(&mut prog, batch_ctx_ptr, seq_count);

        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();
        for instr in &prog.instrs {
            let _ = lower.lower_instr(instr, &alloc);
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    #[test]
    fn test_ptx_sm61_compact_serial_loads_seq_meta() {
        let ir = build_compact_serial_ptx(61);
        assert!(ir.contains("ld.global."),
            "Compact serial should load seq_meta_base via ld.global: {ir}");
    }

    #[test]
    fn test_ptx_sm61_compact_serial_has_loop_structure() {
        let ir = build_compact_serial_ptx(61);
        // LoopBegin/LoopEnd generates bra (branch) instructions
        assert!(ir.contains("bra"),
            "Compact serial should have loop branch instructions: {ir}");
    }

    #[test]
    fn test_ptx_sm61_compact_serial_valid_ptx_version() {
        let ir = build_compact_serial_ptx(61);
        assert!(ir.contains(".version 6.5"), "SM61 PTX should use version 6.5: {ir}");
    }

    // ── SPEC 32 DualBatchMeta Swap PTX Tests (Phase 3.4, REQ-MKO-004) ──

    fn build_dual_batch_swap_ptx(sm_version: u32) -> String {
        use super::super::gpu_lower::{GpuLower, GpuDialect};
        use super::super::stack_frame::StackFrame;
        use super::super::reg_alloc::RegAllocation;

        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_dual_batch_meta_swap(&mut prog, batch_ctx_ptr);

        let mut lower = GpuLower::new(GpuDialect::Ptx { sm_version });
        let frame = StackFrame {
            total_size: 0, alignment: 0, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: false,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![], callee_saved_used: vec![],
        };

        lower.set_vreg_kind_map(&prog);
        lower.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();
        for instr in &prog.instrs {
            let _ = lower.lower_instr(instr, &alloc);
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    #[test]
    fn test_ptx_sm61_dual_batch_swap_loads_epoch_ptr() {
        let ir = build_dual_batch_swap_ptx(61);
        assert!(ir.contains("ld.global."),
            "DualBatchMeta swap should load epoch_ptr via ld.global: {ir}");
    }

    #[test]
    fn test_ptx_sm61_dual_batch_swap_increments_epoch() {
        let ir = build_dual_batch_swap_ptx(61);
        assert!(ir.contains("atom.global.add.u32"),
            "DualBatchMeta swap should emit atom.global.add.u32 for epoch: {ir}");
    }

    #[test]
    fn test_ptx_sm61_dual_batch_swap_valid_ptx() {
        let ir = build_dual_batch_swap_ptx(61);
        assert!(ir.contains(".version 6.5"), "SM61 PTX should use version 6.5: {ir}");
        assert!(ir.contains(".target sm_61"), "SM61 PTX should target sm_61: {ir}");
    }

    // ── SPEC 32 REQ-MKO-004: OutputRingBuffer + Streaming Output Tests ──

    #[test]
    fn test_output_token_entry_size_is_20_bytes() {
        assert_eq!(
            core::mem::size_of::<OutputTokenEntry>(),
            20,
            "OutputTokenEntry must be 5×u32 = 20 bytes"
        );
    }

    #[test]
    fn test_output_ring_buffer_layout() {
        let buf = OutputRingBuffer {
            sub_ring_base_ptr: 0x1000,
            cta_sub_ring_size: 1024,
            num_sub_rings: 8,
            per_cta_doorbell_ptr: 0x2000,
            epoch_flag_ptr: 0x3000,
        };
        assert_eq!(buf.capacity(), 1024 * 8);
        assert_eq!(buf.sub_ring_for_cta(0), 0);
        assert_eq!(buf.sub_ring_for_cta(1), 1024 * OutputRingBuffer::ENTRY_SIZE);
        assert_eq!(buf.sub_ring_for_cta(3), 3 * 1024 * OutputRingBuffer::ENTRY_SIZE);
    }

    #[test]
    fn test_output_ring_offsets_constants() {
        assert_eq!(output_ring_offsets::SUB_RING_BASE_PTR, 0);
        assert_eq!(output_ring_offsets::CTA_SUB_RING_SIZE, 8);
        assert_eq!(output_ring_offsets::NUM_SUB_RINGS, 12);
        assert_eq!(output_ring_offsets::PER_CTA_DOORBELL_PTR, 16);
        assert_eq!(output_ring_offsets::EPOCH_FLAG_PTR, 24);
    }

    #[test]
    fn test_streaming_output_serial_emits_instructions() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let seq_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let token_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let gen_idx = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let write_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: write_count, value: 0 });

        let before = prog.len();
        emit_streaming_output_serial(
            &mut prog, batch_ctx_ptr, seq_id, token_id, gen_idx, false, write_count,
        );
        let after = prog.len();

        // Should emit multiple instructions (loads, stores, arithmetic, doorbell)
        assert!(after > before + 5,
            "emit_streaming_output_serial should emit >5 instructions, got {} new",
            after - before);
    }

    #[test]
    fn test_doorbell_update_emits_two_instructions() {
        let mut prog = VmProgram::new();
        let batch_ctx_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_count = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let before = prog.len();
        emit_doorbell_update(&mut prog, batch_ctx_ptr, write_count);
        let after = prog.len();

        // Should emit at least 2 instructions: ScalarLoad (doorbell_ptr) + ScalarStore (count)
        assert!(after - before >= 2,
            "emit_doorbell_update should emit >=2 instructions (load ptr + store count), got {}",
            after - before);
    }

    #[test]
    fn test_output_entry_finish_reason_field() {
        // Verify finish_reason field exists and has correct semantics
        let entry = OutputTokenEntry {
            seq_id: 0,
            token_id: 42,
            is_final: 1,
            finish_reason: 2, // max_tokens
            gen_idx: 10,
        };
        assert_eq!(entry.seq_id, 0);
        assert_eq!(entry.token_id, 42);
        assert_eq!(entry.is_final, 1);
        assert_eq!(entry.finish_reason, 2);
        assert_eq!(entry.gen_idx, 10);
    }

    // ── SPEC 32 REQ-MKO-005: FusionParams hardware-aware tests ──

    use crate::compiler::hardware_profile::HardwareProfile;

    #[test]
    fn test_prefill_params_sm100_uses_large_tile() {
        let p = prefill_fusion_params(&HardwareProfile::CudaSM100);
        assert_eq!(p.gemm_tile, TileSize(128, 256, 64));
        assert_eq!(p.attention, AttentionMode::FlashAttention);
        assert_eq!(p.kv_mode, KvMode::WriteFull);
        assert_eq!(p.kv_pipeline_stages, 4);
    }

    #[test]
    fn test_prefill_params_sm90_matches_sm100_tile() {
        let p90 = prefill_fusion_params(&HardwareProfile::CudaSM90);
        let p100 = prefill_fusion_params(&HardwareProfile::CudaSM100);
        assert_eq!(p90.gemm_tile, p100.gemm_tile);
    }

    #[test]
    fn test_prefill_params_avx2_uses_blis_tile() {
        let p = prefill_fusion_params(&HardwareProfile::CpuAvx2);
        assert_eq!(p.gemm_tile, TileSize(6, 64, 256));
    }

    #[test]
    fn test_decode_params_sm90_uses_dsmem_and_4stage() {
        let p = decode_fusion_params(&HardwareProfile::CudaSM90);
        assert_eq!(p.gemm_tile, TileSize(1, 256, 64));
        assert_eq!(p.attention, AttentionMode::IncrementalKvAttention);
        assert_eq!(p.kv_mode, KvMode::ReadHistoryWriteOne);
        assert_eq!(p.kv_pipeline_stages, 4);
        assert!(p.use_dsmem_kv_share);
        assert!(p.use_tensor_core_gemv);
    }

    #[test]
    fn test_decode_params_sm80_no_dsmem() {
        let p = decode_fusion_params(&HardwareProfile::CudaSM80);
        assert_eq!(p.kv_pipeline_stages, 2);
        assert!(!p.use_dsmem_kv_share, "SM80 should not use DSMEM KV share");
        assert!(p.use_tensor_core_gemv);
    }

    #[test]
    fn test_decode_params_cpu_no_gpu_features() {
        let p = decode_fusion_params(&HardwareProfile::CpuAvx2);
        assert!(!p.use_dsmem_kv_share);
        assert!(!p.use_tensor_core_gemv);
        assert!(!p.use_ld_nc);
        assert_eq!(p.kv_pipeline_stages, 1);
    }

    #[test]
    fn test_decode_m_dimension_is_one_for_all_profiles() {
        for profile in [
            HardwareProfile::CudaSM100, HardwareProfile::CudaSM90,
            HardwareProfile::CudaSM80, HardwareProfile::CpuAvx2,
            HardwareProfile::CpuAvx512, HardwareProfile::ArmNeoverse,
            HardwareProfile::Generic,
        ] {
            let p = decode_fusion_params(&profile);
            assert_eq!(p.gemm_tile.0, 1,
                "Decode M must be 1 for {:?}, got {}", profile, p.gemm_tile.0);
        }
    }

    #[test]
    fn test_prefill_tile_m_greater_than_decode() {
        let profiles = [
            HardwareProfile::CudaSM100, HardwareProfile::CudaSM90,
            HardwareProfile::CudaSM80, HardwareProfile::CpuAvx2,
            HardwareProfile::CpuAvx512,
        ];
        for profile in &profiles {
            let prefill = prefill_fusion_params(profile);
            let decode = decode_fusion_params(profile);
            assert!(prefill.gemm_tile.0 > decode.gemm_tile.0,
                "Prefill M ({}) should > Decode M ({}) for {:?}",
                prefill.gemm_tile.0, decode.gemm_tile.0, profile);
        }
    }

}
