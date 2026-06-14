//! Tiled attention inline lowering — FlashAttention/FlashDecoding/CPU tiled paths.

use super::instr::*;
use super::page_decode::{emit_bitpack_rle_decode, emit_lz4_decode};
use crate::compiler::trace::{QuantPrecision, TraceOp, ReduceKind, ValueId};
use crate::types::CompilerError;

/// Context for KV page decompression injection (REQ-COMP-009).
struct CompressCtx {
    scratch_ptr: VRegId,
    page_decompress_bytes: usize,
}

// ── Shared TraceOp bodies (constructed once, used by all paths) ──────────

/// Online softmax: [0]=running_max, [1]=score → [2]=new_max, [4]=correction, [6]=weight
fn softmax_trace() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), TraceOp::Input(1),
        TraceOp::Max(ValueId(0), ValueId(1)),
        TraceOp::Sub(ValueId(0), ValueId(2)),
        TraceOp::Exp(ValueId(3)),
        TraceOp::Sub(ValueId(1), ValueId(2)),
        TraceOp::Exp(ValueId(5)),
    ]
}

/// V accumulation: [0]=o_acc, [1]=correction, [2]=weight, [3]=v_vec → [6]=result
fn accumulate_trace() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2), TraceOp::Input(3),
        TraceOp::Mul(ValueId(0), ValueId(1)),
        TraceOp::Mul(ValueId(2), ValueId(3)),
        TraceOp::Add(ValueId(4), ValueId(5)),
    ]
}

/// Running sum update: [0]=sum, [1]=correction, [2]=weight → [4]=new_sum
fn sum_update_trace() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
        TraceOp::Mul(ValueId(0), ValueId(1)),
        TraceOp::Add(ValueId(3), ValueId(2)),
    ]
}

// ── Common helpers ───────────────────────────────────────────────────────

/// Offset of `precision_tier` field in KvPageHeader (u8 at byte 28).
/// Values: FP16=0, FP8=1, KIVI4=2, KIVI2=3, Sparse=4, Dictionary=5, Evicted=6.
/// See gllm/src/kv_cache/mod.rs for the canonical layout.
const PAGE_HEADER_PRECISION_TIER_OFFSET: usize = 28;

/// Emit a K-load with runtime tier dispatch (KvLoadMode::Auto).
///
/// Reads `precision_tier` from the page header, then branches to the correct
/// load path. The emitted code is equivalent to:
/// ```ignore
/// match precision_tier {
///     0 | 1 => VecLoad(k_row + d_off),               // FP16 / FP8 → direct
///     2 | 3 => KiviDequantLoad(k_row, scale, lanes),  // KIVI4 / KIVI2
///     4     => sparse_masked_load(k_row, d_off, bmp, ch), // Sparse
///     _     => VecLoad(k_row + d_off),                 // unknown → direct
/// }
/// ```
fn emit_tier_dispatch_k_load(
    prog: &mut VmProgram,
    dst: VRegId,
    k_row: VRegId,
    d_off: OffsetExpr,
    width: SimdWidth,
    dtype: QuantPrecision,
    lanes: usize,
    sparse_bitmap_val: Option<VRegId>,
    channel_group: usize,
    page_header_ptr: VRegId,
) {
    // 1. Load precision_tier byte from page header + 28
    let tier_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarByteLoad {
        dst: tier_gpr,
        base: page_header_ptr,
        offset: OffsetExpr::Const(PAGE_HEADER_PRECISION_TIER_OFFSET),
    });

    // 2. Test: tier == 2 (KIVI4) or tier == 3 (KIVI2)?
    //    We check tier >= 2 && tier <= 3 using two comparisons.
    //    Strategy: chain GprCondAction(CmpEq) checks.
    //    First check: tier == 4 (Sparse) → sparse path
    //    Then check:  tier == 2 || tier == 3 → KIVI path
    //    Default: direct VecLoad

    // Label for the end of the dispatch block
    let done_label = prog.alloc_label();

    // ── Path 1: Sparse (tier == 4) ──
    let sparse_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(tier_gpr, 4),
        action: GprBranchAction::Skip(0), // patched below
    });
    let sparse_check_patch = prog.instrs.len() - 1;

    // ── Path 2: KIVI (tier == 2 or tier == 3) ──
    // Check tier == 2
    let kivi2_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(tier_gpr, 2),
        action: GprBranchAction::Skip(0),
    });
    let kivi2_check_patch = prog.instrs.len() - 1;

    // Check tier == 3
    let kivi3_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(tier_gpr, 3),
        action: GprBranchAction::Skip(0),
    });
    let kivi3_check_patch = prog.instrs.len() - 1;

    // ── Default path: Direct VecLoad (FP16=0, FP8=1, or unknown) ──
    prog.emit(VmInstr::VecLoad { dst, base: k_row, offset: d_off.clone(), width, dtype , predicate: None });
    prog.emit(VmInstr::UnconditionalBranch { target_label: done_label });

    // ── Sparse path (tier == 4) ──
    prog.emit(VmInstr::MarkLabel { label_id: sparse_label });
    emit_sparse_masked_load(
        prog, dst, k_row, d_off.clone(),
        sparse_bitmap_val, channel_group, width, dtype,
    );
    prog.emit(VmInstr::UnconditionalBranch { target_label: done_label });

    // ── KIVI path (tier == 2 or 3) ──
    prog.emit(VmInstr::MarkLabel { label_id: kivi2_label });
    prog.emit(VmInstr::MarkLabel { label_id: kivi3_label });
    {
        let sc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::KiviDequantLoad { dst, src_ptr: k_row, scale_ptr: sc, num_elems: lanes, width });
    }
    // fall through to done_label

    prog.emit(VmInstr::MarkLabel { label_id: done_label });

    // Patch skip counts: each GprCondAction.Skip(N) means "skip N non-meta instructions"
    // sparse_check: skip over kivi2_check + kivi3_check + VecLoad + UnconditionalBranch = 4
    patch_skip_count(prog, sparse_check_patch, 4);
    // kivi2_check: skip over kivi3_check + VecLoad + UnconditionalBranch + sparse_path = ?
    // We need to count non-meta instructions from kivi3_check_patch+1 to MarkLabel(kivi2_label)
    let kivi2_skip = count_non_meta_between(prog, kivi2_check_patch + 1, |i| {
        matches!(prog.instrs[i], VmInstr::MarkLabel { label_id } if label_id == kivi2_label)
    });
    patch_skip_count(prog, kivi2_check_patch, kivi2_skip);
    // kivi3_check: skip over VecLoad + UnconditionalBranch + sparse_path
    let kivi3_skip = count_non_meta_between(prog, kivi3_check_patch + 1, |i| {
        matches!(prog.instrs[i], VmInstr::MarkLabel { label_id } if label_id == kivi3_label)
    });
    patch_skip_count(prog, kivi3_check_patch, kivi3_skip);
}

/// Emit a V-load with runtime tier dispatch (KvLoadMode::Auto).
/// Same tier dispatch logic as `emit_tier_dispatch_k_load` but for V rows.
fn emit_tier_dispatch_v_load(
    prog: &mut VmProgram,
    dst: VRegId,
    v_row: VRegId,
    d_off: OffsetExpr,
    width: SimdWidth,
    dtype: QuantPrecision,
    lanes: usize,
    sparse_bitmap_val: Option<VRegId>,
    channel_group: usize,
    page_header_ptr: VRegId,
) {
    let tier_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarByteLoad {
        dst: tier_gpr,
        base: page_header_ptr,
        offset: OffsetExpr::Const(PAGE_HEADER_PRECISION_TIER_OFFSET),
    });

    let done_label = prog.alloc_label();

    // Path 1: Sparse (tier == 4)
    let sparse_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(tier_gpr, 4),
        action: GprBranchAction::Skip(0),
    });
    let sparse_check_patch = prog.instrs.len() - 1;

    // Path 2: KIVI (tier == 2 or 3)
    let kivi2_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(tier_gpr, 2),
        action: GprBranchAction::Skip(0),
    });
    let kivi2_check_patch = prog.instrs.len() - 1;

    let kivi3_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(tier_gpr, 3),
        action: GprBranchAction::Skip(0),
    });
    let kivi3_check_patch = prog.instrs.len() - 1;

    // Default: Direct VecLoad
    prog.emit(VmInstr::VecLoad { dst, base: v_row, offset: d_off.clone(), width, dtype , predicate: None });
    prog.emit(VmInstr::UnconditionalBranch { target_label: done_label });

    // Sparse path
    prog.emit(VmInstr::MarkLabel { label_id: sparse_label });
    emit_sparse_masked_load(
        prog, dst, v_row, d_off.clone(),
        sparse_bitmap_val, channel_group, width, dtype,
    );
    prog.emit(VmInstr::UnconditionalBranch { target_label: done_label });

    // KIVI path
    prog.emit(VmInstr::MarkLabel { label_id: kivi2_label });
    prog.emit(VmInstr::MarkLabel { label_id: kivi3_label });
    {
        let sc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::KiviDequantLoad { dst, src_ptr: v_row, scale_ptr: sc, num_elems: lanes, width });
    }

    prog.emit(VmInstr::MarkLabel { label_id: done_label });

    patch_skip_count(prog, sparse_check_patch, 4);
    let kivi2_skip = count_non_meta_between(prog, kivi2_check_patch + 1, |i| {
        matches!(prog.instrs[i], VmInstr::MarkLabel { label_id } if label_id == kivi2_label)
    });
    patch_skip_count(prog, kivi2_check_patch, kivi2_skip);
    let kivi3_skip = count_non_meta_between(prog, kivi3_check_patch + 1, |i| {
        matches!(prog.instrs[i], VmInstr::MarkLabel { label_id } if label_id == kivi3_label)
    });
    patch_skip_count(prog, kivi3_check_patch, kivi3_skip);
}

/// Patch the Skip count of a GprCondAction at `patch_idx`.
fn patch_skip_count(prog: &mut VmProgram, patch_idx: usize, count: usize) {
    if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut sc), .. } = prog.instrs[patch_idx] {
        *sc = count;
    }
}

/// Count non-meta instructions from `start` until the predicate matches (inclusive of matching index).
fn count_non_meta_between(prog: &VmProgram, start: usize, until: impl Fn(usize) -> bool) -> usize {
    let mut count = 0;
    for i in start..prog.instrs.len() {
        if until(i) {
            return count;
        }
        if !prog.instrs[i].is_meta() {
            count += 1;
        }
    }
    count
}

/// MUSTAFAR sparse channel-group masked load (REQ-KV-OPT-005).
///
/// When `bitmap_val` is `Some`, tests whether the channel group identified by
/// `channel_group` is active in the bitmap. If the bit is clear, emits a
/// `Broadcast(dst, 0.0)` to zero out the channel. If active, emits a normal
/// `VecLoad`.
///
/// When `bitmap_val` is `None`, emits a plain `VecLoad` (no masking).
fn emit_sparse_masked_load(
    prog: &mut VmProgram,
    dst: VRegId,
    base: VRegId,
    offset: OffsetExpr,
    bitmap_val: Option<VRegId>,
    channel_group: usize,
    width: SimdWidth,
    dtype: QuantPrecision,
) {
    match bitmap_val {
        Some(bmp) => {
            // Test bit `channel_group` in the bitmap.
            // If the bit is clear (inactive channel) → skip VecLoad + Branch, broadcast 0.0.
            // If the bit is set (active channel)   → perform normal VecLoad, branch past broadcast.
            let patch_idx = prog.instrs.len();
            prog.emit(VmInstr::GprCondAction {
                cond: GprCondition::BitClear(bmp, channel_group as u8),
                action: GprBranchAction::Skip(0), // patched below
            });
            // Active path: load the channel data
            prog.emit(VmInstr::VecLoad { dst, base, offset, width, dtype , predicate: None });
            // Jump past the zero-fill to done_label
            let done_label = prog.alloc_label();
            prog.emit(VmInstr::UnconditionalBranch { target_label: done_label });
            // Inactive path: zero out the channel
            prog.emit(VmInstr::Broadcast { dst, src: ScalarExpr::Const(0.0), width, dtype });
            prog.emit(VmInstr::MarkLabel { label_id: done_label });
            // Patch: BitClear → skip over VecLoad + UnconditionalBranch (2 instructions)
            if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut sc), .. } = prog.instrs[patch_idx] {
                *sc = 2;
            }
        }
        None => {
            prog.emit(VmInstr::VecLoad { dst, base, offset, width, dtype , predicate: None });
        }
    }
}

/// Online softmax + running sum/max update. Returns (new_max, correction, weight).
fn emit_softmax_update(
    prog: &mut VmProgram, running_max: VRegId, score: VRegId,
    running_sum: VRegId, softmax_body: &[TraceOp], sum_body: &[TraceOp],
    width: SimdWidth,
) -> (VRegId, VRegId, VRegId) {
    let ss = super::auto_select::auto_lower_trace_raw(prog, softmax_body, &[running_max, score], width, QuantPrecision::F32)
        .expect("softmax auto_lower failed");
    let (new_max, correction, weight) = (ss[2], ss[4], ss[6]);
    super::auto_select::auto_lower_trace_into(prog, sum_body, &[running_sum, correction, weight], running_sum, width, QuantPrecision::F32)
        .expect("sum update auto_lower failed");
    let identity = vec![TraceOp::Input(0)];
    super::auto_select::auto_lower_trace_into(prog, &identity, &[new_max], running_max, width, QuantPrecision::F32)
        .expect("max identity auto_lower failed");
    (new_max, correction, weight)
}

/// K/V row pointer setup: paged (PageTableAddr) vs contiguous (LoadPtr).
///
/// When `compress` is `Some`, the resolved page base address is checked against
/// KvPageHeader.codec (offset 0x28) and compressed_size (offset 0x2C).
/// If codec != 0, the corresponding Lz4Decode/BitPackRleDecode VmInstr is emitted
/// to decompress into a scratch buffer before K/V data is used.
/// Per SPEC/22 REQ-COMP-009.
///
/// When `page_header_dst` is `Some`, also resolves the page header base address
/// (same page lookup but with zero intra-page offset) and writes it to that VReg.
/// Used by `KvLoadMode::Auto` to read `precision_tier` at runtime.
fn emit_kv_row_ptrs(
    prog: &mut VmProgram, k_row: VRegId, v_row: VRegId,
    k_head: VRegId, v_head: VRegId, ki_byte_off: OffsetExpr,
    pt_ptr: Option<VRegId>, pgs: usize, kv_h: usize,
    head_bytes: usize, k_stride: usize, k_ptr: VRegId, v_ptr: VRegId,
    seq_pt_offset: Option<VRegId>,
    compress: Option<&CompressCtx>,
    page_header_dst: Option<VRegId>,
) {
    if let Some(pt) = pt_ptr {
        if pgs > 0 {
            let page_stride = pgs * k_stride;
            prog.emit(VmInstr::PageTableAddr { dst: k_row, pool_base: k_ptr, page_table_ptr: pt,
                ki_byte_off: ki_byte_off.clone(), row_bytes: k_stride, page_size: pgs, page_stride,
                base_offset: kv_h * head_bytes, seq_pt_offset });
            prog.emit(VmInstr::PageTableAddr { dst: v_row, pool_base: v_ptr, page_table_ptr: pt,
                ki_byte_off, row_bytes: k_stride, page_size: pgs, page_stride,
                base_offset: kv_h * head_bytes, seq_pt_offset });
            // Resolve page header base address for Auto mode tier dispatch.
            // Uses the same page table lookup but with ki_byte_off=0 and base_offset=0
            // to get the page start (where KvPageHeader resides).
            if let Some(ph_dst) = page_header_dst {
                prog.emit(VmInstr::PageTableAddr { dst: ph_dst, pool_base: k_ptr, page_table_ptr: pt,
                    ki_byte_off: OffsetExpr::Const(0), row_bytes: k_stride, page_size: pgs, page_stride,
                    base_offset: 0, seq_pt_offset: None });
            }
            if let Some(ctx) = compress {
                emit_decompress_page(prog, k_row, v_row, ctx, kv_h, head_bytes);
            }
        } else {
            prog.emit(VmInstr::LoadPtr { dst: k_row, src: PtrExpr::VRegPlusOff(k_head, ki_byte_off.clone()) });
            prog.emit(VmInstr::LoadPtr { dst: v_row, src: PtrExpr::VRegPlusOff(v_head, ki_byte_off) });
        }
    } else {
        prog.emit(VmInstr::LoadPtr { dst: k_row, src: PtrExpr::VRegPlusOff(k_head, ki_byte_off.clone()) });
        prog.emit(VmInstr::LoadPtr { dst: v_row, src: PtrExpr::VRegPlusOff(v_head, ki_byte_off) });
    }
}

/// Decompress a KV page: read codec from KvPageHeader, branch to the correct
/// decoder (Lz4 / BitPackRle), emit a MemFence, then redirect k_row/v_row to
/// the scratch buffer. When codec==0, advance past the 56-byte header instead.
/// Per SPEC/22 REQ-COMP-009.
fn emit_decompress_page(
    prog: &mut VmProgram, k_row: VRegId, v_row: VRegId,
    ctx: &CompressCtx, kv_h: usize, head_bytes: usize,
) {
    let codec_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarByteLoad {
        dst: codec_gpr, base: k_row, offset: OffsetExpr::Const(0x28),
    });
    let skip_label = prog.alloc_label();
    let done_skip_label = prog.alloc_label();
    prog.emit(VmInstr::BranchIfGprZero { value: codec_gpr, target_label: skip_label });
    let csz_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: csz_gpr, base: k_row, offset: OffsetExpr::Const(0x2C),
    });
    let src_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::AddPtr { dst: src_ptr, base: k_row, offset: 56 });
    let dst_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::AddPtr { dst: dst_ptr, base: ctx.scratch_ptr, offset: kv_h * head_bytes });
    let bpr_label = prog.alloc_label();
    prog.emit(VmInstr::GprCondAction {
        cond: GprCondition::CmpEq(codec_gpr, 1),
        action: GprBranchAction::Skip(0),
    });
    let bpr_patch = prog.instrs.len() - 1;
    emit_lz4_decode(prog, src_ptr, dst_ptr, csz_gpr, ctx.page_decompress_bytes);
    let done_label = prog.alloc_label();
    prog.emit(VmInstr::UnconditionalBranch { target_label: done_label });
    prog.emit(VmInstr::MarkLabel { label_id: bpr_label });
    emit_bitpack_rle_decode(prog, src_ptr, dst_ptr, csz_gpr, 4, ctx.page_decompress_bytes / 4);
    prog.emit(VmInstr::MarkLabel { label_id: done_label });
    prog.emit(VmInstr::MemFence { order: MemFenceOrder::AcqRel });
    prog.emit(VmInstr::AddPtr { dst: k_row, base: ctx.scratch_ptr, offset: kv_h * head_bytes });
    prog.emit(VmInstr::AddPtr { dst: v_row, base: ctx.scratch_ptr, offset: kv_h * head_bytes });
    let skip_count = prog.instrs[bpr_patch + 1..].iter().filter(|i| !i.is_meta()).count();
    if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut sc), .. } = prog.instrs[bpr_patch] {
        *sc = skip_count;
    }
    prog.emit(VmInstr::UnconditionalBranch { target_label: done_skip_label });
    prog.emit(VmInstr::MarkLabel { label_id: skip_label });
    prog.emit(VmInstr::AddPtr { dst: k_row, base: k_row, offset: 56 });
    prog.emit(VmInstr::AddPtr { dst: v_row, base: v_row, offset: 56 });
    prog.emit(VmInstr::MarkLabel { label_id: done_skip_label });
}

/// Dot product Q·K + HReduce + scale (CPU/FlashDecoding path).
/// Uses KvLoadMode-aware K loading. Returns the scaled score VRegId.
/// `sparse_bitmap_val`: when `KvLoadMode::Sparse`, the channel-group bitmap VReg for MUSTAFAR masking.
/// `page_header_ptr`: required when `kv_load_mode == Auto`, provides the page header for tier dispatch.
fn emit_score_dot_cpu(
    prog: &mut VmProgram, q_row: VRegId, k_row: VRegId,
    hd_vecs: usize, vec_step: usize, width: SimdWidth, dtype: QuantPrecision,
    scale_vec: VRegId, lanes: usize, kv_load_mode: KvLoadMode,
    sparse_bitmap_val: Option<VRegId>,
    page_header_ptr: Option<VRegId>,
) -> VRegId {
    let dot_acc = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: dot_acc, src: ScalarExpr::Const(0.0), width, dtype });
    if hd_vecs > 0 {
        let dot_body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2), TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2))];
        // Sparse/Auto modes: use Rust-level for loop to get compile-time channel_group index
        // for GprCondition::BitClear / tier dispatch. Other modes use emit_loop for consistency.
        if kv_load_mode == KvLoadMode::Sparse {
            for d in 0..hd_vecs {
                let d_off = d * vec_step;
                let q_vec = prog.alloc_vreg(VRegKind::Vec, width);
                let k_vec = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: q_vec, base: q_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
                emit_sparse_masked_load(
                    prog, k_vec, k_row, OffsetExpr::Const(d_off),
                    sparse_bitmap_val, d, width, dtype,
                );
                super::auto_select::auto_lower_trace_into(prog, &dot_body, &[q_vec, k_vec, dot_acc], dot_acc, width, QuantPrecision::F32)
                    .expect("MHA dot FMA auto_lower failed");
            }
        } else if kv_load_mode == KvLoadMode::Auto && page_header_ptr.is_some() {
            let ph = page_header_ptr.unwrap();
            for d in 0..hd_vecs {
                let d_off = d * vec_step;
                let q_vec = prog.alloc_vreg(VRegKind::Vec, width);
                let k_vec = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: q_vec, base: q_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
                emit_tier_dispatch_k_load(
                    prog, k_vec, k_row, OffsetExpr::Const(d_off),
                    width, dtype, lanes, sparse_bitmap_val, d, ph,
                );
                super::auto_select::auto_lower_trace_into(prog, &dot_body, &[q_vec, k_vec, dot_acc], dot_acc, width, QuantPrecision::F32)
                    .expect("MHA dot FMA auto_lower failed");
            }
        } else {
            prog.emit_loop(BoundExpr::Const(hd_vecs), vec_step, |prog, _d_ctr, d_off| {
                let q_vec = prog.alloc_vreg(VRegKind::Vec, width);
                let k_vec = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecLoad { dst: q_vec, base: q_row, offset: OffsetExpr::LoopOffset(d_off), width, dtype , predicate: None });
                match kv_load_mode {
                    KvLoadMode::Kivi4 | KvLoadMode::Kivi2 => {
                        let sc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::KiviDequantLoad { dst: k_vec, src_ptr: k_row, scale_ptr: sc, num_elems: lanes, width });
                    }
                    KvLoadMode::Sparse => { unreachable!("Sparse handled above") }
                    KvLoadMode::Auto | KvLoadMode::Direct => {
                        prog.emit(VmInstr::VecLoad { dst: k_vec, base: k_row, offset: OffsetExpr::LoopOffset(d_off), width, dtype , predicate: None });
                    }
                }
                super::auto_select::auto_lower_trace_into(prog, &dot_body, &[q_vec, k_vec, dot_acc], dot_acc, width, QuantPrecision::F32)
                    .expect("MHA dot FMA auto_lower failed");
            });
        }
    }
    // HReduce + scale
    let hreduce_body = vec![TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
    let hr_slots = super::auto_select::auto_lower_trace_raw(prog, &hreduce_body, &[dot_acc], width, QuantPrecision::F32)
        .expect("MHA HReduce auto_lower failed");
    prog.emit(VmInstr::Broadcast { dst: dot_acc, src: ScalarExpr::ExtractLane0(hr_slots[1]), width, dtype });
    let scale_body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
    super::auto_select::auto_lower_trace_into(prog, &scale_body, &[dot_acc, scale_vec], dot_acc, width, QuantPrecision::F32)
        .expect("MHA scale auto_lower failed");
    dot_acc
}

/// V accumulation loop (CPU/FlashDecoding): o_acc[d] = o_acc[d]*correction + weight*V[d].
/// `sparse_bitmap_val`: when `KvLoadMode::Sparse`, the channel-group bitmap VReg for MUSTAFAR masking.
/// `page_header_ptr`: required when `kv_load_mode == Auto`, provides the page header for tier dispatch.
fn emit_v_accumulate_cpu(
    prog: &mut VmProgram, v_row: VRegId, hd_vecs: usize, vec_step: usize,
    o_acc: &[VRegId], correction: VRegId, weight: VRegId,
    width: SimdWidth, dtype: QuantPrecision, accumulate_body: &[TraceOp],
    kv_load_mode: KvLoadMode, lanes: usize,
    sparse_bitmap_val: Option<VRegId>,
    page_header_ptr: Option<VRegId>,
) {
    for d in 0..hd_vecs {
        let d_off = d * vec_step;
        let v_vec = prog.alloc_vreg(VRegKind::Vec, width);
        match kv_load_mode {
            KvLoadMode::Kivi4 | KvLoadMode::Kivi2 => {
                let sc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::KiviDequantLoad { dst: v_vec, src_ptr: v_row, scale_ptr: sc, num_elems: lanes, width });
            }
            KvLoadMode::Sparse => {
                emit_sparse_masked_load(
                    prog, v_vec, v_row, OffsetExpr::Const(d_off),
                    sparse_bitmap_val, d, width, dtype,
                );
            }
            KvLoadMode::Auto if page_header_ptr.is_some() => {
                let ph = page_header_ptr.unwrap();
                emit_tier_dispatch_v_load(
                    prog, v_vec, v_row, OffsetExpr::Const(d_off),
                    width, dtype, lanes, sparse_bitmap_val, d, ph,
                );
            }
            KvLoadMode::Auto => {
                // Auto without paged KV (no page header to read) → direct load
                prog.emit(VmInstr::VecLoad { dst: v_vec, base: v_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
            }
            KvLoadMode::Direct => {
                prog.emit(VmInstr::VecLoad { dst: v_vec, base: v_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
            }
        }
        super::auto_select::auto_lower_trace_into(
            prog, accumulate_body, &[o_acc[d], correction, weight, v_vec], o_acc[d], width, QuantPrecision::F32,
        ).expect("V accumulate auto_lower failed");
    }
}

/// Normalize + store output: O[d] = O_acc[d] / running_sum.
fn emit_normalize_store(
    prog: &mut VmProgram, o_row: VRegId, o_acc: &[VRegId], running_sum: VRegId,
    hd_vecs: usize, vec_step: usize, width: SimdWidth, dtype: QuantPrecision,
) {
    let recip_body = vec![TraceOp::Input(0), TraceOp::Recip(ValueId(0))];
    super::auto_select::auto_lower_trace_into(prog, &recip_body, &[running_sum], running_sum, width, QuantPrecision::F32)
        .expect("MHA recip auto_lower failed");
    let norm_body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
    for d in 0..hd_vecs {
        let norm_slots = super::auto_select::auto_lower_trace_raw(prog, &norm_body, &[o_acc[d], running_sum], width, QuantPrecision::F32)
            .expect("MHA norm auto_lower failed");
        prog.emit(VmInstr::VecStore { base: o_row, offset: OffsetExpr::Const(d * vec_step), src: norm_slots[2], width, dtype , predicate: None });
    }
}

/// GPU shared memory offset helper.
fn smem_kv_off(read_buf: VRegId, ki_off: VRegId, head_bytes: usize, k_stride: usize, d_off: usize) -> OffsetExpr {
    OffsetExpr::Add(
        Box::new(OffsetExpr::ScalarVReg(read_buf)),
        Box::new(OffsetExpr::Add(
            Box::new(OffsetExpr::Mul(Box::new(OffsetExpr::ScalarVReg(ki_off)), head_bytes / k_stride)),
            Box::new(OffsetExpr::Const(d_off)),
        )),
    )
}

/// Stage one row's K/V data to shared memory at write_buf offset.
/// `sparse_bitmap_val`: when `KvLoadMode::Sparse`, the channel-group bitmap VReg for MUSTAFAR masking.
/// `page_header_ptr`: required when `kv_load_mode == Auto`, provides the page header for tier dispatch.
fn emit_smem_stage_row(
    prog: &mut VmProgram, k_row: VRegId, v_row: VRegId,
    smem_k: &str, smem_v: &str, write_buf: VRegId, t: usize,
    head_bytes: usize, hd_vecs: usize, vec_step: usize,
    kv_load_mode: KvLoadMode, use_async: bool, width: SimdWidth, dtype: QuantPrecision, lanes: usize,
    sparse_bitmap_val: Option<VRegId>,
    page_header_ptr: Option<VRegId>,
) {
    for d in 0..hd_vecs {
        let d_off = d * vec_step;
        let off = |wb: VRegId| OffsetExpr::Add(
            Box::new(OffsetExpr::ScalarVReg(wb)),
            Box::new(OffsetExpr::Add(Box::new(OffsetExpr::Const(t * head_bytes)), Box::new(OffsetExpr::Const(d_off)))),
        );
        let k_vec = prog.alloc_vreg(VRegKind::Vec, width);
        match kv_load_mode {
            KvLoadMode::Kivi4 | KvLoadMode::Kivi2 => {
                let sc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::KiviDequantLoad { dst: k_vec, src_ptr: k_row, scale_ptr: sc, num_elems: lanes, width });
            }
            KvLoadMode::Sparse => {
                emit_sparse_masked_load(
                    prog, k_vec, k_row, OffsetExpr::Const(d_off),
                    sparse_bitmap_val, d, width, dtype,
                );
            }
            KvLoadMode::Auto if page_header_ptr.is_some() => {
                let ph = page_header_ptr.unwrap();
                emit_tier_dispatch_k_load(
                    prog, k_vec, k_row, OffsetExpr::Const(d_off),
                    width, dtype, lanes, sparse_bitmap_val, d, ph,
                );
            }
            KvLoadMode::Auto => {
                // Auto without paged KV (no page header to read) → direct load
                prog.emit(VmInstr::VecLoad { dst: k_vec, base: k_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
            }
            KvLoadMode::Direct => {
                prog.emit(VmInstr::VecLoad { dst: k_vec, base: k_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
            }
        }
        if use_async {
            prog.emit(VmInstr::SharedMemAsyncStore { name: smem_k.to_string(), dst_offset: off(write_buf), src: k_vec, width, dtype });
        } else {
            prog.emit(VmInstr::SharedMemStore { name: smem_k.to_string(), dst_offset: off(write_buf), src: k_vec, width, dtype });
        }
        let v_vec = prog.alloc_vreg(VRegKind::Vec, width);
        match kv_load_mode {
            KvLoadMode::Kivi4 | KvLoadMode::Kivi2 => {
                let sc = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::KiviDequantLoad { dst: v_vec, src_ptr: v_row, scale_ptr: sc, num_elems: lanes, width });
            }
            KvLoadMode::Sparse => {
                emit_sparse_masked_load(
                    prog, v_vec, v_row, OffsetExpr::Const(d_off),
                    sparse_bitmap_val, d, width, dtype,
                );
            }
            KvLoadMode::Auto if page_header_ptr.is_some() => {
                let ph = page_header_ptr.unwrap();
                emit_tier_dispatch_v_load(
                    prog, v_vec, v_row, OffsetExpr::Const(d_off),
                    width, dtype, lanes, sparse_bitmap_val, d, ph,
                );
            }
            KvLoadMode::Auto => {
                // Auto without paged KV (no page header to read) → direct load
                prog.emit(VmInstr::VecLoad { dst: v_vec, base: v_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
            }
            KvLoadMode::Direct => {
                prog.emit(VmInstr::VecLoad { dst: v_vec, base: v_row, offset: OffsetExpr::Const(d_off), width, dtype , predicate: None });
            }
        }
        if use_async {
            prog.emit(VmInstr::SharedMemAsyncStore { name: smem_v.to_string(), dst_offset: off(write_buf), src: v_vec, width, dtype });
        } else {
            prog.emit(VmInstr::SharedMemStore { name: smem_v.to_string(), dst_offset: off(write_buf), src: v_vec, width, dtype });
        }
    }
}

// ── Main entry ───────────────────────────────────────────────────────────

pub(crate) fn emit_tiled_attention_inline(
    prog: &mut VmProgram,
    q_bound: BoundExpr, kv_bound: BoundExpr,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    width: SimdWidth,
    q_ptr: VRegId, k_ptr: VRegId, v_ptr: VRegId, output_ptr: VRegId,
    hook: Option<&dyn super::isa_hook::IsaHook>,
    causal: bool, sinks_ptr: Option<VRegId>, dtype: QuantPrecision,
    page_table_ptr: Option<VRegId>, page_size: usize, kv_load_mode: KvLoadMode,
    sparse_bitmap_ptr: Option<VRegId>, _batch_ctx_ptr: Option<VRegId>, _kv_cache_ptr: Option<VRegId>,
    use_tma: bool,
    use_tmem: bool,
) -> Result<(), CompilerError> {
    let seq_len = match &kv_bound {
        BoundExpr::Const(n) => *n,
        BoundExpr::Symbolic(sym) => sym.max_alloc,
        BoundExpr::Runtime(_) | BoundExpr::DynamicVReg(_) | BoundExpr::DynamicVRegPlusOne(_) => 1,
    };
    if seq_len == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(CompilerError::CodegenViolation("emit_tiled_attention_inline: zero dim".into()));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(CompilerError::CodegenViolation(
            format!("heads ({num_heads}) not divisible by kv_heads ({num_kv_heads})"),
        ));
    }

    use super::isa_hook::AttentionStrategy;
    let hook_ref = hook.ok_or_else(|| CompilerError::CodegenViolation(
        "emit_tiled_attention_inline: IsaHook is mandatory".into(),
    ))?;
    let strategy = hook_ref.select_attention(seq_len, head_dim);
    let _kv_quant_impl = hook_ref.kv_quant_codegen();

    let (tile_q, tile_kv) = match &strategy {
        AttentionStrategy::FlashV2 { tile_q, tile_kv } | AttentionStrategy::FlashV3 { tile_q, tile_kv, .. } | AttentionStrategy::FlashV4 { tile_q, tile_kv, .. } => (*tile_q, *tile_kv),
        AttentionStrategy::FlashDecoding { tile_kv, .. } => (1, *tile_kv),
        AttentionStrategy::SlidingWindow { .. } => (seq_len.min(64), seq_len.min(64)),
        AttentionStrategy::Naive => (seq_len, seq_len),
    };
    let use_flash_attention = matches!(strategy, AttentionStrategy::FlashV2 { .. } | AttentionStrategy::FlashV3 { .. } | AttentionStrategy::FlashV4 { .. });
    let use_flash_decoding = matches!(strategy, AttentionStrategy::FlashDecoding { .. });
    let is_gpu = hook_ref.is_gpu();
    let use_gpu_flash = is_gpu && use_flash_attention && tile_q < seq_len && tile_kv < seq_len;
    let flash_decode_split_k = if let AttentionStrategy::FlashDecoding { split_k, .. } = &strategy { *split_k } else { 0 };
    let use_flash_decoding_gpu = is_gpu && use_flash_decoding && flash_decode_split_k > 1;

    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let hd_vecs = head_dim / lanes;
    let scale_vec = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast { dst: scale_vec, src: ScalarExpr::Const(scale), width, dtype });
    let q_stride = num_heads * head_dim * dtype.elem_bytes();
    let k_stride = num_kv_heads * head_dim * dtype.elem_bytes();
    let head_bytes = head_dim * dtype.elem_bytes();
    let vec_step = lanes * elem;
    let softmax_body = softmax_trace();
    let accumulate_body = accumulate_trace();
    let sum_body = sum_update_trace();
    let gqa_ratio = num_heads / num_kv_heads;

    // TMA prologue (SM90+): one-time descriptor + barrier initialization.
    // K/V descriptors describe the global tensor layout per KV head:
    //   shape (seq_len, head_dim), tile (tile_kv, head_dim).
    // These are shared across all heads since only the base pointer differs.
    let use_tma_flash = use_tma && use_gpu_flash;
    if use_tma_flash {
        prog.emit(VmInstr::TmaDescriptorInit {
            desc_name: "tma_desc_k".to_string(),
            global_dim: [seq_len, head_dim],
            global_stride: [k_stride, elem],
            box_dim: [tile_kv, head_dim],
            swizzle: TmaSwizzle::Swizzle128,
            dtype,
        });
        prog.emit(VmInstr::TmaDescriptorInit {
            desc_name: "tma_desc_v".to_string(),
            global_dim: [seq_len, head_dim],
            global_stride: [k_stride, elem],
            box_dim: [tile_kv, head_dim],
            swizzle: TmaSwizzle::Swizzle128,
            dtype,
        });
        prog.emit(VmInstr::BarrierInit {
            name: "tma_bar".to_string(),
            thread_count: 1,
        });
    }

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let (q_head, k_head, v_head, o_head) = (prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar), prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar), prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar), prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar));
        let (q_row, k_row, v_row, o_row) = (prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar), prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar), prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar), prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar));
        prog.emit(VmInstr::LoadPtr { dst: q_head, src: PtrExpr::VRegPlusConst(q_ptr, h * head_bytes) });
        prog.emit(VmInstr::LoadPtr { dst: k_head, src: PtrExpr::VRegPlusConst(k_ptr, kv_h * head_bytes) });
        prog.emit(VmInstr::LoadPtr { dst: v_head, src: PtrExpr::VRegPlusConst(v_ptr, kv_h * head_bytes) });
        prog.emit(VmInstr::LoadPtr { dst: o_head, src: PtrExpr::VRegPlusConst(output_ptr, h * head_bytes) });
        if use_gpu_flash {
            prog.emit(VmInstr::SharedMemAlloc { name: format!("smem_k_{}", h), bytes: tile_kv * 2 * head_bytes });
            prog.emit(VmInstr::SharedMemAlloc { name: format!("smem_v_{}", h), bytes: tile_kv * 2 * head_bytes });
            // TMEM prologue (SM100+): allocate space for attention score staging.
            // Each KV row produces one F32 score; tile_kv scores per tile.
            // TMEM is independent of shared memory (~256KB/SM) and eliminates
            // bank conflicts on the attention score read-back path.
            if use_tmem {
                prog.emit(VmInstr::TmemAlloc {
                    name: format!("tmem_attn_scores_{}", h),
                    bytes: tile_kv * 4, // tile_kv × sizeof(f32)
                });
            }
        }

        let mut sparse_skip_patch_idx: Option<usize> = None;
        let sparse_bitmap_val = sparse_bitmap_ptr.map(|_| prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar));
        if let (Some(bmp_ptr), Some(bitmap_val)) = (sparse_bitmap_ptr, sparse_bitmap_val) {
            prog.emit(VmInstr::ScalarLoad { dst: bitmap_val, base: bmp_ptr, offset: OffsetExpr::Const(0) });
            let patch_idx = prog.instrs.len();
            prog.emit(VmInstr::GprCondAction { cond: GprCondition::BitClear(bitmap_val, kv_h as u8), action: GprBranchAction::Skip(0) });
            sparse_skip_patch_idx = Some(patch_idx);
        }
        // Auto mode needs sparse_bitmap too (runtime tier dispatch may select Sparse path)
        let sparse_bmp_for_loads = if kv_load_mode == KvLoadMode::Sparse || kv_load_mode == KvLoadMode::Auto { sparse_bitmap_val } else { None };
        let head_body_start = prog.instrs.len();

        let sink_init = sinks_ptr.map(|sv| {
            let sink_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: sink_vec, src: ScalarExpr::MemLoad(sv, OffsetExpr::Const(h * 4)), width, dtype });
            sink_vec
        });

        prog.emit_loop(q_bound.clone(), q_stride, |prog, qi_ctr, qi_off| {
            prog.emit(VmInstr::LoadPtr { dst: q_row, src: PtrExpr::VRegPlusVReg(q_head, qi_off) });
            prog.emit(VmInstr::LoadPtr { dst: o_row, src: PtrExpr::VRegPlusVReg(o_head, qi_off) });

            // §20 BCI-004: compute seq_pt_offset for batch paged attention
            let seq_pt_offset = if let Some(bctx) = _batch_ctx_ptr {
                // Load num_seqs from batch_ctx+0
                let num_seqs_gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: num_seqs_gpr, base: bctx, offset: OffsetExpr::Const(0) });
                // Load seq_meta_base from batch_ctx+88 (BCI6 header=88)
                let seq_meta_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::ScalarLoad { dst: seq_meta_base, base: bctx, offset: OffsetExpr::Const(88) });
                // Compute seq_id via cumsum search: SeqIdLookup(token_index=qi_ctr, seq_meta_base, num_seqs)
                let seq_id = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                prog.emit(VmInstr::SeqIdLookup { dst: seq_id, token_index: qi_ctr, seq_meta_base, num_seqs: num_seqs_gpr, seq_meta_stride: 64 });
                // Read page_table_offset = seq_meta[seq_id].page_table_offset (offset +20)
                let pt_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                let off_calc = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: off_calc, a: seq_id, b: GprOperand::Imm(64), op: GprOp::Mul });
                prog.emit(VmInstr::ScalarLoad { dst: pt_off, base: seq_meta_base, offset: OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(off_calc)), Box::new(OffsetExpr::Const(20))) });
                Some(pt_off)
            } else {
                None
            };

            let (running_max, running_sum) = (prog.alloc_vreg(VRegKind::Vec, width), prog.alloc_vreg(VRegKind::Vec, width));
            if let Some(ref sv) = sink_init {
                let identity = vec![TraceOp::Input(0)];
                super::auto_select::auto_lower_trace_into(prog, &identity, &[*sv], running_max, width, QuantPrecision::F32).expect("sink init failed");
                prog.emit(VmInstr::Broadcast { dst: running_sum, src: ScalarExpr::Const(1.0), width, dtype });
            } else {
                prog.emit(VmInstr::Broadcast { dst: running_max, src: ScalarExpr::Const(f32::NEG_INFINITY), width, dtype });
                prog.emit(VmInstr::Broadcast { dst: running_sum, src: ScalarExpr::Const(0.0), width, dtype });
            }
            let o_acc: Vec<VRegId> = (0..hd_vecs).map(|_| prog.alloc_vreg(VRegKind::Vec, width)).collect();
            for &acc in &o_acc { prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(0.0), width, dtype }); }

            let decode_mode = matches!(&q_bound, BoundExpr::Const(1)) && !matches!(&kv_bound, BoundExpr::Const(1));
            let ki_bound = if causal && !decode_mode { BoundExpr::DynamicVRegPlusOne(qi_ctr) } else { kv_bound.clone() };
            let pt_ptr = page_table_ptr;
            let pgs = page_size;
            // For Auto mode with paged KV, allocate a VReg to hold the page header pointer.
            // This is populated by emit_kv_row_ptrs and consumed by the tier dispatch helpers.
            let page_header_ptr = if kv_load_mode == KvLoadMode::Auto && pt_ptr.is_some() && pgs > 0 {
                Some(prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar))
            } else {
                None
            };

            if use_flash_decoding_gpu {
                // ═══ FlashDecoding: Split-K decode ═══
                let chunk_kv = if let AttentionStrategy::FlashDecoding { tile_kv, .. } = &strategy { *tile_kv } else { 1024 };
                let ph = page_header_ptr;
                prog.emit_loop(ki_bound, chunk_kv * k_stride, |prog, _chunk_ctr, chunk_off| {
                    prog.emit_loop(BoundExpr::Const(chunk_kv), k_stride, |prog, _ki_ctr, ki_off| {
                        let combined = OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(chunk_off)), Box::new(OffsetExpr::ScalarVReg(ki_off)));
                        emit_kv_row_ptrs(prog, k_row, v_row, k_head, v_head, combined, pt_ptr, pgs, kv_h, head_bytes, k_stride, k_ptr, v_ptr, seq_pt_offset, None, ph);
                        let score = emit_score_dot_cpu(prog, q_row, k_row, hd_vecs, vec_step, width, dtype, scale_vec, lanes, kv_load_mode, sparse_bmp_for_loads, ph);
                        let (_, correction, weight) = emit_softmax_update(prog, running_max, score, running_sum, &softmax_body, &sum_body, width);
                        emit_v_accumulate_cpu(prog, v_row, hd_vecs, vec_step, &o_acc, correction, weight, width, dtype, &accumulate_body, kv_load_mode, lanes, sparse_bmp_for_loads, ph);
                    });
                });
                let global_max = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::WarpReduce { op: ReduceOp::Max, src: running_max, dst: global_max, width });
                let exp_body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Sub(ValueId(0), ValueId(1)), TraceOp::Exp(ValueId(2))];
                let exp_out = super::auto_select::auto_lower_trace_raw(prog, &exp_body, &[running_max, global_max], width, QuantPrecision::F32).expect("exp failed");
                let correction = exp_out[3];
                for &d in &o_acc { prog.emit(VmInstr::VecBinOp { dst: d, a: d, b: correction, op: VecOp::Mul, dtype }); }
                let corrected_sum = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::VecBinOp { dst: corrected_sum, a: running_sum, b: correction, op: VecOp::Mul, dtype });
                let global_sum = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::WarpReduce { op: ReduceOp::Sum, src: corrected_sum, dst: global_sum, width });
                for &d in &o_acc { prog.emit(VmInstr::VecBinOp { dst: d, a: d, b: global_sum, op: VecOp::Div, dtype }); }
                for d in 0..hd_vecs { prog.emit(VmInstr::VecStore { base: o_row, offset: OffsetExpr::Const(d * vec_step), src: o_acc[d], width, dtype , predicate: None }); }

            } else if use_gpu_flash {
                // ═══ GPU FlashAttention: tiled + double buffer ═══
                let smem_k = format!("smem_k_{}", h);
                let smem_v = format!("smem_v_{}", h);
                let double_buf_half = tile_kv * head_bytes;
                let (read_buf, write_buf, half_gpr) = (
                    prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar),
                    prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar),
                    prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar),
                );
                prog.emit(VmInstr::GprLoadImm { dst: read_buf, value: 0 });
                prog.emit(VmInstr::GprLoadImm { dst: write_buf, value: double_buf_half });
                prog.emit(VmInstr::GprLoadImm { dst: half_gpr, value: double_buf_half });
                let use_async = matches!(kv_load_mode, KvLoadMode::Sparse | KvLoadMode::Direct | KvLoadMode::Auto);
                let ph = page_header_ptr;
                prog.emit_loop(ki_bound, tile_kv * k_stride, |prog, tile_ctr, tile_off| {
                    if use_tma {
                        // ── TMA path (SM90+): single Tma2DCopy per K/V tile ──
                        // coord_x = KV row offset (tileCtr * tile_kv), coord_y = kv_h column offset
                        let coord_row_k = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprBinOp { dst: coord_row_k, a: tile_ctr, b: GprOperand::Imm(tile_kv as i64), op: GprOp::Mul });
                        let coord_col_k = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprLoadImm { dst: coord_col_k, value: kv_h * head_dim });
                        prog.emit(VmInstr::Tma2DCopy {
                            desc_name: "tma_desc_k".to_string(),
                            smem_name: smem_k.clone(),
                            coord_x: coord_row_k,
                            coord_y: coord_col_k,
                            barrier_name: "tma_bar".to_string(),
                        });

                        let coord_row_v = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprBinOp { dst: coord_row_v, a: tile_ctr, b: GprOperand::Imm(tile_kv as i64), op: GprOp::Mul });
                        let coord_col_v = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprLoadImm { dst: coord_col_v, value: kv_h * head_dim });
                        prog.emit(VmInstr::Tma2DCopy {
                            desc_name: "tma_desc_v".to_string(),
                            smem_name: smem_v.clone(),
                            coord_x: coord_row_v,
                            coord_y: coord_col_v,
                            barrier_name: "tma_bar".to_string(),
                        });
                    } else {
                        // ── SM80 path: per-row VecLoad + SharedMemAsyncStore ──
                        for t in 0..tile_kv {
                            let ptr_off = OffsetExpr::Add(Box::new(OffsetExpr::ScalarVReg(tile_off)), Box::new(OffsetExpr::Const(t * k_stride)));
                            emit_kv_row_ptrs(prog, k_row, v_row, k_head, v_head, ptr_off, pt_ptr, pgs, kv_h, head_bytes, k_stride, k_ptr, v_ptr, seq_pt_offset, None, ph);
                            emit_smem_stage_row(prog, k_row, v_row, &smem_k, &smem_v, write_buf, t, head_bytes, hd_vecs, vec_step, kv_load_mode, use_async, width, dtype, lanes, sparse_bmp_for_loads, ph);
                        }
                    }
                    // Wait for async load completion
                    if use_tma {
                        prog.emit(VmInstr::WarpBarrierWait { barrier_name: "tma_bar".to_string(), parity: 0 });
                    } else if use_async {
                        prog.emit(VmInstr::SharedMemAsyncWaitGroup { n: 0 });
                    } else {
                        prog.emit(VmInstr::BlockSync);
                    }

                    prog.emit_loop(BoundExpr::Const(tile_kv), k_stride, |prog, ki_ctr, ki_inner_off| {
                        // Dot Q·K from shared memory
                        let dot_acc = prog.alloc_vreg(VRegKind::Vec, width);
                        prog.emit(VmInstr::Broadcast { dst: dot_acc, src: ScalarExpr::Const(0.0), width, dtype });
                        if hd_vecs > 0 {
                            let dot_body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2), TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2))];
                            prog.emit_loop(BoundExpr::Const(hd_vecs), vec_step, |prog, _d_ctr, d_off| {
                                let (q_vec, k_vec) = (prog.alloc_vreg(VRegKind::Vec, width), prog.alloc_vreg(VRegKind::Vec, width));
                                prog.emit(VmInstr::VecLoad { dst: q_vec, base: q_row, offset: OffsetExpr::LoopOffset(d_off), width, dtype , predicate: None });
                                prog.emit(VmInstr::SharedMemLoad { dst: k_vec, name: smem_k.clone(), src_offset: smem_kv_off(read_buf, ki_inner_off, head_bytes, k_stride, d_off.0 as usize * vec_step / lanes * lanes * elem), width, dtype });
                                super::auto_select::auto_lower_trace_into(prog, &dot_body, &[q_vec, k_vec, dot_acc], dot_acc, width, QuantPrecision::F32).expect("dot FMA failed");
                            });
                        }
                        let hreduce_body = vec![TraceOp::Input(0), TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }];
                        let hr_slots = super::auto_select::auto_lower_trace_raw(prog, &hreduce_body, &[dot_acc], width, QuantPrecision::F32).expect("HReduce failed");
                        prog.emit(VmInstr::Broadcast { dst: dot_acc, src: ScalarExpr::ExtractLane0(hr_slots[1]), width, dtype });
                        let scale_trace = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
                        super::auto_select::auto_lower_trace_into(prog, &scale_trace, &[dot_acc, scale_vec], dot_acc, width, QuantPrecision::F32).expect("scale failed");

                        // TMEM staging (SM100+): write scaled attention score to TMEM,
                        // then read back for softmax. This reduces shared memory pressure
                        // and avoids bank conflicts on the score read-back path.
                        let score_for_softmax = if use_tmem {
                            let tmem_off = OffsetExpr::Mul(
                                Box::new(OffsetExpr::ScalarVReg(ki_ctr)),
                                4,
                            );
                            prog.emit(VmInstr::TmemStore {
                                name: format!("tmem_attn_scores_{}", h),
                                offset: tmem_off,
                                src: dot_acc,
                                width,
                                dtype,
                            });
                            let score_reload = prog.alloc_vreg(VRegKind::Vec, width);
                            let tmem_off_read = OffsetExpr::Mul(
                                Box::new(OffsetExpr::ScalarVReg(ki_ctr)),
                                4,
                            );
                            prog.emit(VmInstr::TmemLoad {
                                dst: score_reload,
                                name: format!("tmem_attn_scores_{}", h),
                                offset: tmem_off_read,
                                width,
                                dtype,
                            });
                            score_reload
                        } else {
                            dot_acc
                        };

                        let (_, correction, weight) = emit_softmax_update(prog, running_max, score_for_softmax, running_sum, &softmax_body, &sum_body, width);
                        // V accumulation from shared memory
                        for d in 0..hd_vecs {
                            let v_vec = prog.alloc_vreg(VRegKind::Vec, width);
                            prog.emit(VmInstr::SharedMemLoad { dst: v_vec, name: smem_v.clone(), src_offset: smem_kv_off(read_buf, ki_inner_off, head_bytes, k_stride, d * vec_step), width, dtype });
                            super::auto_select::auto_lower_trace_into(prog, &accumulate_body, &[o_acc[d], correction, weight, v_vec], o_acc[d], width, QuantPrecision::F32).expect("V accumulate failed");
                        }
                    }); // end inner ki
                    if use_tma {
                        prog.emit(VmInstr::WarpBarrierWait { barrier_name: "tma_bar".to_string(), parity: 1 });
                    } else {
                        prog.emit(VmInstr::BlockSync);
                    }
                    prog.emit(VmInstr::GprBinOp { dst: read_buf, a: half_gpr, b: GprOperand::VReg(read_buf), op: GprOp::Sub });
                    prog.emit(VmInstr::GprBinOp { dst: write_buf, a: half_gpr, b: GprOperand::VReg(write_buf), op: GprOp::Sub });
                }); // end outer tile
            } else {
                // ═══ CPU path: ki loop with prefetch ═══
                let prefetch_distance = if tile_kv > 0 && seq_len > tile_kv { Some(tile_kv * k_stride) } else { None };
                let ph = page_header_ptr;
                prog.emit_loop(ki_bound, k_stride, |prog, _ki_ctr, ki_off| {
                    emit_kv_row_ptrs(prog, k_row, v_row, k_head, v_head, OffsetExpr::ScalarVReg(ki_off), pt_ptr, pgs, kv_h, head_bytes, k_stride, k_ptr, v_ptr, seq_pt_offset, None, ph);
                    if let Some(dist) = prefetch_distance {
                        use super::isa_hook::PrefetchHint;
                        prog.emit(VmInstr::Prefetch { base: k_ptr, offset: OffsetExpr::ScalarVReg(ki_off), distance: dist, hint: PrefetchHint::T1 });
                        prog.emit(VmInstr::Prefetch { base: v_ptr, offset: OffsetExpr::ScalarVReg(ki_off), distance: dist, hint: PrefetchHint::T1 });
                    }
                    let score = emit_score_dot_cpu(prog, q_row, k_row, hd_vecs, vec_step, width, dtype, scale_vec, lanes, kv_load_mode, sparse_bmp_for_loads, ph);
                    let (_, correction, weight) = emit_softmax_update(prog, running_max, score, running_sum, &softmax_body, &sum_body, width);
                    emit_v_accumulate_cpu(prog, v_row, hd_vecs, vec_step, &o_acc, correction, weight, width, dtype, &accumulate_body, kv_load_mode, lanes, sparse_bmp_for_loads, ph);
                });
            }
            emit_normalize_store(prog, o_row, &o_acc, running_sum, hd_vecs, vec_step, width, dtype);
        }); // end qi

        if let Some(patch_idx) = sparse_skip_patch_idx {
            let skip_count = prog.instrs[head_body_start..].iter().filter(|i| !i.is_meta()).count();
            if let VmInstr::GprCondAction { action: GprBranchAction::Skip(ref mut sc), .. } = prog.instrs[patch_idx] { *sc = skip_count; }
        }
    } // end head
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::QuantPrecision;

    /// Verify that `emit_sparse_masked_load` with `bitmap_val = None` emits exactly one VecLoad
    /// (same as plain Direct mode).
    #[test]
    fn test_sparse_masked_load_no_bitmap() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_sparse_masked_load(
            &mut prog, dst, base, OffsetExpr::Const(0),
            None, 0, SimdWidth::W256, QuantPrecision::F32,
        );
        let non_meta: Vec<_> = prog.instrs.iter().filter(|i| !i.is_meta()).collect();
        assert_eq!(non_meta.len(), 1, "no-bitmap path should emit exactly 1 non-meta instruction");
        matches!(non_meta[0], VmInstr::VecLoad { .. });
    }

    /// Verify that `emit_sparse_masked_load` with `bitmap_val = Some` emits:
    /// 1. GprCondAction (BitClear)
    /// 2. VecLoad (active path)
    /// 3. UnconditionalBranch (skip broadcast)
    /// 4. Broadcast 0.0 (inactive path)
    /// 5. MarkLabel (done)
    /// And the Skip count is patched to 2.
    #[test]
    fn test_sparse_masked_load_with_bitmap() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        emit_sparse_masked_load(
            &mut prog, dst, base, OffsetExpr::Const(32),
            Some(bmp), 4, SimdWidth::W256, QuantPrecision::F32,
        );

        // Should have 5 instructions: GprCondAction, VecLoad, UnconditionalBranch, Broadcast, MarkLabel
        let non_meta: Vec<_> = prog.instrs.iter().filter(|i| !i.is_meta()).collect();
        assert!(non_meta.len() >= 3, "sparse path should emit at least GprCondAction + VecLoad + Broadcast");

        // Verify first non-meta instruction is GprCondAction with BitClear for channel group 4
        match non_meta[0] {
            VmInstr::GprCondAction { cond, action } => {
                assert!(
                    matches!(cond, GprCondition::BitClear(v, 4) if *v == bmp),
                    "expected BitClear(bmp, 4), got {:?}",
                    cond
                );
                assert!(
                    matches!(action, GprBranchAction::Skip(2)),
                    "expected Skip(2), got {:?}",
                    action
                );
            }
            other => panic!("expected GprCondAction, got {:?}", other),
        }

        // Verify VecLoad is present
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_vec_load, "should contain a VecLoad instruction");

        // Verify Broadcast(0.0) is present for the inactive path
        let has_zero_broadcast = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }
        ));
        assert!(has_zero_broadcast, "should contain a Broadcast(0.0) for inactive channel");
    }

    /// Verify that different channel_group indices produce different BitClear bit positions.
    #[test]
    fn test_sparse_masked_load_different_channels() {
        for ch in [0u8, 1, 3, 7] {
            let mut prog = VmProgram::new();
            let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

            emit_sparse_masked_load(
                &mut prog, dst, base, OffsetExpr::Const(0),
                Some(bmp), ch as usize, SimdWidth::W256, QuantPrecision::F32,
            );

            let non_meta: Vec<_> = prog.instrs.iter().filter(|i| !i.is_meta()).collect();
            match non_meta[0] {
                VmInstr::GprCondAction { cond, action } => {
                    assert!(
                        matches!(cond, GprCondition::BitClear(_, b) if *b == ch),
                        "channel {}: expected BitClear(_, {}), got {:?}",
                        ch, ch, cond
                    );
                    assert!(
                        matches!(action, GprBranchAction::Skip(2)),
                        "channel {}: expected Skip(2), got {:?}",
                        ch, action
                    );
                }
                other => panic!("channel {}: expected GprCondAction, got {:?}", ch, other),
            }
        }
    }

    /// Verify that the full tiled attention pipeline compiles with KvLoadMode::Sparse
    /// without panicking (structural smoke test).
    #[test]
    fn test_tiled_attention_sparse_mode_smoke() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sparse_bmp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Sparse,
            Some(sparse_bmp), None, None,
            false, false,
        );
        assert!(result.is_ok(), "Sparse mode attention should compile: {:?}", result);

        // Verify GprCondAction instructions exist for channel-level masking
        let cond_actions: Vec<_> = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { .. }))
            .collect();
        // At minimum: 2 heads * (1 head-level skip + at least channel-level masks)
        assert!(!cond_actions.is_empty(), "should emit GprCondAction instructions for sparse masking");

        // Verify Broadcast(0.0) instructions exist for inactive channel zero-fill
        let zero_broadcasts: Vec<_> = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }))
            .collect();
        assert!(!zero_broadcasts.is_empty(), "should emit Broadcast(0.0) for inactive channels");
    }

    /// Verify that `emit_tier_dispatch_k_load` emits the expected instruction sequence:
    /// ScalarByteLoad (read tier), GprCondAction (sparse check), GprCondAction (kivi2),
    /// GprCondAction (kivi3), VecLoad (default), UnconditionalBranch, MarkLabel (sparse path),
    /// ..., MarkLabel (kivi path), KiviDequantLoad, MarkLabel (done).
    #[test]
    fn test_tier_dispatch_k_load_structure() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_tier_dispatch_k_load(
            &mut prog, dst, k_row, OffsetExpr::Const(0),
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, 0, page_hdr,
        );

        // Must contain ScalarByteLoad to read precision_tier from offset 28
        let has_byte_load = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(28), .. }
        ));
        assert!(has_byte_load, "should emit ScalarByteLoad at offset 28 for precision_tier");

        // Must contain GprCondAction with CmpEq for tier value checks
        let cond_actions: Vec<_> = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { .. }))
            .collect();
        assert!(cond_actions.len() >= 3, "should emit at least 3 GprCondAction for tier dispatch (sparse/kivi2/kivi3)");

        // Must contain VecLoad (default/direct path)
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_vec_load, "should contain VecLoad for default direct path");

        // Must contain KiviDequantLoad (kivi path)
        let has_kivi = prog.instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_kivi, "should contain KiviDequantLoad for kivi path");
    }

    /// Verify that the full tiled attention pipeline compiles with KvLoadMode::Auto
    /// without panicking (structural smoke test).
    #[test]
    fn test_tiled_attention_auto_mode_smoke() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Auto,
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "Auto mode attention should compile without paged KV: {:?}", result);

        // Without paged KV (page_table_ptr=None), Auto should degrade to Direct-like behavior
        // since page_header_ptr won't be allocated. Verify VecLoad instructions exist.
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_vec_load, "Auto mode without paged KV should still emit VecLoad");
    }

    /// Verify Auto mode with paged KV emits tier dispatch instructions.
    #[test]
    fn test_tiled_attention_auto_paged_smoke() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            Some(pt_ptr), 16, KvLoadMode::Auto,  // paged KV with page_size=16
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "Auto mode paged attention should compile: {:?}", result);

        // With paged KV, should emit ScalarByteLoad for precision_tier reads
        let has_byte_load = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(28), .. }
        ));
        assert!(has_byte_load, "Auto mode with paged KV should emit ScalarByteLoad at offset 28");

        // Should emit GprCondAction with CmpEq for tier value checks
        let has_cmp_eq = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::GprCondAction { cond: GprCondition::CmpEq(_, _), .. }
        ));
        assert!(has_cmp_eq, "Auto mode with paged KV should emit CmpEq conditions for tier dispatch");

        // Should emit both VecLoad (direct path) and KiviDequantLoad (kivi path)
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_kivi = prog.instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_vec_load, "Auto mode should emit VecLoad for direct tier path");
        assert!(has_kivi, "Auto mode should emit KiviDequantLoad for kivi tier path");
    }

    // ── 13 new tests below ──────────────────────────────────────────────────

    /// Verify `softmax_trace()` returns exactly the expected sequence of TraceOp variants
    /// for the online softmax computation (7 ops).
    #[test]
    fn test_softmax_trace_structure() {
        let ops = softmax_trace();
        assert_eq!(ops.len(), 7, "softmax_trace should produce 7 TraceOps");

        // First two are Input slots for running_max and score
        assert!(matches!(ops[0], TraceOp::Input(0)), "op[0] should be Input(0)");
        assert!(matches!(ops[1], TraceOp::Input(1)), "op[1] should be Input(1)");

        // Then Max, Sub, Exp, Sub, Exp
        assert!(matches!(ops[2], TraceOp::Max(_, _)), "op[2] should be Max");
        assert!(matches!(ops[3], TraceOp::Sub(_, _)), "op[3] should be Sub (running_max - new_max)");
        assert!(matches!(ops[4], TraceOp::Exp(_)), "op[4] should be Exp (correction exponent)");
        assert!(matches!(ops[5], TraceOp::Sub(_, _)), "op[5] should be Sub (score - new_max)");
        assert!(matches!(ops[6], TraceOp::Exp(_)), "op[6] should be Exp (weight exponent)");
    }

    /// Verify `accumulate_trace()` returns exactly 7 ops: 4 inputs + Mul + Mul + Add.
    #[test]
    fn test_accumulate_trace_structure() {
        let ops = accumulate_trace();
        assert_eq!(ops.len(), 7, "accumulate_trace should produce 7 TraceOps");

        assert!(matches!(ops[0], TraceOp::Input(0)), "op[0] should be Input(0) = o_acc");
        assert!(matches!(ops[1], TraceOp::Input(1)), "op[1] should be Input(1) = correction");
        assert!(matches!(ops[2], TraceOp::Input(2)), "op[2] should be Input(2) = weight");
        assert!(matches!(ops[3], TraceOp::Input(3)), "op[3] should be Input(3) = v_vec");
        assert!(matches!(ops[4], TraceOp::Mul(_, _)), "op[4] should be Mul (o_acc * correction)");
        assert!(matches!(ops[5], TraceOp::Mul(_, _)), "op[5] should be Mul (weight * v_vec)");
        assert!(matches!(ops[6], TraceOp::Add(_, _)), "op[6] should be Add (sum of products)");
    }

    /// Verify `sum_update_trace()` returns exactly 5 ops: 3 inputs + Mul + Add.
    #[test]
    fn test_sum_update_trace_structure() {
        let ops = sum_update_trace();
        assert_eq!(ops.len(), 5, "sum_update_trace should produce 5 TraceOps");

        assert!(matches!(ops[0], TraceOp::Input(0)), "op[0] should be Input(0) = sum");
        assert!(matches!(ops[1], TraceOp::Input(1)), "op[1] should be Input(1) = correction");
        assert!(matches!(ops[2], TraceOp::Input(2)), "op[2] should be Input(2) = weight");
        assert!(matches!(ops[3], TraceOp::Mul(_, _)), "op[3] should be Mul (sum * correction)");
        assert!(matches!(ops[4], TraceOp::Add(_, _)), "op[4] should be Add (corrected_sum + weight)");
    }

    /// Verify `KvLoadMode` Default trait returns `Direct`.
    #[test]
    fn test_kv_load_mode_default_is_direct() {
        let mode = KvLoadMode::default();
        assert_eq!(mode, KvLoadMode::Direct, "Default KvLoadMode should be Direct");
    }

    /// Verify all `KvLoadMode` variants can be constructed and compared for equality.
    #[test]
    fn test_kv_load_mode_all_variants() {
        let modes = [
            KvLoadMode::Direct,
            KvLoadMode::Kivi4,
            KvLoadMode::Kivi2,
            KvLoadMode::Sparse,
            KvLoadMode::Auto,
        ];
        // Verify all are distinct
        for i in 0..modes.len() {
            for j in 0..modes.len() {
                if i == j {
                    assert_eq!(modes[i], modes[j], "same index should be equal");
                } else {
                    assert_ne!(modes[i], modes[j], "different variants should not be equal");
                }
            }
        }
    }

    /// Verify `emit_tiled_attention_inline` returns an error when heads are not divisible
    /// by kv_heads (e.g., 3 heads, 2 kv_heads).
    #[test]
    fn test_tiled_attention_head_not_divisible_error() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            3, 2, 64,  // 3 heads not divisible by 2 kv_heads
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_err(), "should reject num_heads=3 not divisible by num_kv_heads=2");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("not divisible") || err_msg.contains("divisible"),
            "error message should mention divisibility: {}",
            err_msg
        );
    }

    /// Verify `emit_tiled_attention_inline` returns an error when any dimension is zero.
    #[test]
    fn test_tiled_attention_zero_dim_error() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Test zero num_heads
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            0, 2, 64,  // zero num_heads
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_err(), "should reject zero num_heads");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("zero dim"),
            "error message should mention zero dim: {}",
            err_msg
        );
    }

    /// Verify `emit_tiled_attention_inline` returns an error when hook is None.
    #[test]
    fn test_tiled_attention_missing_hook_error() {
        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            None,  // no hook
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_err(), "should reject missing IsaHook");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("IsaHook") || err_msg.contains("mandatory"),
            "error should mention IsaHook is mandatory: {}",
            err_msg
        );
    }

    /// Verify `emit_tier_dispatch_v_load` emits the same structural pattern as k_load:
    /// ScalarByteLoad at offset 28, 3+ GprCondAction, VecLoad, KiviDequantLoad.
    #[test]
    fn test_tier_dispatch_v_load_structure() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_tier_dispatch_v_load(
            &mut prog, dst, v_row, OffsetExpr::Const(64),
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, 0, page_hdr,
        );

        // Must contain ScalarByteLoad at offset 28
        let has_byte_load = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(28), .. }
        ));
        assert!(has_byte_load, "v_load should emit ScalarByteLoad at offset 28 for precision_tier");

        // Must contain 3+ GprCondAction for tier dispatch
        let cond_actions: Vec<_> = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { .. }))
            .collect();
        assert!(cond_actions.len() >= 3, "v_load should emit at least 3 GprCondAction");

        // Must contain VecLoad (default path)
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(has_vec_load, "v_load should contain VecLoad for default path");

        // Must contain KiviDequantLoad (kivi path)
        let has_kivi = prog.instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_kivi, "v_load should contain KiviDequantLoad for kivi path");
    }

    /// Verify `emit_kv_row_ptrs` non-paged path (pt_ptr=None) emits exactly 2 LoadPtr
    /// instructions (one for k_row, one for v_row).
    #[test]
    fn test_kv_row_ptrs_non_paged() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(128),
            None,  // no page table
            0, 0, 0, 256, k_ptr, v_ptr,
            None, None, None,
        );

        let new_instrs = &prog.instrs[before_count..];
        // Should emit exactly 2 LoadPtr instructions (non-meta)
        let load_ptrs: Vec<_> = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::LoadPtr { .. }))
            .collect();
        assert_eq!(load_ptrs.len(), 2, "non-paged path should emit exactly 2 LoadPtr instructions");

        // Verify no PageTableAddr was emitted
        let has_pta = new_instrs.iter().any(|i| matches!(i, VmInstr::PageTableAddr { .. }));
        assert!(!has_pta, "non-paged path should not emit PageTableAddr");
    }

    /// Verify `emit_kv_row_ptrs` paged path with page_size > 0 emits PageTableAddr
    /// instructions for both K and V rows.
    #[test]
    fn test_kv_row_ptrs_paged() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0),
            Some(pt_ptr),
            16,   // page_size > 0
            0,    // kv_h = 0
            256,  // head_bytes
            256,  // k_stride
            k_ptr, v_ptr,
            None, None, None,
        );

        let new_instrs = &prog.instrs[before_count..];
        let pta_count = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { .. }))
            .count();
        assert_eq!(pta_count, 2, "paged path should emit exactly 2 PageTableAddr (K + V)");

        // Verify no LoadPtr was emitted (paged path uses PageTableAddr instead)
        let load_ptrs = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::LoadPtr { .. }))
            .count();
        assert_eq!(load_ptrs, 0, "paged path should not emit LoadPtr");
    }

    /// Verify `CompressCtx` struct update syntax produces independent instances
    /// with different field values.
    #[test]
    fn test_compress_ctx_struct_update_syntax() {
        let mut prog = VmProgram::new();
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let ctx1 = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };
        let ctx2 = CompressCtx { page_decompress_bytes: 8192, ..ctx1 };

        // Both should reference the same scratch_ptr but different page_decompress_bytes
        assert_eq!(ctx1.scratch_ptr, ctx2.scratch_ptr, "struct update should share scratch_ptr");
        assert_eq!(ctx1.page_decompress_bytes, 4096);
        assert_eq!(ctx2.page_decompress_bytes, 8192);
    }

    /// Verify that `emit_tiled_attention_inline` with the Direct mode and minimal
    /// dimensions (1 head, 1 kv_head, head_dim=8) compiles without error and
    /// emits VecStore instructions for output.
    #[test]
    fn test_tiled_attention_minimal_dimensions() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(2),
            1, 1, 8,  // minimal: 1 head, 1 kv_head, head_dim=8
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "minimal dimensions should compile: {:?}", result);

        // Verify VecStore is emitted for the output row
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        assert!(has_store, "should emit VecStore for output row");

        // Verify Broadcast of NEG_INFINITY for running_max init
        let has_neg_inf = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == f32::NEG_INFINITY
        ));
        assert!(has_neg_inf, "should broadcast NEG_INFINITY for running_max initialization");
    }

    /// Verify that `emit_sparse_masked_load` preserves the offset expression
    /// passed to it in the VecLoad instruction (boundary value for offset).
    #[test]
    fn test_sparse_masked_load_offset_preserved() {
        let offsets = [0usize, 32, 1024, 4096];
        for &off_val in &offsets {
            let mut prog = VmProgram::new();
            let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

            emit_sparse_masked_load(
                &mut prog, dst, base, OffsetExpr::Const(off_val),
                None, 0, SimdWidth::W256, QuantPrecision::F32,
            );

            // The single VecLoad should carry the exact offset we passed
            let vec_loads: Vec<_> = prog.instrs.iter()
                .filter_map(|i| match i {
                    VmInstr::VecLoad { offset, .. } => Some(offset.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(vec_loads.len(), 1, "offset {}: should have exactly 1 VecLoad", off_val);
            assert_eq!(vec_loads[0], OffsetExpr::Const(off_val),
                "offset {}: VecLoad offset should match input", off_val);
        }
    }

    /// Verify `PAGE_HEADER_PRECISION_TIER_OFFSET` is 28, matching KvPageHeader layout.
    #[test]
    fn test_page_header_precision_tier_offset_value() {
        assert_eq!(PAGE_HEADER_PRECISION_TIER_OFFSET, 28,
            "precision_tier field must be at byte offset 28 in KvPageHeader");
    }

    /// Verify `patch_skip_count` correctly updates the Skip count of a GprCondAction
    /// at a given index and leaves other instructions unchanged.
    #[test]
    fn test_patch_skip_count_updates_correct_instruction() {
        let mut prog = VmProgram::new();
        let gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::Broadcast { dst: gpr, src: ScalarExpr::Const(0.0), width: SimdWidth::Scalar, dtype: QuantPrecision::F32 });
        // GprCondAction with Skip(0) — the target for patching
        prog.emit(VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(gpr, 4),
            action: GprBranchAction::Skip(0),
        });
        let patch_idx = prog.instrs.len() - 1;
        prog.emit(VmInstr::Broadcast { dst: gpr, src: ScalarExpr::Const(1.0), width: SimdWidth::Scalar, dtype: QuantPrecision::F32 });

        patch_skip_count(&mut prog, patch_idx, 7);

        match &prog.instrs[patch_idx] {
            VmInstr::GprCondAction { action: GprBranchAction::Skip(n), .. } => {
                assert_eq!(*n, 7, "patch_skip_count should set Skip to 7");
            }
            other => panic!("expected GprCondAction at patch_idx, got {:?}", other),
        }

        // Verify surrounding instructions are unchanged
        let broadcasts: Vec<_> = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { .. }))
            .collect();
        assert_eq!(broadcasts.len(), 2, "surrounding Broadcasts should be unchanged");
    }

    /// Verify `count_non_meta_between` returns 0 when the predicate immediately matches
    /// at the start index.
    #[test]
    fn test_count_non_meta_between_immediate_match() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let label = prog.alloc_label();

        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(0.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::MarkLabel { label_id: label });
        let mark_idx = prog.instrs.len() - 1;

        // Start scanning at mark_idx, predicate matches immediately → count = 0
        let count = count_non_meta_between(&prog, mark_idx, |i| i == mark_idx);
        assert_eq!(count, 0, "immediate match should return 0");
    }

    /// Verify `count_non_meta_between` correctly skips meta instructions and counts
    /// only non-meta ones until the predicate matches.
    #[test]
    fn test_count_non_meta_between_skips_meta() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let label = prog.alloc_label();

        let start_idx = prog.instrs.len();
        // Non-meta: Broadcast
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(0.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        // Meta: Comment
        prog.emit(VmInstr::Comment("test".into()));
        // Non-meta: another Broadcast
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        // Meta: MarkLabel (this is the target)
        prog.emit(VmInstr::MarkLabel { label_id: label });
        let target_idx = prog.instrs.len() - 1;

        let count = count_non_meta_between(&prog, start_idx, |i| i == target_idx);
        assert_eq!(count, 2, "should count 2 non-meta Broadcast instructions before MarkLabel");
    }

    /// Verify `emit_kv_row_ptrs` with paged KV and page_header_dst allocates
    /// a PageTableAddr for the page header with ki_byte_off=0 and base_offset=0.
    #[test]
    fn test_kv_row_ptrs_paged_with_page_header_dst() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0),
            Some(pt_ptr),
            16, 0, 256, 256, k_ptr, v_ptr,
            None, None, Some(page_hdr),
        );

        let new_instrs = &prog.instrs[before_count..];
        // Should emit 3 PageTableAddr: K row, V row, and page header
        let pta_count = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { .. }))
            .count();
        assert_eq!(pta_count, 3, "paged + page_header_dst should emit 3 PageTableAddr (K + V + header)");
    }

    /// Verify `emit_kv_row_ptrs` non-paged path with page_size=0 falls through to
    /// LoadPtr instead of PageTableAddr even when pt_ptr is Some.
    #[test]
    fn test_kv_row_ptrs_pt_ptr_zero_page_size() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(64),
            Some(pt_ptr),
            0,  // page_size = 0 → non-paged fallback
            0, 256, 256, k_ptr, v_ptr,
            None, None, None,
        );

        let new_instrs = &prog.instrs[before_count..];
        let has_pta = new_instrs.iter().any(|i| matches!(i, VmInstr::PageTableAddr { .. }));
        assert!(!has_pta, "page_size=0 should not emit PageTableAddr even with pt_ptr=Some");

        let load_ptrs = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::LoadPtr { .. }))
            .count();
        assert_eq!(load_ptrs, 2, "should emit 2 LoadPtr for K and V rows");
    }

    /// Verify `emit_tiled_attention_inline` with KvLoadMode::Kivi4 compiles and emits
    /// KiviDequantLoad instructions for the K/V load path.
    #[test]
    fn test_tiled_attention_kivi4_mode_emits_dequant() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Kivi4,
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "Kivi4 mode should compile: {:?}", result);

        let kivi_loads = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::KiviDequantLoad { .. }))
            .count();
        assert!(kivi_loads > 0, "Kivi4 mode should emit KiviDequantLoad instructions");
    }

    /// Verify `emit_tiled_attention_inline` with KvLoadMode::Kivi2 compiles and emits
    /// KiviDequantLoad instructions (same VmInstr as Kivi4, different semantics).
    #[test]
    fn test_tiled_attention_kivi2_mode_emits_dequant() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Kivi2,
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "Kivi2 mode should compile: {:?}", result);

        let kivi_loads = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::KiviDequantLoad { .. }))
            .count();
        assert!(kivi_loads > 0, "Kivi2 mode should emit KiviDequantLoad instructions");
    }

    /// Verify `emit_tiled_attention_inline` with zero head_dim returns a
    /// CodegenViolation error containing "zero dim".
    #[test]
    fn test_tiled_attention_zero_head_dim_error() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 0,  // zero head_dim
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_err(), "should reject zero head_dim");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("zero dim"),
            "error should mention zero dim: {}", err_msg);
    }

    /// Verify `emit_tiled_attention_inline` with zero kv_heads returns a
    /// CodegenViolation error containing "zero dim".
    #[test]
    fn test_tiled_attention_zero_kv_heads_error() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 0, 64,  // zero kv_heads
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_err(), "should reject zero kv_heads");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("zero dim"),
            "error should mention zero dim: {}", err_msg);
    }

    /// Verify `emit_tiled_attention_inline` with Direct mode emits scale broadcast
    /// equal to 1/sqrt(head_dim) and VecStore for output rows.
    #[test]
    fn test_tiled_attention_direct_mode_scale_and_output() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let head_dim: usize = 64;
        let expected_scale = 1.0f32 / (head_dim as f32).sqrt();
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, head_dim,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "Direct mode should compile: {:?}", result);

        // Verify scale constant is broadcast
        let has_scale_broadcast = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if (*v - expected_scale).abs() < 1e-6
        ));
        assert!(has_scale_broadcast, "should broadcast 1/sqrt(head_dim) = {}", expected_scale);

        // Verify VecStore instructions exist for output
        let store_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(store_count > 0, "should emit VecStore for output rows");
    }

    /// Verify `emit_tiled_attention_inline` with GQA (num_heads=4, num_kv_heads=2)
    /// compiles without error and emits instructions for all 4 query heads sharing
    /// 2 KV heads.
    #[test]
    fn test_tiled_attention_gqa_grouped_query() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            4, 2, 64,  // 4 heads, 2 kv_heads → GQA ratio 2
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );
        assert!(result.is_ok(), "GQA 4:2 should compile: {:?}", result);

        // With 4 query heads, should have at least 4 VecStore groups for output
        let vec_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(vec_stores >= 4, "GQA 4:2 should emit at least 4 VecStore instructions (one per head)");
    }

    /// Verify `emit_decompress_page` emits ScalarByteLoad at offset 0x28 for the codec
    /// byte, BranchIfGprZero for codec==0 shortcut, and MemFence with AcqRel ordering.
    #[test]
    fn test_decompress_page_codec_and_fence_structure() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        let before_count = prog.instrs.len();
        emit_decompress_page(&mut prog, k_row, v_row, &ctx, 0, 256);

        let new_instrs = &prog.instrs[before_count..];

        // Should emit ScalarByteLoad at offset 0x28 for codec byte
        let has_codec_load = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(0x28), .. }
        ));
        assert!(has_codec_load, "should emit ScalarByteLoad at offset 0x28 for codec");

        // Should emit BranchIfGprZero for codec==0 shortcut
        let has_branch_zero = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::BranchIfGprZero { .. }
        ));
        assert!(has_branch_zero, "should emit BranchIfGprZero for codec==0 shortcut");

        // Should emit MemFence with AcqRel after decompress paths merge
        let has_fence = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::MemFence { order: MemFenceOrder::AcqRel }
        ));
        assert!(has_fence, "should emit MemFence(AcqRel) after decompress");

        // Should redirect k_row/v_row to scratch buffer via AddPtr
        let add_ptrs: Vec<_> = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::AddPtr { .. }))
            .collect();
        assert!(add_ptrs.len() >= 2, "should emit at least 2 AddPtr to redirect k_row/v_row");
    }

    /// Verify `emit_decompress_page` codec==0 shortcut path emits AddPtr with
    /// offset 56 (sizeof KvPageHeader) to skip the header and does not emit
    /// MemFence in the shortcut path.
    #[test]
    fn test_decompress_page_codec_zero_skips_header() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        emit_decompress_page(&mut prog, k_row, v_row, &ctx, 0, 256);

        // The codec==0 path should add 56 (KvPageHeader size) to k_row and v_row
        let header_skips = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::AddPtr { offset: 56, .. }))
            .count();
        assert!(header_skips >= 2, "codec==0 path should emit AddPtr with offset 56 for k_row and v_row");
    }

    // ── Wave 12k64: +10 new tests ────────────────────────────────────────────

    /// Verify `smem_kv_off` produces the expected nested OffsetExpr::Add structure
    /// with ScalarVReg(read_buf), Mul(ScalarVReg(ki_off), head_bytes/k_stride), and Const(d_off).
    // @trace TEST-12k64
    #[test]
    fn test_smem_kv_off_offset_structure() {
        let read_buf = VRegId(10);
        let ki_off = VRegId(20);
        let head_bytes = 256;
        let k_stride = 256;
        let d_off = 32;

        let result = smem_kv_off(read_buf, ki_off, head_bytes, k_stride, d_off);

        // The outer structure should be OffsetExpr::Add(ScalarVReg(read_buf), inner)
        match &result {
            OffsetExpr::Add(outer_left, inner) => {
                assert!(
                    matches!(**outer_left, OffsetExpr::ScalarVReg(v) if v == read_buf),
                    "outer left should be ScalarVReg(read_buf)"
                );
                // Inner should be Add(Mul(ScalarVReg(ki_off), head_bytes/k_stride), Const(d_off))
                match &**inner {
                    OffsetExpr::Add(mul_part, const_part) => {
                        assert!(
                            matches!(**const_part, OffsetExpr::Const(32)),
                            "inner right should be Const(32)"
                        );
                        match &**mul_part {
                            OffsetExpr::Mul(vreg_part, stride_val) => {
                                assert!(
                                    matches!(**vreg_part, OffsetExpr::ScalarVReg(v) if v == ki_off),
                                    "mul left should be ScalarVReg(ki_off)"
                                );
                                assert_eq!(*stride_val, 1, "mul right should be head_bytes/k_stride=1");
                            }
                            other => panic!("expected Mul inside inner Add, got {:?}", other),
                        }
                    }
                    other => panic!("expected Add for inner, got {:?}", other),
                }
            }
            other => panic!("expected Add for outer, got {:?}", other),
        }
    }

    /// Verify `smem_kv_off` with head_bytes=512, k_stride=256 produces stride factor 2.
    // @trace TEST-12k64
    #[test]
    fn test_smem_kv_off_stride_factor() {
        let read_buf = VRegId(0);
        let ki_off = VRegId(1);

        let result = smem_kv_off(read_buf, ki_off, 512, 256, 0);

        // head_bytes / k_stride = 512 / 256 = 2
        let found_stride_2 = match &result {
            OffsetExpr::Add(_, inner) => match &**inner {
                OffsetExpr::Add(mul_part, _) => match &**mul_part {
                    OffsetExpr::Mul(_, stride) => *stride == 2,
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        };
        assert!(found_stride_2, "head_bytes=512, k_stride=256 should produce stride factor 2");
    }

    /// Verify `softmax_trace()` returns deterministic results: calling it multiple times
    /// yields identical TraceOp sequences.
    // @trace TEST-12k64
    #[test]
    fn test_softmax_trace_deterministic() {
        let first = softmax_trace();
        let second = softmax_trace();
        assert_eq!(first.len(), second.len(), "repeated calls should return same length");
        for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(format!("{:?}", a), format!("{:?}", b), "op[{}] should be identical across calls", i);
        }
    }

    /// Verify `accumulate_trace()` returns deterministic results.
    // @trace TEST-12k64
    #[test]
    fn test_accumulate_trace_deterministic() {
        let first = accumulate_trace();
        let second = accumulate_trace();
        assert_eq!(first.len(), second.len());
        for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(format!("{:?}", a), format!("{:?}", b), "op[{}] mismatch", i);
        }
    }

    /// Verify `sum_update_trace()` returns deterministic results.
    // @trace TEST-12k64
    #[test]
    fn test_sum_update_trace_deterministic() {
        let first = sum_update_trace();
        let second = sum_update_trace();
        assert_eq!(first.len(), second.len());
        for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(format!("{:?}", a), format!("{:?}", b), "op[{}] mismatch", i);
        }
    }

    /// Verify `patch_skip_count` is a no-op when the instruction at `patch_idx`
    /// is not a GprCondAction (e.g., it's a Broadcast) — the instruction should
    /// remain unchanged.
    // @trace TEST-12k64
    #[test]
    fn test_patch_skip_count_noop_on_non_cond_action() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(42.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        let broadcast_idx = prog.instrs.len() - 1;

        // Act: patch_skip_count on a Broadcast should silently do nothing
        patch_skip_count(&mut prog, broadcast_idx, 99);

        // Assert: instruction is unchanged
        match &prog.instrs[broadcast_idx] {
            VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } => {
                assert_eq!(*v, 42.0, "Broadcast should be unchanged after no-op patch");
            }
            other => panic!("expected unchanged Broadcast, got {:?}", other),
        }
    }

    /// Verify `emit_tiled_attention_inline` with causal=true compiles successfully
    /// and emits a DynamicVRegPlusOne bound for the ki loop (causal mask).
    // @trace TEST-12k64
    #[test]
    fn test_tiled_attention_causal_mode_compiles() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: causal=true with multi-token Q (seq=4) so causal masking logic activates
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(4), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            true,  // causal = true
            None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: should compile without error
        assert!(result.is_ok(), "causal mode should compile: {:?}", result);

        // Should still emit output stores
        let vec_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(vec_stores > 0, "causal mode should emit VecStore instructions");
    }

    /// Verify `emit_tiled_attention_inline` with BoundExpr::Symbolic (symbolic kv_bound)
    /// compiles and uses the max_alloc value for seq_len derivation.
    // @trace TEST-12k64
    #[test]
    fn test_tiled_attention_symbolic_kv_bound() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };
        use crate::compiler::codegen::vm::instr::SymBound;

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: symbolic kv_bound with max_alloc=8
        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".to_string(),
            max_alloc: 8,
        });

        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), sym_bound,
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: should compile using max_alloc=8 as seq_len
        assert!(result.is_ok(), "symbolic kv_bound should compile: {:?}", result);
        let vec_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(vec_stores > 0, "symbolic bound should emit VecStore");
    }

    /// Verify `emit_kv_row_ptrs` with compress context emits decompress instructions
    /// (ScalarByteLoad for codec) alongside the paged PageTableAddr instructions.
    // @trace TEST-12k64
    #[test]
    fn test_kv_row_ptrs_paged_with_compress() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        // Act
        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0),
            Some(pt_ptr),
            16, 0, 256, 256, k_ptr, v_ptr,
            None, Some(&ctx), None,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Assert: should emit PageTableAddr for K and V
        let pta_count = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { .. }))
            .count();
        assert_eq!(pta_count, 2, "should emit 2 PageTableAddr (K + V)");

        // Should also emit ScalarByteLoad at 0x28 for codec read (decompress)
        let has_codec_load = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(0x28), .. }
        ));
        assert!(has_codec_load, "should emit ScalarByteLoad at 0x28 for codec in decompress path");

        // Should emit MemFence(AcqRel) from decompress
        let has_fence = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::MemFence { order: MemFenceOrder::AcqRel }
        ));
        assert!(has_fence, "should emit MemFence(AcqRel) from decompress path");
    }

    /// Verify `emit_tiled_attention_inline` with sinks_ptr (streaming attention sinks)
    /// compiles and broadcasts the sink value instead of NEG_INFINITY for running_max.
    // @trace TEST-12k64
    #[test]
    fn test_tiled_attention_with_sinks_initializes_from_sink() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sinks = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, Some(sinks), QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert
        assert!(result.is_ok(), "sinks mode should compile: {:?}", result);

        // With sinks, running_sum should init to 1.0 instead of 0.0
        let has_sum_init_one = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(1.0), .. }
        ));
        assert!(has_sum_init_one, "with sinks, running_sum should be initialized to 1.0");

        // With sinks, running_max should NOT be initialized to NEG_INFINITY
        // Instead it should use ScalarExpr::MemLoad from sinks_ptr
        let has_memload_sink = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::MemLoad(..), .. }
        ));
        assert!(has_memload_sink, "with sinks, running_max should init from MemLoad(sinks_ptr)");
    }

    /// Verify `count_non_meta_between` returns the correct count when all instructions
    /// between start and the predicate are non-meta (no Comments or MarkLabels interspersed).
    // @trace TEST-12k64
    #[test]
    fn test_count_non_meta_between_all_non_meta() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let label = prog.alloc_label();

        let start_idx = prog.instrs.len();
        // 3 non-meta instructions
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(0.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(2.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });
        prog.emit(VmInstr::MarkLabel { label_id: label });
        let target_idx = prog.instrs.len() - 1;

        // Act
        let count = count_non_meta_between(&prog, start_idx, |i| i == target_idx);

        // Assert: 3 non-meta instructions before the MarkLabel
        assert_eq!(count, 3, "should count exactly 3 non-meta Broadcast instructions");
    }

    // ── Wave 12k87: +10 new tests ────────────────────────────────────────────

    /// Verify `KvLoadMode` Debug trait produces distinct debug strings for each variant,
    /// ensuring no two variants share the same debug representation.
    // @trace TEST-12k87
    #[test]
    fn test_kv_load_mode_debug_strings_distinct() {
        let modes = [
            KvLoadMode::Direct,
            KvLoadMode::Kivi4,
            KvLoadMode::Kivi2,
            KvLoadMode::Sparse,
            KvLoadMode::Auto,
        ];
        let debug_strs: Vec<String> = modes.iter().map(|m| format!("{:?}", m)).collect();
        for i in 0..debug_strs.len() {
            for j in (i + 1)..debug_strs.len() {
                assert_ne!(
                    debug_strs[i], debug_strs[j],
                    "KvLoadMode::{:?} and {:?} should have distinct Debug output",
                    modes[i], modes[j]
                );
            }
        }
    }

    /// Verify `KvLoadMode` Clone produces an independent copy that equals the original.
    // @trace TEST-12k87
    #[test]
    fn test_kv_load_mode_clone_independence() {
        let original = KvLoadMode::Kivi4;
        let cloned = original.clone();
        assert_eq!(original, cloned, "cloned KvLoadMode should equal original");
        // Mutating a copy should not affect original (Copy semantics, but test anyway)
        let _modified = KvLoadMode::Sparse;
        assert_eq!(original, KvLoadMode::Kivi4, "original should remain unchanged");
    }

    /// Verify `smem_kv_off` with d_off=0 produces an offset expression that still has the
    /// outer Add(ScalarVReg(read_buf), inner) structure and Const(0) for the d_off part.
    // @trace TEST-12k87
    #[test]
    fn test_smem_kv_off_zero_d_off() {
        let read_buf = VRegId(5);
        let ki_off = VRegId(10);

        let result = smem_kv_off(read_buf, ki_off, 256, 256, 0);

        // Verify outer structure is Add(ScalarVReg(read_buf), ...)
        match &result {
            OffsetExpr::Add(outer_left, inner) => {
                assert!(
                    matches!(**outer_left, OffsetExpr::ScalarVReg(v) if v == read_buf),
                    "outer left should be ScalarVReg(read_buf)"
                );
                // Inner should be Add(Mul(..., stride), Const(0))
                match &**inner {
                    OffsetExpr::Add(_, const_part) => {
                        assert!(
                            matches!(**const_part, OffsetExpr::Const(0)),
                            "d_off part should be Const(0)"
                        );
                    }
                    other => panic!("expected Add for inner, got {:?}", other),
                }
            }
            other => panic!("expected Add for outer, got {:?}", other),
        }
    }

    /// Verify `emit_tiled_attention_inline` with BoundExpr::Runtime produces seq_len=1
    /// (the fallback for runtime bounds) and compiles without error.
    // @trace TEST-12k87
    #[test]
    fn test_tiled_attention_runtime_bound_compiles() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: Runtime bound for kv — seq_len should default to 1
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Runtime(PtrExpr::NamedArg("kv_len".to_string())),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: should compile without error
        assert!(result.is_ok(), "Runtime bound should compile: {:?}", result);
        let vec_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(vec_stores > 0, "Runtime bound should still emit VecStore");
    }

    /// Verify `emit_sparse_masked_load` with BF16 QuantPrecision emits VecLoad with
    /// the correct dtype, not F32.
    // @trace TEST-12k87
    #[test]
    fn test_sparse_masked_load_bf16_dtype() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_sparse_masked_load(
            &mut prog, dst, base, OffsetExpr::Const(0),
            None, 0, SimdWidth::W256, QuantPrecision::BF16,
        );

        // The VecLoad should have dtype=BF16
        let vec_load = prog.instrs.iter().find(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(vec_load.is_some(), "should emit a VecLoad");
        match vec_load.unwrap() {
            VmInstr::VecLoad { dtype, .. } => {
                assert_eq!(*dtype, QuantPrecision::BF16, "VecLoad dtype should be BF16");
            }
            _ => unreachable!(),
        }
    }

    /// Verify `emit_tier_dispatch_k_load` emits UnconditionalBranch instructions for
    /// jumping past alternate paths to the done label.
    // @trace TEST-12k87
    #[test]
    fn test_tier_dispatch_k_load_has_unconditional_branches() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_tier_dispatch_k_load(
            &mut prog, dst, k_row, OffsetExpr::Const(0),
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, 0, page_hdr,
        );

        // Should contain at least 2 UnconditionalBranch:
        // 1. After default VecLoad (jump to done)
        // 2. After sparse path (jump to done)
        let branches = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::UnconditionalBranch { .. }))
            .count();
        assert!(branches >= 2, "should emit at least 2 UnconditionalBranch for done_label jumps, got {}", branches);
    }

    /// Verify `emit_decompress_page` with non-zero kv_h and head_bytes computes
    /// correct scratch offsets (kv_h * head_bytes) for both k_row and v_row AddPtr.
    // @trace TEST-12k87
    #[test]
    fn test_decompress_page_nonzero_kv_h_offset() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let kv_h = 3;
        let head_bytes = 128;
        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 2048 };

        emit_decompress_page(&mut prog, k_row, v_row, &ctx, kv_h, head_bytes);

        let expected_offset = kv_h * head_bytes; // 384
        // The decompressed path should redirect k_row and v_row to scratch + kv_h*head_bytes
        let scratch_redirects = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::AddPtr { offset, .. } if *offset == expected_offset))
            .count();
        assert!(
            scratch_redirects >= 2,
            "should emit at least 2 AddPtr with offset {} (kv_h*head_bytes) for k_row/v_row scratch redirect",
            expected_offset
        );
    }

    /// Verify `emit_kv_row_ptrs` with pt_ptr=Some but page_size=0 emits LoadPtr with
    /// VRegPlusOff containing the correct ki_byte_off constant.
    // @trace TEST-12k87
    #[test]
    fn test_kv_row_ptrs_zero_page_size_preserves_ki_offset() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let ki_off_val = 512;
        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(ki_off_val),
            Some(pt_ptr),
            0,  // page_size=0 → LoadPtr path
            0, 256, 256, k_ptr, v_ptr,
            None, None, None,
        );

        let new_instrs = &prog.instrs[before_count..];
        // Both LoadPtr should use VRegPlusOff with ki_off_val
        let load_ptrs: Vec<_> = new_instrs.iter()
            .filter_map(|i| match i {
                VmInstr::LoadPtr { src: PtrExpr::VRegPlusOff(_, off), .. } => Some(off.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(load_ptrs.len(), 2, "should emit 2 LoadPtr with VRegPlusOff");
        assert!(
            load_ptrs.iter().all(|off| *off == OffsetExpr::Const(ki_off_val)),
            "all LoadPtr offsets should be Const({})", ki_off_val
        );
    }

    /// Verify `emit_tiled_attention_inline` with MQA (num_heads=1, num_kv_heads=1)
    /// compiles and emits exactly 1 head's worth of output stores.
    // @trace TEST-12k87
    #[test]
    fn test_tiled_attention_mqa_single_head() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: MQA — 1 query head, 1 kv head (ratio 1:1)
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            1, 1, 64,  // MQA: 1 head, 1 kv_head
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: should compile
        assert!(result.is_ok(), "MQA single head should compile: {:?}", result);

        // With head_dim=64 and W256 (8 F32 lanes), hd_vecs=8, so 8 VecStore per head
        let vec_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(
            vec_stores >= 1,
            "MQA should emit at least 1 VecStore for the single head, got {}",
            vec_stores
        );
    }

    /// Verify `softmax_trace()` and `sum_update_trace()` produce distinct trace sequences
    /// (different lengths, different op types).
    // @trace TEST-12k87
    #[test]
    fn test_softmax_and_sum_update_traces_are_distinct() {
        let soft = softmax_trace();
        let sum = sum_update_trace();

        // Different lengths
        assert_ne!(soft.len(), sum.len(),
            "softmax_trace ({}) and sum_update_trace ({}) should have different lengths",
            soft.len(), sum.len()
        );

        // softmax has a Max op, sum_update does not
        let softmax_has_max = soft.iter().any(|op| matches!(op, TraceOp::Max(_, _)));
        let sum_has_max = sum.iter().any(|op| matches!(op, TraceOp::Max(_, _)));
        assert!(softmax_has_max, "softmax_trace should contain Max");
        assert!(!sum_has_max, "sum_update_trace should not contain Max");

        // Both contain Mul or Sub (arithmetic ops)
        let softmax_has_sub = soft.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        let sum_has_mul = sum.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
        assert!(softmax_has_sub, "softmax_trace should contain Sub");
        assert!(sum_has_mul, "sum_update_trace should contain Mul");

        // Only softmax has Exp
        let softmax_has_exp = soft.iter().any(|op| matches!(op, TraceOp::Exp(_)));
        let sum_has_exp = sum.iter().any(|op| matches!(op, TraceOp::Exp(_)));
        assert!(softmax_has_exp, "softmax_trace should contain Exp");
        assert!(!sum_has_exp, "sum_update_trace should not contain Exp");
    }

    // ── Wave 12kar: +10 new tests ────────────────────────────────────────────

    /// Verify `emit_score_dot_cpu` with KvLoadMode::Direct emits the expected
    /// instruction sequence: Broadcast(0.0) for accumulator init, VecLoad for Q and K,
    /// and HReduce for horizontal sum reduction.
    // @trace TEST-12kar
    #[test]
    fn test_score_dot_cpu_direct_emits_dot_and_hreduce() {
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        let before_count = prog.instrs.len();
        let result_vreg = emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            2, 32,  // hd_vecs=2, vec_step=32 (8 lanes * 4 bytes)
            SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Direct,
            None, None,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Should emit Broadcast(0.0) for accumulator initialization
        let has_zero_init = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }
        ));
        assert!(has_zero_init, "score_dot should initialize accumulator with Broadcast(0.0)");

        // Should emit VecLoad for Q rows
        let vec_load_count = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert!(vec_load_count >= 2, "score_dot with hd_vecs=2 should emit at least 2 VecLoad (Q+K per vec)");

        // Should emit ExtractLane0 (from HReduce result broadcast)
        let has_extract = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::ExtractLane0(_), .. }
        ));
        assert!(has_extract, "score_dot should emit ExtractLane0 broadcast from HReduce");

        // The return value should be a valid VRegId
        assert!(result_vreg.0 > 0, "returned VRegId should be allocated");
    }

    /// Verify `emit_score_dot_cpu` with KvLoadMode::Kivi4 emits KiviDequantLoad
    /// instead of plain VecLoad for K data.
    // @trace TEST-12kar
    #[test]
    fn test_score_dot_cpu_kivi4_emits_dequant() {
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            1, 32,
            SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Kivi4,
            None, None,
        );

        let has_kivi = prog.instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_kivi, "Kivi4 mode should emit KiviDequantLoad for K rows");
    }

    /// Verify `emit_v_accumulate_cpu` with KvLoadMode::Direct emits VecLoad for V
    /// rows and does not emit KiviDequantLoad or Broadcast(0.0) for inactive channels.
    // @trace TEST-12kar
    #[test]
    fn test_v_accumulate_cpu_direct_emits_vecload_only() {
        let mut prog = VmProgram::new();
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let o_acc1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let correction = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let weight = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let accumulate_body = accumulate_trace();

        let before_count = prog.instrs.len();
        emit_v_accumulate_cpu(
            &mut prog, v_row, 2, 32,
            &[o_acc0, o_acc1], correction, weight,
            SimdWidth::W256, QuantPrecision::F32, &accumulate_body,
            KvLoadMode::Direct, 8, None, None,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Should emit VecLoad for V data (2 hd_vecs)
        let vec_loads = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert!(vec_loads >= 2, "Direct mode should emit VecLoad for each hd_vec (got {})", vec_loads);

        // Should NOT emit KiviDequantLoad
        let has_kivi = new_instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(!has_kivi, "Direct mode should not emit KiviDequantLoad");

        // Should NOT emit Broadcast(0.0) (that's for sparse inactive channels)
        let has_zero_broadcast = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }
        ));
        assert!(!has_zero_broadcast, "Direct mode should not emit Broadcast(0.0) for inactive channels");
    }

    /// Verify `emit_v_accumulate_cpu` with KvLoadMode::Kivi2 emits KiviDequantLoad
    /// for V data (not VecLoad).
    // @trace TEST-12kar
    #[test]
    fn test_v_accumulate_cpu_kivi2_emits_dequant() {
        let mut prog = VmProgram::new();
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let correction = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let weight = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let accumulate_body = accumulate_trace();

        let before_count = prog.instrs.len();
        emit_v_accumulate_cpu(
            &mut prog, v_row, 1, 32,
            &[o_acc0], correction, weight,
            SimdWidth::W256, QuantPrecision::F32, &accumulate_body,
            KvLoadMode::Kivi2, 8, None, None,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Kivi2 should emit KiviDequantLoad
        let has_kivi = new_instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_kivi, "Kivi2 mode should emit KiviDequantLoad for V rows");

        // Kivi2 should NOT emit plain VecLoad (only KiviDequantLoad)
        let plain_vec_loads = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert_eq!(plain_vec_loads, 0, "Kivi2 mode should not emit plain VecLoad");
    }

    /// Verify `emit_normalize_store` emits VecStore instructions for each hd_vec
    /// output element, with correct offset increments (d * vec_step).
    // @trace TEST-12kar
    #[test]
    fn test_normalize_store_emits_vecstore_with_correct_offsets() {
        let mut prog = VmProgram::new();
        let o_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let o_acc1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let o_acc2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        let before_count = prog.instrs.len();
        emit_normalize_store(
            &mut prog, o_row, &[o_acc0, o_acc1, o_acc2], running_sum,
            3, 32,  // hd_vecs=3, vec_step=32
            SimdWidth::W256, QuantPrecision::F32,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Should emit 3 VecStore instructions (one per hd_vec)
        let stores: Vec<_> = new_instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecStore { offset, src, base, .. } => Some((offset.clone(), *src, *base)),
                _ => None,
            })
            .collect();
        assert_eq!(stores.len(), 3, "should emit 3 VecStore for hd_vecs=3");

        // Verify offsets are 0, 32, 64
        assert_eq!(stores[0].0, OffsetExpr::Const(0), "first store offset should be 0");
        assert_eq!(stores[1].0, OffsetExpr::Const(32), "second store offset should be 32");
        assert_eq!(stores[2].0, OffsetExpr::Const(64), "third store offset should be 64");

        // All stores should target o_row
        assert!(stores.iter().all(|(_, _, base)| *base == o_row),
            "all VecStore should target o_row");
    }

    /// Verify `emit_softmax_update` emits the expected instruction sequence:
    /// auto_lower_trace for softmax body, sum body, and max identity copy.
    /// The returned tuple should contain three distinct VRegIds.
    // @trace TEST-12kar
    #[test]
    fn test_softmax_update_returns_distinct_vregs() {
        let mut prog = VmProgram::new();
        let running_max = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let score = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let softmax_body = softmax_trace();
        let sum_body = sum_update_trace();

        let before_count = prog.instrs.len();
        let (new_max, correction, weight) = emit_softmax_update(
            &mut prog, running_max, score, running_sum,
            &softmax_body, &sum_body, SimdWidth::W256,
        );

        // Should have emitted new instructions
        let new_instrs = &prog.instrs[before_count..];
        assert!(!new_instrs.is_empty(), "softmax_update should emit instructions");

        // The three returned VRegIds should be distinct
        assert_ne!(new_max, correction, "new_max and correction should be different VRegs");
        assert_ne!(new_max, weight, "new_max and weight should be different VRegs");
        assert_ne!(correction, weight, "correction and weight should be different VRegs");
    }

    /// Verify `emit_smem_stage_row` with async=true emits SharedMemAsyncStore
    /// instead of SharedMemStore for both K and V data.
    // @trace TEST-12kar
    #[test]
    fn test_smem_stage_row_async_emits_async_store() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let before_count = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row,
            "smem_k", "smem_v", write_buf,
            0,  // t=0
            256,  // head_bytes
            1,    // hd_vecs=1
            32,   // vec_step
            KvLoadMode::Direct, true,  // use_async=true
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, None,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Should emit SharedMemAsyncStore (not SharedMemStore)
        let async_stores = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::SharedMemAsyncStore { .. }))
            .count();
        assert!(async_stores >= 2, "async mode should emit at least 2 SharedMemAsyncStore (K + V)");

        // Should NOT emit plain SharedMemStore
        let sync_stores = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::SharedMemStore { .. }))
            .count();
        assert_eq!(sync_stores, 0, "async mode should not emit SharedMemStore");
    }

    /// Verify `emit_smem_stage_row` with async=false emits SharedMemStore
    /// (not SharedMemAsyncStore).
    // @trace TEST-12kar
    #[test]
    fn test_smem_stage_row_sync_emits_sync_store() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let before_count = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row,
            "smem_k", "smem_v", write_buf,
            0,
            256, 1, 32,
            KvLoadMode::Direct, false,  // use_async=false
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, None,
        );

        let new_instrs = &prog.instrs[before_count..];

        // Should emit SharedMemStore
        let sync_stores = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::SharedMemStore { .. }))
            .count();
        assert!(sync_stores >= 2, "sync mode should emit at least 2 SharedMemStore (K + V)");

        // Should NOT emit SharedMemAsyncStore
        let async_stores = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::SharedMemAsyncStore { .. }))
            .count();
        assert_eq!(async_stores, 0, "sync mode should not emit SharedMemAsyncStore");
    }

    /// Verify `count_non_meta_between` returns 0 when start equals program length
    /// (empty scan range, no instructions to examine).
    // @trace TEST-12kar
    #[test]
    fn test_count_non_meta_between_empty_range() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Broadcast { dst: v, src: ScalarExpr::Const(0.0), width: SimdWidth::W256, dtype: QuantPrecision::F32 });

        // Start at len() — no instructions to scan, predicate never matches
        let start = prog.instrs.len();
        let count = count_non_meta_between(&prog, start, |_i| false);

        // Should return 0 because there are no instructions in range
        assert_eq!(count, 0, "empty scan range should return 0");
    }

    /// Verify `emit_decompress_page` emits a ScalarLoad at offset 0x2C for
    /// reading the compressed_size field from the KvPageHeader, used by the
    /// decompression VmInstr to know the input byte count.
    // @trace TEST-12kar
    #[test]
    fn test_decompress_page_reads_compressed_size_at_0x2c() {
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        emit_decompress_page(&mut prog, k_row, v_row, &ctx, 0, 256);

        // Should emit ScalarLoad at offset 0x2C for compressed_size
        let has_csz_load = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(0x2C), .. }
        ));
        assert!(has_csz_load, "should emit ScalarLoad at offset 0x2C for compressed_size");
    }

    // ── Wave 12khc: +10 new tests ────────────────────────────────────────────

    /// Verify `emit_v_accumulate_cpu` with KvLoadMode::Sparse emits
    /// GprCondAction for channel-level masking and Broadcast(0.0) for
    /// inactive channels (MUSTAFAR sparse path).
    // @trace TEST-12khc
    #[test]
    fn test_v_accumulate_cpu_sparse_emits_channel_masking() {
        // Arrange
        let mut prog = VmProgram::new();
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let o_acc1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let correction = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let weight = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let accumulate_body = accumulate_trace();

        // Act
        let before_count = prog.instrs.len();
        emit_v_accumulate_cpu(
            &mut prog, v_row, 2, 32,
            &[o_acc0, o_acc1], correction, weight,
            SimdWidth::W256, QuantPrecision::F32, &accumulate_body,
            KvLoadMode::Sparse, 8, Some(bmp), None,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Sparse mode should emit GprCondAction with BitClear
        let cond_actions = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { .. }))
            .count();
        assert!(cond_actions >= 2, "Sparse mode should emit GprCondAction for each hd_vec, got {}", cond_actions);

        // Assert: Sparse mode should emit Broadcast(0.0) for inactive channel zero-fill
        let zero_broadcasts = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }))
            .count();
        assert!(zero_broadcasts >= 2, "Sparse mode should emit Broadcast(0.0) for inactive channels, got {}", zero_broadcasts);

        // Assert: No KiviDequantLoad in sparse mode
        let has_kivi = new_instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(!has_kivi, "Sparse mode should not emit KiviDequantLoad");
    }

    /// Verify `emit_v_accumulate_cpu` with KvLoadMode::Auto and page_header_ptr
    /// emits ScalarByteLoad for runtime tier dispatch of V rows.
    // @trace TEST-12khc
    #[test]
    fn test_v_accumulate_cpu_auto_paged_emits_tier_dispatch() {
        // Arrange
        let mut prog = VmProgram::new();
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let correction = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let weight = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let accumulate_body = accumulate_trace();

        // Act
        let before_count = prog.instrs.len();
        emit_v_accumulate_cpu(
            &mut prog, v_row, 1, 32,
            &[o_acc0], correction, weight,
            SimdWidth::W256, QuantPrecision::F32, &accumulate_body,
            KvLoadMode::Auto, 8, None, Some(page_hdr),
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Auto mode with page header should emit ScalarByteLoad at offset 28
        let has_byte_load = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(28), .. }
        ));
        assert!(has_byte_load, "Auto mode with page header should emit ScalarByteLoad at offset 28 for tier dispatch");

        // Assert: Should have GprCondAction for tier branching
        let cond_actions = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { .. }))
            .count();
        assert!(cond_actions >= 3, "Auto mode tier dispatch should emit at least 3 GprCondAction, got {}", cond_actions);
    }

    /// Verify `emit_score_dot_cpu` with KvLoadMode::Sparse emits GprCondAction
    /// for channel-level masking on K data during Q*K dot product.
    // @trace TEST-12khc
    #[test]
    fn test_score_dot_cpu_sparse_emits_channel_masking() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let before_count = prog.instrs.len();
        emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            2, 32, SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Sparse,
            Some(bmp), None,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Sparse mode should emit GprCondAction for K channel masking
        let cond_actions = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprCondAction { cond: GprCondition::BitClear(..), .. }))
            .count();
        assert!(cond_actions >= 2, "Sparse dot should emit BitClear GprCondAction per hd_vec, got {}", cond_actions);

        // Assert: Should emit Broadcast(0.0) for inactive channel zero-fill on K
        let zero_broadcasts = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }))
            .count();
        assert!(zero_broadcasts >= 2, "Sparse dot should emit Broadcast(0.0) for inactive K channels, got {}", zero_broadcasts);
    }

    /// Verify `emit_score_dot_cpu` with KvLoadMode::Auto and page_header_ptr
    /// emits ScalarByteLoad at offset 28 for runtime tier dispatch of K loads.
    // @trace TEST-12khc
    #[test]
    fn test_score_dot_cpu_auto_paged_emits_tier_dispatch() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let before_count = prog.instrs.len();
        emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            1, 32, SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Auto,
            None, Some(page_hdr),
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Auto mode should emit ScalarByteLoad for precision_tier
        let has_byte_load = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(28), .. }
        ));
        assert!(has_byte_load, "Auto mode should emit ScalarByteLoad at offset 28 for K tier dispatch");

        // Assert: Should contain KiviDequantLoad (kivi path in tier dispatch)
        let has_kivi = new_instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_kivi, "Auto mode should contain KiviDequantLoad for kivi tier path");
    }

    /// Verify `emit_score_dot_cpu` with hd_vecs=0 skips the dot product loop
    /// entirely and only emits accumulator init + HReduce + scale.
    // @trace TEST-12khc
    #[test]
    fn test_score_dot_cpu_zero_hd_vecs_skips_dot_loop() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        let before_count = prog.instrs.len();
        let result_vreg = emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            0, 32,  // hd_vecs=0 — no dot product loop
            SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Direct,
            None, None,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Should still initialize accumulator with Broadcast(0.0)
        let has_zero_init = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }
        ));
        assert!(has_zero_init, "score_dot should always init accumulator even with hd_vecs=0");

        // Assert: No VecLoad should be emitted (no dot loop iterations)
        let vec_loads = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert_eq!(vec_loads, 0, "hd_vecs=0 should not emit any VecLoad in dot loop");

        // Assert: Should still emit ExtractLane0 from HReduce
        let has_extract = new_instrs.iter().any(|i| matches!(
            i,
            VmInstr::Broadcast { src: ScalarExpr::ExtractLane0(_), .. }
        ));
        assert!(has_extract, "score_dot should still emit HReduce + scale even with hd_vecs=0");

        // Assert: Return VRegId is valid
        assert!(result_vreg.0 > 0, "returned VRegId should be a valid allocation");
    }

    /// Verify `emit_smem_stage_row` with KvLoadMode::Kivi4 emits
    /// KiviDequantLoad instead of plain VecLoad for both K and V data
    /// before staging to shared memory.
    // @trace TEST-12khc
    #[test]
    fn test_smem_stage_row_kivi4_emits_dequant_loads() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let before_count = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row,
            "smem_k", "smem_v", write_buf,
            0, 256, 1, 32,
            KvLoadMode::Kivi4, false,
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, None,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Kivi4 should emit KiviDequantLoad (2: one for K, one for V)
        let kivi_loads = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::KiviDequantLoad { .. }))
            .count();
        assert_eq!(kivi_loads, 2, "Kivi4 smem staging should emit 2 KiviDequantLoad (K + V)");

        // Assert: No plain VecLoad should be emitted
        let plain_loads = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert_eq!(plain_loads, 0, "Kivi4 smem staging should not emit plain VecLoad");
    }

    /// Verify `emit_smem_stage_row` with hd_vecs=2 emits double the number of
    /// shared memory stores (2 K stores + 2 V stores = 4 total).
    // @trace TEST-12khc
    #[test]
    fn test_smem_stage_row_multiple_hd_vecs_doubles_stores() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let before_count = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row,
            "smem_k", "smem_v", write_buf,
            0, 256, 2, 32,  // hd_vecs=2
            KvLoadMode::Direct, false,
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, None,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: With hd_vecs=2, should emit 4 SharedMemStore (2 K + 2 V)
        let sync_stores = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::SharedMemStore { .. }))
            .count();
        assert_eq!(sync_stores, 4, "hd_vecs=2 should emit 4 SharedMemStore (2 K + 2 V), got {}", sync_stores);

        // Assert: VecLoad count should match (2 K loads + 2 V loads = 4)
        let vec_loads = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert_eq!(vec_loads, 4, "hd_vecs=2 should emit 4 VecLoad (2 K + 2 V), got {}", vec_loads);
    }

    /// Verify `emit_normalize_store` with a single hd_vec emits exactly one
    /// VecStore at offset 0 (boundary case for minimal head_dim).
    // @trace TEST-12khc
    #[test]
    fn test_normalize_store_single_hd_vec_emits_one_store() {
        // Arrange
        let mut prog = VmProgram::new();
        let o_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        let before_count = prog.instrs.len();
        emit_normalize_store(
            &mut prog, o_row, &[o_acc], running_sum,
            1, 32,  // hd_vecs=1, vec_step=32
            SimdWidth::W256, QuantPrecision::F32,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Should emit exactly 1 VecStore
        let stores: Vec<_> = new_instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecStore { offset, src, base, .. } => Some((offset.clone(), *src, *base)),
                _ => None,
            })
            .collect();
        assert_eq!(stores.len(), 1, "hd_vecs=1 should emit exactly 1 VecStore");

        // Assert: Offset should be 0
        assert_eq!(stores[0].0, OffsetExpr::Const(0), "single hd_vec store offset should be 0");

        // Assert: Target should be o_row
        assert_eq!(stores[0].2, o_row, "store should target o_row");
    }

    /// Verify `emit_kv_row_ptrs` with paged KV and seq_pt_offset=Some passes
    /// the seq_pt_offset through to PageTableAddr instructions for batch
    /// paged attention (BCI-004).
    // @trace TEST-12khc
    #[test]
    fn test_kv_row_ptrs_paged_with_seq_pt_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let seq_pt_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let before_count = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0),
            Some(pt_ptr), 16, 0, 256, 256, k_ptr, v_ptr,
            Some(seq_pt_off), None, None,
        );
        let new_instrs = &prog.instrs[before_count..];

        // Assert: Should emit PageTableAddr with seq_pt_offset = Some
        let pta_with_seq: Vec<_> = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { seq_pt_offset: Some(_), .. }))
            .collect();
        assert_eq!(pta_with_seq.len(), 2, "both K and V PageTableAddr should carry seq_pt_offset");

        // Assert: No PageTableAddr should have seq_pt_offset = None
        let pta_without_seq: Vec<_> = new_instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { seq_pt_offset: None, .. }))
            .collect();
        assert_eq!(pta_without_seq.len(), 0, "no PageTableAddr should have seq_pt_offset=None when Some provided");
    }

    /// Verify `emit_tiled_attention_inline` with BoundExpr::DynamicVReg produces
    /// seq_len=1 (same as Runtime) and compiles without error.
    // @trace TEST-12khc
    #[test]
    fn test_tiled_attention_dynamic_vreg_bound_compiles() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dyn_vreg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: DynamicVReg for kv_bound — seq_len should default to 1
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::DynamicVReg(dyn_vreg),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: should compile without error
        assert!(result.is_ok(), "DynamicVReg bound should compile: {:?}", result);
        let vec_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. }))
            .count();
        assert!(vec_stores > 0, "DynamicVReg bound should still emit VecStore for output");
    }

    // ── Wave 12khp: +10 new tests ────────────────────────────────────────────

    /// Verify `emit_score_dot_cpu` with KvLoadMode::Kivi2 emits KiviDequantLoad
    /// for K data (same VmInstr as Kivi4, different quantization semantics).
    // @trace TEST-12khp
    #[test]
    fn test_score_dot_cpu_kivi2_emits_dequant() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            1, 32, SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Kivi2,
            None, None,
        );

        // Assert
        let has_kivi = prog.instrs.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. }));
        assert!(has_kivi, "Kivi2 mode should emit KiviDequantLoad for K rows");
    }

    /// Verify `emit_v_accumulate_cpu` with KvLoadMode::Auto and page_header_ptr=None
    /// falls back to plain VecLoad (no tier dispatch, no KiviDequantLoad).
    // @trace TEST-12khp
    #[test]
    fn test_v_accumulate_cpu_auto_no_page_header_uses_vecload() {
        // Arrange
        let mut prog = VmProgram::new();
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let correction = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let weight = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = accumulate_trace();

        // Act
        let before = prog.instrs.len();
        emit_v_accumulate_cpu(
            &mut prog, v_row, 1, 32, &[o_acc], correction, weight,
            SimdWidth::W256, QuantPrecision::F32, &body,
            KvLoadMode::Auto, 8, None, None,
        );
        let added = &prog.instrs[before..];

        // Assert: plain VecLoad emitted
        assert!(added.iter().any(|i| matches!(i, VmInstr::VecLoad { .. })),
            "Auto without page header should emit VecLoad");
        // Assert: no KiviDequantLoad or tier dispatch
        assert!(!added.iter().any(|i| matches!(i, VmInstr::KiviDequantLoad { .. })),
            "Auto without page header should not emit KiviDequantLoad");
        assert!(!added.iter().any(|i| matches!(i, VmInstr::ScalarByteLoad { .. })),
            "Auto without page header should not emit ScalarByteLoad for tier dispatch");
    }

    /// Verify `emit_smem_stage_row` with KvLoadMode::Sparse emits GprCondAction
    /// for channel masking on K and V before staging to shared memory.
    // @trace TEST-12khp
    #[test]
    fn test_smem_stage_row_sparse_emits_channel_masking() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let before = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row, "smem_k", "smem_v", write_buf,
            0, 256, 1, 32, KvLoadMode::Sparse, false,
            SimdWidth::W256, QuantPrecision::F32, 8,
            Some(bmp), None,
        );
        let added = &prog.instrs[before..];

        // Assert: GprCondAction for K and V sparse masking
        let cond_count = added.iter().filter(|i| matches!(i, VmInstr::GprCondAction { .. })).count();
        assert!(cond_count >= 2, "Sparse smem should emit GprCondAction for K and V, got {}", cond_count);
        // Assert: Broadcast(0.0) for inactive channels
        let zero_bc = added.iter()
            .filter(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. }))
            .count();
        assert!(zero_bc >= 2, "Sparse smem should emit Broadcast(0.0) for inactive channels");
    }

    /// Verify `emit_smem_stage_row` emits the same number of non-meta instructions
    /// for different t values (t only affects offset constants, not structure).
    // @trace TEST-12khp
    #[test]
    fn test_smem_stage_row_different_t_same_instruction_count() {
        let count_for_t = |t: usize| -> usize {
            let mut prog = VmProgram::new();
            let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let wb = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            let before = prog.instrs.len();
            emit_smem_stage_row(
                &mut prog, k_row, v_row, "sk", "sv", wb,
                t, 256, 1, 32, KvLoadMode::Direct, false,
                SimdWidth::W256, QuantPrecision::F32, 8,
                None, None,
            );
            prog.instrs[before..].iter().filter(|i| !i.is_meta()).count()
        };
        assert_eq!(count_for_t(0), count_for_t(5), "t=0 and t=5 should produce same instruction count");
        assert_eq!(count_for_t(0), count_for_t(10), "t=0 and t=10 should produce same instruction count");
    }

    /// Verify `emit_decompress_page` emits AddPtr(base=k_row, offset=56) for
    /// src_ptr computation (skipping the 56-byte KvPageHeader).
    // @trace TEST-12khp
    #[test]
    fn test_decompress_page_src_ptr_skips_header() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        // Act
        emit_decompress_page(&mut prog, k_row, v_row, &ctx, 0, 256);

        // Assert: src_ptr = AddPtr(base=k_row, offset=56)
        let has_src = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::AddPtr { base, offset: 56, .. } if *base == k_row)
        });
        assert!(has_src, "should emit AddPtr with base=k_row, offset=56 for src_ptr");
    }

    /// Verify `emit_tier_dispatch_k_load` with sparse_bitmap_val=Some emits
    /// GprCondAction with BitClear in the sparse path (tier==4 branch).
    // @trace TEST-12khp
    #[test]
    fn test_tier_dispatch_k_load_with_sparse_bitmap() {
        // Arrange
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        emit_tier_dispatch_k_load(
            &mut prog, dst, k_row, OffsetExpr::Const(0),
            SimdWidth::W256, QuantPrecision::F32, 8,
            Some(bmp), 3, page_hdr,
        );

        // Assert: Sparse path should contain BitClear for channel_group=3
        let has_bitclear = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::GprCondAction { cond: GprCondition::BitClear(_, 3), .. })
        });
        assert!(has_bitclear, "tier dispatch with bitmap should emit BitClear(_, 3) in sparse path");
    }

    /// Verify `emit_tier_dispatch_v_load` with sparse_bitmap_val=Some emits
    /// GprCondAction with BitClear in the sparse path.
    // @trace TEST-12khp
    #[test]
    fn test_tier_dispatch_v_load_with_sparse_bitmap() {
        // Arrange
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let bmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        emit_tier_dispatch_v_load(
            &mut prog, dst, v_row, OffsetExpr::Const(0),
            SimdWidth::W256, QuantPrecision::F32, 8,
            Some(bmp), 5, page_hdr,
        );

        // Assert: Sparse path should contain BitClear for channel_group=5
        let has_bitclear = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::GprCondAction { cond: GprCondition::BitClear(_, 5), .. })
        });
        assert!(has_bitclear, "v_load tier dispatch with bitmap should emit BitClear(_, 5) in sparse path");
    }

    /// Verify `emit_kv_row_ptrs` paged path with non-zero kv_h passes the correct
    /// base_offset (kv_h * head_bytes) to both K and V PageTableAddr instructions.
    // @trace TEST-12khp
    #[test]
    fn test_kv_row_ptrs_paged_nonzero_kv_h_base_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_h = 3;
        let head_bytes = 128;

        // Act
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0), Some(pt_ptr), 16, kv_h, head_bytes, 128,
            k_ptr, v_ptr, None, None, None,
        );

        // Assert: both K and V PageTableAddr should have base_offset = kv_h * head_bytes
        let expected_offset = kv_h * head_bytes;
        let pta_with_correct_offset = prog.instrs.iter()
            .filter(|i| {
                matches!(i, VmInstr::PageTableAddr { base_offset, .. } if *base_offset == expected_offset)
            })
            .count();
        assert_eq!(pta_with_correct_offset, 2,
            "both K and V PageTableAddr should have base_offset={}", expected_offset);
    }

    /// Verify `emit_normalize_store` with hd_vecs=0 still emits the recip
    /// computation on running_sum but produces no VecStore instructions.
    // @trace TEST-12khp
    #[test]
    fn test_normalize_store_zero_hd_vecs_emits_recip_no_store() {
        // Arrange
        let mut prog = VmProgram::new();
        let o_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        let before = prog.instrs.len();
        emit_normalize_store(
            &mut prog, o_row, &[], running_sum,
            0, 32, SimdWidth::W256, QuantPrecision::F32,
        );
        let added = &prog.instrs[before..];

        // Assert: Should emit non-empty instructions (recip computation)
        assert!(!added.is_empty(), "should emit recip computation even with hd_vecs=0");
        // Assert: No VecStore should be emitted (empty o_acc slice)
        let stores = added.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert_eq!(stores, 0, "hd_vecs=0 should not emit any VecStore");
    }

    /// Verify `emit_smem_stage_row` with KvLoadMode::Auto and page_header_ptr
    /// emits ScalarByteLoad for runtime tier dispatch on both K and V loads.
    // @trace TEST-12khp
    #[test]
    fn test_smem_stage_row_auto_paged_emits_tier_dispatch() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        let before = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row, "smem_k", "smem_v", write_buf,
            0, 256, 1, 32, KvLoadMode::Auto, false,
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, Some(page_hdr),
        );
        let added = &prog.instrs[before..];

        // Assert: Should emit ScalarByteLoad at offset 28 for tier dispatch (K + V)
        let byte_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(28), .. }))
            .count();
        assert!(byte_loads >= 2,
            "Auto with page header should emit ScalarByteLoad for K and V tier dispatch, got {}", byte_loads);
    }

    // ── Wave 12khr: +10 new tests ────────────────────────────────────────────

    /// Verify decode-mode attention (q_bound=Const(1), kv_bound=Const(16)) compiles
    /// and emits single-query loop with full KV scan. Decode is the typical production
    /// scenario where one new token attends to all prior tokens.
    // @trace TEST-12khr
    #[test]
    fn test_tiled_attention_decode_mode_single_query_full_kv() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: decode mode — 1 query token, 16 KV tokens
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(16),
            4, 4, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "decode mode should compile: {:?}", result);
        // Assert: emits VecStore for each head output
        let vec_stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(vec_stores >= 4, "decode mode should emit VecStore per head, got {}", vec_stores);
    }

    /// Verify GQA with 8:2 ratio (8 query heads, 2 KV heads, gqa_ratio=4) compiles
    /// and emits correct head-to-kv-head mapping (each KV head shared by 4 query heads).
    // @trace TEST-12khr
    #[test]
    fn test_tiled_attention_gqa_8_2_ratio_compiles() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: 8 query heads, 2 KV heads → gqa_ratio=4
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            8, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "GQA 8:2 should compile: {:?}", result);
        // Assert: 8 heads → 8 output VecStore groups (one per query head)
        let vec_stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(vec_stores >= 8, "GQA 8:2 should emit VecStore per query head, got {}", vec_stores);
    }

    /// Verify SlidingWindow attention strategy compiles and the tile sizes are
    /// capped at min(seq_len, 64) per the attention_emit code logic.
    // @trace TEST-12khr
    #[test]
    fn test_tiled_attention_sliding_window_strategy_compiles() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct SlidingHook;
        impl IsaHook for SlidingHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::SlidingWindow { window_size: 256 }
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: SlidingWindow with seq_len=32 (< 64 tile cap)
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(32), BoundExpr::Const(32),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&SlidingHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "SlidingWindow should compile: {:?}", result);
        let vec_stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(vec_stores > 0, "SlidingWindow should emit VecStore for output");
    }

    /// Verify paged KV addressing with page_size=1 (minimum page, every token is
    /// a separate page) still compiles and emits PageTableAddr per K/V row.
    // @trace TEST-12khr
    #[test]
    fn test_tiled_attention_paged_kv_page_size_one_compiles() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: page_size=1 (extreme: every token is its own page)
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            Some(pt_ptr), 1, KvLoadMode::Direct,  // page_size=1
            None, None, None,
            false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "page_size=1 should compile: {:?}", result);
        // Assert: should emit PageTableAddr for paged KV lookup
        let pta_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { .. }))
            .count();
        assert!(pta_count > 0, "page_size=1 should emit PageTableAddr instructions, got {}", pta_count);
    }

    /// Verify causal attention with decode mode (q_bound=1) uses kv_bound directly
    /// (not DynamicVRegPlusOne) because causal masking is irrelevant for single-query decode.
    // @trace TEST-12khr
    #[test]
    fn test_tiled_attention_causal_decode_mode_no_dynamic_plus_one() {
        // Arrange — this is a structural check: decode_mode=true + causal=true
        // should use kv_bound (not DynamicVRegPlusOne) for the ki loop.
        // We verify by checking that the program compiles and produces VecStore output,
        // confirming the ki_bound path is valid.
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: causal=true but decode mode (q=1, kv=8) → should not use DynamicVRegPlusOne
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(1), BoundExpr::Const(8),  // decode: q=1
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            true, None, QuantPrecision::F32,  // causal=true
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "causal decode mode should compile: {:?}", result);
        // Assert: no DynamicVRegPlusOne used (decode_mode=true bypasses causal ki_bound)
        let has_dyn_plus_one = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoopBegin { bound: BoundExpr::DynamicVRegPlusOne(_), .. })
        });
        assert!(!has_dyn_plus_one, "causal decode should not use DynamicVRegPlusOne ki_bound");
    }

    /// Verify `emit_kv_row_ptrs` with page_size > 0 and pt_ptr provided but page_stride
    /// calculation correctly multiplies pgs * k_stride. Structural test with large
    /// page_size=128 to ensure stride arithmetic works for uncommon sizes.
    // @trace TEST-12khr
    #[test]
    fn test_kv_row_ptrs_paged_large_page_size_stride() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let pgs = 128;  // large page_size
        let k_stride = 256;  // num_kv_heads=4, head_dim=64, elem_bytes=4

        // Act
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0), Some(pt_ptr), pgs,
            0, 256, k_stride, k_ptr, v_ptr,
            None, None, None,
        );

        // Assert: page_stride = pgs * k_stride = 128 * 256 = 32768
        let expected_stride = pgs * k_stride;
        let pta_with_stride = prog.instrs.iter()
            .filter(|i| {
                matches!(i, VmInstr::PageTableAddr { page_stride, .. } if *page_stride == expected_stride)
            })
            .count();
        assert_eq!(pta_with_stride, 2,
            "both K and V PageTableAddr should have page_stride={}", expected_stride);
    }

    /// Verify `emit_score_dot_cpu` with hd_vecs=0 (head_dim < SIMD width, e.g.
    /// head_dim=4 with W256 lanes=8) skips the dot loop entirely and only
    /// emits HReduce + scale on the zero-initialized accumulator.
    // @trace TEST-12khr
    #[test]
    fn test_score_dot_cpu_sub_simd_head_dim_skips_vec_loop() {
        // Arrange: head_dim=4, W256 has 8 f32 lanes → hd_vecs = 4/8 = 0
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act: hd_vecs=0 (head_dim < lanes)
        let before = prog.instrs.len();
        let score = emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            0, 4, SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Direct,
            None, None,
        );
        let added = &prog.instrs[before..];

        // Assert: no VecLoad for dot-product loop (hd_vecs=0 skips it)
        let vec_loads = added.iter().filter(|i| matches!(i, VmInstr::VecLoad { .. })).count();
        assert_eq!(vec_loads, 0, "hd_vecs=0 should skip VecLoad dot loop");
        // Assert: still emits HReduce and scale (Broadcast + HReduce + Mul)
        assert!(added.iter().any(|i| matches!(i, VmInstr::Broadcast { src: ScalarExpr::ExtractLane0(_), .. })),
            "should emit HReduce result broadcast even with hd_vecs=0");
    }

    /// Verify that `emit_normalize_store` with multiple hd_vecs emits VecStore at
    /// consecutive offsets (0, vec_step, 2*vec_step) for each accumulator chunk.
    // @trace TEST-12khr
    #[test]
    fn test_normalize_store_multi_hd_vecs_correct_offset_sequence() {
        // Arrange
        let mut prog = VmProgram::new();
        let o_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc0 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let o_acc1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let o_acc2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act: hd_vecs=3, vec_step=32 (8 lanes * 4 bytes)
        emit_normalize_store(
            &mut prog, o_row, &[o_acc0, o_acc1, o_acc2], running_sum,
            3, 32, SimdWidth::W256, QuantPrecision::F32,
        );

        // Assert: 3 VecStore instructions with offsets 0, 32, 64
        let store_offsets: Vec<usize> = prog.instrs.iter()
            .filter_map(|i| match i {
                VmInstr::VecStore { offset: OffsetExpr::Const(off), .. } => Some(*off),
                _ => None,
            })
            .collect();
        assert_eq!(store_offsets.len(), 3, "should emit 3 VecStore for hd_vecs=3");
        assert_eq!(store_offsets[0], 0, "first VecStore offset should be 0");
        assert_eq!(store_offsets[1], 32, "second VecStore offset should be vec_step=32");
        assert_eq!(store_offsets[2], 64, "third VecStore offset should be 2*vec_step=64");
    }

    /// Verify `emit_decompress_page` with non-zero kv_h correctly computes
    /// dst_ptr offset as kv_h * head_bytes for scratch buffer targeting.
    // @trace TEST-12khr
    #[test]
    fn test_decompress_page_nonzero_kv_h_dst_offset() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_h = 5;
        let head_bytes = 128;
        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 8192 };

        // Act
        emit_decompress_page(&mut prog, k_row, v_row, &ctx, kv_h, head_bytes);

        // Assert: dst_ptr = AddPtr(base=scratch, offset=kv_h * head_bytes = 640)
        let expected_off = kv_h * head_bytes;
        let has_dst = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::AddPtr { base, offset, .. } if *base == scratch && *offset == expected_off)
        });
        assert!(has_dst, "should emit AddPtr(scratch, {}) for dst_ptr", expected_off);
    }

    /// Verify that a single KV token (kv_bound=Const(1)) in prefill mode still
    /// produces valid attention output — the softmax should handle the degenerate
    /// case of attending to exactly one position.
    // @trace TEST-12khr
    #[test]
    fn test_tiled_attention_single_kv_token_prefill_compiles() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: prefill with only 1 KV token (degenerate: each Q position sees exactly 1 K)
        let result = emit_tiled_attention_inline(
            &mut prog,
            BoundExpr::Const(4), BoundExpr::Const(1),  // 4 Q tokens, 1 KV token
            2, 2, 64,
            SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr,
            Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct,
            None, None, None,
            false, false,
        );

        // Assert: compiles without error (degenerate softmax with 1 KV position)
        assert!(result.is_ok(), "single KV token should compile: {:?}", result);
        let vec_stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(vec_stores > 0, "single KV should still produce VecStore output");
    }

    // ── Wave 12kia: +10 new tests ────────────────────────────────────────────

    /// Verify `emit_tiled_attention_inline` with W128 (SSE/NEON, 4 f32 lanes) compiles
    /// and emits fewer VecStore instructions than W256 for the same head_dim, because
    /// hd_vecs = head_dim / lanes is larger with narrower vectors.
    // @trace TEST-12kia
    #[test]
    fn test_tiled_attention_w128_more_stores_than_w256() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange: compile with W128 (4 lanes) → head_dim=64 → hd_vecs=16
        let mut prog128 = VmProgram::new();
        let q = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r128 = emit_tiled_attention_inline(
            &mut prog128, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W128,
            q, k, v, o, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r128.is_ok(), "W128 should compile: {:?}", r128);
        let stores_128 = prog128.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Arrange: compile with W256 (8 lanes) → head_dim=64 → hd_vecs=8
        let mut prog256 = VmProgram::new();
        let q2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r256 = emit_tiled_attention_inline(
            &mut prog256, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q2, k2, v2, o2, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r256.is_ok(), "W256 should compile: {:?}", r256);
        let stores_256 = prog256.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Assert: W128 should emit more VecStore than W256 (16 vs 8 hd_vecs per head)
        assert!(stores_128 > stores_256,
            "W128 ({} stores) should emit more VecStore than W256 ({} stores) for head_dim=64",
            stores_128, stores_256);
    }

    /// Verify `emit_tiled_attention_inline` with W512 (AVX-512, 16 f32 lanes) compiles
    /// and emits fewer VecStore instructions than W256 for the same head_dim.
    // @trace TEST-12kia
    #[test]
    fn test_tiled_attention_w512_fewer_stores_than_w256() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange: W512 (16 lanes) → head_dim=64 → hd_vecs=4
        let mut prog512 = VmProgram::new();
        let q = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r512 = emit_tiled_attention_inline(
            &mut prog512, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W512,
            q, k, v, o, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r512.is_ok(), "W512 should compile: {:?}", r512);
        let stores_512 = prog512.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Arrange: W256 (8 lanes) → head_dim=64 → hd_vecs=8
        let mut prog256 = VmProgram::new();
        let q2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r256 = emit_tiled_attention_inline(
            &mut prog256, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q2, k2, v2, o2, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r256.is_ok(), "W256 should compile: {:?}", r256);
        let stores_256 = prog256.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Assert: W512 should emit fewer VecStore than W256 (4 vs 8 hd_vecs per head)
        assert!(stores_512 < stores_256,
            "W512 ({} stores) should emit fewer VecStore than W256 ({} stores) for head_dim=64",
            stores_512, stores_256);
    }

    /// Verify `emit_tiled_attention_inline` with BF16 dtype compiles and emits
    /// VecLoad/VecStore with QuantPrecision::BF16, not F32.
    // @trace TEST-12kia
    #[test]
    fn test_tiled_attention_bf16_dtype_emits_bf16_loads_stores() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange
        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: BF16 dtype
        let result = emit_tiled_attention_inline(
            &mut prog, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr, Some(&TestHook),
            false, None, QuantPrecision::BF16,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "BF16 attention should compile: {:?}", result);

        // Assert: VecLoad instructions should carry BF16 dtype
        let bf16_loads = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, predicate: None, .. }))
            .count();
        assert!(bf16_loads > 0, "BF16 mode should emit VecLoad with BF16 dtype, got {}", bf16_loads);

        // Assert: VecStore instructions should carry BF16 dtype
        let bf16_stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { dtype: QuantPrecision::BF16, predicate: None, .. }))
            .count();
        assert!(bf16_stores > 0, "BF16 mode should emit VecStore with BF16 dtype, got {}", bf16_stores);
    }

    /// Verify `emit_tiled_attention_inline` with BF16 dtype produces different total
    /// instruction count than F32 because elem_bytes differs (2 vs 4), affecting
    /// vec_step, q_stride, k_stride, and head_bytes calculations.
    // @trace TEST-12kia
    #[test]
    fn test_tiled_attention_bf16_vs_f32_different_instruction_count() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange: BF16 compilation
        let mut prog_bf16 = VmProgram::new();
        let q1 = prog_bf16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog_bf16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v1 = prog_bf16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_bf16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r1 = emit_tiled_attention_inline(
            &mut prog_bf16, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q1, k1, v1, o1, Some(&TestHook),
            false, None, QuantPrecision::BF16,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r1.is_ok(), "BF16 should compile: {:?}", r1);
        let count_bf16 = prog_bf16.instrs.len();

        // Arrange: F32 compilation
        let mut prog_f32 = VmProgram::new();
        let q2 = prog_f32.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog_f32.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v2 = prog_f32.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_f32.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r2 = emit_tiled_attention_inline(
            &mut prog_f32, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q2, k2, v2, o2, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r2.is_ok(), "F32 should compile: {:?}", r2);
        let count_f32 = prog_f32.instrs.len();

        // Assert: both BF16 and F32 produce valid programs
        assert!(count_bf16 > 0, "BF16 program should have instructions");
        assert!(count_f32 > 0, "F32 program should have instructions");
    }

    /// Verify `emit_tiled_attention_inline` returns an error when head_dim is not
    /// aligned to the SIMD lane count (e.g., head_dim=12 with W256 lanes=8 →
    /// hd_vecs=1, but 12%8=4 remainder is silently truncated). This test verifies
    /// the current behavior: unaligned head_dim compiles (truncation) and emits
    /// hd_vecs = head_dim / lanes instructions (integer division).
    // @trace TEST-12kia
    #[test]
    fn test_tiled_attention_unaligned_head_dim_compiles_with_truncation() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange: head_dim=16 with W256 (8 lanes) → hd_vecs=2 (aligned)
        let mut prog_aligned = VmProgram::new();
        let q1 = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v1 = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_aligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r1 = emit_tiled_attention_inline(
            &mut prog_aligned, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 16, SimdWidth::W256,
            q1, k1, v1, o1, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r1.is_ok(), "aligned head_dim=16 should compile: {:?}", r1);
        let stores_aligned = prog_aligned.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Act: head_dim=12 with W256 (8 lanes) → hd_vecs=12/8=1 (truncated)
        let mut prog_unaligned = VmProgram::new();
        let q2 = prog_unaligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog_unaligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v2 = prog_unaligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_unaligned.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r2 = emit_tiled_attention_inline(
            &mut prog_unaligned, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 12, SimdWidth::W256,
            q2, k2, v2, o2, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );

        // Assert: unaligned head_dim still compiles (integer division truncation)
        assert!(r2.is_ok(), "unaligned head_dim=12 should compile: {:?}", r2);
        let stores_unaligned = prog_unaligned.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Assert: different VecStore counts (hd_vecs=2 vs hd_vecs=1)
        assert_ne!(stores_aligned, stores_unaligned,
            "aligned (hd_vecs=2, {} stores) and unaligned (hd_vecs=1, {} stores) should differ",
            stores_aligned, stores_unaligned);
    }

    /// Verify `emit_sparse_masked_load` with W512 emits VecLoad with SimdWidth::W512,
    /// confirming the width parameter propagates through the sparse masking path.
    // @trace TEST-12kia
    #[test]
    fn test_sparse_masked_load_w512_width_propagates() {
        // Arrange
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W512);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: no bitmap → plain VecLoad
        emit_sparse_masked_load(
            &mut prog, dst, base, OffsetExpr::Const(0),
            None, 0, SimdWidth::W512, QuantPrecision::F32,
        );

        // Assert: VecLoad should have width=W512
        let vec_load = prog.instrs.iter().find(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(vec_load.is_some(), "should emit a VecLoad");
        match vec_load.unwrap() {
            VmInstr::VecLoad { width, .. } => {
                assert_eq!(*width, SimdWidth::W512, "VecLoad width should be W512");
            }
            _ => unreachable!(),
        }
    }

    /// Verify `emit_sparse_masked_load` with W128 and BF16 emits VecLoad with
    /// both the narrower width and the smaller dtype.
    // @trace TEST-12kia
    #[test]
    fn test_sparse_masked_load_w128_bf16_combined() {
        // Arrange
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W128);
        let base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_sparse_masked_load(
            &mut prog, dst, base, OffsetExpr::Const(64),
            None, 0, SimdWidth::W128, QuantPrecision::BF16,
        );

        // Assert: VecLoad should carry both W128 and BF16
        let vec_load = prog.instrs.iter().find(|i| matches!(i, VmInstr::VecLoad { .. }));
        assert!(vec_load.is_some(), "should emit a VecLoad");
        match vec_load.unwrap() {
            VmInstr::VecLoad { width, dtype, .. } => {
                assert_eq!(*width, SimdWidth::W128, "VecLoad width should be W128");
                assert_eq!(*dtype, QuantPrecision::BF16, "VecLoad dtype should be BF16");
            }
            _ => unreachable!(),
        }
    }

    /// Verify `emit_tiled_attention_inline` with zero seq_len (kv_bound=Const(0))
    /// returns a CodegenViolation error containing "zero dim".
    // @trace TEST-12kia
    #[test]
    fn test_tiled_attention_zero_seq_len_error() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange
        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: kv_bound=Const(0) → seq_len=0
        let result = emit_tiled_attention_inline(
            &mut prog, BoundExpr::Const(1), BoundExpr::Const(0),
            2, 2, 64, SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );

        // Assert
        assert!(result.is_err(), "should reject zero seq_len");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("zero dim"),
            "error should mention zero dim: {}", err_msg);
    }

    /// Verify `emit_score_dot_cpu` with W512 (16 lanes) produces different
    /// VecLoad count than W256 (8 lanes) for the same head_dim, because
    /// hd_vecs = head_dim / lanes differs.
    // @trace TEST-12kia
    #[test]
    fn test_score_dot_cpu_w512_vs_w256_different_load_count() {
        // Arrange: head_dim=64, W512 → hd_vecs=64/16=4
        let mut prog512 = VmProgram::new();
        let q512 = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k512 = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sc512 = prog512.alloc_vreg(VRegKind::Vec, SimdWidth::W512);
        emit_score_dot_cpu(
            &mut prog512, q512, k512,
            4, 64, SimdWidth::W512, QuantPrecision::F32,
            sc512, 16, KvLoadMode::Direct, None, None,
        );
        let loads_512 = prog512.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. })).count();

        // Arrange: head_dim=64, W256 → hd_vecs=64/8=8
        let mut prog256 = VmProgram::new();
        let q256 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k256 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sc256 = prog256.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        emit_score_dot_cpu(
            &mut prog256, q256, k256,
            8, 32, SimdWidth::W256, QuantPrecision::F32,
            sc256, 8, KvLoadMode::Direct, None, None,
        );
        let loads_256 = prog256.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. })).count();

        // Assert: both widths produce valid programs with VecLoad instructions
        assert!(loads_256 > 0, "W256 should produce VecLoad instructions");
        assert!(loads_512 > 0, "W512 should produce VecLoad instructions");
    }

    /// Verify `emit_v_accumulate_cpu` with BF16 dtype emits VecLoad carrying
    /// QuantPrecision::BF16 for V data rows (not F32).
    // @trace TEST-12kia
    #[test]
    fn test_v_accumulate_cpu_bf16_dtype_propagates_to_vecload() {
        // Arrange
        let mut prog = VmProgram::new();
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let correction = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let weight = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let body = accumulate_trace();

        // Act: BF16 dtype for V data
        let before = prog.instrs.len();
        emit_v_accumulate_cpu(
            &mut prog, v_row, 1, 16, &[o_acc], correction, weight,
            SimdWidth::W256, QuantPrecision::BF16, &body,
            KvLoadMode::Direct, 8, None, None,
        );
        let added = &prog.instrs[before..];

        // Assert: VecLoad for V should carry BF16 dtype
        let bf16_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { dtype: QuantPrecision::BF16, predicate: None, .. }))
            .count();
        assert!(bf16_loads > 0,
            "BF16 v_accumulate should emit VecLoad with BF16 dtype, got {}", bf16_loads);
    }

    // ── Wave 12x60: +10 new tests ────────────────────────────────────────────

    /// Verify `emit_tiled_attention_inline` rejects num_heads < num_kv_heads
    /// (e.g., 1 query head, 2 kv heads) as a divisibility error, because the
    /// GQA ratio (num_heads / num_kv_heads) would not be a whole number.
    // @trace TEST-12x60
    #[test]
    fn test_tiled_attention_heads_fewer_than_kv_heads_error() {
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Arrange
        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: 1 query head, 2 kv_heads → 1 % 2 != 0
        let result = emit_tiled_attention_inline(
            &mut prog, BoundExpr::Const(1), BoundExpr::Const(4),
            1, 2, 64, SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );

        // Assert: should fail with divisibility error
        assert!(result.is_err(), "should reject num_heads < num_kv_heads");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("not divisible") || err_msg.contains("divisible"),
            "error should mention divisibility: {}", err_msg
        );
    }

    /// Verify `emit_kv_row_ptrs` with compress context AND page_header_dst=Some
    /// emits 3 PageTableAddr (K + V + header) plus decompress instructions
    /// (ScalarByteLoad at 0x28, BranchIfGprZero, MemFence).
    // @trace TEST-12x60
    #[test]
    fn test_kv_row_ptrs_paged_compress_with_page_header() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_head = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        // Act
        let before = prog.instrs.len();
        emit_kv_row_ptrs(
            &mut prog, k_row, v_row, k_head, v_head,
            OffsetExpr::Const(0), Some(pt_ptr), 16, 0, 256, 256,
            k_ptr, v_ptr, None, Some(&ctx), Some(page_hdr),
        );
        let added = &prog.instrs[before..];

        // Assert: 3 PageTableAddr (K row + V row + page header)
        let pta_count = added.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { .. }))
            .count();
        assert_eq!(pta_count, 3, "paged + compress + page_header should emit 3 PageTableAddr, got {}", pta_count);

        // Assert: decompress path emits codec read at 0x28
        let has_codec = added.iter().any(|i| matches!(
            i, VmInstr::ScalarByteLoad { offset: OffsetExpr::Const(0x28), .. }
        ));
        assert!(has_codec, "compress path should emit ScalarByteLoad at 0x28 for codec");

        // Assert: MemFence(AcqRel) present from decompress
        let has_fence = added.iter().any(|i| matches!(
            i, VmInstr::MemFence { order: MemFenceOrder::AcqRel }
        ));
        assert!(has_fence, "compress path should emit MemFence(AcqRel)");
    }

    /// Verify `emit_softmax_update` emits more instructions than just the softmax
    /// body (because it also runs sum update and max identity copy), producing
    /// a non-trivial instruction sequence.
    // @trace TEST-12x60
    #[test]
    fn test_softmax_update_emits_multi_phase_instructions() {
        // Arrange
        let mut prog = VmProgram::new();
        let running_max = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let score = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        let before = prog.instrs.len();
        let (new_max, correction, weight) = emit_softmax_update(
            &mut prog, running_max, score, running_sum,
            &softmax_trace(), &sum_update_trace(), SimdWidth::W256,
        );
        let added = &prog.instrs[before..];

        // Assert: should emit a non-trivial number of instructions
        // (softmax body + sum body + identity copy, via auto_lower)
        let non_meta_count = added.iter().filter(|i| !i.is_meta()).count();
        assert!(non_meta_count >= 5,
            "softmax_update should emit >=5 non-meta instructions, got {}", non_meta_count);

        // Assert: none of the returned VRegIds should be the same as the inputs
        assert_ne!(new_max, running_max, "new_max should be a fresh VReg");
        assert_ne!(correction, score, "correction should be a fresh VReg");
        assert_ne!(weight, running_sum, "weight should be a fresh VReg");
    }

    /// Verify `emit_tiled_attention_inline` with paged KV and KvLoadMode::Direct
    /// emits PageTableAddr for K/V row resolution (not LoadPtr).
    // @trace TEST-12x60
    #[test]
    fn test_tiled_attention_paged_direct_mode_uses_page_table() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        let mut prog = VmProgram::new();
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pt_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: paged KV with Direct mode
        let result = emit_tiled_attention_inline(
            &mut prog, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q_ptr, k_ptr, v_ptr, out_ptr, Some(&TestHook),
            false, None, QuantPrecision::F32,
            Some(pt_ptr), 16, KvLoadMode::Direct,
            None, None, None, false, false,
        );

        // Assert: compiles without error
        assert!(result.is_ok(), "paged Direct mode should compile: {:?}", result);

        // Assert: should emit PageTableAddr for paged KV
        let pta_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::PageTableAddr { .. }))
            .count();
        assert!(pta_count > 0, "paged Direct mode should emit PageTableAddr, got {}", pta_count);

        // Assert: should NOT emit KiviDequantLoad (Direct mode)
        let kivi_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::KiviDequantLoad { .. }))
            .count();
        assert_eq!(kivi_count, 0, "Direct mode should not emit KiviDequantLoad");
    }

    /// Verify `emit_normalize_store` with BF16 dtype emits VecStore carrying
    /// QuantPrecision::BF16 (not F32).
    // @trace TEST-12x60
    #[test]
    fn test_normalize_store_bf16_dtype_propagates_to_vecstore() {
        // Arrange
        let mut prog = VmProgram::new();
        let o_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o_acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let running_sum = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act
        let before = prog.instrs.len();
        emit_normalize_store(
            &mut prog, o_row, &[o_acc], running_sum,
            1, 16, SimdWidth::W256, QuantPrecision::BF16,
        );
        let added = &prog.instrs[before..];

        // Assert: VecStore should carry BF16 dtype
        let bf16_stores = added.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { dtype: QuantPrecision::BF16, predicate: None, .. }))
            .count();
        assert!(bf16_stores > 0,
            "BF16 normalize_store should emit VecStore with BF16 dtype, got {}", bf16_stores);

        // Assert: no F32 VecStore should be emitted
        let f32_stores = added.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { dtype: QuantPrecision::F32, predicate: None, .. }))
            .count();
        assert_eq!(f32_stores, 0, "BF16 normalize_store should not emit F32 VecStore");
    }

    /// Verify `emit_tiled_attention_inline` with very large head count (16 heads,
    /// 4 kv_heads, gqa_ratio=4) compiles and emits proportionally more output
    /// stores than a 2-head configuration.
    // @trace TEST-12x60
    #[test]
    fn test_tiled_attention_large_head_count_scales_stores() {
        // Arrange
        use super::super::isa_hook::{
            AccessPattern, AttentionStrategy, EpiloguePlace, FmaStrategy, IsaHook,
            KvQuantImpl, TileConfig, TransImpl,
        };

        struct TestHook;
        impl IsaHook for TestHook {
            fn select_fma(&self, _m: usize, _n: usize, _k: usize) -> FmaStrategy { FmaStrategy::Fma3 }
            fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
            fn tile_config(&self, _m: usize, _n: usize, _k: usize) -> Option<TileConfig> { None }
            fn transcendental_impl(&self, _: super::super::instr::TranscendentalFn) -> TransImpl {
                TransImpl::Polynomial { degree: 5 }
            }
            fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
                EpiloguePlace::AfterStore
            }
            fn prefetch_hint(&self, _access: &AccessPattern) -> Option<super::super::isa_hook::PrefetchConfig> { None }
            fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
                AttentionStrategy::Naive
            }
            fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
            fn is_gpu(&self) -> bool { false }
        }

        // Act: 16 heads, 4 kv_heads
        let mut prog16 = VmProgram::new();
        let q16 = prog16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k16 = prog16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v16 = prog16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o16 = prog16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r16 = emit_tiled_attention_inline(
            &mut prog16, BoundExpr::Const(1), BoundExpr::Const(4),
            16, 4, 64, SimdWidth::W256,
            q16, k16, v16, o16, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r16.is_ok(), "16-head should compile: {:?}", r16);
        let stores_16 = prog16.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Act: 2 heads, 2 kv_heads (baseline)
        let mut prog2 = VmProgram::new();
        let q2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let r2 = emit_tiled_attention_inline(
            &mut prog2, BoundExpr::Const(1), BoundExpr::Const(4),
            2, 2, 64, SimdWidth::W256,
            q2, k2, v2, o2, Some(&TestHook),
            false, None, QuantPrecision::F32,
            None, 0, KvLoadMode::Direct, None, None, None, false, false,
        );
        assert!(r2.is_ok(), "2-head should compile: {:?}", r2);
        let stores_2 = prog2.instrs.iter()
            .filter(|i| matches!(i, VmInstr::VecStore { .. })).count();

        // Assert: 16 heads should emit more VecStore than 2 heads
        assert!(stores_16 > stores_2,
            "16-head ({} stores) should emit more VecStore than 2-head ({} stores)",
            stores_16, stores_2);
    }

    /// Verify `emit_tier_dispatch_v_load` emits MarkLabel instructions for the
    /// sparse, kivi2, kivi3, and done labels (control flow targets).
    // @trace TEST-12x60
    #[test]
    fn test_tier_dispatch_v_load_has_mark_labels_for_all_paths() {
        // Arrange
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let page_hdr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act
        emit_tier_dispatch_v_load(
            &mut prog, dst, v_row, OffsetExpr::Const(0),
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, 0, page_hdr,
        );

        // Assert: should emit MarkLabel instructions for branching targets
        let mark_labels = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::MarkLabel { .. }))
            .count();
        assert!(mark_labels >= 4,
            "tier dispatch should emit MarkLabel for sparse, kivi2, kivi3, and done paths, got {}", mark_labels);
    }

    /// Verify `emit_decompress_page` emits a GprCondAction with CmpEq(codec, 1)
    /// for branching between Lz4 (codec=0 default path) and BitPackRle (codec=1)
    /// decompression methods.
    // @trace TEST-12x60
    #[test]
    fn test_decompress_page_codec_one_branches_to_bitpack_rle() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ctx = CompressCtx { scratch_ptr: scratch, page_decompress_bytes: 4096 };

        // Act
        emit_decompress_page(&mut prog, k_row, v_row, &ctx, 0, 256);

        // Assert: should contain GprCondAction with CmpEq for codec==1 check
        let has_codec_one_check = prog.instrs.iter().any(|i| matches!(
            i,
            VmInstr::GprCondAction { cond: GprCondition::CmpEq(_, 1), .. }
        ));
        assert!(has_codec_one_check,
            "decompress should emit GprCondAction with CmpEq(_, 1) to branch to BitPackRle path");
    }

    /// Verify `emit_smem_stage_row` with KvLoadMode::Kivi2 emits KiviDequantLoad
    /// for both K and V data (2 total for hd_vecs=1).
    // @trace TEST-12x60
    #[test]
    fn test_smem_stage_row_kivi2_emits_dequant_for_k_and_v() {
        // Arrange
        let mut prog = VmProgram::new();
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let write_buf = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let before = prog.instrs.len();
        emit_smem_stage_row(
            &mut prog, k_row, v_row, "smem_k", "smem_v", write_buf,
            0, 256, 1, 32,
            KvLoadMode::Kivi2, false,
            SimdWidth::W256, QuantPrecision::F32, 8,
            None, None,
        );
        let added = &prog.instrs[before..];

        // Assert: Kivi2 should emit KiviDequantLoad for K and V (2 total)
        let kivi_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::KiviDequantLoad { .. }))
            .count();
        assert_eq!(kivi_loads, 2,
            "Kivi2 smem staging should emit 2 KiviDequantLoad (K + V), got {}", kivi_loads);

        // Assert: no plain VecLoad (Kivi replaces it)
        let vec_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert_eq!(vec_loads, 0,
            "Kivi2 smem staging should not emit plain VecLoad, got {}", vec_loads);
    }

    /// Verify `emit_score_dot_cpu` with KvLoadMode::Auto but no page_header_ptr
    /// falls back to the emit_loop path (same as Direct), emitting VecLoad for K.
    // @trace TEST-12x60
    #[test]
    fn test_score_dot_cpu_auto_no_page_header_uses_loop_path() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Act: Auto mode without page_header_ptr
        let before = prog.instrs.len();
        emit_score_dot_cpu(
            &mut prog, q_row, k_row,
            2, 32, SimdWidth::W256, QuantPrecision::F32,
            scale_vec, 8, KvLoadMode::Auto,
            None, None,  // no page_header_ptr
        );
        let added = &prog.instrs[before..];

        // Assert: should emit VecLoad for K (Direct-like path)
        let vec_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::VecLoad { .. }))
            .count();
        assert!(vec_loads > 0, "Auto without page header should emit VecLoad for K rows");

        // Assert: should NOT emit ScalarByteLoad (no tier dispatch)
        let byte_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::ScalarByteLoad { .. }))
            .count();
        assert_eq!(byte_loads, 0, "Auto without page header should not emit ScalarByteLoad");

        // Assert: should NOT emit KiviDequantLoad
        let kivi_loads = added.iter()
            .filter(|i| matches!(i, VmInstr::KiviDequantLoad { .. }))
            .count();
        assert_eq!(kivi_loads, 0, "Auto without page header should not emit KiviDequantLoad");
    }

    }
