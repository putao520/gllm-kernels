//! Buffer lifetime analysis and interval graph coloring.
//!
//! Analyzes tensor lifetimes across the execution schedule and assigns
//! buffer offsets to minimize total scratchpad memory via greedy interval
//! graph coloring.
//!
//! SPEC 28 REQ-GRP-001: Extended with `TensorLifetime` cross-layer dimension
//! and `CrossLayerLifetime` for single mega-kernel resource planning.

use std::collections::{HashMap, HashSet};
use crate::compiler::graph::{CompilerGraph, TensorId, OpId};
use crate::compiler::fusion::{FusionPlan, FusionMode};
use crate::compiler::virtual_tensor::VirtualTensorMap;
use crate::compiler::virtual_activation::VirtualActivationMap;
use crate::compiler::layout_negotiator::LayoutAssignment;
use crate::types::DType;

/// §0.2.3 一个 CompilerGraph 张量在运行时的物理位置分类。
///
/// R3 BufferAllocation 输出 — 在 buffer 分配阶段一次性推导，
/// ISA Lowering codegen 直接消费，无需独立推导逻辑。
#[derive(Debug, Clone, Copy)]
pub enum TensorPtrSource {
    /// graph.inputs[0] — ABI activation arg。
    Activation,
    /// §0.2.8 Ping-pong activation input — scratch[ping_offset]。
    /// 层循环中当前层读取的 activation buffer。
    ActivationPing,
    /// §0.2.8 Ping-pong activation output — scratch[pong_offset]。
    /// 层循环中当前层写入的 activation buffer。
    /// 每层末尾 ActivationSwap 交换 ping/pong，下一层 input=pong, output=ping。
    ActivationPong,
    /// graph.inputs[1..] — ABI weights arg + weight_layout 内 byte offset。
    Weight { offset: usize },
    /// 中间张量 (非 graph.inputs/outputs) — scratchpad + BufferAllocation offset。
    Intermediate { offset: usize },
    /// graph.outputs[i] — ABI output arg + max_alloc 累加偏移。
    Output { offset: usize },
}

/// Trace alias chain to find the physical (non-aliased) tensor.
///
/// When activation_input is an alias tensor (e.g. SgInject output, which
/// `output_aliases_input()` maps to its input), we need to find the actual
/// tensor with a scratchpad allocation so we can extend its lifetime.
fn resolve_alias_to_physical(
    tid: TensorId,
    alias_outputs: &HashSet<TensorId>,
    graph: &CompilerGraph,
) -> Option<TensorId> {
    let mut current = tid;
    let mut visited = HashSet::new();
    while alias_outputs.contains(&current) {
        if !visited.insert(current) {
            break; // cycle detection
        }
        let producer_op = graph.ops.iter().find(|op| op.outputs.contains(&current))?;
        let alias_input = producer_op.op_output_aliases_input(graph)?;
        current = *producer_op.inputs.get(alias_input)?;
    }
    Some(current)
}

/// A tensor's lifetime interval: [first_use, last_use] in schedule order.
#[derive(Debug, Clone, Copy)]
pub struct Lifetime {
    /// Which tensor this lifetime belongs to.
    pub tensor_id: TensorId,
    /// Schedule step of first use (read or write).
    pub first_use: usize,
    /// Schedule step of last use (read or write).
    pub last_use: usize,
    /// Buffer size in bytes.
    pub size_bytes: usize,
}

/// Buffer assignment: tensor mapped to (offset, size) in the scratchpad.
#[derive(Debug, Clone, Copy)]
pub struct BufferSlot {
    /// Which tensor occupies this slot.
    pub tensor_id: TensorId,
    /// Byte offset within the scratchpad.
    pub offset: usize,
    /// Size of this buffer in bytes.
    pub size_bytes: usize,
}

/// Result of buffer allocation.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct BufferAllocation {
    /// Per-tensor buffer assignments.
    pub slots: Vec<BufferSlot>,
    /// Total scratchpad bytes required.
    pub total_bytes: usize,
    /// Number of intermediate tensors allocated.
    pub num_tensors: usize,
    /// Bytes saved compared to naive allocation (sum of all tensor sizes).
    pub bytes_saved: usize,
    /// R2: Virtual tensor → physical source tensor ID mapping.
    pub virtual_source_map: HashMap<TensorId, usize>,
    /// R2.5: Activation tensor → fixed slot index.
    pub activation_slots: HashMap<TensorId, usize>,
    /// R2: Tensors skipped because they are virtual (no physical buffer needed).
    pub skipped_virtual: HashSet<TensorId>,
    /// §0.2.3 R3: Tensor → 物理位置分类 (Activation/Weight/Intermediate/Output)
    /// 在 buffer 分配阶段一次性推导，ISA Lowering codegen 直接消费。
    pub tensor_sources: HashMap<TensorId, TensorPtrSource>,
}

impl BufferAllocation {
    /// Look up scratchpad offset for an intermediate tensor.
    pub fn offset_of(&self, tid: TensorId) -> Option<usize> {
        self.slots.iter()
            .find(|s| s.tensor_id == tid)
            .map(|s| s.offset)
    }

    fn empty() -> Self {
        Self::default()
    }
}


/// Analyze tensor lifetimes from the fusion plan's execution order.
///
/// Only intermediate tensors are included — graph inputs and outputs are
/// externally managed and excluded from scratchpad allocation.
pub fn analyze_lifetimes(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    vtm: Option<&VirtualTensorMap>,
    vam: Option<&VirtualActivationMap>,
) -> Vec<Lifetime> {
    // Build schedule order: flatten fusion groups
    let schedule: Vec<OpId> = plan
        .groups
        .iter()
        .flat_map(|g| g.ops.iter().copied())
        .collect();

    let mut first_use: HashMap<TensorId, usize> = HashMap::new();
    let mut last_use: HashMap<TensorId, usize> = HashMap::new();

    for (step, &op_id) in schedule.iter().enumerate() {
        if let Some(op) = graph.op(op_id) {
            for &tid in op.inputs.iter().chain(op.outputs.iter()) {
                first_use.entry(tid).or_insert(step);
                last_use.insert(tid, step);
            }
        }
    }


    // Exclude graph inputs/outputs (externally managed)
    let graph_io: HashSet<TensorId> = graph
        .inputs
        .iter()
        .chain(graph.outputs.iter())
        .copied()
        .collect();

    // Collect all activation alias output tensors — they share the same physical
    // buffer as the activation input (in-place update for layer loop) and must
    // not get their own scratchpad allocation.
    let mut alias_outputs: HashSet<TensorId> = HashSet::new();
    if let Some(cfg) = &graph.layer_loop_config {
        if let Some((_, out_tid)) = cfg.activation_alias {
            alias_outputs.insert(out_tid);
        }
    }
    if let Some(cfg) = &graph.hetero_layer_loop_config {
        for &(_, out_tid) in &cfg.activation_aliases {
            alias_outputs.insert(out_tid);
        }
    }
    // Generic output-alias detection: any op whose output_aliases_input() returns
    // Some(n) means output[0] shares the same physical slot as input[n]. No separate
    // scratchpad allocation needed — the output reuses the input's slot via
    // TensorPtrSource aliasing (see TensorPtrResolver::build).
    for op_id in &schedule {
        if let Some(op) = graph.op(*op_id) {
            if op.op_output_aliases_input(graph).is_some() {
                if let Some(&out_tid) = op.outputs.first() {
                    alias_outputs.insert(out_tid);
                }
            }
        }
    }

    // The activation input tensor must stay alive for the entire schedule
    // because all alias outputs write back to it in-place across iterations.
    // Without extending its lifetime, the allocator would freely reuse offset 0
    // for other intermediates, causing data corruption.
    //
    // When activation_input is an alias tensor (e.g. SgInject output), trace
    // through alias chain to find the physical tensor with actual allocation.
    let mut activation_input: Option<TensorId> = None;
    if let Some(cfg) = &graph.layer_loop_config {
        if let Some((in_tid, _)) = cfg.activation_alias {
            activation_input = resolve_alias_to_physical(in_tid, &alias_outputs, graph);
        }
    }
    if let Some(cfg) = &graph.hetero_layer_loop_config {
        if let Some(&(in_tid, _)) = cfg.activation_aliases.first() {
            activation_input = Some(in_tid);
        }
    }

    let schedule_len = schedule.len();

    // DEBUG: print schedule ops and their inputs/outputs
    if std::env::var("GLLM_DEBUG_BUFFER_ALLOC").is_ok() {
        eprintln!("[buf-alloc] === SCHEDULE ({} ops) ===", schedule.len());
        for (step, &op_id) in schedule.iter().enumerate() {
            if let Some(op) = graph.op(op_id) {
                let in_names: Vec<String> = op.inputs.iter()
                    .filter_map(|&tid| graph.tensor(tid).map(|t| format!("{}({:?})", t.name, tid)))
                    .collect();
                let out_names: Vec<String> = op.outputs.iter()
                    .filter_map(|&tid| graph.tensor(tid).map(|t| format!("{}({:?})", t.name, tid)))
                    .collect();
                eprintln!("[buf-alloc]   step {:3}: {:?} inputs=[{}] outputs=[{}]",
                    step, op.op, in_names.join(", "), out_names.join(", "));
            }
        }
    }

    // §0.2.3: Logits tensor (last GEMM output before Argmax) is managed by
    // mega-kernel output_ptr (scratchpad logits region), NOT by intermediate
    // scratchpad allocation.  Exclude it to avoid allocating max_seq_len *
    // vocab_size bytes (e.g. 32768 * 151669 * 4 = 18.5 GB) in scratchpad.
    let logits_output_tid: Option<TensorId> = graph.ops.iter().rev()
        .find_map(|op| {
            // The logits producer is the last QuantGemm or Gemm op whose output
            // feeds into Argmax/WriteLogits/StoreToken downstream.
            if op.op_is_quant_gemm(graph) || op.op_is_gemm_like(graph) {
                op.outputs.first().copied()
            } else {
                None
            }
        })
        .and_then(|tid| {
            // Only exclude if this tensor is consumed by Argmax or WriteLogits
            // (i.e. it IS the logits, not a regular GEMM output).
            let consumer_is_logits_sink = graph.ops.iter().any(|op| {
                op.inputs.contains(&tid) &&
                    matches!(op.op_resolved(graph),
                        Some(crate::compiler::graph::Op::Argmax { .. }) |
                        Some(crate::compiler::graph::Op::WriteLogits { .. }))
            });
            if consumer_is_logits_sink { Some(tid) } else { None }
        });

    let mut lifetimes = Vec::new();
    for tensor in &graph.tensors {
        if graph_io.contains(&tensor.id) {
            continue;
        }
        if alias_outputs.contains(&tensor.id) {
            continue;
        }
        // §0.2.3: Logits tensor managed by mega-kernel output_ptr — skip
        if logits_output_tid == Some(tensor.id) {
            continue;
        }
        // R2: 跳过虚拟 tensor — 已被 DataFlowOptimizer 消除，不需要物理 buffer
        if let Some(vtm) = &vtm {
            if vtm.is_virtual(tensor.id) {
                continue;
            }
        }
        if let (Some(&first), Some(&last)) =
            (first_use.get(&tensor.id), last_use.get(&tensor.id))
        {
            // Use max_for_allocation to account for symbolic dimensions (SymDim).
            // concrete_bytes() treats Symbolic as 1, causing massive under-allocation
            // for intermediate tensors with symbolic seq_len (e.g. FusedQkvRope Q/K outputs).
            // ARCH-DTYPE-JIT-TYPED: elem_bytes 始终从 tensor.dtype 推断，禁止硬编码 F32。
            // 中间张量的 dtype 由 dtype 传播链在图构建时设置，已反映真实计算精度。
            let elem_bytes = tensor.dtype.size_bytes();
            // ARCH-SYMDIM: buffer 分配用 graph.max_seq_len 作为 Symbolic 维度上界
            let numel: usize = tensor.shape.iter()
                .map(|d| d.max_for_allocation(graph.max_seq_len))
                .product::<usize>()
                .max(1);
            let size_bytes = numel * elem_bytes;
            if size_bytes > 0 {
                // Activation input must stay live for the entire schedule so
                // that alias outputs can write back to it in-place across all
                // layer iterations without other tensors clobbering offset 0.
                let effective_last = if activation_input == Some(tensor.id) {
                    schedule_len.saturating_sub(1)
                } else {
                    last
                };
                if std::env::var("GLLM_DEBUG_BUFFER_ALLOC").is_ok() {
                    eprintln!("[buf-alloc] {:40} {:?} lifetime=[{},{}] size={}",
                        tensor.name, tensor.id, first, effective_last, size_bytes);
                }
                lifetimes.push(Lifetime {
                    tensor_id: tensor.id,
                    first_use: first,
                    last_use: effective_last,
                    size_bytes,
                });
            }
        }
    }

    lifetimes
}

/// Greedy interval graph coloring: assign buffer offsets to minimize total memory.
///
/// Algorithm: sort by first_use (ties broken by larger size first for better
/// packing), then greedily assign each tensor to the lowest offset where it
/// doesn't overlap with any currently live tensor.
pub fn allocate_buffers(lifetimes: &[Lifetime]) -> BufferAllocation {
    if lifetimes.is_empty() {
        return BufferAllocation::empty();
    }

    // Sort by first_use, ties broken by larger size first
    let mut sorted: Vec<&Lifetime> = lifetimes.iter().collect();
    sorted.sort_by_key(|l| (l.first_use, std::cmp::Reverse(l.size_bytes)));

    // Active allocations: (offset, end_offset, last_use)
    let mut active: Vec<(usize, usize, usize)> = Vec::new();
    let mut slots = Vec::new();
    let mut total_bytes = 0usize;

    for lt in &sorted {
        let offset = find_offset(lt, &active);
        let end = offset + lt.size_bytes;
        total_bytes = total_bytes.max(end);

        active.push((offset, end, lt.last_use));
        slots.push(BufferSlot {
            tensor_id: lt.tensor_id,
            offset,
            size_bytes: lt.size_bytes,
        });
    }

    let naive_total: usize = lifetimes.iter().map(|l| l.size_bytes).sum();

    BufferAllocation {
        num_tensors: slots.len(),
        slots,
        total_bytes,
        bytes_saved: naive_total.saturating_sub(total_bytes),
        virtual_source_map: HashMap::new(),
        activation_slots: HashMap::new(),
        skipped_virtual: HashSet::new(),
        tensor_sources: HashMap::new(),
    }
}

/// Greedy interval graph coloring with explicit cache-line alignment.
///
/// `cacheline_bytes` is the hardware cache line size (typically 64 or 128).
/// Every buffer offset is rounded up to this alignment boundary, which
/// eliminates false sharing and improves prefetch efficiency.
///
/// When `layout` (R1.5 LayoutAssignment) is provided, tensors with layout
/// constraints (e.g. PanelPacked, SharedMemTile) get enhanced alignment
/// based on the layout's tile boundaries (SPEC §0.2.11).
pub fn allocate_buffers_aligned(
    lifetimes: &[Lifetime],
    cacheline_bytes: usize,
    vtm: Option<&VirtualTensorMap>,
    vam: Option<&VirtualActivationMap>,
    graph: &CompilerGraph,
    layout: Option<&LayoutAssignment>,
) -> BufferAllocation {
    let align = cacheline_bytes.max(64);

    if lifetimes.is_empty() {
        return BufferAllocation::empty();
    }

    let mut sorted: Vec<&Lifetime> = lifetimes.iter().collect();
    sorted.sort_by_key(|l| (l.first_use, std::cmp::Reverse(l.size_bytes)));

    // §0.2.8: Activation ping-pong compression — 将 N 个 activation tensor 压缩为 2 个 buffer
    // VirtualActivationMap 已经将 activation tensor 映射到 buffer_idx=0 (ping) / 1 (pong)
    // 这里把它们从 sorted 中移除，单独分配 2 个等大 slot
    let mut activation_tids: HashSet<TensorId> = HashSet::new();
    let mut activation_buffer_size: usize = 0;
    if let Some(vam) = &vam {
        if vam.num_buffers == 2 && !vam.activation_assignments.is_empty() {
            activation_buffer_size = vam.buffer_size_bytes;
            for &tid in vam.activation_assignments.keys() {
                activation_tids.insert(tid);
            }
        }
    }

    // 分离 activation 和 non-activation lifetimes
    let (activation_lifetimes, non_activation_lifetimes): (Vec<_>, Vec<_>) =
        sorted.into_iter().partition(|lt| activation_tids.contains(&lt.tensor_id));
    sorted = non_activation_lifetimes;

    let mut active: Vec<(usize, usize, usize)> = Vec::new();
    let mut slots = Vec::new();
    let mut total_bytes = 0usize;

    // 先分配 2 个 ping-pong activation buffer (如果有的话)
    if activation_buffer_size > 0 {
        let buf_align = align;
        // Ping buffer at offset 0
        let ping_size = (activation_buffer_size + buf_align - 1) & !(buf_align - 1);
        slots.push(BufferSlot {
            tensor_id: TensorId(0xFFFF_FF00), // sentinel: ping buffer
            offset: 0,
            size_bytes: ping_size,
        });
        // Pong buffer at offset ping_size
        let pong_offset = ping_size;
        slots.push(BufferSlot {
            tensor_id: TensorId(0xFFFF_FF01), // sentinel: pong buffer
            offset: pong_offset,
            size_bytes: ping_size,
        });
        total_bytes = pong_offset + ping_size;
        active.push((0, ping_size, usize::MAX)); // live for entire schedule
        active.push((pong_offset, pong_offset + ping_size, usize::MAX));

        // 记录每个 activation tensor 的 slot index (ping=0, pong=1)
        if let Some(vam) = &vam {
            for (&tid, slot) in &vam.activation_assignments {
                let slot_idx = slot.buffer_idx; // 0=ping, 1=pong
                if std::env::var("GLLM_DEBUG_BUFFER_ALLOC").is_ok() {
                    eprintln!("[buf-alloc] activation {:?} → {} buffer idx={}", tid, slot_idx, slot.buffer_idx);
                }
                let _ = (tid, slot_idx);
            }
        }
    }

    for lt in &sorted {
        // R1.5 布局感知对齐: 如果 tensor 对应的 op 有 PanelPacked 等布局约束,
        // 使用更大的对齐 (tile 边界)
        let tensor_align = layout_align_for_tensor(lt.tensor_id, graph, layout, align);
        let offset = find_offset_aligned(lt, &active, tensor_align);
        let end = offset + lt.size_bytes;
        total_bytes = total_bytes.max(end);

        active.push((offset, end, lt.last_use));
        slots.push(BufferSlot {
            tensor_id: lt.tensor_id,
            offset,
            size_bytes: lt.size_bytes,
        });
    }

    // Round total up to alignment for consistent scratchpad sizing
    total_bytes = (total_bytes + align - 1) & !(align - 1);

    let naive_total: usize = lifetimes.iter().map(|l| l.size_bytes).sum();


    if std::env::var("GLLM_DEBUG_BUFFER_ALLOC").is_ok() {
        eprintln!("[buf-alloc] === ALLOCATION RESULTS ===");
        eprintln!("[buf-alloc] total_bytes={}, naive={}, saved={}", total_bytes, naive_total, naive_total.saturating_sub(total_bytes));
        for slot in &slots {
            eprintln!("[buf-alloc]   {:?} offset={} size={}", slot.tensor_id, slot.offset, slot.size_bytes);
        }
    }

    // R2: Build virtual source map and skipped set
    let mut virtual_source_map = HashMap::new();
    let mut skipped_virtual = HashSet::new();
    if let Some(vtm) = &vtm {
        for tensor in &graph.tensors {
            if vtm.is_virtual(tensor.id) {
                let root = vtm.physical_root(tensor.id);
                virtual_source_map.insert(tensor.id, root.0 as usize);
                skipped_virtual.insert(tensor.id);
            }
            // §0.2.7: PackMap 虚拟化的权重 tensor — 跳过 pack buffer 分配
            if let Some(pm) = vtm.pack_maps.get(&tensor.id) {
                if pm.requires_physical_pack() {
                    skipped_virtual.insert(tensor.id);
                }
            }
        }
    }

    // R2.5: Build activation slot map — 映射到 ping(0)/pong(1) sentinel slot
    let mut activation_slots = HashMap::new();
    if let Some(vam) = &vam {
        for (tid, a_slot) in &vam.activation_assignments {
            // 每个 activation tensor 映射到 buffer_idx 对应的 sentinel slot
            activation_slots.insert(*tid, a_slot.buffer_idx);
        }
    }

    // R3: Build tensor source classification (§0.2.3 虚拟内存)
    // ISA Lowering codegen 直接消费, 无需独立推导。
    let tensor_sources = build_tensor_sources(graph, &slots, &activation_tids, activation_buffer_size, vam);


    BufferAllocation {
        num_tensors: slots.len(),
        slots,
        total_bytes,
        bytes_saved: naive_total.saturating_sub(total_bytes)
            + skipped_virtual.iter().filter_map(|tid| graph.tensor(*tid).map(|t| t.concrete_bytes())).sum::<usize>()
            + activation_lifetimes.iter().map(|lt| lt.size_bytes).sum::<usize>().saturating_sub(2 * activation_buffer_size),
        virtual_source_map,
        activation_slots,
        skipped_virtual,
        tensor_sources,
    }
}

/// R3: 构建 tensor → TensorPtrSource 分类映射
fn build_tensor_sources(
    graph: &CompilerGraph,
    slots: &[BufferSlot],
    activation_tids: &HashSet<TensorId>,
    activation_buffer_size: usize,
    vam: Option<&VirtualActivationMap>,
) -> HashMap<TensorId, TensorPtrSource> {
    let mut map: HashMap<TensorId, TensorPtrSource> = HashMap::new();

    // Activation + Weight: graph.inputs
    for (i, &tid) in graph.inputs.iter().enumerate() {
        if i == 0 {
            map.insert(tid, TensorPtrSource::Activation);
        } else {
            map.insert(tid, TensorPtrSource::Weight { offset: 0 });
        }
    }
    let wl = graph.weight_layout();
    for &(tid, off) in &wl.offsets {
        map.insert(tid, TensorPtrSource::Weight { offset: off });
    }

    // Outputs: 按 graph.outputs 顺序累加
    // ARCH-DTYPE-JIT-TYPED: elem_bytes 从输出张量 dtype 推断，禁止硬编码 4。
    {
        let mut cursor = 0usize;
        for &tid in &graph.outputs {
            let numel = graph.tensor_numel_for_alloc(tid, graph.max_seq_len).unwrap_or(0);
            let elem_bytes = graph.tensor(tid)
                .map(|t| t.dtype.size_bytes())
                .unwrap_or(DType::F32.size_bytes()); // 安全回退: 无 tensor meta 时默认 F32
            map.insert(tid, TensorPtrSource::Output { offset: cursor });
            cursor += numel * elem_bytes;
        }
    }

    // Intermediate: BufferAllocation slots
    for slot in slots {
        map.entry(slot.tensor_id).or_insert(TensorPtrSource::Intermediate { offset: slot.offset });
    }

    // Activation alias tensors: the output of activation_alias shares the same
    // physical buffer as the activation input (in-place overwrite during layer loop).
    // In mega-kernel path (activation_buffer_size > 0), VAM provides ping-pong slots.
    // In forward-only path (compile_graph), the alias output must map to the same
    // scratchpad offset as the activation input so that post-loop ops (e.g. MeanPool)
    // can read the final layer's output from the correct location.
    if activation_buffer_size > 0 {
        if let Some(vam_ref) = vam {
            // Map activation alias tensors to ActivationPing/ActivationPong
            // based on their VAM buffer_idx assignment (0=ping, 1=pong).
            // This is critical: post-loop ops (MeanPool, etc.) must read from
            // the correct pong buffer, not from scratchpad offset 0.
            if let Some(cfg) = &graph.layer_loop_config {
                if let Some((in_tid, out_tid)) = cfg.activation_alias {
                    map.insert(in_tid, TensorPtrSource::ActivationPing);
                    map.insert(out_tid, TensorPtrSource::ActivationPong);
                }
            }
            if let Some(cfg) = &graph.hetero_layer_loop_config {
                for &(in_tid, out_tid) in &cfg.activation_aliases {
                    map.insert(in_tid, TensorPtrSource::ActivationPing);
                    map.insert(out_tid, TensorPtrSource::ActivationPong);
                }
            }
            // Also map any VAM-assigned tensors not covered by activation_alias
            // (shouldn't happen in practice, but ensures completeness).
            for (&tid, slot) in &vam_ref.activation_assignments {
                if !map.contains_key(&tid) {
                    match slot.buffer_idx {
                        0 => { map.insert(tid, TensorPtrSource::ActivationPing); }
                        1 => { map.insert(tid, TensorPtrSource::ActivationPong); }
                        _ => { map.insert(tid, TensorPtrSource::Intermediate { offset: 0 }); }
                    }
                }
            }
        }
    } else {
        // Forward-only path (compile_graph): alias output inherits activation input's source.
        if let Some(cfg) = &graph.layer_loop_config {
            if let Some((in_tid, out_tid)) = cfg.activation_alias {
                if let Some(src) = map.get(&in_tid).copied() {
                    map.insert(out_tid, src);
                }
            }
        }
        if let Some(cfg) = &graph.hetero_layer_loop_config {
            for &(in_tid, out_tid) in &cfg.activation_aliases {
                if let Some(src) = map.get(&in_tid).copied() {
                    map.insert(out_tid, src);
                }
            }
        }
    }

    // Generic output-alias
    for op in &graph.ops {
        if let Some(input_idx) = op.op_output_aliases_input(graph) {
            if let (Some(&in_tid), Some(&out_tid)) =
                (op.inputs.get(input_idx), op.outputs.first())
            {
                if let Some(&src) = map.get(&in_tid) {
                    map.insert(out_tid, src);
                }
            }
        }
    }

    // §0.2.8 Activation alias — ping-pong 双 buffer
    // input_tid → ActivationPing (当前层读取), output_tid → ActivationPong (当前层写入)
    // 每层末尾 ActivationSwap 交换 ping/pong ptr，实现双 buffer 交替。
    // 仅当 activation_buffer_size > 0 (即 VAM 分析分配了 sentinel slots) 时启用。
    // Per-layer 编译路径 activation_buffer_size=0，保持原始 Activation 映射。
    if activation_buffer_size > 0 {
        if let Some(ref cfg) = graph.layer_loop_config {
            if let Some((ref input_tid, ref output_tid)) = cfg.activation_alias {
                map.insert(*input_tid, TensorPtrSource::ActivationPing);
                map.insert(*output_tid, TensorPtrSource::ActivationPong);
            }
        }
        if let Some(ref cfg) = graph.hetero_layer_loop_config {
            for (input_tid, output_tid) in &cfg.activation_aliases {
                map.insert(*input_tid, TensorPtrSource::ActivationPing);
                map.insert(*output_tid, TensorPtrSource::ActivationPong);
            }
        }
    }

    map
}

/// R1.5 布局感知对齐: 根据 LayoutAssignment 中该 tensor 对应 op 的布局约束,
/// 返回增强的对齐值。PanelPacked/SharedMemTile 需要 tile 边界对齐。
fn layout_align_for_tensor(
    tid: TensorId,
    graph: &CompilerGraph,
    layout: Option<&LayoutAssignment>,
    default_align: usize,
) -> usize {
    use crate::compiler::accel_registry::LayoutConstraint;
    let Some(la) = layout else { return default_align };

    let Some(producer_op_id) = graph.tensor(tid).and_then(|t| t.producer) else {
        return default_align;
    };

    let Some(assign) = la.group_assignments.iter()
        .find_map(|ga| ga.op_layouts.get(&producer_op_id))
    else {
        return default_align;
    };

    match &assign.output_layout {
        LayoutConstraint::PanelPacked { mr, nr } => {
            (mr * nr * 4).max(default_align)
        }
        LayoutConstraint::SharedMemTile { tile_rows, tile_cols, .. } => {
            (tile_rows * tile_cols * 4).max(default_align)
        }
        LayoutConstraint::TmaAligned2D { tile_m: _, tile_n: _ } => {
            128_usize.max(default_align)
        }
        LayoutConstraint::AmxTileBF16 { rows, cols } => {
            (rows * cols * 2).max(default_align)
        }
        _ => default_align,
    }
}

/// Find the lowest aligned offset where a tensor fits without overlapping live allocations.
fn find_offset_aligned(lt: &Lifetime, active: &[(usize, usize, usize)], align: usize) -> usize {
    let mut live_ranges: Vec<(usize, usize)> = active
        .iter()
        .filter(|(_, _, last)| *last >= lt.first_use)
        .map(|(start, end, _)| (*start, *end))
        .collect();

    live_ranges.sort_by_key(|(start, _)| *start);

    let mut candidate = 0usize;
    for &(start, end) in &live_ranges {
        // Align candidate before checking gap
        let aligned = (candidate + align - 1) & !(align - 1);
        if aligned + lt.size_bytes <= start {
            return aligned;
        }
        candidate = candidate.max(end);
    }

    (candidate + align - 1) & !(align - 1)
}

/// Find the lowest offset where a tensor can be placed without overlapping
/// any currently live allocation.
fn find_offset(lt: &Lifetime, active: &[(usize, usize, usize)]) -> usize {
    // Collect all live allocations that overlap with this tensor's lifetime
    let mut live_ranges: Vec<(usize, usize)> = active
        .iter()
        .filter(|(_, _, last)| *last >= lt.first_use)
        .map(|(start, end, _)| (*start, *end))
        .collect();

    live_ranges.sort_by_key(|(start, _)| *start);

    // Find first gap that fits
    let mut candidate = 0usize;
    for &(start, end) in &live_ranges {
        if candidate + lt.size_bytes <= start {
            return candidate; // fits in the gap
        }
        candidate = candidate.max(end);
    }

    // Align to 64 bytes for cache line alignment
    (candidate + 63) & !63
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 28 REQ-GRP-001: TensorLifetime cross-layer extension
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Cross-layer lifetime classification for single mega-kernel planning.
///
/// Determines whether a tensor's buffer can be reused between layer iterations
/// or must persist across multiple (or all) layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossLayerLifetime {
    /// Only alive within a single layer iteration (e.g. FFN intermediate).
    /// Buffer offset can be fully reused by the next layer.
    PerLayer,
    /// Alive across all layer iterations (e.g. activation, KV cache).
    /// Buffer must persist for the entire layer loop.
    SpanningAllLayers,
    /// Alive across N layer iterations (e.g. sliding window KV).
    SpanningNLayers(usize),
    /// Persistent activation: written once before the layer loop, read-only
    /// during each layer iteration (e.g. AltUp per_layer_inputs [L,S,hpl]).
    /// Semantically distinct from SpanningAllLayers: no per-layer write-back,
    /// enabling read-only memory placement optimization.
    PersistentActivation,
}

/// Extended tensor lifetime with cross-layer dimension (SPEC 28).
///
/// Wraps the existing `Lifetime` intra-layer interval with a cross-layer
/// classification for whole-graph resource planning.
#[derive(Debug, Clone)]
pub struct TensorLifetimeExt {
    pub tensor_id: TensorId,
    /// Intra-layer lifetime: [first_step, last_step] within a single layer.
    pub intra_layer: (usize, usize),
    /// Cross-layer classification.
    pub cross_layer: CrossLayerLifetime,
    /// Buffer size in bytes.
    pub size_bytes: usize,
}

impl TensorLifetimeExt {
    /// Convert from a basic `Lifetime` with cross-layer classification derived from the graph.
    pub fn from_lifetime(lt: &Lifetime, graph: &CompilerGraph) -> Self {
        let cross_layer = determine_cross_layer_lifetime(lt.tensor_id, graph);
        Self {
            tensor_id: lt.tensor_id,
            intra_layer: (lt.first_use, lt.last_use),
            cross_layer,
            size_bytes: lt.size_bytes,
        }
    }
}

/// Determine a tensor's cross-layer lifetime classification from the graph.
///
/// Rules:
/// - Activation ping/pong tensors → SpanningAllLayers
/// - KV cache tensors → SpanningAllLayers
/// - Output tensors → SpanningAllLayers
/// - Tensors produced by layer-specific ops (GEMM, Norm, etc.) → PerLayer
/// - Virtual activation aliases → inherit from their physical source
fn determine_cross_layer_lifetime(tid: TensorId, graph: &CompilerGraph) -> CrossLayerLifetime {
    // Activation input tensor — spans all layers (ping-pong)
    if let Some(cfg) = &graph.layer_loop_config {
        if let Some((in_tid, _)) = cfg.activation_alias {
            if tid == in_tid {
                return CrossLayerLifetime::SpanningAllLayers;
            }
        }
    }
    if let Some(cfg) = &graph.hetero_layer_loop_config {
        for (in_tid, _) in &cfg.activation_aliases {
            if tid == *in_tid {
                return CrossLayerLifetime::SpanningAllLayers;
            }
        }
    }

    // Output tensors — produced every layer, consumed by next layer or as final output
    if graph.outputs.contains(&tid) {
        return CrossLayerLifetime::SpanningAllLayers;
    }

    // Check if this is a KV-related tensor (name-based heuristic)
    if let Some(tensor) = graph.tensor(tid) {
        let name = &tensor.name;
        if name.contains("_k_rope") || name.contains("_v_") || name.contains("_attn_out")
            || name.contains("_kv_") || name.contains("_k_cache") || name.contains("_v_cache")
        {
            return CrossLayerLifetime::SpanningAllLayers;
        }
        // AltUp PersistentActivation: per_layer_inputs [L,S,hpl] written once
        // before the layer loop, read-only during each layer iteration.
        if name.contains("per_layer_input") || name.contains("altup.fat")
            || name.contains("ple_combined")
        {
            return CrossLayerLifetime::PersistentActivation;
        }
    }

    // Default: per-layer intermediate
    CrossLayerLifetime::PerLayer
}

/// Extended lifetime analysis with cross-layer classification (SPEC 28 REQ-GRP-001).
///
/// Wraps `analyze_lifetimes` and augments each lifetime with a cross-layer dimension.
/// Used by `GraphResourcePlan` for whole-graph buffer planning.
pub fn analyze_lifetimes_extended(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    vtm: Option<&VirtualTensorMap>,
    vam: Option<&VirtualActivationMap>,
) -> Vec<TensorLifetimeExt> {
    let base = analyze_lifetimes(graph, plan, vtm, vam);
    base.iter().map(|lt| TensorLifetimeExt::from_lifetime(lt, graph)).collect()
}

/// Per-group scratch buffer requirement for TileLevelFusion.
#[derive(Debug, Clone)]
pub struct GroupScratch {
    /// Fusion group ID.
    pub group_id: usize,
    /// Scratch bytes needed (tile_rows × K × sizeof(f32)).
    pub scratch_bytes: usize,
}

/// Compute scratch buffer requirements for TileLevelFusion groups.
///
/// Each TileLevelFusion group needs a scratch buffer to hold the tiled norm
/// output (tile_rows × K × element_size). This is separate from the
/// intermediate tensor allocation because it's a temporary within the
/// microkernel's MC loop, not a full tensor.
pub fn compute_scratch_requirements(
    plan: &FusionPlan,
    graph: &CompilerGraph,
) -> Vec<GroupScratch> {
    plan.groups
        .iter()
        .filter_map(|group| {
            if let FusionMode::TileLevelFusion { tile_rows, .. } = group.mode {
                // Find the GEMM op to get K dimension (胖 opcode 自描述)
                let k = group.ops.iter().find_map(|&oid| {
                    graph.op(oid).and_then(|o| o.op_gemm_dims(graph).map(|(_, _, k)| k))
                }).unwrap_or(0);

                let elem_size = group.ops.iter().find_map(|&oid| {
                    graph.op(oid).and_then(|o| {
                        o.outputs.first().and_then(|&tid| {
                            graph.tensor(tid).map(|t| t.dtype.size_bytes())
                        })
                    })
                }).unwrap_or(DType::F32.size_bytes());

                let scratch_bytes = tile_rows * k * elem_size;
                if scratch_bytes > 0 {
                    Some(GroupScratch {
                        group_id: group.id,
                        scratch_bytes,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, TensorId};
    use crate::compiler::ir::LayerIR;
    use crate::compiler::fusion;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::planner::ExecutionPlan;
    use crate::dispatch::DeviceProfile;
    use crate::types::ModelConfig;

    #[test]
    fn test_non_overlapping_reuse() {
        // Two tensors with non-overlapping lifetimes should share buffer space
        let lifetimes = vec![
            Lifetime {
                tensor_id: TensorId(0),
                first_use: 0,
                last_use: 1,
                size_bytes: 1024,
            },
            Lifetime {
                tensor_id: TensorId(1),
                first_use: 2,
                last_use: 3,
                size_bytes: 1024,
            },
        ];

        let alloc = allocate_buffers(&lifetimes);

        assert_eq!(alloc.num_tensors, 2);
        // Non-overlapping: second tensor can reuse the first's space
        // Total should be <= 1024 + alignment overhead, not 2048
        assert!(
            alloc.total_bytes < 2048,
            "Non-overlapping tensors should share buffer, total={} (expected < 2048)",
            alloc.total_bytes
        );
        assert!(alloc.bytes_saved > 0, "Should save bytes via reuse");
    }

    #[test]
    fn test_overlapping_no_reuse() {
        // Two tensors with overlapping lifetimes cannot share buffer space
        let lifetimes = vec![
            Lifetime {
                tensor_id: TensorId(0),
                first_use: 0,
                last_use: 3,
                size_bytes: 1024,
            },
            Lifetime {
                tensor_id: TensorId(1),
                first_use: 1,
                last_use: 2,
                size_bytes: 1024,
            },
        ];

        let alloc = allocate_buffers(&lifetimes);

        assert_eq!(alloc.num_tensors, 2);
        // Overlapping: both must be live simultaneously
        assert!(
            alloc.total_bytes >= 2048,
            "Overlapping tensors cannot share buffer, total={} (expected >= 2048)",
            alloc.total_bytes
        );
    }

    #[test]
    fn test_llama_buffer_allocation() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fusion::fuse_with_dag(&graph, &registry, &exec_plan);

        let lifetimes = analyze_lifetimes(&graph, &plan, None, None);
        assert!(!lifetimes.is_empty(), "LLaMA graph should have intermediate tensors");

        let alloc = allocate_buffers(&lifetimes);

        assert!(alloc.num_tensors > 0);
        assert!(alloc.total_bytes > 0);
        assert!(
            alloc.bytes_saved > 0,
            "LLaMA graph should benefit from buffer reuse"
        );

        let naive: usize = lifetimes.iter().map(|l| l.size_bytes).sum();
        eprintln!(
            "LLaMA buffer alloc: {} tensors, naive={} bytes, optimized={} bytes, saved={} bytes ({:.0}%)",
            alloc.num_tensors,
            naive,
            alloc.total_bytes,
            alloc.bytes_saved,
            alloc.bytes_saved as f64 / naive as f64 * 100.0,
        );
    }

    #[test]
    fn test_empty_lifetimes() {
        let alloc = allocate_buffers(&[]);
        assert_eq!(alloc.total_bytes, 0);
        assert_eq!(alloc.num_tensors, 0);
        assert_eq!(alloc.bytes_saved, 0);
        assert!(alloc.slots.is_empty());
    }

    #[test]
    fn cacheline_128_alignment() {
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 2, size_bytes: 100 },
            Lifetime { tensor_id: TensorId(1), first_use: 1, last_use: 3, size_bytes: 200 },
        ];
        let alloc = allocate_buffers_aligned(&lifetimes, 128, None, None, &CompilerGraph::new(), None);
        for slot in &alloc.slots {
            assert_eq!(slot.offset % 128, 0, "offset {} not aligned to 128", slot.offset);
        }
    }

    #[test]
    fn default_64_alignment() {
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 2, size_bytes: 100 },
        ];
        let alloc = allocate_buffers_aligned(&lifetimes, 64, None, None, &CompilerGraph::new(), None);
        assert_eq!(alloc.slots[0].offset % 64, 0);
    }

    #[test]
    fn aligned_empty() {
        let alloc = allocate_buffers_aligned(&[], 128, None, None, &CompilerGraph::new(), None);
        assert_eq!(alloc.total_bytes, 0);
        assert_eq!(alloc.num_tensors, 0);
    }

    #[test]
    fn aligned_non_overlapping_reuse() {
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 1, size_bytes: 1024 },
            Lifetime { tensor_id: TensorId(1), first_use: 2, last_use: 3, size_bytes: 1024 },
        ];
        let alloc = allocate_buffers_aligned(&lifetimes, 64, None, None, &CompilerGraph::new(), None);
        // Non-overlapping: second tensor can reuse first's space
        assert!(
            alloc.total_bytes < 2048 + 64,
            "Non-overlapping should share buffer, total={}",
            alloc.total_bytes
        );
    }

    // ── Lifetime struct tests ──────────────────────────────────────

    #[test]
    fn lifetime_construction_and_field_access() {
        let lt = Lifetime {
            tensor_id: TensorId(42),
            first_use: 5,
            last_use: 10,
            size_bytes: 2048,
        };
        assert_eq!(lt.tensor_id, TensorId(42));
        assert_eq!(lt.first_use, 5);
        assert_eq!(lt.last_use, 10);
        assert_eq!(lt.size_bytes, 2048);
    }

    #[test]
    fn lifetime_clone_and_copy() {
        let lt = Lifetime {
            tensor_id: TensorId(7),
            first_use: 0,
            last_use: 3,
            size_bytes: 512,
        };
        // Copy semantic (Copy trait)
        let lt_copy = lt;
        assert_eq!(lt_copy.tensor_id, TensorId(7));
        assert_eq!(lt_copy.size_bytes, 512);
        // Original still accessible after copy
        assert_eq!(lt.tensor_id, TensorId(7));
    }

    #[test]
    fn lifetime_debug_format() {
        let lt = Lifetime {
            tensor_id: TensorId(1),
            first_use: 2,
            last_use: 3,
            size_bytes: 64,
        };
        let debug_str = format!("{:?}", lt);
        assert!(debug_str.contains("Lifetime"));
        assert!(debug_str.contains("tensor_id"));
        assert!(debug_str.contains("first_use"));
        assert!(debug_str.contains("last_use"));
        assert!(debug_str.contains("size_bytes"));
    }

    // ── BufferSlot struct tests ────────────────────────────────────

    #[test]
    fn buffer_slot_construction_and_field_access() {
        let slot = BufferSlot {
            tensor_id: TensorId(99),
            offset: 256,
            size_bytes: 128,
        };
        assert_eq!(slot.tensor_id, TensorId(99));
        assert_eq!(slot.offset, 256);
        assert_eq!(slot.size_bytes, 128);
    }

    #[test]
    fn buffer_slot_clone_and_copy() {
        let slot = BufferSlot {
            tensor_id: TensorId(3),
            offset: 0,
            size_bytes: 1024,
        };
        let cloned = slot.clone();
        assert_eq!(cloned.tensor_id, slot.tensor_id);
        assert_eq!(cloned.offset, slot.offset);
        assert_eq!(cloned.size_bytes, slot.size_bytes);
        // Copy also works
        let copied = slot;
        assert_eq!(copied.tensor_id, slot.tensor_id);
    }

    #[test]
    fn buffer_slot_debug_format() {
        let slot = BufferSlot {
            tensor_id: TensorId(5),
            offset: 1024,
            size_bytes: 512,
        };
        let debug_str = format!("{:?}", slot);
        assert!(debug_str.contains("BufferSlot"));
        assert!(debug_str.contains("tensor_id"));
        assert!(debug_str.contains("offset"));
    }

    // ── BufferAllocation struct tests ──────────────────────────────

    #[test]
    fn buffer_allocation_default_is_empty() {
        let alloc = BufferAllocation::default();
        assert!(alloc.slots.is_empty());
        assert_eq!(alloc.total_bytes, 0);
        assert_eq!(alloc.num_tensors, 0);
        assert_eq!(alloc.bytes_saved, 0);
        assert!(alloc.virtual_source_map.is_empty());
        assert!(alloc.activation_slots.is_empty());
        assert!(alloc.skipped_virtual.is_empty());
        assert!(alloc.tensor_sources.is_empty());
    }

    #[test]
    fn buffer_allocation_empty_returns_default() {
        let alloc = BufferAllocation::empty();
        assert_eq!(alloc.total_bytes, 0);
        assert_eq!(alloc.num_tensors, 0);
        assert!(alloc.slots.is_empty());
    }

    #[test]
    fn buffer_allocation_offset_of_finds_tensor() {
        let alloc = BufferAllocation {
            slots: vec![
                BufferSlot { tensor_id: TensorId(0), offset: 0, size_bytes: 100 },
                BufferSlot { tensor_id: TensorId(1), offset: 128, size_bytes: 200 },
                BufferSlot { tensor_id: TensorId(2), offset: 384, size_bytes: 64 },
            ],
            total_bytes: 448,
            num_tensors: 3,
            bytes_saved: 0,
            virtual_source_map: HashMap::new(),
            activation_slots: HashMap::new(),
            skipped_virtual: HashSet::new(),
            tensor_sources: HashMap::new(),
        };
        assert_eq!(alloc.offset_of(TensorId(0)), Some(0));
        assert_eq!(alloc.offset_of(TensorId(1)), Some(128));
        assert_eq!(alloc.offset_of(TensorId(2)), Some(384));
    }

    #[test]
    fn buffer_allocation_offset_of_missing_returns_none() {
        let alloc = BufferAllocation {
            slots: vec![
                BufferSlot { tensor_id: TensorId(0), offset: 0, size_bytes: 100 },
            ],
            total_bytes: 100,
            num_tensors: 1,
            bytes_saved: 0,
            virtual_source_map: HashMap::new(),
            activation_slots: HashMap::new(),
            skipped_virtual: HashSet::new(),
            tensor_sources: HashMap::new(),
        };
        assert_eq!(alloc.offset_of(TensorId(999)), None);
    }

    #[test]
    fn buffer_allocation_clone() {
        let alloc = BufferAllocation {
            slots: vec![BufferSlot { tensor_id: TensorId(0), offset: 0, size_bytes: 64 }],
            total_bytes: 64,
            num_tensors: 1,
            bytes_saved: 0,
            virtual_source_map: HashMap::new(),
            activation_slots: HashMap::new(),
            skipped_virtual: HashSet::new(),
            tensor_sources: HashMap::new(),
        };
        let cloned = alloc.clone();
        assert_eq!(cloned.num_tensors, 1);
        assert_eq!(cloned.total_bytes, 64);
        assert_eq!(cloned.slots.len(), 1);
        assert_eq!(cloned.slots[0].tensor_id, TensorId(0));
    }

    // ── TensorPtrSource enum tests ─────────────────────────────────

    #[test]
    fn tensor_ptr_source_variants_debug_and_copy() {
        let activation = TensorPtrSource::Activation;
        let ping = TensorPtrSource::ActivationPing;
        let pong = TensorPtrSource::ActivationPong;
        let weight = TensorPtrSource::Weight { offset: 4096 };
        let intermediate = TensorPtrSource::Intermediate { offset: 8192 };
        let output = TensorPtrSource::Output { offset: 16384 };

        // Debug format
        assert!(format!("{:?}", activation).contains("Activation"));
        assert!(format!("{:?}", ping).contains("ActivationPing"));
        assert!(format!("{:?}", pong).contains("ActivationPong"));
        assert!(format!("{:?}", weight).contains("Weight"));
        assert!(format!("{:?}", intermediate).contains("Intermediate"));
        assert!(format!("{:?}", output).contains("Output"));

        // Clone
        let weight_cloned = weight.clone();
        assert!(matches!(weight_cloned, TensorPtrSource::Weight { offset: 4096 }));

        // Copy
        let output_copied = output;
        assert!(matches!(output_copied, TensorPtrSource::Output { offset: 16384 }));
    }

    #[test]
    fn tensor_ptr_source_weight_offset_access() {
        let src = TensorPtrSource::Weight { offset: 12345 };
        if let TensorPtrSource::Weight { offset } = src {
            assert_eq!(offset, 12345);
        } else {
            panic!("Expected Weight variant");
        }
    }

    #[test]
    fn tensor_ptr_source_intermediate_offset_access() {
        let src = TensorPtrSource::Intermediate { offset: 67890 };
        if let TensorPtrSource::Intermediate { offset } = src {
            assert_eq!(offset, 67890);
        } else {
            panic!("Expected Intermediate variant");
        }
    }

    // ── CrossLayerLifetime enum tests ──────────────────────────────

    #[test]
    fn cross_layer_lifetime_variants_and_equality() {
        let per_layer = CrossLayerLifetime::PerLayer;
        let spanning = CrossLayerLifetime::SpanningAllLayers;
        let n_layers = CrossLayerLifetime::SpanningNLayers(4);

        assert_eq!(per_layer, CrossLayerLifetime::PerLayer);
        assert_ne!(per_layer, spanning);
        assert_eq!(n_layers, CrossLayerLifetime::SpanningNLayers(4));
        assert_ne!(n_layers, CrossLayerLifetime::SpanningNLayers(2));
    }

    #[test]
    fn cross_layer_lifetime_debug_format() {
        assert!(format!("{:?}", CrossLayerLifetime::PerLayer).contains("PerLayer"));
        assert!(format!("{:?}", CrossLayerLifetime::SpanningAllLayers).contains("SpanningAllLayers"));
        let n = format!("{:?}", CrossLayerLifetime::SpanningNLayers(3));
        assert!(n.contains("SpanningNLayers"));
    }

    #[test]
    fn cross_layer_lifetime_clone_and_copy() {
        let orig = CrossLayerLifetime::SpanningNLayers(7);
        let cloned = orig.clone();
        assert_eq!(cloned, CrossLayerLifetime::SpanningNLayers(7));
        let copied = orig;
        assert_eq!(copied, CrossLayerLifetime::SpanningNLayers(7));
    }

    // ── TensorLifetimeExt struct tests ─────────────────────────────

    #[test]
    fn tensor_lifetime_ext_construction_and_field_access() {
        let ext = TensorLifetimeExt {
            tensor_id: TensorId(10),
            intra_layer: (1, 5),
            cross_layer: CrossLayerLifetime::PerLayer,
            size_bytes: 4096,
        };
        assert_eq!(ext.tensor_id, TensorId(10));
        assert_eq!(ext.intra_layer, (1, 5));
        assert_eq!(ext.cross_layer, CrossLayerLifetime::PerLayer);
        assert_eq!(ext.size_bytes, 4096);
    }

    #[test]
    fn tensor_lifetime_ext_from_lifetime_per_layer() {
        let lt = Lifetime {
            tensor_id: TensorId(3),
            first_use: 2,
            last_use: 7,
            size_bytes: 2048,
        };
        let graph = CompilerGraph::new();
        let ext = TensorLifetimeExt::from_lifetime(&lt, &graph);
        assert_eq!(ext.tensor_id, TensorId(3));
        assert_eq!(ext.intra_layer, (2, 7));
        assert_eq!(ext.size_bytes, 2048);
        // No special graph config, so default is PerLayer
        assert_eq!(ext.cross_layer, CrossLayerLifetime::PerLayer);
    }

    #[test]
    fn tensor_lifetime_ext_debug_format() {
        let ext = TensorLifetimeExt {
            tensor_id: TensorId(1),
            intra_layer: (0, 3),
            cross_layer: CrossLayerLifetime::SpanningAllLayers,
            size_bytes: 1024,
        };
        let debug_str = format!("{:?}", ext);
        assert!(debug_str.contains("TensorLifetimeExt"));
        assert!(debug_str.contains("tensor_id"));
        assert!(debug_str.contains("intra_layer"));
        assert!(debug_str.contains("cross_layer"));
        assert!(debug_str.contains("size_bytes"));
    }

    // ── GroupScratch struct tests ──────────────────────────────────

    #[test]
    fn group_scratch_construction_and_field_access() {
        let gs = GroupScratch {
            group_id: 5,
            scratch_bytes: 8192,
        };
        assert_eq!(gs.group_id, 5);
        assert_eq!(gs.scratch_bytes, 8192);
    }

    #[test]
    fn group_scratch_clone() {
        let gs = GroupScratch {
            group_id: 2,
            scratch_bytes: 4096,
        };
        let cloned = gs.clone();
        assert_eq!(cloned.group_id, 2);
        assert_eq!(cloned.scratch_bytes, 4096);
    }

    #[test]
    fn group_scratch_debug_format() {
        let gs = GroupScratch {
            group_id: 1,
            scratch_bytes: 256,
        };
        let debug_str = format!("{:?}", gs);
        assert!(debug_str.contains("GroupScratch"));
        assert!(debug_str.contains("group_id"));
        assert!(debug_str.contains("scratch_bytes"));
    }

    // ── find_offset / find_offset_aligned pure function tests ──────

    #[test]
    fn find_offset_finds_gap_in_live_ranges() {
        // A live range [0, 100) that ended before our tensor starts
        // Active: (offset=0, end=100, last_use=0)
        // Tensor starts at step 1, so active range is not live
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 5, last_use: 10, size_bytes: 50 };
        let active = vec![(0usize, 100usize, 0usize)]; // last_use=0 < first_use=5, not live
        let offset = find_offset(&lt, &active);
        assert_eq!(offset, 0); // No live ranges overlap, should place at 0
    }

    #[test]
    fn find_offset_stacks_after_live_range() {
        // One live range covering our first_use
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 50 };
        let active = vec![(0usize, 100usize, 10usize)]; // last_use=10 >= first_use=0
        let offset = find_offset(&lt, &active);
        // Should be placed after the live range, aligned to 64
        assert!(offset >= 100);
        assert_eq!(offset % 64, 0, "should be 64-byte aligned");
    }

    #[test]
    fn find_offset_aligned_with_small_align() {
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 5, last_use: 10, size_bytes: 50 };
        // No live ranges
        let offset = find_offset_aligned(&lt, &[], 32);
        assert_eq!(offset, 0);
    }

    #[test]
    fn find_offset_aligned_fits_in_gap() {
        // Live range [0, 200), our tensor starts at step 5 which is after last_use=3
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 5, last_use: 10, size_bytes: 50 };
        let active = vec![(0usize, 200usize, 3usize)];
        let offset = find_offset_aligned(&lt, &active, 64);
        assert_eq!(offset, 0);
    }

    #[test]
    fn find_offset_aligned_stacks_after() {
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 80 };
        let active = vec![(0usize, 100usize, 10usize)];
        let offset = find_offset_aligned(&lt, &active, 128);
        assert!(offset >= 100);
        assert_eq!(offset % 128, 0, "should be 128-byte aligned");
    }

    // ── allocate_buffers edge cases ────────────────────────────────

    #[test]
    fn allocate_buffers_single_tensor() {
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 256 },
        ];
        let alloc = allocate_buffers(&lifetimes);
        assert_eq!(alloc.num_tensors, 1);
        assert_eq!(alloc.slots.len(), 1);
        assert_eq!(alloc.slots[0].tensor_id, TensorId(0));
        assert_eq!(alloc.slots[0].offset, 0);
        assert_eq!(alloc.slots[0].size_bytes, 256);
        assert_eq!(alloc.bytes_saved, 0);
    }

    #[test]
    fn allocate_buffers_three_tensors_partial_overlap() {
        // A: [0,5] 1024 bytes — overlaps B but not C
        // B: [2,4] 512 bytes  — overlaps A
        // C: [6,8] 1024 bytes — no overlap with A or B
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 1024 },
            Lifetime { tensor_id: TensorId(1), first_use: 2, last_use: 4, size_bytes: 512 },
            Lifetime { tensor_id: TensorId(2), first_use: 6, last_use: 8, size_bytes: 1024 },
        ];
        let alloc = allocate_buffers(&lifetimes);
        assert_eq!(alloc.num_tensors, 3);
        // C can reuse either A or B's space (they're both dead by step 6)
        let naive: usize = lifetimes.iter().map(|l| l.size_bytes).sum();
        assert!(alloc.total_bytes < naive, "Partial overlap should enable reuse");
        assert!(alloc.bytes_saved > 0);
    }

    #[test]
    fn allocate_buffers_sorted_by_first_use_then_size() {
        // Two tensors with same first_use — larger should come first for better packing
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 2, size_bytes: 100 },
            Lifetime { tensor_id: TensorId(1), first_use: 0, last_use: 3, size_bytes: 500 },
        ];
        let alloc = allocate_buffers(&lifetimes);
        assert_eq!(alloc.num_tensors, 2);
        // Both overlap at step 0, so both must be live simultaneously
        assert!(alloc.total_bytes >= 600, "Both tensors alive at step 0");
    }

    #[test]
    fn allocate_buffers_bytes_saved_calculation() {
        // Non-overlapping: naive=2048, actual should be close to 1024 (reuse)
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 1, size_bytes: 1024 },
            Lifetime { tensor_id: TensorId(1), first_use: 2, last_use: 3, size_bytes: 1024 },
        ];
        let alloc = allocate_buffers(&lifetimes);
        let naive: usize = 2048;
        assert_eq!(alloc.bytes_saved, naive.saturating_sub(alloc.total_bytes));
    }

    #[test]
    fn allocate_buffers_aligned_minimum_alignment_is_64() {
        // Pass alignment=32, should be clamped to 64 minimum
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 2, size_bytes: 100 },
        ];
        let alloc = allocate_buffers_aligned(&lifetimes, 32, None, None, &CompilerGraph::new(), None);
        assert_eq!(alloc.slots[0].offset, 0);
        // Total should be rounded up to at least 64
        assert!(alloc.total_bytes >= 64);
    }

    #[test]
    fn allocate_buffers_aligned_total_is_aligned() {
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 100 },
            Lifetime { tensor_id: TensorId(1), first_use: 1, last_use: 4, size_bytes: 200 },
        ];
        let alloc = allocate_buffers_aligned(&lifetimes, 128, None, None, &CompilerGraph::new(), None);
        assert_eq!(alloc.total_bytes % 128, 0, "total_bytes should be 128-aligned");
    }

    // ── compute_scratch_requirements tests ─────────────────────────

    #[test]
    fn compute_scratch_requirements_empty_plan() {
        let plan = FusionPlan { groups: vec![], op_to_group: HashMap::new() };
        let graph = CompilerGraph::new();
        let result = compute_scratch_requirements(&plan, &graph);
        assert!(result.is_empty());
    }

    // ── Additional tests (wave-12kna) ─────────────────────────────────

    #[test]
    fn find_offset_multiple_gaps_picks_first_fit() {
        // Arrange: two live ranges with a gap between them, plus a gap before the first.
        // Active: (0, 50, 10), (200, 300, 10) — both live through step 5.
        // Tensor size = 80 bytes. Gap [50, 200) is 150 bytes wide -> fits.
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 5, last_use: 8, size_bytes: 80 };
        let active = vec![
            (0usize, 50usize, 10usize),
            (200usize, 300usize, 10usize),
        ];
        // Act
        let offset = find_offset(&lt, &active);
        // Assert: should land in the first gap after the first live range.
        // find_offset does not align within gap scanning, so offset = 50.
        assert!(offset >= 50);
        assert!(offset + 80 <= 200, "must fit in gap [50, 200)");
    }

    #[test]
    fn find_offset_no_gap_stacks_at_end() {
        // Arrange: three fully-overlapping live ranges with no usable gap.
        // total live span = [0, 300), all still live at first_use=1.
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 1, last_use: 5, size_bytes: 64 };
        let active = vec![
            (0usize, 100usize, 10usize),
            (100usize, 200usize, 10usize),
            (200usize, 300usize, 10usize),
        ];
        // Act
        let offset = find_offset(&lt, &active);
        // Assert: no gap fits, so offset = align_up(300) = 320 (64-byte aligned).
        assert!(offset >= 300);
        assert_eq!(offset % 64, 0, "offset must be 64-byte aligned");
    }

    #[test]
    fn find_offset_aligned_alignment_pushes_past_gap() {
        // Arrange: live range [0, 50) with last_use=10; our tensor first_use=1 (overlaps).
        // Align=128, so candidate 0 is aligned, but [0, 50) is live. Next candidate
        // after 50 aligned to 128 = 128.
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 1, last_use: 5, size_bytes: 40 };
        let active = vec![(0usize, 50usize, 10usize)];
        // Act
        let offset = find_offset_aligned(&lt, &active, 128);
        // Assert
        assert_eq!(offset, 128);
        assert_eq!(offset % 128, 0);
    }

    #[test]
    fn allocate_buffers_identical_lifetimes_all_stacked() {
        // Arrange: three tensors all alive during [0, 5] — none can share space.
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 256 },
            Lifetime { tensor_id: TensorId(1), first_use: 0, last_use: 5, size_bytes: 256 },
            Lifetime { tensor_id: TensorId(2), first_use: 0, last_use: 5, size_bytes: 256 },
        ];
        // Act
        let alloc = allocate_buffers(&lifetimes);
        // Assert: all three must be live simultaneously, no reuse possible.
        assert_eq!(alloc.num_tensors, 3);
        assert!(
            alloc.total_bytes >= 768,
            "identical lifetimes need independent slots, total={}",
            alloc.total_bytes
        );
        // Verify each slot has a distinct, non-overlapping offset range.
        let mut ranges: Vec<(usize, usize)> = alloc.slots.iter()
            .map(|s| (s.offset, s.offset + s.size_bytes))
            .collect();
        ranges.sort_by_key(|r| r.0);
        for window in ranges.windows(2) {
            assert!(
                window[0].1 <= window[1].0,
                "slots must not overlap: {:?}",
                window
            );
        }
    }

    #[test]
    fn allocate_buffers_zero_size_tensor() {
        // Arrange: a tensor with size_bytes=0 should not contribute to total.
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 0 },
        ];
        // Act
        let alloc = allocate_buffers(&lifetimes);
        // Assert: slot exists but total_bytes is effectively zero (plus alignment rounding).
        assert_eq!(alloc.num_tensors, 1);
        assert_eq!(alloc.slots[0].offset, 0);
        assert_eq!(alloc.slots[0].size_bytes, 0);
    }

    #[test]
    fn buffer_allocation_offset_of_returns_first_match() {
        // Arrange: two slots with the same TensorId — offset_of should return the first.
        let alloc = BufferAllocation {
            slots: vec![
                BufferSlot { tensor_id: TensorId(5), offset: 100, size_bytes: 50 },
                BufferSlot { tensor_id: TensorId(5), offset: 200, size_bytes: 50 },
            ],
            total_bytes: 250,
            num_tensors: 2,
            bytes_saved: 0,
            virtual_source_map: HashMap::new(),
            activation_slots: HashMap::new(),
            skipped_virtual: HashSet::new(),
            tensor_sources: HashMap::new(),
        };
        // Act
        let result = alloc.offset_of(TensorId(5));
        // Assert
        assert_eq!(result, Some(100), "offset_of should return the first matching slot");
    }

    #[test]
    fn tensor_ptr_source_output_offset_access() {
        // Arrange
        let src = TensorPtrSource::Output { offset: 99999 };
        // Act & Assert
        if let TensorPtrSource::Output { offset } = src {
            assert_eq!(offset, 99999);
        } else {
            panic!("Expected Output variant");
        }
    }

    #[test]
    fn tensor_ptr_source_activation_ping_pong_variants() {
        // Arrange: construct both ping and pong variants.
        let ping = TensorPtrSource::ActivationPing;
        let pong = TensorPtrSource::ActivationPong;
        // Act & Assert: Debug output should contain variant name.
        assert!(format!("{:?}", ping).contains("ActivationPing"));
        assert!(format!("{:?}", pong).contains("ActivationPong"));
        // Copy semantics: both are Copy, assign and verify.
        let ping2 = ping;
        let pong2 = pong;
        assert!(matches!(ping2, TensorPtrSource::ActivationPing));
        assert!(matches!(pong2, TensorPtrSource::ActivationPong));
    }

    #[test]
    fn cross_layer_lifetime_spanning_n_layers_boundary_values() {
        // Arrange: boundary values N=0 and N=1.
        let zero = CrossLayerLifetime::SpanningNLayers(0);
        let one = CrossLayerLifetime::SpanningNLayers(1);
        // Act & Assert: equality checks
        assert_eq!(zero, CrossLayerLifetime::SpanningNLayers(0));
        assert_eq!(one, CrossLayerLifetime::SpanningNLayers(1));
        assert_ne!(zero, one);
        // Debug format contains the inner value
        let dbg_zero = format!("{:?}", zero);
        assert!(dbg_zero.contains("SpanningNLayers"));
    }

    #[test]
    fn tensor_lifetime_ext_from_lifetime_with_graph_output() {
        // Arrange: a graph with an output tensor; lifetime for that tensor.
        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor("output_tensor", vec![], crate::types::DType::F32);
        graph.outputs.push(tid);
        let lt = Lifetime {
            tensor_id: tid,
            first_use: 0,
            last_use: 3,
            size_bytes: 512,
        };
        // Act
        let ext = TensorLifetimeExt::from_lifetime(&lt, &graph);
        // Assert: output tensors should be SpanningAllLayers.
        assert_eq!(ext.tensor_id, tid);
        assert_eq!(ext.intra_layer, (0, 3));
        assert_eq!(ext.size_bytes, 512);
        assert_eq!(ext.cross_layer, CrossLayerLifetime::SpanningAllLayers);
    }

    // ── Additional tests (wave-12x61) ────────────────────────────────────

    #[test]
    fn cross_layer_lifetime_ordering_per_layer_not_equal_to_spanning() {
        // Arrange: all four variants.
        let per_layer = CrossLayerLifetime::PerLayer;
        let spanning_all = CrossLayerLifetime::SpanningAllLayers;
        let spanning_n = CrossLayerLifetime::SpanningNLayers(2);
        let persistent = CrossLayerLifetime::PersistentActivation;
        // Act & Assert: PerLayer is distinct from both Spanning variants.
        assert_ne!(per_layer, spanning_all);
        assert_ne!(per_layer, spanning_n);
        assert_ne!(per_layer, persistent);
        // SpanningAllLayers is distinct from SpanningNLayers(2) and PersistentActivation.
        assert_ne!(spanning_all, spanning_n);
        assert_ne!(spanning_all, persistent);
        // Same variant with same inner value is equal.
        assert_eq!(spanning_n, CrossLayerLifetime::SpanningNLayers(2));
        assert_eq!(persistent, CrossLayerLifetime::PersistentActivation);
    }

    #[test]
    fn tensor_lifetime_ext_intra_layer_preserves_lifetime_bounds() {
        // Arrange: Lifetime with non-zero first_use and last_use.
        let lt = Lifetime {
            tensor_id: TensorId(42),
            first_use: 3,
            last_use: 9,
            size_bytes: 4096,
        };
        let graph = CompilerGraph::new();
        // Act
        let ext = TensorLifetimeExt::from_lifetime(&lt, &graph);
        // Assert: intra_layer tuple exactly mirrors first_use and last_use.
        assert_eq!(ext.intra_layer.0, 3);
        assert_eq!(ext.intra_layer.1, 9);
        assert_eq!(ext.size_bytes, 4096);
        assert_eq!(ext.tensor_id, TensorId(42));
    }

    #[test]
    fn buffer_slot_offset_range_covers_exactly_size_bytes() {
        // Arrange: a slot at offset 512 with size 256.
        let slot = BufferSlot {
            tensor_id: TensorId(10),
            offset: 512,
            size_bytes: 256,
        };
        // Act & Assert: the slot occupies [512, 768).
        assert_eq!(slot.offset, 512);
        assert_eq!(slot.offset + slot.size_bytes, 768);
    }

    #[test]
    fn allocate_buffers_large_alignment_clamps_to_minimum_64() {
        // Arrange: pass cacheline_bytes=1, should be clamped to 64.
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 2, size_bytes: 200 },
        ];
        // Act
        let alloc = allocate_buffers_aligned(&lifetimes, 1, None, None, &CompilerGraph::new(), None);
        // Assert: offset is 0 (trivially aligned), total is rounded up to at least 64.
        assert_eq!(alloc.slots[0].offset, 0);
        assert!(alloc.total_bytes >= 64, "total_bytes should be at least 64 after rounding, got {}", alloc.total_bytes);
    }

    #[test]
    fn allocate_buffers_many_non_overlapping_reuse_single_slot() {
        // Arrange: four sequential tensors, none overlapping, all 1024 bytes.
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 1, size_bytes: 1024 },
            Lifetime { tensor_id: TensorId(1), first_use: 2, last_use: 3, size_bytes: 1024 },
            Lifetime { tensor_id: TensorId(2), first_use: 4, last_use: 5, size_bytes: 1024 },
            Lifetime { tensor_id: TensorId(3), first_use: 6, last_use: 7, size_bytes: 1024 },
        ];
        // Act
        let alloc = allocate_buffers(&lifetimes);
        // Assert: all four can reuse the same slot, so total is approximately 1024 + alignment.
        assert_eq!(alloc.num_tensors, 4);
        assert!(
            alloc.total_bytes < 2048,
            "four non-overlapping 1024-byte tensors should share one slot, total={}",
            alloc.total_bytes
        );
    }

    #[test]
    fn allocate_buffers_overlapping_chain_stacks_incrementally() {
        // Arrange: A=[0,2], B=[1,3], C=[2,4] — each overlaps its neighbor.
        // A and C do NOT overlap (A.last=2, C.first=2 — step 2 is shared but
        // allocate_buffers treats last_use >= first_use as overlap).
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 2, size_bytes: 512 },
            Lifetime { tensor_id: TensorId(1), first_use: 1, last_use: 3, size_bytes: 512 },
            Lifetime { tensor_id: TensorId(2), first_use: 2, last_use: 4, size_bytes: 512 },
        ];
        // Act
        let alloc = allocate_buffers(&lifetimes);
        // Assert: all three must be live at some point with A+B overlapping and B+C overlapping.
        assert_eq!(alloc.num_tensors, 3);
        // Verify no two overlapping slots share the same offset range.
        for i in 0..alloc.slots.len() {
            for j in (i + 1)..alloc.slots.len() {
                let a = &alloc.slots[i];
                let b = &alloc.slots[j];
                let a_lt = lifetimes.iter().find(|l| l.tensor_id == a.tensor_id).unwrap();
                let b_lt = lifetimes.iter().find(|l| l.tensor_id == b.tensor_id).unwrap();
                let overlaps = a_lt.last_use >= b_lt.first_use && b_lt.last_use >= a_lt.first_use;
                if overlaps {
                    let a_range = (a.offset, a.offset + a.size_bytes);
                    let b_range = (b.offset, b.offset + b.size_bytes);
                    let ranges_overlap = a_range.0 < b_range.1 && b_range.0 < a_range.1;
                    assert!(!ranges_overlap,
                        "overlapping lifetimes must not share buffer: {:?} vs {:?}", a_range, b_range);
                }
            }
        }
    }

    #[test]
    fn find_offset_aligned_empty_active_returns_zero() {
        // Arrange: no live ranges, any alignment.
        let lt = Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 5, size_bytes: 1000 };
        // Act
        let offset = find_offset_aligned(&lt, &[], 256);
        // Assert: with no active ranges, offset is 0 (aligned to any value).
        assert_eq!(offset, 0);
    }

    #[test]
    fn group_scratch_zero_scratch_bytes_is_valid() {
        // Arrange: a GroupScratch with scratch_bytes=0 (e.g., tile_rows=0 or k=0).
        let gs = GroupScratch {
            group_id: 0,
            scratch_bytes: 0,
        };
        // Act & Assert: zero scratch is valid, fields accessible.
        assert_eq!(gs.group_id, 0);
        assert_eq!(gs.scratch_bytes, 0);
        // Debug format still works.
        let debug_str = format!("{:?}", gs);
        assert!(debug_str.contains("GroupScratch"));
    }

    #[test]
    fn tensor_ptr_source_intermediate_and_weight_have_different_offsets() {
        // Arrange: two sources with distinct offsets.
        let weight = TensorPtrSource::Weight { offset: 0 };
        let intermediate = TensorPtrSource::Intermediate { offset: 4096 };
        // Act & Assert: variant discriminators differ even if offset values coincided.
        assert!(matches!(weight, TensorPtrSource::Weight { .. }));
        assert!(matches!(intermediate, TensorPtrSource::Intermediate { .. }));
        // Extract and compare offsets.
        if let TensorPtrSource::Weight { offset: w } = weight {
            if let TensorPtrSource::Intermediate { offset: i } = intermediate {
                assert_ne!(w, i, "weight and intermediate offsets should differ in this test");
            }
        }
    }

    #[test]
    fn allocate_buffers_aligned_two_non_overlapping_reuse_with_larger_align() {
        // Arrange: two non-overlapping tensors with 256-byte alignment.
        let lifetimes = vec![
            Lifetime { tensor_id: TensorId(0), first_use: 0, last_use: 1, size_bytes: 100 },
            Lifetime { tensor_id: TensorId(1), first_use: 2, last_use: 3, size_bytes: 100 },
        ];
        // Act
        let alloc = allocate_buffers_aligned(&lifetimes, 256, None, None, &CompilerGraph::new(), None);
        // Assert: both offsets are 256-aligned, total is 256-aligned, and reuse happens.
        for slot in &alloc.slots {
            assert_eq!(slot.offset % 256, 0, "offset {} not 256-aligned", slot.offset);
        }
        assert_eq!(alloc.total_bytes % 256, 0, "total not 256-aligned");
        assert!(
            alloc.total_bytes < 200 + 256,
            "non-overlapping should reuse, total={}",
            alloc.total_bytes
        );
    }

    // ── PersistentActivation tests (AltUp per_layer_inputs) ──────────────

    #[test]
    fn persistent_activation_variant_equality_and_debug() {
        let pa = CrossLayerLifetime::PersistentActivation;
        assert_eq!(pa, CrossLayerLifetime::PersistentActivation);
        assert_ne!(pa, CrossLayerLifetime::SpanningAllLayers);
        assert_ne!(pa, CrossLayerLifetime::PerLayer);
        let dbg = format!("{:?}", pa);
        assert!(dbg.contains("PersistentActivation"));
    }

    #[test]
    fn determine_cross_layer_detects_per_layer_input() {
        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor_concrete("altup.per_layer_inputs", &[35, 512, 256], crate::types::DType::F32);
        let lt = Lifetime { tensor_id: tid, first_use: 0, last_use: 50, size_bytes: 35 * 512 * 256 * 4 };
        let ext = TensorLifetimeExt::from_lifetime(&lt, &graph);
        assert_eq!(ext.cross_layer, CrossLayerLifetime::PersistentActivation);
    }

    #[test]
    fn determine_cross_layer_detects_altup_fat_buffer() {
        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor_concrete("altup.fat_in", &[512, 4096], crate::types::DType::F32);
        let lt = Lifetime { tensor_id: tid, first_use: 0, last_use: 50, size_bytes: 512 * 4096 * 4 };
        let ext = TensorLifetimeExt::from_lifetime(&lt, &graph);
        assert_eq!(ext.cross_layer, CrossLayerLifetime::PersistentActivation);
    }

    #[test]
    fn determine_cross_layer_default_intermediate_is_per_layer() {
        let mut graph = CompilerGraph::new();
        let tid = graph.add_tensor_concrete("layer.ffn_intermediate", &[512, 2048], crate::types::DType::F32);
        let lt = Lifetime { tensor_id: tid, first_use: 2, last_use: 5, size_bytes: 512 * 2048 * 4 };
        let ext = TensorLifetimeExt::from_lifetime(&lt, &graph);
        assert_eq!(ext.cross_layer, CrossLayerLifetime::PerLayer);
    }
}
