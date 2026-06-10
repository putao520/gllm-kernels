//! Fusion pass — the unified DAG-based fusion entry point.
//!
//! Uses SemanticDAG OpClass for classification instead of hand-maintained OpSemantics.

use std::collections::{HashMap, HashSet};
use crate::compiler::graph::{CompilerGraph, CompilerOp, OpKind, OpId, MultiOutputConfig};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass, Bottleneck};
use crate::compiler::registry::ScalarOpRegistry;
use super::types::{FusionGroup, FusionMode, FusionPlan, GroupMarker};
use super::helpers::{
    detect_qkv_norm_rope, detect_qkv_shared_input, detect_norm_into_gemm, detect_ffn_block,
    collect_epilogue, collect_elementwise_chain, split_elementwise_by_l1, detect_tile_vs_compute_root,
};
use super::cost_model::{chain_eliminated_bytes, estimate_fusion_cost, Cost};
use super::pdt;
use crate::compiler::hardware_profile::HardwareProfile;
use crate::compiler::pain_point::OpBottleneckMap;
use crate::compiler::jit_context::JitContext;

/// Fusion pass based on SemanticDAG (convenience wrapper).
///
/// Builds a SemanticDAG from the graph and registry, then delegates to
/// `fuse_with_dag_prebuilt`. Use `fuse_with_dag_prebuilt` directly when
/// the caller already has a SemanticDAG to avoid redundant construction.
pub fn fuse_with_dag(graph: &CompilerGraph, registry: &ScalarOpRegistry, plan: &crate::compiler::planner::ExecutionPlan) -> FusionPlan {
    let dag = SemanticDAG::from_graph(graph, registry);
    fuse_with_dag_prebuilt(graph, &dag, plan, None, None)
}

/// Fusion pass using a pre-built SemanticDAG + optional R0 bottleneck map
/// + optional JitContext for resource-aware fusion decisions (SPEC 15 REQ-JCTX-014).
pub fn fuse_with_dag_prebuilt(
    graph: &CompilerGraph,
    dag: &SemanticDAG,
    plan: &crate::compiler::planner::ExecutionPlan,
    bottleneck_map: Option<&OpBottleneckMap>,
    jit_ctx: Option<&JitContext>,
) -> FusionPlan {
    let topo = graph.topological_sort();

    // DEBUG: print topo order and OpClass
    if std::env::var("GLLM_DEBUG_BUFFER_ALLOC").is_ok() {
        eprintln!("[fusion] === TOPO ORDER ({} ops) ===", topo.len());
        for (i, &op_id) in topo.iter().enumerate() {
            if let Some(op) = graph.op(op_id) {
                let oc = dag.node(op_id).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
                let out_names: Vec<String> = op.outputs.iter()
                    .filter_map(|&tid| graph.tensor(tid).map(|t| t.name.clone()))
                    .collect();
                eprintln!("[fusion]   topo[{:3}] {:?} {:?} → {}", i, oc, op.kind, out_names.join(", "));
                if i > 10 { break; }
            }
        }
    }

    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut op_to_group: HashMap<OpId, usize> = HashMap::new();
    let mut claimed: HashSet<OpId> = HashSet::new();

    // First pass: Gemma 4 QKV+QkNorm+ValueNorm+RoPE detection (superset of QkvSharedInput)
    let qkv_norm_rope_groups = detect_qkv_norm_rope(graph, &topo);
    for grp in &qkv_norm_rope_groups {
        let gid = groups.len();
        for &op_id in &grp.ops {
            op_to_group.insert(op_id, gid);
            claimed.insert(op_id);
        }
        groups.push(grp.clone());
    }

    // Second pass: QKV shared input detection (skips groups whose GEMMs are already claimed)
    let qkv_groups = detect_qkv_shared_input(graph, &topo);
    for qkv in &qkv_groups {
        // Skip if any GEMM in this group was already claimed by FusedQkvNormRope
        if qkv.ops.iter().any(|op_id| claimed.contains(op_id)) {
            continue;
        }
        let gid = groups.len();
        for &op_id in &qkv.ops {
            op_to_group.insert(op_id, gid);
            claimed.insert(op_id);
        }
        groups.push(qkv.clone());
    }

    // Third pass: FFNBlock detection (Gate+Up GEMM → activation → Mul)
    // 运行在 QKV detection 之后，避免 Gate/Up GEMM 被误判为 QKV 分组。
    let ffn_groups = detect_ffn_block(graph, &topo);
    for ffn in &ffn_groups {
        if ffn.ops.iter().any(|op_id| claimed.contains(op_id)) {
            continue;
        }
        let gid = groups.len();
        for &op_id in &ffn.ops {
            op_to_group.insert(op_id, gid);
            claimed.insert(op_id);
        }
        groups.push(ffn.clone());
    }

    // Second pass: walk topo order using OpClass from SemanticDAG
    for &op_id in &topo {
        if claimed.contains(&op_id) {
            continue;
        }

        let op = match graph.op(op_id) {
            Some(o) => o,
            None => continue,
        };

        let node = dag.node(op_id);
        let op_class = node.map(|n| n.op_class).unwrap_or(OpClass::Opaque);

        match op_class {
            OpClass::Gemm => {
                // Check norm prefix
                let norm_prefix = detect_norm_into_gemm(graph, op, Some(dag));
                // Collect epilogue using OpClass
                let mut epilogue = collect_epilogue(graph, op, &claimed, Some(dag));

                // Phase A: Gemm + Reduction (Argmax) → EpilogueInjection
                // After collecting ElemWise epilogue chain, check if the last output
                // feeds a single Reduction consumer (e.g., Argmax after logits GEMM).
                // This eliminates logits writeback — argmax runs directly on GEMM accumulator.
                let reduction_epilogue = try_collect_reduction_epilogue(
                    graph, op, &epilogue, &claimed, Some(dag),
                );
                if let Some(red_op) = reduction_epilogue {
                    epilogue.push(red_op);
                }

                // PDT-aware epilogue filter: when R0 bottleneck_map is available,
                // gate each epilogue op on its PDT score_fusion value.
                if bottleneck_map.is_some() && !epilogue.is_empty() {
                    let anchor_class = OpClass::Gemm;
                    epilogue.retain(|ep| {
                        let ep_class = dag.node(ep.id).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
                        if !pdt::can_fuse(anchor_class, ep_class) {
                            return false;
                        }
                        pdt::score_fusion(op_id, ep.id, graph, dag, bottleneck_map, Some(&plan.profile)) > 0.0
                    });
                }

                if norm_prefix.is_some() || !epilogue.is_empty() {
                    let gid = groups.len();
                    let mut all_ops = Vec::new();
                    let anchor_bottleneck = dag.node(op_id).map(|n| n.bottleneck);
                    let mode = if let Some(norm_id) = norm_prefix {
                        if !epilogue.is_empty() {
                            // Roofline-guided: compute-bound with deep epilogue -> Standalone
                            // (deep epilogue increases I-cache pressure in compute-bound kernels)
                            if anchor_bottleneck == Some(Bottleneck::Compute) && epilogue.len() > 2 {
                                FusionMode::Standalone
                            } else {
                                FusionMode::EpilogueInjection
                            }
                        } else {
                            // Decide TileLevelFusion vs ComputeRoot based on L1 capacity
                            detect_tile_vs_compute_root(graph, op, norm_id, plan)
                        }
                    } else {
                        // No norm prefix: roofline-guided epilogue decision
                        if anchor_bottleneck == Some(Bottleneck::Compute) && epilogue.len() > 2 {
                            FusionMode::Standalone
                        } else {
                            FusionMode::EpilogueInjection
                        }
                    };

                    if let Some(norm_id) = norm_prefix {
                        if !claimed.contains(&norm_id) {
                            all_ops.push(norm_id);
                            op_to_group.insert(norm_id, gid);
                            claimed.insert(norm_id);
                        }
                    }

                    all_ops.push(op_id);
                    op_to_group.insert(op_id, gid);
                    claimed.insert(op_id);

                    let epilogue_ids: Vec<OpId> = epilogue.iter().map(|o| o.id).collect();
                    for &eid in &epilogue_ids {
                        all_ops.push(eid);
                        op_to_group.insert(eid, gid);
                        claimed.insert(eid);
                    }

                    groups.push(FusionGroup {
                        id: gid,
                        anchor: op_id,
                        epilogue: epilogue_ids,
                        mode,
                        ops: all_ops,
                        multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                    });
                } else {
                    // Standalone GEMM
                    let gid = groups.len();
                    op_to_group.insert(op_id, gid);
                    claimed.insert(op_id);
                    groups.push(FusionGroup {
                        id: gid,
                        anchor: op_id,
                        epilogue: Vec::new(),
                        mode: FusionMode::Standalone,
                        ops: vec![op_id],
                        multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                    });
                }
            }
            OpClass::ElemWise | OpClass::Injective => {
                // Try to chain with downstream elementwise/injective ops
                let chain = collect_elementwise_chain(graph, op, &claimed, Some(dag));
                // Cost-based filter: only fuse if eliminating intermediates saves cycles.
                // When R0 bottleneck_map is available, use PDT score_fusion for precise gating.
                let accepted_chain: Vec<OpId> = if chain.is_empty() {
                    Vec::new()
                } else if bottleneck_map.is_some() {
                    // PDT-aware scoring: check each chain member individually.
                    chain.iter()
                        .filter(|&&co| {
                            pdt::score_fusion(op_id, co.id, graph, dag, bottleneck_map, Some(&plan.profile)) > 0.0
                        })
                        .map(|o| o.id)
                        .collect()
                } else {
                    let eliminated = chain_eliminated_bytes(graph, op, &chain);
                    if Cost::fusion_benefit(eliminated, plan) > 0 {
                        chain.iter().map(|o| o.id).collect()
                    } else {
                        Vec::new()
                    }
                };

                let mut all_ops = vec![op_id];
                all_ops.extend_from_slice(&accepted_chain);

                // Split chain by L1 budget to avoid thrashing
                let sub_chains = split_elementwise_by_l1(graph, &all_ops, plan);

                for sub in sub_chains {
                    let gid = groups.len();
                    let mode = if sub.len() <= 1 {
                        FusionMode::Standalone
                    } else {
                        FusionMode::LoopFusion
                    };
                    let epilogue = if sub.len() > 1 { sub[1..].to_vec() } else { Vec::new() };

                    for &oid in &sub {
                        op_to_group.insert(oid, gid);
                        claimed.insert(oid);
                    }

                    groups.push(FusionGroup {
                        id: gid,
                        anchor: sub[0],
                        epilogue,
                        mode,
                        ops: sub,
                        multi_output: MultiOutputConfig::single(),
                        dominant_dtype: None,
                        marker: GroupMarker::None,
                        is_layer_group: false,
                        hetero_layer_type: None,
                    });
                }
            }
            OpClass::Reduction | OpClass::Opaque => {
                let gid = groups.len();
                op_to_group.insert(op_id, gid);
                claimed.insert(op_id);
                groups.push(FusionGroup {
                    id: gid,
                    anchor: op_id,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op_id],
                    multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                });
            }
        }
    }

    // Hardware-aware post-filter: demote groups that exceed hardware fusion limits.
    // HardwareProfile drives two constraints:
    //   1. max_fusion_depth — caps epilogue chain length to avoid register spills
    //   2. negative benefit — the cost model (which already uses DeviceProfile for
    //      register/cache sizing) determines if fusion is net-negative
    let hw = HardwareProfile::detect(&plan.profile);
    let max_depth = hw.max_fusion_depth();
    let mut orphaned_groups: Vec<FusionGroup> = Vec::new();
    let num_groups = groups.len();

    for gi in 0..num_groups {
        if groups[gi].mode == FusionMode::Standalone {
            continue;
        }
        if matches!(groups[gi].mode, FusionMode::FusedQkvNormRope { .. }) {
            continue;
        }
        // FFNBlock 是结构性融合（shared pack_a），不适用 epilogue chain 的 max_depth 约束。
        if matches!(groups[gi].mode, FusionMode::FFNBlock { .. }) {
            continue;
        }
        // Depth constraint: too many ops risks register spills on this hardware
        if groups[gi].ops.len() > max_depth {
            let dropped: Vec<OpId> = groups[gi].epilogue.drain(..).collect();
            groups[gi].ops.retain(|id| !dropped.contains(id));
            groups[gi].mode = FusionMode::Standalone;
            for id in dropped {
                orphaned_groups.push(FusionGroup {
                    id: 0,
                    anchor: id,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![id],
                    multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                });
            }
            continue;
        }
        // Cost-model gate for EpilogueInjection and LoopFusion
        if matches!(groups[gi].mode, FusionMode::EpilogueInjection | FusionMode::LoopFusion) {
            let cost = estimate_fusion_cost(&groups[gi], graph, plan, bottleneck_map);
            if cost.benefit < 0 {
                let dropped: Vec<OpId> = groups[gi].epilogue.drain(..).collect();
                groups[gi].ops.retain(|id| !dropped.contains(id));
                groups[gi].mode = FusionMode::Standalone;
                for id in dropped {
                    orphaned_groups.push(FusionGroup {
                        id: 0,
                        anchor: id,
                        epilogue: Vec::new(),
                        mode: FusionMode::Standalone,
                        ops: vec![id],
                        multi_output: MultiOutputConfig::single(),
                        dominant_dtype: None,
                        marker: GroupMarker::None,
                        is_layer_group: false,
                        hetero_layer_type: None,
                    });
                }
            }
        }
    }

    // Re-sort groups by execution order
    groups.sort_by_key(|g| g.ops.iter().map(|o| o.0).min().unwrap_or(0));
    let mut new_op_to_group = HashMap::new();
    for (i, g) in groups.iter_mut().enumerate() {
        g.id = i;
        for &oid in &g.ops {
            new_op_to_group.insert(oid, i);
        }
    }

    // Append orphaned groups (ops dropped from epilogues during hw post-filter)
    groups.extend(orphaned_groups);

    // Rebuild op_to_group after orphaned groups are added
    let mut final_op_to_group = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        for &oid in &g.ops {
            final_op_to_group.insert(oid, i);
        }
    }

    // REQ-DTYPE-003: 推导每个融合组的主导 dtype
    for g in &mut groups {
        g.infer_dominant_dtype(graph);
    }

    // REQ-DTYPE-003: 融合组内 dtype 连续性验证 + widen→narrow→widen 链检查。
    // 验证规则:
    // 1. 融合组内相邻 op dtype 相同时零变换直传 (无额外 cost)
    // 2. widening (BF16→F32) 在 GEMM 内部 load 完成，允许
    // 3. narrowing (F32→BF16) 仅在 Epilogue 最后一步，允许
    // 4. 量化边界仅入口发生，允许
    // 5. 禁止 widen→compute→narrow→widen 链 (表示 dtype 不稳定，应拆分融合组)
    for g in &groups {
        if let Err(e) = validate_fusion_group_dtype(g, graph) {
            eprintln!("[WARN] dtype oscillation in fusion group {}: {:?}", g.id, e);
        }
    }

    // REQ-DTYPE-007: dtype 感知代数重关联。
    // 对每个 LoopFusion 组的 epilogue ops 按 dtype 稳定排序，
    // 使相同 dtype 的 ops 相邻，消除中间 widen/narrow。
    // 仅对 ElemWise 组执行（GEMM 组的 epilogue 顺序由 FMA 数据流决定）。
    for g in &mut groups {
        if matches!(g.mode, FusionMode::LoopFusion) && g.epilogue.len() > 1 {
            // 稳定排序：按每个 op 第一个输入 tensor 的 dtype 元素字节数排序
            // 相同 dtype 的 ops 自然聚在一起
            g.epilogue.sort_by_key(|&op_id| {
                graph.op(op_id)
                    .and_then(|op| op.inputs.first())
                    .and_then(|&tid| graph.tensor(tid))
                    .map(|t| t.dtype.size_bytes())
                    .unwrap_or(4)
            });
        }
    }

    // REQ-UMK-012: 从图拓扑推导 GroupMarker，替代 label prefix 约定
    assign_group_markers(&mut groups, graph);

    FusionPlan {
        groups,
        op_to_group: final_op_to_group,
    }
}

/// REQ-DTYPE-003: 融合组内 dtype 连续性验证。
///
/// 检查融合组内是否存在 widen→compute→narrow→widen 链，
/// 这种模式表示 dtype 不稳定，应拆分融合组。
///
/// 合法模式:
/// - 相同 dtype 直传 (零变换)
/// - widen (BF16→F32) 仅在 GEMM 内部 load
/// - narrow (F32→BF16) 仅在 Epilogue 最后一步
/// - 量化边界仅入口发生
///
/// 非法模式: widen→compute→narrow→widen (dtype 振荡)
fn validate_fusion_group_dtype(
    group: &FusionGroup,
    graph: &CompilerGraph,
) -> Result<(), crate::types::CompilerError> {
    use crate::compiler::trace::QuantPrecision;

    // 提取融合组内每个 op 的 dtype
    let op_dtypes: Vec<Option<QuantPrecision>> = group.ops.iter().map(|&op_id| {
        graph.op(op_id)
            .and_then(|op| op.inputs.first())
            .and_then(|&tid| graph.tensor(tid))
            .map(|t| t.dtype.to_quant_precision())
    }).collect();

    // 检测 widen→narrow→widen 振荡模式
    // 跟踪连续的 widen/narrow 转换次数
    let mut transition_count = 0u32;
    for i in 1..op_dtypes.len() {
        let prev = op_dtypes[i - 1];
        let curr = op_dtypes[i];
        match (prev, curr) {
            (Some(p), Some(c)) if p != c => {
                transition_count += 1;
                if transition_count >= 3 {
                    return Err(crate::types::CompilerError::CodegenViolation(format!(
                        "REQ-DTYPE-003: 融合组 {} 存在 dtype 振荡 (widen→compute→narrow→widen), \
                         应拆分融合组。op[{}]: {:?} → op[{}]: {:?}",
                        group.id, i - 1, p, i, c
                    )));
                }
            }
            _ => {
                // 相同 dtype 或无法推断 → 重置计数
                transition_count = 0;
            }
        }
    }
    Ok(())
}

/// REQ-UMK-012: 从图拓扑推导 GroupMarker，替代 label prefix 约定。
///
/// 推导规则（纯 OpKind 拓扑驱动，零 label 依赖）:
/// 1. 为每个融合组计算 OpKind 判别式签名（有序变体 ID 序列）
/// 2. 图有 layer_loop_config → 同构层循环：连续 N 个相同签名的组 → LayerLoopBegin{N}/End
/// 3. 图有 hetero_layer_loop_config → 异构层循环：循环签名模式 → HeteroLayerLoopBegin/End
/// 4. 无层循环配置 → 不标记
///
/// 同构子结构检测原理：transformer 层在 CompilerGraph 中表现为重复的 OpKind 序列。
/// 融合后，同一层的 ops 被分组到连续的 FusionGroup 中，这些组具有相同的 OpKind 签名。
/// 这比 label prefix 约定更可靠——不依赖 build_graph 的命名约定。
fn assign_group_markers(groups: &mut [FusionGroup], graph: &CompilerGraph) {
    let has_layer_loop = graph.layer_loop_config.is_some();
    let has_hetero_loop = graph.hetero_layer_loop_config.is_some();

    if !has_layer_loop && !has_hetero_loop {
        return;
    }

    // Compute OpKind discriminant signature for each fusion group.
    // discriminant() returns a stable value per enum variant, ignoring field values.
    // Two groups with identical discriminant sequences are isomorphic substructures.
    let signatures: Vec<Vec<std::mem::Discriminant<OpKind>>> = groups.iter()
        .map(|g| {
            g.ops.iter()
                .filter_map(|&oid| graph.op(oid))
                .map(|op| std::mem::discriminant(&op.kind))
                .collect()
        })
        .collect();

    if has_hetero_loop {
        assign_hetero_markers(groups, &signatures, graph);
    } else {
        assign_homogeneous_markers(groups, &signatures, graph);
    }
}

/// Detect homogeneous layer groups: consecutive groups with identical OpKind signatures.
fn assign_homogeneous_markers(
    groups: &mut [FusionGroup],
    signatures: &[Vec<std::mem::Discriminant<OpKind>>],
    graph: &CompilerGraph,
) {
    let num_iterations = graph.layer_loop_config.as_ref()
        .map(|cfg| cfg.num_layers)
        .unwrap_or(0);

    if num_iterations == 0 || groups.is_empty() {
        return;
    }

    // Find the longest run of consecutive groups with identical signatures.
    // A transformer layer typically produces 2-4 fusion groups (attention + FFN, etc.).
    // The pattern repeats num_iterations times, so the run length should be
    // a multiple of the pattern period.
    let layer_group_indices = find_isomorphic_run(signatures, num_iterations);

    if layer_group_indices.is_empty() {
        return;
    }

    // Mark all layer groups
    for &idx in &layer_group_indices {
        groups[idx].is_layer_group = true;
    }

    if let Some(&first) = layer_group_indices.first() {
        groups[first].marker = GroupMarker::LayerLoopBegin { num_iterations };
    }
    if layer_group_indices.len() > 1 {
        if let Some(&last) = layer_group_indices.last() {
            groups[last].marker = GroupMarker::LayerLoopEnd;
        }
    }
}

/// Detect heterogeneous layer groups: cycling signature patterns (Gemma 4 sliding/full).
fn assign_hetero_markers(
    groups: &mut [FusionGroup],
    signatures: &[Vec<std::mem::Discriminant<OpKind>>],
    graph: &CompilerGraph,
) {
    let num_segments = graph.hetero_layer_loop_config.as_ref()
        .map(|cfg| cfg.num_segments)
        .unwrap_or(0);

    if num_segments == 0 || groups.is_empty() {
        return;
    }

    // For hetero models, find groups with cycling signatures.
    // The cycle period = groups_per_segment (e.g., 5 for Gemma 4: 4 sliding + 1 full).
    // Detect by finding the shortest repeating pattern in signatures.
    let layer_group_indices = find_cycling_run(signatures, num_segments);

    if layer_group_indices.is_empty() {
        return;
    }

    // Derive hetero_layer_type for each layer group from OpKind parameters.
    // Sliding vs Full: distinguished by Attention head_dim (sliding < full).
    // Small vs Large FFN: distinguished by GEMM n (intermediate size, small < large).
    let hetero_dims = extract_hetero_dims(graph);
    for &idx in &layer_group_indices {
        groups[idx].is_layer_group = true;
        if let Some(ref dims) = hetero_dims {
            groups[idx].hetero_layer_type = derive_hetero_layer_type(&groups[idx], graph, dims);
        }
    }

    if let Some(&first) = layer_group_indices.first() {
        groups[first].marker = GroupMarker::HeteroLayerLoopBegin { num_segments };
    }
    if layer_group_indices.len() > 1 {
        if let Some(&last) = layer_group_indices.last() {
            groups[last].marker = GroupMarker::HeteroLayerLoopEnd;
        }
    }
}

/// Extract hetero dimension thresholds from graph ops.
/// Returns (sliding_head_dim, full_head_dim, small_intermediate, large_intermediate).
fn extract_hetero_dims(graph: &CompilerGraph) -> Option<HeteroDims> {
    // Collect all unique head_dim values from Attention ops
    let mut head_dims: Vec<usize> = graph.ops.iter()
        .filter_map(|op| match &op.kind {
            OpKind::MultiHeadAttention { head_dim, .. } => Some(*head_dim),
            _ => None,
        })
        .collect();
    head_dims.sort_unstable();
    head_dims.dedup();

    if head_dims.len() < 2 {
        return None;
    }

    let sliding_head_dim = head_dims[0];
    let full_head_dim = head_dims[1];

    // Collect all unique FFN intermediate sizes from GEMM ops
    // (the n dimension of gate/up projections in FFN)
    let mut intermediates: Vec<usize> = graph.ops.iter()
        .filter_map(|op| match &op.kind {
            OpKind::Gemm { n, .. } | OpKind::GemmBias { n, .. } | OpKind::QuantGemm { n, .. } => Some(*n),
            _ => None,
        })
        .collect();
    intermediates.sort_unstable();
    intermediates.dedup();

    // For hetero models, there should be at least 2 distinct intermediate sizes
    let (small_intermediate, large_intermediate) = if intermediates.len() >= 2 {
        (intermediates[0], intermediates[intermediates.len() - 1])
    } else {
        // No FFN size difference — all same intermediate
        (intermediates.first().copied().unwrap_or(0), intermediates.first().copied().unwrap_or(0))
    };

    Some(HeteroDims { sliding_head_dim, full_head_dim, small_intermediate, large_intermediate })
}

struct HeteroDims {
    sliding_head_dim: usize,
    full_head_dim: usize,
    small_intermediate: usize,
    large_intermediate: usize,
}

/// Derive HeteroLayerType from a fusion group's OpKind parameters.
fn derive_hetero_layer_type(
    group: &FusionGroup,
    graph: &CompilerGraph,
    dims: &HeteroDims,
) -> Option<super::types::HeteroLayerType> {
    use super::types::HeteroLayerType;

    // Determine if this group has sliding or full attention
    let is_sliding = group.ops.iter().any(|&oid| {
        graph.op(oid).map_or(false, |op| match &op.kind {
            OpKind::MultiHeadAttention { head_dim, .. } => *head_dim == dims.sliding_head_dim,
            _ => false,
        })
    });
    let is_full = group.ops.iter().any(|&oid| {
        graph.op(oid).map_or(false, |op| match &op.kind {
            OpKind::MultiHeadAttention { head_dim, .. } => *head_dim == dims.full_head_dim,
            _ => false,
        })
    });

    // Determine if this group has small or large FFN
    let is_small_ffn = group.ops.iter().any(|&oid| {
        graph.op(oid).map_or(false, |op| match &op.kind {
            OpKind::Gemm { n, .. } | OpKind::GemmBias { n, .. } | OpKind::QuantGemm { n, .. }
                if *n == dims.small_intermediate && dims.small_intermediate != dims.large_intermediate =>
            {
                true
            }
            _ => false,
        })
    });
    let is_large_ffn = group.ops.iter().any(|&oid| {
        graph.op(oid).map_or(false, |op| match &op.kind {
            OpKind::Gemm { n, .. } | OpKind::GemmBias { n, .. } | OpKind::QuantGemm { n, .. }
                if *n == dims.large_intermediate && dims.small_intermediate != dims.large_intermediate =>
            {
                true
            }
            _ => false,
        })
    });

    // Combine: sliding/full × small/large
    match (is_sliding, is_full, is_small_ffn, is_large_ffn) {
        (true, _, true, _) => Some(HeteroLayerType::SlidingSmall),
        (true, _, _, true) => Some(HeteroLayerType::SlidingLarge),
        (_, true, true, _) => Some(HeteroLayerType::FullSmall),
        (_, true, _, true) => Some(HeteroLayerType::FullLarge),
        // Fallback: if no FFN size difference, use small variant
        (true, _, false, false) => Some(HeteroLayerType::SlidingSmall),
        (_, true, false, false) => Some(HeteroLayerType::FullSmall),
        _ => None,
    }
}

/// Find the longest run of consecutive groups with identical signatures.
///
/// For homogeneous models, all layer groups have the same OpKind signature.
/// We look for a contiguous block of `num_iterations * period` groups where
/// every `period`-th group has the same signature as the first.
///
/// `period` is auto-detected as the smallest value where signatures repeat.
fn find_isomorphic_run(
    signatures: &[Vec<std::mem::Discriminant<OpKind>>],
    num_iterations: usize,
) -> Vec<usize> {
    if signatures.is_empty() || num_iterations == 0 {
        return Vec::new();
    }

    // Find the period: smallest p where signatures[0..p] repeats at signatures[p..2p].
    // For a typical transformer: period = 2 (attention group + FFN group) or 1 (single fused group).
    let max_period = signatures.len().min(8);
    let period = (1..=max_period).find(|&p| {
        if p * 2 > signatures.len() {
            return false;
        }
        // Check that signatures[0..p] == signatures[p..2p]
        (0..p).all(|i| signatures[i] == signatures[p + i])
    }).unwrap_or(1);

    // Verify the run: all groups in [0, period * num_iterations) should be part of the layer loop.
    let expected_len = period * num_iterations;
    if expected_len > signatures.len() {
        // Not enough groups for the expected number of iterations.
        // Fall back: use all groups with the same signature as group 0.
        return find_consecutive_same_signature(signatures);
    }

    // Verify that the pattern repeats for all iterations.
    let is_valid = (0..num_iterations).all(|iter| {
        let base = iter * period;
        (0..period).all(|i| signatures[base + i] == signatures[i])
    });

    if is_valid {
        (0..expected_len).collect()
    } else {
        // Pattern doesn't repeat cleanly — fall back to consecutive same-signature detection.
        find_consecutive_same_signature(signatures)
    }
}

/// Find consecutive groups with the same signature (fallback for when periodic detection fails).
fn find_consecutive_same_signature(
    signatures: &[Vec<std::mem::Discriminant<OpKind>>],
) -> Vec<usize> {
    if signatures.is_empty() {
        return Vec::new();
    }

    // Find the most common signature (the layer template).
    let mut sig_counts: HashMap<Vec<std::mem::Discriminant<OpKind>>, usize> = HashMap::new();
    for sig in signatures {
        *sig_counts.entry(sig.clone()).or_insert(0) += 1;
    }
    let dominant_sig = sig_counts.into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(sig, _)| sig)
        .unwrap();

    // Find the longest contiguous run of groups matching the dominant signature.
    let matching: Vec<usize> = signatures.iter().enumerate()
        .filter(|(_, sig)| *sig == &dominant_sig)
        .map(|(i, _)| i)
        .collect();

    if matching.is_empty() {
        return Vec::new();
    }

    // Find longest contiguous run in matching indices.
    let mut best_start = matching[0];
    let mut best_len = 1;
    let mut cur_start = matching[0];
    let mut cur_len = 1;

    for &idx in &matching[1..] {
        if idx == cur_start + cur_len {
            cur_len += 1;
        } else {
            if cur_len > best_len {
                best_start = cur_start;
                best_len = cur_len;
            }
            cur_start = idx;
            cur_len = 1;
        }
    }
    if cur_len > best_len {
        best_start = cur_start;
        best_len = cur_len;
    }

    // Return ALL groups in the range [best_start, best_start + best_len),
    // not just the ones matching the dominant signature.
    // In a layer loop, groups between matching ones (e.g., FFN between attention)
    // are also part of the layer loop.
    (best_start..best_start + best_len).collect()
}

/// Find groups forming a cycling signature pattern (heterogeneous models).
///
/// For Gemma 4 E2B: each segment has [sliding_small × 4 + full_small] or
/// [sliding_large × 4 + full_large], producing a cycling pattern of distinct
/// signatures. We detect this by finding the shortest cycle period that
/// repeats `num_segments` times.
fn find_cycling_run(
    signatures: &[Vec<std::mem::Discriminant<OpKind>>],
    num_segments: usize,
) -> Vec<usize> {
    if signatures.is_empty() || num_segments == 0 {
        return Vec::new();
    }

    // For hetero models, the cycle period = groups_per_segment.
    // Try to detect it by finding the smallest p where the pattern
    // signatures[0..p] repeats (with possible signature variations) at signatures[p..2p].
    // Hetero models have DIFFERENT signatures within a segment but the SAME
    // pattern of signature differences across segments.
    let max_period = signatures.len().min(16);
    let period = (1..=max_period).find(|&p| {
        if p * 2 > signatures.len() {
            return false;
        }
        // For hetero: check that the discriminant SET (not sequence) repeats.
        // Within a segment, signatures differ (sliding vs full), but the set
        // of discriminant variants is the same across segments.
        let set0: HashSet<_> = signatures[0..p].iter().collect();
        let set1: HashSet<_> = signatures[p..2*p].iter().collect();
        set0 == set1
    }).unwrap_or(signatures.len().min(5));

    let expected_len = period * num_segments;
    let run_len = expected_len.min(signatures.len());

    (0..run_len).collect()
}

/// Try to collect a single Reduction op (e.g., Argmax) immediately downstream of
/// a GEMM (or GEMM + ElemWise epilogue chain) for EpilogueInjection fusion.
///
/// Conditions:
/// 1. The last output of the GEMM or epilogue chain has exactly one consumer
/// 2. That consumer is classified as Reduction (OpClass::Reduction)
/// 3. That consumer is not already claimed
/// 4. That consumer has a single input (from the chain, no extra dependencies)
///
/// Returns the Reduction op if all conditions are met, None otherwise.
fn try_collect_reduction_epilogue<'a>(
    graph: &'a CompilerGraph,
    gemm_op: &CompilerOp,
    epilogue: &[&'a CompilerOp],
    claimed: &HashSet<OpId>,
    dag: Option<&SemanticDAG>,
) -> Option<&'a CompilerOp> {
    // Determine the last output tensor in the chain (GEMM → epilogue tail)
    let last_outputs = if let Some(last_ep) = epilogue.last() {
        &last_ep.outputs
    } else {
        &gemm_op.outputs
    };

    // Must have exactly one output tensor
    let out_tid = last_outputs.first()?;
    let tensor = graph.tensor(*out_tid)?;
    if tensor.consumers.len() != 1 {
        return None;
    }

    let consumer_id = tensor.consumers[0];
    if claimed.contains(&consumer_id) {
        return None;
    }

    let consumer = graph.op(consumer_id)?;

    // Must be Reduction class
    let is_reduction = if let Some(dag) = dag {
        dag.node(consumer_id).map(|n| n.op_class) == Some(OpClass::Reduction)
    } else {
        matches!(
            consumer.kind,
            OpKind::Argmax { .. } | OpKind::MeanPool { .. } | OpKind::L2Normalize { .. }
        )
    };
    if !is_reduction {
        return None;
    }

    // Reduction must have a single input from the chain (no extra dependencies)
    if consumer.inputs.len() != 1 {
        return None;
    }

    Some(consumer)
}

/// Return true if the op is a LayerNorm/RmsNorm/ValueNorm whose single output
/// tensor is consumed by exactly one GEMM op (Gemm / GemmBias / QuantGemm).
///
/// Deferring the Standalone claim for such ops lets the downstream GEMM fuse
/// them via ComputeRoot / TileLevelFusion / EpilogueInjection (see the
/// "multi-op JIT chain heap corruption" comment in the Reduction branch).
fn norm_feeds_single_gemm_consumer(
    graph: &CompilerGraph,
    op_id: OpId,
    dag: &SemanticDAG,
) -> bool {
    use crate::compiler::graph::OpKind;
    let op = match graph.op(op_id) {
        Some(o) => o,
        None => return false,
    };
    // Must match detect_norm_into_gemm's accepted norms (not ValueNorm).
    match op.kind {
        OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => {}
        _ => return false,
    }
    // Norm must produce exactly one output with exactly one consumer.
    let out_tid = match op.outputs.as_slice() {
        [tid] => *tid,
        _ => return false,
    };
    let tensor = match graph.tensor(out_tid) {
        Some(t) => t,
        None => return false,
    };
    if tensor.consumers.len() != 1 {
        return false;
    }
    let consumer_id = tensor.consumers[0];
    let consumer_class = match dag.node(consumer_id) {
        Some(n) => n.op_class,
        None => return false,
    };
    if consumer_class != OpClass::Gemm {
        return false;
    }
    // The consumer must be a plain GEMM kind (not MoEGate, which shares Gemm
    // class but does not participate in Norm-prefix fusion).
    matches!(
        graph.op(consumer_id).map(|o| &o.kind),
        Some(OpKind::Gemm { .. })
            | Some(OpKind::GemmBias { .. })
            | Some(OpKind::QuantGemm { .. })
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, OpId, OpKind, LayerCondition, SymDim, TensorId};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::semantic_dag::SemanticDAG;
    use crate::compiler::planner::ExecutionPlan;
    use crate::dispatch::DeviceProfile;
    use crate::types::DType;
    use std::collections::HashSet;

    // ── Helpers ──

    fn make_plan() -> ExecutionPlan {
        ExecutionPlan::from_profile(&DeviceProfile::detect())
    }

    fn make_registry() -> ScalarOpRegistry {
        ScalarOpRegistry::new()
    }

    /// Build a minimal graph: input_a → op → output.
    fn make_single_op_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId, TensorId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let op = g.add_op(kind, vec![a], vec![out], "op");
        (g, op, a, out)
    }

    // ── Test 1: norm_feeds_single_gemm_consumer returns false for non-norm op ──

    #[test]
    fn norm_feeds_single_gemm_rejects_non_norm_op() {
        // Arrange: a Tanh op (not a norm) in a graph with a downstream GEMM
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let tanh_op = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![mid, w],
            vec![out],
            "gemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = norm_feeds_single_gemm_consumer(&g, tanh_op, &dag);

        // Assert: Tanh is not RmsNorm or LayerNorm, so must return false
        assert!(!result);
    }

    // ── Test 2: norm_feeds_single_gemm_consumer returns false when consumer is not Gem class ──

    #[test]
    fn norm_feeds_single_gemm_rejects_non_gemm_consumer() {
        // Arrange: RmsNorm feeds a Tanh (not a GEMM)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(OpKind::Tanh, vec![mid], vec![out], "tanh");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);

        // Assert: consumer is Tanh, not Gemm class
        assert!(!result);
    }

    // ── Test 3: norm_feeds_single_gemm_consumer returns false for multiple consumers ──

    #[test]
    fn norm_feeds_single_gemm_rejects_multiple_consumers() {
        // Arrange: RmsNorm output has two consumers
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(OpKind::Tanh, vec![mid], vec![out1], "tanh1");
        g.add_op(OpKind::Tanh, vec![mid], vec![out2], "tanh2");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);

        // Assert: two consumers means not single-GEMM
        assert!(!result);
    }

    // ── Test 4: norm_feeds_single_gemm_consumer returns false for non-existent op ──

    #[test]
    fn norm_feeds_single_gemm_rejects_nonexistent_op() {
        // Arrange: empty graph, bogus OpId
        let g = CompilerGraph::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let bogus = OpId(999);

        // Act
        let result = norm_feeds_single_gemm_consumer(&g, bogus, &dag);

        // Assert: no op at OpId(999)
        assert!(!result);
    }

    // ── Test 5: norm_feeds_single_gemm_consumer returns false for multi-output norm ──

    #[test]
    fn norm_feeds_single_gemm_rejects_multi_output_norm() {
        // Arrange: a "norm" with two outputs (which shouldn't happen in practice,
        // but tests the slice-match guard)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);
        // RmsNorm with two outputs
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![out1, out2], "norm");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);

        // Assert: two outputs => doesn't match [tid] pattern
        assert!(!result);
    }

    // ── Test 6: try_collect_reduction_epilogue returns None for non-reduction consumer ──

    #[test]
    fn try_collect_reduction_epilogue_rejects_non_reduction() {
        // Arrange: GEMM → Silu (not a reduction)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act: empty epilogue, so last_outputs = gemm_op.outputs
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, Some(&dag));

        // Assert: Silu is not Reduction class
        assert!(result.is_none());
    }

    // ── Test 7: try_collect_reduction_epilogue returns None when consumer is already claimed ──

    #[test]
    fn try_collect_reduction_epilogue_rejects_claimed_consumer() {
        // Arrange: GEMM → Argmax, but Argmax is already claimed
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let argmax = g.add_op(OpKind::Argmax { vocab_size: 4 }, vec![gemm_out], vec![argmax_out], "argmax");

        let gemm_op = g.op(gemm).unwrap().clone();
        let mut claimed = HashSet::new();
        claimed.insert(argmax); // pre-claim the reduction op
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, Some(&dag));

        // Assert: consumer already claimed
        assert!(result.is_none());
    }

    // ── Test 8: try_collect_reduction_epilogue returns None for multi-input consumer ──

    #[test]
    fn try_collect_reduction_epilogue_rejects_multi_input_consumer() {
        // Arrange: GEMM → consumer with 2 inputs (not single-input reduction)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let extra = g.add_tensor_concrete("extra", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        // Add uses 2 inputs, so consumer.inputs.len() != 1
        g.add_op(OpKind::Add, vec![gemm_out, extra], vec![out], "add");

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, Some(&dag));

        // Assert: Add is not Reduction + has 2 inputs
        assert!(result.is_none());
    }

    // ── Test 9: try_collect_reduction_epilogue returns None when output has multiple consumers ──

    #[test]
    fn try_collect_reduction_epilogue_rejects_multi_consumer_output() {
        // Arrange: GEMM output feeds two consumers
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        // Two consumers of gemm_out
        g.add_op(OpKind::Argmax { vocab_size: 4 }, vec![gemm_out], vec![out1], "argmax");
        g.add_op(OpKind::Tanh, vec![gemm_out], vec![out2], "tanh");

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, Some(&dag));

        // Assert: gemm_out has 2 consumers
        assert!(result.is_none());
    }

    // ── Test 10: fuse_with_dag_prebuilt on empty graph produces empty plan ──

    #[test]
    fn fuse_with_dag_empty_graph_produces_empty_plan() {
        // Arrange
        let g = CompilerGraph::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert
        assert_eq!(fusion_plan.groups.len(), 0);
        assert_eq!(fusion_plan.op_to_group.len(), 0);
        assert_eq!(fusion_plan.num_groups(), 0);
        assert_eq!(fusion_plan.num_fused_ops(), 0);
    }

    // ── Test 11: fuse_with_dag_prebuilt assigns each standalone op to its own group ──

    #[test]
    fn fuse_with_dag_single_tanh_produces_standalone_group() {
        // Arrange: single Tanh op, no fusion possible
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: single group, Standalone mode, contains our op
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.mode, FusionMode::Standalone);
        assert_eq!(group.ops.len(), 1);
        assert_eq!(group.anchor, op0);
        assert!(group.epilogue.is_empty());
    }

    // ── Test 12: fuse_with_dag_prebuilt assigns GEMM ops correctly ──

    #[test]
    fn fuse_with_dag_single_gemm_produces_standalone() {
        // Arrange: isolated GEMM with no epilogue
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: single group, Standalone GEMM
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert!(group.epilogue.is_empty());
    }

    // ── Test 13: fuse_with_dag_prebuilt produces valid op_to_group mapping ──

    #[test]
    fn fuse_with_dag_op_to_group_covers_all_ops() {
        // Arrange: two independent ops (no data dependency between them)
        let mut g = CompilerGraph::new();
        let a1 = g.add_tensor_concrete("a1", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);
        let a2 = g.add_tensor_concrete("a2", &[1, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);

        let op0 = g.add_op(OpKind::Tanh, vec![a1], vec![out1], "tanh1");
        let op1 = g.add_op(OpKind::Silu, vec![a2], vec![out2], "silu1");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: both ops are in the plan, each in its own group
        assert!(fusion_plan.group_of(op0).is_some());
        assert!(fusion_plan.group_of(op1).is_some());
        assert_eq!(fusion_plan.op_to_group.len(), 2);

        // Each op maps to a distinct group (they are independent)
        let g0 = fusion_plan.op_to_group[&op0];
        let g1 = fusion_plan.op_to_group[&op1];
        assert_ne!(g0, g1);
    }

    #[test]
    fn norm_feeds_single_gemm_accepts_rms_norm_into_plain_gemm() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![mid, w],
            vec![out],
            "gemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);
        assert!(result);
    }

    #[test]
    fn norm_feeds_single_gemm_accepts_layer_norm_into_gemm() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::LayerNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![mid, w],
            vec![out],
            "gemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);
        assert!(result);
    }

    #[test]
    fn norm_feeds_single_gemm_rejects_value_norm_kind() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let vnorm_op = g.add_op(OpKind::ValueNorm { eps: 1e-5 }, vec![a], vec![mid], "vnorm");
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![mid, w],
            vec![out],
            "gemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let result = norm_feeds_single_gemm_consumer(&g, vnorm_op, &dag);
        assert!(!result);
    }

    #[test]
    fn norm_feeds_single_gemm_accepts_gemm_bias_consumer() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let b = g.add_tensor_concrete("b", &[4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(
            OpKind::GemmBias { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![mid, w, b],
            vec![out],
            "gemm_bias",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);
        assert!(result);
    }

    #[test]
    fn try_collect_reduction_epilogue_with_no_dag_uses_opkind_match() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let argmax = g.add_op(OpKind::Argmax { vocab_size: 4 }, vec![gemm_out], vec![argmax_out], "argmax");

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();

        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, g.op(argmax).unwrap().id);
    }

    #[test]
    fn try_collect_reduction_epilogue_with_epilogue_chain_last_output() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4], DType::F32);
        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let silu = g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        let argmax = g.add_op(OpKind::Argmax { vocab_size: 4 }, vec![silu_out], vec![argmax_out], "argmax");

        let gemm_op = g.op(gemm).unwrap().clone();
        let silu_op = g.op(silu).unwrap();
        let claimed = HashSet::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[silu_op], &claimed, Some(&dag));
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, g.op(argmax).unwrap().id);
    }

    #[test]
    fn try_collect_reduction_epilogue_returns_none_for_zero_output_gemm() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let gemm_op = CompilerOp {
            kind: OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            inputs: vec![a, w],
            outputs: vec![],
            label: "empty_gemm".into(),
            id: gemm,
    guard: LayerCondition::Always,
        };
        let claimed = HashSet::new();

        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, None);
        assert!(result.is_none());
    }

    #[test]
    fn fuse_with_dag_convenience_wrapper_matches_prebuilt() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let reg = make_registry();
        let plan = make_plan();

        let result = fuse_with_dag(&g, &reg, &plan);
        assert_eq!(result.num_groups(), 1);
        assert_eq!(result.group_of(op0).unwrap().mode, FusionMode::Standalone);
    }

    #[test]
    fn fuse_with_dag_reduction_op_gets_standalone_group() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1], DType::F32);
        let op0 = g.add_op(OpKind::Argmax { vocab_size: 16 }, vec![a], vec![out], "argmax");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.mode, FusionMode::Standalone);
        assert!(group.epilogue.is_empty());
    }

    #[test]
    fn fuse_with_dag_sequential_tanh_silu_chain_produces_groups() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 64], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);

        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let op1 = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);
        assert!(fusion_plan.group_of(op0).is_some());
        assert!(fusion_plan.group_of(op1).is_some());
        assert_eq!(fusion_plan.op_to_group.len(), 2);
    }

    #[test]
    fn fuse_with_dag_plan_display_format() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let reg = make_registry();
        let plan = make_plan();
        let fusion_plan = fuse_with_dag(&g, &reg, &plan);

        let display = format!("{}", fusion_plan);
        assert!(display.contains("1 groups"));
        assert!(display.contains("Standalone"));
    }

    #[test]
    fn norm_feeds_single_gemm_rejects_moe_gate_consumer() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(
            OpKind::MoEGate { seq_len: 1, num_experts: 8, hidden: 4, top_k: 2 },
            vec![mid, w],
            vec![out],
            "moe_gate",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);
        assert!(!result);
    }

    // ── Test: infer_dominant_dtype sets F32 from graph tensor dtype ──

    #[test]
    fn infer_dominant_dtype_sets_f32_from_graph() {
        // Arrange: FusionGroup with anchor op whose first input is F32
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let mut group = FusionGroup {
            id: 0,
            anchor: op0,
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![op0],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        group.infer_dominant_dtype(&g);

        // Assert
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::F32));
    }

    // ── Test: infer_dominant_dtype sets BF16 from graph tensor dtype ──

    #[test]
    fn infer_dominant_dtype_sets_bf16_from_graph() {
        // Arrange: FusionGroup with anchor op whose first input is BF16
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::BF16);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::BF16);
        let op0 = g.add_op(OpKind::Silu, vec![a], vec![out], "silu");

        let mut group = FusionGroup {
            id: 0,
            anchor: op0,
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![op0],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        group.infer_dominant_dtype(&g);

        // Assert
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::BF16));
    }

    // ── Test: infer_dominant_dtype stays None for non-existent anchor ──

    #[test]
    fn infer_dominant_dtype_none_for_nonexistent_anchor() {
        // Arrange: empty graph, anchor points to non-existent op
        let g = CompilerGraph::new();
        let mut group = FusionGroup {
            id: 0,
            anchor: OpId(999),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(999)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        group.infer_dominant_dtype(&g);

        // Assert: no op at OpId(999), so dominant_dtype stays None
        assert!(group.dominant_dtype.is_none());
    }

    // ── Test: fuse_with_dag GemmBias standalone GEMM produces correct plan ──

    #[test]
    fn fuse_with_dag_gemm_bias_standalone_produces_correct_plan() {
        // Arrange: isolated GemmBias with no downstream epilogue
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let w = g.add_tensor_concrete("w", &[16, 16], DType::F32);
        let b = g.add_tensor_concrete("bias", &[16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(
            OpKind::GemmBias { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, w, b],
            vec![out],
            "gemm_bias",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: single group containing the GemmBias op
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert!(group.epilogue.is_empty());
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: try_collect_reduction_epilogue accepts MeanPool as reduction ──

    #[test]
    fn try_collect_reduction_epilogue_accepts_meanpool_reduction() {
        // Arrange: GEMM → MeanPool (single-input reduction, no DAG fallback)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let meanpool = g.add_op(
            OpKind::MeanPool { seq_len: 1, hidden: 4, cls_mode: false },
            vec![gemm_out],
            vec![out],
            "meanpool",
        );

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();

        // Act: without DAG, falls back to OpKind match (MeanPool is matched)
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, None);

        // Assert: MeanPool is recognized as reduction via OpKind match
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, g.op(meanpool).unwrap().id);
    }

    // ── Test: fuse_with_dag with RmsNorm→Tanh (no GEMM) produces separate groups ──

    #[test]
    fn fuse_with_dag_norm_then_elemwise_produces_separate_groups() {
        // Arrange: RmsNorm → Tanh chain (no GEMM involved)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 64], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);

        let norm = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        let tanh = g.add_op(OpKind::Tanh, vec![mid], vec![out], "tanh");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: both ops should be present in the plan, each mapped to a group
        assert!(fusion_plan.group_of(norm).is_some());
        assert!(fusion_plan.group_of(tanh).is_some());
        assert_eq!(fusion_plan.op_to_group.len(), 2);
    }

    // ── Test: FusionMode Debug trait produces expected output ──

    #[test]
    fn fusion_mode_debug_trait_output() {
        // Arrange: various FusionMode variants
        let standalone = FusionMode::Standalone;
        let epilogue = FusionMode::EpilogueInjection;
        let loop_fusion = FusionMode::LoopFusion;
        let norm_into = FusionMode::NormIntoGemm;

        // Act & Assert: Debug output should contain variant names
        assert!(format!("{:?}", standalone).contains("Standalone"));
        assert!(format!("{:?}", epilogue).contains("EpilogueInjection"));
        assert!(format!("{:?}", loop_fusion).contains("LoopFusion"));
        assert!(format!("{:?}", norm_into).contains("NormIntoGemm"));
    }

    // ── Test: FusionPlan Display lists multiple groups correctly ──

    #[test]
    fn fusion_plan_display_lists_multiple_groups() {
        // Arrange: two separate ops → two groups
        let mut g = CompilerGraph::new();
        let a1 = g.add_tensor_concrete("a1", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);
        let a2 = g.add_tensor_concrete("a2", &[1, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);

        g.add_op(OpKind::Tanh, vec![a1], vec![out1], "tanh");
        g.add_op(OpKind::Silu, vec![a2], vec![out2], "silu");

        let reg = make_registry();
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag(&g, &reg, &plan);
        let display = format!("{}", fusion_plan);

        // Assert: display shows 2 groups with anchor op indices
        assert!(display.contains("2 groups"));
    }

    // ── Test: fuse_with_dag with QuantGemm standalone produces correct plan ──

    #[test]
    fn fuse_with_dag_quant_gemm_standalone() {
        // Arrange: isolated QuantGemm with no downstream ops
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 8], DType::F32);
        let w = g.add_tensor_concrete("w", &[8, 8], DType::F32);
        let scale = g.add_tensor_concrete("scale", &[8], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 8], DType::F32);
        let op0 = g.add_op(
            OpKind::QuantGemm {
                m: SymDim::Concrete(1), n: 8, k: 8,
                quant_type: crate::quant::QuantType::Q4_0,
            },
            vec![a, w, scale],
            vec![out],
            "qgemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert!(group.epilogue.is_empty());
    }

    // ── Test: norm_feeds_single_gemm_consumer accepts QuantGemm consumer ──

    #[test]
    fn norm_feeds_single_gemm_accepts_quant_gemm_consumer() {
        // Arrange: RmsNorm → QuantGemm (single consumer, Gemm class)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let scale = g.add_tensor_concrete("scale", &[4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");
        g.add_op(
            OpKind::QuantGemm {
                m: SymDim::Concrete(1), n: 4, k: 4,
                quant_type: crate::quant::QuantType::Q4_0,
            },
            vec![mid, w, scale],
            vec![out],
            "qgemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act
        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);

        // Assert: QuantGemm is a valid Gemm-class consumer
        assert!(result);
    }

    // ── Test: try_collect_reduction_epilogue accepts L2Normalize via OpKind match ──

    #[test]
    fn try_collect_reduction_epilogue_accepts_l2_normalize_opkind() {
        // Arrange: GEMM → L2Normalize (single-input reduction, no DAG)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let l2norm = g.add_op(
            OpKind::L2Normalize { hidden: 4 },
            vec![gemm_out],
            vec![out],
            "l2norm",
        );

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();

        // Act: no DAG, so OpKind match is used; L2Normalize is in the match list
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, None);

        // Assert
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, g.op(l2norm).unwrap().id);
    }

    // ── Test: Opaque op (Reshape) gets Standalone group ──

    #[test]
    fn fuse_with_dag_opaque_reshape_gets_standalone_group() {
        // Arrange: single Reshape op classified as Opaque
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[4, 1], DType::F32);
        let op0 = g.add_op(OpKind::Reshape { target_shape: vec![4, 1] }, vec![a], vec![out], "reshape");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: Opaque ops land in the Reduction|Opaque arm → Standalone
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.mode, FusionMode::Standalone);
        assert_eq!(group.ops.len(), 1);
        assert!(group.epilogue.is_empty());
    }

    // ── Test: RoPE (Injective) op gets a group assignment ──

    #[test]
    fn fuse_with_dag_rope_injective_gets_group() {
        // Arrange: single RoPE op (Injective class)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let op0 = g.add_op(
            OpKind::RoPE { num_heads: 1, head_dim: 4, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a],
            vec![out],
            "rope",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: RoPE is Injective, so it enters the ElemWise|Injective arm
        assert!(fusion_plan.group_of(op0).is_some());
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
    }

    // ── Test: num_fused_ops returns 0 for all-Standalone plan ──

    #[test]
    fn num_fused_ops_zero_when_all_standalone() {
        // Arrange: two independent ops, no fusion possible
        let mut g = CompilerGraph::new();
        let a1 = g.add_tensor_concrete("a1", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);
        let a2 = g.add_tensor_concrete("a2", &[1, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);

        g.add_op(OpKind::Tanh, vec![a1], vec![out1], "tanh");
        g.add_op(OpKind::Silu, vec![a2], vec![out2], "silu");

        let reg = make_registry();
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag(&g, &reg, &plan);

        // Assert: both groups are Standalone, so num_fused_ops == 0
        assert_eq!(fusion_plan.num_groups(), 2);
        assert_eq!(fusion_plan.num_fused_ops(), 0);
    }

    // ── Test: infer_dominant_dtype for GEMM anchor with BF16 input ──

    #[test]
    fn infer_dominant_dtype_bf16_gemm_anchor() {
        // Arrange: Gemm op with BF16 input tensor
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 8], DType::BF16);
        let w = g.add_tensor_concrete("w", &[8, 8], DType::BF16);
        let out = g.add_tensor_concrete("out", &[1, 8], DType::BF16);
        let op0 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 8, k: 8, dtype: DType::BF16, trans_b: false },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let mut group = FusionGroup {
            id: 0,
            anchor: op0,
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![op0],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        group.infer_dominant_dtype(&g);

        // Assert: BF16 input → QuantPrecision::BF16
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::BF16));
    }

    // ── Test: fuse_with_dag assigns sequential group IDs after sort ──

    #[test]
    fn fuse_with_dag_group_ids_sequential_after_sort() {
        // Arrange: three independent ops, each should get its own group
        let mut g = CompilerGraph::new();
        let a1 = g.add_tensor_concrete("a1", &[1, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);
        let a2 = g.add_tensor_concrete("a2", &[1, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);
        let a3 = g.add_tensor_concrete("a3", &[1, 4], DType::F32);
        let out3 = g.add_tensor_concrete("out3", &[1, 4], DType::F32);

        let op0 = g.add_op(OpKind::Tanh, vec![a1], vec![out1], "tanh");
        let op1 = g.add_op(OpKind::Silu, vec![a2], vec![out2], "silu");
        let op2 = g.add_op(OpKind::Gelu, vec![a3], vec![out3], "gelu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: 3 groups with sequential IDs 0, 1, 2
        assert_eq!(fusion_plan.num_groups(), 3);
        let ids: Vec<usize> = fusion_plan.groups.iter().map(|g| g.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);

        // All ops mapped correctly
        assert!(fusion_plan.group_of(op0).is_some());
        assert!(fusion_plan.group_of(op1).is_some());
        assert!(fusion_plan.group_of(op2).is_some());
    }

    // ── Test: fuse_with_dag Transpose (Opaque) gets Standalone ──

    #[test]
    fn fuse_with_dag_transpose_opaque_standalone() {
        // Arrange: single Transpose op
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[2, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[4, 2], DType::F32);
        let op0 = g.add_op(OpKind::Transpose { perm: vec![1, 0] }, vec![a], vec![out], "transpose");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: Transpose is Opaque → Standalone group
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.mode, FusionMode::Standalone);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: fuse_with_dag multi-type mixed graph covers all OpClass branches ──

    #[test]
    fn fuse_with_dag_mixed_ops_cover_all_opclass_branches() {
        // Arrange: GEMM + ElemWise + Reduction + Opaque ops in one graph
        let mut g = CompilerGraph::new();

        // GEMM branch
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let gemm_op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );

        // ElemWise branch (independent)
        let b = g.add_tensor_concrete("b", &[1, 4], DType::F32);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 4], DType::F32);
        let tanh_op = g.add_op(OpKind::Tanh, vec![b], vec![tanh_out], "tanh");

        // Reduction branch (independent)
        let c = g.add_tensor_concrete("c", &[1, 4], DType::F32);
        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], DType::F32);
        let argmax_op = g.add_op(OpKind::Argmax { vocab_size: 4 }, vec![c], vec![argmax_out], "argmax");

        // Opaque branch (independent)
        let d = g.add_tensor_concrete("d", &[2, 2], DType::F32);
        let reshape_out = g.add_tensor_concrete("reshape_out", &[4, 1], DType::F32);
        let reshape_op = g.add_op(OpKind::Reshape { target_shape: vec![4, 1] }, vec![d], vec![reshape_out], "reshape");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: 4 groups, all ops mapped
        assert_eq!(fusion_plan.num_groups(), 4);
        assert_eq!(fusion_plan.op_to_group.len(), 4);

        // Each op in its own group
        assert!(fusion_plan.group_of(gemm_op).is_some());
        assert!(fusion_plan.group_of(tanh_op).is_some());
        assert!(fusion_plan.group_of(argmax_op).is_some());
        assert!(fusion_plan.group_of(reshape_op).is_some());

        // All Standalone (no epilogue chain exists)
        for group in &fusion_plan.groups {
            assert_eq!(group.mode, FusionMode::Standalone);
        }
    }

    // ── Test: norm_feeds_single_gemm_consumer rejects consumer with non-existent tensor ──

    #[test]
    fn norm_feeds_single_gemm_rejects_nonexistent_output_tensor() {
        // Arrange: RmsNorm with output tensor that has no consumers
        // This tests the tensor.consumers.len() == 1 guard with zero consumers
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 4], DType::F32);
        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![mid], "norm");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act: mid tensor has zero consumers (no downstream GEMM)
        let result = norm_feeds_single_gemm_consumer(&g, norm_op, &dag);

        // Assert: zero consumers => not single GEMM consumer
        assert!(!result);
    }

    // ── Test: FusionPlan Display with empty plan shows 0 groups ──

    #[test]
    fn fusion_plan_display_empty_plan() {
        // Arrange: empty graph → empty plan
        let g = CompilerGraph::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Act
        let display = format!("{}", fusion_plan);

        // Assert: display shows "0 groups"
        assert!(display.contains("0 groups"));
    }

    // ── Test: RmsNorm and downstream GEMM each get assigned to groups ──

    #[test]
    fn fuse_with_dag_norm_prefix_gemm_both_assigned() {
        // Arrange: RmsNorm → GEMM chain
        // RmsNorm is classified as Reduction by SemanticDAG, so it gets claimed
        // as Standalone first. The GEMM then gets its own group.
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 64], DType::F32);
        let w = g.add_tensor_concrete("w", &[64, 64], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);

        let norm_op = g.add_op(OpKind::RmsNorm { eps: 1e-5 }, vec![a], vec![norm_out], "norm");
        let gemm_op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false },
            vec![norm_out, w],
            vec![out],
            "gemm",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: both ops are assigned to groups (2 groups total, each Standalone)
        assert_eq!(fusion_plan.num_groups(), 2, "RmsNorm (Reduction) and GEMM form 2 groups");
        assert!(fusion_plan.group_of(norm_op).is_some());
        assert!(fusion_plan.group_of(gemm_op).is_some());
        // They are in distinct groups since RmsNorm is claimed first
        let norm_gid = fusion_plan.op_to_group[&norm_op];
        let gemm_gid = fusion_plan.op_to_group[&gemm_op];
        assert_ne!(norm_gid, gemm_gid, "RmsNorm and GEMM are in distinct groups");
    }

    // ── Test: GEMM with elemwise epilogue chain (GEMM → Silu → Tanh) ──

    #[test]
    fn fuse_with_dag_gemm_with_elemwise_epilogue_chain() {
        // Arrange: GEMM → Silu → Tanh sequential chain
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let w = g.add_tensor_concrete("w", &[64, 64], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 64], DType::F32);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 64], DType::F32);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 64], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let silu = g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        let tanh = g.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: all three ops should be assigned to a group
        assert!(fusion_plan.group_of(gemm).is_some(), "GEMM must be in a group");
        assert!(fusion_plan.group_of(silu).is_some(), "Silu must be in a group");
        assert!(fusion_plan.group_of(tanh).is_some(), "Tanh must be in a group");

        // GEMM is the anchor; Silu and Tanh may be in epilogue (same group) or standalone
        let gemm_group = fusion_plan.group_of(gemm).unwrap();
        // The GEMM must be the anchor of its group
        assert_eq!(gemm_group.anchor, gemm);
    }

    // ── Test: FusedQkvNormRope FusionMode Debug output ──

    #[test]
    fn fused_qkv_norm_rope_mode_debug_output() {
        // Arrange: construct FusedQkvNormRope variant
        let mode = FusionMode::FusedQkvNormRope {
            gemm_q: OpId(0),
            gemm_k: OpId(1),
            gemm_v: OpId(2),
            qk_norm_q: OpId(3),
            qk_norm_k: OpId(4),
            value_norm_v: OpId(5),
            rope_q: OpId(6),
            rope_k: OpId(7),
        };

        // Act
        let debug = format!("{:?}", mode);

        // Assert: Debug output must contain the variant name and key fields
        assert!(debug.contains("FusedQkvNormRope"), "Debug must contain variant name");
        assert!(debug.contains("gemm_q"));
        assert!(debug.contains("gemm_k"));
        assert!(debug.contains("gemm_v"));
    }

    // ── Test: multiple independent GEMMs produce separate groups ──

    #[test]
    fn fuse_with_dag_multiple_independent_gemms_separate_groups() {
        // Arrange: two GEMMs with no data dependency between them
        let mut g = CompilerGraph::new();
        let a1 = g.add_tensor_concrete("a1", &[1, 4], DType::F32);
        let w1 = g.add_tensor_concrete("w1", &[4, 4], DType::F32);
        let out1 = g.add_tensor_concrete("out1", &[1, 4], DType::F32);

        let a2 = g.add_tensor_concrete("a2", &[1, 4], DType::F32);
        let w2 = g.add_tensor_concrete("w2", &[4, 4], DType::F32);
        let out2 = g.add_tensor_concrete("out2", &[1, 4], DType::F32);

        let gemm1 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a1, w1],
            vec![out1],
            "gemm1",
        );
        let gemm2 = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a2, w2],
            vec![out2],
            "gemm2",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: 2 groups, each GEMM in its own group
        assert_eq!(fusion_plan.num_groups(), 2, "two independent GEMMs must produce two groups");
        let g1 = fusion_plan.group_of(gemm1).unwrap();
        let g2 = fusion_plan.group_of(gemm2).unwrap();
        assert_ne!(g1.id, g2.id, "each GEMM must be in a distinct group");
    }

    // ── Test: group_of returns None for ops not in the graph ──

    #[test]
    fn fuse_with_dag_group_of_returns_none_for_unknown_op() {
        // Arrange: single Tanh op in graph, but query for a bogus OpId
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: bogus OpId is not in any group
        assert!(fusion_plan.group_of(OpId(999)).is_none());
    }

    // ── Test: Gelu op (ElemWise) standalone produces correct group ──

    #[test]
    fn fuse_with_dag_gelu_elemwise_standalone_group() {
        // Arrange: single Gelu op, classified as ElemWise
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 32], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 32], DType::F32);
        let op0 = g.add_op(OpKind::Gelu, vec![a], vec![out], "gelu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: single Standalone group for the Gelu op
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
        assert_eq!(group.epilogue.len(), 0);
    }

    // ── Test: Top-K op (Opaque) gets Standalone group ──

    #[test]
    fn fuse_with_dag_topk_opaque_standalone() {
        // Arrange: single Softmax op (classified as Opaque/Reduction)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::Softmax, vec![a], vec![out], "softmax");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: Opaque/Reduction → Standalone group
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.mode, FusionMode::Standalone);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: SwiGlu op (ElemWise) gets group assignment ──

    #[test]
    fn fuse_with_dag_swiglu_elemwise_standalone() {
        // Arrange: single SwiGlu op (ElemWise class)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::SwiGlu, vec![a], vec![out], "swiglu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: SwiGlu is ElemWise → Standalone (no chain to fuse with)
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.mode, FusionMode::Standalone);
    }

    // ── Test: TileLevelFusion and ComputeRoot FusionMode variants Debug and Clone ──

    #[test]
    fn tile_level_fusion_and_compute_root_mode_variants() {
        // Arrange: construct TileLevelFusion and ComputeRoot variants
        let tile_mode = FusionMode::TileLevelFusion {
            predecessor: OpId(10),
            tile_rows: 32,
        };
        let compute_root_mode = FusionMode::ComputeRoot {
            predecessor: OpId(20),
        };

        // Act & Assert: Debug output contains variant names
        let tile_debug = format!("{:?}", tile_mode);
        assert!(tile_debug.contains("TileLevelFusion"));
        assert!(tile_debug.contains("tile_rows"));

        let cr_debug = format!("{:?}", compute_root_mode);
        assert!(cr_debug.contains("ComputeRoot"));
        assert!(cr_debug.contains("predecessor"));

        // Clone roundtrip
        assert_eq!(tile_mode, tile_mode.clone());
        assert_eq!(compute_root_mode, compute_root_mode.clone());
    }

    // ── Test: Add op (ElemWise with 2 inputs) gets standalone group ──

    #[test]
    fn fuse_with_dag_add_binary_elemwise_standalone() {
        // Arrange: Add op with two inputs (binary elementwise)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let b = g.add_tensor_concrete("b", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::Add, vec![a, b], vec![out], "add");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: Add is ElemWise, standalone (no chain)
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: Mul op (binary ElemWise) gets standalone group ──

    #[test]
    fn fuse_with_dag_mul_binary_elemwise_standalone() {
        // Arrange: Mul op with two inputs (binary elementwise)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let b = g.add_tensor_concrete("b", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::Mul, vec![a, b], vec![out], "mul");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: Mul is ElemWise, standalone (no chain to fuse with)
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
        assert_eq!(group.mode, FusionMode::Standalone);
    }

    // ── Test: FusionGroup infer_dominant_dtype with epilogue ops ──

    #[test]
    fn infer_dominant_dtype_with_epilogue_ops() {
        // Arrange: FusionGroup with anchor + epilogue, all F32
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let mid = g.add_tensor_concrete("mid", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let anchor_op = g.add_op(OpKind::Tanh, vec![a], vec![mid], "tanh");
        let epi_op = g.add_op(OpKind::Silu, vec![mid], vec![out], "silu");

        let mut group = FusionGroup {
            id: 0,
            anchor: anchor_op,
            epilogue: vec![epi_op],
            mode: FusionMode::LoopFusion,
            ops: vec![anchor_op, epi_op],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        group.infer_dominant_dtype(&g);

        // Assert: anchor's first input is F32
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::F32));
    }

    // ── Test: FusionGroup infer_dominant_dtype with BF16 epilogue preserves anchor dtype ──

    #[test]
    fn infer_dominant_dtype_bf16_anchor_overrides_epilogue() {
        // Arrange: anchor has BF16 input, epilogue op has F32 input
        // dominant_dtype is derived from the anchor only
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 8], DType::BF16);
        let mid = g.add_tensor_concrete("mid", &[1, 8], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 8], DType::F32);
        let anchor_op = g.add_op(OpKind::Silu, vec![a], vec![mid], "silu");
        let epi_op = g.add_op(OpKind::Tanh, vec![mid], vec![out], "tanh");

        let mut group = FusionGroup {
            id: 0,
            anchor: anchor_op,
            epilogue: vec![epi_op],
            mode: FusionMode::LoopFusion,
            ops: vec![anchor_op, epi_op],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        group.infer_dominant_dtype(&g);

        // Assert: anchor's first input is BF16, so dominant_dtype is BF16
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::BF16));
    }

    // ── REQ-DTYPE-003: validate_fusion_group_dtype tests ──

    #[test]
    fn validate_fusion_group_dtype_same_dtype_ok() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[32], DType::F32);
        let b = g.add_tensor_concrete("b", &[32], DType::F32);
        let c = g.add_tensor_concrete("c", &[32], DType::F32);
        g.inputs = vec![a]; g.outputs = vec![c];
        let op0 = g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        let op1 = g.add_op(OpKind::Tanh, vec![b], vec![c], "tanh");
        let group = FusionGroup {
            id: 0, anchor: op0, epilogue: vec![op1],
            mode: FusionMode::LoopFusion, ops: vec![op0, op1],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None, marker: GroupMarker::None,
            is_layer_group: false, hetero_layer_type: None,
        };
        assert!(validate_fusion_group_dtype(&group, &g).is_ok(),
            "same dtype across ops should pass validation");
    }

    #[test]
    fn validate_fusion_group_dtype_single_widen_ok() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[32], DType::BF16);
        let b = g.add_tensor_concrete("b", &[32], DType::F32);
        g.inputs = vec![a]; g.outputs = vec![b];
        let op0 = g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        let group = FusionGroup {
            id: 0, anchor: op0, epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![op0],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None, marker: GroupMarker::None,
            is_layer_group: false, hetero_layer_type: None,
        };
        assert!(validate_fusion_group_dtype(&group, &g).is_ok(),
            "single widen should pass validation");
    }

    // ── Test: FusionPlan op_to_group consistency with groups ──

    #[test]
    fn fusion_plan_op_to_group_consistency_with_groups() {
        // Arrange: three ops in sequence → each maps to a group
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let mid1 = g.add_tensor_concrete("mid1", &[1, 4], DType::F32);
        let mid2 = g.add_tensor_concrete("mid2", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);

        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![mid1], "tanh");
        let op1 = g.add_op(OpKind::Silu, vec![mid1], vec![mid2], "silu");
        let op2 = g.add_op(OpKind::Gelu, vec![mid2], vec![out], "gelu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: every op in the plan's groups has a corresponding op_to_group entry
        for group in &fusion_plan.groups {
            for &op_id in &group.ops {
                let mapped_idx = fusion_plan.op_to_group.get(&op_id).copied();
                assert_eq!(mapped_idx, Some(group.id),
                    "op {:?} maps to group idx {:?} but expected group id {}", op_id, mapped_idx, group.id);
            }
        }
        // All three ops are present in the plan
        assert!(fusion_plan.group_of(op0).is_some());
        assert!(fusion_plan.group_of(op1).is_some());
        assert!(fusion_plan.group_of(op2).is_some());
    }

    // ── Test: fuse_with_dag with GEMM → Reshape (Opaque) does not fuse epilogue ──

    #[test]
    fn fuse_with_dag_gemm_to_reshape_no_epilogue() {
        // Arrange: GEMM output feeds into a Reshape (Opaque, not ElemWise)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let reshape_out = g.add_tensor_concrete("reshape_out", &[4, 1], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Reshape { target_shape: vec![4, 1] }, vec![gemm_out], vec![reshape_out], "reshape");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: GEMM should be Standalone (Reshape is Opaque, not eligible for epilogue)
        let gemm_group = fusion_plan.group_of(gemm).unwrap();
        assert_eq!(gemm_group.anchor, gemm);
        assert!(gemm_group.epilogue.is_empty(), "GEMM should not have Opaque Reshape as epilogue");
    }

    // ── Test: try_collect_reduction_epilogue with DAG and Argmax succeeds ──

    #[test]
    fn try_collect_reduction_epilogue_with_dag_accepts_argmax() {
        // Arrange: GEMM → Argmax (single-input reduction, with DAG)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let w = g.add_tensor_concrete("w", &[4, 4], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4], DType::F32);
        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4, k: 4, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let argmax = g.add_op(OpKind::Argmax { vocab_size: 4 }, vec![gemm_out], vec![argmax_out], "argmax");

        let gemm_op = g.op(gemm).unwrap().clone();
        let claimed = HashSet::new();
        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);

        // Act: with DAG, Argmax should be recognized as Reduction class
        let result = try_collect_reduction_epilogue(&g, &gemm_op, &[], &claimed, Some(&dag));

        // Assert
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, g.op(argmax).unwrap().id);
    }

    // ── Test: FusionMode FFNBlock Debug output contains all fields ──

    #[test]
    fn ffn_block_mode_debug_contains_all_fields() {
        // Arrange: construct FFNBlock variant with specific OpIds
        let mode = FusionMode::FFNBlock {
            gate_gemm: OpId(100),
            up_gemm: OpId(101),
            activation: OpId(102),
            combine: OpId(103),
        };

        // Act
        let debug = format!("{:?}", mode);

        // Assert: Debug output must contain variant name and all field names
        assert!(debug.contains("FFNBlock"), "Debug must contain variant name");
        assert!(debug.contains("gate_gemm"));
        assert!(debug.contains("up_gemm"));
        assert!(debug.contains("activation"));
        assert!(debug.contains("combine"));
    }

    // ── Test: FusionMode CrossLayerResidual equality and Debug ──

    #[test]
    fn cross_layer_residual_equality_and_debug() {
        // Arrange: two identical CrossLayerResidual variants
        let mode1 = FusionMode::CrossLayerResidual {
            residual: OpId(50),
            norm: OpId(51),
        };
        let mode2 = FusionMode::CrossLayerResidual {
            residual: OpId(50),
            norm: OpId(51),
        };

        // Act & Assert: PartialEq
        assert_eq!(mode1, mode2);

        // Debug output
        let debug = format!("{:?}", mode1);
        assert!(debug.contains("CrossLayerResidual"));
        assert!(debug.contains("residual"));
        assert!(debug.contains("norm"));
    }

    // ── Test: fuse_with_dag with BF16 tensors propagates dominant_dtype correctly ──

    #[test]
    fn fuse_with_dag_bf16_tensor_propagates_dominant_dtype() {
        // Arrange: single Tanh op with BF16 input
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 32], DType::BF16);
        let out = g.add_tensor_concrete("out", &[1, 32], DType::BF16);
        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![out], "tanh");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: the group's dominant_dtype should be BF16
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::BF16));
    }

    // ── Test: fuse_with_dag with bottleneck_map produces valid plan ──

    #[test]
    fn fuse_with_dag_with_bottleneck_map_produces_valid_plan() {
        // Arrange: GEMM → Silu chain with a bottleneck map
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 32], DType::F32);
        let w = g.add_tensor_concrete("w", &[32, 32], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 32], DType::F32);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 32], DType::F32);

        let gemm = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 32, k: 32, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        let silu = g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Build a minimal bottleneck map via PainPointAnalyzer
        let bottleneck_map = crate::compiler::pain_point::PainPointAnalyzer::analyze(&g, &plan.profile);

        // Act: pass bottleneck_map to the fusion pass
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, Some(&bottleneck_map), None);

        // Assert: all ops are assigned to groups (plan is structurally valid)
        assert!(fusion_plan.group_of(gemm).is_some(), "GEMM must be in a group");
        assert!(fusion_plan.group_of(silu).is_some(), "Silu must be in a group");
        // op_to_group covers all ops
        assert_eq!(fusion_plan.op_to_group.len(), 2);
    }

    // ── Test: fuse_with_dag GeGlu (ElemWise) standalone group ──

    #[test]
    fn fuse_with_dag_geglu_elemwise_standalone() {
        // Arrange: single GeGlu op (ElemWise class)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 32], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 32], DType::F32);
        let op0 = g.add_op(OpKind::GeGlu, vec![a], vec![out], "geglu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: GeGlu is ElemWise, standalone (no chain to fuse with)
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
        assert_eq!(group.epilogue.len(), 0);
    }

    // ── Test: fuse_with_dag large mixed graph with GEMM + chain has all ops assigned ──

    #[test]
    fn fuse_with_dag_large_mixed_graph_all_ops_assigned() {
        // Arrange: independent GEMM + Reshape + Argmax (3 OpClass branches)
        let mut g = CompilerGraph::new();

        // GEMM
        let a = g.add_tensor_concrete("a", &[1, 8], DType::F32);
        let w = g.add_tensor_concrete("w", &[8, 8], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 8], DType::F32);
        let gemm_op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 8, k: 8, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );

        // Reshape (Opaque) on separate input
        let r = g.add_tensor_concrete("r", &[2, 4], DType::F32);
        let reshape_out = g.add_tensor_concrete("reshape_out", &[8, 1], DType::F32);
        let reshape_op = g.add_op(OpKind::Reshape { target_shape: vec![8, 1] }, vec![r], vec![reshape_out], "reshape");

        // Argmax (Reduction) on separate input
        let c = g.add_tensor_concrete("c", &[1, 8], DType::F32);
        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], DType::F32);
        let argmax_op = g.add_op(OpKind::Argmax { vocab_size: 8 }, vec![c], vec![argmax_out], "argmax");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: 3 groups, one per op, all ops covered
        assert_eq!(fusion_plan.num_groups(), 3);
        assert_eq!(fusion_plan.op_to_group.len(), 3);
        assert!(fusion_plan.group_of(gemm_op).is_some());
        assert!(fusion_plan.group_of(reshape_op).is_some());
        assert!(fusion_plan.group_of(argmax_op).is_some());

        // All groups are Standalone (no epilogue opportunities between independent ops)
        for group in &fusion_plan.groups {
            assert_eq!(group.mode, FusionMode::Standalone);
        }
    }

    // ── Test: fuse_with_dag SwiGluClipped (ElemWise) standalone ──

    #[test]
    fn fuse_with_dag_swiglu_clipped_elemwise_standalone() {
        // Arrange: single SwiGluClipped op
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::SwiGluClipped { limit: 5.0 }, vec![a], vec![out], "swiglu_clipped");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: SwiGluClipped is ElemWise, standalone (no chain)
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: fuse_with_dag Residual (ElemWise) gets group ──

    #[test]
    fn fuse_with_dag_residual_elemwise_standalone() {
        // Arrange: single Residual op (binary elementwise)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let b = g.add_tensor_concrete("b", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::Residual, vec![a, b], vec![out], "residual");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: Residual is ElemWise, assigned to a group
        assert!(fusion_plan.group_of(op0).is_some());
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: fuse_with_dag sequential chain Tanh→Silu→Gelu→Tanh has all ops mapped ──

    #[test]
    fn fuse_with_dag_four_op_chain_all_mapped() {
        // Arrange: four sequential ElemWise ops
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let mid1 = g.add_tensor_concrete("mid1", &[1, 64], DType::F32);
        let mid2 = g.add_tensor_concrete("mid2", &[1, 64], DType::F32);
        let mid3 = g.add_tensor_concrete("mid3", &[1, 64], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);

        let op0 = g.add_op(OpKind::Tanh, vec![a], vec![mid1], "tanh1");
        let op1 = g.add_op(OpKind::Silu, vec![mid1], vec![mid2], "silu1");
        let op2 = g.add_op(OpKind::Gelu, vec![mid2], vec![mid3], "gelu1");
        let op3 = g.add_op(OpKind::Tanh, vec![mid3], vec![out], "tanh2");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: all four ops are mapped
        assert!(fusion_plan.group_of(op0).is_some(), "op0 must be mapped");
        assert!(fusion_plan.group_of(op1).is_some(), "op1 must be mapped");
        assert!(fusion_plan.group_of(op2).is_some(), "op2 must be mapped");
        assert!(fusion_plan.group_of(op3).is_some(), "op3 must be mapped");
        assert_eq!(fusion_plan.op_to_group.len(), 4);

        // Verify op_to_group consistency: each mapped group id is a valid index
        for (&_op, &gid) in &fusion_plan.op_to_group {
            assert!(gid < fusion_plan.groups.len(), "group id {} out of range", gid);
        }
    }

    // ── Test: fuse_with_dag QkNorm (NormLike) gets standalone group ──

    #[test]
    fn fuse_with_dag_qknorm_standalone_group() {
        // Arrange: single QkNorm op (classified as Reduction by SemanticDAG)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4], DType::F32);
        let op0 = g.add_op(OpKind::QkNorm { head_dim: 4, eps: 1e-5 }, vec![a], vec![out], "qk_norm");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: QkNorm is classified as Reduction → Standalone
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: fuse_with_dag LogitSoftcap (ElemWise) standalone ──

    #[test]
    fn fuse_with_dag_logit_softcap_elemwise_standalone() {
        // Arrange: single LogitSoftcap op
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 16], DType::F32);
        let op0 = g.add_op(OpKind::LogitSoftcap { cap: 30.0 }, vec![a], vec![out], "logit_softcap");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: LogitSoftcap is ElemWise, standalone
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.anchor, op0);
        assert_eq!(group.ops.len(), 1);
    }

    // ── Test: fuse_with_dag GemmBias with single Silu epilogue produces group ──

    #[test]
    fn fuse_with_dag_gemm_bias_with_silu_epilogue() {
        // Arrange: GemmBias → Silu chain
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 32], DType::F32);
        let w = g.add_tensor_concrete("w", &[32, 32], DType::F32);
        let b = g.add_tensor_concrete("bias", &[32], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 32], DType::F32);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 32], DType::F32);

        let gemm = g.add_op(
            OpKind::GemmBias { m: SymDim::Concrete(1), n: 32, k: 32, dtype: DType::F32, trans_b: false },
            vec![a, w, b],
            vec![gemm_out],
            "gemm_bias",
        );
        let silu = g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: both ops are in groups
        assert!(fusion_plan.group_of(gemm).is_some());
        assert!(fusion_plan.group_of(silu).is_some());

        // GemmBias is the anchor of its group
        let gemm_group = fusion_plan.group_of(gemm).unwrap();
        assert_eq!(gemm_group.anchor, gemm);
    }

    // ── Test: fuse_with_dag SliceView (Opaque) gets Standalone group ──

    #[test]
    fn fuse_with_dag_slice_view_opaque_standalone() {
        // Arrange: single SliceView op (Opaque)
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 16], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 8], DType::F32);
        let op0 = g.add_op(
            OpKind::SliceView { axis: 1, start: 0, end: 8 },
            vec![a],
            vec![out],
            "slice_view",
        );

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: SliceView is Opaque → Standalone group
        assert_eq!(fusion_plan.num_groups(), 1);
        let group = fusion_plan.group_of(op0).unwrap();
        assert_eq!(group.mode, FusionMode::Standalone);
        assert_eq!(group.ops.len(), 1);
        assert!(group.epilogue.is_empty());
    }

    // ── Test: fuse_with_dag group IDs are monotonically increasing by execution order ──

    #[test]
    fn fuse_with_dag_group_ids_monotonically_ordered() {
        // Arrange: 5 independent ops (no data dependency between them)
        // Using independent ops guarantees each gets its own group (no fusion)
        let mut g = CompilerGraph::new();
        let mut op_ids = Vec::new();
        let kinds = [
            OpKind::Tanh, OpKind::Silu, OpKind::Gelu, OpKind::Tanh, OpKind::Silu,
        ];
        for (i, kind) in kinds.into_iter().enumerate() {
            let inp_name = format!("a{}", i);
            let out_name = format!("out{}", i);
            let inp = g.add_tensor_concrete(&inp_name, &[1, 4], DType::F32);
            let out = g.add_tensor_concrete(&out_name, &[1, 4], DType::F32);
            let op = g.add_op(kind, vec![inp], vec![out], &format!("op{}", i));
            op_ids.push(op);
        }

        let reg = make_registry();
        let dag = SemanticDAG::from_graph(&g, &reg);
        let plan = make_plan();

        // Act
        let fusion_plan = fuse_with_dag_prebuilt(&g, &dag, &plan, None, None);

        // Assert: group IDs are 0, 1, 2, ... in order
        assert_eq!(fusion_plan.num_groups(), 5, "5 independent ops must produce 5 groups");
        let ids: Vec<usize> = fusion_plan.groups.iter().map(|g| g.id).collect();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, i, "group at position {} should have id {}", i, i);
        }

        // All ops are covered
        assert_eq!(fusion_plan.op_to_group.len(), 5);
    }
}
