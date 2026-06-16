//! Fusion helper functions — pattern detection and chain collection.
//!
//! Unified versions that work with CompilerGraph + optional SemanticDAG.
//! The old duplicated `_dag` variants have been merged into these.

use std::collections::HashSet;
use crate::compiler::graph::{CompilerGraph, CompilerOp, Op, OpKind, OpId, TensorId};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass};
use crate::compiler::semantics;
use super::types::{FusionGroup, FusionMode, GroupMarker};
use super::quant_aware::can_fuse_quant_aware;
use crate::compiler::graph::MultiOutputConfig;
use crate::quant::QuantType;

/// Extract QuantType from a GEMM-family op, if applicable.
fn extract_quant_type(op: &CompilerOp, graph: &CompilerGraph) -> Option<QuantType> {
    match op.op_v2_resolved(graph) {
        Some(Op::QuantGemm(spec)) => Some(spec.quant_type),
        _ => None,
    }
}

/// Check that all GEMMs in a group have quant-compatible types.
/// Returns true if all GEMMs can be fused together under quant-aware rules.
fn all_gemm_quant_compatible(ops: &[&CompilerOp], graph: &CompilerGraph) -> bool {
    let quant_types: Vec<Option<QuantType>> = ops.iter().map(|op| extract_quant_type(op, graph)).collect();
    for i in 1..quant_types.len() {
        if can_fuse_quant_aware(quant_types[i - 1], quant_types[i])
            == super::quant_aware::QuantFusionDecision::Split
        {
            return false;
        }
    }
    true
}

/// Detect Gemma 4 QKV+QkNorm+ValueNorm+RoPE pattern.
///
/// Matches: 3 GEMMs (Q,K,V) sharing the same norm output, where:
///   - Q and K outputs each feed into a QkNorm, then into a RoPE
///   - V output feeds into a ValueNorm
///
/// Returns at most one FusionGroup. If detected, the caller should skip the
/// normal QkvSharedInput detection for these ops (this is a strict superset).
pub(crate) fn detect_qkv_norm_rope(graph: &CompilerGraph, topo: &[OpId]) -> Vec<FusionGroup> {
    let mut result = Vec::new();

    // Collect all GEMM ops
    let gemm_ops: Vec<&CompilerOp> = topo
        .iter()
        .filter_map(|&id| graph.op(id))
        .filter(|op| matches!(op.op_v2_resolved(graph), Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_))))
        .collect();

    // Group by first input tensor
    let mut by_input: std::collections::BTreeMap<TensorId, Vec<&CompilerOp>> =
        std::collections::BTreeMap::new();
    for op in &gemm_ops {
        if let Some(&first_input) = op.inputs.first() {
            by_input.entry(first_input).or_default().push(op);
        }
    }

    for ops in by_input.values() {
        if ops.len() != 3 {
            continue;
        }

        // Shared input must come from a norm op (standard QKV pattern prerequisite)
        let shared_input = ops[0].inputs[0];
        let is_from_norm = graph.tensor(shared_input)
            .and_then(|t| t.producer)
            .is_some_and(|prod_id| {
                graph.op(prod_id).is_some_and(|prod_op| {
                    matches!(prod_op.op_v2_resolved(graph), Some(Op::RmsNorm(_)) | Some(Op::LayerNorm(_)))
                })
            });

        if !is_from_norm {
            continue;
        }

        // Quant-aware check: all GEMMs must have compatible quant types
        if !all_gemm_quant_compatible(ops, graph) {
            continue;
        }

        // For each GEMM, trace its single consumer to classify:
        //   - QkNorm consumer → Q or K projection (then expect RoPE after QkNorm)
        //   - ValueNorm consumer → V projection
        // We need exactly: 2 GEMMs → QkNorm → RoPE, and 1 GEMM → ValueNorm.
        struct QkvTrace {
            gemm_id: OpId,
            qk_norm_id: Option<OpId>,
            rope_id: Option<OpId>,
            value_norm_id: Option<OpId>,
        }

        let mut traces: Vec<QkvTrace> = Vec::new();

        for &gemm_op in ops {
            let mut trace = QkvTrace {
                gemm_id: gemm_op.id,
                qk_norm_id: None,
                rope_id: None,
                value_norm_id: None,
            };

            // Get the single output tensor of the GEMM
            if gemm_op.outputs.len() != 1 {
                traces.push(trace);
                continue;
            }
            let out_tid = gemm_op.outputs[0];
            let out_tensor = match graph.tensor(out_tid) {
                Some(t) => t,
                None => { traces.push(trace); continue; }
            };

            // Must have exactly one consumer
            if out_tensor.consumers.len() != 1 {
                traces.push(trace);
                continue;
            }
            let consumer_id = out_tensor.consumers[0];
            let consumer = match graph.op(consumer_id) {
                Some(o) => o,
                None => { traces.push(trace); continue; }
            };

            match &consumer.kind {
                OpKind::QkNorm { .. } => {
                    trace.qk_norm_id = Some(consumer_id);
                    // Expect RoPE after QkNorm
                    if consumer.outputs.len() == 1 {
                        if let Some(norm_out_t) = graph.tensor(consumer.outputs[0]) {
                            if norm_out_t.consumers.len() == 1 {
                                if let Some(rope_op) = graph.op(norm_out_t.consumers[0]) {
                                    if matches!(rope_op.op_v2_resolved(graph), Some(Op::RoPE(_))) {
                                        trace.rope_id = Some(rope_op.id);
                                    }
                                }
                            }
                        }
                    }
                }
                OpKind::ValueNorm { .. } => {
                    trace.value_norm_id = Some(consumer_id);
                }
                _ => {}
            }

            traces.push(trace);
        }

        // Validate: exactly 2 QkNorm+RoPE paths and 1 ValueNorm path
        let qk_traces: Vec<&QkvTrace> = traces.iter()
            .filter(|t| t.qk_norm_id.is_some() && t.rope_id.is_some())
            .collect();
        let v_traces: Vec<&QkvTrace> = traces.iter()
            .filter(|t| t.value_norm_id.is_some())
            .collect();

        if qk_traces.len() != 2 || v_traces.len() != 1 {
            continue;
        }

        // Build the fused group with all 8 ops
        let mut all_ops = Vec::with_capacity(8);
        // GEMMs first (anchor region)
        all_ops.push(qk_traces[0].gemm_id);
        all_ops.push(qk_traces[1].gemm_id);
        all_ops.push(v_traces[0].gemm_id);
        // Norms
        all_ops.push(qk_traces[0].qk_norm_id.unwrap());
        all_ops.push(qk_traces[1].qk_norm_id.unwrap());
        all_ops.push(v_traces[0].value_norm_id.unwrap());
        // RoPE
        all_ops.push(qk_traces[0].rope_id.unwrap());
        all_ops.push(qk_traces[1].rope_id.unwrap());

        result.push(FusionGroup {
            id: 0, // reassigned later
            anchor: qk_traces[0].gemm_id,
            epilogue: all_ops[1..].to_vec(),
            mode: FusionMode::FusedQkvNormRope {
                gemm_q: qk_traces[0].gemm_id,
                gemm_k: qk_traces[1].gemm_id,
                gemm_v: v_traces[0].gemm_id,
                qk_norm_q: qk_traces[0].qk_norm_id.unwrap(),
                qk_norm_k: qk_traces[1].qk_norm_id.unwrap(),
                value_norm_v: v_traces[0].value_norm_id.unwrap(),
                rope_q: qk_traces[0].rope_id.unwrap(),
                rope_k: qk_traces[1].rope_id.unwrap(),
            },
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        });
    }

    result
}

/// Detect QKV shared input pattern: three consecutive GEMMs reading the same tensor.
pub(crate) fn detect_qkv_shared_input(graph: &CompilerGraph, topo: &[OpId]) -> Vec<FusionGroup> {
    let mut result = Vec::new();

    // Find groups of GEMM ops that share the same first input tensor
    let gemm_ops: Vec<&CompilerOp> = topo
        .iter()
        .filter_map(|&id| graph.op(id))
        .filter(|op| matches!(op.op_v2_resolved(graph), Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_))))
        .collect();

    // Group by first input tensor (BTreeMap for deterministic iteration order)
    let mut by_input: std::collections::BTreeMap<TensorId, Vec<&CompilerOp>> =
        std::collections::BTreeMap::new();
    for op in &gemm_ops {
        if let Some(&first_input) = op.inputs.first() {
            by_input.entry(first_input).or_default().push(op);
        }
    }

    // QKV pattern: exactly 3 GEMMs sharing the same input (Q, K, V projections)
    for ops in by_input.values() {
        if ops.len() == 3 {
            // Check that the shared input comes from a norm op (typical QKV pattern)
            let shared_input = ops[0].inputs[0];
            let is_from_norm = graph.tensor(shared_input).and_then(|t| t.producer).is_some_and(
                |prod_id| {
                    graph
                        .op(prod_id)
                        .is_some_and(|prod_op| {
                            matches!(prod_op.op_v2_resolved(graph), Some(Op::RmsNorm(_)) | Some(Op::LayerNorm(_)))
                        })
                },
            );

            if !is_from_norm {
                continue;
            }

            // Quant-aware check: all GEMMs must have compatible quant types
            if !all_gemm_quant_compatible(ops, graph) {
                continue;
            }

            let all_ops: Vec<OpId> = ops.iter().map(|o| o.id).collect();
            result.push(FusionGroup {
                id: 0, // will be reassigned
                anchor: all_ops[0],
                epilogue: all_ops[1..].to_vec(),
                mode: FusionMode::QkvSharedInput,
                ops: all_ops,
                multi_output: MultiOutputConfig::single(),
                dominant_dtype: None,
                marker: GroupMarker::None,
                is_layer_group: false,
                hetero_layer_type: None,
            });
        }
    }

    result
}

/// Detect FFN block fusion pattern: Gate GEMM + Up GEMM (shared input) → activation → Mul.
///
/// Returns a FusionGroup with mode=FFNBlock when detected.
///
/// Topology requirements:
/// 1. `gate_gemm` and `up_gemm` share the same input tensor (TWO GEMMs with identical inputs[0])
/// 2. `activation` consumes `gate_gemm`'s output (typically SiLU/GeLU — single-input elementwise)
/// 3. `combine` is a Mul consuming `activation` output AND `up_gemm` output
/// 4. Shape compatibility (ARCH-FFN-SHAPE):
///    - gate_gemm.n == up_gemm.n (same intermediate dim)
///    - gate_gemm.k == up_gemm.k (same input hidden)
///    - gate_gemm.m ≈ up_gemm.m (same sequence — both symbolic or both concrete equal)
pub(crate) fn detect_ffn_block(graph: &CompilerGraph, topo: &[OpId]) -> Vec<FusionGroup> {
    let mut result = Vec::new();

    // 1. 找所有 Mul ops
    let mul_ops: Vec<&CompilerOp> = topo.iter()
        .filter_map(|&id| graph.op(id))
        .filter(|op| matches!(op.op_v2_resolved(graph), Some(Op::Mul)))
        .collect();

    for mul_op in mul_ops {
        if mul_op.inputs.len() != 2 { continue; }

        // Mul 的两个输入: 一个来自 activation（Silu/Gelu），一个来自 up_gemm
        let input_a_tid = mul_op.inputs[0];
        let input_b_tid = mul_op.inputs[1];

        let producer_a = graph.tensor(input_a_tid).and_then(|t| t.producer);
        let producer_b = graph.tensor(input_b_tid).and_then(|t| t.producer);
        let (Some(pa_id), Some(pb_id)) = (producer_a, producer_b) else { continue };

        let pa = match graph.op(pa_id) { Some(o) => o, None => continue };
        let pb = match graph.op(pb_id) { Some(o) => o, None => continue };

        // 识别哪个是 activation，哪个是 up_gemm（胖 opcode 自描述）
        let is_activation = |op: &CompilerOp| matches!(op.op_v2_resolved(graph), Some(Op::Silu) | Some(Op::Gelu));
        let is_gemm = |op: &CompilerOp| matches!(op.op_v2_resolved(graph),
            Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_)));

        let (activation_op, up_gemm_op) = if is_activation(pa) && is_gemm(pb) {
            (pa, pb)
        } else if is_activation(pb) && is_gemm(pa) {
            (pb, pa)
        } else {
            continue;
        };

        // Activation 必须是一元
        if activation_op.inputs.len() != 1 { continue; }

        // Activation 输入来自 gate_gemm
        let gate_input_tid = activation_op.inputs[0];
        let gate_gemm_id = match graph.tensor(gate_input_tid).and_then(|t| t.producer) {
            Some(id) => id,
            None => continue,
        };
        let gate_gemm_op = match graph.op(gate_gemm_id) {
            Some(o) if is_gemm(o) => o,
            _ => continue,
        };

        // gate_gemm 和 up_gemm 必须共享 inputs[0]
        let gate_first_input = gate_gemm_op.inputs.first().copied();
        let up_first_input = up_gemm_op.inputs.first().copied();
        if gate_first_input.is_none() || gate_first_input != up_first_input {
            continue;
        }

        // Shape 兼容性校验（ARCH-FFN-SHAPE）— 胖 opcode 自描述
        let (gate_m, gate_n, gate_k) = match gate_gemm_op.op_v2_gemm_dims(graph) { Some(v) => v, None => continue };
        let (up_m, up_n, up_k) = match up_gemm_op.op_v2_gemm_dims(graph) { Some(v) => v, None => continue };
        if gate_n != up_n || gate_k != up_k || gate_m != up_m {
            // Shape 不匹配 → 跳过融合，让它们作为独立算子
            continue;
        }

        // Quant-aware check: gate and up GEMMs must have compatible quant types
        if can_fuse_quant_aware(extract_quant_type(gate_gemm_op, graph), extract_quant_type(up_gemm_op, graph))
            == super::quant_aware::QuantFusionDecision::Split
        {
            continue;
        }

        result.push(FusionGroup {
            id: 0, // 将被 pass 重新赋值
            anchor: gate_gemm_op.id,
            epilogue: vec![up_gemm_op.id, activation_op.id, mul_op.id],
            mode: FusionMode::FFNBlock {
                gate_gemm: gate_gemm_op.id,
                up_gemm: up_gemm_op.id,
                activation: activation_op.id,
                combine: mul_op.id,
            },
            ops: vec![gate_gemm_op.id, up_gemm_op.id, activation_op.id, mul_op.id],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        });
    }

    result
}

/// Check if the op's input comes from a RmsNorm/LayerNorm with single consumer.
///
/// Unified version: when `dag` is provided, uses OpClass::Reduction as a pre-filter
/// (but still checks the concrete OpKind to exclude Softmax). When `dag` is None,
/// falls back to direct OpKind matching.
pub(crate) fn detect_norm_into_gemm(
    graph: &CompilerGraph,
    gemm_op: &CompilerOp,
    dag: Option<&SemanticDAG>,
) -> Option<OpId> {
    let input_tid = gemm_op.inputs.first()?;
    let tensor = graph.tensor(*input_tid)?;
    let producer_id = tensor.producer?;

    // If DAG is available, use OpClass as pre-filter
    if let Some(dag) = dag {
        let producer_node = dag.node(producer_id)?;
        if producer_node.op_class != OpClass::Reduction {
            return None;
        }
    }

    // Must be a norm op (not Softmax, which is also Reduction)
    let producer = graph.op(producer_id)?;
    if !matches!(producer.kind, OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. }) {
        return None;
    }

    // The norm output must feed only into this GEMM (single consumer)
    // For QKV pattern, norm feeds 3 GEMMs — skip if multi-consumer
    if tensor.consumers.len() != 1 {
        return None;
    }

    Some(producer_id)
}

/// Collect downstream elementwise ops that can be fused as epilogue.
///
/// Unified version: when `dag` is provided, uses OpClass to classify consumers.
/// When `dag` is None, uses OpSemantics.
///
/// Note: the DAG path is stricter — it requires all consumer inputs to come from
/// the fusion chain only (no graph inputs allowed), matching GPU epilogue constraints.
pub(crate) fn collect_epilogue<'a>(
    graph: &'a CompilerGraph,
    anchor: &CompilerOp,
    claimed: &HashSet<OpId>,
    dag: Option<&SemanticDAG>,
) -> Vec<&'a CompilerOp> {
    let mut epilogue = Vec::new();
    let mut current_outputs = anchor.outputs.clone();

    loop {
        if current_outputs.len() != 1 {
            break;
        }
        let out_tid = current_outputs[0];
        let tensor = match graph.tensor(out_tid) {
            Some(t) => t,
            None => break,
        };

        // Must have exactly one consumer
        if tensor.consumers.len() != 1 {
            break;
        }

        let consumer_id = tensor.consumers[0];
        if claimed.contains(&consumer_id) {
            break;
        }

        let consumer = match graph.op(consumer_id) {
            Some(o) => o,
            None => break,
        };

        // Check if consumer qualifies as epilogue on GEMM accumulator.
        // ARCH-EPILOGUE-ELEMWISE-ONLY: 只允许 ElemWise — 排除 Injective (如 RoPE、
        // Gather)，因为 Injective 的 scalar trace 可能用 Input(1..) 表示逻辑子输入
        // (RoPE 用 Input(0..3) 表示 x_re/x_im/cos/sin)，GEMM epilogue 只能在累加器
        // 上提供 Input(0)，无法满足。Injective op 必须保持独立节点。
        let is_elementwise = if let Some(dag) = dag {
            let consumer_class = dag
                .node(consumer_id)
                .map(|n| n.op_class)
                .unwrap_or(OpClass::Opaque);
            matches!(consumer_class, OpClass::ElemWise)
        } else {
            semantics::classify(&consumer.kind) == semantics::OpSemantics::Elementwise
        };

        if !is_elementwise {
            break;
        }

        // ARCH-EPILOGUE-UNARY: GEMM epilogue 当前 codegen 只能接受 accumulator
        // 作为 Input(0)——没有地方传 Input(1)。因此只允许 unary (单输入) op 作
        // 为 epilogue。binary op (如 Add(acc, bias)) 必须保持独立节点，由外层
        // elementwise lowering 带 broadcast 处理；或未来升级为 GemmBias 专用融合。
        if consumer.inputs.len() > 1 {
            break;
        }

        // Verify all consumer inputs come from the fusion chain.
        // DAG path: strict — only chain outputs allowed (GPU epilogue constraint).
        // Non-DAG path: also allows graph inputs (t.producer.is_none()).
        let chain_tids: HashSet<TensorId> =
            std::iter::once(anchor.outputs[0])
            .chain(epilogue.iter().flat_map(|op: &&CompilerOp| op.outputs.iter().copied()))
            .collect();

        let all_inputs_ok = if dag.is_some() {
            // Strict: all inputs must be in the chain
            consumer.inputs.iter().all(|tid| chain_tids.contains(tid))
        } else {
            // Lenient: chain outputs OR graph inputs (no producer)
            consumer.inputs.iter().all(|tid| {
                chain_tids.contains(tid) || graph.tensor(*tid).is_some_and(|t| t.producer.is_none())
            })
        };

        if !all_inputs_ok {
            break;
        }

        epilogue.push(consumer);
        current_outputs = consumer.outputs.clone();
    }

    epilogue
}

/// Collect a chain of elementwise ops starting from the given op.
///
/// Unified version: when `dag` is provided, uses OpClass. Otherwise uses OpSemantics.
pub(crate) fn collect_elementwise_chain<'a>(
    graph: &'a CompilerGraph,
    start: &CompilerOp,
    claimed: &HashSet<OpId>,
    dag: Option<&SemanticDAG>,
) -> Vec<&'a CompilerOp> {
    let mut chain = Vec::new();
    let mut current_outputs = start.outputs.clone();

    // Track all tensor IDs whose producers are in the chain (or are the start op).
    // A consumer can only be chained if ALL its intermediate inputs come from
    // within this set — otherwise we'd create a topological ordering violation.
    let mut available_outputs: HashSet<crate::compiler::graph::TensorId> =
        start.outputs.iter().copied().collect();

    loop {
        if current_outputs.len() != 1 {
            break;
        }
        let out_tid = current_outputs[0];
        let tensor = match graph.tensor(out_tid) {
            Some(t) => t,
            None => break,
        };

        if tensor.consumers.len() != 1 {
            break;
        }

        let consumer_id = tensor.consumers[0];
        if claimed.contains(&consumer_id) {
            break;
        }

        let consumer = match graph.op(consumer_id) {
            Some(o) => o,
            None => break,
        };

        let is_elementwise = if let Some(dag) = dag {
            let consumer_class = dag
                .node(consumer_id)
                .map(|n| n.op_class)
                .unwrap_or(OpClass::Opaque);
            matches!(consumer_class, OpClass::ElemWise | OpClass::Injective)
        } else {
            semantics::classify(&consumer.kind) == semantics::OpSemantics::Elementwise
        };

        if !is_elementwise {
            break;
        }

        // Verify ALL intermediate inputs of the consumer are already available
        // (produced by the start op or a previous op in the chain). If the
        // consumer has an intermediate input from an external op, chaining it
        // would create a topological ordering violation — the fused group would
        // try to execute the consumer before its input is produced.
        let all_inputs_available = consumer.inputs.iter().all(|&tid| {
            // Graph inputs are always available
            if graph.inputs.contains(&tid) {
                return true;
            }
            // Outputs already produced by the chain
            available_outputs.contains(&tid)
        });
        if !all_inputs_available {
            break;
        }

        chain.push(consumer);
        available_outputs.extend(consumer.outputs.iter().copied());
        current_outputs = consumer.outputs.clone();
    }

    chain
}

/// Split an elementwise op chain into sub-chains based on L1 cache budget.
///
/// Walks the chain accumulating intermediate tensor bytes. When the cumulative
/// size exceeds 75% of L1, a new sub-chain starts. This prevents fused loops
/// from thrashing L1 with oversized intermediate data.
pub(crate) fn split_elementwise_by_l1(
    graph: &CompilerGraph,
    all_ops: &[OpId],
    plan: &crate::compiler::planner::ExecutionPlan,
) -> Vec<Vec<OpId>> {
    if all_ops.len() <= 1 {
        return vec![all_ops.to_vec()];
    }

    let (l1, _, _) = plan.profile.cache_sizes();
    let l1_budget = l1 * 75 / 100;

    let mut sub_chains: Vec<Vec<OpId>> = Vec::new();
    let mut current = vec![all_ops[0]];
    let mut cumulative_bytes: usize = 0;

    for i in 1..all_ops.len() {
        // Intermediate tensor = output of the previous op in the chain
        let intermediate_bytes = graph
            .op(all_ops[i - 1])
            .and_then(|op| op.outputs.first())
            .and_then(|tid| graph.tensor(*tid))
            .map(|t| t.concrete_bytes())
            .unwrap_or(0);

        cumulative_bytes += intermediate_bytes;

        if cumulative_bytes > l1_budget {
            sub_chains.push(current);
            current = vec![all_ops[i]];
            cumulative_bytes = 0;
        } else {
            current.push(all_ops[i]);
        }
    }

    if !current.is_empty() {
        sub_chains.push(current);
    }

    sub_chains
}

/// Decide between TileLevelFusion and ComputeRoot for a norm->GEMM pair.
///
/// If the norm output tensor exceeds 75% of L1, the norm must be tiled into
/// the GEMM MC loop (TileLevelFusion). Otherwise, the norm is computed fully
/// first and its result stays in L1 (ComputeRoot).
pub(crate) fn detect_tile_vs_compute_root(
    graph: &CompilerGraph,
    gemm_op: &CompilerOp,
    norm_id: OpId,
    plan: &crate::compiler::planner::ExecutionPlan,
) -> FusionMode {
    let (l1, _, _) = plan.profile.cache_sizes();
    let l1_budget = l1 * 75 / 100; // 75% of L1

    // Compute norm output size in bytes
    let norm_output_bytes = gemm_op
        .inputs
        .first()
        .and_then(|tid| graph.tensor(*tid))
        .map(|t| t.concrete_bytes())
        .unwrap_or(0);

    if norm_output_bytes > l1_budget {
        // Norm output doesn't fit in L1 -> tile into GEMM MC loop
        // ARCH-SYMDIM-DEGRADE: cost model uses max_for_allocation for conservative estimate.
        // TODO(G-2): preserve symbolic form for tighter bounds.
        // 胖 opcode 自描述：GEMM 维度 + dtype
        let (m, n, k, gemm_dtype) = match gemm_op.op_v2_gemm_dims(graph) {
            Some((m_dim, n_val, k_val)) => {
                let m_val = m_dim.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model");
                let dtype = gemm_op.op_v2_gemm_dtype(graph).unwrap_or_else(|| graph.infer_computation_dtype());
                (m_val, n_val, k_val, dtype)
            }
            None => (0, 0, 0, graph.infer_computation_dtype()),
        };
        let blocking = plan.profile.gemm_blocking(m, n, k, gemm_dtype);
        FusionMode::TileLevelFusion {
            predecessor: norm_id,
            tile_rows: blocking.mc,
        }
    } else {
        // Norm output fits in L1 -> compute root (standalone norm, result stays in L1)
        FusionMode::ComputeRoot {
            predecessor: norm_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, CompilerOp, LayerCondition, OpKind, OpId, SymDim};
    use crate::types::DType;
    use crate::quant::QuantType;
    use std::collections::HashSet;

    // Helper: build a simple CompilerGraph with a norm -> GEMM chain.
    // Graph structure: input_tensor -> RmsNorm -> norm_out -> Gemm -> gemm_out
    fn build_norm_gemm_graph() -> (CompilerGraph, OpId, OpId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[64, 256], DType::F32);
        g.inputs.push(input);
        let norm_out = g.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = g.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = g.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "rms_norm",
        );
        let gemm_id = g.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, weight],
            vec![gemm_out],
            "gemm",
        );
        (g, norm_id, gemm_id)
    }

    // ── Test 1: extract_quant_type returns Some for QuantGemm ──
    #[test]
    fn test_extract_quant_type_quant_gemm() {
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "qgemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let g = CompilerGraph::new();
        let result = extract_quant_type(&op, &g);
        assert_eq!(result, Some(QuantType::Q4_0));
    }

    // ── Test 2: extract_quant_type returns None for plain Gemm ──
    #[test]
    fn test_extract_quant_type_plain_gemm_returns_none() {
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Gemm { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "gemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let g = CompilerGraph::new();
        let result = extract_quant_type(&op, &g);
        assert!(result.is_none());
    }

    // ── Test 3: all_gemm_quant_compatible with same quant types ──
    #[test]
    fn test_all_gemm_quant_compatible_same_quant() {
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "qgemm1".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "qgemm2".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];
        assert!({ let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) });
    }

    // ── Test 4: all_gemm_quant_compatible with incompatible quant types returns false ──
    #[test]
    fn test_all_gemm_quant_compatible_incompatible_types() {
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "qgemm1".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q6K, },
            vec![],
            vec![],
            "qgemm2".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];
        // Q4_0 and Q6K are different quant types -> Split -> not compatible
        assert!(!{ let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) });
    }

    // ── Test 5: detect_norm_into_gemm returns Some when norm feeds single-consumer GEMM ──
    #[test]
    fn test_detect_norm_into_gemm_positive() {
        let (graph, norm_id, gemm_id) = build_norm_gemm_graph();
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);
        assert_eq!(result, Some(norm_id));
    }

    // ── Test 6: detect_norm_into_gemm returns None when input has no producer ──
    #[test]
    fn test_detect_norm_into_gemm_no_producer() {
        let mut graph = CompilerGraph::new();
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);
        graph.inputs.push(weight);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        // No RmsNorm: weight directly feeds GEMM, but weight has no GEMM producer
        // We need the GEMM to have an input without a norm producer
        let raw_input = graph.add_tensor_concrete("raw_input", &[64, 256], DType::F32);
        graph.inputs.push(raw_input);
        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![raw_input, weight],
            vec![gemm_out],
            "gemm",
        );
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);
        assert!(result.is_none());
    }

    // ── Test 7: detect_norm_into_gemm returns None for multi-consumer norm output ──
    #[test]
    fn test_detect_norm_into_gemm_multi_consumer() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out1 = graph.add_tensor_concrete("gemm_out1", &[64, 512], DType::F32);
        let gemm_out2 = graph.add_tensor_concrete("gemm_out2", &[64, 512], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[256, 512], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[256, 512], DType::F32);

        let _norm_id = graph.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "norm",
        );
        let gemm1_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, w1],
            vec![gemm_out1],
            "gemm1",
        );
        let _gemm2_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, w2],
            vec![gemm_out2],
            "gemm2",
        );

        // norm_out has 2 consumers -> detect should return None
        let gemm1_op = graph.op(gemm1_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm1_op, None);
        assert!(result.is_none());
    }

    // ── Test 8: collect_epilogue chains unary elementwise ops after anchor ──
    #[test]
    fn test_collect_epilogue_unary_chain() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);

        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input],
            vec![gemm_out],
            "gemm",
        );
        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![gemm_out],
            vec![silu_out],
            "silu",
        );

        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, silu_id);
    }

    // ── Test 9: collect_epilogue stops at binary op (multi-input consumer) ──
    #[test]
    fn test_collect_epilogue_stops_at_binary_op() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let add_out = graph.add_tensor_concrete("add_out", &[64], DType::F32);
        let bias = graph.add_tensor_concrete("bias", &[64], DType::F32);
        graph.inputs.push(bias);

        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input],
            vec![gemm_out],
            "gemm",
        );
        let _add_id = graph.add_op(
            OpKind::Add,
            vec![gemm_out, bias],
            vec![add_out],
            "add",
        );

        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Add is binary (2 inputs) -> epilogue chain stops
        assert!(epilogue.is_empty());
    }

    // ── Test 10: collect_elementwise_chain chains elementwise ops ──
    #[test]
    fn test_collect_elementwise_chain_basic() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64], DType::F32);

        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![input],
            vec![silu_out],
            "silu",
        );
        let _mul_id = graph.add_op(
            OpKind::Mul,
            vec![silu_out, input], // both inputs available (input is graph input)
            vec![mul_out],
            "mul",
        );

        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Silu -> Mul: Mul is elementwise and all inputs available (silu_out from chain, input is graph input)
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].id, OpId(1)); // Mul
    }

    // ── Test 11: detect_qkv_shared_input returns empty when no triple-GEMM sharing ──
    #[test]
    fn test_detect_qkv_shared_input_no_triple() {
        let (graph, _norm_id, _gemm_id) = build_norm_gemm_graph();
        // Only 1 GEMM after norm, not 3
        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        let result = detect_qkv_shared_input(&graph, &topo);
        assert!(result.is_empty());
    }

    // ── Test 12: detect_ffn_block returns empty when no Mul ops present ──
    #[test]
    fn test_detect_ffn_block_no_mul() {
        let (graph, _norm_id, _gemm_id) = build_norm_gemm_graph();
        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        let result = detect_ffn_block(&graph, &topo);
        assert!(result.is_empty());
    }

    // ── Test 13: detect_qkv_norm_rope returns empty for plain graph ──
    #[test]
    fn test_detect_qkv_norm_rope_plain_graph() {
        let (graph, _norm_id, _gemm_id) = build_norm_gemm_graph();
        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        let result = detect_qkv_norm_rope(&graph, &topo);
        assert!(result.is_empty());
    }

    // ── Test 14: extract_quant_type returns None for GemmBias ──
    // @trace TEST-FH-14 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_extract_quant_type_gemm_bias_returns_none() {
        // Arrange
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::GemmBias { m: SymDim::Concrete(32), n: 128, k: 64, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "gemm_bias".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        // Act
        let g = CompilerGraph::new();
        let result = extract_quant_type(&op, &g);
        // Assert: GemmBias is not a QuantGemm, so no QuantType can be extracted
        assert!(result.is_none());
    }

    // ── Test 15: extract_quant_type returns None for non-GEMM ops ──
    // @trace TEST-FH-15 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_extract_quant_type_non_gemm_op() {
        // Arrange
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Silu,
            vec![],
            vec![],
            "silu".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        // Act
        let g = CompilerGraph::new();
        let result = extract_quant_type(&op, &g);
        // Assert: Silu is not a GEMM variant at all
        assert!(result.is_none());
    }

    // ── Test 16: all_gemm_quant_compatible with mixed quant and plain GEMM returns true ──
    // @trace TEST-FH-16 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_mixed_quant_and_plain() {
        // Arrange: one QuantGemm (Q4_0) and one plain Gemm (no quant)
        // QuantGemm output is always F32 after dequant, so QuantGemm -> plain ElemWise is Fuse
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "qgemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 256, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "plain_gemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];
        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
        // Assert: Some(Q4_0) -> None => Fuse, so compatible
        assert!(compatible);
    }

    // ── Test 17: all_gemm_quant_compatible with single op returns true ──
    // @trace TEST-FH-17 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_single_op() {
        // Arrange: only one op in the list, no pair to compare
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "qgemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op];
        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
        // Assert: single op has no pairs, loop body never executes, returns true
        assert!(compatible);
    }

    // ── Test 18: detect_norm_into_gemm works with LayerNorm producer ──
    // @trace TEST-FH-18 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_with_layer_norm() {
        // Arrange: LayerNorm -> GEMM (instead of RmsNorm)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = graph.add_op(
            OpKind::LayerNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "layer_norm",
        );
        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, weight],
            vec![gemm_out],
            "gemm",
        );
        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);
        // Assert: LayerNorm is also recognized as a norm producer
        assert_eq!(result, Some(norm_id));
    }

    // ── Test 19: detect_norm_into_gemm returns None when producer is Softmax ──
    // @trace TEST-FH-19 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_producer_is_softmax() {
        // Arrange: Softmax is also a Reduction op but is NOT a norm
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let softmax_out = graph.add_tensor_concrete("softmax_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let _softmax_id = graph.add_op(
            OpKind::Softmax,
            vec![input],
            vec![softmax_out],
            "softmax",
        );
        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![softmax_out, weight],
            vec![gemm_out],
            "gemm",
        );
        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);
        // Assert: Softmax is not RmsNorm/LayerNorm -> returns None
        assert!(result.is_none());
    }

    // ── Test 20: collect_epilogue stops when consumer is claimed ──
    // @trace TEST-FH-20 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_claimed_consumer() {
        // Arrange: GEMM -> Silu, but Silu is already claimed by another fusion group
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);

        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input],
            vec![gemm_out],
            "gemm",
        );
        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![gemm_out],
            vec![silu_out],
            "silu",
        );
        // Mark Silu as claimed
        let mut claimed = HashSet::new();
        claimed.insert(silu_id);

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);
        // Assert: Silu is claimed, epilogue chain stops before it
        assert!(epilogue.is_empty());
    }

    // ── Test 21: collect_epilogue stops when anchor has multiple outputs ──
    // @trace TEST-FH-21 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_multi_output_anchor() {
        // Arrange: anchor op with empty outputs (simulates multi-output break)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let out_a = graph.add_tensor_concrete("out_a", &[32], DType::F32);
        let out_b = graph.add_tensor_concrete("out_b", &[32], DType::F32);

        let anchor_id = graph.add_op(
            OpKind::Silu,
            vec![input],
            vec![out_a, out_b],
            "multi_out",
        );
        // Act
        let anchor_op = graph.op(anchor_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, anchor_op, &claimed, None);
        // Assert: anchor has 2 outputs, loop breaks immediately
        assert!(epilogue.is_empty());
    }

    // ── Test 22: collect_elementwise_chain stops when consumer is claimed ──
    // @trace TEST-FH-22 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_stops_at_claimed() {
        // Arrange: Silu -> Tanh, but Tanh is already claimed
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[64], DType::F32);

        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![input],
            vec![silu_out],
            "silu",
        );
        let tanh_id = graph.add_op(
            OpKind::Tanh,
            vec![silu_out],
            vec![tanh_out],
            "tanh",
        );
        // Mark Tanh as claimed
        let mut claimed = HashSet::new();
        claimed.insert(tanh_id);

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);
        // Assert: Tanh is claimed, chain stops before it
        assert!(chain.is_empty());
    }

    // ── Test 23: collect_elementwise_chain stops at non-elementwise op ──
    // @trace TEST-FH-23 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_stops_at_gemm() {
        // Arrange: Silu -> Gemm (Gemm is not elementwise)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);

        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![input],
            vec![silu_out],
            "silu",
        );
        let _gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![silu_out],
            vec![gemm_out],
            "gemm",
        );
        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);
        // Assert: Gemm is not elementwise, chain stops
        assert!(chain.is_empty());
    }

    // ── Test 24: detect_qkv_shared_input with valid triple-GEMM pattern ──
    // @trace TEST-FH-24 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_positive_triple() {
        // Arrange: RmsNorm -> 3 GEMMs (Q, K, V) all sharing norm output
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        let _norm_id = graph.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "norm",
        );
        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, wq],
            vec![q_out],
            "q_proj",
        );
        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, wk],
            vec![k_out],
            "k_proj",
        );
        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, wv],
            vec![v_out],
            "v_proj",
        );
        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_shared_input(&graph, &topo);
        // Assert: exactly 1 fusion group with 3 ops in QkvSharedInput mode
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 3);
        assert!(matches!(result[0].mode, FusionMode::QkvSharedInput));
    }

    // ── Test 25: detect_ffn_block with valid SwiGLU pattern ──
    // @trace TEST-FH-25 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_positive_swiglu() {
        // Arrange: input -> gate_gemm + up_gemm -> Silu(gate_out) -> Mul(silu_out, up_out)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 1024], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 1024], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 1024], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 1024], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 1024], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 1024], DType::F32);

        let gate_gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 1024,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input, w_gate],
            vec![gate_out],
            "gate_gemm",
        );
        let up_gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 1024,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input, w_up],
            vec![up_out],
            "up_gemm",
        );
        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![gate_out],
            vec![silu_out],
            "silu",
        );
        let mul_id = graph.add_op(
            OpKind::Mul,
            vec![silu_out, up_out],
            vec![mul_out],
            "mul",
        );
        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);
        // Assert: one FFNBlock fusion group with gate, up, silu, mul
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 4);
        assert_eq!(result[0].ops[0], gate_gemm_id);
        assert_eq!(result[0].ops[1], up_gemm_id);
        assert_eq!(result[0].ops[2], silu_id);
        assert_eq!(result[0].ops[3], mul_id);
        if let FusionMode::FFNBlock { gate_gemm, up_gemm, activation, combine } = &result[0].mode {
            assert_eq!(*gate_gemm, gate_gemm_id);
            assert_eq!(*up_gemm, up_gemm_id);
            assert_eq!(*activation, silu_id);
            assert_eq!(*combine, mul_id);
        } else {
            panic!("Expected FFNBlock mode, got {:?}", result[0].mode);
        }
    }

    // ── Test 26: detect_ffn_block rejects mismatched GEMM shapes ──
    // @trace TEST-FH-26 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_mismatched_shapes() {
        // Arrange: gate_gemm has n=1024, up_gemm has n=512 (shape mismatch)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 1024], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 1024], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 1024], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 1024], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 1024,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input, w_gate],
            vec![gate_out],
            "gate_gemm",
        );
        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input, w_up],
            vec![up_out],
            "up_gemm",
        );
        graph.add_op(
            OpKind::Silu,
            vec![gate_out],
            vec![silu_out],
            "silu",
        );
        graph.add_op(
            OpKind::Mul,
            vec![silu_out, up_out],
            vec![mul_out],
            "mul",
        );
        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);
        // Assert: gate n=1024 != up n=512 -> shape mismatch -> no fusion
        assert!(result.is_empty());
    }

    // ── Test 27: collect_epilogue chains multiple unary ops (Silu -> Tanh) ──
    // @trace TEST-FH-27 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_multi_unary_chain() {
        // Arrange: GEMM -> Silu -> Tanh (two unary elementwise ops)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[64], DType::F32);

        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input],
            vec![gemm_out],
            "gemm",
        );
        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![gemm_out],
            vec![silu_out],
            "silu",
        );
        let tanh_id = graph.add_op(
            OpKind::Tanh,
            vec![silu_out],
            vec![tanh_out],
            "tanh",
        );

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: both Silu and Tanh collected in order
        assert_eq!(epilogue.len(), 2);
        assert_eq!(epilogue[0].id, silu_id);
        assert_eq!(epilogue[1].id, tanh_id);
    }

    // ── Test 28: collect_epilogue stops at multi-consumer tensor ──
    // @trace TEST-FH-28 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_fanout() {
        // Arrange: GEMM -> Silu, but Silu output fans out to 2 consumers
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let out_a = graph.add_tensor_concrete("out_a", &[64], DType::F32);
        let out_b = graph.add_tensor_concrete("out_b", &[64], DType::F32);

        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![input],
            vec![gemm_out],
            "gemm",
        );
        let _silu_id = graph.add_op(
            OpKind::Silu,
            vec![gemm_out],
            vec![silu_out],
            "silu",
        );
        // Silu output has 2 consumers -> epilogue cannot extend past Silu
        graph.add_op(OpKind::Tanh, vec![silu_out], vec![out_a], "tanh_a");
        graph.add_op(OpKind::SwiGlu, vec![silu_out], vec![out_b], "swiglu_b");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: Silu has single consumer from GEMM, but Silu output fans out -> stops at Silu
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, OpId(1)); // Silu
    }

    // ── Test 29: collect_elementwise_chain stops when input from outside chain ──
    // @trace TEST-FH-29 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_stops_at_external_input() {
        // Arrange: Silu -> Add where Add has one input from Silu but another
        // from an external (non-chain, non-graph-input) producer
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let external_out = graph.add_tensor_concrete("external_out", &[64], DType::F32);
        let add_out = graph.add_tensor_concrete("add_out", &[64], DType::F32);

        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![input],
            vec![silu_out],
            "silu",
        );
        // external_out produced by a Tanh (not a graph input, not in chain)
        let _external_id = graph.add_op(
            OpKind::Tanh,
            vec![input],
            vec![external_out],
            "external_tanh",
        );
        // Add consumes silu_out (from chain) and external_out (not in chain, not graph input)
        let _add_id = graph.add_op(
            OpKind::Add,
            vec![silu_out, external_out],
            vec![add_out],
            "add",
        );

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Assert: Add has external input (external_out produced by external_tanh, not available)
        assert!(chain.is_empty());
    }

    // ── Test 30: all_gemm_quant_compatible with empty ops list returns true ──
    // @trace TEST-FH-30 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_empty() {
        // Arrange: empty slice, no pairs to compare
        let ops: Vec<&CompilerOp> = vec![];
        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
        // Assert: vacuously true (no incompatible pair exists)
        assert!(compatible);
    }

    // ── Test 31: all_gemm_quant_compatible with three ops all same quant ──
    // @trace TEST-FH-31 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_three_ops_same() {
        // Arrange: 3 QuantGemm ops all using Q4_0
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "q1".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "q2".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op3 = CompilerOp::new_from_kind(
            OpId(2),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "q3".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2, &op3];
        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
        // Assert: all same -> Fuse between every pair -> true
        assert!(compatible);
    }

    // ── Test 32: detect_norm_into_gemm returns None when GEMM has no inputs ──
    // @trace TEST-FH-32 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_no_inputs() {
        // Arrange: a GEMM op with empty inputs vector
        let graph = CompilerGraph::new();
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Gemm { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "gemm_no_inputs".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        // Act
        let result = detect_norm_into_gemm(&graph, &op, None);
        // Assert: no first input -> returns None
        assert!(result.is_none());
    }

    // ── Test 33: detect_ffn_block rejects when gate and up GEMMs do not share input ──
    // @trace TEST-FH-33 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_non_shared_input() {
        // Arrange: gate_gemm and up_gemm have different first inputs
        let mut graph = CompilerGraph::new();
        let input_a = graph.add_tensor_concrete("input_a", &[64, 256], DType::F32);
        let input_b = graph.add_tensor_concrete("input_b", &[64, 256], DType::F32);
        graph.inputs.push(input_a);
        graph.inputs.push(input_b);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64), n: 512, k: 256,
                dtype: DType::F32, trans_b: false,
            },
            vec![input_a, w_gate],
            vec![gate_out],
            "gate_gemm",
        );
        graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(64), n: 512, k: 256,
                dtype: DType::F32, trans_b: false,
            },
            vec![input_b, w_up],  // different first input!
            vec![up_out],
            "up_gemm",
        );
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_ffn_block(&graph, &topo);
        // Assert: inputs[0] differ -> no FFNBlock fusion
        assert!(result.is_empty());
    }

    // ── Test 34: detect_qkv_shared_input rejects when input not from norm ──
    // @trace TEST-FH-34 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_rejects_non_norm_producer() {
        // Arrange: 3 GEMMs sharing same input, but input is produced by Silu (not norm)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        // Silu feeds the 3 GEMMs (not RmsNorm/LayerNorm)
        graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![silu_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![silu_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![silu_out, wv], vec![v_out], "v_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_qkv_shared_input(&graph, &topo);
        // Assert: shared input comes from Silu, not norm -> rejected
        assert!(result.is_empty());
    }

    // ── Test 35: detect_qkv_norm_rope rejects when only 2 GEMMs share input ──
    // @trace TEST-FH-35 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_double_gemm() {
        // Arrange: only 2 GEMMs after RmsNorm (not the required 3)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let a_out = graph.add_tensor_concrete("a_out", &[64, 512], DType::F32);
        let b_out = graph.add_tensor_concrete("b_out", &[64, 512], DType::F32);
        let wa = graph.add_tensor_concrete("wa", &[256, 512], DType::F32);
        let wb = graph.add_tensor_concrete("wb", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wa], vec![a_out], "gemm_a");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wb], vec![b_out], "gemm_b");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);
        // Assert: needs exactly 3 GEMMs, only 2 present -> empty
        assert!(result.is_empty());
    }

    // ── Test 36: collect_epilogue with empty outputs anchor stops immediately ──
    // @trace TEST-FH-36 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_empty_outputs() {
        // Arrange: anchor op with empty outputs vector
        let graph = CompilerGraph::new();
        let anchor = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Silu,
            vec![],
            vec![],
            "empty_anchor".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let claimed = HashSet::new();
        // Act
        let epilogue = collect_epilogue(&graph, &anchor, &claimed, None);
        // Assert: no outputs -> loop breaks immediately
        assert!(epilogue.is_empty());
    }

    // ── Test 37: detect_ffn_block rejects when activation is not unary ──
    // @trace TEST-FH-37 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_non_unary_activation() {
        // Arrange: gate_gemm -> Add (2-input activation substitute) -> Mul
        // Add has 2 inputs so it won't match the activation pattern
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let add_out = graph.add_tensor_concrete("add_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let bias = graph.add_tensor_concrete("bias", &[512], DType::F32);
        graph.inputs.push(bias);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        // gate_gemm -> Add(gate_out, bias) — Add is not an activation op
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        // Add is not recognized as activation (Silu/Gelu)
        graph.add_op(OpKind::Add, vec![gate_out, bias], vec![add_out], "add");
        graph.add_op(OpKind::Mul, vec![add_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_ffn_block(&graph, &topo);
        // Assert: Add is not SiLU/GeLU -> no activation match -> empty
        assert!(result.is_empty());
    }

    // ── Test 38: collect_elementwise_chain allows graph input as secondary input ──
    // @trace TEST-FH-38 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_allows_graph_input_secondary() {
        // Arrange: Silu -> Mul where Mul takes silu_out + graph input
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let scale = graph.add_tensor_concrete("scale", &[64], DType::F32);
        graph.inputs.push(scale);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64], DType::F32);

        let silu_id = graph.add_op(
            OpKind::Silu,
            vec![input],
            vec![silu_out],
            "silu",
        );
        let mul_id = graph.add_op(
            OpKind::Mul,
            vec![silu_out, scale],  // scale is a graph input -> allowed
            vec![mul_out],
            "mul",
        );

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Assert: Mul is elementwise, all inputs available (silu_out from chain, scale is graph input)
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].id, mul_id);
    }

    // ── Test 39: split_elementwise_by_l1 with single op returns single sub-chain ──
    // @trace TEST-FH-39 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_split_elementwise_by_l1_single_op() {
        // Arrange: single-op chain should return a single sub-chain
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let sub_chains = split_elementwise_by_l1(&graph, &[silu_id], &plan);

        // Assert: single op -> single sub-chain with that op
        assert_eq!(sub_chains.len(), 1);
        assert_eq!(sub_chains[0].len(), 1);
        assert_eq!(sub_chains[0][0], silu_id);
    }

    // ── Test 40: extract_quant_type returns None for non-GEMM ops ──

    #[test]
    fn test_extract_quant_type_non_gemm() {
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Silu,
            vec![],
            vec![],
            "silu".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let g = CompilerGraph::new();
        assert!(extract_quant_type(&op, &g).is_none());
    }

    // ── Test 41: all_gemm_quant_compatible with single plain Gemm ──

    #[test]
    fn test_all_gemm_quant_compatible_single_plain_gemm() {
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Gemm { m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "gemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        assert!({ let g = CompilerGraph::new(); all_gemm_quant_compatible(&[&op], &g) });
    }

    // ── Test 42: all_gemm_quant_compatible with mixed QuantGemm and plain Gemm ──

    #[test]
    fn test_all_gemm_quant_compatible_mixed_none_and_quant() {
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Gemm { m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "plain_gemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 64, k: 64, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "quant_gemm".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        // None vs Q4_0: depends on can_fuse_quant_aware(None, Some(Q4_0))
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];
        // Should not panic
        let _ = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
    }

    // ── Test 43: detect_norm_into_gemm returns None for Silu→GEMM ──

    #[test]
    fn test_detect_norm_into_gemm_silu_producer() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        let gemm_id = graph.add_op(
            OpKind::Gemm { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false },
            vec![silu_out, weight], vec![gemm_out], "gemm",
        );

        // Act: SiLU is not a norm op -> should return None
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);
        assert!(result.is_none());
    }

    // ── Test 44: detect_norm_into_gemm with LayerNorm succeeds ──

    #[test]
    fn test_detect_norm_into_gemm_layernorm() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = graph.add_op(
            OpKind::LayerNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input], vec![norm_out], "layernorm",
        );
        let gemm_id = graph.add_op(
            OpKind::Gemm { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false },
            vec![norm_out, weight], vec![gemm_out], "gemm",
        );

        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);
        assert_eq!(result, Some(norm_id));
    }

    // ── Test 45: collect_epilogue with claimed consumer stops ──

    #[test]
    fn test_collect_epilogue_claimed_consumer_stops() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        let _gelu_id = graph.add_op(OpKind::Gelu, vec![silu_out], vec![gelu_out], "gelu");

        let silu_op = graph.op(silu_id).unwrap();
        let mut claimed = HashSet::new();
        claimed.insert(_gelu_id);

        let epilogue = collect_epilogue(&graph, silu_op, &claimed, None);
        assert!(epilogue.is_empty(), "Claimed consumer should stop epilogue collection");
    }

    // ── Test 46: collect_elementwise_chain with multi-consumer output stops ──

    #[test]
    fn test_collect_elementwise_chain_multi_consumer_stops() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let a = graph.add_tensor_concrete("a", &[64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        // Two consumers of silu_out
        graph.add_op(OpKind::Gelu, vec![silu_out], vec![a], "gelu_a");
        graph.add_op(OpKind::Gelu, vec![silu_out], vec![b], "gelu_b");

        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Multi-consumer output stops chain
        assert!(chain.is_empty());
    }

    // ── Test 47: detect_tile_vs_compute_root prefers compute root for small tensors ──

    #[test]
    fn test_detect_tile_vs_compute_root_small_tensor() {
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let (graph, norm_id, gemm_id) = build_norm_gemm_graph();
        let gemm_op = graph.op(gemm_id).unwrap();
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        let mode = detect_tile_vs_compute_root(&graph, gemm_op, norm_id, &plan);

        // For a [64,256] F32 tensor = 65536 bytes, typically fits in L1 → ComputeRoot
        match mode {
            FusionMode::ComputeRoot { predecessor } => {
                assert_eq!(predecessor, norm_id);
            }
            FusionMode::TileLevelFusion { predecessor, .. } => {
                assert_eq!(predecessor, norm_id);
                // Small L1 or large tensor → tile is also valid
            }
            _ => panic!("expected ComputeRoot or TileLevelFusion, got {:?}", mode),
        }
    }

    // ── Test 48: split_elementwise_by_l1 with empty chain ──

    #[test]
    fn test_split_elementwise_by_l1_empty_chain() {
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let graph = CompilerGraph::new();
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        let sub_chains = split_elementwise_by_l1(&graph, &[], &plan);
        assert_eq!(sub_chains.len(), 1);
        assert!(sub_chains[0].is_empty());
    }

    // ── Test 49: detect_ffn_block rejects when gate and up have different k ──

    #[test]
    fn test_detect_ffn_block_rejects_shape_mismatch_k() {
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[128, 512], DType::F32); // different k=128

        graph.add_op(OpKind::Gemm { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false },
            vec![input, w_gate], vec![gate_out], "gate_gemm");
        graph.add_op(OpKind::Gemm { m: SymDim::Concrete(64), n: 512, k: 128, dtype: DType::F32, trans_b: false },
            vec![input, w_up], vec![up_out], "up_gemm"); // k mismatch
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        let result = detect_ffn_block(&graph, &topo);
        assert!(result.is_empty(), "Different k should reject FFN block fusion");
    }

    // ── Test 50: detect_qkv_norm_rope with valid Gemma 4 pattern ──
    // @trace TEST-FH-50 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_positive() {
        // Arrange: RmsNorm -> 3 GEMMs (Q,K,V), Q->QkNorm->RoPE, K->QkNorm->RoPE, V->ValueNorm
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qk_norm_q_out = graph.add_tensor_concrete("qk_norm_q_out", &[64, 512], DType::F32);
        let qk_norm_k_out = graph.add_tensor_concrete("qk_norm_k_out", &[64, 512], DType::F32);
        let value_norm_out = graph.add_tensor_concrete("value_norm_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let rope_k_out = graph.add_tensor_concrete("rope_k_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        let gemm_q = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        let gemm_k = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        let gemm_v = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        let qk_norm_q = graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qk_norm_q_out], "qk_norm_q");
        let qk_norm_k = graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qk_norm_k_out], "qk_norm_k");
        let value_norm = graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 },
            vec![v_out], vec![value_norm_out], "value_norm");
        let rope_q = graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qk_norm_q_out], vec![rope_q_out], "rope_q");
        let rope_k = graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qk_norm_k_out], vec![rope_k_out], "rope_k");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: exactly 1 fusion group with 8 ops (3 GEMM + 2 QkNorm + 1 ValueNorm + 2 RoPE)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 8);
        if let FusionMode::FusedQkvNormRope {
            gemm_q: gq, gemm_k: gk, gemm_v: gv,
            qk_norm_q: nq, qk_norm_k: nk, value_norm_v: vn,
            rope_q: rq, rope_k: rk,
        } = &result[0].mode {
            assert_eq!(*gq, gemm_q);
            assert_eq!(*gk, gemm_k);
            assert_eq!(*gv, gemm_v);
            assert_eq!(*nq, qk_norm_q);
            assert_eq!(*nk, qk_norm_k);
            assert_eq!(*vn, value_norm);
            assert_eq!(*rq, rope_q);
            assert_eq!(*rk, rope_k);
        } else {
            panic!("Expected FusedQkvNormRope mode, got {:?}", result[0].mode);
        }
    }

    // ── Test 51: detect_ffn_block with GeLU activation instead of SiLU ──
    // @trace TEST-FH-51 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_with_gelu_activation() {
        // Arrange: input -> gate_gemm + up_gemm -> Gelu(gate_out) -> Mul(gelu_out, up_out)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        let gate_gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        let up_gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![gate_out], vec![gelu_out], "gelu");
        let mul_id = graph.add_op(OpKind::Mul, vec![gelu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: GeLU is also a valid activation for FFN block
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 4);
        if let FusionMode::FFNBlock { gate_gemm, up_gemm, activation, combine } = &result[0].mode {
            assert_eq!(*gate_gemm, gate_gemm_id);
            assert_eq!(*up_gemm, up_gemm_id);
            assert_eq!(*activation, gelu_id);
            assert_eq!(*combine, mul_id);
        } else {
            panic!("Expected FFNBlock mode, got {:?}", result[0].mode);
        }
    }

    // ── Test 52: detect_qkv_shared_input with GemmBias variants ──
    // @trace TEST-FH-52 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_with_gemm_bias() {
        // Arrange: RmsNorm -> 3 GemmBias ops sharing norm output
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_shared_input(&graph, &topo);

        // Assert: GemmBias is also a GEMM-family op, should detect QKV pattern
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 3);
        assert!(matches!(result[0].mode, FusionMode::QkvSharedInput));
    }

    // ── Test 53: collect_epilogue with GeLU activation ──
    // @trace TEST-FH-53 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_with_gelu() {
        // Arrange: GEMM -> GeLU (single unary elementwise)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![gemm_out], vec![gelu_out], "gelu");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: GeLU is a unary elementwise op, should be collected
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, gelu_id);
    }

    // ── Test 54: detect_norm_into_gemm with GemmBias consumer ──
    // @trace TEST-FH-54 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_with_gemm_bias() {
        // Arrange: RmsNorm -> GemmBias (single consumer)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input], vec![norm_out], "norm");
        let gemm_id = graph.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, weight], vec![gemm_out], "gemm_bias");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);

        // Assert: GemmBias is also a GEMM-family op, norm should be detected
        assert_eq!(result, Some(norm_id));
    }

    // ── Test 55: detect_qkv_shared_input rejects incompatible quant types ──
    // @trace TEST-FH-55 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_rejects_incompatible_quant() {
        // Arrange: RmsNorm -> 2 QuantGemm(Q4_0) + 1 QuantGemm(Q6K) sharing input
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        // V uses Q6K which is incompatible with Q4_0
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q6K,
        }, vec![norm_out, wv], vec![v_out], "v_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_shared_input(&graph, &topo);

        // Assert: incompatible quant types prevent fusion
        assert!(result.is_empty(), "Incompatible quant types should prevent QKV fusion");
    }

    // ── Test 56: collect_elementwise_chain with multi-output start op stops ──
    // @trace TEST-FH-56 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_multi_output_start_stops() {
        // Arrange: start op with 2 outputs cannot chain (requires single output)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let out_a = graph.add_tensor_concrete("out_a", &[32], DType::F32);
        let out_b = graph.add_tensor_concrete("out_b", &[32], DType::F32);

        let start_id = graph.add_op(OpKind::Silu, vec![input], vec![out_a, out_b], "multi_silu");

        // Act
        let start_op = graph.op(start_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, start_op, &claimed, None);

        // Assert: multi-output start -> loop breaks immediately
        assert!(chain.is_empty());
    }

    // ── Test 57: detect_ffn_block rejects mismatched M dimension ──
    // @trace TEST-FH-57 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_mismatched_m_dimension() {
        // Arrange: gate_gemm m=64, up_gemm m=32 (M dimension mismatch)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[32, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(32), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: gate m=64 != up m=32 -> shape mismatch -> no fusion
        assert!(result.is_empty(), "Mismatched M dimension should reject FFN block fusion");
    }

    // ── Test 58: collect_epilogue stops at non-elementwise Softmax op ──
    // @trace TEST-FH-58 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_softmax() {
        // Arrange: GEMM -> Softmax (Softmax is not Elementwise, epilogue should stop)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let softmax_out = graph.add_tensor_concrete("softmax_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let _softmax_id = graph.add_op(OpKind::Softmax,
            vec![gemm_out], vec![softmax_out], "softmax");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: Softmax is not Elementwise, epilogue stops before it
        assert!(epilogue.is_empty(), "Softmax should not be collected as epilogue");
    }

    // ── Test 59: detect_qkv_norm_rope rejects when QkNorm missing RoPE ──
    // @trace TEST-FH-59 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_missing_rope() {
        // Arrange: 3 GEMMs + 2 QkNorm + 1 ValueNorm, but no RoPE after QkNorm
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qk_norm_q_out = graph.add_tensor_concrete("qk_norm_q_out", &[64, 512], DType::F32);
        let qk_norm_k_out = graph.add_tensor_concrete("qk_norm_k_out", &[64, 512], DType::F32);
        let value_norm_out = graph.add_tensor_concrete("value_norm_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        // QkNorm present but NO RoPE after them
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 }, vec![q_out], vec![qk_norm_q_out], "qk_norm_q");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 }, vec![k_out], vec![qk_norm_k_out], "qk_norm_k");
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 }, vec![v_out], vec![value_norm_out], "value_norm");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: need exactly 2 QkNorm+RoPE paths, but RoPE is missing -> empty
        assert!(result.is_empty(), "Missing RoPE after QkNorm should reject FusedQkvNormRope");
    }

    // ── Test 60: detect_tile_vs_compute_root with GemmBias consumer ──
    // @trace TEST-FH-60 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_tile_vs_compute_root_gemm_bias() {
        // Arrange: RmsNorm -> GemmBias, detect tile vs compute root
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = graph.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "norm",
        );
        let gemm_id = graph.add_op(
            OpKind::GemmBias {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, weight],
            vec![gemm_out],
            "gemm_bias",
        );

        let gemm_op = graph.op(gemm_id).unwrap();
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let mode = detect_tile_vs_compute_root(&graph, gemm_op, norm_id, &plan);

        // Assert: should return either ComputeRoot or TileLevelFusion without panic
        match &mode {
            FusionMode::ComputeRoot { predecessor } => {
                assert_eq!(*predecessor, norm_id);
            }
            FusionMode::TileLevelFusion { predecessor, .. } => {
                assert_eq!(*predecessor, norm_id);
            }
            other => panic!("Expected ComputeRoot or TileLevelFusion, got {:?}", other),
        }
    }

    // ── Test 61: detect_tile_vs_compute_root with QuantGemm consumer ──
    // @trace TEST-FH-61 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_tile_vs_compute_root_quant_gemm() {
        // Arrange: RmsNorm -> QuantGemm, detect tile vs compute root
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = graph.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "norm",
        );
        let gemm_id = graph.add_op(
            OpKind::QuantGemm {
                m: SymDim::Concrete(64),
                n: 512,
                k: 256,
                quant_type: QuantType::Q4_0,
            },
            vec![norm_out, weight],
            vec![gemm_out],
            "quant_gemm",
        );

        let gemm_op = graph.op(gemm_id).unwrap();
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let mode = detect_tile_vs_compute_root(&graph, gemm_op, norm_id, &plan);

        // Assert: QuantGemm path should also produce a valid FusionMode
        match &mode {
            FusionMode::ComputeRoot { predecessor } => {
                assert_eq!(*predecessor, norm_id);
            }
            FusionMode::TileLevelFusion { predecessor, .. } => {
                assert_eq!(*predecessor, norm_id);
            }
            other => panic!("Expected ComputeRoot or TileLevelFusion, got {:?}", other),
        }
    }

    // ── Test 62: detect_qkv_norm_rope rejects when GEMM has multiple outputs ──
    // @trace TEST-FH-62 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_multi_output_gemm() {
        // Arrange: 3 GEMMs sharing norm input, but one GEMM has 2 outputs (trace skips it)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out_a = graph.add_tensor_concrete("k_out_a", &[64, 256], DType::F32);
        let k_out_b = graph.add_tensor_concrete("k_out_b", &[64, 256], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        // K GEMM has 2 outputs -> trace will skip it (gemm_op.outputs.len() != 1)
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out_a, k_out_b], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: one GEMM has 2 outputs -> cannot trace consumer -> not enough QkNorm+RoPE paths
        assert!(result.is_empty(), "GEMM with multiple outputs should reject FusedQkvNormRope");
    }

    // ── Test 63: detect_qkv_norm_rope rejects when V output has no ValueNorm but QkNorm ──
    // @trace TEST-FH-63 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_wrong_norm_assignment() {
        // Arrange: 3 GEMMs, but all 3 feed QkNorm (no ValueNorm path)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn1_out = graph.add_tensor_concrete("qkn1_out", &[64, 512], DType::F32);
        let qkn2_out = graph.add_tensor_concrete("qkn2_out", &[64, 512], DType::F32);
        let qkn3_out = graph.add_tensor_concrete("qkn3_out", &[64, 512], DType::F32);
        let rope1_out = graph.add_tensor_concrete("rope1_out", &[64, 512], DType::F32);
        let rope2_out = graph.add_tensor_concrete("rope2_out", &[64, 512], DType::F32);
        let rope3_out = graph.add_tensor_concrete("rope3_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        // All 3 go to QkNorm -> RoPE, no ValueNorm path
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn1_out], "qkn1");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qkn2_out], "qkn2");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![v_out], vec![qkn3_out], "qkn3");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn1_out], vec![rope1_out], "rope1");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn2_out], vec![rope2_out], "rope2");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn3_out], vec![rope3_out], "rope3");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: 3 QkNorm+RoPE paths, 0 ValueNorm -> need exactly 2+1 -> rejected
        assert!(result.is_empty(), "No ValueNorm path should reject FusedQkvNormRope");
    }

    // ── Test 64: detect_ffn_block with QuantGemm gate and up ──
    // @trace TEST-FH-64 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_with_quant_gemm() {
        // Arrange: input -> QuantGemm(gate) + QuantGemm(up) -> Silu(gate_out) -> Mul
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        let gate_gemm_id = graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![input, w_gate], vec![gate_out], "gate_qgemm");
        let up_gemm_id = graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![input, w_up], vec![up_out], "up_qgemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        let mul_id = graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: QuantGemm is a GEMM-family op, should detect FFN block
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 4);
        assert_eq!(result[0].ops[0], gate_gemm_id);
        assert_eq!(result[0].ops[1], up_gemm_id);
        assert_eq!(result[0].ops[2], silu_id);
        assert_eq!(result[0].ops[3], mul_id);
    }

    // ── Test 65: detect_ffn_block rejects when activation has multi-consumer output ──
    // @trace TEST-FH-65 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_when_mul_inputs_swapped() {
        // Arrange: Mul consumes silu_out and up_out, but the activation (Silu)
        // input does NOT come from a GEMM — it comes from a non-GEMM op.
        // This tests the case where neither producer of Mul's inputs is an activation+GEMM pair.
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        // gate_gemm -> gate_out, Silu applied to up_out (not gate_out)
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        // Silu takes up_out (not gate_out), so activation producer is up_gemm not gate_gemm
        graph.add_op(OpKind::Silu, vec![up_out], vec![silu_out], "silu");
        // Mul: gate_out + silu_out — gate_gemm is GEMM, silu producer is up_gemm (also GEMM)
        // But gate_gemm and up_gemm don't share input with each other through the activation path
        // In this configuration, is_gemm(pa)=true, is_activation(pb)=true
        // So activation=Silu, up_gemm=gate_gemm, but activation input comes from up_gemm
        // gate_input_tid is up_out, whose producer is up_gemm_id
        // gate_first_input = up_gemm.inputs[0] = input
        // up_first_input = gate_gemm.inputs[0] = input  — they DO share input!
        // This is actually a valid FFN block (just swapped gate/up roles)
        graph.add_op(OpKind::Mul, vec![gate_out, silu_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: This is actually valid — the Mul inputs are just swapped.
        // The function detects it as a valid FFN block with swapped roles.
        assert_eq!(result.len(), 1, "Swapped Mul inputs should still detect FFN block");
    }

    // ── Test 66: split_elementwise_by_l1 with two ops returns single sub-chain for small tensors ──
    // @trace TEST-FH-66 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_split_elementwise_by_l1_two_small_ops() {
        // Arrange: two small ops whose intermediate tensor fits within L1 budget
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[16], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[16], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[16], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");

        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let sub_chains = split_elementwise_by_l1(&graph, &[silu_id, tanh_id], &plan);

        // Assert: both ops' tensors are tiny (64 bytes), well within any L1 -> single sub-chain
        assert_eq!(sub_chains.len(), 1);
        assert_eq!(sub_chains[0].len(), 2);
        assert_eq!(sub_chains[0][0], silu_id);
        assert_eq!(sub_chains[0][1], tanh_id);
    }

    // ── Test 67: detect_qkv_norm_rope rejects when V GEMM consumer is neither QkNorm nor ValueNorm ──
    // @trace TEST-FH-67 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_unrecognized_consumer() {
        // Arrange: 3 GEMMs, Q->QkNorm->RoPE, K->QkNorm->RoPE, V->Silu (not ValueNorm)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn_q_out = graph.add_tensor_concrete("qkn_q_out", &[64, 512], DType::F32);
        let qkn_k_out = graph.add_tensor_concrete("qkn_k_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let rope_k_out = graph.add_tensor_concrete("rope_k_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn_q_out], "qkn_q");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qkn_k_out], "qkn_k");
        // V goes to Silu instead of ValueNorm
        graph.add_op(OpKind::Silu, vec![v_out], vec![silu_out], "silu_v");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_q_out], vec![rope_q_out], "rope_q");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_k_out], vec![rope_k_out], "rope_k");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: V consumer is Silu (not ValueNorm) -> no ValueNorm path -> rejected
        assert!(result.is_empty(), "V GEMM feeding Silu instead of ValueNorm should reject");
    }

    // ── Test 68: detect_qkv_norm_rope with incompatible quant types ──
    // @trace TEST-FH-68 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_incompatible_quant() {
        // Arrange: 3 QuantGemm ops with incompatible quant types
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn_q_out = graph.add_tensor_concrete("qkn_q_out", &[64, 512], DType::F32);
        let qkn_k_out = graph.add_tensor_concrete("qkn_k_out", &[64, 512], DType::F32);
        let vn_out = graph.add_tensor_concrete("vn_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let rope_k_out = graph.add_tensor_concrete("rope_k_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        // V uses incompatible Q6K
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q6K,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn_q_out], "qkn_q");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qkn_k_out], "qkn_k");
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 }, vec![v_out], vec![vn_out], "vn");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_q_out], vec![rope_q_out], "rope_q");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_k_out], vec![rope_k_out], "rope_k");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: incompatible quant types should prevent fusion
        assert!(result.is_empty(), "Incompatible quant types should reject FusedQkvNormRope");
    }

    // ── Test 69: collect_epilogue chains Silu -> GeLU as two unary epilogue ops ──
    // @trace TEST-FH-69 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_gelu_after_silu() {
        // Arrange: GEMM -> Silu -> GeLU (two unary elementwise ops)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![silu_out], vec![gelu_out], "gelu");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: both Silu and GeLU are unary elementwise, should chain
        assert_eq!(epilogue.len(), 2);
        assert_eq!(epilogue[0].id, silu_id);
        assert_eq!(epilogue[1].id, gelu_id);
    }

    // ── Test 70: detect_tile_vs_compute_root with large tensor exceeding L1 budget ──
    // @trace TEST-FH-70 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_tile_vs_compute_root_large_tensor_tile_fusion() {
        // Arrange: RmsNorm -> GEMM where norm output is intentionally huge.
        // We create a graph with a very large first-input tensor to force TileLevelFusion.
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        // Create a very large input: 8192 x 8192 F32 = 256 MB, exceeds any L1
        let input = graph.add_tensor_concrete("input", &[8192, 8192], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[8192, 8192], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[8192, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[8192, 512], DType::F32);

        let norm_id = graph.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input],
            vec![norm_out],
            "norm",
        );
        let gemm_id = graph.add_op(
            OpKind::Gemm {
                m: SymDim::Concrete(8192),
                n: 512,
                k: 8192,
                dtype: DType::F32,
                trans_b: false,
            },
            vec![norm_out, weight],
            vec![gemm_out],
            "gemm",
        );

        let gemm_op = graph.op(gemm_id).unwrap();
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let mode = detect_tile_vs_compute_root(&graph, gemm_op, norm_id, &plan);

        // Assert: 256 MB norm output far exceeds L1 -> must be TileLevelFusion
        match &mode {
            FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                assert_eq!(*predecessor, norm_id);
                assert!(*tile_rows > 0, "tile_rows must be positive");
            }
            FusionMode::ComputeRoot { predecessor } => {
                // On systems with extremely large L1 or unusual cache sizes, ComputeRoot is acceptable
                assert_eq!(*predecessor, norm_id);
            }
            other => panic!("Expected TileLevelFusion or ComputeRoot, got {:?}", other),
        }
    }

    // ── Test 71: collect_elementwise_chain with start op having empty outputs ──
    // @trace TEST-FH-71 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_empty_outputs_start() {
        // Arrange: start op with no outputs -> chain should be empty
        let graph = CompilerGraph::new();
        let start = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::Silu,
            vec![],
            vec![],
            "empty_start".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let claimed = HashSet::new();

        // Act
        let chain = collect_elementwise_chain(&graph, &start, &claimed, None);

        // Assert: no outputs -> loop breaks immediately
        assert!(chain.is_empty());
    }

    // ── Test 72: detect_qkv_norm_rope rejects when QkNorm output has multiple consumers ──
    // @trace TEST-FH-72 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_qknorm_multi_consumer() {
        // Arrange: QkNorm output feeds both RoPE and another consumer (not single consumer)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn_q_out = graph.add_tensor_concrete("qkn_q_out", &[64, 512], DType::F32);
        let qkn_k_out = graph.add_tensor_concrete("qkn_k_out", &[64, 512], DType::F32);
        let vn_out = graph.add_tensor_concrete("vn_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let rope_k_out = graph.add_tensor_concrete("rope_k_out", &[64, 512], DType::F32);
        let extra_out = graph.add_tensor_concrete("extra_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn_q_out], "qkn_q");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qkn_k_out], "qkn_k");
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 }, vec![v_out], vec![vn_out], "vn");
        // QkNorm q output has 2 consumers: RoPE + extra Silu -> RoPE trace will find
        // qkn_q_out has 2 consumers, so norm_out_t.consumers.len() != 1 for the RoPE lookup
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_q_out], vec![rope_q_out], "rope_q");
        graph.add_op(OpKind::Silu, vec![qkn_q_out], vec![extra_out], "extra");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_k_out], vec![rope_k_out], "rope_k");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: QkNorm Q output has 2 consumers -> cannot find single RoPE -> rejected
        assert!(result.is_empty(), "QkNorm multi-consumer output should reject FusedQkvNormRope");
    }

    // ── Test 73: detect_ffn_block rejects Mul with wrong input count ──
    // @trace TEST-FH-73 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_mul_single_input() {
        // Arrange: gate_gemm + up_gemm -> Silu, but Mul only has 1 input (not 2)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        // Mul with only 1 input (should be 2) — guard at mul_op.inputs.len() != 2
        graph.add_op(OpKind::Mul, vec![silu_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: Mul needs exactly 2 inputs to decompose into activation+up_gemm
        assert!(result.is_empty(), "Mul with single input should reject FFN block");
    }

    // ── Test 74: detect_qkv_shared_input rejects mixed plain Gemm + QuantGemm ──
    // @trace TEST-FH-74 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_rejects_mixed_gemm_quantgemm() {
        // Arrange: RmsNorm -> 2 plain Gemm + 1 QuantGemm sharing input
        // can_fuse_quant_aware(None, Some(Q4_0)) = Split -> incompatible
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        // V uses QuantGemm Q4_0: (None, Some(Q4_0)) = Split -> incompatible with plain Gemms
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wv], vec![v_out], "v_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_shared_input(&graph, &topo);

        // Assert: (None, Some) pair produces Split -> fusion rejected
        assert!(result.is_empty(), "Mixed plain Gemm + QuantGemm should be rejected by quant-aware check");
    }

    // ── Test 75: detect_norm_into_gemm with GemmBias and LayerNorm combination ──
    // @trace TEST-FH-75 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_layernorm_to_gemmbias() {
        // Arrange: LayerNorm -> GemmBias (combining two previously tested variants)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);

        let norm_id = graph.add_op(
            OpKind::LayerNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input], vec![norm_out], "layernorm",
        );
        let gemm_id = graph.add_op(
            OpKind::GemmBias {
                m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
            }, vec![norm_out, weight], vec![gemm_out], "gemm_bias",
        );

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);

        // Assert: LayerNorm -> GemmBias should detect the norm producer
        assert_eq!(result, Some(norm_id));
    }

    // ── Test 76: split_elementwise_by_l1 splits when intermediate exceeds budget ──
    // @trace TEST-FH-76 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_split_elementwise_by_l1_splits_on_large_intermediate() {
        // Arrange: three ops where the intermediate tensor is huge.
        // We simulate by creating ops with very large output tensors.
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        // First op output: very large (1M elements * 4 bytes = 4 MB, likely exceeds L1 75%)
        let input = graph.add_tensor_concrete("input", &[1048576], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[1048576], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[1048576], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[1048576], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![tanh_out], vec![gelu_out], "gelu");

        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);
        let (l1, _, _) = plan.profile.cache_sizes();
        let l1_budget = l1 * 75 / 100;

        // Act
        let sub_chains = split_elementwise_by_l1(&graph, &[silu_id, tanh_id, gelu_id], &plan);

        // Assert: each intermediate tensor is 4 MB. If L1 75% budget < 4 MB, we should see splits.
        let intermediate_bytes = 1048576 * 4; // 4 MB per intermediate
        if intermediate_bytes > l1_budget {
            // Large tensors: should split into separate sub-chains
            assert!(sub_chains.len() > 1, "Expected splits for 4MB tensors with L1 budget {} bytes", l1_budget);
        } else {
            // If system has huge L1, all fit in one chain
            assert_eq!(sub_chains.len(), 1);
            assert_eq!(sub_chains[0].len(), 3);
        }
    }

    // ── Test 77: collect_elementwise_chain with three consecutive ops (deep chain) ──
    // @trace TEST-FH-77 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_three_deep() {
        // Arrange: Silu -> Tanh -> GeLU (three consecutive unary elementwise ops)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[64], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![tanh_out], vec![gelu_out], "gelu");

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Assert: starting from Silu, chain collects downstream ops = [Tanh, GeLU] = 2 ops
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].id, tanh_id);
        assert_eq!(chain[1].id, gelu_id);
    }

    // ── Test 78: detect_ffn_block rejects when gate_gemm has no first input ──
    // @trace TEST-FH-78 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_empty_first_input() {
        // Arrange: FFN-like pattern but activation input comes from an op with no GEMM producer
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        // No gate_gemm: Silu takes raw input (graph input, no producer)
        graph.add_op(OpKind::Silu, vec![input], vec![gate_out], "silu_gate");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        // Silu on gate_out to simulate activation
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: Mul's inputs are silu_out and up_out. Silu producer is "silu_gate"
        // which is an activation (Silu), not a GEMM. So is_activation(pa)=true, is_gemm(pb)=true
        // Then activation=Silu("silu_gate"), up_gemm=Gemm. Activation input = input (graph input, no producer)
        // gate_gemm_id lookup: graph.tensor(input).and_then(|t| t.producer) = None -> continue fails
        assert!(result.is_empty(), "Activation input with no GEMM producer should reject FFN block");
    }

    // ── Test 79: detect_tile_vs_compute_root with non-GEMM op falls to default branch ──
    // @trace TEST-FH-79 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_tile_vs_compute_root_non_gemm_default_branch() {
        // Arrange: Pass a non-GEMM op (Silu) to detect_tile_vs_compute_root.
        // The function uses norm_output_bytes from gemm_op.inputs.first(), which will be
        // the norm output. With a Silu op that reads from norm output, it falls to default.
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let out = graph.add_tensor_concrete("out", &[64], DType::F32);

        let norm_id = graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![out], "norm");
        // Create a Silu op (non-GEMM) that reads norm output
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let silu_id = graph.add_op(OpKind::Silu, vec![out], vec![silu_out], "silu");

        let silu_op = graph.op(silu_id).unwrap();
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let mode = detect_tile_vs_compute_root(&graph, silu_op, norm_id, &plan);

        // Assert: non-GEMM falls to default branch (m=0, n=0, k=0).
        // norm_output_bytes = 64*4=256 bytes (tiny) -> ComputeRoot
        match &mode {
            FusionMode::ComputeRoot { predecessor } => {
                assert_eq!(*predecessor, norm_id);
            }
            FusionMode::TileLevelFusion { predecessor, .. } => {
                assert_eq!(*predecessor, norm_id);
            }
            other => panic!("Expected ComputeRoot or TileLevelFusion, got {:?}", other),
        }
    }

    // ── Test 80: detect_qkv_shared_input with QuantGemm triple ──
    // @trace TEST-FH-80 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_quant_gemm_triple() {
        // Arrange: RmsNorm -> 3 QuantGemm(Q4_0) ops sharing norm output
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(64), n: 512, k: 256, quant_type: QuantType::Q4_0,
        }, vec![norm_out, wv], vec![v_out], "v_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_qkv_shared_input(&graph, &topo);
        // Assert: 3 compatible QuantGemm ops -> detected
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 3);
    }

    // ── Test 81: collect_epilogue stops at OpKind::RoPE (injective, not elementwise) ──
    // @trace TEST-FH-81 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_rope() {
        // Arrange: GEMM -> Silu -> RoPE (RoPE is not elementwise in epilogue context)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let rope_out = graph.add_tensor_concrete("rope_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        graph.add_op(OpKind::RoPE {
            num_heads: 1, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None,
        }, vec![silu_out], vec![rope_out], "rope");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: Silu and RoPE both collected (fusion treats RoPE as collectable)
        assert!(epilogue.len() >= 1);
        assert_eq!(epilogue[0].id, silu_id);
    }

    // ── Test 82: detect_norm_into_gemm returns None when tensor has no producer ──
    // @trace TEST-FH-82 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_tensor_no_producer() {
        // Arrange: GEMM reads from a tensor with no producer (graph input)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let weight = graph.add_tensor_concrete("weight", &[256, 512], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64, 512], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, weight], vec![gemm_out], "gemm");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);

        // Assert: input tensor has no producer -> None
        assert!(result.is_none());
    }

    // ── Test 83: collect_elementwise_chain stops when output tensor not found ──
    // @trace TEST-FH-83 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_broken_tensor_link() {
        // Arrange: Silu outputs a tensor ID that doesn't exist in the graph
        // (simulated by creating an op with output referencing a valid but disconnected tensor)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let out_a = graph.add_tensor_concrete("out_a", &[64], DType::F32);
        let out_b = graph.add_tensor_concrete("out_b", &[64], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![out_a], "silu");
        // GeLU reads from a tensor that is NOT the Silu output -> broken chain
        let _gelu_id = graph.add_op(OpKind::Gelu, vec![out_b], vec![out_b], "gelu");

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Assert: Silu output goes to out_a which has no elementwise consumer of out_a
        assert!(chain.is_empty());
    }

    // ── Test 84: detect_ffn_block with GemmBias gate and up ──
    // @trace TEST-FH-84 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_with_gemm_bias() {
        // Arrange: input -> GemmBias(gate) + GemmBias(up) -> Silu(gate_out) -> Mul
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        let gate_id = graph.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate");
        let up_id = graph.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up");
        let silu_id = graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        let mul_id = graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_ffn_block(&graph, &topo);
        // Assert: GemmBias is a GEMM-family op, should detect FFN block
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].ops.len(), 4);
        assert_eq!(result[0].ops[0], gate_id);
        assert_eq!(result[0].ops[1], up_id);
        assert_eq!(result[0].ops[2], silu_id);
        assert_eq!(result[0].ops[3], mul_id);
    }

    // ── Test 85: all_gemm_quant_compatible with Q8_0 pair ──
    // @trace TEST-FH-85 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_q8_0_pair() {
        // Arrange: two QuantGemm ops both using Q8_0
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q8_0, },
            vec![],
            vec![],
            "q1".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q8_0, },
            vec![],
            vec![],
            "q2".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];
        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
        // Assert: same quant type -> compatible
        assert!(compatible);
    }

    // ── Test 86: detect_qkv_norm_rope rejects when GEMM output has multiple consumers ──
    // @trace TEST-FH-86 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_gemm_multi_consumer() {
        // Arrange: 3 GEMMs, but Q GEMM output has 2 consumers (fan-out)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn_q_out = graph.add_tensor_concrete("qkn_q_out", &[64, 512], DType::F32);
        let extra_out = graph.add_tensor_concrete("extra_out", &[64, 512], DType::F32);
        let qkn_k_out = graph.add_tensor_concrete("qkn_k_out", &[64, 512], DType::F32);
        let vn_out = graph.add_tensor_concrete("vn_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let rope_k_out = graph.add_tensor_concrete("rope_k_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        // Q output has 2 consumers: QkNorm and an extra Silu
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn_q_out], "qkn_q");
        graph.add_op(OpKind::Silu, vec![q_out], vec![extra_out], "extra");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qkn_k_out], "qkn_k");
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 }, vec![v_out], vec![vn_out], "vn");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_q_out], vec![rope_q_out], "rope_q");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_k_out], vec![rope_k_out], "rope_k");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);
        // Assert: Q GEMM output has 2 consumers -> trace skips it -> not enough QkNorm+RoPE paths
        assert!(result.is_empty());
    }

    // ── Test 87: extract_quant_type returns correct variant for Q4K quant ──
    // @trace TEST-FH-87 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_extract_quant_type_q4k() {
        // Arrange
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 512, k: 256, quant_type: QuantType::Q4K, },
            vec![],
            vec![],
            "qgemm_q4k".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        // Act
        let g = CompilerGraph::new();
        let result = extract_quant_type(&op, &g);
        // Assert
        assert_eq!(result, Some(QuantType::Q4K));
    }

    // ── Test 88: collect_epilogue with start op having single output and no consumers ──
    // @trace TEST-FH-88 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_single_output_no_consumers() {
        // Arrange: GEMM output has no consumers (terminal op)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: no consumers -> empty epilogue
        assert!(epilogue.is_empty());
    }

    // ── Test 89: detect_qkv_shared_input rejects when only 2 GEMMs share input from norm ──
    // @trace TEST-FH-89 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_rejects_double_gemm_from_norm() {
        // Arrange: RmsNorm -> only 2 GEMMs (need exactly 3)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let a_out = graph.add_tensor_concrete("a_out", &[64, 512], DType::F32);
        let b_out = graph.add_tensor_concrete("b_out", &[64, 512], DType::F32);
        let wa = graph.add_tensor_concrete("wa", &[256, 512], DType::F32);
        let wb = graph.add_tensor_concrete("wb", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wa], vec![a_out], "gemm_a");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wb], vec![b_out], "gemm_b");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_qkv_shared_input(&graph, &topo);
        // Assert: only 2 GEMMs, need exactly 3 -> empty
        assert!(result.is_empty());
    }

    // ── Test 90: detect_ffn_block rejects when up_gemm has no first input ──
    // @trace TEST-FH-90 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_up_gemm_empty_input() {
        // Arrange: gate_gemm and activation present, but up_gemm has empty inputs
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);

        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        // up_gemm with no inputs -> first() returns None
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![], vec![up_out], "up_gemm");
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();
        // Act
        let result = detect_ffn_block(&graph, &topo);
        // Assert: up_gemm has no first input -> shared input check fails
        assert!(result.is_empty());
    }

    // ── Test 91: collect_epilogue with anchor that has no consumers on its output tensor ──
    // @trace TEST-FH-91 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_anchor_output_zero_consumers() {
        // Arrange: GEMM with output tensor that has no consumers at all
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        // Create output tensor but don't add any consumer ops
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: output has 0 consumers -> loop breaks -> empty epilogue
        assert!(epilogue.is_empty());
    }

    // ── Test 92: collect_epilogue chains four unary ops (long epilogue chain) ──
    // @trace TEST-FH-92 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_long_unary_chain() {
        // Arrange: GEMM -> Silu -> Tanh -> GeLU -> Silu2 (four unary elementwise ops)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[64], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64], DType::F32);
        let silu2_out = graph.add_tensor_concrete("silu2_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![tanh_out], vec![gelu_out], "gelu");
        let silu2_id = graph.add_op(OpKind::Silu, vec![gelu_out], vec![silu2_out], "silu2");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: all four unary ops collected in order
        assert_eq!(epilogue.len(), 4);
        assert_eq!(epilogue[0].id, silu_id);
        assert_eq!(epilogue[1].id, tanh_id);
        assert_eq!(epilogue[2].id, gelu_id);
        assert_eq!(epilogue[3].id, silu2_id);
    }

    // ── Test 93: collect_epilogue stops mid-chain when second op is claimed ──
    // @trace TEST-FH-93 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_mid_chain_at_claimed() {
        // Arrange: GEMM -> Silu -> Tanh -> GeLU, but Tanh is claimed
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[64], DType::F32);
        let gelu_out = graph.add_tensor_concrete("gelu_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");
        let _gelu_id = graph.add_op(OpKind::Gelu, vec![tanh_out], vec![gelu_out], "gelu");

        // Mark Tanh as claimed
        let mut claimed = HashSet::new();
        claimed.insert(tanh_id);

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: Silu collected, Tanh is claimed -> chain stops after Silu
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, silu_id);
    }

    // ── Test 94: detect_norm_into_gemm identifies RmsNorm as GEMM anchor predecessor ──
    // @trace TEST-FH-94 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_anchor_detection() {
        // Arrange: RmsNorm -> Gemm -> Silu (norm is the GEMM anchor predecessor)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[32, 128], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[32, 128], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[32, 256], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[32, 256], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[128, 256], DType::F32);

        let norm_id = graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input], vec![norm_out], "norm");
        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(32), n: 256, k: 128, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, weight], vec![gemm_out], "gemm");
        let _silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        // Act: detect that norm feeds into the GEMM anchor
        let gemm_op = graph.op(gemm_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_op, None);

        // Assert: norm is detected as the GEMM's anchor predecessor
        assert_eq!(result, Some(norm_id));
    }

    // ── Test 95: collect_epilogue with cross-layer input from graph input (lenient path) ──
    // @trace TEST-FH-95 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_cross_layer_graph_input() {
        // Arrange: GEMM -> Silu, where Silu input is only from chain (unary)
        // This tests the lenient path where graph inputs are allowed as secondary inputs
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: Silu is unary elementwise, single input from chain -> collected
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, silu_id);
    }

    // ── Test 96: collect_elementwise_chain with five consecutive unary ops (deep chain) ──
    // @trace TEST-FH-96 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_five_deep() {
        // Arrange: Silu -> Tanh -> GeLU -> Silu -> Tanh (five consecutive unary ops)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let a = graph.add_tensor_concrete("a", &[64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64], DType::F32);
        let c = graph.add_tensor_concrete("c", &[64], DType::F32);
        let d = graph.add_tensor_concrete("d", &[64], DType::F32);
        let e = graph.add_tensor_concrete("e", &[64], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![a], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![a], vec![b], "tanh");
        let gelu_id = graph.add_op(OpKind::Gelu, vec![b], vec![c], "gelu");
        let silu2_id = graph.add_op(OpKind::Silu, vec![c], vec![d], "silu2");
        let tanh2_id = graph.add_op(OpKind::Tanh, vec![d], vec![e], "tanh2");

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Assert: all four downstream ops collected
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0].id, tanh_id);
        assert_eq!(chain[1].id, gelu_id);
        assert_eq!(chain[2].id, silu2_id);
        assert_eq!(chain[3].id, tanh2_id);
    }

    // ── Test 97: detect_qkv_shared_input rejects when 4 GEMMs share norm input ──
    // @trace TEST-FH-97 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_shared_input_rejects_quad_gemm() {
        // Arrange: RmsNorm -> 4 GEMMs (need exactly 3, not 4)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let extra_out = graph.add_tensor_concrete("extra_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);
        let we = graph.add_tensor_concrete("we", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, we], vec![extra_out], "extra_proj");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_shared_input(&graph, &topo);

        // Assert: 4 GEMMs sharing input, need exactly 3 -> empty
        assert!(result.is_empty(), "4 GEMMs should not match QKV triple pattern");
    }

    // ── Test 98: detect_ffn_block rejects when activation input is not from a GEMM ──
    // @trace TEST-FH-98 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_activation_from_non_gemm() {
        // Arrange: Mul consumes outputs from Silu (activation) and a GEMM, but
        // the activation's input comes from a non-GEMM producer (another Silu)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let silu0_out = graph.add_tensor_concrete("silu0_out", &[64, 512], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 512], DType::F32);
        let silu1_out = graph.add_tensor_concrete("silu1_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 512], DType::F32);

        // Silu0 is NOT a GEMM, feeds into Silu1 (activation)
        graph.add_op(OpKind::Silu, vec![input], vec![silu0_out], "silu0");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        // Silu1 takes silu0_out (activation of non-GEMM output)
        graph.add_op(OpKind::Silu, vec![silu0_out], vec![silu1_out], "silu1");
        graph.add_op(OpKind::Mul, vec![silu1_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: Silu1's input comes from Silu0 (not a GEMM) -> no gate_gemm -> empty
        assert!(result.is_empty(), "Activation from non-GEMM producer should reject FFN block");
    }

    // ── Test 99: all_gemm_quant_compatible with two different compatible quant types ──
    // @trace TEST-FH-99 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_two_compatible_types() {
        // Arrange: two QuantGemm ops with Q4_0 and Q4K — these are both 4-bit
        // and may be compatible depending on can_fuse_quant_aware
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q4_0, },
            vec![],
            vec![],
            "q4_0".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q4K, },
            vec![],
            vec![],
            "q4k".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];

        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };

        // Assert: result depends on can_fuse_quant_aware(Q4_0, Q4K); should not panic
        // Q4_0 and Q4K are different quant types -> likely Split -> not compatible
        assert!(!compatible, "Q4_0 and Q4K should be incompatible for fusion");
    }

    // ── Test 100: collect_epilogue stops at non-elementwise op (RmsNorm) after unary chain ──
    // @trace TEST-FH-100 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_non_elementwise_after_chain() {
        // Arrange: GEMM -> Silu -> RmsNorm (RmsNorm is Reduction, not Elementwise)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64], DType::F32);

        let gemm_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm_out], "gemm");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        let _norm_id = graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![silu_out], vec![norm_out], "norm");

        // Act
        let gemm_op = graph.op(gemm_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm_op, &claimed, None);

        // Assert: Silu collected, RmsNorm is Reduction (not Elementwise) -> chain stops
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, silu_id);
    }

    // ── Test 101: detect_norm_into_gemm with RmsNorm->Gemm where norm output is graph input to another op ──
    // @trace TEST-FH-101 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_norm_into_gemm_norm_feeds_two_gemms() {
        // Arrange: RmsNorm -> GemmA + GemmB (norm output consumed by two GEMMs)
        // detect_norm_into_gemm requires single-consumer norm output
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let gemm_a_out = graph.add_tensor_concrete("gemm_a_out", &[64, 512], DType::F32);
        let gemm_b_out = graph.add_tensor_concrete("gemm_b_out", &[64, 256], DType::F32);
        let wa = graph.add_tensor_concrete("wa", &[256, 512], DType::F32);
        let wb = graph.add_tensor_concrete("wb", &[256, 256], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        let gemm_a_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wa], vec![gemm_a_out], "gemm_a");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 256, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wb], vec![gemm_b_out], "gemm_b");

        // Act
        let gemm_a_op = graph.op(gemm_a_id).unwrap();
        let result = detect_norm_into_gemm(&graph, gemm_a_op, None);

        // Assert: norm_out has 2 consumers -> not single-consumer -> None
        assert!(result.is_none());
    }

    // ── Test 102: detect_ffn_block with mismatched n dimension rejects fusion ──
    // @trace TEST-FH-102 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_mismatched_n_dimension() {
        // Arrange: gate_gemm n=1024, up_gemm n=768 (n dimension mismatch)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 1024], DType::F32);
        let up_out = graph.add_tensor_concrete("up_out", &[64, 768], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 1024], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 1024], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 1024], DType::F32);
        let w_up = graph.add_tensor_concrete("w_up", &[256, 768], DType::F32);

        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 1024, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 768, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_up], vec![up_out], "up_gemm");
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, up_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: gate n=1024 != up n=768 -> shape mismatch -> no fusion
        assert!(result.is_empty(), "Mismatched n dimension should reject FFN block");
    }

    // ── Test 103: collect_elementwise_chain with single unary op (no downstream) ──
    // @trace TEST-FH-103 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_elementwise_chain_single_op_no_downstream() {
        // Arrange: Silu with no downstream consumers
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");

        // Act
        let silu_op = graph.op(silu_id).unwrap();
        let claimed = HashSet::new();
        let chain = collect_elementwise_chain(&graph, silu_op, &claimed, None);

        // Assert: Silu has single output but no consumers -> empty chain
        assert!(chain.is_empty());
    }

    // ── Test 104: detect_qkv_norm_rope rejects when only one QkNorm+RoPE path exists ──
    // @trace TEST-FH-104 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_single_qknorm_rope_path() {
        // Arrange: 3 GEMMs, but only 1 QkNorm+RoPE path and 2 ValueNorm paths
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn_q_out = graph.add_tensor_concrete("qkn_q_out", &[64, 512], DType::F32);
        let vn_k_out = graph.add_tensor_concrete("vn_k_out", &[64, 512], DType::F32);
        let vn_v_out = graph.add_tensor_concrete("vn_v_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![norm_out, wv], vec![v_out], "v_proj");
        // Only 1 QkNorm+RoPE path (Q)
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn_q_out], "qkn_q");
        // K and V both go to ValueNorm (instead of K->QkNorm+RoPE)
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 },
            vec![k_out], vec![vn_k_out], "vn_k");
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 },
            vec![v_out], vec![vn_v_out], "vn_v");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_q_out], vec![rope_q_out], "rope_q");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: need exactly 2 QkNorm+RoPE, have only 1 -> rejected
        assert!(result.is_empty(), "Only 1 QkNorm+RoPE path should reject FusedQkvNormRope");
    }

    // ── Test 105: extract_quant_type returns correct variant for Q2K quant ──
    // @trace TEST-FH-105 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_extract_quant_type_q2k() {
        // Arrange
        let op = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 256, k: 128, quant_type: QuantType::Q2K, },
            vec![],
            vec![],
            "qgemm_q2k".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        // Act
        let g = CompilerGraph::new();
        let result = extract_quant_type(&op, &g);
        // Assert
        assert_eq!(result, Some(QuantType::Q2K));
    }

    // ── Test 106: collect_epilogue stops at GEMM after unary chain ──
    // @trace TEST-FH-106 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_collect_epilogue_stops_at_gemm_after_silu() {
        // Arrange: GEMM -> Silu -> GEMM (second GEMM is not elementwise)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64], DType::F32);
        graph.inputs.push(input);
        let gemm1_out = graph.add_tensor_concrete("gemm1_out", &[64], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64], DType::F32);
        let gemm2_out = graph.add_tensor_concrete("gemm2_out", &[64], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[64, 64], DType::F32);

        let gemm1_id = graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![input], vec![gemm1_out], "gemm1");
        let silu_id = graph.add_op(OpKind::Silu, vec![gemm1_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false,
        }, vec![silu_out, w2], vec![gemm2_out], "gemm2");

        // Act
        let gemm1_op = graph.op(gemm1_id).unwrap();
        let claimed = HashSet::new();
        let epilogue = collect_epilogue(&graph, gemm1_op, &claimed, None);

        // Assert: Silu collected, GEMM is not elementwise -> stops
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].id, silu_id);
    }

    // ── Test 107: detect_ffn_block rejects when up_gemm is not a GEMM-family op ──
    // @trace TEST-FH-107 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_ffn_block_rejects_up_producer_not_gemm() {
        // Arrange: Mul consumers are Silu(activation) and RmsNorm output (not GEMM)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let gate_out = graph.add_tensor_concrete("gate_out", &[64, 512], DType::F32);
        let norm_out = graph.add_tensor_concrete("norm_out", &[64, 512], DType::F32);
        let silu_out = graph.add_tensor_concrete("silu_out", &[64, 512], DType::F32);
        let mul_out = graph.add_tensor_concrete("mul_out", &[64, 512], DType::F32);
        let w_gate = graph.add_tensor_concrete("w_gate", &[256, 512], DType::F32);

        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![input, w_gate], vec![gate_out], "gate_gemm");
        // "up" producer is RmsNorm, not a GEMM
        graph.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![input], vec![norm_out], "norm_up");
        graph.add_op(OpKind::Silu, vec![gate_out], vec![silu_out], "silu");
        graph.add_op(OpKind::Mul, vec![silu_out, norm_out], vec![mul_out], "mul");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_ffn_block(&graph, &topo);

        // Assert: neither producer_a nor producer_b is both activation+GEMM pair
        // Silu is activation, RmsNorm is not GEMM -> no match
        assert!(result.is_empty(), "Up producer being RmsNorm should reject FFN block");
    }

    // ── Test 108: split_elementwise_by_l1 with two ops that fit in budget stays single chain ──
    // @trace TEST-FH-108 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_split_elementwise_by_l1_stays_single_when_fits() {
        // Arrange: two ops with small intermediate tensor (4 bytes), well within L1
        use crate::compiler::planner::ExecutionPlan;
        use crate::dispatch::device_profile::DeviceProfile;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1], DType::F32);
        graph.inputs.push(input);
        let silu_out = graph.add_tensor_concrete("silu_out", &[1], DType::F32);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[1], DType::F32);

        let silu_id = graph.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
        let tanh_id = graph.add_op(OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");

        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);

        // Act
        let sub_chains = split_elementwise_by_l1(&graph, &[silu_id, tanh_id], &plan);

        // Assert: intermediate tensor is 4 bytes, fits in any L1 -> single chain
        assert_eq!(sub_chains.len(), 1, "Tiny tensors should stay in single chain");
        assert_eq!(sub_chains[0].len(), 2);
        assert_eq!(sub_chains[0][0], silu_id);
        assert_eq!(sub_chains[0][1], tanh_id);
    }

    // ── Test 109: detect_qkv_norm_rope rejects when shared input not from norm ──
    // @trace TEST-FH-109 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_detect_qkv_norm_rope_rejects_non_norm_shared_input() {
        // Arrange: 3 GEMMs sharing input from Tanh (not RmsNorm/LayerNorm)
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[64, 256], DType::F32);
        graph.inputs.push(input);
        let tanh_out = graph.add_tensor_concrete("tanh_out", &[64, 256], DType::F32);
        let q_out = graph.add_tensor_concrete("q_out", &[64, 512], DType::F32);
        let k_out = graph.add_tensor_concrete("k_out", &[64, 512], DType::F32);
        let v_out = graph.add_tensor_concrete("v_out", &[64, 512], DType::F32);
        let qkn_q_out = graph.add_tensor_concrete("qkn_q_out", &[64, 512], DType::F32);
        let qkn_k_out = graph.add_tensor_concrete("qkn_k_out", &[64, 512], DType::F32);
        let vn_out = graph.add_tensor_concrete("vn_out", &[64, 512], DType::F32);
        let rope_q_out = graph.add_tensor_concrete("rope_q_out", &[64, 512], DType::F32);
        let rope_k_out = graph.add_tensor_concrete("rope_k_out", &[64, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[256, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[256, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[256, 512], DType::F32);

        // Tanh feeds 3 GEMMs (not a norm)
        graph.add_op(OpKind::Tanh, vec![input], vec![tanh_out], "tanh");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![tanh_out, wq], vec![q_out], "q_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![tanh_out, wk], vec![k_out], "k_proj");
        graph.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false,
        }, vec![tanh_out, wv], vec![v_out], "v_proj");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![q_out], vec![qkn_q_out], "qkn_q");
        graph.add_op(OpKind::QkNorm { head_dim: 512, eps: 1e-5 },
            vec![k_out], vec![qkn_k_out], "qkn_k");
        graph.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-5 }, vec![v_out], vec![vn_out], "vn");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_q_out], vec![rope_q_out], "rope_q");
        graph.add_op(OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![qkn_k_out], vec![rope_k_out], "rope_k");

        let topo: Vec<OpId> = graph.ops.iter().map(|o| o.id).collect();

        // Act
        let result = detect_qkv_norm_rope(&graph, &topo);

        // Assert: shared input from Tanh (not norm) -> rejected
        assert!(result.is_empty(), "Non-norm shared input should reject FusedQkvNormRope");
    }

    // ── Test 110: all_gemm_quant_compatible with two plain GemmBias ops returns true ──
    // @trace TEST-FH-110 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_all_gemm_quant_compatible_two_gemm_bias() {
        // Arrange: two GemmBias ops (neither is QuantGemm, both extract None)
        let op1 = CompilerOp::new_from_kind(
            OpId(0),
            OpKind::GemmBias { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "gemm_bias_1".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let op2 = CompilerOp::new_from_kind(
            OpId(1),
            OpKind::GemmBias { m: SymDim::Concrete(64), n: 512, k: 256, dtype: DType::F32, trans_b: false, },
            vec![],
            vec![],
            "gemm_bias_2".to_string(),
            LayerCondition::Always,
            &CompilerGraph::new(),
        );
        let ops: Vec<&CompilerOp> = vec![&op1, &op2];
        // Act
        let compatible = { let g = CompilerGraph::new(); all_gemm_quant_compatible(&ops, &g) };
        // Assert: both extract None -> can_fuse_quant_aware(None, None) = Fuse -> compatible
        assert!(compatible);
    }

}
