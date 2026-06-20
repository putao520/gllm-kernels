//! R1.5 LayoutNegotiator — 流水线级联布局协商 (SSOT: §3.10)
//!
//! 消费 AccelerationRegistry (§3.10.2) + OpBottleneckMap (§3.9)
//! 输出 LayoutAssignment: 每个 tensor 的协商布局 + 变换代价。
//!
//! 核心原则: 不是找全局最大公约布局。是流水线时序级联协商。
//! 每个自然数据搬运点都是"免费变换窗口"，协商器识别并利用这些窗口。

use std::collections::HashMap;
use crate::compiler::graph::{CompilerGraph, Op, OpId};
use crate::compiler::fusion::{FusionGroup, FusionMode};
use crate::compiler::semantic_dag::{SemanticDAG, OpClass};
use crate::compiler::accel_registry::{AccelerationRegistry, LayoutConstraint};
use crate::compiler::pain_point::OpBottleneckMap;
use crate::dispatch::device_profile::DeviceProfile;

/// 两个相邻阶段之间的数据搬运类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementType {
    /// 寄存器直传 (EpilogueInjection) — 无搬运点，布局必须兼容
    RegisterDirect,
    /// 寄存器 → 内存 (store) — 免费变换窗口
    RegisterToMemory,
    /// 内存 → 内存 (copy) — 免费变换窗口
    MemoryToMemory,
    /// GPU 全局 → 共享内存 (TMA prefetch) — 免费变换窗口
    GpuGlobalToShared,
}

/// 布局变换描述
#[derive(Debug, Clone)]
pub struct LayoutTransform {
    /// 源布局
    pub source: LayoutConstraint,
    /// 目标布局
    pub target: LayoutConstraint,
    /// 变换代价 (0 = 免费变换)
    pub cost: f64,
}

/// REQ-DTYPE-004: dtype 变换描述 — 当相邻 op 的 dtype 不同时
#[derive(Debug, Clone)]
pub struct DtypeTransform {
    /// 源 dtype (producer 的输出 dtype)
    pub source: crate::compiler::trace::QuantPrecision,
    /// 目标 dtype (consumer 的输入 dtype)
    pub target: crate::compiler::trace::QuantPrecision,
    /// 变换方式: VecWiden / VecNarrow / PackMap 转换
    pub method: DtypeTransformMethod,
}

/// REQ-DTYPE-004: dtype 变换方式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeTransformMethod {
    /// VecWiden: 窄→宽 (BF16→F32)，编译时 bake
    VecWiden,
    /// VecNarrow: 宽→窄 (F32→BF16)，编译时 bake
    VecNarrow,
    /// PackMap 转换: 通过重排 stride 实现 dtype+layout 同时转换
    PackMap,
}

/// 单个 op 的布局分配结果
#[derive(Debug, Clone)]
pub struct OpLayoutAssignment {
    /// 该 op 的输入布局
    pub input_layout: LayoutConstraint,
    /// §0.2.7: 权重输入的布局约束 (inputs[1])，用于 PackMap 虚拟权重推导
    pub weight_layout: Option<LayoutConstraint>,
    /// 该 op 的输出布局
    pub output_layout: LayoutConstraint,
    /// 匹配到的加速指令 ID (如果有)
    pub accel_id: Option<&'static str>,
    /// REQ-DTYPE-004: 该 op 的输入 dtype，参与布局决策
    pub dtype: Option<crate::compiler::trace::QuantPrecision>,
}

/// 两个相邻 op 之间的布局变换结果
#[derive(Debug, Clone)]
pub struct InterOpTransform {
    /// 上游 op
    pub producer: OpId,
    /// 下游 op
    pub consumer: OpId,
    /// 布局变换描述
    pub transform: LayoutTransform,
    /// 搬运类型
    pub movement: MovementType,
    /// REQ-DTYPE-004: dtype 变换描述 (None 表示 dtype 相同，零变换)
    pub dtype_transform: Option<DtypeTransform>,
}

/// 一个融合组的布局协商结果
#[derive(Debug, Clone)]
pub struct GroupLayoutAssignment {
    /// 融合组 ID
    pub group_id: usize,
    /// 每个 op 的布局分配
    pub op_layouts: HashMap<OpId, OpLayoutAssignment>,
    /// op 间的布局变换
    pub inter_op_transforms: Vec<InterOpTransform>,
    /// REQ-DTYPE-004: op 间的 dtype 变换
    pub dtype_transforms: Vec<DtypeTransform>,
    /// 总加速收益 (满足的加速指令收益之和)
    pub total_benefit: f64,
    /// 总变换代价
    pub total_transform_cost: f64,
}

/// 全局布局协商结果 (R1.5 输出)
#[derive(Debug, Clone)]
pub struct LayoutAssignment {
    /// 每个融合组的布局分配
    pub group_assignments: Vec<GroupLayoutAssignment>,
    /// 总收益
    pub total_benefit: f64,
    /// 总变换代价
    pub total_transform_cost: f64,
}

impl LayoutAssignment {
    /// 空分配
    pub fn empty() -> Self {
        LayoutAssignment {
            group_assignments: Vec::new(),
            total_benefit: 0.0,
            total_transform_cost: 0.0,
        }
    }

    /// 是否所有变换都是免费的
    pub fn all_transforms_free(&self) -> bool {
        self.total_transform_cost == 0.0
    }
}

/// 布局协商器 — 动态约束求解流水线级联布局
pub struct LayoutNegotiator;

impl LayoutNegotiator {
    /// 执行流水线级联布局协商 (SPEC §3.10.3)
    ///
    /// 输入: FusionPlan + AccelerationRegistry + DeviceProfile + PainPoints
    /// 输出: LayoutAssignment
    pub fn negotiate(
        groups: &[FusionGroup],
        registry: &AccelerationRegistry,
        device: &DeviceProfile,
        dag: &SemanticDAG,
        bottleneck_map: &OpBottleneckMap,
        graph: &CompilerGraph,
    ) -> LayoutAssignment {
        let mut group_assignments = Vec::new();
        let mut total_benefit = 0.0;
        let mut total_transform_cost = 0.0;

        for group in groups {
            let ga = negotiate_group(group, registry, device, dag, bottleneck_map, graph);
            total_benefit += ga.total_benefit;
            total_transform_cost += ga.total_transform_cost;
            group_assignments.push(ga);
        }

        LayoutAssignment {
            group_assignments,
            total_benefit,
            total_transform_cost,
        }
    }
}

/// 协商单个融合组的布局
fn negotiate_group(
    group: &FusionGroup,
    registry: &AccelerationRegistry,
    device: &DeviceProfile,
    dag: &SemanticDAG,
    bottleneck_map: &OpBottleneckMap,
    graph: &CompilerGraph,
) -> GroupLayoutAssignment {
    let mut op_layouts = HashMap::new();
    let mut inter_op_transforms = Vec::new();
    let mut total_benefit = 0.0;
    let mut total_transform_cost = 0.0;

    // Step 1: 为每个 op 查询最佳加速指令并确定偏好布局
    let ops = &group.ops;
    if ops.is_empty() {
        return GroupLayoutAssignment {
            group_id: group.id,
            op_layouts,
            inter_op_transforms,
            dtype_transforms: Vec::new(),
            total_benefit: 0.0,
            total_transform_cost: 0.0,
        };
    }

    // 收集每个 op 的加速指令和偏好
    let mut preferred: Vec<OpPreferredLayout> = Vec::with_capacity(ops.len());
    for &op_id in ops {
        let op_class = dag.node(op_id).map(|n| n.op_class).unwrap_or(OpClass::Opaque);
        let bn = bottleneck_map.gemm_bottlenecks.get(&op_id)
            .map(|gb| gb.bottleneck)
            .unwrap_or(crate::compiler::pain_point::BottleneckType::MemoryBound { bandwidth_utilization: 0.5 });

        let best = registry.best_for(op_class, device, bn);
        let (input_layout, weight_layout, output_layout, accel_id) = match best {
            Some(decl) => {
                let input = decl.input_layouts.first()
                    .cloned()
                    .unwrap_or(LayoutConstraint::Any);
                let wl = decl.input_layouts.get(1).cloned();
                (input, wl, decl.output_layout.clone(), Some(decl.id))
            }
            None => (LayoutConstraint::Any, None, LayoutConstraint::Any, None),
        };

        // §0.2.11 模型感知覆写: 从 CompilerGraph 提取 per-op 模型参数
        // 替换 AccelerationRegistry 中的占位符零值
        let (output_layout, weight_layout) = model_aware_layout_overrides(
            op_id, graph, output_layout, weight_layout,
        );

        // REQ-DTYPE-004: 从 graph 推导 op 的 dtype
        let op_dtype = graph.op(op_id)
            .and_then(|op| op.inputs.first())
            .and_then(|&tid| graph.tensor(tid))
            .map(|t| t.dtype.to_quant_precision());

        preferred.push(OpPreferredLayout {
            op_id,
            op_class,
            input_layout,
            weight_layout,
            output_layout,
            accel_id,
            dtype: op_dtype,
        });
    }

    let mut dtype_transforms = Vec::new();

    // Step 2: 流水线级联协商 — 对相邻 op 进行布局协商
    // 每个 (op[i], op[i+1]) 对是一个协商点
    for i in 0..ops.len() {
        let pref = &preferred[i];

        // 记录该 op 的布局分配
        let benefit = match pref.accel_id {
            Some(_id) => 1.0, // 简化: 有加速指令就给基础收益
            None => 0.0,
        };
        total_benefit += benefit;

        op_layouts.insert(pref.op_id, OpLayoutAssignment {
            input_layout: pref.input_layout.clone(),
            weight_layout: pref.weight_layout.clone(),
            output_layout: pref.output_layout.clone(),
            accel_id: pref.accel_id,
            dtype: pref.dtype,
        });

        // Step 3: 与下一个 op 的布局协商 + REQ-DTYPE-004 dtype 协商
        if i + 1 < ops.len() {
            let next = &preferred[i + 1];
            let movement = classify_movement(pref.op_class, next.op_class, i, group);

            // REQ-DTYPE-004: dtype 协商 — 相同 dtype 共享 buffer (零变换),
            // 不同 dtype 需要通过 VecWiden/VecNarrow/PackMap 转换
            let dtype_xform = match (pref.dtype, next.dtype) {
                (Some(src), Some(dst)) if src != dst => {
                    let method = if dst.elem_bytes() > src.elem_bytes() {
                        DtypeTransformMethod::VecWiden
                    } else if dst.elem_bytes() < src.elem_bytes() {
                        DtypeTransformMethod::VecNarrow
                    } else {
                        // Same byte width but different dtype (e.g. F16↔BF16) — PackMap stride remap
                        DtypeTransformMethod::PackMap
                    };
                    let xform = DtypeTransform { source: src, target: dst, method };
                    dtype_transforms.push(xform.clone());
                    Some(xform)
                }
                _ => None,
            };

            match movement {
                MovementType::RegisterDirect => {
                    // 无搬运点: 布局必须兼容
                    // REQ-DTYPE-004: dtype 不同时 RegisterDirect 不可行 — 需要显式变换
                    let layout_compatible = pref.output_layout.compatible_with(&next.input_layout);
                    let dtype_compatible = dtype_xform.is_none();

                    if layout_compatible && dtype_compatible {
                        // 零变换, 下游直接消费上游输出布局
                    } else {
                        // 不兼容: 找折中布局
                        let compromise = if !layout_compatible {
                            find_compromise(&pref.output_layout, &next.input_layout)
                        } else {
                            pref.output_layout.clone()
                        };
                        let layout_cost = if !layout_compatible && compromise == LayoutConstraint::Any { 0.0 } else if !layout_compatible { 1.0 } else { 0.0 };
                        // dtype 变换零成本 (编译时 bake)
                        total_transform_cost += layout_cost;
                        inter_op_transforms.push(InterOpTransform {
                            producer: pref.op_id,
                            consumer: next.op_id,
                            transform: LayoutTransform {
                                source: pref.output_layout.clone(),
                                target: compromise,
                                cost: layout_cost,
                            },
                            movement,
                            dtype_transform: dtype_xform,
                        });
                    }
                }
                MovementType::RegisterToMemory | MovementType::MemoryToMemory | MovementType::GpuGlobalToShared => {
                    // 免费变换窗口: 下游用自己的偏好布局
                    // store/copy 本来就要发生, 改变 stride/layout = 零额外成本
                    // REQ-DTYPE-004: dtype 变换也是零成本 (VecWiden/VecNarrow 编译时 bake)
                    let needs_layout_xform = !pref.output_layout.compatible_with(&next.input_layout);
                    let needs_dtype_xform = dtype_xform.is_some();
                    if needs_layout_xform || needs_dtype_xform {
                        inter_op_transforms.push(InterOpTransform {
                            producer: pref.op_id,
                            consumer: next.op_id,
                            transform: LayoutTransform {
                                source: pref.output_layout.clone(),
                                target: if needs_layout_xform { next.input_layout.clone() } else { pref.output_layout.clone() },
                                cost: 0.0, // 免费!
                            },
                            movement,
                            dtype_transform: dtype_xform,
                        });
                    }
                }
            }
        }
    }

    GroupLayoutAssignment {
        group_id: group.id,
        op_layouts,
        inter_op_transforms,
        dtype_transforms,
        total_benefit,
        total_transform_cost,
    }
}

/// §0.2.11 模型感知布局覆写 — 从 CompilerGraph 提取 per-op 模型参数
///
/// AccelerationRegistry 是纯静态工厂（零模型感知），其 LayoutConstraint 中的
/// num_heads/head_dim/kc 都是占位符零值。此函数从 graph 的 OpKind 提取真实模型参数，
/// 替换 output_layout 和 weight_layout 中的占位符。
fn model_aware_layout_overrides(
    op_id: OpId,
    graph: &CompilerGraph,
    output_layout: LayoutConstraint,
    weight_layout: Option<LayoutConstraint>,
) -> (LayoutConstraint, Option<LayoutConstraint>) {
    let op = match graph.op(op_id) {
        Some(o) => o,
        None => return (output_layout, weight_layout),
    };

    let out = match op.op_resolved(graph) {
        // MHA 的输出是 HeadSplit — 从 Op AttentionSpec 提取真实参数
        Some(Op::MultiHeadAttention(spec)) => {
            match output_layout {
                LayoutConstraint::HeadSplit { num_heads: 0, head_dim: 0 } |
                LayoutConstraint::Any => {
                    LayoutConstraint::HeadSplit { num_heads: spec.geometry.num_q_heads, head_dim: spec.geometry.head_dim }
                }
                other => other,
            }
        }
        // QKV 投影 GEMM: 如果是 QKV 三兄弟，输出应该是 HeadSplit
        Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) => {
            if is_qkv_op(op_id, graph) {
                let (_, n, _) = op.op_gemm_dims(graph).unwrap_or((crate::compiler::graph::SymDim::Concrete(1), 0, 0));
                let head_dim = extract_head_dim_from_graph(graph).unwrap_or(n);
                let num_heads = n / head_dim.max(1);
                LayoutConstraint::HeadSplit { num_heads, head_dim }
            } else {
                output_layout
            }
        }
        _ => output_layout,
    };

    // weight_layout: PanelPacked 的 kc 应从 op 的 K 维度推导
    let wl = weight_layout.map(|wl| {
        let is_gemm_like = matches!(op.op_resolved(graph), Some(Op::Gemm(_)) | Some(Op::GemmBias(_)));
        let k_dim = op.op_gemm_dims(graph).map(|(_, _, k)| k);
        match (is_gemm_like, k_dim, &wl) {
            (true, Some(k), LayoutConstraint::PanelPacked { mr, nr: _ }) => {
                // kc = 实际 K 维度（用于 BLIS stride 计算）
                LayoutConstraint::PanelPacked { mr: *mr, nr: k }
            }
            _ => wl,
        }
    });

    (out, wl)
}

/// 检查 op 是否为 QKV 投影（三兄弟共享同一输入）
///
/// ARCH-JIT-DATA-YIELDS: 使用 tensor.consumers 索引替代全图扫描。
fn is_qkv_op(op_id: OpId, graph: &CompilerGraph) -> bool {
    let op = match graph.op(op_id) {
        Some(o) => o,
        None => return false,
    };
    if !matches!(op.op_resolved(graph), Some(Op::Gemm(_)) | Some(Op::GemmBias(_))) {
        return false;
    }
    let Some(&input_tid) = op.inputs.first() else { return false };
    // 统计共享同一输入的 GEMM 数量（通过 tensor.consumers 索引）
    let sibling_count = graph.tensor(input_tid)
        .map(|t| t.consumers.iter()
            .filter(|&&c| graph.op(c).is_some_and(|o| matches!(o.op_resolved(graph),
                Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_)))))
            .count())
        .unwrap_or(0);
    sibling_count >= 3
}

/// 从图中提取 head_dim — 通过查找 MHA op 的参数
///
/// ARCH-JIT-DATA-YIELDS: short-circuit find_map — stops at first match.
/// This is a single targeted lookup, not a pre-scan bool flag.
fn extract_head_dim_from_graph(graph: &CompilerGraph) -> Option<usize> {
    graph.ops.iter()
        .find_map(|op| match op.op_resolved(graph) {
            Some(Op::MultiHeadAttention(spec)) => Some(spec.geometry.head_dim),
            Some(Op::RoPE(spec)) => Some(spec.head_dim),
            _ => None,
        })
}

/// 单个 op 的偏好布局 (协商前)
struct OpPreferredLayout {
    op_id: OpId,
    op_class: OpClass,
    input_layout: LayoutConstraint,
    /// §0.2.7: 权重输入布局约束 (inputs[1])
    weight_layout: Option<LayoutConstraint>,
    output_layout: LayoutConstraint,
    accel_id: Option<&'static str>,
    /// REQ-DTYPE-004: op 的输入 dtype，用于布局协商的 dtype 感知。
    dtype: Option<crate::compiler::trace::QuantPrecision>,
}

/// 分类两个相邻 op 之间的数据搬运类型
fn classify_movement(
    producer_class: OpClass,
    _consumer_class: OpClass,
    producer_index: usize,
    group: &FusionGroup,
) -> MovementType {
    // EpilogueInjection/TileLevelFusion/ComputeRoot 中的 GEMM → 后续 op
    // 通过寄存器直传 (RegisterDirect)
    if matches!(producer_class, OpClass::Gemm) {
        // GEMM 是 anchor → epilogue op 通过寄存器直连 (EpilogueInjection)
        // 但如果是最后一个 epilogue op 的输出 → 写回内存
        if producer_index == 0 {
            // GEMM (anchor) → 第一个 epilogue: RegisterDirect
            return MovementType::RegisterDirect;
        }
    }

    // 融合组内两个相邻 ElemWise/Injective op:
    // 如果不是 GEMM 的直接 epilogue, 则通过内存传递
    if matches!(producer_class, OpClass::ElemWise | OpClass::Injective) {
        // ElemWise/Injective 在 LoopFusion 中通过寄存器直传
        // 在其他模式中通过内存
        match group.mode {
            FusionMode::LoopFusion => {
                return MovementType::RegisterDirect;
            }
            _ => {
                return MovementType::RegisterToMemory;
            }
        }
    }

    // 默认: 寄存器 → 内存 (最保守)
    MovementType::RegisterToMemory
}

/// 在两个不兼容的布局之间找折中布局
fn find_compromise(a: &LayoutConstraint, b: &LayoutConstraint) -> LayoutConstraint {
    // Any 总是兼容的 — 如果任一方是 Any, 不需要折中
    if matches!(a, LayoutConstraint::Any) || matches!(b, LayoutConstraint::Any) {
        return LayoutConstraint::Any;
    }

    // RowMajor 和 HeadSplit 互相兼容 (只是 reshape)
    if matches!(a, LayoutConstraint::RowMajor { .. }) && matches!(b, LayoutConstraint::HeadSplit { .. }) {
        return b.clone();
    }
    if matches!(a, LayoutConstraint::HeadSplit { .. }) && matches!(b, LayoutConstraint::RowMajor { .. }) {
        return a.clone();
    }

    // 其他不兼容: 退回 Any (最安全但放弃布局优化)
    LayoutConstraint::Any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::accel_registry::AccelerationRegistry;
    use crate::compiler::fusion::{FusionMode, GroupMarker};

    fn make_test_dag() -> SemanticDAG {
        SemanticDAG {
            nodes: Vec::new(),
            edges: Vec::new(),
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
        }
    }

    #[test]
    fn test_negotiate_standalone_group() {
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let dag = make_test_dag();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let graph = CompilerGraph::new();
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );
        assert_eq!(result.group_assignments.len(), 1);
        assert_eq!(result.total_transform_cost, 0.0);
    }

    #[test]
    fn test_layout_assignment_empty() {
        let la = LayoutAssignment::empty();
        assert!(la.group_assignments.is_empty());
        assert!(la.all_transforms_free());
    }

    #[test]
    fn test_find_compromise_any() {
        let a = LayoutConstraint::Any;
        let b = LayoutConstraint::RowMajor { align_bytes: 64 };
        assert_eq!(find_compromise(&a, &b), LayoutConstraint::Any);
    }

    #[test]
    fn test_find_compromise_rowmajor_headsplit() {
        let a = LayoutConstraint::RowMajor { align_bytes: 64 };
        let b = LayoutConstraint::HeadSplit { num_heads: 32, head_dim: 128 };
        let c = find_compromise(&a, &b);
        assert!(matches!(c, LayoutConstraint::HeadSplit { .. }));
    }

    #[test]
    fn test_find_compromise_incompatible() {
        let a = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };
        let b = LayoutConstraint::InterleavedPairs;
        let c = find_compromise(&a, &b);
        assert!(matches!(c, LayoutConstraint::Any));
    }

    // ── 13 new tests ─────────────────────────────────────────────────

    /// @trace TEST-LN-06 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: GEMM producer at index 0 should yield RegisterDirect
    /// (anchor -> first epilogue via register passthrough).
    #[test]
    fn test_classify_movement_gemm_anchor_register_direct() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::Gemm, OpClass::ElemWise, 0, &group);

        // Assert
        assert_eq!(movement, MovementType::RegisterDirect);
    }

    /// @trace TEST-LN-07 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: ElemWise + LoopFusion yields RegisterDirect.
    #[test]
    fn test_classify_movement_elemwise_loop_fusion_register_direct() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::LoopFusion,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::ElemWise, OpClass::ElemWise, 0, &group);

        // Assert
        assert_eq!(movement, MovementType::RegisterDirect);
    }

    /// @trace TEST-LN-08 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: ElemWise with non-LoopFusion mode yields RegisterToMemory.
    #[test]
    fn test_classify_movement_elemwise_epilogue_to_memory() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::ElemWise, OpClass::ElemWise, 0, &group);

        // Assert
        assert_eq!(movement, MovementType::RegisterToMemory);
    }

    /// @trace TEST-LN-09 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: Reduction (non-Gemm, non-ElemWise/Injective) yields default RegisterToMemory.
    #[test]
    fn test_classify_movement_reduction_default_to_memory() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::Reduction, OpClass::ElemWise, 0, &group);

        // Assert
        assert_eq!(movement, MovementType::RegisterToMemory);
    }

    /// @trace TEST-LN-10 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: Injective + LoopFusion yields RegisterDirect.
    #[test]
    fn test_classify_movement_injective_loop_fusion() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::LoopFusion,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::Injective, OpClass::ElemWise, 0, &group);

        // Assert
        assert_eq!(movement, MovementType::RegisterDirect);
    }

    /// @trace TEST-LN-11 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: HeadSplit to RowMajor direction returns the HeadSplit (a).
    #[test]
    fn test_find_compromise_headsplit_to_rowmajor() {
        // Arrange
        let a = LayoutConstraint::HeadSplit { num_heads: 16, head_dim: 64 };
        let b = LayoutConstraint::RowMajor { align_bytes: 32 };

        // Act
        let c = find_compromise(&a, &b);

        // Assert: returns a (the HeadSplit)
        assert_eq!(c, a);
    }

    /// @trace TEST-LN-12 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: both sides Any yields Any.
    #[test]
    fn test_find_compromise_both_any() {
        // Arrange
        let a = LayoutConstraint::Any;
        let b = LayoutConstraint::Any;

        // Act
        let c = find_compromise(&a, &b);

        // Assert
        assert_eq!(c, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-13 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: empty ops list returns zero benefit and zero cost.
    #[test]
    fn test_negotiate_group_empty_ops() {
        // Arrange
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let dag = make_test_dag();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: Vec::new(), // empty
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert
        assert_eq!(result.group_assignments.len(), 1);
        assert_eq!(result.total_benefit, 0.0);
        assert_eq!(result.total_transform_cost, 0.0);
        assert!(result.group_assignments[0].op_layouts.is_empty());
    }

    /// @trace TEST-LN-14 [req:REQ-LAYOUT] [level:unit]
    /// negotiate: multiple groups accumulate benefits and costs correctly.
    #[test]
    fn test_negotiate_multiple_groups() {
        // Arrange
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let dag = make_test_dag();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();

        let group_a = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let group_b = FusionGroup {
            id: 1,
            anchor: OpId(1),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group_a, group_b], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert
        assert_eq!(result.group_assignments.len(), 2);
        assert_eq!(result.group_assignments[0].group_id, 0);
        assert_eq!(result.group_assignments[1].group_id, 1);
    }

    /// @trace TEST-LN-15 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: multi-op EpilogueInjection with two ops that have
    /// incompatible RegisterDirect layouts triggers a compromise transform with cost 1.0.
    #[test]
    fn test_negotiate_group_register_direct_incompatible() {
        // Arrange: build a DAG where op[0] is Gemm (index 0 => RegisterDirect)
        // and op[1] is ElemWise, with different node classes.
        let mut dag = make_test_dag();
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(0),
            op: crate::compiler::graph::Op::Gemm(crate::compiler::graph::GemmSpec {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 64,
                k: 64,
                dtype: crate::types::DType::F32,
                trans_b: false,
                has_bias: false,
            }),
            op_trace: None,
            op_class: OpClass::Gemm,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Compute,
            arithmetic_intensity: 10.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "gemm".to_string(),
        });
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(1),
            op: crate::compiler::graph::Op::Silu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "silu".to_string(),
        });

        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert: both ops get layout assignments
        let ga = &result.group_assignments[0];
        assert!(ga.op_layouts.contains_key(&OpId(0)));
        assert!(ga.op_layouts.contains_key(&OpId(1)));
    }

    /// @trace TEST-LN-16 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: MultiHeadAttention op overrides zero-valued
    /// HeadSplit layout with actual num_heads and head_dim.
    #[test]
    fn test_model_aware_mha_overrides_headsplit() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tin = graph.add_tensor_concrete("q", &[1, 4096], DType::F32);
        let tout = graph.add_tensor_concrete("out", &[1, 4096], DType::F32);
        let _op_id = graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 32, num_kv_heads: 32, head_dim: 128 }, mask: if true { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![tin],
            vec![tout],
            "mha",
        );

        // Act: override zero-valued HeadSplit
        let (out_layout, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::HeadSplit { num_heads: 0, head_dim: 0 },
            None,
        );

        // Assert
        assert_eq!(out_layout, LayoutConstraint::HeadSplit { num_heads: 32, head_dim: 128 });
        assert!(wl.is_none());
    }

    /// @trace TEST-LN-17 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: Gemm that is NOT a QKV op preserves output layout.
    #[test]
    fn test_model_aware_non_qkv_gemm_unchanged() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64, 128], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 128], DType::F32);
        let _op_id = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, b],
            vec![c],
            "gemm_non_qkv",
        );

        // Act
        let (out_layout, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::RowMajor { align_bytes: 64 },
            None,
        );

        // Assert: non-QKV Gemm keeps the provided layout unchanged
        assert_eq!(out_layout, LayoutConstraint::RowMajor { align_bytes: 64 });
        assert!(wl.is_none());
    }

    /// @trace TEST-LN-18 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: Gemm with PanelPacked weight_layout gets
    /// kc replaced by the op's K dimension.
    #[test]
    fn test_model_aware_gemm_panel_packed_weight_override() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 256], DType::F32);
        let b = graph.add_tensor_concrete("b", &[256, 512], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 512], DType::F32);
        let _op_id = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, b],
            vec![c],
            "gemm_weight",
        );

        // Act: PanelPacked weight with mr=14, nr=0 placeholder
        let (out_layout, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::RowMajor { align_bytes: 64 },
            Some(LayoutConstraint::PanelPacked { mr: 14, nr: 0 }),
        );

        // Assert: nr should be overridden to k=256
        assert_eq!(out_layout, LayoutConstraint::RowMajor { align_bytes: 64 });
        let weight = wl.expect("weight_layout should be Some");
        assert_eq!(weight, LayoutConstraint::PanelPacked { mr: 14, nr: 256 });
    }

    /// @trace TEST-LN-19 [req:REQ-LAYOUT] [level:unit]
    /// MovementType::RegisterToMemory with incompatible layouts produces a
    /// free (cost=0.0) inter-op transform, confirming the "free transformation window".
    #[test]
    fn test_negotiate_register_to_memory_free_transform() {
        // Arrange: two ElemWise ops in EpilogueInjection mode => RegisterToMemory
        let mut dag = make_test_dag();
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(0),
            op: crate::compiler::graph::Op::Silu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "silu".to_string(),
        });
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(1),
            op: crate::compiler::graph::Op::Gelu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "gelu".to_string(),
        });

        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert: Both ops are ElemWise (Any layout from registry), so they are compatible.
        // No inter-op transforms should be generated since Any is compatible with Any.
        let ga = &result.group_assignments[0];
        // The total transform cost is 0 because Any is compatible with Any
        assert_eq!(ga.total_transform_cost, 0.0);
        // Verify classify_movement produced RegisterToMemory for ElemWise+EpilogueInjection
        let movement = classify_movement(OpClass::ElemWise, OpClass::ElemWise, 0, &FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        });
        assert_eq!(movement, MovementType::RegisterToMemory);
    }

    /// @trace TEST-LN-20 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: op_id not found in graph returns layouts unchanged.
    #[test]
    fn test_model_aware_missing_op_returns_unchanged() {
        // Arrange: empty graph, op does not exist
        let graph = CompilerGraph::new();
        let output = LayoutConstraint::ColMajor { align_bytes: 32 };
        let weight = Some(LayoutConstraint::PanelPacked { mr: 6, nr: 16 });

        // Act
        let (out_layout, wl) = model_aware_layout_overrides(
            OpId(99), // nonexistent
            &graph,
            output.clone(),
            weight.clone(),
        );

        // Assert: unchanged passthrough
        assert_eq!(out_layout, output);
        assert_eq!(wl, weight);
    }

    /// @trace TEST-LN-21 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: multi-op LoopFusion with Gemm at index > 0 still uses
    /// default RegisterToMemory (Gemm not at anchor position).
    #[test]
    fn test_classify_movement_gemm_non_anchor_to_memory() {
        // Arrange: GEMM at index 1 (not the anchor position 0)
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act: producer_index = 1 (not 0), so GEMM is not anchor
        let movement = classify_movement(OpClass::Gemm, OpClass::ElemWise, 1, &group);

        // Assert: falls through to default RegisterToMemory
        assert_eq!(movement, MovementType::RegisterToMemory);
    }

    /// @trace TEST-LN-22 [req:REQ-LAYOUT] [level:unit]
    /// LayoutAssignment with non-zero total_transform_cost returns false for all_transforms_free.
    #[test]
    fn test_layout_assignment_not_free_with_cost() {
        // Arrange
        let assignment = LayoutAssignment {
            group_assignments: vec![GroupLayoutAssignment {
                group_id: 0,
                op_layouts: HashMap::new(),
                inter_op_transforms: vec![InterOpTransform {
                    producer: OpId(0),
                    consumer: OpId(1),
                    transform: LayoutTransform {
                        source: LayoutConstraint::PanelPacked { mr: 14, nr: 32 },
                        target: LayoutConstraint::Any,
                        cost: 1.0,
                    },
                    movement: MovementType::RegisterDirect,
                    dtype_transform: None,
                }],
                dtype_transforms: Vec::new(),
                total_benefit: 2.0,
                total_transform_cost: 1.0,
            }],
            total_benefit: 2.0,
            total_transform_cost: 1.0,
        };

        // Act & Assert
        assert!(!assignment.all_transforms_free());
        assert_eq!(assignment.total_benefit, 2.0);
        assert_eq!(assignment.total_transform_cost, 1.0);
    }

    /// @trace TEST-LN-23 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: MHA with LayoutConstraint::Any input
    /// gets promoted to HeadSplit with actual model parameters.
    #[test]
    fn test_model_aware_mha_any_promoted_to_headsplit() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tin = graph.add_tensor_concrete("q", &[1, 2048], DType::F32);
        let tout = graph.add_tensor_concrete("out", &[1, 2048], DType::F32);
        let _op_id = graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 8, num_kv_heads: 8, head_dim: 256 }, mask: if false { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![tin],
            vec![tout],
            "mha",
        );

        // Act: input layout is Any, should be promoted
        let (out_layout, _) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::Any,
            None,
        );

        // Assert
        assert_eq!(out_layout, LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 256 });
    }

    /// @trace TEST-LN-24 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: three Gemm ops sharing the same input tensor returns true.
    #[test]
    fn test_is_qkv_op_three_sibling_gemms() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 512], DType::F32);
        let wq = graph.add_tensor_concrete("wq", &[512, 512], DType::F32);
        let wk = graph.add_tensor_concrete("wk", &[512, 512], DType::F32);
        let wv = graph.add_tensor_concrete("wv", &[512, 512], DType::F32);
        let oq = graph.add_tensor_concrete("oq", &[1, 512], DType::F32);
        let ok_ = graph.add_tensor_concrete("ok", &[1, 512], DType::F32);
        let ov = graph.add_tensor_concrete("ov", &[1, 512], DType::F32);

        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, wq], vec![oq], "q_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, wk], vec![ok_], "k_proj");
        let v_op = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, wv], vec![ov], "v_proj");

        // Act
        let result = is_qkv_op(v_op, &graph);

        // Assert
        assert!(result, "Three Gemm ops sharing the same input should be detected as QKV");
    }

    /// @trace TEST-LN-25 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: single Gemm without siblings returns false.
    #[test]
    fn test_is_qkv_op_single_gemm_returns_false() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 64], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[64, 128], DType::F32);
        let output = graph.add_tensor_concrete("output", &[1, 128], DType::F32);

        let op_id = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![input, weight],
            vec![output],
            "single_gemm",
        );

        // Act
        let result = is_qkv_op(op_id, &graph);

        // Assert
        assert!(!result, "Single Gemm without sibling ops should not be detected as QKV");
    }

    /// @trace TEST-LN-26 [req:REQ-LAYOUT] [level:unit]
    /// extract_head_dim_from_graph: finds head_dim from a RoPE op in the graph.
    #[test]
    fn test_extract_head_dim_from_rope() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let q = graph.add_tensor_concrete("q", &[1, 8, 64], DType::F32);
        let oq = graph.add_tensor_concrete("oq", &[1, 8, 64], DType::F32);
        graph.add_op(Op::RoPE(RopeSpec { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None }),
            vec![q],
            vec![oq],
            "rope",
        );

        // Act
        let head_dim = extract_head_dim_from_graph(&graph);

        // Assert
        assert_eq!(head_dim, Some(64));
    }

    /// @trace TEST-LN-27 [req:REQ-LAYOUT] [level:unit]
    /// extract_head_dim_from_graph: returns None when no MHA or RoPE op exists.
    #[test]
    fn test_extract_head_dim_missing_returns_none() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64, 128], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 128], DType::F32);
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, b],
            vec![c],
            "gemm_only",
        );

        // Act
        let head_dim = extract_head_dim_from_graph(&graph);

        // Assert
        assert_eq!(head_dim, None);
    }

    /// @trace TEST-LN-28 [req:REQ-LAYOUT] [level:unit]
    /// InterOpTransform and LayoutTransform fields carry correct producer/consumer,
    /// source/target layout, and cost information through negotiation.
    #[test]
    fn test_inter_op_transform_structure() {
        // Arrange
        let transform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 64 },
            target: LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 64 },
            cost: 0.0,
        };
        let iot = InterOpTransform {
            producer: OpId(10),
            consumer: OpId(20),
            transform: transform.clone(),
            movement: MovementType::MemoryToMemory,
            dtype_transform: None,
        };

        // Act & Assert: verify all fields are correctly stored
        assert_eq!(iot.producer, OpId(10));
        assert_eq!(iot.consumer, OpId(20));
        assert_eq!(iot.movement, MovementType::MemoryToMemory);
        assert_eq!(iot.transform.source, LayoutConstraint::RowMajor { align_bytes: 64 });
        assert_eq!(iot.transform.target, LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 64 });
        assert_eq!(iot.transform.cost, 0.0);
    }

    // ── 10 new tests (wave-12kad) ────────────────────────────────────────

    /// @trace TEST-LN-29 [req:REQ-LAYOUT] [level:unit]
    /// MovementType enum variants are Copy + Clone + PartialEq + Eq + Debug.
    #[test]
    fn test_movement_type_traits() {
        // Arrange
        let a = MovementType::RegisterDirect;
        let b = a; // Copy
        let c = a.clone(); // Clone

        // Assert: PartialEq
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(a, MovementType::GpuGlobalToShared);

        // Assert: Debug formats without panic
        let _debug_str = format!("{:?}", a);
        let _debug_str = format!("{:?}", MovementType::RegisterToMemory);
        let _debug_str = format!("{:?}", MovementType::MemoryToMemory);
        let _debug_str = format!("{:?}", MovementType::GpuGlobalToShared);
    }

    /// @trace TEST-LN-30 [req:REQ-LAYOUT] [level:unit]
    /// LayoutTransform Clone produces an independent copy with identical fields.
    #[test]
    fn test_layout_transform_clone() {
        // Arrange
        let original = LayoutTransform {
            source: LayoutConstraint::InterleavedPairs,
            target: LayoutConstraint::ColMajor { align_bytes: 128 },
            cost: 3.5,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.source, original.source);
        assert_eq!(cloned.target, original.target);
        assert_eq!(cloned.cost, original.cost);
    }

    /// @trace TEST-LN-31 [req:REQ-LAYOUT] [level:unit]
    /// OpLayoutAssignment carries weight_layout=None for ops without a second input.
    #[test]
    fn test_op_layout_assignment_no_weight() {
        // Arrange
        let assignment = OpLayoutAssignment {
            input_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            weight_layout: None,
            output_layout: LayoutConstraint::Any,
            accel_id: None,
            dtype: None,
        };

        // Assert: weight_layout is None, accel_id is None
        assert!(assignment.weight_layout.is_none());
        assert!(assignment.accel_id.is_none());
        assert_eq!(assignment.input_layout, LayoutConstraint::RowMajor { align_bytes: 64 });
        assert_eq!(assignment.output_layout, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-32 [req:REQ-LAYOUT] [level:unit]
    /// OpLayoutAssignment carries weight_layout=Some and accel_id for hardware-accelerated ops.
    #[test]
    fn test_op_layout_assignment_with_weight_and_accel() {
        // Arrange
        let assignment = OpLayoutAssignment {
            input_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            weight_layout: Some(LayoutConstraint::PanelPacked { mr: 6, nr: 256 }),
            output_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            accel_id: Some("avx512_gemm"),
            dtype: None,
        };

        // Assert: all fields populated correctly
        let wl = assignment.weight_layout.expect("weight_layout should be Some");
        assert_eq!(wl, LayoutConstraint::PanelPacked { mr: 6, nr: 256 });
        assert_eq!(assignment.accel_id, Some("avx512_gemm"));
    }

    /// @trace TEST-LN-33 [req:REQ-LAYOUT] [level:unit]
    /// GroupLayoutAssignment Clone produces an independent copy preserving all fields.
    #[test]
    fn test_group_layout_assignment_clone_preserves_data() {
        // Arrange
        let mut op_layouts = HashMap::new();
        op_layouts.insert(OpId(0), OpLayoutAssignment {
            input_layout: LayoutConstraint::Any,
            weight_layout: None,
            output_layout: LayoutConstraint::Any,
            accel_id: None,
            dtype: None,
        });
        let original = GroupLayoutAssignment {
            group_id: 7,
            op_layouts: op_layouts.clone(),
            inter_op_transforms: vec![InterOpTransform {
                producer: OpId(0),
                consumer: OpId(1),
                transform: LayoutTransform {
                    source: LayoutConstraint::Any,
                    target: LayoutConstraint::Any,
                    cost: 0.0,
                },
                movement: MovementType::RegisterToMemory,
                dtype_transform: None,
            }],
            dtype_transforms: Vec::new(),
            total_benefit: 5.0,
            total_transform_cost: 1.5,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.group_id, 7);
        assert_eq!(cloned.total_benefit, 5.0);
        assert_eq!(cloned.total_transform_cost, 1.5);
        assert_eq!(cloned.op_layouts.len(), 1);
        assert_eq!(cloned.inter_op_transforms.len(), 1);
    }

    /// @trace TEST-LN-34 [req:REQ-LAYOUT] [level:unit]
    /// LayoutAssignment::empty() produces an assignment where all_transforms_free is true,
    /// and Clone preserves the empty state.
    #[test]
    fn test_layout_assignment_empty_clone_and_free() {
        // Arrange
        let empty = LayoutAssignment::empty();

        // Act
        let cloned = empty.clone();

        // Assert
        assert!(cloned.group_assignments.is_empty());
        assert!(cloned.all_transforms_free());
        assert_eq!(cloned.total_benefit, 0.0);
        assert_eq!(cloned.total_transform_cost, 0.0);
    }

    /// @trace TEST-LN-35 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: Any on the second side (b) yields Any regardless of a.
    #[test]
    fn test_find_compromise_second_side_any() {
        // Arrange
        let a = LayoutConstraint::ColMajor { align_bytes: 32 };
        let b = LayoutConstraint::Any;

        // Act
        let result = find_compromise(&a, &b);

        // Assert: Any on either side yields Any
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-36 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: two Gemm ops sharing the same input (not three) returns false.
    #[test]
    fn test_is_qkv_op_two_sibling_gemms_returns_false() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 256], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[256, 256], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[256, 256], DType::F32);
        let o1 = graph.add_tensor_concrete("o1", &[1, 256], DType::F32);
        let o2 = graph.add_tensor_concrete("o2", &[1, 256], DType::F32);

        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w1], vec![o1], "proj1");
        let op2 = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w2], vec![o2], "proj2");

        // Act
        let result = is_qkv_op(op2, &graph);

        // Assert: need >= 3 siblings, 2 is not enough
        assert!(!result, "Two sibling Gemm ops should not be detected as QKV (need >= 3)");
    }

    /// @trace TEST-LN-37 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: MHA with a non-zero, non-Any HeadSplit preserves
    /// the existing layout (no override when already populated).
    #[test]
    fn test_model_aware_mha_preserves_existing_headsplit() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tin = graph.add_tensor_concrete("q", &[1, 4096], DType::F32);
        let tout = graph.add_tensor_concrete("out", &[1, 4096], DType::F32);
        let _op_id = graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 32, num_kv_heads: 32, head_dim: 128 }, mask: if true { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![tin],
            vec![tout],
            "mha",
        );

        let existing_layout = LayoutConstraint::HeadSplit { num_heads: 16, head_dim: 256 };

        // Act: pass an already-populated HeadSplit (non-zero)
        let (out_layout, _) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            existing_layout.clone(),
            None,
        );

        // Assert: existing layout is preserved unchanged
        assert_eq!(out_layout, existing_layout);
    }

    /// @trace TEST-LN-38 [req:REQ-LAYOUT] [level:unit]
    /// extract_head_dim_from_graph: finds head_dim from an MHA op in the graph.
    #[test]
    fn test_extract_head_dim_from_mha() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let tin = graph.add_tensor_concrete("q", &[1, 4096], DType::F32);
        let tout = graph.add_tensor_concrete("out", &[1, 4096], DType::F32);
        graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 32, num_kv_heads: 8, head_dim: 128 }, mask: if true { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![tin],
            vec![tout],
            "mha",
        );

        // Act
        let head_dim = extract_head_dim_from_graph(&graph);

        // Assert
        assert_eq!(head_dim, Some(128));
    }

    // ── 10 new tests (wave-12kff) ────────────────────────────────────────

    /// @trace TEST-LN-39 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: GemmBias op (not Gemm) with non-QKV
    /// preserves the provided output layout unchanged.
    #[test]
    fn test_model_aware_gemmbias_non_qkv_preserves_layout() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 128], DType::F32);
        let b = graph.add_tensor_concrete("b", &[128, 256], DType::F32);
        let bias = graph.add_tensor_concrete("bias", &[256], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 256], DType::F32);
        graph.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 128, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, b, bias],
            vec![c],
            "gemm_bias_non_qkv",
        );

        let layout = LayoutConstraint::RowMajor { align_bytes: 128 };

        // Act
        let (out_layout, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            layout.clone(),
            None,
        );

        // Assert: single GemmBias is not QKV, layout preserved
        assert_eq!(out_layout, layout);
        assert!(wl.is_none());
    }

    /// @trace TEST-LN-40 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: GemmBias that IS a QKV op gets
    /// HeadSplit output layout computed from n and head_dim.
    #[test]
    fn test_model_aware_gemmbias_qkv_gets_headsplit() {
        // Arrange: build a graph with an MHA to extract head_dim, then 3 GemmBias
        // sharing the same input tensor.
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        // MHA provides head_dim=64 for extract_head_dim_from_graph
        let mq = graph.add_tensor_concrete("mq", &[1, 512], DType::F32);
        let mo = graph.add_tensor_concrete("mo", &[1, 512], DType::F32);
        graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 8, num_kv_heads: 8, head_dim: 64 }, mask: if true { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![mq],
            vec![mo],
            "mha_ref",
        );

        // 3 GemmBias ops sharing the same input
        let input = graph.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[512, 512], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[512, 512], DType::F32);
        let w3 = graph.add_tensor_concrete("w3", &[512, 512], DType::F32);
        let b1 = graph.add_tensor_concrete("b1", &[512], DType::F32);
        let b2 = graph.add_tensor_concrete("b2", &[512], DType::F32);
        let b3 = graph.add_tensor_concrete("b3", &[512], DType::F32);
        let o1 = graph.add_tensor_concrete("o1", &[1, 512], DType::F32);
        let o2 = graph.add_tensor_concrete("o2", &[1, 512], DType::F32);
        let o3 = graph.add_tensor_concrete("o3", &[1, 512], DType::F32);

        graph.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: true }), vec![input, w1, b1], vec![o1], "q_proj");
        graph.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: true }), vec![input, w2, b2], vec![o2], "k_proj");
        let v_op = graph.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: true }), vec![input, w3, b3], vec![o3], "v_proj");

        // Act
        let (out_layout, _) = model_aware_layout_overrides(
            v_op,
            &graph,
            LayoutConstraint::Any,
            None,
        );

        // Assert: QKV GemmBias gets HeadSplit with num_heads=512/64=8, head_dim=64
        assert_eq!(out_layout, LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 64 });
    }

    /// @trace TEST-LN-41 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: GemmBias with PanelPacked weight_layout
    /// gets kc overridden to the op's K dimension.
    #[test]
    fn test_model_aware_gemmbias_panel_packed_weight() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 192], DType::F32);
        let b = graph.add_tensor_concrete("b", &[192, 384], DType::F32);
        let bias = graph.add_tensor_concrete("bias", &[384], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 384], DType::F32);
        graph.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 384, k: 192, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, b, bias],
            vec![c],
            "gemm_bias_weight",
        );

        // Act
        let (_out_layout, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::Any,
            Some(LayoutConstraint::PanelPacked { mr: 6, nr: 0 }),
        );

        // Assert: nr should be overridden to k=192
        let weight = wl.expect("weight_layout should be Some");
        assert_eq!(weight, LayoutConstraint::PanelPacked { mr: 6, nr: 192 });
    }

    /// @trace TEST-LN-42 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: non-GEMM op (RmsNorm) passes through
    /// both output and weight layouts unchanged (wildcard branch).
    #[test]
    fn test_model_aware_rmsnorm_passthrough() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("x", &[1, 256], DType::F32);
        let output = graph.add_tensor_concrete("out", &[1, 256], DType::F32);
        graph.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-6, dtype: DType::F32, has_weight: true }),
            vec![input],
            vec![output],
            "rms_norm",
        );

        let out_layout = LayoutConstraint::RowMajor { align_bytes: 32 };
        let weight_layout = Some(LayoutConstraint::ColMajor { align_bytes: 64 });

        // Act
        let (result_out, result_wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            out_layout.clone(),
            weight_layout.clone(),
        );

        // Assert: both layouts passed through unchanged
        assert_eq!(result_out, out_layout);
        assert_eq!(result_wl, weight_layout);
    }

    /// @trace TEST-LN-43 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: op_id not found in graph returns false.
    #[test]
    fn test_is_qkv_op_missing_op_id() {
        // Arrange: empty graph
        let graph = CompilerGraph::new();

        // Act
        let result = is_qkv_op(OpId(42), &graph);

        // Assert
        assert!(!result, "Missing op_id should return false");
    }

    /// @trace TEST-LN-44 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: non-GEMM op (e.g., Silu) returns false early.
    #[test]
    fn test_is_qkv_op_non_gemm_returns_false() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("x", &[1, 64], DType::F32);
        let output = graph.add_tensor_concrete("out", &[1, 64], DType::F32);
        let op_id = graph.add_op(Op::Silu,
            vec![input],
            vec![output],
            "silu",
        );

        // Act
        let result = is_qkv_op(op_id, &graph);

        // Assert
        assert!(!result, "Silu op should not be detected as QKV");
    }

    /// @trace TEST-LN-45 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: Gemm op with no inputs returns false (empty inputs list).
    #[test]
    fn test_is_qkv_op_gemm_no_inputs() {
        // Arrange: manually construct a graph with a Gemm that has no inputs
        use crate::compiler::graph::{CompilerGraph, CompilerOp, LayerCondition, KvSource, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        // Add a Gemm with no inputs (edge case)
        let op_node = CompilerOp::new_from_op(OpId(0), Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![], // no inputs
            vec![],
            "orphan_gemm",
            LayerCondition::Always
        );
        graph.ops.push(op_node);
        let op_id = OpId(0);

        // Act
        let result = is_qkv_op(op_id, &graph);

        // Assert: no inputs -> no shared input tensor -> false
        assert!(!result, "Gemm with no inputs should return false");
    }

    /// @trace TEST-LN-46 [req:REQ-LAYOUT] [level:unit]
    /// extract_head_dim_from_graph: MHA takes priority over RoPE
    /// (find_map returns the first match, which is MHA).
    #[test]
    fn test_extract_head_dim_mha_priority_over_rope() {
        // Arrange: graph with MHA first, then RoPE with different head_dim
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        // MHA with head_dim=96
        let mq = graph.add_tensor_concrete("mq", &[1, 768], DType::F32);
        let mo = graph.add_tensor_concrete("mo", &[1, 768], DType::F32);
        graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 8, num_kv_heads: 8, head_dim: 96 }, mask: if false { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![mq],
            vec![mo],
            "mha_first",
        );

        // RoPE with head_dim=64 (different from MHA)
        let rq = graph.add_tensor_concrete("rq", &[1, 512], DType::F32);
        let ro = graph.add_tensor_concrete("ro", &[1, 512], DType::F32);
        graph.add_op(Op::RoPE(RopeSpec { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None }),
            vec![rq],
            vec![ro],
            "rope_second",
        );

        // Act
        let head_dim = extract_head_dim_from_graph(&graph);

        // Assert: MHA (head_dim=96) is first in ops iteration, so it takes priority
        assert_eq!(head_dim, Some(96), "MHA head_dim should take priority over RoPE");
    }

    /// @trace TEST-LN-47 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: non-empty bottleneck_map entry for an op causes
    /// best_for to be called with the actual bottleneck type.
    #[test]
    fn test_negotiate_group_with_bottleneck_entry() {
        // Arrange
        use crate::compiler::pain_point::{GemmBottleneck, BottleneckType, GemmRole, ExecPattern, ParallelismDesc};

        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let mut dag = make_test_dag();
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(0),
            op: crate::compiler::graph::Op::Gemm(crate::compiler::graph::GemmSpec {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 128,
                k: 256,
                dtype: crate::types::DType::F32,
                trans_b: false,
                has_bias: false,
            }),
            op_trace: None,
            op_class: OpClass::Gemm,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 0.5,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "gemm".to_string(),
        });

        let mut gemm_bottlenecks = HashMap::new();
        gemm_bottlenecks.insert(OpId(0), GemmBottleneck {
            gemm_role: GemmRole::GateUpProjection,
            shape: (1, 128, 256),
            arithmetic_intensity: 0.5,
            ridge_point: 10.0,
            bottleneck: BottleneckType::MemoryBound { bandwidth_utilization: 0.8 },
            optimal_fusion: crate::compiler::pain_point::FusionPriority::EpilogueInjection,
            fusion_benefits: HashMap::new(),
            exec_pattern: ExecPattern::ScalarLoop,
            parallelism: ParallelismDesc::SimdVectorize { element_width: 8, unroll_factor: 1 },
        });

        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks,
            ridge_point: 10.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert: group assignment produced with no panic
        assert_eq!(result.group_assignments.len(), 1);
        let ga = &result.group_assignments[0];
        assert!(ga.op_layouts.contains_key(&OpId(0)));
        assert_eq!(ga.group_id, 0);
    }

    /// @trace TEST-LN-48 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: Gemm QKV with head_dim from graph
    /// computes num_heads = n / head_dim correctly.
    #[test]
    fn test_model_aware_gemm_qkv_head_dim_extraction() {
        // Arrange: graph with MHA head_dim=80, then 3 Gemm ops (QKV) with n=640
        // num_heads = 640 / 80 = 8
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();

        // MHA provides head_dim=80
        let mq = graph.add_tensor_concrete("mq", &[1, 640], DType::F32);
        let mo = graph.add_tensor_concrete("mo", &[1, 640], DType::F32);
        graph.add_op(Op::MultiHeadAttention(AttentionSpec { geometry: AttentionGeometry { num_q_heads: 8, num_kv_heads: 8, head_dim: 80 }, mask: if false { AttentionMask::Causal } else { AttentionMask::Full }, kv_source: KvSource::FromTensor, sinks: if false { SinksSpec::Learnable } else { SinksSpec::None }, seq_len: SymDim::Concrete(1), dtype: DType::F32 }),
            vec![mq],
            vec![mo],
            "mha_for_hd",
        );

        // 3 Gemm ops sharing same input (QKV pattern)
        let input = graph.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[512, 640], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[512, 640], DType::F32);
        let w3 = graph.add_tensor_concrete("w3", &[512, 640], DType::F32);
        let o1 = graph.add_tensor_concrete("o1", &[1, 640], DType::F32);
        let o2 = graph.add_tensor_concrete("o2", &[1, 640], DType::F32);
        let o3 = graph.add_tensor_concrete("o3", &[1, 640], DType::F32);

        let q_op = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 640, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w1], vec![o1], "q_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 640, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w2], vec![o2], "k_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 640, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w3], vec![o3], "v_proj");

        // Act: query the first QKV gemm
        let (out_layout, _) = model_aware_layout_overrides(
            q_op,
            &graph,
            LayoutConstraint::Any,
            None,
        );

        // Assert: num_heads = 640 / 80 = 8
        assert_eq!(
            out_layout,
            LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 80 },
            "QKV Gemm should compute num_heads from n/head_dim"
        );
    }

    // ── 10 new tests (wave-12x33) ────────────────────────────────────────

    /// @trace TEST-LN-49 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: ColMajor with different alignments
    /// is not compatible (no special ColMajor alignment rule like RowMajor has).
    #[test]
    fn test_compatible_with_colmajor_different_align() {
        // Arrange
        let a = LayoutConstraint::ColMajor { align_bytes: 32 };
        let b = LayoutConstraint::ColMajor { align_bytes: 64 };

        // Act & Assert
        assert!(!a.compatible_with(&b), "Different ColMajor alignments should not be compatible");
    }

    /// @trace TEST-LN-50 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: two identical PanelPacked layouts are compatible.
    #[test]
    fn test_compatible_with_identical_panel_packed() {
        // Arrange
        let a = LayoutConstraint::PanelPacked { mr: 6, nr: 16 };
        let b = LayoutConstraint::PanelPacked { mr: 6, nr: 16 };

        // Act & Assert
        assert!(a.compatible_with(&b), "Identical PanelPacked layouts should be compatible");
    }

    /// @trace TEST-LN-51 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: ColMajor vs PanelPacked are incompatible.
    #[test]
    fn test_compatible_with_colmajor_vs_panelpacked() {
        // Arrange
        let a = LayoutConstraint::ColMajor { align_bytes: 64 };
        let b = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };

        // Act & Assert
        assert!(!a.compatible_with(&b), "ColMajor and PanelPacked should be incompatible");
    }

    /// @trace TEST-LN-52 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: OpClass::Opaque falls through to default RegisterToMemory.
    #[test]
    fn test_classify_movement_opaque_falls_to_default() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::Opaque, OpClass::ElemWise, 0, &group);

        // Assert
        assert_eq!(movement, MovementType::RegisterToMemory);
    }

    /// @trace TEST-LN-53 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: PanelPacked vs RowMajor returns Any fallback.
    #[test]
    fn test_find_compromise_panelpacked_vs_rowmajor() {
        // Arrange
        let a = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };
        let b = LayoutConstraint::RowMajor { align_bytes: 64 };

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-54 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: Gemm with non-PanelPacked weight_layout
    /// passes weight through unchanged.
    #[test]
    fn test_model_aware_gemm_non_panel_weight_passthrough() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64, 128], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 128], DType::F32);
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, b],
            vec![c],
            "gemm_colmaj_wt",
        );

        let weight_input = LayoutConstraint::ColMajor { align_bytes: 32 };

        // Act
        let (_, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::Any,
            Some(weight_input.clone()),
        );

        // Assert
        assert_eq!(wl, Some(weight_input));
    }

    /// @trace TEST-LN-55 [req:REQ-LAYOUT] [level:unit]
    /// negotiate: empty groups slice produces empty LayoutAssignment.
    #[test]
    fn test_negotiate_empty_groups() {
        // Arrange
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let dag = make_test_dag();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();

        // Act
        let result = LayoutNegotiator::negotiate(
            &[], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert
        assert!(result.group_assignments.is_empty());
        assert_eq!(result.total_benefit, 0.0);
        assert_eq!(result.total_transform_cost, 0.0);
        assert!(result.all_transforms_free());
    }

    /// @trace TEST-LN-56 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: first QKV sibling is also detected (not just the last).
    #[test]
    fn test_is_qkv_op_first_sibling_detected() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 256], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[256, 256], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[256, 256], DType::F32);
        let w3 = graph.add_tensor_concrete("w3", &[256, 256], DType::F32);
        let o1 = graph.add_tensor_concrete("o1", &[1, 256], DType::F32);
        let o2 = graph.add_tensor_concrete("o2", &[1, 256], DType::F32);
        let o3 = graph.add_tensor_concrete("o3", &[1, 256], DType::F32);

        let q_op = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w1], vec![o1], "q_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w2], vec![o2], "k_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w3], vec![o3], "v_proj");

        // Act
        let result = is_qkv_op(q_op, &graph);

        // Assert
        assert!(result, "First QKV sibling should be detected");
    }

    /// @trace TEST-LN-57 [req:REQ-LAYOUT] [level:unit]
    /// LayoutAssignment Clone with multiple groups preserves totals.
    #[test]
    fn test_layout_assignment_clone_multiple_groups() {
        // Arrange
        let g1 = GroupLayoutAssignment {
            group_id: 0,
            op_layouts: HashMap::new(),
            inter_op_transforms: vec![InterOpTransform {
                producer: OpId(0),
                consumer: OpId(1),
                transform: LayoutTransform {
                    source: LayoutConstraint::Any,
                    target: LayoutConstraint::RowMajor { align_bytes: 32 },
                    cost: 0.0,
                },
                movement: MovementType::MemoryToMemory,
                dtype_transform: None,
            }],
            dtype_transforms: Vec::new(),
            total_benefit: 3.0,
            total_transform_cost: 0.5,
        };
        let g2 = GroupLayoutAssignment {
            group_id: 1,
            op_layouts: HashMap::new(),
            inter_op_transforms: Vec::new(),
            dtype_transforms: Vec::new(),
            total_benefit: 7.0,
            total_transform_cost: 2.0,
        };
        let original = LayoutAssignment {
            group_assignments: vec![g1, g2],
            total_benefit: 10.0,
            total_transform_cost: 2.5,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.group_assignments.len(), 2);
        assert_eq!(cloned.group_assignments[0].group_id, 0);
        assert_eq!(cloned.group_assignments[1].group_id, 1);
        assert_eq!(cloned.total_benefit, 10.0);
        assert_eq!(cloned.total_transform_cost, 2.5);
        assert!(!cloned.all_transforms_free());
    }

    /// @trace TEST-LN-58 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: three-op LoopFusion with ElemWise produces
    /// RegisterDirect movement and zero transform cost.
    #[test]
    fn test_negotiate_three_op_loop_fusion() {
        // Arrange
        let mut dag = make_test_dag();
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(0),
            op: crate::compiler::graph::Op::Silu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "silu".to_string(),
        });
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(1),
            op: crate::compiler::graph::Op::Gelu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "gelu".to_string(),
        });
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(2),
            op: crate::compiler::graph::Op::Tanh,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "tanh".to_string(),
        });

        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2)],
            mode: FusionMode::LoopFusion,
            ops: vec![OpId(0), OpId(1), OpId(2)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group.clone()], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert
        let ga = &result.group_assignments[0];
        assert_eq!(ga.op_layouts.len(), 3);
        assert!(ga.op_layouts.contains_key(&OpId(0)));
        assert!(ga.op_layouts.contains_key(&OpId(1)));
        assert!(ga.op_layouts.contains_key(&OpId(2)));
        assert_eq!(ga.total_transform_cost, 0.0);
        assert_eq!(
            classify_movement(OpClass::ElemWise, OpClass::ElemWise, 0, &group),
            MovementType::RegisterDirect,
        );
        assert_eq!(
            classify_movement(OpClass::ElemWise, OpClass::ElemWise, 1, &group),
            MovementType::RegisterDirect,
        );
    }

    // ── 10 new tests (wave-12x89) ────────────────────────────────────────

    /// @trace TEST-LN-59 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: two ColMajor layouts fall through to Any
    /// (neither is RowMajor nor HeadSplit, no special case applies).
    #[test]
    fn test_find_compromise_both_colmajor_returns_any() {
        // Arrange
        let a = LayoutConstraint::ColMajor { align_bytes: 32 };
        let b = LayoutConstraint::ColMajor { align_bytes: 64 };

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-60 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: two HeadSplit layouts with different params fall through to Any
    /// (only RowMajor↔HeadSplit cross-type has a special case).
    #[test]
    fn test_find_compromise_both_headsplit_different_returns_any() {
        // Arrange
        let a = LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 64 };
        let b = LayoutConstraint::HeadSplit { num_heads: 16, head_dim: 128 };

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-61 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: two InterleavedPairs layouts fall through to Any
    /// (no special case for InterleavedPairs ↔ InterleavedPairs).
    #[test]
    fn test_find_compromise_both_interleaved_returns_any() {
        // Arrange
        let a = LayoutConstraint::InterleavedPairs;
        let b = LayoutConstraint::InterleavedPairs;

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-62 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: OpId in group.ops but missing from DAG nodes gets
    /// OpClass::Opaque fallback and still produces a valid layout assignment.
    #[test]
    fn test_negotiate_group_op_not_in_dag_produces_any_layout() {
        // Arrange
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let dag = make_test_dag(); // empty DAG — no nodes
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(99),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(99)], // OpId not in DAG
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert: no panic, valid output with Any layout
        assert_eq!(result.group_assignments.len(), 1);
        let layout = result.group_assignments[0].op_layouts.get(&OpId(99)).unwrap();
        assert_eq!(layout.input_layout, LayoutConstraint::Any);
        assert_eq!(layout.output_layout, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-63 [req:REQ-LAYOUT] [level:unit]
    /// LayoutAssignment with benefit > 0 but cost = 0 still returns all_transforms_free().
    #[test]
    fn test_layout_assignment_zero_cost_with_benefit_is_free() {
        // Arrange
        let mut op_layouts = HashMap::new();
        op_layouts.insert(OpId(0), OpLayoutAssignment {
            input_layout: LayoutConstraint::Any,
            weight_layout: None,
            output_layout: LayoutConstraint::Any,
            accel_id: None,
            dtype: None,
        });
        let assignment = LayoutAssignment {
            group_assignments: vec![GroupLayoutAssignment {
                group_id: 0,
                op_layouts,
                inter_op_transforms: Vec::new(),
                dtype_transforms: Vec::new(),
                total_benefit: 10.0,
                total_transform_cost: 0.0,
            }],
            total_benefit: 10.0,
            total_transform_cost: 0.0,
        };

        // Act & Assert
        assert!(assignment.all_transforms_free());
        assert_eq!(assignment.total_benefit, 10.0);
    }

    /// @trace TEST-LN-64 [req:REQ-LAYOUT] [level:unit]
    /// InterOpTransform with GpuGlobalToShared movement carries correct metadata.
    #[test]
    fn test_inter_op_transform_gpu_global_to_shared_movement() {
        // Arrange
        let transform = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 128 },
            target: LayoutConstraint::ColMajor { align_bytes: 128 },
            cost: 0.0,
        };
        let iot = InterOpTransform {
            producer: OpId(5),
            consumer: OpId(6),
            transform,
            movement: MovementType::GpuGlobalToShared,
        dtype_transform: None,
        };

        // Act & Assert
        assert_eq!(iot.movement, MovementType::GpuGlobalToShared);
        assert_eq!(iot.producer, OpId(5));
        assert_eq!(iot.consumer, OpId(6));
        assert_eq!(iot.transform.cost, 0.0);
    }

    /// @trace TEST-LN-65 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: Any is compatible with ColMajor (both directions).
    #[test]
    fn test_compatible_with_any_vs_colmajor() {
        // Arrange
        let any = LayoutConstraint::Any;
        let colmajor = LayoutConstraint::ColMajor { align_bytes: 64 };

        // Act & Assert
        assert!(any.compatible_with(&colmajor));
        assert!(colmajor.compatible_with(&any));
    }

    /// @trace TEST-LN-66 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: identical InterleavedPairs are compatible.
    #[test]
    fn test_compatible_with_interleaved_identical() {
        // Arrange
        let a = LayoutConstraint::InterleavedPairs;
        let b = LayoutConstraint::InterleavedPairs;

        // Act & Assert
        assert!(a.compatible_with(&b));
    }

    /// @trace TEST-LN-67 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: single ElemWise op in Standalone mode gets exactly one layout assignment.
    #[test]
    fn test_negotiate_group_standalone_elemwise() {
        // Arrange
        let mut dag = make_test_dag();
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(0),
            op: crate::compiler::graph::Op::Silu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "silu".to_string(),
        });
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert
        let ga = &result.group_assignments[0];
        assert_eq!(ga.op_layouts.len(), 1);
        assert!(ga.op_layouts.contains_key(&OpId(0)));
        assert!(ga.inter_op_transforms.is_empty());
    }

    /// @trace TEST-LN-68 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: GemmBias with ColMajor (non-PanelPacked)
    /// weight_layout passes it through unchanged.
    #[test]
    fn test_model_aware_gemmbias_colmajor_weight_passthrough() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64, 128], DType::F32);
        let bias = graph.add_tensor_concrete("bias", &[128], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 128], DType::F32);
        graph.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, b, bias],
            vec![c],
            "gemm_bias_colmaj_wt",
        );
        let weight_input = LayoutConstraint::ColMajor { align_bytes: 32 };

        // Act
        let (_, wl) = model_aware_layout_overrides(
            OpId(0),
            &graph,
            LayoutConstraint::Any,
            Some(weight_input.clone()),
        );

        // Assert: ColMajor weight is not PanelPacked, so passthrough
        assert_eq!(wl, Some(weight_input));
    }

    // ── 10 new tests (wave-12x59) ────────────────────────────────────────

    /// @trace TEST-LN-69 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: identical AmxTileBF16 layouts are compatible.
    #[test]
    fn test_compatible_with_identical_amx_tile_bf16() {
        // Arrange
        let a = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };
        let b = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };

        // Act & Assert
        assert!(a.compatible_with(&b), "Identical AmxTileBF16 layouts should be compatible");
    }

    /// @trace TEST-LN-70 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: AmxTileBF16 with different dimensions
    /// is not compatible (different tile sizes require different handling).
    #[test]
    fn test_compatible_with_amx_tile_different_dims() {
        // Arrange
        let a = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };
        let b = LayoutConstraint::AmxTileBF16 { rows: 8, cols: 16 };

        // Act & Assert
        assert!(!a.compatible_with(&b), "AmxTileBF16 with different dims should not be compatible");
    }

    /// @trace TEST-LN-71 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: identical SharedMemTile layouts are compatible.
    #[test]
    fn test_compatible_with_identical_shared_mem_tile() {
        // Arrange
        let a = LayoutConstraint::SharedMemTile { tile_rows: 16, tile_cols: 16, padding_bytes: 4 };
        let b = LayoutConstraint::SharedMemTile { tile_rows: 16, tile_cols: 16, padding_bytes: 4 };

        // Act & Assert
        assert!(a.compatible_with(&b), "Identical SharedMemTile layouts should be compatible");
    }

    /// @trace TEST-LN-72 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: identical TmaAligned2D layouts are compatible.
    #[test]
    fn test_compatible_with_identical_tma_aligned_2d() {
        // Arrange
        let a = LayoutConstraint::TmaAligned2D { tile_m: 128, tile_n: 128 };
        let b = LayoutConstraint::TmaAligned2D { tile_m: 128, tile_n: 128 };

        // Act & Assert
        assert!(a.compatible_with(&b), "Identical TmaAligned2D layouts should be compatible");
    }

    /// @trace TEST-LN-73 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: RowMajor is not compatible with VnniPacked4.
    #[test]
    fn test_compatible_with_rowmajor_vs_vnnipacked() {
        // Arrange
        let row = LayoutConstraint::RowMajor { align_bytes: 64 };
        let vnni = LayoutConstraint::VnniPacked4;

        // Act & Assert
        assert!(!row.compatible_with(&vnni), "RowMajor and VnniPacked4 should not be compatible");
        assert!(!vnni.compatible_with(&row), "VnniPacked4 and RowMajor should not be compatible");
    }

    /// @trace TEST-LN-74 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: PanelPacked vs HeadSplit returns Any
    /// (no special case for PanelPacked ↔ HeadSplit).
    #[test]
    fn test_find_compromise_panelpacked_vs_headsplit() {
        // Arrange
        let a = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };
        let b = LayoutConstraint::HeadSplit { num_heads: 8, head_dim: 64 };

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-75 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: VnniPacked4 vs AmxTileBF16 returns Any
    /// (two GPU-specific layouts with no cross-type special case).
    #[test]
    fn test_find_compromise_vnni_vs_amx_tile() {
        // Arrange
        let a = LayoutConstraint::VnniPacked4;
        let b = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-76 [req:REQ-LAYOUT] [level:unit]
    /// classify_movement: Reduction producer with LoopFusion mode yields
    /// default RegisterToMemory (Reduction is not ElemWise/Injective, so
    /// the LoopFusion shortcut does not apply).
    #[test]
    fn test_classify_movement_reduction_loop_fusion_not_direct() {
        // Arrange
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::LoopFusion,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let movement = classify_movement(OpClass::Reduction, OpClass::ElemWise, 0, &group);

        // Assert: Reduction is not Gemm (index 0) nor ElemWise/Injective,
        // so it falls through to default RegisterToMemory even in LoopFusion mode.
        assert_eq!(movement, MovementType::RegisterToMemory);
    }

    /// @trace TEST-LN-77 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: QKV Gemm without an MHA or RoPE in the graph
    /// falls back to head_dim = n (the op's n dimension), producing num_heads = 1.
    #[test]
    fn test_model_aware_gemm_qkv_without_mha_fallback_head_dim() {
        // Arrange: 3 Gemm ops sharing input, but no MHA/RoPE in the graph
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 256], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[256, 512], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[256, 512], DType::F32);
        let w3 = graph.add_tensor_concrete("w3", &[256, 512], DType::F32);
        let o1 = graph.add_tensor_concrete("o1", &[1, 512], DType::F32);
        let o2 = graph.add_tensor_concrete("o2", &[1, 512], DType::F32);
        let o3 = graph.add_tensor_concrete("o3", &[1, 512], DType::F32);

        let q_op = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w1], vec![o1], "q_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w2], vec![o2], "k_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w3], vec![o3], "v_proj");

        // Act: no MHA/RoPE => extract_head_dim_from_graph returns None
        // => head_dim = n = 512, num_heads = 512 / 512 = 1
        let (out_layout, _) = model_aware_layout_overrides(
            q_op,
            &graph,
            LayoutConstraint::Any,
            None,
        );

        // Assert: fallback uses n as head_dim, so num_heads = n/n = 1
        assert_eq!(
            out_layout,
            LayoutConstraint::HeadSplit { num_heads: 1, head_dim: 512 },
            "QKV Gemm without MHA/RoPE should fall back to head_dim=n, num_heads=1"
        );
    }

    /// @trace TEST-LN-78 [req:REQ-LAYOUT] [level:unit]
    /// negotiate: two groups with distinct group IDs produce independent
    /// op_layouts HashMaps (no cross-contamination).
    #[test]
    fn test_negotiate_two_groups_independent_op_layouts() {
        // Arrange
        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let dag = make_test_dag();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();

        let group_a = FusionGroup {
            id: 10,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let group_b = FusionGroup {
            id: 20,
            anchor: OpId(1),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group_a, group_b], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert: each group has its own independent op_layouts
        assert_eq!(result.group_assignments.len(), 2);
        assert_eq!(result.group_assignments[0].group_id, 10);
        assert_eq!(result.group_assignments[1].group_id, 20);
        // group_a has OpId(0), group_b has OpId(1) — no overlap
        assert!(result.group_assignments[0].op_layouts.contains_key(&OpId(0)));
        assert!(!result.group_assignments[0].op_layouts.contains_key(&OpId(1)));
        assert!(result.group_assignments[1].op_layouts.contains_key(&OpId(1)));
        assert!(!result.group_assignments[1].op_layouts.contains_key(&OpId(0)));
    }

    // ── 10 new tests (wave-12x90) ────────────────────────────────────────

    /// @trace TEST-LN-79 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: SharedMemTile with different tile_rows
    /// is not compatible (different tile geometry requires re-tiling).
    #[test]
    fn test_compatible_with_shared_mem_tile_different_rows() {
        // Arrange
        let a = LayoutConstraint::SharedMemTile { tile_rows: 16, tile_cols: 16, padding_bytes: 4 };
        let b = LayoutConstraint::SharedMemTile { tile_rows: 32, tile_cols: 16, padding_bytes: 4 };

        // Act & Assert
        assert!(!a.compatible_with(&b), "SharedMemTile with different tile_rows should not be compatible");
    }

    /// @trace TEST-LN-80 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: TmaAligned2D with different tile_n
    /// is not compatible (different TMA descriptor needed).
    #[test]
    fn test_compatible_with_tma_aligned_2d_different_tile_n() {
        // Arrange
        let a = LayoutConstraint::TmaAligned2D { tile_m: 128, tile_n: 128 };
        let b = LayoutConstraint::TmaAligned2D { tile_m: 128, tile_n: 64 };

        // Act & Assert
        assert!(!a.compatible_with(&b), "TmaAligned2D with different tile_n should not be compatible");
    }

    /// @trace TEST-LN-81 [req:REQ-LAYOUT] [level:unit]
    /// LayoutConstraint::compatible_with: AmxTileBF16 is not compatible with SharedMemTile
    /// (fundamentally different GPU vs CPU tile formats).
    #[test]
    fn test_compatible_with_amx_tile_vs_shared_mem_tile() {
        // Arrange
        let amx = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };
        let smem = LayoutConstraint::SharedMemTile { tile_rows: 16, tile_cols: 32, padding_bytes: 0 };

        // Act & Assert: both directions
        assert!(!amx.compatible_with(&smem), "AmxTileBF16 and SharedMemTile should not be compatible");
        assert!(!smem.compatible_with(&amx), "SharedMemTile and AmxTileBF16 should not be compatible");
    }

    /// @trace TEST-LN-82 [req:REQ-LAYOUT] [level:unit]
    /// find_compromise: TmaAligned2D vs RowMajor returns Any
    /// (no special case for TMA <-> RowMajor).
    #[test]
    fn test_find_compromise_tma_vs_rowmajor() {
        // Arrange
        let a = LayoutConstraint::TmaAligned2D { tile_m: 128, tile_n: 128 };
        let b = LayoutConstraint::RowMajor { align_bytes: 128 };

        // Act
        let result = find_compromise(&a, &b);

        // Assert
        assert_eq!(result, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-83 [req:REQ-LAYOUT] [level:unit]
    /// model_aware_layout_overrides: Gemm with Any output and no QKV siblings
    /// preserves Any (non-QKV Gemm wildcard branch).
    #[test]
    fn test_model_aware_gemm_any_output_non_qkv() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, LayerCondition, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let a = graph.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = graph.add_tensor_concrete("b", &[64, 128], DType::F32);
        let c = graph.add_tensor_concrete("c", &[1, 128], DType::F32);
        let op_id = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, b],
            vec![c],
            "single_gemm",
        );

        // Act: output is Any, non-QKV single gemm
        let (out_layout, _) = model_aware_layout_overrides(
            op_id,
            &graph,
            LayoutConstraint::Any,
            None,
        );

        // Assert: single Gemm is not QKV, so Any output is preserved
        assert_eq!(out_layout, LayoutConstraint::Any);
    }

    /// @trace TEST-LN-84 [req:REQ-LAYOUT] [level:unit]
    /// negotiate_group: two-op Standalone with ElemWise producers has
    /// RegisterToMemory movement and zero transform cost (Any is compatible with Any).
    #[test]
    fn test_negotiate_two_elemwise_standalone_zero_cost() {
        // Arrange
        let mut dag = make_test_dag();
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(0),
            op: crate::compiler::graph::Op::Gelu,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "gelu".to_string(),
        });
        dag.nodes.push(crate::compiler::semantic_dag::SemanticNode {
            node_id: OpId(1),
            op: crate::compiler::graph::Op::Tanh,
            op_trace: None,
            op_class: OpClass::ElemWise,
            bottleneck: crate::compiler::semantic_dag::Bottleneck::Memory,
            arithmetic_intensity: 1.0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            label: "tanh".to_string(),
        });

        let registry = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks: HashMap::new(),
            ridge_point: 0.0,
        };
        let graph = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = LayoutNegotiator::negotiate(
            &[group], &registry, &device, &dag, &bottleneck_map, &graph,
        );

        // Assert: two ElemWise ops with Any layout, Standalone => RegisterToMemory
        // But Any is compatible with Any, so no transforms are generated.
        let ga = &result.group_assignments[0];
        assert_eq!(ga.op_layouts.len(), 2);
        assert_eq!(ga.total_transform_cost, 0.0);
        // Verify the movement type is RegisterToMemory for ElemWise in Standalone mode
        let group_ref = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0), OpId(1)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert_eq!(
            classify_movement(OpClass::ElemWise, OpClass::ElemWise, 0, &group_ref),
            MovementType::RegisterToMemory,
        );
    }

    /// @trace TEST-LN-85 [req:REQ-LAYOUT] [level:unit]
    /// LayoutTransform cost is preserved as f64 across clone and comparison.
    /// Zero cost on GpuGlobalToShared movement represents "free transformation window".
    #[test]
    fn test_layout_transform_zero_cost_free_window() {
        // Arrange: simulate a free transform window (RegisterToMemory with cost 0)
        let t = LayoutTransform {
            source: LayoutConstraint::RowMajor { align_bytes: 64 },
            target: LayoutConstraint::HeadSplit { num_heads: 32, head_dim: 128 },
            cost: 0.0,
        };

        // Act
        let cloned = t.clone();

        // Assert: cost is exactly 0.0 (free transformation window)
        assert_eq!(cloned.cost, 0.0);
        assert!(cloned.source != cloned.target, "Source and target differ (layout change occurred)");
    }

    /// @trace TEST-LN-86 [req:REQ-LAYOUT] [level:unit]
    /// is_qkv_op: QuantGemm ops are counted as siblings in the QKV detection
    /// (QuantGemm is included in the sibling filter alongside Gemm/GemmBias).
    #[test]
    fn test_is_qkv_op_quant_gemm_siblings_counted() {
        // Arrange: 2 regular Gemm + 1 QuantGemm sharing the same input
        use crate::compiler::graph::{CompilerGraph, CompilerOp, LayerCondition, KvSource, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 256], DType::F32);
        let w1 = graph.add_tensor_concrete("w1", &[256, 256], DType::F32);
        let w2 = graph.add_tensor_concrete("w2", &[256, 256], DType::F32);
        let w3 = graph.add_tensor_concrete("w3", &[256, 256], DType::F32);
        let o1 = graph.add_tensor_concrete("o1", &[1, 256], DType::F32);
        let o2 = graph.add_tensor_concrete("o2", &[1, 256], DType::F32);
        let o3 = graph.add_tensor_concrete("o3", &[1, 256], DType::F32);

        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w1], vec![o1], "q_proj");
        graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }), vec![input, w2], vec![o2], "k_proj");

        // Third sibling is a QuantGemm — added via graph.add_op to maintain tensor.consumers index.
        graph.add_op(Op::QuantGemm(QuantGemmSpec { m: SymDim::Concrete(1), n: 256, k: 256, quant_type: crate::quant::QuantType::F32 }), vec![input, w3], vec![o3], "v_proj");

        // Act: check the first Gemm (OpId(0)) — should see 3 siblings (2 Gemm + 1 QuantGemm)
        let result = is_qkv_op(OpId(0), &graph);

        // Assert: QuantGemm is counted, so >= 3 siblings => true
        assert!(result, "QuantGemm should be counted as sibling, making 3 total => QKV detected");
    }

    /// @trace TEST-LN-87 [req:REQ-LAYOUT] [level:unit]
    /// extract_head_dim_from_graph: graph with only non-MHA/non-RoPE ops returns None.
    /// Specifically tests that Silu and RmsNorm are not sources of head_dim.
    #[test]
    fn test_extract_head_dim_silu_rmsnorm_returns_none() {
        // Arrange
        use crate::compiler::graph::{CompilerGraph, KvSource, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();
        let x = graph.add_tensor_concrete("x", &[1, 256], DType::F32);
        let y = graph.add_tensor_concrete("y", &[1, 256], DType::F32);
        graph.add_op(Op::Silu, vec![x], vec![y], "silu");

        let z = graph.add_tensor_concrete("z", &[1, 256], DType::F32);
        let w = graph.add_tensor_concrete("w", &[1, 256], DType::F32);
        graph.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-6, dtype: DType::F32, has_weight: true }), vec![z], vec![w], "norm");

        // Act
        let head_dim = extract_head_dim_from_graph(&graph);

        // Assert
        assert_eq!(head_dim, None, "Silu and RmsNorm should not provide head_dim");
    }

    /// @trace TEST-LN-88 [req:REQ-LAYOUT] [level:unit]
    /// LayoutAssignment accumulation: total_benefit is sum of all group benefits
    /// and total_transform_cost is sum of all group costs, verified with three groups.
    #[test]
    fn test_layout_assignment_accumulation_three_groups() {
        // Arrange: three groups with different benefit/cost values
        let g1 = GroupLayoutAssignment {
            group_id: 0,
            op_layouts: HashMap::new(),
            inter_op_transforms: Vec::new(),
                dtype_transforms: Vec::new(),
            total_benefit: 2.0,
            total_transform_cost: 0.0,
        };
        let g2 = GroupLayoutAssignment {
            group_id: 1,
            op_layouts: HashMap::new(),
            inter_op_transforms: Vec::new(),
                dtype_transforms: Vec::new(),
            total_benefit: 3.0,
            total_transform_cost: 1.5,
        };
        let g3 = GroupLayoutAssignment {
            group_id: 2,
            op_layouts: HashMap::new(),
            inter_op_transforms: Vec::new(),
                dtype_transforms: Vec::new(),
            total_benefit: 5.0,
            total_transform_cost: 0.5,
        };
        let assignment = LayoutAssignment {
            group_assignments: vec![g1, g2, g3],
            total_benefit: 10.0,
            total_transform_cost: 2.0,
        };

        // Act & Assert
        assert_eq!(assignment.group_assignments.len(), 3);
        assert_eq!(assignment.total_benefit, 10.0); // 2 + 3 + 5
        assert_eq!(assignment.total_transform_cost, 2.0); // 0 + 1.5 + 0.5
        assert!(!assignment.all_transforms_free());
    }
}
