//! R2 虚拟数据求解 — VirtualTensor + IndexMap (SSOT: §3.3)
//!
//! VTC (Virtual Tensor Opportunity Graph) + Global Greedy 消除中间 tensor 物理化。
//! 消费 R1.5 LayoutAssignment — 虚拟 tensor 的 IndexMap 必须尊重协商布局。
//!
//! 消除所有可消除的中间 tensor 物理化。
//! Type I (始终有利): 全连续虚拟 tensor (Reshape, Slice) → 零成本索引变换
//! Type II (需要分析): 部分连续 (Transpose, Permute) → 需评估 memory coalescing 影响
//! 不可虚拟化: 跨融合组边界, 多消费者外部引用

use std::collections::{HashMap, HashSet};
use crate::compiler::graph::{CompilerGraph, OpId, TensorId};
use crate::compiler::fusion::FusionPlan;
use crate::compiler::layout_negotiator::LayoutAssignment;
use crate::compiler::pack_map::PackMap;
use crate::dispatch::device_profile::DeviceProfile;

/// 索引变换类型 (VTC §3.2 mapping function F)
#[derive(Debug, Clone, PartialEq)]
pub enum IndexMap {
    /// f(i) = i — 零开销 (相同布局)
    Identity,
    /// f(i) = i + offset — Reshape, Slice
    Offset(isize),
    /// f(i, j) = (j, i) — 矩阵转置
    Transpose2D,
    /// f(i) = perm[i] — Permute (需要维度重排)
    Permute(Vec<usize>),
    /// f(i) = scale * i — 广播/重复
    Broadcast { factor: usize },
    /// f(i) = head_split(i, num_heads, head_dim) — RowMajor → HeadSplit reshape
    /// [seq, hidden] → [seq, num_heads, head_dim], 零成本 stride 重计算
    HeadSplit { num_heads: usize, head_dim: usize },
    /// f(i, j) = interleave_pair(i, j) — Gate/Up 交织
    /// [seq, 2×inter] → InterleavedPairs, SwiGLU 直接消费
    InterleavePair,
}

impl IndexMap {
    /// 估算变换的运行时成本 (0 = 零成本)，参数化于硬件能力。
    pub fn cost_for(&self, profile: &DeviceProfile) -> f64 {
        match self {
            IndexMap::Identity => 0.0,
            IndexMap::Offset(_) => 0.0,
            IndexMap::HeadSplit { .. } => 0.0,
            IndexMap::InterleavePair => 0.0,
            IndexMap::Transpose2D => {
                // AVX-512 VPERM/SVE compact: 近零代价; 无硬件转置: 非合并访问惩罚
                if profile.has_hw_transpose() { 0.1 } else { 1.0 }
            }
            IndexMap::Permute(_) => {
                if profile.has_hw_permute() { 0.2 } else { 2.0 }
            }
            IndexMap::Broadcast { .. } => {
                if profile.has_hw_broadcast() { 0.0 } else { 0.5 }
            }
        }
    }

    /// 是否为 Type I (始终有利, 零成本)
    pub fn is_type_i_for(&self, profile: &DeviceProfile) -> bool {
        self.cost_for(profile) == 0.0
    }
}

/// 虚拟 tensor (VTC §3.1: (F, P₁, ..., Pₙ))
#[derive(Debug, Clone)]
pub struct VirtualTensor {
    /// 物理来源 tensor
    pub source: TensorId,
    /// 索引变换函数
    pub index_map: IndexMap,
    /// 字节偏移 (叠加在 source offset 之上)
    pub byte_offset: usize,
    /// 逻辑元素数
    pub num_elements: usize,
    /// 每元素字节数
    pub elem_bytes: usize,
}

impl VirtualTensor {
    /// 虚拟化节省的字节数 (一次写 + 一次读)
    pub fn bytes_saved(&self) -> usize {
        self.num_elements * self.elem_bytes * 2
    }
}

/// VTC Round 2 输出: 虚拟化决策映射
#[derive(Debug, Clone)]
pub struct VirtualTensorMap {
    /// TensorId → VirtualTensor (如果虚拟化)
    pub virtual_map: HashMap<TensorId, VirtualTensor>,
    /// 必须物理化的 tensor (跨组/多消费者)
    pub physical_set: HashSet<TensorId>,
    /// 总节省字节数
    pub bytes_saved: usize,
    /// §0.2.7 虚拟权重 — PackMap 索引映射替代物理 pack
    /// 权重 tensor → PackMap: 编译时生成 stride 计算, 运行时零 pack buffer
    pub pack_maps: HashMap<TensorId, PackMap>,
}

impl VirtualTensorMap {
    /// 空映射
    pub fn empty() -> Self {
        VirtualTensorMap {
            virtual_map: HashMap::new(),
            physical_set: HashSet::new(),
            bytes_saved: 0,
            pack_maps: HashMap::new(),
        }
    }

    /// 是否某个 tensor 已虚拟化
    pub fn is_virtual(&self, tid: TensorId) -> bool {
        self.virtual_map.contains_key(&tid)
    }

    /// 获取虚拟 tensor 的物理来源链 (递归追踪到最终物理 tensor)
    pub fn physical_root(&self, tid: TensorId) -> TensorId {
        let mut current = tid;
        while let Some(vt) = self.virtual_map.get(&current) {
            current = vt.source;
        }
        current
    }
}

/// VTC 虚拟 tensor 机会 (VTOG 边)
#[derive(Debug, Clone)]
struct VirtualOpportunity {
    /// 可虚拟化的 tensor
    tid: TensorId,
    /// 物理来源
    source: TensorId,
    /// 索引变换
    index_map: IndexMap,
    /// 节省字节数
    benefit: usize,
}

/// 数据流优化器 — VTC VTOG + Global Greedy
pub struct DataFlowOptimizer;

impl DataFlowOptimizer {
    /// 构建 VTOG 并用 Global Greedy 消除中间 tensor (SPEC §3.3)
    /// 消费 R1.5 LayoutAssignment — 虚拟 tensor 的 IndexMap 必须尊重协商布局。
    /// 消费 DeviceProfile — IndexMap 代价参数化于硬件能力。
    pub fn eliminate(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        layout: Option<&LayoutAssignment>,
        profile: &DeviceProfile,
    ) -> VirtualTensorMap {
        let mut opportunities = Vec::new();
        let mut virtual_map = HashMap::new();
        let mut physical_set = HashSet::new();
        let mut pack_maps = HashMap::new();

        // §0.2.7: 推导 PackMap — 从 LayoutConstraint 转换为虚拟权重索引映射
        if let Some(la) = layout {
            for ga in &la.group_assignments {
                for (&op_id, assign) in &ga.op_layouts {
                    // 输出 tensor 的 PackMap
                    let pm = crate::compiler::pack_map::pack_map_from_layout(&assign.output_layout);
                    if pm.requires_physical_pack() {
                        if let Some(op) = graph.op(op_id) {
                            for &out_tid in &op.outputs {
                                pack_maps.insert(out_tid, pm.clone());
                            }
                        }
                    }
                    // 输入 (权重) tensor 的 PackMap — B-matrix panel pack 等
                    if let Some(wl) = &assign.weight_layout {
                        if let Some(op) = graph.op(op_id) {
                            let pm = crate::compiler::pack_map::pack_map_from_layout(wl);
                            if pm.requires_physical_pack() {
                                if let Some(&tid) = op.inputs.get(1) {
                                    pack_maps.insert(tid, pm);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 1: 分析每个 tensor 的 def-use 链
        for tensor in &graph.tensors {
            let tid = tensor.id;

            // 跳过图输入 (权重/activation — 已经物理化)
            if tensor.producer.is_none() {
                physical_set.insert(tid);
                continue;
            }

            // 分析消费者情况
            let consumers = &tensor.consumers;

            // Step 2: 检查是否可虚拟化
            if let Some(opportunity) = analyze_virtual_opportunity(graph, tid, consumers, plan, layout) {
                opportunities.push(opportunity);
            } else {
                physical_set.insert(tid);
            }
        }

        // Step 3: Global Greedy — 按收益排序, 贪心选择
        opportunities.sort_by(|a, b| b.benefit.cmp(&a.benefit));

        let mut claimed_sources: HashSet<TensorId> = HashSet::new();
        let mut total_saved = 0;

        for opp in opportunities {
            // 冲突检测: 一个 tensor 只能有一个虚拟来源
            // 一个 source 不能被多个虚拟 tensor 引用 (除非 Type I 零成本)
            if !opp.index_map.is_type_i_for(profile) && claimed_sources.contains(&opp.source) {
                physical_set.insert(opp.tid);
                continue;
            }

            let elem_bytes = graph.tensor(opp.tid)
                    .map(|t| t.dtype.size_bytes())
                    .unwrap_or(4);
            virtual_map.insert(opp.tid, VirtualTensor {
                source: opp.source,
                index_map: opp.index_map,
                byte_offset: 0,
                num_elements: opp.benefit / (elem_bytes.max(1) * 2),
                elem_bytes,
            });
            claimed_sources.insert(opp.source);
            total_saved += opp.benefit;
        }

        VirtualTensorMap {
            virtual_map,
            physical_set,
            bytes_saved: total_saved,
            pack_maps,
        }
    }
}

/// 分析单个 tensor 的虚拟化机会
fn analyze_virtual_opportunity(
    graph: &CompilerGraph,
    tid: TensorId,
    consumers: &[OpId],
    plan: &FusionPlan,
    layout: Option<&LayoutAssignment>,
) -> Option<VirtualOpportunity> {
    let tensor = graph.tensor(tid)?;

    // 必须有生产者 (非图输入)
    let producer_id = tensor.producer?;

    // 单消费者: 检查是否可以虚拟化
    if consumers.len() == 1 {
        let consumer_id = consumers[0];

        // 检查 producer 和 consumer 是否在同一融合组
        let same_group = plan.op_to_group.get(&producer_id)
            .zip(plan.op_to_group.get(&consumer_id))
            .map(|(a, b)| a == b)
            .unwrap_or(false);

        if same_group {
            // 布局感知: 检查 R1.5 是否要求此 tensor 使用物理 pack 布局
            if let Some(la) = layout {
                if requires_physical_layout(la, producer_id) {
                    return None;
                }
            }

            // 同组内单消费者: 可以虚拟化
            let elem_bytes = tensor.dtype.size_bytes();
            let num_elements = tensor.concrete_numel();
            let benefit = num_elements * elem_bytes * 2;

            // 推断 IndexMap 类型 (布局感知, 按 producer OpId 精确查找)
            let index_map = infer_index_map(producer_id, layout);

            let source = find_physical_source(graph, producer_id)?;

            return Some(VirtualOpportunity {
                tid,
                source,
                index_map,
                benefit,
            });
        }
    }

    // 多消费者: 检查是否所有消费者都在同一融合组
    if consumers.len() > 1 {
        let first_group = plan.op_to_group.get(&consumers[0]);
        let all_same = first_group.map(|fg| {
            consumers.iter().all(|c| plan.op_to_group.get(c) == Some(fg))
        }).unwrap_or(false);

        if all_same {
            // 组内多消费者: 部分虚拟化 (一次物理化, 多消费点虚拟引用)
            // 暂时标记为 physical (Phase D 深化处理)
        }
    }

    None
}

/// 推断索引变换类型 (布局感知)
/// 消费 R1.5 LayoutAssignment: 按 producer OpId 精确查找协商布局。
fn infer_index_map(
    producer_id: OpId,
    layout: Option<&LayoutAssignment>,
) -> IndexMap {
    use crate::compiler::accel_registry::LayoutConstraint;

    if let Some(la) = layout {
        if let Some(assign) = la.group_assignments.iter()
            .find_map(|ga| ga.op_layouts.get(&producer_id))
        {
            return match &assign.output_layout {
                LayoutConstraint::HeadSplit { num_heads, head_dim } =>
                    IndexMap::HeadSplit { num_heads: *num_heads, head_dim: *head_dim },
                LayoutConstraint::InterleavedPairs =>
                    IndexMap::InterleavePair,
                LayoutConstraint::ColMajor { .. } =>
                    IndexMap::Transpose2D,
                _ => IndexMap::Identity,
            };
        }
    }

    IndexMap::Identity
}

/// 追踪物理来源 tensor
fn find_physical_source(
    graph: &CompilerGraph,
    producer_id: OpId,
) -> Option<TensorId> {
    graph.op(producer_id).and_then(|op| op.inputs.first().copied())
}

/// 检查 R1.5 是否要求此 tensor 使用物理 pack 布局 (不可虚拟化)
/// §0.2.7: PackMap 虚拟化的布局不再需要物理 pack — 返回 false
fn requires_physical_layout(layout: &LayoutAssignment, producer_op: OpId) -> bool {
    use crate::compiler::accel_registry::LayoutConstraint;
    layout.group_assignments.iter()
        .find_map(|ga| ga.op_layouts.get(&producer_op))
        .is_some_and(|assign| {
            let lc = &assign.output_layout;
            // §0.2.7: PanelPacked/Vnni/AmxTile 已由 PackMap 虚拟化 — 不需要物理 pack
            if matches!(lc,
                LayoutConstraint::PanelPacked { .. } |
                LayoutConstraint::VnniPacked4 |
                LayoutConstraint::AmxTileBF16 { .. }
            ) {
                return false;
            }
            matches!(lc,
                LayoutConstraint::SharedMemTile { .. } |
                LayoutConstraint::TmaAligned2D { .. }
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_map_cost() {
        let profile = DeviceProfile::detect();
        assert_eq!(IndexMap::Identity.cost_for(&profile), 0.0);
        assert_eq!(IndexMap::Offset(16).cost_for(&profile), 0.0);
        assert_eq!(IndexMap::HeadSplit { num_heads: 32, head_dim: 128 }.cost_for(&profile), 0.0);
        assert_eq!(IndexMap::InterleavePair.cost_for(&profile), 0.0);
        assert!(IndexMap::Transpose2D.cost_for(&profile) > 0.0);
        assert!(IndexMap::Permute(vec![1, 0]).cost_for(&profile) > 0.0);
    }

    #[test]
    fn test_index_map_type_i() {
        let profile = DeviceProfile::detect();
        assert!(IndexMap::Identity.is_type_i_for(&profile));
        assert!(IndexMap::Offset(0).is_type_i_for(&profile));
        assert!(IndexMap::HeadSplit { num_heads: 32, head_dim: 128 }.is_type_i_for(&profile));
        assert!(!IndexMap::Transpose2D.is_type_i_for(&profile));
    }

    #[test]
    fn test_virtual_tensor_bytes_saved() {
        let vt = VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 1024,
            elem_bytes: 4,
        };
        assert_eq!(vt.bytes_saved(), 1024 * 4 * 2);
    }

    #[test]
    fn test_virtual_tensor_map_empty() {
        let vtm = VirtualTensorMap::empty();
        assert!(vtm.virtual_map.is_empty());
        assert!(vtm.physical_set.is_empty());
        assert!(vtm.pack_maps.is_empty());
        assert_eq!(vtm.bytes_saved, 0);
        assert!(!vtm.is_virtual(TensorId(0)));
    }

    #[test]
    fn test_virtual_tensor_map_physical_root() {
        let mut vtm = VirtualTensorMap::empty();
        vtm.virtual_map.insert(TensorId(2), VirtualTensor {
            source: TensorId(1),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 100,
            elem_bytes: 4,
        });
        vtm.virtual_map.insert(TensorId(1), VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Offset(16),
            byte_offset: 0,
            num_elements: 100,
            elem_bytes: 4,
        });

        // 追踪到物理根
        assert_eq!(vtm.physical_root(TensorId(2)), TensorId(0));
        assert_eq!(vtm.physical_root(TensorId(1)), TensorId(0));
        assert_eq!(vtm.physical_root(TensorId(0)), TensorId(0)); // 物理tensor返回自己
    }

    // @trace TEST-VT-06 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_broadcast_cost() {
        // Arrange: two profiles — one with hw broadcast, one without
        let profile = DeviceProfile::detect();

        // Act
        let broadcast = IndexMap::Broadcast { factor: 4 };
        let cost = broadcast.cost_for(&profile);
        let is_type_i = broadcast.is_type_i_for(&profile);

        // Assert: cost is either 0.0 (hw broadcast) or 0.5 (no hw broadcast)
        assert!(cost == 0.0 || cost == 0.5, "unexpected broadcast cost: {cost}");
        assert_eq!(is_type_i, cost == 0.0);
    }

    // @trace TEST-VT-07 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_permute_cost_bounds() {
        // Arrange
        let profile = DeviceProfile::detect();
        let permute = IndexMap::Permute(vec![2, 0, 1]);

        // Act
        let cost = permute.cost_for(&profile);

        // Assert: must be one of the two defined cost tiers
        assert!(cost == 0.2 || cost == 2.0, "unexpected permute cost: {cost}");
    }

    // @trace TEST-VT-08 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_transpose_cost_bounds() {
        // Arrange
        let profile = DeviceProfile::detect();
        let transpose = IndexMap::Transpose2D;

        // Act
        let cost = transpose.cost_for(&profile);

        // Assert: must be one of the two defined cost tiers
        assert!(cost == 0.1 || cost == 1.0, "unexpected transpose cost: {cost}");
    }

    // @trace TEST-VT-09 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_interleave_pair_type_i() {
        // Arrange
        let profile = DeviceProfile::detect();
        let interleave = IndexMap::InterleavePair;

        // Act & Assert: InterleavePair is always zero-cost (Type I)
        assert_eq!(interleave.cost_for(&profile), 0.0);
        assert!(interleave.is_type_i_for(&profile));
    }

    // @trace TEST-VT-10 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_head_split_various_configs() {
        // Arrange
        let profile = DeviceProfile::detect();

        let configs = vec![
            (1, 1),
            (32, 128),
            (8, 4096),
            (128, 4),
        ];

        for (num_heads, head_dim) in configs {
            // Act
            let hs = IndexMap::HeadSplit { num_heads, head_dim };
            let cost = hs.cost_for(&profile);

            // Assert: HeadSplit is always zero-cost regardless of parameters
            assert_eq!(cost, 0.0, "HeadSplit cost should be 0 for heads={num_heads}, dim={head_dim}");
            assert!(hs.is_type_i_for(&profile));
        }
    }

    // @trace TEST-VT-11 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_offset_positive_and_negative() {
        // Arrange
        let profile = DeviceProfile::detect();

        let offsets: Vec<isize> = vec![0, 1, -1, 1024, -1024, isize::MAX, isize::MIN];

        for offset in offsets {
            // Act
            let idx = IndexMap::Offset(offset);
            let cost = idx.cost_for(&profile);

            // Assert: Offset is always zero-cost
            assert_eq!(cost, 0.0, "Offset({offset}) should be zero cost");
            assert!(idx.is_type_i_for(&profile));
        }
    }

    // @trace TEST-VT-12 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_bytes_saved_boundary() {
        // Arrange: zero elements — should save zero bytes
        let vt_zero = VirtualTensor {
            source: TensorId(99),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 0,
            elem_bytes: 4,
        };

        // Act & Assert
        assert_eq!(vt_zero.bytes_saved(), 0);

        // Arrange: large element size (e.g. f64 = 8 bytes)
        let vt_large = VirtualTensor {
            source: TensorId(100),
            index_map: IndexMap::Offset(64),
            byte_offset: 0,
            num_elements: 256,
            elem_bytes: 8,
        };

        // Act & Assert: 256 * 8 * 2 = 4096
        assert_eq!(vt_large.bytes_saved(), 4096);

        // Arrange: single element, single byte
        let vt_one = VirtualTensor {
            source: TensorId(101),
            index_map: IndexMap::HeadSplit { num_heads: 1, head_dim: 1 },
            byte_offset: 0,
            num_elements: 1,
            elem_bytes: 1,
        };

        // Act & Assert: 1 * 1 * 2 = 2
        assert_eq!(vt_one.bytes_saved(), 2);
    }

    // @trace TEST-VT-13 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_is_virtual_present_and_absent() {
        // Arrange: map with one virtual tensor
        let mut vtm = VirtualTensorMap::empty();
        let tid = TensorId(42);
        vtm.virtual_map.insert(tid, VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 100,
            elem_bytes: 4,
        });

        // Act & Assert: present
        assert!(vtm.is_virtual(tid));
        // Act & Assert: absent
        assert!(!vtm.is_virtual(TensorId(43)));
        assert!(!vtm.is_virtual(TensorId(0)));
    }

    // @trace TEST-VT-14 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_physical_root_deep_chain() {
        // Arrange: 4-level chain: 4 → 3 → 2 → 1 → 0 (physical)
        let mut vtm = VirtualTensorMap::empty();
        for i in 1u32..=4u32 {
            vtm.virtual_map.insert(TensorId(i), VirtualTensor {
                source: TensorId(i - 1),
                index_map: IndexMap::Offset(i as isize * 16),
                byte_offset: 0,
                num_elements: 50,
                elem_bytes: 4,
            });
        }

        // Act & Assert: all resolve to TensorId(0)
        for i in 1u32..=4u32 {
            assert_eq!(vtm.physical_root(TensorId(i)), TensorId(0),
                "TensorId({i}) should resolve to TensorId(0)");
        }

        // Act & Assert: physical tensor returns itself
        assert_eq!(vtm.physical_root(TensorId(0)), TensorId(0));
        // Act & Assert: unknown tensor (not in map) returns itself
        assert_eq!(vtm.physical_root(TensorId(999)), TensorId(999));
    }

    // @trace TEST-VT-15 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_physical_root_single_step() {
        // Arrange: one virtual tensor pointing directly to a physical source
        let mut vtm = VirtualTensorMap::empty();
        vtm.virtual_map.insert(TensorId(5), VirtualTensor {
            source: TensorId(10),
            index_map: IndexMap::Transpose2D,
            byte_offset: 256,
            num_elements: 512,
            elem_bytes: 2,
        });

        // Act & Assert: resolves in one step
        assert_eq!(vtm.physical_root(TensorId(5)), TensorId(10));
    }

    // @trace TEST-VT-16 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_equality() {
        // Arrange & Act & Assert: PartialEq-derived equality
        assert_eq!(IndexMap::Identity, IndexMap::Identity);
        assert_eq!(IndexMap::Offset(0), IndexMap::Offset(0));
        assert_ne!(IndexMap::Offset(0), IndexMap::Offset(1));
        assert_eq!(
            IndexMap::Permute(vec![1, 0, 2]),
            IndexMap::Permute(vec![1, 0, 2])
        );
        assert_ne!(
            IndexMap::Permute(vec![1, 0, 2]),
            IndexMap::Permute(vec![2, 0, 1])
        );
        assert_eq!(
            IndexMap::HeadSplit { num_heads: 8, head_dim: 64 },
            IndexMap::HeadSplit { num_heads: 8, head_dim: 64 }
        );
        assert_ne!(
            IndexMap::HeadSplit { num_heads: 8, head_dim: 64 },
            IndexMap::HeadSplit { num_heads: 8, head_dim: 128 }
        );
        assert_eq!(IndexMap::InterleavePair, IndexMap::InterleavePair);
        assert_eq!(IndexMap::Transpose2D, IndexMap::Transpose2D);
        assert_eq!(IndexMap::Broadcast { factor: 3 }, IndexMap::Broadcast { factor: 3 });
        assert_ne!(IndexMap::Broadcast { factor: 3 }, IndexMap::Broadcast { factor: 4 });
    }

    // @trace TEST-VT-17 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_empty_pack_maps() {
        // Arrange
        let vtm = VirtualTensorMap::empty();

        // Act & Assert: pack_maps starts empty
        assert!(vtm.pack_maps.is_empty(), "empty map should have no pack maps");
    }

    // @trace TEST-VT-18 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_fields_preserved() {
        // Arrange: construct a VirtualTensor with specific byte_offset
        let vt = VirtualTensor {
            source: TensorId(7),
            index_map: IndexMap::Permute(vec![1, 0]),
            byte_offset: 2048,
            num_elements: 4096,
            elem_bytes: 2,
        };

        // Act & Assert: all fields preserved correctly
        assert_eq!(vt.source, TensorId(7));
        assert_eq!(vt.index_map, IndexMap::Permute(vec![1, 0]));
        assert_eq!(vt.byte_offset, 2048);
        assert_eq!(vt.num_elements, 4096);
        assert_eq!(vt.elem_bytes, 2);
        assert_eq!(vt.bytes_saved(), 4096 * 2 * 2); // 16384
    }

    // @trace TEST-VT-19 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_physical_set_tracking() {
        // Arrange: map with both virtual and physical tensors
        let mut vtm = VirtualTensorMap::empty();
        vtm.virtual_map.insert(TensorId(10), VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 64,
            elem_bytes: 4,
        });
        vtm.physical_set.insert(TensorId(0));
        vtm.physical_set.insert(TensorId(20));
        vtm.bytes_saved = 512;

        // Act & Assert: physical set contains correct entries
        assert!(vtm.physical_set.contains(&TensorId(0)));
        assert!(vtm.physical_set.contains(&TensorId(20)));
        assert!(!vtm.physical_set.contains(&TensorId(10))); // virtual, not physical
        assert_eq!(vtm.bytes_saved, 512);
    }

    // @trace TEST-VT-20 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_multiple_pack_maps() {
        // Arrange: map with multiple pack_maps for different tensors
        use crate::compiler::pack_map::PackMap;
        let mut vtm = VirtualTensorMap::empty();
        let pm_panel = PackMap::PanelPack { nr: 4, kc: 256 };
        let pm_vnni = PackMap::VnniPack { interleave: 4 };
        let pm_tile = PackMap::TilePack { tile_rows: 16, tile_cols: 16 };
        vtm.pack_maps.insert(TensorId(100), pm_panel.clone());
        vtm.pack_maps.insert(TensorId(200), pm_vnni.clone());
        vtm.pack_maps.insert(TensorId(300), pm_tile.clone());

        // Act & Assert: all pack_maps present and correct
        assert_eq!(vtm.pack_maps.len(), 3);
        assert_eq!(vtm.pack_maps.get(&TensorId(100)), Some(&pm_panel));
        assert_eq!(vtm.pack_maps.get(&TensorId(200)), Some(&pm_vnni));
        assert_eq!(vtm.pack_maps.get(&TensorId(300)), Some(&pm_tile));
        assert_eq!(vtm.pack_maps.get(&TensorId(400)), None);
    }

    // @trace TEST-VT-21 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_large_physical_set() {
        // Arrange: map with many physical tensors and a few virtual ones
        let mut vtm = VirtualTensorMap::empty();
        for i in 0..100u32 {
            vtm.physical_set.insert(TensorId(i));
        }
        // Virtual tensors pointing to physical sources
        for i in 100..110u32 {
            vtm.virtual_map.insert(TensorId(i), VirtualTensor {
                source: TensorId(i - 100),
                index_map: IndexMap::Offset(0),
                byte_offset: 0,
                num_elements: 64,
                elem_bytes: 4,
            });
        }

        // Act & Assert: physical set has correct size
        assert_eq!(vtm.physical_set.len(), 100);
        // Virtual tensors resolve to their physical roots
        for i in 100..110u32 {
            assert_eq!(vtm.physical_root(TensorId(i)), TensorId(i - 100));
        }
        // Physical tensors are not virtual
        assert!(!vtm.is_virtual(TensorId(0)));
        assert!(!vtm.is_virtual(TensorId(99)));
        // Virtual tensors are virtual
        assert!(vtm.is_virtual(TensorId(100)));
        assert!(vtm.is_virtual(TensorId(109)));
    }

    // @trace TEST-VT-22 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_all_variants_type_i_classification() {
        // Arrange: check Type I classification for every IndexMap variant
        let profile = DeviceProfile::detect();

        let type_i_variants: Vec<IndexMap> = vec![
            IndexMap::Identity,
            IndexMap::Offset(0),
            IndexMap::Offset(isize::MAX),
            IndexMap::Offset(isize::MIN),
            IndexMap::HeadSplit { num_heads: 1, head_dim: 1 },
            IndexMap::HeadSplit { num_heads: 128, head_dim: 4096 },
            IndexMap::InterleavePair,
        ];

        for variant in &type_i_variants {
            // Act & Assert
            assert!(variant.is_type_i_for(&profile),
                "{:?} should be Type I (zero cost)", variant);
        }

        // Transpose2D is never Type I
        assert!(!IndexMap::Transpose2D.is_type_i_for(&profile));
    }

    // @trace TEST-VT-23 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_bytes_saved_various_elem_bytes() {
        // Arrange: test bytes_saved with different element sizes
        let configs: Vec<(usize, usize, usize)> = vec![
            (0, 1, 0),    // zero elements -> 0 bytes saved
            (1, 1, 2),    // 1 elem, 1 byte -> 2 bytes
            (1, 2, 4),    // 1 elem, 2 bytes -> 4 bytes
            (1, 4, 8),    // 1 elem, 4 bytes -> 8 bytes
            (1, 8, 16),   // 1 elem, 8 bytes -> 16 bytes
            (1024, 2, 4096), // 1024 elems, 2 bytes -> 4096 bytes
        ];

        for (num_elements, elem_bytes, expected) in configs {
            // Act
            let vt = VirtualTensor {
                source: TensorId(0),
                index_map: IndexMap::Identity,
                byte_offset: 0,
                num_elements,
                elem_bytes,
            };

            // Assert
            assert_eq!(vt.bytes_saved(), expected,
                "bytes_saved mismatch for num_elements={num_elements}, elem_bytes={elem_bytes}");
        }
    }

    // @trace TEST-VT-24 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_empty_is_virtual_for_all_absent() {
        // Arrange: empty map, check several TensorIds
        let vtm = VirtualTensorMap::empty();
        let absent_ids: Vec<TensorId> = vec![
            TensorId(0), TensorId(1), TensorId(100), TensorId(u32::MAX),
        ];

        // Act & Assert: none should be virtual
        for tid in &absent_ids {
            assert!(!vtm.is_virtual(*tid),
                "empty map should report TensorId({}) as non-virtual", tid.0);
        }
    }

    // @trace TEST-VT-25 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_pack_map_overwrite() {
        // Arrange: inserting pack_maps for the same TensorId should overwrite
        use crate::compiler::pack_map::PackMap;
        let mut vtm = VirtualTensorMap::empty();
        let pm_first = PackMap::PanelPack { nr: 4, kc: 128 };
        let pm_second = PackMap::VnniPack { interleave: 4 };

        vtm.pack_maps.insert(TensorId(50), pm_first);

        // Act: overwrite with different PackMap
        vtm.pack_maps.insert(TensorId(50), pm_second.clone());

        // Assert: only one entry, the latest
        assert_eq!(vtm.pack_maps.len(), 1);
        assert_eq!(vtm.pack_maps.get(&TensorId(50)), Some(&pm_second));
    }

    // @trace TEST-VT-26 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_physical_root_with_mixed_types() {
        // Arrange: chain with different IndexMap types
        let mut vtm = VirtualTensorMap::empty();
        vtm.virtual_map.insert(TensorId(3), VirtualTensor {
            source: TensorId(2),
            index_map: IndexMap::Transpose2D,
            byte_offset: 0,
            num_elements: 64,
            elem_bytes: 4,
        });
        vtm.virtual_map.insert(TensorId(2), VirtualTensor {
            source: TensorId(1),
            index_map: IndexMap::Permute(vec![1, 0]),
            byte_offset: 128,
            num_elements: 64,
            elem_bytes: 4,
        });
        vtm.virtual_map.insert(TensorId(1), VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Broadcast { factor: 4 },
            byte_offset: 256,
            num_elements: 64,
            elem_bytes: 4,
        });

        // Act & Assert: 3 → 2 → 1 → 0 (physical)
        assert_eq!(vtm.physical_root(TensorId(3)), TensorId(0));
        assert_eq!(vtm.physical_root(TensorId(2)), TensorId(0));
        assert_eq!(vtm.physical_root(TensorId(1)), TensorId(0));
        assert_eq!(vtm.physical_root(TensorId(0)), TensorId(0));
    }

    // @trace TEST-VT-27 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_cost_non_negative() {
        // Arrange: all IndexMap variants must have non-negative cost
        let profile = DeviceProfile::detect();
        let variants: Vec<IndexMap> = vec![
            IndexMap::Identity,
            IndexMap::Offset(0),
            IndexMap::Offset(4096),
            IndexMap::Transpose2D,
            IndexMap::Permute(vec![1, 0, 2]),
            IndexMap::Broadcast { factor: 1 },
            IndexMap::Broadcast { factor: 1024 },
            IndexMap::HeadSplit { num_heads: 32, head_dim: 128 },
            IndexMap::InterleavePair,
        ];

        for variant in &variants {
            // Act
            let cost = variant.cost_for(&profile);

            // Assert: cost must be >= 0
            assert!(cost >= 0.0, "cost for {:?} is negative: {}", variant, cost);
        }
    }

    // @trace TEST-VT-28 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_virtual_and_physical_coexist() {
        // Arrange: a VirtualTensorMap where the same TensorId(0) is both
        // a physical_set member and the source of a virtual tensor
        let mut vtm = VirtualTensorMap::empty();
        vtm.virtual_map.insert(TensorId(10), VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Offset(64),
            byte_offset: 0,
            num_elements: 256,
            elem_bytes: 4,
        });
        vtm.physical_set.insert(TensorId(0));
        vtm.physical_set.insert(TensorId(20));
        vtm.bytes_saved = 2048;

        // Act & Assert: is_virtual distinguishes virtual from physical
        assert!(vtm.is_virtual(TensorId(10)));
        assert!(!vtm.is_virtual(TensorId(0)));
        assert!(!vtm.is_virtual(TensorId(20)));

        // Act & Assert: physical_root of virtual tensor returns physical source
        assert_eq!(vtm.physical_root(TensorId(10)), TensorId(0));

        // Act & Assert: physical tensors in physical_set are tracked
        assert!(vtm.physical_set.contains(&TensorId(0)));
        assert!(vtm.physical_set.contains(&TensorId(20)));
        assert_eq!(vtm.physical_set.len(), 2);
    }

    // @trace TEST-VT-29 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_bytes_saved_proportional_to_elements_and_size() {
        // Arrange: two tensors with same num_elements but different elem_bytes
        let vt_f32 = VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 512,
            elem_bytes: 4,
        };
        let vt_f16 = VirtualTensor {
            source: TensorId(1),
            index_map: IndexMap::Offset(0),
            byte_offset: 0,
            num_elements: 512,
            elem_bytes: 2,
        };

        // Act & Assert: f32 tensor saves exactly 2x what f16 tensor saves
        let saved_f32 = vt_f32.bytes_saved();
        let saved_f16 = vt_f16.bytes_saved();
        assert!(saved_f32 > 0);
        assert!(saved_f16 > 0);
        assert_eq!(saved_f32, saved_f16 * 2);
    }

    // @trace TEST-VT-30 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_permute_different_lengths() {
        // Arrange: permute vectors of different lengths should all have valid cost
        let profile = DeviceProfile::detect();
        let permutes: Vec<IndexMap> = vec![
            IndexMap::Permute(vec![1, 0]),
            IndexMap::Permute(vec![2, 0, 1]),
            IndexMap::Permute(vec![3, 0, 1, 2]),
        ];

        for permute in &permutes {
            // Act
            let cost = permute.cost_for(&profile);

            // Assert: cost is always one of the two valid tiers
            assert!(
                cost == 0.2 || cost == 2.0,
                "unexpected permute cost for {:?}: {cost}",
                permute
            );
        }
    }

    // @trace TEST-VT-31 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_physical_set_deduplication() {
        // Arrange: inserting the same TensorId into physical_set twice
        let mut vtm = VirtualTensorMap::empty();
        vtm.physical_set.insert(TensorId(42));
        vtm.physical_set.insert(TensorId(42));
        vtm.physical_set.insert(TensorId(42));

        // Act & Assert: HashSet deduplicates, so len is 1
        assert_eq!(vtm.physical_set.len(), 1);
        assert!(vtm.physical_set.contains(&TensorId(42)));
    }

    // @trace TEST-VT-33 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_broadcast_factor_does_not_affect_type_i() {
        // Arrange: broadcast with factor=1 vs factor=1000000
        let profile = DeviceProfile::detect();
        let bcast_small = IndexMap::Broadcast { factor: 1 };
        let bcast_large = IndexMap::Broadcast { factor: 1_000_000 };

        // Act
        let cost_small = bcast_small.cost_for(&profile);
        let cost_large = bcast_large.cost_for(&profile);

        // Assert: cost depends only on hardware support, not factor magnitude
        assert_eq!(cost_small, cost_large);
        assert_eq!(bcast_small.is_type_i_for(&profile), bcast_large.is_type_i_for(&profile));
    }

    // @trace TEST-VT-34 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_bytes_saved_accumulated() {
        // Arrange: map with multiple virtual tensors, manually compute total
        let mut vtm = VirtualTensorMap::empty();
        let vt_a = VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 256,
            elem_bytes: 4,
        };
        let vt_b = VirtualTensor {
            source: TensorId(1),
            index_map: IndexMap::Offset(128),
            byte_offset: 0,
            num_elements: 1024,
            elem_bytes: 2,
        };
        let expected_total = vt_a.bytes_saved() + vt_b.bytes_saved();
        vtm.virtual_map.insert(TensorId(10), vt_a);
        vtm.virtual_map.insert(TensorId(11), vt_b);
        vtm.bytes_saved = expected_total;

        // Act & Assert: bytes_saved matches sum of individual savings
        assert!(vtm.bytes_saved > 0);
        assert_eq!(vtm.bytes_saved, 256 * 4 * 2 + 1024 * 2 * 2);
    }

    // @trace TEST-VT-35 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_offset_negative_values() {
        // Arrange: negative offsets are semantically valid (e.g., slicing from end)
        let profile = DeviceProfile::detect();
        let negative_offsets: Vec<isize> = vec![-1, -128, -4096, isize::MIN / 2];

        for offset in negative_offsets {
            let idx = IndexMap::Offset(offset);

            // Act & Assert: negative offsets are still zero-cost and Type I
            assert_eq!(idx.cost_for(&profile), 0.0);
            assert!(idx.is_type_i_for(&profile));
        }
    }

    // @trace TEST-VT-36 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_map_physical_root_non_virtual_returns_self() {
        // Arrange: tensor that is in physical_set but not in virtual_map
        let mut vtm = VirtualTensorMap::empty();
        vtm.physical_set.insert(TensorId(7));

        // Act & Assert: non-virtual tensor returns itself
        assert_eq!(vtm.physical_root(TensorId(7)), TensorId(7));
        // Unknown tensor (not in either set) also returns itself
        assert_eq!(vtm.physical_root(TensorId(99)), TensorId(99));
    }

    // @trace TEST-VT-37 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_virtual_tensor_byte_offset_independent_of_bytes_saved() {
        // Arrange: two tensors with same num_elements/elem_bytes but different byte_offset
        let vt_offset_0 = VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 0,
            num_elements: 100,
            elem_bytes: 4,
        };
        let vt_offset_large = VirtualTensor {
            source: TensorId(0),
            index_map: IndexMap::Identity,
            byte_offset: 999999,
            num_elements: 100,
            elem_bytes: 4,
        };

        // Act & Assert: bytes_saved is independent of byte_offset
        assert_eq!(vt_offset_0.bytes_saved(), vt_offset_large.bytes_saved());
        assert!(vt_offset_0.bytes_saved() > 0);
    }

    // @trace TEST-VT-38 [req:REQ-VTC] [level:unit]
    #[test]
    fn test_index_map_type_i_implies_zero_cost() {
        // Arrange: for every IndexMap variant, if it is Type I then cost must be 0
        let profile = DeviceProfile::detect();
        let variants: Vec<IndexMap> = vec![
            IndexMap::Identity,
            IndexMap::Offset(0),
            IndexMap::Offset(999),
            IndexMap::Transpose2D,
            IndexMap::Permute(vec![1, 0]),
            IndexMap::Permute(vec![2, 1, 0]),
            IndexMap::Broadcast { factor: 1 },
            IndexMap::Broadcast { factor: 16 },
            IndexMap::HeadSplit { num_heads: 4, head_dim: 64 },
            IndexMap::HeadSplit { num_heads: 128, head_dim: 8 },
            IndexMap::InterleavePair,
        ];

        for variant in &variants {
            let cost = variant.cost_for(&profile);
            let is_type_i = variant.is_type_i_for(&profile);

            // Assert: Type I implies exactly zero cost
            if is_type_i {
                assert_eq!(cost, 0.0, "Type I variant {:?} should have zero cost", variant);
            } else {
                assert!(cost > 0.0, "Non-Type I variant {:?} should have positive cost", variant);
            }
        }
    }
}
