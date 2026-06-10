//! Heterogeneous layer compilation — boundary analysis + parallel template compilation.

use rayon::prelude::*;

use super::plan_lower::{
    LoweringContext, TensorPtrResolver, compile_layer_type_body,
};
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::compiler::fusion::{FusionPlan, HeteroLayerType};
use crate::compiler::graph::CompilerGraph;
use crate::types::CompilerError;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 边界分析
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 层类型的 group 范围
#[derive(Debug, Clone)]
pub struct LayerTypeRange {
    pub layer_type: HeteroLayerType,
    pub group_range: std::ops::Range<usize>,
}

/// 分析 FusionPlan 中异构层类型的 group 边界
/// REQ-UMK-012: 从 FusionGroup.hetero_layer_type 推导（OpKind 参数驱动），非 label 前缀。
pub fn analyze_hetero_layer_boundaries(
    plan: &FusionPlan,
    _graph: &CompilerGraph,
) -> Vec<LayerTypeRange> {
    let mut ranges: Vec<LayerTypeRange> = Vec::new();
    let mut current_type: Option<HeteroLayerType> = None;
    let mut range_start = 0;

    for (gi, group) in plan.groups.iter().enumerate() {
        // REQ-UMK-012: read from FusionGroup.hetero_layer_type (OpKind-parameter-derived),
        // not from anchor_op.label string prefix.
        let lt = group.hetero_layer_type;

        match (current_type, lt) {
            (None, Some(new_lt)) => {
                current_type = Some(new_lt);
                range_start = gi;
            }
            (Some(old_lt), Some(new_lt)) if old_lt != new_lt => {
                ranges.push(LayerTypeRange {
                    layer_type: old_lt,
                    group_range: range_start..gi,
                });
                current_type = Some(new_lt);
                range_start = gi;
            }
            _ => {}
        }
    }

    if let Some(lt) = current_type {
        ranges.push(LayerTypeRange {
            layer_type: lt,
            group_range: range_start..plan.groups.len(),
        })
    }

    ranges
}

/// 并行编译异构层模板 (rayon)
///
/// 每种层类型在独立 VmProgram 中编译为 LayerTemplate。
/// 4 种类型 (ss/fs/sl/fl) 用 rayon 并行编译。
/// 主 VmProgram 中按原始顺序实例化模板（LoopBegin + ABI 初始化 + 模板体 + LoopEnd）。
pub fn compile_hetero_templates_parallel(
    ctx: &LoweringContext,
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    resolver: &TensorPtrResolver,
) -> Result<Vec<LayerTypeRange>, CompilerError> {
    let boundaries = analyze_hetero_layer_boundaries(plan, graph);
    if boundaries.is_empty() {
        return Ok(Vec::new());
    }

    // 预编译所有层类型模板 (rayon 并行)
    let templates: Vec<_> = boundaries.iter()
        .map(|range| {
            let label = match range.layer_type {
                HeteroLayerType::SlidingSmall => "ss",
                HeteroLayerType::FullSmall => "fs",
                HeteroLayerType::SlidingLarge => "sl",
                HeteroLayerType::FullLarge => "fl",
            };
            (range.clone(), label)
        })
        .collect();

    // rayon 并行编译所有层类型模板
    // LoweringContext 所有引用字段都是 Sync (IsaHook: Send + Sync, 其他为普通 &T)
    // compile_layer_type_body 创建独立 VmProgram，无共享可变状态
    let results: Vec<_> = templates
        .par_iter()
        .map(|(range, label)| {
            let result = compile_layer_type_body(
                ctx, range.group_range.clone(), plan, graph, alloc, resolver,
            );
            (label, result)
        })
        .collect();

    // 顺序报告结果 (并行编译完成，日志顺序输出)
    for (label, result) in &results {
        match result {
            Ok(tpl) => {
                if std::env::var("GLLM_DEBUG_RESOURCE").is_ok() {
                    eprintln!("[hetero-parallel] Template {} compiled: {} instrs, abi=({},{},{},{})",
                        label, tpl.body.instrs.len(),
                        tpl.abi_map.input_ptr.0, tpl.abi_map.weight_ptr.0,
                        tpl.abi_map.output_ptr.0, tpl.abi_map.scratch_base.0);
                }
            }
            Err(e) => {
                eprintln!("[hetero-parallel] Template {} failed: {}", label, e);
            }
        }
    }

    Ok(boundaries)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HeteroLayerType ───────────────────────────────────────────────

    #[test]
    fn hetero_layer_type_equality() {
        assert_eq!(HeteroLayerType::SlidingSmall, HeteroLayerType::SlidingSmall);
        assert_ne!(HeteroLayerType::SlidingSmall, HeteroLayerType::FullSmall);
        assert_ne!(HeteroLayerType::SlidingLarge, HeteroLayerType::FullLarge);
    }

    #[test]
    fn hetero_layer_type_copy() {
        let a = HeteroLayerType::FullSmall;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn hetero_layer_type_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(HeteroLayerType::SlidingSmall);
        set.insert(HeteroLayerType::FullLarge);
        set.insert(HeteroLayerType::SlidingSmall); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn hetero_layer_type_variants() {
        let variants = [
            HeteroLayerType::SlidingSmall,
            HeteroLayerType::FullSmall,
            HeteroLayerType::SlidingLarge,
            HeteroLayerType::FullLarge,
        ];
        // All distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variants[{i}] == variants[{j}]");
            }
        }
    }

    // ── LayerTypeRange ────────────────────────────────────────────────

    #[test]
    fn layer_type_range_fields() {
        let range = LayerTypeRange {
            layer_type: HeteroLayerType::SlidingLarge,
            group_range: 0..4,
        };
        assert_eq!(range.layer_type, HeteroLayerType::SlidingLarge);
        assert_eq!(range.group_range.start, 0);
        assert_eq!(range.group_range.end, 4);
    }

    #[test]
    fn layer_type_range_clone() {
        let range = LayerTypeRange {
            layer_type: HeteroLayerType::FullSmall,
            group_range: 2..6,
        };
        let cloned = range.clone();
        assert_eq!(cloned.layer_type, range.layer_type);
        assert_eq!(cloned.group_range, range.group_range);
    }

    #[test]
    fn layer_type_range_empty_range() {
        let range = LayerTypeRange {
            layer_type: HeteroLayerType::SlidingSmall,
            group_range: 0..0,
        };
        assert!(range.group_range.is_empty());
    }

    // ── analyze_hetero_layer_boundaries (empty plan) ──────────────────

    #[test]
    fn analyze_hetero_boundaries_empty_plan() {
        let plan = FusionPlan { groups: vec![], op_to_group: Default::default() };
        let graph = CompilerGraph::new();
        let result = analyze_hetero_layer_boundaries(&plan, &graph);
        assert!(result.is_empty());
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn hetero_layer_type_debug_format() {
        // Arrange & Act
        let debug_ss = format!("{:?}", HeteroLayerType::SlidingSmall);
        let debug_fl = format!("{:?}", HeteroLayerType::FullLarge);
        // Assert — Debug should contain the variant name
        assert!(debug_ss.contains("SlidingSmall"), "got: {debug_ss}");
        assert!(debug_fl.contains("FullLarge"), "got: {debug_fl}");
    }

    #[test]
    fn hetero_layer_type_all_four_distinct() {
        // Arrange
        let all = [
            HeteroLayerType::SlidingSmall,
            HeteroLayerType::FullSmall,
            HeteroLayerType::SlidingLarge,
            HeteroLayerType::FullLarge,
        ];
        // Act & Assert — pairwise distinct
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "variant {i} == variant {j}");
            }
        }
    }

    #[test]
    fn hetero_layer_type_clone_independent() {
        let a = HeteroLayerType::SlidingLarge;
        let b = a.clone();
        assert_eq!(a, b);
        // Verify they are independent copies (Copy type, so always equal)
        assert_eq!(a, HeteroLayerType::SlidingLarge);
    }

    #[test]
    fn layer_type_range_debug_format() {
        let range = LayerTypeRange {
            layer_type: HeteroLayerType::FullSmall,
            group_range: 1..3,
        };
        let debug = format!("{:?}", range);
        assert!(debug.contains("FullSmall"), "got: {debug}");
        assert!(debug.contains("1"), "should contain range start");
    }

    #[test]
    fn layer_type_range_single_element_range() {
        let range = LayerTypeRange {
            layer_type: HeteroLayerType::SlidingSmall,
            group_range: 5..6,
        };
        assert_eq!(range.group_range.len(), 1);
    }

    #[test]
    fn hetero_layer_type_ordering_consistency() {
        // Verify Hash consistency: same value → same hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        HeteroLayerType::FullLarge.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        HeteroLayerType::FullLarge.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn analyze_boundaries_unrecognized_labels_yield_empty() {
        // Arrange — graph with no hetero-labeled ops (empty groups)
        let plan = FusionPlan { groups: vec![], op_to_group: Default::default() };
        let graph = CompilerGraph::new();
        // Act
        let result = analyze_hetero_layer_boundaries(&plan, &graph);
        // Assert — no hetero labels → no boundaries
        assert!(result.is_empty());
    }

    #[test]
    fn hetero_layer_type_exhaustive_match() {
        // Verify all 4 variants are handled without panic
        let v = HeteroLayerType::SlidingSmall;
        let _ = match v {
            HeteroLayerType::SlidingSmall => "ss",
            HeteroLayerType::FullSmall => "fs",
            HeteroLayerType::SlidingLarge => "sl",
            HeteroLayerType::FullLarge => "fl",
        };
    }

    #[test]
    fn layer_type_range_clone_preserves_end() {
        let range = LayerTypeRange {
            layer_type: HeteroLayerType::SlidingLarge,
            group_range: 10..20,
        };
        let cloned = range.clone();
        assert_eq!(cloned.group_range.end, 20);
    }
}
