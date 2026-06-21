//! ParallelCompileScheduler — REQ-UMK-31 编译管线三层并行化
//!
//! Phase 0: ScalarOpRegistry trace 按 op 并行提取，合并到统一 registry。
//! Phase 2: 融合组按 TopoLevel 并行 emit（同层级无数据依赖）。
//! Phase 3: ISA lowering 按 TopoLevel 并行（同层级融合组独立 lowerer 实例）。
//!
//! 并行度由编译器自动调度，保证语义正确性：
//! - Phase 0 并行提取 trace 后串行注入 registry（HashMap 非 Sync）。
//! - Phase 2/3 依赖 GroupDependencyAnalyzer 的 TopoLevel 分层，
//!   同层级 groups 无数据依赖可并行，跨层级串行。

use rayon::prelude::*;

use crate::compiler::graph::CompilerGraph;
use crate::compiler::registry::{OpKindKey, ScalarOpRegistry};
use crate::compiler::trace::OpTrace;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::group_dep::{GroupDependencyAnalyzer, TopoLevel};

// ── Phase 0: 并行 trace 提取 ──────────────────────────────────────

/// Phase 0 并行 trace 提取结果
#[derive(Debug)]
pub struct ParallelTraceResult {
    /// 每个操作的 OpKindKey → OpTrace 映射
    pub traces: Vec<(OpKindKey, OpTrace)>,
    /// 并行提取的 op 数量
    pub ops_processed: usize,
    /// 并行化实际使用的线程数
    pub threads_used: usize,
}

/// Phase 0: 从 CompilerGraph 的 ops 中并行提取 OpTrace。
///
/// 对于每个 op，使用 ScalarOpRegistry::key_from_op 获取 key，
/// 然后从预构建的全量 registry 中查找缓存的 trace。
/// 由于 registry 的 get_trace 是只读操作（&self），多线程可安全并行读取。
///
/// 返回 (key, trace) 对列表，调用方负责串行注入到目标 registry。
// @trace REQ-UMK-31 [entity:ENT-MEGA-KERNEL-FN] [api:POST /compile]
pub fn parallel_extract_traces(
    graph: &CompilerGraph,
    registry: &ScalarOpRegistry,
) -> ParallelTraceResult {
    let ops: Vec<_> = graph.ops.iter().collect();
    let ops_processed = ops.len();

    // 并行从 registry 读取缓存的 trace（只读操作，线程安全）
    let traces: Vec<(OpKindKey, OpTrace)> = ops
        .par_iter()
        .filter_map(|cop| {
            let key = ScalarOpRegistry::key_from_op(&cop.op);
            registry.get_trace(&key).map(|t| (key, t.clone()))
        })
        .collect();

    let threads_used = rayon::current_num_threads().min(ops_processed.max(1));

    ParallelTraceResult {
        traces,
        ops_processed,
        threads_used,
    }
}

// ── Phase 2: 融合组并行 emit 调度 ──────────────────────────────────

/// Phase 2 并行 emit 调度统计
#[derive(Debug, Clone)]
pub struct ParallelEmitStats {
    /// TopoLevel 总数
    pub num_levels: usize,
    /// 最大并行度（最宽层级的 group 数量）
    pub max_parallelism: usize,
    /// 每个层级的 group 数量
    pub groups_per_level: Vec<usize>,
}

/// Phase 2: 分析融合组的并行调度计划。
///
/// 使用 GroupDependencyAnalyzer 分析融合组间依赖，返回 TopoLevel 列表。
/// 同一层级的 groups 可以并行 emit，跨层级必须串行。
// @trace REQ-UMK-31 [entity:ENT-MEGA-KERNEL-FN] [api:POST /compile]
pub fn analyze_parallel_emit_plan(
    plan: &FusionPlan,
    graph: &CompilerGraph,
) -> (Vec<TopoLevel>, ParallelEmitStats) {
    let levels = GroupDependencyAnalyzer::analyze(plan, graph);
    let max_parallelism = levels
        .iter()
        .map(|l| l.groups.len())
        .max()
        .unwrap_or(0);
    let groups_per_level: Vec<usize> = levels.iter().map(|l| l.groups.len()).collect();

    let stats = ParallelEmitStats {
        num_levels: levels.len(),
        max_parallelism,
        groups_per_level,
    };

    (levels, stats)
}

// ── Phase 3: ISA Lowering 并行调度 ──────────────────────────────────

/// Phase 3 并行 lowering 调度统计
#[derive(Debug, Clone)]
pub struct ParallelLowerStats {
    /// TopoLevel 总数
    pub num_levels: usize,
    /// 最大并行度（最宽层级的 group 数量）
    pub max_parallelism: usize,
    /// 每个层级的 group 数量
    pub groups_per_level: Vec<usize>,
    /// 总 group 数量
    pub total_groups: usize,
}

/// Phase 3: 分析 ISA lowering 的并行调度计划。
///
/// 复用 GroupDependencyAnalyzer 的 TopoLevel 分层。
/// 同一层级的 groups 可以使用独立 lowerer 实例并行 lowering，
/// 跨层级必须按顺序串行 lowering（数据依赖）。
// @trace REQ-UMK-31 [entity:ENT-MEGA-KERNEL-FN] [api:POST /compile]
pub fn analyze_parallel_lower_plan(
    plan: &FusionPlan,
    graph: &CompilerGraph,
) -> (Vec<TopoLevel>, ParallelLowerStats) {
    let levels = GroupDependencyAnalyzer::analyze(plan, graph);
    let max_parallelism = levels
        .iter()
        .map(|l| l.groups.len())
        .max()
        .unwrap_or(0);
    let total_groups: usize = levels.iter().map(|l| l.groups.len()).sum();
    let groups_per_level: Vec<usize> = levels.iter().map(|l| l.groups.len()).collect();

    let stats = ParallelLowerStats {
        num_levels: levels.len(),
        max_parallelism,
        groups_per_level,
        total_groups,
    };

    (levels, stats)
}

// ── 统一调度器 ──────────────────────────────────────────────────────

/// 编译管线三层并行调度器
///
/// 管理三阶段并行调度策略，输出调度计划供各阶段使用。
// @trace REQ-UMK-31 [entity:ENT-MEGA-KERNEL-FN] [api:POST /compile]
pub struct ParallelCompileScheduler {
    /// Phase 0 是否启用并行 trace 提取
    pub phase0_parallel: bool,
    /// Phase 2 是否启用并行 fusion group emit
    pub phase2_parallel: bool,
    /// Phase 3 是否启用并行 ISA lowering
    pub phase3_parallel: bool,
    /// Phase 2/3 共享的 TopoLevel 调度计划
    topo_levels: Option<Vec<TopoLevel>>,
    /// Phase 2 调度统计
    emit_stats: Option<ParallelEmitStats>,
    /// Phase 3 调度统计
    lower_stats: Option<ParallelLowerStats>,
}

impl ParallelCompileScheduler {
    /// 创建新的并行调度器，根据条件自动决定是否启用并行。
    ///
    /// 启用条件：
    /// - ops 数量 > 1 时启用 Phase 0 并行 trace 提取
    /// - 融合组数量 > 1 且存在并行层级时启用 Phase 2 并行 emit
    /// - 融合组数量 > 1 且存在并行层级时启用 Phase 3 并行 lowering
    pub fn new() -> Self {
        ParallelCompileScheduler {
            phase0_parallel: true,
            phase2_parallel: true,
            phase3_parallel: true,
            topo_levels: None,
            emit_stats: None,
            lower_stats: None,
        }
    }

    /// 创建禁用所有并行的调度器（用于调试/对比）
    pub fn new_serial() -> Self {
        ParallelCompileScheduler {
            phase0_parallel: false,
            phase2_parallel: false,
            phase3_parallel: false,
            topo_levels: None,
            emit_stats: None,
            lower_stats: None,
        }
    }

    /// 执行 Phase 0: 并行 trace 提取，合并到目标 registry。
    ///
    /// 如果 phase0_parallel 为 true 且 ops 数量 > 1，使用 rayon 并行提取；
    /// 否则退化为串行遍历。
    pub fn phase0_extract_traces(
        &self,
        graph: &CompilerGraph,
        source_registry: &ScalarOpRegistry,
        target_registry: &mut ScalarOpRegistry,
    ) -> ParallelTraceResult {
        if !self.phase0_parallel || graph.ops.len() <= 1 {
            // 串行退路：逐 op 提取 trace
            let mut traces = Vec::new();
            for cop in &graph.ops {
                let key = ScalarOpRegistry::key_from_op(&cop.op);
                if let Some(t) = source_registry.get_trace(&key) {
                    traces.push((key, t.clone()));
                }
            }
            return ParallelTraceResult {
                traces,
                ops_processed: graph.ops.len(),
                threads_used: 1,
            };
        }

        let result = parallel_extract_traces(graph, source_registry);

        // 串行合并到目标 registry（HashMap 写入非线程安全）
        for (key, trace) in &result.traces {
            target_registry.inject_trace(key.clone(), trace.clone());
        }

        result
    }

    /// 分析 Phase 2/3 的并行调度计划。
    ///
    /// 必须在 fusion_plan 构建完成后调用。
    /// TopoLevel 分析结果被 Phase 2 和 Phase 3 共享（同一个依赖关系）。
    pub fn analyze_schedule(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
    ) {
        let (levels, emit_stats) = analyze_parallel_emit_plan(plan, graph);
        let total_groups: usize = levels.iter().map(|l| l.groups.len()).sum();
        let max_par = levels.iter().map(|l| l.groups.len()).max().unwrap_or(0);
        let groups_per_level: Vec<usize> = levels.iter().map(|l| l.groups.len()).collect();

        let lower_stats = ParallelLowerStats {
            num_levels: levels.len(),
            max_parallelism: max_par,
            groups_per_level,
            total_groups,
        };

        // 自动判断是否启用并行：只有当存在宽度 > 1 的层级时才有并行收益
        let has_parallelism = max_par > 1;
        if !has_parallelism {
            self.phase2_parallel = false;
            self.phase3_parallel = false;
        }

        self.topo_levels = Some(levels);
        self.emit_stats = Some(emit_stats);
        self.lower_stats = Some(lower_stats);
    }

    /// 获取 Phase 2/3 的 TopoLevel 调度计划。
    ///
    /// 必须在 analyze_schedule() 之后调用。
    pub fn topo_levels(&self) -> &[TopoLevel] {
        self.topo_levels.as_deref().unwrap_or(&[])
    }

    /// 获取 Phase 2 调度统计
    pub fn emit_stats(&self) -> Option<&ParallelEmitStats> {
        self.emit_stats.as_ref()
    }

    /// 获取 Phase 3 调度统计
    pub fn lower_stats(&self) -> Option<&ParallelLowerStats> {
        self.lower_stats.as_ref()
    }

    /// 判断 Phase 2 是否应该对给定层级并行执行
    pub fn should_parallelize_level(&self, level: &TopoLevel) -> bool {
        self.phase2_parallel && level.groups.len() > 1
    }

    /// 判断 Phase 3 是否应该对给定层级并行执行
    pub fn should_parallelize_lower_level(&self, level: &TopoLevel) -> bool {
        self.phase3_parallel && level.groups.len() > 1
    }

    /// 获取调度摘要报告（调试用）
    pub fn summary(&self) -> String {
        let p0 = if self.phase0_parallel { "ON" } else { "OFF" };
        let p2 = if self.phase2_parallel { "ON" } else { "OFF" };
        let p3 = if self.phase3_parallel { "ON" } else { "OFF" };
        let levels = self.topo_levels.as_ref().map(|l| l.len()).unwrap_or(0);
        let max_par = self.emit_stats.as_ref().map(|s| s.max_parallelism).unwrap_or(0);
        format!(
            "ParallelCompileScheduler: P0={} P2={} P3={} levels={} max_par={}",
            p0, p2, p3, levels, max_par
        )
    }
}

impl Default for ParallelCompileScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{SymDim, Op};
    use crate::compiler::fusion::{FusionGroup, FusionMode, GroupMarker};
    use crate::types::DType;
    use std::collections::HashMap;

    // ── Phase 0 tests ──────────────────────────────────────────────

    #[test]
    fn test_phase0_parallel_extract_empty_graph() {
        let graph = CompilerGraph::new();
        let registry = ScalarOpRegistry::with_defaults();
        let scheduler = ParallelCompileScheduler::new();
        let mut target = ScalarOpRegistry::new();
        let result = scheduler.phase0_extract_traces(&graph, &registry, &mut target);
        assert_eq!(result.ops_processed, 0);
        assert!(result.traces.is_empty());
    }

    #[test]
    fn test_phase0_parallel_extract_single_op() {
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("in", vec![SymDim::Concrete(4)], DType::F32);
        let t_out = graph.add_tensor("out", vec![SymDim::Concrete(4)], DType::F32);
        let _op = graph.add_op(Op::Silu, vec![t_in], vec![t_out], "silu");

        let registry = ScalarOpRegistry::with_defaults();
        let scheduler = ParallelCompileScheduler::new();
        let mut target = ScalarOpRegistry::new();
        let result = scheduler.phase0_extract_traces(&graph, &registry, &mut target);

        // Single op → serial fallback (ops.len() <= 1)
        assert_eq!(result.ops_processed, 1);
        assert_eq!(result.threads_used, 1);
    }

    #[test]
    fn test_phase0_serial_mode() {
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("in", vec![SymDim::Concrete(4)], DType::F32);
        let t_a = graph.add_tensor("a", vec![SymDim::Concrete(4)], DType::F32);
        let t_b = graph.add_tensor("b", vec![SymDim::Concrete(4)], DType::F32);
        let _op0 = graph.add_op(Op::Silu, vec![t_in], vec![t_a], "s0");
        let _op1 = graph.add_op(Op::Gelu, vec![t_in], vec![t_b], "g0");

        let registry = ScalarOpRegistry::with_defaults();
        let scheduler = ParallelCompileScheduler::new_serial();
        let mut target = ScalarOpRegistry::new();
        let result = scheduler.phase0_extract_traces(&graph, &registry, &mut target);

        assert_eq!(result.ops_processed, 2);
        assert_eq!(result.threads_used, 1);
    }

    #[test]
    fn test_phase0_parallel_vs_serial_equivalence() {
        // Build a graph with multiple ops
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("in", vec![SymDim::Concrete(4)], DType::F32);
        let t_a = graph.add_tensor("a", vec![SymDim::Concrete(4)], DType::F32);
        let t_b = graph.add_tensor("b", vec![SymDim::Concrete(4)], DType::F32);
        let _op0 = graph.add_op(Op::Silu, vec![t_in], vec![t_a], "s0");
        let _op1 = graph.add_op(Op::Gelu, vec![t_in], vec![t_b], "g0");

        let registry = ScalarOpRegistry::with_defaults();

        // Parallel extraction
        let par_scheduler = ParallelCompileScheduler::new();
        let mut par_target = ScalarOpRegistry::new();
        let par_result = par_scheduler.phase0_extract_traces(&graph, &registry, &mut par_target);

        // Serial extraction
        let ser_scheduler = ParallelCompileScheduler::new_serial();
        let mut ser_target = ScalarOpRegistry::new();
        let ser_result = ser_scheduler.phase0_extract_traces(&graph, &registry, &mut ser_target);

        // Same number of ops processed
        assert_eq!(par_result.ops_processed, ser_result.ops_processed);

        // Same trace keys extracted
        let par_keys: HashMap<_, _> = par_result
            .traces
            .iter()
            .map(|(k, _)| (format!("{:?}", k), ()))
            .collect();
        let ser_keys: HashMap<_, _> = ser_result
            .traces
            .iter()
            .map(|(k, _)| (format!("{:?}", k), ()))
            .collect();
        assert_eq!(par_keys.len(), ser_keys.len());
    }

    // ── Phase 2/3 scheduling tests ─────────────────────────────────

    #[test]
    fn test_scheduler_new_defaults() {
        let s = ParallelCompileScheduler::new();
        assert!(s.phase0_parallel);
        assert!(s.phase2_parallel);
        assert!(s.phase3_parallel);
        assert!(s.topo_levels.is_none());
    }

    #[test]
    fn test_scheduler_new_serial() {
        let s = ParallelCompileScheduler::new_serial();
        assert!(!s.phase0_parallel);
        assert!(!s.phase2_parallel);
        assert!(!s.phase3_parallel);
    }

    #[test]
    fn test_analyze_schedule_empty_plan() {
        let graph = CompilerGraph::new();
        let plan = FusionPlan {
            groups: Vec::new(),
            op_to_group: HashMap::new(),
        };
        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        // Empty plan → no parallelism
        assert!(!scheduler.phase2_parallel);
        assert!(!scheduler.phase3_parallel);
        assert_eq!(scheduler.topo_levels().len(), 0);
    }

    #[test]
    fn test_analyze_schedule_single_group() {
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("in", vec![SymDim::Concrete(4)], DType::F32);
        let t_out = graph.add_tensor("out", vec![SymDim::Concrete(4)], DType::F32);
        let op0 = graph.add_op(Op::Silu, vec![t_in], vec![t_out], "silu");

        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op0,
                epilogue: Vec::new(),
                mode: FusionMode::Standalone,
                ops: vec![op0],
                multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                dominant_dtype: None,
                marker: GroupMarker::None,
                is_layer_group: false,
                hetero_layer_type: None,
            }],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0);
                m
            },
        };

        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        // Single group → max_parallelism = 1, no parallel benefit
        assert!(!scheduler.phase2_parallel);
        assert!(!scheduler.phase3_parallel);
        assert_eq!(scheduler.topo_levels().len(), 1);
    }

    #[test]
    fn test_analyze_schedule_parallel_groups() {
        let mut graph = CompilerGraph::new();
        let t_in0 = graph.add_tensor("in0", vec![SymDim::Concrete(4)], DType::F32);
        let t_in1 = graph.add_tensor("in1", vec![SymDim::Concrete(4)], DType::F32);
        let t_out0 = graph.add_tensor("out0", vec![SymDim::Concrete(4)], DType::F32);
        let t_out1 = graph.add_tensor("out1", vec![SymDim::Concrete(4)], DType::F32);

        let op0 = graph.add_op(Op::Silu, vec![t_in0], vec![t_out0], "s0");
        let op1 = graph.add_op(Op::Gelu, vec![t_in1], vec![t_out1], "g0");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0,
                    anchor: op0,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op0],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                },
                FusionGroup {
                    id: 1,
                    anchor: op1,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op1],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                },
            ],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0);
                m.insert(op1, 1);
                m
            },
        };

        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        // Two independent groups → max_parallelism = 2, should enable parallel
        assert!(scheduler.phase2_parallel);
        assert!(scheduler.phase3_parallel);
        assert_eq!(scheduler.topo_levels().len(), 1);
        assert_eq!(scheduler.topo_levels()[0].groups.len(), 2);

        let stats = scheduler.emit_stats().unwrap();
        assert_eq!(stats.max_parallelism, 2);
    }

    #[test]
    fn test_analyze_schedule_sequential_groups() {
        // Chain: op0 → op1 → op2 (all sequential, no parallelism)
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("input", vec![SymDim::Concrete(4)], DType::F32);
        let t_a = graph.add_tensor("a", vec![SymDim::Concrete(4)], DType::F32);
        let t_b = graph.add_tensor("b", vec![SymDim::Concrete(4)], DType::F32);

        let op0 = graph.add_op(Op::Silu, vec![t_in], vec![t_a], "s0");
        let op1 = graph.add_op(Op::Gelu, vec![t_a], vec![t_b], "g0");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0,
                    anchor: op0,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op0],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                },
                FusionGroup {
                    id: 1,
                    anchor: op1,
                    epilogue: Vec::new(),
                    mode: FusionMode::Standalone,
                    ops: vec![op1],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
                    hetero_layer_type: None,
                },
            ],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0);
                m.insert(op1, 1);
                m
            },
        };

        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        // Sequential chain → max_parallelism = 1, no parallel benefit
        assert!(!scheduler.phase2_parallel);
        assert!(!scheduler.phase3_parallel);
        assert_eq!(scheduler.topo_levels().len(), 2);
    }

    #[test]
    fn test_should_parallelize_level() {
        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.phase2_parallel = true;

        let wide_level = TopoLevel { level: 0, groups: vec![0, 1, 2, 3] };
        let narrow_level = TopoLevel { level: 1, groups: vec![4] };

        assert!(scheduler.should_parallelize_level(&wide_level));
        assert!(!scheduler.should_parallelize_level(&narrow_level));
    }

    #[test]
    fn test_should_parallelize_lower_level() {
        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.phase3_parallel = true;

        let wide_level = TopoLevel { level: 0, groups: vec![0, 1] };
        let narrow_level = TopoLevel { level: 1, groups: vec![2] };

        assert!(scheduler.should_parallelize_lower_level(&wide_level));
        assert!(!scheduler.should_parallelize_lower_level(&narrow_level));
    }

    #[test]
    fn test_should_parallelize_disabled() {
        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.phase2_parallel = false;
        scheduler.phase3_parallel = false;

        let wide_level = TopoLevel { level: 0, groups: vec![0, 1, 2] };
        assert!(!scheduler.should_parallelize_level(&wide_level));
        assert!(!scheduler.should_parallelize_lower_level(&wide_level));
    }

    #[test]
    fn test_summary_report() {
        let mut scheduler = ParallelCompileScheduler::new();
        let graph = CompilerGraph::new();
        let plan = FusionPlan {
            groups: Vec::new(),
            op_to_group: HashMap::new(),
        };
        scheduler.analyze_schedule(&plan, &graph);

        let summary = scheduler.summary();
        assert!(summary.contains("P0=ON"));
        assert!(summary.contains("P2=OFF"));
        assert!(summary.contains("P3=OFF"));
    }

    #[test]
    fn test_diamond_graph_parallelism() {
        // Diamond: op0 → op1, op2 → op3
        // Level 0: [g0], Level 1: [g1, g2] (parallel), Level 2: [g3]
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("input", vec![SymDim::Concrete(4)], DType::F32);
        let t_a = graph.add_tensor("a", vec![SymDim::Concrete(4)], DType::F32);
        let t_b = graph.add_tensor("b", vec![SymDim::Concrete(4)], DType::F32);
        let t_c = graph.add_tensor("c", vec![SymDim::Concrete(4)], DType::F32);
        let t_out = graph.add_tensor("out", vec![SymDim::Concrete(4)], DType::F32);

        let op0 = graph.add_op(Op::Silu, vec![t_in], vec![t_a], "src");
        let op1 = graph.add_op(Op::Silu, vec![t_a], vec![t_b], "left");
        let op2 = graph.add_op(Op::Silu, vec![t_a], vec![t_c], "right");
        let op3 = graph.add_op(Op::Add, vec![t_b, t_c], vec![t_out], "merge");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0, anchor: op0, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op0],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
                FusionGroup {
                    id: 1, anchor: op1, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op1],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
                FusionGroup {
                    id: 2, anchor: op2, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op2],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
                FusionGroup {
                    id: 3, anchor: op3, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op3],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
            ],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0); m.insert(op1, 1); m.insert(op2, 2); m.insert(op3, 3);
                m
            },
        };

        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        // Diamond has parallel level → should enable parallel
        assert!(scheduler.phase2_parallel);
        assert!(scheduler.phase3_parallel);
        assert_eq!(scheduler.topo_levels().len(), 3);

        let stats = scheduler.emit_stats().unwrap();
        assert_eq!(stats.max_parallelism, 2);
        assert_eq!(stats.groups_per_level, vec![1, 2, 1]);
    }

    #[test]
    fn test_lower_stats_total_groups() {
        let mut graph = CompilerGraph::new();
        let t_in0 = graph.add_tensor("in0", vec![SymDim::Concrete(4)], DType::F32);
        let t_in1 = graph.add_tensor("in1", vec![SymDim::Concrete(4)], DType::F32);
        let t_out0 = graph.add_tensor("out0", vec![SymDim::Concrete(4)], DType::F32);
        let t_out1 = graph.add_tensor("out1", vec![SymDim::Concrete(4)], DType::F32);

        let op0 = graph.add_op(Op::Silu, vec![t_in0], vec![t_out0], "s0");
        let op1 = graph.add_op(Op::Gelu, vec![t_in1], vec![t_out1], "g0");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0, anchor: op0, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op0],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
                FusionGroup {
                    id: 1, anchor: op1, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op1],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
            ],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0); m.insert(op1, 1);
                m
            },
        };

        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        let lower_stats = scheduler.lower_stats().unwrap();
        assert_eq!(lower_stats.total_groups, 2);
        assert_eq!(lower_stats.num_levels, 1);
    }

    #[test]
    fn test_default_trait() {
        let scheduler = ParallelCompileScheduler::default();
        assert!(scheduler.phase0_parallel);
        assert!(scheduler.phase2_parallel);
        assert!(scheduler.phase3_parallel);
    }

    #[test]
    fn test_parallel_extract_traces_direct() {
        // Test the direct parallel_extract_traces function
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor("in", vec![SymDim::Concrete(4)], DType::F32);
        let t_a = graph.add_tensor("a", vec![SymDim::Concrete(4)], DType::F32);
        let t_b = graph.add_tensor("b", vec![SymDim::Concrete(4)], DType::F32);
        let _op0 = graph.add_op(Op::Silu, vec![t_in], vec![t_a], "s0");
        let _op1 = graph.add_op(Op::Gelu, vec![t_in], vec![t_b], "g0");

        let registry = ScalarOpRegistry::with_defaults();
        let result = parallel_extract_traces(&graph, &registry);

        assert_eq!(result.ops_processed, 2);
        assert!(result.threads_used >= 1);
    }

    #[test]
    fn test_analyze_parallel_emit_plan_direct() {
        let mut graph = CompilerGraph::new();
        let t_in0 = graph.add_tensor("in0", vec![SymDim::Concrete(4)], DType::F32);
        let t_in1 = graph.add_tensor("in1", vec![SymDim::Concrete(4)], DType::F32);
        let t_out0 = graph.add_tensor("out0", vec![SymDim::Concrete(4)], DType::F32);
        let t_out1 = graph.add_tensor("out1", vec![SymDim::Concrete(4)], DType::F32);

        let op0 = graph.add_op(Op::Silu, vec![t_in0], vec![t_out0], "s0");
        let op1 = graph.add_op(Op::Gelu, vec![t_in1], vec![t_out1], "g0");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0, anchor: op0, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op0],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
                FusionGroup {
                    id: 1, anchor: op1, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op1],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
            ],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0); m.insert(op1, 1);
                m
            },
        };

        let (levels, stats) = analyze_parallel_emit_plan(&plan, &graph);
        assert_eq!(levels.len(), 1);
        assert_eq!(stats.max_parallelism, 2);
        assert_eq!(stats.num_levels, 1);
    }

    #[test]
    fn test_analyze_parallel_lower_plan_direct() {
        let mut graph = CompilerGraph::new();
        let t_in0 = graph.add_tensor("in0", vec![SymDim::Concrete(4)], DType::F32);
        let t_in1 = graph.add_tensor("in1", vec![SymDim::Concrete(4)], DType::F32);
        let t_out0 = graph.add_tensor("out0", vec![SymDim::Concrete(4)], DType::F32);
        let t_out1 = graph.add_tensor("out1", vec![SymDim::Concrete(4)], DType::F32);

        let op0 = graph.add_op(Op::Silu, vec![t_in0], vec![t_out0], "s0");
        let op1 = graph.add_op(Op::Gelu, vec![t_in1], vec![t_out1], "g0");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0, anchor: op0, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op0],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
                FusionGroup {
                    id: 1, anchor: op1, epilogue: Vec::new(),
                    mode: FusionMode::Standalone, ops: vec![op1],
                    multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                    dominant_dtype: None, marker: GroupMarker::None,
                    is_layer_group: false, hetero_layer_type: None,
                },
            ],
            op_to_group: {
                let mut m = HashMap::new();
                m.insert(op0, 0); m.insert(op1, 1);
                m
            },
        };

        let (levels, stats) = analyze_parallel_lower_plan(&plan, &graph);
        assert_eq!(levels.len(), 1);
        assert_eq!(stats.max_parallelism, 2);
        assert_eq!(stats.total_groups, 2);
    }

    #[test]
    fn test_wide_fan_out_schedule() {
        // One producer feeds 5 independent consumers → 2 levels
        let mut graph = CompilerGraph::new();
        let t_root = graph.add_tensor("root", vec![SymDim::Concrete(4)], DType::F32);
        let t_src = graph.add_tensor("src", vec![SymDim::Concrete(4)], DType::F32);
        let op0 = graph.add_op(Op::Silu, vec![t_root], vec![t_src], "src");

        let mut consumer_ops = Vec::new();
        for i in 0..5 {
            let t_out = graph.add_tensor(&format!("out{}", i), vec![SymDim::Concrete(4)], DType::F32);
            let opi = graph.add_op(Op::Silu, vec![t_src], vec![t_out], &format!("c{}", i));
            consumer_ops.push(opi);
        }

        let mut groups = vec![FusionGroup {
            id: 0, anchor: op0, epilogue: Vec::new(),
            mode: FusionMode::Standalone, ops: vec![op0],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None, marker: GroupMarker::None,
            is_layer_group: false, hetero_layer_type: None,
        }];
        let mut op_map = HashMap::new();
        op_map.insert(op0, 0);
        for (i, &opi) in consumer_ops.iter().enumerate() {
            groups.push(FusionGroup {
                id: i + 1, anchor: opi, epilogue: Vec::new(),
                mode: FusionMode::Standalone, ops: vec![opi],
                multi_output: crate::compiler::graph::MultiOutputConfig::single(),
                dominant_dtype: None, marker: GroupMarker::None,
                is_layer_group: false, hetero_layer_type: None,
            });
            op_map.insert(opi, i + 1);
        }

        let plan = FusionPlan { groups, op_to_group: op_map };

        let mut scheduler = ParallelCompileScheduler::new();
        scheduler.analyze_schedule(&plan, &graph);

        assert!(scheduler.phase2_parallel);
        assert!(scheduler.phase3_parallel);

        let stats = scheduler.emit_stats().unwrap();
        assert_eq!(stats.max_parallelism, 5);
        assert_eq!(stats.num_levels, 2);

        // Level 0 has 1 group, level 1 has 5 groups
        assert!(scheduler.should_parallelize_level(&scheduler.topo_levels()[1]));
        assert!(!scheduler.should_parallelize_level(&scheduler.topo_levels()[0]));
    }
}
