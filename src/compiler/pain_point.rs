//! R0 PainPointAnalyzer — (Model×Device) 编译时瓶颈推导 (SSOT: §3.9)
//!
//! 纯静态分析: 从 CompilerGraph 的 GEMM 形状 + DeviceProfile 的峰值性能
//! 推导每个 GEMM 的瓶颈类型和最优融合策略。
//! 零运行时依赖 — 所有输入在编译时已知。

use std::collections::HashMap;
use crate::compiler::graph::{CompilerGraph, Op, OpId, OpKind};
use crate::dispatch::device_profile::DeviceProfile;

/// GEMM 在模型中的角色 (影响融合策略选择)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmRole {
    QkvProjection,
    OutputProjection,
    GateUpProjection,
    DownProjection,
    LmHead,
    Other,
}

/// 瓶颈类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BottleneckType {
    /// AI < ridge: 带宽瓶颈 → 融合消除内存访问收益最大
    MemoryBound { bandwidth_utilization: f64 },
    /// AI >= ridge: 计算瓶颈 → 融合收益缩减但寄存器效率仍有益
    ComputeBound { compute_utilization: f64 },
    /// 延迟瓶颈 (小矩阵, kernel launch overhead)
    LatencyBound { estimated_latency_ns: f64 },
}

/// §0.2.9 虚拟执行模式 — 硬件无关的执行意图 (SSOT: §3.9)
///
/// 编码计算策略的意图，由 DeviceProfile 决定具体物化方式。
/// R0 PainPointAnalyzer 根据 (GemmShape × BottleneckType × DeviceProfile) 推导，
/// ISA Lowering codegen 消费此字段选择物化策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecPattern {
    /// 分块 GEMM: 最大化计算吞吐 (compute-bound 大矩阵)
    /// CPU: BLIS 风格 MC×KC×NC 分块; GPU: tensor core wgmma/wmma
    TileGemm {
        /// 分块行数 (CPU: mr 微核行数; GPU: CTA M 分块)
        tile_m: usize,
        /// 分块列数 (CPU: nr 微核列数; GPU: CTA N 分块)
        tile_n: usize,
        /// 分块深度 (CPU: kc 缓存分块; GPU: CTA K 分块)
        tile_k: usize,
        /// GPU Warp 级 M 分块 (0 = 不使用 GPU 三级分块, CPU 忽略此字段)
        warp_m: usize,
        /// GPU Warp 级 N 分块 (0 = 不使用 GPU 三级分块, CPU 忽略此字段)
        warp_n: usize,
        /// GPU MMA 指令 K 深度 (0 = 默认, CPU 忽略此字段)
        mma_k: usize,
        /// 双缓冲流水线深度 (0/1 = 无流水线, 2 = ping-pong double buffer, 3 = 三缓冲)
        /// GPU: cp.async/TMA 加载与 MMA 计算重叠; CPU: prefetch 与计算重叠
        pipeline_depth: usize,
    },
    /// 共享内存 tile: 最大化缓存利用 (memory-bound 中等矩阵)
    /// CPU: cache-resident tile; GPU: shared memory tiling
    SharedMemTile {
        tile_rows: usize,
        tile_cols: usize,
    },
    /// 异步流水线: producer-consumer 双缓冲
    /// CPU: prefetch + compute overlap; GPU: TMA + wgmma pipeline
    AsyncPipeline,
    /// 标量循环: 小矩阵 / 调试 / 不支持 SIMD
    ScalarLoop,
}

/// 从 (DeviceProfile × GemmShape × BottleneckType) 推导 ExecPattern
///
/// §0.2.9: tile 大小随 GEMM M/N/K 比例变化 — 不是固定 ISA 常量。
/// - decode (M=1): GEMV，tile_m=1, tile_n 按缓存行对齐
/// - 窄高 (M<tm, N>tk): 减小 tile_m, 保持 tile_n
/// - K 很大: 减小 tile_k 适应 L1
pub fn derive_exec_pattern(
    m: usize, n: usize, k: usize,
    bottleneck: &BottleneckType,
    profile: &DeviceProfile,
) -> ExecPattern {
    if m <= 1 && n * k <= 256 {
        return ExecPattern::ScalarLoop;
    }

    let (tm_base, tn_base, tk_base) = profile.gemm_tile_sizes();

    // GEMV (decode M=1): tile_m=1, 无 MC 分块
    if m == 1 {
        return match bottleneck {
            BottleneckType::LatencyBound { .. } => ExecPattern::ScalarLoop,
            _ => ExecPattern::TileGemm {
                tile_m: 1,
                tile_n: tn_base,
                tile_k: tk_base,
                warp_m: 0,
                warp_n: 0,
                mma_k: 0,
                pipeline_depth: 0,
            },
        };
    }

    // 窄高矩阵 (M < tm): 收缩 tile_m 以减少尾部浪费
    let tm = if m < tm_base { m } else { tm_base };

    // K 很大: 收缩 tile_k 以适应 L1 缓存
    let tk = if k > 4 * tk_base {
        (tk_base / 2).max(1)
    } else {
        tk_base
    };

    // 宽矮矩阵 (N < tn): 收缩 tile_n
    let tn = if n < tn_base { n } else { tn_base };

    match bottleneck {
        BottleneckType::ComputeBound { .. } => {
            ExecPattern::TileGemm { tile_m: tm, tile_n: tn, tile_k: tk, warp_m: 0, warp_n: 0, mma_k: 0, pipeline_depth: 0 }
        }
        BottleneckType::MemoryBound { .. } => {
            ExecPattern::SharedMemTile { tile_rows: tm, tile_cols: tn }
        }
        BottleneckType::LatencyBound { .. } => ExecPattern::ScalarLoop,
    }
}

/// §0.2.10 虚拟并行 — 逻辑并行度 → 物理 SIMD/warp/wavefront 映射
///
/// 编码并行度意图，由 DeviceProfile 决定物化方式。
/// CPU: YMM/ZMM/NEON/SVE 向量寄存器; GPU: warp/wavefront SIMT 并行
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismDesc {
    /// SIMD 向量化: 单指令多数据并行
    /// CPU: for lane in 0..simd_width; GPU: 单条 vector 指令
    SimdVectorize {
        /// SIMD 元素宽度 (CPU: f32 lanes per register; GPU: warp size)
        element_width: usize,
        /// 循环展开因子 (ILP)
        unroll_factor: usize,
    },
    /// 线程并行: 多核/多线程
    /// CPU: Rayon MC 循环; GPU: Grid/Block 调度
    ThreadParallel {
        /// 并行维度 (CPU: physical_cores; GPU: num_sms)
        parallel_dim: usize,
        /// 并行粒度 (cache-line / warp 粒度)
        granularity: usize,
    },
    /// Wave 并行: Multi-Wave 内部调度
    /// GPU: Grid launch 多 Thread Block; CPU: NUMA 绑定多线程
    WaveParallel {
        num_waves: usize,
    },
    /// Warp 协作: producer-consumer 双缓冲
    /// GPU: Thread Block Cluster; CPU: N/A
    WarpCooperative,
}

/// 从 (DeviceProfile × GemmShape) 推导 ParallelismDesc
///
/// §0.2.10: 并行度随 GEMM M/N/K 差异化 — decode vs prefill。
/// - decode (M=1): GEMV，unroll_factor=1（K 展开无收益）
/// - prefill (M>1): 正常 SIMD 向量化 + K 展开
pub fn derive_parallelism(
    profile: &DeviceProfile,
    m: usize, _n: usize, _k: usize,
) -> ParallelismDesc {
    let simd_width = profile.simd_width_f32();
    let k_unroll = profile.k_unroll_factor();

    let unroll_factor = if m <= 1 {
        1 // GEMV: K 展开无收益，单行扫描
    } else {
        k_unroll
    };

    ParallelismDesc::SimdVectorize {
        element_width: if simd_width > 0 { simd_width } else { 1 },
        unroll_factor,
    }
}

/// 从 DeviceProfile 推导线程并行参数
pub fn derive_thread_parallelism(profile: &DeviceProfile) -> ParallelismDesc {
    ParallelismDesc::ThreadParallel {
        parallel_dim: profile.physical_cores,
        granularity: 64, // cache line size
    }
}

/// GEMM 融合策略优先级 (§3.2.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPriority {
    /// P0: GEMM 累加器 → 后续 op 在寄存器完成
    EpilogueInjection,
    /// P1: Norm 输出直喂 GEMM
    NormIntoGemm,
    /// P1: Norm 嵌入 GEMM MC 循环
    TileLevelFusion,
    /// P1: Norm 完整计算后驻留 L1/L2
    ComputeRoot,
    /// P2: 结构性融合
    QkvSharedInput,
    /// P2: Gate+Up→Act→Mul→Down
    FfnBlock,
    /// P3: Elementwise 链合并
    LoopFusion,
}

/// 单个 GEMM 的瓶颈分析结果
#[derive(Debug, Clone)]
pub struct GemmBottleneck {
    pub gemm_role: GemmRole,
    pub shape: (usize, usize, usize), // (M, N, K) — M 用 max_for_allocation
    pub arithmetic_intensity: f64,
    pub ridge_point: f64,
    pub bottleneck: BottleneckType,
    pub optimal_fusion: FusionPriority,
    pub fusion_benefits: HashMap<FusionPriority, f64>,
    /// §0.2.9 虚拟执行模式: 硬件感知的 GEMM 执行策略
    pub exec_pattern: ExecPattern,
    /// §0.2.10 虚拟并行: per-op 并行度 (decode vs prefill 差异化)
    pub parallelism: ParallelismDesc,
}

/// 全模型性能分析结果 (R0 输出)
#[derive(Debug, Clone)]
pub struct OpBottleneckMap {
    pub gemm_bottlenecks: HashMap<OpId, GemmBottleneck>,
    pub ridge_point: f64,
}

/// 编译时痛点分析器
pub struct PainPointAnalyzer;

impl PainPointAnalyzer {
    /// 纯静态分析: (CompilerGraph × DeviceProfile) → OpBottleneckMap
    ///
    /// 零运行时依赖 — M 使用 max_for_allocation，结果保守但安全。
    pub fn analyze(
        graph: &CompilerGraph,
        device: &DeviceProfile,
    ) -> OpBottleneckMap {
        let ridge = if device.peak_bandwidth_gbs > 0.0 {
            device.peak_gflops_f32 * 1e9 / (device.peak_bandwidth_gbs * 1e9)
        } else {
            10.0
        };

        let mut gemm_bottlenecks = HashMap::new();

        // ARCH-JIT-DATA-YIELDS: pre-compute GEMM op list in topological order
        // to avoid per-call full-graph scans in classify_gemm_role.
        let all_gemms_in_order: Vec<OpId> = graph.topological_sort()
            .into_iter()
            .filter(|&op_id| {
                graph.op(op_id).is_some_and(|op| matches!(op.op_v2_resolved(graph),
                    Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_))))
            })
            .collect();

        for op_id in graph.topological_sort() {
            let op = match graph.op(op_id) {
                Some(o) => o,
                None => continue,
            };

            // 胖 opcode 自描述：从 Op v2 Spec 读 GEMM 维度
            let (m, n, k) = match op.op_v2_gemm_dims(graph) {
                Some((m_dim, n_val, k_val)) => {
                    // ARCH-SYMDIM-DEGRADE: cost model uses max_for_allocation for conservative estimate.
                    // TODO(G-2): preserve symbolic form for tighter bounds.
                    let m_val = m_dim.max_for_allocation_strict()
                        .expect("ARCH-SYMDIM: SymDim must have max_value in pain_point analysis");
                    (m_val, n_val, k_val)
                }
                None => continue,
            };

            let role = classify_gemm_role(op_id, graph, &all_gemms_in_order);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            // PERF: bytes 用 F32 (4 字节) 作为保守估算上界,非统一精度假设
            // (实际 dtype 从 op inputs 推导,BF16/F16 buffer 更小,F32 是上界)
            let bytes = (m * k + k * n + m * n) as f64 * 4.0;
            let ai = if bytes > 0.0 { flops / bytes } else { 0.0 };

            let bottleneck = if m <= 1 && n * k <= 256 {
                BottleneckType::LatencyBound { estimated_latency_ns: 100.0 }
            } else if ai < ridge {
                BottleneckType::MemoryBound {
                    bandwidth_utilization: if ridge > 0.0 { ai / ridge } else { 1.0 },
                }
            } else {
                BottleneckType::ComputeBound {
                    compute_utilization: if ai > 0.0 { ridge / ai } else { 1.0 },
                }
            };

            let optimal_fusion = pick_strategy(role, &bottleneck, m, device);
            let fusion_benefits = compute_fusion_benefits(role, &bottleneck, m, n, k);
            let exec_pattern = derive_exec_pattern(m, n, k, &bottleneck, device);
            let parallelism = derive_parallelism(device, m, n, k);

            gemm_bottlenecks.insert(op_id, GemmBottleneck {
                gemm_role: role,
                shape: (m, n, k),
                arithmetic_intensity: ai,
                ridge_point: ridge,
                bottleneck,
                optimal_fusion,
                fusion_benefits,
                exec_pattern,
                parallelism,
            });
        }

        OpBottleneckMap {
            gemm_bottlenecks,
            ridge_point: ridge,
        }
    }
}

/// 从图拓扑推导 GEMM 语义角色。
/// 优先使用结构特征（共享输入 = QKV，紧跟 attention 输出 = Output），
/// label 匹配作为 fallback。
///
/// ARCH-JIT-DATA-YIELDS: 全部基于 tensor.consumers / tensor.producer 索引和预计算的
/// `all_gemms_in_order` 列表，零全图扫描。
fn classify_gemm_role(
    op_id: OpId,
    graph: &CompilerGraph,
    all_gemms_in_order: &[OpId],
) -> GemmRole {
    let op = match graph.op(op_id) {
        Some(o) => o,
        None => return GemmRole::Other,
    };

    let is_gemm_kind = |kind: &OpKind| matches!(kind,
        OpKind::Gemm { .. } | OpKind::GemmBias { .. } | OpKind::QuantGemm { .. });

    // 拓扑推导 1: 共享输入的 GEMM 三兄弟 → QkvProjection
    // ARCH-JIT-DATA-YIELDS: 使用 tensor.consumers 索引替代全图扫描。
    if let Some(&input_tid) = op.inputs.first() {
        let shared_count = graph.tensor(input_tid)
            .map(|t| t.consumers.iter()
                .filter(|&&c| graph.op(c).is_some_and(|o| is_gemm_kind(&o.kind)))
                .count())
            .unwrap_or(0);
        if shared_count >= 3 {
            return GemmRole::QkvProjection;
        }
    }

    // 拓扑推导 2: 此 GEMM 的输出被 softmax 或 attention 算子消费 → OutputProjection
    for &output_tid in &op.outputs {
        let consumers = graph.tensor(output_tid).map(|t| &t.consumers).cloned().unwrap_or_default();
        for consumer_id in &consumers {
            if let Some(consumer) = graph.op(*consumer_id) {
                if matches!(consumer.kind,
                    OpKind::Softmax | OpKind::MultiHeadAttention { .. }
                ) {
                    return GemmRole::OutputProjection;
                }
            }
        }
    }

    // 拓扑推导 3: 共享输入的 GEMM 两兄弟 + 后接 SiLU/SwiGLU → GateUpProjection
    if let Some(&input_tid) = op.inputs.first() {
        let sibling_gemm_count = graph.tensor(input_tid)
            .map(|t| t.consumers.iter()
                .filter(|&&c| c != op_id && graph.op(c).is_some_and(|o| is_gemm_kind(&o.kind)))
                .count())
            .unwrap_or(0);
        if sibling_gemm_count == 1 {
            // 检查输出是否被 SwiGLU 或 SiLU 消费
            for &output_tid in &op.outputs {
                let consumers = graph.tensor(output_tid).map(|t| &t.consumers).cloned().unwrap_or_default();
                for consumer_id in &consumers {
                    if let Some(consumer) = graph.op(*consumer_id) {
                        if matches!(consumer.kind,
                            OpKind::SwiGlu | OpKind::Silu | OpKind::SwiGluClipped { .. } | OpKind::GeGlu
                        ) {
                            return GemmRole::GateUpProjection;
                        }
                    }
                }
            }
        }
    }

    // 拓扑推导 4: 输入是 SwiGLU/SiLU 输出的 GEMM → DownProjection
    if let Some(&input_tid) = op.inputs.first() {
        if let Some(input_tensor) = graph.tensor(input_tid) {
            if let Some(producer_id) = input_tensor.producer {
                if let Some(producer) = graph.op(producer_id) {
                    if matches!(producer.kind,
                        OpKind::SwiGlu | OpKind::Silu | OpKind::SwiGluClipped { .. } | OpKind::GeGlu
                    ) {
                        return GemmRole::DownProjection;
                    }
                }
            }
        }
    }

    // 拓扑推导 5: 图中最后一个 GEMM + 无后续 GEMM → LmHead
    // ARCH-JIT-DATA-YIELDS: 使用预计算的 all_gemms_in_order 列表，零全图扫描。
    if all_gemms_in_order.len() >= 3 {
        if all_gemms_in_order.last() == Some(&op_id) {
            return GemmRole::LmHead;
        }
    }

    GemmRole::Other
}

/// 选择最优融合策略
fn pick_strategy(role: GemmRole, bottleneck: &BottleneckType, m: usize, device: &DeviceProfile) -> FusionPriority {
    let (mc, _, _) = device.gemm_tile_sizes();
    let tile_threshold = mc.max(2);
    match role {
        GemmRole::LmHead => FusionPriority::EpilogueInjection,
        GemmRole::QkvProjection => FusionPriority::QkvSharedInput,
        GemmRole::GateUpProjection => FusionPriority::FfnBlock,
        GemmRole::OutputProjection | GemmRole::DownProjection => {
            match bottleneck {
                BottleneckType::MemoryBound { .. } => FusionPriority::EpilogueInjection,
                BottleneckType::ComputeBound { .. } => {
                    if m > tile_threshold {
                        FusionPriority::TileLevelFusion
                    } else {
                        FusionPriority::EpilogueInjection
                    }
                }
                BottleneckType::LatencyBound { .. } => FusionPriority::EpilogueInjection,
            }
        }
        GemmRole::Other => match bottleneck {
            BottleneckType::MemoryBound { .. } => FusionPriority::EpilogueInjection,
            _ => FusionPriority::LoopFusion,
        },
    }
}

/// 计算每种融合策略的收益倍率
fn compute_fusion_benefits(
    role: GemmRole,
    bottleneck: &BottleneckType,
    m: usize, n: usize, k: usize,
) -> HashMap<FusionPriority, f64> {
    let mut benefits = HashMap::new();
    // PERF: F32 (4 字节) 作为保守估算上界,非统一精度假设
    let output_bytes = (m * n) as f64 * 4.0;
    let input_bytes = (m * k) as f64 * 4.0;

    let scale = match bottleneck {
        BottleneckType::MemoryBound { .. } => 1.0,
        BottleneckType::ComputeBound { compute_utilization } => compute_utilization.max(0.1),
        BottleneckType::LatencyBound { .. } => 0.5,
    };

    // P0: EpilogueInjection — 消除 output 写回
    benefits.insert(FusionPriority::EpilogueInjection, output_bytes * scale);

    // P1: NormIntoGemm — 消除 norm output 写回 + 重读
    benefits.insert(FusionPriority::NormIntoGemm, input_bytes * scale * 0.5);

    // P1: TileLevelFusion — 仅 M > mc_min 有效
    let tile_benefit = if m > 16 { input_bytes * scale * 0.8 } else { 0.0 };
    benefits.insert(FusionPriority::TileLevelFusion, tile_benefit);

    // P1: ComputeRoot — norm output 驻留 L1/L2
    benefits.insert(FusionPriority::ComputeRoot, input_bytes * scale * 0.3);

    // P2: QkvSharedInput — 消除 2× pack_a
    let qkv_benefit = match role {
        GemmRole::QkvProjection => input_bytes * 2.0 * scale,
        _ => 0.0,
    };
    benefits.insert(FusionPriority::QkvSharedInput, qkv_benefit);

    // P2: FFNBlock — Gate+Up 共享 pack_a + activation 融合
    let ffn_benefit = match role {
        GemmRole::GateUpProjection => (output_bytes + input_bytes) * scale,
        _ => 0.0,
    };
    benefits.insert(FusionPriority::FfnBlock, ffn_benefit);

    // P3: LoopFusion — elementwise chain
    benefits.insert(FusionPriority::LoopFusion, output_bytes * scale * 0.2);

    benefits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, KvSource, SymDim};
    use crate::types::DType;
    use crate::dispatch::device_profile::DeviceProfile;

    #[test]
    fn test_pain_point_decode_memory_bound() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[1, 4096], DType::F32);
        let w_gate = g.add_tensor_concrete("w_gate", &[4096, 11008], DType::F32);
        let w_up = g.add_tensor_concrete("w_up", &[4096, 11008], DType::F32);
        let gate_out = g.add_tensor_concrete("gate_out", &[1, 11008], DType::F32);
        let up_out = g.add_tensor_concrete("up_out", &[1, 11008], DType::F32);
        let swiglu_out = g.add_tensor_concrete("swiglu_out", &[1, 11008], DType::F32);
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 11008, k: 4096,
            dtype: DType::F32, trans_b: false }, vec![inp, w_gate], vec![gate_out], "gate_proj");
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 11008, k: 4096,
            dtype: DType::F32, trans_b: false }, vec![inp, w_up], vec![up_out], "up_proj");
        g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        assert_eq!(map.gemm_bottlenecks.len(), 2);
        let bn = map.gemm_bottlenecks.values().next().unwrap();
        assert_eq!(bn.gemm_role, GemmRole::GateUpProjection);
        assert_eq!(bn.optimal_fusion, FusionPriority::FfnBlock);
        assert!(bn.arithmetic_intensity < bn.ridge_point);
        assert!(matches!(bn.bottleneck, BottleneckType::MemoryBound { .. }));
    }

    #[test]
    fn test_pain_point_prefill_compute_bound() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[512, 4096], DType::F32);
        let w = g.add_tensor_concrete("weight", &[4096, 4096], DType::F32);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[512, 4096], DType::F32);
        let soft_out = g.add_tensor_concrete("soft_out", &[512, 4096], DType::F32);
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(512), n: 4096, k: 4096,
            dtype: DType::F32, trans_b: false }, vec![inp, w], vec![gemm_out], "o_proj");
        g.add_op(OpKind::Softmax, vec![gemm_out], vec![soft_out], "attn_softmax");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        let bn = map.gemm_bottlenecks.values().next().unwrap();
        assert_eq!(bn.gemm_role, GemmRole::OutputProjection);
        assert!(bn.arithmetic_intensity >= 1.0);
    }

    #[test]
    fn test_classify_gemm_role_topology() {
        // QKV: 3 GEMM 共享同一输入
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 4096], DType::F32);
        let wq = g.add_tensor_concrete("wq", &[4096, 4096], DType::F32);
        let wk = g.add_tensor_concrete("wk", &[4096, 4096], DType::F32);
        let wv = g.add_tensor_concrete("wv", &[4096, 4096], DType::F32);
        let oq = g.add_tensor_concrete("oq", &[1, 4096], DType::F32);
        let ok_ = g.add_tensor_concrete("ok", &[1, 4096], DType::F32);
        let ov = g.add_tensor_concrete("ov", &[1, 4096], DType::F32);
        let q_op = g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![inp, wq], vec![oq], "q");
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![inp, wk], vec![ok_], "k");
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![inp, wv], vec![ov], "v");
        assert_eq!(classify_gemm_role(q_op, &g, &[q_op]), GemmRole::QkvProjection);

        // OutputProjection: GEMM 输出被 Softmax 消费
        let mut g2 = CompilerGraph::new();
        let inp2 = g2.add_tensor_concrete("x", &[1, 4096], DType::F32);
        let w2 = g2.add_tensor_concrete("w", &[4096, 4096], DType::F32);
        let gemm_out2 = g2.add_tensor_concrete("go", &[1, 4096], DType::F32);
        let soft_out = g2.add_tensor_concrete("so", &[1, 4096], DType::F32);
        let o_proj = g2.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![inp2, w2], vec![gemm_out2], "o");
        g2.add_op(OpKind::Softmax, vec![gemm_out2], vec![soft_out], "soft");
        assert_eq!(classify_gemm_role(o_proj, &g2, &[o_proj]), GemmRole::OutputProjection);

        // 孤立 GEMM: Other
        let mut g3 = CompilerGraph::new();
        let inp3 = g3.add_tensor_concrete("x", &[1, 4096], DType::F32);
        let w3 = g3.add_tensor_concrete("w", &[4096, 4096], DType::F32);
        let out3 = g3.add_tensor_concrete("o", &[1, 4096], DType::F32);
        let isolated = g3.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![inp3, w3], vec![out3], "generic");
        assert_eq!(classify_gemm_role(isolated, &g3, &[isolated]), GemmRole::Other);
    }

    #[test]
    fn test_fusion_benefits_qkv() {
        let benefits = compute_fusion_benefits(
            GemmRole::QkvProjection,
            &BottleneckType::MemoryBound { bandwidth_utilization: 0.5 },
            1, 4096, 4096,
        );
        assert!(benefits[&FusionPriority::QkvSharedInput] > 0.0);
        assert!(benefits[&FusionPriority::FfnBlock] == 0.0);
    }

    #[test]
    fn test_derive_exec_pattern_gemv_decode() {
        // M=1 (decode GEMV) should produce TileGemm with tile_m=1.
        let device = DeviceProfile::detect();
        let pattern = derive_exec_pattern(1, 4096, 4096, &BottleneckType::MemoryBound { bandwidth_utilization: 0.3 }, &device);
        match pattern {
            ExecPattern::TileGemm { tile_m, .. } => assert_eq!(tile_m, 1),
            ExecPattern::ScalarLoop => {} // tiny matrix edge case acceptable
            other => panic!("expected TileGemm or ScalarLoop for GEMV, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_exec_pattern_latency_bound_tiny() {
        // Very tiny matrix (M<=1, N*K<=256) should be LatencyBound → ScalarLoop.
        let device = DeviceProfile::detect();
        let pattern = derive_exec_pattern(1, 8, 16, &BottleneckType::LatencyBound { estimated_latency_ns: 50.0 }, &device);
        assert_eq!(pattern, ExecPattern::ScalarLoop);
    }

    #[test]
    fn test_derive_exec_pattern_memory_bound_shared_mem_tile() {
        let device = DeviceProfile::detect();
        let pattern = derive_exec_pattern(64, 256, 256, &BottleneckType::MemoryBound { bandwidth_utilization: 0.2 }, &device);
        assert!(matches!(pattern, ExecPattern::SharedMemTile { .. }), "memory-bound mid-size should be SharedMemTile, got {:?}", pattern);
    }

    #[test]
    fn test_derive_exec_pattern_compute_bound_tile_gemm() {
        let device = DeviceProfile::detect();
        let pattern = derive_exec_pattern(512, 1024, 1024, &BottleneckType::ComputeBound { compute_utilization: 0.8 }, &device);
        assert!(matches!(pattern, ExecPattern::TileGemm { .. }), "compute-bound large should be TileGemm, got {:?}", pattern);
    }

    #[test]
    fn test_derive_parallelism_decode_unroll_one() {
        // Decode (M=1): unroll_factor should be 1.
        let device = DeviceProfile::detect();
        let par = derive_parallelism(&device, 1, 4096, 4096);
        match par {
            ParallelismDesc::SimdVectorize { unroll_factor, .. } => assert_eq!(unroll_factor, 1),
            other => panic!("expected SimdVectorize, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_parallelism_prefill_unroll() {
        // Prefill (M>1): unroll_factor should be >= 1.
        let device = DeviceProfile::detect();
        let par = derive_parallelism(&device, 512, 4096, 4096);
        match par {
            ParallelismDesc::SimdVectorize { unroll_factor, element_width } => {
                assert!(unroll_factor >= 1);
                assert!(element_width >= 1);
            }
            other => panic!("expected SimdVectorize, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_thread_parallelism_structure() {
        let device = DeviceProfile::detect();
        let par = derive_thread_parallelism(&device);
        match par {
            ParallelismDesc::ThreadParallel { parallel_dim, granularity } => {
                assert_eq!(parallel_dim, device.physical_cores);
                assert_eq!(granularity, 64);
            }
            other => panic!("expected ThreadParallel, got {:?}", other),
        }
    }

    #[test]
    fn test_compute_fusion_benefits_gate_up() {
        let benefits = compute_fusion_benefits(
            GemmRole::GateUpProjection,
            &BottleneckType::MemoryBound { bandwidth_utilization: 1.0 },
            1, 11008, 4096,
        );
        assert!(benefits[&FusionPriority::FfnBlock] > 0.0);
        assert!(benefits[&FusionPriority::QkvSharedInput] == 0.0);
        assert!(benefits[&FusionPriority::EpilogueInjection] > 0.0);
    }

    #[test]
    fn test_compute_fusion_benefits_compute_bound_scales_down() {
        let mem_benefits = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::MemoryBound { bandwidth_utilization: 1.0 },
            64, 256, 256,
        );
        let comp_benefits = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::ComputeBound { compute_utilization: 0.2 },
            64, 256, 256,
        );
        // Compute-bound should scale benefits down by compute_utilization.
        assert!(comp_benefits[&FusionPriority::EpilogueInjection] < mem_benefits[&FusionPriority::EpilogueInjection]);
    }

    #[test]
    fn test_compute_fusion_benefits_tile_level_requires_large_m() {
        let small = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::MemoryBound { bandwidth_utilization: 1.0 },
            4, 256, 256,
        );
        let large = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::MemoryBound { bandwidth_utilization: 1.0 },
            128, 256, 256,
        );
        // TileLevelFusion benefit is 0 for small M, positive for large M.
        assert_eq!(small[&FusionPriority::TileLevelFusion], 0.0);
        assert!(large[&FusionPriority::TileLevelFusion] > 0.0);
    }

    #[test]
    fn test_pick_strategy_lm_head_always_epilogue() {
        let device = DeviceProfile::detect();
        let strat = pick_strategy(GemmRole::LmHead, &BottleneckType::MemoryBound { bandwidth_utilization: 0.5 }, 1, &device);
        assert_eq!(strat, FusionPriority::EpilogueInjection);

        let strat2 = pick_strategy(GemmRole::LmHead, &BottleneckType::ComputeBound { compute_utilization: 0.9 }, 512, &device);
        assert_eq!(strat2, FusionPriority::EpilogueInjection);
    }

    #[test]
    fn test_classify_gemm_role_down_projection_after_silu() {
        // DownProjection: GEMM input is produced by SiLU.
        let mut g = CompilerGraph::new();
        let silu_in = g.add_tensor_concrete("silu_in", &[1, 4096], DType::F32);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4096], DType::F32);
        g.add_op(OpKind::Silu, vec![silu_in], vec![silu_out], "silu");

        let w_down = g.add_tensor_concrete("w_down", &[4096, 4096], DType::F32);
        let down_out = g.add_tensor_concrete("down_out", &[1, 4096], DType::F32);
        let down_op = g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 4096, k: 4096,
            dtype: DType::F32, trans_b: false,
        }, vec![silu_out, w_down], vec![down_out], "down_proj");

        assert_eq!(classify_gemm_role(down_op, &g, &[down_op]), GemmRole::DownProjection);
    }

    #[test]
    fn test_classify_gemm_role_lm_head_last_gemm() {
        // LmHead: last GEMM in a graph with >= 3 GEMMs, each with distinct inputs
        // (no shared-input topology that would trigger QKV classification).
        let mut g = CompilerGraph::new();
        let mut last_id = None;
        let mut gemm_ids_in_order: Vec<OpId> = Vec::new();
        for i in 0..3 {
            let inp = g.add_tensor_concrete(&format!("x{}", i), &[1, 256], DType::F32);
            let w = g.add_tensor_concrete(&format!("w{}", i), &[256, 256], DType::F32);
            let out = g.add_tensor_concrete(&format!("o{}", i), &[1, 256], DType::F32);
            let op_id = g.add_op(OpKind::Gemm {
                m: SymDim::Concrete(1), n: 256, k: 256,
                dtype: DType::F32, trans_b: false,
            }, vec![inp, w], vec![out], &format!("gemm{}", i));
            last_id = Some(op_id);
            gemm_ids_in_order.push(op_id);
        }
        assert_eq!(classify_gemm_role(last_id.unwrap(), &g, &gemm_ids_in_order), GemmRole::LmHead);
    }

    #[test]
    fn test_ridge_point_computation() {
        // Verify ridge point = peak_gflops / peak_bandwidth.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 256], DType::F32);
        let out = g.add_tensor_concrete("o", &[1, 256], DType::F32);
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 256, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w], vec![out], "gemm");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        let expected_ridge = if device.peak_bandwidth_gbs > 0.0 {
            device.peak_gflops_f32 * 1e9 / (device.peak_bandwidth_gbs * 1e9)
        } else {
            10.0
        };
        assert!((map.ridge_point - expected_ridge).abs() < 1e-6);
    }

    // ── 10 new tests covering uncovered logic paths ───────────────────

    #[test]
    fn test_analyze_gemmbias_variant() {
        // Arrange: GemmBias ops should be analyzed just like Gemm ops.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 256], DType::F32);
        let out = g.add_tensor_concrete("o", &[1, 256], DType::F32);
        g.add_op(OpKind::GemmBias {
            m: SymDim::Concrete(1), n: 256, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w], vec![out], "gemm_bias");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        // Assert: the GemmBias is picked up as a single GEMM bottleneck.
        assert_eq!(map.gemm_bottlenecks.len(), 1);
        let bn = map.gemm_bottlenecks.values().next().unwrap();
        assert_eq!(bn.shape, (1, 256, 256));
    }

    #[test]
    fn test_analyze_quantgemm_variant() {
        // Arrange: QuantGemm ops should be analyzed and classified.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 256], DType::F32);
        let out = g.add_tensor_concrete("o", &[1, 256], DType::F32);
        g.add_op(OpKind::QuantGemm {
            m: SymDim::Concrete(1), n: 256, k: 256,
            quant_type: crate::quant::QuantType::Q4K,
        }, vec![inp, w], vec![out], "qgemm");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        // Assert: QuantGemm is treated as a GEMM and analyzed.
        assert_eq!(map.gemm_bottlenecks.len(), 1);
        let bn = map.gemm_bottlenecks.values().next().unwrap();
        assert_eq!(bn.shape.1, 256);
        assert_eq!(bn.shape.2, 256);
    }

    #[test]
    fn test_analyze_symbolic_m_with_max_value() {
        // Arrange: SymDim::Symbolic with max_value should use that max for M.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[512, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 256], DType::F32);
        let out = g.add_tensor_concrete("o", &[512, 256], DType::F32);
        g.add_op(OpKind::Gemm {
            m: SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(1024) },
            n: 256, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w], vec![out], "sym_gemm");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        // Assert: M uses max_value=1024 for the analysis.
        let bn = map.gemm_bottlenecks.values().next().unwrap();
        assert_eq!(bn.shape.0, 1024);
    }

    #[test]
    fn test_classify_gate_up_projection_via_geglu() {
        // Arrange: Two GEMMs sharing input + GeGlu consumer → GateUpProjection.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 512], DType::F32);
        let w_gate = g.add_tensor_concrete("w_gate", &[512, 1024], DType::F32);
        let w_up = g.add_tensor_concrete("w_up", &[512, 1024], DType::F32);
        let gate_out = g.add_tensor_concrete("gate_out", &[1, 1024], DType::F32);
        let up_out = g.add_tensor_concrete("up_out", &[1, 1024], DType::F32);
        let geglu_out = g.add_tensor_concrete("geglu_out", &[1, 1024], DType::F32);

        let gate_op = g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 1024, k: 512,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w_gate], vec![gate_out], "gate");
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 1024, k: 512,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w_up], vec![up_out], "up");
        g.add_op(OpKind::GeGlu, vec![gate_out, up_out], vec![geglu_out], "geglu");

        // Act
        let role = classify_gemm_role(gate_op, &g, &[gate_op]);

        // Assert: GateUpProjection via GeGlu topology.
        assert_eq!(role, GemmRole::GateUpProjection);
    }

    #[test]
    fn test_classify_gate_up_projection_via_swiglu_clipped() {
        // Arrange: Two GEMMs sharing input + SwiGluClipped consumer → GateUpProjection.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 256], DType::F32);
        let w_gate = g.add_tensor_concrete("w_gate", &[256, 512], DType::F32);
        let w_up = g.add_tensor_concrete("w_up", &[256, 512], DType::F32);
        let gate_out = g.add_tensor_concrete("gate_out", &[1, 512], DType::F32);
        let up_out = g.add_tensor_concrete("up_out", &[1, 512], DType::F32);
        let clipped_out = g.add_tensor_concrete("clipped_out", &[1, 512], DType::F32);

        let gate_op = g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w_gate], vec![gate_out], "gate");
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w_up], vec![up_out], "up");
        g.add_op(OpKind::SwiGluClipped { limit: 7.0 },
            vec![gate_out, up_out], vec![clipped_out], "clipped");

        // Act
        let role = classify_gemm_role(gate_op, &g, &[gate_op]);

        // Assert: GateUpProjection via SwiGluClipped topology.
        assert_eq!(role, GemmRole::GateUpProjection);
    }

    #[test]
    fn test_classify_output_projection_via_mha() {
        // Arrange: GEMM output consumed by MultiHeadAttention → OutputProjection.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[1, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 256], DType::F32);
        let gemm_out = g.add_tensor_concrete("go", &[1, 256], DType::F32);
        let mha_out = g.add_tensor_concrete("mha_o", &[1, 256], DType::F32);

        let o_proj = g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(1), n: 256, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w], vec![gemm_out], "o_proj");
        g.add_op(OpKind::MultiHeadAttention {
            seq_len: SymDim::Concrete(1),
            num_heads: 4, num_kv_heads: 2, head_dim: 64,
            causal: true, attention_sinks: false,
            kv_source: KvSource::FromTensor,
        }, vec![gemm_out], vec![mha_out], "mha");

        // Act
        let role = classify_gemm_role(o_proj, &g, &[o_proj]);

        // Assert: OutputProjection via MHA consumer.
        assert_eq!(role, GemmRole::OutputProjection);
    }

    #[test]
    fn test_pick_strategy_other_compute_bound_loop_fusion() {
        // Arrange: GemmRole::Other + ComputeBound → LoopFusion.
        let device = DeviceProfile::detect();

        // Act
        let strat = pick_strategy(
            GemmRole::Other,
            &BottleneckType::ComputeBound { compute_utilization: 0.7 },
            64,
            &device,
        );

        // Assert
        assert_eq!(strat, FusionPriority::LoopFusion);
    }

    #[test]
    fn test_pick_strategy_output_projection_compute_large_m_tile_fusion() {
        // Arrange: OutputProjection + ComputeBound with M > tile_threshold → TileLevelFusion.
        let device = DeviceProfile::detect();
        let (mc, _, _) = device.gemm_tile_sizes();
        let large_m = mc.max(2) + 100; // definitely > tile_threshold

        // Act
        let strat = pick_strategy(
            GemmRole::OutputProjection,
            &BottleneckType::ComputeBound { compute_utilization: 0.9 },
            large_m,
            &device,
        );

        // Assert
        assert_eq!(strat, FusionPriority::TileLevelFusion);
    }

    #[test]
    fn test_pick_strategy_down_projection_latency_bound() {
        // Arrange: DownProjection + LatencyBound → EpilogueInjection.
        let device = DeviceProfile::detect();

        // Act
        let strat = pick_strategy(
            GemmRole::DownProjection,
            &BottleneckType::LatencyBound { estimated_latency_ns: 200.0 },
            1,
            &device,
        );

        // Assert
        assert_eq!(strat, FusionPriority::EpilogueInjection);
    }

    #[test]
    fn test_compute_fusion_benefits_latency_bound_halved() {
        // Arrange: LatencyBound should halve the scale factor to 0.5.
        let benefits = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::LatencyBound { estimated_latency_ns: 100.0 },
            32, 128, 128,
        );

        // Act: compute expected EpilogueInjection benefit = output_bytes * 0.5
        let output_bytes = (32usize * 128) as f64 * 4.0;

        // Assert: LatencyBound scale=0.5, so EpilogueInjection = output_bytes * 0.5
        let expected = output_bytes * 0.5;
        let actual = benefits[&FusionPriority::EpilogueInjection];
        assert!((actual - expected).abs() < 1e-6,
            "expected EpilogueInjection benefit {}, got {}", expected, actual);

        // Also verify QkvSharedInput and FfnBlock are 0 for Other role.
        assert_eq!(benefits[&FusionPriority::QkvSharedInput], 0.0);
        assert_eq!(benefits[&FusionPriority::FfnBlock], 0.0);
    }

    // ── Wave 12kmf: 10 additional tests ────────────────────────────────

    #[test]
    fn test_derive_exec_pattern_gemv_latency_returns_scalar_loop() {
        // Arrange: M=1 with N*K > 256 (not the tiny shortcut), but LatencyBound.
        let device = DeviceProfile::detect();
        let bottleneck = BottleneckType::LatencyBound { estimated_latency_ns: 80.0 };

        // Act
        let pattern = derive_exec_pattern(1, 512, 512, &bottleneck, &device);

        // Assert: GEMV + LatencyBound branch returns ScalarLoop.
        assert_eq!(pattern, ExecPattern::ScalarLoop);
    }

    #[test]
    fn test_derive_exec_pattern_narrow_tall_shrinks_tile_m() {
        // Arrange: M smaller than tm_base — tile_m should shrink to M.
        let device = DeviceProfile::detect();
        let (tm_base, _, _) = device.gemm_tile_sizes();
        let m = if tm_base > 2 { tm_base / 2 } else { 1 };
        let bottleneck = BottleneckType::ComputeBound { compute_utilization: 0.9 };

        // Act
        let pattern = derive_exec_pattern(m, 1024, 1024, &bottleneck, &device);

        // Assert: tile_m == m (shrunk from tm_base).
        match pattern {
            ExecPattern::TileGemm { tile_m, .. } => assert_eq!(tile_m, m),
            other => panic!("expected TileGemm for compute-bound, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_exec_pattern_wide_short_shrinks_tile_n() {
        // Arrange: N smaller than tn_base — tile_n should shrink to N.
        let device = DeviceProfile::detect();
        let (_, tn_base, _) = device.gemm_tile_sizes();
        let n = if tn_base > 4 { tn_base / 2 } else { 2 };
        let bottleneck = BottleneckType::ComputeBound { compute_utilization: 0.8 };

        // Act
        let pattern = derive_exec_pattern(256, n, 1024, &bottleneck, &device);

        // Assert: tile_n == n (shrunk from tn_base).
        match pattern {
            ExecPattern::TileGemm { tile_n, .. } => assert_eq!(tile_n, n),
            other => panic!("expected TileGemm, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_exec_pattern_large_k_shrinks_tile_k() {
        // Arrange: K > 4 * tk_base — tile_k should halve.
        let device = DeviceProfile::detect();
        let (_, _, tk_base) = device.gemm_tile_sizes();
        let k = tk_base * 5 + 10; // definitely > 4 * tk_base
        let expected_tk = (tk_base / 2).max(1);
        let bottleneck = BottleneckType::ComputeBound { compute_utilization: 0.9 };

        // Act
        let pattern = derive_exec_pattern(256, 256, k, &bottleneck, &device);

        // Assert: tile_k == expected_tk (halved).
        match pattern {
            ExecPattern::TileGemm { tile_k, .. } => assert_eq!(tile_k, expected_tk),
            other => panic!("expected TileGemm, got {:?}", other),
        }
    }

    #[test]
    fn test_compute_fusion_benefits_compute_bound_clamps_utilization() {
        // Arrange: ComputeBound with very low utilization should clamp to 0.1.
        let benefits = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::ComputeBound { compute_utilization: 0.01 },
            64, 256, 256,
        );
        let output_bytes = (64usize * 256) as f64 * 4.0;

        // Assert: scale clamped to 0.1, so EpilogueInjection = output_bytes * 0.1.
        let expected = output_bytes * 0.1;
        let actual = benefits[&FusionPriority::EpilogueInjection];
        assert!((actual - expected).abs() < 1e-6,
            "expected clamped benefit {}, got {}", expected, actual);
    }

    #[test]
    fn test_derive_parallelism_m0_treated_as_gemv() {
        // Arrange: M=0 should be treated like decode (unroll_factor=1).
        let device = DeviceProfile::detect();

        // Act
        let par = derive_parallelism(&device, 0, 4096, 4096);

        // Assert: unroll_factor == 1 (same as M=1 decode path).
        match par {
            ParallelismDesc::SimdVectorize { unroll_factor, element_width } => {
                assert_eq!(unroll_factor, 1);
                assert!(element_width >= 1);
            }
            other => panic!("expected SimdVectorize, got {:?}", other),
        }
    }

    #[test]
    fn test_pick_strategy_output_projection_memory_bound_epilogue() {
        // Arrange: OutputProjection + MemoryBound → EpilogueInjection.
        let device = DeviceProfile::detect();

        // Act
        let strat = pick_strategy(
            GemmRole::OutputProjection,
            &BottleneckType::MemoryBound { bandwidth_utilization: 0.4 },
            1,
            &device,
        );

        // Assert
        assert_eq!(strat, FusionPriority::EpilogueInjection);
    }

    #[test]
    fn test_pick_strategy_other_memory_bound_epilogue() {
        // Arrange: GemmRole::Other + MemoryBound → EpilogueInjection.
        let device = DeviceProfile::detect();

        // Act
        let strat = pick_strategy(
            GemmRole::Other,
            &BottleneckType::MemoryBound { bandwidth_utilization: 0.6 },
            32,
            &device,
        );

        // Assert
        assert_eq!(strat, FusionPriority::EpilogueInjection);
    }

    #[test]
    fn test_analyze_populates_all_gemm_bottleneck_fields() {
        // Arrange: Build a simple GEMM graph and run full analysis.
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("x", &[64, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 512], DType::F32);
        let out = g.add_tensor_concrete("o", &[64, 512], DType::F32);
        g.add_op(OpKind::Gemm {
            m: SymDim::Concrete(64), n: 512, k: 256,
            dtype: DType::F32, trans_b: false,
        }, vec![inp, w], vec![out], "gemm");

        let device = DeviceProfile::detect();
        let map = PainPointAnalyzer::analyze(&g, &device);

        // Assert: verify all fields are populated correctly.
        assert_eq!(map.gemm_bottlenecks.len(), 1);
        let bn = map.gemm_bottlenecks.values().next().unwrap();

        // Shape matches input.
        assert_eq!(bn.shape, (64, 512, 256));

        // Arithmetic intensity is non-negative.
        assert!(bn.arithmetic_intensity >= 0.0);

        // Ridge point matches map-level ridge.
        assert!((bn.ridge_point - map.ridge_point).abs() < 1e-10);

        // Fusion benefits has entries for all 7 priorities.
        assert_eq!(bn.fusion_benefits.len(), 7);
        for priority in [
            FusionPriority::EpilogueInjection,
            FusionPriority::NormIntoGemm,
            FusionPriority::TileLevelFusion,
            FusionPriority::ComputeRoot,
            FusionPriority::QkvSharedInput,
            FusionPriority::FfnBlock,
            FusionPriority::LoopFusion,
        ] {
            assert!(bn.fusion_benefits.contains_key(&priority),
                "missing benefit for {:?}", priority);
        }

        // ExecPattern and parallelism are populated (not default-zero).
        assert!(matches!(bn.exec_pattern,
            ExecPattern::TileGemm { .. } | ExecPattern::SharedMemTile { .. }));
        assert!(matches!(bn.parallelism,
            ParallelismDesc::SimdVectorize { .. }));
    }

    #[test]
    fn test_compute_fusion_benefits_down_projection_compute_root_positive() {
        // Arrange: DownProjection role — ComputeRoot should always be positive.
        let benefits = compute_fusion_benefits(
            GemmRole::DownProjection,
            &BottleneckType::MemoryBound { bandwidth_utilization: 1.0 },
            64, 2048, 4096,
        );

        // Assert: ComputeRoot is input_bytes * 1.0 * 0.3 > 0.
        let input_bytes = (64usize * 4096) as f64 * 4.0;
        let expected_cr = input_bytes * 0.3;
        assert!((benefits[&FusionPriority::ComputeRoot] - expected_cr).abs() < 1e-6);

        // Assert: QkvSharedInput and FfnBlock are zero for non-matching roles.
        assert_eq!(benefits[&FusionPriority::QkvSharedInput], 0.0);
        assert_eq!(benefits[&FusionPriority::FfnBlock], 0.0);

        // Assert: NormIntoGemm is also positive.
        assert!(benefits[&FusionPriority::NormIntoGemm] > 0.0);
    }

    // ── Wave 12x60: 10 additional tests ────────────────────────────────

    #[test]
    fn test_bottleneck_memory_bound_propagates_utilization() {
        // Arrange: Create a MemoryBound bottleneck with a specific utilization.
        let bn = BottleneckType::MemoryBound { bandwidth_utilization: 0.37 };

        // Act & Assert: Verify the value is correctly stored and retrievable.
        match bn {
            BottleneckType::MemoryBound { bandwidth_utilization } => {
                assert!((bandwidth_utilization - 0.37).abs() < 1e-10);
            }
            other => panic!("expected MemoryBound, got {:?}", other),
        }
    }

    #[test]
    fn test_bottleneck_compute_bound_propagates_utilization() {
        // Arrange: Create a ComputeBound bottleneck with a specific utilization.
        let bn = BottleneckType::ComputeBound { compute_utilization: 0.83 };

        // Act & Assert: Verify the value is correctly stored and retrievable.
        match bn {
            BottleneckType::ComputeBound { compute_utilization } => {
                assert!((compute_utilization - 0.83).abs() < 1e-10);
            }
            other => panic!("expected ComputeBound, got {:?}", other),
        }
    }

    #[test]
    fn test_exec_pattern_tile_gemm_equality() {
        // Arrange: Two identical TileGemm patterns.
        let a = ExecPattern::TileGemm {
            tile_m: 64, tile_n: 32, tile_k: 128,
            warp_m: 16, warp_n: 8, mma_k: 4,
            pipeline_depth: 2,
        };
        let b = ExecPattern::TileGemm {
            tile_m: 64, tile_n: 32, tile_k: 128,
            warp_m: 16, warp_n: 8, mma_k: 4,
            pipeline_depth: 2,
        };
        let c = ExecPattern::TileGemm {
            tile_m: 64, tile_n: 32, tile_k: 128,
            warp_m: 16, warp_n: 8, mma_k: 4,
            pipeline_depth: 3, // different pipeline_depth
        };

        // Assert: PartialEq derives field-wise comparison.
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_exec_pattern_shared_mem_tile_equality() {
        // Arrange: Two SharedMemTile patterns with same and different fields.
        let a = ExecPattern::SharedMemTile { tile_rows: 32, tile_cols: 64 };
        let b = ExecPattern::SharedMemTile { tile_rows: 32, tile_cols: 64 };
        let c = ExecPattern::SharedMemTile { tile_rows: 16, tile_cols: 64 };

        // Assert: PartialEq works on SharedMemTile fields.
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_parallelism_desc_simd_vectorize_fields() {
        // Arrange: Construct a SimdVectorize with specific parameters.
        let desc = ParallelismDesc::SimdVectorize {
            element_width: 16,
            unroll_factor: 4,
        };

        // Act & Assert: Verify field extraction.
        match desc {
            ParallelismDesc::SimdVectorize { element_width, unroll_factor } => {
                assert_eq!(element_width, 16);
                assert_eq!(unroll_factor, 4);
            }
            other => panic!("expected SimdVectorize, got {:?}", other),
        }
    }

    #[test]
    fn test_fusion_priority_hashmap_key_consistency() {
        // Arrange: Use FusionPriority as HashMap keys to verify Hash + Eq.
        let mut map = HashMap::new();
        let key1 = FusionPriority::EpilogueInjection;
        let key2 = FusionPriority::EpilogueInjection;
        let key3 = FusionPriority::LoopFusion;

        // Act: Insert with key1, read with key2 (same value, different variable).
        map.insert(key1, 42.0);
        let val = map.get(&key2);
        let val3 = map.get(&key3);

        // Assert: Same-value keys map to same entry; different key maps to None.
        assert_eq!(*val.unwrap(), 42.0);
        assert!(val3.is_none());
    }

    #[test]
    fn test_compute_fusion_benefits_always_produces_seven_entries() {
        // Arrange: compute_fusion_benefits for any valid input should produce 7 entries.
        let benefits = compute_fusion_benefits(
            GemmRole::LmHead,
            &BottleneckType::ComputeBound { compute_utilization: 0.5 },
            1, 4096, 4096,
        );

        // Assert: exactly 7 FusionPriority variants.
        assert_eq!(benefits.len(), 7);

        // All keys present.
        for priority in [
            FusionPriority::EpilogueInjection,
            FusionPriority::NormIntoGemm,
            FusionPriority::TileLevelFusion,
            FusionPriority::ComputeRoot,
            FusionPriority::QkvSharedInput,
            FusionPriority::FfnBlock,
            FusionPriority::LoopFusion,
        ] {
            assert!(benefits.contains_key(&priority),
                "missing {:?} in benefits map", priority);
        }
    }

    #[test]
    fn test_compute_fusion_benefits_zero_dimensions() {
        // Arrange: M=0 yields zero output_bytes and zero input_bytes.
        let benefits = compute_fusion_benefits(
            GemmRole::Other,
            &BottleneckType::MemoryBound { bandwidth_utilization: 1.0 },
            0, 256, 256,
        );

        // Assert: All benefits should be zero since M=0 → output_bytes=0, input_bytes=0.
        for (&priority, &val) in &benefits {
            assert_eq!(val, 0.0, "benefit for {:?} should be 0 when M=0, got {}", priority, val);
        }
    }

    #[test]
    fn test_derive_exec_pattern_scalar_loop_for_tiny_matrix() {
        // Arrange: M=1, N*K <= 256 triggers the tiny-matrix shortcut.
        let device = DeviceProfile::detect();

        // Act: M=1, N=8, K=32 → N*K = 256 <= 256.
        let pattern = derive_exec_pattern(1, 8, 32, &BottleneckType::MemoryBound { bandwidth_utilization: 0.5 }, &device);

        // Assert: tiny matrix → ScalarLoop regardless of bottleneck type.
        assert_eq!(pattern, ExecPattern::ScalarLoop);
    }

    #[test]
    fn test_derive_exec_pattern_gemv_non_latency_bottleneck() {
        // Arrange: M=1 (GEMV) with MemoryBound — not LatencyBound and not tiny.
        let device = DeviceProfile::detect();
        let (_, tn_base, tk_base) = device.gemm_tile_sizes();
        let bottleneck = BottleneckType::MemoryBound { bandwidth_utilization: 0.2 };

        // Act
        let pattern = derive_exec_pattern(1, 2048, 4096, &bottleneck, &device);

        // Assert: GEMV + non-LatencyBound → TileGemm with tile_m=1.
        match pattern {
            ExecPattern::TileGemm { tile_m, tile_n, tile_k, .. } => {
                assert_eq!(tile_m, 1, "GEMV tile_m must be 1");
                assert_eq!(tile_n, tn_base, "tile_n should be tn_base");
                assert_eq!(tile_k, tk_base, "tile_k should be tk_base");
            }
            other => panic!("expected TileGemm for GEMV non-latency, got {:?}", other),
        }
    }
}
