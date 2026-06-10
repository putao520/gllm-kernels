//! Graph Topology Analysis — 编译器 = 喂什么编译什么
//!
//! 编译器是"无脑编译器"——只看图里有什么 op 就生成什么代码。
//! 不需要任何 isXXX/hasXXX 先验知识。
//!
//! 设计模式: Strategy Pattern
//! - BUILD 阶段: 模型文件 → 知道是 reranker/embedding/llm → 选择策略 → 裁剪图 + 配置
//! - 编译阶段: 拿到裁剪后的图 + CompileConfig → 无脑生成机器码 → 返回 CALL 指针
//!
//! 裁剪规则: OutputMode 只裁剪尾部 ops，不修改模型大类。
//! - decoder 裁剪为 ClassifyBinary: 裁掉 Argmax+StoreToken+CheckStopCondition
//! - decoder 裁剪为 EncodeToLayer: 裁掉 lm_head 及后续
//! - 编译器看裁剪后的图 ops 推导一切，完全不读 output_modes

use crate::compiler::graph::{CompilerGraph, OpKind, TensorId};
use crate::compiler::mega_kernel_abi::MtpKernelConfig;

/// seq_len 的来源 — 从图 ops 推导
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqLenSource {
    /// seq_len = prompt_len（图无 Argmax: 所有 token 一次处理）
    PromptLen,
    /// seq_len = gen_counter + 1（图有 Argmax: prefill 递增, decode 恒定）
    LoopCounterPlusOne,
}

/// 循环 bound — 从图 ops 推导
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyBound {
    /// 单遍循环（图无 Argmax: Const(1)）
    Const(usize),
    /// 动态循环（图有 Argmax: total_iters = prompt_len - 1 + max_new_tokens）
    DynamicTotalIters,
}

/// 从 CompilerGraph 推导的编译控制流信息
///
/// 编译器只看图 ops，不读任何外部配置。
/// 每个字段都从图中的 OpKind 存在性/参数推导。
#[derive(Debug, Clone)]
pub struct GraphTopologyAnalysis {
    /// 图有 Argmax + StoreToken + CheckStopCondition → 生成循环
    pub has_generate_loop: bool,
    /// Const(1) (图无 Argmax) vs DynamicTotalIters (图有 Argmax)
    pub outer_loop_bound: TopologyBound,
    /// PromptLen (图无 Argmax) vs LoopCounterPlusOne (图有 Argmax)
    pub seq_len_source: SeqLenSource,

    // ── 从图 OpKind 存在性推导的属性 ──

    /// 图有 MHA ops → 加载 kv_cache_ptr
    pub has_mha: bool,

    /// 词表大小 — 从 OpKind::Argmax { vocab_size } 推导。
    /// 图无 Argmax 时为 None。
    pub vocab_size: Option<usize>,

    /// 产出 logits 的 op 索引 — Argmax 的直接前驱 op。
    /// 用于 logits 重定向（替代 lm_head label 搜索）。
    pub logits_producer_op_idx: Option<usize>,

    /// logits 产出的 tensor ID — logits_producer_op_idx 对应 op 的 output[0]。
    pub logits_output_tid: Option<TensorId>,

    /// MTP 配置 — 从 OpKind::MtpDraft { depth, hidden_size, vocab_size } 推导。
    pub mtp_config: Option<MtpKernelConfig>,

    // ── 层循环拓扑属性 ──

    /// 同构层循环的层数
    pub layer_num_layers: Option<usize>,

    /// 层循环 activation alias
    pub layer_activation_alias: Option<(TensorId, TensorId)>,
}

impl GraphTopologyAnalysis {
    /// 从 CompilerGraph 推导拓扑信息。
    pub fn analyze(graph: &CompilerGraph) -> Self {
        let has_argmax = graph.ops.iter().any(|op| matches!(op.kind, OpKind::Argmax { .. }));
        let has_store_token = graph.ops.iter().any(|op| matches!(op.kind, OpKind::StoreToken));
        let has_check_stop = graph.ops.iter().any(|op| matches!(op.kind, OpKind::CheckStopCondition));

        let has_generate_loop = has_argmax && has_store_token && has_check_stop;

        let outer_loop_bound = if has_generate_loop {
            TopologyBound::DynamicTotalIters
        } else {
            TopologyBound::Const(1)
        };

        let seq_len_source = if has_generate_loop {
            SeqLenSource::LoopCounterPlusOne
        } else {
            SeqLenSource::PromptLen
        };

        // has_mha: 图有 MHA ops → 加载 kv_cache_ptr
        let has_mha = graph.ops.iter().any(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. }));

        let vocab_size = graph.ops.iter().find_map(|op| {
            match op.kind {
                OpKind::Argmax { vocab_size } => Some(vocab_size),
                _ => None,
            }
        });

        // logits_producer: Argmax 的直接前驱 op 索引
        let logits_producer_op_idx = if has_argmax {
            let argmax_idx = graph.ops.iter().position(|op| matches!(op.kind, OpKind::Argmax { .. }));
            argmax_idx.and_then(|ai| {
                if let Some(&argmax_input_tid) = graph.ops.get(ai)?.inputs.first() {
                    graph.ops.iter().position(|op| op.outputs.contains(&argmax_input_tid))
                } else {
                    None
                }
            })
        } else {
            None
        };

        let logits_output_tid = logits_producer_op_idx.and_then(|idx| {
            graph.ops.get(idx)?.outputs.first().copied()
        });

        // MTP 配置从 OpKind::MtpDraft 推导
        let mtp_config = graph.ops.iter().find_map(|op| {
            match op.kind {
                OpKind::MtpDraft { depth, hidden_size, vocab_size } =>
                    Some(MtpKernelConfig { depth, hidden_size, vocab_size }),
                _ => None,
            }
        });

        // 层循环拓扑属性
        let layer_num_layers = graph.layer_loop_config.as_ref().map(|cfg| cfg.num_layers);
        let layer_activation_alias = graph.layer_loop_config.as_ref()
            .and_then(|cfg| cfg.activation_alias);

        Self {
            has_generate_loop,
            outer_loop_bound,
            seq_len_source,
            has_mha,
            vocab_size,
            logits_producer_op_idx,
            logits_output_tid,
            mtp_config,
            layer_num_layers,
            layer_activation_alias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, OpKind, SymDim};
    use crate::types::DType;

    fn make_test_graph(ops: Vec<OpKind>) -> CompilerGraph {
        let mut graph = CompilerGraph::default();
        for (i, kind) in ops.into_iter().enumerate() {
            let input_tid = graph.add_tensor_concrete(
                &format!("input_{}", i),
                &[1],
                DType::F32,
            );
            let output_tid = graph.add_tensor_concrete(
                &format!("output_{}", i),
                &[1],
                DType::F32,
            );
            graph.add_op(kind, vec![input_tid], vec![output_tid], &format!("op_{}", i));
        }
        graph
    }

    #[test]
    fn generate_loop_detected_from_ops() {
        let graph = make_test_graph(vec![
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            OpKind::Argmax { vocab_size: 32000 },
            OpKind::StoreToken,
            OpKind::CheckStopCondition,
        ]);
        let topo = GraphTopologyAnalysis::analyze(&graph);

        assert!(topo.has_generate_loop);
        assert_eq!(topo.outer_loop_bound, TopologyBound::DynamicTotalIters);
        assert_eq!(topo.seq_len_source, SeqLenSource::LoopCounterPlusOne);
        assert_eq!(topo.vocab_size, Some(32000));
    }

    #[test]
    fn no_generate_loop_without_argmax() {
        let graph = make_test_graph(vec![
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            OpKind::MeanPool { seq_len: 128, hidden: 768, cls_mode: false },
        ]);
        let topo = GraphTopologyAnalysis::analyze(&graph);

        assert!(!topo.has_generate_loop);
        assert_eq!(topo.outer_loop_bound, TopologyBound::Const(1));
        assert_eq!(topo.seq_len_source, SeqLenSource::PromptLen);
        assert_eq!(topo.vocab_size, None);
    }

    #[test]
    fn partial_argmax_without_store_token() {
        let graph = make_test_graph(vec![
            OpKind::Argmax { vocab_size: 32000 },
        ]);
        let topo = GraphTopologyAnalysis::analyze(&graph);

        assert!(!topo.has_generate_loop);
    }

    #[test]
    fn vocab_size_derived_from_argmax() {
        let graph = make_test_graph(vec![
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            OpKind::Argmax { vocab_size: 152344 },
            OpKind::StoreToken,
            OpKind::CheckStopCondition,
        ]);
        let topo = GraphTopologyAnalysis::analyze(&graph);

        assert_eq!(topo.vocab_size, Some(152344));
    }

    #[test]
    fn logits_producer_derived_from_argmax_predecessor() {
        let mut graph = CompilerGraph::default();
        let gemm_in = graph.add_tensor_concrete("gemm_in", &[1], DType::F32);
        let gemm_out = graph.add_tensor_concrete("gemm_out", &[1], DType::F32);
        graph.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![gemm_in], vec![gemm_out], "final_proj",
        );
        let argmax_out = graph.add_tensor_concrete("argmax_out", &[1], DType::F32);
        graph.add_op(
            OpKind::Argmax { vocab_size: 32000 },
            vec![gemm_out], vec![argmax_out], "argmax",
        );
        let store_out = graph.add_tensor_concrete("store_out", &[1], DType::F32);
        graph.add_op(OpKind::StoreToken, vec![argmax_out], vec![store_out], "store");
        let check_out = graph.add_tensor_concrete("check_out", &[1], DType::F32);
        graph.add_op(OpKind::CheckStopCondition, vec![store_out], vec![check_out], "check");

        let topo = GraphTopologyAnalysis::analyze(&graph);

        assert_eq!(topo.logits_producer_op_idx, Some(0));
        assert_eq!(topo.logits_output_tid, Some(gemm_out));
    }
}
