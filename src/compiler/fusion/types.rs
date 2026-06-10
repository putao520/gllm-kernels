//! Fusion types — FusionGroup, FusionMode, FusionPlan, FusionCost, GroupMarker, HeteroLayerType.

use std::collections::HashMap;
use crate::compiler::graph::{CompilerGraph, MultiOutputConfig, OpId};
use crate::compiler::trace::QuantPrecision;

/// REQ-UMK-012: 异构层类型枚举 (Gemma 4 风格: sliding/full × small/large FFN).
/// 携带在 FusionGroup.hetero_layer_type 上，从图拓扑推导而非 label 前缀。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeteroLayerType {
    /// Sliding attention + small FFN
    SlidingSmall,
    /// Full attention + small FFN
    FullSmall,
    /// Sliding attention + large FFN
    SlidingLarge,
    /// Full attention + large FFN
    FullLarge,
}

/// Fusion 融合引擎在组序列中插入的结构标记 (REQ-UMK-012)。
///
/// 标记位置由 CompilerGraph 的层拓扑推导，不由编译器假设。
/// 替代 pipeline.inc.rs 中 `anchor_op.label.starts_with("layer.")` 的 label 约定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupMarker {
    /// 同构层循环开始。num_iterations 从图拓扑推导（如 35 层 → LayerLoopBegin { num_iterations: 35 }）。
    LayerLoopBegin { num_iterations: usize },
    /// 同构层循环结束。
    LayerLoopEnd,
    /// 异构层循环开始（Gemma 4 E2B: sliding/full 交替）。
    /// 4 种层类型模板按段循环，num_segments 从图拓扑推导。
    HeteroLayerLoopBegin { num_segments: usize },
    /// 异构层循环结束。
    HeteroLayerLoopEnd,
    /// ForwardPhaseDispatch 三路分支（仅多步生成图：含 Argmax 的图）。
    PhaseDispatch,
    /// 无标记——普通融合组。
    None,
}

impl Default for GroupMarker {
    fn default() -> Self {
        GroupMarker::None
    }
}

/// A group of fused operations that will be compiled as a single unit.
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Unique group ID.
    pub id: usize,
    /// The "anchor" op — determines the primary computation pattern.
    /// For GEMM fusion, this is the GEMM op.
    /// For elementwise chains, this is the first op.
    pub anchor: OpId,
    /// Ops absorbed into this group's epilogue (in execution order).
    pub epilogue: Vec<OpId>,
    /// The fusion mode that was applied.
    pub mode: FusionMode,
    /// All op IDs in this group (anchor + epilogue), in execution order.
    pub ops: Vec<OpId>,
    /// Multi-output ABI configuration (DEEP-001).
    /// Default: single output (legacy ABI, zero overhead).
    pub multi_output: MultiOutputConfig,
    /// REQ-DTYPE-003: 融合组的主导 dtype (dominant dtype)。
    /// 从 anchor op 的第一个输入 tensor 的 dtype 推导。
    /// None = 无法推导（使用默认 F32）。
    pub dominant_dtype: Option<QuantPrecision>,
    /// REQ-UMK-012: 融合引擎插入的结构标记（LayerLoopBegin/End, PhaseDispatch, etc.）。
    /// 默认 GroupMarker::None。编译器从 marker 读取层循环信息，不搜索 op.label。
    pub marker: GroupMarker,
    /// REQ-UMK-012: 此融合组属于层循环体（由 assign_group_markers 从图拓扑推导）。
    /// 替代 pipeline.inc.rs 中 `anchor_op.label.starts_with("layer.")` 的 label 约定。
    pub is_layer_group: bool,
    /// REQ-UMK-012: 异构层子类型（从 OpKind 签名推导，非 label 前缀）。
    /// 替代 hetero_emit.rs/pipeline.inc.rs 中 `anchor_op.label.starts_with("layer_sliding_small.")` 等。
    /// None = 非异构层组或非层组。
    pub hetero_layer_type: Option<HeteroLayerType>,
}

impl FusionGroup {
    /// 从 graph 推导融合组的主导 dtype (REQ-DTYPE-003)。
    /// 策略: anchor op 的第一个输入 tensor 的 dtype。
    pub fn infer_dominant_dtype(&mut self, graph: &CompilerGraph) {
        if let Some(op) = graph.op(self.anchor) {
            if let Some(&tid) = op.inputs.first() {
                if let Some(tensor) = graph.tensor(tid) {
                    self.dominant_dtype = Some(tensor.dtype.to_quant_precision());
                    return;
                }
            }
        }
        self.dominant_dtype = None;
    }
}

/// Named fusion modes recognized by the pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionMode {
    /// Single op, no fusion applied.
    Standalone,
    /// GEMM with fused elementwise epilogue (e.g., GEMM + SiLU, GEMM + Add).
    EpilogueInjection,
    /// Chain of elementwise ops collapsed into a single loop.
    LoopFusion,
    /// Tile-level fusion: predecessor tile computation embedded in GEMM MC loop.
    /// Used when predecessor output > 75% L1 — tiles are computed per MC strip.
    TileLevelFusion {
        /// The predecessor op (e.g. RmsNorm) whose output is tiled into the GEMM MC loop.
        predecessor: OpId,
        /// Number of rows per tile (= MC from GEMM blocking).
        tile_rows: usize,
    },
    /// Compute root: predecessor computed fully before GEMM, result stays in L1/L2.
    /// Used when predecessor output ≤ 75% L1.
    ComputeRoot {
        /// The predecessor op computed as a standalone root.
        predecessor: OpId,
    },
    /// Three QKV GEMMs sharing the same input → single pack_a.
    QkvSharedInput,
    /// RmsNorm output feeds directly into GEMM (no intermediate writeback).
    NormIntoGemm,
    /// FFN full fusion: Gate+Up GEMMs (shared input) → activation → Mul → Down GEMM.
    FFNBlock {
        /// Gate GEMM op.
        gate_gemm: OpId,
        /// Up GEMM op.
        up_gemm: OpId,
        /// Activation op (SiLU/GeLU).
        activation: OpId,
        /// Combine op (Mul).
        combine: OpId,
    },
    /// Cross-layer residual: Layer N's Residual Add + Layer N+1's RmsNorm.
    CrossLayerResidual {
        /// The Residual Add op.
        residual: OpId,
        /// The RmsNorm op from the next layer.
        norm: OpId,
    },
    /// Fused QKV projection + QkNorm + ValueNorm + RoPE (Gemma 4 pattern).
    ///
    /// Fuses: Gemm(Q) + Gemm(K) + Gemm(V) + QkNorm(Q,K) + ValueNorm(V) + RoPE(Q) + RoPE(K)
    /// into a single kernel. Triggered only when QkNorm and ValueNorm are present
    /// (standard models without these norms are unaffected).
    ///
    /// The fused node walks the NormLike codegen path: the QkNorm/ValueNorm/RoPE
    /// operations are folded into the epilogue of the shared QKV GEMM pack_a region.
    FusedQkvNormRope {
        /// The three GEMM ops (Q, K, V projections).
        gemm_q: OpId,
        gemm_k: OpId,
        gemm_v: OpId,
        /// QkNorm applied to Q output.
        qk_norm_q: OpId,
        /// QkNorm applied to K output.
        qk_norm_k: OpId,
        /// ValueNorm applied to V output.
        value_norm_v: OpId,
        /// RoPE applied to normalized Q.
        rope_q: OpId,
        /// RoPE applied to normalized K.
        rope_k: OpId,
    },
}

/// Result of the fusion pass.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Fusion groups in execution order.
    pub groups: Vec<FusionGroup>,
    /// Map from OpId → group index (for quick lookup).
    pub op_to_group: HashMap<OpId, usize>,
}

impl FusionPlan {
    /// Number of fusion groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get the group containing a specific op.
    pub fn group_of(&self, op: OpId) -> Option<&FusionGroup> {
        self.op_to_group.get(&op).map(|&idx| &self.groups[idx])
    }

    /// Count how many ops were fused (not standalone).
    pub fn num_fused_ops(&self) -> usize {
        self.groups
            .iter()
            .filter(|g| g.mode != FusionMode::Standalone)
            .map(|g| g.ops.len())
            .sum()
    }
}

impl std::fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FusionPlan: {} groups", self.groups.len())?;
        for g in &self.groups {
            let ops_str: Vec<String> = g.ops.iter().map(|o| format!("{}", o.0)).collect();
            writeln!(
                f,
                "  [{}] {:?} anchor=Op({}) marker={:?} ops=[{}]",
                g.id,
                g.mode,
                g.anchor.0,
                g.marker,
                ops_str.join(", ")
            )?;
        }
        Ok(())
    }
}

/// Cost estimate for a fusion decision.
#[derive(Debug, Clone)]
pub struct FusionCost {
    /// Bytes of intermediate data eliminated by fusion (saved memory traffic).
    pub bytes_saved: usize,
    /// Extra registers consumed by the fused kernel vs separate kernels.
    pub extra_regs: usize,
    /// Scratch buffer bytes needed for tiled fusion (0 for epilogue/loop fusion).
    pub scratch_bytes: usize,
    /// Net benefit score: positive means fusion is profitable.
    /// `benefit = bytes_saved - penalty`, where penalty accounts for register
    /// spill cost and scratch buffer overhead.
    pub benefit: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_group(id: usize, mode: FusionMode, ops: Vec<OpId>) -> FusionGroup {
        FusionGroup {
            id,
            anchor: ops[0],
            epilogue: ops[1..].to_vec(),
            mode,
            ops,
            multi_output: MultiOutputConfig::default(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }
    }

    // ── FusionMode ──

    #[test]
    fn fusion_mode_equality() {
        assert_eq!(FusionMode::Standalone, FusionMode::Standalone);
        assert_eq!(FusionMode::EpilogueInjection, FusionMode::EpilogueInjection);
        assert_ne!(FusionMode::Standalone, FusionMode::LoopFusion);
        assert_ne!(FusionMode::EpilogueInjection, FusionMode::NormIntoGemm);
    }

    // ── FusionPlan ──

    #[test]
    fn plan_empty() {
        let plan = FusionPlan {
            groups: vec![],
            op_to_group: HashMap::new(),
        };
        assert_eq!(plan.num_groups(), 0);
        assert_eq!(plan.num_fused_ops(), 0);
    }

    #[test]
    fn plan_group_of() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let g = make_group(0, FusionMode::EpilogueInjection, vec![op0, op1]);
        let plan = FusionPlan {
            groups: vec![g],
            op_to_group: HashMap::from([(op0, 0), (op1, 0)]),
        };
        assert_eq!(plan.num_groups(), 1);
        let found = plan.group_of(op0).unwrap();
        assert_eq!(found.id, 0);
        assert!(plan.group_of(OpId(99)).is_none());
    }

    #[test]
    fn plan_num_fused_ops_excludes_standalone() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let op2 = OpId(2);
        let g_standalone = make_group(0, FusionMode::Standalone, vec![op0]);
        let g_fused = make_group(1, FusionMode::LoopFusion, vec![op1, op2]);
        let plan = FusionPlan {
            groups: vec![g_standalone, g_fused],
            op_to_group: HashMap::from([(op0, 0), (op1, 1), (op2, 1)]),
        };
        assert_eq!(plan.num_fused_ops(), 2); // only the fused group's ops
    }

    #[test]
    fn plan_display() {
        let op0 = OpId(0);
        let g = make_group(0, FusionMode::Standalone, vec![op0]);
        let plan = FusionPlan {
            groups: vec![g],
            op_to_group: HashMap::from([(op0, 0)]),
        };
        let s = plan.to_string();
        assert!(s.contains("1 groups"));
        assert!(s.contains("Standalone"));
    }

    // ── FusionCost ──

    #[test]
    fn cost_benefit_positive() {
        let cost = FusionCost {
            bytes_saved: 4096,
            extra_regs: 2,
            scratch_bytes: 0,
            benefit: 4096,
        };
        assert!(cost.benefit > 0);
    }

    // ── FusionMode Debug format ──

    #[test]
    fn fusion_mode_debug_format() {
        assert!(format!("{:?}", FusionMode::Standalone).contains("Standalone"));
        assert!(format!("{:?}", FusionMode::QkvSharedInput).contains("QkvSharedInput"));
        assert!(format!("{:?}", FusionMode::NormIntoGemm).contains("NormIntoGemm"));
    }

    // ── FusionMode Clone ──

    #[test]
    fn fusion_mode_clone() {
        let mode = FusionMode::TileLevelFusion { predecessor: OpId(5), tile_rows: 64 };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    // ── FusionGroup construction ──

    #[test]
    fn fusion_group_fields() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let g = FusionGroup {
            id: 42,
            anchor: op0,
            epilogue: vec![op1],
            mode: FusionMode::LoopFusion,
            ops: vec![op0, op1],
            multi_output: MultiOutputConfig::default(),
            dominant_dtype: Some(QuantPrecision::BF16),
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert_eq!(g.id, 42);
        assert_eq!(g.ops.len(), 2);
        assert_eq!(g.dominant_dtype, Some(QuantPrecision::BF16));
    }

    // ── FusionGroup Debug ──

    #[test]
    fn fusion_group_debug() {
        let op0 = OpId(0);
        let g = make_group(0, FusionMode::Standalone, vec![op0]);
        let debug = format!("{:?}", g);
        assert!(debug.contains("Standalone"));
    }

    // ── FusionPlan with multiple groups ──

    #[test]
    fn plan_multiple_groups_display() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let op2 = OpId(2);
        let g0 = make_group(0, FusionMode::Standalone, vec![op0]);
        let g1 = make_group(1, FusionMode::EpilogueInjection, vec![op1, op2]);
        let plan = FusionPlan {
            groups: vec![g0, g1],
            op_to_group: HashMap::from([(op0, 0), (op1, 1), (op2, 1)]),
        };
        let s = plan.to_string();
        assert!(s.contains("2 groups"));
        assert!(s.contains("EpilogueInjection"));
        assert_eq!(plan.num_groups(), 2);
    }

    // ── FusionCost clone preserves fields ──

    #[test]
    fn fusion_cost_clone() {
        let cost = FusionCost {
            bytes_saved: 1024,
            extra_regs: 4,
            scratch_bytes: 256,
            benefit: 512,
        };
        let cloned = cost.clone();
        assert_eq!(cloned.bytes_saved, 1024);
        assert_eq!(cloned.extra_regs, 4);
        assert_eq!(cloned.scratch_bytes, 256);
        assert_eq!(cloned.benefit, 512);
    }

    // ── FusionCost Debug ──

    #[test]
    fn fusion_cost_debug() {
        let cost = FusionCost {
            bytes_saved: 0,
            extra_regs: 0,
            scratch_bytes: 0,
            benefit: -100,
        };
        let debug = format!("{:?}", cost);
        assert!(debug.contains("FusionCost"));
    }

    // ── FFNBlock mode construction ──

    #[test]
    fn ffn_block_mode_construction() {
        let mode = FusionMode::FFNBlock {
            gate_gemm: OpId(0),
            up_gemm: OpId(1),
            activation: OpId(2),
            combine: OpId(3),
        };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
        assert!(format!("{:?}", mode).contains("FFNBlock"));
    }

    // ── CrossLayerResidual mode ──

    #[test]
    fn cross_layer_residual_mode() {
        let mode = FusionMode::CrossLayerResidual {
            residual: OpId(10),
            norm: OpId(11),
        };
        assert!(format!("{:?}", mode).contains("CrossLayerResidual"));
        assert_eq!(mode, mode.clone());
    }

    // ── FusionPlan num_fused_ops all standalone ──

    #[test]
    fn plan_all_standalone_zero_fused() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let g0 = make_group(0, FusionMode::Standalone, vec![op0]);
        let g1 = make_group(1, FusionMode::Standalone, vec![op1]);
        let plan = FusionPlan {
            groups: vec![g0, g1],
            op_to_group: HashMap::from([(op0, 0), (op1, 1)]),
        };
        assert_eq!(plan.num_fused_ops(), 0);
    }

    // ── Additional tests ──

    #[test]
    fn tile_level_fusion_mode_construction() {
        let mode = FusionMode::TileLevelFusion {
            predecessor: OpId(7),
            tile_rows: 128,
        };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
        let debug = format!("{:?}", mode);
        assert!(debug.contains("TileLevelFusion"));
        assert!(debug.contains("tile_rows"));
    }

    #[test]
    fn compute_root_mode_construction() {
        let mode = FusionMode::ComputeRoot {
            predecessor: OpId(3),
        };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
        let debug = format!("{:?}", mode);
        assert!(debug.contains("ComputeRoot"));
    }

    #[test]
    fn qkv_shared_input_mode() {
        let mode = FusionMode::QkvSharedInput;
        assert_eq!(mode, FusionMode::QkvSharedInput);
        assert_ne!(mode, FusionMode::Standalone);
        assert!(format!("{:?}", mode).contains("QkvSharedInput"));
    }

    #[test]
    fn fused_qkv_norm_rope_mode_construction() {
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
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
        let debug = format!("{:?}", mode);
        assert!(debug.contains("FusedQkvNormRope"));
        assert!(debug.contains("gemm_q"));
        assert!(debug.contains("rope_k"));
    }

    #[test]
    fn fusion_mode_inequality_cross_variants() {
        // Different variants should never be equal
        assert_ne!(FusionMode::Standalone, FusionMode::LoopFusion);
        assert_ne!(FusionMode::QkvSharedInput, FusionMode::NormIntoGemm);
        assert_ne!(FusionMode::EpilogueInjection, FusionMode::QkvSharedInput);
        assert_ne!(
            FusionMode::TileLevelFusion { predecessor: OpId(0), tile_rows: 64 },
            FusionMode::ComputeRoot { predecessor: OpId(0) }
        );
    }

    #[test]
    fn plan_display_with_loop_fusion() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let g = make_group(5, FusionMode::LoopFusion, vec![op0, op1]);
        let plan = FusionPlan {
            groups: vec![g],
            op_to_group: HashMap::from([(op0, 0), (op1, 0)]),
        };
        let s = plan.to_string();
        assert!(s.contains("LoopFusion"));
        assert!(s.contains("anchor=Op(0)"));
        assert!(s.contains("[5]"));
    }

    #[test]
    fn plan_group_of_returns_correct_mode() {
        let op0 = OpId(0);
        let op1 = OpId(1);
        let op2 = OpId(2);
        let g0 = make_group(0, FusionMode::NormIntoGemm, vec![op0]);
        let g1 = make_group(1, FusionMode::FFNBlock {
            gate_gemm: op1,
            up_gemm: OpId(3),
            activation: OpId(4),
            combine: OpId(5),
        }, vec![op1, OpId(3), OpId(4), OpId(5)]);
        let plan = FusionPlan {
            groups: vec![g0, g1],
            op_to_group: HashMap::from([(op0, 0), (op1, 1)]),
        };
        assert!(matches!(plan.group_of(op0).unwrap().mode, FusionMode::NormIntoGemm));
        assert!(matches!(plan.group_of(op1).unwrap().mode, FusionMode::FFNBlock { .. }));
    }

    #[test]
    fn fusion_group_default_dominant_dtype() {
        let op0 = OpId(0);
        let g = FusionGroup {
            id: 0,
            anchor: op0,
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op0],
            multi_output: MultiOutputConfig::default(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert!(g.dominant_dtype.is_none());
    }

    #[test]
    fn fusion_cost_negative_benefit_unprofitable() {
        let cost = FusionCost {
            bytes_saved: 64,
            extra_regs: 32,
            scratch_bytes: 4096,
            benefit: -1024,
        };
        assert!(cost.benefit < 0);
        let debug = format!("{:?}", cost);
        assert!(debug.contains("-1024"));
    }

    #[test]
    fn fusion_plan_display_shows_ops_list() {
        let op0 = OpId(10);
        let op1 = OpId(11);
        let op2 = OpId(12);
        let g = make_group(0, FusionMode::EpilogueInjection, vec![op0, op1, op2]);
        let plan = FusionPlan {
            groups: vec![g],
            op_to_group: HashMap::from([(op0, 0), (op1, 0), (op2, 0)]),
        };
        let s = plan.to_string();
        assert!(s.contains("10, 11, 12"));
    }
}
