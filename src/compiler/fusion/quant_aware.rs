//! Quantization-aware fusion rules.
//!
//! Per SPEC `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md §5`.
//! Extends the base `can_fuse` with dtype/quant compatibility checks.
//! Core principle: same-precision ops attract (fuse freely), cross-precision
//! ops need explicit cast insertion at group boundaries.

use crate::quant::QuantType;
use crate::compiler::trace::QuantPrecision;

/// Fusion decision for quant-aware analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFusionDecision {
    /// Same precision — fuse directly, zero-copy in register.
    Fuse,
    /// Different precision but compatible — fuse with implicit widen (e.g. F16→F32).
    FuseWithWiden,
    /// Quantized boundary — must insert dequant/cast, split into separate groups.
    Split,
}

/// Check if two adjacent ops can fuse considering their quantization formats.
///
/// This is called AFTER the structural `can_fuse` check passes. It adds a
/// dtype/quant compatibility layer on top.
pub fn can_fuse_quant_aware(
    prev_output_quant: Option<QuantType>,
    next_input_quant: Option<QuantType>,
) -> QuantFusionDecision {
    match (prev_output_quant, next_input_quant) {
        // Both unquantized (F32/F16/BF16) — always fuse.
        (None, None) => QuantFusionDecision::Fuse,

        // Same quant type — fuse (e.g. Q4_0 GEMM → Q4_0 GEMM epilogue chain).
        (Some(a), Some(b)) if a == b => QuantFusionDecision::Fuse,

        // QuantGemm output is always F32 (dequant happens inside the kernel).
        // So QuantGemm → ElemWise(F32) is always fusable.
        (Some(_), None) => QuantFusionDecision::Fuse,

        // Unquantized → Quantized input: need requantize, split.
        (None, Some(_)) => QuantFusionDecision::Split,

        // Different quant types: split (can't fuse Q4_0 output into Q6K input).
        (Some(_a), Some(_b)) => QuantFusionDecision::Split,
    }
}

/// Compute fusion cost between two quantization formats.
/// Lower cost = more attractive to fuse.
pub fn fusion_cost(quant_a: Option<QuantType>, quant_b: Option<QuantType>) -> f32 {
    match (quant_a, quant_b) {
        (None, None) => 0.0,
        (Some(a), Some(b)) if a == b => 0.0,
        (Some(_), None) | (None, Some(_)) => 0.5,
        (Some(a), Some(b)) => {
            if both_classic(a, b) { 0.5 }
            else if one_is_kquant(a, b) { 1.5 }
            else if either_is_iq(a, b) { 3.0 }
            else { 1.0 }
        }
    }
}

fn both_classic(a: QuantType, b: QuantType) -> bool {
    is_classic(a) && is_classic(b)
}

fn is_classic(qt: QuantType) -> bool {
    matches!(qt, QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_0
        | QuantType::Q5_1 | QuantType::Q8_0 | QuantType::Q8_1)
}

fn one_is_kquant(a: QuantType, b: QuantType) -> bool {
    is_kquant(a) || is_kquant(b)
}

fn is_kquant(qt: QuantType) -> bool {
    matches!(qt, QuantType::Q2K | QuantType::Q3K | QuantType::Q4K
        | QuantType::Q5K | QuantType::Q6K | QuantType::Q8K)
}

fn either_is_iq(a: QuantType, b: QuantType) -> bool {
    is_iq(a) || is_iq(b)
}

fn is_iq(qt: QuantType) -> bool {
    matches!(qt, QuantType::IQ1S | QuantType::IQ1M | QuantType::IQ2XXS
        | QuantType::IQ2XS | QuantType::IQ2S | QuantType::IQ3XXS
        | QuantType::IQ3S | QuantType::IQ4NL | QuantType::IQ4XS)
}

/// Greedy fusion group selection: consecutive same-precision ops form a group.
/// Group boundaries get dtype cast insertion.
///
/// Returns: Vec of (start_idx, end_idx_exclusive, group_quant_type).
pub fn select_fusion_groups(
    op_quant_types: &[Option<QuantType>],
) -> Vec<(usize, usize, Option<QuantType>)> {
    if op_quant_types.is_empty() {
        return Vec::new();
    }
    let mut groups = Vec::new();
    let mut group_start = 0;
    let mut group_qt = op_quant_types[0];

    for i in 1..op_quant_types.len() {
        let decision = can_fuse_quant_aware(group_qt, op_quant_types[i]);
        match decision {
            QuantFusionDecision::Fuse | QuantFusionDecision::FuseWithWiden => {
                // Continue current group. If prev was quant and next is None (F32 output),
                // the group stays at the quant type (QuantGemm outputs F32 but is still "quant group").
                if group_qt.is_none() && op_quant_types[i].is_some() {
                    group_qt = op_quant_types[i];
                }
            }
            QuantFusionDecision::Split => {
                groups.push((group_start, i, group_qt));
                group_start = i;
                group_qt = op_quant_types[i];
            }
        }
    }
    groups.push((group_start, op_quant_types.len(), group_qt));
    groups
}

/// REQ-DTYPE-003: 检测 widen→compute→narrow→widen 链，返回应合并的融合组对。
///
/// 当相邻融合组形成 widen→compute→narrow→widen 模式时，中间的 narrow→widen
/// 是冗余的（F32→BF16→F32），应将两个组合并为 widen→compute→narrow，
/// 消除不必要的精度转换。
///
/// 参数 `group_dtypes` 是每个融合组的 dominant_dtype 列表。
/// 返回值是应合并的 (left_group_idx, right_group_idx) 对列表。
pub fn detect_widen_narrow_chains(
    group_dtypes: &[Option<QuantPrecision>],
) -> Vec<(usize, usize)> {
    use crate::compiler::trace::QuantPrecision;

    let mut merges = Vec::new();
    if group_dtypes.len() < 2 {
        return merges;
    }

    for i in 0..group_dtypes.len() - 1 {
        let left = group_dtypes[i];
        let right = group_dtypes[i + 1];

        // 检测 narrow→widen 模式：
        // left 组的 dominant dtype > right 组的 dominant dtype
        // 即 left 输出窄 dtype (BF16)，right 输入也窄 dtype (BF16)
        // 但中间经过 F32 计算 → 形成 BF16→F32→BF16 冗余链
        match (left, right) {
            (Some(QuantPrecision::F32), Some(QuantPrecision::BF16)) |
            (Some(QuantPrecision::F32), Some(QuantPrecision::F16)) => {
                // F32→BF16/F16 narrowing at boundary
                // 检查下一个边界是否是 widening（BF16/F16→F32）
                if i + 2 < group_dtypes.len() {
                    if let Some(next_next) = group_dtypes[i + 2] {
                        if next_next.elem_bytes() > right.unwrap().elem_bytes() {
                            // 完整链: F32→BF16→F32 (narrow→widen)
                            // 合并 i+1 和 i+2 为一组
                            merges.push((i + 1, i + 2));
                        }
                    }
                }
            }
            (Some(QuantPrecision::BF16), Some(QuantPrecision::F32)) |
            (Some(QuantPrecision::F16), Some(QuantPrecision::F32)) => {
                // BF16/F16→F32 widening at boundary
                // 检查前一个边界是否是 narrowing
                if i > 0 {
                    if let Some(prev_prev) = group_dtypes[i - 1] {
                        if prev_prev.elem_bytes() > left.unwrap().elem_bytes() {
                            // 完整链: F32→BF16→F32 (narrow→widen)
                            // 合并 i-1 和 i 为一组
                            merges.push((i - 1, i));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_quant_fuses() {
        assert_eq!(
            can_fuse_quant_aware(Some(QuantType::Q4_0), Some(QuantType::Q4_0)),
            QuantFusionDecision::Fuse
        );
    }

    #[test]
    fn quant_to_f32_fuses() {
        assert_eq!(
            can_fuse_quant_aware(Some(QuantType::Q6K), None),
            QuantFusionDecision::Fuse
        );
    }

    #[test]
    fn different_quant_splits() {
        assert_eq!(
            can_fuse_quant_aware(Some(QuantType::Q4_0), Some(QuantType::Q6K)),
            QuantFusionDecision::Split
        );
    }

    #[test]
    fn fusion_groups_mixed_model() {
        // Simulates: Q4_0 layers → Q4_0 layers → Q6K lm_head
        let ops = vec![
            Some(QuantType::Q4_0), // layer.q_proj
            None,                   // layer.norm (F32)
            Some(QuantType::Q4_0), // layer.k_proj
            None,                   // layer.swiglu (F32)
            Some(QuantType::Q6K),  // lm_head
        ];
        let groups = select_fusion_groups(&ops);
        // Q4_0 group: [0..4], Q6K group: [4..5]
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].2, Some(QuantType::Q4_0));
        assert_eq!(groups[1].2, Some(QuantType::Q6K));
    }

    #[test]
    fn fusion_cost_same_is_zero() {
        assert_eq!(fusion_cost(Some(QuantType::Q4_0), Some(QuantType::Q4_0)), 0.0);
        assert_eq!(fusion_cost(None, None), 0.0);
    }

    #[test]
    fn fusion_cost_kquant_is_high() {
        let c = fusion_cost(Some(QuantType::Q4_0), Some(QuantType::Q6K));
        assert!(c > 1.0);
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn both_none_fuses() {
        // Arrange: two unquantized ops
        // Act
        let decision = can_fuse_quant_aware(None, None);
        // Assert
        assert_eq!(decision, QuantFusionDecision::Fuse);
    }

    #[test]
    fn none_to_quant_splits() {
        // Arrange: unquantized output → quantized input
        // Act
        let decision = can_fuse_quant_aware(None, Some(QuantType::Q4_0));
        // Assert
        assert_eq!(decision, QuantFusionDecision::Split);
    }

    #[test]
    fn same_quant_q8_0_fuses() {
        assert_eq!(
            can_fuse_quant_aware(Some(QuantType::Q8_0), Some(QuantType::Q8_0)),
            QuantFusionDecision::Fuse,
        );
    }

    #[test]
    fn quant_fusion_decision_is_copy() {
        let d = QuantFusionDecision::Fuse;
        let copied = d;
        assert_eq!(d, copied);
    }

    #[test]
    fn fusion_groups_empty_input() {
        let groups = select_fusion_groups(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn fusion_groups_single_none() {
        let groups = select_fusion_groups(&[None]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], (0, 1, None));
    }

    #[test]
    fn fusion_groups_all_same_quant() {
        let ops = vec![Some(QuantType::Q5_0); 5];
        let groups = select_fusion_groups(&ops);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].2, Some(QuantType::Q5_0));
        assert_eq!(groups[0].0, 0);
        assert_eq!(groups[0].1, 5);
    }

    #[test]
    fn fusion_cost_cross_quant_nonzero() {
        // Q4_1 and Q5_0 are both classic but different types → cost 0.5
        let c = fusion_cost(Some(QuantType::Q4_1), Some(QuantType::Q5_0));
        assert_eq!(c, 0.5, "cross-classic (different types) should be 0.5");

        // Same type → cost 0
        let same = fusion_cost(Some(QuantType::Q4_0), Some(QuantType::Q4_0));
        assert_eq!(same, 0.0, "same type should have zero cost");

        let c2 = fusion_cost(Some(QuantType::Q4_0), Some(QuantType::Q5K));
        assert!(c2 > 0.0, "cross-precision should have nonzero cost");
    }

    #[test]
    fn fusion_cost_iq_is_highest() {
        let c = fusion_cost(Some(QuantType::Q4_0), Some(QuantType::IQ1S));
        assert!(c >= 3.0, "IQ cross-type should be expensive, got {c}");
    }

    #[test]
    fn select_fusion_groups_quant_then_none_then_different() {
        // Q4_0 → None → Q6K
        let ops = vec![Some(QuantType::Q4_0), None, Some(QuantType::Q6K)];
        let groups = select_fusion_groups(&ops);
        // Q4_0→None fuses (quant output → unquant fuses), then None→Q6K splits
        assert!(groups.len() >= 2, "should split at Q6K boundary");
    }

    // ── REQ-DTYPE-003: widen-narrow chain detection tests ──

    #[test]
    fn detect_widen_narrow_chain_f32_bf16_f32() {
        // F32 → BF16 → F32 pattern: should detect merge at (1, 2)
        let dtypes = vec![
            Some(QuantPrecision::F32),
            Some(QuantPrecision::BF16),
            Some(QuantPrecision::F32),
        ];
        let merges = detect_widen_narrow_chains(&dtypes);
        assert!(!merges.is_empty(), "should detect F32→BF16→F32 chain");
        // The merge should be for the BF16→F32 boundary
        assert!(merges.iter().any(|&(l, r)| l == 1 && r == 2),
            "expected merge at (1, 2), got {:?}", merges);
    }

    #[test]
    fn detect_widen_narrow_chain_no_chain_same_dtype() {
        // BF16 → BF16 → BF16: no chain
        let dtypes = vec![
            Some(QuantPrecision::BF16),
            Some(QuantPrecision::BF16),
            Some(QuantPrecision::BF16),
        ];
        let merges = detect_widen_narrow_chains(&dtypes);
        assert!(merges.is_empty(), "same-dtype chain should have no merges");
    }

    #[test]
    fn detect_widen_narrow_chain_empty() {
        let merges = detect_widen_narrow_chains(&[]);
        assert!(merges.is_empty());
    }

    #[test]
    fn detect_widen_narrow_chain_single_group() {
        let dtypes = vec![Some(QuantPrecision::F32)];
        let merges = detect_widen_narrow_chains(&dtypes);
        assert!(merges.is_empty(), "single group cannot form a chain");
    }

    #[test]
    fn detect_widen_narrow_chain_f16_variant() {
        // F32 → F16 → F32 pattern
        let dtypes = vec![
            Some(QuantPrecision::F32),
            Some(QuantPrecision::F16),
            Some(QuantPrecision::F32),
        ];
        let merges = detect_widen_narrow_chains(&dtypes);
        assert!(!merges.is_empty(), "should detect F32→F16→F32 chain");
    }
}
