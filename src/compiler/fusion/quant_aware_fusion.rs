//! Quantization-aware fusion engine.
//!
//! Implements the **same-precision attraction rule** (SPEC 23-QUANT-CODEGEN-ALGO.md §4):
//! - Adjacent ops with identical quant types fuse into a single group
//! - Precision boundaries are marked for explicit dequant/requant insertion
//! - QuantGemm operators sharing the same quant type are preferentially grouped
//!
//! This is the high-level integration layer that connects the low-level
//! `can_fuse_quant_aware` / `select_fusion_groups` functions to the fusion
//! pipeline (`FusionEngine` → `fuse_with_dag_prebuilt`).

use crate::quant::QuantType;
use super::quant_aware::{can_fuse_quant_aware, QuantFusionDecision, select_fusion_groups, fusion_cost};

/// Fusion rule identification.
///
/// Each rule encodes a distinct strategy. Rules are ordered by priority
/// (higher = applied first). `QuantAwareFusion` sits at the top to ensure
/// same-precision attraction is checked before structural fusion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionRule {
    /// Quantization-aware fusion: same-precision attraction rule.
    /// Adjacent ops with identical quant types fuse preferentially.
    /// Cross-precision boundaries get explicit dequant/requant insertion.
    QuantAwareFusion,
    /// PDT-guided fusion (pain-point driven).
    /// Uses the R0 bottleneck map to gate fusion decisions.
    PdtGuided,
    /// Hardware-constrained fusion (register/cache budget).
    /// Caps epilogue chain length and demotes over-budget groups.
    HardwareConstrained,
    /// Standard fusion (no quantization awareness).
    /// Pure structural compatibility, no dtype/quant checks.
    Standard,
}

impl FusionRule {
    /// Returns `true` if this rule prioritizes same-precision fusion.
    pub fn prefers_same_precision(&self) -> bool {
        matches!(self, FusionRule::QuantAwareFusion)
    }

    /// Priority of this rule (higher = applied earlier in the pipeline).
    pub fn priority(&self) -> u32 {
        match self {
            FusionRule::QuantAwareFusion => 90,
            FusionRule::PdtGuided => 80,
            FusionRule::HardwareConstrained => 70,
            FusionRule::Standard => 60,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            FusionRule::QuantAwareFusion => {
                "same-precision attraction: quant-identical ops fuse, cross-precision split"
            }
            FusionRule::PdtGuided => "PDT-guided fusion: pain-point driven cost model",
            FusionRule::HardwareConstrained => {
                "Hardware-constrained: register/cache budget caps fusion depth"
            }
            FusionRule::Standard => "Standard fusion: structural compatibility only",
        }
    }
}

/// Quantization-aware fusion engine.
///
/// This engine orchestrates fusion decisions with quantization awareness.
/// It wraps the lower-level `can_fuse_quant_aware` and `select_fusion_groups`
/// functions and integrates them with the fusion pipeline.
///
/// # Same-Precision Attraction
///
/// The core principle: **ops using the same quant type attract** and are
/// fused together, while **precision boundaries** get explicit dequant/requant
/// operations inserted. This avoids unnecessary dequant → compute → requant
/// round-trips between ops of the same precision.
///
/// # Usage
///
/// ```ignore
/// let engine = FusionEngine::with_quant_aware();
/// if engine.should_merge_groups(prev_quant, next_quant) {
///     // fuse into same group
/// } else {
///     // split groups, insert dequant/requant at boundary
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FusionEngine {
    /// Active fusion rules, in priority order (first = highest).
    pub rules: Vec<FusionRule>,
}

impl FusionEngine {
    /// Create a new `FusionEngine` with the given rules.
    /// Rules retain their caller-specified order (no auto-sort).
    pub fn new(rules: Vec<FusionRule>) -> Self {
        Self { rules }
    }

    /// Create a `FusionEngine` with all available rules enabled.
    /// Rules are ordered by descending priority.
    pub fn all() -> Self {
        let mut rules = vec![
            FusionRule::QuantAwareFusion,
            FusionRule::PdtGuided,
            FusionRule::HardwareConstrained,
            FusionRule::Standard,
        ];
        rules.sort_by_key(|r| std::cmp::Reverse(r.priority()));
        Self { rules }
    }

    /// Create a `FusionEngine` with only the quant-aware rule enabled.
    pub fn quant_aware_only() -> Self {
        Self {
            rules: vec![FusionRule::QuantAwareFusion],
        }
    }

    /// Create a `FusionEngine` with standard (non-quant-aware) rules only.
    pub fn standard_only() -> Self {
        Self {
            rules: vec![FusionRule::Standard],
        }
    }

    /// Check if a specific rule is active in this engine.
    pub fn has_rule(&self, rule: FusionRule) -> bool {
        self.rules.contains(&rule)
    }

    /// Returns `true` if quantization-aware fusion is active.
    pub fn is_quant_aware(&self) -> bool {
        self.has_rule(FusionRule::QuantAwareFusion)
    }

    /// Return the active rules sorted by priority (highest first).
    pub fn sorted_rules(&self) -> Vec<FusionRule> {
        let mut sorted = self.rules.clone();
        sorted.sort_by_key(|r| std::cmp::Reverse(r.priority()));
        sorted
    }

    /// Determine if two adjacent fusion groups should be merged
    /// based on their quantization types.
    ///
    /// When `QuantAwareFusion` is active, same-precision groups merge
    /// preferentially and cross-precision groups are kept separate.
    /// Without quant awareness, always returns `true` (structural merge).
    ///
    /// # Arguments
    ///
    /// * `group_a_quant` — Quant type of the first group (output precision).
    /// * `group_b_quant` — Quant type of the second group (input precision).
    ///
    /// # Returns
    ///
    /// * `true` — groups should be merged (same precision or unquantized).
    /// * `false` — groups must be split, dequant/requant needed at boundary.
    pub fn should_merge_groups(
        &self,
        group_a_quant: Option<QuantType>,
        group_b_quant: Option<QuantType>,
    ) -> bool {
        if !self.is_quant_aware() {
            return true;
        }
        match can_fuse_quant_aware(group_a_quant, group_b_quant) {
            QuantFusionDecision::Fuse | QuantFusionDecision::FuseWithWiden => true,
            QuantFusionDecision::Split => false,
        }
    }

    /// Compute the cost of fusing two groups with the given quant types.
    ///
    /// Lower cost = more attractive to fuse. Same-precision pairs cost 0.
    /// Cross-precision pairs cost more depending on the quant family distance.
    pub fn merge_cost(
        &self,
        quant_a: Option<QuantType>,
        quant_b: Option<QuantType>,
    ) -> f32 {
        if !self.is_quant_aware() {
            return 0.0;
        }
        fusion_cost(quant_a, quant_b)
    }

    /// Select fusion groups from a sequence of ops with their quant types.
    ///
    /// Uses `select_fusion_groups` internally when quant-aware.
    /// Returns `(start_idx, end_idx_exclusive, group_quant_type)` tuples.
    pub fn select_groups(
        &self,
        op_quant_types: &[Option<QuantType>],
    ) -> Vec<(usize, usize, Option<QuantType>)> {
        if !self.is_quant_aware() {
            // Without quant awareness, everything goes in one group
            if op_quant_types.is_empty() {
                return Vec::new();
            }
            return vec![(0, op_quant_types.len(), op_quant_types[0])];
        }
        select_fusion_groups(op_quant_types)
    }

    /// Insert explicit dequant/requant operations at precision boundaries.
    ///
    /// When `QuantAwareFusion` is active, cross-precision boundaries between
    /// groups need explicit cast operations (dequantize old precision → F32 →
    /// requantize to new precision). This function identifies those boundaries.
    ///
    /// Returns a list of boundary indices where casts are needed.
    pub fn find_cast_boundaries(
        &self,
        op_quant_types: &[Option<QuantType>],
    ) -> Vec<usize> {
        if !self.is_quant_aware() || op_quant_types.is_empty() {
            return Vec::new();
        }

        let groups = self.select_groups(op_quant_types);
        let mut boundaries = Vec::new();

        for i in 1..groups.len() {
            let (_, prev_end, prev_quant) = groups[i - 1];
            let (curr_start, _, curr_quant) = groups[i];
            // Boundary needs cast if quant types differ
            if prev_quant != curr_quant {
                boundaries.push(curr_start);
            }
            // Also record if we went from quant → none → quant (cast to F32 at boundary)
            if prev_quant.is_some() && curr_quant.is_some() && prev_quant != curr_quant {
                boundaries.push(curr_start);
            }
        }

        boundaries
    }
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FusionRule tests ──────────────────────────────────────────────

    #[test]
    fn test_quant_aware_fusion_prefers_same_precision() {
        assert!(FusionRule::QuantAwareFusion.prefers_same_precision());
        assert!(!FusionRule::Standard.prefers_same_precision());
    }

    #[test]
    fn test_fusion_rule_priority_ordering() {
        assert!(FusionRule::QuantAwareFusion.priority() > FusionRule::Standard.priority());
        assert!(FusionRule::PdtGuided.priority() > FusionRule::HardwareConstrained.priority());
    }

    #[test]
    fn test_fusion_rule_descriptions() {
        let desc = FusionRule::QuantAwareFusion.description();
        assert!(desc.contains("same-precision"), "description should mention same-precision");
    }

    // ── FusionEngine tests ────────────────────────────────────────────

    #[test]
    fn test_engine_quant_aware_detection() {
        let engine = FusionEngine::quant_aware_only();
        assert!(engine.is_quant_aware());
        assert!(engine.has_rule(FusionRule::QuantAwareFusion));
        assert!(!engine.has_rule(FusionRule::Standard));
    }

    #[test]
    fn test_engine_standard_only() {
        let engine = FusionEngine::standard_only();
        assert!(!engine.is_quant_aware());
        assert!(engine.has_rule(FusionRule::Standard));
        assert!(!engine.has_rule(FusionRule::QuantAwareFusion));
    }

    #[test]
    fn test_engine_all_has_quant_aware() {
        let engine = FusionEngine::all();
        assert!(engine.is_quant_aware());
        assert!(engine.has_rule(FusionRule::QuantAwareFusion));
        assert!(engine.has_rule(FusionRule::Standard));
        assert!(engine.has_rule(FusionRule::PdtGuided));
        assert!(engine.has_rule(FusionRule::HardwareConstrained));
    }

    #[test]
    fn test_same_precision_merge() {
        let engine = FusionEngine::all();
        // Same quant type → fuse
        assert!(engine.should_merge_groups(Some(QuantType::Q4_0), Some(QuantType::Q4_0)));
        // Both unquantized → fuse
        assert!(engine.should_merge_groups(None, None));
        // QuantGemm output (F32) → elementwise → fuse
        assert!(engine.should_merge_groups(Some(QuantType::Q4_0), None));
    }

    #[test]
    fn test_cross_precision_split() {
        let engine = FusionEngine::all();
        // Different quant types → split
        assert!(!engine.should_merge_groups(Some(QuantType::Q4_0), Some(QuantType::Q6K)));
        // None → quantized → split (need requant)
        assert!(!engine.should_merge_groups(None, Some(QuantType::Q4_0)));
    }

    #[test]
    fn test_standard_engine_merges_all() {
        let engine = FusionEngine::standard_only();
        // Without quant awareness, all pairs merge
        assert!(engine.should_merge_groups(Some(QuantType::Q4_0), Some(QuantType::Q6K)));
        assert!(engine.should_merge_groups(None, Some(QuantType::Q4_0)));
    }

    #[test]
    fn test_merge_cost_same_precision_zero() {
        let engine = FusionEngine::all();
        assert_eq!(engine.merge_cost(Some(QuantType::Q4_0), Some(QuantType::Q4_0)), 0.0);
        assert_eq!(engine.merge_cost(None, None), 0.0);
    }

    #[test]
    fn test_merge_cost_cross_precision_positive() {
        let engine = FusionEngine::all();
        let cost = engine.merge_cost(Some(QuantType::Q4_0), Some(QuantType::Q6K));
        assert!(cost > 0.0, "cross-precision merge should have positive cost");
    }

    #[test]
    fn test_standard_engine_cost_zero() {
        let engine = FusionEngine::standard_only();
        assert_eq!(engine.merge_cost(Some(QuantType::Q4_0), Some(QuantType::Q6K)), 0.0);
    }

    // ── Group selection tests ─────────────────────────────────────────

    #[test]
    fn test_select_groups_quant_aware() {
        let engine = FusionEngine::all();
        let ops = vec![
            Some(QuantType::Q4_0),
            Some(QuantType::Q4_0),
            None,
            Some(QuantType::Q6K),
        ];
        let groups = engine.select_groups(&ops);
        // Q4_0 group [0..3], Q6K group [3..4]
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].2, Some(QuantType::Q4_0));
        assert_eq!(groups[1].2, Some(QuantType::Q6K));
    }

    #[test]
    fn test_select_groups_standard() {
        let engine = FusionEngine::standard_only();
        let ops = vec![
            Some(QuantType::Q4_0),
            Some(QuantType::Q6K),
        ];
        let groups = engine.select_groups(&ops);
        // Without quant awareness, everything is one group
        assert_eq!(groups.len(), 1);
    }

    // ── Cast boundary tests ───────────────────────────────────────────

    #[test]
    fn test_find_cast_boundaries_same_precision() {
        let engine = FusionEngine::all();
        let ops = vec![
            Some(QuantType::Q4_0),
            Some(QuantType::Q4_0),
            None,
        ];
        let boundaries = engine.find_cast_boundaries(&ops);
        // No boundaries: same Q4_0 throughout, then None (fusable)
        assert!(boundaries.is_empty(), "same precision chain should have no cast boundaries");
    }

    #[test]
    fn test_find_cast_boundaries_cross_precision() {
        let engine = FusionEngine::all();
        let ops = vec![
            Some(QuantType::Q4_0),
            Some(QuantType::Q4_0),
            Some(QuantType::Q6K),
        ];
        let boundaries = engine.find_cast_boundaries(&ops);
        // Boundary at index 2 (transition from Q4_0 to Q6K)
        assert!(!boundaries.is_empty(), "cross-precision should have cast boundaries");
    }

    #[test]
    fn test_find_cast_boundaries_empty() {
        let engine = FusionEngine::all();
        assert!(engine.find_cast_boundaries(&[]).is_empty());
    }

    #[test]
    fn test_find_cast_boundaries_standard_engine() {
        let engine = FusionEngine::standard_only();
        let ops = vec![
            Some(QuantType::Q4_0),
            Some(QuantType::Q6K),
        ];
        // Standard engine doesn't find boundaries
        assert!(engine.find_cast_boundaries(&ops).is_empty());
    }

    // ── Sorted rules tests ────────────────────────────────────────────

    #[test]
    fn test_sorted_rules_priority_order() {
        let engine = FusionEngine::new(vec![
            FusionRule::Standard,
            FusionRule::QuantAwareFusion,
            FusionRule::HardwareConstrained,
        ]);
        let sorted = engine.sorted_rules();
        assert_eq!(sorted[0], FusionRule::QuantAwareFusion, "QuantAwareFusion should be first");
        assert_eq!(sorted[1], FusionRule::HardwareConstrained);
        assert_eq!(sorted[2], FusionRule::Standard);
    }

    #[test]
    fn test_default_engine() {
        let engine = FusionEngine::default();
        assert!(engine.is_quant_aware());
        assert!(engine.has_rule(FusionRule::Standard));
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn test_fusion_rule_all_descriptions_non_empty() {
        let rules = [
            FusionRule::QuantAwareFusion,
            FusionRule::PdtGuided,
            FusionRule::HardwareConstrained,
            FusionRule::Standard,
        ];
        for rule in &rules {
            let desc = rule.description();
            assert!(!desc.is_empty(), "description for {:?} should not be empty", rule);
        }
    }

    #[test]
    fn test_fusion_rule_hash_and_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FusionRule::QuantAwareFusion);
        set.insert(FusionRule::Standard);
        set.insert(FusionRule::QuantAwareFusion); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_fusion_rule_copy() {
        let a = FusionRule::PdtGuided;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_fusion_rule_debug_format() {
        let debug = format!("{:?}", FusionRule::HardwareConstrained);
        assert!(debug.contains("HardwareConstrained"));
    }

    #[test]
    fn test_engine_new_custom_rules() {
        let engine = FusionEngine::new(vec![FusionRule::PdtGuided]);
        assert!(!engine.is_quant_aware());
        assert!(engine.has_rule(FusionRule::PdtGuided));
        assert!(!engine.has_rule(FusionRule::Standard));
    }

    #[test]
    fn test_engine_new_empty_rules() {
        let engine = FusionEngine::new(vec![]);
        assert!(!engine.is_quant_aware());
        assert!(engine.rules.is_empty());
    }

    #[test]
    fn test_engine_clone_preserves_rules() {
        let engine = FusionEngine::all();
        let cloned = engine.clone();
        assert_eq!(engine.rules.len(), cloned.rules.len());
        assert!(cloned.is_quant_aware());
    }

    #[test]
    fn test_engine_debug_format() {
        let engine = FusionEngine::quant_aware_only();
        let debug = format!("{:?}", engine);
        assert!(debug.contains("FusionEngine"));
    }

    #[test]
    fn test_select_groups_empty_ops() {
        let engine = FusionEngine::all();
        let groups = engine.select_groups(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_select_groups_single_op() {
        let engine = FusionEngine::all();
        let ops = vec![Some(QuantType::Q4_0)];
        let groups = engine.select_groups(&ops);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, 0);
        assert_eq!(groups[0].1, 1);
        assert_eq!(groups[0].2, Some(QuantType::Q4_0));
    }

    #[test]
    fn test_select_groups_all_same_precision() {
        let engine = FusionEngine::all();
        let ops = vec![
            Some(QuantType::Q4_0),
            Some(QuantType::Q4_0),
            Some(QuantType::Q4_0),
        ];
        let groups = engine.select_groups(&ops);
        assert_eq!(groups.len(), 1, "all same precision should form one group");
        assert_eq!(groups[0].1, 3);
    }

    #[test]
    fn test_merge_cost_none_quant_pair() {
        let engine = FusionEngine::all();
        let cost = engine.merge_cost(None, Some(QuantType::Q4_0));
        // None → QuantType is a cross-precision boundary
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_sorted_rules_empty_engine() {
        let engine = FusionEngine::new(vec![]);
        let sorted = engine.sorted_rules();
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_should_merge_groups_none_none() {
        let engine = FusionEngine::quant_aware_only();
        assert!(engine.should_merge_groups(None, None));
    }
}
