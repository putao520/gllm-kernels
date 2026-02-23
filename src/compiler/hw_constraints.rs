use crate::compiler::fusion::{FusionGroup, FusionPattern};
use crate::compiler::graph::{CompilerGraph, OpKind};
use crate::dispatch::DeviceProfile;

/// Result of hardware constraint checking for a fusion group.
#[derive(Debug, Clone)]
pub struct HwConstraintResult {
    /// Which group was checked.
    pub group_id: usize,
    /// Whether all constraints are satisfied.
    pub valid: bool,
    /// Estimated SIMD register pressure (number of registers needed).
    pub register_pressure: usize,
    /// Hardware register limit.
    pub register_limit: usize,
    /// Estimated L1 working set in bytes.
    pub l1_working_set_bytes: usize,
    /// L1 budget (85% of L1D) in bytes.
    pub l1_budget_bytes: usize,
    /// Number of epilogue ops in this group.
    pub epilogue_depth: usize,
    /// Maximum allowed epilogue depth.
    pub max_epilogue_depth: usize,
    /// List of constraint violations (empty if valid).
    pub violations: Vec<ConstraintViolation>,
}

/// A specific hardware constraint that was violated.
#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    /// Register pressure exceeds available SIMD registers.
    RegisterPressure { needed: usize, available: usize },
    /// L1 working set exceeds the L1 budget.
    L1Overflow { working_set: usize, l1_size: usize },
    /// Epilogue chain is too deep for register-resident execution.
    EpilogueTooDeep { depth: usize, max: usize },
}

/// Stateless hardware constraint checker bound to a device profile.
///
/// Validates that fusion groups respect register file size, L1 capacity,
/// and epilogue depth limits for the target microarchitecture.
pub struct HwConstraintChecker<'a> {
    pub profile: &'a DeviceProfile,
}

impl<'a> HwConstraintChecker<'a> {
    pub fn new(profile: &'a DeviceProfile) -> Self {
        HwConstraintChecker { profile }
    }

    /// Validate that the fused group's register pressure fits in the SIMD
    /// register file.
    ///
    /// Budget: MR × ceil(NR / simd_w) accumulators + ceil(MR / simd_w)
    /// A-panel regs + 1 B broadcast reg + epilogue scratch (1 per fused
    /// epilogue op, capped at 4).
    pub fn validate_register_pressure(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
    ) -> Result<(), ConstraintViolation> {
        let total = estimate_register_pressure(group, graph, self.profile);
        let available = self.profile.num_simd_regs();

        if total > available {
            return Err(ConstraintViolation::RegisterPressure {
                needed: total,
                available,
            });
        }
        Ok(())
    }

    /// Validate that the epilogue chain is not too deep.
    ///
    /// Deep epilogues increase register pressure and instruction-cache
    /// footprint inside the microkernel inner loop. Cap at 16 fused ops.
    pub fn validate_epilogue_depth(
        &self,
        group: &FusionGroup,
    ) -> Result<(), ConstraintViolation> {
        let max_depth: usize = 16;
        let depth = group.epilogue.len();
        if depth > max_depth {
            return Err(ConstraintViolation::EpilogueTooDeep {
                depth,
                max: max_depth,
            });
        }
        Ok(())
    }

    /// Validate that the GEMM micropanels fit in L1 (85% budget).
    ///
    /// Working set = packed_A micropanel (MR × KC × 4) + packed_B micropanel
    /// (KC × NR × 4). Non-GEMM groups are always valid here.
    pub fn validate_l1_working_set(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
    ) -> Result<(), ConstraintViolation> {
        let (m, n, k) = match extract_gemm_dims(group, graph) {
            Some(dims) => dims,
            None => return Ok(()), // non-GEMM groups always pass
        };

        let (mr, nr) = self.profile.microkernel_mr_nr();
        let blocking = self.profile.gemm_blocking(m, n, k);
        let kc = blocking.kc;
        let working_set = (mr * kc + kc * nr) * 4; // packed_a + packed_b micropanels
        let (l1, _, _) = self.profile.cache_sizes();

        if working_set > l1 * 85 / 100 {
            return Err(ConstraintViolation::L1Overflow {
                working_set,
                l1_size: l1,
            });
        }
        Ok(())
    }

    /// Run all constraint checks on a fusion group.
    ///
    /// Returns `Ok(true)` if all constraints pass, or the first violation.
    pub fn validate_group(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
    ) -> Result<bool, ConstraintViolation> {
        self.validate_register_pressure(group, graph)?;
        self.validate_epilogue_depth(group)?;
        self.validate_l1_working_set(group, graph)?;
        Ok(true)
    }

    /// Validate all groups in a fusion plan, collecting full results.
    pub fn validate_plan(
        &self,
        groups: &[FusionGroup],
        graph: &CompilerGraph,
    ) -> Vec<HwConstraintResult> {
        groups
            .iter()
            .map(|g| self.build_result(g, graph))
            .collect()
    }

    /// Build a detailed `HwConstraintResult` for a single group.
    fn build_result(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
    ) -> HwConstraintResult {
        let register_limit = self.profile.num_simd_regs();
        let (l1, _, _) = self.profile.cache_sizes();
        let l1_budget = l1 * 85 / 100;
        let max_epilogue_depth: usize = 16;

        let register_pressure = estimate_register_pressure(group, graph, self.profile);
        let l1_working_set = estimate_l1_working_set(group, graph, self.profile);
        let epilogue_depth = group.epilogue.len();

        let mut violations = Vec::new();

        if register_pressure > register_limit {
            violations.push(ConstraintViolation::RegisterPressure {
                needed: register_pressure,
                available: register_limit,
            });
        }

        if l1_working_set > l1_budget {
            violations.push(ConstraintViolation::L1Overflow {
                working_set: l1_working_set,
                l1_size: l1,
            });
        }

        if epilogue_depth > max_epilogue_depth {
            violations.push(ConstraintViolation::EpilogueTooDeep {
                depth: epilogue_depth,
                max: max_epilogue_depth,
            });
        }

        HwConstraintResult {
            group_id: group.id,
            valid: violations.is_empty(),
            register_pressure,
            register_limit,
            l1_working_set_bytes: l1_working_set,
            l1_budget_bytes: l1_budget,
            epilogue_depth,
            max_epilogue_depth,
            violations,
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy free-function API (delegates to HwConstraintChecker)
// ---------------------------------------------------------------------------

/// Check hardware constraints for a single fusion group.
pub fn check_group(
    group: &FusionGroup,
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> HwConstraintResult {
    HwConstraintChecker::new(profile).build_result(group, graph)
}

/// Check all groups in a fusion plan.
pub fn check_plan(
    groups: &[FusionGroup],
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> Vec<HwConstraintResult> {
    HwConstraintChecker::new(profile).validate_plan(groups, graph)
}

// ---------------------------------------------------------------------------
// Internal estimation helpers
// ---------------------------------------------------------------------------

/// Extract GEMM (m, n, k) from the anchor op, if it is a GEMM variant.
fn extract_gemm_dims(group: &FusionGroup, graph: &CompilerGraph) -> Option<(usize, usize, usize)> {
    let op = graph.op(group.anchor)?;
    match &op.kind {
        OpKind::Gemm { m, n, k }
        | OpKind::GemmBias { m, n, k }
        | OpKind::QuantGemm { m, n, k, .. } => Some((*m, *n, *k)),
        _ => None,
    }
}

/// Estimate register pressure for a fusion group.
///
/// GEMM microkernel register budget:
///   - MR × ceil(NR / simd_w) accumulator registers
///   - ceil(MR / simd_w) A-panel broadcast registers
///   - 1 B-panel broadcast register
///   - For epilogue fusion: 1 scratch register per fused op (capped at 4)
///
/// The A-panel registers double as pack scratch, so no extra allocation
/// is needed for packing.
fn estimate_register_pressure(
    group: &FusionGroup,
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> usize {
    match group.pattern {
        FusionPattern::Standalone => {
            if let Some(op) = graph.op(group.anchor) {
                match &op.kind {
                    OpKind::Gemm { .. } | OpKind::GemmBias { .. } | OpKind::QuantGemm { .. } => {
                        gemm_base_regs(profile)
                    }
                    _ => 2, // input + output
                }
            } else {
                2
            }
        }
        FusionPattern::GemmEpilogue => {
            let epilogue_scratch = group.epilogue.len().min(4);
            gemm_base_regs(profile) + epilogue_scratch
        }
        FusionPattern::ElementwiseChain => 3 + group.ops.len(),
        FusionPattern::QkvSharedInput => gemm_base_regs(profile),
        FusionPattern::NormIntoGemm => gemm_base_regs(profile) + 2,
    }
}

/// Base register count for a GEMM microkernel:
///   MR × ceil(NR / simd_w) accumulators
/// + ceil(MR / simd_w) A-panel registers
/// + 1 B broadcast register
fn gemm_base_regs(profile: &DeviceProfile) -> usize {
    let (mr, nr) = profile.microkernel_mr_nr();
    let simd_w = profile.simd_width_f32();
    let acc_regs = mr * (nr / simd_w).max(1);
    let a_regs = (mr / simd_w).max(1);
    acc_regs + a_regs + 1
}

/// Estimate L1 working set for a fusion group.
///
/// For GEMM groups: A micropanel (MR × KC × 4) + B micropanel (KC × NR × 4).
/// For elementwise: input tile + output tile sized to fit in L1.
fn estimate_l1_working_set(
    group: &FusionGroup,
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> usize {
    if let Some((m, n, k)) = extract_gemm_dims(group, graph) {
        let (mr, nr) = profile.microkernel_mr_nr();
        let blocking = profile.gemm_blocking(m, n, k);
        let a_panel = mr * blocking.kc * 4;
        let b_panel = blocking.kc * nr * 4;
        a_panel + b_panel
    } else {
        let tile = profile.elem_tile_size();
        tile * 4 * 2 // input + output, f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::fusion;
    use crate::compiler::graph::{CompilerGraph, OpId};
    use crate::compiler::ir::LayerIR;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::{DType, ModelConfig};

    // -----------------------------------------------------------------------
    // Legacy API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_standalone_gemm() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let out = g.add_tensor("out", vec![32, 4096], dt);
        g.add_op(
            OpKind::Gemm { m: 32, n: 4096, k: 4096 },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let plan = fusion::fuse(&g);
        let profile = DeviceProfile::detect();
        let results = check_plan(&plan.groups, &g, &profile);

        assert_eq!(results.len(), 1);
        assert!(
            results[0].valid,
            "Standalone GEMM should pass constraints, violations: {:?}",
            results[0].violations
        );
    }

    #[test]
    fn test_check_gemm_epilogue() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![32, 4096], dt);
        let silu_out = g.add_tensor("silu_out", vec![32, 4096], dt);

        g.add_op(
            OpKind::Gemm { m: 32, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let plan = fusion::fuse(&g);
        let profile = DeviceProfile::detect();
        let results = check_plan(&plan.groups, &g, &profile);

        assert_eq!(results.len(), 1);
        assert!(
            results[0].valid,
            "GEMM+SiLU epilogue should pass constraints, violations: {:?}",
            results[0].violations
        );
    }

    #[test]
    fn test_check_llama_plan() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&graph, &registry);

        let results = check_plan(&plan.groups, &graph, &profile);

        for r in &results {
            assert!(
                r.valid,
                "Group {} ({:?}) failed constraints: {:?}",
                r.group_id,
                plan.groups[r.group_id].pattern,
                r.violations
            );
        }

        eprintln!(
            "LLaMA plan: {} groups, all passed hw constraints",
            results.len()
        );
    }

    #[test]
    fn test_register_pressure_increases_with_epilogue() {
        let profile = DeviceProfile::detect();

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            pattern: FusionPattern::Standalone,
            ops: vec![OpId(0)],
        };

        let with_epilogue = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2)],
            pattern: FusionPattern::GemmEpilogue,
            ops: vec![OpId(0), OpId(1), OpId(2)],
        };

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let out = g.add_tensor("out", vec![32, 4096], dt);
        g.add_op(
            OpKind::Gemm { m: 32, n: 4096, k: 4096 },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let p_standalone = estimate_register_pressure(&standalone, &g, &profile);
        let p_epilogue = estimate_register_pressure(&with_epilogue, &g, &profile);

        assert!(
            p_epilogue > p_standalone,
            "Epilogue should increase register pressure: standalone={}, epilogue={}",
            p_standalone,
            p_epilogue
        );
    }

    // -----------------------------------------------------------------------
    // HwConstraintChecker tests
    // -----------------------------------------------------------------------

    /// Simple GEMM + single epilogue (like SiLU) should pass validation.
    #[test]
    fn test_checker_gemm_plus_relu_passes() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let gemm_out = g.add_tensor("gemm_out", vec![32, 4096], dt);
        let silu_out = g.add_tensor("silu_out", vec![32, 4096], dt);

        g.add_op(
            OpKind::Gemm { m: 32, n: 4096, k: 4096 },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let plan = fusion::fuse(&g);
        let profile = DeviceProfile::detect();
        let checker = HwConstraintChecker::new(&profile);

        for group in &plan.groups {
            let result = checker.validate_group(group, &g);
            assert!(
                result.is_ok(),
                "GEMM+SiLU should pass all checks: {:?}",
                result.err()
            );
        }
    }

    /// An epilogue with 20 ops should fail the depth check (max 16).
    #[test]
    fn test_checker_deep_epilogue_fails() {
        let profile = DeviceProfile::detect();
        let checker = HwConstraintChecker::new(&profile);

        let epilogue_ops: Vec<OpId> = (1..=20).map(|i| OpId(i)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            pattern: FusionPattern::GemmEpilogue,
            ops: all_ops,
        };

        let result = checker.validate_epilogue_depth(&group);
        assert!(result.is_err(), "20-deep epilogue should fail");
        match result.unwrap_err() {
            ConstraintViolation::EpilogueTooDeep { depth, max } => {
                assert_eq!(depth, 20);
                assert_eq!(max, 16);
            }
            other => panic!("Expected EpilogueTooDeep, got {:?}", other),
        }
    }

    /// Epilogue at exactly the limit (16) should pass.
    #[test]
    fn test_checker_epilogue_at_limit_passes() {
        let profile = DeviceProfile::detect();
        let checker = HwConstraintChecker::new(&profile);

        let epilogue_ops: Vec<OpId> = (1..=16).map(|i| OpId(i)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            pattern: FusionPattern::GemmEpilogue,
            ops: all_ops,
        };

        assert!(checker.validate_epilogue_depth(&group).is_ok());
    }

    /// AVX2 (16 regs) has tighter register pressure than AVX-512 (32 regs).
    /// Verify that epilogue fusion increases pressure, and that the base
    /// GEMM microkernel fits within the register file on the detected ISA.
    #[test]
    fn test_checker_register_pressure_avx2_vs_avx512() {
        let profile = DeviceProfile::detect();
        let regs = profile.num_simd_regs();

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let out = g.add_tensor("out", vec![32, 4096], dt);
        g.add_op(
            OpKind::Gemm { m: 32, n: 4096, k: 4096 },
            vec![a, w],
            vec![out],
            "gemm",
        );

        // Standalone GEMM — must always fit
        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            pattern: FusionPattern::Standalone,
            ops: vec![OpId(0)],
        };
        let base_pressure = estimate_register_pressure(&standalone, &g, &profile);
        assert!(
            base_pressure <= regs,
            "Base GEMM pressure {} exceeds {} regs",
            base_pressure,
            regs
        );

        // GEMM + 3 epilogue ops — pressure should increase
        let with_epilogue = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2), OpId(3)],
            pattern: FusionPattern::GemmEpilogue,
            ops: vec![OpId(0), OpId(1), OpId(2), OpId(3)],
        };
        let epi_pressure = estimate_register_pressure(&with_epilogue, &g, &profile);
        assert!(
            epi_pressure > base_pressure,
            "Epilogue should increase pressure: base={}, epi={}",
            base_pressure,
            epi_pressure
        );

        // On AVX2 (16 regs) the headroom is tighter than AVX-512 (32 regs)
        let headroom = regs.saturating_sub(epi_pressure);
        eprintln!(
            "ISA regs={}, base={}, epi={}, headroom={}",
            regs, base_pressure, epi_pressure, headroom
        );
        // AVX2: 16 - ~15 = ~1 headroom; AVX-512: 32 - ~31 = ~1 headroom
        // (both are tight because MR*NR/simd_w scales with the register file)
    }

    /// validate_group collects all checks — a valid standalone GEMM should pass.
    #[test]
    fn test_checker_validate_group_standalone_gemm() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let w = g.add_tensor("w", vec![4096, 4096], dt);
        let out = g.add_tensor("out", vec![32, 4096], dt);
        g.add_op(
            OpKind::Gemm { m: 32, n: 4096, k: 4096 },
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            pattern: FusionPattern::Standalone,
            ops: vec![OpId(0)],
        };

        let profile = DeviceProfile::detect();
        let checker = HwConstraintChecker::new(&profile);
        assert!(checker.validate_group(&group, &g).is_ok());
    }

    /// validate_plan returns one result per group.
    #[test]
    fn test_checker_validate_plan() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&graph, &registry);

        let checker = HwConstraintChecker::new(&profile);
        let results = checker.validate_plan(&plan.groups, &graph);

        assert_eq!(results.len(), plan.groups.len());
        for r in &results {
            assert!(
                r.valid,
                "Group {} failed: {:?}",
                r.group_id,
                r.violations
            );
        }
    }

    /// L1 working set for a non-GEMM group always passes.
    #[test]
    fn test_checker_l1_non_gemm_passes() {
        let profile = DeviceProfile::detect();
        let checker = HwConstraintChecker::new(&profile);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![32, 4096], dt);
        let out = g.add_tensor("out", vec![32, 4096], dt);
        g.add_op(OpKind::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            pattern: FusionPattern::Standalone,
            ops: vec![OpId(0)],
        };

        assert!(checker.validate_l1_working_set(&group, &g).is_ok());
    }
}
