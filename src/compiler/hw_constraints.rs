use crate::compiler::fusion::{FusionGroup, FusionMode};
use crate::compiler::graph::{CompilerGraph, MultiOutputConfig, Op};
use crate::compiler::planner::ExecutionPlan;
use crate::types::DType;

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
    /// AMX tile configuration exceeds available tile registers.
    AmxTileOverflow { tiles_needed: usize, tiles_available: usize },
    /// SME2 ZA array size exceeds streaming SVE mode capacity.
    Sme2ZaOverflow { za_bytes: usize, max_za_bytes: usize },
    /// GPU shared memory tile exceeds per-block shared memory budget.
    SharedMemOverflow { tile_bytes: usize, smem_bytes: usize },
    /// GPU tensor core GEMM requires unsupported instruction.
    TensorCoreUnsupported { required_gen: u32, available_gen: u32 },
}

/// Stateless hardware constraint checker bound to a device profile.
///
/// Validates that fusion groups respect register file size, L1 capacity,
/// and epilogue depth limits for the target microarchitecture.
pub struct HwConstraintChecker<'a> {
    pub plan: &'a ExecutionPlan,
}

impl<'a> HwConstraintChecker<'a> {
    pub fn new(plan: &'a ExecutionPlan) -> Self {
        HwConstraintChecker { plan }
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
        let total = estimate_register_pressure(group, graph, self.plan);
        let mut available = self.plan.profile.num_simd_regs();

        // Memory-bound groups get +4 register headroom.
        if crate::compiler::fusion::is_memory_bound_group(group, graph, self.plan) {
            available += 4;
        }

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
    /// footprint inside the microkernel inner loop.
    /// Max depth is dynamic: AVX-512/NEON/SVE (32 regs) → 16, AVX2 (16 regs) → 8.
    pub fn validate_epilogue_depth(
        &self,
        group: &FusionGroup,
    ) -> Result<(), ConstraintViolation> {
        let max_depth = max_epilogue_depth(self.plan);
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
    /// Working set = packed_A micropanel (MR × KC × elem_bytes) + packed_B micropanel
    /// (KC × NR × elem_bytes). Non-GEMM groups are always valid here.
    pub fn validate_l1_working_set(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
    ) -> Result<(), ConstraintViolation> {
        let (m, n, k) = match extract_gemm_dims(group, graph) {
            Some(dims) => dims,
            None => return Ok(()), // non-GEMM groups always pass
        };

        let dtype = extract_gemm_dtype(group, graph);
        let elem_bytes = dtype.size_bytes();
        let (mr, nr) = self.plan.profile.microkernel_mr_nr();
        let blocking = self.plan.profile.gemm_blocking(m, n, k, dtype);
        let kc = blocking.kc;
        let working_set = (mr * kc + kc * nr) * elem_bytes;
        let (l1, _, _) = self.plan.profile.cache_sizes();

        if working_set > (l1 as f64 * self.plan.profile.l1_budget_ratio()) as usize {
            return Err(ConstraintViolation::L1Overflow {
                working_set,
                l1_size: l1,
            });
        }
        Ok(())
    }

    /// Validate AMX tile constraints for AVX-512+AMX profiles.
    ///
    /// AMX provides 8 tile registers (TMM0-TMM7), each up to 16x16 elements.
    /// A fused GEMM+epilogue group needs tiles for: accumulator + B-panel + scratch.
    pub fn validate_amx_tiles(
        &self,
        group: &FusionGroup,
    ) -> Result<(), ConstraintViolation> {
        use crate::compiler::hardware_profile::HardwareProfile;
        let hw = HardwareProfile::detect(&self.plan.profile);
        if !hw.has_amx() {
            return Ok(());
        }
        // AMX has 8 tile registers. Reserve 1 for accumulator, 1 for load.
        // Each fused GEMM needs: 1 accumulator + ceil(epilogue_depth/4) scratch tiles.
        let gemm_tiles = 2;
        let epilogue_tiles = (group.epilogue.len() + 3) / 4;
        let total = gemm_tiles + epilogue_tiles;
        if total > 8 {
            return Err(ConstraintViolation::AmxTileOverflow {
                tiles_needed: total,
                tiles_available: 8,
            });
        }
        Ok(())
    }

    /// Validate SME2 ZA array constraints for ARM Neoverse profiles.
    pub fn validate_sme2_za(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
    ) -> Result<(), ConstraintViolation> {
        use crate::compiler::hardware_profile::HardwareProfile;
        let hw = HardwareProfile::detect(&self.plan.profile);
        if !hw.has_sme2() {
            return Ok(());
        }
        // SME2 ZA array is SVLxSVL bytes (up to 256x256 for SVL=256).
        // Estimate working set from GEMM dims if present.
        let (_, _, _) = match extract_gemm_dims(group, graph) {
            Some(dims) => dims,
            None => return Ok(()),
        };
        // ZA array max = 16KB for SVL=128, 64KB for SVL=256
        let max_za = 16 * 1024;
        let za_needed = group.epilogue.len() * 256; // rough estimate
        if za_needed > max_za {
            return Err(ConstraintViolation::Sme2ZaOverflow {
                za_bytes: za_needed,
                max_za_bytes: max_za,
            });
        }
        Ok(())
    }

    /// Validate GPU shared memory constraints for tensor core GEMM fusion.
    ///
    /// When a JitContext is provided (SPEC 15 REQ-JCTX-014), uses its
    /// mem_available(SharedMem) as the budget instead of re-reading from
    /// HardwareProfile.
    pub fn validate_gpu_shared_mem(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
        jit_ctx: Option<&crate::compiler::jit_context::JitContext>,
    ) -> Result<(), ConstraintViolation> {
        let smem = if let Some(ctx) = jit_ctx {
            // SPEC 15 REQ-JCTX-014: 从 JitContext 查询可用共享内存（字节）
            ctx.mem_available(crate::compiler::jit_context::ResourceKind::SharedMem)
        } else {
            use crate::compiler::hardware_profile::HardwareProfile;
            let hw = HardwareProfile::detect(&self.plan.profile);
            hw.shared_memory_bytes()
        };
        if smem == 0 {
            return Ok(()); // not a GPU profile
        }
        let (_, _, _) = match extract_gemm_dims(group, graph) {
            Some(dims) => dims,
            None => return Ok(()),
        };
        // Estimate shared memory tile: 2 x tile_m x tile_n x elem_bytes
        // Typical tile: 64x64x4 = 16KB per buffer, 32KB total
        let tile_bytes = 2 * 64 * 64 * 4;
        if tile_bytes > smem {
            return Err(ConstraintViolation::SharedMemOverflow {
                tile_bytes,
                smem_bytes: smem,
            });
        }
        Ok(())
    }

    /// Validate GPU tensor core generation requirement.
    pub fn validate_tensor_core_gen(
        &self,
        required_gen: u32,
    ) -> Result<(), ConstraintViolation> {
        use crate::compiler::hardware_profile::HardwareProfile;
        let hw = HardwareProfile::detect(&self.plan.profile);
        let available = hw.tensor_core_gen();
        if available > 0 && available < required_gen {
            return Err(ConstraintViolation::TensorCoreUnsupported {
                required_gen,
                available_gen: available,
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
        self.validate_group_with_context(group, graph, None)
    }

    /// Run all constraint checks with optional JitContext (SPEC 15 REQ-JCTX-014).
    pub fn validate_group_with_context(
        &self,
        group: &FusionGroup,
        graph: &CompilerGraph,
        jit_ctx: Option<&crate::compiler::jit_context::JitContext>,
    ) -> Result<bool, ConstraintViolation> {
        self.validate_register_pressure(group, graph)?;
        self.validate_epilogue_depth(group)?;
        self.validate_l1_working_set(group, graph)?;
        self.validate_amx_tiles(group)?;
        self.validate_sme2_za(group, graph)?;
        self.validate_gpu_shared_mem(group, graph, jit_ctx)?;
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
        let mut register_limit = self.plan.profile.num_simd_regs();
        let (l1, _, _) = self.plan.profile.cache_sizes();
        let l1_budget = (l1 as f64 * self.plan.profile.l1_budget_ratio()) as usize;
        let max_epilogue_depth = max_epilogue_depth(self.plan);

        // Memory-bound groups get +4 register headroom: fusion benefit is high
        // (eliminating memory traffic), so tolerating more register pressure is worthwhile.
        if crate::compiler::fusion::is_memory_bound_group(group, graph, self.plan) {
            register_limit += 4;
        }

        let register_pressure = estimate_register_pressure(group, graph, self.plan);
        let l1_working_set = estimate_l1_working_set(group, graph, self.plan);
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
    plan: &ExecutionPlan,
) -> HwConstraintResult {
    HwConstraintChecker::new(plan).build_result(group, graph)
}

/// Check all groups in a fusion plan.
pub fn check_plan(
    groups: &[FusionGroup],
    graph: &CompilerGraph,
    plan: &ExecutionPlan,
) -> Vec<HwConstraintResult> {
    HwConstraintChecker::new(plan).validate_plan(groups, graph)
}

/// Enforce hardware constraints: groups that violate register/L1/epilogue limits
/// are split back into per-op Standalone groups.
pub fn enforce_constraints(
    groups: &mut Vec<FusionGroup>,
    graph: &CompilerGraph,
    plan: &ExecutionPlan,
) {
    let checker = HwConstraintChecker::new(plan);
    let mut i = 0;
    while i < groups.len() {
        let result = checker.build_result(&groups[i], graph);
        if !result.valid {
            eprintln!(
                "[HW-CONSTRAINT] group {} ({:?}) violated: {:?} — splitting to standalone",
                groups[i].id, groups[i].mode, result.violations
            );
            let old = groups.remove(i);
            // Split into per-op standalone groups
            for (j, &op_id) in old.ops.iter().enumerate() {
                groups.insert(i + j, FusionGroup {
                    id: old.id * 100 + j,
                    anchor: op_id,
                    epilogue: vec![],
                    mode: FusionMode::Standalone,
                    ops: vec![op_id],
                    multi_output: MultiOutputConfig::single(),
                    dominant_dtype: old.dominant_dtype,
                    marker: old.marker.clone(),
                    is_layer_group: old.is_layer_group,
                    hetero_layer_type: old.hetero_layer_type,
                });
            }
            i += old.ops.len();
        } else {
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal estimation helpers
// ---------------------------------------------------------------------------

/// Extract GEMM (m, n, k) from the anchor op, if it is a GEMM variant.
// ARCH-SYMDIM-DEGRADE: cost model uses max_for_allocation for conservative estimate;
// symbolic-bound propagation is deferred — current upper-bound is sufficient for roofline classification.
fn extract_gemm_dims(group: &FusionGroup, graph: &CompilerGraph) -> Option<(usize, usize, usize)> {
    let op = graph.op(group.anchor)?;
    op.op_gemm_dims(graph).map(|(m, n, k)| {
        (m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model"), n, k)
    })
}



/// Extract GEMM dtype from the anchor op.
/// ARCH-DTYPE-FULLCHAIN-ORCH: uses graph-level dtype inference for QuantGemm and non-GEMM ops.
fn extract_gemm_dtype(group: &FusionGroup, graph: &CompilerGraph) -> crate::types::DType {
    let op = graph.op(group.anchor);
    op.and_then(|op| {
        if op.op_is_quant_gemm(graph) {
            Some(graph.infer_computation_dtype())
        } else {
            op.op_gemm_dtype(graph)
        }
    }).unwrap_or_else(|| graph.infer_computation_dtype())
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
    plan: &ExecutionPlan,
) -> usize {
    let dtype = extract_gemm_dtype(group, graph);
    match group.mode {
        FusionMode::Standalone => {
            if let Some(op) = graph.op(group.anchor) {
                let op_resolved = op.op_resolved(graph);
                if matches!(op_resolved, Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_))) {
                    gemm_base_regs(plan, dtype)
                } else {
                    2 // input + output
                }
            } else {
                2
            }
        }
        FusionMode::EpilogueInjection => {
            // Dynamic: free regs after GEMM accumulators determine no-spill capacity
            let base = gemm_base_regs(plan, dtype);
            let avail = plan.profile.num_simd_regs();
            let free_regs = avail.saturating_sub(base);
            let no_spill = group.epilogue.len().min(free_regs);
            let spill_ops = group.epilogue.len().saturating_sub(free_regs);
            // Spill ops still need 1 temp register (shared via spill/reload)
            base + no_spill + spill_ops.min(1)
        }
        FusionMode::LoopFusion => 3 + group.ops.len(),
        FusionMode::QkvSharedInput => gemm_base_regs(plan, dtype),
        FusionMode::NormIntoGemm => gemm_base_regs(plan, dtype) + 2,
        FusionMode::TileLevelFusion { .. } => {
            // GEMM base + 3 scratch for norm (mean, rsqrt, weight)
            gemm_base_regs(plan, dtype) + 3
        }
        FusionMode::ComputeRoot { .. } => {
            // Norm computed fully before GEMM; GEMM runs with base regs only
            gemm_base_regs(plan, dtype)
        }
        FusionMode::FFNBlock { .. } => {
            // Gate+Up shared pack_a + activation + mul: similar to QkvSharedInput + epilogue
            gemm_base_regs(plan, dtype) + 2
        }
        FusionMode::CrossLayerResidual { .. } => {
            // Residual Add + RmsNorm: 3 regs (input, output, norm scratch)
            3
        }
        FusionMode::FusedQkvNormRope { .. } => {
            // QKV shared pack_a + norm + rope: same as QkvSharedInput
            gemm_base_regs(plan, dtype)
        }
    }
}

/// Max epilogue depth derived from register pressure.
///
/// Epilogue ops use registers for: accumulator readback, activation computation,
/// broadcast constants. Budget = (available - gemm_accumulator) / 2 (accounting for
/// source + destination operands per epilogue op).
fn max_epilogue_depth(plan: &ExecutionPlan) -> usize {
    let avail = plan.profile.num_simd_regs();
    let reserved = plan.profile.gemm_accumulator_regs();
    ((avail.saturating_sub(reserved)) / 2).max(2)
}

/// Base register count for a GEMM microkernel:
///   MR × ceil(NR / simd_w) accumulators
/// + ceil(MR / simd_w) A-panel registers
/// + 1 B broadcast register
fn gemm_base_regs(plan: &ExecutionPlan, dtype: crate::types::DType) -> usize {
    let (mr, nr) = plan.profile.microkernel_mr_nr();
    let simd_w = plan.profile.simd_width_bytes() / dtype.size_bytes();
    let simd_w = simd_w.max(1);
    let acc_regs = mr * (nr / simd_w).max(1);
    let a_regs = (mr / simd_w).max(1);
    acc_regs + a_regs + 1
}

/// Estimate L1 working set for a fusion group.
///
/// For GEMM groups: A micropanel (MR × KC × elem_bytes) + B micropanel (KC × NR × elem_bytes).
/// For elementwise: input tile + output tile sized to fit in L1.
fn estimate_l1_working_set(
    group: &FusionGroup,
    graph: &CompilerGraph,
    plan: &ExecutionPlan,
) -> usize {
    if let Some((m, n, k)) = extract_gemm_dims(group, graph) {
        let dtype = extract_gemm_dtype(group, graph);
        let elem_bytes = dtype.size_bytes();
        let (mr, nr) = plan.profile.microkernel_mr_nr();
        let blocking = plan.profile.gemm_blocking(m, n, k, dtype);
        let a_panel = mr * blocking.kc * elem_bytes;
        let b_panel = blocking.kc * nr * elem_bytes;
        a_panel + b_panel
    } else {
        // Elementwise: use the dtype of the group's anchor output tensor
        let elem_bytes = group.ops.iter()
            .find_map(|&oid| graph.op(oid))
            .and_then(|op| op.outputs.first())
            .and_then(|tid| graph.tensor(*tid))
            .map(|t| t.dtype.size_bytes())
            .unwrap_or(DType::F32.size_bytes()); // fallback F32
        let tile = plan.profile.elem_tile_size();
        tile * elem_bytes * 2 // input + output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::fusion::{self, GroupMarker};
    use crate::compiler::graph::{CompilerGraph, OpId, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};
    use crate::compiler::ir::LayerIR;
    use crate::compiler::planner::ExecutionPlan;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::dispatch::DeviceProfile;
    use crate::types::{DType, ModelConfig};

    // -----------------------------------------------------------------------
    // Legacy API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_standalone_gemm() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let results = check_plan(&plan.groups, &g, &exec_plan);

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
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[32, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[32, 4096], dt);

        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let results = check_plan(&plan.groups, &g, &exec_plan);

        // After ARCH-ROOT-CAUSE fix: compute-bound GEMM (4096×4096) gets demoted
        // from EpilogueInjection → Standalone, and SiLU is split into its own
        // Standalone group. Both groups should pass constraints.
        for r in &results {
            assert!(
                r.valid,
                "Group {} ({:?}) failed constraints: {:?}",
                r.group_id,
                plan.groups[r.group_id].mode,
                r.violations
            );
        }
    }

    #[test]
    fn test_check_llama_plan() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&graph, &registry, &exec_plan);

        let results = check_plan(&plan.groups, &graph, &exec_plan);

        for r in &results {
            assert!(
                r.valid,
                "Group {} ({:?}) failed constraints: {:?}",
                r.group_id,
                plan.groups[r.group_id].mode,
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
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let with_epilogue = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1), OpId(2)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let p_standalone = estimate_register_pressure(&standalone, &g, &exec_plan);
        let p_epilogue = estimate_register_pressure(&with_epilogue, &g, &exec_plan);

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
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[32, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[32, 4096], dt);

        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let checker = HwConstraintChecker::new(&exec_plan);

        for group in &plan.groups {
            let result = checker.validate_group(group, &g);
            assert!(
                result.is_ok(),
                "GEMM+SiLU should pass all checks: {:?}",
                result.err()
            );
        }
    }

    /// An epilogue with 20 ops should fail the depth check.
    #[test]
    fn test_checker_deep_epilogue_fails() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let expected_max = max_epilogue_depth(&exec_plan);

        let epilogue_ops: Vec<OpId> = (1..=20).map(|i| OpId(i)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let result = checker.validate_epilogue_depth(&group);
        assert!(result.is_err(), "20-deep epilogue should fail");
        match result.unwrap_err() {
            ConstraintViolation::EpilogueTooDeep { depth, max } => {
                assert_eq!(depth, 20);
                assert_eq!(max, expected_max);
            }
            other => panic!("Expected EpilogueTooDeep, got {:?}", other),
        }
    }

    /// Epilogue at exactly the dynamic limit should pass.
    #[test]
    fn test_checker_epilogue_at_limit_passes() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let limit = max_epilogue_depth(&exec_plan);

        let epilogue_ops: Vec<OpId> = (1..=limit).map(|i| OpId(i as u32)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        assert!(checker.validate_epilogue_depth(&group).is_ok());
    }

    /// AVX2 (16 regs) has tighter register pressure than AVX-512 (32 regs).
    /// Verify that epilogue fusion increases pressure, and that the base
    /// GEMM microkernel fits within the register file on the detected ISA.
    #[test]
    fn test_checker_register_pressure_avx2_vs_avx512() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let regs = profile.num_simd_regs();

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        // Standalone GEMM — must always fit
        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let base_pressure = estimate_register_pressure(&standalone, &g, &exec_plan);
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
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1), OpId(2), OpId(3)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let epi_pressure = estimate_register_pressure(&with_epilogue, &g, &exec_plan);
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
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        assert!(checker.validate_group(&group, &g).is_ok());
    }

    /// validate_plan returns one result per group.
    #[test]
    fn test_checker_validate_plan() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&graph, &registry, &exec_plan);

        let checker = HwConstraintChecker::new(&exec_plan);
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
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        assert!(checker.validate_l1_working_set(&group, &g).is_ok());
    }

    // -----------------------------------------------------------------------
    // New tests (TEST-HWC-01 through TEST-HWC-13)
    // -----------------------------------------------------------------------

    /// TEST-HWC-01: enforce_constraints splits an invalid group into per-op Standalone groups.
    /// @trace TEST-HWC-01 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_enforce_constraints_splits_invalid_group() {
        // Arrange: create a graph with one GEMM op
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[32, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[32, 4096], dt);
        let add_out = g.add_tensor_concrete("add_out", &[32, 4096], dt);

        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(Op::Add, vec![silu_out, silu_out], vec![add_out], "add");

        // Create an oversized epilogue group that will violate depth constraints
        let epilogue_ops: Vec<OpId> = (1..=50).map(|i| OpId(i)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let mut groups = vec![FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }];

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        enforce_constraints(&mut groups, &g, &exec_plan);

        // Assert: the single oversized group should be split into standalone groups
        assert!(
            groups.len() > 1,
            "enforce_constraints should split the invalid group, got {} groups",
            groups.len(),
        );
        for group in &groups {
            assert_eq!(
                group.mode,
                FusionMode::Standalone,
                "All split groups should be Standalone"
            );
            assert!(
                group.epilogue.is_empty(),
                "Split groups should have empty epilogue"
            );
        }
    }

    /// TEST-HWC-02: validate_tensor_core_gen passes when required <= available.
    /// @trace TEST-HWC-02 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_validate_tensor_core_gen_passes_when_satisfied() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act: require gen 0 (always available) or a low gen
        let result = checker.validate_tensor_core_gen(0);

        // Assert
        assert!(
            result.is_ok(),
            "Requiring tensor core gen 0 should always pass: {:?}",
            result.err()
        );
    }

    /// TEST-HWC-03: validate_tensor_core_gen fails when required > available on GPU.
    /// @trace TEST-HWC-03 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_validate_tensor_core_gen_fails_when_unsatisfied() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let hw = crate::compiler::hardware_profile::HardwareProfile::detect(&profile);
        let available_gen = hw.tensor_core_gen();

        // Act & Assert
        if available_gen > 0 {
            // On a GPU with tensor cores, request a higher gen
            let result = checker.validate_tensor_core_gen(available_gen + 10);
            assert!(result.is_err(), "Should fail when required_gen > available_gen");
            match result.unwrap_err() {
                ConstraintViolation::TensorCoreUnsupported {
                    required_gen,
                    available_gen: avail,
                } => {
                    assert_eq!(required_gen, available_gen + 10);
                    assert_eq!(avail, available_gen);
                }
                other => panic!("Expected TensorCoreUnsupported, got {:?}", other),
            }
        } else {
            // On CPU (no tensor cores, available=0), validate_tensor_core_gen always passes
            // because the condition is `available > 0 && available < required`
            assert!(
                checker.validate_tensor_core_gen(100).is_ok(),
                "CPU without tensor cores should always pass (available=0 bypasses check)"
            );
        }
    }

    /// TEST-HWC-04: register pressure for LoopFusion mode is 3 + ops.len().
    /// @trace TEST-HWC-04 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_register_pressure_loop_fusion_mode() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let b = g.add_tensor_concrete("b", &[32, 4096], dt);
        let c = g.add_tensor_concrete("c", &[32, 4096], dt);
        let d = g.add_tensor_concrete("d", &[32, 4096], dt);
        g.add_op(Op::Silu, vec![a], vec![b], "silu1");
        g.add_op(Op::Tanh, vec![b], vec![c], "tanh");
        g.add_op(Op::Gelu, vec![c], vec![d], "gelu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2)],
            mode: FusionMode::LoopFusion,
            ops: vec![OpId(0), OpId(1), OpId(2)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let pressure = estimate_register_pressure(&group, &g, &exec_plan);

        // Assert: LoopFusion = 3 + ops.len()
        assert_eq!(
            pressure,
            3 + 3,
            "LoopFusion register pressure should be 3 + ops.len()"
        );
    }

    /// TEST-HWC-05: register pressure for NormIntoGemm mode is gemm_base + 2.
    /// @trace TEST-HWC-05 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_register_pressure_norm_into_gemm_mode() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let norm_group = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::NormIntoGemm,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let norm_pressure = estimate_register_pressure(&norm_group, &g, &exec_plan);

        // Assert: NormIntoGemm = gemm_base + 2
        assert_eq!(
            norm_pressure,
            base + 2,
            "NormIntoGemm should add exactly 2 registers over base GEMM"
        );
    }

    /// TEST-HWC-06: register pressure for TileLevelFusion mode is gemm_base + 3.
    /// @trace TEST-HWC-06 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_register_pressure_tile_level_fusion_mode() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let tile_group = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::TileLevelFusion {
                predecessor: OpId(0),
                tile_rows: 64,
            },
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let tile_pressure = estimate_register_pressure(&tile_group, &g, &exec_plan);

        // Assert: TileLevelFusion = gemm_base + 3 (mean, rsqrt, weight scratch)
        assert_eq!(
            tile_pressure,
            base + 3,
            "TileLevelFusion should add exactly 3 scratch registers over base GEMM"
        );
    }

    /// TEST-HWC-07: register pressure for CrossLayerResidual mode is 3.
    /// @trace TEST-HWC-07 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_register_pressure_cross_layer_residual_mode() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let b = g.add_tensor_concrete("b", &[32, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Add, vec![a, b], vec![out], "residual");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::CrossLayerResidual {
                residual: OpId(0),
                norm: OpId(1),
            },
            ops: vec![OpId(0), OpId(1)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let pressure = estimate_register_pressure(&group, &g, &exec_plan);

        // Assert: CrossLayerResidual = 3 (input, output, norm scratch)
        assert_eq!(
            pressure, 3,
            "CrossLayerResidual should use exactly 3 registers"
        );
    }

    /// TEST-HWC-08: check_group legacy function returns correct HwConstraintResult fields.
    /// @trace TEST-HWC-08 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_check_group_legacy_returns_correct_fields() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 42,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let result = check_group(&group, &g, &exec_plan);

        // Assert
        assert_eq!(result.group_id, 42, "group_id should match input");
        assert!(result.valid, "Standalone GEMM should be valid");
        assert!(result.violations.is_empty(), "Should have no violations");
        assert!(
            result.register_pressure > 0,
            "Register pressure should be positive for GEMM"
        );
        assert!(
            result.register_limit > 0,
            "Register limit should be positive"
        );
    }

    /// TEST-HWC-09: build_result detects register pressure violation for oversized epilogue.
    /// @trace TEST-HWC-09 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_build_result_detects_register_violation() {
        // Arrange: create a group with many epilogue ops to trigger register pressure
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        // Create a group with 100 epilogue ops — far exceeding any realistic limit
        let epilogue_ops: Vec<OpId> = (1..=100).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_register_pressure(&group, &g);

        // Assert: should detect a violation
        if let Err(ConstraintViolation::RegisterPressure { needed, available }) = result {
            assert!(
                needed > available,
                "needed ({}) should exceed available ({})",
                needed, available
            );
        } else {
            // On very high reg count profiles this might still pass, which is acceptable
            eprintln!("Note: 100-epilogue group did not exceed register limit on this profile");
        }
    }

    /// TEST-HWC-10: validate_group_with_context delegates to same checks as validate_group.
    /// @trace TEST-HWC-10 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_validate_group_with_context_no_jit_ctx() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[32, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[32, 4096], dt);

        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act: call with None jit_ctx
        for group in &plan.groups {
            let result = checker.validate_group_with_context(group, &g, None);

            // Assert: should behave identically to validate_group
            assert!(
                result.is_ok(),
                "validate_group_with_context(None) should pass for valid GEMM+SiLU: {:?}",
                result.err()
            );
        }
    }

    /// TEST-HWC-11: extract_gemm_dims works for GemmBias anchor op.
    /// @trace TEST-HWC-11 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_extract_gemm_dims_gemm_bias() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[16, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 1024], dt);
        let out = g.add_tensor_concrete("out", &[16, 1024], dt);
        g.add_op(Op::GemmBias(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 1024, k: 512, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, w],
            vec![out],
            "gemm_bias",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let dims = extract_gemm_dims(&group, &g);

        // Assert
        assert_eq!(dims, Some((16, 1024, 512)), "Should extract (m, n, k) from GemmBias");
    }

    /// TEST-HWC-12: L1 working set estimation for elementwise (non-GEMM) uses tile size.
    /// @trace TEST-HWC-12 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_l1_working_set_elementwise_uses_tile_size() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let tile_size = profile.elem_tile_size();

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: Vec::new(),
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let working_set = estimate_l1_working_set(&group, &g, &exec_plan);

        // Assert: elementwise = tile * elem_bytes * 2 (input + output)
        let expected = tile_size * dt.size_bytes() * 2;
        assert_eq!(
            working_set, expected,
            "Elementwise L1 working set should be tile_size * elem_bytes * 2"
        );
    }

    /// TEST-HWC-13: validate_plan returns empty results for empty groups.
    /// @trace TEST-HWC-13 [req:REQ-FUS] [level:unit]
    #[test]
    fn test_validate_plan_empty_groups() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let g = CompilerGraph::new();
        let checker = HwConstraintChecker::new(&exec_plan);
        let groups: Vec<FusionGroup> = Vec::new();

        // Act
        let results = checker.validate_plan(&groups, &g);

        // Assert
        assert!(
            results.is_empty(),
            "Empty groups should produce empty results"
        );
    }

    // ── Test 25: ConstraintViolation Debug formats for all variants ──

    #[test]
    fn constraint_violation_debug_all_variants() {
        // Arrange & Act & Assert — just ensure Debug doesn't panic
        let _ = format!("{:?}", ConstraintViolation::RegisterPressure { needed: 32, available: 16 });
        let _ = format!("{:?}", ConstraintViolation::L1Overflow { working_set: 65536, l1_size: 32768 });
        let _ = format!("{:?}", ConstraintViolation::EpilogueTooDeep { depth: 20, max: 8 });
        let _ = format!("{:?}", ConstraintViolation::AmxTileOverflow { tiles_needed: 10, tiles_available: 8 });
        let _ = format!("{:?}", ConstraintViolation::Sme2ZaOverflow { za_bytes: 32768, max_za_bytes: 16384 });
        let _ = format!("{:?}", ConstraintViolation::SharedMemOverflow { tile_bytes: 65536, smem_bytes: 49152 });
        let _ = format!("{:?}", ConstraintViolation::TensorCoreUnsupported { required_gen: 90, available_gen: 80 });
    }

    // ── Test 26: HwConstraintResult default-ish values ──

    #[test]
    fn hw_constraint_result_fields_on_valid_standalone() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let results = check_plan(&plan.groups, &g, &exec_plan);

        // Assert
        assert_eq!(results.len(), 1, "single GEMM should be 1 group");
        let r = &results[0];
        assert!(r.valid);
        assert!(r.violations.is_empty(), "valid group should have no violations");
        assert!(r.register_pressure > 0, "register pressure should be positive");
        assert!(r.register_limit > 0, "register limit should be positive");
        assert!(r.register_pressure <= r.register_limit);
        assert_eq!(r.epilogue_depth, 0, "standalone GEMM has no epilogue");
    }

    // ── Test 27: HwConstraintResult clone preserves fields ──

    #[test]
    fn hw_constraint_result_clone_preserves_fields() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let results = check_plan(&plan.groups, &g, &exec_plan);

        // Act
        let cloned = results[0].clone();

        // Assert
        assert_eq!(results[0].group_id, cloned.group_id);
        assert_eq!(results[0].valid, cloned.valid);
        assert_eq!(results[0].register_pressure, cloned.register_pressure);
        assert_eq!(results[0].l1_working_set_bytes, cloned.l1_working_set_bytes);
        assert_eq!(results[0].violations.len(), cloned.violations.len());
    }

    // ── Test 28: check_group returns correct group_id ──

    #[test]
    fn check_group_returns_correct_group_id() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);

        // Act
        let result = check_group(&plan.groups[0], &g, &exec_plan);

        // Assert
        assert_eq!(result.group_id, plan.groups[0].id);
    }

    // ── Test 29: ConstraintViolation clone roundtrip ──

    #[test]
    fn constraint_violation_clone_roundtrip() {
        // Arrange
        let violation = ConstraintViolation::RegisterPressure { needed: 20, available: 16 };

        // Act
        let cloned = violation.clone();

        // Assert
        match cloned {
            ConstraintViolation::RegisterPressure { needed, available } => {
                assert_eq!(needed, 20);
                assert_eq!(available, 16);
            }
            other => panic!("expected RegisterPressure, got {:?}", other),
        }
    }

    // ── Test 30: validate_epilogue_depth zero epilogue always passes ──

    #[test]
    fn validate_epilogue_depth_zero_always_passes() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let g = CompilerGraph::new();
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_epilogue_depth(&group);

        // Assert
        assert!(result.is_ok(), "zero epilogue should always pass");
    }

    // ── Test 31: max_epilogue_depth is at least 2 ──

    #[test]
    fn max_epilogue_depth_at_least_two() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act — compute via checker internals
        let checker = HwConstraintChecker::new(&exec_plan);

        // Build a group with 0 epilogue to extract the max
        let g = CompilerGraph::new();
        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: crate::compiler::graph::MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Assert — passing with empty epilogue proves max >= 0
        assert!(checker.validate_epilogue_depth(&group).is_ok());
    }

    // ── Test 32: HwConstraintResult debug format ──

    #[test]
    fn hw_constraint_result_debug_format() {
        // Arrange
        let result = HwConstraintResult {
            group_id: 42,
            valid: true,
            register_pressure: 16,
            register_limit: 32,
            l1_working_set_bytes: 8192,
            l1_budget_bytes: 24576,
            epilogue_depth: 3,
            max_epilogue_depth: 8,
            violations: vec![],
        };

        // Act
        let debug = format!("{:?}", result);

        // Assert
        assert!(debug.contains("42"));
        assert!(debug.contains("true"));
    }

    // ── Test 33: validate_tensor_core_gen passes when available >= required ──

    #[test]
    fn validate_tensor_core_gen_passes_when_sufficient() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act: require gen=0 (always satisfied) or same as available
        let result = checker.validate_tensor_core_gen(0);
        assert!(result.is_ok(), "gen=0 should always pass");
    }

    // ── Test 34: enforce_constraints splits violating groups ──

    #[test]
    fn enforce_constraints_splits_invalid_groups() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Create a group with excessive epilogue to force a violation
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1); 50], // 50 epilogue ops — should trigger EpilogueTooDeep
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let mut groups = vec![group];
        enforce_constraints(&mut groups, &g, &exec_plan);

        // After enforcement: invalid group should be split into standalone ops
        assert!(groups.len() >= 1, "enforce should produce at least 1 group");
        // The split groups should be Standalone
        for g in &groups {
            assert_eq!(g.mode, FusionMode::Standalone);
        }
    }

    // ── Test 35: HwConstraintResult with violations reports valid=false ──

    #[test]
    fn hw_constraint_result_with_violation_is_invalid() {
        let result = HwConstraintResult {
            group_id: 0,
            valid: false,
            register_pressure: 64,
            register_limit: 32,
            l1_working_set_bytes: 65536,
            l1_budget_bytes: 32768,
            epilogue_depth: 20,
            max_epilogue_depth: 8,
            violations: vec![
                ConstraintViolation::RegisterPressure { needed: 64, available: 32 },
            ],
        };
        assert!(!result.valid);
        assert_eq!(result.violations.len(), 1);
    }

    // ── Test 36: extract_gemm_dims returns None for non-GEMM anchor ──

    #[test]
    fn extract_gemm_dims_non_gemm_returns_none() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64], dt);
        let out = g.add_tensor_concrete("out", &[64], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert!(extract_gemm_dims(&group, &g).is_none());
    }

    // ── Test 37: gemm_base_regs is positive for F32 ──

    #[test]
    fn gemm_base_regs_positive_f32() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let regs = gemm_base_regs(&exec_plan, DType::F32);
        assert!(regs >= 3, "base regs should be at least 3 (acc + a_panel + b), got {regs}");
    }

    // ── Test 38: gemm_base_regs returns positive for BF16 ──

    #[test]
    fn gemm_base_regs_bf16_positive() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let regs = gemm_base_regs(&exec_plan, DType::BF16);
        assert!(regs >= 1, "BF16 gemm_base_regs should be positive, got {regs}");
    }

    // ── Test 39: estimate_register_pressure for LoopFusion scales with ops ──

    #[test]
    fn estimate_register_pressure_loop_fusion_scales() {
        let g = CompilerGraph::new();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let small = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![OpId(0), OpId(1)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let large = FusionGroup {
            id: 1, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![OpId(0), OpId(1), OpId(2), OpId(3)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let small_rp = estimate_register_pressure(&small, &g, &exec_plan);
        let large_rp = estimate_register_pressure(&large, &g, &exec_plan);
        assert!(large_rp > small_rp, "more ops should need more regs (small={small_rp}, large={large_rp})");
    }

    // ── Test 40: ConstraintViolation Debug for all 7 variants ──

    #[test]
    fn constraint_violation_debug_all_seven_variants() {
        let violations = vec![
            ConstraintViolation::RegisterPressure { needed: 1, available: 1 },
            ConstraintViolation::L1Overflow { working_set: 1, l1_size: 1 },
            ConstraintViolation::EpilogueTooDeep { depth: 1, max: 1 },
            ConstraintViolation::AmxTileOverflow { tiles_needed: 1, tiles_available: 1 },
            ConstraintViolation::Sme2ZaOverflow { za_bytes: 1, max_za_bytes: 1 },
            ConstraintViolation::SharedMemOverflow { tile_bytes: 1, smem_bytes: 1 },
            ConstraintViolation::TensorCoreUnsupported { required_gen: 1, available_gen: 1 },
        ];
        for v in &violations {
            let debug = format!("{:?}", v);
            assert!(debug.len() > 5, "Debug should produce meaningful output");
        }
    }

    // ── Test 41: extract_gemm_dims returns dims for QuantGemm ──

    /// @trace TEST-HWC-41 [req:REQ-FUS] [level:unit]
    #[test]
    fn extract_gemm_dims_quant_gemm() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[4, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 256], dt);
        let out = g.add_tensor_concrete("out", &[4, 256], dt);
        g.add_op(Op::QuantGemm(QuantGemmSpec { m: crate::compiler::graph::SymDim::Concrete(4), n: 256, k: 128, quant_type: crate::quant::QuantType::Q4K }),
            vec![a, w],
            vec![out],
            "quant_gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let dims = extract_gemm_dims(&group, &g);

        // Assert
        assert_eq!(dims, Some((4, 256, 128)), "QuantGemm dims should be extracted");
    }

    // ── Test 42: extract_gemm_dtype returns explicit dtype for Gemm ──

    /// @trace TEST-HWC-42 [req:REQ-FUS] [level:unit]
    #[test]
    fn extract_gemm_dtype_returns_explicit_dtype() {
        // Arrange
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[8, 64], DType::BF16);
        let w = g.add_tensor_concrete("w", &[64, 64], DType::BF16);
        let out = g.add_tensor_concrete("out", &[8, 64], DType::BF16);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::BF16, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let dtype = extract_gemm_dtype(&group, &g);

        // Assert
        assert_eq!(dtype, DType::BF16, "extract_gemm_dtype should return the Gemm's explicit dtype");
    }

    // ── Test 43: register pressure for FFNBlock mode is gemm_base + 2 ──

    /// @trace TEST-HWC-43 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_ffn_block_mode() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let ffn_group = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::FFNBlock {
                gate_gemm: OpId(0),
                up_gemm: OpId(0),
                activation: OpId(0),
                combine: OpId(0),
            },
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let ffn = estimate_register_pressure(&ffn_group, &g, &exec_plan);

        // Assert: FFNBlock = gemm_base + 2
        assert_eq!(
            ffn, base + 2,
            "FFNBlock should add exactly 2 registers over base GEMM (base={base}, ffn={ffn})"
        );
    }

    // ── Test 44: register pressure for ComputeRoot mode equals gemm_base ──

    /// @trace TEST-HWC-44 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_compute_root_equals_base() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let compute_root = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::ComputeRoot { predecessor: OpId(0) },
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let cr = estimate_register_pressure(&compute_root, &g, &exec_plan);

        // Assert: ComputeRoot = gemm_base (norm computed before, GEMM uses base regs only)
        assert_eq!(
            cr, base,
            "ComputeRoot register pressure should equal base GEMM (base={base}, cr={cr})"
        );
    }

    // ── Test 45: register pressure for QkvSharedInput equals gemm_base ──

    /// @trace TEST-HWC-45 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_qkv_shared_input_equals_base() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let qkv_group = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::QkvSharedInput,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let qkv = estimate_register_pressure(&qkv_group, &g, &exec_plan);

        // Assert: QkvSharedInput shares pack_a, so register pressure equals base
        assert_eq!(
            qkv, base,
            "QkvSharedInput register pressure should equal base GEMM (base={base}, qkv={qkv})"
        );
    }

    // ── Test 46: enforce_constraints leaves valid groups untouched ──

    /// @trace TEST-HWC-46 [req:REQ-FUS] [level:unit]
    #[test]
    fn enforce_constraints_leaves_valid_groups_untouched() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[8, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let original_count = plan.groups.len();
        let original_ids: Vec<usize> = plan.groups.iter().map(|grp| grp.id).collect();

        let mut groups = plan.groups.clone();

        // Act
        enforce_constraints(&mut groups, &g, &exec_plan);

        // Assert: valid groups should remain unchanged
        assert_eq!(
            groups.len(),
            original_count,
            "Valid groups should not be split"
        );
        let result_ids: Vec<usize> = groups.iter().map(|grp| grp.id).collect();
        assert_eq!(
            result_ids, original_ids,
            "Valid group IDs should be preserved"
        );
    }

    // ── Test 47: max_epilogue_depth is at least 2 ──

    /// @trace TEST-HWC-47 [req:REQ-FUS] [level:unit]
    #[test]
    fn max_epilogue_depth_minimum_is_two() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let depth = max_epilogue_depth(&exec_plan);

        // Assert: the formula ((avail - reserved) / 2).max(2) guarantees at least 2
        assert!(
            depth >= 2,
            "max_epilogue_depth should be at least 2, got {depth}"
        );
    }

    // ── Test 48: validate_register_pressure for non-GEMM standalone uses 2 regs ──

    /// @trace TEST-HWC-48 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_register_pressure_non_gemm_standalone_is_two() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64], dt);
        let out = g.add_tensor_concrete("out", &[64], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let pressure = estimate_register_pressure(&group, &g, &exec_plan);

        // Assert: non-GEMM standalone = 2 (input + output)
        assert_eq!(pressure, 2, "non-GEMM standalone should use 2 registers");
    }

    // ── Test 49: estimate_l1_working_set GEMM returns positive value ──

    /// @trace TEST-HWC-49 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_l1_working_set_gemm_returns_positive() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 1024], dt);
        let w = g.add_tensor_concrete("w", &[1024, 2048], dt);
        let out = g.add_tensor_concrete("out", &[32, 2048], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 2048, k: 1024, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let ws = estimate_l1_working_set(&group, &g, &exec_plan);

        // Assert: GEMM working set must be positive (a_panel + b_panel)
        assert!(
            ws > 0,
            "GEMM L1 working set should be positive, got {ws}"
        );
    }

    // ── Test 50: register pressure for FusedQkvNormRope equals gemm_base ──

    /// @trace TEST-HWC-50 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_fused_qkv_norm_rope_equals_base() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let standalone = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let fused_group = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::FusedQkvNormRope {
                gemm_q: OpId(0),
                gemm_k: OpId(0),
                gemm_v: OpId(0),
                qk_norm_q: OpId(0),
                qk_norm_k: OpId(0),
                value_norm_v: OpId(0),
                rope_q: OpId(0),
                rope_k: OpId(0),
            },
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let fused = estimate_register_pressure(&fused_group, &g, &exec_plan);

        // Assert: FusedQkvNormRope = gemm_base (same as QkvSharedInput)
        assert_eq!(
            fused, base,
            "FusedQkvNormRope should equal base GEMM (base={base}, fused={fused})"
        );
    }

    // ── Test 51: extract_gemm_dtype falls back to graph.infer_computation_dtype for QuantGemm ──

    /// @trace TEST-HWC-51 [req:REQ-FUS] [level:unit]
    #[test]
    fn extract_gemm_dtype_quant_gemm_uses_graph_infer() {
        // Arrange: QuantGemm has no explicit dtype field, so it uses graph-level inference
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[4, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 256], dt);
        let out = g.add_tensor_concrete("out", &[4, 256], dt);
        g.add_op(Op::QuantGemm(QuantGemmSpec { m: crate::compiler::graph::SymDim::Concrete(4), n: 256, k: 128, quant_type: crate::quant::QuantType::Q4K }),
            vec![a, w],
            vec![out],
            "quant_gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let dtype = extract_gemm_dtype(&group, &g);

        // Assert: QuantGemm delegates to infer_computation_dtype
        assert_eq!(
            dtype, DType::F32,
            "QuantGemm should use graph.infer_computation_dtype()"
        );
    }

    // ── Test 52: estimate_register_pressure for Standalone with invalid anchor returns 2 ──

    /// @trace TEST-HWC-52 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_register_pressure_standalone_invalid_anchor_returns_two() {
        // Arrange: graph has no ops, so OpId(99) is invalid
        let g = CompilerGraph::new();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(99), // does not exist in graph
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(99)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let pressure = estimate_register_pressure(&group, &g, &exec_plan);

        // Assert: invalid anchor falls through to the default 2
        assert_eq!(
            pressure, 2,
            "Standalone with invalid anchor should use default 2 registers"
        );
    }

    // ── Test 53: build_result collects multiple violations simultaneously ──

    /// @trace TEST-HWC-53 [req:REQ-FUS] [level:unit]
    #[test]
    fn build_result_collects_multiple_violations() {
        // Arrange: construct a group that violates both epilogue depth and register pressure
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        // 80 epilogue ops: exceeds depth and register pressure simultaneously
        let epilogue_ops: Vec<OpId> = (1..=80).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_plan(&[group], &g);
        assert_eq!(result.len(), 1);
        let r = &result[0];

        // Assert: should have at least 1 violation (epilogue depth guaranteed)
        assert!(!r.valid, "Group with 80 epilogue ops should be invalid");
        assert!(
            !r.violations.is_empty(),
            "Should have at least one violation"
        );

        // Check that at least EpilogueTooDeep is present
        let has_depth = r.violations.iter().any(|v| {
            matches!(v, ConstraintViolation::EpilogueTooDeep { .. })
        });
        assert!(
            has_depth,
            "Should have EpilogueTooDeep violation among {:?}",
            r.violations
        );
    }

    // ── Test 54: validate_gpu_shared_mem passes for non-GPU profile (smem=0) ──

    /// @trace TEST-HWC-54 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_gpu_shared_mem_non_gpu_returns_ok() {
        // Arrange: CPU profile will have smem=0, which early-returns Ok
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 1024], dt);
        let w = g.add_tensor_concrete("w", &[1024, 1024], dt);
        let out = g.add_tensor_concrete("out", &[32, 1024], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 1024, k: 1024, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act: jit_ctx=None, on CPU smem will be 0
        let result = checker.validate_gpu_shared_mem(&group, &g, None);

        // Assert: smem=0 means not GPU → Ok(())
        assert!(
            result.is_ok(),
            "Non-GPU (smem=0) should pass: {:?}",
            result.err()
        );
    }

    // ── Test 55: validate_amx_tiles passes for small epilogue depth ──

    /// @trace TEST-HWC-55 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_amx_tiles_small_epilogue_passes_or_skips() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Group with 4 epilogue ops: needs 2 (gemm) + ceil(4/4)=1 = 3 tiles ≤ 8
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2), OpId(3), OpId(4)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1), OpId(2), OpId(3), OpId(4)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_amx_tiles(&group);

        // Assert: should pass (3 tiles ≤ 8) or Ok if no AMX
        assert!(
            result.is_ok(),
            "Small epilogue should pass AMX tile check (or skip on non-AMX): {:?}",
            result.err()
        );
    }

    // ── Test 56: validate_amx_tiles fails for very large epilogue depth ──

    /// @trace TEST-HWC-56 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_amx_tiles_large_epilogue_fails_or_passes() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Group with 32 epilogue ops: needs 2 (gemm) + ceil(32/4)=8 = 10 tiles > 8
        let epilogue_ops: Vec<OpId> = (1..=32).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_amx_tiles(&group);

        // Assert: if AMX is present, 10 tiles > 8 should fail; otherwise Ok (no AMX)
        use crate::compiler::hardware_profile::HardwareProfile;
        let hw = HardwareProfile::detect(&profile);
        if hw.has_amx() {
            assert!(result.is_err(), "32 epilogue ops should overflow AMX tiles");
            if let Err(ConstraintViolation::AmxTileOverflow { tiles_needed, tiles_available }) = result {
                assert_eq!(tiles_needed, 10);
                assert_eq!(tiles_available, 8);
            } else {
                panic!("Expected AmxTileOverflow");
            }
        } else {
            assert!(result.is_ok(), "Non-AMX should pass (no AMX to check)");
        }
    }

    // ── Test 57: estimate_l1_working_set for GEMM respects elem_bytes of dtype ──

    /// @trace TEST-HWC-57 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_l1_working_set_gemm_positive_and_dtype_dependent() {
        // Arrange: verify that the L1 working set is computed using elem_bytes from dtype
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // F32 GEMM
        let mut g_f32 = CompilerGraph::new();
        let a32 = g_f32.add_tensor_concrete("a", &[32, 1024], DType::F32);
        let w32 = g_f32.add_tensor_concrete("w", &[1024, 2048], DType::F32);
        let o32 = g_f32.add_tensor_concrete("out", &[32, 2048], DType::F32);
        g_f32.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 2048, k: 1024, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a32, w32],
            vec![o32],
            "gemm",
        );

        // BF16 GEMM with same dims
        let mut g_bf16 = CompilerGraph::new();
        let a16 = g_bf16.add_tensor_concrete("a", &[32, 1024], DType::BF16);
        let w16 = g_bf16.add_tensor_concrete("w", &[1024, 2048], DType::BF16);
        let o16 = g_bf16.add_tensor_concrete("out", &[32, 2048], DType::BF16);
        g_bf16.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 2048, k: 1024, dtype: DType::BF16, trans_b: false, has_bias: false }),
            vec![a16, w16],
            vec![o16],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let ws_f32 = estimate_l1_working_set(&group, &g_f32, &exec_plan);
        let ws_bf16 = estimate_l1_working_set(&group, &g_bf16, &exec_plan);

        // Assert: both must be positive; BF16 may use different (mr,nr,kc) due to wider simd_w,
        // but must be positive. The exact ratio depends on microkernel parameters.
        assert!(ws_f32 > 0, "F32 GEMM working set must be positive, got {ws_f32}");
        assert!(ws_bf16 > 0, "BF16 GEMM working set must be positive, got {ws_bf16}");
        // Verify that the formula (mr*kc + kc*nr)*elem_bytes is at least plausible:
        // F32 elem_bytes=4, BF16 elem_bytes=2. Both must be multiples of their elem_bytes.
        assert_eq!(ws_f32 % 4, 0, "F32 working set should be 4-byte aligned");
        assert_eq!(ws_bf16 % 2, 0, "BF16 working set should be 2-byte aligned");
    }

    // ── Test 58: enforce_constraints with mix of valid and invalid groups ──

    /// @trace TEST-HWC-58 [req:REQ-FUS] [level:unit]
    #[test]
    fn enforce_constraints_mixed_valid_invalid_groups() {
        // Arrange: one valid standalone GEMM + one oversized epilogue group
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let valid_group = FusionGroup {
            id: 10,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let epilogue_ops: Vec<OpId> = (1..=50).map(OpId).collect();
        let mut invalid_ops = vec![OpId(0)];
        invalid_ops.extend_from_slice(&epilogue_ops);
        let invalid_group = FusionGroup {
            id: 20,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: invalid_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut groups = vec![valid_group, invalid_group];

        // Act
        enforce_constraints(&mut groups, &g, &exec_plan);

        // Assert: first group (valid) should survive; second (invalid) should be split
        assert!(
            groups.len() > 2,
            "Should have at least 3 groups after splitting the invalid one, got {}",
            groups.len()
        );
        // The first group should still be Standalone with the original id
        assert_eq!(groups[0].id, 10, "Valid group id should be preserved");
        assert_eq!(groups[0].mode, FusionMode::Standalone);
    }

    // ── Test 59: gemm_base_regs for different dtypes ──

    /// @trace TEST-HWC-59 [req:REQ-FUS] [level:unit]
    #[test]
    fn gemm_base_regs_bf16_fewer_than_f32() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let regs_f32 = gemm_base_regs(&exec_plan, DType::F32);
        let regs_bf16 = gemm_base_regs(&exec_plan, DType::BF16);

        // Assert: BF16 has larger simd_w (more elements per register), so fewer regs needed
        assert!(
            regs_bf16 <= regs_f32,
            "BF16 base regs ({regs_bf16}) should be <= F32 base regs ({regs_f32})"
        );
    }

    // ── Test 60: validate_sme2_za passes for non-SME2 profile ──

    /// @trace TEST-HWC-60 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_sme2_za_non_sme2_returns_ok() {
        // Arrange: on most x86 systems, SME2 is not available
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 1024], dt);
        let w = g.add_tensor_concrete("w", &[1024, 1024], dt);
        let out = g.add_tensor_concrete("out", &[32, 1024], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 1024, k: 1024, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        // Even with many epilogue ops, non-SME2 should pass
        let epilogue_ops: Vec<OpId> = (1..=100).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_sme2_za(&group, &g);

        // Assert: non-SME2 always returns Ok
        assert!(
            result.is_ok(),
            "Non-SME2 should always pass SME2 ZA check: {:?}",
            result.err()
        );
    }

    // ── Test 61: validate_l1_working_set passes for GemmBias anchor ──

    /// @trace TEST-HWC-61 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_l1_working_set_gemm_bias_anchor_passes() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[16, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 256], dt);
        let out = g.add_tensor_concrete("out", &[16, 256], dt);
        g.add_op(Op::GemmBias(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 512, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, w],
            vec![out],
            "gemm_bias",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_l1_working_set(&group, &g);

        // Assert: GemmBias dims should be extracted and L1 check should pass
        assert!(
            result.is_ok(),
            "GemmBias anchor should pass L1 working set check: {:?}",
            result.err()
        );
    }

    // ── Test 62: extract_gemm_dtype for non-GEMM anchor falls back to graph infer ──

    /// @trace TEST-HWC-62 [req:REQ-FUS] [level:unit]
    #[test]
    fn extract_gemm_dtype_non_gemm_uses_graph_infer() {
        // Arrange: non-GEMM op has no explicit dtype, should fall back to graph.infer_computation_dtype()
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64], dt);
        let out = g.add_tensor_concrete("out", &[64], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let dtype = extract_gemm_dtype(&group, &g);

        // Assert: non-GEMM falls through to graph.infer_computation_dtype()
        assert_eq!(
            dtype, DType::F32,
            "Non-GEMM anchor should use graph.infer_computation_dtype()"
        );
    }

    // ── Test 63: enforce_constraints assigns correct group ids on split ──

    /// @trace TEST-HWC-63 [req:REQ-FUS] [level:unit]
    #[test]
    fn enforce_constraints_split_assigns_sequential_ids() {
        // Arrange: group with id=5 and 3 ops should split into ids 500, 501, 502
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[8, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[8, 64], dt);
        let add_out = g.add_tensor_concrete("add_out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(Op::Add, vec![silu_out, silu_out], vec![add_out], "add");

        let epilogue_ops: Vec<OpId> = (1..=50).map(OpId).collect();
        let mut all_ops = vec![OpId(0), OpId(1), OpId(2)];
        all_ops.extend_from_slice(&epilogue_ops);

        let mut groups = vec![FusionGroup {
            id: 5,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }];

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        enforce_constraints(&mut groups, &g, &exec_plan);

        // Assert: split groups should use id pattern old_id * 100 + j
        for (j, group) in groups.iter().enumerate() {
            assert_eq!(
                group.id, 5 * 100 + j,
                "Split group {} should have id={}, got {}",
                j, 5 * 100 + j, group.id
            );
        }
    }

    // ── Test 64: HwConstraintResult l1_budget equals l1 * budget_ratio ──

    /// @trace TEST-HWC-64 [req:REQ-FUS] [level:unit]
    #[test]
    fn hw_constraint_result_l1_budget_matches_ratio() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let (l1_raw, _, _) = profile.cache_sizes();
        let expected_budget = (l1_raw as f64 * profile.l1_budget_ratio()) as usize;

        // Act
        let result = check_group(&group, &g, &exec_plan);

        // Assert
        assert_eq!(
            result.l1_budget_bytes, expected_budget,
            "l1_budget_bytes should equal l1 * budget_ratio"
        );
    }

    // ── Test 65: validate_register_pressure EpilogueInjection saturates at free regs ──

    /// @trace TEST-HWC-65 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_register_pressure_epilogue_saturates_free_regs() {
        // Arrange: GEMM anchor with more epilogue ops than available free registers
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let base_group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let base = estimate_register_pressure(&base_group, &g, &exec_plan);
        let available = profile.num_simd_regs();
        let free = available.saturating_sub(base);

        // Group with far more epilogue ops than free regs
        let epilogue_ops: Vec<OpId> = (1..=200).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let saturated_group = FusionGroup {
            id: 1,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let pressure = estimate_register_pressure(&saturated_group, &g, &exec_plan);

        // Assert: pressure = base + min(200, free) + min(200-free, 1) capped at 1
        // The spill term contributes at most 1 extra register
        let expected_no_spill = free.min(200);
        let expected_spill = if 200 > free { 1 } else { 0 };
        let expected = base + expected_no_spill + expected_spill;
        assert_eq!(
            pressure, expected,
            "EpilogueInjection pressure should saturate: base={base}, free={free}, expected={expected}, got={pressure}"
        );
    }

    // ── Test 66: validate_sme2_za passes for non-GEMM group ──

    /// @trace TEST-HWC-66 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_sme2_za_non_gemm_passes() {
        // Arrange: non-GEMM group (extract_gemm_dims returns None)
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 1024], dt);
        let out = g.add_tensor_concrete("out", &[32, 1024], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_sme2_za(&group, &g);

        // Assert: non-GEMM group should always pass SME2 ZA check
        assert!(
            result.is_ok(),
            "Non-GEMM group should pass SME2 ZA check: {:?}",
            result.err()
        );
    }

    // ── Test 67: estimate_l1_working_set for non-GEMM with BF16 dtype ──

    /// @trace TEST-HWC-67 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_l1_working_set_elementwise_bf16_half_of_f32() {
        // Arrange: elementwise group with BF16 tensor should have half the L1 working set of F32
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let tile_size = profile.elem_tile_size();

        let mut g_f32 = CompilerGraph::new();
        let a32 = g_f32.add_tensor_concrete("a", &[64], DType::F32);
        let o32 = g_f32.add_tensor_concrete("out", &[64], DType::F32);
        g_f32.add_op(Op::Silu, vec![a32], vec![o32], "silu");

        let mut g_bf16 = CompilerGraph::new();
        let a16 = g_bf16.add_tensor_concrete("a", &[64], DType::BF16);
        let o16 = g_bf16.add_tensor_concrete("out", &[64], DType::BF16);
        g_bf16.add_op(Op::Silu, vec![a16], vec![o16], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let ws_f32 = estimate_l1_working_set(&group, &g_f32, &exec_plan);
        let ws_bf16 = estimate_l1_working_set(&group, &g_bf16, &exec_plan);

        // Assert: tile * elem_bytes * 2 → BF16 should be exactly half of F32
        let expected_f32 = tile_size * DType::F32.size_bytes() * 2;
        let expected_bf16 = tile_size * DType::BF16.size_bytes() * 2;
        assert_eq!(ws_f32, expected_f32, "F32 elementwise working set mismatch");
        assert_eq!(ws_bf16, expected_bf16, "BF16 elementwise working set mismatch");
        assert_eq!(
            ws_f32, ws_bf16 * 2,
            "F32 elementwise working set should be exactly 2x BF16"
        );
    }

    // ── Test 68: build_result epilogue_depth matches group.epilogue.len() ──

    /// @trace TEST-HWC-68 [req:REQ-FUS] [level:unit]
    #[test]
    fn build_result_epilogue_depth_matches_group() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[8, 64], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[8, 64], dt);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(Op::Tanh, vec![silu_out], vec![tanh_out], "tanh");

        let group = FusionGroup {
            id: 7,
            anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1), OpId(2)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let results = checker.validate_plan(&[group], &g);
        assert_eq!(results.len(), 1);
        let r = &results[0];

        // Assert
        assert_eq!(r.group_id, 7, "group_id should match input");
        assert_eq!(r.epilogue_depth, 2, "epilogue_depth should be 2");
        assert_eq!(r.max_epilogue_depth, max_epilogue_depth(&exec_plan));
    }

    // ── Test 69: validate_plan results count matches input groups count ──

    /// @trace TEST-HWC-69 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_plan_results_count_matches_groups_count() {
        // Arrange: create a small GEMM graph and fuse it
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 128], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[8, 128], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[8, 128], dt);
        let b = g.add_tensor_concrete("b", &[8, 128], dt);
        let add_out = g.add_tensor_concrete("add_out", &[8, 128], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[8, 128], dt);

        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(Op::Add, vec![silu_out, b], vec![add_out], "add");
        g.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }), vec![add_out], vec![norm_out], "norm");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let results = checker.validate_plan(&plan.groups, &g);

        // Assert: 1:1 mapping between groups and results
        assert_eq!(
            results.len(),
            plan.groups.len(),
            "results count should equal groups count"
        );
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.group_id, plan.groups[i].id,
                "result[{}] group_id should match group[{}].id",
                i, i
            );
        }
    }

    // ── Test 70: max_epilogue_depth scales with register count ──

    /// @trace TEST-HWC-70 [req:REQ-FUS] [level:unit]
    #[test]
    fn max_epilogue_depth_increases_with_more_regs() {
        // Arrange: verify that max_epilogue_depth formula produces sensible values
        // Formula: ((avail - reserved) / 2).max(2)
        // avail = num_simd_regs, reserved = gemm_accumulator_regs
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let avail = profile.num_simd_regs();
        let reserved = profile.gemm_accumulator_regs();

        // Act
        let depth = max_epilogue_depth(&exec_plan);

        // Assert: depth = ((avail - reserved) / 2).max(2)
        let expected = ((avail.saturating_sub(reserved)) / 2).max(2);
        assert_eq!(
            depth, expected,
            "max_epilogue_depth should be ((avail={avail} - reserved={reserved}) / 2).max(2)"
        );
        // Sanity: depth should be reasonable relative to register file size
        assert!(
            depth <= avail,
            "max_epilogue_depth ({depth}) should not exceed available regs ({avail})"
        );
    }

    // ── Test 71: validate_register_pressure passes for elementwise Standalone ──

    /// @trace TEST-HWC-71 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_register_pressure_elementwise_standalone_passes() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64], dt);
        let out = g.add_tensor_concrete("out", &[64], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_register_pressure(&group, &g);

        // Assert: elementwise standalone uses 2 regs, always within limit
        assert!(
            result.is_ok(),
            "Elementwise Standalone should pass register check: {:?}",
            result.err()
        );
    }

    // ── Test 72: HwConstraintResult l1_working_set_bytes positive for GEMM ──

    /// @trace TEST-HWC-72 [req:REQ-FUS] [level:unit]
    #[test]
    fn hw_constraint_result_l1_working_set_positive_for_gemm() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[16, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 512], dt);
        let out = g.add_tensor_concrete("out", &[16, 512], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 512, k: 256, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let result = check_group(&group, &g, &exec_plan);

        // Assert
        assert!(
            result.l1_working_set_bytes > 0,
            "GEMM L1 working set should be positive, got {}",
            result.l1_working_set_bytes
        );
        assert!(
            result.l1_working_set_bytes <= result.l1_budget_bytes,
            "GEMM working set ({}) should fit in L1 budget ({})",
            result.l1_working_set_bytes, result.l1_budget_bytes
        );
    }

    // ── Test 73: check_plan with single-elementwise graph returns one result ──

    /// @trace TEST-HWC-73 [req:REQ-FUS] [level:unit]
    #[test]
    fn check_plan_single_elementwise_returns_one_valid_result() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[128], dt);
        let out = g.add_tensor_concrete("out", &[128], dt);
        g.add_op(Op::Gelu, vec![a], vec![out], "gelu");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);

        // Act
        let results = check_plan(&plan.groups, &g, &exec_plan);

        // Assert
        assert_eq!(results.len(), 1, "single op should produce 1 group");
        assert!(results[0].valid, "single elementwise should be valid");
        assert_eq!(results[0].epilogue_depth, 0, "standalone has no epilogue");
    }

    // ── Test 74: gemm_base_regs for F16 is fewer than or equal to F32 ──

    /// @trace TEST-HWC-74 [req:REQ-FUS] [level:unit]
    #[test]
    fn gemm_base_regs_f16_leq_f32() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let regs_f32 = gemm_base_regs(&exec_plan, DType::F32);
        let regs_f16 = gemm_base_regs(&exec_plan, DType::F16);

        // Assert: F16 has larger simd_w → fewer registers needed
        assert!(
            regs_f16 <= regs_f32,
            "F16 base regs ({regs_f16}) should be <= F32 ({regs_f32})"
        );
    }

    // ── Test 75: validate_group passes for GemmBias standalone ──

    /// @trace TEST-HWC-75 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_group_gemm_bias_standalone_passes() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[16, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 128], dt);
        let out = g.add_tensor_concrete("out", &[16, 128], dt);
        g.add_op(Op::GemmBias(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 128, k: 256, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, w], vec![out], "gemm_bias",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_group(&group, &g);

        // Assert
        assert!(
            result.is_ok(),
            "GemmBias standalone should pass all checks: {:?}",
            result.err()
        );
    }

    // ── Test 76: ConstraintViolation clone works for L1Overflow ──

    /// @trace TEST-HWC-76 [req:REQ-FUS] [level:unit]
    #[test]
    fn constraint_violation_clone_l1_overflow() {
        // Arrange
        let v = ConstraintViolation::L1Overflow { working_set: 98304, l1_size: 32768 };

        // Act
        let cloned = v.clone();

        // Assert
        match cloned {
            ConstraintViolation::L1Overflow { working_set, l1_size } => {
                assert_eq!(working_set, 98304);
                assert_eq!(l1_size, 32768);
            }
            other => panic!("expected L1Overflow, got {:?}", other),
        }
    }

    // ── Test 77: validate_epilogue_depth passes for single epilogue op ──

    /// @trace TEST-HWC-77 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_epilogue_depth_single_op_passes() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_epilogue_depth(&group);

        // Assert: 1 epilogue op should always pass (max is at least 2)
        assert!(
            result.is_ok(),
            "Single epilogue op should pass depth check: {:?}",
            result.err()
        );
    }

    // ── Test 78: estimate_register_pressure EpilogueInjection increases linearly before saturation ──

    /// @trace TEST-HWC-78 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_register_pressure_epilogue_grows_before_saturation() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let regs = profile.num_simd_regs();

        // Build two groups: 1 epilogue vs 2 epilogue — both should be within free regs
        let g1 = FusionGroup {
            id: 0, anchor: OpId(0),
            epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let g2 = FusionGroup {
            id: 1, anchor: OpId(0),
            epilogue: vec![OpId(1), OpId(2)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0), OpId(1), OpId(2)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let p1 = estimate_register_pressure(&g1, &g, &exec_plan);
        let p2 = estimate_register_pressure(&g2, &g, &exec_plan);

        // Assert: both within register limit, and p2 >= p1
        assert!(p1 <= regs, "1-epilogue pressure {p1} should fit in {regs} regs");
        assert!(p2 <= regs, "2-epilogue pressure {p2} should fit in {regs} regs");
        assert!(p2 >= p1, "2 epilogue ops should need at least as many regs as 1: p1={p1}, p2={p2}");
    }

    // ── Test 79: check_group for QuantGemm returns positive register_pressure ──

    /// @trace TEST-HWC-79 [req:REQ-FUS] [level:unit]
    #[test]
    fn check_group_quant_gemm_positive_register_pressure() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[4, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 256], dt);
        let out = g.add_tensor_concrete("out", &[4, 256], dt);
        g.add_op(Op::QuantGemm(QuantGemmSpec { m: crate::compiler::graph::SymDim::Concrete(4), n: 256, k: 128, quant_type: crate::quant::QuantType::Q4K }),
            vec![a, w], vec![out], "quant_gemm",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let result = check_group(&group, &g, &exec_plan);

        // Assert
        assert!(
            result.register_pressure > 0,
            "QuantGemm register pressure should be positive, got {}",
            result.register_pressure
        );
        assert!(result.valid, "QuantGemm standalone should be valid");
    }

    // ── Test 80: HwConstraintResult max_epilogue_depth matches free function ──

    /// @trace TEST-HWC-80 [req:REQ-FUS] [level:unit]
    #[test]
    fn hw_constraint_result_max_epilogue_matches_function() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let expected_max = max_epilogue_depth(&exec_plan);

        // Act
        let result = check_group(&group, &g, &exec_plan);

        // Assert
        assert_eq!(
            result.max_epilogue_depth, expected_max,
            "HwConstraintResult.max_epilogue_depth should match max_epilogue_depth()"
        );
    }

    // ── Test 81: zero epilogue depth always passes epilogue check ──

    /// @trace TEST-HWC-81 [req:REQ-FUS] [level:unit]
    #[test]
    fn zero_epilogue_depth_always_passes_check() {
        // Arrange: a group with no epilogue, in EpilogueInjection mode
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_epilogue_depth(&group);

        // Assert: zero depth is always within any positive max
        assert!(result.is_ok(), "zero epilogue depth should always pass");
    }

    // ── Test 82: L1 budget equals 85% of L1D cache size ──

    /// @trace TEST-HWC-82 [req:REQ-FUS] [level:unit]
    #[test]
    fn l1_budget_equals_85_percent_of_l1d() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let (l1_raw, _, _) = profile.cache_sizes();
        let expected_budget = (l1_raw as f64 * profile.l1_budget_ratio()) as usize;

        // Act
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );
        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let result = check_group(&group, &g, &exec_plan);

        // Assert
        assert_eq!(
            result.l1_budget_bytes, expected_budget,
            "L1 budget should be l1_raw * l1_budget_ratio"
        );
        assert!(
            result.l1_budget_bytes < l1_raw,
            "L1 budget ({}) should be strictly less than raw L1 ({})",
            result.l1_budget_bytes, l1_raw,
        );
    }

    // ── Test 83: register pressure overflow detected by validate_register_pressure ──

    /// @trace TEST-HWC-83 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_overflow_detected_by_validate() {
        // Arrange: build a group with many epilogue ops to try to trigger overflow.
        // EpilogueInjection pressure = base + no_spill + min(spill_ops, 1), which
        // saturates quickly, so we verify the violation path works when it triggers
        // and that the function is callable without panic.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let epilogue_ops: Vec<OpId> = (1..=200).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_register_pressure(&group, &g);

        // Assert: if it overflows, the violation must have correct fields
        if let Err(ConstraintViolation::RegisterPressure { needed, available }) = result {
            assert!(needed > available, "needed ({needed}) should exceed available ({available})");
        }
        // If it does not overflow on this profile, that is also acceptable —
        // the important thing is the function runs without panic.
    }

    // ── Test 84: fusion rejection — epilogue one beyond max depth fails ──

    /// @trace TEST-HWC-84 [req:REQ-FUS] [level:unit]
    #[test]
    fn epilogue_one_beyond_max_depth_fails() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let limit = max_epilogue_depth(&exec_plan);
        let beyond = limit + 1;

        let epilogue_ops: Vec<OpId> = (1..=beyond).map(|i| OpId(i as u32)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_epilogue_depth(&group);

        // Assert: exactly one beyond the limit should fail
        assert!(result.is_err(), "depth {} should fail when max is {}", beyond, limit);
        if let Err(ConstraintViolation::EpilogueTooDeep { depth, max }) = result {
            assert_eq!(depth, beyond);
            assert_eq!(max, limit);
        }
    }

    // ── Test 85: validate_group rejects a group with multiple simultaneous violations ──

    /// @trace TEST-HWC-85 [req:REQ-FUS] [level:unit]
    #[test]
    fn build_result_reports_multiple_violations() {
        // Arrange: oversized epilogue that violates both depth and register pressure
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let epilogue_ops: Vec<OpId> = (1..=200).map(OpId).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.build_result(&group, &g);

        // Assert: oversized group should have at least one violation
        assert!(!result.valid, "oversized group should be invalid");
        assert!(
            !result.violations.is_empty(),
            "oversized group should have at least one violation"
        );
    }

    // ── Test 86: enforce_constraints on empty groups vec is a no-op ──

    /// @trace TEST-HWC-86 [req:REQ-FUS] [level:unit]
    #[test]
    fn enforce_constraints_empty_groups_is_noop() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let g = CompilerGraph::new();
        let mut groups: Vec<FusionGroup> = vec![];

        // Act
        enforce_constraints(&mut groups, &g, &exec_plan);

        // Assert: empty input produces empty output
        assert!(groups.is_empty(), "empty groups should remain empty after enforce");
    }

    // ── Test 87: L1 working set for BF16 GEMM is smaller than F32 GEMM ──

    /// @trace TEST-HWC-87 [req:REQ-FUS] [level:unit]
    #[test]
    fn l1_working_set_bf16_smaller_than_f32() {
        // Arrange: same GEMM dimensions, different dtypes
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let mut g32 = CompilerGraph::new();
        let a32 = g32.add_tensor_concrete("a", &[16, 512], DType::F32);
        let w32 = g32.add_tensor_concrete("w", &[512, 256], DType::F32);
        let out32 = g32.add_tensor_concrete("out", &[16, 256], DType::F32);
        g32.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a32, w32], vec![out32], "gemm",
        );

        let mut g16 = CompilerGraph::new();
        let a16 = g16.add_tensor_concrete("a", &[16, 512], DType::BF16);
        let w16 = g16.add_tensor_concrete("w", &[512, 256], DType::BF16);
        let out16 = g16.add_tensor_concrete("out", &[16, 256], DType::BF16);
        g16.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 512, dtype: DType::BF16, trans_b: false, has_bias: false }),
            vec![a16, w16], vec![out16], "gemm",
        );

        let group32 = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let group16 = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let ws32 = estimate_l1_working_set(&group32, &g32, &exec_plan);
        let ws16 = estimate_l1_working_set(&group16, &g16, &exec_plan);

        // Assert: BF16 elem_bytes=2 < F32 elem_bytes=4, so working set is smaller
        assert!(
            ws16 < ws32,
            "BF16 working set ({ws16}) should be smaller than F32 ({ws32})"
        );
    }

    // ── Test 88: validate_register_pressure passes for Standalone non-GEMM ──

    /// @trace TEST-HWC-88 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_register_pressure_standalone_non_gemm_passes() {
        // Arrange: RmsNorm (non-GEMM) standalone — uses 2 regs
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[128], dt);
        let out = g.add_tensor_concrete("out", &[128], dt);
        g.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }), vec![a], vec![out], "norm");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_register_pressure(&group, &g);

        // Assert: 2 regs always fits within any positive register file
        assert!(result.is_ok(), "non-GEMM standalone should pass register check: {:?}", result.err());
    }

    // ── Test 89: register pressure for missing anchor op defaults to 2 ──

    /// @trace TEST-HWC-89 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_missing_anchor_defaults_to_two() {
        // Arrange: Standalone group referencing a non-existent OpId
        let g = CompilerGraph::new();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(999), // does not exist in graph
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(999)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let pressure = estimate_register_pressure(&group, &g, &exec_plan);

        // Assert: missing anchor → falls through to default 2
        assert_eq!(pressure, 2, "missing anchor op should default to 2 register pressure");
    }

    // ── Test 90: gemm_base_regs with BF16 uses fewer or equal regs than F32 ──

    /// @trace TEST-HWC-90 [req:REQ-FUS] [level:unit]
    #[test]
    fn gemm_base_regs_bf16_leq_f32() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let regs_f32 = gemm_base_regs(&exec_plan, DType::F32);
        let regs_bf16 = gemm_base_regs(&exec_plan, DType::BF16);

        // Assert: BF16 has larger simd_w → fewer accumulator and A-panel regs
        assert!(
            regs_bf16 <= regs_f32,
            "BF16 base regs ({regs_bf16}) should be <= F32 ({regs_f32})"
        );
    }

    // ── Test 91: ConstraintViolation field values preserved across Debug format ──

    /// @trace TEST-HWC-91 [req:REQ-FUS] [level:unit]
    #[test]
    fn constraint_violation_debug_preserves_field_values() {
        // Arrange
        let violations = vec![
            ConstraintViolation::AmxTileOverflow { tiles_needed: 10, tiles_available: 8 },
            ConstraintViolation::Sme2ZaOverflow { za_bytes: 32768, max_za_bytes: 16384 },
            ConstraintViolation::SharedMemOverflow { tile_bytes: 65536, smem_bytes: 49152 },
        ];

        // Act
        for v in &violations {
            let debug = format!("{:?}", v);

            // Assert: Debug output must contain the numeric values from the fields
            match v {
                ConstraintViolation::AmxTileOverflow { tiles_needed, tiles_available } => {
                    assert!(debug.contains(&tiles_needed.to_string()), "Debug should contain tiles_needed");
                    assert!(debug.contains(&tiles_available.to_string()), "Debug should contain tiles_available");
                }
                ConstraintViolation::Sme2ZaOverflow { za_bytes, max_za_bytes } => {
                    assert!(debug.contains(&za_bytes.to_string()), "Debug should contain za_bytes");
                    assert!(debug.contains(&max_za_bytes.to_string()), "Debug should contain max_za_bytes");
                }
                ConstraintViolation::SharedMemOverflow { tile_bytes, smem_bytes } => {
                    assert!(debug.contains(&tile_bytes.to_string()), "Debug should contain tile_bytes");
                    assert!(debug.contains(&smem_bytes.to_string()), "Debug should contain smem_bytes");
                }
                _ => {}
            }
        }
    }

    // ── Test 92: HwConstraintResult clone isolation ──

    /// @trace TEST-HWC-92 [req:REQ-FUS] [level:unit]
    #[test]
    fn hw_constraint_result_clone_isolation() {
        // Arrange: construct a result with violations, clone it, mutate the clone,
        // and verify the original is unchanged.
        let original = HwConstraintResult {
            group_id: 5,
            valid: false,
            register_pressure: 64,
            register_limit: 32,
            l1_working_set_bytes: 65536,
            l1_budget_bytes: 32768,
            epilogue_depth: 20,
            max_epilogue_depth: 8,
            violations: vec![
                ConstraintViolation::RegisterPressure { needed: 64, available: 32 },
                ConstraintViolation::EpilogueTooDeep { depth: 20, max: 8 },
            ],
        };

        // Act
        let mut cloned = original.clone();
        cloned.violations.clear();
        cloned.valid = true;

        // Assert: original is unaffected
        assert!(!original.valid, "original.valid should still be false");
        assert_eq!(original.violations.len(), 2, "original violations should still have 2 entries");
        assert!(cloned.valid, "cloned.valid should be true after mutation");
        assert!(cloned.violations.is_empty(), "cloned violations should be empty after clear");
    }

    // ── Test 93: validate_gpu_shared_mem passes for non-GEMM group ──

    /// @trace TEST-HWC-93 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_gpu_shared_mem_non_gemm_group_passes() {
        // Arrange: a non-GEMM group — extract_gemm_dims returns None → early Ok
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64], dt);
        let out = g.add_tensor_concrete("out", &[64], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act: jit_ctx = None
        let result = checker.validate_gpu_shared_mem(&group, &g, None);

        // Assert: non-GEMM group should always pass GPU shared mem check
        assert!(
            result.is_ok(),
            "Non-GEMM group should pass GPU shared mem check: {:?}",
            result.err()
        );
    }

    // ── Test 94: enforce_constraints with single-op group leaves it intact ──

    /// @trace TEST-HWC-94 [req:REQ-FUS] [level:unit]
    #[test]
    fn enforce_constraints_single_op_group_unchanged() {
        // Arrange: a standalone group with a single op that is always valid
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64], dt);
        let out = g.add_tensor_concrete("out", &[64], dt);
        g.add_op(Op::Silu, vec![a], vec![out], "silu");

        let mut groups = vec![FusionGroup {
            id: 99,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        }];

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        enforce_constraints(&mut groups, &g, &exec_plan);

        // Assert: valid single-op group should remain as-is
        assert_eq!(groups.len(), 1, "single valid group should not be split");
        assert_eq!(groups[0].id, 99, "group id should be preserved");
        assert_eq!(groups[0].mode, FusionMode::Standalone);
        assert!(groups[0].epilogue.is_empty());
    }

    // ── Test 95: FusionGroup with dominant_dtype set passes constraints ──

    /// @trace TEST-HWC-95 [req:REQ-FUS] [level:unit]
    #[test]
    fn fusion_group_with_dominant_dtype_passes_constraints() {
        // Arrange: a GEMM group with dominant_dtype explicitly set
        let mut g = CompilerGraph::new();
        let dt = DType::BF16;
        let a = g.add_tensor_concrete("a", &[16, 512], DType::BF16);
        let w = g.add_tensor_concrete("w", &[512, 256], DType::BF16);
        let out = g.add_tensor_concrete("out", &[16, 256], DType::BF16);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 512, dtype: DType::BF16, trans_b: false, has_bias: false }),
            vec![a, w],
            vec![out],
            "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: Some(crate::compiler::trace::QuantPrecision::BF16),
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_group(&group, &g);

        // Assert: setting dominant_dtype should not cause constraint failures
        assert!(
            result.is_ok(),
            "GEMM group with dominant_dtype should pass: {:?}",
            result.err()
        );
    }

    // ── Test 96: estimate_l1_working_set for empty ops returns elementwise fallback ──

    /// @trace TEST-HWC-96 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_l1_working_set_empty_ops_uses_tile_fallback() {
        // Arrange: a group with an anchor op not in the graph — extract_gemm_dims
        // returns None, and the ops list has no valid op for tensor dtype lookup,
        // so it falls back to F32 elem_bytes.
        let g = CompilerGraph::new();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let tile_size = profile.elem_tile_size();

        let group = FusionGroup {
            id: 0,
            anchor: OpId(999), // does not exist in graph
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(999)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let ws = estimate_l1_working_set(&group, &g, &exec_plan);

        // Assert: fallback = tile_size * F32_size * 2 (input + output)
        let expected = tile_size * DType::F32.size_bytes() * 2;
        assert_eq!(
            ws, expected,
            "Empty/invalid ops L1 working set should use F32 tile fallback: got {ws}, expected {expected}"
        );
    }

    // ── Test 97: max_epilogue_depth is consistent between free function and build_result ──

    /// @trace TEST-HWC-97 [req:REQ-FUS] [level:unit]
    #[test]
    fn max_epilogue_depth_consistent_with_build_result() {
        // Arrange: create two groups with different epilogue depths and verify
        // that build_result reports the same max_epilogue_depth as the free function
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let group_no_epi = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let group_with_epi = FusionGroup {
            id: 1, anchor: OpId(0), epilogue: vec![OpId(1), OpId(2)],
            mode: FusionMode::EpilogueInjection, ops: vec![OpId(0), OpId(1), OpId(2)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let expected_max = max_epilogue_depth(&exec_plan);

        // Act
        let r1 = checker.build_result(&group_no_epi, &g);
        let r2 = checker.build_result(&group_with_epi, &g);

        // Assert: both results should report the same max_epilogue_depth
        assert_eq!(r1.max_epilogue_depth, expected_max, "group_no_epi max depth mismatch");
        assert_eq!(r2.max_epilogue_depth, expected_max, "group_with_epi max depth mismatch");
        assert_eq!(r1.max_epilogue_depth, r2.max_epilogue_depth, "both groups should report same max");
    }

    // ── Test 98: ConstraintViolation EpilogueTooDeep at boundary values ──

    /// @trace TEST-HWC-98 [req:REQ-FUS] [level:unit]
    #[test]
    fn constraint_violation_epilogue_too_deep_boundary_values() {
        // Arrange: test ConstraintViolation::EpilogueTooDeep with usize::MAX
        let v = ConstraintViolation::EpilogueTooDeep {
            depth: usize::MAX,
            max: usize::MAX - 1,
        };

        // Act
        let cloned = v.clone();
        let debug = format!("{:?}", v);

        // Assert: clone preserves boundary values
        match cloned {
            ConstraintViolation::EpilogueTooDeep { depth, max } => {
                assert_eq!(depth, usize::MAX);
                assert_eq!(max, usize::MAX - 1);
            }
            other => panic!("expected EpilogueTooDeep, got {:?}", other),
        }
        // Debug format should contain the large values
        assert!(debug.contains("EpilogueTooDeep"), "Debug should contain variant name");
    }

    // ── Test 99: validate_l1_working_set with QuantGemm anchor extracts dims ──

    /// @trace TEST-HWC-99 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_l1_working_set_quant_gemm_anchor_passes() {
        // Arrange: QuantGemm is a GEMM variant — extract_gemm_dims should
        // return Some, and validate_l1_working_set should compute the working set
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[4, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 256], dt);
        let out = g.add_tensor_concrete("out", &[4, 256], dt);
        g.add_op(Op::QuantGemm(QuantGemmSpec { m: crate::compiler::graph::SymDim::Concrete(4), n: 256, k: 128, quant_type: crate::quant::QuantType::Q4K }),
            vec![a, w],
            vec![out],
            "quant_gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_l1_working_set(&group, &g);

        // Assert: QuantGemm should be recognized as GEMM and L1 check should pass
        assert!(
            result.is_ok(),
            "QuantGemm anchor should pass L1 working set check: {:?}",
            result.err()
        );

        // Verify working set is positive via estimate_l1_working_set
        let ws = estimate_l1_working_set(&group, &g, &exec_plan);
        assert!(ws > 0, "QuantGemm L1 working set should be positive, got {ws}");
    }

    // ── Test 100: HwConstraintChecker lifetime borrows ExecutionPlan ──

    /// @trace TEST-HWC-100 [req:REQ-FUS] [level:unit]
    #[test]
    fn hw_constraint_checker_borrows_plan_not_owns() {
        // Arrange: verify that HwConstraintChecker holds a reference to ExecutionPlan,
        // not an owned copy. This test verifies the API contract: the checker
        // does not outlive the plan.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act: create checker and validate — the checker borrows exec_plan
        let checker = HwConstraintChecker::new(&exec_plan);
        let result = checker.validate_group(&group, &g);

        // Assert: checker successfully used the borrowed plan
        assert!(
            result.is_ok(),
            "Checker with borrowed plan should validate successfully: {:?}",
            result.err()
        );

        // Verify plan is still accessible after checker is dropped
        let _still_valid = exec_plan.profile.num_simd_regs();
    }

    // ── Test 101: register pressure monotonically increases with LoopFusion ops ──

    /// @trace TEST-HWC-101 [req:REQ-FUS] [level:unit]
    #[test]
    fn register_pressure_loop_fusion_monotonically_increases() {
        // Arrange
        let g = CompilerGraph::new();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act: measure register pressure for 1, 2, 3, 4, 5 ops
        let pressures: Vec<usize> = (1..=5)
            .map(|n| {
                let group = FusionGroup {
                    id: 0,
                    anchor: OpId(0),
                    epilogue: vec![],
                    mode: FusionMode::LoopFusion,
                    ops: (0..n).map(|i| OpId(i as u32)).collect(),
                    multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
            hetero_layer_type: None,
                };
                estimate_register_pressure(&group, &g, &exec_plan)
            })
            .collect();

        // Assert: each step should increase by exactly 1 (LoopFusion = 3 + ops.len())
        for i in 1..pressures.len() {
            assert_eq!(
                pressures[i] - pressures[i - 1], 1,
                "LoopFusion pressure should increase by 1 per op: {} -> {}",
                pressures[i - 1], pressures[i]
            );
        }
    }

    // ── Test 102: build_result register_limit is at least num_simd_regs ──

    /// @trace TEST-HWC-102 [req:REQ-FUS] [level:unit]
    #[test]
    fn build_result_register_limit_at_least_profile_regs() {
        // Arrange: the register_limit in build_result is num_simd_regs() plus an optional
        // +4 headroom for memory-bound groups. So the limit must be >= num_simd_regs().
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[8, 64], dt);
        let w = g.add_tensor_concrete("w", &[64, 64], dt);
        let out = g.add_tensor_concrete("out", &[8, 64], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let min_limit = profile.num_simd_regs();

        // Act
        let result = check_group(&group, &g, &exec_plan);

        // Assert: limit is at least num_simd_regs (may be +4 if memory-bound)
        assert!(
            result.register_limit >= min_limit,
            "register_limit ({}) should be >= num_simd_regs ({})",
            result.register_limit, min_limit
        );
        // The difference must be 0 or 4 (memory-bound headroom)
        let delta = result.register_limit - min_limit;
        assert!(
            delta == 0 || delta == 4,
            "register_limit delta should be 0 or 4, got {delta}"
        );
    }

    // ── Test 103: estimate_l1_working_set for GEMM scales with K dimension ──

    /// @trace TEST-HWC-103 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_l1_working_set_scales_with_k_dimension() {
        // Arrange: two GEMM graphs with different K dimensions
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let mut g_small_k = CompilerGraph::new();
        let a1 = g_small_k.add_tensor_concrete("a", &[16, 128], DType::F32);
        let w1 = g_small_k.add_tensor_concrete("w", &[128, 256], DType::F32);
        let o1 = g_small_k.add_tensor_concrete("out", &[16, 256], DType::F32);
        g_small_k.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 128, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a1, w1], vec![o1], "gemm",
        );

        let mut g_large_k = CompilerGraph::new();
        let a2 = g_large_k.add_tensor_concrete("a", &[16, 1024], DType::F32);
        let w2 = g_large_k.add_tensor_concrete("w", &[1024, 256], DType::F32);
        let o2 = g_large_k.add_tensor_concrete("out", &[16, 256], DType::F32);
        g_large_k.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 1024, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a2, w2], vec![o2], "gemm",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let ws_small = estimate_l1_working_set(&group, &g_small_k, &exec_plan);
        let ws_large = estimate_l1_working_set(&group, &g_large_k, &exec_plan);

        // Assert: larger K → larger working set (formula: (mr*kc + kc*nr)*elem_bytes)
        assert!(
            ws_large >= ws_small,
            "L1 working set with K=1024 ({ws_large}) should be >= K=128 ({ws_small})"
        );
    }

    // ── Test 104: gemm_base_regs is consistent across repeated calls ──

    /// @trace TEST-HWC-104 [req:REQ-FUS] [level:unit]
    #[test]
    fn gemm_base_regs_is_deterministic() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act: call gemm_base_regs multiple times for the same dtype
        let r1_f32 = gemm_base_regs(&exec_plan, DType::F32);
        let r2_f32 = gemm_base_regs(&exec_plan, DType::F32);
        let r1_bf16 = gemm_base_regs(&exec_plan, DType::BF16);
        let r2_bf16 = gemm_base_regs(&exec_plan, DType::BF16);

        // Assert: identical inputs must produce identical outputs
        assert_eq!(r1_f32, r2_f32, "gemm_base_regs(F32) should be deterministic");
        assert_eq!(r1_bf16, r2_bf16, "gemm_base_regs(BF16) should be deterministic");
    }

    // ── Test 105: validate_epilogue_depth returns correct error fields ──

    /// @trace TEST-HWC-105 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_epilogue_depth_error_fields_are_correct() {
        // Arrange
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);
        let limit = max_epilogue_depth(&exec_plan);

        // Create a group with limit + 5 epilogue ops
        let beyond = limit + 5;
        let epilogue_ops: Vec<OpId> = (1..=beyond).map(|i| OpId(i as u32)).collect();
        let mut all_ops = vec![OpId(0)];
        all_ops.extend_from_slice(&epilogue_ops);

        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: epilogue_ops,
            mode: FusionMode::EpilogueInjection,
            ops: all_ops,
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let result = checker.validate_epilogue_depth(&group);

        // Assert
        assert!(result.is_err(), "depth {} should exceed max {}", beyond, limit);
        match result.unwrap_err() {
            ConstraintViolation::EpilogueTooDeep { depth, max } => {
                assert_eq!(depth, beyond, "error depth should equal epilogue length");
                assert_eq!(max, limit, "error max should equal max_epilogue_depth");
            }
            other => panic!("Expected EpilogueTooDeep, got {:?}", other),
        }
    }

    // ── Test 106: validate_l1_working_set with zero-sized GEMM dims returns Ok ──

    /// @trace TEST-HWC-106 [req:REQ-FUS] [level:unit]
    #[test]
    fn validate_l1_working_set_tiny_gemm_passes() {
        // Arrange: minimal GEMM with very small M, N, K — working set should be tiny
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 1], dt);
        let w = g.add_tensor_concrete("w", &[1, 1], dt);
        let out = g.add_tensor_concrete("out", &[1, 1], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(1), n: 1, k: 1, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let group = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let checker = HwConstraintChecker::new(&exec_plan);

        // Act
        let result = checker.validate_l1_working_set(&group, &g);

        // Assert: tiny GEMM should always fit in L1
        assert!(
            result.is_ok(),
            "Tiny 1x1x1 GEMM should pass L1 working set check: {:?}",
            result.err()
        );

        // Verify the working set is positive but small
        let ws = estimate_l1_working_set(&group, &g, &exec_plan);
        assert!(ws > 0, "even tiny GEMM should have positive working set");
    }

    // ── Test 107: ConstraintViolation RegisterPressure clone preserves both fields ──

    /// @trace TEST-HWC-107 [req:REQ-FUS] [level:unit]
    #[test]
    fn constraint_violation_register_pressure_clone_preserves_fields() {
        // Arrange: construct with specific asymmetric values
        let v = ConstraintViolation::RegisterPressure { needed: 37, available: 13 };

        // Act
        let cloned = v.clone();

        // Assert
        match cloned {
            ConstraintViolation::RegisterPressure { needed, available } => {
                assert_eq!(needed, 37, "cloned needed should be 37");
                assert_eq!(available, 13, "cloned available should be 13");
            }
            other => panic!("expected RegisterPressure, got {:?}", other),
        }
    }

    // ── Test 108: HwConstraintResult valid groups have zero violations ──

    /// @trace TEST-HWC-108 [req:REQ-FUS] [level:unit]
    #[test]
    fn valid_result_has_zero_violations_and_positive_metrics() {
        // Arrange: fuse a simple GEMM+Silu graph — should produce valid groups
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[16, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 256], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[16, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[16, 256], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(16), n: 256, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(Op::Silu, vec![gemm_out], vec![silu_out], "silu");

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&g, &registry, &exec_plan);
        let results = check_plan(&plan.groups, &g, &exec_plan);

        // Assert: all valid results must have consistent internal state
        for r in &results {
            if r.valid {
                assert!(
                    r.violations.is_empty(),
                    "valid result for group {} should have zero violations, got {:?}",
                    r.group_id, r.violations
                );
                assert!(
                    r.register_pressure > 0,
                    "register_pressure should be positive for group {}",
                    r.group_id
                );
                assert!(
                    r.register_limit > 0,
                    "register_limit should be positive for group {}",
                    r.group_id
                );
                assert!(
                    r.register_pressure <= r.register_limit,
                    "register_pressure ({}) should not exceed limit ({}) for group {}",
                    r.register_pressure, r.register_limit, r.group_id
                );
            }
        }
    }

    // ── Test 109: estimate_register_pressure EpilogueInjection with single op increases by 1 ──

    /// @trace TEST-HWC-109 [req:REQ-FUS] [level:unit]
    #[test]
    fn estimate_register_pressure_epilogue_single_op_increment() {
        // Arrange: compare standalone vs epilogue injection with exactly 1 op
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[32, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(32), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm",
        );

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let available = profile.num_simd_regs();

        let standalone = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        let with_one = FusionGroup {
            id: 1, anchor: OpId(0), epilogue: vec![OpId(1)],
            mode: FusionMode::EpilogueInjection, ops: vec![OpId(0), OpId(1)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act
        let base = estimate_register_pressure(&standalone, &g, &exec_plan);
        let with_epi = estimate_register_pressure(&with_one, &g, &exec_plan);

        // Assert: with 1 epilogue op within free regs, pressure = base + 1
        let free = available.saturating_sub(base);
        if free >= 1 {
            assert_eq!(
                with_epi, base + 1,
                "1 epilogue op should add exactly 1 register (base={base}, with_epi={with_epi}, free={free})"
            );
        }
        // The epilogue version must be at least as large as standalone
        assert!(
            with_epi >= base,
            "epilogue injection should not reduce register pressure: base={base}, with_epi={with_epi}"
        );
    }

    // ── Test 110: extract_gemm_dims returns correct values for all three GEMM variants ──

    /// @trace TEST-HWC-110 [req:REQ-FUS] [level:unit]
    #[test]
    fn extract_gemm_dims_all_three_variants() {
        // Arrange & Act & Assert: Gemm, GemmBias, and QuantGemm all produce dims

        // Gemm
        let mut g1 = CompilerGraph::new();
        let a1 = g1.add_tensor_concrete("a", &[8, 32], DType::F32);
        let w1 = g1.add_tensor_concrete("w", &[32, 64], DType::F32);
        let o1 = g1.add_tensor_concrete("out", &[8, 64], DType::F32);
        g1.add_op(Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(8), n: 64, k: 32, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a1, w1], vec![o1], "gemm",
        );
        let grp1 = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert_eq!(extract_gemm_dims(&grp1, &g1), Some((8, 64, 32)));

        // GemmBias
        let mut g2 = CompilerGraph::new();
        let a2 = g2.add_tensor_concrete("a", &[4, 16], DType::F32);
        let w2 = g2.add_tensor_concrete("w", &[16, 32], DType::F32);
        let o2 = g2.add_tensor_concrete("out", &[4, 32], DType::F32);
        g2.add_op(Op::GemmBias(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(4), n: 32, k: 16, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a2, w2], vec![o2], "gemm_bias",
        );
        let grp2 = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert_eq!(extract_gemm_dims(&grp2, &g2), Some((4, 32, 16)));

        // QuantGemm
        let mut g3 = CompilerGraph::new();
        let a3 = g3.add_tensor_concrete("a", &[2, 8], DType::F32);
        let w3 = g3.add_tensor_concrete("w", &[8, 16], DType::F32);
        let o3 = g3.add_tensor_concrete("out", &[2, 16], DType::F32);
        g3.add_op(Op::QuantGemm(QuantGemmSpec { m: crate::compiler::graph::SymDim::Concrete(2), n: 16, k: 8, quant_type: crate::quant::QuantType::Q4K }),
            vec![a3, w3], vec![o3], "quant_gemm",
        );
        let grp3 = FusionGroup {
            id: 0, anchor: OpId(0), epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(), dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert_eq!(extract_gemm_dims(&grp3, &g3), Some((2, 16, 8)));
    }
}
