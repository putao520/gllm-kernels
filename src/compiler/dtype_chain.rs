//! DTYPE-CHAIN: End-to-end dtype chain validation and propagation.
//!
//! REQ-DTYPE-CHAIN-001~005: Verifies dtype preservation from Loader through CompilerGraph
//! to codegen, detects breakpoints, selects dequantization paths, links to kernel selection,
//! and gates compilation on chain validity.

use crate::types::{CompilerError, DType, InferenceError};
use crate::compiler::graph::{CompilerGraph, Op, OpId, TensorId, TensorMeta};
use crate::dispatch::device_profile::DeviceProfile;

// ── REQ-DTYPE-CHAIN-001: DType chain end-to-end tracking ─────────────

/// Legal dtype conversion scenarios (REQ-DTYPE-CHAIN-001 criterion 4).
/// Only these 3 conversions are permitted in the dtype chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalDtypeConversion {
    /// GGUF quantized → JIT dequantization (inside GEMM load microkernel)
    QuantDequant,
    /// BF16/F16 weight → F32 accumulation (inside VDPBF16PS / tensor core instruction)
    WideAccumulate,
    /// Accumulator result → storage writeback (Epilogue last step)
    NarrowWriteback,
}

/// Validate that a dtype conversion is among the 3 legal scenarios.
/// Returns Err for illegal conversions.
pub fn validate_dtype_conversion(
    source: DType,
    target: DType,
    context: &str,
) -> Result<LegalDtypeConversion, InferenceError> {
    // Legal 1: Quantized → F32 dequantization
    if is_quantized_dtype(source) && target == DType::F32 {
        return Ok(LegalDtypeConversion::QuantDequant);
    }
    // Legal 2: BF16/F16 → F32 widening (GEMM accumulation)
    if (source == DType::BF16 || source == DType::F16) && target == DType::F32 {
        return Ok(LegalDtypeConversion::WideAccumulate);
    }
    // Legal 3: F32 → BF16/F16 narrowing (Epilogue writeback)
    if source == DType::F32 && (target == DType::BF16 || target == DType::F16) {
        return Ok(LegalDtypeConversion::NarrowWriteback);
    }
    // Same dtype — no conversion needed
    if source == target {
        return Ok(LegalDtypeConversion::WideAccumulate); // identity
    }
    Err(InferenceError::CompileError(CompilerError::InvalidGraph(format!(
        "DTYPE-CHAIN-001: Illegal dtype conversion {:?} -> {:?} in context '{}'. \
         Only QuantDequant/WideAccumulate/NarrowWriteback are permitted.",
        source, target, context
    ))))
}

/// Check if a DType represents a quantized format.
fn is_quantized_dtype(dt: DType) -> bool {
    matches!(dt, DType::U8 | DType::F8E4M3 | DType::F8E5M2
        | DType::F6E3M2 | DType::F6E2M3 | DType::F4E2M1)
}

// ── REQ-DTYPE-CHAIN-002: DType chain breakpoint detection ───────────

/// A detected breakpoint in the dtype chain.
#[derive(Debug, Clone)]
pub struct DtypeBreakpoint {
    /// The op where the breakpoint was detected.
    pub op_id: OpId,
    /// The op label (for human-readable error messages).
    pub op_label: String,
    /// Input tensor ID with mismatched dtype.
    pub tensor_id: TensorId,
    /// Expected dtype (from producer).
    pub expected_dtype: DType,
    /// Actual dtype (from tensor metadata).
    pub actual_dtype: DType,
    /// Suggested fix.
    pub suggestion: String,
}

/// Detect dtype breakpoints in the CompilerGraph.
///
/// Scans all ops and verifies that input tensor dtypes are compatible
/// with the op's expected dtype. Reports mismatches with fix suggestions.
pub fn detect_dtype_breakpoints(graph: &CompilerGraph) -> Vec<DtypeBreakpoint> {
    let mut breakpoints = Vec::new();

    for op in &graph.ops {
        // Check each input tensor's dtype against the op's expected dtype.
        let expected = op_expected_dtype(op, graph);

        for &input_tid in &op.inputs {
            let Some(tensor) = graph.tensor(input_tid) else { continue };
            let Some(exp) = &expected else { continue };

            if tensor.dtype != *exp && !is_legal_chain_transition(tensor.dtype, *exp) {
                breakpoints.push(DtypeBreakpoint {
                    op_id: op.id,
                    op_label: op.label.clone(),
                    tensor_id: input_tid,
                    expected_dtype: *exp,
                    actual_dtype: tensor.dtype,
                    suggestion: format!(
                        "Insert dtype conversion ({:?}→{:?}) before op '{}', or fix tensor '{}' dtype",
                        tensor.dtype, exp, op.label, tensor.name
                    ),
                });
            }
        }
    }

    breakpoints
}

/// Derive the expected dtype for an op from its OpKind.
fn op_expected_dtype(op: &crate::compiler::graph::CompilerOp, graph: &CompilerGraph) -> Option<DType> {
    match op.op_resolved(graph) {
        Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) => op.op_gemm_dtype(graph),
        Some(Op::QuantGemm(_)) => Some(DType::F32), // QuantGemm always outputs F32
        // Norm ops expect the same dtype as their first input
        _ => None,
    }
}

/// Check if a dtype transition is legal in the chain (same dtype or legal conversion).
fn is_legal_chain_transition(source: DType, target: DType) -> bool {
    if source == target {
        return true;
    }
    validate_dtype_conversion(source, target, "chain transition").is_ok()
}

// ── REQ-DTYPE-CHAIN-003: Dequantization path selection ──────────────

/// Dequantization strategy for a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantStrategy {
    /// Immediate dequantization: convert at load time (for small tensors like norms).
    /// Only used when the tensor is consumed by non-GEMM ops (norms, biases).
    Immediate,
    /// Deferred dequantization: dequant inside GEMM microkernel.
    /// Weight stays in quantized format until the compute kernel reads it.
    /// This is the preferred path for GEMM weight tensors.
    DeferredInKernel,
}

/// Select the optimal dequantization path for a weight tensor.
///
/// Strategy:
/// - Norm/bias tensors (1D, consumed by non-GEMM ops) → Immediate
/// - GEMM weight tensors (2D, consumed by Gemm/QuantGemm) → DeferredInKernel
/// - Other tensors → DeferredInKernel (safe default)
pub fn select_dequant_path(
    tensor: &TensorMeta,
    graph: &CompilerGraph,
) -> DequantStrategy {
    // 1D tensors consumed by norm/bias ops → immediate dequant
    if tensor.shape.len() == 1 {
        // ARCH-JIT-DATA-YIELDS: use tensor.consumers index instead of graph.ops.iter()
        let consumed_by_norm = tensor.consumers.iter()
            .filter_map(|&op_id| graph.op(op_id))
            .any(|op| matches!(op.op_resolved(graph), Some(Op::RmsNorm(_)) | Some(Op::LayerNorm(_))));
        if consumed_by_norm {
            return DequantStrategy::Immediate;
        }
    }

    // 2D tensors consumed by GEMM ops → deferred dequant
    let consumed_by_gemm = tensor.consumers.iter()
        .filter_map(|&op_id| graph.op(op_id))
        .any(|op| matches!(op.op_resolved(graph), Some(Op::Gemm(_)) | Some(Op::GemmBias(_)) | Some(Op::QuantGemm(_))));

    if consumed_by_gemm {
        return DequantStrategy::DeferredInKernel;
    }

    // Default: deferred (safer, matches JIT pipeline)
    DequantStrategy::DeferredInKernel
}

// ── REQ-DTYPE-CHAIN-004: DType chain & kernel selection linkage ─────

/// Compute dtype derived from (TensorMeta.dtype, DeviceProfile).
///
/// This is the dtype used for buffer layout and kernel selection.
/// It represents the actual computation precision after hardware promotion.
///
/// Rules (from SPEC REQ-DTYPE-CHAIN-004):
/// - BF16 model + AVX-512 VDPBF16PS → F32 (BF16×BF16→F32 accumulation)
/// - BF16 model + GPU tensor core WGMMA → F32 (BF16×BF16→F32 accumulation)
/// - BF16 model + AMX TDPBF16PS → F32 (BF16×BF16→F32 accumulation)
/// - F32 model + any hardware → F32
/// - FP16 model + GPU tensor core → F32 (FP16×FP16→F32 accumulation)
/// - Quantized model → F32 (dequant then F32 accumulation)
/// - Future: BF16 residual path may keep compute_dtype = BF16
pub fn derive_compute_dtype(storage_dtype: DType, device: &DeviceProfile) -> DType {
    match storage_dtype {
        // BF16/F16 always widen to F32 on current hardware
        DType::BF16 | DType::F16 => DType::F32,
        // Quantized types always dequant to F32
        DType::U8 | DType::F8E4M3 | DType::F8E5M2
        | DType::F6E3M2 | DType::F6E2M3 | DType::F4E2M1 => DType::F32,
        // F32 stays F32
        DType::F32 => DType::F32,
    }
    // Note: device parameter is reserved for future hardware that supports
    // native BF16 accumulation (e.g. future GPU architectures). Currently
    // all paths result in F32, but the function signature allows future
    // evolution without API break.
}

/// Kernel selection key derived from (storage_dtype, compute_dtype, DeviceProfile).
///
/// Different dtype combinations correspond to different kernel implementations:
/// - (BF16, F32, AVX-512) → VDPBF16PS GEMM microkernel
/// - (BF16, F32, GPU TC) → HMMA.16816 BF16 tensor core
/// - (F32, F32, AVX-512) → F32 FMA GEMM microkernel
/// - (Q4_K, F32, any) → QuantGemm with deferred dequant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelDtypeKey {
    pub storage_dtype: DType,
    pub compute_dtype: DType,
}

impl KernelDtypeKey {
    /// Derive kernel selection key from graph and device profile.
    pub fn from_graph(graph: &CompilerGraph, device: &DeviceProfile) -> Self {
        let storage_dtype = derive_storage_dtype_from_graph(graph);
        let compute_dtype = derive_compute_dtype(storage_dtype, device);
        Self { storage_dtype, compute_dtype }
    }
}

/// Derive storage dtype from graph (most common weight tensor dtype).
fn derive_storage_dtype_from_graph(graph: &CompilerGraph) -> DType {
    let mut f32_count = 0usize;
    let mut bf16_count = 0usize;
    let mut f16_count = 0usize;
    // Skip first input (activation), scan weight inputs.
    for &tid in graph.inputs.iter().skip(1) {
        if let Some(t) = graph.tensors.get(tid.0 as usize) {
            match t.dtype {
                DType::F32 => f32_count += 1,
                DType::BF16 => bf16_count += 1,
                DType::F16 => f16_count += 1,
                _ => {} // Quantized/other types don't affect float storage dtype
            }
        }
    }
    if bf16_count >= f32_count && bf16_count >= f16_count && bf16_count > 0 {
        DType::BF16
    } else if f16_count >= f32_count && f16_count >= bf16_count && f16_count > 0 {
        DType::F16
    } else {
        DType::F32
    }
}

// ── REQ-DTYPE-CHAIN-005: DType chain validation gate ────────────────

/// Result of dtype chain validation.
#[derive(Debug, Clone)]
pub struct DtypeChainValidation {
    /// Whether the chain is valid (no breakpoints).
    pub is_valid: bool,
    /// Number of breakpoints detected.
    pub num_breakpoints: usize,
    /// Breakpoint details (empty if is_valid).
    pub breakpoints: Vec<DtypeBreakpoint>,
    /// Compute dtype derived from (storage_dtype, DeviceProfile).
    pub compute_dtype: DType,
    /// Storage dtype (most common weight dtype).
    pub storage_dtype: DType,
    /// Per-tensor dequantization strategies.
    pub dequant_strategies: Vec<(TensorId, DequantStrategy)>,
    /// Kernel dtype key for selection.
    pub kernel_key: KernelDtypeKey,
}

impl DtypeChainValidation {
    /// Run full dtype chain validation on a CompilerGraph.
    ///
    /// This is the compilation gate: if validation fails, Mega-Kernel
    /// generation is blocked.
    pub fn validate(graph: &CompilerGraph, device: &DeviceProfile) -> Self {
        let storage_dtype = derive_storage_dtype_from_graph(graph);
        let compute_dtype = derive_compute_dtype(storage_dtype, device);
        let kernel_key = KernelDtypeKey::from_graph(graph, device);

        // Detect breakpoints
        let breakpoints = detect_dtype_breakpoints(graph);

        // Compute per-tensor dequant strategies
        let dequant_strategies: Vec<(TensorId, DequantStrategy)> = graph.tensors.iter()
            .filter(|t| is_quantized_dtype(t.dtype))
            .map(|t| (t.id, select_dequant_path(t, graph)))
            .collect();

        let is_valid = breakpoints.is_empty();

        Self {
            is_valid,
            num_breakpoints: breakpoints.len(),
            breakpoints,
            compute_dtype,
            storage_dtype,
            dequant_strategies,
            kernel_key,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, SymDim, Op, GemmSpec, NormSpec, QuantGemmSpec, RopeSpec, AttentionSpec, AttentionGeometry, AttentionMask, SinksSpec, CachedGqaSpec, MlaSpec, DualRopeSpec};

    // ── REQ-DTYPE-CHAIN-001 tests ────────────────────────────────

    #[test]
    fn test_legal_conversion_quant_dequant() {
        let result = validate_dtype_conversion(DType::U8, DType::F32, "test");
        assert_eq!(result.unwrap(), LegalDtypeConversion::QuantDequant);

        let result = validate_dtype_conversion(DType::F8E4M3, DType::F32, "test");
        assert_eq!(result.unwrap(), LegalDtypeConversion::QuantDequant);
    }

    #[test]
    fn test_legal_conversion_wide_accumulate() {
        let result = validate_dtype_conversion(DType::BF16, DType::F32, "test");
        assert_eq!(result.unwrap(), LegalDtypeConversion::WideAccumulate);

        let result = validate_dtype_conversion(DType::F16, DType::F32, "test");
        assert_eq!(result.unwrap(), LegalDtypeConversion::WideAccumulate);
    }

    #[test]
    fn test_legal_conversion_narrow_writeback() {
        let result = validate_dtype_conversion(DType::F32, DType::BF16, "test");
        assert_eq!(result.unwrap(), LegalDtypeConversion::NarrowWriteback);

        let result = validate_dtype_conversion(DType::F32, DType::F16, "test");
        assert_eq!(result.unwrap(), LegalDtypeConversion::NarrowWriteback);
    }

    #[test]
    fn test_illegal_conversion() {
        let result = validate_dtype_conversion(DType::BF16, DType::F16, "test");
        assert!(result.is_err());

        let result = validate_dtype_conversion(DType::F32, DType::U8, "test");
        assert!(result.is_err());

        let result = validate_dtype_conversion(DType::F8E4M3, DType::BF16, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_same_dtype_identity() {
        let result = validate_dtype_conversion(DType::F32, DType::F32, "test");
        assert!(result.is_ok());

        let result = validate_dtype_conversion(DType::BF16, DType::BF16, "test");
        assert!(result.is_ok());
    }

    // ── REQ-DTYPE-CHAIN-002 tests ────────────────────────────────

    #[test]
    fn test_detect_breakpoints_empty_graph() {
        let graph = CompilerGraph::new();
        let breakpoints = detect_dtype_breakpoints(&graph);
        assert!(breakpoints.is_empty());
    }

    #[test]
    fn test_detect_breakpoints_consistent_dtype() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 64], dt);
        let b = g.add_tensor_concrete("b", &[64, 128], dt);
        let c = g.add_tensor_concrete("c", &[1, 128], dt);
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![a, b], vec![c], "gemm"
        );
        let breakpoints = detect_dtype_breakpoints(&g);
        assert!(breakpoints.is_empty(), "Consistent dtype should have no breakpoints");
    }

    #[test]
    fn test_detect_breakpoints_dtype_mismatch() {
        let mut g = CompilerGraph::new();
        // Input tensor is BF16, but op expects F32 — this is a LEGAL transition (WideAccumulate)
        let a = g.add_tensor_concrete("a", &[1, 64], DType::BF16);
        let b = g.add_tensor_concrete("b", &[64, 128], DType::BF16);
        let c = g.add_tensor_concrete("c", &[1, 128], DType::F32);
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, b], vec![c], "gemm"
        );
        // BF16→F32 is a legal WideAccumulate transition — should not be a breakpoint
        let breakpoints = detect_dtype_breakpoints(&g);
        assert!(breakpoints.is_empty(), "Legal BF16→F32 transition should not be a breakpoint");
    }

    // ── REQ-DTYPE-CHAIN-003 tests ────────────────────────────────

    #[test]
    fn test_dequant_strategy_norm_tensor() {
        let mut g = CompilerGraph::new();
        // Norm weight is 1D, consumed by RmsNorm as second input
        let x = g.add_tensor_concrete("x", &[1, 64], DType::F32);
        let norm_w = g.add_tensor_concrete("norm_w", &[64], DType::U8);
        let out = g.add_tensor_concrete("out", &[1, 64], DType::F32);
        g.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-6, dtype: DType::F32, has_weight: true }), vec![x, norm_w], vec![out], "norm");

        let tensor = g.tensor(norm_w).unwrap();
        let strategy = select_dequant_path(tensor, &g);
        assert_eq!(strategy, DequantStrategy::Immediate, "1D tensor consumed by norm → Immediate");
    }

    #[test]
    fn test_dequant_strategy_gemm_weight() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = g.add_tensor_concrete("b", &[64, 128], DType::U8);
        let c = g.add_tensor_concrete("c", &[1, 128], DType::F32);
        g.add_op(Op::QuantGemm(QuantGemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, quant_type: crate::quant::QuantType::Q4K }),
            vec![a, b], vec![c], "qgemm"
        );

        let tensor = g.tensor(b).unwrap();
        let strategy = select_dequant_path(tensor, &g);
        assert_eq!(strategy, DequantStrategy::DeferredInKernel, "2D weight consumed by GEMM → DeferredInKernel");
    }

    #[test]
    fn test_dequant_strategy_1d_bias() {
        let mut g = CompilerGraph::new();
        let bias = g.add_tensor_concrete("bias", &[128], DType::U8);
        // Bias consumed by GemmBias (not a norm)
        let a = g.add_tensor_concrete("a", &[1, 64], DType::F32);
        let b = g.add_tensor_concrete("b", &[64, 128], DType::BF16);
        let c = g.add_tensor_concrete("c", &[1, 128], DType::F32);
        g.add_op(Op::GemmBias(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: DType::F32, trans_b: false, has_bias: true }),
            vec![a, b, bias], vec![c], "gemm_bias"
        );

        let tensor = g.tensor(bias).unwrap();
        let strategy = select_dequant_path(tensor, &g);
        assert_eq!(strategy, DequantStrategy::DeferredInKernel, "1D tensor consumed by GEMM → DeferredInKernel");
    }

    // ── REQ-DTYPE-CHAIN-004 tests ────────────────────────────────

    #[test]
    fn test_derive_compute_dtype_bf16_to_f32() {
        let device = DeviceProfile::detect();
        assert_eq!(derive_compute_dtype(DType::BF16, &device), DType::F32);
    }

    #[test]
    fn test_derive_compute_dtype_f16_to_f32() {
        let device = DeviceProfile::detect();
        assert_eq!(derive_compute_dtype(DType::F16, &device), DType::F32);
    }

    #[test]
    fn test_derive_compute_dtype_f32_identity() {
        let device = DeviceProfile::detect();
        assert_eq!(derive_compute_dtype(DType::F32, &device), DType::F32);
    }

    #[test]
    fn test_derive_compute_dtype_quant_to_f32() {
        let device = DeviceProfile::detect();
        assert_eq!(derive_compute_dtype(DType::U8, &device), DType::F32);
        assert_eq!(derive_compute_dtype(DType::F8E4M3, &device), DType::F32);
        assert_eq!(derive_compute_dtype(DType::F4E2M1, &device), DType::F32);
    }

    #[test]
    fn test_kernel_dtype_key_from_graph() {
        let mut g = CompilerGraph::new();
        let dt = DType::BF16;
        let a = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w = g.add_tensor_concrete("weight", &[512, 512], dt);
        let out = g.add_tensor_concrete("out", &[1, 512], DType::F32);
        g.inputs = vec![a, w];
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm"
        );

        let device = DeviceProfile::detect();
        let key = KernelDtypeKey::from_graph(&g, &device);
        assert_eq!(key.storage_dtype, DType::BF16);
        assert_eq!(key.compute_dtype, DType::F32);
    }

    // ── REQ-DTYPE-CHAIN-005 tests ────────────────────────────────

    #[test]
    fn test_dtype_chain_validation_valid_graph() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 64], dt);
        let w = g.add_tensor_concrete("weight", &[64, 128], dt);
        let out = g.add_tensor_concrete("out", &[1, 128], dt);
        g.inputs = vec![a, w];
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 128, k: 64, dtype: dt, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm"
        );

        let device = DeviceProfile::detect();
        let validation = DtypeChainValidation::validate(&g, &device);
        assert!(validation.is_valid, "F32 graph should be valid");
        assert_eq!(validation.num_breakpoints, 0);
        assert_eq!(validation.compute_dtype, DType::F32);
        assert_eq!(validation.storage_dtype, DType::F32);
    }

    #[test]
    fn test_dtype_chain_validation_bf16_model() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w = g.add_tensor_concrete("weight", &[512, 512], DType::BF16);
        let out = g.add_tensor_concrete("out", &[1, 512], DType::F32);
        g.inputs = vec![a, w];
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![a, w], vec![out], "gemm"
        );

        let device = DeviceProfile::detect();
        let validation = DtypeChainValidation::validate(&g, &device);
        assert!(validation.is_valid, "BF16 model with F32 compute should be valid");
        assert_eq!(validation.compute_dtype, DType::F32);
        assert_eq!(validation.storage_dtype, DType::BF16);
        assert_eq!(validation.kernel_key, KernelDtypeKey {
            storage_dtype: DType::BF16,
            compute_dtype: DType::F32,
        });
    }

    #[test]
    fn test_dtype_chain_validation_separates_weight_and_activation() {
        let mut g = CompilerGraph::new();
        // Norm weight is F32, GEMM weight is BF16 — mixed dtypes
        let input = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let norm_w = g.add_tensor_concrete("norm_w", &[512], DType::F32);
        let gemm_w = g.add_tensor_concrete("gemm_w", &[512, 2048], DType::BF16);
        g.inputs = vec![input, norm_w, gemm_w];

        let normed = g.add_tensor_concrete("normed", &[1, 512], DType::F32);
        g.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-6, dtype: DType::F32, has_weight: true }), vec![input, norm_w], vec![normed], "norm");

        let out = g.add_tensor_concrete("out", &[1, 2048], DType::F32);
        g.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 2048, k: 512, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![normed, gemm_w], vec![out], "gemm"
        );

        let device = DeviceProfile::detect();
        let validation = DtypeChainValidation::validate(&g, &device);
        assert!(validation.is_valid, "Mixed F32 norm + BF16 GEMM should be valid");
        assert_eq!(validation.storage_dtype, DType::BF16, "BF16 GEMM weight dominates storage_dtype");
        assert_eq!(validation.compute_dtype, DType::F32, "compute_dtype derived from (BF16, DeviceProfile)");
    }
}
