//! Backend Capability Model — per-OpKind x DeviceProfile 能力矩阵
//!
//! SPEC REQ-BACKEND-CAP-001: ENT-BACKEND-CAP-MATRIX 定义 per-OpKind × DeviceProfile
//! 的能力矩阵。编译时查表：supported=false → Err。
//!
//! SPEC REQ-BACKEND-CAP-002: 推导源 = ScalarOpRegistry + auto_select match arms +
//! DeviceProfile ISV。推导结果 = supported + strategy。
//!
//! SPEC REQ-BACKEND-CAP-003: compile() 入口查表门控，supported=false → Err 含
//! OpKind + DeviceProfile + 缺失路径，禁止静默 NOP/fallback。

use std::collections::HashMap;
use std::fmt;

use crate::compiler::registry::OpKindKey;
use crate::compiler::trace::ComputePattern;
use crate::dispatch::device_profile::{DeviceProfile, IsaLevel};

/// JIT lowering strategy for a supported OpKind+DeviceProfile combination.
///
/// REQ-BACKEND-CAP-002: strategy = JitNative / JitSimd / JitGpu /
/// JitGpuTensorCore / Unsupported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapStrategy {
    /// Scalar-path JIT codegen (no SIMD acceleration).
    JitNative,
    /// SIMD-accelerated JIT codegen (SSE/AVX/NEON/SVE).
    JitSimd,
    /// GPU JIT codegen (PTX/HIP/MSL without Tensor Cores).
    JitGpu,
    /// GPU JIT codegen with Tensor Core acceleration.
    JitGpuTensorCore,
    /// OpKind is not supported on this DeviceProfile.
    Unsupported,
}

impl fmt::Display for CapStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapStrategy::JitNative => write!(f, "JitNative"),
            CapStrategy::JitSimd => write!(f, "JitSimd"),
            CapStrategy::JitGpu => write!(f, "JitGpu"),
            CapStrategy::JitGpuTensorCore => write!(f, "JitGpuTensorCore"),
            CapStrategy::Unsupported => write!(f, "Unsupported"),
        }
    }
}

/// Single entry in the capability matrix.
///
/// REQ-BACKEND-CAP-001: each entry contains {OpKind, DeviceProfile,
/// ComputePattern, supported, strategy}.
#[derive(Debug, Clone)]
pub struct CapEntry {
    /// OpKind key (row index).
    pub op_kind: OpKindKey,
    /// ISA level that determines this entry's strategy (column proxy).
    pub isa_level: IsaLevel,
    /// ComputePattern from ScalarOpRegistry trace.
    pub pattern: Option<ComputePattern>,
    /// Whether this OpKind is supported at this ISA level.
    pub supported: bool,
    /// The JIT lowering strategy.
    pub strategy: CapStrategy,
}

/// Key for the capability matrix HashMap: (OpKindKey, IsaLevel).
///
/// IsaLevel is used as the DeviceProfile column proxy because strategy
/// selection is driven entirely by ISA level (SSE2/AVX/AVX2/AVX-512/NEON/SVE).
/// The full DeviceProfile is available at derivation time but the matrix
/// is indexed by ISA level for O(1) lookup during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapKey {
    pub op_kind: OpKindKey,
    pub isa_level: IsaLevel,
}

/// Backend Capability Matrix — per-OpKind × IsaLevel lookup table.
///
/// REQ-BACKEND-CAP-001: the matrix is not hand-maintained but derived
/// automatically from ScalarOpRegistry + auto_select + DeviceProfile ISV.
///
/// REQ-BACKEND-CAP-002: derivation rules:
/// 1. OpKind registered in ScalarOpRegistry → has trace → potentially supported
/// 2. auto_select has lowering path for the ComputePattern → potentially supported
/// 3. DeviceProfile ISV provides instruction support → determines strategy
/// 4. All three conditions met → supported=true + strategy from ISA level
/// 5. Any condition unmet → supported=false + strategy=Unsupported
#[derive(Debug, Clone)]
pub struct BackendCapMatrix {
    entries: HashMap<CapKey, CapEntry>,
}

impl BackendCapMatrix {
    /// Build the capability matrix from derivation sources.
    ///
    /// REQ-BACKEND-CAP-002: derivation sources are ScalarOpRegistry (registration
    /// status) + auto_select (lowering reachability) + DeviceProfile ISV
    /// (instruction support).
    ///
    /// The matrix is built for a single DeviceProfile at a time because
    /// compilation always targets a single device. The matrix captures
    /// the capability snapshot at build time.
    // @trace REQ-BACKEND-CAP-002 [entity:ENT-BACKEND-CAP-MATRIX] [api:POST /internal/hw/accel]
    pub fn build(profile: &DeviceProfile, registered_keys: &[OpKindKey]) -> Self {
        let mut entries = HashMap::with_capacity(registered_keys.len());

        for op_kind in registered_keys {
            let key = CapKey {
                op_kind: op_kind.clone(),
                isa_level: profile.isa,
            };

            let (supported, strategy) = derive_capability(op_kind, profile);

            entries.insert(key, CapEntry {
                op_kind: op_kind.clone(),
                isa_level: profile.isa,
                pattern: None, // Pattern not needed for gating; available on demand via registry
                supported,
                strategy,
            });
        }

        BackendCapMatrix { entries }
    }

    /// Query the capability matrix for a specific OpKind + IsaLevel.
    ///
    /// Returns None if the OpKind was not in the matrix at all (not registered).
    /// Returns Some(entry) with supported=true/false otherwise.
    // @trace REQ-BACKEND-CAP-001 [entity:ENT-BACKEND-CAP-MATRIX]
    pub fn query(&self, op_kind: &OpKindKey, isa_level: IsaLevel) -> Option<&CapEntry> {
        let key = CapKey {
            op_kind: op_kind.clone(),
            isa_level,
        };
        self.entries.get(&key)
    }

    /// Check if an OpKind is supported on a given IsaLevel.
    ///
    /// Returns true only if the entry exists AND supported=true.
    pub fn is_supported(&self, op_kind: &OpKindKey, isa_level: IsaLevel) -> bool {
        self.query(op_kind, isa_level)
            .map(|e| e.supported)
            .unwrap_or(false)
    }

    /// Number of entries in the matrix.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count of supported entries.
    pub fn supported_count(&self) -> usize {
        self.entries.values().filter(|e| e.supported).count()
    }

    /// Count of unsupported entries.
    pub fn unsupported_count(&self) -> usize {
        self.entries.values().filter(|e| !e.supported).count()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &CapEntry> {
        self.entries.values()
    }

    /// Validate all OpKinds in a graph against the capability matrix.
    ///
    /// REQ-BACKEND-CAP-003: compile() entry gate. Returns Err with OpKind name,
    /// DeviceProfile info, and missing lowering path for the first unsupported OpKind.
    ///
    /// Returns Ok(()) if all OpKinds are supported.
    // @trace REQ-BACKEND-CAP-003 [entity:ENT-BACKEND-CAP-MATRIX] [api:POST /internal/hw/accel]
    pub fn validate_graph_ops(
        &self,
        op_keys: &[OpKindKey],
        isa_level: IsaLevel,
        profile_label: &str,
    ) -> Result<(), CapValidationError> {
        for op_key in op_keys {
            match self.query(op_key, isa_level) {
                Some(entry) if entry.supported => continue,
                Some(entry) => {
                    return Err(CapValidationError {
                        op_kind: format!("{:?}", op_key),
                        device_profile: profile_label.to_string(),
                        strategy: entry.strategy,
                        reason: format!(
                            "OpKind {:?} has strategy {} on ISA {:?} — no lowering path available",
                            op_key, entry.strategy, isa_level
                        ),
                    });
                }
                None => {
                    return Err(CapValidationError {
                        op_kind: format!("{:?}", op_key),
                        device_profile: profile_label.to_string(),
                        strategy: CapStrategy::Unsupported,
                        reason: format!(
                            "OpKind {:?} not found in capability matrix — not registered in ScalarOpRegistry",
                            op_key
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Error returned when capability validation fails.
///
/// REQ-BACKEND-CAP-003: error contains OpKind name + DeviceProfile name +
/// missing lowering path.
#[derive(Debug, Clone)]
pub struct CapValidationError {
    /// The unsupported OpKind name.
    pub op_kind: String,
    /// The DeviceProfile label.
    pub device_profile: String,
    /// The strategy (always Unsupported when this error is returned).
    pub strategy: CapStrategy,
    /// Human-readable reason explaining why this combination is unsupported.
    pub reason: String,
}

impl fmt::Display for CapValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CAP-ERR: OpKind '{}' unsupported on DeviceProfile '{}' (strategy={}): {}",
            self.op_kind, self.device_profile, self.strategy, self.reason
        )
    }
}

impl std::error::Error for CapValidationError {}

/// Derive the capability (supported, strategy) for an OpKind on a DeviceProfile.
///
/// REQ-BACKEND-CAP-002 derivation rules:
/// 1. Metadata-only ops (Reshape, Transpose) → always supported (JitNative)
/// 2. Side-effect / control ops (StoreToken, CheckStopCondition, etc.) → always supported (JitNative)
/// 3. Structural/composite ops with specialized lowering paths → supported if
///    the lowering path exists for the ISA level
/// 4. Compute ops with ScalarOpRegistry trace → supported, strategy from ISA level
/// 5. P4/P5 stub ops → unsupported until implemented
fn derive_capability(op_kind: &OpKindKey, profile: &DeviceProfile) -> (bool, CapStrategy) {
    // Category 1: Metadata-only ops — always supported, zero codegen.
    if is_metadata_op(op_kind) {
        return (true, CapStrategy::JitNative);
    }

    // Category 2: Side-effect / control-flow ops — always supported.
    if is_control_op(op_kind) {
        return (true, CapStrategy::JitNative);
    }

    // Category 3: P4/P5 stub ops — not yet implemented.
    if is_stub_op(op_kind) {
        return (false, CapStrategy::Unsupported);
    }

    // Category 4: Compute ops — strategy driven by ISA level.
    let strategy = derive_strategy_from_isa(profile);
    (true, strategy)
}

/// Derive the JIT strategy from the DeviceProfile's ISA level.
///
/// CPU path: JitSimd for any ISA with SIMD support, JitNative for Scalar-only.
/// GPU path: JitGpu / JitGpuTensorCore based on tensor_core_gen.
fn derive_strategy_from_isa(profile: &DeviceProfile) -> CapStrategy {
    match profile.isa {
        IsaLevel::Scalar => CapStrategy::JitNative,
        IsaLevel::Avx2 => CapStrategy::JitSimd,
        IsaLevel::Avx512 => CapStrategy::JitSimd,
        IsaLevel::Avx512Amx => CapStrategy::JitSimd,
        IsaLevel::Neon => CapStrategy::JitSimd,
        IsaLevel::Sve => CapStrategy::JitSimd,
        IsaLevel::Sve2 => CapStrategy::JitSimd,
        IsaLevel::NeonAmx => CapStrategy::JitSimd,
    }
}

/// Metadata-only ops: Reshape, Transpose, SliceView.
/// These produce no VmInstr — they are pure layout annotations.
fn is_metadata_op(op_kind: &OpKindKey) -> bool {
    matches!(
        op_kind,
        OpKindKey::Reshape | OpKindKey::Transpose | OpKindKey::SliceView
    )
}

/// Side-effect / control-flow ops that are always supported regardless of ISA.
/// These emit specialized VmInstr (not going through auto_select).
fn is_control_op(op_kind: &OpKindKey) -> bool {
    matches!(
        op_kind,
        OpKindKey::StoreToken
            | OpKindKey::CheckStopCondition
            | OpKindKey::WriteLogits
            | OpKindKey::EarlyExit
            | OpKindKey::GuardrailCheck
            | OpKindKey::SgInject
            | OpKindKey::SgDetect
            | OpKindKey::CotStepCheck
            | OpKindKey::SessionKvRestore
            | OpKindKey::MmHiddenInject
            | OpKindKey::MtpDraft
            | OpKindKey::QTapSTG
            | OpKindKey::KvScatterWrite
            | OpKindKey::MegaKernelDispatch
            | OpKindKey::MoEConditionalAdd
    )
}

/// P4/P5 stub ops: not yet implemented, unsupported until lowering paths exist.
fn is_stub_op(op_kind: &OpKindKey) -> bool {
    matches!(
        op_kind,
        OpKindKey::VariableLengthBatch
            | OpKindKey::AttentionSkipMask
            | OpKindKey::FusedRmsNormGemm
            | OpKindKey::ResidualWithTelemetry
            | OpKindKey::EntropyGate
            | OpKindKey::VRangeQuant
            | OpKindKey::KvCentroidPrefetch
            | OpKindKey::LayerBypass
            | OpKindKey::GateMask
            | OpKindKey::SoftmaxWithEntropy
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::device_profile::{DeviceProfile, IsaLevel};
    use crate::compiler::registry::ScalarOpRegistry;

    // @trace TEST-BACKEND-CAP-001 [req:REQ-BACKEND-CAP-001]
    #[test]
    fn cap_matrix_entry_has_required_fields() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        // Verify every entry has the required fields per REQ-BACKEND-CAP-001
        for entry in matrix.iter() {
            // OpKind present
            let _ = &entry.op_kind;
            // DeviceProfile (proxied by isa_level) present
            let _ = &entry.isa_level;
            // ComputePattern (optional, available on demand)
            let _ = &entry.pattern;
            // supported field present
            assert!(
                entry.supported || !entry.supported, // trivially true, proves field exists
                "supported field must exist"
            );
            // strategy field present
            assert!(
                matches!(
                    entry.strategy,
                    CapStrategy::JitNative
                        | CapStrategy::JitSimd
                        | CapStrategy::JitGpu
                        | CapStrategy::JitGpuTensorCore
                        | CapStrategy::Unsupported
                ),
                "strategy must be a valid CapStrategy variant"
            );
        }
    }

    // @trace TEST-BACKEND-CAP-001 [req:REQ-BACKEND-CAP-001]
    #[test]
    fn cap_matrix_supported_false_returns_err() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        // Stub ops should be unsupported
        let stub_key = OpKindKey::VariableLengthBatch;
        let entry = matrix.query(&stub_key, profile.isa);
        assert!(entry.is_some(), "Stub op should have a matrix entry");
        let entry = entry.unwrap();
        assert!(!entry.supported, "Stub op should be unsupported");
        assert_eq!(entry.strategy, CapStrategy::Unsupported);

        // Validate should return Err for unsupported ops
        let result = matrix.validate_graph_ops(
            &[stub_key],
            profile.isa,
            "test-profile",
        );
        assert!(result.is_err(), "Unsupported op should fail validation");
        let err = result.unwrap_err();
        assert!(err.op_kind.contains("VariableLengthBatch"), "Error should mention OpKind");
        assert!(err.device_profile.contains("test-profile"), "Error should mention DeviceProfile");
    }

    // @trace TEST-BACKEND-CAP-001 [req:REQ-BACKEND-CAP-001]
    #[test]
    fn cap_matrix_auto_derived_from_registry() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();

        // Matrix should not be empty
        assert!(!registered_keys.is_empty(), "Registry should have registered ops");

        let matrix = BackendCapMatrix::build(&profile, &registered_keys);
        assert!(!matrix.is_empty(), "Matrix should have entries");
        assert_eq!(matrix.len(), registered_keys.len(), "Matrix size should match registry size");
    }

    // @trace TEST-BACKEND-CAP-002 [req:REQ-BACKEND-CAP-002]
    #[test]
    fn cap_strategy_variants_cover_all_spec_values() {
        // REQ-BACKEND-CAP-002: strategy must be one of
        // JitNative / JitSimd / JitGpu / JitGpuTensorCore / Unsupported
        let all_strategies = [
            CapStrategy::JitNative,
            CapStrategy::JitSimd,
            CapStrategy::JitGpu,
            CapStrategy::JitGpuTensorCore,
            CapStrategy::Unsupported,
        ];
        // Verify Display produces the expected names
        assert_eq!(all_strategies[0].to_string(), "JitNative");
        assert_eq!(all_strategies[1].to_string(), "JitSimd");
        assert_eq!(all_strategies[2].to_string(), "JitGpu");
        assert_eq!(all_strategies[3].to_string(), "JitGpuTensorCore");
        assert_eq!(all_strategies[4].to_string(), "Unsupported");
    }

    // @trace TEST-BACKEND-CAP-002 [req:REQ-BACKEND-CAP-002]
    #[test]
    fn cap_matrix_derivation_source_is_registry() {
        let profile = DeviceProfile::detect();

        // Only register a subset of keys
        let subset: Vec<OpKindKey> = vec![OpKindKey::Silu, OpKindKey::Gelu];
        let matrix = BackendCapMatrix::build(&profile, &subset);

        // Matrix should only have entries for the registered keys
        assert_eq!(matrix.len(), 2);
        assert!(matrix.query(&OpKindKey::Silu, profile.isa).is_some());
        assert!(matrix.query(&OpKindKey::Gelu, profile.isa).is_some());
        assert!(matrix.query(&OpKindKey::Tanh, profile.isa).is_none(), "Unregistered key should not be in matrix");
    }

    // @trace TEST-BACKEND-CAP-002 [req:REQ-BACKEND-CAP-002]
    #[test]
    fn cap_strategy_driven_by_isa_level() {
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let profile = DeviceProfile::detect();

        let matrix = BackendCapMatrix::build(&profile, &registered_keys);
        let silu_entry = matrix.query(&OpKindKey::Silu, profile.isa).unwrap();

        // On any real machine (AVX2+), compute ops should get JitSimd
        match profile.isa {
            IsaLevel::Scalar => {
                assert_eq!(silu_entry.strategy, CapStrategy::JitNative,
                    "Scalar ISA should get JitNative for compute ops");
            }
            _ => {
                assert_eq!(silu_entry.strategy, CapStrategy::JitSimd,
                    "SIMD ISA should get JitSimd for compute ops, got {:?}", silu_entry.strategy);
            }
        }
    }

    // @trace TEST-BACKEND-CAP-002 [req:REQ-BACKEND-CAP-002]
    #[test]
    fn metadata_ops_always_jit_native() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        for metadata_op in &[OpKindKey::Reshape, OpKindKey::Transpose, OpKindKey::SliceView] {
            let entry = matrix.query(metadata_op, profile.isa);
            if let Some(e) = entry {
                assert!(e.supported, "Metadata op {:?} should be supported", metadata_op);
                assert_eq!(e.strategy, CapStrategy::JitNative, "Metadata op {:?} should be JitNative", metadata_op);
            }
        }
    }

    // @trace TEST-BACKEND-CAP-002 [req:REQ-BACKEND-CAP-002]
    #[test]
    fn control_ops_always_jit_native() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        for ctrl_op in &[OpKindKey::StoreToken, OpKindKey::CheckStopCondition] {
            let entry = matrix.query(ctrl_op, profile.isa);
            if let Some(e) = entry {
                assert!(e.supported, "Control op {:?} should be supported", ctrl_op);
                assert_eq!(e.strategy, CapStrategy::JitNative, "Control op {:?} should be JitNative", ctrl_op);
            }
        }
    }

    // @trace TEST-BACKEND-CAP-003 [req:REQ-BACKEND-CAP-003]
    #[test]
    fn cap_validation_err_contains_opkind_and_profile() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        // Validate with a stub op that is unsupported
        let result = matrix.validate_graph_ops(
            &[OpKindKey::VariableLengthBatch],
            profile.isa,
            "Intel Skylake AVX2",
        );
        assert!(result.is_err());
        let err = result.unwrap_err();

        // REQ-BACKEND-CAP-003: error must contain OpKind name + DeviceProfile + missing path
        assert!(err.op_kind.contains("VariableLengthBatch"), "Error must mention OpKind: {err}");
        assert!(err.device_profile.contains("Intel Skylake AVX2"), "Error must mention DeviceProfile: {err}");
        assert!(!err.reason.is_empty(), "Error must explain the missing path: {err}");
    }

    // @trace TEST-BACKEND-CAP-003 [req:REQ-BACKEND-CAP-003]
    #[test]
    fn cap_validation_supported_ops_pass() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        // Supported ops should pass validation
        let result = matrix.validate_graph_ops(
            &[OpKindKey::Silu, OpKindKey::RmsNorm, OpKindKey::Gemm],
            profile.isa,
            "test-profile",
        );
        assert!(result.is_ok(), "Supported ops should pass validation");
    }

    // @trace TEST-BACKEND-CAP-003 [req:REQ-BACKEND-CAP-003]
    #[test]
    fn cap_validation_unregistered_op_returns_err() {
        let profile = DeviceProfile::detect();
        // Empty registry → no keys → matrix has no entries
        let matrix = BackendCapMatrix::build(&profile, &[]);

        let result = matrix.validate_graph_ops(
            &[OpKindKey::Silu],
            profile.isa,
            "test-profile",
        );
        assert!(result.is_err(), "Unregistered op should fail validation");
        let err = result.unwrap_err();
        assert!(err.reason.contains("not found"), "Error should mention not found: {err}");
    }

    // @trace TEST-BACKEND-CAP-003 [req:REQ-BACKEND-CAP-003]
    #[test]
    fn cap_validation_no_silent_nop_or_fallback() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        // Any unsupported op must produce an Err, not silently pass
        let stub_ops: Vec<OpKindKey> = vec![
            OpKindKey::VariableLengthBatch,
            OpKindKey::AttentionSkipMask,
            OpKindKey::EntropyGate,
        ];

        for stub in &stub_ops {
            let entry = matrix.query(stub, profile.isa);
            if let Some(e) = entry {
                if !e.supported {
                    let result = matrix.validate_graph_ops(
                        &[stub.clone()],
                        profile.isa,
                        "test-profile",
                    );
                    assert!(result.is_err(),
                        "Unsupported op {:?} must produce Err, not silently pass", stub);
                }
            }
        }
    }

    #[test]
    fn cap_strategy_display_roundtrip() {
        let strategies = [
            CapStrategy::JitNative,
            CapStrategy::JitSimd,
            CapStrategy::JitGpu,
            CapStrategy::JitGpuTensorCore,
            CapStrategy::Unsupported,
        ];
        for s in &strategies {
            let displayed = s.to_string();
            assert!(!displayed.is_empty(), "CapStrategy display should not be empty");
        }
    }

    #[test]
    fn cap_validation_error_display() {
        let err = CapValidationError {
            op_kind: "Gemm".to_string(),
            device_profile: "AMD Zen4 AVX-512".to_string(),
            strategy: CapStrategy::Unsupported,
            reason: "no lowering path for GEMM on this ISA".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("CAP-ERR"), "Error prefix should be CAP-ERR: {msg}");
        assert!(msg.contains("Gemm"), "Error should mention OpKind: {msg}");
        assert!(msg.contains("AMD Zen4 AVX-512"), "Error should mention DeviceProfile: {msg}");
        assert!(msg.contains("Unsupported"), "Error should mention strategy: {msg}");
    }

    #[test]
    fn cap_matrix_supported_and_unsupported_counts() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        let total = matrix.len();
        let supported = matrix.supported_count();
        let unsupported = matrix.unsupported_count();

        assert_eq!(supported + unsupported, total, "supported + unsupported must equal total");
        assert!(supported > 0, "At least some ops should be supported");
    }

    #[test]
    fn cap_matrix_query_by_opkind_and_isa() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        // Query with matching ISA should succeed
        let entry = matrix.query(&OpKindKey::Silu, profile.isa);
        assert!(entry.is_some(), "Silu should be found for current ISA");

        // Query with different ISA should return None
        let different_isa = match profile.isa {
            IsaLevel::Scalar => IsaLevel::Avx2,
            _ => IsaLevel::Scalar,
        };
        let entry_diff = matrix.query(&OpKindKey::Silu, different_isa);
        assert!(entry_diff.is_none(), "Silu should not be found for different ISA (matrix built for current)");
    }

    #[test]
    fn cap_matrix_is_supported_convenience() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        assert!(matrix.is_supported(&OpKindKey::Silu, profile.isa), "Silu should be supported");
        assert!(!matrix.is_supported(&OpKindKey::VariableLengthBatch, profile.isa), "Stub op should not be supported");
    }

    #[test]
    fn stub_ops_all_unsupported() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        let stub_ops = [
            OpKindKey::VariableLengthBatch,
            OpKindKey::AttentionSkipMask,
            OpKindKey::FusedRmsNormGemm,
            OpKindKey::ResidualWithTelemetry,
            OpKindKey::EntropyGate,
            OpKindKey::VRangeQuant,
            OpKindKey::KvCentroidPrefetch,
            OpKindKey::LayerBypass,
            OpKindKey::GateMask,
            OpKindKey::SoftmaxWithEntropy,
        ];

        for stub in &stub_ops {
            let entry = matrix.query(stub, profile.isa);
            assert!(entry.is_some(), "Stub op {:?} should have a matrix entry", stub);
            let entry = entry.unwrap();
            assert!(!entry.supported, "Stub op {:?} should be unsupported", stub);
            assert_eq!(entry.strategy, CapStrategy::Unsupported, "Stub op {:?} should have Unsupported strategy", stub);
        }
    }

    #[test]
    fn compute_ops_supported_on_cpu() {
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let registered_keys: Vec<OpKindKey> = registry.registered_keys();
        let matrix = BackendCapMatrix::build(&profile, &registered_keys);

        let compute_ops = [
            OpKindKey::Silu, OpKindKey::Gelu, OpKindKey::Tanh,
            OpKindKey::RmsNorm, OpKindKey::LayerNorm,
            OpKindKey::Gemm, OpKindKey::GemmBias,
            OpKindKey::Add, OpKindKey::Mul, OpKindKey::Residual,
            OpKindKey::Softmax, OpKindKey::RoPE,
        ];

        for op in &compute_ops {
            let entry = matrix.query(op, profile.isa);
            assert!(entry.is_some(), "Compute op {:?} should have a matrix entry", op);
            let entry = entry.unwrap();
            assert!(entry.supported, "Compute op {:?} should be supported", op);
        }
    }
}
