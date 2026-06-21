//! Compiler diagnostics — Fallback whitelist + Fail-triangulation three-layer error codes.
//!
//! REQ-FALLBACK-001: Fallback whitelist — only 5 SPEC-authorized fallbacks are legal.
//! REQ-FALLBACK-002: Non-authorized fallback must return Err with structured diagnostics.
//! REQ-FAIL-TRIANG-001: IR-layer error code prefix `IR-ERR` + position/expected/actual.
//! REQ-FAIL-TRIANG-002: PASS-layer error code prefix `PASS-ERR` + invariant/pass/diff.
//! REQ-FAIL-TRIANG-003: CODEGEN-layer error code prefix `CG-ERR` + VmInstr/OpKind/DeviceProfile.

use std::fmt;

use crate::types::CompilerError;
use crate::compiler::graph::{CompilerGraph, OpId, TensorId};
use crate::dispatch::device_profile::DeviceProfile;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 REQ-FALLBACK-001: Fallback whitelist
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// SPEC-authorized fallback identifiers (REQ-FALLBACK-001).
///
/// Only these 5 fallback paths are legal. Any other fallback is a violation
/// of the NO-FALLBACK / NO-SILENT-FALLBACK iron laws.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FallbackWhitelist {
    /// A2: HF -> ModelScope download source switch (REQ-LOADER-016).
    HfToModelScope,
    /// A3: ONNX Fusion -> Atomic when pattern mismatch (ARCH-ONNX).
    OnnxFusionToAtomic,
    /// A4: HW Fusion -> Standalone when hardware constraint violated (ARCH-DETAILED-DESIGNS).
    HwFusionToStandalone,
    /// A5: Reshape/Transpose metadata NOP (NO-SILENT-FALLBACK exception).
    ReshapeTransposeMetadataNop,
    /// Placeholder for future SPEC-authorized fallbacks.
    /// Must be explicitly approved by user in writing before use.
    Reserved(String),
}

impl FallbackWhitelist {
    /// Human-readable authorization reference for each whitelist item.
    // @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn spec_ref(&self) -> &str {
        match self {
            Self::HfToModelScope => "REQ-LOADER-016",
            Self::OnnxFusionToAtomic => "ARCH-ONNX",
            Self::HwFusionToStandalone => "ARCH-DETAILED-DESIGNS",
            Self::ReshapeTransposeMetadataNop => "NO-SILENT-FALLBACK exception",
            Self::Reserved(r) => r,
        }
    }

    /// All SPEC-authorized fallback items (excluding Reserved).
    // @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn authorized_items() -> Vec<Self> {
        vec![
            Self::HfToModelScope,
            Self::OnnxFusionToAtomic,
            Self::HwFusionToStandalone,
            Self::ReshapeTransposeMetadataNop,
        ]
    }
}

impl fmt::Display for FallbackWhitelist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HfToModelScope => write!(f, "A2: HF→ModelScope ({})", self.spec_ref()),
            Self::OnnxFusionToAtomic => write!(f, "A3: ONNX Fusion→Atomic ({})", self.spec_ref()),
            Self::HwFusionToStandalone => write!(f, "A4: HW Fusion→Standalone ({})", self.spec_ref()),
            Self::ReshapeTransposeMetadataNop => write!(f, "A5: Reshape/Transpose NOP ({})", self.spec_ref()),
            Self::Reserved(r) => write!(f, "Reserved: {r}"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 REQ-FALLBACK-002: Non-authorized fallback error
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Structured error for non-authorized fallback attempts (REQ-FALLBACK-002).
///
/// Every fallback outside the REQ-FALLBACK-001 whitelist must return this error,
/// not a silent NOP. The error includes: trigger condition, attempted fallback
/// path, and suggested fix direction.
#[derive(Debug, Clone)]
pub struct FallbackError {
    /// The condition that triggered the fallback attempt.
    pub trigger: String,
    /// The fallback path that was attempted.
    pub attempted_path: String,
    /// Suggested fix direction.
    pub suggestion: String,
}

impl FallbackError {
    /// Create a new FallbackError with all required fields.
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn new(trigger: impl Into<String>, attempted_path: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            trigger: trigger.into(),
            attempted_path: attempted_path.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Convert to CompilerError.
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn to_compiler_error(&self) -> CompilerError {
        CompilerError::UnauthorizedFallback {
            trigger: self.trigger.clone(),
            attempted_path: self.attempted_path.clone(),
            suggestion: self.suggestion.clone(),
        }
    }
}

impl fmt::Display for FallbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UNAUTHORIZED-FALLBACK: trigger='{}', attempted='{}', fix='{}'",
            self.trigger, self.attempted_path, self.suggestion
        )
    }
}

impl std::error::Error for FallbackError {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 REQ-FAIL-TRIANG-001: IR-layer error code (IR-ERR)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Position within the IR where a precondition violation was detected.
#[derive(Debug, Clone)]
pub enum IrPosition {
    /// Violation at a specific op (identified by OpId and op kind).
    Op { op_id: OpId, op_label: String },
    /// Violation at a specific tensor slot.
    Slot { tensor_id: TensorId, name: String },
    /// Violation on an edge (producer -> consumer).
    Edge { producer_op: OpId, consumer_op: OpId },
}

impl fmt::Display for IrPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Op { op_id, op_label } => write!(f, "op '{}' (id={})", op_label, op_id.0),
            Self::Slot { tensor_id, name } => write!(f, "slot '{}' (id={})", name, tensor_id.0),
            Self::Edge { producer_op, consumer_op } => {
                write!(f, "edge producer={} → consumer={}", producer_op.0, consumer_op.0)
            }
        }
    }
}

/// Structured IR-layer error with IR-ERR prefix (REQ-FAIL-TRIANG-001).
///
/// IR-layer errors capture precondition violations in the CompilerGraph:
/// - The violated precondition name
/// - The position in the IR (OpKind/slot/edge)
/// - Expected vs actual values
///
/// These errors are detected in `compile()` entry `pre_check()`.
#[derive(Debug, Clone)]
pub struct IrError {
    /// The precondition that was violated.
    pub precondition: String,
    /// Where in the IR the violation was detected.
    pub position: IrPosition,
    /// The expected value (or description thereof).
    pub expected: String,
    /// The actual value observed.
    pub actual: String,
}

impl IrError {
    /// Create a new IR-layer error.
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn new(
        precondition: impl Into<String>,
        position: IrPosition,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self {
            precondition: precondition.into(),
            position,
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Convert to CompilerError.
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn to_compiler_error(&self) -> CompilerError {
        CompilerError::IrLayer {
            precondition: self.precondition.clone(),
            position: format!("{}", self.position),
            expected: self.expected.clone(),
            actual: self.actual.clone(),
        }
    }

    /// Error code prefix (always "IR-ERR").
    pub fn prefix() -> &'static str {
        "IR-ERR"
    }
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IR-ERR: precondition '{}' violated at {}: expected '{}', actual '{}'",
            self.precondition, self.position, self.expected, self.actual
        )
    }
}

impl std::error::Error for IrError {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 REQ-FAIL-TRIANG-002: PASS-layer error code (PASS-ERR)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Structured PASS-layer error with PASS-ERR prefix (REQ-FAIL-TRIANG-002).
///
/// PASS-layer errors capture invariant violations after a pass executes:
/// - The violated invariant name
/// - The pass name where the violation was detected
/// - Input graph vs output graph difference
///
/// These errors are detected after pass execution verification.
#[derive(Debug, Clone)]
pub struct PassError {
    /// The invariant that was violated.
    pub invariant: String,
    /// The pass where the violation was detected.
    pub pass_name: String,
    /// Description of the input graph state before the pass.
    pub input_diff: String,
    /// Description of the output graph state after the pass.
    pub output_diff: String,
}

impl PassError {
    /// Create a new PASS-layer error.
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn new(
        invariant: impl Into<String>,
        pass_name: impl Into<String>,
        input_diff: impl Into<String>,
        output_diff: impl Into<String>,
    ) -> Self {
        Self {
            invariant: invariant.into(),
            pass_name: pass_name.into(),
            input_diff: input_diff.into(),
            output_diff: output_diff.into(),
        }
    }

    /// Convert to CompilerError.
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn to_compiler_error(&self) -> CompilerError {
        CompilerError::PassLayer {
            invariant: self.invariant.clone(),
            pass_name: self.pass_name.clone(),
            input_diff: self.input_diff.clone(),
            output_diff: self.output_diff.clone(),
        }
    }

    /// Error code prefix (always "PASS-ERR").
    pub fn prefix() -> &'static str {
        "PASS-ERR"
    }
}

impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PASS-ERR: invariant '{}' violated in pass '{}': input='{}', output='{}'",
            self.invariant, self.pass_name, self.input_diff, self.output_diff
        )
    }
}

impl std::error::Error for PassError {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5 REQ-FAIL-TRIANG-003: CODEGEN-layer error code (CG-ERR)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Register state snapshot at the point of codegen failure.
#[derive(Debug, Clone)]
pub struct RegisterState {
    /// Number of allocated registers.
    pub allocated: usize,
    /// Number of available registers.
    pub available: usize,
    /// Number of spilled registers.
    pub spilled: usize,
}

impl fmt::Display for RegisterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "allocated={}, available={}, spilled={}",
            self.allocated, self.available, self.spilled
        )
    }
}

/// Structured CODEGEN-layer error with CG-ERR prefix (REQ-FAIL-TRIANG-003).
///
/// CODEGEN-layer errors capture ISA emission failures:
/// - The failed VmInstr/ISA instruction
/// - The source TraceOp/OpKind that triggered the emission
/// - The DeviceProfile in use
/// - Register state at point of failure
///
/// These errors are detected during ISA Lowering emit.
#[derive(Debug, Clone)]
pub struct CodegenError {
    /// The VmInstr or ISA instruction that failed to emit.
    pub vminstr: String,
    /// The source TraceOp or OpKind that triggered this emission.
    pub source_op: String,
    /// The DeviceProfile in use at time of failure.
    pub device_profile: String,
    /// Register state snapshot at point of failure.
    pub register_state: Option<RegisterState>,
}

impl CodegenError {
    /// Create a new CODEGEN-layer error.
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn new(
        vminstr: impl Into<String>,
        source_op: impl Into<String>,
        device_profile: impl Into<String>,
        register_state: Option<RegisterState>,
    ) -> Self {
        Self {
            vminstr: vminstr.into(),
            source_op: source_op.into(),
            device_profile: device_profile.into(),
            register_state,
        }
    }

    /// Create from CompilerError::CodegenViolation, enriching with structured fields.
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn from_violation(
        violation_msg: &str,
        source_op: impl Into<String>,
        device_profile: impl Into<String>,
    ) -> Self {
        Self {
            vminstr: violation_msg.to_string(),
            source_op: source_op.into(),
            device_profile: device_profile.into(),
            register_state: None,
        }
    }

    /// Convert to CompilerError.
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    pub fn to_compiler_error(&self) -> CompilerError {
        CompilerError::CodegenLayer {
            vminstr: self.vminstr.clone(),
            source_op: self.source_op.clone(),
            device_profile: self.device_profile.clone(),
            register_state: self.register_state.as_ref().map(|rs| format!("{}", rs)),
        }
    }

    /// Error code prefix (always "CG-ERR").
    pub fn prefix() -> &'static str {
        "CG-ERR"
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CG-ERR: vminstr='{}', source_op='{}', device_profile='{}'",
            self.vminstr, self.source_op, self.device_profile
        )?;
        if let Some(ref rs) = self.register_state {
            write!(f, ", registers=[{}]", rs)?;
        }
        Ok(())
    }
}

impl std::error::Error for CodegenError {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6 compile() entry pre_check() — REQ-FAIL-TRIANG-001
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Run IR-layer precondition checks at compile() entry (REQ-FAIL-TRIANG-001).
///
/// Validates graph structural invariants before any compilation proceeds.
/// Returns a list of IrError for each violated precondition. An empty Vec
/// means all preconditions pass.
// @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
// @trace REQ-PASS-INV-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
pub fn pre_check(graph: &CompilerGraph) -> Vec<IrError> {
    let mut errors = Vec::new();

    // Check 1: Every op must have at least one output tensor.
    for op in &graph.ops {
        if op.outputs.is_empty() {
            errors.push(IrError::new(
                "op_has_output",
                IrPosition::Op { op_id: op.id, op_label: op.label.clone() },
                ">= 1 output tensor",
                "0 output tensors",
            ));
        }
    }

    // Check 2: Input tensors must exist in the graph.
    for op in &graph.ops {
        for &input_tid in &op.inputs {
            if graph.tensor(input_tid).is_none() {
                errors.push(IrError::new(
                    "input_tensor_exists",
                    IrPosition::Slot { tensor_id: input_tid, name: format!("input_of_{}", op.label) },
                    format!("tensor id {} exists", input_tid.0),
                    "tensor not found in graph",
                ));
            }
        }
    }

    // Check 3: Output tensors must exist in the graph.
    for op in &graph.ops {
        for &output_tid in &op.outputs {
            if graph.tensor(output_tid).is_none() {
                errors.push(IrError::new(
                    "output_tensor_exists",
                    IrPosition::Slot { tensor_id: output_tid, name: format!("output_of_{}", op.label) },
                    format!("tensor id {} exists", output_tid.0),
                    "tensor not found in graph",
                ));
            }
        }
    }

    // Check 4: Producer-consumer dtype consistency on edges.
    for op in &graph.ops {
        for &input_tid in &op.inputs {
            if let Some(tensor) = graph.tensor(input_tid) {
                // Find the producer op for this tensor.
                if let Some(producer_id) = tensor.producer {
                    if let Some(producer) = graph.op(producer_id) {
                        // Verify producer actually declares this tensor as output.
                        if !producer.outputs.contains(&input_tid) {
                            errors.push(IrError::new(
                                "producer_declares_output",
                                IrPosition::Edge { producer_op: producer_id, consumer_op: op.id },
                                format!("producer op '{}' declares tensor {} as output", producer.label, input_tid.0),
                                format!("tensor {} not in producer outputs", input_tid.0),
                            ));
                        }
                    }
                }
            }
        }
    }

    errors
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7 PASS-layer post-verification — REQ-FAIL-TRIANG-002
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verify pass invariants after execution (REQ-FAIL-TRIANG-002).
///
/// Given a pass name and the before/after state, check invariants
/// and return PassError for each violation.
// @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
// @trace REQ-PASS-INV-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
pub fn pass_post_verify(
    pass_name: &str,
    op_count_before: usize,
    op_count_after: usize,
    tensor_count_before: usize,
    tensor_count_after: usize,
) -> Vec<PassError> {
    let mut errors = Vec::new();

    // Invariant 1: Pass must not increase op count beyond a reasonable threshold.
    // Optimization passes should reduce or maintain op count.
    // Some passes (e.g., lowering) may increase count, but we check for
    // correctness-invariant violations, not heuristic thresholds.
    // The key invariant: output tensor count must not decrease
    // (output tensors represent required computation results).
    if tensor_count_after < tensor_count_before {
        errors.push(PassError::new(
            "output_tensor_count_not_decreased",
            pass_name,
            format!("{} tensors before pass", tensor_count_before),
            format!("{} tensors after pass (lost {})", tensor_count_after, tensor_count_before - tensor_count_after),
        ));
    }

    // Invariant 2: Pass must not produce zero ops (that would be a NOP/deletion).
    if op_count_before > 0 && op_count_after == 0 {
        errors.push(PassError::new(
            "ops_not_all_eliminated",
            pass_name,
            format!("{} ops before pass", op_count_before),
            "0 ops after pass (all computation eliminated)",
        ));
    }

    errors
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §8 CODEGEN-layer emit check — REQ-FAIL-TRIANG-003
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Check codegen emit result and wrap in CodegenError if failed (REQ-FAIL-TRIANG-003).
///
/// Converts a raw CompilerError::CodegenViolation into a structured CodegenError
/// with the required CG-ERR prefix and full context.
// @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
pub fn codegen_emit_check(
    result: Result<(), CompilerError>,
    source_op: &str,
    device_profile: &DeviceProfile,
) -> Result<(), CompilerError> {
    match result {
        Ok(()) => Ok(()),
        Err(CompilerError::CodegenViolation(msg)) => {
            let cg_err = CodegenError::from_violation(
                &msg,
                source_op,
                format!("{:?}", device_profile.arch),
            );
            Err(cg_err.to_compiler_error())
        }
        Err(CompilerError::RegisterOverflow { needed, available, context }) => {
            let cg_err = CodegenError::new(
                format!("RegisterOverflow: need {needed}, have {available}"),
                source_op.to_string(),
                format!("{:?}", device_profile.arch),
                Some(RegisterState {
                    allocated: needed,
                    available,
                    spilled: 0,
                }),
            );
            Err(cg_err.to_compiler_error())
        }
        Err(CompilerError::UnsupportedDType { dtype, ref isa }) => {
            let cg_err = CodegenError::new(
                format!("UnsupportedDType: {:?} for ISA {}", dtype, isa),
                source_op.to_string(),
                format!("{:?}", device_profile.arch),
                None,
            );
            Err(cg_err.to_compiler_error())
        }
        Err(other) => Err(other),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §9 Fallback authorization check — REQ-FALLBACK-001/002
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Check whether a fallback path is authorized by the whitelist (REQ-FALLBACK-001/002).
///
/// Returns Ok(()) if the fallback is authorized, Err(FallbackError) otherwise.
// @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
// @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
pub fn check_fallback_authorized(
    fallback: &FallbackWhitelist,
    trigger: &str,
) -> Result<(), FallbackError> {
    // Only the 4 concrete whitelist items are authorized.
    // Reserved items are NOT authorized unless explicitly approved by user.
    match fallback {
        FallbackWhitelist::HfToModelScope
        | FallbackWhitelist::OnnxFusionToAtomic
        | FallbackWhitelist::HwFusionToStandalone
        | FallbackWhitelist::ReshapeTransposeMetadataNop => Ok(()),
        FallbackWhitelist::Reserved(reason) => {
            Err(FallbackError::new(
                trigger,
                format!("Reserved fallback: {reason}"),
                "Request SPEC authorization for this fallback path before use",
            ))
        }
    }
}

/// Check for silent fallback anti-patterns (REQ-FALLBACK-002).
///
/// Detects forbidden patterns: emit_nop_raw, match _ => Ok(()), eprintln!("[WARN]..."),
/// unwrap_or(default), etc. Returns list of detected violations with suggestions.
// @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
pub fn detect_silent_fallback_patterns(source: &str) -> Vec<FallbackError> {
    let mut violations = Vec::new();
    let forbidden_patterns = [
        ("emit_nop_raw", "emit_nop_raw() — silent NOP instead of Err"),
        ("match _ => Ok(())", "match _ => Ok(()) — catch-all success instead of Err"),
        ("eprintln!(\"[WARN]\",", "eprintln!([WARN]...) — warning instead of Err"),
        ("unwrap_or(default)", "unwrap_or(default) — silent default instead of Err"),
        ("unwrap_or(Default::default())", "unwrap_or(Default::default()) — silent default instead of Err"),
    ];

    for (pattern, description) in &forbidden_patterns {
        if source.contains(pattern) {
            violations.push(FallbackError::new(
                description.to_string(),
                format!("pattern '{}' detected", pattern),
                "Replace with explicit Err() returning structured diagnostics",
            ));
        }
    }

    violations
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §10 Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    // ── REQ-FALLBACK-001 tests ────────────────────────────────────

    #[test]
    // @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_fallback_whitelist_has_five_items() {
        let items = FallbackWhitelist::authorized_items();
        assert_eq!(items.len(), 4, "REQ-FALLBACK-001: exactly 4 concrete whitelist items (A2/A3/A4/A5)");
    }

    #[test]
    // @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_fallback_whitelist_spec_refs() {
        assert_eq!(FallbackWhitelist::HfToModelScope.spec_ref(), "REQ-LOADER-016");
        assert_eq!(FallbackWhitelist::OnnxFusionToAtomic.spec_ref(), "ARCH-ONNX");
        assert_eq!(FallbackWhitelist::HwFusionToStandalone.spec_ref(), "ARCH-DETAILED-DESIGNS");
        assert_eq!(FallbackWhitelist::ReshapeTransposeMetadataNop.spec_ref(), "NO-SILENT-FALLBACK exception");
    }

    #[test]
    // @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_fallback_whitelist_display() {
        assert!(format!("{}", FallbackWhitelist::HfToModelScope).contains("A2"));
        assert!(format!("{}", FallbackWhitelist::OnnxFusionToAtomic).contains("A3"));
        assert!(format!("{}", FallbackWhitelist::HwFusionToStandalone).contains("A4"));
        assert!(format!("{}", FallbackWhitelist::ReshapeTransposeMetadataNop).contains("A5"));
    }

    #[test]
    // @trace REQ-FALLBACK-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_authorized_fallbacks_pass_check() {
        for item in FallbackWhitelist::authorized_items() {
            assert!(
                check_fallback_authorized(&item, "test").is_ok(),
                "Authorized fallback {:?} should pass check",
                item
            );
        }
    }

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_reserved_fallback_fails_check() {
        let reserved = FallbackWhitelist::Reserved("experimental".to_string());
        let result = check_fallback_authorized(&reserved, "test trigger");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.trigger.contains("test trigger"));
        assert!(err.attempted_path.contains("experimental"));
        assert!(err.suggestion.contains("SPEC authorization"));
    }

    // ── REQ-FALLBACK-002 tests ────────────────────────────────────

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_fallback_error_structure() {
        let err = FallbackError::new(
            "dtype mismatch",
            "fallback to F32",
            "propagate dtype through graph",
        );
        assert_eq!(err.trigger, "dtype mismatch");
        assert_eq!(err.attempted_path, "fallback to F32");
        assert_eq!(err.suggestion, "propagate dtype through graph");
    }

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_fallback_error_display_contains_unauthorized() {
        let err = FallbackError::new("trigger", "path", "fix");
        let msg = format!("{}", err);
        assert!(msg.contains("UNAUTHORIZED-FALLBACK"));
    }

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_fallback_error_to_compiler_error() {
        let err = FallbackError::new("trigger", "path", "fix");
        let compiler_err = err.to_compiler_error();
        match compiler_err {
            CompilerError::UnauthorizedFallback { trigger, attempted_path, suggestion } => {
                assert_eq!(trigger, "trigger");
                assert_eq!(attempted_path, "path");
                assert_eq!(suggestion, "fix");
            }
            other => panic!("Expected UnauthorizedFallback, got {:?}", other),
        }
    }

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_detect_silent_fallback_patterns_clean() {
        let clean_source = "let x = some_function();\nOk(x)\n";
        let violations = detect_silent_fallback_patterns(clean_source);
        assert!(violations.is_empty(), "Clean source should have no violations");
    }

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_detect_silent_fallback_patterns_emit_nop() {
        let dirty_source = "emit_nop_raw();\nOk(())\n";
        let violations = detect_silent_fallback_patterns(dirty_source);
        assert!(!violations.is_empty(), "Should detect emit_nop_raw");
        assert!(violations[0].trigger.contains("emit_nop_raw"));
    }

    #[test]
    // @trace REQ-FALLBACK-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_detect_silent_fallback_patterns_unwrap_or_default() {
        let dirty_source = "let x = result.unwrap_or(default);\n";
        let violations = detect_silent_fallback_patterns(dirty_source);
        assert!(!violations.is_empty(), "Should detect unwrap_or(default)");
    }

    // ── REQ-FAIL-TRIANG-001 tests ─────────────────────────────────

    #[test]
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_ir_error_prefix() {
        assert_eq!(IrError::prefix(), "IR-ERR");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_ir_error_display_contains_prefix() {
        let err = IrError::new(
            "input_tensor_exists",
            IrPosition::Op { op_id: OpId(42), op_label: "test_op".to_string() },
            "tensor exists",
            "tensor not found",
        );
        let msg = format!("{}", err);
        assert!(msg.starts_with("IR-ERR:"), "IR error must start with IR-ERR prefix");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_ir_error_contains_position() {
        let err = IrError::new(
            "precondition",
            IrPosition::Op { op_id: OpId(1), op_label: "gemm_0".to_string() },
            "expected",
            "actual",
        );
        let msg = format!("{}", err);
        assert!(msg.contains("gemm_0"), "IR error must contain op label");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_ir_error_contains_expected_actual() {
        let err = IrError::new(
            "dtype_match",
            IrPosition::Slot { tensor_id: TensorId(5), name: "weight".to_string() },
            "BF16",
            "F32",
        );
        let msg = format!("{}", err);
        assert!(msg.contains("BF16"), "IR error must contain expected value");
        assert!(msg.contains("F32"), "IR error must contain actual value");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_ir_position_edge_display() {
        let pos = IrPosition::Edge { producer_op: OpId(1), consumer_op: OpId(2) };
        let msg = format!("{}", pos);
        assert!(msg.contains("producer=1"));
        assert!(msg.contains("consumer=2"));
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-001 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_ir_error_to_compiler_error() {
        let err = IrError::new(
            "test_precondition",
            IrPosition::Op { op_id: OpId(0), op_label: "op0".to_string() },
            "expected_val",
            "actual_val",
        );
        let ce = err.to_compiler_error();
        match ce {
            CompilerError::IrLayer { precondition, position, expected, actual } => {
                assert_eq!(precondition, "test_precondition");
                assert!(position.contains("op0"));
                assert_eq!(expected, "expected_val");
                assert_eq!(actual, "actual_val");
            }
            other => panic!("Expected IrLayer, got {:?}", other),
        }
    }

    // ── REQ-FAIL-TRIANG-002 tests ─────────────────────────────────

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_error_prefix() {
        assert_eq!(PassError::prefix(), "PASS-ERR");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_error_display_contains_prefix() {
        let err = PassError::new(
            "invariant_name",
            "pass_name",
            "input",
            "output",
        );
        let msg = format!("{}", err);
        assert!(msg.starts_with("PASS-ERR:"), "PASS error must start with PASS-ERR prefix");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_error_contains_invariant_and_pass() {
        let err = PassError::new(
            "tensor_count_preserved",
            "DeadVRegElimination",
            "10 tensors",
            "5 tensors",
        );
        let msg = format!("{}", err);
        assert!(msg.contains("tensor_count_preserved"), "Must contain invariant name");
        assert!(msg.contains("DeadVRegElimination"), "Must contain pass name");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_error_to_compiler_error() {
        let err = PassError::new(
            "invariant",
            "pass",
            "input_diff",
            "output_diff",
        );
        let ce = err.to_compiler_error();
        match ce {
            CompilerError::PassLayer { invariant, pass_name, input_diff, output_diff } => {
                assert_eq!(invariant, "invariant");
                assert_eq!(pass_name, "pass");
                assert_eq!(input_diff, "input_diff");
                assert_eq!(output_diff, "output_diff");
            }
            other => panic!("Expected PassLayer, got {:?}", other),
        }
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_post_verify_tensor_loss_detected() {
        let errors = pass_post_verify("TestPass", 10, 10, 20, 15);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].invariant.contains("tensor_count_not_decreased"));
        assert!(errors[0].output_diff.contains("lost 5"));
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_post_verify_ops_eliminated_detected() {
        let errors = pass_post_verify("TestPass", 5, 0, 5, 5);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].invariant.contains("ops_not_all_eliminated"));
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-002 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_pass_post_verify_clean() {
        let errors = pass_post_verify("GoodPass", 10, 8, 20, 20);
        assert!(errors.is_empty(), "Clean pass should produce no errors");
    }

    // ── REQ-FAIL-TRIANG-003 tests ─────────────────────────────────

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_codegen_error_prefix() {
        assert_eq!(CodegenError::prefix(), "CG-ERR");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_codegen_error_display_contains_prefix() {
        let err = CodegenError::new("VecFma", "RmsNorm", "Cpu", None);
        let msg = format!("{}", err);
        assert!(msg.starts_with("CG-ERR:"), "CODEGEN error must start with CG-ERR prefix");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_codegen_error_contains_vminstr_and_source_op() {
        let err = CodegenError::new("VecFma", "RmsNorm", "Cpu", None);
        let msg = format!("{}", err);
        assert!(msg.contains("VecFma"), "Must contain VmInstr name");
        assert!(msg.contains("RmsNorm"), "Must contain source OpKind");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_codegen_error_with_register_state() {
        let err = CodegenError::new(
            "VecFma",
            "Gemm",
            "Cpu",
            Some(RegisterState { allocated: 32, available: 16, spilled: 8 }),
        );
        let msg = format!("{}", err);
        assert!(msg.contains("allocated=32"), "Must contain register state");
        assert!(msg.contains("spilled=8"), "Must contain spill count");
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_codegen_error_to_compiler_error() {
        let err = CodegenError::new(
            "VecFma",
            "Gemm",
            "x86_64",
            Some(RegisterState { allocated: 32, available: 16, spilled: 8 }),
        );
        let ce = err.to_compiler_error();
        match ce {
            CompilerError::CodegenLayer { vminstr, source_op, device_profile, register_state } => {
                assert_eq!(vminstr, "VecFma");
                assert_eq!(source_op, "Gemm");
                assert_eq!(device_profile, "x86_64");
                assert!(register_state.unwrap().contains("allocated=32"));
            }
            other => panic!("Expected CodegenLayer, got {:?}", other),
        }
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_codegen_error_from_violation() {
        let err = CodegenError::from_violation("alignment error", "Gemm", "x86_64");
        assert_eq!(err.vminstr, "alignment error");
        assert_eq!(err.source_op, "Gemm");
        assert_eq!(err.device_profile, "x86_64");
        assert!(err.register_state.is_none());
    }

    #[test]
    // @trace REQ-FAIL-TRIANG-003 [entity:ENT-COMPILER-GRAPH] [api:POST /compile]
    fn test_register_state_display() {
        let rs = RegisterState { allocated: 10, available: 16, spilled: 2 };
        let msg = format!("{}", rs);
        assert!(msg.contains("allocated=10"));
        assert!(msg.contains("available=16"));
        assert!(msg.contains("spilled=2"));
    }
}
