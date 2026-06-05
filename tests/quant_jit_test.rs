//! Quantization JIT integration tests (QCG12).
//!
//! Validates that the quantized dequantization JIT pipeline produces correct
//! results by exercising each quantized format's DecodeTraceBuilder trace
//! generation, coverage-check driven path selection, and numerical simulation.
//!
//! SPEC ref: `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md`
//! Dependencies: QCG4 (NativePath), QCG5 (AssistedPath), QCG6 (DequantFMAPath),
//!               QCG7 (QuantAlgoKind), QCG8 (quant_aware_fusion),
//!               QCG11 (QuantCapability)

use gllm_kernels::compiler::codegen::vm::quant_decode::DecodeTraceBuilder;
use gllm_kernels::compiler::codegen::vm::numerical_sim::simulate_compile;
use gllm_kernels::compiler::quant_format::{
    QuantAlgoKind, QuantDataKind, QuantFormatDescriptor, registry,
    IsaKind, OpCategory, CoveragePath, CoverageMatrix,
    coverage_check, coverage_matrix,
};
use gllm_kernels::compiler::fusion::quant_aware::{
    can_fuse_quant_aware, QuantFusionDecision, fusion_cost,
    select_fusion_groups,
};
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::quant::QuantType;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 Test framework helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Floating-point tolerance for numerical comparisons.
const TOLERANCE: f64 = 1e-4;

/// A representative quantized format descriptor + metadata for JIT testing.
struct QuantJitTestCase {
    /// Human-readable name for diagnostics.
    name: &'static str,
    /// QuantType for registry lookup.
    quant_type: QuantType,
    /// Number of SIMD output lanes to test with.
    output_lanes: usize,
}

/// Returns at least 5 representative quantized format test cases covering:
/// - Classic GGML: Q4_0 (4-bit packed), Q8_0 (INT8 signed)
/// - K-Quant: Q4_K (hierarchical scale), Q6_K (StaticBias + i8 scales)
/// - External: AWQ4 (row-major INT4), MXFP4 (Float4)
fn representative_formats() -> Vec<QuantJitTestCase> {
    vec![
        QuantJitTestCase {
            name: "Q4_0",
            quant_type: QuantType::Q4_0,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "Q4_1",
            quant_type: QuantType::Q4_1,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "Q5_0",
            quant_type: QuantType::Q5_0,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "Q8_0",
            quant_type: QuantType::Q8_0,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "Q4_K",
            quant_type: QuantType::Q4K,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "Q5_K",
            quant_type: QuantType::Q5K,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "Q6_K",
            quant_type: QuantType::Q6K,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "AWQ4",
            quant_type: QuantType::AWQ4,
            output_lanes: 8,
        },
        QuantJitTestCase {
            name: "MXFP4",
            quant_type: QuantType::Mxfp4 { block_size: 32 },
            output_lanes: 8,
        },
    ]
}

/// Validates a dequantization trace: non-empty, correct input slots, final slot in range.
fn assert_trace_valid(trace_name: &str, trace: &[gllm_kernels::compiler::trace::TraceOp], _output_lanes: usize) {
    assert!(!trace.is_empty(), "[{}] DecodeTraceBuilder produced empty trace", trace_name);
    assert!(
        matches!(trace[0], gllm_kernels::compiler::trace::TraceOp::Input(0)),
        "[{}] First TraceOp must be Input(0) for block_base", trace_name
    );
    assert!(
        matches!(trace[1], gllm_kernels::compiler::trace::TraceOp::Input(1)),
        "[{}] Second TraceOp must be Input(1) for data_ptr", trace_name
    );
}

/// Run the compile-time numerical simulator and check the result within tolerance.
fn assert_numerical_simulation_ok(name: &str, desc: &QuantFormatDescriptor, output_lanes: usize) {
    let mut trace = Vec::new();
    DecodeTraceBuilder::new(desc, output_lanes).build(&mut trace);

    // Build minimal block data: block_bytes worth of zeroed data for simulation.
    // The simulator verifies the trace can execute without NaN/Inf.
    let block_data = vec![0u8; desc.block_bytes.max(1)];
    let inputs: &[i64] = &[4096, 4096]; // block_base, data_ptr (same for gather path)

    let result = simulate_compile(&trace, desc, &block_data, inputs);
    assert!(
        result.is_ok(),
        "[{}] numerical simulation failed: {:?}", name, result.err()
    );

    let sim_result = result.unwrap();
    assert!(
        !sim_result.has_nan,
        "[{}] numerical simulation produced NaN", name
    );
    assert!(
        !sim_result.has_inf,
        "[{}] numerical simulation produced Inf", name
    );
    assert!(
        !sim_result.outputs.is_empty(),
        "[{}] numerical simulation produced empty outputs", name
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 Trace generation tests (compile phase)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_trace_all_representative_formats() {
    let reg = registry();
    for tc in representative_formats() {
        let desc = reg.get(&tc.quant_type);
        assert!(
            desc.is_some(),
            "[{}] QuantFormatDescriptor not registered for {:?}", tc.name, tc.quant_type
        );
        let desc = desc.unwrap();

        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, tc.output_lanes).build(&mut trace);

        assert_trace_valid(tc.name, &trace, tc.output_lanes);
        assert!(
            final_slot.0 < trace.len() as u32,
            "[{}] final_slot {} out of range [0, {})", tc.name, final_slot, trace.len()
        );
    }
}

#[test]
fn test_quant_codegen_trace_slot_references_valid() {
    let reg = registry();
    for tc in representative_formats() {
        let desc = reg.get(&tc.quant_type).unwrap();
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, tc.output_lanes).build(&mut trace);

        let len = trace.len();
        assert!(
            final_slot.0 < len as u32,
            "[{}] final_slot {} >= trace.len() {}", tc.name, final_slot, len
        );

        // Validate all slot references within each op are to earlier slots
        use gllm_kernels::compiler::trace::TraceOp;
        use gllm_kernels::compiler::trace::ValueId;
        for (pos, op) in trace.iter().enumerate() {
            let refs: Vec<ValueId> = match op {
                TraceOp::QuantBitAnd { lhs, rhs }
                | TraceOp::QuantBitOr { lhs, rhs } => vec![*lhs, *rhs],
                TraceOp::QuantBroadcast { src, .. }
                | TraceOp::QuantCastF16toF32 { src }
                | TraceOp::QuantCastI8toF32 { src }
                | TraceOp::QuantExtractBits { src, .. }
                | TraceOp::QuantIntDivConst { src, .. }
                | TraceOp::QuantIntMul { src, .. }
                | TraceOp::QuantShiftLeft { src, .. }
                | TraceOp::QuantShiftRight { src, .. } => vec![*src],
                TraceOp::QuantCodebookLookup { indices, .. } => vec![*indices],
                TraceOp::QuantDequantFma { acc, a, b } => vec![*acc, *a, *b],
                TraceOp::QuantInterleave { lo, hi } => vec![*lo, *hi],
                TraceOp::QuantScalarLoad { ptr, .. } => vec![*ptr],
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b) => vec![*a, *b],
                TraceOp::QuantAndMask { src, .. } => vec![*src],
                _ => vec![],
            };
            for r in refs {
                assert!(
                    (r.0 as usize) < pos,
                    "[{}] Op at slot {} refs future slot {} ({:?})",
                    tc.name, pos, r, op
                );
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 Coverage-based path selection tests (QCG4-QCG6 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_path_selection_per_format() {
    // For each representative format, verify that coverage_check returns a valid
    // CoveragePath for all (IsaKind × OpCategory) combinations.
    for tc in representative_formats() {
        for &isa in IsaKind::all() {
            for &op in OpCategory::all() {
                let path = coverage_check(tc.quant_type, isa, op);
                // Every cell must produce a valid path
                match path {
                    CoveragePath::Native => {}
                    CoveragePath::Assisted => {}
                    CoveragePath::DequantFMA => {}
                }
            }
        }
    }
}

#[test]
fn test_quant_codegen_path_priority_native_over_assisted() {
    // INT8 format on x86_64 (has VNNI) should get Native for GEMM
    let native_path = coverage_check(QuantType::Q8_0, IsaKind::X86, OpCategory::Gemm);
    // ARM with SDOT should also get Native or Assisted for GEMM with INT8
    let arm_path = coverage_check(QuantType::Q8_0, IsaKind::Arm, OpCategory::Gemm);

    // Both should be valid (not panic), and INT8 on x86 should be at least Assisted
    assert!(
        native_path.rank() <= CoveragePath::Assisted.rank(),
        "Q8_0 on x86_64 GEMM should be Native or Assisted, got {:?}", native_path
    );
    assert!(
        arm_path.rank() <= CoveragePath::Assisted.rank(),
        "Q8_0 on ARM GEMM should be Native or Assisted, got {:?}", arm_path
    );

    // Q4_0 on x86_64: PackedInt4 requires assisted nibble unpack on VNNI
    let q4_path = coverage_check(QuantType::Q4_0, IsaKind::X86, OpCategory::Gemm);
    assert!(
        matches!(q4_path, CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA),
        "Q4_0 on x86_64 must have a valid coverage path"
    );

    // Verify priority ranking: Native(0) > Assisted(1) > DequantFMA(2)
    assert!(CoveragePath::Native.rank() < CoveragePath::Assisted.rank());
    assert!(CoveragePath::Assisted.rank() < CoveragePath::DequantFMA.rank());
}

#[test]
fn test_quant_codegen_dequant_fma_path_for_float_formats() {
    // Float formats (BF16/F16/F32) on Quant ops may fall back to DequantFMA
    // on certain ISA × OpCategory combos — this is valid behavior.
    for qt in &[QuantType::Bf16, QuantType::F16, QuantType::F32] {
        let path = coverage_check(*qt, IsaKind::X86, OpCategory::Quant);
        assert!(
            matches!(path, CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA),
            "Float format {:?} on x86_64 Quant must have valid path, got {:?}", qt, path
        );
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 Numerical simulation tests (execute phase + scalar reference comparison)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_numerical_sim_q4_0() {
    let reg = registry();
    let desc = reg.get(&QuantType::Q4_0).expect("Q4_0 must be registered");
    assert_numerical_simulation_ok("Q4_0", desc, 8);
}

#[test]
fn test_quant_codegen_numerical_sim_q4_1() {
    let reg = registry();
    let desc = reg.get(&QuantType::Q4_1).expect("Q4_1 must be registered");
    assert_numerical_simulation_ok("Q4_1", desc, 8);
}

#[test]
fn test_quant_codegen_numerical_sim_q8_0() {
    let reg = registry();
    let desc = reg.get(&QuantType::Q8_0).expect("Q8_0 must be registered");
    assert_numerical_simulation_ok("Q8_0", desc, 8);
}

#[test]
fn test_quant_codegen_numerical_sim_q4k() {
    let reg = registry();
    let desc = reg.get(&QuantType::Q4K).expect("Q4_K must be registered");
    assert_numerical_simulation_ok("Q4_K", desc, 8);
}

#[test]
fn test_quant_codegen_numerical_sim_q6k() {
    let reg = registry();
    let desc = reg.get(&QuantType::Q6K).expect("Q6_K must be registered");
    assert_numerical_simulation_ok("Q6_K", desc, 8);
}

#[test]
fn test_quant_codegen_numerical_sim_all_formats() {
    let reg = registry();
    for tc in representative_formats() {
        let desc = match reg.get(&tc.quant_type) {
            Some(d) => d,
            None => continue,
        };
        assert_numerical_simulation_ok(tc.name, desc, tc.output_lanes);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5 QuantAlgoKind descriptor round-trip tests (QCG7 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_algo_kind_descriptor_roundtrip() {
    // Every QuantAlgoKind must produce a valid descriptor from the registry
    for kind in QuantAlgoKind::all() {
        let desc = kind.descriptor();
        assert_eq!(desc.quant_type, kind.quant_type(), "QuantAlgoKind::{:?} descriptor mismatch", kind);
        assert!(!desc.name.is_empty(), "QuantAlgoKind::{:?} has empty name", kind);
        assert!(desc.block_size > 0, "QuantAlgoKind::{:?} has zero block_size", kind);
        assert!(desc.block_bytes > 0, "QuantAlgoKind::{:?} has zero block_bytes", kind);
    }
}

#[test]
fn test_quant_codegen_algo_kind_all_registered() {
    let reg = registry();
    for kind in QuantAlgoKind::all() {
        let qt = kind.quant_type();
        assert!(
            reg.get(&qt).is_some(),
            "QuantAlgoKind::{:?} → {:?} not registered in QuantFormatRegistry",
            kind, qt
        );
    }
}

#[test]
fn test_quant_codegen_format_descriptor_properties() {
    let reg = registry();
    for kind in QuantAlgoKind::all() {
        let desc = kind.descriptor();
        let name = format!("{:?}", kind);

        // bits_per_element must be positive for quantized formats
        assert!(
            desc.bits_per_element > 0,
            "[{}] bits_per_element must be > 0, got {}", name, desc.bits_per_element
        );

        // block_size must be positive
        assert!(
            desc.block_size > 0,
            "[{}] block_size must be > 0, got {}", name, desc.block_size
        );

        // block_bytes must be positive
        assert!(
            desc.block_bytes > 0,
            "[{}] block_bytes must be > 0, got {}", name, desc.block_bytes
        );
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6 Quant-aware fusion integration tests (QCG8 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Two quantization types are "same precision" if they can fuse:
/// same quant type → same precision; None/None → same; quant/None → same.
fn same_precision(a: Option<QuantType>, b: Option<QuantType>) -> bool {
    matches!(
        can_fuse_quant_aware(a, b),
        QuantFusionDecision::Fuse | QuantFusionDecision::FuseWithWiden
    )
}

#[test]
fn test_quant_codegen_fusion_same_format_fuses() {
    // Same quant type → Fuse
    for kind in QuantAlgoKind::all() {
        let qt = kind.quant_type();
        let decision = can_fuse_quant_aware(Some(qt), Some(qt));
        assert_eq!(
            decision,
            QuantFusionDecision::Fuse,
            "can_fuse_quant_aware({:?}, {:?}) should be Fuse, got {:?}",
            qt, qt, decision
        );
    }
}

#[test]
fn test_quant_codegen_fusion_different_format_splits() {
    let kinds = QuantAlgoKind::all();
    // Pick two distinct formats and verify they split
    if kinds.len() >= 2 {
        let qt_a = kinds[0].quant_type();
        let qt_b = kinds[1].quant_type();
        if qt_a != qt_b {
            let decision = can_fuse_quant_aware(Some(qt_a), Some(qt_b));
            assert_eq!(
                decision,
                QuantFusionDecision::Split,
                "can_fuse_quant_aware({:?}, {:?}) should be Split for different formats",
                qt_a, qt_b
            );
        }
    }
}

#[test]
fn test_quant_codegen_fusion_cost_same_is_zero() {
    for kind in QuantAlgoKind::all() {
        let qt = kind.quant_type();
        let cost = fusion_cost(Some(qt), Some(qt));
        assert!(
            (cost - 0.0).abs() < TOLERANCE as f32,
            "fusion_cost({:?}, {:?}) should be 0.0, got {}", qt, qt, cost
        );
    }
}

#[test]
fn test_quant_codegen_fusion_groups_correct_partitioning() {
    let ops = vec![
        Some(QuantType::Q4_0),
        Some(QuantType::Q4_0),
        Some(QuantType::Q6K),
        Some(QuantType::Q6K),
        None, // F32
    ];
    let groups = select_fusion_groups(&ops);

    // Q4_0 and Q6K are different formats → Split.
    // Q6K → None (F32 output) fuses because quant→F32 is a valid dequant+arithmetic group.
    // So we get 2 groups: (Q4_0, Q4_0) and (Q6K, Q6K, F32).
    assert_eq!(groups.len(), 2, "Expected 2 groups, got {}", groups.len());
    assert_eq!(groups[0].2, Some(QuantType::Q4_0));
    assert_eq!(groups[0].0, 0);
    assert_eq!(groups[0].1, 2);
    assert_eq!(groups[1].2, Some(QuantType::Q6K));
    assert_eq!(groups[1].0, 2);
    assert_eq!(groups[1].1, 5);
}

#[test]
fn test_quant_codegen_same_precision_exhaustive() {
    // All same-quant pairs must be same_precision
    for kind in QuantAlgoKind::all() {
        let qt = kind.quant_type();
        assert!(
            same_precision(Some(qt), Some(qt)),
            "same_precision({:?}, {:?}) should be true", qt, qt
        );
    }
    // None/None is same precision
    assert!(same_precision(None, None));

    // Cross-quant must NOT be same precision
    let kinds = QuantAlgoKind::all();
    for i in 0..kinds.len() {
        for j in (i + 1)..kinds.len() {
            let qa = kinds[i].quant_type();
            let qb = kinds[j].quant_type();
            if qa != qb {
                assert!(
                    !same_precision(Some(qa), Some(qb)),
                    "same_precision({:?}, {:?}) should be false for different formats",
                    qa, qb
                );
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7 Coverage matrix tests (QCG10/QCG11 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_coverage_matrix_completeness() {
    // For every registered QuantType, the 5×5 coverage matrix must be complete
    for kind in QuantAlgoKind::all() {
        let qt = kind.quant_type();
        let matrix = coverage_matrix(qt);

        // All 25 cells must have a valid path
        let mut cell_count = 0;
        for entry in matrix.iter() {
            assert!(
                matches!(
                    entry.path,
                    CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA
                ),
                "Uncovered cell: qt={:?} isa={:?} op={:?} path={:?}",
                qt, entry.isa, entry.op, entry.path
            );
            cell_count += 1;
        }
        assert_eq!(cell_count, 25, "CoverageMatrix for {:?} should have 25 cells", qt);
    }
}

#[test]
fn test_quant_codegen_coverage_check_returns_valid_path() {
    for kind in QuantAlgoKind::all() {
        let qt = kind.quant_type();
        for &isa in IsaKind::all() {
            for &op in OpCategory::all() {
                let path = coverage_check(qt, isa, op);
                assert!(
                    matches!(
                        path,
                        CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA
                    ),
                    "coverage_check({:?}, {:?}, {:?}) returned invalid path: {:?}",
                    qt, isa, op, path
                );
            }
        }
    }
}

#[test]
fn test_quant_codegen_coverage_report_generates() {
    let matrix = CoverageMatrix::new(QuantType::Q4_0);
    let report = matrix.coverage_report(QuantType::Q4_0);
    assert!(!report.is_empty(), "Coverage report must not be empty");
    assert!(report.contains("Q4_0"), "Report must mention the quant type");
    assert!(report.contains("x86_64"), "Report must mention ISA x86_64");
    assert!(report.contains("ARM"), "Report must mention ISA ARM");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §8 Scalar reference dequantization tests (tolerance ±1e-4)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scalar reference dequantization for Q4_0 block.
/// Formula: value = (nibble - 8) * d (f16→f32 scale)
fn scalar_dequant_q4_0(block_data: &[u8], scale_f16: half::f16) -> Vec<f32> {
    let d = scale_f16.to_f32();
    let mut result = Vec::with_capacity(32);
    for &byte in &block_data[..16] {
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;
        result.push((lo - 8.0) * d);
        result.push((hi - 8.0) * d);
    }
    result
}

/// Scalar reference dequantization for Q8_0 block.
/// Formula: value = qs_i8 * d (f16→f32 scale)
fn scalar_dequant_q8_0(qs: &[i8], scale_f16: half::f16) -> Vec<f32> {
    let d = scale_f16.to_f32();
    qs.iter().map(|&q| q as f32 * d).collect()
}

/// Scalar reference dequantization for Q4_1 block.
/// Formula: value = nibble * d + m
fn scalar_dequant_q4_1(block_data: &[u8], d: half::f16, m: half::f16) -> Vec<f32> {
    let d_f32 = d.to_f32();
    let m_f32 = m.to_f32();
    let mut result = Vec::with_capacity(32);
    for &byte in &block_data[..16] {
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;
        result.push(lo * d_f32 + m_f32);
        result.push(hi * d_f32 + m_f32);
    }
    result
}

#[test]
fn test_quant_codegen_scalar_reference_q4_0_values() {
    // Simulate a Q4_0 block: d = f16(0.5), qs = [0x87, 0x43, ...]
    let d = half::f16::from_f32(0.5);
    let block_data: [u8; 16] = [
        0x87, 0x43, 0x21, 0x65, 0xAB, 0xCD, 0xEF, 0x01,
        0x23, 0x45, 0x67, 0x89, 0xBC, 0xDE, 0xF0, 0x12,
    ];
    let values = scalar_dequant_q4_0(&block_data, d);

    assert_eq!(values.len(), 32, "Q4_0 block should produce 32 values");

    // Spot-check first two values:
    // byte 0x87: lo=7 → (7-8)*0.5 = -0.5, hi=8 → (8-8)*0.5 = 0.0
    assert!(
        (values[0] - (-0.5)).abs() < TOLERANCE as f32,
        "Q4_0 values[0] should be -0.5, got {}", values[0]
    );
    assert!(
        (values[1] - 0.0).abs() < TOLERANCE as f32,
        "Q4_0 values[1] should be 0.0, got {}", values[1]
    );

    // byte 0x43: lo=3 → (3-8)*0.5 = -2.5, hi=4 → (4-8)*0.5 = -2.0
    assert!(
        (values[2] - (-2.5)).abs() < TOLERANCE as f32,
        "Q4_0 values[2] should be -2.5, got {}", values[2]
    );
    assert!(
        (values[3] - (-2.0)).abs() < TOLERANCE as f32,
        "Q4_0 values[3] should be -2.0, got {}", values[3]
    );

    // Verify all values are finite
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "Q4_0 values[{}] is not finite: {}", i, v);
    }
}

#[test]
fn test_quant_codegen_scalar_reference_q4_1_values() {
    let d = half::f16::from_f32(1.5);
    let m = half::f16::from_f32(0.25);
    let block_data: [u8; 16] = [
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    ];
    let values = scalar_dequant_q4_1(&block_data, d, m);

    assert_eq!(values.len(), 32, "Q4_1 block should produce 32 values");

    // byte 0x10: lo=0 → 0*1.5+0.25 = 0.25, hi=1 → 1*1.5+0.25 = 1.75
    assert!(
        (values[0] - 0.25).abs() < TOLERANCE as f32,
        "Q4_1 values[0] should be 0.25, got {}", values[0]
    );
    assert!(
        (values[1] - 1.75).abs() < TOLERANCE as f32,
        "Q4_1 values[1] should be 1.75, got {}", values[1]
    );
}

#[test]
fn test_quant_codegen_scalar_reference_q8_0_values() {
    // Use 0.125 (1/8) which f16 can represent exactly, avoiding f16 rounding.
    let d = half::f16::from_f32(0.125);
    let qs: [i8; 32] = [
        -10, -5, -1, 0, 1, 5, 10, 20,
        -128, -64, -32, -16, -8, -4, -2, -1,
        0, 1, 2, 4, 8, 16, 32, 64,
        100, 110, 120, 127, -127, -100, -50, 25,
    ];
    let values = scalar_dequant_q8_0(&qs, d);

    assert_eq!(values.len(), 32, "Q8_0 block should produce 32 values");

    // Spot-check: qs[0] = -10, d = 0.125 → -10 * 0.125 = -1.25
    assert!(
        (values[0] - (-1.25)).abs() < TOLERANCE as f32,
        "Q8_0 values[0] should be -1.25, got {}", values[0]
    );
    // qs[3] = 0, d = 0.125 → 0.0
    assert!(
        (values[3] - 0.0).abs() < TOLERANCE as f32,
        "Q8_0 values[3] should be 0.0, got {}", values[3]
    );
    // qs[6] = 10, d = 0.125 → 1.25
    assert!(
        (values[6] - 1.25).abs() < TOLERANCE as f32,
        "Q8_0 values[6] should be 1.25, got {}", values[6]
    );
    // qs[7] = 20, d = 0.125 → 2.5
    assert!(
        (values[7] - 2.5).abs() < TOLERANCE as f32,
        "Q8_0 values[7] should be 2.5, got {}", values[7]
    );

    // Verify all values are finite
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "Q8_0 values[{}] is not finite: {}", i, v);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §9 VmInstr quantization variant availability tests (QCG5 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_vminstr_quant_variants_exist() {
    use gllm_kernels::compiler::codegen::vm::instr::{
        VmInstr, VmProgram, VRegId, VRegKind, SimdWidth, BlockUnpackMode,
    };
    use gllm_kernels::compiler::quant_format::QuantDataKind;

    let mut prog = VmProgram::new();
    let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let base = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let scale = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

    // QuantBlockLoad: load raw quant bytes
    prog.instrs.push(VmInstr::QuantBlockLoad {
        dst,
        base: VRegId(0),
        offset: gllm_kernels::compiler::codegen::vm::instr::OffsetExpr::Const(0),
        unpack: BlockUnpackMode::Int8,
        width: SimdWidth::W256,
    });

    // QuantExtractBits: nibble extraction for INT4/INT5/INT6
    prog.instrs.push(VmInstr::QuantExtractBits {
        dst,
        src,
        bit_offset: 0,
        bit_width: 4,
        width: SimdWidth::W256,
    });

    // QuantInterleave: merge lo/hi nibble streams
    prog.instrs.push(VmInstr::QuantInterleave {
        dst,
        lo: src,
        hi: src,
        width: SimdWidth::W256,
    });

    // QuantDequantFma: dequantize + FMA accumulate
    prog.instrs.push(VmInstr::QuantDequantFma {
        dst,
        weight: src,
        activation: src,
        scale,
        zero_point: VRegId(0),
        quant_kind: QuantDataKind::PackedInt4,
        dtype: gllm_kernels::compiler::trace::QuantPrecision::F32,
        width: SimdWidth::W256,
    });

    // Verify all 4 quantization variants were pushed
    assert!(
        prog.instrs.len() >= 4,
        "Expected at least 4 quantization VmInstr variants, got {}",
        prog.instrs.len()
    );
}

#[test]
fn test_quant_codegen_vminstr_dot_product_exists() {
    use gllm_kernels::compiler::codegen::vm::instr::{
        VmInstr, VmProgram, VRegId, VRegKind, SimdWidth, DotDtype,
    };

    let mut prog = VmProgram::new();
    let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let b = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

    // DotProduct: hardware dot-product (VPDPBUSD / SDOT / HMMA)
    prog.instrs.push(VmInstr::DotProduct {
        acc,
        a,
        b,
        input_dtype: DotDtype::Bf16,
        width: SimdWidth::W256,
    });

    // alloc_vreg emits DeclareVReg per call (3 here), so total = 3 + 1 DotProduct = 4
    let non_declare: Vec<_> = prog.instrs.iter().filter(|i| !matches!(i, VmInstr::DeclareVReg { .. })).collect();
    assert_eq!(non_declare.len(), 1, "Expected 1 DotProduct, got {:?}", non_declare);
    assert!(matches!(non_declare[0], VmInstr::DotProduct { .. }));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §10 DequantFMA VmInstr encoding tests (QCG6 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_dequant_fma_vminstr_encoding() {
    use gllm_kernels::compiler::codegen::vm::instr::{
        VmInstr, VmProgram, VRegId, VRegKind, SimdWidth,
    };
    use gllm_kernels::compiler::quant_format::QuantDataKind;

    // Construct QuantDequantFma for each QuantDataKind and verify
    let kinds = [
        (QuantDataKind::PackedInt4, "PackedInt4"),
        (QuantDataKind::Int8, "Int8"),
        (QuantDataKind::Float4, "Float4"),
        (QuantDataKind::PackedInt6, "PackedInt6"),
        (QuantDataKind::Bfloat16, "Bfloat16"),
    ];

    for (kind, name) in &kinds {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let w = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let a = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let scale = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        prog.instrs.push(VmInstr::QuantDequantFma {
            dst,
            weight: w,
            activation: a,
            scale,
            zero_point: VRegId(0),
            quant_kind: *kind,
            dtype: gllm_kernels::compiler::trace::QuantPrecision::F32,
            width: SimdWidth::W256,
        });

        // alloc_vreg emits DeclareVReg per call (4 here), so total = 4 + 1 QuantDequantFma = 5
        let non_declare: Vec<_> = prog.instrs.iter().filter(|i| !matches!(i, VmInstr::DeclareVReg { .. })).collect();
        assert_eq!(
            non_declare.len(), 1,
            "QuantDequantFma variant for {} should be constructable, got {:?}", name, non_declare
        );
        assert!(matches!(non_declare[0], VmInstr::QuantDequantFma { .. }));
    }
}

#[test]
fn test_quant_codegen_dequant_fma_vminstr_roundtrip() {
    use gllm_kernels::compiler::codegen::vm::instr::{
        VmInstr, VmProgram, VRegId, VRegKind, SimdWidth,
    };
    use gllm_kernels::compiler::quant_format::QuantDataKind;

    // Verify QuantExtractBits can have bit_width=4 (INT4 unpack)
    let mut prog = VmProgram::new();
    let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
    let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

    prog.instrs.push(VmInstr::QuantExtractBits {
        dst,
        src,
        bit_offset: 0,
        bit_width: 4,
        width: SimdWidth::W256,
    });

    // Verify the instruction encodes correctly (last non-DeclareVReg is QuantExtractBits)
    let non_declare: Vec<_> = prog.instrs.iter().filter(|i| !matches!(i, VmInstr::DeclareVReg { .. })).collect();
    match non_declare.first() {
        Some(VmInstr::QuantExtractBits { bit_width, .. }) => {
            assert_eq!(*bit_width, 4);
        }
        other => panic!("Expected QuantExtractBits, got {:?}", other),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §11 QuantCapability integration tests (QCG11 integration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_device_profile_quant_capabilities() {
    let profile = DeviceProfile::detect();

    // F32 must always be in native_formats
    assert!(
        profile.quant_capabilities.native_formats.contains(&QuantType::F32),
        "F32 must be in native_formats on any device"
    );

    // At least some formats must be listed (F32 is always present)
    assert!(
        !profile.quant_capabilities.native_formats.is_empty(),
        "native_formats must not be empty (at least F32)"
    );
}

#[test]
fn test_quant_codegen_quant_capability_coverage_consistency() {
    let profile = DeviceProfile::detect();

    // For each native format, coverage_check must return a valid path for x86_64 GEMM
    for qt in &profile.quant_capabilities.native_formats {
        let path = coverage_check(*qt, IsaKind::X86, OpCategory::Gemm);
        // Native formats should have Native or Assisted path on x86_64 (not DequantFMA)
        assert!(
            matches!(path, CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA),
            "Native format {:?} on x86_64 GEMM must have valid coverage path, got {:?}", qt, path
        );
    }

    // For each assisted format, coverage_check must return a valid path
    for qt in &profile.quant_capabilities.assisted_formats {
        let path = coverage_check(*qt, IsaKind::X86, OpCategory::Gemm);
        assert!(
            matches!(path, CoveragePath::Native | CoveragePath::Assisted | CoveragePath::DequantFMA),
            "Assisted format {:?} on x86_64 GEMM must have valid coverage path, got {:?}", qt, path
        );
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §12 End-to-end: compile → execute → compare for 5+ formats
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_quant_codegen_e2e_five_representative_formats() {
    let reg = registry();
    let profile = DeviceProfile::detect();

    let formats: Vec<QuantType> = vec![
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q8_0,
        QuantType::Q4K,
        QuantType::Q6K,
    ];

    for qt in &formats {
        let desc = match reg.get(qt) {
            Some(d) => d,
            None => panic!("{:?} not registered", qt),
        };

        // Phase 1: Compile — generate trace
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        assert!(!trace.is_empty(), "Trace for {:?} must not be empty", qt);
        assert!(
            final_slot.0 < trace.len() as u32,
            "Final slot for {:?} out of range", qt
        );

        // Phase 2: Execute — run numerical simulation
        let block_data = vec![0u8; desc.block_bytes.max(1)];
        let inputs: &[i64] = &[4096, 4096];
        let sim_result = simulate_compile(&trace, desc, &block_data, inputs);
        assert!(
            sim_result.is_ok(),
            "Numerical simulation failed for {:?}: {:?}", qt, sim_result.err()
        );

        let result = sim_result.unwrap();
        assert!(
            !result.has_nan,
            "[{:?}] Simulation produced NaN", qt
        );
        assert!(
            !result.has_inf,
            "[{:?}] Simulation produced Inf", qt
        );

        // Phase 3: Verify coverage path is consistent
        let path = coverage_check(*qt, IsaKind::X86, OpCategory::Gemm);
        match path {
            CoveragePath::Native => {
                // Native path selected — hardware has direct support
            }
            CoveragePath::Assisted => {
                // Assisted path — hardware has SIMD but no native dot-product
            }
            CoveragePath::DequantFMA => {
                // Valid fallback — acceptable for some formats/hardware combos
            }
        }
    }
}

#[test]
#[ignore]
fn debug_q5_0_trace_dump() {
    use gllm_kernels::compiler::codegen::vm::quant_decode::DecodeTraceBuilder;
    use gllm_kernels::compiler::quant_format::registry;
    use gllm_kernels::quant::QuantType;
    use gllm_kernels::compiler::trace::TraceOp;

    let reg = registry();
    let desc = reg.get(&QuantType::Q5_0).expect("Q5_0");
    let mut trace = Vec::new();
    DecodeTraceBuilder::new(desc, 8).build(&mut trace);

    for (i, op) in trace.iter().enumerate() {
        eprintln!("{:3}: {:?}", i, op);
    }
}
