//! Shared codegen completeness tests.
//!
//! Ensures every non-metadata OpKind either produces real machine code
//! (code_size > 4) or returns an explicit Err — never a silent NOP.
//! Runs under both `jit-x86` and `jit-aarch64` feature flags.

#[cfg(test)]
mod completeness_tests {
    use crate::compiler::graph::{CompilerGraph, OpKind, TensorId};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::compiler::codegen::emitter::MachineCodeEmitter;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::DType;
    use crate::compiler::registry::ScalarOpRegistry;
    use std::collections::HashMap;

    const SEQ: usize = 4;
    const HIDDEN: usize = 384;
    const HEADS: usize = 6;
    const HEAD_DIM: usize = 64;

    fn build_graph(
        kind: OpKind,
        input_shapes: &[(&str, Vec<usize>)],
        output_shapes: &[(&str, Vec<usize>)],
    ) -> (CompilerGraph, crate::compiler::graph::OpId) {
        let mut g = CompilerGraph::new();
        let inputs: Vec<TensorId> = input_shapes
            .iter()
            .map(|(name, shape)| g.add_tensor(name, shape.clone(), DType::F32))
            .collect();
        let outputs: Vec<TensorId> = output_shapes
            .iter()
            .map(|(name, shape)| g.add_tensor(name, shape.clone(), DType::F32))
            .collect();
        g.inputs = inputs.clone();
        g.outputs = outputs.clone();
        let op_id = g.add_op(kind, inputs, outputs, "test_op");
        (g, op_id)
    }

    /// Generic try_compile: takes any MachineCodeEmitter.
    fn try_compile_with<F: FnOnce(&DeviceProfile) -> Box<dyn MachineCodeEmitter>>(
        kind: OpKind,
        input_shapes: &[(&str, Vec<usize>)],
        output_shapes: &[(&str, Vec<usize>)],
        make_emitter: F,
    ) -> Result<usize, String> {
        let (graph, op_id) = build_graph(kind, input_shapes, output_shapes);
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![op_id],
            }],
            op_to_group,
        };
        let alloc = BufferAllocation {
            slots: vec![],
            total_bytes: 0,
            num_tensors: 0,
            bytes_saved: 0,
        };
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let mut emitter = make_emitter(&profile);
        let output = emitter.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry))?;
        Ok(output.code.len())
    }

    fn assert_no_silent_nop(label: &str, result: Result<usize, String>) {
        match result {
            Ok(size) => {
                assert!(
                    size > 4,
                    "{}: produced only {} bytes (silent NOP detected!)",
                    label, size
                );
                eprintln!("[completeness] {}: OK ({} bytes)", label, size);
            }
            Err(msg) => {
                eprintln!("[completeness] {}: Err (expected) — {}", label, msg);
                // Explicit error is fine — the op is not silently ignored.
            }
        }
    }

    fn compute_op_cases() -> Vec<(&'static str, OpKind, Vec<(&'static str, Vec<usize>)>, Vec<(&'static str, Vec<usize>)>)> {
        vec![
            // ── Normalization ──
            ("RmsNorm", OpKind::RmsNorm { eps: 1e-5 },
                vec![("input", vec![HIDDEN])], vec![("output", vec![HIDDEN])]),
            ("LayerNorm", OpKind::LayerNorm { eps: 1e-5 },
                vec![("input", vec![HIDDEN])], vec![("output", vec![HIDDEN])]),

            // ── Linear algebra ──
            ("Gemm", OpKind::Gemm { m: SEQ, n: HIDDEN, k: HIDDEN },
                vec![("a", vec![SEQ, HIDDEN]), ("b", vec![HIDDEN, HIDDEN])],
                vec![("c", vec![SEQ, HIDDEN])]),
            ("GemmBias", OpKind::GemmBias { m: SEQ, n: HIDDEN, k: HIDDEN },
                vec![("a", vec![SEQ, HIDDEN]), ("b", vec![HIDDEN, HIDDEN]), ("bias", vec![HIDDEN])],
                vec![("c", vec![SEQ, HIDDEN])]),

            // ── Activations (unary) ──
            ("Silu", OpKind::Silu,
                vec![("input", vec![HIDDEN])], vec![("output", vec![HIDDEN])]),
            ("Gelu", OpKind::Gelu,
                vec![("input", vec![HIDDEN])], vec![("output", vec![HIDDEN])]),

            // ── Activations (binary gated) ──
            ("SwiGlu", OpKind::SwiGlu,
                vec![("gate", vec![HIDDEN]), ("up", vec![HIDDEN])],
                vec![("output", vec![HIDDEN])]),
            ("GeGlu", OpKind::GeGlu,
                vec![("gate", vec![HIDDEN]), ("up", vec![HIDDEN])],
                vec![("output", vec![HIDDEN])]),

            // ── Attention ──
            ("Softmax", OpKind::Softmax,
                vec![("input", vec![HIDDEN])], vec![("output", vec![HIDDEN])]),
            ("MultiHeadAttention", OpKind::MultiHeadAttention { seq_len: SEQ, num_heads: HEADS, head_dim: HEAD_DIM },
                vec![("q", vec![SEQ, HEADS * HEAD_DIM]), ("k", vec![SEQ, HEADS * HEAD_DIM]), ("v", vec![SEQ, HEADS * HEAD_DIM])],
                vec![("output", vec![SEQ, HEADS * HEAD_DIM])]),
            ("RoPE", OpKind::RoPE { head_dim: HEAD_DIM, theta: 10000.0 },
                vec![("input", vec![SEQ, HEADS * HEAD_DIM])],
                vec![("output", vec![SEQ, HEADS * HEAD_DIM])]),

            // ── Elementwise (binary) ──
            ("Add", OpKind::Add,
                vec![("a", vec![HIDDEN]), ("b", vec![HIDDEN])],
                vec![("output", vec![HIDDEN])]),
            ("Mul", OpKind::Mul,
                vec![("a", vec![HIDDEN]), ("b", vec![HIDDEN])],
                vec![("output", vec![HIDDEN])]),
            ("Residual", OpKind::Residual,
                vec![("x", vec![HIDDEN]), ("residual", vec![HIDDEN])],
                vec![("output", vec![HIDDEN])]),

            // ── Pooling ──
            ("MeanPool", OpKind::MeanPool { seq_len: SEQ, hidden: HIDDEN },
                vec![("input", vec![SEQ, HIDDEN])],
                vec![("output", vec![HIDDEN])]),

            // ── L2 Normalize ──
            ("L2Normalize", OpKind::L2Normalize { hidden: HIDDEN },
                vec![("input", vec![HIDDEN])], vec![("output", vec![HIDDEN])]),

            // ── Quantization ──
            ("QuantGemm", OpKind::QuantGemm { m: SEQ, n: HIDDEN, k: HIDDEN, block_size: 32, bits: 4 },
                vec![("a", vec![SEQ, HIDDEN]), ("b_quant", vec![HIDDEN * HIDDEN / 2])],
                vec![("c", vec![SEQ, HIDDEN])]),
            ("Dequantize", OpKind::Dequantize { num_elements: HIDDEN, block_size: 32, bits: 4 },
                vec![("quant", vec![HIDDEN / 2])],
                vec![("output", vec![HIDDEN])]),
        ]
    }

    fn metadata_op_cases() -> Vec<(&'static str, OpKind, Vec<(&'static str, Vec<usize>)>, Vec<(&'static str, Vec<usize>)>)> {
        vec![
            ("Reshape", OpKind::Reshape { target_shape: vec![SEQ, HIDDEN] },
                vec![("input", vec![SEQ * HIDDEN])],
                vec![("output", vec![SEQ, HIDDEN])]),
            ("Transpose", OpKind::Transpose { perm: vec![1, 0] },
                vec![("input", vec![SEQ, HIDDEN])],
                vec![("output", vec![HIDDEN, SEQ])]),
        ]
    }

    // ── x86_64 tests ──────────────────────────────────────────────────────────

    #[cfg(all(target_arch = "x86_64", feature = "jit-x86"))]
    mod x86 {
        use super::*;
        use crate::compiler::codegen::x86_64::jit::X86CodeGen;

        fn try_compile(kind: OpKind, inputs: &[(&str, Vec<usize>)], outputs: &[(&str, Vec<usize>)]) -> Result<usize, String> {
            try_compile_with(kind, inputs, outputs, |profile| {
                Box::new(X86CodeGen::new(profile))
            })
        }

        #[test]
        fn codegen_completeness_no_silent_nop_x86() {
            let mut failures = Vec::new();
            for (label, kind, inputs, outputs) in &compute_op_cases() {
                let result = try_compile(kind.clone(), inputs, outputs);
                match &result {
                    Ok(size) if *size <= 4 => {
                        failures.push(format!("{}: silent NOP ({} bytes)", label, size));
                    }
                    _ => {}
                }
                assert_no_silent_nop(label, result);
            }
            if !failures.is_empty() {
                panic!("x86 completeness violations:\n  {}", failures.join("\n  "));
            }
        }

        #[test]
        fn codegen_completeness_metadata_ops_x86() {
            for (label, kind, inputs, outputs) in &metadata_op_cases() {
                let result = try_compile(kind.clone(), inputs, outputs);
                match &result {
                    Ok(_) => eprintln!("[completeness] {} (metadata/x86): OK", label),
                    Err(msg) => eprintln!("[completeness] {} (metadata/x86): Err — {}", label, msg),
                }
                // Both Ok (even small code) and Err are acceptable for metadata ops.
            }
        }
    }

    // ── aarch64 tests ─────────────────────────────────────────────────────────

    #[cfg(all(target_arch = "aarch64", feature = "jit-aarch64"))]
    mod aarch64 {
        use super::*;
        use crate::compiler::codegen::aarch64_dynasm::jit::DynasmAArch64CodeGen;

        fn try_compile(kind: OpKind, inputs: &[(&str, Vec<usize>)], outputs: &[(&str, Vec<usize>)]) -> Result<usize, String> {
            try_compile_with(kind, inputs, outputs, |profile| {
                Box::new(DynasmAArch64CodeGen::new(profile))
            })
        }

        #[test]
        fn codegen_completeness_no_silent_nop_aarch64() {
            let mut failures = Vec::new();
            for (label, kind, inputs, outputs) in &compute_op_cases() {
                let result = try_compile(kind.clone(), inputs, outputs);
                match &result {
                    Ok(size) if *size <= 4 => {
                        failures.push(format!("{}: silent NOP ({} bytes)", label, size));
                    }
                    _ => {}
                }
                assert_no_silent_nop(label, result);
            }
            if !failures.is_empty() {
                panic!("aarch64 completeness violations:\n  {}", failures.join("\n  "));
            }
        }

        #[test]
        fn codegen_completeness_metadata_ops_aarch64() {
            for (label, kind, inputs, outputs) in &metadata_op_cases() {
                let result = try_compile(kind.clone(), inputs, outputs);
                match &result {
                    Ok(_) => eprintln!("[completeness] {} (metadata/aarch64): OK", label),
                    Err(msg) => eprintln!("[completeness] {} (metadata/aarch64): Err — {}", label, msg),
                }
                // Both Ok (even small code) and Err are acceptable for metadata ops.
            }
        }
    }
}
