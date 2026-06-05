#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use crate::compiler::codegen::vm::plan_lower::compile_layer;
    use crate::compiler::executable::CompiledLayer;
    use crate::compiler::graph::*;
    use crate::compiler::fusion::*;
    use crate::compiler::buffer_alloc::*;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::dispatch::DeviceProfile;
    use crate::types::DType;
    use std::collections::HashMap;

    const N: usize = 32; // 4 AVX2 vectors

    fn compile(graph: &CompilerGraph, plan: &FusionPlan, alloc: &BufferAllocation) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(plan, graph, alloc, &exec_plan, Some(&registry)).unwrap();
        assert!(!output.code.is_empty(), "compile_layer produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    fn build_unary(kind: OpKind) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[N], DType::F32);
        let out = g.add_tensor_concrete("output", &[N], DType::F32);
        g.inputs = vec![inp];
        g.outputs = vec![out];
        let op = g.add_op(kind, vec![inp], vec![out], "op");
        let mut m = HashMap::new();
        m.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::LoopFusion, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: m,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    fn build_binary(kind: OpKind) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[N], DType::F32);
        let b = g.add_tensor_concrete("b", &[N], DType::F32);
        let out = g.add_tensor_concrete("output", &[N], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![out];
        let op = g.add_op(kind, vec![a, b], vec![out], "op");
        let mut m = HashMap::new();
        m.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::LoopFusion, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: m,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    fn test_input() -> Vec<f32> {
        (0..N).map(|i| (i as f32 - N as f32 / 2.0) * 0.1).collect()
    }

    fn test_input_b() -> Vec<f32> {
        (0..N).map(|i| (i as f32 + 1.0) * 0.05).collect()
    }

    /// Execute unary op via CompiledLayerFn ABI.
    /// output 在第 8 个参数 = [rsp+8] = [rbp+24]。
    unsafe fn exec_unary(layer: &CompiledLayer, input: &[f32], output: &mut [f32]) {
        let mut telemetry = vec![0u8; crate::compiler::graph::telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES];
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,       // rdi: activation
            std::ptr::null(),                   // rsi: weights (unused)
            std::ptr::null_mut(),               // rdx: kv_cache
            std::ptr::null(),                   // rcx: positions
            std::ptr::null(),                   // r8: seq_lens
            1,                                  // r9: batch_size
            input.len(),                        // [rsp+0]: seq_len (元素数)
            output.as_mut_ptr() as *mut u8,     // [rsp+8] → [rbp+24]: output
            std::ptr::null_mut(),               // [rsp+16]: scratchpad
            telemetry.as_mut_ptr() as *mut u8,  // [rsp+24]: telemetry
        );
    }

    /// Execute binary op via CompiledLayerFn ABI.
    unsafe fn exec_binary(layer: &CompiledLayer, a: &[f32], b: &[f32], output: &mut [f32]) {
        let mut telemetry = vec![0u8; crate::compiler::graph::telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES];
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,            // rdi: activation (input A)
            b.as_ptr() as *const u8,            // rsi: weights (input B)
            std::ptr::null_mut(),               // rdx: kv_cache
            std::ptr::null(),                   // rcx: positions
            std::ptr::null(),                   // r8: seq_lens
            1,                                  // r9: batch_size
            a.len(),                            // [rsp+0]: seq_len
            output.as_mut_ptr() as *mut u8,     // [rsp+8] → [rbp+24]: output
            std::ptr::null_mut(),               // [rsp+16]: scratchpad
            telemetry.as_mut_ptr() as *mut u8,  // [rsp+24]: telemetry
        );
    }

    fn ref_silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
    fn ref_gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
    }

    #[test]
    fn test_vm_e2e_silu() {
        let (g, p, a) = build_unary(OpKind::Silu);
        let layer = compile(&g, &p, &a);
        let input = test_input();
        let mut output = vec![0.0f32; N];
        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(diff < 1e-2, "SiLU[{i}]: got={}, expected={}, diff={diff}", output[i], expected);
        }
    }

    #[test]
    fn test_vm_e2e_add() {
        let (g, p, a) = build_binary(OpKind::Add);
        let layer = compile(&g, &p, &a);
        let input_a = test_input();
        let input_b = test_input_b();
        let mut output = vec![0.0f32; N];
        unsafe { exec_binary(&layer, &input_a, &input_b, &mut output) };

        let mut max_diff = 0.0f32;
        for i in 0..N {
            let expected = input_a[i] + input_b[i];
            let diff = (output[i] - expected).abs();
            max_diff = max_diff.max(diff);
            if diff >= 1e-5 {
                eprintln!("Add[{i}]: got={}, expected={}, diff={diff}", output[i], expected);
            }
        }
        assert!(max_diff < 1e-5, "Add max_diff={max_diff}");
    }

    #[test]
    fn test_vm_e2e_mul() {
        let (g, p, a) = build_binary(OpKind::Mul);
        let layer = compile(&g, &p, &a);
        let input_a = test_input();
        let input_b = test_input_b();
        let mut output = vec![0.0f32; N];
        unsafe { exec_binary(&layer, &input_a, &input_b, &mut output) };

        for i in 0..N {
            let expected = input_a[i] * input_b[i];
            let diff = (output[i] - expected).abs();
            assert!(diff < 1e-5, "Mul[{i}]: got={}, expected={}, diff={diff}", output[i], expected);
        }
    }

    // ── SwiGLU ────────────────────────────────────────────────────────

    fn ref_swiglu(gate: f32, up: f32) -> f32 {
        let silu = gate / (1.0 + (-gate).exp());
        silu * up
    }

    fn ref_swiglu_clipped(gate: f32, up: f32, limit: f32) -> f32 {
        let g = gate.max(-limit).min(limit);
        let u = up.max(-limit).min(limit);
        let silu = g / (1.0 + (-g).exp());
        silu * u
    }

    /// Unclipped SwiGLU baseline — serves as the control against which
    /// the clipped variant must differ for large-magnitude inputs.
    #[test]
    fn test_vm_e2e_swiglu() {
        let (g, p, a) = build_binary(OpKind::SwiGlu);
        let layer = compile(&g, &p, &a);
        let gate_in = test_input();
        let up_in = test_input_b();
        let mut output = vec![0.0f32; N];
        unsafe { exec_binary(&layer, &gate_in, &up_in, &mut output) };

        for i in 0..N {
            let expected = ref_swiglu(gate_in[i], up_in[i]);
            let diff = (output[i] - expected).abs();
            assert!(diff < 1e-4,
                "SwiGlu[{i}]: got={}, expected={}, diff={diff}",
                output[i], expected);
        }
    }

    /// Clipped SwiGLU (gpt-oss-20b). Verifies the JIT-emitted clamp+silu+mul
    /// matches the scalar reference for both in-range (|x| <= limit) and
    /// saturating (|x| > limit) inputs.
    #[test]
    fn test_vm_e2e_swiglu_clipped_limit7() {
        const LIMIT: f32 = 7.0;
        let (g, p, a) = build_binary(OpKind::SwiGluClipped { limit: LIMIT });
        let layer = compile(&g, &p, &a);

        // Inputs spanning the clip boundary: [-100, -7, -1, 0, 1, 7, 100, ...]
        // paired against `up` values that also exercise ± saturation.
        let mut gate_in = vec![0.0f32; N];
        let mut up_in = vec![0.0f32; N];
        let sample_gate = [-100.0f32, -50.0, -7.0, -1.0, 0.0, 1.0, 7.0, 50.0];
        let sample_up   = [100.0f32,   50.0,  -7.0,  1.0, 2.0, 3.0, 4.0, -50.0];
        for i in 0..N {
            gate_in[i] = sample_gate[i % sample_gate.len()];
            up_in[i]   = sample_up[i % sample_up.len()];
        }

        let mut output = vec![0.0f32; N];
        unsafe { exec_binary(&layer, &gate_in, &up_in, &mut output) };

        for i in 0..N {
            let expected = ref_swiglu_clipped(gate_in[i], up_in[i], LIMIT);
            let diff = (output[i] - expected).abs();
            assert!(diff < 1e-4,
                "SwiGluClipped[{i}] (gate={}, up={}): got={}, expected={}, diff={diff}",
                gate_in[i], up_in[i], output[i], expected);
        }
    }

    /// Verifies the limit is actually honoured by the JIT: clipped output
    /// at large magnitudes must deviate from the unclipped baseline.
    #[test]
    fn test_vm_e2e_swiglu_clipped_differs_from_unclipped() {
        const LIMIT: f32 = 7.0;

        // Build two separate layers on the same large-magnitude inputs.
        let gate_in = vec![50.0f32; N];
        let up_in = vec![50.0f32; N];

        let mut out_unclipped = vec![0.0f32; N];
        {
            let (g, p, a) = build_binary(OpKind::SwiGlu);
            let layer = compile(&g, &p, &a);
            unsafe { exec_binary(&layer, &gate_in, &up_in, &mut out_unclipped) };
        }

        let mut out_clipped = vec![0.0f32; N];
        {
            let (g, p, a) = build_binary(OpKind::SwiGluClipped { limit: LIMIT });
            let layer = compile(&g, &p, &a);
            unsafe { exec_binary(&layer, &gate_in, &up_in, &mut out_clipped) };
        }

        // Unclipped ≈ 50 * 50 = 2500; clipped ≈ silu(7) * 7 ≈ 48.95.
        let expected_clipped = ref_swiglu_clipped(50.0, 50.0, LIMIT);
        let expected_unclipped = ref_swiglu(50.0, 50.0);

        for i in 0..N {
            assert!((out_clipped[i] - expected_clipped).abs() < 1e-3,
                "clipped[{i}]={}, expected={}", out_clipped[i], expected_clipped);
            assert!((out_unclipped[i] - expected_unclipped).abs() < 1e-1,
                "unclipped[{i}]={}, expected~{}", out_unclipped[i], expected_unclipped);
            assert!((out_unclipped[i] - out_clipped[i]).abs() > 100.0,
                "clipped and unclipped must differ substantially at |x|=50: \
                 clipped={}, unclipped={}",
                out_clipped[i], out_unclipped[i]);
        }
    }

    /// Verifies per-op limit parameter actually propagates through the JIT.
    /// Two layers built with distinct `limit` values on the same inputs must
    /// produce distinct outputs.
    #[test]
    fn test_vm_e2e_swiglu_clipped_limit_parameterised() {
        // Input magnitude = 10, so limit=3 saturates harder than limit=7.
        let gate_in = vec![10.0f32; N];
        let up_in = vec![10.0f32; N];

        let mut out_limit3 = vec![0.0f32; N];
        {
            let (g, p, a) = build_binary(OpKind::SwiGluClipped { limit: 3.0 });
            let layer = compile(&g, &p, &a);
            unsafe { exec_binary(&layer, &gate_in, &up_in, &mut out_limit3) };
        }

        let mut out_limit7 = vec![0.0f32; N];
        {
            let (g, p, a) = build_binary(OpKind::SwiGluClipped { limit: 7.0 });
            let layer = compile(&g, &p, &a);
            unsafe { exec_binary(&layer, &gate_in, &up_in, &mut out_limit7) };
        }

        let expected_3 = ref_swiglu_clipped(10.0, 10.0, 3.0);
        let expected_7 = ref_swiglu_clipped(10.0, 10.0, 7.0);

        for i in 0..N {
            assert!((out_limit3[i] - expected_3).abs() < 1e-4,
                "limit3[{i}]={}, expected={}", out_limit3[i], expected_3);
            assert!((out_limit7[i] - expected_7).abs() < 1e-4,
                "limit7[{i}]={}, expected={}", out_limit7[i], expected_7);
        }

        // Sanity: outputs with different limits must differ.
        assert!((out_limit3[0] - out_limit7[0]).abs() > 1e-2,
            "limit=3 output must differ from limit=7: {} vs {}",
            out_limit3[0], out_limit7[0]);
    }

    #[test]
    fn test_vm_e2e_gemm() {
        let (m, n, k) = (4, 8, 16);
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm",
        );
        let mut map = HashMap::new();
        map.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: map,
        };
        let alloc = BufferAllocation::default();
        let layer = compile(&g, &plan, &alloc);

        // 填充数据
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.007).cos()).collect();
        let mut c_jit = vec![0.0f32; m * n];

        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8,       // rdi: A matrix
                b_data.as_ptr() as *const u8,       // rsi: B matrix
                std::ptr::null_mut(),               // rdx: kv_cache
                std::ptr::null(),                   // rcx: positions
                std::ptr::null(),                   // r8: seq_lens
                1,                                  // r9: batch_size
                m,                                  // [rsp+0]: seq_len (= M dim)
                c_jit.as_mut_ptr() as *mut u8,      // [rsp+8] → [rbp+24]: output
                std::ptr::null_mut(),               // [rsp+16]: scratchpad
                std::ptr::null_mut(),               // [rsp+24]: telemetry (no SiLU op → safe to null)
            );
        }

        // 标量参考
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k { sum += a_data[i*k+p] * b_data[p*n+j]; }
                c_ref[i*n+j] = sum;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let diff = (c_jit[i*n+j] - c_ref[i*n+j]).abs();
                assert!(diff < k as f32 * 1e-5,
                    "GEMM[{i},{j}]: got={}, expected={}, diff={diff}", c_jit[i*n+j], c_ref[i*n+j]);
            }
        }
    }

    #[test]
    fn test_vm_compile_layer_code_size() {
        let (g, p, a) = build_unary(OpKind::Silu);
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&p, &g, &a, &exec_plan, Some(&registry)).unwrap();
        // SiLU kernel 应该产出实质性代码
        assert!(output.code.len() > 50, "code too small: {}", output.code.len());
        assert!(output.code.len() < 10000, "code too large: {}", output.code.len());
    }

    /// 复现 GELU 在大负输入下的数值 bug (cosine=0.02 vs PyTorch)。
    /// 用 InferenceCompiler::compile_graph 路径 (与真实推理一致), Symbolic
    /// seq_len + Concrete feature_dim。
    ///
    /// 当前 ignored: GELU 13-op trace 需要 14 个 Vec VReg 同时活跃 (slot 0-13),
    /// 加上 acc/sec 超出 AVX2 YMM 池 (13 可用)。真实推理路径走 emit_elementwise_inline
    /// 循环分摊 VReg 生命周期, 所以 compile_graph 成功 — 单测的 lower_elementwise
    /// 路径压力更集中。修复方向: 对 trace body 长度 > N 的场景强制使用临时 spill。
    #[test]
    fn test_vm_e2e_gelu_boundary() {
        use crate::compiler::{InferenceCompiler, SymDim};

        let feature = 32usize;  // < 4096 避免 RegAlloc 爆, 但仍走 plan_lower 路径
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(16) };
        let input = g.add_tensor("input", vec![seq_sym.clone(), SymDim::Concrete(feature)], DType::F32);
        let output = g.add_tensor("output", vec![seq_sym, SymDim::Concrete(feature)], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        g.add_op(OpKind::Gelu, vec![input], vec![output], "gelu");

        let mut compiler = InferenceCompiler::new();
        let layer = compiler.compile_graph(&g).expect("compile GELU");

        // 边界值: [-21.3, -10, -5, -1, 0, 1, 5, 21.3] + 0.3 filler
        let seq_actual = 1usize;
        let mut input_data = vec![0.3f32; 16 * feature];  // max seq buffer
        let boundary = [-21.3f32, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 21.3];
        for (i, &v) in boundary.iter().enumerate() { input_data[i] = v; }
        let mut output_data = vec![0.0f32; 16 * feature];

        unsafe {
            let f = layer.entry_point();
            f(
                input_data.as_ptr() as *const u8,
                std::ptr::null(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(),
                1, seq_actual,
                output_data.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }

        let mut expected = vec![0.0f32; 16 * feature];
        unsafe { gllm_scalar_ops::activations::scalar_gelu(
            input_data.as_ptr(), expected.as_mut_ptr(), 16 * feature,
        ) };

        eprintln!("GELU 边界值对比:");
        for i in 0..boundary.len() {
            eprintln!("  gelu({:+7.3}) = jit:{:+13.5} ref:{:+13.5} diff:{:+12.4e}",
                input_data[i], output_data[i], expected[i], output_data[i] - expected[i]);
        }

        for i in 0..boundary.len() {
            let diff = (output_data[i] - expected[i]).abs();
            let tol = 1e-2 * expected[i].abs().max(1.0);
            assert!(diff < tol,
                "GELU[{i}] input={} got={} expected={} diff={} tol={}",
                input_data[i], output_data[i], expected[i], diff, tol);
        }
    }

    /// 复现 gllm encoder 真实路径: Symbolic M GEMM。
    /// 期望: JIT 输出 ≈ scalar 参考。
    #[test]
    fn test_vm_e2e_gemm_symbolic_m() {
        use crate::compiler::SymDim;
        let (m_actual, n, k) = (4usize, 8usize, 16usize);
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(2048) };
        let a = g.add_tensor("A", vec![seq_sym.clone(), SymDim::Concrete(k)], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor("C", vec![seq_sym.clone(), SymDim::Concrete(n)], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op = g.add_op(
            OpKind::Gemm { m: seq_sym.clone(), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_sym",
        );
        let mut map = HashMap::new();
        map.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: map,
        };
        let alloc = BufferAllocation::default();
        let layer = compile(&g, &plan, &alloc);

        let a_data: Vec<f32> = (0..m_actual*k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.007).cos()).collect();
        let mut c_jit = vec![0.0f32; m_actual * n];

        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, m_actual, c_jit.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }

        let mut c_ref = vec![0.0f32; m_actual * n];
        for i in 0..m_actual { for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k { s += a_data[i*k+p] * b_data[p*n+j]; }
            c_ref[i*n+j] = s;
        } }

        eprintln!("c_jit[0..8]={:?}", &c_jit[..8]);
        eprintln!("c_ref[0..8]={:?}", &c_ref[..8]);
        for i in 0..m_actual { for j in 0..n {
            let diff = (c_jit[i*n+j] - c_ref[i*n+j]).abs();
            assert!(diff < k as f32 * 1e-5,
                "GEMM[{i},{j}]: got={}, expected={}, diff={diff}", c_jit[i*n+j], c_ref[i*n+j]);
        } }
    }

    /// 复现 E2E XLMR q_proj: K=1024 N=1024 Symbolic M GEMM。
    #[test]
    fn test_vm_e2e_gemm_symbolic_m_large() {
        use crate::compiler::SymDim;
        let (m_actual, n, k) = (16usize, 1024usize, 1024usize);
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(2048) };
        let a = g.add_tensor("A", vec![seq_sym.clone(), SymDim::Concrete(k)], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor("C", vec![seq_sym.clone(), SymDim::Concrete(n)], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op = g.add_op(
            OpKind::Gemm { m: seq_sym.clone(), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_sym_large",
        );
        let mut map = HashMap::new();
        map.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: map,
        };
        let alloc = BufferAllocation::default();
        let layer = compile(&g, &plan, &alloc);

        // Buffer padded to max_value to avoid OOB if JIT loops to max_value
        let m_max = 2048usize;
        let mut a_data: Vec<f32> = vec![0.0; m_max * k];
        for i in 0..m_actual*k { a_data[i] = (i as f32 * 0.001).sin(); }
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.0007).cos()).collect();
        let mut c_jit = vec![0.0f32; m_max * n];

        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, m_actual, c_jit.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }

        let non_zero = c_jit.iter().filter(|&&x| x != 0.0).count();
        eprintln!("c_jit[0..8]={:?}", &c_jit[..8]);
        eprintln!("non_zero_count={} / total={}", non_zero, m_actual * n);
        assert!(non_zero > 0, "all outputs zero! bug reproduced");

        let mut c_ref = vec![0.0f32; m_actual * n];
        for i in 0..m_actual { for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k { s += a_data[i*k+p] * b_data[p*n+j]; }
            c_ref[i*n+j] = s;
        } }
        for i in 0..m_actual { for j in 0..n {
            let diff = (c_jit[i*n+j] - c_ref[i*n+j]).abs();
            assert!(diff < k as f32 * 1e-4,
                "GEMM[{i},{j}]: got={}, expected={}, diff={diff}", c_jit[i*n+j], c_ref[i*n+j]);
        } }
    }

    /// 复现 E2E MHA 路径: Symbolic seq_len + online softmax。
    /// 验证 JIT attention 输出与 scalar 参考数值一致。
    /// 当前 GPR 池在 head_dim=8 时不够装下 attention 所有同时活跃 VReg
    /// (q_head/k_head/v_head/kv_v_base/o_head/q_row/k_row/v_row/o_row = 9 Ptr
    /// + 4 层嵌套 loop = 8 Counter/ByteOff ≥ 17 GPR VReg), 暂时 ignore,
    /// 待 Ptr VReg spill 或扩 GPR 池后启用。ARCH-REGALLOC-COUNTER-NOSPILL。
    #[test]
    fn test_vm_e2e_mha_symbolic() {
        use crate::compiler::SymDim;
        let (seq_actual, seq_max, num_heads, head_dim) = (4usize, 8usize, 1usize, 8usize);
        let num_kv_heads = num_heads;
        let h = num_heads * head_dim;
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(seq_max) };
        let q = g.add_tensor("q", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
        let k = g.add_tensor("k", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
        let v = g.add_tensor("v", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
        let out = g.add_tensor("out", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
        g.inputs = vec![q, k, v];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::MultiHeadAttention {
                seq_len: seq_sym, num_heads, num_kv_heads, head_dim, causal: false, attention_sinks: false,
            },
            vec![q, k, v], vec![out], "mha",
        );
        let mut map = HashMap::new();
        map.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: map,
        };
        let alloc = BufferAllocation::default();
        let layer = compile(&g, &plan, &alloc);

        // 准备 Q (seq_actual 行), K 和 V 拼接到 weight_blob。K/V 各占 seq_max 行。
        let q_data: Vec<f32> = (0..seq_actual * h).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut kv_blob = vec![0.0f32; 2 * seq_max * h];
        // K in [0..seq_max*h]
        for i in 0..seq_actual * h {
            kv_blob[i] = (i as f32 * 0.07).cos();
        }
        // V in [seq_max*h..2*seq_max*h]
        for i in 0..seq_actual * h {
            kv_blob[seq_max * h + i] = (i as f32 * 0.05).sin();
        }
        let mut out_jit = vec![0.0f32; seq_max * h]; // 分配 max 大小避免 JIT 越界
        // Q buffer 也要 max 大小避免越界
        let mut q_buf = vec![0.0f32; seq_max * h];
        q_buf[..seq_actual * h].copy_from_slice(&q_data);

        unsafe {
            let f = layer.entry_point();
            f(
                q_buf.as_ptr() as *const u8, kv_blob.as_ptr() as *const u8,
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, seq_actual, out_jit.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }

        // Scalar 参考 — 经典 attention: out[qi,h,d] = sum_ki softmax(QK/sqrt(d))[ki] * V[ki,h,d]
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out_ref = vec![0.0f32; seq_actual * h];
        for qi in 0..seq_actual {
            for hd in 0..num_heads {
                let q_row = &q_data[qi*h + hd*head_dim .. qi*h + (hd+1)*head_dim];
                // scores[ki] = Q · K[ki,h] * scale
                let mut scores = vec![0.0f32; seq_actual];
                for ki in 0..seq_actual {
                    let k_row = &kv_blob[ki*h + hd*head_dim .. ki*h + (hd+1)*head_dim];
                    let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a,b)| a*b).sum();
                    scores[ki] = dot * scale;
                }
                // softmax
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                let probs: Vec<f32> = scores.iter().map(|s| (s - max).exp() / exp_sum).collect();
                // out[qi,h,:] = sum_ki probs[ki] * V[ki,h,:]
                for d in 0..head_dim {
                    let mut s = 0.0f32;
                    for ki in 0..seq_actual {
                        let v = kv_blob[seq_max * h + ki*h + hd*head_dim + d];
                        s += probs[ki] * v;
                    }
                    out_ref[qi*h + hd*head_dim + d] = s;
                }
            }
        }

        eprintln!("out_jit[0..8]={:?}", &out_jit[..8]);
        eprintln!("out_ref[0..8]={:?}", &out_ref[..8]);
        eprintln!("out_jit seq[1][0..8]={:?}", &out_jit[h..h+8]);
        eprintln!("out_ref seq[1][0..8]={:?}", &out_ref[h..h+8]);

        for qi in 0..seq_actual {
            for d in 0..h {
                let idx = qi*h + d;
                let diff = (out_jit[idx] - out_ref[idx]).abs();
                assert!(diff < 1e-4,
                    "MHA[qi={qi},d={d}]: got={}, expected={}, diff={diff}",
                    out_jit[idx], out_ref[idx]);
            }
        }
    }

    /// 重现 E2E 场景：通过 InferenceCompiler::compile_graph 编译 2D Symbolic Add。
    /// 这是 gllm executor 调用的确切路径。
    #[test]
    fn test_compile_graph_binary_add_symbolic() {
        use crate::compiler::{InferenceCompiler, SymDim};

        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic {
            name: "seq_len".to_string(),
            max_value: Some(2048),
        };
        let hidden = 768;
        let dt = DType::F32;

        let input_0 = g.add_tensor("input_0", vec![seq_sym.clone(), SymDim::Concrete(hidden)], dt);
        let input_1 = g.add_tensor("input_1", vec![seq_sym.clone(), SymDim::Concrete(hidden)], dt);
        let output = g.add_tensor("output", vec![seq_sym, SymDim::Concrete(hidden)], dt);
        g.inputs = vec![input_0, input_1];
        g.outputs = vec![output];
        g.add_op(OpKind::Add, vec![input_0, input_1], vec![output], "add0");

        let mut compiler = InferenceCompiler::new();
        let result = compiler.compile_graph(&g);
        match &result {
            Ok(layer) => eprintln!("✅ compile_graph Add succeeded, code_size={}", layer.code_size()),
            Err(e) => eprintln!("❌ compile_graph Add failed: {}", e),
        }
        result.unwrap();
    }

    /// 验证 register-only 路径在 hd_vecs > 4 时数值正确 (task #6)。
    /// hd_vecs=8 (BERT-style head_dim=64) 和 hd_vecs=16 (Qwen3-style head_dim=128)
    /// 都必须严格匹配 scalar reference,且**重复调用 bit-exact 一致**
    /// (旧 memory-based 路径会因 mmap lazy commit 产生不确定性)。
    #[test]
    fn test_vm_e2e_mha_register_only_hd_vecs_8_and_16() {
        use crate::compiler::SymDim;
        for &(num_heads, head_dim) in &[(1usize, 64usize), (1usize, 128usize)] {
            let hd_vecs = head_dim / 8; // AVX2 lanes=8
            assert!(hd_vecs > 4, "test 目标是 hd_vecs > 4 路径");
            let (seq_actual, seq_max) = (4usize, 8usize);
            let num_kv_heads = num_heads;
            let h = num_heads * head_dim;
            let mut g = CompilerGraph::new();
            let seq_sym = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(seq_max) };
            let q = g.add_tensor("q", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
            let k = g.add_tensor("k", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
            let v = g.add_tensor("v", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
            let out = g.add_tensor("out", vec![seq_sym.clone(), SymDim::Concrete(h)], DType::F32);
            g.inputs = vec![q, k, v];
            g.outputs = vec![out];
            let op = g.add_op(
                OpKind::MultiHeadAttention {
                    seq_len: seq_sym, num_heads, num_kv_heads, head_dim, causal: false, attention_sinks: false,
                },
                vec![q, k, v], vec![out], "mha",
            );
            let mut map = HashMap::new();
            map.insert(op, 0);
            let plan = FusionPlan {
                groups: vec![FusionGroup {
                    id: 0, anchor: op, epilogue: vec![],
                    mode: FusionMode::Standalone, ops: vec![op],
                    multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                }],
                op_to_group: map,
            };
            let alloc = BufferAllocation::default();
            let layer = compile(&g, &plan, &alloc);

            // 准备 Q (seq_actual 行), K 和 V 拼接到 weight_blob。K/V 各占 seq_max 行。
            let q_data: Vec<f32> = (0..seq_actual * h).map(|i| (i as f32 * 0.013).sin()).collect();
            let mut kv_blob = vec![0.0f32; 2 * seq_max * h];
            for i in 0..seq_actual * h {
                kv_blob[i] = (i as f32 * 0.017).cos();
                kv_blob[seq_max * h + i] = (i as f32 * 0.011).sin();
            }
            let mut q_buf = vec![0.0f32; seq_max * h];
            q_buf[..seq_actual * h].copy_from_slice(&q_data);

            // 跑两次,验证 bit-exact determinism
            let mut out_jit_a = vec![0.0f32; seq_max * h];
            let mut out_jit_b = vec![0.0f32; seq_max * h];
            unsafe {
                let f = layer.entry_point();
                f(
                    q_buf.as_ptr() as *const u8, kv_blob.as_ptr() as *const u8,
                    std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                    1, seq_actual, out_jit_a.as_mut_ptr() as *mut u8,
                    std::ptr::null_mut(), std::ptr::null_mut(),
                );
                f(
                    q_buf.as_ptr() as *const u8, kv_blob.as_ptr() as *const u8,
                    std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                    1, seq_actual, out_jit_b.as_mut_ptr() as *mut u8,
                    std::ptr::null_mut(), std::ptr::null_mut(),
                );
            }

            // Determinism: 同输入两次执行必须 bit-exact 一致
            for i in 0..seq_actual * h {
                assert_eq!(out_jit_a[i].to_bits(), out_jit_b[i].to_bits(),
                    "MHA hd_vecs={} not deterministic at idx {}: {} vs {}",
                    hd_vecs, i, out_jit_a[i], out_jit_b[i]);
            }

            // Scalar 参考
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut out_ref = vec![0.0f32; seq_actual * h];
            for qi in 0..seq_actual {
                for hd in 0..num_heads {
                    let q_row = &q_data[qi*h + hd*head_dim .. qi*h + (hd+1)*head_dim];
                    let mut scores = vec![0.0f32; seq_actual];
                    for ki in 0..seq_actual {
                        let k_row = &kv_blob[ki*h + hd*head_dim .. ki*h + (hd+1)*head_dim];
                        let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a,b)| a*b).sum();
                        scores[ki] = dot * scale;
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                    let probs: Vec<f32> = scores.iter().map(|s| (s - max).exp() / exp_sum).collect();
                    for d in 0..head_dim {
                        let mut s = 0.0f32;
                        for ki in 0..seq_actual {
                            let v = kv_blob[seq_max * h + ki*h + hd*head_dim + d];
                            s += probs[ki] * v;
                        }
                        out_ref[qi*h + hd*head_dim + d] = s;
                    }
                }
            }

            for qi in 0..seq_actual {
                for d in 0..h {
                    let idx = qi*h + d;
                    let diff = (out_jit_a[idx] - out_ref[idx]).abs();
                    assert!(diff < 5e-3,
                        "MHA hd_vecs={hd_vecs} [qi={qi},d={d}]: got={}, expected={}, diff={diff}",
                        out_jit_a[idx], out_ref[idx]);
                }
            }
            eprintln!("[MHA-RegOnly] hd_vecs={hd_vecs} (head_dim={head_dim}) PASS");
        }
    }


    // ────────────────────────────────────────────────────────────────
    // AttentionSinks (OpenAI gpt-oss-20b) — 数值正确性测试
    // ────────────────────────────────────────────────────────────────

    /// Pure scalar CPU reference for MHA with per-head attention sinks.
    /// Layout: Q/K/V/Out are `[seq, num_heads*head_dim]` row-major.
    fn scalar_mha_with_sinks_ref(
        q: &[f32], k: &[f32], v: &[f32], sinks: &[f32],
        seq: usize, num_heads: usize, head_dim: usize,
    ) -> Vec<f32> {
        let hidden = num_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq * hidden];
        for h in 0..num_heads {
            let sink_h = sinks[h];
            for qi in 0..seq {
                // raw scores
                let mut scores = vec![0.0f32; seq];
                for ki in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qi*hidden + h*head_dim + d] * k[ki*hidden + h*head_dim + d];
                    }
                    scores[ki] = dot * scale;
                }
                // stable softmax with sink denominator
                let mut m = sink_h;
                for &s in &scores { if s > m { m = s; } }
                let mut l = (sink_h - m).exp();
                let mut w = vec![0.0f32; seq];
                for (ki, &s) in scores.iter().enumerate() {
                    w[ki] = (s - m).exp();
                    l += w[ki];
                }
                let inv_l = 1.0 / l;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..seq {
                        acc += w[ki] * v[ki*hidden + h*head_dim + d];
                    }
                    out[qi*hidden + h*head_dim + d] = acc * inv_l;
                }
            }
        }
        out
    }

    /// Standard MHA reference (no sinks) for sanity comparison.
    fn scalar_mha_ref(
        q: &[f32], k: &[f32], v: &[f32],
        seq: usize, num_heads: usize, head_dim: usize,
    ) -> Vec<f32> {
        let hidden = num_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq * hidden];
        for h in 0..num_heads {
            for qi in 0..seq {
                let mut scores = vec![0.0f32; seq];
                for ki in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qi*hidden + h*head_dim + d] * k[ki*hidden + h*head_dim + d];
                    }
                    scores[ki] = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let l: f32 = scores.iter().map(|s| (s - m).exp()).sum();
                let inv_l = 1.0 / l;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..seq {
                        acc += (scores[ki] - m).exp() * v[ki*hidden + h*head_dim + d];
                    }
                    out[qi*hidden + h*head_dim + d] = acc * inv_l;
                }
            }
        }
        out
    }

    /// attention_sinks=true 的 JIT 输出必须匹配 scalar reference (含 sink 的 softmax 分母)。
    /// 同时,当 sinks=very negative 时,输出应退化到 no-sink 参考;当 sinks=very positive 时,
    /// sink 吸收大部分概率质量,输出近似 0 (每个 V_j 的权重被推向零)。
    #[test]
    fn test_vm_e2e_mha_attention_sinks_numerical() {
        let seq: usize = 4;
        let num_heads: usize = 2;
        let num_kv_heads: usize = num_heads;
        let head_dim: usize = 8; // hd_vecs = 1 for AVX2 W256 (8 f32 lanes)
        let hidden = num_heads * head_dim;

        // 构图: inputs = [q, k, v, sinks]
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Concrete(seq);
        let q = g.add_tensor("q", vec![seq_sym.clone(), SymDim::Concrete(hidden)], DType::F32);
        let k = g.add_tensor("k", vec![seq_sym.clone(), SymDim::Concrete(hidden)], DType::F32);
        let v = g.add_tensor("v", vec![seq_sym.clone(), SymDim::Concrete(hidden)], DType::F32);
        let sinks_t = g.add_tensor("sinks", vec![SymDim::Concrete(num_heads)], DType::F32);
        let out_t = g.add_tensor("out", vec![seq_sym, SymDim::Concrete(hidden)], DType::F32);
        g.inputs = vec![q, k, v, sinks_t];
        g.outputs = vec![out_t];
        let op = g.add_op(
            OpKind::MultiHeadAttention {
                seq_len: SymDim::Concrete(seq),
                num_heads, num_kv_heads, head_dim,
                causal: false,
                attention_sinks: true,
            },
            vec![q, k, v, sinks_t],
            vec![out_t],
            "mha_sinks",
        );
        let mut map = HashMap::new();
        map.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: map,
        };
        let alloc = BufferAllocation::default();
        let layer = compile(&g, &plan, &alloc);

        // 输入数据
        let q_data: Vec<f32> = (0..seq*hidden).map(|i| (i as f32 * 0.019).sin()).collect();
        let k_data: Vec<f32> = (0..seq*hidden).map(|i| (i as f32 * 0.023).cos()).collect();
        let v_data: Vec<f32> = (0..seq*hidden).map(|i| (i as f32 * 0.031).sin()).collect();

        // Weight blob 按 g.inputs[1..] = [k, v, sinks] 顺序 pack
        let wl = g.weight_layout();
        let weight_bytes = wl.total_bytes;
        let mut weight_blob = vec![0u8; weight_bytes];

        // k 的 offset + 写入 data
        let k_off = wl.offsets.iter().find(|(tid, _)| *tid == k).unwrap().1;
        let v_off = wl.offsets.iter().find(|(tid, _)| *tid == v).unwrap().1;
        let s_off = wl.offsets.iter().find(|(tid, _)| *tid == sinks_t).unwrap().1;
        unsafe {
            std::ptr::copy_nonoverlapping(
                k_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(k_off),
                seq * hidden * 4);
            std::ptr::copy_nonoverlapping(
                v_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(v_off),
                seq * hidden * 4);
        }

        // Case 1: "normal" sinks (不太大不太小,有实际分担效应)
        let sinks_normal: Vec<f32> = (0..num_heads).map(|h| -0.5 + 0.25 * h as f32).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                sinks_normal.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(s_off),
                num_heads * 4);
        }
        let mut out_jit = vec![0.0f32; seq * hidden];
        unsafe {
            let f = layer.entry_point();
            f(
                q_data.as_ptr() as *const u8,
                weight_blob.as_ptr(),
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, seq, out_jit.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
        let out_ref = scalar_mha_with_sinks_ref(&q_data, &k_data, &v_data, &sinks_normal,
            seq, num_heads, head_dim);
        for i in 0..seq*hidden {
            let diff = (out_jit[i] - out_ref[i]).abs();
            assert!(diff < 5e-4,
                "sinks(normal): idx={} jit={} ref={} diff={}", i, out_jit[i], out_ref[i], diff);
        }

        // Case 2: extremely negative sinks ⇒ exp(sink - m) → 0,退化为 no-sink softmax。
        let sinks_neg: Vec<f32> = vec![-50.0; num_heads];
        unsafe {
            std::ptr::copy_nonoverlapping(
                sinks_neg.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(s_off),
                num_heads * 4);
        }
        let mut out_jit_neg = vec![0.0f32; seq * hidden];
        unsafe {
            let f = layer.entry_point();
            f(
                q_data.as_ptr() as *const u8,
                weight_blob.as_ptr(),
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, seq, out_jit_neg.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
        let out_nosink = scalar_mha_ref(&q_data, &k_data, &v_data, seq, num_heads, head_dim);
        for i in 0..seq*hidden {
            let diff = (out_jit_neg[i] - out_nosink[i]).abs();
            // exp(-50) ≈ 1.9e-22 不是完全零,但数量级远小于典型 exp(s-m) ∈ [1e-3, 1]。
            // JIT 走 register FMA 路径,IEEE 754 rounding 导致与 scalar 序列化求和相比
            // 存在 ~1e-4 级别的 relative error,5e-4 absolute tolerance 完全覆盖。
            assert!(diff < 5e-4,
                "sinks(-50): idx={} should ≈ no-sink; jit={} nosink={} diff={}",
                i, out_jit_neg[i], out_nosink[i], diff);
        }

        // Case 3: extremely positive sinks ⇒ sink 吸收所有概率质量,输出应 ≈ 0。
        let sinks_pos: Vec<f32> = vec![50.0; num_heads];
        unsafe {
            std::ptr::copy_nonoverlapping(
                sinks_pos.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(s_off),
                num_heads * 4);
        }
        let mut out_jit_pos = vec![0.0f32; seq * hidden];
        unsafe {
            let f = layer.entry_point();
            f(
                q_data.as_ptr() as *const u8,
                weight_blob.as_ptr(),
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, seq, out_jit_pos.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
        for i in 0..seq*hidden {
            assert!(out_jit_pos[i].abs() < 1e-5,
                "sinks(+50): idx={} should ≈ 0 (sink absorbs mass); got {}", i, out_jit_pos[i]);
        }

        // Case 4: causal 路径一致性
        let mut g2 = CompilerGraph::new();
        let q2 = g2.add_tensor("q", vec![SymDim::Concrete(seq), SymDim::Concrete(hidden)], DType::F32);
        let k2 = g2.add_tensor("k", vec![SymDim::Concrete(seq), SymDim::Concrete(hidden)], DType::F32);
        let v2 = g2.add_tensor("v", vec![SymDim::Concrete(seq), SymDim::Concrete(hidden)], DType::F32);
        let s2 = g2.add_tensor("sinks", vec![SymDim::Concrete(num_heads)], DType::F32);
        let out2 = g2.add_tensor("out", vec![SymDim::Concrete(seq), SymDim::Concrete(hidden)], DType::F32);
        g2.inputs = vec![q2, k2, v2, s2];
        g2.outputs = vec![out2];
        let op2 = g2.add_op(
            OpKind::MultiHeadAttention {
                seq_len: SymDim::Concrete(seq),
                num_heads, num_kv_heads, head_dim,
                causal: true,
                attention_sinks: true,
            },
            vec![q2, k2, v2, s2],
            vec![out2],
            "mha_sinks_causal",
        );
        let mut map2 = HashMap::new();
        map2.insert(op2, 0);
        let plan2 = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op2, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op2],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: map2,
        };
        let alloc2 = BufferAllocation::default();
        let layer2 = compile(&g2, &plan2, &alloc2);

        let wl2 = g2.weight_layout();
        let mut weight_blob2 = vec![0u8; wl2.total_bytes];
        let k_off2 = wl2.offsets.iter().find(|(tid, _)| *tid == k2).unwrap().1;
        let v_off2 = wl2.offsets.iter().find(|(tid, _)| *tid == v2).unwrap().1;
        let s_off2 = wl2.offsets.iter().find(|(tid, _)| *tid == s2).unwrap().1;
        unsafe {
            std::ptr::copy_nonoverlapping(
                k_data.as_ptr() as *const u8,
                weight_blob2.as_mut_ptr().add(k_off2),
                seq * hidden * 4);
            std::ptr::copy_nonoverlapping(
                v_data.as_ptr() as *const u8,
                weight_blob2.as_mut_ptr().add(v_off2),
                seq * hidden * 4);
            std::ptr::copy_nonoverlapping(
                sinks_normal.as_ptr() as *const u8,
                weight_blob2.as_mut_ptr().add(s_off2),
                num_heads * 4);
        }

        let mut out_jit_causal = vec![0.0f32; seq * hidden];
        unsafe {
            let f = layer2.entry_point();
            f(
                q_data.as_ptr() as *const u8,
                weight_blob2.as_ptr(),
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                1, seq, out_jit_causal.as_mut_ptr() as *mut u8,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
        // Causal reference: 每个 qi 只 attend 到 ki ≤ qi 的位置
        let mut out_causal_ref = vec![0.0f32; seq * hidden];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..num_heads {
            let sink_h = sinks_normal[h];
            for qi in 0..seq {
                let ki_end = qi + 1;
                let mut scores = vec![0.0f32; ki_end];
                for ki in 0..ki_end {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[qi*hidden + h*head_dim + d] * k_data[ki*hidden + h*head_dim + d];
                    }
                    scores[ki] = dot * scale;
                }
                let mut m = sink_h;
                for &s in &scores { if s > m { m = s; } }
                let mut l = (sink_h - m).exp();
                let mut w = vec![0.0f32; ki_end];
                for (ki, &s) in scores.iter().enumerate() {
                    w[ki] = (s - m).exp();
                    l += w[ki];
                }
                let inv_l = 1.0 / l;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..ki_end {
                        acc += w[ki] * v_data[ki*hidden + h*head_dim + d];
                    }
                    out_causal_ref[qi*hidden + h*head_dim + d] = acc * inv_l;
                }
            }
        }
        for i in 0..seq*hidden {
            let diff = (out_jit_causal[i] - out_causal_ref[i]).abs();
            assert!(diff < 5e-4,
                "causal sinks: idx={} jit={} ref={} diff={}", i, out_jit_causal[i], out_causal_ref[i], diff);
        }
        eprintln!("[MHA-Sinks] PASS: normal + extreme_neg + extreme_pos + causal");
    }
    fn test_head_rms_norm_numerical() {
        let head_dim = 2;
        let num_heads = 2;
        let hidden = head_dim * num_heads;
        let seq = 1;

        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[seq, hidden], DType::F32);
        let wt = g.add_tensor_concrete("weight", &[head_dim], DType::F32);
        let out = g.add_tensor_concrete("output", &[seq, hidden], DType::F32);
        g.inputs = vec![inp, wt];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::HeadRmsNorm { head_dim, eps: 1e-6 },
            vec![inp, wt],
            vec![out],
            "head_rms_norm",
        );
        let mut m = HashMap::new();
        m.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group: m,
        };
        let alloc = BufferAllocation::default();

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("HeadRmsNorm compile");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("from_code");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weight_data: Vec<f32> = vec![1.0, 1.0];
        let mut out_data = vec![0.0f32; hidden];
        let mut telemetry = vec![0u8; crate::compiler::graph::telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES];
        let mut scratch = vec![0u8; output.scratchpad_bytes.max(1024)];

        unsafe {
            let f = layer.entry_point();
            f(
                input_data.as_ptr() as *const u8,
                weight_data.as_ptr() as *const u8,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                seq,
                out_data.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
                telemetry.as_mut_ptr() as *mut u8,
            );
        }

        eprintln!("[HEAD-RMS] output: {:?}", &out_data);

        let expected: [f32; 4] = [
            1.0 / (2.5_f32 + 1e-6).sqrt(),
            2.0 / (2.5_f32 + 1e-6).sqrt(),
            3.0 / (12.5_f32 + 1e-6).sqrt(),
            4.0 / (12.5_f32 + 1e-6).sqrt(),
        ];
        // tolerance 1e-3:VecUnaryOp::Rsqrt 在 x86_64 用 vrsqrtps approximation
        // (~12 bit 精度),非精确 1/sqrt。误差 ~1-3e-4 为正常范围。
        for i in 0..hidden {
            assert!(
                (out_data[i] - expected[i]).abs() < 1e-3,
                "out[{i}] = {}, expected {} (diff {})",
                out_data[i], expected[i], (out_data[i] - expected[i]).abs(),
            );
        }
    }
}

