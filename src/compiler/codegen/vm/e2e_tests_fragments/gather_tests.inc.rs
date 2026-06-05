#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod gather_tests {
    use crate::compiler::{CompilerGraph, InferenceCompiler, SymDim};
    use crate::compiler::graph::{OpKind, MultiOutputConfig};
    use crate::compiler::codegen::vm::plan_lower::compile_layer;
    use crate::compiler::executable::CompiledLayer;
    use crate::compiler::fusion::{FusionPlan, FusionGroup, FusionMode};
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::dispatch::DeviceProfile;
    use crate::types::DType;
    #[allow(unused_imports)]
    use std::collections::HashMap;

    #[test]
    fn test_compile_graph_gather() {
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic {
            name: "seq_len".to_string(),
            max_value: Some(2048),
        };
        let embed_dim = 768;
        let table_rows = 50265;
        let dt = DType::F32;

        let indices = g.add_tensor("indices", vec![seq_sym.clone()], dt);
        let table = g.add_tensor_concrete("embed_table", &[table_rows, embed_dim], dt);
        let output = g.add_tensor("embed_out", vec![seq_sym.clone(), SymDim::Concrete(embed_dim)], dt);
        g.inputs = vec![indices, table];
        g.outputs = vec![output];
        g.add_op(
            OpKind::Gather { table_rows, embed_dim, index_dim: seq_sym, indices_kind: Default::default() },
            vec![indices, table], vec![output], "gather",
        );

        let mut compiler = InferenceCompiler::new();
        let result = compiler.compile_graph(&g);
        match &result {
            Ok(layer) => eprintln!("✅ Gather compilation OK, code_size={}", layer.code_size()),
            Err(e) => eprintln!("❌ Gather failed: {}", e),
        }
        result.unwrap();
    }

    /// 重现完整 embedding 流程: Gather → Gather → Add
    #[test]
    fn test_compile_graph_gather_then_add() {
        // 单独编译两个 Gather + 一个 Add (模拟 xlmr embedding 流程)
        let mut compiler = InferenceCompiler::new();
        let seq_sym = || SymDim::Symbolic {
            name: "seq_len".to_string(),
            max_value: Some(2048),
        };
        let hidden = 768;
        let dt = DType::F32;

        // 1. Gather: embed_tokens
        {
            let mut g = CompilerGraph::new();
            let indices = g.add_tensor("indices", vec![seq_sym()], dt);
            let table = g.add_tensor_concrete("embed_table", &[50265, hidden], dt);
            let output = g.add_tensor("embed_out", vec![seq_sym(), SymDim::Concrete(hidden)], dt);
            g.inputs = vec![indices, table];
            g.outputs = vec![output];
            g.add_op(
                OpKind::Gather { table_rows: 50265, embed_dim: hidden, index_dim: seq_sym(), indices_kind: Default::default() },
                vec![indices, table], vec![output], "gather_tokens",
            );
            compiler.compile_graph(&g).expect("Gather tokens");
        }

        // 2. Gather: embed_pos
        {
            let mut g = CompilerGraph::new();
            let indices = g.add_tensor("indices", vec![seq_sym()], dt);
            let table = g.add_tensor_concrete("embed_table", &[514, hidden], dt);
            let output = g.add_tensor("embed_out", vec![seq_sym(), SymDim::Concrete(hidden)], dt);
            g.inputs = vec![indices, table];
            g.outputs = vec![output];
            g.add_op(
                OpKind::Gather { table_rows: 514, embed_dim: hidden, index_dim: seq_sym(), indices_kind: Default::default() },
                vec![indices, table], vec![output], "gather_pos",
            );
            compiler.compile_graph(&g).expect("Gather pos");
        }

        // 3. Add: embed_tok + embed_pos
        {
            let mut g = CompilerGraph::new();
            let a = g.add_tensor("input_0", vec![seq_sym(), SymDim::Concrete(hidden)], dt);
            let b = g.add_tensor("input_1", vec![seq_sym(), SymDim::Concrete(hidden)], dt);
            let output = g.add_tensor("output", vec![seq_sym(), SymDim::Concrete(hidden)], dt);
            g.inputs = vec![a, b];
            g.outputs = vec![output];
            g.add_op(OpKind::Add, vec![a, b], vec![output], "embed_add_pos");
            compiler.compile_graph(&g).expect("Add embed_add_pos");
        }

        eprintln!("✅ Full Gather→Gather→Add pipeline compiled successfully");
    }

    // ═══════════════════════════════════════════════════════════════
    //  AltUp (formerly PerLayerEmbed) — REMOVED
    //  PerLayerEmbed has been replaced by AltUpPredict/AltUpCorrect/AltUpInject
    //  (Injective ops). AltUp E2E tests will be added when AltUp scalar
    //  reference implementations and graph builders are available.
    // ═══════════════════════════════════════════════════════════════

    // ══════════════════════════════════════════════════════════════════
    //  T37: ColumnSlice (PleSlice runtime 列切片真实实现)
    // ══════════════════════════════════════════════════════════════════

    /// Build ColumnSlice plan as single standalone op.
    /// Input [seq, num_layers * dim] → Output [seq, dim], start = layer_idx * dim。
    fn build_column_slice_plan(
        seq: usize, num_layers: usize, dim: usize, layer_idx: usize,
    ) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input_inner = num_layers * dim;
        let seq_dim = SymDim::Concrete(seq);
        let inp = g.add_tensor("ple_full", vec![seq_dim.clone(), SymDim::Concrete(input_inner)], dt);
        let out = g.add_tensor("ple_slice", vec![seq_dim.clone(), SymDim::Concrete(dim)], dt);
        g.inputs = vec![inp];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::ColumnSlice {
                seq_len: seq_dim,
                input_inner,
                start: layer_idx * dim,
                slice_dim: dim,
            },
            vec![inp], vec![out], "ple_col_slice",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// Execute ColumnSlice layer via CompiledLayer ABI.
    unsafe fn exec_column_slice(
        layer: &CompiledLayer,
        input: &[f32], output: &mut [f32], seq: usize,
    ) {
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,        // rdi: input (ple_full)
            std::ptr::null(),                    // rsi: weights (unused)
            std::ptr::null_mut(),                // rdx: kv_cache
            std::ptr::null(),                    // rcx: positions
            std::ptr::null(),                    // r8: seq_lens
            1,                                    // r9: batch_size
            seq,                                  // [rsp+0]: seq_len
            output.as_mut_ptr() as *mut u8,      // [rsp+8] → output
            std::ptr::null_mut(),                // [rsp+16]: scratchpad
            std::ptr::null_mut(),                // [rsp+24]: telemetry
        );
    }

    /// T37 核心数值正确性验证: 每个 layer_idx 的切片都必须精确等于
    /// `input[s, layer_idx * dim + j]`, 而不是共享同一起点。
    #[test]
    fn test_column_slice_per_layer_correctness() {
        let (seq, num_layers, dim) = (4usize, 6usize, 4usize);
        let input_inner = num_layers * dim;

        // 确定性输入: 每行前 dim 元素标记 layer 0, 中间 dim 元素标记 layer 1, ...
        // ple_full[s, layer_idx * dim + j] = (s + 1) * 100 + layer_idx * 10 + j
        let input: Vec<f32> = (0..seq)
            .flat_map(|s| {
                (0..input_inner).map(move |col| {
                    let layer_idx = col / dim;
                    let j = col % dim;
                    ((s + 1) * 100 + layer_idx * 10 + j) as f32
                })
            })
            .collect();

        for layer_idx in 0..num_layers {
            let (g, plan, alloc) = build_column_slice_plan(seq, num_layers, dim, layer_idx);
            let profile = DeviceProfile::detect();
            let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
            let registry = ScalarOpRegistry::with_defaults();
            let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
                .expect("compile ColumnSlice");
            assert!(!output.code.is_empty(), "ColumnSlice codegen produced empty machine code");
            let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
                .expect("CompiledLayer::from_code ColumnSlice");

            let mut out = vec![0.0_f32; seq * dim];
            unsafe { exec_column_slice(&layer, &input, &mut out, seq) };

            // 逐行逐列验证 output[s, j] == input[s, layer_idx * dim + j]
            for s in 0..seq {
                for j in 0..dim {
                    let expected = ((s + 1) * 100 + layer_idx * 10 + j) as f32;
                    let actual = out[s * dim + j];
                    assert_eq!(
                        actual, expected,
                        "layer_idx={layer_idx} s={s} j={j}: expected {expected}, got {actual} \
                         (input row[s={s}] = {:?})",
                        &input[s * input_inner..(s + 1) * input_inner]
                    );
                }
            }
        }
    }

    /// T37 tail 路径: slice_dim 不是 SIMD lanes 的整数倍 (e.g. dim=7 on AVX2 lanes=8)。
    #[test]
    fn test_column_slice_tail_path() {
        // 选择 dim < AVX2 lanes(8) 以强制走 pure tail 路径
        let (seq, num_layers, dim) = (3usize, 3usize, 5usize);
        let layer_idx = 1usize;
        let input_inner = num_layers * dim;

        // 输入: input[s * input_inner + col] = s * 1000 + col
        let input: Vec<f32> = (0..seq * input_inner).map(|i| {
            let s = i / input_inner;
            let col = i % input_inner;
            (s * 1000 + col) as f32
        }).collect();

        let (g, plan, alloc) = build_column_slice_plan(seq, num_layers, dim, layer_idx);
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile ColumnSlice tail");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer::from_code ColumnSlice tail");

        let mut out = vec![0.0_f32; seq * dim];
        unsafe { exec_column_slice(&layer, &input, &mut out, seq) };

        for s in 0..seq {
            for j in 0..dim {
                let expected = (s * 1000 + layer_idx * dim + j) as f32;
                let actual = out[s * dim + j];
                assert_eq!(actual, expected,
                    "tail path: layer_idx={layer_idx} s={s} j={j}: expected {expected}, got {actual}");
            }
        }
    }

    /// T37 Symbolic seq_len 路径: 运行时传入 seq_actual, 验证 JIT 从 [rbp+16] 读取。
    #[test]
    fn test_column_slice_symbolic_seq() {
        let (seq_max, num_layers, dim, seq_actual) = (16usize, 4usize, 8usize, 5usize);
        let layer_idx = 2usize;
        let input_inner = num_layers * dim;

        // 构造 Symbolic 版本的 plan
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let seq_sym = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(seq_max) };
        let inp = g.add_tensor("ple_full", vec![seq_sym.clone(), SymDim::Concrete(input_inner)], dt);
        let out = g.add_tensor("ple_slice", vec![seq_sym.clone(), SymDim::Concrete(dim)], dt);
        g.inputs = vec![inp];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::ColumnSlice {
                seq_len: seq_sym,
                input_inner,
                start: layer_idx * dim,
                slice_dim: dim,
            },
            vec![inp], vec![out], "ple_col_slice_sym",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();

        // 输入按 seq_max 分配 (weight/input buffer 必须按 max 上界准备)
        let input: Vec<f32> = (0..seq_max * input_inner).map(|i| i as f32).collect();
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile ColumnSlice symbolic");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer::from_code ColumnSlice symbolic");

        let mut out = vec![0.0_f32; seq_max * dim];
        unsafe { exec_column_slice(&layer, &input, &mut out, seq_actual) };

        // 仅校验前 seq_actual 行
        for s in 0..seq_actual {
            for j in 0..dim {
                let expected = (s * input_inner + layer_idx * dim + j) as f32;
                let actual = out[s * dim + j];
                assert_eq!(actual, expected,
                    "symbolic seq: s={s} j={j}: expected {expected}, got {actual}");
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  T55: LearnedPos2D (SigLIP / ViT learned 2D positional embedding)
    // ══════════════════════════════════════════════════════════════════

    /// 构造 LearnedPos2D plan: 两输入 pure elementwise add。
    fn build_learned_pos_2d_plan(
        num_patches: usize, embed_dim: usize,
    ) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let patches = g.add_tensor_concrete("patches", &[num_patches, embed_dim], dt);
        let pos_table = g.add_tensor_concrete("pos_table", &[num_patches, embed_dim], dt);
        let out = g.add_tensor_concrete("out", &[num_patches, embed_dim], dt);
        g.inputs = vec![patches, pos_table];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::LearnedPos2D { num_patches, embed_dim },
            vec![patches, pos_table], vec![out], "learned_pos_2d",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// Execute binary elementwise layer (pattern 与 tests::exec_binary 等价,
    /// 但 gather_tests 模块内独立定义避免跨模块依赖)。
    #[allow(dead_code)]
    unsafe fn exec_binary_direct(
        layer: &CompiledLayer, a: &[f32], b: &[f32], out: &mut [f32], seq: usize,
    ) {
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8, b.as_ptr() as *const u8,
            std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 1,
            seq,
            out.as_mut_ptr() as *mut u8,
            std::ptr::null_mut(), std::ptr::null_mut(),
        );
    }

    /// T55 LearnedPos2D 数值对齐: pure elementwise add,验证全部输出与标量参考一致。
    #[test]
    fn test_vm_e2e_learned_pos_2d_basic() {
        let (num_patches, embed_dim) = (4usize, 16usize); // 16 = 2 AVX2 vectors, 无 tail
        let (g, plan, alloc) = build_learned_pos_2d_plan(num_patches, embed_dim);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile LearnedPos2D basic");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from LearnedPos2D basic");

        let patches: Vec<f32> = (0..num_patches * embed_dim).map(|i| (i as f32 * 0.013).sin()).collect();
        let pos_table: Vec<f32> = (0..num_patches * embed_dim).map(|i| (i as f32 * 0.019 + 0.3).cos()).collect();
        let mut out = vec![0.0_f32; num_patches * embed_dim];

        unsafe { exec_binary_direct(&layer, &patches, &pos_table, &mut out, num_patches) };

        let mut max_diff = 0.0_f32;
        for i in 0..num_patches * embed_dim {
            let expected = patches[i] + pos_table[i];
            let diff = (out[i] - expected).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("LearnedPos2D basic: max_abs_diff={max_diff:.2e} (num_patches={num_patches}, embed_dim={embed_dim})");
        assert!(max_diff < 1e-5, "LearnedPos2D basic: max_diff={max_diff}");
    }

    /// T55 LearnedPos2D tail path: embed_dim=7 触发 tail (AVX2 lanes=8)。
    #[test]
    fn test_vm_e2e_learned_pos_2d_tail() {
        let (num_patches, embed_dim) = (3usize, 7usize);
        let (g, plan, alloc) = build_learned_pos_2d_plan(num_patches, embed_dim);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile LearnedPos2D tail");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from LearnedPos2D tail");

        let patches: Vec<f32> = (0..num_patches * embed_dim).map(|i| (i as f32 * 0.07).sin()).collect();
        let pos_table: Vec<f32> = (0..num_patches * embed_dim).map(|i| (i as f32 * 0.09 - 0.5).cos()).collect();
        let mut out = vec![0.0_f32; num_patches * embed_dim];

        unsafe { exec_binary_direct(&layer, &patches, &pos_table, &mut out, num_patches) };

        let mut max_diff = 0.0_f32;
        for i in 0..num_patches * embed_dim {
            let expected = patches[i] + pos_table[i];
            let diff = (out[i] - expected).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("LearnedPos2D tail: max_abs_diff={max_diff:.2e}");
        assert!(max_diff < 1e-5, "LearnedPos2D tail: max_diff={max_diff}");
    }

    // ══════════════════════════════════════════════════════════════════
    //  T55: DepthwiseConv1D (USM Conformer convolution module)
    // ══════════════════════════════════════════════════════════════════

    /// Scalar reference (独立实现,语义与 scalar-ops/src/depthwise_conv1d.rs 一致)。
    fn scalar_depthwise_conv1d_reference(
        x: &[f32], weight: &[f32], seq_len: usize, channels: usize,
        kernel_size: usize, causal: bool,
    ) -> Vec<f32> {
        let left_pad: isize = if causal {
            kernel_size as isize - 1
        } else {
            (kernel_size as isize - 1) / 2
        };
        let mut out = vec![0.0_f32; seq_len * channels];
        for t in 0..seq_len {
            for c in 0..channels {
                let mut acc = 0.0_f32;
                for k in 0..kernel_size {
                    let t_in_signed = t as isize + k as isize - left_pad;
                    if t_in_signed < 0 || t_in_signed >= seq_len as isize {
                        continue;
                    }
                    let t_in = t_in_signed as usize;
                    acc += x[t_in * channels + c] * weight[c * kernel_size + k];
                }
                out[t * channels + c] = acc;
            }
        }
        out
    }

    /// 构造 DepthwiseConv1D plan — seq_len Concrete, 输入 `[seq, channels]`, weight `[channels, K]`。
    fn build_dwc_plan(
        seq: usize, channels: usize, kernel_size: usize, causal: bool,
    ) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let x = g.add_tensor_concrete("x", &[seq, channels], dt);
        let w = g.add_tensor_concrete("w", &[channels, kernel_size], dt);
        let out = g.add_tensor_concrete("out", &[seq, channels], dt);
        g.inputs = vec![x, w];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::DepthwiseConv1D { channels, kernel_size, causal },
            vec![x, w], vec![out], "dwc",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// Execute DepthwiseConv1D layer via CompiledLayer ABI (需要 scratchpad)。
    unsafe fn exec_dwc(
        layer: &CompiledLayer,
        x: &[f32], weight: &[f32], out: &mut [f32], scratch: &mut [u8],
        seq: usize,
    ) {
        let f = layer.entry_point();
        f(
            x.as_ptr() as *const u8,
            weight.as_ptr() as *const u8,
            std::ptr::null_mut(),
            std::ptr::null(), std::ptr::null(), 1,
            seq,
            out.as_mut_ptr() as *mut u8,
            scratch.as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }

    /// T55 DepthwiseConv1D causal, kernel_size=3, 验证小规模手算场景 (与 scalar-ops 测试同值)。
    #[test]
    fn test_vm_e2e_dwc_causal_small() {
        let (seq, channels, k) = (8usize, 2usize, 3usize);
        let (g, plan, alloc) = build_dwc_plan(seq, channels, k, true);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile DWC causal small");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from DWC causal small");

        // x[t, c] = t + 1 (两通道同值) — 跟 scalar-ops 测试一致
        let x: Vec<f32> = (0..seq).flat_map(|t| (0..channels).map(move |_| (t + 1) as f32)).collect();
        // w[c, k] = c*10 + k + 1
        let weight: Vec<f32> = (0..channels)
            .flat_map(|c| (0..k).map(move |kk| (c * 10 + kk + 1) as f32))
            .collect();
        let mut out = vec![0.0_f32; seq * channels];
        let mut scratch = vec![0u8; output.scratchpad_bytes];
        unsafe { exec_dwc(&layer, &x, &weight, &mut out, &mut scratch, seq) };

        let expected = scalar_depthwise_conv1d_reference(&x, &weight, seq, channels, k, true);
        let mut max_diff = 0.0_f32;
        for i in 0..seq * channels {
            let diff = (out[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("DWC causal small: max_abs_diff={max_diff:.2e} scratch={}B", output.scratchpad_bytes);
        assert!(max_diff < 1e-4,
            "DWC causal small: max_diff={max_diff}\nout[0..8]={:?}\nexp[0..8]={:?}",
            &out[..8.min(out.len())], &expected[..8.min(expected.len())]);
    }

    /// T55 DepthwiseConv1D non-causal, 1 channel (隔离 c-loop unroll 影响)。
    #[test]
    fn test_vm_e2e_dwc_noncausal_single_channel() {
        let (seq, channels, k) = (6usize, 1usize, 5usize);
        let (g, plan, alloc) = build_dwc_plan(seq, channels, k, false);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile DWC noncausal 1ch");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from DWC noncausal 1ch");

        let x: Vec<f32> = (0..seq * channels).map(|i| ((i as f32 + 1.0) * 0.11).sin()).collect();
        let weight: Vec<f32> = (0..channels * k).map(|i| ((i as f32 + 2.0) * 0.13).cos() * 0.4).collect();
        let mut out = vec![0.0_f32; seq * channels];
        let mut scratch = vec![0u8; output.scratchpad_bytes];
        unsafe { exec_dwc(&layer, &x, &weight, &mut out, &mut scratch, seq) };

        let expected = scalar_depthwise_conv1d_reference(&x, &weight, seq, channels, k, false);
        let mut max_diff = 0.0_f32;
        for i in 0..seq * channels {
            let diff = (out[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("DWC noncausal 1ch: out={:?}", out);
        eprintln!("DWC noncausal 1ch: exp={:?}", expected);
        assert!(max_diff < 1e-4, "DWC noncausal 1ch: max_diff={max_diff}");
    }

    /// T55 DepthwiseConv1D non-causal, 对称 SAME pad, odd kernel, 多通道数值对齐。
    #[test]
    fn test_vm_e2e_dwc_noncausal_same() {
        let (seq, channels, k) = (6usize, 4usize, 5usize);
        let (g, plan, alloc) = build_dwc_plan(seq, channels, k, false);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile DWC noncausal");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from DWC noncausal");

        let x: Vec<f32> = (0..seq * channels).map(|i| ((i as f32 + 1.0) * 0.11).sin()).collect();
        let weight: Vec<f32> = (0..channels * k).map(|i| ((i as f32 + 2.0) * 0.13).cos() * 0.4).collect();
        let mut out = vec![0.0_f32; seq * channels];
        let mut scratch = vec![0u8; output.scratchpad_bytes];
        unsafe { exec_dwc(&layer, &x, &weight, &mut out, &mut scratch, seq) };

        let expected = scalar_depthwise_conv1d_reference(&x, &weight, seq, channels, k, false);
        let mut max_diff = 0.0_f32;
        for i in 0..seq * channels {
            let diff = (out[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("DWC noncausal: max_abs_diff={max_diff:.2e} scratch={}B", output.scratchpad_bytes);
        assert!(max_diff < 1e-4, "DWC noncausal: max_diff={max_diff}");
    }

    /// T55 DepthwiseConv1D Symbolic seq_len 路径: 编译时绑定 max_value=32,运行时 seq_actual=5。
    #[test]
    fn test_vm_e2e_dwc_symbolic_seq() {
        use crate::compiler::SymDim;
        let (channels, k) = (8usize, 3usize);
        let seq_max = 32usize;
        let seq_actual = 5usize;
        let causal = true;

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let seq_sym = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(seq_max) };
        let x = g.add_tensor("x", vec![seq_sym.clone(), SymDim::Concrete(channels)], dt);
        let w = g.add_tensor_concrete("w", &[channels, k], dt);
        let out_t = g.add_tensor("out", vec![seq_sym.clone(), SymDim::Concrete(channels)], dt);
        g.inputs = vec![x, w];
        g.outputs = vec![out_t];
        let op = g.add_op(
            OpKind::DepthwiseConv1D { channels, kernel_size: k, causal },
            vec![x, w], vec![out_t], "dwc_sym",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile DWC symbolic");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from DWC symbolic");

        // Input padded by max_value 以避免 OOB read
        let mut x_buf = vec![0.0_f32; seq_max * channels];
        for i in 0..seq_actual * channels {
            x_buf[i] = ((i as f32 + 1.0) * 0.017).sin();
        }
        let weight: Vec<f32> = (0..channels * k).map(|i| ((i as f32 + 3.0) * 0.019).cos() * 0.3).collect();
        let mut out = vec![0.0_f32; seq_max * channels];
        let mut scratch = vec![0u8; output.scratchpad_bytes];
        unsafe { exec_dwc(&layer, &x_buf, &weight, &mut out, &mut scratch, seq_actual) };

        // 仅校验前 seq_actual 行
        let x_actual = &x_buf[..seq_actual * channels];
        let expected = scalar_depthwise_conv1d_reference(x_actual, &weight, seq_actual, channels, k, causal);
        let mut max_diff = 0.0_f32;
        for i in 0..seq_actual * channels {
            let diff = (out[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("DWC symbolic: max_abs_diff={max_diff:.2e} (seq_actual={seq_actual}, seq_max={seq_max}) scratch={}B",
                  output.scratchpad_bytes);
        assert!(max_diff < 1e-4,
            "DWC symbolic: max_diff={max_diff}\nout[0..8]={:?}\nexp[0..8]={:?}",
            &out[..8.min(out.len())], &expected[..8.min(expected.len())]);
    }

    // ══════════════════════════════════════════════════════════════════
    //  T65: PatchEmbed (SigLIP / ViT vision tower)
    // ══════════════════════════════════════════════════════════════════

    /// Scalar reference (独立实现,语义与 scalar-ops/src/patch_embed.rs 一致).
    /// 不调用 scalar_ops (NO_SCALAR 铁律) — 复制独立实现做 ground truth。
    fn scalar_patch_embed_reference(
        image: &[f32], kernel: &[f32],
        patch_size: usize, embed_dim: usize, in_channels: usize, image_size: usize,
    ) -> Vec<f32> {
        let num_patches_side = image_size / patch_size;
        let num_patches = num_patches_side * num_patches_side;
        let mut patches = vec![0.0_f32; num_patches * embed_dim];
        let image_plane = image_size * image_size;
        let kernel_plane = patch_size * patch_size;
        for p_row in 0..num_patches_side {
            for p_col in 0..num_patches_side {
                let p = p_row * num_patches_side + p_col;
                for e in 0..embed_dim {
                    let mut acc = 0.0_f32;
                    for c in 0..in_channels {
                        for kr in 0..patch_size {
                            for kc in 0..patch_size {
                                let img_row = p_row * patch_size + kr;
                                let img_col = p_col * patch_size + kc;
                                let img_idx = c * image_plane + img_row * image_size + img_col;
                                let ker_idx = e * in_channels * kernel_plane
                                    + c * kernel_plane + kr * patch_size + kc;
                                acc += image[img_idx] * kernel[ker_idx];
                            }
                        }
                    }
                    patches[p * embed_dim + e] = acc;
                }
            }
        }
        patches
    }

    /// 构造 PatchEmbed plan — image_size/patch_size/embed_dim/in_channels 全部 Concrete。
    fn build_patch_embed_plan(
        patch_size: usize, embed_dim: usize, in_channels: usize, image_size: usize,
    ) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let num_patches_side = image_size / patch_size;
        let num_patches = num_patches_side * num_patches_side;
        let image = g.add_tensor_concrete("image", &[in_channels, image_size, image_size], dt);
        let kernel = g.add_tensor_concrete("kernel",
            &[embed_dim, in_channels, patch_size, patch_size], dt);
        let patches = g.add_tensor_concrete("patches", &[num_patches, embed_dim], dt);
        g.inputs = vec![image, kernel];
        g.outputs = vec![patches];
        let op = g.add_op(
            OpKind::PatchEmbed { patch_size, embed_dim, in_channels, image_size },
            vec![image, kernel], vec![patches], "patch_embed",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// Execute PatchEmbed layer via CompiledLayer ABI (无 scratchpad 需求)。
    unsafe fn exec_patch_embed(
        layer: &CompiledLayer,
        image: &[f32], kernel: &[f32], patches: &mut [f32], scratch: &mut [u8],
    ) {
        let f = layer.entry_point();
        f(
            image.as_ptr() as *const u8,
            kernel.as_ptr() as *const u8,
            std::ptr::null_mut(),
            std::ptr::null(), std::ptr::null(), 1,
            1, // seq_len unused
            patches.as_mut_ptr() as *mut u8,
            scratch.as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }

    /// T65 PatchEmbed tiny: 1 channel, patch_size=2, image_size=4, embed_dim=4
    /// → 4 patches × 4 embed_dim, 可完整手算对拍。
    #[test]
    fn test_vm_e2e_patch_embed_tiny() {
        let (patch_size, embed_dim, in_channels, image_size) = (2_usize, 4, 1, 4);
        let (g, plan, alloc) = build_patch_embed_plan(patch_size, embed_dim, in_channels, image_size);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile PatchEmbed tiny");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from PatchEmbed tiny");

        let image: Vec<f32> = (0..in_channels * image_size * image_size)
            .map(|i| (i as f32 + 1.0) * 0.1).collect();
        let kernel: Vec<f32> = (0..embed_dim * in_channels * patch_size * patch_size)
            .map(|i| ((i as f32 + 1.0) * 0.07).sin() * 0.5).collect();
        let num_patches = (image_size / patch_size).pow(2);
        let mut patches = vec![0.0_f32; num_patches * embed_dim];
        let mut scratch = vec![0u8; output.scratchpad_bytes.max(1)];
        unsafe { exec_patch_embed(&layer, &image, &kernel, &mut patches, &mut scratch) };

        let expected = scalar_patch_embed_reference(
            &image, &kernel, patch_size, embed_dim, in_channels, image_size);
        let mut max_diff = 0.0_f32;
        for i in 0..num_patches * embed_dim {
            let diff = (patches[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("PatchEmbed tiny: max_abs_diff={max_diff:.2e} scratch={}B",
                  output.scratchpad_bytes);
        assert!(max_diff < 1e-5,
            "PatchEmbed tiny: max_diff={max_diff}\nout={:?}\nexp={:?}",
            patches, expected);
    }

    /// T65 PatchEmbed single_patch: 手算验证 (in=1, patch=2, img=2, embed=1):
    ///   image=[1,2,3,4], kernel=[1,2,3,4] → patches[0]=1·1+2·2+3·3+4·4=30
    /// 与 scalar-ops/src/patch_embed.rs::patch_embed_single_patch_hand_computed 对齐。
    #[test]
    fn test_vm_e2e_patch_embed_single_patch_hand_computed() {
        let (patch_size, embed_dim, in_channels, image_size) = (2_usize, 1, 1, 2);
        let (g, plan, alloc) = build_patch_embed_plan(patch_size, embed_dim, in_channels, image_size);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile PatchEmbed single_patch");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from PatchEmbed single_patch");

        let image = vec![1.0_f32, 2.0, 3.0, 4.0];
        let kernel = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut patches = vec![0.0_f32; 1];
        let mut scratch = vec![0u8; output.scratchpad_bytes.max(1)];
        unsafe { exec_patch_embed(&layer, &image, &kernel, &mut patches, &mut scratch) };

        // 1*1 + 2*2 + 3*3 + 4*4 = 30
        eprintln!("PatchEmbed single_patch: got={} (expected 30.0)", patches[0]);
        assert!((patches[0] - 30.0).abs() < 1e-5,
            "PatchEmbed single_patch: got {} expected 30.0", patches[0]);
    }

    /// T65 PatchEmbed SigLIP-style: in=3, patch=14, img=28 → 4 patches,
    /// embed=16 (小 embed 避免运行时过慢)。随机 image/kernel, 与 scalar ref 对拍。
    #[test]
    fn test_vm_e2e_patch_embed_siglip_style() {
        let (patch_size, embed_dim, in_channels, image_size) = (14_usize, 16, 3, 28);
        let (g, plan, alloc) = build_patch_embed_plan(patch_size, embed_dim, in_channels, image_size);

        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&registry))
            .expect("compile PatchEmbed siglip-style");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0)
            .expect("CompiledLayer from PatchEmbed siglip-style");

        let image_n = in_channels * image_size * image_size;
        let image: Vec<f32> = (0..image_n)
            .map(|i| ((i as f32 + 1.0) * 0.013).sin() * 0.3).collect();
        let kernel_n = embed_dim * in_channels * patch_size * patch_size;
        let kernel: Vec<f32> = (0..kernel_n)
            .map(|i| ((i as f32 + 2.0) * 0.017).cos() * 0.25).collect();
        let num_patches = (image_size / patch_size).pow(2);
        let mut patches = vec![0.0_f32; num_patches * embed_dim];
        let mut scratch = vec![0u8; output.scratchpad_bytes.max(1)];
        unsafe { exec_patch_embed(&layer, &image, &kernel, &mut patches, &mut scratch) };

        let expected = scalar_patch_embed_reference(
            &image, &kernel, patch_size, embed_dim, in_channels, image_size);
        let mut max_diff = 0.0_f32;
        for i in 0..num_patches * embed_dim {
            let diff = (patches[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("PatchEmbed siglip-style (in=3, patch=14, img=28, embed=16): \
                   max_abs_diff={max_diff:.2e} scratch={}B",
                  output.scratchpad_bytes);
        assert!(max_diff < 1e-4,
            "PatchEmbed siglip-style: max_diff={max_diff}\nout[0..8]={:?}\nexp[0..8]={:?}",
            &patches[..8.min(patches.len())], &expected[..8.min(expected.len())]);
    }
}
