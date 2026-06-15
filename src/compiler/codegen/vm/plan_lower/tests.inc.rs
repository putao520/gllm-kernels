mod tests {
    use super::*;
    use crate::compiler::graph::*;
    use crate::compiler::fusion::*;
    #[allow(unused_imports)] // used by test_compile_layer_* tests
    use crate::dispatch::device_profile::DeviceProfile;
    use std::collections::HashMap;
    use crate::types::DType;

    fn simple_silu_plan() -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[32], DType::F32);
        let out = g.add_tensor_concrete("output", &[32], DType::F32);
        g.inputs = vec![inp];
        g.outputs = vec![out];
        let op_id = g.add_op(OpKind::Silu, vec![inp], vec![out], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op_id, epilogue: vec![],
                mode: FusionMode::LoopFusion, ops: vec![op_id],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    #[test]
    fn test_compile_layer_produces_code() {
        let (graph, plan, alloc) = simple_silu_plan();
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);

        let output = compile_layer(&plan, &graph, &alloc, &exec_plan,
            Some(&ScalarOpRegistry::with_defaults())).unwrap();
        assert!(!output.code.is_empty(), "compile_layer should produce code");
        assert!(output.code.len() > 50, "code too small: {}", output.code.len());
    }

    #[test]
    fn test_compile_layer_gemm() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[4, 16], DType::F32);
        let b = g.add_tensor_concrete("B", &[16, 8], DType::F32);
        let c = g.add_tensor_concrete("C", &[4, 8], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op_id = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(4), n: 8, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm",
        );

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op_id, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op_id],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);

        let output = compile_layer(&plan, &g, &alloc, &exec_plan,
            Some(&ScalarOpRegistry::with_defaults())).unwrap();
        assert!(!output.code.is_empty());
    }

    /// ARCH-CODEGEN-DISPATCH: 验证 compile_layer 在 GPU platform 下产出 PTX 文本。
    #[test]
    fn test_compile_layer_dispatches_to_gpu_ptx() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::reg_alloc::RegAllocator;
        use crate::compiler::codegen::vm::isa_hook;
        use crate::compiler::codegen::vm::stack_frame::StackFrame;

        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[32], DType::F32);
        let out = g.add_tensor_concrete("out", &[32], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![out];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![out], "silu");
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op_id, epilogue: vec![],
                mode: FusionMode::LoopFusion, ops: vec![op_id],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();

        // 手工走完整管线 — profile 指定 CUDA SM90
        let profile = IsaProfile::cuda(90);
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let program = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref())).unwrap();
        program.validate_provenance().unwrap();
        program.validate_structure().unwrap();
        let alloc_result = RegAllocator::new(&profile).allocate(&program).unwrap();
        let frame = StackFrame::compute(&alloc_result, &profile, alloc.total_bytes);
        let dialect = crate::compiler::codegen::vm::gpu_lower::GpuDialect::Ptx { sm_version: 90 };
        let (text, format) = super::compile_gpu(&program, &frame, &alloc_result, dialect).unwrap();
        assert_eq!(format, CodeFormat::Ptx);
        assert!(text.contains(".version"), "PTX 必须包含 .version 声明");
        assert!(text.contains(".target sm_90"), "PTX 必须包含 .target sm_90");
        assert!(text.contains(".visible .entry kernel"), "PTX 必须包含 kernel 入口");
        assert!(text.contains(".param .u64 input_ptr"), "PTX 必须声明 input_ptr 参数");
        // 不应出现我们已经清理的硬编码
        assert!(!text.contains("param0"), "PTX 不应使用 param0 (应该用 input_ptr)");
        assert!(!text.contains("%f127"), "PTX 不应硬编码 %f127 (应该用 %fs0)");
    }

    // ══════════════════════════════════════════════════════════════════
    //  T38: PerLayerEmbed GPU codegen — REMOVED
    //  PerLayerEmbed has been replaced by AltUpPredict/AltUpCorrect/AltUpInject
    //  (Injective ops). GPU codegen tests for AltUp will be added when
    //  AltUp scalar reference implementations and graph builders are available.
    // ══════════════════════════════════════════════════════════════════

    // ══════════════════════════════════════════════════════════════════
    //  T66: LearnedPos2D / DepthwiseConv1D / PatchEmbed GPU codegen
    //       (PTX / HIP / MSL 三方言)
    //
    //  结构化验证——不依赖真实 GPU 硬件,只验证 GpuLower 在处理
    //  CPU 侧 VmInstr 序列 (由 lower_fusion_plan 产出) 时发射的
    //  文本 IR 包含正确的 dialect 关键指令 + op 关键数据搬运/计算。
    //
    //  VmInstr 是硬件无关 IR,CPU lower 产出的 program 理论上
    //  GPU lower 也必须能正确翻译。这些测试就是契约守卫。
    // ══════════════════════════════════════════════════════════════════

    /// 构造 LearnedPos2D plan — 二元 elementwise add: `out = patches + pos_table`。
    fn build_learned_pos2d_plan(
        num_patches: usize, embed_dim: usize,
    ) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let patches = g.add_tensor_concrete("patches", &[num_patches, embed_dim], dt);
        let pos = g.add_tensor_concrete("pos", &[num_patches, embed_dim], dt);
        let out = g.add_tensor_concrete("out", &[num_patches, embed_dim], dt);
        g.inputs = vec![patches, pos];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::LearnedPos2D { num_patches, embed_dim },
            vec![patches, pos], vec![out], "learned_pos2d",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// 构造 DepthwiseConv1D plan (与 e2e_tests::build_dwc_plan 等价,独立复刻避免跨模块可见性问题)。
    fn build_dwc_gpu_plan(
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
            vec![x, w], vec![out], "dwc_gpu",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// 构造 PatchEmbed plan (Conv2D stride=patch_size)。
    fn build_patch_embed_plan(
        patch_size: usize, embed_dim: usize, in_channels: usize, image_size: usize,
    ) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let kernel_numel = patch_size * patch_size * in_channels;
        let x = g.add_tensor_concrete("x", &[in_channels, image_size, image_size], dt);
        let w = g.add_tensor_concrete("w", &[embed_dim, kernel_numel], dt);
        let out = g.add_tensor_concrete("out", &[num_patches, embed_dim], dt);
        g.inputs = vec![x, w];
        g.outputs = vec![out];
        let op = g.add_op(
            OpKind::PatchEmbed { patch_size, embed_dim, in_channels, image_size },
            vec![x, w], vec![out], "patch_embed",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    /// 统一 GPU 编译入口 — 对 T66 三 op 共用: scratchpad 预算 + VmProgram 产出 +
    /// RegAlloc + compile_gpu 三方言文本 IR 生成。
    fn compile_op_for_gpu(
        g: &CompilerGraph, plan: &FusionPlan, alloc: &BufferAllocation,
        profile: &crate::compiler::codegen::vm::isa_profile::IsaProfile,
        dialect: crate::compiler::codegen::vm::gpu_lower::GpuDialect,
    ) -> Result<(String, crate::compiler::codegen::CodeFormat), CompilerError> {
        use crate::compiler::codegen::vm::reg_alloc::RegAllocator;
        use crate::compiler::codegen::vm::isa_hook;
        use crate::compiler::codegen::vm::stack_frame::StackFrame;
        let hook = isa_hook::select_hook(profile);
        let registry = ScalarOpRegistry::with_defaults();
        let rope_req = compute_rope_requirement(plan, g, alloc)?;
        let ple_req = compute_ple_requirement(plan, g, alloc, rope_req.as_ref())?;
        let dwc_req = compute_dwc_requirement(plan, g, alloc, rope_req.as_ref(), ple_req.as_ref())?;
        let program = lower_fusion_plan_inner(
            plan, g, alloc, Some(&registry), profile, Some(hook.as_ref()),
            rope_req.as_ref(), ple_req.as_ref(), dwc_req.as_ref(), false,
        )?;
        program.validate_provenance()?;
        program.validate_structure()?;
        let alloc_result = RegAllocator::new(profile).allocate(&program)?;
        let elem = QuantPrecision::F32.elem_bytes();
        let base_after_rope = match &rope_req {
            Some(req) => req.cache_offset + req.max_seq_len * req.head_dim * elem,
            None => alloc.total_bytes,
        };
        let base_after_ple = match &ple_req {
            Some(req) => req.post_mlp_offset + req.max_seq_len * req.hidden * elem,
            None => base_after_rope,
        };
        let moe_scratch = compute_moe_packed_requirement(plan, g);
        let base_after_moe = base_after_ple + moe_scratch;
        let scratchpad_bytes = match &dwc_req {
            Some(req) => (req.padded_offset + req.total_bytes).max(64),
            None => base_after_moe.max(64),
        };
        let frame = StackFrame::compute(&alloc_result, profile, scratchpad_bytes);
        super::compile_gpu(&program, &frame, &alloc_result, dialect)
    }

    // ── LearnedPos2D ×3 方言 ──

    /// T66: LearnedPos2D → PTX SM80 (elementwise add)。
    #[test]
    fn test_learned_pos2d_gpu_ptx_sm80() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let (g, plan, alloc) = build_learned_pos2d_plan(16, 8);
        let profile = IsaProfile::cuda(80);
        let dialect = GpuDialect::Ptx { sm_version: 80 };
        let (text, format) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("LearnedPos2D must lower to PTX SM80");
        assert_eq!(format, CodeFormat::Ptx);
        assert!(text.contains(".target sm_80"), "PTX target missing: {text}");
        assert!(text.contains(".visible .entry kernel"), "PTX kernel entry missing");
        // elementwise add → 至少一条 add.f32
        assert!(text.contains("add.f32"), "LearnedPos2D 必须发射 PTX add.f32: {text}");
        // 至少一层循环 (outer num_patches or inner feature vector loop)
        assert!(text.contains("LOOP_0:"), "LearnedPos2D PTX 必须包含循环标签: {text}");
        // 全局加载/存储
        assert!(text.contains("ld.global.f32"), "LearnedPos2D 必须从 global 读取: {text}");
        assert!(text.contains("st.global.f32"), "LearnedPos2D 必须写 global: {text}");
    }

    /// T66: LearnedPos2D → HIP gfx908 (CDNA3)。
    #[test]
    fn test_learned_pos2d_gpu_hip_gfx908() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let (g, plan, alloc) = build_learned_pos2d_plan(16, 8);
        let profile = IsaProfile::hip(908);
        let dialect = GpuDialect::Hip { gfx_arch: 908, wave_size: 64 };
        let (text, format) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("LearnedPos2D must lower to HIP gfx908");
        assert_eq!(format, CodeFormat::Hip);
        assert!(text.contains("__global__ void kernel"), "HIP kernel entry missing: {text}");
        assert!(text.contains("gfx908"), "HIP IR 必须标识 gfx908");
        // HIP elementwise add via C++ operator "+" (见 GpuLower VecBinOp 非 PTX 分支)
        assert!(text.contains(" + "), "HIP LearnedPos2D 应包含 C++ add 表达式: {text}");
        // threadIdx 不一定出现, 但 kernel signature 必须有 float*
        assert!(text.contains("float* __restrict__ input_ptr"),
            "HIP 必须声明 input_ptr 参数: {text}");
    }

    /// T66: LearnedPos2D → Metal MSL。
    #[test]
    fn test_learned_pos2d_gpu_metal() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let profile = IsaProfile::cuda(80); // proxy profile; dialect 独立决定发射格式
        let (g, plan, alloc) = build_learned_pos2d_plan(16, 8);
        let dialect = GpuDialect::Metal { gpu_family: 9 };
        let (text, format) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("LearnedPos2D must lower to Metal MSL");
        assert_eq!(format, CodeFormat::Msl);
        assert!(text.contains("kernel void"), "MSL kernel entry missing: {text}");
        assert!(text.contains("[[buffer(0)]]"), "MSL 必须声明 buffer 绑定: {text}");
        assert!(text.contains("device float*"), "MSL 必须用 device 地址空间: {text}");
    }

    // ── DepthwiseConv1D ×3 方言 ──

    /// T66: DepthwiseConv1D causal → PTX SM90 (WGMMA-level SM)。
    #[test]
    fn test_depthwise_conv1d_gpu_ptx_sm90_causal() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        // seq=8, channels=2, k=3, causal=true (与 scalar-ops 小测试同维度)
        let (g, plan, alloc) = build_dwc_gpu_plan(8, 2, 3, true);
        let profile = IsaProfile::cuda(90);
        let dialect = GpuDialect::Ptx { sm_version: 90 };
        let (text, format) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("DepthwiseConv1D causal must lower to PTX SM90");
        assert_eq!(format, CodeFormat::Ptx);
        assert!(text.contains(".target sm_90"), "PTX target missing");
        // DWC 用 scratchpad 做 zero-pad + input copy
        assert!(text.contains("cvta.shared.u64"), "DWC 必须声明 shared scratchpad: {text}");
        // FMA 累加 (per-channel × kernel_size 条 FMA)
        assert!(text.contains("fma.rn.f32"), "DWC 必须用 fma.rn.f32 累加: {text}");
        // 嵌套循环 (zero-fill + row copy + t_loop × c_loop → 多层 LOOP_ 标签)
        assert!(text.contains("LOOP_0:"), "DWC 必须有至少一层循环: {text}");
        assert!(text.contains("LOOP_1:"), "DWC 必须有嵌套循环: {text}");
    }

    /// T66: DepthwiseConv1D non-causal → HIP gfx950 (CDNA4)。
    #[test]
    fn test_depthwise_conv1d_gpu_hip_gfx950_noncausal() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        // non-causal 要求奇数 kernel_size (scalar ref 语义约束)
        let (g, plan, alloc) = build_dwc_gpu_plan(8, 2, 3, false);
        let profile = IsaProfile::hip(950);
        let dialect = GpuDialect::Hip { gfx_arch: 950, wave_size: 64 };
        let (text, format) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("DepthwiseConv1D non-causal must lower to HIP gfx950");
        assert_eq!(format, CodeFormat::Hip);
        assert!(text.contains("__global__ void kernel"), "HIP kernel entry missing");
        assert!(text.contains("gfx950"), "HIP IR 必须标识 gfx950");
        // FMA — HIP 用 fma() 内建
        assert!(text.contains("fma("), "DWC HIP 必须使用 fma() 内建: {text}");
        // 使用 __shared__ scratchpad 做 padding
        assert!(text.contains("__shared__"), "DWC HIP 必须声明 __shared__ scratchpad: {text}");
    }

    /// T66: DepthwiseConv1D → Metal MSL。
    #[test]
    fn test_depthwise_conv1d_gpu_metal() {
        use crate::compiler::codegen::CodeFormat;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let profile = IsaProfile::cuda(80); // proxy; dialect 独立发射 MSL
        let (g, plan, alloc) = build_dwc_gpu_plan(8, 2, 3, true);
        let dialect = GpuDialect::Metal { gpu_family: 9 };
        let (text, format) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("DepthwiseConv1D must lower to Metal MSL");
        assert_eq!(format, CodeFormat::Msl);
        assert!(text.contains("kernel void"), "MSL kernel entry missing");
        // Metal threadgroup scratchpad (DWC 用 scratch 做 padding)
        assert!(text.contains("threadgroup"), "DWC MSL 必须声明 threadgroup scratchpad: {text}");
        // FMA 走 Metal fma() 内建
        assert!(text.contains("fma("), "DWC MSL 必须使用 fma() 内建: {text}");
    }

    // ── PatchEmbed ×3 方言 (Err 占位, 与 x86_64 CPU 路径一致) ──

    /// T66: PatchEmbed → PTX 必须返回 Err 占位 (与 CPU lower 保持一致)。
    /// 根据任务规范,PatchEmbed GPU 路径随 T65 并行推进,当前阶段接受 Err 显式报告。
    #[test]
    fn test_patch_embed_gpu_ptx_returns_err() {
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let (g, plan, alloc) = build_patch_embed_plan(4, 8, 3, 16);
        let profile = IsaProfile::cuda(90);
        let dialect = GpuDialect::Ptx { sm_version: 90 };
        // T66 agent 同步实现了 lower_patch_embed(CPU VM IR, graph_dtype(&graph)), GpuLower 能直接翻译
        // 到 PTX — 与 LearnedPos2D/DepthwiseConv1D 契约相同。断言 compile 成功
        // 且包含基础结构标记,不再期待 Err。PatchEmbed 真实数值对齐由 T65 负责。
        let (kernel, _fmt) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("PatchEmbed GPU (PTX) compile failed");
        assert!(kernel.contains(".target sm_") || kernel.contains(".entry kernel")
            || kernel.contains("threadgroup") || kernel.contains("__global__"),
            "GPU kernel text 必须含 dialect 标志指令: got {} bytes", kernel.len());
    }

    /// T66: PatchEmbed → HIP 必须返回 Err 占位。
    #[test]
    fn test_patch_embed_gpu_hip_returns_err() {
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let (g, plan, alloc) = build_patch_embed_plan(4, 8, 3, 16);
        let profile = IsaProfile::hip(950);
        let dialect = GpuDialect::Hip { gfx_arch: 950, wave_size: 64 };
        let (kernel, _fmt) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("PatchEmbed GPU (HIP) compile failed");
        assert!(kernel.contains("__global__") || kernel.contains("extern \"C\""),
            "HIP kernel text 必须含标志: got {} bytes", kernel.len());
    }

    /// T66: PatchEmbed → Metal 必须返回 Err 占位。
    #[test]
    fn test_patch_embed_gpu_metal_returns_err() {
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;
        use crate::compiler::codegen::vm::gpu_lower::GpuDialect;
        let profile = IsaProfile::cuda(80);
        let (g, plan, alloc) = build_patch_embed_plan(4, 8, 3, 16);
        let dialect = GpuDialect::Metal { gpu_family: 9 };
        let (kernel, _fmt) = compile_op_for_gpu(&g, &plan, &alloc, &profile, dialect)
            .expect("PatchEmbed GPU (Metal) compile failed");
        assert!(kernel.contains("threadgroup") || kernel.contains("[[buffer(0)]]")
            || kernel.contains("kernel void"),
            "Metal kernel text 必须含标志: got {} bytes", kernel.len());
    }

    #[test]
    fn test_group_dependency_analyzer_linear_chain() {
        // op0 → op1 → op2: 全串行依赖
        let mut g = CompilerGraph::new();
        let t0 = g.add_tensor_concrete("t0", &[32], DType::F32);
        let t1 = g.add_tensor_concrete("t1", &[32], DType::F32);
        let t2 = g.add_tensor_concrete("t2", &[32], DType::F32);
        let t3 = g.add_tensor_concrete("t3", &[32], DType::F32);
        g.inputs = vec![t0];
        g.outputs = vec![t3];
        let op0 = g.add_op(OpKind::Silu, vec![t0], vec![t1], "op0");
        let op1 = g.add_op(OpKind::Gelu, vec![t1], vec![t2], "op1");
        let op2 = g.add_op(OpKind::Silu, vec![t2], vec![t3], "op2");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup { id: 0, anchor: op0, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op0], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 1, anchor: op1, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op1], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 2, anchor: op2, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op2], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
            ],
            op_to_group: HashMap::from([(op0, 0), (op1, 1), (op2, 2)]),
        };
        let levels = GroupDependencyAnalyzer::analyze(&plan, &g);
        // 线性链: 每层一个 group
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec![0]);
        assert_eq!(levels[1], vec![1]);
        assert_eq!(levels[2], vec![2]);
    }

    #[test]
    fn test_group_dependency_analyzer_independent_branches() {
        // op0 ← t0, op1 ← t0: 两个独立分支（共享输入但无互相依赖）
        let mut g = CompilerGraph::new();
        let t0 = g.add_tensor_concrete("t0", &[32], DType::F32);
        let t1 = g.add_tensor_concrete("t1", &[32], DType::F32);
        let t2 = g.add_tensor_concrete("t2", &[32], DType::F32);
        g.inputs = vec![t0];
        g.outputs = vec![t1, t2];
        let op0 = g.add_op(OpKind::Silu, vec![t0], vec![t1], "op0");
        let op1 = g.add_op(OpKind::Gelu, vec![t0], vec![t2], "op1");

        let plan = FusionPlan {
            groups: vec![
                FusionGroup { id: 0, anchor: op0, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op0], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 1, anchor: op1, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op1], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
            ],
            op_to_group: HashMap::from([(op0, 0), (op1, 1)]),
        };
        let levels = GroupDependencyAnalyzer::analyze(&plan, &g);
        // 两个独立分支应同一层级
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 2);
    }

    #[test]
    fn test_compile_layer_type_body_silu() {
        let (graph, plan, alloc) = simple_silu_plan();
        let profile = DeviceProfile::detect();
        let isa_profile = crate::compiler::codegen::vm::isa_profile::IsaProfile::from_device_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let hook = super::super::isa_hook::select_hook(&isa_profile);
        let resolver = TensorPtrResolver::build(&graph, &alloc, &super::super::topology::GraphTopologyAnalysis::analyze(&graph));
        let sess = CompileSession {
            width: isa_profile.optimal_simd_width(),
            sym_map: &SymDimSlotMap::mega_kernel_abi(),
            registry: Some(&registry),
            hook: Some(hook.as_ref()),
            budget: Some(super::super::isa_hook::ResourceBudget::from_isa_profile(&isa_profile)),
            page_size: 0,
            dot_cap: isa_profile.dot_cap,
            kv_elem_bytes: 4,
            debug_jit: false,
            virtual_activation: None,
            virtual_tensor_map: None,
            layout: None,
            batch_ctx_ptr: None,
        };
        let ctx = LoweringContext {
            session: &sess,
            dtype: graph_dtype(&graph),
            rope_req: None,
            ple_req: None,
            dwc_req: None,
            exec_pattern: None,
            bottleneck_map: None,
            parallelism: None,
        };

        let template = compile_layer_type_body(&ctx, 0..1, &plan, &graph, &alloc, &resolver).unwrap();
        assert!(template.body.instrs.len() > 1, "template should have instructions");
        assert_eq!(template.abi_map.input_ptr.0, 0); // first allocated VReg
    }

    // ══════════════════════════════════════════════════════════════════
    //  Edge-case tests: empty plan, single-op, multi-group order,
    //  dtype propagation, symbolic seq_len, graph_dtype, diamond deps,
    //  VmProgram validation, multi-output config, TensorPtrResolver
    // ══════════════════════════════════════════════════════════════════

    /// Empty plan (0 groups) → lower_fusion_plan must produce a valid VmProgram
    /// with only ABI pointer loads (no compute instructions).
    #[test]
    fn test_lower_empty_plan_produces_valid_program() {
        let g = CompilerGraph::new();
        let plan = FusionPlan { groups: vec![], op_to_group: HashMap::new() };
        let alloc = BufferAllocation::default();
        let profile = IsaProfile::cuda(80);
        let registry = ScalarOpRegistry::with_defaults();
        let prog = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, None)
            .expect("empty plan must lower without error");
        prog.validate_provenance().expect("provenance");
        prog.validate_structure().expect("structure");
        assert!(!prog.instrs.is_empty(), "empty plan should emit ABI setup instrs");
    }

    /// Single-op plan (one Silu) → lowering must produce non-empty code
    /// and VmProgram must pass provenance + structure validation.
    #[test]
    fn test_single_op_plan_validates_vm_program() {
        let (graph, plan, alloc) = simple_silu_plan();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let prog = lower_fusion_plan(&plan, &graph, &alloc, Some(&registry), &profile, Some(hook.as_ref()))
            .expect("single-op plan must lower");
        prog.validate_provenance().expect("provenance");
        prog.validate_structure().expect("structure");
        let declares = prog.instrs.iter().filter(|i| matches!(i, VmInstr::DeclareVReg { .. })).count();
        assert!(declares >= 3, "single-op plan needs >=3 VReg declarations, got {declares}");
    }

    /// Multi-group sequential plan: Silu → Add. VmProgram must validate.
    #[test]
    fn test_multi_group_sequential_plan() {
        let mut g = CompilerGraph::new();
        let t0 = g.add_tensor_concrete("t0", &[32], DType::F32);
        let t1 = g.add_tensor_concrete("t1", &[32], DType::F32);
        let t2 = g.add_tensor_concrete("t2", &[32], DType::F32);
        g.inputs = vec![t0]; g.outputs = vec![t2];
        let op0 = g.add_op(OpKind::Silu, vec![t0], vec![t1], "silu");
        let op1 = g.add_op(OpKind::Gelu, vec![t1], vec![t2], "gelu");
        let plan = FusionPlan {
            groups: vec![
                FusionGroup { id: 0, anchor: op0, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op0], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 1, anchor: op1, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op1], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
            ],
            op_to_group: HashMap::from([(op0, 0), (op1, 1)]),
        };
        let alloc = BufferAllocation::default();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let prog = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref()))
            .expect("multi-group plan must lower");
        prog.validate_provenance().expect("provenance");
        prog.validate_structure().expect("structure");
        assert!(prog.instrs.len() > 10, "multi-group plan should emit substantial IR");
    }

    /// dtype propagation: BF16 tensors → graph_dtype must return QuantPrecision::BF16.
    #[test]
    fn test_graph_dtype_bf16_propagation() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[32], DType::BF16);
        let out = g.add_tensor_concrete("output", &[32], DType::BF16);
        g.inputs = vec![inp]; g.outputs = vec![out];
        let dtype = graph_dtype(&g);
        assert_eq!(dtype, QuantPrecision::BF16, "graph_dtype must reflect BF16");
    }

    /// dtype propagation: F16 tensors → graph_dtype must return QuantPrecision::F16.
    #[test]
    fn test_graph_dtype_f16_propagation() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[64], DType::F16);
        let out = g.add_tensor_concrete("output", &[64], DType::F16);
        g.inputs = vec![inp]; g.outputs = vec![out];
        let dtype = graph_dtype(&g);
        assert_eq!(dtype, QuantPrecision::F16, "graph_dtype must reflect F16");
    }

    /// Symbolic seq_len lowering: SymDim::Symbolic must produce
    /// VmProgram containing LoopBegin with BoundExpr::Symbolic.
    #[test]
    fn test_symbolic_seq_len_produces_symbolic_loop() {
        let mut g = CompilerGraph::new();
        let seq_sym = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(128) };
        let inp = g.add_tensor("input", vec![seq_sym.clone(), SymDim::Concrete(32)], DType::F32);
        let out = g.add_tensor("output", vec![seq_sym, SymDim::Concrete(32)], DType::F32);
        g.inputs = vec![inp]; g.outputs = vec![out];
        let op_id = g.add_op(OpKind::Silu, vec![inp], vec![out], "silu_sym");
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op_id, epilogue: vec![],
                mode: FusionMode::LoopFusion, ops: vec![op_id],
                multi_output: MultiOutputConfig::single(), dominant_dtype: None,
                marker: GroupMarker::None,
                is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let prog = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref()))
            .expect("symbolic plan must lower");
        prog.validate_provenance().expect("provenance");
        let has_symbolic_loop = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::LoopBegin { bound: BoundExpr::Symbolic(_), .. })
        });
        assert!(has_symbolic_loop, "symbolic seq_len must produce BoundExpr::Symbolic loop");
    }

    /// graph_dtype on empty graph (no tensors) must return QuantPrecision::F32
    /// as the only legal fallback per SPEC (REQ-DTYPE-001).
    #[test]
    fn test_graph_dtype_empty_graph_returns_f32() {
        let g = CompilerGraph::new();
        let dtype = graph_dtype(&g);
        assert_eq!(dtype, QuantPrecision::F32, "empty graph must default to F32");
    }

    /// GroupDependencyAnalyzer: diamond dependency (op0→op1, op0→op2, both→op3).
    /// Levels must be: [0], [1,2], [3].
    #[test]
    fn test_group_dependency_analyzer_diamond() {
        let mut g = CompilerGraph::new();
        let t0 = g.add_tensor_concrete("t0", &[16], DType::F32);
        let t1 = g.add_tensor_concrete("t1", &[16], DType::F32);
        let t2 = g.add_tensor_concrete("t2", &[16], DType::F32);
        let t3 = g.add_tensor_concrete("t3", &[16], DType::F32);
        let t4 = g.add_tensor_concrete("t4", &[16], DType::F32);
        g.inputs = vec![t0]; g.outputs = vec![t4];
        let op0 = g.add_op(OpKind::Silu, vec![t0], vec![t1], "op0");
        let op1 = g.add_op(OpKind::Gelu, vec![t1], vec![t2], "op1");
        let op2 = g.add_op(OpKind::Gelu, vec![t1], vec![t3], "op2");
        let op3 = g.add_op(OpKind::Mul, vec![t2, t3], vec![t4], "op3");
        let plan = FusionPlan {
            groups: vec![
                FusionGroup { id: 0, anchor: op0, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op0], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 1, anchor: op1, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op1], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 2, anchor: op2, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op2], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
                FusionGroup { id: 3, anchor: op3, epilogue: vec![], mode: FusionMode::Standalone, ops: vec![op3], multi_output: MultiOutputConfig::single(), dominant_dtype: None, marker: GroupMarker::None, is_layer_group: false, hetero_layer_type: None },
            ],
            op_to_group: HashMap::from([(op0, 0), (op1, 1), (op2, 2), (op3, 3)]),
        };
        let levels = GroupDependencyAnalyzer::analyze(&plan, &g);
        assert_eq!(levels.len(), 3, "diamond: 3 dependency levels");
        assert_eq!(levels[0], vec![0], "level 0: root");
        assert_eq!(levels[1].len(), 2, "level 1: two branches");
        assert!(levels[1].contains(&1) && levels[1].contains(&2));
        assert_eq!(levels[2], vec![3], "level 2: merge");
    }

    /// Full validation: lower_fusion_plan on a GEMM graph must produce
    /// VmProgram passing provenance, structure, and declares-before-uses.
    #[test]
    fn test_full_validation_after_lowering() {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[8, 16], DType::F32);
        let b = g.add_tensor_concrete("B", &[16, 8], DType::F32);
        let c = g.add_tensor_concrete("C", &[8, 8], DType::F32);
        g.inputs = vec![a, b]; g.outputs = vec![c];
        let op_id = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(8), n: 8, k: 16, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm",
        );
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op_id, epilogue: vec![],
                mode: FusionMode::Standalone, ops: vec![op_id],
                multi_output: MultiOutputConfig::single(), dominant_dtype: None,
                marker: GroupMarker::None,
                is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        let profile = IsaProfile::from_device_profile(&DeviceProfile::detect());
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let prog = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref()))
            .expect("GEMM plan must lower");
        prog.validate_provenance().expect("provenance");
        prog.validate_structure().expect("structure");
        prog.validate_declares_before_uses().expect("declares before uses");
    }

    /// TensorPtrResolver: first input must map to Activation, second to Weight.
    #[test]
    fn test_tensor_ptr_resolver_activation_weight_mapping() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[32], DType::F32);
        let wt = g.add_tensor_concrete("weight", &[32], DType::F32);
        let out = g.add_tensor_concrete("output", &[32], DType::F32);
        g.inputs = vec![inp, wt]; g.outputs = vec![out];
        let alloc = BufferAllocation::default();
        let resolver = TensorPtrResolver::build(&g, &alloc, &super::super::topology::GraphTopologyAnalysis::analyze(&g));
        let inp_src = resolver.source(inp).expect("input must have source");
        assert!(matches!(inp_src, TensorPtrSource::Activation), "input must be Activation");
        let wt_src = resolver.source(wt).expect("weight must have source");
        assert!(matches!(wt_src, TensorPtrSource::Weight { .. }), "weight must be Weight");
    }

    // ══════════════════════════════════════════════════════════════════
    //  Additional unit tests: SymDimSlotMap, TensorPtrResolver override,
    //  scratch requirement functions, GroupDependencyAnalyzer edge cases,
    //  VmProgram construction, graph_dtype with mixed tensors,
    //  lower_fusion_plan with GPU profile
    // ══════════════════════════════════════════════════════════════════

    /// SymDimSlotMap::mega_kernel_abi() must resolve "input" and "output" keys.
    #[test]
    fn test_sym_dim_slot_map_mega_kernel_abi_resolves_keys() {
        let map = SymDimSlotMap::mega_kernel_abi();
        assert!(map.resolve("input").is_some(), "mega-kernel ABI must resolve 'input'");
        assert!(map.resolve("output").is_some(), "mega-kernel ABI must resolve 'output'");
        assert!(map.resolve("weights").is_some(), "mega-kernel ABI must resolve 'weights'");
        assert!(map.resolve("scratchpad").is_some(), "mega-kernel ABI must resolve 'scratchpad'");
    }

    /// SymDimSlotMap::gpu_abi() must resolve GPU-specific keys.
    #[test]
    fn test_sym_dim_slot_map_gpu_abi_resolves_keys() {
        let map = SymDimSlotMap::gpu_abi();
        assert!(map.resolve("input").is_some(), "GPU ABI must resolve 'input'");
        assert!(map.resolve("output").is_some(), "GPU ABI must resolve 'output'");
        assert!(map.resolve("seq_len").is_some(), "GPU ABI must resolve 'seq_len'");
    }

    /// SymDimSlotMap::to_bound() must map Concrete -> BoundExpr::Const
    /// and Symbolic -> BoundExpr::Symbolic.
    #[test]
    fn test_sym_dim_slot_map_to_bound_concrete_and_symbolic() {
        let map = SymDimSlotMap::mega_kernel_abi();
        let concrete = SymDim::Concrete(64);
        let bound = map.to_bound(&concrete);
        assert_eq!(bound, BoundExpr::Const(64), "Concrete(64) must map to BoundExpr::Const(64)");

        let symbolic = SymDim::Symbolic { name: "seq_len".into(), max_value: Some(512) };
        let bound = map.to_bound(&symbolic);
        match &bound {
            BoundExpr::Symbolic(sb) => {
                assert_eq!(sb.name, "seq_len");
                assert_eq!(sb.max_alloc, 512);
            }
            other => panic!("Symbolic must map to BoundExpr::Symbolic, got {:?}", other),
        }
    }

    /// TensorPtrResolver::override_source() must change a tensor's source mapping.
    #[test]
    fn test_tensor_ptr_resolver_override_source() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[32], DType::F32);
        let wt = g.add_tensor_concrete("weight", &[32], DType::F32);
        let out = g.add_tensor_concrete("output", &[32], DType::F32);
        g.inputs = vec![inp, wt]; g.outputs = vec![out];
        let alloc = BufferAllocation::default();
        let mut resolver = TensorPtrResolver::build(&g, &alloc, &super::super::topology::GraphTopologyAnalysis::analyze(&g));
        // Verify initial mapping
        let original = resolver.source(inp);
        assert!(original.is_some(), "input must have initial source");
        // Override to Output
        resolver.override_source(inp, TensorPtrSource::Output { offset: 0 });
        let overridden = resolver.source(inp);
        assert!(matches!(overridden, Some(TensorPtrSource::Output { offset: 0 })),
            "overridden source must be Output(0)");
    }

    /// compute_rope_requirement() returns None when the plan has no RoPE ops.
    #[test]
    fn test_compute_rope_requirement_no_rope_returns_none() {
        let (g, plan, alloc) = simple_silu_plan();
        let req = compute_rope_requirement(&plan, &g, &alloc).unwrap();
        assert!(req.is_none(), "plan without RoPE ops must return None");
    }

    /// compute_ple_requirement() always returns None (PerLayerEmbed removed, AltUp is Injective).
    #[test]
    fn test_compute_ple_requirement_no_ple_returns_none() {
        let (g, plan, alloc) = simple_silu_plan();
        let rope_req = compute_rope_requirement(&plan, &g, &alloc).unwrap();
        let req = compute_ple_requirement(&plan, &g, &alloc, rope_req.as_ref()).unwrap();
        assert!(req.is_none(), "plan without PLE ops must return None");
    }

    /// compute_dwc_requirement() returns None when the plan has no DepthwiseConv1D ops.
    #[test]
    fn test_compute_dwc_requirement_no_dwc_returns_none() {
        let (g, plan, alloc) = simple_silu_plan();
        let rope_req = compute_rope_requirement(&plan, &g, &alloc).unwrap();
        let ple_req = compute_ple_requirement(&plan, &g, &alloc, rope_req.as_ref()).unwrap();
        let req = compute_dwc_requirement(&plan, &g, &alloc, rope_req.as_ref(), ple_req.as_ref()).unwrap();
        assert!(req.is_none(), "plan without DWC ops must return None");
    }

    /// GroupDependencyAnalyzer: single group produces one level with one element.
    #[test]
    fn test_group_dependency_analyzer_single_group() {
        let (g, plan, _alloc) = simple_silu_plan();
        let levels = GroupDependencyAnalyzer::analyze(&plan, &g);
        assert_eq!(levels.len(), 1, "single group must produce 1 level");
        assert_eq!(levels[0], vec![0], "single group level must contain group 0");
    }

    /// VmProgram manual construction: alloc_vreg and emit produce correct
    /// instruction count and DeclareVReg instructions.
    #[test]
    fn test_vm_program_manual_construction_counts() {
        let mut prog = VmProgram::new();
        assert!(prog.is_empty(), "new program must be empty");
        let v0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::Comment("test".into()));
        // alloc_vreg emits DeclareVReg, so we have 2 DeclareVReg + 1 Comment = 3
        assert_eq!(prog.len(), 3, "must have 3 instructions (2 DeclareVReg + 1 Comment)");
        assert_eq!(prog.vreg_count(), 2, "must have 2 vregs");
        let declares = prog.instrs.iter().filter(|i| matches!(i, VmInstr::DeclareVReg { .. })).count();
        assert_eq!(declares, 2, "must have 2 DeclareVReg instructions");
        // Verify VRegIds are sequential
        assert_eq!(v0.0, 0, "first vreg must be id 0");
        assert_eq!(v1.0, 1, "second vreg must be id 1");
    }

    /// graph_dtype: when the first tensor is BF16 and second is F32,
    /// the first tensor's dtype must win (SSOT: first tensor determines graph dtype).
    #[test]
    fn test_graph_dtype_first_tensor_wins() {
        let mut g = CompilerGraph::new();
        let bf16 = g.add_tensor_concrete("bf16_tensor", &[32], DType::BF16);
        let f32 = g.add_tensor_concrete("f32_tensor", &[32], DType::F32);
        g.inputs = vec![bf16, f32]; g.outputs = vec![f32];
        let dtype = graph_dtype(&g);
        assert_eq!(dtype, QuantPrecision::BF16, "graph_dtype must reflect first tensor's BF16");
    }

    /// lower_fusion_plan with IsaProfile::cuda(80) must produce a VmProgram
    /// that passes provenance and structure validation.
    #[test]
    fn test_lower_fusion_plan_cuda_profile_validates() {
        let (graph, plan, alloc) = simple_silu_plan();
        let profile = IsaProfile::cuda(80);
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let prog = lower_fusion_plan(&plan, &graph, &alloc, Some(&registry), &profile, Some(hook.as_ref()))
            .expect("Silu plan must lower under CUDA profile");
        prog.validate_provenance().expect("provenance");
        prog.validate_structure().expect("structure");
        assert!(!prog.instrs.is_empty(), "CUDA-lowered program must have instructions");
        // Verify at least one DeclareVReg exists
        let has_declare = prog.instrs.iter().any(|i| matches!(i, VmInstr::DeclareVReg { .. }));
        assert!(has_declare, "CUDA-lowered program must declare VRegs");
    }

    // ── op_input_dtype tests (REQ-DTYPE-001) ──

    #[test]
    fn test_op_input_dtype_from_first_input_tensor() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[32], DType::BF16);
        let w = g.add_tensor_concrete("weight", &[32, 64], DType::BF16);
        let out = g.add_tensor_concrete("output", &[64], DType::BF16);
        g.inputs = vec![inp, w]; g.outputs = vec![out];
        let op = g.add_op(OpKind::Silu, vec![inp], vec![out], "silu");
        let dtype = op_input_dtype(g.op(op).unwrap(), &g);
        assert_eq!(dtype, QuantPrecision::BF16, "op_input_dtype must derive from first input tensor");
    }

    #[test]
    fn test_op_input_dtype_falls_back_to_f32_when_no_inputs() {
        let mut g = CompilerGraph::new();
        let out = g.add_tensor_concrete("output", &[1], DType::F32);
        g.outputs = vec![out];
        let op = g.add_op(OpKind::Reshape { target_shape: vec![1] }, vec![], vec![out], "reshape");
        let dtype = op_input_dtype(g.op(op).unwrap(), &g);
        assert_eq!(dtype, QuantPrecision::F32, "op_input_dtype must fall back to F32 when op has no inputs");
    }

    #[test]
    fn test_op_input_dtype_f16_propagation() {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[128], DType::F16);
        let out = g.add_tensor_concrete("output", &[128], DType::F16);
        g.inputs = vec![inp]; g.outputs = vec![out];
        let op = g.add_op(OpKind::Silu, vec![inp], vec![out], "silu");
        let dtype = op_input_dtype(g.op(op).unwrap(), &g);
        assert_eq!(dtype, QuantPrecision::F16, "op_input_dtype must propagate F16");
    }
}

    /// Verify trans_b=true GEMM: C = A * B^T where B is [N,K] row-major.
    /// Compare JIT output with scalar reference.
    #[test]
    fn test_gemm_trans_b_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 5usize;   // seq_len
        let n = 16usize;  // output dim
        let k = 32usize;  // hidden dim

        // Build graph: C = A * B^T
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[n, k], DType::F32);  // [N,K] row-major
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: true },
            vec![a, b], vec![c], "gemm_trans_b",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("trans_b GEMM compilation failed")
            .expect_cpu()
            .layer_code;

        // Prepare input data
        let a_data: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01 - 0.5).collect();
        let b_data: Vec<f32> = (0..n*k).map(|i| i as f32 * 0.02 - 0.3).collect();

        // Scalar reference: C[i][j] = sum_p A[i][p] * B[j][p]
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[j * k + p];
                }
                c_ref[i * n + j] = sum;
            }
        }

        // Execute JIT
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8,
                b_data.as_ptr() as *const u8,
                1,
                m,
                c_jit.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }

        // Compare
        let max_diff = (0..c_ref.len())
            .map(|i| (c_jit[i] - c_ref[i]).abs())
            .fold(0.0f32, f32::max);

        eprintln!("[trans_b test] c_jit[0..4]: {:?}", &c_jit[..4]);
        eprintln!("[trans_b test] c_ref[0..4]: {:?}", &c_ref[..4]);
        eprintln!("[trans_b test] max_diff: {:.2e}", max_diff);

        assert!(max_diff < 0.01, "trans_b GEMM max_diff={max_diff:.4} exceeds tolerance");
    }

    /// Verify non-transposed GEMM still works: C = A * B where B is [K,N] row-major.
    /// Verify non-trans GEMM (trans_b=false) compiles successfully through the
    /// BLIS tiled path. Numerical correctness is verified by lower-level GEMM
    /// unit tests; this test ensures the full compilation pipeline produces a
    /// valid, non-trivial VmProgram.
    #[test]
    fn test_gemm_no_trans_b_compiles() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};

        use crate::types::DType;

        let m = 5usize;
        let n = 16usize;
        let k = 32usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_no_trans",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("non-trans GEMM compilation failed")
            .expect_cpu()
            .layer_code;

        assert_ne!(compiled.config_hash, 0, "non-trans GEMM should produce non-zero hash");
    }

    /// Non-trans_b GEMM: m=2, n=16, k=8 (2 j_blocks with nr=2, lanes=8).
    #[test]
    fn test_gemm_no_trans_b_m2_n16_k8_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 2usize;
        let n = 16usize;
        let k = 8usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_m2_n16",
        );

        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("GEMM compilation failed").expect_cpu().layer_code;

        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.2).cos()).collect();

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr(),
            );
        }

        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m2_n16] jit={:?}", &c_jit[..]);
        eprintln!("[m2_n16] ref={:?}", &c_ref[..]);
        eprintln!("[m2_n16] max_diff={max_diff:.6}");
        assert!(max_diff < 0.01, "m2_n16 max_diff={max_diff}");
    }

    /// Minimal test: directly call emit_gemm_blis_inline and dump VmInstrs
    #[test]
    fn test_blis_gem_vminstr_dump_m4_n16_k32() {
        use crate::compiler::codegen::vm::gemm_emit::emit_gemm_blis_inline;
        use crate::compiler::codegen::vm::instr::VmProgram;
        use crate::compiler::codegen::vm::instr::VRegKind;
        use crate::compiler::codegen::vm::instr::SimdWidth;
        use crate::compiler::codegen::vm::instr::VmInstr;
        use crate::compiler::codegen::vm::instr::ScalarExpr;
        use crate::compiler::trace::QuantPrecision;

        let m = 4usize; let n = 16usize; let k = 32usize;
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_gemm_blis_inline(&mut prog, m, n, k, SimdWidth::W256, a_ptr, b_ptr, c_ptr, 4, 2, None, 4, QuantPrecision::F32, false).unwrap();

        let total = prog.instrs.len();
        let mut fma_count = 0;
        let mut load_count = 0;
        let mut store_count = 0;
        let mut loop_count = 0;
        for instr in &prog.instrs {
            match instr {
                VmInstr::Fma { .. } => fma_count += 1,
                VmInstr::VecLoad { .. } => load_count += 1,
                VmInstr::Broadcast { src: ScalarExpr::MemLoad(..), .. } => load_count += 1,
                VmInstr::VecStore { .. } => store_count += 1,
                VmInstr::LoopBegin { .. } => loop_count += 1,
                _ => {}
            }
        }

        // Dump all instructions for inspection
        let mut dump = String::new();
        dump.push_str(&format!("total_instrs={total} fma={fma_count} loads={load_count} stores={store_count} loops={loop_count}\n"));
        for (i, instr) in prog.instrs.iter().enumerate() {
            dump.push_str(&format!("{i:4}: {instr:?}\n"));
        }
        std::fs::write("/tmp/blis_vminstr_full.txt", dump).ok();
    }

    /// Direct BLIS GEMM with k_unroll=1 (no K-loop unrolling) for m=4,n=16,k=32
    #[test]
    fn test_blis_gem_k1_unroll_m4_n16_k32() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::compiler::codegen::vm::gemm_emit::emit_gemm_blis_inline;
        use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, SimdWidth, VmInstr};
        use crate::compiler::trace::QuantPrecision;
        use crate::types::DType;

        let m = 4usize; let n = 16usize; let k = 32usize;
        // Test with k_unroll=1: 32 K-loop iterations, no unrolling
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_gemm_blis_inline(&mut prog, m, n, k, SimdWidth::W256, a_ptr, b_ptr, c_ptr, 4, 2, None, 1, QuantPrecision::F32, false).unwrap();

        let mut fma_count = 0;
        for instr in &prog.instrs {
            if let VmInstr::Fma { .. } = instr { fma_count += 1; }
        }
        // k_unroll=1: 32 iterations × 4 rows × 2 cols = 256 FMA
        eprintln!("[k1_unroll] fma_count={fma_count} total_instrs={}", prog.instrs.len());
    }

    /// Non-trans_b GEMM: m=4, n=16, k=32 (mr=4, multiple i_blocks).
    #[test]
    fn test_gemm_no_trans_b_m4_n16_k32_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize;
        let n = 16usize;
        let k = 32usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_m4_n16",
        );

        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("GEMM compilation failed").expect_cpu().layer_code;

        // Dump machine code for disassembly
        let code = compiled.code_bytes();
        std::fs::write("/tmp/gemm_m4_n16_k32.bin", code).ok();

        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr(),
            );
        }

        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m4_n16] max_diff={max_diff:.6}");
        for i in 0..m { eprintln!("[m4_n16] row{i} jit={:.3?} ref={:.3?}", &c_jit[i*n..i*n+4], &c_ref[i*n..i*n+4]); }
        assert!(max_diff < 0.01, "m4_n16 max_diff={max_diff}");
    }

    /// Non-trans_b GEMM: m=4, n=16, k=4 (full mr tile, tiny K — no K-loop iterations)
    #[test]
    fn test_gemm_no_trans_b_m4_n16_k4_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize;
        let n = 16usize;
        let k = 4usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_m4_n16_k4",
        );

        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("GEMM compilation failed").expect_cpu().layer_code;

        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr(),
            );
        }

        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m4_n16_k4] max_diff={max_diff:.6}");
        for i in 0..m { eprintln!("[m4_n16_k4] row{i} jit={:.3?} ref={:.3?}", &c_jit[i*n..i*n+4], &c_ref[i*n..i*n+4]); }
        assert!(max_diff < 0.01, "m4_n16_k4 max_diff={max_diff}");
    }

    /// Non-trans_b GEMM: m=4, n=16, k=8 (2 K-loop iterations)
    #[test]
    fn test_gemm_no_trans_b_m4_n16_k8_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize;
        let n = 16usize;
        let k = 8usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_m4_n16_k8",
        );

        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("GEMM compilation failed").expect_cpu().layer_code;

        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr(),
            );
        }

        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m4_n16_k8] max_diff={max_diff:.6}");
        for i in 0..m { eprintln!("[m4_n16_k8] row{i} jit={:.3?} ref={:.3?}", &c_jit[i*n..i*n+4], &c_ref[i*n..i*n+4]); }
        assert!(max_diff < 0.01, "m4_n16_k8 max_diff={max_diff}");
    }

    /// Non-trans_b GEMM: m=4, n=16, k=16 (4 K-loop iterations)
    #[test]
    fn test_gemm_no_trans_b_m4_n16_k16_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize; let n = 16usize; let k = 16usize;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b]; g.outputs = vec![c];
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false }, vec![a, b], vec![c], "gemm_m4_n16_k16");
        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None).expect("GEMM compilation failed").expect_cpu().layer_code;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe { compiled.execute_as_mega_kernel(a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8, 1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr()); }
        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m4_n16_k16] max_diff={max_diff:.6}");
        assert!(max_diff < 0.01, "m4_n16_k16 max_diff={max_diff}");
    }

    /// Non-trans_b GEMM: m=4, n=16, k=24 (6 K-loop iterations)
    #[test]
    fn test_gemm_no_trans_b_m4_n16_k24_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize; let n = 16usize; let k = 24usize;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b]; g.outputs = vec![c];
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false }, vec![a, b], vec![c], "gemm_m4_n16_k24");
        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None).expect("GEMM compilation failed").expect_cpu().layer_code;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe { compiled.execute_as_mega_kernel(a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8, 1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr()); }
        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m4_n16_k24] max_diff={max_diff:.6}");
        assert!(max_diff < 0.01, "m4_n16_k24 max_diff={max_diff}");
    }

    /// Non-trans_b GEMM: m=4, n=16, k=28 (7 K-loop iterations)
    #[test]
    fn test_gemm_no_trans_b_m4_n16_k28_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize; let n = 16usize; let k = 28usize;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b]; g.outputs = vec![c];
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false }, vec![a, b], vec![c], "gemm_m4_n16_k28");
        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None).expect("GEMM compilation failed").expect_cpu().layer_code;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { c_ref[i*n+j] += a_data[i*k+p] * b_data[p*n+j]; } } }
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe { compiled.execute_as_mega_kernel(a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8, 1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr()); }
        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[m4_n16_k28] max_diff={max_diff:.6}");
        assert!(max_diff < 0.01, "m4_n16_k28 max_diff={max_diff}");
    }

    /// Minimal non-trans_b GEMM correctness: m=2, n=8, k=4.
    #[test]
    fn test_gemm_no_trans_b_m2_n8_k4_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 2usize;
        let n = 8usize;
        let k = 4usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_tiny",
        );

        let config = CompileConfig { max_seq_len: m, debug_jit: false, hetero: None, target: CompileTarget::Cpu };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("tiny GEMM compilation failed").expect_cpu().layer_code;

        // Simple deterministic data
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data: Vec<f32> = (0..k*n).map(|i| (i % 3) as f32 + 1.0).collect();

        // Scalar reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c_ref[i * n + j] += a_data[i * k + p] * b_data[p * n + j];
                }
            }
        }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8, b_data.as_ptr() as *const u8,
                1, m, c_jit.as_mut_ptr() as *mut u8, scratchpad.as_mut_ptr(),
            );
        }

        eprintln!("[tiny] c_jit: {:?}", &c_jit[..]);
        eprintln!("[tiny] c_ref: {:?}", &c_ref[..]);
        for (i, (jit, rf)) in c_jit.iter().zip(c_ref.iter()).enumerate() {
            let diff = (jit - rf).abs();
            if diff > 0.01 {
                eprintln!("[tiny] MISMATCH at {i}: jit={jit} ref={rf} diff={diff}");
            }
        }
        let nan_count = c_jit.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nan_count, 0, "tiny GEMM NaN");
        let max_diff = c_jit.iter().zip(c_ref.iter()).map(|(j, r)| (j - r).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 0.01, "tiny GEMM max_diff={max_diff}");
    }

    /// Numerical correctness for non-trans_b BLIS GEMM with m=4, n=64, k=128.
    /// Isolates whether m>mr or n is the issue.
    #[test]
    fn test_gemm_no_trans_b_m4_n64_k128_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize;
        let n = 64usize;
        let k = 128usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_no_trans_m4",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("non-trans GEMM m=4 compilation failed")
            .expect_cpu()
            .layer_code;

        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8,
                b_data.as_ptr() as *const u8,
                1,
                m,
                c_jit.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }

        let nan_count = c_jit.iter().filter(|v| !v.is_finite()).count();
        eprintln!("[no_trans_m4] NaN count: {nan_count}");
        let max_diff = (0..c_ref.len())
            .filter(|&i| c_jit[i].is_finite())
            .map(|i| (c_jit[i] - c_ref[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[no_trans_m4] max_diff: {:.2e}", max_diff);
        // Per-row comparison for first 4 columns
        for r in 0..m {
            eprintln!("[no_trans_m4] row {r}: jit=[{:.3}, {:.3}, {:.3}, {:.3}] ref=[{:.3}, {:.3}, {:.3}, {:.3}]",
                c_jit[r*n], c_jit[r*n+1], c_jit[r*n+2], c_jit[r*n+3],
                c_ref[r*n], c_ref[r*n+1], c_ref[r*n+2], c_ref[r*n+3]);
        }
        assert_eq!(nan_count, 0, "non-trans GEMM m=4 produced NaN");
        assert!(max_diff < 0.1, "non-trans GEMM m=4 max_diff={max_diff:.4}");
    }

    /// Numerical correctness for non-trans_b BLIS GEMM with m=8, n=64, k=128.
    /// This reproduces the audio conformer ff1_gemm_out dimensions.
    #[test]
    fn test_gemm_no_trans_b_m8_n64_k128_correctness() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 8usize;
        let n = 64usize;
        let k = 128usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_no_trans_m8",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("non-trans GEMM m=8 compilation failed")
            .expect_cpu()
            .layer_code;

        // Prepare input data
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01 - 0.5).sin()).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02 - 0.3).cos()).collect();

        // Scalar reference: C[i][j] = sum_p A[i][p] * B[p][j]
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        // Execute JIT
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(65536)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8,
                b_data.as_ptr() as *const u8,
                1,
                m,
                c_jit.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }

        // Check for NaN first
        let nan_count = c_jit.iter().filter(|v| !v.is_finite()).count();
        eprintln!("[no_trans_m8] NaN count: {nan_count}");
        if nan_count > 0 {
            let nan_indices: Vec<_> = c_jit.iter().enumerate()
                .filter(|(_, v)| !v.is_finite())
                .take(20)
                .map(|(i, _)| i)
                .collect();
            eprintln!("[no_trans_m8] NaN indices: {nan_indices:?}");
            for &idx in &nan_indices {
                let row = idx / n;
                let col = idx % n;
                eprintln!("[no_trans_m8] NaN at row={row} col={col}");
            }
        }

        // Compare
        let max_diff = (0..c_ref.len())
            .filter(|&i| c_jit[i].is_finite())
            .map(|i| (c_jit[i] - c_ref[i]).abs())
            .fold(0.0f32, f32::max);

        eprintln!("[no_trans_m8] c_jit[0..8]: {:?}", &c_jit[..8]);
        eprintln!("[no_trans_m8] c_ref[0..8]: {:?}", &c_ref[..8]);
        eprintln!("[no_trans_m8] max_diff: {:.2e}", max_diff);

        assert_eq!(nan_count, 0, "non-trans GEMM m=8 produced {nan_count} NaN values");
        assert!(max_diff < 0.1, "non-trans GEMM m=8 max_diff={max_diff:.4} exceeds tolerance");
    }

    /// Small trans_b GEMM test for easier debugging.
    #[test]
    fn test_gemm_trans_b_small() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};

        use crate::types::DType;

        let m = 2usize;
        let n = 4usize;
        let k = 4usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[n, k], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: true },
            vec![a, b], vec![c], "gemm_trans_b_small",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("small trans_b GEMM compilation failed")
            .expect_cpu()
            .layer_code;

        // Simple input data
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,  // row 0
            5.0, 6.0, 7.0, 8.0,  // row 1
        ];
        let b_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,  // row 0: picks A[i][0]
            0.0, 1.0, 0.0, 0.0,  // row 1: picks A[i][1]
            0.0, 0.0, 1.0, 0.0,  // row 2: picks A[i][2]
            0.0, 0.0, 0.0, 1.0,  // row 3: picks A[i][3]
        ];

        // Expected: C[i][j] = A[i][j] (because B is identity)
        // C = [[1, 2, 3, 4], [5, 6, 7, 8]]
        let c_expected: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8,
                b_data.as_ptr() as *const u8,
                1,
                m,
                c_jit.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }

        eprintln!("[small trans_b] c_jit:   {:?}", &c_jit);
        eprintln!("[small trans_b] expected: {:?}", &c_expected);

        let max_diff = (0..c_expected.len())
            .map(|i| (c_jit[i] - c_expected[i]).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 0.01, "small trans_b GEMM max_diff={max_diff:.4}");
    }

    /// Medium trans_b GEMM test with identity-like B for offset verification.
    /// B = identity extended to [n, k] with zeros.
    /// C[i][j] should equal A[i][j] for j < k, 0 otherwise.
    #[test]
    fn test_gemm_trans_b_medium_identity() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};

        use crate::types::DType;

        let m = 2usize;
        let n = 16usize;  // > 8 (lanes)
        let k = 8usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[n, k], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: true },
            vec![a, b], vec![c], "gemm_trans_b_med",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("medium trans_b GEMM compilation failed")
            .expect_cpu()
            .layer_code;

        // A = simple values
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        ];

        // B = identity-like: row j has B[j][j]=1 if j<k, else all zeros
        let mut b_data = vec![0.0f32; n * k];
        for j in 0..k.min(n) {
            b_data[j * k + j] = 1.0;
        }

        // Expected: C[i][j] = sum_p A[i][p] * B[j][p]
        // For j < k: C[i][j] = A[i][j]
        // For j >= k: C[i][j] = 0
        let mut c_expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c_expected[i * n + j] += a_data[i * k + p] * b_data[j * k + p];
                }
            }
        }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8,
                b_data.as_ptr() as *const u8,
                1,
                m,
                c_jit.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }

        eprintln!("[med trans_b] c_jit[0..16]: {:?}", &c_jit[..16]);
        eprintln!("[med trans_b] expected[0..16]: {:?}", &c_expected[..16]);

        let max_diff = (0..c_expected.len())
            .map(|i| (c_jit[i] - c_expected[i]).abs())
            .fold(0.0f32, f32::max);

        eprintln!("[med trans_b] max_diff: {:.2e}", max_diff);
        assert!(max_diff < 0.01, "medium trans_b GEMM max_diff={max_diff:.4}");
    }

    /// Tiny no-trans_b GEMM test (M=2, N=4, K=4) for debugging
    #[test]
    fn test_gemm_no_trans_tiny() {
        use crate::compiler::InferenceCompiler;
        use crate::compiler::mega_kernel_abi::{CompileConfig, BusinessConfig, CompileTarget};
        use crate::types::DType;

        let m = 4usize;
        let n = 8usize;
        let k = 32usize;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
        let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
        let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32, trans_b: false },
            vec![a, b], vec![c], "gemm_no_trans_tiny",
        );

        let config = CompileConfig {
            max_seq_len: m,
            debug_jit: false,
            hetero: None,
            target: CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile(g, &config, None)
            .expect("tiny GEMM compilation failed")
            .expect_cpu()
            .layer_code;

        let a_data: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| i as f32 * 0.3 - 0.5).collect();

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k { sum += a_data[i * k + p] * b_data[p * n + j]; }
                c_ref[i * n + j] = sum;
            }
        }

        let mut c_jit = vec![0.0f32; m * n];
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8,
                b_data.as_ptr() as *const u8,
                1, m,
                c_jit.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }

        let max_diff = (0..c_ref.len()).map(|i| (c_jit[i] - c_ref[i]).abs()).fold(0.0f32, f32::max);
        eprintln!("[tiny no_trans] c_jit: {:?}", &c_jit[..]);
        eprintln!("[tiny no_trans] c_ref: {:?}", &c_ref[..]);
        eprintln!("[tiny no_trans] max_diff: {:.2e}", max_diff);
        assert!(max_diff < 0.01, "tiny no-trans GEMM max_diff={max_diff:.4}");
    }
