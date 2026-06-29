
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0.1 编译会话共享状态
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 编译会话级共享状态 — 持有 lowering 过程中不变的硬件/配置信息。
///
/// 从 LoweringContext 中提取出来的全局常量字段。
/// 注意：AbiPtrs 不在此处，因为它包含运行时分配的 VReg，
/// 需要在 VmProgram 分配 VReg 之后才能构造。
// @trace REQ-FATOP-008 [entity:CompileSession] CompileSession 提取 13 字段
pub struct CompileSession<'a> {
    /// SIMD 宽度（硬件全局）
    pub width: SimdWidth,
    /// SymDim slot 映射（编译会话级）
    pub sym_map: &'a SymDimSlotMap,
    /// 标量算子注册表
    pub registry: Option<&'a ScalarOpRegistry>,
    /// ISA hook
    pub hook: Option<&'a dyn super::isa_hook::IsaHook>,
    /// 硬件能力位集合 (CR-002: OpImpl.requires() 谓词子集匹配的源)。
    pub feature_set: super::op_impl::FeatureSet,
    /// 资源预算
    pub budget: Option<super::isa_hook::ResourceBudget>,
    /// PagedAttention page size
    pub page_size: usize,
    /// Hardware dot-product capability
    pub dot_cap: DotProductCap,
    /// KV cache element bytes
    pub kv_elem_bytes: usize,
    /// JIT debug instrumentation
    pub debug_jit: bool,
    /// 虚拟化映射
    pub virtual_activation: Option<&'a VirtualActivationMap>,
    pub virtual_tensor_map: Option<&'a VirtualTensorMap>,
    pub layout: Option<&'a crate::compiler::layout_negotiator::LayoutAssignment>,
    /// §20 BCI batch_ctx_ptr
    pub batch_ctx_ptr: Option<VRegId>,
}

/// Op-local 状态 — 持有每个 op 不同的局部状态 + 引用 CompileSession。
///
/// 替代 emit_fusion_groups/emit_standalone_op 中 16+ 参数的重复传递。
/// 局部参数（prog, op, input_ptr, weight_ptr, output_ptr 等）仍作为函数参数。
///
/// 通过 &'a CompileSession<'a> 引用会话级状态，保持单生命周期 'a。
pub struct LoweringContext<'a> {
    /// 编译会话级共享状态（全局常量）
    pub session: &'a CompileSession<'a>,
    /// 当前图/融合组的计算 dtype（op-local）
    pub dtype: QuantPrecision,
    /// §0.2.9 虚拟执行模式: 当前 op 的 ExecPattern (from R0 PainPointAnalyzer)
    pub exec_pattern: Option<ExecPattern>,
    /// §0.2.9 R0 PainPointAnalyzer 输出: per-GEMM 瓶颈分析 (含 ExecPattern)
    pub bottleneck_map: Option<&'a OpBottleneckMap>,
    /// RoPE/PLE/DWC scratchpad 需求（op-local，按 op 关联）
    pub rope_req: Option<&'a RopeCacheRequirement>,
    pub ple_req: Option<&'a PleScratchRequirement>,
    pub dwc_req: Option<&'a DwcScratchRequirement>,
    /// §0.2.10 虚拟并行: SIMD/warp 并行度描述 (from DeviceProfile)
    pub parallelism: Option<ParallelismDesc>,
}

impl<'a> LoweringContext<'a> {
    /// 查找指定 op 的 ExecPattern，优先从瓶颈映射获取，fallback 到 ctx.exec_pattern。
    pub fn exec_pattern_for_op(&self, op_id: crate::compiler::graph::OpId) -> Option<ExecPattern> {
        self.bottleneck_map
            .and_then(|bm| bm.gemm_bottlenecks.get(&op_id))
            .map(|bn| bn.exec_pattern)
            .or(self.exec_pattern)
    }

    /// §0.2.10: 查找 GEMM op 的 per-op ParallelismDesc (decode vs prefill 差异化)。
    pub fn parallelism_for_op(&self, op_id: crate::compiler::graph::OpId) -> Option<ParallelismDesc> {
        self.bottleneck_map
            .and_then(|bm| bm.gemm_bottlenecks.get(&op_id))
            .map(|bn| bn.parallelism)
            .or(self.parallelism)
    }

    /// §0.2.7: 查找 GEMM op 权重 (inputs[1]) 的 PackMap。
    pub fn pack_map_for_gemm(&self, weight_tid: Option<TensorId>) -> Option<&'a crate::compiler::pack_map::PackMap> {
        let tid = weight_tid?;
        self.session.virtual_tensor_map?.pack_maps.get(&tid)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Layer 6: Debug instrumentation helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub(super) fn maybe_debug_bp(prog: &mut VmProgram, ctx: &LoweringContext<'_>, label: &str) {
    if ctx.session.debug_jit {
        prog.emit(VmInstr::DebugBreakpoint { label: label.to_string() });
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0.0 SymDim 解析辅助
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Resolve SymDim to BoundExpr, using mega-kernel decode seq_len VReg when available.
/// Shared by `emit_standalone_op` and `lower_op`.
pub(crate) fn resolve_sym_dim(dim: &SymDim, abi: &AbiPtrs, sym_map: &SymDimSlotMap) -> BoundExpr {
    if let Some(seq_vreg) = abi.mega_decode_seq_len {
        if dim.is_symbolic() {
            return BoundExpr::DynamicVReg(seq_vreg);
        }
    }
    sym_map.to_bound(dim)
}

/// 从图推断激活**计算** dtype（REQ-DTYPE-010, CR-DTYPE-SOVEREIGNTY-001）。
///
/// BCE-20260629-003 第六层: 区分**存储 dtype** 与**计算 dtype**。
/// 激活 tensor 标的是存储 dtype（BF16），但 JIT 计算精度应该是 F32
/// （REQ-DTYPE-CHAIN-005: BF16 存储 + F32 计算/累加）。
/// 旧实现取首个浮点 tensor 的存储 dtype(BF16) 当 compute dtype → 与
/// MegaKernelAbi.compute_dtype=F32 不同步 → harness 读 NaN。
///
/// 正确: graph_dtype 返回**计算精度** F32（激活累加精度），存储 dtype 由各
/// OpImpl 的 weight_dtype 参数单独传播。BF16/F16 权重在 VecLoad WidenCompute
/// 时自动 widen 到 F32 寄存器计算。
pub(super) fn graph_dtype(graph: &CompilerGraph) -> QuantPrecision {
    use crate::compiler::trace::DTypeKind;
    // 取第一个浮点 tensor 确认有浮点数据（跳过索引类 I32/U8）
    let has_float = graph.tensors.iter()
        .map(|t| t.dtype.to_quant_precision())
        .any(|qp| matches!(qp.kind, DTypeKind::F32 | DTypeKind::BF16 | DTypeKind::F16 | DTypeKind::TF32));
    if !has_float {
        return QuantPrecision::F32;
    }
    // 计算精度统一 F32（激活累加）。存储 dtype(BF16/F16) 由 VecLoad WidenCompute
    // 在寄存器内 widen 到 F32，不在 graph_dtype 层混入存储精度。
    // 这保证 JIT ctx.dtype(F32) == MegaKernelAbi.compute_dtype(F32) 同步。
    QuantPrecision::F32
}

/// 从 op 的第一个输入 tensor 推断计算 dtype (REQ-DTYPE-001)。
///
/// dtype 传播链: TensorMeta.dtype → op_input_dtype() → ctx.dtype → emit_*(ctx) → VmInstr{dtype}
///
/// 推断方向严格单向：从 op.inputs[0].dtype 推导，禁止反向推断。
/// 无输入 tensor 时安全回退到 F32 (SPEC §0.8 REQ-DTYPE-001 唯一合法的 F32 默认)。
pub(crate) fn op_input_dtype(
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
) -> QuantPrecision {
    op.inputs.first()
        .and_then(|&tid| graph.tensor(tid))
        .map(|t| t.dtype.to_quant_precision())
        .unwrap_or(QuantPrecision::F32)
}

/// Derive KV cache element bytes from graph weight tensors (majority vote).
/// Must match GraphDerivedGeometry::derive_storage_dtype which the executor
/// uses to allocate the KV cache buffer. Returns the elem_bytes (2 for BF16,
/// 4 for F32) that the JIT should use for KV cache row stride computation.
pub(super) fn kv_cache_elem_bytes(graph: &CompilerGraph) -> usize {
    let mut f32_count = 0usize;
    let mut bf16_count = 0usize;
    let mut f16_count = 0usize;
    for &tid in graph.inputs.iter().skip(1) {
        if let Some(t) = graph.tensors.get(tid.0 as usize) {
            match t.dtype {
                crate::types::DType::F32 => f32_count += 1,
                crate::types::DType::BF16 => bf16_count += 1,
                crate::types::DType::F16 => f16_count += 1,
                _ => {}
            }
        }
    }
    if bf16_count >= f32_count && bf16_count >= f16_count && bf16_count > 0 {
        2 // BF16
    } else if f16_count >= f32_count && f16_count >= bf16_count && f16_count > 0 {
        2 // F16
    } else {
        4 // F32
    }
}

/// 从 op 的第一个输入 tensor 获取计算 dtype 的 elem_bytes。
/// dtype 传播链: TensorMeta.dtype → QuantPrecision → elem_bytes (SPEC 00-PHILOSOPHY §4)。
/// 如果 op 没有输入 tensor（罕见），默认返回 F32 的 4 字节。
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0.1 TensorPtrResolver (ARCH-DATA-FLOW-CONTRACT §3 统一解析器)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 统一解析 CompilerGraph 每个 tensor 的运行时 base ptr 分类 + byte offset。
///
/// R3 输出: tensor_sources 已在 BufferAllocation 中预计算，
/// TensorPtrResolver 提供额外的 override 和 materialize 能力。
pub struct TensorPtrResolver {
    map: HashMap<TensorId, TensorPtrSource>,
}

impl TensorPtrResolver {
    /// 从 BufferAllocation 的预计算 tensor_sources 构建。
    /// 当 R3 未预计算 tensor_sources（简单 allocate_buffers 路径）时 fallback 到自行构建。
    pub fn build(graph: &CompilerGraph, alloc: &BufferAllocation, topology: &super::topology::GraphTopologyAnalysis) -> Self {
        let map = if !alloc.tensor_sources.is_empty() {
            let mut m = alloc.tensor_sources.clone();
            // BCE-20260629-005: 强制 Gather/QuantGather 输出为 Intermediate{offset}
            // 从 alloc.slots 查找真实 offset，避免和 ping buffer 冲突。
            eprintln!("[RESOLVER] using alloc.tensor_sources ({} entries)", m.len());
            for op in &graph.ops {
                if let crate::compiler::graph::Op::Gather { .. } | crate::compiler::graph::Op::QuantGather { .. } = &op.op {
                    if let Some(&out_tid) = op.outputs.first() {
                        let off = alloc.offset_of(out_tid).unwrap_or(0);
                        eprintln!("[RESOLVER] Gather output tid={} → Intermediate{{offset={}}}", out_tid.0, off);
                        m.insert(out_tid, TensorPtrSource::Intermediate { offset: off });
                    }
                }
            }
            m
        } else {
            build_tensor_sources_fallback(graph, alloc, topology)
        };

        // Diagnostic: dump key tensor mappings for debugging
        if std::env::var("GLLM_DEBUG_RESOURCE").is_ok() {
            // Dump ALL tensors that map to ActivationPing/ActivationPong/Intermediate
            let mut ping_count = 0;
            let mut pong_count = 0;
            let mut inter_count = 0;
            let mut activation_count = 0;
            for meta in &graph.tensors {
                if let Some(src) = map.get(&meta.id) {
                    match src {
                        TensorPtrSource::ActivationPing => {
                            ping_count += 1;
                            eprintln!("[resolver] PING  {:40} tid={}", meta.name, meta.id.0);
                        }
                        TensorPtrSource::ActivationPong => {
                            pong_count += 1;
                            eprintln!("[resolver] PONG  {:40} tid={}", meta.name, meta.id.0);
                        }
                        TensorPtrSource::Intermediate { offset: io } => {
                            eprintln!("[resolver] INTER {:40} tid={} offset={}", meta.name, meta.id.0, io);
                            inter_count += 1;
                        }
                        TensorPtrSource::Activation => { activation_count += 1; }
                        TensorPtrSource::Output { offset: oo } => {
                            eprintln!("[resolver] OUTPUT {:40} tid={} offset={}", meta.name, meta.id.0, oo);
                        }
                        TensorPtrSource::Weight { offset: wo } => {
                            if *wo > 0 {
                                eprintln!("[resolver] WEIGHT {:40} tid={} offset={}", meta.name, meta.id.0, wo);
                            }
                        }
                    }
                }
            }
            eprintln!("[resolver] SUMMARY: ping={} pong={} intermediate={} activation={}",
                ping_count, pong_count, inter_count, activation_count);
            // Dump activation_alias pairs
            // SPEC/39: activation_alias 从 topology 推导，替代 graph.layer_loop_config 读取
            if let Some((in_tid, out_tid)) = &topology.layer_activation_alias {
                eprintln!("[resolver] HOMO activation_alias: input_tid={} output_tid={}", in_tid.0, out_tid.0);
            }
            // HETERO: hetero_layer_loop_config.activation_aliases 仍保留在 graph 上
            // (hetero 拓扑尚未迁移到 GraphTopologyAnalysis)
            if let Some(cfg) = graph.hetero_layer_loop_config.as_ref() {
                for (i, (in_tid, out_tid)) in cfg.activation_aliases.iter().enumerate() {
                    let in_name = graph.tensors.iter().find(|t| t.id == *in_tid).map(|t| t.name.as_str()).unwrap_or("?");
                    let out_name = graph.tensors.iter().find(|t| t.id == *out_tid).map(|t| t.name.as_str()).unwrap_or("?");
                    eprintln!("[resolver] HETERO activation_alias[{}]: input={} ({}) output={} ({})", i, in_tid.0, in_name, out_tid.0, out_name);
                }
            }
            eprintln!("[resolver] graph.inputs: {} tensors, weight_layout: {} offsets",
                graph.inputs.len(), graph.weight_layout().offsets.len());
        }

        Self { map }
    }

    pub fn source(&self, tid: TensorId) -> Option<TensorPtrSource> {
        self.map.get(&tid).copied()
    }

    /// Override a tensor's source mapping (mega-kernel: redirect logits to output_ptr).
    pub fn override_source(&mut self, tid: TensorId, source: TensorPtrSource) {
        self.map.insert(tid, source);
    }

    /// 为 tensor 生成指向其数据起点的 Ptr VReg。
    pub fn materialize(
        &self,
        prog: &mut VmProgram,
        tid: TensorId,
        abi: &AbiPtrs,
    ) -> Option<VRegId> {
        let src = self.map.get(&tid)?;
        match src {
            TensorPtrSource::Activation => Some(abi.input_ptr),
            TensorPtrSource::ActivationPing => Some(abi.activation_ping_ptr?),
            TensorPtrSource::ActivationPong => Some(abi.activation_pong_ptr?),
            TensorPtrSource::Weight { offset } => {
                let base = abi.weight_ptr?;
                if *offset == 0 {
                    Some(base)
                } else {
                    let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr, src: PtrExpr::VRegPlusConst(base, *offset) });
                    Some(ptr)
                }
            }
            TensorPtrSource::Intermediate { offset } => {
                // ARCH-SPILL-SAFE: Root cause fixed in ScopedSpillAllocator —
                // each VReg now gets a unique spill offset, preventing corruption.
                // No longer need to reload scratchpad base from StackArg(24) here.
                let base = abi.scratch_ptr?;
                if *offset == 0 {
                    Some(base)
                } else {
                    let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr, src: PtrExpr::VRegPlusConst(base, *offset) });
                    Some(ptr)
                }
            }
            TensorPtrSource::Output { offset } => {
                let base = abi.output_ptr;
                if *offset == 0 {
                    Some(base)
                } else {
                    let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr, src: PtrExpr::VRegPlusConst(base, *offset) });
                    Some(ptr)
                }
            }
        }
    }
}

/// Fallback: 当 R3 未预计算 tensor_sources 时，在此处构建。
/// 正式路径通过 allocate_buffers_aligned → build_tensor_sources 完成。
fn build_tensor_sources_fallback(
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    topology: &super::topology::GraphTopologyAnalysis,
) -> HashMap<TensorId, TensorPtrSource> {
    let dtype = graph_dtype(graph);
    let mut map: HashMap<TensorId, TensorPtrSource> = HashMap::new();

    for (i, &tid) in graph.inputs.iter().enumerate() {
        if i == 0 {
            map.insert(tid, TensorPtrSource::Activation);
        } else {
            map.insert(tid, TensorPtrSource::Weight { offset: 0 });
        }
    }
    let wl = graph.weight_layout();
    for &(tid, off) in &wl.offsets {
        map.insert(tid, TensorPtrSource::Weight { offset: off });
    }

    {
        let elem = dtype.elem_bytes();
        let mut cursor = 0usize;
        for &tid in &graph.outputs {
            let numel = graph.tensor_numel_for_alloc(tid, graph.max_seq_len).unwrap_or(0);
            map.insert(tid, TensorPtrSource::Output { offset: cursor });
            cursor += numel * elem;
        }
    }

    for slot in &alloc.slots {
        map.entry(slot.tensor_id).or_insert(TensorPtrSource::Intermediate { offset: slot.offset });
    }

    for op in &graph.ops {
        if let Some(input_idx) = op.op_output_aliases_input(graph) {
            if let (Some(&in_tid), Some(&out_tid)) =
                (op.inputs.get(input_idx), op.outputs.first())
            {
                if let Some(&src) = map.get(&in_tid) {
                    map.insert(out_tid, src);
                }
            }
        }
    }

    // Ping-pong 仅当 VAM 分配了 sentinel slots 时启用。
    // Per-layer 编译路径没有 VAM 分析 → 无 sentinel slots → 保持原始 Activation 映射。
    let has_ping_pong_slots = alloc.slots.iter().any(|s| s.tensor_id.0 == 0xFFFF_FF00);
    if has_ping_pong_slots {
        // BCE-20260629-005: 跳过 Gather/QuantGather 输出
        let gather_outs: std::collections::HashSet<TensorId> = graph.ops.iter()
            .filter_map(|op| match &op.op {
                crate::compiler::graph::Op::Gather { .. } | crate::compiler::graph::Op::QuantGather { .. } => op.outputs.first().copied(),
                _ => None,
            })
            .collect();
        if let Some((ref input_tid, ref output_tid)) = topology.layer_activation_alias {
            if !gather_outs.contains(input_tid) {
                map.entry(*input_tid).or_insert(TensorPtrSource::ActivationPing);
            }
            if !gather_outs.contains(output_tid) {
                map.entry(*output_tid).or_insert(TensorPtrSource::ActivationPong);
            }
        }
        // HETERO
        if let Some(ref cfg) = graph.hetero_layer_loop_config {
            for (input_tid, output_tid) in &cfg.activation_aliases {
                if !gather_outs.contains(input_tid) {
                    map.entry(*input_tid).or_insert(TensorPtrSource::ActivationPing);
                }
                if !gather_outs.contains(output_tid) {
                    map.entry(*output_tid).or_insert(TensorPtrSource::ActivationPong);
                }
            }
        }
    } else {
        // Forward-only path (compile_graph): alias output inherits activation input's source
        // so that post-loop ops (e.g. MeanPool) can read the final layer's output.
        // SPEC/39: activation_alias 从 topology 推导，替代 graph.layer_loop_config 读取
        if let Some((ref input_tid, ref output_tid)) = topology.layer_activation_alias {
            if let Some(src) = map.get(input_tid).copied() {
                map.insert(*output_tid, src);
            }
        }
        // HETERO: hetero_layer_loop_config.activation_aliases 仍保留在 graph 上
        if let Some(ref cfg) = graph.hetero_layer_loop_config {
            for (input_tid, output_tid) in &cfg.activation_aliases {
                if let Some(src) = map.get(input_tid).copied() {
                    map.insert(*output_tid, src);
                }
            }
        }
    }

    map
}

/// 扫描 plan 中所有 RoPE 算子,推导 cos/sin 表需求 (ARCH-ROPE-CACHE)。
///
/// 支持异构模型（如 Gemma-4 E2B）中多种 head_dim 的 RoPE 参数。
/// 最多支持 2 种不同的 (head_dim, theta, partial) 组合（primary + secondary cache）。
pub(crate) fn compute_rope_requirement(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
) -> Result<Option<RopeCacheRequirement>, CompilerError> {
    use crate::compiler::graph::RopeScaling;
    // Collect unique (head_dim, theta, partial, rope_scaling) sets.
    // (head_dim, theta, partial, max_seq, rope_scaling)
    let mut primary: Option<(usize, f64, f32, usize, Option<RopeScaling>)> = None;
    let mut secondary: Option<(usize, f64, f32, usize, Option<RopeScaling>)> = None;

    for group in &plan.groups {
        for &op_id in &group.ops {
            let Some(op) = graph.op(op_id) else { continue };
            // 从 Op（胖 opcode）读取 RoPE 参数 — 从 Op 读取。
            let op_resolved = &op.op;
            // Collect RoPE params from both standard RoPE and DualRoPE ops.
            // DualRoPE contributes two entries (sliding + global) simultaneously.
            let rope_params: Vec<(usize, f64, f32, Option<RopeScaling>)> = match op_resolved {
                crate::compiler::graph::Op::RoPE(spec) => {
                    vec![(spec.head_dim, spec.theta, spec.partial, spec.rope_scaling.clone())]
                }
                crate::compiler::graph::Op::DualRoPE(spec) => {
                    vec![
                        (spec.head_dim, spec.sliding_theta, spec.sliding_partial, spec.rope_scaling.clone()),
                        (spec.head_dim, spec.global_theta, spec.global_partial, spec.rope_scaling.clone()),
                    ]
                }
                _ => continue,
            };
            for (head_dim, theta, partial, rope_scaling) in rope_params {
                let out_tensor = op.outputs.first()
                    .and_then(|&tid| graph.tensor(tid))
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "RoPE op {:?}: 无输出张量", op_id)))?;
                let max_seq = out_tensor.shape.iter().find_map(|d| match d {
                    SymDim::Symbolic { max_value, .. } => *max_value,
                    _ => None,
                }).or_else(|| out_tensor.shape.first().and_then(|d| d.as_concrete()))
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "RoPE op {:?}: 无法推导 seq 维度 (shape={:?})", op_id, out_tensor.shape)))?;

                let entry = (head_dim, theta, partial, max_seq, rope_scaling);

                let matches = |e: &(usize, f64, f32, usize, Option<RopeScaling>)| -> bool {
                    let (h, t, p, _, s) = e;
                    *h == head_dim && (t - theta).abs() < 1e-6 && (p - partial).abs() < 1e-6
                        && s.as_ref().map(|s| s.fingerprint_bytes()) == rope_scaling.as_ref().map(|s| s.fingerprint_bytes())
                };

                if primary.as_ref().is_none_or(&matches) {
                    if let Some(ref mut p) = primary {
                        p.3 = p.3.max(max_seq);
                    } else {
                        primary = Some(entry);
                    }
                } else if secondary.as_ref().is_none_or(matches) {
                    if let Some(ref mut s) = secondary {
                        s.3 = s.3.max(max_seq);
                    } else {
                        secondary = Some(entry);
                    }
                } else {
                    return Err(CompilerError::CodegenViolation("ARCH-ROPE-CACHE: plan 内发现 3+ 种不同 RoPE 参数组合,最多支持 2 种".to_string()));
                }
            }
        }
    }
    let Some((head_dim, theta, partial, max_seq_len, rope_scaling)) = primary else {
        return Ok(None);
    };
    let cache_offset = (alloc.total_bytes + 63) & !63;
    let attention_scaling = crate::compiler::rope_scaling::compute_attention_scaling(rope_scaling);

    // Secondary cache allocated right after primary cache (64B aligned).
    let secondary_cache = secondary.map(|(hd, sec_theta, sec_partial, ms, sec_scaling)| {
        let primary_bytes = max_seq_len * head_dim * 4; // f32
        let sec_offset = ((cache_offset + primary_bytes) + 63) & !63;
        super::super::SecondaryRopeCache {
            head_dim: hd,
            cache_offset: sec_offset,
            theta: sec_theta,
            partial: sec_partial,
            rope_scaling: sec_scaling,
        }
    });

    Ok(Some(RopeCacheRequirement {
        cache_offset,
        head_dim,
        theta,
        partial,
        max_seq_len,
        rope_scaling,
        attention_scaling,
        secondary_cache,
    }))
}

/// PerLayerEmbed scratch 布局计算 (已废弃: PerLayerEmbed 已迁移为 AltUp 3-op 拆分)。
/// 始终返回 None。保留函数签名以维持 LoweringContext 接口兼容。
pub(crate) fn compute_ple_requirement(
    _plan: &FusionPlan,
    _graph: &CompilerGraph,
    _alloc: &BufferAllocation,
    _rope_req: Option<&RopeCacheRequirement>,
) -> Result<Option<PleScratchRequirement>, CompilerError> {
    Ok(None)
}

/// 扫描 plan 中所有 DepthwiseConv1D 算子,推导 padded input buffer 在 scratchpad 中的布局。
///
/// 返回 None 当图中无 DWC 算子。多个 DWC 算子要求 (channels, kernel_size, causal) 一致,
/// `max_seq_len` 取最大值;签名不一致直接 Err。
///
/// Padding 策略 (scalar_depthwise_conv1d 语义):
/// - causal=true  → 左 pad `kernel_size - 1`, 右 pad 0
/// - causal=false → 对称 SAME pad, 前后各 `(kernel_size - 1) / 2`
///
/// 布局顺序: BufferAllocation → RoPE cache → PLE scratch → DWC padded buffer。
pub(crate) fn compute_dwc_requirement(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    rope_req: Option<&RopeCacheRequirement>,
    ple_req: Option<&PleScratchRequirement>,
) -> Result<Option<DwcScratchRequirement>, CompilerError> {
    let dtype = graph_dtype(graph);
    let elem = dtype.elem_bytes();
    let mut common: Option<(usize, usize, usize, bool)> = None; // (max_seq, channels, kernel_size, causal)
    for group in &plan.groups {
        for &op_id in &group.ops {
            let Some(op) = graph.op(op_id) else { continue };
            // 从 Op（胖 opcode）读取 DepthwiseConv1D 参数 — 从 Op 读取。
            let op_resolved = &op.op;
            if let crate::compiler::graph::Op::DepthwiseConv1D { channels, kernel_size, causal } = op_resolved {
                if *kernel_size == 0 {
                    return Err(CompilerError::CodegenViolation(
                        "DepthwiseConv1D: kernel_size=0 非法".into()));
                }
                if *channels == 0 {
                    return Err(CompilerError::CodegenViolation(
                        "DepthwiseConv1D: channels=0 非法".into()));
                }
                // non-causal 只接受奇数 kernel_size (scalar ref 语义约束)
                if !*causal && kernel_size % 2 == 0 {
                    return Err(CompilerError::CodegenViolation(format!(
                        "DepthwiseConv1D non-causal 要求奇数 kernel_size, 实际 {}", kernel_size,
                    )));
                }
                // 从输出张量首维推导 seq_len 上界 (Symbolic max_value 或 Concrete 值)
                let out_tensor = op.outputs.first()
                    .and_then(|&tid| graph.tensor(tid))
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "DepthwiseConv1D op {:?}: 无输出张量", op_id)))?;
                let max_seq = out_tensor.shape.iter().find_map(|d| match d {
                    SymDim::Symbolic { max_value, .. } => *max_value,
                    _ => None,
                }).or_else(|| out_tensor.shape.first().and_then(|d| d.as_concrete()))
                    .ok_or_else(|| CompilerError::CodegenViolation(format!(
                        "DepthwiseConv1D op {:?}: 无法推导 seq 维度 (shape={:?})",
                        op_id, out_tensor.shape)))?;
                match common {
                    None => common = Some((max_seq, *channels, *kernel_size, *causal)),
                    Some((m0, c0, k0, causal0)) => {
                        if c0 != *channels || k0 != *kernel_size || causal0 != *causal {
                            return Err(CompilerError::CodegenViolation(format!(
                                "DepthwiseConv1D plan 内签名不一致: \
                                 ({c0}, {k0}, {causal0}) vs ({}, {}, {})",
                                channels, kernel_size, causal,
                            )));
                        }
                        common = Some((m0.max(max_seq), c0, k0, causal0));
                    }
                }
            }
        }
    }
    let Some((max_seq_len, channels, kernel_size, causal)) = common else { return Ok(None); };

    let left_pad = if causal { kernel_size - 1 } else { (kernel_size - 1) / 2 };
    let right_pad = if causal { 0 } else { (kernel_size - 1) / 2 };
    let total_pad = left_pad + right_pad;
    let padded_rows = max_seq_len + total_pad;
    let padded_bytes = padded_rows * channels * elem;

    // 基址: 先于 DWC 的最后一个块之后 (PLE > RoPE > alloc), 64B 对齐。
    let base = if let Some(req) = ple_req {
        req.post_mlp_offset + req.max_seq_len * req.hidden * elem
    } else if let Some(r) = rope_req {
        r.cache_offset + r.max_seq_len * r.head_dim * elem
    } else {
        alloc.total_bytes
    };
    let padded_offset = (base + 63) & !63;
    let total_bytes = (padded_bytes + 63) & !63;

    Ok(Some(DwcScratchRequirement {
        padded_offset,
        total_bytes,
        max_seq_len,
        channels,
        kernel_size,
        causal,
        left_pad,
    }))
}

/// 扫描 plan 中所有 MoEDispatchPacked 算子，计算 scratchpad 大小。
/// Layout:
///   [top_k weights + top_k indices] = 2 * top_k * sizeof(f32)
///   [num_experts logits]             = num_experts * sizeof(f32)
///   [gate_up_buf]                    = 2 * intermediate_size * sizeof(f32)
///   [activ_buf]                      = intermediate_size * sizeof(f32)
/// 总计 = (2*top_k + num_experts + 3*intermediate_size) * sizeof(f32) 字节。
fn compute_moe_packed_requirement(
    plan: &FusionPlan,
    graph: &CompilerGraph,
) -> usize {
    let dtype = graph_dtype(graph);
    let elem = dtype.elem_bytes();
    for group in &plan.groups {
        for &op_id in &group.ops {
            let Some(op) = graph.op(op_id) else { continue };
            // 从 Op（胖 opcode）读取 MoEDispatchPacked 参数 — 从 Op 读取。
            let op_resolved = &op.op;
            if let crate::compiler::graph::Op::MoEDispatchPacked { num_experts, top_k, intermediate_size, .. } = op_resolved {
                let routing_overhead = (2 * top_k + num_experts) * elem;
                let compute_buffers = 3 * intermediate_size * elem;
                return routing_overhead + compute_buffers;
            }
        }
    }
    0
}

use std::collections::HashMap;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0 SymDimSlotMap (ARCH-SYMDIM-THREADING §3.3)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ABI 参数和符号维度的统一映射。
///
/// **权威来源**: `vm_state::VmState` (ARCH-VM-STATE-TRACKING)。
/// 本结构从 VmState 构建，传递给所有 lower 函数。
///
/// 三类映射:
/// 1. 固定参数 (input/weights/output/scratchpad) → VmState 查询 → PtrExpr
/// 2. 符号维度 (seq_len/batch_size) → VmState 查询 → PtrExpr (运行时绑定)
/// 3. SymDim → BoundExpr (Concrete→Const, Symbolic→Symbolic)
#[derive(Clone)]
pub struct SymDimSlotMap {
    slots: HashMap<String, PtrExpr>,
}

impl SymDimSlotMap {
    /// 从 VmState 构建——所有位置由 VM 状态机计算，零硬编码。
    /// (ARCH-VM-STATE-TRACKING §4.1)
    ///
    /// `param_names` 指定哪些参数需要查询（CPU ABI 10 参数 / GPU ABI 5 参数）。
    pub fn from_vm_state_with_params(
        state: &super::vm_state::VmState,
        param_names: &[&str],
    ) -> Result<Self, CompilerError> {
        let mut slots = HashMap::new();
        for name in param_names {
            slots.insert((*name).into(), state.arg_ptr_expr(name)?);
        }
        // 别名（仅当目标参数存在时）
        for (alias, target) in super::vm_state::VmState::sym_dim_aliases() {
            if param_names.contains(&target) {
                slots.insert(alias.into(), state.arg_ptr_expr(target)?);
            }
        }
        Ok(Self { slots })
    }

    /// 从 CPU (x86/AArch64) VmState 构建完整 SymDimSlotMap。
    pub fn from_vm_state(state: &super::vm_state::VmState) -> Result<Self, CompilerError> {
        Self::from_vm_state_with_params(state, &[
            "input", "weights", "kv_cache", "positions", "seq_lens",
            "output", "scratchpad", "telemetry", "seq_len", "batch_size",
        ])
    }

    /// GPU kernel ABI SymDimSlotMap（6 参数）。
    /// ARCH-GPU-ABI: input/weights/output/seq_len/telemetry 为 `.param` 入参;
    /// scratchpad 为片上共享内存 (ARCH-GPU-SHARED-SCRATCH),通过符号 `smem` 访问。
    pub fn gpu_abi() -> Self {
        let state = super::vm_state::VmState::init_gpu_kernel();
        Self::from_vm_state_with_params(&state, &[
            "input", "weights", "output", "seq_len", "telemetry", "scratchpad",
        ]).expect("GPU ABI init failed")
    }

    /// MegaKernelFn ABI SymDimSlotMap。
    ///
    /// 将 CompiledLayerFn 语义名映射到 MegaKernelFn 物理位置:
    /// - 寄存器参数 (input/weights/kv_cache/positions/seq_lens/batch_size) 完全相同
    /// - scratchpad → MegaKernelFn 的 scratchpad_ptr (StackArg(24))
    /// - output → scratchpad + logits_offset (需要 VmInstr 计算,此处不映射)
    /// - seq_len → 不映射 (mega-kernel decode 时为常量 1)
    /// - telemetry → MegaKernelFn 的 telemetry_ptr (StackArg(88))
    pub fn mega_kernel_abi() -> Self {
        let state = super::vm_state::VmState::init_mega_kernel_x86();
        let mut slots = HashMap::new();

        // 寄存器参数 — CompiledLayerFn 和 MegaKernelFn 的前 6 个参数物理位置相同
        slots.insert("input".into(), state.arg_ptr_expr("input_ids_ptr").unwrap());
        slots.insert("weights".into(), state.arg_ptr_expr("weight_blob_ptr").unwrap());
        slots.insert("kv_cache".into(), state.arg_ptr_expr("kv_cache_ptr").unwrap());
        slots.insert("positions".into(), state.arg_ptr_expr("positions_ptr").unwrap());
        slots.insert("seq_lens".into(), state.arg_ptr_expr("aux_ptr").unwrap());
        slots.insert("batch_size".into(), state.arg_ptr_expr("batch_size").unwrap());

        // scratchpad — MegaKernelFn 的 scratchpad_ptr 在 StackArg(24)
        slots.insert("scratchpad".into(), state.arg_ptr_expr("scratchpad_ptr").unwrap());

        // output — MegaKernelFn 的 output_tokens_ptr (arg 8).
        // For simple graphs compiled via compile_layer_with_sym_map, the output
        // buffer is passed via this ABI parameter.
        slots.insert("output".into(), state.arg_ptr_expr("output_tokens_ptr").unwrap());

        // telemetry — MegaKernelFn 的 telemetry_ptr 在 StackArg(96)
        slots.insert("telemetry".into(), state.arg_ptr_expr("telemetry_ptr").unwrap());

        // ABI stack args — resolved from MEGA_KERNEL_PARAMS via VmState.
        // Single source of truth: arg positions defined in mega_kernel_abi.rs.
        // Fall back to known offsets for ABIs without these params (small graphs).
        slots.insert("seq_len".into(),
            state.arg_ptr_expr("prompt_len").unwrap_or(PtrExpr::StackArg(16)));
        slots.insert("hook_ctx_ptr".into(),
            state.arg_ptr_expr("hook_ctx_ptr").unwrap_or(PtrExpr::StackArg(88)));
        slots.insert("callback_table_ptr".into(),
            state.arg_ptr_expr("callback_table_ptr").unwrap_or(PtrExpr::StackArg(128)));

        // Remaining ABI args from MEGA_KERNEL_PARAMS — added symbolically so
        // lower_op arms can use sym_map.resolve("name") instead of
        // hardcoded StackArg(N).  Eliminates fragile manual offset calculation.
        for name in ["output_tokens_ptr", "max_new_tokens", "eos_token_id", "prompt_len", "page_table_ptr"] {
            if let Ok(expr) = state.arg_ptr_expr(name) {
                slots.insert(name.into(), expr);
            }
        }

        // 别名
        for (alias, target) in super::vm_state::VmState::sym_dim_aliases() {
            if slots.contains_key(target) {
                slots.insert(alias.into(), slots[target].clone());
            }
        }

        Self { slots }
    }

    /// SymDim → BoundExpr（ARCH-SYMDIM-NO-UNWRAP）。
    pub fn to_bound(&self, dim: &SymDim) -> BoundExpr {
        match dim {
            SymDim::Concrete(v) => BoundExpr::Const(*v),
            SymDim::Symbolic { name, max_value } => {
                BoundExpr::Symbolic(SymBound {
                    name: name.clone(),
                    max_alloc: max_value.expect("ARCH-SYMDIM: Symbolic dim must have max_value for buffer allocation"),
                })
            }
        }
    }

    /// 查找参数/维度名的物理位置。
    pub fn resolve(&self, name: &str) -> Option<&PtrExpr> {
        self.slots.get(name)
    }

    pub fn resolve_all_keys(&self) -> Vec<&String> {
        self.slots.keys().collect()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0 内部张量地址辅助 (ARCH-DATA-FLOW-CONTRACT §3.1)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 为 op 的首个输出张量加载指针。
///
/// Resolver override 优先于 BufferAllocation scratch 偏移。
/// Mega-kernel output redirect (Output { offset: 0 }) overrides scratch offsets
/// so that the final output tensor writes to the caller-provided output buffer
/// instead of scratchpad (which the caller never reads from).
pub(super) fn load_op_scratch_ptr(
    prog: &mut VmProgram,
    scratch_base: VRegId,
    op: &crate::compiler::graph::CompilerOp,
    alloc: &BufferAllocation,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
) -> Result<VRegId, CompilerError> {
    let out_tid = op.outputs.first().copied().ok_or_else(|| CompilerError::CodegenViolation(
        format!("op {:?} 无输出张量", op.id)))?;
    // Resolver override takes priority: if the tensor was redirected to Output,
    // Weight, or Activation (not Intermediate), use the resolver's answer.
    // This ensures mega-kernel output redirect works correctly — the final
    // output tensor must write to abi.output_ptr (caller-provided buffer),
    // not scratchpad (which the caller never reads).
    if let Some(src) = resolver.source(out_tid) {
        match src {
            TensorPtrSource::Output { .. } | TensorPtrSource::Activation
            | TensorPtrSource::ActivationPing | TensorPtrSource::ActivationPong
            | TensorPtrSource::Weight { .. } => {
                if let Some(ptr) = resolver.materialize(prog, out_tid, abi) {
                    return Ok(ptr);
                }
            }
            TensorPtrSource::Intermediate { .. } => {
                // Intermediate tensors still use scratchpad offsets
            }
        }
    }
    if let Some(offset) = alloc.offset_of(out_tid) {
        // ARCH-SPILL-SAFE: Root cause fixed in ScopedSpillAllocator —
        // each VReg now gets a unique spill offset, preventing corruption.
        // No longer need to reload scratchpad base from StackArg(24) here.
        let base = abi.scratch_ptr
            .ok_or_else(|| CompilerError::CodegenViolation(
                "load_op_scratch_ptr: scratch_ptr not available".into()))?;
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: ptr,
            src: PtrExpr::VRegPlusConst(base, offset),
        });
        Ok(ptr)
    } else if let Some(ptr) = resolver.materialize(prog, out_tid, abi) {
        Ok(ptr)
    } else {
        Err(CompilerError::CodegenViolation(
            format!("无法为 op {:?} 的输出 tensor {:?} 分配指针（非 scratchpad 也非 resolver 可解析）", op.id, out_tid)))
    }
}
