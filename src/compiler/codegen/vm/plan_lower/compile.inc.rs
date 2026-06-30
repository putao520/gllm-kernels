// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 compile_layer — 全链路入口
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Register VM 全管线编译入口。
///
/// Graph → FusionPlan → lower → OptPass → RegAlloc → StackFrame → IsaLower → 物理代码
///
/// SPEC/39 REQ-UMK-001: All compilation produces MegaKernelFn ABI code.
pub fn compile_layer(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    exec_plan: &crate::compiler::planner::ExecutionPlan,
    registry: Option<&ScalarOpRegistry>,
) -> Result<CodegenOutput, CompilerError> {
    let sym_map = SymDimSlotMap::mega_kernel_abi();
    compile_layer_with_sym_map(plan, graph, alloc, exec_plan, registry, &sym_map)
}

/// Same as `compile_layer` but accepts an explicit `SymDimSlotMap`.
/// Used internally when callers need to pass a custom ABI layout.
pub fn compile_layer_with_sym_map(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    exec_plan: &crate::compiler::planner::ExecutionPlan,
    registry: Option<&ScalarOpRegistry>,
    sym_map: &SymDimSlotMap,
) -> Result<CodegenOutput, CompilerError> {
    let dtype = graph_dtype(graph);
    let profile = IsaProfile::from_device_profile(&exec_plan.profile);
    let hook = isa_hook::select_hook(&profile);

    // Topology analysis: single-pass derivation replacing per-site graph.ops scans.
    let topology = super::topology::GraphTopologyAnalysis::analyze(graph);

    // ARCH-ROPE-CACHE: 预先扫描 plan 推导 RoPE cos/sin 表需求,
    // 再把 cache_offset 传入 lower_fusion_plan_inner (供 lower_rope_full 使用)。
    let rope_req = compute_rope_requirement(plan, graph, alloc)?;

    // PLE 中间张量 (ple_ctx / post_mlp_out) scratch 布局 (task #28.1)。
    let ple_req = compute_ple_requirement(plan, graph, alloc, rope_req.as_ref())?;

    // DWC padded input buffer scratch 布局 (T55 USM Conformer)。
    let dwc_req = compute_dwc_requirement(plan, graph, alloc, rope_req.as_ref(), ple_req.as_ref())?;

    // Stage 1: FusionPlan → VmProgram (IsaHook 驱动多算法选择)
    let mut program = lower_fusion_plan_inner_with_sym_map(plan, graph, alloc, registry, &profile,
        Some(hook.as_ref()), rope_req.as_ref(), ple_req.as_ref(), dwc_req.as_ref(), false, Some(sym_map), &topology)?;

    // Stage 1.5: 符号验证 — catch 低级错误 (栈对齐, 寄存器配对, 嵌套 skip)
    // 在 ISA lowering 前运行, 违规返回 Err 而非产生错误机器码。
    super::verify::verify_vm_program(&program)?;

    // §13 IsaHook 查询: 从实际 FusionPlan 中提取 GEMM 维度驱动 hook 决策
    let (mr, nr) = hook.gemm_microkernel_shape();
    let _ = (mr, nr);
    // 从 plan 中找到第一个 GEMM op 的真实维度 (用于 epi_place 推导)
    if let Some((m_dim, n, k)) = plan.groups.iter()
        .filter_map(|g| graph.op(g.anchor))
        .filter_map(|op| extract_gemm_dims_sym(op, graph).ok())
        .next()
    {
        let m_alloc = match &m_dim {
            SymDim::Concrete(v) => *v,
            SymDim::Symbolic { max_value, .. } => max_value.expect("GEMM M Symbolic needs max_value"),
        };
        // ARCH-EPILOGUE-PLACE: epilogue_strategy 基于累加器数量和 epilogue 操作数决定执行位置
        let epi_place = hook.epilogue_strategy(m_alloc * n.min(k), 2);
        let _ = (m_alloc, epi_place);
    }

    // §14.1 管线铁律: validate_provenance + validate_structure
    program.validate_provenance()
        .map_err(|e| CompilerError::CodegenViolation(format!("provenance: {e}")))?;
    program.validate_structure()
        .map_err(|e| CompilerError::CodegenViolation(format!("structure: {e}")))?;

    // D2: 操作数类型一致性 + D3: SIMD 宽度一致性 + D1: 值域验证
    if let Err(e) = program.validate_type_consistency() {
        return Err(CompilerError::CodegenViolation(format!("type-check: {e}")));
    }
    if let Err(e) = program.validate_width_consistency() {
        return Err(CompilerError::CodegenViolation(format!("width-check: {e}")));
    }
    if let Err(e) = program.validate_value_domains() {
        return Err(CompilerError::CodegenViolation(format!("value-domain: {e}")));
    }

    // Stage 2: VM 优化
    let pass_registry = PassRegistry::with_defaults();
    let _stats = pass_registry.run_all(&mut program, &profile, hook.as_ref());

    // ARCH-VREG-DECLARE-BEFORE-USE: opt pass 后验证 DeclareVReg 出现在所有 use
    // 之前。目前作为 warning (GLLM_STRICT_VALIDATE=1 启用严格模式)，因为 lower
    // 层面存在跨 TraceOp 的 VReg 前向引用 (gemma 的 GELU trace lowering 会 lazy
    // alloc)。严格检查需要先修复 lower 层的 declare 顺序。
    if let Err(e) = program.validate_declares_before_uses() {
        if std::env::var("GLLM_STRICT_VALIDATE").is_ok() {
            return Err(CompilerError::CodegenViolation(format!("post-opt: {e}")));
        }
        eprintln!("[WARN][ARCH-VREG-DECLARE-BEFORE-USE] {e}");
    }

    // GLLM_DUMP_VM_PRE=<dir>: dump VmProgram (opt pass 后,reg alloc 前),
    // 用于在 RegAlloc 失败时仍能看到完整的 instruction stream。
    if let Ok(dir) = std::env::var("GLLM_DUMP_VM_PRE") {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_IDX: AtomicUsize = AtomicUsize::new(0);
        let idx = CALL_IDX.fetch_add(1, Ordering::SeqCst);
        let _ = std::fs::create_dir_all(&dir);
        let anchor = plan.groups.first()
            .and_then(|g| graph.op(g.anchor))
            .map(|op| format!("{:?}", op.op).chars().take(30).collect::<String>())
            .unwrap_or_else(|| "unknown".to_string())
            .replace(|c: char| !c.is_alphanumeric(), "_");
        let path = format!("{}/{:04}_{}.pre.txt", dir, idx, anchor);
        if let Ok(mut f) = std::fs::File::create(&path) {
            writeln!(f, "=== VmProgram ({} instrs, pre-RegAlloc) ===", program.instrs.len()).ok();
            for (i, instr) in program.instrs.iter().enumerate() {
                writeln!(f, "{:4}: {:?}", i, instr).ok();
            }
        }
    }

    // Stage 3: 寄存器分配
    let alloc_result = RegAllocator::new(&profile).allocate(&program)
        .map_err(|e| CompilerError::CodegenViolation(format!("RegAlloc: {e}")))?;

    // GLLM_DUMP_VM_REG=<dir>: 每个 compile_layer dump VmProgram + RegAlloc
    // mapping 到 dir/<idx>_<anchor_op>.txt, 供逐层数值调试。
    if let Ok(dir) = std::env::var("GLLM_DUMP_VM_REG") {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_IDX: AtomicUsize = AtomicUsize::new(0);
        let idx = CALL_IDX.fetch_add(1, Ordering::SeqCst);
        let _ = std::fs::create_dir_all(&dir);
        let anchor = plan.groups.first()
            .and_then(|g| graph.op(g.anchor))
            .map(|op| format!("{:?}", op.op).chars().take(30).collect::<String>())
            .unwrap_or_else(|| "unknown".to_string())
            .replace(|c: char| !c.is_alphanumeric(), "_");
        let path = format!("{}/{:04}_{}.txt", dir, idx, anchor);
        if let Ok(mut f) = std::fs::File::create(&path) {
            writeln!(f, "=== VmProgram ({} instrs) ===", program.instrs.len()).ok();
            for (i, instr) in program.instrs.iter().enumerate() {
                writeln!(f, "{:4}: {:?}", i, instr).ok();
            }
            writeln!(f, "\n=== RegAlloc Mapping ===").ok();
            let mut pairs: Vec<_> = alloc_result.mapping.iter().collect();
            pairs.sort_by_key(|(v, _)| v.0);
            for (vreg, phys) in pairs {
                writeln!(f, "  v{} → {:?}", vreg.0, phys).ok();
            }
            writeln!(f, "\n=== Spill Slots ===").ok();
            writeln!(f, "  total_spills={}", alloc_result.spills.len()).ok();
            for (i, spill) in alloc_result.spills.iter().enumerate() {
                writeln!(f, "  slot[{}]: vreg={} offset={} size={}", i, spill.vreg.0, spill.offset, spill.size).ok();
            }
            writeln!(f, "\n=== Spill Offset → VReg ===").ok();
            for (i, spill) in alloc_result.spills.iter().enumerate() {
                if spill.vreg.0 != u32::MAX {
                    // Calculate rbp_offset for this spill
                    let rbp_off = -(88 + spill.offset as i32 + spill.size as i32);
                    writeln!(f, "  offset={} → slot[{}] vreg={} rbp_off={}", spill.offset, i, spill.vreg.0, rbp_off).ok();
                }
            }
        }
    }

    // Register allocation mapping available in alloc_result.mapping for debugging.

    // Stage 4: 栈帧计算
    let frame = StackFrame::compute(&alloc_result, &profile, alloc.total_bytes);

    // Stage 5: ISA Lower — 根据 profile.platform 分派到对应 backend
    // ARCH-CODEGEN-DISPATCH: X86_64 → X86Lower, AArch64 → AArch64Lower, GPU → GpuLower
    let (code, format) = match &profile.platform {
        super::isa_profile::Platform::X86_64 { has_avx512, .. } => {
            let mut lowerer = X86Lower::with_sym_map(*has_avx512, sym_map.clone());
            lowerer.set_scratch_gprs(&profile.scratch_gprs)?;
            lowerer.set_scratch_vec_regs(&profile.scratch_vec_regs)?;
            lowerer.precompute_zero_vregs(&program);
            lowerer.emit_prologue(&frame, &alloc_result)?;
            // StackLayout 现在在 emit_prologue 内部直接构建，无需 set_spill_base
            for instr in &program.instrs {
                lowerer.lower_instr(instr, &alloc_result)?;
            }
            lowerer.emit_epilogue(&frame, &alloc_result)?;
            (lowerer.finalize()?, crate::compiler::codegen::CodeFormat::MachineCode)
        }
        super::isa_profile::Platform::AArch64 { .. } => {
            let mut lowerer = super::aarch64_lower::AArch64Lower::with_profile(&profile);
            lowerer.emit_prologue(&frame, &alloc_result)?;
            for instr in &program.instrs {
                lowerer.lower_instr(instr, &alloc_result)?;
            }
            lowerer.emit_epilogue(&frame, &alloc_result)?;
            (lowerer.finalize()?, crate::compiler::codegen::CodeFormat::MachineCode)
        }
        super::isa_profile::Platform::Cuda { sm_version, .. } => {
            let dialect = super::gpu_lower::GpuDialect::Ptx { sm_version: *sm_version };
            let (text, fmt) = compile_gpu(&program, &frame, &alloc_result, dialect)?;
            (text.into_bytes(), fmt)
        }
        super::isa_profile::Platform::Hip { gfx_arch, wave_size, .. } => {
            let dialect = super::gpu_lower::GpuDialect::Hip { gfx_arch: *gfx_arch, wave_size: *wave_size };
            let (text, fmt) = compile_gpu(&program, &frame, &alloc_result, dialect)?;
            (text.into_bytes(), fmt)
        }
        super::isa_profile::Platform::Metal { gpu_family, .. } => {
            let dialect = super::gpu_lower::GpuDialect::Metal { gpu_family: *gpu_family };
            let (text, fmt) = compile_gpu(&program, &frame, &alloc_result, dialect)?;
            (text.into_bytes(), fmt)
        }
    };

    // ARCH-DATA-FLOW-CONTRACT §3: scratchpad_bytes 是调用方分配的外部 buffer
    // （传入 arg[8] = scratchpad_ptr），包含中间张量 + (可选) RoPE cos/sin 表
    // + (可选) PerLayerEmbed 中间张量 + (可选) DWC padded input buffer。
    let elem = dtype.elem_bytes();
    let base_after_rope = match &rope_req {
        Some(req) => {
            // cos/sin 表跟在中间张量之后,大小 = max_seq × head_dim × 4B。
            req.cache_offset + req.max_seq_len * req.head_dim * elem
        }
        None => alloc.total_bytes,
    };
    let base_after_ple = match &ple_req {
        Some(req) => req.post_mlp_offset + req.max_seq_len * req.hidden * elem,
        None => base_after_rope,
    };
    // MoEDispatchPacked scratchpad: gu_buf (2*intermediate_size*f32) + activ_buf (intermediate_size*f32)
    let moe_scratch = compute_moe_packed_requirement(plan, graph);
    let base_after_moe = base_after_ple + moe_scratch;
    let total_scratchpad = match &dwc_req {
        Some(req) => (req.padded_offset + req.total_bytes).max(64),
        None => base_after_moe.max(64),
    };

    Ok(CodegenOutput {
        code,
        format,
        scratchpad_bytes: total_scratchpad,
        hotpatch_points: vec![],
        rope_cache: rope_req,
    })
}

/// GPU Lower 统一入口 — Cuda/Hip/Metal 三种方言共用此函数。
/// ARCH-GPU-REG-KIND + ARCH-GPU-REG-COUNT: 注入 vreg_kinds 映射 + 按实际 VReg 数量动态声明。
fn compile_gpu(
    program: &VmProgram,
    frame: &StackFrame,
    alloc: &super::reg_alloc::RegAllocation,
    dialect: super::gpu_lower::GpuDialect,
) -> Result<(String, crate::compiler::codegen::CodeFormat), CompilerError> {
    let mut lowerer = super::gpu_lower::GpuLower::new(dialect);
    lowerer.set_vreg_kind_map(program);
    let counts = program.vreg_counts_by_kind();
    lowerer.emit_prologue(frame, alloc, counts)?;
    for instr in &program.instrs {
        lowerer.lower_instr(instr, alloc)?;
    }
    lowerer.emit_epilogue(frame, alloc)?;
    let text = lowerer.finalize()?;
    let format = match dialect {
        super::gpu_lower::GpuDialect::Ptx { .. } => crate::compiler::codegen::CodeFormat::Ptx,
        super::gpu_lower::GpuDialect::Hip { .. } => crate::compiler::codegen::CodeFormat::Hip,
        super::gpu_lower::GpuDialect::Metal { .. } => crate::compiler::codegen::CodeFormat::Msl,
    };
    Ok((text, format))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 辅助函数
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 从 ScalarOpRegistry 获取算子的 TraceOp body。
///
/// **铁律 (§14.1)**: trace 必须从 registry 获取 (SymExec 阶段 产出)。
/// 不允许 hardcode。如果 registry 没有该算子的 trace → 返回 Err (NO_SCALAR)。
pub(crate) fn extract_op_trace(
    op: &crate::compiler::graph::CompilerOp,
    registry: Option<&ScalarOpRegistry>,
    graph: &CompilerGraph,
) -> Result<Vec<TraceOp>, CompilerError> {
    // 从 Op (单 IR) 派生 OpKindKey
    let key = Some(ScalarOpRegistry::key_from_op(&op.op));

    // 从 registry 查询 SymExec trace
    if let (Some(reg), Some(k)) = (registry, &key) {
        if let Some(trace) = reg.get_trace(k) {
            // 从 ComputePattern 提取 body
            let mut body = extract_body_from_pattern(&trace.pattern);

            // ── OpKind 参数化 trace 重写 ──
            // 某些 OpKind 变体在单个 OpKindKey 下承载 per-op 数值参数
            // (例如 Op::SwiGluClipped { limit:  })。registry 存储的是
            // 规范模板 trace (canonical limit=7.0), codegen 时按 OpKind
            // 的实际参数重写对应 Const 槽。
            //
            // 这符合 §14.1 四阶段管线铁律:
            //   Scalar (scalar_swiglu_clipped) → SymExec 模板 trace
            //   → IR (per-op Const 重写) → ISA Lowering
            //
            // 不是 fallback, 是 ComputePattern 自动分发 IR 层的合法常量折叠。
            if let Some(Op::SwiGluClipped { limit }) = op.op_resolved(graph) {
                rewrite_swiglu_clipped_limit(&mut body, limit);
            }

            return Ok(body);
        }
    }

    // 没有 registry 或 registry 中没有该算子 → 返回 Err
    // (除了 Reshape/Transpose 是元数据操作，不需要 trace)
    match op.op_resolved(graph) {
        Some(Op::Reshape { .. }) | Some(Op::Transpose { .. }) | Some(Op::SliceView { .. }) => Ok(vec![]),
        // Gather has its own dedicated lower path, no scalar trace needed
        Some(Op::Gather { .. }) => Ok(vec![]),
        // QuantGather has its own dedicated lower path (emit_quant_gather_inline), no scalar trace needed
        Some(Op::QuantGather { .. }) => Ok(vec![]),
        // ColumnSlice: memory-bound row-major copy, dedicated lower path (lower_column_slice)
        Some(Op::ColumnSlice { .. }) => Ok(vec![]),
        // AltUp ops: Injective, dispatched via emit_injective_inline
        Some(Op::AltUpPredict { .. }) | Some(Op::AltUpCorrect { .. }) | Some(Op::AltUpInject { .. }) => Ok(vec![]),
        // MoERouter: opaque composite op (GEMM + softmax + top-k), specialized dispatch
        Some(Op::MoERouter { .. }) => Ok(vec![]),
        // MoEDispatchPacked: opaque 复合算子 (mxfp4 dequant + SwiGLU + GEMV), 专用分发
        Some(Op::MoEDispatchPacked { .. }) => Ok(vec![]),
        // ARCH-SG-QTAP: pure side-effect op, 无 scalar trace (直接由 lower_qtap_stg 发射)
        Some(Op::QTapSTG { .. }) => Ok(vec![]),
        // Argmax: 独立 lowering (VmInstr::Argmax), 无需 scalar trace
        Some(Op::Argmax { .. }) => Ok(vec![]),
        // MtpDraft: structural op lowered via TraceOp::MtpDraft → auto_select
        Some(Op::MtpDraft { .. }) => Ok(vec![]),
        // LogitSoftcap: 生成 per-cap 的 elementwise trace (cap * tanh(x / cap))
        // 用于 GEMM EpilogueInjection 融合。
        Some(Op::LogitSoftcap { cap }) => {
            let cap_val = cap as f64;
            let inv_cap = 1.0 / cap_val;
            Ok(vec![
                TraceOp::Input(0),       // [0] x
                TraceOp::Const(inv_cap), // [1] 1/cap
                TraceOp::Mul(ValueId(0), ValueId(1)),      // [2] x * (1/cap)
                TraceOp::Tanh(ValueId(2)),        // [3] tanh(x * (1/cap))
                TraceOp::Const(cap_val), // [4] cap
                TraceOp::Mul(ValueId(3), ValueId(4)),      // [5] cap * tanh(...)
            ])
        }
        _ => Err(CompilerError::CodegenViolation(
            format!(
                "extract_op_trace: Op {:?} 没有在 ScalarOpRegistry 中注册。\
                 违反 §14.1 四阶段管线铁律——所有算子必须走 Scalar→SymExec→TraceOp 管线。",
                op.op
            )
        )),
    }
}

/// 按 `Op::SwiGluClipped { limit:  }` 的实际 `limit` 值重写 trace body。
///
/// 模板 trace 使用 canonical `Const(+7.0)` / `Const(-7.0)` 表达 ±limit 钳位。
/// 调用方为每个 op 实例把这两个常量换成真实的 `±limit`。
///
/// 约束: 模板 trace 形状必须精确匹配 `register_swiglu_clipped` 中的 14-op SSA
/// 布局——slot [2]=+limit, slot [3]=-limit。任何不匹配都说明 registry 注册被
/// 破坏, 必须 panic 防止静默错误结果 (NO_SILENT_FALLBACK)。
fn rewrite_swiglu_clipped_limit(body: &mut [TraceOp], limit: f32) {
    use crate::compiler::trace::TraceOp;
    // 防御性校验: body 长度至少 14 (见 registry 中的 SSA 布局注释)。
    assert!(
        body.len() >= 14,
        "rewrite_swiglu_clipped_limit: template trace shorter than 14 ops \
         (got {}), registry 注册已被破坏",
        body.len()
    );
    // slot [2] 必须是 +limit 常量,slot [3] 必须是 -limit 常量。
    match (&body[2], &body[3]) {
        (TraceOp::Const(p), TraceOp::Const(n)) => {
            debug_assert!(
                *p > 0.0 && *n < 0.0,
                "template limit constants must be (+, -) canonical pair"
            );
        }
        _ => panic!(
            "rewrite_swiglu_clipped_limit: template trace slots [2]/[3] are \
             not Const — registry 注册布局已变化,更新 rewrite 逻辑"
        ),
    }
    body[2] = TraceOp::Const(limit as f64);
    body[3] = TraceOp::Const(-(limit as f64));
}

/// Rewrite LogitSoftcap trace placeholders with actual cap value.
/// Template: [Input(0), Const(1/cap_ph), Mul(0,1), Tanh(2), Const(cap_ph), Mul(3,4)]
/// Rewrites slot [1] to 1.0/cap and slot [4] to cap.
fn rewrite_logit_softcap_cap(body: &mut [TraceOp], cap: f32) {
    use crate::compiler::trace::TraceOp;
    assert!(
        body.len() >= 6,
        "rewrite_logit_softcap_cap: template trace shorter than 6 ops (got {})",
        body.len()
    );
    body[1] = TraceOp::Const(1.0 / cap as f64);
    body[4] = TraceOp::Const(cap as f64);
}



/// 从 ComputePattern 提取主体 TraceOp 序列。
fn extract_body_from_pattern(pattern: &ComputePattern) -> Vec<TraceOp> {
    match pattern {
        ComputePattern::Elementwise { body } => body.clone(),
        ComputePattern::BinaryElementwise { body } => body.clone(),
        ComputePattern::Injective { body, .. } => body.clone(),
        ComputePattern::Reduction { combine, .. } => combine.clone(),
        ComputePattern::NormLike { reduce, .. } => reduce.clone(),
        ComputePattern::Gemm => vec![TraceOp::Input(0)],
        ComputePattern::QuantDecode { decode, .. } => decode.clone(),
    }
}

/// ComputePattern 自动分发: ComputePattern-driven auto dispatch.
///
/// 如果 ScalarOpRegistry 中有该 OpKind 的 trace 且 ComputePattern 是
/// Elementwise 或 BinaryElementwise，自动通过 `emit_elementwise_inline`
/// 生成代码。这消除了 GELU/Tanh/Sigmoid/Relu/Add/Mul 等纯 elementwise
/// 算子的手写 match arm 需求。
///
/// 返回 Ok(true) 表示已自动分发，Ok(false) 表示需要专用 lower 函数。
#[allow(clippy::too_many_arguments)]
fn try_auto_dispatch_elementwise(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    seq_bound_override: Option<&BoundExpr>,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
) -> Result<bool, CompilerError> {
    use crate::compiler::trace::ComputePattern;

    // Residual with telemetry needs fused compute+telemetry loop (emit_residual_with_telemetry),
    // not separate elementwise + post-hook. Skip auto-dispatch in that case.
    if matches!(op.op_resolved(graph), Some(Op::Residual)) && graph.telemetry.residual_cosine_sim {
        return Ok(false);
    }

    // Ops with dedicated dispatch paths must be excluded from auto-dispatch.
    // - Structural ops: Injective traces are skeletal, need dedicated lowering
    // - NormLike/Reduction ops: dedicated lower_op paths with
    //   specialized weight/bias handling not supported by generic emission
    if matches!(op.op_resolved(graph),
        Some(Op::ColumnSlice { .. }) | Some(Op::Gather { .. }) |
        Some(Op::DepthwiseConv1D { .. }) | Some(Op::PatchEmbed { .. }) |
        Some(Op::LearnedPos2D { .. }) |
        Some(Op::AltUpPredict { .. }) | Some(Op::AltUpCorrect { .. }) | Some(Op::AltUpInject { .. }) |
        Some(Op::RmsNorm(_)) | Some(Op::LayerNorm(_)) |
        Some(Op::ValueNorm(_)) | Some(Op::QkNorm { .. }) |
        Some(Op::HeadRmsNorm { .. }) | Some(Op::Softmax) |
        Some(Op::L2Normalize { .. }) | Some(Op::MeanPool { .. }) |
        Some(Op::Argmax { .. }) | Some(Op::MtpDraft { .. })
    ) {
        return Ok(false);
    }

    let reg = match ctx.session.registry {
        Some(r) => r,
        None => return Ok(false),
    };

    // 从 Op (单 IR) 派生 OpKindKey
    let key = ScalarOpRegistry::key_from_op(&op.op);

    let trace = match reg.get_trace(&key) {
        Some(t) => t,
        None => return Ok(false),
    };

    // 仅 dispatch Elementwise / BinaryElementwise / Injective 模式
    // NormLike/Gemm 由 lower_op 处理
    // Injective 仅支持 num_inputs ≤ 2（emit_injective_inline 最多 2 输入）
    // 和 num_outputs ≤ 1（通用 elementwise 单输出）。
    let body = match &trace.pattern {
        ComputePattern::Elementwise { body } => body.clone(),
        ComputePattern::BinaryElementwise { body } => body.clone(),
        ComputePattern::Injective { body, num_inputs, num_outputs } => {
            if *num_inputs > 2 || *num_outputs > 1 {
                return Ok(false);
            }
            body.clone()
        }
        ComputePattern::Reduction { .. } => {
            return try_dispatch_reduction(prog, op, graph, &trace.pattern, ctx,
                input_ptr, output_ptr, seq_bound_override);
        }
        ComputePattern::NormLike { .. } => {
            return try_dispatch_normlike(prog, op, graph, &trace.pattern, ctx,
                input_ptr, weight_ptr, output_ptr, seq_bound_override);
        }
        _ => return Ok(false),
    };

    if body.is_empty() {
        return Ok(false);
    }

    // Op 参数化 trace 重写 (与 extract_op_trace 逻辑一致)
    let mut body = body;
    if let Some(Op::SwiGluClipped { limit }) = op.op_resolved(graph) {
        rewrite_swiglu_clipped_limit(&mut body, limit);
    }
    if let Some(Op::LogitSoftcap { cap }) = op.op_resolved(graph) {
        rewrite_logit_softcap_cap(&mut body, cap);
    }

    // ── Auto dispatch: trace + pointer resolution + emit ──
    let (out_shape, _) = infer_output_shape_sym(op, graph)?;
    let is_binary = op.inputs.len() > 1;

    let resolved_input = op.inputs.first().copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .unwrap_or(input_ptr);
    let resolved_weight = if is_binary {
        op.inputs.get(1).copied()
            .and_then(|tid| resolver.materialize(prog, tid, abi))
            .unwrap_or(weight_ptr)
    } else {
        weight_ptr
    };
    let resolved_output = op.outputs.first().copied()
        .and_then(|tid| resolver.materialize(prog, tid, abi))
        .unwrap_or(output_ptr);

    // ARCH-BROADCAST: 若 secondary operand shape 不含输出的 Symbolic 外层维度
    // (典型: bias [hidden] vs activation [seq, hidden])，则按广播语义，
    // row_weight 不按 outer offset 偏移，始终指向 weight_ptr[0..feature_dim]。
    let weight_is_broadcast = if is_binary {
        let weight_tid = op.inputs[1];
        let weight_shape = graph.tensor(weight_tid)
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        let outer_sym_in_out = out_shape.iter().find(|d| d.is_symbolic());
        match outer_sym_in_out {
            Some(sym) => !weight_shape.iter().any(|d| d == sym),
            None => false,
        }
    } else {
        false
    };

    let _acc = emit_elementwise_inline(prog, &body, &out_shape, ctx.session.width, is_binary,
        weight_is_broadcast,
        resolved_input, resolved_weight, resolved_output, ctx.session.sym_map, seq_bound_override, ctx.dtype)?;

    Ok(true)
}

/// Reduction 模式通用处理器 (ARCH-AUTO-INSTR-SELECT)。
///
/// 从 ComputePattern::Reduction 提取 combine + normalize trace，
/// 自动生成累加循环 → HReduce → normalize 写回。
/// NormLike 模式通用处理器 (ARCH-AUTO-INSTR-SELECT).
///
/// 从 ComputePattern::NormLike 的 reduce/finalize/transform trace 自动生成
/// 三阶段归一化代码 (reduce → finalize → transform)。
/// 覆盖 SoftmaxWithEntropy 等所有 NormLike 算子（已有 OpKind 的除外）。
fn try_dispatch_normlike(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    pattern: &ComputePattern,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    seq_bound_override: Option<&BoundExpr>,
) -> Result<bool, CompilerError> {
    let (out_shape, feature_dim) = infer_output_shape_sym(op, graph)?;
    if feature_dim == 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "NormLike op {:?}: feature_dim=0, cannot emit normlike", op.id)));
    }
    let seq_bound = seq_bound_override
        .cloned()
        .or_else(|| out_shape.first().map(|d| ctx.session.sym_map.to_bound(d)))
        .unwrap_or(BoundExpr::Const(1));
    let norm_kind = match op.op_resolved(graph).and_then(|o| o.norm_meta()) {
        Some(meta) => match meta.variant {
            crate::compiler::graph::NormVariant::RmsNorm => NormKind::RmsNorm,
            crate::compiler::graph::NormVariant::HeadRmsNorm => NormKind::HeadRmsNorm,
            crate::compiler::graph::NormVariant::LayerNorm => NormKind::LayerNorm,
            crate::compiler::graph::NormVariant::ValueNorm => NormKind::ValueNorm,
        },
        None => NormKind::RmsNorm,
    };
    let weight_dtype = op.inputs.get(1)
        .and_then(|&tid| graph.tensor(tid))
        .map(|t| t.dtype.to_quant_precision())
        .unwrap_or(ctx.dtype);
    emit_normlike_inline(
        prog, pattern, feature_dim, /*groups_per_row=*/1,
        /*broadcast_weight=*/false, norm_kind,
        ctx.session.width, seq_bound, input_ptr, weight_ptr, output_ptr,
        ctx.dtype,
        weight_dtype, // BCE-20260629-011: 传递 weight dtype
    )?;
    Ok(true)
}

/// Reduction 模式通用处理器 (ARCH-AUTO-INSTR-SELECT)。
///
/// 几何参数 (seq_bound, feature_dim) 从 OpKind 中提取，
/// 但算术逻辑完全由 trace 驱动。
pub(crate) fn try_dispatch_reduction(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    pattern: &ComputePattern,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    output_ptr: VRegId,
    seq_bound_override: Option<&BoundExpr>,
) -> Result<bool, CompilerError> {
    use crate::compiler::trace::ComputePattern;

    let (identity, combine, normalize) = match pattern {
        ComputePattern::Reduction { identity, combine, normalize, .. } => {
            eprintln!("[REDUCTION-DIAG] identity={} combine={:?} normalize={:?}", identity, combine, normalize);
            (*identity, combine.clone(), normalize.clone())
        }
        _ => return Ok(false),
    };

    // 从 Op 提取几何参数（胖 opcode 自描述）
    let geom = match op.op_resolved(graph).and_then(|o| o.reduction_geometry()) {
        Some(g) => g,
        None => return Ok(false),
    };
    let feature_dim = geom.hidden;
    let seq_bound = if geom.cls_mode {
        BoundExpr::Const(1)
    } else if let Some(override_bound) = seq_bound_override.cloned() {
        override_bound
    } else if matches!(op.op_resolved(graph), Some(Op::L2Normalize { .. })) {
        BoundExpr::Const(1)
    } else {
        let input_dim = op.inputs.first()
            .and_then(|&tid| graph.tensor(tid))
            .and_then(|t| t.shape.first());
        match input_dim {
            Some(SymDim::Symbolic { name, max_value }) => {
                BoundExpr::Symbolic(SymBound {
                    name: name.clone(),
                    max_alloc: max_value.expect("Symbolic dim needs max_value"),
                })
            }
            _ => BoundExpr::Const(geom.seq_len),
        }
    };

    if feature_dim == 0 {
        return Err(CompilerError::CodegenViolation(
            "try_dispatch_reduction: zero feature_dim".into()));
    }

    let width = ctx.session.width;
    let dtype = ctx.dtype;
    let lanes = width.f32_lanes().max(1);
    let elem = dtype.elem_bytes();
    let vec_count = feature_dim / lanes;
    let step = width.bytes();
    let row_bytes = feature_dim * dtype.elem_bytes();

    // MeanPool with Symbolic seq_len: convert to Runtime BEFORE scale computation
    // so that 1/N uses the actual runtime seq_len, not 1/max_alloc.
    let seq_bound = match &seq_bound {
        BoundExpr::Symbolic(_sb) => {
            let resolved = ctx.session.sym_map.resolve("seq_len").cloned();
            eprintln!("[MEANPOOL-SYMDIM] Symbolic seq_bound, sym_map.resolve('seq_len') = {:?}", resolved);
            resolved
                .map(BoundExpr::Runtime)
                .unwrap_or_else(|| { eprintln!("[MEANPOOL-SYMDIM] WARNING: no seq_len in sym_map, falling back to Symbolic"); seq_bound.clone() })
        }
        other => other.clone(),
    };

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let tmp = prog.alloc_vreg(VRegKind::Vec, width);
    let scale = prog.alloc_vreg(VRegKind::Vec, width);

    // 计算 1/N 缩放因子
    match &seq_bound {
        BoundExpr::Const(n) if *n > 0 => {
            let inv_n = 1.0f32 / *n as f32;
            prog.emit(VmInstr::Broadcast { dst: scale, src: ScalarExpr::Const(inv_n), width, dtype: ctx.dtype, });
        }
        BoundExpr::Const(_) => {
            prog.emit(VmInstr::Broadcast { dst: scale, src: ScalarExpr::Const(1.0), width, dtype: ctx.dtype, });
        }
        BoundExpr::Symbolic(sb) => {
            // Fallback: no sym_map entry — use 1/max_alloc approximation
            let inv_n = 1.0f32 / sb.max_alloc as f32;
            prog.emit(VmInstr::Broadcast { dst: scale, src: ScalarExpr::Const(inv_n), width, dtype: ctx.dtype, });
        }
        BoundExpr::Runtime(ptr_expr) => {
            // Runtime: load seq_len from stack/ABI, convert to float, compute 1/N.
            // ptr_expr points to an integer (i32/u32) in memory — LoadPtr reads it
            // into a GPR, then IndexToScalar converts to f32 for 1/N division.
            let n_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::LoadPtr { dst: n_gpr, src: ptr_expr.clone() });
            let n_float = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::IndexToScalar { dst: n_float, src: n_gpr });
            let n_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: n_vec, src: ScalarExpr::VReg(n_float), width, dtype: ctx.dtype, });
            let ones = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: ones, src: ScalarExpr::Const(1.0), width, dtype: ctx.dtype, });
            prog.emit(VmInstr::VecBinOp { dst: scale, a: ones, b: n_vec, op: VecOp::Div, dtype: ctx.dtype, });
        }
        BoundExpr::DynamicVReg(vreg) => {
            // DynamicVReg: 外层 loop counter 的 GPR 值通过 IndexToScalar 转为 float,
            // 计算 1/N — 与 Runtime 分支同路径,跳过 LoadPtr(值已在 GPR 中).
            let n_float = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::IndexToScalar { dst: n_float, src: *vreg });
            let n_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: n_vec, src: ScalarExpr::VReg(n_float), width, dtype: ctx.dtype, });
            let ones = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: ones, src: ScalarExpr::Const(1.0), width, dtype: ctx.dtype, });
            prog.emit(VmInstr::VecBinOp { dst: scale, a: ones, b: n_vec, op: VecOp::Div, dtype: ctx.dtype, });
        }
        BoundExpr::DynamicVRegPlusOne(vreg) => {
            // DynamicVRegPlusOne: N = vreg + 1. IndexToScalar + float add 1.0 + div.
            let n_float = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);
            prog.emit(VmInstr::IndexToScalar { dst: n_float, src: *vreg });
            let n_vec = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: n_vec, src: ScalarExpr::VReg(n_float), width, dtype: ctx.dtype, });
            let one = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast { dst: one, src: ScalarExpr::Const(1.0), width, dtype: ctx.dtype, });
            let n_plus_one = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecBinOp { dst: n_plus_one, a: n_vec, b: one, op: VecOp::Add, dtype: ctx.dtype, });
            prog.emit(VmInstr::VecBinOp { dst: scale, a: one, b: n_plus_one, op: VecOp::Div, dtype: ctx.dtype, });
        }
    }

    let row_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // Determine if combine is a simple Add — if so, use Accumulate directly
    // without also running the combine trace (which would double-add).
    let combine_is_simple_add = combine.len() == 3
        && matches!(combine[0], TraceOp::Input(_))
        && matches!(combine[1], TraceOp::Input(_))
        && matches!(combine[2], TraceOp::Add(_, _));

    // 外层: 列向量化循环 (Const)
    // 内层: 行循环 — 用 LoadPtr 计算行基地址
    if vec_count > 0 {
        prog.emit_loop(BoundExpr::Const(vec_count), step, |prog, _col_ctr, col_off| {
            prog.emit(VmInstr::Broadcast { dst: acc, src: ScalarExpr::Const(identity as f32), width, dtype: ctx.dtype, });

            prog.emit_loop(seq_bound.clone(), row_bytes, |prog, _row_ctr, row_off| {
                prog.emit(VmInstr::LoadPtr {
                    dst: row_base,
                    src: PtrExpr::VRegPlusVReg(input_ptr, row_off),
                });
                prog.emit(VmInstr::VecLoad {
                    dst: tmp, base: row_base, offset: OffsetExpr::LoopOffset(col_off), width,
                    dtype: ctx.dtype, predicate: None,
                });
                if combine_is_simple_add {
                    // Simple Add: Accumulate handles acc += tmp directly
                    prog.emit(VmInstr::Accumulate { acc, src: tmp });
                } else {
                    // Complex combine: let trace handle it, skip redundant Accumulate
                    super::auto_select::auto_lower_trace(prog, &combine, &[acc, tmp], width, dtype)
                        .expect("try_dispatch_reduction: combine trace invariant violation");
                }
            });

            if let Some(ref norm_body) = normalize {
                super::auto_select::auto_lower_trace(prog, norm_body, &[acc, scale], width, dtype)
                    .expect("try_dispatch_reduction: normalize trace invariant violation");
            }

            prog.emit(VmInstr::VecStore {
                base: output_ptr, offset: OffsetExpr::LoopOffset(col_off), src: acc, width,
                dtype: ctx.dtype, predicate: None,
            });
        });
    }

    Ok(true)
}

/// Injective 模式通用处理器 (ARCH-AUTO-INSTR-SELECT)。
///
/// 从 ComputePattern::Injective 的 body trace 自动生成算术 VmInstr。
/// 纯算术处理器——循环结构、内存加载由调用方负责。
///
/// 调用方在每次循环迭代中：
/// 1. 准备好 `inputs` VReg（已加载的数据）
/// 2. 调用本函数生成算术
/// 3. 从返回的 slots 中取出所需的输出 slot，存储到目标位置
///
/// 返回所有 SSA slot VRegIds，调用方按需取用。
fn emit_injective_inline(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    width: SimdWidth,
    dtype: crate::compiler::trace::QuantPrecision,
) -> Result<Vec<VRegId>, CompilerError> {
    super::auto_select::auto_lower_trace_raw(prog, body, inputs, width, dtype)
}

/// Standalone/LoopFusion: 两层 Op lowering dispatch。
///
/// Layer 1: `try_auto_dispatch_elementwise` — 纯 Elementwise/BinaryElementwise/Reduction
///   算子自动通过 registry trace pipeline dispatch (auto_lower_trace)。
///   覆盖: Silu, Gelu, Tanh, Sigmoid, Relu, Add, Mul, SwiGlu, GeGlu, MeanPool,
///   Residual (no telemetry), LogitSoftcap 等。
///
/// Layer 2: `lower_op` — 胖 opcode 驱动，所有 NormLike/Gemm/Attention/MoE/Structural
///   算子通过 Op Spec struct 自描述参数，直接 emit VmInstr。
///   包括: RmsNorm, LayerNorm, ValueNorm, QkNorm, HeadRmsNorm, Softmax, Gemm/GemmBias,
///   MHA, CachedGqa, MlaAttention, RoPE, MoE, Argmax, StoreToken, WriteLogits, etc.
// @trace REQ-AIS-005 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/standalone-op]
pub(super) fn emit_standalone_op(
    prog: &mut VmProgram,
    op: &crate::compiler::graph::CompilerOp,
    graph: &CompilerGraph,
    ctx: &LoweringContext,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    rope_cache_offset: Option<usize>,
    resolver: &TensorPtrResolver,
    abi: &AbiPtrs,
) -> Result<(), CompilerError> {
    // BCE-20260630-MIXED-P1: per-op ctx（杠杆总闸）。
    // @trace CTX-PER-OP-DTYPE [req:REQ-DTYPE-006] [level:unit]
    // 在派发瓶颈点刷新 ctx.dtype = 当前 op 的激活计算精度（op_input_dtype → accumulator_dtype
    // promote），替代旧 graph_dtype(graph) 全图统一 F32。下游 RoPE/elementwise/softmax/
    // attention 调用点经 &op_ctx 自动获 per-op 激活 dtype（B1 下激活 F32 → accumulator_dtype
    // 仍 F32，行为等价；B2-ready）。详见 LoweringContext::for_op。
    let op_ctx = ctx.for_op(op, graph);
    let ctx = &op_ctx;
    let width = ctx.session.width;
    let sym_map = ctx.session.sym_map;
    let registry = ctx.session.registry;
    // Elementwise seq_bound_override: reuse resolve_dim logic for outer Symbolic dim.
    let seq_bound_override: Option<BoundExpr> = abi.mega_decode_seq_len.map(BoundExpr::DynamicVReg);

    // ── ComputePattern 自动分发: ComputePattern 自动指令选择 ──
    // 纯 Elementwise/BinaryElementwise 算子自动通过 registry trace pipeline dispatch，
    // 不需要手写 match arm。GELU/Tanh/Sigmoid/Relu/Add/Mul/SwiGLU 等全走此路径。
    if try_auto_dispatch_elementwise(
        prog, op, graph, ctx,
        input_ptr, weight_ptr, output_ptr,
        seq_bound_override.as_ref(), resolver, abi,
    )? {
        // ── Telemetry post-hooks for elementwise ops ──
        // §13.5 SiLU dead neuron telemetry: post-hook after auto-dispatch.
        // Residual telemetry is NOT a post-hook — it needs fused compute+telemetry loop,
        // handled by lower_op when graph.telemetry.residual_cosine_sim is true.
        if matches!(op.op_resolved(graph), Some(Op::Silu)) && graph.telemetry.silu_dead_neuron {
            let (out_shape, _) = infer_output_shape_sym(op, graph)?;
            emit_silu_dead_neuron_telemetry(prog, input_ptr, &out_shape, width, sym_map, ctx.dtype)?;
        }
        return Ok(());
    }

    // ── Op dispatch (胖 opcode 驱动) ──
    if lower_op(prog, op, graph, ctx, resolver, abi)? {
        return Ok(());
    }

    // lower_op 未处理 → 报错（所有 ops 应通过 lower_op 处理）
    Err(CompilerError::CodegenViolation(format!(
        "emit_standalone_op: op {:?} 未被 lower_op 处理", op.op
    )))
}

/// Structural op dispatch (ARCH-AUTO-INSTR-SELECT Category C/D).
///
/// **Category C** (手写 lower 委托, awaiting TraceOp extension for auto_select migration):
/// Gather, ColumnSlice, QTapSTG.
///
/// **Category D** (permanent, cannot use auto_select):
/// GEMM 维度提取——保留完整 SymDim（ARCH-SYMDIM-NO-UNWRAP）。
/// 返回 (m_sym, n, k)。调用方通过 sym_map.to_bound(&m_sym) 获取循环 bound。
pub(super) fn extract_gemm_dims_sym(op: &crate::compiler::graph::CompilerOp, graph: &CompilerGraph) -> Result<(SymDim, usize, usize), CompilerError> {
    op.op_gemm_dims(graph).ok_or_else(|| CompilerError::CodegenViolation(format!("not a GEMM op: {:?}", op.op)))
}

/// 从 FusionGroup 的 epilogue ops 收集合并的 TraceOp 链。
///
/// ARCH-EPILOGUE-CHAIN: 多 op 串联需要 slot 索引重映射（下游 Input(0) → 上游输出 slot）。
/// 当前架构暂只支持单 op epilogue；多 op 需要扩展 TraceOp 引入 Copy/Move 变体或
/// 改用"每个 op 一次 lower_trace_body 调用"模式。
pub(super) fn collect_epilogue_trace(
    group: &crate::compiler::fusion::FusionGroup,
    graph: &CompilerGraph,
    registry: Option<&ScalarOpRegistry>,
) -> Result<Vec<TraceOp>, CompilerError> {
    // Filter to ops that have trace bodies (Argmax, StoreToken, etc. return empty
    // traces — they have specialized lowering and should not participate in epilogue
    // trace chaining). Only trace-bearing ops count toward the multi-op limit.
    let trace_ops: Vec<_> = group.epilogue.iter()
        .filter_map(|&op_id| {
            let op = graph.op(op_id)?;
            Some((op_id, op))
        })
        .collect();
    let trace_bodies: Vec<Vec<TraceOp>> = trace_ops.iter()
        .map(|(_, op)| extract_op_trace(op, registry, graph))
        .collect::<Result<Vec<_>, _>>()?;
    let trace_bearing_count = trace_bodies.iter().filter(|b| !b.is_empty()).count();
    if trace_bearing_count > 1 {
        return Err(CompilerError::CodegenViolation(format!(
            "collect_epilogue_trace: {} 个有 trace 的 epilogue op 需要 SSA slot 重映射，\
             当前只支持单 op（ARCH-EPILOGUE-CHAIN 待实现）",
            trace_bearing_count
        )));
    }
    let mut chain: Vec<TraceOp> = Vec::new();
    for body in &trace_bodies {
        chain.extend(body.iter().cloned());
    }
    Ok(chain)
}

/// 推导 op 的输出形状信息 (保留 SymDim 语义)。
///
/// 返回 (outer_dims: Vec<SymDim>, feature_dim: usize)。
/// outer_dims 可能包含 Symbolic 维度（如 seq_len），feature_dim 始终 Concrete。
/// ARCH-SYMDIM-NO-CONST-DEGRADE: 禁止用 max_for_allocation 压平 Symbolic 维度。
pub(crate) fn infer_output_shape_sym(op: &crate::compiler::graph::CompilerOp, graph: &CompilerGraph) -> Result<(Vec<SymDim>, usize), CompilerError> {
    let out_tid = op.outputs.first()
        .ok_or_else(|| CompilerError::CodegenViolation(format!("op '{}' has no outputs", op.label)))?;
    let tensor = graph.tensor(*out_tid)
        .ok_or_else(|| CompilerError::CodegenViolation(format!("tensor {:?} not in graph", out_tid)))?;
    if tensor.shape.is_empty() {
        return Err(CompilerError::CodegenViolation(format!("op '{}' output has empty shape", op.label)));
    }
    let feature_dim = tensor.shape.last()
        .and_then(|d| d.as_concrete())
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("ARCH-SYMDIM-OUTER-ONLY: op '{}' output tensor '{}' shape={:?} — inner dim must be Concrete",
                op.label, tensor.name, tensor.shape)
        ))?;
    Ok((tensor.shape.clone(), feature_dim))
}

/// RmsNorm / ValueNorm 的标量 pattern。LayerNorm 不用此 pattern — 它走
/// RmsNorm/ValueNorm NormLike pattern builder (LayerNorm uses emit_layernorm_auto)。
pub(crate) fn build_norm_pattern(op: &crate::compiler::graph::CompilerOp, graph: &CompilerGraph) -> Result<ComputePattern, CompilerError> {
    let meta = op.op_resolved(graph).and_then(|o| o.norm_meta())
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("build_norm_pattern: expected NormLike op, got op {:?}", op.op)
        ))?;
    // 仅 RmsNorm/ValueNorm 走 build_norm_pattern（LayerNorm 走 emit_layernorm_auto）
    if !matches!(meta.variant, crate::compiler::graph::NormVariant::RmsNorm | crate::compiler::graph::NormVariant::ValueNorm) {
        return Err(CompilerError::CodegenViolation(
            format!("build_norm_pattern: only RmsNorm/ValueNorm supported, got {:?}", op.op)
        ));
    }
    let eps = meta.eps;
    let has_weight = meta.has_weight;
    let transform = if has_weight {
        vec![
            TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
            TraceOp::Mul(ValueId(0), ValueId(1)),  // x * scale
            TraceOp::Mul(ValueId(3), ValueId(2)),  // (x * scale) * weight
        ]
    } else {
        vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))]
    };
    Ok(ComputePattern::NormLike {
        reduce: vec![TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(0))],
        finalize: vec![
            TraceOp::Input(0), TraceOp::Input(1), TraceOp::Div(ValueId(0), ValueId(1)),
            TraceOp::Const(eps as f64), TraceOp::Add(ValueId(2), ValueId(3)), TraceOp::Rsqrt(ValueId(4)),
        ],
        transform,
    })
}

/// HeadRmsNorm pattern: 与 RmsNorm 一致 (mean-based + weight),仅 feature_dim 不同。
/// 与 build_norm_pattern 区别:HeadRmsNorm transform 含 weight × (Input(2)),
/// 实际数学等同 RmsNorm with weight。
pub(crate) fn build_norm_pattern_head_rms(eps: f32) -> Result<ComputePattern, CompilerError> {
    Ok(ComputePattern::NormLike {
        reduce: vec![TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(0))],
        finalize: vec![
            TraceOp::Input(0), TraceOp::Input(1), TraceOp::Div(ValueId(0), ValueId(1)),
            TraceOp::Const(eps as f64), TraceOp::Add(ValueId(2), ValueId(3)), TraceOp::Rsqrt(ValueId(4)),
        ],
        transform: vec![
            TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2),
            TraceOp::Mul(ValueId(0), ValueId(1)),  // x * scale
            TraceOp::Mul(ValueId(3), ValueId(2)),  // (x * scale) * weight
        ],
    })
}

/// QkNorm pattern: L2 normalization + √head_dim rescaling (no learned weight).
///
/// Math per head d ∈ [head_dim]:
///   sum_sq  = Σ_d x²
///   inv_norm = 1 / (sqrt(sum_sq) + eps)   // NOT 1/sqrt(mean+eps) like RmsNorm
///   out     = x * inv_norm * √head_dim
///
/// finalize body: [Input(0)=sum_sq, Sqrt(0)=√sum_sq, Const(eps), Add(1,2), Const(1.0), Div(4,3), Const(√head_dim), Mul(5,6)]
/// transform body: [Input(0)=x, Input(1)=inv_norm_scale, Mul(0,1)]  (no weight)
pub(crate) fn build_norm_pattern_qk(eps: f32, head_dim: usize) -> Result<ComputePattern, CompilerError> {
    let sqrt_head_dim = (head_dim as f64).sqrt();
    Ok(ComputePattern::NormLike {
        reduce: vec![TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(0))],
        finalize: vec![
            TraceOp::Input(0),          // slot 0: sum_sq
            TraceOp::Sqrt(ValueId(0)),           // slot 1: sqrt(sum_sq)
            TraceOp::Const(eps as f64), // slot 2: eps
            TraceOp::Add(ValueId(1), ValueId(2)),         // slot 3: sqrt(sum_sq) + eps
            TraceOp::Const(1.0),        // slot 4: 1.0
            TraceOp::Div(ValueId(4), ValueId(3)),         // slot 5: 1 / (sqrt(sum_sq) + eps) = inv_norm
            TraceOp::Const(sqrt_head_dim), // slot 6: √head_dim
            TraceOp::Mul(ValueId(5), ValueId(6)),         // slot 7: inv_norm * √head_dim
        ],
        transform: vec![
            TraceOp::Input(0),  // x
            TraceOp::Input(1),  // inv_norm * √head_dim (from finalize result)
            TraceOp::Mul(ValueId(0), ValueId(1)), // x * inv_norm_scale
        ],
    })
}

