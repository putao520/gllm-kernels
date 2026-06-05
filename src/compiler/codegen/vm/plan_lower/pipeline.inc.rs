// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 FusionPlan → VmProgram
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 将 FusionPlan 翻译为 VmProgram。
pub fn lower_fusion_plan(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    registry: Option<&ScalarOpRegistry>,
    profile: &IsaProfile,
    hook: Option<&dyn super::isa_hook::IsaHook>,
) -> Result<VmProgram, CompilerError> {
    let rope_req = compute_rope_requirement(plan, graph, alloc)?;
    let ple_req = compute_ple_requirement(plan, graph, alloc, rope_req.as_ref())?;
    let dwc_req = compute_dwc_requirement(plan, graph, alloc, rope_req.as_ref(), ple_req.as_ref())?;
    lower_fusion_plan_inner(plan, graph, alloc, registry, profile, hook, rope_req.as_ref(), ple_req.as_ref(), dwc_req.as_ref(), false)
}

pub(crate) fn lower_fusion_plan_inner(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    registry: Option<&ScalarOpRegistry>,
    profile: &IsaProfile,
    hook: Option<&dyn super::isa_hook::IsaHook>,
    rope_req: Option<&RopeCacheRequirement>,
    ple_req: Option<&PleScratchRequirement>,
    dwc_req: Option<&DwcScratchRequirement>,
    debug_jit: bool,
) -> Result<VmProgram, CompilerError> {
    let width = profile.optimal_simd_width();
    // ARCH-CODEGEN-DISPATCH: 按 platform 选择 ABI SymDimSlotMap
    let sym_map = match &profile.platform {
        super::isa_profile::Platform::X86_64 { .. } | super::isa_profile::Platform::AArch64 { .. } => {
            SymDimSlotMap::default_abi()
        }
        super::isa_profile::Platform::Cuda { .. }
        | super::isa_profile::Platform::Hip { .. }
        | super::isa_profile::Platform::Metal { .. } => {
            SymDimSlotMap::gpu_abi()
        }
    };

    let ctx = LoweringContext {
        width,
        dtype: graph_dtype(graph),
        sym_map: &sym_map,
        registry,
        hook,
        budget: None,
        rope_req,
        ple_req,
        dwc_req,
        exec_pattern: None,
        bottleneck_map: None,
        virtual_activation: None,
        parallelism: Some(ParallelismDesc::SimdVectorize {
            element_width: width.f32_lanes().max(1),
            unroll_factor: profile.k_unroll_factor,
        }),
        virtual_tensor_map: None,
        layout: None,
        page_size: 0,
        dot_cap: profile.dot_cap,
        batch_ctx_ptr: None,
        debug_jit,
    };

    let mut prog = VmProgram::new();

    // ARCH-LOADPTR-ORDER: 所有 AbiArg 源的 LoadPtr 必须先 emit，StackArg/其他源的 LoadPtr 后 emit。
    // 原因: AbiArg 直接从物理 ABI 寄存器读取 (rdi/rsi/rdx/rcx/r8/r9)，这些寄存器同时是 GPR 池成员，
    // RegAllocator 可能在其他 VReg 的 LoadPtr StackArg 时把它们当 dst 使用（如 `mov rsi, [rbp+0x18]`），
    // 从而破坏尚未消费的 AbiArg 值。重排保证 AbiArg → 目标 VReg 的 mov 先执行，释放 ABI 寄存器供后续复用。
    //
    // 按需分配，只分配实际使用的 VReg。
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    let needs_weight = plan.groups.iter().any(|g| {
        graph.op(g.anchor).is_some_and(|op| op.inputs.len() > 1)
    });
    let weight_ptr = if needs_weight {
        prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar)
    } else {
        input_ptr
    };
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // Phase 1: AbiArg 源 (input, weight) — 读取物理 ABI 寄存器
    prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: ctx.sym_map.resolve("input").cloned().expect("ABI: input") });
    if needs_weight {
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: ctx.sym_map.resolve("weights").cloned().expect("ABI: weights") });
    }

    // Phase 2: StackArg/其他源 (output, scratchpad)
    prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: ctx.sym_map.resolve("output").cloned().expect("ABI: output") });

    // scratchpad_ptr: 在需要 scratch 的融合模式（NormIntoGemm/QkvSharedInput/FFNBlock/
    // TileLevelFusion/ComputeRoot/CrossLayerResidual/FusedQkvNormRope）使用。
    // ARCH-DATA-FLOW-CONTRACT §3.1: Standalone/LoopFusion 组的 op 若 input/output
    // 是图内 intermediate (非 graph.inputs/outputs), 需要从 scratchpad 读写。
    // 因此当图存在 intermediate 张量 (alloc.total_bytes > 0) 时一律加载 scratchpad。
    let needs_scratch = alloc.total_bytes > 0 || ple_req.is_some() || dwc_req.is_some() || plan.groups.iter().any(|g| matches!(&g.mode,
        FusionMode::NormIntoGemm
        | FusionMode::QkvSharedInput
        | FusionMode::FFNBlock { .. }
        | FusionMode::TileLevelFusion { .. }
        | FusionMode::ComputeRoot { .. }
        | FusionMode::CrossLayerResidual { .. }
        | FusionMode::FusedQkvNormRope { .. }
    ));
    let scratch_base = if needs_scratch {
        let sp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: sp,
            src: ctx.sym_map.resolve("scratchpad").cloned().expect("ABI: scratchpad"),
        });
        sp
    } else {
        input_ptr
    };

    // ARCH-DATA-FLOW-CONTRACT §3 (D#1 统一根治):
    // 先前 group_input_ptr / group_weight_ptr / group_output_ptr 在三处 ad-hoc
    // 判断 tensor 的物理位置(Activation/Weight/Intermediate/Output),每处缺
    // 覆盖就出 bug(multi-output 覆写、gemm_k 读 w_q、SwiGLU Mul 读 weight_blob
    // 起点、rope_q 读原 activation 等)。统一收敛到 TensorPtrResolver,一次
    // 建表,每个 op 按 tensor_id 直接 materialize。
    let resolver = TensorPtrResolver::build(graph, alloc);
    let original_weight_vreg = if needs_weight { Some(weight_ptr) } else { None };
    let mut current_abi = AbiPtrs {
        input_ptr,
        weight_ptr: original_weight_vreg,
        weight_abi_expr: if needs_weight {
            Some(ctx.sym_map.resolve("weights").cloned().expect("ABI: weights"))
        } else {
            None
        },
        output_ptr,
        scratch_ptr: if needs_scratch { Some(scratch_base) } else { None },
        gen_loop_counter: None,
        layer_loop_counter: None,
        mega_decode_seq_len: None,
        hook_ctx_ptr: None,
        sg_detect_scratch_offset: None,
        sg_knowledge_scratch_offset: None,
        callback_table_ptr: None,
        page_table_ptr: None,
        kv_load_mode: graph.kv_load_mode,
        kv_cache_ptr: None,
        activation_ping_ptr: None,
        activation_pong_ptr: None,
    };

    emit_fusion_groups(
        &mut prog, plan, graph, alloc, &ctx,
        rope_req.map(|r| r.cache_offset),
        &mut current_abi, original_weight_vreg, &resolver,
    )?;

    prog.validate_structure().map_err(CompilerError::CodegenViolation)?;
    Ok(prog)
}

/// Process all fusion groups in a FusionPlan, emitting VmInstrs for each.
///
/// Handles layer loop entry/exit when `graph.layer_loop_config` is set:
/// ops whose anchor has label starting with "layer." are wrapped in a
/// LoopBegin/LoopEnd pair that strides the weight pointer.
///
/// This function is the shared core between `lower_fusion_plan_inner()`
/// (per-layer compilation) and `compile_mega_kernel_vm()` (whole-model
/// single-function compilation).
pub(super) fn emit_fusion_groups(
    prog: &mut VmProgram,
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    ctx: &LoweringContext,
    rope_cache_offset: Option<usize>,
    current_abi: &mut AbiPtrs,
    original_weight_vreg: Option<VRegId>,
    resolver: &TensorPtrResolver,
) -> Result<(), CompilerError> {
    let width = ctx.width;
    let dtype = ctx.dtype;
    let sym_map = ctx.sym_map;
    let registry = ctx.registry;
    let hook = ctx.hook;
    let rope_req = ctx.rope_req;
    let ple_req = ctx.ple_req;
    let dwc_req = ctx.dwc_req;
    let layer_loop_cfg = graph.layer_loop_config.as_ref();
    let hetero_loop_cfg = graph.hetero_layer_loop_config.as_ref();

    // ── 异构模型预编译: 并行编译 4 种层类型模板 ──
    if hetero_loop_cfg.is_some() {
        let _boundaries = compile_hetero_templates_parallel(
            ctx, plan, graph, alloc, resolver,
        )?;
        // TODO: 用预编译的模板替代逐 group emit — Phase D 深化
    }

    let mut state = EmitState {
        abi: current_abi.clone(),
        hetero_phase: HeteroPhase::BeforeLayers,
        in_layer_loop: false,
        hetero_seg_byte_offset: None,
        hetero_seg_weight_base: None,
        active_guard: LayerCondition::Always,
        guard_skip_patch: None,
    };

    // Alias ABI ptrs for convenient access inside the loop.
    let input_ptr = state.abi.input_ptr;
    let weight_ptr = state.abi.weight_ptr.unwrap_or(input_ptr);
    let output_ptr = state.abi.output_ptr;
    let scratch_base = state.abi.scratch_ptr.unwrap_or(input_ptr);
    // Mega-kernel: override Symbolic seq_len bound with DynamicVReg(decode_seq_len).
    let seq_bound_override: Option<BoundExpr> = state.abi.mega_decode_seq_len.map(BoundExpr::DynamicVReg);

    // §0.2.8 ActivationSwap: 预分配 ping-pong buffer 指针 VReg
    // buffer_alloc 在 scratch 中分配了 ping (offset 0) 和 pong (offset ping_size) 两个 slot。
    // ping_ptr = scratch_base + 0, pong_ptr = scratch_base + ping_size。
    // 每层迭代末尾 ActivationSwap 交换 ptr 值，下一层的 input/output 自动切换。
    let activation_swap_vregs: Option<(VRegId, VRegId)> = if layer_loop_cfg.is_some() || hetero_loop_cfg.is_some() {
        // 从 alloc 中查找 ping/pong sentinel slot 的 offset
        let ping_offset = alloc.slots.iter()
            .find(|s| s.tensor_id.0 == 0xFFFF_FF00)
            .map(|s| s.offset);
        let pong_offset = alloc.slots.iter()
            .find(|s| s.tensor_id.0 == 0xFFFF_FF01)
            .map(|s| s.offset);
        if let (Some(ping_off), Some(pong_off)) = (ping_offset, pong_offset) {
            let ping_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let pong_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            // ping/pong buffers are at scratch_base + offset (pointer arithmetic, NOT dereference).
            // scratchpad is zero-initialized — LoadPtr would read NULL from it.
            prog.emit(VmInstr::AddPtr { dst: ping_ptr, base: scratch_base, offset: ping_off });
            prog.emit(VmInstr::AddPtr { dst: pong_ptr, base: scratch_base, offset: pong_off });
            // 设置到 current_abi 供 resolver materialize 使用
            current_abi.activation_ping_ptr = Some(ping_ptr);
            current_abi.activation_pong_ptr = Some(pong_ptr);
            eprintln!("[PING-PONG] enabled: ping_ptr=v{} (off={}), pong_ptr=v{} (off={})",
                ping_ptr.0, ping_off, pong_ptr.0, pong_off);
            Some((ping_ptr, pong_ptr))
        } else {
            // No sentinel slots allocated (e.g. no layer_loop_config.activation_alias)
            eprintln!("[PING-PONG] skipped: no sentinel slots found in buffer allocation");
            None
        }
    } else {
        None
    };

    for (gi, group) in plan.groups.iter().enumerate() {
        let anchor_op = graph.op(group.anchor).ok_or_else(|| {
            CompilerError::CodegenViolation(format!("anchor op {:?} not found", group.anchor))
        })?;

        // Mega-kernel: skip ops that are handled by Phase 5-7 (Argmax, StoreToken,
        // CheckStopCondition, WriteLogits). Processing them here would emit duplicate
        // instructions with WRONG ABI parameters (CompiledLayerFn offsets instead of
        // MegaKernelFn offsets), causing memory corruption and wrong control flow.
        if matches!(anchor_op.kind,
            OpKind::Argmax { .. } | OpKind::StoreToken | OpKind::CheckStopCondition
            | OpKind::WriteLogits { .. } | OpKind::MtpDraft { .. }
        ) {
            continue;
        }

        let is_sliding_small_op = anchor_op.label.starts_with("layer_sliding_small.");
        let is_full_small_op = anchor_op.label.starts_with("layer_full_small.");
        let is_sliding_large_op = anchor_op.label.starts_with("layer_sliding_large.");
        let is_full_large_op = anchor_op.label.starts_with("layer_full_large.");
        let is_sliding_op = is_sliding_small_op || is_sliding_large_op;
        let is_full_op = is_full_small_op || is_full_large_op;
        let is_layer_op = anchor_op.label.starts_with("layer.") || is_sliding_op || is_full_op;
        let group_op_labels: Vec<String> = group.ops.iter()
            .filter_map(|&oid| graph.op(oid).map(|o| o.label.clone()))
            .collect();
        eprintln!("[LAYER-DETECT] gi={gi} label='{}' is_layer={is_layer_op} ops=[{}] mode={:?}", anchor_op.label, group_op_labels.join(", "), group.mode);

        // ── Heterogeneous layer loop handling (4-type: sliding/full × small/large FFN) ──
        //
        // Gemma-4 E2B: 7 segments × [4 sliding + 1 full] = 35 layers.
        // Segments 0-2 use small FFN (ss+fs templates), segments 3-6 use large FFN (sl+fl).
        //
        // JIT structure:
        //   LoopBegin(num_small_segments=3, step=small_seg_stride)  ── outer small loop
        //     LoopBegin(sliding_per_segment=4, step=ss_stride)      ── inner sliding loop
        //       ss ops
        //     LoopEnd
        //     fs full body ops
        //   LoopEnd                                                  ── outer small end
        //   LoopBegin(num_large_segments=4, step=large_seg_stride)  ── outer large loop
        //     LoopBegin(sliding_per_segment=4, step=sl_stride)      ── inner sliding loop
        //       sl ops
        //     LoopEnd
        //     fl full body ops
        //   LoopEnd                                                  ── outer large end
        if let Some(hcfg) = hetero_loop_cfg {
            let num_small_segs = hcfg.large_ffn_start_segment;
            let num_large_segs = hcfg.num_segments - num_small_segs;

            // ── Phase 1: Small segment entry (ss ops) ──
            if is_sliding_small_op && state.hetero_phase == HeteroPhase::BeforeLayers {
                // Outer loop for small segments
                let seg_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                let seg_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::LoopBegin {
                    counter: seg_counter,
                    byte_offset: seg_byte_off,
                    bound: BoundExpr::Const(num_small_segs),
                    step_bytes: hcfg.small_segment_stride,
                });
                // Compute segment weight base = weight_ptr + layer_blob_base_offset + seg_byte_off
                let seg_base_tmp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: seg_base_tmp, value: hcfg.layer_blob_base_offset });
                let seg_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seg_base, a: weight_ptr, b: GprOperand::VReg(seg_base_tmp ), op: GprOp::Add });
                let seg_wb = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seg_wb, a: seg_base, b: GprOperand::VReg(seg_byte_off ), op: GprOp::Add });
                // Save for Phase 2 (full layer base calculation within the outer loop)
                state.hetero_seg_byte_offset = Some(seg_byte_off);
                state.hetero_seg_weight_base = Some(seg_wb);
                // Inner sliding loop
                let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::LoopBegin {
                    counter, byte_offset,
                    bound: BoundExpr::Const(hcfg.sliding_per_segment),
                    step_bytes: hcfg.sliding_small_stride,
                });
                let layer_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: layer_weight_base, a: seg_wb, b: GprOperand::VReg(byte_offset ), op: GprOp::Add });
                state.abi.weight_ptr = Some(layer_weight_base);
                state.abi.layer_loop_counter = Some(counter);
                state.in_layer_loop = true;
                state.hetero_phase = HeteroPhase::InSlidingLoop;
            }
            // ── Phase 2: ss → fs transition (sliding → full within small segment) ──
            if is_full_small_op && state.hetero_phase == HeteroPhase::InSlidingLoop {
                prog.emit(VmInstr::LoopEnd); // end inner sliding loop
                state.in_layer_loop = false;
                // Full layer base = seg_weight_base + sliding_per_segment * ss_stride
                let full_off = hcfg.sliding_per_segment * hcfg.sliding_small_stride;
                let seg_wb = state.hetero_seg_weight_base.expect("seg_weight_base not set");
                let full_off_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: full_off_gpr, value: full_off });
                let full_base_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: full_base_ptr, a: seg_wb, b: GprOperand::VReg(full_off_gpr ), op: GprOp::Add });
                state.abi.weight_ptr = Some(full_base_ptr);
                state.abi.layer_loop_counter = None;
                state.hetero_phase = HeteroPhase::InFullBody;
            }
            // ── Phase 3: fs → sl transition (end small outer loop, start large outer loop) ──
            if is_sliding_large_op && state.hetero_phase == HeteroPhase::InFullBody {
                prog.emit(VmInstr::LoopEnd); // end outer small segment loop
                // Start outer large segment loop
                let seg_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                let seg_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::LoopBegin {
                    counter: seg_counter,
                    byte_offset: seg_byte_off,
                    bound: BoundExpr::Const(num_large_segs),
                    step_bytes: hcfg.large_segment_stride,
                });
                // Large segments start after all small segments
                let large_base_start = num_small_segs * hcfg.small_segment_stride;
                let seg_base_tmp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: seg_base_tmp, value: hcfg.layer_blob_base_offset + large_base_start });
                let seg_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seg_base, a: weight_ptr, b: GprOperand::VReg(seg_base_tmp ), op: GprOp::Add });
                let seg_wb = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: seg_wb, a: seg_base, b: GprOperand::VReg(seg_byte_off ), op: GprOp::Add });
                // Save for Phase 4
                state.hetero_seg_byte_offset = Some(seg_byte_off);
                state.hetero_seg_weight_base = Some(seg_wb);
                // Inner sliding loop
                let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                prog.emit(VmInstr::LoopBegin {
                    counter, byte_offset,
                    bound: BoundExpr::Const(hcfg.sliding_per_segment),
                    step_bytes: hcfg.sliding_large_stride,
                });
                let layer_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: layer_weight_base, a: seg_wb, b: GprOperand::VReg(byte_offset ), op: GprOp::Add });
                state.abi.weight_ptr = Some(layer_weight_base);
                state.abi.layer_loop_counter = Some(counter);
                state.in_layer_loop = true;
                state.hetero_phase = HeteroPhase::InLargeSlidingLoop;
            }
            // ── Phase 4: sl → fl transition ──
            if is_full_large_op && state.hetero_phase == HeteroPhase::InLargeSlidingLoop {
                prog.emit(VmInstr::LoopEnd); // end inner sliding loop
                state.in_layer_loop = false;
                let full_off = hcfg.sliding_per_segment * hcfg.sliding_large_stride;
                let seg_wb = state.hetero_seg_weight_base.expect("seg_weight_base not set for large");
                let full_off_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprLoadImm { dst: full_off_gpr, value: full_off });
                let full_base_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: full_base_ptr, a: seg_wb, b: GprOperand::VReg(full_off_gpr ), op: GprOp::Add });
                state.abi.weight_ptr = Some(full_base_ptr);
                state.abi.layer_loop_counter = None;
                state.hetero_phase = HeteroPhase::InLargeFullBody;
            }
            // ── Phase 5: End of all layer ops ──
            if !is_layer_op && matches!(state.hetero_phase,
                HeteroPhase::InSlidingLoop | HeteroPhase::InFullBody
                | HeteroPhase::InLargeSlidingLoop | HeteroPhase::InLargeFullBody
            ) {
                if state.in_layer_loop {
                    prog.emit(VmInstr::LoopEnd);
                }
                // End the outer segment loop
                prog.emit(VmInstr::LoopEnd);
                state.abi.weight_ptr = original_weight_vreg;
                state.abi.layer_loop_counter = None;
                state.in_layer_loop = false;
                state.hetero_phase = HeteroPhase::Done;
            }
        } else {
            // ── Standard (homogeneous) layer loop ──
            // ── Layer loop entry: emit LoopBegin + compute layer_weight_base ──
            if is_layer_op && !state.in_layer_loop {
                eprintln!("[LOOP-CHECK] gi={gi} label='{}' is_layer_op={is_layer_op} in_layer_loop={} layer_loop_cfg={}",
                    anchor_op.label, state.in_layer_loop, layer_loop_cfg.is_some());
                if let Some(cfg) = layer_loop_cfg {
                    eprintln!("[LOOP-EMIT] num_layers={} weight_stride={} step_bytes={}", cfg.num_layers, cfg.weight_stride, cfg.weight_stride);
                    let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
                    let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
                    // DEBUG: configurable layer count via GLLM_DEBUG_LAYERS env var
                    let layer_bound = if let Ok(n) = std::env::var("GLLM_DEBUG_LAYERS") {
                        if let Ok(count) = n.parse::<usize>() {
                            eprintln!("[DEBUG-LAYERS] Overriding num_layers {} -> {}", cfg.num_layers, count);
                            BoundExpr::Const(count)
                        } else {
                            BoundExpr::Const(cfg.num_layers)
                        }
                    } else if std::env::var("GLLM_SINGLE_LAYER").is_ok() {
                        eprintln!("[SINGLE-LAYER] Overriding num_layers {} -> 1", cfg.num_layers);
                        BoundExpr::Const(1)
                    } else {
                        BoundExpr::Const(cfg.num_layers)
                    };

                    // §0.2.8 Cross-layer weight prefetch: allocate shared memory for
                    // prefetching next layer's weight tile. GPU-only: CPU backends
                    // do not have shared memory or async DMA.
                    let is_gpu = matches!(width, SimdWidth::Warp(_));
                    if is_gpu {
                        let smem_weight_prefetch_name = "smem_w_prefetch";
                        let weight_prefetch_size = cfg.weight_stride;
                        prog.emit(VmInstr::SharedMemAlloc {
                            name: smem_weight_prefetch_name.to_string(),
                            bytes: weight_prefetch_size,
                        });
                    }

                    prog.emit(VmInstr::LoopBegin {
                        counter, byte_offset,
                        bound: layer_bound,
                        step_bytes: cfg.weight_stride,
                    });

                    // Reload weight_ptr from ABI stack slot on every iteration.
                    // Under extreme register pressure, weight_ptr's spill slot may be
                    // overwritten during the loop body. Reloading from the ABI stack
                    // slot (callee-save, never touched by regalloc) guarantees a
                    // correct base pointer for weight offset computation.
                    let fresh_weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr {
                        dst: fresh_weight,
                        src: ctx.sym_map.resolve("weights").cloned().expect("ABI: weights"),
                    });

                    // §0.2.8 WeightPrefetchWait: wait for the prefetch issued at the
                    // end of the *previous* iteration. GPU-only.
                    if is_gpu {
                        prog.emit(VmInstr::WeightPrefetchWait { group: 0 });
                    }

                    // layer_weight_base = fresh_weight + byte_offset
                    // Do NOT add layer_blob_base_offset here — ops' graph offsets already include it.
                    let layer_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::GprBinOp { dst: layer_weight_base, a: fresh_weight, b: GprOperand::VReg(byte_offset), op: GprOp::Add });
                    state.abi.weight_ptr = Some(layer_weight_base);
                    state.abi.layer_loop_counter = Some(counter);
                    state.in_layer_loop = true;
                }
            }

            // ── Layer loop exit: emit ActivationSwap + LoopEnd, adjust weight_ptr for globals ──
            if !is_layer_op && state.in_layer_loop {
                // §0.2.8 ActivationSwap: 每层迭代末尾交换 ping-pong buffer 指针
                if let Some((ping, pong)) = activation_swap_vregs {
                    prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
                }
                // §0.2.8 WeightPrefetchAsync: issue async load of the NEXT layer's
                // weights. This fires at the end of each iteration so the DMA
                // transfer overlaps with the LoopEnd → LoopBegin back-edge.
                // The next iteration's WeightPrefetchWait will synchronize.
                // On the last iteration, the prefetch reads past the weight blob
                // but the data is never consumed — WeightPrefetchWait on the
                // post-loop path is never reached.
                // §0.2.8 WeightPrefetchAsync: GPU-only — CPU backends lack
                // shared memory and async DMA engines.
                if matches!(width, SimdWidth::Warp(_)) {
                    if let Some(cfg) = layer_loop_cfg.as_ref() {
                        let wp = state.abi.weight_ptr.unwrap_or(weight_ptr);
                        prog.emit(VmInstr::WeightPrefetchAsync {
                            smem_name: "smem_w_prefetch".to_string(),
                            weight_base: wp,
                            weight_offset: cfg.weight_stride,
                            size: cfg.weight_stride,
                        });
                    }
                }
                prog.emit(VmInstr::LoopEnd);
                // After the layer loop, reload weight base from ABI args.
                // original_weight_vreg's spill slot may have been overwritten by the
                // register allocator during the multi-iteration layer loop. Reloading
                // from the ABI stack slot guarantees the correct weight_blob base.
                // Global weights (final_norm, lm_head, embed) are packed at the
                // beginning of the blob, before layer template weights. Their graph
                // offsets are absolute from blob start, so weight_ptr must point to
                // blob offset 0.
                let fresh_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: fresh_weight_base,
                    src: ctx.sym_map.resolve("weights").cloned().expect("ABI: weights"),
                });
                state.abi.weight_ptr = Some(fresh_weight_base);
                state.abi.layer_loop_counter = None;
                state.in_layer_loop = false;
            }
        }

        // Sync current_abi from state.abi — layer loop setup mutates state.abi.weight_ptr.
        // resolver.materialize() reads from current_abi. Without this sync, the resolver
        // emits LoadPtr using stale VRegs.
        current_abi.weight_ptr = state.abi.weight_ptr;
        current_abi.scratch_ptr = state.abi.scratch_ptr;
        current_abi.layer_loop_counter = state.abi.layer_loop_counter;
        current_abi.gen_loop_counter = state.abi.gen_loop_counter;
        current_abi.kv_cache_ptr = state.abi.kv_cache_ptr;

        // ARCH-DATA-FLOW-CONTRACT §3 (D#1 统一根治):
        // group 内 anchor_op 的 input[0] / input[1] / output[0] 统一经
        // TensorPtrResolver 查询, 物理位置由建表阶段一次性决定 (Activation /
        // Weight / Intermediate / Output), 每处按 tensor_id 取真实 base+offset。
        let group_input_ptr = anchor_op.inputs.first()
            .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
            .unwrap_or(input_ptr);
        let group_weight_ptr = anchor_op.inputs.get(1)
            .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
            .unwrap_or(weight_ptr);
        let group_output_ptr = anchor_op.outputs.first()
            .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
            .unwrap_or(output_ptr);

        // ── Layer guard (NO_LAYER_EXPAND, SPEC 03 §1.3.1) ──
        // Emit GprCondAction to conditionally skip ops based on layer_idx.
        // Consecutive ops with the same guard are merged into a single skip range.
        let op_guard = anchor_op.guard;
        if op_guard != state.active_guard {
            // Close previous guard run (patch-back Skip count)
            if let Some(patch_idx) = state.guard_skip_patch.take() {
                let skip_n = prog.instrs.len() - patch_idx - 1;
                if let VmInstr::GprCondAction {
                    action: GprBranchAction::Skip(ref mut n), ..
                } = prog.instrs[patch_idx] {
                    *n = skip_n;
                }
            }
            state.active_guard = op_guard;

            // Open new guard run if non-Always and inside layer loop
            if op_guard != LayerCondition::Always && state.in_layer_loop {
                let counter = state.abi.layer_loop_counter
                    .expect("guarded op requires active layer loop");
                let skip_cond = match op_guard {
                    LayerCondition::LayerIdxLt(t) => {
                        // Guard = "donor executes" → skip when consumer (idx >= t)
                        GprCondition::CmpGeU(counter, t as u64)
                    }
                    LayerCondition::LayerIdxGe(t) => {
                        // Guard = "consumer executes" → skip when donor (idx < t)
                        GprCondition::CmpLtU(counter, t as u64)
                    }
                    LayerCondition::Always => unreachable!(),
                };
                prog.emit(VmInstr::GprCondAction {
                    cond: skip_cond,
                    action: GprBranchAction::Skip(0),
                });
                state.guard_skip_patch = Some(prog.instrs.len() - 1);
            }
        }

        // §4 CompoundExecution: 先按 FusionMode dispatch，再按 OpKind
        emit_fusion_group_by_mode(
            prog, group, anchor_op, graph, alloc, ctx,
            group_input_ptr, group_weight_ptr, group_output_ptr,
            scratch_base, input_ptr, weight_ptr, output_ptr,
            rope_cache_offset, seq_bound_override.as_ref(),
            resolver, current_abi,
        )?;
    }

    // Close any pending guard run after all groups processed
    if let Some(patch_idx) = state.guard_skip_patch.take() {
        let skip_n = prog.instrs.len() - patch_idx - 1;
        if let VmInstr::GprCondAction {
            action: GprBranchAction::Skip(ref mut n), ..
        } = prog.instrs[patch_idx] {
            *n = skip_n;
        }
    }

    // Close layer loop if still open (all groups were layer ops)
    if state.in_layer_loop {
        // §0.2.8 ActivationSwap: 最终层迭代末尾交换 ping-pong buffer
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        // §0.2.8 WeightPrefetchAsync: GPU-only — CPU backends lack
        // shared memory and async DMA engines.
        if matches!(width, SimdWidth::Warp(_)) {
            if let Some(cfg) = layer_loop_cfg.as_ref() {
                let wp = state.abi.weight_ptr.unwrap_or(weight_ptr);
                prog.emit(VmInstr::WeightPrefetchAsync {
                    smem_name: "smem_w_prefetch".to_string(),
                    weight_base: wp,
                    weight_offset: cfg.weight_stride,
                    size: cfg.weight_stride,
                });
            }
        }
        prog.emit(VmInstr::LoopEnd);
        // After the layer loop, reset weight_ptr to original (offset 0).
        // Global weights are at the beginning of the blob with absolute offsets.
        state.abi.weight_ptr = original_weight_vreg;
    }

    // Write back mutated ABI state to caller.
    *current_abi = state.abi;

    Ok(())
}

/// Elementwise 内联 (ARCH-SYMDIM-NO-CONST-DEGRADE): 返回 acc VRegId。
///
/// `output_shape`: 输出张量的完整 SymDim 形状。
/// 外层 Symbolic 维度 → BoundExpr::Symbolic（运行时 seq_len）。
/// 内层 Concrete 维度 → BoundExpr::Const（编译时固定）。
pub(crate) fn emit_elementwise_inline(
    prog: &mut VmProgram,
    body: &[TraceOp],
    output_shape: &[SymDim],
    width: SimdWidth,
    is_binary: bool,
    weight_is_broadcast: bool,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    sym_map: &SymDimSlotMap,
    seq_bound_override: Option<&BoundExpr>,
    dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    let lanes = width.f32_lanes();
    if lanes == 0 {
        return Err(CompilerError::CodegenViolation("zero lanes".into()));
    }

    // 计算总元素的循环结构:
    // 外层: seq 维度 (Symbolic → BoundExpr::Symbolic)
    // 内层: feature 维度 (Concrete → BoundExpr::Const, 向量化)
    let feature_dim: usize = output_shape.iter()
        .filter(|d| !d.is_symbolic())
        .map(|d| d.as_concrete().expect("ARCH-SYMDIM-OUTER-ONLY: inner dim must be Concrete after is_symbolic filter"))
        .product::<usize>()
        .max(1);
    let feature_vecs = feature_dim / lanes;
    let step_bytes = width.bytes();
    // ARCH-DATA-FLOW-CONTRACT §2.3: 行字节数来自 row_stride_bytes，不手工乘 elem
    let row_bytes = feature_dim * dtype.elem_bytes();

    // 是否有 Symbolic 外层维度
    let outer_sym = output_shape.iter().find(|d| d.is_symbolic());

    let acc = prog.alloc_vreg(VRegKind::Vec, width);
    let sec = if is_binary { Some(prog.alloc_vreg(VRegKind::Vec, width)) } else { None };

    if let Some(sym_dim) = outer_sym {
        // 二层循环: 外层 Symbolic (seq_len)，内层 Const (feature_dim/lanes)
        // 外层 byte_offset 定位到行首 → LoadPtr 计算行基地址
        // 内层 byte_offset 是列内偏移 → 直接用 LoopOffset（无嵌套 Add）
        let outer_bound = seq_bound_override.cloned().unwrap_or_else(|| sym_map.to_bound(sym_dim));
        let row_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let row_weight = if is_binary { prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar) } else { input_ptr };
        let row_output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit_loop(outer_bound, row_bytes, |prog, _row_ctr, row_off| {
            // 行基地址 = base_ptr + row_off (VM 自动跟踪的 byte_offset)
            prog.emit(VmInstr::LoadPtr { dst: row_input, src: PtrExpr::VRegPlusVReg(input_ptr, row_off) });
            if is_binary {
                // ARCH-BROADCAST: 广播维度上 weight 始终指向第 0 行，不加 row_off
                let weight_src = if weight_is_broadcast {
                    PtrExpr::VRegPlusConst(weight_ptr, 0)
                } else {
                    PtrExpr::VRegPlusVReg(weight_ptr, row_off)
                };
                prog.emit(VmInstr::LoadPtr { dst: row_weight, src: weight_src });
            }
            prog.emit(VmInstr::LoadPtr { dst: row_output, src: PtrExpr::VRegPlusVReg(output_ptr, row_off) });

            // 内层: 列向量化循环，偏移从行首开始
            if feature_vecs > 0 {
                prog.emit_loop(BoundExpr::Const(feature_vecs), step_bytes, |prog, _ctr, col_off| {
                    prog.emit(VmInstr::VecLoad {
                        dst: acc, base: row_input, offset: OffsetExpr::LoopOffset(col_off), width,
                        dtype,
                    });
                    if let Some(s) = sec {
                        prog.emit(VmInstr::VecLoad {
                            dst: s, base: row_weight, offset: OffsetExpr::LoopOffset(col_off), width,
                            dtype,
                        });
                    }
                    lower::lower_trace_body_compat(prog, body, acc, sec, width)
                        .expect("lower_trace_body: OpTrace invariant violation");
                    prog.emit(VmInstr::VecStore {
                        base: row_output, offset: OffsetExpr::LoopOffset(col_off), src: acc, width,
                        dtype,
                    });
                });
            }
            // Scalar tail: feature_dim % lanes != 0 的剩余列。典型场景: N=1 的
            // classifier_out_proj bias Add (feature_dim=1, lanes=8, feature_vecs=0)
            // → 纯 tail。原 code 只在 feature_vecs > 0 时 emit 主循环, tail 被忽略
            // → output 全 0 → classifier rerank_logit = 0。
            let tail = feature_dim - feature_vecs * lanes;
            if tail > 0 {
                let elem = dtype.elem_bytes();
                let tail_base_bytes = feature_vecs * step_bytes;
                let s_width = SimdWidth::Scalar;
                let s_acc = prog.alloc_vreg(VRegKind::Vec, s_width);
                let s_sec = if is_binary { Some(prog.alloc_vreg(VRegKind::Vec, s_width)) } else { None };
                for t in 0..tail {
                    let col_off_const = tail_base_bytes + t * elem;
                    prog.emit(VmInstr::VecLoad {
                        dst: s_acc, base: row_input, offset: OffsetExpr::Const(col_off_const), width: s_width,
                        dtype,
                    });
                    if let Some(s) = s_sec {
                        prog.emit(VmInstr::VecLoad {
                            dst: s, base: row_weight, offset: OffsetExpr::Const(col_off_const), width: s_width,
                            dtype,
                        });
                    }
                    lower::lower_trace_body_compat(prog, body, s_acc, s_sec, s_width)
                        .expect("lower_trace_body: OpTrace invariant violation (scalar tail)");
                    prog.emit(VmInstr::VecStore {
                        base: row_output, offset: OffsetExpr::Const(col_off_const), src: s_acc, width: s_width,
                        dtype,
                    });
                }
            }
        });
    } else {
        // 单层循环: 全 Concrete (编译时已知总元素数)
        let total_vecs = feature_dim / lanes; // 所有维度都 Concrete
        if total_vecs > 0 {
            prog.emit_loop(BoundExpr::Const(total_vecs), step_bytes, |prog, _counter, byte_off| {
                prog.emit(VmInstr::VecLoad {
                    dst: acc, base: input_ptr,
                    offset: OffsetExpr::LoopOffset(byte_off), width,
                    dtype,
                });
                if let Some(s) = sec {
                    prog.emit(VmInstr::VecLoad {
                        dst: s, base: weight_ptr,
                        offset: OffsetExpr::LoopOffset(byte_off), width,
                        dtype,
                    });
                }
                lower::lower_trace_body_compat(prog, body, acc, sec, width)
                    .expect("lower_trace_body: OpTrace invariant violation");
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: OffsetExpr::LoopOffset(byte_off), src: acc, width,
                    dtype,
                });
            });
        }
        // Scalar tail: feature_dim % lanes 的剩余元素 (全 Concrete 路径)
        let tail = feature_dim - total_vecs * lanes;
        if tail > 0 {
            let elem = dtype.elem_bytes();
            let tail_base_bytes = total_vecs * step_bytes;
            let s_width = SimdWidth::Scalar;
            let s_acc = prog.alloc_vreg(VRegKind::Vec, s_width);
            let s_sec = if is_binary { Some(prog.alloc_vreg(VRegKind::Vec, s_width)) } else { None };
            for t in 0..tail {
                let col_off_const = tail_base_bytes + t * elem;
                prog.emit(VmInstr::VecLoad {
                    dst: s_acc, base: input_ptr,
                    offset: OffsetExpr::Const(col_off_const), width: s_width,
                    dtype,
                });
                if let Some(s) = s_sec {
                    prog.emit(VmInstr::VecLoad {
                        dst: s, base: weight_ptr,
                        offset: OffsetExpr::Const(col_off_const), width: s_width,
                        dtype,
                    });
                }
                lower::lower_trace_body_compat(prog, body, s_acc, s_sec, s_width)
                    .expect("lower_trace_body: OpTrace invariant violation (scalar tail)");
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: OffsetExpr::Const(col_off_const),
                    src: s_acc, width: s_width,
                    dtype,
                });
            }
        }
    }
    Ok(acc)
}

