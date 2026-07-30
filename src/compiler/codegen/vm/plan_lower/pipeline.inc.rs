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
    let topology = super::topology::GraphTopologyAnalysis::analyze(graph);
    lower_fusion_plan_inner_with_sym_map(
        plan, graph, alloc, registry, profile, hook,
        rope_req, ple_req, dwc_req, debug_jit, None, &topology,
    )
}

pub(crate) fn lower_fusion_plan_inner_with_sym_map(
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
    sym_map_override: Option<&SymDimSlotMap>,
    topology: &super::topology::GraphTopologyAnalysis,
) -> Result<VmProgram, CompilerError> {
    let width = profile.optimal_simd_width();
    // ARCH-CODEGEN-DISPATCH: 按 platform 选择 ABI SymDimSlotMap
    let owned_sym_map;
    let sym_map = match sym_map_override {
        Some(override_map) => override_map,
        None => {
            owned_sym_map = match &profile.platform {
                super::isa_profile::Platform::X86_64 { .. } | super::isa_profile::Platform::AArch64 { .. } => {
                    SymDimSlotMap::mega_kernel_abi()
                }
                super::isa_profile::Platform::Cuda { .. }
                | super::isa_profile::Platform::Hip { .. }
                | super::isa_profile::Platform::Metal { .. } => {
                    SymDimSlotMap::gpu_abi()
                }
            };
            &owned_sym_map
        }
    };

    let sess = CompileSession {
        width,
        sym_map,
        registry,
        hook,
        feature_set: profile.feature_set(),
        budget: None,
        page_size: 0,
        dot_cap: profile.dot_cap,
        kv_elem_bytes: kv_cache_elem_bytes(graph),
        debug_jit,
        virtual_activation: None,
        virtual_tensor_map: None,
        layout: None,
        batch_ctx_ptr: None,
    };
    let ctx = LoweringContext {
        session: &sess,
        accum_dtype: graph_dtype(graph),
        rope_req,
        ple_req,
        dwc_req,
        exec_pattern: None,
        bottleneck_map: None,
        parallelism: Some(ParallelismDesc::SimdVectorize {
            element_width: width.f32_lanes().max(1),
            unroll_factor: profile.k_unroll_factor,
        }),
    };

    let mut prog = VmProgram::new();

    // ARCH-LOADPTR-ORDER: 所有 AbiArg 源的 LoadPtr 必须先 emit，StackArg/其他源的 LoadPtr 后 emit。
    // 原因: AbiArg 直接从物理 ABI 寄存器读取 (rdi/rsi/rdx/rcx/r8/r9)，这些寄存器同时是 GPR 池成员，
    // RegAllocator 可能在其他 VReg 的 LoadPtr StackArg 时把它们当 dst 使用（如 `mov rsi, [rbp+0x18]`），
    // 从而破坏尚未消费的 AbiArg 值。重排保证 AbiArg → 目标 VReg 的 mov 先执行，释放 ABI 寄存器供后续复用。
    //
    // 按需分配，只分配实际使用的 VReg。
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    let needs_weight = topology.weight_source == super::topology::WeightSource::WeightRequired;
    let weight_ptr = if needs_weight {
        prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar)
    } else {
        input_ptr
    };
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    // AbiArg sources (input, weight) — load from physical ABI registers
    prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: ctx.session.sym_map.resolve("input").cloned().expect("ABI: input") });
    if needs_weight {
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: ctx.session.sym_map.resolve("weights").cloned().expect("ABI: weights") });
    }

    // StackArg/other sources (output, scratchpad)
    prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: ctx.session.sym_map.resolve("output").cloned().expect("ABI: output") });

    // scratchpad_ptr: 在需要 scratch 的融合模式（NormIntoGemm/QkvSharedInput/FFNBlock/
    // TileLevelFusion/ComputeRoot/CrossLayerResidual/FusedQkvNormRope）使用。
    // ARCH-DATA-FLOW-CONTRACT §3.1: Standalone/LoopFusion 组的 op 若 input/output
    // 是图内 intermediate (非 graph.inputs/outputs), 需要从 scratchpad 读写。
    // 因此当图存在 intermediate 张量 (alloc.total_bytes > 0) 时一律加载 scratchpad。
    let needs_scratch = crate::compiler::codegen::vm::vm_state::needs_scratch_for_plan(
        alloc.total_bytes, ple_req.is_some(), dwc_req.is_some(), plan,
    );
    let scratch_base = if needs_scratch {
        let sp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: sp,
            src: ctx.session.sym_map.resolve("scratchpad").cloned().expect("ABI: scratchpad"),
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
    let resolver = TensorPtrResolver::build(graph, alloc, topology);
    let original_weight_vreg = if needs_weight { Some(weight_ptr) } else { None };
    let mut current_abi = AbiPtrs {
        input_ptr,
        weight_ptr: original_weight_vreg,
        output_ptr,
        scratch_ptr: if needs_scratch { Some(scratch_base) } else { None },
        gen_loop_counter: None,
        layer_loop_counter: None,
        layer_loop_counter_fresh: None,
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

    let fctx = FusionEmitCtx {
        plan,
        graph,
        alloc,
        ctx: &ctx,
        topology,
        rope_cache_offset: rope_req.map(|r| r.cache_offset),
        original_weight_vreg,
    };
    emit_fusion_groups(
        &fctx, &mut prog, &mut current_abi, &resolver,
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

/// 不可变引用打包: emit_fusion_groups 的只读上下文 (BCE-20260630-PIPELINE)。
///
/// `prog` / `current_abi` / `resolver` 为 &mut 独立参数保留在函数签名上,
/// 不进 ctx, 避免 borrow 冲突 (参考 lower_op e0fee8e7 的参数风格)。
pub(super) struct FusionEmitCtx<'a> {
    pub plan: &'a FusionPlan,
    pub graph: &'a CompilerGraph,
    pub alloc: &'a BufferAllocation,
    pub ctx: &'a LoweringContext<'a>,
    pub topology: &'a super::topology::GraphTopologyAnalysis,
    pub rope_cache_offset: Option<usize>,
    pub original_weight_vreg: Option<VRegId>,
}

/// 主循环内每个 group 共享的派生不变量 (从 EmitState.abi 派生, 循环外计算一次)。
struct LoopLocals {
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    scratch_base: VRegId,
    /// Mega-kernel: override Symbolic seq_len bound with DynamicVReg(decode_seq_len).
    seq_bound_override: Option<BoundExpr>,
    /// §0.2.8 ActivationSwap ping-pong buffer VReg 对 (None = 无层循环 / 无 sentinel slot).
    activation_swap_vregs: Option<(VRegId, VRegId)>,
    /// §diagnostic-layer-capture: capture region base VReg + per-layer stride.
    /// None when diagnostic-layer-capture feature is disabled or no capture region allocated.
    /// When Some, emit_layer_capture_copy is called before ActivationSwap at each
    /// layer iteration boundary, copying the pong buffer to capture_base + counter×stride.
    layer_capture: Option<LayerCaptureInfo>,
}

/// §diagnostic-layer-capture: capture region info for ring-buffer layer capture.
struct LayerCaptureInfo {
    /// VReg holding capture_base = scratch_base + layer_capture_offset
    capture_base: VRegId,
    /// Per-layer stride in bytes (max_seq_len × hidden × elem_bytes)
    per_layer_stride: usize,
    /// Hidden dimension (elements per row)
    hidden_dim: usize,
}

pub(super) fn emit_fusion_groups(
    fctx: &FusionEmitCtx,
    prog: &mut VmProgram,
    current_abi: &mut AbiPtrs,
    resolver: &TensorPtrResolver,
) -> Result<(), CompilerError> {
    let width = fctx.ctx.session.width;
    let dtype = fctx.ctx.accum_dtype;
    let _sym_map = fctx.ctx.session.sym_map;
    let _registry = fctx.ctx.session.registry;
    let _hook = fctx.ctx.session.hook;
    let _rope_req = fctx.ctx.rope_req;
    let _ple_req = fctx.ctx.ple_req;
    let _dwc_req = fctx.ctx.dwc_req;
    let layer_loop_cfg = fctx.graph.layer_loop_config.as_ref();
    let hetero_loop_cfg = fctx.graph.hetero_layer_loop_config.as_ref();
    let mixed_quant_loop_cfg = fctx.graph.mixed_quant_layer_loop_config.as_ref();

    // ── 异构模型预编译: 并行编译 4 种层类型模板 ──
    if hetero_loop_cfg.is_some() {
        let _boundaries = compile_hetero_templates_parallel(
            fctx.ctx, fctx.plan, fctx.graph, fctx.alloc, resolver,
        )?;
        // PERF: 后续可引入预编译模板替代逐 group emit,降低 JIT 编译时延(非功能项)
    }

    let mut state = EmitState {
        abi: current_abi.clone(),
        hetero_phase: HeteroPhase::BeforeLayers,
        in_layer_loop: false,
        hetero_seg_byte_offset: None,
        hetero_seg_weight_base: None,
        hetero_global_layer_idx: None,
        hetero_outer_seg_counter: None,
        active_guard: LayerCondition::Always,
        guard_end_label: None,
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
    let activation_swap_vregs: Option<(VRegId, VRegId)> = if layer_loop_cfg.is_some() || hetero_loop_cfg.is_some() || mixed_quant_loop_cfg.is_some() {
        // 从 alloc 中查找 ping/pong sentinel slot 的 offset
        let ping_offset = fctx.alloc.slots.iter()
            .find(|s| s.tensor_id.0 == 0xFFFF_FF00)
            .map(|s| s.offset);
        let pong_offset = fctx.alloc.slots.iter()
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
            Some((ping_ptr, pong_ptr))
        } else {
            // No sentinel slots allocated (e.g. no layer_loop_config.activation_alias)
            None
        }
    } else {
        None
    };

    // §diagnostic-layer-capture: allocate capture_base VReg if capture region exists.
    // The capture region is allocated in buffer_alloc when the
    // `diagnostic-layer-capture` feature is enabled. capture_base =
    // scratch_base + layer_capture_offset. The per-layer side-channel copy
    // is emitted at each layer iteration boundary (close_layer_loop /
    // handle_standard_layer_loop exit) BEFORE ActivationSwap.
    let layer_capture: Option<LayerCaptureInfo> = {
        #[cfg(not(feature = "diagnostic-layer-capture"))]
        {
            None
        }
        #[cfg(feature = "diagnostic-layer-capture")]
        {
            eprintln!("[CAP-ALLOC] alloc.layer_capture_offset={} bytes={} stride={}", fctx.alloc.layer_capture_offset, fctx.alloc.layer_capture_bytes, fctx.alloc.layer_capture_stride);
            if fctx.alloc.layer_capture_bytes > 0 && fctx.alloc.layer_capture_stride > 0 {
                let capture_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::AddPtr {
                    dst: capture_base,
                    base: scratch_base,
                    offset: fctx.alloc.layer_capture_offset,
                });
                // Derive hidden_dim from the per-layer stride.
                // stride = max_seq_len × hidden × elem_bytes (F32 → 4 bytes)
                // hidden = stride / (max_seq_len × 4)
                let max_seq = fctx.graph.max_seq_len.min(ALLOC_SEQ_CAP).max(1);
                let elem_bytes = crate::types::DType::F32.size_bytes();
                let hidden_dim = fctx.alloc.layer_capture_stride / (max_seq * elem_bytes);
                if hidden_dim > 0 {
                    Some(LayerCaptureInfo {
                        capture_base,
                        per_layer_stride: fctx.alloc.layer_capture_stride,
                        hidden_dim,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        }
    };

    let locals = LoopLocals {
        input_ptr,
        weight_ptr,
        output_ptr,
        scratch_base,
        seq_bound_override,
        activation_swap_vregs,
        layer_capture,
    };

    // BCE-20260702-GPU-OOM: per-group RSS probe to locate the exact fusion group
    // that triggers the 993MB → 23.5GB RSS explosion on the GPU codegen path.
    // CPU path peak is ~680MB; GPU path explodes during emit_fusion_groups.
    let _group_total = fctx.plan.groups.len();
    for (gi, group) in fctx.plan.groups.iter().enumerate() {
        if gi % 1 == 0 {
            let rss = (|| -> usize {
                if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
                    for line in s.lines() {
                        if let Some(rest) = line.strip_prefix("VmRSS:") {
                            if let Some(kb) = rest.trim().split_whitespace().next() {
                                if let Ok(n) = kb.parse::<usize>() {
                                    return n / 1024;
                                }
                            }
                        }
                    }
                }
                0
            })();
            eprintln!("[OOMPROBE-GROUP] group {}/{} RSS={}MB instrs={}",
                gi, _group_total, rss, prog.len());
            use std::io::Write;
            std::io::stderr().flush().ok();
        }
        // BCE-20260724-PLAN-C-RESIDUAL-BREAK: Plan C (per-group ping/pong AddPtr
        // reset) DELETED. Root cause of Q5_K_M SIGSEGV was the Skip(N) count
        // over-reaching into subsequent groups' instructions (including Plan C's
        // AddPtr) in close_guard_run/close_pending_guard_run — now fixed by
        // JumpToLabel + MarkLabel. Plan C's AddPtr was papering over the count
        // bug (shifting regalloc decisions to "happen to" avoid spill conflicts)
        // while simultaneously breaking BF16 residual stream (each group reset
        // ping=embedding → attn_resid=emb+attn_output instead of
        // prev_layer_out+attn_output → N=28 logits converge, argmax frozen).
        // With the Skip(N) count bug eradicated, Plan C is no longer needed.
        emit_one_fusion_group(
            fctx, prog, current_abi, resolver, &mut state, group, &locals,
        )?;
    }

    #[cfg(feature = "diagnostic-layer-capture")]
    {
        eprintln!("[CAP-SEQ] VmInstr around capture (instrs 385-415):");
        for (i, instr) in prog.instrs.iter().enumerate() {
            if (385..415).contains(&i) {
                eprintln!("  [{}] {:?}", i, instr);
            }
        }
    }

    // Close any pending guard run after all groups processed
    close_pending_guard_run(prog, &mut state);

    // §diagnostic-layer-capture: dump VmInstr sequence around LoopBegin/capture/LoopEnd
    #[cfg(feature = "diagnostic-layer-capture")]
    {
        eprintln!("[CAP-SEQ] VmInstr sequence (LoopBegin/ActivationSwap/capture/LoopEnd):");
        for (i, instr) in prog.instrs.iter().enumerate() {
            let mark = match instr {
                VmInstr::LoopBegin { .. } => Some("LoopBegin"),
                VmInstr::LoopEnd => Some("LoopEnd"),
                VmInstr::ActivationSwap { .. } => Some("ActivationSwap"),
                VmInstr::GprBinOp { op: GprOp::Mul, .. } => Some("GprMul(capture?)"),
                VmInstr::GprLoadImm { value, .. } if *value > 1000000 => Some("GprLoadImm(big)"),
                _ => None,
            };
            if let Some(m) = mark {
                eprintln!("  [{}] {} {:?}", i, m, instr);
            }
        }
    }

    // Close layer loop if still open (all groups were layer ops)
    if state.in_layer_loop {
        close_layer_loop(prog, &mut state, &locals, width, layer_loop_cfg, fctx.original_weight_vreg, dtype);
    }

    // Write back mutated ABI state to caller.
    *current_abi = state.abi;

    Ok(())
}

/// 关闭未闭合的 guard run: emit MarkLabel to anchor the JumpToLabel target.
///
/// The guard's `GprCondAction { JumpToLabel(end_label) }` was emitted with a
/// label_id allocated up-front; here we emit the matching `MarkLabel` at the
/// guard run's end. This replaces the fragile `Skip(N)` count-back mechanism
/// (BCE-20260724-PLAN-C-RESIDUAL-BREAK) whose non-meta instruction count
/// over-reached into subsequent groups' instructions.
fn close_pending_guard_run(prog: &mut VmProgram, state: &mut EmitState) {
    if let Some(end_label) = state.guard_end_label.take() {
        prog.emit(VmInstr::MarkLabel { label_id: end_label });
    }
}

/// 关闭仍开启的层循环 (所有 group 都是层 op 的情况): ActivationSwap + WeightPrefetchAsync + LoopEnd.
fn close_layer_loop(
    prog: &mut VmProgram,
    state: &mut EmitState,
    locals: &LoopLocals,
    width: SimdWidth,
    layer_loop_cfg: Option<&crate::compiler::graph::LayerLoopConfig>,
    original_weight_vreg: Option<VRegId>,
    dtype: QuantPrecision,
) {
    // §diagnostic-layer-capture: emit counter-scaled side-channel copy BEFORE
    // ActivationSwap. The pong buffer holds the current layer's output; we
    // copy it to capture_base + layer_loop_counter × per_layer_stride.
    // This must happen BEFORE the swap so we capture the just-written buffer.
    #[cfg(feature = "diagnostic-layer-capture")]
    {
        if let Some(ref cap) = locals.layer_capture {
            if let Some(counter) = state.abi.layer_loop_counter {
                if let Some((_, pong)) = locals.activation_swap_vregs {
                    let _ = super::structural_builder::StructuralOpBuilder::emit_layer_capture_copy(
                        prog, pong, cap.capture_base, counter,
                        cap.per_layer_stride, cap.hidden_dim, width, dtype,
                    );
                }
            }
        }
    }
    // §0.2.8 ActivationSwap: 最终层迭代末尾交换 ping-pong buffer
    if let Some((ping, pong)) = locals.activation_swap_vregs {
        prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
    }
    // §0.2.8 WeightPrefetchAsync: GPU-only — CPU backends lack
    // shared memory and async DMA engines.
    if matches!(width, SimdWidth::Warp(_)) {
        if let Some(cfg) = layer_loop_cfg {
            let wp = state.abi.weight_ptr.unwrap_or(locals.weight_ptr);
            prog.emit(VmInstr::WeightPrefetchAsync {
                smem_name: "smem_w_prefetch".to_string(),
                weight_base: wp,
                weight_offset: cfg.weight_stride,
                size: cfg.weight_stride,
            });
        }
    }
    prog.emit(VmInstr::LoopEnd);
    // §0.2.8 Parity fix: after the layer loop, N swaps have occurred.
    // The last iteration wrote to pong, then swapped — so pong_ptr now
    // points to the WRONG buffer. One more swap restores correctness.
    // BCE-20260713-ACTIVATION-SWAP-PARITY: restored (was removed in 210b0d6e
    // which broke N=odd post-loop).
    // Note: Plan C's per-group AddPtr reset (which previously "covered" the
    // cross-group flip) was DELETED in BCE-20260724-PLAN-C-RESIDUAL-BREAK
    // (root-caused as papering over the Skip(N) count bug in guard close,
    // and breaking BF16 residual stream). This post-loop parity swap remains.
    if let Some((ping, pong)) = locals.activation_swap_vregs {
        prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
    }
    // After the layer loop, reset weight_ptr to original (offset 0).
    // Global weights are at the beginning of the blob with absolute offsets.
    state.abi.weight_ptr = original_weight_vreg;
}

/// 处理单个融合组: 层循环 entry/exit (异构或同构) + guard + 算子 emit.
#[allow(clippy::too_many_arguments)]
fn emit_one_fusion_group(
    fctx: &FusionEmitCtx,
    prog: &mut VmProgram,
    current_abi: &mut AbiPtrs,
    resolver: &TensorPtrResolver,
    state: &mut EmitState,
    group: &crate::compiler::fusion::FusionGroup,
    locals: &LoopLocals,
) -> Result<(), CompilerError> {
    let anchor_op = fctx.graph.op(group.anchor).ok_or_else(|| {
        CompilerError::CodegenViolation(format!("anchor op {:?} not found", group.anchor))
    })?;

    // Mega-kernel: skip sampling ops that are manually emitted by
    // compile_mega_kernel_vm's conditional pipeline
    // (GraphTopologyAnalysis.has_sampling_ops). Processing them here would emit
    // duplicate instructions with WRONG ABI parameters (CompiledLayerFn offsets
    // instead of MegaKernelFn offsets), causing memory corruption and wrong
    // control flow.
    // Op 必填，直接读缓存识别采样 op。
    let is_sampling_op = matches!(
        &anchor_op.op,
        crate::compiler::graph::Op::Argmax { .. }
        | crate::compiler::graph::Op::StoreToken
        | crate::compiler::graph::Op::CheckStopCondition
        | crate::compiler::graph::Op::WriteLogits { .. }
        | crate::compiler::graph::Op::MtpDraft { .. }
    );
    if is_sampling_op {
        return Ok(());
    }

    // ── Heterogeneous / mixed-quant / standard layer loop handling ──
    if fctx.graph.hetero_layer_loop_config.is_some() {
        handle_hetero_layer_loop(fctx, prog, state, group, locals)?;
    } else if fctx.graph.mixed_quant_layer_loop_config.is_some() {
        handle_mixed_quant_layer_loop(fctx, prog, state, group, locals)?;
    } else {
        handle_standard_layer_loop(fctx, prog, state, group, locals)?;
    }

    // Sync current_abi from state.abi — layer loop setup mutates state.abi.weight_ptr.
    // resolver.materialize() reads from current_abi. Without this sync, the resolver
    // emits LoadPtr using stale VRegs.
    current_abi.weight_ptr = state.abi.weight_ptr;
    current_abi.scratch_ptr = state.abi.scratch_ptr;
    current_abi.layer_loop_counter = state.abi.layer_loop_counter;
    current_abi.layer_loop_counter_fresh = state.abi.layer_loop_counter_fresh;
    current_abi.gen_loop_counter = state.abi.gen_loop_counter;
    current_abi.kv_cache_ptr = state.abi.kv_cache_ptr;

    // ── Guard + materialize + emit_fusion_group_by_mode ──
    emit_group_guard_and_body(
        prog, fctx, current_abi, resolver, state, group, anchor_op, locals,
    )?;

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §emit_fusion_groups 子fn: 按层循环类型 (异构 / 同构) + guard/body 拆分
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 处理异构层循环 (4-type: sliding/full × small/large FFN)。
///
/// Gemma-4 E2B: 7 segments × [4 sliding + 1 full] = 35 layers.
/// Segments 0-2 use small FFN (ss+fs templates), segments 3-6 use large FFN (sl+fl).
///
/// JIT structure:
///   LoopBegin(num_small_segments=3, step=small_seg_stride)  ── outer small loop
///     LoopBegin(sliding_per_segment=4, step=ss_stride)      ── inner sliding loop
///       ss ops
///     LoopEnd
///     fs full body ops
///   LoopEnd                                                  ── outer small end
///   LoopBegin(num_large_segments=4, step=large_seg_stride)  ── outer large loop
///     LoopBegin(sliding_per_segment=4, step=sl_stride)      ── inner sliding loop
///       sl ops
///     LoopEnd
///     fl full body ops
///   LoopEnd                                                  ── outer large end
fn handle_hetero_layer_loop(
    fctx: &FusionEmitCtx,
    prog: &mut VmProgram,
    state: &mut EmitState,
    group: &crate::compiler::fusion::FusionGroup,
    locals: &LoopLocals,
) -> Result<(), CompilerError> {
    let hcfg = match fctx.graph.hetero_layer_loop_config.as_ref() {
        Some(c) => c,
        None => return Ok(()),
    };
    let num_small_segs = hcfg.large_ffn_start_segment;
    let num_large_segs = hcfg.num_segments - num_small_segs;
    let layers_per_seg = hcfg.sliding_per_segment + 1;
    let weight_ptr = locals.weight_ptr;
    let activation_swap_vregs = locals.activation_swap_vregs;
    let original_weight_vreg = fctx.original_weight_vreg;

    // ── Small segment entry (sliding+small ops) ──
    if group.hetero_layer_type == Some(HeteroLayerType::SlidingSmall) && state.hetero_phase == HeteroPhase::BeforeLayers {
        // Outer loop for small segments
        let seg_counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let seg_byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter: seg_counter,
            byte_offset: seg_byte_off,
            bound: BoundExpr::Const(num_small_segs),
            step_bytes: hcfg.small_segment_stride,
        });
        let seg_wb = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: seg_wb, a: weight_ptr, b: GprOperand::VReg(seg_byte_off), op: GprOp::Add });
        state.hetero_seg_byte_offset = Some(seg_byte_off);
        state.hetero_seg_weight_base = Some(seg_wb);
        state.hetero_outer_seg_counter = Some(seg_counter);
        // Inner sliding loop
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset,
            bound: BoundExpr::Const(hcfg.sliding_per_segment),
            step_bytes: hcfg.sliding_small_stride,
        });
        // Type 0: no correction needed (template base = lbb_off, rel correction = 0)
        let layer_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: layer_weight_base, a: seg_wb, b: GprOperand::VReg(byte_offset), op: GprOp::Add });
        // Compute global layer_idx = seg_counter * layers_per_seg + counter
        let seg_layer_base = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let lps_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: lps_gpr, value: layers_per_seg });
        prog.emit(VmInstr::GprBinOp { dst: seg_layer_base, a: seg_counter, b: GprOperand::VReg(lps_gpr), op: GprOp::Mul });
        let global_layer_idx = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: global_layer_idx, a: seg_layer_base, b: GprOperand::VReg(counter), op: GprOp::Add });
        state.abi.weight_ptr = Some(layer_weight_base);
        state.abi.layer_loop_counter = Some(global_layer_idx);
        state.abi.layer_loop_counter_fresh = Some(global_layer_idx);
        state.hetero_global_layer_idx = Some(global_layer_idx);
        state.in_layer_loop = true;
        state.hetero_phase = HeteroPhase::InSlidingLoop;
    }
    // ── Sliding→Full transition within small segment ──
    if group.hetero_layer_type == Some(HeteroLayerType::FullSmall) && state.hetero_phase == HeteroPhase::InSlidingLoop {
        // ActivationSwap before inner LoopEnd: each sliding iteration swaps ping-pong
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        prog.emit(VmInstr::LoopEnd); // end inner sliding loop
        // Full layer base = seg_weight_base + sliding_per_segment * ss_stride
        // then subtract (type_1_template_base - lbb_off) = ss_stride
        // to align VmInstr weight_offsets (which use graph template layout)
        // with the expanded blob layout.
        let full_off = hcfg.sliding_per_segment * hcfg.sliding_small_stride;
        let seg_wb = state.hetero_seg_weight_base.expect("seg_weight_base not set");
        let full_off_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: full_off_gpr, value: full_off });
        let full_base_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: full_base_ptr, a: seg_wb, b: GprOperand::VReg(full_off_gpr ), op: GprOp::Add });
        // Subtract type 1 relative offset (ss_stride)
        let type1_rel = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: type1_rel, value: hcfg.sliding_small_stride });
        prog.emit(VmInstr::GprBinOp { dst: full_base_ptr, a: full_base_ptr, b: GprOperand::VReg(type1_rel), op: GprOp::Sub });
        // Compute global layer_idx for full layer = seg_counter * layers_per_seg + sliding_per_segment
        let seg_ctr = state.hetero_outer_seg_counter.expect("outer seg_counter not stored in small segment entry");
        let lps_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: lps_gpr, value: layers_per_seg });
        let seg_layer_base = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: seg_layer_base, a: seg_ctr, b: GprOperand::VReg(lps_gpr), op: GprOp::Mul });
        let sps_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: sps_gpr, value: hcfg.sliding_per_segment });
        let full_layer_idx = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: full_layer_idx, a: seg_layer_base, b: GprOperand::VReg(sps_gpr), op: GprOp::Add });
        state.abi.weight_ptr = Some(full_base_ptr);
        state.abi.layer_loop_counter = Some(full_layer_idx);
        state.abi.layer_loop_counter_fresh = Some(full_layer_idx);
        state.hetero_global_layer_idx = Some(full_layer_idx);
        state.in_layer_loop = true;
        state.hetero_phase = HeteroPhase::InFullBody;
    }
    // ── Small→Large segment transition (end small outer loop, start large outer loop) ──
    if group.hetero_layer_type == Some(HeteroLayerType::SlidingLarge) && state.hetero_phase == HeteroPhase::InFullBody {
        // ActivationSwap before outer small segment LoopEnd
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
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
        prog.emit(VmInstr::GprLoadImm { dst: seg_base_tmp, value: large_base_start });
        let seg_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: seg_base, a: weight_ptr, b: GprOperand::VReg(seg_base_tmp ), op: GprOp::Add });
        let seg_wb = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: seg_wb, a: seg_base, b: GprOperand::VReg(seg_byte_off ), op: GprOp::Add });
        // Save for large→full transition
        state.hetero_seg_byte_offset = Some(seg_byte_off);
        state.hetero_seg_weight_base = Some(seg_wb);
        state.hetero_outer_seg_counter = Some(seg_counter);
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
        // Subtract type 2 relative offset (ss_stride + fs_stride)
        let type2_rel = hcfg.sliding_small_stride + hcfg.full_small_stride;
        let type2_rel_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: type2_rel_gpr, value: type2_rel });
        prog.emit(VmInstr::GprBinOp { dst: layer_weight_base, a: layer_weight_base, b: GprOperand::VReg(type2_rel_gpr), op: GprOp::Sub });
        // Compute global layer_idx = (num_small_segs + seg_counter) * layers_per_seg + counter
        let large_seg_offset = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let nss_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: nss_gpr, value: num_small_segs });
        prog.emit(VmInstr::GprBinOp { dst: large_seg_offset, a: nss_gpr, b: GprOperand::VReg(seg_counter), op: GprOp::Add });
        let seg_layer_base = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let lps_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: lps_gpr, value: layers_per_seg });
        prog.emit(VmInstr::GprBinOp { dst: seg_layer_base, a: large_seg_offset, b: GprOperand::VReg(lps_gpr), op: GprOp::Mul });
        let global_layer_idx = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: global_layer_idx, a: seg_layer_base, b: GprOperand::VReg(counter), op: GprOp::Add });
        state.abi.weight_ptr = Some(layer_weight_base);
        state.abi.layer_loop_counter = Some(global_layer_idx);
        state.abi.layer_loop_counter_fresh = Some(global_layer_idx);
        state.hetero_global_layer_idx = Some(global_layer_idx);
        state.in_layer_loop = true;
        state.hetero_phase = HeteroPhase::InLargeSlidingLoop;
    }
    // ── Large sliding→Full transition ──
    if group.hetero_layer_type == Some(HeteroLayerType::FullLarge) && state.hetero_phase == HeteroPhase::InLargeSlidingLoop {
        // ActivationSwap before inner sliding LoopEnd
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        prog.emit(VmInstr::LoopEnd); // end inner sliding loop
        let full_off = hcfg.sliding_per_segment * hcfg.sliding_large_stride;
        let seg_wb = state.hetero_seg_weight_base.expect("seg_weight_base not set for large");
        let full_off_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: full_off_gpr, value: full_off });
        let full_base_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: full_base_ptr, a: seg_wb, b: GprOperand::VReg(full_off_gpr ), op: GprOp::Add });
        // Subtract type 3 relative offset (ss_stride + fs_stride + sl_stride)
        let type3_rel = hcfg.sliding_small_stride + hcfg.full_small_stride + hcfg.sliding_large_stride;
        let type3_rel_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: type3_rel_gpr, value: type3_rel });
        prog.emit(VmInstr::GprBinOp { dst: full_base_ptr, a: full_base_ptr, b: GprOperand::VReg(type3_rel_gpr), op: GprOp::Sub });
        // Compute global layer_idx for full_large = (num_small_segs + seg_counter) * layers_per_seg + sliding_per_segment
        let seg_ctr = state.hetero_outer_seg_counter.expect("outer seg_counter not stored in small→large transition");
        let nss_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: nss_gpr, value: num_small_segs });
        let abs_seg = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: abs_seg, a: nss_gpr, b: GprOperand::VReg(seg_ctr), op: GprOp::Add });
        let lps_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: lps_gpr, value: layers_per_seg });
        let seg_layer_base = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: seg_layer_base, a: abs_seg, b: GprOperand::VReg(lps_gpr), op: GprOp::Mul });
        let sps_gpr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: sps_gpr, value: hcfg.sliding_per_segment });
        let full_layer_idx = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: full_layer_idx, a: seg_layer_base, b: GprOperand::VReg(sps_gpr), op: GprOp::Add });
        state.abi.weight_ptr = Some(full_base_ptr);
        state.abi.layer_loop_counter = Some(full_layer_idx);
        state.abi.layer_loop_counter_fresh = Some(full_layer_idx);
        state.hetero_global_layer_idx = Some(full_layer_idx);
        state.in_layer_loop = true;
        state.hetero_phase = HeteroPhase::InLargeFullBody;
    }
    // ── End of all layer ops ──
    // Close any open loops when we hit the first non-layer group after the hetero phase.
    // If we're in an inner sliding loop (InSlidingLoop/InLargeSlidingLoop),
    // we need to close both inner and outer loops (2 LoopEnds).
    // If we're in a full body phase (InFullBody/InLargeFullBody),
    // the inner loop was already closed by the phase transition,
    // so we only need to close the outer segment loop (1 LoopEnd).
    if !group.is_layer_group && matches!(state.hetero_phase,
        HeteroPhase::InSlidingLoop | HeteroPhase::InFullBody
        | HeteroPhase::InLargeSlidingLoop | HeteroPhase::InLargeFullBody
    ) {
        // Close inner sliding loop if still open
        if matches!(state.hetero_phase, HeteroPhase::InSlidingLoop | HeteroPhase::InLargeSlidingLoop) {
            // ActivationSwap before inner LoopEnd
            if let Some((ping, pong)) = activation_swap_vregs {
                prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
            }
            prog.emit(VmInstr::LoopEnd);
        }
        // ActivationSwap before outer segment LoopEnd
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        // Close outer segment loop
        prog.emit(VmInstr::LoopEnd);
        // §0.2.8 Parity fix: after the hetero loop, N swaps have occurred.
        // One more swap restores pong_ptr to the last-written buffer.
        // BCE-20260713-ACTIVATION-SWAP-PARITY: restored (210b0d6e removed it).
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        state.abi.weight_ptr = original_weight_vreg;
        state.abi.layer_loop_counter = None;
        state.abi.layer_loop_counter_fresh = None;
        state.in_layer_loop = false;
        state.hetero_phase = HeteroPhase::Done;
    }
    Ok(())
}

/// 处理同构 (标准) 层循环: LoopBegin entry + LoopEnd exit + WeightPrefetchAsync.
fn handle_standard_layer_loop(
    fctx: &FusionEmitCtx,
    prog: &mut VmProgram,
    state: &mut EmitState,
    group: &crate::compiler::fusion::FusionGroup,
    locals: &LoopLocals,
) -> Result<(), CompilerError> {
    let layer_loop_cfg = fctx.graph.layer_loop_config.as_ref();
    let topology = fctx.topology;
    let ctx = fctx.ctx;
    let width = ctx.session.width;

    // ── Layer loop entry: emit LoopBegin + compute layer_weight_base ──
    if group.is_layer_group && !state.in_layer_loop {
        if let Some(cfg) = layer_loop_cfg {
            let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
            let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
            // DEBUG: configurable layer count via GLLM_DEBUG_LAYERS env var
            // SPEC/39: num_layers 从 topology 推导，替代 cfg.num_layers 读取
            let topology_num_layers = topology.layer_num_layers.unwrap_or(cfg.num_layers);
            let layer_bound = if let Ok(n) = std::env::var("GLLM_DEBUG_LAYERS") {
                if let Ok(count) = n.parse::<usize>() {
                    BoundExpr::Const(count)
                } else {
                    BoundExpr::Const(topology_num_layers)
                }
            } else if std::env::var("GLLM_SINGLE_LAYER").is_ok() {
                BoundExpr::Const(1)
            } else {
                BoundExpr::Const(topology_num_layers)
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
                src: ctx.session.sym_map.resolve("weights").cloned().expect("ABI: weights"),
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
            // §BCE-20260715-KV-COUNTER-SPILL: fresh per-iteration copy of the layer
            // counter for KV-cache offset computation. The raw `counter` is
            // loop-carried + regalloc-spillable; its spill slot may be polluted by
            // other counters (gen-counter) inside the loop body, blowing up the
            // KV-cache offset → SIGSEGV. Mirrors the mixed-quant path + fresh_weight
            // reload pattern (BCE-20260706-008). Also fixes the standard-path latent
            // bug where KV copy previously hardcoded GprLoadImm{value:0} (KV always
            // wrote layer 0, multi-layer KV cache was silently wrong but did not crash).
            let fresh_layer_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp {
                dst: fresh_layer_ctr, a: counter,
                b: GprOperand::Imm(0), op: GprOp::Add,
            });
            state.abi.layer_loop_counter_fresh = Some(fresh_layer_ctr);
            state.in_layer_loop = true;
        }
    }

    // ── Layer loop exit: emit ActivationSwap + LoopEnd, adjust weight_ptr for globals ──
    if !group.is_layer_group && state.in_layer_loop {
        // §diagnostic-layer-capture: emit counter-scaled side-channel copy BEFORE
        // ActivationSwap. The pong buffer holds the current layer's output.
        // Uses the fresh counter copy (regalloc-safe) — BCE-20260715-KV-COUNTER-SPILL.
        #[cfg(feature = "diagnostic-layer-capture")]
        {
            if let Some(ref cap) = locals.layer_capture {
                let ctr = state.abi.layer_loop_counter_fresh
                    .or(state.abi.layer_loop_counter);
                if let Some(counter) = ctr {
                    if let Some((_, pong)) = locals.activation_swap_vregs {
                        let _ = super::structural_builder::StructuralOpBuilder::emit_layer_capture_copy(
                            prog, pong, cap.capture_base, counter,
                            cap.per_layer_stride, cap.hidden_dim, width, ctx.accum_dtype,
                        );
                    }
                }
            }
        }
        // §0.2.8 ActivationSwap: 每层迭代末尾交换 ping-pong buffer 指针
        if let Some((ping, pong)) = locals.activation_swap_vregs {
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
            if let Some(cfg) = layer_loop_cfg {
                let wp = state.abi.weight_ptr.unwrap_or(locals.weight_ptr);
                prog.emit(VmInstr::WeightPrefetchAsync {
                    smem_name: "smem_w_prefetch".to_string(),
                    weight_base: wp,
                    weight_offset: cfg.weight_stride,
                    size: cfg.weight_stride,
                });
            }
        }
        prog.emit(VmInstr::LoopEnd);
        // §0.2.8 Parity fix: after the layer loop, N swaps have occurred.
        // The last iteration wrote to pong, then swapped — so pong_ptr now
        // points to the WRONG buffer (the one NOT written by the last layer).
        // One more swap restores pong_ptr to the last-written buffer, ensuring
        // post-loop ops (MeanPool, final_norm) read from the correct data.
        // BCE-20260713-ACTIVATION-SWAP-PARITY: restored (210b0d6e removed it).
        if let Some((ping, pong)) = locals.activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        // After the layer loop, reload weight base from ABI args.
        // original_weight_vreg's spill slot may have been overwritten by the
        // register allocator during the multi-iteration layer loop. Reloading
        // from the ABI stack slot guarantees the correct weight_blob base.
        // Global weights (final_norm, logits-producer, embed) are packed at the
        // beginning of the blob, before layer template weights. Their graph
        // offsets are absolute from blob start, so weight_ptr must point to
        // blob offset 0.
        let fresh_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: fresh_weight_base,
            src: ctx.session.sym_map.resolve("weights").cloned().expect("ABI: weights"),
        });
        state.abi.weight_ptr = Some(fresh_weight_base);
        state.abi.layer_loop_counter = None;
        state.abi.layer_loop_counter_fresh = None;
        state.in_layer_loop = false;
    }
    Ok(())
}

/// 处理 mixed-quant (per-layer varying quant dtype) 层循环 (task #6 方案 B)。
///
/// Mixed-quant 模型 (Q5_K_M: 14 Q6K + 14 Q5K 层在 attn_v/ffn_down 不规则交错)
/// 按文件层序单循环迭代, 但每层 weight dtype 不同 → offset_table 非线性
/// (文件层序 running sum, 非组聚集)。
///
/// JIT 结构 (单层 LoopBegin, step=0, 非双层非逐层展开):
///   LoopBegin(counter=layer_idx, bound=num_layers, step_bytes=0)
///     fresh_weight = reload from ABI weights slot (防 regalloc 覆盖)
///     offset = LoadLayerWeightOffset(offset_table, layer_idx)  // 查表
///     weight_ptr = fresh_weight + offset
///     // per-group 模板分化 (v_proj/down_proj per-dtype + LayerInGroup guard)
///     // 由 emit_group_guard_and_body 走 LayerInGroup guard 路径处理
///     ActivationSwap
///   LoopEnd
///   + parity-fix ActivationSwap (恢复 pong_ptr 到最后写入 buffer)
///   + reload weight_ptr from ABI (global ops 用绝对偏移)
///
/// 合宪性 (ARCH-JIT-DATA-YIELDS 铁律4): offset_table 是层序控制流数据
/// (layer_idx → byte offset), 非权重 dtype 数据。LoadLayerWeightOffset 按
/// layer_idx 选 pre-bake 偏移是控制流, 非运行时 dtype match。
fn handle_mixed_quant_layer_loop(
    fctx: &FusionEmitCtx,
    prog: &mut VmProgram,
    state: &mut EmitState,
    group: &crate::compiler::fusion::FusionGroup,
    locals: &LoopLocals,
) -> Result<(), CompilerError> {
    let mcfg = match fctx.graph.mixed_quant_layer_loop_config.as_ref() {
        Some(c) => c,
        None => return Ok(()),
    };
    let topology = fctx.topology;
    let ctx = fctx.ctx;
    let width = ctx.session.width;
    let activation_swap_vregs = locals.activation_swap_vregs;
    let original_weight_vreg = fctx.original_weight_vreg;

    // ── Layer loop entry: emit LoopBegin (step=0) + LoadLayerWeightOffset ──
    if group.is_layer_group && !state.in_layer_loop {
        let counter = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        // SPEC/39: num_layers 从 topology 推导, 替代 cfg.num_layers 读取
        let topology_num_layers = topology.layer_num_layers.unwrap_or(mcfg.num_layers);
        let layer_bound = if let Ok(n) = std::env::var("GLLM_DEBUG_LAYERS") {
            if let Ok(count) = n.parse::<usize>() {
                BoundExpr::Const(count)
            } else {
                BoundExpr::Const(topology_num_layers)
            }
        } else if std::env::var("GLLM_SINGLE_LAYER").is_ok() {
            BoundExpr::Const(1)
        } else {
            BoundExpr::Const(topology_num_layers)
        };

        // step_bytes=0: no linear weight stride (per-layer dtype varies → non-linear).
        // weight offset is computed via LoadLayerWeightOffset(offset_table, layer_idx).
        prog.emit(VmInstr::LoopBegin {
            counter, byte_offset,
            bound: layer_bound,
            step_bytes: 0,
        });

        // Reload weight_ptr from ABI stack slot on every iteration (regalloc safety,
        // mirrors standard path line 908-912).
        let fresh_weight = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: fresh_weight,
            src: ctx.session.sym_map.resolve("weights").cloned().expect("ABI: weights"),
        });

        // BCE-20260716-BUG-A: dump ping/pong ptr values at layer body entry.
        // env=GLLM_TRACE_SWAP compile-time gate. idx = compile-time counter (per
        // emit call) — runtime iterations overwrite the same idx slot → last
        // iteration (layer1 for N=2) value survives. base_offset=4096 (swap-log
        // uses 1024; 16B/entry: [+0]=ping, [+8]=pong).
        if std::env::var("GLLM_TRACE_SWAP").map(|v| !v.is_empty() && v != "0").unwrap_or(false) {
            if let Some((ping_ptr, pong_ptr)) = activation_swap_vregs {
                static TRACE_BODY_IDX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let bidx = TRACE_BODY_IDX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                prog.emit(VmInstr::TracePtrs {
                    a: ping_ptr, b: pong_ptr,
                    base_offset: 4096, idx: bidx,
                });
            }
        }

        // LoadLayerWeightOffset: offset = offset_table[layer_idx] (runtime table lookup).
        let layer_offset = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadLayerWeightOffset {
            dst: layer_offset,
            offset_table: mcfg.offset_table.clone(),
            layer_idx_reg: counter,
        });

        // weight_ptr = fresh_weight + layer_offset
        // (layer_blob_base_offset is already baked into offset_table entries by build_graph)
        let layer_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: layer_weight_base, a: fresh_weight,
            b: GprOperand::VReg(layer_offset), op: GprOp::Add,
        });
        state.abi.weight_ptr = Some(layer_weight_base);
        state.abi.layer_loop_counter = Some(counter);
        // §BCE-20260715-KV-COUNTER-SPILL: fresh per-iteration copy of the layer
        // counter for KV-cache offset computation. The raw `counter` is loop-carried
        // and regalloc-spillable; its spill slot may be polluted by other counters
        // (gen-counter) inside the loop body, blowing up the KV-cache offset → SIGSEGV.
        // The fresh copy has a short live-range (used only by KV copy + capture),
        // so it is far less likely to be spilled, and even if spilled it is rewritten
        // from the authoritative `counter` every iteration (mirrors fresh_weight
        // reload-from-ABI pattern, BCE-20260706-008).
        let fresh_layer_ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: fresh_layer_ctr, a: counter,
            b: GprOperand::Imm(0), op: GprOp::Add,
        });
        state.abi.layer_loop_counter_fresh = Some(fresh_layer_ctr);
        state.in_layer_loop = true;
    }

    // ── Layer loop exit: emit ActivationSwap + LoopEnd + reload weight base ──
    if !group.is_layer_group && state.in_layer_loop {
        // §diagnostic-layer-capture: counter-scaled side-channel copy BEFORE swap.
        // Uses the fresh counter copy (regalloc-safe) so the capture offset is not
        // corrupted by spill-slot pollution (BCE-20260715-KV-COUNTER-SPILL).
        #[cfg(feature = "diagnostic-layer-capture")]
        {
            if let Some(ref cap) = locals.layer_capture {
                let ctr = state.abi.layer_loop_counter_fresh
                    .or(state.abi.layer_loop_counter);
                if let Some(counter) = ctr {
                    if let Some((_, pong)) = activation_swap_vregs {
                        let _ = super::structural_builder::StructuralOpBuilder::emit_layer_capture_copy(
                            prog, pong, cap.capture_base, counter,
                            cap.per_layer_stride, cap.hidden_dim, width, ctx.accum_dtype,
                        );
                    }
                }
            }
        }
        // ActivationSwap: each iteration swaps ping-pong buffer pointers.
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        prog.emit(VmInstr::LoopEnd);
        // §0.2.8 Parity fix: N swaps occurred; one more restores pong_ptr to the
        // last-written buffer (BCE-20260713-ACTIVATION-SWAP-PARITY, same as standard path).
        if let Some((ping, pong)) = activation_swap_vregs {
            prog.emit(VmInstr::ActivationSwap { ptr_a: ping, ptr_b: pong });
        }
        // After the loop, reload weight base from ABI (global ops use absolute offsets).
        let fresh_weight_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr {
            dst: fresh_weight_base,
            src: ctx.session.sym_map.resolve("weights").cloned().expect("ABI: weights"),
        });
        state.abi.weight_ptr = Some(fresh_weight_base);
        state.abi.layer_loop_counter = None;
        state.abi.layer_loop_counter_fresh = None;
        state.in_layer_loop = false;
        // Restore the original weight vreg for post-loop global ops (mirrors hetero path).
        if let Some(orig) = original_weight_vreg {
            state.abi.weight_ptr = Some(orig);
        }
    }
    Ok(())
}

/// Emit a bitset membership test: returns a VReg holding
/// `is_member = (bitset >> layer_idx) & 1` (1 iff `layer_idx` is in `bitset`).
///
/// Used by `LayerCondition::LayerInGroup` / `LayerNotInGroup` guards for
/// mixed-quant irregular interleaving (Q5_K_M: 14 Q6K + 14 Q5K layers).
/// Reuses the production-proven `GprBinOp(BitTest)` primitive (`dst = (a >> b) & 1`),
/// which lowers to x86 `shr` + `and` / AArch64 `lsr` + `and` — no new VmInstr needed.
///
/// Emitted instructions:
///   GprLoadImm(bitset_reg, bitset)
///   GprBinOp(is_member, bitset_reg, layer_idx, BitTest)  // (bitset >> layer_idx) & 1
fn emit_bitset_membership(
    prog: &mut VmProgram,
    bitset: u64,
    layer_idx: VRegId,
) -> VRegId {
    let bitset_reg = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: bitset_reg, value: bitset as usize });
    let is_member = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp {
        dst: is_member,
        a: bitset_reg,
        b: GprOperand::VReg(layer_idx),
        op: GprOp::BitTest,
    });
    is_member
}

/// 处理 group 的 guard (mixed / standard) + tensor materialize + emit_fusion_group_by_mode。
#[allow(clippy::too_many_arguments)]
fn emit_group_guard_and_body(
    prog: &mut VmProgram,
    fctx: &FusionEmitCtx,
    current_abi: &mut AbiPtrs,
    resolver: &TensorPtrResolver,
    state: &mut EmitState,
    group: &crate::compiler::fusion::FusionGroup,
    anchor_op: &crate::compiler::graph::CompilerOp,
    locals: &LoopLocals,
) -> Result<(), CompilerError> {
    let ctx = fctx.ctx;
    let graph = fctx.graph;
    let alloc = fctx.alloc;
    let input_ptr = locals.input_ptr;
    let weight_ptr = locals.weight_ptr;
    let output_ptr = locals.output_ptr;
    let scratch_base = locals.scratch_base;

    // ── Layer guard (NO_LAYER_EXPAND, SPEC 03 §1.3.1) ──
    // Emit GprCondAction to conditionally skip ops based on layer_idx.
    // Consecutive ops with the same guard are merged into a single skip range.
    // IMPORTANT: Guard detection MUST happen before materialize. If materialize
    // runs first, the new group's LoadPtr instructions get included in the
    // previous guard's Skip range. When that guard fires, the LoadPtr is skipped
    // but the computation (outside the skip range) still executes with
    // uninitialized pointers → SIGSEGV.
    //
    // QkvSharedInput special case: when ops within the group have mixed guards
    // (e.g. Q proj = Always, K/V proj = kv_guard for SharedKvRef), the group
    // cannot use the anchor's guard for all ops. Each op must be emitted with
    // its own guard. We break the group into per-op emission with individual
    // guard transitions.
    let has_mixed_guards = group.mode == FusionMode::QkvSharedInput
        && group.ops.iter().any(|&oid| {
            graph.op(oid).map_or(false, |o| o.guard != anchor_op.guard)
        });

    if has_mixed_guards {
        // Per-op guard handling for QkvSharedInput with mixed guards.
        // Each op gets its own guard transition and individual GEMM emission.
        for &op_id in &group.ops {
            let op = match graph.op(op_id) {
                Some(o) => o,
                None => continue,
            };
            let per_op_guard = op.guard;

            // Close previous guard run if guard changed
            if per_op_guard != state.active_guard {
                close_guard_run(prog, state);
                state.active_guard = per_op_guard;

                if per_op_guard != LayerCondition::Always && state.in_layer_loop {
                    let counter = state.hetero_global_layer_idx
                        .or(state.abi.layer_loop_counter)
                        .expect("guarded op requires active layer loop");
                    let skip_cond = match per_op_guard {
                        LayerCondition::LayerIdxLt(t) => {
                            GprCondition::CmpGeU(counter, t as u64)
                        }
                        LayerCondition::LayerIdxGe(t) => {
                            GprCondition::CmpLtU(counter, t as u64)
                        }
                        LayerCondition::LayerInGroup(bitset) => {
                            // skip when layer_idx NOT in bitset → is_member == 0.
                            // GprOp::BitTest computes dst = (a >> b) & 1, so
                            // is_member = (bitset >> layer_idx) & 1.
                            let is_member = emit_bitset_membership(prog, bitset, counter);
                            GprCondition::CmpEq(is_member, 0)
                        }
                        LayerCondition::LayerNotInGroup(bitset) => {
                            // skip when layer_idx IS in bitset → is_member != 0.
                            let is_member = emit_bitset_membership(prog, bitset, counter);
                            GprCondition::IsNonNull(is_member)
                        }
                        LayerCondition::Always => unreachable!(),
                    };
                    // BCE-20260724-PLAN-C-RESIDUAL-BREAK: JumpToLabel + MarkLabel
                    // replaces Skip(N) count-back (count over-reached into later
                    // groups' instrs → Q5_K_M SIGSEGV when Plan C shifted counts).
                    let end_label = prog.alloc_label();
                    prog.emit(VmInstr::GprCondAction {
                        cond: skip_cond,
                        action: GprBranchAction::JumpToLabel(end_label),
                    });
                    state.guard_end_label = Some(end_label);
                }
            }

            // Materialize per-op tensors
            let op_input_ptr = op.inputs.first()
                .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
                .unwrap_or(input_ptr);
            let op_weight_ptr = op.inputs.get(1)
                .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
                .unwrap_or(weight_ptr);

            // QuantGemm must stay on the quantized lowering path.  This per-op
            // branch exists for QkvSharedInput groups with mixed layer guards; the
            // regular group dispatcher has the same guard in
            // `emit_fusion_group_by_mode`, but this branch used to route every op
            // through the generic F32 GEMM emitter.  That reinterpreted raw Q4_0
            // bytes as F32 (and dropped the DecodeTraceBuilder QuantLoadBytesVec).
            if matches!(&op.op, crate::compiler::graph::Op::QuantGemm(_)) {
                let out_ptr = load_op_scratch_ptr(prog, scratch_base, op, alloc, resolver, current_abi)?;
                prog.emit_scope(|p| {
                    emit_standalone_op(
                        p, op, graph, ctx,
                        op_input_ptr, op_weight_ptr, out_ptr,
                        fctx.rope_cache_offset, resolver, current_abi,
                    )
                })?;
            } else if let Ok((m_dim, n, k)) = extract_gemm_dims_sym(op, graph) {
                let out_ptr = load_op_scratch_ptr(prog, scratch_base, op, alloc, resolver, current_abi)?;
                let pm = ctx.pack_map_for_gemm(op.inputs.get(1).copied());
                // Op 必填，直接读 trans_b。
                let trans_b = match &op.op {
                    crate::compiler::graph::Op::Gemm(spec)
                    | crate::compiler::graph::Op::GemmBias(spec) => spec.trans_b,
                    _ => false,
                };
                prog.emit_scope(|p| -> Result<(), CompilerError> {
                    // BCE-20260629-003 (Pattern c): per-matrix dtype — b_dtype 从权重推断
                    let b_dt = op.inputs.get(1)
                        .and_then(|&tid| graph.tensor(tid))
                        .map(|t| t.dtype.to_quant_precision())
                        .unwrap_or(ctx.accum_dtype);
                    emit_gemm_inline_with_hook(p, &m_dim, n, k, ctx,
                        op_input_ptr, op_weight_ptr, out_ptr,
                        locals.seq_bound_override.as_ref(), Some(op.id), pm, trans_b,
                        ctx.accum_dtype, b_dt, ctx.accum_dtype)?;
                    Ok(())
                })?;
            }
        }
    } else {
        // Standard group-level guard handling
        let op_guard = anchor_op.guard;
        if op_guard != state.active_guard {
            // Close previous guard run (patch-back Skip count)
            close_guard_run(prog, state);
            state.active_guard = op_guard;

            // Open new guard run if non-Always and inside layer loop
            if op_guard != LayerCondition::Always && state.in_layer_loop {
                // In hetero mode, use the computed global layer_idx register
                // (which accounts for segment × layers_per_seg + inner position)
                // rather than the raw inner loop counter.
                let counter = state.hetero_global_layer_idx
                    .or(state.abi.layer_loop_counter)
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
                    LayerCondition::LayerInGroup(bitset) => {
                        // skip when layer_idx NOT in bitset → is_member == 0
                        let is_member = emit_bitset_membership(prog, bitset, counter);
                        GprCondition::CmpEq(is_member, 0)
                    }
                    LayerCondition::LayerNotInGroup(bitset) => {
                        // skip when layer_idx IS in bitset → is_member != 0
                        let is_member = emit_bitset_membership(prog, bitset, counter);
                        GprCondition::IsNonNull(is_member)
                    }
                    LayerCondition::Always => unreachable!(),
                };
                // BCE-20260724-PLAN-C-RESIDUAL-BREAK: JumpToLabel + MarkLabel
                // replaces Skip(N) count-back (count over-reached into later
                // groups' instrs → Q5_K_M SIGSEGV when Plan C shifted counts).
                let end_label = prog.alloc_label();
                prog.emit(VmInstr::GprCondAction {
                    cond: skip_cond,
                    action: GprBranchAction::JumpToLabel(end_label),
                });
                state.guard_end_label = Some(end_label);
            }
        }

        // ARCH-DATA-FLOW-CONTRACT §3 (D#1 统一根治):
        // group 内 anchor_op 的 input[0] / input[1] / output[0] 统一经
        // TensorPtrResolver 查询, 物理位置由建表阶段一次性决定 (Activation /
        // Weight / Intermediate / Output), 每处按 tensor_id 取真实 base+offset.
        // Materialize runs AFTER guard detection so that LoadPtr instructions
        // are correctly placed relative to the guard's skip range.
        let group_input_ptr = anchor_op.inputs.first()
            .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
            .unwrap_or(input_ptr);
        let group_weight_ptr = anchor_op.inputs.get(1)
            .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
            .unwrap_or(weight_ptr);
        let group_output_ptr = anchor_op.outputs.first()
            .and_then(|&tid| resolver.materialize(prog, tid, current_abi))
            .or_else(|| {
                // Fallback: for EpilogueInjection, the anchor's output tensor
                // may not be in the resolver map (it's an intermediate consumed
                // by the epilogue op). Try the terminal op's output instead,
                // since GEMM+epilogue share the same physical buffer.
                group.epilogue.last()
                    .and_then(|&oid| graph.op(oid))
                    .and_then(|op| op.outputs.first().copied())
                    .and_then(|tid| resolver.materialize(prog, tid, current_abi))
            })
            .unwrap_or(output_ptr);

        // §4 CompoundExecution: 先按 FusionMode dispatch，再按 OpKind
        emit_fusion_group_by_mode(
            prog, group, anchor_op, graph, alloc, ctx,
            group_input_ptr, group_weight_ptr, group_output_ptr,
            scratch_base, input_ptr, weight_ptr, output_ptr,
            fctx.rope_cache_offset, locals.seq_bound_override.as_ref(),
            resolver, current_abi,
        )?;
    }
    Ok(())
}

/// 关闭一个 guard run: emit MarkLabel to anchor the JumpToLabel target.
///
/// Mirrors `close_pending_guard_run`: emits the matching `MarkLabel` for the
/// pending `GprCondAction { JumpToLabel(end_label) }` so the conditional jump
/// lands exactly at the guard run boundary — no fragile instruction counting.
/// (BCE-20260724-PLAN-C-RESIDUAL-BREAK)
fn close_guard_run(prog: &mut VmProgram, state: &mut EmitState) {
    if let Some(end_label) = state.guard_end_label.take() {
        prog.emit(VmInstr::MarkLabel { label_id: end_label });
    }
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
    // ARCH-DATA-FLOW-CONTRACT §2.3: 行字节数 = feature_dim × dtype.elem_bytes()（dtype 参数化，不硬编码）
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
                        dtype, predicate: None,
                    });
                    if let Some(s) = sec {
                        prog.emit(VmInstr::VecLoad {
                            dst: s, base: row_weight, offset: OffsetExpr::LoopOffset(col_off), width,
                            dtype, predicate: None,
                        });
                    }
                    lower::lower_trace_body_compat(prog, body, acc, sec, width, dtype)
                        .expect("lower_trace_body: OpTrace invariant violation");
                    prog.emit(VmInstr::VecStore {
                        base: row_output, offset: OffsetExpr::LoopOffset(col_off), src: acc, width,
                        dtype, predicate: None,
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
                        dtype, predicate: None,
                    });
                    if let Some(s) = s_sec {
                        prog.emit(VmInstr::VecLoad {
                            dst: s, base: row_weight, offset: OffsetExpr::Const(col_off_const), width: s_width,
                            dtype, predicate: None,
                        });
                    }
                    lower::lower_trace_body_compat(prog, body, s_acc, s_sec, s_width, dtype)
                        .expect("lower_trace_body: OpTrace invariant violation (scalar tail)");
                    prog.emit(VmInstr::VecStore {
                        base: row_output, offset: OffsetExpr::Const(col_off_const), src: s_acc, width: s_width,
                        dtype, predicate: None,
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
                    dtype, predicate: None,
                });
                if let Some(s) = sec {
                    prog.emit(VmInstr::VecLoad {
                        dst: s, base: weight_ptr,
                        offset: OffsetExpr::LoopOffset(byte_off), width,
                        dtype, predicate: None,
                    });
                }
                lower::lower_trace_body_compat(prog, body, acc, sec, width, dtype)
                    .expect("lower_trace_body: OpTrace invariant violation");
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: OffsetExpr::LoopOffset(byte_off), src: acc, width,
                    dtype, predicate: None,
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
                    dtype, predicate: None,
                });
                if let Some(s) = s_sec {
                    prog.emit(VmInstr::VecLoad {
                        dst: s, base: weight_ptr,
                        offset: OffsetExpr::Const(col_off_const), width: s_width,
                        dtype, predicate: None,
                    });
                }
                lower::lower_trace_body_compat(prog, body, s_acc, s_sec, s_width, dtype)
                    .expect("lower_trace_body: OpTrace invariant violation (scalar tail)");
                prog.emit(VmInstr::VecStore {
                    base: output_ptr, offset: OffsetExpr::Const(col_off_const),
                    src: s_acc, width: s_width,
                    dtype, predicate: None,
                });
            }
        }
    }
    Ok(acc)
}

