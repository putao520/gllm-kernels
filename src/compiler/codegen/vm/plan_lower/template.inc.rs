// §14 LayerTemplate 编译 — 异构模型并行编译支持
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 融合组间 dtype/布局转换 (预留)。
///
/// 当前模型架构中融合组间无显式 Cast 算子（融合 pass 已合并可融合的算子链），
/// 所以本函数为 no-op。当未来需要支持异构层间 dtype 变化时在此处扩展。
fn emit_inter_group_casts(
    _prog: &mut VmProgram,
    _group: &crate::compiler::fusion::FusionGroup,
    _graph: &CompilerGraph,
    _width: SimdWidth,
    _abi: &AbiPtrs,
    _resolver: &TensorPtrResolver,
) -> Result<(), CompilerError> {
    // No-op: 当前所有融合组间无 dtype 转换需求
    Ok(())
}

/// 在独立 VmProgram 中编译一组 fusion groups，返回 LayerTemplate。
///
/// 用于第一层并行化：异构模型的每种层类型独立编译为模板，
/// 然后在主 VmProgram 中通过 append_with_mapping 实例化。
pub fn compile_layer_type_body(
    ctx: &LoweringContext,
    group_range: std::ops::Range<usize>,
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    resolver: &TensorPtrResolver,
) -> Result<super::instr::LayerTemplate, CompilerError> {
    let mut template_prog = VmProgram::new();

    let tpl_input = template_prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tpl_weight = template_prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tpl_output = template_prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let tpl_scratch = template_prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    let tpl_abi = AbiPtrs {
        input_ptr: tpl_input,
        weight_ptr: Some(tpl_weight),
        output_ptr: tpl_output,
        scratch_ptr: Some(tpl_scratch),
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

    let rope_cache_offset = ctx.rope_req.map(|r| r.cache_offset);
    let seq_bound_override: Option<BoundExpr> = None;

    template_prog.emit(VmInstr::Comment("LayerTemplate: ABI placeholders".into()));

    for gi in group_range {
        let group = &plan.groups[gi];
        let anchor_op = graph.op(group.anchor).ok_or_else(|| {
            CompilerError::CodegenViolation(format!("anchor op {:?} not found", group.anchor))
        })?;

        let group_input_ptr = anchor_op.inputs.first()
            .and_then(|&tid| resolver.materialize(&mut template_prog, tid, &tpl_abi))
            .unwrap_or(tpl_input);
        let group_weight_ptr = anchor_op.inputs.get(1)
            .and_then(|&tid| resolver.materialize(&mut template_prog, tid, &tpl_abi))
            .unwrap_or(tpl_weight);
        let group_output_ptr = anchor_op.outputs.first()
            .and_then(|&tid| resolver.materialize(&mut template_prog, tid, &tpl_abi))
            .unwrap_or(tpl_output);

        emit_fusion_group_by_mode(
            &mut template_prog, group, anchor_op, graph, alloc, ctx,
            group_input_ptr, group_weight_ptr, group_output_ptr,
            tpl_scratch, tpl_input, tpl_weight, tpl_output,
            rope_cache_offset, seq_bound_override.as_ref(),
            resolver, &tpl_abi,
        )?;

        emit_inter_group_casts(&mut template_prog, group, graph, ctx.width, &tpl_abi, resolver)?;
    }

    Ok(super::instr::LayerTemplate {
        body: template_prog,
        abi_map: super::instr::LayerAbiMap {
            input_ptr: tpl_input,
            weight_ptr: tpl_weight,
            output_ptr: tpl_output,
            scratch_base: tpl_scratch,
        },
    })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §N GroupDependencyAnalyzer — 融合组间依赖分析
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 融合组间依赖分析器 — 分析 FusionPlan 的 group 间数据依赖。
///
/// 返回拓扑层级列表：同一层级的 groups 之间无数据依赖，可以并行处理。
struct GroupDependencyAnalyzer;

impl GroupDependencyAnalyzer {
    /// 分析 FusionPlan 的 group 间数据依赖。
    ///
    /// 依赖基于 tensor 生产-消费关系：
    /// - group A 的 output tensor → group B 的 input tensor → A blocks B
    /// - 同一层级的 groups 之间没有 tensor 流转关系
    ///
    /// Returns: Vec<Vec<usize>> — 拓扑层级，每组包含可并行处理的 group 索引列表
    fn analyze(plan: &FusionPlan, graph: &CompilerGraph) -> Vec<Vec<usize>> {
        let n = plan.groups.len();
        if n == 0 { return vec![]; }

        // Build: for each group, which groups produce its inputs?
        // group_deps[gi] = set of group indices that gi depends on
        let mut group_deps: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); n];

        for (gi, group) in plan.groups.iter().enumerate() {
            // Collect all input tensor IDs of this group
            let mut input_tensors: std::collections::HashSet<TensorId> =
                std::collections::HashSet::new();
            for &op_id in &group.ops {
                if let Some(op) = graph.op(op_id) {
                    for &tid in &op.inputs {
                        input_tensors.insert(tid);
                    }
                }
            }

            // Find which other groups produce these tensors
            for (other_gi, other_group) in plan.groups.iter().enumerate() {
                if other_gi == gi { continue; }
                for &op_id in &other_group.ops {
                    if let Some(op) = graph.op(op_id) {
                        for &tid in &op.outputs {
                            if input_tensors.contains(&tid) {
                                group_deps[gi].insert(other_gi);
                            }
                        }
                    }
                }
            }
        }

        // Topological sort into levels
        let mut levels: Vec<Vec<usize>> = vec![];
        let mut assigned = vec![false; n];
        let mut remaining = n;

        while remaining > 0 {
            let mut level = vec![];
            for gi in 0..n {
                if assigned[gi] { continue; }
                // Check if all dependencies are assigned
                let all_deps_assigned = group_deps[gi].iter().all(|&dep| assigned[dep]);
                if all_deps_assigned {
                    level.push(gi);
                }
            }
            if level.is_empty() {
                // Circular dependency — assign remaining to one level
                for gi in 0..n {
                    if !assigned[gi] { level.push(gi); }
                }
            }
            for &gi in &level {
                assigned[gi] = true;
            }
            remaining -= level.len();
            levels.push(level);
        }

        levels
    }
}
