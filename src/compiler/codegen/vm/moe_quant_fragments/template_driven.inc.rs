
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 27 REQ-AT-009: 模板驱动 MoE 发射桥接
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 尝试通过模板驱动路径发射 MoE Router+TopK 算子。
pub(crate) fn emit_moe_template_driven(
    prog: &mut VmProgram,
    num_experts: usize,
    hidden: usize,
    top_k: usize,
    width: SimdWidth,
    input_ptr: VRegId,
    weight_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Option<()> {
    use super::algo_registry;
    use super::algo_interpreter::{TemplateInterpreter, ParamTable, TemplateInputs};
    use crate::dispatch::device_profile::DeviceProfile;

    let profile = DeviceProfile::detect();
    let template = algo_registry::select_template(
        &crate::compiler::codegen::vm::algo_template::AlgoStrategy::MoeRouterTopk,
        &profile,
    )?;

    let mut params = ParamTable::new();
    params.set("num_experts", num_experts);
    params.set("hidden_dim", hidden);
    params.set("top_k", top_k);

    let seq_offset = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    let inputs = TemplateInputs::moe();

    let mut interp = TemplateInterpreter::new(params);
    let trace_ops = interp.instantiate(template, &inputs);

    super::auto_select::auto_lower_trace_raw(
        prog, &trace_ops, &[input_ptr, weight_ptr, output_ptr, seq_offset],
        width, dtype,
    ).ok()?;

    Some(())
}
