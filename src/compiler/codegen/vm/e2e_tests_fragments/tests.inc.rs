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

    fn build_unary(op: Op) -> (CompilerGraph, FusionPlan, BufferAllocation) {
        let mut g = CompilerGraph::new();
        let inp = g.add_tensor_concrete("input", &[32], DType::F32);
        let out = g.add_tensor_concrete("output", &[32], DType::F32);
        g.inputs = vec![inp];
        g.outputs = vec![out];
        let op = g.add_op(op, vec![inp], vec![out], "op");
        let mut m = HashMap::new();
        m.insert(op, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0, anchor: op, epilogue: vec![],
                mode: FusionMode::LoopFusion, ops: vec![op],
                multi_output: MultiOutputConfig::single(),
                    dominant_dtype: None,
                    marker: GroupMarker::None,
                    is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group: m,
        };
        let alloc = BufferAllocation::default();
        (g, plan, alloc)
    }

    #[test]
    fn test_vm_compile_layer_code_size() {
        let (g, p, a) = build_unary(Op::Silu);
        let profile = DeviceProfile::detect();
        let exec_plan = crate::compiler::planner::ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let output = compile_layer(&p, &g, &a, &exec_plan, Some(&registry)).unwrap();
        // SiLU kernel 应该产出实质性代码
        assert!(output.code.len() > 50, "code too small: {}", output.code.len());
        assert!(output.code.len() < 10000, "code too large: {}", output.code.len());
    }
}
