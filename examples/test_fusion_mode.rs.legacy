use gllm_kernels::compiler::graph::{CompilerGraph, OpKind, SymDim};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::compiler::fusion::fuse_graph;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::DType;

fn main() {
    let n = 16;
    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[n], DType::F32);
    let bias = g.add_tensor_concrete("bias", &[n], DType::F32);
    let mid = g.add_tensor_concrete("mid", &[n], DType::F32);
    let output = g.add_tensor_concrete("output", &[n], DType::F32);
    g.inputs = vec![input, bias];
    g.outputs = vec![output];
    g.add_op(OpKind::Gelu, vec![input], vec![mid], "gelu");
    g.add_op(OpKind::Add, vec![mid, bias], vec![output], "add");

    let registry = gllm_kernels::compiler::symexec::ScalarOpRegistry::with_defaults();
    let profile = DeviceProfile::detect();
    let exec_plan = gllm_kernels::compiler::planner::ExecutionPlan::from_profile(&profile);
    let plan = fuse_graph(&g, &exec_plan, Some(&registry));
    
    for (i, group) in plan.groups.iter().enumerate() {
        eprintln!("Group {}: mode={:?}, anchor={:?}, ops={:?}, epilogue={:?}", 
            i, group.mode, group.anchor, group.ops, group.epilogue);
        for &op_id in &group.ops {
            if let Some(op) = g.op(op_id) {
                eprintln!("  op {:?}: {:?} inputs={:?} outputs={:?}", 
                    op_id, op.kind, op.inputs, op.outputs);
            }
        }
    }
}
