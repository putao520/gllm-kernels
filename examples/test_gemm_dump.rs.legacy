use gllm_kernels::compiler::graph::{CompilerGraph, OpKind, SymDim};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::DType;

fn main() {
    let (m, n, k) = (4usize, 16usize, 16usize);
    let mut g = CompilerGraph::new();
    let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
    let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
    let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
    g.inputs = vec![a, b];
    g.outputs = vec![c];
    g.add_op(OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32 }, vec![a, b], vec![c], "gemm");
    
    let mut compiler = InferenceCompiler::with_profile(DeviceProfile::detect());
    let layer = compiler.compile_graph(&g).expect("compile failed");
    
    let code = layer.code_bytes();
    eprintln!("code_size = {} bytes", code.len());
    
    // Write code to temp file for objdump
    std::fs::write("/tmp/gemm_4_16_16.bin", code).expect("write failed");
    eprintln!("Written to /tmp/gemm_4_16_16.bin");
}
