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
    
    let a_data = vec![1.0f32; m * k];
    let mut b_data = vec![0.0f32; k * n];
    for i in 0..k { b_data[i * n + i] = 1.0; }
    
    let mut output = vec![0.0f32; m * n];
    let mut scratch = vec![0u8; layer.scratchpad_bytes];
    
    let entry = unsafe { layer.entry_point() };
    eprintln!("entry = {:p}", entry as *const ());
    eprintln!("a = {:p}, b = {:p}, c = {:p}, scratch = {:p}",
        a_data.as_ptr(), b_data.as_ptr(), output.as_ptr(), scratch.as_ptr());
    
    // Busy loop to allow debugger attach
    eprintln!("PID = {}, waiting for debugger...", std::process::id());
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    unsafe {
        layer.execute(
            a_data.as_ptr() as *const u8,
            b_data.as_ptr() as *const u8,
            std::ptr::null_mut(),
            std::ptr::null(), std::ptr::null(),
            0, 0,
            output.as_mut_ptr() as *mut u8,
            if layer.scratchpad_bytes > 0 { scratch.as_mut_ptr() } else { std::ptr::null_mut() },
        );
    }
    
    eprintln!("C[0..8] = {:?}", &output[0..8]);
    eprintln!("Expected all 1.0");
}
