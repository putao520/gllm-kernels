use gllm_kernels::compiler::graph::{CompilerGraph, OpKind, SymDim};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::DType;

fn main() {
    let (m, n, k) = (8usize, 16usize, 16usize);
    let mut g = CompilerGraph::new();
    let a = g.add_tensor_concrete("A", &[m, k], DType::F32);
    let b = g.add_tensor_concrete("B", &[k, n], DType::F32);
    let c = g.add_tensor_concrete("C", &[m, n], DType::F32);
    g.inputs = vec![a, b];
    g.outputs = vec![c];
    g.add_op(OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: DType::F32 }, vec![a, b], vec![c], "gemm");
    
    let mut compiler = InferenceCompiler::with_profile(DeviceProfile::detect());
    let layer = compiler.compile_graph(&g).expect("compile failed");
    
    let mut b_data = vec![0.0f32; k * n];
    for i in 0..k { b_data[i * n + i] = 1.0; }
    let a_data: Vec<f32> = (0..(m*k)).map(|i| (i as f32) / 10.0).collect();
    
    let wl = g.weight_layout();
    let mut weight_blob = vec![0u8; wl.total_bytes];
    for &(tid, off) in &wl.offsets {
        let input_idx = g.inputs.iter().position(|&t| t == tid).unwrap();
        let src = if input_idx == 0 { &a_data } else { &b_data };
        weight_blob[off..off + src.len() * 4].copy_from_slice(
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) });
    }
    
    let mut output = vec![0.0f32; m * n];
    let mut scratch = vec![0u8; layer.scratchpad_bytes];
    
    eprintln!("A[0..4] = {:?}", &a_data[0..4]);
    
    unsafe {
        layer.execute(
            a_data.as_ptr() as *const u8,
            weight_blob.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(), std::ptr::null(),
            0, 0,
            output.as_mut_ptr() as *mut u8,
            if layer.scratchpad_bytes > 0 { scratch.as_mut_ptr() } else { std::ptr::null_mut() },
        );
    }
    
    eprintln!("C[0..4] = {:?}", &output[0..4]);
    eprintln!("Expected = {:?}", &a_data[0..4]);
    let mut max_diff = 0.0f32;
    for i in 0..(m*n) {
        let diff = (output[i] - a_data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    eprintln!("max_diff = {}", max_diff);
}
