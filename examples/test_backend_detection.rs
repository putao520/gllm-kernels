//! Backend Detection Test

use gllm_kernels::KernelDispatcher;

fn main() {
    println!("=== Backend Detection Analysis ===\n");
    
    // Check WGPU availability
    println!("Checking WGPU...");
    match gllm_kernels::wgpu_kernels::EmbeddingOpsKernel::create_default() {
        Ok(_) => println!("  ✅ WGPU kernel available"),
        Err(e) => println!("  ❌ WGPU unavailable: {}", e),
    }
    
    // Check runtime detection
    println!("\nRuntime detection result:");
    let dispatcher = KernelDispatcher::new();
    println!("  Selected: {:?}", dispatcher.backend());
    
    // Force WGPU test
    println!("\n=== Direct WGPU Test ===");
    if let Ok(kernel) = gllm_kernels::wgpu_kernels::EmbeddingOpsKernel::create_default() {
        let scores: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        match kernel.top_k_select_f32(&scores, 10, false) {
            Ok((indices, values)) => {
                println!("✅ Top-K works: indices={:?}, values={:?}", 
                         &indices[..5], &values[..5]);
            }
            Err(e) => println!("❌ Top-K failed: {}", e),
        }
        
        // Small rerank test
        let binary_q: Vec<u32> = vec![0xAAAA5555; 16];  // dim=512/32=16
        let binary_db: Vec<u32> = (0..1000*16).map(|i| i as u32).collect();
        let int8_q: Vec<u32> = vec![0x7F7F7F7F; 128];   // dim=512/4=128
        let int8_db: Vec<u32> = (0..1000*128).map(|i| i as u32).collect();
        
        let config = gllm_kernels::wgpu_kernels::GpuRerankConfig {
            binary_k: 100, int8_k: 10, dim: 512
        };
        
        match kernel.rerank_pipeline(&binary_q, &binary_db, &int8_q, &int8_db, 1000, &config, 0.01) {
            Ok(r) => println!("✅ Rerank pipeline works: {} results", r.indices.len()),
            Err(e) => println!("❌ Rerank pipeline failed: {}", e),
        }
    }
}
