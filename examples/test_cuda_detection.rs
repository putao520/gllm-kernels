//! CUDA Detection Debug

fn main() {
    println!("=== CUDA Detection Debug ===\n");
    
    // Check nvidia-smi
    println!("1. nvidia-smi check:");
    match std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version,memory.total")
        .arg("--format=csv,noheader")
        .output() 
    {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("   ✅ GPU found:\n   {}", stdout.trim());
            } else {
                println!("   ❌ nvidia-smi failed");
            }
        }
        Err(e) => println!("   ❌ nvidia-smi not found: {}", e),
    }
    
    // Check cudarc init
    println!("\n2. cudarc initialization:");
    match cudarc::driver::result::init() {
        Ok(_) => println!("   ✅ CUDA driver initialized"),
        Err(e) => println!("   ❌ CUDA init failed: {:?}", e),
    }
    
    // Check device count
    println!("\n3. CUDA device count:");
    match cudarc::driver::result::device::get_count() {
        Ok(count) => println!("   ✅ {} CUDA device(s) found", count),
        Err(e) => println!("   ❌ Failed to get device count: {:?}", e),
    }

    // Runtime detection
    println!("\n4. Runtime backend detection:");
    let dispatcher = gllm_kernels::KernelDispatcher::new();
    println!("   Selected: {:?}", dispatcher.backend());

    // Also test detect_backend directly
    println!("\n5. Direct detect_backend call:");
    let result = gllm_kernels::detect_backend();
    println!("   Result: {:?}", result);
}
