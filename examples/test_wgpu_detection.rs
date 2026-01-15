//! WGPU Detection Debug

fn main() {
    println!("=== WGPU Detection Debug ===\n");
    
    // Direct WGPU test
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    
    println!("Instance created");
    
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).ok();

    match adapter {
        Some(a) => {
            let info = a.get_info();
            println!("✅ Adapter found:");
            println!("   Name: {}", info.name);
            println!("   Vendor: {}", info.vendor);
            println!("   Device: {}", info.device);
            println!("   Backend: {:?}", info.backend);
            println!("   Driver: {}", info.driver);
        }
        None => {
            println!("❌ No adapter found");
        }
    }
    
    // Try with different power preference
    let adapter_low = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        ..Default::default()
    })).ok();

    if adapter_low.is_some() {
        println!("\n✅ LowPower adapter also available");
    }
    
    // Enumerate all adapters
    println!("\n=== All Available Adapters ===");
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    for (i, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        println!("Adapter {}: {} ({:?})", i, info.name, info.backend);
    }
}
