use std::sync::Arc;
use metal::{Device, ComputePipelineDescriptor, ComputePipelineState, MTLSize};
// use crate::metal_kernels::metallib_loader::{MetallibCollection, MetallibLoadError}; // Imports handled via crate::metal_kernels usually
use crate::metal_kernels::MetallibCollection;
use crate::wgpu_kernels::LinearParams; // Share params struct
use crate::gpu_types::GpuTensor;

/// Embedded metallib data
static LINEAR_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "linear",
    metallib_data: include_bytes!("kernels/linear.metallib"),
};

pub struct MetalLinear {
    pipeline: ComputePipelineState,
    fused_pipeline: ComputePipelineState,
}

impl MetalLinear {
    pub fn new(device: &Device) -> Result<Self, String> {
        // Try loading metallib first
        let library = LINEAR_METALLIB.load(device).map_err(|e| e.to_string())?;
        
        let function = library.get_function("linear_forward_kernel", None)
            .map_err(|e| format!("Failed to finding function: {:?}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        let fused_function = library.get_function("fused_gate_up_silu", None)
            .map_err(|e| format!("Failed to finding fused function: {:?}", e))?;
            
        let fused_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_function)
            .map_err(|e| format!("Failed to create fused pipeline: {:?}", e))?;

        Ok(Self { pipeline, fused_pipeline })
    }

    pub fn forward(
        &self,
        params: LinearParams,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        bias: Option<&metal::Buffer>,
        output: &metal::Buffer,
        batch_size: usize,
    ) -> Result<(), String> {
        let command_queue = self.pipeline.device().new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        
        // Handle bias
        if let Some(b) = bias {
             encoder.set_buffer(2, Some(b), 0);
        } else {
             // Pass dummy or null? implementation in shader checks has_bias flag.
             // If has_bias is 0, buffer(2) access should be avoided.
             // Metal allows null buffer if not accessed.
             encoder.set_buffer(2, None, 0); 
        }

        encoder.set_buffer(3, Some(output), 0);
        
        // Scalars
        let in_features = params.in_features as i32;
        let out_features = params.out_features as i32;
        let has_bias = if bias.is_some() { 1i32 } else { 0i32 };
        
        encoder.set_bytes(4, 4, &in_features as *const i32 as *const _);
        encoder.set_bytes(5, 4, &out_features as *const i32 as *const _);
        encoder.set_bytes(6, 4, &has_bias as *const i32 as *const _);

        // Grid (x=out, y=batch)
        let grid_size = MTLSize {
            width: params.out_features as u64,
            height: batch_size as u64,
            depth: 1
        };
        
        let thread_group_size = MTLSize {
            width: 32,
            height: 1, // threads per group
            depth: 1
        }; // Basic tuning

        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed(); // Blocking for now to match sync dispatcher capability
        
        Ok(())
    }

    pub fn fused_gate_up_silu(
        &self,
        params: LinearParams,
        input: &metal::Buffer,
        weight_gate: &metal::Buffer,
        weight_up: &metal::Buffer,
        output: &metal::Buffer,
        batch_size: usize,
    ) -> Result<(), String> {
        let command_queue = self.fused_pipeline.device().new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.fused_pipeline);

        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight_gate), 0);
        encoder.set_buffer(2, Some(weight_up), 0);
        encoder.set_buffer(3, Some(output), 0);
        
        let in_features = params.in_features as i32;
        let out_features = params.out_features as i32;
        encoder.set_bytes(4, 4, &in_features as *const i32 as *const _);
        encoder.set_bytes(5, 4, &out_features as *const i32 as *const _);

        let grid_size = MTLSize {
            width: params.out_features as u64,
            height: batch_size as u64,
            depth: 1
        };
        
        let thread_group_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1
        };

        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
}
