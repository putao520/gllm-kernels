use metal::{ComputePipelineState, Device, MTLSize};

use crate::metal_kernels::MetallibCollection;

static RMS_NORM_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "rms_norm",
    metallib_data: include_bytes!("kernels/rms_norm.metallib"),
};

pub struct MetalRmsNorm {
    pipeline: ComputePipelineState,
}

impl MetalRmsNorm {
    pub fn new(device: &Device) -> Result<Self, String> {
        let library = RMS_NORM_METALLIB.load(device).map_err(|e| e.to_string())?;
        let function = library
            .get_function("rms_norm_f32", None)
            .map_err(|e| format!("Failed to find function: {:?}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        Ok(Self { pipeline })
    }

    pub fn forward(
        &self,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        output: &metal::Buffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        if rows == 0 || hidden == 0 {
            return Ok(());
        }

        let command_queue = self.pipeline.device().new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);

        let hidden_i32 = hidden as i32;
        let rows_i32 = rows as i32;
        encoder.set_bytes(3, 4, &hidden_i32 as *const i32 as *const _);
        encoder.set_bytes(4, 4, &rows_i32 as *const i32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);

        let grid_size = MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}
