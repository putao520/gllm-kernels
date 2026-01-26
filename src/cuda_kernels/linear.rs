use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;
use crate::cuda_kernels::ptx_loader::PtxCollection;
use crate::wgpu_kernels::LinearParams;

const KERNEL_FORWARD: &str = "linear_forward_kernel";
const KERNEL_FUSED_GATE_UP_SILU: &str = "fused_gate_up_silu_kernel";

/// SM-aware PTX collection for Linear kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static LINEAR_PTX: PtxCollection = PtxCollection {
    kernel_name: "linear",
    ptx_versions: &[
        (61, include_str!("kernels/linear_sm61.ptx")),
        (80, include_str!("kernels/linear.ptx")),
    ],
};

pub struct CudaLinear {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    func: CudaFunction,
    fused_func: CudaFunction,
}

impl CudaLinear {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = LINEAR_PTX
            .load(ctx)
            .map_err(|e| format!("Failed to load linear PTX: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Failed to load linear PTX module: {:?}", e))?;

        let func = module
            .load_function(KERNEL_FORWARD)
            .map_err(|_| format!("Failed to find kernel function: {KERNEL_FORWARD}"))?;

        let fused_func = module
            .load_function(KERNEL_FUSED_GATE_UP_SILU)
            .map_err(|_| format!("Failed to find kernel function: {KERNEL_FUSED_GATE_UP_SILU}"))?;

        Ok(Self { module, func, fused_func })
    }

    pub fn forward(
        &self,
        stream: &cudarc::driver::CudaStream,
        params: LinearParams,
        input: &CudaSlice<u8>,
        weight: &CudaSlice<u8>,
        bias: Option<&CudaSlice<u8>>,
        output: &CudaSlice<u8>,
    ) -> Result<(), String> {
        let has_bias = if bias.is_some() { 1 } else { 0 };
        let cfg = LaunchConfig::for_num_elems(params.out_features);
        
        // Prepare scalars
        let in_features_i32 = params.in_features as i32;
        let out_features_i32 = params.out_features as i32;
        let has_bias_i32 = has_bias as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.func);
            builder.arg(input);
            builder.arg(weight);
            // Handle optional bias
            match bias {
                Some(b) => builder.arg(b),
                None => builder.arg(&0u64), // Null pointer
            };

            builder.arg(output);
            builder.arg(&in_features_i32);
            builder.arg(&out_features_i32);
            builder.arg(&has_bias_i32);
            
            builder.launch(cfg)
        }.map_err(|e| format!("Kernel launch failed: {:?}", e))?;

        Ok(())
    }

    pub fn fused_gate_up_silu(
        &self,
        stream: &cudarc::driver::CudaStream,
        params: LinearParams,
        input: &CudaSlice<u8>,
        weight_gate: &CudaSlice<u8>,
        weight_up: &CudaSlice<u8>,
        output: &CudaSlice<u8>,
        batch_size: usize,
    ) -> Result<(), String> {
        // Grid: [out_features, batch_size]
        let cfg = LaunchConfig {
            grid_dim: (
                (params.out_features as u32 + 255) / 256,
                batch_size as u32,
                1
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let in_features_i32 = params.in_features as i32;
        let out_features_i32 = params.out_features as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.fused_func);
            builder.arg(input);
            builder.arg(weight_gate);
            builder.arg(weight_up);
            builder.arg(output);
            builder.arg(&in_features_i32);
            builder.arg(&out_features_i32);
            
            builder.launch(cfg)
        }.map_err(|e| format!("Fused kernel launch failed: {:?}", e))?;

        Ok(())
    }
}
