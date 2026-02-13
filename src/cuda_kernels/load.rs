use std::sync::Arc;
use cudarc::driver::{CudaDevice, DriverError};
use crate::quant::QuantType;

const PTX_SRC: &str = include_str!("gemm.ptx");

/// Loads the appropriate CUDA module for the given device architecture and quantization type.
pub fn load_kernels(device: &Arc<CudaDevice>, _quant_type: QuantType) -> Result<(), DriverError> {
    // Determine architecture (e.g. sm_80, sm_86)
    // cudarc device.attribute(Attribute::ComputeCapabilityMajor)?
    // For now, we stub this connection.
    
    // Load the PTX module
    // We use a fixed module name "gemm" and function name "sgemm"
    // In a real system, we'd select the PTX based on arch, but single file for now.
    
    // Note: ptx file must be valid. The nvcc command should have succeeded.
    
    device.load_ptx(PTX_SRC.into(), "gemm", &["sgemm"])?;
    
    Ok(())
}
