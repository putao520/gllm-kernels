use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_Q4_DEQUANTIZE: &str = "q4_0_dequantize_f32";
const KERNEL_AWQ_DEQUANTIZE: &str = "awq_dequantize_f32";
const DEFAULT_BLOCK_SIZE: u32 = 256;

static QUANTIZED_PTX: PtxCollection = PtxCollection {
    kernel_name: "quantized",
    ptx_versions: &[
        (61, include_str!("../kernels/quantized_sm61.ptx")),
        (80, include_str!("../kernels/quantized.ptx")),
    ],
};

#[derive(Debug)]
pub enum QuantizedDequantError {
    Driver(DriverError),
    InvalidConfig(String),
    KernelMissing(&'static str),
    PtxLoad(PtxLoadError),
}

impl fmt::Display for QuantizedDequantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for QuantizedDequantError {}

impl From<DriverError> for QuantizedDequantError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for QuantizedDequantError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

pub struct QuantizedDequantKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_q4: CudaFunction,
    kernel_awq: CudaFunction,
}

impl QuantizedDequantKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, QuantizedDequantError> {
        let ptx = QUANTIZED_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_q4 = module
            .load_function(KERNEL_Q4_DEQUANTIZE)
            .map_err(|_| QuantizedDequantError::KernelMissing(KERNEL_Q4_DEQUANTIZE))?;
        let kernel_awq = module
            .load_function(KERNEL_AWQ_DEQUANTIZE)
            .map_err(|_| QuantizedDequantError::KernelMissing(KERNEL_AWQ_DEQUANTIZE))?;

        Ok(Self {
            module,
            kernel_q4,
            kernel_awq,
        })
    }

    pub fn dequantize_q4(
        &self,
        stream: &Arc<CudaStream>,
        q_weight: &CudaSlice<u8>,
        scales: &CudaSlice<f16>,
        num_blocks: usize,
    ) -> Result<CudaSlice<f32>, QuantizedDequantError> {
        if num_blocks == 0 {
            return Err(QuantizedDequantError::InvalidConfig(
                "num_blocks must be > 0".into(),
            ));
        }
        let total_values = num_blocks
            .checked_mul(32)
            .ok_or_else(|| QuantizedDequantError::InvalidConfig("output overflow".into()))?;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(total_values)?;

        let cfg = LaunchConfig {
            grid_dim: ((total_values as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_blocks_i32 = num_blocks as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_q4);
            builder.arg(q_weight);
            builder.arg(scales);
            builder.arg(&mut output);
            builder.arg(&num_blocks_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    pub fn dequantize_awq(
        &self,
        stream: &Arc<CudaStream>,
        qweight: &CudaSlice<u32>,
        qzeros: &CudaSlice<u32>,
        scales: &CudaSlice<f16>,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<CudaSlice<f32>, QuantizedDequantError> {
        if n == 0 || k == 0 || group_size == 0 {
            return Err(QuantizedDequantError::InvalidConfig(
                "n, k, and group_size must be > 0".into(),
            ));
        }
        let total_values = n
            .checked_mul(k)
            .ok_or_else(|| QuantizedDequantError::InvalidConfig("output overflow".into()))?;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(total_values)?;

        let groups = (k / group_size) as i32;
        let cfg = LaunchConfig {
            grid_dim: ((total_values as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let group_size_i32 = group_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_awq);
            builder.arg(qweight);
            builder.arg(qzeros);
            builder.arg(scales);
            builder.arg(&mut output);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&group_size_i32);
            builder.arg(&groups);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}
