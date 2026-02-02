pub mod backend;
pub mod backend_trait;
pub mod cpu_backend;
pub mod cpu_kernels;
pub mod cuda_backend;
pub mod cuda_kernels;
pub mod gpu_types;
pub mod kernel_types;
pub mod ops;
pub mod quantization;

pub use backend::{auto_select_backend, BackendKind};
pub use backend_trait::{
    AttentionTopology, Backend, BackendError, BackendResult, KvCacheHandle, LogitsHandle,
    LogitsTensor, TensorLookup,
};
pub use cpu_backend::CpuBackend;
pub use cuda_backend::{CudaBackend, PerfMetrics};
pub use kernel_types::{DType, GeneratorForwardConfig, PackedBits, PositionEncoding, SamplingConfig};
pub use quantization::{Block, BlockwiseMatrix, PackedU8};
