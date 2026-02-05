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
pub mod swap_manager;

pub use backend::{auto_select_backend, BackendKind};
pub use backend_trait::{
    AttentionTopology, Backend, BackendError, BackendResult, BatchInput, KvCacheHandle,
    LogitsHandle, LogitsTensor, SequenceInput, TensorLookup,
};
pub use cpu_backend::CpuBackend;
pub use cuda_backend::{CudaBackend, PerfMetrics};
pub use kernel_types::{
    DType, GeneratorForwardConfig, PackedBits, PageMetadata, PageState, PositionEncoding,
    SamplingConfig, SwapConfig,
};
pub use quantization::{
    dequantize_q4_0, dequantize_q5_k, dequantize_q8_0, Block, BlockwiseMatrix, PackedU8,
    Q4_0Block, Q4_0Matrix, Q4_0_BLOCK_BYTES, Q5_KBlock, Q5_KMatrix, Q8_0Block, Q8_0Matrix,
    Q8_0_BLOCK_BYTES, QK4_0, QK8_0, QuantizedType,
};
pub use swap_manager::{SwapManager, SwapStats};
