use std::sync::Arc;

use crate::runtime_detection::{detect_backend, BackendType};

pub use crate::backend_trait::{Backend, TensorSlice, TensorSliceMut};
pub use crate::cpu_backend::CpuBackend;
pub use crate::cuda_backend::CudaBackend;
pub use crate::metal_backend::MetalBackend;
pub use crate::rocm_backend::RocmBackend;
pub use crate::wgpu_backend::WgpuBackend;

pub fn auto_select_backend() -> Arc<dyn Backend> {
    match detect_backend() {
        BackendType::Cuda => Arc::new(CudaBackend::new()),
        BackendType::Rocm => Arc::new(RocmBackend::new()),
        BackendType::Metal => Arc::new(MetalBackend::new()),
        BackendType::Wgpu => Arc::new(WgpuBackend::new()),
        BackendType::Cpu => Arc::new(CpuBackend::new()),
    }
}
