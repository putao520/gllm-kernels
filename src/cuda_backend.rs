use crate::traits::{Backend, Element};
use crate::cuda_kernels::CudaKernels;

use cudarc::driver::CudaDevice;

/// The CUDA backend (SPEC/03 §2.2).
pub struct CudaBackend;

impl Backend for CudaBackend {
    const NAME: &'static str = "cuda";

    type Kernels<E: Element> = CudaKernels<E>;

    fn init<E: Element>() -> Self::Kernels<E> {
        let device = CudaDevice::new(0).expect("Failed to initialize CUDA device");
        CudaKernels::new(device)
    }
}
