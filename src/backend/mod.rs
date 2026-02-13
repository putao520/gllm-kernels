use crate::traits::{Backend, Element};
use crate::cpu_kernels::CpuKernels;

/// The CPU backend (SPEC/03 §2.2).
pub struct CpuBackend;

impl Backend for CpuBackend {
    const NAME: &'static str = "cpu";

    type Kernels<E: Element> = CpuKernels<E>;

    fn init<E: Element>() -> Self::Kernels<E> {
        CpuKernels::new()
    }
}
