use crate::traits::{Element, Kernels};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::marker::PhantomData;

pub mod load;

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct CudaKernels<E> {
    device: Arc<CudaDevice>,
    _phantom: PhantomData<E>,
}

impl<E> CudaKernels<E> {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            _phantom: PhantomData,
        }
    }
}

impl<E: Element> Kernels<E> for CudaKernels<E> {
    fn vec_add(&self, _a: &[E], _b: &[E], _out: &mut [E]) {
        unimplemented!("CUDA vec_add kernel not implemented");
    }

    fn vec_mul(&self, _a: &[E], _b: &[E], _out: &mut [E]) {
        unimplemented!("CUDA vec_mul kernel not implemented");
    }

    fn silu(&self, _a: &[E], _out: &mut [E]) {
         unimplemented!("CUDA silu kernel not implemented");
    }

    fn gemm(&self, _a: &[E], _b: &[E], _c: &mut [E], _m: usize, _n: usize, _k: usize) {
         unimplemented!("CUDA gemm kernel not implemented");
    }
    
    fn dequant_q4_k(&self, _block: &[u8], _out: &mut [f32]) {
        unimplemented!("CUDA dequant_q4_k not implemented");
    }

    fn dequant_q8_k(&self, _block: &[u8], _out: &mut [f32]) {
        unimplemented!("CUDA dequant_q8_k not implemented");
    }
}
