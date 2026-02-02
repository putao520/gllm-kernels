use cudarc::driver::CudaSlice;

#[derive(Debug)]
pub struct GpuBuffer<T> {
    inner: CudaSlice<T>,
}

impl<T> GpuBuffer<T> {
    pub(crate) fn new(inner: CudaSlice<T>) -> Self {
        Self { inner }
    }

    pub(crate) fn as_inner(&self) -> &CudaSlice<T> {
        &self.inner
    }

    pub(crate) fn as_inner_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.inner
    }
}
