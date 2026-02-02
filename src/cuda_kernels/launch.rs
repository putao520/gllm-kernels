use std::ffi::c_void;

use cudarc::driver::{result, sys, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, SyncOnDrop};

use crate::backend_trait::BackendResult;

pub(crate) struct KernelLaunch<'a> {
    func: sys::CUfunction,
    stream: &'a CudaStream,
    config: LaunchConfig,
    args: Vec<*mut c_void>,
    device_ptrs: Vec<Box<sys::CUdeviceptr>>,
    _syncs: Vec<SyncOnDrop<'a>>,
}

impl<'a> KernelLaunch<'a> {
    pub fn new(
        func: sys::CUfunction,
        stream: &'a CudaStream,
        config: LaunchConfig,
        arg_capacity: usize,
    ) -> Self {
        Self {
            func,
            stream,
            config,
            args: Vec::with_capacity(arg_capacity),
            device_ptrs: Vec::new(),
            _syncs: Vec::new(),
        }
    }

    pub fn arg_device<T, S>(&mut self, arg: &'a S) -> &mut Self
    where
        S: DevicePtr<T>,
    {
        let (ptr, sync) = arg.device_ptr(self.stream);
        self.device_ptrs.push(Box::new(ptr));
        let ptr_ref = self.device_ptrs.last_mut().expect("device ptr just pushed");
        self.args
            .push((&mut **ptr_ref) as *mut sys::CUdeviceptr as *mut c_void);
        self._syncs.push(sync);
        self
    }

    pub fn arg_device_mut<T, S>(&mut self, arg: &'a mut S) -> &mut Self
    where
        S: DevicePtrMut<T>,
    {
        let (ptr, sync) = arg.device_ptr_mut(self.stream);
        self.device_ptrs.push(Box::new(ptr));
        let ptr_ref = self.device_ptrs.last_mut().expect("device ptr just pushed");
        self.args
            .push((&mut **ptr_ref) as *mut sys::CUdeviceptr as *mut c_void);
        self._syncs.push(sync);
        self
    }

    pub fn arg_device_ptr(&mut self, ptr: sys::CUdeviceptr) -> &mut Self {
        self.device_ptrs.push(Box::new(ptr));
        let ptr_ref = self.device_ptrs.last_mut().expect("device ptr just pushed");
        self.args
            .push((&mut **ptr_ref) as *mut sys::CUdeviceptr as *mut c_void);
        self
    }

    pub fn arg_scalar<T>(&mut self, arg: &'a T) -> &mut Self {
        self.args.push(arg as *const T as *mut c_void);
        self
    }

    pub unsafe fn launch(mut self) -> BackendResult<()> {
        self.stream.context().bind_to_thread()?;
        result::launch_kernel(
            self.func,
            self.config.grid_dim,
            self.config.block_dim,
            self.config.shared_mem_bytes,
            self.stream.cu_stream(),
            &mut self.args,
        )?;
        Ok(())
    }
}
