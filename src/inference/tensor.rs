//! Unified tensor handle for CPU and GPU backends.

use crate::traits::Element;
use crate::inference::types::{DType, InferenceError};

/// Device kind for tensor placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda(u32),
    Metal(u32),
}

/// A device-resident tensor with type-erased element type.
///
/// On CPU, `ptr` is a host pointer (zero overhead).
/// On GPU, `ptr` is a device pointer (data stays on device).
pub struct DeviceTensor {
    ptr: *mut u8,
    len_bytes: usize,
    num_elements: usize,
    dtype: DType,
    device: DeviceKind,
    owned: bool,
    /// True when created from `from_slice` (immutable source).
    /// Mutable access (`as_mut_ptr`, `as_mut_slice`) will panic.
    immutable: bool,
}

unsafe impl Send for DeviceTensor {}
unsafe impl Sync for DeviceTensor {}

impl DeviceTensor {
    /// Allocate a zeroed CPU tensor.
    pub fn alloc_cpu(num_elements: usize, dtype: DType) -> Result<Self, InferenceError> {
        let len_bytes = num_elements * dtype.size_bytes();
        if len_bytes == 0 {
            return Ok(DeviceTensor {
                ptr: std::ptr::null_mut(),
                len_bytes: 0,
                num_elements: 0,
                dtype,
                device: DeviceKind::Cpu,
                owned: false,
                immutable: false,
            });
        }
        let layout = std::alloc::Layout::from_size_align(len_bytes, 64)
            .map_err(|e| InferenceError::RuntimeError(format!("layout error: {e}")))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(InferenceError::OutOfMemory {
                requested: len_bytes,
                available: 0,
            });
        }
        Ok(DeviceTensor {
            ptr,
            len_bytes,
            num_elements,
            dtype,
            device: DeviceKind::Cpu,
            owned: true,
            immutable: false,
        })
    }

    /// Wrap an existing typed slice as a non-owning CPU tensor.
    ///
    /// # Safety
    /// The caller must ensure the slice outlives this tensor.
    pub unsafe fn from_slice<E: Element>(data: &[E]) -> Self {
        DeviceTensor {
            ptr: data.as_ptr() as *mut u8,
            len_bytes: data.len() * std::mem::size_of::<E>(),
            num_elements: data.len(),
            dtype: dtype_from_elem_id(E::ELEM_ID),
            device: DeviceKind::Cpu,
            owned: false,
            immutable: true,
        }
    }

    /// Wrap an existing mutable typed slice as a non-owning CPU tensor.
    ///
    /// # Safety
    /// The caller must ensure the slice outlives this tensor.
    pub unsafe fn from_mut_slice<E: Element>(data: &mut [E]) -> Self {
        DeviceTensor {
            ptr: data.as_mut_ptr() as *mut u8,
            len_bytes: data.len() * std::mem::size_of::<E>(),
            num_elements: data.len(),
            dtype: dtype_from_elem_id(E::ELEM_ID),
            device: DeviceKind::Cpu,
            owned: false,
            immutable: false,
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 { self.ptr }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        assert!(!self.immutable, "cannot mutate immutable tensor created from from_slice");
        self.ptr
    }

    /// View as a typed slice (CPU only).
    ///
    /// # Safety
    /// Caller must ensure the dtype matches E and the tensor is on CPU.
    #[inline]
    pub unsafe fn as_slice<E: Element>(&self) -> &[E] {
        debug_assert_eq!(self.device, DeviceKind::Cpu);
        std::slice::from_raw_parts(self.ptr as *const E, self.num_elements)
    }

    /// View as a mutable typed slice (CPU only).
    ///
    /// # Safety
    /// Caller must ensure the dtype matches E and the tensor is on CPU.
    #[inline]
    pub unsafe fn as_mut_slice<E: Element>(&mut self) -> &mut [E] {
        assert!(!self.immutable, "cannot mutate immutable tensor created from from_slice");
        debug_assert_eq!(self.device, DeviceKind::Cpu);
        std::slice::from_raw_parts_mut(self.ptr as *mut E, self.num_elements)
    }

    #[inline]
    pub fn len_bytes(&self) -> usize { self.len_bytes }
    #[inline]
    pub fn num_elements(&self) -> usize { self.num_elements }
    #[inline]
    pub fn dtype(&self) -> DType { self.dtype }
    #[inline]
    pub fn device(&self) -> DeviceKind { self.device }
    #[inline]
    pub fn is_cpu(&self) -> bool { self.device == DeviceKind::Cpu }
}

impl Drop for DeviceTensor {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() && self.len_bytes > 0 {
            if let DeviceKind::Cpu = self.device {
                let layout = std::alloc::Layout::from_size_align(self.len_bytes, 64)
                    .expect("invalid layout in drop");
                unsafe { std::alloc::dealloc(self.ptr, layout); }
            }
        }
    }
}

fn dtype_from_elem_id(id: u8) -> DType {
    match id {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        _ => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_cpu() {
        let t = DeviceTensor::alloc_cpu(1024, DType::F32).unwrap();
        assert_eq!(t.num_elements(), 1024);
        assert_eq!(t.len_bytes(), 4096);
        assert!(t.is_cpu());
        assert_eq!(t.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0];
        let t = unsafe { DeviceTensor::from_slice(&data) };
        assert_eq!(t.num_elements(), 3);
        assert_eq!(t.dtype(), DType::F32);
        let s: &[f32] = unsafe { t.as_slice() };
        assert_eq!(s, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_alloc_zero_elements() {
        let t = DeviceTensor::alloc_cpu(0, DType::F32).unwrap();
        assert_eq!(t.num_elements(), 0);
        assert_eq!(t.len_bytes(), 0);
    }

    #[test]
    #[should_panic(expected = "cannot mutate immutable tensor created from from_slice")]
    fn test_immutable_tensor_as_mut_ptr_panics() {
        let data = vec![1.0f32, 2.0, 3.0];
        let mut t = unsafe { DeviceTensor::from_slice(&data) };
        let _ = t.as_mut_ptr();
    }

    #[test]
    #[should_panic(expected = "cannot mutate immutable tensor created from from_slice")]
    fn test_immutable_tensor_as_mut_slice_panics() {
        let data = vec![1.0f32, 2.0, 3.0];
        let mut t = unsafe { DeviceTensor::from_slice(&data) };
        let _ = unsafe { t.as_mut_slice::<f32>() };
    }

    #[test]
    fn test_mutable_tensor_from_mut_slice() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        let mut t = unsafe { DeviceTensor::from_mut_slice(&mut data) };
        // Should not panic — created from mutable source
        let _ = t.as_mut_ptr();
        let s: &mut [f32] = unsafe { t.as_mut_slice() };
        s[0] = 42.0;
        assert_eq!(s[0], 42.0);
    }
}
