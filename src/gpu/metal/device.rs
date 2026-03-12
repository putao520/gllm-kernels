//! MetalDevice — `GpuDevice` implementation backed by Metal.framework.
//!
//! Uses the Objective-C runtime FFI from `objc_runtime` to call Metal APIs.
//! No external crate dependencies.

use std::sync::Arc;

use super::objc_runtime::{self, Id, MetalFramework, NSUInteger, NIL};
use crate::gpu::{GpuBuffer, GpuDevice, GpuError, GpuStream};

// ── MetalBuffer ─────────────────────────────────────────────────────────────

/// A Metal device buffer (`MTLBuffer`).
pub struct MetalBuffer {
    /// Raw `id<MTLBuffer>` pointer.
    raw: Id,
    len: usize,
}

// SAFETY: Metal buffers are thread-safe once created.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl GpuBuffer for MetalBuffer {
    fn as_device_ptr(&self) -> u64 {
        // [buffer contents] returns the CPU-visible pointer for shared/managed buffers.
        unsafe {
            let sel = objc_runtime::sel("contents");
            let ptr: Id = objc_runtime::objc_msgSend(self.raw, sel);
            ptr as u64
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let sel = objc_runtime::sel("release");
                objc_runtime::objc_msgSend(self.raw, sel);
            }
        }
    }
}

// ── MetalStream ─────────────────────────────────────────────────────────────

/// A Metal command queue (`MTLCommandQueue`), analogous to a CUDA stream.
pub struct MetalStream {
    /// Raw `id<MTLCommandQueue>` pointer.
    raw: Id,
}

// SAFETY: Metal command queues are thread-safe.
unsafe impl Send for MetalStream {}
unsafe impl Sync for MetalStream {}

impl GpuStream for MetalStream {
    fn synchronize(&self) -> Result<(), GpuError> {
        // Create a command buffer, commit it, and wait until completed.
        // This acts as a barrier for all previously enqueued work.
        unsafe {
            let cmd_buf_sel = objc_runtime::sel("commandBuffer");
            let cmd_buf: Id = objc_runtime::objc_msgSend(self.raw, cmd_buf_sel);
            if cmd_buf.is_null() {
                return Err(GpuError::Driver("failed to create command buffer for sync".into()));
            }

            let commit_sel = objc_runtime::sel("commit");
            objc_runtime::objc_msgSend(cmd_buf, commit_sel);

            let wait_sel = objc_runtime::sel("waitUntilCompleted");
            objc_runtime::objc_msgSend(cmd_buf, wait_sel);
        }
        Ok(())
    }
}

impl Drop for MetalStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let sel = objc_runtime::sel("release");
                objc_runtime::objc_msgSend(self.raw, sel);
            }
        }
    }
}

// ── MetalDevice ─────────────────────────────────────────────────────────────

/// Apple Metal GPU device.
///
/// Wraps a `MTLDevice` obtained via `MTLCreateSystemDefaultDevice()`.
/// All Metal API calls go through `objc_msgSend`.
pub struct MetalDevice {
    /// Raw `id<MTLDevice>` pointer.
    raw: Id,
    /// Cached device name.
    name: String,
    /// Default command queue.
    default_stream: MetalStream,
    /// Keep framework alive.
    _framework: Arc<MetalFramework>,
}

// SAFETY: MTLDevice is thread-safe.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

/// Metal resource storage mode: shared (CPU+GPU visible).
const MTL_RESOURCE_STORAGE_MODE_SHARED: NSUInteger = 0;

impl MetalDevice {
    /// Create a `MetalDevice` using the system default Metal device.
    ///
    /// Returns `Err` if Metal.framework is unavailable or no GPU is found.
    pub fn new() -> Result<Self, GpuError> {
        let framework = MetalFramework::load()
            .ok_or_else(|| GpuError::DeviceNotFound("Metal.framework not available".into()))?;

        let raw = unsafe { (framework.create_default_device)() };
        if raw.is_null() {
            return Err(GpuError::DeviceNotFound("MTLCreateSystemDefaultDevice returned nil".into()));
        }

        // Read device name: [device name] -> NSString
        let name = unsafe {
            let name_sel = objc_runtime::sel("name");
            let ns_name: Id = objc_runtime::objc_msgSend(raw, name_sel);
            objc_runtime::nsstring_to_string(ns_name)
        };

        // Create default command queue: [device newCommandQueue]
        let queue = unsafe {
            let sel = objc_runtime::sel("newCommandQueue");
            objc_runtime::objc_msgSend(raw, sel)
        };
        if queue.is_null() {
            return Err(GpuError::Driver("failed to create default command queue".into()));
        }

        let framework = Arc::new(framework);

        Ok(Self {
            raw,
            name,
            default_stream: MetalStream { raw: queue },
            _framework: framework,
        })
    }

    /// Get the raw `MTLDevice` pointer (for advanced usage).
    pub fn raw_device(&self) -> Id {
        self.raw
    }

    /// Compile MSL source into a `MTLLibrary`.
    ///
    /// Uses `[device newLibraryWithSource:options:error:]`.
    /// Returns the raw `id<MTLLibrary>` pointer.
    pub fn compile_msl(&self, source: &str) -> Result<Id, GpuError> {
        use std::ffi::CString;

        unsafe {
            // Create NSString from source
            let ns_string_class = objc_runtime::class("NSString");
            let alloc_sel = objc_runtime::sel("alloc");
            let init_sel = objc_runtime::sel("initWithUTF8String:");
            let c_source = CString::new(source)
                .map_err(|e| GpuError::ShaderCompilation(format!("invalid MSL source: {e}")))?;

            let ns_alloc: Id = objc_runtime::objc_msgSend(ns_string_class, alloc_sel);
            let ns_source: Id = objc_runtime::objc_msgSend(ns_alloc, init_sel, c_source.as_ptr());
            if ns_source.is_null() {
                return Err(GpuError::ShaderCompilation(
                    "failed to create NSString from MSL source".into(),
                ));
            }

            // [device newLibraryWithSource:options:error:]
            let sel = objc_runtime::sel("newLibraryWithSource:options:error:");
            let mut error: Id = NIL;
            let library: Id = objc_runtime::objc_msgSend(
                self.raw,
                sel,
                ns_source,
                NIL, // options (nil = defaults)
                &mut error as *mut Id,
            );

            // Release the NSString
            let release_sel = objc_runtime::sel("release");
            objc_runtime::objc_msgSend(ns_source, release_sel);

            if library.is_null() {
                let err_msg = if !error.is_null() {
                    let desc_sel = objc_runtime::sel("localizedDescription");
                    let desc: Id = objc_runtime::objc_msgSend(error, desc_sel);
                    objc_runtime::nsstring_to_string(desc)
                } else {
                    "unknown error".into()
                };
                return Err(GpuError::ShaderCompilation(format!(
                    "MSL compilation failed: {err_msg}"
                )));
            }

            Ok(library)
        }
    }

    /// Load a pre-compiled Metal library from AIR bitcode data.
    ///
    /// Uses `[device newLibraryWithData:error:]`.
    /// Returns the raw `id<MTLLibrary>` pointer.
    pub fn load_library_data(&self, data: &[u8]) -> Result<Id, GpuError> {
        unsafe {
            // Wrap data in NSData: [NSData dataWithBytes:length:]
            let nsdata_class = objc_runtime::class("NSData");
            let sel = objc_runtime::sel("dataWithBytes:length:");
            let nsdata: Id = objc_runtime::objc_msgSend(
                nsdata_class,
                sel,
                data.as_ptr(),
                data.len() as NSUInteger,
            );
            if nsdata.is_null() {
                return Err(GpuError::ShaderCompilation(
                    "failed to create NSData from AIR bitcode".into(),
                ));
            }

            // [device newLibraryWithData:error:]
            // Note: Metal expects dispatch_data_t, but NSData toll-free bridges to it.
            let lib_sel = objc_runtime::sel("newLibraryWithData:error:");
            let mut error: Id = NIL;
            let library: Id = objc_runtime::objc_msgSend(
                self.raw,
                lib_sel,
                nsdata,
                &mut error as *mut Id,
            );

            if library.is_null() {
                let err_msg = if !error.is_null() {
                    let desc_sel = objc_runtime::sel("localizedDescription");
                    let desc: Id = objc_runtime::objc_msgSend(error, desc_sel);
                    objc_runtime::nsstring_to_string(desc)
                } else {
                    "unknown error".into()
                };
                return Err(GpuError::ShaderCompilation(format!(
                    "AIR library load failed: {err_msg}"
                )));
            }

            Ok(library)
        }
    }
}

impl GpuDevice for MetalDevice {
    type Buffer = MetalBuffer;
    type Stream = MetalStream;

    fn name(&self) -> &str {
        &self.name
    }

    fn total_memory(&self) -> usize {
        // [device recommendedMaxWorkingSetSize] -> uint64_t
        unsafe {
            let sel = objc_runtime::sel("recommendedMaxWorkingSetSize");
            let size: u64 = std::mem::transmute(objc_runtime::objc_msgSend(self.raw, sel));
            size as usize
        }
    }

    fn free_memory(&self) -> usize {
        // Metal doesn't expose free memory directly.
        // currentAllocatedSize gives us how much is currently allocated.
        // free ~ total - allocated (best-effort).
        let total = self.total_memory();
        let allocated = unsafe {
            let sel = objc_runtime::sel("currentAllocatedSize");
            let size: u64 = std::mem::transmute(objc_runtime::objc_msgSend(self.raw, sel));
            size as usize
        };
        total.saturating_sub(allocated)
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Buffer, GpuError> {
        // [device newBufferWithLength:options:]
        unsafe {
            let sel = objc_runtime::sel("newBufferWithLength:options:");
            let buf: Id = objc_runtime::objc_msgSend(
                self.raw,
                sel,
                bytes as NSUInteger,
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            if buf.is_null() {
                return Err(GpuError::OutOfMemory {
                    requested: bytes,
                    available: self.free_memory(),
                });
            }
            Ok(MetalBuffer { raw: buf, len: bytes })
        }
    }

    fn alloc_zeros(&self, bytes: usize) -> Result<Self::Buffer, GpuError> {
        let buf = self.alloc(bytes)?;
        // Zero-fill: shared buffers are CPU-accessible via [buffer contents]
        unsafe {
            let contents_sel = objc_runtime::sel("contents");
            let ptr: *mut u8 =
                std::mem::transmute(objc_runtime::objc_msgSend(buf.raw, contents_sel));
            if !ptr.is_null() {
                std::ptr::write_bytes(ptr, 0, bytes);
            }
        }
        Ok(buf)
    }

    fn htod(
        &self,
        src: &[u8],
        dst: &mut Self::Buffer,
        _stream: &Self::Stream,
    ) -> Result<(), GpuError> {
        // For shared storage mode, CPU and GPU share the same memory.
        // Just memcpy into the buffer's contents pointer.
        unsafe {
            let contents_sel = objc_runtime::sel("contents");
            let ptr: *mut u8 =
                std::mem::transmute(objc_runtime::objc_msgSend(dst.raw, contents_sel));
            if ptr.is_null() {
                return Err(GpuError::Transfer("buffer contents pointer is null".into()));
            }
            if src.len() > dst.len {
                return Err(GpuError::Transfer(format!(
                    "source ({} bytes) exceeds buffer size ({} bytes)",
                    src.len(),
                    dst.len
                )));
            }
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
        }
        Ok(())
    }

    fn dtoh(
        &self,
        src: &Self::Buffer,
        dst: &mut [u8],
        _stream: &Self::Stream,
    ) -> Result<(), GpuError> {
        unsafe {
            let contents_sel = objc_runtime::sel("contents");
            let ptr: *const u8 =
                std::mem::transmute(objc_runtime::objc_msgSend(src.raw, contents_sel));
            if ptr.is_null() {
                return Err(GpuError::Transfer("buffer contents pointer is null".into()));
            }
            let copy_len = dst.len().min(src.len);
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), copy_len);
        }
        Ok(())
    }

    fn dtod(
        &self,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        stream: &Self::Stream,
    ) -> Result<(), GpuError> {
        // Use a blit command encoder for device-to-device copy.
        unsafe {
            let cmd_buf_sel = objc_runtime::sel("commandBuffer");
            let cmd_buf: Id = objc_runtime::objc_msgSend(stream.raw, cmd_buf_sel);
            if cmd_buf.is_null() {
                return Err(GpuError::Transfer(
                    "failed to create command buffer for dtod".into(),
                ));
            }

            let blit_sel = objc_runtime::sel("blitCommandEncoder");
            let blit: Id = objc_runtime::objc_msgSend(cmd_buf, blit_sel);
            if blit.is_null() {
                return Err(GpuError::Transfer("failed to create blit encoder".into()));
            }

            // [blit copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:]
            let copy_sel =
                objc_runtime::sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
            let copy_len = src.len.min(dst.len) as NSUInteger;
            objc_runtime::objc_msgSend(
                blit,
                copy_sel,
                src.raw,
                0 as NSUInteger,
                dst.raw,
                0 as NSUInteger,
                copy_len,
            );

            let end_sel = objc_runtime::sel("endEncoding");
            objc_runtime::objc_msgSend(blit, end_sel);

            let commit_sel = objc_runtime::sel("commit");
            objc_runtime::objc_msgSend(cmd_buf, commit_sel);
        }
        Ok(())
    }

    fn create_stream(&self) -> Result<Self::Stream, GpuError> {
        unsafe {
            let sel = objc_runtime::sel("newCommandQueue");
            let queue: Id = objc_runtime::objc_msgSend(self.raw, sel);
            if queue.is_null() {
                return Err(GpuError::Driver("failed to create command queue".into()));
            }
            Ok(MetalStream { raw: queue })
        }
    }

    fn default_stream(&self) -> &Self::Stream {
        &self.default_stream
    }

    fn sync(&self) -> Result<(), GpuError> {
        self.default_stream.synchronize()
    }
}
