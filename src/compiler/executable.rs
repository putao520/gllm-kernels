//! CompiledLayer — mmap'd executable code + metadata.
//!
//! Wraps a block of JIT-compiled machine code that implements a complete
//! transformer layer. The code is stored in an executable memory region
//! (via mmap) and called through a function pointer.

use crate::types::InferenceError;

/// Signature of a compiled layer function.
///
/// ```text
/// fn(input: *const u8, weights: *const u8, kv_cache: *mut u8,
///    positions: *const u32, seq_lens: *const usize,
///    batch_size: usize, seq_len: usize,
///    output: *mut u8, scratchpad: *mut u8)
/// ```
pub type CompiledLayerFn = unsafe extern "C" fn(
    *const u8, // input
    *const u8, // weights
    *mut u8,   // kv_cache
    *const u32, // positions
    *const usize, // seq_lens
    usize,     // batch_size
    usize,     // seq_len
    *mut u8,   // output
    *mut u8,   // scratchpad
);

/// A JIT-compiled transformer layer.
///
/// Contains executable machine code in an mmap'd region, plus metadata
/// for validation and cache management.
pub struct CompiledLayer {
    /// Executable memory region
    code: ExecutableBuffer,
    /// Entry point offset within the code buffer
    entry_offset: usize,
    /// Required scratchpad size in bytes
    pub scratchpad_bytes: usize,
    /// Hash of the LayerIR + ExecutionPlan (for cache validation)
    pub config_hash: u64,
}

impl CompiledLayer {
    /// Create a CompiledLayer from raw machine code bytes.
    pub fn from_code(
        code_bytes: &[u8],
        scratchpad_bytes: usize,
        config_hash: u64,
    ) -> Result<Self, InferenceError> {
        let code = ExecutableBuffer::new(code_bytes)?;
        Ok(CompiledLayer {
            code,
            entry_offset: 0,
            scratchpad_bytes,
            config_hash,
        })
    }

    /// Get the entry point function pointer.
    ///
    /// # Safety
    /// The caller must ensure the compiled code is valid and the arguments
    /// match the expected signature.
    #[inline]
    pub unsafe fn entry_point(&self) -> CompiledLayerFn {
        let ptr = self.code.ptr.add(self.entry_offset);
        std::mem::transmute(ptr)
    }

    /// Size of the compiled code in bytes.
    pub fn code_size(&self) -> usize {
        self.code.len
    }

    /// Execute the compiled layer.
    ///
    /// # Safety
    /// The caller must ensure all pointers are valid and the buffer sizes
    /// match the compiled graph's expected layout.
    #[inline]
    pub unsafe fn execute(
        &self,
        input: *const u8,
        weights: *const u8,
        kv_cache: *mut u8,
        positions: *const u32,
        seq_lens: *const usize,
        batch_size: usize,
        seq_len: usize,
        output: *mut u8,
        scratchpad: *mut u8,
    ) {
        let f = self.entry_point();
        f(
            input, weights, kv_cache, positions, seq_lens,
            batch_size, seq_len, output, scratchpad,
        );
    }
}

/// An executable memory buffer backed by mmap.
struct ExecutableBuffer {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: ExecutableBuffer owns its mmap'd memory exclusively. The pointer is
// never aliased and the buffer is immutable (PROT_READ|PROT_EXEC) after construction.
unsafe impl Send for ExecutableBuffer {}
unsafe impl Sync for ExecutableBuffer {}

impl ExecutableBuffer {
    /// Allocate an executable memory region and copy code into it.
    fn new(code: &[u8]) -> Result<Self, InferenceError> {
        if code.is_empty() {
            return Ok(ExecutableBuffer {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }

        // Round up to page size
        let page_size = page_size();
        let len = (code.len() + page_size - 1) & !(page_size - 1);

        // SAFETY: mmap with MAP_PRIVATE|MAP_ANONYMOUS creates a new anonymous mapping.
        // No file descriptor is used (-1). Return value is checked for MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(InferenceError::CompileError(
                "mmap failed for executable buffer".into(),
            ));
        }

        let ptr = ptr as *mut u8;

        // Copy code
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
        }

        // Make executable (and read-only)
        let ret = unsafe { libc::mprotect(ptr as *mut _, len, libc::PROT_READ | libc::PROT_EXEC) };
        if ret != 0 {
            unsafe { libc::munmap(ptr as *mut _, len); }
            return Err(InferenceError::CompileError(
                "mprotect failed for executable buffer".into(),
            ));
        }

        Ok(ExecutableBuffer { ptr, len })
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut _, self.len);
            }
        }
    }
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executable_buffer_empty() {
        let buf = ExecutableBuffer::new(&[]).unwrap();
        assert!(buf.ptr.is_null());
        assert_eq!(buf.len, 0);
    }

    #[test]
    fn test_executable_buffer_alloc() {
        // x86_64: `ret` instruction = 0xC3
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8]; // ret
            let buf = ExecutableBuffer::new(&code).unwrap();
            assert!(!buf.ptr.is_null());
            assert!(buf.len >= 1);
            // Verify we can call it (it just returns immediately)
            unsafe {
                let f: extern "C" fn() = std::mem::transmute(buf.ptr);
                f();
            }
        }

        // aarch64: `ret` instruction = 0xD65F03C0
        #[cfg(target_arch = "aarch64")]
        {
            let code = 0xD65F03C0u32.to_le_bytes();
            let buf = ExecutableBuffer::new(&code).unwrap();
            assert!(!buf.ptr.is_null());
            unsafe {
                let f: extern "C" fn() = std::mem::transmute(buf.ptr);
                f();
            }
        }
    }

    #[test]
    fn test_compiled_layer_from_code() {
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8]; // ret
            let layer = CompiledLayer::from_code(&code, 4096, 0x1234).unwrap();
            assert_eq!(layer.code_size(), page_size()); // rounded up
            assert_eq!(layer.scratchpad_bytes, 4096);
            assert_eq!(layer.config_hash, 0x1234);
        }
    }
}
