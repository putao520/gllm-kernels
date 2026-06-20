//! CompiledLayer — mmap'd executable code + metadata.
//!
//! Wraps a block of JIT-compiled machine code that implements a complete
//! transformer layer. The code is stored in an executable memory region
//! (via mmap) and called through a function pointer.

use crate::types::InferenceError;
use crate::compiler::graph::WeightLayout;
use crate::compiler::hotpatch::HotPatchRegistry;
use crate::types::CompilerError;

// ═══════════════════════════════════════════════════════════════
//  ABI 参数位置由 VmState 动态计算 (ARCH-VM-STATE-TRACKING)
//
//  旧的 abi_slots 硬编码常量模块已删除。
//  所有参数位置通过 vm_state::VmState::arg_ptr_expr("name") 查询。
//  VmState 的初始化由平台 ABI 规则驱动 (init_mega_kernel_x86)。
//  参数名序列内联在 VmState::init_mega_kernel_x86() 中。
// ═══════════════════════════════════════════════════════════════

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
    /// Weight blob layout for multi-weight graphs (BERT etc.)
    pub weight_layout: Option<WeightLayout>,
    /// RoPE cos/sin 表需求 (ARCH-ROPE-CACHE)。
    /// Some 表示 kernel 依赖预填 cos/sin 表,caller 必须按 layout 写入 scratchpad。
    pub rope_cache: Option<crate::compiler::codegen::RopeCacheRequirement>,
    /// Logits region offset within scratchpad. All graphs (generate and non-generate)
    /// write output to scratchpad[logits_scratch_offset]. Callers must read results from there.
    pub logits_scratch_offset: usize,
    /// Number of f32 output elements (for execute_as_mega_kernel post-copy).
    /// 0 means no post-copy (generate graphs write tokens via StoreToken).
    pub output_float_elems: usize,
    /// Hot patch registry (optional, for runtime code modification)
    hotpatch_registry: Option<HotPatchRegistry>,
}

impl CompiledLayer {
    /// Create a CompiledLayer from raw machine code bytes.
    pub fn from_code(
        code_bytes: &[u8],
        scratchpad_bytes: usize,
        config_hash: u64,
    ) -> Result<Self, InferenceError> {
        let code = ExecutableBuffer::new(code_bytes)?;
        let hotpatch_registry = if !code.ptr.is_null() {
            Some(HotPatchRegistry::new(code.ptr, code.len))
        } else {
            None
        };
        Ok(CompiledLayer {
            code,
            entry_offset: 0,
            scratchpad_bytes,
            config_hash,
            weight_layout: None,
            rope_cache: None,
            logits_scratch_offset: 0,
            output_float_elems: 0,
            hotpatch_registry,
        })
    }

    /// Get the entry point as a MegaKernelFn (mega-kernel single-entry-point).
    ///
    /// # Safety
    /// The caller must ensure the compiled code was emitted by `emit_mega_kernel_x86()`
    /// and follows the MegaKernelFn ABI (22 params → usize return).
    #[inline]
    pub unsafe fn entry_point_as_mega_kernel(&self) -> super::MegaKernelFn {
        let ptr = self.code.ptr.add(self.entry_offset);
        std::mem::transmute(ptr)
    }

    /// Size of the compiled code in bytes.
    pub fn code_size(&self) -> usize {
        self.code.len
    }

    /// Raw machine code bytes (for disk caching).
    pub fn code_bytes(&self) -> &[u8] {
        if self.code.ptr.is_null() || self.code.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.code.ptr, self.code.len) }
    }

    /// Execute via MegaKernelFn ABI (22-param). All graphs now use the unified
    /// `compile_mega_kernel_vm` entry point (SPEC/39).
    ///
    /// # Safety
    /// The caller must ensure all pointers are valid and the buffer sizes
    /// match the compiled graph's expected layout.
    #[inline]
    pub unsafe fn execute_as_mega_kernel(
        &self,
        input: *const u8,
        weights: *const u8,
        batch_size: usize,
        seq_len: usize,
        output: *mut u8,
        scratchpad: *mut u8,
    ) {
        let f = self.entry_point_as_mega_kernel();
        let output_tokens = output as *mut u32;
        f(
            input as *const u32,   // arg 0: input_ids_ptr
            weights,               // arg 1: weight_blob_ptr
            std::ptr::null_mut(),  // arg 2: kv_cache_ptr
            std::ptr::null(),      // arg 3: positions_ptr
            std::ptr::null(),      // arg 4: aux_ptr
            batch_size,            // arg 5: batch_size
            seq_len,               // arg 6: prompt_len
            scratchpad,            // arg 7: scratchpad_ptr
            output_tokens,         // arg 8: output_tokens_ptr
            0,                     // arg 9: temperature_u32
            0,                     // arg 10: top_k
            0,                     // arg 11: top_p_u32
            0,                     // arg 12: max_new_tokens
            0,                     // arg 13: eos_token_id
            std::ptr::null(),      // arg 14: hook_ctx_ptr
            std::ptr::null_mut(),  // arg 15: telemetry_ptr
            0,                     // arg 16: session_position
            std::ptr::null(),      // arg 17: fused_hidden_ptr
            0,                     // arg 18: num_mm_tokens
            std::ptr::null(),      // arg 19: callback_table_ptr
            std::ptr::null(),      // arg 20: page_table_ptr
            std::ptr::null(),      // arg 21: batch_ctx_ptr
        );
        // Post-execution: copy output from scratchpad logits region to caller's output buffer.
        // All graphs write their output tensor to scratchpad[logits_scratch_offset].
        // For non-generate graphs (output_float_elems > 0), copy back to the ABI output arg.
        if self.output_float_elems > 0 {
            let src = scratchpad.add(self.logits_scratch_offset) as *const f32;
            let dst = output as *mut f32;
            std::ptr::copy_nonoverlapping(src, dst, self.output_float_elems);
        }
    }

    /// Register a hot-patchable jump point.
    ///
    /// This allows runtime modification of JMP instructions in the compiled code.
    /// Used by the JIT Director Daemon for topology reconfiguration (§9.2).
    pub fn register_hotpatch(&mut self, offset: usize, target: u64, alternatives: Vec<u64>) {
        if let Some(ref mut registry) = self.hotpatch_registry {
            registry.register_patch(offset, target, alternatives);
        }
    }

    /// Apply a hot patch to modify a jump target.
    ///
    /// # Safety
    /// - `patch_id` must be a valid patch registered via `register_hotpatch()`
    /// - `new_target` must be a valid code address within the compiled layer
    pub unsafe fn apply_hotpatch(&self, patch_id: usize, new_target: u64) -> Result<(), CompilerError> {
        if let Some(ref registry) = self.hotpatch_registry {
            registry.patch(patch_id, new_target)
        } else {
            Err("No hotpatch registry available".to_string().into())
        }
    }

    /// Get the current target of a hot patch.
    pub fn get_hotpatch_target(&self, patch_id: usize) -> Option<u64> {
        self.hotpatch_registry.as_ref()?.get_target(patch_id)
    }

    /// Get the number of registered hot patches.
    pub fn hotpatch_count(&self) -> usize {
        self.hotpatch_registry.as_ref().map_or(0, |r| r.len())
    }

    /// NOP out a code region (x86_64: multi-byte NOP sled; aarch64: NOP).
    ///
    /// Used by Hot JMP Patching to physically remove evicted expert code.
    /// Temporarily upgrades memory to RWX, writes NOPs, then restores RX.
    pub fn nop_code_region(&self, offset: usize, len: usize) -> Result<(), InferenceError> {
        self.code.with_write_access(|ptr, total| {
            if offset + len > total {
                return Err(InferenceError::CompileError(
                    format!("NOP region [{}, {}) out of bounds (code size {})", offset, offset + len, total).into(),
                ));
            }
            unsafe { fill_nop_sled(ptr.add(offset), len) }
            Ok(())
        })?
    }

    /// Write arbitrary code bytes at the given offset.
    ///
    /// Used for Restore operations (writing back saved original code).
    pub fn write_code_region(&self, offset: usize, data: &[u8]) -> Result<(), InferenceError> {
        self.code.with_write_access(|ptr, total| {
            if offset + data.len() > total {
                return Err(InferenceError::CompileError(
                    format!("write region [{}, {}) out of bounds (code size {})", offset, offset + data.len(), total).into(),
                ));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
            }
            Ok(())
        })?
    }

    /// Save (copy out) code bytes for later restore.
    pub fn save_code_region(&self, offset: usize, len: usize) -> Result<Vec<u8>, InferenceError> {
        if self.code.ptr.is_null() || offset + len > self.code.len {
            return Err(InferenceError::CompileError(
                format!("save region [{}, {}) out of bounds (code size {})", offset, offset + len, self.code.len).into(),
            ));
        }
        let mut buf = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.code.ptr.add(offset), buf.as_mut_ptr(), len);
        }
        Ok(buf)
    }

    /// Raw code base pointer (for offset calculations by caller).
    pub fn code_base(&self) -> *const u8 {
        self.code.ptr
    }
}

/// Fill a region with NOP sled (x86_64 multi-byte NOP or aarch64 NOP).
///
/// x86_64 uses Intel-recommended multi-byte NOP sequences for decoder efficiency.
fn fill_nop_sled(ptr: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        const MBNOP: &[&[u8]] = &[
            &[0x90],                                           // 1B
            &[0x66, 0x90],                                     // 2B
            &[0x0F, 0x1F, 0x00],                               // 3B
            &[0x0F, 0x1F, 0x40, 0x00],                         // 4B
            &[0x0F, 0x1F, 0x44, 0x00, 0x00],                   // 5B
            &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],             // 6B
            &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],       // 7B
            &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00], // 8B
            &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00], // 9B
        ];
        let mut off = 0;
        while off < len {
            let remaining = len - off;
            let nop_len = remaining.min(9);
            let nop = MBNOP[nop_len - 1];
            unsafe {
                std::ptr::copy_nonoverlapping(nop.as_ptr(), ptr.add(off), nop_len);
            }
            off += nop_len;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let nop_bytes = 0xD503201Fu32.to_le_bytes();
        let mut off = 0;
        while off + 4 <= len {
            unsafe {
                std::ptr::copy_nonoverlapping(nop_bytes.as_ptr(), ptr.add(off), 4);
            }
            off += 4;
        }
        // Pad remaining bytes with 0x00 (shouldn't happen if caller aligns to 4)
        while off < len {
            unsafe { *ptr.add(off) = 0x00; }
            off += 1;
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback: zero-fill (no-op on unknown arch, caller should not use this)
        for i in 0..len {
            unsafe { *ptr.add(i) = 0x00; }
        }
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
    /// Temporarily make the buffer RWX, execute `f`, then restore to RX.
    ///
    /// Handles mprotect → write → mprotect → fence + icache flush.
    fn with_write_access<F, R>(&self, f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(*mut u8, usize) -> R,
    {
        if self.ptr.is_null() || self.len == 0 {
            return Err(InferenceError::CompileError("buffer is empty".into()));
        }

        // RWX
        let ret = unsafe {
            libc::mprotect(self.ptr as *mut _, self.len, libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC)
        };
        if ret != 0 {
            return Err(InferenceError::CompileError("mprotect RWX failed".into()));
        }

        let result = f(self.ptr, self.len);

        // Restore RX
        let ret = unsafe {
            libc::mprotect(self.ptr as *mut _, self.len, libc::PROT_READ | libc::PROT_EXEC)
        };
        if ret != 0 {
            return Err(InferenceError::CompileError("mprotect RX restore failed".into()));
        }

        // x86_64: serializing fence; aarch64: icache flush
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_mfence();
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            libc::cacheflush(self.ptr as *mut _, self.ptr.add(self.len) as *mut _, 0);
        }

        Ok(result)
    }

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

        // Flush instruction cache — required when the same virtual address
        // is reused across mmap/munmap cycles. Without this, stale icache
        // lines from a previous mapping may contain garbage, causing
        // non-deterministic execution failures.
        // On x86_64, this is a no-op (coherent icache), but we include it
        // for correctness on ARM/other platforms and as a safety measure.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            libc::cacheflush(ptr as *mut _, ptr.add(len) as *mut _, 0);
        }
        // x86_64: ensure stores are visible before execution.
        // Serializing instruction prevents speculative execution of
        // stale code from a previous mapping at the same address.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_mfence();
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

// ── GPU Compiled Layer ──────────────────────────────────────────────

/// GPU 编译层 — 持有 GPU kernel module + launch 参数。
#[cfg(feature = "jit-cuda")]
pub struct GpuCompiledKernel {
    /// 编译后的 GPU module handle
    module: crate::gpu::cuda::driver::CUmodule,
    /// Kernel function handle
    function: crate::gpu::cuda::driver::CUfunction,
    /// Kernel 名称
    kernel_name: String,
    /// Launch 参数
    pub grid_dim: [u32; 3],
    pub block_dim: [u32; 3],
    pub shared_mem_bytes: u32,
    /// Driver reference (for cleanup)
    driver: std::sync::Arc<crate::gpu::cuda::driver::CudaDriver>,
}

#[cfg(feature = "jit-cuda")]
impl GpuCompiledKernel {
    /// Kernel 名称。
    pub fn name(&self) -> &str {
        &self.kernel_name
    }

    /// Raw function handle for advanced usage.
    pub fn function(&self) -> crate::gpu::cuda::driver::CUfunction {
        self.function
    }
}

#[cfg(feature = "jit-cuda")]
impl Drop for GpuCompiledKernel {
    fn drop(&mut self) {
        if self.module != 0 {
            unsafe { (self.driver.cuModuleUnload)(self.module); }
        }
    }
}

/// GPU 编译层 — 包含一个或多个 GPU kernel。
#[cfg(feature = "jit-cuda")]
pub struct GpuCompiledLayer {
    /// 按 FusionGroup 顺序排列的 kernel 列表
    kernels: Vec<GpuCompiledKernel>,
    /// 所需的 scratchpad 大小
    pub scratchpad_bytes: usize,
    /// 配置哈希
    pub config_hash: u64,
}

#[cfg(feature = "jit-cuda")]
impl GpuCompiledLayer {
    /// 从 PTX 文本编译 GPU kernel。
    pub fn from_ptx(
        device: &crate::gpu::cuda::device::CudaDevice,
        ptx_source: &[u8],
        kernel_names: &[&str],
        launch_configs: &[crate::gpu::LaunchConfig],
        scratchpad_bytes: usize,
        config_hash: u64,
    ) -> Result<Self, crate::gpu::GpuError> {
        use crate::gpu::cuda::driver::*;
        use crate::gpu::GpuError;

        if kernel_names.len() != launch_configs.len() {
            return Err(GpuError::KernelLaunch(
                "kernel_names and launch_configs length mismatch".into(),
            ));
        }

        let driver = device.driver();

        // Load PTX module
        let mut module: CUmodule = 0;
        let res = unsafe { (driver.cuModuleLoadData)(&mut module, ptx_source.as_ptr() as *const _) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::ShaderCompilation(format!(
                "cuModuleLoadData failed with error {res}"
            )));
        }

        let mut kernels = Vec::with_capacity(kernel_names.len());
        for (i, name) in kernel_names.iter().enumerate() {
            let c_name = std::ffi::CString::new(*name).map_err(|e| {
                GpuError::KernelLaunch(format!("invalid kernel name: {e}"))
            })?;
            let mut function: CUfunction = 0;
            let res = unsafe {
                (driver.cuModuleGetFunction)(&mut function, module, c_name.as_ptr())
            };
            if res != CUDA_SUCCESS {
                // Cleanup module on failure
                unsafe { (driver.cuModuleUnload)(module); }
                return Err(GpuError::KernelLaunch(format!(
                    "cuModuleGetFunction({name}) failed with error {res}"
                )));
            }

            let lc = &launch_configs[i];
            kernels.push(GpuCompiledKernel {
                module: if i == 0 { module } else { 0 }, // Only first kernel owns the module
                function,
                kernel_name: name.to_string(),
                grid_dim: lc.grid_dim,
                block_dim: lc.block_dim,
                shared_mem_bytes: lc.shared_mem_bytes,
                driver: std::sync::Arc::clone(driver),
            });
        }

        Ok(GpuCompiledLayer {
            kernels,
            scratchpad_bytes,
            config_hash,
        })
    }

    /// 返回 kernel 数量。
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// 按索引获取 kernel。
    pub fn kernel(&self, index: usize) -> Option<&GpuCompiledKernel> {
        self.kernels.get(index)
    }

    /// 执行所有 kernel（按顺序）。
    ///
    /// # Safety
    /// 调用者必须确保所有 device pointer 有效且 buffer 大小匹配。
    pub unsafe fn execute(
        &self,
        input: u64,
        weights: u64,
        output: u64,
        n_elements: u32,
        stream: &crate::gpu::cuda::device::CudaStream,
    ) -> Result<(), crate::gpu::GpuError> {
        use crate::gpu::cuda::driver::*;
        use crate::gpu::GpuError;

        for kernel in &self.kernels {
            let mut input_val = input;
            let mut weights_val = weights;
            let mut output_val = output;
            let mut n_val = n_elements;

            let mut params: [*mut std::ffi::c_void; 4] = [
                &mut input_val as *mut u64 as *mut _,
                &mut weights_val as *mut u64 as *mut _,
                &mut output_val as *mut u64 as *mut _,
                &mut n_val as *mut u32 as *mut _,
            ];

            let res = (kernel.driver.cuLaunchKernel)(
                kernel.function,
                kernel.grid_dim[0], kernel.grid_dim[1], kernel.grid_dim[2],
                kernel.block_dim[0], kernel.block_dim[1], kernel.block_dim[2],
                kernel.shared_mem_bytes,
                stream.handle(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if res != CUDA_SUCCESS {
                return Err(GpuError::KernelLaunch(format!(
                    "cuLaunchKernel({}) failed with error {res}",
                    kernel.kernel_name
                )));
            }
        }

        Ok(())
    }
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

    #[cfg(feature = "jit-cuda")]
    #[test]
    fn test_gpu_compiled_layer_struct() {
        // 验证结构体可以构造（不需要真实 GPU）
        let _lc = crate::gpu::LaunchConfig {
            grid_dim: [1, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
        };
    }

    #[test]
    fn test_hotpatch_integration() {
        #[cfg(target_arch = "x86_64")]
        {
            // 创建一个简单的 ret 指令
            let code = [0xC3u8];
            let mut layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // 验证 hotpatch registry 已创建
            assert_eq!(layer.hotpatch_count(), 0);

            // 注册一个热修补点（虽然这个简单代码不会真正使用）
            layer.register_hotpatch(0, 0x1000, vec![0x2000, 0x3000]);
            assert_eq!(layer.hotpatch_count(), 1);
            assert_eq!(layer.get_hotpatch_target(0), Some(0x1000));
        }
    }

    #[test]
    fn test_nop_code_region() {
        #[cfg(target_arch = "x86_64")]
        {
            // Fill with 0x90 (single-byte NOP), ret at end
            let mut code = [0x90u8; 16];
            code[15] = 0xC3; // ret

            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // NOP out bytes 2..10 with multi-byte NOP sled
            layer.nop_code_region(2, 8).unwrap();

            // Multi-byte NOP 8: 0x0F 0x1F 0x84 0x00 0x00 0x00 0x00 0x00
            // Verify it differs from single-byte NOP (0x90) pattern
            let saved = layer.save_code_region(2, 8).unwrap();
            assert_ne!(saved, [0x90u8; 8], "region should be multi-byte NOP, not 0x90 fill");
            assert_eq!(saved[0], 0x0F); // multi-byte NOP starts with 0x0F

            // ret at offset 15 still intact
            let tail = layer.save_code_region(15, 1).unwrap();
            assert_eq!(tail[0], 0xC3);

            // Callable: all NOPs + ret = safe
            unsafe {
                let f: extern "C" fn() = std::mem::transmute(layer.code_base());
                f();
            }
        }
    }

    #[test]
    fn test_write_code_region() {
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xCCu8; 16]; // all INT3
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Write a ret at offset 0
            layer.write_code_region(0, &[0xC3]).unwrap();

            let saved = layer.save_code_region(0, 1).unwrap();
            assert_eq!(saved[0], 0xC3);
        }
    }

    #[test]
    fn test_save_code_region() {
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0x90u8; 64]; // 64 NOPs
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            let saved = layer.save_code_region(10, 5).unwrap();
            assert_eq!(saved.len(), 5);
            assert_eq!(saved, &[0x90, 0x90, 0x90, 0x90, 0x90]);
        }
    }

    #[test]
    fn test_nop_out_of_bounds() {
        let code = [0xC3u8; 4];
        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
        // offset + len > code_size
        assert!(layer.nop_code_region(0, page_size() + 1).is_err());
    }

    #[test]
    fn test_code_base_ptr() {
        let code = [0xC3u8];
        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
        assert!(!layer.code_base().is_null());
    }

    // @trace TEST-EXE-11 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_compiled_layer_from_empty_code() {
        let layer = CompiledLayer::from_code(&[], 0, 0).unwrap();
        assert!(layer.code_base().is_null());
        assert_eq!(layer.code_size(), 0);
        assert_eq!(layer.scratchpad_bytes, 0);
        assert_eq!(layer.config_hash, 0);
        assert!(layer.weight_layout.is_none());
        assert!(layer.rope_cache.is_none());
        assert_eq!(layer.hotpatch_count(), 0);
    }

    // @trace TEST-EXE-12 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_code_bytes_empty_and_nonempty() {
        // Empty layer returns empty slice
        let empty_layer = CompiledLayer::from_code(&[], 0, 0).unwrap();
        assert!(empty_layer.code_bytes().is_empty());

        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8]; // ret
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            let bytes = layer.code_bytes();
            // First byte matches input; rest is zero-filled page alignment
            assert_eq!(bytes[0], 0xC3);
            assert!(bytes.len() >= 1);
        }
    }

    // @trace TEST-EXE-13 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_save_code_region_empty_layer_error() {
        let layer = CompiledLayer::from_code(&[], 0, 0).unwrap();
        let result = layer.save_code_region(0, 1);
        assert!(result.is_err(), "saving from empty layer should fail");
    }

    // @trace TEST-EXE-14 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_save_code_region_out_of_bounds() {
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8; 8];
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            // Request beyond page-aligned buffer
            assert!(layer.save_code_region(0, page_size() + 1).is_err());
            // offset > len
            assert!(layer.save_code_region(page_size(), 1).is_err());
        }
    }

    // @trace TEST-EXE-15 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_write_code_region_out_of_bounds() {
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8; 4];
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            // Write exceeds code size: allocate a Vec larger than page-aligned buffer
            let big_data = vec![0x90u8; page_size() + 1];
            assert!(layer.write_code_region(0, &big_data).is_err());
        }
    }

    // @trace TEST-EXE-17 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_entry_point_as_mega_kernel_returns_valid_ptr() {
        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8]; // ret
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            unsafe {
                let f = layer.entry_point_as_mega_kernel();
                let fptr = f as *const u8;
                assert!(!fptr.is_null());
                assert!(fptr >= layer.code_base());
                let offset = fptr as usize - layer.code_base() as usize;
                assert!(offset < layer.code_size());
            }
        }
    }

    // @trace TEST-EXE-18 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_apply_hotpatch_no_registry_error() {
        // Empty code => no hotpatch registry => apply should error
        let layer = CompiledLayer::from_code(&[], 0, 0).unwrap();
        let result = unsafe { layer.apply_hotpatch(0, 0xDEAD) };
        assert!(result.is_err(), "apply_hotpatch on empty layer should fail");
    }

    // @trace TEST-EXE-19 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_get_hotpatch_target_none_cases() {
        // Empty layer has no registry
        let empty = CompiledLayer::from_code(&[], 0, 0).unwrap();
        assert_eq!(empty.get_hotpatch_target(0), None);

        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8];
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            // Non-existent patch ID
            assert_eq!(layer.get_hotpatch_target(99), None);
        }
    }

    // @trace TEST-EXE-20 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_register_hotpatch_on_empty_layer_noop() {
        // Empty layer has no registry, so register_hotpatch should be a no-op
        let mut layer = CompiledLayer::from_code(&[], 0, 0).unwrap();
        layer.register_hotpatch(0, 0x100, vec![0x200]);
        assert_eq!(layer.hotpatch_count(), 0, "no registry => register is no-op");
    }

    // @trace TEST-EXE-21 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_nop_code_region_fill_patterns() {
        #[cfg(target_arch = "x86_64")]
        {
            // Create code with known non-NOP content and a trailing ret
            let mut code = [0xCCu8; 64]; // INT3 fill
            code[63] = 0xC3; // ret at end

            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // NOP a 1-byte region
            layer.nop_code_region(0, 1).unwrap();
            let one = layer.save_code_region(0, 1).unwrap();
            assert_eq!(one[0], 0x90, "1-byte NOP should be 0x90");

            // NOP a 2-byte region
            layer.nop_code_region(1, 2).unwrap();
            let two = layer.save_code_region(1, 2).unwrap();
            assert_eq!(two, [0x66, 0x90], "2-byte NOP should be 66 90");

            // NOP a 9-byte region (max single multi-byte NOP)
            layer.nop_code_region(10, 9).unwrap();
            let nine = layer.save_code_region(10, 9).unwrap();
            assert_eq!(nine[0], 0x66, "9-byte NOP starts with 66 prefix");
            assert_eq!(nine[1], 0x0F);

            // NOP a 20-byte region (uses 9+9+2 pattern)
            layer.nop_code_region(30, 20).unwrap();
            let twenty = layer.save_code_region(30, 20).unwrap();
            // First 9 bytes: 9B NOP
            assert_eq!(twenty[0], 0x66);
            // Bytes 9-17: another 9B NOP
            assert_eq!(twenty[9], 0x66);
            // Bytes 18-19: 2B NOP
            assert_eq!(twenty[18], 0x66);
            assert_eq!(twenty[19], 0x90);

            // ret still intact
            let tail = layer.save_code_region(63, 1).unwrap();
            assert_eq!(tail[0], 0xC3);
        }
    }

    // @trace TEST-EXE-22 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_nop_code_region_on_empty_layer_error() {
        let layer = CompiledLayer::from_code(&[], 0, 0).unwrap();
        // Empty layer has null ptr, so nop_code_region should fail
        let result = layer.nop_code_region(0, 1);
        assert!(result.is_err(), "nop_code_region on empty layer should fail");
    }

    // @trace TEST-EXE-23 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_page_size_is_reasonable() {
        let ps = page_size();
        // Page size must be a power of 2
        assert!(ps > 0, "page size must be non-zero");
        assert_eq!(ps & (ps - 1), 0, "page size must be a power of 2");
        // Common values: 4096 (x86_64), 65536 (some ARM)
        assert!(ps >= 4096, "page size should be at least 4096");
        assert!(ps <= 65536, "page size should be at most 65536");
    }

    // @trace TEST-EXE-24 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_execute_calls_compiled_ret() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: single ret instruction
            let code = [0xC3u8];
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: call execute with null pointers — it should just return
            unsafe {
                layer.execute_as_mega_kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
            }
            // Assert: no crash = success
        }
    }

    // @trace TEST-EXE-26 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_write_then_save_roundtrip() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 64 bytes of INT3 with ret at end
            let mut code = [0xCCu8; 64];
            code[63] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            let payload: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

            // Act: write payload at offset 10, then save it back
            layer.write_code_region(10, &payload).unwrap();
            let saved = layer.save_code_region(10, 8).unwrap();

            // Assert: saved bytes exactly match what was written
            assert_eq!(saved.as_slice(), payload.as_slice());

            // Assert: ret at offset 63 is still intact
            let tail = layer.save_code_region(63, 1).unwrap();
            assert_eq!(tail[0], 0xC3, "ret instruction should be untouched");
        }
    }

    // @trace TEST-EXE-27 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_write_code_region_on_empty_layer_error() {
        // Arrange: empty layer has null pointer internally
        let layer = CompiledLayer::from_code(&[], 0, 0).unwrap();

        // Act
        let result = layer.write_code_region(0, &[0xC3]);

        // Assert: with_write_access should reject null/empty buffer
        assert!(result.is_err(), "write_code_region on empty layer should fail");
    }

    // @trace TEST-EXE-28 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_nop_code_region_exact_boundary() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: code of exactly 10 bytes with ret at the end
            let mut code = [0xCCu8; 10];
            code[9] = 0xC3; // ret
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: NOP exactly bytes 0..9, leaving the ret intact
            layer.nop_code_region(0, 9).unwrap();

            // Assert: first 9 bytes are NOP sled, last byte is ret
            let saved = layer.save_code_region(0, 10).unwrap();
            // 9-byte NOP starts with 0x66 0x0F 0x1F 0x84 ...
            assert_eq!(saved[0], 0x66, "9-byte NOP should start with 0x66 prefix");
            assert_eq!(saved[9], 0xC3, "ret at byte 9 should be intact");

            // Assert: callable (NOPs + ret)
            unsafe {
                let f: extern "C" fn() = std::mem::transmute(layer.code_base());
                f();
            }
        }
    }

    // @trace TEST-EXE-29 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_fill_nop_sled_all_x86_64_sizes() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: test fill_nop_sled for all sizes 1..9 on a live buffer
            // Expected multi-byte NOP encodings (from MBNOP table)
            let expected: &[&[u8]] = &[
                &[0x90],                                           // 1B
                &[0x66, 0x90],                                     // 2B
                &[0x0F, 0x1F, 0x00],                               // 3B
                &[0x0F, 0x1F, 0x40, 0x00],                         // 4B
                &[0x0F, 0x1F, 0x44, 0x00, 0x00],                   // 5B
                &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],             // 6B
                &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],       // 7B
                &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00], // 8B
                &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00], // 9B
            ];

            // Use a 128-byte buffer, put ret at end
            let mut code = [0xCCu8; 128];
            code[127] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            let mut offset = 0usize;
            for (size_idx, nop_bytes) in expected.iter().enumerate() {
                let nop_len = nop_bytes.len();
                // Act: NOP exactly `nop_len` bytes starting at `offset`
                layer.nop_code_region(offset, nop_len).unwrap();
                // Assert: bytes match expected NOP encoding
                let saved = layer.save_code_region(offset, nop_len).unwrap();
                assert_eq!(
                    saved.as_slice(),
                    *nop_bytes,
                    "NOP size {} at offset {} mismatch",
                    nop_len,
                    offset
                );
                offset += nop_len;
            }

            // Assert: total NOP bytes = 1+2+3+4+5+6+7+8+9 = 45
            assert_eq!(offset, 45);
            // Assert: ret at offset 127 still intact
            let tail = layer.save_code_region(127, 1).unwrap();
            assert_eq!(tail[0], 0xC3);
        }
    }

    // @trace TEST-EXE-30 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_multiple_hotpatches_with_alternatives() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange
            let code = [0xC3u8];
            let mut layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Register three patches with varying alternatives
            layer.register_hotpatch(10, 0x1000, vec![0x2000]);
            layer.register_hotpatch(20, 0x3000, vec![0x4000, 0x5000]);
            layer.register_hotpatch(30, 0x6000, vec![]);

            // Act & Assert: count
            assert_eq!(layer.hotpatch_count(), 3);

            // Assert: targets
            assert_eq!(layer.get_hotpatch_target(0), Some(0x1000));
            assert_eq!(layer.get_hotpatch_target(1), Some(0x3000));
            assert_eq!(layer.get_hotpatch_target(2), Some(0x6000));

            // Assert: non-existent patch returns None
            assert_eq!(layer.get_hotpatch_target(3), None);
            assert_eq!(layer.get_hotpatch_target(99), None);
        }
    }

    // @trace TEST-EXE-31 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_apply_hotpatch_invalid_id_on_registered_layer() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: create layer with one registered patch
            let code = [0xC3u8];
            let mut layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            layer.register_hotpatch(0, 0x1000, vec![]);

            // Act: try to apply patch with invalid index
            let result = unsafe { layer.apply_hotpatch(99, 0xDEAD) };

            // Assert: should error because patch_id 99 does not exist
            assert!(result.is_err(), "applying non-existent patch_id should fail");
        }
    }

    // @trace TEST-EXE-32 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_nop_code_region_offset_nearly_at_end() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 32-byte code with ret at byte 31
            let mut code = [0xCCu8; 32];
            code[31] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: NOP only byte 30 (1-byte NOP), leaving ret at 31 untouched
            layer.nop_code_region(30, 1).unwrap();

            // Assert: byte 30 is 0x90 (single-byte NOP)
            let saved = layer.save_code_region(30, 2).unwrap();
            assert_eq!(saved[0], 0x90);
            assert_eq!(saved[1], 0xC3, "ret at byte 31 should be untouched");
        }
    }

    // @trace TEST-EXE-33 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_code_bytes_preserves_original_content() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: craft specific bytes with ret at end
            let mut code = [0u8; 16];
            for i in 0..15 {
                code[i] = (i as u8).wrapping_mul(17); // arbitrary pattern
            }
            code[15] = 0xC3; // ret
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act
            let bytes = layer.code_bytes();

            // Assert: first 16 bytes exactly match original code
            assert_eq!(&bytes[..16], &code[..], "code_bytes must preserve original content");
        }
    }

    // @trace TEST-EXE-34 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_from_code_default_fields_are_none_or_zero() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange
            let code = [0xC3u8];

            // Act
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Assert: optional fields default to None
            assert!(layer.weight_layout.is_none(), "weight_layout should default to None");
            assert!(layer.rope_cache.is_none(), "rope_cache should default to None");
            assert_eq!(layer.scratchpad_bytes, 0);
            assert_eq!(layer.config_hash, 0);
        }
    }

    // @trace TEST-EXE-35 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_from_code_large_scratchpad_and_max_hash() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: use maximum u64 hash and a large scratchpad
            let code = [0xC3u8];
            let large_scratch = 1024 * 1024 * 128; // 128 MiB
            let max_hash = u64::MAX;

            // Act
            let layer = CompiledLayer::from_code(&code, large_scratch, max_hash).unwrap();

            // Assert: values are stored verbatim without overflow
            assert_eq!(layer.scratchpad_bytes, large_scratch);
            assert_eq!(layer.config_hash, max_hash);
        }
    }

    // @trace TEST-EXE-36 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_save_code_region_zero_length() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: non-empty code buffer
            let code = [0xC3u8; 8];
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: save zero bytes from offset 0
            let saved = layer.save_code_region(0, 0).unwrap();

            // Assert: empty vec returned, no error
            assert!(saved.is_empty(), "zero-length save should return empty vec");
        }
    }

    // @trace TEST-EXE-37 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_nop_code_region_zero_length_noop() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 16 bytes with known pattern + ret at end
            let mut code = [0xABu8; 16];
            code[15] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: NOP zero bytes — should be a no-op
            let result = layer.nop_code_region(4, 0);

            // Assert: succeeds and original content is untouched
            assert!(result.is_ok(), "zero-length NOP should succeed");
            let saved = layer.save_code_region(0, 16).unwrap();
            assert_eq!(saved[0], 0xAB, "content before offset should be untouched");
            assert_eq!(saved[4], 0xAB, "content at offset should be untouched");
            assert_eq!(saved[15], 0xC3, "ret should be untouched");
        }
    }

    // @trace TEST-EXE-38 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_write_code_region_zero_length_noop() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 16 bytes with known content
            let mut code = [0xCDu8; 16];
            code[15] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: write zero bytes
            let result = layer.write_code_region(5, &[]);

            // Assert: succeeds, content unchanged
            assert!(result.is_ok(), "zero-length write should succeed");
            let saved = layer.save_code_region(5, 1).unwrap();
            assert_eq!(saved[0], 0xCD, "content at write offset should be untouched");
        }
    }

    // @trace TEST-EXE-39 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_interleaved_nop_write_nop_preserves_ret() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 64-byte buffer with ret at end
            let mut code = [0xCCu8; 64];
            code[63] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: sequence of NOP → write → NOP on different regions
            layer.nop_code_region(0, 4).unwrap();
            layer.write_code_region(8, &[0xAA, 0xBB, 0xCC, 0xDD]).unwrap();
            layer.nop_code_region(20, 6).unwrap();

            // Assert: each region has expected content, ret intact
            let nop4 = layer.save_code_region(0, 4).unwrap();
            assert_eq!(nop4.as_slice(), &[0x0F, 0x1F, 0x40, 0x00], "4B NOP");

            let written = layer.save_code_region(8, 4).unwrap();
            assert_eq!(written.as_slice(), &[0xAA, 0xBB, 0xCC, 0xDD], "written payload");

            let nop6 = layer.save_code_region(20, 6).unwrap();
            assert_eq!(nop6.as_slice(), &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00], "6B NOP");

            let tail = layer.save_code_region(63, 1).unwrap();
            assert_eq!(tail[0], 0xC3, "ret must survive interleaved ops");
        }
    }

    // @trace TEST-EXE-40 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_code_bytes_reflects_nop_modification() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 8 bytes of INT3 + ret
            let mut code = [0xCCu8; 8];
            code[7] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

            // Act: NOP first 7 bytes, then read via code_bytes()
            layer.nop_code_region(0, 7).unwrap();
            let bytes = layer.code_bytes();

            // Assert: code_bytes reflects the NOP sled, not original INT3
            assert_ne!(bytes[0], 0xCC, "byte 0 should no longer be INT3");
            assert_eq!(bytes[0], 0x0F, "7-byte NOP starts with 0x0F");
            assert_eq!(bytes[7], 0xC3, "ret byte preserved");
        }
    }

    // @trace TEST-EXE-41 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_executable_buffer_drop_does_not_double_free() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: create and drop two buffers sequentially to verify no double-free
            let code = [0xC3u8];
            let ptr1 = {
                let buf = ExecutableBuffer::new(&code).unwrap();
                buf.ptr
            };
            // buf dropped here — munmap called once

            // Act: allocate a new buffer (may reuse same address)
            let buf2 = ExecutableBuffer::new(&code).unwrap();

            // Assert: new buffer is valid and callable
            assert!(!buf2.ptr.is_null());
            // ptr1 is dangling after munmap — do NOT dereference it
            // Just verify buf2 works independently
            unsafe {
                let f: extern "C" fn() = std::mem::transmute(buf2.ptr);
                f(); // no crash = drop did not corrupt state
            }

            // Verify sequential drops don't panic (RAII correctness)
            drop(buf2);
        }
    }

    // @trace TEST-EXE-42 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_from_code_code_size_rounds_up_to_page() {
        // Arrange: small code (1 byte) and larger code (page_size - 1 bytes)
        let small_code = [0xC3u8];

        // Act
        let small_layer = CompiledLayer::from_code(&small_code, 0, 0).unwrap();
        let ps = page_size();

        // Assert: 1 byte rounds up to at least 1 page
        assert_eq!(small_layer.code_size(), ps, "1-byte code should round up to page_size");

        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: code that's exactly page_size bytes (all NOP + ret at end)
            let mut big_code = vec![0x90u8; ps];
            big_code[ps - 1] = 0xC3; // ret at end

            // Act
            let exact_layer = CompiledLayer::from_code(&big_code, 0, 0).unwrap();

            // Assert: exactly page_size fits in exactly one page
            assert_eq!(exact_layer.code_size(), ps, "page_size code should fit in one page");

            // Arrange: code that's page_size + 1 bytes
            let mut over_code = vec![0x90u8; ps + 1];
            over_code[ps] = 0xC3;
            let over_layer = CompiledLayer::from_code(&over_code, 0, 0).unwrap();

            // Assert: page_size + 1 rounds up to 2 pages
            assert_eq!(over_layer.code_size(), ps * 2, "page_size+1 should round up to 2 pages");
        }
    }

    // @trace TEST-EXE-43 [req:REQ-JIT] [level:unit]
    #[test]
    fn test_write_code_region_at_exact_end_boundary() {
        #[cfg(target_arch = "x86_64")]
        {
            // Arrange: 16-byte code, ret at last byte
            let mut code = [0xCCu8; 16];
            code[15] = 0xC3;
            let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
            let total = layer.code_size();

            // Act: write 1 byte at the last position of the page-aligned buffer
            layer.write_code_region(total - 1, &[0xC3]).unwrap();

            // Assert: succeeds without out-of-bounds error
            let saved = layer.save_code_region(total - 1, 1).unwrap();
            assert_eq!(saved[0], 0xC3, "last byte should be the written value");

            // Act: write exactly at the boundary (offset + len == total) should succeed
            layer.write_code_region(total - 2, &[0x90, 0xC3]).unwrap();

            // Assert: boundary write succeeds
            let saved2 = layer.save_code_region(total - 2, 2).unwrap();
            assert_eq!(saved2, [0x90, 0xC3]);
        }
    }
}
