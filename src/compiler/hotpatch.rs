//! Hot JMP Patching — 运行时原子修改 JIT 代码跳转目标
//!
//! 支持 Mega-Kernel 架构的全域热修补机制（§9.2）。
//! 允许在运行时原子覆写 JIT 生成代码中的 JMP 指令，实现零停机的拓扑重构。

use std::sync::atomic::{AtomicU64, Ordering};
use crate::types::CompilerError;

/// 可热修补的跳转点
#[derive(Debug)]
pub struct PatchableJump {
    /// JMP 指令在代码中的偏移（字节）
    pub offset: usize,
    /// 当前跳转目标的绝对地址
    pub target: AtomicU64,
    /// 可选的跳转目标列表（用于多路径切换）
    pub alternatives: Vec<u64>,
}

/// 热修补注册表
pub struct HotPatchRegistry {
    patches: Vec<PatchableJump>,
    code_base: *mut u8,
    code_len: usize,
}

unsafe impl Send for HotPatchRegistry {}
unsafe impl Sync for HotPatchRegistry {}

impl HotPatchRegistry {
    /// 创建新的热修补注册表
    pub fn new(code_base: *mut u8, code_len: usize) -> Self {
        Self {
            patches: Vec::new(),
            code_base,
            code_len,
        }
    }

    /// 注册一个可热修补的跳转点
    pub fn register_patch(&mut self, offset: usize, target: u64, alternatives: Vec<u64>) {
        self.patches.push(PatchableJump {
            offset,
            target: AtomicU64::new(target),
            alternatives,
        });
    }

    /// 原子修改跳转目标（x86_64 JMP rel32）
    ///
    /// # Safety
    /// - 必须确保 code_base 指向有效的可执行内存
    /// - 必须确保 patch_id 有效
    /// - 必须确保 new_target 是有效的代码地址
    pub unsafe fn patch(&self, patch_id: usize, new_target: u64) -> Result<(), CompilerError> {
        if patch_id >= self.patches.len() {
            return Err(format!("Invalid patch ID: {}", patch_id).into());
        }

        let patch = &self.patches[patch_id];
        let jmp_addr = self.code_base.add(patch.offset);

        // 验证偏移在代码范围内
        if patch.offset + 5 > self.code_len {
            return Err("Patch offset out of bounds".to_string().into());
        }

        // x86_64: JMP rel32 = E9 [4-byte signed offset]
        // 计算相对偏移: target - (jmp_addr + 5)
        let rel_offset = (new_target as i64) - (jmp_addr as i64) - 5;
        if rel_offset < i32::MIN as i64 || rel_offset > i32::MAX as i64 {
            return Err("Jump target out of rel32 range".to_string().into());
        }

        let rel_bytes = (rel_offset as i32).to_le_bytes();

        // 临时修改内存保护（RWX）
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            use libc::{mprotect, PROT_EXEC, PROT_READ, PROT_WRITE};
            let page_size = 4096;
            let page_addr = (jmp_addr as usize & !(page_size - 1)) as *mut libc::c_void;
            if mprotect(page_addr, page_size, PROT_READ | PROT_WRITE | PROT_EXEC) != 0 {
                return Err("mprotect failed".to_string().into());
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::memoryapi::VirtualProtect;
            use winapi::um::winnt::PAGE_EXECUTE_READWRITE;
            let page_size = 4096;
            let page_addr = (jmp_addr as usize & !(page_size - 1)) as *mut winapi::ctypes::c_void;
            let mut old_protect = 0u32;
            if VirtualProtect(page_addr, page_size, PAGE_EXECUTE_READWRITE, &mut old_protect) == 0 {
                return Err("VirtualProtect failed".to_string().into());
            }
        }

        // 原子写入 4-byte offset（跳过 E9 opcode）
        std::ptr::write_volatile(jmp_addr.add(1) as *mut [u8; 4], rel_bytes);

        // 刷新指令缓存（确保 CPU 看到新指令）
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            // ARM 需要显式刷新 icache
            std::arch::asm!(
                "dc cvau, {addr}",
                "dsb ish",
                "ic ivau, {addr}",
                "dsb ish",
                "isb",
                addr = in(reg) jmp_addr,
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 有强内存序，但仍需要内存屏障
            std::sync::atomic::fence(Ordering::SeqCst);
        }

        // 更新记录的目标地址
        patch.target.store(new_target, Ordering::Release);

        Ok(())
    }

    /// 获取当前跳转目标
    pub fn get_target(&self, patch_id: usize) -> Option<u64> {
        self.patches.get(patch_id).map(|p| p.target.load(Ordering::Acquire))
    }

    /// 获取可选目标列表
    pub fn get_alternatives(&self, patch_id: usize) -> Option<&[u64]> {
        self.patches.get(patch_id).map(|p| p.alternatives.as_slice())
    }

    /// 获取注册的跳转点数量
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let code = vec![0u8; 1024];
        let registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_patch() {
        let code = vec![0u8; 1024];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        registry.register_patch(0, 0x1000, vec![0x2000, 0x3000]);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get_target(0), Some(0x1000));

        let alts = registry.get_alternatives(0).unwrap();
        assert_eq!(alts, &[0x2000, 0x3000]);
    }

    #[test]
    fn test_invalid_patch_id() {
        let code = vec![0u8; 1024];
        let registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        assert_eq!(registry.get_target(0), None);
        assert_eq!(registry.get_alternatives(0), None);
    }

    // ── Test 4: register multiple patches ──

    #[test]
    fn register_multiple_patches() {
        let code = vec![0u8; 1024];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        registry.register_patch(10, 0xAAAA, vec![]);
        registry.register_patch(20, 0xBBBB, vec![0xCCCC]);
        registry.register_patch(30, 0xDDDD, vec![0xEEEE, 0xFFFF]);

        assert_eq!(registry.len(), 3);
        assert!(!registry.is_empty());
        assert_eq!(registry.get_target(0), Some(0xAAAA));
        assert_eq!(registry.get_target(1), Some(0xBBBB));
        assert_eq!(registry.get_target(2), Some(0xDDDD));
        assert_eq!(registry.get_alternatives(2).unwrap().len(), 2);
    }

    // ── Test 5: get_target for out-of-bounds returns None ──

    #[test]
    fn get_target_out_of_bounds() {
        let code = vec![0u8; 1024];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        registry.register_patch(0, 0x1000, vec![]);

        assert_eq!(registry.get_target(1), None);
        assert_eq!(registry.get_target(100), None);
    }

    // ── Test 6: get_alternatives for out-of-bounds returns None ──

    #[test]
    fn get_alternatives_out_of_bounds() {
        let code = vec![0u8; 1024];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        registry.register_patch(0, 0x1000, vec![]);

        assert_eq!(registry.get_alternatives(5), None);
    }

    // ── Test 7: register with empty alternatives ──

    #[test]
    fn register_with_empty_alternatives() {
        let code = vec![0u8; 256];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        registry.register_patch(0, 0xDEAD, vec![]);
        let alts = registry.get_alternatives(0).unwrap();
        assert!(alts.is_empty());
    }

    // ── Test 8: patch invalid patch_id returns error ──

    #[test]
    fn patch_invalid_id_returns_error() {
        let code = vec![0u8; 1024];
        let registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        let result = unsafe { registry.patch(99, 0x1000) };
        assert!(result.is_err());
    }

    // ── Test 9: patch offset out of bounds returns error ──

    #[test]
    fn patch_offset_out_of_bounds_returns_error() {
        let code = vec![0u8; 64];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        // offset=60, need 5 bytes for JMP, but only 4 bytes remain (60..64)
        registry.register_patch(60, 0x1000, vec![]);

        let result = unsafe { registry.patch(0, 0x2000) };
        assert!(result.is_err());
    }

    // ── Test 10: PatchableJump Debug format ──

    #[test]
    fn patchable_jump_debug_format() {
        let pj = PatchableJump {
            offset: 42,
            target: AtomicU64::new(0xBEEF),
            alternatives: vec![0x1234],
        };
        let debug = format!("{:?}", pj);
        assert!(debug.contains("42"), "should contain offset");
    }

    // ── Test 11: HotPatchRegistry len and is_empty ──

    #[test]
    fn registry_len_and_is_empty() {
        let code = vec![0u8; 128];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register_patch(0, 0x1000, vec![]);
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    // ── Test 12: patch offset exactly at boundary (5 bytes available) ──

    #[test]
    fn patch_at_exact_boundary_succeeds() {
        // Arrange — 128 bytes of code, register patch at offset 123 (5 bytes: 123..128)
        let mut code = vec![0xE9u8, 0x00, 0x00, 0x00, 0x00, 0]; // 6 bytes
        let code_ptr = code.as_mut_ptr();
        let code_len = code.len();
        let mut registry = HotPatchRegistry::new(code_ptr, code_len);
        // offset 0, target within rel32 range of code area
        let target = code_ptr as u64 + 5; // jump to just after the JMP instruction
        registry.register_patch(0, 0xDEAD, vec![]);

        // Act
        let result = unsafe { registry.patch(0, target) };

        // Assert — offset 0 + 5 bytes = exactly fits in 6 byte buffer
        assert!(result.is_ok());
        assert_eq!(registry.get_target(0), Some(target));
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn patchable_jump_offset_stored() {
        // Arrange
        let pj = PatchableJump {
            offset: 99,
            target: AtomicU64::new(0xABCD),
            alternatives: vec![0x1111, 0x2222],
        };
        // Assert
        assert_eq!(pj.offset, 99);
        assert_eq!(pj.target.load(Ordering::Relaxed), 0xABCD);
        assert_eq!(pj.alternatives.len(), 2);
    }

    #[test]
    fn patchable_jump_atomic_target_update() {
        // Arrange
        let pj = PatchableJump {
            offset: 0,
            target: AtomicU64::new(100),
            alternatives: vec![],
        };
        // Act
        pj.target.store(200, Ordering::Release);
        // Assert
        assert_eq!(pj.target.load(Ordering::Acquire), 200);
    }

    #[test]
    fn patch_offset_at_last_byte_fails() {
        // Arrange — 10 bytes of code, patch at offset 9: needs 5 bytes but only 1 available
        let code = vec![0u8; 10];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        registry.register_patch(9, 0x1000, vec![]);

        // Act
        let result = unsafe { registry.patch(0, 0x2000) };

        // Assert
        assert!(result.is_err(), "should fail: only 1 byte available, need 5");
    }

    #[test]
    fn patch_offset_at_boundary_minus_one_fails() {
        // Arrange — 10 bytes, patch at offset 6: needs 5 bytes but only 4 available (6+5=11>10)
        let code = vec![0u8; 10];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        registry.register_patch(6, 0x1000, vec![]);

        let result = unsafe { registry.patch(0, 0x2000) };
        assert!(result.is_err());
    }

    #[test]
    fn registry_register_preserves_alternatives_order() {
        let code = vec![0u8; 64];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        registry.register_patch(0, 0x100, vec![0xAAA, 0xBBB, 0xCCC]);

        let alts = registry.get_alternatives(0).unwrap();
        assert_eq!(alts[0], 0xAAA);
        assert_eq!(alts[1], 0xBBB);
        assert_eq!(alts[2], 0xCCC);
    }

    #[test]
    fn registry_get_target_returns_latest_after_multiple_registers() {
        let code = vec![0u8; 64];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());

        registry.register_patch(0, 0x111, vec![]);
        registry.register_patch(10, 0x222, vec![]);
        registry.register_patch(20, 0x333, vec![]);

        // Each patch_id maps to the registered target
        assert_eq!(registry.get_target(0), Some(0x111));
        assert_eq!(registry.get_target(1), Some(0x222));
        assert_eq!(registry.get_target(2), Some(0x333));
    }

    #[test]
    fn hot_patch_registry_is_send_sync() {
        // Compile-time check: HotPatchRegistry implements Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HotPatchRegistry>();
    }

    #[test]
    fn patch_zero_code_len_fails() {
        // Arrange — zero-length code buffer, any patch should fail
        let code: Vec<u8> = vec![];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, 0);
        registry.register_patch(0, 0x1000, vec![]);

        let result = unsafe { registry.patch(0, 0x2000) };
        assert!(result.is_err(), "should fail: offset 0 + 5 > code_len 0");
    }

    #[test]
    fn patchable_jump_debug_contains_all_fields() {
        let pj = PatchableJump {
            offset: 7,
            target: AtomicU64::new(0xFEED),
            alternatives: vec![0xDEAD],
        };
        let debug = format!("{:?}", pj);
        assert!(debug.contains("7"), "should contain offset");
        // alternatives should appear in debug output
        assert!(debug.contains("DEAD") || debug.contains("dead") || debug.contains("57005"),
            "debug should reference alternatives");
    }

    #[test]
    fn registry_len_increments_per_register() {
        let code = vec![0u8; 128];
        let mut registry = HotPatchRegistry::new(code.as_ptr() as *mut u8, code.len());
        assert_eq!(registry.len(), 0);

        registry.register_patch(0, 0x1, vec![]);
        assert_eq!(registry.len(), 1);

        registry.register_patch(10, 0x2, vec![]);
        assert_eq!(registry.len(), 2);
    }
}
